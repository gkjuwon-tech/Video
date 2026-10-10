"""FND-07 실험 매니페스트: 신원(manifest_id), 검증, 원자적 쓰기, 변조 탐지.

계약: ops/harness/contracts/FND-07.md. 같은 (설정, 코드, 시드, 입력, 파라미터)면 같은 manifest_id.
created_utc는 신원에서 뺀다.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import blake3

from buc.core.config import Config, ConfigError, canonical_json

__all__ = ["SCHEMA", "ManifestError", "Manifest", "make_manifest", "read_git_commit",
           "write_manifest", "read_manifest"]

SCHEMA = "buc.manifest/v1"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TS_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_SEED_MAX = 2**64
_TOP_KEYS = ("schema", "name", "seed", "config", "config_hash", "git_commit", "inputs", "params", "created_utc")
_GIT_TIMEOUT_S = 10


class ManifestError(ValueError):
    """매니페스트가 잘못됐거나 변조됐다."""


def _json_copy(obj, what):
    try:
        return json.loads(canonical_json(obj))
    except (TypeError, ValueError) as e:
        raise ManifestError(f"{what}: JSON 직렬화 가능한 값이어야 합니다 ({e})") from None


def _validate(m: "Manifest") -> None:
    if m.schema != SCHEMA:
        raise ManifestError(f"schema: {SCHEMA!r}여야 합니다 (받은 값 {m.schema!r})")
    if not isinstance(m.name, str) or not _NAME_RE.fullmatch(m.name):
        raise ManifestError(f"name 형식 오류: {m.name!r}")
    if not isinstance(m.git_commit, str) or not (m.git_commit == "unknown" or _SHA_RE.fullmatch(m.git_commit)):
        raise ManifestError(f"git_commit 형식 오류: {m.git_commit!r}")
    if not isinstance(m.inputs, dict):
        raise ManifestError("inputs: dict여야 합니다")
    for k, v in m.inputs.items():
        if not isinstance(k, str) or k == "":
            raise ManifestError(f"inputs 키 오류: {k!r}")
        if not isinstance(v, str) or not _HASH_RE.fullmatch(v):
            raise ManifestError(f"inputs[{k!r}]는 64자 소문자 hex여야 합니다")
    if not isinstance(m.created_utc, str) or not _TS_RE.fullmatch(m.created_utc):
        raise ManifestError(f"created_utc 형식 오류: {m.created_utc!r}")
    try:
        datetime.strptime(m.created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ManifestError(f"created_utc가 실존하지 않는 시각입니다: {m.created_utc!r}") from None
    if not isinstance(m.seed, int) or isinstance(m.seed, bool) or not (0 <= m.seed < _SEED_MAX):
        raise ManifestError(f"seed: [0, 2^64) 정수여야 합니다 (받은 값 {m.seed!r})")
    if not isinstance(m.config, dict):
        raise ManifestError("config: dict여야 합니다")
    try:
        cfg = Config.from_dict(m.config)
    except ConfigError as e:
        raise ManifestError(f"config 검증 실패: {e}") from None
    if m.seed != cfg.seed:
        raise ManifestError("seed가 config.seed와 다릅니다")
    if m.config_hash != cfg.hash():
        raise ManifestError("config_hash가 config와 맞지 않습니다")
    if not isinstance(m.params, dict):
        raise ManifestError("params: dict여야 합니다")
    _json_copy(m.params, "params")


@dataclass(frozen=True, kw_only=True)
class Manifest:
    schema: str
    name: str
    seed: int
    config: dict
    config_hash: str
    git_commit: str
    inputs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    created_utc: str

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "config", _json_copy(self.config, "config") if isinstance(self.config, dict) else self.config)
        set_(self, "params", _json_copy(self.params, "params") if isinstance(self.params, dict) else self.params)
        set_(self, "inputs", copy.deepcopy(self.inputs) if isinstance(self.inputs, dict) else self.inputs)
        _validate(self)

    @property
    def manifest_id(self) -> str:
        d = self.to_dict()
        d.pop("created_utc", None)
        return blake3.blake3(canonical_json(d)).hexdigest()

    def to_dict(self) -> dict:
        return {k: copy.deepcopy(getattr(self, k)) for k in _TOP_KEYS}

    @classmethod
    def from_dict(cls, d) -> "Manifest":
        if not isinstance(d, dict):
            raise ManifestError("매니페스트는 dict여야 합니다")
        unknown = set(d) - set(_TOP_KEYS)
        if unknown:
            raise ManifestError(f"알 수 없는 키: {sorted(unknown)}")
        missing = [k for k in _TOP_KEYS if k not in d]
        if missing:
            raise ManifestError(f"누락된 키: {missing}")
        try:
            return cls(**{k: d[k] for k in _TOP_KEYS})
        except TypeError as e:
            raise ManifestError(str(e)) from None


def read_git_commit(repo_dir=None) -> str:
    """HEAD 커밋 해시. 실패하면 'unknown'. 예외를 던지지 않는다."""
    cwd = repo_dir if repo_dir is not None else Path(__file__).resolve().parents[2]
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True,
                           timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = p.stdout.strip()
    if p.returncode != 0 or not _SHA_RE.fullmatch(out):
        return "unknown"
    return out


def make_manifest(name, config: Config, *, inputs=None, params=None, git_commit=None,
                  created_utc=None) -> Manifest:
    if not isinstance(config, Config):
        raise TypeError("config는 Config 객체여야 합니다")
    if git_commit is None:
        git_commit = read_git_commit()
    if created_utc is None:
        created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return Manifest(
        schema=SCHEMA,
        name=name,
        seed=config.seed,
        config=config.to_dict(),
        config_hash=config.hash(),
        git_commit=git_commit,
        inputs={} if inputs is None else inputs,
        params={} if params is None else params,
        created_utc=created_utc,
    )


def write_manifest(m: Manifest, path) -> Path:
    """임시 파일에 쓴 뒤 os.replace로 원자적으로 옮긴다. 실패하면 임시 파일을 지운다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json({**m.to_dict(), "manifest_id": m.manifest_id}) + b"\n"
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def read_manifest(path) -> Manifest:
    raw = Path(path).read_bytes()  # 없으면 FileNotFoundError
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ManifestError(f"매니페스트가 JSON이 아닙니다: {e}") from None
    if not isinstance(d, dict):
        raise ManifestError("매니페스트의 최상위는 객체여야 합니다")
    stored = d.pop("manifest_id", None)
    if not isinstance(stored, str):
        raise ManifestError("manifest_id가 없습니다")
    m = Manifest.from_dict(d)
    if m.manifest_id != stored:
        raise ManifestError("manifest_id가 내용과 맞지 않습니다(변조 의심)")
    return m
