"""FND-07 manifest.py: 신원(manifest_id)·검증·원자적 쓰기·변조 탐지."""
import dataclasses
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import blake3
import pytest

from buc.core.config import Config, canonical_json
from buc.core.manifest import SCHEMA, Manifest, ManifestError, make_manifest, read_git_commit, read_manifest, write_manifest

REPO = Path(__file__).resolve().parents[2]
SHA = "0123456789abcdef0123456789abcdef01234567"
H1, H2 = "ab" * 32, "cd" * 32
TOP_KEYS = {"schema", "name", "seed", "config", "config_hash", "git_commit", "inputs", "params", "created_utc"}


def mk(**over):
    kw = dict(name="exp-001", config=Config(seed=7), inputs={"img": H1}, params={"lr": 1e-3, "steps": 100},
              git_commit=SHA, created_utc="2026-10-10T12:00:00Z")
    kw.update(over)
    name, cfg = kw.pop("name"), kw.pop("config")
    return make_manifest(name, cfg, **kw)


def oracle_id(d):
    d = {k: v for k, v in d.items() if k not in ("created_utc", "manifest_id")}
    return blake3.blake3(canonical_json(d)).hexdigest()


# ---------------------------------------------------------------- 필드와 id
def test_fields_come_from_config():
    cfg = Config(seed=7, budget_daily_usd=2)
    m = mk(config=cfg)
    assert SCHEMA == "buc.manifest/v1" and m.schema == SCHEMA
    assert m.seed == 7 and m.config == cfg.to_dict() and m.config_hash == cfg.hash()
    assert m.name == "exp-001" and m.git_commit == SHA and m.created_utc == "2026-10-10T12:00:00Z"
    assert m.inputs == {"img": H1} and m.params == {"lr": 1e-3, "steps": 100}
    empty = make_manifest("e", cfg, git_commit="unknown", created_utc="2026-01-01T00:00:00Z")
    assert empty.inputs == {} and empty.params == {} and empty.git_commit == "unknown"


def test_to_dict_keys_and_manifest_id_oracle():
    m = mk()
    d = m.to_dict()
    assert set(d) == TOP_KEYS
    mid = m.manifest_id
    assert re.fullmatch(r"[0-9a-f]{64}", mid)
    assert mid == oracle_id(d)
    assert mid == m.manifest_id  # 안정


def test_id_ignores_created_utc_only():
    base = mk()
    assert mk(created_utc="2030-05-05T05:05:05Z").manifest_id == base.manifest_id
    variants = [
        mk(name="exp-002"), mk(config=Config(seed=8)), mk(config=Config(seed=7, budget_total_usd=99)),
        mk(git_commit="f" * 40), mk(git_commit="unknown"), mk(inputs={"img": H2}), mk(inputs={"img": H1, "b": H2}),
        mk(inputs={"image": H1}), mk(params={"lr": 1e-3, "steps": 101}), mk(params={"lr": 1e-3}),
        mk(params={"lr": 1e-3, "steps": 100.0}),  # 정수/실수 구분
    ]
    ids = {v.manifest_id for v in variants} | {base.manifest_id}
    assert len(ids) == len(variants) + 1


def test_id_independent_of_dict_insertion_order():
    a = mk(inputs={"a": H1, "b": H2}, params={"x": {"p": 1, "q": 2}, "y": [3, 1]})
    b = mk(inputs={"b": H2, "a": H1}, params={"y": [3, 1], "x": {"q": 2, "p": 1}})
    assert a.manifest_id == b.manifest_id and a == b


# ---------------------------------------------------------------- 검증
@pytest.mark.parametrize("name", ["", "-bad", ".hidden", "a b", "x" * 129, "한글", "a/b", "a\n"])
def test_bad_names(name):
    with pytest.raises(ManifestError):
        mk(name=name)


def test_good_names():
    for n in ["a", "A1", "exp_1.2-3", "x" * 128]:
        assert mk(name=n).name == n


@pytest.mark.parametrize("commit", ["XYZ", "A" * 40, "abc", "a" * 39, "a" * 41, "", "Unknown"])
def test_bad_git_commit(commit):
    with pytest.raises(ManifestError):
        mk(git_commit=commit)


@pytest.mark.parametrize("inputs", [
    {"i": "short"}, {"i": "G" * 64}, {"i": "A" * 64}, {"i": 5}, {"": H1}, {"i": "a" * 63}, {"i": "a" * 65},
])
def test_bad_inputs(inputs):
    with pytest.raises(ManifestError):
        mk(inputs=inputs)


@pytest.mark.parametrize("ts", [
    "2026-10-10 12:00:00", "2026-13-01T00:00:00Z", "2026-02-30T00:00:00Z", "2026-10-10T12:00:00+00:00",
    "2026-10-10T12:00:00", "2026-10-10T24:00:00Z", "2026-10-10T12:00:00.5Z", "", "now",
])
def test_bad_created_utc(ts):
    with pytest.raises(ManifestError):
        mk(created_utc=ts)


@pytest.mark.parametrize("params", [{"a": object()}, {"a": float("nan")}, {1: 2}, [1, 2], "x"])
def test_bad_params(params):
    with pytest.raises(ManifestError):
        mk(params=params)


def test_make_manifest_needs_a_config_object():
    with pytest.raises(TypeError):
        make_manifest("x", {"seed": 1}, git_commit="unknown", created_utc="2026-01-01T00:00:00Z")


def test_manifest_error_is_value_error():
    assert issubclass(ManifestError, ValueError)


def test_from_dict_detects_inconsistency():
    m = mk()
    assert Manifest.from_dict(m.to_dict()) == m
    d = m.to_dict()
    d["config_hash"] = "0" * 64
    with pytest.raises(ManifestError):
        Manifest.from_dict(d)
    d = m.to_dict()
    d["config"]["seed"] = 8  # 해시 갱신 없이 설정만 바꿈
    with pytest.raises(ManifestError):
        Manifest.from_dict(d)
    d = m.to_dict()
    d["config"]["seed"] = -1
    with pytest.raises(ValueError):  # ConfigError 또는 ManifestError
        Manifest.from_dict(d)
    d = m.to_dict()
    d["zzz"] = 1
    with pytest.raises(ManifestError):
        Manifest.from_dict(d)
    d = m.to_dict()
    del d["name"]
    with pytest.raises(ManifestError):
        Manifest.from_dict(d)


# ---------------------------------------------------------------- 불변성
def test_frozen_and_copy_semantics():
    params = {"a": [1]}
    inputs = {"i": H1}
    m = mk(params=params, inputs=inputs)
    mid = m.manifest_id
    params["a"].append(2)
    inputs["j"] = H2
    assert m.manifest_id == mid and m.params == {"a": [1]} and m.inputs == {"i": H1}
    d = m.to_dict()
    d["params"]["a"].append(9)
    d["config"]["seed"] = 1234
    assert m.manifest_id == mid and m.params == {"a": [1]} and m.seed == 7
    with pytest.raises(AttributeError):
        m.name = "other"  # type: ignore[misc]
    assert dataclasses.is_dataclass(m)


# ---------------------------------------------------------------- 기본값(시계·git)
def test_default_created_utc_is_now():
    m = make_manifest("t", Config(), git_commit="unknown")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", m.created_utc)
    t = datetime.strptime(m.created_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    assert abs(t - time.time()) < 120


def test_default_git_commit_is_valid():
    m = make_manifest("t", Config(), created_utc="2026-01-01T00:00:00Z")
    assert m.git_commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", m.git_commit)


def test_read_git_commit(tmp_path, monkeypatch):
    try:
        want = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20)
    except FileNotFoundError:
        want = None
    if want is not None and want.returncode == 0:
        assert read_git_commit(REPO) == want.stdout.strip()
    got = read_git_commit(tmp_path)
    assert got == "unknown" or re.fullmatch(r"[0-9a-f]{40}", got)  # tmp가 저장소 안이면 해시일 수 있다
    assert read_git_commit(tmp_path / "does" / "not" / "exist") == "unknown"
    monkeypatch.setenv("PATH", str(tmp_path))  # git 없음
    assert read_git_commit(REPO) == "unknown"


# ---------------------------------------------------------------- 파일 입출력
def test_write_read_roundtrip_and_exact_bytes(tmp_path):
    m = mk()
    target = tmp_path / "a" / "b" / "manifest.json"
    out = write_manifest(m, target)
    assert Path(out) == target and target.is_file()
    assert target.read_bytes() == canonical_json({**m.to_dict(), "manifest_id": m.manifest_id}) + b"\n"
    back = read_manifest(target)
    assert back == m and back.manifest_id == m.manifest_id
    first = target.read_bytes()
    write_manifest(m, target)
    assert target.read_bytes() == first
    assert sorted(os.listdir(target.parent)) == ["manifest.json"]  # 임시 파일 잔여 없음
    other = mk(name="exp-002")
    write_manifest(other, str(target))
    assert read_manifest(str(target)) == other


def test_failed_write_leaves_old_file_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "m.json"
    write_manifest(mk(), target)
    old = target.read_bytes()

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_manifest(mk(name="exp-002"), target)
    monkeypatch.undo()
    assert target.read_bytes() == old
    assert sorted(os.listdir(tmp_path)) == ["m.json"]


def _rewrite(path, fn):
    d = json.loads(path.read_bytes())
    fn(d)
    path.write_text(json.dumps(d), encoding="utf-8")


def test_tamper_detection(tmp_path):
    m = mk()
    p = tmp_path / "m.json"

    def fresh():
        write_manifest(m, p)

    fresh()
    _rewrite(p, lambda d: d["params"].__setitem__("steps", 101))
    with pytest.raises(ManifestError):
        read_manifest(p)
    fresh()
    _rewrite(p, lambda d: d["config"].__setitem__("seed", 8))
    with pytest.raises(ManifestError):
        read_manifest(p)
    fresh()
    _rewrite(p, lambda d: d.__setitem__("git_commit", "f" * 40))
    with pytest.raises(ManifestError):
        read_manifest(p)
    fresh()
    _rewrite(p, lambda d: d.pop("manifest_id"))
    with pytest.raises(ManifestError):
        read_manifest(p)
    fresh()
    _rewrite(p, lambda d: d.__setitem__("zzz", 1))
    with pytest.raises(ManifestError):
        read_manifest(p)
    fresh()
    p.write_bytes(b"{not json")
    with pytest.raises(ManifestError):
        read_manifest(p)
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "missing.json")


def test_schema_major_mismatch_is_rejected_even_with_valid_id(tmp_path):
    p = tmp_path / "m.json"
    write_manifest(mk(), p)

    def to_v2(d):
        d["schema"] = "buc.manifest/v2"
        d["manifest_id"] = oracle_id(d)

    _rewrite(p, to_v2)
    with pytest.raises(ManifestError):
        read_manifest(p)


def test_created_utc_is_not_part_of_identity_in_file(tmp_path):
    m = mk()
    p = tmp_path / "m.json"
    write_manifest(m, p)
    _rewrite(p, lambda d: d.__setitem__("created_utc", "2031-01-01T00:00:00Z"))
    back = read_manifest(p)
    assert back.created_utc == "2031-01-01T00:00:00Z" and back.manifest_id == m.manifest_id
