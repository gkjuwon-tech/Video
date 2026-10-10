"""RUN-03 체크포인트·HF Hub 비공개 저장소 동기.

바이트 I/O만 한다. 표준 라이브러리만 쓰며 ``huggingface_hub`` 는 ``HfHubClient`` 생성 때만 지연 import 한다.
계약서: ops/harness/contracts/RUN-03.md
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Protocol

DEFAULT_REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
DEFAULT_REPO_TYPE = "model"
TOKEN_ENV = "HF_TOKEN"
DEFAULT_PREFIX = "ckpt"
SIDECAR_SUFFIX = ".sha256.json"

_CHUNK = 1024 * 1024
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_EXT_RE = re.compile(r"[A-Za-z0-9]+")
_CKPT_RE = re.compile(r"step_([0-9]{9,})\.([A-Za-z0-9]+)")
_HEX64_RE = re.compile(r"[0-9a-fA-F]{64}")
_TEMP_SUFFIXES = (".tmp", ".part")


# ---------------------------------------------------------------- 예외
class HubSyncError(RuntimeError):
    """허브 동기 계열 예외의 뿌리."""


class HubAuthError(HubSyncError):
    pass


class HubNotFoundError(HubSyncError, FileNotFoundError):
    pass


class TransientHubError(HubSyncError):
    pass


class IntegrityError(HubSyncError):
    pass


class ConflictError(HubSyncError):
    pass


class RepoNotPrivateError(HubSyncError):
    pass


# ---------------------------------------------------------------- 클라이언트 이음매
class HubClient(Protocol):
    def list_files(self, prefix: str) -> list[str]: ...

    def upload_file(self, local: Path, remote: str) -> None: ...

    def download_file(self, remote: str, local: Path) -> None: ...

    def is_private(self) -> bool: ...


# ---------------------------------------------------------------- 비밀 처리
def redact(text: str, secrets: Iterable[str]) -> str:
    """비밀 문자열을 ``***`` 로 치환한다. 빈 비밀은 무시."""
    out = text
    for s in sorted({s for s in secrets if s}, key=len, reverse=True):
        out = out.replace(s, "***")
    return out


def resolve_token(env=None, token_path=None) -> Optional[str]:
    """환경변수 HF_TOKEN 우선, 없으면 토큰 파일. 비면 None."""
    e = os.environ if env is None else env
    v = e.get(TOKEN_ENV)
    if v is not None and v.strip():
        return v.strip()
    p = Path(token_path) if token_path is not None else Path.home() / ".cache" / "huggingface" / "token"
    try:
        txt = p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return txt or None


@dataclass(frozen=True)
class HubConfig:
    token: str = field(repr=False)
    repo_id: str = DEFAULT_REPO_ID
    repo_type: str = DEFAULT_REPO_TYPE


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 1.0
    factor: float = 2.0
    max_delay: float = 30.0

    def __post_init__(self):
        if not (self.max_attempts >= 1 and self.base_delay >= 0 and self.factor >= 1
                and self.max_delay >= self.base_delay):
            raise ValueError("invalid RetryPolicy")

    def delay(self, k: int) -> float:
        """k번째(1부터) 실패 뒤 대기 초."""
        return min(self.max_delay, self.base_delay * self.factor ** (k - 1))


@dataclass(frozen=True)
class SyncResult:
    ok: bool
    uploaded: bool = False
    remote_path: Optional[str] = None
    sha256: Optional[str] = None
    size: Optional[int] = None
    skipped: Optional[str] = None
    error: Optional[str] = None


class CheckpointRef(NamedTuple):
    run_id: str
    step: int
    remote_path: str


# ---------------------------------------------------------------- 순수 함수
def validate_run_id(s) -> str:
    if not isinstance(s, str) or _RUN_ID_RE.fullmatch(s) is None:
        raise ValueError(f"invalid run_id: {s!r}")
    return s


def validate_remote_path(p) -> str:
    if not isinstance(p, str) or not p:
        raise ValueError("remote path must be a non-empty string")
    if len(p) > 512:
        raise ValueError("remote path too long")
    if p.startswith("/") or "\\" in p or "\x00" in p:
        raise ValueError(f"invalid remote path: {p!r}")
    for seg in p.split("/"):
        if seg in ("", ".", ".."):
            raise ValueError(f"invalid remote path segment in {p!r}")
    return p


def checkpoint_name(step, ext: str = "pt") -> str:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be int")
    if step < 0:
        raise ValueError("step must be >= 0")
    if not isinstance(ext, str) or _EXT_RE.fullmatch(ext) is None:
        raise ValueError(f"invalid ext: {ext!r}")
    return "step_%09d.%s" % (step, ext)


def parse_checkpoint_name(name: str):
    if not isinstance(name, str):
        return None
    m = _CKPT_RE.fullmatch(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def checkpoint_path(run_id: str, step: int, ext: str = "pt", prefix: str = DEFAULT_PREFIX) -> str:
    validate_run_id(run_id)
    return f"{prefix}/{run_id}/{checkpoint_name(step, ext)}"


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def _sidecar_bytes(sha: str, size: int) -> bytes:
    return json.dumps({"v": 1, "sha256": sha, "size": size}, sort_keys=True, separators=(",", ":")).encode()


def _parse_sidecar(raw: bytes) -> tuple[str, int]:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise IntegrityError("sidecar is not valid JSON") from e
    if not isinstance(obj, dict) or obj.get("v") != 1 or isinstance(obj.get("v"), bool):
        raise IntegrityError("sidecar: unsupported or missing version")
    sha, size = obj.get("sha256"), obj.get("size")
    if not isinstance(sha, str) or _HEX64_RE.fullmatch(sha) is None:
        raise IntegrityError("sidecar: bad sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise IntegrityError("sidecar: bad size")
    return sha.lower(), size


# ---------------------------------------------------------------- 실물 어댑터
class HfHubClient:
    """huggingface_hub 어댑터. 저장소를 만들 때는 항상 비공개."""

    def __init__(self, *, repo_id: str, repo_type: str, token: str):
        try:
            import huggingface_hub as hub  # 지연 import
        except ImportError as e:
            raise HubSyncError("huggingface_hub is not installed") from None
        self._hub = hub
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._token = token
        self._api = hub.HfApi(token=token)

    def _translate(self, e: BaseException) -> Exception:
        if isinstance(e, HubSyncError):
            return e
        msg = redact(f"{type(e).__name__}: {e}", [self._token])
        status = getattr(getattr(e, "response", None), "status_code", None)
        names = {c.__name__ for c in type(e).__mro__}
        if names & {"EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError",
                    "LocalEntryNotFoundError"} and status != 401:
            return HubNotFoundError(msg)
        if status in (401, 403):
            return HubAuthError(msg)
        if status == 404:
            return HubNotFoundError(msg)
        if status is not None and (status in (408, 425, 429) or status >= 500):
            return TransientHubError(msg)
        mod = type(e).__module__ or ""
        if (isinstance(e, (ConnectionError, TimeoutError))
                or mod.startswith(("requests", "urllib3", "httpx", "httpcore", "ssl", "socket"))):
            return TransientHubError(msg)
        return HubSyncError(msg)

    def _ensure_private_repo(self) -> None:
        self._api.create_repo(repo_id=self.repo_id, repo_type=self.repo_type, private=True, exist_ok=True)

    def list_files(self, prefix: str) -> list[str]:
        try:
            files = self._api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
        except Exception as e:
            raise self._translate(e) from None
        return [f for f in files if f.startswith(prefix)]

    def upload_file(self, local: Path, remote: str) -> None:
        try:
            self._api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                  repo_id=self.repo_id, repo_type=self.repo_type)
        except Exception as e:
            raise self._translate(e) from None

    def download_file(self, remote: str, local: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="buc-hf-") as td:
            try:
                got = self._hub.hf_hub_download(repo_id=self.repo_id, filename=remote, repo_type=self.repo_type,
                                                token=self._token, local_dir=td)
            except Exception as e:
                raise self._translate(e) from None
            shutil.copyfile(got, local)

    def is_private(self) -> bool:
        try:
            try:
                info = self._api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type)
            except Exception as e:
                if isinstance(self._translate(e), HubNotFoundError):
                    self._ensure_private_repo()
                    return True
                raise
            return bool(getattr(info, "private", False))
        except Exception as e:
            raise self._translate(e) from None


# ---------------------------------------------------------------- HubSync
class HubSync:
    def __init__(self, client: Optional[HubClient], *, prefix: str = DEFAULT_PREFIX,
                 retry: RetryPolicy = RetryPolicy(), sleep: Callable[[float], None] = time.sleep,
                 secrets: Iterable[str] = (), log: Optional[logging.Logger] = None):
        validate_remote_path(prefix)
        self._client = client
        self.prefix = prefix
        self._retry_policy = retry
        self._sleep = sleep
        self._secrets = tuple(s for s in secrets if s)
        self._log = log if log is not None else logging.getLogger(__name__)
        self._private_ok = False

    @classmethod
    def from_env(cls, *, env=None, token_path=None, repo_id: str = DEFAULT_REPO_ID,
                 repo_type: str = DEFAULT_REPO_TYPE, client_factory=None, **kw) -> "HubSync":
        token = resolve_token(env, token_path)
        if token is None:
            return cls(None, **kw)
        factory = client_factory if client_factory is not None else HfHubClient
        client = factory(repo_id=repo_id, repo_type=repo_type, token=token)
        kw["secrets"] = tuple(kw.get("secrets", ())) + (token,)
        return cls(client, **kw)

    # --- 내부 보조
    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _require_client(self):
        if self._client is None:
            raise HubAuthError("hf_token=no")
        return self._client

    def _scrub(self, e: BaseException) -> BaseException:
        if not isinstance(e, HubSyncError):
            return e
        msg = str(e)
        red = redact(msg, self._secrets)
        if red == msg:
            return e
        try:
            return type(e)(red)
        except Exception:
            return HubSyncError(red)

    def _retry(self, fn):
        pol = self._retry_policy
        k = 1
        while True:
            try:
                return fn()
            except (TransientHubError, IntegrityError) as e:
                if k >= pol.max_attempts:
                    raise self._scrub(e) from None
                self._sleep(pol.delay(k))
                k += 1
            except HubSyncError as e:
                raise self._scrub(e) from None

    def verify_private(self) -> None:
        client = self._require_client()
        if self._private_ok:
            return
        if not self._retry(client.is_private):
            raise RepoNotPrivateError("repository is not private; refusing to upload")
        self._private_ok = True

    def _read_remote_sidecar(self, remote: str) -> tuple[str, int]:
        """원격 사이드카를 받아 파싱(없으면 HubNotFoundError, 불량이면 IntegrityError)."""
        client = self._require_client()

        def once():
            with tempfile.TemporaryDirectory(prefix="buc-sc-") as td:
                p = Path(td) / "sidecar.json"
                client.download_file(remote + SIDECAR_SUFFIX, p)
                return _parse_sidecar(p.read_bytes())

        return self._retry(once)

    # --- 업로드
    @staticmethod
    def _check_local(local) -> Path:
        p = Path(local)
        if p.name.lower().endswith(_TEMP_SUFFIXES):
            raise ValueError(f"refusing to upload temporary file: {p.name}")
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p

    def upload_file(self, local, remote: str, *, overwrite: bool = False) -> SyncResult:
        validate_remote_path(remote)
        p = self._check_local(local)
        client = self._require_client()
        self.verify_private()
        sha, size = _hash_file(p)
        try:
            rsha, rsize = self._read_remote_sidecar(remote)
        except (HubNotFoundError, IntegrityError):
            rsha = rsize = None  # 미완성 또는 불량 사이드카: 다시 올린다
        if rsha is not None:
            if (rsha, rsize) == (sha, size):
                return SyncResult(ok=True, uploaded=False, remote_path=remote, sha256=sha, size=size,
                                  skipped="up_to_date")
            if not overwrite:
                raise ConflictError(f"remote {remote} exists with different content")
        self._retry(lambda: client.upload_file(p, remote))
        with tempfile.TemporaryDirectory(prefix="buc-sc-") as td:
            sp = Path(td) / "sidecar.json"
            sp.write_bytes(_sidecar_bytes(sha, size))
            self._retry(lambda: client.upload_file(sp, remote + SIDECAR_SUFFIX))
        return SyncResult(ok=True, uploaded=True, remote_path=remote, sha256=sha, size=size)

    def _prepare_checkpoint(self, local, run_id, step) -> tuple[Path, str]:
        validate_run_id(run_id)
        p = Path(local)
        ext = p.suffix.lstrip(".")
        if not ext:
            raise ValueError(f"local file has no extension: {p.name}")
        name = checkpoint_name(step, ext)
        self._check_local(p)
        return p, f"{self.prefix}/{run_id}/{name}"

    def upload_checkpoint(self, local, run_id: str, step: int, *, overwrite: bool = False) -> SyncResult:
        p, remote = self._prepare_checkpoint(local, run_id, step)
        return self.upload_file(p, remote, overwrite=overwrite)

    def try_upload_checkpoint(self, local, run_id: str, step: int, *, overwrite: bool = False) -> SyncResult:
        validate_run_id(run_id)
        p = Path(local)
        if p.name.lower().endswith(_TEMP_SUFFIXES):
            raise ValueError(f"refusing to upload temporary file: {p.name}")
        ext = p.suffix.lstrip(".")
        if not ext:
            raise ValueError(f"local file has no extension: {p.name}")
        checkpoint_name(step, ext)  # 인자 오류는 그대로 올린다
        if self._client is None:
            self._log.info("checkpoint upload skipped: hf_token=no")
            return SyncResult(ok=False, skipped="no_token")
        try:
            return self.upload_checkpoint(p, run_id, step, overwrite=overwrite)
        except (HubSyncError, OSError) as e:
            err = redact(f"{type(e).__name__}: {e}", self._secrets)
            self._log.warning("checkpoint upload failed (best effort): %s", err)
            return SyncResult(ok=False, error=err)

    # --- 목록
    def list_checkpoints(self, run_id: str) -> list[CheckpointRef]:
        validate_run_id(run_id)
        client = self._require_client()
        base = f"{self.prefix}/{run_id}"
        files = set(self._retry(lambda: client.list_files(base)))
        out: list[tuple[int, str, str]] = []
        head = base + "/"
        for f in files:
            if not f.startswith(head):
                continue
            rest = f[len(head):]
            if "/" in rest:
                continue
            parsed = parse_checkpoint_name(rest)
            if parsed is None or (f + SIDECAR_SUFFIX) not in files:
                continue
            out.append((parsed[0], parsed[1], f))
        out.sort()
        return [CheckpointRef(run_id, s, f) for s, _e, f in out]

    def latest_checkpoint(self, run_id: str) -> Optional[CheckpointRef]:
        refs = self.list_checkpoints(run_id)
        return refs[-1] if refs else None

    # --- 다운로드
    def download_file(self, remote: str, dest, *, force: bool = False) -> Path:
        validate_remote_path(remote)
        client = self._require_client()
        dest = Path(dest)
        sha, size = self._read_remote_sidecar(remote)
        if not force and dest.is_file() and dest.stat().st_size == size:
            try:
                if _hash_file(dest)[0] == sha:
                    return dest
            except OSError:
                pass
        part = dest.with_name(dest.name + ".part")

        def once():
            part.parent.mkdir(parents=True, exist_ok=True)
            try:
                if part.exists():
                    part.unlink()
                client.download_file(remote, part)
                got_sha, got_size = _hash_file(part)
                if got_size != size or got_sha != sha:
                    raise IntegrityError(f"integrity check failed for {remote} "
                                         f"(size {got_size} vs {size})")
                os.replace(part, dest)
            finally:
                try:
                    part.unlink()
                except FileNotFoundError:
                    pass

        self._retry(once)
        return dest

    def download_checkpoint(self, run_id: str, dest_dir, step: Optional[int] = None, *,
                            force: bool = False) -> Path:
        refs = self.list_checkpoints(run_id)
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError("step must be int")
            refs = [r for r in refs if r.step == step]
        if not refs:
            raise HubNotFoundError(f"no checkpoint for run {run_id!r}" + ("" if step is None else f" step {step}"))
        ref = refs[-1]
        name = ref.remote_path.rsplit("/", 1)[-1]
        return self.download_file(ref.remote_path, Path(dest_dir) / name, force=force)

    def fetch_latest(self, run_id: str, dest_dir) -> Optional[Path]:
        ref = self.latest_checkpoint(run_id)
        if ref is None:
            return None
        return self.download_file(ref.remote_path, Path(dest_dir) / ref.remote_path.rsplit("/", 1)[-1])
