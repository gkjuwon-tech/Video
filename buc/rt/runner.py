"""RUN-02 pull 러너 v0: GitHub 브랜치에서 잡 명세를 당겨 실행하고 상태·지표·작은 산출물을 같은 브랜치로 푸시한다.

GPU 서버는 SSH가 막혀 있어 서버 쪽에서 당겨 가는 방식이 유일한 경로다. 멱등·재개가 핵심이다.
  * 같은 (id, spec_hash, attempt)는 최대 1회 실행·1회 계상한다(로컬 저널 + 장부 idem_key).
  * 푸시 경쟁에서 진 쪽은 실행하지 않고, 살아 있는 남의 리스는 빼앗지 않는다.
  * 잡 디렉터리는 시도 사이에 지우지 않는다(재개 재료).
  * 서버 환경의 비밀은 저장소에 올리지 않는다(log_tail·error·지표에서 ``***`` 로 가림).

표준 라이브러리만 쓴다. PyYAML 은 ``parse_job_yaml`` 안에서 지연 import 하고, 장부·허브는 주입받는다.
계약서: ops/harness/contracts/RUN-02.md
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import fcntl
import glob as _glob
import hashlib
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

__all__ = [
    "JOB_SCHEMA", "STATE_SCHEMA", "RESULT_SCHEMA", "KINDS", "JOBS_DIR", "RUNS_DIR",
    "RunnerError", "JobSpecError", "PushConflict", "LockHeld",
    "JobSpec", "parse_job_yaml", "JobRemote", "GitRemote", "ExecResult", "Executor",
    "SubprocessExecutor", "JobResult", "Runner", "main",
]

JOB_SCHEMA = "buc.job/v1"
STATE_SCHEMA = "buc.job_state/v1"
RESULT_SCHEMA = "buc.job_result/v1"
KINDS = ("train", "render")
JOBS_DIR = "jobs"
RUNS_DIR = "runs/jobs"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_ENV_KEY_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_SECRET_KEY_RE = re.compile(r"TOKEN|SECRET|PASSWORD|KEY", re.IGNORECASE)
_URL_CRED_RE = re.compile(r"(?<=//)[^/@\s]+(?::[^/@\s]*)?@")
_LOG_MAX_LINES = 200
_LOG_MAX_BYTES = 16 * 1024
_CLAIM_TRIES = 3
_PUBLISH_TRIES = 3
_SPEC_FIELDS = ("id", "kind", "cmd", "env", "timeout_s", "max_attempts", "outputs", "gpu", "priority", "enabled")
_HASH_FIELDS = ("id", "kind", "cmd", "env", "timeout_s", "outputs", "gpu")

_logger = logging.getLogger("buc.rt.runner")


# ---------------------------------------------------------------- 예외
class RunnerError(RuntimeError):
    """러너 계열 예외의 뿌리."""


class JobSpecError(RunnerError, ValueError):
    """잡 명세 위반."""


class PushConflict(RunnerError):
    """원격이 앞서 있어 푸시가 거절됨(로컬은 원격으로 되돌려진 뒤)."""


class LockHeld(RunnerError):
    """같은 work_root 의 다른 러너가 잠금을 쥐고 있음."""


# ---------------------------------------------------------------- 잡 명세
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm_outputs_glob(g) -> str:
    if not isinstance(g, str) or not g.strip():
        raise JobSpecError("outputs 항목은 비지 않은 문자열이어야 합니다")
    if g.startswith("/") or g.startswith("\\") or re.match(r"^[A-Za-z]:", g):
        raise JobSpecError(f"outputs 는 상대 glob 이어야 합니다: {g!r}")
    if ".." in re.split(r"[\\/]", g):
        raise JobSpecError(f"outputs 에 '..' 금지: {g!r}")
    return g


@dataclass(frozen=True, kw_only=True)
class JobSpec:
    id: str
    kind: str
    cmd: tuple
    env: dict
    timeout_s: float = 3600.0
    max_attempts: int = 3
    outputs: tuple = ()
    gpu: bool = True
    priority: int = 0
    enabled: bool = True

    def __post_init__(self):
        set_ = object.__setattr__
        if not isinstance(self.id, str) or _ID_RE.fullmatch(self.id) is None:
            raise JobSpecError(f"id 가 올바르지 않습니다: {self.id!r}")
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise JobSpecError(f"kind 는 {KINDS} 중 하나여야 합니다: {self.kind!r}")
        cmd = self.cmd
        if isinstance(cmd, (str, bytes)) or not isinstance(cmd, (list, tuple)):
            raise JobSpecError("cmd 는 문자열 리스트여야 합니다")
        if len(cmd) == 0 or not all(isinstance(c, str) and c != "" for c in cmd):
            raise JobSpecError("cmd 는 비지 않은 문자열의 비지 않은 리스트여야 합니다")
        set_(self, "cmd", tuple(cmd))
        env = self.env
        if env is None:
            env = {}
        if not isinstance(env, Mapping):
            raise JobSpecError("env 는 매핑이어야 합니다")
        for k, v in env.items():
            if not isinstance(k, str) or _ENV_KEY_RE.fullmatch(k) is None:
                raise JobSpecError(f"env 키가 올바르지 않습니다: {k!r}")
            if _SECRET_KEY_RE.search(k):
                raise JobSpecError(f"env 키에 비밀 이름 금지(서버 환경에서만 주입): {k!r}")
            if not isinstance(v, str):
                raise JobSpecError(f"env 값은 문자열이어야 합니다: {k}")
        set_(self, "env", dict(env))
        t = self.timeout_s
        if not _is_num(t) or not math.isfinite(float(t)) or float(t) <= 0:
            raise JobSpecError(f"timeout_s 는 유한한 양수여야 합니다: {t!r}")
        set_(self, "timeout_s", float(t))
        m = self.max_attempts
        if not isinstance(m, int) or isinstance(m, bool) or m < 1:
            raise JobSpecError(f"max_attempts 는 1 이상 정수여야 합니다: {m!r}")
        outs = self.outputs
        if outs is None:
            outs = ()
        if isinstance(outs, (str, bytes)) or not isinstance(outs, (list, tuple)):
            raise JobSpecError("outputs 는 문자열 리스트여야 합니다")
        set_(self, "outputs", tuple(_norm_outputs_glob(g) for g in outs))
        for name in ("gpu", "enabled"):
            if not isinstance(getattr(self, name), bool):
                raise JobSpecError(f"{name} 는 bool 이어야 합니다")
        p = self.priority
        if not isinstance(p, int) or isinstance(p, bool):
            raise JobSpecError(f"priority 는 정수여야 합니다: {p!r}")

    @classmethod
    def from_dict(cls, d) -> "JobSpec":
        if not isinstance(d, Mapping):
            raise JobSpecError("잡 명세는 매핑이어야 합니다")
        d = dict(d)
        if "schema" in d:
            if d["schema"] != JOB_SCHEMA:
                raise JobSpecError(f"schema 는 {JOB_SCHEMA} 여야 합니다: {d['schema']!r}")
            del d["schema"]
        unknown = sorted(set(map(str, d)) - set(_SPEC_FIELDS))
        if unknown:
            raise JobSpecError(f"모르는 키: {unknown}")
        for req in ("id", "kind", "cmd"):
            if req not in d:
                raise JobSpecError(f"필수 키 없음: {req}")
        d.setdefault("env", {})
        return cls(**d)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "cmd": list(self.cmd), "env": dict(self.env),
                "timeout_s": self.timeout_s, "max_attempts": self.max_attempts, "outputs": list(self.outputs),
                "gpu": self.gpu, "priority": self.priority, "enabled": self.enabled}

    @property
    def spec_hash(self) -> str:
        d = self.to_dict()
        h = {k: d[k] for k in _HASH_FIELDS}
        return hashlib.sha256(json.dumps(h, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()


def parse_job_yaml(text: str, *, stem: Optional[str] = None) -> JobSpec:
    try:
        import yaml  # 지연 import (import 순수성)
    except ImportError as e:  # pragma: no cover
        raise JobSpecError("PyYAML 이 필요합니다(pyyaml>=6)") from e
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise JobSpecError(f"YAML 파싱 실패: {e}") from None
    if not isinstance(doc, dict):
        raise JobSpecError("잡 명세 최상위는 매핑이어야 합니다")
    spec = JobSpec.from_dict(doc)
    if stem is not None and spec.id != stem:
        raise JobSpecError(f"id {spec.id!r} 가 파일 이름 {stem!r} 과 다릅니다")
    return spec


# ---------------------------------------------------------------- 원격 이음매
class JobRemote(Protocol):
    def sync(self) -> None: ...

    def read(self, path: str) -> Optional[bytes]: ...

    def list(self, prefix: str) -> list: ...

    def publish(self, files: Mapping[str, bytes], message: str) -> str: ...


def _scrub_url(text: str) -> str:
    return _URL_CRED_RE.sub("***@", text)


def _safe_rel(path: str) -> str:
    if not isinstance(path, str) or path == "" or path.startswith("/") or "\\" in path or "\x00" in path:
        raise RunnerError(f"저장소 상대 경로가 올바르지 않습니다: {path!r}")
    if any(p in ("", ".", "..") for p in path.split("/")):
        raise RunnerError(f"저장소 상대 경로가 올바르지 않습니다: {path!r}")
    return path


class GitRemote:
    """git CLI 어댑터. 작업 트리는 항상 원격 브랜치와 같게 유지하려 한다(sync = fetch + reset --hard)."""

    def __init__(self, repo_dir, *, branch: str, remote: str = "origin", git: str = "git",
                 timeout_s: float = 600.0):
        self.repo_dir = Path(repo_dir)
        self.branch = branch
        self.remote = remote
        self.git = git
        self.timeout_s = timeout_s

    # -- 내부
    def _run(self, *args, input_bytes: Optional[bytes] = None, ok=(0,)):
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        cmd = [self.git, *args]
        try:
            p = subprocess.run(cmd, cwd=str(self.repo_dir), capture_output=True, input=input_bytes, env=env,
                               timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            raise RunnerError(f"git {args[0]} 시간 초과") from None
        except OSError as e:
            raise RunnerError(f"git 실행 실패: {_scrub_url(str(e))}") from None
        if p.returncode not in ok:
            err = _scrub_url((p.stderr or b"").decode("utf-8", "replace").strip())
            raise _GitFailure(f"git {args[0]} 실패(rc={p.returncode}): {err}", p.returncode, err)
        return p

    def _tracking(self) -> str:
        return f"{self.remote}/{self.branch}"

    # -- JobRemote
    def sync(self) -> None:
        try:
            self._run("fetch", "--quiet", self.remote,
                      f"+refs/heads/{self.branch}:refs/remotes/{self.remote}/{self.branch}")
            self._run("reset", "--hard", "--quiet", self._tracking())
        except _GitFailure as e:
            raise RunnerError(str(e)) from None

    def read(self, path: str) -> Optional[bytes]:
        _safe_rel(path)
        try:
            p = self._run("cat-file", "blob", f"HEAD:{path}", ok=(0, 128))
        except _GitFailure as e:
            raise RunnerError(str(e)) from None
        return p.stdout if p.returncode == 0 else None

    def list(self, prefix: str) -> list:
        pre = prefix.strip("/")
        args = ["ls-files", "-z"]
        if pre:
            args += ["--", pre + "/"]
        try:
            p = self._run(*args)
        except _GitFailure as e:
            raise RunnerError(str(e)) from None
        return sorted(x for x in p.stdout.decode("utf-8", "replace").split("\x00") if x)

    def _head(self) -> str:
        return self._run("rev-parse", "HEAD").stdout.decode().strip()

    def publish(self, files: Mapping[str, bytes], message: str) -> str:
        paths = []
        try:
            for rel, data in files.items():
                _safe_rel(rel)
                dst = self.repo_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data.encode() if isinstance(data, str) else bytes(data))
                paths.append(rel)
            if paths:
                self._run("add", "-f", "--", *paths)
            diff = self._run("diff", "--cached", "--quiet", ok=(0, 1))
            if diff.returncode == 0:
                return self._head()
            self._run("-c", "user.name=buc-runner", "-c", "user.email=buc-runner@users.noreply.invalid",
                      "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message)
        except _GitFailure as e:
            self._recover()
            raise RunnerError(str(e)) from None
        try:
            self._run("push", "--quiet", self.remote, f"HEAD:refs/heads/{self.branch}")
        except _GitFailure as e:
            conflict = re.search(r"rejected|non-fast-forward|fetch first|stale info|failed to update ref|cannot lock ref",
                                 e.stderr, re.IGNORECASE) is not None
            self._recover()
            if conflict:
                raise PushConflict(_scrub_url(str(e))) from None
            raise RunnerError(str(e)) from None
        return self._head()

    def _recover(self) -> None:
        """실패 뒤 로컬을 원격 상태로 되돌린다(원격에 닿지 않으면 다음 sync 가 한다)."""
        try:
            self.sync()
        except RunnerError:
            try:
                self._run("reset", "--hard", "--quiet", "HEAD")
            except RunnerError:
                pass


class _GitFailure(RunnerError):
    def __init__(self, msg: str, rc: int, stderr: str):
        super().__init__(msg)
        self.rc = rc
        self.stderr = stderr


# ---------------------------------------------------------------- 실행기 이음매
@dataclass(frozen=True)
class ExecResult:
    exit_code: Optional[int]
    timed_out: bool
    wall_s: float
    log_tail: str


class Executor(Protocol):
    def run(self, cmd: Sequence[str], *, cwd, env: Mapping[str, str], timeout_s: float, heartbeat_s: float,
            on_heartbeat: Callable[[], None]) -> ExecResult: ...


def _tail_text(raw: bytes) -> str:
    raw = raw[-_LOG_MAX_BYTES:]
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        trailing = True
        lines = lines[:-1]
    else:
        trailing = False
    lines = lines[-_LOG_MAX_LINES:]
    out = b"\n".join(lines) + (b"\n" if trailing and lines else b"")
    text = out.decode("utf-8", "replace")
    enc = text.encode("utf-8")
    if len(enc) > _LOG_MAX_BYTES:
        text = enc[-_LOG_MAX_BYTES:].decode("utf-8", "ignore")
    return text


class SubprocessExecutor:
    """새 프로세스 그룹에서 실행. stdout+stderr 합침. 기한을 넘기면 그룹 전체를 kill 한다."""

    def run(self, cmd, *, cwd, env, timeout_s, heartbeat_s, on_heartbeat) -> ExecResult:
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(list(cmd), cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            return ExecResult(exit_code=127, timed_out=False, wall_s=time.monotonic() - t0,
                              log_tail=f"실행 실패: {e}\n")
        buf = bytearray()
        lock = threading.Lock()

        def reader():
            try:
                while True:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    with lock:
                        buf.extend(chunk)
                        if len(buf) > 4 * _LOG_MAX_BYTES:
                            del buf[:len(buf) - 2 * _LOG_MAX_BYTES]
            except (OSError, ValueError):
                pass

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        deadline = t0 + float(timeout_s)
        hb = max(float(heartbeat_s), 0.01)
        next_hb = t0 + hb
        timed_out = False
        while True:
            now = time.monotonic()
            wait = min(next_hb, deadline) - now
            try:
                proc.wait(timeout=max(wait, 0.0))
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                self._kill_group(proc)
                break
            if now >= next_hb:
                try:
                    on_heartbeat()
                except Exception as e:  # noqa: BLE001 - 하트비트 실패는 치명적이지 않다
                    _logger.warning("heartbeat 실패: %s", e)
                next_hb = time.monotonic() + hb
        th.join(timeout=2.0)
        if th.is_alive():
            self._kill_group(proc)
            th.join(timeout=2.0)
        else:
            self._kill_group(proc, wait=False)  # 남은 손자 프로세스 정리
        try:
            proc.stdout.close()
        except OSError:
            pass
        with lock:
            raw = bytes(buf)
        return ExecResult(exit_code=proc.returncode, timed_out=timed_out, wall_s=time.monotonic() - t0,
                          log_tail=_tail_text(raw))

    @staticmethod
    def _kill_group(proc, wait: bool = True) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if wait:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass


# ---------------------------------------------------------------- 러너
@dataclass(frozen=True, kw_only=True)
class JobResult:
    job_id: str
    spec_hash: str
    status: str
    attempt: int
    exit_code: Optional[int]
    timed_out: bool
    wall_s: float
    metrics: Optional[dict]
    outputs: tuple = ()


def _dumps(d: Mapping) -> bytes:
    return (json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def _loads_obj(b: Optional[bytes]) -> Optional[dict]:
    if b is None:
        return None
    try:
        d = json.loads(b.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return d if isinstance(d, dict) else None


class Runner:
    def __init__(self, remote, *, runner_id: str, work_root, executor=None, ledger=None, hub=None,
                 clock: Callable[[], float] = time.time, environ: Optional[Mapping[str, str]] = None,
                 lease_s: float = 900.0, heartbeat_s: float = 60.0, max_push_bytes: int = 5_000_000,
                 usd_per_h: float = 0.0, log: Optional[Callable[[str], None]] = None):
        if not isinstance(runner_id, str) or not runner_id:
            raise ValueError("runner_id 는 비지 않은 문자열이어야 합니다")
        self.remote = remote
        self.runner_id = runner_id
        self.work_root = Path(work_root).absolute()
        self.executor = executor if executor is not None else SubprocessExecutor()
        self.ledger = ledger
        self.hub = hub
        self._clock = clock
        self._environ = environ if environ is not None else os.environ
        self.lease_s = float(lease_s)
        self.heartbeat_s = float(heartbeat_s)
        self.max_push_bytes = int(max_push_bytes)
        self.usd_per_h = float(usd_per_h)
        self._logfn = log
        self.last_activity: Optional[float] = None

    # ------------------------------------------------------------ 보조
    def _log(self, msg: str) -> None:
        msg = self._redact(msg)
        if self._logfn is not None:
            try:
                self._logfn(msg)
            except Exception:  # noqa: BLE001
                pass
        else:
            _logger.info(msg)

    def _secrets(self) -> list:
        out = []
        for k, v in dict(self._environ).items():
            if isinstance(k, str) and isinstance(v, str) and len(v) >= 4 and _SECRET_KEY_RE.search(k):
                out.append(v)
        return sorted(set(out), key=len, reverse=True)

    def _redact(self, text):
        if not isinstance(text, str):
            return text
        for s in self._secrets():
            text = text.replace(s, "***")
        return text

    def _redact_obj(self, o):
        if isinstance(o, str):
            return self._redact(o)
        if isinstance(o, dict):
            return {self._redact(k) if isinstance(k, str) else k: self._redact_obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._redact_obj(v) for v in o]
        return o

    def _touch(self) -> None:
        self.last_activity = float(self._clock())

    @staticmethod
    def _state_path(job_id: str) -> str:
        return f"{RUNS_DIR}/{job_id}/state.json"

    @staticmethod
    def _result_path(job_id: str) -> str:
        return f"{RUNS_DIR}/{job_id}/result.json"

    def _job_dir(self, spec: JobSpec) -> Path:
        return self.work_root / spec.id / spec.spec_hash[:12]

    def _journal_path(self, spec: JobSpec, attempt: int) -> Path:
        return self._job_dir(spec) / "attempts" / f"{attempt}.json"

    def _read_state(self, job_id: str):
        """(state|None, unreadable:bool)"""
        raw = self.remote.read(self._state_path(job_id))
        if raw is None:
            return None, False
        d = _loads_obj(raw)
        if d is None or not isinstance(d.get("attempt"), int) or not isinstance(d.get("status"), str):
            return None, True
        return d, False

    def _state_doc(self, spec: JobSpec, status: str, attempt: int, *, lease_until: Optional[float],
                   error: Optional[str] = None) -> dict:
        return {"schema": STATE_SCHEMA, "job_id": spec.id, "spec_hash": spec.spec_hash, "status": status,
                "attempt": int(attempt), "runner_id": self.runner_id, "lease_until": lease_until,
                "updated_at": float(self._clock()), "error": self._redact(error)}

    # ------------------------------------------------------------ pending
    def _load_specs(self) -> list:
        specs = []
        for path in self.remote.list(JOBS_DIR):
            rest = path[len(JOBS_DIR) + 1:] if path.startswith(JOBS_DIR + "/") else path
            if "/" in rest or not rest.endswith(".yaml"):
                continue
            stem = rest[:-len(".yaml")]
            raw = self.remote.read(path)
            if raw is None:
                continue
            try:
                specs.append(parse_job_yaml(raw.decode("utf-8"), stem=stem))
            except (JobSpecError, UnicodeDecodeError) as e:
                self._log(f"잡 명세 건너뜀 {path}: {e}")
        return specs

    def _claimable(self, spec: JobSpec, st: Optional[dict]) -> bool:
        if st is None or st.get("spec_hash") != spec.spec_hash:
            return True
        status = st.get("status")
        if status == "retry":
            return True
        if status == "running":
            lease = st.get("lease_until")
            if not _is_num(lease) or float(lease) <= self._clock():
                return True
            return st.get("runner_id") == self.runner_id
        return False

    def pending(self) -> list:
        self.remote.sync()
        out = []
        for spec in self._load_specs():
            if not spec.enabled:
                continue
            st, bad = self._read_state(spec.id)
            if bad:
                self._log(f"상태 파일을 읽을 수 없어 건너뜀: {spec.id}")
                continue
            if self._claimable(spec, st):
                out.append(spec)
        out.sort(key=lambda s: (-s.priority, s.id))
        return out

    # ------------------------------------------------------------ claim
    def _decide(self, spec: JobSpec, st: Optional[dict]) -> Optional[int]:
        """다음 시도 번호. 가질 수 없으면 None."""
        if st is None or st.get("spec_hash") != spec.spec_hash:
            return 1
        status = st["status"]
        attempt = int(st["attempt"])
        if status == "retry":
            return attempt + 1
        if status == "running":
            lease = st.get("lease_until")
            expired = not _is_num(lease) or float(lease) <= self._clock()
            if expired:
                return attempt + 1
            if st.get("runner_id") == self.runner_id:
                if self._journal_path(spec, attempt).is_file():
                    return attempt
                return attempt + 1
        return None

    def claim(self, spec: JobSpec) -> Optional[int]:
        for i in range(_CLAIM_TRIES):
            if i > 0:
                self.remote.sync()
            st, bad = self._read_state(spec.id)
            if bad:
                self._log(f"상태 파일을 읽을 수 없어 claim 포기: {spec.id}")
                return None
            n = self._decide(spec, st)
            if n is None:
                return None
            if n > spec.max_attempts:
                doc = self._state_doc(spec, "failed", n - 1, lease_until=None, error="attempts_exhausted")
                try:
                    self.remote.publish({self._state_path(spec.id): _dumps(doc)},
                                        f"job {spec.id}: attempts exhausted")
                except PushConflict:
                    continue
                self._touch()
                return None
            doc = self._state_doc(spec, "running", n, lease_until=float(self._clock()) + self.lease_s)
            try:
                self.remote.publish({self._state_path(spec.id): _dumps(doc)},
                                    f"job {spec.id}: claim attempt {n} ({self.runner_id})")
            except PushConflict:
                continue
            self._touch()
            return n
        return None

    # ------------------------------------------------------------ 실행
    def _write_journal(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(_dumps(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read_journal(self, path: Path) -> Optional[dict]:
        try:
            d = _loads_obj(path.read_bytes())
        except OSError:
            return None
        if d is None or "wall_s" not in d or "timed_out" not in d or "exit_code" not in d:
            return None
        return d

    def _heartbeat(self, spec: JobSpec, attempt: int) -> None:
        try:
            self.remote.sync()
            st, _ = self._read_state(spec.id)
            if (st is None or st.get("runner_id") != self.runner_id or st.get("attempt") != attempt
                    or st.get("spec_hash") != spec.spec_hash):
                self._log(f"하트비트: 리스를 잃었음 {spec.id}#{attempt}")
                return
            doc = self._state_doc(spec, "running", attempt, lease_until=float(self._clock()) + self.lease_s)
            self.remote.publish({self._state_path(spec.id): _dumps(doc)}, f"job {spec.id}: heartbeat #{attempt}")
            self._touch()
        except Exception as e:  # noqa: BLE001 - 하트비트 실패는 로그만
            self._log(f"하트비트 실패 {spec.id}#{attempt}: {e}")

    def _job_env(self, spec: JobSpec, attempt: int, jd: Path) -> dict:
        env = dict(self._environ)
        env.update(spec.env)
        out = jd / "out"
        env.update({"BUC_JOB_ID": spec.id, "BUC_JOB_KIND": spec.kind, "BUC_ATTEMPT": str(attempt),
                    "BUC_RESUME": "1" if attempt > 1 else "0", "BUC_JOB_DIR": str(jd),
                    "BUC_OUT_DIR": str(out), "BUC_METRICS": str(out / "metrics.json")})
        return env

    def _execute(self, spec: JobSpec, attempt: int) -> dict:
        jd = self._job_dir(spec)
        (jd / "out").mkdir(parents=True, exist_ok=True)
        try:
            (jd / "out" / "metrics.json").unlink()  # 이전 시도의 낡은 지표 제거
        except FileNotFoundError:
            pass
        env = self._job_env(spec, attempt, jd)
        res = self.executor.run(list(spec.cmd), cwd=str(jd), env=env, timeout_s=spec.timeout_s,
                                heartbeat_s=self.heartbeat_s,
                                on_heartbeat=lambda: self._heartbeat(spec, attempt))
        journal = {"attempt": attempt, "exit_code": res.exit_code, "timed_out": bool(res.timed_out),
                   "wall_s": float(res.wall_s), "log_tail": res.log_tail, "finished_at": float(self._clock())}
        self._write_journal(self._journal_path(spec, attempt), journal)
        self._touch()
        return journal

    def _collect_metrics(self, jd: Path):
        p = jd / "out" / "metrics.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return d if isinstance(d, dict) else None

    def _collect_outputs(self, spec: JobSpec, jd: Path):
        """(결과 항목 목록, 저장소로 올릴 {경로: 바이트})"""
        out_dir = jd / "out"
        real_out = os.path.realpath(out_dir)
        rels = set()
        for pat in spec.outputs:
            for m in _glob.glob(pat, root_dir=str(out_dir), recursive=True):
                full = os.path.realpath(os.path.join(real_out, m))
                if not (full == real_out or full.startswith(real_out + os.sep)):
                    continue
                if os.path.isfile(full):
                    rels.add(m.replace(os.sep, "/"))
        items, repo_files, used = [], {}, 0
        secrets = [s.encode("utf-8") for s in self._secrets()]
        for rel in sorted(rels):
            local = os.path.realpath(os.path.join(real_out, rel))
            try:
                data = Path(local).read_bytes()
            except OSError as e:
                self._log(f"산출물 읽기 실패 {rel}: {e}")
                continue
            sha = hashlib.sha256(data).hexdigest()
            item = {"path": rel, "size": len(data), "sha256": sha, "pushed": False, "remote": None}
            leaky = any(s in data for s in secrets)
            if leaky:
                self._log(f"산출물에 비밀 값이 있어 저장소에 올리지 않음: {rel}")
            if (not leaky and len(data) <= self.max_push_bytes and used + len(data) <= self.max_push_bytes):
                used += len(data)
                item["pushed"] = True
                repo_files[f"{RUNS_DIR}/{spec.id}/out/{rel}"] = data
            elif self.hub is not None:
                remote_path = f"jobs/{spec.id}/{spec.spec_hash[:12]}/{rel}"
                try:
                    self.hub.upload_file(Path(local), remote_path, overwrite=True)
                    item["remote"] = remote_path
                except Exception as e:  # noqa: BLE001 - 허브 실패는 치명적이지 않다
                    self._log(f"허브 업로드 실패 {rel}: {e}")
            items.append(item)
        return items, repo_files

    def _record_ledger(self, spec: JobSpec, attempt: int, wall_s: float) -> None:
        if self.ledger is None:
            return
        try:
            from buc.core.ledger import LedgerEvent
        except ImportError:  # pragma: no cover
            return
        gpu_h = wall_s / 3600.0 if spec.gpu else 0.0
        usd = self.usd_per_h * wall_s / 3600.0
        meta = {"job_id": spec.id, "attempt": attempt, "spec_hash": spec.spec_hash, "runner_id": self.runner_id}
        for kind, amount in (("gpu_h", gpu_h), ("usd", usd)):
            if amount <= 0:
                continue
            try:
                self.ledger.record(LedgerEvent(
                    kind=kind, amount=amount, run_id=f"job:{spec.id}", stage=spec.kind, meta=meta,
                    idem_key=f"job:{spec.id}:{spec.spec_hash[:16]}:{attempt}:{kind}"))
            except Exception as e:  # noqa: BLE001
                self._log(f"장부 기록 실패 {spec.id}#{attempt} {kind}: {e}")

    def _owns(self, spec: JobSpec, attempt: int) -> bool:
        st, _ = self._read_state(spec.id)
        return (st is not None and st.get("runner_id") == self.runner_id and st.get("attempt") == attempt
                and st.get("spec_hash") == spec.spec_hash)

    def run_job(self, spec: JobSpec, attempt: int) -> JobResult:
        jd = self._job_dir(spec)
        journal = self._read_journal(self._journal_path(spec, attempt))
        if journal is None:
            journal = self._execute(spec, attempt)
        else:
            self._log(f"저널로 마무리(재실행 없음): {spec.id}#{attempt}")
        exit_code = journal["exit_code"]
        timed_out = bool(journal["timed_out"])
        wall_s = float(journal["wall_s"])
        log_tail = self._redact(str(journal.get("log_tail", "")))
        ok = exit_code == 0 and not timed_out
        status = "done" if ok else ("retry" if attempt < spec.max_attempts else "failed")
        metrics = self._redact_obj(self._collect_metrics(jd))
        items, repo_files = self._collect_outputs(spec, jd)
        gpu_h = wall_s / 3600.0 if spec.gpu else 0.0
        error = None
        if not ok:
            error = "timeout" if timed_out else f"exit_code={exit_code}"

        def lost() -> JobResult:
            return JobResult(job_id=spec.id, spec_hash=spec.spec_hash, status="lost", attempt=attempt,
                             exit_code=exit_code, timed_out=timed_out, wall_s=wall_s, metrics=metrics,
                             outputs=tuple(items))

        recorded = False
        for _ in range(_PUBLISH_TRIES):
            self.remote.sync()
            if not self._owns(spec, attempt):
                self._log(f"리스 상실, 쓰지 않음: {spec.id}#{attempt}")
                return lost()
            if not recorded:
                self._record_ledger(spec, attempt, wall_s)
                recorded = True
            state = self._state_doc(spec, status, attempt, lease_until=None, error=error)
            result = {"schema": RESULT_SCHEMA, "job_id": spec.id, "spec_hash": spec.spec_hash, "status": status,
                      "attempt": attempt, "runner_id": self.runner_id, "exit_code": exit_code,
                      "timed_out": timed_out, "wall_s": wall_s, "gpu_h": gpu_h, "metrics": metrics,
                      "outputs": items, "log_tail": log_tail}
            files = {self._state_path(spec.id): _dumps(state), self._result_path(spec.id): _dumps(result)}
            files.update(repo_files)
            try:
                self.remote.publish(files, f"job {spec.id}: attempt {attempt} {status}")
            except PushConflict:
                continue
            self._touch()
            return JobResult(job_id=spec.id, spec_hash=spec.spec_hash, status=status, attempt=attempt,
                             exit_code=exit_code, timed_out=timed_out, wall_s=wall_s, metrics=metrics,
                             outputs=tuple(items))
        raise PushConflict(f"결과 푸시가 {_PUBLISH_TRIES}회 연속 충돌: {spec.id}#{attempt}")

    # ------------------------------------------------------------ 구동
    def run_once(self) -> Optional[JobResult]:
        self.work_root.mkdir(parents=True, exist_ok=True)
        fh = open(self.work_root / "runner.lock", "a+")
        try:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise LockHeld(f"다른 러너가 {self.work_root} 를 쓰는 중") from None
            try:
                for spec in self.pending():
                    n = self.claim(spec)
                    if n is None:
                        continue
                    return self.run_job(spec, n)
                return None
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()

    def serve(self, *, max_jobs: Optional[int] = None, poll_s: float = 300.0, sleep=time.sleep,
              stop: Optional[Callable[[], bool]] = None) -> int:
        n = 0
        while True:
            if stop is not None and stop():
                break
            if max_jobs is not None and n >= max_jobs:
                break
            try:
                res = self.run_once()
            except LockHeld:
                raise
            except (RunnerError, OSError) as e:
                self._log(f"run_once 실패: {e}")
                res = None
            if res is not None:
                n += 1
                continue
            sleep(poll_s)
        return n


# ---------------------------------------------------------------- CLI
def _build_hub():
    if not os.environ.get("HF_TOKEN"):
        return None
    try:
        from buc.rt.hf_sync import HubSync
        return HubSync.from_env()
    except Exception as e:  # noqa: BLE001 - 허브는 선택 사항
        _logger.warning("허브를 쓸 수 없음: %s", type(e).__name__)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="buc.rt.runner", description="BUC pull 러너")
    ap.add_argument("--repo", required=True, help="잡 저장소의 로컬 클론 경로")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--runner-id", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--once", action="store_true", help="한 번만 당겨 실행하고 끝낸다")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--lease-s", type=float, default=900.0)
    ap.add_argument("--heartbeat-s", type=float, default=60.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    runner = Runner(GitRemote(a.repo, branch=a.branch), runner_id=a.runner_id, work_root=a.work_root,
                    hub=_build_hub(), lease_s=a.lease_s, heartbeat_s=a.heartbeat_s)
    try:
        if a.once:
            res = runner.run_once()
            if res is None:
                _logger.info("실행할 잡 없음")
            else:
                _logger.info("%s#%d -> %s", res.job_id, res.attempt, res.status)
        else:
            runner.serve(max_jobs=a.max_jobs)
    except LockHeld as e:
        print(f"LockHeld: {e}", file=sys.stderr)
        return 75
    except RunnerError as e:
        print(f"RunnerError: {_scrub_url(str(e))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
