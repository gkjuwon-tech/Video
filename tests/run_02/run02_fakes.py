"""RUN-02 테스트용 가짜 원격(공유 origin + 클라이언트별 뷰)·가짜 실행기·주입 시계·가짜 허브."""
import copy
import json
from pathlib import Path

from buc.rt import runner as R

T0 = 1_791_600_000.0
SECRET = "hf_SuperSecretValue_9f8e7d"


class Clock:
    def __init__(self, t: float = T0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class Origin:
    """원격 브랜치 흉내: 파일 dict + 버전(커밋 수)."""

    def __init__(self, files=None):
        self.files = {k: (v.encode() if isinstance(v, str) else v) for k, v in (files or {}).items()}
        self.version = 0
        self.commits = []  # (message, sorted paths)

    def put(self, path, data):
        """다른 러너가 푸시한 것처럼 origin을 직접 바꾼다."""
        self.files[path] = data.encode() if isinstance(data, str) else data
        self.version += 1

    def json(self, path):
        b = self.files.get(path)
        return None if b is None else json.loads(b.decode())


class FakeRemote:
    """JobRemote 이음매. 로컬 뷰는 sync 때 origin을 복사한다. 뷰가 낡았으면 publish는 되돌리고 PushConflict."""

    def __init__(self, origin: Origin):
        self.origin = origin
        self.view = {}
        self.base = -1
        self.before_publish = None  # callable(files) -> None : 경쟁 주입
        self.fail_publish = None    # callable(files) -> bool : True면 OSError
        self.publishes = 0
        self.conflicts = 0
        self.syncs = 0

    def sync(self):
        self.syncs += 1
        self.view = copy.deepcopy(self.origin.files)
        self.base = self.origin.version

    def read(self, path):
        return self.view.get(path)

    def list(self, prefix):
        p = prefix.rstrip("/") + "/"
        return sorted(k for k in self.view if k.startswith(p))

    def publish(self, files, message):
        files = {k: (v.encode() if isinstance(v, str) else bytes(v)) for k, v in files.items()}
        if self.before_publish is not None:
            hook, self.before_publish = self.before_publish, None
            hook(files)
        if self.fail_publish is not None and self.fail_publish(files):
            raise OSError("network down")
        if self.base != self.origin.version:
            self.conflicts += 1
            self.sync()
            raise R.PushConflict("non-fast-forward")
        changed = {k: v for k, v in files.items() if self.origin.files.get(k) != v}
        if changed:
            self.origin.files.update(changed)
            self.origin.version += 1
            self.origin.commits.append((message, sorted(changed)))
        self.publishes += 1
        self.view = copy.deepcopy(self.origin.files)
        self.base = self.origin.version
        return "c%d" % self.origin.version


class FakeExecutor:
    """스크립트된 실행기. script는 callable(call_index, cmd, cwd, env, on_heartbeat) -> ExecResult 또는 ExecResult 목록."""

    def __init__(self, script=None, wall_s=10.0):
        self.calls = []
        self.script = script
        self.wall_s = wall_s

    def run(self, cmd, *, cwd, env, timeout_s, heartbeat_s, on_heartbeat):
        i = len(self.calls)
        self.calls.append({"cmd": list(cmd), "cwd": str(cwd), "env": dict(env), "timeout_s": timeout_s,
                           "heartbeat_s": heartbeat_s})
        s = self.script
        if s is None:
            return R.ExecResult(exit_code=0, timed_out=False, wall_s=self.wall_s, log_tail="ok\n")
        if isinstance(s, list):
            r = s[min(i, len(s) - 1)]
            return r(i, cmd, cwd, env, on_heartbeat) if callable(r) else r
        return s(i, cmd, cwd, env, on_heartbeat)


def ok(wall_s=10.0, log="ok\n"):
    return R.ExecResult(exit_code=0, timed_out=False, wall_s=wall_s, log_tail=log)


def fail(code=1, wall_s=5.0, timed_out=False):
    return R.ExecResult(exit_code=code, timed_out=timed_out, wall_s=wall_s, log_tail="boom\n")


class FakeHub:
    def __init__(self, fail=False):
        self.uploads = []
        self.fail = fail

    def upload_file(self, local, remote, *, overwrite=False):
        if self.fail:
            raise OSError("hub down")
        self.uploads.append((Path(local).read_bytes(), remote))
        return {"ok": True, "remote_path": remote}


def job_yaml(id="j1", kind="train", cmd=("python", "train.py"), **kw):
    lines = ["id: " + json.dumps(id), "kind: " + json.dumps(kind), "cmd: " + json.dumps(list(cmd))]
    for k, v in kw.items():
        lines.append(f"{k}: {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def state_doc(job_id, spec_hash, status, attempt, runner_id, lease_until, updated_at=T0, error=None):
    return json.dumps({"schema": R.STATE_SCHEMA, "job_id": job_id, "spec_hash": spec_hash, "status": status,
                       "attempt": attempt, "runner_id": runner_id, "lease_until": lease_until,
                       "updated_at": updated_at, "error": error}, sort_keys=True)


def spec_of(text, stem=None):
    return R.parse_job_yaml(text, stem=stem)
