"""run_job/run_once: 성공·재시도·소진·타임아웃·저널 재개·리스 상실·하트비트·지표·산출·장부·비밀·락."""
import fcntl
import hashlib
import json
from pathlib import Path

import pytest
from run02_fakes import SECRET, T0, FakeExecutor, FakeHub, FakeRemote, fail, job_yaml, ok, spec_of, state_doc

from buc.core.ledger import open_ledger
from buc.rt import runner as R


def put_job(origin, text):
    s = spec_of(text)
    origin.put(f"jobs/{s.id}.yaml", text)
    return s


def write_out(env, rel, data):
    p = Path(env["BUC_OUT_DIR"]) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def test_success_result_state_env_and_idempotent(origin, make, tmp_path):
    s = put_job(origin, job_yaml("j1", kind="render", env={"MODE": "fast"}))

    def body(i, cmd, cwd, env, hb):
        write_out(env, "metrics.json", json.dumps({"loss": 0.25, "psnr": 31.5}))
        return ok(wall_s=12.5)
    r, rem, ex = make(executor=FakeExecutor([body]))
    res = r.run_once()
    assert isinstance(res, R.JobResult)
    assert res.status == "done" and res.attempt == 1 and res.exit_code == 0 and res.timed_out is False
    assert res.metrics == {"loss": 0.25, "psnr": 31.5}
    assert res.spec_hash == s.spec_hash and res.job_id == "j1"

    call = ex.calls[0]
    env = call["env"]
    jd = tmp_path / "work_r1" / "j1" / s.spec_hash[:12]
    assert Path(call["cwd"]).resolve() == jd.resolve()
    assert env["BUC_JOB_ID"] == "j1" and env["BUC_JOB_KIND"] == "render"
    assert env["BUC_ATTEMPT"] == "1" and env["BUC_RESUME"] == "0"
    assert Path(env["BUC_JOB_DIR"]).resolve() == jd.resolve()
    assert Path(env["BUC_OUT_DIR"]).resolve() == (jd / "out").resolve()
    assert Path(env["BUC_METRICS"]).resolve() == (jd / "out" / "metrics.json").resolve()
    assert env["MODE"] == "fast" and env["PATH"] == "/usr/bin"
    assert call["cmd"] == ["python", "train.py"]
    assert abs(call["timeout_s"] - 3600.0) < 1e-9

    st = origin.json("runs/jobs/j1/state.json")
    assert st["status"] == "done" and st["attempt"] == 1 and st["lease_until"] is None
    rs = origin.json("runs/jobs/j1/result.json")
    assert rs["schema"] == "buc.job_result/v1" and rs["status"] == "done" and rs["runner_id"] == "r1"
    assert rs["exit_code"] == 0 and rs["timed_out"] is False and abs(rs["wall_s"] - 12.5) < 1e-9
    assert abs(rs["gpu_h"] - 12.5 / 3600) < 1e-12
    assert rs["metrics"] == {"loss": 0.25, "psnr": 31.5} and rs["spec_hash"] == s.spec_hash
    assert (jd / "attempts" / "1.json").is_file()
    # state와 result는 한 커밋
    last_paths = origin.commits[-1][1]
    assert "runs/jobs/j1/state.json" in last_paths and "runs/jobs/j1/result.json" in last_paths

    # 멱등: 다시 돌려도 실행 안 함
    assert r.run_once() is None
    assert len(ex.calls) == 1
    r2, _, ex2 = make(runner_id="r2")
    assert r2.run_once() is None and ex2.calls == []


def test_retry_then_done_preserves_job_dir(origin, make):
    s = put_job(origin, job_yaml("j1", max_attempts=3))

    def first(i, cmd, cwd, env, hb):
        Path(env["BUC_JOB_DIR"], "ckpt.txt").write_text("step=100")
        return fail(code=3)

    def second(i, cmd, cwd, env, hb):
        assert env["BUC_RESUME"] == "1" and env["BUC_ATTEMPT"] == "2"
        assert Path(env["BUC_JOB_DIR"], "ckpt.txt").read_text() == "step=100"
        return ok()
    r, _, ex = make(executor=FakeExecutor([first, second]))
    a = r.run_once()
    assert a.status == "retry" and a.attempt == 1 and a.exit_code == 3
    st = origin.json("runs/jobs/j1/state.json")
    assert st["status"] == "retry" and st["attempt"] == 1
    assert origin.json("runs/jobs/j1/result.json")["status"] == "retry"
    b = r.run_once()
    assert b.status == "done" and b.attempt == 2
    assert ex.calls[0]["cwd"] == ex.calls[1]["cwd"]
    assert s.id == "j1"


def test_exhaustion_and_timeout(origin, make):
    put_job(origin, job_yaml("j1", max_attempts=2))
    r, _, ex = make(executor=FakeExecutor([fail(timed_out=True, code=None), fail(code=2)]))
    a = r.run_once()
    assert a.status == "retry" and a.timed_out is True
    assert origin.json("runs/jobs/j1/result.json")["timed_out"] is True
    b = r.run_once()
    assert b.status == "failed" and b.attempt == 2
    assert origin.json("runs/jobs/j1/state.json")["status"] == "failed"
    assert r.run_once() is None and len(ex.calls) == 2


def test_spec_change_reruns_fresh(origin, make):
    s1 = put_job(origin, job_yaml("j1"))
    r, _, ex = make()
    assert r.run_once().status == "done"
    s2 = put_job(origin, job_yaml("j1", cmd=("python", "train.py", "--v2")))
    assert s2.spec_hash != s1.spec_hash
    res = r.run_once()
    assert res.status == "done" and res.attempt == 1 and res.spec_hash == s2.spec_hash
    assert ex.calls[1]["cmd"][-1] == "--v2"
    assert ex.calls[0]["cwd"] != ex.calls[1]["cwd"]


def test_journal_resume_no_reexecution_no_double_count(origin, make, tmp_path, clock):
    s = put_job(origin, job_yaml("j1"))
    led = open_ledger(":memory:", clock=clock)
    work = tmp_path / "shared_work"
    r, rem, ex = make(work=work, ledger=led, usd_per_h=0.5, executor=FakeExecutor([ok(wall_s=7200.0)]))
    rem.fail_publish = lambda files: "runs/jobs/j1/result.json" in files
    try:
        r.run_once()
    except Exception:
        pass
    assert len(ex.calls) == 1
    st = origin.json("runs/jobs/j1/state.json")
    assert st["status"] == "running" and st["attempt"] == 1 and st["runner_id"] == "r1"
    assert (work / "j1" / s.spec_hash[:12] / "attempts" / "1.json").is_file()

    # 같은 runner_id 로 재시작(리스는 아직 살아 있음)
    clock.advance(30)
    ex2 = FakeExecutor([ok(wall_s=1.0)])
    r2 = R.Runner(FakeRemote(origin), runner_id="r1", work_root=work, executor=ex2, clock=clock,
                  ledger=led, usd_per_h=0.5, environ={"PATH": "/usr/bin"})
    res = r2.run_once()
    assert ex2.calls == []
    assert res.status == "done" and res.attempt == 1 and abs(res.wall_s - 7200.0) < 1e-9
    assert origin.json("runs/jobs/j1/result.json")["status"] == "done"
    gpu = led.events(kind="gpu_h", run_id="job:j1")
    usd = led.events(kind="usd", run_id="job:j1")
    assert len(gpu) == 1 and abs(gpu[0].amount - 2.0) < 1e-9
    assert len(usd) == 1 and abs(usd[0].amount - 1.0) < 1e-9
    assert gpu[0].idem_key == f"job:j1:{s.spec_hash[:16]}:1:gpu_h"
    assert usd[0].idem_key == f"job:j1:{s.spec_hash[:16]}:1:usd"
    assert gpu[0].stage == "train"
    led.close()


def test_ledger_non_gpu_job(origin, make, clock):
    put_job(origin, job_yaml("c1", gpu=False))
    led = open_ledger(":memory:", clock=clock)
    r, _, _ = make(ledger=led, usd_per_h=0.0, executor=FakeExecutor([ok(wall_s=3600.0)]))
    assert r.run_once().status == "done"
    assert led.total("gpu_h") == 0.0
    assert origin.json("runs/jobs/c1/result.json")["gpu_h"] == 0.0
    led.close()


def test_lost_lease_writes_nothing(origin, make):
    s = put_job(origin, job_yaml("j1"))

    def body(i, cmd, cwd, env, hb):
        origin.put("runs/jobs/j1/state.json", state_doc("j1", s.spec_hash, "running", 2, "thief", T0 + 9999))
        return ok()
    r, _, _ = make(executor=FakeExecutor([body]))
    res = r.run_once()
    assert res.status == "lost"
    st = origin.json("runs/jobs/j1/state.json")
    assert st["runner_id"] == "thief" and st["attempt"] == 2
    assert "runs/jobs/j1/result.json" not in origin.files


def test_heartbeat_renews_lease(origin, make, clock):
    put_job(origin, job_yaml("j1"))
    seen = []

    def body(i, cmd, cwd, env, hb):
        for _ in range(2):
            clock.advance(100)
            hb()
            seen.append(origin.json("runs/jobs/j1/state.json"))
        return ok()
    r, _, ex = make(executor=FakeExecutor([body]), lease_s=300.0, heartbeat_s=60.0)
    assert r.run_once().status == "done"
    assert abs(ex.calls[0]["heartbeat_s"] - 60.0) < 1e-9
    assert seen[0]["status"] == "running" and abs(seen[0]["lease_until"] - (T0 + 100 + 300)) < 1e-6
    assert abs(seen[1]["lease_until"] - (T0 + 200 + 300)) < 1e-6
    assert r.last_activity >= T0 + 200


def test_heartbeat_failure_is_not_fatal(origin, make, clock):
    put_job(origin, job_yaml("j1"))
    holder = {}

    def body(i, cmd, cwd, env, hb):
        holder["rem"].fail_publish = lambda files: "runs/jobs/j1/result.json" not in files
        clock.advance(10)
        hb()
        holder["rem"].fail_publish = None
        return ok()
    r, rem, _ = make(executor=FakeExecutor([body]))
    holder["rem"] = rem
    assert r.run_once().status == "done"


def test_outputs_small_pushed_big_to_hub(origin, make):
    s = put_job(origin, job_yaml("j1", outputs=["*.txt", "img/*.png", "*.bin", "missing.*"]))
    small, png, big = b"hello world", b"\x89PNG" + b"x" * 100, bytes(range(256)) * 8

    def body(i, cmd, cwd, env, hb):
        write_out(env, "small.txt", small)
        write_out(env, "img/a.png", png)
        write_out(env, "big.bin", big)
        write_out(env, "ignored.log", b"not matched")
        return ok()
    hub = FakeHub()
    r, _, _ = make(executor=FakeExecutor([body]), hub=hub, max_push_bytes=1000)
    res = r.run_once()
    assert res.status == "done"
    outs = {o["path"]: o for o in origin.json("runs/jobs/j1/result.json")["outputs"]}
    assert set(outs) == {"small.txt", "img/a.png", "big.bin"}
    for rel, data in (("small.txt", small), ("img/a.png", png), ("big.bin", big)):
        assert outs[rel]["size"] == len(data) and outs[rel]["sha256"] == hashlib.sha256(data).hexdigest()
    assert outs["small.txt"]["pushed"] is True and outs["img/a.png"]["pushed"] is True
    assert origin.files["runs/jobs/j1/out/small.txt"] == small
    assert origin.files["runs/jobs/j1/out/img/a.png"] == png
    assert "runs/jobs/j1/out/big.bin" not in origin.files
    assert outs["big.bin"]["pushed"] is False
    want = f"jobs/j1/{s.spec_hash[:12]}/big.bin"
    assert outs["big.bin"]["remote"] == want
    assert hub.uploads == [(big, want)]
    assert "runs/jobs/j1/out/ignored.log" not in origin.files


def test_outputs_without_hub_and_hub_failure(origin, make):
    put_job(origin, job_yaml("j1", outputs=["*.bin"]))
    put_job(origin, job_yaml("j2", outputs=["*.bin"]))

    def body(i, cmd, cwd, env, hb):
        write_out(env, "big.bin", b"z" * 5000)
        return ok()
    r, _, _ = make(executor=FakeExecutor([body]), max_push_bytes=100)
    assert r.run_once().status == "done"
    o = origin.json("runs/jobs/j1/result.json")["outputs"][0]
    assert o["pushed"] is False and o["remote"] is None
    r2, _, _ = make(runner_id="r2", executor=FakeExecutor([body]), max_push_bytes=100, hub=FakeHub(fail=True))
    assert r2.run_once().status == "done"
    o2 = origin.json("runs/jobs/j2/result.json")["outputs"][0]
    assert o2["pushed"] is False and o2["remote"] is None


def test_invalid_metrics_is_not_fatal(origin, make):
    put_job(origin, job_yaml("j1"))

    def body(i, cmd, cwd, env, hb):
        write_out(env, "metrics.json", "{not json")
        return ok()
    r, _, _ = make(executor=FakeExecutor([body]))
    res = r.run_once()
    assert res.status == "done" and res.metrics is None
    assert origin.json("runs/jobs/j1/result.json")["metrics"] is None


def test_secrets_redacted_everywhere(origin, make):
    put_job(origin, job_yaml("j1"))

    def body(i, cmd, cwd, env, hb):
        assert env["HF_TOKEN"] == SECRET  # 서버 환경의 비밀은 잡에 전달된다
        return R.ExecResult(exit_code=1, timed_out=False, wall_s=1.0, log_tail=f"auth with {SECRET} failed\n")
    r, _, _ = make(executor=FakeExecutor([body]), environ={"PATH": "/usr/bin", "HF_TOKEN": SECRET})
    res = r.run_once()
    assert res.status == "retry"
    rs = origin.json("runs/jobs/j1/result.json")
    assert "***" in rs["log_tail"] and SECRET not in rs["log_tail"]
    for path, data in origin.files.items():
        assert SECRET.encode() not in data, path


def test_local_lock(origin, make, tmp_path):
    put_job(origin, job_yaml("j1"))
    work = tmp_path / "locked"
    work.mkdir()
    fh = open(work / "runner.lock", "a+")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        r, _, ex = make(work=work)
        with pytest.raises(R.LockHeld):
            r.run_once()
        assert ex.calls == []
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
    assert r.run_once().status == "done"


def test_serve_runs_until_empty(origin, make):
    for j in ("a", "b", "c"):
        put_job(origin, job_yaml(j))
    sleeps = []
    r, _, ex = make()
    n = r.serve(max_jobs=3, poll_s=5.0, sleep=sleeps.append)
    assert n == 3 and [c["env"]["BUC_JOB_ID"] for c in ex.calls] == ["a", "b", "c"]
    stops = iter([False, True])
    n2 = r.serve(poll_s=5.0, sleep=sleeps.append, stop=lambda: next(stops, True))
    assert n2 == 0
