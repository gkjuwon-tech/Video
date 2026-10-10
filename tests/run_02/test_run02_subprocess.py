"""SubprocessExecutor: 종료 코드·로그 꼬리·타임아웃(그룹 kill)·하트비트."""
import os
import sys
import time

from buc.rt import runner as R

PY = sys.executable


def run(code, tmp_path, timeout_s=20.0, heartbeat_s=60.0, hb=None, env=None):
    ex = R.SubprocessExecutor()
    e = dict(os.environ)
    e.update(env or {})
    return ex.run([PY, "-c", code], cwd=str(tmp_path), env=e, timeout_s=timeout_s, heartbeat_s=heartbeat_s,
                  on_heartbeat=hb or (lambda: None))


def test_exit_code_cwd_env_and_stderr(tmp_path):
    code = ("import os,sys; print('cwd', os.getcwd()); print('v', os.environ['RUN02_X']); "
            "sys.stderr.write('err-line\\n'); sys.exit(7)")
    r = run(code, tmp_path, env={"RUN02_X": "hello"})
    assert isinstance(r, R.ExecResult)
    assert r.exit_code == 7 and r.timed_out is False
    assert "v hello" in r.log_tail and "err-line" in r.log_tail
    assert os.path.realpath(str(tmp_path)) in r.log_tail
    assert 0.0 < r.wall_s < 15.0


def test_log_tail_limits(tmp_path):
    r = run("for i in range(5000): print('line%05d' % i)", tmp_path)
    assert r.exit_code == 0
    lines = [ln for ln in r.log_tail.splitlines() if ln]
    assert 1 <= len(lines) <= 200
    assert lines[-1] == "line04999"
    assert len(r.log_tail.encode()) <= 16 * 1024


def test_timeout_kills_process_group(tmp_path):
    marker = tmp_path / "grandchild_alive"
    code = ("import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c','import time,pathlib; time.sleep(2.5); "
            f"pathlib.Path(r\"{marker}\").write_text(\"x\")']); time.sleep(60)")
    t = time.monotonic()
    r = run(code, tmp_path, timeout_s=0.7)
    el = time.monotonic() - t
    assert r.timed_out is True and r.exit_code != 0
    assert el < 10.0
    assert r.wall_s >= 0.6
    time.sleep(3.0)
    assert not marker.exists(), "타임아웃 시 프로세스 그룹 전체를 죽여야 한다"


def test_heartbeat_called_periodically(tmp_path):
    n = []
    r = run("import time; time.sleep(1.0)", tmp_path, heartbeat_s=0.1, hb=lambda: n.append(1))
    assert r.exit_code == 0 and r.timed_out is False
    assert 3 <= len(n) <= 20
