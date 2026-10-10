"""GitRemote·Runner 끝-끝: 로컬 bare 저장소를 GitHub 대역으로 쓴다(네트워크 없음)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from buc.rt import runner as R

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git 없음")
PY = sys.executable


def git(cwd, *args):
    p = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=cwd,
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    return p.stdout


def clone(origin, dst):
    subprocess.run(["git", "clone", "-q", "-b", "main", str(origin), str(dst)], check=True, capture_output=True,
                   timeout=30)
    git(dst, "config", "commit.gpgsign", "false")
    return dst


def job_text(id, code, outputs=()):
    return (f"id: {id}\nkind: train\ngpu: false\ntimeout_s: 30\n"
            f"cmd: {json.dumps([PY, '-c', code])}\noutputs: {json.dumps(list(outputs))}\n")


METRIC_CODE = ("import json,os,pathlib; o=pathlib.Path(os.environ['BUC_OUT_DIR']); o.mkdir(parents=True,exist_ok=True); "
               "(o/'metrics.json').write_text(json.dumps({'loss':0.5,'attempt':int(os.environ['BUC_ATTEMPT'])})); "
               "(o/'s.txt').write_text('sample'); print('trained')")


@pytest.fixture
def repo(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    git(seed, "config", "commit.gpgsign", "false")
    (seed / "jobs").mkdir()
    (seed / "jobs" / "a.yaml").write_text(job_text("a", METRIC_CODE, outputs=["s.txt"]))
    (seed / "README.md").write_text("seed\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "seed")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True, capture_output=True)
    return origin


def show(origin, path):
    p = subprocess.run(["git", "--git-dir", str(origin), "show", f"main:{path}"], capture_output=True, timeout=30)
    return p.stdout if p.returncode == 0 else None


def test_gitremote_publish_conflict_and_recovery(repo, tmp_path):
    a = R.GitRemote(clone(repo, tmp_path / "A"), branch="main")
    b = R.GitRemote(clone(repo, tmp_path / "B"), branch="main")
    a.sync()
    b.sync()
    assert a.read("README.md") == b"seed\n" and a.read("nope.txt") is None
    assert "jobs/a.yaml" in a.list("jobs")
    a.publish({"x/one.txt": b"from A"}, "A writes")
    with pytest.raises(R.PushConflict):
        b.publish({"x/two.txt": b"from B"}, "B writes")
    assert b.read("x/one.txt") == b"from A"      # 로컬이 원격으로 되돌려짐
    assert b.read("x/two.txt") is None
    b.publish({"x/two.txt": b"from B"}, "B again")
    assert show(repo, "x/one.txt") == b"from A" and show(repo, "x/two.txt") == b"from B"
    head = git(tmp_path / "B", "rev-parse", "HEAD").strip()
    assert b.publish({"x/two.txt": b"from B"}, "noop") == head   # 변화 없음 = 커밋 없음


def test_end_to_end_run_and_idempotent(repo, tmp_path):
    rem = R.GitRemote(clone(repo, tmp_path / "A"), branch="main")
    r = R.Runner(rem, runner_id="gpu1", work_root=tmp_path / "work", heartbeat_s=0.2)
    res = r.run_once()
    assert res is not None and res.status == "done" and res.exit_code == 0
    assert res.metrics == {"loss": 0.5, "attempt": 1}
    st = json.loads(show(repo, "runs/jobs/a/state.json"))
    rs = json.loads(show(repo, "runs/jobs/a/result.json"))
    assert st["status"] == "done" and st["runner_id"] == "gpu1"
    assert rs["status"] == "done" and "trained" in rs["log_tail"] and rs["metrics"]["loss"] == 0.5
    assert show(repo, "runs/jobs/a/out/s.txt") == b"sample"
    n_commits = git(tmp_path / "A", "rev-list", "--count", "HEAD")
    # 다른 클론의 러너도 다시 돌리지 않는다
    r2 = R.Runner(R.GitRemote(clone(repo, tmp_path / "B"), branch="main"), runner_id="gpu2",
                  work_root=tmp_path / "work2")
    assert r2.run_once() is None
    assert git(tmp_path / "B", "rev-list", "--count", "HEAD") == n_commits


def test_two_clones_race_one_winner(repo, tmp_path):
    ra = R.Runner(R.GitRemote(clone(repo, tmp_path / "A"), branch="main"), runner_id="A", work_root=tmp_path / "wa")
    rb = R.Runner(R.GitRemote(clone(repo, tmp_path / "B"), branch="main"), runner_id="B", work_root=tmp_path / "wb")
    pa = ra.pending()
    pb = rb.pending()   # B의 뷰는 A의 claim 이전
    assert [s.id for s in pa] == ["a"] and [s.id for s in pb] == ["a"]
    assert ra.claim(pa[0]) == 1
    assert rb.claim(pb[0]) is None
    st = json.loads(show(repo, "runs/jobs/a/state.json"))
    assert st["runner_id"] == "A" and st["status"] == "running"


def test_main_cli_once(repo, tmp_path):
    a = clone(repo, tmp_path / "A")
    rc = R.main(["--repo", str(a), "--branch", "main", "--runner-id", "cli1", "--work-root", str(tmp_path / "w"),
                 "--once"])
    assert rc == 0
    assert json.loads(show(repo, "runs/jobs/a/state.json"))["status"] == "done"
    rc2 = R.main(["--repo", str(a), "--branch", "main", "--runner-id", "cli1", "--work-root", str(tmp_path / "w"),
                  "--once"])
    assert rc2 == 0
    assert Path(tmp_path / "w").is_dir()
