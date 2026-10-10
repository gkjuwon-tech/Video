"""여러 핸들(스레드)·여러 프로세스가 같은 큐·장부 파일을 열어도 캡·슬롯·유일 claim이 지켜진다."""
import json
import subprocess
import sys
import threading
from pathlib import Path

from buc.core.config import Config
from buc.core.ledger import open_ledger
from buc.rt import queue as Q

ROOT = Path(Q.__file__).resolve().parents[2]  # buc 패키지가 있는 디렉터리
T0 = 1_767_225_600.0 + 36_000.0


def _seed(tmp_path, n_jobs, est, cfg):
    qp, lp = tmp_path / "queue.sqlite", tmp_path / "ledger.sqlite"
    led = open_ledger(lp, clock=lambda: T0)
    q = Q.open_queue(qp, ledger=led, config=cfg, clock=lambda: T0, max_running=1000)
    for i in range(n_jobs):
        q.enqueue(f"j{i:03d}", est_gpu_h=est)
    q.close()
    led.close()
    return qp, lp


def _thread_drain(qp, lp, cfg, max_running, n_threads=8):
    got, errs = [], []
    barrier = threading.Barrier(n_threads)

    def worker(k):
        try:
            led = open_ledger(lp, clock=lambda: T0)
            q = Q.open_queue(qp, ledger=led, config=cfg, clock=lambda: T0, max_running=max_running)
            barrier.wait()
            mine = []
            while (j := q.claim(f"w{k}")) is not None:
                mine.append(j.id)
            got.append(mine)
            q.close()
            led.close()
        except BaseException as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=45)
    assert not errs, errs
    return [x for mine in got for x in mine]


def test_eight_threads_claim_exactly_cap_many(tmp_path):
    cfg = Config(budget_daily_gpu_h=5.0, budget_daily_usd=1000.0, budget_total_usd=1000.0)
    qp, lp = _seed(tmp_path, 24, 1.0, cfg)
    claimed = _thread_drain(qp, lp, cfg, max_running=1000)
    assert sorted(claimed) == [f"j{i:03d}" for i in range(5)]  # 같은 크기면 앞의 5개
    with open_ledger(lp) as led, Q.open_queue(qp, ledger=led, config=cfg, clock=lambda: T0) as q:
        assert q.counts()["running"] == 5 and q.counts()["queued"] == 19
        assert q.reserved("gpu_h") <= 5.0 + 1e-9


def test_eight_threads_respect_max_running_with_no_cap(tmp_path):
    cfg = Config()
    qp, lp = _seed(tmp_path, 20, 0.5, cfg)
    claimed = _thread_drain(qp, lp, cfg, max_running=3)
    assert len(claimed) == 3 and len(set(claimed)) == 3


def test_threads_with_priorities_claim_the_top_ones(tmp_path):
    cfg = Config(budget_daily_gpu_h=4.0)
    qp, lp = tmp_path / "q.sqlite", tmp_path / "l.sqlite"
    led = open_ledger(lp, clock=lambda: T0)
    q = Q.open_queue(qp, ledger=led, config=cfg, clock=lambda: T0, max_running=100)
    for i in range(10):
        q.enqueue(f"n{i}", cls="n", est_gpu_h=1.0)
    for i in range(3):
        q.enqueue(f"ab{i}", cls="ab", est_gpu_h=1.0)
    q.enqueue("m0", cls="mrh", est_gpu_h=1.0)
    q.close()
    led.close()
    claimed = _thread_drain(qp, lp, cfg, max_running=100)
    assert sorted(claimed) == ["ab0", "ab1", "ab2", "m0"]


_PROC = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from buc.core.config import Config
from buc.core.ledger import open_ledger
from buc.rt import queue as Q
qp, lp, name = sys.argv[2], sys.argv[3], sys.argv[4]
T0 = float(sys.argv[5])
cfg = Config(budget_daily_gpu_h=7.0)
led = open_ledger(lp, clock=lambda: T0)
q = Q.open_queue(qp, ledger=led, config=cfg, clock=lambda: T0, max_running=1000)
mine = []
while True:
    j = q.claim(name)
    if j is None:
        break
    mine.append(j.id)
print(json.dumps(mine))
"""


def test_three_processes_claim_exactly_cap_many(tmp_path):
    cfg = Config(budget_daily_gpu_h=7.0)
    qp, lp = _seed(tmp_path, 30, 1.0, cfg)
    procs = [subprocess.Popen([sys.executable, "-I", "-c", _PROC, str(ROOT), str(qp), str(lp), f"p{k}", str(T0)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for k in range(3)]
    claimed = []
    for p in procs:
        out, err = p.communicate(timeout=50)
        assert p.returncode == 0, err
        claimed += json.loads(out.strip().splitlines()[-1])
    assert sorted(claimed) == [f"j{i:03d}" for i in range(7)] and len(claimed) == 7
    with open_ledger(lp) as led, Q.open_queue(qp, ledger=led, config=cfg, clock=lambda: T0) as q:
        assert [j.id for j in q.list("running")] == sorted(claimed)
        assert all(j.attempt == 1 for j in q.list("running"))
