"""입력 검증, 멱등 enqueue, 영속(재오픈), 스키마 버전, 닫기, import 순수성."""
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import spend

from buc.core.config import Config
from buc.rt import queue as Q

ROOT = Path(Q.__file__).resolve().parents[2]  # buc 패키지가 있는 디렉터리


@pytest.mark.parametrize("bad", ["", "-x", ".x", "a b", "a/b", "ok\n", "x" * 65, "é"])
def test_bad_ids_rejected(make, bad):
    with pytest.raises(ValueError):
        make().enqueue(bad, est_gpu_h=1.0)


def test_good_ids_accepted(make):
    q = make()
    for jid in ("a", "A1", "x.y-z_1", "0abc", "k" * 64):
        assert q.enqueue(jid, est_gpu_h=0.1).id == jid


@pytest.mark.parametrize("kw", [
    dict(cls="bogus"), dict(cls=None), dict(priority=500_000), dict(priority=-500_000),
    dict(priority=1.5), dict(priority=True), dict(est_gpu_h=0.0), dict(est_gpu_h=-1.0),
    dict(est_gpu_h=float("nan")), dict(est_gpu_h=float("inf")), dict(est_gpu_h=True),
    dict(est_gpu_h="1"), dict(max_attempts=0), dict(max_attempts=True), dict(max_attempts=1.5),
    dict(payload=[1]), dict(payload={"x": {1, 2}}), dict(payload={"x": float("nan")}),
    dict(gpu=1), dict(not_before=float("nan")), dict(not_before="soon"),
])
def test_invalid_enqueue_arguments(make, kw):
    q = make()
    args = dict(est_gpu_h=1.0)
    args.update(kw)
    with pytest.raises((ValueError, TypeError)):
        q.enqueue("j", **args)
    assert q.counts()["queued"] == 0


def test_gpu_false_requires_zero_estimate(make):
    q = make()
    with pytest.raises(ValueError):
        q.enqueue("c", gpu=False, est_gpu_h=0.5)
    j = q.enqueue("c", gpu=False)
    assert j.gpu is False and j.est_gpu_h == 0.0


def test_priority_limit_edges_accepted(make):
    q = make()
    assert q.enqueue("hi", priority=499_999, est_gpu_h=0.1).priority == 499_999
    assert q.enqueue("lo", priority=-499_999, est_gpu_h=0.1).priority == -499_999


def test_int_estimate_normalized_to_float(make):
    j = make().enqueue("a", est_gpu_h=2)
    assert j.est_gpu_h == 2.0 and isinstance(j.est_gpu_h, float)


def test_enqueue_defaults(make, clock):
    j = make().enqueue("a", est_gpu_h=1.0)
    assert (j.cls, j.priority, j.gpu, j.payload, j.max_attempts, j.not_before) == ("train", 0, True, {}, 3, None)
    assert (j.state, j.attempt, j.worker_id, j.lease_until, j.started_at, j.finished_at, j.error, j.result) == \
        ("queued", 0, None, None, None, None, None, None)
    assert j.enqueued_at == pytest.approx(clock()) and j.seq == 1


def test_enqueue_idempotent_and_conflict(make):
    q = make()
    a = q.enqueue("a", cls="mrh", priority=2, est_gpu_h=1.5, payload={"k": 1, "z": [1, 2]})
    again = q.enqueue("a", cls="mrh", priority=2, est_gpu_h=1.5, payload={"z": [1, 2], "k": 1})
    assert again == a and q.counts()["queued"] == 1
    q.claim("w")
    running = q.enqueue("a", cls="mrh", priority=2, est_gpu_h=1.5, payload={"k": 1, "z": [1, 2]})
    assert running.state == "running" and running.attempt == 1   # 상태를 되돌리지 않는다
    for kw in (dict(cls="train"), dict(priority=3), dict(est_gpu_h=2.0), dict(payload={"k": 2}),
               dict(max_attempts=5), dict(not_before=1.0), dict(gpu=False)):
        args = dict(cls="mrh", priority=2, est_gpu_h=1.5, payload={"k": 1, "z": [1, 2]})
        args.update(kw)
        if kw.get("gpu") is False:
            args["est_gpu_h"] = 0.0
        with pytest.raises(Q.QueueError):
            q.enqueue("a", **args)


def test_payload_is_deep_copied_on_input(make):
    q = make()
    p = {"cmd": ["a", "b"]}
    q.enqueue("a", est_gpu_h=1.0, payload=p)
    p["cmd"].append("c")
    assert q.get("a").payload == {"cmd": ["a", "b"]}


def test_seq_monotonic_and_counts(make):
    q = make()
    for i in range(5):
        q.enqueue(f"j{i}", est_gpu_h=0.1)
    assert [j.seq for j in q.list()] == [1, 2, 3, 4, 5]
    assert q.counts() == {"queued": 5, "running": 0, "done": 0, "failed": 0, "cancelled": 0}


def test_invalid_open_arguments(ledger):
    cfg = Config()
    for kw in (dict(max_running=0), dict(max_running=True), dict(max_running=1.5),
               dict(usd_per_h=-1.0), dict(usd_per_h=float("nan")), dict(usd_per_h=True)):
        with pytest.raises((ValueError, TypeError)):
            Q.open_queue(":memory:", ledger=ledger, config=cfg, **kw)


def test_persistence_across_reopen(tmp_path, ledger, clock):
    path = tmp_path / "sub" / "queue.sqlite"
    cfg = Config(budget_daily_gpu_h=5.0)
    q = Q.open_queue(path, ledger=ledger, config=cfg, clock=clock, max_running=2)
    q.enqueue("a", cls="ab", est_gpu_h=1.0, payload={"p": 1})
    q.enqueue("b", cls="n", est_gpu_h=1.0)
    q.enqueue("c", cls="train", est_gpu_h=1.0)
    t0 = clock()
    ja = q.claim("w1", lease_s=100)
    q.close()
    q.close()  # 멱등
    assert path.exists()
    q2 = Q.open_queue(path, ledger=ledger, config=cfg, clock=clock, max_running=2)
    assert q2.get("a") == ja
    assert q2.get("a").lease_until == pytest.approx(t0 + 100)
    assert [j.id for j in q2.list("queued")] == ["c", "b"]
    assert q2.reserved("gpu_h") == pytest.approx(1.0)
    assert q2.heartbeat("a", "w1").lease_until == pytest.approx(t0 + 900)
    assert q2.enqueue("d", est_gpu_h=0.5).seq == 4          # seq 이어짐
    q2.close()


def test_file_schema_and_wal(tmp_path, ledger, clock):
    path = tmp_path / "q.sqlite"
    q = Q.open_queue(path, ledger=ledger, config=Config(), clock=clock)
    q.enqueue("a", est_gpu_h=1.0)
    con = sqlite3.connect(path)
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 1
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        cols = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
        assert {"seq", "id", "cls", "priority", "est_gpu_h", "gpu", "payload", "max_attempts", "not_before",
                "state", "attempt", "worker_id", "lease_until", "enqueued_at", "started_at", "finished_at",
                "error", "result"} <= cols
        assert con.execute("SELECT id, state, attempt FROM jobs").fetchall() == [("a", "queued", 0)]
    finally:
        con.close()
    q.close()


def test_newer_schema_version_refused(tmp_path, ledger, clock):
    path = tmp_path / "q.sqlite"
    Q.open_queue(path, ledger=ledger, config=Config(), clock=clock).close()
    con = sqlite3.connect(path)
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    with pytest.raises(Q.QueueError):
        Q.open_queue(path, ledger=ledger, config=Config(), clock=clock)


def test_closed_queue_is_unusable_and_context_manager(tmp_path, ledger, clock):
    with Q.open_queue(tmp_path / "q.sqlite", ledger=ledger, config=Config(), clock=clock) as q:
        q.enqueue("a", est_gpu_h=1.0)
    for call in (lambda: q.enqueue("b", est_gpu_h=1.0), lambda: q.claim("w"), lambda: q.get("a"),
                 lambda: q.list(), lambda: q.counts(), lambda: q.reserved("gpu_h")):
        with pytest.raises(Q.QueueError):
            call()
    q.close()


def test_memory_queues_are_independent(ledger, clock):
    a = Q.open_queue(":memory:", ledger=ledger, config=Config(), clock=clock)
    b = Q.open_queue(":memory:", ledger=ledger, config=Config(), clock=clock)
    a.enqueue("x", est_gpu_h=1.0)
    assert b.counts()["queued"] == 0
    a.close()
    b.close()


def test_import_is_pure_and_light(tmp_path):
    code = (
        "import sys, os; sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[2]); "
        "import buc.rt.queue as q; "
        "bad = [m for m in ('torch', 'numpy', 'yaml') if m in sys.modules]; "
        "print(repr(bad)); print(sorted(os.listdir('.')))"
    )
    work = tmp_path / "cwd"
    work.mkdir()
    r = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT), str(work)], capture_output=True, text=True,
                       timeout=50)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["[]", "[]"]


def test_ledger_unchanged_by_reading_paths(make, ledger, clock):
    q = make(daily=1.0)
    spend(ledger, "gpu_h", 0.5, "x", ts=clock(), key="k")
    before = ledger.events()
    q.enqueue("a", est_gpu_h=0.5)
    q.claim("w")
    assert ledger.events() == before
