"""디스패치 순서: 클래스 > 우선순위 > 삽입 순서. 입력 순서와 무관. not_before 게이트."""
import random

import pytest

from buc.rt import queue as Q


def _drain(q, clock=None):
    out = []
    while True:
        j = q.claim("w")
        if j is None:
            return out
        out.append(j.id)


def test_constants():
    assert Q.CLASSES == ("ab", "mrh", "train", "n")
    assert Q.STATES == ("queued", "running", "done", "failed", "cancelled")
    assert Q.SCHEMA_VERSION == 1
    assert Q.PRIORITY_LIMIT == 500_000


def test_class_order_ab_mrh_train_n_regardless_of_enqueue_order(make):
    q = make()
    for jid, cls in [("n1", "n"), ("t1", "train"), ("m1", "mrh"), ("a1", "ab")]:
        q.enqueue(jid, cls=cls, est_gpu_h=1.0)
    assert [j.id for j in q.list("queued")] == ["a1", "m1", "t1", "n1"]
    assert _drain(q) == ["a1", "m1", "t1", "n1"]


def test_priority_beats_fifo_within_class_and_class_beats_priority(make):
    q = make()
    q.enqueue("lo", cls="train", priority=1, est_gpu_h=1.0)
    q.enqueue("hi", cls="train", priority=9, est_gpu_h=1.0)
    q.enqueue("neg", cls="train", priority=-5, est_gpu_h=1.0)
    q.enqueue("ab_low", cls="ab", priority=-499_999, est_gpu_h=1.0)
    q.enqueue("n_high", cls="n", priority=499_999, est_gpu_h=1.0)
    assert _drain(q) == ["ab_low", "hi", "lo", "neg", "n_high"]


def test_ties_are_fifo_by_seq(make):
    q = make()
    ids = [f"j{i:02d}" for i in (7, 3, 9, 1, 5)]
    for jid in ids:
        q.enqueue(jid, est_gpu_h=0.5)
    assert [j.seq for j in q.list()] == sorted(j.seq for j in q.list())
    assert _drain(q) == ids  # id 사전순이 아니라 삽입 순서


def test_random_inputs_match_reference_order(make):
    rng = random.Random(20261010)
    q = make(max_running=1000)
    ref = []
    for i in range(120):
        cls = rng.choice(Q.CLASSES)
        pr = rng.randint(-50, 50)
        jid = f"job-{i}"
        q.enqueue(jid, cls=cls, priority=pr, est_gpu_h=0.25)
        ref.append((Q.CLASSES.index(cls), -pr, i, jid))
    expected = [r[3] for r in sorted(ref)]
    assert [j.id for j in q.list("queued")] == expected
    assert _drain(q) == expected


def test_effective_priority_matches_dispatch_order(make):
    q = make()
    rng = random.Random(7)
    for i in range(60):
        q.enqueue(f"e{i}", cls=rng.choice(Q.CLASSES), priority=rng.randint(-1000, 1000), est_gpu_h=0.1)
    jobs = q.list("queued")
    eff = [j.effective_priority for j in jobs]
    assert eff == sorted(eff, reverse=True)
    j = jobs[0]
    assert j.effective_priority == (3 - j.rank) * 1_000_000 + j.priority
    assert j.run_id == f"job:{j.id}"
    ab = q.enqueue("ab-x", cls="ab", priority=3, est_gpu_h=0.1)
    nn = q.enqueue("n-x", cls="n", priority=-3, est_gpu_h=0.1)
    assert (ab.rank, nn.rank) == (0, 3)
    assert ab.effective_priority == 3_000_003 and nn.effective_priority == -3


def test_not_before_gates_claim_until_time(make, clock):
    q = make()
    t0 = clock()
    q.enqueue("later", cls="ab", est_gpu_h=1.0, not_before=t0 + 100)
    q.enqueue("now", cls="n", est_gpu_h=1.0)
    assert q.claim("w").id == "now"
    assert q.claim("w") is None
    clock.advance(99.9)
    assert q.claim("w") is None
    clock.advance(0.1)  # 정확히 not_before: 통과
    assert q.claim("w").id == "later"


def test_claim_sets_running_fields(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0, payload={"cmd": ["python", "x.py"], "k": {"z": 1}})
    t = clock()
    j = q.claim("worker-1", lease_s=120.0)
    assert (j.state, j.attempt, j.worker_id) == ("running", 1, "worker-1")
    assert j.lease_until == pytest.approx(t + 120.0)
    assert j.started_at == pytest.approx(t) and j.error is None
    assert j.payload == {"cmd": ["python", "x.py"], "k": {"z": 1}}
    assert q.get("a") == j
    assert q.counts() == {"queued": 0, "running": 1, "done": 0, "failed": 0, "cancelled": 0}


def test_claim_empty_returns_none(make):
    assert make().claim("w") is None
