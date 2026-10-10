"""슬롯 상한, 리스·하트비트·reap, 재시도·소진, 취소, 멱등 complete."""
import pytest

from buc.rt import queue as Q


def test_max_running_one_serializes_gpu_jobs(make):
    q = make(max_running=1)
    q.enqueue("a", est_gpu_h=1.0)
    q.enqueue("b", est_gpu_h=1.0)
    assert q.claim("w1").id == "a"
    assert q.claim("w2") is None
    q.complete("a", "w1")
    assert q.claim("w2").id == "b"


def test_max_running_two(make):
    q = make(max_running=2)
    for i in range(4):
        q.enqueue(f"j{i}", est_gpu_h=1.0)
    assert [q.claim("w").id for _ in range(2)] == ["j0", "j1"]
    assert q.claim("w") is None
    q.fail("j0", "w", "x", retry=False)
    assert q.claim("w").id == "j2"
    assert q.counts()["running"] == 2


def test_complete_records_result_and_is_idempotent_for_same_worker(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0)
    q.claim("w")
    clock.advance(30)
    done = q.complete("a", "w", result={"loss": 0.25, "ckpt": "a.pt"})
    assert done.state == "done" and done.result == {"loss": 0.25, "ckpt": "a.pt"}
    assert done.finished_at == pytest.approx(clock()) and done.lease_until is None
    clock.advance(10)
    again = q.complete("a", "w", result={"loss": 99})
    assert again == done                  # 멱등: 기존 done 그대로
    with pytest.raises(Q.LeaseLost):
        q.complete("a", "other")
    with pytest.raises(Q.LeaseLost):
        q.fail("a", "w", "late")
    with pytest.raises(Q.LeaseLost):
        q.heartbeat("a", "w")


def test_unknown_job_raises(make):
    q = make()
    for call in (lambda: q.get("nope"), lambda: q.complete("nope", "w"), lambda: q.heartbeat("nope", "w"),
                 lambda: q.fail("nope", "w"), lambda: q.cancel("nope")):
        with pytest.raises(Q.JobNotFound):
            call()
    assert issubclass(Q.JobNotFound, KeyError) and issubclass(Q.JobNotFound, Q.QueueError)


def test_wrong_worker_cannot_touch_running_job(make):
    q = make()
    q.enqueue("a", est_gpu_h=1.0)
    q.claim("w1")
    for call in (lambda: q.complete("a", "w2"), lambda: q.heartbeat("a", "w2"), lambda: q.fail("a", "w2")):
        with pytest.raises(Q.LeaseLost):
            call()
    assert q.get("a").state == "running" and q.get("a").worker_id == "w1"


def test_lease_expiry_requeues_and_next_claim_increments_attempt(make, clock):
    q = make(max_running=1)
    q.enqueue("a", est_gpu_h=1.0, max_attempts=3)
    t0 = clock()
    assert q.claim("w1", lease_s=100).attempt == 1
    clock.t = t0 + 99.999
    assert q.claim("w2") is None          # 아직 유효: 슬롯도 차 있음
    clock.t = t0 + 100                    # lease_until <= now 는 만료
    j = q.claim("w2", lease_s=100)
    assert (j.id, j.attempt, j.worker_id) == ("a", 2, "w2")
    for call in (lambda: q.complete("a", "w1"), lambda: q.heartbeat("a", "w1"), lambda: q.fail("a", "w1")):
        with pytest.raises(Q.LeaseLost):
            call()
    assert q.complete("a", "w2").state == "done"


def test_reap_returns_ids_in_seq_order_and_sets_error(make, clock):
    q = make()
    for i in range(3):
        q.enqueue(f"j{i}", est_gpu_h=0.1)
    t0 = clock()
    q.claim("w", lease_s=10)
    q.claim("w", lease_s=50)
    q.claim("w", lease_s=10)
    assert q.reap() == []
    clock.t = t0 + 20
    assert q.reap() == ["j0", "j2"]
    assert q.reap() == []
    j0 = q.get("j0")
    assert (j0.state, j0.error, j0.worker_id, j0.lease_until) == ("queued", "lease_expired", None, None)
    assert q.get("j1").state == "running"


def test_heartbeat_extends_lease(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0)
    t0 = clock()
    q.claim("w", lease_s=100)
    clock.t = t0 + 90
    j = q.heartbeat("a", "w", lease_s=100)
    assert j.lease_until == pytest.approx(t0 + 190)
    clock.t = t0 + 150
    assert q.reap() == []
    clock.t = t0 + 190
    assert q.reap() == ["a"]


def test_heartbeat_after_expiry_still_ok_if_not_reaped(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0)
    t0 = clock()
    q.claim("w", lease_s=100)
    clock.t = t0 + 150
    assert q.heartbeat("a", "w", lease_s=100).lease_until == pytest.approx(t0 + 250)
    assert q.reap() == []


def test_lease_expiry_exhausts_attempts(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0, max_attempts=2)
    for attempt in (1, 2):
        j = q.claim("w", lease_s=10)
        assert j.attempt == attempt
        clock.advance(10)
    assert q.reap() == ["a"]
    j = q.get("a")
    assert (j.state, j.attempt, j.error) == ("failed", 2, "lease_expired")
    assert q.claim("w") is None
    assert q.reserved("gpu_h") == 0.0


def test_fail_retry_until_max_attempts(make):
    q = make()
    q.enqueue("a", est_gpu_h=1.0, max_attempts=3)
    for attempt in (1, 2):
        j = q.claim("w")
        assert j.attempt == attempt
        f = q.fail("a", "w", f"err{attempt}")
        assert (f.state, f.error, f.worker_id, f.lease_until) == ("queued", f"err{attempt}", None, None)
    q.claim("w")
    f = q.fail("a", "w", "final")
    assert (f.state, f.attempt, f.error) == ("failed", 3, "final")
    assert f.finished_at is not None
    assert q.claim("w") is None


def test_fail_no_retry_and_retry_delay(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0)
    q.enqueue("b", est_gpu_h=1.0)
    q.claim("w")
    assert q.fail("a", "w", "fatal", retry=False).state == "failed"
    t0 = clock()
    q.claim("w")
    r = q.fail("b", "w", "flaky", retry_delay_s=60.0)
    assert r.state == "queued" and r.not_before == pytest.approx(t0 + 60)
    assert q.claim("w") is None
    clock.t = t0 + 60
    assert q.claim("w").id == "b" and q.get("b").attempt == 2


def test_cancel_queued_running_and_terminal(make):
    q = make()
    for jid in "abc":
        q.enqueue(jid, est_gpu_h=1.0)
    q.claim("w")                                   # a running
    assert q.cancel("b").state == "cancelled"      # queued
    assert q.cancel("a").state == "cancelled"      # running
    with pytest.raises(Q.LeaseLost):
        q.complete("a", "w")
    with pytest.raises(Q.LeaseLost):
        q.heartbeat("a", "w")
    assert q.claim("w").id == "c"
    assert q.claim("w") is None
    q.complete("c", "w")
    assert q.cancel("c").state == "done"           # 종결 상태는 불변
    assert q.cancel("a").state == "cancelled"      # 멱등
    assert q.counts() == {"queued": 0, "running": 0, "done": 1, "failed": 0, "cancelled": 2}


def test_terminal_states_are_final(make, clock):
    q = make()
    q.enqueue("a", est_gpu_h=1.0, max_attempts=1)
    q.claim("w")
    q.fail("a", "w", "x")
    assert q.get("a").state == "failed"
    clock.advance(10_000)
    assert q.reap() == [] and q.claim("w") is None
    assert q.get("a").state == "failed"


def test_list_filters_and_unknown_state(make):
    q = make()
    q.enqueue("a", est_gpu_h=1.0)
    q.enqueue("b", est_gpu_h=1.0)
    q.claim("w")
    assert [j.id for j in q.list("running")] == ["a"]
    assert [j.id for j in q.list("queued")] == ["b"]
    assert [j.id for j in q.list()] == ["a", "b"]
    with pytest.raises(ValueError):
        q.list("paused")


def test_jobs_are_frozen_snapshots(make):
    q = make()
    j = q.enqueue("a", est_gpu_h=1.0, payload={"x": [1]})
    with pytest.raises(Exception):
        j.state = "done"  # type: ignore[misc]
    j.payload["x"].append(2)
    assert q.get("a").payload == {"x": [1]}
