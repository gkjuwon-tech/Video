"""GPU-h·USD 예산 집행: 경계, 허용 오차, UTC일 재개, 백필, held 공식."""
import pytest
from conftest import DAY0, H, spend

from buc.rt import queue as Q


def _claim_all(q, worker="w"):
    out = []
    while True:
        j = q.claim(worker)
        if j is None:
            return out
        out.append(j.id)


def test_daily_cap_exact_boundary(make):
    q = make(daily=3.0)
    for i in range(5):
        q.enqueue(f"j{i}", est_gpu_h=1.0)
    assert _claim_all(q) == ["j0", "j1", "j2"]  # 3.0 <= 3.0 통과, 4번째는 4.0 > 3.0
    assert q.reserved("gpu_h") == pytest.approx(3.0, abs=1e-12)
    assert q.counts()["queued"] == 2


def test_float_sum_tolerance_at_cap(make):
    q = make(daily=0.3)  # 0.1+0.1+0.1 == 0.30000000000000004 이지만 +1e-9 허용
    for i in range(4):
        q.enqueue(f"j{i}", est_gpu_h=0.1)
    assert _claim_all(q) == ["j0", "j1", "j2"]


def test_just_over_cap_is_refused(make):
    q = make(daily=3.0)
    q.enqueue("a", est_gpu_h=2.0)
    q.enqueue("b", est_gpu_h=1.0 + 1e-6)
    assert q.claim("w").id == "a"
    assert q.claim("w") is None
    assert q.get("b").state == "queued"


def test_total_cap_enforced_across_days(make, ledger, clock):
    q = make(daily=10.0, total=2.5)
    spend(ledger, "gpu_h", 2.0, "old", ts=DAY0 - 5 * 86400, key="old")  # 지난 5일 전 지출, 누적에만 잡힘
    q.enqueue("a", est_gpu_h=0.5)
    q.enqueue("b", est_gpu_h=0.5 + 1e-3)
    assert q.claim("w").id == "a"      # 2.0 + 0.5 = 2.5
    assert q.claim("w") is None        # 누적 초과


def test_ledger_spend_counts_against_daily_cap(make, ledger, clock):
    q = make(daily=3.0)
    spend(ledger, "gpu_h", 2.5, "elsewhere", ts=clock(), key="s1")
    q.enqueue("big", est_gpu_h=1.0)
    q.enqueue("small", est_gpu_h=0.5)
    assert q.claim("w").id == "small"  # 2.5 + 0.5 = 3.0
    assert q.claim("w") is None


def test_exhausted_daily_budget_resumes_next_utc_day(make, ledger, clock):
    q = make(daily=2.0)
    spend(ledger, "gpu_h", 2.0, "r", ts=DAY0 + 23 * H + 1800, key="s")
    clock.t = DAY0 + 23 * H + 3000          # 같은 UTC일, 23:50
    q.enqueue("nightly", cls="ab", est_gpu_h=1.5)
    assert q.claim("w") is None
    clock.t = DAY0 + 86400 - 1e-3           # 자정 직전
    assert q.claim("w") is None
    clock.t = DAY0 + 86400                  # 다음 UTC일 시작: 일 합이 0으로
    assert q.claim("w").id == "nightly"


def test_claim_with_explicit_now_uses_that_utc_day(make, ledger, clock):
    q = make(daily=1.0)
    spend(ledger, "gpu_h", 1.0, "r", ts=DAY0 + H, key="s")
    q.enqueue("a", est_gpu_h=0.5)
    assert q.claim("w", now=DAY0 + 2 * H) is None
    assert q.claim("w", now=DAY0 + 86400 + H).id == "a"


def test_unaffordable_head_is_skipped_backfill_keeps_it_queued(make):
    q = make(daily=2.0)
    q.enqueue("s1", cls="n", est_gpu_h=1.0)
    assert q.claim("w").id == "s1"
    q.enqueue("big", cls="ab", est_gpu_h=1.8)
    q.enqueue("s2", cls="n", est_gpu_h=0.9)
    q.enqueue("s3", cls="n", est_gpu_h=0.9)
    assert q.claim("w").id == "s2"       # big(1.0+1.8>2.0)을 건너뛰고 s2(1.0+0.9<=2.0)
    assert q.get("big").state == "queued"
    assert q.claim("w") is None          # s3: 1.9+0.9>2.0, big 도 불가
    assert [j.id for j in q.list("queued")] == ["big", "s3"]


def test_release_on_complete_frees_budget_when_job_spent_nothing(make):
    q = make(daily=2.0)
    q.enqueue("a", est_gpu_h=1.5)
    q.enqueue("b", est_gpu_h=1.5)
    assert q.claim("w").id == "a"
    assert q.claim("w") is None
    assert q.reserved("gpu_h") == pytest.approx(1.5)
    q.complete("a", "w")                 # 아무것도 장부에 안 썼다 = 예약만 풀림
    assert q.reserved("gpu_h") == 0.0
    assert q.claim("w").id == "b"


def test_held_subtracts_what_the_job_already_spent(make, ledger, clock):
    q = make(daily=3.0)
    q.enqueue("a", est_gpu_h=2.0)
    assert q.claim("w").id == "a"
    spend(ledger, "gpu_h", 1.5, "job:a", ts=clock(), key="a:1")  # 러너가 1.5h 계상
    assert q.reserved("gpu_h") == pytest.approx(0.5)             # max(0, 2.0 - 1.5)
    q.enqueue("too_big", est_gpu_h=1.5)   # 1.5(spent) + 0.5(held) + 1.5 = 3.5 > 3.0
    q.enqueue("fits", est_gpu_h=1.0)      # 3.0 <= 3.0
    assert q.claim("w2").id == "fits"
    assert q.claim("w2") is None


def test_held_never_negative_when_actual_exceeds_estimate(make, ledger, clock):
    q = make(daily=10.0)
    q.enqueue("a", est_gpu_h=1.0)
    q.claim("w")
    spend(ledger, "gpu_h", 4.0, "job:a", ts=clock(), key="a:1")
    assert q.reserved("gpu_h") == 0.0


def test_double_booking_after_complete_does_not_leak(make, ledger, clock):
    q = make(daily=2.0)
    q.enqueue("a", est_gpu_h=1.0)
    q.claim("w")
    spend(ledger, "gpu_h", 1.0, "job:a", ts=clock(), key="a:1")
    q.complete("a", "w")
    assert q.reserved("gpu_h") == 0.0
    q.enqueue("b", est_gpu_h=1.0)
    assert q.claim("w").id == "b"        # 1.0 + 1.0 <= 2.0
    q.enqueue("c", est_gpu_h=0.5)
    assert q.claim("w") is None          # 1.0(spent) + 1.0(held b) + 0.5


def test_usd_cap_via_usd_per_h(make):
    q = make(usd_daily=1.0, usd_total=100.0, usd_per_h=0.5)
    q.enqueue("a", est_gpu_h=1.5)        # 0.75 usd
    q.enqueue("b", est_gpu_h=1.5)        # 0.75 usd -> 합 1.5 > 1.0
    q.enqueue("c", est_gpu_h=0.5)        # 0.25 usd -> 합 1.0 <= 1.0
    assert q.claim("w").id == "a"
    assert q.claim("w").id == "c"
    assert q.claim("w") is None
    assert q.reserved("usd") == pytest.approx(1.0)


def test_usd_spent_elsewhere_blocks(make, ledger, clock):
    q = make(usd_daily=10.0, usd_total=100.0, usd_per_h=0.5)
    spend(ledger, "usd", 9.8, "vast:1", ts=clock(), key="v")   # 임대료
    q.enqueue("a", est_gpu_h=1.0)        # 0.5 usd -> 10.3 > 10.0
    q.enqueue("b", est_gpu_h=0.4)        # 0.2 usd -> 10.0 <= 10.0
    assert q.claim("w").id == "b"
    assert q.claim("w") is None


def test_usd_not_checked_when_usd_per_h_is_zero(make, ledger, clock):
    q = make(usd_daily=1.0, usd_total=1.0, usd_per_h=0.0)
    spend(ledger, "usd", 5.0, "vast:1", ts=clock(), key="v")   # 이미 USD 캡 초과
    q.enqueue("a", est_gpu_h=1.0)
    assert q.claim("w").id == "a"


def test_non_gpu_jobs_bypass_budget_and_slots(make, ledger, clock):
    q = make(daily=1.0, max_running=1)
    spend(ledger, "gpu_h", 5.0, "r", ts=clock(), key="s")      # GPU 예산 이미 초과
    q.enqueue("gpu1", est_gpu_h=0.1)                          # train: 순서상 먼저지만 예산 때문에 건너뜀
    q.enqueue("cpu1", gpu=False, cls="n")
    q.enqueue("cpu2", gpu=False, cls="n")
    ids = []
    while (j := q.claim("w")) is not None:
        ids.append(j.id)
    assert ids == ["cpu1", "cpu2"]
    assert q.get("gpu1").state == "queued"
    assert q.reserved("gpu_h") == 0.0


def test_uncapped_gpu_h_allows_everything(make):
    q = make(daily=None, total=None)
    for i in range(10):
        q.enqueue(f"j{i}", est_gpu_h=1000.0)
    assert len(_claim_all(q)) == 10


def test_enqueue_rejects_job_that_can_never_fit(make):
    q = make(daily=4.0, total=100.0)
    with pytest.raises(Q.QueueBudgetError) as ei:
        q.enqueue("huge", est_gpu_h=4.5)
    assert ei.value.reason == "daily" and ei.value.kind == "gpu_h"
    assert q.counts()["queued"] == 0
    with pytest.raises(Q.JobNotFound):
        q.get("huge")
    q.enqueue("edge", est_gpu_h=4.0)  # 정확히 캡: 허용
    q2 = make(daily=10.0, total=3.0)
    with pytest.raises(Q.QueueBudgetError) as ei2:
        q2.enqueue("huge", est_gpu_h=3.5)
    assert ei2.value.reason == "total"


def test_enqueue_rejects_usd_that_can_never_fit(make):
    q = make(usd_daily=1.0, usd_total=100.0, usd_per_h=1.0)
    with pytest.raises(Q.QueueBudgetError) as ei:
        q.enqueue("huge", est_gpu_h=1.5)
    assert ei.value.kind == "usd" and ei.value.reason == "daily"


def test_queue_never_writes_ledger(make, ledger):
    q = make(daily=5.0)
    n0 = ledger.count()
    for i in range(3):
        q.enqueue(f"j{i}", est_gpu_h=1.0)
    j = q.claim("w")
    q.heartbeat(j.id, "w")
    q.complete(j.id, "w")
    q.fail(q.claim("w").id, "w", "x")
    q.cancel("j2")
    assert ledger.count() == n0


def test_reserved_rejects_unknown_kind(make):
    with pytest.raises(ValueError):
        make().reserved("tfu")


def test_budget_is_checked_with_cancelled_and_failed_not_held(make):
    q = make(daily=2.0)
    q.enqueue("a", est_gpu_h=2.0)
    q.enqueue("b", est_gpu_h=2.0)
    q.claim("w")
    assert q.claim("w") is None
    q.cancel("a")
    assert q.reserved("gpu_h") == 0.0
    assert q.claim("w").id == "b"
    q.fail("b", "w", "boom", retry=False)
    assert q.get("b").state == "failed" and q.reserved("gpu_h") == 0.0
