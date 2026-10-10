"""난수 시뮬레이션: 러너를 흉내 내며 어느 시점에도 `장부 합 + 예약 <= 캡`이 큐 때문에 깨지지 않음을 확인."""
import math
import random

import pytest
from conftest import DAY0, H, Clock, spend

from buc.core.config import Config
from buc.core.ledger import open_ledger, utc_day_bounds
from buc.rt import queue as Q

TOL = 1e-9


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_invariants_under_random_operations(seed):
    rng = random.Random(seed)
    clock = Clock(DAY0 + 6 * H)
    daily, total = 6.0, 14.0
    cfg = Config(budget_daily_gpu_h=daily, budget_total_gpu_h=total, budget_daily_usd=1000.0,
                 budget_total_usd=1000.0)
    ledger = open_ledger(":memory:", clock=clock)
    q = Q.open_queue(":memory:", ledger=ledger, config=cfg, clock=clock, max_running=2)
    running = {}   # id -> (worker, est)
    done_ids = set()
    n = 0
    for step in range(400):
        clock.advance(rng.choice([60.0, 600.0, 1800.0, 7200.0]))
        for rid in q.reap():          # 리스(3h)가 만료된 잡 정리
            running.pop(rid, None)
        op = rng.random()
        if op < 0.30:
            n += 1
            try:
                q.enqueue(f"j{n}", cls=rng.choice(Q.CLASSES), priority=rng.randint(-5, 5),
                          est_gpu_h=rng.choice([0.25, 0.5, 1.0, 2.0, 3.5]))
            except Q.QueueBudgetError:
                pass
        elif op < 0.65:
            j = q.claim(f"w{step}", lease_s=3 * H)
            if j is not None:
                assert j.id not in running and j.id not in done_ids
                running[j.id] = (j.worker_id, j.est_gpu_h)
        elif running:
            jid = rng.choice(sorted(running))
            worker, est = running[jid]
            actual = est * rng.choice([0.0, 0.5, 0.8, 1.0])   # 실제 지출은 추정을 넘지 않는다고 가정
            if actual:
                spend(ledger, "gpu_h", actual, f"job:{jid}", ts=clock(), key=f"{jid}:{step}")
            if rng.random() < 0.8:
                q.complete(jid, worker)
                done_ids.add(jid)
            else:
                q.fail(jid, worker, "boom", retry=False)
                done_ids.add(jid)
            del running[jid]
        # 불변식
        now = clock()
        start, end = utc_day_bounds(now)
        spent_day = ledger.total("gpu_h", since=start, until=end)
        spent_all = ledger.total("gpu_h")
        held = q.reserved("gpu_h")
        assert held >= 0.0 and math.isfinite(held)
        assert held == pytest.approx(
            math.fsum(max(0.0, e - ledger.total("gpu_h", run_id=f"job:{i}")) for i, (_, e) in running.items()),
            abs=1e-9)
        assert q.counts()["running"] == len(running) <= 2
        assert spent_day + held <= daily + TOL, (step, spent_day, held)
        assert spent_all + held <= total + TOL, (step, spent_all, held)
    assert n > 50 and q.counts()["done"] > 5   # 시뮬레이션이 실제로 일을 했다
    ledger.close()
    q.close()


def test_simulation_is_deterministic_given_seed():
    def run():
        rng = random.Random(99)
        clock = Clock(DAY0)
        led = open_ledger(":memory:", clock=clock)
        q = Q.open_queue(":memory:", ledger=led, config=Config(budget_daily_gpu_h=3.0), clock=clock, max_running=3)
        order = []
        for i in range(40):
            q.enqueue(f"j{i}", cls=rng.choice(Q.CLASSES), priority=rng.randint(-3, 3),
                      est_gpu_h=rng.choice([0.5, 1.0]))
        for _ in range(40):
            j = q.claim("w")
            if j is None:
                clock.advance(86400)
                continue
            order.append(j.id)
            q.complete(j.id, "w")
        q.close()
        led.close()
        return order

    a, b = run(), run()
    assert a == b and len(a) == 40
