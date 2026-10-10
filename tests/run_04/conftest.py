"""RUN-04 공용 픽스처: 주입 시계, 인메모리 장부, 큐 공장."""
import pytest

from buc.core.config import Config
from buc.core.ledger import LedgerEvent, open_ledger
from buc.rt import queue as Q

H = 3600.0
DAY0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z (UTC 자정)


class Clock:
    def __init__(self, t=DAY0 + 10 * H):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def spend(ledger, kind, amount, run_id, ts, key=None):
    """러너가 하는 일을 흉내: 장부에 지출을 기록한다."""
    return ledger.record(LedgerEvent(kind=kind, amount=amount, run_id=run_id, ts=ts, idem_key=key))


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(clock):
    led = open_ledger(":memory:", clock=clock)
    yield led
    led.close()


@pytest.fixture
def make(ledger, clock):
    """make(daily=None, total=None, usd_daily=..., ...) -> JobQueue. gpu_h 캡 기본은 무제한."""
    made = []

    def _make(daily=None, total=None, usd_daily=1000.0, usd_total=1000.0, max_running=100,
              usd_per_h=0.0, path=":memory:", led=None):
        cfg = Config(budget_daily_gpu_h=daily, budget_total_gpu_h=total,
                     budget_daily_usd=usd_daily, budget_total_usd=usd_total)
        q = Q.open_queue(path, ledger=led or ledger, config=cfg, clock=clock,
                         usd_per_h=usd_per_h, max_running=max_running)
        made.append(q)
        return q

    yield _make
    for q in made:
        q.close()
