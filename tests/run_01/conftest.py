"""RUN-01 공용 픽스처: 주입 시계, 가짜 Vast, 인메모리 장부, 오케스트레이터 공장."""
import pytest
from run01_fakes import DAY0, H, Clock, FakeVast, raw_offer  # noqa: F401

from buc.core.config import Config
from buc.core.ledger import open_ledger
from buc.rt import vast as V


@pytest.fixture
def clock():
    return Clock(DAY0 + 10 * H)


@pytest.fixture
def api(clock):
    return FakeVast([raw_offer(1, dph=0.40), raw_offer(2, dph=0.45)], clock=clock)


@pytest.fixture
def ledger(clock):
    led = open_ledger(":memory:", clock=clock)
    yield led
    led.close()


@pytest.fixture
def make(api, ledger, clock):
    """make(daily=100, total=1000, **policy) -> VastOrchestrator. 기본은 유휴·부팅 기한이 사실상 꺼진 상태."""
    def _make(daily=100.0, total=1000.0, config=None, activity=None, **pol):
        cfg = config or Config(budget_daily_usd=daily, budget_total_usd=total)
        kw = dict(idle_timeout_s=1e9, boot_timeout_s=1e9)
        kw.update(pol)
        return V.VastOrchestrator(api, ledger, cfg, policy=V.VastPolicy(**kw), clock=clock, activity=activity)
    return _make
