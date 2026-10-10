import pytest
from run07_helpers import fill_ledger, load_module, make_items

from buc.core.config import Config
from buc.core.ledger import open_ledger


@pytest.fixture(scope="session")
def hr():
    return load_module()


@pytest.fixture
def ledger():
    led = open_ledger(":memory:")
    fill_ledger(led)
    yield led
    led.close()


@pytest.fixture
def items():
    return make_items()


@pytest.fixture
def cfg():
    return Config(budget_daily_usd=10.0, budget_total_usd=35.0)


@pytest.fixture
def data(hr, items, ledger, cfg):
    from run07_helpers import NOW
    return hr.collect(now=NOW, items=items, ledger=ledger, config=cfg)
