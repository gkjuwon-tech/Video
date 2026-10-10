import pytest

import tch03_fakes as F
from buc.teacher import taps as T


@pytest.fixture(scope="module")
def basis():
    return F.make_basis()


@pytest.fixture()
def model():
    return F.FakeDiT().eval()


@pytest.fixture()
def bus(basis, model):
    b = T.TapBus(basis).attach(model.layers)
    yield b
    b.detach()
