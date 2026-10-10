"""RUN-03 공용 픽스처. 가짜 허브·결정론적 sleep 기록기."""
import pytest
from run03_fakes import TOKEN, FakeHub, blob  # noqa: F401

from buc.rt import hf_sync as H


@pytest.fixture
def hub():
    return FakeHub()


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def sync(hub, sleeps):
    return H.HubSync(hub, sleep=sleeps.append, secrets=(TOKEN,))


@pytest.fixture
def ckpt(tmp_path):
    """작은 로컬 체크포인트 파일 만들기: ckpt(name, data) -> Path."""
    def make(name="model.pt", data=None, seed=0):
        p = tmp_path / "src" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if data is not None else blob(1000, seed))
        return p
    return make
