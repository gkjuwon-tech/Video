"""RUN-02 공용 픽스처: origin·가짜 원격·시계·러너 공장."""
import pytest
from run02_fakes import T0, Clock, FakeExecutor, FakeRemote, Origin  # noqa: F401

from buc.rt import runner as R


@pytest.fixture
def clock():
    return Clock(T0)


@pytest.fixture
def origin():
    return Origin()


@pytest.fixture
def make(origin, clock, tmp_path):
    """make(runner_id="r1", executor=None, work=None, **kw) -> (Runner, FakeRemote, FakeExecutor)."""
    def _make(runner_id="r1", executor=None, work=None, remote=None, **kw):
        rem = remote or FakeRemote(origin)
        ex = executor or FakeExecutor()
        kw.setdefault("environ", {"PATH": "/usr/bin"})
        r = R.Runner(rem, runner_id=runner_id, work_root=work or (tmp_path / ("work_" + runner_id)),
                     executor=ex, clock=clock, **kw)
        return r, rem, ex
    return _make
