"""층별 max-abs 모니터: 100스텝마다(기본), 2e4 초과·비유한이면 경보, 훅은 출력을 바꾸지 않고 정리된다."""
import math

import pytest
import torch

from buc.train.amp import Alarm, Amp, AmpConfig, MaxAbsMonitor, MaxAbsReport


class Scale(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x * self.k


class Multi(torch.nn.Module):
    def forward(self, x):
        return (x * 2, {"a": x * 5, "idx": torch.tensor([10 ** 9])}, [x * 3])


def seq():
    return torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), Scale(1.0), Scale(1.0))


def test_defaults_and_leaf_names():
    m = seq()
    mon = MaxAbsMonitor(m)
    assert (mon.every, mon.threshold) == (100, 2e4)
    assert mon.begin(0) is True and mon.begin(99) is False and mon.begin(100) is True and mon.begin(1) is False
    mon.begin(0)
    m(torch.ones(1, 2))
    rep = mon.end()
    assert isinstance(rep, MaxAbsReport) and rep.step == 0
    assert set(rep.values) == {"0", "1", "2"}  # 잎 모듈, 루트("") 제외
    mon.close()


def test_values_are_exact_max_abs_and_repeat_calls_take_the_max():
    m = torch.nn.Sequential(Scale(3.0))
    mon = MaxAbsMonitor(m, every=1)
    mon.begin(5)
    m(torch.tensor([1.0, -2.0]))
    m(torch.tensor([0.5, -7.0]))  # 같은 스텝에서 두 번 호출: 최댓값
    rep = mon.end()
    assert rep.values == {"0": 21.0}
    assert rep.step == 5 and rep.alarms == ()
    assert rep.worst == ("0", 21.0)
    mon.close()


def test_unarmed_steps_record_nothing_and_do_not_leak():
    m = torch.nn.Sequential(Scale(2.0))
    mon = MaxAbsMonitor(m, every=10)
    assert mon.begin(3) is False
    m(torch.full((2,), 1e9))
    assert mon.end() is None
    assert mon.begin(10) is True
    m(torch.ones(2))
    assert mon.end().values == {"0": 2.0}  # 3스텝의 1e9가 새지 않는다
    assert mon.end() is None  # end 뒤에는 다시 무장 해제


def test_alarm_threshold_is_strictly_greater():
    m = torch.nn.Sequential(Scale(1.0))
    mon = MaxAbsMonitor(m, every=1, threshold=2.0)
    mon.begin(0)
    m(torch.tensor([2.0]))
    assert mon.end().alarms == ()
    mon.begin(1)
    m(torch.tensor([2.0001]))
    (a,) = mon.end().alarms
    assert isinstance(a, Alarm) and (a.step, a.layer, a.threshold) == (1, "0", 2.0)
    assert a.value == pytest.approx(2.0001, rel=1e-6)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_activations_raise_an_alarm(bad):
    m = torch.nn.Sequential(Scale(1.0), Scale(1.0))
    mon = MaxAbsMonitor(m, every=1)
    mon.begin(0)
    m(torch.tensor([1.0, bad]))
    rep = mon.end()
    assert {a.layer for a in rep.alarms} == {"0", "1"}
    assert all(not math.isfinite(a.value) for a in rep.alarms)
    assert rep.worst[0] in ("0", "1") and not math.isfinite(rep.worst[1])


def test_default_threshold_catches_fp16_range_blowup_under_autocast():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        m[0].weight.copy_(torch.tensor([[3e4, 0.0], [0.0, 1.0]]))
    amp = Amp(AmpConfig(dtype="fp16", device_type="cpu"))
    mon = MaxAbsMonitor(m, every=1)
    mon.begin(0)
    with amp.autocast():
        y = m(torch.tensor([[1.0, 1.0]]))
    assert y.dtype == torch.float16
    rep = mon.end()
    assert rep.values["0"] == pytest.approx(3e4, rel=1e-3)
    assert [a.layer for a in rep.alarms] == ["0"]


def test_nested_outputs_and_non_float_tensors():
    m = torch.nn.Sequential(Multi(), )
    mon = MaxAbsMonitor(m, every=1)
    mon.begin(0)
    m(torch.tensor([2.0]))
    assert mon.end().values == {"0": 10.0}  # 정수 텐서(1e9)는 무시, 튜플·딕트·리스트 재귀
    mon.close()


def test_include_filter():
    m = seq()
    mon = MaxAbsMonitor(m, every=1, include=lambda name, mod: isinstance(mod, Scale))
    mon.begin(0)
    m(torch.ones(1, 2))
    assert set(mon.end().values) == {"1", "2"}
    mon.close()


def test_monitor_is_transparent_and_close_removes_hooks():
    torch.manual_seed(0)
    m1, m2 = seq(), seq()
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(3, 2)
    y1 = m1(x).sum()
    y1.backward()
    with MaxAbsMonitor(m2, every=1) as mon:
        mon.begin(0)
        y2 = m2(x).sum()
        y2.backward()
        mon.end()
    assert torch.equal(y1, y2)
    assert torch.equal(m1[0].weight.grad, m2[0].weight.grad)
    assert all(len(mod._forward_hooks) == 0 for mod in m2.modules())  # 컨텍스트 종료 = close
    mon.close()  # 두 번 불러도 안전
