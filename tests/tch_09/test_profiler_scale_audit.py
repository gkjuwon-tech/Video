"""MaxAbsProfiler, choose_scale, upper95, audit_images."""
import math

import pytest
import torch
from scipy import stats
from torch import nn

from buc.teacher import t4_fp16 as R


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()


def test_profiler_exact_running_max_and_hook_cleanup():
    m = _net()
    x1 = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64)
    x2 = torch.tensor([[-7.0, 0.5, 0.0]], dtype=torch.float64)
    with R.MaxAbsProfiler(m) as p, torch.no_grad():
        y1 = m(x1)
        m(x2)
    assert set(p.out_max) == {"0", "1", "2"}  # 루트 제외 잎 모듈
    with torch.no_grad():
        h1, h2 = m[0](x1), m[0](x2)
        o1, o2 = m(x1), m(x2)
    assert p.out_max["0"] == pytest.approx(max(h1.abs().max().item(), h2.abs().max().item()), rel=1e-12)
    assert p.in_max["0"] == 7.0
    assert p.out_max["2"] == pytest.approx(max(o1.abs().max().item(), o2.abs().max().item()), rel=1e-12)
    assert p.calls == {"0": 2, "1": 2, "2": 2}
    assert p.nonfinite == {"0": 0, "1": 0, "2": 0}
    for mod in m.modules():
        assert len(mod._forward_hooks) == 0 and len(mod._forward_pre_hooks) == 0
    with torch.no_grad():
        assert torch.equal(m(x1), y1)  # 출력 불변


def test_profiler_named_and_nonfinite():
    m = _net()
    x = torch.tensor([[float("inf"), 1.0, 2.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    with R.MaxAbsProfiler(m, names=["0"]) as p, torch.no_grad():
        m(x)
        m(x[1:])
    assert set(p.out_max) == {"0"}
    assert p.nonfinite["0"] == 1 and p.calls["0"] == 2
    assert math.isfinite(p.out_max["0"]) and p.in_max["0"] == 2.0  # 비유한 값은 최댓값에서 제외
    with pytest.raises(R.RescuePlanError), R.MaxAbsProfiler(m, names=["nope"]):
        pass


@pytest.mark.parametrize("max_abs,headroom,expected", [
    (0.0, 8.0, 1.0),
    (1000.0, 8.0, 1.0),
    (8188.0, 8.0, 1.0),
    (8189.0, 8.0, 2.0),
    (1e5, 8.0, 16.0),
    (1e5, 1.0, 2.0),
    (2.5e6, 8.0, 512.0),
])
def test_choose_scale_values(max_abs, headroom, expected):
    s = R.choose_scale(max_abs, headroom=headroom)
    assert s == expected
    assert max_abs / s <= R.FP16_MAX / headroom
    if s > 1:
        assert max_abs / (s / 2) > R.FP16_MAX / headroom  # 가장 작은 거듭제곱


def test_choose_scale_mapping_and_errors():
    assert R.choose_scale({"a": 10.0, "b": 1e5, "c": 3.0}) == 16.0
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(R.RescueError):
            R.choose_scale(bad)
    with pytest.raises(R.RescueError):
        R.choose_scale(1e12, max_s=2.0 ** 16)
    assert R.choose_scale(1e12, max_s=2.0 ** 30) == 2.0 ** 27


def test_upper95():
    assert R.upper95(0, 1000) == pytest.approx(1 - 0.05 ** (1 / 1000), rel=1e-12)
    assert R.upper95(0, 1000) == pytest.approx(0.0029912, abs=1e-6)  # PARADIGM: 0/1,000의 95% 상한 0.3%
    assert R.upper95(0, 10000) == pytest.approx(0.00029953, abs=1e-7)
    assert R.upper95(7, 7) == 1.0
    for k, n in [(1, 50), (5, 100), (30, 2000)]:
        u = R.upper95(k, n)
        assert k / n < u < 1
        assert stats.binom.cdf(k, n, u) == pytest.approx(0.05, abs=1e-6)
    for k, n in [(0, 0), (-1, 5), (6, 5)]:
        with pytest.raises(ValueError):
            R.upper95(k, n)


def test_audit_images():
    ok = torch.rand(4, 3, 8, 8) * 2 - 1
    black = torch.full((2, 3, 8, 8), -1.0)
    black[0, 0, 0, 0] = -0.97  # (x+1)/2 = 0.015 <= 0.02 -> 여전히 검다
    nan = torch.zeros(3, 8, 8)
    nan[1, 2, 3] = float("nan")
    inf_black = torch.full((3, 8, 8), -1.0)
    inf_black[0, 0, 0] = float("inf")  # 비유한이 우선(배타)
    rep = R.audit_images([ok, black, nan, inf_black])
    assert rep.n_images == 4 + 2 + 1 + 1
    assert rep.n_nonfinite == 2 and rep.n_black == 2
    assert rep.ok is False
    assert rep.upper95_fail == pytest.approx(R.upper95(4, 8), rel=1e-12)
    good = R.audit_images([ok, ok[0]])
    assert (good.n_images, good.n_nonfinite, good.n_black, good.ok) == (5, 0, 0, True)
    assert good.upper95_fail == pytest.approx(1 - 0.05 ** (1 / 5), rel=1e-12)
    assert R.audit_images([]).ok is False
    assert R.audit_images([black], black_thresh=0.01).n_black == 1  # 0.015 > 0.01 이면 첫 장은 검지 않다
