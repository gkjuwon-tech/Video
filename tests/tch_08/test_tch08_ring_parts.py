"""Reference Ring 부품: 띠 합성, 캔버스, 띠 훅, 재노이즈 수식·통계, RingRConfig."""
import math

import numpy as np
import pytest
import torch

from buc.teacher import ref_slot as R
from buc.teacher.sample import SigmaError, StepInfo
from buc.teacher.zimage import make_noise


def _const(rgb, h, w):
    return torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).expand(3, h, w).clone()


def test_compose_strip_slots_constant_swatches():
    cols = [(0.1, -0.2, 0.3), (0.9, 0.0, -0.9), (-0.5, 0.5, 0.25)]
    sw = [_const(c, 40 + 7 * i, 30 + 11 * i) for i, c in enumerate(cols)]
    s = R.compose_strip(sw, strip_px=128, width=1024)
    assert tuple(s.shape) == (3, 128, 1024) and s.dtype == torch.float32
    bounds = [(j * 1024 // 3, (j + 1) * 1024 // 3) for j in range(3)]
    for (lo, hi), c in zip(bounds, cols):
        want = torch.tensor(c).view(3, 1, 1)
        assert torch.allclose(s[:, :, lo:hi], want.expand(3, 128, hi - lo), atol=1e-6)


def test_compose_strip_exact_size_is_bit_copy():
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(3, 32, 24, generator=g), torch.rand(3, 32, 24, generator=g)
    s = R.compose_strip([a, b], strip_px=32, width=48)
    assert torch.equal(s[:, :, :24], a) and torch.equal(s[:, :, 24:], b)


def test_compose_strip_bilinear_definition():
    g = torch.Generator().manual_seed(1)
    a = torch.rand(3, 20, 50, generator=g)
    s = R.compose_strip([a], strip_px=16, width=64)
    ref = torch.nn.functional.interpolate(a[None], size=(16, 64), mode="bilinear", align_corners=False,
                                          antialias=False)[0]
    assert torch.allclose(s, ref, atol=1e-6)


@pytest.mark.parametrize("bad", [[], [torch.zeros(3, 8, 8)] * 5, [torch.zeros(4, 8, 8)],
                                 [torch.full((3, 8, 8), float("nan"))]])
def test_compose_strip_rejects(bad):
    with pytest.raises(ValueError):
        R.compose_strip(bad, strip_px=16, width=64)


def test_compose_strip_rejects_bad_geometry():
    with pytest.raises(ValueError):
        R.compose_strip([torch.zeros(3, 8, 8)], strip_px=20, width=64)
    with pytest.raises(ValueError):
        R.compose_strip([torch.zeros(3, 8, 8)], strip_px=16, width=70)


def test_ring_canvas_and_split():
    view = torch.randn(2, 3, 64, 48)
    strip = torch.randn(3, 16, 48)
    c = R.ring_canvas(view, strip)
    assert tuple(c.shape) == (2, 3, 80, 48)
    st, body = R.split_canvas(c, 16)
    assert torch.equal(body, view) and torch.equal(st, strip.expand(2, 3, 16, 48))
    c1 = R.ring_canvas(view[0], strip)
    assert tuple(c1.shape) == (1, 3, 80, 48) and torch.equal(c1[0], c[0])
    with pytest.raises(ValueError):
        R.ring_canvas(view, torch.randn(3, 16, 40))
    # 기본 기하: 1024^2 뷰 + 128 px 띠 = 1152 x 1024 캔버스
    big = R.ring_canvas(torch.zeros(1, 3, 1024, 1024), torch.zeros(3, R.RING_STRIP_PX, 1024))
    assert tuple(big.shape[-2:]) == tuple(R.RING_CANVAS_HW)


def test_strip_hook_overwrites_only_strip_and_keeps_input():
    x0 = torch.randn(3, 4, 6, 5)
    ref = torch.randn(4, 2, 5)
    hook = R.strip_hook(ref, 2)
    keep = x0.clone()
    info = StepInfo(step=0, sigma=0.5, sigma_next=0.3, view_lo=0, view_hi=3)
    y = hook(x0, info)
    assert torch.equal(x0, keep)
    assert torch.equal(y[:, :, :2], ref.expand(3, 4, 2, 5))
    assert torch.equal(y[:, :, 2:], x0[:, :, 2:])
    refb = torch.randn(3, 4, 2, 5)
    yb = R.strip_hook(refb, 2)(x0, info)
    assert torch.equal(yb[:, :, :2], refb)
    with pytest.raises(ValueError):
        R.strip_hook(torch.randn(4, 3, 5), 2)(x0, info)


def test_renoise_up_exact_coefficients():
    z = torch.randn(2, 4, 3, 3)
    n = torch.randn(2, 4, 3, 3)
    out = R.renoise_up(z, 0.5, 0.75, n)
    a, b = 0.25 / 0.5, math.sqrt(0.75 ** 2 - (0.5 * 0.5) ** 2)
    assert out.dtype == torch.float32
    assert torch.allclose(out, a * z + b * n, atol=1e-6)
    top = R.renoise_up(z, 0.3, 1.0, n)
    assert torch.allclose(top, n, atol=1e-6)


@pytest.mark.parametrize("sf,st", [(0.5, 0.5), (0.75, 0.5), (0.0, 0.5), (1.0, 1.0), (0.5, 1.2), (float("nan"), 0.9)])
def test_renoise_up_rejects(sf, st):
    z = torch.zeros(1, 1, 2, 2)
    with pytest.raises(SigmaError):
        R.renoise_up(z, sf, st, z)


def test_renoise_up_preserves_flow_marginal():
    """z=(1-sf)x+sf*e 이면 결과 - (1-st)x 의 표준편차가 st, 원래 노이즈와의 상관이 a*sf/st."""
    n = 200_000
    sf, st = 5 / 6, 0.9
    x = torch.linspace(-1, 1, n).view(1, 1, 1, n)
    e1 = make_noise(1, (1, 1, 1, n))
    e2 = make_noise(2, (1, 1, 1, n))
    z = (1 - sf) * x + sf * e1
    out = R.renoise_up(z, sf, st, e2)
    res = (out - (1 - st) * x).flatten().double().numpy()
    assert abs(res.mean()) < 0.01
    assert res.std() == pytest.approx(st, rel=0.01)
    a = (1 - st) / (1 - sf)
    corr = np.corrcoef(res, e1.flatten().double().numpy())[0, 1]
    assert corr == pytest.approx(a * sf / st, abs=0.01)


def test_ring_r_config():
    c = R.RingRConfig()
    assert tuple(c.resample_at) == (3,) and c.r == 2
    assert c.extra_nfe(0) == 2 and c.extra_nfe(2) == 2 and c.extra_nfe(3) == 0
    c2 = R.RingRConfig(resample_at=(2, 3, 4), r=2)
    assert c2.extra_nfe(0) == 6 and c2.extra_nfe(3) == 2
    assert R.RingRConfig(r=0).extra_nfe(0) == 0
    for bad in (dict(resample_at=(0,)), dict(resample_at=(3, 3)), dict(r=-1), dict(r=65)):
        with pytest.raises(ValueError):
            R.RingRConfig(**bad)
