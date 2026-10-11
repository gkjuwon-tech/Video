"""R2 횡색수차: 적·청 채널의 녹색 대비 방사 확대율 eps_c 와 모서리 이동(px). 정의: 채널 내용이 중심에서 (1+eps) 배 커져
보이면(바깥 이동 = eps·r) eps > 0."""
from functools import lru_cache

import numpy as np
import pytest

import rd03_helpers as H
from buc.read import screen_fx as fx

HH, WW = H.H0, H.W0
HD = H.half_diag(HH, WW)
TOL_PX = 0.25                      # 계약 목표 ±0.3 px 보다 약간 빡빡하게


@lru_cache(maxsize=None)
def ca_fit(shift_r, shift_b, seed=0, grain=0.0):
    x = H.rect_scene(seed=seed, soft=1.2)
    y = H.with_ca(x, shift_r / HD, shift_b / HD)
    if grain:
        y = H.full_chain(y, H.theta(grain=grain, lut="none"), seed=seed, noise=True)
    return fx.estimate_ca(y.astype(np.float32))


@pytest.mark.parametrize("sr,sb", [(0.7, -0.45), (-0.9, 0.7), (0.0, 0.0)])
def test_recovers_corner_shifts(sr, sb):
    f = ca_fit(sr, sb)
    assert abs(f.shift_corner_px[0] - sr) <= TOL_PX
    assert abs(f.shift_corner_px[1] - sb) <= TOL_PX
    assert abs(f.eps_r * f.r_corner - f.shift_corner_px[0]) < 1e-9
    assert abs(f.eps_b * f.r_corner - f.shift_corner_px[1]) < 1e-9
    assert abs(f.r_corner - HD) < 1e-6                                  # 기본 r_corner = 반대각선
    assert f.n_samples >= 200 and f.reliable is True
    assert np.isfinite(f.rms_resid_px) and f.rms_resid_px <= 0.5


def test_shift_model_is_linear_in_radius():
    f = ca_fit(0.7, -0.45)
    assert f.shift_px(0.0, "r") == 0.0 and f.shift_px(0.0, "b") == 0.0
    assert abs(f.shift_px(HD / 2, "r") - 0.5 * f.shift_corner_px[0]) < 1e-9
    assert abs(f.shift_px(HD, "b") - f.shift_corner_px[1]) < 1e-9
    assert np.allclose(np.asarray(f.shift_px(np.array([0.0, 50.0, 100.0]), "r")), np.array([0.0, 50.0, 100.0]) * f.eps_r)
    with pytest.raises(ValueError):
        f.shift_px(10.0, "g")


def test_robust_to_grain():
    f = ca_fit(0.7, -0.45, seed=1, grain=0.5)
    assert abs(f.shift_corner_px[0] - 0.7) <= 0.3
    assert abs(f.shift_corner_px[1] + 0.45) <= 0.3


def test_gray_input_has_no_ca_and_warns():
    g = np.repeat(H.rect_scene(seed=0, soft=1.2, colored=False)[..., :1], 3, axis=2).astype(np.float32)
    f = fx.estimate_ca(g)
    assert abs(f.shift_corner_px[0]) <= 0.05 and abs(f.shift_corner_px[1]) <= 0.05
    assert "ca_no_color" in f.warnings and f.reliable is False


def test_center_argument_for_cropped_input():
    cx, cy = 130.0, 90.0
    x = H.rect_scene(seed=2, soft=1.2)
    y = H.with_ca(x, 0.004, -0.003, center=(cx, cy))
    rc = float(np.hypot(max(cx, WW - 1 - cx), max(cy, HH - 1 - cy)))
    f = fx.estimate_ca(y.astype(np.float32), center=(cx, cy))
    assert abs(f.eps_r - 0.004) <= 0.0008 and abs(f.eps_b + 0.003) <= 0.0008
    assert abs(f.center[0] - cx) < 1e-9 and abs(f.center[1] - cy) < 1e-9
    assert f.r_corner > 0 and abs(f.r_corner - rc) < 1.0
