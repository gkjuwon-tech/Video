"""plumb-line 왜곡 피팅: 방사 k1·k2 복원(≤0.5 px@1080p), 직선은 0, 에셔 '휜 직선'은 렌즈로 흡수하지 않는다."""
import numpy as np
import pytest

import rd02_helpers as H
from buc.read import geom_cam as gc

W, HH = 1920, 1080
K = H.K_from_fov(70, W, HH)
R = H.camera_R(25, 5, 2)


@pytest.mark.parametrize("k1,k2", [(-0.1, 0.02), (0.06, 0.0), (-0.18, 0.05)])
def test_recovers_radial_distortion(k1, k2):
    polys = H.distorted_polylines(K, R, W, HH, k1, k2, noise_px=0.2, seed=1)
    fit = gc.fit_plumb_line(polys, K)
    assert isinstance(fit, gc.PlumbFit)
    assert fit.n_lines >= 20
    err = H.max_displacement_px(K, W, HH, fit.k1, fit.k2, k1, k2) * 1080.0 / HH
    assert err <= 0.5
    assert fit.rms_px <= 0.4  # 잡음 0.2 px 수준


def test_straight_lines_give_no_distortion():
    polys = H.distorted_polylines(K, R, W, HH, 0.0, 0.0, noise_px=0.2, seed=2)
    fit = gc.fit_plumb_line(polys, K)
    assert H.max_displacement_px(K, W, HH, fit.k1, fit.k2, 0.0, 0.0) <= 0.5


def test_escher_wiggle_is_not_absorbed_into_lens():
    polys = H.distorted_polylines(K, R, W, HH, 0.0, 0.0, noise_px=0.2, seed=3, wiggle_px=10.0)
    opts = gc.GeomCamOpts()
    fit = gc.fit_plumb_line(polys, K, opts=opts)
    assert abs(fit.k1) <= opts.k_bound + 1e-12 and abs(fit.k2) <= opts.k_bound + 1e-12
    assert fit.rms_px >= 2.0  # 설명되지 않은 곡률은 잔차로 보고된다


def test_too_few_lines_returns_zero_and_nan():
    polys = H.distorted_polylines(K, R, W, HH, -0.1, 0.0, n_lines=2, seed=4)
    fit = gc.fit_plumb_line(polys, K)
    assert fit.k1 == 0.0 and fit.k2 == 0.0 and np.isnan(fit.rms_px)
    assert fit.n_lines < gc.GeomCamOpts().min_plumb_lines


def test_k_bound_is_respected():
    polys = H.distorted_polylines(K, R, W, HH, -0.1, 0.02, noise_px=0.2, seed=5)
    opts = gc.GeomCamOpts(k_bound=0.05)
    fit = gc.fit_plumb_line(polys, K, opts=opts)
    assert abs(fit.k1) <= 0.05 + 1e-12 and abs(fit.k2) <= 0.05 + 1e-12
