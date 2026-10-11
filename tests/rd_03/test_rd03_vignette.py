"""R2 비네팅: 정상 질감 장면에 LNS-04 비네팅을 입혀 이득 곡선·모서리 EV·θ.vignette 동치를 복원한다.

식별성 한계(계약서): 저주파 구조(경사·큰 물체)는 방사 이득과 구분되지 않는다. 그래서 시험 장면은 저주파가 없는 정상 질감
이고(`texture_scene`), 경사 장면은 오탐 상한만 본다."""
from functools import lru_cache

import numpy as np
import pytest

import rd03_helpers as H
from buc.lens import sensor_film as sf
from buc.read import screen_fx as fx

HH, WW = H.H0, H.W0
TOL_V = 0.08           # 모서리 이득 V(1) 와 곡선 오차(절대)


@lru_cache(maxsize=None)
def fit(seed, s):
    x = H.texture_scene(seed=seed)
    y = H.with_vignette(x, s) if s else x
    return fx.estimate_vignette(y.astype(np.float32))


@pytest.mark.parametrize("seed,s", [(0, 0.3), (1, 0.3), (0, 0.7), (1, 0.7)])
def test_recovers_lns04_vignette(seed, s):
    f = fit(seed, s)
    v1_true = 1.0 - 0.8 * s
    assert abs(float(f.gain_at(1.0)) - v1_true) <= TOL_V
    assert abs(f.strength - s) <= TOL_V / 0.8 + 1e-9          # θ.vignette 동치
    gt = sf.vignette_gain(HH, WW, s)
    assert np.abs(f.gain() - gt).max() <= TOL_V
    assert f.reliable is True
    assert f.shape == (HH, WW)
    cy, cx = (HH - 1) / 2.0, (WW - 1) / 2.0
    assert abs(f.center[0] - cx) < 1e-6 and abs(f.center[1] - cy) < 1e-6          # 기본 중심 = 프레임 중심
    assert abs(f.r_norm - H.half_diag(HH, WW)) < 1e-6                              # 기본 r_norm = 반대각선


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_vignette_stays_near_identity(seed):
    f = fit(seed, 0.0)
    assert f.strength <= 0.06
    assert abs(float(f.gain_at(1.0)) - 1.0) <= 0.06
    assert f.corner_ev >= -0.1 and f.corner_ev <= 0.0


def test_gain_curve_invariants():
    f = fit(0, 0.7)
    rho = np.linspace(0.0, 1.0, 101)
    g = np.asarray(f.gain_at(rho))
    assert g.shape == rho.shape
    assert g[0] == 1.0                                          # V(0) = 1 (중심 정규화)
    assert np.all(np.diff(g) <= 1e-12)                          # 비증가
    assert g.max() <= 1.0 and g.min() >= 0.05                   # 이득 [0.05, 1]
    assert np.asarray(f.gain_at(np.array([[0.2, 0.5], [0.9, 1.0]]))).shape == (2, 2)
    # corner_ev 는 V(1) 의 log2
    assert abs(f.corner_ev - np.log2(float(f.gain_at(1.0)))) < 1e-6
    # 화소 지도는 같은 곡선을 중심·r_norm 으로 놓은 것
    yy, xx = np.mgrid[0:HH, 0:WW]
    rr = np.hypot(xx - f.center[0], yy - f.center[1]) / f.r_norm
    ref = np.asarray(f.gain_at(rr))
    gm = f.gain()
    assert gm.dtype == np.float32 and gm.shape == (HH, WW)
    assert np.abs(gm - ref).max() <= 1e-5
    # 계수는 V(ρ)=1+a2ρ²+a4ρ⁴+a6ρ⁶ 의 (a2, a4, a6) 이고 a2 ≤ 0
    assert len(f.coeffs) == 3 and f.coeffs[0] <= 0.0
    assert abs(f.strength - min(1.0, max(0.0, (1.0 - float(f.gain_at(1.0))) / 0.8))) < 1e-9


def test_dividing_by_gain_flattens_corners():
    """ExplainAway 이음매: 영상 / gain() 후 모서리와 중앙의 평균 휘도비가 1 에 가깝다."""
    s = 0.7
    x = H.texture_scene(seed=1)
    y = H.with_vignette(x, s).astype(np.float32)
    f = fit(1, s)
    flat = y / f.gain()[..., None]
    lum = lambda a: a[..., 1].astype(np.float64)
    yy, xx = np.mgrid[0:HH, 0:WW]
    rr = np.hypot(xx - (WW - 1) / 2, yy - (HH - 1) / 2) / H.half_diag(HH, WW)
    outer, inner = rr > 0.8, rr < 0.25
    raw_ratio = lum(y)[outer].mean() / lum(y)[inner].mean()
    new_ratio = lum(flat)[outer].mean() / lum(flat)[inner].mean()
    assert raw_ratio < 0.65                              # 입력은 실제로 어둡다
    assert abs(new_ratio - 1.0) <= 0.1


@pytest.mark.parametrize("axis", [0, 1])
def test_illumination_ramp_is_not_a_strong_vignette(axis):
    """한 방향 밝기 경사(약 0.7 EV)를 강한 비네팅으로 오인하지 않는다(상한)."""
    x = H.ramp_scene(lo=0.3, hi=0.5, axis=axis, seed=0)
    f = fx.estimate_vignette(x.astype(np.float32))
    assert float(f.gain_at(1.0)) >= 0.8
    assert f.strength <= 0.25


def test_robust_to_grain_and_halo_in_full_chain():
    th = H.theta(grain=0.5, vignette=0.7, halation=0.15, lut="none")
    x = H.texture_scene(seed=2)
    y = H.full_chain(x, th, seed=3, noise=True)
    f = fx.estimate_vignette(y)
    assert abs(float(f.gain_at(1.0)) - (1.0 - 0.8 * 0.7)) <= 0.1


def test_cropped_input_with_center_and_r_norm():
    """크롭 영상: 광학 중심과 반경 척도를 크롭 좌표로 주면 크롭 지도를 맞춘다(RD-01 crop_box 이음매)."""
    Hf, Wf, s = 300, 480, 0.7
    x_full = H.texture_scene(Hf, Wf, seed=4)
    y_full = H.with_vignette(x_full, s)
    x0, y0, w, h = 120, 60, 300, 200
    crop = y_full[y0:y0 + h, x0:x0 + w].astype(np.float32)
    cx, cy = (Wf - 1) / 2.0 - x0, (Hf - 1) / 2.0 - y0
    rn = H.half_diag(Hf, Wf)
    f = fx.estimate_vignette(crop, center=(cx, cy), r_norm=rn)
    assert f.shape == (h, w)
    assert abs(f.center[0] - cx) < 1e-6 and abs(f.center[1] - cy) < 1e-6 and abs(f.r_norm - rn) < 1e-6
    gt = sf.vignette_gain(Hf, Wf, s)[y0:y0 + h, x0:x0 + w]
    assert np.abs(f.gain() - gt).max() <= 0.1


def test_small_image_flagged():
    x = H.texture_scene(48, 48, seed=0, lf_cut_px=8.0)
    f = fx.estimate_vignette(x.astype(np.float32))
    assert f.reliable is False
    r = fx.estimate_screen_fx(x.astype(np.float32))
    assert "small_image" in r.warnings
    assert r.reliable is False
