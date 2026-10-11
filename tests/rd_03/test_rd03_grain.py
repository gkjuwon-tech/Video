"""R2 그레인: 평탄 영역 잡음에서 θ.grain 동치 세기, 상관 반경, 신호 의존 σ(u), 신호 무관 성분(비물리 플래그), PSD."""
from functools import lru_cache

import numpy as np
import pytest

import rd03_helpers as H
from buc.core import color
from buc.read import screen_fx as fx


@lru_cache(maxsize=None)
def grain_fit(s, r, seed):
    x = H.flat_scene(seed=seed)
    y = H.with_grain(x, s, r, seed=seed) if s else x
    return fx.estimate_grain(H.quantize8(y).astype(np.float32))


def sigma_true(s, u):
    return s * 2.0 * H.GRAIN_SIGMA_MAX * np.sqrt(u * (1.0 - u))


@pytest.mark.parametrize("s,r,seed", [(0.25, 1.5, 0), (0.25, 2.5, 1), (0.5, 1.5, 2), (0.5, 2.5, 0), (1.0, 1.5, 1), (1.0, 2.5, 2)])
def test_strength_and_radius(s, r, seed):
    g = grain_fit(s, r, seed)
    assert abs(g.strength - s) <= max(0.04, 0.2 * s)
    assert abs(g.radius_px - r) <= 0.25 * r
    assert g.reliable is True
    assert g.n_flat_blocks >= 24
    assert g.nonphysical is False and g.floor_frac <= 0.3             # 순수 필름 그레인은 신호 무관 성분이 작다
    assert abs(g.sigma_coef - s * 2.0 * H.GRAIN_SIGMA_MAX) <= max(0.0064, 0.2 * s * 0.16)
    assert abs(g.channel_corr) <= 0.2                                    # LNS-04 그레인은 채널 독립


@pytest.mark.parametrize("seed", [0, 1])
def test_grain_free_input_reads_near_zero(seed):
    g = grain_fit(0.0, 1.5, seed)
    assert g.strength <= 0.08
    assert 0.0 <= g.strength <= 1.0 and np.isfinite(g.radius_px) and g.radius_px > 0


def test_signal_dependent_sigma_curve():
    s, r = 0.6, 2.0
    g = grain_fit(s, r, 3)
    for u in (0.3, 0.5, 0.7):
        got = float(g.sigma_at(u))
        assert abs(got / sigma_true(s, u) - 1.0) <= 0.25, u
    arr = np.asarray(g.sigma_at(np.linspace(0.0, 1.0, 11)))
    assert arr.shape == (11,) and np.all(np.isfinite(arr)) and np.all(arr >= 0)
    assert arr[5] > arr[1] and arr[5] > arr[9]            # 중간톤에서 최대(σ ∝ √(u(1-u)) + 바닥)
    assert len(g.levels) == len(g.sigma_by_level) >= 3
    assert all(0.0 < u < 1.0 for u in g.levels) and all(np.isfinite(v) and v >= 0 for v in g.sigma_by_level)
    assert abs(g.strength - s) <= max(0.04, 0.2 * s)
    assert abs(g.sigma_coef - g.strength * 2.0 * H.GRAIN_SIGMA_MAX) < 1e-9     # c = 0.16 * strength


def test_signal_independent_noise_is_flagged_nonphysical():
    rng = np.random.default_rng(5)
    x = H.flat_scene(seed=1)
    d = color.linear_to_display(x.astype(np.float32)).astype(np.float64)
    d = np.clip(d + rng.normal(0.0, 0.02, d.shape), 0.0, 1.0)             # 레벨과 무관한 백색 가산 잡음
    y = H.quantize8(color.srgb_to_linear(d.astype(np.float32)))
    g = fx.estimate_grain(y.astype(np.float32))
    assert g.nonphysical is True
    assert 0.01 <= g.floor_sigma <= 0.03
    assert g.floor_frac > 0.5
    assert g.radius_px <= 1.2                                              # 백색: 상관 거의 없음


def test_luma_correlated_noise_has_high_channel_corr():
    rng = np.random.default_rng(9)
    from scipy import ndimage as ndi
    x = H.flat_scene(seed=2)
    d = color.linear_to_display(x.astype(np.float32)).astype(np.float64)
    n = ndi.gaussian_filter(rng.normal(size=d.shape[:2]), 0.75)
    n = 0.03 * n / n.std()
    y = H.quantize8(color.srgb_to_linear(np.clip(d + n[..., None], 0, 1).astype(np.float32)))
    g = fx.estimate_grain(y.astype(np.float32))
    assert g.channel_corr >= 0.8


def test_psd_diagnostics_and_model():
    g = grain_fit(0.6, 2.5, 0)
    nu = np.asarray(g.psd_nu)
    p = np.asarray(g.psd)
    assert nu.shape == p.shape and len(nu) >= 8
    assert nu[0] == 0.0 and nu[-1] <= 0.5 + 1e-9 and np.all(np.diff(nu) > 0)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    lo, hi = p[nu <= 0.1].mean(), p[nu >= 0.4].mean()
    assert lo >= 3.0 * hi                                      # 상관 그레인: 저주파 > 고주파
    # 해석 모델: 2D 가우시안 스펙트럼, 평면 적분 = σ(u)^2
    u = 0.5
    grid = np.linspace(0.0, 3.0, 30001)
    S = np.asarray(g.psd_model(grid, u))
    integral = np.trapezoid(S * 2.0 * np.pi * grid, grid) if hasattr(np, "trapezoid") else np.trapz(S * 2.0 * np.pi * grid, grid)
    assert abs(integral / float(g.sigma_at(u)) ** 2 - 1.0) <= 0.02
    rr = g.radius_px
    assert abs(float(np.asarray(g.psd_model(np.array([0.0]), u))[0]) - float(g.sigma_at(u)) ** 2 * np.pi * rr * rr) \
        <= 1e-9 * max(1.0, np.pi * rr * rr)


def test_jpeg_quality_downgrades_reliability():
    x = H.flat_scene(seed=0)
    img = H.quantize8(H.with_grain(x, 0.5, 2.0, seed=0)).astype(np.float32)
    lossy = fx.estimate_grain(img, jpeg_quality=60)
    clean = fx.estimate_grain(img, jpeg_quality=95)
    none = fx.estimate_grain(img)
    assert "jpeg_lossy" in lossy.warnings and lossy.reliable is False
    assert "jpeg_lossy" not in clean.warnings and clean.reliable is True
    assert none.strength == clean.strength and "jpeg_lossy" not in none.warnings   # 품질 미지정은 손실 가정 없음
    assert lossy.strength == clean.strength                           # 경고만 하고 수치는 바꾸지 않는다


def test_texture_only_input_is_finite():
    """평탄 영역이 없는 입력도 예외 없이 유한한 값(상한)을 낸다."""
    g = fx.estimate_grain(H.texture_scene(seed=3).astype(np.float32))
    assert 0.0 <= g.strength <= 1.0 and np.isfinite(g.radius_px) and np.isfinite(g.floor_sigma)
    assert np.isfinite(g.channel_corr)
