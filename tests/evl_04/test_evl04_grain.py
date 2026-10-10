"""EVL-04 ΔG: 대역별 그레인 PSD 로그비(log2)의 평균."""
import numpy as np
import pytest

from evl04_support import ff, noisy, smooth_noise, white_noise_gray


def test_identity_is_exactly_zero_and_shapes():
    m = ff()
    a = white_noise_gray(128, 128, 0.5, 0.05, seed=1)
    assert m.delta_g(a, a) == 0.0
    r = m.grain_band_ratios(a, a, n_bands=6)
    assert r.shape == (6,) and np.all(r == 0.0)
    assert m.grain_band_ratios(a, a, n_bands=4).shape == (4,)


def test_white_noise_amplitude_ratio_gives_two_log2_units_per_band():
    m = ff()
    a = white_noise_gray(256, 256, 0.5, 0.04, seed=2)
    b = white_noise_gray(256, 256, 0.5, 0.08, seed=3)  # 진폭 2배 => 파워 4배 => log2 = 2
    r = m.grain_band_ratios(a, b, n_bands=6)
    assert r.shape == (6,) and np.all(np.abs(r - 2.0) < 0.4), r
    assert m.delta_g(a, b) == pytest.approx(2.0, abs=0.15)
    assert m.delta_g(b, a) == pytest.approx(2.0, abs=0.15)  # ΔG 는 대칭(절댓값 평균)
    rev = m.grain_band_ratios(b, a, n_bands=6)
    assert np.all(rev < -1.5)  # 대역별 비는 부호가 있다: log2(P_F0 / P_입력)


def test_independent_realizations_of_same_grain_are_near_zero():
    m = ff()
    a = white_noise_gray(256, 256, 0.5, 0.05, seed=4)
    b = white_noise_gray(256, 256, 0.5, 0.05, seed=5)
    assert m.delta_g(a, b) < 0.15


def test_extra_high_frequency_grain_shows_only_in_high_bands():
    m = ff()
    base = np.clip(0.5 + 0.12 * smooth_noise(256, 256, 3.0, seed=6), 0, 1)
    g = lambda s, seed: np.repeat(np.clip(base + s * np.random.default_rng(seed).standard_normal(base.shape), 0, 1)[..., None], 3, 2).astype(np.float32)
    a, b = g(0.01, 7), g(0.05, 8)
    r = m.grain_band_ratios(a, b, n_bands=6)
    assert np.all(np.abs(r[:2]) < 0.35), r  # 저역은 매끈한 질감이 지배
    assert np.all(r[-2:] > 1.5), r  # 최고역은 그레인이 지배: 진폭 5배 => log2(25) = 4.6
    assert r[-1] > r[0] + 1.5


def test_band_layout_constants():
    m = ff()
    assert m.GRAIN_F_LO == pytest.approx(1 / 16) and m.GRAIN_F_HI == 0.5  # 사이클/픽셀, 로그 간격 대역


def test_validation_and_purity():
    m = ff()
    a = white_noise_gray(64, 64, 0.5, 0.05, seed=9)
    keep = a.copy()
    m.delta_g(a, noisy(a, 0.01, seed=10))
    assert np.array_equal(a, keep)
    with pytest.raises(ValueError):
        m.delta_g(a, a[:, :-1])
    with pytest.raises(ValueError):
        m.delta_g(a, a, n_bands=0)
    with pytest.raises(ValueError):
        m.delta_g(a[:8, :8], a[:8, :8])  # 대역에 점이 모자란 작은 영상
    flat = np.full((64, 64, 3), 0.5, np.float32)
    v = m.delta_g(flat, a)  # 한쪽에 그레인이 전혀 없어도 inf/NaN 없이 유한값(파워 바닥값)
    assert np.isfinite(v) and v > 5.0
