"""RD-01 실효 해상도 r_eff: 방사 파워 스펙트럼 절단점으로 추정한다.

기준 정의: 방사 PSD 를 거듭제곱 법칙으로 맞추고, 그 모델의 25 % 아래로 지속해서 내려가는 첫 주파수를 ν_c 로 한다.
ratio = ν_c / ν_Nyq (cycles/px, Nyquist = 0.5), r_in = 짧은 변, r_eff = ratio · r_in.
합성 입력은 진폭 ν^-1 (전력 ν^-2) 장이라 물리적 정답이 알려져 있다: k 배 업샘플 -> ratio ≈ 1/k,
σ 가우시안 블러 -> 전력 응답 exp(-(2πνσ)²) = 0.25 에서 ratio ≈ 0.375/σ."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import rd01_helpers as H
from buc.read import ingest as ing

N = 512


def _ratio_of(img, **kw):
    r = ing.ingest(img, color_space="srgb", **kw)
    return r.r_eff / r.r_in, r


def test_native_resolution_field_keeps_full_resolution():
    for seed in range(3):
        ratio, r = _ratio_of(H.rgb3(H.upsampled_field(N, 1, seed=seed)))
        assert r.r_in == N and ratio >= 0.9 and r.r_eff <= r.r_in
        assert "r_eff_unreliable" not in r.warnings


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_upsampled_x2_is_about_half(method):
    for seed in range(3):
        ratio, _ = _ratio_of(H.rgb3(H.upsampled_field(N, 2, seed=seed, method=method)))
        assert 0.38 <= ratio <= 0.65, (method, seed, ratio)


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_upsampled_x4_is_about_a_quarter(method):
    for seed in range(3):
        ratio, _ = _ratio_of(H.rgb3(H.upsampled_field(N, 4, seed=seed, method=method)))
        assert 0.19 <= ratio <= 0.36, (method, seed, ratio)


def test_more_upsampling_means_lower_r_eff_with_clear_margins():
    rs = [_ratio_of(H.rgb3(H.upsampled_field(N, k, seed=1)))[0] for k in (1, 2, 4)]
    assert rs[0] > rs[1] + 0.2 and rs[1] > rs[2] + 0.12


def test_gaussian_blur_follows_the_analytic_cutoff_and_is_monotone():
    base = H.field(N, seed=5)
    sig = [0.0, 0.7, 1.0, 1.5, 2.0, 3.0]
    rs = [_ratio_of(H.rgb3(np.clip(gaussian_filter(base, s) if s else base, 0, 1).astype(np.float32)))[0] for s in sig]
    assert rs[0] >= 0.9
    for a, b in zip(rs, rs[1:]):
        assert b <= a + 0.02
    assert 0.28 <= rs[2] <= 0.50          # σ=1.0 -> 0.375
    assert 0.18 <= rs[3] <= 0.37          # σ=1.5 -> 0.25
    assert 0.14 <= rs[4] <= 0.32          # σ=2.0 -> 0.19
    assert rs[5] <= 0.25


def test_heavy_blur_is_small_but_never_below_the_floor():
    img = H.rgb3(np.clip(gaussian_filter(H.field(N, seed=5), 6), 0, 1).astype(np.float32))
    r = ing.ingest(img, color_space="srgb")
    ratio = r.r_eff / r.r_in
    assert 1 / 16 - 1e-9 <= ratio <= 0.3


def test_white_noise_is_not_mistaken_for_a_low_resolution_image():
    rng = np.random.default_rng(0)
    img = H.rgb3(np.clip(rng.normal(0.5, 0.15, (N, N)), 0, 1).astype(np.float32))
    ratio, r = _ratio_of(img)
    assert ratio >= 0.9


def test_8bit_png_and_jpeg_paths_agree_with_the_float_path():
    f = H.upsampled_field(N, 2, seed=0)
    ratio_f, _ = _ratio_of(H.rgb3(f))
    ratio_png, r = _ratio_of(H.png_bytes(H.to_u8(H.rgb3(f))))
    assert abs(ratio_png - ratio_f) <= 0.05 and r.fmt == "png"
    native = H.to_u8(H.rgb3(H.upsampled_field(N, 1, seed=2)))
    ratio_jpg, rj = _ratio_of(H.jpeg_bytes(native, quality=90))
    assert rj.fmt == "jpeg" and ratio_jpg >= 0.8


def test_non_square_uses_the_short_side_and_respects_the_ratio():
    img = H.rgb3(H.field(320, 512, seed=8).astype(np.float32))
    r = ing.ingest(img, color_space="srgb")
    assert r.r_in == 320 and r.r_eff >= 0.9 * 320 and r.r_eff <= 320
    from buc.core import pyramid
    up = np.clip(pyramid.resize(H.field(160, 256, seed=8).astype(np.float32), (320, 512), "bicubic"), 0, 1)
    r2 = ing.ingest(H.rgb3(up), color_space="srgb")
    assert 0.38 <= r2.r_eff / r2.r_in <= 0.65


def test_color_space_does_not_change_the_cutoff_of_a_gray_image():
    g = H.rgb3(H.upsampled_field(N, 2, seed=1))
    rs = [_ratio_of(g)[0], ing.ingest(g, color_space="display-p3").r_eff / N, ing.ingest(g, color_space="adobe-rgb").r_eff / N]
    assert max(rs) - min(rs) <= 0.05


# ---------------------------------------------------------------------------------------------------------------
# estimate_r_eff 직접 호출: 입력 계약과 퇴화 경우
# ---------------------------------------------------------------------------------------------------------------
def test_estimate_r_eff_fields_and_ranges():
    est = ing.estimate_r_eff(H.upsampled_field(N, 2, seed=0).astype(np.float64))
    assert est.r_in == N and est.reliable is True
    assert est.ratio == pytest.approx(est.r_eff / est.r_in, rel=1e-12)
    assert 1 / 16 <= est.ratio <= 1.0 and 0 < est.r_eff <= est.r_in
    est32 = ing.estimate_r_eff(H.upsampled_field(N, 2, seed=0))
    assert abs(est32.ratio - est.ratio) <= 0.02


def test_constant_image_is_degenerate_and_flagged_not_crashing():
    est = ing.estimate_r_eff(np.full((128, 128), 0.4, np.float32))
    assert est.reliable is False and est.r_eff == est.r_in == 128 and est.ratio == 1.0
    r = ing.ingest(np.full((128, 128, 3), 0.4, np.float32), color_space="srgb")
    assert r.r_eff == r.r_in and "r_eff_unreliable" in r.warnings


def test_tiny_images_are_flagged_unreliable():
    img = H.rgb3(H.field(40, seed=1).astype(np.float32))
    r = ing.ingest(img, color_space="srgb")
    assert r.r_in == 40 and r.r_eff == 40 and "r_eff_unreliable" in r.warnings
    est = ing.estimate_r_eff(np.random.default_rng(0).uniform(size=(63, 200)).astype(np.float32))
    assert est.reliable is False and est.r_in == 63


def test_estimate_r_eff_input_checks_and_purity():
    y = H.upsampled_field(N, 2, seed=0)
    keep = y.copy()
    a, b = ing.estimate_r_eff(y), ing.estimate_r_eff(y)
    assert a == b
    np.testing.assert_array_equal(y, keep)
    for bad in (np.zeros((8,)), np.zeros((8, 8, 3), np.float32)):
        with pytest.raises(ValueError):
            ing.estimate_r_eff(bad)
    with pytest.raises(ValueError):
        ing.estimate_r_eff(np.zeros((0, 8), np.float32))
    with pytest.raises(TypeError):
        ing.estimate_r_eff((y * 255).astype(np.uint8))


def test_estimate_does_not_touch_global_rng():
    s0 = np.random.get_state()[1].copy()
    ing.estimate_r_eff(H.upsampled_field(N, 2, seed=0))
    assert (np.random.get_state()[1] == s0).all()
