"""대역 제한 Wiener `wiener_band`: 관측 대역 안에서만 보정, 밖은 건드리지 않는다(발명 금지). 순수 코사인으로 이득을 직접 잰다."""
import numpy as np
import pytest

import rd04_helpers as H
from buc.read import explain_away as ea

HH, WW = 128, 192
MARGIN = 32


def tone(nu0, angle=0.3, amp=0.12):
    x, f = H.tone_scene(HH, WW, nu0, angle, amp)
    return x, f


@pytest.mark.parametrize("c", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("rel", [0.15, 0.4, 0.7, 0.85, 0.95, 1.3, 1.7])
def test_tone_gain_matches_contract_filter(c, rel):
    """진폭 이득 = G(nu;c) = 1 + t·((1+K)H/(H²+K) − 1)  (t: ν<=0.7ν_c 에서 1, 코사인 테이퍼, ν>=ν_c 에서 0)."""
    nu_c = H.X_DEFOCUS_HALF / (np.pi * c)
    nu0 = rel * nu_c
    if nu0 >= 0.45:
        pytest.skip("Nyquist 근처")
    x, f = tone(nu0)
    out = ea.wiener_band(x.astype(np.float32), c)
    assert out.dtype == np.float32 and out.shape == x.shape
    got = H.tone_amplitude(out, f, MARGIN) / H.tone_amplitude(x, f, MARGIN)
    assert got == pytest.approx(float(H.band_gain(nu0, c)), abs=0.03)
    if rel >= 1.0:
        assert got == pytest.approx(1.0, abs=0.01)                # 대역 밖은 손대지 않는다


def test_blurred_tone_is_restored_in_band_but_not_beyond():
    """흐린 코사인: 보정 전 진폭 비 |H|, 보정 후 대역 안(<=0.7 ν_c)에서 1 ± 0.03, 대역 밖에서는 흐린 진폭 그대로."""
    c = 5.0
    nu_c = H.X_DEFOCUS_HALF / (np.pi * c)
    for rel, lo, hi in ((0.3, 0.97, 1.03), (0.7, 0.97, 1.03), (1.4, None, None)):
        nu0 = rel * nu_c
        x, f = tone(nu0)
        blurred = H.blur_otf_periodic(x, c)
        a_sharp = H.tone_amplitude(x, f, MARGIN)
        a_blur = H.tone_amplitude(blurred, f, MARGIN)
        out = ea.wiener_band(blurred.astype(np.float32), c)
        a_out = H.tone_amplitude(out, f, MARGIN)
        if lo is not None:
            assert lo <= a_out / a_sharp <= hi, (rel, a_out / a_sharp)
            assert a_blur / a_sharp < (0.99 if rel < 0.5 else 0.8)             # 시험이 변별력이 있다(흐린 영상은 실제로 감쇠)
        else:
            assert a_out == pytest.approx(a_blur, rel=0.02)


def test_constant_image_is_preserved_and_dc_gain_is_one():
    x = np.full((64, 80, 3), 0.37, np.float32)
    out = ea.wiener_band(x, 6.0)
    assert np.allclose(out, 0.37, atol=1e-6)
    x2 = np.stack([np.full((64, 80), v) for v in (0.1, 0.5, 0.9)], axis=2).astype(np.float32)
    assert np.allclose(ea.wiener_band(x2, 4.0), x2, atol=1e-6)


def test_small_coc_is_identity_and_per_pixel_floor():
    x = H.texture_scene(64, 80, 3).astype(np.float32)
    for c in (0.0, 0.2, 0.49):
        assert np.array_equal(ea.wiener_band(x, c), x), c
    coc = np.zeros((64, 80))
    coc[:, 40:] = 5.0
    out = ea.wiener_band(x, coc)
    assert np.array_equal(out[:, :40], x[:, :40])                     # c < COC_MIN_PX 인 화소는 비트 동일
    assert not np.array_equal(out[:, 40:], x[:, 40:])
    big = ea.wiener_band(x, 0.5)                                      # 경계값 0.5 는 보정 대상(>=)
    assert big.shape == x.shape


def test_two_zone_coc_map_matches_each_uniform_filter():
    """c 가 구역마다 다른 지도: 각 구역 안쪽에서 해당 c 의 균일 필터 출력과 5 % 안에서 같다."""
    nu0 = 0.08
    x, f = tone(nu0)
    coc = np.full((HH, WW), 3.0)
    coc[:, WW // 2:] = 6.0
    out = ea.wiener_band(x.astype(np.float32), coc)
    rows = slice(MARGIN, HH - MARGIN)
    for c, cols in ((3.0, slice(MARGIN, WW // 2 - 24)), (6.0, slice(WW // 2 + 24, WW - MARGIN))):
        uni = ea.wiener_band(x.astype(np.float32), c)
        d = np.abs(out[rows, cols] - uni[rows, cols]).max()
        assert d <= 0.05 * 0.12, (c, d)               # 진폭 0.12 의 5 % 이하


def test_valid_mask_pixels_are_unchanged_and_far_pixels_unaffected():
    x = H.texture_scene(HH, WW, 4).astype(np.float32)
    valid = np.ones((HH, WW), bool)
    valid[20:30, 20:30] = False
    valid[60:64, 100:140] = False
    ref = ea.wiener_band(x, 4.0)
    out = ea.wiener_band(x, 4.0, valid=valid)
    assert np.array_equal(out[~valid], x[~valid])                     # 무효 화소는 입력 그대로
    far = np.ones_like(valid)
    far[8:44, 8:44] = False
    far[44:80, 84:156] = False
    err = np.abs(out - ref)[far].max()
    assert err <= 0.02 * float(x.std()) + 1e-6, err                  # 멀리 떨어진 화소의 보정은 무효 구역 처리에 거의 무관
    allv = ea.wiener_band(x, 4.0, valid=np.ones((HH, WW), bool))
    assert np.array_equal(allv, ref)


def test_output_is_nonnegative_finite_float32_and_pure():
    x = H.texture_scene(64, 96, 5, ev_std=1.2, mean=0.05).astype(np.float32)
    x[20:30, 30:40] = 0.0
    before = x.copy()
    coc = np.full((64, 96), 7.0)
    out = ea.wiener_band(x, coc)
    assert np.array_equal(x, before)
    assert out.dtype == np.float32 and np.all(np.isfinite(out)) and out.min() >= 0.0
    assert out.tobytes() == ea.wiener_band(x, coc).tobytes()


def test_noise_gain_follows_the_filter_and_k_trades_gain_for_noise():
    """백색 잡음의 RMS 이득 = 필터 이득의 주파수 평균 제곱근(오라클 G 로 예측), k 가 작을수록 증폭이 크다."""
    rng = np.random.default_rng(0)
    h, w = 128, 192
    n = rng.normal(0.0, 0.01, (h, w, 3))
    x = (0.3 + n).astype(np.float32)
    c = 4.0
    res = {}
    for k in (0.003, 0.01, 0.1):
        out = ea.wiener_band(x, c, k=k)
        got = H.rms((out - 0.3)[MARGIN:-MARGIN, MARGIN:-MARGIN]) / H.rms(n[MARGIN:-MARGIN, MARGIN:-MARGIN])
        nu = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
        pred = float(np.sqrt(np.mean(H.band_gain(nu, c, k=k) ** 2)))
        assert got == pytest.approx(pred, rel=0.08), (k, got, pred)
        res[k] = got
    assert res[0.003] > res[0.01] > res[0.1] >= 1.0
    # 대역 안 진폭 복원도 k 가 작을수록 1 에 가깝다
    nu_c = H.X_DEFOCUS_HALF / (np.pi * c)
    x1, f = tone(0.7 * nu_c)
    bl = H.blur_otf_periodic(x1, c).astype(np.float32)
    a = [H.tone_amplitude(ea.wiener_band(bl, c, k=k), f, MARGIN) / H.tone_amplitude(x1, f, MARGIN) for k in (0.003, 0.01, 0.1)]
    assert abs(a[0] - 1) <= abs(a[1] - 1) + 5e-3 and abs(a[1] - 1) < abs(a[2] - 1)


def test_h_min_moves_the_band_edge():
    c = 5.0
    x, f = tone(0.9 * H.X_DEFOCUS_HALF / (np.pi * c))
    a05 = H.tone_amplitude(ea.wiener_band(x.astype(np.float32), c, h_min=0.5), f, MARGIN)
    a08 = H.tone_amplitude(ea.wiener_band(x.astype(np.float32), c, h_min=0.8), f, MARGIN)    # 대역이 좁아져 이 톤은 대역 밖
    assert a08 == pytest.approx(H.tone_amplitude(x, f, MARGIN), rel=0.01)
    assert a05 > 1.1 * a08


def test_validation():
    x = np.full((16, 16, 3), 0.3, np.float32)
    with pytest.raises(TypeError):
        ea.wiener_band(x.tolist(), 3.0)
    with pytest.raises(TypeError):
        ea.wiener_band((x * 255).astype(np.uint8), 3.0)
    for bad in (x[..., 0], x[:, :, :2], np.full((16, 16, 3), -0.1, np.float32), np.full((16, 16, 3), np.nan, np.float32)):
        with pytest.raises(ValueError):
            ea.wiener_band(bad, 3.0)
    for bad in (True, "3", [3.0]):
        with pytest.raises(TypeError):
            ea.wiener_band(x, bad)
    for bad in (-1.0, np.nan, np.inf, np.ones((16, 15))):
        with pytest.raises(ValueError):
            ea.wiener_band(x, bad)
    for bad in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            ea.wiener_band(x, 3.0, k=bad)
    for bad in (0.0, 1.0, 2.0):
        with pytest.raises(ValueError):
            ea.wiener_band(x, 3.0, h_min=bad)
    for bad in (np.ones((16, 15), bool), np.ones((16, 16), np.uint8)):
        with pytest.raises((ValueError, TypeError)):
            ea.wiener_band(x, 3.0, valid=bad)
