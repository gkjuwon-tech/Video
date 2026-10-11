"""LNS-05 플레어: veiling glare(넓은 꼬리 PSF), 회절 별(|F{조리개}|^2)."""
import math

import numpy as np
import pytest

from buc.lens import optics
from lns05_helpers import (blob, centroid, expected_rays, glare_ref, hdr_scene, impulse, match_rays, ring_peaks,
                           ring_profile)


# ----------------------------------------------------------------------------------------------- veiling glare
def test_glare_zero_amount_is_copy_and_input_immutable():
    for dt in (np.float32, np.float64):
        x = hdr_scene().astype(dt)
        keep = x.copy()
        y = optics.veiling_glare(x, 0.0)
        assert y is not x and y.dtype == dt and np.array_equal(y, x) and np.array_equal(x, keep)
        z = optics.veiling_glare(x, 0.2)
        assert z.dtype == dt and np.array_equal(x, keep)


def test_glare_matches_oracle():
    """out = (1-a) x + a (h*x)/(h*1),  h(r) = (1 + (r/r0)^2)^-1.5,  r0 = r0_frac * W."""
    h, w = 64, 96
    x = np.full((h, w, 3), 0.02, np.float32)
    x[20:34, 30:50] = np.array([1.0, 0.9, 0.7])  # 밝은 사각형
    x += 0.01 * np.random.default_rng(4).random((h, w, 3)).astype(np.float32)
    for amount, r0f in ((0.1, 0.03), (0.3, 0.03), (0.2, 0.08)):
        kw = {} if r0f == 0.03 else dict(r0_frac=r0f)
        out = optics.veiling_glare(x, amount, **kw)
        ref = glare_ref(x, amount, r0f)
        assert np.abs(out - ref).max() < 1.5e-3, (amount, r0f, np.abs(out - ref).max())
    # 임펄스: 꼬리 비율도 맞는다 (2% 이내, 값이 큰 곳만)
    imp = impulse(h, w, 48, 32, 0.0)
    imp[32, 48, :] = 100.0
    out = optics.veiling_glare(imp, 0.5)
    ref = glare_ref(imp, 0.5)
    m = ref > 1e-3 * ref.max()
    assert np.abs(out[m] / ref[m] - 1).max() < 0.02


def test_glare_amount_interpolates_linearly_and_is_linear_in_light():
    x = hdr_scene(48, 64)
    g1 = optics.veiling_glare(x, 1.0)
    for a in (0.1, 0.37, 0.8):
        assert np.abs(optics.veiling_glare(x, a) - ((1 - a) * x + a * g1)).max() < 1e-4 * x.max()
    y = hdr_scene(48, 64, seed=9)
    a, b = optics.veiling_glare(x, 0.3), optics.veiling_glare(y, 0.3)
    s = optics.veiling_glare(x + y, 0.3)
    assert np.abs(s - (a + b)).max() < 1e-4 * s.max()  # 가산성
    k = optics.veiling_glare(7.0 * x, 0.3)
    assert np.abs(k - 7.0 * a).max() < 1e-4 * k.max()  # 동차성
    assert optics.veiling_glare(x, 0.3).min() >= 0  # 비음수 입력 -> 비음수 출력


def test_glare_uniform_field_and_channel_independence():
    u = np.full((50, 70, 3), 0.4, np.float32)
    assert np.abs(optics.veiling_glare(u, 0.6) - 0.4).max() < 1e-5  # 가장자리 정규화: 균일 장은 불변
    x = np.zeros((50, 70, 3), np.float32)
    x[..., 0] = blob(50, 70, 35, 25, 3.0)
    out = optics.veiling_glare(x, 0.3)
    assert out[..., 1].max() == 0 and out[..., 2].max() == 0 and out[..., 0].max() > 0  # 같은 PSF, 섞이지 않음


def test_glare_lifts_black_with_distance_decay():
    """밝은 반평면 옆의 검정: 경계에서 멀어질수록 단조 감소하는 양의 들뜸 -> 대비 저하."""
    h, w = 40, 128
    x = np.zeros((h, w, 3), np.float32)
    x[:, :48] = 1.0
    out = optics.veiling_glare(x, 0.1)
    row = out[20, 48:, 0]
    assert row[0] > 0.01 and np.all(np.diff(row[:60]) <= 1e-7) and row[-1] > 0  # (프레임 가장자리 근처는 정규화로 다시 오른다)
    assert row[0] > 3 * row[40]
    assert out[20, 10, 0] < 1.0  # 밝은 쪽은 약간 어두워진다(에너지가 퍼짐)
    assert 0.04 < row[40] / row[10] < 0.2  # 꼬리는 가우시안(~0)보다 훨씬 두껍다 (멱법칙, 지수 1.5)


def test_glare_validation():
    x = hdr_scene(30, 40)
    for bad in (-0.1, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            optics.veiling_glare(x, bad)
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            optics.veiling_glare(x, 0.1, r0_frac=bad)
    with pytest.raises(TypeError):
        optics.veiling_glare(x.astype(np.uint8), 0.1)
    xn = x.copy()
    xn[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        optics.veiling_glare(xn, 0.1)
    with pytest.raises(ValueError):
        optics.veiling_glare(x[..., 0], 0.1)
    optics.veiling_glare(x, 1.0)


# ----------------------------------------------------------------------------------------------- 회절 별 PSF
def test_star_psf_basic_properties():
    for blades in (0, 5, 6):
        p = optics.star_psf(blades, 10.0)
        assert p.shape == (257, 257, 3) and p.dtype == np.float32 and p.min() >= 0
        for c in range(3):
            assert abs(float(p[..., c].sum()) - 1.0) < 1e-4
            assert np.unravel_index(int(np.argmax(p[..., c])), p[..., c].shape) == (128, 128)  # 중심이 최대
            m = p[..., c]
            assert np.abs(m - m[::-1, ::-1]).max() < 1e-5 * m.max()  # |F|^2 는 점대칭
    small = optics.star_psf(6, 0.0, size=129)
    assert small.shape == (129, 129, 3)
    assert np.array_equal(optics.star_psf(6, 5.0), optics.star_psf(6, 5.0))
    # 파장 배율: 조리개를 1/s 로 줄인 것과 같다. 정규화된 PSF 의 중심값은 개구 면적에 비례하므로 s^-2 비
    for blades in (0, 6):
        p = optics.star_psf(blades, 0.0)
        s_r, s_b = optics.STAR_CHROMA[0], optics.STAR_CHROMA[2]
        assert abs(p[128, 128, 0] / p[128, 128, 1] - 1.0 / s_r ** 2) < 0.04
        assert abs(p[128, 128, 2] / p[128, 128, 1] - 1.0 / s_b ** 2) < 0.04


@pytest.mark.parametrize("blades", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("rot", [0.0, 17.0])
def test_star_psf_spike_count_and_angles(blades, rot):
    """짝수 날 N -> N 개, 홀수 날 N -> 2N 개 광선. 광선은 변 법선(rot + (2i+1)*180/N)과 그 반대 방향."""
    p = optics.star_psf(blades, rot)[..., 1]
    ang, prof = ring_profile(p, 128.0, 128.0, 50, 120)
    found = ring_peaks(ang, prof)
    want = expected_rays(blades, rot)
    assert len(want) == (blades if blades % 2 == 0 else 2 * blades)
    assert match_rays(found, want, 3.0), (blades, rot, found, want)


def test_star_psf_circular_aperture_is_isotropic():
    p = optics.star_psf(0)[..., 1]
    ang, prof = ring_profile(p, 128.0, 128.0, 50, 120)
    assert prof.max() / prof.min() < 3.5  # 광선 없음 (에어리 원판; 화소 계단 때문에 완전한 등방은 아니다)
    ang6, prof6 = ring_profile(optics.star_psf(6)[..., 1], 128.0, 128.0, 50, 120)
    assert prof6.max() / prof6.min() > 20.0  # 6 날은 뚜렷한 광선
    assert not match_rays(ring_peaks(ang, prof), expected_rays(6, 0.0), 3.0)


def test_star_psf_validation():
    for bad in (1, 2, 19, 5.5, True, -1):
        with pytest.raises(ValueError):
            optics.star_psf(bad)
    for bad in (128, 16, 0):
        with pytest.raises(ValueError):
            optics.star_psf(6, size=bad)  # 홀수 >= 33
    with pytest.raises(ValueError):
        optics.star_psf(6, float("nan"))
    with pytest.raises(ValueError):
        optics.star_psf(6, aperture_frac=0.0)


# ----------------------------------------------------------------------------------------------- 별 층 (영상)
def test_star_layer_energy_position_and_spikes():
    n = 161
    c = 80
    img = impulse(n, n, c, c, 0.0)
    img[c, c, :] = 101.0  # 초과분 100.2 (문턱 0.8)
    keep = img.copy()
    for blades, rot in ((6, 0.0), (5, 17.0)):
        lay = optics.star_layer(img, blades=blades, rotation_deg=rot, threshold=0.8, size_frac=0.9)
        assert np.array_equal(img, keep) and lay.shape == img.shape and lay.dtype == np.float32
        assert lay.min() >= -1e-6 * lay.max()
        for ch in range(3):
            m = lay[..., ch].astype(np.float64)
            assert abs(m.sum() / 100.2 - 1.0) < 2e-3  # PSF 합 1 -> 광속 보존
            gx, gy = centroid(m)
            assert math.hypot(gx - c, gy - c) < 0.3
        ang, prof = ring_profile(lay[..., 1], float(c), float(c), 30, 70)
        found = ring_peaks(ang, prof)
        want = expected_rays(blades, rot)
        assert match_rays(found, want, 4.0), (blades, rot, found, want)
    # 파장 배율: 중심값 비 R/G = 1/1.12^2, B/G = 1/0.88^2 (재표본 오차 포함 8%)
    lay = optics.star_layer(img, blades=6, threshold=0.8, size_frac=0.9)
    assert abs(lay[c, c, 0] / lay[c, c, 1] - 1 / 1.12 ** 2) < 0.08
    assert abs(lay[c, c, 2] / lay[c, c, 1] - 1 / 0.88 ** 2) < 0.1


def test_star_layer_threshold_linearity_and_validation():
    n = 81
    x = np.full((n, n, 3), 0.5, np.float32)
    x[40, 40, :] = 0.79
    assert not np.any(optics.star_layer(x, blades=6, threshold=0.8, size_frac=0.5))
    a = impulse(n, n, 40, 40, 2.0)
    b = impulse(n, n, 40, 40, 3.0)
    la = optics.star_layer(a, blades=6, threshold=1.0, size_frac=0.5)
    lb = optics.star_layer(b, blades=6, threshold=1.0, size_frac=0.5)
    assert np.allclose(lb, 2.0 * la, rtol=1e-4, atol=1e-7)
    assert optics.star_layer(a.astype(np.float64), blades=6, threshold=1.0, size_frac=0.5).dtype == np.float64
    assert np.array_equal(la, optics.star_layer(a, blades=6, threshold=1.0, size_frac=0.5))
    for bad in (dict(blades=2), dict(threshold=-1.0), dict(size_frac=0.0), dict(size_frac=1.5), dict(rotation_deg=float("nan"))):
        kw = dict(blades=6, threshold=1.0, size_frac=0.5)
        kw.update(bad)
        with pytest.raises(ValueError):
            optics.star_layer(a, **kw)
    with pytest.raises(TypeError):
        optics.star_layer(a.astype(np.uint8), blades=6)
    with pytest.raises(TypeError):
        optics.star_layer(a, 6)  # 키워드 전용
