"""대역 한계 마스크 M_b: 전달함수·컷오프 오라클, 옥타브 규칙, 단조, 소스 결합(min), 검증."""
import numpy as np
import pytest

import rd04_helpers as H
from buc.read import explain_away as ea


def a(x):
    return np.asarray(x, np.float64)


# ------------------------------------------------------------------------------------------------ 전달함수
def test_defocus_mtf_matches_pillbox_oracle():
    nu = np.linspace(0.0, 0.5, 51)
    for c in (0.0, 0.7, 2.0, 5.5, 12.0):
        got = a(ea.defocus_mtf(nu, c))
        assert got.shape == nu.shape
        assert np.allclose(got, np.abs(H.defocus_otf(nu, c)), atol=1e-9, rtol=0), c
    assert np.all(a(ea.defocus_mtf(nu, 0.0)) == 1.0)
    assert float(a(ea.defocus_mtf(0.0, 7.0))) == 1.0
    # 짝함수, 방송(broadcast)
    assert np.allclose(a(ea.defocus_mtf(-nu, 3.0)), a(ea.defocus_mtf(nu, 3.0)))
    assert a(ea.defocus_mtf(nu[:, None], np.array([1.0, 2.0, 3.0])[None, :])).shape == (51, 3)


def test_motion_mtf_is_abs_sinc():
    nu = np.linspace(0.0, 0.5, 41)
    for L in (0.0, 1.0, 4.0, 9.5, 21.0):
        assert np.allclose(a(ea.motion_mtf(nu, L)), np.abs(np.sinc(L * nu)), atol=1e-12, rtol=0), L
    assert float(a(ea.motion_mtf(0.0, 8.0))) == 1.0


def test_mtf_validation():
    for fn in (ea.defocus_mtf, ea.motion_mtf):
        for bad in (-1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                fn(0.1, bad)
        with pytest.raises(ValueError):
            fn(np.nan, 1.0)


# ------------------------------------------------------------------------------------------------ 컷오프
@pytest.mark.parametrize("c", [0.8, 2.0, 3.0, 4.0, 7.5, 12.0, 30.0])
def test_defocus_cutoff_is_half_amplitude_point(c):
    nu_c = float(a(ea.defocus_cutoff_nu(c)))
    assert nu_c == pytest.approx(H.X_DEFOCUS_HALF / (np.pi * c), rel=2e-5)
    assert float(a(ea.defocus_mtf(nu_c, c))) == pytest.approx(0.5, abs=1e-8)
    assert float(a(ea.defocus_mtf(0.97 * nu_c, c))) > 0.5 > float(a(ea.defocus_mtf(1.03 * nu_c, c)))
    # 첫 로브 안: 컷오프 아래에서 MTF 는 단조 감소 (>= 0.5)
    g = a(ea.defocus_mtf(np.linspace(0, nu_c, 30), c))
    assert np.all(np.diff(g) <= 1e-12) and g[-1] == pytest.approx(0.5, abs=1e-8)


@pytest.mark.parametrize("L", [1.5, 4.0, 9.0, 21.0])
def test_motion_cutoff_is_half_amplitude_point(L):
    nu_c = float(a(ea.motion_cutoff_nu(L)))
    assert nu_c == pytest.approx(H.X_MOTION_HALF / (np.pi * L), rel=2e-5)
    assert float(a(ea.motion_mtf(nu_c, L))) == pytest.approx(0.5, abs=1e-8)


def test_cutoff_zero_blur_is_infinite_and_scaling():
    assert np.isinf(float(a(ea.defocus_cutoff_nu(0.0)))) and np.isinf(float(a(ea.motion_cutoff_nu(0.0))))
    cs = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    out = a(ea.defocus_cutoff_nu(cs))
    assert out.shape == cs.shape and np.isinf(out[0])
    assert np.allclose(out[1:] * cs[1:], out[1] * cs[1])            # nu_c ∝ 1/c
    assert np.all(np.diff(out) < 0)


def test_cutoff_depends_on_h_min():
    c = 5.0
    lo, mid, hi = (float(a(ea.defocus_cutoff_nu(c, h))) for h in (0.3, 0.5, 0.8))
    assert lo > mid > hi > 0
    for h in (0.3, 0.8):
        assert float(a(ea.defocus_mtf(float(a(ea.defocus_cutoff_nu(c, h))), c))) == pytest.approx(h, abs=1e-8)
    for h in (0.3, 0.8):
        assert float(a(ea.motion_mtf(float(a(ea.motion_cutoff_nu(c, h))), c))) == pytest.approx(h, abs=1e-8)
    for bad in (0.0, 1.0, -0.1, 1.2, np.nan):
        with pytest.raises(ValueError):
            ea.defocus_cutoff_nu(2.0, bad)
        with pytest.raises(ValueError):
            ea.motion_cutoff_nu(2.0, bad)
    for fn in (ea.defocus_cutoff_nu, ea.motion_cutoff_nu):
        for bad in (-1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                fn(bad)


# ------------------------------------------------------------------------------------------------ 옥타브 지도
def test_no_blur_full_resolution_all_octaves_observed():
    b = ea.observed_band((6, 7))
    assert b.cutoff_nu.shape == (6, 7) and b.cutoff_nu.dtype == np.float32
    assert np.all(b.cutoff_nu == np.float32(0.5))
    assert b.mask.shape == (4, 6, 7) and b.mask.all()
    assert np.all(b.min_octave == 0) and b.min_octave.dtype == np.uint8


@pytest.mark.parametrize("r,expect_min", [(1.0, 0), (0.5, 1), (0.25, 2), (0.125, 3), (1.0 / 16.0, 4), (0.3, 2), (0.9, 1)])
def test_r_ratio_caps_the_top_octaves(r, expect_min):
    """RD-01 r_eff/r_in: 컷오프는 0.5·r_ratio. 옥타브 b 는 윗 가장자리 0.5·2^-b <= 컷오프일 때만 관측."""
    b = ea.observed_band((3, 4), r_ratio=r)
    assert np.all(b.cutoff_nu == np.float32(0.5 * r))
    assert np.all(b.min_octave == expect_min)
    for k in range(4):
        assert bool(b.mask[k, 0, 0]) == (k >= expect_min)


@pytest.mark.parametrize("c,expect_min", [(0.5, 0), (1.0, 0), (1.5, 1), (2.0, 1), (3.0, 2), (4.0, 2), (5.0, 2), (6.0, 3), (10.0, 3), (12.0, 4), (40.0, 4)])
def test_defocus_octave_rule(c, expect_min):
    b = ea.observed_band((2, 2), coc_px=c)
    nu_c = H.X_DEFOCUS_HALF / (np.pi * c)
    want = next((k for k in range(4) if H.OCTAVE_HI[k] <= nu_c * (1 + 1e-9)), 4) if True else None
    assert want == expect_min                                   # 오라클 자체 점검
    assert np.all(b.min_octave == expect_min)
    assert np.all(b.cutoff_nu == pytest.approx(min(0.5, nu_c), rel=1e-6))


def test_motion_and_defocus_combine_by_min():
    h, w = 4, 5
    coc = np.full((h, w), 2.0)                                   # nu_c ~ 0.352
    mot = np.full((h, w), 0.0)
    mot[:, 2:] = 12.0                                            # nu_c ~ 0.050 (< 1/16: 어느 옥타브도 아님)
    r_ratio = 0.6                                                # 0.3
    b = ea.observed_band((h, w), coc_px=coc, motion_len_px=mot, r_ratio=r_ratio)
    want = np.minimum.reduce([np.full((h, w), 0.5 * r_ratio),
                              H.X_DEFOCUS_HALF / (np.pi * coc),
                              np.where(mot > 0, H.X_MOTION_HALF / (np.pi * np.where(mot > 0, mot, 1)), np.inf)])
    assert np.allclose(b.cutoff_nu, want, rtol=1e-5)
    assert np.all(b.min_octave[:, :2] == 1) and np.all(b.min_octave[:, 2:] == 4)   # 0.3 -> 옥타브 1(0.25) 부터 / 0.050 -> 없음
    # 어느 한 요인이 컷오프를 낮추면 관측 옥타브는 줄기만 한다
    base = ea.observed_band((h, w), coc_px=coc, r_ratio=r_ratio)
    assert np.all(b.mask <= base.mask)


def test_band_map_is_per_pixel_and_scalar_equals_map():
    h, w = 5, 6
    coc = np.zeros((h, w))
    coc[:, 3:] = 4.0
    b = ea.observed_band((h, w), coc_px=coc)
    assert np.all(b.min_octave[:, :3] == 0) and np.all(b.min_octave[:, 3:] == 2)
    s = ea.observed_band((h, w), coc_px=4.0)
    m = ea.observed_band((h, w), coc_px=np.full((h, w), 4.0))
    assert np.array_equal(s.mask, m.mask) and np.array_equal(s.cutoff_nu, m.cutoff_nu)
    # 단조: 흐림이 클수록 관측 옥타브는 단조 감소
    cs = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 14.0])[None, :]
    mo = ea.observed_band((1, cs.size), coc_px=cs).min_octave[0]
    assert np.all(np.diff(mo.astype(int)) >= 0)


def test_h_min_changes_the_octave_rule():
    strict = ea.observed_band((1, 1), coc_px=3.0, h_min=0.8)      # 컷오프 더 낮음
    loose = ea.observed_band((1, 1), coc_px=3.0, h_min=0.2)
    assert strict.cutoff_nu[0, 0] < loose.cutoff_nu[0, 0]
    assert strict.min_octave[0, 0] >= loose.min_octave[0, 0]


def test_observed_band_validation():
    for bad in ((0, 4), (4, 0), (4,), (2, 3, 4), (2.5, 3), (True, 3), "ab", None):
        with pytest.raises((ValueError, TypeError)):
            ea.observed_band(bad)
    for bad in (True, "2"):
        with pytest.raises(TypeError):
            ea.observed_band((3, 3), coc_px=bad)
        with pytest.raises(TypeError):
            ea.observed_band((3, 3), motion_len_px=bad)
    for bad in (-1.0, np.nan, np.inf, np.ones((3, 4)), -np.ones((3, 3))):
        with pytest.raises(ValueError):
            ea.observed_band((3, 3), coc_px=bad)
    for bad in (0.0, 0.03, 1.5, np.nan):
        with pytest.raises(ValueError):
            ea.observed_band((3, 3), r_ratio=bad)
    for bad in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError):
            ea.observed_band((3, 3), h_min=bad)
