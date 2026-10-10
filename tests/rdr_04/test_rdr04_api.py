"""RDR-04 API·검증·결정론: 상수, 옵션 검증, 잘못된 입력은 ValueError/TypeError, 입력 불변, 경고 없음."""
import dataclasses
import warnings

import numpy as np
import pytest

from buc.render import filters as F
from buc.render.filters import (DEGREES, MODES, EwaOpts, PrefilteredTexture, WarpResult, bspline_basis,
                                bspline_sample, ewa_covariance, ewa_sample, homography_uv_jacobian,
                                interp_mtf, prefilter_texture, warp_homography)
from rdr04_helpers import band_image

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def test_constants_and_exports():
    assert tuple(DEGREES) == (1, 3, 5)
    assert tuple(MODES) == ("bilinear", "bspline", "ewa")
    for n in ("EwaOpts", "PrefilteredTexture", "WarpResult", "bspline_basis", "interp_mtf", "prefilter_texture",
              "bspline_sample", "ewa_covariance", "ewa_sample", "homography_uv_jacobian", "warp_homography"):
        assert n in F.__all__ and hasattr(F, n)


def test_ewa_opts_defaults_frozen_kwonly():
    o = EwaOpts()
    assert (o.aa_scale, o.max_aniso, o.max_sigma_px, o.recon_sigma_px) == (0.5, 16.0, 8.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.aa_scale = 1.0
    with pytest.raises(TypeError):
        EwaOpts(0.5)                                      # 위치 인자 금지
    assert EwaOpts(aa_scale=0.0, max_aniso=1.0).max_aniso == 1.0     # 경계값 허용


@pytest.mark.parametrize("kw", [
    {"aa_scale": -0.1}, {"aa_scale": float("nan")}, {"aa_scale": float("inf")}, {"aa_scale": "0.5"},
    {"max_aniso": 0.99}, {"max_aniso": float("nan")}, {"max_sigma_px": 0.0}, {"max_sigma_px": -1.0},
    {"max_sigma_px": float("inf")}, {"recon_sigma_px": -0.01}, {"recon_sigma_px": float("nan")},
])
def test_ewa_opts_rejects(kw):
    with pytest.raises(ValueError):
        EwaOpts(**kw)


@pytest.mark.parametrize("deg", [0, 2, 4, 6, -1, 3.5, "3", None, True])
def test_degree_validation(deg):
    img = np.zeros((6, 6), np.float32)
    with pytest.raises(ValueError):
        prefilter_texture(img, degree=deg)
    with pytest.raises(ValueError):
        bspline_basis(0.0, deg)
    with pytest.raises(ValueError):
        interp_mtf(0.1, deg)


def test_prefilter_input_validation():
    ok = np.zeros((5, 7, 3), np.float32)
    prefilter_texture(ok)
    prefilter_texture(ok[..., 0])
    prefilter_texture(ok.astype(np.float64))
    with pytest.raises(TypeError):
        prefilter_texture(np.zeros((5, 7), np.uint8))
    with pytest.raises(TypeError):
        prefilter_texture(np.zeros((5, 7), np.int32))
    for bad in (np.zeros(5, np.float32), np.zeros((2, 3, 4, 5), np.float32), np.zeros((0, 4), np.float32),
                np.zeros((4, 0, 3), np.float32), np.zeros((4, 4, 0), np.float32)):
        with pytest.raises(ValueError):
            prefilter_texture(bad)
    for v in (np.nan, np.inf, -np.inf):
        a = ok.copy()
        a[2, 3, 1] = v
        with pytest.raises(ValueError):
            prefilter_texture(a)
    with pytest.raises(TypeError):
        prefilter_texture(ok, 3)                           # degree 는 키워드 전용


def test_prefiltered_texture_fields_and_immutable_input():
    img = band_image(24)
    img.flags.writeable = False                           # 읽기 전용 입력도 받는다
    before = img.copy()
    t = prefilter_texture(img, degree=5)
    assert isinstance(t, PrefilteredTexture)
    assert t.degree == 5 and tuple(t.shape) == img.shape
    assert t.coef.shape == img.shape and t.coef.dtype == np.float64
    assert np.array_equal(img, before)
    t2 = prefilter_texture(img[..., 0])
    assert t2.degree == 3 and tuple(t2.shape) == img.shape[:2]


def test_sample_input_validation():
    t = prefilter_texture(band_image(16))
    uv = np.zeros((4, 2))
    assert bspline_sample(t, uv).shape == (4, 3)
    with pytest.raises(TypeError):
        bspline_sample(np.zeros((16, 16, 3)), uv)         # 사전필터 안 한 배열 금지
    with pytest.raises(ValueError):
        bspline_sample(t, np.zeros((4, 3)))
    with pytest.raises(ValueError):
        bspline_sample(t, np.zeros(2)[:0])
    for v in (np.nan, np.inf):
        bad = uv.copy()
        bad[1, 0] = v
        with pytest.raises(ValueError):
            bspline_sample(t, bad)
        with pytest.raises(ValueError):
            ewa_sample(t, bad, np.eye(2))
    with pytest.raises(TypeError):
        bspline_sample(t, np.array([["a", "b"]]))


def test_ewa_sample_and_covariance_validation():
    t = prefilter_texture(band_image(16))
    uv = np.full((5, 2), 4.0)
    ewa_sample(t, uv, np.eye(2))
    ewa_sample(t, uv, np.tile(np.eye(2), (5, 1, 1)))
    with pytest.raises(ValueError):
        ewa_sample(t, uv, np.eye(3))
    with pytest.raises(ValueError):
        ewa_sample(t, uv, np.tile(np.eye(2), (4, 1, 1)))   # 브로드캐스트 불가
    bad = np.eye(2)
    bad[0, 1] = np.nan
    with pytest.raises(ValueError):
        ewa_sample(t, uv, bad)
    with pytest.raises(ValueError):
        ewa_covariance(bad)
    with pytest.raises(ValueError):
        ewa_covariance(np.eye(3))
    with pytest.raises(TypeError):
        ewa_sample(t, uv, np.eye(2), EwaOpts())            # opts 는 키워드 전용
    with pytest.raises(TypeError):
        ewa_covariance(np.eye(2), EwaOpts())
    with pytest.raises(TypeError):
        ewa_sample(np.zeros((16, 16, 3)), uv, np.eye(2))


def test_warp_input_validation():
    img = band_image(24)
    I3 = np.eye(3)
    r = warp_homography(img, I3, (10, 12))
    assert isinstance(r, WarpResult)
    assert r.image.shape == (10, 12, 3) and r.image.dtype == np.float32
    assert r.valid.shape == (10, 12) and r.valid.dtype == bool
    r2 = warp_homography(img[..., 0], I3, (10, 12))
    assert r2.image.shape == (10, 12) and r2.image.dtype == np.float32
    for bad_hw in ((0, 5), (5, 0), (-1, 5), (5.5, 5), (5,), (5, 5, 5), ("a", 4), (True, 4)):
        with pytest.raises(ValueError):
            warp_homography(img, I3, bad_hw)
    for bad_H in (np.eye(2), np.ones((3, 3)), np.zeros((3, 3)), np.diag([1.0, 1.0, 0.0]),
                  np.array([[1, 0, np.nan], [0, 1, 0], [0, 0, 1.0]]), np.array([[1, 0, np.inf], [0, 1, 0], [0, 0, 1.0]])):
        with pytest.raises(ValueError):
            warp_homography(img, bad_H, (4, 4))
    with pytest.raises(ValueError):
        warp_homography(img, I3, (4, 4), mode="lanczos")
    with pytest.raises(ValueError):
        warp_homography(img, I3, (4, 4), fill=float("nan"))
    with pytest.raises(TypeError):
        warp_homography(img, I3, (4, 4), "ewa")             # mode 는 키워드 전용
    with pytest.raises(TypeError):
        warp_homography(np.zeros((8, 8), np.uint8), I3, (4, 4))
    with pytest.raises(ValueError):
        warp_homography(np.full((8, 8), np.nan, np.float32), I3, (4, 4))
    with pytest.raises(TypeError):
        homography_uv_jacobian(np.array([["a"] * 3] * 3), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        homography_uv_jacobian(I3, np.zeros((2, 3)))


@pytest.mark.parametrize("mode", ["bilinear", "bspline", "ewa"])
def test_warp_deterministic_and_input_untouched(mode):
    img = band_image(40)
    img.flags.writeable = False
    before = img.copy()
    H = np.array([[0.8, 0.1, 3.0], [-0.05, 0.9, 2.0], [0.0004, 0.0002, 1.0]])
    a = warp_homography(img, H, (30, 33), mode=mode)
    b = warp_homography(img, H, (30, 33), mode=mode)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.valid.tobytes() == b.valid.tobytes()
    assert np.array_equal(img, before)
    assert np.isfinite(a.image).all()


def test_no_global_rng_change_and_no_warnings():
    img = band_image(32)
    np.random.seed(1234)
    s0 = np.random.get_state()[1].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = prefilter_texture(img)
        bspline_sample(t, np.array([[3.3, 4.4], [-5.0, 100.0]]))
        ewa_sample(t, np.array([[3.3, 4.4], [-5.0, 100.0]]), np.diag([3.0, 0.5]))
        ewa_covariance(np.zeros((2, 2)))
        warp_homography(img, np.diag([0.3, 4.0, 1.0]), (16, 16))
    assert np.array_equal(np.random.get_state()[1], s0)


def test_sample_shapes_and_dtypes():
    t3 = prefilter_texture(band_image(20))
    t2 = prefilter_texture(band_image(20)[..., 0])
    for lead in ((), (7,), (3, 4), (2, 3, 4)):
        uv = np.full(lead + (2,), 5.5)
        assert bspline_sample(t3, uv).shape == lead + (3,)
        assert bspline_sample(t2, uv).shape == lead
        assert ewa_sample(t3, uv, np.eye(2)).shape == lead + (3,)
        assert ewa_sample(t2, uv, np.eye(2)).shape == lead
        assert ewa_covariance(np.zeros(lead + (2, 2))).shape == lead + (2, 2)
    out = bspline_sample(t3, np.array([[1.0, 2.0]], np.float32))     # float32 uv 허용
    assert out.dtype == np.float64
    out = ewa_sample(t3, np.array([[1, 2]]), np.eye(2))               # 정수 uv 허용
    assert out.dtype == np.float64
