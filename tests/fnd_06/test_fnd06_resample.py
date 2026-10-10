"""FND-06 resample: resize(반화소 정렬, 축소 시 안티앨리어싱, 기준 구현 = Pillow 'F' 모드)와 sample_bilinear."""
import numpy as np
import pytest
import scipy.ndimage as ndi
from PIL import Image

from buc.core import geom, pyramid

RNG = np.random.default_rng(23)
METHODS = ("nearest", "area", "bilinear", "bicubic", "lanczos")
PIL_FILTER = {"nearest": Image.NEAREST, "area": Image.BOX, "bilinear": Image.BILINEAR,
              "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def pil_resize(img2d, out_hw, method):
    """기준: 같은 float32 평면을 Pillow 'F' 모드로 리사이즈. Pillow 가 좌표 규약(반화소)·지지 폭·정규화의 정의다."""
    im = Image.fromarray(np.ascontiguousarray(img2d, dtype=np.float32))
    assert im.mode == "F"
    return np.asarray(im.resize((out_hw[1], out_hw[0]), PIL_FILTER[method]), dtype=np.float32)


def smooth_noise(h, w, c=None, seed=0):
    r = np.random.default_rng(seed)
    shape = (h, w) if c is None else (h, w, c)
    base = r.uniform(0, 1, shape)
    return ndi.gaussian_filter(base, (1.0, 1.0) + (() if c is None else (0,)), mode="reflect").astype(np.float32)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("src,dst", [((47, 65), (19, 29)), ((21, 31), (47, 71)), ((33, 41), (60, 17)), ((31, 31), (31, 62))])  # 원본 변이 홀수: nearest 경계 동점 없음
def test_resize_matches_pillow(method, src, dst):
    img = smooth_noise(*src, seed=100 * METHODS.index(method) + src[0])
    got = pyramid.resize(img, dst, method)
    assert got.shape == dst and got.dtype == np.float32
    np.testing.assert_allclose(got, pil_resize(img, dst, method), atol=2e-5, rtol=0)


@pytest.mark.parametrize("method", ("area", "bilinear", "bicubic", "lanczos"))
def test_resize_three_channels_and_float64(method):
    img = smooth_noise(30, 40, 3, seed=5)
    got = pyramid.resize(img, (17, 55), method)
    assert got.shape == (17, 55, 3)
    for c in range(3):
        np.testing.assert_allclose(got[..., c], pil_resize(img[..., c], (17, 55), method), atol=2e-5)
    g64 = pyramid.resize(img.astype(np.float64), (17, 55), method)
    assert g64.dtype == np.float64
    np.testing.assert_allclose(g64, got, atol=2e-5)


@pytest.mark.parametrize("method", METHODS)
def test_resize_identity_and_constant(method):
    img = RNG.uniform(0, 1, (13, 17, 3)).astype(np.float32)
    same = pyramid.resize(img, (13, 17), method)
    np.testing.assert_array_equal(same, img)
    keep = img.copy()
    pyramid.resize(img, (5, 40), method)
    np.testing.assert_array_equal(img, keep)
    c = np.full((21, 25), 0.37, np.float64)
    for dst in [(7, 9), (21, 50), (40, 11), (22, 26)]:
        np.testing.assert_allclose(pyramid.resize(c, dst, method), 0.37, atol=1e-6 if method != "nearest" else 0)


def test_resize_area_integer_factor_is_block_mean():
    img = RNG.uniform(0, 1, (12, 18))
    got = pyramid.resize(img, (4, 6), "area")
    want = img.reshape(4, 3, 6, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(got, want, atol=1e-13)
    got2 = pyramid.resize(img, (6, 9), "area")
    np.testing.assert_allclose(got2, img.reshape(6, 2, 9, 2).mean(axis=(1, 3)), atol=1e-13)
    assert got.sum() * 9 == pytest.approx(img.sum(), rel=1e-12)       # 면적 가중이라 질량 보존


def test_resize_nearest_is_center_sampling():
    img = np.arange(20, dtype=np.float32).reshape(4, 5)
    up = pyramid.resize(img, (8, 10), "nearest")
    np.testing.assert_array_equal(up, np.repeat(np.repeat(img, 2, 0), 2, 1))
    down = pyramid.resize(np.arange(10, dtype=np.float32).reshape(1, 10), (1, 5), "nearest")
    np.testing.assert_array_equal(down, [[1, 3, 5, 7, 9]])            # floor((x+0.5)*2) = 2x+1


def test_resize_bilinear_upsample_preserves_ramps_in_half_pixel_coordinates():
    h, w, (a, b, c) = 14, 20, (0.5, -0.25, 3.0)
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    img = a * u + b * v + c
    oh, ow = 33, 47
    out = pyramid.resize(img, (oh, ow), "bilinear")
    j, i = np.mgrid[0:oh, 0:ow].astype(np.float64)
    src_u, src_v = (i + 0.5) * w / ow - 0.5, (j + 0.5) * h / oh - 0.5     # 출력 중심이 가리키는 입력 좌표
    m = (src_u >= 0) & (src_u <= w - 1) & (src_v >= 0) & (src_v <= h - 1)
    np.testing.assert_allclose(out[m], (a * src_u + b * src_v + c)[m], atol=1e-12)


def _blob(h, w, cu, cv, s):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.exp(-((u - cu) ** 2 + (v - cv) ** 2) / (2 * s * s))


def _centroid(img):
    v, u = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
    return (img * u).sum() / img.sum(), (img * v).sum() / img.sum()


@pytest.mark.parametrize("method,scale", [("area", 0.5), ("bilinear", 2.0), ("bilinear", 0.5), ("lanczos", 0.5), ("bicubic", 2.0)])
def test_resize_moves_pixels_exactly_as_scale_intrinsics(method, scale):
    """영상 리샘플과 K 스케일이 같은 좌표 규약(u' = (u+0.5)s - 0.5)을 쓴다. 이음매 테스트."""
    H, W = 64, 80
    cu, cv = 36.3, 28.7
    img = _blob(H, W, cu, cv, 7.0)
    out = pyramid.resize(img, (int(H * scale), int(W * scale)), method)
    cu2, cv2 = _centroid(out)
    cu, cv = _centroid(img)                                            # 입력의 실제 무게중심(경계 절단 영향 제거)
    K = geom.intrinsics_from_fov(60.0, W, H)
    K2 = geom.scale_intrinsics(K, scale)
    # 광선 하나를 골라 두 K 로 투영: 같은 규칙이 영상 위의 점에도 적용된다
    X = np.array([[0.2, -0.1, 3.0]])
    uv, _ = geom.project(K, np.eye(3), np.zeros(3), X)
    uv2, _ = geom.project(K2, np.eye(3), np.zeros(3), X)
    np.testing.assert_allclose(uv2, (uv + 0.5) * scale - 0.5, atol=1e-9)
    assert cu2 == pytest.approx((cu + 0.5) * scale - 0.5, abs=2e-4)
    assert cv2 == pytest.approx((cv + 0.5) * scale - 0.5, abs=2e-4)


def test_resize_argument_validation():
    img = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        pyramid.resize(img, (4, 4), "spline")
    for bad in [(0, 4), (4, 0), (-1, 4)]:
        with pytest.raises(ValueError):
            pyramid.resize(img, bad, "bilinear")
    with pytest.raises(TypeError):
        pyramid.resize(np.zeros((8, 8), np.uint8), (4, 4), "bilinear")
    with pytest.raises(ValueError):
        pyramid.resize(np.zeros(8, np.float32), (4,), "bilinear")
    assert pyramid.resize(img, (4, 4)).shape == (4, 4)                 # 기본 방법은 bilinear


# ---------------------------------------------------------------------------------------------------- sample_bilinear
def test_sample_bilinear_pixel_centers_and_scipy():
    img = RNG.uniform(0, 1, (11, 14, 3))
    v, u = np.mgrid[0:11, 0:14]
    exact = pyramid.sample_bilinear(img, np.stack([u, v], -1).astype(np.float64))
    np.testing.assert_allclose(exact, img, atol=1e-14)                  # 정수 좌표 = 픽셀 중심
    uv = np.stack([RNG.uniform(-2.5, 15.5, 400), RNG.uniform(-2.5, 12.5, 400)], axis=1)
    for border, mode in [("edge", "nearest"), ("zero", "grid-constant")]:
        got = pyramid.sample_bilinear(img, uv, border=border)
        assert got.shape == (400, 3)
        for c in range(3):
            want = ndi.map_coordinates(img[..., c], [uv[:, 1], uv[:, 0]], order=1, mode=mode, cval=0.0)
            np.testing.assert_allclose(got[:, c], want, atol=1e-12)
    gray = img[..., 0]
    g = pyramid.sample_bilinear(gray, uv.reshape(20, 20, 2), border="edge")
    assert g.shape == (20, 20)
    np.testing.assert_allclose(g.reshape(-1), ndi.map_coordinates(gray, [uv[:, 1], uv[:, 0]], order=1, mode="nearest"), atol=1e-12)


def test_sample_bilinear_known_values_and_dtype():
    img = np.array([[0.0, 1.0], [2.0, 4.0]], np.float32)
    got = pyramid.sample_bilinear(img, np.array([[0.5, 0.5], [0.0, 0.5], [0.25, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(got, [1.75, 1.0, 0.25, 4.0], atol=1e-7)
    assert got.dtype == np.float32
    assert pyramid.sample_bilinear(img.astype(np.float64), np.array([[0.5, 0.5]])).dtype == np.float64
    out = pyramid.sample_bilinear(img, np.array([[-1.0, 0.0], [-0.5, 0.0], [5.0, 5.0]]), border="zero")
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-7)            # (-0.5,0): 0.5·0 + 0.5·(밖=0) = 0
    half = pyramid.sample_bilinear(img, np.array([[-0.5, 1.0]]), border="zero")
    assert half[0] == pytest.approx(1.0, abs=1e-7)                         # 0.5·2 + 0.5·0
    edge = pyramid.sample_bilinear(img, np.array([[-0.5, 1.0], [9.0, 9.0]]), border="edge")
    np.testing.assert_allclose(edge, [2.0, 4.0], atol=1e-7)
    nan = pyramid.sample_bilinear(img, np.array([[np.nan, 0.0]]), border="edge")
    assert np.isnan(nan[0])                                                # NaN 좌표는 NaN 으로(조용히 0 이 되지 않는다)
    with pytest.raises(ValueError):
        pyramid.sample_bilinear(img, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        pyramid.sample_bilinear(img, np.zeros((3, 2)), border="wrap")
