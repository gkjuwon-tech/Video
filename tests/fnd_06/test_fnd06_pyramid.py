"""FND-06 pyramid: 가우시안 블러, 반화소 정렬 2배 축소/확대, 가우시안·라플라시안 피라미드, 저대역 교체."""
import numpy as np
import pytest
import scipy.ndimage as ndi

from buc.core import pyramid

RNG = np.random.default_rng(11)


# ---- 독립 오라클 ----
def down1d(a, axis, pad):
    """out[i] = (a[2i-1] + 3 a[2i] + 3 a[2i+1] + a[2i+2]) / 8, 출력 길이 n//2, 경계는 pad 모드."""
    a = np.moveaxis(np.asarray(a, np.float64), axis, 0)
    n = a.shape[0]
    p = np.pad(a, [(1, 2)] + [(0, 0)] * (a.ndim - 1), mode=pad)      # p[k+1] = a[k]
    out = np.stack([(p[2 * i] + 3 * p[2 * i + 1] + 3 * p[2 * i + 2] + p[2 * i + 3]) / 8.0 for i in range(n // 2)])
    return np.moveaxis(out, 0, axis)


def up1d(a, axis):
    """out[2i] = .25 a[i-1] + .75 a[i], out[2i+1] = .75 a[i] + .25 a[i+1], 가장자리 복제."""
    a = np.moveaxis(np.asarray(a, np.float64), axis, 0)
    n = a.shape[0]
    p = np.pad(a, [(1, 1)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    out = np.empty((2 * n,) + a.shape[1:])
    for i in range(n):
        out[2 * i] = 0.25 * p[i] + 0.75 * p[i + 1]
        out[2 * i + 1] = 0.75 * p[i + 1] + 0.25 * p[i + 2]
    return np.moveaxis(out, 0, axis)


def ramp(h, w, a=0.7, b=-0.3, c=2.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    return a * u + b * v + c, (a, b, c)


# ---------------------------------------------------------------------------------------------------- 가우시안 블러
def test_gaussian_kernel_1d():
    for s in [0.5, 1.0, 2.3, 5.0]:
        k = pyramid.gaussian_kernel_1d(s)
        r = int(4.0 * s + 0.5)
        assert k.dtype == np.float64 and k.shape == (2 * r + 1,)
        assert k.sum() == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_allclose(k, k[::-1], atol=0)
        x = np.arange(-r, r + 1)
        want = np.exp(-0.5 * (x / s) ** 2)
        np.testing.assert_allclose(k, want / want.sum(), atol=1e-15)
    assert pyramid.gaussian_kernel_1d(1.0, truncate=2.0).shape == (5,)
    with pytest.raises(ValueError):
        pyramid.gaussian_kernel_1d(0.0)
    with pytest.raises(ValueError):
        pyramid.gaussian_kernel_1d(-1.0)


def test_gaussian_blur_matches_scipy_all_borders():
    img = RNG.uniform(0, 1, (23, 31))
    cmap = {"reflect": ("reflect", 0.0), "edge": ("nearest", 0.0), "zero": ("constant", 0.0)}
    for border, (mode, cval) in cmap.items():
        for s in [0.8, 1.7, 3.0]:
            got = pyramid.gaussian_blur(img, s, border=border)
            want = ndi.gaussian_filter(img, s, mode=mode, cval=cval, truncate=4.0)
            np.testing.assert_allclose(got, want, atol=1e-12)
    rgb = RNG.uniform(0, 1, (17, 19, 3))
    got = pyramid.gaussian_blur(rgb, 1.5)
    for c in range(3):
        np.testing.assert_allclose(got[..., c], ndi.gaussian_filter(rgb[..., c], 1.5, mode="reflect", truncate=4.0), atol=1e-12)
    g32 = pyramid.gaussian_blur(rgb.astype(np.float32), 1.5)
    assert g32.dtype == np.float32 and g32.shape == rgb.shape
    np.testing.assert_allclose(g32, got, atol=2e-6)


def test_gaussian_blur_invariants():
    c = np.full((12, 14, 3), 0.42)
    np.testing.assert_allclose(pyramid.gaussian_blur(c, 2.0), c, atol=1e-14)                    # 상수 보존(경계 포함)
    d = np.zeros((41, 41))
    d[20, 20] = 1.0
    b = pyramid.gaussian_blur(d, 2.0)
    assert b.sum() == pytest.approx(1.0, abs=1e-12)                                             # 질량 보존
    v, u = np.mgrid[0:41, 0:41]
    assert (b * u).sum() == pytest.approx(20.0, abs=1e-12)
    assert (b * (u - 20) ** 2).sum() == pytest.approx(4.0, rel=2e-3)                            # 분산 σ²
    np.testing.assert_allclose(pyramid.gaussian_blur(d, 2.0), pyramid.gaussian_blur(d, 2.0).T, atol=1e-15)
    keep = d.copy()
    pyramid.gaussian_blur(d, 2.0)
    np.testing.assert_array_equal(d, keep)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            pyramid.gaussian_blur(d, bad)
    with pytest.raises(ValueError):
        pyramid.gaussian_blur(d, 1.0, border="wrap")
    with pytest.raises(TypeError):
        pyramid.gaussian_blur(np.zeros((8, 8), np.uint8), 1.0)
    with pytest.raises(ValueError):
        pyramid.gaussian_blur(np.zeros(8), 1.0)


# ---------------------------------------------------------------------------------------------------- 축소/확대
def test_pyr_down_matches_oracle_and_shape_rule():
    for shape in [(8, 12), (9, 13), (6, 7, 3)]:
        img = RNG.uniform(0, 1, shape)
        got = pyramid.pyr_down(img)
        assert got.shape == (shape[0] // 2, shape[1] // 2) + shape[2:]
        want = down1d(down1d(img, 0, "symmetric"), 1, "symmetric")
        np.testing.assert_allclose(got, want, atol=1e-12)
    img = RNG.uniform(0, 1, (8, 8))
    np.testing.assert_allclose(pyramid.pyr_down(img, border="edge"), down1d(down1d(img, 0, "edge"), 1, "edge"), atol=1e-12)
    np.testing.assert_allclose(pyramid.pyr_down(img, border="zero"), down1d(down1d(img, 0, "constant"), 1, "constant"), atol=1e-12)
    g32 = pyramid.pyr_down(img.astype(np.float32))
    assert g32.dtype == np.float32
    with pytest.raises(ValueError):
        pyramid.pyr_down(np.zeros((1, 8)))
    with pytest.raises(TypeError):
        pyramid.pyr_down(np.zeros((8, 8), np.uint8))


def test_pyr_down_half_pixel_alignment_and_mass():
    h, w = 20, 26
    img, (a, b, c) = ramp(h, w)
    out = pyramid.pyr_down(img)
    j, i = np.mgrid[1:h // 2 - 1, 1:w // 2 - 1]                       # 필터 지지가 영상 안에 드는 내부
    want = a * (2 * i + 0.5) + b * (2 * j + 0.5) + c                  # 출력 i 의 중심은 입력 좌표 2i+0.5
    np.testing.assert_allclose(out[1:h // 2 - 1, 1:w // 2 - 1], want, atol=1e-12)
    x = RNG.uniform(0, 1, (16, 24))
    assert pyramid.pyr_down(x).mean() == pytest.approx(x.mean(), abs=1e-14)   # 짝수 크기 + reflect: 평균 정확 보존
    assert pyramid.pyr_down(np.full((7, 9, 3), 0.3)).min() == pytest.approx(0.3, abs=1e-15)


def test_pyr_up_matches_oracle_ramp_and_out_shape():
    img = RNG.uniform(0, 1, (5, 7, 3))
    up = pyramid.pyr_up(img)
    assert up.shape == (10, 14, 3)
    np.testing.assert_allclose(up, up1d(up1d(img, 0), 1), atol=1e-12)
    r, (a, b, c) = ramp(10, 12)
    ru = pyramid.pyr_up(r)
    j, i = np.mgrid[1:19, 1:23]
    src_u, src_v = (i + 0.5) / 2 - 0.5, (j + 0.5) / 2 - 0.5            # 출력 중심이 가리키는 입력 좌표
    np.testing.assert_allclose(ru[1:19, 1:23], a * src_u + b * src_v + c, atol=1e-12)
    np.testing.assert_allclose(pyramid.pyr_up(np.full((4, 4), 0.25)), 0.25, atol=0)              # 경계 포함 상수 보존
    o = pyramid.pyr_up(img, out_shape=(11, 15))                        # 2h+1: 마지막 행/열은 가장자리 복제
    assert o.shape == (11, 15, 3)
    np.testing.assert_allclose(o[:10, :14], up, atol=0)
    np.testing.assert_allclose(o[10, :14], up[9], atol=0)
    np.testing.assert_allclose(o[:10, 14], up[:, 13], atol=0)
    assert pyramid.pyr_up(img, out_shape=(10, 14)).shape == (10, 14, 3)
    for bad in [(9, 14), (12, 14), (10, 16)]:
        with pytest.raises(ValueError):
            pyramid.pyr_up(img, out_shape=bad)
    assert pyramid.pyr_up(img.astype(np.float32)).dtype == np.float32


# ---------------------------------------------------------------------------------------------------- 피라미드
def test_gaussian_pyramid_shapes_and_levels():
    img = RNG.uniform(0, 1, (37, 50, 3)).astype(np.float32)
    keep = img.copy()
    g = pyramid.gaussian_pyramid(img, 3)
    assert len(g) == 4
    np.testing.assert_array_equal(g[0], img)
    assert [x.shape[:2] for x in g] == [(37, 50), (18, 25), (9, 12), (4, 6)]
    assert all(x.dtype == np.float32 for x in g)
    np.testing.assert_allclose(g[1], pyramid.pyr_down(img), atol=0)
    np.testing.assert_array_equal(img, keep)
    assert len(pyramid.gaussian_pyramid(img, 0)) == 1
    with pytest.raises(ValueError):
        pyramid.gaussian_pyramid(np.zeros((8, 8)), 4)                  # 8->4->2->1->(0): 너무 작다
    with pytest.raises(ValueError):
        pyramid.gaussian_pyramid(img, -1)


def test_laplacian_perfect_reconstruction():
    for shape, dt, tol in [((32, 48), np.float64, 1e-12), ((37, 50, 3), np.float64, 1e-12), ((35, 29, 3), np.float32, 2e-6)]:
        img = RNG.uniform(-1, 2, shape).astype(dt)
        for levels in (1, 2, 3):
            lp = pyramid.laplacian_pyramid(img, levels)
            assert len(lp) == levels + 1
            g = pyramid.gaussian_pyramid(img, levels)
            assert [x.shape for x in lp] == [x.shape for x in g]
            np.testing.assert_allclose(lp[-1], g[-1], atol=0)         # 마지막은 잔차(가우시안 최상위)
            rec = pyramid.laplacian_reconstruct(lp)
            assert rec.dtype == dt and rec.shape == img.shape
            np.testing.assert_allclose(rec, img, atol=tol)
    lp = pyramid.laplacian_pyramid(np.full((16, 16, 3), 0.6), 3)
    for band in lp[:-1]:
        np.testing.assert_allclose(band, 0.0, atol=1e-14)             # 상수 영상: 대역 신호 없음
    np.testing.assert_allclose(lp[-1], 0.6, atol=1e-14)
    with pytest.raises(ValueError):
        pyramid.laplacian_reconstruct([])


def test_laplacian_bands_separate_frequencies():
    n = 128
    v, u = np.mgrid[0:n, 0:n].astype(np.float64)
    low = np.cos(2 * np.pi * u / 64.0)                                # 주기 64 px
    high = np.cos(2 * np.pi * u / 4.0)                                # 주기 4 px
    lp_low = pyramid.laplacian_pyramid(low, 3)
    lp_high = pyramid.laplacian_pyramid(high, 3)
    inner = (slice(16, -16), slice(16, -16))
    res = (slice(2, -2), slice(2, -2))                                # 잔차는 16x16
    assert np.abs(lp_low[0][inner]).max() < 0.03                      # 저주파는 band0 에 거의 없고
    assert np.abs(lp_low[3][res]).max() > 0.8                         # 잔차에 남는다
    assert np.abs(lp_high[0][inner]).max() > 0.7                      # 고주파는 band0 에 걸린다
    assert np.abs(lp_high[3][res]).max() < 0.02                       # 잔차에는 거의 없다


def test_laplacian_low_swap():
    img = RNG.uniform(0, 1, (40, 56, 3))
    levels = 3
    g = pyramid.gaussian_pyramid(img, levels)
    same = pyramid.laplacian_low_swap(img, g[levels], levels)          # 자기 저대역으로 바꾸면 항등
    assert same.shape == img.shape and same.dtype == img.dtype
    np.testing.assert_allclose(same, img, atol=1e-12)
    low2 = g[levels] + 0.25                                           # 저대역에 상수를 더하면 전 해상도에서 정확히 +0.25
    np.testing.assert_allclose(pyramid.laplacian_low_swap(img, low2, levels), img + 0.25, atol=1e-12)
    lp = pyramid.laplacian_pyramid(img, levels)                        # 정의: 고대역은 img 의 것, 잔차만 교체
    lp[-1] = low2
    np.testing.assert_allclose(pyramid.laplacian_low_swap(img, low2, levels), pyramid.laplacian_reconstruct(lp), atol=1e-12)
    rnd = RNG.uniform(0, 1, g[levels].shape)
    d = pyramid.laplacian_low_swap(img, rnd, levels) - pyramid.laplacian_low_swap(img, np.zeros_like(rnd), levels)
    up = rnd
    for lvl in range(levels, 0, -1):                                   # 두 결과의 차이는 저대역 차이를 위로 확대한 것뿐
        up = pyramid.pyr_up(up, out_shape=g[lvl - 1].shape[:2])
    np.testing.assert_allclose(d, up, atol=1e-12)
    with pytest.raises(ValueError):
        pyramid.laplacian_low_swap(img, np.zeros((5, 5, 3)), levels)   # 저대역 크기가 맞지 않음
    with pytest.raises(ValueError):
        pyramid.laplacian_low_swap(img, np.zeros_like(g[levels]), 0)
