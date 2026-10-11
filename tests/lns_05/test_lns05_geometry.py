"""LNS-05 기하 수차: 방사 왜곡(Brown-Conrady 역사상), 점 사상, 역왜곡, 횡색수차."""
import inspect
import math

import numpy as np
import pytest

from buc.lens import optics
from lns05_helpers import (blob, center, centroid, grid, half_diag, impulse, ridge_x, smooth_fn, smooth_img,
                           solve_rd)


# ----------------------------------------------------------------------------------------------- API 형태
def test_public_names_and_signatures():
    for n in ("distortion_scale", "apply_distortion", "undistort_image", "ideal_to_distorted", "distorted_to_ideal",
              "apply_lateral_ca", "catseye_kernel", "apply_catseye", "GhostSpec", "DEFAULT_GHOSTS", "ghost_layer",
              "veiling_glare", "star_psf", "star_layer", "OpticsSpec", "spec_from_theta", "apply_optics",
              "apply_optics_frames", "GLARE_EXPONENT", "GLARE_R0_FRAC", "STAR_CHROMA", "STAR_PSF_SIZE",
              "STAR_APERTURE_FRAC", "CATSEYE_MAX_SHIFT"):
        assert hasattr(optics, n), n
    assert optics.GLARE_EXPONENT == 1.5 and optics.GLARE_R0_FRAC == 0.03
    assert tuple(optics.STAR_CHROMA) == (1.12, 1.0, 0.88)
    assert optics.STAR_PSF_SIZE == 257 and optics.STAR_APERTURE_FRAC == 0.25 and optics.CATSEYE_MAX_SHIFT == 0.5
    kwonly = inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(optics.apply_distortion).parameters["border"].kind is kwonly
    assert inspect.signature(optics.apply_distortion).parameters["nearest"].kind is kwonly
    assert inspect.signature(optics.ghost_layer).parameters["threshold"].kind is kwonly
    assert inspect.signature(optics.veiling_glare).parameters["r0_frac"].kind is kwonly
    assert inspect.signature(optics.apply_catseye).parameters["strength"].kind is kwonly


# ----------------------------------------------------------------------------------------------- 왜곡: 영상
def test_distortion_zero_is_copy_and_dtype_preserved():
    for dt in (np.float32, np.float64):
        x = smooth_img(40, 56, dt)
        keep = x.copy()
        y = optics.apply_distortion(x, 0.0, 0.0)
        assert y is not x and y.dtype == dt and np.array_equal(y, x) and np.array_equal(x, keep)
        z = optics.undistort_image(x, 0, 0)
        assert z is not x and np.array_equal(z, x)


def test_distortion_matches_analytic_backward_map():
    """out(p) = in(c + (p-c) s(r)), s = 1 + k1 r^2 + k2 r^4, r = |p-c| / R. 해석 장면과 직접 비교."""
    h, w = 90, 120
    k1, k2 = 0.08, -0.02
    x, y = grid(h, w)
    cx, cy = center(h, w)
    R = half_diag(h, w)
    r = np.hypot(x - cx, y - cy) / R
    s = 1 + k1 * r ** 2 + k2 * r ** 4
    assert np.allclose(optics.distortion_scale(r, k1, k2), s, atol=1e-12)
    sx, sy = cx + (x - cx) * s, cy + (y - cy) * s
    inside = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    ref = smooth_fn(sx, sy)
    for dt in (np.float32, np.float64):
        out = optics.apply_distortion(smooth_img(h, w, dt), k1, k2)
        assert out.dtype == dt and out.shape == (h, w, 3)
        err = np.abs(out.astype(np.float64) - ref)[inside]
        assert err.max() < 6e-3, err.max()
    # 중심 화소와 상(2-D) 입력: 채널 슬라이스와 같다
    img = smooth_img(h, w)
    out3 = optics.apply_distortion(img, k1, k2)
    out2 = optics.apply_distortion(np.ascontiguousarray(img[..., 1]), k1, k2)
    assert out2.shape == (h, w) and np.allclose(out2, out3[..., 1], atol=1e-6)


def test_distortion_center_param():
    h, w = 61, 81
    img = smooth_img(h, w)
    c = (20.0, 30.0)
    out = optics.apply_distortion(img, 0.1, 0.0, center=c)
    assert np.allclose(out[30, 20], img[30, 20], atol=1e-6)  # 주점은 고정점
    cc = optics.apply_distortion(img, 0.1, 0.0)
    assert np.allclose(cc[30, 40], img[30, 40], atol=1e-6)  # 기본 주점 = 영상 중심
    assert not np.allclose(out, cc, atol=1e-3)


def test_distortion_border_modes():
    h, w = 41, 61
    img = np.ones((h, w, 3), np.float32)
    edge = optics.apply_distortion(img, 0.3, 0.0, border="edge")
    zero = optics.apply_distortion(img, 0.3, 0.0, border="zero")
    assert np.allclose(edge, 1.0, atol=1e-6)  # 가장자리 클램프: 상수 영상은 그대로
    assert zero[20, 30, 0] > 0.999 and zero[0, 0, 0] == 0.0  # 바깥을 가리키는 모서리는 0
    assert zero.min() == 0.0 and zero.max() <= 1.0 + 1e-6


def test_barrel_vs_pincushion_straight_line_bows():
    """세로 직선은 배럴(k1>0)에서 바깥으로, 핀쿠션(k1<0)에서 안쪽으로 휜다. 점 사상 오라클과 0.4 px 일치."""
    h, w = 121, 161
    cx, cy = center(h, w)
    x_line = cx + 50.0
    x, y = grid(h, w)
    ridge = np.exp(-((x - x_line) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
    img = np.repeat(ridge[..., None], 3, axis=-1)
    mids = {}
    for k1 in (0.1, -0.1):
        out = optics.apply_distortion(img, k1, 0.0)
        ys = np.arange(5, h - 5, 10)
        pred = optics.ideal_to_distorted(np.stack([np.full(len(ys), x_line), ys.astype(float)], -1), k1, 0.0, (h, w))
        for yy, p in zip(ys, pred):
            got = ridge_x(out[..., 0], int(yy), p[0])
            assert abs(got - p[0]) < 0.4, (k1, yy, got, p[0])
        mids[k1] = (ridge_x(out[..., 0], int(cy), x_line), ridge_x(out[..., 0], 5, x_line + (0.5 if k1 > 0 else -0.5)))
    assert mids[0.1][0] > mids[0.1][1] + 1.0  # 배럴: 가운데 행이 위쪽 행보다 바깥(우측)
    assert mids[-0.1][0] < mids[-0.1][1] - 1.0  # 핀쿠션: 반대


# ----------------------------------------------------------------------------------------------- 점 사상·역왜곡
def test_point_maps_roundtrip_corner_and_center():
    h, w = 90, 160
    cx, cy = center(h, w)
    R = half_diag(h, w)
    rng = np.random.default_rng(7)
    pts = np.stack([rng.uniform(0, w - 1, 400), rng.uniform(0, h - 1, 400)], -1)
    for k1, k2 in ((0.1, 0.0), (-0.1, 0.0), (0.15, -0.05), (0.05, 0.05)):
        d = optics.ideal_to_distorted(pts, k1, k2, (h, w))
        assert d.shape == pts.shape
        back = optics.distorted_to_ideal(d, k1, k2, (h, w))
        assert np.abs(back - pts).max() < 1e-6, (k1, k2)
        # 정의: 이상 -> 왜곡 반지름은 r_d (1 + k1 r_d^2 + k2 r_d^4) = r_q 의 근
        rq = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / R
        rd = np.hypot(d[:, 0] - cx, d[:, 1] - cy) / R
        for i in range(0, 400, 40):
            assert abs(rd[i] - solve_rd(rq[i], k1, k2)) < 1e-7
        # 방향 보존
        v0, v1 = pts - [cx, cy], d - [cx, cy]
        assert np.abs(v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]).max() < 1e-6
    corner = np.array([[0.0, 0.0], [w - 1.0, h - 1.0]])
    dc = optics.ideal_to_distorted(corner, 0.05, 0.0, (h, w))
    rd = np.hypot(dc[:, 0] - cx, dc[:, 1] - cy) / R
    assert np.allclose(rd, solve_rd(1.0, 0.05, 0.0), atol=1e-7) and rd[0] < 1.0  # 배럴: 모서리가 안쪽으로
    assert np.allclose(optics.ideal_to_distorted(np.array([cx, cy]), 0.1, 0.02, (h, w)), [cx, cy], atol=1e-9)
    ident = optics.ideal_to_distorted(pts, 0.0, 0.0, (h, w))
    assert np.allclose(ident, pts, atol=1e-9)
    assert optics.ideal_to_distorted(pts.reshape(20, 20, 2), 0.1, 0.0, (h, w)).shape == (20, 20, 2)


def test_point_map_agrees_with_image_warp():
    """영상에 심은 가우시안 점은 ideal_to_distorted 가 예측한 곳에 나타난다."""
    h, w = 101, 141
    k1, k2 = 0.12, 0.0
    for (px, py) in ((100.0, 30.0), (30.0, 80.0), (110.0, 90.0)):
        img = np.repeat(blob(h, w, px, py, sigma=2.0)[..., None], 3, -1).astype(np.float32)
        out = optics.apply_distortion(img, k1, k2)
        pred = optics.ideal_to_distorted(np.array([px, py]), k1, k2, (h, w))
        cxo, cyo = centroid(out[..., 1])
        assert math.hypot(cxo - pred[0], cyo - pred[1]) < 0.3, (px, py, (cxo, cyo), pred)


def test_undistort_roundtrip():
    h, w = 100, 140
    cx, cy = center(h, w)
    R = half_diag(h, w)
    x, y = grid(h, w)
    inner = np.hypot(x - cx, y - cy) / R < 0.55  # 핀쿠션에서도 표본이 프레임 안
    img = smooth_img(h, w)
    for k1, k2 in ((0.08, 0.0), (-0.08, 0.0), (0.1, -0.03)):
        back = optics.undistort_image(optics.apply_distortion(img, k1, k2), k1, k2)
        err = np.abs(back - img)[inner]
        assert err.mean() < 4e-3 and err.max() < 2e-2, (k1, k2, err.mean(), err.max())
        fwd = optics.apply_distortion(optics.undistort_image(img, k1, k2), k1, k2)
        assert np.abs(fwd - img)[inner].mean() < 4e-3


def test_nearest_keeps_label_values_and_moves_edges():
    h, w = 64, 96
    lab = np.zeros((h, w), np.float32)
    lab[:, 60:] = 7.0
    lab[20:40, 10:30] = 3.0
    out = optics.apply_distortion(lab, 0.1, 0.0, nearest=True)
    assert set(np.unique(out)) <= {0.0, 3.0, 7.0}  # 보간 없음
    lin = optics.apply_distortion(lab, 0.1, 0.0)
    assert len(np.unique(lin)) > 3  # 쌍선형은 경계에서 중간값을 만든다
    assert not np.array_equal(out, lab)


# ----------------------------------------------------------------------------------------------- 왜곡 검증
def test_distortion_validation():
    img = smooth_img(32, 48)
    with pytest.raises(TypeError):
        optics.apply_distortion((img * 255).astype(np.uint8), 0.1, 0.0)
    with pytest.raises(TypeError):
        optics.apply_distortion(img.astype(np.float16), 0.1, 0.0)
    bad = img.copy()
    bad[3, 3, 0] = np.nan
    with pytest.raises(ValueError):
        optics.apply_distortion(bad, 0.1, 0.0)
    for k in (float("nan"), float("inf"), True, "0.1"):
        with pytest.raises(ValueError):
            optics.apply_distortion(img, k, 0.0)
        with pytest.raises(ValueError):
            optics.apply_distortion(img, 0.0, k)
    with pytest.raises(ValueError):  # d/dr (r s(r)) = 1 + 3 k1 r^2 + 5 k2 r^4 가 [0,1] 에서 양수가 아니면 역사상이 없다
        optics.apply_distortion(img, -0.4, 0.0)
    with pytest.raises(ValueError):
        optics.undistort_image(img, -0.4, 0.0)
    with pytest.raises(ValueError):
        optics.ideal_to_distorted(np.zeros((3, 2)), -0.4, 0.0, (32, 48))
    with pytest.raises(ValueError):
        optics.apply_distortion(img, 5.0, 0.0)  # |k| <= 2
    with pytest.raises(ValueError):
        optics.apply_distortion(img[0, :, 0], 0.1, 0.0)  # ndim 1
    with pytest.raises(ValueError):
        optics.apply_distortion(img, 0.1, 0.0, border="mirror")
    optics.apply_distortion(img, -0.3, 0.0)  # 경계: 1 - 0.9 = 0.1 > 0 허용


# ----------------------------------------------------------------------------------------------- 횡색수차
def test_lateral_ca_zero_green_exact_and_constant():
    h, w = 61, 81
    img = smooth_img(h, w)
    keep = img.copy()
    z = optics.apply_lateral_ca(img, 0.0, 0.0)
    assert z is not img and np.array_equal(z, img)
    out = optics.apply_lateral_ca(img, 3.0, -2.0)
    assert out.dtype == np.float32 and np.array_equal(out[..., 1], img[..., 1])  # 초록은 기준
    assert np.array_equal(img, keep)
    assert not np.allclose(out[..., 0], img[..., 0], atol=1e-3) and not np.allclose(out[..., 2], img[..., 2], atol=1e-3)
    const = np.full((h, w, 3), 0.37, np.float32)
    assert np.abs(optics.apply_lateral_ca(const, 5.0, -4.0) - 0.37).max() < 1e-6  # 복사(휘도 보존) 표본화
    cx, cy = center(h, w)
    assert np.allclose(out[int(cy), int(cx)], img[int(cy), int(cx)], atol=1e-6)  # 광축 불변


def test_lateral_ca_shifts_channel_centroids_radially():
    h, w = 121, 161
    cx, cy = center(h, w)
    R = half_diag(h, w)
    red_px, blue_px = 6.0, -4.0
    for (dx, dy) in ((40.0, 10.0), (-35.0, 25.0), (20.0, -30.0)):
        x0, y0 = cx + dx, cy + dy
        b = blob(h, w, x0, y0, sigma=2.0)
        img = np.repeat(b[..., None], 3, -1).astype(np.float32)
        out = optics.apply_lateral_ca(img, red_px, blue_px)
        r0 = math.hypot(dx, dy)
        for ch, px in ((0, red_px), (2, blue_px)):
            gx, gy = centroid(out[..., ch])
            rr = math.hypot(gx - cx, gy - cy)
            assert abs(rr - r0 * (1 + px / R)) < 0.12, (dx, dy, ch, rr, r0 * (1 + px / R))
            cross = (gx - cx) * dy - (gy - cy) * dx  # 방향 불변
            assert abs(cross) / r0 < 0.12
        gx, gy = centroid(out[..., 1])
        assert math.hypot(gx - x0, gy - y0) < 1e-3
    # 변위는 반지름에 선형 (절반 반지름 -> 절반 변위)
    sh = []
    for f in (1.0, 0.5):
        x0, y0 = cx + 60 * f, cy
        img = np.repeat(blob(h, w, x0, y0)[..., None], 3, -1).astype(np.float32)
        sh.append(centroid(optics.apply_lateral_ca(img, 8.0, 0.0)[..., 0])[0] - x0)
    assert abs(sh[0] - 2 * sh[1]) < 0.15 and abs(sh[0] - 60 / R * 8.0) < 0.12


def test_lateral_ca_channels_do_not_mix():
    h, w = 61, 81
    img = np.zeros((h, w, 3), np.float32)
    img[..., 1] = blob(h, w, 60, 20)
    out = optics.apply_lateral_ca(img, 4.0, -4.0)
    assert np.array_equal(out[..., 1], img[..., 1]) and out[..., 0].max() == 0 and out[..., 2].max() == 0
    img2 = impulse(h, w, 60, 20, 1.0)
    img2[..., 1] = 0
    out2 = optics.apply_lateral_ca(img2, 4.0, 0.0)
    assert out2[..., 1].max() == 0 and out2[..., 2].max() == pytest.approx(img2[..., 2].max(), abs=1e-6)  # B 는 0 px 이동
    assert np.array_equal(out2[..., 2], img2[..., 2])


def test_lateral_ca_validation():
    img = smooth_img(30, 40)
    with pytest.raises(TypeError):
        optics.apply_lateral_ca(img.astype(np.float16), 1.0, 1.0)
    with pytest.raises(ValueError):
        optics.apply_lateral_ca(img, float("nan"), 0.0)
    with pytest.raises(ValueError):
        optics.apply_lateral_ca(img[..., :2], 1.0, 0.0)  # 채널 3개 필요
    with pytest.raises(ValueError):
        optics.apply_lateral_ca(img, 1e4, 0.0)  # 확대율 > 2 (|px| <= R) 는 거부
