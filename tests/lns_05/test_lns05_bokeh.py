"""LNS-05 보케·고스트: cat-eye 렌즈 모양 커널(반경별), 단일 층 산란 블러, Hullin 고스트 층."""
import math

import numpy as np
import pytest

from buc.lens import dof, optics
from lns05_helpers import (aniso, ang_diff180, center, centroid, cov2, half_diag, impulse, minor_axis_deg,
                           second_moment)


def lens_area(d, o):
    """지름 d 원 두 개를 중심 간격 o 로 겹친 교집합 넓이 (해석)."""
    R = d / 2.0
    return 2 * R * R * math.acos(o / (2 * R)) - (o / 2.0) * math.sqrt(4 * R * R - o * o)


# ----------------------------------------------------------------------------------------------- cat-eye 커널
def test_catseye_reduces_to_aperture_kernel():
    for d in (1.0, 0.4, 9.0, 20.0):
        for blades, rot in ((0, 0.0), (6, 10.0)):
            ref = dof.aperture_kernel(d, blades, rot)
            for rf, st in ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)):
                k = optics.catseye_kernel(d, rf, 33.0, st, blades, rot)
                assert k.dtype == np.float32 and np.array_equal(k, ref), (d, blades, rf, st)
    assert optics.catseye_kernel(1.0, 1.0, 0.0, 1.0).shape == (1, 1)


def test_catseye_kernel_basic_properties():
    d = 24.0
    ref = dof.aperture_kernel(d)
    for rf in (0.25, 0.5, 1.0):
        for ang in (0.0, 30.0, 90.0, 135.0):
            k = optics.catseye_kernel(d, rf, ang, 1.0)
            assert k.dtype == np.float32 and k.shape == ref.shape and k.shape[0] % 2 == 1
            assert k.min() >= 0 and abs(float(k.sum()) - 1.0) < 1e-5
            cx, cy = centroid(k)
            h = (k.shape[0] - 1) / 2.0
            assert math.hypot(cx - h, cy - h) < 0.1  # 질량중심이 커널 중심
            assert np.allclose(k, k[::-1, ::-1], atol=1e-6)  # 원형 렌즈 모양은 점대칭
    k0 = optics.catseye_kernel(d, 0.7, 0.0, 1.0)
    k180 = optics.catseye_kernel(d, 0.7, 180.0, 1.0)
    assert np.allclose(k0, k180, atol=1e-6)
    ka, kb = optics.catseye_kernel(d, 0.7, 0.0, 1.0), optics.catseye_kernel(d, 0.7, 90.0, 1.0)
    assert np.allclose(ka.T, kb, atol=1e-6)  # 방향 (1,0) 과 (0,1) 은 x/y 전치
    assert np.allclose(ka, ka[:, ::-1], atol=1e-6) and np.allclose(ka, ka[::-1, :], atol=1e-6)  # 축 대칭


def test_catseye_geometry_area_and_orientation():
    """렌즈 모양 = 반지름 d/2 원 두 개(간격 o = MAX_SHIFT*strength*radial_frac*d)의 교집합. 작은 축이 반경 방향."""
    d = 30.0
    for st, rf in ((1.0, 1.0), (1.0, 0.6), (0.5, 1.0)):
        o = optics.CATSEYE_MAX_SHIFT * st * rf * d
        k = optics.catseye_kernel(d, rf, 20.0, st)
        area = 1.0 / float(k.max())  # 완전히 덮인 화소의 값 = 1 / 넓이
        assert abs(area / lens_area(d, o) - 1.0) < 0.03, (st, rf, area, lens_area(d, o))
    areas, ratios = [], []
    for rf in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = optics.catseye_kernel(d, rf, 0.0, 1.0)
        areas.append(1.0 / float((k.astype(np.float64) ** 2).sum()))  # 유효 넓이
        ratios.append(aniso(cov2(k)))
    assert all(a > b for a, b in zip(areas, areas[1:]))  # 가장자리일수록 작아진다 (엄격 단조)
    assert all(b > a for a, b in zip(ratios, ratios[1:])) and ratios[0] < 1.02 and ratios[-1] > 1.4
    for ang in (0.0, 30.0, 90.0, 135.0):
        k = optics.catseye_kernel(d, 1.0, ang, 1.0)
        assert ang_diff180(minor_axis_deg(cov2(k)), ang) < 6.0, ang
    # 세기 단조: 같은 rf 에서 strength 가 클수록 더 비등방
    rs = [aniso(cov2(optics.catseye_kernel(d, 1.0, 0.0, s))) for s in (0.0, 0.3, 0.6, 1.0)]
    assert all(b > a for a, b in zip(rs, rs[1:]))
    # 연속성: 작은 세기는 원형 조리개와 L1 거리가 작다
    near = optics.catseye_kernel(d, 1.0, 0.0, 0.02)
    assert np.abs(near.astype(np.float64) - dof.aperture_kernel(d)).sum() < 0.12


def test_catseye_polygon_aperture():
    d = 28.0
    for blades, rot in ((6, 0.0), (8, 11.0)):
        base = dof.aperture_kernel(d, blades, rot)
        k = optics.catseye_kernel(d, 1.0, 0.0, 1.0, blades, rot)
        assert k.shape == base.shape and k.min() >= 0 and abs(float(k.sum()) - 1.0) < 1e-5
        assert (k > 0).sum() < (base > 0).sum()  # 지지 영역이 줄어든다
        assert aniso(cov2(k)) > 1.05
        h = (k.shape[0] - 1) / 2.0
        cx, cy = centroid(k)
        assert math.hypot(cx - h, cy - h) < 0.1


def test_catseye_kernel_validation():
    for bad in (dict(diameter_px=-1.0), dict(diameter_px=float("nan")), dict(radial_frac=-0.1),
                dict(radial_frac=1.5), dict(strength=-0.1), dict(strength=1.2), dict(blades=2), dict(blades=19),
                dict(blades=5.5), dict(blades=True), dict(angle_deg=float("inf"))):
        a = dict(diameter_px=10.0, radial_frac=0.5, angle_deg=0.0, strength=1.0, blades=0)
        a.update(bad)
        with pytest.raises(ValueError):
            optics.catseye_kernel(**a)


# ----------------------------------------------------------------------------------------------- apply_catseye
def test_apply_catseye_zero_coc_is_copy_and_input_immutable():
    x = np.random.default_rng(1).random((40, 52, 3)).astype(np.float32)
    keep = x.copy()
    y = optics.apply_catseye(x, np.zeros((40, 52)), strength=1.0)
    assert y is not x and y.dtype == np.float32 and np.allclose(y, x, atol=1e-6) and np.array_equal(x, keep)
    y2 = optics.apply_catseye(x.astype(np.float64), np.full((40, 52), 0.3), strength=0.5)
    assert y2.dtype == np.float64 and np.allclose(y2, x, atol=1e-6)  # CoC <= 1 px 는 선명(커널 [[1]])


def test_apply_catseye_impulse_matches_lens_kernel():
    n, c = 97, 14.0
    cx, cy = center(n, n)
    R = half_diag(n, n)
    coc = np.full((n, n), c)
    for (dx, dy) in ((30, 0), (0, 30), (21, 21), (-28, 12)):
        img = impulse(n, n, cx + dx, cy + dy, 0.0)
        img[int(cy + dy), int(cx + dx), :] = 10.0
        out = optics.apply_catseye(img, coc, strength=1.0)
        assert out.dtype == np.float32 and out.min() >= -1e-6
        for ch in range(3):
            m = out[..., ch].astype(np.float64)
            assert abs(m.sum() - 10.0) < 0.05  # 에너지 보존 0.5%
            gx, gy = centroid(m)
            assert math.hypot(gx - (cx + dx), gy - (cy + dy)) < 0.3
        m = out[..., 1].astype(np.float64)
        rf = math.hypot(dx, dy) / R
        ang = math.degrees(math.atan2(dy, dx))
        ref = optics.catseye_kernel(c, rf, ang, 1.0).astype(np.float64)
        cm = cov2(m)
        assert ang_diff180(minor_axis_deg(cm), ang) < 12.0, (dx, dy)  # 반경 방향이 짧다
        assert abs(aniso(cm) - aniso(cov2(ref))) < 0.12, (dx, dy, aniso(cm), aniso(cov2(ref)))
        assert abs(cm.trace() / cov2(ref).trace() - 1.0) < 0.15
    # 광축 위 임펄스: 원형 조리개 커널 그대로
    ix, iy = int(cx), int(cy)
    img = impulse(n, n, ix, iy, 0.0)
    img[iy, ix, :] = 10.0
    out = optics.apply_catseye(img, coc, strength=1.0)
    k = dof.aperture_kernel(c).astype(np.float64)
    h = (k.shape[0] - 1) // 2
    patch = out[iy - h:iy + h + 1, ix - h:ix + h + 1, 1].astype(np.float64) / 10.0
    assert np.abs(patch - k).sum() < 0.02
    assert abs(out[..., 1].sum() - 10.0) < 0.05


def test_apply_catseye_strength_zero_equals_plain_aperture_blur():
    n, c = 81, 12.0
    img = impulse(n, n, 60, 20, 0.0)
    img[20, 60, :] = 8.0
    out = optics.apply_catseye(img, np.full((n, n), c), strength=0.0, blades=6, blade_rot_deg=10.0)
    k = dof.aperture_kernel(c, 6, 10.0).astype(np.float64)
    h = (k.shape[0] - 1) // 2
    patch = out[20 - h:20 + h + 1, 60 - h:60 + h + 1, 0].astype(np.float64) / 8.0
    assert np.abs(patch - k).sum() < 1e-3
    # 균일 영상 + 균일 CoC: 내부는 상수 (strength=0 이면 정확히, 1 이면 평균 6% 이내)
    u = np.full((n, n, 3), 0.5, np.float32)
    o0 = optics.apply_catseye(u, np.full((n, n), c), strength=0.0)
    assert np.abs(o0[16:-16, 16:-16] - 0.5).max() < 1e-4
    o1 = optics.apply_catseye(u, np.full((n, n), c), strength=1.0)
    inner = o1[16:-16, 16:-16]
    assert abs(inner.mean() - 0.5) < 0.03 and inner.std() < 0.05


def test_apply_catseye_per_pixel_coc_scatter():
    """CoC 는 광원(산란하는 화소)의 값: 선명한 쪽 임펄스는 그대로, 흐린 쪽 임펄스는 퍼진다."""
    h, w = 64, 128
    coc = np.zeros((h, w))
    coc[:, 64:] = 12.0
    img = np.zeros((h, w, 3), np.float32)
    img[32, 20, :] = 5.0
    img[32, 96, :] = 5.0
    out = optics.apply_catseye(img, coc, strength=0.5)
    assert out[32, 20, 0] == pytest.approx(5.0, abs=1e-5) and out[32, 21, 0] < 1e-5
    right = out[:, 80:112, 0]
    assert abs(float(right.sum()) - 5.0) < 0.03 and right.max() < 0.2
    assert out.dtype == np.float32


def test_apply_catseye_deterministic_and_validation():
    x = np.random.default_rng(2).random((30, 40, 3)).astype(np.float32)
    c = np.full((30, 40), 6.0)
    a = optics.apply_catseye(x, c, strength=0.7, blades=7)
    b = optics.apply_catseye(x, c, strength=0.7, blades=7)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        optics.apply_catseye(x, c[:, :-1], strength=0.5)
    with pytest.raises(ValueError):
        optics.apply_catseye(x, -c, strength=0.5)
    cn = c.copy()
    cn[0, 0] = np.nan
    with pytest.raises(ValueError):
        optics.apply_catseye(x, cn, strength=0.5)
    with pytest.raises(ValueError):
        optics.apply_catseye(x, c, strength=1.5)
    with pytest.raises(ValueError):
        optics.apply_catseye(x, c, strength=0.5, blades=2)
    with pytest.raises(TypeError):
        optics.apply_catseye(x.astype(np.uint8), c, strength=0.5)
    with pytest.raises(TypeError):
        optics.apply_catseye(x, c, 0.5)  # strength 는 키워드 전용


# ----------------------------------------------------------------------------------------------- 고스트
def test_ghostspec_validation_and_defaults():
    g = optics.GhostSpec(scale=-0.5, size_frac=0.05, gain=0.1)
    assert tuple(g.tint) == (1.0, 1.0, 1.0)
    with pytest.raises(Exception):
        g.gain = 0.2  # frozen
    for bad in (dict(scale=float("nan")), dict(scale=3.5), dict(size_frac=0.0), dict(size_frac=0.6),
                dict(gain=-0.1), dict(tint=(1.0, 1.0)), dict(tint=(1.0, -0.1, 1.0)), dict(gain=float("inf"))):
        a = dict(scale=-0.5, size_frac=0.05, gain=0.1)
        a.update(bad)
        with pytest.raises(ValueError):
            optics.GhostSpec(**a)
    assert len(optics.DEFAULT_GHOSTS) >= 4 and all(isinstance(x, optics.GhostSpec) for x in optics.DEFAULT_GHOSTS)
    sc = [x.scale for x in optics.DEFAULT_GHOSTS]
    assert min(sc) < 0 < max(sc) and max(sc) > 1.0  # 중심 반대편 고스트와 광원 너머 고스트가 모두 있다


def test_ghost_layer_position_flux_and_shape():
    h, w = 121, 161
    cx, cy = center(h, w)
    thr = 0.8
    sx, sy = int(cx + 30), int(cy - 10)
    img = np.zeros((h, w, 3), np.float32)
    img[sy, sx, :] = 5.8  # 초과분 5.0
    gs = optics.GhostSpec(scale=-0.5, size_frac=0.10, gain=0.2, tint=(1.0, 0.5, 0.25))
    keep = img.copy()
    lay = optics.ghost_layer(img, [gs], threshold=thr)
    assert np.array_equal(img, keep) and lay.shape == img.shape and lay.dtype == np.float32 and lay.min() >= -1e-6 * lay.max()
    gx, gy = cx - 15.0, cy + 5.0  # c + scale (p - c)
    d = 0.10 * w
    for ch, t in enumerate(gs.tint):
        m = lay[..., ch].astype(np.float64)
        assert abs(m.sum() - 0.2 * t * 5.0) < 0.01 * 0.2 * t * 5.0 + 1e-6  # 광속 = gain * tint * 초과분
        px, py = centroid(m)
        assert math.hypot(px - gx, py - gy) < 0.7
        assert abs(cov2(m).trace() / (d * d / 8.0) - 1.0) < 0.15  # 원형 조리개: E[r^2] = d^2/8
    # 정수 위치에 떨어지면 정확히 조리개 커널 (blades=6)
    lay6 = optics.ghost_layer(img, [gs], threshold=thr, blades=6, blade_rot_deg=15.0)
    k = dof.aperture_kernel(d, 6, 15.0).astype(np.float64)
    hh = (k.shape[0] - 1) // 2
    x0, y0 = int(round(gx)), int(round(gy))
    patch = lay6[y0 - hh:y0 + hh + 1, x0 - hh:x0 + hh + 1, 0].astype(np.float64)
    assert np.abs(patch - 0.2 * 5.0 * k).max() < 1e-4 * 0.2 * 5.0 * k.max() + 1e-5
    assert abs(float(lay6[..., 0].sum()) - 1.0) < 1e-3


def test_ghost_layer_collinear_and_scale_semantics():
    h, w = 121, 161
    cx, cy = center(h, w)
    sx, sy = cx + 44.0, cy + 20.0
    img = np.zeros((h, w, 3), np.float32)
    img[int(sy), int(sx), :] = 3.0
    v = np.array([sx - cx, sy - cy])
    for scale in (-0.7, -0.2, 0.4, 1.0, 1.3):
        lay = optics.ghost_layer(img, [optics.GhostSpec(scale, 0.05, 0.5)], threshold=1.0)
        gx, gy = centroid(lay[..., 1].astype(np.float64))
        want = np.array([cx, cy]) + scale * v
        assert math.hypot(gx - want[0], gy - want[1]) < 0.8, scale
        cross = (gx - cx) * v[1] - (gy - cy) * v[0]
        assert abs(cross) / np.linalg.norm(v) < 0.8
    two = optics.ghost_layer(img, [optics.GhostSpec(-0.3, 0.05, 0.5), optics.GhostSpec(0.6, 0.05, 0.25)], threshold=1.0)
    s1 = optics.ghost_layer(img, [optics.GhostSpec(-0.3, 0.05, 0.5)], threshold=1.0)
    s2 = optics.ghost_layer(img, [optics.GhostSpec(0.6, 0.05, 0.25)], threshold=1.0)
    assert np.allclose(two, s1 + s2, atol=1e-5)  # 고스트는 가산적


def test_ghost_layer_threshold_linearity_and_edge_cases():
    h, w = 60, 80
    base = np.full((h, w, 3), 0.3, np.float32)
    base[30, 50, :] = 0.79
    gs = (optics.GhostSpec(-0.4, 0.06, 0.3),)
    assert not np.any(optics.ghost_layer(base, gs, threshold=0.8))  # 문턱 아래는 정확히 0
    assert not np.any(optics.ghost_layer(base, (), threshold=0.0))
    a = np.zeros((h, w, 3), np.float32)
    a[30, 50, :] = 2.0
    b = np.zeros((h, w, 3), np.float32)
    b[30, 50, :] = 3.0
    la = optics.ghost_layer(a, gs, threshold=1.0)
    lb = optics.ghost_layer(b, gs, threshold=1.0)
    assert np.allclose(lb, 2.0 * la, rtol=1e-4, atol=1e-6)  # (3-1) = 2 (2-1)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        big = optics.ghost_layer(a * 1e4, gs, threshold=1.0)
    assert np.all(np.isfinite(big))
    d64 = optics.ghost_layer(a.astype(np.float64), gs, threshold=1.0)
    assert d64.dtype == np.float64 and np.allclose(d64, la, atol=1e-5)
    for bad in (dict(threshold=-0.1), dict(threshold=float("nan")), dict(blades=2)):
        with pytest.raises(ValueError):
            optics.ghost_layer(a, gs, **bad)
    with pytest.raises(ValueError):
        optics.ghost_layer(a, [object()])
    with pytest.raises(TypeError):
        optics.ghost_layer(a.astype(np.uint8), gs)
    with pytest.raises(ValueError):
        optics.ghost_layer(a[..., 0], gs)


def test_second_moment_helper_sanity():
    k = dof.aperture_kernel(16.0)
    assert abs(second_moment(k) / (16.0 ** 2 / 8.0) - 1.0) < 0.12
