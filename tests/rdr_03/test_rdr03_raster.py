"""RDR-03 래스터 수치: 독립 오라클 일치, n=2 상수 페이지 = RDR-01, 초타원 에지 프로파일, 텍셀 착지·페이지 방향,
기울어진 TSQ 깊이·노멀, edge-on·카메라 뒤, 하드 쿼드 타일링 피복."""
import numpy as np
import pytest
import torch

from buc.render.splat2d import SplatOpts, render as splat_render
from buc.render.tsq import TSQ, TSQOpts, render
from rdr03_helpers import (ALPHA_MAX, ALPHA_MIN, K_of, Scene, make_surfels_from, make_tsq, oracle_bilinear,
                           oracle_render, pose, random_scene, ray_plane_depth, rot_axis, to_np)

W, H = 40, 32
K = K_of(36.0, W, H)
C2W = pose(rot_axis([0.3, 1, 0], 7), [0.05, -0.03, 0.1])


def _cmp(r, o, atol=1e-6):
    for k in ("rgb", "alpha", "normal"):
        np.testing.assert_allclose(to_np(getattr(r, k)), o[k], atol=atol, err_msg=k)
    d = to_np(r.depth)
    m = np.isfinite(o["depth"])
    assert np.array_equal(np.isfinite(d), m)
    np.testing.assert_allclose(d[m], o["depth"][m], rtol=atol)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_independent_oracle_random_scenes(seed):
    sc = random_scene(14, 10 + seed, edge_lo=2.0, edge_hi=20.0)
    Kc = K.copy()
    if seed == 2:
        Kc[0, 1] = 0.7      # skew
        Kc[1, 1] = 33.0
    r = render(make_tsq(sc), c2w=C2W, K=Kc, width=W, height=H, opts=TSQOpts(bg=(0.1, 0.2, 0.3)))
    _cmp(r, oracle_render(sc, C2W, Kc, W, H, bg=(0.1, 0.2, 0.3)))


def test_n2_constant_pages_equals_rdr01_2dgs():
    for seed in range(3):
        sc = random_scene(12, 20 + seed, edge_lo=2.0, edge_hi=2.0, const_pages=True)
        r = render(make_tsq(sc), c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(bg=(0.3, 0.1, 0.2)))
        s = splat_render(make_surfels_from(sc), c2w=C2W, K=K, width=W, height=H,
                         opts=SplatOpts(bg=(0.3, 0.1, 0.2), backend="torch"))
        for k in ("rgb", "alpha", "normal"):
            np.testing.assert_allclose(to_np(getattr(r, k)), to_np(getattr(s, k)), atol=1e-7, err_msg=k)
        d1, d2 = to_np(r.depth), to_np(s.depth)
        assert np.array_equal(np.isfinite(d1), np.isfinite(d2))
        m = np.isfinite(d1)
        np.testing.assert_allclose(d1[m], d2[m], rtol=1e-7)
        # from_surfels 경로도 같다
        r2 = render(TSQ.from_surfels(make_surfels_from(sc)), c2w=C2W, K=K, width=W, height=H,
                    opts=TSQOpts(bg=(0.3, 0.1, 0.2)))
        np.testing.assert_allclose(to_np(r2.rgb), to_np(s.rgb), atol=1e-7)


def _fronto_single(n, z=3.0, s=0.4, o=0.9, pages=None, R=None):
    pages = np.full((1, 8, 8, 3), 0.5) if pages is None else pages
    R = np.eye(3) if R is None else R
    return Scene([[0.0, 0.0, z]], R[None], [[s, s]], [o], pages, [n])


@pytest.mark.parametrize("n", [2.0, 6.0, 16.0])
def test_edge_profile_analytic(n):
    """정면 TSQ: u = (x-cx)/σ_px, σ_px = fx·s/z. α = min(.999, o·exp(-min(½(|u|^n+|v|^n), d²)))."""
    Kc = K_of(40.0, 64, 48, cx=32.0, cy=24.0)
    z, s, o = 3.0, 0.6, 0.9
    sc = _fronto_single(n, z, s, o)
    r = render(make_tsq(sc), c2w=np.eye(4), K=Kc, width=64, height=48)
    sig = 40.0 * s / z
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float64)
    u, v = (xx - 32.0) / sig, (yy - 24.0) / sig
    E = 0.5 * (np.abs(u) ** n + np.abs(v) ** n)
    a = np.minimum(ALPHA_MAX, o * np.exp(-np.minimum(E, (xx - 32) ** 2 + (yy - 24) ** 2)))
    a = np.where(a >= ALPHA_MIN, a, 0.0)
    np.testing.assert_allclose(to_np(r.alpha), a, atol=1e-9)
    if n == 16.0:
        al = to_np(r.alpha)
        inside = (np.abs(u) <= 0.8) & (np.abs(v) <= 0.8)
        outside = (np.abs(u) >= 1.3) | (np.abs(v) >= 1.3)
        assert al[inside].min() > 0.98 * o
        assert al[outside].max() == 0.0


def test_texel_centers_land_exactly_and_page_orientation():
    """σ_px = 8 → 텍셀 간격 2 px. 텍셀 (i, j) 중심 = 화소 (cx + 2i - 7, cy + 2j - 7). 열 i <-> x, 행 j <-> y."""
    Kc = K_of(40.0, 48, 40, cx=24.0, cy=20.0)
    z = 3.0
    s = 8.0 * z / 40.0
    rng = np.random.default_rng(0)
    pages = rng.uniform(0.0, 1.0, (1, 8, 8, 3))
    sc = _fronto_single(32.0, z, s, 0.999, pages)
    r = render(make_tsq(sc), c2w=np.eye(4), K=Kc, width=48, height=40)
    rgb, al = to_np(r.rgb), to_np(r.alpha)
    for j in range(8):
        for i in range(8):
            x, y = 24 + 2 * i - 7, 20 + 2 * j - 7
            np.testing.assert_allclose(rgb[y, x] / al[y, x], pages[0, j, i], atol=1e-9)
    # 사이 화소는 바이리니어 (텍셀 0,0 과 1,0 사이 중점)
    np.testing.assert_allclose(rgb[13, 18] / al[13, 18], 0.5 * (pages[0, 0, 0] + pages[0, 0, 1]), atol=1e-9)
    # z축 +90° 회전: t_u = 카메라 y, t_v = 카메라 -x  →  텍셀 (i, j) 는 화소 (cx - (2j-7), cy + (2i-7))
    sc2 = _fronto_single(32.0, z, s, 0.999, pages, R=rot_axis([0, 0, 1], 90))
    r2 = render(make_tsq(sc2), c2w=np.eye(4), K=Kc, width=48, height=40)
    rgb2, al2 = to_np(r2.rgb), to_np(r2.alpha)
    for j in range(8):
        for i in range(8):
            x, y = 24 - (2 * j - 7), 20 + (2 * i - 7)
            np.testing.assert_allclose(rgb2[y, x] / al2[y, x], pages[0, j, i], atol=1e-9)


def test_texture_bilinear_everywhere_and_clamp():
    Kc = K_of(40.0, 64, 48, cx=32.0, cy=24.0)
    z, s = 3.0, 0.45
    rng = np.random.default_rng(1)
    pages = rng.uniform(0.0, 1.0, (1, 8, 8, 3))
    sc = _fronto_single(4.0, z, s, 0.95, pages)
    r = render(make_tsq(sc), c2w=np.eye(4), K=Kc, width=64, height=48, opts=TSQOpts(bg=(0.0, 0.0, 0.0)))
    al, rgb = to_np(r.alpha), to_np(r.rgb)
    sig = 40.0 * s / z
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float64)
    m = al > 0
    col = oracle_bilinear(pages[0], (xx - 32) / sig, (yy - 24) / sig)
    np.testing.assert_allclose(rgb[m] / al[m][:, None], col[m], atol=1e-9)
    # |u| > 1 영역(클램프) 이 실제로 그려졌다
    assert (m & (np.abs(xx - 32) / sig > 1.2)).any()


def test_tilted_tsq_depth_normal_and_texture():
    R = rot_axis([0, 1, 0], 35) @ rot_axis([1, 0, 0], -20)
    mean = np.array([0.1, -0.05, 3.0])
    rng = np.random.default_rng(2)
    pages = rng.uniform(0.2, 1.0, (1, 8, 8, 3))
    sc = Scene([mean], R[None], [[0.5, 0.35]], [0.95], pages, [10.0])
    c2w = np.eye(4)
    r = render(make_tsq(sc), c2w=c2w, K=K, width=W, height=H)
    al, d, nrm = to_np(r.alpha), to_np(r.depth), to_np(r.normal)
    m = al > 0.5
    assert m.sum() > 50
    ref = ray_plane_depth(K, c2w, mean, R[:, 2], W, H)
    np.testing.assert_allclose(d[m], ref[m], rtol=1e-9)
    n = R[:, 2] * (-1.0 if R[:, 2] @ mean > 0 else 1.0)    # 카메라 쪽
    np.testing.assert_allclose(nrm[m], np.broadcast_to(n, nrm[m].shape), atol=1e-9)
    o = oracle_render(sc, c2w, K, W, H)
    np.testing.assert_allclose(to_np(r.rgb), o["rgb"], atol=1e-9)


def test_edge_on_and_behind_camera_are_finite():
    Kc = K_of(36.0, W, H, cx=20.0, cy=16.0)
    # 평면 x=0 (노멀 = 카메라 x): 열 x=cx 화소의 광선은 평면 안에 있다 → 저역 항만
    R = rot_axis([0, 1, 0], 90)
    sc = Scene([[0.0, 0.0, 3.0]], R[None], [[0.3, 0.3]], [0.8], np.full((1, 8, 8, 3), 0.7), [8.0])
    t = make_tsq(sc).requires_grad_(True)
    r = render(t, c2w=np.eye(4), K=Kc, width=W, height=H)
    for k in ("rgb", "alpha", "normal"):
        assert torch.isfinite(getattr(r, k)).all(), k
    assert not torch.isnan(r.depth).any()
    al = to_np(r.alpha)
    np.testing.assert_allclose(al[16, 20], min(ALPHA_MAX, 0.8), atol=1e-9)   # d² = 0
    np.testing.assert_allclose(al[16, 21], 0.8 * np.exp(-1.0), atol=1e-9)     # 저역 d² = 1
    (r.rgb.sum() + r.alpha.sum()).backward()
    for p in t.params():
        assert torch.isfinite(p.grad).all()
    # 카메라 뒤 · far 밖
    sc2 = Scene([[0.0, 0.0, -2.0], [0.0, 0.0, 50.0]], np.repeat(np.eye(3)[None], 2, 0), [[0.5, 0.5]] * 2,
                [0.9, 0.9], np.full((2, 8, 8, 3), 0.5), [4.0, 4.0])
    r2 = render(make_tsq(sc2), c2w=np.eye(4), K=Kc, width=W, height=H, opts=TSQOpts(far=20.0))
    assert float(r2.alpha.max()) == 0.0 and torch.isinf(r2.depth).all()


def _tiling_alpha(n):
    s, z = 0.1, 3.0
    c = [[2 * s * i, 2 * s * j, z] for i in range(-3, 4) for j in range(-3, 4)]
    N = len(c)
    sc = Scene(c, np.repeat(np.eye(3)[None], N, 0), np.full((N, 2), s), np.full(N, 0.995),
               np.full((N, 8, 8, 3), 0.5), np.full(N, n))
    Kc = K_of(150.0, 128, 128, cx=64.0, cy=64.0)      # 쿼드 이음매가 화소 중심에 온다
    r = render(make_tsq(sc), c2w=np.eye(4), K=Kc, width=128, height=128)
    return to_np(r.alpha)[39:90, 39:90]


def test_hard_quad_tiling_covers_better_than_soft_surfels():
    a2, a16 = _tiling_alpha(2.0), _tiling_alpha(16.0)
    f2, f16 = (a2 > 0.98).mean(), (a16 > 0.98).mean()
    assert f16 >= 0.65, f16
    assert f2 <= 0.15, f2
    assert a16.mean() > a2.mean() + 0.03
    assert a16.min() > 0.8           # 이음매 (|u|=1) 에서도 구멍 없음
