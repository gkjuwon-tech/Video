"""RDR-01 래스터 수치: 가우시안 프로파일·질량·폭, 저역 필터, 중심 착지(geom.project), 비등방 축,
기울어진 surfel의 깊이(광선-평면 교점)·노멀, 양면, over 합성, 컬링, 속도."""
import math
import time

import numpy as np
import torch

from buc.core import geom
from buc.render.splat2d import SplatOpts, render
from rdr01_helpers import (K_of, alpha_moments, fronto, make_surfels, pose, ray_plane_depth, rot_axis, to_np)

W, H = 48, 40
FX = 40.0
K = K_of(FX, W, H)          # cx = 23.5, cy = 19.5
I4 = np.eye(4)


def _at_pixel(u, v, z, K=K):
    """카메라 항등일 때 화소 (u, v) 중심에 투영되는 깊이 z의 점."""
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def _single(sig_px=4.0, o=0.5, z=3.0, u=24, v=20, rot=None, sv_px=None, dtype=torch.float32):
    s = sig_px * z / FX
    sv = s if sv_px is None else sv_px * z / FX
    rot = np.eye(3) if rot is None else rot
    return make_surfels(_at_pixel(u, v, z), rot[None], [[s, sv]], [o], [[0.7, 0.4, 0.2]], dtype=dtype)


def _grid():
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    return u, v


def test_gaussian_profile_exact():
    o, sig = 0.5, 4.0
    r = render(_single(sig, o), c2w=I4, K=K, width=W, height=H)
    a = to_np(r.alpha)
    assert abs(a[20, 24] - o) < 1e-5
    u, v = _grid()
    exp = o * np.exp(-((u - 24) ** 2 + (v - 20) ** 2) / (2 * sig * sig))
    m = exp >= 0.02
    assert m.sum() > 100
    assert np.abs(a[m] - exp[m]).max() < 1e-5
    # 단일 surfel 이면 rgb = alpha * color
    assert np.allclose(to_np(r.rgb), a[..., None] * np.array([0.7, 0.4, 0.2]), atol=1e-6)


def test_alpha_mass_and_width():
    o, sig = 0.5, 4.0
    a = to_np(render(_single(sig, o), c2w=I4, K=K, width=W, height=H).alpha)
    mass = a.sum()
    ideal = o * 2 * math.pi * sig * sig
    assert 0.95 * ideal <= mass <= 1.005 * ideal, (mass, ideal)
    sig_est = math.sqrt(mass / (2 * math.pi * a.max()))
    assert abs(sig_est / sig - 1) < 0.04
    # 폭은 fx·s/z 에 비례: 깊이 두 배 + 크기 두 배 -> 같은 화면 폭
    a2 = to_np(render(_single(sig, o, z=6.0), c2w=I4, K=K, width=W, height=H).alpha)
    assert np.abs(a2 - a).max() < 1e-4


def test_screen_lowpass_filter():
    """아주 작은 surfel 은 화면 저역 필터 exp(-d²) (σ = 1/√2 px)가 지배한다."""
    o = 0.8
    a = to_np(render(_single(0.05, o), c2w=I4, K=K, width=W, height=H).alpha)
    assert abs(a[20, 24] - o) < 1e-5
    for dv, du in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        assert abs(a[20 + dv, 24 + du] - o * math.exp(-1.0)) < 1e-4
    assert abs(a[21, 25] - o * math.exp(-2.0)) < 1e-4
    assert a[20, 27] == 0.0          # o·e^-9 < 1/255 -> 버림


def test_center_lands_on_geom_project():
    """정면 surfel(카메라 평행) 알파 무게중심 = FND-06 geom.project, 비중심 주점·fx≠fy·회전 카메라."""
    Kc = K_of(45.0, W, H, fy=42.0, cx=22.1, cy=17.6)
    c2w = pose(rot_axis([0.3, 1.0, 0.2], 20), [0.4, -0.2, 0.1])
    w2c = geom.invert_pose(c2w)
    for pc in ([0.13, -0.07, 3.0], [-0.21, 0.11, 2.5], [0.02, 0.05, 4.0]):
        pc = np.array(pc)
        X = c2w[:3, :3] @ pc + c2w[:3, 3]
        s = 3.0 * pc[2] / 45.0
        sur = make_surfels(X, fronto(c2w, 1), [[s, s]], [0.6], [[0.5, 0.5, 0.5]])
        a = to_np(render(sur, c2w=c2w, K=Kc, width=W, height=H).alpha)
        _, cen, _ = alpha_moments(a)
        uv, z = geom.project(Kc, w2c[:3, :3], w2c[:3, 3], X[None])
        assert z[0] > 0
        assert np.abs(cen - uv[0]).max() < 0.03, (cen, uv[0])


def test_anisotropic_axes_follow_quaternion():
    """접선 t_u = R(q)[:,0] 이 화면 장축. 카메라 z 축 둘레 +30° 회전 -> 화면각 +30° (y 아래)."""
    rot = rot_axis([0, 0, 1], 30)
    a = to_np(render(_single(4.0, 0.9, rot=rot, sv_px=1.5), c2w=I4, K=K, width=W, height=H).alpha)
    _, cen, cov = alpha_moments(a)
    assert np.abs(cen - [24, 20]).max() < 0.02
    ev, V = np.linalg.eigh(cov)
    major = V[:, 1]
    ang = math.degrees(math.atan2(major[1], major[0])) % 180.0
    assert abs(ang - 30.0) < 1.0, ang
    ratio = math.sqrt(ev[1] / ev[0])
    assert abs(ratio / (4.0 / 1.5) - 1) < 0.05, ratio


def test_tilted_surfel_depth_is_ray_plane_intersection_and_normal():
    rot = rot_axis([0, 1, 0], 40)
    p = np.array([0.1, -0.05, 3.0])
    sur = make_surfels(p, rot[None], [[0.6, 0.6]], [0.9], [[0.3, 0.6, 0.9]])
    for c2w in (I4, pose(rot_axis([1, 0, 0], 8), [0.05, 0.1, -0.2])):
        r = render(sur, c2w=c2w, K=K, width=W, height=H)
        a, d, n = to_np(r.alpha), to_np(r.depth), to_np(r.normal)
        exp = ray_plane_depth(K, c2w, p, rot[:, 2], W, H)
        m = a > 0.3
        assert m.sum() > 200
        assert np.abs(d[m] / exp[m] - 1).max() < 1e-4
        # 깊이가 화소마다 달라야 한다(중심 깊이 하나가 아님)
        assert d[m].max() - d[m].min() > 0.3
        nw = rot[:, 2]
        o = c2w[:3, 3]
        if nw @ (o - p) < 0:
            nw = -nw
        assert np.abs(n[m] - nw).max() < 1e-4


def test_two_sided():
    rot = rot_axis([0, 1, 0], 35)
    flip = rot @ np.diag([1.0, -1.0, -1.0])     # 같은 평면, 노멀 반대
    p = np.array([0.0, 0.05, 3.0])
    a = render(make_surfels(p, rot[None], [[0.3, 0.2]], [0.7], [[1, 1, 1]]), c2w=I4, K=K, width=W, height=H)
    b = render(make_surfels(p, flip[None], [[0.3, 0.2]], [0.7], [[1, 1, 1]]), c2w=I4, K=K, width=W, height=H)
    assert float(a.alpha.max()) > 0.6
    assert torch.allclose(a.alpha, b.alpha, atol=1e-6)
    assert torch.allclose(a.normal, b.normal, atol=1e-5)


def test_over_compositing_formula():
    o1, o2 = 0.6, 0.7
    c1, c2 = np.array([0.9, 0.1, 0.2]), np.array([0.1, 0.8, 0.3])
    bg = np.array([0.05, 0.1, 0.15])
    sig = 20.0
    means = [_at_pixel(24, 20, 2.0), _at_pixel(24, 20, 3.0)]
    scales = [[sig * 2.0 / FX] * 2, [sig * 3.0 / FX] * 2]
    for order in ((0, 1), (1, 0)):
        sur = make_surfels(np.array(means)[list(order)], fronto(I4, 2), np.array(scales)[list(order)],
                           np.array([o1, o2])[list(order)], np.array([c1, c2])[list(order)])
        r = render(sur, c2w=I4, K=K, width=W, height=H, opts=SplatOpts(bg=tuple(bg)))
        rgb, a, d = to_np(r.rgb), to_np(r.alpha), to_np(r.depth)
        u, v = _grid()
        g = np.exp(-((u - 24) ** 2 + (v - 20) ** 2) / (2 * sig * sig))
        a1, a2 = o1 * g, o2 * g
        win = (slice(16, 25), slice(20, 29))
        ea = 1 - (1 - a1) * (1 - a2)
        erg = a1[..., None] * c1 + (a2 * (1 - a1))[..., None] * c2 + ((1 - a1) * (1 - a2))[..., None] * bg
        ed = (a1 * 2.0 + a2 * (1 - a1) * 3.0) / ea
        assert np.abs(a[win] - ea[win]).max() < 1e-5
        assert np.abs(rgb[win] - erg[win]).max() < 1e-5
        assert np.abs(d[win] - ed[win]).max() < 1e-4
        assert abs(a[20, 24] - 0.88) < 1e-5


def test_opacity_clamp():
    sur = make_surfels(_at_pixel(24, 20, 3.0), fronto(I4, 1), [[0.5, 0.5]], [0.99999], [[1, 1, 1]])
    a = to_np(render(sur, c2w=I4, K=K, width=W, height=H).alpha)
    assert abs(a.max() - 0.999) < 1e-6


def test_culling_near_far_behind():
    def one(z, **kw):
        sur = make_surfels([0, 0, z], fronto(I4, 1), [[0.5, 0.5]], [0.8], [[1, 1, 1]])
        return render(sur, c2w=I4, K=K, width=W, height=H, opts=SplatOpts(**kw))
    for z, kw in ((-2.0, {}), (0.005, {}), (20.0, dict(far=10.0))):
        r = one(z, **kw)
        assert float(r.alpha.max()) == 0.0, (z, kw)
        assert bool(torch.isinf(r.depth).all())
    assert float(one(20.0).alpha.max()) > 0.5
    assert float(one(0.05, near=0.01).alpha.max()) > 0.5


def test_camera_motion_parallax():
    """카메라가 +x 로 b 만큼 가면 깊이 z 의 surfel 은 화면에서 fx·b/z 만큼 왼쪽으로."""
    sur = make_surfels([_at_pixel(24, 11, 2.0), _at_pixel(24, 29, 4.0)], fronto(I4, 2),
                       [[0.1, 0.1], [0.2, 0.2]], [0.5, 0.5], [[1, 0, 0], [0, 1, 0]])
    b = 0.1
    r = render(sur, c2w=pose(None, [b, 0, 0]), K=K, width=W, height=H)
    rgb = to_np(r.rgb)
    _, cr, _ = alpha_moments(rgb[..., 0])
    _, cg, _ = alpha_moments(rgb[..., 1])
    assert abs(cr[0] - (24 - FX * b / 2.0)) < 0.03 and abs(cr[1] - 11) < 0.03
    assert abs(cg[0] - (24 - FX * b / 4.0)) < 0.03 and abs(cg[1] - 29) < 0.03


def test_speed_cpu_fwd_bwd():
    rng = np.random.default_rng(7)
    n = 1000
    w, h = 64, 48
    Kp = K_of(50.0, w, h)
    means = np.c_[rng.uniform(-1, 1, n), rng.uniform(-0.75, 0.75, n), rng.uniform(2.5, 5.0, n)]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, 90)) for _ in range(n)])
    sur = make_surfels(means, rots, rng.uniform(0.02, 0.08, (n, 2)), rng.uniform(0.2, 0.9, n),
                       rng.uniform(0, 1, (n, 3)))
    sur.requires_grad_(True)
    t0 = time.perf_counter()
    r = render(sur, c2w=I4, K=Kp, width=w, height=h)
    (r.rgb.sum() + r.alpha.sum()).backward()
    dt = time.perf_counter() - t0
    assert dt < 8.0, dt
    assert float(r.alpha.mean()) > 0.15
    for p in sur.params():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
