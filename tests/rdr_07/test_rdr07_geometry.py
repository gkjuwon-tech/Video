"""RDR-07 기하: 반사 행렬, 가상 카메라(geom.project 대조), 화소 중심 거울 적중, 평면 클립 오라클."""
import numpy as np
import pytest

from buc.core import geom
from buc.render.mirror import Mirror, clip_world, ray_mirror_hits, reflection_matrix, virtual_camera
from rdr07_helpers import (H, I4, W, K_default, clip_oracle, householder, layer, make_obs, make_world, pitch,
                           pixel_rays, rect_hit, smooth_tex, translate, yaw)


@pytest.mark.parametrize("n,d", [((0, 0, -1), -3.0), ((0, -1, 0), -1.0), ((0.3, -0.8, 0.2), 0.7)])
def test_reflection_matrix_properties(n, d):
    M = reflection_matrix(n, d)
    assert M.shape == (4, 4) and M.dtype == np.float64
    np.testing.assert_allclose(M, householder(n, d), atol=1e-12)
    np.testing.assert_allclose(M @ M, np.eye(4), atol=1e-12)
    assert np.linalg.det(M[:3, :3]) == pytest.approx(-1.0)
    nn = np.asarray(n, float) / np.linalg.norm(n)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    on = X - ((X @ nn) - d)[:, None] * nn                              # 평면 위 점은 고정
    on_h = np.c_[on, np.ones(20)]
    np.testing.assert_allclose((on_h @ M.T)[:, :3], on, atol=1e-12)
    Xh = np.c_[X, np.ones(20)]
    Xr = (Xh @ M.T)[:, :3]
    np.testing.assert_allclose(Xr @ nn - d, -(X @ nn - d), atol=1e-12)   # 부호 있는 거리 반전


CAMS = [
    I4,
    translate(yaw(I4, 12.0), (0.2, -0.1, 0.3)),
    translate(pitch(yaw(I4, -20.0), 15.0), (-0.3, 0.2, -0.1)),
]
MIRRORS = [
    Mirror(center=(0, 0, 3), normal=(0, 0, -1)),
    Mirror(center=(0, 1.0, 0), normal=(0, -1, 0)),
    Mirror(center=(1.0, 0.2, 2.5), normal=(-1.0, 0.1, -0.4)),
]


@pytest.mark.parametrize("ci", range(len(CAMS)))
@pytest.mark.parametrize("mi", range(len(MIRRORS)))
@pytest.mark.parametrize("k1", [0.0, -0.08])
def test_virtual_camera_flip_equals_projection_of_reflected_point(ci, mi, k1):
    c2w, m = CAMS[ci], MIRRORS[mi]
    K = K_default()
    K[0, 1] = 0.7                                                       # 스큐도 뒤집혀야 한다
    c2w_v, K_v = virtual_camera(c2w, K, W, m)
    assert c2w_v.shape == (4, 4) and K_v.shape == (3, 3)
    assert geom.is_rotation(c2w_v[:3, :3], atol=1e-9)
    np.testing.assert_allclose(c2w_v, householder(m.normal, m.d) @ c2w @ np.diag([-1.0, 1, 1, 1]), atol=1e-12)
    Kexp = K.copy()
    Kexp[0, 1] = -K[0, 1]
    Kexp[0, 2] = W - 1 - K[0, 2]
    np.testing.assert_allclose(K_v, Kexp, atol=1e-12)
    rng = np.random.default_rng(ci * 10 + mi)
    R, t = np.linalg.inv(c2w)[:3, :3], np.linalg.inv(c2w)[:3, 3]
    Xc = np.c_[rng.uniform(-1, 1, 50), rng.uniform(-0.7, 0.7, 50), rng.uniform(1.5, 6, 50)]
    Xw = (Xc - t) @ R                                                   # 원 카메라 앞 점(= 거울상 위치)
    M = householder(m.normal, m.d)
    X = (np.c_[Xw, np.ones(50)] @ M.T)[:, :3]                           # 실제 점 X, 거울상 = M X = Xw
    Rv, tv = np.linalg.inv(c2w_v)[:3, :3], np.linalg.inv(c2w_v)[:3, 3]
    uv_v, z_v = geom.project(K_v, Rv, tv, X, k1, 0.0)
    uv, z = geom.project(K, R, t, Xw, k1, 0.0)
    np.testing.assert_allclose(z_v, z, atol=1e-9)                       # z-깊이 보존
    np.testing.assert_allclose(W - 1 - uv_v[:, 0], uv[:, 0], atol=1e-8)
    np.testing.assert_allclose(uv_v[:, 1], uv[:, 1], atol=1e-8)


def test_virtual_camera_rejects_bad_input():
    m = MIRRORS[0]
    with pytest.raises(ValueError):
        virtual_camera(np.eye(3), K_default(), W, m)
    with pytest.raises(ValueError):
        virtual_camera(I4, K_default(), 0, m)


def _hits_oracle(mirrors, c2w, K, w, h, k1=0.0):
    eye, dw, dc = pixel_rays(c2w, K, w, h, k1)
    o = np.broadcast_to(eye, (h * w, 3))
    dv = dw.reshape(-1, 3)
    best_t = np.full(h * w, np.inf)
    best_i = np.full(h * w, -1)
    for i, m in enumerate(mirrors):
        hs = m.half_size
        t = rect_hit(o, dv, m.center, m.normal, m.u_axis if hs is not None else np.zeros(3),
                     None if hs is None else hs[0], None if hs is None else hs[1], m.two_sided)
        t = np.where(t * dc.reshape(-1, 3)[:, 2] >= 1e-3, t, np.inf)   # z-깊이 = t (dc z=1)
        upd = t < best_t
        best_t = np.where(upd, t, best_t)
        best_i = np.where(upd, i, best_i)
    return best_i.reshape(h, w), best_t.reshape(h, w)


@pytest.mark.parametrize("k1", [0.0, -0.06])
def test_ray_mirror_hits_matches_ray_oracle(k1):
    ms = [
        Mirror(center=(0, 0, 3), normal=(0, 0, -1), u_axis=(1, 0, 0), half_size=(0.8, 0.5)),
        Mirror(center=(0.6, 0.2, 2.0), normal=(-0.3, 0, -1), u_axis=(1, 0, -0.3), half_size=(0.3, 0.3)),
        Mirror(center=(0, 1.0, 0), normal=(0, -1, 0)),                  # 무한 바닥
        Mirror(center=(-0.5, 0, 1.5), normal=(0, 0, 1), u_axis=(1, 0, 0), half_size=(0.4, 0.4)),  # 뒷면
    ]
    c2w = translate(yaw(I4, 5.0), (0.05, -0.1, 0.0))
    mid, z = ray_mirror_hits(ms, c2w=c2w, K=K_default(), width=W, height=H, k1=k1)
    assert mid.shape == (H, W) and mid.dtype == np.int16
    assert z.shape == (H, W) and z.dtype == np.float64
    oi, ot = _hits_oracle(ms, c2w, K_default(), W, H, k1)
    agree = mid == oi
    assert agree.mean() >= 0.995, agree.mean()
    assert {0, 1, 2} <= set(np.unique(mid)) and 3 not in set(np.unique(mid))
    hit = agree & (oi >= 0)
    np.testing.assert_allclose(z[hit], ot[hit], rtol=1e-6)
    assert np.isinf(z[mid < 0]).all()


def test_ray_mirror_hits_two_sided_back_face():
    back = Mirror(center=(0, 0, 3), normal=(0, 0, 1), u_axis=(1, 0, 0), half_size=(0.5, 0.5))
    both = Mirror(center=(0, 0, 3), normal=(0, 0, 1), u_axis=(1, 0, 0), half_size=(0.5, 0.5), two_sided=True)
    m1, _ = ray_mirror_hits([back], c2w=I4, K=K_default(), width=W, height=H)
    m2, z2 = ray_mirror_hits([both], c2w=I4, K=K_default(), width=W, height=H)
    assert (m1 == -1).all()
    assert (m2 == 0).sum() > 100
    np.testing.assert_allclose(z2[m2 == 0], 3.0, atol=1e-9)


@pytest.mark.parametrize("side", [1, -1])
def test_clip_world_matches_oracle(side):
    obs = make_obs(translate(yaw(I4, 10.0), (0.1, 0.0, 0.0)))
    v, u = np.mgrid[0:H, 0:W]
    depth = (2.0 + 3.0 * u / W + 0.5 * v / H).astype(np.float32)        # 경사 깊이 -> 평면이 텍셀을 가른다
    w = make_world([layer(smooth_tex(), depth), layer(smooth_tex(alpha=0.6), 6.0)], obs)
    n, d = np.array([0.2, 0.1, -1.0]), -3.4
    n_unit = n / np.linalg.norm(n)
    d_unit = d / np.linalg.norm(n)
    # 계약: 평면은 n.X = d, 내부에서 (n, d) 를 |n| 으로 나눠 단위화한 뒤 eps 를 미터로 적용한다
    cw = clip_world(w, n, d, side=side, eps=1e-4)
    ref = clip_oracle(w, n_unit, d_unit, side=side, eps=1e-4)
    cw_unit = clip_world(w, n_unit, d_unit, side=side, eps=1e-4)
    for a, b in zip(cw.layers, cw_unit.layers):
        np.testing.assert_array_equal(a.rgba, b.rgba)
    assert cw is not w and len(cw.layers) == 2
    for a, b, o in zip(cw.layers, ref.layers, w.layers):
        np.testing.assert_array_equal(a.rgba[..., 3], b.rgba[..., 3])
        np.testing.assert_array_equal(a.rgba[..., :3], o.rgba[..., :3])
        np.testing.assert_array_equal(a.depth, o.depth)
    a0 = cw.layers[0].rgba[..., 3]
    assert 0.2 < (a0 == 0).mean() < 0.8                                  # 실제로 가른다
    assert np.array_equal(w.layers[0].rgba[..., 3], smooth_tex()[..., 3])  # 입력 불변
