"""RDR-07 테스트 공용 도우미: 작은 결정론 월드, 장면(바닥 거울·정면 거울·복도), 독립 오라클.

오라클은 FND-06 geom(동결)과 RDR-02 render_ldi(동결)만 쓰고 RDR-07 구현은 참조하지 않는다."""
import math

import numpy as np

from buc.core import contracts as C
from buc.core import geom

W, H = 96, 64
FOV = 60.0
I4 = np.eye(4)


def K_default(w=W, h=H, fov=FOV):
    return geom.intrinsics_from_fov(fov, w, h)


def make_obs(c2w=None, w=W, h=H, fov=FOV):
    c2w = I4 if c2w is None else np.asarray(c2w, np.float64)
    w2c = geom.invert_pose(c2w)
    return C.Obs(K=K_default(w, h, fov), R=w2c[:3, :3], t=w2c[:3, 3], width=w, height=h)


def make_world(layers, obs=None):
    obs = make_obs() if obs is None else obs
    return C.World(obs=obs, layers=tuple(layers), theta_in=C.Theta(), level="L1", read_hash="r" * 8,
                   prov=C.Prov())


def layer(rgba, depth):
    rgba = np.asarray(rgba, np.float32)
    d = np.broadcast_to(np.asarray(depth, np.float32), rgba.shape[:2]).copy()
    return C.LDILayer(rgba=rgba, depth=d)


def const_rgba(rgb, alpha, w=W, h=H):
    rgba = np.empty((h, w, 4), np.float32)
    rgba[..., :3] = np.asarray(rgb, np.float32)
    rgba[..., 3] = alpha
    return rgba


def smooth_tex(w=W, h=H, alpha=1.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rgba = np.empty((h, w, 4), np.float64)
    rgba[..., 0] = 0.5 + 0.25 * np.sin(2 * np.pi * u / 52.0) + 0.1 * (v / h - 0.5)
    rgba[..., 1] = 0.5 + 0.25 * np.cos(2 * np.pi * (u / 44.0 + v / 60.0))
    rgba[..., 2] = 0.2 + 0.6 * (u / w) * (0.6 + 0.4 * np.sin(2 * np.pi * v / 48.0))
    rgba[..., 3] = alpha
    return rgba.astype(np.float32)


def translate(c2w, d):
    T = np.array(c2w, np.float64, copy=True)
    T[:3, 3] = T[:3, 3] + np.asarray(d, np.float64)
    return T


def yaw(c2w, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.array(c2w, np.float64, copy=True)
    T[:3, :3] = T[:3, :3] @ Ry
    return T


def pitch(c2w, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    T = np.array(c2w, np.float64, copy=True)
    T[:3, :3] = T[:3, :3] @ Rx
    return T


def householder(n, d):
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    M = np.eye(4)
    M[:3, :3] -= 2.0 * np.outer(n, n)
    M[:3, 3] = 2.0 * d * n
    return M


def w2c_of(c2w):
    T = geom.invert_pose(np.asarray(c2w, np.float64))
    return T[:3, :3], T[:3, 3]


def project_pt(c2w, K, X, k1=0.0, k2=0.0):
    R, t = w2c_of(c2w)
    uv, z = geom.project(K, R, t, np.asarray(X, np.float64).reshape(1, 3), k1, k2)
    return np.asarray(uv).reshape(2), float(np.asarray(z).reshape(-1)[0])


def pixel_rays(c2w, K, w, h, k1=0.0, k2=0.0):
    """화소 중심 광선: (eye (3,), dirs_world (h,w,3), dirs_cam (h,w,3) z=1 정규화)."""
    K = np.asarray(K, np.float64)
    uv = geom.pixel_grid(h, w).reshape(-1, 2)
    xd = np.stack([(uv[:, 0] - K[0, 2] - K[0, 1] * (uv[:, 1] - K[1, 2]) / K[1, 1]) / K[0, 0],
                   (uv[:, 1] - K[1, 2]) / K[1, 1]], -1)
    xn = geom.undistort(xd, k1, k2) if (k1 or k2) else xd
    dc = np.concatenate([xn, np.ones((xn.shape[0], 1))], -1)
    T = np.asarray(c2w, np.float64)
    dw = dc @ T[:3, :3].T
    return T[:3, 3].copy(), dw.reshape(h, w, 3), dc.reshape(h, w, 3)


def clip_oracle(world, n, d, side=1, eps=1e-4):
    """텍셀 X = unproject_world(obs) 에 대해 side*(n.X-d) <= eps 면 알파 0."""
    obs = world.obs
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    out = []
    for ly in world.layers:
        h, w = ly.rgba.shape[:2]
        uv = geom.pixel_grid(h, w).reshape(-1, 2)
        X = geom.unproject_world(obs.K, obs.R, obs.t, uv, ly.depth.reshape(-1).astype(np.float64), obs.k1, obs.k2)
        s = side * (np.asarray(X).reshape(-1, 3) @ n - d)
        rgba = ly.rgba.copy()
        a = rgba[..., 3].reshape(-1)
        a[s <= eps] = 0.0
        rgba[..., 3] = a.reshape(h, w)
        out.append(C.LDILayer(rgba=rgba, depth=ly.depth.copy()))
    return C.World(obs=obs, layers=tuple(out), theta_in=world.theta_in, level=world.level,
                   read_hash=world.read_hash, prov=world.prov)


def rect_hit(o, dvec, center, n, u, hu, hv, two_sided=False):
    """광선 묶음 vs 사각 거울. 반환 t (N,) (미스 +inf). n,u 단위, v = n x u."""
    n = np.asarray(n, float)
    u = np.asarray(u, float)
    v = np.cross(n, u)
    dn = dvec @ n
    d = float(n @ np.asarray(center, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (d - o @ n) / dn
    P = o + t[:, None] * dvec
    rel = P - np.asarray(center, float)
    ok = np.isfinite(t) & (t > 1e-9)
    ok &= (dn < 0) | (two_sided & (dn != 0))
    if hu is not None:
        ok &= (np.abs(rel @ u) <= hu) & (np.abs(rel @ v) <= hv)
    return np.where(ok, t, np.inf)
