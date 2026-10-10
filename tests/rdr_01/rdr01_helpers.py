"""RDR-01 테스트 공용 도우미: 카메라, 쿼터니언 오라클(구현 독립), surfel 생성, 광선-평면 깊이 오라클.

오라클은 numpy와 FND-06 geom(동결·검증된 모듈)만 쓰고 RDR-01 구현은 참조하지 않는다."""
import math

import numpy as np
import torch

from buc.core import geom


def K_of(fx, w, h, fy=None, cx=None, cy=None):
    fy = fx if fy is None else fy
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rot_axis(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    return geom.rodrigues(a * math.radians(deg))


def quat_from_R(R):
    """회전행렬 -> 단위 쿼터니언 (w, x, y, z). R(q)의 열 = (t_u, t_v, n)."""
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return np.array(q)


def R_from_quat(q):
    q = np.asarray(q, np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def logit(p):
    p = np.asarray(p, np.float64)
    return np.log(p / (1 - p))


def make_surfels(means, rots, scales, opac, colors, dtype=torch.float32):
    """rots: (N,3,3) 회전행렬(열 = t_u, t_v, n), scales: (N,2) 미터, opac: (N,) (0,1)."""
    from buc.render.splat2d import Surfels
    means = np.asarray(means, np.float64).reshape(-1, 3)
    n = means.shape[0]
    rots = np.asarray(rots, np.float64).reshape(n, 3, 3)
    quats = np.stack([quat_from_R(r) for r in rots])
    scales = np.asarray(scales, np.float64).reshape(n, 2)
    opac = np.asarray(opac, np.float64).reshape(n)
    colors = np.asarray(colors, np.float64).reshape(n, 3)
    t = lambda a: torch.tensor(a, dtype=dtype)  # noqa: E731
    return Surfels(means=t(means), quats=t(quats), log_scales=t(np.log(scales)), opacity_logits=t(logit(opac)),
                   colors=t(colors))


def fronto(c2w, n):
    """카메라와 평행한 surfel 회전(접선 = 카메라 x, y축) n개."""
    return np.repeat(np.asarray(c2w, np.float64)[None, :3, :3], n, axis=0)


def ray_plane_depth(K, c2w, point_w, normal_w, w, h):
    """각 화소 광선과 평면(point, normal, 세계)의 교점의 카메라 z-깊이 (h, w). 교점이 뒤면 nan."""
    w2c = geom.invert_pose(np.asarray(c2w, np.float64))
    p = w2c[:3, :3] @ point_w + w2c[:3, 3]
    n = w2c[:3, :3] @ normal_w
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    lam = (n @ p) / (d @ n)
    lam[lam <= 0] = np.nan
    return lam


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float64)


def alpha_moments(alpha):
    """알파 맵의 질량, 무게중심 (u, v), 2x2 공분산."""
    a = np.asarray(alpha, np.float64)
    h, w = a.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    m = a.sum()
    cu, cv = (a * u).sum() / m, (a * v).sum() / m
    du, dv = u - cu, v - cv
    cov = np.array([[(a * du * du).sum(), (a * du * dv).sum()], [(a * du * dv).sum(), (a * dv * dv).sum()]]) / m
    return m, np.array([cu, cv]), cov


def pose(R=None, t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T
