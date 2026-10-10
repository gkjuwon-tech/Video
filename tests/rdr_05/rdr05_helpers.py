"""RDR-05 테스트 공용 도우미: 작은 결정론 LDI 월드, 카메라 포즈, 기하 오라클.

오라클은 FND-06 geom(동결·검증된 모듈)과 numpy 만 쓰고 RDR-05 구현은 참조하지 않는다."""
import math

import numpy as np

from buc.core import contracts as C
from buc.core import geom

W, H = 96, 64
FOV = 60.0
I4 = np.eye(4)


def K_default(w=W, h=H, fov=FOV):
    return geom.intrinsics_from_fov(fov, w, h)


def FX(w=W, fov=FOV):
    return (w / 2.0) / math.tan(math.radians(fov) / 2.0)


def make_obs(c2w=None, w=W, h=H, fov=FOV, k1=0.0, k2=0.0, mask=None):
    c2w = np.eye(4) if c2w is None else np.asarray(c2w, np.float64)
    w2c = geom.invert_pose(c2w)
    return C.Obs(K=K_default(w, h, fov), R=w2c[:3, :3], t=w2c[:3, 3], width=w, height=h, k1=k1, k2=k2, mask=mask)


def make_world(layers, obs=None):
    obs = make_obs() if obs is None else obs
    return C.World(obs=obs, layers=tuple(layers), theta_in=C.Theta(), level="L1", read_hash="r" * 8, prov=C.Prov())


def layer(rgba, depth):
    """깊이가 스칼라면 평면 카드, 배열이면 픽셀별 깊이."""
    rgba = np.asarray(rgba, np.float32)
    d = np.broadcast_to(np.asarray(depth, np.float32), rgba.shape[:2]).copy()
    return C.LDILayer(rgba=rgba, depth=d)


def const_layer(rgb, alpha, depth, w=W, h=H):
    rgba = np.empty((h, w, 4), np.float32)
    rgba[..., :3] = np.asarray(rgb, np.float32)
    rgba[..., 3] = alpha
    return layer(rgba, depth)


def rand_tex(seed, w=W, h=H, alpha=1.0):
    """결정론 잡음 텍스처. 값 [0.1, 0.9]."""
    rng = np.random.default_rng(seed)
    rgba = np.empty((h, w, 4), np.float32)
    rgba[..., :3] = rng.uniform(0.1, 0.9, (h, w, 3))
    rgba[..., 3] = alpha
    return rgba


def smooth_tex(w=W, h=H, alpha=1.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rgba = np.empty((h, w, 4), np.float64)
    rgba[..., 0] = 0.5 + 0.25 * np.sin(2 * np.pi * u / 52.0)
    rgba[..., 1] = 0.5 + 0.25 * np.cos(2 * np.pi * (u / 44.0 + v / 60.0))
    rgba[..., 2] = 0.2 + 0.6 * (u / w)
    rgba[..., 3] = alpha
    return rgba.astype(np.float32)


def translate(c2w, d):
    T = np.array(c2w, np.float64, copy=True)
    T[:3, 3] = T[:3, 3] + np.asarray(d, np.float64)
    return T


def yaw(c2w, deg):
    """카메라 자신의 y축(아래) 둘레 회전. 위치 고정."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.array(c2w, np.float64, copy=True)
    T[:3, :3] = T[:3, :3] @ Ry
    return T


def plane_layer(n_face, dist, rgba=None, obs=None):
    """관측 카메라 좌표의 평면 n·X = d (n_face: 카메라를 향하는 법선, n_z<0). 중심 광선 위 거리 dist 지점을 지난다.
    반환: LDILayer(픽셀별 z-깊이)"""
    obs = make_obs() if obs is None else obs
    n = np.asarray(n_face, np.float64)
    n = n / np.linalg.norm(n)
    d = float(n @ np.array([0.0, 0.0, dist]))
    uv = geom.pixel_grid(obs.height, obs.width)
    r = geom.unproject(obs.K, uv, 1.0, obs.k1, obs.k2)              # z = 1 광선
    z = d / (r @ n)
    assert np.all(z > 0)
    rgba = smooth_tex(obs.width, obs.height) if rgba is None else rgba
    return layer(rgba, z), n


def rot_o2t(obs, c2w_t):
    """관측 카메라 좌표 -> 목표 카메라 좌표 회전 R_t R_oᵀ (R_t = c2w_tᵀ 의 회전부)."""
    Rt = np.asarray(c2w_t, np.float64)[:3, :3].T
    return Rt @ np.asarray(obs.R, np.float64).T


def atom_world_oracle(obs, depth_map, layer_idx, v, u):
    """원자(층, v, u)의 세계 점 -- geom.unproject 로 관측 카메라 점을 구한 뒤 R^T (X_c - t)."""
    Xc = geom.unproject(obs.K, np.array([float(u), float(v)]), float(depth_map[v, u]), obs.k1, obs.k2)
    return np.asarray(obs.R, np.float64).T @ (Xc - np.asarray(obs.t, np.float64))


def world_to_cam(c2w, X):
    c2w = np.asarray(c2w, np.float64)
    return (np.asarray(X, np.float64) - c2w[:3, 3]) @ c2w[:3, :3]      # R_wc (X - t)


def shift_for_pixels(s, z, fx=None):
    """깊이 z 의 평면이 s 픽셀 왼쪽으로 가려면 카메라가 +x 로 갈 거리(m)."""
    fx = FX() if fx is None else fx
    return s * z / fx
