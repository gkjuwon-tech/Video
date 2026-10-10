"""RDR-02 테스트 공용 도우미: 작은 결정론 LDI 월드, 카메라 포즈, 평면 호모그래피 오라클.

오라클은 FND-06 geom/pyramid(동결·검증된 모듈)만 쓰고 RDR-02 구현은 참조하지 않는다."""
import math

import numpy as np

from buc.core import contracts as C
from buc.core import geom, pyramid

W, H = 96, 64
FOV = 60.0


def K_default(w=W, h=H, fov=FOV):
    return geom.intrinsics_from_fov(fov, w, h)


def FX(w=W, fov=FOV):
    return (w / 2.0) / math.tan(math.radians(fov) / 2.0)


def make_obs(c2w=None, w=W, h=H, fov=FOV, k1=0.0, k2=0.0):
    c2w = np.eye(4) if c2w is None else np.asarray(c2w, np.float64)
    w2c = geom.invert_pose(c2w)
    return C.Obs(K=K_default(w, h, fov), R=w2c[:3, :3], t=w2c[:3, 3], width=w, height=h, k1=k1, k2=k2)


def make_world(layers, obs=None):
    obs = make_obs() if obs is None else obs
    return C.World(obs=obs, layers=tuple(layers), theta_in=C.Theta(), level="L1", read_hash="r" * 8,
                   prov=C.Prov())


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


def smooth_tex(w=W, h=H, alpha=1.0):
    """저주파 텍스처(파장 >= 40 px) -> 쌍선형·타일 필터 차이가 작다. 값 [0.15, 0.85]."""
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
    """카메라 자신의 y축(아래) 둘레 회전. 위치 고정."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.array(c2w, np.float64, copy=True)
    T[:3, :3] = T[:3, :3] @ Ry
    return T


def plane_oracle(tex_rgb, src_obs, c2w_t, K_t, w_t, h_t, n, d, margin=3.0):
    """src 카메라 좌표계의 평면 n.X=d 를 목표 카메라에서 본 영상(역방향 호모그래피 + 쌍선형).
    반환: (rgb (h_t,w_t,3) float64, inside (h_t,w_t) bool: 원본 영상 안쪽 margin px 이상), src_uv"""
    w2c_s = np.eye(4)
    w2c_s[:3, :3] = src_obs.R
    w2c_s[:3, 3] = src_obs.t
    rel = geom.invert_pose(c2w_t) @ geom.invert_pose(w2c_s)          # src cam -> dst cam
    Hm = geom.plane_homography(src_obs.K, K_t, rel[:3, :3], rel[:3, 3], np.asarray(n, float), float(d))
    Hinv = np.linalg.inv(Hm)
    Hinv = Hinv / Hinv[2, 2]
    uv_dst = geom.pixel_grid(h_t, w_t)
    uv_src = geom.apply_homography(Hinv, uv_dst)
    hs, ws = tex_rgb.shape[:2]
    inside = ((uv_src[..., 0] >= margin) & (uv_src[..., 0] <= ws - 1 - margin) &
              (uv_src[..., 1] >= margin) & (uv_src[..., 1] <= hs - 1 - margin))
    rgb = pyramid.sample_bilinear(tex_rgb.astype(np.float64), uv_src, border="edge")
    return rgb, inside, uv_src


def edge_left(alpha_row, mid):
    """왼쪽 가장자리의 연속 좌표 추정: 픽셀 u 는 [u-.5, u+.5). mid 는 카드 안쪽 열."""
    return (mid + 0.5) - float(alpha_row[: mid + 1].sum())


def edge_right(alpha_row, mid):
    return (mid - 0.5) + float(alpha_row[mid:].sum())
