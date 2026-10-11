"""LNS-07 테스트 공용 도우미. 오라클은 계약서 수식과 numpy 만 쓰고 구현 내부를 참조하지 않는다."""
import math

import numpy as np

from buc.core import contracts as C

# ---- 계약서 상수(정본). 테스트는 이 값으로 기대치를 만들고, 모듈 상수가 같은지도 따로 검사한다.
P_MAX = 0.75
TAU = 0.40
UP = (0.0, 0.0, 1.0)
NORTH = (1.0, 0.0, 0.0)  # 이 up/north 에서 east = north x up = (0,-1,0)


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def rayleigh_ref(gamma, p=P_MAX):
    g = np.asarray(gamma, np.float64)
    return p * np.sin(g) ** 2 / (1.0 + np.cos(g) ** 2)


def fresnel_ref(theta_i, n=1.33):
    """스넬 + 진폭 반사율 제곱. 반사광 편광도 |Rs-Rp|/(Rs+Rp)."""
    ti = np.asarray(theta_i, np.float64)
    tt = np.arcsin(np.sin(ti) / n)
    rs = (np.sin(ti - tt) / np.sin(ti + tt)) ** 2
    rp = (np.tan(ti - tt) / np.tan(ti + tt)) ** 2
    return np.abs(rs - rp) / (rs + rp)


def sun_world_ref(elev, azim, up=UP, north=NORTH):
    up = unit(up)
    n = np.asarray(north, np.float64)
    n = unit(n - (n @ up) * up)
    east = np.cross(n, up)
    e, a = math.radians(elev), math.radians(azim)
    return math.cos(e) * (math.cos(a) * n + math.sin(a) * east) + math.sin(e) * up


def cam_R_w2c(heading_deg, pitch_deg=0.0, roll_deg=0.0, up=UP, north=NORTH):
    """OpenCV 카메라(x 오른쪽, y 아래, z 앞)의 세계->카메라 회전. heading: north 에서 east 쪽으로 잰 시선 방위.
    roll: 광축 둘레로 카메라를 돌린 각(x 에서 y 쪽이 양)."""
    up = unit(up)
    n = np.asarray(north, np.float64)
    n = unit(n - (n @ up) * up)
    east = np.cross(n, up)
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    f = math.cos(p) * (math.cos(h) * n + math.sin(h) * east) + math.sin(p) * up
    x = unit(np.cross(f, up))
    y = np.cross(f, x)
    c2w = np.stack([x, y, f], axis=1)
    r = math.radians(roll_deg)
    rz = np.array([[math.cos(r), -math.sin(r), 0.0], [math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]])
    return (c2w @ rz).T


def rays_ref(theta, h, w):
    fx = theta.focal_mm / theta.sensor_width_mm * w
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.stack([(u - cx) / fx, (v - cy) / fx, np.ones((h, w))], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def gain_ref(theta, h, w, R_w2c, up=UP, north=NORTH, mask=None, angle=None, strength=None):
    """계약서 한 줄 수식을 독립적으로 푼 화소 이득 (H,W)."""
    s = theta.cpl if strength is None else strength
    ang = theta.extra.get("cpl_angle_deg", 0.0) if angle is None else angle
    m = np.ones((h, w)) if mask is None else np.asarray(mask, np.float64)
    if theta.sun_elev_deg is None or theta.sun_azim_deg is None:
        return np.full((h, w), (1 - s) + s * TAU)
    sun = R_w2c @ sun_world_ref(theta.sun_elev_deg, theta.sun_azim_deg, up, north)
    rays = rays_ref(theta, h, w)
    gamma = np.arccos(np.clip(rays @ sun, -1.0, 1.0))
    dop = rayleigh_ref(gamma) * m
    e = np.cross(rays, sun)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-300)
    a = math.radians(ang)
    proj = e[..., 0] * math.cos(a) + e[..., 1] * math.sin(a)
    cos2 = 2.0 * proj ** 2 - 1.0
    return (1 - s) + s * TAU * (1.0 + dop * cos2)


def sky_scene(h=48, w=64, seed=3):
    """결정론 장면: 파란 하늘 그라데이션 + 밝은 HDR 한 점. 선형, 최대 > 1."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([0.15 + 0.1 * yy / h, 0.3 + 0.1 * yy / h, 0.7 - 0.1 * yy / h], -1) + 0.01 * rng.random((h, w, 3))
    img[5:8, 10:13] = (30.0, 28.0, 25.0)
    return img.astype(np.float32)


def theta(**kw):
    extra = kw.pop("extra", {})
    return C.Theta(extra=dict(extra), **kw)
