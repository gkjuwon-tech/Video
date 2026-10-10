"""WLD-17 테스트 공용 도우미. 구현(buc.world.farfield)은 테스트 안에서 import 하고, 여기서는 오라클·합성 입력만 만든다.
모든 난수는 지역 Generator. 규약: 팔면체 맵은 세계축에 고정(+z = 맵 중심, -z = 네 모서리), 방위는 위에서 볼 때 시계방향."""
import math

import numpy as np

from buc.core import contracts as C

LUM = np.array([0.2126, 0.7152, 0.0722])


def lum(rgb):
    return np.asarray(rgb, dtype=np.float64) @ LUM


def rand_dirs(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dirs_near(center, max_deg, n, seed):
    """center 로부터 max_deg 이내의 균일(각도 기준 근사) 방향 n개."""
    c = np.asarray(center, dtype=np.float64)
    c = c / np.linalg.norm(c)
    rng = np.random.default_rng(seed)
    a = np.cross(c, [0.3, 0.5, 0.81])
    a /= np.linalg.norm(a)
    b = np.cross(c, a)
    ang = np.radians(max_deg) * np.sqrt(rng.random(n))
    phi = rng.random(n) * 2 * np.pi
    d = (np.cos(ang)[:, None] * c + np.sin(ang)[:, None] * (np.cos(phi)[:, None] * a + np.sin(phi)[:, None] * b))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def elev_az_dir(spec, elev_deg, az_deg):
    """spec 의 (north, east, up) 기저에서 (고도, 방위[시계방향]) 방향. 테스트 쪽 독립 구현."""
    up = np.asarray(spec.up, dtype=np.float64)
    north = np.asarray(spec.north, dtype=np.float64)
    east = np.cross(north, up)
    e, a = math.radians(elev_deg), math.radians(az_deg)
    return math.cos(e) * (math.cos(a) * north + math.sin(a) * east) + math.sin(e) * up


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_obs(R, W=96, H=64, fov_x_deg=60.0):
    f = (W / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    K = np.array([[f, 0, (W - 1) / 2.0], [0, f, (H - 1) / 2.0], [0, 0, 1.0]])
    return C.Obs(K=K, R=np.asarray(R, dtype=np.float64), t=np.array([0.1, -0.2, 0.3]), width=W, height=H)


def make_prov():
    return C.Prov(taint=frozenset({"ENGINE"}), mode="engine", roots=(), ns="engine")


def make_layer(h, w, depth=2.0, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rgba = rng.random((h, w, 4)).astype(np.float32)
    rgba[..., 3] = alpha
    return C.LDILayer(rgba=rgba, depth=np.full((h, w), depth, dtype=np.float32))


def sphere_quadrature(spec, nt=256, nph=512):
    """상반구 중점 구적(cosθ 가중). 테스트 쪽 독립 구현: (dirs (nt,nph,3), weights (nt,nph))."""
    up = np.asarray(spec.up, dtype=np.float64)
    north = np.asarray(spec.north, dtype=np.float64)
    east = np.cross(north, up)
    th = (np.arange(nt) + 0.5) / nt * (np.pi / 2)
    ph = (np.arange(nph) + 0.5) / nph * 2 * np.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = (np.sin(TH)[..., None] * (np.cos(PH)[..., None] * north + np.sin(PH)[..., None] * east) + np.cos(TH)[..., None] * up)
    w = np.sin(TH) * np.cos(TH) * (np.pi / 2 / nt) * (2 * np.pi / nph)
    return d, w
