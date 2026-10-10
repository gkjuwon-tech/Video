"""SCRATCH REFERENCE (test calibration only; not part of the repo)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize

from buc.core import contracts as C

FARFIELD_VERSION = "farfield/v0"
DEFAULT_SIZE = 256
MIN_SIZE, MAX_SIZE = 8, 4096
SUN_ANGULAR_RADIUS_DEG = 0.2665
TURBIDITY_RANGE = (1.0, 10.0)
DEFAULT_SUN_ELEV_DEG = 45.0
DEFAULT_SUN_AZIM_DEG = 180.0
HW_NCOEF = 9
_LUM = np.array([0.2126, 0.7152, 0.0722])


class FarFieldError(C.ContractError):
    pass


def _unit(v, name="v"):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(v).all() or (n <= 0).any():
        raise FarFieldError(f"{name}: bad vector")
    return v / n


def _sgn(x):
    return np.where(x >= 0, 1.0, -1.0)


def oct_encode(d):
    d = _unit(d, "d")
    l1 = np.abs(d).sum(-1, keepdims=True)
    p = d[..., :2] / l1
    z = d[..., 2]
    px, py = p[..., 0], p[..., 1]
    neg = z < 0
    fx = (1 - np.abs(py)) * _sgn(px)
    fy = (1 - np.abs(px)) * _sgn(py)
    px = np.where(neg, fx, px)
    py = np.where(neg, fy, py)
    return (np.stack([px, py], -1) + 1) * 0.5


def oct_decode(uv):
    uv = np.asarray(uv, dtype=np.float64)
    p = uv * 2 - 1
    px, py = p[..., 0], p[..., 1]
    z = 1 - np.abs(px) - np.abs(py)
    neg = z < 0
    fx = (1 - np.abs(py)) * _sgn(px)
    fy = (1 - np.abs(px)) * _sgn(py)
    px2 = np.where(neg, fx, px)
    py2 = np.where(neg, fy, py)
    d = np.stack([px2, py2, z], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _chk_size(size):
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or not (MIN_SIZE <= size <= MAX_SIZE):
        raise FarFieldError("size")


def texel_centers(size):
    _chk_size(size)
    t = (np.arange(size) + 0.5) / size
    v, u = np.meshgrid(t, t, indexing="ij")
    return np.stack([u, v], -1)


def texel_dirs(size):
    return oct_decode(texel_centers(size))


def texel_solid_angle(size):
    d = texel_dirs(size)
    l1 = np.abs(d).sum(-1)
    return 4.0 / (size * size) * l1 ** 3


def sample(far, d):
    rgb = far.rgb if hasattr(far, "rgb") else np.asarray(far)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1]:
        raise FarFieldError("rgb")
    S = rgb.shape[0]
    d = np.asarray(d, dtype=np.float64)
    shp = d.shape[:-1]
    uv = oct_encode(d.reshape(-1, 3))
    x = uv[:, 0] * S - 0.5
    y = uv[:, 1] * S - 0.5
    j0 = np.floor(x).astype(int)
    i0 = np.floor(y).astype(int)
    fx = x - j0
    fy = y - i0
    out = np.zeros((uv.shape[0], 3))

    def fetch(i, j):
        i = i.copy()
        j = j.copy()
        m = (j < 0) | (j >= S)
        j2 = np.where(j < 0, -1 - j, np.where(j >= S, 2 * S - 1 - j, j))
        i2 = np.where(m, S - 1 - i, i)
        m2 = (i2 < 0) | (i2 >= S)
        i3 = np.where(i2 < 0, -1 - i2, np.where(i2 >= S, 2 * S - 1 - i2, i2))
        j3 = np.where(m2, S - 1 - j2, j2)
        return rgb[i3, j3].astype(np.float64)

    for di, dj, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        out += w[:, None] * fetch(i0 + di, j0 + dj)
    return out.reshape(*shp, 3)


def hw_shape(coeffs, cos_theta, gamma):
    A, B, Cc, D, E, F, G, H, I = [float(x) for x in coeffs]
    ct = np.clip(np.asarray(cos_theta, dtype=np.float64), 0.0, 1.0)
    g = np.asarray(gamma, dtype=np.float64)
    cg = np.cos(g)
    chi = (1 + cg ** 2) / (1 + H * H - 2 * H * cg) ** 1.5
    return (1 + A * np.exp(B / (ct + 0.01))) * (Cc + D * np.exp(E * g) + F * cg ** 2 + G * chi + I * np.sqrt(ct))


# ---- Preetham target for the fit -------------------------------------------------------------------------------
def _preetham_rgb(T, elev_deg, view_elev, view_dazim):
    ts = math.radians(90 - max(elev_deg, 0.0))
    tv = np.pi / 2 - view_elev
    cosg = np.cos(tv) * math.cos(ts) + np.sin(tv) * math.sin(ts) * np.cos(view_dazim)
    gam = np.arccos(np.clip(cosg, -1, 1))
    cm = {"Y": (0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703),
          "x": (-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452),
          "y": (-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529)}

    def perez(k, tv_, gam_):
        A, B, Cc, D, E = cm[k]
        return (1 + A * np.exp(B / np.maximum(np.cos(tv_), 1e-3))) * (1 + Cc * np.exp(D * gam_) + E * np.cos(gam_) ** 2)

    chi = (4.0 / 9.0 - T / 120.0) * (math.pi - 2 * ts)
    Yz = (4.0453 * T - 4.9710) * math.tan(chi) - 0.2155 * T + 2.4192
    th = ts
    xz = (T * T * (0.00166 * th ** 3 - 0.00375 * th ** 2 + 0.00209 * th) + T * (-0.02903 * th ** 3 + 0.06377 * th ** 2 - 0.03202 * th + 0.00394)
          + (0.11693 * th ** 3 - 0.21196 * th ** 2 + 0.06052 * th + 0.25886))
    yz = (T * T * (0.00275 * th ** 3 - 0.00610 * th ** 2 + 0.00317 * th) + T * (-0.04214 * th ** 3 + 0.08970 * th ** 2 - 0.04153 * th + 0.00516)
          + (0.15346 * th ** 3 - 0.26756 * th ** 2 + 0.06670 * th + 0.26688))
    out = []
    Y = Yz * perez("Y", tv, gam) / perez("Y", 0.0, ts)
    x = xz * perez("x", tv, gam) / perez("x", 0.0, ts)
    y = yz * perez("y", tv, gam) / perez("y", 0.0, ts)
    X = x / y * Y
    Z = (1 - x - y) / y * Y
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return np.maximum(np.stack([r, g, b], -1), 1e-6)


@lru_cache(maxsize=256)
def _fit(Tq, eq):
    ve = np.linspace(0.0, 89.0, 30) * np.pi / 180
    va = np.linspace(0, np.pi, 24)
    VE, VA = np.meshgrid(ve, va, indexing="ij")
    tgt = _preetham_rgb(Tq, eq, VE, VA)  # (..,3)
    ct = np.sin(VE)
    ts = math.radians(90 - eq)
    cosg = np.cos(np.pi / 2 - VE) * math.cos(ts) + np.sin(np.pi / 2 - VE) * math.sin(ts) * np.cos(VA)
    gam = np.arccos(np.clip(cosg, -1, 1))
    zen = _preetham_rgb(Tq, eq, np.array(np.pi / 2), np.array(0.0))
    lo = [-3, -3, -1, -3, -12, -3, -3, 0.5, -1]
    hi = [3, 0, 3, 6, 0, 3, 3, 0.999, 3]
    x0 = np.array([-0.9, -0.25, 0.3, 0.8, -2.0, 0.1, 0.3, 0.9, 0.05])
    cs = []
    for c in range(3):
        t = tgt[..., c] / zen[c]

        def res(p):
            return ((hw_shape(p, ct, gam) - t) / (t + 0.05)).ravel()

        sol = optimize.least_squares(res, x0, bounds=(lo, hi), max_nfev=200)
        cs.append(sol.x)
    return np.array(cs), zen.copy()


def hw_params(turbidity, albedo, sun_elev_deg):
    for v in (turbidity, albedo, sun_elev_deg):
        if isinstance(v, (bool, np.bool_)) or not math.isfinite(float(v)):
            raise FarFieldError("hw_params")
    if not (1.0 <= turbidity <= 10.0 and 0.0 <= albedo <= 1.0 and -90.0 <= sun_elev_deg <= 90.0):
        raise FarFieldError("hw_params range")
    T = round(float(turbidity), 3)
    e = round(float(min(max(sun_elev_deg, 0.0), 90.0)), 2)
    cs, z = _fit(T, e)
    z = z * (1 + 0.1 * float(albedo))
    if sun_elev_deg < 0:
        z = z * math.exp(sun_elev_deg / 2.5)
    return cs.copy(), z.copy()


@dataclass(frozen=True, kw_only=True)
class FarFieldSpec:
    up: tuple = (0.0, 0.0, 1.0)
    north: tuple | None = None
    sun_elev_deg: float = DEFAULT_SUN_ELEV_DEG
    sun_azim_deg: float = DEFAULT_SUN_AZIM_DEG
    turbidity: float = 2.5
    albedo: float = 0.3
    exposure: float = 1.0
    camera_height_m: float = 1.6
    horizon_blend_deg: float = 1.5
    ground_albedo_rgb: tuple | None = None
    size: int = DEFAULT_SIZE
    include_sun: bool = False

    def __post_init__(self):
        def fin(name, lo=None, hi=None, strict_lo=False):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)):
                raise FarFieldError(f"{name}")
            v = float(v)
            if lo is not None and (v < lo or (strict_lo and v == lo)):
                raise FarFieldError(f"{name}")
            if hi is not None and v > hi:
                raise FarFieldError(f"{name}")
            object.__setattr__(self, name, v)
        fin("sun_elev_deg", -90, 90)
        fin("sun_azim_deg")
        object.__setattr__(self, "sun_azim_deg", self.sun_azim_deg % 360.0)
        fin("turbidity", *TURBIDITY_RANGE)
        fin("albedo", 0, 1)
        fin("exposure", 0, None, True)
        fin("camera_height_m", 0, None, True)
        fin("horizon_blend_deg", 0, None, True)
        s = self.size
        if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or not (MIN_SIZE <= s <= MAX_SIZE):
            raise FarFieldError("size")
        object.__setattr__(self, "size", int(s))
        if not isinstance(self.include_sun, (bool, np.bool_)):
            raise FarFieldError("include_sun")
        up = np.asarray(self.up, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) <= 0:
            raise FarFieldError("up")
        up = up / np.linalg.norm(up)
        if self.north is None:
            k = int(np.argmin(np.abs(up)))
            e = np.zeros(3)
            e[k] = 1.0
            n = e - up * (e @ up)
        else:
            n = np.asarray(self.north, dtype=np.float64)
            if n.shape != (3,) or not np.isfinite(n).all():
                raise FarFieldError("north")
            n = n - up * (n @ up)
        if np.linalg.norm(n) < 1e-6:
            raise FarFieldError("north parallel to up")
        n = n / np.linalg.norm(n)
        object.__setattr__(self, "up", tuple(float(x) for x in up))
        object.__setattr__(self, "north", tuple(float(x) for x in n))
        if self.ground_albedo_rgb is not None:
            g = np.asarray(self.ground_albedo_rgb, dtype=np.float64)
            if g.shape != (3,) or not np.isfinite(g).all() or g.min() < 0 or g.max() > 1:
                raise FarFieldError("ground_albedo_rgb")
            object.__setattr__(self, "ground_albedo_rgb", tuple(float(x) for x in g))

    def basis(self):
        up = np.array(self.up)
        n = np.array(self.north)
        east = np.cross(n, up)
        return n, east, up

    def sun_dir(self):
        n, e, u = self.basis()
        el, az = math.radians(self.sun_elev_deg), math.radians(self.sun_azim_deg)
        return math.cos(el) * (math.cos(az) * n + math.sin(az) * e) + math.sin(el) * u

    def to_meta(self):
        return {"version": FARFIELD_VERSION, "up": list(self.up), "north": list(self.north), "sun_elev_deg": self.sun_elev_deg,
                "sun_azim_deg": self.sun_azim_deg, "turbidity": self.turbidity, "albedo": self.albedo, "exposure": self.exposure,
                "camera_height_m": self.camera_height_m, "horizon_blend_deg": self.horizon_blend_deg,
                "ground_albedo_rgb": None if self.ground_albedo_rgb is None else list(self.ground_albedo_rgb),
                "size": self.size, "include_sun": self.include_sun}

    @classmethod
    def from_meta(cls, m):
        m = dict(m)
        if m.pop("version", None) != FARFIELD_VERSION:
            raise FarFieldError("version")
        try:
            return cls(**m)
        except TypeError as e:
            raise FarFieldError(str(e))

    def build(self):
        d = texel_dirs(self.size)
        rgb = environment_radiance(self, d)
        if self.include_sun and self.sun_elev_deg > 0:
            om = texel_solid_angle(self.size)
            sd = self.sun_dir()
            ang = np.arccos(np.clip(d @ sd, -1, 1))
            sig = 1.5 * math.sqrt(4 * math.pi / self.size ** 2)
            k = np.exp(-0.5 * (ang / sig) ** 2)
            k = k / (k * om).sum()
            rgb = rgb + k[..., None] * sun_irradiance(self)[None, None, :]
        return C.FarField(rgb=rgb.astype(np.float32))


def sky_radiance(spec, d):
    d = _unit(d, "d")
    n, e, u = spec.basis()
    sd = spec.sun_dir()
    ct = np.clip(d @ u, 0.0, 1.0)
    g = np.arccos(np.clip(d @ sd, -1, 1))
    cs, z = hw_params(spec.turbidity, spec.albedo, spec.sun_elev_deg)
    out = np.stack([hw_shape(cs[c], ct, g) * z[c] for c in range(3)], -1)
    return np.maximum(out, 0.0) * spec.exposure


def _quad_dirs(spec, nt=64, nph=128):
    n, e, u = spec.basis()
    th = (np.arange(nt) + 0.5) / nt * (np.pi / 2)
    ph = (np.arange(nph) + 0.5) / nph * 2 * np.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = (np.sin(TH)[..., None] * (np.cos(PH)[..., None] * n + np.sin(PH)[..., None] * e) + np.cos(TH)[..., None] * u)
    w = np.sin(TH) * np.cos(TH) * (np.pi / 2 / nt) * (2 * np.pi / nph)
    return d, w


def sky_irradiance(spec):
    d, w = _quad_dirs(spec)
    return (sky_radiance(spec, d) * w[..., None]).sum((0, 1))


def sun_irradiance(spec):
    el = spec.sun_elev_deg
    if el <= 0:
        return np.zeros(3)
    m = 1.0 / (math.sin(math.radians(el)) + 0.15 * (el + 3.885) ** -1.253)
    tau = np.array([0.07, 0.10, 0.15]) * (0.5 + 0.2 * spec.turbidity)
    rgb = np.exp(-tau * m)
    lum = rgb @ _LUM
    En = 100.0 * math.exp(-0.1 * (0.5 + 0.2 * spec.turbidity) * m) * rgb / lum
    return En * spec.exposure


def ground_lit(spec):
    alb = np.full(3, spec.albedo) if spec.ground_albedo_rgb is None else np.array(spec.ground_albedo_rgb)
    es = math.sin(math.radians(max(spec.sun_elev_deg, 0.0)))
    return alb / math.pi * (sky_irradiance(spec) + sun_irradiance(spec) * es)


def ground_distance(spec, d):
    d = _unit(d, "d")
    s = -(d @ np.array(spec.up))
    with np.errstate(divide="ignore"):
        return np.where(s > 1e-12, spec.camera_height_m / np.maximum(s, 1e-300), np.inf)


def environment_radiance(spec, d):
    d = _unit(d, "d")
    n, e, u = spec.basis()
    el = d @ u
    sky = sky_radiance(spec, d)
    dh = d - el[..., None] * u
    nh = np.linalg.norm(dh, axis=-1, keepdims=True)
    dh = np.where(nh > 1e-9, dh / np.maximum(nh, 1e-300), n)
    hor = sky_radiance(spec, dh)
    delta = np.degrees(np.arcsin(np.clip(-el, 0, 1)))
    w = np.exp(-delta / spec.horizon_blend_deg)[..., None]
    gnd = w * hor + (1 - w) * ground_lit(spec)
    return np.where((el >= 0)[..., None], sky, gnd)


def up_world_from_obs(obs, up_cam):
    return _unit(np.asarray(obs.R, dtype=np.float64).T @ np.asarray(up_cam, dtype=np.float64), "up_cam")


def spec_for_world(obs, up_cam, theta_in, **overrides):
    up = up_world_from_obs(obs, up_cam)
    R = np.asarray(obs.R, dtype=np.float64)
    f = R.T @ np.array([0.0, 0.0, 1.0])
    fh = f - up * (f @ up)
    if np.linalg.norm(fh) < 1e-3:
        fh = R.T @ np.array([1.0, 0.0, 0.0])
        fh = fh - up * (fh @ up)
    kw = dict(up=tuple(up), north=tuple(fh / np.linalg.norm(fh)))
    if theta_in.sun_elev_deg is not None:
        kw["sun_elev_deg"] = theta_in.sun_elev_deg
        kw["sun_azim_deg"] = theta_in.sun_azim_deg
    kw.update(overrides)
    return FarFieldSpec(**kw)
