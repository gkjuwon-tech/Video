"""WLD-17 far-field: octahedral HDR environment, Hosek-Wilkie-form analytic sky, horizon blend, ground extension.

Frame outside the camera (back of the head, orbit ends) is filled by an analytic environment with zero learned
parameters. The result is a FND-02 ``FarField`` ((S,S,3) float32 octahedral map) that is seam-free at -z and at the four
fold edges because ``sample`` mirrors out-of-range texels across the map edges.

Units: linear Rec.709, luminance in kcd/m^2 (zenith ~1-20), illuminance in klux.

HONESTY NOTE on the sky coefficients. The original Hosek-Wilkie coefficient tables are NOT in this repository and could
not be verified, so none of their numbers are used here. ``hw_shape`` is the Hosek-Wilkie *functional form*; the nine
coefficients per colour channel come from ``hw_params``, a deterministic closed-form approximation hand-set in this file:
  * zenith luminance Yz(T, sun elevation): the Preetham et al. (1999) closed form, in kcd/m^2, with a smooth twilight
    fall-off below the horizon;
  * zenith chroma: a smooth analytic blue/red ratio that whitens with turbidity;
  * (A..I) shape coefficients: simple analytic functions of turbidity and sun elevation (horizon brightening, Mie
    circumsolar lobe, forward scattering), chosen by hand to match the qualitative behaviour of Perez-type skies. No
    least-squares fit against Perez/Preetham data was run; this is an approximation, not HW data.
The zenith scale Z is derived analytically so that the sky at the zenith equals the target zenith luminance/chroma.
Sun direct illuminance uses Rayleigh + Angstrom aerosol transmittance with the Kasten-Young air mass.
No iterative solver, no torch/scipy, no global RNG, no import side effects.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from buc.core import contracts as C

FARFIELD_VERSION = "farfield/v0"
DEFAULT_SIZE = 256
MIN_SIZE = 8
MAX_SIZE = 4096
SUN_ANGULAR_RADIUS_DEG = 0.2665
TURBIDITY_RANGE = (1.0, 10.0)
DEFAULT_SUN_ELEV_DEG = 45.0
DEFAULT_SUN_AZIM_DEG = 180.0


class FarFieldError(C.ContractError):
    """far-field 입력·파라미터 위반."""


# ---------------------------------------------------------------------------------------------------------------
# 검증 도우미
# ---------------------------------------------------------------------------------------------------------------
def _num(name, v, lo=None, hi=None, lo_open=False):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise FarFieldError(f"{name}: 숫자여야 한다 ({type(v).__name__})")
    x = float(v)
    if not math.isfinite(x):
        raise FarFieldError(f"{name}: 유한하지 않다")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise FarFieldError(f"{name}={x}: 범위 밖")
    if hi is not None and x > hi:
        raise FarFieldError(f"{name}={x}: 범위 밖")
    return x


def _vec3(name, v):
    if isinstance(v, (str, bytes)):
        raise FarFieldError(f"{name}: 3-벡터여야 한다")
    try:
        a = np.asarray(v)
    except Exception as e:  # noqa: BLE001
        raise FarFieldError(f"{name}: 배열로 바꿀 수 없다") from e
    if a.shape != (3,) or a.dtype == bool or a.dtype.kind not in "iuf":
        raise FarFieldError(f"{name}: 실수 3-벡터여야 한다 (shape {a.shape}, dtype {a.dtype})")
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise FarFieldError(f"{name}: 유한하지 않다")
    return a


def _check_size(size):
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
        raise FarFieldError(f"size: int 여야 한다 ({type(size).__name__})")
    s = int(size)
    if s < MIN_SIZE or s > MAX_SIZE:
        raise FarFieldError(f"size={s}: [{MIN_SIZE}, {MAX_SIZE}] 밖")
    return s


def _unit_dirs(d, name="d"):
    a = np.asarray(d, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise FarFieldError(f"{name}: 마지막 차원이 3 이어야 한다 (shape {a.shape})")
    if not np.isfinite(a).all():
        raise FarFieldError(f"{name}: 유한하지 않다")
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if (n <= 0.0).any():
        raise FarFieldError(f"{name}: 영벡터")
    return a / n


def _normalize_fixed(v):
    """정규화를 (거의) 멱등이 되도록 반복한다."""
    v = np.asarray(v, dtype=np.float64)
    for _ in range(4):
        w = v / np.linalg.norm(v)
        if np.array_equal(w, v):
            break
        v = w
    return v


# ---------------------------------------------------------------------------------------------------------------
# 팔면체 코덱
# ---------------------------------------------------------------------------------------------------------------
def _sgn(x):
    return np.where(x >= 0.0, 1.0, -1.0)


def oct_encode(d):
    d = _unit_dirs(d)
    l1 = np.abs(d).sum(-1, keepdims=True)
    p = d[..., :2] / l1
    z = d[..., 2]
    px, py = p[..., 0], p[..., 1]
    fx = (1.0 - np.abs(py)) * _sgn(px)
    fy = (1.0 - np.abs(px)) * _sgn(py)
    neg = z < 0.0
    px = np.where(neg, fx, px)
    py = np.where(neg, fy, py)
    uv = (np.stack([px, py], axis=-1) + 1.0) * 0.5
    return np.clip(uv, 0.0, 1.0)


def oct_decode(uv):
    a = np.asarray(uv, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise FarFieldError(f"uv: 마지막 차원이 2 여야 한다 (shape {a.shape})")
    if not np.isfinite(a).all():
        raise FarFieldError("uv: 유한하지 않다")
    p = a * 2.0 - 1.0
    px, py = p[..., 0], p[..., 1]
    z = 1.0 - np.abs(px) - np.abs(py)
    neg = z < 0.0
    fx = (1.0 - np.abs(py)) * _sgn(px)
    fy = (1.0 - np.abs(px)) * _sgn(py)
    x = np.where(neg, fx, px)
    y = np.where(neg, fy, py)
    d = np.stack([x, y, z], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def texel_centers(size):
    s = _check_size(size)
    c = (np.arange(s, dtype=np.float64) + 0.5) / s
    u, v = np.meshgrid(c, c, indexing="xy")  # [i, j] -> u from j, v from i
    return np.stack([u, v], axis=-1)


def texel_dirs(size):
    return oct_decode(texel_centers(size))


def texel_solid_angle(size):
    s = _check_size(size)
    l1 = np.abs(texel_dirs(s)).sum(-1)
    return 4.0 / (s * s) * l1 ** 3


def _fold(i, j, s):
    """범위 밖 텍셀 인덱스를 변 거울 접힘으로 되돌린다(모서리는 두 번 적용)."""
    for _ in range(2):
        m = j < 0
        i = np.where(m, s - 1 - i, i)
        j = np.where(m, -1 - j, j)
        m = j >= s
        i = np.where(m, s - 1 - i, i)
        j = np.where(m, 2 * s - 1 - j, j)
        m = i < 0
        j = np.where(m, s - 1 - j, j)
        i = np.where(m, -1 - i, i)
        m = i >= s
        j = np.where(m, s - 1 - j, j)
        i = np.where(m, 2 * s - 1 - i, i)
    return i, j


def sample(far, d):
    rgb = getattr(far, "rgb", far)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[0] < 1:
        raise FarFieldError(f"far: (S,S,3) 여야 한다 (shape {rgb.shape})")
    s = rgb.shape[0]
    du = _unit_dirs(d)
    lead = du.shape[:-1]
    uv = oct_encode(du.reshape(-1, 3))
    x = uv[:, 0] * s - 0.5  # 열
    y = uv[:, 1] * s - 0.5  # 행
    j0 = np.floor(x).astype(np.int64)
    i0 = np.floor(y).astype(np.int64)
    fx = (x - j0)[:, None]
    fy = (y - i0)[:, None]
    out = np.zeros((uv.shape[0], 3), dtype=np.float64)
    for di, wy in ((0, 1.0 - fy), (1, fy)):
        for dj, wx in ((0, 1.0 - fx), (1, fx)):
            ii, jj = _fold(i0 + di, j0 + dj, s)
            out += rgb[ii, jj].astype(np.float64) * (wy * wx)
    return out.reshape(lead + (3,))


# ---------------------------------------------------------------------------------------------------------------
# Hosek-Wilkie 함수형과 계수
# ---------------------------------------------------------------------------------------------------------------
def hw_shape(coeffs, cos_theta, gamma):
    c = np.asarray(coeffs, dtype=np.float64)
    if c.shape[-1:] != (9,):
        raise FarFieldError(f"coeffs: 마지막 차원이 9 여야 한다 (shape {c.shape})")
    A, B, Cc, D, E, Ff, G, H, I = (c[..., k] for k in range(9))
    ct = np.clip(np.asarray(cos_theta, dtype=np.float64), 0.0, 1.0)
    g = np.asarray(gamma, dtype=np.float64)
    cg = np.cos(g)
    chi = (1 + cg ** 2) / (1 + H * H - 2 * H * cg) ** 1.5
    return (1 + A * np.exp(B / (ct + 0.01))) * (Cc + D * np.exp(E * g) + Ff * cg ** 2 + G * chi + I * np.sqrt(ct))


_A_RGB = (-0.80, -0.70, -0.55)  # 붉은 채널이 지평선에서 더 밝아진다 -> 지평선이 더 희다
_LUM = np.array([0.2126, 0.7152, 0.0722])


def _zenith_luminance(turb, elev_deg):
    """Preetham 천정 휘도(kcd/m^2). 고도 65도 이상은 같은 값으로 고정, 지평선 아래는 박명 감쇠."""
    e = min(max(elev_deg, 0.0), 65.0)
    theta_s = math.radians(90.0 - e)
    chi = (4.0 / 9.0 - turb / 120.0) * (math.pi - 2.0 * theta_s)
    y = (4.0453 * turb - 4.9710) * math.tan(chi) - 0.2155 * turb + 2.4192
    y = max(y, 0.05)
    if elev_deg < 0.0:
        y *= math.exp(elev_deg / 3.0)
    return y


def _zenith_chroma(turb):
    rho = 1.0 + 2.2 * math.exp(-0.18 * (turb - 1.0))  # B/R
    c = np.array([1.0, 1.0 + 0.62 * (rho - 1.0), rho])
    return c / float(c @ _LUM)


@functools.lru_cache(maxsize=256)
def _hw_params_cached(turb, albedo, elev):
    boost = 1.0 + 0.6 * math.cos(math.radians(min(max(elev, 0.0), 90.0)))
    h = min(0.78 + 0.018 * (turb - 1.0), 0.97)
    gm = 0.25 * (1.0 + 0.25 * (turb - 1.0)) * boost
    gk = gm * (1.0 - h) ** 3 / 2.0  # G*chi(0) = gm
    ee = -3.5 / (1.0 + 0.15 * (turb - 1.0))
    ds = 1.4 * boost
    sa = 1.0 + 0.03 * (turb - 1.0)
    coeffs = np.zeros((3, 9))
    for k in range(3):
        coeffs[k] = (max(_A_RGB[k] * sa, -1.2), -0.30, 0.50, ds, ee, 0.10, gk, h, 0.10)
    g_ref = max(math.pi / 2.0 - math.radians(elev), 0.6 if elev > 0.0 else 0.0)
    g_ref = max(g_ref, 0.6) if elev > 0.0 else g_ref
    shape_z = np.array([float(hw_shape(coeffs[k], 1.0, g_ref)) for k in range(3)])
    z = _zenith_luminance(turb, elev) * _zenith_chroma(turb) * (1.0 + 0.12 * albedo) / shape_z
    coeffs.setflags(write=False)
    z.setflags(write=False)
    return coeffs, z


def hw_params(turbidity, albedo, sun_elev_deg):
    t = _num("turbidity", turbidity, TURBIDITY_RANGE[0], TURBIDITY_RANGE[1])
    a = _num("albedo", albedo, 0.0, 1.0)
    e = _num("sun_elev_deg", sun_elev_deg, -90.0, 90.0)
    cs, z = _hw_params_cached(t, a, e)
    return cs.copy(), z.copy()


# ---------------------------------------------------------------------------------------------------------------
# 스펙
# ---------------------------------------------------------------------------------------------------------------
_META_KEYS = ("version", "up", "north", "sun_elev_deg", "sun_azim_deg", "turbidity", "albedo", "exposure",
              "camera_height_m", "horizon_blend_deg", "ground_albedo_rgb", "size", "include_sun")


@dataclass(frozen=True, kw_only=True)
class FarFieldSpec:
    up: tuple
    north: Optional[tuple] = None
    sun_elev_deg: float = DEFAULT_SUN_ELEV_DEG
    sun_azim_deg: float = DEFAULT_SUN_AZIM_DEG
    turbidity: float = 2.5
    albedo: float = 0.3
    exposure: float = 1.0
    camera_height_m: float = 1.6
    horizon_blend_deg: float = 1.5
    ground_albedo_rgb: Optional[tuple] = None
    size: int = DEFAULT_SIZE
    include_sun: bool = False

    def __post_init__(self):
        sv = object.__setattr__
        up = _vec3("up", self.up)
        n = float(np.linalg.norm(up))
        if n <= 0.0:
            raise FarFieldError("up: 영벡터")
        up = _normalize_fixed(up)
        if self.north is None:
            k = int(np.argmin(np.abs(up)))
            nv = np.zeros(3)
            nv[k] = 1.0
        else:
            nv = _vec3("north", self.north)
        nv = nv - float(nv @ up) * up
        nn = float(np.linalg.norm(nv))
        if nn <= 1e-9 * max(1.0, float(np.linalg.norm(_vec3("north", self.north)) if self.north is not None else 1.0)):
            raise FarFieldError("north: up 과 평행하거나 영벡터")
        nv = _normalize_fixed(nv)
        sv(self, "up", tuple(float(x) for x in up))
        sv(self, "north", tuple(float(x) for x in nv))
        sv(self, "sun_elev_deg", _num("sun_elev_deg", self.sun_elev_deg, -90.0, 90.0))
        az = math.fmod(_num("sun_azim_deg", self.sun_azim_deg), 360.0)
        if az < 0.0:
            az += 360.0
        if az >= 360.0:
            az = 0.0
        sv(self, "sun_azim_deg", az)
        sv(self, "turbidity", _num("turbidity", self.turbidity, TURBIDITY_RANGE[0], TURBIDITY_RANGE[1]))
        sv(self, "albedo", _num("albedo", self.albedo, 0.0, 1.0))
        sv(self, "exposure", _num("exposure", self.exposure, 0.0, None, lo_open=True))
        sv(self, "camera_height_m", _num("camera_height_m", self.camera_height_m, 0.0, None, lo_open=True))
        sv(self, "horizon_blend_deg", _num("horizon_blend_deg", self.horizon_blend_deg, 0.0, None, lo_open=True))
        if self.ground_albedo_rgb is not None:
            g = _vec3("ground_albedo_rgb", self.ground_albedo_rgb)
            if (g < 0.0).any() or (g > 1.0).any():
                raise FarFieldError("ground_albedo_rgb: [0,1] 밖")
            sv(self, "ground_albedo_rgb", tuple(float(x) for x in g))
        sv(self, "size", _check_size(self.size))
        if not isinstance(self.include_sun, (bool, np.bool_)):
            raise FarFieldError("include_sun: bool 이어야 한다")
        sv(self, "include_sun", bool(self.include_sun))

    # -- 기하 ------------------------------------------------------------------------------------------------
    def _basis(self):
        up = np.array(self.up)
        north = np.array(self.north)
        return north, np.cross(north, up), up

    def sun_dir(self):
        north, east, up = self._basis()
        e, a = math.radians(self.sun_elev_deg), math.radians(self.sun_azim_deg)
        return math.cos(e) * (math.cos(a) * north + math.sin(a) * east) + math.sin(e) * up

    # -- 직렬화 ----------------------------------------------------------------------------------------------
    def to_meta(self):
        return {
            "version": FARFIELD_VERSION,
            "up": list(self.up),
            "north": list(self.north),
            "sun_elev_deg": self.sun_elev_deg,
            "sun_azim_deg": self.sun_azim_deg,
            "turbidity": self.turbidity,
            "albedo": self.albedo,
            "exposure": self.exposure,
            "camera_height_m": self.camera_height_m,
            "horizon_blend_deg": self.horizon_blend_deg,
            "ground_albedo_rgb": None if self.ground_albedo_rgb is None else list(self.ground_albedo_rgb),
            "size": self.size,
            "include_sun": self.include_sun,
        }

    @classmethod
    def from_meta(cls, m):
        if not isinstance(m, dict):
            raise FarFieldError("meta: dict 여야 한다")
        keys = set(m)
        if keys != set(_META_KEYS):
            raise FarFieldError(f"meta 키 불일치: 누락 {sorted(set(_META_KEYS) - keys)}, 모르는 키 {sorted(keys - set(_META_KEYS))}")
        if m["version"] != FARFIELD_VERSION:
            raise FarFieldError(f"meta version {m['version']!r} != {FARFIELD_VERSION!r}")
        kw = {k: m[k] for k in _META_KEYS if k != "version"}
        for k in ("up", "north", "ground_albedo_rgb"):
            if kw[k] is not None:
                if not isinstance(kw[k], (list, tuple)):
                    raise FarFieldError(f"meta {k}: 리스트여야 한다")
                kw[k] = tuple(kw[k])
        if kw["north"] is None:
            raise FarFieldError("meta north: 필요")
        s = cls(**kw)
        # 저장된 up/north 를 비트 그대로 복원한다(재정규화의 1ulp 차이 방지). 값이 거의 같을 때만.
        if (np.allclose(s.up, kw["up"], atol=1e-12) and np.allclose(s.north, kw["north"], atol=1e-12)):
            fixed = object.__new__(cls)
            for f in cls.__dataclass_fields__:
                object.__setattr__(fixed, f, getattr(s, f))
            object.__setattr__(fixed, "up", tuple(float(x) for x in kw["up"]))
            object.__setattr__(fixed, "north", tuple(float(x) for x in kw["north"]))
            return fixed
        return s

    # -- 빌드 ------------------------------------------------------------------------------------------------
    def build(self):
        d = texel_dirs(self.size)
        rgb = environment_radiance(self, d)
        if self.include_sun and self.sun_elev_deg > 0.0:
            rgb = rgb + _sun_disk_delta(self, d)
        out = np.ascontiguousarray(np.maximum(rgb, 0.0).astype(np.float32))
        return C.FarField(rgb=out)


def _sun_disk_delta(spec, d):
    s = spec.size
    om = texel_solid_angle(s)
    sd = spec.sun_dir()
    chord = np.linalg.norm(d - sd, axis=-1)
    ang = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    sigma = max(1.5 * math.sqrt(float(om.max())), math.radians(SUN_ANGULAR_RADIUS_DEG))
    k = np.exp(-0.5 * (ang / sigma) ** 2)
    k = k / float((k * om).sum())
    return sun_irradiance(spec)[None, None, :] * k[..., None]


# ---------------------------------------------------------------------------------------------------------------
# 하늘·조도·지면·환경
# ---------------------------------------------------------------------------------------------------------------
def sky_radiance(spec, d):
    du = _unit_dirs(d)
    cs, z = _hw_params_cached(spec.turbidity, spec.albedo, spec.sun_elev_deg)
    up = np.array(spec.up)
    ct = du @ up
    gam = np.arccos(np.clip(du @ spec.sun_dir(), -1.0, 1.0))
    out = np.stack([hw_shape(cs[k], ct, gam) * z[k] for k in range(3)], axis=-1) * spec.exposure
    return np.maximum(out, 0.0)


_E0_KLUX = 128.0
_SUN_LAMBDA_UM = (0.60, 0.55, 0.465)


def sun_irradiance(spec):
    e = spec.sun_elev_deg
    if e <= 0.0:
        return np.zeros(3)
    se = math.sin(math.radians(e))
    m = 1.0 / (se + 0.15 * (e + 3.885) ** -1.253)  # Kasten-Young
    beta = max(0.04608 * spec.turbidity - 0.04586, 0.0)
    out = np.empty(3)
    for k, lam in enumerate(_SUN_LAMBDA_UM):
        tau = 0.008735 * lam ** -4.08 + beta * lam ** -1.3
        out[k] = _E0_KLUX * math.exp(-tau * m)
    return out * spec.exposure


@functools.lru_cache(maxsize=64)
def _sky_irr_cached(spec):
    nt, nph = 256, 512
    north, east, up = spec._basis()
    th = (np.arange(nt) + 0.5) / nt * (math.pi / 2)
    ph = (np.arange(nph) + 0.5) / nph * 2 * math.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = (np.sin(TH)[..., None] * (np.cos(PH)[..., None] * north + np.sin(PH)[..., None] * east) + np.cos(TH)[..., None] * up)
    w = np.sin(TH) * np.cos(TH) * (math.pi / 2 / nt) * (2 * math.pi / nph)
    v = (sky_radiance(spec, d) * w[..., None]).sum((0, 1))
    v.setflags(write=False)
    return v


def sky_irradiance(spec):
    return _sky_irr_cached(spec).copy()


def ground_lit(spec):
    alb = np.array(spec.ground_albedo_rgb) if spec.ground_albedo_rgb is not None else np.full(3, spec.albedo)
    es = math.sin(math.radians(max(spec.sun_elev_deg, 0.0)))
    return alb / math.pi * (sky_irradiance(spec) + sun_irradiance(spec) * es)


def environment_radiance(spec, d):
    du = _unit_dirs(d)
    lead = du.shape[:-1]
    flat = du.reshape(-1, 3)
    up = np.array(spec.up)
    e = flat @ up
    below = e < 0.0
    dh = flat - e[:, None] * up
    hn = np.linalg.norm(dh, axis=-1, keepdims=True)
    degenerate = hn[:, 0] < 1e-12
    dh = np.where(degenerate[:, None], np.array(spec.north)[None, :], dh / np.where(hn > 0, hn, 1.0))
    eff = np.where(below[:, None], dh, flat)
    sky = sky_radiance(spec, eff)
    g = ground_lit(spec)
    delta = np.degrees(np.arcsin(np.clip(-e, 0.0, 1.0)))
    w = np.exp(-delta / spec.horizon_blend_deg)[:, None]
    out = np.where(below[:, None], w * sky + (1.0 - w) * g[None, :], sky)
    return out.reshape(lead + (3,))


def ground_distance(spec, d):
    du = _unit_dirs(d)
    e = du @ np.array(spec.up)
    down = e < -1e-12
    with np.errstate(divide="ignore"):
        return np.where(down, spec.camera_height_m / np.where(down, -e, 1.0), np.inf)


# ---------------------------------------------------------------------------------------------------------------
# 세계(Obs) 이음매
# ---------------------------------------------------------------------------------------------------------------
def up_world_from_obs(obs, up_cam):
    u = _vec3("up_cam", up_cam)
    v = np.asarray(obs.R, dtype=np.float64).T @ u
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 0.0:
        raise FarFieldError("up_cam: 영벡터")
    return v / n


def _horizontal(v, up):
    h = v - float(v @ up) * up
    n = float(np.linalg.norm(h))
    return (h / n, True) if n > 1e-6 else (h, False)


def spec_for_world(obs, up_cam, theta_in, **overrides):
    if "up" in overrides or "north" in overrides:
        raise FarFieldError("overrides: up·north 는 지정할 수 없다")
    up = up_world_from_obs(obs, up_cam)
    Rt = np.asarray(obs.R, dtype=np.float64).T
    north, ok = _horizontal(Rt @ np.array([0.0, 0.0, 1.0]), up)
    if not ok:
        north, ok = _horizontal(Rt @ np.array([1.0, 0.0, 0.0]), up)
        if not ok:
            north = None
    kw = dict(up=tuple(float(x) for x in up),
              north=None if north is None else tuple(float(x) for x in north),
              sun_elev_deg=DEFAULT_SUN_ELEV_DEG, sun_azim_deg=DEFAULT_SUN_AZIM_DEG)
    el = getattr(theta_in, "sun_elev_deg", None)
    az = getattr(theta_in, "sun_azim_deg", None)
    if el is not None:
        kw["sun_elev_deg"] = el
    if az is not None:
        kw["sun_azim_deg"] = az
    kw.update(overrides)
    return FarFieldSpec(**kw)
