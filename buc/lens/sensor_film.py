"""LNS-04: 센서·필름 (Poisson-Gauss 센서, ADC, 로그 부호화, 65^3 LUT, 할레이션, 비네팅, Newson 계열 그레인).

체인 순서: EV 게인 -> 비네팅 -> 할레이션 -> 센서(Poisson-Gauss, 만충전 클립, ADC) -> 로그 부호화 -> 65^3 LUT -> 그레인.
학습 파라미터 0, 순수 수식. torch 를 import 하지 않는다. 전역 RNG·시각 없음: 난수는 ``np.random.Generator`` 이며
스트림은 ``(seed, frame, 용도)`` 로 분리한다. 입력 배열은 바꾸지 않고 새 배열을 돌려준다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as _ndi

from buc.core import color as _color
from buc.core import contracts as _C
from buc.lens.theta import LUT_NAMES as _LUT_NAMES

__all__ = [
    "LUT_SIZE", "LOG_MIDGRAY", "LOG_STOPS", "GRAIN_SIGMA_MAX", "GRAIN_RADIUS_FRAC", "GRAIN_RADIUS_MIN_PX",
    "HALATION_THRESHOLD", "HALATION_SIGMA_FRAC", "HALATION_TINT", "HALATION_GAIN", "VIGNETTE_MAX_DROP",
    "FULL_WELL_PER_UM2", "FULL_WELL_MIN", "FULL_WELL_MAX", "READ_NOISE_E", "ADC_BITS",
    "SensorSpec", "sensor_spec_from_theta", "adc_quantize", "sensor_capture", "log_encode", "log_decode",
    "identity_lut", "make_lut", "apply_lut3d", "apply_look", "vignette_gain", "apply_vignette",
    "apply_halation", "halation_layer", "grain_field", "apply_sensor_film", "apply_sensor_film_frames",
]

LUT_SIZE = 65
LOG_MIDGRAY = 0.18
LOG_STOPS = (-8.0, 6.0)
GRAIN_SIGMA_MAX = 0.08
GRAIN_RADIUS_FRAC = 0.0012
GRAIN_RADIUS_MIN_PX = 0.75
HALATION_THRESHOLD = 0.8
HALATION_SIGMA_FRAC = 0.02
HALATION_TINT = (1.0, 0.35, 0.12)
HALATION_GAIN = 1.0
VIGNETTE_MAX_DROP = 0.8
FULL_WELL_PER_UM2 = 1500.0
FULL_WELL_MIN = 1500.0
FULL_WELL_MAX = 100000.0
READ_NOISE_E = 2.5
ADC_BITS = 12

_SENSOR_STREAM = 1
_GRAIN_STREAM = 2
_LAMBDA_CAP = 1.5      # 이 이상의 노출은 어차피 만충전 위: 포아송 평균을 여기서 자른다
_TINY = 1e-30


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _float_array(x, name="x") -> np.ndarray:
    a = np.asarray(x)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    return a


def _img3(x, name="rgb") -> np.ndarray:
    a = _float_array(x, name)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape) < 1:
        raise ValueError(f"{name}: (H,W,3) 이어야 한다 (shape {a.shape})")
    return a


def _int_arg(name, v, hi):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{name}: int 여야 한다 ({type(v).__name__})")
    v = int(v)
    if not 0 <= v < hi:
        raise ValueError(f"{name}={v}: [0, {hi}) 밖")
    return v


def _seed_frame(seed, frame):
    return _int_arg("seed", seed, 1 << 64), _int_arg("frame", frame, 1 << 32)


def _unit(name, v) -> float:
    if not _is_real(v):
        raise TypeError(f"{name}: 실수여야 한다")
    v = float(v)
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"{name}={v}: [0,1] 밖")
    return v


def _rng(seed, frame, stream) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, frame, stream])))


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------------------------------------------
# 센서
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class SensorSpec:
    full_well_e: float = 20000.0
    read_noise_e: float = 3.0
    bits: int = 12

    def __post_init__(self):
        for n, lo_ok in (("full_well_e", False), ("read_noise_e", True)):
            v = getattr(self, n)
            if not _is_real(v):
                raise ValueError(f"{n}: 실수여야 한다 ({v!r})")
            f = float(v)
            if not math.isfinite(f) or (f < 0.0 if lo_ok else f <= 0.0):
                raise ValueError(f"{n}={f}: {'>= 0' if lo_ok else '> 0'} 유한이어야 한다")
        b = self.bits
        if isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) or not 1 <= int(b) <= 16:
            raise ValueError(f"bits={b!r}: int 1..16 이어야 한다")


def sensor_spec_from_theta(theta, width_px) -> SensorSpec:
    if isinstance(width_px, (bool, np.bool_)) or not isinstance(width_px, (int, np.integer)) or width_px <= 0:
        raise ValueError(f"width_px={width_px!r}: 양의 정수여야 한다")
    pitch = float(theta.sensor_width_mm) * 1000.0 / float(width_px)
    fw = float(np.clip(FULL_WELL_PER_UM2 * pitch * pitch, FULL_WELL_MIN, FULL_WELL_MAX))
    return SensorSpec(full_well_e=fw, read_noise_e=READ_NOISE_E, bits=ADC_BITS)


def _bits_arg(bits) -> int:
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, np.integer)) or not 1 <= int(bits) <= 16:
        raise ValueError(f"bits={bits!r}: int 1..16 이어야 한다")
    return int(bits)


def adc_quantize(x, bits) -> np.ndarray:
    a = _float_array(x)
    m = float((1 << _bits_arg(bits)) - 1)
    q = np.round(np.clip(a.astype(np.float64), 0.0, 1.0) * m) / m
    return q.astype(np.float32)


def sensor_capture(rgb_lin, spec, *, seed=0, frame=0, noise=True) -> np.ndarray:
    a = _float_array(rgb_lin, "rgb_lin")
    if not isinstance(spec, SensorSpec):
        raise TypeError("spec: SensorSpec 여야 한다")
    seed, frame = _seed_frame(seed, frame)
    x = np.clip(a.astype(np.float64), 0.0, None)
    if not noise:
        return adc_quantize(np.minimum(x, 1.0), spec.bits)
    fw = float(spec.full_well_e)
    rng = _rng(seed, frame, _SENSOR_STREAM)
    e = rng.poisson(np.minimum(x, _LAMBDA_CAP) * fw).astype(np.float64)
    e = np.minimum(e, fw)                                  # 만충전에서 자르고, 그 뒤에 읽기 잡음
    if spec.read_noise_e > 0.0:
        e += rng.normal(0.0, float(spec.read_noise_e), size=e.shape)
    return adc_quantize(e / fw, spec.bits)


# ---------------------------------------------------------------------------------------------------------------
# 로그 부호화
# ---------------------------------------------------------------------------------------------------------------
def _ret(r64, dt):
    r = r64.astype(dt, copy=False)
    return r[()] if r.ndim == 0 else r


def log_encode(x):
    a = _float_array(x)
    lo, hi = LOG_STOPS
    v = a.astype(np.float64)
    e = (np.log2(np.maximum(v, _TINY) / LOG_MIDGRAY) - lo) / (hi - lo)
    return _ret(np.clip(e, 0.0, 1.0), a.dtype)


def log_decode(e):
    a = _float_array(e)
    lo, hi = LOG_STOPS
    return _ret(LOG_MIDGRAY * 2.0 ** (lo + a.astype(np.float64) * (hi - lo)), a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------------------------------------------
def identity_lut(size=LUT_SIZE) -> np.ndarray:
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError(f"size={size!r}: 2 이상의 정수여야 한다")
    g = np.linspace(0.0, 1.0, int(size))
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([r, gg, b], axis=-1).astype(np.float32)


def apply_lut3d(coded, lut) -> np.ndarray:
    c = _float_array(coded, "coded")
    L = np.asarray(lut)
    if L.ndim != 4 or L.shape[3] != 3 or not (L.shape[0] == L.shape[1] == L.shape[2]) or L.shape[0] < 2:
        raise ValueError(f"lut: (S,S,S,3) 정육면체(S>=2) 여야 한다 (shape {L.shape})")
    if L.dtype.kind != "f":
        raise TypeError("lut: 부동소수여야 한다")
    if c.ndim < 1 or c.shape[-1] != 3:
        raise ValueError(f"coded: 마지막 축이 3 이어야 한다 (shape {c.shape})")
    S = L.shape[0]
    flat = np.ascontiguousarray(L).reshape(-1, 3)
    p = np.clip(c.astype(np.float64).reshape(-1, 3), 0.0, 1.0) * (S - 1)
    i0 = np.minimum(np.floor(p).astype(np.intp), S - 2)
    f = p - i0
    base = (i0[:, 0] * S + i0[:, 1]) * S + i0[:, 2]
    out = np.zeros(p.shape, np.float64)
    for dr in (0, 1):
        wr = f[:, 0] if dr else 1.0 - f[:, 0]
        for dg in (0, 1):
            wg = f[:, 1] if dg else 1.0 - f[:, 1]
            for db in (0, 1):
                wb = f[:, 2] if db else 1.0 - f[:, 2]
                idx = base + dr * S * S + dg * S + db
                out += (wr * wg * wb)[:, None] * flat[idx]
    return out.astype(np.float32).reshape(c.shape)


def _luma(d):
    return d[..., 0] * 0.2126 + d[..., 1] * 0.7152 + d[..., 2] * 0.0722


def _sat(d, s):
    y = _luma(d)[..., None]
    return y + s * (d - y)


def _scurve(x, a):
    return x + a * (_smooth(x) - x)


def _grade_teal_orange(d):
    y = _luma(d)
    w_sh = 1.0 - _smooth(y / 0.6)
    w_hi = _smooth((y - 0.4) / 0.6)
    sh = np.array([-0.04, 0.0, 0.05])
    hi = np.array([0.05, 0.01, -0.06])
    return _scurve(d, 0.15) + w_sh[..., None] * sh + w_hi[..., None] * hi


def _grade_bleach_bypass(d):
    return _scurve(_sat(d, 0.55), 0.5)


def _grade_kodak_warm(d):
    y = _luma(d)
    w = 4.0 * np.clip(y, 0, 1) * (1.0 - np.clip(y, 0, 1))
    return _sat(_scurve(d, 0.2), 1.1) + w[..., None] * np.array([0.035, 0.008, -0.045])


def _grade_fuji_cool(d):
    y = _luma(d)
    w = 4.0 * np.clip(y, 0, 1) * (1.0 - np.clip(y, 0, 1))
    return _sat(_scurve(d, 0.1), 1.05) + w[..., None] * np.array([-0.035, 0.008, 0.045])


def _grade_mono_silver(d):
    g = _scurve(np.clip(_luma(d), 0.0, 1.0), 0.3)
    return np.repeat(g[..., None], 3, axis=-1)


def _grade_faded_matte(d):
    return 0.035 + 0.94 * _sat(d, 0.85) * 1.0


def _grade_neon_night(d):
    out = _scurve(_sat(d, 1.7), 0.2)
    y = _luma(d)
    w_sh = 1.0 - _smooth(y / 0.5)
    return out + w_sh[..., None] * np.array([0.0, -0.01, 0.04])


def _grade_cross_process(d):
    r = _scurve(d[..., 0], 0.8)
    g = np.power(d[..., 1], 0.85)
    b = 0.08 + 0.85 * np.power(d[..., 2], 0.75)
    return np.stack([r, g, b], axis=-1)


_GRADES = {
    "none": None, "neutral_709": None,
    "teal_orange": _grade_teal_orange, "bleach_bypass": _grade_bleach_bypass, "kodak_warm": _grade_kodak_warm,
    "fuji_cool": _grade_fuji_cool, "mono_silver": _grade_mono_silver, "faded_matte": _grade_faded_matte,
    "neon_night": _grade_neon_night, "cross_process": _grade_cross_process,
}
_LUT_CACHE: dict = {}


def _check_lut_name(name, size=LUT_SIZE):
    if not isinstance(name, str) or name not in _LUT_NAMES or name not in _GRADES:
        raise ValueError(f"lut {name!r}: {tuple(_LUT_NAMES)} 중 하나여야 한다")
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError(f"size={size!r}: 2 이상의 정수여야 한다")


def _lut_table(name, size=LUT_SIZE) -> np.ndarray:
    """캐시된 읽기 전용 표(내부용)."""
    _check_lut_name(name, size)
    key = (name, int(size))
    t = _LUT_CACHE.get(key)
    if t is None:
        e = np.linspace(0.0, 1.0, int(size))
        f = _color.linear_to_display(np.asarray(LOG_MIDGRAY * 2.0 ** (LOG_STOPS[0] + e * (LOG_STOPS[1] - LOG_STOPS[0])),
                                                np.float32)).astype(np.float64)
        r, g, b = np.meshgrid(f, f, f, indexing="ij")
        d = np.stack([r, g, b], axis=-1)
        fn = _GRADES[name]
        if fn is not None:
            d = fn(d)
        t = np.clip(d, 0.0, 1.0).astype(np.float32)
        t.setflags(write=False)
        _LUT_CACHE[key] = t
    return t


def make_lut(name, size=LUT_SIZE) -> np.ndarray:
    return _lut_table(name, size).copy()


def apply_look(rgb_lin, lut) -> np.ndarray:
    a = _float_array(rgb_lin, "rgb_lin")
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"rgb_lin: 마지막 축이 3 이어야 한다 (shape {a.shape})")
    if isinstance(lut, str):
        _check_lut_name(lut)
        if lut == "none":
            return _color.linear_to_display(a)
        table = _lut_table(lut)
    elif isinstance(lut, np.ndarray):
        table = lut
    else:
        raise TypeError("lut: 이름(str) 또는 ndarray 여야 한다")
    return np.clip(apply_lut3d(log_encode(a), table), 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------------------------
# 비네팅
# ---------------------------------------------------------------------------------------------------------------
def vignette_gain(h, w, strength) -> np.ndarray:
    for n, v in (("h", h), ("w", w)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{n}={v!r}: 양의 정수여야 한다")
    s = _unit("strength", strength)
    h, w = int(h), int(w)
    hd2 = ((w - 1) / 2.0) ** 2 + ((h - 1) / 2.0) ** 2
    if hd2 == 0.0:
        return np.ones((h, w), np.float32)
    yy = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
    xx = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    r2 = (yy[:, None] ** 2 + xx[None, :] ** 2) / hd2
    return (1.0 - VIGNETTE_MAX_DROP * s * r2).astype(np.float32)


def apply_vignette(rgb, strength) -> np.ndarray:
    a = _img3(rgb)
    s = _unit("strength", strength)
    x = a.astype(np.float32)
    if s == 0.0:
        return x.copy() if x is a else x
    return (x * vignette_gain(a.shape[0], a.shape[1], s)[..., None]).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------------------------
# 할레이션
# ---------------------------------------------------------------------------------------------------------------
def halation_layer(rgb_lin, amount) -> np.ndarray:
    a = _img3(rgb_lin, "rgb_lin")
    amt = _unit("amount", amount)
    h, w = a.shape[:2]
    if amt == 0.0:
        return np.zeros(a.shape, np.float32)
    hl = np.maximum(_color.luminance(a.astype(np.float64)) - HALATION_THRESHOLD, 0.0)
    if not np.any(hl > 0.0):
        return np.zeros(a.shape, np.float32)
    blur = _ndi.gaussian_filter(hl, HALATION_SIGMA_FRAC * w, mode="constant", truncate=5.0)
    layer = amt * HALATION_GAIN * np.maximum(blur, 0.0)[..., None] * np.asarray(HALATION_TINT, np.float64)
    return layer.astype(np.float32)


def apply_halation(rgb_lin, amount) -> np.ndarray:
    a = _img3(rgb_lin, "rgb_lin")
    layer = halation_layer(a, amount)
    return (a.astype(np.float32) + layer).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------------------------
# 그레인 (Newson 계열: 신호 의존 진폭 x 상관 입자 무늬)
# ---------------------------------------------------------------------------------------------------------------
def grain_field(u, *, strength, radius_px, seed=0, frame=0) -> np.ndarray:
    a = _float_array(u, "u")
    if a.ndim not in (2, 3) or min(a.shape) < 1:
        raise ValueError(f"u: (H,W) 또는 (H,W,C) 여야 한다 (shape {a.shape})")
    if a.min() < 0.0 or a.max() > 1.0:
        raise ValueError("u: [0,1] 표시값이어야 한다")
    s = _unit("strength", strength)
    if not _is_real(radius_px):
        raise TypeError("radius_px: 실수여야 한다")
    r = float(radius_px)
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError(f"radius_px={r}: 양의 유한값이어야 한다")
    seed, frame = _seed_frame(seed, frame)
    u32 = a.astype(np.float32)
    if s == 0.0:
        return u32.copy() if u32 is a else u32
    two_d = a.ndim == 2
    v = u32[..., None] if two_d else u32
    H, W, C = v.shape
    sg = r / 2.0                                  # 가우시안 입자: 자기상관 exp(-d^2 / (4 sg^2)) = exp(-d^2 / r^2)
    rad = int(4.0 * sg + 0.5)
    ax = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (ax / sg) ** 2)
    k /= k.sum()
    scale = 1.0 / float(np.sum(k * k))            # 2D 분산 = (sum k^2)^2 -> 단위 분산
    noise = _rng(seed, frame, _GRAIN_STREAM).standard_normal((H + 2 * rad, W + 2 * rad, C), dtype=np.float32)
    if rad > 0:
        noise = _ndi.convolve1d(noise, k.astype(np.float32), axis=0, mode="constant")
        noise = _ndi.convolve1d(noise, k.astype(np.float32), axis=1, mode="constant")
        noise = noise[rad:rad + H, rad:rad + W]
    amp = (s * GRAIN_SIGMA_MAX * 2.0) * np.sqrt(np.maximum(v.astype(np.float64) * (1.0 - v), 0.0))
    out = np.clip(v + (amp * noise * scale).astype(np.float32), 0.0, 1.0).astype(np.float32)
    return out[..., 0] if two_d else out


# ---------------------------------------------------------------------------------------------------------------
# 체인
# ---------------------------------------------------------------------------------------------------------------
def apply_sensor_film(rgb_lin, theta, *, seed=0, frame=0, spec=None, noise=True) -> np.ndarray:
    a = _img3(rgb_lin, "rgb_lin")
    seed, frame = _seed_frame(seed, frame)
    _check_lut_name(theta.lut)
    if spec is not None and not isinstance(spec, SensorSpec):
        raise TypeError("spec: SensorSpec 또는 None 이어야 한다")
    H, W = a.shape[:2]
    if spec is None:
        spec = sensor_spec_from_theta(theta, W)
    x = a.astype(np.float32) * np.float32(2.0 ** float(theta.ev))
    x = apply_vignette(x, theta.vignette)
    x = apply_halation(x, theta.halation)
    x = sensor_capture(x, spec, seed=seed, frame=frame, noise=noise)
    y = apply_look(x, theta.lut)
    if noise and float(theta.grain) > 0.0:
        radius = max(GRAIN_RADIUS_MIN_PX, GRAIN_RADIUS_FRAC * W)
        y = grain_field(y, strength=float(theta.grain), radius_px=radius, seed=seed, frame=frame)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def apply_sensor_film_frames(frames, theta, *, seed=0, spec=None, noise=True):
    if not isinstance(frames, _C.Frames):
        raise TypeError("frames: Frames 여야 한다")
    _check_lut_name(theta.lut)
    rgb = frames.rgb
    out = np.empty(rgb.shape, np.float32)
    for i in range(rgb.shape[0]):
        lin = _color.srgb_to_linear(rgb[i])
        out[i] = apply_sensor_film(lin, theta, seed=seed, frame=i, spec=spec, noise=noise)
    return _C.Frames(rgb=out, fps=frames.fps, aov={k: np.array(v, copy=True) for k, v in frames.aov.items()},
                     shot_hash=frames.shot_hash, world_hash=frames.world_hash)
