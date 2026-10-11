"""RD-14 R10 관측 대역 지도: 화소 풋프린트 phi_eff, l_obs 경/부분, w(nu)/nu_obs, 관측 집합 O.

입력이 어디서 어느 레벨까지 "진짜 정보"를 갖는지 숫자로 낸다(PARADIGM 2.10, 2단 소유). 경소유 tau_l >= 2 phi_eff,
부분 소유 phi_eff <= tau_l < 2 phi_eff. numpy / scipy 만 쓴다(torch·cv2·skimage·yaml 없음). 순수·결정론·입력 불변,
import 부작용·전역 RNG 없음. ``buc.world`` 는 import 하지 않는다(격자 상수는 world/strata.py 와 같은 값을 복제).
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import j1 as _j1

from buc.core import geom as _geom

__all__ = [
    "LEVEL_RANGE", "H0_M", "BLOCK_CELLS", "OCT_HI", "OCT_CENTER", "NU_GRID_N", "W_TRUST", "NO_LEVEL",
    "OWN_INPUT", "OWN_INPUT_PART", "OWN_INPUT_REFL",
    "ObsMapError", "NoiseModel", "ObsOpts", "ObsMap", "ObsBlock", "ObsSet",
    "texel_pitch_m", "cell_size_m", "block_extent_m", "footprint_m", "jpeg_band_factor", "effective_footprint",
    "levels_from_footprint", "owner_of_level", "defocus_mtf", "motion_mtf", "w_of_nu", "nu_obs_cutoff",
    "mrh_nu_cap", "build_obs_map", "build_obs_set",
]

LEVEL_RANGE = (-8, 24)
H0_M = 64.0
BLOCK_CELLS = 16
OCT_HI = (0.5, 0.25, 0.125, 0.0625)
OCT_CENTER = tuple(h / math.sqrt(2.0) for h in OCT_HI)
NU_GRID_N = 512
W_TRUST = 0.5
NO_LEVEL = -128
OWN_INPUT = 1
OWN_INPUT_PART = 2
OWN_INPUT_REFL = 3

_EPS = 1e-9
_R_MIN = 1.0 / 16.0
_QUANT = 64.0          # (c, m) 양자화: 1/64 px
_CHUNK = 4096


class ObsMapError(ValueError):
    """RD-14 검증 실패."""


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _num(x, name) -> float:
    if not _is_real(x):
        raise TypeError(f"{name}: 실수여야 한다 ({type(x).__name__})")
    return float(x)


def _fin(x, name) -> float:
    v = _num(x, name)
    if not math.isfinite(v):
        raise ObsMapError(f"{name}={v}: 유한해야 한다")
    return v


def _int(x, name) -> int:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Integral):
        raise TypeError(f"{name}: 정수여야 한다 ({type(x).__name__})")
    return int(x)


def _num_array(x, name) -> np.ndarray:
    """수치(실수) ndarray 또는 스칼라 -> float64 배열. bool/complex/객체는 TypeError."""
    if isinstance(x, np.ndarray):
        if x.dtype.kind not in "iuf":
            raise TypeError(f"{name}: 수치 dtype 이어야 한다 ({x.dtype})")
        return x.astype(np.float64)
    if _is_real(x):
        return np.asarray(float(x), dtype=np.float64)
    raise TypeError(f"{name}: 실수 또는 ndarray 여야 한다 ({type(x).__name__})")


def _level_check(level, name="level") -> int:
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, numbers.Integral):
        raise TypeError(f"{name}: 정수여야 한다 ({type(level).__name__})")
    level = int(level)
    if not (LEVEL_RANGE[0] <= level <= LEVEL_RANGE[1]):
        raise ObsMapError(f"{name}={level}: LEVEL_RANGE {LEVEL_RANGE} 밖")
    return level


def _scalar_out(a):
    a = np.asarray(a)
    return a[()] if a.ndim == 0 else a


# ---------------------------------------------------------------------------------------------------------------
# 격자 상수
# ---------------------------------------------------------------------------------------------------------------
def cell_size_m(level) -> float:
    return H0_M * 2.0 ** (-_level_check(level))


def texel_pitch_m(level) -> float:
    return cell_size_m(level) / 8.0


def block_extent_m(level) -> float:
    return BLOCK_CELLS * cell_size_m(level)


# ---------------------------------------------------------------------------------------------------------------
# 풋프린트·레벨
# ---------------------------------------------------------------------------------------------------------------
def footprint_m(depth_m, f_px, cos_theta=1.0):
    """phi_in = z / (f cos). z 비유한·<=0 -> NaN."""
    f = _fin(f_px, "f_px")
    if f <= 0.0:
        raise ObsMapError(f"f_px={f}: > 0 이어야 한다")
    z = _num_array(depth_m, "depth_m")
    c = _num_array(cos_theta, "cos_theta")
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0) or np.any(c > 1.0):
        raise ObsMapError("cos_theta: (0, 1] 이어야 한다")
    z = np.where(np.isfinite(z) & (z > 0.0), z, np.nan)
    with np.errstate(invalid="ignore"):
        return _scalar_out(z / (f * c))


def jpeg_band_factor(quality) -> float:
    if quality is None:
        return 1.0
    q = _num(quality, "quality")
    if not (1.0 <= q <= 100.0):          # NaN 도 여기서 걸린다
        raise ObsMapError(f"quality={q}: [1, 100] 밖")
    if q >= 90.0:
        return 1.0
    return float(min(4.0, 2.0 ** ((90.0 - q) / 40.0)))


def _nonneg_map(x, name) -> np.ndarray:
    a = _num_array(x, name)
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ObsMapError(f"{name}: 유한한 0 이상이어야 한다")
    return a


def effective_footprint(phi_in, *, r_ratio=1.0, coc_px=0.0, motion_px=0.0, q_jpeg=1.0, nu_cap=None):
    """phi_eff = phi_in * max(1, 1/r_ratio, c/1.5, m/1.5, q_jpeg, 0.5/nu_cap). NaN 전파."""
    phi = _num_array(phi_in, "phi_in")
    with np.errstate(invalid="ignore"):
        if np.any(phi <= 0.0):
            raise ObsMapError("phi_in: > 0 이어야 한다")
    r = _fin(r_ratio, "r_ratio")
    if not (_R_MIN <= r <= 1.0):
        raise ObsMapError(f"r_ratio={r}: [1/16, 1] 밖")
    c = _nonneg_map(coc_px, "coc_px")
    m = _nonneg_map(motion_px, "motion_px")
    q = _fin(q_jpeg, "q_jpeg")
    if q < 1.0:
        raise ObsMapError(f"q_jpeg={q}: >= 1 이어야 한다")
    fac = np.maximum(np.maximum(1.0, 1.0 / r), np.maximum(c / 1.5, m / 1.5))
    fac = np.maximum(fac, q)
    if nu_cap is not None:
        cap = _num_array(nu_cap, "nu_cap")
        if not np.all(np.isfinite(cap)) or np.any(cap <= 0.0) or np.any(cap > 0.5):
            raise ObsMapError("nu_cap: (0, 0.5] 이어야 한다")
        fac = np.maximum(fac, 0.5 / cap)
    return _scalar_out(phi * fac)


def _check_part_extra(part_extra) -> float:
    e = _num(part_extra, "part_extra")
    if not (0.0 <= e <= 1.0):
        raise ObsMapError(f"part_extra={e}: [0, 1] 밖")
    return e


def levels_from_footprint(phi_eff, *, part_extra=0.0):
    """(hard, part) 정수 레벨. a = log2(4/phi); hard = floor(a + 1e-9), part = floor(a + 1 + extra + 1e-9)."""
    e = _check_part_extra(part_extra)
    phi = _num_array(phi_eff, "phi_eff")
    if not np.all(np.isfinite(phi)) or np.any(phi <= 0.0):
        raise ObsMapError("phi_eff: 유한한 양수여야 한다")
    a = np.log2(4.0 / phi)
    lo, hi = LEVEL_RANGE
    hard = np.clip(np.floor(a + _EPS), lo, hi).astype(np.int64)
    part = np.clip(np.floor(a + 1.0 + e + _EPS), lo, hi).astype(np.int64)
    return _scalar_out(hard), _scalar_out(part)


def owner_of_level(level, phi_eff, *, part_extra=0.0):
    """1: tau >= 2 phi, 2: tau >= phi 2^-extra, 0. 동률은 소유(상대 1e-9)."""
    l = _level_check(level)
    e = _check_part_extra(part_extra)
    phi = _num_array(phi_eff, "phi_eff")
    tau = 8.0 * 2.0 ** (-l)
    with np.errstate(invalid="ignore"):
        hard = tau >= 2.0 * phi * (1.0 - _EPS)
        part = tau >= phi * 2.0 ** (-e) * (1.0 - _EPS)
    out = np.where(hard, 1, np.where(part, 2, 0)).astype(np.uint8)
    return _scalar_out(out)


# ---------------------------------------------------------------------------------------------------------------
# MTF, 잡음, w(nu), nu_obs
# ---------------------------------------------------------------------------------------------------------------
def _mtf_param(x, name):
    a = _num_array(x, name)
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ObsMapError(f"{name}: 유한한 0 이상이어야 한다")
    return a


def defocus_mtf(nu, coc_px):
    """원판 착란원 MTF |2 J1(pi c nu) / (pi c nu)|."""
    n = _num_array(nu, "nu")
    c = _mtf_param(coc_px, "coc_px")
    x = np.pi * c * n
    safe = np.where(x == 0.0, 1.0, x)
    return _scalar_out(np.where(x == 0.0, 1.0, np.abs(2.0 * _j1(safe) / safe)))


def motion_mtf(nu, length_px):
    """직선 모션 MTF |sinc(L nu)|."""
    n = _num_array(nu, "nu")
    L = _mtf_param(length_px, "length_px")
    return _scalar_out(np.abs(np.sinc(L * n)))


@dataclass(frozen=True)
class NoiseModel:
    nsr_ref: float = 0.01
    nu_ref: float = 0.25
    alpha: float = 2.0
    jpeg_gain: float = 0.04

    def __post_init__(self):
        for n in ("nsr_ref", "nu_ref", "alpha", "jpeg_gain"):
            _fin(getattr(self, n), n)
        if self.nsr_ref <= 0 or self.nu_ref <= 0 or self.alpha < 0 or self.jpeg_gain < 0:
            raise ObsMapError("NoiseModel: nsr_ref, nu_ref > 0, alpha, jpeg_gain >= 0 이어야 한다")

    def nsr(self, nu, q_jpeg=1.0):
        n = _num_array(nu, "nu")
        return _scalar_out((self.nsr_ref + self.jpeg_gain * (float(q_jpeg) - 1.0)) * (n / self.nu_ref) ** self.alpha)


def _check_noise(noise) -> NoiseModel:
    if noise is None:
        return NoiseModel()
    if not isinstance(noise, NoiseModel):
        raise TypeError("noise: NoiseModel 이어야 한다")
    return noise


def _w_smooth(nu, c, m, q_jpeg, noise):
    H = np.asarray(defocus_mtf(nu, c)) * np.asarray(motion_mtf(nu, m))
    N = np.asarray(noise.nsr(nu, q_jpeg))
    h2 = H * H
    return h2 / (h2 + N)


def w_of_nu(nu, *, r_ratio=1.0, coc_px=0.0, motion_px=0.0, q_jpeg=1.0, noise=None):
    """w = |H|^2 / (|H|^2 + NSR); nu > 0.5 r_ratio -> 0 (r_eff 절단)."""
    n = _num_array(nu, "nu")
    if not np.all(np.isfinite(n)) or np.any(n < 0.0) or np.any(n > 0.5):
        raise ObsMapError("nu: [0, 0.5] 이어야 한다")
    r = _fin(r_ratio, "r_ratio")
    if not (_R_MIN <= r <= 1.0):
        raise ObsMapError(f"r_ratio={r}: [1/16, 1] 밖")
    q = _fin(q_jpeg, "q_jpeg")
    if q < 1.0:
        raise ObsMapError("q_jpeg: >= 1 이어야 한다")
    noise = _check_noise(noise)
    w = _w_smooth(n, coc_px, motion_px, q, noise)
    w = np.where(n > 0.5 * r * (1.0 + 1e-12), 0.0, w)
    return _scalar_out(w)


def _nu_obs_flat(r, c, m, q, noise):
    """c, m 1-D 배열 -> w_smooth 가 처음 0.5 아래로 내려가는 지점(512 구간 격자 + 선형 보간), r 로 상한."""
    g = np.linspace(0.0, 0.5, NU_GRID_N + 1)
    out = np.empty(c.shape[0], dtype=np.float64)
    for s in range(0, c.shape[0], _CHUNK):
        cc, mm = c[s:s + _CHUNK, None], m[s:s + _CHUNK, None]
        w = _w_smooth(g[None, :], cc, mm, q, noise)
        below = w[:, 1:] < W_TRUST
        has = below.any(axis=1)
        i = below.argmax(axis=1) + 1
        rows = np.arange(w.shape[0])
        w0, w1 = w[rows, i - 1], w[rows, i]
        g0, g1 = g[i - 1], g[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            nw = g0 + (W_TRUST - w0) / (w1 - w0) * (g1 - g0)
        out[s:s + _CHUNK] = np.where(has, nw, 0.5)
    return np.minimum(out, 0.5 * r)


def nu_obs_cutoff(*, r_ratio=1.0, coc_px=0.0, motion_px=0.0, q_jpeg=1.0, noise=None):
    """min(0.5 r_ratio, w_smooth = 0.5 의 첫 교차). c, m 방송, 스칼라면 float."""
    r = _fin(r_ratio, "r_ratio")
    if not (_R_MIN <= r <= 1.0):
        raise ObsMapError(f"r_ratio={r}: [1/16, 1] 밖")
    q = _fin(q_jpeg, "q_jpeg")
    if q < 1.0:
        raise ObsMapError("q_jpeg: >= 1 이어야 한다")
    noise = _check_noise(noise)
    c = _mtf_param(coc_px, "coc_px")
    m = _mtf_param(motion_px, "motion_px")
    c, m = np.broadcast_arrays(c, m)
    shape = c.shape
    out = _nu_obs_flat(r, np.ascontiguousarray(c).ravel(), np.ascontiguousarray(m).ravel(), q, noise)
    out = out.reshape(shape)
    return float(out) if out.ndim == 0 else out


def mrh_nu_cap(mrh_obs, tau=0.5):
    """옥타브 b 관측 <=> p_b > tau. 가장 낮은 대역(b=3)부터 연속 관측인 최소 b*: cap = 0.5 2^-b*."""
    if not isinstance(mrh_obs, np.ndarray):
        raise TypeError("mrh_obs: ndarray 여야 한다")
    if mrh_obs.dtype.kind not in "iuf":
        raise TypeError(f"mrh_obs: 수치 dtype 이어야 한다 ({mrh_obs.dtype})")
    t = _fin(tau, "tau")
    if not (0.0 < t < 1.0):
        raise ObsMapError(f"tau={t}: (0, 1) 밖")
    if mrh_obs.ndim != 3 or mrh_obs.shape[0] != 4 or mrh_obs.shape[1] < 1 or mrh_obs.shape[2] < 1:
        raise ObsMapError(f"mrh_obs: shape (4, h, w), h,w >= 1 이어야 한다 (got {mrh_obs.shape})")
    p = mrh_obs.astype(np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ObsMapError("mrh_obs: [0, 1] 유한값이어야 한다")
    flags = p > t
    chain = np.ones(flags.shape[1:], dtype=bool)
    bstar = np.full(flags.shape[1:], 4, dtype=np.int64)
    for b in (3, 2, 1, 0):
        chain &= flags[b]
        bstar = np.where(chain, b, bstar)
    return 0.5 * 2.0 ** (-bstar.astype(np.float64))


# ---------------------------------------------------------------------------------------------------------------
# 옵션·지도
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ObsOpts:
    block_px: int = 16
    block_quantile: float = 0.1
    cos_min: float = 0.1
    part_extra: float = 0.0
    scale_margin_k: float = 0.0
    mrh_tau: float = 0.5
    noise: NoiseModel = field(default_factory=NoiseModel)

    def __post_init__(self):
        if _int(self.block_px, "block_px") < 1:
            raise ObsMapError("block_px >= 1")
        q = _fin(self.block_quantile, "block_quantile")
        if not (0.0 <= q <= 0.5):
            raise ObsMapError("block_quantile: [0, 0.5]")
        c = _fin(self.cos_min, "cos_min")
        if not (0.0 < c <= 1.0):
            raise ObsMapError("cos_min: (0, 1]")
        _check_part_extra(self.part_extra)
        if _fin(self.scale_margin_k, "scale_margin_k") < 0.0:
            raise ObsMapError("scale_margin_k >= 0")
        t = _fin(self.mrh_tau, "mrh_tau")
        if not (0.0 < t < 1.0):
            raise ObsMapError("mrh_tau: (0, 1)")
        _check_noise(self.noise)
        if self.noise is None:
            raise TypeError("noise: NoiseModel 이어야 한다")


@dataclass(frozen=True, eq=False)
class ObsMap:
    size: tuple
    valid: np.ndarray
    phi_in: np.ndarray
    phi_eff: np.ndarray
    l_hard: np.ndarray
    l_part: np.ndarray
    nu_obs: np.ndarray
    block_px: int
    blk_valid: np.ndarray
    blk_l_hard: np.ndarray
    blk_l_part: np.ndarray
    blk_nu_obs: np.ndarray
    blk_w: np.ndarray
    r_ratio: float
    q_jpeg: float
    sigma_log_s: float
    l_sigma_octaves: float
    used_mrh: bool
    warnings: tuple

    def owner_at(self, level) -> np.ndarray:
        l = _level_check(level)
        out = np.where(self.l_hard >= l, 1, np.where(self.l_part >= l, 2, 0)).astype(np.uint8)
        out[~self.valid] = 0
        return out

    def block_owner_at(self, level) -> np.ndarray:
        l = _level_check(level)
        out = np.where(self.blk_l_hard >= l, 1, np.where(self.blk_l_part >= l, 2, 0)).astype(np.uint8)
        out[~self.blk_valid] = 0
        return out

    def obs_levels(self) -> tuple:
        if not self.valid.any():
            return (0.0, 0.0)
        hard = max(0.0, float(np.median(self.l_hard[self.valid])))
        part = max(hard, max(0.0, float(np.median(self.l_part[self.valid]))))
        return (hard, part)

    def to_obs_fields(self) -> dict:
        hard, part = self.obs_levels()
        return {"l_obs_hard": hard, "l_obs_part": part}

    def to_meta(self) -> dict:
        meta = {
            "size": [int(self.size[0]), int(self.size[1])],
            "block_px": int(self.block_px),
            "valid_frac": float(self.valid.mean()) if self.valid.size else 0.0,
            "r_ratio": float(self.r_ratio),
            "q_jpeg": float(self.q_jpeg),
            "mrh": bool(self.used_mrh),
            "sigma_log_s": float(self.sigma_log_s),
            "l_sigma_octaves": float(self.l_sigma_octaves),
            "warnings": list(self.warnings),
        }
        hard, part = self.obs_levels()
        meta["l_obs_hard"], meta["l_obs_part"] = hard, part
        json.dumps(meta, allow_nan=False)
        return meta


def _check_K(K) -> np.ndarray:
    if not isinstance(K, np.ndarray):
        raise TypeError("K: ndarray 여야 한다")
    if K.dtype.kind not in "iuf":
        raise TypeError(f"K: 수치 dtype 이어야 한다 ({K.dtype})")
    if K.shape != (3, 3):
        raise ObsMapError(f"K: shape (3, 3) 이어야 한다 (got {K.shape})")
    K = K.astype(np.float64)
    if not np.all(np.isfinite(K)):
        raise ObsMapError("K: 비유한")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ObsMapError("K: fx, fy > 0 이어야 한다")
    return K


def _deg_map(x, shape, name) -> np.ndarray:
    a = _nonneg_map(x, name)
    if a.ndim == 0:
        return np.full(shape, float(a))
    if a.shape != shape:
        raise ObsMapError(f"{name}: shape {a.shape} != {shape}")
    return a


def _block_reduce_sorted(vals, ok, bp, q):
    """블록별 유효 값 오름차순 v_(floor(q(n-1))). 반환 (blk_valid, value)."""
    Hh, Ww = vals.shape
    gh, gw = -(-Hh // bp), -(-Ww // bp)
    pad = np.full((gh * bp, gw * bp), np.inf)
    pad[:Hh, :Ww] = np.where(ok, vals, np.inf)
    t = pad.reshape(gh, bp, gw, bp).transpose(0, 2, 1, 3).reshape(gh, gw, bp * bp)
    t = np.sort(t, axis=-1)
    n = np.isfinite(t).sum(axis=-1)
    idx = np.floor(q * np.maximum(n - 1, 0)).astype(np.int64)
    val = np.take_along_axis(t, idx[..., None], axis=-1)[..., 0]
    return n > 0, val


def build_obs_map(depth_m, K, *, r_ratio=1.0, coc_px=None, motion_px=None, jpeg_quality=None, normal_cam=None,
                  valid=None, mrh_obs=None, sigma_log_s=0.0, opts=None) -> ObsMap:
    # ---- 검증 (계산 전에 끝낸다)
    if not isinstance(depth_m, np.ndarray):
        raise TypeError("depth_m: ndarray 여야 한다")
    if depth_m.dtype.kind not in "iuf":
        raise TypeError(f"depth_m: 수치 dtype 이어야 한다 ({depth_m.dtype})")
    if depth_m.ndim != 2 or depth_m.size == 0:
        raise ObsMapError(f"depth_m: 비어 있지 않은 (H, W) 여야 한다 (got {depth_m.shape})")
    K = _check_K(K)
    if opts is None:
        opts = ObsOpts()
    elif not isinstance(opts, ObsOpts):
        raise TypeError("opts: ObsOpts 여야 한다")
    shape = depth_m.shape
    Hh, Ww = shape
    r = _fin(r_ratio, "r_ratio")
    if not (_R_MIN <= r <= 1.0):
        raise ObsMapError(f"r_ratio={r}: [1/16, 1] 밖")
    c_map = _deg_map(0.0 if coc_px is None else coc_px, shape, "coc_px")
    m_map = _deg_map(0.0 if motion_px is None else motion_px, shape, "motion_px")
    qj = jpeg_band_factor(jpeg_quality)
    sig = _fin(sigma_log_s, "sigma_log_s")
    if sig < 0.0:
        raise ObsMapError("sigma_log_s >= 0")
    vmask = None
    if valid is not None:
        if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
            raise TypeError("valid: bool ndarray 여야 한다")
        if valid.shape != shape:
            raise ObsMapError(f"valid: shape {valid.shape} != {shape}")
        vmask = valid
    nrm = None
    if normal_cam is not None:
        if not isinstance(normal_cam, np.ndarray):
            raise TypeError("normal_cam: ndarray 여야 한다")
        if normal_cam.dtype.kind not in "iuf":
            raise TypeError("normal_cam: 수치 dtype 이어야 한다")
        if normal_cam.shape != shape + (3,):
            raise ObsMapError(f"normal_cam: shape {normal_cam.shape} != {shape + (3,)}")
        nrm = normal_cam.astype(np.float64)
    cap = None
    if mrh_obs is not None:
        cap_small = mrh_nu_cap(mrh_obs, tau=opts.mrh_tau)
        gh_, gw_ = cap_small.shape
        si = np.minimum(gh_ - 1, np.floor((np.arange(Hh) + 0.5) * gh_ / Hh).astype(np.int64))
        sj = np.minimum(gw_ - 1, np.floor((np.arange(Ww) + 0.5) * gw_ / Ww).astype(np.int64))
        cap = cap_small[si][:, sj]

    # ---- 유효 화소
    z = depth_m.astype(np.float64)
    ok = np.isfinite(z) & (z > 0.0)
    if vmask is not None:
        ok &= vmask
    warnings = []
    if not ok.any():
        warnings.append("no_valid_depth")

    cos = np.ones(shape)
    if nrm is not None:
        if not np.all(np.isfinite(nrm[ok])):
            raise ObsMapError("normal_cam: 유효 화소에 비유한 값")
        nn = np.linalg.norm(nrm, axis=-1)
        if np.any(nn[ok] <= 0.0):
            raise ObsMapError("normal_cam: 유효 화소에 영벡터")
        v, u = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
        yd = (v - K[1, 2]) / K[1, 1]
        xd = (u - K[0, 2] - K[0, 1] * yd) / K[0, 0]
        dn = np.sqrt(xd * xd + yd * yd + 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dot = np.abs(nrm[..., 0] * xd + nrm[..., 1] * yd + nrm[..., 2]) / (dn * nn)
        cos = np.clip(np.where(ok, dot, 1.0), opts.cos_min, 1.0)

    f = 0.5 * (K[0, 0] + K[1, 1])
    zz = np.where(ok, z, 1.0)
    phi_in = zz / (f * cos) * math.exp(opts.scale_margin_k * sig)
    phi_eff = np.asarray(effective_footprint(phi_in, r_ratio=r, coc_px=c_map, motion_px=m_map, q_jpeg=qj,
                                             nu_cap=cap))
    lh, lp = levels_from_footprint(phi_eff, part_extra=opts.part_extra)
    lh = np.where(ok, lh, NO_LEVEL).astype(np.int16)
    lp = np.where(ok, lp, NO_LEVEL).astype(np.int16)

    # ---- nu_obs, w(nu) : (c, m) 을 1/64 px 로 양자화한 고유 쌍만 평가
    ci = np.rint(c_map[ok] * _QUANT).astype(np.int64)
    mi = np.rint(m_map[ok] * _QUANT).astype(np.int64)
    nu_full = np.full(shape, np.nan)
    wpix = np.zeros((4,) + shape)
    if ok.any():
        key = ci * (int(mi.max()) + 1) + mi
        uniq, inv = np.unique(key, return_inverse=True)
        inv = inv.ravel()
        first = np.zeros(uniq.shape[0], dtype=np.int64)
        first[inv[::-1]] = np.arange(inv.shape[0] - 1, -1, -1)
        uc, um = ci[first] / _QUANT, mi[first] / _QUANT
        nu_pair = _nu_obs_flat(r, uc, um, qj, opts.noise)
        w_pair = np.stack([np.asarray(w_of_nu(np.full(uc.shape, OCT_CENTER[b]), r_ratio=r, coc_px=uc,
                                              motion_px=um, q_jpeg=qj, noise=opts.noise)) for b in range(4)])
        nu_pix = nu_pair[inv]
        cap_ok = cap[ok] if cap is not None else np.full(inv.shape, 0.5)
        if cap is not None and np.any(nu_pix > cap_ok):
            warnings.append("mrh_veto")
        nu_full[ok] = np.minimum(nu_pix, cap_ok)
        for b in range(4):
            wpix[b][ok] = np.where(OCT_CENTER[b] > cap_ok, 0.0, w_pair[b][inv])
    if sig > 0.35:
        warnings.append("scale_uncertain")

    # ---- 블록 요약
    bp = int(opts.block_px)
    q = float(opts.block_quantile)
    blk_valid, bh = _block_reduce_sorted(lh.astype(np.float64), ok, bp, q)
    _, bpart = _block_reduce_sorted(lp.astype(np.float64), ok, bp, q)
    _, bnu = _block_reduce_sorted(np.where(ok, nu_full, 0.0), ok, bp, q)
    blk_l_hard = np.where(blk_valid, bh, NO_LEVEL).astype(np.int16)
    blk_l_part = np.where(blk_valid, bpart, NO_LEVEL).astype(np.int16)
    blk_nu = np.where(blk_valid, bnu, 0.0).astype(np.float32)
    gh, gw = blk_valid.shape
    cnt = np.zeros((gh * bp, gw * bp))
    cnt[:Hh, :Ww] = ok
    cnt = cnt.reshape(gh, bp, gw, bp).sum(axis=(1, 3))
    blk_w = np.zeros((4, gh, gw))
    for b in range(4):
        pad = np.zeros((gh * bp, gw * bp))
        pad[:Hh, :Ww] = wpix[b]
        s = pad.reshape(gh, bp, gw, bp).sum(axis=(1, 3))
        blk_w[b] = np.where(cnt > 0, s / np.maximum(cnt, 1.0), 0.0)

    return ObsMap(
        size=(Hh, Ww), valid=ok.copy(),
        phi_in=np.where(ok, phi_in, np.nan).astype(np.float32),
        phi_eff=np.where(ok, phi_eff, np.nan).astype(np.float32),
        l_hard=lh, l_part=lp, nu_obs=nu_full.astype(np.float32),
        block_px=bp, blk_valid=blk_valid, blk_l_hard=blk_l_hard, blk_l_part=blk_l_part,
        blk_nu_obs=blk_nu, blk_w=blk_w.astype(np.float32),
        r_ratio=r, q_jpeg=qj, sigma_log_s=sig, l_sigma_octaves=sig / math.log(2.0),
        used_mrh=cap is not None, warnings=tuple(sorted(set(warnings))),
    )


# ---------------------------------------------------------------------------------------------------------------
# 관측 집합 O
# ---------------------------------------------------------------------------------------------------------------
_KINDS = ("direct", "refl")


def _popcount(a: np.ndarray) -> int:
    return int(np.unpackbits(np.ascontiguousarray(a, dtype="<u8").view(np.uint8)).sum())


@dataclass(frozen=True, eq=False)
class ObsBlock:
    hard: np.ndarray      # uint64[64]
    part: np.ndarray      # uint64[64]


@dataclass(frozen=True, eq=False)
class ObsSet:
    levels: tuple
    kind: str
    stride: int
    blocks: dict

    def keys(self, l=None) -> list:
        ks = sorted(self.blocks)
        if l is None:
            return ks
        return [k for k in ks if k[0] == l]

    def n_cells(self, l=None, state=None) -> int:
        if state not in (None, "hard", "part"):
            raise ObsMapError(f"state={state!r}: None|'hard'|'part'")
        total = 0
        for k in self.keys(l):
            b = self.blocks[k]
            if state in (None, "hard"):
                total += _popcount(b.hard)
            if state in (None, "part"):
                total += _popcount(b.part)
        return total

    def cell_state(self, points, l) -> np.ndarray:
        l = _level_check(l, "l")
        if not isinstance(points, np.ndarray):
            raise TypeError("points: ndarray 여야 한다")
        if points.dtype.kind not in "iuf":
            raise TypeError("points: 수치 dtype 이어야 한다")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ObsMapError(f"points: (N, 3) 이어야 한다 (got {points.shape})")
        p = points.astype(np.float64)
        if not np.all(np.isfinite(p)):
            raise ObsMapError("points: 비유한")
        out = np.zeros(p.shape[0], dtype=np.uint8)
        if p.shape[0] == 0 or l not in self.levels:
            return out
        cells = np.floor(p / cell_size_m(l)).astype(np.int64)
        blk = cells // BLOCK_CELLS
        loc = cells % BLOCK_CELLS
        idx = loc[:, 0] + 16 * loc[:, 1] + 256 * loc[:, 2]
        ub, inv = np.unique(blk, axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(ub.shape[0] + 1))
        for j in range(ub.shape[0]):
            key = (l, int(ub[j, 0]), int(ub[j, 1]), int(ub[j, 2]))
            b = self.blocks.get(key)
            if b is None:
                continue
            sel = order[bounds[j]:bounds[j + 1]]
            wi = (idx[sel] >> 6).astype(np.int64)
            bi = (idx[sel] & 63).astype(np.uint64)
            hb = (b.hard[wi] >> bi) & np.uint64(1)
            pb = (b.part[wi] >> bi) & np.uint64(1)
            out[sel] = np.where(hb == 1, 1, np.where(pb == 1, 2, 0)).astype(np.uint8)
        return out

    def block_own(self, key) -> int:
        key = tuple(key)
        if key not in self.blocks:
            raise KeyError(key)
        if self.kind == "refl":
            return OWN_INPUT_REFL
        return OWN_INPUT_PART if self.blocks[key].part.any() else OWN_INPUT

    def union(self, other) -> "ObsSet":
        if not isinstance(other, ObsSet):
            raise TypeError("union: ObsSet 이어야 한다")
        if other.kind != self.kind or other.levels != self.levels:
            raise ObsMapError("union: kind·levels 가 같아야 한다")
        blocks = {}
        for key in set(self.blocks) | set(other.blocks):
            a, b = self.blocks.get(key), other.blocks.get(key)
            if a is None or b is None:
                src = a if a is not None else b
                blocks[key] = ObsBlock(src.hard.copy(), src.part.copy())
                continue
            hard = a.hard | b.hard
            part = (a.part | b.part) & ~hard
            blocks[key] = ObsBlock(hard, part)
        return ObsSet(self.levels, self.kind, min(self.stride, other.stride), blocks)

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(f"buc.rd14.obsset.v0|{self.kind}|{','.join(map(str, self.levels))}|".encode())
        for key in sorted(self.blocks):
            b = self.blocks[key]
            h.update(np.asarray(key, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(b.hard, dtype="<u8").tobytes())
            h.update(np.ascontiguousarray(b.part, dtype="<u8").tobytes())
        return h.hexdigest()

    def to_meta(self) -> dict:
        return {
            "kind": self.kind, "stride": int(self.stride), "levels": [int(l) for l in self.levels],
            "n_blocks": len(self.blocks), "n_cells_hard": self.n_cells(None, "hard"),
            "n_cells_part": self.n_cells(None, "part"), "digest": self.digest(),
        }


def _check_levels(levels, default) -> tuple:
    if levels is None:
        return default
    try:
        items = list(levels)
    except TypeError as e:
        raise TypeError("levels: 정수 반복자여야 한다") from e
    ints = [_level_check(v, "levels") if isinstance(v, numbers.Integral) and not isinstance(v, bool)
            else _level_check(v, "levels") for v in items]
    if len(set(ints)) != len(ints):
        raise ObsMapError("levels: 중복")
    return tuple(sorted(ints))


def _unique_cells(cells: np.ndarray, nonhard: np.ndarray):
    """cells (N,3) int64 -> (고유 셀 (M,3), 셀별 '경소유 아님' 표본 존재 여부 (M,))."""
    shifted = cells - cells.min(axis=0)
    dims = shifted.max(axis=0).astype(np.float64) + 1.0
    if float(np.prod(dims)) < 2.0 ** 62:
        dy, dz = int(dims[1]), int(dims[2])
        key = (shifted[:, 0] * dy + shifted[:, 1]) * dz + shifted[:, 2]
        uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
        uc = cells[first]
    else:
        uc, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv, weights=nonhard.astype(np.float64), minlength=uc.shape[0])
    return uc, cnt > 0


def build_obs_set(obs_map, depth_m, K, R, t, *, k1=0.0, k2=0.0, levels=None, stride=1, kind="direct") -> ObsSet:
    if not isinstance(obs_map, ObsMap):
        raise TypeError("obs_map: ObsMap 이어야 한다")
    if not isinstance(depth_m, np.ndarray):
        raise TypeError("depth_m: ndarray 여야 한다")
    if depth_m.dtype.kind not in "iuf":
        raise TypeError("depth_m: 수치 dtype 이어야 한다")
    if depth_m.shape != tuple(obs_map.size):
        raise ObsMapError(f"depth_m: shape {depth_m.shape} != 지도 크기 {obs_map.size}")
    K = _check_K(K)
    if not isinstance(R, np.ndarray) or R.dtype.kind not in "iuf":
        raise TypeError("R: 수치 ndarray 여야 한다")
    if R.shape != (3, 3):
        raise ObsMapError("R: (3, 3) 이어야 한다")
    R = R.astype(np.float64)
    if (not np.all(np.isfinite(R)) or not np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            or abs(np.linalg.det(R) - 1.0) > 1e-6):
        raise ObsMapError("R: 회전행렬이 아니다")
    if not isinstance(t, np.ndarray) or t.dtype.kind not in "iuf":
        raise TypeError("t: 수치 ndarray 여야 한다")
    if t.shape != (3,):
        raise ObsMapError("t: (3,) 이어야 한다")
    t = t.astype(np.float64)
    if not np.all(np.isfinite(t)):
        raise ObsMapError("t: 비유한")
    k1 = _fin(k1, "k1")
    k2 = _fin(k2, "k2")
    st = _int(stride, "stride")
    if st < 1:
        raise ObsMapError("stride >= 1")
    if kind not in _KINDS:
        raise ObsMapError(f"kind={kind!r}: {_KINDS}")
    ok = obs_map.valid
    top = int(obs_map.l_part[ok].max()) if ok.any() else -1
    lv = _check_levels(levels, tuple(range(0, top + 1)))

    ii, jj = np.mgrid[0:obs_map.size[0]:st, 0:obs_map.size[1]:st]
    z = depth_m.astype(np.float64)[ii, jj]
    sel = ok[ii, jj] & np.isfinite(z) & (z > 0.0)
    ii, jj, z = ii[sel], jj[sel], z[sel]
    blocks = {}
    if ii.size and lv:
        uv = np.stack([jj, ii], axis=-1).astype(np.float64)
        X = _geom.unproject_world(K, R, t, uv, z, k1, k2)
        good = np.all(np.isfinite(X), axis=-1)
        X, ii, jj = X[good], ii[good], jj[good]
        lh = obs_map.l_hard[ii, jj].astype(np.int64)
        lp = obs_map.l_part[ii, jj].astype(np.int64)
        for l in lv:
            use = lp >= l
            if not use.any():
                continue
            cells = np.floor(X[use] / cell_size_m(l)).astype(np.int64)
            uc, nonhard = _unique_cells(cells, lh[use] < l)
            blk = uc // BLOCK_CELLS
            loc = uc % BLOCK_CELLS
            idx = loc[:, 0] + 16 * loc[:, 1] + 256 * loc[:, 2]
            ub, binv = np.unique(blk, axis=0, return_inverse=True)
            binv = binv.ravel()
            hard = np.zeros((ub.shape[0], 64), dtype=np.uint64)
            part = np.zeros((ub.shape[0], 64), dtype=np.uint64)
            bit = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
            word = idx >> 6
            h = ~nonhard
            np.bitwise_or.at(hard, (binv[h], word[h]), bit[h])
            np.bitwise_or.at(part, (binv[nonhard], word[nonhard]), bit[nonhard])
            for j in range(ub.shape[0]):
                blocks[(l, int(ub[j, 0]), int(ub[j, 1]), int(ub[j, 2]))] = ObsBlock(hard[j].copy(), part[j].copy())
    return ObsSet(levels=lv, kind=kind, stride=st, blocks=blocks)
