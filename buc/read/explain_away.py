"""RD-04: ExplainAway 연산자 v0 -- 역사슬 · 그레인 Wiener · 광원 HDR · 고스트 보간 · 대역 마스크 · 대역 제한 Wiener.

"설명할 수 있는 것은 세계에 넣지 않는다". ``image_lin`` 에서 θ̂_in·RD-03 추정치로 설명되는 화면 고정 효과를 빼 Ĩ(WB 이전 중립 장면
선형)를 내고, 디포커스·모션블러·해상도 한계는 디컨볼브하지 않고 대역 마스크 ``M_b`` 로 표시한다. 관측 대역(|H(ν)|>=h_min) 안에서만
Wiener 보정한다(그 위는 발명이다). 학습 파라미터 0, 결정론. numpy·scipy 만 쓴다(torch·cv2·skimage 없음). import 부작용·전역 RNG 없음,
입력은 바꾸지 않고 새 배열을 돌려준다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import fft as _fft
from scipy import ndimage as _ndi
from scipy import optimize as _opt
from scipy import sparse as _sp
from scipy import special as _special
from scipy.sparse import linalg as _spla

from buc.core import color as _color
from buc.core import contracts as _C
from buc.lens import chain as _chain
from buc.lens import sensor_film as _sf
from buc.read import screen_fx as _fxm

__all__ = [
    "OCTAVES", "OCTAVE_HI", "H_MIN", "BAND_TAPER", "WIENER_K", "COC_MIN_PX", "GHOST_MIN_OCTAVE", "R_RATIO_MIN",
    "HDR_NEAR_SAT", "OWN_OK", "OWN_SENSOR_BLACK", "OWN_SENSOR_SAT", "OWN_LUT_LOST", "OWN_HALO_TAINT", "OWN_GHOST",
    "OWN_HDR", "ExplainAwayError", "ExplainOpts", "defocus_mtf", "motion_mtf", "defocus_cutoff_nu",
    "motion_cutoff_nu", "BandMap", "observed_band", "wiener_grain", "wiener_band", "AwayResult", "explain_away",
]

OCTAVES = 4
OCTAVE_HI = (0.5, 0.25, 0.125, 0.0625)
H_MIN = 0.5
BAND_TAPER = 0.7
WIENER_K = 0.01
COC_MIN_PX = 0.5
GHOST_MIN_OCTAVE = 3
R_RATIO_MIN = 1.0 / 16.0
HDR_NEAR_SAT = 0.97

OWN_OK = _chain.OWN_OK
OWN_SENSOR_BLACK = _chain.OWN_SENSOR_BLACK
OWN_SENSOR_SAT = _chain.OWN_SENSOR_SAT
OWN_LUT_LOST = _chain.OWN_LUT_LOST
OWN_HALO_TAINT = _chain.OWN_HALO_TAINT
OWN_GHOST = 16
OWN_HDR = 32

# ---- 내부 상수
_OCT_EPS = 1e-9               # 옥타브 판정 상대 여유
_HDR_WINDOW = 3               # 블롭 중심 ±3 px 창
_PAD_FACTOR = 8.0             # 대역 Wiener 반사 패딩 = 8·c_max
_LEVEL_RATIO = 1.06           # c 지도 양자화 간격(기하)
_LEVEL_EXACT_MAX = 24         # 서로 다른 c 가 이 수 이하면 양자화하지 않는다
_LEVEL_MAX = 64
_HARMONIC_MAX = 40000         # 조화 보간(희소 LU) 미지수 상한
_GRAIN_R_MIN = 3
_GRAIN_R_MAX = 8
_TINY = 1e-30
_CROSS = _ndi.generate_binary_structure(2, 1)       # 4-연결 십자


class ExplainAwayError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _num_opt(name, v, *, lo=None, hi=None, lo_open=True, hi_open=True):
    """ExplainOpts 용 실수 검증(위반은 ValueError, bool 포함)."""
    if not _is_real(v):
        raise ValueError(f"{name}: 실수여야 한다 ({type(v).__name__})")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name}={x}: 유한값이어야 한다")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f"{name}={x}: 범위 밖")
    if hi is not None and (x >= hi if hi_open else x > hi):
        raise ValueError(f"{name}={x}: 범위 밖")
    return x


def _check_h_min(h_min) -> float:
    if isinstance(h_min, (bool, np.bool_)) or not isinstance(h_min, numbers.Real):
        raise TypeError(f"h_min: 실수여야 한다 ({type(h_min).__name__})")
    x = float(h_min)
    if not (math.isfinite(x) and 0.0 < x < 1.0):
        raise ValueError(f"h_min={x}: (0,1) 이어야 한다")
    return x


def _check_k(k) -> float:
    if not _is_real(k):
        raise TypeError(f"k: 실수여야 한다 ({type(k).__name__})")
    x = float(k)
    if not (math.isfinite(x) and x > 0.0):
        raise ValueError(f"k={x}: 양의 유한값이어야 한다")
    return x


def _check_r_ratio(r) -> float:
    if not _is_real(r):
        raise TypeError(f"r_ratio: 실수여야 한다 ({type(r).__name__})")
    x = float(r)
    if not (math.isfinite(x) and R_RATIO_MIN <= x <= 1.0):
        raise ValueError(f"r_ratio={x}: [{R_RATIO_MIN}, 1] 이어야 한다")
    return x


def _check_shape(shape):
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape: 정수 2-튜플이어야 한다")
    if len(shape) != 2:
        raise ValueError("shape: 길이 2 여야 한다")
    for s in shape:
        if not _is_int(s):
            raise TypeError("shape: 정수여야 한다")
        if int(s) < 1:
            raise ValueError("shape: 양수여야 한다")
    return int(shape[0]), int(shape[1])


def _blur_arg(v, name, shape):
    """None | 실수 스칼라 | (H,W) 배열 -> None | float | float64 (H,W). 유한·비음수."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        if v.dtype.kind not in "fiu":
            raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {v.dtype})")
        a = v.astype(np.float64)
        if a.ndim == 0:
            a = float(a)
        elif a.shape != tuple(shape):
            raise ValueError(f"{name}: shape {a.shape} != {tuple(shape)}")
    elif _is_real(v):
        a = float(v)
    else:
        raise TypeError(f"{name}: None, 실수 또는 (H,W) 배열이어야 한다 ({type(v).__name__})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    if np.any(np.asarray(a) < 0.0):
        raise ValueError(f"{name}: 음수가 있다")
    return a


def _float_img(x, name, *, strict01=False):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name}: ndarray 여야 한다 ({type(x).__name__})")
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {x.dtype})")
    if x.ndim != 3 or x.shape[2] != 3 or min(x.shape) < 1:
        raise ValueError(f"{name}: (H,W,3) 이어야 한다 (shape {x.shape})")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    if float(x.min()) < 0.0:
        raise ValueError(f"{name}: 음수가 있다")
    if strict01 and float(x.max()) > 1.0:
        raise ValueError(f"{name}: [0,1] 이어야 한다")
    return x


# ---------------------------------------------------------------------------------------------------------------
# 전달함수 · 컷오프
# ---------------------------------------------------------------------------------------------------------------
def _f64(v, name, *, nonneg):
    if isinstance(v, (bool, np.bool_, str, bytes)):
        raise TypeError(f"{name}: 수치여야 한다")
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name}: 수치여야 한다") from e
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: 유한값이어야 한다")
    if nonneg and np.any(a < 0.0):
        raise ValueError(f"{name}: 음수가 있다")
    return a


def defocus_mtf(nu, coc_px):
    """원판 흐림 MTF |2 J1(π c ν)/(π c ν)| (c=0 -> 1). float64, 방송."""
    n = _f64(nu, "nu", nonneg=False)
    c = _f64(coc_px, "coc_px", nonneg=True)
    x = np.pi * c * np.abs(n)
    small = x < 1e-12
    xs = np.where(small, 1.0, x)
    return np.asarray(np.abs(np.where(small, 1.0, 2.0 * _special.j1(xs) / xs)), dtype=np.float64)


def motion_mtf(nu, length_px):
    """균일 선형 모션 MTF |sinc(L ν)| (numpy sinc: sin(πx)/(πx)). float64, 방송."""
    n = _f64(nu, "nu", nonneg=False)
    L = _f64(length_px, "length_px", nonneg=True)
    return np.asarray(np.abs(np.sinc(L * n)), dtype=np.float64)


@lru_cache(maxsize=64)
def _x_defocus(h: float) -> float:
    return float(_opt.brentq(lambda x: 2.0 * _special.j1(x) / x - h, 1e-9, 3.8317059702075125, xtol=1e-15, rtol=1e-15))


@lru_cache(maxsize=64)
def _x_motion(h: float) -> float:
    return float(_opt.brentq(lambda x: math.sin(x) / x - h, 1e-9, math.pi, xtol=1e-15, rtol=1e-15))


def _cutoff(x_root, size, h_min, name):
    h = _check_h_min(h_min)
    s = _f64(size, name, nonneg=True)
    xr = x_root(h)
    with np.errstate(divide="ignore"):
        out = np.where(s > 0.0, xr / (np.pi * np.where(s > 0.0, s, 1.0)), np.inf)
    return np.asarray(out, dtype=np.float64)


def defocus_cutoff_nu(coc_px, h_min=H_MIN):
    """디포커스 MTF 가 h_min 이 되는 첫 ν (cyc/px). c=0 이면 inf."""
    return _cutoff(_x_defocus, coc_px, h_min, "coc_px")


def motion_cutoff_nu(length_px, h_min=H_MIN):
    """모션 MTF 가 h_min 이 되는 첫 ν (cyc/px). L=0 이면 inf."""
    return _cutoff(_x_motion, length_px, h_min, "length_px")


# ---------------------------------------------------------------------------------------------------------------
# 옵션
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ExplainOpts:
    grain: bool = True
    ghost: bool = True
    hdr: bool = True
    defocus: bool = True
    wiener_k: float = WIENER_K
    grain_nsr: float = 1.0
    h_min: float = H_MIN

    def __post_init__(self):
        for n in ("grain", "ghost", "hdr", "defocus"):
            if not isinstance(getattr(self, n), bool):
                raise ValueError(f"{n}: bool 이어야 한다")
        object.__setattr__(self, "wiener_k", _num_opt("wiener_k", self.wiener_k, lo=0.0))
        object.__setattr__(self, "grain_nsr", _num_opt("grain_nsr", self.grain_nsr, lo=0.0))
        object.__setattr__(self, "h_min", _num_opt("h_min", self.h_min, lo=0.0, hi=1.0))


# ---------------------------------------------------------------------------------------------------------------
# 관측 대역 지도
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class BandMap:
    """cutoff_nu f32 (H,W); mask bool (4,H,W) (옥타브 0=최고 대역); min_octave u8 (H,W) (첫 관측 옥타브, 없으면 4)."""
    cutoff_nu: np.ndarray
    mask: np.ndarray
    min_octave: np.ndarray


def _min_octave(mask):
    return np.where(mask.any(axis=0), mask.argmax(axis=0), OCTAVES).astype(np.uint8)


def _band_from_cutoff(nu64):
    hi = np.asarray(OCTAVE_HI, np.float64)[:, None, None]
    mask = hi <= nu64[None] * (1.0 + _OCT_EPS)
    return BandMap(cutoff_nu=nu64.astype(np.float32), mask=mask, min_octave=_min_octave(mask))


def _cutoff_map(shape, coc, mot, r_ratio, h_min):
    nu = np.full(shape, 0.5 * r_ratio, np.float64)
    if coc is not None:
        nu = np.minimum(nu, defocus_cutoff_nu(coc, h_min))
    if mot is not None:
        nu = np.minimum(nu, motion_cutoff_nu(mot, h_min))
    return np.asarray(nu, np.float64)


def observed_band(shape, *, coc_px=None, motion_len_px=None, r_ratio=1.0, h_min=H_MIN) -> BandMap:
    """컷오프 ν_c = min(0.5·r_ratio, 디포커스, 모션). 옥타브 b 는 윗 가장자리 <= ν_c 일 때 관측."""
    shp = _check_shape(shape)
    coc = _blur_arg(coc_px, "coc_px", shp)
    mot = _blur_arg(motion_len_px, "motion_len_px", shp)
    r = _check_r_ratio(r_ratio)
    h = _check_h_min(h_min)
    return _band_from_cutoff(_cutoff_map(shp, coc, mot, r, h))


# ---------------------------------------------------------------------------------------------------------------
# 구멍 메우기 (저대역 보간)
# ---------------------------------------------------------------------------------------------------------------
def _neighbor_pairs(h, w):
    lin = np.arange(h * w, dtype=np.int64).reshape(h, w)
    ps, qs = [], []
    if w > 1:
        a, b = lin[:, :-1].ravel(), lin[:, 1:].ravel()
        ps += [a, b]
        qs += [b, a]
    if h > 1:
        a, b = lin[:-1, :].ravel(), lin[1:, :].ravel()
        ps += [a, b]
        qs += [b, a]
    return np.concatenate(ps), np.concatenate(qs)


def _harmonic_fill(img, hole):
    """구멍 안 값을 라플라스 방정식(주변 값 고정, 영상 가장자리는 자연 경계)으로 채운다. img (H,W,C) float64."""
    h, w = hole.shape
    n = int(hole.sum())
    flat = img.reshape(h * w, -1)
    idx = np.full(h * w, -1, np.int64)
    hf = hole.ravel()
    idx[hf] = np.arange(n)
    P, Q = _neighbor_pairs(h, w)
    sel = hf[P]
    P, Q = P[sel], Q[sel]
    deg = np.bincount(idx[P], minlength=n).astype(np.float64)
    unk = hf[Q]
    rows = np.concatenate([np.arange(n), idx[P[unk]]])
    cols = np.concatenate([np.arange(n), idx[Q[unk]]])
    vals = np.concatenate([deg, -np.ones(int(unk.sum()))])
    A = _sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    Pk, Qk = P[~unk], Q[~unk]
    rhs = np.stack([np.bincount(idx[Pk], weights=flat[Qk, c], minlength=n) for c in range(flat.shape[1])], axis=1)
    sol = _spla.splu(A).solve(rhs)
    out = flat.copy()
    out[hf] = sol
    return out.reshape(img.shape)


def _nc_fill(img, hole):
    """큰 구멍용 근사: 정규화 합성곱(가우시안 σ 를 두 배씩 키우며)."""
    known = ~hole
    wk = known.astype(np.float64)
    out = img.copy()
    todo = hole.copy()
    sigma = 2.0
    for _ in range(12):
        if not todo.any():
            break
        den = _ndi.gaussian_filter(wk, sigma, mode="nearest")
        num = np.stack([_ndi.gaussian_filter(img[..., c] * wk, sigma, mode="nearest") for c in range(img.shape[2])], 2)
        ok = todo & (den > 1e-3)
        out[ok] = num[ok] / den[ok][:, None]
        todo &= ~ok
        sigma *= 2.0
    if todo.any():
        out[todo] = img[known].mean(axis=0)
    return out


def _fill_holes(img, hole):
    """img (H,W,C) float64 의 hole 화소를 hole 밖 화소로 저대역 보간한 새 배열."""
    n = int(hole.sum())
    if n == 0:
        return img.copy()
    if n >= hole.size:
        return img.copy()
    if n <= _HARMONIC_MAX:
        return _harmonic_fill(img, hole)
    return _nc_fill(img, hole)


# ---------------------------------------------------------------------------------------------------------------
# 대역 제한 Wiener
# ---------------------------------------------------------------------------------------------------------------
def _taper(u):
    return np.where(u <= BAND_TAPER, 1.0,
                    np.where(u >= 1.0, 0.0, 0.5 * (1.0 + np.cos(np.pi * (u - BAND_TAPER) / (1.0 - BAND_TAPER)))))


def _band_gain(nu, c, k, h_min):
    """G(ν) = 1 + t(ν/ν_c)·((1+K)H/(H²+K) − 1)."""
    h = defocus_mtf(nu, c)
    gw = (1.0 + k) * h / (h * h + k)
    nc = float(defocus_cutoff_nu(c, h_min))
    return 1.0 + _taper(nu / nc) * (gw - 1.0)


def _levels(c_active):
    u = np.unique(c_active)
    if u.size <= _LEVEL_EXACT_MAX:
        return u
    lo, hi = float(u[0]), float(u[-1])
    n = int(min(_LEVEL_MAX, math.ceil(math.log(hi / lo) / math.log(_LEVEL_RATIO)) + 1))
    n = max(n, 2)
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))


def wiener_band(linear, coc_px, *, k=WIENER_K, h_min=H_MIN, valid=None) -> np.ndarray:
    """관측 대역(|H|>=h_min) 안에서만 하는 Wiener 보정. c<COC_MIN_PX·무효 화소는 입력과 비트 동일. float32 (H,W,3), 0 이상."""
    x = _float_img(linear, "linear")
    h, w = x.shape[:2]
    if coc_px is None:
        raise TypeError("coc_px: 실수 또는 (H,W) 배열이어야 한다")
    coc = _blur_arg(coc_px, "coc_px", (h, w))
    kk = _check_k(k)
    hm = _check_h_min(h_min)
    if valid is not None:
        if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
            raise TypeError("valid: bool 배열이어야 한다")
        if valid.shape != (h, w):
            raise ValueError(f"valid: shape {valid.shape} != {(h, w)}")
        if valid.all():
            valid = None
    x32 = x.astype(np.float32)
    cmap = np.broadcast_to(np.asarray(coc, np.float64), (h, w))
    active = cmap >= COC_MIN_PX
    if valid is not None:
        active = active & valid
    if not active.any():
        return x32.copy()

    src = x.astype(np.float64)
    if valid is not None:
        src = _fill_holes(src, ~valid)
    levels = _levels(cmap[active])
    cmax = float(levels[-1])
    pad = int(math.ceil(_PAD_FACTOR * cmax))
    py, px = min(pad, h - 1) if h > 1 else 0, min(pad, w - 1) if w > 1 else 0
    hp, wp = _fft.next_fast_len(h + 2 * py, real=True), _fft.next_fast_len(w + 2 * px, real=True)
    padded = np.pad(src, ((py, hp - h - py), (px, wp - w - px), (0, 0)), mode="reflect")
    F = _fft.rfft2(padded, axes=(0, 1))
    nu = np.hypot(np.fft.fftfreq(hp)[:, None], np.fft.rfftfreq(wp)[None, :])

    # 화소별 보간 가중치(인접한 두 수준)
    ca = np.where(active, cmap, levels[0])
    n = levels.size
    if n == 1:
        i0 = np.zeros((h, w), np.intp)
        t = np.zeros((h, w))
    else:
        i0 = np.clip(np.searchsorted(levels, ca, side="right") - 1, 0, n - 2)
        t = np.clip((ca - levels[i0]) / (levels[i0 + 1] - levels[i0]), 0.0, 1.0)
    out = np.zeros((h, w, 3), np.float64)
    for j in range(n):
        wj = np.where(i0 == j, 1.0 - t, 0.0)
        if n > 1:
            wj = wj + np.where(i0 + 1 == j, t, 0.0)
        wj = np.where(active, wj, 0.0)
        if not wj.any():
            continue
        G = _band_gain(nu, float(levels[j]), kk, hm)
        f = _fft.irfft2(F * G[..., None], s=(hp, wp), axes=(0, 1))[py:py + h, px:px + w]
        out += wj[..., None] * f
    res = np.maximum(out, 0.0).astype(np.float32)
    return np.where(active[..., None], res, x32)


# ---------------------------------------------------------------------------------------------------------------
# 그레인 Wiener
# ---------------------------------------------------------------------------------------------------------------
def wiener_grain(display, grain, *, nsr=1.0) -> np.ndarray:
    """표시값 영역 에지 보존 Wiener(국소 Lee 형): σ(u)=grain.sigma_at(u), 잡음 분산 배율 nsr. σ=0 이면 입력과 비트 동일."""
    d = _float_img(display, "display", strict01=True)
    if not isinstance(grain, _fxm.GrainFit):
        raise TypeError("grain: GrainFit 이어야 한다")
    if not _is_real(nsr):
        raise TypeError("nsr: 실수여야 한다")
    ns = float(nsr)
    if not (math.isfinite(ns) and ns > 0.0):
        raise ValueError(f"nsr={ns}: 양의 유한값이어야 한다")
    d32 = d.astype(np.float32)
    x = d.astype(np.float64)
    if not np.any(np.asarray(grain.sigma_at(x)) > 0.0):
        return d32.copy()
    r = float(grain.radius_px) if math.isfinite(float(grain.radius_px)) else 0.75
    rad = int(min(_GRAIN_R_MAX, max(_GRAIN_R_MIN, math.ceil(2.5 * max(r, 0.0) + 1.0))))
    size = (2 * rad + 1, 2 * rad + 1, 1)
    m = _ndi.uniform_filter(x, size=size, mode="reflect")
    m2 = _ndi.uniform_filter(x * x, size=size, mode="reflect")
    v = np.maximum(m2 - m * m, 0.0)
    s2 = ns * np.asarray(grain.sigma_at(m), np.float64) ** 2
    wgt = np.where(s2 > 0.0, np.minimum(s2 / np.maximum(v, _TINY), 1.0), 0.0)     # 잡음 몫(1 - Wiener 이득)
    out = x - wgt * (x - m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class AwayResult:
    linear: np.ndarray
    status: np.ndarray
    error_bound: np.ndarray
    band: BandMap
    sigma_grain: np.ndarray

    @property
    def owned(self) -> np.ndarray:
        return self.status == OWN_OK

    def to_meta(self) -> dict:
        h, w = self.status.shape
        return {
            "size": [int(h), int(w)],
            "owned_frac": float(np.mean(self.status == OWN_OK)),
            "ghost_px": int(np.count_nonzero(self.status & OWN_GHOST)),
            "hdr_px": int(np.count_nonzero(self.status & OWN_HDR)),
            "band_frac": [float(np.mean(self.band.mask[b])) for b in range(OCTAVES)],
            "cutoff_nu_median": float(np.median(self.band.cutoff_nu.astype(np.float64))),
            "sigma_grain_mean": float(np.mean(self.sigma_grain.astype(np.float64))),
        }


# ---------------------------------------------------------------------------------------------------------------
# 단계 보조
# ---------------------------------------------------------------------------------------------------------------
def _recover_hdr(linear, status, d_in, theta, fx):
    """H⁻¹: 포화 광원 성분을 헤일로 진폭이 말하는 노출 영역 휘도로 올린다(중립 광원 가정). (linear, status) 새 배열."""
    hl = fx.halation
    a = float(theta.halation) if float(theta.halation) > 0.0 else float(hl.amount)
    if not (hl.detected and hl.n_blobs > 0 and len(hl.blobs) > 0 and math.isfinite(a) and a > 0.0):
        return linear, status
    y = min(float(hl.source_luminance(min(a, 1.0))), float(_chain.HALO_SOURCE_MAX))
    if not math.isfinite(y):
        return linear, status
    s, _ = _chain.invert_look(d_in, theta.lut)
    near = (s >= HDR_NEAR_SAT).all(axis=-1)
    if not near.any():
        return linear, status
    lab, _n = _ndi.label(near, structure=_CROSS)
    h, w = near.shape
    chosen = set()
    for b in hl.blobs:
        cx, cy = int(math.floor(float(b.x) + 0.5)), int(math.floor(float(b.y) + 0.5))
        y0, y1 = max(0, cy - _HDR_WINDOW), min(h, cy + _HDR_WINDOW + 1)
        x0, x1 = max(0, cx - _HDR_WINDOW), min(w, cx + _HDR_WINDOW + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        sub = lab[y0:y1, x0:x1]
        sub = sub[sub > 0]
        if sub.size:
            chosen.add(int(np.bincount(sub).argmax()))
    if not chosen:
        return linear, status
    src = np.isin(lab, sorted(chosen))
    core = _ndi.binary_erosion(src, structure=_CROSS, border_value=1)
    if not core.any():
        return linear, status
    v = _sf.vignette_gain(h, w, float(theta.vignette)).astype(np.float64)
    ev = float(np.float32(2.0 ** float(theta.ev)))
    g = _chain.wb_gains(theta.wb_kelvin, theta.wb_tint).astype(np.float32).astype(np.float64)
    scale = v[..., None] * ev * g[None, None, :]
    u = linear.astype(np.float64) * scale
    raised = (u < y) & core[..., None]
    anyr = raised.any(axis=-1)
    if not anyr.any():
        return linear, status
    new = np.where(raised, y / scale, linear.astype(np.float64)).astype(np.float32)
    lin2 = np.where(anyr[..., None], new, linear)
    st2 = status.copy()
    st2[anyr] |= np.uint8(OWN_HDR)
    return lin2, st2


def _demote_ghost(band: BandMap, gmask) -> BandMap:
    if not gmask.any():
        return band
    mask = band.mask.copy()
    mask[:GHOST_MIN_OCTAVE, gmask] = False
    cut = band.cutoff_nu.copy()
    cut[gmask] = np.minimum(cut[gmask], np.float32(OCTAVE_HI[GHOST_MIN_OCTAVE]))
    return BandMap(cutoff_nu=cut, mask=mask, min_octave=_min_octave(mask))


# ---------------------------------------------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------------------------------------------
def explain_away(image_lin, theta, *, screen_fx=None, coc_px=None, motion_len_px=None, r_ratio=1.0, spec=None,
                 opts=None) -> AwayResult:
    """입력 선형 영상에서 θ̂·RD-03 으로 설명되는 것을 뺀 Ĩ 와 소유 상태·오차 한계·대역 지도를 돌려준다."""
    img = _float_img(image_lin, "image_lin")
    h, w = img.shape[:2]
    if not isinstance(theta, _C.Theta):
        raise TypeError("theta: Theta 여야 한다")
    fx = screen_fx
    if fx is not None:
        if not isinstance(fx, _fxm.ScreenFx):
            raise TypeError("screen_fx: ScreenFx 또는 None 이어야 한다")
        if tuple(int(v) for v in fx.size) != (h, w):
            raise ValueError(f"screen_fx.size={tuple(fx.size)} != {(h, w)}")
    if spec is not None and not isinstance(spec, _sf.SensorSpec):
        raise TypeError("spec: SensorSpec 또는 None 이어야 한다")
    if opts is None:
        o = ExplainOpts()
    elif isinstance(opts, ExplainOpts):
        o = opts
    else:
        raise TypeError("opts: ExplainOpts 또는 None 이어야 한다")
    coc = _blur_arg(coc_px, "coc_px", (h, w))
    mot = _blur_arg(motion_len_px, "motion_len_px", (h, w))
    rr = _check_r_ratio(r_ratio)

    # 1. 표시값, 그레인 통계
    d_in = _color.linear_to_display(np.clip(img, 0.0, 1.0).astype(np.float32))
    if fx is not None:
        sigma = np.mean(np.asarray(fx.grain.sigma_at(d_in.astype(np.float64)), np.float64), axis=2).astype(np.float32)
    else:
        sigma = np.zeros((h, w), np.float32)

    # 2. 그레인 Wiener -> 역사슬
    d = wiener_grain(d_in, fx.grain, nsr=o.grain_nsr) if (fx is not None and o.grain) else d_in
    e = _chain.explain(d, theta, spec=spec)
    linear, status = e.linear, e.status.copy()

    # 3. 광원 HDR
    if fx is not None and o.hdr:
        linear, status = _recover_hdr(linear, status, d_in, theta, fx)

    # 4. 고스트 저대역 보간
    if fx is not None and o.ghost and len(fx.ghost.ghosts) > 0:
        m = fx.ghost.mask(h, w)
        if m.any():
            filled = _fill_holes(linear.astype(np.float64), m)
            linear = np.where(m[..., None], np.maximum(filled, 0.0).astype(np.float32), linear)
            status[m] |= np.uint8(OWN_GHOST)

    # 5. 대역 Wiener (소유 화소에서만)
    if coc is not None and o.defocus:
        linear = wiener_band(linear, coc, k=o.wiener_k, h_min=o.h_min, valid=(status == OWN_OK))

    # 6. 오차 한계, 대역 지도
    err = np.where((status != OWN_OK)[..., None], np.float32(np.inf), e.error_bound).astype(np.float32)
    band = _band_from_cutoff(_cutoff_map((h, w), coc, mot, rr, o.h_min))
    band = _demote_ghost(band, (status & OWN_GHOST) != 0)
    return AwayResult(linear=np.array(linear, np.float32, copy=True), status=status, error_bound=err, band=band,
                      sigma_grain=sigma)
