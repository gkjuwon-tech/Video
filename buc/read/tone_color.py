"""RD-06: R4 톤·색 v0 -- 4파라미터 필름 곡선 · WB · CDL · 룩 잔차 LUT 17^3.

입력의 톤·색 처리 G 를 모형으로 두고 역추정(estimate_tone_color)하고, 역 G^-1 로 "중립 룩 장면 선형" L̂ 을 낸다.
명제 L2: 곡선·CDL·LUT 가 같으면 WB·EV 의 선형 오귀속은 다시 찍을 때 정확히 상쇄된다.

모형(float64, 표시값 = sRGB 부호화): x=L*g*2^ev -> y=clip(F(x),0,1) -> c=srgb_enc(y) -> CDL -> clip -> LUT17 -> clip -> D=srgb_dec(c).
numpy/scipy 만 쓴다(torch 금지). import 부작용·전역 RNG 없음. 결정론. 입력 불변, 새 배열 반환.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as _ndi
from scipy import sparse as _sp
from scipy.optimize import least_squares as _least_squares
from scipy.sparse import linalg as _spla
from scipy.special import ndtri as _ndtri

from buc.lens import chain as _chain
from buc.lens import sensor_film as _sf

__all__ = [
    "LUT_N", "MID_GRAY", "OWN_OK", "OWN_CLIP_LO", "OWN_CLIP_HI", "OWN_LUT_LOST",
    "FilmCurve", "CDL", "ToneParams", "ToneInverse", "film_curve", "film_curve_inv", "apply_cdl", "invert_cdl",
    "apply_residual_lut", "invert_residual_lut", "forward", "inverse", "bake_lut", "theta_fields",
    "fit_residual_lut", "r_look", "SkyAnchor", "TonePriors", "ToneOpts", "estimate_tone_color", "ToneColor",
]

LUT_N = 17
MID_GRAY = 0.18
OWN_OK = 0
OWN_CLIP_LO = 1
OWN_CLIP_HI = 2
OWN_LUT_LOST = 4          # LNS-08 explain 비트와 같은 자리

_TOE_A = 0.02
_W709 = np.array([0.2126, 0.7152, 0.0722], np.float64)
_CLIP_EDGE = 1e-3         # 클립 판정 경계(표시/곡선 천장)
_CDL_LOST = 1e-6
_LUT_OK_TOL = 1e-5
_WB_REF_M = 1e6 / 6500.0


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _num(name, v, lo=None, hi=None, *, lo_open=False) -> float:
    if not _is_real(v):
        raise TypeError(f"{name}: 실수여야 한다 ({type(v).__name__})")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name}={v!r}: 유한값이어야 한다")
    if lo is not None and (f <= lo if lo_open else f < lo):
        raise ValueError(f"{name}={f}: 하한 {lo} 미만")
    if hi is not None and f > hi:
        raise ValueError(f"{name}={f}: 상한 {hi} 초과")
    return f


def _triplet(name, v, lo, hi):
    if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
        raise TypeError(f"{name}: 길이 3 의 수열이어야 한다")
    t = tuple(v)
    if len(t) != 3:
        raise ValueError(f"{name}: 길이 3 이어야 한다 (길이 {len(t)})")
    return tuple(_num(f"{name}[{i}]", e, lo, hi) for i, e in enumerate(t))


def _float_arr(x, name):
    a = np.asarray(x)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    return a


def _rgb_arr(x, name, *, nonneg=True):
    """(...,3) float32/64 유한 -> float64 사본 읽기. 0-d 는 거부."""
    a = _float_arr(x, name)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"{name}: 마지막 축이 3 이어야 한다 (shape {a.shape})")
    if nonneg and a.size and float(a.min()) < 0.0:
        raise ValueError(f"{name}: 음수가 있다")
    return a.astype(np.float64)


def _coded_arr(x, name):
    """부호화 값 (...,3): 수치형, 유한. float64 로 반환."""
    a = np.asarray(x)
    if a.dtype.kind not in "fiu":
        raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    a = a.astype(np.float64)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"{name}: 마지막 축이 3 이어야 한다 (shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    return a


def _lut_arr(lut, name="lut"):
    L = np.asarray(lut)
    if L.dtype.kind not in "fiu":
        raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {L.dtype})")
    if L.shape != (LUT_N, LUT_N, LUT_N, 3):
        raise ValueError(f"{name}: ({LUT_N},{LUT_N},{LUT_N},3) 이어야 한다 (shape {L.shape})")
    L = L.astype(np.float64)
    if not np.all(np.isfinite(L)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    return L


# ---------------------------------------------------------------------------------------------------------------
# sRGB
# ---------------------------------------------------------------------------------------------------------------
def _enc(v):
    v = np.asarray(v, np.float64)
    a = np.maximum(v, 0.0)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.maximum(a, 1e-300) ** (1.0 / 2.4) - 0.055)


def _dec(c):
    c = np.maximum(np.asarray(c, np.float64), 0.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ---------------------------------------------------------------------------------------------------------------
# 데이터클래스
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class FilmCurve:
    toe: float = 0.0
    shoulder: float = 0.0
    slope: float = 1.0
    white: float = 1.0

    def __post_init__(self):
        for n, lo, hi in (("toe", 0.0, 1.0), ("shoulder", 0.0, 1.0), ("slope", 0.5, 2.0), ("white", 0.5, 2.0)):
            object.__setattr__(self, n, _num(f"FilmCurve.{n}", getattr(self, n), lo, hi))


@dataclass(frozen=True, kw_only=True)
class CDL:
    slope: tuple = (1.0, 1.0, 1.0)
    offset: tuple = (0.0, 0.0, 0.0)
    power: tuple = (1.0, 1.0, 1.0)
    sat: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "slope", _triplet("CDL.slope", self.slope, 0.25, 4.0))
        object.__setattr__(self, "offset", _triplet("CDL.offset", self.offset, -0.25, 0.25))
        object.__setattr__(self, "power", _triplet("CDL.power", self.power, 0.4, 2.5))
        object.__setattr__(self, "sat", _num("CDL.sat", self.sat, 0.05, 2.0))

    def is_identity(self) -> bool:
        return (self.slope == (1.0, 1.0, 1.0) and self.offset == (0.0, 0.0, 0.0)
                and self.power == (1.0, 1.0, 1.0) and self.sat == 1.0)


@dataclass(frozen=True, kw_only=True, eq=False)
class ToneParams:
    ev: float = 0.0
    wb_kelvin: float = 6500.0
    wb_tint: float = 0.0
    curve: FilmCurve = field(default_factory=FilmCurve)
    cdl: CDL = field(default_factory=CDL)
    lut: object = None

    def __post_init__(self):
        object.__setattr__(self, "ev", _num("ToneParams.ev", self.ev, -5.0, 5.0))
        object.__setattr__(self, "wb_kelvin", _num("ToneParams.wb_kelvin", self.wb_kelvin, 2000.0, 12000.0))
        object.__setattr__(self, "wb_tint", _num("ToneParams.wb_tint", self.wb_tint, -0.6, 0.6))
        if not isinstance(self.curve, FilmCurve):
            raise TypeError("ToneParams.curve: FilmCurve 여야 한다")
        if not isinstance(self.cdl, CDL):
            raise TypeError("ToneParams.cdl: CDL 이어야 한다")
        if self.lut is not None:
            L = _lut_arr(self.lut, "ToneParams.lut").astype(np.float32)
            L.setflags(write=False)
            object.__setattr__(self, "lut", L)

    def is_identity(self) -> bool:
        return (self.ev == 0.0 and self.wb_kelvin == 6500.0 and self.wb_tint == 0.0 and self.curve == FilmCurve()
                and self.cdl.is_identity() and self.lut is None)


@dataclass(frozen=True)
class ToneInverse:
    linear: np.ndarray
    status: np.ndarray

    @property
    def owned(self) -> np.ndarray:
        return self.status == OWN_OK


# ---------------------------------------------------------------------------------------------------------------
# 필름 곡선
# ---------------------------------------------------------------------------------------------------------------
def _curve_core(x, toe, shoulder, slope, white):
    """검증 없는 곡선: x (float64, >=0) -> y."""
    b = _TOE_A * (1.0 + 2.0 * toe)
    with np.errstate(all="ignore"):
        p = MID_GRAY * (x / MID_GRAY) ** slope
        t = p * (p + _TOE_A) / (p + b)
        if shoulder == 0.0:
            return np.minimum(t, white)
        S = 1.0 / shoulder
        r = t / white
        big = r > 1.0
        rb = np.where(big, r, 2.0)
        rs = np.where(big, 0.0, r)
        d = np.where(big, rb * (1.0 + rb ** -S) ** (1.0 / S), (1.0 + rs ** S) ** (1.0 / S))
        return t / d


def _curve_inv_core(y, toe, shoulder, slope, white):
    """검증 없는 역: y (float64, >=0) -> (x, clipped)."""
    b = _TOE_A * (1.0 + 2.0 * toe)
    edge = white * (1.0 - _CLIP_EDGE)
    clipped = y >= edge
    yc = np.minimum(y, edge)
    with np.errstate(all="ignore"):
        if shoulder == 0.0:
            t = yc
        else:
            S = 1.0 / shoulder
            t = yc / (1.0 - (yc / white) ** S) ** (1.0 / S)
        u = t - _TOE_A
        disc = np.sqrt(u * u + 4.0 * b * t)
        # 두 근 공식 중 상쇄가 없는 쪽을 쓴다
        p = np.where(u >= 0.0, 0.5 * (u + disc), 2.0 * b * t / np.maximum(disc - u, 1e-300))
        p = np.where(t <= 0.0, 0.0, p)
        x = MID_GRAY * (p / MID_GRAY) ** (1.0 / slope)
    return x, clipped


def _curve_args(curve):
    if not isinstance(curve, FilmCurve):
        raise TypeError("curve: FilmCurve 여야 한다")
    return curve.toe, curve.shoulder, curve.slope, curve.white


def _nonneg_f64(x, name):
    a = np.asarray(x)
    if a.dtype.kind not in "fiu":
        raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    a = a.astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    if a.size and float(a.min()) < 0.0:
        raise ValueError(f"{name}: 음수가 있다")
    return a


def film_curve(x, curve) -> np.ndarray:
    """장면 선형 x(>=0) -> 곡선 출력 y (float64)."""
    args = _curve_args(curve)
    return np.asarray(_curve_core(_nonneg_f64(x, "x"), *args), np.float64)


def film_curve_inv(y, curve):
    """곡선 역. (x f64, clipped bool). y >= white(1-1e-3) 는 clipped 이고 그 값으로 자른 하한을 돌려준다."""
    args = _curve_args(curve)
    x, clipped = _curve_inv_core(_nonneg_f64(y, "y"), *args)
    return np.asarray(x, np.float64), np.asarray(clipped, bool)


# ---------------------------------------------------------------------------------------------------------------
# CDL (부호화 영역)
# ---------------------------------------------------------------------------------------------------------------
def _cdl_vecs(cdl):
    if not isinstance(cdl, CDL):
        raise TypeError("cdl: CDL 이어야 한다")
    return np.array(cdl.slope), np.array(cdl.offset), np.array(cdl.power), cdl.sat


def _apply_cdl(c, slope, offset, power, sat):
    with np.errstate(all="ignore"):
        v = np.maximum(c * slope + offset, 0.0) ** power
    Y = (v @ _W709)[..., None]
    return Y + sat * (v - Y)


def apply_cdl(c, cdl) -> np.ndarray:
    sl, off, pw, sat = _cdl_vecs(cdl)
    return np.asarray(_apply_cdl(_coded_arr(c, "c"), sl, off, pw, sat), np.float64)


def _invert_cdl(out, slope, offset, power, sat):
    Y = (out @ _W709)[..., None]
    v = Y + (out - Y) / sat
    lost = np.any(v <= _CDL_LOST, axis=-1)
    with np.errstate(all="ignore"):
        c = (np.maximum(v, _CDL_LOST) ** (1.0 / power) - offset) / slope
    return c, lost


def invert_cdl(out, cdl):
    sl, off, pw, sat = _cdl_vecs(cdl)
    c, lost = _invert_cdl(_coded_arr(out, "out"), sl, off, pw, sat)
    return np.asarray(c, np.float64), np.asarray(lost, bool)


# ---------------------------------------------------------------------------------------------------------------
# 잔차 LUT 17^3
# ---------------------------------------------------------------------------------------------------------------
def _trilinear(c, lut, jac=False):
    """c (n,3) 부호화 입력(클램프) -> (n,3) 출력[, (n,3,3) 야코비안 d out_k / d in_j]. lut (S,S,S,3) float64."""
    S = lut.shape[0]
    flat = lut.reshape(-1, 3)
    p = np.clip(c, 0.0, 1.0) * (S - 1)
    i0 = np.minimum(np.floor(p).astype(np.intp), S - 2)
    f = p - i0
    base = (i0[:, 0] * S + i0[:, 1]) * S + i0[:, 2]
    out = np.zeros(p.shape, np.float64)
    J = np.zeros((p.shape[0], 3, 3), np.float64) if jac else None
    corners = {}
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                corners[(dr, dg, db)] = flat[base + dr * S * S + dg * S + db]
    w1 = (1.0 - f, f)
    for (dr, dg, db), v in corners.items():
        out += (w1[dr][:, 0] * w1[dg][:, 1] * w1[db][:, 2])[:, None] * v
    if jac:
        sc = float(S - 1)
        for ax in range(3):
            o = [(0, 1), (0, 1), (0, 1)]
            others = [a for a in range(3) if a != ax]
            for d1 in (0, 1):
                for d2 in (0, 1):
                    key0 = [0, 0, 0]
                    key1 = [0, 0, 0]
                    key0[others[0]] = key1[others[0]] = d1
                    key0[others[1]] = key1[others[1]] = d2
                    key1[ax] = 1
                    w = w1[d1][:, others[0]] * w1[d2][:, others[1]]
                    J[:, :, ax] += (w * sc)[:, None] * (corners[tuple(key1)] - corners[tuple(key0)])
            del o
    return (out, J) if jac else out


def apply_residual_lut(coded, lut) -> np.ndarray:
    """부호화 입력 -> 삼선형 LUT 출력(입력 [0,1] 클램프). float64."""
    c = _coded_arr(coded, "coded")
    L = _lut_arr(lut)
    return _trilinear(c.reshape(-1, 3), L).reshape(c.shape)


def _lut_invert(y, L):
    """y (n,3) -> (x (n,3) in [0,1], resid (n,)): 뉴턴 + 고정점 후보, 활성 집합."""
    n = y.shape[0]
    x = np.clip(y, 0.0, 1.0)
    res = np.max(np.abs(_trilinear(x, L) - y), axis=1) if n else np.zeros(0)
    active = np.flatnonzero(res >= 1e-12)
    eye = np.eye(3)
    for _ in range(80):
        if active.size == 0:
            break
        xa, ya = x[active], y[active]
        fa, Ja = _trilinear(xa, L, jac=True)
        r = fa - ya
        cands = [np.clip(xa - r, 0.0, 1.0)]
        try:
            dx = np.linalg.solve(Ja + 1e-9 * eye, -r[..., None])[..., 0]
            for lam in (1.0, 0.5, 0.25, 0.125):
                cands.append(np.clip(xa + lam * dx, 0.0, 1.0))
        except np.linalg.LinAlgError:
            pass
        best_x, best_r = xa, np.max(np.abs(r), axis=1)
        for cx in cands:
            cr = np.max(np.abs(_trilinear(cx, L) - ya), axis=1)
            better = cr < best_r
            best_x = np.where(better[:, None], cx, best_x)
            best_r = np.where(better, cr, best_r)
        improved = best_r < res[active] - 1e-15
        x[active] = best_x
        res[active] = best_r
        active = active[improved & (best_r >= 1e-12)]
    return x, res


def invert_residual_lut(coded, lut):
    """LUT 역(고정점/뉴턴). (coded f64, ok bool (...,)): 잔차 >= 1e-5 이면 ok=False (치역 밖)."""
    y = _coded_arr(coded, "coded")
    L = _lut_arr(lut)
    x, res = _lut_invert(y.reshape(-1, 3), L)
    return x.reshape(y.shape), (res < _LUT_OK_TOL).reshape(y.shape[:-1])


# ---------------------------------------------------------------------------------------------------------------
# 순방향 / 역방향
# ---------------------------------------------------------------------------------------------------------------
def _check_params(params):
    if not isinstance(params, ToneParams):
        raise TypeError("params: ToneParams 여야 한다")
    return params


def _gains(params):
    return _chain.wb_gains(params.wb_kelvin, params.wb_tint)


def _lut64(params):
    return None if params.lut is None else params.lut.astype(np.float64)


def _forward64(L, params):
    g = _gains(params)
    x = L * g * 2.0 ** params.ev
    y = np.clip(_curve_core(x, *_curve_args(params.curve)), 0.0, 1.0)
    c = _enc(y)
    if not params.cdl.is_identity():
        c = np.clip(_apply_cdl(c, *_cdl_vecs(params.cdl)), 0.0, 1.0)
    if params.lut is not None:
        c = np.clip(_trilinear(c.reshape(-1, 3), _lut64(params)).reshape(c.shape), 0.0, 1.0)
    return _dec(c)


def forward(scene_lin, params) -> np.ndarray:
    """장면 선형 (...,3) -> 표시 선형 D (float32)."""
    _check_params(params)
    L = _rgb_arr(scene_lin, "scene_lin")
    return _forward64(L, params).astype(np.float32)


def inverse(display_lin, params) -> ToneInverse:
    """표시 선형 -> 중립 룩 장면 선형 L̂ (WB·EV 제거) 와 소유 상태."""
    _check_params(params)
    D = _rgb_arr(display_lin, "display_lin")
    shape = D.shape[:-1]
    D = D.reshape(-1, 3)
    n = D.shape[0]
    status = np.zeros(n, np.uint8)
    has_cdl = not params.cdl.is_identity()
    has_lut = params.lut is not None
    hi_disp = np.any(D >= 1.0 - _CLIP_EDGE, axis=1)
    c = _enc(D)
    status[hi_disp] |= OWN_CLIP_HI
    if has_cdl or has_lut:
        status[np.any(D <= _CLIP_EDGE, axis=1)] |= OWN_CLIP_LO
    c = np.minimum(c, 1.0)
    if has_lut:
        c, ok = _lut_invert(c, _lut64(params))
        ok = ok < _LUT_OK_TOL
        status[~ok] |= OWN_LUT_LOST
        # LUT 입력(= CDL 출력 클립 후 값)이 경계에 붙었으면 클립 되었을 수 있다
        status[np.any(c >= 1.0 - 1e-4, axis=1)] |= OWN_CLIP_HI
        status[np.any(c <= 1e-4, axis=1)] |= OWN_CLIP_LO
    if has_cdl:
        c, lost = _invert_cdl(c, *_cdl_vecs(params.cdl))
        status[lost] |= OWN_CLIP_LO
        status[np.any(c >= 1.0 - 1e-4, axis=1)] |= OWN_CLIP_HI
        status[np.any(c <= 1e-4, axis=1)] |= OWN_CLIP_LO
        c = np.clip(c, 0.0, 1.0)
    y = _dec(c)
    x, clipped = _curve_inv_core(y, *_curve_args(params.curve))
    status[np.any(clipped, axis=1)] |= OWN_CLIP_HI
    Lh = x / (_gains(params) * 2.0 ** params.ev)
    return ToneInverse(linear=Lh.reshape(shape + (3,)).astype(np.float32), status=status.reshape(shape))


def bake_lut(params, size=65) -> np.ndarray:
    """WB·EV 이후(F·CDL·LUT)만 구운 LNS-04 룩 LUT: 노드 c(로그 부호화) -> srgb_enc(F(log_decode(c))) 합성."""
    _check_params(params)
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
        raise ValueError(f"size={size!r}: 2 이상의 정수여야 한다")
    size = int(size)
    if size < 2:
        raise ValueError(f"size={size}: 2 이상의 정수여야 한다")
    e = np.linspace(0.0, 1.0, size)
    r, g, b = np.meshgrid(e, e, e, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)
    x = np.asarray(_sf.log_decode(grid), np.float64)
    y = np.clip(_curve_core(x, *_curve_args(params.curve)), 0.0, 1.0)
    c = _enc(y)
    if not params.cdl.is_identity():
        c = np.clip(_apply_cdl(c, *_cdl_vecs(params.cdl)), 0.0, 1.0)
    if params.lut is not None:
        c = np.clip(_trilinear(c.reshape(-1, 3), _lut64(params)).reshape(c.shape), 0.0, 1.0)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def theta_fields(params) -> dict:
    _check_params(params)
    return {"ev": float(params.ev), "wb_kelvin": float(params.wb_kelvin), "wb_tint": float(params.wb_tint)}


# ---------------------------------------------------------------------------------------------------------------
# 룩 잔차 LUT 적합 / r_look
# ---------------------------------------------------------------------------------------------------------------
_EDGE_CACHE: dict = {}


def _edge_op():
    if "G" not in _EDGE_CACHE:
        N = LUT_N
        idx = np.arange(N ** 3).reshape(N, N, N)
        ea, eb = [], []
        for ax in range(3):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[ax], b[ax] = slice(0, -1), slice(1, None)
            ea.append(idx[tuple(a)].ravel())
            eb.append(idx[tuple(b)].ravel())
        ea, eb = np.concatenate(ea), np.concatenate(eb)
        E = len(ea)
        G = _sp.csr_matrix((np.concatenate([np.ones(E), -np.ones(E)]),
                            (np.concatenate([np.arange(E)] * 2), np.concatenate([ea, eb]))), shape=(E, N ** 3))
        _EDGE_CACHE["G"] = (G, E, (G.T @ G).tocsc())
    return _EDGE_CACHE["G"]


def _design(sc):
    N = LUT_N
    n = len(sc)
    p = sc * (N - 1)
    i0 = np.minimum(np.floor(p).astype(np.intp), N - 2)
    f = p - i0
    rows, cols, vals = [], [], []
    ar = np.arange(n)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                wgt = ((f[:, 0] if dr else 1 - f[:, 0]) * (f[:, 1] if dg else 1 - f[:, 1])
                       * (f[:, 2] if db else 1 - f[:, 2]))
                rows.append(ar)
                cols.append(((i0[:, 0] + dr) * N + i0[:, 1] + dg) * N + i0[:, 2] + db)
                vals.append(wgt)
    return _sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, N ** 3))


def _lin_pair(a, b, na, nb):
    A = _float_arr(a, na)
    B = _float_arr(b, nb)
    if A.ndim < 1 or A.shape[-1] != 3:
        raise ValueError(f"{na}: 마지막 축이 3 이어야 한다 (shape {A.shape})")
    if A.shape != B.shape:
        raise ValueError(f"{na}/{nb}: 모양이 같아야 한다 ({A.shape} vs {B.shape})")
    return A.astype(np.float64), B.astype(np.float64)


def fit_residual_lut(source_lin, target_lin, *, weight=None, smooth=0.01, ridge=0.001) -> np.ndarray:
    """source(물리 θ̂ 로 다시 찍은 표시 선형) -> target(관측) 의 잔차 LUT 17^3 (float32).

    J = (1/N) Σ w̃ ‖tri·R - d‖² + smooth (1/E) Σ_edge ‖R_a - R_b‖² + ridge (1/V) Σ ‖R‖², R = LUT - 항등 (볼록, 유일해)."""
    src, tgt = _lin_pair(source_lin, target_lin, "source_lin", "target_lin")
    smooth = _num("smooth", smooth, 0.0)
    ridge = _num("ridge", ridge, 0.0, lo_open=True)
    sc = np.clip(_enc(np.clip(src, 0.0, None)).reshape(-1, 3), 0.0, 1.0)
    tc = np.clip(_enc(np.clip(tgt, 0.0, None)).reshape(-1, 3), 0.0, 1.0)
    n = sc.shape[0]
    if weight is None:
        w = np.ones(n)
    else:
        w = np.asarray(weight)
        if w.dtype.kind not in "fiu":
            raise TypeError("weight: 수치 배열이어야 한다")
        w = w.astype(np.float64).ravel()
        if w.size != n:
            raise ValueError(f"weight: 화소 수 {n} 와 같아야 한다 ({w.size})")
        if not np.all(np.isfinite(w)) or float(w.min()) < 0.0:
            raise ValueError("weight: 유한 비음수여야 한다")
    if not float(w.sum()) > 0.0:
        raise ValueError("weight: 합이 양수여야 한다")
    wt = w / w.mean()
    d = tc - sc
    V = LUT_N ** 3
    _, E, GtG = _edge_op()
    keep = wt > 0.0
    A = _design(sc[keep])
    Aw = _sp.diags(wt[keep]) @ A
    M = (A.T @ Aw) / n + (smooth / E) * GtG + (ridge / V) * _sp.identity(V, format="csc")
    rhs = (Aw.T @ d[keep]) / n
    lu = _spla.splu(M.tocsc())
    R = lu.solve(np.ascontiguousarray(rhs))
    ident = np.asarray(_sf.identity_lut(LUT_N), np.float64).reshape(V, 3)
    return (ident + R).reshape(LUT_N, LUT_N, LUT_N, 3).astype(np.float32)


def r_look(observed_lin, predicted_lin, *, sigma_frac=0.02) -> float:
    """부호화 영역 ‖lp(O-P)‖ / ‖lp(O)‖, lp = 가우시안 σ = sigma_frac * W. 분모 ≈ 0 이면 0.0."""
    O, P = _lin_pair(observed_lin, predicted_lin, "observed_lin", "predicted_lin")
    if O.ndim != 3:
        raise ValueError(f"observed_lin: (H,W,3) 이어야 한다 (shape {O.shape})")
    sigma_frac = _num("sigma_frac", sigma_frac, 0.0, lo_open=True)
    co = _enc(np.clip(O, 0.0, None))
    cp = _enc(np.clip(P, 0.0, None))
    s = sigma_frac * O.shape[1]
    lp_d = _ndi.gaussian_filter(co - cp, sigma=(s, s, 0.0), mode="nearest")
    lp_o = _ndi.gaussian_filter(co, sigma=(s, s, 0.0), mode="nearest")
    den = float(np.linalg.norm(lp_o))
    if den < 1e-12:
        return 0.0
    return float(np.linalg.norm(lp_d) / den)


# ---------------------------------------------------------------------------------------------------------------
# 추정: 입력 객체
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class SkyAnchor:
    mask: object
    luminance: float

    def __post_init__(self):
        m = self.mask
        if not isinstance(m, np.ndarray) or m.dtype != np.bool_:
            raise TypeError("SkyAnchor.mask: bool ndarray 여야 한다")
        if m.ndim != 2 or m.size == 0:
            raise ValueError(f"SkyAnchor.mask: (H,W) 이어야 한다 (shape {m.shape})")
        m = m.copy()
        m.setflags(write=False)
        object.__setattr__(self, "mask", m)
        object.__setattr__(self, "luminance", _num("SkyAnchor.luminance", self.luminance, 0.0, lo_open=True))


@dataclass(frozen=True, kw_only=True)
class TonePriors:
    ev: object = None
    wb_kelvin: object = None
    wb_tint: object = None
    ev_sigma: float = 1.0
    wb_mired_sigma: float = 120.0
    wb_tint_sigma: float = 0.15
    lum_median: float = 0.18
    lum_sigma_stops: float = 1.5
    chroma_log_sigma: float = 0.35
    sky: object = None

    def __post_init__(self):
        for n, lo, hi in (("ev", -5.0, 5.0), ("wb_kelvin", 2000.0, 12000.0), ("wb_tint", -0.6, 0.6)):
            v = getattr(self, n)
            if v is not None:
                object.__setattr__(self, n, _num(f"TonePriors.{n}", v, lo, hi))
        for n in ("ev_sigma", "wb_mired_sigma", "wb_tint_sigma", "lum_median", "lum_sigma_stops", "chroma_log_sigma"):
            object.__setattr__(self, n, _num(f"TonePriors.{n}", getattr(self, n), 0.0, lo_open=True))
        if self.sky is not None and not isinstance(self.sky, SkyAnchor):
            raise TypeError("TonePriors.sky: SkyAnchor 여야 한다")


@dataclass(frozen=True, kw_only=True)
class ToneOpts:
    fit_sat: bool = False
    max_samples: int = 20000
    min_range_stops: float = 1.0
    poor_fit_stops: float = 0.35
    clip_dominant_frac: float = 0.4

    def __post_init__(self):
        if not isinstance(self.fit_sat, (bool, np.bool_)):
            raise TypeError("ToneOpts.fit_sat: bool 이어야 한다")
        object.__setattr__(self, "fit_sat", bool(self.fit_sat))
        if not _is_int(self.max_samples):
            raise TypeError("ToneOpts.max_samples: int 여야 한다")
        if int(self.max_samples) < 256:
            raise ValueError("ToneOpts.max_samples: 256 이상이어야 한다")
        object.__setattr__(self, "max_samples", int(self.max_samples))
        object.__setattr__(self, "min_range_stops", _num("ToneOpts.min_range_stops", self.min_range_stops, 0.0, lo_open=True))
        object.__setattr__(self, "poor_fit_stops", _num("ToneOpts.poor_fit_stops", self.poor_fit_stops, 0.0, lo_open=True))
        object.__setattr__(self, "clip_dominant_frac", _num("ToneOpts.clip_dominant_frac", self.clip_dominant_frac, 0.0, 1.0))


@dataclass(frozen=True)
class ToneColor:
    params: ToneParams
    reliable: bool
    warnings: tuple
    fit_rms_stops: float
    n_used: int
    clip_hi_frac: float
    clip_lo_frac: float

    def to_meta(self) -> dict:
        p = self.params
        rms = float(self.fit_rms_stops)
        return {
            "ev": float(p.ev), "wb_kelvin": float(p.wb_kelvin), "wb_tint": float(p.wb_tint),
            "curve": {"toe": float(p.curve.toe), "shoulder": float(p.curve.shoulder),
                      "slope": float(p.curve.slope), "white": float(p.curve.white)},
            "cdl": {"slope": [float(v) for v in p.cdl.slope], "offset": [float(v) for v in p.cdl.offset],
                    "power": [float(v) for v in p.cdl.power], "sat": float(p.cdl.sat)},
            "lut_present": p.lut is not None,
            "reliable": bool(self.reliable),
            "warnings": list(self.warnings),
            "fit_rms_stops": rms if math.isfinite(rms) else None,
            "n_used": int(self.n_used),
            "clip_hi_frac": float(self.clip_hi_frac),
            "clip_lo_frac": float(self.clip_lo_frac),
        }


# ---------------------------------------------------------------------------------------------------------------
# 추정: 내부
# ---------------------------------------------------------------------------------------------------------------
_N_FIT = 4000                 # 최적화에 쓰는 최대 화소 수(균등 솎기, 결정론)
_Q_LEVELS = np.linspace(0.03, 0.97, 33)
_S_Q = 0.25                   # 분위수 잔차 시그마(stop)
_S_GRAY = 0.02                # 회색 세계 시그마(ln)
_S_SKY = 0.15                 # 하늘 앵커 시그마(stop)
_PRIOR_TOE = 1.0
_PRIOR_SHOULDER = 1.0
_PRIOR_LN_SLOPE = 0.5
_PRIOR_LN_WHITE = 0.5
_M_LO = 1e6 / 12000.0 + 1e-6
_M_HI = 1e6 / 2000.0 - 1e-6


def _gains_mired(m, tint):
    return _chain.wb_gains(1e6 / m, tint)


class _Problem:
    """고정 화소 집합(솎은 부분집합)과 클립 비율로 정의되는 잔차 문제."""

    def __init__(self, y, f_lo, f_u, sky_sel, priors, shape_only):
        self.y = y                              # (m,3) 곡선 출력 y 로 환원된 값(미클립 화소)
        self.m = y.shape[0]
        self.f_lo = f_lo
        self.f_u = f_u
        self.sky_sel = sky_sel                  # (m,) bool | None
        self.pr = priors
        self.shape_only = shape_only
        self.gw_ok = np.all(y > 1e-9, axis=1)
        self.ev0 = 0.0 if priors.ev is None else priors.ev
        self.m0 = 1e6 / (6500.0 if priors.wb_kelvin is None else priors.wb_kelvin)
        self.t0 = 0.0 if priors.wb_tint is None else priors.wb_tint
        self.z = _ndtri(_Q_LEVELS)
        self.q_target = math.log2(priors.lum_median) + priors.lum_sigma_stops * self.z
        self.white_lb = float(np.max(y)) / (1.0 - _CLIP_EDGE) * (1.0 + 1e-9) if self.m else 0.5

    def unpack(self, th):
        ev, m, tint, toe, sh, lsl, lwh = th
        return ev, m, tint, toe, sh, math.exp(lsl), math.exp(lwh)

    def lhat(self, th):
        ev, m, tint, toe, sh, sl, wh = self.unpack(th)
        x, clipped = _curve_inv_core(self.y, toe, sh, sl, wh)
        g = _gains_mired(m, tint)
        return x / (g * 2.0 ** ev), np.any(clipped, axis=1)

    def quantile_resid(self, lum_log2_sorted):
        """usable 분위 수준의 (관측 - 목표) stop. 반환 (resid, usable bool)."""
        mu = lum_log2_sorted.size
        if mu < 2:
            return np.zeros(_Q_LEVELS.size), np.zeros(_Q_LEVELS.size, bool)
        pos = (_Q_LEVELS - self.f_lo) / self.f_u * self.m - 0.5
        usable = (pos >= 0.0) & (pos <= mu - 1.0)
        obs = np.interp(np.clip(pos, 0.0, mu - 1.0), np.arange(mu), lum_log2_sorted)
        return obs - self.q_target, usable

    def residuals(self, th, detail=False):
        ev, m, tint, toe, sh, sl, wh = self.unpack(th)
        L, clipped = self.lhat(th)
        keep = ~clipped
        lum = L @ _W709
        lk = np.sort(np.log2(np.maximum(lum[keep], 1e-12)))
        r, usable = self.quantile_resid(lk)
        nu = int(usable.sum())
        if self.shape_only and nu:
            r = r - float(r[usable].mean())
        rq = np.where(usable, r, 0.0) / _S_Q
        rms = float(np.sqrt(np.mean(r[usable] ** 2))) if nu else float("nan")
        sel = keep & self.gw_ok
        if sel.sum() >= 2:
            lL = np.log(np.maximum(L[sel], 1e-12))
            gw = np.array([np.mean(lL[:, 0] - lL[:, 1]), np.mean(lL[:, 2] - lL[:, 1])]) / _S_GRAY
        else:
            gw = np.zeros(2)
        res = [rq, gw]
        if self.sky_sel is not None:
            ss = self.sky_sel & keep
            if ss.sum() >= 8:
                res.append(np.array([(float(np.median(np.log2(np.maximum(lum[ss], 1e-12)))) - math.log2(self.pr.sky.luminance)) / _S_SKY]))
            else:
                res.append(np.zeros(1))
        pr = self.pr
        res.append(np.array([
            (ev - self.ev0) / pr.ev_sigma, (m - self.m0) / pr.wb_mired_sigma, (tint - self.t0) / pr.wb_tint_sigma,
            toe / _PRIOR_TOE, sh / _PRIOR_SHOULDER, math.log(sl) / _PRIOR_LN_SLOPE, math.log(wh) / _PRIOR_LN_WHITE]))
        out = np.concatenate(res)
        return (out, rms) if detail else out

    def init(self):
        """회색 세계 + 중앙값 휘도로 만든 시작점."""
        y = self.y
        ok = self.gw_ok
        if ok.sum() >= 2:
            lr = np.log(y[ok])
            a1 = float(np.mean(lr[:, 0] - lr[:, 1]))
            a2 = float(np.mean(lr[:, 2] - lr[:, 1]))
        else:
            a1 = a2 = 0.0
        ln2 = math.log(2.0)
        tint = (a1 + a2) / (2.0 * ln2 * _chain.WB_TINT_STOPS)
        mm = (a2 - a1) / (2.0 * ln2 * _chain.WB_MIRED_STOPS)
        m = float(np.clip(_WB_REF_M + mm, _M_LO + 1.0, _M_HI - 1.0))
        tint = float(np.clip(tint, -0.55, 0.55))
        g = _gains_mired(m, tint)
        med = float(np.median((y / g) @ _W709))
        ev = float(np.clip(math.log2(max(med, 1e-6) / MID_GRAY) + 0.0, -4.5, 4.5))
        # 사전분포가 주어졌으면 그 가까이에서 시작
        if self.pr.ev is not None:
            ev = float(self.pr.ev) if abs(ev - self.pr.ev) > 2.0 else ev
        return np.array([ev, m, tint, 0.0, 0.0, 0.0, math.log(max(1.0, self.white_lb * 1.0005))])

    def bounds(self):
        lo = np.array([-5.0, _M_LO, -0.6, 0.0, 0.0, math.log(0.5), math.log(max(0.5, self.white_lb))])
        hi = np.array([5.0, _M_HI, 0.6, 1.0, 1.0, math.log(2.0), math.log(2.0)])
        hi[6] = max(hi[6], lo[6] + 1e-6)
        return lo, hi


def _solve1(prob, x0):
    lo, hi = prob.bounds()
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    scale = np.array([0.5, 30.0, 0.1, 0.3, 0.3, 0.2, 0.2])
    sol = _least_squares(lambda t: prob.residuals(t), x0, bounds=(lo, hi), method="trf", x_scale=scale,
                         diff_step=1e-4, max_nfev=120, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    return sol.x, float(2.0 * sol.cost)


# 곡선 시작점 (toe, shoulder, ln slope): 경계(어깨 0)에서 시작하면 국소해에 걸리므로 내부점에서 여러 개 시작한다
_CURVE_STARTS = ((0.3, 0.3, 0.0), (0.0, 0.7, 0.1), (0.7, 0.1, -0.1))


def _solve(prob, x0, starts=_CURVE_STARTS):
    best = None
    for st in starts:
        x = np.array(x0, np.float64)
        if st is not None:
            x[3], x[4], x[5] = st
        sol, cost = _solve1(prob, x)
        if best is None or cost < best[1] - 1e-9:
            best = (sol, cost)
    return best[0]


def _params_from(prob, th, sat=1.0):
    ev, m, tint, toe, sh, sl, wh = prob.unpack(th)
    return ToneParams(
        ev=float(np.clip(ev, -5.0, 5.0)), wb_kelvin=float(np.clip(1e6 / m, 2000.0, 12000.0)),
        wb_tint=float(np.clip(tint, -0.6, 0.6)),
        curve=FilmCurve(toe=float(np.clip(toe, 0.0, 1.0)), shoulder=float(np.clip(sh, 0.0, 1.0)),
                        slope=float(np.clip(sl, 0.5, 2.0)), white=float(np.clip(wh, 0.5, 2.0))),
        cdl=CDL(sat=float(np.clip(sat, 0.05, 2.0))) if sat != 1.0 else CDL())


def _prior_params(priors):
    return ToneParams(ev=0.0 if priors.ev is None else priors.ev,
                      wb_kelvin=6500.0 if priors.wb_kelvin is None else priors.wb_kelvin,
                      wb_tint=0.0 if priors.wb_tint is None else priors.wb_tint)


def _desat(c, sat):
    """CDL sat 역(부호화 영역): v = Y + (c - Y)/sat. 반환 (v, ok)."""
    Y = (c @ _W709)[:, None]
    v = Y + (c - Y) / sat
    ok = np.all((v > _CDL_LOST) & (v < 1.0), axis=1)
    return v, ok


def _joint_chroma_std(y_lin, prob_params, th, prob):
    """곡선·WB·EV 역을 거친 L̂ 의 (ln R/G, ln B/G) 합동 표준편차."""
    ev, m, tint, toe, sh, sl, wh = prob.unpack(th)
    x, clipped = _curve_inv_core(y_lin, toe, sh, sl, wh)
    g = _gains_mired(m, tint)
    L = x / (g * 2.0 ** ev)
    sel = ~np.any(clipped, axis=1) & np.all(L > 1e-9, axis=1)
    if sel.sum() < 16:
        return float("nan"), int(sel.sum())
    lL = np.log(L[sel])
    a, b = lL[:, 0] - lL[:, 1], lL[:, 2] - lL[:, 1]
    return float(math.sqrt(0.5 * (np.var(a) + np.var(b)))), int(sel.sum())


def _validate_image(image_lin):
    a = _float_arr(image_lin, "image_lin")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image_lin: (H,W,3) 이어야 한다 (shape {a.shape})")
    if a.shape[0] < 8 or a.shape[1] < 8:
        raise ValueError(f"image_lin: H,W >= 8 이어야 한다 (shape {a.shape})")
    if float(a.min()) < 0.0:
        raise ValueError("image_lin: 음수가 있다")
    return a


def estimate_tone_color(image_lin, *, priors=None, opts=None) -> ToneColor:
    """표시 선형 영상 -> G 의 역추정(ev, WB, 4파라미터 곡선[, sat]). CDL(sat 외)·LUT 는 추정하지 않는다."""
    a = _validate_image(image_lin)
    if priors is None:
        priors = TonePriors()
    elif not isinstance(priors, TonePriors):
        raise TypeError("priors: TonePriors 여야 한다")
    if opts is None:
        opts = ToneOpts()
    elif not isinstance(opts, ToneOpts):
        raise TypeError("opts: ToneOpts 여야 한다")
    H, W = a.shape[:2]
    sky = priors.sky
    if sky is not None and sky.mask.shape != (H, W):
        raise ValueError(f"sky.mask: 영상 {(H, W)} 와 같은 모양이어야 한다 ({sky.mask.shape})")

    k = max(1, math.ceil(math.sqrt(H * W / opts.max_samples)))
    sub = a[::k, ::k].astype(np.float64)
    sky_sub = None if sky is None else sky.mask[::k, ::k].reshape(-1)
    D = sub.reshape(-1, 3)
    n_used = D.shape[0]
    c = _enc(D)
    hi = np.any(c >= 1.0 - _CLIP_EDGE, axis=1)
    lo = np.all(c <= _CLIP_EDGE, axis=1)
    unc = ~hi & ~lo
    clip_hi_frac = float(hi.sum() / n_used)
    clip_lo_frac = float(lo.sum() / n_used)
    n_u = int(unc.sum())

    warns = set()
    if n_u < 64:
        warns.add("insufficient_pixels")
    if clip_hi_frac > opts.clip_dominant_frac:
        warns.add("clipped_dominant")
    if clip_lo_frac > opts.clip_dominant_frac:
        warns.add("black_dominant")
    if n_u >= 8:
        ll = np.log2(np.maximum(D[unc] @ _W709, 1e-6))
        p5, p95 = np.percentile(ll, [5.0, 95.0])
        if p95 - p5 < opts.min_range_stops:
            warns.add("insufficient_dynamic_range")
    can_fit = not ({"insufficient_pixels", "insufficient_dynamic_range"} & warns)

    def finish(params, rms):
        w = tuple(sorted(warns))
        return ToneColor(params=params, reliable=not w, warnings=w, fit_rms_stops=rms, n_used=int(n_used),
                         clip_hi_frac=clip_hi_frac, clip_lo_frac=clip_lo_frac)

    if not can_fit:
        return finish(_prior_params(priors), float("nan"))

    # 최적화용 부분집합: 미클립 화소를 균등 솎기
    ui = np.flatnonzero(unc)
    if ui.size > _N_FIT:
        ui = ui[np.linspace(0, ui.size - 1, _N_FIT).round().astype(np.intp)]
    f_lo = clip_lo_frac
    f_u = n_u / n_used
    y_all = _dec(c[ui])
    sky_sel = None if sky_sub is None else sky_sub[ui]
    shape_only = sky is not None
    prob = _Problem(y_all, f_lo, f_u, sky_sel, priors, shape_only)
    th = _solve(prob, prob.init())
    sat = 1.0
    if opts.fit_sat:
        # 채도: (ln R/G, ln B/G) 합동 표준편차가 chroma_log_sigma 가 되도록
        cu = c[ui]
        grid = np.geomspace(0.05, 2.0, 60)
        best = None
        for s in grid:
            v, ok = _desat(cu, s)
            if ok.mean() < 0.5:
                continue
            sd, cnt = _joint_chroma_std(_dec(v[ok]), None, th, prob)
            if not math.isfinite(sd):
                continue
            err = abs(sd - priors.chroma_log_sigma)
            if best is None or err < best[0]:
                best = (err, float(s))
        if best is not None:
            sat = best[1]
            v, ok = _desat(cu, sat)
            if ok.sum() >= 64:
                prob2 = _Problem(_dec(v[ok]), f_lo, f_u * ok.mean(), None if sky_sel is None else sky_sel[ok],
                                 priors, shape_only)
                th = _solve(prob2, th, starts=(None,))
                prob = prob2
    res, rms = prob.residuals(th, detail=True)
    if not math.isfinite(rms) or rms > opts.poor_fit_stops:
        warns.add("poor_fit")
    return finish(_params_from(prob, th, sat), rms if math.isfinite(rms) else float("nan"))
