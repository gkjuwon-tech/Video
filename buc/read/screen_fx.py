"""RD-03: R2 화면 고정 효과 추정기 v0 -- 비네팅 / 그레인 PSD / 횡색수차 / 할레이션 / 플레어 고스트.

이미지 평면에 붙은 효과(L3)를 RD-01 ``image_lin`` 한 장에서 고전 통계로 측정한다(LNS-04 순방향의 역문제, 학습 파라미터 0).
numpy / scipy 만 쓴다(torch / cv2 / skimage 금지). import 부작용 없음, 전역 RNG 불변(난수 불사용), 순수·결정론, 입력 불변.
결과는 모두 ``frozen kw_only`` 데이터클래스이며 배열 필드는 없다(튜플).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares, nnls

from buc.core import color as _color

__all__ = [
    "GRAIN_SIGMA_MAX", "VIGNETTE_MAX_DROP", "HALATION_THRESHOLD", "HALATION_SIGMA_FRAC", "HALATION_TINT",
    "HALO_SOURCE_MAX", "HALO_ASSUMED_SOURCE_Y",
    "ScreenFxError", "ScreenFxOpts", "VignetteFit", "GrainFit", "CAFit", "HaloBlob", "HalationFit", "GhostSpot",
    "GhostFit", "ScreenFx", "estimate_vignette", "estimate_grain", "estimate_ca", "estimate_halation",
    "estimate_ghost", "estimate_screen_fx", "identity_screen_fx",
]

# LNS-04 정본 상수(추정 모델이 순방향과 어긋나면 안 된다), LNS-08 HALO_SOURCE_MAX, 본 단위 HALO_ASSUMED_SOURCE_Y
GRAIN_SIGMA_MAX = 0.08
VIGNETTE_MAX_DROP = 0.8
HALATION_THRESHOLD = 0.8
HALATION_SIGMA_FRAC = 0.02
HALATION_TINT = (1.0, 0.35, 0.12)
HALO_SOURCE_MAX = 16.0
HALO_ASSUMED_SOURCE_Y = 2.0

_NAN = float("nan")                    # 같은 객체를 재사용해 프로즌 데이터클래스 ``==`` 가 nan 에서도 성립하게 한다
_MIN_SIDE = 32
_SMALL_SIDE = 64
_VIG_GRID = np.linspace(0.0, 1.0, 101)
_VIG_FLOOR = 0.05
_LUMA = np.array([0.2126, 0.7152, 0.0722])

W_SMALL = "small_image"
W_FEW_FLAT = "few_flat_blocks"
W_FEW_LEVELS = "few_levels"
W_JPEG = "jpeg_lossy"
W_CA_NO_COLOR = "ca_no_color"
W_NO_SAT = "no_saturation"
W_NEUTRAL = "neutral_halo"
W_EXTRAP = "vignette_extrapolated"


class ScreenFxError(ValueError):
    """입력이 너무 작아 화면 고정 효과를 읽을 수 없다."""


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _f(v) -> float:
    """유한하지 않은 값은 단일 nan 객체로 통일한 Python float."""
    v = float(v)
    return v if math.isfinite(v) else _NAN


def _tup(a) -> tuple:
    return tuple(_f(v) for v in np.asarray(a, np.float64).ravel())


@dataclass(frozen=True, kw_only=True)
class ScreenFxOpts:
    max_side: int = 1024
    block: int = 16
    sat_level: float = 0.98
    min_blob_px: int = 30
    ghost_max: int = 4

    def __post_init__(self):
        for n, lo, hi in (("max_side", 64, None), ("block", 8, 64), ("min_blob_px", 4, None), ("ghost_max", 0, None)):
            v = getattr(self, n)
            if not _is_int(v):
                raise ValueError(f"{n}={v!r}: 정수여야 한다")
            if v < lo or (hi is not None and v > hi):
                raise ValueError(f"{n}={v}: 범위 밖")
        s = self.sat_level
        if not _is_real(s) or not math.isfinite(float(s)) or not (0.5 < float(s) <= 1.0):
            raise ValueError(f"sat_level={s!r}: (0.5, 1] 이어야 한다")


def _opts(opts) -> ScreenFxOpts:
    if opts is None:
        return ScreenFxOpts()
    if not isinstance(opts, ScreenFxOpts):
        raise TypeError("opts: ScreenFxOpts 또는 None 이어야 한다")
    return opts


def _check_image(x) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError("image_lin: numpy ndarray 여야 한다")
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"image_lin: float32/float64 여야 한다 (dtype {x.dtype})")
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"image_lin: (H,W,3) 이어야 한다 (shape {x.shape})")
    if not np.all(np.isfinite(x)):
        raise ValueError("image_lin: NaN/inf 가 있다")
    if min(x.shape[:2]) < _MIN_SIDE:
        raise ScreenFxError(f"image_lin: 짧은 변이 {_MIN_SIDE} 미만이다 (shape {x.shape})")
    return np.maximum(x.astype(np.float64), 0.0)


def _xy(name, v):
    if v is None:
        return None
    try:
        a = [x for x in v]
    except TypeError:
        raise ValueError(f"{name}: 길이 2 의 수열이어야 한다") from None
    if len(a) != 2 or not all(_is_real(t) and math.isfinite(float(t)) for t in a):
        raise ValueError(f"{name}: 유한한 실수 2 개여야 한다")
    return float(a[0]), float(a[1])


def _norm_r(v):
    if v is None:
        return None
    if not _is_real(v) or not math.isfinite(float(v)) or float(v) <= 0.0:
        raise ValueError("r_norm: 양의 유한 실수여야 한다")
    return float(v)


def _default_center(h, w):
    return ((w - 1) / 2.0, (h - 1) / 2.0)


def _half_diag(h, w):
    return float(math.hypot((w - 1) / 2.0, (h - 1) / 2.0))


def _downscale(lin, opts):
    """긴 변이 max_side 를 넘으면 정수배 블록 평균. 반환 (영상, 배율 f). 작은 화소 i 의 원 좌표 = i*f + (f-1)/2."""
    h, w = lin.shape[:2]
    f = max(1, int(math.ceil(max(h, w) / float(opts.max_side))))
    if f == 1:
        return lin, 1
    hh, ww = h // f, w // f
    s = lin[:hh * f, :ww * f].reshape(hh, f, ww, f, 3).mean(axis=(1, 3))
    return s, f


def _coords(hs, ws, f):
    xs = np.arange(ws, dtype=np.float64) * f + (f - 1) / 2.0
    ys = np.arange(hs, dtype=np.float64) * f + (f - 1) / 2.0
    return xs, ys


def _luma(a):
    return a @ _LUMA


def _warn(*items) -> tuple:
    return tuple(sorted({w for w in items if w}))


def _small_warn(h, w):
    return W_SMALL if min(h, w) < _SMALL_SIDE else None


def _huber_fit(A, y, delta, iters=8, w0=None):
    """Huber IRLS 선형 최소제곱. 반환 (계수, 비용)."""
    n = len(y)
    w = np.ones(n) if w0 is None else np.asarray(w0, np.float64)
    coef = np.zeros(A.shape[1])
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ coef
        a = np.abs(r)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))
        if w0 is not None:
            w = w * w0
    r = y - A @ coef
    a = np.abs(r)
    cost = float(np.sum(np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))))
    return coef, cost


# ---------------------------------------------------------------------------------------------------------------
# 비네팅
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class VignetteFit:
    coeffs: tuple
    center: tuple
    r_norm: float
    shape: tuple
    corner_ev: float
    strength: float
    rms_resid_ev: float
    reliable: bool
    curve: tuple
    warnings: tuple = ()

    def gain_at(self, rho):
        """V(ρ): V(0)=1, 비증가, [0.05,1]. ρ>1 은 V(1)."""
        r = np.clip(np.asarray(rho, np.float64), 0.0, 1.0)
        return np.interp(r, _VIG_GRID, np.asarray(self.curve, np.float64))

    def gain(self) -> np.ndarray:
        h, w = self.shape
        xs = np.arange(w, dtype=np.float64) - self.center[0]
        ys = np.arange(h, dtype=np.float64) - self.center[1]
        rho = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) / self.r_norm
        return np.asarray(self.gain_at(rho)).astype(np.float32)


def _vig_from_curve(curve, h, w, center, r_norm, rms, reliable, warnings):
    c = np.asarray(curve, np.float64)
    v1 = float(c[-1])
    r2 = _VIG_GRID ** 2
    X = np.stack([r2, r2 ** 2, r2 ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(X, c - 1.0, rcond=None)
    coef = np.asarray(coef)
    if coef[0] > 0.0:
        coef = np.zeros(3)
    return VignetteFit(
        coeffs=_tup(coef), center=(float(center[0]), float(center[1])), r_norm=float(r_norm), shape=(int(h), int(w)),
        corner_ev=_f(math.log2(v1)), strength=_f(min(1.0, max(0.0, (1.0 - v1) / VIGNETTE_MAX_DROP))),
        rms_resid_ev=_f(rms), reliable=bool(reliable), curve=_tup(c), warnings=_warn(*warnings))


def _vignette_core(lin_full, center, r_norm, opts) -> VignetteFit:
    h, w = lin_full.shape[:2]
    cx, cy = center if center is not None else _default_center(h, w)
    rn = r_norm if r_norm is not None else _half_diag(h, w)
    center_t = (cx, cy)
    warns = [_small_warn(h, w)]
    lin, f = _downscale(lin_full, opts)
    hs, ws = lin.shape[:2]
    Y = _luma(lin)
    valid = (Y > 0.004) & (lin.max(axis=2) < 0.985)
    xs, ys = _coords(hs, ws, f)
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    rho = np.sqrt(dx * dx + dy * dy) / rn
    ang = np.arctan2(dy, dx)
    rho_v, ang_v, ly = rho[valid], ang[valid], np.log2(Y[valid])
    identity = np.ones_like(_VIG_GRID)
    if ly.size < 50:
        return _vig_from_curve(identity, h, w, center_t, rn, _NAN, False, warns)
    ring = np.minimum((rho_v / 0.05).astype(np.int64), 60)
    sector = np.minimum(((ang_v + math.pi) / (2.0 * math.pi) * 12.0).astype(np.int64), 11)
    cell = ring * 12 + sector
    ncell = 61 * 12
    order = np.lexsort((ly, cell))
    cs, yv = cell[order], ly[order]
    counts = np.bincount(cell, minlength=ncell)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ok = counts >= 12
    if ok.sum() < 8:
        return _vig_from_curve(identity, h, w, center_t, rn, _NAN, False, warns)
    lo_i = starts[ok] + (counts[ok] - 1) // 2
    hi_i = starts[ok] + counts[ok] // 2
    med = 0.5 * (yv[lo_i] + yv[hi_i])
    rho_c = (np.bincount(cell, weights=rho_v, minlength=ncell)[ok] / counts[ok])
    n_c = counts[ok].astype(np.float64)
    wt = np.sqrt(np.minimum(n_c, 600.0) / 600.0) + 0.1
    r2, r4, r6 = rho_c ** 2, rho_c ** 4, rho_c ** 6
    g = np.linspace(0.0, 1.0, 41)
    g2, g3, g4, g5, g6 = g ** 2, g ** 3, g ** 4, g ** 5, g ** 6
    sig, lam, kpen = 0.15, 1.0 / 0.08, 25.0

    def resid(p):
        c0, a2, a4, a6 = p
        V = np.maximum(1.0 + a2 * r2 + a4 * r4 + a6 * r6, _VIG_FLOOR)
        res = wt * (med - c0 - np.log2(V)) / sig
        dV = 2.0 * a2 * g + 4.0 * a4 * g3 + 6.0 * a6 * g5
        Vg = 1.0 + a2 * g2 + a4 * g4 + a6 * g6
        return np.concatenate([res, [lam * a4, lam * a6], kpen * np.maximum(dV, 0.0),
                               kpen * np.maximum(_VIG_FLOOR - Vg, 0.0)])

    p0 = np.array([float(np.median(med)), 0.0, 0.0, 0.0])
    sol = least_squares(resid, p0, bounds=([-30.0, -0.97, -1.0, -1.0], [30.0, 0.0, 1.0, 1.0]), loss="soft_l1",
                        f_scale=1.0, method="trf")
    c0, a2, a4, a6 = sol.x
    V = np.clip(1.0 + a2 * _VIG_GRID ** 2 + a4 * _VIG_GRID ** 4 + a6 * _VIG_GRID ** 6, _VIG_FLOOR, 1.0)
    V = np.minimum.accumulate(V)
    V[0] = 1.0
    # 투영된 곡선에 대한 잔차(상수는 다시 맞춘다)
    Vc = np.interp(np.minimum(rho_c, 1.0), _VIG_GRID, V)
    base = med - np.log2(Vc)
    c0f = float(np.median(base))
    res = med - c0f - np.log2(Vc)
    rms = float(np.sqrt(np.sum(wt * res * res) / np.sum(wt)))
    rho_obs = float(np.max(rho_c))
    if rho_obs < 0.85:
        warns.append(W_EXTRAP)
    reliable = min(h, w) >= _SMALL_SIDE and int(ok.sum()) >= 40 and rho_obs >= 0.5 and rms <= 0.6
    return _vig_from_curve(V, h, w, center_t, rn, rms, reliable, warns)


def estimate_vignette(image_lin, *, center=None, r_norm=None, opts=None) -> VignetteFit:
    o = _opts(opts)
    lin = _check_image(image_lin)
    return _vignette_core(lin, _xy("center", center), _norm_r(r_norm), o)


# ---------------------------------------------------------------------------------------------------------------
# 그레인
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class GrainFit:
    strength: float
    sigma_coef: float
    floor_sigma: float
    floor_frac: float
    nonphysical: bool
    radius_px: float
    channel_corr: float
    n_flat_blocks: int
    levels: tuple
    sigma_by_level: tuple
    psd_nu: tuple
    psd: tuple
    reliable: bool
    warnings: tuple = ()

    def sigma_at(self, u):
        a = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
        return np.sqrt(self.sigma_coef ** 2 * a * (1.0 - a) + self.floor_sigma ** 2)

    def psd_model(self, nu, u=0.5):
        """σ(u)² π r² exp(-π² r² ν²): 평면 적분 = σ(u)²."""
        n = np.asarray(nu, np.float64)
        s2 = np.asarray(self.sigma_at(u)) ** 2
        r2 = self.radius_px ** 2
        return s2 * math.pi * r2 * np.exp(-(math.pi ** 2) * r2 * n * n)


@lru_cache(maxsize=8)
def _resid_tables(B):
    """가우시안 상관 exp(-d²/r²) 잡음에 B×B 평면 제거를 한 잔차의 기대 (분산 유지율, 지연 1·2 자기상관비)."""
    rs = np.geomspace(0.25, 12.0, 64)
    idx = np.arange(B, dtype=np.float64) - (B - 1) / 2.0
    e0 = np.full((B, B), 1.0 / B)
    ex = np.broadcast_to(idx[None, :], (B, B)).copy()
    ey = np.broadcast_to(idx[:, None], (B, B)).copy()
    ex /= np.sqrt(np.sum(ex * ex))
    ey /= np.sqrt(np.sum(ey * ey))
    E = [e0, ex, ey]
    ii = np.arange(B)
    N = float(B * B)

    def pair_sum(u, v, d):
        return float(np.sum(u[:, :-d] * v[:, d:]) + np.sum(u[:-d, :] * v[d:, :]))

    ret = np.zeros(len(rs))
    rat = np.zeros((2, len(rs)))
    for k, r in enumerate(rs):
        kk = np.exp(-((ii[:, None] - ii[None, :]) ** 2) / (r * r))
        S = [kk @ e @ kk for e in E]
        G = np.array([[float(np.sum(E[a] * S[b])) for b in range(3)] for a in range(3)])
        trc = N - (G[0, 0] + G[1, 1] + G[2, 2])
        ret[k] = trc / N
        for li, d in enumerate((1, 2)):
            npairs = 2.0 * (B - d) * B
            t1 = npairs * math.exp(-d * d / (r * r))
            t2 = sum(pair_sum(E[a], S[a], d) + pair_sum(S[a], E[a], d) for a in range(3))
            t3 = sum(G[a, b] * pair_sum(E[a], E[b], d) for a in range(3) for b in range(3))
            rat[li, k] = ((t1 - t2 + t3) / npairs) / ret[k]
    return rs, ret, rat[0], rat[1]


@lru_cache(maxsize=8)
def _plane_basis(B):
    idx = np.arange(B, dtype=np.float64) - (B - 1) / 2.0
    X = np.stack([np.ones(B * B), np.tile(idx, B), np.repeat(idx, B)], axis=1)
    return X, np.linalg.pinv(X)


def _grain_fallback(warns, reliable=False, n_flat=0):
    nu = np.arange(9) / 16.0
    return GrainFit(
        strength=0.0, sigma_coef=0.0, floor_sigma=0.0, floor_frac=0.0, nonphysical=False, radius_px=0.75,
        channel_corr=0.0, n_flat_blocks=int(n_flat), levels=(), sigma_by_level=(), psd_nu=_tup(nu),
        psd=_tup(np.zeros_like(nu)), reliable=reliable, warnings=_warn(*warns))


def _grain_core(lin, jpeg_quality, opts) -> GrainFit:
    h, w = lin.shape[:2]
    warns = [_small_warn(h, w)]
    if jpeg_quality is not None and jpeg_quality < 85.0:
        warns.append(W_JPEG)
    lossy = W_JPEG in warns
    B = opts.block
    nby, nbx = h // B, w // B
    if nby < 1 or nbx < 1:
        warns.append(W_FEW_FLAT)
        return _grain_fallback(warns)
    d = np.asarray(_color.linear_to_display(lin.astype(np.float32)), np.float64)
    oy, ox = (h - nby * B) // 2, (w - nbx * B) // 2
    N = B * B
    blocks = d[oy:oy + nby * B, ox:ox + nbx * B].reshape(nby, B, nbx, B, 3).transpose(4, 0, 2, 1, 3).reshape(3, nby * nbx, N)
    X, Xp = _plane_basis(B)
    res = blocks - (blocks @ Xp.T) @ X.T                     # (3, nb, N)
    s2 = np.mean(res * res, axis=2)                           # (3, nb)
    u = blocks.mean(axis=2)
    bmin, bmax = blocks.min(axis=2), blocks.max(axis=2)
    valid = (bmin > 0.0) & (bmax < 1.0) & (u > 0.04) & (u < 0.96)
    cc, bb = np.nonzero(valid)
    if cc.size < 6:
        warns.append(W_FEW_FLAT)
        return _grain_fallback(warns)
    uv, sv = u[cc, bb], np.sqrt(s2[cc, bb])
    order = np.argsort(uv, kind="stable")
    n_bins = int(np.clip(cc.size // 30, 1, 8))
    sel_mask = np.zeros(cc.size, bool)
    for chunk in np.array_split(order, n_bins):
        if chunk.size < 3:
            continue
        s = sv[chunk]
        thr1 = 1.3 * np.percentile(s, 25.0)
        s1 = s <= thr1
        ref = float(np.median(s[s1]))
        sel = s <= 1.4 * ref
        sel_mask[chunk[sel]] = True
    sel_idx = np.nonzero(sel_mask)[0]
    if sel_idx.size < 3:
        warns.append(W_FEW_FLAT)
        return _grain_fallback(warns)
    sc, sb, su = cc[sel_idx], bb[sel_idx], uv[sel_idx]
    n_flat = int(np.unique(sb).size)
    sres = res[sc, sb]                                         # (m, N)
    norm = np.sqrt(su * (1.0 - su))
    nres = sres / norm[:, None]
    # --- 반경: 잔차 자기상관 지연 1·2 를 평면 제거까지 반영한 기대값에 맞춘다
    ms = float(np.mean(nres * nres))
    if ms < 1e-12:
        r_hat, ret_hat = 0.75, 1.0
    else:
        R = nres.reshape(-1, B, B)
        m1 = (np.sum(R[:, :, :-1] * R[:, :, 1:]) + np.sum(R[:, :-1, :] * R[:, 1:, :])) / (R.shape[0] * 2.0 * (B - 1) * B)
        m2 = (np.sum(R[:, :, :-2] * R[:, :, 2:]) + np.sum(R[:, :-2, :] * R[:, 2:, :])) / (R.shape[0] * 2.0 * (B - 2) * B)
        rs, ret, r1, r2 = _resid_tables(B)
        a1, a2 = m1 / ms, m2 / ms
        cost = (a1 - r1) ** 2 + 0.5 * (a2 - r2) ** 2
        k = int(np.argmin(cost))
        r_hat, ret_hat = float(rs[k]), float(ret[k])
    # --- 레벨별 σ² 를 c² u(1-u) + n0² 에 NNLS
    n_lv = int(np.clip(sel_idx.size // 6, 1, 8))
    ord2 = np.argsort(su, kind="stable")
    lv, lv_s2, lv_uu, lv_n = [], [], [], []
    for chunk in np.array_split(ord2, n_lv):
        if chunk.size < 2:
            continue
        lv.append(float(np.mean(su[chunk])))
        lv_s2.append(float(np.mean(s2[sc[chunk], sb[chunk]])) / ret_hat)
        lv_uu.append(float(np.mean(su[chunk] * (1.0 - su[chunk]))))
        lv_n.append(chunk.size)
    lv, lv_s2, lv_uu, lv_n = map(np.asarray, (lv, lv_s2, lv_uu, lv_n))
    if len(lv) < 3:
        warns.append(W_FEW_LEVELS)
    wgt = np.sqrt(lv_n) / np.maximum(lv_s2, 1e-12)
    A = np.stack([lv_uu, np.ones_like(lv_uu)], axis=1) * wgt[:, None]
    (c2, n02), _ = nnls(A, lv_s2 * wgt)
    c, n0 = math.sqrt(max(c2, 0.0)), math.sqrt(max(n02, 0.0))
    um = float(np.median(su))
    denom = c2 * um * (1.0 - um) + n02
    floor_frac = float(n02 / denom) if denom > 0 else 0.0
    nonphysical = bool(floor_frac > 0.5 and n0 > 0.004)
    strength = float(min(1.0, max(0.0, c / (2.0 * GRAIN_SIGMA_MAX))))
    # --- 채널 간 상관(세 채널 모두 평탄한 위치)
    sel_pos = {}
    for ch in range(3):
        sel_pos[ch] = {int(b): i for b, i in zip(sb[sc == ch], np.nonzero(sc == ch)[0])}
    common = sorted(set(sel_pos[0]) & set(sel_pos[1]) & set(sel_pos[2]))
    corr = 0.0
    if len(common) >= 3:
        Rm = [np.stack([nres[sel_pos[ch][b]] for b in common]) for ch in range(3)]
        cs = []
        for a_, b_ in ((0, 1), (0, 2), (1, 2)):
            den = math.sqrt(float(np.sum(Rm[a_] ** 2) * np.sum(Rm[b_] ** 2)))
            cs.append(float(np.sum(Rm[a_] * Rm[b_]) / den) if den > 0 else 0.0)
        corr = float(np.mean(cs))
    # --- 실측 방사 PSD (u=0.5 환산, Hann 창)
    nu_grid = np.arange(9) / 16.0
    if B != 16:
        nu_grid = np.arange(B // 2 + 1) / float(B)
    sub = nres[np.linspace(0, len(nres) - 1, min(len(nres), 400)).astype(np.int64)] * 0.5
    win1 = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(B) + 0.5) / B)
    win = np.outer(win1, win1)
    F = np.fft.fft2(sub.reshape(-1, B, B) * win[None])
    P = (np.abs(F) ** 2) / np.sum(win * win)
    P = P.mean(axis=0)
    fr = np.fft.fftfreq(B)
    nu_r = np.hypot(fr[:, None], fr[None, :])
    kbin = np.rint(nu_r * B).astype(np.int64)
    psd = np.array([P[(kbin == k)].mean() if np.any(kbin == k) else 0.0 for k in range(len(nu_grid))])
    reliable = bool(n_flat >= 24 and min(h, w) >= _SMALL_SIDE and not lossy)
    if n_flat < 24:
        warns.append(W_FEW_FLAT)
    return GrainFit(
        strength=_f(strength), sigma_coef=_f(c), floor_sigma=_f(n0), floor_frac=_f(floor_frac), nonphysical=nonphysical,
        radius_px=_f(r_hat), channel_corr=_f(corr), n_flat_blocks=n_flat, levels=_tup(lv),
        sigma_by_level=_tup(np.sqrt(np.maximum(lv_s2, 0.0))), psd_nu=_tup(nu_grid), psd=_tup(psd), reliable=reliable,
        warnings=_warn(*warns))


def _check_jpeg(q):
    if q is None:
        return None
    if not _is_real(q) or not math.isfinite(float(q)) or not (0.0 < float(q) <= 100.0):
        raise ValueError("jpeg_quality: None 또는 (0,100] 실수여야 한다")
    return float(q)


def estimate_grain(image_lin, *, jpeg_quality=None, opts=None) -> GrainFit:
    o = _opts(opts)
    lin = _check_image(image_lin)
    return _grain_core(lin, _check_jpeg(jpeg_quality), o)


# ---------------------------------------------------------------------------------------------------------------
# 횡색수차
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class CAFit:
    eps_r: float
    eps_b: float
    center: tuple
    r_corner: float
    shift_corner_px: tuple
    n_samples: int
    rms_resid_px: float
    reliable: bool
    warnings: tuple = ()

    def shift_px(self, r_px, channel):
        if channel == "r":
            e = self.eps_r
        elif channel == "b":
            e = self.eps_b
        else:
            raise ValueError("channel: 'r' 또는 'b'")
        out = e * np.asarray(r_px, np.float64)
        return out[()] if out.ndim == 0 else out


def _ca_channel(S, ch, ux, uy, rr, valid, W):
    """채널 ch 를 녹색에 대해 R≈aG+b+β∂_rG 로 국소 LS 해 -> (s=-β/a, 정보 가중, a)."""
    G, R = S[..., 1], S[..., ch]
    gy, gx = np.gradient(G)
    D = gx * ux + gy * uy
    bx = lambda z: ndi.uniform_filter(z, size=W, mode="nearest")  # noqa: E731
    mG, mR, mD = bx(G), bx(R), bx(D)
    cGG = bx(G * G) - mG * mG
    cGD = bx(G * D) - mG * mD
    cDD = bx(D * D) - mD * mD
    cRG = bx(R * G) - mR * mG
    cRD = bx(R * D) - mR * mD
    det = cGG * cDD - cGD * cGD
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (cRG * cDD - cRD * cGD) / det
        beta = (cGG * cRD - cGD * cRG) / det
        s = -beta / a
        vdg = cDD - cGD * cGD / cGG
    return s, vdg, a, cGG, valid


def _ca_core(lin_full, center, opts) -> CAFit:
    h, w = lin_full.shape[:2]
    cx, cy = center if center is not None else _default_center(h, w)
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    r_corner = float(max(math.hypot(x - cx, y - cy) for x, y in corners))
    warns = [_small_warn(h, w)]
    lin, f = _downscale(lin_full, opts)
    hs, ws = lin.shape[:2]
    G = lin[..., 1]
    scale = max(float(np.mean(G)), 1e-6)
    no_color = False
    for ch in (0, 2):
        if float(np.sqrt(np.mean((lin[..., ch] - G) ** 2))) < 1e-3 * scale:
            no_color = True
    base_kw = dict(center=(float(cx), float(cy)), r_corner=r_corner)
    if no_color:
        warns.append(W_CA_NO_COLOR)
        return CAFit(eps_r=0.0, eps_b=0.0, shift_corner_px=(0.0, 0.0), n_samples=0, rms_resid_px=_NAN, reliable=False,
                     warnings=_warn(*warns), **base_kw)
    S = np.stack([ndi.gaussian_filter(lin[..., c], 1.0, mode="nearest") for c in range(3)], axis=-1)
    xs, ys = _coords(hs, ws, f)
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    rr = np.sqrt(dx * dx + dy * dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux, uy = dx / rr, dy / rr
    W = 9
    clip = (lin.max(axis=2) >= 0.97).astype(np.float64)
    bad = ndi.maximum_filter(clip, size=W + 4) > 0
    Ysat = _luma(lin) >= opts.sat_level
    if Ysat.any():
        near = ndi.distance_transform_edt(~Ysat) <= (0.06 * ws)
        bad = bad | near
    inside = np.zeros((hs, ws), bool)
    m = W // 2 + 1
    inside[m:hs - m, m:ws - m] = True
    stride = 3
    sub = np.zeros((hs, ws), bool)
    sub[::stride, ::stride] = True
    far = rr >= 0.1 * r_corner
    out = []
    n_used = []
    for ch in (0, 2):
        s, vdg, a, cgg, _ = _ca_channel(S, ch, ux, uy, rr, None, W)
        cand = inside & sub & far & ~bad & np.isfinite(s) & np.isfinite(vdg) & (a > 0.15) & (a < 6.0) & (cgg > 1e-8)
        if cand.sum() < 20:
            out.append((0.0, _NAN, 0))
            n_used.append(0)
            continue
        vv = vdg[cand]
        thr = max(4.0 * float(np.median(vv)), 1e-9)
        sel = cand & (vdg >= thr)
        if sel.sum() < 20:
            sel = cand
        r_s = rr[sel] / f
        s_s = s[sel]
        wt = np.minimum(vdg[sel] * a[sel] ** 2, np.percentile(vdg[sel] * a[sel] ** 2, 95.0))
        wt = wt / np.max(wt)
        keep = np.abs(s_s) < 8.0
        r_s, s_s, wt = r_s[keep], s_s[keep], wt[keep]
        if r_s.size < 20:
            out.append((0.0, _NAN, 0))
            n_used.append(0)
            continue
        eps = 0.0
        rw = np.ones_like(wt)
        for _ in range(12):
            ww = wt * rw
            eps = float(np.sum(ww * r_s * s_s) / max(np.sum(ww * r_s * r_s), 1e-12))
            res = s_s - eps * r_s
            mad = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-6
            k = 1.5 * mad
            ar = np.abs(res)
            rw = np.where(ar <= k, 1.0, k / np.maximum(ar, 1e-12))
        res = s_s - eps * r_s
        top = wt >= np.median(wt)
        rms = float(np.sqrt(np.sum(rw[top] * res[top] ** 2) / max(np.sum(rw[top]), 1e-12))) * f
        out.append((eps, rms, int(r_s.size)))
        n_used.append(int(r_s.size))
    # eps 는 화소 단위 비율(작은 화소 반경 / 작은 화소 이동 -> 비율은 배율 불변)
    eps_r, eps_b = out[0][0], out[1][0]
    n_samples = int(min(n_used)) if n_used else 0
    rms_list = [o[1] for o in out if math.isfinite(o[1])]
    rms = max(rms_list) if rms_list else _NAN
    reliable = bool(n_samples >= 200 and min(h, w) >= _SMALL_SIDE)
    return CAFit(eps_r=_f(eps_r), eps_b=_f(eps_b), shift_corner_px=(_f(eps_r * r_corner), _f(eps_b * r_corner)),
                 n_samples=n_samples, rms_resid_px=_f(rms), reliable=reliable, warnings=_warn(*warns), **base_kw)


def estimate_ca(image_lin, *, center=None, opts=None) -> CAFit:
    o = _opts(opts)
    lin = _check_image(image_lin)
    return _ca_core(lin, _xy("center", center), o)


# ---------------------------------------------------------------------------------------------------------------
# 할레이션
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class HaloBlob:
    x: float
    y: float
    area_px: float


@dataclass(frozen=True, kw_only=True)
class HalationFit:
    blobs: tuple
    n_blobs: int
    halo_gain: float
    sigma_px: float
    sigma_frac: float
    tint: tuple
    detected: bool
    amount: float
    amount_min: float
    reliable: bool
    warnings: tuple = ()

    def source_luminance(self, amount):
        if not _is_real(amount) or not math.isfinite(float(amount)) or not (0.0 < float(amount) <= 1.0):
            raise ValueError("amount: (0,1] 실수여야 한다")
        return float(HALATION_THRESHOLD + self.halo_gain / float(amount))


def _find_blobs(lin, f, opts):
    """포화 연결성분(Y>=sat_level, 면적>=min_blob_px). 반환 (blobs [(x,y,area)...] 면적 큰 순, 광원 마스크)."""
    Y = _luma(lin)
    mask = Y >= opts.sat_level
    if not mask.any():
        return [], np.zeros_like(mask)
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    min_area = max(1, int(math.ceil(opts.min_blob_px / float(f * f))))
    keep = np.nonzero(areas >= min_area)[0] + 1
    if keep.size == 0:
        return [], np.zeros_like(mask)
    cms = ndi.center_of_mass(mask, lab, keep)
    blobs = []
    for lbl, (cy, cx) in zip(keep, cms):
        blobs.append((float(cx * f + (f - 1) / 2.0), float(cy * f + (f - 1) / 2.0), float(areas[lbl - 1]) * f * f))
    blobs.sort(key=lambda b: (-b[2], b[1], b[0]))
    return blobs, np.isin(lab, keep)


def _halation_core(lin_full, opts) -> HalationFit:
    h, w = lin_full.shape[:2]
    warns = [_small_warn(h, w)]
    lin, f = _downscale(lin_full, opts)
    hs, ws = lin.shape[:2]
    blobs, M = _find_blobs(lin, f, opts)
    prior_sigma_px = HALATION_SIGMA_FRAC * w
    prior_tint = tuple(float(t) for t in HALATION_TINT)

    def result(P, sigma_px, tint, detected, reliable):
        amount_min = P / (HALO_SOURCE_MAX - HALATION_THRESHOLD) if detected else 0.0
        amount = min(1.0, P / (HALO_ASSUMED_SOURCE_Y - HALATION_THRESHOLD)) if detected else 0.0
        return HalationFit(
            blobs=tuple(HaloBlob(x=_f(x), y=_f(y), area_px=_f(a)) for x, y, a in blobs), n_blobs=len(blobs),
            halo_gain=_f(P), sigma_px=_f(sigma_px), sigma_frac=_f(sigma_px / w), tint=_tup(tint), detected=bool(detected),
            amount=_f(amount), amount_min=_f(amount_min), reliable=bool(reliable), warnings=_warn(*warns))

    if not blobs:
        warns.append(W_NO_SAT)
        return result(0.0, prior_sigma_px, prior_tint, False, False)
    dist = ndi.distance_transform_edt(~M)
    excl = dist <= 2.0
    region = (dist <= 0.1 * ws) & ~excl
    chan_ok = [region & (lin[..., c] < 0.985) for c in range(3)]
    if min(int(m.sum()) for m in chan_ok) < 200:
        return result(0.0, prior_sigma_px, prior_tint, False, False)
    xn = (np.arange(ws, dtype=np.float64) / ws - 0.5)[None, :] * np.ones((hs, 1))
    yn = (np.arange(hs, dtype=np.float64) / ws - 0.5)[:, None] * np.ones((1, ws))
    Mf = M.astype(np.float64)
    fr_grid = np.geomspace(0.006, 0.06, 28)
    # 채널별 Huber 척도: 광원 항 없이 평면만 맞춘 잔차의 MAD
    deltas = []
    for c in range(3):
        m_ = chan_ok[c]
        A0 = np.stack([np.ones(m_.sum()), xn[m_], yn[m_]], axis=1)
        y_ = lin[..., c][m_]
        co, *_ = np.linalg.lstsq(A0, y_, rcond=None)
        r0 = y_ - A0 @ co
        deltas.append(max(1.5 * 1.4826 * float(np.median(np.abs(r0 - np.median(r0)))), 0.004))
    costs, sols = [], []
    for fr in fr_grid:
        K = ndi.gaussian_filter(Mf, fr * ws, mode="constant", truncate=5.0)
        cost_t, ps = 0.0, []
        for c in range(3):
            m_ = chan_ok[c]
            A = np.stack([np.ones(m_.sum()), xn[m_], yn[m_], K[m_]], axis=1)
            coef, cost = _huber_fit(A, lin[..., c][m_], deltas[c], iters=6)
            cost_t += cost
            ps.append(float(coef[3]))
        costs.append(cost_t)
        sols.append(ps)
    costs = np.asarray(costs)
    k = int(np.argmin(costs))
    lf = np.log(fr_grid)
    if 0 < k < len(fr_grid) - 1:
        y0, y1, y2 = costs[k - 1], costs[k], costs[k + 1]
        den = y0 - 2.0 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 1e-30 else 0.0
        off = float(np.clip(off, -1.0, 1.0))
    else:
        off = 0.0
    fr_best = float(np.exp(lf[k] + off * (lf[1] - lf[0])))
    # 보간한 σ 에서 다시 맞춘다
    K = ndi.gaussian_filter(Mf, fr_best * ws, mode="constant", truncate=5.0)
    ps = []
    for c in range(3):
        m_ = chan_ok[c]
        A = np.stack([np.ones(m_.sum()), xn[m_], yn[m_], K[m_]], axis=1)
        coef, _ = _huber_fit(A, lin[..., c][m_], deltas[c], iters=8)
        ps.append(float(coef[3]))
    P = max(ps[0], 0.0)
    if ps[0] >= 0.01:
        tint = (1.0, float(np.clip(ps[1] / ps[0], 0.0, 3.0)), float(np.clip(ps[2] / ps[0], 0.0, 3.0)))
    else:
        tint = prior_tint
    red = tint[1] <= 0.8 and tint[2] <= 0.8
    detected = bool(P >= 0.03 and red)
    if P >= 0.03 and not red:
        warns.append(W_NEUTRAL)
    sigma_px = fr_best * w
    return result(P, sigma_px, tint, detected, detected)


def estimate_halation(image_lin, *, opts=None) -> HalationFit:
    o = _opts(opts)
    lin = _check_image(image_lin)
    return _halation_core(lin, o)


# ---------------------------------------------------------------------------------------------------------------
# 고스트
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class GhostSpot:
    x: float
    y: float
    radius_px: float
    z: float
    k: float


@dataclass(frozen=True, kw_only=True)
class GhostFit:
    source: tuple | None
    center: tuple
    ghosts: tuple
    warnings: tuple = ()

    def mask(self, h, w) -> np.ndarray:
        if not _is_int(h) or not _is_int(w) or h < 1 or w < 1:
            raise ValueError("h, w: 1 이상의 정수여야 한다")
        out = np.zeros((int(h), int(w)), bool)
        for g in self.ghosts:
            r = 1.25 * g.radius_px
            x0, x1 = max(0, int(math.floor(g.x - r))), min(w, int(math.ceil(g.x + r)) + 1)
            y0, y1 = max(0, int(math.floor(g.y - r))), min(h, int(math.ceil(g.y + r)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            out[y0:y1, x0:x1] |= (xx - g.x) ** 2 + (yy - g.y) ** 2 <= r * r
        return out


_GHOST_BAND_PX = 6.0
_GHOST_Z = 6.0
_DOG_RATIO = 1.6
_DOG_RADIUS_FACTOR = 1.757            # 원판 반지름 R 의 DoG(σ, 1.6σ) 최대 응답 척도: R = 1.757 σ


def _ghost_core(lin_full, center, opts) -> GhostFit:
    h, w = lin_full.shape[:2]
    cx, cy = center if center is not None else _default_center(h, w)
    warns = [_small_warn(h, w)]
    lin, f = _downscale(lin_full, opts)
    hs, ws = lin.shape[:2]
    blobs, _ = _find_blobs(lin, f, opts)
    ctr = (float(cx), float(cy))
    if not blobs:
        warns.append(W_NO_SAT)
        return GhostFit(source=None, center=ctr, ghosts=(), warnings=_warn(*warns))
    sx, sy, sarea = blobs[0]
    source = (float(sx), float(sy))
    vx, vy = sx - cx, sy - cy
    vlen = math.hypot(vx, vy)
    if vlen < 1e-6 or opts.ghost_max == 0:
        return GhostFit(source=source, center=ctr, ghosts=(), warnings=_warn(*warns))
    Y = np.minimum(_luma(lin), 1.0)
    xs, ys = _coords(hs, ws, f)
    X = np.broadcast_to(xs[None, :], (hs, ws))
    Yg = np.broadcast_to(ys[:, None], (hs, ws))
    perp = np.abs((X - cx) * vy - (Yg - cy) * vx) / vlen
    band = perp <= _GHOST_BAND_PX
    dsrc = np.hypot(X - sx, Yg - sy)
    r_src = math.sqrt(sarea / math.pi)
    sig_s = np.geomspace(2.0, 16.0, 10)
    dogs = []
    for s in sig_s:
        sg = s / f
        dogs.append(ndi.gaussian_filter(Y, sg, mode="nearest") - ndi.gaussian_filter(Y, _DOG_RATIO * sg, mode="nearest"))
    cands = []
    for j, s in enumerate(sig_s):
        D = dogs[j]
        sg = s / f
        med = float(np.median(D))
        mad = 1.4826 * float(np.median(np.abs(D - med)))
        mad = max(mad, 3e-4)
        z = (D - med) / mad
        size = 2 * int(math.ceil(sg)) + 1
        ismax = (D == ndi.maximum_filter(D, size=size, mode="nearest")) & (D > 0)
        if j > 0:
            ismax &= D >= dogs[j - 1]
        if j < len(sig_s) - 1:
            ismax &= D >= dogs[j + 1]
        far = dsrc > (2.0 * r_src + 1.0 * s + 3.0)
        sel = ismax & band & far & (z >= _GHOST_Z)
        for iy, ix in zip(*np.nonzero(sel)):
            # 척도 방향 포물선 보간(로그 σ)
            sstar = s
            if 0 < j < len(sig_s) - 1:
                y0, y1, y2 = dogs[j - 1][iy, ix], D[iy, ix], dogs[j + 1][iy, ix]
                den = y0 - 2.0 * y1 + y2
                if den < -1e-30:
                    off = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
                    sstar = float(s * (sig_s[1] / sig_s[0]) ** off)
            cands.append((float(z[iy, ix]), float(ix * f + (f - 1) / 2.0), float(iy * f + (f - 1) / 2.0),
                          _DOG_RADIUS_FACTOR * sstar))
    cands.sort(key=lambda c: (-c[0], c[2], c[1]))
    acc = []
    for z, x, y, r in cands:
        if all(math.hypot(x - a[1], y - a[2]) >= 1.2 * max(r, a[3]) for a in acc):
            acc.append((z, x, y, r))
    acc = acc[:opts.ghost_max]
    ghosts = []
    for z, x, y, r in acc:
        k = ((x - cx) * vx + (y - cy) * vy) / (vlen * vlen)
        ghosts.append(GhostSpot(x=_f(x), y=_f(y), radius_px=_f(r), z=_f(z), k=_f(k)))
    return GhostFit(source=source, center=ctr, ghosts=tuple(ghosts), warnings=_warn(*warns))


def estimate_ghost(image_lin, *, center=None, opts=None) -> GhostFit:
    o = _opts(opts)
    lin = _check_image(image_lin)
    return _ghost_core(lin, _xy("center", center), o)


# ---------------------------------------------------------------------------------------------------------------
# 합 / 항등 / 메타
# ---------------------------------------------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        x = float(v)
        return x if math.isfinite(x) else None
    return v


@dataclass(frozen=True, kw_only=True)
class ScreenFx:
    size: tuple
    vignette: VignetteFit
    grain: GrainFit
    ca: CAFit
    halation: HalationFit
    ghost: GhostFit
    warnings: tuple = ()

    @property
    def reliable(self) -> bool:
        return bool(self.vignette.reliable and self.grain.reliable)

    def theta_fields(self) -> dict:
        def u(v):
            return float(min(1.0, max(0.0, v))) if math.isfinite(v) else 0.0
        return {"vignette": u(self.vignette.strength), "grain": u(self.grain.strength), "halation": u(self.halation.amount)}

    def to_meta(self) -> dict:
        v, g, c, hl, gh = self.vignette, self.grain, self.ca, self.halation, self.ghost
        meta = {
            "size": list(self.size),
            "vignette": {"strength": v.strength, "corner_ev": v.corner_ev, "coeffs": list(v.coeffs),
                         "center": list(v.center), "r_norm": v.r_norm, "rms_resid_ev": v.rms_resid_ev,
                         "reliable": v.reliable},
            "grain": {"strength": g.strength, "radius_px": g.radius_px, "floor_sigma": g.floor_sigma,
                      "sigma_coef": g.sigma_coef, "nonphysical": g.nonphysical, "channel_corr": g.channel_corr,
                      "n_flat_blocks": g.n_flat_blocks, "reliable": g.reliable},
            "ca": {"eps_r": c.eps_r, "eps_b": c.eps_b, "shift_corner_px": list(c.shift_corner_px),
                   "n_samples": c.n_samples, "rms_resid_px": c.rms_resid_px, "reliable": c.reliable},
            "halation": {"detected": hl.detected, "n_blobs": hl.n_blobs, "sigma_frac": hl.sigma_frac,
                         "tint": list(hl.tint), "halo_gain": hl.halo_gain, "amount": hl.amount,
                         "amount_min": hl.amount_min, "reliable": hl.reliable},
            "ghost": {"n": len(gh.ghosts), "source": None if gh.source is None else list(gh.source),
                      "spots": [{"x": s.x, "y": s.y, "radius_px": s.radius_px, "z": s.z, "k": s.k} for s in gh.ghosts]},
            "warnings": sorted(set(self.warnings)),
            "reliable": self.reliable,
        }
        return _clean(meta)


def estimate_screen_fx(image_lin, *, center=None, r_norm=None, jpeg_quality=None, opts=None) -> ScreenFx:
    o = _opts(opts)
    lin = _check_image(image_lin)
    c, rn, jq = _xy("center", center), _norm_r(r_norm), _check_jpeg(jpeg_quality)
    h, w = lin.shape[:2]
    vig = _vignette_core(lin, c, rn, o)
    gr = _grain_core(lin, jq, o)
    ca = _ca_core(lin, c, o)
    hal = _halation_core(lin, o)
    gh = _ghost_core(lin, c, o)
    warns = _warn(*(vig.warnings + gr.warnings + ca.warnings + hal.warnings + gh.warnings))
    return ScreenFx(size=(int(h), int(w)), vignette=vig, grain=gr, ca=ca, halation=hal, ghost=gh, warnings=warns)


def identity_screen_fx(h, w) -> ScreenFx:
    """효과 없음(강도 0, reliable=False). RD-21 폴백값."""
    if not _is_int(h) or not _is_int(w) or h < 1 or w < 1:
        raise ValueError("h, w: 1 이상의 정수여야 한다")
    h, w = int(h), int(w)
    ctr = _default_center(h, w)
    rn = max(_half_diag(h, w), 1e-9)
    vig = _vig_from_curve(np.ones_like(_VIG_GRID), h, w, ctr, rn, _NAN, False, ())
    nu = np.arange(9) / 16.0
    gr = _grain_fallback(())
    ca = CAFit(eps_r=0.0, eps_b=0.0, center=ctr, r_corner=rn, shift_corner_px=(0.0, 0.0), n_samples=0,
               rms_resid_px=_NAN, reliable=False, warnings=())
    hal = HalationFit(blobs=(), n_blobs=0, halo_gain=0.0, sigma_px=HALATION_SIGMA_FRAC * w,
                      sigma_frac=HALATION_SIGMA_FRAC, tint=tuple(float(t) for t in HALATION_TINT), detected=False,
                      amount=0.0, amount_min=0.0, reliable=False, warnings=())
    gh = GhostFit(source=None, center=ctr, ghosts=(), warnings=())
    del nu
    return ScreenFx(size=(h, w), vignette=vig, grain=gr, ca=ca, halation=hal, ghost=gh, warnings=())
