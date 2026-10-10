"""RD-05: R3 광학 블러 추정 v0 -- c(z) 강건 회귀 -> z_f, N_eq, 모션블러 방향장.

모델: u=1/z, c(u)=D|u-u_f|, D=f^2/N*(W/s)/(1-f*u_f)  (LNS-02 coc_px 와 같은 정의). 디컨볼브하지 않는다.
numpy/scipy 만 쓴다. torch 텐서는 duck typing(detach/cpu/numpy)으로 받으며 import 하지 않는다.
import 부작용·전역 RNG 변경 없음. 결정론.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls

__all__ = [
    "U_INF", "FF_SENSOR_MM", "HYPERFOCAL_COC_MM", "BlurEstError", "BlurOpts", "DefocusFit", "MotionField",
    "BlurRead", "estimate_defocus", "estimate_motion_field", "motion_depth_consistency", "read_blur",
]

U_INF = 1e-3
FF_SENSOR_MM = 36.0
HYPERFOCAL_COC_MM = 0.03
_SIGMA_CAP = 10.0


class BlurEstError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# 옵션
# ---------------------------------------------------------------------------------------------------------------
def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _pos(name, x):
    if not _is_real(x) or not math.isfinite(float(x)) or float(x) <= 0:
        raise BlurEstError(f"{name}={x!r}: 유한한 양수여야 한다")
    return float(x)


def _frac(name, x, lo_open=True):
    if not _is_real(x) or not math.isfinite(float(x)):
        raise BlurEstError(f"{name}={x!r}: 유한한 실수여야 한다")
    v = float(x)
    if not ((0.0 < v if lo_open else 0.0 <= v) and v <= 1.0):
        raise BlurEstError(f"{name}={x!r}: 범위를 벗어났다")
    return v


@dataclass(frozen=True, kw_only=True)
class BlurOpts:
    min_points: int = 50
    min_log_depth_span: float = 0.1
    pan_focus_px: float = 1.0
    default_n_eq: float = 11.0
    inlier_abs_px: float = 1.0
    inlier_rel: float = 0.25
    min_inlier_frac: float = 0.7
    sigma_s_max: float = 0.35
    n_range: tuple = (1.0, 32.0)
    max_points: int = 50_000
    tile_px: int = 64
    aniso_min: float = 0.4
    min_length_px: float = 2.0
    tau_motion: float = 0.3
    min_motion_tiles: int = 4

    def __post_init__(self):
        if not _is_int(self.min_points) or self.min_points < 2:
            raise BlurEstError(f"min_points={self.min_points!r}: 2 이상의 정수여야 한다")
        _pos("min_log_depth_span", self.min_log_depth_span)
        _pos("pan_focus_px", self.pan_focus_px)
        _pos("default_n_eq", self.default_n_eq)
        _pos("inlier_abs_px", self.inlier_abs_px)
        _pos("inlier_rel", self.inlier_rel)
        _frac("min_inlier_frac", self.min_inlier_frac)
        _pos("sigma_s_max", self.sigma_s_max)
        nr = self.n_range
        if not isinstance(nr, (tuple, list)) or len(nr) != 2:
            raise BlurEstError("n_range 는 (lo, hi) 여야 한다")
        lo, hi = _pos("n_range[0]", nr[0]), _pos("n_range[1]", nr[1])
        if lo > hi:
            raise BlurEstError("n_range 가 역순이다")
        object.__setattr__(self, "n_range", (lo, hi))
        if not _is_int(self.max_points) or self.max_points < 2:
            raise BlurEstError(f"max_points={self.max_points!r}: 2 이상의 정수여야 한다")
        if not _is_int(self.tile_px) or self.tile_px < 16:
            raise BlurEstError(f"tile_px={self.tile_px!r}: 16 이상의 정수여야 한다")
        _frac("aniso_min", self.aniso_min, lo_open=False)
        _pos("min_length_px", self.min_length_px)
        _pos("tau_motion", self.tau_motion)
        if not _is_int(self.min_motion_tiles) or self.min_motion_tiles < 1:
            raise BlurEstError(f"min_motion_tiles={self.min_motion_tiles!r}: 1 이상의 정수여야 한다")


# ---------------------------------------------------------------------------------------------------------------
# 입력 변환·검증
# ---------------------------------------------------------------------------------------------------------------
def _to_numpy(x, name) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    try:
        a = np.array(x, dtype=np.float64)          # 복사: 입력 불변
    except (TypeError, ValueError) as e:
        raise BlurEstError(f"{name}: 숫자 배열로 바꿀 수 없다") from e
    return a


def _to_2d(x, name) -> np.ndarray:
    a = _to_numpy(x, name)
    if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 1:
        a = a[0, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.size == 0:
        raise BlurEstError(f"{name}: 2D(또는 [1,1,h,w]) 배열이어야 한다 (shape={a.shape})")
    return a


def _check_inputs(coc_px, depth_m, weight, f_px, width_px, sigma_s):
    c = _to_2d(coc_px, "coc_px")
    z = _to_2d(depth_m, "depth_m")
    if c.shape != z.shape:
        raise BlurEstError(f"coc_px{c.shape} 와 depth_m{z.shape} 의 모양이 다르다")
    if not np.all(np.isfinite(c)) or np.any(c < 0):
        raise BlurEstError("coc_px 는 유한한 비음수여야 한다")
    if np.any(np.isnan(z)) or np.any(z <= 0):
        raise BlurEstError("depth_m 은 양수(+inf 허용)여야 한다")
    if weight is None:
        w = np.ones_like(c)
    else:
        w = _to_2d(weight, "weight")
        if w.shape != c.shape:
            raise BlurEstError("weight 의 모양이 다르다")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise BlurEstError("weight 는 유한한 비음수여야 한다")
    f_px = _pos("f_px", f_px)
    if not _is_int(width_px) or int(width_px) < 1:
        raise BlurEstError(f"width_px={width_px!r}: 1 이상의 정수여야 한다")
    if not _is_real(sigma_s) or not math.isfinite(float(sigma_s)) or float(sigma_s) < 0:
        raise BlurEstError(f"sigma_s={sigma_s!r}: 유한한 비음수여야 한다")
    return c, z, w, f_px, int(width_px), float(sigma_s)


# ---------------------------------------------------------------------------------------------------------------
# 결과 직렬화 도우미
# ---------------------------------------------------------------------------------------------------------------
def _jf(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _jarr(a):
    a = np.asarray(a)
    if a.dtype == bool:
        return a.tolist()
    return [[_jf(v) for v in row] for row in a.astype(np.float64)]


# ---------------------------------------------------------------------------------------------------------------
# DefocusFit
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class DefocusFit:
    status: str
    z_f_m: float | None
    u_f: float | None
    d_px_m: float
    n_eq: float
    f_ff_mm: float
    sigma_log_zf: float | None
    sigma_log_n: float | None
    rel_resid: float | None
    inlier_frac: float
    n_used: int
    reliable: bool
    warnings: tuple

    def coc_pred(self, depth_m, signed=False) -> np.ndarray:
        z = np.asarray(_to_numpy(depth_m, "depth_m"), dtype=np.float64)
        if self.status != "fit":
            return np.zeros_like(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = 1.0 / z
        s = self.d_px_m * (self.u_f - u)          # 뒤(z>z_f) 양수, 앞 음수 (LNS-02 부호)
        return s if signed else np.abs(s)

    def residual(self, coc_px, depth_m) -> np.ndarray:
        c = np.asarray(_to_numpy(coc_px, "coc_px"), dtype=np.float64)
        return c - self.coc_pred(depth_m)

    def theta_fields(self) -> dict:
        f_mm = self.f_ff_mm
        n = min(max(self.n_eq, 0.7), 64.0)
        if self.status == "fit":
            focus = float(self.z_f_m)
        else:
            focus = (f_mm * f_mm / (self.n_eq * HYPERFOCAL_COC_MM) + f_mm) / 1000.0
        return {"focal_mm": float(f_mm), "sensor_width_mm": FF_SENSOR_MM, "f_stop": float(n), "focus_m": focus}

    def to_meta(self) -> dict:
        return {
            "status": self.status, "z_f_m": _jf(self.z_f_m), "n_eq": _jf(self.n_eq), "f_ff_mm": _jf(self.f_ff_mm),
            "d_px_m": _jf(self.d_px_m), "u_f": _jf(self.u_f), "sigma_log_zf": _jf(self.sigma_log_zf),
            "sigma_log_n": _jf(self.sigma_log_n), "rel_resid": _jf(self.rel_resid),
            "inlier_frac": _jf(self.inlier_frac), "n_used": int(self.n_used), "reliable": bool(self.reliable),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------------------------------------------
# 가중 백분위
# ---------------------------------------------------------------------------------------------------------------
def _wpercentile(x, w, q) -> float:
    o = np.argsort(x, kind="stable")
    xs, ws = x[o], w[o]
    cw = np.cumsum(ws)
    tot = cw[-1]
    if tot <= 0:
        return float(np.percentile(xs, q))
    pos = (cw - 0.5 * ws) / tot
    return float(np.interp(q / 100.0, pos, xs))


# ---------------------------------------------------------------------------------------------------------------
# c(u) = D|u-u_f| 강건 회귀 (Cauchy 손실, 재하강 IRLS, 전역 격자 탐색)
# ---------------------------------------------------------------------------------------------------------------
def _scales(c, opts):
    return 0.5 * opts.inlier_abs_px + 0.5 * opts.inlier_rel * c


def _loss(u, c, w, s, a, uf):
    r = c - a * np.abs(u - uf)
    return float(np.sum(w * np.log1p((r / s) ** 2)))


def _global_candidates(u, f_ff, nsub):
    umax = 1.0 / (2.0 * f_ff)
    idx = np.unique(np.linspace(0, len(u) - 1, min(len(u), nsub)).astype(np.int64))
    us = u[idx]
    grid = np.concatenate([[0.0], np.linspace(0.0, umax, 160), np.quantile(us, np.linspace(0.0, 1.0, 120))])
    grid = np.unique(np.clip(grid, 0.0, umax))
    return idx, grid


def _scan_d(us, cs, ws, ss, grid, iters=12):
    """각 후보 u_f 에 대해 D 를 IRLS 로 구하고 손실을 돌려준다. 모양 (G,)"""
    x = np.abs(us[None, :] - grid[:, None])
    wc = ws[None, :] * cs[None, :]
    num = np.sum(wc * x, axis=1)
    den = np.sum(ws[None, :] * x * x, axis=1)
    d = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    s2 = ss[None, :] ** 2
    for _ in range(iters):
        r = cs[None, :] - d[:, None] * x
        om = ws[None, :] / (s2 + r * r)
        num = np.sum(om * cs[None, :] * x, axis=1)
        den = np.sum(om * x * x, axis=1)
        d = np.where(den > 0, num / np.maximum(den, 1e-300), d)
    r = cs[None, :] - d[:, None] * x
    loss = np.sum(ws[None, :] * np.log1p(r * r / s2), axis=1)
    loss = np.where(d > 0, loss, np.inf)
    return d, loss


def _refine(u, c, w, s, a, uf, umax, iters=300):
    s2 = s * s
    best = (a, uf)
    for _ in range(iters):
        d = u - uf
        sg = np.where(d >= 0, 1.0, -1.0)
        x = np.abs(d)
        r = c - a * x
        om = w / (s2 + r * r)
        c1 = sg * u            # a 의 계수
        c2 = -sg               # b=a*u_f 의 계수
        m11 = np.sum(om * c1 * c1)
        m12 = np.sum(om * c1 * c2)
        m22 = np.sum(om * c2 * c2)
        v1 = np.sum(om * c1 * c)
        v2 = np.sum(om * c2 * c)
        det = m11 * m22 - m12 * m12
        ok = det > 1e-12 * max(m11 * m22, 1e-300)
        if ok:
            na = (v1 * m22 - v2 * m12) / det
            nb = (m11 * v2 - m12 * v1) / det
            if na <= 0:
                ok = False
        if ok:
            nuf = nb / na
        if not ok or nuf < 0 or nuf > umax:
            nuf = min(max(uf if not ok else nuf, 0.0), umax)
            x1 = np.abs(u - nuf)
            den = np.sum(om * x1 * x1)
            if den <= 0:
                break
            na = np.sum(om * x1 * c) / den
            if na <= 0:
                break
        conv = abs(na - a) <= 1e-12 * abs(a) and abs(nuf - uf) <= 1e-13 * max(abs(uf), 1e-3)
        a, uf = float(na), float(nuf)
        best = (a, uf)
        if conv:
            break
    return best


def _sandwich(u, c, w, a, uf, opts):
    """(a, b=a*u_f) 의 공분산(inlier 위 HC0 샌드위치). 실패하면 None."""
    d = u - uf
    sg = np.where(d >= 0, 1.0, -1.0)
    pred = a * np.abs(d)
    r = c - pred
    inl = np.abs(r) <= np.maximum(opts.inlier_abs_px, opts.inlier_rel * pred)
    if int(inl.sum()) < 3:
        return None
    X = np.stack([sg * u, -sg], axis=1)[inl]
    wi, ri = w[inl], r[inl]
    bread = (X * wi[:, None]).T @ X
    meat = (X * (wi * wi * ri * ri)[:, None]).T @ X
    if abs(np.linalg.det(bread)) <= 1e-14 * max(bread[0, 0] * bread[1, 1], 1e-300):
        return None
    bi = np.linalg.inv(bread)
    return bi @ meat @ bi


def _fit_curve(u, c, w, f_ff, opts):
    umax = 1.0 / (2.0 * f_ff)
    s = _scales(c, opts)
    idx, grid = _global_candidates(u, f_ff, 2500)
    d0, loss0 = _scan_d(u[idx], c[idx], w[idx], s[idx], grid)
    if not np.any(np.isfinite(loss0)):
        return None
    order = np.argsort(loss0, kind="stable")
    cands = []
    for k in order[:6]:
        if np.isfinite(loss0[k]):
            cands.append((float(d0[k]), float(grid[k])))
    best, best_loss = None, math.inf
    for a0, u0 in cands:
        a, uf = _refine(u, c, w, s, a0, u0, umax)
        ls = _loss(u, c, w, s, a, uf)
        if ls < best_loss:
            best, best_loss = (a, uf), ls
    return best, s


def estimate_defocus(coc_px, depth_m, *, f_px, width_px, weight=None, sigma_s=0.0, opts=None) -> DefocusFit:
    opts = BlurOpts() if opts is None else opts
    c2, z2, w2, f_px, width_px, sigma_s = _check_inputs(coc_px, depth_m, weight, f_px, width_px, sigma_s)
    f_ff = FF_SENSOR_MM / 1000.0 * f_px / width_px            # m
    f_ff_mm = 1000.0 * f_ff
    k = w2.ravel() > 0
    c = c2.ravel()[k]
    z = z2.ravel()[k]
    w = w2.ravel()[k]
    n = int(c.size)

    def nofit(status, inl=0.0):
        return DefocusFit(status=status, z_f_m=None, u_f=None, d_px_m=0.0, n_eq=float(opts.default_n_eq),
                          f_ff_mm=f_ff_mm, sigma_log_zf=None, sigma_log_n=None, rel_resid=None,
                          inlier_frac=float(inl), n_used=n, reliable=False, warnings=(status,))

    if n < opts.min_points:
        return nofit("insufficient")
    if n > opts.max_points:
        sel = np.unique(np.linspace(0, n - 1, opts.max_points).astype(np.int64))
        c, z, w = c[sel], z[sel], w[sel]
        n = int(c.size)
    if _wpercentile(c, w, 95.0) < opts.pan_focus_px:
        inl = float(np.sum(w[c <= opts.inlier_abs_px]) / np.sum(w))
        return nofit("pan_focus", inl)
    fin = np.isfinite(z)
    if int(fin.sum()) < 2:
        return nofit("insufficient")
    lz = np.log(z[fin])
    span = _wpercentile(lz, w[fin], 95.0) - _wpercentile(lz, w[fin], 5.0)
    if span < opts.min_log_depth_span:
        return nofit("insufficient")

    u = 1.0 / z
    res = _fit_curve(u, c, w, f_ff, opts)
    if res is None:
        return nofit("insufficient")
    (a, uf), _ = res
    umax = 1.0 / (2.0 * f_ff)
    uf = min(max(uf, 0.0), umax)
    pred = a * np.abs(u - uf)
    r = c - pred
    wsum = float(np.sum(w))
    inl_mask = np.abs(r) <= np.maximum(opts.inlier_abs_px, opts.inlier_rel * pred)
    inlier_frac = float(np.sum(w[inl_mask]) / wsum)
    den = float(np.sum(w * c * c))
    rel_resid = math.sqrt(float(np.sum(w * r * r)) / den) if den > 0 else 0.0

    q = 1.0 - f_ff * uf
    n_eq = f_ff * f_ff * (width_px / (FF_SENSOR_MM / 1000.0)) / (q * a)
    z_f = math.inf if uf <= U_INF else 1.0 / uf

    # 불확실도 (통계 + 스케일)
    cov = _sandwich(u, c, w, a, uf, opts)
    b = a * uf
    if cov is None or not np.all(np.isfinite(cov)):
        s_zf, s_n = _SIGMA_CAP, _SIGMA_CAP
    else:
        bb = max(b, U_INF * a)
        g_z = np.array([1.0 / a, -1.0 / bb])
        g_n = np.array([-f_ff * b / (q * a * a) - 1.0 / a, f_ff / (q * a)])
        s_zf = math.sqrt(max(float(g_z @ cov @ g_z), 0.0))
        s_n = math.sqrt(max(float(g_n @ cov @ g_n), 0.0))
        s_zf, s_n = min(s_zf, _SIGMA_CAP), min(s_n, _SIGMA_CAP)
    sig_zf = math.sqrt(s_zf ** 2 + sigma_s ** 2)
    sig_n = math.sqrt(s_n ** 2 + sigma_s ** 2)

    warns = set()
    if z_f == math.inf:
        warns.add("focus_at_infinity")
    if inlier_frac < opts.min_inlier_frac:
        warns.add("depth_inconsistent")
    if sigma_s > opts.sigma_s_max:
        warns.add("scale_unreliable")
    if not (opts.n_range[0] <= n_eq <= opts.n_range[1]):
        warns.add("n_out_of_range")
    reliable = "depth_inconsistent" not in warns and "scale_unreliable" not in warns
    return DefocusFit(status="fit", z_f_m=float(z_f), u_f=float(uf), d_px_m=float(a), n_eq=float(n_eq),
                      f_ff_mm=f_ff_mm, sigma_log_zf=float(sig_zf), sigma_log_n=float(sig_n),
                      rel_resid=float(rel_resid), inlier_frac=inlier_frac, n_used=n, reliable=bool(reliable),
                      warnings=tuple(sorted(warns)))


# ---------------------------------------------------------------------------------------------------------------
# 모션블러 방향장
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class MotionField:
    theta: np.ndarray
    length_px: np.ndarray
    anisotropy: np.ndarray
    is_motion: np.ndarray
    tile_px: int
    theta_global: float | None
    length_global: float | None
    frac_motion: float

    def to_meta(self) -> dict:
        return {
            "tile_px": int(self.tile_px), "grid": [int(self.theta.shape[0]), int(self.theta.shape[1])],
            "theta": _jarr(self.theta), "length_px": _jarr(self.length_px), "anisotropy": _jarr(self.anisotropy),
            "is_motion": _jarr(self.is_motion), "theta_global": _jf(self.theta_global),
            "length_global": _jf(self.length_global), "frac_motion": _jf(self.frac_motion),
        }


_SCHARR = np.array([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]) / 32.0
_FLAT_EPS = 1e-12
_PEAK_MIN = -0.15
_TAU_LO = 3.0


def _to_luma(image) -> np.ndarray:
    a = _to_numpy(image, "image")
    if a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] != 3:
        a = a[0]
    if a.ndim == 3 and a.shape[-1] == 3:
        a = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    if a.ndim != 2 or a.size == 0:
        raise BlurEstError(f"image: 2D 또는 HxWx3 여야 한다 (shape={a.shape})")
    if not np.all(np.isfinite(a)):
        raise BlurEstError("image 는 유한해야 한다")
    return a


def _wrap_pi(t: float) -> float:
    t = t % math.pi
    return 0.0 if t >= math.pi else t


def _streak_length(ix, iy, theta, t) -> float:
    """줄무늬 방향 미분의 자기상관에서 lag=L 의 음의 봉우리를 찾는다. 못 찾으면 nan."""
    ct, st = math.cos(theta), math.sin(theta)
    g = ct * ix + st * iy
    g = g - g.mean()
    p = 2 * t
    f = np.fft.rfft2(g, (p, p))
    r = np.fft.irfft2(f.real ** 2 + f.imag ** 2, (p, p))
    k = np.arange(p)
    lag = np.where(k < p // 2, k, k - p)
    cnt = np.clip(t - np.abs(lag), 1, None).astype(np.float64)
    r = r / (cnt[:, None] * cnt[None, :])
    r0 = r[0, 0]
    if not r0 > 0:
        return math.nan
    rs = np.fft.fftshift(r) / r0
    tau_hi = t / 2.0
    taus = np.arange(_TAU_LO, tau_hi + 1e-9, 0.25)
    if taus.size < 3:
        return math.nan
    prof = ndimage.map_coordinates(rs, [p // 2 + taus * st, p // 2 + taus * ct], order=1, mode="nearest")
    # 방향 오차에 둔감하도록 가로 방향 +-1 px 에서도 본다
    best = prof
    for off in (-1.0, 1.0):
        p2 = ndimage.map_coordinates(rs, [p // 2 + taus * st + off * ct, p // 2 + taus * ct - off * st], order=1,
                                     mode="nearest")
        best = np.minimum(best, p2)
    i = int(np.argmin(best))
    if best[i] > _PEAK_MIN or i == 0 or i == taus.size - 1:
        return math.nan
    y0, y1, y2 = best[i - 1], best[i], best[i + 1]
    den = y0 - 2 * y1 + y2
    dl = 0.5 * (y0 - y2) / den * 0.25 if den > 0 else 0.0
    return float(taus[i] + dl)


def estimate_motion_field(image, *, opts=None) -> MotionField:
    opts = BlurOpts() if opts is None else opts
    img = _to_luma(image)
    t = opts.tile_px
    gh, gw = img.shape[0] // t, img.shape[1] // t
    if gh == 0 or gw == 0:
        raise BlurEstError(f"image{img.shape}: tile_px={t} 타일이 0개다")
    ix = ndimage.correlate(img, _SCHARR, mode="reflect")
    iy = ndimage.correlate(img, _SCHARR.T, mode="reflect")
    theta = np.full((gh, gw), np.nan)
    length = np.full((gh, gw), np.nan)
    aniso = np.zeros((gh, gw))
    motion = np.zeros((gh, gw), dtype=bool)
    for i in range(gh):
        for j in range(gw):
            sl = (slice(i * t, (i + 1) * t), slice(j * t, (j + 1) * t))
            tx, ty = ix[sl], iy[sl]
            jxx, jyy, jxy = float(np.sum(tx * tx)), float(np.sum(ty * ty)), float(np.sum(tx * ty))
            tr = jxx + jyy
            if tr / (t * t) <= _FLAT_EPS:
                continue
            an = min(max(math.hypot(jxx - jyy, 2.0 * jxy) / tr, 0.0), 1.0)
            phi = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)       # 지배 기울기 방향
            th = _wrap_pi(phi + 0.5 * math.pi)                  # 줄무늬 방향
            theta[i, j], aniso[i, j] = th, an
            if an >= opts.aniso_min:
                ln = _streak_length(tx, ty, th, t)
                if math.isfinite(ln) and ln >= opts.min_length_px:
                    length[i, j] = ln
                    motion[i, j] = True
    th_g = len_g = None
    if motion.any():
        a = aniso[motion]
        th2 = 2.0 * theta[motion]
        th_g = _wrap_pi(0.5 * math.atan2(float(np.sum(a * np.sin(th2))), float(np.sum(a * np.cos(th2)))))
        len_g = float(np.median(length[motion]))
    return MotionField(theta=theta, length_px=length, anisotropy=aniso, is_motion=motion, tile_px=t,
                       theta_global=th_g, length_global=len_g, frac_motion=float(motion.mean()))


def motion_depth_consistency(length_px, depth_m, weight=None) -> tuple:
    L = np.asarray(_to_numpy(length_px, "length_px"), dtype=np.float64).ravel()
    z = np.asarray(_to_numpy(depth_m, "depth_m"), dtype=np.float64).ravel()
    if L.shape != z.shape:
        raise BlurEstError("length_px 와 depth_m 의 길이가 다르다")
    w = np.ones_like(L) if weight is None else np.asarray(_to_numpy(weight, "weight"), dtype=np.float64).ravel()
    if w.shape != L.shape:
        raise BlurEstError("weight 의 길이가 다르다")
    if np.any(np.isnan(z)) or np.any(z <= 0):
        raise BlurEstError("depth_m 은 양수(+inf 허용)여야 한다")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise BlurEstError("weight 는 유한한 비음수여야 한다")
    ok = np.isfinite(L) & (L > 0) & (w > 0)
    if int(ok.sum()) < 2:
        return (math.nan, math.nan, math.nan)
    L, w, u = L[ok], w[ok], 1.0 / z[ok]
    sw = np.sqrt(w)
    A = np.stack([np.ones_like(u), u], axis=1) * sw[:, None]
    (a, b), _ = nnls(A, L * sw)
    r = L - (a + b * u)
    mean = float(np.sum(w * L) / np.sum(w))
    rel = math.sqrt(float(np.sum(w * r * r) / np.sum(w))) / max(mean, 1e-6)
    return (float(a), float(b), float(rel))


# ---------------------------------------------------------------------------------------------------------------
# 통합
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class BlurRead:
    defocus: DefocusFit
    motion: MotionField | None
    motion_consistency: tuple | None
    warnings: tuple

    def to_meta(self) -> dict:
        mc = None if self.motion_consistency is None else [_jf(v) for v in self.motion_consistency]
        return {"defocus": self.defocus.to_meta(), "motion": None if self.motion is None else self.motion.to_meta(),
                "motion_consistency": mc, "warnings": list(self.warnings)}


def _block_median(arr2d, i, j, t, img_hw, reduce):
    hd, wd = arr2d.shape
    h, w = img_hw
    r0 = min(int(math.floor(i * t * hd / h)), hd - 1)
    r1 = max(int(math.ceil((i + 1) * t * hd / h)), r0 + 1)
    c0 = min(int(math.floor(j * t * wd / w)), wd - 1)
    c1 = max(int(math.ceil((j + 1) * t * wd / w)), c0 + 1)
    return reduce(arr2d[r0:r1, c0:c1])


def read_blur(*, defocus, depth_m, f_px, width_px, image=None, weight=None, sigma_s=0.0, opts=None) -> BlurRead:
    opts = BlurOpts() if opts is None else opts
    fit = estimate_defocus(defocus, depth_m, f_px=f_px, width_px=width_px, weight=weight, sigma_s=sigma_s, opts=opts)
    if image is None:
        return BlurRead(defocus=fit, motion=None, motion_consistency=None, warnings=fit.warnings)
    mf = estimate_motion_field(image, opts=opts)
    warns = set(fit.warnings)
    mc = None
    if int(mf.is_motion.sum()) >= opts.min_motion_tiles:
        z = _to_2d(depth_m, "depth_m")
        w = None if weight is None else _to_2d(weight, "weight")
        img = _to_luma(image)
        hw = img.shape
        ls, zs, ws = [], [], []
        for i, j in zip(*np.nonzero(mf.is_motion)):
            ls.append(mf.length_px[i, j])
            zs.append(_block_median(z, int(i), int(j), mf.tile_px, hw, lambda b: float(np.median(b))))
            ws.append(1.0 if w is None else _block_median(w, int(i), int(j), mf.tile_px, hw,
                                                          lambda b: float(np.mean(b))))
        mc = motion_depth_consistency(np.array(ls), np.array(zs), np.array(ws))
        if math.isfinite(mc[2]) and mc[2] > opts.tau_motion:
            warns.add("motion_nonphysical")
    return BlurRead(defocus=fit, motion=mf, motion_consistency=mc, warnings=tuple(sorted(warns)))
