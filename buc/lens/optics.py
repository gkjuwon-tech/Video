"""LNS-05: 렌즈 수차 근사식 부품 (왜곡 · 횡색수차 · cat-eye · Hullin 고스트 · veiling glare · 회절 별).

학습 0, 순수 수식. 선형 장면 영상 (H,W) 또는 (H,W,3) float32/float64 에 작동한다.
좌표 규약: 화소 중심이 정수 (x=열, y=행), 주점 c=((W-1)/2,(H-1)/2), 정규화 반지름 r=|p-c|/R, R=hypot(W-1,H-1)/2.
각도 t 는 방향 (cos t, sin t), 도 단위, y 는 아래. 조리개 규약은 LNS-02 dof.aperture_kernel 과 같다
(날 수 0 또는 3..18, 변 법선 rot+(2i+1)·180/N, 외접 반지름 d/2).
import 부작용·전역 RNG·RuntimeWarning 없음, torch 를 쓰지 않는다. 입력은 바꾸지 않고 새 배열을 돌려준다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import fft as _sfft
from scipy.signal import fftconvolve

from buc.core import color as _color
from buc.core import contracts as _C
from buc.core import pyramid as _pyr
from buc.lens import dof as _dof
from buc.lens import theta as _theta

__all__ = [
    "GLARE_EXPONENT", "GLARE_R0_FRAC", "STAR_CHROMA", "STAR_PSF_SIZE", "STAR_APERTURE_FRAC", "CATSEYE_MAX_SHIFT",
    "distortion_scale", "apply_distortion", "undistort_image", "ideal_to_distorted", "distorted_to_ideal",
    "apply_lateral_ca", "catseye_kernel", "apply_catseye", "GhostSpec", "DEFAULT_GHOSTS", "ghost_layer",
    "veiling_glare", "star_psf", "star_layer", "OpticsSpec", "spec_from_theta", "apply_optics",
    "apply_optics_frames",
]

GLARE_EXPONENT = 1.5
GLARE_R0_FRAC = 0.03
STAR_CHROMA = (1.12, 1.0, 0.88)
STAR_PSF_SIZE = 257
STAR_APERTURE_FRAC = 0.25
CATSEYE_MAX_SHIFT = 0.5


# ---------------------------------------------------------------------------------------------------------------
# 검증 도우미
# ---------------------------------------------------------------------------------------------------------------
def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _num(name, x, lo=None, hi=None, lo_open=False, hi_open=False) -> float:
    """유한 실수(bool 불가) 를 float 로. 범위 [lo,hi] (open 이면 열린 끝) 를 벗어나면 ValueError."""
    if not _is_real(x) or not math.isfinite(float(x)):
        raise ValueError(f"{name}={x!r}: 유한한 실수여야 한다")
    v = float(x)
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ValueError(f"{name}={v}: {'>' if lo_open else '>='} {lo} 이어야 한다")
    if hi is not None and (v >= hi if hi_open else v > hi):
        raise ValueError(f"{name}={v}: {'<' if hi_open else '<='} {hi} 이어야 한다")
    return v


def _blades(blades) -> int:
    if isinstance(blades, (bool, np.bool_)) or not isinstance(blades, numbers.Integral):
        raise ValueError(f"blades={blades!r}: 정수여야 한다")
    if not (blades == 0 or 3 <= blades <= 18):
        raise ValueError(f"blades={blades}: 0 또는 3..18 이어야 한다")
    return int(blades)


def _img(x, name="img", need3=True) -> np.ndarray:
    """float32/float64 영상 검사. need3 이면 (H,W,3) 만, 아니면 (H,W) 도 허용."""
    a = np.asarray(x)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    ok = (a.ndim == 3 and a.shape[2] == 3) or (not need3 and a.ndim == 2)
    if not ok:
        raise ValueError(f"{name}: {'(H,W,3)' if need3 else '(H,W) 또는 (H,W,3)'} 이어야 한다 (shape {a.shape})")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{name}: 빈 영상 (shape {a.shape})")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: 유한하지 않은 값이 있다")
    return a


def _border(border) -> str:
    if border not in ("edge", "zero"):
        raise ValueError(f"border={border!r}: 'edge' 또는 'zero' 여야 한다")
    return border


def _half_diag(h, w) -> float:
    r = math.hypot(w - 1, h - 1) / 2.0
    return r if r > 0.0 else 1.0


def _center_xy(h, w, center):
    if center is None:
        return (w - 1) / 2.0, (h - 1) / 2.0
    try:
        cx, cy = center
    except (TypeError, ValueError) as e:
        raise ValueError(f"center={center!r}: (x, y) 쌍이어야 한다") from e
    return _num("center.x", cx), _num("center.y", cy)


def _shape_hw(shape_hw):
    try:
        h, w = shape_hw
    except (TypeError, ValueError) as e:
        raise ValueError(f"shape_hw={shape_hw!r}: (H, W) 쌍이어야 한다") from e
    for v in (h, w):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) or v < 1:
            raise ValueError(f"shape_hw={shape_hw!r}: 1 이상의 정수여야 한다")
    return int(h), int(w)


# ---------------------------------------------------------------------------------------------------------------
# 표본화
# ---------------------------------------------------------------------------------------------------------------
def _resample(a: np.ndarray, sx: np.ndarray, sy: np.ndarray, border: str, nearest: bool) -> np.ndarray:
    """a (H,W[,C]) 를 화소 위치 (sx, sy) 에서 표본화. 출력 모양 sx.shape + a.shape[2:], dtype 은 a 와 같다."""
    H, W = a.shape[:2]
    extra = (1,) * (a.ndim - 2)
    if nearest:
        ix = np.floor(np.clip(sx, -4.0, W + 3.0) + 0.5).astype(np.int64)
        iy = np.floor(np.clip(sy, -4.0, H + 3.0) + 0.5).astype(np.int64)
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        out = a[np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)]
        if border == "zero":
            out = np.where(inside.reshape(inside.shape + extra), out, np.zeros((), a.dtype))
        return out
    src = a.astype(np.float64, copy=False)
    if border == "edge":
        u, v = np.clip(sx, 0.0, W - 1.0), np.clip(sy, 0.0, H - 1.0)
    else:
        u, v = np.clip(sx, -2.0, W + 1.0), np.clip(sy, -2.0, H + 1.0)
    x0f, y0f = np.floor(u), np.floor(v)
    fx, fy = u - x0f, v - y0f
    x0, y0 = x0f.astype(np.int64), y0f.astype(np.int64)

    def tap(yy, xx):
        val = src[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)]
        if border == "zero":
            inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            val = np.where(inside.reshape(inside.shape + extra), val, 0.0)
        return val

    def wt(t):
        return t.reshape(t.shape + extra)

    out = (wt((1.0 - fx) * (1.0 - fy)) * tap(y0, x0) + wt(fx * (1.0 - fy)) * tap(y0, x0 + 1)
           + wt((1.0 - fx) * fy) * tap(y0 + 1, x0) + wt(fx * fy) * tap(y0 + 1, x0 + 1))
    return out.astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 방사 왜곡
# ---------------------------------------------------------------------------------------------------------------
def _check_k(k1, k2):
    k1 = _num("k1", k1)
    k2 = _num("k2", k2)
    if abs(k1) > 2.0 or abs(k2) > 2.0:
        raise ValueError(f"k1={k1}, k2={k2}: |k| <= 2 이어야 한다")
    # g(u) = 1 + 3 k1 u + 5 k2 u^2 (u = r^2 in [0,1]) 가 양수여야 r s(r) 이 단조 -> 역사상이 있다
    gmin = min(1.0, 1.0 + 3.0 * k1 + 5.0 * k2)
    if k2 > 0.0:
        uv = -3.0 * k1 / (10.0 * k2)
        if 0.0 < uv < 1.0:
            gmin = min(gmin, 1.0 - 9.0 * k1 * k1 / (20.0 * k2))
    if not gmin > 0.0:
        raise ValueError(f"k1={k1}, k2={k2}: 1 + 3 k1 r² + 5 k2 r⁴ > 0 (r in [0,1]) 이어야 한다")
    return k1, k2


def distortion_scale(r, k1, k2) -> np.ndarray:
    """s(r) = 1 + k1 r² + k2 r⁴ (후방 사상 배율)."""
    k1, k2 = _check_k(k1, k2)
    rr = np.asarray(r)
    if rr.dtype.kind not in "iuf":
        raise TypeError(f"r: 수치 배열이어야 한다 (dtype {rr.dtype})")
    r2 = rr.astype(np.float64) ** 2
    return np.asarray(1.0 + k1 * r2 + k2 * r2 * r2)


def _backward_grid(h, w, k1, k2, center):
    """왜곡 영상의 화소 p 가 이상 영상에서 읽는 위치 c + (p-c) s(r)."""
    cx, cy = _center_xy(h, w, center)
    R = _half_diag(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    r2 = (dx * dx + dy * dy) / (R * R)
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return cx + dx * s, cy + dy * s


def _r_peak(k1, k2) -> float:
    """r s(r) 의 첫 극값 반지름 (d/dr = 0). 없으면 inf."""
    if k2 == 0.0:
        return math.sqrt(-1.0 / (3.0 * k1)) if k1 < 0.0 else math.inf
    disc = 9.0 * k1 * k1 - 20.0 * k2
    if disc < 0.0:
        return math.inf
    sq = math.sqrt(disc)
    us = [u for u in ((-3.0 * k1 - sq) / (10.0 * k2), (-3.0 * k1 + sq) / (10.0 * k2)) if u > 0.0]
    return math.sqrt(min(us)) if us else math.inf


def _solve_rd(rq: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """r_d (1 + k1 r_d² + k2 r_d⁴) = r_q 를 푼다 (뉴턴 + 이분법 가드). 해가 없는 r_q 는 극값 반지름으로 자른다."""
    cap = _r_peak(k1, k2)

    def f(r):
        r2 = r * r
        return r * (1.0 + r2 * (k1 + k2 * r2))

    def fp(r):
        r2 = r * r
        return 1.0 + r2 * (3.0 * k1 + 5.0 * k2 * r2)

    hi = np.minimum(rq, cap)
    for _ in range(80):
        need = (f(hi) < rq) & (hi < cap)
        if not need.any():
            break
        hi = np.where(need, np.minimum(hi * 2.0 + 1e-12, cap), hi)
    lo = np.zeros_like(rq)
    r = hi.copy()
    for _ in range(80):
        fr = f(r) - rq
        lo = np.where(fr < 0.0, r, lo)
        hi = np.where(fr >= 0.0, r, hi)
        d = fp(r)
        good = d > 1e-12
        rn = r - fr / np.where(good, d, 1.0)
        ok = good & (rn >= lo) & (rn <= hi)
        rn = np.where(ok, rn, 0.5 * (lo + hi))
        delta = float(np.max(np.abs(rn - r))) if rn.size else 0.0
        r = rn
        if delta <= 1e-15:
            break
    return r


def _ideal_to_distorted_xy(x, y, k1, k2, h, w, center):
    cx, cy = _center_xy(h, w, center)
    R = _half_diag(h, w)
    dx, dy = x - cx, y - cy
    rq = np.hypot(dx, dy) / R
    rd = _solve_rd(rq, k1, k2)
    ratio = np.where(rq > 0.0, rd / np.where(rq > 0.0, rq, 1.0), 1.0)
    return cx + dx * ratio, cy + dy * ratio


def _points(xy):
    a = np.asarray(xy)
    if a.dtype.kind not in "iuf":
        raise TypeError(f"xy: 수치 배열이어야 한다 (dtype {a.dtype})")
    a = a.astype(np.float64)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise ValueError(f"xy: 마지막 축이 2 여야 한다 (shape {a.shape})")
    if not np.isfinite(a).all():
        raise ValueError("xy: 유한하지 않은 값이 있다")
    return a


def ideal_to_distorted(xy, k1, k2, shape_hw, center=None) -> np.ndarray:
    """이상(핀홀) 화소 위치 -> 왜곡 영상 위치. r_d (1 + k1 r_d² + k2 r_d⁴) = r_q 의 근을 뉴턴으로 푼다."""
    k1, k2 = _check_k(k1, k2)
    h, w = _shape_hw(shape_hw)
    a = _points(xy)
    ox, oy = _ideal_to_distorted_xy(a[..., 0], a[..., 1], k1, k2, h, w, center)
    return np.stack([ox, oy], axis=-1)


def distorted_to_ideal(xy, k1, k2, shape_hw, center=None) -> np.ndarray:
    """왜곡 영상 위치 -> 이상 화소 위치 (닫힌 식 c + (p-c) s(r))."""
    k1, k2 = _check_k(k1, k2)
    h, w = _shape_hw(shape_hw)
    a = _points(xy)
    cx, cy = _center_xy(h, w, center)
    R = _half_diag(h, w)
    dx, dy = a[..., 0] - cx, a[..., 1] - cy
    r2 = (dx * dx + dy * dy) / (R * R)
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.stack([cx + dx * s, cy + dy * s], axis=-1)


def _warp(a: np.ndarray, k1, k2, border, nearest, center, inverse) -> np.ndarray:
    h, w = a.shape[:2]
    if k1 == 0.0 and k2 == 0.0:
        return a.copy()
    if inverse:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        sx, sy = _ideal_to_distorted_xy(xx, yy, k1, k2, h, w, center)
    else:
        sx, sy = _backward_grid(h, w, k1, k2, center)
    return _resample(a, sx, sy, border, nearest)


def _distort_api(img, k1, k2, border, nearest, center, inverse):
    a = _img(img, "img", need3=False)
    k1, k2 = _check_k(k1, k2)
    border = _border(border)
    if not isinstance(nearest, (bool, np.bool_)):
        raise ValueError(f"nearest={nearest!r}: bool 이어야 한다")
    _center_xy(a.shape[0], a.shape[1], center)
    return _warp(a, k1, k2, border, bool(nearest), center, inverse)


def apply_distortion(img, k1, k2, *, border="edge", nearest=False, center=None) -> np.ndarray:
    """후방 사상 out(p) = in(c + (p-c) s(r)), s = 1 + k1 r² + k2 r⁴. k1>0 배럴, k1<0 핀쿠션."""
    return _distort_api(img, k1, k2, border, nearest, center, False)


def undistort_image(img, k1, k2, *, border="edge", nearest=False, center=None) -> np.ndarray:
    """apply_distortion 의 역: out(q) = in(ideal_to_distorted(q))."""
    return _distort_api(img, k1, k2, border, nearest, center, True)


# ---------------------------------------------------------------------------------------------------------------
# 횡색수차
# ---------------------------------------------------------------------------------------------------------------
def apply_lateral_ca(rgb, red_px=0.0, blue_px=0.0, *, border="edge") -> np.ndarray:
    """채널 c 를 배율 1+e_c (e = px/R) 로 확대한다 (초록 기준). 후방 쌍선형 표본화(복사 보존)."""
    a = _img(rgb, "rgb")
    border = _border(border)
    h, w = a.shape[:2]
    R = _half_diag(h, w)
    px = {}
    for name, v in (("red_px", red_px), ("blue_px", blue_px)):
        v = _num(name, v)
        if abs(v) >= 0.5 * R:
            raise ValueError(f"{name}={v}: |px| < 0.5 R ({0.5 * R:.2f}) 이어야 한다")
        px[name] = v
    out = a.copy()
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    for ch, name in ((0, "red_px"), (2, "blue_px")):
        if px[name] == 0.0:
            continue
        m = 1.0 + px[name] / R
        out[..., ch] = _resample(a[..., ch], cx + (xx - cx) / m, cy + (yy - cy) / m, border, False)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 조리개 면적 적분 (경계 띠만 부분 표본)
# ---------------------------------------------------------------------------------------------------------------
def _snap(v: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(v) < eps else v


def _margin(X, Y, shape):
    """모양의 부호 있는 여유: 안쪽 < 0 < 바깥 (안쪽은 가장 가까운 변까지 거리, 바깥은 하한)."""
    ox, oy, R, blades, rot = shape
    if blades == 0:
        return np.hypot(X - ox, Y - oy) - R
    m = None
    for i in range(blades):
        phi = rot + (2 * i + 1) * math.pi / blades
        p = (X - ox) * math.cos(phi) + (Y - oy) * math.sin(phi)
        m = p if m is None else np.maximum(m, p)
    return m - R * math.cos(math.pi / blades)


def _margin_all(X, Y, shapes):
    m = _margin(X, Y, shapes[0])
    for s in shapes[1:]:
        m = np.maximum(m, _margin(X, Y, s))
    return m


def _coverage(X: np.ndarray, Y: np.ndarray, shapes, ss: int) -> np.ndarray:
    """화소 (중심 X,Y, 한 변 1) 가 볼록 모양들의 교집합에 덮이는 비율. 경계 띠만 ss×ss 부분 표본."""
    M = _margin_all(X, Y, shapes)
    cov = (M < -0.75).astype(np.float64)
    idx = np.nonzero(np.abs(M) <= 0.75)
    if idx[0].size:
        o = (np.arange(ss) + 0.5) / ss - 0.5
        xs = X[idx][:, None, None] + o[None, None, :]
        ys = Y[idx][:, None, None] + o[None, :, None]
        xs, ys = np.broadcast_to(xs, (xs.shape[0], ss, ss)), np.broadcast_to(ys, (ys.shape[0], ss, ss))
        cov[idx] = (_margin_all(xs, ys, shapes) <= 0.0).mean(axis=(1, 2))
    return cov


# ---------------------------------------------------------------------------------------------------------------
# cat-eye
# ---------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _lens_kernel(d: float, u: float, ang_deg: float, blades: int, rot_deg: float) -> np.ndarray:
    """u = strength·radial_frac. 읽기 전용 float32 커널."""
    if u <= 0.0 or d <= 1.0:
        k = _dof.aperture_kernel(d, blades, rot_deg)
        k.flags.writeable = False
        return k
    half = int(math.ceil(d / 2.0))
    R = d / 2.0
    o = CATSEYE_MAX_SHIFT * u * d
    ang = math.radians(ang_deg)
    ex, ey = _snap(math.cos(ang)), _snap(math.sin(ang))
    rot = math.radians(rot_deg)
    shapes = ((ex * o / 2.0, ey * o / 2.0, R, blades, rot), (-ex * o / 2.0, -ey * o / 2.0, R, blades, rot))
    ax = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(ax, ax)
    cov = _coverage(X, Y, shapes, 8)
    s = float(cov.sum())
    if s <= 0.0:
        cov = np.zeros_like(cov)
        cov[half, half] = 1.0
    else:
        cx0, cy0 = _snap(float((cov * X).sum() / s), 1e-9), _snap(float((cov * Y).sum() / s), 1e-9)
        if cx0 != 0.0 or cy0 != 0.0:  # 질량중심을 커널 중심에 맞춘다
            cov = _coverage(X + cx0, Y + cy0, shapes, 8)
    k = (cov / cov.sum()).astype(np.float32)
    k.flags.writeable = False
    return k


def catseye_kernel(diameter_px, radial_frac, angle_deg, strength, blades=0, rotation_deg=0.0) -> np.ndarray:
    """반지름 d/2 조리개 두 개를 간격 o = CATSEYE_MAX_SHIFT·strength·radial_frac·d 만큼 (방향 angle_deg) 어긋나게
    겹친 교집합 (질량중심을 중심에 맞춤). 합 1, 크기 = dof.aperture_kernel(d) 와 같다."""
    d = _num("diameter_px", diameter_px, lo=0.0)
    rf = _num("radial_frac", radial_frac, lo=0.0, hi=1.0)
    ang = _num("angle_deg", angle_deg)
    st = _num("strength", strength, lo=0.0, hi=1.0)
    b = _blades(blades)
    rot = _num("rotation_deg", rotation_deg)
    return _lens_kernel(d, rf * st, ang, b, rot).copy()


def _quant_d(d: np.ndarray) -> np.ndarray:
    step = np.where(d <= 8.0, 0.25, np.where(d <= 32.0, 0.5, np.where(d <= 96.0, 1.0, 2.0)))
    return np.round(d / step) * step


def apply_catseye(rgb, coc_px, *, strength, blades=0, blade_rot_deg=0.0) -> np.ndarray:
    """단일 층 산란 블러: 화소마다 자기 CoC 와 (r, angle) 의 cat-eye 커널로 퍼뜨린다. CoC<=1 px 는 선명.
    (r, angle) 은 각도 7.5°, 세기·반지름 1/16 단위로 양자화한다. 프레임 경계 d/2 안은 비규정(밖으로 나간 빛은 버린다)."""
    img = _img(rgb, "rgb")
    h, w = img.shape[:2]
    c = np.asarray(coc_px)
    if c.dtype.kind not in "iuf":
        raise TypeError(f"coc_px: 수치 배열이어야 한다 (dtype {c.dtype})")
    if c.shape != (h, w):
        raise ValueError(f"coc_px: (H,W)={(h, w)} 이어야 한다 (shape {c.shape})")
    c = c.astype(np.float64)
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("coc_px: 비음수 유한이어야 한다")
    st = _num("strength", strength, lo=0.0, hi=1.0)
    b = _blades(blades)
    rot = _num("blade_rot_deg", blade_rot_deg)

    src = img.astype(np.float64)
    dq_all = np.where(c > 1.0, _quant_d(c), 0.0)
    sharp = dq_all <= 1.0
    acc = np.where(sharp[..., None], src, 0.0)
    ys, xs = np.nonzero(~sharp)
    if ys.size == 0:
        return acc.astype(img.dtype)
    d = dq_all[ys, xs]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    R = _half_diag(h, w)
    rf = np.minimum(np.hypot(xs - cx, ys - cy) / R, 1.0)
    ang = np.degrees(np.arctan2(ys - cy, xs - cx)) % 180.0
    ui = np.round(st * rf * 16.0).astype(np.int64)
    ai = np.where(ui > 0, np.round(ang / 7.5).astype(np.int64) % 24, 0)
    di = np.round(d * 4.0).astype(np.int64)
    key = (di * 17 + ui) * 24 + ai
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cuts = np.nonzero(np.diff(ks))[0] + 1
    groups = np.split(order, cuts)

    Hm = int(math.ceil(float(d.max()) / 2.0))
    Hp, Wp = h + 2 * Hm, w + 2 * Hm
    Hf, Wf = _sfft.next_fast_len(Hp, real=True), _sfft.next_fast_len(Wp, real=True)
    K2 = 2 * Hm + 1
    buf = np.zeros((Hp, Wp, 3))
    spec_acc = None
    for g in groups:
        k0 = int(g[0])
        dq = float(di[k0]) / 4.0
        u = float(ui[k0]) / 16.0
        a_deg = float(ai[k0]) * 7.5
        ker = _lens_kernel(dq, u, a_deg, b, rot)
        hk = (ker.shape[0] - 1) // 2
        gy, gx = ys[g], xs[g]
        if g.size <= 40:
            kk = ker.astype(np.float64)[:, :, None]
            for y, x in zip(gy.tolist(), gx.tolist()):
                buf[y + Hm - hk:y + Hm + hk + 1, x + Hm - hk:x + Hm + hk + 1] += kk * src[y, x]
            continue
        m = np.zeros((h, w, 3))
        m[gy, gx] = src[gy, gx]
        emb = np.zeros((K2, K2))
        emb[Hm - hk:Hm + hk + 1, Hm - hk:Hm + hk + 1] = ker
        part = _sfft.rfft2(m, s=(Hf, Wf), axes=(0, 1)) * _sfft.rfft2(emb, s=(Hf, Wf))[:, :, None]
        spec_acc = part if spec_acc is None else spec_acc + part
    if spec_acc is not None:
        buf += _sfft.irfft2(spec_acc, s=(Hf, Wf), axes=(0, 1))[:Hp, :Wp]
    return (acc + buf[Hm:Hm + h, Hm:Hm + w]).astype(img.dtype)


# ---------------------------------------------------------------------------------------------------------------
# Hullin 고스트
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GhostSpec:
    scale: float
    size_frac: float
    gain: float
    tint: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "scale", _num("scale", self.scale, lo=-3.0, hi=3.0))
        object.__setattr__(self, "size_frac", _num("size_frac", self.size_frac, lo=0.005, hi=0.5))
        object.__setattr__(self, "gain", _num("gain", self.gain, lo=0.0))
        try:
            t = tuple(self.tint)
        except TypeError as e:
            raise ValueError(f"tint={self.tint!r}: 길이 3 이어야 한다") from e
        if len(t) != 3:
            raise ValueError(f"tint={self.tint!r}: 길이 3 이어야 한다")
        object.__setattr__(self, "tint", tuple(_num("tint", v, lo=0.0) for v in t))


DEFAULT_GHOSTS = (
    GhostSpec(-0.30, 0.060, 0.0200, (1.00, 0.85, 0.65)),
    GhostSpec(-0.75, 0.110, 0.0120, (0.65, 0.85, 1.00)),
    GhostSpec(-1.25, 0.050, 0.0100, (0.80, 1.00, 0.70)),
    GhostSpec(0.55, 0.040, 0.0150, (0.90, 1.00, 0.85)),
    GhostSpec(1.45, 0.080, 0.0080, (1.00, 0.70, 0.90)),
    GhostSpec(2.10, 0.150, 0.0060, (0.70, 0.80, 1.00)),
)


def _splat(E: np.ndarray, scale: float) -> np.ndarray:
    """E (H,W,3) 를 중심 기준 p' = c + scale (p-c) 로 광속 보존 쌍선형 옮긴다 (프레임 밖은 버린다)."""
    h, w = E.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = np.nonzero(E.max(axis=2) > 0.0)
    out = np.zeros((h * w, 3))
    if ys.size == 0:
        return out.reshape(h, w, 3)
    vals = E[ys, xs]
    px = cx + scale * (xs - cx)
    py = cy + scale * (ys - cy)
    x0f, y0f = np.floor(px), np.floor(py)
    fx, fy = px - x0f, py - y0f
    x0, y0 = x0f.astype(np.int64), y0f.astype(np.int64)
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (wgt > 0.0)
        if not ok.any():
            continue
        idx = yi[ok] * w + xi[ok]
        for ch in range(3):
            out[:, ch] += np.bincount(idx, weights=(wgt * vals[:, ch])[ok], minlength=h * w)
    return out.reshape(h, w, 3)


def _conv_same(arr: np.ndarray, ker: np.ndarray) -> np.ndarray:
    if ker.shape == (1, 1):
        return arr.copy()
    return fftconvolve(arr, ker.astype(np.float64)[:, :, None], mode="same", axes=(0, 1))


def _excess(a: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(a.astype(np.float64) - threshold, 0.0)


def ghost_layer(rgb_lin, ghosts, *, threshold=0.8, blades=0, blade_rot_deg=0.0) -> np.ndarray:
    """가산 고스트 층. E = max(rgb - threshold, 0) 를 p' = c + scale (p-c) 로 옮기고 aperture_kernel(size_frac·W) 로
    퍼뜨려 gain·tint 를 곱한다. 총 광속 = gain·tint_c·ΣE_c (프레임 안)."""
    a = _img(rgb_lin, "rgb_lin")
    try:
        gs = tuple(ghosts)
    except TypeError as e:
        raise ValueError("ghosts: GhostSpec 의 시퀀스여야 한다") from e
    for g in gs:
        if not isinstance(g, GhostSpec):
            raise ValueError(f"ghosts: GhostSpec 이 아닌 원소 {g!r}")
    thr = _num("threshold", threshold, lo=0.0)
    b = _blades(blades)
    rot = _num("blade_rot_deg", blade_rot_deg)
    h, w = a.shape[:2]
    out = np.zeros((h, w, 3))
    E = _excess(a, thr)
    if gs and E.any():
        for g in gs:
            if g.gain == 0.0 or not any(g.tint):
                continue
            ker = _dof.aperture_kernel(g.size_frac * w, b, rot)
            lay = _conv_same(_splat(E, g.scale), ker)
            out += lay * (g.gain * np.asarray(g.tint))
    return np.maximum(out, 0.0).astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# veiling glare
# ---------------------------------------------------------------------------------------------------------------
def veiling_glare(rgb_lin, amount, *, r0_frac=GLARE_R0_FRAC) -> np.ndarray:
    """out = (1-a) x + a (h*x)/(h*1),  h(r) = (1 + (r/r0)²)^-1.5,  r0 = r0_frac·W px. 채널 독립."""
    a = _img(rgb_lin, "rgb_lin")
    amt = _num("amount", amount, lo=0.0, hi=1.0)
    r0f = _num("r0_frac", r0_frac, lo=0.0, hi=1.0, lo_open=True)
    if amt == 0.0:
        return a.copy()
    h, w = a.shape[:2]
    r0 = r0f * w
    yy, xx = np.mgrid[-(h - 1):h, -(w - 1):w].astype(np.float64)
    hk = (1.0 + (xx * xx + yy * yy) / (r0 * r0)) ** (-GLARE_EXPONENT)
    x4 = np.concatenate([a.astype(np.float64), np.ones((h, w, 1))], axis=2)
    conv = fftconvolve(x4, hk[:, :, None], mode="same", axes=(0, 1))
    blurred = conv[..., :3] / conv[..., 3:4]
    out = (1.0 - amt) * a.astype(np.float64) + amt * blurred
    return out.astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 회절 별
# ---------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _star_psf_cached(blades: int, rot_deg: float, size: int, aperture_frac: float) -> np.ndarray:
    half = size // 2
    ax = np.arange(size, dtype=np.float64) - half
    X, Y = np.meshgrid(ax, ax)
    rot = math.radians(rot_deg)
    out = np.empty((size, size, 3))
    for ch in range(3):
        R = aperture_frac * size / (2.0 * STAR_CHROMA[ch])
        cov = _coverage(X, Y, ((0.0, 0.0, R, blades, rot),), 16)
        p = np.fft.fftshift(np.abs(np.fft.fft2(cov)) ** 2)
        out[..., ch] = p / p.sum()
    res = out.astype(np.float32)
    res.flags.writeable = False
    return res


def star_psf(blades, rotation_deg=0.0, *, size=STAR_PSF_SIZE, aperture_frac=STAR_APERTURE_FRAC) -> np.ndarray:
    """조리개(지름 aperture_frac·size/s_c 의 N각형·원)의 |FFT|². 중심 index size//2, 채널마다 합 1."""
    b = _blades(blades)
    rot = _num("rotation_deg", rotation_deg)
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, numbers.Integral) or size < 33 or size % 2 == 0:
        raise ValueError(f"size={size!r}: 33 이상의 홀수 정수여야 한다")
    af = _num("aperture_frac", aperture_frac, lo=0.0, hi=0.5, lo_open=True)
    period = 360.0 / b if b else 360.0
    return _star_psf_cached(b, round(rot % period, 9), int(size), af).copy()


@lru_cache(maxsize=64)
def _star_kernel(blades: int, rot_deg: float, width: int) -> np.ndarray:
    psf = _star_psf_cached(blades, rot_deg, STAR_PSF_SIZE, STAR_APERTURE_FRAC).astype(np.float64)
    if width == 1:
        k = np.ones((1, 1, 3))
    elif width == STAR_PSF_SIZE:
        k = psf
    else:
        k = _pyr.resize(psf, (width, width), method="bilinear")
    k = k / k.sum(axis=(0, 1), keepdims=True)
    k.flags.writeable = False
    return k


def star_layer(rgb_lin, *, blades, rotation_deg=0.0, threshold=0.8, size_frac=0.25) -> np.ndarray:
    """단위 세기 별 층: E = max(rgb - threshold, 0) 를 PSF (전체 폭 size_frac·W px, 합 1) 로 컨볼루션한다."""
    a = _img(rgb_lin, "rgb_lin")
    b = _blades(blades)
    rot = _num("rotation_deg", rotation_deg)
    thr = _num("threshold", threshold, lo=0.0)
    sf = _num("size_frac", size_frac, lo=0.0, hi=1.0, lo_open=True)
    h, w = a.shape[:2]
    E = _excess(a, thr)
    if not E.any():
        return np.zeros(a.shape, a.dtype)
    width = max(1, 2 * round((sf * w - 1.0) / 2.0) + 1)
    period = 360.0 / b if b else 360.0
    ker = _star_kernel(b, round(rot % period, 9), int(width))
    if width == 1:
        out = E
    else:
        out = fftconvolve(E, ker, mode="same", axes=(0, 1))
    return np.maximum(out, 0.0).astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 스펙 · 체인
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class OpticsSpec:
    k1: float = 0.0
    k2: float = 0.0
    ca_red_px: float = 0.0
    ca_blue_px: float = 0.0
    catseye: float = 0.0
    blades: int = 0
    blade_rot_deg: float = 0.0
    glare: float = 0.0
    glare_r0_frac: float = GLARE_R0_FRAC
    ghosts: tuple = field(default=())
    ghost_threshold: float = 0.8
    star_amount: float = 0.0
    star_size_frac: float = 0.25
    star_threshold: float = 0.8

    def __post_init__(self):
        k1, k2 = _check_k(self.k1, self.k2)
        s = object.__setattr__
        s(self, "k1", k1)
        s(self, "k2", k2)
        s(self, "ca_red_px", _num("ca_red_px", self.ca_red_px))
        s(self, "ca_blue_px", _num("ca_blue_px", self.ca_blue_px))
        s(self, "catseye", _num("catseye", self.catseye, lo=0.0, hi=1.0))
        s(self, "blades", _blades(self.blades))
        s(self, "blade_rot_deg", _num("blade_rot_deg", self.blade_rot_deg))
        s(self, "glare", _num("glare", self.glare, lo=0.0, hi=1.0))
        s(self, "glare_r0_frac", _num("glare_r0_frac", self.glare_r0_frac, lo=0.0, hi=1.0, lo_open=True))
        try:
            gs = tuple(self.ghosts)
        except TypeError as e:
            raise ValueError(f"ghosts={self.ghosts!r}: GhostSpec 의 시퀀스여야 한다") from e
        for g in gs:
            if not isinstance(g, GhostSpec):
                raise ValueError(f"ghosts: GhostSpec 이 아닌 원소 {g!r}")
        s(self, "ghosts", gs)
        s(self, "ghost_threshold", _num("ghost_threshold", self.ghost_threshold, lo=0.0))
        s(self, "star_amount", _num("star_amount", self.star_amount, lo=0.0, hi=1.0))
        s(self, "star_size_frac", _num("star_size_frac", self.star_size_frac, lo=0.0, hi=1.0, lo_open=True))
        s(self, "star_threshold", _num("star_threshold", self.star_threshold, lo=0.0))


# 렌즈별 기본 스펙. ca 는 폭 대비 비율 (해상도에 선형), ghost 는 DEFAULT_GHOSTS gain 에 곱하는 배수.
_LENS_TABLE = {
    "spherical_clean": dict(k1=0.0, k2=0.0, ca_red=0.0, ca_blue=0.0, catseye=0.05, glare=0.015, ghost=0.5),
    "vintage_petzval": dict(k1=0.05, k2=-0.01, ca_red=0.0022, ca_blue=-0.0018, catseye=0.85, glare=0.04, ghost=0.9),
    "vintage_double_gauss": dict(k1=0.03, k2=0.0, ca_red=0.0012, ca_blue=-0.0010, catseye=0.45, glare=0.05,
                                 ghost=1.2),
    "anamorphic_2x": dict(k1=-0.02, k2=0.0, ca_red=0.0008, ca_blue=-0.0008, catseye=0.55, glare=0.035, ghost=1.6),
    "soft_single_coat": dict(k1=0.015, k2=0.0, ca_red=0.0015, ca_blue=-0.0012, catseye=0.25, glare=0.14, ghost=2.5),
}
_STAR_FSTOP0 = 2.8
_STAR_FSTOP_REF = 16.0
_STAR_AMOUNT_REF = 0.25


def _star_amount(f_stop: float) -> float:
    t = math.log2(f_stop / _STAR_FSTOP0) / math.log2(_STAR_FSTOP_REF / _STAR_FSTOP0)
    return _STAR_AMOUNT_REF * min(1.6, max(0.0, t))


def spec_from_theta(theta, width_px) -> OpticsSpec:
    """θ.lens_rx 별 표 + f_stop(별) + extra.blades -> OpticsSpec. 수차 px 는 폭에 비례한다."""
    if not isinstance(theta, _C.Theta):
        raise TypeError(f"theta: Theta 여야 한다 ({type(theta).__name__})")
    if isinstance(width_px, (bool, np.bool_)) or not isinstance(width_px, numbers.Integral) or width_px < 1:
        raise ValueError(f"width_px={width_px!r}: 1 이상의 정수여야 한다")
    if theta.lens_rx not in _theta.LENS_RX_NAMES or theta.lens_rx not in _LENS_TABLE:
        raise ValueError(f"lens_rx={theta.lens_rx!r}: 미등록 렌즈")
    t = _LENS_TABLE[theta.lens_rx]
    extra = theta.extra if isinstance(theta.extra, dict) else {}
    blades = int(extra.get("blades", 0))
    w = float(width_px)
    ghosts = tuple(GhostSpec(g.scale, g.size_frac, g.gain * t["ghost"], g.tint) for g in DEFAULT_GHOSTS)
    return OpticsSpec(k1=t["k1"], k2=t["k2"], ca_red_px=t["ca_red"] * w, ca_blue_px=t["ca_blue"] * w,
                      catseye=t["catseye"], blades=blades, glare=t["glare"], ghosts=ghosts,
                      star_amount=_star_amount(float(theta.f_stop)))


def apply_optics(rgb_lin, spec) -> np.ndarray:
    """out = glare(x) + ghost_layer(x) + star_amount·star_layer(x) (셋 다 원본 x 에서) -> 횡색수차 -> 왜곡.
    spec.catseye 는 소비하지 않는다 (블러는 DOF 몫, 호출자가 apply_catseye)."""
    a = _img(rgb_lin, "rgb_lin")
    if not isinstance(spec, OpticsSpec):
        raise TypeError(f"spec: OpticsSpec 이어야 한다 ({type(spec).__name__})")
    out = veiling_glare(a, spec.glare, r0_frac=spec.glare_r0_frac) if spec.glare > 0.0 else a.copy()
    if spec.ghosts:
        out = out + ghost_layer(a, spec.ghosts, threshold=spec.ghost_threshold, blades=spec.blades,
                                blade_rot_deg=spec.blade_rot_deg)
    if spec.star_amount > 0.0:
        out = out + spec.star_amount * star_layer(a, blades=spec.blades, rotation_deg=spec.blade_rot_deg,
                                                  threshold=spec.star_threshold, size_frac=spec.star_size_frac)
    out = apply_lateral_ca(out, spec.ca_red_px, spec.ca_blue_px)
    return apply_distortion(out, spec.k1, spec.k2)


def apply_optics_frames(frames, theta, *, spec=None):
    """표시값 Frames -> 선형 -> 프레임마다 apply_optics -> 표시값(클립). (N,H,W) AOV 는 같은 왜곡으로 최근접 사상."""
    if not isinstance(frames, _C.Frames):
        raise TypeError(f"frames: Frames 여야 한다 ({type(frames).__name__})")
    rgb = frames.rgb
    n, h, w = rgb.shape[:3]
    if spec is None:
        spec = spec_from_theta(theta, w)
    elif not isinstance(spec, OpticsSpec):
        raise TypeError(f"spec: OpticsSpec 이어야 한다 ({type(spec).__name__})")
    out_rgb = np.empty(rgb.shape, np.float32)
    for i in range(n):
        out_rgb[i] = _color.linear_to_display(apply_optics(_color.srgb_to_linear(rgb[i]), spec))
    aov = {}
    for k, v in frames.aov.items():
        arr = np.asarray(v)
        if arr.ndim == 3 and arr.shape == (n, h, w):
            aov[k] = np.stack([_warp(arr[i], spec.k1, spec.k2, "edge", True, None, False) for i in range(n)])
        else:
            aov[k] = arr.copy()
    return _C.Frames(rgb=out_rgb, fps=frames.fps, aov=aov, shot_hash=frames.shot_hash,
                     world_hash=frames.world_hash)
