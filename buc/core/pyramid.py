"""FND-06 필터·피라미드·리샘플.

영상은 ``(H,W)`` 또는 ``(H,W,C)`` float32|float64(그 밖은 TypeError), 결과 dtype 보존, 내부 계산 float64.
좌표 규약은 픽셀 중심이 정수(반화소 정렬). ``resize`` 의 기준 구현은 Pillow ``Image.resize`` 의 'F' 모드
(채널별, ``reducing_gap=None``)이며 같은 계수 산식을 numpy/scipy.sparse 로 재현한다(Pillow 비의존).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as _sp

__all__ = [
    "gaussian_kernel_1d", "gaussian_blur", "pyr_down", "pyr_up", "gaussian_pyramid", "laplacian_pyramid",
    "laplacian_reconstruct", "laplacian_low_swap", "resize", "sample_bilinear",
]

_PAD = {"reflect": "symmetric", "edge": "edge", "zero": "constant"}


def _check(img, name="img"):
    a = np.asarray(img)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    if a.ndim not in (2, 3):
        raise ValueError(f"{name}: (H,W) 또는 (H,W,C) 여야 한다 (shape {a.shape})")
    if min(a.shape) < 1:
        raise ValueError(f"{name}: 빈 영상 (shape {a.shape})")
    return a


def _pad_mode(border):
    if border not in _PAD:
        raise ValueError(f"border={border!r}: {tuple(_PAD)} 중 하나여야 한다")
    return _PAD[border]


def _pad_axis(a, axis, before, after, mode):
    widths = [(0, 0)] * a.ndim
    widths[axis] = (before, after)
    if mode == "constant":
        return np.pad(a, widths, mode="constant", constant_values=0.0)
    return np.pad(a, widths, mode=mode)


def _sl(axis, ndim, s):
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


# ---------------------------------------------------------------------------------------------------------------
# 가우시안 블러
# ---------------------------------------------------------------------------------------------------------------
def gaussian_kernel_1d(sigma, truncate=4.0) -> np.ndarray:
    sigma, truncate = float(sigma), float(truncate)
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma={sigma}: 양수여야 한다")
    if not (math.isfinite(truncate) and truncate > 0.0):
        raise ValueError(f"truncate={truncate}: 양수여야 한다")
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


def _conv_axis(a, kernel, axis, mode):
    r = (len(kernel) - 1) // 2
    n = a.shape[axis]
    p = _pad_axis(a, axis, r, r, mode)
    out = None
    for k, w in enumerate(kernel):
        term = w * p[_sl(axis, a.ndim, slice(k, k + n))]
        out = term if out is None else out + term
    return out


def gaussian_blur(img, sigma, border="reflect", truncate=4.0) -> np.ndarray:
    a = _check(img)
    mode = _pad_mode(border)
    k = gaussian_kernel_1d(sigma, truncate)
    v = a.astype(np.float64)
    v = _conv_axis(v, k, 0, mode)
    v = _conv_axis(v, k, 1, mode)
    return v.astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 반화소 정렬 2배 축소/확대
# ---------------------------------------------------------------------------------------------------------------
def _down_axis(v, axis, mode):
    n = v.shape[axis]
    m = n // 2
    p = _pad_axis(v, axis, 1, 2, mode)
    nd = v.ndim
    p0 = p[_sl(axis, nd, slice(0, 2 * m, 2))]
    p1 = p[_sl(axis, nd, slice(1, 2 * m, 2))]
    p2 = p[_sl(axis, nd, slice(2, 2 * m + 1, 2))]
    p3 = p[_sl(axis, nd, slice(3, 2 * m + 2, 2))]
    return (p0 + 3.0 * p1 + 3.0 * p2 + p3) / 8.0


def pyr_down(img, border="reflect") -> np.ndarray:
    a = _check(img)
    mode = _pad_mode(border)
    if a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError(f"pyr_down: 크기가 너무 작다 (shape {a.shape})")
    v = a.astype(np.float64)
    v = _down_axis(v, 0, mode)
    v = _down_axis(v, 1, mode)
    return v.astype(a.dtype)


def _up_axis(v, axis):
    n = v.shape[axis]
    nd = v.ndim
    p = _pad_axis(v, axis, 1, 1, "edge")
    cur = p[_sl(axis, nd, slice(1, n + 1))]
    even = 0.25 * p[_sl(axis, nd, slice(0, n))] + 0.75 * cur
    odd = 0.75 * cur + 0.25 * p[_sl(axis, nd, slice(2, n + 2))]
    shape = list(v.shape)
    shape[axis] = 2 * n
    out = np.empty(shape, dtype=np.float64)
    out[_sl(axis, nd, slice(0, None, 2))] = even
    out[_sl(axis, nd, slice(1, None, 2))] = odd
    return out


def _pyr_up64(v, out_shape=None):
    h, w = v.shape[0], v.shape[1]
    th, tw = (2 * h, 2 * w) if out_shape is None else out_shape
    out = _up_axis(_up_axis(v, 0), 1)
    if th == 2 * h + 1:
        out = np.concatenate([out, out[-1:]], axis=0)
    if tw == 2 * w + 1:
        out = np.concatenate([out, out[:, -1:]], axis=1)
    return out


def _out_shape_check(out_shape, h, w):
    try:
        th, tw = (int(out_shape[0]), int(out_shape[1]))
        if len(out_shape) != 2:
            raise ValueError
    except (TypeError, ValueError, IndexError) as e:
        raise ValueError(f"out_shape={out_shape!r}: (H, W) 정수 쌍이어야 한다") from e
    if th not in (2 * h, 2 * h + 1) or tw not in (2 * w, 2 * w + 1):
        raise ValueError(f"out_shape={out_shape}: 각 축 2h 또는 2h+1 이어야 한다 (입력 {h}x{w})")
    return th, tw


def pyr_up(img, out_shape=None) -> np.ndarray:
    a = _check(img)
    shp = None if out_shape is None else _out_shape_check(out_shape, a.shape[0], a.shape[1])
    return _pyr_up64(a.astype(np.float64), shp).astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 피라미드
# ---------------------------------------------------------------------------------------------------------------
def _levels(levels):
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer)):
        raise TypeError(f"levels={levels!r}: 정수여야 한다")
    if levels < 0:
        raise ValueError(f"levels={levels}: 0 이상이어야 한다")
    return int(levels)


def gaussian_pyramid(img, levels) -> list:
    a = _check(img)
    levels = _levels(levels)
    out = [a.copy()]
    for _ in range(levels):
        out.append(pyr_down(out[-1]))
    return out


def laplacian_pyramid(img, levels) -> list:
    g = gaussian_pyramid(img, levels)
    out = []
    for lvl in range(len(g) - 1):
        up = _pyr_up64(g[lvl + 1].astype(np.float64), g[lvl].shape[:2])
        out.append((g[lvl].astype(np.float64) - up).astype(g[lvl].dtype))
    out.append(g[-1])
    return out


def laplacian_reconstruct(pyr) -> np.ndarray:
    if not isinstance(pyr, (list, tuple)) or len(pyr) == 0:
        raise ValueError("pyr: 비어 있지 않은 리스트여야 한다")
    bands = [_check(p, f"pyr[{i}]") for i, p in enumerate(pyr)]
    cur = bands[-1].astype(np.float64)
    for b in reversed(bands[:-1]):
        shp = b.shape[:2]
        if shp[0] not in (2 * cur.shape[0], 2 * cur.shape[0] + 1) or shp[1] not in (2 * cur.shape[1], 2 * cur.shape[1] + 1):
            raise ValueError("pyr: 단계 크기가 맞지 않는다")
        cur = b.astype(np.float64) + _pyr_up64(cur, shp)
    return cur.astype(bands[0].dtype)


def laplacian_low_swap(img, low, levels) -> np.ndarray:
    a = _check(img)
    levels = _levels(levels)
    if levels < 1:
        raise ValueError("levels >= 1 이어야 한다")
    low = _check(low, "low")
    lp = laplacian_pyramid(a, levels)
    if low.shape != lp[-1].shape:
        raise ValueError(f"low shape {low.shape} != 잔차 shape {lp[-1].shape}")
    lp[-1] = low.astype(a.dtype)
    return laplacian_reconstruct(lp)


# ---------------------------------------------------------------------------------------------------------------
# resize (Pillow 계수 산식)
# ---------------------------------------------------------------------------------------------------------------
_METHODS = ("nearest", "area", "bilinear", "bicubic", "lanczos")


def _f_box(x):
    return np.where((x > -0.5) & (x <= 0.5), 1.0, 0.0)


def _f_bilinear(x):
    x = np.abs(x)
    return np.where(x < 1.0, 1.0 - x, 0.0)


def _f_bicubic(x, a=-0.5):
    x = np.abs(x)
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return np.where(x < 1.0, near, np.where(x < 2.0, far, 0.0))


def _f_lanczos(x):
    return np.where((x >= -3.0) & (x < 3.0), np.sinc(x) * np.sinc(x / 3.0), 0.0)


_FILTERS = {"area": (_f_box, 0.5), "bilinear": (_f_bilinear, 1.0), "bicubic": (_f_bicubic, 2.0),
            "lanczos": (_f_lanczos, 3.0)}


def _coeff_matrix(in_size, out_size, method):
    """(out_size, in_size) 희소 행렬. Pillow precompute_coeffs 와 같은 산식."""
    scale = in_size / out_size
    if method == "nearest":
        idx = np.floor((np.arange(out_size) + 0.5) * scale).astype(np.int64)
        idx = np.clip(idx, 0, in_size - 1)
        return _sp.csr_matrix((np.ones(out_size), (np.arange(out_size), idx)), shape=(out_size, in_size))
    filt, support0 = _FILTERS[method]
    filterscale = max(scale, 1.0)
    support = support0 * filterscale
    ss = 1.0 / filterscale
    rows, cols, vals = [], [], []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        if xmax <= xmin:
            xmin = min(max(int(center), 0), in_size - 1)
            xmax = xmin + 1
        pos = np.arange(xmin, xmax, dtype=np.float64)
        w = filt((pos - center + 0.5) * ss)
        tot = w.sum()
        if tot != 0.0:
            w = w / tot
        rows.append(np.full(xmax - xmin, xx))
        cols.append(pos.astype(np.int64))
        vals.append(w)
    return _sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(out_size, in_size))


def resize(img, out_hw, method="bilinear") -> np.ndarray:
    a = _check(img)
    try:
        if len(out_hw) != 2:
            raise ValueError
        oh, ow = out_hw
    except (TypeError, ValueError) as e:
        raise ValueError(f"out_hw={out_hw!r}: (H, W) 쌍이어야 한다") from e
    for v in (oh, ow):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"out_hw={out_hw!r}: 1 이상의 정수여야 한다")
    oh, ow = int(oh), int(ow)
    if method not in _METHODS:
        raise ValueError(f"method={method!r}: {_METHODS} 중 하나여야 한다")
    h, w = a.shape[:2]
    if (h, w) == (oh, ow):
        return a.copy()
    v = a.astype(np.float64)
    if ow != w:
        Wx = _coeff_matrix(w, ow, method)
        v = np.moveaxis(np.asarray(Wx @ np.moveaxis(v, 1, 0).reshape(w, -1)).reshape((ow,) + v.shape[:1] + v.shape[2:]), 0, 1)
    if oh != h:
        Wy = _coeff_matrix(h, oh, method)
        v = np.asarray(Wy @ v.reshape(h, -1)).reshape((oh,) + v.shape[1:])
    return np.ascontiguousarray(v).astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# sample_bilinear
# ---------------------------------------------------------------------------------------------------------------
def sample_bilinear(img, uv, border="edge") -> np.ndarray:
    a = _check(img)
    if border not in ("edge", "zero"):
        raise ValueError(f"border={border!r}: 'edge' 또는 'zero' 여야 한다")
    uv = np.asarray(uv)
    if uv.dtype.kind not in "biuf":
        raise TypeError(f"uv: 수치 배열이어야 한다 (dtype {uv.dtype})")
    uv = uv.astype(np.float64)
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError(f"uv: 마지막 축이 2 여야 한다 (shape {uv.shape})")
    H, W = a.shape[:2]
    u, v = uv[..., 0], uv[..., 1]
    ok = np.isfinite(u) & np.isfinite(v)
    u = np.where(ok, u, 0.0)
    v = np.where(ok, v, 0.0)
    if border == "edge":
        u, v = np.clip(u, 0.0, W - 1.0), np.clip(v, 0.0, H - 1.0)
    else:
        u, v = np.clip(u, -2.0, W + 1.0), np.clip(v, -2.0, H + 1.0)
    x0f, y0f = np.floor(u), np.floor(v)
    fx, fy = u - x0f, v - y0f
    x0, y0 = x0f.astype(np.int64), y0f.astype(np.int64)
    src = a.astype(np.float64)

    def tap(yy, xx):
        inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        val = src[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)]
        if border == "zero":
            val = np.where(inside.reshape(inside.shape + (1,) * (val.ndim - inside.ndim)), val, 0.0)
        return val

    def wt(x):
        return x.reshape(x.shape + (1,) * (src.ndim - 2))

    out = (wt((1.0 - fx) * (1.0 - fy)) * tap(y0, x0) + wt(fx * (1.0 - fy)) * tap(y0, x0 + 1)
           + wt((1.0 - fx) * fy) * tap(y0 + 1, x0) + wt(fx * fy) * tap(y0 + 1, x0 + 1))
    out = np.where(wt(ok), out, np.nan)
    return out.astype(a.dtype)
