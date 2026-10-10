"""RDR-04 EWA 비등방 필터 / C_0 정합 렌더 필터 (카디널 B-스플라인 사전필터).

모든 프레임이 같은 함수로 텍스처를 읽는다. 보간은 카디널 B-스플라인(사전필터로 노드 보간),
축소는 EWA 비등방 가우시안이며 J 가 확대/항등이면 블러가 0 이다.

좌표 규약: uv=(x,y), 화소 중심이 정수, 영상 영역 [-0.5, W-0.5) x [-0.5, H-0.5), 경계는 거울(whole-sample).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as _ndi

__all__ = [
    "DEGREES", "MODES", "EwaOpts", "PrefilteredTexture", "WarpResult", "bspline_basis", "interp_mtf",
    "prefilter_texture", "bspline_sample", "ewa_covariance", "ewa_sample", "homography_uv_jacobian",
    "warp_homography",
]

DEGREES = (1, 3, 5)
MODES = ("bilinear", "bspline", "ewa")

_TAP_RANGE = 3.5          # 가우시안 절단 반경 (σ 배)
_TAP_MARGIN = 0.8         # 구적 간격 h = 1/(0.5 + margin/σ): 복제 스펙트럼 이득 < 1e-5
_SIGMA_ZERO = 1e-4        # 이보다 작은 σ 는 0 으로 본다
_CHUNK_POINTS = 120_000


# --------------------------------------------------------------------------------------------- 옵션
def _is_real(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


@dataclass(frozen=True, kw_only=True)
class EwaOpts:
    aa_scale: float = 0.5
    max_aniso: float = 16.0
    max_sigma_px: float = 8.0
    recon_sigma_px: float = 0.0

    def __post_init__(self):
        for name in ("aa_scale", "max_aniso", "max_sigma_px", "recon_sigma_px"):
            v = getattr(self, name)
            if not _is_real(v) or not math.isfinite(float(v)):
                raise ValueError(f"EwaOpts.{name} must be a finite real number, got {v!r}")
        if self.aa_scale < 0:
            raise ValueError("aa_scale must be >= 0")
        if self.max_aniso < 1:
            raise ValueError("max_aniso must be >= 1")
        if not self.max_sigma_px > 0:
            raise ValueError("max_sigma_px must be > 0")
        if self.recon_sigma_px < 0:
            raise ValueError("recon_sigma_px must be >= 0")


def _check_degree(degree):
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or int(degree) not in DEGREES:
        raise ValueError(f"degree must be one of {DEGREES}, got {degree!r}")
    return int(degree)


# --------------------------------------------------------------------------------------------- 기저 / MTF
def bspline_basis(x, degree=3):
    """중심 B-스플라인 β^n(x), 지지 |x| < (n+1)/2."""
    n = _check_degree(degree)
    x = np.asarray(x, dtype=np.float64)
    half = (n + 1) / 2.0
    xc = np.clip(np.abs(x), 0.0, half)          # 우함수: |x| 로 계산해 대칭을 비트 단위로 보장
    s = xc + half
    acc = np.zeros_like(xc)
    for k in range(n + 2):
        c = (-1.0) ** k * math.comb(n + 1, k)
        acc = acc + c * np.maximum(s - k, 0.0) ** n
    out = np.maximum(acc / math.factorial(n), 0.0)
    return np.where(np.abs(x) >= half, 0.0, out)


def interp_mtf(freq, degree=3, *, prefiltered=True):
    """보간 필터의 연속 주파수 응답 (freq 는 cy/px)."""
    n = _check_degree(degree)
    f = np.asarray(freq, dtype=np.float64)
    num = np.sinc(f) ** (n + 1)
    if not prefiltered or n == 1:
        return np.asarray(num, dtype=np.float64)
    c1 = np.cos(2.0 * np.pi * f)
    if n == 3:
        b = (4.0 + 2.0 * c1) / 6.0
    else:
        b = (66.0 + 52.0 * c1 + 2.0 * np.cos(4.0 * np.pi * f)) / 120.0
    return np.asarray(num / b, dtype=np.float64)


# --------------------------------------------------------------------------------------------- 입력 검증
def _as_numeric(a, what):
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{what} must be numeric, got dtype {a.dtype}")
    return a


def _check_image(img):
    a = np.asarray(img)
    if a.dtype.kind != "f":
        raise TypeError(f"image dtype must be floating, got {a.dtype}")
    if a.ndim not in (2, 3):
        raise ValueError("image must be (H,W) or (H,W,C)")
    if a.size == 0:
        raise ValueError("image is empty")
    if not np.isfinite(a).all():
        raise ValueError("image has non-finite values")
    return a


def _check_uv(uv):
    a = _as_numeric(uv, "uv").astype(np.float64, copy=False)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise ValueError("uv must have last axis of size 2")
    if not np.isfinite(a).all():
        raise ValueError("uv has non-finite values")
    return a


def _check_jac(jac):
    a = _as_numeric(jac, "jac").astype(np.float64, copy=False)
    if a.ndim < 2 or a.shape[-2:] != (2, 2):
        raise ValueError("jac must have trailing shape (2,2)")
    if not np.isfinite(a).all():
        raise ValueError("jac has non-finite values")
    return a


def _check_opts(opts):
    if opts is None:
        return EwaOpts()
    if not isinstance(opts, EwaOpts):
        raise TypeError("opts must be an EwaOpts")
    return opts


def _check_tex(tex):
    if not isinstance(tex, PrefilteredTexture):
        raise TypeError("tex must be a PrefilteredTexture")
    return tex


# --------------------------------------------------------------------------------------------- 사전필터 / 표본
@dataclass(frozen=True, eq=False)
class PrefilteredTexture:
    coef: np.ndarray
    degree: int
    shape: tuple


def prefilter_texture(img, *, degree=3) -> PrefilteredTexture:
    n = _check_degree(degree)
    a = _check_image(img)
    coef = np.array(a, dtype=np.float64, copy=True)
    if n > 1:
        for ax in (0, 1):
            if coef.shape[ax] > 1:
                coef = _ndi.spline_filter1d(coef, order=n, axis=ax, mode="mirror", output=np.float64)
    coef.flags.writeable = False
    return PrefilteredTexture(coef=coef, degree=n, shape=tuple(a.shape))


def _mirror_index(k, n):
    if n == 1:
        return np.zeros_like(k)
    p = 2 * (n - 1)
    m = np.mod(k, p)
    return np.where(m >= n, p - m, m)


def _eval_points(tex, pts):
    """pts (M,2) float64 -> (M, C) (2D 영상은 C=1)."""
    coef = tex.coef
    d = tex.degree
    H, W = coef.shape[:2]
    c3 = coef if coef.ndim == 3 else coef[..., None]
    x = np.clip(pts[:, 0], -1e12, 1e12)
    y = np.clip(pts[:, 1], -1e12, 1e12)
    off = np.arange(d + 1) - (d - 1) // 2
    kx = np.floor(x).astype(np.int64)[:, None] + off[None, :]
    ky = np.floor(y).astype(np.int64)[:, None] + off[None, :]
    wx = bspline_basis(x[:, None] - kx, d)
    wy = bspline_basis(y[:, None] - ky, d)
    ix = _mirror_index(kx, W)
    iy = _mirror_index(ky, H)
    vals = c3[iy[:, :, None], ix[:, None, :]]            # (M, D, D, C)
    return np.einsum("mi,mj,mijc->mc", wy, wx, vals, optimize=False)


def _eval_chunked(tex, pts):
    c = 1 if tex.coef.ndim == 2 else tex.coef.shape[2]
    out = np.empty((pts.shape[0], c), np.float64)
    for s in range(0, pts.shape[0], _CHUNK_POINTS):
        out[s:s + _CHUNK_POINTS] = _eval_points(tex, pts[s:s + _CHUNK_POINTS])
    return out


def _finish(tex, flat, lead):
    if tex.coef.ndim == 2:
        return flat.reshape(lead)
    return flat.reshape(lead + (tex.coef.shape[2],))


def bspline_sample(tex, uv):
    tex = _check_tex(tex)
    uv = _check_uv(uv)
    lead = uv.shape[:-1]
    flat = _eval_chunked(tex, uv.reshape(-1, 2))
    return _finish(tex, flat, lead)


# --------------------------------------------------------------------------------------------- EWA
def _cov_parts(jac, opts):
    """jac (...,2,2) -> U (...,2,2), var (...,2) (두 주축의 분산, 원본 px²)."""
    shp = jac.shape[:-2]
    J = jac.reshape(-1, 2, 2)
    if J.shape[0] == 0:
        return np.zeros(shp + (2, 2)), np.zeros(shp + (2,))
    U, s, _ = np.linalg.svd(J)
    s1 = s[:, 0]
    s2 = np.maximum(s[:, 1], s1 / opts.max_aniso)
    sp = np.stack([s1, s2], axis=-1)
    sig = np.minimum(opts.aa_scale * np.sqrt(np.maximum(sp * sp - 1.0, 0.0)), opts.max_sigma_px)
    var = sig * sig + opts.recon_sigma_px ** 2
    return U.reshape(shp + (2, 2)), var.reshape(shp + (2,))


def ewa_covariance(jac, *, opts=None):
    opts = _check_opts(opts)
    jac = _check_jac(jac)
    U, var = _cov_parts(jac, opts)
    return np.einsum("...ik,...k,...jk->...ij", U, var, U)


def _axis_taps(var):
    """분산 -> (탭 반경 n, 간격 h). σ≈0 이면 n=0."""
    sig = np.sqrt(var)
    on = sig > _SIGMA_ZERO
    safe = np.where(on, sig, 1.0)
    h = np.minimum(1.0, 1.0 / (0.5 + _TAP_MARGIN / safe))
    n = np.where(on, np.ceil(_TAP_RANGE * safe / h), 0).astype(np.int64)
    return n, np.where(on, h, 0.0)


def ewa_sample(tex, uv, jac, *, opts=None):
    tex = _check_tex(tex)
    opts = _check_opts(opts)
    uv = _check_uv(uv)
    jac = _check_jac(jac)
    lead = uv.shape[:-1]
    try:
        jb = np.broadcast_to(jac, lead + (2, 2))
    except ValueError as e:
        raise ValueError(f"jac shape {jac.shape} cannot broadcast to {lead + (2, 2)}") from e
    P = int(np.prod(lead, dtype=np.int64)) if lead else 1
    uvf = uv.reshape(P, 2)
    U, var = _cov_parts(jb.reshape(P, 2, 2), opts)
    n1, h1 = _axis_taps(var[:, 0])
    n2, h2 = _axis_taps(var[:, 1])
    c = 1 if tex.coef.ndim == 2 else tex.coef.shape[2]
    out = np.empty((P, c), np.float64)
    key = n1 * 100_000 + n2
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [P]])
    for s, e in zip(starts, ends):
        idx = order[s:e]
        a1, a2 = int(n1[idx[0]]), int(n2[idx[0]])
        if a1 == 0 and a2 == 0:
            out[idx] = _eval_chunked(tex, uvf[idx])
            continue
        t1, t2 = 2 * a1 + 1, 2 * a2 + 1
        step = max(1, _CHUNK_POINTS // (t1 * t2))
        i1 = np.arange(-a1, a1 + 1, dtype=np.float64)
        i2 = np.arange(-a2, a2 + 1, dtype=np.float64)
        for cs in range(0, idx.size, step):
            ii = idx[cs:cs + step]
            m = ii.size
            A = i1[None, :] * h1[ii][:, None]                         # (m,t1) 주축 1 좌표
            B = i2[None, :] * h2[ii][:, None]
            v1 = np.maximum(var[ii, 0], 1e-300)[:, None]
            v2 = np.maximum(var[ii, 1], 1e-300)[:, None]
            w1 = np.exp(-0.5 * A * A / v1) if a1 else np.ones((m, 1))
            w2 = np.exp(-0.5 * B * B / v2) if a2 else np.ones((m, 1))
            w1 /= w1.sum(axis=1, keepdims=True)
            w2 /= w2.sum(axis=1, keepdims=True)
            e1 = U[ii, :, 0]                                           # (m,2)
            e2 = U[ii, :, 1]
            pts = (uvf[ii][:, None, None, :] + A[:, :, None, None] * e1[:, None, None, :]
                   + B[:, None, :, None] * e2[:, None, None, :])       # (m,t1,t2,2)
            vals = _eval_points(tex, pts.reshape(-1, 2)).reshape(m, t1, t2, c)
            out[ii] = np.einsum("mi,mj,mijc->mc", w1, w2, vals)
    return _finish(tex, out, lead)


# --------------------------------------------------------------------------------------------- 호모그래피
def _check_H(H):
    Ha = _as_numeric(H, "H").astype(np.float64, copy=False)
    if Ha.shape != (3, 3):
        raise ValueError("H must be 3x3")
    if not np.isfinite(Ha).all():
        raise ValueError("H has non-finite values")
    return Ha


def homography_uv_jacobian(H, xy):
    Ha = _check_H(H)
    xy = _check_uv(xy)
    x, y = xy[..., 0], xy[..., 1]
    a = Ha[0, 0] * x + Ha[0, 1] * y + Ha[0, 2]
    b = Ha[1, 0] * x + Ha[1, 1] * y + Ha[1, 2]
    c = Ha[2, 0] * x + Ha[2, 1] * y + Ha[2, 2]
    ok = c > 0
    cs = np.where(ok, c, 1.0)
    u = a / cs
    v = b / cs
    uv = np.stack([u, v], axis=-1)
    J = np.empty(xy.shape[:-1] + (2, 2), np.float64)
    for j in range(2):
        J[..., 0, j] = (Ha[0, j] - u * Ha[2, j]) / cs
        J[..., 1, j] = (Ha[1, j] - v * Ha[2, j]) / cs
    uv = np.where(ok[..., None], uv, np.nan)
    J = np.where(ok[..., None, None], J, np.nan)
    return uv, J


@dataclass(frozen=True, eq=False)
class WarpResult:
    image: np.ndarray
    valid: np.ndarray


def _check_out_hw(out_hw):
    try:
        ok = len(out_hw) == 2
    except TypeError:
        ok = False
    if not ok:
        raise ValueError("out_hw must be (h, w)")
    for v in out_hw:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0:
            raise ValueError("out_hw must be two positive integers")
    return int(out_hw[0]), int(out_hw[1])


def _bilinear_edge(img, uv):
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0], 0.0, W - 1.0)
    y = np.clip(uv[:, 1], 0.0, H - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), max(W - 2, 0))
    y0 = np.minimum(np.floor(y).astype(np.int64), max(H - 2, 0))
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = x - x0
    fy = y - y0
    im = img.astype(np.float64)
    if im.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((1 - fx) * (1 - fy) * im[y0, x0] + fx * (1 - fy) * im[y0, x1]
            + (1 - fx) * fy * im[y1, x0] + fx * fy * im[y1, x1])


def warp_homography(img, H, out_hw, *, mode="ewa", opts=None, fill=0.0) -> WarpResult:
    a = _check_image(img)
    Ha = _check_H(H)
    h, w = _check_out_hw(out_hw)
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    opts = _check_opts(opts)
    if not _is_real(fill) or not math.isfinite(float(fill)):
        raise ValueError("fill must be a finite real number")
    if abs(np.linalg.det(Ha)) <= 1e-12 * np.linalg.norm(Ha) ** 3:
        raise ValueError("H is singular")

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    uv, J = homography_uv_jacobian(Ha, np.stack([xx, yy], axis=-1))
    okc = np.isfinite(uv[..., 0])
    uvz = np.where(okc[..., None], uv, 0.0)
    H0, W0 = a.shape[:2]
    valid = (okc & (uvz[..., 0] >= -0.5) & (uvz[..., 0] < W0 - 0.5)
             & (uvz[..., 1] >= -0.5) & (uvz[..., 1] < H0 - 0.5))
    out = np.full((h, w) + a.shape[2:], np.float32(fill), dtype=np.float32)
    if valid.any():
        puv = uv[valid]
        if mode == "bilinear":
            vals = _bilinear_edge(a, puv)
        else:
            tex = prefilter_texture(a, degree=3)
            if mode == "bspline":
                vals = bspline_sample(tex, puv)
            else:
                vals = ewa_sample(tex, puv, J[valid], opts=opts)
        out[valid] = vals.astype(np.float32)
    return WarpResult(image=out, valid=valid)
