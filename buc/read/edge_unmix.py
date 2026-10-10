"""RD-18 R13 경계 분해: 깊이 불연속 trimap · Levin 닫힌형 매팅 · 전경/배경 색 분리.

깊이 불연속(상대 점프 > rel_jump)의 ±band_px 띠와 가는 구조 마스크(thin)로 trimap을 만들고,
Levin 닫힌형 매팅으로 알파를, 평활 최소제곱으로 전경색 F, 배경색 B를 푼다. I = aF + (1-a)B.
매팅은 I(와 깊이)만의 함수이며 화면 공간 합성이 아니다.

numpy · scipy 만 쓴다. torch 금지, import 부작용·전역 RNG 변경 없음, 입력 불변, 결정론.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi

__all__ = [
    "BG", "UNKNOWN", "FG", "EdgeUnmixError", "UnmixConfig", "UnmixResult",
    "depth_edges", "upsample_depth", "make_trimap", "matting_laplacian",
    "solve_alpha", "solve_fg_bg", "edge_unmix",
]

BG = 0
UNKNOWN = 128
FG = 255

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_CHUNK = 16384  # 라플라시안 창 처리 단위(임시 메모리 제한)


class EdgeUnmixError(ValueError):
    """경계 분해를 계산할 수 없는 입력(알려진 화소 없음 등)."""


def _pos_finite(name: str, v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if not (math.isfinite(f) and f > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    return f


def _int_ge1(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{name} must be an int >= 1, got {v!r}")
    if int(v) < 1:
        raise ValueError(f"{name} must be an int >= 1, got {v!r}")
    return int(v)


@dataclass(frozen=True, kw_only=True)
class UnmixConfig:
    rel_jump: float = 0.10
    band_px: int = 3
    eps: float = 1e-7
    win_r: int = 1
    ridge: float = 1e-9
    gamma: float = 0.1
    delta: float = 1e-6
    clip_tau: float = 0.2
    recon_tau: float = 0.02

    def __post_init__(self) -> None:
        _pos_finite("rel_jump", self.rel_jump)
        _int_ge1("band_px", self.band_px)
        _pos_finite("eps", self.eps)
        _int_ge1("win_r", self.win_r)
        _pos_finite("ridge", self.ridge)
        _pos_finite("gamma", self.gamma)
        _pos_finite("delta", self.delta)
        for name in ("clip_tau", "recon_tau"):
            v = getattr(self, name)
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a number, got {v!r}")
            if not (math.isfinite(f) and f >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {v!r}")


# --------------------------------------------------------------------------- 검증 도구

def _real_dtype_check(name: str, a: np.ndarray) -> None:
    if a.dtype.kind not in "fiu":
        raise TypeError(f"{name} must have a real numeric dtype, got {a.dtype}")


def _check_depth(depth: Any, name: str = "depth") -> np.ndarray:
    z = np.asarray(depth)
    _real_dtype_check(name, z)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError(f"{name} must be 2-D (H, W), got shape {z.shape}")
    return z


def _check_image(image: Any, name: str = "image") -> np.ndarray:
    a = np.asarray(image)
    _real_dtype_check(name, a)
    if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{name} must have shape (H, W, 3), got {a.shape}")
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def _check_trimap(trimap: Any, shape: Tuple[int, int]) -> np.ndarray:
    t = np.asarray(trimap)
    if t.dtype.kind not in "iu":
        raise TypeError(f"trimap must have an integer dtype, got {t.dtype}")
    if t.shape != tuple(shape):
        raise ValueError(f"trimap shape {t.shape} != image shape {tuple(shape)}")
    if not np.isin(t, (BG, UNKNOWN, FG)).all():
        raise ValueError("trimap codes must be in {0, 128, 255}")
    return t


def _check_bool_mask(mask: Any, shape: Tuple[int, int], name: str) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        raise TypeError(f"{name} must be a bool array, got {m.dtype}")
    if m.shape != tuple(shape):
        raise ValueError(f"{name} shape {m.shape} != {tuple(shape)}")
    return m


# --------------------------------------------------------------------------- 깊이 에지

def depth_edges(depth: np.ndarray, *, rel_jump: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """깊이 불연속 화소 (near, far). 유효(유한·>0) 4-이웃 쌍의 상대 점프가 문턱을 넘으면 작은 z 쪽이 near."""
    z = _check_depth(depth).astype(np.float64)
    rel = _pos_finite("rel_jump", rel_jump)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(z) & (z > 0.0)
    zz = np.where(valid, z, 1.0)
    near = np.zeros(z.shape, bool)
    far = np.zeros(z.shape, bool)
    for sl_p, sl_q in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        zp, zq = zz[sl_p], zz[sl_q]
        ok = valid[sl_p] & valid[sl_q] & (np.abs(zp - zq) > rel * np.minimum(zp, zq))
        p_near = ok & (zp < zq)
        q_near = ok & (zq < zp)
        near[sl_p] |= p_near
        far[sl_q] |= p_near
        near[sl_q] |= q_near
        far[sl_p] |= q_near
    return near, far


def upsample_depth(depth: np.ndarray, hw: Tuple[int, int]) -> Tuple[np.ndarray, int]:
    """저해상도 깊이 (h,w)를 (H,W)로 최근접 확대. ceil(H/s)==h, ceil(W/s)==w 인 가장 작은 s."""
    z = _check_depth(depth).astype(np.float64)
    Hh, Ww = int(hw[0]), int(hw[1])
    if Hh < 1 or Ww < 1:
        raise ValueError(f"invalid target size {hw}")
    h, w = z.shape
    for s in range(1, max(Hh, Ww) + 1):
        if -(-Hh // s) == h and -(-Ww // s) == w:
            up = np.repeat(np.repeat(z, s, axis=0), s, axis=1)[:Hh, :Ww]
            return np.ascontiguousarray(up), s
    raise ValueError(f"depth shape {z.shape} is not a ceil-division of target {(Hh, Ww)} by an integer stride")


# --------------------------------------------------------------------------- trimap

def make_trimap(depth: np.ndarray, *, rel_jump: float = 0.10, band_px: int = 3,
                thin: Optional[np.ndarray] = None) -> np.ndarray:
    """깊이 불연속 띠(+thin)를 UNKNOWN, 나머지를 최근접 에지 화소의 near/far에 따라 FG/BG로."""
    z = _check_depth(depth)
    band_px = _int_ge1("band_px", band_px)
    near, far = depth_edges(z, rel_jump=rel_jump)
    shape = z.shape
    thin_m = None if thin is None else _check_bool_mask(thin, shape, "thin")
    E = near | far
    tri = np.full(shape, FG, dtype=np.uint8)
    if E.any():
        _, idx = ndi.distance_transform_edt(~E, return_indices=True)
        nearest_is_near = near[idx[0], idx[1]]
        tri = np.where(nearest_is_near, FG, BG).astype(np.uint8)
        side = 2 * band_px - 1
        U = ndi.binary_dilation(E, structure=np.ones((side, side), bool))
    else:
        U = np.zeros(shape, bool)
    if thin_m is not None:
        U = U | thin_m
    tri[U] = UNKNOWN
    return tri


# --------------------------------------------------------------------------- 매팅 라플라시안

def matting_laplacian(image: np.ndarray, *, eps: float = 1e-7, win_r: int = 1,
                      centers: Optional[np.ndarray] = None) -> sp.csr_matrix:
    """Levin 닫힌형 매팅 라플라시안 (HW,HW) csr float64, 화소 번호는 행 우선."""
    img = _check_image(image)
    eps = _pos_finite("eps", eps)
    r = _int_ge1("win_r", win_r)
    Hh, Ww = img.shape[:2]
    N = Hh * Ww
    cm = None if centers is None else _check_bool_mask(centers, (Hh, Ww), "centers")
    if Hh < 2 * r + 1 or Ww < 2 * r + 1:
        return sp.csr_matrix((N, N), dtype=np.float64)
    ok = np.zeros((Hh, Ww), bool)
    ok[r:Hh - r, r:Ww - r] = True
    if cm is not None:
        ok &= cm
    cy, cx = np.nonzero(ok)  # 행 우선 순서(결정론)
    K = cy.size
    if K == 0:
        return sp.csr_matrix((N, N), dtype=np.float64)

    n = (2 * r + 1) ** 2
    off = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(off, off, indexing="ij")
    doff = (dy * Ww + dx).ravel()  # (n,)
    flat = img.reshape(N, 3)
    eye_n = np.eye(n)
    eye3 = np.eye(3)
    idx_t = np.int32 if N < 2 ** 31 - 1 else np.int64

    rows_l, cols_l, vals_l = [], [], []
    for s in range(0, K, _CHUNK):
        c = (cy[s:s + _CHUNK] * Ww + cx[s:s + _CHUNK])
        ids = c[:, None] + doff[None, :]  # (k, n)
        X = flat[ids]  # (k, n, 3)
        Xc = X - X.mean(axis=1, keepdims=True)
        S = np.matmul(Xc.transpose(0, 2, 1), Xc) / n + (eps / n) * eye3
        Si = np.linalg.inv(S)
        Si = 0.5 * (Si + Si.transpose(0, 2, 1))
        G = np.matmul(np.matmul(Xc, Si), Xc.transpose(0, 2, 1))  # (k, n, n)
        G = 0.5 * (G + G.transpose(0, 2, 1))
        V = eye_n[None, :, :] - (1.0 + G) / n
        k = ids.shape[0]
        rows_l.append(np.broadcast_to(ids[:, :, None], (k, n, n)).reshape(-1).astype(idx_t))
        cols_l.append(np.broadcast_to(ids[:, None, :], (k, n, n)).reshape(-1).astype(idx_t))
        vals_l.append(V.reshape(-1))
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    L.sum_duplicates()
    return L


# --------------------------------------------------------------------------- 알파 풀이

def solve_alpha(image: np.ndarray, trimap: np.ndarray, *, eps: float = 1e-7, win_r: int = 1,
                ridge: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """하드 제약 Levin 매팅. 반환 (alpha(clip 0..1), alpha_raw), float64 (H,W)."""
    img = _check_image(image)
    Hh, Ww = img.shape[:2]
    tri = _check_trimap(trimap, (Hh, Ww))
    eps = _pos_finite("eps", eps)
    r = _int_ge1("win_r", win_r)
    ridge = _pos_finite("ridge", ridge)
    U = tri == UNKNOWN
    K = ~U
    if not K.any():
        raise EdgeUnmixError("trimap has no known pixels")
    b = (tri == FG).astype(np.float64)
    if not U.any():
        return b.copy(), b.copy()

    side = 2 * r + 1
    centers = ndi.binary_dilation(U, structure=np.ones((side, side), bool))
    L = matting_laplacian(img, eps=eps, win_r=r, centers=centers)
    u_idx = np.flatnonzero(U.ravel())
    bvec = b.ravel().copy()
    bvec[u_idx] = 0.0
    Lr = L[u_idx]  # (nu, N)
    Luu = Lr[:, u_idx] + ridge * sp.identity(u_idx.size, format="csr")
    rhs = -(Lr @ bvec)
    sol = spla.spsolve(Luu.tocsc(), rhs)
    raw = b.ravel().copy()
    raw[u_idx] = np.asarray(sol, dtype=np.float64).reshape(-1)
    raw = raw.reshape(Hh, Ww)
    return np.clip(raw, 0.0, 1.0), raw


# --------------------------------------------------------------------------- 전경/배경 색 분리

def _smooth_rows(P: np.ndarray, Q: np.ndarray, idx: np.ndarray, const: np.ndarray,
                 row0: int, s: float):
    """sqrt(gamma)*(X_p - X_q) 행: (rows, cols, vals, y(len,3))."""
    m = P.size
    r = row0 + np.arange(m)
    ip, iq = idx[P], idx[Q]
    mp, mq = ip >= 0, iq >= 0
    rows = np.concatenate([r[mp], r[mq]])
    cols = np.concatenate([ip[mp], iq[mq]])
    vals = np.concatenate([np.full(mp.sum(), s), np.full(mq.sum(), -s)])
    y = -s * (const[P] - const[Q])
    return rows, cols, vals, y


def solve_fg_bg(image: np.ndarray, alpha: np.ndarray, trimap: np.ndarray, *,
                gamma: float = 0.1, delta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """전경색 F, 배경색 B를 Omega = U ∪ N4(U) 위 2차 에너지의 유일 최소점으로 푼다. float64 (H,W,3)."""
    img = _check_image(image)
    Hh, Ww = img.shape[:2]
    a = np.asarray(alpha)
    _real_dtype_check("alpha", a)
    if a.shape != (Hh, Ww):
        raise ValueError(f"alpha shape {a.shape} != {(Hh, Ww)}")
    a = a.astype(np.float64)
    if not np.isfinite(a).all() or a.min() < 0.0 or a.max() > 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")
    tri = _check_trimap(trimap, (Hh, Ww))
    gamma = _pos_finite("gamma", gamma)
    delta = _pos_finite("delta", delta)

    N = Hh * Ww
    U = tri == UNKNOWN
    fgk = tri == FG
    bgk = tri == BG
    F = np.zeros((Hh, Ww, 3))
    B = np.zeros((Hh, Ww, 3))
    F[fgk] = img[fgk]
    B[bgk] = img[bgk]
    if not U.any():
        return F, B

    om = ndi.binary_dilation(U, structure=_CROSS)
    free_F = om & ~fgk
    free_B = om & ~bgk
    nF = int(free_F.sum())
    nB = int(free_B.sum())
    nv = nF + nB
    idxF = np.full(N, -1, dtype=np.int64)
    idxB = np.full(N, -1, dtype=np.int64)
    idxF[np.flatnonzero(free_F.ravel())] = np.arange(nF)
    idxB[np.flatnonzero(free_B.ravel())] = nF + np.arange(nB)
    flatI = img.reshape(N, 3)
    flatA = a.reshape(N)
    cF = np.where(fgk.reshape(N, 1), flatI, 0.0)  # F 고정 상수(고정 아니면 0)
    cB = np.where(bgk.reshape(N, 1), flatI, 0.0)

    rows_l, cols_l, vals_l, y_l = [], [], [], []
    row0 = 0

    # 데이터 항: aF + (1-a)B - I
    om_ids = np.flatnonzero(om.ravel())
    m = om_ids.size
    rr = row0 + np.arange(m)
    al = flatA[om_ids]
    iF, iB = idxF[om_ids], idxB[om_ids]
    mF, mB = iF >= 0, iB >= 0
    rows_l += [rr[mF], rr[mB]]
    cols_l += [iF[mF], iB[mB]]
    vals_l += [al[mF], 1.0 - al[mB]]
    y_l.append(flatI[om_ids] - al[:, None] * cF[om_ids] - (1.0 - al)[:, None] * cB[om_ids])
    row0 += m

    # 평활 항: Omega 안 4-이웃 쌍, F와 B 각각
    ids = np.arange(N).reshape(Hh, Ww)
    ph = om[:, :-1] & om[:, 1:]
    pv = om[:-1, :] & om[1:, :]
    P = np.concatenate([ids[:, :-1][ph], ids[:-1, :][pv]])
    Q = np.concatenate([ids[:, 1:][ph], ids[1:, :][pv]])
    sg = math.sqrt(gamma)
    for idx, const in ((idxF, cF), (idxB, cB)):
        rw, cl, vl, yy = _smooth_rows(P, Q, idx, const, row0, sg)
        rows_l.append(rw); cols_l.append(cl); vals_l.append(vl); y_l.append(yy)
        row0 += P.size

    # 자유 변수 릿지: sqrt(delta)(v - I)
    sd = math.sqrt(delta)
    for idx in (idxF, idxB):
        pix = np.flatnonzero(idx >= 0)
        rr = row0 + np.arange(pix.size)
        rows_l.append(rr)
        cols_l.append(idx[pix])
        vals_l.append(np.full(pix.size, sd))
        y_l.append(sd * flatI[pix])
        row0 += pix.size

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    Y = np.concatenate(y_l, axis=0)
    M = sp.coo_matrix((vals, (rows, cols)), shape=(row0, nv)).tocsr()
    A = (M.T @ M).tocsc()
    rhs = M.T @ Y
    sol = spla.splu(A).solve(np.ascontiguousarray(rhs))
    Ff = F.reshape(N, 3)
    Bf = B.reshape(N, 3)
    Ff[np.flatnonzero(free_F.ravel())] = sol[:nF]
    Bf[np.flatnonzero(free_B.ravel())] = sol[nF:]
    F = np.maximum(Ff.reshape(Hh, Ww, 3), 0.0)
    B = np.maximum(Bf.reshape(Hh, Ww, 3), 0.0)
    return F, B


# --------------------------------------------------------------------------- 종단

@dataclass(frozen=True, eq=False)
class UnmixResult:
    alpha: np.ndarray
    alpha_raw: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    trimap: np.ndarray
    band: np.ndarray
    edges: np.ndarray
    thin: np.ndarray
    depth_stride: int
    n_unknown: int
    band_frac: float
    clip_frac: float
    recon_err: float
    low_conf: bool
    rel_jump: float = 0.10
    band_px: int = 3
    eps: float = 1e-7
    win_r: int = 1

    def meta(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 요약(배열 없음)."""
        return {
            "n_unknown": int(self.n_unknown),
            "band_frac": float(self.band_frac),
            "clip_frac": float(self.clip_frac),
            "recon_err": float(self.recon_err),
            "low_conf": bool(self.low_conf),
            "depth_stride": int(self.depth_stride),
            "thin_frac": float(self.thin.mean()) if self.thin.size else 0.0,
            "rel_jump": float(self.rel_jump),
            "band_px": int(self.band_px),
            "eps": float(self.eps),
            "win_r": int(self.win_r),
        }


def edge_unmix(image_lin: np.ndarray, depth: np.ndarray, *, thin: Optional[np.ndarray] = None,
               cfg: Optional[UnmixConfig] = None) -> UnmixResult:
    """깊이 불연속 경계의 알파·전경색·배경색 분해."""
    cfg = UnmixConfig() if cfg is None else cfg
    img = _check_image(image_lin, "image_lin")
    if (img < 0.0).any():
        raise ValueError("image_lin must be >= 0")
    Hh, Ww = img.shape[:2]
    if min(Hh, Ww) < 2 * cfg.win_r + 1:
        raise ValueError(f"image too small for win_r={cfg.win_r}: {(Hh, Ww)}")
    z0 = _check_depth(depth)
    thin_m = None if thin is None else _check_bool_mask(thin, (Hh, Ww), "thin")

    z, stride = upsample_depth(z0, (Hh, Ww))
    tri = make_trimap(z, rel_jump=cfg.rel_jump, band_px=cfg.band_px, thin=thin_m)
    near, far = depth_edges(z, rel_jump=cfg.rel_jump)
    edges = near | far
    alpha, alpha_raw = solve_alpha(img, tri, eps=cfg.eps, win_r=cfg.win_r, ridge=cfg.ridge)
    fg, bg = solve_fg_bg(img, alpha, tri, gamma=cfg.gamma, delta=cfg.delta)

    band = tri == UNKNOWN
    n_unknown = int(band.sum())
    a32 = alpha.astype(np.float32)
    raw32 = alpha_raw.astype(np.float32)
    fg32 = fg.astype(np.float32)
    bg32 = bg.astype(np.float32)
    if n_unknown:
        raw_u = raw32.astype(np.float64)[band]
        clip_frac = float(((raw_u < -0.05) | (raw_u > 1.05)).mean())
        a64 = a32.astype(np.float64)[..., None]
        rec = a64 * fg32.astype(np.float64) + (1.0 - a64) * bg32.astype(np.float64)
        recon_err = float(np.abs(rec - img)[band].max())
    else:
        clip_frac = 0.0
        recon_err = 0.0
    low_conf = bool(clip_frac > cfg.clip_tau or recon_err > cfg.recon_tau)
    thin_out = np.zeros((Hh, Ww), bool) if thin_m is None else thin_m.copy()
    return UnmixResult(
        alpha=a32, alpha_raw=raw32, fg=fg32, bg=bg32, trimap=tri, band=band, edges=edges,
        thin=thin_out, depth_stride=int(stride), n_unknown=n_unknown,
        band_frac=float(band.mean()), clip_frac=clip_frac, recon_err=recon_err, low_conf=low_conf,
        rel_jump=float(cfg.rel_jump), band_px=int(cfg.band_px), eps=float(cfg.eps), win_r=int(cfg.win_r),
    )
