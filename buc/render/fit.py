"""RDR-01 피팅: COLMAP points3D 리더, 점/RGB-D 초기화, 미분 래스터 기반 Adam 피팅.

계약서: ops/harness/contracts/RDR-01.md. 래스터는 `buc.render.splat2d`(torch 정본 또는 gsplat).
세계 오른손·미터, 카메라 OpenCV, `c2w` 카메라->세계, 색은 선형 Rec.709.
같은 seed·장치에서 결과는 비트 동일하다(전역 RNG 를 건드리지 않고, 뷰 순열은 전용 Generator 를 쓴다).
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from buc.core import color, geom
from buc.render.splat2d import (Surfels, SplatOpts, SplatRender, _camera, _render_full)

__all__ = ["ColmapPoints", "read_colmap_points3d", "init_from_points", "init_from_rgbd", "FitView", "FitOpts",
           "FitResult", "fit", "psnr"]


# ---------------------------------------------------------------------------------------------------------------
# COLMAP points3D
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ColmapPoints:
    ids: np.ndarray          # int64 (N,)
    xyz: np.ndarray          # float64 (N,3)
    rgb: np.ndarray          # uint8 (N,3)
    error: np.ndarray        # float64 (N,)
    track_len: np.ndarray    # int64 (N,)


def _pack(rows):
    """rows: [(id, x, y, z, r, g, b, err, tlen)] -> id 오름차순 ColmapPoints."""
    n = len(rows)
    ids = np.empty(n, np.int64)
    xyz = np.empty((n, 3), np.float64)
    rgb = np.empty((n, 3), np.uint8)
    err = np.empty(n, np.float64)
    tl = np.empty(n, np.int64)
    for i, (pid, x, y, z, r, g, b, e, t) in enumerate(rows):
        ids[i], xyz[i], rgb[i], err[i], tl[i] = pid, (x, y, z), (r, g, b), e, t
    order = np.argsort(ids, kind="stable")
    return ColmapPoints(ids=ids[order], xyz=xyz[order], rgb=rgb[order], error=err[order], track_len=tl[order])


def _read_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) < 8 or (len(tok) - 8) % 2 != 0:
                raise ValueError(f"{path}:{ln}: points3D.txt 행 형식이 깨졌다 (토큰 {len(tok)}개)")
            try:
                pid = int(tok[0])
                x, y, z = float(tok[1]), float(tok[2]), float(tok[3])
                r, g, b = int(tok[4]), int(tok[5]), int(tok[6])
                e = float(tok[7])
            except ValueError as ex:
                raise ValueError(f"{path}:{ln}: 숫자 파싱 실패 ({ex})") from ex
            if not all(0 <= c <= 255 for c in (r, g, b)):
                raise ValueError(f"{path}:{ln}: RGB 범위 밖")
            rows.append((pid, x, y, z, r, g, b, e, (len(tok) - 8) // 2))
    return _pack(rows)


def _read_bin(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 8:
        raise ValueError(f"{path}: points3D.bin 이 너무 짧다")
    (n,) = struct.unpack_from("<Q", buf, 0)
    off, rows = 8, []
    head = struct.Struct("<Q3d3BdQ")
    # 점 하나는 최소 head.size 바이트: 말도 안 되는 n 을 빨리 거른다
    if n > (len(buf) - 8) // head.size:
        raise ValueError(f"{path}: 점 개수 {n} 이 파일 크기와 맞지 않는다")
    for _ in range(n):
        if off + head.size > len(buf):
            raise ValueError(f"{path}: points3D.bin 이 잘렸다")
        pid, x, y, z, r, g, b, e, tl = head.unpack_from(buf, off)
        off += head.size + 8 * tl
        if off > len(buf):
            raise ValueError(f"{path}: points3D.bin 트랙이 잘렸다")
        rows.append((pid, x, y, z, r, g, b, e, tl))
    if off != len(buf):
        raise ValueError(f"{path}: points3D.bin 끝에 {len(buf) - off} 바이트가 남았다")
    return _pack(rows)


def read_colmap_points3d(path) -> ColmapPoints:
    """points3D.txt|.bin 읽기. 디렉터리면 .bin 우선. id 오름차순. 깨진 형식은 ValueError."""
    p = os.fspath(path)
    if os.path.isdir(p):
        b, t = os.path.join(p, "points3D.bin"), os.path.join(p, "points3D.txt")
        if os.path.exists(b):
            p = b
        elif os.path.exists(t):
            p = t
        else:
            raise FileNotFoundError(f"{p}: points3D.bin|txt 가 없다")
    elif not os.path.exists(p):
        raise FileNotFoundError(p)
    return _read_bin(p) if p.lower().endswith(".bin") else _read_txt(p)


# ---------------------------------------------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------------------------------------------
def _quat_from_R(R: np.ndarray) -> np.ndarray:
    """(N,3,3) 회전행렬 -> (N,4) wxyz 단위 쿼터니언(w>=0)."""
    R = np.asarray(R, np.float64)
    n = R.shape[0]
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c0 = tr > 0
    c1 = ~c0 & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
    c2 = ~c0 & ~c1 & (R[:, 1, 1] >= R[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    if c0.any():
        r = R[c0]
        s = np.sqrt(tr[c0] + 1.0) * 2
        q[c0] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                          (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
    if c1.any():
        r = R[c1]
        s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
        q[c1] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                          (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
    if c2.any():
        r = R[c2]
        s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
        q[c2] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                          (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
    if c3.any():
        r = R[c3]
        s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
        q[c3] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                          (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, :1].repeat(4, 1) < 0] *= -1.0
    return q


def _frame_from_normal(n: np.ndarray) -> np.ndarray:
    """(N,3) 단위 노멀 -> (N,3,3) 회전(열 = t_u, t_v, n), det +1."""
    ax = np.argmin(np.abs(n), axis=1)
    e = np.zeros_like(n)
    e[np.arange(len(n)), ax] = 1.0
    tu = e - (e * n).sum(1, keepdims=True) * n
    tu /= np.linalg.norm(tu, axis=1, keepdims=True)
    tv = np.cross(n, tu)
    return np.stack([tu, tv, n], axis=2)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def init_from_points(xyz, rgb=None, *, k_nn=3, k_normal=8, scale_mult=1.0, min_scale=1e-4, opacity=0.1,
                     cam_centers=None, dtype=torch.float32) -> Surfels:
    """점 구름 -> surfel. 스케일 = k_nn 최근접 거리 평균, 노멀 = k_normal 이웃 PCA 최소 고유벡터."""
    from scipy.spatial import cKDTree
    P = np.asarray(xyz, np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"xyz: (N,3) 여야 한다 (shape {P.shape})")
    N = P.shape[0]
    if N < 2:
        raise ValueError("점이 2개 이상 필요하다")
    if not np.isfinite(P).all():
        raise ValueError("xyz: 비유한 값")
    if not 0.0 < float(opacity) < 1.0:
        raise ValueError(f"opacity: (0,1) 이어야 한다 ({opacity})")
    k_nn, k_normal = int(k_nn), int(k_normal)
    if k_nn < 1 or k_normal < 2:
        raise ValueError("k_nn >= 1, k_normal >= 2 여야 한다")
    # 색
    if rgb is None:
        col = np.full((N, 3), 0.5)
    else:
        a = np.asarray(rgb)
        if a.shape != (N, 3):
            raise ValueError(f"rgb: ({N},3) 여야 한다 (shape {a.shape})")
        col = color.srgb8_to_linear(a).astype(np.float64) if a.dtype == np.uint8 else a.astype(np.float64)
        if not np.isfinite(col).all():
            raise ValueError("rgb: 비유한 값")
    tree = cKDTree(P)
    kq = min(max(k_nn, k_normal), N - 1) + 1
    dist, nbr = tree.query(P, k=kq)
    dist, nbr = dist.reshape(N, kq), nbr.reshape(N, kq)
    kk = min(k_nn, N - 1)
    d_mean = dist[:, 1:kk + 1].mean(1)
    scale = np.maximum(float(min_scale), float(scale_mult) * d_mean)
    # 노멀: self + 이웃 PCA
    kn = min(k_normal, N - 1)
    nb = P[nbr[:, :kn + 1]]                                  # (N, kn+1, 3)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _w, V = np.linalg.eigh(cov)
    nrm = V[:, :, 0]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if cam_centers is not None:
        C = np.asarray(cam_centers, np.float64).reshape(-1, 3)
        if C.shape[0] == 0 or not np.isfinite(C).all():
            raise ValueError("cam_centers: 유한한 (M,3) 이어야 한다")
        dd = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        near = C[np.argmin(dd, axis=1)]
        flip = (nrm * (near - P)).sum(1) < 0
    else:
        # 결정적 기본 방향: 가장 큰 성분이 양수
        big = np.take_along_axis(nrm, np.argmax(np.abs(nrm), axis=1)[:, None], 1)[:, 0]
        flip = big < 0
    nrm = np.where(flip[:, None], -nrm, nrm)
    quats = _quat_from_R(_frame_from_normal(nrm))
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)  # noqa: E731
    return Surfels(means=t(P), quats=t(quats), log_scales=t(np.log(np.stack([scale, scale], 1))),
                   opacity_logits=t(np.full(N, _logit(float(opacity)))), colors=t(col))


def init_from_rgbd(rgb_lin, depth, *, K, c2w, stride=1, alpha=None, opacity=0.99, scale_mult=0.5,
                   dtype=torch.float32) -> Surfels:
    """RGB-D 한 장 -> 정면(카메라 평행) surfel. 유효 화소(유한·양수 깊이, alpha>1e-3)마다 하나."""
    img = np.asarray(rgb_lin, np.float64)
    d = np.asarray(depth, np.float64)
    if d.ndim != 2 or img.shape != d.shape + (3,):
        raise ValueError(f"rgb_lin (H,W,3) 와 depth (H,W) 가 맞지 않는다: {img.shape}, {d.shape}")
    H, W = d.shape
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError(f"stride: 1 이상의 정수 ({stride!r})")
    stride = int(stride)
    cam = _camera(c2w, K, W, H)
    c, Km, w2c_R, w2c_t = cam[0], cam[1], cam[2], cam[3]
    if alpha is None:
        al = np.ones((H, W))
    else:
        al = np.asarray(alpha, np.float64)
        if al.shape != (H, W):
            raise ValueError(f"alpha: ({H},{W}) 여야 한다 (shape {al.shape})")
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    dsub, asub, csub = d[vv, uu], al[vv, uu], img[vv, uu]
    ok = np.isfinite(dsub) & (dsub > 0) & np.isfinite(asub) & (asub > 1e-3)
    if not ok.any():
        return Surfels.empty(dtype)
    u, v, z, a, cc = uu[ok].astype(np.float64), vv[ok].astype(np.float64), dsub[ok], asub[ok], csub[ok]
    X = geom.unproject_world(Km, w2c_R, w2c_t, np.stack([u, v], 1), z)
    n = X.shape[0]
    q = _quat_from_R(np.repeat(c[None, :3, :3], 1, 0))[0]
    su = scale_mult * stride * z / Km[0, 0]
    sv = scale_mult * stride * z / Km[1, 1]
    op = float(opacity) * a
    t = lambda a_: torch.as_tensor(np.ascontiguousarray(a_), dtype=dtype)  # noqa: E731
    return Surfels(means=t(X), quats=t(np.tile(q, (n, 1))), log_scales=t(np.log(np.stack([su, sv], 1))),
                   opacity_logits=t(_logit(op)), colors=t(cc))


# ---------------------------------------------------------------------------------------------------------------
# 피팅
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class FitView:
    image: np.ndarray        # (H,W,3) 선형
    c2w: np.ndarray
    K: np.ndarray
    mask: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class FitOpts:
    iters: int = 300
    lr_means: float = 1.6e-4
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacity: float = 5e-2
    lr_colors: float = 2.5e-3
    lambda_ssim: float = 0.2
    lambda_dist: float = 0.0
    lambda_normal: float = 0.0
    batch_views: int = 0
    prune_opacity: float = 0.0
    bg: tuple = (0.0, 0.0, 0.0)
    seed: int = 0
    backend: str = "auto"


@dataclass(kw_only=True, eq=False)
class FitResult:
    surfels: Surfels
    losses: list
    psnr: tuple


def psnr(a, b, mask=None) -> float:
    """10*log10(1/MSE). 같으면 inf. mask (H,W) bool 이면 그 화소만."""
    x = (a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)).astype(np.float64)
    y = (b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"모양이 다르다: {x.shape} vs {y.shape}")
    se = (x - y) ** 2
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.shape != x.shape[:2]:
            raise ValueError(f"mask: {x.shape[:2]} 여야 한다 (shape {m.shape})")
        if not m.any():
            raise ValueError("mask 가 비어 있다")
        se = se[m]
    mse = float(se.mean())
    return math.inf if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)


def _gauss_win(dtype, device, size=11, sigma=1.5):
    g = torch.exp(-((torch.arange(size, dtype=dtype, device=device) - size // 2) ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _ssim_map(x, y):
    """x, y: (H,W,3) -> (H,W) 채널 평균 SSIM 지도(11x11 가우시안 sigma 1.5, 같은 크기 제로 패딩)."""
    c = x.shape[-1]
    win = _gauss_win(x.dtype, x.device).expand(c, 1, 11, 11)
    xx, yy = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    f = lambda t: F.conv2d(t, win, padding=5, groups=c)  # noqa: E731
    mx, my = f(xx), f(yy)
    sxx, syy, sxy = f(xx * xx) - mx * mx, f(yy * yy) - my * my, f(xx * yy) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s[0].mean(0)


def _depth_normals(depth, alpha, Km):
    """깊이 지도 -> 카메라 좌표 노멀 (H-2,W-2,3)과 유효 마스크. 카메라 쪽으로 뒤집는다."""
    H, W = depth.shape
    ok = torch.isfinite(depth) & (alpha > 1e-3)
    d = torch.where(ok, depth, torch.zeros_like(depth))
    dt, dev = depth.dtype, depth.device
    v, u = torch.meshgrid(torch.arange(H, dtype=dt, device=dev), torch.arange(W, dtype=dt, device=dev),
                          indexing="ij")
    yd = (v - float(Km[1, 2])) / float(Km[1, 1])
    xd = (u - float(Km[0, 2]) - float(Km[0, 1]) * yd) / float(Km[0, 0])
    p = torch.stack([xd * d, yd * d, d], -1)
    dx = p[1:-1, 2:] - p[1:-1, :-2]
    dy = p[2:, 1:-1] - p[:-2, 1:-1]
    n = torch.cross(dx, dy, dim=-1)
    n = n / torch.sqrt((n * n).sum(-1, keepdim=True).clamp_min(1e-30))
    face = (n * p[1:-1, 1:-1]).sum(-1, keepdim=True) > 0          # n.(-p) < 0 이면 뒤집는다
    n = torch.where(face, -n, n)
    valid = ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
    return n, valid


def _view_loss(surf, view, opts, bg_opts, tensors):
    img, mask = tensors
    want_dist = opts.lambda_dist > 0
    r, dist = _render_full(surf, view.c2w, view.K, img.shape[1], img.shape[0], bg_opts, want_dist)
    diff = (r.rgb - img).abs()
    if mask is None:
        l1 = diff.mean()
    else:
        l1 = (diff * mask[..., None]).sum() / (mask.sum() * 3.0).clamp_min(1.0)
    loss = (1.0 - opts.lambda_ssim) * l1
    if opts.lambda_ssim > 0:
        sm = _ssim_map(r.rgb, img)
        if mask is None:
            ssim_term = (1.0 - sm).mean()
        else:
            ssim_term = ((1.0 - sm) * mask).sum() / mask.sum().clamp_min(1.0)
        loss = loss + opts.lambda_ssim * ssim_term
    if want_dist:
        dm = dist.mean() if mask is None else (dist * mask).sum() / mask.sum().clamp_min(1.0)
        loss = loss + opts.lambda_dist * dm
    if opts.lambda_normal > 0 and img.shape[0] >= 3 and img.shape[1] >= 3:
        cam = _camera(view.c2w, view.K, img.shape[1], img.shape[0])
        Rw2c = torch.as_tensor(cam[2], dtype=img.dtype, device=img.device)
        nd, valid = _depth_normals(r.depth, r.alpha.detach(), cam[1])
        nr = r.normal[1:-1, 1:-1] @ Rw2c.T
        wgt = r.alpha[1:-1, 1:-1] * valid
        if mask is not None:
            wgt = wgt * mask[1:-1, 1:-1]
        term = (wgt * (1.0 - (nr * nd).sum(-1))).sum() / wgt.detach().sum().clamp_min(1.0)
        loss = loss + opts.lambda_normal * term
    return loss


def fit(init: Surfels, views, opts: FitOpts | None = None) -> FitResult:
    """미분 래스터 + Adam 으로 surfel 을 뷰들에 맞춘다. init 은 바뀌지 않는다."""
    opts = FitOpts() if opts is None else opts
    views = list(views)
    if not views:
        raise ValueError("views: 하나 이상 필요하다")
    if not isinstance(init, Surfels):
        raise TypeError("init: Surfels 여야 한다")
    if opts.iters < 0 or opts.batch_views < 0:
        raise ValueError("iters, batch_views >= 0 여야 한다")
    dt, dev = init.means.dtype, init.means.device
    ropts = SplatOpts(bg=opts.bg, backend=opts.backend)
    prep = []
    for i, v in enumerate(views):
        img = torch.as_tensor(np.asarray(v.image, np.float64), dtype=dt, device=dev)
        if img.ndim != 3 or img.shape[-1] != 3 or not bool(torch.isfinite(img).all()):
            raise ValueError(f"views[{i}].image: 유한한 (H,W,3) 여야 한다")
        _camera(v.c2w, v.K, img.shape[1], img.shape[0])
        m = None
        if v.mask is not None:
            mk = np.asarray(v.mask, bool)
            if mk.shape != tuple(img.shape[:2]):
                raise ValueError(f"views[{i}].mask: {tuple(img.shape[:2])} 여야 한다")
            m = torch.as_tensor(mk, dtype=dt, device=dev)
        prep.append((img, m))

    surf = init.clone().requires_grad_(True)
    lrs = [opts.lr_means, opts.lr_quats, opts.lr_scales, opts.lr_opacity, opts.lr_colors]
    optim = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(surf.params(), lrs)])
    gen = torch.Generator().manual_seed(int(opts.seed))
    V = len(views)
    k = V if opts.batch_views == 0 else min(int(opts.batch_views), V)
    losses = []
    for _ in range(int(opts.iters)):
        sel = range(V) if k == V else torch.randperm(V, generator=gen)[:k].tolist()
        optim.zero_grad(set_to_none=True)
        total = None
        for i in sel:
            li = _view_loss(surf, views[i], opts, ropts, prep[i])
            total = li if total is None else total + li
        total = total / len(sel)
        total.backward()
        losses.append(total.detach())
        optim.step()
    out = surf.detach().clone()
    if opts.prune_opacity > 0 and out.n:
        out = out.subset(out.opacities() >= opts.prune_opacity)
    scores = []
    with torch.no_grad():
        for v, (img, m) in zip(views, prep):
            r, _ = _render_full(out, v.c2w, v.K, img.shape[1], img.shape[0], ropts)
            scores.append(psnr(r.rgb, img, None if m is None else m > 0.5))
    return FitResult(surfels=out, losses=[float(x) for x in losses], psnr=tuple(scores))
