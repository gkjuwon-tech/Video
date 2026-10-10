"""RDR-01 2DGS 래퍼 v0: 미분 래스터(torch 정본 + gsplat) · dual-cut LOD.

계약서: ops/harness/contracts/RDR-01.md. 세계 오른손·미터, 카메라 OpenCV, `c2w` (4,4) 카메라->세계,
픽셀 중심이 정수 좌표. 색은 선형 Rec.709, 활성화 없이 그대로 합성(SH 없음).

구현 요약
  * 2D 가우시안 surfel: 점 `p + s_u*u*t_u + s_v*v*t_v`, `R(q)` 의 열 = (t_u, t_v, n).
    화소 광선과 surfel 평면의 교점에서 (u, v)를 구한다(2DGS 광선-surfel 교차).
  * 화소 알파 `a = min(0.999, o*exp(-q/2))`, `q = min(u^2+v^2, 2*|x-c|^2)` (c = 중심 투영, 저역 필터
    sigma = 1/sqrt(2) px). 교점이 없거나(edge-on, 카메라 뒤) 퇴화하면 `q = 2*|x-c|^2` 만 남는다.
  * 중심 z 오름차순(안정 정렬)으로 over 합성. `T_i = prod(1-a_j)` 는 log1p 누적합으로 계산한다.
  * torch 백엔드는 화소 덩어리마다 (surfel x 화소) 밀집 텐서를 만든다(정본, CPU/CUDA, float32/64).
    gsplat 백엔드는 `gsplat.rasterization_2dgs`(CUDA float32). gsplat 은 지연 import 한다.
"""
from __future__ import annotations

import importlib
import math
import numbers
import os
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

__all__ = ["BackendUnavailable", "Surfels", "SplatOpts", "SplatRender", "resolve_backend", "render",
           "lod_lambda", "DualCutRender", "render_dual_cut"]

_ALPHA_MAX = 0.999
_ALPHA_MIN = 1.0 / 255.0
_ALPHA_OUT_EPS = 1e-3
_DENOM_EPS = 1e-6
_CHUNK_ELEMS = 1 << 20           # (surfel x 화소) 덩어리 크기 상한(원소 수)
_BACKENDS = ("auto", "torch", "gsplat")


class BackendUnavailable(RuntimeError):
    """요청한 래스터 백엔드를 쓸 수 없다(CPU 텐서, gsplat 없음, JIT 실패 ...)."""


# ---------------------------------------------------------------------------------------------------------------
# Surfels
# ---------------------------------------------------------------------------------------------------------------
_FIELDS = ("means", "quats", "log_scales", "opacity_logits", "colors")
_TAILS = {"means": (3,), "quats": (4,), "log_scales": (2,), "opacity_logits": (), "colors": (3,)}


@dataclass(kw_only=True, eq=False)
class Surfels:
    means: torch.Tensor            # (N,3)
    quats: torch.Tensor            # (N,4) wxyz, 정규화 전
    log_scales: torch.Tensor       # (N,2)
    opacity_logits: torch.Tensor   # (N,)
    colors: torch.Tensor           # (N,3) 선형 RGB

    def __post_init__(self):
        vals = {}
        for f in _FIELDS:
            v = getattr(self, f)
            if not isinstance(v, torch.Tensor):
                try:
                    v = torch.as_tensor(np.asarray(v))
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"{f}: 텐서로 바꿀 수 없다 ({e})") from e
            vals[f] = v
        dt, dev = vals["means"].dtype, vals["means"].device
        if dt not in (torch.float32, torch.float64):
            raise ValueError(f"dtype 은 float32|float64 여야 한다 ({dt})")
        n = vals["means"].shape[0] if vals["means"].ndim >= 1 else -1
        for f in _FIELDS:
            v = vals[f]
            if v.dtype != dt:
                raise ValueError(f"{f}: dtype {v.dtype} != {dt}")
            if v.device != dev:
                raise ValueError(f"{f}: device {v.device} != {dev}")
            if tuple(v.shape) != (n,) + _TAILS[f]:
                raise ValueError(f"{f}: 모양 {tuple(v.shape)} != {(n,) + _TAILS[f]}")
            if n > 0 and not bool(torch.isfinite(v).all()):
                raise ValueError(f"{f}: 비유한 값")
            setattr(self, f, v)

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @property
    def dtype(self):
        return self.means.dtype

    @property
    def device(self):
        return self.means.device

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    def rotations(self) -> torch.Tensor:
        return _quat_to_rot(self.quats)

    def normals(self) -> torch.Tensor:
        return self.rotations()[:, :, 2]

    def params(self) -> list:
        return [self.means, self.quats, self.log_scales, self.opacity_logits, self.colors]

    def requires_grad_(self, flag: bool = True) -> "Surfels":
        for p in self.params():
            p.requires_grad_(flag)
        return self

    def detach(self) -> "Surfels":
        return Surfels(**{f: getattr(self, f).detach() for f in _FIELDS})

    def clone(self) -> "Surfels":
        """독립 복사(기울기 그래프에서 분리)."""
        return Surfels(**{f: getattr(self, f).detach().clone() for f in _FIELDS})

    def to(self, device=None, dtype=None) -> "Surfels":
        return Surfels(**{f: getattr(self, f).to(device=device, dtype=dtype) for f in _FIELDS})

    def subset(self, idx) -> "Surfels":
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(np.asarray(idx))
        idx = idx.to(self.means.device)
        return Surfels(**{f: getattr(self, f)[idx] for f in _FIELDS})

    @classmethod
    def cat(cls, items) -> "Surfels":
        items = list(items)
        if not items:
            raise ValueError("cat: 빈 목록")
        return cls(**{f: torch.cat([getattr(s, f) for s in items], dim=0) for f in _FIELDS})

    @classmethod
    def empty(cls, dtype=torch.float32, device=None) -> "Surfels":
        kw = dict(dtype=dtype, device=device)
        return cls(**{f: torch.zeros((0,) + _TAILS[f], **kw) for f in _FIELDS})

    def save(self, path) -> None:
        arrs = {f: getattr(self, f).detach().cpu().numpy() for f in _FIELDS}
        with open(os.fspath(path), "wb") as fh:
            np.savez(fh, **arrs)

    @classmethod
    def load(cls, path) -> "Surfels":
        with np.load(os.fspath(path)) as z:
            missing = [f for f in _FIELDS if f not in z.files]
            if missing:
                raise ValueError(f"surfel npz 에 필드가 없다: {missing}")
            return cls(**{f: torch.from_numpy(np.array(z[f])) for f in _FIELDS})


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3). 정규화 전 쿼터니언을 받는다."""
    nrm = torch.sqrt((q * q).sum(-1, keepdim=True).clamp_min(1e-30))
    q = q / nrm
    w, x, y, z = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return r.reshape(*q.shape[:-1], 3, 3)


# ---------------------------------------------------------------------------------------------------------------
# 옵션 · 결과
# ---------------------------------------------------------------------------------------------------------------
def _real(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name}: 유한해야 한다 ({v!r})")
    return f


@dataclass(frozen=True, kw_only=True)
class SplatOpts:
    bg: tuple = (0.0, 0.0, 0.0)
    near: float = 0.01
    far: float = 1e4
    backend: str = "auto"

    def __post_init__(self):
        try:
            bg = tuple(_real("bg", x) for x in self.bg)
        except TypeError as e:
            raise ValueError(f"bg: 길이 3 의 수열이어야 한다 ({self.bg!r})") from e
        if len(bg) != 3:
            raise ValueError(f"bg: 길이 3 이어야 한다 ({self.bg!r})")
        near, far = _real("near", self.near), _real("far", self.far)
        if not near > 0:
            raise ValueError(f"near > 0 이어야 한다 ({near})")
        if not far > near:
            raise ValueError(f"far > near 여야 한다 (near={near}, far={far})")
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend: {_BACKENDS} 중 하나 ({self.backend!r})")
        object.__setattr__(self, "bg", bg)
        object.__setattr__(self, "near", near)
        object.__setattr__(self, "far", far)


@dataclass(kw_only=True, eq=False)
class SplatRender:
    rgb: torch.Tensor       # (H,W,3)
    alpha: torch.Tensor     # (H,W)
    depth: torch.Tensor     # (H,W) 카메라 z, 알파<1e-3 은 +inf
    normal: torch.Tensor    # (H,W,3) 세계 단위 노멀(카메라 쪽), 알파<1e-3 은 0
    backend: str = "torch"

    def numpy(self) -> dict:
        return {k: getattr(self, k).detach().cpu().numpy().astype(np.float32)
                for k in ("rgb", "alpha", "depth", "normal")}


# ---------------------------------------------------------------------------------------------------------------
# 백엔드 선택
# ---------------------------------------------------------------------------------------------------------------
_GS = {"mod": None, "err": None}


def _import_gsplat():
    if _GS["mod"] is not None:
        return _GS["mod"]
    if _GS["err"] is not None:
        raise BackendUnavailable(_GS["err"])
    try:
        mod = importlib.import_module("gsplat")
    except Exception as e:  # noqa: BLE001
        _GS["err"] = f"gsplat import 실패: {type(e).__name__}: {e}"
        raise BackendUnavailable(_GS["err"]) from e
    _GS["mod"] = mod
    return mod


def _dev_type(device) -> str:
    if isinstance(device, torch.Tensor):
        device = device.device
    return torch.device(device).type


def resolve_backend(name: str, device) -> str:
    """`name` 을 실제 백엔드("torch"|"gsplat")로 바꾼다. "auto" 는 env BUC_SPLAT_BACKEND 를 먼저 본다."""
    if name not in _BACKENDS:
        raise ValueError(f"backend: {_BACKENDS} 중 하나 ({name!r})")
    on_cuda = _dev_type(device) == "cuda"
    if name == "torch":
        return "torch"
    if name == "gsplat":
        if not on_cuda:
            raise BackendUnavailable("gsplat 백엔드는 CUDA 텐서만 지원한다")
        _import_gsplat()
        return "gsplat"
    env = os.environ.get("BUC_SPLAT_BACKEND", "").strip().lower()
    if env not in ("", "auto", "torch", "gsplat"):
        raise ValueError(f"BUC_SPLAT_BACKEND: torch|gsplat|auto 중 하나 ({env!r})")
    if env == "torch":
        return "torch"
    if on_cuda:
        try:
            _import_gsplat()
            return "gsplat"
        except BackendUnavailable as e:
            warnings.warn(f"gsplat 을 쓸 수 없어 torch 백엔드로 폴백한다: {e}", RuntimeWarning, stacklevel=2)
    elif env == "gsplat":
        warnings.warn("BUC_SPLAT_BACKEND=gsplat 이지만 CPU 텐서라 torch 백엔드를 쓴다", RuntimeWarning,
                      stacklevel=2)
    return "torch"


# ---------------------------------------------------------------------------------------------------------------
# 카메라 검증
# ---------------------------------------------------------------------------------------------------------------
def _size(width, height):
    for name, v in (("width", width), ("height", height)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or int(v) < 1:
            raise ValueError(f"{name}: 1 이상의 정수여야 한다 ({v!r})")
    return int(width), int(height)


def _camera(c2w, K, width, height):
    W, H = _size(width, height)
    try:
        c = np.asarray(c2w, np.float64)
        Km = np.asarray(K, np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"카메라 배열 변환 실패: {e}") from e
    if c.shape != (4, 4) or not np.isfinite(c).all():
        raise ValueError(f"c2w: 유한한 (4,4) 여야 한다 (shape {c.shape})")
    if Km.shape != (3, 3) or not np.isfinite(Km).all():
        raise ValueError(f"K: 유한한 (3,3) 여야 한다 (shape {Km.shape})")
    R = c[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) <= 0:
        raise ValueError("c2w: 회전부가 직교(det>0)가 아니다")
    if not np.allclose(c[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("c2w: 마지막 행이 [0,0,0,1] 이 아니다")
    if not (Km[0, 0] > 0 and Km[1, 1] > 0):
        raise ValueError("K: fx, fy > 0 이어야 한다")
    w2c_R = R.T
    w2c_t = -R.T @ c[:3, 3]
    return c, Km, w2c_R, w2c_t, W, H


# ---------------------------------------------------------------------------------------------------------------
# torch 정본 래스터
# ---------------------------------------------------------------------------------------------------------------
def _background(P, dt, dev, bg, anchor):
    """아무것도 그려지지 않은 결과. 기울기 그래프가 끊기지 않게 0*params 를 더한다."""
    bgt = torch.tensor(bg, dtype=dt, device=dev)
    rgb = bgt.expand(P, 3).clone()
    zero = None
    if anchor is not None and any(p.requires_grad for p in anchor.params()):
        zero = sum((p.sum() for p in anchor.params())) * 0.0
        rgb = rgb + zero
    alpha = torch.zeros(P, dtype=dt, device=dev) if zero is None else zero.expand(P).clone()
    depth = torch.full((P,), float("inf"), dtype=dt, device=dev)
    normal = torch.zeros(P, 3, dtype=dt, device=dev)
    dist = torch.zeros(P, dtype=dt, device=dev)
    return dict(rgb=rgb, alpha=alpha, depth=depth, normal=normal, distort=dist)


def _render_torch(s: Surfels, w2c_R, w2c_t, Km, W, H, opts: SplatOpts, pix=None, want_distort=False):
    """평탄 결과 dict(rgb (P,3), alpha (P,), depth (P,), normal (P,3), distort (P,)). pix: 평탄 화소 번호 (P,)."""
    dt, dev = s.means.dtype, s.means.device
    if pix is None:
        pix = torch.arange(H * W, device=dev)
    P = int(pix.shape[0])
    if s.n == 0 or P == 0:
        return _background(P, dt, dev, opts.bg, s if s.n else None)

    Rw = torch.as_tensor(w2c_R, dtype=dt, device=dev)
    tw = torch.as_tensor(w2c_t, dtype=dt, device=dev)
    fx, fy, skew, cx, cy = (float(Km[0, 0]), float(Km[1, 1]), float(Km[0, 1]), float(Km[0, 2]), float(Km[1, 2]))

    pc_all = s.means @ Rw.T + tw
    z_all = pc_all[:, 2]
    keep = (z_all >= opts.near) & (z_all <= opts.far)
    idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return _background(P, dt, dev, opts.bg, s)
    order = torch.sort(z_all.detach()[idx], stable=True).indices
    idx = idx[order]

    R_world = _quat_to_rot(s.quats[idx])                      # (M,3,3) 열 = (t_u, t_v, n)
    Rcam = Rw @ R_world
    pc = pc_all[idx]                                          # (M,3)
    zc = pc[:, 2]
    tu, tv, nc = Rcam[:, :, 0], Rcam[:, :, 1], Rcam[:, :, 2]
    su, sv = torch.exp(s.log_scales[idx]).unbind(-1)          # (M,)
    op = torch.sigmoid(s.opacity_logits[idx])
    col = s.colors[idx]
    numer = (nc * pc).sum(-1)                                 # n.p (카메라 좌표)
    tu_p = (tu * pc).sum(-1)
    tv_p = (tv * pc).sum(-1)
    sign = torch.where(numer.detach() > 0, -torch.ones_like(numer), torch.ones_like(numer))
    nw = R_world[:, :, 2] * sign[:, None]                     # 카메라 쪽으로 뒤집은 세계 노멀
    cu = (fx * pc[:, 0] + skew * pc[:, 1]) / zc + cx          # 중심 투영 (px)
    cv = fy * pc[:, 1] / zc + cy

    M = int(idx.numel())
    step = max(32, _CHUNK_ELEMS // M)
    outs = {k: [] for k in ("rgb", "alpha", "depth", "normal", "distort")}
    bgt = torch.tensor(opts.bg, dtype=dt, device=dev)
    for a in range(0, P, step):
        pp = pix[a:a + step]
        uu = (pp % W).to(dt)
        vv = torch.div(pp, W, rounding_mode="floor").to(dt)
        yd = (vv - cy) / fy
        xd = (uu - cx - skew * yd) / fx
        Dm = torch.stack([xd, yd, torch.ones_like(xd)], 0)    # (3,p)

        d2 = (uu[None, :] - cu[:, None]) ** 2 + (vv[None, :] - cv[:, None]) ** 2
        Dn = nc @ Dm                                           # (M,p)
        ok = Dn.detach().abs() > _DENOM_EPS
        Dn_safe = torch.where(ok, Dn, torch.ones_like(Dn))
        t = numer[:, None] / Dn_safe
        ok = ok & (t.detach() > 0)
        ru = t * (tu @ Dm) - tu_p[:, None]
        rv = t * (tv @ Dm) - tv_p[:, None]
        q_ray = (ru / su[:, None]) ** 2 + (rv / sv[:, None]) ** 2
        q_flt = 2.0 * d2
        q = torch.where(ok, torch.minimum(q_ray, q_flt), q_flt)
        A = torch.clamp(op[:, None] * torch.exp(-0.5 * q), max=_ALPHA_MAX)
        A = torch.where(A >= _ALPHA_MIN, A, torch.zeros_like(A))

        logt = torch.log1p(-A)
        cum = torch.cumsum(logt, dim=0) - logt                 # 배타적 누적합
        w = A * torch.exp(cum)                                 # (M,p)
        alpha = w.sum(0)
        rgb = w.T @ col + (1.0 - alpha)[:, None] * bgt
        zi = torch.where(ok, t, zc[:, None].expand_as(t))
        dnum = (w * zi).sum(0)
        low = alpha.detach() < _ALPHA_OUT_EPS
        depth = torch.where(low, torch.full_like(alpha, float("inf")),
                            dnum / torch.where(low, torch.ones_like(alpha), alpha))
        nsum = w.T @ nw
        nn = torch.sqrt((nsum * nsum).sum(-1, keepdim=True).clamp_min(1e-30))
        normal = torch.where(low[:, None], torch.zeros_like(nsum), nsum / nn)
        outs["rgb"].append(rgb)
        outs["alpha"].append(alpha)
        outs["depth"].append(depth)
        outs["normal"].append(normal)
        if want_distort:
            wc = torch.cumsum(w, 0) - w
            wzc = torch.cumsum(w * zi, 0) - w * zi
            outs["distort"].append(2.0 * (w * (zi * wc - wzc)).sum(0))
        else:
            outs["distort"].append(torch.zeros_like(alpha))
    return {k: torch.cat(v, 0) for k, v in outs.items()}


# ---------------------------------------------------------------------------------------------------------------
# gsplat 백엔드 (CUDA float32)
# ---------------------------------------------------------------------------------------------------------------
def _render_gsplat(s: Surfels, w2c_R, w2c_t, Km, W, H, opts: SplatOpts, want_distort=False):
    gs = _import_gsplat()
    dev, dt = s.means.device, s.means.dtype
    if dev.type != "cuda":
        raise BackendUnavailable("gsplat 백엔드는 CUDA 텐서만 지원한다")
    if dt != torch.float32:
        raise BackendUnavailable("gsplat 백엔드는 float32 만 지원한다")
    P = H * W
    if s.n == 0:
        return _background(P, dt, dev, opts.bg, None)
    view = np.eye(4)
    view[:3, :3] = w2c_R
    view[:3, 3] = w2c_t
    viewmats = torch.as_tensor(view, dtype=dt, device=dev)[None]
    Ks = torch.as_tensor(Km, dtype=dt, device=dev)[None]
    sc = torch.exp(s.log_scales)
    scales3 = torch.cat([sc, torch.full_like(sc[:, :1], 1e-4)], dim=-1)
    bg = torch.tensor(opts.bg, dtype=dt, device=dev)[None]
    try:
        colors, alphas, normals, _, distort, _, _ = gs.rasterization_2dgs(
            s.means, s.quats, scales3, torch.sigmoid(s.opacity_logits), s.colors, viewmats, Ks, W, H,
            near_plane=opts.near, far_plane=opts.far, render_mode="RGB+ED", backgrounds=bg,
            distloss=bool(want_distort))
    except BackendUnavailable:
        raise
    except Exception as e:  # noqa: BLE001  (JIT 컴파일 실패, 커널 오류 ...)
        _GS["err"] = f"gsplat 실행 실패: {type(e).__name__}: {e}"
        raise BackendUnavailable(_GS["err"]) from e
    rgb = colors[0, ..., :3].reshape(P, 3)
    depth = colors[0, ..., 3].reshape(P)
    alpha = alphas[0, ..., 0].reshape(P)
    normal = normals[0].reshape(P, 3)
    low = alpha.detach() < _ALPHA_OUT_EPS
    depth = torch.where(low, torch.full_like(depth, float("inf")), depth)
    nn = torch.sqrt((normal * normal).sum(-1, keepdim=True).clamp_min(1e-30))
    normal = torch.where(low[:, None], torch.zeros_like(normal), normal / nn)
    dist = distort[0, ..., 0].reshape(P) if (want_distort and distort is not None) else torch.zeros_like(alpha)
    return dict(rgb=rgb, alpha=alpha, depth=depth, normal=normal, distort=dist)


def _render_flat(s, cam, opts, pix=None, want_distort=False):
    """백엔드를 골라 평탄 결과 dict 와 실제 백엔드 이름을 돌려준다."""
    _c, Km, w2c_R, w2c_t, W, H = cam
    want = resolve_backend(opts.backend, s.means.device)
    explicit = opts.backend == "gsplat"
    if want == "gsplat" and s.means.dtype != torch.float32:
        if explicit:
            raise BackendUnavailable("gsplat 백엔드는 float32 만 지원한다")
        want = "torch"
    if want == "gsplat":
        try:
            out = _render_gsplat(s, w2c_R, w2c_t, Km, W, H, opts, want_distort)
            if pix is not None:
                out = {k: v[pix] for k, v in out.items()}
            return out, "gsplat"
        except BackendUnavailable as e:
            if explicit:
                raise
            warnings.warn(f"gsplat 실패, torch 백엔드로 폴백한다: {e}", RuntimeWarning, stacklevel=3)
    return _render_torch(s, w2c_R, w2c_t, Km, W, H, opts, pix, want_distort), "torch"


def _render_full(s, c2w, K, width, height, opts, want_distort=False):
    if not isinstance(s, Surfels):
        raise TypeError("surfels: Surfels 여야 한다")
    opts = SplatOpts() if opts is None else opts
    cam = _camera(c2w, K, width, height)
    out, be = _render_flat(s, cam, opts, None, want_distort)
    W, H = cam[4], cam[5]
    r = SplatRender(rgb=out["rgb"].reshape(H, W, 3), alpha=out["alpha"].reshape(H, W),
                    depth=out["depth"].reshape(H, W), normal=out["normal"].reshape(H, W, 3), backend=be)
    return r, out["distort"].reshape(H, W)


def render(surfels: Surfels, *, c2w, K, width, height, opts: SplatOpts | None = None) -> SplatRender:
    """surfel 을 한 카메라로 찍는다(선형 RGB·알파·z-깊이·세계 노멀). 미분 가능."""
    return _render_full(surfels, c2w, K, width, height, opts)[0]


# ---------------------------------------------------------------------------------------------------------------
# dual-cut LOD
# ---------------------------------------------------------------------------------------------------------------
def lod_lambda(depth, K, h0, kappa=1.0):
    """lambda = log2(h0 * fx / (kappa * depth)). depth=inf -> -inf. numpy->numpy, tensor->tensor."""
    fx = float(np.asarray(K, np.float64)[0, 0])
    h0, kappa = _real("h0", h0), _real("kappa", kappa)
    if isinstance(depth, torch.Tensor):
        return torch.log2(h0 * fx / (kappa * depth))
    d = np.asarray(depth, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(h0 * fx / (kappa * d))


@dataclass(kw_only=True, eq=False)
class DualCutRender:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    level: torch.Tensor
    n_tiles: int
    n_tiles_double: int
    cuts_rendered: tuple


def _tile_lambda(lam, H, W, tile):
    """타일마다 유한 값의 평균(없으면 0). NaN 은 ValueError. -> (Th, Tw) float64."""
    if isinstance(lam, torch.Tensor):
        a = lam.detach().cpu().numpy()
    else:
        a = np.asarray(lam)
    try:
        a = a.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"lam: 수치 배열이어야 한다 ({e})") from e
    if a.ndim == 0:
        a = np.full((H, W), float(a))
    if a.shape != (H, W):
        raise ValueError(f"lam: 스칼라 또는 ({H},{W}) 여야 한다 (shape {a.shape})")
    if np.isnan(a).any():
        raise ValueError("lam: NaN 이 있다")
    Th, Tw = -(-H // tile), -(-W // tile)
    fin = np.isfinite(a)
    val = np.zeros((Th * tile, Tw * tile))
    cnt = np.zeros((Th * tile, Tw * tile))
    val[:H, :W] = np.where(fin, a, 0.0)
    cnt[:H, :W] = fin
    sv = val.reshape(Th, tile, Tw, tile).sum(axis=(1, 3))
    sc = cnt.reshape(Th, tile, Tw, tile).sum(axis=(1, 3))
    return np.where(sc > 0, sv / np.maximum(sc, 1), 0.0)


def render_dual_cut(cuts: Sequence[Surfels], lam, *, c2w, K, width, height, opts: SplatOpts | None = None,
                    tile: int = 16, band=(0.05, 0.95)) -> DualCutRender:
    """타일마다 두 컷을 lambda 의 소수부로 섞는다. cuts[0] 이 가장 거친 레벨."""
    cuts = list(cuts)
    if not cuts or not all(isinstance(c, Surfels) for c in cuts):
        raise ValueError("cuts: 비어 있지 않은 Surfels 목록이어야 한다")
    if isinstance(tile, bool) or not isinstance(tile, numbers.Integral) or int(tile) < 1:
        raise ValueError(f"tile: 1 이상의 정수여야 한다 ({tile!r})")
    tile = int(tile)
    try:
        b0, b1 = float(band[0]), float(band[1])
    except (TypeError, IndexError, ValueError) as e:
        raise ValueError(f"band: (lo, hi) 여야 한다 ({band!r})") from e
    if not (0.0 <= b0 < b1 <= 1.0):
        raise ValueError(f"band: 0 <= lo < hi <= 1 ({band!r})")
    opts = SplatOpts() if opts is None else opts
    cam = _camera(c2w, K, width, height)
    W, H = cam[4], cam[5]
    L = len(cuts)
    lt = np.clip(_tile_lambda(lam, H, W, tile), 0.0, float(L - 1))
    la = np.minimum(np.floor(lt), max(L - 2, 0)).astype(np.int64)
    f = lt - la
    f = np.where(f <= b0, 0.0, np.where(f >= b1, 1.0, f))
    if L == 1:
        f = np.zeros_like(f)
    Th, Tw = lt.shape

    def up(x):
        return np.repeat(np.repeat(x, tile, axis=0), tile, axis=1)[:H, :W].reshape(-1)

    la_p, f_p = up(la), up(f)
    dom_p = np.where(f_p < 0.5, la_p, la_p + 1)
    lev_p = la_p + f_p
    n_double = int(((f > 0) & (f < 1)).sum())

    dev = cuts[0].means.device
    dt = cuts[0].means.dtype
    P = H * W
    rgb = torch.zeros(P, 3, dtype=dt, device=dev)
    alpha = torch.zeros(P, dtype=dt, device=dev)
    depth = torch.zeros(P, dtype=dt, device=dev)
    normal = torch.zeros(P, 3, dtype=dt, device=dev)
    used = []
    for c in range(L):
        w_p = np.where(la_p == c, 1.0 - f_p, 0.0) + np.where(la_p + 1 == c, f_p, 0.0)
        sel = np.nonzero(w_p > 0)[0]
        if sel.size == 0:
            continue
        used.append(c)
        sel_t = torch.as_tensor(sel, device=dev)
        out, _be = _render_flat(cuts[c], cam, opts, sel_t)
        wt = torch.as_tensor(w_p[sel], dtype=dt, device=dev)
        rgb = rgb.index_add(0, sel_t, out["rgb"] * wt[:, None])
        alpha = alpha.index_add(0, sel_t, out["alpha"] * wt)
        dsel = np.nonzero(dom_p == c)[0]
        if dsel.size:
            # 우세 컷은 항상 렌더된 컷 안에 있다(f<0.5 -> la, 아니면 la+1; 둘 다 w>0)
            pos = torch.as_tensor(np.searchsorted(sel, dsel), device=dev)
            dsel_t = torch.as_tensor(dsel, device=dev)
            depth = depth.index_copy(0, dsel_t, out["depth"][pos])
            normal = normal.index_copy(0, dsel_t, out["normal"][pos])
    return DualCutRender(
        rgb=rgb.reshape(H, W, 3), alpha=alpha.reshape(H, W), depth=depth.reshape(H, W),
        normal=normal.reshape(H, W, 3), level=torch.as_tensor(lev_p, dtype=dt, device=dev).reshape(H, W),
        n_tiles=int(Th * Tw), n_tiles_double=n_double, cuts_rendered=tuple(used))
