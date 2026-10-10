"""RDR-03 TSQ(Textured Soft Quad) 미분 래스터 · 8×8 페이지 바이리니어 · 초타원 에지 n≥2 · BC7/BC5 페이지 코덱.

계약서: ops/harness/contracts/RDR-03.md. 규약은 RDR-01(`buc.render.splat2d`)과 같다: 세계 오른손·미터,
카메라 OpenCV, `c2w` (4,4) 카메라->세계, 픽셀 중심이 정수 좌표, 색은 선형 RGB 그대로 합성.

래스터 요약
  * TSQ i: 점 `p + s_u*u*t_u + s_v*v*t_v`, `R(q)` 열 = (t_u, t_v, n). 화소 광선과 평면의 교점에서 (u, v).
  * `E = ½(|u|^n + |v|^n)` (교점 무효면 +inf), `E_lp = |x - c|²` (px), `α = min(.999, o·exp(-min(E, E_lp)))`,
    `α < 1/255` 이면 0. n = max(edge_n, 2).
  * 색 = 페이지 바이리니어(텍셀 중심 `u_i = -1 + (2i+1)/8`, 클램프 주소). 교점 무효면 (u,v)=(0,0).
  * 중심 z 오름차순(안정) over 합성, depth·normal 은 RDR-01 과 같다.
  * `|u|^n` 은 `exp(min(n·log|u|, 80))` 로 계산해 큰 n·큰 |u| (float32 오버플로)에서도 값 0·기울기 유한.

백엔드: "torch"(정본, CPU·CUDA, float32/64)와 "cuda"(`tsq_kernel.cu` 지연 JIT, CUDA float32).
`import buc.render.tsq` 는 CUDA 를 초기화하거나 컴파일하지 않는다.
"""
from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from buc.render.splat2d import BackendUnavailable, Surfels, _camera, _quat_to_rot, _real

__all__ = ["PAGE", "LMIN", "BackendUnavailable", "TSQ", "TSQOpts", "TSQRender", "resolve_backend",
           "load_extension", "edge_falloff", "sample_page", "render", "encode_bc7", "decode_bc7", "encode_bc5",
           "decode_bc5", "PageBlob", "compress_pages", "decompress_pages", "decompress_nh"]

PAGE = 8
LMIN = 2.0 ** -16

_ALPHA_MAX = 0.999
_ALPHA_MIN = 1.0 / 255.0
_ALPHA_OUT_EPS = 1e-3
_DENOM_EPS = 1e-6
_LOGPOW_MAX = 80.0              # |u|^n 의 지수 상한 (exp(80) 은 float32 에서 유한)
_CHUNK_ELEMS = 1 << 20          # (TSQ x 화소) 덩어리 원소 수 상한
_BACKENDS = ("auto", "torch", "cuda")
_NG = 22                        # cuda 백엔드 TSQ 당 기하 특징 수


# ---------------------------------------------------------------------------------------------------------------
# TSQ
# ---------------------------------------------------------------------------------------------------------------
_FIELDS = ("means", "quats", "log_scales", "opacity_logits", "pages", "edge_n")
_TAILS = {"means": (3,), "quats": (4,), "log_scales": (2,), "opacity_logits": (), "pages": (PAGE, PAGE, 3),
          "edge_n": ()}


@dataclass(kw_only=True, eq=False)
class TSQ:
    means: torch.Tensor            # (N,3)
    quats: torch.Tensor            # (N,4) wxyz, 정규화 전
    log_scales: torch.Tensor       # (N,2)
    opacity_logits: torch.Tensor   # (N,)
    edge_n: torch.Tensor           # (N,) ≥ 2
    pages: torch.Tensor            # (N,8,8,3) 선형 RGB, page[j, i] = 행 j(v), 열 i(u)

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
        if n > 0 and bool((vals["edge_n"] < 2.0).any()):
            raise ValueError("edge_n: 2 이상이어야 한다")
        for f in _FIELDS:
            setattr(self, f, vals[f])

    # -- 성질 ---------------------------------------------------------------------------------------------------
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
        return [self.means, self.quats, self.log_scales, self.opacity_logits, self.pages, self.edge_n]

    # -- 도구 ---------------------------------------------------------------------------------------------------
    def _kw(self, fn) -> dict:
        return {f: fn(getattr(self, f)) for f in _FIELDS}

    def requires_grad_(self, flag: bool = True) -> "TSQ":
        for p in self.params():
            p.requires_grad_(flag)
        return self

    def detach(self) -> "TSQ":
        return TSQ(**self._kw(lambda x: x.detach()))

    def clone(self) -> "TSQ":
        """독립 복사(기울기 그래프에서 분리)."""
        return TSQ(**self._kw(lambda x: x.detach().clone()))

    def to(self, device=None, dtype=None) -> "TSQ":
        return TSQ(**self._kw(lambda x: x.to(device=device, dtype=dtype)))

    def subset(self, idx) -> "TSQ":
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(np.asarray(idx))
        idx = idx.to(self.means.device)
        return TSQ(**self._kw(lambda x: x[idx]))

    @classmethod
    def cat(cls, items) -> "TSQ":
        items = list(items)
        if not items:
            raise ValueError("cat: 빈 목록")
        return cls(**{f: torch.cat([getattr(s, f) for s in items], dim=0) for f in _FIELDS})

    @classmethod
    def empty(cls, dtype=torch.float32, device=None) -> "TSQ":
        return cls(**{f: torch.zeros((0,) + _TAILS[f], dtype=dtype, device=device) for f in _FIELDS})

    def save(self, path) -> None:
        arrs = {f: getattr(self, f).detach().cpu().numpy() for f in _FIELDS}
        with open(os.fspath(path), "wb") as fh:
            np.savez(fh, **arrs)

    @classmethod
    def load(cls, path) -> "TSQ":
        with np.load(os.fspath(path)) as z:
            missing = [f for f in _FIELDS if f not in z.files]
            if missing:
                raise ValueError(f"TSQ npz 에 필드가 없다: {missing}")
            return cls(**{f: torch.from_numpy(np.array(z[f])) for f in _FIELDS})

    @classmethod
    def from_surfels(cls, s: Surfels, *, edge_n=2.0) -> "TSQ":
        if not isinstance(s, Surfels):
            raise TypeError("from_surfels: Surfels 가 필요하다")
        en = _real("edge_n", edge_n)
        pages = s.colors[:, None, None, :].expand(s.n, PAGE, PAGE, 3).clone()
        return cls(means=s.means, quats=s.quats, log_scales=s.log_scales, opacity_logits=s.opacity_logits,
                   edge_n=torch.full((s.n,), en, dtype=s.dtype, device=s.device), pages=pages)

    def to_surfels(self) -> Surfels:
        return Surfels(means=self.means, quats=self.quats, log_scales=self.log_scales,
                       opacity_logits=self.opacity_logits, colors=self.pages.mean(dim=(1, 2)))


# ---------------------------------------------------------------------------------------------------------------
# 옵션 · 결과
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class TSQOpts:
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
class TSQRender:
    rgb: torch.Tensor       # (H,W,3)
    alpha: torch.Tensor     # (H,W)
    depth: torch.Tensor     # (H,W) 카메라 z, 알파<1e-3 은 +inf
    normal: torch.Tensor    # (H,W,3) 세계 단위 노멀(카메라 쪽), 알파<1e-3 은 0
    backend: str = "torch"

    def numpy(self) -> dict:
        return {k: getattr(self, k).detach().cpu().numpy().astype(np.float32)
                for k in ("rgb", "alpha", "depth", "normal")}


# ---------------------------------------------------------------------------------------------------------------
# CUDA 확장 (지연 JIT)
# ---------------------------------------------------------------------------------------------------------------
_EXT = {"mod": None, "err": None}
_EXT_LOCK = threading.Lock()
_KERNEL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsq_kernel.cu")


def load_extension():
    """`tsq_kernel.cu` 를 JIT 컴파일·로드한다. CUDA·nvcc 없음/컴파일 실패 → BackendUnavailable."""
    with _EXT_LOCK:
        if _EXT["mod"] is not None:
            return _EXT["mod"]
        if _EXT["err"] is not None:
            raise BackendUnavailable(_EXT["err"])
        try:
            if not torch.cuda.is_available():
                raise BackendUnavailable("CUDA 장치가 없다")
            from torch.utils import cpp_extension
            if cpp_extension.CUDA_HOME is None:
                raise BackendUnavailable("CUDA 툴킷(nvcc)을 찾을 수 없다")
            if not os.path.exists(_KERNEL_SRC):
                raise BackendUnavailable(f"커널 소스가 없다: {_KERNEL_SRC}")
            mod = cpp_extension.load(name="buc_tsq_kernel", sources=[_KERNEL_SRC],
                                     extra_cuda_cflags=["-O3"], verbose=False)
            for fn in ("forward", "backward"):
                if not hasattr(mod, fn):
                    raise BackendUnavailable(f"확장에 {fn} 이 없다")
        except BackendUnavailable as e:
            _EXT["err"] = str(e)
            raise
        except BaseException as e:  # noqa: BLE001  (컴파일 실패, SystemExit, ImportError ...)
            if isinstance(e, KeyboardInterrupt):
                raise
            _EXT["err"] = f"TSQ CUDA 확장 로드 실패: {type(e).__name__}: {e}"
            raise BackendUnavailable(_EXT["err"]) from None
        _EXT["mod"] = mod
        return mod


def _dev_type(device) -> str:
    if isinstance(device, torch.Tensor):
        device = device.device
    return torch.device(device).type


def resolve_backend(name: str, device) -> str:
    """`name` 을 실제 백엔드("torch"|"cuda")로. "auto" 는 env BUC_TSQ_BACKEND 우선, CUDA 장치 + 확장 로드 성공이면
    "cuda", 아니면 "torch"."""
    if name not in _BACKENDS:
        raise ValueError(f"backend: {_BACKENDS} 중 하나 ({name!r})")
    on_cuda = _dev_type(device) == "cuda"
    if name == "torch":
        return "torch"
    if name == "cuda":
        if not on_cuda:
            raise BackendUnavailable("cuda 백엔드는 CUDA 텐서만 지원한다")
        load_extension()
        return "cuda"
    env = os.environ.get("BUC_TSQ_BACKEND", "").strip().lower()
    if env not in ("", "auto", "torch", "cuda"):
        raise ValueError(f"BUC_TSQ_BACKEND: torch|cuda|auto 중 하나 ({env!r})")
    if env == "torch":
        return "torch"
    if on_cuda:
        try:
            load_extension()
            return "cuda"
        except BackendUnavailable as e:
            warnings.warn(f"TSQ CUDA 확장을 쓸 수 없어 torch 백엔드로 폴백한다: {e}", RuntimeWarning, stacklevel=2)
    elif env == "cuda":
        warnings.warn("BUC_TSQ_BACKEND=cuda 이지만 CPU 텐서라 torch 백엔드를 쓴다", RuntimeWarning, stacklevel=2)
    return "torch"


# ---------------------------------------------------------------------------------------------------------------
# 미분 기본 연산
# ---------------------------------------------------------------------------------------------------------------
def _spow(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """|x|^n (n ≥ 0 가정). x=0 -> 0 (기울기 0), 지수 상한으로 오버플로 없음."""
    a = torch.abs(x)
    pos = a > 0
    la = torch.log(torch.where(pos, a, torch.ones_like(a)))
    L = torch.clamp(n * la, max=_LOGPOW_MAX)
    return torch.where(pos, torch.exp(L), torch.zeros_like(L))


def edge_falloff(u, v, n) -> torch.Tensor:
    """exp(-½(|u|^n + |v|^n)), 브로드캐스트. u=0 · 큰 n 에서도 기울기 유한."""
    u, v, n = (x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in (u, v, n))
    return torch.exp(-0.5 * (_spow(u, n) + _spow(v, n)))


def _bilerp(table: torch.Tensor, base: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """table (N*64, C) 평탄 페이지, base = 페이지 번호*64 (u 와 브로드캐스트). -> (..., C)."""
    x = torch.clamp(4.0 * (u + 1.0) - 0.5, 0.0, PAGE - 1.0)
    y = torch.clamp(4.0 * (v + 1.0) - 0.5, 0.0, PAGE - 1.0)
    x0 = torch.clamp(torch.floor(x.detach()), max=PAGE - 2.0)
    y0 = torch.clamp(torch.floor(y.detach()), max=PAGE - 2.0)
    fx = (x - x0).unsqueeze(-1)
    fy = (y - y0).unsqueeze(-1)
    i00 = base + (y0 * PAGE + x0).long()
    c00 = table[i00]
    c01 = table[i00 + 1]
    c10 = table[i00 + PAGE]
    c11 = table[i00 + PAGE + 1]
    return ((1 - fx) * (1 - fy) * c00 + fx * (1 - fy) * c01 + (1 - fx) * fy * c10 + fx * fy * c11)


def sample_page(pages, idx, uv) -> torch.Tensor:
    """pages (N,8,8,C), idx (M,) long, uv (M,2) -> (M,C). 바이리니어, 클램프 주소."""
    if not isinstance(pages, torch.Tensor):
        pages = torch.as_tensor(np.asarray(pages))
    if pages.ndim != 4 or pages.shape[1:3] != (PAGE, PAGE):
        raise ValueError(f"pages: (N,8,8,C) 여야 한다 ({tuple(pages.shape)})")
    idx = torch.as_tensor(idx, device=pages.device).long().reshape(-1)
    uv = torch.as_tensor(uv, device=pages.device)
    if uv.shape != (idx.shape[0], 2):
        raise ValueError(f"uv: ({idx.shape[0]},2) 여야 한다 ({tuple(uv.shape)})")
    if not uv.is_floating_point():
        uv = uv.to(pages.dtype)
    N, C = pages.shape[0], pages.shape[3]
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= N):
        raise ValueError("idx: 페이지 범위 밖")
    table = pages.reshape(N * PAGE * PAGE, C)
    return _bilerp(table, idx * (PAGE * PAGE), uv[:, 0], uv[:, 1])


# ---------------------------------------------------------------------------------------------------------------
# 준비: 컬링·정렬·카메라 좌표 특징 (두 백엔드 공용, 미분 가능)
# ---------------------------------------------------------------------------------------------------------------
def _prepare(t: TSQ, w2c_R, w2c_t, Km, opts: TSQOpts):
    dt, dev = t.dtype, t.device
    Rw = torch.as_tensor(w2c_R, dtype=dt, device=dev)
    tw = torch.as_tensor(w2c_t, dtype=dt, device=dev)
    fx, fy, skew, cx, cy = (float(Km[0, 0]), float(Km[1, 1]), float(Km[0, 1]), float(Km[0, 2]), float(Km[1, 2]))
    pc_all = t.means @ Rw.T + tw
    z_all = pc_all[:, 2]
    keep = (z_all >= opts.near) & (z_all <= opts.far)
    idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return None
    order = torch.sort(z_all.detach()[idx], stable=True).indices
    idx = idx[order]
    R_world = _quat_to_rot(t.quats[idx])                      # (M,3,3) 열 = (t_u, t_v, n)
    Rcam = Rw @ R_world
    pc = pc_all[idx]
    zc = pc[:, 2]
    tu, tv, nc = Rcam[:, :, 0], Rcam[:, :, 1], Rcam[:, :, 2]
    su, sv = torch.exp(t.log_scales[idx]).unbind(-1)
    numer = (nc * pc).sum(-1)
    sign = torch.where(numer.detach() > 0, -torch.ones_like(numer), torch.ones_like(numer))
    return dict(
        idx=idx, tu=tu, tv=tv, nc=nc, numer=numer, tu_p=(tu * pc).sum(-1), tv_p=(tv * pc).sum(-1), su=su, sv=sv,
        op=torch.sigmoid(t.opacity_logits[idx]), n=torch.clamp(t.edge_n[idx], min=2.0),
        cu=(fx * pc[:, 0] + skew * pc[:, 1]) / zc + cx, cv=fy * pc[:, 1] / zc + cy, zc=zc,
        nw=R_world[:, :, 2] * sign[:, None], pages=t.pages[idx], cam=(fx, fy, skew, cx, cy))


def _background(P, dt, dev, bg, anchor):
    bgt = torch.tensor(bg, dtype=dt, device=dev)
    rgb = bgt.expand(P, 3).clone()
    zero = None
    if anchor is not None and any(p.requires_grad for p in anchor.params()):
        zero = sum((p.sum() for p in anchor.params())) * 0.0
        rgb = rgb + zero
    alpha = torch.zeros(P, dtype=dt, device=dev) if zero is None else zero.expand(P).clone()
    depth = torch.full((P,), float("inf"), dtype=dt, device=dev)
    normal = torch.zeros(P, 3, dtype=dt, device=dev)
    return dict(rgb=rgb, alpha=alpha, depth=depth, normal=normal)


def _finish(rgb_acc, alpha, dnum, nsum, bgt):
    rgb = rgb_acc + (1.0 - alpha)[:, None] * bgt
    low = alpha.detach() < _ALPHA_OUT_EPS
    depth = torch.where(low, torch.full_like(alpha, float("inf")),
                        dnum / torch.where(low, torch.ones_like(alpha), alpha))
    nn = torch.sqrt((nsum * nsum).sum(-1, keepdim=True).clamp_min(1e-30))
    normal = torch.where(low[:, None], torch.zeros_like(nsum), nsum / nn)
    return rgb, depth, normal


# ---------------------------------------------------------------------------------------------------------------
# torch 정본 래스터
# ---------------------------------------------------------------------------------------------------------------
def _render_torch(t: TSQ, pre, W, H, opts: TSQOpts):
    dt, dev = t.dtype, t.device
    P = H * W
    fx, fy, skew, cx, cy = pre["cam"]
    tu, tv, nc = pre["tu"], pre["tv"], pre["nc"]
    numer, tu_p, tv_p = pre["numer"], pre["tu_p"], pre["tv_p"]
    su, sv, op, nn_, cu, cv, zc, nw = (pre[k] for k in ("su", "sv", "op", "n", "cu", "cv", "zc", "nw"))
    M = int(pre["idx"].numel())
    table = pre["pages"].reshape(M * PAGE * PAGE, 3)
    base = (torch.arange(M, device=dev) * (PAGE * PAGE))[:, None]
    bgt = torch.tensor(opts.bg, dtype=dt, device=dev)
    pix = torch.arange(P, device=dev)
    step = max(32, _CHUNK_ELEMS // M)
    outs = {k: [] for k in ("rgb", "alpha", "depth", "normal")}
    for a in range(0, P, step):
        pp = pix[a:a + step]
        uu = (pp % W).to(dt)
        vv = torch.div(pp, W, rounding_mode="floor").to(dt)
        yd = (vv - cy) / fy
        xd = (uu - cx - skew * yd) / fx
        Dm = torch.stack([xd, yd, torch.ones_like(xd)], 0)     # (3,p)
        d2 = (uu[None, :] - cu[:, None]) ** 2 + (vv[None, :] - cv[:, None]) ** 2
        Dn = nc @ Dm                                            # (M,p)
        ok = Dn.detach().abs() > _DENOM_EPS
        tt = numer[:, None] / torch.where(ok, Dn, torch.ones_like(Dn))
        ok = ok & (tt.detach() > 0)
        u = (tt * (tu @ Dm) - tu_p[:, None]) / su[:, None]
        v = (tt * (tv @ Dm) - tv_p[:, None]) / sv[:, None]
        n_ = nn_[:, None]
        E = 0.5 * (_spow(u, n_) + _spow(v, n_))
        E = torch.where(ok, E, torch.full_like(E, float("inf")))
        A = torch.clamp(op[:, None] * torch.exp(-torch.minimum(E, d2)), max=_ALPHA_MAX)
        A = torch.where(A >= _ALPHA_MIN, A, torch.zeros_like(A))
        zero = torch.zeros_like(u)
        col = _bilerp(table, base, torch.where(ok, u, zero), torch.where(ok, v, zero))   # (M,p,3)
        logt = torch.log1p(-A)
        cum = torch.cumsum(logt, dim=0) - logt
        w = A * torch.exp(cum)
        alpha = w.sum(0)
        rgb_acc = torch.einsum("mp,mpc->pc", w, col)
        zi = torch.where(ok, tt, zc[:, None].expand_as(tt))
        dnum = (w * zi).sum(0)
        nsum = w.T @ nw
        rgb, depth, normal = _finish(rgb_acc, alpha, dnum, nsum, bgt)
        outs["rgb"].append(rgb)
        outs["alpha"].append(alpha)
        outs["depth"].append(depth)
        outs["normal"].append(normal)
    return {k: torch.cat(v, 0) for k, v in outs.items()}


# ---------------------------------------------------------------------------------------------------------------
# cuda 백엔드 (tsq_kernel.cu, float32)
# ---------------------------------------------------------------------------------------------------------------
class _TSQRasterFn(torch.autograd.Function):
    """G (M,22) 카메라 좌표 기하 특징, pages (M,64,3) -> out (P,8) = [rgb_acc(3), alpha, dnum, nsum(3)]."""

    @staticmethod
    def forward(ctx, G, pages, W, H, cam):
        mod = load_extension()
        fx, fy, skew, cx, cy = cam
        out = mod.forward(G, pages, int(W), int(H), fx, fy, skew, cx, cy)[0]
        ctx.save_for_backward(G, pages, out)
        ctx.meta = (int(W), int(H), cam)
        return out

    @staticmethod
    def backward(ctx, g_out):
        G, pages, out = ctx.saved_tensors
        W, H, cam = ctx.meta
        fx, fy, skew, cx, cy = cam
        mod = load_extension()
        gG, gP = mod.backward(G, pages, out, g_out.contiguous(), W, H, fx, fy, skew, cx, cy)
        return gG, gP, None, None, None


def _render_cuda(t: TSQ, pre, W, H, opts: TSQOpts):
    dt, dev = t.dtype, t.device
    if dev.type != "cuda":
        raise BackendUnavailable("cuda 백엔드는 CUDA 텐서만 지원한다")
    if dt != torch.float32:
        raise BackendUnavailable("cuda 백엔드는 float32 만 지원한다")
    load_extension()
    M = int(pre["idx"].numel())
    G = torch.cat([pre["tu"], pre["tv"], pre["nc"], pre["numer"][:, None], pre["tu_p"][:, None],
                   pre["tv_p"][:, None], pre["su"][:, None], pre["sv"][:, None], pre["op"][:, None],
                   pre["n"][:, None], pre["cu"][:, None], pre["cv"][:, None], pre["zc"][:, None], pre["nw"]],
                  dim=1).contiguous()
    assert G.shape == (M, _NG)
    pages = pre["pages"].reshape(M, PAGE * PAGE, 3).contiguous()
    try:
        out = _TSQRasterFn.apply(G, pages, W, H, tuple(float(c) for c in pre["cam"]))
    except BackendUnavailable:
        raise
    except Exception as e:  # noqa: BLE001  (커널 실행 오류)
        raise BackendUnavailable(f"TSQ CUDA 커널 실행 실패: {type(e).__name__}: {e}") from e
    bgt = torch.tensor(opts.bg, dtype=dt, device=dev)
    rgb, depth, normal = _finish(out[:, 0:3], out[:, 3], out[:, 4], out[:, 5:8], bgt)
    return dict(rgb=rgb, alpha=out[:, 3], depth=depth, normal=normal)


# ---------------------------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------------------------
def render(tsq: TSQ, *, c2w, K, width, height, opts: TSQOpts | None = None) -> TSQRender:
    """TSQ 를 한 카메라로 찍는다(선형 RGB·알파·z-깊이·세계 노멀). 모든 params() 에 미분 가능."""
    if not isinstance(tsq, TSQ):
        raise TypeError("tsq: TSQ 여야 한다")
    opts = TSQOpts() if opts is None else opts
    if not isinstance(opts, TSQOpts):
        raise TypeError("opts: TSQOpts 여야 한다")
    _c, Km, w2c_R, w2c_t, W, H = _camera(c2w, K, width, height)
    want = resolve_backend(opts.backend, tsq.device)
    explicit = opts.backend == "cuda"
    if want == "cuda" and tsq.dtype != torch.float32:
        if explicit:
            raise BackendUnavailable("cuda 백엔드는 float32 만 지원한다")
        want = "torch"
    P = H * W
    pre = _prepare(tsq, w2c_R, w2c_t, Km, opts) if tsq.n else None
    if pre is None:
        out, be = _background(P, tsq.dtype, tsq.device, opts.bg, tsq if tsq.n else None), want
    else:
        out = None
        be = "torch"
        if want == "cuda":
            try:
                out, be = _render_cuda(tsq, pre, W, H, opts), "cuda"
            except BackendUnavailable as e:
                if explicit:
                    raise
                warnings.warn(f"TSQ cuda 실패, torch 백엔드로 폴백한다: {e}", RuntimeWarning, stacklevel=2)
        if out is None:
            out = _render_torch(tsq, pre, W, H, opts)
    return TSQRender(rgb=out["rgb"].reshape(H, W, 3), alpha=out["alpha"].reshape(H, W),
                     depth=out["depth"].reshape(H, W), normal=out["normal"].reshape(H, W, 3), backend=be)


# ---------------------------------------------------------------------------------------------------------------
# BC 코덱 (numpy, CPU, 결정론)
# ---------------------------------------------------------------------------------------------------------------
_BC7_W4 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], np.int64)


def _check_img(img, channels, name):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError(f"{name}: uint8 배열이어야 한다")
    if img.ndim != 3 or img.shape[2] != channels:
        raise ValueError(f"{name}: (H,W,{channels}) 여야 한다 ({img.shape})")
    h, w = img.shape[:2]
    if h < 4 or w < 4 or h % 4 or w % 4:
        raise ValueError(f"{name}: H, W 는 4 의 배수여야 한다 ({h}x{w})")
    return img


def _check_size(data, width, height, bpb):
    for nm, v in (("width", width), ("height", height)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 4 or v % 4:
            raise ValueError(f"{nm}: 4 의 배수인 양의 정수여야 한다 ({v!r})")
    try:
        data = bytes(data)
    except TypeError as e:
        raise ValueError(f"data: 바이트열이어야 한다 ({e})") from e
    nb = (int(width) // 4) * (int(height) // 4)
    if len(data) != nb * bpb:
        raise ValueError(f"data: 길이 {len(data)} != {nb * bpb}")
    return np.frombuffer(data, np.uint8).reshape(nb, bpb), int(width), int(height)


def _to_blocks(img):
    """(H,W,C) -> (B,16,C) 블록 래스터 순서, 블록 안 화소 행 우선."""
    h, w, c = img.shape
    return img.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)


def _from_blocks(blk, h, w):
    c = blk.shape[-1]
    return blk.reshape(h // 4, w // 4, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def _put(bits, pos, val, nb):
    for b in range(nb):
        bits[:, pos + b] = (val >> b) & 1


def _get(bits, pos, nb):
    v = np.zeros(bits.shape[0], np.int64)
    for b in range(nb):
        v |= bits[:, pos + b].astype(np.int64) << b
    return v


# -- BC7 모드 6 -------------------------------------------------------------------------------------------------
def _bc7_palette(e0, e1):
    """e0, e1 (B,4) int -> (B,16,4) int."""
    w = _BC7_W4[None, :, None]
    return ((64 - w) * e0[:, None, :] + w * e1[:, None, :] + 32) >> 6


def _bc7_assign(X, e0, e1, wch):
    pal = _bc7_palette(e0, e1).astype(np.float64)                         # (B,16,4)
    d = (((X[:, :, None, :] - pal[:, None, :, :]) ** 2) * wch).sum(-1)    # (B,16pix,16pal)
    idx = np.argmin(d, axis=2)
    err = np.take_along_axis(d, idx[..., None], 2)[..., 0].sum(1)
    return idx, err


def _bc7_quant(e, p):
    """연속 끝점 (B,4) -> 7 비트 k, 8 비트 값 (패리티 p (B,))."""
    k = np.clip(np.rint((e - p[:, None]) / 2.0), 0, 127).astype(np.int64)
    return k, 2 * k + p[:, None]


def _bc7_try(X, lo, hi, wch, best):
    B = X.shape[0]
    for p0 in (0, 1):
        for p1 in (0, 1):
            P0 = np.full(B, p0, np.int64)
            P1 = np.full(B, p1, np.int64)
            k0, e0 = _bc7_quant(lo, P0)
            k1, e1 = _bc7_quant(hi, P1)
            idx, err = _bc7_assign(X, e0, e1, wch)
            better = err < best["err"]
            if better.any():
                for key, val in (("err", err), ("k0", k0), ("k1", k1), ("p0", P0), ("p1", P1), ("idx", idx)):
                    best[key][better] = val[better]


def _bc7_encode_blocks(X, wch):
    """X (B,16,4) float64 [0,255] -> (B,16) uint8 모드 6 블록."""
    B = X.shape[0]
    m = X.mean(1)
    Xc = X - m[:, None, :]
    cov = np.einsum("bpi,bpj->bij", Xc, Xc)
    _, vecs = np.linalg.eigh(cov)
    ax = vecs[:, :, -1]
    proj = np.einsum("bpi,bi->bp", Xc, ax)
    lo = m + proj.min(1)[:, None] * ax
    hi = m + proj.max(1)[:, None] * ax
    best = dict(err=np.full(B, np.inf), k0=np.zeros((B, 4), np.int64), k1=np.zeros((B, 4), np.int64),
                p0=np.zeros(B, np.int64), p1=np.zeros(B, np.int64), idx=np.zeros((B, 16), np.int64))
    # 후보: 주축 극단, 바운딩 박스, 상수(평균)
    _bc7_try(X, np.clip(lo, 0, 255), np.clip(hi, 0, 255), wch, best)
    _bc7_try(X, X.min(1), X.max(1), wch, best)
    _bc7_try(X, m, m, wch, best)
    # 최소제곱 끝점 정제 (지수 고정)
    for _ in range(3):
        a = _BC7_W4[best["idx"]] / 64.0                                 # (B,16)
        b_ = 1.0 - a
        s00 = (b_ * b_).sum(1)
        s01 = (a * b_).sum(1)
        s11 = (a * a).sum(1)
        r0 = np.einsum("bp,bpc->bc", b_, X)
        r1 = np.einsum("bp,bpc->bc", a, X)
        det = s00 * s11 - s01 * s01
        okd = np.abs(det) > 1e-9
        dets = np.where(okd, det, 1.0)[:, None]
        e0 = (s11[:, None] * r0 - s01[:, None] * r1) / dets
        e1 = (s00[:, None] * r1 - s01[:, None] * r0) / dets
        e0 = np.where(okd[:, None], e0, m)
        e1 = np.where(okd[:, None], e1, m)
        _bc7_try(X, np.clip(e0, 0, 255), np.clip(e1, 0, 255), wch, best)
    k0, k1, p0, p1, idx = best["k0"], best["k1"], best["p0"], best["p1"], best["idx"]
    # 앵커(화소 0) 지수 MSB = 0: 끝점 교환 + 지수 반전 (가중치 표가 대칭이라 정확)
    sw = idx[:, 0] >= 8
    k0, k1 = np.where(sw[:, None], k1, k0), np.where(sw[:, None], k0, k1)
    p0, p1 = np.where(sw, p1, p0), np.where(sw, p0, p1)
    idx = np.where(sw[:, None], 15 - idx, idx)
    bits = np.zeros((B, 128), np.uint8)
    bits[:, 6] = 1
    pos = 7
    for c in range(4):
        _put(bits, pos, k0[:, c], 7)
        _put(bits, pos + 7, k1[:, c], 7)
        pos += 14
    bits[:, 63] = p0
    bits[:, 64] = p1
    pos = 65
    for i in range(16):
        nb = 3 if i == 0 else 4
        _put(bits, pos, idx[:, i], nb)
        pos += nb
    return np.packbits(bits, axis=1, bitorder="little")


def encode_bc7(img) -> bytes:
    """uint8 (H,W,4) RGBA -> BC7 (모드 6 만). H, W 는 4 의 배수."""
    img = _check_img(img, 4, "img")
    X = _to_blocks(img).astype(np.float64)
    return _bc7_encode_blocks(X, np.ones(4)).tobytes()


def decode_bc7(data, width, height) -> np.ndarray:
    """BC7 -> uint8 (H,W,4). 모드 6 은 비트 정확, 예약 모드(바이트0=0)는 0, 그 밖의 모드는 ValueError."""
    raw, w, h = _check_size(data, width, height, 16)
    b0 = raw[:, 0]
    reserved = b0 == 0
    mode6 = (b0 & 0x7F) == 0x40
    bad = ~(reserved | mode6)
    if bad.any():
        k = int(np.nonzero(bad)[0][0])
        raise ValueError(f"BC7 블록 {k}: 지원하지 않는 모드 (byte0={int(b0[k]):#04x}), 모드 6 만 지원")
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    p0 = _get(bits, 63, 1)
    p1 = _get(bits, 64, 1)
    e0 = np.stack([(_get(bits, 7 + 14 * c, 7) << 1) | p0 for c in range(4)], 1)
    e1 = np.stack([(_get(bits, 14 + 14 * c, 7) << 1) | p1 for c in range(4)], 1)
    idx = np.zeros((raw.shape[0], 16), np.int64)
    pos = 65
    for i in range(16):
        nb = 3 if i == 0 else 4
        idx[:, i] = _get(bits, pos, nb)
        pos += nb
    w_ = _BC7_W4[idx][..., None]
    out = ((64 - w_) * e0[:, None, :] + w_ * e1[:, None, :] + 32) >> 6
    out[reserved] = 0
    return _from_blocks(out.astype(np.uint8), h, w)


# -- BC4 / BC5 --------------------------------------------------------------------------------------------------
def _bc4_palette(r0, r1):
    """r0, r1 (B,) int -> (B,8) float64 (0..255), D3D 부동 팔레트."""
    r0f, r1f = r0.astype(np.float64)[:, None], r1.astype(np.float64)[:, None]
    k7 = np.arange(1, 7, dtype=np.float64)[None]
    k5 = np.arange(1, 5, dtype=np.float64)[None]
    p8 = np.concatenate([r0f, r1f, ((7 - k7) * r0f + k7 * r1f) / 7.0], 1)
    p6 = np.concatenate([r0f, r1f, ((5 - k5) * r0f + k5 * r1f) / 5.0, np.zeros_like(r0f),
                         np.full_like(r0f, 255.0)], 1)
    return np.where((r0 > r1)[:, None], p8, p6)


def _bc4_eval(X, r0, r1):
    pal = _bc4_palette(r0, r1)
    d = (X[:, :, None] - pal[:, None, :]) ** 2
    idx = np.argmin(d, 2)
    return idx, np.take_along_axis(d, idx[..., None], 2)[..., 0].sum(1)


def _bc4_encode_blocks(X):
    """X (B,16) float64 [0,255] -> (B,8) uint8."""
    B = X.shape[0]
    mx = np.rint(X.max(1)).astype(np.int64)
    mn = np.rint(X.min(1)).astype(np.int64)
    best_r0, best_r1 = mx.copy(), mn.copy()
    best_idx, best_err = _bc4_eval(X, best_r0, best_r1)

    def consider(r0, r1):
        nonlocal best_r0, best_r1, best_idx, best_err
        r0 = np.clip(r0, 0, 255).astype(np.int64)
        r1 = np.clip(r1, 0, 255).astype(np.int64)
        idx, err = _bc4_eval(X, r0, r1)
        b = err < best_err
        best_r0 = np.where(b, r0, best_r0)
        best_r1 = np.where(b, r1, best_r1)
        best_idx = np.where(b[:, None], idx, best_idx)
        best_err = np.where(b, err, best_err)

    # 6분할 모드(r0 <= r1)도 후보로
    consider(mn, mx)
    # 최소제곱 정제 (8분할: 지수 -> 보간 계수)
    frac8 = np.array([0, 7, 1, 2, 3, 4, 5, 6], np.float64) / 7.0          # 팔레트 지수 -> r1 비중
    for _ in range(3):
        mode8 = best_r0 > best_r1
        a = frac8[best_idx]
        b_ = 1.0 - a
        s00, s01, s11 = (b_ * b_).sum(1), (a * b_).sum(1), (a * a).sum(1)
        q0, q1 = (b_ * X).sum(1), (a * X).sum(1)
        det = s00 * s11 - s01 * s01
        okd = (np.abs(det) > 1e-9) & mode8
        dets = np.where(okd, det, 1.0)
        e0 = np.where(okd, (s11 * q0 - s01 * q1) / dets, best_r0)
        e1 = np.where(okd, (s00 * q1 - s01 * q0) / dets, best_r1)
        for d0 in (0, -1, 1):
            for d1 in (0, -1, 1):
                r0 = np.rint(e0).astype(np.int64) + d0
                r1 = np.rint(e1).astype(np.int64) + d1
                r0, r1 = np.maximum(r0, r1), np.minimum(r0, r1)
                r0 = np.where(r0 == r1, np.minimum(r0 + 1, 255), r0)
                r1 = np.where(r0 == r1, r1 - 1, r1)
                consider(r0, r1)
    out = np.zeros((B, 8), np.uint8)
    out[:, 0] = best_r0
    out[:, 1] = best_r1
    bits = np.zeros((B, 48), np.uint8)
    for i in range(16):
        _put(bits, 3 * i, best_idx[:, i], 3)
    out[:, 2:] = np.packbits(bits, axis=1, bitorder="little")
    return out


def _bc4_decode_blocks(raw8):
    r0 = raw8[:, 0].astype(np.int64)
    r1 = raw8[:, 1].astype(np.int64)
    pal = _bc4_palette(r0, r1)
    bits = np.unpackbits(raw8[:, 2:], axis=1, bitorder="little")
    idx = np.stack([_get(bits, 3 * i, 3) for i in range(16)], 1)
    return np.take_along_axis(pal, idx, 1) / 255.0


def encode_bc5(img) -> bytes:
    """uint8 (H,W,2) -> BC5 (BC4(R) 8 B + BC4(G) 8 B 블록)."""
    img = _check_img(img, 2, "img")
    X = _to_blocks(img).astype(np.float64)
    out = np.concatenate([_bc4_encode_blocks(X[:, :, 0]), _bc4_encode_blocks(X[:, :, 1])], 1)
    return out.tobytes()


def decode_bc5(data, width, height) -> np.ndarray:
    """BC5 -> float32 (H,W,2) ∈ [0,1] (D3D 부동 팔레트 / 255)."""
    raw, w, h = _check_size(data, width, height, 16)
    r = _bc4_decode_blocks(raw[:, 0:8])
    g = _bc4_decode_blocks(raw[:, 8:16])
    return _from_blocks(np.stack([r, g], -1), h, w).astype(np.float32)


# -- 페이지 코덱 -------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PageBlob:
    n: int
    lo: np.ndarray          # float32 (N,) log2 하한
    hi: np.ndarray          # float32 (N,) log2 상한
    bc7: bytes              # 64 N 바이트, 페이지 p = bc7[64p:64p+64]
    bc5: bytes | None = None  # 64 N 바이트 (노멀/높이) 또는 None

    @property
    def nbytes(self) -> int:
        return len(self.bc7) + (len(self.bc5) if self.bc5 is not None else 0)


def _as_np64(x, name):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 수치 배열이어야 한다 ({e})") from e
    return a


def compress_pages(pages, nh=None) -> PageBlob:
    """pages (N,8,8,3) ≥0 선형 -> BC7 로그 코드 (64 B/페이지). nh (N,8,8,2) ∈[-1,1] -> BC5 (64 B/페이지)."""
    p = _as_np64(pages, "pages")
    if p.ndim != 4 or p.shape[1:] != (PAGE, PAGE, 3):
        raise ValueError(f"pages: (N,8,8,3) 여야 한다 ({p.shape})")
    if not np.isfinite(p).all():
        raise ValueError("pages: 비유한 값")
    N = p.shape[0]
    y = np.log2(np.maximum(p, LMIN))
    yf = y.reshape(N, -1)
    lo = (yf.min(1) if N else np.zeros(0)).astype(np.float32)
    hi = (np.maximum(yf.max(1), lo.astype(np.float64) + 1e-3) if N else np.zeros(0)).astype(np.float32)
    L64, H64 = lo.astype(np.float64)[:, None, None, None], hi.astype(np.float64)[:, None, None, None]
    code = np.clip(np.rint(255.0 * (y - L64) / (H64 - L64)), 0, 255).astype(np.uint8)
    bc5 = None
    if nh is not None:
        q = _as_np64(nh, "nh")
        if q.shape != (N, PAGE, PAGE, 2):
            raise ValueError(f"nh: ({N},8,8,2) 여야 한다 ({q.shape})")
        if not np.isfinite(q).all():
            raise ValueError("nh: 비유한 값")
        qc = np.clip(np.rint((np.clip(q, -1.0, 1.0) + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
        bc5 = encode_bc5(qc.reshape(N * PAGE, PAGE, 2)) if N else b""
    if N:
        img = np.concatenate([code.reshape(N * PAGE, PAGE, 3), np.full((N * PAGE, PAGE, 1), 255, np.uint8)], -1)
        X = _to_blocks(img).astype(np.float64)
        bc7 = _bc7_encode_blocks(X, np.array([1.0, 1.0, 1.0, 0.25])).tobytes()
    else:
        bc7 = b""
    return PageBlob(n=int(N), lo=lo, hi=hi, bc7=bc7, bc5=bc5)


def _check_blob(blob):
    if not isinstance(blob, PageBlob):
        raise TypeError("blob: PageBlob 이어야 한다")
    if len(blob.bc7) != 64 * blob.n:
        raise ValueError("blob.bc7: 길이 불일치")


def decompress_pages(blob: PageBlob) -> np.ndarray:
    """PageBlob -> float32 (N,8,8,3) 선형. L = 2^(lo + c/255·(hi-lo))."""
    _check_blob(blob)
    N = blob.n
    if N == 0:
        return np.zeros((0, PAGE, PAGE, 3), np.float32)
    codes = decode_bc7(blob.bc7, PAGE, PAGE * N)[..., :3].reshape(N, PAGE, PAGE, 3).astype(np.float64)
    L64 = np.asarray(blob.lo, np.float32).astype(np.float64)[:, None, None, None]
    H64 = np.asarray(blob.hi, np.float32).astype(np.float64)[:, None, None, None]
    return np.exp2(L64 + codes / 255.0 * (H64 - L64)).astype(np.float32)


def decompress_nh(blob: PageBlob):
    """PageBlob -> float32 (N,8,8,2) ∈[-1,1] 또는 None."""
    _check_blob(blob)
    if blob.bc5 is None:
        return None
    N = blob.n
    if N == 0:
        return np.zeros((0, PAGE, PAGE, 2), np.float32)
    d = decode_bc5(blob.bc5, PAGE, PAGE * N).astype(np.float64).reshape(N, PAGE, PAGE, 2)
    return (2.0 * d - 1.0).astype(np.float32)

