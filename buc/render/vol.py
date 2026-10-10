"""RDR-08 STRAND/SGGX 체적 v0: 털·잎·솜털을 싣는 그릇. 매팅 알파를 체적 밀도로 바꾸는 그릇이기도 하다.

계약서: ops/harness/contracts/RDR-08.md. 세계 오른손·미터, 카메라 OpenCV, `c2w` (4,4) 카메라->세계, 화소 중심 정수.
색은 선형, 직선 알파. 순수 PyTorch(CPU 정본, autograd), numpy / scipy / torch 만 쓴다. import 시 gsplat import·CUDA
초기화·전역 RNG 변경 없음.

구성
  * SGGX 미세플레이크 수학: `sggx_fiber / sggx_surface / sggx_sigma / sggx_D / sggx_phase_specular`.
  * `VolGaussians`: SGGX 비등방 소광을 갖는 3D 가우시안 체적. 화소 광선 하나가 가우시안 하나를 지나는 직선 적분이
    광학 깊이 `tau = rho * sigma_S(w) * sqrt(2 pi / a) * exp(-m^2 / 2)` (닫힌 형).
  * `Strands`: 화면 공간 가닥 래스터(선분 = 캡슐, 가닥당 화소당 조각 하나).
  * `render_vol`: 가우시안·가닥·backdrop 조각을 화소별 깊이 정렬해 over 합성(RDR-01과 같은 규칙).
  * 변환: `alpha_to_tau / tau_to_alpha / matte_to_vol / strands_to_vol / scatter_colors`.

구현 메모
  * 렌더는 카메라 좌표계에서 계산한다(공분산 역행렬·SGGX 를 카메라 좌표로 한 번 옮긴다). 화소 덩어리마다
    (조각 x 화소) 밀집 텐서를 만든다. 조각 선택(argmax·정렬 색인·마스크)만 기울기가 없고 나머지는 모두 미분된다.
  * `strands_to_vol` 은 means / log_scales / log_density / colors 로는 기울기가 흐르지만 자세(quats)·sggx 는 접선을
    분리(detach)해서 만든다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import torch

from buc.core import geom

__all__ = ["VolGaussians", "Strands", "VolOpts", "VolRender", "render_vol", "sggx_fiber", "sggx_surface",
           "sggx_sigma", "sggx_D", "sggx_phase_specular", "alpha_to_tau", "tau_to_alpha", "matte_to_vol",
           "strands_to_vol", "scatter_colors"]

_TWO_PI = 2.0 * math.pi
_SQRT2 = math.sqrt(2.0)
_ALPHA_OUT_EPS = 1e-3
_CHUNK_ELEMS = 1 << 19          # (조각 x 화소) 덩어리 크기 상한(원소 수)
_SEG_EPS2 = 1e-12               # 퇴화 선분 문턱 |b-a|^2 (px^2)
_D_EPS2 = 1e-18                 # sqrt 기울기 안전용
_LEN_EPS = 1e-9


# ---------------------------------------------------------------------------------------------------------------
# 공용 보조
# ---------------------------------------------------------------------------------------------------------------
def _real(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name}: 유한해야 한다 ({v!r})")
    return f


def _as_tensor(name, v):
    if isinstance(v, torch.Tensor):
        return v
    try:
        return torch.as_tensor(np.asarray(v))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{name}: 텐서로 바꿀 수 없다 ({e})") from e


def _tiny(t: torch.Tensor) -> float:
    return float(torch.finfo(t.dtype).tiny)


def _unit(x: torch.Tensor) -> torch.Tensor:
    n = torch.sqrt((x * x).sum(-1, keepdim=True))
    return x / n.clamp_min(_tiny(x))


def _quad(S: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """wᵀ S w (브로드캐스트)."""
    return torch.einsum("...i,...ij,...j->...", w, S, w)


def _rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) 회전 -> (...,4) wxyz (w >= 0 쪽 부호 규약). 기울기 없음(색인 선택)."""
    R = R.detach()
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    c0 = torch.stack([1 + tr, m21 - m12, m02 - m20, m10 - m01], -1)
    c1 = torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1)
    c2 = torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1)
    c3 = torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1)
    cands = torch.stack([c0, c1, c2, c3], -2)                       # (...,4,4)
    pick = torch.stack([tr, m00, m11, m22], -1).argmax(-1)          # (...)
    sel = torch.gather(cands, -2, pick[..., None, None].expand(*pick.shape, 1, 4)).squeeze(-2)
    return sel / torch.sqrt((sel * sel).sum(-1, keepdim=True)).clamp_min(_tiny(sel))


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz (정규화 전) -> (N,3,3)."""
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True).clamp_min(1e-30))
    w, x, y, z = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return r.reshape(*q.shape[:-1], 3, 3)


# ---------------------------------------------------------------------------------------------------------------
# SGGX
# ---------------------------------------------------------------------------------------------------------------
def _prep(*xs):
    """numpy/list/tensor 혼합 입력 -> (텐서들, numpy 로 돌려줄지). 텐서가 하나라도 있으면 torch 로 계산한다."""
    ref = next((x for x in xs if isinstance(x, torch.Tensor)), None)
    if ref is None:
        out = [torch.as_tensor(np.array(x, dtype=np.float64)) for x in xs]
        return out, True
    dt = ref.dtype if ref.dtype.is_floating_point else torch.float64
    out = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            out.append(x if x.dtype.is_floating_point else x.to(dt))
        else:
            out.append(torch.as_tensor(np.array(x, dtype=np.float64), dtype=dt, device=ref.device))
    return out, False


def _ret(x: torch.Tensor, as_np: bool):
    return x.detach().cpu().numpy() if as_np else x


def _roughness(r, like: torch.Tensor):
    if isinstance(r, torch.Tensor):
        ok = bool(torch.isfinite(r).all()) and bool(((r > 0) & (r <= 1)).all())
        if not ok:
            raise ValueError(f"roughness: (0, 1] 이어야 한다 ({r!r})")
        return r.to(dtype=like.dtype, device=like.device)
    try:
        arr = np.asarray(r, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"roughness: 실수여야 한다 ({r!r})") from e
    if not (np.isfinite(arr).all() and (arr > 0).all() and (arr <= 1).all()):
        raise ValueError(f"roughness: (0, 1] 이어야 한다 ({r!r})")
    return torch.as_tensor(arr, dtype=like.dtype, device=like.device)


def _unit_checked(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.shape[-1] != 3:
        raise ValueError(f"{name}: 마지막 축이 3 이어야 한다 (shape {tuple(x.shape)})")
    nrm = torch.sqrt((x.detach() * x.detach()).sum(-1))
    if not bool((nrm > 0).all()) or not bool(torch.isfinite(nrm).all()):
        raise ValueError(f"{name}: 영벡터 또는 비유한 방향")
    return _unit(x)


def sggx_fiber(t, roughness):
    """섬유 SGGX 행렬 S = I - t tᵀ + r² t tᵀ (t 는 정규화된다). (...,3) -> (...,3,3)."""
    (t,), as_np = _prep(t)
    r = _roughness(roughness, t)
    tn = _unit_checked(t, "t")
    tt = tn[..., :, None] * tn[..., None, :]
    eye = torch.eye(3, dtype=tn.dtype, device=tn.device)
    r2 = (r * r)
    r2 = r2[..., None, None] if r2.ndim else r2
    return _ret(eye - tt + r2 * tt, as_np)


def sggx_surface(n, roughness):
    """표면 SGGX 행렬 S = n nᵀ + r² (I - n nᵀ)."""
    (n,), as_np = _prep(n)
    r = _roughness(roughness, n)
    nn_ = _unit_checked(n, "n")
    nnt = nn_[..., :, None] * nn_[..., None, :]
    eye = torch.eye(3, dtype=nn_.dtype, device=nn_.device)
    r2 = (r * r)
    r2 = r2[..., None, None] if r2.ndim else r2
    return _ret(nnt + r2 * (eye - nnt), as_np)


def sggx_sigma(S, w):
    """투영 면적 σ(w) = sqrt(wᵀ S w) (w 는 정규화된다)."""
    (S, w), as_np = _prep(S, w)
    wn = _unit(w)
    return _ret(torch.sqrt(_quad(S, wn).clamp_min(_tiny(S))), as_np)


def _sggx_D_t(S: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    Sinv = torch.linalg.inv(S)
    det = torch.linalg.det(S)
    q = _quad(Sinv, mn)
    return 1.0 / (math.pi * torch.sqrt(det) * q * q)


def sggx_D(S, m):
    """SGGX 법선 분포 D(m) = 1 / (π sqrt(det S) (mᵀ S⁻¹ m)²) (m 은 정규화된다)."""
    (S, m), as_np = _prep(S, m)
    return _ret(_sggx_D_t(S, _unit(m)), as_np)


def sggx_phase_specular(S, wi, wo):
    """정반사 위상 f_p = D(h) / (4 σ(wi)), h = normalize(wi + wo). |wi + wo| < 1e-9 이면 0."""
    (S, wi, wo), as_np = _prep(S, wi, wo)
    wi, wo = _unit(wi), _unit(wo)
    h = wi + wo
    hn = torch.sqrt((h * h).sum(-1))
    ok = hn >= 1e-9
    ez = torch.zeros_like(h)
    ez[..., 2] = 1.0
    hs = torch.where(ok[..., None], h, ez)
    hs = hs / torch.sqrt((hs * hs).sum(-1, keepdim=True))
    f = _sggx_D_t(S, hs) / (4.0 * torch.sqrt(_quad(S, wi).clamp_min(_tiny(S))))
    return _ret(torch.where(ok, f, torch.zeros_like(f)), as_np)


# ---------------------------------------------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------------------------------------------
_VFIELDS = ("means", "quats", "log_scales", "log_density", "colors", "sggx")
_VTAILS = {"means": (3,), "quats": (4,), "log_scales": (3,), "log_density": (), "colors": (3,), "sggx": (3, 3)}


@dataclass(kw_only=True, eq=False)
class VolGaussians:
    means: torch.Tensor          # (N,3)
    quats: torch.Tensor          # (N,4) wxyz, 정규화 전
    log_scales: torch.Tensor     # (N,3)
    log_density: torch.Tensor    # (N,)
    colors: torch.Tensor         # (N,3)
    sggx: torch.Tensor           # (N,3,3) SPD

    def __post_init__(self):
        vals = {f: _as_tensor(f, getattr(self, f)) for f in _VFIELDS}
        dt, dev = vals["means"].dtype, vals["means"].device
        if dt not in (torch.float32, torch.float64):
            raise ValueError(f"dtype 은 float32|float64 여야 한다 ({dt})")
        n = vals["means"].shape[0] if vals["means"].ndim >= 1 else -1
        for f in _VFIELDS:
            v = vals[f]
            if v.dtype != dt:
                raise ValueError(f"{f}: dtype {v.dtype} != {dt}")
            if v.device != dev:
                raise ValueError(f"{f}: device {v.device} != {dev}")
            if tuple(v.shape) != (n,) + _VTAILS[f]:
                raise ValueError(f"{f}: 모양 {tuple(v.shape)} != {(n,) + _VTAILS[f]}")
            if n > 0 and not bool(torch.isfinite(v.detach()).all()):
                raise ValueError(f"{f}: 비유한 값")
        if n > 0:
            if not bool((vals["quats"].detach().abs().sum(-1) > 0).all()):
                raise ValueError("quats: 영 쿼터니언")
            S = vals["sggx"].detach().to(torch.float64)
            scale = float(S.abs().max())
            if float((S - S.transpose(-1, -2)).abs().max()) > 1e-6 * scale:
                raise ValueError("sggx: 대칭이 아니다")
            ev = torch.linalg.eigvalsh(0.5 * (S + S.transpose(-1, -2)))
            if not bool((ev[..., 0] > 0).all()):
                raise ValueError("sggx: 양정치가 아니다")
        for f in _VFIELDS:
            setattr(self, f, vals[f])

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @property
    def dtype(self):
        return self.means.dtype

    @property
    def device(self):
        return self.means.device

    def densities(self) -> torch.Tensor:
        return torch.exp(self.log_density)

    def rotations(self) -> torch.Tensor:
        return _quat_to_rot(self.quats)

    def covariances(self) -> torch.Tensor:
        R = self.rotations()
        s2 = torch.exp(2.0 * self.log_scales)
        return (R * s2[:, None, :]) @ R.transpose(-1, -2)

    def params(self) -> list:
        return [self.means, self.quats, self.log_scales, self.log_density, self.colors]

    def requires_grad_(self, flag: bool = True) -> "VolGaussians":
        for p in self.params():
            p.requires_grad_(flag)
        return self

    def detach(self) -> "VolGaussians":
        return VolGaussians(**{f: getattr(self, f).detach() for f in _VFIELDS})

    def to(self, device=None, dtype=None) -> "VolGaussians":
        return VolGaussians(**{f: getattr(self, f).to(device=device, dtype=dtype) for f in _VFIELDS})

    def subset(self, idx) -> "VolGaussians":
        idx = _as_tensor("idx", idx).to(self.means.device)
        return VolGaussians(**{f: getattr(self, f)[idx] for f in _VFIELDS})


_SFIELDS = ("points", "radius", "opacity", "colors")


@dataclass(kw_only=True, eq=False)
class Strands:
    points: torch.Tensor     # (S,P,3) P >= 2
    radius: torch.Tensor     # (S,P) m  ((S,) 입력은 (S,P)로 확장)
    opacity: torch.Tensor    # (S,) in (0,1]
    colors: torch.Tensor     # (S,3)

    def __post_init__(self):
        vals = {f: _as_tensor(f, getattr(self, f)) for f in _SFIELDS}
        pts = vals["points"]
        dt, dev = pts.dtype, pts.device
        if dt not in (torch.float32, torch.float64):
            raise ValueError(f"dtype 은 float32|float64 여야 한다 ({dt})")
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"points: 모양 {tuple(pts.shape)} != (S,P,3)")
        S, P = int(pts.shape[0]), int(pts.shape[1])
        if P < 2:
            raise ValueError(f"points: 가닥당 점이 2 개 이상이어야 한다 (P={P})")
        rad = vals["radius"]
        if rad.ndim == 1 and tuple(rad.shape) == (S,):
            rad = rad[:, None].expand(S, P).contiguous()
            vals["radius"] = rad
        want = {"points": (S, P, 3), "radius": (S, P), "opacity": (S,), "colors": (S, 3)}
        for f in _SFIELDS:
            v = vals[f]
            if v.dtype != dt:
                raise ValueError(f"{f}: dtype {v.dtype} != {dt}")
            if v.device != dev:
                raise ValueError(f"{f}: device {v.device} != {dev}")
            if tuple(v.shape) != want[f]:
                raise ValueError(f"{f}: 모양 {tuple(v.shape)} != {want[f]}")
            if S > 0 and not bool(torch.isfinite(v.detach()).all()):
                raise ValueError(f"{f}: 비유한 값")
        if S > 0:
            if not bool((vals["radius"].detach() > 0).all()):
                raise ValueError("radius: 양수여야 한다")
            o = vals["opacity"].detach()
            if not bool(((o > 0) & (o <= 1)).all()):
                raise ValueError("opacity: (0, 1] 이어야 한다")
        for f in _SFIELDS:
            setattr(self, f, vals[f])

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def dtype(self):
        return self.points.dtype

    @property
    def device(self):
        return self.points.device

    def params(self) -> list:
        return [self.points, self.radius, self.opacity, self.colors]

    def requires_grad_(self, flag: bool = True) -> "Strands":
        for p in self.params():
            p.requires_grad_(flag)
        return self

    def detach(self) -> "Strands":
        return Strands(**{f: getattr(self, f).detach() for f in _SFIELDS})

    def to(self, device=None, dtype=None) -> "Strands":
        return Strands(**{f: getattr(self, f).to(device=device, dtype=dtype) for f in _SFIELDS})

    def subset(self, idx) -> "Strands":
        idx = _as_tensor("idx", idx).to(self.points.device)
        return Strands(**{f: getattr(self, f)[idx] for f in _SFIELDS})


# ---------------------------------------------------------------------------------------------------------------
# 옵션 · 결과
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class VolOpts:
    bg: tuple = (0.0, 0.0, 0.0)
    near: float = 0.01
    far: float = 1e4
    filter_sigma: float = 0.5
    max_alpha: float = 0.999
    min_alpha: float = 1.0 / 255.0
    cutoff: float = 4.0

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
        fs = _real("filter_sigma", self.filter_sigma)
        if not fs > 0:
            raise ValueError(f"filter_sigma > 0 이어야 한다 ({fs})")
        mx = _real("max_alpha", self.max_alpha)
        if not 0 < mx < 1:
            raise ValueError(f"max_alpha: (0, 1) 이어야 한다 ({mx})")
        mn = _real("min_alpha", self.min_alpha)
        if not 0 <= mn < 1:
            raise ValueError(f"min_alpha: [0, 1) 이어야 한다 ({mn})")
        co = _real("cutoff", self.cutoff)
        if not co > 0:
            raise ValueError(f"cutoff > 0 이어야 한다 ({co})")
        for k, v in (("bg", bg), ("near", near), ("far", far), ("filter_sigma", fs), ("max_alpha", mx),
                     ("min_alpha", mn), ("cutoff", co)):
            object.__setattr__(self, k, v)


@dataclass(kw_only=True, eq=False)
class VolRender:
    rgb: torch.Tensor       # (H,W,3)
    alpha: torch.Tensor     # (H,W)
    depth: torch.Tensor     # (H,W) 카메라 z, 알파 < 1e-3 은 +inf

    def numpy(self) -> dict:
        return {k: getattr(self, k).detach().cpu().numpy().astype(np.float32) for k in ("rgb", "alpha", "depth")}


# ---------------------------------------------------------------------------------------------------------------
# 카메라
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
    return c, Km, R, c[:3, 3].copy(), W, H


def _pixel_rays(Km, W, H):
    """모든 화소의 카메라 좌표 단위 광선 (P,3), 행 우선(v*W+u), 화소 좌표 (u,v) (P,)."""
    fx, fy, sk, cx, cy = Km[0, 0], Km[1, 1], Km[0, 1], Km[0, 2], Km[1, 2]
    vv, uu = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    yd = (vv - cy) / fy
    xd = (uu - cx - sk * yd) / fx
    r = np.stack([xd, yd, np.ones_like(xd)], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True), uu, vv


# ---------------------------------------------------------------------------------------------------------------
# 조각 계산
# ---------------------------------------------------------------------------------------------------------------
class _GaussPre:
    """가우시안 -> 카메라 좌표 전처리(청크 밖에서 한 번)."""

    def __init__(self, vol: VolGaussians, Rt: torch.Tensor, Ct: torch.Tensor):
        Rg = vol.rotations()
        A = Rt.T @ Rg                                              # 가우시안 축 -> 카메라 좌표
        inv_s2 = torch.exp(-2.0 * vol.log_scales)
        self.Sinv = (A * inv_s2[:, None, :]) @ A.transpose(-1, -2)  # Σ⁻¹ (카메라 좌표)
        self.S = Rt.T @ vol.sggx @ Rt
        self.d = -((vol.means - Ct) @ Rt)                           # C - μ (카메라 좌표; 카메라 원점)
        self.Sd = (self.Sinv @ self.d[..., None]).squeeze(-1)
        self.rho = torch.exp(vol.log_density)
        self.n = vol.n

    def frag(self, OmT: torch.Tensor, omz: torch.Tensor, opts: VolOpts):
        """OmT (3,Pc) 단위 광선, omz (Pc,) -> (alpha (N,Pc), z (N,Pc))."""
        Wm = self.Sinv @ OmT                                       # Σ⁻¹ω   (N,3,Pc)
        SW = self.S @ OmT                                          # Sω
        d = self.d[:, :, None]
        a = (Wm * OmT[None]).sum(1)
        sig2 = (SW * OmT[None]).sum(1).clamp_min(_tiny(a))
        b = (Wm * d).sum(1)
        tstar = -b / a
        p = d + tstar[:, None, :] * OmT[None]
        g = self.Sd[:, :, None] + tstar[:, None, :] * Wm
        q = (p * g).sum(1)                                         # m²
        tau = self.rho[:, None] * torch.sqrt(sig2) * torch.sqrt(_TWO_PI / a) * torch.exp(-0.5 * q)
        alpha = torch.clamp(-torch.expm1(-tau), max=opts.max_alpha)
        z = tstar * omz[None]
        valid = (q <= opts.cutoff ** 2) & (z >= opts.near) & (z <= opts.far)
        return torch.where(valid, alpha, torch.zeros_like(alpha)), z


class _StrandPre:
    """가닥 선분 -> 화면 좌표 전처리."""

    def __init__(self, st: Strands, Km, Rt: torch.Tensor, Ct: torch.Tensor, opts: VolOpts):
        S, P = st.n, int(st.points.shape[1])
        Xc = (st.points - Ct) @ Rt                                 # (S,P,3)
        z = Xc[..., 2]
        ok = (z >= opts.near) & (z <= opts.far)
        zs = torch.where(ok, z, torch.ones_like(z))
        fx, fy, sk, cx, cy = (float(Km[0, 0]), float(Km[1, 1]), float(Km[0, 1]), float(Km[0, 2]), float(Km[1, 2]))
        y = Xc[..., 1] / zs
        x = Xc[..., 0] / zs
        u = fx * x + sk * y + cx
        v = fy * y + cy
        self.S, self.nseg = S, P - 1
        self.valid = (ok[:, :-1] & ok[:, 1:]).reshape(-1)
        self.au, self.av = u[:, :-1].reshape(-1), v[:, :-1].reshape(-1)
        self.bu, self.bv = u[:, 1:].reshape(-1), v[:, 1:].reshape(-1)
        self.za, self.zb = zs[:, :-1].reshape(-1), zs[:, 1:].reshape(-1)
        self.ra, self.rb = st.radius[:, :-1].reshape(-1), st.radius[:, 1:].reshape(-1)
        self.op = st.opacity[:, None].expand(S, P - 1).reshape(-1)
        self.f = math.sqrt(fx * fy)
        self.k = S * (P - 1)
        abu, abv = self.bu - self.au, self.bv - self.av
        self.abu, self.abv = abu, abv
        len2 = abu * abu + abv * abv
        self.deg = len2 < _SEG_EPS2
        self.len2s = torch.where(self.deg, torch.ones_like(len2), len2)

    def frag(self, ux: torch.Tensor, vy: torch.Tensor, opts: VolOpts):
        """ux, vy (Pc,) 화소 좌표 -> (alpha (S,Pc), z (S,Pc))."""
        apu = ux[None, :] - self.au[:, None]
        apv = vy[None, :] - self.av[:, None]
        t = (apu * self.abu[:, None] + apv * self.abv[:, None]) / self.len2s[:, None]
        t = torch.where(self.deg[:, None], torch.zeros_like(t), t.clamp(0.0, 1.0))
        du = apu - t * self.abu[:, None]
        dv = apv - t * self.abv[:, None]
        d = torch.sqrt(du * du + dv * dv + _D_EPS2)
        zt = self.za[:, None] + t * (self.zb - self.za)[:, None]
        rt = self.ra[:, None] + t * (self.rb - self.ra)[:, None]
        w = rt * self.f / zt
        inv = 1.0 / (opts.filter_sigma * _SQRT2)
        a_seg = 0.5 * self.op[:, None] * (torch.special.erfc((d - w) * inv) - torch.special.erfc((d + w) * inv))
        a_seg = torch.where(self.valid[:, None], a_seg, torch.zeros_like(a_seg))
        Pc = a_seg.shape[1]
        a3 = a_seg.reshape(self.S, self.nseg, Pc)
        z3 = zt.reshape(self.S, self.nseg, Pc)
        best = a3.detach().argmax(1, keepdim=True)                  # 동률은 작은 번호
        a_b = a3.gather(1, best).squeeze(1)
        z_b = z3.gather(1, best).squeeze(1)
        a_b = torch.clamp(a_b, max=opts.max_alpha)
        a_b = torch.where(a_b >= opts.min_alpha, a_b, torch.zeros_like(a_b))
        return a_b, z_b


def _backdrop_get(backdrop, key):
    if isinstance(backdrop, dict):
        if key not in backdrop:
            raise ValueError(f"backdrop: '{key}' 키가 없다")
        return backdrop[key]
    if not hasattr(backdrop, key):
        raise ValueError(f"backdrop: '{key}' 속성이 없다")
    return getattr(backdrop, key)


def _backdrop_tensors(backdrop, W, H, dt, dev):
    out = {}
    want = {"rgb": (H, W, 3), "alpha": (H, W), "depth": (H, W)}
    for k, shp in want.items():
        v = _as_tensor(f"backdrop.{k}", _backdrop_get(backdrop, k))
        if tuple(v.shape) != shp:
            raise ValueError(f"backdrop.{k}: 모양 {tuple(v.shape)} != {shp}")
        out[k] = v.to(device=dev, dtype=dt)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 렌더
# ---------------------------------------------------------------------------------------------------------------
def render_vol(*, vol=None, strands=None, c2w, K, width, height, opts=None, backdrop=None) -> VolRender:
    """가우시안 체적 + 가닥 + (선택) backdrop 을 깊이 정렬 over 합성으로 찍는다."""
    opts = VolOpts() if opts is None else opts
    if not isinstance(opts, VolOpts):
        raise ValueError("opts: VolOpts 여야 한다")
    if vol is not None and not isinstance(vol, VolGaussians):
        raise ValueError("vol: VolGaussians 여야 한다")
    if strands is not None and not isinstance(strands, Strands):
        raise ValueError("strands: Strands 여야 한다")
    _, Km, Rw, Cw, W, H = _camera(c2w, K, width, height)
    P = W * H

    refs = [x for x in (vol, strands) if x is not None]
    if refs:
        dt, dev = refs[0].dtype, refs[0].device
        for x in refs[1:]:
            if x.dtype != dt or x.device != dev:
                raise ValueError("vol 과 strands 의 dtype·device 가 같아야 한다")
    else:
        bd_ref = backdrop.get("rgb") if isinstance(backdrop, dict) else getattr(backdrop, "rgb", None)
        if isinstance(bd_ref, torch.Tensor) and bd_ref.dtype in (torch.float32, torch.float64):
            dt, dev = bd_ref.dtype, bd_ref.device
        else:
            dt, dev = torch.float32, torch.device("cpu")
    bd = _backdrop_tensors(backdrop, W, H, dt, dev) if backdrop is not None else None

    n_g = vol.n if vol is not None else 0
    n_s = strands.n if strands is not None else 0
    bgt = torch.tensor(opts.bg, dtype=dt, device=dev)
    if n_g + n_s == 0 and bd is None:
        rgb = bgt.expand(P, 3).clone().reshape(H, W, 3)
        return VolRender(rgb=rgb, alpha=torch.zeros(H, W, dtype=dt, device=dev),
                         depth=torch.full((H, W), float("inf"), dtype=dt, device=dev))

    Rt = torch.tensor(Rw, dtype=dt, device=dev)
    Ct = torch.tensor(Cw, dtype=dt, device=dev)
    om_np, ux_np, vy_np = _pixel_rays(Km, W, H)
    Om = torch.tensor(om_np, dtype=dt, device=dev)                 # (P,3)
    omz = Om[:, 2].contiguous()
    ux = torch.tensor(ux_np, dtype=dt, device=dev)
    vy = torch.tensor(vy_np, dtype=dt, device=dev)

    gp = _GaussPre(vol, Rt, Ct) if n_g > 0 else None
    sp = _StrandPre(strands, Km, Rt, Ct, opts) if n_s > 0 else None
    cols = [x.colors for x in (vol, strands) if x is not None and x.n > 0]
    colors_all = torch.cat(cols, 0) if cols else torch.zeros(0, 3, dtype=dt, device=dev)
    F0 = n_g + n_s
    if bd is not None:
        bd_alpha = bd["alpha"].reshape(-1).clamp(0.0, 1.0)
        bd_rgb = bd["rgb"].reshape(-1, 3)
        bd_depth = bd["depth"].reshape(-1)
        bd_on = bd_alpha > 0
        bd_rgb = torch.where(bd_on[:, None], bd_rgb, torch.zeros_like(bd_rgb))
        bd_fin = bd_on & torch.isfinite(bd_depth)
        bd_zval = torch.where(bd_fin, bd_depth, torch.zeros_like(bd_depth))
    unit = max(n_g, sp.k if sp is not None else 0, 1)
    step = max(1, _CHUNK_ELEMS // unit)

    outs_rgb, outs_alpha, outs_depth = [], [], []
    for lo in range(0, P, step):
        hi = min(P, lo + step)
        Pc = hi - lo
        alphas, zs = [], []
        if gp is not None:
            a_g, z_g = gp.frag(Om[lo:hi].T.contiguous(), omz[lo:hi], opts)
            alphas.append(a_g.T)
            zs.append(z_g.T)
        if sp is not None:
            a_s, z_s = sp.frag(ux[lo:hi], vy[lo:hi], opts)
            alphas.append(a_s.T)
            zs.append(z_s.T)
        if bd is not None:
            alphas.append(bd_alpha[lo:hi, None])
            zs.append(bd_zval[lo:hi, None])
        alpha_f = torch.cat(alphas, 1)                              # (Pc,F)
        z_f = torch.cat(zs, 1)
        live = alpha_f.detach() > 0
        key = torch.where(live, z_f.detach(), torch.full_like(z_f.detach(), float("inf")))
        if bd is not None:
            key[:, -1] = torch.where(live[:, -1] & bd_fin[lo:hi], bd_zval[lo:hi].detach(),
                                     torch.full_like(key[:, -1], float("inf")))
        idx = torch.sort(key, dim=1, stable=True).indices
        a_sorted = alpha_f.gather(1, idx)
        cp = torch.cumprod(1.0 - a_sorted, dim=1)
        T_sorted = torch.cat([torch.ones(Pc, 1, dtype=dt, device=dev), cp[:, :-1]], 1)
        w_sorted = a_sorted * T_sorted
        w = torch.zeros_like(w_sorted).scatter(1, idx, w_sorted)    # 입력 순서로 복원
        alpha_o = w_sorted.sum(1)
        t_final = cp[:, -1]
        rgb_o = t_final[:, None] * bgt
        if F0 > 0:
            rgb_o = rgb_o + w[:, :F0] @ colors_all
        if bd is not None:
            T_orig = torch.zeros_like(T_sorted).scatter(1, idx, T_sorted)
            rgb_o = rgb_o + T_orig[:, -1:] * bd_rgb[lo:hi]
        z_val = torch.where(live, z_f, torch.zeros_like(z_f))
        if bd is not None:
            z_val = torch.cat([z_val[:, :-1], bd_zval[lo:hi, None] * live[:, -1:]], 1)
        dnum = (w * z_val).sum(1)
        low = alpha_o.detach() < _ALPHA_OUT_EPS
        depth_o = torch.where(low, torch.full_like(alpha_o, float("inf")),
                              dnum / torch.where(low, torch.ones_like(alpha_o), alpha_o))
        outs_rgb.append(rgb_o)
        outs_alpha.append(alpha_o)
        outs_depth.append(depth_o)
    return VolRender(rgb=torch.cat(outs_rgb, 0).reshape(H, W, 3), alpha=torch.cat(outs_alpha, 0).reshape(H, W),
                     depth=torch.cat(outs_depth, 0).reshape(H, W))


# ---------------------------------------------------------------------------------------------------------------
# 변환
# ---------------------------------------------------------------------------------------------------------------
def alpha_to_tau(alpha, max_alpha=0.999):
    """τ = -log(1 - clip(α, 0, max_alpha)). numpy -> numpy, tensor -> tensor."""
    mx = _real("max_alpha", max_alpha)
    if isinstance(alpha, torch.Tensor):
        return -torch.log1p(-alpha.clamp(0.0, mx))
    a = np.asarray(alpha, dtype=np.float64)
    return -np.log1p(-np.clip(a, 0.0, mx))


def tau_to_alpha(tau):
    """α = 1 - exp(-τ) (τ < 0 은 0 으로)."""
    if isinstance(tau, torch.Tensor):
        return -torch.expm1(-tau.clamp_min(0.0))
    t = np.asarray(tau, dtype=np.float64)
    return -np.expm1(-np.maximum(t, 0.0))


def _np_map(name, v, shape):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 수치 배열이어야 한다 ({e})") from e
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name}: 모양 {a.shape} != {shape}")
    return a


def matte_to_vol(alpha, fg_rgb, depth, *, K, c2w, stride=1, footprint=1.0, thickness=None, tangent=None,
                 roughness=0.1, min_alpha=1.0 / 255.0, dtype=torch.float32) -> VolGaussians:
    """매팅 α (+ 전경색·깊이) -> 화소당 가우시안 하나인 체적. 원천 뷰에서 α 가 재현되고 시점을 옮기면 시차가 난다."""
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()
    a_np = np.asarray(alpha, dtype=np.float64) if not isinstance(alpha, np.ndarray) else alpha.astype(np.float64)
    if a_np.ndim != 2:
        raise ValueError(f"alpha: (H,W) 여야 한다 (shape {a_np.shape})")
    H, W = a_np.shape
    rgb = _np_map("fg_rgb", fg_rgb, (H, W, 3))
    dep = _np_map("depth", depth, (H, W))
    _, Km, Rw, Cw, _, _ = _camera(c2w, K, W, H)
    if isinstance(stride, bool) or not isinstance(stride, numbers.Integral) or int(stride) < 1:
        raise ValueError(f"stride: 1 이상의 정수여야 한다 ({stride!r})")
    stride = int(stride)
    f = _real("footprint", footprint)
    if not f > 0:
        raise ValueError(f"footprint > 0 이어야 한다 ({footprint!r})")
    if thickness is not None:
        thickness = _real("thickness", thickness)
        if not thickness > 0:
            raise ValueError(f"thickness > 0 이어야 한다 ({thickness!r})")
    mina = _real("min_alpha", min_alpha)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype 은 float32|float64 여야 한다 ({dtype})")
    _roughness(roughness, torch.zeros(1, dtype=torch.float64))
    if tangent is not None:
        if isinstance(tangent, torch.Tensor):
            tangent = tangent.detach().cpu().numpy()
        tangent = np.asarray(tangent, dtype=np.float64)
        if tangent.shape not in ((3,), (H, W, 3)):
            raise ValueError(f"tangent: (3,) 또는 (H,W,3) 이어야 한다 (shape {tangent.shape})")

    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    vs, us = vs.reshape(-1), us.reshape(-1)
    a_sel, d_sel = a_np[vs, us], dep[vs, us]
    keep = np.isfinite(d_sel) & (d_sel > 0) & np.isfinite(a_sel) & (a_sel >= mina)
    vs, us, a_sel, d_sel = vs[keep], us[keep], a_sel[keep], d_sel[keep]
    M = int(vs.shape[0])
    if M == 0:
        return VolGaussians(means=torch.zeros(0, 3, dtype=dtype), quats=torch.zeros(0, 4, dtype=dtype),
                            log_scales=torch.zeros(0, 3, dtype=dtype), log_density=torch.zeros(0, dtype=dtype),
                            colors=torch.zeros(0, 3, dtype=dtype), sggx=torch.zeros(0, 3, 3, dtype=dtype))
    uv = np.stack([us, vs], -1).astype(np.float64)
    mu = geom.unproject_world(Km, Rw.T, -Rw.T @ Cw, uv, d_sel)
    col = rgb[vs, us]
    fx, fy = Km[0, 0], Km[1, 1]
    sx = f * stride * d_sel / fx
    sy = f * stride * d_sel / fy
    sz = np.full(M, thickness) if thickness is not None else 0.25 * sx
    scales = np.stack([sx, sy, sz], -1)
    if tangent is None:
        S = np.broadcast_to(np.eye(3), (M, 3, 3)).copy()
    else:
        tg = np.broadcast_to(tangent, (M, 3)) if tangent.shape == (3,) else tangent[vs, us]
        S = sggx_fiber(tg, roughness)
    w = mu - Cw
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    w_cam = w @ Rw                                                  # Rwᵀ ω (행벡터)
    a_q = ((w_cam / scales) ** 2).sum(-1)
    sig = np.sqrt(np.einsum("ni,nij,nj->n", w, S, w))
    tau_t = alpha_to_tau(a_sel) / (_TWO_PI * f * f)
    rho = tau_t / (sig * np.sqrt(_TWO_PI / a_q))
    quat = _rot_to_quat(torch.as_tensor(np.broadcast_to(Rw, (M, 3, 3)).copy()))
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)  # noqa: E731
    return VolGaussians(means=t(mu), quats=t(quat.numpy()), log_scales=t(np.log(scales)), log_density=t(np.log(rho)),
                        colors=t(col), sggx=t(S))


def strands_to_vol(strands: Strands, *, roughness=0.1, min_std=0.0) -> VolGaussians:
    """가닥을 선분당 SGGX 가우시안 하나로 바꾼다(원거리 LOD). 투영 커버리지 2·o·r̄·L·σ(ω) 를 보존한다."""
    if not isinstance(strands, Strands):
        raise ValueError("strands: Strands 여야 한다")
    ms = _real("min_std", min_std)
    if ms < 0:
        raise ValueError(f"min_std >= 0 이어야 한다 ({min_std!r})")
    dt, dev = strands.dtype, strands.device
    _roughness(roughness, torch.zeros(1, dtype=torch.float64))
    S, P = strands.n, int(strands.points.shape[1])
    pa, pb = strands.points[:, :-1], strands.points[:, 1:]
    seg = (pb - pa).reshape(-1, 3)
    L = torch.sqrt((seg * seg).sum(-1))
    keep = L.detach() > _LEN_EPS
    kidx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if kidx.numel() == 0:
        return VolGaussians(means=torch.zeros(0, 3, dtype=dt, device=dev), quats=torch.zeros(0, 4, dtype=dt, device=dev),
                            log_scales=torch.zeros(0, 3, dtype=dt, device=dev),
                            log_density=torch.zeros(0, dtype=dt, device=dev),
                            colors=torch.zeros(0, 3, dtype=dt, device=dev),
                            sggx=torch.zeros(0, 3, 3, dtype=dt, device=dev))
    A = pa.reshape(-1, 3)[kidx]
    Bp = pb.reshape(-1, 3)[kidx]
    Lk = L[kidx]
    sd = seg[kidx]
    rbar = (0.5 * (strands.radius[:, :-1] + strands.radius[:, 1:])).reshape(-1)[kidx]
    op = strands.opacity[:, None].expand(S, P - 1).reshape(-1)[kidx]
    col = strands.colors[:, None, :].expand(S, P - 1, 3).reshape(-1, 3)[kidx]
    tdir = (sd / Lk[:, None]).detach()                              # 접선(기울기 분리)
    # 접선에 가장 덜 정렬된 좌표축으로 det +1 정규직교 기저를 만든다.
    e = torch.zeros_like(tdir)
    e.scatter_(1, tdir.abs().argmin(1, keepdim=True), 1.0)
    u2 = e - (e * tdir).sum(-1, keepdim=True) * tdir
    u2 = u2 / torch.sqrt((u2 * u2).sum(-1, keepdim=True))
    u3 = torch.cross(tdir, u2, dim=-1)
    R = torch.stack([tdir, u2, u3], -1)                             # 열 = 축
    s_t = torch.clamp(0.5 * Lk, min=ms)
    s_p = torch.clamp(rbar, min=ms)
    rho = 2.0 * op * rbar * Lk / ((_TWO_PI ** 1.5) * s_t * s_p * s_p)
    log_scales = torch.log(torch.stack([s_t, s_p, s_p], -1))
    sg = sggx_fiber(tdir, roughness)
    return VolGaussians(means=0.5 * (A + Bp), quats=_rot_to_quat(R).to(dt), log_scales=log_scales,
                        log_density=torch.log(rho), colors=col, sggx=sg.to(dt))


def scatter_colors(vol: VolGaussians, *, light_dir, light_rgb, cam_center, albedo=None) -> torch.Tensor:
    """가우시안마다 SGGX 정반사 위상으로 빛을 카메라 쪽으로 산란시킨 색 (N,3). 그림자 없음(v0)."""
    if not isinstance(vol, VolGaussians):
        raise ValueError("vol: VolGaussians 여야 한다")
    dt, dev = vol.dtype, vol.device

    def conv(name, x, shape):
        try:
            t = x.to(dtype=dt, device=dev) if isinstance(x, torch.Tensor) else \
                torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dt, device=dev)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name}: 수치 배열이어야 한다 ({e})") from e
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: 모양 {tuple(t.shape)} != {shape}")
        return t

    ld = conv("light_dir", light_dir, (3,))
    lrgb = conv("light_rgb", light_rgb, (3,))
    cc = conv("cam_center", cam_center, (3,))
    alb = vol.colors if albedo is None else conv("albedo", albedo, (vol.n, 3))
    nrm = torch.sqrt((ld.detach() * ld.detach()).sum())
    if not float(nrm) > 0:
        raise ValueError("light_dir: 영벡터")
    wi = -ld / nrm
    wo = cc - vol.means
    wo = wo / torch.sqrt((wo * wo).sum(-1, keepdim=True)).clamp_min(_tiny(wo))
    wi_b = wi.expand(vol.n, 3)
    f = sggx_phase_specular(vol.sggx, wi_b, wo)
    return alb * lrgb * f[:, None]
