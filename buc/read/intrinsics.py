"""RD-09: R7 내재 분해 v0 -- Retinex 평활 + 기하 음영 사전 (알베도 A, 음영 S, 소유 L_d).

중립 룩 장면 선형 L̂ (RD-06 ``inverse(...).linear``)을 ``L_d = A * S`` (S 는 무채색 스칼라 음영)로 가른다.
첫 프레임 외관은 L_d(입력 그대로, 비트 동일)가 소유한다. A 는 ``L_d / S`` 로 파생되고 재조명·프레임 밖 팔레트에만 쓴다.
S 는 ln S 에 대한 볼록 2차 에너지(계약서 RD-09.md 의 E(s))의 유일해로, 희소 직접 풀이로 구한다.
광원(R8) <-> 음영(R7)을 ``n_alt`` 번 교대한다(광원 <- S, S <- 광원).

numpy/scipy 만 쓴다(torch·yaml 금지; 텐서는 ``.detach().cpu().numpy()`` 덕 타이핑). import 부작용·전역 RNG 없음.
결정론(같은 입력 -> 비트 동일), 입력 불변, 새 배열 반환.
"""
from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import optimize as _opt
from scipy import sparse as _sp
from scipy.sparse import csgraph as _csgraph
from scipy.sparse import linalg as _spla

__all__ = [
    "Y_FLOOR", "S_MIN", "GEO_FLOOR", "NORMAL_MIN", "MIN_LIGHT_PIXELS",
    "IntrinsicsError", "IntrinsicsOpts", "GeoLight", "LightFit", "IntrinsicsResult",
    "geometric_shading", "fit_geo_light", "decompose", "relight", "heads_from_readout",
]

Y_FLOOR = 1e-6
S_MIN = 1e-4
GEO_FLOOR = 1e-3
NORMAL_MIN = 0.5
MIN_LIGHT_PIXELS = 16

_W709 = np.array([0.2126, 0.7152, 0.0722], np.float64)
_GEO_SOURCES = ("none", "light_fit", "light", "geo_shading")
_COARSE_DIRS = 2048          # 광원 방향 전역 탐색용 피보나치 구면 격자
_COARSE_PIXELS = 4096        # 전역 탐색에 쓰는 화소 표본 상한(결정론적 등간격)
_REFINE_STARTS = 4           # 국소 정밀화 시작점 수
_REFINE_SEP_COS = math.cos(math.radians(10.0))


class IntrinsicsError(ValueError):
    """내재 분해가 정의되지 않는 입력(사전 부족, 화소 부족, 미정 성분 등)."""


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _real(name, v, *, lo=None, lo_open=False) -> float:
    if not _is_real(v):
        raise TypeError(f"{name}: 실수여야 한다 ({type(v).__name__})")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name}: 유한해야 한다 ({x})")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f"{name}: {'>' if lo_open else '>='} {lo} 이어야 한다 ({x})")
    return x


def _farray(name, a, *, shape=None, ndim=None, last=None, nonneg=False, finite=True) -> np.ndarray:
    """float32/float64 ndarray 검사(아니면 TypeError), 모양·유한·범위 위반은 ValueError. 복사하지 않는다."""
    if not isinstance(a, np.ndarray) or a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 ndarray 여야 한다 ({type(a).__name__}"
                        f"{'' if not isinstance(a, np.ndarray) else ', ' + str(a.dtype)})")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name}: {ndim}차원이어야 한다 (shape={a.shape})")
    if last is not None and a.shape[-1] != last:
        raise ValueError(f"{name}: 마지막 축이 {last} 이어야 한다 (shape={a.shape})")
    if shape is not None and a.shape[:len(shape)] != tuple(shape):
        raise ValueError(f"{name}: 모양 {tuple(shape)} 이어야 한다 (shape={a.shape})")
    if finite and not np.isfinite(a).all():
        raise ValueError(f"{name}: 비유한 값이 있다")
    if nonneg and (a < 0).any():
        raise ValueError(f"{name}: 음수가 있다")
    return a


def _barray(name, a, shape) -> np.ndarray:
    if not isinstance(a, np.ndarray) or a.dtype != np.bool_:
        raise TypeError(f"{name}: bool ndarray 여야 한다")
    if a.shape != tuple(shape):
        raise ValueError(f"{name}: 모양 {tuple(shape)} 이어야 한다 (shape={a.shape})")
    return a


# ---------------------------------------------------------------------------------------------------------------
# 공개 dataclass
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class IntrinsicsOpts:
    alpha: float = 1.0          # Retinex 데이터항(휘도 기울기 일치)
    beta: float = 0.1           # 음영 평활(법선이 연속인 곳)
    gamma: float = 1.0          # 기하 음영 사전
    eta: float = 0.3            # 음영 헤드 사전
    delta: float = 0.1          # 알베도 헤드 사전(채널당)
    tau_chroma: float = 0.05
    tau_normal: float = 0.5
    light_poor_rel: float = 0.25
    n_alt: int = 2
    max_side: int = 320

    def __post_init__(self):
        for n in ("alpha", "beta", "gamma", "eta", "delta"):
            object.__setattr__(self, n, _real(n, getattr(self, n), lo=0.0))
        for n in ("tau_chroma", "tau_normal", "light_poor_rel"):
            object.__setattr__(self, n, _real(n, getattr(self, n), lo=0.0, lo_open=True))
        for n, lo, hi in (("n_alt", 1, 5), ("max_side", 8, None)):
            v = getattr(self, n)
            if not _is_int(v):
                raise TypeError(f"{n}: 정수여야 한다 ({type(v).__name__})")
            v = int(v)
            if v < lo or (hi is not None and v > hi):
                raise ValueError(f"{n}: {lo}..{hi if hi is not None else '∞'} 이어야 한다 ({v})")
            object.__setattr__(self, n, v)


@dataclass(frozen=True, kw_only=True)
class GeoLight:
    """단일 방향광 + 주변광. direction = 표면 -> 광원 단위벡터(카메라 좌표, OpenCV)."""
    direction: tuple
    intensity: float
    ambient: float

    def __post_init__(self):
        try:
            d = tuple(self.direction)
        except TypeError:
            raise TypeError("direction: 길이 3 시퀀스여야 한다") from None
        if len(d) != 3:
            raise ValueError(f"direction: 길이 3 이어야 한다 ({len(d)})")
        d = tuple(_real("direction", v) for v in d)
        if abs(math.sqrt(sum(v * v for v in d)) - 1.0) >= 1e-6:
            raise ValueError(f"direction: 단위 벡터여야 한다 ({d})")
        object.__setattr__(self, "direction", d)
        object.__setattr__(self, "intensity", _real("intensity", self.intensity, lo=0.0))
        object.__setattr__(self, "ambient", _real("ambient", self.ambient, lo=0.0))

    def to_meta(self) -> dict:
        return {"direction": [float(v) for v in self.direction], "intensity": float(self.intensity),
                "ambient": float(self.ambient)}


@dataclass(frozen=True)
class LightFit:
    light: GeoLight
    rms: float
    n_used: int


@dataclass(frozen=True, eq=False)
class IntrinsicsResult:
    L_d: np.ndarray                 # f32 (H,W,3) 입력의 float32 복사(비트 동일)
    albedo: np.ndarray              # f32 (H,W,3) = L_d / shading (제한 없음)
    shading: np.ndarray             # f32 (H,W) > 0
    valid: np.ndarray               # bool (H,W) = ~제외
    geo_shading: np.ndarray | None  # f32 (H,W) | None
    light: GeoLight | None
    geo_source: str                 # "none" | "light_fit" | "light" | "geo_shading"
    r_intr: float                   # nan 가능
    n_solves: int
    solve_stride: int
    warnings: tuple
    opts: IntrinsicsOpts

    @property
    def reliable(self) -> bool:
        return not self.warnings

    def meta(self) -> dict:
        n_valid = int(self.valid.sum())
        n = int(self.valid.size)
        r = float(self.r_intr)
        return {
            "light": None if self.light is None else self.light.to_meta(),
            "geo_source": self.geo_source,
            "r_intr": None if math.isnan(r) else r,
            "n_solves": int(self.n_solves),
            "solve_stride": int(self.solve_stride),
            "n_valid": n_valid,
            "excluded_frac": float(1.0 - n_valid / n),
            "warnings": list(self.warnings),
            "reliable": self.reliable,
            "opts": dataclasses.asdict(self.opts),
        }


# ---------------------------------------------------------------------------------------------------------------
# 기하 음영
# ---------------------------------------------------------------------------------------------------------------
def _geo(normal64: np.ndarray, light: GeoLight, visibility: np.ndarray | None = None) -> np.ndarray:
    d = np.asarray(light.direction, np.float64)
    m = np.maximum(normal64 @ d, 0.0)
    if visibility is None:
        return light.ambient + light.intensity * m
    return light.ambient + light.intensity * visibility * m


def geometric_shading(normal, light, *, visibility=None) -> np.ndarray:
    """Ŝ_geo = ambient + intensity * vis * max(0, n·ω). 법선은 정규화하지 않는다(영벡터면 ambient). f64 (H,W)."""
    n = _farray("normal", normal, ndim=3, last=3)
    if not isinstance(light, GeoLight):
        raise TypeError(f"light: GeoLight 여야 한다 ({type(light).__name__})")
    vis = None
    if visibility is not None:
        v = _farray("visibility", visibility, shape=n.shape[:2], ndim=2)
        if (v < 0).any() or (v > 1).any():
            raise ValueError("visibility: [0,1] 범위여야 한다")
        vis = v.astype(np.float64)
    return _geo(n.astype(np.float64), light, vis)


# ---------------------------------------------------------------------------------------------------------------
# 조명 적합: min_{ω∈S², amb,k>=0} Σ w (S - amb - k max(0, n·ω))²   (전역: 구면 격자 + 국소 정밀화)
# ---------------------------------------------------------------------------------------------------------------
def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    ph = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def _amb_k_candidates(W, St, Stt, SS, SSt, SSS):
    """주어진 모멘트에서 (amb,k>=0) 후보 3개의 (목적식, amb, k). 벡터화(원소별)."""
    obj, amb, kk = [], [], []
    # 내점 해
    det = W * Stt - St * St
    ok = det > 1e-12 * np.maximum(W * Stt, 1e-300)
    sd = np.where(ok, det, 1.0)
    a1 = (SS * Stt - SSt * St) / sd
    k1 = (W * SSt - St * SS) / sd
    ok = ok & (a1 >= 0) & (k1 >= 0)
    o1 = SSS - 2 * a1 * SS - 2 * k1 * SSt + a1 * a1 * W + 2 * a1 * k1 * St + k1 * k1 * Stt
    obj.append(np.where(ok, o1, np.inf)); amb.append(a1); kk.append(k1)
    # k = 0
    a2 = np.maximum(SS / W, 0.0) + 0.0 * Stt
    o2 = SSS - 2 * a2 * SS + a2 * a2 * W
    obj.append(o2); amb.append(a2); kk.append(0.0 * Stt)
    # amb = 0
    pos = Stt > 0
    k3 = np.where(pos, np.maximum(SSt / np.where(pos, Stt, 1.0), 0.0), 0.0)
    o3 = SSS - 2 * k3 * SSt + k3 * k3 * Stt
    obj.append(np.where(pos, o3, np.inf)); amb.append(0.0 * Stt); kk.append(k3)
    return np.stack(obj), np.stack(amb), np.stack(kk)


class _LightProblem:
    """사용 화소(단위 법선 N, 음영 S, 가중 w)에 대한 프로파일 목적식 f(ω) = min_{amb,k>=0} Σ w r²."""

    def __init__(self, N, S, w):
        self.N, self.S, self.w = N, S, w
        self.W = float(w.sum())
        self.wS = w * S
        self.SS = float(self.wS.sum())
        self.SSS = float((self.wS * S).sum())

    def solve(self, omega):
        t = np.maximum(self.N @ omega, 0.0)
        wt = self.w * t
        St = float(wt.sum())
        Stt = float(wt @ t)
        SSt = float(self.wS @ t)
        o, a, k = _amb_k_candidates(self.W, St, Stt, self.SS, SSt, self.SSS)
        best = (math.inf, 0.0, 0.0)
        for j in range(3):
            if not math.isfinite(float(o[j])):
                continue
            aj, kj = max(float(a[j]), 0.0), max(float(k[j]), 0.0)
            r = self.S - aj - kj * t
            ex = float(self.w @ (r * r))
            if ex < best[0]:
                best = (ex, aj, kj)
        return best

    def objective(self, omega):
        return self.solve(omega)[0]


def _tangent_basis(omega):
    ref = np.array([1.0, 0.0, 0.0]) if abs(omega[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(omega, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(omega, e1)
    return e1, e2


def _used_mask(nrm, weight, mask):
    return (weight > 0) & (nrm >= NORMAL_MIN) & mask


def _fit_core(normal64, S64, weight, mask) -> LightFit:
    nrm = np.linalg.norm(normal64, axis=-1)
    used = _used_mask(nrm, weight, mask)
    m = int(used.sum())
    if m < MIN_LIGHT_PIXELS:
        raise IntrinsicsError(f"조명 적합에 쓸 화소가 부족하다 ({m} < {MIN_LIGHT_PIXELS})")
    N = normal64[used] / nrm[used][:, None]
    S = S64[used]
    w = weight[used]
    prob = _LightProblem(N, S, w)
    # 1) 구면 격자 전역 탐색(화소 표본)
    if m > _COARSE_PIXELS:
        sel = np.linspace(0, m - 1, _COARSE_PIXELS).astype(np.int64)
        cp = _LightProblem(N[sel], S[sel], w[sel])
    else:
        cp = prob
    D = _fib_sphere(_COARSE_DIRS)
    T = np.maximum(cp.N @ D.T, 0.0)
    WT = cp.w[:, None] * T
    St = WT.sum(0)
    Stt = (WT * T).sum(0)
    SSt = cp.wS @ T
    o, _, _ = _amb_k_candidates(cp.W, St, Stt, cp.SS, SSt, cp.SSS)
    best = o.min(0)
    order = np.argsort(best, kind="stable")
    starts = []
    for j in order:
        if all(float(D[j] @ s) < _REFINE_SEP_COS for s in starts):
            starts.append(D[j])
            if len(starts) >= _REFINE_STARTS:
                break
    # 2) 각 시작점에서 접평면 좌표 Nelder-Mead 정밀화
    scale = max(prob.SSS, 1e-300)
    cand = (math.inf, None)
    for om0 in starts:
        e1, e2 = _tangent_basis(om0)

        def f(p, om0=om0, e1=e1, e2=e2):
            v = om0 + p[0] * e1 + p[1] * e2
            return prob.objective(v / np.linalg.norm(v)) / scale

        res = _opt.minimize(f, np.zeros(2), method="Nelder-Mead",
                            options={"initial_simplex": np.array([[0.0, 0.0], [0.08, 0.0], [0.0, 0.08]]),
                                     "xatol": 1e-9, "fatol": 1e-15, "maxiter": 400, "maxfev": 600})
        v = om0 + res.x[0] * e1 + res.x[1] * e2
        v = v / np.linalg.norm(v)
        val = prob.objective(v)
        if val < cand[0]:
            cand = (val, v)
        v0 = om0
        val0 = prob.objective(v0)
        if val0 < cand[0]:
            cand = (val0, v0)
    om = cand[1] / np.linalg.norm(cand[1])
    ex, amb, k = prob.solve(om)
    r = S - amb - k * np.maximum(N @ om, 0.0)
    rms = math.sqrt(max(float(w @ (r * r)) / float(w.sum()), 0.0))
    light = GeoLight(direction=tuple(float(x) for x in om), intensity=float(k), ambient=float(amb))
    return LightFit(light=light, rms=rms, n_used=m)


def fit_geo_light(normal, shading, *, weight=None, mask=None) -> LightFit:
    """음영 S 에 단일 방향광+주변광 모형을 전역 최소제곱으로 맞춘다. 사용 화소 = weight>0 ∧ ‖n‖≥NORMAL_MIN ∧ mask."""
    n = _farray("normal", normal, ndim=3, last=3)
    hw = n.shape[:2]
    s = _farray("shading", shading, shape=hw, ndim=2)
    w = np.ones(hw) if weight is None else _farray("weight", weight, shape=hw, ndim=2, nonneg=True)
    mk = np.ones(hw, bool) if mask is None else _barray("mask", mask, hw)
    return _fit_core(n.astype(np.float64), s.astype(np.float64), w.astype(np.float64), mk)


# ---------------------------------------------------------------------------------------------------------------
# R7 풀이: ln S 에 대한 볼록 2차 에너지의 희소 직접 풀이
# ---------------------------------------------------------------------------------------------------------------
def _fallback_ln(I, geo, head, ahead):
    """제외 화소의 ln S 폴백: max(Ŝ_geo,S_MIN) > max(head,S_MIN) > exp(s^A)."""
    if geo is not None:
        return np.log(np.maximum(geo, S_MIN))
    if head is not None:
        return np.log(np.maximum(head, S_MIN))
    return _albedo_prior_ln(I, ahead)


def _albedo_prior_ln(I, ahead):
    return (np.log(np.maximum(I, Y_FLOOR)) - np.log(np.maximum(ahead, Y_FLOOR))).mean(-1)


def _solve_ln(I, valid, normal, geo, g, head, ahead, opts: IntrinsicsOpts) -> np.ndarray:
    """계약서 E(s) 의 최소점 ln S (h,w). 제외 화소는 폴백 값."""
    h, w = valid.shape
    s_out = _fallback_ln(I, geo, head, ahead).astype(np.float64)
    nv = int(valid.sum())
    if nv == 0:
        return s_out
    Y = np.maximum(I @ _W709, Y_FLOOR)
    lum = np.log(Y)
    csum = np.maximum(I.sum(-1), 1e-6)
    chrom = I[..., :2] / csum[..., None]
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(nv)

    ia_l, ib_l, wo_l, we_l, dl_l = [], [], [], [], []
    for sa, sb in (((slice(None), slice(0, w - 1)), (slice(None), slice(1, w))),
                   ((slice(0, h - 1), slice(None)), (slice(1, h), slice(None)))):
        ok = valid[sa] & valid[sb]
        om = np.exp(-((np.linalg.norm(chrom[sa] - chrom[sb], axis=-1) / opts.tau_chroma) ** 2))
        if normal is None:
            bn = np.ones_like(om)
        else:
            bn = np.exp(-((np.linalg.norm(normal[sa] - normal[sb], axis=-1) / opts.tau_normal) ** 2))
        ia_l.append(idx[sa][ok]); ib_l.append(idx[sb][ok])
        wo = opts.alpha * om[ok]
        wo_l.append(wo); we_l.append(wo + opts.beta * bn[ok]); dl_l.append((lum[sa] - lum[sb])[ok])
    ia, ib = np.concatenate(ia_l), np.concatenate(ib_l)
    wo, we, dl = np.concatenate(wo_l), np.concatenate(we_l), np.concatenate(dl_l)

    pd = np.zeros(nv)           # 사전 가중
    pr = np.zeros(nv)           # 사전 가중 * 목표
    if geo is not None:
        gx = opts.gamma * g[valid]
        pd += gx
        pr += gx * np.log(np.maximum(geo[valid], S_MIN))
    if head is not None:
        pd += opts.eta
        pr += opts.eta * np.log(np.maximum(head[valid], S_MIN))
    if ahead is not None:
        pd += 3.0 * opts.delta
        pr += 3.0 * opts.delta * _albedo_prior_ln(I, ahead)[valid]

    # 연결 성분(양의 간선 가중)마다 사전 가중 합이 있어야 수준이 정해진다
    pos = we > 0
    graph = _sp.coo_matrix((np.ones(int(pos.sum())), (ia[pos], ib[pos])), shape=(nv, nv))
    ncomp, lab = _csgraph.connected_components(graph, directed=False)
    comp_prior = np.bincount(lab, weights=pd, minlength=ncomp)
    if (comp_prior <= 0).any():
        raise IntrinsicsError("음영 수준이 미정이다: 사전 가중 합이 0 인 연결 성분이 있다")

    diag = np.bincount(ia, weights=we, minlength=nv) + np.bincount(ib, weights=we, minlength=nv) + pd
    rhs = np.bincount(ia, weights=wo * dl, minlength=nv) - np.bincount(ib, weights=wo * dl, minlength=nv) + pr
    A = _sp.coo_matrix((np.concatenate([diag, -we, -we]),
                        (np.concatenate([np.arange(nv), ia, ib]), np.concatenate([np.arange(nv), ib, ia]))),
                       shape=(nv, nv)).tocsc()
    sol = np.atleast_1d(_spla.spsolve(A, rhs))
    if not np.isfinite(sol).all():
        raise IntrinsicsError("풀이가 유한하지 않다(조건수 붕괴)")
    s_out[valid] = sol
    return s_out


def _axis_interp(n, ns, f):
    t = np.minimum(np.arange(n, dtype=np.float64) / f, ns - 1)
    i0 = np.minimum(np.floor(t).astype(np.int64), ns - 1)
    i1 = np.minimum(i0 + 1, ns - 1)
    return i0, i1, t - i0


def _upsample_ln(s, f, h, w):
    """표본 격자 ln S 를 표본 좌표 (y/f, x/f)에서 쌍선형 보간(끝은 마지막 표본으로 고정)."""
    y0, y1, fy = _axis_interp(h, s.shape[0], f)
    x0, x1, fx = _axis_interp(w, s.shape[1], f)
    fy = fy[:, None]
    fx = fx[None, :]
    top = (1 - fx) * s[y0][:, x0] + fx * s[y0][:, x1]
    bot = (1 - fx) * s[y1][:, x0] + fx * s[y1][:, x1]
    return (1 - fy) * top + fy * bot


def decompose(linear, *, normal=None, shading_head=None, albedo_head=None, geo_shading=None, light=None,
              geo_weight=None, exclude=None, status=None, opts=None) -> IntrinsicsResult:
    """L_d = linear 를 알베도·음영으로 가른다. 계약서 RD-09.md 의 에너지 E(s)를 풀고 L_d 는 그대로 소유한다."""
    I_in = _farray("linear", linear, ndim=3, last=3, nonneg=True)
    h, w = I_in.shape[:2]
    if h < 4 or w < 4:
        raise ValueError(f"linear: H,W >= 4 이어야 한다 ({h}x{w})")
    hw = (h, w)
    if opts is None:
        opts = IntrinsicsOpts()
    elif not isinstance(opts, IntrinsicsOpts):
        raise TypeError("opts: IntrinsicsOpts 여야 한다")
    nrm_in = None if normal is None else _farray("normal", normal, shape=hw, ndim=3, last=3)
    head_in = None if shading_head is None else _farray("shading_head", shading_head, shape=hw, ndim=2, nonneg=True)
    ahead_in = None if albedo_head is None else _farray("albedo_head", albedo_head, shape=hw, ndim=3, last=3,
                                                         nonneg=True)
    geo_in = None if geo_shading is None else _farray("geo_shading", geo_shading, shape=hw, ndim=2, nonneg=True)
    gw_in = np.ones(hw) if geo_weight is None else _farray("geo_weight", geo_weight, shape=hw, ndim=2, nonneg=True)
    if light is not None and not isinstance(light, GeoLight):
        raise TypeError(f"light: GeoLight 여야 한다 ({type(light).__name__})")
    ex_in = None if exclude is None else _barray("exclude", exclude, hw)
    if status is not None:
        if not isinstance(status, np.ndarray) or status.dtype != np.uint8:
            raise TypeError("status: uint8 ndarray 여야 한다")
        if status.shape != hw:
            raise ValueError(f"status: 모양 {hw} 이어야 한다 (shape={status.shape})")
    if light is not None and geo_in is not None:
        raise ValueError("light 와 geo_shading 을 동시에 줄 수 없다")
    if light is not None and nrm_in is None:
        raise ValueError("light 에는 normal 이 필요하다")

    excl = np.zeros(hw, bool)
    if ex_in is not None:
        excl |= ex_in
    if status is not None:
        excl |= status != 0
    valid_full = ~excl

    if geo_in is not None:
        mode = "geo_shading"
    elif light is not None:
        mode = "light"
    elif nrm_in is not None and (head_in is not None or ahead_in is not None):
        mode = "light_fit"
    else:
        mode = "none"
    if mode == "none" and head_in is None and ahead_in is None:
        raise IntrinsicsError("사전 원천이 없다: geo_shading/light/(normal+헤드)/shading_head/albedo_head 중 하나는 필요")

    # ---- 표본 격자(스트라이드) -------------------------------------------------------------------------------
    f = (max(h, w) + opts.max_side - 1) // opts.max_side
    sl = (slice(None, None, f), slice(None, None, f))
    I64 = I_in.astype(np.float64)
    I_s = I64[sl]
    valid_s = valid_full[sl]
    nrm_s = None if nrm_in is None else nrm_in.astype(np.float64)[sl]
    head_s = None if head_in is None else head_in.astype(np.float64)[sl]
    ahead_s = None if ahead_in is None else ahead_in.astype(np.float64)[sl]
    gw_s = gw_in.astype(np.float64)[sl]
    nrm_ok_s = None if nrm_s is None else (np.linalg.norm(nrm_s, axis=-1) >= NORMAL_MIN)

    warn = set()
    n_solves = 1
    final_light = light
    first_fit = None
    if mode == "geo_shading":
        geo_s = geo_in.astype(np.float64)[sl]
        s_s = _solve_ln(I_s, valid_s, nrm_s, geo_s, gw_s, head_s, ahead_s, opts)
    elif mode == "light":
        geo_s = _geo(nrm_s, light)
        s_s = _solve_ln(I_s, valid_s, nrm_s, geo_s, gw_s * nrm_ok_s, head_s, ahead_s, opts)
    elif mode == "light_fit":
        if head_s is not None:
            S_prev = head_s
        else:
            S_prev = (I_s @ _W709) / np.maximum(ahead_s @ _W709, 1e-3)
        n_solves = opts.n_alt
        for k in range(opts.n_alt):
            fit = _fit_core(nrm_s, S_prev, gw_s, valid_s)
            if k == 0:
                first_fit = fit
                used = _used_mask(np.linalg.norm(nrm_s, axis=-1), gw_s, valid_s)
                s0_mean = float((gw_s[used] * S_prev[used]).sum() / gw_s[used].sum())
                if fit.rms > opts.light_poor_rel * s0_mean:
                    warn.add("light_fit_poor")
            final_light = fit.light
            geo_s = _geo(nrm_s, final_light)
            s_s = _solve_ln(I_s, valid_s, nrm_s, geo_s, gw_s * nrm_ok_s, head_s, ahead_s, opts)
            S_prev = np.exp(s_s)
    else:
        geo_s = None
        s_s = _solve_ln(I_s, valid_s, nrm_s, None, gw_s, head_s, ahead_s, opts)
        warn.add("no_geometry")

    # ---- 전해상도로 올리고 제외 화소는 전해상도 폴백으로 덮어쓴다 ------------------------------------------------
    s_full = s_s if f == 1 else _upsample_ln(s_s, f, h, w)
    if mode == "geo_shading":
        geo_full = geo_in.astype(np.float64)
    elif mode in ("light", "light_fit"):
        geo_full = _geo(nrm_in.astype(np.float64), final_light)
    else:
        geo_full = None
    if excl.any():
        fb = _fallback_ln(I64, geo_full, None if head_in is None else head_in.astype(np.float64),
                          None if ahead_in is None else ahead_in.astype(np.float64))
        s_full = np.where(excl, fb, s_full)
    shading = np.maximum(np.exp(s_full).astype(np.float32), np.finfo(np.float32).tiny)
    L_d = I_in.astype(np.float32)               # 입력의 float32 복사(새 배열)
    albedo = (L_d.astype(np.float64) / shading.astype(np.float64)[..., None]).astype(np.float32)
    geo_out = None if geo_full is None else geo_full.astype(np.float32)

    n_valid = int(valid_full.sum())
    if geo_out is not None and n_valid > 0:
        d = (np.log(shading[valid_full].astype(np.float64))
             - np.log(np.maximum(geo_out[valid_full].astype(np.float64), S_MIN)))
        r_intr = float(math.sqrt(float(np.mean(d * d))))
    else:
        r_intr = float("nan")
    if 2 * int(excl.sum()) > excl.size:
        warn.add("excluded_dominant")

    return IntrinsicsResult(
        L_d=L_d, albedo=albedo, shading=shading, valid=valid_full.copy(), geo_shading=geo_out,
        light=final_light if mode != "none" else None, geo_source=mode if mode != "none" else "none",
        r_intr=r_intr, n_solves=int(n_solves), solve_stride=int(f), warnings=tuple(sorted(warn)), opts=opts)


# ---------------------------------------------------------------------------------------------------------------
# 재조명
# ---------------------------------------------------------------------------------------------------------------
def relight(result, *, light=None, geo_shading=None, normal=None, visibility=None, floor=GEO_FLOOR) -> np.ndarray:
    """out = L_d * max(Ŝ_new,floor) / max(Ŝ_old,floor). Ŝ_old = result.geo_shading. f32 (H,W,3)."""
    if not isinstance(result, IntrinsicsResult):
        raise TypeError(f"result: IntrinsicsResult 여야 한다 ({type(result).__name__})")
    if result.geo_shading is None:
        raise IntrinsicsError("기하 음영이 없는 결과는 재조명할 수 없다(geo_source='none')")
    fl = _real("floor", floor, lo=0.0, lo_open=True)
    if (light is None) == (geo_shading is None):
        raise ValueError("light 와 geo_shading 중 정확히 하나만 줘야 한다")
    hw = result.L_d.shape[:2]
    if light is not None:
        if not isinstance(light, GeoLight):
            raise TypeError(f"light: GeoLight 여야 한다 ({type(light).__name__})")
        if normal is None:
            raise ValueError("light 에는 normal 이 필요하다")
        n = _farray("normal", normal, shape=hw, ndim=3, last=3)
        if n.shape[:2] != hw:
            raise ValueError("normal: 모양이 결과와 다르다")
        vis = None
        if visibility is not None:
            v = _farray("visibility", visibility, shape=hw, ndim=2)
            if (v < 0).any() or (v > 1).any():
                raise ValueError("visibility: [0,1] 범위여야 한다")
            vis = v.astype(np.float64)
        new = _geo(n.astype(np.float64), light, vis)
    else:
        if normal is not None or visibility is not None:
            raise ValueError("geo_shading 과 함께 normal/visibility 를 줄 수 없다")
        new = _farray("geo_shading", geo_shading, shape=hw, ndim=2, nonneg=True).astype(np.float64)
    old = result.geo_shading.astype(np.float64)
    ratio = np.maximum(new, fl) / np.maximum(old, fl)
    return (result.L_d.astype(np.float64) * ratio[..., None]).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# RD-07 MRH 헤드 어댑터
# ---------------------------------------------------------------------------------------------------------------
def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        x = x.numpy()
    return np.asarray(x)


def _bilinear(a, H, W):
    """torch F.interpolate(mode='bilinear', align_corners=False) 와 같은 (c,h,w) -> (c,H,W) (float64)."""
    c, h, w = a.shape

    def axis(n_out, n_in):
        scale = n_in / n_out
        src = np.maximum((np.arange(n_out) + 0.5) * scale - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, src - i0

    y0, y1, fy = axis(H, h)
    x0, x1, fx = axis(W, w)
    fy = fy[None, :, None]
    fx = fx[None, None, :]
    top = (1 - fx) * a[:, y0][:, :, x0] + fx * a[:, y0][:, :, x1]
    bot = (1 - fx) * a[:, y1][:, :, x0] + fx * a[:, y1][:, :, x1]
    return (1 - fy) * top + fy * bot


def heads_from_readout(outputs, hw) -> dict:
    """MrhReadout.outputs 의 albedo/shading/normal 헤드 [1,c,h,w] 를 (H,W)로 키운다(f64). 없는 키는 None."""
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs: 매핑이어야 한다")
    try:
        H, W = hw
    except (TypeError, ValueError):
        raise ValueError("hw: (H, W) 여야 한다") from None
    if not (_is_int(H) and _is_int(W)) or H < 1 or W < 1:
        raise ValueError(f"hw: 양의 정수 쌍이어야 한다 ({hw})")
    H, W = int(H), int(W)
    out = {"albedo": None, "shading": None, "normal": None}
    for key, ch in (("albedo", 3), ("shading", 1), ("normal", 3)):
        v = outputs.get(key)
        if v is None:
            continue
        a = _to_numpy(v)
        if a.ndim != 4 or a.shape[0] != 1 or a.shape[1] != ch or a.shape[2] < 1 or a.shape[3] < 1:
            raise ValueError(f"{key}: [1,{ch},h,w] 여야 한다 (shape={a.shape})")
        a = a[0].astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f"{key}: 비유한 값이 있다")
        r = _bilinear(a, H, W)
        if key == "shading":
            out[key] = r[0]
        elif key == "albedo":
            out[key] = np.ascontiguousarray(np.moveaxis(r, 0, -1))
        else:
            n = np.ascontiguousarray(np.moveaxis(r, 0, -1))
            nn = np.linalg.norm(n, axis=-1, keepdims=True)
            out[key] = np.where(nn > 1e-6, n / np.maximum(nn, 1e-6), 0.0)
    return out
