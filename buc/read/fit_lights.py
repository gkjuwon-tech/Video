"""RD-10: R8 FitLights + 치트 라이트 v0 -- NNLS 세기 · CMA-ES 방향 · 그림자 에지 · CL1-CL6.

R7 음영 S (RD-09 ``decompose().shading``)와 법선으로 입력의 빛을 ``S ~ a + sum_j I_j V_j max(0, n.w_j)`` (a, I >= 0)로
역적합한다. 먼저 동기 광원 1개(키, ``"key"``)를 풀고, 남은 잔차만 치트 라이트(CL1)가 설명한다. 못 설명하면 폴백 =
복사휘도 굽기(``status="fallback_bake"``, ``geo_shading=None``). 음의 광원·light linking 없음(CL2), 치트는 유령(CL3),
광원 <= 4 / 플래그 <= 4 / 치트 에너지 몫 <= 35 % 는 넘으면 ``r_light`` 를 적색으로 올려 보고한다(CL4; 막지는 않는다).

좌표: 카메라(OpenCV), 법선은 카메라를 향함, w = 표면 -> 광원 단위벡터, 위 = -y.
numpy/scipy 만 쓴다(torch·yaml·cma 금지; 텐서는 ``.detach().cpu().numpy()`` 덕 타이핑). CMA-ES 는 직접 구현.
import 부작용·전역 RNG 없음(지역 ``np.random.default_rng(seed)``), 결정론(같은 입력 -> 비트 동일), 입력 불변.
RD-09 ``GeoLight`` 는 ``FitLightsResult.geo_light()`` 안에서만 지연 import 한다. RD-08 ``MetricScale`` 은 덕 타이핑.
"""
from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import optimize as _opt

__all__ = [
    "MAX_CHEAT_LIGHTS", "MAX_FLAGS", "CHEAT_SHARE_MAX", "SUN_ANGULAR_MAX", "PRIOR_LAMBDA", "NORMAL_MIN", "MIN_PIXELS",
    "CHEAT_ANGULAR_RADIUS", "KEY_NAME",
    "FitLightsError", "FitLightsOpts", "Light", "Flag", "DirectionPrior", "ShadowEdge", "IntensityFit", "CmaResult",
    "EnergyReport", "LightHint", "FitLightsResult",
    "transport", "render_shading", "solve_intensities", "cmaes_minimize", "find_shadow_edges",
    "shadow_direction_prior", "penumbra_angular_size", "classify_source", "direction_to_az_el", "energy_report",
    "cheat_shadow_index", "promote", "light_hint", "fit_lights",
]

MAX_CHEAT_LIGHTS = 4
MAX_FLAGS = 4
CHEAT_SHARE_MAX = 0.35
SUN_ANGULAR_MAX = 0.02
PRIOR_LAMBDA = 0.01
NORMAL_MIN = 0.5
MIN_PIXELS = 16
CHEAT_ANGULAR_RADIUS = 0.25
KEY_NAME = "key"

_S_FLOOR = 1e-4              # 그림자 에지의 ln 바닥
_VAR_MIN = 1e-12             # 음영 분산이 이 이하면 R^2 가 정의되지 않는다
_UNIT_TOL = 1e-6
_ROLES = ("motivated", "cheat")
_KINDS = ("sun", "area", "spot", "bounce")
_LOSSES = ("l2", "huber")
_SOURCES = ("user", "shadow", "probe")
_CAUSES = {"sun": "sky_open", "area": "window_or_lamp", "spot": "lamp", "bounce": "reflector"}
_COARSE_STARTS = 3           # 키 탐색에서 CMA-ES 를 시작할 격자 극소 수
_COARSE_SEP_COS = math.cos(math.radians(20.0))
_TANGENT_BOUND = 1.5         # 접선 평면 탐색 반경(비정규 방향 d0 + u e1 + v e2 의 u, v)
_REFINE_ROUNDS = 3
_HUBER_ITERS = 12


class FitLightsError(ValueError):
    """조명 적합이 정의되지 않는 입력(사용 화소 부족, 비유한 목적값, 상쇄된 방향 등)."""


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _real(name, v, *, lo=None, lo_open=False, hi=None) -> float:
    if not _is_real(v):
        raise TypeError(f"{name}: 실수여야 한다 ({type(v).__name__})")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name}: 유한해야 한다 ({x})")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f"{name}: {'>' if lo_open else '>='} {lo} 이어야 한다 ({x})")
    if hi is not None and x > hi:
        raise ValueError(f"{name}: <= {hi} 이어야 한다 ({x})")
    return x


def _int(name, v, *, lo=None, hi=None) -> int:
    if not _is_int(v):
        raise TypeError(f"{name}: 정수여야 한다 ({type(v).__name__})")
    x = int(v)
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ValueError(f"{name}: {lo}..{hi if hi is not None else '∞'} 이어야 한다 ({x})")
    return x


def _str(name, v) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name}: 문자열이어야 한다 ({type(v).__name__})")
    return v


def _to_np(a):
    """torch 텐서 등은 ``.detach().cpu().numpy()`` 로 덕 타이핑(torch import 없음)."""
    if not isinstance(a, np.ndarray) and hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return a


def _farray(name, a, *, shape=None, ndim=None, last=None, nonneg=False, finite=True) -> np.ndarray:
    """float32/float64 ndarray 검사(아니면 TypeError), 모양·유한·범위 위반은 ValueError. 복사하지 않는다."""
    a = _to_np(a)
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
    a = _to_np(a)
    if not isinstance(a, np.ndarray) or a.dtype != np.bool_:
        raise TypeError(f"{name}: bool ndarray 여야 한다")
    if a.shape != tuple(shape):
        raise ValueError(f"{name}: 모양 {tuple(shape)} 이어야 한다 (shape={a.shape})")
    return a


def _vec3(name, v, *, unit=False, norm_min=None) -> tuple:
    try:
        t = tuple(v)
    except TypeError:
        raise TypeError(f"{name}: 길이 3 시퀀스여야 한다") from None
    if len(t) != 3:
        raise ValueError(f"{name}: 길이 3 이어야 한다 ({len(t)})")
    t = tuple(_real(f"{name}[{i}]", x) for i, x in enumerate(t))
    nrm = math.sqrt(sum(x * x for x in t))
    if unit and abs(nrm - 1.0) >= _UNIT_TOL:
        raise ValueError(f"{name}: 단위 벡터여야 한다 ({t})")
    if norm_min is not None and nrm <= norm_min:
        raise ValueError(f"{name}: 길이 > {norm_min} 이어야 한다 ({nrm})")
    return t


def _unit_np(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _check_lights(lights) -> tuple:
    if isinstance(lights, (str, bytes)) or not hasattr(lights, "__iter__"):
        raise TypeError("lights: Light 시퀀스여야 한다")
    ls = tuple(lights)
    for x in ls:
        if not isinstance(x, Light):
            raise TypeError(f"lights: Light 여야 한다 ({type(x).__name__})")
    names = [x.name for x in ls]
    if len(set(names)) != len(names):
        raise ValueError(f"lights: 이름이 중복된다 ({names})")
    return ls


def _check_flags(flags, shape, *, limit=True) -> tuple:
    if isinstance(flags, (str, bytes)) or not hasattr(flags, "__iter__"):
        raise TypeError("flags: Flag 시퀀스여야 한다")
    fs = tuple(flags)
    for f in fs:
        if not isinstance(f, Flag):
            raise TypeError(f"flags: Flag 여야 한다 ({type(f).__name__})")
        if f.coverage.shape != tuple(shape):
            raise ValueError(f"flag {f.name!r}: coverage 모양 {tuple(shape)} 이어야 한다 ({f.coverage.shape})")
    if limit and len(fs) > MAX_FLAGS:
        raise ValueError(f"flags: 최대 {MAX_FLAGS}개 ({len(fs)})")
    return fs


def _check_vis_map(visibility, shape) -> dict:
    if visibility is None:
        return {}
    if not isinstance(visibility, Mapping):
        raise TypeError("visibility: {이름: (H,W)} 매핑이어야 한다")
    out = {}
    for k, v in visibility.items():
        a = _farray(f"visibility[{k!r}]", v, shape=shape, ndim=2)
        if (a < 0).any() or (a > 1).any():
            raise ValueError(f"visibility[{k!r}]: [0,1] 범위여야 한다")
        out[k] = a.astype(np.float64)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 자료형
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class FitLightsOpts:
    popsize: int = 16
    generations: int = 30
    max_cheat: int = MAX_CHEAT_LIGHTS
    min_gain: float = 0.02
    resid_tau: float = 0.02
    r2_min: float = 0.5
    n_coarse: int = 512
    seed: int = 0
    loss: str = "l2"

    def __post_init__(self):
        for n, lo, hi in (("popsize", 4, None), ("generations", 1, None), ("max_cheat", 0, MAX_CHEAT_LIGHTS),
                          ("n_coarse", 16, None), ("seed", 0, None)):
            object.__setattr__(self, n, _int(n, getattr(self, n), lo=lo, hi=hi))
        object.__setattr__(self, "min_gain", _real("min_gain", self.min_gain, lo=0.0, lo_open=True))
        object.__setattr__(self, "resid_tau", _real("resid_tau", self.resid_tau, lo=0.0))
        object.__setattr__(self, "r2_min", _real("r2_min", self.r2_min, lo=0.0, hi=1.0))
        if not isinstance(self.loss, str):
            raise TypeError(f"loss: 문자열이어야 한다 ({type(self.loss).__name__})")
        if self.loss not in _LOSSES:
            raise ValueError(f"loss: {_LOSSES} 중 하나여야 한다 ({self.loss!r})")


def direction_to_az_el(d) -> tuple:
    """(방위°, 고도°). 고도 = asin(-d_y)(위 = -y 가 +), 방위 = atan2(d_x, d_z)(정면 +z 가 0, 오른쪽 +)."""
    x, y, z = _vec3("direction", d, unit=True)
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, -y))))


@dataclass(frozen=True, kw_only=True)
class Light:
    name: str
    role: str
    kind: str
    direction: tuple
    intensity: float
    angular_radius: float = 0.0

    def __post_init__(self):
        if _str("name", self.name) == "":
            raise ValueError("name: 비어 있지 않아야 한다")
        if _str("role", self.role) not in _ROLES:
            raise ValueError(f"role: {_ROLES} 중 하나여야 한다 ({self.role!r})")
        if _str("kind", self.kind) not in _KINDS:
            raise ValueError(f"kind: {_KINDS} 중 하나여야 한다 ({self.kind!r})")
        if self.role == "cheat" and self.kind == "sun":
            raise ValueError("치트 라이트는 sun 일 수 없다 (면광·스폿·반사판만)")
        if self.role == "motivated" and self.kind == "bounce":
            raise ValueError("bounce 는 치트 전용이다")
        object.__setattr__(self, "direction", _vec3("direction", self.direction, unit=True))
        object.__setattr__(self, "intensity", _real("intensity", self.intensity, lo=0.0))
        object.__setattr__(self, "angular_radius",
                           _real("angular_radius", self.angular_radius, lo=0.0, hi=math.pi / 2))

    @property
    def ghost(self) -> bool:
        """CL3: 치트 = 카메라 1차·정반사 경로에서 보이지 않는 유령."""
        return self.role == "cheat"

    def to_meta(self) -> dict:
        az, el = direction_to_az_el(self.direction)
        return {"name": self.name, "role": self.role, "kind": self.kind, "direction": [float(v) for v in self.direction],
                "intensity": float(self.intensity), "angular_radius": float(self.angular_radius),
                "ghost": self.ghost, "cheat": 1 if self.role == "cheat" else 0,
                "azimuth_deg": az, "elevation_deg": el}


@dataclass(frozen=True, eq=False)
class Flag:
    """차광판: 지명한 광원의 수송에만 ``(1 - coverage)`` 를 곱한다."""
    name: str
    light: str
    coverage: np.ndarray

    def __post_init__(self):
        if _str("name", self.name) == "":
            raise ValueError("name: 비어 있지 않아야 한다")
        _str("light", self.light)
        c = _farray("coverage", self.coverage, ndim=2)
        if (c < 0).any() or (c > 1).any():
            raise ValueError("coverage: [0,1] 범위여야 한다")
        c = np.array(c, dtype=np.float64)
        c.flags.writeable = False
        object.__setattr__(self, "coverage", c)


@dataclass(frozen=True)
class DirectionPrior:
    direction: tuple
    sigma_rad: float
    source: str = "user"

    def __post_init__(self):
        object.__setattr__(self, "direction", _vec3("direction", self.direction, unit=True))
        object.__setattr__(self, "sigma_rad", _real("sigma_rad", self.sigma_rad, lo=0.0, lo_open=True))
        if _str("source", self.source) not in _SOURCES:
            raise ValueError(f"source: {_SOURCES} 중 하나여야 한다 ({self.source!r})")


@dataclass(frozen=True)
class ShadowEdge:
    """드리운 그림자의 점(표면)과 가림체 에지 점의 쌍: 광선 방향 = 표면 -> 광원."""
    shadow_point: tuple
    occluder_point: tuple
    weight: float = 1.0

    def __post_init__(self):
        sp = _vec3("shadow_point", self.shadow_point)
        op = _vec3("occluder_point", self.occluder_point)
        if math.dist(sp, op) <= 1e-9:
            raise ValueError("shadow_point 와 occluder_point 의 거리는 > 1e-9 여야 한다")
        object.__setattr__(self, "shadow_point", sp)
        object.__setattr__(self, "occluder_point", op)
        object.__setattr__(self, "weight", _real("weight", self.weight, lo=0.0, lo_open=True))


@dataclass(frozen=True, eq=False)
class IntensityFit:
    intensities: np.ndarray
    ambient: float
    rms: float
    r2: float
    nmse: float
    n_used: int


@dataclass(frozen=True, eq=False)
class CmaResult:
    x: np.ndarray
    fun: float
    n_evals: int
    history: tuple


@dataclass(frozen=True)
class EnergyReport:
    energies: tuple
    ambient_energy: float
    shares: tuple
    cheat_share: float
    n_cheat: int
    n_flags: int
    r_light: float
    red: bool


@dataclass(frozen=True)
class LightHint:
    az_deg: float
    el_deg: float
    cause: str
    distance_m: tuple | None
    distance_rel: tuple | None
    metric_reliable: bool | None


def _clean(o):
    """JSON(allow_nan=False)용: nan/inf -> None, numpy 스칼라·배열·튜플 -> 파이썬 값."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, numbers.Integral):
        return int(o)
    if isinstance(o, numbers.Real):
        x = float(o)
        return x if math.isfinite(x) else None
    return o


@dataclass(frozen=True, eq=False)
class FitLightsResult:
    status: str
    lights: tuple
    ambient: float
    flags: tuple
    geo_shading: np.ndarray | None
    r2: float
    rms: float
    nmse: float
    cheat_share: float
    r_light: float
    red: bool
    n_used: int
    n_evals: int
    warnings: tuple
    priors: tuple
    energies: tuple
    opts: FitLightsOpts

    @property
    def recommend_safe(self) -> bool:
        return self.red

    @property
    def reliable(self) -> bool:
        return self.status == "ok" and not self.warnings

    def geo_light(self):
        """키 방향·세기와 ambient 를 RD-09 ``GeoLight`` 로. 폴백이면 None."""
        if self.status != "ok" or not self.lights:
            return None
        from buc.read.intrinsics import GeoLight        # 지연 import (RD-09)
        key = self.lights[0]
        return GeoLight(direction=key.direction, intensity=key.intensity, ambient=self.ambient)

    def meta(self) -> dict:
        total = self.ambient + float(sum(self.energies))
        share = [(e / total if total > 0 else 0.0) for e in self.energies]
        mot, cheat = [], []
        for light, sh in zip(self.lights, share):
            m = light.to_meta()
            m["energy_share"] = sh
            (cheat if light.role == "cheat" else mot).append(m)
        out = {"status": self.status, "motivated": mot, "cheat": cheat,
               "probe": [{"direction": list(p.direction), "sigma_rad": p.sigma_rad}
                         for p in self.priors if p.source == "probe"],
               "priors": [{"direction": list(p.direction), "sigma_rad": p.sigma_rad, "source": p.source}
                          for p in self.priors],
               "ambient": self.ambient, "r2": self.r2, "rms": self.rms, "nmse": self.nmse,
               "cheat_share": self.cheat_share, "r_light": self.r_light, "red": self.red,
               "recommend_safe": self.recommend_safe, "n_used": self.n_used, "n_evals": self.n_evals,
               "warnings": list(self.warnings), "reliable": self.reliable,
               "flags": [{"name": f.name, "light": f.light} for f in self.flags]}
        return _clean(out)


# ---------------------------------------------------------------------------------------------------------------
# 수송 · 렌더
# ---------------------------------------------------------------------------------------------------------------
def transport(normal, direction, *, visibility=None) -> np.ndarray:
    """T = V * max(0, n.w) (n 정규화 안 함, V in [0,1]). f64 (H,W). RD-09 ``geometric_shading(ambient=0, intensity=1)`` 와 같다."""
    n = _farray("normal", normal, ndim=3, last=3)
    d = np.asarray(_vec3("direction", direction, unit=True), np.float64)
    t = np.maximum(n.astype(np.float64) @ d, 0.0)
    if visibility is not None:
        v = _farray("visibility", visibility, shape=n.shape[:2], ndim=2)
        if (v < 0).any() or (v > 1).any():
            raise ValueError("visibility: [0,1] 범위여야 한다")
        t = t * v.astype(np.float64)
    return t


def _flag_att(flags, name, shape) -> np.ndarray | None:
    att = None
    for f in flags:
        if f.light == name:
            att = (1.0 - f.coverage) if att is None else att * (1.0 - f.coverage)
    return att


def _light_transport(n64, light, vis_map, flags) -> np.ndarray:
    t = np.maximum(n64 @ np.asarray(light.direction, np.float64), 0.0)
    v = vis_map.get(light.name)
    if v is not None:
        t = t * v
    att = _flag_att(flags, light.name, t.shape)
    if att is not None:
        t = t * att
    return t


def render_shading(lights, normal, *, ambient=0.0, visibility=None, flags=()) -> np.ndarray:
    """``a + sum_j I_j T_j prod_{플래그 -> j}(1 - cov)``. visibility 는 {이름: (H,W)}. f64 (H,W)."""
    n = _farray("normal", normal, ndim=3, last=3)
    ls = _check_lights(lights)
    a = _real("ambient", ambient, lo=0.0)
    fs = _check_flags(flags, n.shape[:2])
    vm = _check_vis_map(visibility, n.shape[:2])
    n64 = n.astype(np.float64)
    s = np.full(n.shape[:2], a, np.float64)
    for light in ls:
        s = s + light.intensity * _light_transport(n64, light, vm, fs)
    return s


# ---------------------------------------------------------------------------------------------------------------
# NNLS 세기
# ---------------------------------------------------------------------------------------------------------------
def _nnls(A, b):
    try:
        p, _ = _opt.nnls(A, b)
    except RuntimeError:                       # 반복 한도: 같은 문제를 경계 최소제곱으로
        p = _opt.lsq_linear(A, b, bounds=(0.0, np.inf)).x
    return np.maximum(p, 0.0)


def _solve_core(T, s, w, ambient, loss):
    """T (J,M), s (M,), w (M,) > 0. 반환 (p (J[+1],), 통계 dict). p = [I..., a]."""
    J, M = T.shape
    K = J + (1 if ambient else 0)
    A = np.empty((M, K), np.float64)
    A[:, :J] = T.T
    if ambient:
        A[:, J] = 1.0
    sw = np.sqrt(w)
    p = _nnls(A * sw[:, None], s * sw)
    if loss == "huber":
        for _ in range(_HUBER_ITERS):
            r = s - A @ p
            sig = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            if sig <= 1e-12:
                break
            c = 1.345 * sig
            wh = w * np.minimum(1.0, c / np.maximum(np.abs(r), 1e-300))
            sh = np.sqrt(wh)
            p = _nnls(A * sh[:, None], s * sh)
    r = s - A @ p
    sw_sum = float(w.sum())
    sse = float(np.sum(w * r * r))
    mean = float(np.sum(w * s) / sw_sum)
    den = float(np.sum(w * (s - mean) ** 2))
    if den / sw_sum <= _VAR_MIN:
        nmse = r2 = math.nan
    else:
        nmse = sse / den
        r2 = 1.0 - nmse
    return p, {"nmse": nmse, "r2": r2, "rms": math.sqrt(sse / sw_sum), "sse": sse, "den": den}


def solve_intensities(T, shading, *, weight=None, ambient=True, loss="l2") -> IntensityFit:
    """``min sum w (S - a - sum I_j T_j)^2`` s.t. ``I, a >= 0`` (NNLS, 전역 최적). huber 는 12회 IRLS."""
    T = _farray("T", T, ndim=3)
    S = _farray("shading", shading, shape=T.shape[1:], ndim=2)
    if not isinstance(loss, str):
        raise TypeError("loss: 문자열이어야 한다")
    if loss not in _LOSSES:
        raise ValueError(f"loss: {_LOSSES} 중 하나여야 한다 ({loss!r})")
    if weight is None:
        w = np.ones(S.shape, np.float64)
    else:
        w = _farray("weight", weight, shape=S.shape, ndim=2, nonneg=True).astype(np.float64)
    use = w > 0
    n_used = int(use.sum())
    J = T.shape[0]
    K = J + (1 if ambient else 0)
    if n_used < K:
        raise FitLightsError(f"사용 화소 {n_used} 가 미지수 {K} 보다 적다")
    Tm = T.astype(np.float64)[:, use]
    p, st = _solve_core(Tm, S.astype(np.float64)[use], w[use], bool(ambient), loss)
    return IntensityFit(intensities=p[:J].copy(), ambient=float(p[J]) if ambient else 0.0, rms=st["rms"],
                        r2=st["r2"], nmse=st["nmse"], n_used=n_used)


# ---------------------------------------------------------------------------------------------------------------
# CMA-ES (직접 구현: 표준 (mu/mu_w, lambda), 경계는 후보 클립)
# ---------------------------------------------------------------------------------------------------------------
def _bound(name, b, n, default):
    if b is None:
        return np.full(n, default, np.float64)
    a = np.asarray(b, np.float64) if not isinstance(b, np.ndarray) else b.astype(np.float64)
    if a.ndim == 0:
        a = np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name}: 스칼라 또는 길이 {n} 이어야 한다 (shape={a.shape})")
    if np.isnan(a).any():
        raise ValueError(f"{name}: nan 이 있다")
    return a


def cmaes_minimize(f, x0, sigma0, *, lower=None, upper=None, popsize=16, generations=30, seed=0) -> CmaResult:
    """f 의 극소를 표준 CMA-ES 로. 평가는 정확히 ``popsize * generations`` 회, ``history`` 는 세대별 최선값(비증가)."""
    if not callable(f):
        raise TypeError("f: 호출 가능해야 한다")
    popsize = _int("popsize", popsize, lo=4)
    generations = _int("generations", generations, lo=1)
    seed = _int("seed", seed, lo=0)
    sigma0 = _real("sigma0", sigma0, lo=0.0, lo_open=True)
    x0 = _to_np(x0)
    try:
        x0 = np.array(x0, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError("x0: 실수 1차원 배열이어야 한다") from None
    if x0.ndim != 1 or x0.size < 1:
        raise ValueError(f"x0: 1차원이어야 한다 (shape={x0.shape})")
    if not np.isfinite(x0).all():
        raise ValueError("x0: 비유한 값이 있다")
    n = x0.size
    lo = _bound("lower", lower, n, -np.inf)
    hi = _bound("upper", upper, n, np.inf)
    if not (lo < hi).all():
        raise ValueError("lower < upper 여야 한다")

    rng = np.random.default_rng(seed)
    lam = popsize
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts = wts / wts.sum()
    mueff = 1.0 / float(np.sum(wts ** 2))
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    m = np.clip(x0, lo, hi)
    sigma = float(sigma0)
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    best_x = m.copy()
    best_f = math.inf
    history = []
    n_evals = 0
    for g in range(generations):
        z = rng.standard_normal((lam, n))
        y = (z * D) @ B.T
        x = np.clip(m + sigma * y, lo, hi)
        fx = np.empty(lam)
        for k in range(lam):
            v = float(f(x[k].copy()))
            n_evals += 1
            if not math.isfinite(v):
                raise FitLightsError(f"목적값이 비유한이다 ({v})")
            fx[k] = v
        order = np.argsort(fx, kind="stable")
        if fx[order[0]] < best_f:
            best_f = float(fx[order[0]])
            best_x = x[order[0]].copy()
        history.append(best_f)
        ye = (x - m) / sigma
        ysel = ye[order[:mu]]
        yw = wts @ ysel
        m_new = np.clip(m + sigma * yw, lo, hi)
        inv_sqrt_yw = B @ ((B.T @ yw) / D)
        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) * inv_sqrt_yw
        hsig = 1.0 if (np.linalg.norm(ps) / math.sqrt(1.0 - (1.0 - cs) ** (2 * (g + 1))) / chi_n
                       < 1.4 + 2.0 / (n + 1.0)) else 0.0
        pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * yw
        rank_mu = (ysel * wts[:, None]).T @ ysel
        C = ((1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) + cmu * rank_mu)
        sigma = float(np.clip(sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0)), 1e-300, 1e300))
        C = 0.5 * (C + C.T)
        ev, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(ev, 1e-30))
        m = m_new
    return CmaResult(x=best_x, fun=best_f, n_evals=n_evals, history=tuple(history))


# ---------------------------------------------------------------------------------------------------------------
# 그림자 에지 · 반그림자 · 방향 사전분포
# ---------------------------------------------------------------------------------------------------------------
def find_shadow_edges(shading, normal, *, valid=None, grad_tau=0.15, normal_tau=0.1) -> np.ndarray:
    """4-이웃 쌍이 둘 다 유효하고 ``|ln S| 차 > grad_tau`` ∧ ``||n_x - n_y|| < normal_tau`` 이면 두 화소를 표시한다."""
    s = _farray("shading", shading, ndim=2)
    n = _farray("normal", normal, shape=s.shape, ndim=3, last=3)
    gt = _real("grad_tau", grad_tau, lo=0.0, lo_open=True)
    nt = _real("normal_tau", normal_tau, lo=0.0, lo_open=True)
    v = np.ones(s.shape, bool) if valid is None else _barray("valid", valid, s.shape)
    ls = np.log(np.maximum(s.astype(np.float64), _S_FLOOR))
    n64 = n.astype(np.float64)
    out = np.zeros(s.shape, bool)
    for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        ok = v[sl_a] & v[sl_b]
        ok &= np.abs(ls[sl_a] - ls[sl_b]) > gt
        ok &= np.linalg.norm(n64[sl_a] - n64[sl_b], axis=-1) < nt
        out[sl_a] |= ok
        out[sl_b] |= ok
    return out


def shadow_direction_prior(edges, *, sigma_floor=0.02) -> DirectionPrior:
    """``u_i = unit(occluder - shadow)``; 방향 = ``unit(sum w u)``; ``sigma = max(floor, sqrt(sum w th^2 / sum w))``."""
    if isinstance(edges, (str, bytes)) or not hasattr(edges, "__iter__"):
        raise TypeError("edges: ShadowEdge 시퀀스여야 한다")
    es = tuple(edges)
    for e in es:
        if not isinstance(e, ShadowEdge):
            raise TypeError(f"edges: ShadowEdge 여야 한다 ({type(e).__name__})")
    floor = _real("sigma_floor", sigma_floor, lo=0.0, lo_open=True)
    if not es:
        raise FitLightsError("그림자 에지가 없다")
    u = np.array([_unit_np(np.subtract(e.occluder_point, e.shadow_point)) for e in es])
    w = np.array([e.weight for e in es])
    mean = w @ u
    nm = float(np.linalg.norm(mean))
    if nm < 1e-9 * float(w.sum()):
        raise FitLightsError("에지 방향이 상쇄되어 방향이 정해지지 않는다")
    d = mean / nm
    th = np.arctan2(np.linalg.norm(np.cross(u, d), axis=1), u @ d)
    sigma = max(floor, math.sqrt(float(np.sum(w * th * th) / w.sum())))
    return DirectionPrior(direction=tuple(float(v) for v in d), sigma_rad=sigma, source="shadow")


def penumbra_angular_size(width, distance) -> float:
    """반그림자 폭 / 거리 = 광원 각크기 (rad)."""
    return _real("width", width, lo=0.0, lo_open=True) / _real("distance", distance, lo=0.0, lo_open=True)


def classify_source(theta_s) -> str:
    return "sun" if _real("theta_s", theta_s, lo=0.0) < SUN_ANGULAR_MAX else "area"


# ---------------------------------------------------------------------------------------------------------------
# 에너지 몫 (CL4) · 치트 그림자 지표 (CL5)
# ---------------------------------------------------------------------------------------------------------------
def energy_report(lights, normal, *, ambient, weight=None, visibility=None, flags=()) -> EnergyReport:
    """``E_j = sum w I_j T_j / sum w``, 몫 = ``E_j / (a + sum E)``. ``r_light = max(cheat_share/0.35, n_cheat/4, n_flags/4)``."""
    n = _farray("normal", normal, ndim=3, last=3)
    ls = _check_lights(lights)
    a = _real("ambient", ambient, lo=0.0)
    fs = _check_flags(flags, n.shape[:2], limit=False)          # CL4: 넘으면 막지 않고 적색으로 보고
    vm = _check_vis_map(visibility, n.shape[:2])
    if weight is None:
        w = np.ones(n.shape[:2], np.float64)
    else:
        w = _farray("weight", weight, shape=n.shape[:2], ndim=2, nonneg=True).astype(np.float64)
    sw = float(w.sum())
    if sw <= 0:
        raise FitLightsError("weight 의 합이 0 이다")
    n64 = n.astype(np.float64)
    e = tuple(float(np.sum(w * (light.intensity * _light_transport(n64, light, vm, fs))) / sw) for light in ls)
    total = a + float(sum(e))
    shares = tuple((x / total if total > 0 else 0.0) for x in e)
    cheat_share = float(sum(sh for sh, light in zip(shares, ls) if light.role == "cheat"))
    n_cheat = sum(1 for light in ls if light.role == "cheat")
    r_light = max(cheat_share / CHEAT_SHARE_MAX, n_cheat / MAX_CHEAT_LIGHTS, len(fs) / MAX_FLAGS)
    return EnergyReport(energies=e, ambient_energy=a, shares=shares, cheat_share=cheat_share, n_cheat=n_cheat,
                        n_flags=len(fs), r_light=float(r_light), red=bool(r_light > 1.0))


def cheat_shadow_index(lights, normal, visibility, *, valid=None, vis_thresh=0.5) -> float:
    """유효 화소 중 어떤 치트 광원이 비추는 면(n.w > 0)인데 그 광원의 V < vis_thresh 인 비율(CL5)."""
    n = _farray("normal", normal, ndim=3, last=3)
    ls = _check_lights(lights)
    vm = _check_vis_map(visibility, n.shape[:2])
    thr = _real("vis_thresh", vis_thresh)
    v = np.ones(n.shape[:2], bool) if valid is None else _barray("valid", valid, n.shape[:2])
    nv = int(v.sum())
    if nv == 0:
        return 0.0
    n64 = n.astype(np.float64)
    hit = np.zeros(n.shape[:2], bool)
    for light in ls:
        if light.role != "cheat" or light.name not in vm:
            continue
        lit = (n64 @ np.asarray(light.direction, np.float64)) > 0.0
        hit |= lit & (vm[light.name] < thr)
    return float((hit & v).sum()) / float(nv)


# ---------------------------------------------------------------------------------------------------------------
# 승격 · 인과 힌트
# ---------------------------------------------------------------------------------------------------------------
def promote(light) -> Light:
    """치트 -> 동기(이름·종류·값 유지, 새 객체). 동기·bounce 는 ValueError."""
    if not isinstance(light, Light):
        raise TypeError(f"light: Light 여야 한다 ({type(light).__name__})")
    if light.role != "cheat":
        raise ValueError("치트 라이트만 승격할 수 있다")
    if light.kind == "bounce":
        raise ValueError("bounce 는 동기 광원이 될 수 없다")
    return dataclasses.replace(light, role="motivated")


def light_hint(light, *, source_size_m=(0.5, 1.5), metric=None) -> LightHint:
    """MassingFM 인과 힌트: 방위·고도·원인, 면광이면 광원 거리 구간(미터, 스케일 있으면 상대 단위)."""
    if not isinstance(light, Light):
        raise TypeError(f"light: Light 여야 한다 ({type(light).__name__})")
    try:
        lo_s, hi_s = source_size_m
    except (TypeError, ValueError):
        raise TypeError("source_size_m: (최소, 최대) 두 값이어야 한다") from None
    lo_s = _real("source_size_m[0]", lo_s, lo=0.0, lo_open=True)
    hi_s = _real("source_size_m[1]", hi_s, lo=0.0, lo_open=True)
    if not lo_s < hi_s:
        raise ValueError(f"source_size_m: 0 < 최소 < 최대 여야 한다 ({lo_s}, {hi_s})")
    az, el = direction_to_az_el(light.direction)
    dist_m = dist_rel = None
    if light.kind == "area" and light.angular_radius > 0:
        theta = 2.0 * light.angular_radius
        dist_m = (lo_s / theta, hi_s / theta)
    reliable = None
    if metric is not None:
        s_lo, s_hi = metric.interval(2.0)
        reliable = bool(metric.reliable)
        if dist_m is not None:
            dist_rel = (dist_m[0] / s_hi, dist_m[1] / s_lo)
    return LightHint(az_deg=az, el_deg=el, cause=_CAUSES[light.kind], distance_m=dist_m, distance_rel=dist_rel,
                     metric_reliable=reliable)


# ---------------------------------------------------------------------------------------------------------------
# fit_lights
# ---------------------------------------------------------------------------------------------------------------
def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    ph = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def _tangent_basis(d):
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(d)))] = 1.0
    e1 = np.cross(d, axis)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def _angle(a, b) -> float:
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


class _Ctx:
    """사용 화소로 줄인 문제: 수송·가시성·플래그·NNLS 를 한곳에서. 평가 횟수를 센다."""

    def __init__(self, nu, su, wu, used, vis_fn, flags, loss, priors, lam):
        self.nu, self.su, self.wu, self.used = nu, su, wu, used
        self.vis_fn = vis_fn
        self.loss = loss
        self.priors = priors
        self.lam = lam
        self.att = {}
        for f in flags:
            c = 1.0 - f.coverage[used]
            self.att[f.light] = c if f.light not in self.att else self.att[f.light] * c
        self.n_evals = 0

    def T(self, name, d, *, use_vis=True):
        t = np.maximum(self.nu @ d, 0.0)
        if use_vis and self.vis_fn is not None:
            t = t * self.vis_fn(d)[self.used]
        a = self.att.get(name)
        return t if a is None else t * a

    def fit(self, Ts):
        self.n_evals += 1
        return _solve_core(np.asarray(Ts), self.su, self.wu, True, self.loss)

    def penalty(self, d) -> float:
        return self.lam * sum((_angle(d, np.asarray(p.direction)) / p.sigma_rad) ** 2 for p in self.priors)


def _refine_dir(ctx, names, dirs, Ts, j, opts, sigma0, seed):
    """광원 j 의 방향만 접선 평면 2차원 CMA-ES 로 정련(다른 광원 고정). 비용 = NMSE (+ 키면 사전 벌점)."""
    d0 = dirs[j]
    e1, e2 = _tangent_basis(d0)
    def to_dir(x):
        return _unit_np(d0 + x[0] * e1 + x[1] * e2)

    def cost(x):
        d = to_dir(x)
        trial = list(Ts)
        trial[j] = ctx.T(names[j], d)
        _, st = ctx.fit(trial)
        return st["nmse"] + (ctx.penalty(d) if j == 0 else 0.0)

    res = cmaes_minimize(cost, [0.0, 0.0], sigma0, lower=-_TANGENT_BOUND, upper=_TANGENT_BOUND,
                         popsize=opts.popsize, generations=opts.generations, seed=seed)
    return to_dir(res.x), res.fun


def _model_cost(ctx, names, dirs):
    Ts = [ctx.T(nm, d) for nm, d in zip(names, dirs)]
    p, st = ctx.fit(Ts)
    return Ts, p, st, st["nmse"] + ctx.penalty(dirs[0])


def _alternate(ctx, names, dirs, opts, sigma0, seed, order):
    """좌표 교대 정련: order 순으로 한 번에 한 광원의 방향을 CMA-ES 로. 개선이 없으면 일찍 끝낸다."""
    dirs = [d.copy() for d in dirs]
    Ts, _, _, cost = _model_cost(ctx, names, dirs)
    for rnd in range(_REFINE_ROUNDS):
        before = cost
        for j in order:
            nd, fun = _refine_dir(ctx, names, dirs, Ts, j, opts, sigma0 * (0.5 ** rnd), seed + 97 * rnd + 13 * j)
            if fun < cost - 1e-15:
                dirs[j] = nd
                Ts[j] = ctx.T(names[j], nd)
                cost = fun
        if before - cost < 1e-9:
            break
    Ts, p, st, cost = _model_cost(ctx, names, dirs)
    return dirs, p, st, cost


def _fallback(status_reason, n_used, n_evals, flags, priors, opts, r2=math.nan, rms=math.nan, nmse=math.nan):
    warns = tuple(sorted({"fallback_bake", status_reason}))
    return FitLightsResult(status="fallback_bake", lights=(), ambient=0.0, flags=(), geo_shading=None, r2=r2, rms=rms,
                           nmse=nmse, cheat_share=0.0, r_light=0.0, red=False, n_used=n_used, n_evals=n_evals,
                           warnings=warns, priors=priors, energies=(), opts=opts)


def fit_lights(shading, normal, *, valid=None, weight=None, visibility=None, init_light=None,
               direction_priors=(), flags=(), penumbra=None, opts=None) -> FitLightsResult:
    """R8: 키(동기) 1개를 NNLS+CMA-ES 로 풀고, 남은 잔차만 치트 라이트로 설명한다. 못 설명하면 복사휘도 굽기 폴백."""
    s = _farray("shading", shading, ndim=2, nonneg=True)
    n = _farray("normal", normal, ndim=3, last=3)
    if s.shape[0] < 4 or s.shape[1] < 4:
        raise ValueError(f"shading: H, W >= 4 여야 한다 (shape={s.shape})")
    if n.shape[:2] != s.shape:
        raise ValueError(f"normal: 모양 {s.shape + (3,)} 이어야 한다 (shape={n.shape})")
    shape = s.shape
    v = np.ones(shape, bool) if valid is None else _barray("valid", valid, shape)
    w = (np.ones(shape, np.float64) if weight is None
         else _farray("weight", weight, shape=shape, ndim=2, nonneg=True).astype(np.float64))
    if visibility is not None and not callable(visibility):
        raise TypeError("visibility: 호출 가능 (단위방향 (3,)) -> (H,W) 여야 한다")
    fs = _check_flags(flags, shape)
    if isinstance(direction_priors, (str, bytes)) or not hasattr(direction_priors, "__iter__"):
        raise TypeError("direction_priors: DirectionPrior 시퀀스여야 한다")
    priors = tuple(direction_priors)
    for p in priors:
        if not isinstance(p, DirectionPrior):
            raise TypeError(f"direction_priors: DirectionPrior 여야 한다 ({type(p).__name__})")
    init_dir = None
    if init_light is not None:
        if not hasattr(init_light, "direction"):
            raise TypeError("init_light: .direction 이 있는 객체여야 한다")
        init_dir = _unit_np(_vec3("init_light.direction", init_light.direction, norm_min=1e-9))
    theta_s = None
    if penumbra is not None:
        try:
            pw, pd = penumbra
        except (TypeError, ValueError):
            raise TypeError("penumbra: (폭, 거리) 두 값이어야 한다") from None
        theta_s = penumbra_angular_size(pw, pd)
    if opts is None:
        opts = FitLightsOpts()
    elif not isinstance(opts, FitLightsOpts):
        raise TypeError(f"opts: FitLightsOpts 여야 한다 ({type(opts).__name__})")

    # ---- 사용 화소 --------------------------------------------------------------------------------------------
    n64 = n.astype(np.float64)
    nrm = np.linalg.norm(n64, axis=-1)
    used = v & (w > 0) & (nrm >= NORMAL_MIN)
    n_used = int(used.sum())
    if n_used < MIN_PIXELS:
        return _fallback("few_pixels", n_used, 0, fs, priors, opts)
    n_eff = np.where((nrm >= NORMAL_MIN)[..., None], n64 / np.maximum(nrm, 1e-300)[..., None], n64)
    nu = n_eff[used]
    su = s.astype(np.float64)[used]
    wu = w[used]
    sw = float(wu.sum())
    mean = float(np.sum(wu * su) / sw)
    if float(np.sum(wu * (su - mean) ** 2)) / sw <= _VAR_MIN:
        return _fallback("degenerate_shading", n_used, 0, fs, priors, opts)

    def vis_fn(d):
        """콜백을 단위 방향으로 부르고 [0,1] (H,W) 임을 확인한다."""
        out = _to_np(visibility(np.array(d, dtype=np.float64)))
        if not isinstance(out, np.ndarray) or out.dtype not in (np.float32, np.float64):
            raise TypeError("visibility 콜백: float32/float64 ndarray 를 돌려줘야 한다")
        if out.shape != shape:
            raise ValueError(f"visibility 콜백: 모양 {shape} 이어야 한다 (shape={out.shape})")
        if not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
            raise ValueError("visibility 콜백: [0,1] 유한값이어야 한다")
        return out.astype(np.float64)

    ctx = _Ctx(nu, su, wu, used, vis_fn if visibility is not None else None, fs, opts.loss, priors, PRIOR_LAMBDA)
    seed = opts.seed

    # ---- 1. 키 ------------------------------------------------------------------------------------------------
    grid = _fib_sphere(opts.n_coarse)
    key_prior_pen = np.array([ctx.penalty(g) for g in grid])
    cg = np.empty(len(grid))
    for i, g in enumerate(grid):
        _, st = ctx.fit([ctx.T(KEY_NAME, g, use_vis=False)])
        cg[i] = st["nmse"] + key_prior_pen[i]
    starts = []          # (방향, sigma0)
    for i in np.argsort(cg, kind="stable"):
        if all(float(g @ grid[i]) < _COARSE_SEP_COS for g, _ in starts):
            starts.append((grid[i].copy(), 0.25))
        if len(starts) >= _COARSE_STARTS:
            break
    extra = []
    if init_dir is not None:
        extra.append((init_dir, 0.2))
    for p in priors:
        extra.append((_unit_np(p.direction), min(0.3, max(0.05, p.sigma_rad))))
    for d, sg in extra:
        if all(float(d @ g) < math.cos(math.radians(1.0)) for g, _ in starts):
            starts.append((d, sg))
    best = None
    for si, (d0, sg) in enumerate(starts):
        e1, e2 = _tangent_basis(d0)

        def cost(x, d0=d0, e1=e1, e2=e2):
            d = _unit_np(d0 + x[0] * e1 + x[1] * e2)
            _, st = ctx.fit([ctx.T(KEY_NAME, d)])
            return st["nmse"] + ctx.penalty(d)

        res = cmaes_minimize(cost, [0.0, 0.0], sg, lower=-_TANGENT_BOUND, upper=_TANGENT_BOUND,
                             popsize=opts.popsize, generations=opts.generations, seed=seed + 1000 * (si + 1))
        d = _unit_np(d0 + res.x[0] * e1 + res.x[1] * e2)
        if best is None or res.fun < best[0]:
            best = (res.fun, d)
    names = [KEY_NAME]
    dirs = [best[1]]
    dirs, p, st, _ = _alternate(ctx, names, dirs, opts, 0.05, seed + 5000, [0])
    cur_nmse = st["nmse"]

    # ---- 2. 치트 라이트 (CL1) ---------------------------------------------------------------------------------
    k = 0
    while len(names) - 1 < opts.max_cheat and cur_nmse >= opts.resid_tau:
        k += 1
        cname = f"cheat{k}"
        Tfix = [ctx.T(nm, d) for nm, d in zip(names, dirs)]
        cc = np.empty(len(grid))
        for i, g in enumerate(grid):
            _, stg = ctx.fit(Tfix + [ctx.T(cname, g, use_vis=False)])
            cc[i] = stg["nmse"]
        g0 = grid[int(np.argmin(cc))].copy()
        t_names = names + [cname]
        t_dirs = dirs + [g0]
        order = [len(t_names) - 1] + list(range(len(t_names) - 1))
        t_dirs, tp, tst, _ = _alternate(ctx, t_names, t_dirs, opts, 0.15, seed + 9000 * k, order)
        gain = cur_nmse - tst["nmse"]
        if gain >= opts.min_gain and tp[len(t_names) - 1] > 0.0:
            names, dirs, cur_nmse = t_names, t_dirs, tst["nmse"]
        else:
            break

    # ---- 3. 최종 풀이 · 결과 조립 -----------------------------------------------------------------------------
    Ts = [ctx.T(nm, d) for nm, d in zip(names, dirs)]
    p, st = ctx.fit(Ts)
    keep = [0] + [i for i in range(1, len(names)) if p[i] > 0.0]
    if len(keep) != len(names):
        names = [names[i] for i in keep]
        dirs = [dirs[i] for i in keep]
        p, st = ctx.fit([Ts[i] for i in keep])
    J = len(names)
    amb = float(p[J])
    lights = []
    for i in range(J):
        d = tuple(float(x) for x in dirs[i])
        if i == 0:
            kind, rad = "sun", 0.0
            if theta_s is not None:
                kind, rad = classify_source(theta_s), min(theta_s / 2.0, math.pi / 2)
            lights.append(Light(name=names[i], role="motivated", kind=kind, direction=d, intensity=float(p[i]),
                                angular_radius=rad))
        else:
            lights.append(Light(name=names[i], role="cheat", kind="area", direction=d, intensity=float(p[i]),
                                angular_radius=CHEAT_ANGULAR_RADIUS))
    r2, rms, nmse = st["r2"], st["rms"], st["nmse"]
    if r2 < opts.r2_min or lights[0].intensity <= 0.0:
        return _fallback("low_r2", n_used, ctx.n_evals, fs, priors, opts, r2=r2, rms=rms, nmse=nmse)

    vis_maps = {}
    if visibility is not None:
        for light in lights:
            vis_maps[light.name] = vis_fn(np.asarray(light.direction))
    lights = tuple(lights)
    wfull = np.where(used, w, 0.0)
    rep = energy_report(lights, n_eff, ambient=amb, weight=wfull, visibility=vis_maps, flags=fs)
    geo = render_shading(lights, n_eff, ambient=amb, visibility=vis_maps, flags=fs).astype(np.float32)

    warns = set()
    if rep.cheat_share > CHEAT_SHARE_MAX:
        warns.add("cheat_share_exceeded")
    if rep.n_cheat == opts.max_cheat and nmse >= opts.resid_tau:
        warns.add("cheat_limit")
    have = {light.name for light in lights}
    if any(f.light not in have for f in fs):
        warns.add("flag_unused")
    return FitLightsResult(status="ok", lights=lights, ambient=amb, flags=fs, geo_shading=geo, r2=r2, rms=rms,
                           nmse=nmse, cheat_share=rep.cheat_share, r_light=rep.r_light, red=rep.red, n_used=n_used,
                           n_evals=ctx.n_evals, warnings=tuple(sorted(warns)), priors=priors, energies=rep.energies,
                           opts=opts)
