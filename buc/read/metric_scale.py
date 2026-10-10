"""RD-08 R6 미터 스케일: 크기 사전(로그정규) 가중 최소제곱 + 앵커 + 저술된 스케일.

MRH 깊이는 affine-불변이라 미터가 없다. 검출 객체의 상대 크기 ``ĥ_k``(미터 = s·ĥ)와 크기 사전(DAT-16)으로
전역 스케일 ``s`` 와 불확실도 ``σ_s`` 를 푼다. 한 객체에 몰린 크기 잔차는 "저술된 스케일"로 분리해 풀이에서 뺀다.
증거가 없으면 모른다고 말한다(``reliable=False``).

numpy · 표준 라이브러리만(torch·cv2·yaml 금지; 표 적재는 ``default_table()`` 호출 때만). import 부작용 없음,
전역 RNG 불변, 순수·결정론·입력 불변.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SIGMA_LOG_F", "MetricScaleError", "SizeObs", "Anchor", "ScaleOpts", "ObsFit", "AuthoredScale", "MetricScale",
    "default_table", "f_px_from_fov", "rel_size", "make_obs", "anchor_subject_distance", "anchor_camera_height",
    "estimate_metric_scale",
]

SIGMA_LOG_F = {"vp3": 0.03, "vp2": 0.05, "prior": 0.10, "default": 0.30}


class MetricScaleError(ValueError):
    pass


# ------------------------------------------------------------------ 검증 도우미
def _num(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: 숫자여야 한다: {x!r}")
    return float(x)


def _finite(x, name):
    v = _num(x, name)
    if not math.isfinite(v):
        raise ValueError(f"{name}: 유한해야 한다: {x!r}")
    return v


def _pos(x, name):
    v = _finite(x, name)
    if not v > 0:
        raise ValueError(f"{name}: 양수여야 한다: {x!r}")
    return v


# ------------------------------------------------------------------ 입력 자료형
@dataclass(frozen=True, kw_only=True)
class SizeObs:
    cls: str
    h_rel: float
    sigma_h: float = 0.1
    q: float = 1.0
    saliency: float = 0.0
    id: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.cls, str):
            raise TypeError(f"cls: 문자열이어야 한다: {self.cls!r}")
        if not self.cls.strip():
            raise ValueError("cls: 비어 있다")
        _pos(self.h_rel, "h_rel")
        sh = _finite(self.sigma_h, "sigma_h")
        if sh < 0:
            raise ValueError(f"sigma_h ≥ 0 이어야 한다: {self.sigma_h!r}")
        q = _finite(self.q, "q")
        if not 0 < q <= 1:
            raise ValueError(f"q ∈ (0,1] 이어야 한다: {self.q!r}")
        sal = _finite(self.saliency, "saliency")
        if not 0 <= sal <= 1:
            raise ValueError(f"saliency ∈ [0,1] 이어야 한다: {self.saliency!r}")


@dataclass(frozen=True, kw_only=True)
class Anchor:
    name: str
    log_s: float
    sigma: float

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"name: 문자열이어야 한다: {self.name!r}")
        if not self.name:
            raise ValueError("name: 비어 있다")
        _finite(self.log_s, "log_s")
        _pos(self.sigma, "sigma")


@dataclass(frozen=True, kw_only=True)
class ScaleOpts:
    z_author: float = 3.0
    z_inlier: float = 2.5
    min_obs_author: int = 3
    sigma_reliable: float = 0.35
    chi2_warn: float = 4.0
    default_log_s: float = 0.0
    default_sigma: float = 1.5

    def __post_init__(self):
        za = _finite(self.z_author, "z_author")
        zi = _finite(self.z_inlier, "z_inlier")
        if not (za > zi > 0):
            raise ValueError("z_author > z_inlier > 0 이어야 한다")
        if isinstance(self.min_obs_author, bool) or not isinstance(self.min_obs_author, (int, np.integer)):
            raise TypeError("min_obs_author: 정수여야 한다")
        if self.min_obs_author < 3:
            raise ValueError("min_obs_author ≥ 3 이어야 한다")
        _pos(self.sigma_reliable, "sigma_reliable")
        _pos(self.chi2_warn, "chi2_warn")
        _finite(self.default_log_s, "default_log_s")
        _pos(self.default_sigma, "default_sigma")


# ------------------------------------------------------------------ 결과 자료형
@dataclass(frozen=True)
class ObsFit:
    index: int
    id: Optional[str]
    cls: str
    name: Optional[str]
    dim: Optional[str]
    status: str
    log_s_k: Optional[float]
    sigma_k: Optional[float]
    weight: Optional[float]
    z: Optional[float]
    log_size_resid: Optional[float]


@dataclass(frozen=True)
class AuthoredScale:
    index: int
    id: Optional[str]
    cls: str
    factor: float


@dataclass(frozen=True)
class MetricScale:
    s: float
    log_s: float
    sigma_s: float
    sigma_stat: float
    source: str
    n_used: int
    n_unknown: int
    fits: Tuple[ObsFit, ...]
    authored: Tuple[AuthoredScale, ...]
    anchors: Tuple[str, ...]
    chi2_red: Optional[float]
    reliable: bool
    warnings: Tuple[str, ...]

    def to_meters(self, x):
        if isinstance(x, np.ndarray):
            return x * self.s
        return self.s * x

    def interval(self, k: float = 2.0):
        return (self.s * math.exp(-k * self.sigma_s), self.s * math.exp(k * self.sigma_s))

    def to_meta(self) -> dict:
        return {
            "s": self.s, "log_s": self.log_s, "sigma_s": self.sigma_s, "sigma_stat": self.sigma_stat,
            "source": self.source, "n_used": self.n_used, "n_unknown": self.n_unknown,
            "authored": [{"id": a.id, "cls": a.cls, "factor": a.factor} for a in self.authored],
            "anchors": list(self.anchors), "chi2_red": self.chi2_red, "reliable": self.reliable,
            "warnings": list(self.warnings),
        }


# ------------------------------------------------------------------ 표
_TABLE = None


def default_table():
    """data/size_prior.yaml 을 DAT-16 도구로 적재(캐시)."""
    global _TABLE
    if _TABLE is None:
        tool = Path(__file__).resolve().parents[2] / "tools" / "size_prior_check.py"
        name = "size_prior_check_rd08_default"
        mod = sys.modules.get(name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(name, tool)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        _TABLE = mod.load_table()
    return _TABLE


# ------------------------------------------------------------------ 헬퍼
def f_px_from_fov(fov_x_deg, width_px) -> float:
    fov = _finite(fov_x_deg, "fov_x_deg")
    w = _pos(width_px, "width_px")
    if not 0 < fov < 180:
        raise ValueError(f"fov_x_deg ∈ (0,180) 이어야 한다: {fov_x_deg!r}")
    return (w / 2.0) / math.tan(math.radians(fov) / 2.0)


def rel_size(extent_px, depth_rel, f_px) -> float:
    return _pos(extent_px, "extent_px") * _pos(depth_rel, "depth_rel") / _pos(f_px, "f_px")


def make_obs(cls, *, extent_px, depth_rel, f_px, sigma_extent=0.05, sigma_depth=0.05, q=1.0, saliency=0.0,
             id=None) -> SizeObs:
    h = rel_size(extent_px, depth_rel, f_px)
    se = _finite(sigma_extent, "sigma_extent")
    sd = _finite(sigma_depth, "sigma_depth")
    if se < 0 or sd < 0:
        raise ValueError("sigma_extent, sigma_depth ≥ 0 이어야 한다")
    return SizeObs(cls=cls, h_rel=h, sigma_h=math.hypot(se, sd), q=q, saliency=saliency, id=id)


def anchor_subject_distance(distance_m, depth_rel, *, sigma_log=0.10) -> Anchor:
    d = _pos(distance_m, "distance_m")
    r = _pos(depth_rel, "depth_rel")
    return Anchor(name="exif_subject_distance", log_s=math.log(d / r), sigma=_pos(sigma_log, "sigma_log"))


def anchor_camera_height(height_rel, *, median_m=1.5, sigma_log=0.30) -> Anchor:
    h = _pos(height_rel, "height_rel")
    m = _pos(median_m, "median_m")
    return Anchor(name="camera_height_prior", log_s=math.log(m / h), sigma=_pos(sigma_log, "sigma_log"))


# ------------------------------------------------------------------ 추정
def _wsum(ws, xs):
    return math.fsum(w * x for w, x in zip(ws, xs))


def estimate_metric_scale(obs, *, table=None, anchors=(), sigma_log_f=0.0, opts=None) -> MetricScale:
    if isinstance(obs, (str, bytes)) or not hasattr(obs, "__iter__"):
        raise TypeError("obs: SizeObs 시퀀스여야 한다")
    obs = tuple(obs)
    if not all(isinstance(o, SizeObs) for o in obs):
        raise TypeError("obs: 모든 원소가 SizeObs 여야 한다")
    if isinstance(anchors, (str, bytes)) or not hasattr(anchors, "__iter__"):
        raise TypeError("anchors: Anchor 시퀀스여야 한다")
    anchors = tuple(anchors)
    if not all(isinstance(a, Anchor) for a in anchors):
        raise TypeError("anchors: 모든 원소가 Anchor 여야 한다")
    if opts is None:
        opts = ScaleOpts()
    elif not isinstance(opts, ScaleOpts):
        raise TypeError("opts: ScaleOpts 여야 한다")
    slf = _finite(sigma_log_f, "sigma_log_f")
    if slf < 0:
        raise ValueError("sigma_log_f ≥ 0 이어야 한다")
    if table is None:
        table = default_table()

    # 객체별 단독 증언
    rec = []  # (index, o, prior|None, a, v, w)
    for i, o in enumerate(obs):
        try:
            p = table.resolve(o.cls)
        except KeyError:
            rec.append((i, o, None, None, None, None))
            continue
        mu = float(p.mu_log)
        sc = float(p.sigma_log)
        v = sc * sc + o.sigma_h * o.sigma_h
        rec.append((i, o, p, mu - math.log(o.h_rel), v, o.q / v))
    used = [r for r in rec if r[2] is not None]
    n_unknown = len(rec) - len(used)

    anc_a = [a.log_s for a in anchors]
    anc_v = [a.sigma * a.sigma for a in anchors]
    anc_w = [1.0 / v for v in anc_v]

    # 저술된 스케일 판정
    authored_idx = None
    authored_factor = None
    if len(used) >= opts.min_obs_author:
        sal = [r[1].saliency for r in used]
        mx = max(sal)
        if mx > 0 and sal.count(mx) == 1:
            kstar = used[sal.index(mx)]
            rest = [r for r in used if r is not kstar]
            ws = [r[5] for r in rest] + anc_w
            aa = [r[3] for r in rest] + anc_a
            sw = math.fsum(ws)
            if sw > 0:
                ls_m = _wsum(ws, aa) / sw
                var_m = 1.0 / sw
                resid = kstar[3] - ls_m
                if abs(resid) / math.sqrt(kstar[4] + var_m) > opts.z_author and all(
                        abs(r[3] - ls_m) / math.sqrt(r[4]) <= opts.z_inlier for r in rest):
                    authored_idx = kstar[0]
                    authored_factor = math.exp(
                        ls_m + math.log(kstar[1].h_rel) - float(kstar[2].mu_log))

    pool = [r for r in used if r[0] != authored_idx]
    p_w = [r[5] for r in pool] + anc_w
    p_a = [r[3] for r in pool] + anc_a
    p_v = [r[4] for r in pool] + anc_v

    warnings = set()
    if n_unknown:
        warnings.add("unknown_class")
    if authored_idx is not None:
        warnings.add("authored_scale")

    chi2 = None
    if p_w:
        sw = math.fsum(p_w)
        log_s = _wsum(p_w, p_a) / sw
        sigma_stat = sw ** -0.5
        if len(p_w) > 1:
            chi2 = math.fsum((a - log_s) ** 2 / v for a, v in zip(p_a, p_v)) / (len(p_w) - 1)
        if pool and anchors:
            source = "size_prior+anchor"
        elif pool:
            source = "size_prior"
        else:
            source = "anchor"
    else:
        log_s = opts.default_log_s
        sigma_stat = opts.default_sigma
        source = "default"
        warnings.add("no_scale_evidence")
    sigma_s = math.hypot(sigma_stat, slf)

    if chi2 is not None and chi2 > opts.chi2_warn:
        warnings.add("scale_inconsistent")
    if sigma_s > opts.sigma_reliable:
        warnings.add("scale_ambiguous")
    reliable = (source != "default" and sigma_s <= opts.sigma_reliable and "scale_inconsistent" not in warnings)

    fits = []
    for i, o, p, a, v, w in rec:
        if p is None:
            fits.append(ObsFit(i, o.id, o.cls, None, None, "unknown_class", None, None, None, None, None))
            continue
        resid = log_s + math.log(o.h_rel) - float(p.mu_log)
        if i == authored_idx:
            status, z = "authored", None
        else:
            status, z = "used", (a - log_s) / math.sqrt(v)
        fits.append(ObsFit(i, o.id, o.cls, p.name, p.dim, status, a, math.sqrt(v), w, z, resid))

    authored = ()
    if authored_idx is not None:
        o = obs[authored_idx]
        authored = (AuthoredScale(authored_idx, o.id, o.cls, authored_factor),)

    return MetricScale(
        s=math.exp(log_s), log_s=log_s, sigma_s=sigma_s, sigma_stat=sigma_stat, source=source,
        n_used=len(pool), n_unknown=n_unknown, fits=tuple(fits), authored=authored,
        anchors=tuple(a.name for a in anchors), chi2_red=chi2, reliable=reliable,
        warnings=tuple(sorted(warnings)),
    )
