"""RD-11 에셔 점수: 7성분 + r_dis + r_refl 의 파라미터 0 잔차 합 (파이프라인 §2.9.2).

``Esc = Σ_c w_c · min(1, r_c / τ_c) ∈ [0, 1]``. 학습·적합 없음. 상류 판독(RD-02/07/08/10)이 낸 잔차를 표의 τ·w 로 묶을 뿐이며,
메뉴 등급(안전/확장/풀)의 근거가 된다. 잔차를 못 재면 0 으로 두고 "측정 못 함"으로 표시한다(Esc 는 하한).

numpy 와 표준 라이브러리만 쓴다(import 시 torch·scipy·yaml 미로드). 텐서는 ``.detach().cpu().numpy()`` 덕 타이핑으로 받는다.
순수·결정론, 입력 불변, 전역 RNG 불변. 상류 모듈은 import 하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = [
    "EscherError", "EscherInputs", "EscComponent", "EscherScore", "MenuHint", "SizeResidual", "PlaneResidual",
    "escher_score", "esc_band", "menu_hint", "size_residual", "plane_residual", "circular_sd_deg",
    "light_residual_deg", "px_at_1080p", "rms_px1080", "light_cue_dirs", "inputs_from_reads", "escher_from_reads",
    "COMPONENTS", "WEIGHTS", "TAU", "GREEN_MAX", "RED_MIN", "CARDS_MIN", "CHEAT_SHARE_MAX", "PX_REF_HEIGHT",
]

COMPONENTS = ("vp", "multi_vp", "size", "plane", "light", "topo", "reconcile", "r_dis", "r_refl")
WEIGHTS = {"vp": 0.20, "multi_vp": 0.15, "size": 0.15, "plane": 0.05, "light": 0.15, "topo": 0.10,
           "reconcile": 0.05, "r_dis": 0.10, "r_refl": 0.05}
TAU = {"vp": 2.0, "multi_vp": 0.10, "size": 0.25, "plane": 0.05, "light": 15.0, "topo": 2.0,
       "reconcile": 4.0, "r_dis": 0.05, "r_refl": 4.0}
_UNIT = {"vp": "deg", "multi_vp": "frac", "size": "log", "plane": "rel", "light": "deg", "topo": "count",
         "reconcile": "px@1080p", "r_dis": "rel", "r_refl": "px@1080p"}
GREEN_MAX = 0.15
RED_MIN = 0.35
CARDS_MIN = 0.6
CHEAT_SHARE_MAX = 0.35
PX_REF_HEIGHT = 1080
_MAD_TO_SD = 1.4826
_ESC_DECIMALS = 12      # 부동소수 합의 꼬리를 정리한다(띠 경계 0.15/0.35/0.6 에서 반올림 잡음이 등급을 흔들지 않게)


class EscherError(ValueError):
    """RD-11 입력 오류."""


# ------------------------------------------------------------------------------------------ 검증 도우미
def _num(v: Any, name: str) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: 수치여야 한다 ({type(v).__name__})")
    return float(v)


def _nonneg(v: Any, name: str, upper: float | None = None) -> float:
    x = _num(v, name)
    if not math.isfinite(x) or x < 0.0:
        raise EscherError(f"{name}: 유한한 비음수여야 한다 ({x})")
    if upper is not None and x > upper:
        raise EscherError(f"{name}: {upper} 이하여야 한다 ({x})")
    return x


def _to_np(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


# ------------------------------------------------------------------------------------------ 입력
_FIELDS = ("r_vp_deg", "orphan_frac", "size_sd", "plane_rel", "light_sd_deg", "cheat_share", "topo_count",
           "ffg_px1080", "r_dis", "r_refl_px1080")
_UPPER = {"orphan_frac": 1.0, "cheat_share": 1.0}


@dataclass(frozen=True, kw_only=True)
class EscherInputs:
    r_vp_deg: float | None = None
    orphan_frac: float | None = None
    size_sd: float | None = None
    plane_rel: float | None = None
    light_sd_deg: float | None = None
    cheat_share: float | None = None
    topo_count: float | None = None
    ffg_px1080: float | None = None
    r_dis: float | None = None
    r_refl_px1080: float | None = None

    def __post_init__(self):
        for f in _FIELDS:
            v = getattr(self, f)
            if v is not None:
                object.__setattr__(self, f, _nonneg(v, f, _UPPER.get(f)))


@dataclass(frozen=True)
class EscComponent:
    name: str
    r: float | None
    tau: float
    w: float
    w_eff: float
    x: float
    contrib: float
    measured: bool
    unit: str


@dataclass(frozen=True)
class EscherScore:
    esc: float
    band: str
    components: tuple
    pending: tuple
    evidence: float
    refl_redirected: bool
    dominant: str | None
    cards_2_5d: bool
    inputs: EscherInputs
    tau_scale: float

    def component(self, name: str) -> EscComponent:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    def with_reconcile(self, ffg_px1080: float) -> "EscherScore":
        """화해 후 비용(RD-12)을 넣어 다시 계산한다. 원본은 그대로."""
        return escher_score(replace(self.inputs, ffg_px1080=ffg_px1080), tau_scale=self.tau_scale)

    def to_meta(self) -> dict:
        return {
            "esc": self.esc, "band": self.band,
            "components": {c.name: {"r": c.r, "tau": c.tau, "w": c.w, "w_eff": c.w_eff, "x": c.x,
                                    "contrib": c.contrib, "measured": c.measured, "unit": c.unit}
                           for c in self.components},
            "pending": list(self.pending), "evidence": self.evidence, "dominant": self.dominant,
            "cards_2_5d": self.cards_2_5d, "refl_redirected": self.refl_redirected, "tau_scale": self.tau_scale,
        }


@dataclass(frozen=True)
class MenuHint:
    band: str
    safe: bool
    ext: bool
    full: bool
    full_warn: bool
    cards_2_5d: bool


# ------------------------------------------------------------------------------------------ 점수
def _light_r(sd: float | None, share: float | None) -> float | None:
    if sd is None and share is None:
        return None
    return light_residual_deg(0.0 if sd is None else sd, 0.0 if share is None else share)


def escher_score(inputs: EscherInputs, *, tau_scale: float = 1.0) -> EscherScore:
    if not isinstance(inputs, EscherInputs):
        raise TypeError("inputs 는 EscherInputs 여야 한다")
    scale = _num(tau_scale, "tau_scale")
    if not math.isfinite(scale) or scale <= 0.0:
        raise EscherError(f"tau_scale 은 유한한 양수여야 한다 ({scale})")
    r = {"vp": inputs.r_vp_deg, "multi_vp": inputs.orphan_frac, "size": inputs.size_sd, "plane": inputs.plane_rel,
         "light": _light_r(inputs.light_sd_deg, inputs.cheat_share), "topo": inputs.topo_count,
         "reconcile": inputs.ffg_px1080, "r_dis": inputs.r_dis, "r_refl": inputs.r_refl_px1080}
    redirected = r["r_refl"] is None
    comps = []
    for name in COMPONENTS:
        rc = r[name]
        tau = TAU[name] * scale
        w = WEIGHTS[name]
        if rc is None:
            w_eff, x, contrib = 0.0, 0.0, 0.0
        else:
            w_eff = w + (WEIGHTS["r_refl"] if (name == "vp" and redirected) else 0.0)
            x = min(1.0, rc / tau)
            contrib = w_eff * x
        comps.append(EscComponent(name=name, r=rc, tau=tau, w=w, w_eff=w_eff, x=x, contrib=contrib,
                                  measured=rc is not None, unit=_UNIT[name]))
    esc = round(min(1.0, max(0.0, math.fsum(c.contrib for c in comps))), _ESC_DECIMALS)
    evidence = math.fsum(c.w_eff for c in comps if c.measured)
    dominant = None
    if esc > 0.0:
        best = -1.0
        for c in comps:
            if c.contrib > best:
                best, dominant = c.contrib, c.name
    return EscherScore(esc=esc, band=esc_band(esc), components=tuple(comps),
                       pending=tuple(c.name for c in comps if not c.measured), evidence=evidence,
                       refl_redirected=redirected, dominant=dominant, cards_2_5d=esc > CARDS_MIN,
                       inputs=inputs, tau_scale=scale)


def _check_esc(esc: Any) -> float:
    e = _num(esc, "esc")
    if not math.isfinite(e) or e < 0.0 or e > 1.0:
        raise EscherError(f"esc 는 [0,1] 이어야 한다 ({e})")
    return e


def esc_band(esc: float) -> str:
    e = _check_esc(esc)
    if e < GREEN_MAX:
        return "green"
    if e > RED_MIN:
        return "red"
    return "yellow"


def menu_hint(esc: float) -> MenuHint:
    e = _check_esc(esc)
    band = esc_band(e)
    ext = e <= RED_MIN
    return MenuHint(band=band, safe=True, ext=ext, full=ext, full_warn=(e >= GREEN_MAX and ext),
                    cards_2_5d=e > CARDS_MIN)


# ------------------------------------------------------------------------------------------ 크기
@dataclass(frozen=True)
class SizeResidual:
    sd: float | None
    per_class: dict
    n_groups: int
    n_objects: int


def size_residual(fits, *, min_group: int = 2) -> SizeResidual:
    if isinstance(min_group, bool) or not isinstance(min_group, (int, np.integer)):
        raise TypeError("min_group 은 정수여야 한다")
    if min_group < 2:
        raise EscherError("min_group 은 2 이상이어야 한다")
    groups: dict = {}
    for f in iter(fits):
        if getattr(f, "status", None) != "used":
            continue
        resid = getattr(f, "log_size_resid", None)
        if resid is None:
            continue
        resid = float(resid)
        if not math.isfinite(resid):
            continue
        key = getattr(f, "name", None) or getattr(f, "cls", None)
        if key is None:
            continue
        groups.setdefault(key, []).append(resid)
    per_class, n_obj = {}, 0
    for key, vals in groups.items():
        if len(vals) >= min_group:
            v = np.asarray(vals, np.float64)
            per_class[key] = _MAD_TO_SD * float(np.median(np.abs(v - np.median(v))))
            n_obj += len(vals)
    sd = max(per_class.values()) if per_class else None
    return SizeResidual(sd=sd, per_class=per_class, n_groups=len(per_class), n_objects=n_obj)


# ------------------------------------------------------------------------------------------ 평면
@dataclass(frozen=True)
class PlaneResidual:
    rel: float | None
    regions: dict
    n_regions: int
    n_pixels: int
    rel_map: np.ndarray


def plane_residual(log_depth, plane_labels, *, min_pixels: int = 12) -> PlaneResidual:
    labels = _to_np(plane_labels)
    if labels.dtype.kind not in "iu":
        raise TypeError("plane_labels 는 정수 배열이어야 한다")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, (int, np.integer)):
        raise TypeError("min_pixels 는 정수여야 한다")
    if min_pixels < 3:
        raise EscherError("min_pixels 는 3 이상이어야 한다")
    ld = _to_np(log_depth)
    if ld.ndim == 4:
        if ld.shape[:2] != (1, 1):
            raise EscherError(f"log_depth 모양이 (H,W) 또는 (1,1,H,W) 가 아니다: {ld.shape}")
        ld = ld[0, 0]
    if ld.ndim != 2:
        raise EscherError(f"log_depth 모양이 (H,W) 또는 (1,1,H,W) 가 아니다: {ld.shape}")
    if ld.dtype.kind not in "fiu":
        raise TypeError("log_depth 는 수치 배열이어야 한다")
    ld = ld.astype(np.float64)
    if labels.shape != ld.shape:
        raise EscherError(f"모양 불일치: log_depth {ld.shape} vs plane_labels {labels.shape}")
    h, w = ld.shape
    rel_map = np.full((h, w), np.nan, np.float64)
    usable = (labels >= 0) & np.isfinite(ld)
    if not usable.any():
        return PlaneResidual(rel=None, regions={}, n_regions=0, n_pixels=0, rel_map=rel_map)
    ref = float(ld[usable].max())
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    regions: dict = {}
    n_pix = 0
    for lab in np.unique(labels[usable]):
        m = usable & (labels == lab)
        n = int(m.sum())
        if n < min_pixels:
            continue
        inv = np.exp(ref - ld[m])                       # w = 1/z (스케일은 상대 오차에 무관)
        a = np.stack([cols[m], rows[m], np.ones(n)], 1)
        coef, *_ = np.linalg.lstsq(a, inv, rcond=None)
        fit = a @ coef
        rel = np.abs(inv - fit) / np.maximum(np.abs(fit), 1e-3 * float(np.median(inv)))
        rel_map[m] = rel
        regions[int(lab)] = float(np.median(rel))
        n_pix += n
    if not regions:
        return PlaneResidual(rel=None, regions={}, n_regions=0, n_pixels=0, rel_map=rel_map)
    rel_all = float(np.median(rel_map[np.isfinite(rel_map)]))
    return PlaneResidual(rel=rel_all, regions=regions, n_regions=len(regions), n_pixels=n_pix, rel_map=rel_map)


# ------------------------------------------------------------------------------------------ 빛
def circular_sd_deg(dirs, weights=None) -> float:
    d = _to_np(dirs).astype(np.float64) if not isinstance(dirs, np.ndarray) else dirs.astype(np.float64)
    if d.size == 0:
        return 0.0
    if d.ndim != 2 or d.shape[1] != 3:
        raise EscherError(f"dirs 모양이 (N,3) 이 아니다: {d.shape}")
    n = d.shape[0]
    if not np.isfinite(d).all():
        raise EscherError("dirs 에 비유한 값이 있다")
    norm = np.linalg.norm(d, axis=1)
    if (norm <= 0.0).any():
        raise EscherError("dirs 에 영벡터가 있다")
    if weights is None:
        wt = np.ones(n)
    else:
        wt = _to_np(weights).astype(np.float64).reshape(-1)
        if wt.shape[0] != n:
            raise EscherError("weights 길이가 dirs 와 다르다")
        if not np.isfinite(wt).all() or (wt < 0).any():
            raise EscherError("weights 는 유한한 비음수여야 한다")
        if wt.sum() <= 0.0:
            raise EscherError("weights 합이 0 이다")
    if n < 2:
        return 0.0
    u = d / norm[:, None]
    rr = float(np.linalg.norm((wt[:, None] * u).sum(0)) / wt.sum())
    if rr < 1e-12:
        return 180.0
    rr = min(rr, 1.0)
    return min(180.0, math.degrees(math.sqrt(max(0.0, -2.0 * math.log(rr)))))


def light_residual_deg(sd_deg: float, cheat_share: float) -> float:
    sd = _nonneg(sd_deg, "sd_deg")
    share = _nonneg(cheat_share, "cheat_share", 1.0)
    return sd + 15.0 * max(0.0, share - CHEAT_SHARE_MAX) / CHEAT_SHARE_MAX


# ------------------------------------------------------------------------------------------ px@1080p
def _height(h: Any) -> float:
    x = _num(h, "image_height_px")
    if not math.isfinite(x) or x <= 0.0:
        raise EscherError(f"image_height_px 는 양수여야 한다 ({x})")
    return x


def px_at_1080p(px: float, image_height_px: float) -> float:
    p = _nonneg(px, "px")
    return p * PX_REF_HEIGHT / _height(image_height_px)


def rms_px1080(displacements_px, image_height_px: float) -> float:
    hh = _height(image_height_px)
    d = _to_np(displacements_px).astype(np.float64)
    if d.size == 0:
        raise EscherError("변위가 비었다")
    if d.ndim == 1:
        sq = d * d
    elif d.ndim == 2 and d.shape[1] == 2:
        sq = (d * d).sum(1)
    else:
        raise EscherError(f"변위 모양이 (N,) 또는 (N,2) 가 아니다: {d.shape}")
    if not np.isfinite(d).all():
        raise EscherError("변위에 비유한 값이 있다")
    return math.sqrt(float(sq.mean())) * PX_REF_HEIGHT / hh


# ------------------------------------------------------------------------------------------ 상류 어댑터
def _unit(v) -> np.ndarray:
    a = np.asarray(v, np.float64).reshape(-1)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or not math.isfinite(n) or n <= 0.0:
        raise EscherError("광원 방향이 올바른 3-벡터가 아니다")
    return a / n


def light_cue_dirs(lights) -> list:
    """RD-10 결과에서 방향 단서를 뽑는다: (ok 일 때) 첫 동기 광원 + 그림자 사전분포."""
    cues: list = []
    if getattr(lights, "status", None) == "ok":
        for lt in getattr(lights, "lights", ()) or ():
            if getattr(lt, "role", None) == "motivated":
                cues.append(_unit(lt.direction))
                break
    for p in getattr(lights, "priors", ()) or ():
        if getattr(p, "source", None) == "shadow":
            cues.append(_unit(p.direction))
    return cues


def _finite_or_none(v):
    if v is None:
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def inputs_from_reads(*, geom=None, mrh=None, scale=None, lights=None, light_dirs=(), plane_labels=None,
                      topo_count=None, ffg_px1080=None, refl_px1080=None) -> EscherInputs:
    r_vp = orphan = r_dis = size_sd = plane_rel = light_sd = cheat = None
    if geom is not None:
        r_vp = _finite_or_none(geom.r_vp_deg)
        orphan = _finite_or_none(geom.orphan_frac)
    if mrh is not None:
        r_dis = _finite_or_none(mrh.r_dis)
    if plane_labels is not None:
        outputs = getattr(mrh, "outputs", None) if mrh is not None else None
        if outputs is None or "log_depth" not in outputs or outputs["log_depth"] is None:
            raise EscherError("plane_labels 에는 mrh.outputs['log_depth'] 가 필요하다")
        plane_rel = plane_residual(outputs["log_depth"], plane_labels).rel
    if scale is not None:
        size_sd = size_residual(scale.fits).sd
    extra = [_unit(v) for v in (light_dirs if light_dirs is not None else ())]
    if lights is not None:
        cheat = _finite_or_none(lights.cheat_share)
        cues = light_cue_dirs(lights) + extra
        light_sd = circular_sd_deg(np.asarray(cues, np.float64).reshape(-1, 3))
    elif len(extra) >= 2:
        light_sd = circular_sd_deg(np.asarray(extra, np.float64))
    return EscherInputs(r_vp_deg=r_vp, orphan_frac=orphan, size_sd=size_sd, plane_rel=plane_rel,
                        light_sd_deg=light_sd, cheat_share=cheat, topo_count=topo_count, ffg_px1080=ffg_px1080,
                        r_dis=r_dis, r_refl_px1080=refl_px1080)


def escher_from_reads(*, tau_scale: float = 1.0, **reads) -> EscherScore:
    return escher_score(inputs_from_reads(**reads), tau_scale=tau_scale)
