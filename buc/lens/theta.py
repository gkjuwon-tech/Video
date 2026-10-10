"""LNS-01: theta(룩 프리셋) 스키마 어휘·물리 범위, 결정론 사전분포 샘플러, 수제 프리셋 로더.

학습 파라미터 0, 순수 수식·데이터. torch 를 import 하지 않는다.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml

from buc.core.contracts import ContractError, Theta

SCHEMA_VERSION = "1.0.0"

LUT_NAMES = ("none", "neutral_709", "teal_orange", "bleach_bypass", "kodak_warm", "fuji_cool", "mono_silver",
             "faded_matte", "neon_night", "cross_process")
LENS_RX_NAMES = ("spherical_clean", "vintage_petzval", "vintage_double_gauss", "anamorphic_2x", "soft_single_coat")
EXTRA_DEFAULTS = {"squeeze": 1.0, "blades": 9, "cpl_angle_deg": 0.0, "rolloff": 0.5, "atmos_g": 0.0}

HARD_BOUNDS = {
    "aspect": (1.0, 3.0), "focal_mm": (4.0, 600.0), "sensor_width_mm": (4.8, 70.0), "f_stop": (0.95, 32.0),
    "focus_m": (0.3, 1000.0), "shutter_deg": (1.0, 360.0), "fps": (8.0, 240.0), "nd_stops": (0.0, 12.0),
    "cpl": (0.0, 1.0), "ev": (-5.0, 5.0), "wb_kelvin": (2000.0, 12000.0), "wb_tint": (-0.6, 0.6),
    "grain": (0.0, 1.0), "halation": (0.0, 1.0), "vignette": (0.0, 1.0), "rim": (0.0, 1.0),
    "atmos_beta": (0.0, 0.1), "sun_elev_deg": (-20.0, 90.0), "sun_azim_deg": (0.0, 360.0),
    "blades": (3, 18), "cpl_angle_deg": (0.0, 180.0), "rolloff": (0.0, 1.0), "atmos_g": (0.0, 0.95),
}
_EXCL_HI = frozenset({"sun_azim_deg", "cpl_angle_deg"})  # [lo, hi)
_FLOAT_EXTRA = ("squeeze", "cpl_angle_deg", "rolloff", "atmos_g")

PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "presets"

_THETA_FIELDS = tuple(f.name for f in fields(Theta))
_NUM_FIELDS = tuple(n for n in _THETA_FIELDS
                    if n not in ("lut", "lens_rx", "extra", "sun_elev_deg", "sun_azim_deg"))


def _err(msg: str) -> ContractError:
    return ContractError("theta: " + msg)


def _is_real(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


# ------------------------------------------------------------------------------------------------ extra
def _canon_extra(extra: Mapping) -> dict:
    out = copy.deepcopy(dict(extra))
    for k in _FLOAT_EXTRA:
        if k in out:
            v = out[k]
            if not _is_real(v):
                raise _err(f"extra.{k}: 숫자여야 한다 ({v!r})")
            out[k] = float(v)
        else:
            out[k] = EXTRA_DEFAULTS[k]
    if "blades" in out:
        v = out["blades"]
        if not _is_real(v) or float(v) != math.floor(float(v)) or not math.isfinite(float(v)):
            raise _err(f"extra.blades: 정수여야 한다 ({v!r})")
        out["blades"] = int(v)
    else:
        out["blades"] = EXTRA_DEFAULTS["blades"]
    return out


def default_theta() -> Theta:
    return Theta(extra=dict(EXTRA_DEFAULTS))


def complete_extra(theta: Theta) -> Theta:
    kw = {f.name: getattr(theta, f.name) for f in fields(Theta)}
    extra = kw["extra"]
    if not isinstance(extra, Mapping):
        raise _err("extra 는 dict")
    kw["extra"] = _canon_extra(extra)
    return Theta(**kw)


# ------------------------------------------------------------------------------------------------ validate
def _in_bounds(name: str, v: float) -> bool:
    lo, hi = HARD_BOUNDS[name]
    if not math.isfinite(v):
        return False
    return lo <= v < hi if name in _EXCL_HI else lo <= v <= hi


def validate_theta(theta: Theta) -> None:
    if not isinstance(theta, Theta):
        raise _err("Theta 가 아니다")
    theta.validate()
    for n in _NUM_FIELDS:
        v = getattr(theta, n)
        if not _in_bounds(n, float(v)):
            raise _err(f"{n}={v!r} 가 범위 {HARD_BOUNDS[n]} 밖")
    for n in ("sun_elev_deg", "sun_azim_deg"):
        v = getattr(theta, n)
        if v is not None and not _in_bounds(n, float(v)):
            raise _err(f"{n}={v!r} 가 범위 {HARD_BOUNDS[n]} 밖")
    if (theta.sun_elev_deg is None) != (theta.sun_azim_deg is None):
        raise _err("sun_elev_deg 와 sun_azim_deg 는 함께 None 이거나 함께 값")
    if theta.lut not in LUT_NAMES:
        raise _err(f"lut {theta.lut!r} 미등록")
    if theta.lens_rx not in LENS_RX_NAMES:
        raise _err(f"lens_rx {theta.lens_rx!r} 미등록")
    if not 3.0 <= theta.fov_x_deg() <= 130.0:
        raise _err(f"가로 화각 {theta.fov_x_deg():.2f} 도가 [3,130] 밖")
    ex = _canon_extra(theta.extra)
    for k in ("blades", "cpl_angle_deg", "rolloff", "atmos_g"):
        if not _in_bounds(k, float(ex[k])):
            raise _err(f"extra.{k}={ex[k]!r} 가 범위 {HARD_BOUNDS[k]} 밖")
    sq = ex["squeeze"]
    if sq not in (1.0, 2.0):
        raise _err(f"squeeze={sq} 는 1.0 또는 2.0")
    if (theta.lens_rx == "anamorphic_2x") != (sq == 2.0):
        raise _err("lens_rx==anamorphic_2x 와 squeeze==2.0 은 함께여야 한다")
    if theta.lens_rx == "anamorphic_2x" and theta.aspect < 2.0:
        raise _err("애너모픽은 aspect>=2.0")


# ------------------------------------------------------------------------------------------------ 직렬화
def _plain(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def theta_to_dict(theta: Theta) -> dict:
    t = complete_extra(theta)
    return {f.name: _plain(getattr(t, f.name)) for f in fields(Theta)}


def theta_from_dict(d: Any) -> Theta:
    if not isinstance(d, dict):
        raise _err("dict 가 필요하다")
    unknown = set(d) - set(_THETA_FIELDS)
    if unknown:
        raise _err(f"모르는 키 {sorted(map(str, unknown))}")
    kw = copy.deepcopy(d)
    if "extra" in kw and not isinstance(kw["extra"], dict):
        raise _err("extra 는 dict")
    try:
        t = complete_extra(Theta(**kw))
    except ContractError:
        raise
    except (TypeError, ValueError, OverflowError, KeyError, AttributeError) as e:
        raise _err(str(e)) from e
    validate_theta(t)
    return t


def exposure_time_s(theta: Theta) -> float:
    return float(theta.shutter_deg) / (360.0 * float(theta.fps))


# ------------------------------------------------------------------------------------------------ 샘플러
_SENSORS = ((8.8, .05), (17.3, .15), (24.89, .40), (36.0, .30), (54.12, .10))
_SHUTTERS = ((45.0, .05), (90.0, .15), (144.0, .05), (172.8, .05), (180.0, .60), (270.0, .10))
_FPS = ((24.0, .50), (25.0, .10), (30.0, .15), (48.0, .05), (60.0, .10), (120.0, .10))
_ZERO_INFLATED = (  # (name, p_zero, lo, hi, log)
    ("nd_stops", .6, 0.3, 10.0, False), ("cpl", .7, 0.2, 1.0, False), ("grain", .3, 0.05, 0.7, False),
    ("halation", .5, 0.05, 0.6, False), ("vignette", .3, 0.05, 0.7, False), ("rim", .5, 0.1, 1.0, False),
    ("atmos_beta", .4, 1e-4, 2e-2, True),
)
_LUT_P = (.35, .15) + (.0625,) * 8
_LENS_P = (.6, .1, .1, .1, .1)  # LENS_RX_NAMES 순서가 아니라 아래 _LENS_ORDER 순서
_LENS_ORDER = ("spherical_clean", "vintage_petzval", "vintage_double_gauss", "anamorphic_2x", "soft_single_coat")
_ASPECT_SPH = ((1.3333, .10), (1.5, .05), (1.7778, .55), (1.85, .10), (2.0, .05), (2.39, .15))
_ASPECT_ANA = ((2.0, .2), (2.39, .7), (2.76, .1))
_SAMPLE_KEYS = frozenset(_THETA_FIELDS) - {"extra"} | frozenset(EXTRA_DEFAULTS)


def _choice(rng, table):
    p = np.array([q for _, q in table], dtype=np.float64)
    return float(table[int(rng.choice(len(table), p=p / p.sum()))][0])


def _loguni(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_theta(rng, *, fixed: Optional[Mapping[str, Any]] = None) -> Theta:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    fixed = dict(fixed or {})
    for k in fixed:
        if k not in _SAMPLE_KEYS:
            raise _err(f"fixed 에 허용되지 않는 키 {k!r}")
    # 고정된 값 자체가 비정상인 경우를 먼저 거른다(Theta 생성자가 최종 검사).

    # 난수는 고정 여부와 무관하게 항상 같은 순서로 소비한다(고정하지 않은 필드가 안정적).
    sensor_s = _choice(rng, _SENSORS)
    fov_s = _loguni(rng, 15.0, 95.0)
    f_stop = _loguni(rng, 1.2, 22.0)
    focus_m = _loguni(rng, 0.5, 50.0)
    wb_kelvin = _loguni(rng, 2500.0, 10000.0)
    shutter = _choice(rng, _SHUTTERS)
    fps = _choice(rng, _FPS)
    zi = {}
    for name, p0, lo, hi, log in _ZERO_INFLATED:
        is_zero = rng.random() < p0
        val = _loguni(rng, lo, hi) if log else float(rng.uniform(lo, hi))
        zi[name] = 0.0 if is_zero else val
    ev = float(np.clip(rng.normal(0.0, 0.7), -3.0, 3.0))
    tint = float(np.clip(rng.normal(0.0, 0.15), -0.5, 0.5))
    lut = LUT_NAMES[int(rng.choice(len(LUT_NAMES), p=np.array(_LUT_P)))]
    lens_s = _LENS_ORDER[int(rng.choice(len(_LENS_ORDER), p=np.array(_LENS_P)))]
    u_asp = float(rng.random())
    sun_none = bool(rng.random() < 0.25)
    elev_s = float(rng.uniform(-5.0, 75.0))
    azim_s = float(rng.uniform(0.0, 360.0))
    blades = int(rng.integers(5, 13))
    cpl_angle = float(rng.uniform(0.0, 180.0))
    rolloff = float(rng.uniform(0.0, 1.0))
    atmos_g = float(rng.uniform(0.0, 0.9))

    def pick(table, u):
        c = 0.0
        for v, p in table:
            c += p
            if u < c:
                return float(v)
        return float(table[-1][0])

    # --- 렌즈 처방과 squeeze, aspect 결합
    lens = fixed.get("lens_rx", None)
    if lens is None:
        lens = lens_s
        if "squeeze" in fixed:
            sq = fixed["squeeze"]
            if sq == 2.0 and _is_real(sq):
                lens = "anamorphic_2x"
            elif lens == "anamorphic_2x":
                lens = "spherical_clean"
        elif "aspect" in fixed and _is_real(fixed["aspect"]) and fixed["aspect"] < 2.0 and lens == "anamorphic_2x":
            lens = "spherical_clean"
    ana = lens == "anamorphic_2x"
    aspect = pick(_ASPECT_ANA if ana else _ASPECT_SPH, u_asp)
    squeeze = 2.0 if ana else 1.0

    # --- 센서·초점거리
    if "sensor_width_mm" in fixed:
        sensor = fixed["sensor_width_mm"]
    elif "focal_mm" in fixed:
        sensor = 24.89
    else:
        sensor = sensor_s
    if "focal_mm" in fixed:
        focal = fixed["focal_mm"]
    else:
        if not _is_real(sensor) or not float(sensor) > 0:
            raise _err(f"sensor_width_mm={sensor!r}")
        focal = float(sensor) / (2.0 * math.tan(math.radians(fov_s) / 2.0))

    # --- 태양
    if "sun_elev_deg" in fixed or "sun_azim_deg" in fixed:
        elev = fixed.get("sun_elev_deg", "_")
        azim = fixed.get("sun_azim_deg", "_")
        if elev is None and azim == "_":
            azim = None
        if azim is None and elev == "_":
            elev = None
        if elev == "_":
            elev = elev_s
        if azim == "_":
            azim = azim_s
    elif sun_none:
        elev = azim = None
    else:
        elev, azim = elev_s, azim_s

    kw = dict(aspect=aspect, focal_mm=focal, sensor_width_mm=sensor, f_stop=f_stop, focus_m=focus_m,
              shutter_deg=shutter, fps=fps, ev=ev, wb_kelvin=wb_kelvin, wb_tint=tint, lut=lut, lens_rx=lens,
              sun_elev_deg=elev, sun_azim_deg=azim, **zi)
    extra = {"squeeze": squeeze, "blades": blades, "cpl_angle_deg": cpl_angle, "rolloff": rolloff,
             "atmos_g": atmos_g}
    for k, v in fixed.items():
        if k in EXTRA_DEFAULTS:
            extra[k] = v
        elif k in _THETA_FIELDS:
            kw[k] = v
    try:
        t = complete_extra(Theta(**kw, extra=extra))
    except ContractError:
        raise
    except (TypeError, ValueError, OverflowError, KeyError, AttributeError) as e:
        raise _err(str(e)) from e
    validate_theta(t)
    return t


def sample_thetas(n: int, seed: int = 0, *, fixed: Optional[Mapping[str, Any]] = None) -> list:
    if n < 0:
        raise ValueError("n 은 0 이상")
    return [sample_theta(np.random.default_rng(np.random.SeedSequence([int(seed), i])), fixed=fixed)
            for i in range(int(n))]


# ------------------------------------------------------------------------------------------------ 프리셋
@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    theta: Theta
    provenance: str
    tags: tuple = ()


def list_presets() -> tuple:
    return tuple(sorted(p.stem for p in PRESETS_DIR.glob("*.yaml")))


def _parse_preset(path: Path) -> Preset:
    try:
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise _err(f"{path.name}: YAML 오류 {e}") from e
    if not isinstance(d, dict):
        raise _err(f"{path.name}: 최상위는 매핑이어야 한다")
    for k in ("name", "description", "provenance", "theta"):
        if k not in d:
            raise _err(f"{path.name}: 필수 키 {k!r} 없음")
    unknown = set(d) - {"name", "description", "provenance", "tags", "theta"}
    if unknown:
        raise _err(f"{path.name}: 모르는 키 {sorted(map(str, unknown))}")
    for k in ("name", "description", "provenance"):
        if not isinstance(d[k], str) or not d[k].strip():
            raise _err(f"{path.name}: {k} 는 비어 있지 않은 문자열")
    tags = d.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
        raise _err(f"{path.name}: tags 는 문자열 목록")
    if not isinstance(d["theta"], dict):
        raise _err(f"{path.name}: theta 는 매핑이어야 한다")
    return Preset(name=d["name"], description=d["description"], theta=theta_from_dict(d["theta"]),
                  provenance=d["provenance"], tags=tuple(tags))


def load_preset(name_or_path) -> Preset:
    if isinstance(name_or_path, Path) or str(name_or_path).lower().endswith((".yaml", ".yml")):
        path = Path(name_or_path)
    else:
        path = PRESETS_DIR / (str(name_or_path) + ".yaml")
    if not path.is_file():
        raise KeyError(f"프리셋 없음: {name_or_path}")
    return _parse_preset(path)


def load_presets() -> dict:
    return {n: load_preset(n) for n in list_presets()}
