"""LNS-01 테스트 공용 도우미. 구현(buc.lens.theta)의 내부를 참조하지 않고 계약서 값만 쓴다."""
import functools
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PRESETS = REPO / "presets"

LUTS = ("none", "neutral_709", "teal_orange", "bleach_bypass", "kodak_warm", "fuji_cool", "mono_silver",
        "faded_matte", "neon_night", "cross_process")
LENSES = ("spherical_clean", "vintage_petzval", "vintage_double_gauss", "anamorphic_2x", "soft_single_coat")
EXTRA_DEFAULTS = {"squeeze": 1.0, "blades": 9, "cpl_angle_deg": 0.0, "rolloff": 0.5, "atmos_g": 0.0}

# 계약서 HARD_BOUNDS 표 그대로
HARD = {
    "aspect": (1.0, 3.0), "focal_mm": (4.0, 600.0), "sensor_width_mm": (4.8, 70.0), "f_stop": (0.95, 32.0),
    "focus_m": (0.3, 1000.0), "shutter_deg": (1.0, 360.0), "fps": (8.0, 240.0), "nd_stops": (0.0, 12.0),
    "cpl": (0.0, 1.0), "ev": (-5.0, 5.0), "wb_kelvin": (2000.0, 12000.0), "wb_tint": (-0.6, 0.6),
    "grain": (0.0, 1.0), "halation": (0.0, 1.0), "vignette": (0.0, 1.0), "rim": (0.0, 1.0),
    "atmos_beta": (0.0, 0.1), "sun_elev_deg": (-20.0, 90.0), "sun_azim_deg": (0.0, 360.0),
    "blades": (3, 18), "cpl_angle_deg": (0.0, 180.0), "rolloff": (0.0, 1.0), "atmos_g": (0.0, 0.95),
}
HARD_EXCLUSIVE_HI = {"sun_azim_deg", "cpl_angle_deg"}  # 반열린 구간 [lo, hi)

N_SAMPLES = 4000
SEED = 20260101


@functools.lru_cache(maxsize=1)
def samples():
    from buc.lens import theta as T
    return tuple(T.sample_thetas(N_SAMPLES, SEED))


def col(name, ts=None):
    ts = samples() if ts is None else ts
    return np.array([getattr(t, name) for t in ts], dtype=np.float64)


def ecol(name, ts=None):
    ts = samples() if ts is None else ts
    return np.array([t.extra[name] for t in ts], dtype=np.float64)


def fov_of(ts=None):
    ts = samples() if ts is None else ts
    return np.array([t.fov_x_deg() for t in ts], dtype=np.float64)


def sigma_frac(p, n):
    """이항 비율의 표준편차."""
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


def assert_frac(mask, p, label=""):
    """관측 비율이 p에서 6 시그마(+1e-3) 안."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    f = mask.mean()
    tol = 6 * sigma_frac(p, n) + 1e-3
    assert abs(f - p) <= tol, f"{label}: 비율 {f:.4f} != {p} (허용 {tol:.4f}, n={n})"


def ks_uniform01(u):
    """[0,1] 균등과의 KS 통계량."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = u.size
    i = np.arange(1, n + 1)
    return max(float(np.max(i / n - u)), float(np.max(u - (i - 1) / n)))


def assert_uniform(x, lo, hi, label="", log=False):
    x = np.asarray(x, dtype=np.float64)
    assert x.size > 100, f"{label}: 표본 부족"
    assert x.min() >= lo - 1e-9 * max(1, abs(lo)) and x.max() <= hi + 1e-9 * max(1, abs(hi)), \
        f"{label}: 범위 밖 [{x.min()}, {x.max()}] vs [{lo}, {hi}]"
    u = (np.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (x - lo) / (hi - lo)
    d = ks_uniform01(np.clip(u, 0, 1))
    crit = 2.2 / math.sqrt(x.size)  # 유의수준 약 1e-4
    assert d <= crit, f"{label}: KS D={d:.4f} > {crit:.4f}"
    # 양 끝까지 덮는지(균등이면 끝 2% 구간에 표본이 있어야 한다)
    assert (u < 0.02).any() and (u > 0.98).any(), f"{label}: 끝 구간 표본 없음"


def assert_choices(values, table, label="", abs_tol=1e-9):
    """values 가 table(값->확률) 의 키만 갖고 빈도가 확률과 맞는지."""
    values = np.asarray(values, dtype=np.float64)
    keys = np.array(sorted(table), dtype=np.float64)
    for v in np.unique(values):
        assert np.min(np.abs(keys - v)) <= abs_tol, f"{label}: 표에 없는 값 {v}"
    for k, p in table.items():
        assert_frac(np.abs(values - k) <= abs_tol, p, f"{label}={k}")
