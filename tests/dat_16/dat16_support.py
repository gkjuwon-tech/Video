"""DAT-16 테스트 공용 도우미. 도구 모듈은 파일 경로로 적재한다(tools/는 패키지가 아니다)."""
import copy
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "size_prior_check.py"
DATA = ROOT / "data" / "size_prior.yaml"


@lru_cache(maxsize=1)
def tool():
    spec = importlib.util.spec_from_file_location("size_prior_check_dat16", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass가 모듈을 찾을 수 있게
    spec.loader.exec_module(mod)
    return mod


def mk(median, sigma, dim, group, tier, aliases=()):
    """일관된 클래스 항목: range_m = 중앙 95% = median*exp(±2σ)."""
    lo = float(f"{median * math.exp(-2 * sigma):.5g}")
    hi = float(f"{median * math.exp(2 * sigma):.5g}")
    d = {"dim": dim, "median_m": median, "sigma_log": sigma, "range_m": [lo, hi], "group": group, "tier": tier}
    if aliases:
        d["aliases"] = list(aliases)
    return d


def small_raw():
    """손으로 만든 유효 소형 표(12 클래스, 핵심 6). 매 호출 새 사본."""
    c = {
        "door_interior": mk(2.03, 0.04, "height", "architecture", 1, ["interior_door", "door"]),
        "stair_riser": mk(0.17, 0.06, "height", "architecture", 1),
        "chair_seat": mk(0.45, 0.06, "height", "furniture", 1, ["seat_height"]),
        "dining_table": mk(0.75, 0.04, "height", "furniture", 1),
        "person_adult": mk(1.70, 0.06, "height", "person", 1, ["person", "human"]),
        "car": mk(1.5, 0.06, "height", "vehicle", 1, ["automobile"]),
        "coffee_mug": mk(0.095, 0.08, "height", "tableware", 2),
        "wine_bottle": mk(0.30, 0.04, "height", "container", 2),
        "smartphone": mk(0.15, 0.05, "height", "electronics", 2),
        "dog": mk(0.5, 0.3, "height", "animal", 2),
        "tree": mk(15, 0.4, "height", "plant", 2),
        "tennis_ball": mk(0.067, 0.03, "diameter", "sport", 2),
    }
    return {"schema": 1, "unit": "m", "classes": copy.deepcopy(c)}


def small_limits():
    return tool().Limits(min_classes=10, max_classes=50, min_core=5)


def real_raw():
    return tool().read_raw(DATA)
