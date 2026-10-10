"""PRD-03 스텁 팩 공용 params 검증 도우미. numpy 와 표준 라이브러리만 쓴다(import 부작용 없음)."""
import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional


def check_keys(params: Any, allowed: Iterable[str]) -> dict:
    """알 수 없는 키를 거부한다. ``_seed`` 는 스파인이 채우는 키라 허용한다."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError("params 는 dict 여야 한다")
    allowed_set = set(allowed)
    out = {k: v for k, v in params.items() if k != "_seed"}
    bad = sorted(str(k) for k in out if k not in allowed_set)
    if bad:
        raise ValueError(f"알 수 없는 params 키 {bad} (허용: {sorted(allowed_set)})")
    return out


def finite_float(name: str, v: Any, *, positive: bool = False, lo: Optional[float] = None,
                 hi: Optional[float] = None) -> float:
    if isinstance(v, bool) or not isinstance(v, Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name}: 유한해야 한다 ({f})")
    if positive and f <= 0.0:
        raise ValueError(f"{name}: 0보다 커야 한다 ({f})")
    if lo is not None and f < lo:
        raise ValueError(f"{name}: {lo} 이상이어야 한다 ({f})")
    if hi is not None and f > hi:
        raise ValueError(f"{name}: {hi} 이하여야 한다 ({f})")
    return f


def int_in(name: str, v: Any, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise ValueError(f"{name}: 정수여야 한다 ({v!r})")
    i = int(v)
    if lo is not None and i < lo:
        raise ValueError(f"{name}: {lo} 이상이어야 한다 ({i})")
    if hi is not None and i > hi:
        raise ValueError(f"{name}: {hi} 이하여야 한다 ({i})")
    return i
