"""FND-07 설정: 검증된 불변 Config, 정규 JSON, blake3 해시, 우선순위 로더.

계약: ops/harness/contracts/FND-07.md. stdlib + blake3만 쓴다(numpy/torch 없음).
우선순위: 기본값 < JSON 파일 < 환경변수 BUC_<FIELD> < overrides.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import blake3

__all__ = ["ConfigError", "canonical_json", "Config", "load_config"]

_SEED_MAX = 2**64


class ConfigError(ValueError):
    """설정 값이 잘못됐다(생성자, from_dict, replace, load_config 공통)."""


# --------------------------------------------------------------------------- 정규 JSON

def _check_json(obj, path="$"):
    if obj is None or isinstance(obj, (bool, str)):
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float at {path}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_json(v, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-str key {k!r} at {path}")
            _check_json(v, f"{path}.{k}")
        return
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__} at {path}")


def canonical_json(obj) -> bytes:
    """키 정렬·공백 없음·UTF-8. 정수/실수/불리언은 서로 다른 바이트를 낸다."""
    _check_json(obj)
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return text.encode("utf-8")


# --------------------------------------------------------------------------- 검증 도우미

def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num(name, x, *, nullable=False):
    if x is None:
        if nullable:
            return None
        raise ConfigError(f"{name}: None 불가")
    if not _is_real(x):
        raise ConfigError(f"{name}: 숫자여야 합니다 (받은 타입 {type(x).__name__})")
    v = float(x)
    if not math.isfinite(v) or v < 0:
        raise ConfigError(f"{name}: 유한하고 0 이상이어야 합니다 (받은 값 {x!r})")
    return v


def _nonempty_str(name, x):
    if not isinstance(x, str) or x == "":
        raise ConfigError(f"{name}: 비어 있지 않은 문자열이어야 합니다")
    return x


# --------------------------------------------------------------------------- Config

_CAP_FIELDS = {
    "usd": ("budget_daily_usd", "budget_total_usd"),
    "gpu_h": ("budget_daily_gpu_h", "budget_total_gpu_h"),
    "tfu": ("budget_daily_tfu", "budget_total_tfu"),
}


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0
    runs_dir: str = "runs"
    ledger_path: str = "runs/ledger.sqlite"
    budget_daily_usd: float = 10.0
    budget_total_usd: float = 35.0
    budget_daily_gpu_h: float | None = None
    budget_total_gpu_h: float | None = None
    budget_daily_tfu: float | None = None
    budget_total_tfu: float | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        s = self.seed
        if not isinstance(s, int) or isinstance(s, bool) or not (0 <= s < _SEED_MAX):
            raise ConfigError(f"seed: [0, 2^64) 정수여야 합니다 (받은 값 {s!r})")
        set_ = object.__setattr__
        set_(self, "runs_dir", _nonempty_str("runs_dir", self.runs_dir))
        set_(self, "ledger_path", _nonempty_str("ledger_path", self.ledger_path))
        set_(self, "budget_daily_usd", _num("budget_daily_usd", self.budget_daily_usd))
        set_(self, "budget_total_usd", _num("budget_total_usd", self.budget_total_usd))
        for name in ("budget_daily_gpu_h", "budget_total_gpu_h", "budget_daily_tfu", "budget_total_tfu"):
            set_(self, name, _num(name, getattr(self, name), nullable=True))
        if not isinstance(self.extra, dict):
            raise ConfigError("extra: dict여야 합니다")
        try:
            _check_json(self.extra)
            set_(self, "extra", copy.deepcopy(self.extra))
        except (TypeError, ValueError) as e:
            raise ConfigError(f"extra: JSON 값만 허용됩니다 ({e})") from None

    def to_dict(self) -> dict:
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def _field_names(cls) -> set:
        return {f.name for f in dataclasses.fields(cls)}

    @classmethod
    def from_dict(cls, d) -> "Config":
        if not isinstance(d, dict):
            raise ConfigError("설정은 dict여야 합니다")
        unknown = set(d) - cls._field_names()
        if unknown:
            raise ConfigError(f"알 수 없는 설정 키: {sorted(unknown)}")
        return cls(**d)

    def replace(self, **kw) -> "Config":
        unknown = set(kw) - self._field_names()
        if unknown:
            raise ConfigError(f"알 수 없는 설정 키: {sorted(unknown)}")
        return dataclasses.replace(self, **kw)

    def caps(self, kind: str):
        if kind not in _CAP_FIELDS:
            raise ValueError(f"kind must be one of {tuple(_CAP_FIELDS)}, got {kind!r}")
        daily, total = _CAP_FIELDS[kind]
        return getattr(self, daily), getattr(self, total)

    def hash(self) -> str:
        return blake3.blake3(canonical_json(self.to_dict())).hexdigest()


# --------------------------------------------------------------------------- 로더

_NULLABLE = {"budget_daily_gpu_h", "budget_total_gpu_h", "budget_daily_tfu", "budget_total_tfu"}
_INT_FIELDS = {"seed"}
_FLOAT_FIELDS = {"budget_daily_usd", "budget_total_usd"} | _NULLABLE
_ENV_FIELDS = [f.name for f in dataclasses.fields(Config) if f.name != "extra"]


def _parse_env(name: str, text: str):
    if name in _INT_FIELDS:
        try:
            return int(text)
        except ValueError:
            raise ConfigError(f"BUC_{name.upper()}: 정수가 아닙니다 ({text!r})") from None
    if name in _FLOAT_FIELDS:
        if name in _NULLABLE and text.strip().lower() in ("", "none"):
            return None
        try:
            return float(text)
        except ValueError:
            raise ConfigError(f"BUC_{name.upper()}: 실수가 아닙니다 ({text!r})") from None
    return text


def load_config(path=None, *, env=None, overrides=None) -> Config:
    """기본값 < JSON 파일 < 환경변수 BUC_<FIELD> < overrides. 파일·환경을 바꾸지 않는다."""
    merged: dict = {}
    if path is not None:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as fh:  # 없으면 FileNotFoundError
            text = fh.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(f"설정 파일이 JSON이 아닙니다: {e}") from None
        if not isinstance(data, dict):
            raise ConfigError("설정 파일의 최상위는 객체여야 합니다")
        unknown = set(data) - Config._field_names()
        if unknown:
            raise ConfigError(f"설정 파일의 알 수 없는 키: {sorted(unknown)}")
        merged.update(data)

    source = os.environ if env is None else env
    for name in _ENV_FIELDS:
        key = "BUC_" + name.upper()
        if key in source:
            merged[name] = _parse_env(name, source[key])

    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ConfigError("overrides는 dict여야 합니다")
        unknown = set(overrides) - Config._field_names()
        if unknown:
            raise ConfigError(f"overrides의 알 수 없는 키: {sorted(unknown)}")
        merged.update(overrides)

    return Config.from_dict(merged)
