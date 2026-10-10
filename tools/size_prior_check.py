"""DAT-16 크기 사전 검사기 + 적재기.

표준 라이브러리 + PyYAML + numpy만 쓴다. 파일 경로로 적재 가능하며(tools/는 패키지가 아니다),
import 시 파일을 읽지 않는다. 순수·결정론.

CLI: python3 tools/size_prior_check.py [--path P --min-classes N --max-classes N --min-core N]
  종료 코드: 0 유효, 1 검사 문제, 2 읽기·파싱 불가.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCHEMA_VERSION = 1
DIMS = frozenset({"height", "width", "length", "diameter", "depth"})
GROUPS = frozenset({
    "animal", "appliance", "architecture", "clothing", "container", "electronics", "food",
    "furniture", "misc", "nature", "person", "plant", "sport", "street", "tableware", "tool",
    "vehicle",
})
GROUP_BOUNDS = {
    "architecture": (0.02, 150.0), "furniture": (0.1, 4.0), "appliance": (0.05, 3.0),
    "tableware": (0.01, 1.0), "electronics": (0.01, 3.0), "vehicle": (0.5, 30.0),
    "person": (0.03, 2.2), "animal": (0.005, 8.0), "plant": (0.005, 60.0), "street": (0.05, 30.0),
    "tool": (0.01, 3.0), "sport": (0.02, 3.0), "container": (0.02, 3.0), "clothing": (0.02, 2.0),
    "food": (0.01, 1.0), "nature": (0.005, 200.0), "misc": (0.005, 100.0),
}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "size_prior.yaml"

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{1,39}")
_TOP_REQ = ("schema", "unit", "classes")
_CLS_REQ = ("dim", "median_m", "sigma_log", "range_m", "group", "tier")
_CLS_OPT = ("aliases", "note")
_MED_LO, _MED_HI = 0.005, 200.0
_SIG_LO, _SIG_HI = 0.02, 0.9
_LOG_MED_TOL = 0.15
_SIG_TOL_REL, _SIG_TOL_ABS = 0.35, 0.01
_NORMAL = statistics.NormalDist()


class SizePriorError(ValueError):
    """적재 실패. problems: 문제 문자열 튜플. str(e)는 줄바꿈으로 이은 문제 목록."""

    def __init__(self, problems):
        self.problems = tuple(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Limits:
    min_classes: int = 480
    max_classes: int = 600
    min_core: int = 150

    def __post_init__(self):
        if not (0 <= self.min_core <= self.min_classes <= self.max_classes):
            raise ValueError(
                f"Limits 관계 위반: 0 <= min_core <= min_classes <= max_classes "
                f"({self.min_core}, {self.min_classes}, {self.max_classes})"
            )


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class ClassPrior:
    name: str
    dim: str
    median_m: float
    sigma_log: float
    range_m: tuple
    group: str
    tier: int
    aliases: tuple = ()
    note: object = None

    @property
    def mu_log(self) -> float:
        return math.log(self.median_m)

    def quantile(self, p: float) -> float:
        if not (isinstance(p, (int, float)) and 0 < p < 1):
            raise ValueError(f"분위수 p는 (0,1) 안이어야 한다: {p!r}")
        return self.median_m * math.exp(self.sigma_log * _NORMAL.inv_cdf(float(p)))

    def z(self, size_m: float) -> float:
        if not (isinstance(size_m, (int, float)) and math.isfinite(size_m) and size_m > 0):
            raise ValueError(f"크기는 유한한 양수여야 한다: {size_m!r}")
        return (math.log(size_m) - self.mu_log) / self.sigma_log

    def contains(self, size_m: float, k: float = 2.0) -> bool:
        return abs(self.z(size_m)) <= k


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


@dataclass(frozen=True)
class SizePriorTable:
    classes: dict
    _alias: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        amap = {}
        for name, c in self.classes.items():
            for a in c.aliases:
                amap[a] = name
        object.__setattr__(self, "_alias", amap)

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(sorted(self.classes))

    def __contains__(self, name) -> bool:
        return isinstance(name, str) and name in self.classes

    def __getitem__(self, name: str) -> ClassPrior:
        return self.classes[name]

    def resolve(self, text: str) -> ClassPrior:
        if not isinstance(text, str):
            raise KeyError(text)
        key = _normalize(text)
        if not key:
            raise KeyError(text)
        if key in self.classes:
            return self.classes[key]
        if key in self._alias:
            return self.classes[self._alias[key]]
        raise KeyError(text)

    def names(self, *, group=None, tier=None) -> tuple:
        return tuple(
            n for n in sorted(self.classes)
            if (group is None or self.classes[n].group == group)
            and (tier is None or self.classes[n].tier == tier)
        )

    def core(self) -> tuple:
        return self.names(tier=1)

    def groups(self) -> tuple:
        return tuple(sorted({c.group for c in self.classes.values()}))

    def arrays(self, names):
        names = list(names)
        if not names:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        cs = [self.resolve(n) for n in names]
        mu = np.array([c.mu_log for c in cs], dtype=np.float64)
        sg = np.array([c.sigma_log for c in cs], dtype=np.float64)
        return mu, sg


# ---------------------------------------------------------------- 검사

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_name(x) -> bool:
    return isinstance(x, str) and _NAME_RE.fullmatch(x) is not None


def _check_entry(name, e, probs: list) -> None:
    label = str(name)
    if not isinstance(e, dict):
        probs.append(f"E_SCHEMA: {label}: 항목은 매핑이어야 한다")
        return
    for k in _CLS_REQ:
        if k not in e:
            probs.append(f"E_SCHEMA: {label}: 필수 키 누락 '{k}'")
    for k in e:
        if k not in _CLS_REQ and k not in _CLS_OPT:
            probs.append(f"E_SCHEMA: {label}: 알 수 없는 키 '{k}'")

    if "dim" in e:
        d = e["dim"]
        if not isinstance(d, str):
            probs.append(f"E_SCHEMA: {label}: dim은 문자열이어야 한다")
        elif d not in DIMS:
            probs.append(f"E_ENUM: {label}: dim '{d}'은 허용 값이 아니다")
    if "group" in e:
        g = e["group"]
        if not isinstance(g, str):
            probs.append(f"E_SCHEMA: {label}: group은 문자열이어야 한다")
        elif g not in GROUPS:
            probs.append(f"E_ENUM: {label}: group '{g}'은 허용 값이 아니다")
    if "tier" in e:
        t = e["tier"]
        if type(t) is not int:
            probs.append(f"E_SCHEMA: {label}: tier는 정수여야 한다")
        elif t not in (1, 2):
            probs.append(f"E_ENUM: {label}: tier {t}은 1 또는 2여야 한다")

    med = e.get("median_m")
    sig = e.get("sigma_log")
    rng = e.get("range_m")
    if "median_m" in e and not _is_num(med):
        probs.append(f"E_SCHEMA: {label}: median_m은 숫자여야 한다")
    if "sigma_log" in e and not _is_num(sig):
        probs.append(f"E_SCHEMA: {label}: sigma_log는 숫자여야 한다")
    rng_ok_type = isinstance(rng, list) and len(rng) == 2 and all(_is_num(x) for x in rng)
    if "range_m" in e and not rng_ok_type:
        probs.append(f"E_SCHEMA: {label}: range_m은 숫자 2개짜리 리스트여야 한다")

    if "aliases" in e:
        al = e["aliases"]
        if not isinstance(al, list):
            probs.append(f"E_SCHEMA: {label}: aliases는 리스트여야 한다")
        else:
            for a in al:
                if not isinstance(a, str):
                    probs.append(f"E_SCHEMA: {label}: 별칭은 문자열이어야 한다")
                elif not _is_name(a):
                    probs.append(f"E_NAME: {label}: 별칭 '{a}'은 정규형(소문자·밑줄, 2–40자)이 아니다")
    if "note" in e and not isinstance(e["note"], str):
        probs.append(f"E_SCHEMA: {label}: note는 문자열이어야 한다")

    # ---- 값 검사 (타입이 맞는 것만)
    med_ok = _is_num(med) and math.isfinite(med)
    if _is_num(med) and not (med_ok and _MED_LO <= med <= _MED_HI):
        probs.append(f"E_VALUE: {label}: 중앙값 {med} m가 [{_MED_LO}, {_MED_HI}] 밖이거나 비유한")
    group = e.get("group")
    if med_ok and med > 0 and isinstance(group, str) and group in GROUP_BOUNDS:
        lo_g, hi_g = GROUP_BOUNDS[group]
        if not (lo_g <= med <= hi_g):
            probs.append(f"E_UNIT: {label}: 중앙값 {med} m가 {group} 허용 구간 [{lo_g}, {hi_g}] 밖 (단위 오류 의심)")

    sig_ok = _is_num(sig) and math.isfinite(sig) and sig > 0
    if _is_num(sig) and not (sig_ok and _SIG_LO <= sig <= _SIG_HI):
        probs.append(f"E_VALUE: {label}: sigma_log {sig}이 [{_SIG_LO}, {_SIG_HI}] 밖이거나 비유한")

    rng_finite = rng_ok_type and all(math.isfinite(x) for x in rng)
    if rng_ok_type and not rng_finite:
        probs.append(f"E_VALUE: {label}: range_m이 비유한")
    rng_valid = False
    if rng_finite:
        lo, hi = float(rng[0]), float(rng[1])
        if not (0 < lo < hi):
            probs.append(f"E_VALUE: {label}: range_m은 0 < lo < hi여야 한다 ({lo}, {hi})")
        elif med_ok and not (lo <= med <= hi):
            probs.append(f"E_VALUE: {label}: 중앙값 {med}이 range_m 밖")
        else:
            rng_valid = True

    if rng_valid and med_ok and med > 0 and sig_ok:
        lo, hi = float(rng[0]), float(rng[1])
        if abs(math.log(med) - 0.5 * (math.log(lo) + math.log(hi))) > _LOG_MED_TOL:
            probs.append(f"E_CONSIST: {label}: 중앙값 {med}이 range_m의 기하 중심에서 치우쳤다")
        if abs(sig - math.log(hi / lo) / 4) > _SIG_TOL_REL * sig + _SIG_TOL_ABS:
            probs.append(f"E_CONSIST: {label}: sigma_log {sig}과 range_m 폭 ln(hi/lo)/4가 어긋났다")


def check_raw(raw, *, limits: Limits = DEFAULT_LIMITS) -> list:
    """원시 표를 검사해 문제 목록을 돌려준다. 예외를 던지지 않고, 입력을 바꾸지 않는다.
    빈 목록 = 유효. 결과는 정렬·중복 없음."""
    probs: list = []
    try:
        _check_all(raw, limits, probs)
    except Exception as exc:  # 방어선: 어떤 입력이든 문제 목록으로 돌려준다
        probs.append(f"E_SCHEMA: <table>: 검사 중 내부 오류 ({type(exc).__name__})")
    return sorted(set(probs))


def _check_all(raw, limits: Limits, probs: list) -> None:
    if not isinstance(raw, dict):
        probs.append("E_SCHEMA: <table>: 최상위는 매핑이어야 한다")
        return
    for k in _TOP_REQ:
        if k not in raw:
            probs.append(f"E_SCHEMA: <table>: 최상위 키 누락 '{k}'")
    for k in raw:
        if k not in _TOP_REQ:
            probs.append(f"E_SCHEMA: <table>: 알 수 없는 최상위 키 '{k}'")
    if "schema" in raw and not (type(raw["schema"]) is int and raw["schema"] == SCHEMA_VERSION):
        probs.append(f"E_SCHEMA: <table>: schema는 정수 {SCHEMA_VERSION}이어야 한다")
    if "unit" in raw and raw["unit"] != "m":
        probs.append("E_SCHEMA: <table>: unit은 'm'이어야 한다")
    classes = raw.get("classes")
    if "classes" in raw and not isinstance(classes, dict):
        probs.append("E_SCHEMA: <table>: classes는 매핑이어야 한다")
        return

    if not isinstance(classes, dict):
        return

    # 이름 형식과 항목 검사
    for name in sorted(classes, key=str):
        if not _is_name(name):
            probs.append(f"E_NAME: {name}: 이름은 ^[a-z][a-z0-9_]{{1,39}}$ 형식이어야 한다")
        _check_entry(name, classes[name], probs)

    # 전역 유일성: 정본 이름 먼저, 그다음 별칭
    owners: dict = {}
    for name in sorted(classes, key=str):
        owners[str(name)] = str(name)
    for name in sorted(classes, key=str):
        e = classes[name]
        if not isinstance(e, dict):
            continue
        al = e.get("aliases")
        if not isinstance(al, list):
            continue
        for a in al:
            if not isinstance(a, str):
                continue
            if a in owners:
                probs.append(f"E_DUP: {name}: 별칭 '{a}'가 이미 쓰였다 (소유: {owners[a]})")
            else:
                owners[a] = str(name)

    # 규모
    n = len(classes)
    if not (limits.min_classes <= n <= limits.max_classes):
        probs.append(
            f"E_COUNT: <table>: 클래스 수 {n}이 허용 범위 [{limits.min_classes}, {limits.max_classes}] 밖"
        )
    core = sum(
        1 for e in classes.values()
        if isinstance(e, dict) and type(e.get("tier")) is int and e.get("tier") == 1
    )
    if core < limits.min_core:
        probs.append(f"E_COUNT: <table>: 핵심(tier 1) 클래스 {core}개가 최소 {limits.min_core}개보다 적다")


# ---------------------------------------------------------------- 적재

class _StrictLoader(yaml.SafeLoader):
    """중복 매핑 키를 기록하는 안전 로더. 덮어쓰기 대신 E_DUP으로 보고한다."""


def _strict_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in seen:
                loader.dups.append(key)
            seen.add(key)
        except TypeError:
            pass
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping)


def read_raw(path) -> dict:
    """YAML을 안전하게 적재한다. 중복 키 -> E_DUP, 파싱·파일 오류 -> E_PARSE (SizePriorError)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SizePriorError([f"E_PARSE: <table>: 파일을 읽을 수 없다 ({exc})"]) from None
    loader = _StrictLoader(text)
    loader.dups = []
    try:
        data = loader.get_single_data()
    except yaml.YAMLError as exc:
        raise SizePriorError([f"E_PARSE: <table>: YAML 파싱 실패 ({type(exc).__name__})"]) from None
    finally:
        loader.dispose()
    if loader.dups:
        raise SizePriorError([f"E_DUP: {k}: 같은 매핑에 키가 두 번 나온다" for k in loader.dups])
    if not isinstance(data, dict):
        raise SizePriorError(["E_PARSE: <table>: 최상위가 매핑이 아니다"])
    return data


def _build_table(raw: dict) -> SizePriorTable:
    classes = {}
    for name, e in raw["classes"].items():
        rng = e["range_m"]
        classes[name] = ClassPrior(
            name=name,
            dim=e["dim"],
            median_m=float(e["median_m"]),
            sigma_log=float(e["sigma_log"]),
            range_m=(float(rng[0]), float(rng[1])),
            group=e["group"],
            tier=int(e["tier"]),
            aliases=tuple(e.get("aliases", ())),
            note=e.get("note"),
        )
    return SizePriorTable(classes)


def load_table(path=None, *, limits: Limits = DEFAULT_LIMITS) -> SizePriorTable:
    raw = read_raw(DEFAULT_PATH if path is None else path)
    probs = check_raw(raw, limits=limits)
    if probs:
        raise SizePriorError(probs)
    return _build_table(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DAT-16 크기 사전 검사")
    ap.add_argument("--path", default=None)
    ap.add_argument("--min-classes", type=int, default=DEFAULT_LIMITS.min_classes)
    ap.add_argument("--max-classes", type=int, default=DEFAULT_LIMITS.max_classes)
    ap.add_argument("--min-core", type=int, default=DEFAULT_LIMITS.min_core)
    args = ap.parse_args(argv)
    try:
        limits = Limits(min_classes=args.min_classes, max_classes=args.max_classes, min_core=args.min_core)
    except ValueError as exc:
        print(f"E_COUNT: <table>: {exc}")
        return 2
    path = DEFAULT_PATH if args.path is None else Path(args.path)
    try:
        raw = read_raw(path)
    except SizePriorError as exc:
        for p in exc.problems:
            print(p)
        return 2
    probs = check_raw(raw, limits=limits)
    if probs:
        for p in probs:
            print(p)
        return 1
    table = _build_table(raw)
    print(f"OK {len(table)} classes, core {len(table.core())}, groups {len(table.groups())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
