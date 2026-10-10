"""FND-04 계보 태그·prov 비트 v1 (taint, ext/comp/hold, BLOCK_TRAIN/DEV).

방화벽의 어휘. 표준 라이브러리만 사용하며 import 부작용이 없다.
모든 판정은 닫힌 쪽 실패: 모르는 태그·라이선스·역할은 금지 쪽으로 처리한다.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TagError", "ProvError", "FirewallError", "KNOWN_BASES", "EXT_BASES", "BLOCK_TRAIN", "BLOCK_DEV",
    "TRAIN_LICENSES", "MODES", "NAMESPACES", "PROV_TAG", "ProvRecord", "normalize_tag", "parse_tag",
    "as_record", "implied_ns", "most_restrictive", "ext_bit", "comp_bit", "hold_bit", "prov_bits",
    "merkle_parts", "content_address_kind", "derive", "why_not_train", "train_ok", "why_not_dev",
    "dev_ok", "signal_class", "assert_train_ok", "assert_dev_ok",
]


class TagError(ValueError):
    """태그 문법 위반 또는 모르는 BASE."""


class ProvError(ValueError):
    """ProvRecord 필드 위반."""


class FirewallError(RuntimeError):
    """방화벽 위반."""


KNOWN_BASES = frozenset({
    "ENGINE", "CAPTURE", "ASSET", "OPEN-GEN", "HUMAN", "SI", "TWIN", "COMMISSION", "USER",
    "USER-DERIVED", "BIO", "EXTGEN", "EXTGEN-DERIVED", "EXTGEN-INSPIRED", "COMP", "HOLDOUT",
    "PUB-SEALED", "EVAL-ONLY", "IP-FLAG", "OPTIN", "TEL-SYS", "TEL-CAGG", "D3", "NC",
})
EXT_BASES = frozenset({"USER", "USER-DERIVED", "BIO", "EXTGEN", "EXTGEN-DERIVED", "EXTGEN-INSPIRED"})
BLOCK_TRAIN = EXT_BASES | {"COMP", "HOLDOUT", "PUB-SEALED", "EVAL-ONLY", "TEL-CAGG", "OPTIN", "D3", "NC"}
BLOCK_DEV = EXT_BASES | {"COMP", "HOLDOUT", "PUB-SEALED", "EVAL-ONLY", "TEL-CAGG", "OPTIN"}
TRAIN_LICENSES = frozenset({"CC0", "PD", "CC-BY", "Apache-2.0"})
MODES = ("none", "inspired", "derived", "authored")
NAMESPACES = ("user_vault", "ext_vault", "comp_vault", "pub_bench", "train_ok")
PROV_TAG = "buc.prov.v1"

_NS_RANK = {n: i for i, n in enumerate(NAMESPACES)}
_MODE_RANK = {m: i for i, m in enumerate(MODES)}
_QUAL_REQUIRED = frozenset({"EXTGEN", "EXTGEN-DERIVED", "EXTGEN-INSPIRED", "TWIN"})
_OPEN_ROLES = ("train", "dev-eval", "bench")
_SCOPES = ("eval", "train", "publish")
_ALIASES = {"SOUL": "EXTGEN:soul", "SOUL-DERIVED": "EXTGEN-DERIVED:soul",
            "SOUL-INSPIRED": "EXTGEN-INSPIRED:soul"}


def _split(s: str) -> tuple[str, str | None]:
    base, sep, qual = s.partition(":")
    return base, (qual if sep else None)


def normalize_tag(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"태그는 str 이어야 한다: {type(s).__name__}")
    if s in _ALIASES:
        s = _ALIASES[s]
    if not s or any(c.isspace() for c in s):
        raise TagError(f"태그 문법 위반: {s!r}")
    base, qual = _split(s)
    if base not in KNOWN_BASES:
        raise TagError(f"모르는 태그 BASE: {s!r}")
    if qual is not None and qual == "":
        raise TagError(f"한정자가 비었다: {s!r}")
    if base in _QUAL_REQUIRED and qual is None:
        raise TagError(f"한정자 필수: {s!r}")
    if base == "OPEN-GEN":
        parts = qual.split(":") if qual is not None else []
        if len(parts) != 3 or any(not p for p in parts) or parts[2] not in _OPEN_ROLES:
            raise TagError(f"OPEN-GEN:<m>:<h>:<role> 위반: {s!r}")
    elif base == "COMMISSION":
        if qual is None:
            raise TagError(f"한정자 필수: {s!r}")
        scopes = qual.split("+")
        if any(sc not in _SCOPES for sc in scopes) or len(set(scopes)) != len(scopes):
            raise TagError(f"COMMISSION scope 위반: {s!r}")
        qual = "+".join(sorted(scopes, key=_SCOPES.index))
        return f"COMMISSION:{qual}"
    elif base == "IP-FLAG":
        if qual not in (None, "masked"):
            raise TagError(f"IP-FLAG 한정자 위반: {s!r}")
    return s


def parse_tag(s: str) -> tuple[str, str | None]:
    return _split(normalize_tag(s))


def _norm_taint(taint: Any) -> frozenset:
    if isinstance(taint, (str, bytes)) or taint is None:
        raise TypeError("taint 는 str 아닌 iterable[str] 이어야 한다")
    return frozenset(normalize_tag(t) for t in taint)


def implied_ns(taint) -> str:
    bases = {_split(t)[0] for t in _norm_taint(taint)}
    out = "train_ok"
    for b in bases:
        if b in ("USER", "USER-DERIVED", "BIO"):
            n = "user_vault"
        elif b in ("EXTGEN", "EXTGEN-DERIVED", "EXTGEN-INSPIRED"):
            n = "ext_vault"
        elif b in ("COMP", "EVAL-ONLY"):
            n = "comp_vault"
        elif b in ("HOLDOUT", "PUB-SEALED"):
            n = "pub_bench"
        else:
            continue
        out = most_restrictive(out, n)
    return out


def most_restrictive(*ns: str) -> str:
    if not ns:
        raise ValueError("most_restrictive: 빈 입력")
    for n in ns:
        if n not in _NS_RANK:
            raise ValueError(f"모르는 ns: {n!r}")
    return min(ns, key=_NS_RANK.__getitem__)


@dataclass(frozen=True)
class ProvRecord:
    taint: frozenset = frozenset()
    mode: str = "none"
    roots: tuple = ()
    ns: str | None = None

    def __post_init__(self) -> None:
        taint = _norm_taint(self.taint)
        if isinstance(self.roots, (str, bytes)):
            raise ProvError("roots: str 시퀀스여야 한다")
        roots = tuple(sorted(set(self.roots)))
        if any(not isinstance(r, str) for r in roots):
            raise ProvError("roots 원소는 str")
        if self.mode not in _MODE_RANK:
            raise ProvError(f"모르는 mode: {self.mode!r}")
        imp = implied_ns(taint)
        ns = self.ns
        if ns is None:
            ns = imp
        elif ns not in _NS_RANK:
            raise ProvError(f"모르는 ns: {ns!r}")
        elif _NS_RANK[ns] > _NS_RANK[imp]:
            raise ProvError(f"ns {ns!r} 가 implied {imp!r} 보다 느슨하다")
        object.__setattr__(self, "taint", taint)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "ns", ns)

    @property
    def _bases(self) -> set:
        return {_split(t)[0] for t in self.taint}

    @property
    def ext(self) -> bool:
        return bool(self._bases & EXT_BASES)

    @property
    def comp(self) -> bool:
        return "COMP" in self._bases

    @property
    def hold(self) -> bool:
        return bool(self._bases & {"HOLDOUT", "PUB-SEALED"})

    @property
    def bits(self) -> int:
        return int(self.ext) | (int(self.comp) << 1) | (int(self.hold) << 2)

    def to_dict(self) -> dict:
        return {"v": PROV_TAG, "taint": sorted(self.taint), "mode": self.mode,
                "roots": list(self.roots), "ns": self.ns}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ProvRecord":
        if not isinstance(d, Mapping):
            raise TypeError("from_dict: Mapping 이어야 한다")
        if d.get("v") != PROV_TAG:
            raise ProvError(f"prov 버전 불일치: {d.get('v')!r}")
        return cls._from_mapping(d)

    @classmethod
    def _from_mapping(cls, d: Mapping) -> "ProvRecord":
        return cls(taint=frozenset(d.get("taint", ())), mode=d.get("mode", "none"),
                   roots=tuple(d.get("roots", ())), ns=d.get("ns"))


def _from_object(x) -> ProvRecord:
    mode = {"engine": "none", "mixed": "derived"}.get(x.mode, x.mode)
    ns = getattr(x, "ns", None)
    taint = _norm_taint(x.taint)
    if ns == "engine":
        ns = "train_ok"
    if ns is not None:
        if ns not in _NS_RANK:
            raise ProvError(f"모르는 ns: {ns!r}")
        ns = most_restrictive(ns, implied_ns(taint))
    return ProvRecord(taint=taint, mode=mode, roots=tuple(x.roots), ns=ns)


def as_record(x) -> ProvRecord:
    if isinstance(x, ProvRecord):
        return x
    if x is None or isinstance(x, (str, bytes)):
        raise TypeError(f"as_record: 태그 집합이 아니다: {type(x).__name__}")
    if isinstance(x, Mapping):
        if "v" in x and x["v"] != PROV_TAG:
            raise ProvError(f"prov 버전 불일치: {x['v']!r}")
        return ProvRecord._from_mapping(x)
    if all(hasattr(x, a) for a in ("taint", "mode", "roots", "ns")):
        return _from_object(x)
    try:
        it = iter(x)
    except TypeError:
        raise TypeError(f"as_record: 변환 불가 {type(x).__name__}") from None
    return ProvRecord(taint=frozenset(it))


def ext_bit(x) -> bool:
    return as_record(x).ext


def comp_bit(x) -> bool:
    return as_record(x).comp


def hold_bit(x) -> bool:
    return as_record(x).hold


def prov_bits(x) -> int:
    return as_record(x).bits


def merkle_parts(x) -> tuple[str, int, str]:
    r = as_record(x)
    return (PROV_TAG, r.bits, r.ns)


def content_address_kind(x) -> str:
    return "hmac" if as_record(x).ext else "sha256"


def derive(op: str, parents: Sequence, *, human_ref: bool = False,
           updates_shared_weights: bool = False) -> ProvRecord:
    if not isinstance(op, str) or not op:
        raise ValueError("derive: op 는 비어 있지 않은 str")
    if isinstance(parents, (str, bytes)) or parents is None:
        raise TypeError("derive: parents 는 시퀀스")
    recs = [as_record(p) for p in parents]
    if not recs:
        raise ValueError("derive: parents 가 비었다")
    taint: set = set()
    for r in recs:
        for t in r.taint:
            base, qual = _split(t)
            if base == "USER":
                t = "USER-DERIVED" + (f":{qual}" if qual is not None else "")
            elif base == "EXTGEN":
                t = f"EXTGEN-DERIVED:{qual}"
            if human_ref and _split(t)[0] in ("EXTGEN-DERIVED", "EXTGEN-INSPIRED"):
                t = f"EXTGEN-INSPIRED:{_split(t)[1]}"
            taint.add(t)
    mode = max((r.mode for r in recs), key=_MODE_RANK.__getitem__)
    if human_ref and any(r.ext for r in recs):
        mode = "inspired"
    roots = tuple(sorted(set().union(*(r.roots for r in recs))))
    result_taint = frozenset(taint)
    ns = most_restrictive(*(r.ns for r in recs), implied_ns(result_taint))
    out = ProvRecord(taint=result_taint, mode=mode, roots=roots, ns=ns)
    if updates_shared_weights and out.ext:
        raise FirewallError("CT-M5: ext 계보 입력으로 공유 가중치를 갱신할 수 없다")
    return out


def why_not_train(x, branch: str = "main") -> tuple[str, ...]:
    if branch not in ("main", "optin-v1"):
        raise ValueError(f"모르는 branch: {branch!r}")
    r = as_record(x)
    why: set = set()
    if branch == "main":
        for t in r.taint:
            base, qual = _split(t)
            if base in BLOCK_TRAIN:
                why.add(t)
            elif base == "IP-FLAG" and qual is None:
                why.add(t)
            elif base == "OPEN-GEN" and qual.split(":")[2] != "train":
                why.add(t)
            elif base == "COMMISSION" and "train" not in qual.split("+"):
                why.add(t)
            elif base == "ASSET" and qual not in TRAIN_LICENSES:
                why.add(t)
    else:
        allowed = {"CAPTURE", "ENGINE", "ASSET", "HUMAN", "OPTIN"}
        why.update(t for t in r.taint if _split(t)[0] not in allowed)
    if r.mode == "inspired" and r.ext:
        why.add("Q17")
    return tuple(sorted(why))


def train_ok(x, branch: str = "main") -> bool:
    return not why_not_train(x, branch)


def why_not_dev(x) -> tuple[str, ...]:
    r = as_record(x)
    why: set = set()
    for t in r.taint:
        base, qual = _split(t)
        if base in BLOCK_DEV:
            why.add(t)
        elif base == "IP-FLAG" and qual is None:
            why.add(t)
        elif base == "OPEN-GEN" and qual.split(":")[2] not in ("train", "dev-eval"):
            why.add(t)
        elif base == "COMMISSION" and "eval" not in qual.split("+"):
            why.add(t)
    return tuple(sorted(why))


def dev_ok(x) -> bool:
    return not why_not_dev(x)


def signal_class(x) -> str:
    return "S" if dev_ok(x) else "J"


def assert_train_ok(x, what: str = "") -> None:
    w = why_not_train(x)
    if w:
        raise FirewallError(f"train 금지{f' ({what})' if what else ''}: {', '.join(w)}")


def assert_dev_ok(x, what: str = "") -> None:
    w = why_not_dev(x)
    if w:
        raise FirewallError(f"dev 금지{f' ({what})' if what else ''}: {', '.join(w)}")
