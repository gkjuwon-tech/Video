#!/usr/bin/env python3
"""EVL-01 고정 입력 24장 매니페스트 검사기.

표준 라이브러리 + PyYAML + buc.gov.prov(FND-04)만 쓴다. torch·numpy를 들이지 않는다.
사용: python3 tools/inputs_qa.py [--path P] [--ready] [--matrix]
종료 코드: 0 통과, 1 문제 있음(줄마다 문제), 2 읽기·파싱 불가.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from buc.gov import prov  # noqa: E402  (FND-04 계보 어휘, 표준 라이브러리 전용)

SCHEMA_VERSION = 1
N_ITEMS = 24
NAME = "inputs24"
VAULT_REPO = "sdsdsdsdsdsdsd123/buc-artifacts"
MOVES = ("I1", "I2", "I3", "I4", "I5")
KINDS = ("photo", "gen", "painting")
ORIGINS = ("capture", "asset", "extgen", "engine", "unverified")
CATEGORIES = ("C1", "C2", "C3", "C4", "C5", "C6")
FEATURES = ("text", "reflective", "fur", "people", "face", "night", "indoor", "outdoor", "macro", "far", "bokeh")
DIFFICULTIES = ("easy", "mid", "hard")
PICKED_BY = ("founder", "random", "third_party", "hero")
STATUSES = ("planned", "ready")
CROP_MODES = ("center", "manual")
ASPECTS: dict[str, float | None] = {
    "native": None,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
    "2.39:1": 2.39,
    "9:16": 9 / 16,
}
KIND_ORIGINS = {
    "photo": frozenset({"capture", "asset", "unverified"}),
    "gen": frozenset({"extgen", "engine", "unverified"}),
    "painting": frozenset({"asset", "extgen", "engine", "unverified"}),
}
VAULT_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg", ".heic", ".exr", ".tif", ".tiff")
SIZE_MIN, SIZE_MAX = 512, 8192
CROP_MIN_SIDE = 512
CAPTURE_MIN_SIDE = 1920
MANUAL_ASPECT_TOL = 0.01
SHA_RE = re.compile(r"[0-9a-f]{64}")
TITLE_RE = re.compile(r"[a-z][a-z0-9_]{1,39}")
SECRET_RES = (re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"sk-[A-Za-z0-9]{20,}"))
DEFAULT_PATH = ROOT / "eval" / "inputs24" / "manifest.yaml"

# D15 데모 히어로 (docs/DEMO-D15.md). 슬롯 1 고정값.
HERO: dict[str, Any] = {
    "title": "d15_hero_yak",
    "vault_path": "demo/d15_hero_yak.webp",
    "sha256": "7c1fc8dcebcfb485e93cff2b206405dd4af155acf3085213df82128876362bda",
    "size": [2000, 861],
    "status": "ready",
    "origin": "unverified",
    "kinds": ("photo", "gen"),
    "category": "C3",
    "picked_by": "hero",
    "features": ("fur", "people", "bokeh", "far", "outdoor"),
    "crop": {"aspect": "native", "mode": "center", "box": [0, 0, 2000, 861]},
}

# 칸 커버리지 하한·상한. (lo, hi), None = 제한 없음.
COVERAGE: dict[str, Any] = {
    "kind": {"photo": (10, 14), "gen": (6, 10), "painting": (3, 5)},
    "capture": (8, None),
    "extgen": (None, 10),
    "category": (2, None),        # C1..C5 각각
    "category_photo": (1, None),  # C1..C5 각각의 photo
    "text": (2, None),
    "text_indoor": (1, None),
    "reflective": (2, None),
    "people": (3, None),
    "night": (2, None),
    "fur": (1, None),
    "hard": (6, None),
    "easy": (None, 10),
    "curated": (6, None),         # random + third_party
    "public_ok": (10, None),
}

_TOP_KEYS = frozenset({"schema", "name", "vault_repo", "moves", "selection", "items"})
_SEL_KEYS = frozenset({"method", "candidate_pool", "seed", "disclosure"})
_ITEM_REQ = frozenset({
    "slot", "id", "title", "status", "kind", "origin", "category", "features", "difficulty",
    "picked_by", "taint", "generator", "vault_path", "sha256", "size", "crop",
})
_ITEM_OPT = frozenset({"note", "moves"})
_CROP_REQ = frozenset({"aspect", "mode", "box"})
_CROP_OPT = frozenset({"reason"})
_M = "<manifest>"
_SECRET_MSG = "비밀 패턴(hf_/sk-) 탐지, 값은 기록하지 않는다"


class ManifestError(ValueError):
    """매니페스트 읽기·검사 실패. problems는 문제 문자열 전부."""

    def __init__(self, problems) -> None:
        self.problems = tuple(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Item:
    slot: int
    id: str
    title: str
    status: str
    kind: str
    origin: str
    category: str
    features: frozenset
    difficulty: str
    picked_by: str
    taint: tuple
    generator: str | None
    vault_path: str | None
    sha256: str | None
    size: tuple | None
    crop_aspect: str | None
    crop_box: tuple | None
    public_ok: bool


@dataclass(frozen=True)
class Manifest:
    items: tuple
    moves: tuple
    selection: dict
    vault_repo: str

    @property
    def hero(self) -> Item:
        return self.by_slot(1)

    def by_slot(self, n: int) -> Item:
        for x in self.items:
            if x.slot == n:
                return x
        raise KeyError(n)

    def ready(self) -> tuple:
        return tuple(x for x in self.items if x.status == "ready")


# ---------------------------------------------------------------- 공통 도우미

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _p(out: list, code: str, loc: str, msg: str) -> None:
    out.append(f"{code}: {loc}: {msg}")


def _loc(it: Any, i: int) -> str:
    if isinstance(it, dict):
        s = it.get("slot")
        if _is_int(s) and 1 <= s <= N_ITEMS:
            return f"in{s:02d}"
    return f"item[{i}]"


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def center_crop_box(w: int, h: int, aspect: str) -> tuple[int, int, int, int]:
    """가운데 최대 크롭 박스 (x0, y0, x1, y1). native는 전체 프레임."""
    if not (_is_int(w) and _is_int(h)) or w <= 0 or h <= 0:
        raise ValueError(f"크기는 양의 정수여야 한다: {w!r}x{h!r}")
    if not isinstance(aspect, str) or aspect not in ASPECTS:
        raise ValueError(f"모르는 화면비: {aspect!r}")
    a = ASPECTS[aspect]
    if a is None:
        return (0, 0, w, h)
    if w / h > a:
        bh = h
        bw = _round_half_up(h * a)
    else:
        bw = w
        bh = _round_half_up(w / a)
    x0 = (w - bw) // 2
    y0 = (h - bh) // 2
    return (x0, y0, x0 + bw, y0 + bh)


# ---------------------------------------------------------------- 읽기

class _StrictLoader(yaml.SafeLoader):
    """안전 로더 + 매핑 중복 키 기록(E_DUP)."""

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self.dup_keys: list[str] = []

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, _ in node.value:
                if isinstance(key_node, yaml.ScalarNode) and key_node.tag != "tag:yaml.org,2002:merge":
                    if key_node.value in seen:
                        self.dup_keys.append(key_node.value)
                    seen.add(key_node.value)
        return super().construct_mapping(node, deep=deep)


def read_raw(path) -> Any:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise ManifestError([f"E_PARSE: {_M}: 읽기 실패 {p.name} ({type(e).__name__})"]) from None
    loader = _StrictLoader(text)
    try:
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    except yaml.YAMLError as e:
        raise ManifestError([f"E_PARSE: {_M}: YAML 파싱 실패 ({type(e).__name__})"]) from None
    if loader.dup_keys:
        raise ManifestError([f"E_DUP: {_M}: 매핑 중복 키 {k!r}" for k in sorted(set(loader.dup_keys))])
    return data


# ---------------------------------------------------------------- 검사

def _vault_path_problem(vp: str) -> str | None:
    if "\\" in vp:
        return "백슬래시 금지"
    if vp.startswith("/") or re.match(r"^[A-Za-z]:", vp):
        return "절대 경로 금지"
    if "://" in vp:
        return "URL 금지"
    if any(part in ("", "..") for part in vp.split("/")):
        return "'..' 또는 빈 경로 요소 금지"
    if PurePosixPath(vp).suffix.lower() not in VAULT_SUFFIXES:
        return f"허용 확장자 밖 ({', '.join(VAULT_SUFFIXES)})"
    return None


def _public(norm, origin) -> bool:
    if norm is None or origin is None:
        return False
    return origin != "unverified" and not prov.ext_bit(norm)


def _parse_item(it: Any, loc: str, out: list) -> dict | None:
    if not isinstance(it, dict):
        _p(out, "E_SCHEMA", loc, "항목이 매핑이 아니다")
        return None
    keys = set(it.keys())
    missing = _ITEM_REQ - keys
    for k in sorted(str(x) for x in missing):
        _p(out, "E_SCHEMA", loc, f"누락 키 {k}")
    for k in sorted(str(x) for x in keys - _ITEM_REQ - _ITEM_OPT):
        _p(out, "E_SCHEMA", loc, f"여분 키 {k}")
    if missing:
        return None

    row: dict[str, Any] = {"loc": loc, "raw": it, "ok": True}

    def schema(msg: str) -> None:
        _p(out, "E_SCHEMA", loc, msg)
        row["ok"] = False

    def enum(field: str, allowed: tuple) -> str | None:
        v = it[field]
        if not isinstance(v, str):
            schema(f"{field} 는 문자열이어야 한다")
            return None
        if v not in allowed:
            _p(out, "E_ENUM", loc, f"{field}={v!r} 는 허용 밖")
            return None
        return v

    # slot / id
    s = it["slot"]
    row["slot"] = None
    if not _is_int(s):
        schema("slot 은 정수여야 한다")
    elif not 1 <= s <= N_ITEMS:
        _p(out, "E_SLOT", loc, f"slot {s} 는 1..{N_ITEMS} 밖")
    else:
        row["slot"] = s
        if it["id"] != f"in{s:02d}":
            _p(out, "E_SLOT", loc, f"id {it['id']!r} 는 in{s:02d} 여야 한다")

    t = it["title"]
    row["title"] = t
    if not (isinstance(t, str) and TITLE_RE.fullmatch(t)):
        schema("title 형식 위반 (^[a-z][a-z0-9_]{1,39}$)")

    row["status"] = enum("status", STATUSES)
    row["kind"] = enum("kind", KINDS)
    row["origin"] = enum("origin", ORIGINS)
    row["category"] = enum("category", CATEGORIES)
    row["difficulty"] = enum("difficulty", DIFFICULTIES)
    row["picked_by"] = enum("picked_by", PICKED_BY)

    # features
    f = it["features"]
    row["features"] = None
    if not isinstance(f, list) or not all(isinstance(x, str) for x in f):
        schema("features 는 문자열 리스트여야 한다")
    else:
        bad = [x for x in f if x not in FEATURES]
        for x in bad:
            _p(out, "E_ENUM", loc, f"feature {x!r} 는 허용 밖")
        if len(set(f)) != len(f):
            _p(out, "E_DUP", loc, "features 에 중복")
        if not bad:
            row["features"] = frozenset(f)

    # taint (계보)
    tn = it["taint"]
    row["norm"] = None
    if not isinstance(tn, list) or not tn:
        schema("taint 는 비지 않은 리스트여야 한다")
    elif not all(isinstance(x, str) for x in tn):
        schema("taint 원소는 문자열이어야 한다")
    else:
        norm: list[str] = []
        bad_tag: str | None = None
        for x in tn:
            try:
                norm.append(prov.normalize_tag(x))
            except Exception:  # 닫힌 쪽: 모르는 태그는 불합격
                bad_tag = x
                break
        if bad_tag is not None:
            _p(out, "E_TAINT", loc, f"태그 문법 또는 모르는 BASE: {bad_tag!r}")
        else:
            row["norm"] = tuple(sorted(set(norm)))

    # generator / vault_path / sha256 / size / crop
    g = it["generator"]
    row["generator"] = None
    if g is not None and not (isinstance(g, str) and g):
        schema("generator 는 null 또는 비지 않은 문자열")
    else:
        row["generator"] = g

    vp = it["vault_path"]
    if vp is not None and not (isinstance(vp, str) and vp):
        schema("vault_path 는 null 또는 비지 않은 문자열")

    sh = it["sha256"]
    if sh is not None and not isinstance(sh, str):
        schema("sha256 은 null 또는 문자열")
    elif isinstance(sh, str) and not SHA_RE.fullmatch(sh):
        _p(out, "E_VALUE", loc, "sha256 은 소문자 hex 64자여야 한다")

    sz = it["size"]
    row["size"] = None
    if sz is None:
        pass
    elif not (isinstance(sz, list) and len(sz) == 2 and all(_is_int(v) for v in sz)):
        schema("size 는 null 또는 정수 2개 [w, h]")
    else:
        w, h = sz
        if min(w, h) < 1 or max(w, h) < SIZE_MIN or max(w, h) > SIZE_MAX:
            _p(out, "E_VALUE", loc, f"size {w}x{h}: 긴 변은 {SIZE_MIN}-{SIZE_MAX}")
        if min(w, h) >= 1:
            row["size"] = (w, h)

    cr = it["crop"]
    row["crop"] = None
    if cr is None:
        pass
    elif not isinstance(cr, dict):
        schema("crop 은 null 또는 매핑")
    else:
        ck = set(cr.keys())
        cmiss = _CROP_REQ - ck
        cextra = ck - _CROP_REQ - _CROP_OPT
        if cmiss or cextra:
            schema(f"crop 키 불일치 (누락 {sorted(str(x) for x in cmiss)}, 여분 {sorted(str(x) for x in cextra)})")
        else:
            asp, mode, box = cr["aspect"], cr["mode"], cr["box"]
            reason = cr.get("reason")
            good = True
            if not isinstance(asp, str):
                schema("crop.aspect 는 문자열이어야 한다")
                good = False
            elif asp not in ASPECTS:
                _p(out, "E_ENUM", loc, f"crop.aspect={asp!r} 는 허용 밖")
                good = False
            if not isinstance(mode, str):
                schema("crop.mode 는 문자열이어야 한다")
                good = False
            elif mode not in CROP_MODES:
                _p(out, "E_ENUM", loc, f"crop.mode={mode!r} 는 허용 밖")
                good = False
            if not (isinstance(box, list) and len(box) == 4 and all(_is_int(v) for v in box)):
                schema("crop.box 는 정수 4개 [x0, y0, x1, y1]")
                good = False
            if reason is not None and not isinstance(reason, str):
                schema("crop.reason 은 문자열이어야 한다")
                good = False
            if good:
                row["crop"] = {"aspect": asp, "mode": mode, "box": tuple(box), "reason": reason}

    if "note" in it and not isinstance(it["note"], str):
        schema("note 는 문자열이어야 한다")
    if "moves" in it:
        mv = it["moves"]
        if not (isinstance(mv, list) and all(isinstance(x, str) for x in mv)):
            schema("moves 는 문자열 리스트여야 한다")
        else:
            for x in mv:
                if x not in MOVES:
                    _p(out, "E_ENUM", loc, f"moves 원소 {x!r} 는 허용 밖")
            if len(set(mv)) != len(mv):
                _p(out, "E_DUP", loc, "moves 에 중복")
    return row


def _cross_item(row: dict, out: list) -> None:
    if not row["ok"]:
        return
    loc, it = row["loc"], row["raw"]
    kind, origin, cat, gen = row["kind"], row["origin"], row["category"], row["generator"]

    # 칸 일관성
    if kind is not None and origin is not None and origin not in KIND_ORIGINS[kind]:
        _p(out, "E_CONSIST", loc, f"kind={kind} 에 origin={origin} 은 올 수 없다")
    if kind is not None and cat is not None and ((kind == "painting") != (cat == "C6")):
        _p(out, "E_CONSIST", loc, "painting 이면 C6, C6 이면 painting 이어야 한다")
    if origin is not None and ((gen is not None) != (origin in ("extgen", "engine"))):
        _p(out, "E_CONSIST", loc, "generator 는 origin 이 extgen·engine 일 때만 non-null")

    # 계보 (FND-04 어휘)
    norm = row["norm"]
    if norm is not None and origin is not None:
        if "EVAL-ONLY" not in norm:
            _p(out, "E_TAINT", loc, "EVAL-ONLY 누락 (24장은 학습·보정·회귀 금지)")
        if origin == "capture" and "CAPTURE" not in norm:
            _p(out, "E_TAINT", loc, "capture 인데 CAPTURE 태그 없음")
        if origin == "asset" and not any(t.partition(":")[0] == "ASSET" and t.partition(":")[2] for t in norm):
            _p(out, "E_TAINT", loc, "asset 인데 ASSET:<license> 태그 없음")
        if origin == "extgen" and gen is not None and f"EXTGEN:{gen}" not in norm:
            _p(out, "E_TAINT", loc, f"extgen 인데 EXTGEN:{gen} 태그 없음")
        if origin == "engine" and "ENGINE" not in norm:
            _p(out, "E_TAINT", loc, "engine 인데 ENGINE 태그 없음")
        if origin == "unverified" and not prov.ext_bit(norm):
            _p(out, "E_TAINT", loc, "권리 미확인(unverified)은 ext 비트가 켜져야 한다")

    # 상태
    status = row["status"]
    if status == "ready":
        empty = [k for k in ("vault_path", "sha256", "size", "crop") if it[k] is None]
        if empty:
            _p(out, "E_READY", loc, f"ready 인데 null: {', '.join(empty)}")
    elif status == "planned":
        filled = [k for k in ("sha256", "size", "crop") if it[k] is not None]
        if filled:
            _p(out, "E_READY", loc, f"planned 인데 값이 있다: {', '.join(filled)}")

    vp = it["vault_path"]
    if isinstance(vp, str) and vp:
        problem = _vault_path_problem(vp)
        if problem:
            _p(out, "E_PATH", loc, f"vault_path {problem}")

    # 크롭 기하
    size, crop = row["size"], row["crop"]
    if size is not None and crop is not None:
        w, h = size
        asp, box = crop["aspect"], crop["box"]
        x0, y0, x1, y1 = box
        geo = 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        if not geo:
            _p(out, "E_CROP", loc, f"box {list(box)} 가 {w}x{h} 밖이거나 뒤집혔다")
        if crop["mode"] == "center":
            want = center_crop_box(w, h, asp)
            if box != want:
                _p(out, "E_CROP", loc, f"center 크롭이 아니다 (기대 {list(want)})")
        else:
            reason = crop["reason"]
            if not (isinstance(reason, str) and reason.strip()):
                _p(out, "E_CROP", loc, "manual 크롭은 reason 필수")
            if geo:
                a = ASPECTS[asp] if ASPECTS[asp] is not None else w / h
                ratio = ((x1 - x0) / (y1 - y0)) / a
                if abs(ratio - 1.0) > MANUAL_ASPECT_TOL:
                    _p(out, "E_CROP", loc, f"manual 화면비 오차 {ratio - 1:+.2%} (허용 1%)")
        if geo:
            side = max(x1 - x0, y1 - y0)
            if side < CROP_MIN_SIDE:
                _p(out, "E_VALUE", loc, f"크롭 긴 변 {side} < {CROP_MIN_SIDE}")
            if origin == "capture" and side < CAPTURE_MIN_SIDE:
                _p(out, "E_VALUE", loc, f"capture 크롭 긴 변 {side} < {CAPTURE_MIN_SIDE} (미러리스 2K)")


def _check_moves(mv: Any, out: list) -> None:
    if not isinstance(mv, list) or not mv:
        _p(out, "E_SCHEMA", _M, "moves 는 비지 않은 리스트")
        return
    strs = [x for x in mv if isinstance(x, str)]
    if len(strs) != len(mv):
        _p(out, "E_SCHEMA", _M, "moves 원소는 문자열")
    for x in strs:
        if x not in MOVES:
            _p(out, "E_ENUM", _M, f"moves 원소 {x!r} 는 허용 밖")
    if len(set(strs)) != len(strs):
        _p(out, "E_DUP", _M, "moves 에 중복")


def _check_selection(sel: Any, out: list) -> None:
    if not isinstance(sel, dict):
        _p(out, "E_SCHEMA", _M, "selection 은 매핑")
        return
    keys = set(sel.keys())
    for k in sorted(str(x) for x in _SEL_KEYS - keys):
        _p(out, "E_SCHEMA", _M, f"selection 누락 키 {k}")
    for k in sorted(str(x) for x in keys - _SEL_KEYS):
        _p(out, "E_SCHEMA", _M, f"selection 여분 키 {k}")
    m = sel.get("method")
    if not (isinstance(m, str) and m.strip()):
        _p(out, "E_SCHEMA", _M, "selection.method 는 비지 않은 문자열")
    cp = sel.get("candidate_pool")
    if not _is_int(cp):
        _p(out, "E_SCHEMA", _M, "selection.candidate_pool 은 정수")
    elif cp < N_ITEMS:
        _p(out, "E_VALUE", _M, f"candidate_pool {cp} < {N_ITEMS}")
    if not _is_int(sel.get("seed")):
        _p(out, "E_SCHEMA", _M, "selection.seed 는 정수")
    d = sel.get("disclosure")
    if not (isinstance(d, str) and d.strip()):
        _p(out, "E_SCHEMA", _M, "selection.disclosure 는 비지 않은 문자열")


def _check_slots(items: list, out: list) -> None:
    cnt: Counter = Counter()
    for it in items:
        if isinstance(it, dict) and _is_int(it.get("slot")):
            cnt[it["slot"]] += 1
    for s in range(1, N_ITEMS + 1):
        if cnt.get(s, 0) != 1:
            _p(out, "E_SLOT", _M, f"슬롯 {s} 이 {cnt.get(s, 0)}번 (정확히 1번이어야 한다)")


def _check_secrets(raw: dict, out: list) -> None:
    def hit(v: Any) -> bool:
        return any(rx.search(s) for s in _iter_strings(v) for rx in SECRET_RES)

    items = raw.get("items")
    top = {k: v for k, v in raw.items() if k != "items"}
    if hit(top):
        _p(out, "E_SECRET", _M, _SECRET_MSG)
    if isinstance(items, list):
        for i, it in enumerate(items):
            if hit(it):
                _p(out, "E_SECRET", _loc(it, i), _SECRET_MSG)
    elif hit(items):
        _p(out, "E_SECRET", _M, _SECRET_MSG)


def _check_dups(rows: list[dict], out: list) -> None:
    for field in ("title", "sha256", "vault_path"):
        seen: dict[str, str] = {}
        for r in rows:
            v = r["raw"][field]
            if not isinstance(v, str):
                continue
            if v in seen:
                _p(out, "E_DUP", r["loc"], f"{field} 중복 (먼저 {seen[v]})")
            else:
                seen[v] = r["loc"]


def _check_hero(rows: list[dict], out: list) -> None:
    for r in rows:
        if r["picked_by"] == "hero" and r["slot"] != 1:
            _p(out, "E_HERO", r["loc"], "picked_by=hero 는 슬롯 1 전용")
    heroes = [r for r in rows if r["slot"] == 1]
    if not heroes:
        _p(out, "E_HERO", "in01", "슬롯 1 항목이 없다")
        return
    r = heroes[0]
    raw, crop = r["raw"], r["crop"]
    checks = {
        "title": raw["title"] == HERO["title"],
        "vault_path": raw["vault_path"] == HERO["vault_path"],
        "sha256": raw["sha256"] == HERO["sha256"],
        "size": isinstance(raw["size"], list) and raw["size"] == HERO["size"],
        "status": r["status"] == HERO["status"],
        "origin": r["origin"] == HERO["origin"],
        "kind": r["kind"] in HERO["kinds"],
        "category": r["category"] == HERO["category"],
        "picked_by": r["picked_by"] == HERO["picked_by"],
        "features": r["features"] is not None and set(HERO["features"]) <= r["features"],
        "crop": crop is not None and crop["aspect"] == "native" and crop["mode"] == "center"
        and list(crop["box"]) == HERO["crop"]["box"],
    }
    for name, ok in checks.items():
        if not ok:
            _p(out, "E_HERO", r["loc"], f"{name} 가 히어로 고정값과 다르다")


def _coverage_ready(r: dict) -> bool:
    return all(r[k] is not None for k in ("kind", "origin", "category", "difficulty", "picked_by", "features", "norm"))


def _check_coverage(rows: list[dict], out: list) -> None:
    kind = Counter(r["kind"] for r in rows)
    origin = Counter(r["origin"] for r in rows)
    cat = Counter(r["category"] for r in rows)
    cat_photo = Counter(r["category"] for r in rows if r["kind"] == "photo")
    diff = Counter(r["difficulty"] for r in rows)
    picked = Counter(r["picked_by"] for r in rows)
    public = sum(1 for r in rows if _public(r["norm"], r["origin"]))

    def nf(*fs: str) -> int:
        return sum(1 for r in rows if all(f in r["features"] for f in fs))

    def band(label: str, n: int, rng: tuple) -> None:
        lo, hi = rng
        if (lo is not None and n < lo) or (hi is not None and n > hi):
            _p(out, "E_COVER", _M, f"{label}={n} 가 허용 범위 {rng} 밖")

    for k in KINDS:
        band(f"kind {k}", kind.get(k, 0), COVERAGE["kind"][k])
    band("origin capture", origin.get("capture", 0), COVERAGE["capture"])
    band("origin extgen", origin.get("extgen", 0), COVERAGE["extgen"])
    for c in ("C1", "C2", "C3", "C4", "C5"):
        band(f"category {c}", cat.get(c, 0), COVERAGE["category"])
        band(f"photo {c}", cat_photo.get(c, 0), COVERAGE["category_photo"])
    band("feature text", nf("text"), COVERAGE["text"])
    band("feature text+indoor", nf("text", "indoor"), COVERAGE["text_indoor"])
    band("feature reflective", nf("reflective"), COVERAGE["reflective"])
    band("feature people", nf("people"), COVERAGE["people"])
    band("feature night", nf("night"), COVERAGE["night"])
    band("feature fur", nf("fur"), COVERAGE["fur"])
    band("difficulty hard", diff.get("hard", 0), COVERAGE["hard"])
    band("difficulty easy", diff.get("easy", 0), COVERAGE["easy"])
    band("picked_by random+third_party", picked.get("random", 0) + picked.get("third_party", 0), COVERAGE["curated"])
    band("public_ok", public, COVERAGE["public_ok"])


def check_raw(raw: Any, *, require_ready: bool = False) -> list[str]:
    """쓰레기 입력에도 예외 없이 문제 목록(정렬·중복 없음)을 돌려준다. 입력을 바꾸지 않는다."""
    out: list[str] = []
    try:
        _check(raw, require_ready, out)
    except Exception as e:  # 최후의 안전망: 검사기 자체 오류도 문제로 보고
        out.append(f"E_SCHEMA: {_M}: 검사 중 내부 오류 ({type(e).__name__})")
    return sorted(set(out))


def _check(raw: Any, require_ready: bool, out: list) -> None:
    if not isinstance(raw, dict):
        _p(out, "E_SCHEMA", _M, "최상위가 매핑이 아니다")
        return
    keys = set(raw.keys())
    for k in sorted(str(x) for x in _TOP_KEYS - keys):
        _p(out, "E_SCHEMA", _M, f"누락 키 {k}")
    for k in sorted(str(x) for x in keys - _TOP_KEYS):
        _p(out, "E_SCHEMA", _M, f"여분 키 {k}")
    sch = raw.get("schema")
    if not (_is_int(sch) and sch == SCHEMA_VERSION):
        _p(out, "E_SCHEMA", _M, f"schema 는 {SCHEMA_VERSION}")
    if raw.get("name") != NAME:
        _p(out, "E_SCHEMA", _M, f"name 은 {NAME!r}")
    if raw.get("vault_repo") != VAULT_REPO:
        _p(out, "E_SCHEMA", _M, "vault_repo 가 고정값과 다르다")
    _check_moves(raw.get("moves"), out)
    _check_selection(raw.get("selection"), out)
    _check_secrets(raw, out)

    items = raw.get("items")
    if not isinstance(items, list):
        _p(out, "E_SCHEMA", _M, "items 는 리스트")
        return
    if len(items) != N_ITEMS:
        _p(out, "E_SLOT", _M, f"항목 {len(items)}개 (정확히 {N_ITEMS})")
    _check_slots(items, out)

    rows: list[dict] = []
    for i, it in enumerate(items):
        row = _parse_item(it, _loc(it, i), out)
        if row is not None:
            rows.append(row)
    for row in rows:
        _cross_item(row, out)
    _check_dups(rows, out)
    _check_hero(rows, out)

    complete = len(items) == N_ITEMS and len(rows) == N_ITEMS and all(_coverage_ready(r) for r in rows)
    if complete:
        _check_coverage(rows, out)
    if require_ready:
        for r in rows:
            if r["status"] == "planned":
                _p(out, "E_READY", r["loc"], "ready 모드: planned (촬영 후 sha256·size·crop 확정 필요)")


# ---------------------------------------------------------------- 적재·집계

def _build(raw: dict) -> Manifest:
    items = []
    for r in sorted(raw["items"], key=lambda x: x["slot"]):
        taint = tuple(sorted({prov.normalize_tag(t) for t in r["taint"]}))
        crop = r["crop"]
        items.append(Item(
            slot=r["slot"],
            id=r["id"],
            title=r["title"],
            status=r["status"],
            kind=r["kind"],
            origin=r["origin"],
            category=r["category"],
            features=frozenset(r["features"]),
            difficulty=r["difficulty"],
            picked_by=r["picked_by"],
            taint=taint,
            generator=r["generator"],
            vault_path=r["vault_path"],
            sha256=r["sha256"],
            size=tuple(r["size"]) if r["size"] is not None else None,
            crop_aspect=crop["aspect"] if crop is not None else None,
            crop_box=tuple(crop["box"]) if crop is not None else None,
            public_ok=_public(taint, r["origin"]),
        ))
    return Manifest(
        items=tuple(items),
        moves=tuple(raw["moves"]),
        selection=dict(raw["selection"]),
        vault_repo=raw["vault_repo"],
    )


def load_manifest(path=None, *, require_ready: bool = False) -> Manifest:
    p = DEFAULT_PATH if path is None else Path(path)
    raw = read_raw(p)
    problems = check_raw(raw, require_ready=require_ready)
    if problems:
        raise ManifestError(problems)
    return _build(raw)


def _tally(keys: tuple, values) -> dict:
    c = {k: 0 for k in keys}
    for v in values:
        c[v] += 1
    return c


def coverage(m: Manifest) -> dict:
    items = m.items
    feats = {f: 0 for f in FEATURES}
    for x in items:
        for f in x.features:
            feats[f] += 1
    matrix = {c: {k: 0 for k in KINDS} for c in CATEGORIES}
    for x in items:
        matrix[x.category][x.kind] += 1
    return {
        "kind": _tally(KINDS, (x.kind for x in items)),
        "origin": _tally(ORIGINS, (x.origin for x in items)),
        "category": _tally(CATEGORIES, (x.category for x in items)),
        "difficulty": _tally(DIFFICULTIES, (x.difficulty for x in items)),
        "picked_by": _tally(PICKED_BY, (x.picked_by for x in items)),
        "features": feats,
        "matrix": matrix,
        "public_ok": sum(1 for x in items if x.public_ok),
        "ready": len(m.ready()),
        "total": len(items),
    }


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EVL-01 고정 입력 24장 매니페스트 검사")
    ap.add_argument("--path", default=None, help="매니페스트 경로 (기본: eval/inputs24/manifest.yaml)")
    ap.add_argument("--ready", action="store_true", help="ready 모드: planned 항목이 있으면 실패")
    ap.add_argument("--matrix", action="store_true", help="커버리지 JSON을 출력")
    args = ap.parse_args(argv)

    path = DEFAULT_PATH if args.path is None else Path(args.path)
    try:
        raw = read_raw(path)
    except ManifestError as e:
        for line in e.problems:
            print(line)
        return 2
    problems = check_raw(raw, require_ready=args.ready)
    if problems:
        for line in problems:
            print(line)
        return 1
    m = _build(raw)
    n = len(m.items)
    print(f"OK {n} items (ready {len(m.ready())}/{n}) hero={m.hero.title}")
    if args.matrix:
        print(json.dumps(coverage(m), ensure_ascii=False, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
