"""EVL-01 테스트 공용 도우미. 도구 모듈은 파일 경로로 적재한다(tools/는 패키지가 아니다)."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "inputs_qa.py"
MANIFEST = ROOT / "eval" / "inputs24" / "manifest.yaml"

HERO_SHA = "7c1fc8dcebcfb485e93cff2b206405dd4af155acf3085213df82128876362bda"


@lru_cache(maxsize=1)
def tool():
    spec = importlib.util.spec_from_file_location("inputs_qa_evl01", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass가 모듈을 찾을 수 있게
    spec.loader.exec_module(mod)
    return mod


def fake_sha(slot):
    return hashlib.sha256(f"evl01-test-{slot}".encode()).hexdigest()


# (slot, kind, origin, category, features, difficulty, picked_by, generator, size, aspect, taint)
_CAP = (6000, 4000)
SPEC = [
    (1, "gen", "unverified", "C3", ["fur", "people", "bokeh", "far", "outdoor"], "hard", "hero", None, (2000, 861), "native", ["EXTGEN:unknown", "EVAL-ONLY"]),
    (2, "photo", "capture", "C1", ["text", "indoor", "night"], "hard", "random", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (3, "photo", "capture", "C1", ["text", "outdoor", "night", "reflective"], "hard", "founder", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (4, "photo", "capture", "C2", ["indoor", "reflective"], "mid", "third_party", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (5, "photo", "capture", "C2", ["indoor", "reflective", "text"], "mid", "founder", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (6, "photo", "capture", "C3", ["outdoor", "macro"], "mid", "random", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (7, "photo", "capture", "C4", ["outdoor", "far"], "easy", "founder", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (8, "photo", "capture", "C5", ["people", "outdoor"], "hard", "random", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (9, "photo", "capture", "C5", ["people", "indoor", "face"], "mid", "third_party", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (10, "photo", "capture", "C3", ["outdoor", "fur"], "easy", "founder", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (11, "photo", "capture", "C4", ["outdoor", "far", "night"], "mid", "founder", None, _CAP, "16:9", ["CAPTURE", "EVAL-ONLY"]),
    (12, "photo", "asset", "C2", ["indoor"], "easy", "random", None, (4000, 3000), "3:2", ["ASSET:CC0", "EVAL-ONLY"]),
    (13, "photo", "asset", "C1", ["text", "outdoor"], "mid", "founder", None, (4000, 3000), "3:2", ["ASSET:CC0", "EVAL-ONLY"]),
    (14, "gen", "extgen", "C1", ["night", "text", "outdoor"], "hard", "founder", "soul", (2048, 1152), "16:9", ["EXTGEN:soul", "EVAL-ONLY"]),
    (15, "gen", "extgen", "C5", ["people", "outdoor"], "hard", "founder", "soul", (2048, 1152), "16:9", ["EXTGEN:soul", "EVAL-ONLY"]),
    (16, "gen", "extgen", "C3", ["outdoor", "macro"], "easy", "founder", "soul", (2048, 1152), "16:9", ["EXTGEN:soul", "EVAL-ONLY"]),
    (17, "gen", "extgen", "C2", ["indoor", "reflective"], "mid", "founder", "gpt-image", (1536, 1024), "3:2", ["EXTGEN:gpt-image", "EVAL-ONLY"]),
    (18, "gen", "extgen", "C4", ["outdoor", "far"], "easy", "founder", "gpt-image", (1536, 1024), "3:2", ["EXTGEN:gpt-image", "EVAL-ONLY"]),
    (19, "gen", "engine", "C5", ["people", "indoor"], "mid", "random", "z-image-turbo", (2048, 2048), "1:1", ["ENGINE", "EVAL-ONLY"]),
    (20, "gen", "engine", "C4", ["outdoor"], "easy", "founder", "z-image-turbo", (2048, 2048), "1:1", ["ENGINE", "EVAL-ONLY"]),
    (21, "painting", "asset", "C6", ["outdoor"], "hard", "third_party", None, (3000, 2000), "3:2", ["ASSET:PD", "EVAL-ONLY"]),
    (22, "painting", "asset", "C6", ["indoor"], "mid", "founder", None, (3000, 2000), "3:2", ["ASSET:PD", "EVAL-ONLY"]),
    (23, "painting", "extgen", "C6", ["outdoor", "people"], "hard", "founder", "soul", (2048, 2048), "1:1", ["EXTGEN:soul", "EVAL-ONLY"]),
    (24, "painting", "engine", "C6", ["indoor"], "easy", "founder", "z-image-turbo", (2048, 2048), "1:1", ["ENGINE", "EVAL-ONLY"]),
]

# 아래 값은 계약서의 center_crop_box 규칙을 손으로 계산한 것이다(도구에 의존하지 않는다).
BOX = {
    ((6000, 4000), "16:9"): [0, 312, 6000, 3687],
    ((4000, 3000), "3:2"): [0, 166, 4000, 2833],
    ((2048, 1152), "16:9"): [0, 0, 2048, 1152],
    ((1536, 1024), "3:2"): [0, 0, 1536, 1024],
    ((2048, 2048), "1:1"): [0, 0, 2048, 2048],
    ((3000, 2000), "3:2"): [0, 0, 3000, 2000],
    ((2000, 861), "native"): [0, 0, 2000, 861],
}


def item(slot):
    s = next(x for x in SPEC if x[0] == slot)
    _, kind, origin, cat, feats, diff, picked, gen, size, aspect, taint = s
    hero = slot == 1
    return {
        "slot": slot,
        "id": f"in{slot:02d}",
        "title": "d15_hero_yak" if hero else f"input_{slot:02d}",
        "status": "ready",
        "kind": kind,
        "origin": origin,
        "category": cat,
        "features": list(feats),
        "difficulty": diff,
        "picked_by": picked,
        "taint": list(taint),
        "generator": gen,
        "vault_path": "demo/d15_hero_yak.webp" if hero else f"eval/inputs24/in{slot:02d}.webp",
        "sha256": HERO_SHA if hero else fake_sha(slot),
        "size": list(size),
        "crop": {"aspect": aspect, "mode": "center", "box": list(BOX[(size, aspect)])},
    }


def valid_raw():
    """손으로 만든 유효 매니페스트(24 항목, 전부 ready). 매 호출 새 사본."""
    return {
        "schema": 1,
        "name": "inputs24",
        "vault_repo": "sdsdsdsdsdsdsd123/buc-artifacts",
        "moves": ["I1", "I2", "I3", "I4", "I5"],
        "selection": {
            "method": "D0 candidate pool, stratified by category x kind",
            "candidate_pool": 80,
            "seed": 20261010,
            "disclosure": "Inputs were selected by the team; see coverage matrix.",
        },
        "items": [item(i) for i in range(1, 25)],
    }


def planned(it):
    """ready 항목을 planned로 바꾼다(칸 정보는 유지)."""
    it["status"] = "planned"
    it["sha256"] = None
    it["size"] = None
    it["crop"] = None
    return it


def codes(problems):
    return {p.split(":", 1)[0] for p in problems}
