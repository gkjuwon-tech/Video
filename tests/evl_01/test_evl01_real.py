"""EVL-01 실제 매니페스트: 계획 모드 통과, 히어로 고정값, 구성(실사·생성·그림), 계보 방화벽, 이미지·비밀 부재."""
import re

import evl01_support as S
import yaml

from buc.gov import prov

T = S.tool()
IMAGE_EXT = {".webp", ".png", ".jpg", ".jpeg", ".heic", ".heif", ".exr", ".tif", ".tiff", ".bmp", ".gif", ".avif"}


def _m():
    return T.load_manifest()  # DEFAULT_PATH, 계획 모드


def test_real_manifest_exists_and_valid():
    assert S.MANIFEST.is_file()
    raw = T.read_raw(S.MANIFEST)
    assert T.check_raw(raw) == []
    m = _m()
    assert len(m.items) == 24
    assert set(m.moves) <= set(T.MOVES) and m.moves
    sel = m.selection
    assert sel["candidate_pool"] >= 24 and sel["disclosure"].strip()


def test_real_hero_is_d15_yak():
    h = _m().hero
    assert h.slot == 1 and h.id == "in01" and h.title == "d15_hero_yak"
    assert h.status == "ready"
    assert h.vault_path == "demo/d15_hero_yak.webp"
    assert h.sha256 == S.HERO_SHA
    assert h.size == (2000, 861)
    assert h.crop_aspect == "native" and h.crop_box == (0, 0, 2000, 861)
    assert h.origin == "unverified" and h.public_ok is False
    assert h.category == "C3"
    assert {"fur", "people", "bokeh", "far", "outdoor"} <= h.features
    assert prov.ext_bit(h.taint)


def test_real_composition_matches_founder_plan():
    cov = T.coverage(_m())
    k = cov["kind"]
    # 실사 12 안팎, 외부 생성기 8 안팎, 그림 4 안팎 (r-inventory §12)
    assert 10 <= k["photo"] <= 14 and 6 <= k["gen"] <= 10 and 3 <= k["painting"] <= 5
    assert cov["origin"]["capture"] >= 8
    assert cov["origin"]["extgen"] <= 10
    for c in ("C1", "C2", "C3", "C4", "C5"):
        assert cov["category"][c] >= 2 and cov["matrix"][c]["photo"] >= 1, c
    assert cov["category"]["C6"] == k["painting"]
    assert cov["public_ok"] >= 10
    assert cov["picked_by"]["random"] + cov["picked_by"]["third_party"] >= 6
    assert cov["difficulty"]["hard"] >= 6 and cov["difficulty"]["easy"] <= 10


def test_real_has_indoor_text_capture_and_mirrorless():
    m = _m()
    # 간판·글자 있는 실내를 자체 촬영으로
    assert any({"text", "indoor"} <= x.features and x.origin == "capture" for x in m.items)
    for x in m.items:
        if x.origin == "capture" and x.size is not None:
            x0, y0, x1, y1 = x.crop_box
            assert max(x1 - x0, y1 - y0) >= 1920


def test_real_firewall_every_item_blocked_from_training():
    m = _m()
    for x in m.items:
        assert "EVAL-ONLY" in x.taint
        assert not prov.train_ok(x.taint), x.id
        assert not prov.dev_ok(x.taint), x.id
        if x.origin in ("extgen", "unverified"):
            assert prov.ext_bit(x.taint) and x.public_ok is False


def test_real_planned_items_have_no_invented_hashes():
    m = _m()
    for x in m.items:
        if x.status == "planned":
            assert x.sha256 is None and x.size is None and x.crop_box is None, x.id
    # 현재 ready는 계약상 히어로뿐이거나, 그 밖의 ready 항목은 고유 해시
    shas = [x.sha256 for x in m.items if x.sha256]
    assert len(shas) == len(set(shas)) and S.HERO_SHA in shas


def test_no_image_originals_in_repo_eval():
    root = S.ROOT / "eval"
    assert root.is_dir()
    imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    assert imgs == [], imgs
    m = _m()
    for x in m.items:
        if x.vault_path:
            assert not (S.ROOT / x.vault_path).exists(), f"vault original present in repo: {x.vault_path}"


def test_no_secrets_in_manifest_text():
    text = S.MANIFEST.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)
    assert not re.search(r"(?i)(token|api[_-]?key|password)\s*[:=]\s*\S{8,}", text)
    yaml.safe_load(text)
