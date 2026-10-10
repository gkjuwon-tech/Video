"""EVL-01 검사기: 유효 입력 통과, 변조 탐지(코드별), 쓰레기 입력 무예외·무변경, ready 모드."""
import copy

import evl01_support as S
import pytest

T = S.tool()


def it(raw, slot):
    return next(x for x in raw["items"] if x["slot"] == slot)


def test_valid_manifest_passes_both_modes():
    raw = S.valid_raw()
    assert T.check_raw(raw) == []
    assert T.check_raw(raw, require_ready=True) == []


def test_constants():
    assert T.SCHEMA_VERSION == 1 and T.N_ITEMS == 24
    assert T.VAULT_REPO == "sdsdsdsdsdsdsd123/buc-artifacts"
    assert set(T.MOVES) == {"I1", "I2", "I3", "I4", "I5"}
    assert set(T.KINDS) == {"photo", "gen", "painting"}
    assert set(T.ORIGINS) == {"capture", "asset", "extgen", "engine", "unverified"}
    assert set(T.CATEGORIES) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert set(T.FEATURES) == {"text", "reflective", "fur", "people", "face", "night", "indoor", "outdoor",
                               "macro", "far", "bokeh"}
    assert set(T.ASPECTS) == {"native", "1:1", "4:3", "3:2", "16:9", "21:9", "2.39:1", "9:16"}
    assert T.ASPECTS["native"] is None
    assert T.ASPECTS["16:9"] == pytest.approx(16 / 9, rel=1e-12)
    assert T.ASPECTS["2.39:1"] == pytest.approx(2.39, rel=1e-12)
    assert T.HERO["sha256"] == S.HERO_SHA
    assert T.HERO["vault_path"] == "demo/d15_hero_yak.webp"
    assert tuple(T.HERO["size"]) == (2000, 861)
    assert T.DEFAULT_PATH.resolve() == S.MANIFEST.resolve()


def m_schema2(r): r["schema"] = 2
def m_extra_top(r): r["extra"] = 1
def m_bad_repo(r): r["vault_repo"] = "someone/public-repo"
def m_bad_name(r): r["name"] = "inputs25"
def m_move_enum(r): r["moves"] = ["I1", "I9"]
def m_moves_empty(r): r["moves"] = []
def m_disclosure_empty(r): r["selection"]["disclosure"] = ""
def m_pool_small(r): r["selection"]["candidate_pool"] = 10
def m_slot_bool(r): it(r, 7)["slot"] = True
def m_extra_item_key(r): it(r, 7)["mood"] = "sad"
def m_missing_item_key(r): del it(r, 7)["difficulty"]
def m_features_not_list(r): it(r, 7)["features"] = "outdoor"
def m_drop_item(r): r["items"].pop()
def m_dup_slot(r):
    x = it(r, 5); x["slot"] = 4; x["id"] = "in04"
def m_id_mismatch(r): it(r, 7)["id"] = "in99"
def m_dup_title(r): it(r, 8)["title"] = it(r, 7)["title"]
def m_bad_title(r): it(r, 7)["title"] = "Bad Title"
def m_kind_enum(r): it(r, 7)["kind"] = "video"
def m_feature_enum(r): it(r, 7)["features"] = ["outdoor", "dragon"]
def m_feature_dup(r): it(r, 7)["features"] = ["outdoor", "outdoor", "far"]
def m_diff_enum(r): it(r, 7)["difficulty"] = "brutal"
def m_painting_not_c6(r): it(r, 21)["category"] = "C2"
def m_c6_not_painting(r): it(r, 7)["category"] = "C6"
def m_extgen_no_gen(r): it(r, 14)["generator"] = None
def m_capture_with_gen(r): it(r, 7)["generator"] = "soul"
def m_photo_extgen(r):
    x = it(r, 7); x["origin"] = "extgen"; x["generator"] = "soul"; x["taint"] = ["EXTGEN:soul", "EVAL-ONLY"]
def m_taint_unknown(r): it(r, 7)["taint"] = ["FOO", "EVAL-ONLY"]
def m_taint_capture_missing(r): it(r, 7)["taint"] = ["ASSET:CC0", "EVAL-ONLY"]
def m_taint_no_evalonly(r): it(r, 7)["taint"] = ["CAPTURE"]
def m_taint_wrong_gen(r): it(r, 14)["taint"] = ["EXTGEN:gpt-image", "EVAL-ONLY"]
def m_taint_asset_nolic(r): it(r, 12)["taint"] = ["ASSET", "EVAL-ONLY"]
def m_taint_engine_missing(r): it(r, 19)["taint"] = ["CAPTURE", "EVAL-ONLY"]
def m_taint_empty(r): it(r, 7)["taint"] = []
def m_crop_offcenter(r): it(r, 7)["crop"]["box"] = [0, 0, 6000, 3375]
def m_crop_oob(r): it(r, 7)["crop"]["box"] = [0, 312, 6001, 3687]
def m_crop_inverted(r): it(r, 7)["crop"]["box"] = [100, 312, 100, 3687]
def m_crop_manual_noreason(r): it(r, 7)["crop"].update(mode="manual", box=[0, 0, 6000, 3375])
def m_crop_manual_wrong_aspect(r): it(r, 7)["crop"].update(mode="manual", box=[0, 0, 6000, 4000], reason="keep sky")
def m_crop_aspect_enum(r): it(r, 7)["crop"]["aspect"] = "5:4"
def m_crop_mode_enum(r): it(r, 7)["crop"]["mode"] = "smart"
def m_capture_too_small(r):
    x = it(r, 2); x["size"] = [1600, 1200]; x["crop"] = {"aspect": "native", "mode": "center", "box": [0, 0, 1600, 1200]}
def m_size_too_big(r):
    x = it(r, 7); x["size"] = [9000, 6000]; x["crop"] = {"aspect": "native", "mode": "center", "box": [0, 0, 9000, 6000]}
def m_crop_too_small(r):
    x = it(r, 19); x["crop"] = {"aspect": "1:1", "mode": "manual", "box": [0, 0, 400, 400], "reason": "detail"}
def m_sha_upper(r): it(r, 7)["sha256"] = it(r, 7)["sha256"].upper()
def m_sha_short(r): it(r, 7)["sha256"] = "abc123"
def m_dup_sha(r): it(r, 8)["sha256"] = it(r, 7)["sha256"]
def m_dup_vault(r): it(r, 8)["vault_path"] = it(r, 7)["vault_path"]
def m_path_dotdot(r): it(r, 7)["vault_path"] = "../secret/in07.webp"
def m_path_abs(r): it(r, 7)["vault_path"] = "/data/in07.webp"
def m_path_url(r): it(r, 7)["vault_path"] = "https://example.com/in07.webp"
def m_path_ext(r): it(r, 7)["vault_path"] = "eval/inputs24/in07.txt"
def m_path_backslash(r): it(r, 7)["vault_path"] = "eval\\inputs24\\in07.webp"
def m_ready_missing_sha(r): it(r, 7)["sha256"] = None
def m_planned_with_sha(r): S.planned(it(r, 7))["sha256"] = S.fake_sha(7)
def m_hero_sha(r): it(r, 1)["sha256"] = S.fake_sha(1)
def m_hero_planned(r): S.planned(it(r, 1))
def m_hero_origin(r):
    x = it(r, 1); x["origin"] = "capture"; x["taint"] = ["CAPTURE", "EVAL-ONLY"]
def m_hero_crop(r): it(r, 1)["crop"] = {"aspect": "16:9", "mode": "manual", "box": [470, 0, 1531, 597], "reason": "x"}
def m_hero_features(r): it(r, 1)["features"] = ["people", "bokeh", "far", "outdoor"]
def m_hero_category(r): it(r, 1)["category"] = "C5"
def m_hero_swapped(r):
    a, b = it(r, 1), it(r, 2)
    a["slot"], b["slot"] = 2, 1
    a["id"], b["id"] = "in02", "in01"
def m_hero_tag_elsewhere(r): it(r, 7)["picked_by"] = "hero"
def m_secret_note(r): it(r, 7)["note"] = "token hf_" + "Ab3" * 10
def m_secret_sk(r): r["selection"]["method"] = "key sk-" + "Zz9" * 10
def m_cover_category(r):
    for s in (2, 3, 13, 14):
        it(r, s)["category"] = "C2"
def m_cover_paintings(r):
    for s in (21, 22):
        it(r, s)["kind"] = "photo"; it(r, s)["category"] = "C3"
def m_cover_easy(r):
    for x in r["items"]:
        x["difficulty"] = "easy"
def m_cover_curated(r):
    for x in r["items"]:
        if x["picked_by"] in ("random", "third_party"):
            x["picked_by"] = "founder"
def m_cover_no_text(r):
    for x in r["items"]:
        x["features"] = [f for f in x["features"] if f != "text"]
def m_cover_text_indoor(r):
    for s in (2, 5):
        it(r, s)["features"] = [f for f in it(r, s)["features"] if f != "indoor"] + ["outdoor"] * (s == 2)
def m_cover_no_reflective(r):
    for x in r["items"]:
        x["features"] = [f for f in x["features"] if f != "reflective"]
def m_cover_c4_no_photo(r):
    for s in (7, 11):
        x = it(r, s); x["category"] = "C3"


MUTATIONS = [
    (m_schema2, "E_SCHEMA"), (m_extra_top, "E_SCHEMA"), (m_bad_repo, "E_SCHEMA"), (m_bad_name, "E_SCHEMA"),
    (m_move_enum, "E_ENUM"), (m_moves_empty, "E_SCHEMA"), (m_disclosure_empty, "E_SCHEMA"),
    (m_pool_small, "E_VALUE"), (m_slot_bool, "E_SCHEMA"), (m_extra_item_key, "E_SCHEMA"),
    (m_missing_item_key, "E_SCHEMA"), (m_features_not_list, "E_SCHEMA"),
    (m_drop_item, "E_SLOT"), (m_dup_slot, "E_SLOT"), (m_id_mismatch, "E_SLOT"), (m_dup_title, "E_DUP"),
    (m_bad_title, "E_SCHEMA"), (m_kind_enum, "E_ENUM"), (m_feature_enum, "E_ENUM"), (m_feature_dup, "E_DUP"),
    (m_diff_enum, "E_ENUM"),
    (m_painting_not_c6, "E_CONSIST"), (m_c6_not_painting, "E_CONSIST"), (m_extgen_no_gen, "E_CONSIST"),
    (m_capture_with_gen, "E_CONSIST"), (m_photo_extgen, "E_CONSIST"),
    (m_taint_unknown, "E_TAINT"), (m_taint_capture_missing, "E_TAINT"), (m_taint_no_evalonly, "E_TAINT"),
    (m_taint_wrong_gen, "E_TAINT"), (m_taint_asset_nolic, "E_TAINT"), (m_taint_engine_missing, "E_TAINT"),
    (m_taint_empty, "E_SCHEMA"),
    (m_crop_offcenter, "E_CROP"), (m_crop_oob, "E_CROP"), (m_crop_inverted, "E_CROP"),
    (m_crop_manual_noreason, "E_CROP"), (m_crop_manual_wrong_aspect, "E_CROP"), (m_crop_aspect_enum, "E_ENUM"),
    (m_crop_mode_enum, "E_ENUM"),
    (m_capture_too_small, "E_VALUE"), (m_size_too_big, "E_VALUE"), (m_crop_too_small, "E_VALUE"),
    (m_sha_upper, "E_VALUE"), (m_sha_short, "E_VALUE"), (m_dup_sha, "E_DUP"), (m_dup_vault, "E_DUP"),
    (m_path_dotdot, "E_PATH"), (m_path_abs, "E_PATH"), (m_path_url, "E_PATH"), (m_path_ext, "E_PATH"),
    (m_path_backslash, "E_PATH"),
    (m_ready_missing_sha, "E_READY"), (m_planned_with_sha, "E_READY"),
    (m_hero_sha, "E_HERO"), (m_hero_planned, "E_HERO"), (m_hero_origin, "E_HERO"), (m_hero_crop, "E_HERO"),
    (m_hero_features, "E_HERO"), (m_hero_category, "E_HERO"), (m_hero_swapped, "E_HERO"),
    (m_hero_tag_elsewhere, "E_HERO"),
    (m_secret_note, "E_SECRET"), (m_secret_sk, "E_SECRET"),
    (m_cover_category, "E_COVER"), (m_cover_paintings, "E_COVER"), (m_cover_easy, "E_COVER"),
    (m_cover_curated, "E_COVER"), (m_cover_no_text, "E_COVER"), (m_cover_text_indoor, "E_COVER"),
    (m_cover_no_reflective, "E_COVER"), (m_cover_c4_no_photo, "E_COVER"),
]


@pytest.mark.parametrize("mut,code", MUTATIONS, ids=[m.__name__ for m, _ in MUTATIONS])
def test_mutation_detected(mut, code):
    raw = S.valid_raw()
    mut(raw)
    snap = copy.deepcopy(raw)
    probs = T.check_raw(raw)
    assert raw == snap, "check_raw must not mutate input"
    assert isinstance(probs, list) and all(isinstance(p, str) for p in probs)
    assert probs == sorted(set(probs)), "problems must be sorted and unique"
    assert code in S.codes(probs), f"expected {code}, got {probs}"
    for p in probs:
        parts = p.split(": ", 2)
        assert len(parts) == 3 and parts[0].startswith("E_") and parts[1] and parts[2], p


def test_item_problem_locator_uses_id():
    raw = S.valid_raw()
    it(raw, 7)["kind"] = "video"
    probs = T.check_raw(raw)
    assert any(p.startswith("E_ENUM: in07: ") for p in probs), probs
    raw = S.valid_raw()
    raw["schema"] = 2
    assert any(p.startswith("E_SCHEMA: <manifest>: ") for p in T.check_raw(raw))


def _ok(mut):
    raw = S.valid_raw()
    mut(raw)
    return T.check_raw(raw)


def test_accepted_variants():
    # SOUL 별칭은 EXTGEN:soul로 정규화되어 허용
    assert _ok(lambda r: it(r, 14).__setitem__("taint", ["SOUL", "EVAL-ONLY"])) == []
    # 사유 있는 수동 크롭, 화면비 1% 안
    assert _ok(lambda r: it(r, 7)["crop"].update(mode="manual", box=[0, 0, 6000, 3375], reason="keep sign")) == []
    # 선택 키 note·moves
    assert _ok(lambda r: it(r, 7).update(note="long lens", moves=["I1", "I2"])) == []
    # native 크롭 + 4:3 이미지
    def native(r):
        x = it(r, 19); x["size"] = [2048, 1536]; x["crop"] = {"aspect": "native", "mode": "center", "box": [0, 0, 2048, 1536]}
    assert _ok(native) == []
    # 추가 계보 태그(HOLDOUT)는 금지되지 않는다
    assert _ok(lambda r: it(r, 7).__setitem__("taint", ["CAPTURE", "EVAL-ONLY", "HOLDOUT"])) == []


@pytest.mark.parametrize("garbage", [
    None, 3, "x", [], {}, {"items": None}, {"items": [None] * 24},
    {"schema": 1, "items": [{}]}, {"schema": "1", "name": 5, "moves": "I1", "selection": [], "items": {}},
])
def test_garbage_never_raises(garbage):
    snap = copy.deepcopy(garbage)
    probs = T.check_raw(garbage)
    assert isinstance(probs, list) and probs, garbage
    assert garbage == snap


def test_garbage_items_inside_valid_manifest():
    raw = S.valid_raw()
    raw["items"][6] = None
    raw["items"][7] = {"slot": "eight"}
    raw["items"][8]["crop"] = {"box": "nope"}
    raw["items"][9]["size"] = [1, 2, 3]
    raw["items"][10]["taint"] = [None, 5]
    probs = T.check_raw(raw)
    assert probs and "E_SCHEMA" in S.codes(probs)


def test_planned_items_ok_in_planning_mode_but_not_ready_mode():
    raw = S.valid_raw()
    for s in range(2, 25):
        S.planned(it(raw, s))
        it(raw, s)["vault_path"] = None
    assert T.check_raw(raw) == []
    probs = T.check_raw(raw, require_ready=True)
    assert S.codes(probs) == {"E_READY"}
    assert sum(p.startswith("E_READY") for p in probs) == 23


def test_planned_items_still_count_for_coverage():
    raw = S.valid_raw()
    for s in range(2, 25):
        S.planned(it(raw, s))
    for x in raw["items"]:
        x["difficulty"] = "easy"
    assert "E_COVER" in S.codes(T.check_raw(raw))
