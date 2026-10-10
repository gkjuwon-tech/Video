"""EVL-01 API: 크롭 수학, load_manifest·Item·Manifest, coverage 집계, read_raw, CLI, 의존성 위생."""
import dataclasses
import json
import subprocess
import sys

import evl01_support as S
import pytest
import yaml

T = S.tool()


# ---------- center_crop_box ----------
@pytest.mark.parametrize("w,h,aspect,box", [
    (6000, 4000, "16:9", (0, 312, 6000, 3687)),
    (4000, 3000, "3:2", (0, 166, 4000, 2833)),
    (4000, 3000, "1:1", (500, 0, 3500, 3000)),
    (1000, 1000, "4:3", (0, 125, 1000, 875)),
    (1920, 1080, "16:9", (0, 0, 1920, 1080)),
    (2000, 861, "native", (0, 0, 2000, 861)),
    (2000, 861, "16:9", (234, 0, 1765, 861)),   # bw = round_half_up(861*16/9 = 1530.67) = 1531, x0 = 469//2
    (1080, 1920, "9:16", (0, 0, 1080, 1920)),
    (3000, 3000, "2.39:1", (0, 872, 3000, 2127)),  # bh = round(1255.23) = 1255, y0 = 1745//2
])
def test_center_crop_box_values(w, h, aspect, box):
    got = T.center_crop_box(w, h, aspect)
    assert isinstance(got, tuple) and len(got) == 4 and all(type(v) is int for v in got)
    assert got == box


def test_center_crop_box_general_properties():
    for w in (513, 777, 1024, 2000, 4031, 8192):
        for h in (512, 600, 999, 2048, 3001):
            for name, a in T.ASPECTS.items():
                x0, y0, x1, y1 = T.center_crop_box(w, h, name)
                assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                bw, bh = x1 - x0, y1 - y0
                assert bw == w or bh == h, "maximal crop touches two sides"
                assert x0 == (w - bw) // 2 and y0 == (h - bh) // 2, "centered (floor)"
                if a is None:
                    assert (bw, bh) == (w, h)
                else:
                    # 반올림 오차 1px 안에서 화면비 일치
                    assert abs(bw - bh * a) <= 0.5 + 1e-9 or abs(bh - bw / a) <= 0.5 + 1e-9


@pytest.mark.parametrize("args", [(0, 100, "1:1"), (100, -1, "1:1"), (100, 100, "5:4"), (100.0, 100, "1:1"),
                                  (True, 100, "1:1")])
def test_center_crop_box_rejects(args):
    with pytest.raises(ValueError):
        T.center_crop_box(*args)


# ---------- load_manifest / Item / Manifest ----------
def _write(tmp_path, raw, name="m.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def test_load_manifest_items(tmp_path):
    raw = S.valid_raw()
    m = T.load_manifest(_write(tmp_path, raw), require_ready=True)
    assert isinstance(m, T.Manifest)
    assert [x.slot for x in m.items] == list(range(1, 25))
    assert m.moves == ("I1", "I2", "I3", "I4", "I5")
    assert m.vault_repo == T.VAULT_REPO
    assert m.selection["candidate_pool"] == 80
    for x, r in zip(m.items, raw["items"]):
        assert isinstance(x, T.Item)
        assert (x.slot, x.id, x.title, x.status, x.kind, x.origin, x.category) == (
            r["slot"], r["id"], r["title"], r["status"], r["kind"], r["origin"], r["category"])
        assert x.features == frozenset(r["features"])
        assert x.taint == tuple(sorted(r["taint"]))
        assert x.size == tuple(r["size"])
        assert x.crop_box == tuple(r["crop"]["box"]) and x.crop_aspect == r["crop"]["aspect"]
        assert x.sha256 == r["sha256"] and x.vault_path == r["vault_path"] and x.generator == r["generator"]
    h = m.hero
    assert h.slot == 1 and h.title == "d15_hero_yak" and h.sha256 == S.HERO_SHA
    assert h.public_ok is False
    assert m.by_slot(7).title == "input_07"
    with pytest.raises(KeyError):
        m.by_slot(25)
    assert len(m.ready()) == 24
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.slot = 2


def test_load_manifest_normalizes_alias(tmp_path):
    raw = S.valid_raw()
    raw["items"][13]["taint"] = ["SOUL", "EVAL-ONLY"]
    m = T.load_manifest(_write(tmp_path, raw))
    assert m.by_slot(14).taint == ("EVAL-ONLY", "EXTGEN:soul")


def test_public_ok_rule(tmp_path):
    m = T.load_manifest(_write(tmp_path, S.valid_raw()))
    for x in m.items:
        expect = x.origin in ("capture", "asset", "engine")
        assert x.public_ok is expect, x.slot


def test_load_manifest_raises_with_all_problems(tmp_path):
    raw = S.valid_raw()
    raw["schema"] = 2
    raw["items"][6]["kind"] = "video"
    with pytest.raises(T.ManifestError) as ei:
        T.load_manifest(_write(tmp_path, raw))
    e = ei.value
    assert isinstance(e, ValueError)
    assert {"E_SCHEMA", "E_ENUM"} <= S.codes(e.problems)
    assert str(e) == "\n".join(e.problems)


def test_load_manifest_ready_mode(tmp_path):
    raw = S.valid_raw()
    S.planned(raw["items"][5])
    p = _write(tmp_path, raw)
    m = T.load_manifest(p)
    assert len(m.ready()) == 23 and m.by_slot(6).status == "planned" and m.by_slot(6).crop_box is None
    with pytest.raises(T.ManifestError) as ei:
        T.load_manifest(p, require_ready=True)
    assert S.codes(ei.value.problems) == {"E_READY"}


# ---------- coverage ----------
def test_coverage_exact(tmp_path):
    cov = T.coverage(T.load_manifest(_write(tmp_path, S.valid_raw())))
    assert cov["total"] == 24 and cov["ready"] == 24
    assert cov["kind"] == {"photo": 12, "gen": 8, "painting": 4}
    assert cov["origin"] == {"capture": 10, "asset": 4, "extgen": 6, "engine": 3, "unverified": 1}
    assert cov["category"] == {"C1": 4, "C2": 4, "C3": 4, "C4": 4, "C5": 4, "C6": 4}
    assert cov["difficulty"] == {"easy": 7, "mid": 9, "hard": 8}
    assert cov["picked_by"] == {"founder": 15, "random": 5, "third_party": 3, "hero": 1}
    assert cov["public_ok"] == 17
    f = cov["features"]
    assert set(f) == set(T.FEATURES)
    assert (f["text"], f["reflective"], f["people"], f["night"], f["fur"], f["bokeh"], f["face"]) == (5, 4, 6, 4, 2, 1, 1)
    mx = cov["matrix"]
    assert set(mx) == set(T.CATEGORIES) and all(set(v) == set(T.KINDS) for v in mx.values())
    assert mx["C1"] == {"photo": 3, "gen": 1, "painting": 0}
    assert mx["C4"] == {"photo": 2, "gen": 2, "painting": 0}
    assert mx["C6"] == {"photo": 0, "gen": 0, "painting": 4}
    assert sum(sum(v.values()) for v in mx.values()) == 24
    json.dumps(cov)  # JSON 직렬화 가능


def test_coverage_constant_shape():
    c = T.COVERAGE
    assert tuple(c["kind"]["photo"]) == (10, 14)
    assert tuple(c["kind"]["gen"]) == (6, 10)
    assert tuple(c["kind"]["painting"]) == (3, 5)


@pytest.mark.parametrize("n_paint", [3, 5])
def test_coverage_band_edges_inclusive(n_paint):
    raw = S.valid_raw()
    items = {x["slot"]: x for x in raw["items"]}
    if n_paint == 3:  # painting 22 -> photo(asset) C2: photo 13, painting 3
        items[22].update(kind="photo", category="C2")
    else:  # gen 20(engine) -> painting C6: painting 5, gen 7
        items[20].update(kind="painting", category="C6")
    assert T.check_raw(raw) == []


def test_coverage_band_edge_exceeded():
    raw = S.valid_raw()
    items = {x["slot"]: x for x in raw["items"]}
    items[20].update(kind="painting", category="C6")
    items[16].update(kind="painting", category="C6")   # painting 6 > 5
    assert "E_COVER" in S.codes(T.check_raw(raw))


# ---------- read_raw ----------
def test_read_raw_errors(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("schema: 1\nschema: 1\nname: inputs24\n", encoding="utf-8")
    with pytest.raises(T.ManifestError) as ei:
        T.read_raw(p)
    assert "E_DUP" in S.codes(ei.value.problems)
    p2 = tmp_path / "bad.yaml"
    p2.write_text("schema: [1,\n", encoding="utf-8")
    with pytest.raises(T.ManifestError) as ei:
        T.read_raw(p2)
    assert "E_PARSE" in S.codes(ei.value.problems)
    with pytest.raises(T.ManifestError) as ei:
        T.read_raw(tmp_path / "missing.yaml")
    assert "E_PARSE" in S.codes(ei.value.problems)
    p3 = tmp_path / "evil.yaml"
    p3.write_text("x: !!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")
    with pytest.raises(T.ManifestError):
        T.read_raw(p3)


def test_read_raw_roundtrip(tmp_path):
    raw = S.valid_raw()
    assert T.read_raw(_write(tmp_path, raw)) == raw


# ---------- CLI ----------
def test_main_exit_codes(tmp_path, capsys):
    good = _write(tmp_path, S.valid_raw(), "good.yaml")
    assert T.main(["--path", str(good), "--ready"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK 24 items (ready 24/24)")

    assert T.main(["--path", str(good), "--matrix"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("OK")
    cov = json.loads("\n".join(lines[1:]))
    assert cov["kind"] == {"photo": 12, "gen": 8, "painting": 4}

    raw = S.valid_raw()
    S.planned(raw["items"][3])
    part = _write(tmp_path, raw, "part.yaml")
    assert T.main(["--path", str(part)]) == 0
    assert "ready 23/24" in capsys.readouterr().out
    assert T.main(["--path", str(part), "--ready"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert out and all(line.startswith("E_") for line in out)

    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: [1,\n", encoding="utf-8")
    assert T.main(["--path", str(bad)]) == 2
    assert T.main(["--path", str(tmp_path / "nope.yaml")]) == 2


def test_script_runs_and_imports_no_torch():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('iq', {str(S.TOOL)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['iq'] = m; spec.loader.exec_module(m)\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'numpy' not in sys.modules, 'numpy imported'\n"
        "print('clean')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd="/", check=False)
    assert r.returncode == 0 and "clean" in r.stdout, r.stderr
    r = subprocess.run([sys.executable, str(S.TOOL), "--path", str(S.MANIFEST)], capture_output=True, text=True,
                       timeout=60, cwd="/", check=False)
    assert r.returncode == 0 and r.stdout.startswith("OK 24 items"), r.stdout + r.stderr
