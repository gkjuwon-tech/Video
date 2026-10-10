"""GOV-03 preset_lineage_check: 공유 프리셋은 ENGINE·CAPTURE·ASSET 에서만, θ_in 은 공유 프리셋으로 가지 않는다."""
import shutil
import textwrap
from pathlib import Path

import pytest

from buc.core.contracts import Theta
from buc.gov import lints as G
from buc.gov import prov as P
from buc.lens import theta as T

LINT = "preset_lineage_check"
REPO = Path(__file__).resolve().parents[2]


def test_constants():
    assert G.PRESET_ALLOWED_BASES == frozenset({"ENGINE", "CAPTURE", "ASSET"})
    assert G.PRESET_PROVENANCE == {"handmade": "ENGINE", "engine": "ENGINE", "capture": "CAPTURE"}
    assert G.PRESET_SCAN_DIRS == ("buc",) and G.PRESET_EXEMPT_DIRS == ("buc/gov",)


def test_provenance_record():
    assert G.preset_provenance_record("handmade").taint == frozenset({"ENGINE"})
    assert G.preset_provenance_record("engine").taint == frozenset({"ENGINE"})
    assert G.preset_provenance_record("capture").taint == frozenset({"CAPTURE"})
    assert G.preset_provenance_record("asset:CC0").taint == frozenset({"ASSET:CC0"})
    assert isinstance(G.preset_provenance_record("handmade"), P.ProvRecord)
    for bad in ("user", "USER", "soul", "", "asset:", "Handmade ", "theta_in", None, 3):
        with pytest.raises(ValueError):
            G.preset_provenance_record(bad)


@pytest.mark.parametrize("prov,want", [
    (["ENGINE"], ()),
    (["CAPTURE"], ()),
    (["ASSET:CC0"], ()),
    (["ENGINE", "CAPTURE", "ASSET:CC-BY"], ()),
    ({"taint": ["ENGINE"], "mode": "none", "roots": [], "ns": "train_ok"}, ()),
    (["USER"], ("NS:user_vault", "TAINT:USER")),
    (["ENGINE", "USER-DERIVED"], ("NS:user_vault", "TAINT:USER-DERIVED")),
    (["SOUL"], ("NS:ext_vault", "TAINT:EXTGEN:soul")),
    (["HOLDOUT", "CAPTURE"], ("NS:pub_bench", "TAINT:HOLDOUT")),
    (["SI"], ("TAINT:SI",)),
    (["NC", "ASSET:CC-BY-NC"], ("TAINT:NC",)),
    ([], ("NO_TAINT",)),
    # 허용 베이스여도 선언 ns 가 train_ok 보다 좁으면(누군가의 vault 에 있던 것) 공유 금지
    (P.ProvRecord(taint=frozenset({"ENGINE"}), ns="user_vault"), ("NS:user_vault",)),
    (["NOT-A-TAG"], ("UNREADABLE",)),
    ("ENGINE", ("UNREADABLE",)),
    (None, ("UNREADABLE",)),
])
def test_why_not_preset(prov, want):
    assert G.why_not_preset(prov) == want


def _theta(**kw):
    return T.complete_extra(Theta(**kw))


def test_check_presets_entries():
    a = G.PresetEntry(name="eng", theta=_theta(focal_mm=50.0), prov=["ENGINE"])
    b = G.PresetEntry(name="usr", theta=_theta(focal_mm=85.0), prov=["USER"])
    c = G.PresetEntry(name="cap", theta=_theta(focal_mm=24.0), prov=P.ProvRecord(taint=frozenset({"CAPTURE"})))
    vs = G.check_presets([b, a, c])
    assert all(isinstance(v, G.Violation) and v.lint == LINT and v.line == 0 for v in vs)
    assert [(v.path, v.rule, v.detail) for v in vs] == [("usr", "NS", "user_vault"), ("usr", "TAINT", "USER")]
    assert G.check_presets([a, c]) == []
    assert G.check_presets([]) == []


def test_check_presets_lns01_objects():
    th = _theta(focal_mm=35.0)
    ok = T.Preset(name="p1", description="d", theta=th, provenance="handmade", tags=())
    bad = T.Preset(name="p2", description="d", theta=th, provenance="from_user_upload", tags=())
    vs = G.check_presets([ok, bad])
    assert [(v.path, v.rule, v.detail) for v in vs] == [("p2", "PROVENANCE", "from_user_upload")]


def test_theta_in_copy_detected_after_extra_normalisation():
    user_theta_in = Theta(focal_mm=41.0, f_stop=2.8, lut="teal_orange")  # extra 일부 없음
    laundered = G.PresetEntry(name="looks_great", theta=T.complete_extra(user_theta_in), prov=["ENGINE"])
    other = G.PresetEntry(name="other", theta=_theta(focal_mm=42.0, f_stop=2.8, lut="teal_orange"),
                          prov=["ENGINE"])
    vs = G.check_presets([laundered, other], project_thetas=[user_theta_in])
    assert [(v.path, v.rule) for v in vs] == [("looks_great", "THETA_IN_COPY")]
    assert vs[0].detail == T.complete_extra(user_theta_in).content_hash()[:16]
    # 1 ulp 라도 다르면 복사가 아니다(정확 일치 검사; 근사 복제는 범위 밖)
    import math
    near = Theta(focal_mm=math.nextafter(41.0, 99.0), f_stop=2.8, lut="teal_orange")
    assert G.check_presets([laundered], project_thetas=[near]) == []


def test_repo_presets_clean():
    vs = G.check_preset_dir(REPO / "presets")
    assert vs == [], [(v.path, v.rule, v.detail) for v in vs]
    assert len(list((REPO / "presets").glob("*.yaml"))) >= 20


def test_repo_presets_vs_project_theta():
    # 프로젝트 θ_in 이 수제 프리셋과 똑같으면(사용자가 프리셋을 그대로 골랐다) 그 프리셋은 '복사'로 보고된다:
    # 검사는 값 일치만 본다. 하류(LNS/PRD)는 θ_in 이 아닌 프리셋 '이름'을 기록해 이 경우를 피한다.
    p = T.load_preset("neutral_reference")
    vs = G.check_preset_dir(REPO / "presets", project_thetas=[p.theta])
    assert [(v.path, v.rule) for v in vs] == [("neutral_reference", "THETA_IN_COPY")]


def test_preset_dir_bad_files(tmp_path):
    src = REPO / "presets" / "neutral_reference.yaml"
    shutil.copy(src, tmp_path / "neutral_reference.yaml")
    text = src.read_text(encoding="utf-8")
    (tmp_path / "user_look.yaml").write_text(
        text.replace("name: neutral_reference", "name: user_look").replace("provenance: handmade",
                                                                          "provenance: user"),
        encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("name: broken\n: : :\n", encoding="utf-8")
    vs = G.check_preset_dir(tmp_path)
    got = [(v.path, v.rule) for v in vs]
    assert ("user_look", "PROVENANCE") in got
    assert ("broken.yaml", "UNREADABLE") in got
    assert not any(p == "neutral_reference" for p, _ in got)
    assert len(got) == 2 and vs == sorted(vs)


def ast_scan(src, path="buc/x/m.py"):
    vs = G.scan_preset_source(textwrap.dedent(src), path=path, module="buc.x.m")
    for v in vs:
        assert v.lint == LINT and v.path == path and v.line >= 1
    return [(v.line, v.rule) for v in vs]


@pytest.mark.parametrize("src", [
    "def f(lib, world):\n    lib.add_preset(world.theta_in)\n",
    "def f(world):\n    save_preset('mine', theta=world.theta_in)\n",
    "def f(read):\n    PRESET_LIBRARY.register(read.theta_in)\n",
    "def f(theta_in, presets):\n    presets.append(theta_in)\n",
    "def f(theta_in, shared_presets):\n    shared_presets['u1'] = theta_in\n",
    "def f(read, Presets):\n    Presets.store(name='x', theta=fit(read.theta_in))\n",
])
def test_theta_in_to_preset_bad(src):
    got = ast_scan(src)
    assert got == [(2, "THETA_IN_TO_PRESET")], (src, got)


@pytest.mark.parametrize("src", [
    "def f(name):\n    return load_preset(name)\n",
    "def f(world, preset):\n    world.theta_in = preset.theta\n",
    "def f(lib, th):\n    lib.add_preset(th)\n",
    "def f(world):\n    return render(world, theta=world.theta_in)\n",
    "def f(width, height, crf, preset):\n    _check_params(width, height, crf, preset)\n",
    "s = 'presets.add(theta_in)'\n",
])
def test_theta_in_to_preset_good(src):
    assert ast_scan(src) == []


def test_preset_ast_scope(tmp_path):
    for rel in ("buc/world/a.py", "buc/gov/b.py", "buc/lens/c.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("def f(lib, w):\n    lib.add_preset(w.theta_in)\n", encoding="utf-8")
    vs = G.scan_paths(LINT, tmp_path)
    # buc/gov 는 규칙 자체가 사는 곳이라 면제
    assert [v.path for v in vs] == ["buc/lens/c.py", "buc/world/a.py"]


def test_repo_preset_ast_clean():
    vs = G.scan_paths(LINT, REPO)
    assert vs == [], [(v.path, v.line, v.detail) for v in vs]
