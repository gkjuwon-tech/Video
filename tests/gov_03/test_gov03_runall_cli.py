"""GOV-03 run_all·CLI: 네 린트를 한 번에, 저장소 기본 검사 통과, 각 린트 카나리아가 실패를 낸다."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from buc.gov import lints as G

REPO = Path(__file__).resolve().parents[2]


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _canary_tree(root):
    """린트마다 정확히 하나씩 위반을 심은 가짜 저장소."""
    _write(root, "buc/render/frame0.py", "def frame0(world, read):\n    return read.image_lin\n")
    _write(root, "buc/teacher/edge.py", "def f(t, read):\n    return t.sample(init=read.image_lin)\n")
    _write(root, "buc/world/refl.py", "def f(F, I_w, m_refl):\n    return F + I_w * m_refl\n")
    _write(root, "buc/world/look.py", "def f(lib, world):\n    lib.add_preset(world.theta_in)\n")
    _write(root, "buc/render/clean.py", "import math\n")


def _clean_tree(root):
    _write(root, "buc/render/clean.py", "import math\n\n\ndef f(world):\n    return world\n")
    _write(root, "buc/teacher/t.py", "def f(t, z):\n    return t.sample(init=z)\n")


def test_run_all_canaries(tmp_path):
    _canary_tree(tmp_path)
    res = G.run_all(tmp_path)
    assert tuple(res) == G.LINTS
    flat = {L: [(v.path, v.rule) for v in vs] for L, vs in res.items()}
    assert flat["no_input_pixels_in_render_path"] == [("buc/render/frame0.py", "INPUT_NAME")]
    # 캔버스·반사 린트 범위(buc/teacher, buc/world, buc/render …)에는 render/frame0.py 도 들지만
    # 그 파일엔 init 키워드도 반사 마스크도 없다
    assert flat["canvas_extension_lint"] == [("buc/teacher/edge.py", "INPUT_INIT")]
    assert flat["refl_bake_lint"] == [("buc/world/refl.py", "REFL_INPUT")]
    assert flat["preset_lineage_check"] == [("buc/world/look.py", "THETA_IN_TO_PRESET")]
    assert G.run_all(tmp_path) == res
    sub = G.run_all(tmp_path, lints=("refl_bake_lint",))
    assert tuple(sub) == ("refl_bake_lint",) and sub["refl_bake_lint"] == res["refl_bake_lint"]
    with pytest.raises(ValueError):
        G.run_all(tmp_path, lints=("nope",))


def test_run_all_includes_preset_dir(tmp_path):
    _clean_tree(tmp_path)
    src = (REPO / "presets" / "neutral_reference.yaml").read_text(encoding="utf-8")
    _write(tmp_path, "presets/neutral_reference.yaml", src.replace("provenance: handmade", "provenance: user"))
    res = G.run_all(tmp_path)
    assert [(v.path, v.rule) for v in res["preset_lineage_check"]] == [("neutral_reference", "PROVENANCE")]
    assert all(res[L] == [] for L in G.LINTS if L != "preset_lineage_check")


def test_run_all_repo_clean():
    res = G.run_all(REPO)
    bad = [(v.lint, v.path, v.line, v.rule, v.detail) for vs in res.values() for v in vs]
    assert bad == [], bad


def _main(argv, capsys):
    code = G.main(argv)
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


def test_main_exit_codes_and_json(tmp_path, capsys):
    _canary_tree(tmp_path / "bad")
    _clean_tree(tmp_path / "good")
    code, js = _main(["--root", str(tmp_path / "good")], capsys)
    assert code == 0 and js == {"ok": True, "violations": []}
    code, js = _main(["--root", str(tmp_path / "bad")], capsys)
    assert code == 1 and js["ok"] is False
    vs = js["violations"]
    assert all(set(v) == {"lint", "path", "line", "rule", "detail"} for v in vs)
    keys = [(v["path"], v["line"], v["lint"], v["rule"], v["detail"]) for v in vs]
    assert keys == sorted(keys)
    assert sorted({v["lint"] for v in vs}) == sorted(G.LINTS)
    code, js = _main(["--root", str(tmp_path / "bad"), "--lint", "refl_bake_lint"], capsys)
    assert code == 1 and {v["lint"] for v in js["violations"]} == {"refl_bake_lint"}
    with pytest.raises(SystemExit) as e:
        G.main(["--lint", "nope"])
    assert e.value.code == 2


def test_violation_shape():
    v = G.Violation(path="a.py", line=3, lint="refl_bake_lint", rule="REFL_INPUT", detail="where")
    assert v.to_dict() == {"lint": "refl_bake_lint", "path": "a.py", "line": 3, "rule": "REFL_INPUT",
                           "detail": "where"}
    w = G.Violation(path="a.py", line=2, lint="z", rule="Z", detail="")
    assert sorted([v, w]) == [w, v]
    with pytest.raises(Exception):
        v.line = 4  # frozen


def test_module_cli_subprocess(tmp_path):
    _canary_tree(tmp_path)
    p = subprocess.run([sys.executable, "-m", "buc.gov.lints", "--root", str(tmp_path),
                        "--lint", "canvas_extension_lint"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert p.returncode == 1, p.stderr
    js = json.loads(p.stdout.strip().splitlines()[-1])
    assert [(v["path"], v["rule"]) for v in js["violations"]] == [("buc/teacher/edge.py", "INPUT_INIT")]


def test_lints_module_is_light():
    """린트 모듈은 torch·numpy 없이 import 된다(CI 에서 가볍게 돈다)."""
    code = ("import sys; import buc.gov.lints; "
            "bad = [m for m in ('torch', 'numpy') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
