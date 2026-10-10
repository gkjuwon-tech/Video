"""CLI(main)와 check_file: 종료 코드, 마지막 줄 JSON, 파일 취급."""
import json
import subprocess
import sys

import pytest
import yaml

from fnd05_helpers import ROOT, TOOL, D, E, doc, load


def write(tmp_path, d, name="ledger.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def last_json(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return json.loads(lines[-1])


GOOD = doc(E("a", 100), E("b", 50), D("t", 150, "a+b", ["a", "b"]))
BAD = doc(E("a", 100), E("b", 50), D("t", 160, "a+b", ["a", "b"]))        # 6.67% 어긋남


def test_main_ok(tmp_path, capsys):
    p = write(tmp_path, GOOD)
    before = p.read_bytes()
    rc = load().main(["--ledger", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    j = last_json(out)
    assert j == {"ok": True, "checked": 3, "violations": []}
    assert p.read_bytes() == before, "원장 파일을 고치면 안 된다"


def test_main_violation(tmp_path, capsys):
    p = write(tmp_path, BAD)
    rc = load().main(["--ledger", str(p)])
    j = last_json(capsys.readouterr().out)
    assert rc == 1 and j["ok"] is False and j["checked"] == 3
    assert len(j["violations"]) == 1
    v = j["violations"][0]
    assert set(v) == {"check", "id", "why"} and v["check"] == "mismatch" and v["id"] == "t" and v["why"]


def test_main_tol_flag(tmp_path, capsys):
    p = write(tmp_path, BAD)
    assert load().main(["--ledger", str(p), "--tol", "0.1"]) == 0
    assert last_json(capsys.readouterr().out)["ok"] is True
    assert load().main(["--ledger", str(p), "--tol", "0.01"]) == 1
    capsys.readouterr()


@pytest.mark.parametrize("extra", [["--tol", "abc"], ["--tol", "5"], ["--tol", "-1"], ["--bogus"], ["--tol"]])
def test_main_usage_errors_exit_2(tmp_path, capsys, extra):
    p = write(tmp_path, GOOD)
    try:
        rc = load().main(["--ledger", str(p), *extra])
    except SystemExit as e:           # argparse 방식도 허용
        rc = e.code
    assert rc == 2
    capsys.readouterr()


def test_main_missing_file_exit_2(tmp_path, capsys):
    try:
        rc = load().main(["--ledger", str(tmp_path / "nope.yaml")])
    except SystemExit as e:
        rc = e.code
    assert rc == 2
    capsys.readouterr()


@pytest.mark.parametrize("text", ["entries: [unclosed", "- 1\n- 2\n", "just a string\n", "", "schema: [", "{{{"])
def test_broken_yaml_is_violation_exit_1(tmp_path, capsys, text):
    p = tmp_path / "broken.yaml"
    p.write_text(text, encoding="utf-8")
    rc = load().main(["--ledger", str(p)])
    j = last_json(capsys.readouterr().out)
    assert rc == 1 and j["ok"] is False
    assert {v["check"] for v in j["violations"]} == {"schema"}
    r = load().check_file(p)                         # 예외 없이 Report
    assert r.ok is False and r.violations[0].check == "schema"


def test_check_file_api(tmp_path):
    L = load()
    p = write(tmp_path, GOOD)
    for arg in (p, str(p)):
        r = L.check_file(arg)
        assert r.ok and r.checked == 3
    assert L.check_file(write(tmp_path, BAD, "bad.yaml")).ok is False
    assert L.check_file(write(tmp_path, BAD, "bad2.yaml"), tol=0.1).ok is True
    assert L.check_file(p) == L.check_ledger(GOOD)
    with pytest.raises(FileNotFoundError):
        L.check_file(tmp_path / "missing.yaml")


def test_yaml_roundtrip_types(tmp_path):
    """YAML 이 만든 int/float/null/들여쓰기 형태 그대로 읽는다."""
    text = (
        "schema: buc.number-ledger/v1\n"
        "tolerance: 0.01\n"
        "owners: [pipeline, bench]\n"
        "entries:\n"
        "  - {id: a, value: 179, unit: TFU, owner: pipeline}\n"
        "  - id: b\n    value: 10.5\n    unit: TFU\n    owner: pipeline\n    formula: null\n"
        "  - id: s\n    value: 200\n    unit: TFU\n    owner: bench\n"
        "    formula: \"a + 2*b\"\n    inputs: [a, b]\n"
    )
    p = tmp_path / "l.yaml"
    p.write_text(text, encoding="utf-8")
    r = load().check_file(p)
    assert r.ok and r.checked == 3 and r.computed["s"] == pytest.approx(200.0)


def _run(args, cwd):
    return subprocess.run([sys.executable, str(TOOL), *args], cwd=str(cwd), capture_output=True, text=True,
                          timeout=30, env={"PYTHONIOENCODING": "utf-8", "PATH": ""} | _site_env())


def _site_env():
    import os
    keep = ("HOME", "VIRTUAL_ENV", "PYTHONPATH", "LANG", "LC_ALL", "SYSTEMROOT")
    return {k: v for k, v in os.environ.items() if k in keep}


def test_subprocess_end_to_end(tmp_path):
    good = write(tmp_path, GOOD, "good.yaml")
    bad = write(tmp_path, BAD, "bad.yaml")
    r = _run(["--ledger", str(good)], tmp_path)
    assert r.returncode == 0, r.stderr
    assert last_json(r.stdout) == {"ok": True, "checked": 3, "violations": []}
    r = _run(["--ledger", str(bad)], tmp_path)
    assert r.returncode == 1, r.stderr
    j = last_json(r.stdout)
    assert j["ok"] is False and j["violations"][0]["check"] == "mismatch"
    r = _run(["--ledger", str(tmp_path / "none.yaml")], tmp_path)
    assert r.returncode == 2
    r = _run(["--tol", "9"], tmp_path)
    assert r.returncode == 2


def test_default_ledger_resolved_from_repo_root_not_cwd(tmp_path):
    """--ledger 없이 빈 cwd 에서 돌려도 저장소 루트의 docs/number-ledger.yaml 을 본다."""
    L = load()
    r = _run([], tmp_path)
    if (ROOT / L.DEFAULT_LEDGER).exists():
        assert r.returncode in (0, 1)
        assert last_json(r.stdout)["checked"] >= 1
    else:
        assert r.returncode == 2, (r.stdout, r.stderr)
