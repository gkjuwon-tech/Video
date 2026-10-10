"""RUN-07 쓰기·CLI·읽기 전용·import 위생."""
import hashlib
import json
import subprocess
import sys

import pytest
from PIL import Image
from run07_helpers import MODULE_PATH, NOW, ROOT, SECRET, make_items

from buc.core.ledger import open_ledger

STAMP = "20261011T0030Z"


def _write_backlog(tmp_path):
    p = tmp_path / "backlog.json"
    p.write_text(json.dumps({"version": 1, "items": make_items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _file_ledger(tmp_path):
    from run07_helpers import fill_ledger
    p = tmp_path / "led" / "ledger.sqlite"
    led = open_ledger(p)
    fill_ledger(led)
    led.close()
    return p


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_write_report_names_and_contents(hr, data, tmp_path):
    out = tmp_path / "reports"
    png, md = hr.write_report(out, data)
    assert png == out / f"hourly-{STAMP}.png" and md == out / f"hourly-{STAMP}.md"
    assert md.read_text(encoding="utf-8") == hr.summarize_ko(data)
    assert Image.open(png).size == hr.CANVAS
    assert sorted(f.name for f in out.iterdir()) == sorted([png.name, md.name])


def test_write_report_is_idempotent_bytes(hr, data, tmp_path):
    out = tmp_path / "r"
    png, md = hr.write_report(out, data)
    h1 = (_sha(png), _sha(md))
    hr.write_report(out, data)
    assert (_sha(png), _sha(md)) == h1


def test_cli_success(hr, tmp_path, capsys, monkeypatch):
    bl, led, out = _write_backlog(tmp_path), _file_ledger(tmp_path), tmp_path / "out"
    monkeypatch.setenv("BUC_BUDGET_DAILY_USD", "10")
    monkeypatch.setenv("BUC_BUDGET_TOTAL_USD", "35")
    rc = hr.main(["--out", str(out), "--backlog", str(bl), "--ledger", str(led), "--now", str(NOW)])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and res["ok"] is True
    assert res["png"].endswith(f"hourly-{STAMP}.png") and res["md"].endswith(f"hourly-{STAMP}.md")
    text = open(res["md"], encoding="utf-8").read()
    assert "지출(오늘 UTC): $2.00 / 일 캡 $10.00 (20%)" in text
    assert "진행: 완료 3/7 (42.9%)" in text
    assert SECRET not in text
    assert Image.open(res["png"]).size == hr.CANVAS


def test_cli_options_hours_and_window(hr, tmp_path, capsys):
    bl, led, out = _write_backlog(tmp_path), _file_ledger(tmp_path), tmp_path / "out"
    rc = hr.main(["--out", str(out), "--backlog", str(bl), "--ledger", str(led), "--now", str(NOW),
                  "--window-h", "2", "--hours", "12"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0
    assert "지난 2시간 완료: 3건 — A, B, C" in open(res["md"], encoding="utf-8").read()


def test_cli_missing_ledger_is_reported_not_created(hr, tmp_path, capsys):
    bl, out = _write_backlog(tmp_path), tmp_path / "out"
    missing = tmp_path / "nope" / "ledger.sqlite"
    rc = hr.main(["--out", str(out), "--backlog", str(bl), "--ledger", str(missing), "--now", str(NOW)])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and res["ok"] is True
    assert "지출: 장부 없음 — 기록을 읽지 못했다" in open(res["md"], encoding="utf-8").read()
    assert not missing.exists() and not missing.parent.exists()


@pytest.mark.parametrize("content", [None, "{broken", "[]"])
def test_cli_bad_backlog_fails_without_inventing(hr, tmp_path, capsys, content):
    bl, out = tmp_path / "backlog.json", tmp_path / "out"
    if content is not None:
        bl.write_text(content, encoding="utf-8")
    rc = hr.main(["--out", str(out), "--backlog", str(bl), "--ledger", str(tmp_path / "l.sqlite"), "--now", str(NOW)])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 1 and res["ok"] is False and res["why"]
    assert not out.exists() or list(out.iterdir()) == []


def test_cli_bad_args_exit_2(hr):
    with pytest.raises(SystemExit) as e:
        hr.main(["--bogus"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        hr.main(["--now", "abc"])
    assert e.value.code == 2


def test_cli_is_read_only(hr, tmp_path, capsys):
    bl, led, out = _write_backlog(tmp_path), _file_ledger(tmp_path), tmp_path / "out"
    b0 = bl.read_bytes()
    with open_ledger(led) as lg:
        n0 = lg.count()
    assert hr.main(["--out", str(out), "--backlog", str(bl), "--ledger", str(led), "--now", str(NOW)]) == 0
    capsys.readouterr()
    assert bl.read_bytes() == b0
    with open_ledger(led) as lg:
        assert lg.count() == n0


def test_script_runs_as_subprocess(tmp_path):
    bl, led, out = _write_backlog(tmp_path), _file_ledger(tmp_path), tmp_path / "out"
    p = subprocess.run([sys.executable, str(MODULE_PATH), "--out", str(out), "--backlog", str(bl), "--ledger", str(led),
                        "--now", str(NOW)], cwd=tmp_path, capture_output=True, text=True, timeout=50,
                       env={"PATH": "/usr/bin:/bin:/usr/local/bin"})      # cwd·PYTHONPATH 도움 없이 레포 루트를 스스로 찾아야 한다
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["ok"] is True and (out / f"hourly-{STAMP}.md").exists()


def test_import_hygiene_no_heavy_modules_no_side_effects(tmp_path):
    code = (
        "import importlib.util, sys, os\n"
        f"spec = importlib.util.spec_from_file_location('h', {str(MODULE_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['h'] = m; spec.loader.exec_module(m)\n"
        "bad = [k for k in ('torch', 'numpy', 'matplotlib') if k in sys.modules]\n"
        "print(bad, sorted(os.listdir('.')))\n"
    )
    work = tmp_path / "cwd"
    work.mkdir()
    p = subprocess.run([sys.executable, "-c", code], cwd=work, capture_output=True, text=True, timeout=50,
                       env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "[] []"


def test_source_does_not_import_forbidden_packages():
    import ast
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert not mods & {"matplotlib", "torch", "numpy", "scipy"}
