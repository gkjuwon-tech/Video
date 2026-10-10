"""RUN-08: main() CLI. 가짜 torch·주입 시계·주입 케이스."""
import json

import pytest
from run08_helpers import GATE_PARAMS, Clock, fake_cpu, fake_t4, make_case

SECRET = "hf_" + "D" * 24


def last_json(capsys):
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, out
    return json.loads(lines[0])


def test_main_t4_run_writes_json_and_md(t4, tmp_path, capsys):
    tt = fake_t4()
    clock = Clock(100.0)
    gate, _, _ = make_case(t4, name="gate", clock=clock, dur=0.5, params=GATE_PARAMS)
    other, _, _ = make_case(t4, name="fb", clock=clock, dur=0.25,
                            params={"mode": "fwdbwd", "width": 1024, "height": 1024, "n_prims": 7})
    out = tmp_path / "o" / "r.json"
    rc = t4.main(["--out", str(out), "--repeats", "2", "--warmup", "1", "--created-utc", "2026-10-24T00:00:00Z",
                  "--budget-min", "2"], torch_mod=tt, clock=clock, cases=[gate, other])
    assert rc == 0
    res = last_json(capsys)
    assert res == {"ok": True, "json": str(out), "md": str(out.with_suffix(".md")),
                   "shot_on_t4": True, "gate": "pass"}
    rep = t4.load_report(out)
    assert rep.created_utc == "2026-10-24T00:00:00Z" and rep.budget_s == 120.0
    assert [m.name for m in rep.measurements] == ["gate", "fb"]
    assert rep.device.is_t4 and rep.shot_on_t4
    assert out.with_suffix(".md").read_text(encoding="utf-8") == t4.render_appendix(rep)


def test_main_gate_fail_is_reported_not_hidden(t4, tmp_path, capsys):
    clock = Clock(100.0)
    gate, _, _ = make_case(t4, name="gate", clock=clock, dur=0.9, params=GATE_PARAMS)
    rc = t4.main(["--out", str(tmp_path / "r.json"), "--repeats", "1", "--warmup", "0"],
                 torch_mod=fake_t4(), clock=clock, cases=[gate])
    assert rc == 0                                   # 측정은 성공했다. 미달은 결과이지 오류가 아니다
    res = last_json(capsys)
    assert res["ok"] is True and res["gate"] == "fail" and res["shot_on_t4"] is True


def test_main_without_cuda_reports_unmeasured_and_exits_zero(t4, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = t4.main(["--out", str(out)], torch_mod=fake_cpu())
    assert rc == 0
    res = last_json(capsys)
    assert res["ok"] is True and res["shot_on_t4"] is False and res["gate"] == "unmeasured"
    rep = t4.load_report(out)
    assert len(rep.measurements) == 4                          # 기본 케이스 4개
    assert all(m.status == "unmeasured" and m.reason == "no-cuda" for m in rep.measurements)
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "미측정" in md and md.endswith("\n")
    assert rep.created_utc is not None and rep.created_utc.endswith("Z") and len(rep.created_utc) == 20


def test_main_default_out_is_relative_to_cwd(t4, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert t4.main([], torch_mod=fake_cpu()) == 0
    res = last_json(capsys)
    assert (tmp_path / "work" / "t4" / "shot_on_t4.json").is_file()
    assert (tmp_path / "work" / "t4" / "shot_on_t4.md").is_file()
    assert res["json"].endswith("shot_on_t4.json")


def test_main_allow_cpu_real_tiny_run(t4, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA 호스트에서는 CPU 시나리오가 아니다")
    out = tmp_path / "r.json"
    rc = t4.main(["--out", str(out), "--allow-cpu", "--width", "64", "--height", "48", "--prims", "40,80",
                  "--train-size", "32", "--train-prims", "40", "--repeats", "1", "--warmup", "0"])
    assert rc == 0
    res = last_json(capsys)
    assert res["shot_on_t4"] is False and res["gate"] == "unmeasured"
    rep = t4.load_report(out)
    assert [m.name for m in rep.measurements] == ["splat_fwd_64x48_40", "splat_fwd_64x48_80",
                                                  "splat_fwdbwd_32x32_40"]
    assert [m.status for m in rep.measurements] == ["ok", "ok", "ok"]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "> 미측정: 이 장치는 T4가 아니다 (cpu). T4 수치는 없다." in md


@pytest.mark.parametrize("argv", [
    ["--repeats", "0"], ["--repeats", "x"], ["--warmup", "-1"], ["--prims", "a,b"], ["--prims", "0"],
    ["--prims", "5,5"], ["--width", "x"], ["--height", "0"], ["--budget-min", "-1"], ["--budget-min", "nan"],
    ["--bogus"],
])
def test_main_rejects_bad_args(t4, tmp_path, argv):
    with pytest.raises(SystemExit) as ei:
        t4.main(["--out", str(tmp_path / "r.json")] + argv, torch_mod=fake_cpu())
    assert ei.value.code == 2
    assert not (tmp_path / "r.json").exists()


def test_main_write_failure_returns_1(t4, tmp_path, capsys):
    blocker = tmp_path / "f"
    blocker.write_text("x", encoding="utf-8")
    rc = t4.main(["--out", str(blocker / "r.json")], torch_mod=fake_cpu())
    assert rc == 1
    res = last_json(capsys)
    assert res["ok"] is False and isinstance(res["why"], str) and res["why"]


def test_main_never_prints_or_stores_secrets(t4, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("VAST_API_KEY", "k" * 40)
    clock = Clock(100.0)

    def behavior(i):
        raise RuntimeError(f"auth failed token={SECRET}")

    bad, _, _ = make_case(t4, name="bad", clock=clock, behavior=behavior)
    out = tmp_path / "r.json"
    rc = t4.main(["--out", str(out), "--repeats", "1", "--warmup", "0"], torch_mod=fake_t4(), clock=clock,
                 cases=[bad])
    assert rc == 0
    captured = capsys.readouterr()
    blob = captured.out + captured.err + out.read_text(encoding="utf-8") + out.with_suffix(".md").read_text(
        encoding="utf-8")
    assert SECRET not in blob and "k" * 40 not in blob
    assert t4.load_report(out).measurements[0].status == "error"
