"""EVL-02: 실제 매니페스트(EVL-01)와의 이음매, CLI 종료 코드, R-EVAL 격리(torch 미적재, import 부작용 없음)."""
import json
import subprocess
import sys

import pytest

from evl02_support import MODULE, ROOT, FakeRender, ab, make_manifest, qa

M = ab()
D1 = "2026-10-10"


# ------------------------------------------------------------------------------------------------ 실제 매니페스트
def test_load_inputs_partial_vs_strict_on_the_real_manifest():
    m = M.load_inputs(partial=True)
    ready = m.ready()
    assert ready and m.hero.status == "ready"  # 히어로(슬롯 1)는 항상 ready
    if len(ready) < 24:
        with pytest.raises(ValueError):  # EVL-01 ManifestError(ValueError): 촬영 전에는 24장 모드가 실패하는 것이 정상
            M.load_inputs()
    else:
        assert len(M.load_inputs().items) == 24


def test_run_on_real_manifest_carries_disclosure_and_matrix(tmp_path):
    m = M.load_inputs(partial=True)
    res = M.run_daily(m, FakeRender(), tmp_path, D1, partial=True, thumb_size=(64, 48))
    rec = res.record
    ready = [x.id for x in m.ready()]
    assert [e["id"] for e in rec["items"]] == sorted(ready)
    assert rec["n_planned_skipped"] == len(m.items) - len(ready)
    assert rec["disclosure"] == m.selection["disclosure"]
    assert rec["matrix"] == qa().coverage(m)["matrix"]
    assert rec["vault_repo"] == m.vault_repo
    for e in rec["items"]:
        job_crop = next(x for x in m.items if x.id == e["id"]).crop_box
        assert job_crop is not None  # 크롭 박스가 있는 항목만 돈다(같은 크롭 파일 규약)


def test_every_job_gets_the_manifest_crop_box_and_a_seed_independent_of_other_items():
    m = M.load_inputs(partial=True)
    jobs = M.plan_jobs(m, base_seed=4, partial=True)
    by_id = {j.id: j for j in jobs}
    for it in m.ready():
        assert tuple(by_id[it.id].crop_box) == tuple(it.crop_box)
        assert by_id[it.id].seed == M.job_seed(4, it.id, "I1")


# ------------------------------------------------------------------------------------------------ CLI
def run_main(args, render=None):
    return M.main([str(a) for a in args], render=render)


def test_cli_success_failure_and_usage_errors(tmp_path, capsys):
    out = tmp_path / "o"
    assert run_main(["--out", out, "--date", D1, "--partial", "--seed", "9", "--tag", "abc"], FakeRender()) == 0
    rec = M.read_record(out / D1)
    assert rec["base_seed"] == 9 and rec["tag"] == "abc" and rec["complete"] and rec["move"] == "I1"
    assert all(e["seed"] == M.job_seed(9, e["id"], "I1") for e in rec["items"])
    assert D1 in capsys.readouterr().out  # 한 줄 요약

    out2 = tmp_path / "o2"  # 항목 하나가 터지면 1 (야간 잡이 아침에 빨간불)
    assert run_main(["--out", out2, "--date", D1, "--partial"], FakeRender(fail={"in01": RuntimeError("x")})) == 1
    assert M.read_record(out2 / D1)["summary"]["failed"] == 1

    out3 = tmp_path / "o3"
    for bad in (["--out", out3, "--date", "2026-99-99", "--partial"],
                ["--out", out3, "--date", D1, "--partial", "--move", "I9"],
                ["--out", out3, "--partial"],                      # --date 누락
                ["--date", D1, "--partial"],                       # --out 누락
                ["--out", out3, "--date", D1, "--partial", "--manifest", tmp_path / "nope.yaml"]):
        assert run_main(bad, FakeRender()) == 2, bad
    assert not out3.exists()  # 인자·매니페스트 오류는 디스크를 건드리지 않는다
    capsys.readouterr()


def test_cli_without_renderer_or_with_unready_manifest_exits_2(tmp_path):
    out = tmp_path / "o"
    assert run_main(["--out", out, "--date", D1, "--partial"]) == 2        # 렌더러 없음
    if len(M.load_inputs(partial=True).ready()) < 24:
        assert run_main(["--out", out, "--date", D1], FakeRender()) == 2   # planned 가 있는데 --partial 없음
    assert not out.exists()


def test_cli_renderer_factory_by_module_attr(tmp_path):
    out = tmp_path / "o"
    assert run_main(["--out", out, "--date", D1, "--partial", "--renderer", "evl02_support:cli_renderer"]) == 0
    assert M.read_record(out / D1)["summary"]["failed"] == 0
    assert run_main(["--out", tmp_path / "p", "--date", D1, "--partial", "--renderer", "no_such_mod_evl02:f"]) == 2


# ------------------------------------------------------------------------------------------------ 격리 · 부작용
def test_import_is_light_and_side_effect_free(tmp_path):
    code = (
        "import importlib.util, sys, json, os\n"
        f"spec = importlib.util.spec_from_file_location('daily_ab_probe', {str(MODULE)!r})\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)\n"
        "bad = sorted(x for x in sys.modules if x.split('.')[0] in ('lpips', 'torchvision', 'cv2', 'tensorflow'))\n"
        "print(json.dumps({'bad': bad, 'cwd': sorted(os.listdir('.'))}))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    info = json.loads(p.stdout.strip().splitlines()[-1])
    assert info["bad"] == []  # 평가 모델은 싣지 않는다(R-EVAL: LPIPS 는 EVL-04 워커 프로세스). torch 는 buc.core.det 가 적재하므로 허용
    assert info["cwd"] == []  # import 부작용 없음


def test_module_source_has_no_forbidden_imports():
    src = MODULE.read_text(encoding="utf-8")
    for needle in ("import torch", "from torch", "import lpips", "import cv2", "hf_", "api_key", "HF_TOKEN"):
        assert needle not in src, needle


def test_public_names_exist_with_documented_constants():
    names = ("SCHEMA_VERSION", "DEFAULT_MOVE", "SEED_TAG", "RECORD_NAME", "SHEET_NAME", "PSNR_CAP_DB", "NO_PREV_RGB",
             "FAILED_RGB", "BG_RGB", "STATUS_RGB", "PAD", "HEADER_H", "LABEL_H", "DailyABError", "Job", "Rendered",
             "PrevDay", "SheetGeometry", "DailyResult", "job_seed", "plan_jobs", "check_sha256", "spine_renderer",
             "read_record", "find_previous", "canonical_digest", "clip_delta", "sheet_geometry", "run_daily",
             "load_inputs", "main")
    for n in names:
        assert hasattr(M, n), n
    assert (M.SCHEMA_VERSION, M.DEFAULT_MOVE, M.SEED_TAG) == (1, "I1", "buc.daily_ab.v1")
    assert (M.RECORD_NAME, M.SHEET_NAME) == ("run.json", "sheet.png")
    assert make_manifest(1).hero.id == "in01"
