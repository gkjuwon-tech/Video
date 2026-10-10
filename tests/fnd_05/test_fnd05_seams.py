"""이음매: F-M1 식 시나리오, import 위생, 실제 원장(있으면)."""
import subprocess
import sys

import pytest
import yaml

from fnd05_helpers import ROOT, TOOL, D, E, checks, doc, load

PARTS = ["bucket.world", "bucket.teacher", "bucket.data", "bucket.bench", "bucket.misc"]


def ninety_day(reserve):
    """90일 버킷 합계 4,320 (벤치 정본 예비 635)."""
    return doc(
        E("bucket.world", 1200, "T4-h", "pipeline"), E("bucket.teacher", 900, "T4-h", "catalog"),
        E("bucket.data", 750, "T4-h", "catalog"), E("bucket.bench", 510, "T4-h", "bench"),
        E("bucket.misc", 325, "T4-h", "catalog"),
        E("bucket.reserve", reserve, "T4-h", "bench"),
        D("bucket.total", 4320, "sum(" + ",".join(PARTS) + ", bucket.reserve)", PARTS + ["bucket.reserve"],
          "T4-h", "bench"),
    )


def test_bucket_total_4320_reproduced():
    r = load().check_ledger(ninety_day(635))
    assert r.ok and r.computed["bucket.total"] == pytest.approx(4320.0)


@pytest.mark.parametrize("reserve,ok", [
    (635, True), (613, True),            # 합 4298, 0.51% : 통과
    (598, True),                         # 합 4283, 0.86% : 통과
    (590, False),                        # 합 4275, 1.05% : 실패(원 메모의 예비 590-610 오류가 잡힌다)
    (560, False), (700, False),
])
def test_reserve_drift_detected_beyond_one_percent(reserve, ok):
    r = load().check_ledger(ninety_day(reserve))
    assert r.ok is ok
    if not ok:
        assert [v.id for v in checks(r, "mismatch")] == ["bucket.total"]


def test_double_counting_style_world_is_caught():
    """스타일 세계 40 T4-h 이중 계상: 합계가 같은 40 을 두 번 센 값이면 잡힌다."""
    parts = ["l2.photo", "l2.style", "o4.total"]
    d = doc(E("l2.photo", 22, "T4-h"), E("l2.style", 40, "T4-h"), E("o4.extra", 42, "T4-h"),
            D("o4.total", 82, "o4.extra + l2.style", ["o4.extra", "l2.style"], "T4-h"),
            D("grand", 144, "sum(" + ",".join(parts) + ")", parts, "T4-h"))
    r = load().check_ledger(d)       # 22 + 40 + 82 = 144 이면 정상, 40 이 두 번 들어가 있다는 것은 사람이 정할 일
    assert r.ok and r.computed["grand"] == pytest.approx(144.0)
    d["entries"][-1]["value"] = 184          # 40 을 한 번 더 센 값
    assert not load().check_ledger(d).ok


def test_roundtrip_through_yaml_equals_in_memory(tmp_path):
    d = ninety_day(635)
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    assert load().check_file(p) == load().check_ledger(d)


def test_module_imports_only_stdlib_and_yaml():
    code = (
        "import importlib.util,sys\n"
        f"s=importlib.util.spec_from_file_location('lcx',{str(TOOL)!r})\n"
        "m=importlib.util.module_from_spec(s); sys.modules['lcx']=m; s.loader.exec_module(m)\n"
        "bad=[n for n in ('buc','torch','numpy','scipy','PIL') if n in sys.modules]\n"
        "print('BAD:'+','.join(bad))\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "BAD:", r.stdout


def test_import_has_no_side_effects(tmp_path):
    before = sorted(p.name for p in ROOT.iterdir())
    load()
    assert sorted(p.name for p in ROOT.iterdir()) == before


def test_real_ledger_if_present_is_green():
    p = ROOT / load().DEFAULT_LEDGER
    if not p.exists():
        pytest.skip("docs/number-ledger.yaml 은 카탈로그 렌즈가 채운다(이 단위 소유 아님)")
    r = load().check_file(p)
    assert r.ok, [(v.check, v.id, v.why) for v in r.violations]
