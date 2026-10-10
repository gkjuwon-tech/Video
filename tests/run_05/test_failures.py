"""진단 거절, 단계별 종료 코드, 재시도, gsplat 대체 경로, timeout, skip."""
import time

import pytest
from run05_helpers import PROBE_5090, is_pip_install


def _probe(**kw):
    d = dict(PROBE_5090)
    d.update(kw)
    return d


@pytest.mark.parametrize("probe,results,reason", [
    (None, {"probe": [{"rc": 1, "out": ""}]}, "probe_failed"),
    (None, {"probe": [{"rc": 0, "out": "nothing useful here"}]}, "probe_failed"),
    (None, {"probe": [{"rc": 0, "out": "PROBE {not json"}]}, "probe_failed"),
    (_probe(cuda_available=False), {}, "no_cuda"),
    (_probe(cuda="12.6"), {}, "cuda_too_old_for_sm120"),
    (_probe(arch_list=["sm_75", "sm_80", "sm_86", "sm_90"]), {}, "torch_lacks_sm_120"),
])
def test_env_rejections(box, probe, results, reason):
    r = box.run(probe=probe, results=results)
    assert r.code == 10, r.dump()
    assert r.keys == ["probe"], r.dump()
    j = r.result
    assert j is not None, "실패해도 bootstrap.json 은 남는다"
    assert j["ok"] is False and j["code"] == 10
    assert j["env"]["status"] == "fail" and j["env"]["reason"] == reason
    assert j["deps"]["status"] == "skip"
    assert j["prefetch"]["status"] == "skip"
    assert j["gsplat"]["status"] == "skip"
    assert not (box.root / "env.sh").exists()
    res = [x for x in r.out.splitlines() if x.startswith("BOOT_RESULT")]
    assert res and "ok=0" in res[-1] and "code=10" in res[-1]


def test_cuda_version_compared_numerically(box):
    r = box.run(probe=_probe(cuda="12.10"))
    assert r.code == 0, r.dump()
    assert r.result["env"]["status"] == "ok"


def test_deps_failure_stops(box):
    r = box.run(results={"deps": [{"rc": 1}]})
    assert r.code == 11, r.dump()
    assert r.keys == ["probe", "deps"]
    j = r.result
    assert j["deps"]["status"] == "fail"
    assert j["prefetch"]["status"] == "skip" and j["gsplat"]["status"] == "skip"
    assert j["ok"] is False and j["code"] == 11


def test_prefetch_retry_then_success_and_permanent_failure(box):
    r = box.run(extra_env={"BUC_RETRIES": "3"},
                results={"prefetch:org/model-a": [{"rc": 1}, {"rc": 1}, {"rc": 0}],
                         "prefetch:org/model-b": [{"rc": 2}]})
    assert r.code == 20, r.dump()
    assert r.keys.count("prefetch:org/model-a") == 3
    assert r.keys.count("prefetch:org/model-b") == 3
    assert r.keys[-2:] == ["gsplat_install", "gsplat_jit"], "prefetch 실패에도 gsplat 은 계속"
    j = r.result
    assert j["prefetch"]["status"] == "fail"
    assert j["prefetch"]["models"] == [{"repo": "org/model-a", "status": "ok", "attempts": 3},
                                       {"repo": "org/model-b", "status": "fail", "attempts": 3}]
    assert j["gsplat"]["status"] == "ok"
    assert j["ok"] is False and j["code"] == 20


def test_prefetch_retries_respected(box):
    r = box.run(extra_env={"BUC_RETRIES": "1"}, results={"prefetch:org/model-a": [{"rc": 1}, {"rc": 0}]})
    assert r.code == 20, r.dump()
    assert r.keys.count("prefetch:org/model-a") == 1


def test_gsplat_jit_failure_falls_back_to_git_source(box):
    r = box.run(results={"gsplat_jit": [{"rc": 1}, {"rc": 0}]})
    assert r.code == 0, r.dump()
    assert r.keys[-4:] == ["gsplat_install", "gsplat_jit", "gsplat_install", "gsplat_jit"]
    first, second = r.of("gsplat_install")
    assert "gsplat==1.5.3" in first["argv"]
    assert is_pip_install(second)
    assert "--no-build-isolation" in second["argv"]
    assert "git+https://github.com/nerfstudio-project/gsplat@v1.5.3" in second["argv"], second["argv"]
    for c in r.of("gsplat_install"):
        assert c["env"]["TORCH_CUDA_ARCH_LIST"] == "12.0"
    g = r.result["gsplat"]
    assert g["status"] == "ok" and g["source"] == "git" and g["attempts"] == ["pypi", "git"]


def test_gsplat_install_failure_skips_jit_then_git(box):
    r = box.run(results={"gsplat_install": [{"rc": 1}, {"rc": 0}]})
    assert r.code == 0, r.dump()
    assert r.keys[-3:] == ["gsplat_install", "gsplat_install", "gsplat_jit"]
    assert r.result["gsplat"]["source"] == "git"


def test_gsplat_total_failure(box):
    r = box.run(results={"gsplat_jit": [{"rc": 1}]})
    assert r.code == 30, r.dump()
    g = r.result["gsplat"]
    assert g["status"] == "fail" and g["attempts"] == ["pypi", "git"]
    assert r.result["ok"] is False


def test_gsplat_jit_timeout(box):
    t = time.monotonic()
    r = box.run(extra_env={"BUC_GSPLAT_TIMEOUT": "1"}, results={"gsplat_jit": [{"rc": 0, "sleep": 6}]})
    dt = time.monotonic() - t
    assert r.code == 30, r.dump()
    assert dt < 11, f"timeout 이 걸려야 한다: {dt:.1f}s"
    assert r.result["gsplat"]["status"] == "fail"


def test_prefetch_failure_and_gsplat_failure_reports_first_code(box):
    r = box.run(extra_env={"BUC_RETRIES": "1"},
                results={"prefetch:org/model-a": [{"rc": 1}], "gsplat_jit": [{"rc": 1}]})
    assert r.code == 20, r.dump()
    assert r.result["gsplat"]["status"] == "fail"


def test_skip_flags(box):
    r = box.run(extra_env={"BUC_SKIP_DEPS": "1", "BUC_SKIP_PREFETCH": "1", "BUC_SKIP_GSPLAT": "1"})
    assert r.code == 0, r.dump()
    assert r.keys == ["probe"]
    j = r.result
    assert (j["deps"]["status"], j["prefetch"]["status"], j["gsplat"]["status"]) == ("skip", "skip", "skip")
    assert j["ok"] is True
