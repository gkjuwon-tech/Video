"""스탬프 멱등·무효화, 잠금, 비밀 위생."""
import fcntl
import os
import shutil

from run05_helpers import PROBE_5090, TOKEN


def test_rerun_is_noop_except_probe(box):
    r1 = box.run()
    assert r1.code == 0, r1.dump()
    r2 = box.run()
    assert r2.code == 0, r2.dump()
    assert r2.keys == ["probe"], r2.dump()
    j = r2.result
    assert j["deps"]["status"] == "cached"
    assert j["prefetch"]["status"] == "cached"
    assert [m["status"] for m in j["prefetch"]["models"]] == ["cached", "cached"]
    assert j["gsplat"]["status"] == "cached"
    assert j["ok"] is True


def test_torch_change_invalidates_deps_and_gsplat_only(box):
    assert box.run().code == 0
    probe = dict(PROBE_5090, torch="2.9.2+cu128")
    r = box.run(probe=probe)
    assert r.code == 0, r.dump()
    assert r.keys == ["probe", "deps", "gsplat_install", "gsplat_jit"], r.dump()
    cons = [c["constraints"] for c in r.calls if c["constraints"]]
    assert cons and all("torch==2.9.2+cu128" in x.splitlines() for x in cons)


def test_cap_change_invalidates_gsplat(box):
    assert box.run().code == 0
    probe = dict(PROBE_5090, cap=[10, 0], arch_list=["sm_90", "sm_100", "sm_120"])
    r = box.run(probe=probe)
    assert r.code == 0, r.dump()
    assert r.keys == ["probe", "gsplat_install", "gsplat_jit"], r.dump()
    assert r.of("gsplat_jit")[0]["env"]["TORCH_CUDA_ARCH_LIST"] == "10.0"


def test_empty_extension_cache_rebuilds_gsplat(box):
    assert box.run().code == 0
    ext = box.root / "torch_extensions"
    assert ext.is_dir() and any(ext.iterdir())
    shutil.rmtree(ext)
    ext.mkdir()
    r = box.run()
    assert r.code == 0, r.dump()
    assert r.keys == ["probe", "gsplat_install", "gsplat_jit"], r.dump()


def test_force_redoes_everything(box):
    assert box.run().code == 0
    r = box.run(extra_env={"BUC_FORCE": "1"})
    assert r.code == 0, r.dump()
    assert r.keys == ["probe", "deps", "prefetch:org/model-a", "prefetch:org/model-b",
                      "gsplat_install", "gsplat_jit"], r.dump()


def test_failures_are_not_stamped(box):
    r1 = box.run(extra_env={"BUC_RETRIES": "1"},
                 results={"prefetch:org/model-b": [{"rc": 1}], "gsplat_jit": [{"rc": 1}]})
    assert r1.code == 20, r1.dump()
    r2 = box.run(extra_env={"BUC_RETRIES": "1"}, results={})
    assert r2.code == 0, r2.dump()
    assert r2.keys == ["probe", "prefetch:org/model-b", "gsplat_install", "gsplat_jit"], r2.dump()
    assert [m["status"] for m in r2.result["prefetch"]["models"]] == ["cached", "ok"]
    assert r2.result["prefetch"]["status"] == "ok"


def test_new_model_in_list_is_fetched_alone(box):
    assert box.run().code == 0
    r = box.run(extra_env={"BUC_MODELS": "org/model-a org/model-b org/model-c"})
    assert r.code == 0, r.dump()
    assert r.keys == ["probe", "prefetch:org/model-c"], r.dump()


def test_lock_held_exits_75(box):
    box.root.mkdir(parents=True, exist_ok=True)
    fd = os.open(box.root / "bootstrap.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = box.run(timeout=20)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert r.code == 75, r.dump()
    assert "BOOT_LOCKED" in r.out
    assert r.keys == []
    r2 = box.run()
    assert r2.code == 0, r2.dump()


def _all_text_under(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            try:
                out.append(open(os.path.join(dp, fn), "rb").read().decode("utf-8", "replace"))
            except OSError:
                pass
    return out


def test_token_never_leaks(box):
    r = box.run(extra_env={"HF_TOKEN": TOKEN},
                results={"prefetch:org/model-a": [{"rc": 1}, {"rc": 0}], "gsplat_jit": [{"rc": 1}, {"rc": 0}]})
    assert r.code == 0, r.dump()
    assert TOKEN not in r.out and TOKEN not in r.err
    for text in _all_text_under(box.root):
        assert TOKEN not in text
    for c in r.calls:
        assert all(TOKEN not in a for a in c.get("argv", []))
    assert all(c["has_token"] for c in r.of("prefetch")), "prefetch 는 환경으로 토큰을 받아야 한다"
    assert r.result["hf_token"] == "yes"
