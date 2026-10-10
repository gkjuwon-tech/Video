"""정상 경로: 호출 순서, 단계별 환경, torch 제약, 결과 JSON, 마커, env.sh."""
import json
import os
import re
import subprocess

from run05_helpers import SCRIPT, is_pip_install

DEFAULT_PKGS = ["diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_transfer",
                "sentencepiece", "ninja", "jaxtyping", "rich"]
TORCH_TOKEN = re.compile(r"^(torch|torchvision|torchaudio)(\[.*\])?([=<>!~].*)?$")


def test_syntax_and_no_xtrace():
    assert SCRIPT.exists(), f"없음: {SCRIPT}"
    p = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    src = SCRIPT.read_text()
    assert not re.search(r"^\s*set\s+-[a-wyz]*x", src, re.M), "set -x 는 비밀을 흘린다"
    assert "xtrace" not in src
    assert re.search(r"^\s*set\s+-[a-z]*u", src, re.M), "set -u 필요"


def test_happy_path_sequence_and_env(box):
    r = box.run()
    assert r.code == 0, r.dump()
    assert r.keys == ["probe", "deps", "prefetch:org/model-a", "prefetch:org/model-b",
                      "gsplat_install", "gsplat_jit"], r.dump()
    root = str(box.root)
    for c in r.of("prefetch"):
        assert c["env"]["HF_HOME"] == os.path.join(root, "hf")
        assert c["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert [c["model"] for c in r.of("prefetch")] == ["org/model-a", "org/model-b"]
    jobs = str(min(len(os.sched_getaffinity(0)), 16))
    for c in r.of("gsplat_install") + r.of("gsplat_jit"):
        assert c["env"]["TORCH_CUDA_ARCH_LIST"] == "12.0"
        assert c["env"]["TORCH_EXTENSIONS_DIR"] == os.path.join(root, "torch_extensions")
        assert c["env"]["MAX_JOBS"] == jobs
    inst = r.of("gsplat_install")[0]
    assert is_pip_install(inst) and "gsplat==1.5.3" in inst["argv"], inst["argv"]
    deps = r.of("deps")[0]
    assert is_pip_install(deps)
    for pkg in DEFAULT_PKGS:
        assert pkg in deps["argv"], (pkg, deps["argv"])
    assert not [c for c in r.calls if c["key"].startswith("CMD:")], "curl/wget/git 직접 호출 금지"


def test_torch_is_pinned_in_every_pip_call(box):
    r = box.run()
    assert r.code == 0, r.dump()
    pips = [c for c in r.calls if c["step"] and is_pip_install(c)]
    assert len(pips) == 2
    for c in pips:
        assert "-c" in c["argv"], c["argv"]
        assert c["constraints"] is not None, "제약 파일이 실제로 있어야 한다"
        lines = [x.strip() for x in c["constraints"].splitlines()]
        assert "torch==2.9.1+cu128" in lines, lines
        for tok in c["argv"]:
            assert not TORCH_TOKEN.match(tok), f"torch 재설치 위험: {tok}"
    assert (box.root / "constraints.txt").exists()


def test_result_json_and_markers(box):
    r = box.run()
    assert r.code == 0, r.dump()
    j = r.result
    assert j["v"] == 1 and j["ok"] is True and j["code"] == 0
    assert j["hf_token"] == "no"
    assert j["paths"] == {"root": str(box.root), "hf_home": str(box.root / "hf"),
                          "torch_extensions_dir": str(box.root / "torch_extensions")}
    e = j["env"]
    assert e["status"] == "ok" and e["reason"] is None
    assert e["cap"] == [12, 0] and e["arch"] == "12.0"
    assert e["torch"] == "2.9.1+cu128" and e["cuda"] == "12.8" and "5090" in e["gpu"]
    assert j["deps"]["status"] == "ok"
    assert j["prefetch"]["status"] == "ok"
    assert j["prefetch"]["models"] == [{"repo": "org/model-a", "status": "ok", "attempts": 1},
                                       {"repo": "org/model-b", "status": "ok", "attempts": 1}]
    g = j["gsplat"]
    assert g["status"] == "ok" and g["version"] == "1.5.3" and g["source"] == "pypi" and g["attempts"] == ["pypi"]
    assert j["ts_start"] <= j["ts_end"]
    leftovers = [p.name for p in box.root.iterdir() if p.name.startswith("bootstrap.json") and p.name != "bootstrap.json"]
    assert not leftovers, leftovers
    lines = r.out.splitlines()
    boot = [x for x in lines if x.startswith("BOOT_")]
    assert all(len(x) <= 400 for x in boot)
    assert any(x.startswith("BOOT_START") for x in boot)
    env_line = [x for x in boot if x.startswith("BOOT_ENV")]
    assert env_line and "arch=12.0" in env_line[0]
    res = [x for x in boot if x.startswith("BOOT_RESULT")]
    assert res and "ok=1" in res[-1] and "code=0" in res[-1]
    log = box.root / "logs" / "bootstrap.log"
    assert log.exists() and "BOOT_START" in log.read_text()


def test_env_sh_sources_cleanly(box):
    r = box.run(extra_env={"HF_TOKEN": "hf_dummyvalue000"})
    assert r.code == 0, r.dump()
    envsh = box.root / "env.sh"
    assert envsh.exists()
    p = subprocess.run(["bash", "-c", f'unset HF_HOME TORCH_EXTENSIONS_DIR TORCH_CUDA_ARCH_LIST HF_TOKEN; '
                                      f'source "{envsh}"; python3 -c "import json,os;print(json.dumps({{k: os.environ.get(k) '
                                      f'for k in (\'HF_HOME\',\'TORCH_EXTENSIONS_DIR\',\'TORCH_CUDA_ARCH_LIST\','
                                      f'\'HF_HUB_ENABLE_HF_TRANSFER\',\'HF_TOKEN\')}}))"'],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert got == {"HF_HOME": str(box.root / "hf"), "TORCH_EXTENSIONS_DIR": str(box.root / "torch_extensions"),
                   "TORCH_CUDA_ARCH_LIST": "12.0", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_TOKEN": None}
    assert "hf_dummyvalue000" not in envsh.read_text()


def test_presets_are_preserved(box, tmp_path):
    hf = tmp_path / "myhf"
    ext = tmp_path / "myext"
    r = box.run(extra_env={"HF_HOME": str(hf), "TORCH_EXTENSIONS_DIR": str(ext),
                           "TORCH_CUDA_ARCH_LIST": "12.0+PTX", "MAX_JOBS": "3"})
    assert r.code == 0, r.dump()
    for c in r.of("prefetch"):
        assert c["env"]["HF_HOME"] == str(hf)
    for c in r.of("gsplat_install") + r.of("gsplat_jit"):
        assert c["env"]["TORCH_CUDA_ARCH_LIST"] == "12.0+PTX"
        assert c["env"]["TORCH_EXTENSIONS_DIR"] == str(ext)
        assert c["env"]["MAX_JOBS"] == "3"
    assert r.result["paths"]["hf_home"] == str(hf)
    assert r.result["paths"]["torch_extensions_dir"] == str(ext)


def test_non_blackwell_gpu_arch_derived(box):
    probe = {"cuda_available": True, "torch": "2.9.1+cu121", "cuda": "12.1", "gpu": "NVIDIA GeForce RTX 4090",
             "cap": [8, 9], "arch_list": ["sm_80", "sm_86", "sm_89", "sm_90"], "bf16": True}
    r = box.run(probe=probe)
    assert r.code == 0, r.dump()
    assert r.result["env"]["arch"] == "8.9"
    for c in r.of("gsplat_jit"):
        assert c["env"]["TORCH_CUDA_ARCH_LIST"] == "8.9"
