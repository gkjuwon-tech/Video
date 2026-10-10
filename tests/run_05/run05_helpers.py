"""RUN-05 테스트 도우미: bootstrap.sh 를 스텁 파이썬·스텁 PATH 로 돌리고 호출 기록과 결과를 돌려준다."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "ops" / "gpu" / "bootstrap.sh"
STUB = Path(__file__).resolve().parent / "run05_stub.py"
TOKEN = "hf_RUN05fakeSECRETtokenZq9xW"

PROBE_5090 = {"cuda_available": True, "torch": "2.9.1+cu128", "cuda": "12.8", "gpu": "NVIDIA GeForce RTX 5090",
              "cap": [12, 0], "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], "bf16": True}

STRIP = ("HF_HOME", "TORCH_EXTENSIONS_DIR", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "HF_TOKEN",
         "HF_HUB_ENABLE_HF_TRANSFER", "BOOT_STEP", "BOOT_MODEL")


def _exe(path: Path, body: str):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Box:
    """하나의 가짜 GPU 서버. 같은 Box 로 여러 번 돌리면 캐시(BUC_ROOT)가 이어진다."""

    def __init__(self, tmp: Path):
        self.tmp = Path(tmp)
        self.root = self.tmp / "buc"
        self.bin = self.tmp / "bin"
        self.bin.mkdir(parents=True, exist_ok=True)
        self.log = self.tmp / "stub_log.jsonl"
        self.spec_path = self.tmp / "spec.json"
        self.py = self.bin / "gpu-python"
        _exe(self.py, f'#!/bin/sh\nexec "{sys.executable}" "{STUB}" "$@"\n')
        for name in ("curl", "wget", "git"):
            _exe(self.bin / name,
                 f'#!/bin/sh\necho "{{\\"key\\": \\"CMD:{name}\\"}}" >> "$STUB_LOG"\nexit 1\n')
        _exe(self.bin / "nvidia-smi", '#!/bin/sh\necho "NVIDIA GeForce RTX 5090, 32607 MiB, 575.57"\n')
        self.spec = {"probe": dict(PROBE_5090), "results": {}}

    def run(self, extra_env=None, timeout=50, probe=None, results=None):
        if probe is not None:
            self.spec["probe"] = probe
        if results is not None:
            self.spec["results"] = results
        self.spec_path.write_text(json.dumps(self.spec))
        start = self.log.stat().st_size if self.log.exists() else 0
        env = {k: v for k, v in os.environ.items() if k not in STRIP and not k.startswith("BUC_")}
        env.update({
            "PATH": f"{self.bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(self.tmp / "home"),
            "BUC_ROOT": str(self.root),
            "BUC_PYTHON": str(self.py),
            "BUC_MODELS": "org/model-a org/model-b",
            "BUC_GSPLAT_VERSION": "1.5.3",
            "BUC_RETRY_SLEEP": "0",
            "STUB_SPEC": str(self.spec_path),
            "STUB_LOG": str(self.log),
        })
        (self.tmp / "home").mkdir(exist_ok=True)
        if extra_env:
            for k, v in extra_env.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = v
        p = subprocess.run(["bash", str(SCRIPT)], env=env, cwd=str(self.tmp), stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout)
        calls = []
        if self.log.exists():
            with open(self.log) as f:
                f.seek(start)
                calls = [json.loads(x) for x in f.read().splitlines() if x.strip()]
        res_path = self.root / "bootstrap.json"
        result = json.loads(res_path.read_text()) if res_path.exists() else None
        return Run(p, calls, result)


class Run:
    def __init__(self, proc, calls, result):
        self.proc, self.calls, self.result = proc, calls, result
        self.code, self.out, self.err = proc.returncode, proc.stdout, proc.stderr

    @property
    def keys(self):
        return [c["key"] for c in self.calls]

    def of(self, step):
        return [c for c in self.calls if c["step"] == step]

    def dump(self):
        return f"code={self.code}\nkeys={self.keys}\nSTDOUT:\n{self.out[-3000:]}\nSTDERR:\n{self.err[-3000:]}"


def is_pip_install(c):
    a = c["argv"]
    return len(a) >= 3 and a[0] == "-m" and a[1] == "pip" and "install" in a[2:]
