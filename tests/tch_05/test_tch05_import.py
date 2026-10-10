"""TCH-05 import 위생: 부작용 없음, 무거운 의존 없음, 소유 경로의 공개 이름."""
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_import_has_no_heavy_deps_and_no_side_effects():
    code = (
        "import sys, random, numpy as np, torch\n"
        "r0 = random.getstate(); n0 = np.random.get_state()[1].copy(); t0 = torch.get_rng_state().clone()\n"
        "import buc.teacher.decision_curve as m\n"
        "assert random.getstate() == r0 and (np.random.get_state()[1] == n0).all() and torch.equal(torch.get_rng_state(), t0)\n"
        "for bad in ('diffusers', 'transformers', 'huggingface_hub'):\n"
        "    assert bad not in sys.modules, bad\n"
        "assert not torch.cuda.is_initialized()\n"
        "print('ok')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stdout + p.stderr
