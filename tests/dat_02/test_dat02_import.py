"""import 부작용 없음: CUDA·diffusers·transformers 를 끌어오지 않는다."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_import_is_light():
    code = ("import sys, torch\n"
            "import buc.data.realify as R\n"
            "bad = [m for m in ('diffusers', 'transformers') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "assert not torch.cuda.is_initialized()\n"
            "assert callable(R.realify) and callable(R.write_results) and callable(R.item_from_synth)\n")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
