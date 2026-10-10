"""TCH-04 import 안전성: diffusers·transformers·CUDA 없이, 부작용(파일 생성) 없이."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_import_has_no_heavy_deps_and_no_side_effects(tmp_path):
    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import buc.teacher.exhaust as X\n"
        "bad = [m for m in ('diffusers', 'transformers', 'huggingface_hub', 'accelerate') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "import torch\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert os.listdir('.') == [], os.listdir('.')\n"
        "print('ok exhaust', hasattr(X, 'ExhaustLedger'))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
    assert "ok exhaust True" in r.stdout


def test_module_declares_all():
    from buc.teacher import exhaust as X
    assert isinstance(X.__all__, (list, tuple))
    for n in ["ExhaustLedger", "KeyParts", "compute_key", "quantize_x0", "dequantize_x0", "allowed_purposes"]:
        assert n in X.__all__
