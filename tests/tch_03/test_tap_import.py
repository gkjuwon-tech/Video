"""TCH-03 import 부작용 없음·공개 이름."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

NAMES = ["TAP_LAYERS", "PCA_DIM", "INT8_MAX", "TapError", "TapSpecError", "TapBasisError", "TapNumericError",
         "PCABasis", "fit_pca_basis", "TapSet", "TapRing", "TapBus"]


def test_public_names():
    from buc.teacher import taps
    for n in NAMES:
        assert hasattr(taps, n), n
    assert issubclass(taps.TapSpecError, ValueError) and issubclass(taps.TapNumericError, FloatingPointError)
    assert all(issubclass(getattr(taps, n), RuntimeError)
               for n in ("TapError", "TapSpecError", "TapBasisError", "TapNumericError"))


def test_import_is_light_and_side_effect_free():
    code = (
        "import sys, torch\n"
        "st = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,\n"
        "      torch.backends.cudnn.benchmark, torch.initial_seed())\n"
        "import buc.teacher.taps\n"
        "assert st == (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,\n"
        "              torch.backends.cudnn.benchmark, torch.initial_seed()), 'global state changed'\n"
        "bad = [m for m in ('diffusers', 'transformers', 'huggingface_hub', 'safetensors') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
