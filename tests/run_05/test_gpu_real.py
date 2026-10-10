"""실제 GPU 서버에서만: 부트스트랩이 gsplat 을 설치·JIT 빌드한다(가중치 prefetch 는 생략)."""
import json
import os
import subprocess
import sys

import pytest
from run05_helpers import SCRIPT


@pytest.mark.gpu
def test_real_gsplat_bootstrap(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    root = os.environ.get("BUC_ROOT_TEST") or str(tmp_path / "buc")
    env = dict(os.environ, BUC_ROOT=root, BUC_PYTHON=sys.executable, BUC_SKIP_PREFETCH="1")
    p = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=3600)
    j = json.load(open(os.path.join(root, "bootstrap.json")))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    maj, mi = torch.cuda.get_device_capability(0)
    assert j["env"]["cap"] == [maj, mi]
    assert j["gsplat"]["status"] in ("ok", "cached")
    q = subprocess.run(["bash", "-c", f'source "{root}/env.sh"; "{sys.executable}" -c "import gsplat; print(gsplat.__version__)"'],
                       capture_output=True, text=True, timeout=600)
    assert q.returncode == 0, q.stderr[-2000:]
