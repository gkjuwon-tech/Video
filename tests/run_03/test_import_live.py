"""import 순수성(CPU에서 hub 없이 import), 소스 위생, 실제 허브 왕복(GPU 러너 전용)."""
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from buc.rt import hf_sync as H

SRC = Path(H.__file__)
ROOT = SRC.parents[2]


def test_import_has_no_heavy_or_hub_dependency():
    code = (
        "import sys; sys.path.insert(0, %r); sys.modules['huggingface_hub'] = None\n"
        "import buc.rt.hf_sync as m\n"
        "bad = [n for n in ('torch', 'huggingface_hub', 'requests', 'numpy') if sys.modules.get(n) is not None]\n"
        "assert not bad, bad\n"
        "print('OK')\n" % str(ROOT)
    )
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
    assert p.returncode == 0 and p.stdout.strip() == "OK", p.stderr[-2000:]


def test_hf_client_without_hub_raises_hubsyncerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import 시 ImportError
    with pytest.raises(H.HubSyncError) as ei:
        H.HfHubClient(repo_id="a/b", repo_type="model", token="tok-value-123")
    assert "tok-value-123" not in str(ei.value)
    with pytest.raises(H.HubSyncError):
        H.HubSync.from_env(env={"HF_TOKEN": "tok-value-123"})


def test_source_has_no_hardcoded_secret_and_never_makes_public_repo():
    src = SRC.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", src)
    assert not re.search(r"private\s*=\s*False", src)
    assert not re.search(r"(?i)api[_-]?key\s*=\s*['\"][^'\"]{8,}", src)


@pytest.mark.gpu
def test_live_private_repo_roundtrip(tmp_path):
    """GPU 러너 전용: 실제 허브 왕복. 운영 금고에는 쓰지 않도록 BUC_HF_TEST_REPO(시험용 비공개 저장소)를 요구한다."""
    pytest.importorskip("huggingface_hub")
    repo = os.environ.get("BUC_HF_TEST_REPO")
    if not os.environ.get("HF_TOKEN") or not repo:
        pytest.skip("HF_TOKEN 또는 BUC_HF_TEST_REPO 없음")
    assert repo != H.DEFAULT_REPO_ID
    s = H.HubSync.from_env(repo_id=repo)
    s.verify_private()
    run = "ci-" + uuid.uuid4().hex[:10]
    src = tmp_path / "w.pt"
    src.write_bytes(os.urandom(4096))
    assert s.upload_checkpoint(src, run, 7).uploaded
    assert s.upload_checkpoint(src, run, 7).skipped == "up_to_date"
    assert s.latest_checkpoint(run).step == 7
    assert s.fetch_latest(run, tmp_path / "back").read_bytes() == src.read_bytes()
