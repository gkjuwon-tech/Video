"""GPU 러너 전용: 실제 CUDA에서 fp16 GradScaler 학습과 재개. CPU 게이트에서는 제외(-m "not gpu")."""
import pytest
import torch
from run06_helpers import make_model, make_trainer

from buc.train.amp import AmpConfig

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")]


def test_fp16_cuda_training_and_resume(tmp_path):
    amp = AmpConfig(dtype="fp16", device_type="cuda", init_scale=1024.0, growth_interval=5)
    kw = dict(dropout=0.0, amp=amp, lr=5e-2, sched=False, grad_clip=None, device="cuda")
    full, logs = make_trainer(tmp_path / "f", max_steps=20, **kw)
    full.fit()
    assert all(p.dtype == torch.float32 and p.is_cuda for p in full.model.parameters())
    assert full.amp.scaling and full.amp.scale_value >= 1024.0
    d = tmp_path / "ck"
    a, _ = make_trainer(d, max_steps=10, **kw)
    a.fit()
    b, _ = make_trainer(d, max_steps=20, model=make_model(5, 0.0), **kw)
    b.fit()
    for (k, x), (_, y) in zip(full.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.allclose(x.cpu(), y.cpu(), atol=1e-5, rtol=1e-4), k
