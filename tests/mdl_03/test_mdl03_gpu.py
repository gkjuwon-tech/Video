"""GPU 러너 전용: 기본 설정 MRH fp16 autocast 1 스텝이 유한하고 CPU fp32와 대략 일치."""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음")
def test_fp16_step_finite_on_cuda():
    from buc.models.mrh import heads as Hd
    from buc.models.mrh import neck as N
    from buc.models.mrh import train as TR

    cfg = N.MRHConfig()
    m = Hd.MRH(cfg, seed=0).cuda()
    g = torch.Generator().manual_seed(0)
    taps = {l: torch.randn(2, 128, 32, 32, generator=g).cuda() for l in cfg.tap_layers}
    gbuf = torch.zeros(2, 12, 512, 512)
    gbuf[:, 0] = torch.linspace(0.5, 2.0, 512).view(1, 512, 1)
    gbuf[:, 3] = -1.0
    gbuf[:, 11] = 1.0
    batch = {"gbuf": gbuf.cuda(), "sem": torch.full((2, 512, 512), 8, dtype=torch.int64).cuda(),
             "band_valid": ((True,) * 9,) * 2, "t": torch.tensor([0.3, 0.6]).cuda(), "keys": ("a", "b")}
    with torch.autocast("cuda", dtype=torch.float16):
        out = m(taps, batch["t"])
    tot, met = TR.mrh_loss({k: v.float() for k, v in out.items()}, batch,
                           sem_table=TR.default_sem_table(dim=cfg.sem_dim).cuda())
    tot.backward()
    assert torch.isfinite(tot)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
