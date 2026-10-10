"""TCH-01 실물 5090 시험(@gpu). 실제 가중치가 HF 캐시에 있어야 한다(RUN-05 prefetch). CPU 게이트에서는 제외."""
import pytest
import torch

pytestmark = pytest.mark.gpu


def _real_teacher():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.teacher import zimage as Z
    spec = Z.TeacherSpec()
    try:
        Z.snapshot_dir(spec)
    except Z.TeacherWeightsMissing as e:
        pytest.skip(f"가중치 캐시 없음: {e}")
    torch.cuda.reset_peak_memory_stats()
    return Z, Z.load_teacher(spec, device="cuda", seed=0, loaders=Z.default_loaders(), fingerprint=False)


def test_real_load_bf16_frozen_and_memory_budget():
    Z, t = _real_teacher()
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in t.transformer.parameters())
    assert len(t.layers) == 30
    assert next(t.transformer.parameters()).device.type == "cuda"
    assert not t.text.unloaded
    # 설계 수치: 6.15B * 2 B = 12.3 GB + VAE. TE는 지연 적재라 아직 GPU에 없다.
    assert 11e9 < t.info()["weights_bytes"] < 13.5e9
    assert torch.cuda.max_memory_allocated() < 14e9
    assert Z.module_bytes(t.vae.vae) < 0.5e9


def test_real_text_cache_hit_and_release():
    Z, t = _real_teacher()
    a = t.encode_text(["a mossy stone plaque in flat daylight"])[0]
    assert a.shape[-1] == 2560 and torch.isfinite(a.float()).all()
    b = t.encode_text(["a mossy stone plaque in flat daylight"])[0]
    assert torch.equal(a, b) and t.text.stats["encode_calls"] == 1
    t.release_text_encoder()
    assert t.text.unloaded
    assert torch.cuda.memory_allocated() < 14e9


def test_real_vae_roundtrip_psnr_and_determinism():
    Z, t = _real_teacher()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing="ij")
    img = torch.stack([xx * 0.8, yy * 0.8, (xx * yy)]).unsqueeze(0).to("cuda")
    z = t.encode_image(img)
    assert tuple(z.shape) == (1, 16, 32, 32)
    assert torch.equal(z, t.encode_image(img))                          # 비트 동일
    y = t.decode_latent(z)
    mse = ((y - img) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()  # 범위 [-1,1] -> 피크 2
    assert psnr >= 25.0, psnr
    n = t.make_noise(3, 1, 256, 256)
    assert n.device.type == "cuda" and torch.equal(n.cpu(), Z.make_noise(3, (1, 16, 32, 32)))
