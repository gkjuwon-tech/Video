"""TCH-02 실물 검증(5090 + 오프라인 HF 캐시 필요). CPU 게이트에서는 제외(-m "not gpu").

주의: 이 파일은 CPU 컨테이너에서 실행되지 않았다. 임계값은 첫 GPU 실행에서 보정이 필요할 수 있다."""
import math

import pytest
import torch

from buc.teacher import sample as S

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def teacher():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.teacher import zimage
    try:
        return zimage.load_teacher(device="cuda", te_device="cpu", seed=0)
    except (zimage.TeacherWeightsMissing, zimage.TeacherDependencyError) as e:
        pytest.skip(f"실물 가중치/의존성 없음: {e}")


def _image(h=512, w=512):
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    img = torch.stack([0.8 * x, 0.8 * y, 0.5 * torch.sin(3 * x) * torch.cos(2 * y)])[None]
    img[:, :, h // 4:h // 2, w // 4:w // 2] = 0.9
    return img.clamp(-1, 1)


def _psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item() / 4.0       # [-1,1] -> [0,1] 이므로 /4
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def test_default_table_matches_snapshot_scheduler(teacher):
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        sch = FlowMatchEulerDiscreteScheduler.from_pretrained(teacher.snapshot, subfolder="scheduler",
                                                              local_files_only=True)
    except Exception as e:                                           # noqa: BLE001
        pytest.skip(f"스케줄러 설정을 읽을 수 없다: {e}")
    sch.set_timesteps(9)
    ref = [float(s) for s in sch.sigmas[:8]]
    assert S.DEFAULT_TABLE.points == pytest.approx(ref, abs=5e-3), (S.DEFAULT_TABLE.points, ref)


def test_sdedit_roundtrip_preserves_image(teacher):
    smp = S.TeacherSampler(teacher)
    img = _image().to("cuda")
    z = teacher.encode_image(img)
    r = smp.sdedit(z, ["a photo of a smooth gradient wall"], sigma=0.3, seeds=1, steps=2)
    out = teacher.decode_latent(r.z)
    assert torch.isfinite(out).all()
    assert _psnr(out.cpu(), img.cpu()) > 18.0, "낮은 σ SDEdit 은 저대역을 보존해야 한다(부호/t 규약 오류 의심)"
    # 노이즈를 더하지 않고 같은 규약으로 x̂0 만 읽어도 크게 어긋나지 않는다
    x0 = smp.predict_x0(smp.sdedit(z, ["a photo"], sigma=0.3, seeds=1, steps=1).z_start, S.DEFAULT_TABLE.points[7],
                        ["a photo"])
    assert _psnr(teacher.decode_latent(x0).cpu(), img.cpu()) > 15.0


def test_lockstep_batch_matches_single_views_within_tolerance(teacher):
    smp = S.TeacherSampler(teacher, max_views_per_call=4)
    z = teacher.encode_image(torch.cat([_image(), _image().flip(-1), -_image()]).to("cuda"))
    prompts = ["a photo of a wall", "a photo of a floor", "a photo of a door"]
    seeds = [3, 4, 5]
    full = smp.sdedit(z, prompts, sigma=0.3, seeds=seeds, steps=2)
    for i in range(3):
        one = smp.sdedit(z[i:i + 1], prompts[i:i + 1], sigma=0.3, seeds=[seeds[i]], steps=2)
        diff = (one.z[0] - full.z[i]).abs().mean().item()
        assert diff < 0.05, (i, diff)           # bf16 배치 비결정성 허용. 실패 시 임계값은 첫 실측으로 보정


def test_same_inputs_same_bits_on_gpu(teacher):
    smp = S.TeacherSampler(teacher)
    z = teacher.encode_image(_image().to("cuda"))
    a = smp.sdedit(z, ["a photo"], sigma=0.3, seeds=7, steps=2)
    b = smp.sdedit(z, ["a photo"], sigma=0.3, seeds=7, steps=2)
    assert torch.equal(a.z, b.z) and torch.equal(a.x0_hat, b.x0_hat)
