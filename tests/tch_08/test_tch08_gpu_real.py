"""TCH-08 실물 검증(CUDA + 오프라인 HF 캐시의 klein 4B + 사람이 확인한 라이선스). CPU 게이트에서는 제외.

주의: 이 파일은 CPU 컨테이너에서 실행되지 않았다. klein 저장소 이름·잠재 채널·TE 차원·σ 표는 [2차] 잠정값이며
첫 GPU 실행에서 확인·보정한다. 라이선스 확인은 BUC_KLEIN_LICENSE_VERIFIED=1 로 사람이 명시해야 한다."""
import os
from dataclasses import replace

import pytest
import torch

from buc.teacher import ref_slot as R

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def klein():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    if os.environ.get("BUC_KLEIN_LICENSE_VERIFIED") != "1":
        pytest.skip("klein 라이선스 원문 확인 전(N 클래스)")
    from buc.teacher import zimage
    spec = replace(R.KleinSpec(), license_verified=True)
    try:
        return R.load_klein(spec, device="cuda", te_device="cpu", seed=0)
    except (zimage.TeacherWeightsMissing, zimage.TeacherDependencyError) as e:
        pytest.skip(f"실물 가중치/의존성 없음: {e}")


def test_real_load(klein):
    assert klein.license == "Apache-2.0"
    assert klein.dtype == torch.bfloat16
    assert all(not p.requires_grad for p in klein.raw_transformer.parameters())
    assert klein.info()["weights_bytes"] < 12 * 2 ** 30


def test_table_matches_scheduler(klein):
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        sch = FlowMatchEulerDiscreteScheduler.from_pretrained(klein.snapshot, subfolder="scheduler",
                                                              local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"스케줄러 설정 없음: {e}")
    sch.set_timesteps(4)
    got = [float(s) for s in sch.sigmas[:4]]
    assert got == pytest.approx(list(R.KLEIN_TABLE.points), abs=5e-3)


def test_generate_with_two_refs(klein):
    from buc.teacher.sample import TeacherSampler
    y, x = torch.meshgrid(torch.linspace(-1, 1, 512), torch.linspace(-1, 1, 512), indexing="ij")
    imgs = [torch.stack([x, y, x * y]).clamp(-1, 1), torch.stack([y, x, -x * y]).clamp(-1, 1)]
    lat = klein.encode_refs(imgs)
    klein.set_refs(lat)
    s = TeacherSampler(klein, table=R.KLEIN_TABLE)
    out = s.generate(["an oil painting of a harbor, thick impasto strokes"], seeds=[0], height=512, width=512)
    assert out.calls == 4 and bool(torch.isfinite(out.z).all())
    img = klein.decode_latent(out.z)
    assert img.shape[-2:] == (512, 512) and float(img.std()) > 0.05
    assert torch.cuda.max_memory_allocated() < 15 * 2 ** 30
