"""TCH-06 실물 검증(5090 + 오프라인 HF 캐시 필요). CPU 게이트에서는 제외(-m "not gpu").

주의: 이 파일은 CPU 컨테이너에서 실행되지 않았다. 임계값은 첫 GPU 실행에서 보정이 필요할 수 있다.
합성 이미지만 쓰므로 '사진 vs 그림' 분리가 실제 사진에서도 성립하는지는 이 테스트로 알 수 없다(E0 (e)가 따로 잰다)."""
import pytest
import torch

from buc.teacher import dca as D
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


def _smooth(n, h=512, w=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    out = []
    for _ in range(n):
        a, b, c = (torch.rand(3, generator=g) * 2 - 1) * 0.6
        out.append(torch.stack([a * x + 0.3 * c * y, b * y, 0.4 * torch.sin(3 * x + 2 * c) * torch.cos(2 * y)]).clamp(-1, 1))
    return torch.stack(out)


def test_dca_finite_antisymmetric_and_dr_scales_with_noise(teacher):
    smp = S.TeacherSampler(teacher)
    z = teacher.encode_image(_smooth(2).to("cuda"))
    a, b = ["a photograph of a wall"], ["an oil painting of a wall"]
    r = D.teacher_dca(smp, z, a, b, seeds=[1, 2])
    rv = D.teacher_dca(smp, z, b, a, seeds=[1, 2])
    assert torch.isfinite(r.dca).all()
    torch.testing.assert_close(r.dca, -rv.dca, rtol=1e-3, atol=1e-4)
    assert r.calls == 4
    # 잠재에 노이즈를 얹은 '불량 이미지'는 DR 이 더 크다(방향 확인; 크기 임계는 첫 실행에서 보정)
    bad = z + 0.5 * torch.randn_like(z)
    good_dr = D.teacher_dr(smp, z, a, seeds=[1, 2], sigmas=(0.5,)).dr
    bad_dr = D.teacher_dr(smp, bad, a, seeds=[1, 2], sigmas=(0.5,)).dr
    assert bool((bad_dr > good_dr).all()), (good_dr, bad_dr)


def test_dr_auc_real_vs_corrupted_has_a_floor(teacher):
    """E0 (e) 의 축소판: 매끈한 이미지 8장 vs 잠재 오염 8장. 교사 DR 의 판별 AUC 가 0.7 이상이어야 한다."""
    smp = S.TeacherSampler(teacher)
    z = teacher.encode_image(_smooth(8, seed=3).to("cuda"))
    g = torch.Generator(device="cuda").manual_seed(0)
    fake = z + 0.35 * torch.randn(z.shape, generator=g, device="cuda")
    cond = ["a smooth gradient wall"]
    seeds = list(range(8))
    real_dr = D.teacher_dr(smp, z, cond, seeds=seeds, sigmas=(0.5, 0.3)).dr.cpu()
    fake_dr = D.teacher_dr(smp, fake, cond, seeds=seeds, sigmas=(0.5, 0.3)).dr.cpu()
    assert D.dr_auc(real_dr, fake_dr) >= 0.7
    # 일부 σ 가 표 밖이면 거부(교사 스냅 표는 TCH-02 와 같다)
    with pytest.raises(S.SigmaError):
        D.teacher_dr(smp, z, cond, seeds=seeds, sigmas=(0.42,))
