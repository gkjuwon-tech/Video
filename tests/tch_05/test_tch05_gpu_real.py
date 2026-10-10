"""TCH-05 실물 교사 측정(5090 필요, 실행되지 않음): 결정 곡선이 단조이고 폭이 유한하며 해석이 나온다.
수치 임계는 이 단위의 수용 기준이 아니라 E2 (a)의 판독이다: 결과 해석은 사람(Fable)이 한다."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def real():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.teacher.sample import TeacherSampler
    from buc.teacher.zimage import load_teacher
    teacher = load_teacher(seed=0)
    return TeacherSampler(teacher), teacher


def _natural(n, size):
    g = np.random.default_rng(0)
    out = []
    for _ in range(n):
        f = np.fft.fft2(g.standard_normal((size, size, 3)), axes=(0, 1))
        fy, fx = np.fft.fftfreq(size)[:, None, None], np.fft.fftfreq(size)[None, :, None]
        r = np.sqrt(fx ** 2 + fy ** 2)
        r[0, 0] = 1
        im = np.real(np.fft.ifft2(f / r, axes=(0, 1)))
        out.append(np.clip(im / im.std() * 0.3, -1, 1))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float()


def test_real_teacher_curve_is_finite_and_monotone_in_sigma(real):
    from buc.teacher import decision_curve as dc
    sampler, teacher = real
    cur = dc.run_x1_slim(sampler, dc.TeacherCodec(teacher), _natural(2, 512), ["a photo of rough concrete"],
                         seeds=[1, 2], levels=5)
    assert np.all(np.isfinite(cur.decision)) and np.all((cur.decision >= 0) & (cur.decision <= 2))
    assert cur.decision[0, 0] > cur.decision[0, -1]            # σ 최대에서 미세 옥타브가 굵은 옥타브보다 많이 바뀐다
    assert cur.decision[0].sum() > cur.decision[-1].sum()      # σ 가 작아질수록 덜 바뀐다
    it = dc.interpret(cur)
    assert it.fallback in ("none", "widen_level_gap_or_sr_lora", "inconclusive")
