"""TCH-11 GPU 실물 검증(5090 필요, CPU 게이트에서는 제외). 이 컨테이너에서는 실행하지 못했다."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

PROMPTS = ["a mossy stone wall after rain", "an empty corridor with tiled floor",
           "a red kite over a wheat field", "a ceramic teapot on a wooden table"]


@pytest.fixture(scope="module")
def sampler():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.teacher.sample import TeacherSampler
    from buc.teacher.zimage import load_teacher
    return TeacherSampler(load_teacher(device="cuda", seed=0, deterministic=True))


def test_real_entry_gen_8nfe_512(sampler):
    from buc.teacher import entry_gen as E
    out = E.generate_entries(sampler, PROMPTS, seed=1, height=512, width=512, mode="survey", batch_size=2)
    assert len(out) == 4 and all(e.nfe == 8 and e.image.shape == (512, 512, 3) for e in out)
    for e in out:
        assert e.image.std() > 20.0 and 10 < e.image.mean() < 245       # 검정/흰 화면·상수 이미지가 아니다
    imgs = [e.image.astype(np.float32) for e in out]
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.abs(imgs[i] - imgs[j]).mean() > 10.0              # 프롬프트마다 서로 다른 그림


def test_real_entry_gen_reproducible_on_same_gpu(sampler):
    from buc.teacher import entry_gen as E
    a = E.generate_entries(sampler, PROMPTS[:2], seed=3, height=512, width=512, batch_size=2)
    b = E.generate_entries(sampler, PROMPTS[:2], seed=3, height=512, width=512, batch_size=2)
    for x, y in zip(a, b):
        assert x.key == y.key
        assert np.abs(x.image.astype(np.float32) - y.image.astype(np.float32)).mean() <= 0.5
    c = E.generate_entries(sampler, PROMPTS[:2], seed=3, height=512, width=512, batch_size=1)   # lockstep vs 단일 뷰
    for x, y in zip(a, c):
        assert np.abs(x.image.astype(np.float32) - y.image.astype(np.float32)).mean() <= 3.0
