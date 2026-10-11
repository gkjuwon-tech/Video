"""TCH-01 회귀: 결정론 모드가 'torch.empty 를 NaN 으로 채우는 포이즌'을 켠 채 남기지 않는다.

배경: ``torch.use_deterministic_algorithms(True)`` 는 ``torch.utils.deterministic.fill_uninitialized_memory`` (기본 True)
도 켠다. 그러면 diffusers·ATen 내부의 모든 ``empty*`` 가 NaN 으로 채워진다. 같은 박스의 순정 ZImagePipeline 은 결정론 모드를
켜지 않으므로 이 포이즌을 만나지 않는다. 재현성 계약(같은 입력 -> 같은 비트)에 포이즌은 필요 없다."""
from dataclasses import replace

import pytest
import torch

from buc.teacher import zimage as Z


@pytest.fixture(autouse=True)
def restore_fill_flag():
    prev = getattr(torch.utils.deterministic, "fill_uninitialized_memory", None)
    if prev is None:
        pytest.skip("이 torch 에는 fill_uninitialized_memory 스위치가 없다")
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = prev


def test_torch_poison_precondition_is_real():
    """전제 확인: 결정론 알고리즘 + fill 이 켜져 있으면 torch.empty 는 전부 NaN 이다(우리 쪽이 아니라 torch 의 동작)."""
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    assert bool(torch.isnan(torch.empty(4096)).all())


def test_enable_determinism_mode_keeps_determinism_but_drops_the_poison():
    torch.utils.deterministic.fill_uninitialized_memory = True
    rep = Z.enable_determinism_mode(seed=1)
    assert torch.are_deterministic_algorithms_enabled() and rep["deterministic_algorithms"] is True
    assert torch.utils.deterministic.fill_uninitialized_memory is False


def test_load_teacher_deterministic_leaves_no_poison(tmp_path):
    from tch01_fakes import FakeTE, FakeTransformer, FakeVae, make_hub
    spec = replace(Z.TeacherSpec(), num_layers=4, text_dim=8, max_text_len=16)
    torch.utils.deterministic.fill_uninitialized_memory = True
    loaders = Z.Loaders(tokenizer=lambda p: None, text_encoder=lambda p, d: FakeTE(8).to(d),
                        transformer=lambda p, d: FakeTransformer(4).to(d), vae=lambda p, d: FakeVae().to(d),
                        encode_prompts=lambda tok, te, ps, n: [torch.zeros(2, 8) for _ in ps])
    t = Z.load_teacher(spec, device="cpu", dtype=torch.float32, hub_cache=make_hub(tmp_path), loaders=loaders,
                       deterministic=True, seed=3)
    assert t.determinism["deterministic_algorithms"] is True
    assert torch.utils.deterministic.fill_uninitialized_memory is False
