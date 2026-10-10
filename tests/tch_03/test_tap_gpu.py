"""TCH-03 GPU 시험(5090/실제 Z-Image-Turbo 가중치 필요). CPU 게이트에서는 제외된다.

실제 층의 토큰 배치(이미지 토큰이 앞인지, 패딩 여부)는 이 파일의 시험으로 확인한다. 가짜 블록으로는 알 수 없다.
"""
import pytest
import torch

import tch03_fakes as F
from buc.teacher import taps as T

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def teacher():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.teacher import zimage
    try:
        return zimage.load_teacher(fingerprint=False)
    except zimage.TeacherWeightsMissing as e:
        pytest.skip(f"가중치 없음: {e}")


def test_hooks_attach_to_real_layers_and_detach_cleanly(teacher):
    layers = teacher.layers
    assert len(layers) == 30
    dim = teacher.spec.hidden_dim
    g = F.gen(0)
    samples = {l: torch.randn(256, dim, generator=g) for l in T.TAP_LAYERS}
    basis = T.fit_pca_basis(samples, prov=F.ENGINE)               # k=128, D=3840
    assert basis.k == 128 and basis.dim == dim
    bus = T.TapBus(basis).attach(layers)
    assert bus.hook_count == 3
    bus.detach()
    assert all(len(layers[i]._forward_hooks) == 0 for i in range(30))


def test_real_forward_fires_each_tapped_layer_once(teacher):
    pytest.skip("실제 transformer 호출 시그니처는 TCH-02 소관이다. TCH-02 완료 뒤 이 시험을 채운다(token_offset·격자 검증 포함).")
