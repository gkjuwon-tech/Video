"""RD-07 GPU 테스트(실물 교사 필요, CPU 게이트 제외). 이 환경에서는 실행하지 못했다."""
import numpy as np
import pytest
import torch

import rd07_helpers as H


@pytest.mark.gpu
def test_real_teacher_reads_512_square_with_finite_outputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.models.mrh.heads import MRH
    from buc.read import mrh_call as M
    from buc.teacher import taps as T
    from buc.teacher.sample import TeacherSampler
    from buc.teacher.zimage import TeacherWeightsMissing, load_teacher
    try:
        teacher = load_teacher(device="cuda")
    except TeacherWeightsMissing:
        pytest.skip("HF 캐시에 Z-Image-Turbo 가중치 없음")
    g = torch.Generator().manual_seed(0)
    layers = T.TAP_LAYERS
    d, k = 3840, 128
    q = torch.linalg.qr(torch.randn(d, k, generator=g))[0]            # 무작위 정규직교 기저(품질 주장 없음)
    basis = T.PCABasis(layers, torch.zeros(len(layers), d), q[None].repeat(len(layers), 1, 1),
                       torch.full((len(layers), k), 0.5))
    bus = T.TapBus(basis).attach(teacher.layers)
    mrh = MRH(seed=0).cuda().eval()
    reader = M.MrhReader(TeacherSampler(teacher), bus, mrh, token_offset=0, seed=0)   # 이미지 토큰 위치는 미검증
    img = H.make_image(512, 512, seed=1)
    r = M.read_mrh(img, reader, prov=H.USER, cfg=M.ReadConfig(), key="gpu")
    assert r.lane == "cleanroom" and r.n_forward >= 1
    assert tuple(r.outputs["log_depth"].shape) == (1, 1, 128, 128)
    assert all(bool(torch.isfinite(v).all()) for v in r.outputs.values()) and np.isfinite(r.r_dis)
    bus.detach()
