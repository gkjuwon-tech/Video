"""RD-15 GPU 테스트(실물 교사 필요, CPU 게이트 제외). 이 환경에서는 실행하지 못했다."""
import numpy as np
import pytest
import torch

import rd15_helpers as H


@pytest.mark.gpu
def test_real_teacher_face_read_with_h2_reread_is_finite():
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
    q = torch.linalg.qr(torch.randn(d, k, generator=g))[0]
    basis = T.PCABasis(layers, torch.zeros(len(layers), d), q[None].repeat(len(layers), 1, 1),
                       torch.full((len(layers), k), 0.5))
    bus = T.TapBus(basis).attach(teacher.layers)
    mrh = MRH(seed=0).cuda().eval()
    reader = M.MrhReader(TeacherSampler(teacher), bus, mrh, token_offset=0, seed=0)
    fm = H.fm()
    img, _ = H.head_image(fh=160, cy=256, cx=256, hw=(512, 512), tone="medium", bg="foliage")
    ro = M.read_mrh(img, reader, prov=H.USER, cfg=M.ReadConfig(), key="gpu")
    table = np.random.default_rng(0).normal(size=(64, 768)).astype(np.float32)        # 무작위 표(품질 주장 없음)
    m = fm.face_map(img, ro, f_px=600.0, sem_table=table, reader=reader, prov=H.USER,
                    opts=fm.FaceOpts(face_ids=(1,)))
    assert np.isfinite(m.prob).all() and m.n_reread == len(m.faces) + m.n_candidates
    bus.detach()
