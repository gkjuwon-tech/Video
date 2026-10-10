"""실물 교사 스모크(GPU 러너 전용). CPU 게이트에서는 -m "not gpu" 로 빠진다.

검증하지 못한 것: 실제 Z-Image 의 이미지 토큰 오프셋(BUC_TOKEN_OFFSET 환경변수로 준다), 실제 결정 곡선(TCH-05), τ."""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.gpu
def test_real_teacher_realify_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from buc.data import realify as R
    from buc.data.synth_render import render_room
    from buc.gov import prov as P
    from buc.teacher import taps as T
    from buc.teacher.sample import TeacherSampler
    from buc.teacher.zimage import TeacherWeightsMissing, load_teacher

    try:
        teacher = load_teacher()
    except TeacherWeightsMissing:
        pytest.skip("교사 가중치 없음")
    off = int(os.environ.get("BUC_TOKEN_OFFSET", "0"))
    basis_path = os.environ.get("BUC_TAP_BASIS")
    if not basis_path or not os.path.exists(basis_path):
        pytest.skip("BUC_TAP_BASIS (TCH-03 동결 기저) 없음")
    basis = T.PCABasis.load(basis_path)
    bus = T.TapBus(basis).attach(teacher.layers)
    try:
        sampler = TeacherSampler(teacher)
        it = R.item_from_synth(render_room(1, width=512, height=512), key="gpu0", caption="a photo of a room")
        r = R.realify(sampler, bus, [it], curve=R.provisional_curve(sampler.table, 8), sigma=0.5, seed=0,
                      token_offset=off, tau=1.0)[0]
    finally:
        bus.detach()
    assert r.image.shape == (512, 512, 3) and np.isfinite(r.image).all()
    assert r.taps.grid == (32, 32)
    assert r.band_valid[-1] is True
    assert r.lowband_err < 0.1, r.lowband_err          # 잠정 하한: 거친 대역 RMS 0.1 이하
    assert P.train_ok(r.prov)
