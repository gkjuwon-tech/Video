"""WLD-08 GPU 스모크(실제 Z-Image). CPU 게이트에서는 제외된다(-m "not gpu"). 이 작성자는 실행하지 못했다."""
import numpy as np
import pytest

import wld08_helpers as h

S = h.mod()


@pytest.mark.gpu
def test_real_zimage_sdedit_fill_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 필요")
    from buc.teacher.sample import TeacherSampler
    from buc.teacher.zimage import TeacherWeightsMissing, load_teacher
    from buc.world.depth_world import build_depth_world

    try:
        teacher = load_teacher(device="cuda")
    except TeacherWeightsMissing:
        pytest.skip("교사 가중치 없음")
    w, hh = 384, 256
    f = 100.0 * w / 96.0
    obs = h.obs_of(w, hh, f)
    v, u = np.mgrid[0:hh, 0:w]
    img = np.stack([0.5 + 0.3 * np.sin(2 * np.pi * u / 160.0), 0.5 + 0.25 * np.cos(2 * np.pi * v / 120.0),
                    0.3 + 0.4 * u / w], axis=-1).astype(np.float32)
    box = (v >= 80) & (v < 176) & (u >= 160) & (u < 256)
    img[box] = (0.9, 0.1, 0.1)
    depth = np.where(box, 2.0, 6.0).astype(np.float32)
    dw = build_depth_world(h.make_read(img, depth, obs=obs))
    res = S.sdedit_avg_fill(dw, TeacherSampler(teacher), seed=0)
    assert res.filled.any()
    a, b = dw.world, res.world
    assert h.layer_bytes_equal(a.layers[0], b.layers[0])
    assert np.array_equal(a.layers[1].rgba[~dw.hidden], b.layers[1].rgba[~dw.hidden])
    px = b.layers[1].rgba[res.filled][:, :3]
    assert np.isfinite(px).all() and px.min() >= 0 and px.max() <= 1
