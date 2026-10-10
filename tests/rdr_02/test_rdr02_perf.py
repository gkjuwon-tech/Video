"""RDR-02 속도: '빠른 경로'. 값이 아니라 복잡도(층 수에 거의 선형)와 느슨한 절대 상한만 본다."""
import time

import numpy as np

from buc.core import contracts as C
from buc.render.quads import render_ldi, render_shot
from rdr02_helpers import K_default, layer, make_obs, make_world, smooth_tex, translate, yaw


def _world(w, h, n_layers):
    layers = []
    for i in range(n_layers):
        t = smooth_tex(w, h, alpha=0.8 if i < n_layers - 1 else 1.0)
        layers.append(layer(t, 2.0 + 1.5 * i))
    return make_world(layers, make_obs(w=w, h=h))


def test_320x180_four_layers_single_frame_under_budget():
    w, h = 320, 180
    world = _world(w, h, 4)
    c2w = yaw(translate(np.eye(4), (0.1, 0.0, 0.3)), 4.0)
    render_ldi(world, c2w=c2w, K=K_default(w, h), width=w, height=h)                 # 워밍업
    t0 = time.perf_counter()
    out = render_ldi(world, c2w=c2w, K=K_default(w, h), width=w, height=h)
    dt = time.perf_counter() - t0
    assert dt < 8.0, f"{dt:.1f}s"
    assert out.alpha.mean() > 0.9


def test_eight_frame_shot_160x90_under_budget():
    w, h, n = 160, 90, 8
    world = _world(w, h, 3)
    c2w = np.stack([translate(np.eye(4), (0.02 * i, 0.0, 0.05 * i)) for i in range(n)])
    shot = C.Shot(c2w=c2w, t=np.arange(n) / 24.0, K=np.stack([K_default(w, h)] * n), focus_m=np.full(n, 3.0),
                  width=w, height=h, fps=24.0, theta=C.Theta())
    t0 = time.perf_counter()
    fr = render_shot(world, shot)
    dt = time.perf_counter() - t0
    assert fr.rgb.shape == (n, h, w, 3)
    assert dt < 20.0, f"{dt:.1f}s"


def test_cost_scales_roughly_linearly_with_layers():
    w, h = 160, 90
    c2w = translate(np.eye(4), (0.1, 0.0, 0.2))

    def timeit(n):
        world = _world(w, h, n)
        render_ldi(world, c2w=c2w, K=K_default(w, h), width=w, height=h)
        t0 = time.perf_counter()
        render_ldi(world, c2w=c2w, K=K_default(w, h), width=w, height=h)
        return time.perf_counter() - t0

    t2, t8 = timeit(2), timeit(8)
    assert t8 < 8.0 * t2 + 1.0, (t2, t8)                                              # 층 4배 -> 시간 최대 약 8배
