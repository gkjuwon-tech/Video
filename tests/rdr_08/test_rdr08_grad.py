"""RDR-08 미분·속도·GPU: float64 gradcheck(가우시안 means·log_density·colors, 가닥 points·opacity·colors),
CPU 속도, CUDA = CPU."""
import time

import numpy as np
import pytest
import torch

from buc.render.vol import Strands, VolGaussians, VolOpts, render_vol
from rdr08_helpers import K_of, at_pixel, make_strands, make_vol, rot_axis

W, H = 12, 10
K = K_of(30.0, W, H)
I4 = np.eye(4)


def test_gradcheck_float64():
    mus = np.array([at_pixel(K, 5.3, 4.6, 2.0), at_pixel(K, 6.4, 5.2, 2.6)])
    rots = np.stack([rot_axis([1, 0.3, 0], 20), rot_axis([0, 1, 0.5], 40)])
    v0 = make_vol(mus, rots, [[0.15, 0.1, 0.08], [0.2, 0.15, 0.1]], [3.0, 2.0], [[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]])
    pts = np.stack([at_pixel(K, 1.3, 2.2, 1.5), at_pixel(K, 10.6, 7.7, 1.8)])
    s0 = make_strands(pts[None], [[0.04, 0.03]], [0.5], [[0.9, 0.9, 0.2]])
    opts = VolOpts(bg=(0.1, 0.2, 0.3))

    def f(means, log_density, colors, points, opacity, scolors):
        v = VolGaussians(means=means, quats=v0.quats, log_scales=v0.log_scales, log_density=log_density,
                         colors=colors, sggx=v0.sggx)
        s = Strands(points=points, radius=s0.radius, opacity=opacity, colors=scolors)
        out = render_vol(vol=v, strands=s, c2w=I4, K=K, width=W, height=H, opts=opts)
        return out.rgb, out.alpha

    inputs = tuple(x.clone().requires_grad_(True) for x in
                   (v0.means, v0.log_density, v0.colors, s0.points, s0.opacity, s0.colors))
    out = f(*inputs)
    assert out[1].max() > 0.2 and out[1].min() < 0.05
    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradient_flows_through_sggx_and_scales():
    v = make_vol([at_pixel(K, 5.5, 4.5, 2.0)], np.eye(3)[None], [[0.1, 0.1, 0.1]], [2.0], [[1, 1, 1]])
    sg = v.sggx.clone().requires_grad_(True)
    ls = v.log_scales.clone().requires_grad_(True)
    vv = VolGaussians(means=v.means, quats=v.quats, log_scales=ls, log_density=v.log_density, colors=v.colors,
                      sggx=sg)
    out = render_vol(vol=vv, c2w=I4, K=K, width=W, height=H)
    out.alpha.sum().backward()
    assert sg.grad is not None and torch.isfinite(sg.grad).all() and sg.grad.abs().sum() > 0
    assert ls.grad is not None and torch.isfinite(ls.grad).all() and ls.grad.abs().sum() > 0


def _big_scene(dtype=torch.float32, device="cpu"):
    Wb, Hb = 64, 48
    Kb = K_of(60.0, Wb, Hb)
    rng = np.random.default_rng(0)
    n = 1500
    mus = np.c_[rng.uniform(-0.8, 0.8, n), rng.uniform(-0.6, 0.6, n), rng.uniform(2.0, 4.0, n)]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, 90)) for _ in range(n)])
    v = make_vol(mus, rots, rng.uniform(0.01, 0.05, (n, 3)), rng.uniform(1, 20, n), rng.uniform(0, 1, (n, 3)),
                 dtype=dtype)
    roots = np.c_[rng.uniform(-0.8, 0.8, 100), rng.uniform(-0.6, 0.0, 100), rng.uniform(2.0, 3.0, 100)]
    pts = roots[:, None, :] + np.linspace(0, 1, 7)[None, :, None] * np.array([0.02, 0.4, 0.05])[None, None]
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    s = make_strands(pts, np.full((100, 7), 0.003), rng.uniform(0.3, 0.9, 100), rng.uniform(0, 1, (100, 3)),
                     dtype=dtype)
    return v.to(device=device), s.to(device=device), Kb, Wb, Hb


def test_speed_cpu():
    v, s, Kb, Wb, Hb = _big_scene()
    for p in v.params() + s.params():
        p.requires_grad_(True)
    t0 = time.perf_counter()
    out = render_vol(vol=v, strands=s, c2w=I4, K=Kb, width=Wb, height=Hb)
    (out.rgb.sum() + out.alpha.sum()).backward()
    dt = time.perf_counter() - t0
    assert torch.isfinite(v.means.grad).all() and torch.isfinite(s.points.grad).all()
    assert out.alpha.mean() > 0.1
    assert dt < 10.0, dt


@pytest.mark.gpu
def test_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    v, s, Kb, Wb, Hb = _big_scene()
    a = render_vol(vol=v, strands=s, c2w=I4, K=Kb, width=Wb, height=Hb).numpy()
    vg, sg = v.to(device="cuda"), s.to(device="cuda")
    b = render_vol(vol=vg, strands=sg, c2w=I4, K=Kb, width=Wb, height=Hb).numpy()
    for k in ("rgb", "alpha"):
        np.testing.assert_allclose(a[k], b[k], atol=2e-4)
