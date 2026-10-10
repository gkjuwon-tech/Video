"""RDR-01 미분: float64 torch 백엔드에서 autograd = 중앙 차분(모든 파라미터), 컬링된 surfel 기울기 0."""
import numpy as np
import torch

from buc.render.splat2d import SplatOpts, render
from rdr01_helpers import K_of, make_surfels, pose, rot_axis

W, H = 20, 16
K = K_of(18.0, W, H)
C2W = pose(rot_axis([0.2, 1.0, 0.0], 6), [0.03, -0.02, 0.0])


def _scene():
    means = [[0.11, -0.07, 2.5], [-0.17, 0.09, 3.1], [0.04, 0.13, 3.7]]
    rots = [rot_axis([1, 0.3, 0], 25), rot_axis([0, 1, 0.5], -35), rot_axis([0.4, 0.2, 1], 50)]
    return make_surfels(means, rots, [[0.45, 0.3], [0.5, 0.42], [0.62, 0.55]], [0.55, 0.7, 0.8],
                        [[0.9, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.3, 0.95]], dtype=torch.float64)


def _weights():
    g = torch.Generator().manual_seed(11)
    return (torch.rand(H, W, 3, generator=g, dtype=torch.float64) - 0.5,
            torch.rand(H, W, generator=g, dtype=torch.float64) - 0.5,
            torch.rand(H, W, generator=g, dtype=torch.float64) - 0.5)


def _loss(s, wr, wa, wd, mask):
    r = render(s, c2w=C2W, K=K, width=W, height=H, opts=SplatOpts(bg=(0.1, 0.2, 0.3)))
    d = torch.where(mask, r.depth, torch.zeros_like(r.depth))
    return (r.rgb * wr).sum() + (r.alpha * wa).sum() + (d * wd).sum(), r


def test_finite_difference_all_params():
    s = _scene()
    wr, wa, wd = _weights()
    with torch.no_grad():
        _, r0 = _loss(s, wr, wa, wd, torch.ones(H, W, dtype=torch.bool))
    assert r0.rgb.dtype == torch.float64
    mask = r0.alpha > 0.05
    assert int(mask.sum()) > 60
    g = s.clone().requires_grad_(True)
    loss, _ = _loss(g, wr, wa, wd, mask)
    loss.backward()
    h = 1e-6
    names = ("means", "quats", "log_scales", "opacity_logits", "colors")
    for name, p_ad in zip(names, g.params()):
        ad = p_ad.grad.detach().clone().reshape(-1)
        fd = torch.zeros_like(ad)
        for i in range(ad.numel()):
            sp, sm = s.clone(), s.clone()
            getattr(sp, name).view(-1)[i] += h
            getattr(sm, name).view(-1)[i] -= h
            with torch.no_grad():
                fd[i] = (_loss(sp, wr, wa, wd, mask)[0] - _loss(sm, wr, wa, wd, mask)[0]) / (2 * h)
        err = float((ad - fd).norm())
        scale = float(fd.norm())
        assert scale > 1e-3, name
        assert err <= 1e-4 * scale + 1e-6, (name, err, scale, ad, fd)


def test_culled_surfel_has_zero_grad_and_others_finite():
    s = _scene()
    behind = make_surfels([[0.0, 0.0, -1.0]], rot_axis([0, 0, 1], 0)[None], [[0.3, 0.3]], [0.8], [[1, 1, 1]],
                          dtype=torch.float64)
    from buc.render.splat2d import Surfels
    both = Surfels.cat([s, behind]).requires_grad_(True)
    r = render(both, c2w=np.eye(4), K=K, width=W, height=H)
    (r.rgb.sum() + r.alpha.sum()).backward()
    for p in both.params():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        assert float(p.grad[3].abs().max()) == 0.0
    assert float(both.colors.grad[:3].abs().sum()) > 0


def test_grad_flows_float32_and_no_nan_edge_on():
    """edge-on(노멀 ⟂ 시선) surfel 이 섞여도 float32 기울기는 유한하다."""
    s = make_surfels([[0.0, 0.0, 3.0], [0.1, 0.0, 2.0]], [rot_axis([0, 1, 0], 90), np.eye(3)],
                     [[0.4, 0.4], [0.3, 0.3]], [0.8, 0.6], [[1, 0, 0], [0, 1, 0]]).requires_grad_(True)
    r = render(s, c2w=np.eye(4), K=K, width=W, height=H)
    fin = torch.isfinite(r.depth)
    (r.rgb.sum() + r.alpha.sum() + r.depth[fin].sum() + r.normal.sum()).backward()
    for p in s.params():
        assert bool(torch.isfinite(p.grad).all())
