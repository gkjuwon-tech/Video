"""RDR-01 GPU(5090, RUN-05 부트스트랩 후): gsplat 백엔드 ↔ torch 기준 구현 수치·기울기 일치, auto 선택, 피팅."""
import numpy as np
import pytest
import torch

from buc.render.splat2d import SplatOpts, render, resolve_backend
from rdr01_helpers import K_of, make_surfels, pose, rot_axis

pytestmark = pytest.mark.gpu

W, H = 96, 64
K = K_of(80.0, W, H)
C2W = pose(rot_axis([0.2, 1, 0], 5), [0.05, -0.02, 0.0])


def _scene(n=300, seed=0):
    rng = np.random.default_rng(seed)
    means = np.c_[rng.uniform(-1, 1, n), rng.uniform(-0.7, 0.7, n), rng.uniform(2.5, 5, n)]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, 70)) for _ in range(n)])
    return make_surfels(means, rots, rng.uniform(0.03, 0.12, (n, 2)), rng.uniform(0.2, 0.9, n),
                        rng.uniform(0, 1, (n, 3))).to(device="cuda")


def test_auto_picks_gsplat_on_cuda(monkeypatch):
    monkeypatch.delenv("BUC_SPLAT_BACKEND", raising=False)
    assert torch.cuda.is_available()
    assert resolve_backend("auto", "cuda") == "gsplat"


def test_gsplat_matches_torch_reference():
    s = _scene()
    g = render(s, c2w=C2W, K=K, width=W, height=H, opts=SplatOpts(backend="gsplat", bg=(0.1, 0.1, 0.1)))
    t = render(s, c2w=C2W, K=K, width=W, height=H, opts=SplatOpts(backend="torch", bg=(0.1, 0.1, 0.1)))
    assert g.backend == "gsplat" and t.backend == "torch"
    assert float((g.rgb - t.rgb).abs().mean()) < 2e-3
    assert float((g.alpha - t.alpha).abs().mean()) < 2e-3
    m = (t.alpha > 0.5) & (g.alpha > 0.5)
    assert float(((g.depth[m] / t.depth[m]) - 1).abs().mean()) < 1e-3
    assert float((g.normal[m] * t.normal[m]).sum(-1).mean()) > 0.99


def test_gsplat_gradients_match_torch():
    s = _scene(n=120, seed=1)
    grads = {}
    for be in ("gsplat", "torch"):
        x = s.clone().requires_grad_(True)
        r = render(x, c2w=C2W, K=K, width=W, height=H, opts=SplatOpts(backend=be))
        (r.rgb.square().sum() + r.alpha.sum()).backward()
        grads[be] = torch.cat([p.grad.reshape(-1) for p in (x.means, x.log_scales, x.opacity_logits, x.colors)])
    a, b = grads["gsplat"], grads["torch"]
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    assert cos > 0.99, cos


def test_fit_on_cuda_runs():
    from buc.render.fit import FitOpts, FitView, fit
    gt = _scene(n=80, seed=2)
    img = render(gt, c2w=C2W, K=K, width=W, height=H).rgb.detach()
    init = gt.clone()
    init.colors.fill_(0.5)
    res = fit(init, [FitView(image=img.cpu().numpy(), c2w=C2W, K=K)], FitOpts(iters=60, lr_colors=2e-2))
    assert res.losses[-1] < 0.5 * res.losses[0]
