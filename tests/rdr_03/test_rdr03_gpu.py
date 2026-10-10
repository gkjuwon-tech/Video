"""RDR-03 GPU: tsq_kernel.cu (cuda 백엔드) ↔ torch 정본 수치·기울기 일치, auto 선택, 결정론."""
import numpy as np
import pytest
import torch

from buc.render import tsq as T
from buc.render.tsq import TSQOpts, render
from rdr03_helpers import K_of, make_tsq, pose, random_scene, rot_axis

pytestmark = pytest.mark.gpu

W, H = 96, 64
K = K_of(80.0, W, H)
C2W = pose(rot_axis([0.2, 1, 0], 5), [0.05, -0.02, 0.0])
FIELDS = ("means", "quats", "log_scales", "opacity_logits", "pages", "edge_n")


def _scene(seed=0, n=300):
    return make_tsq(random_scene(n, seed, scale=(0.03, 0.15), spread=1.0, depth=(2.5, 5.0)),
                    torch.float32).to(device="cuda")


def test_extension_loads_and_auto_picks_cuda(monkeypatch):
    monkeypatch.delenv("BUC_TSQ_BACKEND", raising=False)
    assert torch.cuda.is_available()
    assert T.load_extension() is not None
    assert T.resolve_backend("auto", "cuda") == "cuda"


def test_cuda_matches_torch_reference():
    t = _scene()
    a = render(t, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(backend="cuda", bg=(0.1, 0.2, 0.3)))
    b = render(t, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(backend="torch", bg=(0.1, 0.2, 0.3)))
    assert a.backend == "cuda" and b.backend == "torch"
    na, nb = a.numpy(), b.numpy()
    for k in ("rgb", "alpha"):
        assert np.abs(na[k] - nb[k]).max() <= 3e-3, k
    m = np.isfinite(nb["depth"]) & (nb["alpha"] > 0.05)
    assert np.array_equal(np.isfinite(na["depth"])[m], m[m])
    np.testing.assert_allclose(na["depth"][m], nb["depth"][m], rtol=1e-3)


def test_cuda_gradients_match_torch():
    rng = np.random.default_rng(1)
    wr = torch.tensor(rng.normal(size=(H, W, 3)), dtype=torch.float32, device="cuda")
    grads = {}
    for be in ("cuda", "torch"):
        t = _scene(2).requires_grad_(True)
        r = render(t, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(backend=be))
        ((r.rgb * wr).sum() + r.alpha.sum()).backward()
        grads[be] = {f: getattr(t, f).grad.detach().double().reshape(-1) for f in FIELDS}
    for f in FIELDS:
        a, b = grads["cuda"][f], grads["torch"][f]
        assert torch.isfinite(a).all(), f
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-30))
        assert cos > 0.99, (f, cos)


def test_cuda_deterministic():
    t = _scene(3)
    o = TSQOpts(backend="cuda")
    a = render(t, c2w=C2W, K=K, width=W, height=H, opts=o).numpy()
    b = render(t, c2w=C2W, K=K, width=W, height=H, opts=o).numpy()
    for k in ("rgb", "alpha", "depth"):
        assert np.array_equal(a[k], b[k]), k
