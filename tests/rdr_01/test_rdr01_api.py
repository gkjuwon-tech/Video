"""RDR-01 API: 이름·시그니처, Surfels 검증·도구, 지연 import, 백엔드 선택, render 입력 검증, 결정론, 입력 불변."""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from buc.render import splat2d
from buc.render.splat2d import (BackendUnavailable, SplatOpts, SplatRender, Surfels, render,
                                resolve_backend)
from rdr01_helpers import K_of, fronto, make_surfels, pose, rot_axis

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W, H = 32, 24
K = K_of(30.0, W, H)


def _scene(dtype=torch.float32, n=6, seed=0):
    rng = np.random.default_rng(seed)
    means = np.c_[rng.uniform(-0.4, 0.4, n), rng.uniform(-0.3, 0.3, n), rng.uniform(2.0, 4.0, n)]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, 60)) for _ in range(n)])
    return make_surfels(means, rots, rng.uniform(0.08, 0.25, (n, 2)), rng.uniform(0.3, 0.9, n),
                        rng.uniform(0.05, 0.95, (n, 3)), dtype=dtype)


def test_public_names_and_signatures():
    from buc.render import fit
    for name in ("Surfels", "SplatOpts", "SplatRender", "render", "resolve_backend", "BackendUnavailable",
                 "lod_lambda", "render_dual_cut", "DualCutRender"):
        assert hasattr(splat2d, name), name
    for name in ("ColmapPoints", "read_colmap_points3d", "init_from_points", "init_from_rgbd", "FitView",
                 "FitOpts", "FitResult", "fit", "psnr"):
        assert hasattr(fit, name), name
    p = inspect.signature(render).parameters
    for k in ("c2w", "K", "width", "height", "opts"):
        assert p[k].kind is inspect.Parameter.KEYWORD_ONLY, k
    p = inspect.signature(splat2d.render_dual_cut).parameters
    assert p["tile"].default == 16 and tuple(p["band"].default) == (0.05, 0.95)
    assert issubclass(BackendUnavailable, RuntimeError)
    o = SplatOpts()
    assert tuple(o.bg) == (0.0, 0.0, 0.0) and o.near == 0.01 and o.far == 1e4 and o.backend == "auto"
    fo = fit.FitOpts()
    assert fo.iters == 300 and fo.lambda_ssim == 0.2 and fo.batch_views == 0 and fo.seed == 0


def test_import_is_lazy_no_gsplat_no_cuda_init():
    code = ("import sys, torch\n"
            "import buc.render.splat2d, buc.render.fit\n"
            "print('gsplat' in sys.modules, torch.cuda.is_initialized())\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"], out.stdout


def test_surfels_validation_and_shapes():
    s = _scene()
    assert s.n == 6
    assert s.scales().shape == (6, 2) and s.opacities().shape == (6,)
    R = s.rotations()
    assert R.shape == (6, 3, 3)
    eye = torch.eye(3).expand(6, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(6), atol=1e-5)
    assert torch.allclose(s.normals(), R[:, :, 2])
    ps = s.params()
    assert [p is q for p, q in zip(ps, [s.means, s.quats, s.log_scales, s.opacity_logits, s.colors])] == [True] * 5
    # 정규화 안 된 쿼터니언도 같은 회전
    s2 = s.clone()
    s2.quats.mul_(3.7)
    assert torch.allclose(s2.rotations(), R, atol=1e-5)
    good = {f: getattr(s, f).clone() for f in ("means", "quats", "log_scales", "opacity_logits", "colors")}
    with pytest.raises(TypeError):
        Surfels(*good.values())
    bad = dict(good, means=good["means"][:5])
    with pytest.raises(ValueError):
        Surfels(**bad)
    bad = dict(good, log_scales=torch.zeros(6, 3))
    with pytest.raises(ValueError):
        Surfels(**bad)
    nan = good["colors"].clone()
    nan[2, 1] = float("nan")
    with pytest.raises(ValueError):
        Surfels(**dict(good, colors=nan))
    with pytest.raises(ValueError):
        Surfels(**dict(good, colors=good["colors"].double()))
    # numpy 입력은 tensor로
    s3 = Surfels(**{k: v.numpy() for k, v in good.items()})
    assert isinstance(s3.means, torch.Tensor) and s3.means.dtype == torch.float32


def test_surfels_tools_roundtrip(tmp_path):
    s = _scene(dtype=torch.float64)
    p = tmp_path / "s.npz"
    s.save(p)
    t = Surfels.load(p)
    for a, b in zip(s.params(), t.params()):
        assert a.dtype == b.dtype and torch.equal(a, b)
    e = Surfels.empty()
    assert e.n == 0 and e.means.shape == (0, 3) and e.means.dtype == torch.float32
    c = Surfels.cat([s.subset(torch.tensor([0, 1])), s.subset(torch.tensor([2, 3, 4, 5]))])
    for a, b in zip(s.params(), c.params()):
        assert torch.equal(a, b)
    m = torch.tensor([True, False, True, False, False, True])
    assert s.subset(m).n == 3 and torch.equal(s.subset(m).means, s.means[m])
    f = s.to(dtype=torch.float32)
    assert f.means.dtype == torch.float32 and f.n == 6
    g = s.clone().requires_grad_(True)
    assert all(p.requires_grad for p in g.params())
    assert not any(p.requires_grad for p in g.detach().params())
    c2 = s.clone()
    c2.means.add_(1.0)
    assert not torch.equal(c2.means, s.means)


def test_backend_resolution(monkeypatch):
    monkeypatch.delenv("BUC_SPLAT_BACKEND", raising=False)
    assert resolve_backend("auto", "cpu") == "torch"
    assert resolve_backend("torch", "cpu") == "torch"
    with pytest.raises(BackendUnavailable):
        resolve_backend("gsplat", "cpu")
    with pytest.raises(ValueError):
        resolve_backend("optix", "cpu")
    monkeypatch.setenv("BUC_SPLAT_BACKEND", "torch")
    assert resolve_backend("auto", "cpu") == "torch"
    s = _scene()
    r = render(s, c2w=np.eye(4), K=K, width=W, height=H)
    assert isinstance(r, SplatRender) and r.backend == "torch"
    with pytest.raises(BackendUnavailable):
        render(s, c2w=np.eye(4), K=K, width=W, height=H, opts=SplatOpts(backend="gsplat"))


def test_opts_validation():
    for kw in (dict(near=0.0), dict(near=-1.0), dict(far=0.005), dict(bg=(0, 0)), dict(bg=(0, float("nan"), 0)),
               dict(backend="nope")):
        with pytest.raises(ValueError):
            SplatOpts(**kw)


def test_render_input_validation():
    s = _scene()
    ok = dict(c2w=np.eye(4), K=K, width=W, height=H)
    render(s, **ok)
    with pytest.raises(TypeError):
        render(s, np.eye(4), K, W, H)
    bads = [dict(width=0), dict(height=-3), dict(width=32.5), dict(K=K_of(-30.0, W, H)), dict(K=np.eye(2)),
            dict(c2w=np.eye(3)), dict(c2w=np.diag([1.0, 2.0, 1.0, 1.0])), dict(c2w=np.full((4, 4), np.nan))]
    for b in bads:
        with pytest.raises(ValueError):
            render(s, **dict(ok, **b))
    # 리스트·float32 카메라 허용
    r = render(s, c2w=np.eye(4).tolist(), K=K.astype(np.float32), width=W, height=H)
    assert r.rgb.shape == (H, W, 3)


def test_output_types_shapes_dtypes():
    for dt in (torch.float32, torch.float64):
        s = _scene(dtype=dt)
        r = render(s, c2w=pose(rot_axis([0, 1, 0], 5), [0.05, 0, 0]), K=K, width=W, height=H)
        assert r.rgb.shape == (H, W, 3) and r.alpha.shape == (H, W)
        assert r.depth.shape == (H, W) and r.normal.shape == (H, W, 3)
        for x in (r.rgb, r.alpha, r.depth, r.normal):
            assert isinstance(x, torch.Tensor) and x.dtype == dt
        a = r.alpha
        assert float(a.min()) >= 0 and float(a.max()) <= 1
        assert bool(torch.isfinite(r.rgb).all()) and bool(torch.isfinite(r.normal).all())
        assert not bool(torch.isnan(r.depth).any())
        low = a < 1e-3
        assert bool(torch.isinf(r.depth[low]).all()) and bool((r.depth[~low] > 0).all())
        assert bool((r.normal[low] == 0).all())
        nn = r.normal[~low].norm(dim=-1)
        assert torch.allclose(nn, torch.ones_like(nn), atol=1e-4)
        d = r.numpy()
        assert set(d) >= {"rgb", "alpha", "depth", "normal"}
        assert all(v.dtype == np.float32 and isinstance(v, np.ndarray) for v in d.values())


def test_empty_and_bg():
    bg = (0.2, 0.4, 0.6)
    r = render(Surfels.empty(), c2w=np.eye(4), K=K, width=W, height=H, opts=SplatOpts(bg=bg))
    assert torch.allclose(r.rgb, torch.tensor(bg).expand(H, W, 3), atol=1e-7)
    assert float(r.alpha.abs().max()) == 0.0 and bool(torch.isinf(r.depth).all())


def test_bg_composite_identity():
    s = _scene()
    r0 = render(s, c2w=np.eye(4), K=K, width=W, height=H)
    bg = (0.3, 0.1, 0.9)
    r1 = render(s, c2w=np.eye(4), K=K, width=W, height=H, opts=SplatOpts(bg=bg))
    exp = r0.rgb + (1 - r0.alpha)[..., None] * torch.tensor(bg)
    assert torch.allclose(r1.rgb, exp, atol=1e-6)
    assert torch.equal(r1.alpha, r0.alpha)


def test_determinism_and_input_untouched():
    s = _scene()
    before = [p.clone() for p in s.params()]
    c2w = pose(rot_axis([1, 2, 0], 7), [0.1, -0.05, 0.2])
    a = render(s, c2w=c2w, K=K, width=W, height=H).numpy()
    b = render(s, c2w=c2w, K=K, width=W, height=H).numpy()
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k
    for p, q in zip(before, s.params()):
        assert torch.equal(p, q)


def test_input_order_invariance():
    s = _scene(n=8, seed=3)
    perm = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    a = render(s, c2w=np.eye(4), K=K, width=W, height=H)
    b = render(s.subset(perm), c2w=np.eye(4), K=K, width=W, height=H)
    assert float(a.alpha.max()) > 0.5
    for k in ("rgb", "alpha", "normal"):
        assert torch.allclose(getattr(a, k), getattr(b, k), atol=1e-6), k
    fin = torch.isfinite(a.depth)
    assert torch.equal(fin, torch.isfinite(b.depth))
    assert torch.allclose(a.depth[fin], b.depth[fin], atol=1e-5)


def test_no_warnings_edge_cases():
    import warnings
    c = np.eye(4)
    # 카메라 뒤, near 안쪽, edge-on, 아주 큰 surfel 을 한 장면에
    rots = np.stack([np.eye(3), np.eye(3), rot_axis([0, 1, 0], 90), np.eye(3)])
    s = make_surfels([[0, 0, -2.0], [0, 0, 0.005], [0.0, 0.0, 3.0], [0.1, 0.1, 50.0]], rots,
                     [[0.2, 0.2], [0.2, 0.2], [0.3, 0.3], [100.0, 100.0]], [0.8, 0.8, 0.8, 0.5],
                     np.full((4, 3), 0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = render(s, c2w=c, K=K, width=W, height=H, opts=SplatOpts(far=1e3))
    for x in (r.rgb, r.alpha, r.normal):
        assert bool(torch.isfinite(x).all())
    assert not bool(torch.isnan(r.depth).any())
