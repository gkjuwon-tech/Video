"""RDR-03 API: 이름·시그니처, TSQ 검증·도구, 지연 import, 백엔드 선택, render 입력 검증, 결정론, 입력 불변, 순서 무관."""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from buc.render import splat2d, tsq as T
from buc.render.tsq import TSQ, TSQOpts, TSQRender, render
from rdr03_helpers import K_of, Scene, make_tsq, pose, random_scene, rot_axis, to_np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W, H = 32, 24
K = K_of(30.0, W, H)
C2W = pose(rot_axis([0.2, 1, 0], 4), [0.03, -0.02, 0.0])


def test_public_names_and_signatures():
    for name in ("PAGE", "LMIN", "TSQ", "TSQOpts", "TSQRender", "render", "resolve_backend", "load_extension",
                 "edge_falloff", "sample_page", "BackendUnavailable", "encode_bc7", "decode_bc7", "encode_bc5",
                 "decode_bc5", "PageBlob", "compress_pages", "decompress_pages", "decompress_nh"):
        assert hasattr(T, name), name
    assert T.PAGE == 8 and T.LMIN == 2.0 ** -16
    assert T.BackendUnavailable is splat2d.BackendUnavailable
    p = inspect.signature(render).parameters
    for k in ("c2w", "K", "width", "height", "opts"):
        assert p[k].kind is inspect.Parameter.KEYWORD_ONLY, k
    o = TSQOpts()
    assert tuple(o.bg) == (0.0, 0.0, 0.0) and o.near == 0.01 and o.far == 1e4 and o.backend == "auto"
    with pytest.raises(ValueError):
        TSQOpts(backend="gsplat")


def test_kernel_source_exists():
    path = os.path.join(ROOT, "buc", "render", "tsq_kernel.cu")
    src = open(path, encoding="utf-8").read()
    assert "__global__" in src and len(src) > 500


def test_import_is_lazy_no_cuda_init():
    code = ("import sys, torch\n"
            "import buc.render.tsq\n"
            "print(torch.cuda.is_initialized(), 'gsplat' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"], out.stdout


def test_tsq_validation():
    sc = random_scene(4, 0)
    t = make_tsq(sc)
    assert t.n == 4 and t.dtype == torch.float64 and t.pages.shape == (4, 8, 8, 3)
    good = {f: getattr(t, f) for f in ("means", "quats", "log_scales", "opacity_logits", "edge_n", "pages")}
    for f, bad in (("edge_n", torch.tensor([2.0, 1.99, 3.0, 4.0], dtype=torch.float64)),
                   ("pages", torch.zeros(4, 4, 4, 3, dtype=torch.float64)),
                   ("pages", torch.zeros(4, 8, 8, 4, dtype=torch.float64)),
                   ("means", torch.full((4, 3), float("nan"), dtype=torch.float64)),
                   ("opacity_logits", torch.zeros(4, dtype=torch.float32)),
                   ("edge_n", torch.full((3,), 2.0, dtype=torch.float64))):
        kw = dict(good)
        kw[f] = bad
        with pytest.raises(ValueError):
            TSQ(**kw)
    kw = {f: v.numpy() for f, v in good.items()}
    t2 = TSQ(**kw)
    assert isinstance(t2.means, torch.Tensor) and torch.equal(t2.pages, t.pages)


def test_tsq_tools(tmp_path):
    sc = random_scene(5, 1)
    t = make_tsq(sc, dtype=torch.float32)
    ps = t.params()
    assert [p.shape for p in ps] == [(5, 3), (5, 4), (5, 2), (5,), (5, 8, 8, 3), (5,)]
    assert ps[4] is t.pages and ps[5] is t.edge_n and ps[0] is t.means
    np.testing.assert_allclose(to_np(t.scales()), sc.scales, rtol=1e-6)
    np.testing.assert_allclose(to_np(t.opacities()), sc.opac, rtol=1e-5)
    np.testing.assert_allclose(to_np(t.rotations()), sc.rots, atol=1e-5)
    np.testing.assert_allclose(to_np(t.normals()), sc.rots[:, :, 2], atol=1e-5)
    s = t.subset(torch.tensor([0, 3]))
    assert s.n == 2 and torch.equal(s.pages[1], t.pages[3])
    m = t.subset(torch.tensor([True, False, True, False, False]))
    assert m.n == 2 and torch.equal(m.means[1], t.means[2])
    c = TSQ.cat([s, m])
    assert c.n == 4 and torch.equal(c.edge_n[2], t.edge_n[0])
    e = TSQ.empty()
    assert e.n == 0 and e.pages.shape == (0, 8, 8, 3) and e.dtype == torch.float32
    d = t.to(dtype=torch.float64)
    assert d.dtype == torch.float64 and d.pages.dtype == torch.float64
    t.requires_grad_(True)
    assert all(p.requires_grad for p in t.params())
    assert not any(p.requires_grad for p in t.detach().params())
    cl = t.clone()
    assert cl.pages is not t.pages and torch.equal(cl.pages, t.pages)
    path = tmp_path / "t.npz"
    t.detach().save(path)
    t3 = TSQ.load(path)
    for a, b in zip(t3.params(), t.params()):
        assert a.dtype == b.dtype and torch.equal(a, b.detach())


def test_from_and_to_surfels():
    from buc.render.splat2d import Surfels
    rng = np.random.default_rng(3)
    s = Surfels(means=torch.tensor(rng.normal(size=(3, 3))), quats=torch.tensor(rng.normal(size=(3, 4))),
                log_scales=torch.tensor(rng.normal(size=(3, 2))), opacity_logits=torch.tensor(rng.normal(size=3)),
                colors=torch.tensor(rng.uniform(size=(3, 3))))
    t = TSQ.from_surfels(s, edge_n=6.0)
    assert t.n == 3 and torch.all(t.edge_n == 6.0)
    assert torch.equal(t.pages, s.colors[:, None, None, :].expand(3, 8, 8, 3))
    assert torch.equal(t.means, s.means) and torch.equal(t.quats, s.quats)
    t2 = TSQ.from_surfels(s)
    assert torch.all(t2.edge_n == 2.0)
    pages = torch.tensor(rng.uniform(size=(3, 8, 8, 3)))
    t.pages = pages
    back = t.to_surfels()
    assert isinstance(back, Surfels)
    torch.testing.assert_close(back.colors, pages.mean(dim=(1, 2)))


def test_backend_selection(monkeypatch):
    monkeypatch.delenv("BUC_TSQ_BACKEND", raising=False)
    assert T.resolve_backend("auto", "cpu") == "torch"
    assert T.resolve_backend("torch", "cpu") == "torch"
    monkeypatch.setenv("BUC_TSQ_BACKEND", "torch")
    assert T.resolve_backend("auto", "cpu") == "torch"
    with pytest.raises(ValueError):
        T.resolve_backend("vulkan", "cpu")
    t = make_tsq(random_scene(3, 2), dtype=torch.float32)
    with pytest.raises(T.BackendUnavailable):
        render(t, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(backend="cuda"))
    r = render(t, c2w=C2W, K=K, width=W, height=H)
    assert r.backend == "torch"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU 전용 확인")
def test_load_extension_without_cuda_raises_backend_unavailable():
    with pytest.raises(T.BackendUnavailable):
        T.load_extension()


def test_render_input_validation():
    t = make_tsq(random_scene(3, 4))
    with pytest.raises(TypeError):
        render(t, C2W, K, W, H)  # noqa
    with pytest.raises(TypeError):
        render(object(), c2w=C2W, K=K, width=W, height=H)
    bad_cams = [dict(c2w=np.eye(3), K=K), dict(c2w=C2W, K=np.eye(2)), dict(c2w=np.full((4, 4), np.nan), K=K),
                dict(c2w=pose(np.diag([1.0, 1.0, 2.0])), K=K), dict(c2w=C2W, K=K_of(-3.0, W, H))]
    for cam in bad_cams:
        with pytest.raises(ValueError):
            render(t, width=W, height=H, **cam)
    for w, h in ((0, H), (W, 2.5), (W, -1)):
        with pytest.raises(ValueError):
            render(t, c2w=C2W, K=K, width=w, height=h)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_render_shapes_dtype_and_numpy(dtype):
    t = make_tsq(random_scene(6, 5), dtype=dtype)
    r = render(t, c2w=C2W, K=K, width=W, height=H)
    assert isinstance(r, TSQRender)
    assert r.rgb.shape == (H, W, 3) and r.alpha.shape == (H, W) and r.depth.shape == (H, W)
    assert r.normal.shape == (H, W, 3)
    for k in ("rgb", "alpha", "depth", "normal"):
        assert getattr(r, k).dtype == dtype
    n = r.numpy()
    assert set(n) >= {"rgb", "alpha", "depth", "normal"}
    assert all(v.dtype == np.float32 for v in n.values())
    assert float(r.alpha.max()) > 0.3 and not torch.isnan(r.rgb).any()


def test_empty_renders_background():
    e = TSQ.empty(dtype=torch.float64)
    r = render(e, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(bg=(0.2, 0.4, 0.6)))
    np.testing.assert_allclose(to_np(r.rgb), np.broadcast_to([0.2, 0.4, 0.6], (H, W, 3)), atol=1e-12)
    assert float(r.alpha.abs().max()) == 0.0
    assert torch.isinf(r.depth).all() and float(r.normal.abs().max()) == 0.0


def test_determinism_input_invariance_and_order_independence():
    sc = random_scene(10, 6)
    t = make_tsq(sc, dtype=torch.float32)
    before = [p.clone() for p in t.params()]
    r1 = render(t, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(bg=(0.1, 0.1, 0.1)))
    r2 = render(t, c2w=C2W, K=K, width=W, height=H, opts=TSQOpts(bg=(0.1, 0.1, 0.1)))
    for k in ("rgb", "alpha", "depth", "normal"):
        assert np.array_equal(r1.numpy()[k], r2.numpy()[k]), k
    for a, b in zip(before, t.params()):
        assert torch.equal(a, b)
    perm = np.random.default_rng(7).permutation(sc.n)
    sc2 = Scene(sc.means[perm], sc.rots[perm], sc.scales[perm], sc.opac[perm], sc.pages[perm], sc.edge_n[perm])
    r3 = render(make_tsq(sc2, dtype=torch.float32), c2w=C2W, K=K, width=W, height=H,
                opts=TSQOpts(bg=(0.1, 0.1, 0.1)))
    for k in ("rgb", "alpha", "normal"):
        np.testing.assert_allclose(r3.numpy()[k], r1.numpy()[k], atol=1e-5, err_msg=k)
    m = np.isfinite(r1.numpy()["depth"])
    np.testing.assert_allclose(r3.numpy()["depth"][m], r1.numpy()["depth"][m], rtol=1e-5)
