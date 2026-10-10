"""RDR-08 API: 이름·시그니처, 지연 import(gsplat·CUDA 없음), 데이터 클래스 검증·도구, 옵션·카메라·backdrop 검증,
빈 입력, numpy() 형식."""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from buc.render import vol as V
from buc.render.vol import Strands, VolGaussians, VolOpts, VolRender, render_vol
from rdr08_helpers import K_of, make_strands, make_vol, rot_axis, to_np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W, H = 20, 16
K = K_of(40.0, W, H)
I4 = np.eye(4)


def _vol(n=3, dtype=torch.float64):
    rng = np.random.default_rng(1)
    mus = np.c_[rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n), rng.uniform(2, 3, n)]
    rots = np.stack([rot_axis(rng.normal(size=3), 30) for _ in range(n)])
    return make_vol(mus, rots, rng.uniform(0.03, 0.08, (n, 3)), rng.uniform(1, 5, n), rng.uniform(0, 1, (n, 3)),
                    dtype=dtype)


def _strands(dtype=torch.float64):
    pts = np.array([[[-0.2, 0.0, 2.0], [0.0, 0.05, 2.1], [0.2, 0.0, 2.2]]])
    return make_strands(pts, [[0.01, 0.01, 0.005]], [0.8], [[0.9, 0.9, 0.8]], dtype=dtype)


def test_public_names_and_signatures():
    for name in ("VolGaussians", "Strands", "VolOpts", "VolRender", "render_vol", "sggx_fiber", "sggx_surface",
                 "sggx_sigma", "sggx_D", "sggx_phase_specular", "alpha_to_tau", "tau_to_alpha", "matte_to_vol",
                 "strands_to_vol", "scatter_colors"):
        assert hasattr(V, name), name
    p = inspect.signature(render_vol).parameters
    for k in ("vol", "strands", "c2w", "K", "width", "height", "opts", "backdrop"):
        assert p[k].kind is inspect.Parameter.KEYWORD_ONLY, k
    p = inspect.signature(V.matte_to_vol).parameters
    for k in ("K", "c2w", "stride", "footprint", "thickness", "tangent", "roughness", "min_alpha", "dtype"):
        assert p[k].kind is inspect.Parameter.KEYWORD_ONLY, k
    assert p["stride"].default == 1 and p["footprint"].default == 1.0 and p["thickness"].default is None
    o = VolOpts()
    assert o.bg == (0.0, 0.0, 0.0) and o.near == 0.01 and o.far == 1e4 and o.filter_sigma == 0.5
    assert o.max_alpha == 0.999 and abs(o.min_alpha - 1 / 255) < 1e-12 and o.cutoff == 4.0


def test_import_is_lazy():
    code = ("import sys, torch\nimport buc.render.vol\n"
            "assert 'gsplat' not in sys.modules\n"
            "assert not torch.cuda.is_initialized()\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_volgaussians_tools():
    v = _vol(4)
    assert v.n == 4
    np.testing.assert_allclose(to_np(v.densities()), np.exp(to_np(v.log_density)), rtol=1e-12)
    R = to_np(v.rotations())
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.repeat(np.eye(3)[None], 4, 0), atol=1e-12)
    s = np.exp(to_np(v.log_scales))
    np.testing.assert_allclose(to_np(v.covariances()), R @ (s[:, :, None] ** 2 * R.transpose(0, 2, 1)), atol=1e-12)
    ps = v.params()
    assert [p is q for p, q in zip(ps, [v.means, v.quats, v.log_scales, v.log_density, v.colors])] == [True] * 5
    v.requires_grad_()
    assert all(p.requires_grad for p in v.params())
    d = v.detach()
    assert not any(p.requires_grad for p in d.params())
    sub = v.subset(torch.tensor([True, False, True, False]))
    assert sub.n == 2
    np.testing.assert_allclose(to_np(sub.means), to_np(v.means)[[0, 2]])
    np.testing.assert_allclose(to_np(sub.sggx), to_np(v.sggx)[[0, 2]])
    assert v.subset([3]).n == 1
    f32 = v.to(dtype=torch.float32)
    assert f32.means.dtype == torch.float32 and f32.sggx.dtype == torch.float32
    # numpy 입력 → tensor
    n = VolGaussians(means=np.zeros((1, 3)), quats=np.array([[1.0, 0, 0, 0]]), log_scales=np.zeros((1, 3)),
                     log_density=np.zeros(1), colors=np.zeros((1, 3)), sggx=np.eye(3)[None])
    assert isinstance(n.means, torch.Tensor)


def _vkw(**over):
    base = dict(means=torch.zeros(2, 3), quats=torch.tensor([[1.0, 0, 0, 0]] * 2), log_scales=torch.zeros(2, 3),
                log_density=torch.zeros(2), colors=torch.zeros(2, 3), sggx=torch.eye(3).repeat(2, 1, 1))
    base.update(over)
    return base


@pytest.mark.parametrize("over", [
    dict(means=torch.zeros(3, 3)),
    dict(log_scales=torch.zeros(2, 2)),
    dict(colors=torch.zeros(2, 3, dtype=torch.float64)),
    dict(log_density=torch.tensor([0.0, float("nan")])),
    dict(sggx=torch.tensor([[[1.0, 0.5, 0], [0, 1, 0], [0, 0, 1]]] * 2)),          # 비대칭
    dict(sggx=torch.tensor([[[1.0, 0, 0], [0, 1, 0], [0, 0, -0.1]]] * 2)),         # 비양정치
    dict(sggx=torch.eye(3).repeat(2, 1, 1)[:, :2]),
])
def test_volgaussians_validation(over):
    with pytest.raises(ValueError):
        VolGaussians(**_vkw(**over))


def test_strands_tools_and_validation():
    s = make_strands(np.zeros((2, 3, 3)) + np.arange(3)[None, :, None] * 0.1, [0.01, 0.02], [0.5, 1.0],
                     [[1, 0, 0], [0, 1, 0]])
    assert s.n == 2 and tuple(s.radius.shape) == (2, 3)
    np.testing.assert_allclose(to_np(s.radius)[1], 0.02)
    assert [p is q for p, q in zip(s.params(), [s.points, s.radius, s.opacity, s.colors])] == [True] * 4
    assert s.subset([1]).n == 1 and s.to(dtype=torch.float32).points.dtype == torch.float32
    pts = torch.zeros(1, 3, 3, dtype=torch.float64)
    good = dict(points=pts, radius=torch.full((1, 3), 0.01, dtype=torch.float64),
                opacity=torch.tensor([0.5], dtype=torch.float64), colors=torch.zeros(1, 3, dtype=torch.float64))
    Strands(**good)
    for over in (dict(points=torch.zeros(1, 1, 3, dtype=torch.float64), radius=torch.full((1, 1), 0.01,
                                                                                            dtype=torch.float64)),
                 dict(radius=torch.zeros(1, 3, dtype=torch.float64)),
                 dict(opacity=torch.tensor([1.5], dtype=torch.float64)),
                 dict(opacity=torch.tensor([0.0], dtype=torch.float64)),
                 dict(colors=torch.zeros(2, 3, dtype=torch.float64)),
                 dict(points=torch.full((1, 3, 3), float("inf"), dtype=torch.float64))):
        with pytest.raises(ValueError):
            Strands(**{**good, **over})


@pytest.mark.parametrize("kw", [dict(bg=(0, 0)), dict(near=0.0), dict(near=2.0, far=1.0), dict(filter_sigma=0.0),
                                dict(max_alpha=1.0), dict(max_alpha=0.0), dict(min_alpha=-0.1), dict(cutoff=0.0),
                                dict(bg=(0, float("nan"), 0))])
def test_volopts_validation(kw):
    with pytest.raises(ValueError):
        VolOpts(**kw)


def test_render_validation_and_empty():
    v = _vol()
    bad_R = np.eye(4)
    bad_R[0, 0] = 2.0
    for kw in (dict(c2w=bad_R, K=K, width=W, height=H), dict(c2w=I4, K=np.eye(3) * 0, width=W, height=H),
               dict(c2w=I4, K=K, width=0, height=H), dict(c2w=I4, K=K, width=W, height=2.5),
               dict(c2w=np.eye(3), K=K, width=W, height=H)):
        with pytest.raises(ValueError):
            render_vol(vol=v, **kw)
    with pytest.raises(ValueError):                         # dtype 불일치
        render_vol(vol=v, strands=_strands(torch.float32), c2w=I4, K=K, width=W, height=H)
    with pytest.raises(ValueError):                         # backdrop 모양 불일치
        render_vol(vol=v, c2w=I4, K=K, width=W, height=H,
                   backdrop=dict(rgb=np.zeros((H, W + 1, 3)), alpha=np.zeros((H, W + 1)), depth=np.zeros((H, W + 1))))
    out = render_vol(c2w=I4, K=K, width=W, height=H, opts=VolOpts(bg=(0.2, 0.4, 0.6)))
    assert isinstance(out, VolRender)
    np.testing.assert_allclose(to_np(out.rgb), np.broadcast_to([0.2, 0.4, 0.6], (H, W, 3)), atol=1e-7)
    assert to_np(out.alpha).max() == 0 and np.isinf(to_np(out.depth)).all()


def test_output_shapes_dtypes_numpy():
    for dt in (torch.float32, torch.float64):
        out = render_vol(vol=_vol(dtype=dt), strands=_strands(dt), c2w=I4, K=K, width=W, height=H)
        assert tuple(out.rgb.shape) == (H, W, 3) and tuple(out.alpha.shape) == (H, W)
        assert tuple(out.depth.shape) == (H, W)
        assert out.rgb.dtype == dt and out.alpha.dtype == dt and out.depth.dtype == dt
        d = out.numpy()
        assert set(d) >= {"rgb", "alpha", "depth"}
        for k in ("rgb", "alpha", "depth"):
            assert isinstance(d[k], np.ndarray) and d[k].dtype == np.float32
        a = d["alpha"]
        assert (a >= 0).all() and (a <= 1).all() and a.max() > 0.3
        assert not np.isnan(d["rgb"]).any() and not np.isnan(d["depth"]).any()
        m = a < 1e-3
        assert np.isinf(d["depth"][m]).all() and np.isfinite(d["depth"][~m]).all()
