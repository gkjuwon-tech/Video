"""RDR-07 API: 자료형 검증, 키워드 전용, 지연 import, 결정론, 입력 불변, 무거울 동일성, shot 이음매."""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from buc.core import color
from buc.core import contracts as C
from buc.render.mirror import Mirror, MirrorOpts, MirrorRender, render_mirror_shot, render_mirrors
from buc.render.quads import RenderOpts, render_ldi
from rdr07_helpers import H, I4, W, K_default, const_rgba, layer, make_world, smooth_tex, translate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _floor():
    return Mirror(center=(0, 1.0, 0), normal=(0, -1, 0), reflectance=(0.7, 0.8, 0.9))


def _world():
    return make_world([layer(smooth_tex(), 4.0)])


# ------------------------------------------------------------------------------------------------ Mirror / opts
def test_mirror_normalizes_and_exposes_plane_offset():
    m = Mirror(center=(1.0, 2.0, 3.0), normal=(0, 0, -2.0), u_axis=(1.0, 0, 0.5), half_size=(0.5, 0.25))
    np.testing.assert_allclose(np.asarray(m.normal), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(np.asarray(m.u_axis), [1, 0, 0], atol=1e-12)   # 법선 성분 제거 후 단위화
    assert m.d == pytest.approx(-3.0)
    assert m.two_sided is False
    assert tuple(m.reflectance) == pytest.approx((0.9, 0.9, 0.9))


@pytest.mark.parametrize("kw", [
    dict(normal=(0, 0, 0)),
    dict(normal=(0, np.nan, 1)),
    dict(center=(0, np.inf, 0)),
    dict(reflectance=(1.2, 0.5, 0.5)),
    dict(reflectance=(-0.1, 0.5, 0.5)),
    dict(reflectance=(0.5, 0.5)),
    dict(half_size=(0.0, 1.0), u_axis=(1, 0, 0)),
    dict(half_size=(1.0, -1.0), u_axis=(1, 0, 0)),
    dict(half_size=(1.0, 1.0)),                         # u_axis 없음
    dict(half_size=(1.0, 1.0), u_axis=(0, 0, 3)),       # 법선과 평행
])
def test_mirror_validation(kw):
    base = dict(center=(0, 0, 3), normal=(0, 0, -1))
    base.update(kw)
    with pytest.raises(ValueError):
        Mirror(**base)


def test_mirror_is_keyword_only_and_frozen():
    with pytest.raises(TypeError):
        Mirror((0, 0, 3), (0, 0, -1))
    m = _floor()
    with pytest.raises(Exception):
        m.two_sided = True


@pytest.mark.parametrize("kw", [
    dict(max_depth=-1), dict(max_depth=17), dict(max_renders=0), dict(min_weight=1.0), dict(min_weight=-0.1),
    dict(clip_eps_m=-1e-3), dict(terminal_rgb=(0, 0)), dict(terminal_rgb=(0, np.nan, 0)),
])
def test_opts_validation(kw):
    with pytest.raises(ValueError):
        MirrorOpts(**kw)


def test_opts_defaults():
    o = MirrorOpts()
    assert (o.max_depth, o.max_renders) == (2, 64)
    assert o.min_weight == pytest.approx(1e-3)
    assert o.clip_eps_m == pytest.approx(1e-4)
    assert o.quad is None
    MirrorOpts(max_depth=0)
    MirrorOpts(max_depth=16)


def test_render_mirrors_argument_errors():
    w = _world()
    K = K_default()
    with pytest.raises(TypeError):
        render_mirrors(w, [_floor()], I4, K, W, H)
    with pytest.raises(TypeError):
        render_mirrors(w, ["not a mirror"], c2w=I4, K=K, width=W, height=H)
    with pytest.raises(ValueError):
        render_mirrors(w, [_floor()], c2w=np.eye(3), K=K, width=W, height=H)
    with pytest.raises(ValueError):
        render_mirrors(w, [_floor()], c2w=I4, K=K, width=0, height=H)
    bad = I4.copy()
    bad[:3, :3] = np.diag([1.0, 1.0, -1.0])                       # det -1
    with pytest.raises(ValueError):
        render_mirrors(w, [_floor()], c2w=bad, K=K, width=W, height=H)


# ------------------------------------------------------------------------------------------------ import
def test_module_import_does_not_import_warp_or_cuda():
    code = ("import sys; import buc.render.mirror as m; "
            "assert 'warp' not in sys.modules, 'warp imported eagerly'; "
            "b = m.available_backends(); assert 'numpy' in b and isinstance(b, tuple); print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


# ------------------------------------------------------------------------------------------------ results
def test_result_types_shapes_and_determinism():
    w = _world()
    c2w = translate(I4, (0.05, -0.1, 0.2))
    st = np.random.get_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a = render_mirrors(w, [_floor()], c2w=c2w, K=K_default(), width=W, height=H)
    b = render_mirrors(w, [_floor()], c2w=c2w, K=K_default(), width=W, height=H)
    s2 = np.random.get_state()
    assert st[0] == s2[0] and np.array_equal(st[1], s2[1])
    assert isinstance(a, MirrorRender)
    spec = dict(rgb=((H, W, 3), np.float32), alpha=((H, W), np.float32), depth=((H, W), np.float32),
                layer_id=((H, W), np.int16), mirror_id=((H, W), np.int16), bounce=((H, W), np.int8),
                virtual_depth=((H, W), np.float32))
    for k, (shape, dt) in spec.items():
        x, y = getattr(a, k), getattr(b, k)
        assert x.shape == shape and x.dtype == dt, k
        assert x.tobytes() == y.tobytes(), f"{k} 비결정"
        if x.dtype.kind == "f":
            assert not np.isnan(x).any(), k
    assert isinstance(a.n_renders, int) and a.n_renders == b.n_renders >= 2
    assert (a.mirror_id == 0).sum() > 100
    assert set(np.unique(a.mirror_id)) <= {-1, 0}


def test_inputs_not_modified():
    w = _world()
    before = [ly.rgba.copy() for ly in w.layers], [ly.depth.copy() for ly in w.layers]
    m = _floor()
    c2w = translate(I4, (0.0, -0.1, 0.0))
    c0 = c2w.copy()
    K = K_default()
    K0 = K.copy()
    render_mirrors(w, [m], c2w=c2w, K=K, width=W, height=H)
    for ly, r, d in zip(w.layers, *before):
        assert np.array_equal(ly.rgba, r) and np.array_equal(ly.depth, d)
    assert np.array_equal(c2w, c0) and np.array_equal(K, K0)


@pytest.mark.parametrize("mirrors", [[], [Mirror(center=(0, 0, -2), normal=(0, 0, 1))]])
def test_no_visible_mirror_is_byte_identical_to_render_ldi(mirrors):
    """거울이 없거나(빈 목록) 카메라 뒤에만 있으면 RDR-02 와 바이트 동일."""
    w = make_world([layer(smooth_tex(), 3.0), layer(const_rgba((0.2, 0.3, 0.4), 1.0), 6.0)])
    c2w = translate(I4, (0.1, 0.0, 0.1))
    q = RenderOpts(bg=(0.1, 0.2, 0.3))
    ref = render_ldi(w, c2w=c2w, K=K_default(), width=W, height=H, opts=q)
    r = render_mirrors(w, mirrors, c2w=c2w, K=K_default(), width=W, height=H, opts=MirrorOpts(quad=q))
    for k in ("rgb", "alpha", "depth", "layer_id"):
        assert getattr(r, k).tobytes() == getattr(ref, k).tobytes(), k
    assert (r.mirror_id == -1).all() and (r.bounce == 0).all()
    np.testing.assert_array_equal(r.virtual_depth, r.depth)
    assert r.n_renders == 1


def test_render_mirror_shot_frames_seam():
    w = _world()
    n = 2
    c2w = np.stack([translate(I4, (0.04 * i, -0.1, 0.05 * i)) for i in range(n)])
    shot = C.Shot(c2w=c2w, t=np.arange(n) / 24.0, K=np.stack([K_default()] * n), focus_m=np.full(n, 3.0),
                  width=W, height=H, fps=24.0, theta=C.Theta())
    ms = [_floor()]
    fr = render_mirror_shot(w, ms, shot)
    assert isinstance(fr, C.Frames)
    assert fr.rgb.shape == (n, H, W, 3) and fr.fps == 24.0
    assert fr.shot_hash == shot.content_hash() and fr.world_hash == w.content_hash()
    keys = {"depth": np.float32, "alpha": np.float32, "layer_id": np.int16, "mirror_id": np.int16,
            "bounce": np.int8, "virtual_depth": np.float32}
    assert set(fr.aov) == set(keys)
    for k, dt in keys.items():
        assert fr.aov[k].shape == (n, H, W) and fr.aov[k].dtype == dt, k
    for i in range(n):
        one = render_mirrors(w, ms, c2w=c2w[i], K=K_default(), width=W, height=H)
        np.testing.assert_allclose(fr.rgb[i], color.linear_to_display(one.rgb), atol=1e-6)
        np.testing.assert_array_equal(fr.aov["mirror_id"][i], one.mirror_id)
        np.testing.assert_array_equal(fr.aov["bounce"][i], one.bounce)
        np.testing.assert_array_equal(fr.aov["virtual_depth"][i], one.virtual_depth)
