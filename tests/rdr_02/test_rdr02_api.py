"""RDR-02 API: 시그니처, 입력 검증, 결정론, 입력 불변, render_shot -> Frames."""
import inspect
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from buc.core import color, geom
from buc.core import contracts as C
from buc.render import quads
from buc.render.quads import QuadRender, RenderOpts, card_layer, render_ldi, render_shot
from rdr02_helpers import H, W, K_default, layer, make_obs, make_world, smooth_tex, translate, yaw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
I4 = np.eye(4)


def _world():
    a = smooth_tex(alpha=0.7)
    b = smooth_tex()
    return make_world([layer(a, 2.0), layer(b, 4.5)])


def test_public_names_and_signatures():
    for name in ("RenderOpts", "QuadRender", "card_layer", "render_ldi", "render_shot"):
        assert hasattr(quads, name), name
    params = inspect.signature(render_ldi).parameters
    assert list(params)[0] == "world"
    for kw in ("c2w", "K", "width", "height", "k1", "k2", "opts"):
        assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY, kw
    assert params["k1"].default == 0.0 and params["k2"].default == 0.0 and params["opts"].default is None
    p2 = inspect.signature(render_shot).parameters
    assert list(p2)[:2] == ["world", "shot"] and p2["opts"].default is None


def test_render_opts_defaults_frozen_and_validation():
    o = RenderOpts()
    assert tuple(o.bg) == (0.0, 0.0, 0.0) and o.edge_px == 1.0 and o.z_tol_rel == 0.1
    assert o.alpha_eps == 1e-3 and o.near_m == 1e-3 and o.max_footprint_px == 8.0
    with pytest.raises(Exception):
        o.edge_px = 2.0                                           # frozen
    with pytest.raises(TypeError):
        RenderOpts((0, 0, 0))                                     # kw_only
    for bad in (dict(edge_px=0.0), dict(edge_px=-1.0), dict(z_tol_rel=-0.1), dict(alpha_eps=-1e-3),
                dict(near_m=0.0), dict(max_footprint_px=0.5), dict(bg=(0.0, 0.0)), dict(bg=(0.0, 0.0, float("nan")))):
        with pytest.raises(ValueError):
            RenderOpts(**bad)


def test_card_layer_makes_a_valid_constant_depth_layer():
    rgba = smooth_tex()
    ly = card_layer(rgba, 2.5)
    assert isinstance(ly, C.LDILayer)
    assert ly.depth.dtype == np.float32 and ly.depth.shape == (H, W) and np.all(ly.depth == np.float32(2.5))
    np.testing.assert_array_equal(ly.rgba, rgba)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):                           # ContractError 도 ValueError 의 하위 클래스
            card_layer(rgba, bad)


def test_bad_arguments_raise_valueerror():
    w = make_world([card_layer(smooth_tex(), 3.0)])
    good = dict(c2w=I4, K=K_default(), width=W, height=H)
    render_ldi(w, **good)
    bad_cases = [
        dict(c2w=np.eye(3)), dict(c2w=I4 * 2.0), dict(c2w=np.where(np.eye(4) == 1, 1.0, np.nan)),
        dict(K=np.eye(2)), dict(K=np.array([[-80.0, 0, 48], [0, 80, 32], [0, 0, 1]])),
        dict(width=0), dict(height=-3), dict(width=96.5), dict(k1=float("nan")),
    ]
    for kw in bad_cases:
        with pytest.raises(ValueError):
            render_ldi(w, **{**good, **kw})
    with pytest.raises(TypeError):
        render_ldi(w, I4, K_default(), W, H)                       # 키워드 전용
    T = I4.copy()
    T[:3, :3] = geom.rodrigues(np.array([0.0, 0.3, 0.0])) * 1.01
    with pytest.raises(ValueError):
        render_ldi(w, **{**good, "c2w": T})                        # 비직교 회전


def test_determinism_bitwise_and_inputs_not_modified():
    w = _world()
    h0 = w.content_hash()
    snap = [(ly.rgba.copy(), ly.depth.copy()) for ly in w.layers]
    c2w = yaw(translate(I4, (0.09, -0.02, 0.3)), 5.0)
    st = np.random.get_state()[1].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = render_ldi(w, c2w=c2w, K=K_default(), width=W, height=H)
        b = render_ldi(w, c2w=c2w, K=K_default(), width=W, height=H)
    assert isinstance(a, QuadRender)
    for f in ("rgb", "alpha", "depth", "layer_id"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), f
    assert w.content_hash() == h0
    for (r0, d0), ly in zip(snap, w.layers):
        assert ly.rgba.tobytes() == r0.tobytes() and ly.depth.tobytes() == d0.tobytes()
    assert (np.random.get_state()[1] == st).all(), "전역 numpy RNG 를 건드렸다"
    with pytest.raises(Exception):
        a.rgb = None                                              # QuadRender 는 frozen


def test_c2w_and_K_accept_lists_and_float32_and_do_not_change_the_result():
    w = _world()
    c2w = translate(I4, (0.05, 0, 0.1))
    ref = render_ldi(w, c2w=c2w, K=K_default(), width=W, height=H)
    alt = render_ldi(w, c2w=c2w.astype(np.float32).tolist(), K=K_default().astype(np.float32), width=W, height=H)
    np.testing.assert_allclose(alt.rgb, ref.rgb, atol=2e-3)


def test_module_import_is_cpu_only_and_side_effect_free():
    code = ("import numpy as np; s=np.random.get_state()[1].copy(); import buc.render.quads as q;"
            "assert (np.random.get_state()[1]==s).all(); assert hasattr(q,'render_ldi'); print('ok')")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr[-800:]


def _shot(n=4, w=W, h=H, dx=0.04):
    c2w = np.stack([translate(I4, (dx * i, 0.0, 0.05 * i)) for i in range(n)])
    return C.Shot(c2w=c2w, t=np.arange(n) / 24.0, K=np.stack([K_default(w, h)] * n), focus_m=np.full(n, 3.0),
                  width=w, height=h, fps=24.0, theta=C.Theta())


def test_render_shot_returns_valid_frames_matching_per_frame_render():
    w = _world()
    shot = _shot()
    assert shot.first_pose_matches(w.obs)
    opts = RenderOpts(bg=(0.05, 0.05, 0.05))
    fr = render_shot(w, shot, opts)
    assert isinstance(fr, C.Frames)
    fr.validate()
    assert fr.rgb.shape == (4, H, W, 3) and fr.rgb.dtype == np.float32
    assert fr.rgb.min() >= 0.0 and fr.rgb.max() <= 1.0
    assert fr.fps == 24.0
    assert fr.shot_hash == shot.content_hash() and fr.world_hash == w.content_hash()
    assert set(fr.aov) >= {"depth", "alpha", "layer_id"}
    assert fr.aov["depth"].shape == (4, H, W) and fr.aov["alpha"].shape == (4, H, W)
    assert fr.aov["layer_id"].shape == (4, H, W)
    assert fr.aov["depth"].dtype == np.float32 and fr.aov["layer_id"].dtype == np.int16
    for i in range(4):
        one = render_ldi(w, c2w=shot.c2w[i], K=shot.K[i], width=W, height=H, opts=opts)
        np.testing.assert_allclose(fr.rgb[i], color.linear_to_display(one.rgb), atol=1e-6)     # rgb 는 표시값(sRGB 부호화)
        np.testing.assert_allclose(fr.aov["alpha"][i], one.alpha, atol=1e-7)
        np.testing.assert_array_equal(fr.aov["depth"][i], one.depth)
        np.testing.assert_array_equal(fr.aov["layer_id"][i], one.layer_id)
    assert render_shot(w, shot, opts).rgb.tobytes() == fr.rgb.tobytes()


def test_render_shot_first_frame_reproduces_the_observed_layers():
    w = make_world([card_layer(smooth_tex(), 3.0)], make_obs())
    fr = render_shot(w, _shot(n=2))
    exp = color.linear_to_display(smooth_tex()[..., :3])
    np.testing.assert_allclose(fr.rgb[0], exp, atol=4e-3)


def test_render_shot_uses_per_frame_intrinsics():
    w = make_world([card_layer(smooth_tex(), 3.0)])
    K = np.stack([K_default(), K_default(fov=40.0)])              # 두 번째 프레임은 줌 인
    shot = C.Shot(c2w=np.stack([I4, I4]), t=np.arange(2) / 24.0, K=K, focus_m=np.full(2, 3.0), width=W, height=H,
                  fps=24.0, theta=C.Theta())
    fr = render_shot(w, shot)
    one = render_ldi(w, c2w=I4, K=K[1], width=W, height=H)
    np.testing.assert_allclose(fr.rgb[1], color.linear_to_display(one.rgb), atol=1e-6)
    assert np.abs(fr.rgb[1] - fr.rgb[0]).mean() > 0.01
