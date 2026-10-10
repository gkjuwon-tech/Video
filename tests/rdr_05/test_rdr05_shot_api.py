"""RDR-05 render_shot_aov 이음매, 결정론, 입력 검증, 카메라 뒤 카드, 옵션 전달, 속도."""
import time

import numpy as np
import pytest

from buc.core import contracts as C
from buc.render import aov as A
from buc.render import quads
from buc.render.quads import RenderOpts, render_ldi
from rdr05_helpers import (H, I4, W, K_default, layer, make_obs, make_world, plane_layer, rand_tex, smooth_tex, translate,
                           yaw)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
HW = H * W


def _world():
    front = rand_tex(40, alpha=0.0)
    front[8:56, 16:64, 3] = 1.0
    back, _ = plane_layer((0.2, -0.1, -1.0), 5.0)
    return make_world([layer(front, 2.5), back])


def _shot(n=3):
    c2w = np.stack([yaw(translate(I4, (0.05 * i, 0.01 * i, 0.1 * i)), 2.0 * i) for i in range(n)])
    K = np.stack([K_default() for _ in range(n)])
    for i in range(n):                                   # 주점은 그대로, 초점거리만 프레임마다 다르다
        K[i][0, 0] *= 1 + 0.05 * i
        K[i][1, 1] *= 1 + 0.05 * i
    return C.Shot(c2w=c2w, t=np.arange(n) / 24.0, K=K, focus_m=np.full(n, 3.0), width=W, height=H, fps=24.0,
                  theta=C.Theta())


def test_render_shot_aov_extends_render_shot_bytewise():
    world, shot = _world(), _shot(3)
    ref = quads.render_shot(world, shot)
    fr = A.render_shot_aov(world, shot)
    assert isinstance(fr, C.Frames)
    assert fr.rgb.tobytes() == ref.rgb.tobytes() and fr.rgb.shape == (3, H, W, 3)
    assert fr.fps == ref.fps and fr.shot_hash == ref.shot_hash == shot.content_hash()
    assert fr.world_hash == ref.world_hash == world.content_hash()
    for k in ("depth", "alpha", "layer_id"):
        assert fr.aov[k].dtype == ref.aov[k].dtype and fr.aov[k].tobytes() == ref.aov[k].tobytes(), k
    assert set(fr.aov) == {"depth", "alpha", "layer_id", "normal", "atom_id", "flow", "flow_valid"}
    assert fr.aov["normal"].shape == (3, H, W, 3) and fr.aov["normal"].dtype == np.float32
    assert fr.aov["atom_id"].shape == (3, H, W) and fr.aov["atom_id"].dtype == np.int64
    assert fr.aov["flow"].shape == (3, H, W, 2) and fr.aov["flow"].dtype == np.float32
    assert fr.aov["flow_valid"].shape == (3, H, W) and fr.aov["flow_valid"].dtype == bool


def test_render_shot_aov_per_frame_values_and_flow_points_to_next_frame():
    world, shot = _world(), _shot(3)
    fr = A.render_shot_aov(world, shot)
    for i in range(3):
        nxt = None
        if i + 1 < 3:
            nxt = A.Cam(c2w=shot.c2w[i + 1], K=shot.K[i + 1])
        one = A.render_aov(world, c2w=shot.c2w[i], K=shot.K[i], width=W, height=H, flow_to=nxt)
        assert fr.aov["atom_id"][i].tobytes() == one.atom_id.tobytes(), i
        assert fr.aov["normal"][i].tobytes() == one.normal.tobytes(), i
        assert fr.aov["flow"][i].tobytes() == one.flow.tobytes(), i
        assert fr.aov["flow_valid"][i].tobytes() == one.flow_valid.tobytes(), i
    assert fr.aov["flow_valid"][0].mean() > 0.9 and np.abs(fr.aov["flow"][0]).max() > 0.5
    assert not fr.aov["flow_valid"][2].any() and (fr.aov["flow"][2] == 0).all()
    assert fr.aov["atom_id"][0].max() >= 0


def test_render_shot_aov_passes_opts():
    world, shot = _world(), _shot(2)
    opts = RenderOpts(bg=(0.2, 0.1, 0.0), edge_px=2.0, z_tol_rel=0.05)
    a = A.render_shot_aov(world, shot, opts)
    b = quads.render_shot(world, shot, opts)
    assert a.rgb.tobytes() == b.rgb.tobytes() and a.aov["depth"].tobytes() == b.aov["depth"].tobytes()
    assert a.aov["layer_id"].tobytes() == b.aov["layer_id"].tobytes()
    assert A.render_shot_aov(world, shot).rgb.tobytes() != a.rgb.tobytes()      # 옵션이 실제로 반영됨


def test_opts_forwarded_by_render_aov():
    world = _world()
    opts = RenderOpts(bg=(0.3, 0.3, 0.3), edge_px=2.5, max_footprint_px=4.0)
    kw = dict(c2w=translate(I4, (0.1, 0, 0)), K=K_default(), width=W, height=H, opts=opts)
    assert A.render_aov(world, **kw).rgb.tobytes() == render_ldi(world, **kw).rgb.tobytes()


def test_determinism_and_world_immutability():
    world = _world()
    h0 = world.content_hash()
    pose = yaw(translate(I4, (0.1, 0.02, 0.2)), 5.0)
    kw = dict(c2w=pose, K=K_default(), width=W, height=H, flow_to=A.Cam(c2w=translate(pose, (0.05, 0, 0)), K=K_default()))
    a, b = A.render_aov(world, **kw), A.render_aov(world, **kw)
    for name in ("rgb", "alpha", "depth", "layer_id", "normal", "atom_id", "flow", "flow_valid"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name
    assert world.content_hash() == h0
    for name in ("rgb", "alpha", "depth", "normal", "flow"):
        arr = getattr(a, name)
        assert arr.flags["C_CONTIGUOUS"] and np.isfinite(arr[np.isfinite(arr)]).all()
    assert not np.isnan(a.depth).any() and not np.isnan(a.flow).any() and not np.isnan(a.normal).any()


def test_camera_inside_or_behind_the_card_gives_empty_aov():
    world = make_world([layer(smooth_tex(), 3.0)])
    for fwd in (3.5, 5.0):                               # 카드가 카메라 뒤
        out = A.render_aov(world, c2w=translate(I4, (0, 0, fwd)), K=K_default(), width=W, height=H,
                           flow_to=A.Cam(c2w=I4, K=K_default()))
        assert (out.atom_id == -1).all() and (out.layer_id == -1).all() and (out.alpha < 1e-3).all()
        assert np.isinf(out.depth).all() and (out.normal == 0).all()
        assert not out.flow_valid.any() and (out.flow == 0).all()


def test_input_validation_matches_render_ldi():
    world = _world()
    good = dict(c2w=I4, K=K_default(), width=W, height=H)
    A.render_aov(world, **good)
    with pytest.raises(TypeError):
        A.render_aov(world, I4, K_default(), W, H)
    bad_pose = np.eye(4)
    bad_pose[:3, :3] *= 1.5
    cases = [
        dict(c2w=np.eye(3)), dict(c2w=bad_pose), dict(c2w=np.full((4, 4), np.nan)),
        dict(K=np.eye(2)), dict(K=np.zeros((3, 3))), dict(width=0), dict(height=-3), dict(width=96.5),
        dict(width=True), dict(k1=float("nan")), dict(k2=float("inf")),
    ]
    for c in cases:
        with pytest.raises(ValueError):
            A.render_aov(world, **{**good, **c})
    with pytest.raises(TypeError):
        A.render_shot_aov(world, _shot(2), RenderOpts(), 5)


def test_speed_160x90_three_layers_with_flow():
    w, h = 160, 90
    obs = make_obs(w=w, h=h)
    ls = [layer(rand_tex(i, w, h, alpha=0.8 if i < 2 else 1.0), 2.0 + 1.5 * i) for i in range(3)]
    world = C.World(obs=obs, layers=tuple(ls), theta_in=C.Theta(), level="L1", read_hash="r" * 8, prov=C.Prov())
    K = K_default(w, h)
    t0 = time.perf_counter()
    out = A.render_aov(world, c2w=translate(I4, (0.1, 0, 0.1)), K=K, width=w, height=h,
                       flow_to=A.Cam(c2w=translate(I4, (0.15, 0, 0.2)), K=K))
    dt = time.perf_counter() - t0
    assert dt < 15.0, dt
    assert out.atom_id.shape == (h, w) and (out.atom_id >= 0).mean() > 0.9
