"""RDR-05 법선·광류 AOV: 해석 오라클(카드·기울어진 평면·회전 카메라), geom 오라클, 무효 규칙."""
import numpy as np
import pytest

from buc.core import geom
from buc.render import aov as A
from rdr05_helpers import (FX, H, I4, W, K_default, const_layer, layer, make_obs, make_world, plane_layer, rand_tex,
                           rot_o2t, smooth_tex, translate, world_to_cam, yaw)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
HW = H * W


def _aov(world, c2w=I4, **kw):
    return A.render_aov(world, c2w=c2w, K=K_default(), width=W, height=H, **kw)


# ---------------------------------------------------------------------------------------------------------------
# 법선
# ---------------------------------------------------------------------------------------------------------------
def test_card_normal_faces_camera_and_zero_where_empty():
    rgba = smooth_tex()
    rgba[:, 60:, 3] = 0.0                                   # 오른쪽은 비어 있다
    out = _aov(make_world([layer(rgba, 3.0)]))
    assert out.normal.dtype == np.float32 and out.normal.shape == (H, W, 3)
    cov = out.atom_id >= 0
    assert cov[:, :55].all() and not cov[:, 62:].any()
    assert np.allclose(out.normal[cov], [0, 0, -1], atol=1e-6)
    assert (out.normal[~cov] == 0).all()


@pytest.mark.parametrize("yaw_deg", [0.0, 20.0, -35.0])
def test_card_normal_rotates_with_target_camera(yaw_deg):
    obs = make_obs()
    world = make_world([layer(smooth_tex(), 4.0)], obs=obs)
    pose = yaw(translate(I4, (0.2, 0.05, -0.1)), yaw_deg)
    out = _aov(world, pose)
    expect = rot_o2t(obs, pose) @ np.array([0.0, 0.0, -1.0])
    sel = out.alpha > 0.98
    assert sel.sum() > 1000
    assert np.allclose(out.normal[sel], expect, atol=2e-5)
    assert np.allclose(np.linalg.norm(out.normal[out.atom_id >= 0], axis=-1), 1.0, atol=1e-5)


@pytest.mark.parametrize("k1", [0.0, -0.1])
def test_tilted_plane_normal_exact_even_with_obs_distortion(k1):
    obs = make_obs(k1=k1)
    ly, n_face = plane_layer((0.3, -0.2, -1.0), 4.0, obs=obs)
    world = make_world([ly], obs=obs)
    pose = yaw(translate(I4, (-0.15, 0.1, 0.2)), -10.0)
    out = _aov(world, pose, k1=0.0)
    expect = rot_o2t(obs, pose) @ n_face
    sel = out.alpha > 0.98
    assert sel.mean() > 0.5
    err = np.abs(out.normal[sel] - expect).max()
    assert err < 2e-4, err
    # 평면 법선이 정말 기울어 있다는 확인(카드 법선과 구별됨)
    assert np.linalg.norm(expect - rot_o2t(obs, pose) @ np.array([0, 0, -1.0])) > 0.3


def test_normals_follow_the_dominant_layer():
    front = const_layer((0.9, 0.1, 0.1), 0.0, 2.0)
    rgba = front.rgba.copy()
    rgba[:, :40, 3] = 1.0
    front = layer(rgba, 2.0)
    back, n_face = plane_layer((-0.4, 0.1, -1.0), 5.0)
    out = _aov(make_world([front, back]))
    in_front = out.layer_id == 0
    in_back = out.layer_id == 1
    assert in_front.sum() > 1000 and in_back.sum() > 1000
    assert np.allclose(out.normal[in_front], [0, 0, -1], atol=1e-5)
    sel = in_back & (out.alpha > 0.98)
    assert np.allclose(out.normal[sel], n_face, atol=2e-4)


def test_normal_never_mixes_across_depth_step():
    # 한 층 안의 깊이 불연속: 두 쪽 모두 정면 평면이므로 모든 텍셀 법선이 (0,0,-1) 이어야 한다(단절된 이웃은 쓰지 않음)
    dep = np.full((H, W), 6.0, np.float32)
    dep[:, : W // 2] = 3.0
    out = _aov(make_world([layer(rand_tex(0), dep)]))
    assert (out.atom_id >= 0).all()
    assert np.allclose(out.normal, [0, 0, -1], atol=1e-6)


def test_normals_unit_facing_camera_in_generic_scene():
    la = layer(rand_tex(3, alpha=0.7), np.linspace(2.0, 2.6, W, dtype=np.float32)[None, :].repeat(H, 0))
    lb, _ = plane_layer((0.2, 0.3, -1.0), 5.0)
    world = make_world([la, lb])
    pose = yaw(translate(I4, (0.1, 0.0, 0.2)), 6.0)
    out = _aov(world, pose)
    cov = out.atom_id >= 0
    n = out.normal[cov].astype(np.float64)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)
    assert (out.normal[~cov] == 0).all()
    X = world_to_cam(pose, A.atom_points(world, out.atom_id[cov]))
    assert ((n * X).sum(-1) <= 1e-6).all()                       # n·P <= 0: 카메라를 향한다


# ---------------------------------------------------------------------------------------------------------------
# 광류
# ---------------------------------------------------------------------------------------------------------------
def _cam(c2w, k1=0.0, k2=0.0, K=None):
    return A.Cam(c2w=c2w, K=K_default() if K is None else K, k1=k1, k2=k2)


def test_flow_lateral_translation_closed_form():
    z, dx = 2.5, 0.2
    world = make_world([layer(smooth_tex(), z)])
    out = _aov(world, I4, flow_to=_cam(translate(I4, (dx, 0, 0))))
    assert out.flow.dtype == np.float32 and out.flow.shape == (H, W, 2) and out.flow_valid.dtype == bool
    assert out.flow_valid.all() and (out.atom_id >= 0).all()
    assert np.allclose(out.flow[..., 0], -FX() * dx / z, atol=1e-3)
    assert np.allclose(out.flow[..., 1], 0.0, atol=1e-3)


def test_flow_depends_on_layer_depth():
    front = const_layer((1, 0, 0), 0.0, 2.0)
    rgba = front.rgba.copy()
    rgba[:, :40, 3] = 1.0
    world = make_world([layer(rgba, 2.0), const_layer((0, 0, 1), 1.0, 5.0)])
    dx = 0.1
    out = _aov(world, I4, flow_to=_cam(translate(I4, (dx, 0, 0))))
    f = out.layer_id == 0
    b = out.layer_id == 1
    assert f.sum() > 1000 and b.sum() > 1000
    assert np.allclose(out.flow[f][:, 0], -FX() * dx / 2.0, atol=1e-3)
    assert np.allclose(out.flow[b][:, 0], -FX() * dx / 5.0, atol=1e-3)


def test_flow_push_in_matches_atom_texel_position():
    z = 3.0
    world = make_world([layer(rand_tex(4), z)])
    out = _aov(world, I4, flow_to=_cam(translate(I4, (0, 0, 0.5))))     # 0.5 m 전진: z 3.0 -> 2.5
    v, u = np.mgrid[0:H, 0:W]
    atom_v, atom_u = (out.atom_id % HW) // W, (out.atom_id % HW) % W
    assert np.array_equal(atom_v, v) and np.array_equal(atom_u, u)       # 항등 카메라: 자기 텍셀
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    gain = z / (z - 0.5) - 1.0
    assert np.allclose(out.flow[..., 0], (u - cx) * gain, atol=2e-3)
    assert np.allclose(out.flow[..., 1], (v - cy) * gain, atol=2e-3)


def test_flow_matches_geom_oracle_for_rotated_cameras_with_distortion():
    obs = make_obs(k1=0.04)
    la = layer(rand_tex(5, alpha=0.9), np.linspace(2.5, 3.5, H, dtype=np.float32)[:, None].repeat(W, 1))
    world = make_world([la, const_layer((0.2, 0.2, 0.8), 1.0, 6.0)], obs=obs)
    pa = yaw(translate(I4, (0.1, 0.0, 0.1)), 7.0)
    pb = yaw(translate(I4, (-0.2, 0.05, 0.4)), -9.0)
    kb = (0.05, -0.01)
    out = _aov(world, pa, flow_to=_cam(pb, *kb))
    sel = out.flow_valid
    assert sel.mean() > 0.7
    X = A.atom_points(world, out.atom_id[sel])
    wa, wb = np.linalg.inv(pa), np.linalg.inv(pb)
    uv_a, _ = geom.project(K_default(), wa[:3, :3], wa[:3, 3], X, 0.0, 0.0)
    uv_b, _ = geom.project(K_default(), wb[:3, :3], wb[:3, 3], X, *kb)
    assert np.abs(out.flow[sel] - (uv_b - uv_a)).max() < 3e-3
    assert (out.flow[~sel] == 0).all()


def test_flow_invalid_rules():
    z = 3.0
    rgba = smooth_tex()
    rgba[:, 70:, 3] = 0.0
    world = make_world([layer(rgba, z)])
    # (a) 일부 화소는 표면 없음
    out = _aov(world, I4, flow_to=_cam(translate(I4, (0.1, 0, 0))))
    empty = out.atom_id < 0
    assert empty.any() and (~empty).any()
    assert not out.flow_valid[empty].any() and (out.flow[empty] == 0).all()
    assert out.flow_valid[~empty].all()
    # (b) B 카메라가 카드를 지나쳐 카드가 뒤에 있다 / 근평면 안
    for fwd in (3.5, 2.9995):
        o2 = _aov(world, I4, flow_to=_cam(translate(I4, (0, 0, fwd))))
        assert not o2.flow_valid.any() and (o2.flow == 0).all()
        assert np.array_equal(o2.atom_id, out.atom_id)                 # 광류는 지배 원자를 바꾸지 않는다
    # (c) flow_to 없음
    o3 = _aov(world)
    assert not o3.flow_valid.any() and (o3.flow == 0).all()
    assert np.array_equal(o3.atom_id, out.atom_id)


def test_flow_to_validation():
    world = make_world([layer(smooth_tex(), 3.0)])
    with pytest.raises(TypeError):
        _aov(world, I4, flow_to={"c2w": I4, "K": K_default()})
    with pytest.raises(ValueError):
        A.Cam(c2w=np.eye(3), K=K_default())
    bad = np.eye(4)
    bad[:3, :3] *= 2.0
    with pytest.raises(ValueError):
        A.Cam(c2w=bad, K=K_default())
    with pytest.raises(ValueError):
        A.Cam(c2w=I4, K=np.zeros((3, 3)))
