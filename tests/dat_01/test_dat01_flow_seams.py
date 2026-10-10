"""DAT-01 광류 GT(닫힌형·geom 오라클·무효 규칙)와 상류(RDR-02/RDR-05) 이음매 교차검증."""
import numpy as np
import pytest

from buc.core import contracts as C
from buc.core import geom
from buc.data import synth_render as S
from buc.invent.grammar import core as G

import dat01_helpers as H

W, HT = 64, 48


def _wall_cam():
    return H.cam_center(W, HT, fov=70.0, eye=(2.0, 2.0, 1.5), target=(2.0, 3.0, 1.5))


def test_flow_closed_form_on_fronto_parallel_wall():
    prog = H.box_program(lights=H.two_lights())
    cam = _wall_cam()
    dx = 0.1
    cam_b = H.shifted(cam, (dx, 0.0, 0.0))
    s = S.render_gbuffer(prog, cam, flow_to=cam_b)
    depth, ax, side, _ = H.box_oracle(cam)
    wall = (ax == 1) & (side == 1)  # y=4 벽: 광축에 수직, z-깊이 = 2
    assert wall.sum() > 0.4 * wall.size
    fx = cam.K[0, 0]
    np.testing.assert_allclose(s.flow[..., 0][wall], -fx * dx / 2.0, atol=2e-3)  # du = -fx dx / Z
    np.testing.assert_allclose(s.flow[..., 1][wall], 0.0, atol=2e-3)
    assert s.flow_valid[wall].all()
    assert s.flow_camera is cam_b or np.array_equal(s.flow_camera.c2w, cam_b.c2w)


def test_flow_matches_geom_oracle_for_rotation_and_translation():
    prog = H.furnished_program()
    cam = H.cam_center(W, HT, fov=110.0)
    cam_b = H.rotated_yaw(H.shifted(cam, (0.2, 0.0, 0.1)), 4.0)
    s = S.render_gbuffer(prog, cam, flow_to=cam_b)
    assert np.isfinite(s.depth).all()
    f, zb = H.flow_oracle(s.depth.astype(np.float64), cam, cam_b)
    valid = zb > 1e-3
    assert (s.flow_valid == valid).all()
    np.testing.assert_allclose(s.flow[valid], f[valid], atol=2e-3, rtol=1e-4)
    assert np.abs(s.flow).max() > 1.0  # 실제로 움직인 광류


def test_flow_with_lens_distortion_uses_both_cameras_distortion():
    prog = H.box_program(lights=H.two_lights())
    cam = H.cam_center(W, HT, fov=90.0, eye=(2.0, 1.0, 1.5), k1=-0.08, k2=0.01)
    cam_b = H.shifted(cam, (0.15, 0.0, 0.0))
    s = S.render_gbuffer(prog, cam, flow_to=cam_b)
    depth, _, _, _ = H.box_oracle(cam)
    np.testing.assert_allclose(s.depth, depth, rtol=3e-5, atol=1e-6)  # 왜곡 광선도 해석 깊이와 일치
    f, zb = H.flow_oracle(s.depth.astype(np.float64), cam, cam_b)
    np.testing.assert_allclose(s.flow[zb > 1e-3], f[zb > 1e-3], atol=3e-3, rtol=1e-4)


def test_flow_invalid_when_point_is_behind_target_camera():
    prog = H.box_program(lights=H.two_lights())
    cam = H.cam_center(W, HT, fov=100.0, eye=(2.0, 2.0, 1.5), target=(2.0, 3.0, 1.5))
    cam_b = H.rotated_yaw(cam, 130.0)  # 같은 위치에서 크게 돌아선다: 일부 점은 B 의 뒤쪽
    s = S.render_gbuffer(prog, cam, flow_to=cam_b)
    f, zb = H.flow_oracle(s.depth.astype(np.float64), cam, cam_b)
    valid = zb > 1e-3
    assert 0.05 < valid.mean() < 0.95, "테스트 설계: 유효/무효가 모두 있어야 한다"
    assert (s.flow_valid == valid).all()
    assert (s.flow[~valid] == 0).all() and np.isfinite(s.flow).all()


def test_no_flow_target_gives_zero_and_invalid():
    s = S.render_gbuffer(H.box_program(lights=H.two_lights()), _wall_cam())
    assert s.flow_camera is None
    assert not s.flow_valid.any() and (s.flow == 0).all()


def test_translate_camera_helper_moves_along_camera_axes():
    cam = _wall_cam()
    d = np.array([0.1, -0.2, 0.3])
    moved = S.translate_camera(cam, d)
    np.testing.assert_allclose(moved.c2w[:3, 3], cam.c2w[:3, 3] + cam.c2w[:3, :3] @ d, atol=1e-12)
    np.testing.assert_allclose(moved.c2w[:3, :3], cam.c2w[:3, :3], atol=1e-15)
    np.testing.assert_allclose(moved.K, cam.K)
    assert (moved.width, moved.height) == (cam.width, cam.height)


# --------------------------------------------------------------------------- 상류 이음매: RDR-05 / RDR-02
def _world_from_sample(s):
    cam = s.camera
    w2c = geom.invert_pose(cam.c2w)
    obs = C.Obs(K=cam.K, R=w2c[:3, :3], t=w2c[:3, 3], width=cam.width, height=cam.height, k1=cam.k1, k2=cam.k2)
    rgba = np.concatenate([np.clip(s.rgb, 0.0, 1.0), np.ones(s.rgb.shape[:2] + (1,), np.float32)], axis=-1)
    layer = C.LDILayer(rgba=rgba, depth=s.depth)
    return C.World(obs=obs, layers=(layer,), theta_in=C.Theta(), level="L1", read_hash="r" * 8, prov=C.Prov())


@pytest.fixture(scope="module")
def seam():
    prog = H.furnished_program()
    cam = H.cam_center(W, HT, fov=100.0)
    cam_b = H.shifted(cam, (0.1, 0.0, 0.0))
    s = S.render_gbuffer(prog, cam, flow_to=cam_b)
    return s, cam, cam_b, _world_from_sample(s)


def test_flow_agrees_with_rdr05_aov_flow(seam):
    from buc.render import aov
    s, cam, cam_b, world = seam
    a = aov.render_aov(world, c2w=cam.c2w, K=cam.K, width=W, height=HT,
                       flow_to=aov.Cam(c2w=cam_b.c2w, K=cam_b.K))
    both = a.flow_valid & s.flow_valid
    assert both.mean() > 0.95
    err = np.abs(a.flow[both] - s.flow[both]).max(axis=-1)
    assert np.median(err) < 5e-3
    assert (err < 0.05).mean() > 0.99


def test_normals_agree_with_rdr05_finite_difference_normals(seam):
    from buc.render import aov
    s, cam, _, world = seam
    a = aov.render_aov(world, c2w=cam.c2w, K=cam.K, width=W, height=HT)
    ours = np.moveaxis(s.normal, 0, -1).astype(np.float64)
    theirs = a.normal.astype(np.float64)
    ok = np.linalg.norm(theirs, axis=-1) > 0.5
    cosang = np.sum(ours * theirs, axis=-1)[ok]
    assert ok.mean() > 0.9
    assert (cosang > np.cos(np.deg2rad(8.0))).mean() > 0.8  # 에지 화소를 빼면 대부분이 8도 이내
    assert np.median(cosang) > 0.999


def test_gbuffer_depth_rgb_roundtrip_through_rdr02_identity_camera(seam):
    from buc.render import quads
    s, cam, _, world = seam
    r = quads.render_ldi(world, c2w=cam.c2w, K=cam.K, width=W, height=HT, k1=cam.k1, k2=cam.k2)
    np.testing.assert_allclose(r.alpha, 1.0, atol=1e-3)
    want = np.clip(s.rgb, 0.0, 1.0)
    err = np.abs(r.rgb - want).max(axis=-1)
    assert err.mean() < 3e-3
    assert (err < 0.02).mean() > 0.98
    ok = np.isfinite(r.depth)
    rel = np.abs(r.depth[ok] - s.depth[ok]) / s.depth[ok]
    assert np.median(rel) < 1e-4 and (rel < 0.02).mean() > 0.98
