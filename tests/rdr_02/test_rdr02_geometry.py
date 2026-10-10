"""RDR-02 기하: 항등, 평면 호모그래피 오라클, 시차, 왜곡, 점 착지, 깊이 AOV."""
import math

import numpy as np
import pytest

from buc.core import geom
from buc.render.quads import render_ldi
from rdr02_helpers import (FX, H, W, K_default, const_layer, layer, make_obs, make_world, plane_oracle,
                           smooth_tex, translate, yaw)

TEX = smooth_tex()
D = 3.0


def _render(world, c2w, K=None, w=W, h=H, **kw):
    return render_ldi(world, c2w=c2w, K=K_default(w, h) if K is None else K, width=w, height=h, **kw)


def _plane_world(d=D):
    return make_world([layer(TEX, d)])


def test_identity_reproduces_layer():
    out = _render(_plane_world(), np.eye(4))
    np.testing.assert_allclose(out.rgb, TEX[..., :3], atol=2e-3)
    assert out.alpha.min() >= 1 - 1e-3
    np.testing.assert_allclose(out.depth, D, rtol=1e-4)


def test_identity_with_nontrivial_pose_and_distortion():
    # 월드 포즈가 임의여도, 목표 카메라 == 관측 카메라(왜곡 포함)이면 항등이어야 한다
    c2w = geom.look_at(np.array([0.7, -0.4, -1.0]), np.array([0.2, 0.1, 2.5]), up=np.array([0.0, -1.0, 0.0]))
    for k1, k2 in [(0.0, 0.0), (-0.08, 0.01)]:
        obs = make_obs(c2w, k1=k1, k2=k2)
        out = _render(make_world([layer(TEX, 2.5)], obs), c2w, k1=k1, k2=k2)
        # 왜곡이 큰 가장자리는 풋프린트가 달라질 수 있어 안쪽 4px 를 비교한다
        sl = (slice(4, H - 4), slice(4, W - 4))
        np.testing.assert_allclose(out.rgb[sl], TEX[..., :3][sl], atol=8e-3)
        assert out.alpha[sl].min() >= 0.99


@pytest.mark.parametrize("name,move,yaw_deg", [
    ("lateral_x", (0.12, 0.0, 0.0), 0.0),
    ("lateral_xy", (-0.08, 0.05, 0.0), 0.0),
    ("dolly_in_1.2x", (0.0, 0.0, 0.5), 0.0),
    ("dolly_in_2x", (0.0, 0.0, 1.5), 0.0),
    ("yaw10", (0.05, 0.0, 0.0), 10.0),
])
def test_fronto_parallel_card_matches_homography_oracle(name, move, yaw_deg):
    c2w_t = yaw(translate(np.eye(4), move), yaw_deg)
    out = _render(_plane_world(), c2w_t)
    obs = make_obs()
    ref, inside, _ = plane_oracle(TEX[..., :3], obs, c2w_t, K_default(), W, H, n=(0, 0, 1), d=D, margin=3.0)
    assert inside.sum() > 0.25 * W * H, "오라클 영역이 너무 작다(테스트 설정 오류)"
    err = np.abs(out.rgb.astype(np.float64) - ref)[inside]
    assert err.mean() < 0.01, f"{name}: 평균 오차 {err.mean():.4f}"
    assert err.max() < 0.05, f"{name}: 최대 오차 {err.max():.4f}"
    assert out.alpha[inside].min() > 0.98, f"{name}: 구멍(핀홀) 발생"


def test_minification_dolly_out_stays_close_to_oracle():
    # 축소(0.75x)는 필터가 화소 간격보다 좁아 알파/색에 약간의 잔물결이 허용된다
    c2w_t = translate(np.eye(4), (0.0, 0.0, -1.0))
    out = _render(_plane_world(), c2w_t)
    ref, inside, _ = plane_oracle(TEX[..., :3], make_obs(), c2w_t, K_default(), W, H, n=(0, 0, 1), d=D, margin=3.0)
    assert inside.sum() > 0.25 * W * H
    err = np.abs(out.rgb.astype(np.float64) - ref)[inside]
    assert err.mean() < 0.03 and err.max() < 0.15, (err.mean(), err.max())
    assert out.alpha[inside].min() > 0.8 and out.alpha[inside].mean() > 0.95


def test_slanted_plane_per_pixel_depth_matches_oracle():
    # 기울어진 평면 n.X = d (n = (sin30, 0, cos30)) 의 픽셀별 깊이로 만든 한 장의 층
    n = np.array([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])
    d = 2.8
    K = K_default()
    uv = geom.pixel_grid(H, W)
    xy1 = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    z = d / (xy1 @ n)
    assert z.min() > 1.0
    world = make_world([layer(TEX, z.astype(np.float32))])
    c2w_t = yaw(translate(np.eye(4), (0.15, 0.0, 0.1)), 8.0)
    out = _render(world, c2w_t)
    ref, inside, _ = plane_oracle(TEX[..., :3], make_obs(), c2w_t, K, W, H, n=n, d=d, margin=4.0)
    assert inside.sum() > 0.3 * W * H
    err = np.abs(out.rgb.astype(np.float64) - ref)[inside]
    assert err.mean() < 0.02 and err.max() < 0.1, (err.mean(), err.max())
    assert out.alpha[inside].min() > 0.97


def test_depth_aov_is_target_camera_zdepth():
    # 광축 z-깊이: 앞으로 0.5 m 가면 3.0 -> 2.5, yaw 하면 평면의 z 가 화소마다 달라진다
    out = _render(_plane_world(), translate(np.eye(4), (0, 0, 0.5)))
    mid = (slice(16, H - 16), slice(24, W - 24))
    np.testing.assert_allclose(out.depth[mid], 2.5, rtol=2e-3)
    c2w_t = yaw(translate(np.eye(4), (0.05, 0, 0)), 10.0)
    out = _render(_plane_world(), c2w_t)
    K = K_default()
    # 목표 카메라 좌표에서의 평면: 월드 평면 z = D (법선 e_z), 점 X = c + z_t * R (x, y, 1)
    uv = geom.pixel_grid(H, W)
    ray = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    Rw = c2w_t[:3, :3]
    ez = Rw[2, :]                                   # (R ray)_z = Rw[2] . ray
    z_t = (D - c2w_t[2, 3]) / (ray @ ez)
    _, inside, _ = plane_oracle(TEX[..., :3], make_obs(), c2w_t, K, W, H, n=(0, 0, 1), d=D, margin=4.0)
    assert inside.sum() > 0.2 * W * H
    np.testing.assert_allclose(out.depth[inside], z_t[inside], rtol=3e-3)


@pytest.mark.parametrize("k1,k2", [(0.0, 0.0), (-0.08, 0.01)])
def test_points_land_where_geom_project_says(k1, k2):
    # 서로 다른 깊이의 작은 불투명 블록 두 개: 알파 무게중심이 geom.project 의 투영과 0.3 px 이내
    rgba = np.zeros((H, W, 4), np.float32)
    depth = np.full((H, W), 6.0, np.float32)
    spots = [((20, 15), 2.0), ((72, 44), 5.0)]            # ((u, v), z)
    for (u, v), z in spots:
        rgba[v - 1:v + 2, u - 1:u + 2] = 1.0
        depth[v - 1:v + 2, u - 1:u + 2] = z
    obs = make_obs()
    world = make_world([layer(rgba, depth)], obs)
    c2w_t = yaw(translate(np.eye(4), (0.10, -0.04, 0.2)), 4.0)
    out = _render(world, c2w_t, k1=k1, k2=k2)
    w2c_t = geom.invert_pose(c2w_t)
    for (u, v), z in spots:
        # 블록 중심 한 점의 월드 위치(깊이는 블록 깊이)
        X = geom.unproject_world(obs.K, obs.R, obs.t, np.array([[float(u), float(v)]]), np.array([z]))
        uv_exp, z_t = geom.project(K_default(), w2c_t[:3, :3], w2c_t[:3, 3], X, k1, k2)
        ue, ve = float(uv_exp[0, 0]), float(uv_exp[0, 1])
        r = 7
        u0, v0 = int(round(ue)), int(round(ve))
        win = (slice(max(v0 - r, 0), v0 + r + 1), slice(max(u0 - r, 0), u0 + r + 1))
        a = out.alpha[win].astype(np.float64)
        assert a.sum() == pytest.approx(9.0 * (z / float(z_t[0])) ** 2, rel=0.25), "블록 면적(원근 축소/확대)이 맞지 않다"
        vv, uu = np.mgrid[win[0], win[1]]
        cu, cv = (a * uu).sum() / a.sum(), (a * vv).sum() / a.sum()
        assert abs(cu - ue) < 0.3 and abs(cv - ve) < 0.3, ((cu, cv), (ue, ve))


def test_resolution_independence_supersample_then_box_matches_1x():
    c2w_t = translate(np.eye(4), (0.11, 0.0, 0.3))
    world = _plane_world()
    one = _render(world, c2w_t)
    K2 = geom.scale_intrinsics(K_default(), 2.0)
    two = render_ldi(world, c2w=c2w_t, K=K2, width=2 * W, height=2 * H)
    box = two.rgb.reshape(H, 2, W, 2, 3).mean(axis=(1, 3))
    sl = (slice(6, H - 6), slice(10, W - 10))
    err = np.abs(box.astype(np.float64) - one.rgb)[sl]
    assert err.mean() < 0.012 and err.max() < 0.05, (err.mean(), err.max())
