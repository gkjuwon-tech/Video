"""FND-06 geom: K / 포즈 / 투영 / 왜곡 / 평면 호모그래피. 정답을 아는 값과 독립 오라클(스칼라 루프, scipy)로 판정한다."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from buc.core import geom

RNG = np.random.default_rng(20261010)


def rand_rot(rng=RNG, max_deg=179.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * math.radians(rng.uniform(0, max_deg))).as_matrix()


def rand_pose(rng=RNG):
    T = np.eye(4)
    T[:3, :3] = rand_rot(rng)
    T[:3, 3] = rng.normal(size=3) * 2
    return T


# ---------------------------------------------------------------------------------------------------- 내부행렬 K
def test_intrinsics_from_fov_exact():
    K = geom.intrinsics_from_fov(60.0, 1920, 1080)
    fx = 960.0 / math.tan(math.radians(30.0))
    assert K.shape == (3, 3) and K.dtype == np.float64
    np.testing.assert_allclose(K, [[fx, 0, 959.5], [0, fx, 539.5], [0, 0, 1]], rtol=0, atol=1e-9)


def test_90deg_fov_edge_ray_lands_on_pixel_edge():
    """수평 FOV 90도의 가장자리 광선은 영상 오른쪽 변(u = W-0.5)에 떨어진다(픽셀 중심이 정수인 규약)."""
    K = geom.intrinsics_from_fov(90.0, 101, 51)
    uv, z = geom.project(K, np.eye(3), np.zeros(3), np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]))
    np.testing.assert_allclose(uv[:, 0], [100.5, -0.5], atol=1e-9)
    np.testing.assert_allclose(uv[:, 1], [25.0, 25.0], atol=1e-9)
    np.testing.assert_allclose(z, [1.0, 1.0])


def test_fov_focal_conversions_known_values():
    assert geom.fov_from_focal(50.0, 36.0) == pytest.approx(39.5978, abs=1e-3)   # 2 atan(18/50)
    assert geom.fov_from_focal(24.0, 36.0) == pytest.approx(73.7398, abs=1e-3)   # 2 atan(18/24)
    for f, s in [(35.0, 24.89), (12.0, 9.6), (200.0, 36.0)]:
        assert geom.focal_from_fov(geom.fov_from_focal(f, s), s) == pytest.approx(f, rel=1e-12)
    with pytest.raises(ValueError):
        geom.fov_from_focal(0.0, 36.0)
    with pytest.raises(ValueError):
        geom.focal_from_fov(180.0, 36.0)


def test_intrinsics_from_focal_matches_fov_form():
    K1 = geom.intrinsics_from_focal(35.0, 24.89, 1920, 1080)
    assert K1[0, 0] == pytest.approx(35.0 / 24.89 * 1920, rel=1e-12) and K1[1, 1] == K1[0, 0]
    K2 = geom.intrinsics_from_fov(geom.fov_from_focal(35.0, 24.89), 1920, 1080)
    np.testing.assert_allclose(K1, K2, rtol=1e-12)
    assert geom.fov_x_deg_from_K(K1, 1920) == pytest.approx(geom.fov_from_focal(35.0, 24.89), abs=1e-9)


def test_scale_and_crop_intrinsics_move_pixels_consistently():
    K = geom.intrinsics_from_fov(70.0, 640, 480)
    R, t = rand_rot(), np.array([0.1, -0.2, 0.3])
    X = RNG.uniform(-1, 1, size=(50, 3)) + np.array([0, 0, 4.0])
    uv, _ = geom.project(K, R, t, X)
    for sx, sy in [(0.5, 0.5), (0.25, 0.25), (2.0, 2.0), (0.5, 0.75)]:
        K2 = geom.scale_intrinsics(K, sx, sy)
        uv2, _ = geom.project(K2, R, t, X)
        want = (uv + 0.5) * np.array([sx, sy]) - 0.5          # 픽셀 중심이 정수: u' = (u + 0.5)·s - 0.5
        np.testing.assert_allclose(uv2, want, atol=1e-9)
    np.testing.assert_allclose(geom.scale_intrinsics(K, 0.5), geom.scale_intrinsics(K, 0.5, 0.5))
    Kc = geom.crop_intrinsics(K, 100, 40)
    uvc, _ = geom.project(Kc, R, t, X)
    np.testing.assert_allclose(uvc, uv - np.array([100, 40]), atol=1e-9)
    assert Kc[0, 0] == K[0, 0] and Kc[1, 1] == K[1, 1]
    with pytest.raises(ValueError):
        geom.scale_intrinsics(K, 0.0)
    assert not np.shares_memory(geom.scale_intrinsics(K, 1.0), K)


# ---------------------------------------------------------------------------------------------------- 회전
def test_rodrigues_matches_scipy_and_roundtrips():
    for _ in range(200):
        rv = RNG.normal(size=3) * RNG.uniform(0.0, 1.0)
        rv *= RNG.uniform(0.0, math.pi) / max(np.linalg.norm(rv), 1e-12)
        R = geom.rodrigues(rv)
        np.testing.assert_allclose(R, Rotation.from_rotvec(rv).as_matrix(), atol=1e-12)
        assert geom.is_rotation(R, atol=1e-12)
        np.testing.assert_allclose(geom.rotvec_from_matrix(R), rv, atol=1e-9)
    np.testing.assert_allclose(geom.rodrigues(np.zeros(3)), np.eye(3), atol=0)
    np.testing.assert_allclose(geom.rotvec_from_matrix(np.eye(3)), np.zeros(3), atol=0)
    rv = np.array([1e-10, -2e-10, 3e-10])                         # 아주 작은 각도에서 0/0이 나면 안 된다
    np.testing.assert_allclose(geom.rotvec_from_matrix(geom.rodrigues(rv)), rv, rtol=1e-4)


def test_rotvec_near_pi_is_stable():
    axis = np.array([2.0, -1.0, 2.0]) / 3.0
    for ang in [math.pi - 1e-3, math.pi - 1e-8, math.pi]:
        R = geom.rodrigues(axis * ang)
        rv = geom.rotvec_from_matrix(R)
        assert np.linalg.norm(rv) == pytest.approx(ang, abs=1e-6)
        np.testing.assert_allclose(geom.rodrigues(rv), R, atol=1e-7)   # 부호 모호성은 R 로 판정한다


def test_is_rotation_and_project_to_so3():
    R = rand_rot()
    assert geom.is_rotation(R)
    assert not geom.is_rotation(np.diag([1.0, 1.0, -1.0]))             # 반사
    assert not geom.is_rotation(R * 1.01)                              # 스케일
    assert not geom.is_rotation(np.eye(4))
    np.testing.assert_allclose(geom.project_to_so3(R), R, atol=1e-12)
    noisy = R + RNG.normal(size=(3, 3)) * 1e-4
    P = geom.project_to_so3(noisy)
    assert geom.is_rotation(P, atol=1e-12) and np.linalg.norm(P - R) < 5e-4
    assert geom.is_rotation(geom.project_to_so3(np.diag([1.0, 1.0, -1.0])), atol=1e-12)   # det +1 로 강제


def test_rotation_angle_deg_is_geodesic_and_precise():
    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    base = rand_rot()
    for deg in [0.0, 1e-4, 0.1, 30.0, 90.0, 150.0, 179.9, 180.0]:
        R2 = base @ Rotation.from_rotvec(axis * math.radians(deg)).as_matrix()
        assert geom.rotation_angle_deg(base, R2) == pytest.approx(deg, rel=1e-6, abs=1e-8)
        assert geom.rotation_angle_deg(R2, base) == pytest.approx(deg, rel=1e-6, abs=1e-8)
    assert geom.rotation_angle_deg(base, base) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------------------------------- 포즈
def test_look_at_known_geometry():
    c2w = geom.look_at(np.zeros(3), np.array([5.0, 0, 0]), up=np.array([0.0, 0, 1]))
    assert c2w.shape == (4, 4) and c2w.dtype == np.float64
    # 카메라 x(오른쪽) = 세계 -Y, y(아래) = 세계 -Z, z(앞) = 세계 +X, 오른손계.
    np.testing.assert_allclose(c2w[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
    np.testing.assert_allclose(c2w[3], [0, 0, 0, 1])


def test_look_at_projects_target_to_principal_point_and_up_is_up():
    for _ in range(20):
        eye = RNG.normal(size=3) * 3
        target = RNG.normal(size=3)
        up = np.array([0.0, 1.0, 0.0])
        if np.linalg.norm(np.cross(target - eye, up)) < 0.3:
            continue
        c2w = geom.look_at(eye, target, up=up)
        assert geom.is_rotation(c2w[:3, :3], atol=1e-12)
        np.testing.assert_allclose(c2w[:3, 3], eye, atol=0)
        w2c = geom.invert_pose(c2w)
        K = geom.intrinsics_from_fov(50.0, 320, 240)
        uv, z = geom.project(K, w2c[:3, :3], w2c[:3, 3], target[None])
        np.testing.assert_allclose(uv[0], [159.5, 119.5], atol=1e-9)
        assert z[0] == pytest.approx(np.linalg.norm(target - eye), rel=1e-12)
        uv_up, _ = geom.project(K, w2c[:3, :3], w2c[:3, 3], (target + 0.05 * up)[None])
        assert uv_up[0, 1] < uv[0, 1]                                  # 세계 위쪽은 영상 위쪽(v 감소)


def test_look_at_degenerate_inputs_raise():
    with pytest.raises(ValueError):
        geom.look_at(np.ones(3), np.ones(3), up=np.array([0.0, 0, 1]))                 # eye == target
    with pytest.raises(ValueError):
        geom.look_at(np.zeros(3), np.array([0.0, 0, 3]), up=np.array([0.0, 0, 1]))     # 시선 ∥ up
    with pytest.raises(ValueError):
        geom.look_at(np.zeros(3), np.array([1.0, 0, 0]), up=np.zeros(3))


def test_pose_inverse_relative_and_interp():
    for _ in range(20):
        T = rand_pose()
        np.testing.assert_allclose(geom.invert_pose(T) @ T, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(geom.make_pose(T[:3, :3], T[:3, 3]), T, atol=0)
    Ta, Tb = rand_pose(), rand_pose()        # w2c
    Tab = geom.relative_pose(Ta, Tb)         # a 카메라 좌표 -> b 카메라 좌표
    np.testing.assert_allclose(Tab, Tb @ np.linalg.inv(Ta), atol=1e-10)
    X = RNG.normal(size=(7, 3))
    xa = X @ Ta[:3, :3].T + Ta[:3, 3]
    xb = X @ Tb[:3, :3].T + Tb[:3, 3]
    np.testing.assert_allclose(xa @ Tab[:3, :3].T + Tab[:3, 3], xb, atol=1e-10)

    T0, T1 = rand_pose(), rand_pose()
    np.testing.assert_allclose(geom.interp_pose(T0, T1, 0.0), T0, atol=1e-12)
    np.testing.assert_allclose(geom.interp_pose(T0, T1, 1.0), T1, atol=1e-12)
    total = geom.rotation_angle_deg(T0[:3, :3], T1[:3, :3])
    for s in [0.1, 0.25, 0.5, 0.9]:
        Ts = geom.interp_pose(T0, T1, s)
        assert geom.is_rotation(Ts[:3, :3], atol=1e-12)
        assert geom.rotation_angle_deg(T0[:3, :3], Ts[:3, :3]) == pytest.approx(s * total, rel=1e-7, abs=1e-8)
        np.testing.assert_allclose(Ts[:3, 3], (1 - s) * T0[:3, 3] + s * T1[:3, 3], atol=1e-12)
        np.testing.assert_allclose(Ts[3], [0, 0, 0, 1])


# ---------------------------------------------------------------------------------------------------- 왜곡·투영
def test_distort_matches_formula_and_undistort_inverts():
    xy = RNG.uniform(-0.8, 0.8, size=(500, 2))
    for k1, k2 in [(0.0, 0.0), (-0.15, 0.03), (0.2, 0.0), (0.1, -0.05)]:
        d = geom.distort(xy, k1, k2)
        r2 = (xy ** 2).sum(1, keepdims=True)
        np.testing.assert_allclose(d, xy * (1 + k1 * r2 + k2 * r2 ** 2), atol=1e-14)
        np.testing.assert_allclose(geom.undistort(d, k1, k2), xy, atol=1e-10)
    np.testing.assert_array_equal(geom.distort(xy, 0.0, 0.0), xy)
    np.testing.assert_allclose(geom.distort(np.zeros((1, 2)), 0.3, 0.1), 0.0)


def test_project_matches_scalar_oracle_and_shapes():
    K = np.array([[900.0, 0.0, 310.25], [0.0, 880.0, 241.75], [0, 0, 1.0]])
    R, t = rand_rot(), np.array([0.2, -0.1, 0.5])
    X = RNG.uniform(-1, 1, size=(4, 5, 3)) + np.array([0, 0, 5.0])
    k1, k2 = -0.12, 0.02
    uv, z = geom.project(K, R, t, X, k1, k2)
    assert uv.shape == (4, 5, 2) and z.shape == (4, 5) and uv.dtype == np.float64
    for i in range(4):
        for j in range(5):
            xc = R @ X[i, j] + t
            x, y = xc[0] / xc[2], xc[1] / xc[2]
            f = 1 + k1 * (x * x + y * y) + k2 * (x * x + y * y) ** 2
            assert uv[i, j, 0] == pytest.approx(900.0 * x * f + 310.25, abs=1e-9)
            assert uv[i, j, 1] == pytest.approx(880.0 * y * f + 241.75, abs=1e-9)
            assert z[i, j] == pytest.approx(xc[2], abs=1e-12)
    uv0, _ = geom.project(K, np.eye(3), np.zeros(3), np.array([[0.0, 0.0, 7.0]]))
    np.testing.assert_allclose(uv0[0], [310.25, 241.75], atol=1e-12)
    _, zb = geom.project(K, np.eye(3), np.zeros(3), np.array([[0.0, 0.0, -2.0]]))
    assert zb[0] == -2.0                                              # 카메라 뒤는 z<0 으로 보고(호출자가 거른다)


def test_unproject_and_project_roundtrip_with_distortion():
    K = geom.intrinsics_from_fov(65.0, 400, 300)
    R, t = rand_rot(), np.array([0.3, 0.1, -0.4])
    uv = np.stack([RNG.uniform(0, 399, 300), RNG.uniform(0, 299, 300)], axis=1)
    depth = RNG.uniform(0.5, 20.0, 300)
    for k1, k2 in [(0.0, 0.0), (-0.1, 0.02)]:
        Xc = geom.unproject(K, uv, depth, k1, k2)
        assert Xc.shape == (300, 3)
        np.testing.assert_allclose(Xc[:, 2], depth, atol=1e-12)
        uv_back, z = geom.project(K, np.eye(3), np.zeros(3), Xc, k1, k2)
        np.testing.assert_allclose(uv_back, uv, atol=1e-7)
        Xw = geom.unproject_world(K, R, t, uv, depth, k1, k2)
        np.testing.assert_allclose(Xw @ R.T + t, Xc, atol=1e-9)
        uv_w, z_w = geom.project(K, R, t, Xw, k1, k2)
        np.testing.assert_allclose(uv_w, uv, atol=1e-7)
        np.testing.assert_allclose(z_w, depth, atol=1e-9)


def test_pixel_grid_and_depth_grid_unprojection():
    g = geom.pixel_grid(3, 5)
    assert g.shape == (3, 5, 2) and g.dtype == np.float64
    np.testing.assert_array_equal(g[0, 0], [0, 0])
    np.testing.assert_array_equal(g[2, 4], [4, 2])                     # (u, v) = (열, 행)
    K = geom.intrinsics_from_fov(60.0, 8, 6)
    depth = np.full((6, 8), 2.5)
    P = geom.unproject(K, geom.pixel_grid(6, 8), depth)
    assert P.shape == (6, 8, 3)
    np.testing.assert_allclose(P[..., 2], 2.5)
    # 주점 근처 두 픽셀(3,2)·(4,3)의 중점이 광축 위에 있다
    np.testing.assert_allclose((P[2, 3] + P[3, 4])[:2] / 2, [0.0, 0.0], atol=1e-12)
    x_step = P[0, 1, 0] - P[0, 0, 0]
    assert x_step == pytest.approx(2.5 / K[0, 0], rel=1e-12)           # 한 픽셀 = depth/fx


def test_visible_mask_half_pixel_boundaries():
    W, H = 10, 6
    uv = np.array([[-0.5, 0.0], [-0.51, 0.0], [9.49, 5.0], [9.5, 5.0], [4.0, -0.5], [4.0, 5.5], [4.0, 5.49], [4.0, 2.0]])
    z = np.ones(len(uv))
    got = geom.visible_mask(uv, z, W, H)
    np.testing.assert_array_equal(got, [True, False, True, False, True, False, True, True])
    assert not geom.visible_mask(np.array([[4.0, 2.0]]), np.array([-1.0]), W, H)[0]
    assert not geom.visible_mask(np.array([[4.0, 2.0]]), np.array([0.0]), W, H)[0]
    assert not geom.visible_mask(np.array([[np.nan, 2.0]]), np.array([1.0]), W, H)[0]


# ---------------------------------------------------------------------------------------------------- 호모그래피
def test_plane_homography_maps_plane_points():
    K1 = geom.intrinsics_from_fov(60.0, 320, 240)
    K2 = geom.intrinsics_from_fov(75.0, 400, 300)
    R = Rotation.from_rotvec([0.05, -0.1, 0.03]).as_matrix()
    t = np.array([0.4, -0.1, 0.2])
    n = np.array([0.1, -0.2, 1.0])
    n /= np.linalg.norm(n)
    d = 6.0
    # 평면 n·X = d (카메라1 좌표). 평면 위 점들을 만든다.
    a = np.cross(n, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    s = RNG.uniform(-1.5, 1.5, size=(40, 2))
    X1 = n * d + s[:, :1] * a + s[:, 1:] * b
    np.testing.assert_allclose(X1 @ n, d, atol=1e-12)
    uv1, _ = geom.project(K1, np.eye(3), np.zeros(3), X1)
    uv2, _ = geom.project(K2, R, t, X1)
    H = geom.plane_homography(K1, K2, R, t, n, d)
    assert H.shape == (3, 3) and H[2, 2] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(geom.apply_homography(H, uv1), uv2, atol=1e-7)


def test_plane_homography_special_cases():
    K = geom.intrinsics_from_fov(60.0, 320, 240)
    H0 = geom.plane_homography(K, K, np.eye(3), np.zeros(3), np.array([0.0, 0, 1]), 5.0)
    np.testing.assert_allclose(H0, np.eye(3), atol=1e-12)
    # 정면 평면 d=5, 옆으로 0.1 m 이동(X_dst = X_src + t): 모든 점이 u 방향으로 fx·0.1/5 px 이동
    H = geom.plane_homography(K, K, np.eye(3), np.array([0.1, 0, 0]), np.array([0.0, 0, 1]), 5.0)
    uv = RNG.uniform(0, 300, size=(10, 2))
    np.testing.assert_allclose(geom.apply_homography(H, uv) - uv, np.tile([K[0, 0] * 0.1 / 5.0, 0.0], (10, 1)), atol=1e-9)
    with pytest.raises(ValueError):
        geom.plane_homography(K, K, np.eye(3), np.zeros(3), np.array([0.0, 0, 1]), 0.0)
    with pytest.raises(ValueError):
        geom.plane_homography(K, K, np.eye(3), np.zeros(3), np.array([0.0, 0, 2.0]), 1.0)      # n 이 단위벡터가 아님
