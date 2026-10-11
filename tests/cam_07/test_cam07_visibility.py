"""CAM-07: 선분 가시성(차폐), 프러스텀+차폐 가시성 행렬, 깊이 버퍼 가시성."""

import numpy as np
import pytest

from cam07_helpers import K_pinhole, cl, dense_min_sdf, pose, poses, rot, seg_point_dist


# ---------------------------------------------------------------- segment_visible: 구
def test_segment_visible_sphere_cases():
    m = cl()
    s = [m.Sphere(center=[0, 0, 5], radius=1.0)]
    sv = lambda a, b: m.segment_visible(s, a, b)
    assert sv([0, 0, 0], [0, 0, 10]) is False                # 관통
    assert sv([0, 0, 0], [0, 0, 3]) is True                  # 구 앞에서 끝
    assert sv([0, 0, 0], [0, 0, 4]) is True                  # 끝점이 표면(바깥에서 본 표면 점은 보인다)
    assert sv([0, 0, 0], [0, 0, 6]) is False                 # 뒷면 점은 구에 가려진다
    assert sv([0, 0, 4], [0, 0, 0]) is True                  # 표면에서 바깥으로
    assert sv([0, 0, 4], [0, 0, 6]) is False                 # 표면에서 안쪽을 지나감
    assert sv([2, 0, 0], [2, 0, 10]) is True                 # 간격 2 > 반지름
    assert sv([0.9, 0, 0], [0.9, 0, 10]) is False
    assert sv([1.1, 0, 0], [1.1, 0, 10]) is True
    assert sv([0, 0, 4.8], [0, 0, 5.2]) is False             # 선분이 구 안에 완전히 포함
    assert sv([0, 0, 0], [0, 0, 0]) is True                  # 길이 0, 바깥
    assert m.segment_visible([], [0, 0, 0], [9, 9, 9]) is True
    assert sv([0, 0, 0], [0, 0, 10]) == sv([0, 0, 10], [0, 0, 0])          # 대칭


def test_segment_visible_box_and_halfspace():
    m = cl()
    thin = [m.Box(center=[0, 0, 5], half_extents=[1.0, 1.0, 0.1])]
    assert m.segment_visible(thin, [0.5, 0, 0], [0.5, 0, 10]) is False
    assert m.segment_visible(thin, [1.5, 0, 0], [1.5, 0, 10]) is True
    assert m.segment_visible(thin, [0, 0, 0], [0, 0, 4.9]) is True               # 앞면 위의 점은 보인다
    assert m.segment_visible(thin, [0, 0, 0], [0, 0, 5.0]) is False           # 고체 속 점은 가려짐
    assert m.segment_visible(thin, [0, 0, 0], [0, 0, 5.1]) is False           # 뒷면 위의 점은 가려짐
    assert m.segment_visible(thin, [0, 0, 5.1], [0, 0, 9]) is True            # 뒷면에서 바깥으로
    side = [m.Box(center=[0, 0, 5], half_extents=[1.0, 1.0, 0.1], rotation=rot([0, 1, 0], 90.0))]   # 얇은 축이 x 로
    assert m.segment_visible(side, [-2, 0, 5], [2, 0, 5]) is False
    assert m.segment_visible(side, [-2, 0, 3], [2, 0, 3]) is True
    assert m.segment_visible(side, [0.5, 0, 0], [0.5, 0, 10]) is True         # 폭 0.1 의 얇은 x 쪽을 벗어남
    floor = [m.HalfSpace(normal=[0, 1, 0], offset=0.0)]
    assert m.segment_visible(floor, [0, 1, 0], [5, 2, 0]) is True
    assert m.segment_visible(floor, [0, 1, 0], [5, -1, 0]) is False
    assert m.segment_visible(floor, [0, 0, 0], [1, 1, 0]) is True             # 바닥 위 점에서 위로
    assert m.segment_visible(floor, [0, 0, 0], [1, -1, 0]) is False
    assert m.segment_visible(floor, [0, 1, 0], [3, 0.0, 0]) is True           # 바닥 위 점을 바라봄
    assert m.segment_visible(floor, [0, -1, 0], [3, -2, 0]) is False          # 바닥 아래는 전부 가려짐


def test_segment_visible_broadcast_and_scalar_type():
    m = cl()
    s = [m.Sphere(center=[0, 0, 5], radius=1.0)]
    a = np.zeros((4, 1, 3))
    b = np.array([[0, 0, 10.0], [3, 0, 10.0], [0, 0, 3.0]])[None]
    r = m.segment_visible(s, a, b)
    assert r.shape == (4, 3) and r.dtype == bool
    assert np.all(r == np.array([False, True, True])[None])
    assert isinstance(m.segment_visible(s, [0, 0, 0], [0, 0, 10]), bool)
    assert m.segment_visible(s, np.zeros((7, 3)), np.tile([0, 0, 10.0], (7, 1))).shape == (7,)
    with pytest.raises(ValueError):
        m.segment_visible(s, [0, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        m.segment_visible(s, [0, 0, float("nan")], [0, 0, 1])


def test_segment_visible_random_vs_dense_sampling():
    m = cl()
    obs = [m.Sphere(center=[0, 0, 5], radius=1.0), m.Sphere(center=[2.5, 0.5, 3], radius=0.6),
           m.Box(center=[-2, 0, 4], half_extents=[0.5, 1.0, 0.4], rotation=rot([0, 1, 0], 25.0)),
           m.HalfSpace(normal=[0, -1, 0], offset=-2.0)]
    sdf = lambda x: m.clearance(obs, x)
    rng = np.random.default_rng(20)
    n_block = n_free = 0
    for _ in range(600):
        a = rng.uniform([-4, -3, -1], [4, 2.5, 7])
        b = rng.uniform([-4, -3, -1], [4, 2.5, 7])
        if sdf(a) < 0.02 or sdf(b) < 0.02:
            continue                                         # 끝점이 고체 안/표면 근처인 경우는 별도 테스트
        mn, _ = dense_min_sdf(sdf, a, b, n=8001)
        if abs(mn) < 2e-3:
            continue                                         # 접선 근처는 표본 해상도로 판정 불가
        got = m.segment_visible(obs, a, b)
        assert got == (mn > 0), (a, b, mn)
        n_block += mn < 0
        n_free += mn > 0
    assert n_block >= 30 and n_free >= 30                    # 두 경우를 모두 실제로 검증했다


# ---------------------------------------------------------------- 카메라-점 가시성 행렬
def test_visibility_matrix_occlusion_sweep_vs_analytic():
    """카메라가 x 로 훑을 때 구 차폐기에 가려지는 프레임을 점-선 거리 해석식과 비교."""
    m = cl()
    occ = m.Sphere(center=[0, 0, 2.0], radius=0.5)
    xs = np.linspace(-2, 2, 41)
    T = poses([[x, 0, 0] for x in xs])
    tgt = np.array([[0.0, 0.0, 4.0]])
    vis = m.visibility_matrix([occ], T, tgt)
    assert vis.shape == (41, 1) and vis.dtype == bool
    want = np.array([seg_point_dist([0, 0, 2.0], [x, 0, 0], [0, 0, 4.0]) >= 0.5 for x in xs])
    # 해석식 경계(|x| = 2*0.5/cos) 근방은 제외하고 비교
    d = np.array([seg_point_dist([0, 0, 2.0], [x, 0, 0], [0, 0, 4.0]) for x in xs])
    mask = np.abs(d - 0.5) > 1e-3
    assert np.array_equal(vis[mask, 0], want[mask])
    assert (~vis[:, 0]).sum() >= 5 and vis[:, 0].sum() >= 5
    assert vis[0, 0] and vis[-1, 0] and not vis[20, 0]


def test_visibility_matrix_frustum_edges_and_behind():
    m = cl()
    W, H = 64, 48
    K = K_pinhole(100.0, W, H)
    T = poses([[0, 0, 0]])
    # u = 100*x/z + 31.5 ; 오른쪽 가장자리 u = W-0.5 = 63.5 <=> x/z = 0.32
    pts = np.array([[0.0, 0.0, 5.0],            # 중앙
                    [0.32 * 5 - 1e-3, 0, 5.0],  # 오른쪽 가장자리 안
                    [0.32 * 5 + 1e-3, 0, 5.0],  # 밖
                    [-0.32 * 5 + 1e-3, 0, 5.0],  # 왼쪽 안 (u=-0.5 에서 안쪽)
                    [-0.32 * 5 - 1e-3, 0, 5.0],  # 왼쪽 밖
                    [0.0, 0.235 * 5 - 1e-3, 5.0],  # v = 100*y/z + 23.5 <= 47.5 <=> y/z <= 0.24
                    [0.0, 0.24 * 5 + 1e-3, 5.0],
                    [0.0, 0.0, -5.0],           # 카메라 뒤
                    [0.0, 0.0, 0.0]])           # 카메라 위치(z=0 <= near)
    v = m.visibility_matrix([], T, pts, K=K, size=(W, H))[0]
    assert list(v) == [True, True, False, True, False, True, False, False, False]
    assert list(m.visibility_matrix([], T, pts)[0]) == [True] * 7 + [False, False]     # K 없음 -> 앞쪽(z>near)만
    v2 = m.visibility_matrix([], T, pts[:3], K=K, size=(W, H), margin_px=2.0)[0]       # 안쪽 2 px 여유
    assert list(v2) == [True, False, False]
    v3 = m.visibility_matrix([], T, pts[[0, 3, 4, 5]], K=K, size=(W, H), margin_px=2.0)[0]   # 왼쪽·아래쪽 가장자리도 여유 적용
    assert list(v3) == [True, False, False, False]
    # 넓은 near
    far = m.visibility_matrix([], T, np.array([[0.0, 0.0, 0.5]]), near=1.0)
    assert far.shape == (1, 1) and not far[0, 0]


def test_visibility_matrix_rotated_camera():
    m = cl()
    W, H = 64, 48
    K = K_pinhole(100.0, W, H)
    R = rot([0, 1, 0], 90.0)                                  # c2w: 카메라 +z 가 세계 +x 를 향함
    T = poses([[1.0, 2.0, 3.0]], R)
    p_front = np.array([[6.0, 2.0, 3.0]])                     # 세계 +x 쪽 5 m
    p_back = np.array([[-4.0, 2.0, 3.0]])
    assert m.visibility_matrix([], T, p_front, K=K, size=(W, H))[0, 0]
    assert not m.visibility_matrix([], T, p_back, K=K, size=(W, H))[0, 0]
    # 카메라 위치가 차폐기 안이면 아무것도 안 보인다
    inside = m.Sphere(center=[1.0, 2.0, 3.0], radius=0.5)
    assert not m.visibility_matrix([inside], T, p_front)[0, 0]


def test_visible_fraction_and_shapes():
    m = cl()
    occ = [m.Sphere(center=[0, 0, 2.0], radius=0.5)]
    T = poses([[x, 0, 0] for x in (-3.0, 0.0, 3.0)])
    pts = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 1.0], [0.0, 3.0, 4.0], [0.0, -3.0, 4.0]])
    M = m.visibility_matrix(occ, T, pts)
    assert M.shape == (3, 4)
    fr = m.visible_fraction(occ, T, pts)
    assert fr.shape == (3,) and np.allclose(fr, M.mean(axis=1))
    assert np.allclose(fr, [1.0, 0.75, 1.0])                  # 가운데 프레임만 (0,0,4) 가려짐
    assert m.visibility_matrix(occ, T[0], pts[:1]).shape == (1, 1)
    with pytest.raises(ValueError):
        m.visibility_matrix(occ, T, pts, K=K_pinhole(100.0, 8, 8))                 # K 만 주고 size 없음
    with pytest.raises(ValueError):
        m.visibility_matrix(occ, T, pts, size=(8, 8))
    bad = T.copy()
    bad[1, :3, :3] *= 2.0                                     # 비회전
    with pytest.raises(ValueError):
        m.visibility_matrix(occ, bad, pts)
    with pytest.raises(ValueError):
        m.visibility_matrix(occ, T, np.zeros((2, 2)))


# ---------------------------------------------------------------- 깊이 버퍼 가시성
def test_depth_visible_wall():
    m = cl()
    W, H = 64, 48
    K = K_pinhole(100.0, W, H)
    depth = np.full((H, W), 5.0, np.float32)
    T = pose([0, 0, 0])
    pts = np.array([[0.0, 0.0, 4.0],      # 벽 앞: 보임
                    [0.0, 0.0, 6.0],      # 벽 뒤: 가려짐
                    [0.0, 0.0, 5.04],     # 허용오차 1% 이내
                    [0.0, 0.0, 5.3],      # 허용오차 밖
                    [3.0, 0.0, 4.0],      # 화면 밖(u = 100*0.75+31.5)
                    [0.0, 0.0, -1.0],     # 카메라 뒤
                    [0.0, 0.0, 5.0]])     # 정확히 벽 위
    got = m.depth_visible(pts, K, T, depth)
    assert got.dtype == bool and got.shape == (7,)
    assert list(got) == [True, False, True, False, False, False, True]
    tight = m.depth_visible(pts[[2]], K, T, depth, rel_tol=0.0)
    assert not tight[0]
    assert m.depth_visible(pts[[2]], K, T, depth, rel_tol=0.0, abs_tol=0.05)[0]


def test_depth_visible_sky_nearest_pixel_and_pose():
    m = cl()
    W, H = 32, 24
    K = K_pinhole(50.0, W, H)
    depth = np.full((H, W), np.inf, np.float32)               # 하늘 +inf 허용
    depth[:, :16] = 3.0                                       # 왼쪽 절반은 벽
    T = pose([0, 0, 0])
    # u = 50*x/z + 15.5 ; 픽셀 열 = round(u)
    left = np.array([[(-5 / 50.0) * 6.0, 0.0, 6.0]])          # u = 10.5 -> 열 10 or 11 (벽): z=6 > 3 -> 가려짐
    right = np.array([[(5 / 50.0) * 6.0, 0.0, 6.0]])          # u = 20.5 -> 하늘: 보임
    assert not m.depth_visible(left, K, T, depth)[0]
    assert m.depth_visible(right, K, T, depth)[0]
    # 카메라 위치·자세를 따른다: 벽 깊이는 카메라 기준 z
    d2 = np.full((H, W), 3.0, np.float32)
    T2 = pose([0, 0, 2.0])
    p2 = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0]])         # 카메라 기준 z = 2, 4
    assert list(m.depth_visible(p2, K, T2, d2)) == [True, False]
    T3 = pose([0, 0, 0], rot([0, 1, 0], 90.0))                 # 카메라 +z 가 세계 +x
    p3 = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert list(m.depth_visible(p3, K, T3, d2)) == [True, False, False]


def test_depth_visible_validation():
    m = cl()
    K = K_pinhole(50.0, 8, 8)
    pts = np.array([[0.0, 0.0, 1.0]])
    d = np.full((8, 8), 2.0)
    bad = d.copy()
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        m.depth_visible(pts, K, pose([0, 0, 0]), bad)          # NaN 깊이 금지(FND-02)
    with pytest.raises(ValueError):
        m.depth_visible(pts, K, pose([0, 0, 0]), np.zeros((8, 8, 1)))
    with pytest.raises(ValueError):
        m.depth_visible(np.zeros((2, 2)), K, pose([0, 0, 0]), d)
