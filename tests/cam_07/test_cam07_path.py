"""CAM-07: 경로 클리어런스(인증 하한·터널링), 충돌 판정, free_distance, 점수, 경로 보고서."""
import math

import numpy as np
import pytest

from cam07_helpers import cl, dense_min_sdf, poses, polyline_sphere_clearance, rot


def _line(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


# ---------------------------------------------------------------- camera_positions
def test_camera_positions_shapes():
    m = cl()
    P = np.random.default_rng(0).normal(size=(5, 3))
    T = poses(P, rot([0, 1, 0], 20))
    assert np.allclose(m.camera_positions(T), P) and m.camera_positions(T).shape == (5, 3)
    assert np.allclose(m.camera_positions(P), P)
    assert m.camera_positions(T[0]).shape == (1, 3) and np.allclose(m.camera_positions(T[0])[0], P[0])
    assert m.camera_positions(P[0]).shape == (1, 3)
    for bad in (np.zeros((0, 3)), np.zeros((3, 5)), np.zeros(4), np.full((2, 4, 4), np.nan)):
        with pytest.raises(ValueError):
            m.camera_positions(bad)
    T2 = T.copy()
    T2[2, 3, :] = [0, 0, 1, 1]
    with pytest.raises(ValueError):
        m.camera_positions(T2)


# ---------------------------------------------------------------- 구: 해석해
def test_path_clearance_sphere_analytic():
    m = cl()
    c, r = np.array([0.0, 1.0, 0.0]), 0.3
    obs = [m.Sphere(center=c, radius=r)]
    P = _line([-2, 0, 0], [2, 0, 0], 9)               # 프레임 간격 0.5 m, 최소는 4번 프레임(x=0)
    pc = m.path_clearance(obs, P, tol=1e-6)
    want = polyline_sphere_clearance(c, r, P)
    assert want == pytest.approx(0.7, abs=1e-12)
    assert pc.min_clearance - want <= 1e-6 + 1e-12 and pc.min_clearance >= want - 1e-12
    assert pc.lower_bound <= want + 1e-12 and pc.min_clearance - pc.lower_bound <= 1e-6 + 1e-12
    assert pc.at_frame == pytest.approx(4.0, abs=0.01)
    assert np.allclose(pc.point, [0, 0, 0], atol=0.01)
    assert m.clearance(obs, np.array(pc.point)) == pytest.approx(pc.min_clearance, abs=1e-9)


def test_path_clearance_frame_in_between_samples():
    """최소가 프레임 사이(비대칭 위치)에 있을 때: 프레임 점만 보면 놓친다. 하한은 진짜 최소 이하여야 한다."""
    m = cl()
    c, r = np.array([0.3, 0.5, 0.0]), 0.2
    obs = [m.Sphere(center=c, radius=r)]
    P = np.array([[-1.0, 0, 0], [1.0, 0, 0]])         # 두 프레임뿐, 진짜 최소는 x=0.3 (s=0.65), 값 0.3
    pc = m.path_clearance(obs, P, tol=1e-2)            # 느슨한 tol: 하한이 상한과 달라야 하고 진짜 값을 감싸야 한다
    assert 0.3 - 1e-12 <= pc.min_clearance <= 0.3 + 1e-2 + 1e-12
    assert pc.lower_bound <= 0.3 + 1e-12 and pc.min_clearance - pc.lower_bound <= 1e-2 + 1e-12
    pc2 = m.path_clearance(obs, P, tol=1e-6)
    assert 0.3 - 1e-12 <= pc2.min_clearance <= 0.3 + 1e-6 + 1e-12 and pc2.lower_bound <= 0.3 + 1e-12
    assert pc2.at_frame == pytest.approx(0.65, abs=0.01)
    assert min(m.clearance(obs, P[0]), m.clearance(obs, P[1])) > 0.6      # 프레임 점 자체의 clearance 는 크다


def test_path_clearance_random_sphere_paths_vs_analytic():
    m = cl()
    rng = np.random.default_rng(10)
    for _ in range(25):
        c = rng.uniform(-1, 1, 3)
        r = rng.uniform(0.2, 0.6)
        P = rng.uniform(-2, 2, size=(int(rng.integers(2, 7)), 3))
        pc = m.path_clearance([m.Sphere(center=c, radius=r)], P, tol=1e-5)
        want = polyline_sphere_clearance(c, r, P)
        assert pc.lower_bound - 1e-9 <= want <= pc.min_clearance + 1e-9
        assert pc.min_clearance - want <= 1e-5 + 1e-9
        assert 0.0 <= pc.at_frame <= len(P) - 1


# ---------------------------------------------------------------- 상자·혼합: 조밀 표본 대조
def test_path_clearance_mixed_scene_vs_dense_sampling():
    m = cl()
    obs = [m.Sphere(center=[0, 0, 5], radius=1.0),
           m.Box(center=[2, 0, 3], half_extents=[0.4, 1.0, 0.3], rotation=rot([0, 1, 0], 30.0)),
           m.HalfSpace(normal=[0, -1, 0], offset=-1.5)]
    rng = np.random.default_rng(11)
    for _ in range(12):
        P = rng.uniform(-3, 4, size=(4, 3))
        pc = m.path_clearance(obs, P, tol=1e-4)
        ref = min(dense_min_sdf(lambda x: m.clearance(obs, x), P[i], P[i + 1])[0] for i in range(3))
        # 조밀 표본의 최소는 진짜 최소의 상계, 구현의 하한은 진짜 최소의 하계
        assert pc.lower_bound <= ref + 1e-9
        assert pc.min_clearance <= ref + 1e-4 + 1e-9                 # 진짜 최소 + tol 이내
        assert pc.min_clearance - pc.lower_bound <= 1e-4 + 1e-12


def test_tunneling_thin_wall_between_frames_is_detected():
    m = cl()
    wall = m.Box(center=[0, 0, 0], half_extents=[0.01, 2.0, 2.0])      # 두께 2 cm 벽
    P = np.array([[-3.0, 0, 0], [3.0, 0, 0]])                         # 두 프레임 모두 벽에서 멀다
    assert m.clearance([wall], P[0]) > 2.9 and m.clearance([wall], P[1]) > 2.9
    pc = m.path_clearance([wall], P)
    assert pc.min_clearance <= -0.01 + 1e-4 + 1e-12                    # 벽 중심면(sdf=-0.01)을 지나감
    assert pc.lower_bound <= -0.01 + 1e-12 and pc.min_clearance - pc.lower_bound <= 1e-4 + 1e-12
    assert not m.collision_free([wall], P, camera_radius=0.0)
    assert not m.collision_free([wall], P)
    # 벽을 비껴 가는 경로는 통과
    P2 = np.array([[-3.0, 2.6, 0], [3.0, 2.6, 0]])
    assert m.collision_free([wall], P2)


def test_floor_halfspace_path_clearance_endpoints_exact():
    m = cl()
    floor = m.HalfSpace(normal=[0, 1, 0], offset=0.0)
    P = np.array([[0, 1.0, 0], [1, 0.4, 0], [2, 2.0, 0]])
    pc = m.path_clearance([floor], P, tol=1e-6)
    assert pc.min_clearance == pytest.approx(0.4, abs=1e-6) and pc.at_frame == pytest.approx(1.0, abs=1e-3)
    assert pc.lower_bound <= 0.4 + 1e-12


# ---------------------------------------------------------------- 충돌 판정
def test_collision_free_threshold_semantics():
    m = cl()
    s = m.Sphere(center=[0, 1.0, 0], radius=0.3)                       # 경로 y=0 에서 clearance 0.7
    P = _line([-2, 0, 0], [2, 0, 0], 9)
    assert m.collision_free([s], P, camera_radius=0.6, margin=0.0)
    assert m.collision_free([s], P, camera_radius=0.5, margin=0.19)
    assert not m.collision_free([s], P, camera_radius=0.5, margin=0.21)   # 0.5 + 0.21 > 0.7
    assert not m.collision_free([s], P, camera_radius=0.8)
    assert m.collision_free([s], P, camera_radius=0.6995) and not m.collision_free([s], P, camera_radius=0.7005)  # tol(1e-4) 바깥에서 정확
    assert m.collision_free([s], P)                                      # 기본 반경 0.10
    assert m.collision_free([], P)                                       # 장애물 없음
    # 경로가 구를 관통
    Q = _line([0, -2, 0], [0, 2, 0], 5)
    assert not m.collision_free([s], Q, camera_radius=0.0)
    # 소리성: 자유라고 답했으면 조밀 표본의 진짜 최소가 임계 이상
    P3 = np.array([[-2.0, 0.0, 0.0], [2.0, 0.55, 0.0]])
    for rad in (0.05, 0.2, 0.3, 0.4):
        if m.collision_free([s], P3, camera_radius=rad):
            assert dense_min_sdf(lambda x: m.clearance([s], x), P3[0], P3[1])[0] >= rad - 1e-9


def test_collision_free_accepts_c2w_and_matches_positions():
    m = cl()
    s = m.Sphere(center=[0, 1.0, 0], radius=0.3)
    P = _line([-2, 0, 0], [2, 0, 0], 9)
    T = poses(P, rot([1, 0, 0], 33.0))
    for rad in (0.2, 0.69, 0.71):
        assert m.collision_free([s], T, camera_radius=rad) == m.collision_free([s], P, camera_radius=rad)
    a, b = m.path_clearance([s], T), m.path_clearance([s], P)
    assert a.min_clearance == b.min_clearance and a.at_frame == b.at_frame


def test_single_frame_and_empty_obstacles():
    m = cl()
    s = m.Sphere(center=[0, 0, 0], radius=1.0)
    pc = m.path_clearance([s], [[0.0, 0.0, 3.0]])
    assert pc.min_clearance == pytest.approx(2.0, abs=1e-12) and pc.lower_bound <= 2.0 + 1e-12 and pc.at_frame == 0.0
    pe = m.path_clearance([], _line([0, 0, 0], [1, 1, 1], 4))
    assert pe.min_clearance == math.inf and pe.lower_bound == math.inf and pe.at_frame == 0.0


def test_path_clearance_validation():
    m = cl()
    s = m.Sphere(center=[0, 0, 0], radius=1.0)
    P = _line([0, 0, 3], [0, 0, 5], 3)
    for tol in (0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.path_clearance([s], P, tol=tol)
    for kw in ({"camera_radius": -0.1}, {"margin": -0.1}):
        with pytest.raises(ValueError):
            m.collision_free([s], P, **kw)
    with pytest.raises(ValueError):
        m.path_clearance([s], np.zeros((0, 3)))


def test_path_clearance_deterministic_bitwise():
    m = cl()
    obs = [m.Sphere(center=[0, 0, 5], radius=1.0), m.Box(center=[1, 0, 3], half_extents=[0.3, 0.3, 0.3])]
    P = np.random.default_rng(12).uniform(-3, 4, size=(8, 3))
    a, b = m.path_clearance(obs, P), m.path_clearance(obs, P)
    assert (a.min_clearance, a.lower_bound, a.at_frame, tuple(a.point)) == (b.min_clearance, b.lower_bound, b.at_frame, tuple(b.point))


# ---------------------------------------------------------------- free_distance
def test_free_distance_head_on():
    m = cl()
    s = m.Sphere(center=[0, 0, 10], radius=2.0)
    d = m.free_distance([s], [0, 0, 0], [0, 0, 3.0], radius=0.1)       # 방향은 정규화된다
    assert d == pytest.approx(7.9, abs=1e-5) and d <= 7.9 + 1e-9        # 보수적: 진짜 거리 이하
    b = m.Box(center=[0, 0, 6], half_extents=[1, 1, 1])
    assert m.free_distance([b], [0, 0, 0], [0, 0, 1], radius=0.25) == pytest.approx(4.75, abs=1e-5)
    both = m.free_distance([s, b], [0, 0, 0], [0, 0, 1], radius=0.25)
    assert both == pytest.approx(4.75, abs=1e-5)


def test_free_distance_oblique_floor_and_away_and_cap():
    m = cl()
    floor = m.HalfSpace(normal=[0, 1, 0], offset=0.0)
    th = math.radians(30.0)
    d = m.free_distance([floor], [0, 2.0, 0], [math.cos(th), -math.sin(th), 0], radius=0.1)
    assert d == pytest.approx((2.0 - 0.1) / math.sin(th), abs=1e-4) and d <= (1.9 / math.sin(th)) + 1e-9
    away = m.free_distance([floor], [0, 2.0, 0], [0, 1.0, 0], radius=0.1, max_dist=50.0)
    assert away == pytest.approx(50.0)                                   # 멀어지면 상한
    assert m.free_distance([], [0, 0, 0], [1, 0, 0], max_dist=12.0) == pytest.approx(12.0)
    inside = m.free_distance([m.Sphere(center=[0, 0, 1], radius=1.0)], [0, 0, 0.9], [0, 0, 1])
    assert inside == 0.0                                                  # 이미 위반이면 0


def test_free_distance_consistent_with_collision_free():
    m = cl()
    obs = [m.Sphere(center=[0.3, 0.1, 8], radius=1.5), m.HalfSpace(normal=[0, -1, 0], offset=-3.0)]
    o, dvec = np.zeros(3), np.array([0.05, 0.02, 1.0])
    dvec = dvec / np.linalg.norm(dvec)
    rad = 0.2
    fd = m.free_distance(obs, o, dvec, radius=rad)
    assert 0 < fd < 100
    assert m.collision_free(obs, np.stack([o, o + (fd - 1e-3) * dvec]), camera_radius=rad)
    assert not m.collision_free(obs, np.stack([o, o + (fd + 1e-2) * dvec]), camera_radius=rad)


def test_free_distance_validation():
    m = cl()
    s = m.Sphere(center=[0, 0, 5], radius=1.0)
    for kw in ({"direction": [0, 0, 0]}, {"direction": [0, 0, float("nan")]}, {"direction": [0, 1]}, {"radius": -1.0},
               {"max_dist": 0.0}):
        args = dict(direction=[0, 0, 1], radius=0.1)
        args.update(kw)
        with pytest.raises(ValueError):
            m.free_distance([s], [0, 0, 0], **args)


# ---------------------------------------------------------------- 점수
def test_clearance_score_properties():
    m = cl()
    r, w = 0.1, 0.5
    f = lambda c: m.clearance_score(c, radius=r, comfort_m=w)
    assert f(r) == 0.0 and f(-3.0) == 0.0 and f(r + w) == pytest.approx(1.0) and f(9.0) == 1.0
    assert f(math.inf) == 1.0
    assert f(r + w / 2) == pytest.approx(0.5, abs=1e-12)               # 스무스스텝 3x^2-2x^3 의 중점
    xs = np.linspace(-0.2, 1.0, 241)
    ys = m.clearance_score(xs, radius=r, comfort_m=w)
    assert ys.shape == xs.shape and np.all(np.diff(ys) >= -1e-15) and ys.min() >= 0 and ys.max() <= 1
    x = 0.3
    assert f(r + x * w) == pytest.approx(3 * x * x - 2 * x ** 3, abs=1e-12)
    assert m.clearance_score(0.35) == pytest.approx(f(0.35))           # 기본 반경·편안 거리
    assert isinstance(f(0.3), float)
    for kw in ({"comfort_m": 0.0}, {"comfort_m": -1.0}, {"radius": -0.1}):
        with pytest.raises(ValueError):
            m.clearance_score(0.3, **kw)
    with pytest.raises(ValueError):
        m.clearance_score(float("nan"))


# ---------------------------------------------------------------- 보고서
def test_evaluate_path_report():
    m = cl()
    s = m.Sphere(center=[0, 1.0, 0], radius=0.3)
    P = _line([-2, 0, 0], [2, 0, 0], 9)
    T = poses(P)
    rep = m.evaluate_path([s], T, camera_radius=0.1, margin=0.1, comfort_m=0.5)
    assert rep.ok and not rep.collision and rep.min_visible_fraction is None
    assert rep.min_clearance == pytest.approx(0.7, abs=2e-4) and rep.worst_frame == pytest.approx(4.0, abs=0.1)
    assert rep.lower_bound <= 0.7 + 1e-12
    assert rep.score == pytest.approx(m.clearance_score(rep.min_clearance, radius=0.1, comfort_m=0.5))
    assert rep.score == 1.0
    bad = m.evaluate_path([s], T, camera_radius=0.5, margin=0.25)
    assert (not bad.ok) and bad.collision and 0.0 <= bad.score < 1.0
    hit = m.evaluate_path([s], poses(_line([0, -2, 0], [0, 2, 0], 5)))
    assert hit.collision and (not hit.ok) and hit.score == 0.0 and hit.min_clearance < 0
    # 가시성 조건
    pts = np.array([[0.0, 0.0, 4.0]])
    ok_vis = m.evaluate_path([s], T, subject_points=pts, min_visible=0.5)
    assert ok_vis.min_visible_fraction == pytest.approx(1.0) and ok_vis.ok
    wall = m.Sphere(center=[0, 0, 2.0], radius=1.5)
    blocked = m.evaluate_path([wall], poses(_line([0, 0, -3], [0, 0, -2], 3)), camera_radius=0.0, subject_points=pts, min_visible=0.5)
    assert blocked.min_visible_fraction == 0.0 and not blocked.ok and not blocked.collision
