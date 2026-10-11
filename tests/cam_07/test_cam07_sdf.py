"""CAM-07: 장애물 프리미티브의 부호거리(해석해·브루트포스 대조·립시츠), 장면 clearance, 검증, 직렬화."""
import json
import math

import numpy as np
import pytest

from cam07_helpers import (box_surface_samples, cl, ref_box_sdf_aabb, ref_sphere_sdf, rot)


# ---------------------------------------------------------------- 상수
def test_constants():
    m = cl()
    assert m.CAMERA_RADIUS_M == pytest.approx(0.10)
    assert m.COMFORT_M == pytest.approx(0.5)
    assert m.SURF_TOL == pytest.approx(1e-6)
    assert m.PATH_TOL_M == pytest.approx(1e-4)
    assert tuple(m.OBSTACLE_TYPES) == ("sphere", "box", "halfspace")
    for n in ("Sphere", "Box", "HalfSpace", "clearance", "path_clearance", "collision_free", "free_distance",
              "segment_visible", "visibility_matrix", "visible_fraction", "depth_visible", "clearance_score",
              "evaluate_path", "obstacle_from_dict", "camera_positions", "PathClearance", "ClearanceReport"):
        assert n in m.__all__ and hasattr(m, n)


# ---------------------------------------------------------------- 구
def test_sphere_sdf_analytic():
    m = cl()
    s = m.Sphere(center=[1.0, 2.0, 3.0], radius=0.5)
    assert s.signed_distance([1.0, 2.0, 3.0]) == pytest.approx(-0.5, abs=1e-12)
    assert s.signed_distance([1.0, 2.0, 5.0]) == pytest.approx(1.5, abs=1e-12)
    assert s.signed_distance([1.0, 2.0, 3.5]) == pytest.approx(0.0, abs=1e-12)
    assert s.signed_distance([1.0, 2.25, 3.0]) == pytest.approx(-0.25, abs=1e-12)
    rng = np.random.default_rng(1)
    p = rng.normal(size=(50, 3)) * 3
    assert np.allclose(s.signed_distance(p), ref_sphere_sdf(np.array([1.0, 2.0, 3.0]), 0.5, p), atol=1e-12)
    assert s.signed_distance(p).shape == (50,)
    assert s.signed_distance(p.reshape(5, 10, 3)).shape == (5, 10)


# ---------------------------------------------------------------- 상자
def test_box_sdf_axis_aligned_known_values():
    m = cl()
    b = m.Box(center=[0, 0, 0], half_extents=[0.5, 0.5, 0.5])
    cases = [((1, 0, 0), 0.5), ((1, 1, 0), math.sqrt(0.5)), ((1, 1, 1), math.sqrt(0.75)),
             ((0.1, 0, 0), -0.4), ((0.4, 0.1, 0), -0.1), ((0, 0, 0), -0.5), ((0.5, 0, 0), 0.0), ((0, -3, 0), 2.5)]
    for p, want in cases:
        assert b.signed_distance(p) == pytest.approx(want, abs=1e-12), p
    h = np.array([1.0, 0.5, 0.25])
    b2 = m.Box(center=[0, 0, 0], half_extents=h)
    rng = np.random.default_rng(2)
    p = rng.uniform(-2, 2, size=(200, 3))
    assert np.allclose(b2.signed_distance(p), ref_box_sdf_aabb(h, p), atol=1e-12)


def test_box_rotation_and_translation_equivariance():
    m = cl()
    h = np.array([1.0, 0.5, 0.25])
    R = rot([1, 2, 3], 37.0)
    c = np.array([0.3, -1.0, 2.0])
    b = m.Box(center=c, half_extents=h, rotation=R)
    rng = np.random.default_rng(3)
    q = rng.uniform(-2.5, 2.5, size=(300, 3))          # 상자 국소 좌표의 점
    world = q @ R.T + c
    assert np.allclose(b.signed_distance(world), ref_box_sdf_aabb(h, q), atol=1e-10)
    # 90도 회전 = 축 교환
    b90 = m.Box(center=[0, 0, 0], half_extents=[1.0, 0.5, 0.25], rotation=rot([0, 0, 1], 90.0))
    bswap = m.Box(center=[0, 0, 0], half_extents=[0.5, 1.0, 0.25])
    assert np.allclose(b90.signed_distance(q), bswap.signed_distance(q), atol=1e-10)


def test_box_sdf_vs_bruteforce_surface_distance():
    m = cl()
    h = np.array([0.5, 0.3, 0.2])
    R = rot([0.3, -1, 0.5], 52.0)
    c = np.array([0.1, 0.2, -0.3])
    b = m.Box(center=c, half_extents=h, rotation=R)
    S = box_surface_samples(c, h, R, n=61)
    spacing = 2 * h.max() / 60
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1.2, 1.2, size=(40, 3)) + c
    sdf = b.signed_distance(pts)
    for p, v in zip(pts, sdf):
        d = np.linalg.norm(S - p, axis=1).min()
        assert abs(abs(v) - d) <= spacing * 0.75 + 1e-9, (p, v, d)
        inside = np.all(np.abs((p - c) @ R) < h)
        assert (v < 0) == bool(inside)


# ---------------------------------------------------------------- 반공간
def test_halfspace_sdf_and_normalization():
    m = cl()
    f = m.HalfSpace(normal=[0, 2.0, 0], offset=2.0)      # 정규화 -> n=(0,1,0), offset=1
    assert np.allclose(f.normal, [0, 1, 0]) and f.offset == pytest.approx(1.0)
    assert f.signed_distance([5, 3.0, -2]) == pytest.approx(2.0)      # 자유 쪽 n.x > offset
    assert f.signed_distance([0, 0.25, 0]) == pytest.approx(-0.75)    # 고체 안
    assert f.signed_distance([9, 1.0, 9]) == pytest.approx(0.0)
    g = m.HalfSpace(normal=[1, 1, 0], offset=0.0)
    assert g.signed_distance([1, 1, 7]) == pytest.approx(math.sqrt(2.0))


# ---------------------------------------------------------------- 립시츠, 장면 clearance
def _mixed(m):
    return [m.Sphere(center=[0, 0, 5], radius=1.0),
            m.Box(center=[2, 0, 3], half_extents=[0.4, 1.0, 0.3], rotation=rot([0, 1, 0], 30.0)),
            m.HalfSpace(normal=[0, -1, 0], offset=-1.5)]


def test_scene_clearance_is_min_and_one_lipschitz():
    m = cl()
    obs = _mixed(m)
    rng = np.random.default_rng(5)
    p = rng.uniform(-3, 4, size=(400, 3))
    want = np.min([o.signed_distance(p) for o in obs], axis=0)
    got = m.clearance(obs, p)
    assert got.shape == (400,) and np.allclose(got, want, atol=1e-12)
    q = p + rng.normal(size=p.shape) * 0.3
    dq = np.linalg.norm(q - p, axis=1)
    for o in obs + [None]:
        a = m.clearance(obs, p) if o is None else o.signed_distance(p)
        b = m.clearance(obs, q) if o is None else o.signed_distance(q)
        assert np.all(np.abs(a - b) <= dq + 1e-9)          # 1-립시츠(경로 클리어런스 상한 증명의 전제)


def test_clearance_scalar_empty_and_shapes():
    m = cl()
    obs = _mixed(m)
    v = m.clearance(obs, [0.0, 0.0, 0.0])
    assert isinstance(v, float) and v == pytest.approx(float(np.min([o.signed_distance([0.0, 0.0, 0.0]) for o in obs])))
    assert m.clearance([], [1.0, 2.0, 3.0]) == math.inf
    e = m.clearance([], np.zeros((4, 3)))
    assert e.shape == (4,) and np.all(np.isinf(e))
    assert m.clearance(obs, np.zeros((2, 3, 3))).shape == (2, 3)


def test_inputs_not_mutated():
    m = cl()
    obs = _mixed(m)
    p = np.random.default_rng(6).normal(size=(10, 3))
    p0 = p.copy()
    m.clearance(obs, p)
    path = np.random.default_rng(7).normal(size=(6, 3)) * 2
    path0 = path.copy()
    m.path_clearance(obs, path)
    m.collision_free(obs, path)
    assert np.array_equal(p, p0) and np.array_equal(path, path0)
    c = np.array([0.0, 0.0, 5.0])
    s = m.Sphere(center=c, radius=1.0)
    c[0] = 99.0                                           # 입력 배열 변경이 객체에 새면 안 된다(복사 보관)
    assert s.center[0] == 0.0


# ---------------------------------------------------------------- 검증
def test_validation_errors():
    m = cl()
    bad = [lambda: m.Sphere(center=[0, 0, 0], radius=0.0), lambda: m.Sphere(center=[0, 0, 0], radius=-1.0),
           lambda: m.Sphere(center=[0, 0, float("nan")], radius=1.0), lambda: m.Sphere(center=[0, 0], radius=1.0),
           lambda: m.Sphere(center=[0, 0, 0], radius=float("inf")),
           lambda: m.Box(center=[0, 0, 0], half_extents=[1, 0, 1]),
           lambda: m.Box(center=[0, 0, 0], half_extents=[1, 1, -1]),
           lambda: m.Box(center=[0, 0, 0], half_extents=[1, 1, 1], rotation=2.0 * np.eye(3)),
           lambda: m.Box(center=[0, 0, 0], half_extents=[1, 1, 1], rotation=np.diag([1.0, 1.0, -1.0])),   # 반사(det -1)
           lambda: m.Box(center=[0, 0, 0], half_extents=[1, 1, 1], rotation=np.eye(2)),
           lambda: m.HalfSpace(normal=[0, 0, 0], offset=0.0),
           lambda: m.HalfSpace(normal=[0, 1, 0], offset=float("inf"))]
    for f in bad:
        with pytest.raises(ValueError):
            f()
    s = m.Sphere(center=[0, 0, 0], radius=1.0)
    with pytest.raises(ValueError):
        s.signed_distance([0.0, 1.0])
    with pytest.raises(ValueError):
        s.signed_distance([0.0, float("nan"), 1.0])
    with pytest.raises(ValueError):
        m.clearance([s], [[0.0, 0.0]])
    with pytest.raises(TypeError):
        m.Sphere([0, 0, 0], 1.0)                          # kw-only


def test_dict_roundtrip_and_errors():
    m = cl()
    obs = _mixed(m)
    for o in obs:
        d = o.to_dict()
        d2 = json.loads(json.dumps(d))                    # JSON 가능
        o2 = m.obstacle_from_dict(d2)
        assert type(o2) is type(o) and d["type"] in m.OBSTACLE_TYPES
        p = np.random.default_rng(8).uniform(-3, 3, size=(30, 3))
        assert np.allclose(o.signed_distance(p), o2.signed_distance(p), atol=1e-12)
        assert o2.to_dict() == json.loads(json.dumps(o2.to_dict()))
    d = m.Box(center=[0, 0, 0], half_extents=[1, 1, 1]).to_dict()
    d.pop("rotation")                                      # rotation 은 선택 키
    assert np.allclose(m.obstacle_from_dict(d).rotation, np.eye(3))
    for bad in ({"type": "cone", "center": [0, 0, 0], "radius": 1.0},
                {"type": "sphere", "center": [0, 0, 0]},
                {"type": "sphere", "center": [0, 0, 0], "radius": 1.0, "extra": 1},
                {"center": [0, 0, 0], "radius": 1.0}, [1, 2]):
        with pytest.raises(ValueError):
            m.obstacle_from_dict(bad)
