"""CAM-04: face_angles 와 미관측 얼굴면 노출 — 해석해, 독립 구적과의 대조, 대칭/불변."""
import math

import numpy as np
import pytest

from cam04_helpers import (cam_pos, frame, fe, make_face, random_rigid, ref_exposure, transform_face)

TOL = 0.006      # 계약서: 노출 비율의 절대 오차 <= 0.005 (+ 부동소수 여유)


# ---------------------------------------------------------------- face_angles
def test_face_angles_analytic():
    m = fe()
    face = make_face(front=(0, 0, -1), d=2.0)           # 카메라(원점)를 보는 얼굴, right = up x front = (+1,0,0)
    r, u, f = frame(face)
    assert np.allclose(r, [1, 0, 0]) and np.allclose(u, [0, -1, 0]) and np.allclose(f, [0, 0, -1])
    yaw, pitch, dist = m.face_angles(face, [1.0, 0.0, 0.0])
    assert yaw == pytest.approx(math.degrees(math.atan2(1.0, 2.0)), abs=1e-9) and pitch == pytest.approx(0.0, abs=1e-9)
    assert dist == pytest.approx(math.sqrt(5.0))
    yaw, pitch, dist = m.face_angles(face, [0.0, -1.0, 0.0])      # 위(OpenCV y 음수)에서 -> pitch > 0
    assert yaw == pytest.approx(0.0, abs=1e-9) and pitch == pytest.approx(math.degrees(math.atan2(1.0, 2.0)), abs=1e-9)
    yaw, pitch, _ = m.face_angles(face, [-2.0, 0.0, 2.0])         # 얼굴 옆 90 도 부근
    assert yaw == pytest.approx(-90.0, abs=1e-9)
    yaw, _, _ = m.face_angles(face, [0.0, 0.0, 4.0])              # 얼굴 뒤
    assert abs(yaw) == pytest.approx(180.0, abs=1e-9)


def test_face_angles_vectorized_matches_scalar_and_roundtrips_helper():
    m = fe()
    face = make_face(front=(0.3, 0.1, -1.0), d=2.5)
    ys = [-170.0, -80.0, -30.0, 0.0, 20.0, 75.0, 140.0]
    pts = np.stack([cam_pos(face, y, 12.0, 1.7) for y in ys])
    yaw, pitch, dist = m.face_angles(face, pts)
    assert yaw.shape == pitch.shape == dist.shape == (len(ys),)
    assert np.allclose(yaw, ys, atol=1e-9) and np.allclose(pitch, 12.0, atol=1e-9) and np.allclose(dist, 1.7)
    for i in range(len(ys)):
        s = m.face_angles(face, pts[i])
        assert np.allclose(s, (yaw[i], pitch[i], dist[i]), atol=1e-12)


def test_face_angles_rejects_camera_inside_head_or_nonfinite():
    m = fe()
    face = make_face(d=2.0)
    with pytest.raises(ValueError):
        m.face_angles(face, [0.0, 0.0, 2.0])               # 얼굴 중심
    with pytest.raises(ValueError):
        m.face_angles(face, [0.0, 0.0, 2.05])              # 머리 구 안 (< 0.10 m)
    with pytest.raises(ValueError):
        m.face_angles(face, [float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError):
        m.face_angles(face, [0.0, 0.0])


# ---------------------------------------------------------------- 노출: 해석해
def test_exposure_zero_for_same_camera():
    m = fe()
    face = make_face()
    p = cam_pos(face, 160.0, 5.0, 2.0)
    assert m.unobserved_exposure(face, p, p) == 0.0


def test_exposure_zero_when_face_is_observed_frontally_from_any_camera():
    # 정면 관측(캡 전체가 보임) -> 어디로 가도 새로 드러나는 얼굴면이 없다.
    m = fe()
    face = make_face(front=(0, 0, -1), d=2.0)
    p0 = np.zeros(3)
    rng = np.random.default_rng(3)
    for _ in range(40):
        yaw, pitch, dist = rng.uniform(-180, 180), rng.uniform(-60, 60), rng.uniform(0.5, 6.0)
        assert m.unobserved_exposure(face, p0, cam_pos(face, yaw, pitch, dist)) == 0.0


def test_exposure_far_camera_analytic_values():
    # 거의 무한 거리 -> 가시 영역 = 반구. 뒷모습(관측 yaw 180)에서 a 도 돈 카메라:
    #  a <= 45 (= 90 - 캡 반각) 이면 0, a >= 135 이면 1, a = 90 이면 대원이 캡 중심을 지나 정확히 1/2.
    m = fe()
    face = make_face(d=500.0)
    p0 = np.zeros(3)                                    # 얼굴 뒤 500 m
    e = lambda a: m.unobserved_exposure(face, p0, cam_pos(face, 180.0 - a, 0.0, 500.0))
    assert e(0.0) == 0.0
    assert e(30.0) == 0.0 and e(40.0) == 0.0
    assert e(90.0) == pytest.approx(0.5, abs=TOL)
    assert e(140.0) == pytest.approx(1.0, abs=TOL) and e(180.0) == pytest.approx(1.0, abs=TOL)
    # 피치로 돌려도 같은 구면 기하: 위쪽 카메라, 관측 pitch 0 -> 위로 90 도
    p_up = cam_pos(face, 180.0, 90.0, 500.0)
    assert m.unobserved_exposure(face, p0, p_up) == pytest.approx(0.5, abs=TOL)


def test_exposure_in_unit_interval_and_monotone_in_orbit_angle():
    m = fe()
    face = make_face(d=2.0)
    p0 = np.zeros(3)
    vals = [m.unobserved_exposure(face, p0, cam_pos(face, 180.0 - a, 0.0, 2.0)) for a in range(0, 181, 5)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a - 2e-3 for a, b in zip(vals, vals[1:]))
    assert vals[0] == 0.0 and vals[-1] > 0.99


def test_exposure_mirror_symmetry_for_axis_observation():
    m = fe()
    face = make_face(d=2.0)
    p0 = np.zeros(3)
    for a in (50.0, 60.0, 90.0, 120.0):
        left = m.unobserved_exposure(face, p0, cam_pos(face, 180.0 + a, 0.0, 2.0))
        right = m.unobserved_exposure(face, p0, cam_pos(face, 180.0 - a, 0.0, 2.0))
        assert left == pytest.approx(right, abs=0.01)


def test_nearer_camera_sees_less_of_the_head():
    # 유한 거리: 가시 조건은 n.(c-p) > r. 가까울수록 보이는 영역이 작다 -> 노출이 줄어든다.
    m = fe()
    p0_of = lambda f, d: np.zeros(3)
    vals = []
    for d in (0.4, 1.0, 2.0, 8.0, 100.0):
        face = make_face(d=d)
        vals.append(m.unobserved_exposure(face, p0_of(face, d), cam_pos(face, 90.0, 0.0, d)))
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[0] < vals[2] - 0.03 and vals[2] < 0.5 - 0.01 and vals[-1] == pytest.approx(0.5, abs=0.01)


def test_larger_head_radius_hides_more():
    m = fe()
    small = make_face(d=2.0, head_radius_m=0.10)
    big = make_face(d=2.0, head_radius_m=0.40)
    p0 = np.zeros(3)
    q = lambda f: cam_pos(f, 90.0, 0.0, 2.0)
    assert m.unobserved_exposure(big, p0, q(big)) < m.unobserved_exposure(small, p0, q(small)) - 0.02


# ---------------------------------------------------------------- 노출: 독립 구적과 대조
CASES = [
    # (front, d0, obs yaw/pitch/dist, cam yaw/pitch/dist)
    ((0, 0, 1), 2.0, (180.0, 0.0, 2.0), (120.0, 0.0, 2.0)),
    ((0, 0, 1), 2.0, (180.0, 0.0, 2.0), (60.0, 15.0, 1.2)),
    ((0, 0, 1), 3.0, (150.0, 10.0, 3.0), (95.0, -10.0, 3.0)),
    ((0, 0, 1), 3.0, (150.0, 10.0, 3.0), (-120.0, 25.0, 5.0)),
    ((0.3, 0.2, 1.0), 2.5, (110.0, -5.0, 2.0), (30.0, 0.0, 2.0)),
    ((0.3, 0.2, 1.0), 2.5, (110.0, -5.0, 2.0), (-40.0, 0.0, 2.0)),
    ((1.0, 0.0, 0.2), 1.5, (0.0, 0.0, 1.5), (70.0, 0.0, 1.5)),
    ((0, 0, -1), 2.0, (60.0, 0.0, 2.0), (-60.0, 0.0, 2.0)),
]


@pytest.mark.parametrize("front,d0,obs,cam", CASES)
def test_exposure_matches_independent_quadrature(front, d0, obs, cam):
    m = fe()
    face = make_face(front=front, d=d0)
    p0, p = cam_pos(face, *obs), cam_pos(face, *cam)
    got = m.unobserved_exposure(face, p0, p)
    assert got == pytest.approx(ref_exposure(face, p0, p), abs=TOL)


def test_exposure_vector_input_matches_scalar_and_is_deterministic():
    m = fe()
    face = make_face(d=2.0)
    p0 = np.zeros(3)
    pts = np.stack([cam_pos(face, y, 5.0, 2.0) for y in (180.0, 150.0, 120.0, 100.0, 60.0)])
    vec = m.unobserved_exposure(face, p0, pts)
    assert vec.shape == (5,)
    for i in range(5):
        assert vec[i] == m.unobserved_exposure(face, p0, pts[i])
    assert np.array_equal(vec, m.unobserved_exposure(face, p0, pts))


def test_exposure_is_invariant_under_rigid_motion():
    m = fe()
    face = make_face(front=(0.2, 0.0, 1.0), d=2.0)
    p0 = cam_pos(face, 165.0, 4.0, 2.0)
    pts = np.stack([cam_pos(face, y, 8.0, 1.8) for y in (150.0, 110.0, 80.0, -100.0)])
    base = m.unobserved_exposure(face, p0, pts)
    for seed in (1, 2):
        R, t = random_rigid(seed)
        f2 = transform_face(face, R, t)
        got = m.unobserved_exposure(f2, R @ p0 + t, (R @ pts.T).T + t)
        assert np.allclose(got, base, atol=1e-3)


def test_exposure_rejects_camera_inside_head_or_bad_shape():
    m = fe()
    face = make_face(d=2.0)
    ok = cam_pos(face, 90.0, 0.0, 2.0)
    with pytest.raises(ValueError):
        m.unobserved_exposure(face, ok, [0.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        m.unobserved_exposure(face, [0.0, 0.0, 2.0], ok)
    with pytest.raises(ValueError):
        m.unobserved_exposure(face, ok, np.zeros((4, 2)))
    with pytest.raises(ValueError):
        m.unobserved_exposure(face, ok, [np.nan, 0.0, 0.0])
