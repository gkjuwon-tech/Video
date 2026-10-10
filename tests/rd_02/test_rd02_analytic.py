"""해석 기본 함수: focal_from_vps, 수심 주점, 수평선, roll/pitch 왕복, 선분-VP 잔차."""
import math

import numpy as np
import pytest

import rd02_helpers as H
from buc.read import geom_cam as gc

W, HH = 1280, 720


def _vps(fov, yaw, pitch, roll, pp=None):
    K = H.K_from_fov(fov, W, HH, pp)
    R = H.camera_R(yaw, pitch, roll)
    return K, R, [K @ (R @ H.AXES[i]) for i in range(3)]


@pytest.mark.parametrize("fov,yaw,pitch,roll", [(60, 30, 6, -3), (45, -40, 15, 5), (80, 20, -10, 0)])
def test_focal_from_two_orthogonal_vps_exact(fov, yaw, pitch, roll):
    K, R, v = _vps(fov, yaw, pitch, roll)
    c = K[:2, 2]
    # 수평 두 VP(축 0, 2): 유한 픽셀 좌표 (2,) 와 동차 (3,) 둘 다 받는다
    p0, p2 = v[0][:2] / v[0][2], v[2][:2] / v[2][2]
    f1 = gc.focal_from_vps(p0, p2, c)
    f2 = gc.focal_from_vps(v[0], v[2] * -3.0, c)
    assert f1 == pytest.approx(K[0, 0], rel=1e-9)
    assert f2 == pytest.approx(K[0, 0], rel=1e-9)


def test_focal_from_vps_rejects_impossible_configs():
    c = np.array([639.5, 359.5])
    # 같은 쪽에 있는 두 VP: (v1-c)·(v2-c) > 0 -> 허수 초점
    assert gc.focal_from_vps(np.array([2000.0, 360.0]), np.array([1500.0, 300.0]), c) is None
    # 무한원점
    assert gc.focal_from_vps(np.array([1.0, 0.0, 0.0]), np.array([-200.0, 360.0, 1.0]), c) is None
    assert gc.focal_from_vps(np.array([np.nan, 1.0]), np.array([-200.0, 360.0]), c) is None


def test_principal_point_is_orthocenter_of_three_vps():
    pp = ((W - 1) / 2 + 40, (HH - 1) / 2 - 25)
    K, R, v = _vps(60, 35, 22, 3, pp)
    pts = [x[:2] / x[2] for x in v]
    c = gc.principal_point_from_vps(*pts)
    np.testing.assert_allclose(c, pp, atol=1e-6)
    c2 = gc.principal_point_from_vps(*v)  # 동차 입력
    np.testing.assert_allclose(c2, pp, atol=1e-6)


@pytest.mark.parametrize("roll,pitch", [(0, 0), (-4, 8), (7.5, -15), (-30, 45), (12, 0)])
def test_roll_pitch_roundtrip_and_definition(roll, pitch):
    up = gc.up_from_roll_pitch(roll, pitch)
    assert up.shape == (3,)
    assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(up, [math.cos(math.radians(pitch)) * math.sin(math.radians(roll)),
                                    -math.cos(math.radians(pitch)) * math.cos(math.radians(roll)),
                                    math.sin(math.radians(pitch))], atol=1e-12)
    r, p = gc.roll_pitch_from_up(up)
    assert r == pytest.approx(roll, abs=1e-9) and p == pytest.approx(pitch, abs=1e-9)
    # 오라클 카메라와 같은 정의
    R = H.camera_R(17.0, pitch, roll)
    r2, p2 = gc.roll_pitch_from_up(H.up_cam(R) * 2.5)  # 노름 무관
    assert r2 == pytest.approx(roll, abs=1e-9) and p2 == pytest.approx(pitch, abs=1e-9)


def test_horizon_line_contains_horizontal_vanishing_points_and_sky_is_positive():
    K, R, v = _vps(60, 30, 8, -4)
    up = H.up_cam(R)
    l = gc.horizon_line(K, up)
    assert l.shape == (3,)
    assert np.linalg.norm(l[:2]) == pytest.approx(1.0, abs=1e-12)
    for ax in (0, 2):  # 수평 축의 VP 는 수평선 위에 있다
        assert abs(v[ax][2]) > 1e-9
        assert abs(l @ (v[ax] / v[ax][2])) < 1e-6  # 픽셀 거리(‖l[:2]‖=1)
    # 하늘 쪽(위쪽 광선)은 양수, 땅 쪽은 음수
    sky = K @ (up + 0.2 * (R @ H.AXES[2]))
    ground = K @ (-up + 0.2 * (R @ H.AXES[2]))
    assert l @ (sky / sky[2]) > 0 and l @ (ground / ground[2]) < 0
    # 수평선 영상 기울기 = roll
    y0 = -(l[2]) / l[1]
    y1 = -(l[0] * 100 + l[2]) / l[1]
    assert math.degrees(math.atan2(y1 - y0, 100)) == pytest.approx(-4.0, abs=1e-9)


def test_segment_vp_residual():
    vp = np.array([2000.0, 300.0, 1.0])
    a = np.array([100.0, 500.0])
    d = (vp[:2] - a) / np.linalg.norm(vp[:2] - a)
    segs = np.array([np.r_[a, a + 200 * d],                    # VP 를 정확히 향함
                     np.r_[a + 200 * d, a],                    # 방향 반대여도 0
                     ])
    n = np.array([-d[1], d[0]])
    rot = math.radians(3.0)
    d3 = d * math.cos(rot) + n * math.sin(rot)
    mid = a + 100 * d
    segs = np.vstack([segs, np.r_[mid - 100 * d3, mid + 100 * d3]])  # 중점 둘레로 3° 돌림
    r = gc.segment_vp_residual_deg(segs, vp)
    assert r.shape == (3,)
    np.testing.assert_allclose(r[:2], 0.0, atol=1e-7)
    assert r[2] == pytest.approx(3.0, abs=1e-6)
    # 무한원점 VP: 방향 그 자체와 비교
    r_inf = gc.segment_vp_residual_deg(segs[:2], np.r_[d, 0.0])
    np.testing.assert_allclose(r_inf, 0.0, atol=1e-7)
    assert (gc.segment_vp_residual_deg(segs, vp) <= 90.0).all()
