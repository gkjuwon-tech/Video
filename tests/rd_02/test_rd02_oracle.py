"""도우미 자체 검사: 합성 오라클이 맞아야 나머지 테스트가 의미 있다. RD-02 구현을 import 하지 않는다."""
import math

import numpy as np

import rd02_helpers as H
from buc.core import geom


def test_camera_R_is_rotation_and_roll_pitch_match_construction():
    for yaw, pitch, roll in [(0, 0, 0), (30, 8, -4), (-50, -15, 7), (35, 22, 3)]:
        R = H.camera_R(yaw, pitch, roll)
        assert geom.is_rotation(R)
        r, p = H.roll_pitch(H.up_cam(R))
        assert abs(r - roll) < 1e-9 and abs(p - pitch) < 1e-9


def test_pitch_positive_puts_horizon_below_center_and_roll_positive_lowers_right_side():
    W, Hh = 1280, 720
    K = H.K_from_fov(60, W, Hh)
    up = H.up_cam(H.camera_R(0, 10, 0))
    l = np.linalg.inv(K).T @ up
    y_c = -(l[0] * K[0, 2] + l[2]) / l[1]
    assert y_c > K[1, 2] + 100  # 위를 보면 수평선은 중심 아래
    up = H.up_cam(H.camera_R(0, 0, 5))
    l = np.linalg.inv(K).T @ up
    yl = -(l[0] * 0 + l[2]) / l[1]
    yr = -(l[0] * (W - 1) + l[2]) / l[1]
    assert yr > yl  # roll + 이면 오른쪽이 내려간다(y 아래)
    assert abs(math.degrees(math.atan2(yr - yl, W - 1)) - 5.0) < 1e-9


def test_segments_point_at_ground_truth_vps():
    W, Hh = 1280, 720
    K = H.K_from_fov(60, W, Hh)
    R = H.camera_R(30, 6, -3)
    segs, labs = H.manhattan_segments(K, R, W, Hh, noise_px=0.0)
    assert len(segs) == 180
    for ax in range(3):
        v = K @ (R @ H.AXES[ax])
        for s in segs[labs == ax][:10]:
            a, b = s[:2], s[2:]
            # 동차 직선 위에 VP 가 있다
            l = np.cross(np.r_[a, 1.0], np.r_[b, 1.0])
            assert abs(l @ v) / (np.linalg.norm(l[:2]) * np.linalg.norm(v)) < 1e-9
        assert (np.hypot(segs[:, 0] - segs[:, 2], segs[:, 1] - segs[:, 3]) >= 40).all()


def test_distorted_polylines_are_straight_after_undistort():
    W, Hh = 1920, 1080
    K = H.K_from_fov(70, W, Hh)
    R = H.camera_R(25, 5, 2)
    polys = H.distorted_polylines(K, R, W, Hh, -0.1, 0.02, noise_px=0.0)
    assert len(polys) == 24
    curv = []
    for P in polys:
        xy = (P - K[:2, 2]) / K[0, 0]
        und = geom.undistort(xy, -0.1, 0.02) * K[0, 0]
        c = und - und.mean(0)
        assert np.linalg.svd(c, compute_uv=False)[1] / math.sqrt(len(c)) < 1e-6
        raw = P - P.mean(0)
        curv.append(np.linalg.svd(raw, compute_uv=False)[1] / math.sqrt(len(raw)))
    assert max(curv) > 2.0  # 왜곡이 실제로 곡률을 만든다


def test_max_displacement_metric():
    K = H.K_from_fov(70, 1920, 1080)
    assert H.max_displacement_px(K, 1920, 1080, -0.1, 0.02, -0.1, 0.02) == 0.0
    d = H.max_displacement_px(K, 1920, 1080, 0.0, 0.0, -0.1, 0.02)
    assert 10 < d < 200


def test_render_room_has_contrast_and_is_deterministic():
    K = H.K_from_fov(60, 160, 120)
    R = H.camera_R(30, 8, -4)
    a = H.render_room(K, R, 160, 120)
    b = H.render_room(K, R, 160, 120)
    assert np.array_equal(a, b)
    assert a.min() < 0.3 and a.max() > 0.7 and np.isfinite(a).all()
