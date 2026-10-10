"""CAM-02: 4종 무브의 이상(해석) 경로 ideal_path — 푸시인·슬라이더·오빗·랙의 정의식."""
import math
import random

import numpy as np
import pytest

from cam02_helpers import (DEFAULT_FRAMES, T4, angle_deg, is_rot, local_poses, mm, odd_home, orbit_local, s_mj, s_rt,
                           subject_in_camera, taus)

Z = 3.0


def _check_pose_stack(c2w, n=DEFAULT_FRAMES):
    assert c2w.shape == (n, 4, 4) and c2w.dtype == np.float64 and np.isfinite(c2w).all()
    assert all(is_rot(c2w[k, :3, :3], 1e-9) for k in range(0, n, 8))
    assert np.abs(c2w[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max() == 0.0


def test_movepath_container_and_c0_handling():
    m = mm()
    sp = m.recipe("orbit", Z)
    p = m.ideal_path(sp)
    assert isinstance(p, m.MovePath) and p.spec == sp and p.rig == "ideal" and p.n_frames == DEFAULT_FRAMES
    np.testing.assert_array_equal(p.c0, np.eye(4))
    np.testing.assert_allclose(p.t, np.arange(DEFAULT_FRAMES) / 24.0, atol=1e-12)
    assert p.focus_m.shape == (DEFAULT_FRAMES,) and np.all(p.focus_m > 0)
    np.testing.assert_allclose(p.subject, [0.0, 0.0, Z], atol=1e-15)
    _check_pose_stack(p.c2w)
    c0 = odd_home()
    c0_before = c0.copy()
    q = m.ideal_path(sp, c0)
    np.testing.assert_array_equal(c0, c0_before)                                    # 입력 불변
    c0[0, 0] = 99.0                                                                 # 내부는 복사본
    np.testing.assert_allclose(q.c0, c0_before, atol=0)
    np.testing.assert_allclose(q.subject, (c0_before @ np.array([0.0, 0.0, Z, 1.0]))[:3], atol=1e-12)
    # 경로는 C_0 에 대해 등변: ideal(c0) = c0 @ ideal(I)
    np.testing.assert_allclose(q.c2w, c0_before[None] @ p.c2w, atol=1e-12)
    np.testing.assert_allclose(q.focus_m, p.focus_m, atol=1e-12)


@pytest.mark.parametrize("c0", [np.zeros((4, 3)), np.full((4, 4), np.nan), np.diag([2.0, 1.0, 1.0, 1.0]),
                                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2.0]])])
def test_invalid_c0_rejected(c0):
    m = mm()
    with pytest.raises(ValueError):
        m.ideal_path(m.recipe("orbit", Z), c0)


@pytest.mark.parametrize("home", [None, "odd"])
@pytest.mark.parametrize("mag", [0.5, 1.0, 1.5, 2.0, 4.0])
def test_push_in_geometry(mag, home):
    m = mm()
    c0 = np.eye(4) if home is None else odd_home()
    p = m.ideal_path(m.recipe("push_in", Z, m=mag), c0)
    _check_pose_stack(p.c2w)
    d_end = Z * (1.0 - 1.0 / mag)
    d = d_end * s_mj(taus(DEFAULT_FRAMES))
    expect = c0[None] @ np.stack([T4(None, [0, 0, x]) for x in d])
    np.testing.assert_allclose(p.c2w, expect, atol=1e-12)                           # 광축 방향 직선 전진, 자세 고정
    np.testing.assert_allclose(p.c2w[:, :3, :3], np.broadcast_to(c0[:3, :3], (DEFAULT_FRAMES, 3, 3)), atol=1e-15)
    np.testing.assert_allclose(p.focus_m, Z - d, atol=1e-12)                        # 초점은 피사체를 따라간다
    # 배율: 피사체 깊이의 점을 투영한 크기의 비 = z_s / (z_s - d)
    X = (c0 @ np.array([0.3, -0.2, Z, 1.0]))[:3]
    xc = subject_in_camera(p.c2w, X)
    ratio = (xc[:, 0] / xc[:, 2]) / (xc[0, 0] / xc[0, 2])
    np.testing.assert_allclose(ratio, Z / (Z - d), atol=1e-12)
    assert abs(ratio[-1] - mag) < 1e-12 and ratio[0] == 1.0


def test_push_in_ease_in_out_shape():
    m = mm()
    p = m.ideal_path(m.recipe("push_in", Z, m=1.5))
    d = p.c2w[:, 2, 3]
    assert abs(d[-1] - 1.0) < 1e-12 and abs(d[96] - 0.5) < 1e-12                    # 8 s 중간 = 절반
    assert np.all(np.diff(d) >= 0)
    step = np.diff(d)
    assert step[0] < 1e-5 and step[-1] < 1e-5                                       # 정지 출발·정지 도착
    assert 92 <= int(np.argmax(step)) <= 99                                         # 최고 속도는 중간
    assert abs(step.max() * 24.0 - 1.875 * 1.0 / 8.0) < 2e-3                        # min-jerk 최고 속도 15/8 * d/T


@pytest.mark.parametrize("sgn", [1.0, -1.0, 0.0])
def test_slider_geometry(sgn):
    m = mm()
    c0 = odd_home()
    sl = 0.10 * sgn
    p = m.ideal_path(m.recipe("slider", Z, slide_zs=sl), c0)
    _check_pose_stack(p.c2w)
    L = local_poses(p.c2w, c0)
    x = sl * Z * s_mj(taus(DEFAULT_FRAMES))
    np.testing.assert_allclose(L[:, 0, 3], x, atol=1e-12)                           # C_0 의 +x 로 평행이동
    np.testing.assert_allclose(L[:, 1:3, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(L[:, :3, :3], np.broadcast_to(np.eye(3), (DEFAULT_FRAMES, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(p.focus_m, np.sqrt(Z ** 2 + x ** 2), atol=1e-12)
    assert abs(L[-1, 0, 3] - sl * Z) < 1e-12


@pytest.mark.parametrize("home", [None, "odd"])
@pytest.mark.parametrize("deg", [20.0, -20.0, 90.0, 180.0])
def test_orbit_geometry(deg, home):
    m = mm()
    c0 = np.eye(4) if home is None else odd_home()
    p = m.ideal_path(m.recipe("orbit", Z, orbit_deg=deg), c0)
    _check_pose_stack(p.c2w)
    a = math.radians(deg) * s_mj(taus(DEFAULT_FRAMES))
    np.testing.assert_allclose(p.c2w, c0[None] @ np.stack([orbit_local(x, Z) for x in a]), atol=1e-12)
    # 피사체는 항상 화면 중앙 광축 위, 거리 일정
    xc = subject_in_camera(p.c2w, p.subject)
    np.testing.assert_allclose(xc, np.broadcast_to([0.0, 0.0, Z], xc.shape), atol=1e-12)
    np.testing.assert_allclose(p.focus_m, Z, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(p.c2w[:, :3, 3] - p.subject, axis=1), Z, atol=1e-12)
    # 오빗 각 = 피사체에서 본 카메라 방향의 변화 = deg * s(tau)
    v = p.c2w[:, :3, 3] - p.subject
    np.testing.assert_allclose(angle_deg(v, v[0][None]), abs(deg) * s_mj(taus(DEFAULT_FRAMES)), atol=1e-9)
    # 부호 규약: +오빗 = 카메라는 C_0 의 -x(왼쪽)로 이동하며 +x(오른쪽)를 향해 회전
    if abs(deg) < 180.0:                                                            # 180 deg 에서는 x 성분이 0
        L = local_poses(p.c2w, c0)
        sgn = math.copysign(1.0, deg)
        assert L[-1, 0, 3] * sgn < 0 and L[-1, 0, 2] * sgn > 0


def test_orbit_mean_alpha_and_full_circle():
    m = mm()
    p = m.ideal_path(m.recipe("orbit", Z, orbit_deg=20.0))
    v = p.c2w[:, :3, 3] - p.subject
    alpha = angle_deg(v, v[0][None])
    assert abs(alpha.mean() - 10.0) < 1e-9 and abs(alpha.max() - 20.0) < 1e-9       # IR 표의 평균 |alpha| 10 deg
    p360 = m.ideal_path(m.recipe("orbit", Z, orbit_deg=360.0))
    np.testing.assert_allclose(p360.c2w[-1], np.eye(4), atol=1e-9)                  # 한 바퀴 = 제자리
    v360 = p360.c2w[:, :3, 3] - p360.subject
    assert abs(angle_deg(v360[96], v360[0]) - 180.0) < 1e-9                         # 중간(s=0.5)에 정반대편


@pytest.mark.parametrize("name,kw", [("push_in", dict(m=1.5)), ("slider", dict(slide_zs=0.1)),
                                     ("orbit", dict(orbit_deg=20.0)), ("orbit", dict(orbit_deg=-90.0)),
                                     ("rack_focus", dict(focus_to_m=9.0))])
def test_round_trip_is_symmetric_and_returns_exactly(name, kw):
    m = mm()
    c0 = odd_home()
    one = m.ideal_path(m.recipe(name, Z, **kw), c0)
    rt = m.ideal_path(m.recipe(name, Z, round_trip=True, **kw), c0)
    n = DEFAULT_FRAMES
    np.testing.assert_allclose(rt.c2w[-1], rt.c2w[0], atol=1e-12)                   # C_0 로 정확히 복귀
    np.testing.assert_allclose(rt.c2w[0], c0, atol=1e-12)
    np.testing.assert_allclose(rt.c2w, rt.c2w[::-1], atol=1e-12)                    # 시간 대칭 경로
    np.testing.assert_allclose(rt.focus_m, rt.focus_m[::-1], atol=1e-12)
    # 가운데(4 s)는 편도의 종점과 같다
    np.testing.assert_allclose(rt.c2w[96], one.c2w[-1], atol=1e-12)
    np.testing.assert_allclose(rt.focus_m[96], one.focus_m[-1], atol=1e-12)
    # 앞 절반은 편도를 2배 속도로 본 것
    s_one = s_mj(np.arange(97) / 96)
    if name == "orbit":
        a = math.radians(kw["orbit_deg"]) * s_one
        np.testing.assert_allclose(rt.c2w[:97], c0[None] @ np.stack([orbit_local(x, Z) for x in a]), atol=1e-12)
    assert n == rt.n_frames


def test_rack_focus_pulls_in_diopter_space_with_a_locked_camera():
    m = mm()
    c0 = odd_home()
    sp = m.recipe("rack_focus", Z, focus_to_m=12.0)
    p = m.ideal_path(sp, c0)
    np.testing.assert_allclose(p.c2w, np.broadcast_to(c0, p.c2w.shape), atol=0)     # 카메라는 정지
    s = s_mj(taus(DEFAULT_FRAMES))
    d0, d1 = 1.0 / Z, 1.0 / 12.0
    np.testing.assert_allclose(1.0 / p.focus_m, d0 + s * (d1 - d0), atol=1e-12)     # 디옵터에서 min-jerk
    assert abs(p.focus_m[0] - Z) < 1e-12 and abs(p.focus_m[-1] - 12.0) < 1e-12
    assert np.all(np.diff(p.focus_m) >= 0)
    # 가까운 쪽으로도, 임의의 시작 거리로도
    q = m.ideal_path(m.recipe("rack_focus", Z, focus_from_m=8.0, focus_to_m=1.5))
    assert abs(q.focus_m[0] - 8.0) < 1e-12 and abs(q.focus_m[-1] - 1.5) < 1e-12 and np.all(np.diff(q.focus_m) <= 0)
    # 편향 없음: 가운데 시각의 디옵터는 양 끝의 평균
    assert abs(1.0 / q.focus_m[96] - 0.5 * (1.0 / 8.0 + 1.0 / 1.5)) < 1e-12
    # 푸시인 등 다른 무브에서는 초점이 피사체 거리를 따라가고 항상 양수
    for name in ("push_in", "slider", "orbit"):
        assert np.all(m.ideal_path(m.recipe(name, Z)).focus_m > 0)


def test_other_moves_follow_focus_on_the_subject():
    m = mm()
    for name, kw in (("push_in", dict(m=4.0)), ("slider", dict(slide_zs=-0.12)), ("orbit", dict(orbit_deg=90.0))):
        p = m.ideal_path(m.recipe(name, Z, **kw), odd_home())
        np.testing.assert_allclose(p.focus_m, np.linalg.norm(p.c2w[:, :3, 3] - p.subject, axis=1), atol=1e-12)


def test_ideal_path_is_deterministic_and_does_not_touch_global_rng():
    m = mm()
    np.random.seed(1234)
    random.seed(4321)
    st_np, st_py = np.random.get_state()[1].copy(), random.getstate()
    a = m.ideal_path(m.recipe("orbit", Z, orbit_deg=33.0), odd_home())
    b = m.ideal_path(m.recipe("orbit", Z, orbit_deg=33.0), odd_home())
    assert a.c2w.tobytes() == b.c2w.tobytes() and a.focus_m.tobytes() == b.focus_m.tobytes()
    assert (np.random.get_state()[1] == st_np).all() and random.getstate() == st_py


def test_other_frame_rates_hit_the_endpoint_exactly():
    m = mm()
    for dur, fps in ((2.5, 30.0), (1.0, 24.0), (8.0, 60.0)):
        p = m.ideal_path(m.recipe("orbit", Z, orbit_deg=20.0, duration_s=dur, fps=fps))
        n = int(round(dur * fps)) + 1
        assert p.c2w.shape == (n, 4, 4) and abs(p.t[-1] - dur) < 1e-12
        np.testing.assert_allclose(p.c2w[-1], orbit_local(math.radians(20.0), Z), atol=1e-12)
        r = m.ideal_path(m.recipe("orbit", Z, orbit_deg=20.0, duration_s=dur, fps=fps, round_trip=True))
        np.testing.assert_allclose(r.c2w, r.c2w[::-1], atol=1e-12)
    assert s_rt(0.5) == 1.0
