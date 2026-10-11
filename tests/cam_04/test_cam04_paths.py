"""CAM-04: 경로 판정 evaluate_path / evaluate_faces / orbit_budget_deg — F0-F3 봉투를 실제로 집행하는가."""
import dataclasses
import math

import numpy as np
import pytest

from cam04_helpers import (cam_pos, fe, make_face, orbit_path, push_path, random_rigid, ref_exposure,
                           solve_angle_for_exposure, transform_face, transform_poses)

P0 = np.zeros(3)             # C_0 = 단위행렬 -> 위치 원점


def front_face(**kw):
    """카메라를 정면으로 보는 얼굴 (관측 yaw 0)."""
    kw.setdefault("d", 2.0)
    return make_face(front=(0, 0, -1), **kw)


def back_face(**kw):
    """카메라를 등진 뒷모습 (관측 yaw 180)."""
    kw.setdefault("d", 2.0)
    return make_face(front=(0, 0, 1), **kw)


def ev(face, c2w, **kw):
    return fe().evaluate_path(face, c2w, **kw)


# ---------------------------------------------------------------- 정면 F2: yaw 봉투
def test_f2_frontal_yaw_30_degree_envelope():
    face = front_face(h_f=0.10, sigma_z=0.005)               # alpha_sigma(0.5 cm)=64 도 -> 표(30)가 이긴다
    ok = ev(face, orbit_path(face, P0, 25.0))
    assert ok.ok and ok.violations == () and ok.category == "F2"
    assert ok.yaw_dev_deg == pytest.approx(25.0, abs=1e-6) and ok.pitch_dev_deg == pytest.approx(0.0, abs=1e-6)
    assert ok.yaw_obs_deg == pytest.approx(0.0, abs=1e-6) and ok.m_peak == pytest.approx(1.0, abs=1e-9)
    assert ok.h_f_max == pytest.approx(0.10, abs=1e-9) and ok.limits.yaw_deg == pytest.approx(30.0)
    edge = ev(face, orbit_path(face, P0, 30.0))
    assert edge.ok                                            # 정확히 30 도는 허용
    bad = ev(face, orbit_path(face, P0, 31.0))
    assert not bad.ok and bad.violations == ("yaw",)
    neg = ev(face, orbit_path(face, P0, -31.0))
    assert not neg.ok and neg.violations == ("yaw",)          # 반대 방향도 같다
    assert ev(face, orbit_path(face, P0, -29.0)).ok


def test_alpha_sigma_tightens_the_yaw_envelope():
    face = front_face(h_f=0.10, sigma_z=0.02)                 # alpha_sigma = 13 도
    assert ev(face, orbit_path(face, P0, 12.0)).ok
    bad = ev(face, orbit_path(face, P0, 14.0))
    assert not bad.ok and bad.violations == ("yaw",) and bad.limits.yaw_deg == pytest.approx(13.0, abs=0.02)
    wide = front_face(h_f=0.10, sigma_z=0.0)
    assert ev(wide, orbit_path(wide, P0, 29.0)).ok            # 불확실도가 없으면 표값까지 열린다


def test_f3_close_face_yaw_15_and_safe_grade_only():
    face = front_face(h_f=0.20, sigma_z=0.003)
    v = ev(face, orbit_path(face, P0, 14.0))
    assert v.ok and v.category == "F3" and v.allowed_grades == ("safe",)
    bad = ev(face, orbit_path(face, P0, 16.0))
    assert not bad.ok and bad.violations == ("yaw",)
    assert ev(face, orbit_path(face, P0, 14.0), grade="safe").ok
    g = ev(face, orbit_path(face, P0, 14.0), grade="extended")
    assert not g.ok and g.violations == ("grade",)


def test_pitch_envelope_f2_15_and_f3_10():
    f2 = front_face(h_f=0.10)
    assert ev(f2, orbit_path(f2, P0, 0.0, 14.0)).ok
    bad = ev(f2, orbit_path(f2, P0, 0.0, 16.0))
    assert not bad.ok and bad.violations == ("pitch",) and bad.pitch_dev_deg == pytest.approx(16.0, abs=1e-6)
    assert not ev(f2, orbit_path(f2, P0, 0.0, -16.0)).ok        # 아래쪽도 대칭
    f3 = front_face(h_f=0.20)
    assert ev(f3, orbit_path(f3, P0, 0.0, 9.0)).ok
    bad3 = ev(f3, orbit_path(f3, P0, 0.0, 11.0))
    assert not bad3.ok and bad3.violations == ("pitch",)


def test_combined_violations_are_listed_in_fixed_order():
    face = front_face(h_f=0.20)
    v = ev(face, orbit_path(face, P0, 40.0, 20.0), grade="extended")
    assert v.violations == ("yaw", "pitch", "grade") and not v.ok


def test_margin_deg_shrinks_yaw_envelope():
    face = front_face(h_f=0.10)
    assert ev(face, orbit_path(face, P0, 25.0)).ok
    v = ev(face, orbit_path(face, P0, 25.0), margin_deg=12.0)     # 한계 30 - 12 = 18
    assert not v.ok and v.violations == ("yaw",) and v.limits.yaw_deg == pytest.approx(18.0)
    assert ev(face, orbit_path(face, P0, 17.0), margin_deg=12.0).ok


# ---------------------------------------------------------------- 푸시인: 범주는 경로 전체의 h_f 로 정한다
def test_push_in_limits_by_category():
    f2 = front_face(h_f=0.10)
    assert ev(f2, push_path(f2, P0, 1.0)).ok                       # m = 1.0
    assert ev(f2, push_path(f2, P0, 1.01)).ok                      # 2% 허용 오차 안
    bad = ev(f2, push_path(f2, P0, 1.05))
    assert not bad.ok and bad.violations == ("push",) and bad.m_peak == pytest.approx(1.05, abs=1e-6)
    assert ev(f2, push_path(f2, P0, 0.7)).ok                       # 후퇴는 자유
    f1 = front_face(h_f=0.015)
    assert ev(f1, push_path(f1, P0, 1.5)).ok and ev(f1, push_path(f1, P0, 1.5)).category == "F1"
    assert ev(f1, push_path(f1, P0, 1.51)).ok                      # m <= 1.5 (+2%)
    assert ev(f1, push_path(f1, P0, 1.7)).violations == ("push",)


def test_category_uses_h_f_max_over_the_path():
    face = front_face(h_f=0.015)                                  # 시작은 F1 크기
    v = ev(face, push_path(face, P0, 2.5))                        # 끝에서 h_f = 0.0375 > 3% -> F2
    assert v.category == "F2" and v.h_f_max == pytest.approx(0.0375, abs=1e-6)
    assert v.violations == ("push",)
    near = front_face(h_f=0.14)
    w = ev(near, push_path(near, P0, 1.2))                         # 0.14 -> 0.168 : F3
    assert w.category == "F3" and w.h_f_max == pytest.approx(0.168, abs=1e-6)


# ---------------------------------------------------------------- F1: 제한 없음, 단 미관측 3%
def test_f1_small_face_has_no_angle_limit():
    face = front_face(h_f=0.015)
    for yaw, pitch in ((90.0, 0.0), (180.0, 0.0), (60.0, 40.0)):
        v = ev(face, orbit_path(face, P0, yaw, pitch))
        assert v.ok and v.category == "F1", (yaw, pitch, v)
        assert v.unobs_max == 0.0


# ---------------------------------------------------------------- F0: 얼굴 없음 / 뒷모습, 5% 규칙
def test_no_face_means_no_constraint():
    r = fe().evaluate_faces([], orbit_path(front_face(), P0, 120.0))
    assert r.ok and r.violations == () and r.verdicts == () and r.allowed_grades == ("safe", "extended", "full")
    assert fe().face_viol_sigma([], orbit_path(front_face(), P0, 120.0)) is False


def test_f0_back_view_five_percent_rule():
    face = back_face(h_f=0.10)
    ok = ev(face, orbit_path(face, P0, 45.0))                      # 미관측 0%
    assert ok.ok and ok.category == "F0" and abs(ok.yaw_obs_deg) == pytest.approx(180.0, abs=1e-6)
    assert ok.unobs_max == 0.0 and ok.limits.unobs_max == pytest.approx(0.05)
    for sgn in (+1.0, -1.0):                                       # yaw 편차는 +-180 도 경계를 감아서 잰다
        w = ev(face, orbit_path(face, P0, sgn * 45.0))
        assert w.yaw_dev_deg == pytest.approx(45.0, abs=1e-6)
    mid = ev(face, orbit_path(face, P0, 50.0))                     # 약 0.7% : 허용
    assert mid.ok and 0.0 < mid.unobs_max < 0.05
    bad = ev(face, orbit_path(face, P0, 65.0))
    assert not bad.ok and bad.violations == ("unobserved",)
    assert bad.unobs_max == pytest.approx(ref_exposure(face, P0, cam_pos(face, 180.0 - 65.0, 0.0, 2.0)), abs=0.006)
    assert bad.unobs_max > 0.05
    # 반대 방향도 같다. 뒷모습은 yaw/pitch/푸시 규칙이 없다(위반은 노출뿐).
    assert not ev(face, orbit_path(face, P0, -65.0)).ok
    assert ev(face, orbit_path(face, P0, 40.0, 30.0)).violations in ((), ("unobserved",))
    assert ev(face, push_path(face, P0, 3.0)).ok


def test_unobserved_cap_uses_per_frame_maximum_not_the_endpoint():
    # 60 도 돌았다가 돌아오는 경로: 끝점은 관측 위치로 복귀(노출 0)지만 도중 5% 를 넘었다.
    face = back_face(h_f=0.10)
    out = orbit_path(face, P0, 65.0, n=31)
    back = np.concatenate([out, out[::-1][1:]])
    v = ev(face, back)
    assert not v.ok and v.violations == ("unobserved",)
    assert v.unobs_max > 0.05


def test_f1_three_percent_vs_f0_five_percent_thresholds():
    # 같은 노출량(약 4%)이 F1(관측 yaw 115, 3% 한계)에서는 위반, F0(관측 yaw 140, 5% 한계)에서는 허용.
    f1 = front_face(h_f=0.02)
    a1 = solve_angle_for_exposure(f1, 115.0, 0.04, -1.0)
    p01 = cam_pos(f1, 115.0, 0.0, 2.0)
    path1 = orbit_path(f1, p01, -a1)
    v1 = ev(f1, path1)
    assert v1.category == "F1" and v1.limits.unobs_max == pytest.approx(0.03)
    assert v1.unobs_max == pytest.approx(0.04, abs=0.008)
    assert not v1.ok and v1.violations == ("unobserved",)

    f0 = front_face(h_f=0.02)
    a0 = solve_angle_for_exposure(f0, 140.0, 0.04, -1.0)
    p00 = cam_pos(f0, 140.0, 0.0, 2.0)
    v0 = ev(f0, orbit_path(f0, p00, -a0))
    assert v0.category == "F0" and v0.limits.unobs_max == pytest.approx(0.05)
    assert v0.unobs_max == pytest.approx(0.04, abs=0.008)
    assert v0.ok


def test_f2_also_obeys_unobserved_five_percent():
    # 옆얼굴(관측 yaw 100, F2) 에서 정면 쪽으로 7 도 돌면 5% 를 넘는다. yaw 한계(30)는 지켰더라도 폐기.
    face = front_face(h_f=0.10)
    p0 = cam_pos(face, 100.0, 0.0, 2.0)
    path = orbit_path(face, p0, -7.0)
    ref = ref_exposure(face, p0, cam_pos(face, 93.0, 0.0, 2.0))
    v = ev(face, path)
    assert ref > 0.05 + 0.006                                       # 시험 자체의 전제
    assert v.category == "F2" and v.yaw_dev_deg == pytest.approx(7.0, abs=1e-6)
    assert not v.ok and v.violations == ("unobserved",)


def test_back_view_exposure_budget_is_about_sixty_degrees_and_depends_on_distance():
    m = fe()
    budgets = []
    for d in (1.5, 2.0, 4.0, 8.0):
        face = back_face(h_f=0.05, d=d)
        b = m.orbit_budget_deg(face, P0, +1.0)
        assert b == pytest.approx(m.orbit_budget_deg(face, P0, -1.0), abs=0.25)      # 좌우 대칭
        assert 48.0 <= b <= 62.0, (d, b)                                            # "대략 +-60 도"
        # 독립 구적으로 5% 를 만드는 각도와 대조
        ang = solve_angle_for_exposure(face, 180.0, 0.05, -1.0, dist=d)
        assert b == pytest.approx(ang, abs=1.5), (d, b, ang)
        budgets.append(b)
    assert all(b2 <= b1 + 0.25 for b1, b2 in zip(budgets, budgets[1:]))             # 멀수록(헤드가 작아 보일수록) 좁다
    big = back_face(h_f=0.05, head_radius_m=0.25)
    assert m.orbit_budget_deg(big, P0, +1.0) >= m.orbit_budget_deg(back_face(h_f=0.05), P0, +1.0) + 3.0


# ---------------------------------------------------------------- orbit_budget_deg: evaluate_path 와 일관
BUDGET_CASES = [
    ("F2 table", lambda: front_face(h_f=0.10, sigma_z=0.0), 30.0),
    ("F2 alpha", lambda: front_face(h_f=0.10, sigma_z=0.02), 13.0),
    ("F2 alpha1cm", lambda: front_face(h_f=0.10, sigma_z=0.01), 26.74),
    ("F3 table", lambda: front_face(h_f=0.20, sigma_z=0.0), 15.0),
    ("F3 alpha", lambda: front_face(h_f=0.20, sigma_z=0.03), 8.63),
    ("F1 free", lambda: front_face(h_f=0.015), 180.0),
]


@pytest.mark.parametrize("name,mk,cap", BUDGET_CASES)
def test_orbit_budget_equals_angle_cap_and_is_consistent_with_evaluate_path(name, mk, cap):
    m = fe()
    face = mk()
    for sign in (+1.0, -1.0):
        b = m.orbit_budget_deg(face, P0, sign)
        assert cap - 0.26 <= b <= cap + 1e-9, (name, b)
        assert ev(face, orbit_path(face, P0, sign * max(b - 0.01, 0.0), n=81)).ok
        if b < 170.0:
            assert not ev(face, orbit_path(face, P0, sign * (b + 1.0), n=81)).ok


def test_orbit_budget_consistent_for_oblique_observation_and_margin():
    m = fe()
    face = back_face(h_f=0.05)
    p0 = cam_pos(face, 150.0, 6.0, 2.0)                          # 비스듬한 뒷모습(F0)
    for sign in (+1.0, -1.0):
        b = m.orbit_budget_deg(face, p0, sign)
        assert 0.0 <= b < 180.0
        assert ev(face, orbit_path(face, p0, sign * max(b - 0.01, 0.0), n=81)).ok
        assert not ev(face, orbit_path(face, p0, sign * (b + 1.5), n=81)).ok
    f2 = front_face(h_f=0.10)
    assert m.orbit_budget_deg(f2, P0, +1.0, margin_deg=12.0) == pytest.approx(18.0, abs=0.26)
    assert m.orbit_budget_deg(f2, P0, +1.0, margin_deg=99.0) == 0.0
    with pytest.raises(ValueError):
        m.orbit_budget_deg(f2, P0, 0.0)
    with pytest.raises(ValueError):
        m.orbit_budget_deg(f2, P0, +1.0, step_deg=0.0)


# ---------------------------------------------------------------- 도착 샷: c0 를 명시
def test_explicit_c0_for_arrival_shot_gives_same_verdict_as_forward_path():
    face = back_face(h_f=0.05)
    fwd = orbit_path(face, P0, 55.0, n=41)
    rev = fwd[::-1].copy()
    a = ev(face, fwd)
    b = ev(face, rev, c0=fwd[0])
    for f in ("category", "violations", "ok"):
        assert getattr(a, f) == getattr(b, f)
    for f in ("yaw_obs_deg", "h_f_max", "m_peak", "yaw_dev_deg", "pitch_dev_deg", "unobs_max"):
        assert getattr(a, f) == pytest.approx(getattr(b, f), abs=1e-9)
    # c0 를 주지 않으면 첫 포즈가 관측 카메라로 간주된다(= 다른 판정).
    c = ev(face, rev)
    assert abs(c.yaw_obs_deg - a.yaw_obs_deg) > 10.0


# ---------------------------------------------------------------- 불변성 / 결정성
def test_verdict_is_mirror_symmetric_and_rigid_motion_invariant():
    face = front_face(h_f=0.12, sigma_z=0.01)
    path = orbit_path(face, P0, 22.0, 9.0)
    base = ev(face, path)
    mirror = ev(face, orbit_path(face, P0, -22.0, -9.0))
    assert base.ok and mirror.ok
    assert mirror.yaw_dev_deg == pytest.approx(base.yaw_dev_deg, abs=1e-9)
    assert mirror.pitch_dev_deg == pytest.approx(base.pitch_dev_deg, abs=1e-9)
    for seed in (5, 6):
        R, t = random_rigid(seed)
        got = ev(transform_face(face, R, t), transform_poses(path, R, t))
        for f in ("yaw_dev_deg", "pitch_dev_deg", "m_peak", "h_f_max", "yaw_obs_deg"):
            assert getattr(got, f) == pytest.approx(getattr(base, f), abs=1e-6), f
        assert got.violations == base.violations and got.category == base.category
    bf = back_face(h_f=0.05)
    bp = orbit_path(bf, P0, 58.0)
    b0 = ev(bf, bp)
    R, t = random_rigid(9)
    b1 = ev(transform_face(bf, R, t), transform_poses(bp, R, t))
    assert b1.unobs_max == pytest.approx(b0.unobs_max, abs=1e-3) and b1.violations == b0.violations


def test_evaluate_is_pure_deterministic_and_does_not_touch_global_rng():
    face = back_face(h_f=0.05)
    path = orbit_path(face, P0, 58.0)
    copy = path.copy()
    s0 = np.random.get_state()[1].copy()
    a, b = ev(face, path), ev(face, path)
    assert a == b and np.array_equal(path, copy)
    assert (np.random.get_state()[1] == s0).all()
    assert dataclasses.is_dataclass(a)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.ok = False


# ---------------------------------------------------------------- 여러 얼굴
def test_evaluate_faces_combines_verdicts():
    m = fe()
    near = front_face(h_f=0.10)
    far = make_face(front=(0, 0, -1), d=6.0, h_f=0.02)
    close = make_face(front=(0, 0, -1), d=1.0, h_f=0.25)
    path = orbit_path(near, P0, 20.0)
    ok = m.evaluate_faces([near, far], path)
    assert ok.ok and not m.face_viol_sigma([near, far], path) and len(ok.verdicts) == 2
    assert ok.allowed_grades == ("safe", "extended")                  # F2 와 F1 의 교집합
    bad = m.evaluate_faces([near, close], path)                       # close(F3) 는 15 도 한계 위반
    assert not bad.ok and m.face_viol_sigma([near, close], path)
    assert bad.violations == ("yaw",) and bad.allowed_grades == ("safe",)
    assert bad.verdicts[0].ok and not bad.verdicts[1].ok
    # 순서 무관, 딕셔너리 입력 허용
    again = m.evaluate_faces([close.to_dict(), near.to_dict()], path)
    assert again.violations == bad.violations and again.allowed_grades == bad.allowed_grades
    assert [v.category for v in again.verdicts] == ["F3", "F2"]
    g = m.evaluate_faces([near, close], orbit_path(near, P0, 5.0), grade="extended")
    assert not g.ok and "grade" in g.violations


# ---------------------------------------------------------------- 입력 검증
def test_evaluate_path_rejects_bad_poses():
    face = front_face()
    path = orbit_path(face, P0, 10.0)
    with pytest.raises(ValueError):
        ev(face, path[:0])
    with pytest.raises(ValueError):
        ev(face, path[:, :3, :])
    bad = path.copy()
    bad[3, 0, 3] = np.nan
    with pytest.raises(ValueError):
        ev(face, bad)
    bad = path.copy()
    bad[2, 3, :] = [0, 0, 0, 2.0]
    with pytest.raises(ValueError):
        ev(face, bad)
    with pytest.raises(ValueError):
        ev(face, path, c0=np.eye(3))
    with pytest.raises(ValueError):
        ev(face, path, margin_deg=-1.0)
    with pytest.raises(ValueError):
        ev(face, path, grade="platinum")
    inside = path.copy()
    inside[5, :3, 3] = face.center
    with pytest.raises(ValueError):
        ev(face, inside)
