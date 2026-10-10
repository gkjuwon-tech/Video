"""CAM-02: 촬영 메뉴 봉투 Π_g — 경로 측정, 등급별 봉투, 최소 등급, 소리(soundness)와 포함 관계."""
import dataclasses
import math

import numpy as np
import pytest

from cam02_helpers import T4, Rx, Ry, Rz, mm, odd_home

Z = 3.0


def _M(**kw):
    m = mm()
    d = dict(alpha_deg=0.0, yaw_deg=0.0, m_peak=1.0, disp_m=0.0)
    d.update(kw)
    return m.PathMeasure(**d)


# ---------------------------------------------------------------- measure
def test_measure_known_paths():
    m = mm()
    for home in (np.eye(4), odd_home()):
        o = m.measure(m.ideal_path(m.recipe("orbit", Z), home).c2w, Z)
        assert isinstance(o, m.PathMeasure)
        assert abs(o.alpha_deg - 20.0) < 1e-9 and o.yaw_deg < 1e-9 and abs(o.m_peak - 1.0) < 1e-12
        assert abs(o.disp_m - 2.0 * Z * math.sin(math.radians(10.0))) < 1e-9        # 현(chord)
        p = m.measure(m.ideal_path(m.recipe("push_in", Z, m=1.5), home).c2w, Z)
        assert p.alpha_deg < 1e-9 and p.yaw_deg < 1e-9
        assert abs(p.m_peak - 1.5) < 1e-12 and abs(p.disp_m - 1.0) < 1e-12
        s = m.measure(m.ideal_path(m.recipe("slider", Z, slide_zs=0.1), home).c2w, Z)
        a = math.degrees(math.atan(0.1))
        assert abs(s.alpha_deg - a) < 1e-9 and abs(s.yaw_deg - a) < 1e-9            # 슬라이더는 재조준하지 않는다
        assert abs(s.m_peak - 1.0) < 1e-12 and abs(s.disp_m - 0.3) < 1e-12
        r = m.measure(m.ideal_path(m.recipe("rack_focus", Z, focus_to_m=9.0), home).c2w, Z)
        assert max(r.alpha_deg, r.yaw_deg, r.disp_m) < 1e-9 and abs(r.m_peak - 1.0) < 1e-12


def test_measure_pull_out_and_single_frame():
    m = mm()
    po = m.measure(m.ideal_path(m.recipe("push_in", Z, m=0.5)).c2w, Z)
    assert abs(po.m_peak - 1.0) < 1e-12 and abs(po.disp_m - Z) < 1e-12                          # 뒤로 빠지면 배율 최대는 첫 프레임
    one = m.measure(np.eye(4)[None], Z)
    assert max(one.alpha_deg, one.yaw_deg, one.disp_m) < 1e-9 and abs(one.m_peak - 1.0) < 1e-12


def test_measure_in_place_pan_counts_as_yaw_and_roll_is_ignored():
    m = mm()
    c0 = odd_home()
    for deg in (10.0, 25.0, 90.0, 180.0):
        c = np.stack([c0, c0 @ T4(Ry(math.radians(deg)))])
        r = m.measure(c, Z)
        assert abs(r.yaw_deg - deg) < 1e-9 and r.alpha_deg < 1e-9 and r.disp_m < 1e-9
    tilt = m.measure(np.stack([c0, c0 @ T4(Rx(math.radians(12.0)))]), Z)
    assert abs(tilt.yaw_deg - 12.0) < 1e-9
    roll = m.measure(np.stack([c0, c0 @ T4(Rz(math.radians(30.0)))]), Z)
    assert roll.yaw_deg < 1e-9 and roll.alpha_deg < 1e-9                            # 롤은 측정하지 않는다


def test_measure_errors():
    m = mm()
    ok = m.ideal_path(m.recipe("orbit", Z)).c2w
    with pytest.raises(ValueError):
        m.measure(ok, 0.0)
    with pytest.raises(ValueError):
        m.measure(ok, -1.0)
    with pytest.raises(ValueError):
        m.measure(ok[0], Z)                                                        # (4,4) 는 열이 아니다
    with pytest.raises(ValueError):
        m.measure(ok[:, :3, :], Z)
    bad = ok.copy()
    bad[5, 0, 0] = np.nan
    with pytest.raises(ValueError):
        m.measure(bad, Z)
    bad = ok.copy()
    bad[7, :3, :3] *= 1.5                                                          # 회전 아님
    with pytest.raises(ValueError):
        m.measure(bad, Z)
    sub = ok.copy()
    sub[3, :3, 3] = (ok[0] @ np.array([0.0, 0.0, Z, 1.0]))[:3]
    with pytest.raises(ValueError):                                                # 카메라가 피사체 위
        m.measure(sub, Z)


# ---------------------------------------------------------------- envelope table
def test_envelope_table():
    m = mm()
    s, e, f = (m.envelope(g) for g in m.GRADES)
    assert all(isinstance(x, m.Envelope) for x in (s, e, f))
    assert (s.grade, e.grade, f.grade) == ("safe", "extended", "full")
    assert (s.max_alpha_deg, s.max_yaw_deg, s.max_slide_zs, s.m_max, s.max_disp_m) == (20.0, 20.0, 0.15, 1.5, math.inf)
    assert (s.ir_cap_mean, s.ir_cap_max) == (0.20, 0.30)
    assert (e.max_alpha_deg, e.max_yaw_deg, e.max_slide_zs, e.m_max, e.max_disp_m) == (90.0, 90.0, math.inf, 1.5, math.inf)
    assert (e.ir_cap_mean, e.ir_cap_max) == (0.65, 0.85)
    assert (f.max_alpha_deg, f.max_yaw_deg, f.max_slide_zs, f.m_max, f.max_disp_m) == (math.inf,) * 5
    assert f.ir_cap_mean is None and f.ir_cap_max is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.max_alpha_deg = 99.0
    mat = m.envelope("safe", m_max=m.M_MAX_MATERIAL, observed_depth_m=5.0)
    assert mat.m_max == 4.0 and mat.max_disp_m == 5.0                               # 관측 깊이는 안전·확장 모두에 건다
    ext = m.envelope("extended", observed_depth_m=5.0)
    assert ext.max_disp_m == 5.0 and ext.m_max == 1.5
    assert m.envelope("full", m_max=2.0, observed_depth_m=1.0).m_max == math.inf    # 풀은 제한 없음
    for bad in ("", "green", "SAFE"):
        with pytest.raises(ValueError):
            m.envelope(bad)
    for kw in (dict(m_max=0.0), dict(m_max=-1.0), dict(m_max=float("nan")), dict(observed_depth_m=0.0),
               dict(observed_depth_m=-2.0), dict(observed_depth_m=float("nan"))):
        with pytest.raises(ValueError):
            m.envelope("safe", **kw)


def test_contains_boundaries():
    m = mm()
    s, e, f = (m.envelope(g) for g in m.GRADES)
    assert s.contains(_M(alpha_deg=20.0, yaw_deg=20.0, m_peak=1.5)) and not s.contains(_M(alpha_deg=20.01))
    assert not s.contains(_M(yaw_deg=20.01)) and not s.contains(_M(m_peak=1.51))
    assert s.contains(_M(alpha_deg=20.0 + 0.5 * m.BOUND_TOL))                     # 경계 허용 오차 안
    assert e.contains(_M(alpha_deg=90.0, yaw_deg=90.0)) and not e.contains(_M(alpha_deg=90.01))
    assert not e.contains(_M(yaw_deg=90.01)) and not e.contains(_M(m_peak=1.51))
    assert f.contains(_M(alpha_deg=1e6, yaw_deg=1e6, m_peak=1e6, disp_m=1e9))
    d = m.envelope("extended", observed_depth_m=4.0)
    assert d.contains(_M(disp_m=4.0)) and not d.contains(_M(disp_m=4.01))
    big = m.envelope("safe", m_max=4.0)
    assert big.contains(_M(m_peak=4.0)) and not big.contains(_M(m_peak=4.01))


def test_envelopes_are_nested_for_random_measures():
    m = mm()
    rng = np.random.default_rng(3)
    for _ in range(400):
        ms = _M(alpha_deg=float(rng.choice([rng.uniform(0, 30), rng.uniform(0, 200)])),
                yaw_deg=float(rng.choice([rng.uniform(0, 30), rng.uniform(0, 200)])),
                m_peak=float(rng.uniform(1.0, 6.0)), disp_m=float(rng.uniform(0, 10)))
        md = float(rng.uniform(0.5, 6.0))
        dep = float(rng.uniform(0.5, 12.0))
        s, e, f = (m.envelope(g, m_max=md, observed_depth_m=dep) for g in m.GRADES)
        assert (not s.contains(ms)) or e.contains(ms)
        assert (not e.contains(ms)) or f.contains(ms)
        assert f.contains(ms)


# ---------------------------------------------------------------- grades of specs / paths
def _g(name, z=Z, **kw):
    m = mm()
    spec_kw = {k: v for k, v in kw.items() if k not in ("m_max", "observed_depth_m")}
    env_kw = {k: v for k, v in kw.items() if k in ("m_max", "observed_depth_m")}
    return m.min_grade(m.recipe(name, z, **spec_kw), **env_kw)


def test_bench_intents_grades():
    m = mm()
    got = [m.min_grade(m.bench_spec(k, Z)) for k in m.BENCH_INTENTS]
    assert got == ["safe", "safe", "safe", "extended", "full"]                      # I1 푸시인, I2·I3 +-20 deg 아크, I4 90 deg, I5 180 deg


@pytest.mark.parametrize("deg,grade", [(0.0, "safe"), (20.0, "safe"), (-20.0, "safe"), (20.5, "extended"),
                                       (-45.0, "extended"), (90.0, "extended"), (90.5, "full"), (180.0, "full"),
                                       (360.0, "full")])
def test_orbit_grades(deg, grade):
    assert _g("orbit", orbit_deg=deg) == grade


@pytest.mark.parametrize("mag,mm_,grade", [(1.5, 1.5, "safe"), (1.0, 1.5, "safe"), (0.5, 1.5, "safe"),
                                           (1.5001, 1.5, "full"), (2.0, 1.5, "full"),
                                           (3.0, 4.0, "safe"), (4.0, 4.0, "safe"), (4.2, 4.0, "full")])
def test_push_in_grades_depend_on_the_cell_m_max(mag, mm_, grade):
    assert _g("push_in", m=mag, m_max=mm_) == grade


@pytest.mark.parametrize("sl,grade", [(0.0, "safe"), (0.10, "safe"), (-0.15, "safe"), (0.16, "extended"),
                                      (-0.5, "extended")])
def test_slider_grades(sl, grade):
    assert _g("slider", slide_zs=sl) == grade


def test_rack_focus_grades_split_between_sharp_planes_and_melted_background():
    assert _g("rack_focus", focus_to_m=9.0) == "safe"                              # 선명면 사이
    assert _g("rack_focus", focus_to_m=9.0, to_blur=True) == "extended"           # 녹은 배경으로 (F-P6)
    m = mm()
    assert not m.envelope("safe").contains_spec(m.recipe("rack_focus", Z, focus_to_m=9.0, to_blur=True))
    assert m.envelope("extended").contains_spec(m.recipe("rack_focus", Z, focus_to_m=9.0, to_blur=True))


def test_observed_depth_limits_the_extended_grade_by_camera_displacement():
    chord90 = 2.0 * Z * math.sin(math.radians(45.0))                                # 4.243 m
    assert _g("orbit", orbit_deg=90.0, observed_depth_m=5.0) == "extended"
    assert _g("orbit", orbit_deg=90.0, observed_depth_m=chord90 * 0.99) == "full"
    assert _g("orbit", orbit_deg=20.0, observed_depth_m=5.0) == "safe"
    assert _g("orbit", orbit_deg=20.0, observed_depth_m=0.5) == "full"             # 현 1.04 m > 깊이 0.5 m: 안전·확장 모두 밖
    assert _g("slider", slide_zs=0.5, observed_depth_m=2.0) == "extended"          # 변위 1.5 m <= 깊이 2 m
    assert _g("slider", slide_zs=0.5, observed_depth_m=1.0) == "full"              # 변위 1.5 m > 깊이 1 m


def test_min_grade_path_matches_min_grade_for_rotating_and_pushing_moves():
    m = mm()
    for home in (np.eye(4), odd_home()):
        for name, kw in (("orbit", dict(orbit_deg=20.0)), ("orbit", dict(orbit_deg=60.0)), ("orbit", dict(orbit_deg=120.0)),
                         ("push_in", dict(m=1.5)), ("push_in", dict(m=3.0)), ("rack_focus", dict(focus_to_m=9.0))):
            sp = m.recipe(name, Z, **kw)
            c2w = m.ideal_path(sp, home).c2w
            for env_kw in (dict(), dict(m_max=4.0), dict(observed_depth_m=4.0)):
                assert m.min_grade_path(c2w, Z, **env_kw) == m.min_grade(sp, **env_kw), (name, kw, env_kw)


def test_min_grade_path_for_pans_and_look_back():
    m = mm()
    c0 = odd_home()
    grade = lambda deg: m.min_grade_path(np.stack([c0, c0 @ T4(Ry(math.radians(deg)))]), Z)
    assert grade(10.0) == "safe" and grade(20.0) == "safe" and grade(25.0) == "extended"
    assert grade(90.0) == "extended" and grade(120.0) == "full" and grade(180.0) == "full"      # 뒤돌아보기


def test_spec_envelope_is_sound_wrt_path_measure_and_nested():
    """contains_spec 이면 그 이상 경로의 측정도 같은 봉투에 든다(소리), 그리고 안전 ⊂ 확장 ⊂ 풀."""
    m = mm()
    specs = []
    for deg in (0.0, 5.0, 20.0, -20.0, 20.5, 45.0, 90.0, 135.0, 180.0, 360.0):
        specs.append(m.recipe("orbit", Z, orbit_deg=deg))
        specs.append(m.recipe("orbit", Z, orbit_deg=deg, round_trip=True))
    for mag in (0.4, 1.0, 1.5, 2.0, 4.0, 6.0):
        specs.append(m.recipe("push_in", Z, m=mag))
    for sl in (0.0, 0.1, 0.15, 0.2, -0.4):
        specs.append(m.recipe("slider", Z, slide_zs=sl))
    specs.append(m.recipe("rack_focus", Z, focus_to_m=7.0))
    specs.append(m.recipe("rack_focus", Z, focus_to_m=7.0, to_blur=True))
    for sp in specs:
        for env_kw in (dict(), dict(m_max=4.0, observed_depth_m=6.0), dict(m_max=2.0, observed_depth_m=2.0)):
            envs = [m.envelope(g, **env_kw) for g in m.GRADES]
            inc = [e.contains_spec(sp) for e in envs]
            assert (not inc[0]) or inc[1]
            assert (not inc[1]) or inc[2]
            assert inc[2]
            ms = m.measure(m.ideal_path(sp).c2w, Z)
            for e, ok in zip(envs, inc):
                if ok:
                    assert e.contains(ms) and e.contains_path(m.ideal_path(sp).c2w, Z), (sp, e.grade)
            mg = m.min_grade(sp, **env_kw)
            assert inc[m.GRADES.index(mg)] and not any(inc[:m.GRADES.index(mg)])


def test_ir_caps_grow_with_grade_and_are_exposed():
    m = mm()
    s, e, f = (m.envelope(g) for g in m.GRADES)
    assert s.ir_cap_mean < e.ir_cap_mean < 1.0 and s.ir_cap_max < e.ir_cap_max <= 1.0
    assert s.ir_cap_mean < s.ir_cap_max and e.ir_cap_mean < e.ir_cap_max
    assert f.ir_cap_mean is None
