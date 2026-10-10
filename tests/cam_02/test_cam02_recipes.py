"""CAM-02: 상수·MoveSpec 검증·recipe/bench_spec·프레임 시간축·ease 프로파일."""
import dataclasses
import math

import numpy as np
import pytest

from cam02_helpers import DEFAULT_FRAMES, mm, s_mj, s_rt


def test_constants():
    m = mm()
    assert tuple(m.MOVE_NAMES) == ("push_in", "slider", "orbit", "rack_focus")
    assert tuple(m.GRADES) == ("safe", "extended", "full")
    assert tuple(m.BENCH_INTENTS) == ("I1", "I2", "I3", "I4", "I5")
    assert m.BENCH_DURATION_S == 8.0 and m.BENCH_FPS == 24.0
    assert m.M_MAX_SEMANTIC == 1.5 and m.M_MAX_MATERIAL == 4.0
    assert 0.0 < m.BOUND_TOL <= 1e-6


def test_recipe_defaults_follow_the_bench_definitions():
    m = mm()
    p = m.recipe("push_in", 3.0)
    assert (p.name, p.z_s, p.m, p.duration_s, p.fps, p.round_trip) == ("push_in", 3.0, 1.5, 8.0, 24.0, False)
    s = m.recipe("slider", 3.0)
    assert s.slide_zs == 0.10 and s.m == 1.0 and s.orbit_deg == 0.0
    o = m.recipe("orbit", 3.0)
    assert o.orbit_deg == 20.0 and o.m == 1.0 and o.slide_zs == 0.0
    r = m.recipe("rack_focus", 3.0, focus_to_m=9.0)
    assert r.focus_from_m == 3.0 and r.focus_to_m == 9.0 and r.to_blur is False
    # 재정의는 그대로 반영, 모든 MoveSpec 은 얼어 있다
    assert m.recipe("orbit", 2.0, orbit_deg=-35.0, duration_s=4.0, fps=30.0).n_frames == 121
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.m = 2.0
    for sp in (p, s, o, r):
        assert isinstance(sp, m.MoveSpec)
        assert sp.n_frames == DEFAULT_FRAMES


@pytest.mark.parametrize("bad", [
    dict(name="dolly_zoom", z_s=3.0),                                    # 모르는 무브
    dict(name="orbit", z_s=0.0), dict(name="orbit", z_s=-1.0), dict(name="orbit", z_s=float("nan")),
    dict(name="orbit", z_s=float("inf")),
    dict(name="orbit", z_s=3.0, duration_s=0.0), dict(name="orbit", z_s=3.0, fps=-24.0),
    dict(name="orbit", z_s=3.0, duration_s=7.3),                         # 7.3*24 = 175.2: 종점이 격자에 안 놓임
    dict(name="push_in", z_s=3.0, m=0.0), dict(name="push_in", z_s=3.0, m=float("nan")),
    dict(name="orbit", z_s=3.0, m=1.5),                                  # 해당 없는 필드는 기본값이어야 한다
    dict(name="push_in", z_s=3.0, slide_zs=0.1), dict(name="slider", z_s=3.0, orbit_deg=20.0),
    dict(name="orbit", z_s=3.0, orbit_deg=361.0), dict(name="orbit", z_s=3.0, orbit_deg=float("nan")),
    dict(name="rack_focus", z_s=3.0),                                    # 랙은 초점 구간 필수
    dict(name="rack_focus", z_s=3.0, focus_from_m=3.0),
    dict(name="rack_focus", z_s=3.0, focus_from_m=3.0, focus_to_m=0.0),
    dict(name="rack_focus", z_s=3.0, focus_from_m=-3.0, focus_to_m=5.0),
    dict(name="orbit", z_s=3.0, focus_from_m=3.0, focus_to_m=5.0),       # 랙 아닌 무브에 초점 구간
    dict(name="slider", z_s=3.0, to_blur=True),
])
def test_movespec_rejects_invalid(bad):
    with pytest.raises(ValueError):
        mm().MoveSpec(**bad)


def test_recipe_errors():
    m = mm()
    with pytest.raises(ValueError):
        m.recipe("barrel_roll", 3.0)
    with pytest.raises(ValueError):
        m.recipe("rack_focus", 3.0)                                      # focus_to_m 없음
    with pytest.raises(ValueError):
        m.recipe("orbit", 3.0, m=2.0)


def test_bench_spec_is_the_registered_definition():
    m = mm()
    for z in (1.0, 3.0, 7.5):
        i1, i2, i3, i4, i5 = (m.bench_spec(k, z) for k in ("I1", "I2", "I3", "I4", "I5"))
        assert (i1.name, i1.m, i1.z_s) == ("push_in", 1.5, z)
        assert (i2.name, i2.orbit_deg, i2.round_trip) == ("orbit", 20.0, False)
        assert (i3.name, i3.orbit_deg, i3.round_trip) == ("orbit", 20.0, True)
        assert (i4.name, i4.orbit_deg, i4.round_trip) == ("orbit", 90.0, False)
        assert (i5.name, i5.orbit_deg, i5.round_trip) == ("orbit", 180.0, False)
        for sp in (i1, i2, i3, i4, i5):
            assert sp.duration_s == 8.0 and sp.fps == 24.0 and sp.n_frames == DEFAULT_FRAMES
    for bad in ("I6", "i1", "", "push_in"):
        with pytest.raises(ValueError):
            m.bench_spec(bad, 3.0)
    with pytest.raises(ValueError):
        m.bench_spec("I1", 0.0)


@pytest.mark.parametrize("dur,fps,n", [(8.0, 24.0, 193), (2.5, 30.0, 76), (1.0, 24.0, 25), (0.5, 10.0, 6)])
def test_frame_times(dur, fps, n):
    m = mm()
    sp = m.recipe("orbit", 3.0, duration_s=dur, fps=fps)
    t = m.frame_times(sp)
    assert sp.n_frames == n and t.shape == (n,) and t.dtype == np.float64
    assert t[0] == 0.0 and abs(t[-1] - dur) < 1e-12
    np.testing.assert_allclose(np.diff(t), 1.0 / fps, atol=1e-12)
    assert m.ideal_path(sp).t.shape == (n,)


def test_ease_profile_matches_min_jerk():
    m = mm()
    tau = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(m.ease_profile(tau), s_mj(tau), atol=1e-14)
    from buc.camera.rigs import min_jerk_profile
    np.testing.assert_allclose(m.ease_profile(tau), min_jerk_profile(tau)[0], atol=1e-14)   # CAM-01 과 같은 곡선
    s = m.ease_profile(tau)
    assert s[0] == 0.0 and s[-1] == 1.0 and abs(m.ease_profile(0.5) - 0.5) < 1e-15
    assert np.all(np.diff(s) >= 0)
    np.testing.assert_allclose(s + s[::-1], 1.0, atol=1e-14)                       # 점대칭
    # 범위 밖 클램프, 스칼라/배열 모양 보존
    assert m.ease_profile(-0.3) == 0.0 and m.ease_profile(1.7) == 1.0
    assert np.asarray(m.ease_profile(0.25)).shape == () and m.ease_profile(np.zeros((2, 3))).shape == (2, 3)


def test_ease_profile_round_trip():
    m = mm()
    tau = np.linspace(0.0, 1.0, 193)
    s = m.ease_profile(tau, round_trip=True)
    np.testing.assert_allclose(s, s_rt(tau), atol=1e-14)
    assert s[0] == 0.0 and s[-1] == 0.0 and abs(float(m.ease_profile(0.5, True)) - 1.0) < 1e-15
    np.testing.assert_allclose(s, s[::-1], atol=1e-14)                              # 시간 대칭
    assert np.all(np.diff(s[:97]) >= 0) and np.all(np.diff(s[96:]) <= 0)
    # 가운데에서 정지: 속도 0
    assert abs(s[96] - s[95]) < 1e-3 and math.isclose(float(m.ease_profile(0.5, True)), 1.0)
