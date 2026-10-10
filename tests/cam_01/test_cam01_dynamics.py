"""CAM-01: 1 kHz RK4 ODE 의 해석해 대조 (질량·감쇠·스프링·진자·마찰) + 제어(PID·ff·지연·포화)."""
import math

import numpy as np
import pytest
from scipy.special import ellipk

from cam01_helpers import DT, mass_damper_exact, odd_home, rigs


def _slider(**p):
    return rigs().make_rig("slider", params=p)


def _crane(**p):
    return rigs().make_rig("crane_jib", params=p)


def test_time_grid_is_exactly_1khz():
    assert rigs().DT == 1e-3 and rigs().RATE_HZ == 1000
    tr = _slider().simulate_force(0.0, 1.234)
    assert tr.t.shape == (1235,)                                    # round(T/dt)+1 개
    np.testing.assert_allclose(tr.t, np.arange(1235) * DT, atol=1e-12)
    assert tr.q.shape == tr.qd.shape == tr.u.shape == (1235, 1)
    assert tr.pose.shape == (1235, 4, 4)
    assert tr.q_ref is None                                         # 개루프에는 기준 궤적이 없다
    assert abs(tr.dt - DT) < 1e-18


def test_mass_damper_matches_closed_form_and_pose_rides_the_rail():
    home = odd_home()
    m = rigs()
    rig = m.make_rig("slider", params=dict(mass=2.0, damping=3.0), home_c2w=home)
    tr = rig.simulate_force(1.5, 2.0, q0=[0.1], qd0=[0.4])
    q, v = mass_damper_exact(tr.t, 2.0, 3.0, 1.5, 0.1, 0.4)
    np.testing.assert_allclose(tr.q[:, 0], q, atol=1e-9)
    np.testing.assert_allclose(tr.qd[:, 0], v, atol=1e-9)
    axis = np.asarray(rig.geom["axis"])
    exp = (home @ np.concatenate([q[:, None] * axis, np.ones((len(q), 1))], axis=1).T).T[:, :3]
    np.testing.assert_allclose(tr.pose[:, :3, 3], exp, atol=1e-9)    # 홈 포즈 합성까지 맞아야 한다
    np.testing.assert_allclose(tr.pose[:, :3, :3], np.broadcast_to(home[:3, :3], (len(q), 3, 3)), atol=1e-14)
    np.testing.assert_allclose(tr.u[:-1, 0], 1.5, atol=0)           # 인가 힘 기록


def test_quadratic_drag_terminal_velocity():
    m_, c2, f = 2.0, 0.5, 2.0
    tr = _slider(mass=m_, damping=0.0, quad_drag=c2).simulate_force(f, 6.0)
    vinf, rate = math.sqrt(f / c2), math.sqrt(f * c2) / m_
    np.testing.assert_allclose(tr.qd[:, 0], vinf * np.tanh(rate * tr.t), atol=1e-8)
    np.testing.assert_allclose(tr.q[:, 0], vinf / rate * np.log(np.cosh(rate * tr.t)), atol=1e-7)
    assert abs(tr.qd[-1, 0] - vinf) < 0.01 * vinf                    # 종단 속도 √(F/c2)


def test_underdamped_spring_mass_damper():
    m_, c, k, q0 = 1.0, 0.8, 25.0, 0.3
    tr = _slider(mass=m_, damping=c, stiffness=k).simulate_force(0.0, 3.0, q0=[q0])
    w0 = math.sqrt(k / m_)
    z = c / (2 * math.sqrt(k * m_))
    wd = w0 * math.sqrt(1 - z * z)
    t = tr.t
    exact = np.exp(-z * w0 * t) * q0 * (np.cos(wd * t) + z * w0 / wd * np.sin(wd * t))
    np.testing.assert_allclose(tr.q[:, 0], exact, atol=1e-8)
    # 무감쇠: 에너지 보존
    tr = _slider(mass=1.0, damping=0.0, stiffness=100.0).simulate_force(0.0, 5.0, q0=[0.2])
    e = 0.5 * tr.qd[:, 0] ** 2 + 0.5 * 100.0 * tr.q[:, 0] ** 2
    assert np.ptp(e) / e[0] < 1e-9
    np.testing.assert_allclose(tr.q[:, 0], 0.2 * np.cos(10.0 * tr.t), atol=1e-7)


def test_pendulum_energy_and_exact_period():
    w2 = 9.0
    th0 = 1.0
    tr = _crane(mass=1.0, damping=0.0, w2=w2).simulate_force(np.zeros(2), 8.5, q0=[th0, 0.0])
    e = 0.5 * tr.qd[:, 0] ** 2 + w2 * (1.0 - np.cos(tr.q[:, 0]))
    assert np.ptp(e) / e[0] < 1e-9                                   # 비선형 진자의 에너지 보존
    assert np.abs(tr.q[:, 1]).max() == 0.0                           # DOF 간 결합 없음
    t, q = tr.t, tr.q[:, 0]
    i = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]                     # 하향 영교차
    tc = t[i] + DT * q[i] / (q[i] - q[i + 1])
    assert len(tc) >= 3
    period = float(np.mean(np.diff(tc)))
    exact = 4.0 * ellipk(math.sin(th0 / 2.0) ** 2) / math.sqrt(w2)   # 큰 진폭의 정확한 주기
    assert abs(period / exact - 1.0) < 2e-4
    assert abs(exact / (2 * math.pi / 3.0) - 1.0) > 0.05             # (큰 진폭이 선형 주기와 확실히 다름을 확인)
    # 작은 진폭은 2π/ω0
    tr = _crane(mass=1.0, damping=0.0, w2=w2).simulate_force(np.zeros(2), 6.5, q0=[0.05, 0.0])
    q = tr.q[:, 0]
    i = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]
    tc = tr.t[i] + DT * q[i] / (q[i] - q[i + 1])
    assert abs(np.mean(np.diff(tc)) / (2 * math.pi / 3.0) - 1.0) < 5e-4


def test_pendulum_equilibrium_is_home_and_force_sign():
    rig = _crane(mass=1.0, damping=0.0, w2=4.0)
    tr = rig.simulate_force(np.zeros(2), 1.0)
    assert np.abs(tr.q).max() == 0.0 and np.abs(tr.qd).max() == 0.0   # q=0 이 평형
    tr = rig.simulate_force(np.array([2.0, 0.0]), 0.05)
    assert tr.qd[-1, 0] > 0 and tr.qd[-1, 1] == 0.0                   # +힘 -> +가속, 다른 DOF 불변


def test_coulomb_friction_sliding_stopping_and_stiction_creep():
    eps = rigs().V_EPS
    assert eps == 1e-3
    fc = 0.5
    tr = _slider(mass=1.0, damping=0.0, coulomb=fc).simulate_force(0.0, 6.0, qd0=[2.0])
    sl = tr.t <= 3.9                                                  # v >= 0.05 >> ε 구간: 등감속 fc/m
    np.testing.assert_allclose(tr.qd[sl, 0], 2.0 - fc * tr.t[sl], atol=1e-5)
    assert abs(tr.qd[-1, 0]) < 1e-3
    assert abs(tr.q[-1, 0] - 4.0) < 0.02                              # 정지 거리 v0²/(2 fc/m)
    tr = _slider(mass=1.0, damping=0.0, coulomb=fc).simulate_force(0.3, 3.0)   # 정지마찰보다 작은 힘
    vss = eps * math.atanh(0.3 / fc)                                  # f_c·tanh(v/ε) = F  의 정상해
    assert abs(tr.qd[-1, 0] - vss) < 3e-5
    assert abs(tr.q[-1, 0]) < 5e-3


def test_disturbance_is_opt_in_for_open_loop_and_seeded():
    m = rigs()
    rig = m.make_rig("shoulder_handheld")
    a = rig.simulate_force(np.zeros(rig.n_q), 1.0)
    assert np.abs(a.q).max() == 0.0                                    # disturbance=False 이면 순수 개루프
    b = rig.simulate_force(np.zeros(rig.n_q), 1.0, disturbance=True, seed=3)
    c = rig.simulate_force(np.zeros(rig.n_q), 1.0, disturbance=True, seed=3)
    d = rig.simulate_force(np.zeros(rig.n_q), 1.0, disturbance=True, seed=4)
    assert np.abs(b.q).max() > 0.0
    assert np.array_equal(b.q, c.q) and np.array_equal(b.pose, c.pose)
    assert not np.array_equal(b.q, d.q)


def test_callable_force_sampled_once_per_step_at_grid_times():
    rig = _slider(mass=2.0, damping=3.0)
    seen = []

    def f(t):
        seen.append(t)
        return np.array([1.5])

    a = rig.simulate_force(f, 0.5)
    b = rig.simulate_force(np.array([1.5]), 0.5)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.u, b.u)
    assert len(seen) == 500
    np.testing.assert_allclose(seen, np.arange(500) * DT, atol=1e-12)


def test_min_jerk_tracking_feedforward_vs_pd_only():
    m = rigs()
    base = dict(mass=5.0, damping=10.0, kp=400.0, kd=60.0, ki=0.0)
    it = m.Intent.point_to_point([0.0], [1.0], 2.0, tail_s=3.0)
    ff = m.make_rig("slider", params=dict(base, ff_gain=1.0)).simulate(it)
    assert ff.q_ref.shape == ff.q.shape
    assert np.abs(ff.q - ff.q_ref).max() < 1e-3                      # 역동역학 ff: ZOH 오차만 남는다
    np.testing.assert_allclose(ff.q[0], [0.0], atol=0)
    assert abs(ff.q[2000, 0] - 1.0) < 1e-3 and abs(ff.qd[2000, 0]) < 5e-3   # t=T 에서 도착, 정지
    pd = m.make_rig("slider", params=dict(base, ff_gain=0.0)).simulate(it)
    lag = np.abs(pd.q - pd.q_ref).max()
    assert lag > 1e-2                                                # 감쇠 c·v/kp ≈ 2 cm 지연
    assert abs(pd.q[-1, 0] - 1.0) < 1e-4 and abs(pd.qd[-1, 0]) < 1e-4  # 그래도 정착
    ref = it.reference(ff.t)
    np.testing.assert_allclose(ff.q_ref, ref[0], atol=1e-12)         # q_ref 는 Intent.reference 와 같다


def test_integral_action_removes_gravity_offset():
    m = rigs()
    it = m.Intent(np.array([0.0, 1.0]), np.full((2, 2), 0.3), tail_s=9.0)
    short = m.Intent(np.array([0.0, 1.0]), np.full((2, 2), 0.3), tail_s=2.0)
    common = dict(mass=1.0, damping=0.5, kp=50.0, kd=20.0, w2=4.0, ff_gain=0.0)
    no_i = m.make_rig("crane_jib", params=dict(common, ki=0.0)).simulate(short, q0=[0.3, 0.3])
    with_i = m.make_rig("crane_jib", params=dict(common, ki=200.0)).simulate(it, q0=[0.3, 0.3])
    assert abs(no_i.q[-1, 0] - 0.3) > 0.01                           # P 만으로는 중력 정상 오차 ≈ w2 sin(0.3)/kp
    assert abs(with_i.q[-1, 0] - 0.3) < 1e-3                         # I 항이 제거
    assert abs(with_i.qd[-1, 0]) < 1e-3


def test_delayed_feedback_uses_stale_error():
    m = rigs()
    it = m.Intent(np.array([0.0, 1.0]), np.zeros((2, 1)), tail_s=2.0)     # 상수 기준 0
    common = dict(mass=1.0, damping=0.5, kp=10.0, kd=0.0, ki=0.0, ff_gain=0.0)
    d0 = m.make_rig("slider", params=dict(common, delay_s=0.0)).simulate(it, q0=[0.1])
    dd = m.make_rig("slider", params=dict(common, delay_s=0.2)).simulate(it, q0=[0.1])
    # 처음 δ 동안은 e(0) = -0.1 을 계속 본다: 일정한 힘 F=-1 의 개루프 해석해
    sel = dd.t <= 0.2 + 1e-12
    q, v = mass_damper_exact(dd.t[sel], 1.0, 0.5, -1.0, 0.1, 0.0)
    np.testing.assert_allclose(dd.q[sel, 0], q, atol=1e-9)
    np.testing.assert_allclose(dd.u[:200, 0], -1.0, atol=1e-12)
    # 지연 없는 루프는 즉시 반응하므로 해석해에서 벗어난다
    assert np.abs(d0.q[sel, 0] - q).max() > 1e-5

    def first_cross(tr):
        i = int(np.argmax(tr.q[:, 0] <= 0.0))
        return tr.t[i]

    assert first_cross(dd) < first_cross(d0) - 0.02                  # 낡은 오차로 더 밀어 먼저(그리고 더 크게) 넘어간다
    assert dd.q[:, 0].min() < d0.q[:, 0].min() - 0.01                # 오버슛이 커진다


def test_control_saturation_u_max():
    m = rigs()
    it = m.Intent.point_to_point([0.0], [1.0], 1.0, tail_s=0.0)
    rig = m.make_rig("slider", params=dict(mass=3.0, damping=0.0, u_max=1.0))
    tr = rig.simulate(it)                                            # 필요한 힘 ≈ 17 N >> 1 N
    assert np.abs(tr.u).max() <= 1.0 + 1e-12                         # 외란이 없으면 u ≤ u_max
    assert np.abs(tr.q - tr.q_ref).max() > 0.1                       # 따라서 추종 실패
    free = m.make_rig("slider", params=dict(mass=3.0, damping=0.0))
    assert np.abs(free.simulate(it).u).max() > 10.0


def test_simulate_does_not_touch_inputs_or_global_rng():
    m = rigs()
    times = np.array([0.0, 1.0])
    wp = np.array([[0.0, 0.0], [0.4, 0.2]])
    it = m.Intent(times, wp, tail_s=0.5)
    kt, kw = times.copy(), wp.copy()
    q0 = np.array([0.0, 0.0])
    s0 = np.random.get_state()[1].copy()
    rig = m.make_rig("crane_jib")
    a = rig.simulate(it, q0=q0)
    b = rig.simulate(it, q0=q0)
    assert np.array_equal(times, kt) and np.array_equal(wp, kw) and np.array_equal(q0, [0.0, 0.0])
    assert np.array_equal(it.times, kt) and np.array_equal(it.waypoints, kw)
    assert (np.random.get_state()[1] == s0).all()                    # 전역 RNG 불변
    assert np.array_equal(a.q, b.q) and np.array_equal(a.pose, b.pose) and np.array_equal(a.u, b.u)


@pytest.mark.parametrize("bad", [
    dict(q0=[0.0, 0.0]),            # 슬라이더는 1자유도
    dict(qd0=[0.0, 0.0]),
    dict(q0=[np.nan]),
])
def test_initial_state_validation(bad):
    with pytest.raises(ValueError):
        _slider().simulate_force(0.0, 0.1, **bad)
    with pytest.raises(ValueError):
        _slider().simulate(rigs().Intent.point_to_point([0.0], [1.0], 1.0), **bad)


def test_duration_and_intent_dof_validation():
    rig = _slider()
    for dur in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            rig.simulate_force(0.0, dur)
    with pytest.raises(ValueError):
        rig.simulate(rigs().Intent.point_to_point([0.0, 0.0], [1.0, 1.0], 1.0))   # 2 DOF intent -> 1 DOF 리그
