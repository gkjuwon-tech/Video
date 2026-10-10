"""CAM-01: min-jerk, 2/3 멱법칙, OU 잡음 (순수 수식)."""
import math

import numpy as np
import pytest

from cam01_helpers import rigs


def test_min_jerk_profile_boundaries_and_symmetry():
    m = rigs()
    s, d1, d2 = (np.asarray(x) for x in m.min_jerk_profile(np.array([0.0, 0.5, 1.0])))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0], atol=1e-15)
    np.testing.assert_allclose(d1, [0.0, 15.0 / 8.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(d2, [0.0, 0.0, 0.0], atol=1e-13)
    tau = np.linspace(0, 1, 101)
    a = np.asarray(m.min_jerk_profile(tau)[0])
    b = np.asarray(m.min_jerk_profile(1.0 - tau)[0])
    np.testing.assert_allclose(a + b, 1.0, atol=1e-14)            # s(τ) + s(1-τ) = 1
    assert np.all(np.diff(a) >= 0)                                # 단조
    # [0,1] 밖은 클램프(속도·가속도 0)
    s_o, d1_o, d2_o = (np.asarray(x) for x in m.min_jerk_profile(np.array([-0.5, 1.7])))
    np.testing.assert_allclose(s_o, [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(d1_o, 0.0, atol=1e-15)
    np.testing.assert_allclose(d2_o, 0.0, atol=1e-15)


def test_min_jerk_profile_derivatives_and_extrema():
    m = rigs()
    tau = np.linspace(0.0, 1.0, 20001)
    s, d1, d2 = (np.asarray(x) for x in m.min_jerk_profile(tau))
    h = tau[1] - tau[0]
    np.testing.assert_allclose(np.gradient(s, h)[2:-2], d1[2:-2], atol=2e-6)
    np.testing.assert_allclose(np.gradient(d1, h)[2:-2], d2[2:-2], atol=2e-5)
    assert abs(d1.max() - 15.0 / 8.0) < 1e-9                       # 최고 속도 15/8
    assert abs(np.abs(d2).max() - 10.0 / math.sqrt(3.0)) < 1e-6    # 최고 가속도 10/√3
    jerk = np.gradient(d2, h)
    cost = float(np.sum(jerk[1:-1] ** 2) * h)                      # ∫ s'''² dτ = 720 (최소 저크 다항식의 값)
    assert abs(cost - 720.0) / 720.0 < 5e-3


def test_min_jerk_path_waypoints_rest_and_continuity():
    m = rigs()
    times = np.array([0.0, 1.0, 3.0])
    wp = np.array([[0.0, 1.0], [2.0, 1.5], [-1.0, 4.0]])
    x, v, a = m.min_jerk_path(times, wp, times)
    np.testing.assert_allclose(x, wp, atol=1e-12)                  # 웨이포인트 통과
    np.testing.assert_allclose(v, 0.0, atol=1e-12)                 # 각 웨이포인트에서 정지
    np.testing.assert_allclose(a, 0.0, atol=1e-10)
    t = np.linspace(0.0, 3.0, 3001)
    x, v, a = m.min_jerk_path(times, wp, t)
    assert x.shape == v.shape == a.shape == (3001, 2)
    np.testing.assert_allclose(np.gradient(x, t, axis=0)[3:-3], v[3:-3], atol=2e-3)
    xm, vm, _ = m.min_jerk_path(times, wp, np.array([0.5, 2.0]))   # 구간 중간점
    np.testing.assert_allclose(xm, [[1.0, 1.25], [0.5, 2.75]], atol=1e-12)
    np.testing.assert_allclose(vm[0], 15.0 / 8.0 * np.array([2.0, 0.5]) / 1.0, atol=1e-12)
    np.testing.assert_allclose(vm[1], 15.0 / 8.0 * np.array([-3.0, 2.5]) / 2.0, atol=1e-12)
    # 범위 밖은 끝점 유지, 스칼라 t는 (n,) 형태
    xo, vo, ao = m.min_jerk_path(times, wp, np.array([-1.0, 9.0]))
    np.testing.assert_allclose(xo, [wp[0], wp[-1]], atol=1e-12)
    np.testing.assert_allclose(vo, 0.0, atol=1e-12)
    np.testing.assert_allclose(ao, 0.0, atol=1e-12)
    xs, vs, as_ = m.min_jerk_path(times, wp, 0.5)
    assert np.asarray(xs).shape == (2,) and np.asarray(vs).shape == (2,) and np.asarray(as_).shape == (2,)


def test_power_law_speed():
    m = rigs()
    assert abs(float(m.power_law_speed(8.0, 1.0)) - 0.5) < 1e-14         # v = K κ^(-1/3)
    assert abs(float(m.power_law_speed(8.0, 3.0)) - 1.5) < 1e-14
    k = np.array([0.01, 0.1, 1.0, 10.0, 123.0])
    v = np.asarray(m.power_law_speed(k, 2.0))
    np.testing.assert_allclose(v * k, 2.0 * k ** (2.0 / 3.0), rtol=1e-12)  # 각속도 = K κ^(2/3)
    assert np.all(np.diff(v) < 0)                                          # 곡률이 클수록 느리다
    for bad in (0.0, -1.0, np.array([1.0, 0.0])):
        with pytest.raises(ValueError):
            m.power_law_speed(bad, 1.0)


def test_ou_noise_statistics_and_determinism():
    m = rigs()
    n, dt, tau, sigma = 300_000, 1e-3, 0.02, 2.0
    x = np.asarray(m.ou_noise(n, dt, tau, sigma, np.random.default_rng(7)))
    assert x.shape == (n,) and np.isfinite(x).all()
    assert abs(x.var() / sigma**2 - 1.0) < 0.10                    # 정상 분산 σ²
    assert abs(x.mean()) < 0.15
    xc = x - x.mean()
    ac = lambda lag: float(np.dot(xc[:-lag], xc[lag:]) / np.dot(xc, xc))   # noqa: E731
    assert abs(ac(1) - math.exp(-dt / tau)) < 0.01                 # 지수 자기상관 exp(-lag·dt/τ)
    assert abs(ac(20) - math.exp(-1.0)) < 0.04
    assert abs(ac(60) - math.exp(-3.0)) < 0.04
    x2 = m.ou_noise(1000, dt, tau, sigma, np.random.default_rng(7))
    x3 = m.ou_noise(1000, dt, tau, sigma, np.random.default_rng(7))
    x4 = m.ou_noise(1000, dt, tau, sigma, np.random.default_rng(8))
    assert np.array_equal(x2, x3) and not np.array_equal(x2, x4)
    assert not np.any(np.asarray(m.ou_noise(50, dt, tau, 0.0, np.random.default_rng(1))))   # σ=0 이면 0


def test_ou_noise_argument_validation():
    m = rigs()
    g = np.random.default_rng(0)
    for args in ((0, 1e-3, 0.1, 1.0), (10, 0.0, 0.1, 1.0), (10, 1e-3, 0.0, 1.0), (10, 1e-3, -1.0, 1.0),
                 (10, 1e-3, 0.1, -1.0)):
        with pytest.raises(ValueError):
            m.ou_noise(*args, g)


def test_intent_validation_and_reference():
    m = rigs()
    ok_t, ok_w = np.array([0.0, 1.0, 2.0]), np.zeros((3, 2))
    it = m.Intent(ok_t, ok_w, tail_s=0.5)
    assert abs(it.duration_s - 2.5) < 1e-15
    bad = [
        (np.array([0.5, 1.0]), np.zeros((2, 2)), 0.0),            # times[0] != 0
        (np.array([0.0, 1.0, 1.0]), ok_w, 0.0),                    # 엄격 증가 아님
        (np.array([0.0, 2.0, 1.0]), ok_w, 0.0),
        (np.array([0.0]), np.zeros((1, 2)), 0.0),                  # 웨이포인트 2개 미만
        (ok_t, np.zeros((2, 2)), 0.0),                             # 길이 불일치
        (ok_t, np.zeros(3), 0.0),                                  # 2차원 아님
        (ok_t, np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]]), 0.0),
        (ok_t, ok_w, -1.0),                                        # tail_s < 0
    ]
    for t, w, tail in bad:
        with pytest.raises(ValueError):
            m.Intent(t, w, tail_s=tail)
    p2p = m.Intent.point_to_point([0.0, 1.0], [2.0, 3.0], 2.0, tail_s=1.0)
    assert abs(p2p.duration_s - 3.0) < 1e-15
    q, v, a = p2p.reference(np.array([0.0, 1.0, 2.0, 2.5]))
    np.testing.assert_allclose(q, [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0]], atol=1e-12)
    np.testing.assert_allclose(v[[0, 2, 3]], 0.0, atol=1e-12)
