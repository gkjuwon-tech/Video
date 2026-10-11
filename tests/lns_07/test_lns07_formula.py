"""LNS-07 수식 조각: Rayleigh 편광도, Fresnel 반사 편광도, 필터 이득, 노출 보상."""
import math

import numpy as np
import pytest

from buc.lens import cpl
from lns07_helpers import P_MAX, TAU, fresnel_ref, rayleigh_ref

E_PERP_X = np.array([1.0, 0.0, 0.0])
E_PERP_Y = np.array([0.0, 1.0, 0.0])


def grid(dop, evec):
    """(2,3) 화소 격자: 모든 화소 같은 dop·evec."""
    return np.full((2, 3), dop), np.broadcast_to(evec, (2, 3, 3)).copy()


# ------------------------------------------------------------------------------------------------ 상수
def test_constants_are_the_contract_values():
    assert cpl.P_MAX == 0.75 and cpl.TAU == 0.40
    assert 0.0 < cpl.P_MAX < 1.0 and 0.35 <= cpl.TAU <= 0.45  # PARADIGM T≈0.35–0.45


# ------------------------------------------------------------------------------------------------ Rayleigh
def test_rayleigh_dop_matches_oracle_and_special_angles():
    g = np.linspace(0.0, math.pi, 361)
    d = cpl.rayleigh_dop(g)
    assert d.dtype == np.float64 and d.shape == g.shape
    assert np.allclose(d, rayleigh_ref(g), atol=1e-12, rtol=0)
    assert abs(float(cpl.rayleigh_dop(0.0))) < 1e-15 and abs(float(cpl.rayleigh_dop(math.pi))) < 1e-15
    assert float(cpl.rayleigh_dop(math.pi / 2)) == pytest.approx(P_MAX, abs=1e-12)  # 90° 에서 최대
    assert float(cpl.rayleigh_dop(math.pi / 4)) == pytest.approx(P_MAX * 0.5 / 1.5, abs=1e-12)
    assert d.max() <= P_MAX + 1e-12 and d.min() >= 0.0


def test_rayleigh_dop_symmetry_monotone_and_pmax_parameter():
    g = np.linspace(0.0, math.pi / 2, 200)
    d = cpl.rayleigh_dop(g)
    assert np.all(np.diff(d) > 0)  # 0..90° 에서 단조 증가
    assert np.allclose(cpl.rayleigh_dop(math.pi - g), d, atol=1e-12)  # γ <-> π-γ 대칭
    assert np.allclose(cpl.rayleigh_dop(g, 0.5), d * (0.5 / P_MAX), atol=1e-12)
    assert cpl.rayleigh_dop(np.zeros((3, 4, 5))).shape == (3, 4, 5)
    assert np.allclose(cpl.rayleigh_dop(np.array(1.0)), rayleigh_ref(1.0))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_rayleigh_dop_rejects_bad_pmax(bad):
    with pytest.raises(ValueError):
        cpl.rayleigh_dop(1.0, bad)


def test_rayleigh_dop_rejects_nonfinite_gamma():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cpl.rayleigh_dop(bad)


# ------------------------------------------------------------------------------------------------ Fresnel
def test_fresnel_dop_oracle_brewster_and_endpoints():
    a = np.linspace(0.02, math.pi / 2 - 0.02, 120)
    assert np.allclose(cpl.fresnel_dop(a), fresnel_ref(a), atol=1e-9)
    n = 1.33
    brew = math.atan(n)
    assert float(cpl.fresnel_dop(brew)) == pytest.approx(1.0, abs=1e-9)  # 브루스터각: 완전 s편광
    assert float(cpl.fresnel_dop(0.0)) == pytest.approx(0.0, abs=1e-12)  # 수직 입사: 무편광
    assert float(cpl.fresnel_dop(math.pi / 2)) == pytest.approx(0.0, abs=1e-9)  # 스치는 입사: 무편광
    up = cpl.fresnel_dop(np.linspace(0.0, brew, 80))
    dn = cpl.fresnel_dop(np.linspace(brew, math.pi / 2, 80))
    assert np.all(np.diff(up) > 0) and np.all(np.diff(dn) < 0)  # 브루스터각에서 정점
    glass = 1.5
    assert float(cpl.fresnel_dop(math.atan(glass), glass)) == pytest.approx(1.0, abs=1e-9)  # 굴절률 인자
    assert np.allclose(cpl.fresnel_dop(a, glass), fresnel_ref(a, glass), atol=1e-9)


@pytest.mark.parametrize("ang,n", [(-0.1, 1.33), (1.6, 1.33), (float("nan"), 1.33), (0.5, 1.0), (0.5, 0.9),
                                   (0.5, float("nan"))])
def test_fresnel_dop_validation(ang, n):
    with pytest.raises(ValueError):
        cpl.fresnel_dop(ang, n)


# ------------------------------------------------------------------------------------------------ 이득 닫힌 꼴
def test_polarizer_axis_is_x_toward_y():
    assert np.allclose(cpl.polarizer_axis(0.0), (1, 0, 0), atol=1e-15)
    assert np.allclose(cpl.polarizer_axis(90.0), (0, 1, 0), atol=1e-15)
    assert np.allclose(cpl.polarizer_axis(45.0), (math.sqrt(0.5), math.sqrt(0.5), 0), atol=1e-15)
    a = cpl.polarizer_axis(30.0)
    assert a.shape == (3,) and a.dtype == np.float64 and abs(np.linalg.norm(a) - 1) < 1e-15


def test_gain_closed_forms_aligned_crossed_diagonal():
    dop, ev = grid(0.6, E_PERP_Y)  # E벡터가 영상 세로
    aligned = cpl.polarizer_gain(dop, ev, 90.0)
    crossed = cpl.polarizer_gain(dop, ev, 0.0)
    diag = cpl.polarizer_gain(dop, ev, 45.0)
    assert aligned.shape == (2, 3) and aligned.dtype == np.float64
    assert np.allclose(aligned, TAU * 1.6, atol=1e-12)  # τ(1+DoP)
    assert np.allclose(crossed, TAU * 0.4, atol=1e-12)  # τ(1-DoP)
    assert np.allclose(diag, TAU, atol=1e-12)  # 45°: 편광 성분이 정확히 반
    # 완전 편광 + 정렬 = 2τ, 완전 편광 + 직교 = 0 (소광)
    d1, e1 = grid(1.0, E_PERP_X)
    assert np.allclose(cpl.polarizer_gain(d1, e1, 0.0), 2 * TAU, atol=1e-12)
    assert np.allclose(cpl.polarizer_gain(d1, e1, 90.0), 0.0, atol=1e-12)
    # 무편광 -> 각 무관 τ
    d0, e0 = grid(0.0, E_PERP_X)
    for ang in (0.0, 17.0, 90.0, 133.0):
        assert np.allclose(cpl.polarizer_gain(d0, e0, ang), TAU, atol=1e-12)


def test_gain_matches_cos2_law_for_in_plane_evec():
    # E벡터가 영상 평면 안에서 φ 방향이면 이득 = τ(1+DoP cos2(φ-φ_f))
    for phi in (0.0, 20.0, 75.0, 120.0, 179.0):
        e = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi)), 0.0])
        dop, ev = grid(0.7, e)
        for ang in (0.0, 33.0, 90.0, 150.0):
            want = TAU * (1.0 + 0.7 * math.cos(2.0 * math.radians(phi - ang)))
            assert np.allclose(cpl.polarizer_gain(dop, ev, ang), want, atol=1e-12), (phi, ang)


def test_gain_angle_average_orthogonal_sum_and_period():
    rng = np.random.default_rng(5)
    dop = rng.uniform(0.0, 1.0, (4, 5))
    angs = np.arange(0.0, 180.0, 5.0)
    # 축 주기 180° (E벡터가 영상 평면을 벗어나도 성립)
    ev3 = rng.normal(size=(4, 5, 3))
    ev3 /= np.linalg.norm(ev3, axis=-1, keepdims=True)
    for a in (0.0, 37.0, 91.0):
        assert np.allclose(cpl.polarizer_gain(dop, ev3, a), cpl.polarizer_gain(dop, ev3, a + 180.0), atol=1e-12)
        assert np.allclose(cpl.polarizer_gain(dop, ev3, a), cpl.polarizer_gain(dop, ev3, a - 180.0), atol=1e-12)
    # 영상 평면 안 E벡터: 모든 각 평균 = τ, 직교 두 축의 합 = 2((1-s)+sτ)
    th = rng.uniform(0, math.pi, (4, 5))
    evp = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1)
    avg = np.mean([cpl.polarizer_gain(dop, evp, a) for a in angs], axis=0)
    assert np.allclose(avg, TAU, atol=1e-12)
    for s in (1.0, 0.6, 0.25):
        tot = cpl.polarizer_gain(dop, evp, 21.0, strength=s) + cpl.polarizer_gain(dop, evp, 111.0, strength=s)
        assert np.allclose(tot, 2.0 * ((1 - s) + s * TAU), atol=1e-12)
        avg_s = np.mean([cpl.polarizer_gain(dop, evp, a, strength=s) for a in angs], axis=0)
        assert np.allclose(avg_s, (1 - s) + s * TAU, atol=1e-12)


def test_gain_strength_blend_and_tau():
    dop, ev = grid(0.5, E_PERP_Y)
    full = cpl.polarizer_gain(dop, ev, 90.0, strength=1.0)
    assert np.all(cpl.polarizer_gain(dop, ev, 90.0, strength=0.0) == 1.0)  # s=0 은 정확히 1
    assert np.allclose(cpl.polarizer_gain(dop, ev, 90.0, strength=0.25), 0.75 + 0.25 * full, atol=1e-12)
    assert np.allclose(cpl.polarizer_gain(dop, ev, 90.0, tau=0.5), 0.5 * 1.5, atol=1e-12)
    assert np.allclose(cpl.polarizer_gain(dop, ev, 90.0), full)  # 기본 tau=TAU, strength=1


def test_gain_bounds_for_unit_evec_and_sky_dop():
    rng = np.random.default_rng(11)
    dop = rng.uniform(0.0, P_MAX, (30, 40))
    ev = rng.normal(size=(30, 40, 3))
    ev /= np.linalg.norm(ev, axis=-1, keepdims=True)
    for ang in (0.0, 40.0, 90.0, 170.0):
        g = cpl.polarizer_gain(dop, ev, ang)
        assert g.min() >= TAU * (1 - P_MAX) - 1e-12 and g.max() <= TAU * (1 + P_MAX) + 1e-12


def test_gain_validation():
    dop, ev = grid(0.5, E_PERP_Y)
    for kw in ({"strength": -0.01}, {"strength": 1.01}, {"strength": float("nan")}, {"tau": 0.0}, {"tau": 1.01},
               {"tau": float("nan")}):
        with pytest.raises(ValueError):
            cpl.polarizer_gain(dop, ev, 10.0, **kw)
    with pytest.raises(ValueError):
        cpl.polarizer_gain(dop * 0 + 1.2, ev, 10.0)  # dop > 1
    with pytest.raises(ValueError):
        cpl.polarizer_gain(dop * 0 - 0.1, ev, 10.0)  # dop < 0
    with pytest.raises(ValueError):
        cpl.polarizer_gain(dop, ev[:, :2], 10.0)  # 모양 불일치
    with pytest.raises(ValueError):
        cpl.polarizer_gain(dop, ev, float("nan"))
    with pytest.raises(TypeError):
        cpl.polarizer_gain(dop, ev, 10.0, 1.0)  # strength 는 키워드 전용


# ------------------------------------------------------------------------------------------------ 노출 보상
def test_compensation_ev():
    assert cpl.compensation_ev(0.0) == 0.0
    assert cpl.compensation_ev(1.0) == pytest.approx(-math.log2(TAU), abs=1e-12)  # ≈ 1.3219 EV
    assert 1.2 < cpl.compensation_ev(1.0) < 1.6
    prev = 0.0
    for s in (0.1, 0.3, 0.5, 0.8, 1.0):
        v = cpl.compensation_ev(s)
        assert v == pytest.approx(-math.log2((1 - s) + s * TAU), abs=1e-12) and v > prev
        prev = v
    assert cpl.compensation_ev(1.0, tau=0.5) == pytest.approx(1.0, abs=1e-12)
    # 비편광 하늘 이득의 역수와 정확히 맞는다
    dop, ev = grid(0.0, E_PERP_X)
    g = float(cpl.polarizer_gain(dop, ev, 12.0, strength=0.7)[0, 0])
    assert 2.0 ** cpl.compensation_ev(0.7) * g == pytest.approx(1.0, abs=1e-12)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            cpl.compensation_ev(bad)
    with pytest.raises(ValueError):
        cpl.compensation_ev(0.5, tau=0.0)
