"""LNS-06 출사동 커널: Prescription 검증, 닫힌형 개구율, exit_pupil_kernel (cat-eye 위에 스월·애너모픽 스퀴즈)."""
import dataclasses
import math

import numpy as np
import pytest

from buc.lens import optics
from buc.lens import prescriptions as P
from lns06_helpers import (ang_diff180, catseye_area_frac, centroid, cov2, minor_axis_deg, ring_mass, second_moment,
                           vertex_phase_deg)


def mk(cat=0.0, blades=0, rot=0.0, **kw):
    return P.Prescription(optics=optics.OpticsSpec(catseye=cat, blades=blades, blade_rot_deg=rot), **kw)


# ----------------------------------------------------------------------------------------------- 표면·상수
def test_public_surface_and_constants():
    for name in ("SWIRL_MAX_TWIST_DEG", "EDGE_SIGMA_FRAC", "EDGE_R_INNER", "SQUEEZE_MAX", "DEFAULT_STREAK_TINT",
                 "RX_TABLE", "Prescription", "pupil_open_fraction", "exit_pupil_kernel", "edge_softness",
                 "streak_layer", "apply_prescription_bokeh", "prescription_from_theta", "apply_prescription",
                 "apply_prescription_frames"):
        assert hasattr(P, name), name
        assert name in P.__all__, name
    assert P.SWIRL_MAX_TWIST_DEG == 60.0
    assert P.EDGE_SIGMA_FRAC == 0.015
    assert P.EDGE_R_INNER == 0.35
    assert P.SQUEEZE_MAX == 3.0
    assert tuple(P.DEFAULT_STREAK_TINT) == (0.55, 0.75, 1.0)


# ----------------------------------------------------------------------------------------------- Prescription
def test_prescription_defaults_frozen_and_properties():
    rx = P.Prescription()
    assert rx.optics == optics.OpticsSpec()
    assert (rx.squeeze, rx.swirl, rx.edge_blur, rx.streak) == (1.0, 0.0, 0.0, 0.0)
    assert rx.streak_len_frac == 0.35 and rx.streak_threshold == 0.8 and rx.name == "custom"
    assert tuple(rx.streak_tint) == (0.55, 0.75, 1.0)
    assert rx == P.Prescription()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rx.swirl = 0.5
    with pytest.raises(TypeError):
        P.Prescription(optics.OpticsSpec())  # 키워드 전용
    rx2 = mk(cat=0.6, blades=7, rot=11.0, streak_tint=[1, 0.5, 0.25])
    assert (rx2.catseye, rx2.blades, rx2.blade_rot_deg) == (0.6, 7, 11.0)  # optics 의 읽기 전용 속성
    assert isinstance(rx2.streak_tint, tuple) and rx2.streak_tint == (1.0, 0.5, 0.25)
    rx3 = P.Prescription(squeeze=2, swirl=1, edge_blur=0, name="x")
    assert isinstance(rx3.squeeze, float) and rx3.squeeze == 2.0


@pytest.mark.parametrize("kw", [
    dict(squeeze=0.99), dict(squeeze=3.01), dict(squeeze=float("nan")), dict(squeeze=True), dict(squeeze="2"),
    dict(swirl=-0.01), dict(swirl=1.01), dict(swirl=float("inf")), dict(edge_blur=-0.1), dict(edge_blur=1.5),
    dict(streak=-1.0), dict(streak=1.2), dict(streak_len_frac=0.0), dict(streak_len_frac=1.01),
    dict(streak_tint=(1.0, 1.0)), dict(streak_tint=(1.0, -0.1, 1.0)), dict(streak_tint=(1.0, float("nan"), 1.0)),
    dict(streak_threshold=-0.5), dict(name=""), dict(name=3),
])
def test_prescription_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        P.Prescription(**kw)


def test_prescription_optics_type():
    with pytest.raises(TypeError):
        P.Prescription(optics={"k1": 0.0})
    with pytest.raises(TypeError):
        P.Prescription(optics=None)


# ----------------------------------------------------------------------------------------------- 개구율
def test_pupil_open_fraction_closed_form():
    assert P.pupil_open_fraction(0.0, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert P.pupil_open_fraction(1.0, 0.0) == pytest.approx(1.0, abs=1e-12)
    for r, c in [(0.3, 0.5), (0.5, 1.0), (1.0, 0.7), (1.0, 1.0), (0.9, 0.9)]:
        assert P.pupil_open_fraction(r, c) == pytest.approx(catseye_area_frac(r * c), abs=1e-9)
    assert P.pupil_open_fraction(1.0, 1.0) == pytest.approx(0.3910, abs=1e-3)
    vals = [P.pupil_open_fraction(r, 1.0) for r in np.linspace(0, 1, 11)]
    assert all(b < a for a, b in zip(vals, vals[1:]))  # 가장자리일수록 좁아진다
    vals = [P.pupil_open_fraction(0.8, c) for c in np.linspace(0, 1, 11)]
    assert all(b < a for a, b in zip(vals, vals[1:]))
    assert isinstance(P.pupil_open_fraction(0.5, 0.5), float)
    for bad in [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1), (float("nan"), 0.5), (True, 0.5)]:
        with pytest.raises(ValueError):
            P.pupil_open_fraction(*bad)


@pytest.mark.parametrize("u", [0.3, 0.7, 1.0])
def test_pupil_open_fraction_matches_kernel_area(u):
    # 평평한 커널의 넓이 = 1/최댓값 (완전 덮인 화소의 값). 개구율×원 넓이와 2% 안
    d = 60.0
    k = P.exit_pupil_kernel(d, u, 30.0, mk(cat=1.0)).astype(np.float64)
    area = 1.0 / k.max()
    assert area == pytest.approx(P.pupil_open_fraction(u, 1.0) * math.pi * (d / 2) ** 2, rel=0.02)


# ----------------------------------------------------------------------------------------------- 커널: 이음매
@pytest.mark.parametrize("d,r,ang,cat,blades,rot", [
    (8.0, 0.5, 0.0, 0.0, 0, 0.0), (12.5, 1.0, 90.0, 1.0, 0, 0.0), (20.0, 0.7, 37.0, 0.6, 6, 13.0),
    (31.0, 0.3, 120.0, 0.9, 5, 0.0), (5.5, 0.9, 45.0, 0.5, 3, 7.0),
])
def test_kernel_equals_catseye_when_neutral(d, r, ang, cat, blades, rot):
    rx = mk(cat=cat, blades=blades, rot=rot)
    want = optics.catseye_kernel(d, r, ang, cat, blades, rot)
    got = P.exit_pupil_kernel(d, r, ang, rx)
    assert got.dtype == np.float32 and np.array_equal(got, want)
    # r=0 이면 스월 (twist ∝ r) 도 영향이 없다
    sw = mk(cat=cat, blades=blades, rot=rot, swirl=1.0)
    assert np.array_equal(P.exit_pupil_kernel(d, 0.0, ang, sw), optics.catseye_kernel(d, 0.0, ang, cat, blades, rot))
    # 스월 0, 스퀴즈 1 이면 r 이 얼마든 cat-eye 와 같다
    assert np.array_equal(P.exit_pupil_kernel(d, r, ang, mk(cat=cat, blades=blades, rot=rot, swirl=0.0)), want)


def test_kernel_tiny_diameter_is_delta():
    for rx in (mk(), mk(cat=1.0, swirl=1.0, squeeze=2.0)):
        for d in (0.0, 0.5, 1.0):
            k = P.exit_pupil_kernel(d, 0.8, 30.0, rx)
            assert k.shape == (1, 1) and k.dtype == np.float32 and k[0, 0] == 1.0


def test_kernel_returns_fresh_array():
    rx = mk(cat=0.8, swirl=0.5, squeeze=1.5)
    a = P.exit_pupil_kernel(10.0, 0.6, 20.0, rx)
    ref = a.copy()
    a[:] = 7.0
    assert np.array_equal(P.exit_pupil_kernel(10.0, 0.6, 20.0, rx), ref)


# ----------------------------------------------------------------------------------------------- 커널: 불변식 격자
@pytest.mark.parametrize("d", [6.5, 17.0, 40.0])
@pytest.mark.parametrize("blades", [0, 5])
def test_kernel_invariants_grid(d, blades):
    for squeeze in (1.0, 1.5, 2.0):
        for swirl in (0.0, 0.5, 1.0):
            for r in (0.3, 1.0):
                for ang in (0.0, 37.0, 90.0):
                    rx = mk(cat=0.8, blades=blades, swirl=swirl, squeeze=squeeze)
                    k = P.exit_pupil_kernel(d, r, ang, rx)
                    K = 2 * math.ceil(d * math.sqrt(squeeze) / 2.0) + 1
                    assert k.shape == (K, K) and k.dtype == np.float32
                    assert np.isfinite(k).all() and k.min() >= 0.0
                    assert float(k.sum(dtype=np.float64)) == pytest.approx(1.0, abs=1e-5)
                    cx, cy = centroid(k)
                    mid = (K - 1) / 2.0
                    tol = 0.1 + 0.005 * d  # LNS-05 다각형 cat-eye 자체의 중심 오차 포함
                    assert abs(cx - mid) < tol and abs(cy - mid) < tol
                    assert ring_mass(k) < (0.01 if d >= 10 else 0.1)  # 모양이 배열 안에 들어간다


# ----------------------------------------------------------------------------------------------- 스월
def test_swirl_rotates_cateye_axis_in_increasing_angle_direction():
    # 카메라 축 위 (angle 0) 의 cat-eye 렌즈는 x 방향으로 좁고 y 로 길다 -> 작은 고유축 0°. 스월은 각도 증가 방향으로 비튼다.
    expect = {0.0: (0.0, 1.0), 0.3: (9.0, 15.5), 0.6: (18.5, 30.5), 1.0: (32.0, 49.0)}  # 도 (원형 오차 ±25%)
    got = {}
    for sw, (lo, hi) in expect.items():
        k = P.exit_pupil_kernel(40.0, 1.0, 0.0, mk(cat=1.0, swirl=sw))
        a = minor_axis_deg(cov2(k))
        got[sw] = a
        assert lo <= a <= hi, (sw, a)
    assert got[0.3] < got[0.6] < got[1.0]


def test_swirl_twist_scales_with_field_radius():
    # 삼각형 조리개 (cat-eye 0): 꼭짓점 방향 위상이 τ = swirl·60°·r_frac 만큼 (바깥 고리 평균이라 약 0.75–1.0 배) 돈다.
    base = P.exit_pupil_kernel(48.0, 1.0, 0.0, mk(blades=3))
    ph0 = vertex_phase_deg(base, 3)
    ph = {}
    for r in (0.5, 1.0):
        k = P.exit_pupil_kernel(48.0, r, 0.0, mk(blades=3, swirl=1.0))
        ph[r] = (vertex_phase_deg(k, 3) - ph0 + 60.0) % 120.0 - 60.0
    assert 0.75 * 30.0 <= ph[0.5] <= 1.02 * 30.0
    assert ph[0.5] > 0.0  # 각도 증가 방향
    k_half = P.exit_pupil_kernel(48.0, 0.25, 0.0, mk(blades=3, swirl=1.0))
    ph_q = (vertex_phase_deg(k_half, 3) - ph0 + 60.0) % 120.0 - 60.0
    assert 0.75 * 15.0 <= ph_q <= 1.02 * 15.0
    assert ph_q < ph[0.5]


def test_swirl_preserves_mass_second_moment_and_centroid():
    for blades in (0, 6):
        plain = P.exit_pupil_kernel(40.0, 1.0, 0.0, mk(cat=1.0, blades=blades))
        sw = P.exit_pupil_kernel(40.0, 1.0, 0.0, mk(cat=1.0, blades=blades, swirl=1.0))
        assert not np.allclose(plain, sw, atol=1e-4)
        assert float(sw.sum(dtype=np.float64)) == pytest.approx(1.0, abs=1e-5)
        assert second_moment(sw) == pytest.approx(second_moment(plain), rel=0.02)  # 스월은 면적·반경 분포를 보존
        cx, cy = centroid(sw)
        assert abs(cx - 20.0) < 0.05 and abs(cy - 20.0) < 0.05


# ----------------------------------------------------------------------------------------------- 애너모픽 스퀴즈
@pytest.mark.parametrize("s", [1.5, 2.0, 3.0])
def test_squeeze_makes_tall_ellipse_preserving_geometric_mean(s):
    base = cov2(optics.catseye_kernel(32.0, 0.0, 0.0, 0.0))
    k = P.exit_pupil_kernel(32.0, 0.0, 0.0, mk(squeeze=s))
    c = cov2(k)
    assert abs(c[0, 1]) < 0.01 * c[0, 0]
    assert c[1, 1] / c[0, 0] == pytest.approx(s * s, rel=0.04)      # 세로로 s 배 길다 (분산비 s²)
    assert c[0, 0] * c[1, 1] == pytest.approx(base[0, 0] * base[1, 1], rel=0.03)  # 넓이(기하평균) 보존
    assert c[0, 0] == pytest.approx(base[0, 0] / s, rel=0.04)
    assert ring_mass(k) < 0.005


def test_squeeze_with_polygon_and_cateye_keeps_unit_mass_and_axes():
    rx = mk(cat=0.8, blades=6, squeeze=2.0)
    k = P.exit_pupil_kernel(30.0, 0.8, 0.0, rx)
    c = cov2(k)
    assert c[1, 1] > 2.0 * c[0, 0]
    assert float(k.sum(dtype=np.float64)) == pytest.approx(1.0, abs=1e-5)
    assert ang_diff180(minor_axis_deg(c), 0.0) < 2.0


# ----------------------------------------------------------------------------------------------- 오류
def test_kernel_validation():
    rx = mk()
    for args in [(-1.0, 0.5, 0.0), (float("nan"), 0.5, 0.0), (10.0, -0.1, 0.0), (10.0, 1.1, 0.0),
                 (10.0, float("nan"), 0.0), (10.0, 0.5, float("nan")), (True, 0.5, 0.0), (10.0, True, 0.0)]:
        with pytest.raises(ValueError):
            P.exit_pupil_kernel(*args, rx)
    for bad in (None, optics.OpticsSpec(), {"swirl": 0.5}):
        with pytest.raises(TypeError):
            P.exit_pupil_kernel(10.0, 0.5, 0.0, bad)
