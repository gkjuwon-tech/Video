"""LNS-07 적용: 독립 오라클 대조, 고전적 CPL 거동, 마스크·태양 없음·항등, 선형성, 롤 동등성, LNS-04 이음매."""
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from buc.core import color
from buc.lens import cpl
from lns07_helpers import NORTH, P_MAX, TAU, UP, cam_R_w2c, fresnel_ref, gain_ref, sky_scene, theta

REPO = Path(__file__).resolve().parents[2]
KW = dict(up=UP, north=NORTH)


def run(x, th, R, **kw):
    return cpl.apply_cpl(x, th, R_w2c=R, **{**KW, **kw})


def flat(h, w, rgb=(0.2, 0.4, 0.8), dtype=np.float32):
    return np.broadcast_to(np.asarray(rgb, dtype), (h, w, 3)).copy()


# ------------------------------------------------------------------------------------------------ 항등·복사
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cpl_zero_is_exact_copy_for_both_dtypes(dtype):
    x = sky_scene().astype(dtype)
    keep = x.copy()
    th = theta(cpl=0.0, sun_elev_deg=10.0, sun_azim_deg=90.0, extra={"cpl_angle_deg": 33.0})
    y = run(x, th, cam_R_w2c(0.0))
    assert np.array_equal(x, keep)  # 입력 불변
    assert y.dtype == dtype and y.shape == x.shape and y.tobytes() == x.tobytes()
    assert y is not x and not np.shares_memory(x, y)  # 새 배열


# ------------------------------------------------------------------------------------------------ 고전적 CPL 거동
def test_classic_blue_sky_darkening_numbers():
    # 북쪽을 보는 수평 카메라, 태양은 동쪽 지평선(γ=90° at 중앙). 망원이라 화면 전체가 거의 같은 dop.
    h, w = 9, 9
    x = flat(h, w)
    R = cam_R_w2c(0.0)
    c = (h // 2, w // 2)
    center = {}
    for ang in (0.0, 45.0, 90.0, 135.0):
        th = theta(cpl=1.0, focal_mm=2000.0, sensor_width_mm=24.89, sun_elev_deg=0.0, sun_azim_deg=90.0,
                   extra={"cpl_angle_deg": ang})
        y = run(x, th, R)
        center[ang] = y[c] / x[c]
        assert np.allclose(center[ang], center[ang][0], rtol=1e-6)  # 중성 필터: 세 채널 같은 이득
    assert center[90.0][0] == pytest.approx(TAU * (1 + P_MAX), rel=1e-5)  # E벡터(세로)와 정렬: 0.70
    assert center[0.0][0] == pytest.approx(TAU * (1 - P_MAX), rel=1e-5)  # 직교: 0.10 -> 하늘이 짙어진다
    assert center[45.0][0] == pytest.approx(TAU, rel=1e-5)
    assert center[135.0][0] == pytest.approx(TAU, rel=1e-5)
    assert center[90.0][0] / center[0.0][0] == pytest.approx((1 + P_MAX) / (1 - P_MAX), rel=1e-5)  # 7 배


def test_sky_vs_cloud_contrast_rises_when_filter_crosses_the_sky_polarization():
    h, w = 8, 16
    x = flat(h, w, (0.5, 0.5, 0.5))
    mask = np.zeros((h, w), bool)
    mask[:, : w // 2] = True  # 왼쪽 반이 하늘, 오른쪽 반은 구름(무편광)
    R = cam_R_w2c(0.0)
    ratios = {}
    for ang in (0.0, 90.0):
        th = theta(cpl=1.0, focal_mm=2000.0, sun_elev_deg=0.0, sun_azim_deg=90.0, extra={"cpl_angle_deg": ang})
        y = run(x, th, R, sky_mask=mask)
        assert np.allclose(y[:, w // 2 :], x[:, w // 2 :] * TAU, rtol=1e-6)  # 구름: 각 무관 τ
        ratios[ang] = float(y[0, 0, 0] / y[0, w - 1, 0])
    # 모서리 화소는 광축에서 아주 조금 벗어나 있어 dop·E벡터가 ~1e-3 달라진다
    assert ratios[0.0] == pytest.approx(1 - P_MAX, rel=2e-3) and ratios[90.0] == pytest.approx(1 + P_MAX, rel=2e-3)


def test_darkest_sky_is_ninety_degrees_from_sun():
    # 광각 카메라, 필터 축은 90°(세로). 태양 방위가 시선과 90° 일 때가 90° 아닐 때보다 중앙 dop 가 크다.
    h, w = 21, 21
    x = flat(h, w, (1.0, 1.0, 1.0))
    R = cam_R_w2c(0.0)
    c = (h // 2, w // 2)
    spread = {}
    for azim in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0):
        th = theta(cpl=1.0, focal_mm=50.0, sun_elev_deg=0.0, sun_azim_deg=azim, extra={"cpl_angle_deg": 0.0})
        spread[azim] = float(run(x, th, R)[c][0])  # 축 0°: 직교 -> 이득 τ(1-dop)
    assert spread[90.0] == min(spread.values())
    assert spread[0.0] == pytest.approx(TAU, rel=1e-6) and spread[180.0] == pytest.approx(TAU, rel=1e-6)  # 태양 앞/뒤: 무편광
    assert spread[30.0] > spread[60.0] > spread[90.0] < spread[120.0] < spread[150.0]


# ------------------------------------------------------------------------------------------------ 전 화면 독립 오라클
CASES = [
    # (heading, pitch, roll, focal_mm, sun_elev, sun_azim, cpl, angle, mask)
    (0.0, 0.0, 0.0, 24.0, 5.0, 90.0, 1.0, 0.0, None),
    (0.0, 0.0, 0.0, 24.0, 5.0, 90.0, 1.0, 90.0, None),
    (35.0, 12.0, 0.0, 14.0, 40.0, 10.0, 0.8, 77.0, None),
    (200.0, -10.0, 7.0, 35.0, 12.0, 300.0, 0.5, 130.0, "soft"),
    (90.0, 25.0, -20.0, 50.0, 60.0, 200.0, 1.0, 20.0, "bool"),
    (310.0, 0.0, 90.0, 28.0, -3.0, 45.0, 0.35, 179.0, None),
    (0.0, 80.0, 0.0, 20.0, 85.0, 0.0, 1.0, 55.0, None),  # 거의 천정을 보며 태양도 높다
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_frame_matches_independent_oracle(case, dtype):
    heading, pitch, roll, focal, e, a, s, ang, mk = case
    h, w = 40, 56
    rng = np.random.default_rng(1)
    mask = None
    if mk == "bool":
        mask = rng.random((h, w)) > 0.5
    elif mk == "soft":
        mask = rng.random((h, w))
    th = theta(cpl=s, focal_mm=focal, sun_elev_deg=e, sun_azim_deg=a, extra={"cpl_angle_deg": ang})
    x = sky_scene(h, w).astype(dtype)
    R = cam_R_w2c(heading, pitch, roll)
    y = run(x, th, R, sky_mask=mask)
    want = x.astype(np.float64) * gain_ref(th, h, w, R, mask=mask)[..., None]
    assert y.dtype == dtype and y.shape == x.shape
    tol = 1e-6 if dtype == np.float32 else 1e-10
    assert np.allclose(y, want, rtol=tol, atol=tol * 1e-3)
    # 밝은 HDR 점(>1)은 잘리지 않고 같은 이득을 받는다
    assert np.isfinite(y).all() and y.min() >= 0.0


def test_default_k_and_explicit_k_agree_and_mask_none_equals_all_ones():
    h, w = 24, 40
    th = theta(cpl=1.0, focal_mm=28.0, sun_elev_deg=20.0, sun_azim_deg=70.0, extra={"cpl_angle_deg": 40.0})
    x = sky_scene(h, w)
    R = cam_R_w2c(20.0, 5.0)
    a = run(x, th, R)
    b = run(x, th, R, K=th.intrinsics(w, h))
    c = run(x, th, R, sky_mask=np.ones((h, w), bool))
    d = run(x, th, R, sky_mask=np.ones((h, w), np.float32))
    assert a.tobytes() == b.tobytes() == c.tobytes() == d.tobytes()
    # 다른 K(더 좁은 화각)는 다른 결과
    K2 = th.intrinsics(w, h) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 1]])
    K2[2] = (0, 0, 1)
    assert run(x, th, R, K=K2).tobytes() != a.tobytes()


def test_mask_semantics_zero_is_unpolarized_and_soft_scales_dop():
    h, w = 10, 12
    x = flat(h, w, (0.3, 0.3, 0.3), np.float64)
    R = cam_R_w2c(0.0)
    th = theta(cpl=1.0, focal_mm=2000.0, sun_elev_deg=0.0, sun_azim_deg=90.0, extra={"cpl_angle_deg": 90.0})
    full = run(x, th, R)[5, 6, 0] / 0.3
    zero = run(x, th, R, sky_mask=np.zeros((h, w), bool))
    half = run(x, th, R, sky_mask=np.full((h, w), 0.5))
    assert np.allclose(zero, 0.3 * TAU, rtol=1e-12)
    assert half[5, 6, 0] / 0.3 == pytest.approx(TAU * (1 + 0.5 * P_MAX), rel=1e-3)  # 중앙(광축) 화소
    assert zero[5, 6, 0] / 0.3 < half[5, 6, 0] / 0.3 < full  # dop 가 클수록 정렬 축에서 더 밝다


def test_no_sun_gives_uniform_unpolarized_gain():
    h, w = 12, 20
    x = sky_scene(h, w)
    R = cam_R_w2c(33.0, 4.0)
    for s in (1.0, 0.4):
        want = (1 - s) + s * TAU
        for kw in ({}, {"sun_elev_deg": 30.0}, {"sun_azim_deg": 120.0}):  # 하나만 있어도 태양 없음
            th = theta(cpl=s, extra={"cpl_angle_deg": 66.0}, **kw)
            y = run(x, th, R)
            assert np.allclose(y, x.astype(np.float64) * want, rtol=1e-6)
    # 각이 달라도 같다
    a = run(x, theta(cpl=1.0, extra={"cpl_angle_deg": 0.0}), R)
    b = run(x, theta(cpl=1.0, extra={"cpl_angle_deg": 90.0}), R)
    assert a.tobytes() == b.tobytes()


def test_default_angle_zero_and_period_180():
    h, w = 20, 30
    x = sky_scene(h, w)
    R = cam_R_w2c(10.0, 3.0)
    base = dict(cpl=1.0, focal_mm=24.0, sun_elev_deg=15.0, sun_azim_deg=80.0)
    no_key = run(x, theta(**base), R)
    zero = run(x, theta(extra={"cpl_angle_deg": 0.0}, **base), R)
    half_turn = run(x, theta(extra={"cpl_angle_deg": 180.0}, **base), R)
    assert no_key.tobytes() == zero.tobytes()  # extra 누락 = 0°
    assert np.allclose(half_turn, zero, rtol=1e-6)
    other = run(x, theta(extra={"cpl_angle_deg": 90.0}, **base), R)
    assert not np.allclose(other, zero, rtol=1e-3)


def test_works_with_lns01_default_and_sampled_thetas():
    from buc.lens import theta as T

    x = sky_scene(16, 24)
    R = cam_R_w2c(0.0)
    d = T.default_theta()  # cpl=0, 태양 없음
    assert run(x, d, R).tobytes() == x.tobytes()
    for th in T.sample_thetas(25, 7):
        y = run(x, th, R)
        assert y.dtype == np.float32 and np.isfinite(y).all() and y.min() >= 0.0
        s = th.cpl
        hi = (1 - s) + s * TAU * (1 + P_MAX)
        lo = (1 - s) + s * TAU * (1 - P_MAX)
        ratio = y.astype(np.float64) / np.maximum(x.astype(np.float64), 1e-12)
        assert ratio.min() >= lo - 1e-5 and ratio.max() <= hi + 1e-5


# ------------------------------------------------------------------------------------------------ 불변식
def test_linearity_hdr_preserved_and_channel_neutral():
    h, w = 16, 20
    th = theta(cpl=0.9, focal_mm=24.0, sun_elev_deg=25.0, sun_azim_deg=110.0, extra={"cpl_angle_deg": 50.0})
    R = cam_R_w2c(15.0, 4.0, 3.0)
    x = sky_scene(h, w).astype(np.float64)
    y1 = run(x, th, R)
    assert np.allclose(run(3.0 * x, th, R), 3.0 * y1, rtol=1e-12)  # 동차성
    z = np.random.default_rng(2).random((h, w, 3))
    assert np.allclose(run(x + z, th, R), y1 + run(z, th, R), rtol=1e-12, atol=1e-14)  # 가법성
    g = y1 / x
    assert np.allclose(g[..., 0], g[..., 1], rtol=1e-12) and np.allclose(g[..., 1], g[..., 2], rtol=1e-12)
    big = np.full((h, w, 3), 500.0, np.float32)
    yb = run(big, th, R)
    assert yb.max() > 1.0 and np.allclose(yb, 500.0 * g[..., :1].astype(np.float32) * np.ones(3), rtol=2e-6)
    zero = run(np.zeros((h, w, 3), np.float32), th, R)
    assert np.array_equal(zero, np.zeros((h, w, 3), np.float32))


def test_gain_bounds_over_a_wide_lens_at_full_strength():
    h, w = 30, 60
    x = flat(h, w, (1.0, 1.0, 1.0), np.float64)
    for ang in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0):
        th = theta(cpl=1.0, focal_mm=10.0, sun_elev_deg=30.0, sun_azim_deg=60.0, extra={"cpl_angle_deg": ang})
        g = run(x, th, cam_R_w2c(0.0, 10.0))[..., 0]
        assert g.min() >= TAU * (1 - P_MAX) - 1e-12 and g.max() <= TAU * (1 + P_MAX) + 1e-12


def test_orthogonal_axes_sum_for_narrow_lens_and_angle_average_for_on_axis_pixels():
    # 망원: E벡터가 거의 영상 평면 안 -> 직교한 두 축 이득의 합이 비편광 이득의 2 배
    h, w = 7, 7
    x = flat(h, w, (1.0, 1.0, 1.0), np.float64)
    R = cam_R_w2c(0.0, 0.0, 12.0)
    for s in (1.0, 0.6):
        g = lambda ang: run(x, theta(cpl=s, focal_mm=2000.0, sun_elev_deg=0.0, sun_azim_deg=90.0,  # noqa: E731
                                     extra={"cpl_angle_deg": ang}), R)[3, 3, 0]
        assert g(25.0) + g(115.0) == pytest.approx(2 * ((1 - s) + s * TAU), rel=1e-6)
        avg = np.mean([g(a) for a in np.arange(0.0, 180.0, 3.0)])
        assert avg == pytest.approx((1 - s) + s * TAU, rel=1e-6)


def test_camera_roll_equals_rotating_the_filter():
    # 카메라를 광축으로 α(x 에서 y 쪽이 양) 굴리고 필터 축 φ = 굴리지 않고 축을 φ+α 로 돌린 것 (중앙 화소, 망원)
    h, w = 5, 5
    x = flat(h, w, (1.0, 1.0, 1.0), np.float64)
    base = dict(cpl=1.0, focal_mm=2000.0, sun_elev_deg=10.0, sun_azim_deg=75.0)
    for alpha in (0.0, 17.0, 60.0, 130.0):
        for ang in (0.0, 40.0, 100.0):
            rolled = run(x, theta(extra={"cpl_angle_deg": ang}, **base), cam_R_w2c(0.0, 0.0, alpha))[2, 2, 0]
            turned = run(x, theta(extra={"cpl_angle_deg": ang + alpha}, **base), cam_R_w2c(0.0, 0.0, 0.0))[2, 2, 0]
            assert rolled == pytest.approx(turned, rel=1e-6), (alpha, ang)


def test_camera_motion_moves_the_pattern_but_frames_are_deterministic():
    x = sky_scene(20, 30)
    th = theta(cpl=1.0, focal_mm=24.0, sun_elev_deg=10.0, sun_azim_deg=90.0, extra={"cpl_angle_deg": 90.0})
    a = run(x, th, cam_R_w2c(0.0))
    b = run(x, th, cam_R_w2c(0.0))
    c = run(x, th, cam_R_w2c(25.0))
    assert a.tobytes() == b.tobytes() and a.tobytes() != c.tobytes()


def test_no_warnings_on_degenerate_geometry():
    # 태양이 정확히 시선 위/뒤, 천정, 지평: 0 나눗셈·NaN·경고 금지
    x = sky_scene(12, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for e, a, heading in ((0.0, 0.0, 0.0), (0.0, 180.0, 0.0), (90.0, 0.0, 0.0), (-20.0, 359.9, 0.0)):
            y = run(x, theta(cpl=1.0, focal_mm=40.0, sun_elev_deg=e, sun_azim_deg=a), cam_R_w2c(heading))
            assert np.isfinite(y).all()
        y = run(x, theta(cpl=1.0, focal_mm=40.0, sun_elev_deg=90.0, sun_azim_deg=0.0), cam_R_w2c(0.0, 90.0 - 1e-9))
        assert np.isfinite(y).all()


# ------------------------------------------------------------------------------------------------ 일반 진입점 (반사)
def test_apply_polarizer_water_reflection_extinction_at_brewster():
    h, w = 6, 8
    n = 1.33
    dop = np.full((h, w), float(cpl.fresnel_dop(np.arctan(n), n)))
    assert dop[0, 0] == pytest.approx(1.0, abs=1e-9)
    ev = np.zeros((h, w, 3))
    ev[..., 0] = 1.0  # s편광 E벡터가 영상 가로
    x = flat(h, w, (0.8, 0.9, 1.0), np.float64)
    killed = cpl.apply_polarizer(x, np.minimum(dop, 1.0), ev, 90.0)  # 축 세로 = E벡터에 직교
    passed = cpl.apply_polarizer(x, np.minimum(dop, 1.0), ev, 0.0)
    assert np.max(killed) < 1e-8  # 브루스터각 반사는 소광
    assert np.allclose(passed, x * 2 * TAU, rtol=1e-8)
    # 보통 입사각(30°)은 일부만 줄어든다: 이득 = τ(1-dop) ~ fresnel_ref
    d30 = float(fresnel_ref(np.radians(30.0)))
    part = cpl.apply_polarizer(x, np.full((h, w), d30), ev, 90.0)
    assert np.allclose(part, x * TAU * (1 - d30), rtol=1e-6)


def test_apply_polarizer_matches_apply_cpl_when_given_sky_polarization():
    h, w = 18, 26
    th = theta(cpl=0.7, focal_mm=22.0, sun_elev_deg=18.0, sun_azim_deg=140.0, extra={"cpl_angle_deg": 65.0})
    R = cam_R_w2c(40.0, 6.0)
    x = sky_scene(h, w)
    rays = cpl.view_rays(th.intrinsics(w, h), h, w)
    sun_cam = R @ cpl.sun_dir_world(18.0, 140.0, **KW)
    dop, ev = cpl.sky_polarization(rays, sun_cam)
    a = cpl.apply_polarizer(x, dop, ev, 65.0, strength=0.7)
    b = run(x, th, R)
    assert np.allclose(a, b, rtol=1e-6) and a.dtype == np.float32
    assert np.allclose(b, x.astype(np.float64) * gain_ref(th, h, w, R)[..., None], rtol=1e-6)


# ------------------------------------------------------------------------------------------------ 오류 경로
def test_validation_errors():
    th = theta(cpl=1.0, sun_elev_deg=10.0, sun_azim_deg=10.0)
    R = cam_R_w2c(0.0)
    x = sky_scene(8, 10)
    with pytest.raises(TypeError):
        run(x.astype(np.int32), th, R)
    with pytest.raises(TypeError):
        run((x * 255).astype(np.uint8), th, R)
    with pytest.raises(ValueError):
        run(np.zeros((8, 10, 4), np.float32), th, R)
    with pytest.raises(ValueError):
        run(np.zeros((8, 10), np.float32), th, R)
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        run(bad, th, R)
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        run(bad, th, R)
    with pytest.raises(TypeError):
        run(x, {"cpl": 1.0}, R)
    for badR in (np.eye(3) * 2.0, np.diag([1.0, 1.0, -1.0]), np.eye(4), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError):
            run(x, th, badR)
    with pytest.raises(ValueError):
        run(x, th, R, K=np.eye(2))
    with pytest.raises(ValueError):
        run(x, th, R, sky_mask=np.ones((8, 9), bool))
    with pytest.raises(ValueError):
        run(x, th, R, sky_mask=np.full((8, 10), 1.5))
    with pytest.raises(ValueError):
        run(x, th, R, sky_mask=np.full((8, 10), -0.1))
    with pytest.raises(ValueError):
        run(x, th, R, up=(0, 0, 0))
    with pytest.raises(TypeError):
        cpl.apply_cpl(x, th, R)  # R_w2c 는 키워드 전용
    # 검증은 s=0 이어도 일어난다
    with pytest.raises(ValueError):
        run(bad, theta(cpl=0.0), R)
    with pytest.raises(ValueError):
        cpl.apply_polarizer(x, np.zeros((8, 9)), np.zeros((8, 9, 3)), 0.0)  # dop 모양 != 영상


# ------------------------------------------------------------------------------------------------ LNS-04 이음매
def test_chain_with_lns04_and_exposure_compensation():
    from buc.lens import sensor_film as sf

    h, w = 24, 32
    mid = flat(h, w, (0.18, 0.18, 0.18))
    R = cam_R_w2c(0.0)
    th_cpl = theta(cpl=1.0)  # 태양 없음 -> 균일 τ
    dark = run(mid, th_cpl, R)
    assert np.allclose(dark, 0.18 * TAU, rtol=1e-6)
    # LNS-04 는 ev 만 적용: 보상 전/후 표시값 비교(노이즈 끔)
    raw = sf.apply_sensor_film(dark, theta(cpl=1.0), noise=False)
    comp = sf.apply_sensor_film(dark, theta(cpl=1.0, ev=cpl.compensation_ev(1.0)), noise=False)
    want_raw = color.linear_to_display(np.float32(0.18 * TAU))
    want_comp = color.linear_to_display(np.float32(0.18))
    assert np.max(np.abs(raw - want_raw)) < 3e-3 and np.max(np.abs(comp - want_comp)) < 3e-3
    assert float(comp.mean()) > float(raw.mean()) + 0.05  # 보상하면 실제로 밝아진다
    # 순서: CPL 은 선형 장면 단계라 센서 이후가 아니다. 선형 곱은 센서 전에만 정확하다.
    assert color.linear_to_display(dark).mean() < color.linear_to_display(mid).mean()


# ------------------------------------------------------------------------------------------------ 격리·성능
def test_module_has_no_torch_or_scipy_and_no_global_state():
    code = (
        "import sys, numpy as np\n"
        "from buc.lens import cpl\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'scipy' not in sys.modules, 'scipy imported'\n"
        "st = np.random.get_state()[1].copy()\n"
        "from buc.core import contracts as C\n"
        "cpl.apply_cpl(np.ones((6,8,3),np.float32), C.Theta(cpl=1.0, sun_elev_deg=5.0, sun_azim_deg=90.0),\n"
        "              R_w2c=np.eye(3), up=(0,0,1))\n"
        "assert (np.random.get_state()[1] == st).all(), 'global RNG touched'\n"
        "print('OK')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "OK" in p.stdout, p.stderr[-600:]


def test_lens_package_stays_a_namespace():
    assert not (REPO / "buc" / "lens" / "__init__.py").exists()  # 네임스페이스 유지(계약서 소유 규칙)


def test_full_hd_class_frame_is_fast():
    th = theta(cpl=1.0, focal_mm=24.0, sun_elev_deg=20.0, sun_azim_deg=100.0, extra={"cpl_angle_deg": 70.0})
    x = np.random.default_rng(0).random((360, 640, 3), dtype=np.float32)
    R = cam_R_w2c(15.0, 3.0)
    run(x, th, R)  # 워밍업
    t0 = time.perf_counter()
    y = run(x, th, R, sky_mask=np.ones((360, 640), bool))
    assert time.perf_counter() - t0 < 3.0
    assert y.shape == x.shape and y.dtype == np.float32
