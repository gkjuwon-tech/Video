"""LNS-07 하늘 기하: 태양 방향 규약, 시선 광선, Rayleigh 편광 장(DoP·E벡터)."""
import math

import numpy as np
import pytest

from buc.lens import cpl
from lns07_helpers import NORTH, P_MAX, UP, cam_R_w2c, rayleigh_ref, rays_ref, sun_world_ref, theta, unit


# ------------------------------------------------------------------------------------------------ 태양 방향
def test_sun_dir_axes_and_unit_norm():
    n = np.array(NORTH)
    east = np.cross(NORTH, UP)
    assert np.allclose(east, (0, -1, 0))
    d = lambda e, a: cpl.sun_dir_world(e, a, up=UP, north=NORTH)  # noqa: E731
    assert np.allclose(d(0, 0), n, atol=1e-12)
    assert np.allclose(d(0, 90), east, atol=1e-12)
    assert np.allclose(d(0, 180), -n, atol=1e-12)
    assert np.allclose(d(90, 0), UP, atol=1e-12)
    assert np.allclose(d(30, 90), 0.5 * np.array(UP) + math.cos(math.radians(30)) * east, atol=1e-12)
    for e in (-20, 0, 33, 90):
        for a in (0, 77, 359.5):
            v = d(e, a)
            assert v.shape == (3,) and v.dtype == np.float64 and abs(np.linalg.norm(v) - 1) < 1e-12
            assert np.allclose(v, sun_world_ref(e, a), atol=1e-12)
            assert v @ np.array(UP) == pytest.approx(math.sin(math.radians(e)), abs=1e-12)  # 고도


def test_sun_dir_matches_farfield_convention():
    from buc.world.farfield import FarFieldSpec

    cases = [((0, 0, 1), (1, 0, 0)), ((0, 0, 1), (0.6, 0.8, 0.0)), ((0, 1, 0), (1, 0, 0)), ((0, 1, 0), None),
             ((0, 0, 1), None), ((1, 2, 3), (0, 0, 1)), ((0, -1, 0), None)]
    for up, north in cases:
        spec_kw = dict(up=up, north=north)
        for e, a in ((10, 20), (-5, 300), (60, 123.4)):
            ref = FarFieldSpec(sun_elev_deg=e, sun_azim_deg=a, **spec_kw).sun_dir()
            got = cpl.sun_dir_world(e, a, up=up, north=north)
            assert np.allclose(got, ref, atol=1e-9), (up, north, e, a)


def test_sun_dir_validation():
    with pytest.raises(ValueError):
        cpl.sun_dir_world(10, 10, up=(0, 0, 0))
    with pytest.raises(ValueError):
        cpl.sun_dir_world(10, 10, up=(0, 0, 1), north=(0, 0, 2))  # north ∥ up
    with pytest.raises(ValueError):
        cpl.sun_dir_world(10, 10, up=(0, 0, float("nan")))
    with pytest.raises(ValueError):
        cpl.sun_dir_world(10, 10, up=(0, 0))  # 모양
    with pytest.raises(TypeError):
        cpl.sun_dir_world(10, 10, (0, 0, 1))  # up 은 키워드 전용


# ------------------------------------------------------------------------------------------------ 시선 광선
def test_view_rays_center_corners_and_unit():
    th = theta(focal_mm=24.0)
    h, w = 33, 61
    K = th.intrinsics(w, h)
    r = cpl.view_rays(K, h, w)
    assert r.shape == (h, w, 3) and r.dtype == np.float64
    assert np.allclose(np.linalg.norm(r, axis=-1), 1.0, atol=1e-12)
    assert np.allclose(r[h // 2, w // 2], (0, 0, 1), atol=1e-12)  # 주점(홀수 크기에서 정중앙 화소)
    assert np.allclose(r, rays_ref(th, h, w), atol=1e-12)
    # x 오른쪽, y 아래, z 앞(OpenCV)
    assert r[h // 2, w - 1, 0] > 0 and r[h // 2, 0, 0] < 0 and r[h - 1, w // 2, 1] > 0 and r[0, w // 2, 1] < 0
    assert np.all(r[..., 2] > 0)
    # 가장자리 화소의 수평각 = atan(((W-1)/2)/fx)
    fx = K[0, 0]
    ang = math.atan2(r[h // 2, w - 1, 0], r[h // 2, w - 1, 2])
    assert ang == pytest.approx(math.atan(((w - 1) / 2) / fx), abs=1e-12)
    # 화각: 화소 가장자리(반 화소 바깥) 기준으로 θ.fov_x_deg()와 일치
    assert 2 * math.degrees(math.atan((w / 2) / fx)) == pytest.approx(th.fov_x_deg(), abs=1e-9)


def test_view_rays_validation():
    with pytest.raises(ValueError):
        cpl.view_rays(np.eye(2), 4, 4)
    with pytest.raises(ValueError):
        cpl.view_rays(np.full((3, 3), np.nan), 4, 4)


# ------------------------------------------------------------------------------------------------ 편광 장
def test_beam_perpendicular_sun_gives_pmax_and_vertical_e_vector():
    # 북쪽을 보는 수평 카메라, 태양은 동쪽 지평선: 중앙 시선과 태양이 정확히 90°
    th = theta(focal_mm=100.0)
    h, w = 31, 41
    R = cam_R_w2c(0.0)
    assert np.allclose(R @ np.cross(NORTH, UP), (1, 0, 0), atol=1e-12)  # east -> 카메라 x(오른쪽)
    assert np.allclose(R @ np.array(UP), (0, -1, 0), atol=1e-12)  # up -> 카메라 -y(위)
    sun_cam = R @ cpl.sun_dir_world(0, 90, up=UP, north=NORTH)
    assert np.allclose(sun_cam, (1, 0, 0), atol=1e-12)
    dop, ev = cpl.sky_polarization(cpl.view_rays(th.intrinsics(w, h), h, w), sun_cam)
    assert dop.shape == (h, w) and ev.shape == (h, w, 3) and dop.dtype == np.float64 and ev.dtype == np.float64
    assert dop[h // 2, w // 2] == pytest.approx(P_MAX, abs=1e-12)
    assert np.allclose(np.abs(ev[h // 2, w // 2]), (0, 1, 0), atol=1e-12)  # 산란면(수평)에 수직 = 영상 세로
    assert dop.max() <= P_MAX + 1e-12 and dop.min() >= 0.0


def test_dop_field_matches_rayleigh_of_angle_to_sun():
    th = theta(focal_mm=18.0)  # 광각
    h, w = 40, 60
    rays = cpl.view_rays(th.intrinsics(w, h), h, w)
    for sun_cam in (unit((0.6, -0.3, 0.2)), unit((-1, 0.2, 0.5)), unit((0, -1, 0)), unit((0.3, 0.3, -1))):
        dop, ev = cpl.sky_polarization(rays, sun_cam)
        gamma = np.arccos(np.clip(rays @ sun_cam, -1, 1))
        assert np.allclose(dop, rayleigh_ref(gamma), atol=1e-9)
        # E벡터: 단위, 시선과 태양 모두에 수직(= 산란면에 수직), 영벡터는 dop 0 인 곳뿐
        assert np.allclose(np.linalg.norm(ev, axis=-1), 1.0, atol=1e-9)
        assert np.max(np.abs(np.einsum("hwc,hwc->hw", ev, rays))) < 1e-9
        assert np.max(np.abs(ev @ sun_cam)) < 1e-9
    dop2, _ = cpl.sky_polarization(rays, unit((0.6, -0.3, 0.2)), p_max=0.5)
    dop1, _ = cpl.sky_polarization(rays, unit((0.6, -0.3, 0.2)))
    assert np.allclose(dop2, dop1 * (0.5 / P_MAX), atol=1e-12)


def test_sun_in_line_with_view_is_unpolarized_without_nan():
    th = theta(focal_mm=35.0)
    h, w = 25, 25
    rays = cpl.view_rays(th.intrinsics(w, h), h, w)
    for sun_cam in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])):  # 태양이 앞/뒤 광축 위
        dop, ev = cpl.sky_polarization(rays, sun_cam)
        assert np.isfinite(dop).all() and np.isfinite(ev).all()
        assert dop[h // 2, w // 2] == 0.0 and np.all(ev[h // 2, w // 2] == 0.0)  # 외적 0 -> dop 0, evec 0
        assert dop[h // 2, w // 2 + 8] > 0.0  # 주변은 편광된다
    # 광축 위 태양: 방사 대칭 -> 같은 반지름의 dop 는 같다
    d, _ = cpl.sky_polarization(rays, np.array([0.0, 0.0, 1.0]))
    assert d[h // 2, w // 2 + 7] == pytest.approx(d[h // 2 - 7, w // 2], abs=1e-12)
    assert d[h // 2, w // 2 + 7] == pytest.approx(d[h // 2, w // 2 - 7], abs=1e-12)


def test_polarization_pattern_is_tangential_around_the_sun():
    # 태양이 영상 안에 있으면 E벡터는 태양 둘레의 접선 방향(반지름 방향에 수직)
    th = theta(focal_mm=20.0)
    h, w = 41, 61
    rays = cpl.view_rays(th.intrinsics(w, h), h, w)
    sun_cam = unit((0.1, 0.05, 1.0))
    dop, ev = cpl.sky_polarization(rays, sun_cam)
    # 시선에서 태양으로 가는 방향을 영상 평면으로 투영 -> 접선 E벡터와 직교
    to_sun = sun_cam - np.einsum("hwc,c->hw", rays, sun_cam)[..., None] * rays
    assert np.max(np.abs(np.einsum("hwc,hwc->hw", ev, to_sun))) < 1e-9


def test_sky_polarization_does_not_touch_inputs_and_validates():
    th = theta()
    rays = cpl.view_rays(th.intrinsics(8, 6), 6, 8)
    keep = rays.copy()
    s = unit((0.3, 0.2, 0.9))
    keeps = s.copy()
    a = cpl.sky_polarization(rays, s)
    b = cpl.sky_polarization(rays, s)
    assert np.array_equal(rays, keep) and np.array_equal(s, keeps)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    with pytest.raises(ValueError):
        cpl.sky_polarization(rays, np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        cpl.sky_polarization(rays, np.array([np.nan, 0.0, 1.0]))
    with pytest.raises(ValueError):
        cpl.sky_polarization(rays[..., :2], s)
    with pytest.raises(ValueError):
        cpl.sky_polarization(rays, s, p_max=0.0)
    with pytest.raises(ValueError):
        cpl.sky_polarization(rays, 2.0 * s)  # 단위 벡터가 아니다
