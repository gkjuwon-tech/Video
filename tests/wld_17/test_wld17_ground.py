"""WLD-17: 지평선(하늘↔지면 연속), 지면 연장(조명된 지면 + 지평선 헤이즈 대역), 태양 조도, 태양 원반 에너지."""
import math

import numpy as np
import pytest

from buc.world import farfield as F
from wld17_helpers import elev_az_dir, lum, rand_dirs, sphere_quadrature


def spec(**kw):
    base = dict(up=(0.0, 0.0, 1.0), north=(1.0, 0.0, 0.0), sun_elev_deg=45.0, sun_azim_deg=0.0, turbidity=2.5, albedo=0.3)
    base.update(kw)
    return F.FarFieldSpec(**base)


def env(s, elev, az):
    return np.asarray(F.environment_radiance(s, elev_az_dir(s, elev, az)))


def test_environment_equals_sky_above_horizon():
    s = spec()
    d = rand_dirs(2000, 1)
    d = d[d @ np.asarray(s.up) >= 0]
    np.testing.assert_allclose(F.environment_radiance(s, d), F.sky_radiance(s, d), rtol=1e-12)


def test_horizon_is_continuous_for_every_azimuth():
    for s in (spec(), spec(sun_elev_deg=8.0, sun_azim_deg=90.0), spec(sun_elev_deg=-10.0)):
        for eps, tol in ((0.05, 0.06), (0.005, 0.01)):
            worst = 0.0
            for az in range(0, 360, 10):
                a, b = env(s, +eps, az), env(s, -eps, az)
                worst = max(worst, float(np.abs(a - b).max() / a.max()))
            assert worst < tol, (eps, worst)


def test_ground_far_below_is_azimuth_independent_and_equals_lit_ground():
    s = spec()
    g = F.ground_lit(s)
    assert g.shape == (3,) and (g > 0).all()
    np.testing.assert_allclose(F.environment_radiance(s, -np.asarray(s.up)), g, rtol=1e-4)
    vals = np.array([env(s, -40.0, az) for az in range(0, 360, 30)])
    np.testing.assert_allclose(vals, np.broadcast_to(vals[0], vals.shape), rtol=1e-4)
    np.testing.assert_allclose(vals[0], g, rtol=1e-3)


def test_ground_lit_is_albedo_over_pi_times_irradiance():
    for s in (spec(), spec(sun_elev_deg=12.0, turbidity=5.0, albedo=0.6), spec(sun_elev_deg=-8.0)):
        es = math.sin(math.radians(max(s.sun_elev_deg, 0.0)))
        want = s.albedo / math.pi * (np.asarray(F.sky_irradiance(s)) + np.asarray(F.sun_irradiance(s)) * es)
        np.testing.assert_allclose(F.ground_lit(s), want, rtol=1e-9)


def test_ground_albedo_override_is_per_channel():
    base = spec(ground_albedo_rgb=(1.0, 1.0, 1.0))
    col = spec(ground_albedo_rgb=(0.1, 0.2, 0.4))
    np.testing.assert_allclose(F.ground_lit(col), F.ground_lit(base) * np.array([0.1, 0.2, 0.4]), rtol=1e-9)
    np.testing.assert_allclose(F.environment_radiance(col, -np.asarray(col.up)), F.ground_lit(col), rtol=1e-4)
    # 지면 알베도 override 는 하늘을 바꾸지 않는다
    d = rand_dirs(100, 2)
    d = d[d @ np.asarray(col.up) > 0]
    np.testing.assert_allclose(F.sky_radiance(col, d), F.sky_radiance(spec(), d), rtol=1e-12)


def test_sky_irradiance_matches_independent_quadrature():
    for s, tol in ((spec(), 0.03), (spec(sun_elev_deg=10.0, sun_azim_deg=70.0), 0.05), (spec(turbidity=7.0), 0.03)):
        d, w = sphere_quadrature(s, 192, 384)
        want = (np.asarray(F.sky_radiance(s, d)) * w[..., None]).sum((0, 1))
        np.testing.assert_allclose(F.sky_irradiance(s), want, rtol=tol)


def test_direct_fraction_of_horizontal_irradiance_is_clear_sky_like():
    s = spec()
    e_sun_h = float(lum(F.sun_irradiance(s))) * math.sin(math.radians(45.0))
    e_sky = float(lum(F.sky_irradiance(s)))
    assert 0.5 < e_sun_h / (e_sun_h + e_sky) < 0.95, (e_sun_h, e_sky)


def test_sun_irradiance_trends():
    z = F.sun_irradiance(spec(sun_elev_deg=0.0))
    assert z.shape == (3,) and (np.asarray(F.sun_irradiance(spec(sun_elev_deg=-5.0))) == 0).all() and (z == 0).all()
    y = [float(lum(F.sun_irradiance(spec(sun_elev_deg=e)))) for e in (10.0, 30.0, 60.0)]
    assert 0 < y[0] < y[1] < y[2], y
    assert float(lum(F.sun_irradiance(spec(turbidity=2.0)))) > float(lum(F.sun_irradiance(spec(turbidity=8.0))))
    low, high = F.sun_irradiance(spec(sun_elev_deg=10.0)), F.sun_irradiance(spec(sun_elev_deg=60.0))
    assert low[2] / low[0] < high[2] / high[0]  # 낮은 해는 더 붉다
    assert (low >= 0).all() and (high >= 0).all()


def test_exposure_scales_everything_linearly():
    a, b = spec(), spec(exposure=2.0)
    d = rand_dirs(300, 3)
    np.testing.assert_allclose(F.environment_radiance(b, d), 2.0 * np.asarray(F.environment_radiance(a, d)), rtol=1e-9)
    np.testing.assert_allclose(F.sky_irradiance(b), 2.0 * np.asarray(F.sky_irradiance(a)), rtol=1e-9)
    np.testing.assert_allclose(F.sun_irradiance(b), 2.0 * np.asarray(F.sun_irradiance(a)), rtol=1e-9)
    np.testing.assert_allclose(F.ground_lit(b), 2.0 * np.asarray(F.ground_lit(a)), rtol=1e-9)


def test_horizon_haze_band_fades_monotonically_to_ground():
    s = spec()
    g = F.ground_lit(s)
    errs = []
    for dep in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        errs.append(float(np.mean([abs(float(lum(env(s, -dep, az))) - float(lum(g))) for az in range(0, 360, 20)])))
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    # 대역폭이 좁을수록 같은 깊이에서 지면에 더 가깝다
    wide, narrow = spec(horizon_blend_deg=3.0), spec(horizon_blend_deg=0.5)
    ew = abs(float(lum(env(wide, -2.0, 90))) - float(lum(F.ground_lit(wide))))
    en = abs(float(lum(env(narrow, -2.0, 90))) - float(lum(F.ground_lit(narrow))))
    assert en < ew
    assert s.horizon_blend_deg == 1.5  # 기본 대역 1.5도


def test_ground_plane_distance():
    s = spec(camera_height_m=1.6)
    up = np.asarray(s.up)
    north = np.asarray(s.north)
    assert float(F.ground_distance(s, -up)) == pytest.approx(1.6, rel=1e-12)
    d30 = -math.sin(math.radians(30)) * up + math.cos(math.radians(30)) * north
    assert float(F.ground_distance(s, d30)) == pytest.approx(3.2, rel=1e-9)
    assert math.isinf(float(F.ground_distance(s, up)))
    assert math.isinf(float(F.ground_distance(s, north)))  # 수평선: 무한원
    out = F.ground_distance(s, rand_dirs(24, 1).reshape(4, 6, 3))
    assert out.shape == (4, 6)
    dirs = rand_dirs(500, 2)
    below = dirs @ up < -1e-6
    assert np.isfinite(np.asarray(F.ground_distance(s, dirs))[below]).all()
    assert np.isinf(np.asarray(F.ground_distance(s, dirs))[~below & (dirs @ up >= 0)]).all()


# ---- 태양 원반 --------------------------------------------------------------------------------------------------------
def test_sun_disk_constant():
    assert 0.26 <= F.SUN_ANGULAR_RADIUS_DEG <= 0.27


def test_include_sun_conserves_irradiance_and_stays_local():
    S = 256
    s0 = spec(size=S)
    s1 = spec(size=S, include_sun=True)
    a = np.asarray(s0.build().rgb, dtype=np.float64)
    b = np.asarray(s1.build().rgb, dtype=np.float64)
    delta = b - a
    assert (delta >= -1e-6).all()
    om = F.texel_solid_angle(S)
    d = F.texel_dirs(S)
    np.testing.assert_allclose((delta * om[..., None]).sum((0, 1)), F.sun_irradiance(s1), rtol=0.01)  # 법선 조도 E_n 보존
    sd = s1.sun_dir()
    ang = np.degrees(np.arccos(np.clip(d @ sd, -1, 1)))
    peak = delta.max()
    assert delta[ang > 15.0].max() <= 1e-3 * peak
    i = np.unravel_index(np.argmax(lum(delta)), ang.shape)
    assert ang[i] < 1.5  # 최대는 해 방향
    assert peak > 20 * a.max()  # 태양 원반은 하늘보다 훨씬 밝다(HDR)


def test_include_sun_ignored_when_sun_below_horizon():
    for e in (-5.0, 0.0):
        a = spec(size=64, sun_elev_deg=e).build().rgb
        b = spec(size=64, sun_elev_deg=e, include_sun=True).build().rgb
        np.testing.assert_array_equal(a, b)
