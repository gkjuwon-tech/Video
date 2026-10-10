"""WLD-17: Hosek–Wilkie 함수형(hw_shape), 계수 공급(hw_params), 하늘 휘도의 물리적 거동. 단위: 선형 Rec.709, 휘도 kcd/m^2."""
import math

import numpy as np
import pytest

from buc.world import farfield as F
from wld17_helpers import elev_az_dir, lum, rand_dirs


def spec(**kw):
    base = dict(up=(0.0, 0.0, 1.0), north=(1.0, 0.0, 0.0), sun_elev_deg=45.0, sun_azim_deg=0.0, turbidity=2.5, albedo=0.3)
    base.update(kw)
    return F.FarFieldSpec(**base)


def L(s, elev, az):
    return np.asarray(F.sky_radiance(s, elev_az_dir(s, elev, az)))


def hw_oracle(c, cos_t, gamma):
    """테스트 쪽 독립 구현. 계수 순서 (A,B,C,D,E,F,G,H,I)."""
    A, B, Cc, D, E, Ff, G, H, I = c
    ct = np.clip(cos_t, 0.0, 1.0)
    cg = np.cos(gamma)
    chi = (1 + cg ** 2) / (1 + H * H - 2 * H * cg) ** 1.5
    return (1 + A * np.exp(B / (ct + 0.01))) * (Cc + D * np.exp(E * gamma) + Ff * cg ** 2 + G * chi + I * np.sqrt(ct))


def test_hw_shape_matches_closed_form():
    rng = np.random.default_rng(0)
    for _ in range(40):
        c = np.concatenate([rng.uniform(-2, 1, 2), rng.uniform(0, 2, 1), rng.uniform(0, 4, 1), rng.uniform(-6, -0.5, 1),
                            rng.uniform(-1, 2, 2), rng.uniform(0.3, 0.98, 1), rng.uniform(-0.5, 2, 1)])
        ct = rng.uniform(0, 1, 50)
        g = rng.uniform(0, math.pi, 50)
        np.testing.assert_allclose(F.hw_shape(c, ct, g), hw_oracle(c, ct, g), rtol=1e-12, atol=1e-12)
    c = np.array([-1.0, -0.3, 0.4, 1.2, -2.0, 0.2, 0.3, 0.9, 0.1])
    # 지평선 아래(cosθ<0)는 cosθ=0 으로 클램프, 브로드캐스트
    np.testing.assert_allclose(F.hw_shape(c, -0.5, 1.0), F.hw_shape(c, 0.0, 1.0), rtol=1e-12)
    assert F.hw_shape(c, np.linspace(0, 1, 6)[:, None], np.linspace(0, 3, 4)[None, :]).shape == (6, 4)


def test_hw_params_shape_range_and_determinism():
    cs, z = F.hw_params(2.5, 0.3, 45.0)
    assert cs.shape == (3, 9) and z.shape == (3,)
    assert np.isfinite(cs).all() and np.isfinite(z).all() and (z > 0).all()
    assert ((cs[:, 7] >= 0) & (cs[:, 7] < 1)).all()  # 이방성 H 는 [0,1): χ 의 분모가 특이하지 않다
    cs2, z2 = F.hw_params(2.5, 0.3, 45.0)
    np.testing.assert_array_equal(cs, cs2)
    np.testing.assert_array_equal(z, z2)
    cs[:] = 0.0
    z[:] = 0.0  # 반환값은 사본: 내부 상태를 오염시키지 않는다
    cs3, z3 = F.hw_params(2.5, 0.3, 45.0)
    np.testing.assert_array_equal(cs3, cs2)
    np.testing.assert_array_equal(z3, z2)
    for bad in [(0.5, 0.3, 45.0), (11.0, 0.3, 45.0), (2.5, -0.1, 45.0), (2.5, 1.1, 45.0), (np.nan, 0.3, 45.0), (2.5, 0.3, 91.0)]:
        with pytest.raises(F.FarFieldError):
            F.hw_params(*bad)


def test_sky_radiance_is_hw_shape_times_zenith_scale():
    rng = np.random.default_rng(3)
    for sp in (spec(), spec(sun_elev_deg=12.0, sun_azim_deg=200.0, turbidity=6.0, albedo=0.8, exposure=1.7)):
        d = rand_dirs(500, 5)
        d = d[d @ np.asarray(sp.up) > -0.3]  # 지평선 아래도 일부 포함: cosθ 클램프
        cs, z = F.hw_params(sp.turbidity, sp.albedo, sp.sun_elev_deg)
        up = np.asarray(sp.up)
        gam = np.arccos(np.clip(d @ sp.sun_dir(), -1, 1))
        want = np.stack([hw_oracle(cs[c], d @ up, gam) * z[c] for c in range(3)], axis=-1) * sp.exposure
        np.testing.assert_allclose(F.sky_radiance(sp, d), want, rtol=1e-9)
    assert F.sky_radiance(spec(), np.array([0.0, 0.0, 1.0])).shape == (3,)
    assert F.sky_radiance(spec(), rand_dirs(12, 1).reshape(3, 4, 3)).shape == (3, 4, 3)
    del rng


def test_sky_is_symmetric_about_the_sun_meridian():
    """HW 는 (θ, γ) 만의 함수: 해 자오면에 대한 거울 대칭. 방위 이음매·좌표 실수를 잡는다."""
    for sp in (spec(), spec(sun_azim_deg=137.0, sun_elev_deg=20.0, up=(0.0, -1.0, 0.0), north=(0.0, 0.0, 1.0))):
        up = np.asarray(sp.up)
        n = np.cross(up, sp.sun_dir())
        n /= np.linalg.norm(n)
        d = rand_dirs(400, 8)
        dm = d - 2 * (d @ n)[:, None] * n
        np.testing.assert_allclose(F.sky_radiance(sp, d), F.sky_radiance(sp, dm), rtol=1e-9)


def test_zenith_luminance_is_physically_anchored():
    s = spec()
    y = float(lum(L(s, 90.0, 0.0)))
    assert 1.0 < y < 20.0, y  # kcd/m^2: 맑은 하늘 천정 휘도 수 kcd/m^2
    assert np.isfinite(L(s, 90.0, 0.0)).all()


def test_zenith_is_bluer_than_horizon_and_blue_overall():
    s = spec()
    z = L(s, 90.0, 0.0)
    h = L(s, 0.0, 180.0)  # 태양 반대편 지평선
    assert 1.15 < z[2] / z[0] < 6.0
    assert h[2] / h[0] < z[2] / z[0]  # 지평선이 더 희다
    assert (z > 0).all() and (h > 0).all()


def test_circumsolar_glow():
    s = spec()
    near = lum(L(s, 45.0, 0.0 + 3.0))  # 해 근처
    far = lum(L(s, 45.0, 180.0))  # 반대 방위, 같은 고도
    assert near > 3.0 * far, (near, far)


def test_luminance_decreases_away_from_sun_at_fixed_elevation():
    s = spec()
    ys = np.array([float(lum(L(s, 45.0, a))) for a in (0, 5, 10, 20, 40, 70, 110, 180)])
    assert (np.diff(ys) <= 1e-9).all(), ys


def test_zenith_luminance_grows_with_sun_elevation():
    ys = [float(lum(L(spec(sun_elev_deg=e), 90.0, 0.0))) for e in (10.0, 30.0, 60.0)]
    assert ys[0] < ys[1] < ys[2], ys


def test_sun_below_horizon_is_dim_but_valid():
    day = float(lum(L(spec(sun_elev_deg=45.0), 90.0, 0.0)))
    for e in (-5.0, -10.0, -20.0):
        sp = spec(sun_elev_deg=e)
        v = np.asarray(F.sky_radiance(sp, rand_dirs(300, 4)))
        assert np.isfinite(v).all() and (v >= 0).all()
        assert float(lum(L(sp, 90.0, 0.0))) < 0.2 * day
    assert float(lum(L(spec(sun_elev_deg=-20.0), 90.0, 0.0))) < float(lum(L(spec(sun_elev_deg=-5.0), 90.0, 0.0)))


def test_turbidity_whitens_the_sky_and_albedo_never_darkens():
    clear = L(spec(turbidity=2.0), 90.0, 0.0)
    hazy = L(spec(turbidity=8.0), 90.0, 0.0)
    assert hazy[2] / hazy[0] < clear[2] / clear[0]
    d = rand_dirs(400, 6)
    d = d[d @ np.array([0.0, 0.0, 1.0]) > 0]
    lo = F.sky_radiance(spec(albedo=0.0), d)
    hi = F.sky_radiance(spec(albedo=1.0), d)
    assert (hi >= lo * (1 - 1e-6)).all()


def test_exposure_scales_linearly_and_orientation_is_free():
    d = rand_dirs(200, 2)
    a = F.sky_radiance(spec(), d)
    np.testing.assert_allclose(F.sky_radiance(spec(exposure=2.5), d), 2.5 * a, rtol=1e-12)
    # 같은 물리 상황을 다른 세계 축으로 적으면 같은 휘도: 세계를 R 로 돌려 up/north/방향을 함께 돌린다
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 축 순환(회전)
    s2 = F.FarFieldSpec(up=tuple(R @ [0, 0, 1.0]), north=tuple(R @ [1.0, 0, 0]), sun_elev_deg=45.0, sun_azim_deg=0.0,
                        turbidity=2.5, albedo=0.3)
    np.testing.assert_allclose(F.sky_radiance(s2, d @ R.T), a, rtol=1e-9)
