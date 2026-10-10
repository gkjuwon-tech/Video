"""LNS-02 CoC 수식, 부호, 무한대, 입력 검증, Defocus Pyramid 단계, 조리개 커널."""
import dataclasses
import inspect
import math

import numpy as np
import pytest

from buc.core import contracts as C
from buc.lens import dof
from lns02_helpers import FOCAL, FSTOP, LENS, SENSOR, W, coc_ref, disk_kernel, second_moment


def test_public_names():
    for n in ("DofOpts", "DofResult", "coc_px", "coc_from_theta", "defocus_level", "aperture_kernel",
              "apply_dof", "apply_dof_theta", "rack_schedule", "breathing_scale", "breathing_K",
              "apply_breathing", "apply_dof_frames"):
        assert hasattr(dof, n), n
    p = inspect.signature(dof.apply_dof).parameters
    for n in ("focal_mm", "f_stop", "focus_m", "sensor_width_mm", "opts"):
        assert p[n].kind is inspect.Parameter.KEYWORD_ONLY, n
    p = inspect.signature(dof.coc_px).parameters
    for n in ("focal_mm", "f_stop", "focus_m", "sensor_width_mm", "width_px", "signed"):
        assert p[n].kind is inspect.Parameter.KEYWORD_ONLY, n


def test_opts_defaults_and_validation():
    o = dof.DofOpts()
    assert (o.blades, o.blade_rot_deg, o.pyramid, o.pyramid_threshold_px, o.max_coc_px, o.in_focus_px,
            o.breathing_beta) == (0, 0.0, True, 8.0, 64.0, 0.5, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.blades = 5  # frozen
    for bad in (dict(blades=2), dict(blades=19), dict(blades=-1), dict(blades=5.5), dict(max_coc_px=0.5),
                dict(pyramid_threshold_px=1.0), dict(in_focus_px=-0.1), dict(breathing_beta=1.5), dict(breathing_beta=-1.5),
                dict(max_coc_px=float("nan")), dict(blade_rot_deg=float("inf"))):
        with pytest.raises(ValueError):
            dof.DofOpts(**bad)
    dof.DofOpts(blades=3)
    dof.DofOpts(blades=18)
    dof.DofOpts(breathing_beta=-0.4)
    dof.DofOpts(pyramid=False, pyramid_threshold_px=4.0, max_coc_px=128.0, in_focus_px=0.0)


@pytest.mark.parametrize("focus", [0.5, 1.2, 3.0, 10.0])
def test_coc_matches_thin_lens_formula(focus):
    z = np.array([0.2, 0.4, 0.9, 1.5, 3.0, 7.0, 25.0, 400.0])
    for signed in (False, True):
        got = dof.coc_px(z, **LENS, focus_m=focus, width_px=W, signed=signed)
        assert got.dtype == np.float32 and got.shape == z.shape
        np.testing.assert_allclose(got, coc_ref(z, focus, signed=signed), rtol=1e-5, atol=1e-5)


def test_coc_sign_zero_and_monotonic():
    zf = 2.0
    assert float(dof.coc_px(np.array([zf]), **LENS, focus_m=zf, width_px=W)[0]) == pytest.approx(0.0, abs=1e-6)
    s = dof.coc_px(np.array([1.0, 4.0]), **LENS, focus_m=zf, width_px=W, signed=True)
    assert s[0] < 0 < s[1]  # 앞(near)은 음, 뒤(far)는 양
    zs = np.linspace(2.0, 500.0, 200)
    c = dof.coc_px(zs, **LENS, focus_m=zf, width_px=W)
    assert np.all(np.diff(c) >= -1e-6)
    zn = np.linspace(0.2, 2.0, 200)
    cn = dof.coc_px(zn, **LENS, focus_m=zf, width_px=W)
    assert np.all(np.diff(cn) <= 1e-6)


def test_coc_infinity_limit_and_scalings():
    zf = 3.0
    inf = float(dof.coc_px(np.array([np.inf]), **LENS, focus_m=zf, width_px=W)[0])
    lim = FOCAL ** 2 / (FSTOP * (zf * 1000 - FOCAL)) * W / SENSOR
    assert inf == pytest.approx(lim, rel=1e-5)
    assert np.isfinite(inf)
    z = np.array([0.7, 9.0])
    a = dof.coc_px(z, **LENS, focus_m=zf, width_px=W)
    b = dof.coc_px(z, focal_mm=FOCAL, f_stop=FSTOP * 2, sensor_width_mm=SENSOR, focus_m=zf, width_px=W)
    np.testing.assert_allclose(b, a / 2, rtol=1e-5)  # 1/N
    c2 = dof.coc_px(z, **LENS, focus_m=zf, width_px=2 * W)
    np.testing.assert_allclose(c2, 2 * a, rtol=1e-5)  # 해상도 비례
    # 스칼라·2D 입력 모양 보존
    assert dof.coc_px(np.full((3, 4), 5.0), **LENS, focus_m=zf, width_px=W).shape == (3, 4)


def test_coc_invalid_inputs():
    z = np.array([1.0, 2.0])
    bad_kw = [dict(focus_m=0.05), dict(focus_m=0.085), dict(focus_m=-1.0), dict(focus_m=float("nan")),
              dict(f_stop=0.0), dict(f_stop=float("nan")), dict(focal_mm=-5.0), dict(sensor_width_mm=0.0),
              dict(width_px=0)]
    for b in bad_kw:
        kw = dict(**LENS, focus_m=2.0, width_px=W)
        kw.update(b)
        with pytest.raises(ValueError):
            dof.coc_px(z, **kw)
    for bz in (np.array([1.0, np.nan]), np.array([0.0, 1.0]), np.array([-1.0, 1.0])):
        with pytest.raises(ValueError):
            dof.coc_px(bz, **LENS, focus_m=2.0, width_px=W)


def test_coc_from_theta():
    th = C.Theta(focal_mm=50.0, f_stop=2.0, sensor_width_mm=24.89, focus_m=2.5)
    z = np.array([0.8, 2.5, 6.0, np.inf])
    got = dof.coc_from_theta(z, th, 320)
    ref = coc_ref(z, 2.5, focal_mm=50.0, f_stop=2.0, sensor_width_mm=24.89, width_px=320)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    got2 = dof.coc_from_theta(z, th, 320, focus_m=1.0, signed=True)
    ref2 = coc_ref(z, 1.0, focal_mm=50.0, f_stop=2.0, sensor_width_mm=24.89, width_px=320, signed=True)
    np.testing.assert_allclose(got2, ref2, rtol=1e-5, atol=1e-5)


def test_defocus_level_table():
    c = np.array([0.0, 3.0, 8.0, 8.01, 16.0, 16.5, 32.0, 33.0, 64.0, 100.0, -20.0])
    exp = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 2])
    got = dof.defocus_level(c)
    assert np.issubdtype(np.asarray(got).dtype, np.integer)
    np.testing.assert_array_equal(got, exp)
    np.testing.assert_array_equal(dof.defocus_level(np.array([4.0, 4.1, 9.0]), threshold_px=4.0), [0, 1, 2])


@pytest.mark.parametrize("d", [0.0, 0.7, 1.0])
def test_aperture_kernel_tiny_is_delta(d):
    k = dof.aperture_kernel(d)
    assert k.shape == (1, 1) and float(k[0, 0]) == pytest.approx(1.0)


@pytest.mark.parametrize("d", [3.0, 6.5, 12.0, 25.0])
def test_aperture_kernel_circle(d):
    k = np.asarray(dof.aperture_kernel(d), np.float64)
    assert k.ndim == 2 and k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert k.shape[0] == 2 * math.ceil(d / 2) + 1
    assert k.min() >= 0 and k.sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-6)
    np.testing.assert_allclose(k, k.T, atol=1e-6)
    m2, cx, cy = second_moment(k)
    c = (k.shape[0] - 1) / 2
    assert abs(cx - c) < 1e-3 and abs(cy - c) < 1e-3
    assert m2 == pytest.approx(d * d / 8.0, rel=0.12)
    # 이상 원판 오라클과 질량 분포가 가깝다(L1 < 0.15)
    ref = disk_kernel(d)
    pad = (ref.shape[0] - k.shape[0]) // 2
    if pad >= 0:
        kk = np.pad(k, pad)
        assert np.abs(kk - ref).sum() < 0.15
    else:
        rr = np.pad(ref, -pad)
        assert np.abs(k - rr).sum() < 0.15


def test_aperture_kernel_polygon():
    d = 21.0
    circ = np.asarray(dof.aperture_kernel(d), np.float64)
    for b in (5, 6, 9):
        k = np.asarray(dof.aperture_kernel(d, blades=b), np.float64)
        assert k.min() >= 0 and k.sum() == pytest.approx(1.0, abs=1e-5)
        m2c, _, _ = second_moment(circ)
        m2, _, _ = second_moment(k)
        assert 0.7 * m2c < m2 <= m2c * 1.02  # 원에 내접하는 다각형
        assert k.shape == circ.shape
        assert np.abs(k - circ).max() > 1e-4
    hexa = np.asarray(dof.aperture_kernel(d, blades=6), np.float64)
    hexr = np.asarray(dof.aperture_kernel(d, blades=6, rotation_deg=30.0), np.float64)
    assert np.abs(hexa - hexr).max() > 1e-4  # 회전이 반영된다
    np.testing.assert_allclose(np.asarray(dof.aperture_kernel(d, blades=6, rotation_deg=60.0)), hexa, atol=2e-3)
    with pytest.raises(ValueError):
        dof.aperture_kernel(-1.0)
    with pytest.raises(ValueError):
        dof.aperture_kernel(float("nan"))
    with pytest.raises(ValueError):
        dof.aperture_kernel(5.0, blades=2)
