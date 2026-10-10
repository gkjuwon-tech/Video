"""LNS-02 랙 스케줄(디옵터 보간·이징)과 포커스 브리딩."""
import math

import numpy as np
import pytest

from buc.lens import dof
from lns02_helpers import coc_ref


def _smooth(u):
    return u * u * (3 - 2 * u)


def test_rack_endpoints_shape_dtype():
    z = dof.rack_schedule(25, 24.0, 1.0, 6.0)
    assert isinstance(z, np.ndarray) and z.shape == (25,) and z.dtype == np.float64
    assert z[0] == 1.0 and z[-1] == 6.0
    assert np.all(np.diff(z) > 0)
    assert dof.rack_schedule(1, 24.0, 2.0, 5.0).tolist() == [2.0]


@pytest.mark.parametrize("ease", ["linear", "smooth"])
def test_rack_interpolates_in_diopters(ease):
    n, fps, a, b = 31, 30.0, 0.8, 12.0
    z = dof.rack_schedule(n, fps, a, b, ease=ease)
    u = np.arange(n) / (n - 1)
    e = u if ease == "linear" else _smooth(u)
    ref = 1.0 / (1.0 / a + (1.0 / b - 1.0 / a) * e)
    np.testing.assert_allclose(z, ref, rtol=1e-9)
    # 디옵터 보간이면 먼 평면의 CoC 가 이징 e 에 거의 선형으로 변한다(z_f >> f 근사, 3.5% 이내)
    c = coc_ref(np.full(n, 1e6), z)
    lin = c[0] + (c[-1] - c[0]) * e
    assert np.abs(c - lin).max() < 0.035 * abs(c[0] - c[-1])


def test_rack_smooth_ease_has_zero_end_velocity():
    z = dof.rack_schedule(49, 24.0, 1.0, 4.0, ease="smooth")
    d = 1.0 / z
    v = np.diff(d)
    assert abs(v[0]) < 0.06 * np.abs(v).max() and abs(v[-1]) < 0.06 * np.abs(v).max()
    assert np.argmax(np.abs(v)) in (23, 24)


def test_rack_window_and_hold():
    fps = 24.0
    z = dof.rack_schedule(48, fps, 3.0, 1.0, start_s=0.5, duration_s=1.0, ease="linear")
    t = np.arange(48) / fps
    before, after = t <= 0.5 + 1e-12, t >= 1.5 - 1e-12
    assert np.all(z[before] == 3.0) and np.all(z[after] == 1.0)
    mid = (~before) & (~after)
    u = (t[mid] - 0.5) / 1.0
    np.testing.assert_allclose(1.0 / z[mid], 1 / 3.0 + (1.0 - 1 / 3.0) * u, rtol=1e-9)
    assert np.all(np.diff(z) <= 0)


def test_rack_to_infinity():
    z = dof.rack_schedule(9, 24.0, 2.0, math.inf, ease="linear")
    assert z[0] == 2.0 and np.isinf(z[-1]) and z[-1] > 0
    np.testing.assert_allclose(1.0 / z[:-1], 0.5 * (1 - np.arange(8) / 8), rtol=1e-9)
    z2 = dof.rack_schedule(5, 24.0, math.inf, 2.0, ease="linear")
    assert np.isinf(z2[0]) and z2[-1] == 2.0


def test_rack_validation():
    for args, kw in [((0, 24.0, 1.0, 2.0), {}), ((3.5, 24.0, 1.0, 2.0), {}), ((5, 0.0, 1.0, 2.0), {}),
                     ((5, 24.0, 0.0, 2.0), {}), ((5, 24.0, 1.0, -2.0), {}), ((5, 24.0, float("nan"), 2.0), {}),
                     ((5, 24.0, 1.0, 2.0), {"ease": "bounce"}), ((5, 24.0, 1.0, 2.0), {"start_s": -1.0}),
                     ((5, 24.0, 1.0, 2.0), {"duration_s": 0.0}), ((5, 24.0, 1.0, 2.0), {"duration_s": float("inf")})]:
        with pytest.raises(ValueError):
            dof.rack_schedule(*args, **kw)


def test_breathing_scale_formula():
    f = 50.0
    assert dof.breathing_scale(f, 2.0, 0.0) == pytest.approx(1.0)
    assert dof.breathing_scale(f, math.inf, 0.3) == pytest.approx(1.0)
    for z, beta, zr in [(0.5, 0.3, math.inf), (2.0, -0.0, 1.0), (1.0, 0.5, 4.0), (4.0, 0.5, 1.0)]:
        m = (f / 1000) / (z - f / 1000)
        mr = 0.0 if math.isinf(zr) else (f / 1000) / (zr - f / 1000)
        exp = (1 + beta * m) / (1 + beta * mr)
        assert dof.breathing_scale(f, z, beta, ref_focus_m=zr) == pytest.approx(exp, rel=1e-9)
    assert dof.breathing_scale(f, 0.5, 0.3) > dof.breathing_scale(f, 2.0, 0.3) > 1.0  # 가까이 맞출수록 확대
    zs = np.array([0.5, 1.0, 2.0])
    arr = dof.breathing_scale(f, zs, 0.3)
    assert np.shape(arr) == (3,)
    assert dof.breathing_scale(f, 0.5, -0.3) < 1.0  # 음의 β: 가까이 맞추면 축소(허용)
    with pytest.raises(ValueError):
        dof.breathing_scale(f, 0.04, 0.3)  # focus <= f
    for bad_beta in (1.5, -1.5, float("nan")):
        with pytest.raises(ValueError):
            dof.breathing_scale(f, 1.0, bad_beta)


def test_breathing_K_and_apply_breathing():
    K = np.array([[100.0, 0, 40.0], [0, 100.0, 30.0], [0, 0, 1]])
    K2 = dof.breathing_K(K, 1.1)
    np.testing.assert_allclose(K2, [[110.0, 0, 40.0], [0, 110.0, 30.0], [0, 0, 1]])
    np.testing.assert_allclose(K, [[100.0, 0, 40.0], [0, 100.0, 30.0], [0, 0, 1]])  # 입력 불변
    h, w = 61, 81
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    blob = np.exp(-((x - 60.0) ** 2 + (y - 20.0) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
    img = np.stack([blob, blob * 0.5, blob * 0.25], -1)
    same = dof.apply_breathing(img, 1.0, K)
    np.testing.assert_array_equal(same, img)
    s = 1.08
    out = dof.apply_breathing(img, s, K)
    assert out.shape == img.shape and out.dtype == np.float32
    o = out[..., 0].astype(np.float64)
    cx, cy = (o * x).sum() / o.sum(), (o * y).sum() / o.sum()
    assert cx == pytest.approx(40.0 + s * 20.0, abs=0.1)
    assert cy == pytest.approx(30.0 + s * -10.0, abs=0.1)
    d2 = dof.apply_breathing(blob, s, K)  # 2D 입력
    assert d2.shape == blob.shape
    dn = dof.apply_breathing(np.where(blob > 0.5, 2.0, 7.0).astype(np.float32), s, K, nearest=True)
    assert set(np.unique(dn).tolist()) <= {2.0, 7.0}  # 깊이용 최근접: 새 값 없음
