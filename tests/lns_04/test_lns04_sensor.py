"""LNS-04 센서: Poisson-Gauss + 만충전 클립 + ADC. 사진 전달 곡선(PTC), 결정론, 독립성, 오류 경로."""
import warnings

import numpy as np
import pytest

from buc.lens import sensor_film as sf
from lns04_helpers import BITS, FW, RN, corr

SPEC = dict(full_well_e=FW, read_noise_e=RN, bits=BITS)
H, W = 128, 128
Q = 1.0 / ((1 << BITS) - 1)


def _spec(**kw):
    return sf.SensorSpec(**{**SPEC, **kw})


def _cap(x, seed=1, frame=0, spec=None, noise=True, shape=(H, W, 3)):
    img = np.full(shape, x, np.float32)
    return sf.sensor_capture(img, spec or _spec(), seed=seed, frame=frame, noise=noise)


def test_constants_match_contract():
    from lns04_helpers import ADC_BITS, FULL_WELL_MAX, FULL_WELL_MIN, FULL_WELL_PER_UM2, READ_NOISE_E
    assert sf.ADC_BITS == ADC_BITS and sf.READ_NOISE_E == READ_NOISE_E
    assert sf.FULL_WELL_PER_UM2 == FULL_WELL_PER_UM2
    assert (sf.FULL_WELL_MIN, sf.FULL_WELL_MAX) == (FULL_WELL_MIN, FULL_WELL_MAX)


def test_sensor_spec_defaults_and_validation():
    s = sf.SensorSpec()
    assert (s.full_well_e, s.read_noise_e, s.bits) == (20000.0, 3.0, 12)
    with pytest.raises(TypeError):
        sf.SensorSpec(20000.0)  # kw_only
    for bad in (dict(full_well_e=0.0), dict(full_well_e=-1.0), dict(full_well_e=float("nan")),
                dict(read_noise_e=-0.1), dict(bits=0), dict(bits=17), dict(bits=8.0), dict(bits=True)):
        with pytest.raises(ValueError):
            sf.SensorSpec(**bad)


# ----------------------------------------------------------------------------------------------- ADC
def test_adc_quantize_grid_and_endpoints():
    x = np.linspace(-0.2, 1.3, 5001).astype(np.float32)
    for bits in (1, 8, 12, 16):
        y = sf.adc_quantize(x, bits)
        assert y.dtype == np.float32 and y.shape == x.shape
        m = (1 << bits) - 1
        code = y.astype(np.float64) * m
        assert np.max(np.abs(code - np.round(code))) < 2e-3 * max(1, m / 4095)
        assert y.min() == 0.0 and y.max() == 1.0
        assert np.all(np.diff(y) >= 0)
        inside = (x >= 0) & (x <= 1)
        assert np.max(np.abs(y[inside] - x[inside])) <= 0.5 / m + 1e-6  # 최근접 반올림
        assert np.array_equal(sf.adc_quantize(y, bits), y)  # 멱등


def test_adc_quantize_errors():
    with pytest.raises(TypeError):
        sf.adc_quantize(np.zeros(3, np.int32), 8)
    with pytest.raises(ValueError):
        sf.adc_quantize(np.array([0.1, np.nan], np.float32), 8)
    for b in (0, 17, 8.0, True):
        with pytest.raises(ValueError):
            sf.adc_quantize(np.zeros(3, np.float32), b)


# ----------------------------------------------------------------------------------------------- 센서
def test_capture_output_contract():
    img = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    keep = img.copy()
    out = sf.sensor_capture(img, _spec(), seed=3)
    assert out.dtype == np.float32 and out.shape == img.shape
    assert np.array_equal(img, keep)  # 입력 불변
    code = out.astype(np.float64) * ((1 << BITS) - 1)
    assert np.max(np.abs(code - np.round(code))) < 1e-3  # ADC 격자 위
    assert out.min() >= 0.0 and out.max() <= 1.0
    out64 = sf.sensor_capture(img.astype(np.float64), _spec(), seed=3)
    assert out64.dtype == np.float32
    # 임의 모양 허용
    assert sf.sensor_capture(np.full((7,), 0.3, np.float32), _spec(), seed=1).shape == (7,)
    assert sf.sensor_capture(np.full((2, 5, 6, 3), 0.3, np.float32), _spec(), seed=1).shape == (2, 5, 6, 3)


@pytest.mark.parametrize("x", [0.1, 0.3, 0.5, 0.8])
def test_capture_unbiased_and_variance_matches_poisson_gauss(x):
    out = _cap(x).astype(np.float64)
    n = out.size
    expect_var = x / FW + RN ** 2 / FW ** 2 + Q ** 2 / 12.0
    assert abs(out.mean() - x) < 6 * np.sqrt(expect_var / n) + 2e-5
    assert out.var() == pytest.approx(expect_var, rel=0.06)


def test_photon_transfer_curve_slope_is_inverse_full_well():
    xs = np.linspace(0.05, 0.9, 9)
    mean = np.array([_cap(x, seed=11).astype(np.float64).mean() for x in xs])
    var = np.array([_cap(x, seed=11).astype(np.float64).var() for x in xs])
    slope, icpt = np.polyfit(mean, var, 1)
    assert slope == pytest.approx(1.0 / FW, rel=0.05)  # 샷 잡음: 분산 ∝ 신호
    assert abs(icpt) < 3 * (RN ** 2 / FW ** 2 + Q ** 2 / 12.0) + 2e-7


def test_noise_grows_with_smaller_full_well():
    a = _cap(0.4, spec=_spec(full_well_e=5000.0)).astype(np.float64).std()
    b = _cap(0.4, spec=_spec(full_well_e=80000.0)).astype(np.float64).std()
    assert a / b == pytest.approx(4.0, rel=0.12)  # sqrt(80000/5000)


def test_read_noise_dominates_in_shadows():
    spec = _spec(full_well_e=50000.0, read_noise_e=12.0)
    out = _cap(0.002, spec=spec).astype(np.float64)
    expect = np.sqrt(0.002 / 50000.0 + 12.0 ** 2 / 50000.0 ** 2 + (1 / 4095) ** 2 / 12.0)
    assert out.std() == pytest.approx(expect, rel=0.08)


def test_saturation_and_black():
    # 만충전 위: 전자 수는 FW 에서 잘리고(그 뒤에 읽기 잡음), ADC 가 [0,1] 로 자른다 -> 1 - O(RN/FW)
    floor = 1.0 - 6.0 * RN / FW - Q
    for v in (_cap(1.5), _cap(40.0), sf.sensor_capture(np.full((32, 32, 3), 1e9, np.float32), _spec(), seed=1)):
        assert v.max() == 1.0 and v.min() >= floor and v.astype(np.float64).mean() > 1.0 - 1.5 * RN / FW - Q
    # 읽기 잡음은 포화 뒤에도 남는다: 코드 4094.5 아래로 내려가는 화소 비율 = Phi(-0.5*FW/4095/RN)
    from math import erfc, sqrt
    p_below = 0.5 * erfc((0.5 * FW / 4095.0 / RN) / sqrt(2.0))
    assert (_cap(1.5) < 1.0).mean() == pytest.approx(p_below, abs=0.03)
    clean = sf.sensor_capture(np.full((8, 8, 3), 3.0, np.float32), _spec(read_noise_e=0.0), seed=1)
    assert np.all(clean == 1.0)  # 읽기 잡음 0 이면 정확히 1.0
    z = _cap(0.0).astype(np.float64)
    assert z.min() >= 0.0 and z.mean() < 4.0 * RN / FW
    neg = sf.sensor_capture(np.full((8, 8, 3), -0.5, np.float32), _spec(), seed=1)  # 음수는 0으로
    assert neg.min() >= 0.0 and neg.mean() < 4.0 * RN / FW
    one = _cap(1.0).astype(np.float64)
    assert 0.99 < one.mean() <= 1.0  # 클립으로 약간 낮다


def test_noise_off_is_pure_quantization():
    x = np.random.default_rng(5).random((16, 16, 3)).astype(np.float32) * 1.4
    out = sf.sensor_capture(x, _spec(), seed=9, noise=False)
    assert np.array_equal(out, sf.adc_quantize(np.clip(x, 0, 1), BITS))
    assert np.array_equal(out, sf.sensor_capture(x, _spec(), seed=123, frame=7, noise=False))  # 시드와 무관


def test_pixel_and_channel_noise_independent():
    out = _cap(0.5, seed=21).astype(np.float64)
    n = out - out.mean(axis=(0, 1), keepdims=True)
    assert abs(corr(n[:, :-1, 0], n[:, 1:, 0])) < 0.03
    assert abs(corr(n[:-1, :, 1], n[1:, :, 1])) < 0.03
    assert abs(corr(n[..., 0], n[..., 1])) < 0.03
    assert abs(corr(n[..., 1], n[..., 2])) < 0.03


def test_determinism_and_seed_frame_semantics():
    a = _cap(0.5, seed=5, frame=2)
    assert a.tobytes() == _cap(0.5, seed=5, frame=2).tobytes()
    b = _cap(0.5, seed=6, frame=2)
    c = _cap(0.5, seed=5, frame=3)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)
    na, nb, nc = (v.astype(np.float64) - 0.5 for v in (a, b, c))
    assert abs(corr(na, nb)) < 0.03 and abs(corr(na, nc)) < 0.03  # 프레임·시드 간 무상관
    big = (1 << 64) - 1
    assert sf.sensor_capture(np.full((4, 4, 3), 0.5, np.float32), _spec(), seed=big, frame=(1 << 32) - 1).shape == (4, 4, 3)


def test_does_not_touch_global_rng():
    np.random.seed(1234)
    before = np.random.get_state()[1].copy()
    _cap(0.5)
    sf.adc_quantize(np.zeros(3, np.float32), 8)
    assert np.array_equal(np.random.get_state()[1], before)


def test_no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = np.array([[[0.0, 1e-9, 1e9]]], np.float32)
        sf.sensor_capture(x, _spec(), seed=1)
        sf.sensor_capture(x, _spec(), seed=1, noise=False)


def test_capture_errors():
    ok = np.full((4, 4, 3), 0.3, np.float32)
    with pytest.raises(TypeError):
        sf.sensor_capture(ok.astype(np.float16), _spec())
    with pytest.raises(TypeError):
        sf.sensor_capture((ok * 255).astype(np.uint8), _spec())
    for bad in (np.nan, np.inf, -np.inf):
        x = ok.copy()
        x[0, 0, 0] = bad
        with pytest.raises(ValueError):
            sf.sensor_capture(x, _spec())
    for kw in (dict(seed=-1), dict(frame=-1), dict(seed=1 << 64), dict(frame=1 << 32)):
        with pytest.raises(ValueError):
            sf.sensor_capture(ok, _spec(), **kw)
    for kw in (dict(seed=1.5), dict(frame=0.0), dict(seed=True), dict(frame="0")):
        with pytest.raises(TypeError):
            sf.sensor_capture(ok, _spec(), **kw)
    with pytest.raises(TypeError):
        sf.sensor_capture(ok, _spec(), 1)  # seed 는 키워드 전용


# ----------------------------------------------------------------------------------------------- θ 매핑
def test_sensor_spec_from_theta_pixel_pitch_physics():
    from buc.core.contracts import Theta
    # 피치 = 센서폭(mm)*1000/폭(px) [um], 만충전 = clip(1500*피치^2, 1500, 100000)
    for sw, wpx in ((12.0, 1920), (8.8, 1280), (17.3, 2048), (5.79, 1920)):
        s = sf.sensor_spec_from_theta(Theta(sensor_width_mm=sw), wpx)
        pitch = sw * 1000.0 / wpx
        assert s.full_well_e == pytest.approx(float(np.clip(1500.0 * pitch ** 2, 1500.0, 100000.0)), rel=1e-9)
        assert s.read_noise_e == sf.READ_NOISE_E and s.bits == sf.ADC_BITS
    big = sf.sensor_spec_from_theta(Theta(sensor_width_mm=24.89), 128)   # 피치 194 um -> 상한
    small = sf.sensor_spec_from_theta(Theta(sensor_width_mm=5.79), 1920)  # 슈퍼8: 피치 3 um
    assert big.full_well_e == 100000.0
    assert small.full_well_e == pytest.approx(1500.0 * (5.79 * 1000 / 1920) ** 2, rel=1e-9)
    assert big.full_well_e > 3 * small.full_well_e  # 작은 센서가 더 시끄럽다
    tiny = sf.sensor_spec_from_theta(Theta(sensor_width_mm=4.8), 16384)  # 피치 0.29 um -> 하한
    assert tiny.full_well_e == 1500.0
    with pytest.raises(ValueError):
        sf.sensor_spec_from_theta(Theta(), 0)
