"""CAM-06: 자료형(Series/RawImu/ImuClip) · GPMF 파서 · 폰/공개 데이터 CSV 파서."""
import dataclasses
import io
import math

import numpy as np
import pytest

from cam06_helpers import (G0, TUM_HEADER, generic_text, imu, make_clip, make_gpmf, make_signal, tum_text)

RNG = np.random.default_rng(7)


def _sig(n, rate=200.0, seed=3, amp=1.0):
    t = np.arange(n) / rate
    return make_signal(seed, amp=amp)(t)


# ------------------------------------------------------------------ 자료형
def test_constants():
    m = imu()
    assert m.RATE_HZ == 200.0 and m.AUG_FACTOR == 5.5 and m.G_STD == pytest.approx(9.80665)
    assert m.MAX_GAP_S == pytest.approx(0.1) and m.WINDOW_LEN == 400 and m.GRAVITY_LP_HZ == pytest.approx(0.5)
    assert tuple(m.RIG_KINDS) == ("lowpass", "highpass")


def test_series_validation_and_readonly():
    m = imu()
    t = np.arange(10) * 0.005
    v = _sig(10)
    s = m.Series(t, v)
    assert s.t.dtype == np.float64 and s.v.shape == (10, 3)
    assert not s.t.flags.writeable and not s.v.flags.writeable
    v2 = v.copy()
    s2 = m.Series(t, v2)
    v2[0, 0] = 99.0                                     # 입력 복사본이어야 한다
    assert s2.v[0, 0] != 99.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.t = t
    bad_t = t.copy()
    bad_t[5] = bad_t[4]
    for args in [(bad_t, v), (t[::-1].copy(), v), (t, v[:, :2]), (t[:5], v), (t[:1], v[:1]),
                 (np.where(np.arange(10) == 3, np.nan, t), v), (t, np.where(np.arange(30).reshape(10, 3) == 4, np.inf, v))]:
        with pytest.raises(ValueError):
            m.Series(*args)


def test_imuclip_fields_and_validation():
    m = imu()
    c = make_clip(1000, t0=3.5, meta={"dataset": "TUM VI", "license": "CC-BY-4.0"})
    assert c.n == 1000 and c.rate_hz == 200.0 and c.t0 == 3.5
    assert c.duration_s == pytest.approx(5.0)
    assert np.allclose(c.t, 3.5 + np.arange(1000) / 200.0, atol=1e-12)
    arr = c.as_array()
    assert arr.shape == (1000, 6) and np.array_equal(arr[:, :3], c.gyro) and np.array_equal(arr[:, 3:], c.accel)
    assert not c.gyro.flags.writeable and not c.accel.flags.writeable
    assert c.meta == {"dataset": "TUM VI", "license": "CC-BY-4.0"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.t0 = 0.0
    g, a = c.gyro, c.accel
    with pytest.raises(ValueError):
        m.ImuClip(0.0, 200.0, g, a[:-1])
    with pytest.raises(ValueError):
        m.ImuClip(0.0, 200.0, g[:, :2], a[:, :2])
    with pytest.raises(ValueError):
        m.ImuClip(0.0, 0.0, g, a)
    with pytest.raises(ValueError):
        m.ImuClip(0.0, 200.0, g[:0], a[:0])
    with pytest.raises(ValueError):
        m.ImuClip(0.0, 200.0, np.where(g > 100, 0, np.nan), a)
    with pytest.raises(TypeError):
        m.ImuClip(0.0, 200.0, g, a, meta={"x": object()})            # meta 는 JSON 가능해야 한다
    meta = {"k": [1, 2]}
    c2 = m.ImuClip(0.0, 200.0, g, a, meta=meta)
    meta["k"].append(3)
    assert c2.meta == {"k": [1, 2]}                                # 깊은 복사


def test_imuclip_slice():
    c = make_clip(1000, t0=2.0)
    s = c.slice(100, 300)
    assert s.n == 200 and s.t0 == pytest.approx(2.0 + 100 / 200.0) and s.rate_hz == c.rate_hz
    assert np.array_equal(s.gyro, c.gyro[100:300]) and np.array_equal(s.accel, c.accel[100:300])
    assert s.source == c.source and s.device == c.device
    with pytest.raises(ValueError):
        c.slice(500, 500)


# ------------------------------------------------------------------ GPMF
def test_gpmf_basic_values_times_and_skipped_streams():
    m = imu()
    g = _sig(950, seed=3, amp=2.0)                                   # 950 = 4.75 블록
    a = G0 + _sig(950, seed=4, amp=3.0)
    data, deq_g, deq_a, st = make_gpmf(g, a)
    raw = m.parse_gpmf(data)
    assert raw.source == "gpmf" and raw.device == "gopro"
    assert raw.gyro.v.shape == (950, 3) and raw.accel is not None and raw.accel.v.shape == (950, 3)
    assert np.allclose(raw.gyro.v, deq_g, atol=1e-12, rtol=0)        # SCAL 로 나눈 값, 축 순서 그대로
    assert np.allclose(raw.accel.v, deq_a, atol=1e-12, rtol=0)
    assert np.max(np.abs(deq_g - g)) <= 0.5 / 64.0 + 1e-12            # (테스트 자체 점검: 양자화 오차)
    span = (st[-1] - st[0]) / 1e6
    rate = 200.0 * (len(st) - 1) / span
    assert rate == pytest.approx(200.0, abs=1e-9)
    expect = np.concatenate([st[k] / 1e6 + np.arange(200 if k < 4 else 150) / rate for k in range(5)])
    assert np.allclose(raw.gyro.t, expect, atol=1e-9, rtol=0)
    assert np.allclose(raw.accel.t, expect, atol=1e-9, rtol=0)
    assert raw.meta["truncated"] is False
    assert raw.meta["gyro_rate_hz"] == pytest.approx(200.0, abs=1e-6)
    assert raw.meta["accel_rate_hz"] == pytest.approx(200.0, abs=1e-6)
    with pytest.raises(TypeError):
        m.parse_gpmf("not bytes")
    assert m.parse_gpmf(bytearray(data)).gyro.v.shape == (950, 3)    # bytes 계열 허용


def test_gpmf_fitted_rate_when_sensor_clock_is_not_nominal():
    """센서가 199.8 Hz 로 도는 카메라: 블록(200표본) STMP 간격이 1.001001 s. 표본 시각은 블록 STMP 에 못박고 속도는 맞춘다."""
    m = imu()
    n = 1400
    g = _sig(n, rate=199.8, seed=5)
    data, deq_g, _, st = make_gpmf(g, None, block_s=1.001001, gps=False)
    raw = m.parse_gpmf(data)
    assert raw.accel is None
    span = (st[-1] - st[0]) / 1e6
    rate = 200.0 * (len(st) - 1) / span
    assert rate == pytest.approx(199.8, rel=2e-6)
    assert raw.meta["gyro_rate_hz"] == pytest.approx(rate, rel=1e-9)
    assert "accel_rate_hz" not in raw.meta
    for k in range(len(st)):
        assert raw.gyro.t[k * 200] == pytest.approx(st[k] / 1e6, abs=1e-9)        # 블록 첫 표본 = STMP
    j = np.arange(200)
    assert np.allclose(raw.gyro.t[:200], st[0] / 1e6 + j / rate, atol=1e-9)
    assert np.allclose(raw.gyro.v, deq_g, atol=1e-12)


def test_gpmf_stmp_jitter_anchors_each_block():
    m = imu()
    g = _sig(2000, seed=6)
    a = G0 + _sig(2000, seed=7)
    jit = RNG.integers(-300, 300, 10)
    data, _, _, st = make_gpmf(g, a, jitter_us=jit)
    raw = m.parse_gpmf(data)
    assert len(st) == 10
    for k in range(10):
        assert raw.gyro.t[k * 200] == pytest.approx(st[k] / 1e6, abs=1e-9)
        assert raw.accel.t[k * 200] == pytest.approx(st[k] / 1e6, abs=1e-9)
    assert np.all(np.diff(raw.gyro.t) > 0)


def test_gpmf_different_gyro_and_accel_rates_and_per_axis_scal():
    m = imu()
    g = _sig(2000, rate=400.0, seed=8)                       # 400 Hz, 5 s
    a = G0 + _sig(1000, rate=200.0, seed=9)                  # 200 Hz, 5 s
    data, deq_g, deq_a, st = make_gpmf(g, a, rate_g=400.0, rate_a=200.0, scal_g=np.array([40.0, 50.0, 60.0]),
                                       scal_a=np.array([400.0, 410.0, 420.0]))
    raw = m.parse_gpmf(data)
    assert raw.gyro.v.shape == (2000, 3) and raw.accel.v.shape == (1000, 3)
    assert np.allclose(raw.gyro.v, deq_g, atol=1e-12) and np.allclose(raw.accel.v, deq_a, atol=1e-12)
    assert raw.meta["gyro_rate_hz"] == pytest.approx(400.0, abs=1e-6)
    assert raw.meta["accel_rate_hz"] == pytest.approx(200.0, abs=1e-6)
    assert raw.gyro.t[-1] == pytest.approx(st[-1] / 1e6 + 399 / 400.0, abs=1e-6)


def test_gpmf_int32_samples_and_device_name():
    m = imu()
    g = _sig(600, seed=10)
    data, deq_g, _, _ = make_gpmf(g, None, typ="l", scal_g=1.0e5, gps=False)
    raw = m.parse_gpmf(data, device="hero11-A")
    assert raw.device == "hero11-A" and raw.source == "gpmf"
    assert np.allclose(raw.gyro.v, deq_g, atol=1e-12)


def test_gpmf_without_stmp_or_single_block_needs_rate():
    m = imu()
    g = _sig(400, seed=11)
    a = G0 + _sig(400, seed=12)
    data, deq_g, _, _ = make_gpmf(g, a, with_stmp=False)
    with pytest.raises(ValueError):
        m.parse_gpmf(data)
    raw = m.parse_gpmf(data, rate_hz=200.0)                  # STMP 없으면 t = i / rate (시작 0)
    assert np.allclose(raw.gyro.t, np.arange(400) / 200.0, atol=1e-12)
    assert np.allclose(raw.gyro.v, deq_g, atol=1e-12)
    one, _, _, st = make_gpmf(g[:200], None, gps=False)       # STMP 는 있지만 블록이 하나뿐
    with pytest.raises(ValueError):
        m.parse_gpmf(one)
    raw1 = m.parse_gpmf(one, rate_hz=200.0)
    assert np.allclose(raw1.gyro.t, st[0] / 1e6 + np.arange(200) / 200.0, atol=1e-9)
    # rate_hz 를 명시하면 그 값이 STMP 로 맞춘 속도보다 우선한다
    data2, *_ = make_gpmf(g, a)
    r2 = m.parse_gpmf(data2, rate_hz=250.0)
    assert np.allclose(np.diff(r2.gyro.t[:200]), 0.004, atol=1e-9)


def test_gpmf_truncated_file_drops_incomplete_devc():
    m = imu()
    g = _sig(1000, seed=13)
    a = G0 + _sig(1000, seed=14)
    data, deq_g, deq_a, _ = make_gpmf(g, a)
    cut = data[:-400]                                        # 마지막 DEVC 한가운데서 잘림
    raw = m.parse_gpmf(cut)
    assert raw.meta["truncated"] is True
    assert raw.gyro.v.shape == (800, 3) and raw.accel.v.shape == (800, 3)
    assert np.allclose(raw.gyro.v, deq_g[:800], atol=1e-12)
    assert m.parse_gpmf(data).meta["truncated"] is False


def test_gpmf_error_cases():
    m = imu()
    with pytest.raises(ValueError):
        m.parse_gpmf(b"")
    with pytest.raises(ValueError):
        m.parse_gpmf(b"\x00" * 64)
    only_accel, *_ = make_gpmf(_sig(400), None, gps=True)    # GYRO 가 있어야 한다: 아래는 GYRO 를 ACCL 로 바꾼 파일
    assert b"GYRO" in only_accel
    with pytest.raises(ValueError):
        m.parse_gpmf(only_accel.replace(b"GYRO", b"GYRX"))
    g = _sig(1000, seed=15)
    jit = np.zeros(5, dtype=int)
    jit[2] = -1_500_000                                      # 블록 2 의 STMP 가 블록 1 보다 앞선다
    bad, *_ = make_gpmf(g, None, jitter_us=jit, gps=False)
    with pytest.raises(ValueError):
        m.parse_gpmf(bad)


# ------------------------------------------------------------------ CSV: TUM VI 형식
def _tum_rows(n=40, step_ns=5_000_000, t_ns0=1403636579758555392):
    t_ns = t_ns0 + step_ns * np.arange(n)
    return t_ns, _sig(n, seed=21), G0 + _sig(n, seed=22)


def test_csv_tumvi_autodetect_and_column_order():
    m = imu()
    t_ns, g, a = _tum_rows()
    raw = m.parse_phone_csv(tum_text(t_ns, g, a))
    assert raw.source == "tumvi" and raw.device == "phone"
    assert np.allclose(raw.gyro.v, g, atol=1e-12) and np.allclose(raw.accel.v, a, atol=1e-12)    # w 먼저, a 나중
    assert np.allclose(raw.gyro.t, t_ns / 1e9, atol=1e-6, rtol=0)
    assert np.allclose(np.diff(raw.gyro.t), 0.005, atol=1e-6)
    assert np.array_equal(raw.gyro.t, raw.accel.t)
    assert m.parse_phone_csv(tum_text(t_ns, g, a), device="cam0-imu").device == "cam0-imu"


def test_csv_input_kinds_comments_and_blank_lines(tmp_path):
    m = imu()
    t_ns, g, a = _tum_rows(20)
    txt = tum_text(t_ns, g, a)
    lines = txt.splitlines()
    noisy = "\n".join(lines[:5] + ["# 중간 주석", ""] + lines[5:]) + "\n\n"
    ref = m.parse_phone_csv(txt)
    for src in (noisy, noisy.encode(), io.StringIO(noisy)):
        r = m.parse_phone_csv(src)
        assert np.array_equal(r.gyro.v, ref.gyro.v) and np.array_equal(r.gyro.t, ref.gyro.t)
    p = tmp_path / "imu0.csv"
    p.write_text(txt)
    for src in (p, str(p)):                                  # 줄바꿈이 없는 str 은 경로로 본다
        r = m.parse_phone_csv(src)
        assert np.array_equal(r.accel.v, ref.accel.v)


# ------------------------------------------------------------------ CSV: 일반 형식 · 단위
def test_csv_generic_units_conversion():
    m = imu()
    t = 100.0 + 0.01 * np.arange(30)
    g_deg = _sig(30, seed=31) * 40.0
    a_g = _sig(30, seed=32) * 0.3 + np.array([0.0, -1.0, 0.0])
    txt = generic_text(t, g_deg, a_g, tname="t", unit_scale=1000.0)          # ms 로 기록
    raw = m.parse_phone_csv(txt, time_unit="ms", gyro_unit="deg/s", accel_unit="g")
    assert raw.source == "csv"
    assert np.allclose(raw.gyro.t, t, atol=1e-9)
    assert np.allclose(raw.gyro.v, np.deg2rad(g_deg), atol=1e-12)
    assert np.allclose(raw.accel.v, a_g * 9.80665, atol=1e-12)
    for unit, scale in (("s", 1.0), ("us", 1e6), ("ns", 1e9)):
        r = m.parse_phone_csv(generic_text(t, g_deg, a_g, tname="time", unit_scale=scale), time_unit=unit)
        assert np.allclose(r.gyro.t, t, atol=1e-6)
    r = m.parse_phone_csv(generic_text(t, g_deg, a_g, tname="Timestamp"))     # 이름은 대소문자 무시, 기본 단위 s, rad/s, m/s²
    assert np.allclose(r.gyro.v, g_deg, atol=1e-12) and np.allclose(r.accel.v, a_g, atol=1e-12)


def test_csv_columns_override_and_sensor_selection_and_merge():
    m = imu()
    t = 0.01 * np.arange(25)
    g = _sig(25, seed=33)
    a = G0 + _sig(25, seed=34)
    with pytest.raises(ValueError):                          # 구분자는 쉼표만: 세미콜론 파일은 열을 못 찾는다
        m.parse_phone_csv("t;gx;gy;gz\n0;1;2;3\n1;1;2;3\n")
    head = "Ts,RX,RY,RZ,UX,UY,UZ,extra\n"
    body = "".join(",".join(repr(float(x)) for x in [t[i], *g[i], *a[i]]) + ",9\n" for i in range(25))
    cols = {"t": "ts", "gx": "rx", "gy": "ry", "gz": "rz", "ax": "ux", "ay": "uy", "az": "uz"}
    raw = m.parse_phone_csv(head + body, columns=cols)       # 열 이름은 대소문자 무시
    assert np.allclose(raw.gyro.v, g) and np.allclose(raw.accel.v, a) and np.allclose(raw.gyro.t, t)
    # 센서별 파일 (안드로이드 센서 로거처럼 자이로·가속도가 따로, 시각도 따로)
    gy = m.parse_phone_csv(generic_text(t, g), device="p1")
    assert gy.accel is None and gy.gyro.v.shape == (25, 3)
    acc_txt = generic_text(t + 0.003, a, None).replace("gx,gy,gz", "ax,ay,az")
    ac = m.parse_phone_csv(acc_txt, sensor="accel", device="p1")
    assert ac.gyro is None and np.allclose(ac.accel.v, a) and np.allclose(ac.accel.t, t + 0.003)
    both = m.merge_raw(gy, ac)
    assert np.array_equal(both.gyro.v, g) and np.array_equal(both.gyro.t, t)
    assert np.array_equal(both.accel.v, a) and np.allclose(both.accel.t, t + 0.003)
    assert both.device == "p1" and both.source == gy.source
    only_g = m.parse_phone_csv(head + body, columns=cols, sensor="gyro")      # 가속도 열이 있어도 무시
    assert only_g.accel is None and np.allclose(only_g.gyro.v, g)
    with pytest.raises(ValueError):
        m.merge_raw(gy, gy)                                  # accel_raw 에 가속도가 없다
    with pytest.raises(ValueError):
        m.merge_raw(ac, ac)                                  # gyro_raw 에 자이로가 없다
    with pytest.raises(ValueError):
        m.parse_phone_csv(generic_text(t, g), sensor="accel")                # 가속도 열 없음
    with pytest.raises(ValueError):
        m.parse_phone_csv(generic_text(t, g), sensor="mag")


def test_csv_error_cases():
    m = imu()
    t = 0.01 * np.arange(10)
    g = _sig(10, seed=35)
    a = G0 + _sig(10, seed=36)
    ok = generic_text(t, g, a)
    lines = ok.splitlines()
    with pytest.raises(ValueError):
        m.parse_phone_csv(ok.replace("gy,", "gq,", 1))                         # 열 없음
    with pytest.raises(ValueError):
        m.parse_phone_csv("\n".join(lines[:5] + [lines[5].replace(",", ",abc,", 1)] + lines[6:]))   # 숫자 아님
    with pytest.raises(ValueError):
        m.parse_phone_csv("\n".join(lines[:5] + [lines[4]] + lines[6:]))       # 시각이 늘지 않음
    with pytest.raises(ValueError):
        m.parse_phone_csv("\n".join(lines[:5] + [lines[5].replace(lines[5].split(",")[1], "nan")] + lines[6:]))
    with pytest.raises(ValueError):
        m.parse_phone_csv("\n".join(lines[:2]))                                # 행 1개
    with pytest.raises(ValueError):
        m.parse_phone_csv(ok, gyro_unit="rpm")
    with pytest.raises(ValueError):
        m.parse_phone_csv(ok, accel_unit="mg")
    with pytest.raises(ValueError):
        m.parse_phone_csv(ok, time_unit="min")
    with pytest.raises(ValueError):
        m.parse_phone_csv(ok, fmt="xml")
    assert TUM_HEADER.startswith("#timestamp [ns]")                              # 테스트 자체 점검
    assert math.isfinite(m.parse_phone_csv(ok).gyro.v.sum())
