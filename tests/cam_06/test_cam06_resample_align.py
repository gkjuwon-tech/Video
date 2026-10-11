"""CAM-06: 균일 리샘플(200 Hz) · 틈 분할 · 기기 간 시각 정렬(자이로 노름 상호상관, 부표본)."""
import math

import numpy as np
import pytest

from cam06_helpers import G0, imu, make_clip, make_raw, make_signal, tone

RATE = 200.0


def _edge(n, rate=RATE, sec=0.25):
    k = int(sec * rate)
    return slice(k, n - k)


# ------------------------------------------------------------------ 리샘플
def test_resample_upsample_jittered_timestamps_and_overlap_grid():
    m = imu()
    rng = np.random.default_rng(5)
    n_g, n_a = 800, 700
    t_g = 1.0 + np.arange(n_g) / 100.0 + rng.uniform(-5e-4, 5e-4, n_g)              # 100 Hz, 타임스탬프 지터
    t_a = 1.3 + np.arange(n_a) / 100.0 + rng.uniform(-5e-4, 5e-4, n_a)              # 늦게 시작, 일찍 끝남
    g = tone(t_g, 5.0, 2.0, 0.4, axis=1)
    a = G0 + tone(t_a, 3.0, 4.0, 1.1, axis=2)
    raw = make_raw(t_g, g, t_a, a, source="gpmf", device="camA")
    clip = m.resample_uniform(raw)                                                  # 기본 200 Hz
    start, end = max(t_g[0], t_a[0]), min(t_g[-1], t_a[-1])
    n = int(math.floor((end - start) * 200.0 + 1e-9)) + 1
    assert clip.rate_hz == 200.0 and clip.n == n and clip.t0 == pytest.approx(start, abs=1e-12)
    assert clip.t[-1] <= end + 1e-9 and clip.t[-1] > end - 1 / 200.0 - 1e-9
    e = _edge(n)
    assert np.max(np.abs(clip.gyro[e] - tone(clip.t, 5.0, 2.0, 0.4, axis=1)[e])) < 2e-3
    assert np.max(np.abs(clip.accel[e] - (G0 + tone(clip.t, 3.0, 4.0, 1.1, axis=2))[e])) < 3e-3
    assert clip.source == "gpmf" and clip.device == "camA"
    c2 = m.resample_uniform(raw, 100.0)
    assert c2.rate_hz == 100.0 and np.allclose(c2.t, start + np.arange(c2.n) / 100.0, atol=1e-12)


def test_resample_downsample_is_antialiased():
    m = imu()
    t = np.arange(4000) / 400.0                                                     # 400 Hz, 10 s
    g = tone(t, 10.0, 1.0, 0.2, axis=0) + tone(t, 150.0, 1.0, 0.7, axis=1)          # 150 Hz 성분은 200 Hz 표본화로 접힌다(50 Hz)
    a = G0 + tone(t, 7.0, 2.0, 0.9, axis=2)
    clip = m.resample_uniform(make_raw(t, g, t, a))
    assert clip.rate_hz == 200.0 and clip.n == 2000
    e = _edge(clip.n)
    assert np.max(np.abs(clip.gyro[e, 0] - tone(clip.t, 10.0, 1.0, 0.2)[e, 0])) < 2e-3          # 통과대역은 보존
    assert np.max(np.abs(clip.gyro[e, 1])) < 0.05                                               # 대역 밖은 >26 dB 감쇠
    assert np.max(np.abs(clip.accel[e, 2] - (G0[2] + tone(clip.t, 7.0, 2.0, 0.9, axis=2)[e, 2]))) < 4e-3


def test_resample_same_rate_is_identity_on_grid_points():
    m = imu()
    t = 2.0 + np.arange(1000) / 200.0
    g = make_signal(3, amp=2.0)(t)
    a = G0 + make_signal(4, amp=2.0)(t)
    clip = m.resample_uniform(make_raw(t, g, t, a))
    assert clip.n == 1000 and clip.t0 == pytest.approx(2.0)
    assert np.max(np.abs(clip.gyro - g)) < 1e-6 and np.max(np.abs(clip.accel - a)) < 1e-6


def test_resample_t0_argument_and_errors():
    m = imu()
    t = np.arange(1000) / 200.0
    raw = make_raw(t, make_signal(1)(t), t, G0 + make_signal(2)(t))
    c = m.resample_uniform(raw, t0=1.0)
    assert c.t0 == 1.0 and c.n == int(math.floor((t[-1] - 1.0) * 200 + 1e-9)) + 1
    with pytest.raises(ValueError):
        m.resample_uniform(raw, t0=-0.5)                                            # 겹치는 구간 밖
    with pytest.raises(ValueError):
        m.resample_uniform(raw, t0=t[-1] + 1.0)
    with pytest.raises(ValueError):
        m.resample_uniform(raw, 0.0)
    with pytest.raises(ValueError):
        m.resample_uniform(make_raw(t, make_signal(1)(t)))                          # 가속도 없음
    far = make_raw(t, make_signal(1)(t), t + 100.0, G0 + make_signal(2)(t))
    with pytest.raises(ValueError):
        m.resample_uniform(far)                                                     # 자이로·가속도 구간이 안 겹침


def test_gap_detection_and_split_segments():
    m = imu()
    t1 = np.arange(0, 400) / 200.0                                                  # 0 .. 1.995
    t2 = 3.0 + np.arange(0, 300) / 200.0                                            # 틈 1.0 s
    t = np.concatenate([t1, t2])
    g = make_signal(8, amp=1.0)(t)
    a = G0 + make_signal(9, amp=1.0)(t)
    raw = make_raw(t, g, t, a, source="csv", device="p")
    with pytest.raises(ValueError):
        m.resample_uniform(raw)                                                     # 기본 max_gap_s = 0.1 초과
    segs = m.split_segments(raw)
    assert len(segs) == 2
    assert segs[0].gyro.v.shape[0] + segs[1].gyro.v.shape[0] == 700
    assert segs[0].gyro.t[-1] == pytest.approx(t1[-1]) and segs[1].gyro.t[0] == pytest.approx(3.0)
    assert np.array_equal(segs[1].accel.v, a[400:]) and segs[0].source == "csv" and segs[0].device == "p"
    for s in segs:
        c = m.resample_uniform(s)                                                   # 조각은 각각 리샘플 가능
        assert c.n >= 299
    assert len(m.split_segments(raw, max_gap_s=2.0)) == 1
    assert m.resample_uniform(raw, max_gap_s=2.0).n > 0
    # 가속도에만 틈이 있어도 자이로까지 같은 지점에서 나눈다
    ta = np.concatenate([np.arange(0, 400) / 200.0, 3.0 + np.arange(0, 300) / 200.0])
    tg = np.arange(0, 1400) / 200.0
    seg2 = m.split_segments(make_raw(tg, make_signal(1)(tg), ta, G0 + make_signal(2)(ta)))
    mid = 0.5 * (399 / 200.0 + 3.0)                                                 # 틈의 중점에서 자른다
    assert len(seg2) == 2 and seg2[0].gyro.t[-1] < mid < seg2[1].gyro.t[0]
    assert seg2[0].gyro.v.shape[0] + seg2[1].gyro.v.shape[0] == 1400
    assert len(m.split_segments(make_raw(t1, g[:400], t1, a[:400]))) == 1


# ------------------------------------------------------------------ 정렬
def _pair(offset, *, other_t0=0.3, n_ref=12000, n_other=11000, noise=0.0, seed=1):
    """ref 시계의 해석적 신호 f. other 는 자기 시계 t_o 에 R f(t_o + offset) 을 기록(= t_ref = t_other + offset)."""
    f = make_signal(seed, amp=1.6, fmax=5.0)
    ref = make_clip(n_ref, gyro=f(np.arange(n_ref) / RATE), seed=seed + 50, source="gpmf")
    t_o = other_t0 + np.arange(n_other) / RATE
    ax = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    th = math.radians(40.0)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    g_o = f(t_o + offset) @ R.T
    if noise:
        g_o = g_o + np.random.default_rng(seed + 7).normal(0, noise, g_o.shape)
    other = make_clip(n_other, t0=other_t0, gyro=g_o, seed=seed + 80, source="csv")
    return ref, other, R


@pytest.mark.parametrize("offset", [0.137, -0.412, 0.0, 1.234, -1.5])
def test_estimate_offset_recovers_known_offset_subsample(offset):
    m = imu()
    ref, other, _ = _pair(offset)
    est = m.estimate_offset(ref, other)                                             # 기본 max_lag_s = 2
    assert est.offset_s == pytest.approx(offset, abs=1.5e-3)                        # 표본 간격 5 ms 의 30% 이내
    assert est.score > 0.99
    assert abs(est.offset_s - ((ref.t0 - other.t0) + est.lag_samples / RATE)) <= 0.5 / RATE + 1e-9


def test_estimate_offset_with_sensor_noise_and_distant_clock_origins():
    m = imu()
    ref, other, _ = _pair(0.2, other_t0=17.25, noise=0.03, seed=2)                  # other 표본 시각 원점이 17.25 s 어긋나 있어도 된다
    est = m.estimate_offset(ref, other, max_lag_s=1.0)                              # max_lag_s 는 |offset_s| 의 한계(t0 차이와 무관)
    assert est.offset_s == pytest.approx(0.2, abs=3e-3)
    assert est.score > 0.9


def test_estimate_offset_unrelated_signals_low_score_and_lag_bound():
    m = imu()
    ref, _, _ = _pair(0.0, seed=1)
    other = make_clip(11000, t0=0.0, gyro=make_signal(99, amp=1.6, fmax=5.0)(np.arange(11000) / RATE), seed=3)
    est = m.estimate_offset(ref, other)
    assert est.score < 0.5
    ref2, other2, _ = _pair(1.234)
    est2 = m.estimate_offset(ref2, other2, max_lag_s=0.5)                           # 정답이 탐색 범위 밖
    assert abs(est2.offset_s) <= 0.5 + 1.0 / RATE + 1e-9
    assert est2.score < 0.9


def test_estimate_offset_validation_and_purity():
    m = imu()
    ref, other, _ = _pair(0.1)
    g0, g1 = ref.gyro.copy(), other.gyro.copy()
    e1 = m.estimate_offset(ref, other)
    e2 = m.estimate_offset(ref, other)
    assert e1 == e2 and np.array_equal(ref.gyro, g0) and np.array_equal(other.gyro, g1)
    other100 = make_clip(5000, rate=100.0, t0=0.0)
    with pytest.raises(ValueError):
        m.estimate_offset(ref, other100)                                            # 속도가 다르면 먼저 리샘플
    with pytest.raises(ValueError):
        m.estimate_offset(ref, other, max_lag_s=0.0)
    with pytest.raises(ValueError):
        m.estimate_offset(ref, other, max_lag_s=-1.0)
    short = make_clip(600)                                                          # 3 s: 겹침이 5 s 미만
    with pytest.raises(ValueError):
        m.estimate_offset(short, short)


def test_shift_clip():
    m = imu()
    c = make_clip(100, t0=1.5, meta={"a": 1})
    s = m.shift_clip(c, -0.25)
    assert s.t0 == pytest.approx(1.25) and np.array_equal(s.gyro, c.gyro) and s.meta == c.meta and c.t0 == 1.5


def test_align_pair_resamples_other_onto_ref_grid():
    m = imu()
    offset = 0.2
    ref, other, R = _pair(offset)
    ref_cut, oa, est = m.align_pair(ref, other)
    assert est.offset_s == pytest.approx(offset, abs=1.5e-3)
    assert ref_cut.n == oa.n and ref_cut.rate_hz == oa.rate_hz == RATE
    assert ref_cut.t0 == pytest.approx(oa.t0, abs=1e-12)
    i0 = int(round((ref_cut.t0 - ref.t0) * RATE))
    assert i0 >= 0 and abs((ref_cut.t0 - ref.t0) * RATE - i0) < 1e-6
    assert np.array_equal(ref_cut.gyro, ref.gyro[i0:i0 + ref_cut.n])               # ref 는 잘라내기만 한다
    # 겹침: other 가 덮는 ref 시각 [0.3 + 0.2, 0.3 + 0.2 + 54.995] ∩ ref [0, 59.995]
    assert ref_cut.t0 == pytest.approx(0.5, abs=1.5 / RATE)
    assert abs(ref_cut.n - 11000) <= 3
    # 같은 물리 운동을 보므로 회전 불변 노름이 일치한다
    d = np.linalg.norm(ref_cut.gyro, axis=1) - np.linalg.norm(oa.gyro, axis=1)
    e = _edge(ref_cut.n)
    assert math.sqrt(np.mean(d[e] ** 2)) < 0.02
    # other 의 각 표본은 자기 프레임 그대로(회전 안 함): R 로 되돌리면 ref 자이로와 일치
    back = oa.gyro @ R
    assert math.sqrt(np.mean((back[e] - ref_cut.gyro[e]) ** 2)) < 0.03
    assert oa.source == "csv" and ref_cut.source == "gpmf"
