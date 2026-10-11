"""CAM-06: 증강 기본 연산(프레임 회전 · 시간 늘이기 · 리그 전달함수 · 센서 열화) · AugmentConfig · 기록형 apply_augmentation."""
import dataclasses
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cam06_helpers import G0, imu, lsq_amp, make_clip, make_signal, tone

RATE = 200.0


def _mid(n, sec=1.0):
    k = int(sec * RATE)
    return slice(k, n - k)


def _tone_clip(n=4000, g_freq=3.0, g_amp=1.5, a_freq=4.0, a_amp=3.0, **kw):
    t = np.arange(n) / RATE
    return make_clip(n, gyro=tone(t, g_freq, g_amp, 0.3, axis=0), accel=G0 + tone(t, a_freq, a_amp, 1.0, axis=2), **kw)


# ------------------------------------------------------------------ rotate_frame
def test_rotate_frame_maps_sensor_axes_and_preserves_norms():
    m = imu()
    c = make_clip(500, meta={"k": 1}, t0=2.0)
    Rz90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    one = m.ImuClip(0.0, RATE, np.array([[1.0, 0, 0]] * 3), np.array([[0, 0, 1.0]] * 3))
    r = m.rotate_frame(one, Rz90)
    assert np.allclose(r.gyro, [[0, 1, 0]] * 3, atol=1e-12) and np.allclose(r.accel, [[0, 0, 1]] * 3, atol=1e-12)
    R = Rotation.from_rotvec([0.3, -0.8, 0.5]).as_matrix()
    g0, a0 = c.gyro.copy(), c.accel.copy()
    r = m.rotate_frame(c, R)
    assert np.allclose(r.gyro, c.gyro @ R.T, atol=1e-12) and np.allclose(r.accel, c.accel @ R.T, atol=1e-12)
    assert np.allclose(np.linalg.norm(r.gyro, axis=1), np.linalg.norm(c.gyro, axis=1), atol=1e-12)
    assert np.allclose(np.linalg.norm(r.accel, axis=1), np.linalg.norm(c.accel, axis=1), atol=1e-12)
    assert r.t0 == c.t0 and r.rate_hz == c.rate_hz and r.meta == c.meta and r.n == c.n
    assert np.array_equal(c.gyro, g0) and np.array_equal(c.accel, a0)                       # 입력 불변
    back = m.rotate_frame(r, R.T)
    assert np.allclose(back.gyro, c.gyro, atol=1e-12) and np.allclose(back.accel, c.accel, atol=1e-12)
    assert np.allclose(m.rotate_frame(c, np.eye(3)).gyro, c.gyro, atol=0)


def test_rotate_frame_rejects_non_rotations():
    m = imu()
    c = make_clip(50)
    for bad in (2.0 * np.eye(3), np.diag([1.0, 1.0, -1.0]), np.eye(4), np.full((3, 3), np.nan), np.ones(3)):
        with pytest.raises(ValueError):
            m.rotate_frame(c, bad)


# ------------------------------------------------------------------ gravity_estimate
def test_gravity_estimate_tracks_constant_and_rejects_shake():
    m = imu()
    c = _tone_clip(4000, a_freq=4.0, a_amp=3.0)
    g = m.gravity_estimate(c)
    assert g.shape == (4000, 3)
    e = _mid(4000, 4.0)                                                                     # 0.5 Hz 필터의 가장자리 과도는 수 초
    assert np.max(np.abs(g[e] - G0)) < 0.01                                                 # 0.5 Hz 저역: 4 Hz 떨림은 거의 사라진다
    wide = m.gravity_estimate(c, fc_hz=20.0)                                                # 컷오프를 올리면 떨림이 남는다
    assert np.max(np.abs(wide[e, 2] - G0[2])) > 2.0
    still = make_clip(600, gyro=np.zeros((600, 3)), accel=np.tile(G0, (600, 1)))
    assert np.allclose(m.gravity_estimate(still), G0, atol=1e-9)


# ------------------------------------------------------------------ time_warp
def test_time_warp_identity_and_length():
    m = imu()
    c = make_clip(1000, t0=1.5)
    w = m.time_warp(c, 1.0)
    assert w.n == 1000 and w.t0 == c.t0 and w.rate_hz == c.rate_hz and w.meta == c.meta
    assert np.max(np.abs(w.gyro - c.gyro)) < 1e-9 and np.max(np.abs(w.accel - c.accel)) < 1e-9
    for s, n in ((1.25, 1000), (0.8, 1000), (1.18, 777), (0.85, 3), (2.0, 1001), (1.0000001, 100)):
        assert m.warped_length(n, s) == int(math.floor((n - 1) / s + 1e-9)) + 1
    assert m.time_warp(make_clip(1000), 1.25).n == m.warped_length(1000, 1.25) == 800
    assert m.time_warp(make_clip(1000), 0.8).n == m.warped_length(1000, 0.8) == 1249
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.time_warp(c, bad)


@pytest.mark.parametrize("s", [1.25, 0.8])
def test_time_warp_scales_rates_and_dynamics_physically(s):
    """운동을 s 배 빠르게: 각속도 ω'(t)=s ω(st) → 진폭 s 배·주파수 s 배, 가속도의 동적 성분은 s² 배, 중력은 그대로."""
    m = imu()
    c = _tone_clip(4000, g_freq=3.0, g_amp=1.5, a_freq=4.0, a_amp=3.0)
    w = m.time_warp(c, s)
    e = _mid(w.n, 4.0)                                                                       # 중력 추정(0.5 Hz)의 가장자리 과도는 수 초
    assert lsq_amp(w.gyro[e, 0], w.t[e], 3.0 * s) == pytest.approx(1.5 * s, rel=0.02)
    assert lsq_amp(w.gyro[e, 0], w.t[e], 3.0) < 0.05 * 1.5                                  # 원래 주파수 성분은 사라진다
    assert lsq_amp(w.accel[e, 2], w.t[e], 4.0 * s) == pytest.approx(3.0 * s * s, rel=0.03)
    assert np.mean(w.accel[e, 1]) == pytest.approx(G0[1], abs=0.01)                          # 떨림 없는 축의 중력은 보존
    assert np.max(np.abs(w.accel[e, 1] - G0[1])) < 0.02
    assert np.allclose(w.gyro[e, 1:], 0.0, atol=1e-9)
    assert w.t0 == c.t0 and w.rate_hz == RATE


# ------------------------------------------------------------------ rig_filter
def _h_low(r, z):
    return 1.0 / math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2)


def _h_high(r, z):
    return r * r * _h_low(r, z)


def test_rig_filter_dc_behaviour_has_no_start_transient():
    m = imu()
    const = make_clip(1000, gyro=np.tile([0.3, -0.2, 0.9], (1000, 1)), accel=np.tile(G0, (1000, 1)))
    lo = m.rig_filter(const, kind="lowpass", fn_hz=4.0, zeta=0.7)
    assert np.allclose(lo.gyro, const.gyro, atol=1e-9) and np.allclose(lo.accel, const.accel, atol=1e-9)
    hi = m.rig_filter(const, kind="highpass", fn_hz=4.0, zeta=0.7)
    assert np.allclose(hi.gyro, 0.0, atol=1e-9)                                              # 고역: 직류 제거, 시작 과도 없음
    assert np.allclose(hi.accel, G0, atol=1e-9)                                              # 가속도는 중력(느린 성분)을 남긴다


@pytest.mark.parametrize("kind,f,z,fn", [("lowpass", 2.0, 0.707, 4.0), ("lowpass", 8.0, 0.707, 4.0), ("lowpass", 4.0, 0.2, 4.0),
                                         ("lowpass", 0.4, 1.0, 4.0), ("highpass", 2.0, 0.707, 4.0), ("highpass", 8.0, 0.5, 4.0),
                                         ("highpass", 12.0, 1.0, 6.0)])
def test_rig_filter_frequency_response_matches_second_order_analytic(kind, f, z, fn):
    m = imu()
    n = 4000
    t = np.arange(n) / RATE
    c = make_clip(n, gyro=tone(t, f, 1.0, 0.5, axis=1), accel=G0 + tone(t, f, 2.0, 0.1, axis=0))
    o = m.rig_filter(c, kind=kind, fn_hz=fn, zeta=z)
    h = (_h_low if kind == "lowpass" else _h_high)(f / fn, z)
    s = slice(n // 2, n)                                                                      # 정상상태 구간
    assert lsq_amp(o.gyro[s, 1], t[s], f) == pytest.approx(h, rel=0.03)
    assert np.max(np.abs(o.gyro[s, [0, 2]])) < 1e-9
    assert lsq_amp(o.accel[s, 0], t[s], f) == pytest.approx(2.0 * h, rel=0.04)               # 동적 가속도도 같은 전달함수
    assert np.mean(o.accel[s, 1]) == pytest.approx(G0[1], abs=0.01)                          # 중력은 리그가 못 거른다
    o2 = m.rig_filter(c, kind=kind, fn_hz=fn, zeta=z, gain=2.5)
    assert lsq_amp(o2.gyro[s, 1], t[s], f) == pytest.approx(2.5 * h, rel=0.03)
    assert o.t0 == c.t0 and o.rate_hz == c.rate_hz and o.n == c.n


def test_rig_filter_is_causal_linear_and_pure():
    m = imu()
    a = make_clip(1500, seed=11)
    g_future = a.gyro.copy()
    g_future[1000:] += 5.0
    b = m.ImuClip(a.t0, RATE, g_future, a.accel)
    oa = m.rig_filter(a, kind="lowpass", fn_hz=6.0, zeta=0.5)
    ob = m.rig_filter(b, kind="lowpass", fn_hz=6.0, zeta=0.5)
    assert np.array_equal(oa.gyro[:1000], ob.gyro[:1000])                                    # 자이로 경로는 인과적: 미래가 과거를 못 바꾼다
    assert not np.allclose(oa.gyro[1000:], ob.gyro[1000:])
    g1, g2 = make_signal(31)(a.t), make_signal(32)(a.t)
    ca, cb, cs = (m.ImuClip(a.t0, RATE, g, a.accel) for g in (g1, g2, g1 + 2 * g2))
    f = lambda c: m.rig_filter(c, kind="highpass", fn_hz=3.0, zeta=0.8).gyro            # noqa: E731
    assert np.allclose(f(cs), f(ca) + 2 * f(cb), atol=1e-8)                                  # 선형
    g0 = a.gyro.copy()
    m.rig_filter(a, kind="lowpass", fn_hz=6.0, zeta=0.5)
    assert np.array_equal(a.gyro, g0)


def test_rig_filter_validation():
    m = imu()
    c = make_clip(300)
    ok = dict(kind="lowpass", fn_hz=5.0, zeta=0.7)
    m.rig_filter(c, **ok)
    for bad in (dict(kind="bandpass"), dict(fn_hz=0.0), dict(fn_hz=-1.0), dict(fn_hz=90.0), dict(fn_hz=100.0), dict(zeta=0.0),
                dict(zeta=-0.2), dict(gain=0.0), dict(gain=-1.0), dict(fn_hz=float("nan"))):
        with pytest.raises(ValueError):
            m.rig_filter(c, **{**ok, **bad})


# ------------------------------------------------------------------ perturb_sensor
def test_perturb_sensor_statistics_bias_and_determinism():
    m = imu()
    n = 40000
    c = make_clip(n, gyro=np.zeros((n, 3)), accel=np.zeros((n, 3)))
    gb, ab = (0.01, -0.02, 0.03), (0.1, 0.0, -0.2)
    p = m.perturb_sensor(c, seed=5, gyro_sigma=0.05, accel_sigma=0.2, gyro_bias=gb, accel_bias=ab)
    assert np.allclose(p.gyro.mean(axis=0), gb, atol=5 * 0.05 / math.sqrt(n))
    assert np.allclose(p.accel.mean(axis=0), ab, atol=5 * 0.2 / math.sqrt(n))
    assert np.allclose(p.gyro.std(axis=0), 0.05, rtol=0.03) and np.allclose(p.accel.std(axis=0), 0.2, rtol=0.03)
    assert abs(np.corrcoef(p.gyro[:, 0], p.gyro[:, 1])[0, 1]) < 0.03 and abs(np.corrcoef(p.gyro[:, 0], p.accel[:, 0])[0, 1]) < 0.03
    q = m.perturb_sensor(c, seed=5, gyro_sigma=0.05, accel_sigma=0.2, gyro_bias=gb, accel_bias=ab)
    assert np.array_equal(p.gyro, q.gyro) and np.array_equal(p.accel, q.accel)
    r = m.perturb_sensor(c, seed=6, gyro_sigma=0.05, accel_sigma=0.2, gyro_bias=gb, accel_bias=ab)
    assert not np.array_equal(p.gyro, r.gyro)
    base = make_clip(300)
    z = m.perturb_sensor(base, seed=1)
    assert np.array_equal(z.gyro, base.gyro) and np.array_equal(z.accel, base.accel)         # 기본값은 항등
    st = np.random.get_state()[1].copy()
    m.perturb_sensor(base, seed=3, gyro_sigma=1.0)
    assert np.array_equal(np.random.get_state()[1], st)                                      # 전역 RNG 불변


def test_perturb_sensor_validation():
    m = imu()
    c = make_clip(100)
    for kw in (dict(gyro_sigma=-0.1), dict(accel_sigma=-0.1), dict(gyro_bias=(1.0, 2.0)), dict(accel_bias=(1, 2, 3, 4)),
               dict(gyro_bias=(0, 0, float("nan")))):
        with pytest.raises(ValueError):
            m.perturb_sensor(c, seed=0, **kw)


# ------------------------------------------------------------------ AugmentConfig · sample_augmentation
def test_augment_config_defaults_frozen_and_validation():
    m = imu()
    cfg = m.AugmentConfig()
    assert dataclasses.is_dataclass(cfg)
    d = dataclasses.asdict(cfg)
    assert d["rot_deg_max"] == 30.0 and tuple(d["time_scale"]) == (0.85, 1.18) and d["rig_prob"] == 0.6
    assert tuple(d["rig_kinds"]) == ("lowpass", "highpass") and tuple(d["rig_fn_hz"]) == (1.0, 20.0)
    assert tuple(d["rig_zeta"]) == (0.3, 1.2) and tuple(d["rig_gain"]) == (0.5, 1.5)
    assert tuple(d["gyro_sigma"]) == (0.0, 0.02) and tuple(d["accel_sigma"]) == (0.0, 0.1)
    assert d["gyro_bias_max"] == 0.01 and d["accel_bias_max"] == 0.05
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.rig_prob = 0.1
    with pytest.raises(TypeError):
        m.AugmentConfig(30.0)                                                                # 키워드 전용
    for bad in (dict(time_scale=(1.2, 0.9)), dict(time_scale=(0.0, 1.0)), dict(rig_fn_hz=(1.0, 90.0)), dict(rig_fn_hz=(0.0, 5.0)),
                dict(rig_prob=1.5), dict(rig_prob=-0.1), dict(rig_kinds=("notch",)), dict(rig_kinds=()), dict(gyro_sigma=(-1.0, 0.1)),
                dict(rot_deg_max=-1.0), dict(rig_zeta=(0.0, 1.0)), dict(rig_gain=(0.0, 1.0)), dict(accel_bias_max=-0.1),
                dict(gyro_sigma=(0.0, float("nan")))):
        with pytest.raises(ValueError):
            m.AugmentConfig(**bad)


def test_sample_augmentation_schema_ranges_and_determinism():
    m = imu()
    cfg = m.AugmentConfig(rot_deg_max=20.0, time_scale=(0.9, 1.1), rig_prob=0.5, rig_kinds=("lowpass",), rig_fn_hz=(2.0, 10.0),
                          rig_zeta=(0.4, 0.9), rig_gain=(0.7, 1.3), gyro_sigma=(0.001, 0.01), accel_sigma=(0.01, 0.05),
                          gyro_bias_max=0.005, accel_bias_max=0.02)
    st = np.random.get_state()[1].copy()
    augs = [m.sample_augmentation(cfg, seed=s) for s in range(400)]
    assert np.array_equal(np.random.get_state()[1], st)
    n_rig = 0
    for a in augs:
        assert set(a) == {"rotvec", "time_scale", "rig", "noise", "crop"}
        json.dumps(a)                                                                        # JSON 직렬화 가능(numpy 형 금지)
        assert len(a["rotvec"]) == 3 and np.linalg.norm(a["rotvec"]) <= math.radians(20.0) + 1e-9
        assert 0.9 - 1e-12 <= a["time_scale"] <= 1.1 + 1e-12 and a["crop"] is None
        nz = a["noise"]
        assert isinstance(nz["seed"], int) and nz["seed"] >= 0
        assert 0.001 <= nz["gyro_sigma"] <= 0.01 and 0.01 <= nz["accel_sigma"] <= 0.05
        assert len(nz["gyro_bias"]) == 3 and max(abs(x) for x in nz["gyro_bias"]) <= 0.005 + 1e-12
        assert len(nz["accel_bias"]) == 3 and max(abs(x) for x in nz["accel_bias"]) <= 0.02 + 1e-12
        if a["rig"] is not None:
            n_rig += 1
            r = a["rig"]
            assert set(r) == {"kind", "fn_hz", "zeta", "gain"} and r["kind"] == "lowpass"
            assert 2.0 <= r["fn_hz"] <= 10.0 and 0.4 <= r["zeta"] <= 0.9 and 0.7 <= r["gain"] <= 1.3
    assert 0.4 <= n_rig / 400 <= 0.6
    assert len({round(a["time_scale"], 9) for a in augs}) > 300 and len({a["noise"]["seed"] for a in augs}) > 390
    assert m.sample_augmentation(cfg, seed=7) == m.sample_augmentation(cfg, seed=7)
    assert m.sample_augmentation(cfg, seed=7) != m.sample_augmentation(cfg, seed=8)
    assert m.sample_augmentation(seed=3) == m.sample_augmentation(m.AugmentConfig(), seed=3)       # cfg 기본값 = AugmentConfig()
    always = m.AugmentConfig(rig_prob=1.0)
    never = m.AugmentConfig(rig_prob=0.0)
    assert all(m.sample_augmentation(always, seed=s)["rig"] is not None for s in range(50))
    assert all(m.sample_augmentation(never, seed=s)["rig"] is None for s in range(50))
    flat = m.AugmentConfig(rot_deg_max=0.0, time_scale=(1.0, 1.0), rig_prob=0.0, gyro_sigma=(0.0, 0.0), accel_sigma=(0.0, 0.0),
                           gyro_bias_max=0.0, accel_bias_max=0.0)
    a = m.sample_augmentation(flat, seed=1)
    assert a["time_scale"] == 1.0 and a["rotvec"] == [0.0, 0.0, 0.0] and a["noise"]["gyro_sigma"] == 0.0
    assert a["noise"]["gyro_bias"] == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------ apply_augmentation
def test_apply_augmentation_order_is_rotate_warp_rig_perturb_crop():
    m = imu()
    c = make_clip(3000, meta={"dataset": "TUM VI"}, seed=4)
    aug = {"rotvec": [0.2, -0.4, 0.3], "time_scale": 1.1,
           "rig": {"kind": "lowpass", "fn_hz": 6.0, "zeta": 0.6, "gain": 0.9},
           "noise": {"seed": 77, "gyro_sigma": 0.01, "accel_sigma": 0.05, "gyro_bias": [0.002, -0.001, 0.003], "accel_bias": [0.01, 0.0, -0.02]},
           "crop": [100, 1500]}
    R = Rotation.from_rotvec(aug["rotvec"]).as_matrix()
    x = m.rotate_frame(c, R)
    x = m.time_warp(x, 1.1)
    x = m.rig_filter(x, kind="lowpass", fn_hz=6.0, zeta=0.6, gain=0.9)
    nz = aug["noise"]
    x = m.perturb_sensor(x, seed=nz["seed"], gyro_sigma=nz["gyro_sigma"], accel_sigma=nz["accel_sigma"], gyro_bias=nz["gyro_bias"],
                         accel_bias=nz["accel_bias"])
    x = x.slice(100, 1600)
    out = m.apply_augmentation(c, aug)
    assert out.n == 1500 and out.rate_hz == RATE and out.t0 == pytest.approx(x.t0, abs=1e-12)
    assert np.allclose(out.gyro, x.gyro, atol=1e-9) and np.allclose(out.accel, x.accel, atol=1e-9)
    assert out.meta["dataset"] == "TUM VI" and out.meta["aug"] == aug and "aug" not in c.meta     # 출처 보존, 입력 불변
    again = m.apply_augmentation(c, json.loads(json.dumps(aug)))
    assert np.array_equal(again.gyro, out.gyro) and np.array_equal(again.accel, out.accel)       # 기록만으로 재현


def test_apply_augmentation_identity_and_validation():
    m = imu()
    c = make_clip(800)
    ident = {"rotvec": [0.0, 0.0, 0.0], "time_scale": 1.0, "rig": None,
             "noise": {"seed": 1, "gyro_sigma": 0.0, "accel_sigma": 0.0, "gyro_bias": [0, 0, 0], "accel_bias": [0, 0, 0]}, "crop": None}
    o = m.apply_augmentation(c, ident)
    assert o.n == 800 and np.max(np.abs(o.gyro - c.gyro)) < 1e-9 and np.max(np.abs(o.accel - c.accel)) < 1e-9
    bad_cases = []
    for k in ident:
        d = dict(ident)
        del d[k]
        bad_cases.append(d)
    bad_cases.append({**ident, "extra": 1})
    bad_cases += [{**ident, "crop": [-1, 10]}, {**ident, "crop": [790, 20]}, {**ident, "crop": [0, 0]}, {**ident, "rotvec": [0.0, 0.0]},
                  {**ident, "time_scale": 0.0}, {**ident, "rig": {"kind": "lowpass", "fn_hz": 5.0}}]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            m.apply_augmentation(c, bad)


def test_rotation_only_augmentation_preserves_norms():
    m = imu()
    cfg = m.AugmentConfig(rot_deg_max=180.0, time_scale=(1.0, 1.0), rig_prob=0.0, gyro_sigma=(0.0, 0.0), accel_sigma=(0.0, 0.0),
                          gyro_bias_max=0.0, accel_bias_max=0.0)
    c = make_clip(900, seed=8)
    for s in range(5):
        o = m.augment_clip(c, cfg, seed=s)
        assert np.allclose(np.linalg.norm(o.gyro, axis=1), np.linalg.norm(c.gyro, axis=1), atol=1e-9)
        assert np.allclose(np.linalg.norm(o.accel, axis=1), np.linalg.norm(c.accel, axis=1), atol=1e-9)
    assert not np.allclose(m.augment_clip(c, cfg, seed=1).gyro, c.gyro)


def test_augment_clip_equals_sample_then_apply():
    m = imu()
    c = make_clip(1200, seed=9, meta={"a": 1})
    cfg = m.AugmentConfig()
    for s in (0, 1, 2, 3):
        a = m.augment_clip(c, cfg, seed=s)
        b = m.apply_augmentation(c, m.sample_augmentation(cfg, seed=s))
        assert np.array_equal(a.gyro, b.gyro) and np.array_equal(a.accel, b.accel) and a.meta == b.meta
    d = m.augment_clip(c, seed=5)
    e = m.augment_clip(c, m.AugmentConfig(), seed=5)
    assert np.array_equal(d.gyro, e.gyro)
    assert not np.array_equal(m.augment_clip(c, cfg, seed=1).gyro, m.augment_clip(c, cfg, seed=2).gyro)
