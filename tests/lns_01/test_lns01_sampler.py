"""LNS-01: 사전분포 샘플러 — 결정론, 접두 안정성, fixed, 한계 분포(허용 오차 6 시그마급), 결합 제약."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from buc.core.contracts import ContractError, Theta
from buc.lens import theta as T
from lns01_helpers import (HARD, HARD_EXCLUSIVE_HI, LENSES, LUTS, N_SAMPLES, REPO, SEED, assert_choices, assert_frac,
                           assert_uniform, col, ecol, fov_of, samples)

SENSORS = {8.8: 0.05, 17.3: 0.15, 24.89: 0.40, 36.0: 0.30, 54.12: 0.10}
ANAMORPHIC_ASPECT = {2.0: 0.2, 2.39: 0.7, 2.76: 0.1}
SPHERICAL_ASPECT = {1.3333: 0.10, 1.5: 0.05, 1.7778: 0.55, 1.85: 0.10, 2.0: 0.05, 2.39: 0.15}


def hashes(ts):
    return [t.content_hash() for t in ts]


# --------------------------------------------------------------------------------------------- 결정론
def test_sample_count_and_validity():
    ts = samples()
    assert len(ts) == N_SAMPLES
    for t in ts:
        assert isinstance(t, Theta)
    for t in ts[:400]:
        T.validate_theta(t)
        t.validate()


def test_all_samples_inside_hard_bounds_and_complete_extra():
    ts = samples()
    for n, (lo, hi) in HARD.items():
        if n in ("blades", "cpl_angle_deg", "rolloff", "atmos_g"):
            x = ecol(n)
        elif n.startswith("sun_"):
            x = np.array([getattr(t, n) for t in ts if getattr(t, n) is not None], dtype=np.float64)
        else:
            x = col(n)
        assert np.isfinite(x).all(), n
        assert x.min() >= lo, (n, x.min())
        assert (x.max() < hi) if n in HARD_EXCLUSIVE_HI else (x.max() <= hi), (n, x.max())
    for t in ts[:500]:
        assert set(T.EXTRA_DEFAULTS) <= set(t.extra)


def test_python_native_types_and_json():
    for t in samples()[:300]:
        for n in ("aspect", "focal_mm", "sensor_width_mm", "f_stop", "focus_m", "shutter_deg", "fps", "nd_stops", "cpl",
                  "ev", "wb_kelvin", "wb_tint", "grain", "halation", "vignette", "atmos_beta", "rim"):
            assert type(getattr(t, n)) is float, (n, type(getattr(t, n)))
        assert t.sun_elev_deg is None or type(t.sun_elev_deg) is float
        assert t.sun_azim_deg is None or type(t.sun_azim_deg) is float
        assert type(t.extra["blades"]) is int
        for k in ("squeeze", "cpl_angle_deg", "rolloff", "atmos_g"):
            assert type(t.extra[k]) is float, k
        s = json.dumps(t.to_dict(), allow_nan=False)
        assert Theta.from_dict(json.loads(s)).content_hash() == t.content_hash()


def test_same_args_same_hashes_and_prefix_stability():
    a = hashes(T.sample_thetas(12, 7))
    assert a == hashes(T.sample_thetas(12, 7))
    assert hashes(T.sample_thetas(5, 7)) == a[:5]
    assert hashes(T.sample_thetas(5, 7)) == hashes(T.sample_thetas(40, 7))[:5]
    assert len(set(a)) == 12  # 서로 다른 표본


def test_definition_of_ith_sample():
    for i in (0, 1, 2, 17):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, i]))
        assert T.sample_theta(rng).content_hash() == samples()[i].content_hash()


def test_different_seeds_differ():
    a = set(hashes(T.sample_thetas(20, 1)))
    b = set(hashes(T.sample_thetas(20, 2)))
    assert not (a & b) and len(a) == 20 and len(b) == 20


def test_int_seed_equals_generator_and_stream_advances():
    assert T.sample_theta(7).content_hash() == T.sample_theta(np.random.default_rng(7)).content_hash()
    g = np.random.default_rng(3)
    first, second = T.sample_theta(g), T.sample_theta(g)
    assert first.content_hash() != second.content_hash()
    g2 = np.random.default_rng(3)
    assert T.sample_theta(g2).content_hash() == first.content_hash()
    assert T.sample_theta(g2).content_hash() == second.content_hash()


def test_sample_thetas_edge_cases():
    assert T.sample_thetas(0, 5) == []
    with pytest.raises(ValueError):
        T.sample_thetas(-1, 5)
    assert hashes(T.sample_thetas(3)) == hashes(T.sample_thetas(3, 0))


def test_no_dependence_on_global_state_or_cwd(tmp_path, monkeypatch):
    base = hashes(T.sample_thetas(6, 11))
    import random
    random.seed(123)
    np.random.seed(456)
    monkeypatch.chdir(tmp_path)
    assert hashes(T.sample_thetas(6, 11)) == base
    state = np.random.get_state()[1].copy()
    T.sample_thetas(10, 11)
    assert (np.random.get_state()[1] == state).all()  # 전역 난수 상태를 건드리지 않는다


def test_deterministic_across_processes_and_hashseed():
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from buc.lens import theta as T\n"
            "print(','.join(t.content_hash() for t in T.sample_thetas(6, 11)))\n"
            "print('torch' in sys.modules)\n" % str(REPO))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50,
                           cwd=str(REPO), env=env)
        assert p.returncode == 0, p.stderr[-500:]
        outs.append(p.stdout.split("\n"))
    assert outs[0] == outs[1]
    assert outs[0][0].split(",") == hashes(T.sample_thetas(6, 11))
    assert outs[0][1] == "False"


# --------------------------------------------------------------------------------------------- fixed
def test_fixed_scalar_field():
    ts = T.sample_thetas(60, 5, fixed={"f_stop": 2.8, "lut": "teal_orange"})
    assert all(t.f_stop == 2.8 and t.lut == "teal_orange" for t in ts)
    assert len({t.focus_m for t in ts}) > 50  # 나머지는 여전히 샘플
    assert hashes(ts) == hashes(T.sample_thetas(60, 5, fixed={"f_stop": 2.8, "lut": "teal_orange"}))
    for t in ts:
        T.validate_theta(t)


def test_fixed_focal_defaults_sensor_super35():
    ts = T.sample_thetas(60, 6, fixed={"focal_mm": 50})
    assert all(t.focal_mm == 50.0 and t.sensor_width_mm == 24.89 for t in ts)
    ts = T.sample_thetas(60, 6, fixed={"focal_mm": 50, "sensor_width_mm": 36.0})
    assert all(t.focal_mm == 50.0 and t.sensor_width_mm == 36.0 for t in ts)


def test_fixed_sensor_only_keeps_fov_prior():
    ts = T.sample_thetas(300, 8, fixed={"sensor_width_mm": 36.0})
    assert all(t.sensor_width_mm == 36.0 for t in ts)
    f = fov_of(ts)
    assert f.min() >= 15 - 1e-6 and f.max() <= 95 + 1e-6


def test_fixed_extra_key_and_none_sun():
    ts = T.sample_thetas(40, 9, fixed={"blades": 12, "rolloff": 0.25})
    assert all(t.extra["blades"] == 12 and type(t.extra["blades"]) is int and t.extra["rolloff"] == 0.25 for t in ts)
    ts = T.sample_thetas(80, 9, fixed={"sun_elev_deg": 30.0})
    assert all(t.sun_elev_deg == 30.0 and t.sun_azim_deg is not None for t in ts)
    ts = T.sample_thetas(80, 9, fixed={"sun_elev_deg": None})
    assert all(t.sun_elev_deg is None and t.sun_azim_deg is None for t in ts)
    ts = T.sample_thetas(80, 9, fixed={"sun_azim_deg": 90.0})
    assert all(t.sun_azim_deg == 90.0 and t.sun_elev_deg is not None for t in ts)


def test_fixed_anamorphic_lens_forces_coupling():
    ts = T.sample_thetas(300, 10, fixed={"lens_rx": "anamorphic_2x"})
    assert all(t.lens_rx == "anamorphic_2x" and t.extra["squeeze"] == 2.0 for t in ts)
    assert_choices([t.aspect for t in ts], ANAMORPHIC_ASPECT, "fixed anamorphic aspect")
    ts = T.sample_thetas(300, 10, fixed={"lens_rx": "vintage_petzval"})
    assert all(t.extra["squeeze"] == 1.0 for t in ts)
    assert_choices([t.aspect for t in ts], SPHERICAL_ASPECT, "fixed spherical aspect")


@pytest.mark.parametrize("fixed", [
    {"fov_x_deg": 40.0}, {"bogus": 1}, {"f_stop": 100.0}, {"lut": "nope"}, {"focal_mm": float("nan")},
    {"lens_rx": "anamorphic_2x", "aspect": 1.7778},  # 애너모픽 + aspect<2
    {"lens_rx": "spherical_clean", "squeeze": 2.0},  # squeeze 결합 위반
    {"focal_mm": 4.0},  # sensor 24.89 -> 화각 144도
    {"sun_azim_deg": 90.0, "sun_elev_deg": None},
    {"blades": 2},
])
def test_fixed_invalid_raises(fixed):
    with pytest.raises(ContractError):
        T.sample_theta(1, fixed=fixed)


def test_fixed_not_mutated():
    f = {"f_stop": 2.8}
    T.sample_theta(1, fixed=f)
    assert f == {"f_stop": 2.8}


# --------------------------------------------------------------------------------------------- 한계 분포
def test_sensor_and_fov_prior():
    assert_choices(col("sensor_width_mm"), SENSORS, "sensor")
    f = fov_of()
    assert_uniform(f, 15.0, 95.0, "fov_x", log=True)
    # 화각 -> 초점거리 관계가 정확하다
    ts = samples()
    for t in ts[:200]:
        fov = math.radians(t.fov_x_deg())
        assert t.focal_mm == pytest.approx(t.sensor_width_mm / (2 * math.tan(fov / 2)), rel=1e-12)


def test_log_uniform_scalars():
    assert_uniform(col("f_stop"), 1.2, 22.0, "f_stop", log=True)
    assert_uniform(col("focus_m"), 0.5, 50.0, "focus_m", log=True)
    assert_uniform(col("wb_kelvin"), 2500.0, 10000.0, "wb_kelvin", log=True)


def test_shutter_and_fps_choices():
    assert_choices(col("shutter_deg"), {45: .05, 90: .15, 144: .05, 172.8: .05, 180: .60, 270: .10}, "shutter")
    assert_choices(col("fps"), {24: .50, 25: .10, 30: .15, 48: .05, 60: .10, 120: .10}, "fps")


@pytest.mark.parametrize("name,p0,lo,hi,log", [
    ("nd_stops", 0.6, 0.3, 10.0, False), ("cpl", 0.7, 0.2, 1.0, False), ("grain", 0.3, 0.05, 0.7, False),
    ("halation", 0.5, 0.05, 0.6, False), ("vignette", 0.3, 0.05, 0.7, False), ("rim", 0.5, 0.1, 1.0, False),
    ("atmos_beta", 0.4, 1e-4, 2e-2, True),
])
def test_zero_inflated(name, p0, lo, hi, log):
    x = col(name)
    assert_frac(x == 0.0, p0, name + " 영 비율")
    nz = x[x != 0.0]
    assert nz.min() >= lo - 1e-12, (name, nz.min())
    assert_uniform(nz, lo, hi, name + " 비영 부분", log=log)


def test_ev_and_tint_clipped_normal():
    ev = col("ev")
    assert ev.min() >= -3 and ev.max() <= 3
    assert abs(ev.mean()) < 6 * 0.7 / math.sqrt(N_SAMPLES)
    assert ev.std() == pytest.approx(0.7, abs=0.05)
    assert_frac(np.abs(ev) < 0.7, 0.6827, "ev 1시그마 비율")
    tint = col("wb_tint")
    assert tint.min() >= -0.5 and tint.max() <= 0.5
    assert abs(tint.mean()) < 6 * 0.15 / math.sqrt(N_SAMPLES)
    assert tint.std() == pytest.approx(0.15, abs=0.02)
    assert_frac(np.abs(tint) < 0.15, 0.6827, "tint 1시그마 비율")


def test_lut_and_lens_choices():
    luts = [t.lut for t in samples()]
    assert set(luts) <= set(LUTS)
    assert_frac(np.array([x == "none" for x in luts]), 0.35, "lut none")
    assert_frac(np.array([x == "neutral_709" for x in luts]), 0.15, "lut neutral_709")
    for name in LUTS[2:]:
        assert_frac(np.array([x == name for x in luts]), 0.0625, "lut " + name)
    lens = [t.lens_rx for t in samples()]
    assert set(lens) <= set(LENSES)
    assert_frac(np.array([x == "spherical_clean" for x in lens]), 0.6, "lens spherical")
    for name in LENSES[1:]:
        assert_frac(np.array([x == name for x in lens]), 0.1, "lens " + name)


def test_aspect_and_squeeze_coupling():
    ts = samples()
    ana = [t for t in ts if t.lens_rx == "anamorphic_2x"]
    sph = [t for t in ts if t.lens_rx != "anamorphic_2x"]
    assert len(ana) > 250 and len(sph) > 3000
    assert all(t.extra["squeeze"] == 2.0 for t in ana)
    assert all(t.extra["squeeze"] == 1.0 for t in sph)
    assert_choices([t.aspect for t in ana], ANAMORPHIC_ASPECT, "aspect(anamorphic)")
    assert_choices([t.aspect for t in sph], SPHERICAL_ASPECT, "aspect(spherical)")


def test_sun_prior():
    ts = samples()
    none_mask = np.array([t.sun_elev_deg is None for t in ts])
    assert (none_mask == np.array([t.sun_azim_deg is None for t in ts])).all()
    assert_frac(none_mask, 0.25, "태양 없음")
    elev = np.array([t.sun_elev_deg for t in ts if t.sun_elev_deg is not None])
    azim = np.array([t.sun_azim_deg for t in ts if t.sun_azim_deg is not None])
    assert_uniform(elev, -5.0, 75.0, "sun_elev")
    assert_uniform(azim, 0.0, 360.0, "sun_azim")
    assert azim.max() < 360.0


def test_extra_prior():
    b = ecol("blades")
    assert set(np.unique(b)) == set(range(5, 13))
    for k in range(5, 13):
        assert_frac(b == k, 1 / 8, f"blades={k}")
    assert_uniform(ecol("cpl_angle_deg"), 0.0, 180.0, "cpl_angle_deg")
    assert ecol("cpl_angle_deg").max() < 180.0
    assert_uniform(ecol("rolloff"), 0.0, 1.0, "rolloff")
    assert_uniform(ecol("atmos_g"), 0.0, 0.9, "atmos_g")


def test_fields_are_independent():
    cols = {
        "ln_f": np.log(col("f_stop")), "ln_focus": np.log(col("focus_m")), "ln_fov": np.log(fov_of()),
        "ev": col("ev"), "ln_wb": np.log(col("wb_kelvin")), "rolloff": ecol("rolloff"), "tint": col("wb_tint"),
        "grain": col("grain"),
    }
    names = list(cols)
    m = np.corrcoef(np.vstack([cols[n] for n in names]))
    lim = 6 / math.sqrt(N_SAMPLES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert abs(m[i, j]) < lim, (names[i], names[j], m[i, j])
    # 영 팽창 두 필드의 영 여부도 독립(결합 확률 = 곱)
    a, b = col("grain") == 0, col("halation") == 0
    assert_frac(a & b, 0.3 * 0.5, "grain=0 & halation=0")


def test_physical_plausibility_of_sampled_exposure():
    """표본 노출 시간(shutter/(360 fps))은 1/960 s 이상 1/20 s 이하 (45도@120fps ~ 270도@24fps 범위 안)."""
    e = np.array([T.exposure_time_s(t) for t in samples()])
    assert e.min() >= 45 / (360 * 120) - 1e-12
    assert e.max() <= 270 / (360 * 24) + 1e-12
