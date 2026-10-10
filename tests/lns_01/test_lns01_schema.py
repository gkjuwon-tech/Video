"""LNS-01: 스키마 상수, 범위 검증, 직렬화 왕복, 노출 시간."""
import dataclasses
import json
import math

import numpy as np
import pytest

from buc.core.contracts import ContractError, Theta
from buc.lens import theta as T
from lns01_helpers import EXTRA_DEFAULTS, HARD, HARD_EXCLUSIVE_HI, LENSES, LUTS, samples


# --------------------------------------------------------------------------------------------- 상수
def test_vocab_and_constants():
    assert T.SCHEMA_VERSION == "1.0.0"
    assert tuple(T.LUT_NAMES) == LUTS
    assert tuple(T.LENS_RX_NAMES) == LENSES
    assert dict(T.EXTRA_DEFAULTS) == EXTRA_DEFAULTS
    assert type(T.EXTRA_DEFAULTS["blades"]) is int
    for k in ("squeeze", "cpl_angle_deg", "rolloff", "atmos_g"):
        assert type(T.EXTRA_DEFAULTS[k]) is float


def test_hard_bounds_table_exact():
    for k, (lo, hi) in HARD.items():
        assert k in T.HARD_BOUNDS, k
        a, b = T.HARD_BOUNDS[k]
        assert float(a) == pytest.approx(lo, abs=1e-12) and float(b) == pytest.approx(hi, abs=1e-12), k


def test_hard_bounds_inside_fnd02_ranges():
    """HARD_BOUNDS 는 FND-02 Theta 의 유효 범위보다 좁거나 같다(교집합이 비어 있지 않다)."""
    for lo_val in ("focal_mm", "sensor_width_mm", "f_stop", "ev", "wb_kelvin", "fps", "shutter_deg"):
        lo, hi = T.HARD_BOUNDS[lo_val]
        Theta(**{lo_val: lo})
        Theta(**{lo_val: hi})


def test_presets_dir_is_repo_presets():
    from lns01_helpers import PRESETS
    assert T.PRESETS_DIR.resolve() == PRESETS.resolve()


def test_torch_free_import_in_fresh_process():
    import subprocess
    import sys
    from lns01_helpers import REPO
    code = ("import sys; sys.path.insert(0, %r); import buc.lens.theta; print('torch' in sys.modules)" % str(REPO))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50, cwd=str(REPO))
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------------------------- 기본값
def test_default_theta_valid_and_complete():
    t = T.default_theta()
    T.validate_theta(t)
    assert t.extra == EXTRA_DEFAULTS
    assert t.lut == "none" and t.lens_rx == "spherical_clean"
    base = Theta()
    for f in dataclasses.fields(Theta):
        if f.name != "extra":
            assert getattr(t, f.name) == getattr(base, f.name), f.name
    assert t.fov_x_deg() == pytest.approx(2 * math.degrees(math.atan(24.89 / 70)), abs=1e-9)


def test_complete_extra_semantics():
    t = Theta(extra={"squeeze": 1, "blades": 8.0, "future_key": [1, 2]})
    c = T.complete_extra(t)
    assert c is not t
    assert t.extra == {"squeeze": 1, "blades": 8.0, "future_key": [1, 2]}  # 입력 불변
    assert c.extra["future_key"] == [1, 2]
    assert type(c.extra["squeeze"]) is float and c.extra["squeeze"] == 1.0
    assert type(c.extra["blades"]) is int and c.extra["blades"] == 8
    for k in ("cpl_angle_deg", "rolloff", "atmos_g"):
        assert c.extra[k] == EXTRA_DEFAULTS[k] and type(c.extra[k]) is float
    # 멱등
    assert T.complete_extra(c).content_hash() == c.content_hash()
    # 다른 필드는 그대로
    assert c.f_stop == t.f_stop and c.lut == t.lut


@pytest.mark.parametrize("bad", [8.5, True, "9", None])
def test_complete_extra_rejects_bad_blades(bad):
    with pytest.raises(ContractError):
        T.complete_extra(Theta(extra={"blades": bad}))


# --------------------------------------------------------------------------------------------- validate_theta
def test_validate_accepts_good_cases():
    T.validate_theta(T.default_theta())
    T.validate_theta(Theta())  # extra 가 비어 있어도 기본값 취급
    T.validate_theta(Theta(extra={"future_key": "x"}))  # 모르는 extra 키 허용
    T.validate_theta(Theta(aspect=2.39, lens_rx="anamorphic_2x", extra={"squeeze": 2.0}))
    T.validate_theta(Theta(sun_elev_deg=30, sun_azim_deg=200))
    # 경계값 포함
    T.validate_theta(Theta(focal_mm=35, f_stop=0.95, focus_m=0.3, fps=8, nd_stops=12, ev=-5, wb_kelvin=2000))
    T.validate_theta(Theta(focal_mm=600, sensor_width_mm=70, f_stop=32, focus_m=1000, shutter_deg=360, fps=240))
    T.validate_theta(Theta(wb_tint=0.6, atmos_beta=0.1, sun_elev_deg=-20, sun_azim_deg=0, aspect=3.0))
    T.validate_theta(Theta(extra={"blades": 3, "cpl_angle_deg": 0.0, "rolloff": 1.0, "atmos_g": 0.95}))
    T.validate_theta(Theta(extra={"blades": 18, "cpl_angle_deg": 179.999, "rolloff": 0.0}))


BAD_KW = [
    {"focal_mm": 3}, {"focal_mm": 700}, {"sensor_width_mm": 4.0}, {"sensor_width_mm": 80},
    {"f_stop": 0.8}, {"f_stop": 40}, {"focus_m": float("inf")}, {"focus_m": 0.2}, {"focus_m": 2000},
    {"fps": 5}, {"shutter_deg": 0.5}, {"nd_stops": 13}, {"ev": 6}, {"ev": -6}, {"wb_kelvin": 1800},
    {"wb_kelvin": 13000}, {"wb_tint": 0.7}, {"wb_tint": -0.7}, {"atmos_beta": 0.2}, {"aspect": 3.5},
    {"aspect": 0.8}, {"sun_elev_deg": -30, "sun_azim_deg": 10}, {"sun_elev_deg": 95, "sun_azim_deg": 10},
    {"lut": "bogus"}, {"lens_rx": "bogus"}, {"lut": ""},
    # 화각 밖: 24.89mm 센서에 4mm = 144도, 4.8mm 센서에 600mm = 0.46도
    {"focal_mm": 4}, {"sensor_width_mm": 4.8, "focal_mm": 600},
    # 태양은 둘이 함께
    {"sun_elev_deg": 30}, {"sun_azim_deg": 120},
    # 애너모픽 결합
    {"lens_rx": "anamorphic_2x", "aspect": 2.39},  # squeeze 기본 1.0 과 불일치
    {"lens_rx": "anamorphic_2x", "aspect": 1.7778, "extra": {"squeeze": 2.0}},  # 애너모픽은 aspect>=2
    {"lens_rx": "spherical_clean", "extra": {"squeeze": 2.0}},
    {"extra": {"squeeze": 1.5}}, {"extra": {"squeeze": "2"}},
    # extra 범위
    {"extra": {"blades": 2}}, {"extra": {"blades": 19}}, {"extra": {"blades": 8.5}}, {"extra": {"blades": True}},
    {"extra": {"rolloff": 1.5}}, {"extra": {"rolloff": -0.1}}, {"extra": {"rolloff": float("nan")}},
    {"extra": {"cpl_angle_deg": 180.0}}, {"extra": {"cpl_angle_deg": -1.0}},
    {"extra": {"atmos_g": 0.96}}, {"extra": {"atmos_g": -0.1}},
]


@pytest.mark.parametrize("kw", BAD_KW, ids=[str(i) for i in range(len(BAD_KW))])
def test_validate_rejects(kw):
    try:
        t = Theta(**kw)
    except ContractError:
        # FND-02 가 생성자에서 먼저 거부하면 그것도 같은 예외 타입이어야 한다.
        return
    with pytest.raises(ContractError):
        T.validate_theta(t)


def test_validate_exception_type_is_contracterror_only():
    # ValueError 하위이긴 하지만 정확히 ContractError(또는 하위) 여야 한다
    with pytest.raises(ContractError) as ei:
        T.validate_theta(Theta(lut="nope"))
    assert isinstance(ei.value, ValueError)


# --------------------------------------------------------------------------------------------- 직렬화
def _only_plain(x):
    if isinstance(x, dict):
        return all(isinstance(k, str) and _only_plain(v) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return all(_only_plain(v) for v in x)
    return x is None or type(x) in (float, int, str, bool)


def test_theta_to_dict_shape():
    t = T.default_theta()
    d = T.theta_to_dict(t)
    names = {f.name for f in dataclasses.fields(Theta)}
    assert set(d) == names
    assert "__contract__" not in d and "__v__" not in d
    assert isinstance(d["extra"], dict) and d["extra"] == EXTRA_DEFAULTS
    assert _only_plain(d)
    json.dumps(d, allow_nan=False)
    assert d["sun_elev_deg"] is None and d["sun_azim_deg"] is None


def test_roundtrip_samples_bit_exact():
    for t in samples()[:300]:
        d = json.loads(json.dumps(T.theta_to_dict(t), allow_nan=False))
        t2 = T.theta_from_dict(d)
        assert t2.content_hash() == t.content_hash()
        assert t.content_hash() == T.complete_extra(t).content_hash()


def test_theta_from_dict_defaults_and_coercion():
    t = T.theta_from_dict({})
    assert t.content_hash() == T.default_theta().content_hash()
    t = T.theta_from_dict({"lens_rx": "anamorphic_2x", "aspect": 2.39, "extra": {"squeeze": 2, "blades": 9.0}})
    assert type(t.extra["squeeze"]) is float and t.extra["squeeze"] == 2.0
    assert type(t.extra["blades"]) is int and t.extra["blades"] == 9
    assert t.extra["rolloff"] == 0.5
    t = T.theta_from_dict({"f_stop": 2, "focal_mm": 50})  # 정수 입력은 float 로
    assert type(t.f_stop) is float and type(t.focal_mm) is float


@pytest.mark.parametrize("bad", [
    None, [], "theta", 3, {"nonsense": 1}, {"__contract__": "Theta"}, {"f_stop": 100}, {"lut": "x"},
    {"focal_mm": float("nan")}, {"extra": {"blades": 2}}, {"extra": "x"}, {"sun_elev_deg": 10},
])
def test_theta_from_dict_rejects(bad):
    with pytest.raises(ContractError):
        T.theta_from_dict(bad)


def test_theta_from_dict_does_not_mutate_input():
    d = {"f_stop": 2.8, "extra": {"blades": 7}}
    import copy
    d0 = copy.deepcopy(d)
    T.theta_from_dict(d)
    assert d == d0


# --------------------------------------------------------------------------------------------- 노출 시간
@pytest.mark.parametrize("deg,fps,exp", [(180, 24, 1 / 48), (90, 25, 90 / 9000), (360, 24, 1 / 24), (172.8, 24, 0.02),
                                         (45, 120, 1 / 960), (270, 18, 270 / (360 * 18))])
def test_exposure_time(deg, fps, exp):
    t = Theta(shutter_deg=deg, fps=fps)
    v = T.exposure_time_s(t)
    assert type(v) is float
    assert v == pytest.approx(exp, rel=1e-12)


def test_hard_bounds_cover_every_sample_field():
    """표본이 아니라 HARD_BOUNDS 자체의 일관성: 모든 수치 필드와 extra 4개가 키로 있다."""
    num = [f.name for f in dataclasses.fields(Theta) if f.name not in ("lut", "lens_rx", "extra")]
    for n in num:
        assert n in T.HARD_BOUNDS, n
    for n in ("blades", "cpl_angle_deg", "rolloff", "atmos_g"):
        assert n in T.HARD_BOUNDS
    # 경계 의미(반열린 구간) 일관성
    for n in HARD_EXCLUSIVE_HI:
        lo, hi = T.HARD_BOUNDS[n]
        assert hi in (360, 180, 360.0, 180.0)
    assert np.isfinite([v for b in T.HARD_BOUNDS.values() for v in b]).all()
