"""LNS-01: 수제 프리셋 20 — 이름, 서명 제약, 다양성, 출처, 로더 오류 경로."""
import dataclasses
import json
import math
import textwrap

import pytest
import yaml

from buc.core.contracts import ContractError
from buc.lens import theta as T
from lns01_helpers import EXTRA_DEFAULTS, HARD, LENSES, LUTS, PRESETS

NAMES = [
    "neutral_reference", "cinema_scope_anamorphic", "portrait_85_shallow", "wide_landscape_deep", "golden_hour_warm",
    "midday_harsh", "blue_hour_dusk", "night_neon", "foggy_morning", "vintage_petzval_swirl", "super8_home_movie",
    "imax_large_format", "telephoto_compression_sports", "macro_closeup", "high_speed_slowmo", "long_exposure_nd_water",
    "cpl_blue_sky_polarized", "bleach_bypass_war", "teal_orange_blockbuster", "noir_monochrome",
]

SIGNATURES = {
    "neutral_reference": lambda t: (abs(t.focal_mm - 35) < 1e-9 and t.lut == "none" and t.grain == 0
                                    and t.halation == 0 and t.vignette == 0 and t.lens_rx == "spherical_clean"),
    "cinema_scope_anamorphic": lambda t: t.lens_rx == "anamorphic_2x" and abs(t.aspect - 2.39) <= 0.01,
    "portrait_85_shallow": lambda t: t.fov_x_deg() <= 30 and t.f_stop <= 2,
    "wide_landscape_deep": lambda t: t.fov_x_deg() >= 70 and t.f_stop >= 8 and t.focus_m >= 5,
    "golden_hour_warm": lambda t: t.sun_elev_deg is not None and 2 <= t.sun_elev_deg <= 15 and t.lut == "kodak_warm",
    "midday_harsh": lambda t: t.sun_elev_deg is not None and t.sun_elev_deg >= 55,
    "blue_hour_dusk": lambda t: t.sun_elev_deg is not None and -6 <= t.sun_elev_deg <= 0,
    "night_neon": lambda t: t.lut == "neon_night" and t.f_stop <= 2 and t.grain >= 0.4,
    "foggy_morning": lambda t: t.atmos_beta >= 0.005 and t.sun_elev_deg is not None and t.sun_elev_deg <= 20,
    "vintage_petzval_swirl": lambda t: t.lens_rx == "vintage_petzval" and t.vignette >= 0.4,
    "super8_home_movie": lambda t: (abs(t.sensor_width_mm - 5.79) <= 0.01 and abs(t.aspect - 1.3333) <= 0.01
                                    and t.fps == 18 and t.grain >= 0.6),
    "imax_large_format": lambda t: t.sensor_width_mm >= 50 and t.grain <= 0.1,
    "telephoto_compression_sports": lambda t: t.focal_mm >= 200 and t.fps >= 48 and t.shutter_deg <= 180,
    "macro_closeup": lambda t: t.focus_m <= 0.6 and t.f_stop <= 5.6,
    "high_speed_slowmo": lambda t: t.fps >= 120 and t.shutter_deg <= 180,
    "long_exposure_nd_water": lambda t: t.nd_stops >= 4 and t.shutter_deg >= 270 and t.cpl > 0,
    "cpl_blue_sky_polarized": lambda t: t.cpl >= 0.8 and t.sun_elev_deg is not None and 30 <= t.sun_elev_deg <= 70,
    "bleach_bypass_war": lambda t: t.lut == "bleach_bypass" and t.grain >= 0.3,
    "teal_orange_blockbuster": lambda t: t.lut == "teal_orange",
    "noir_monochrome": lambda t: t.lut == "mono_silver" and t.vignette >= 0.3 and t.grain >= 0.3,
}


@pytest.fixture(scope="module")
def presets():
    return T.load_presets()


def test_directory_and_file_names():
    assert PRESETS.is_dir()
    files = sorted(p.name for p in PRESETS.glob("*.yaml"))
    assert files == sorted(n + ".yaml" for n in NAMES)
    assert len(NAMES) == 20 == len(set(NAMES))
    assert not list(PRESETS.glob("*.yml"))


def test_listing_and_loading(presets):
    names = T.list_presets()
    assert isinstance(names, tuple) and list(names) == sorted(NAMES)
    assert set(presets) == set(NAMES)
    for n, p in presets.items():
        assert isinstance(p, T.Preset) and p.name == n
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.name = "x"  # type: ignore[misc]


def test_cwd_independent_and_path_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = T.load_preset("night_neon")
    b = T.load_preset(PRESETS / "night_neon.yaml")
    c = T.load_preset(str(PRESETS / "night_neon.yaml"))
    assert a.theta.content_hash() == b.theta.content_hash() == c.theta.content_hash()
    assert a.name == "night_neon"
    assert len(T.list_presets()) == 20


def test_missing_preset_keyerror():
    with pytest.raises(KeyError):
        T.load_preset("no_such_preset")


def test_yaml_files_wellformed_and_handmade():
    for n in NAMES:
        path = PRESETS / (n + ".yaml")
        text = path.read_text(encoding="utf-8")
        assert "soul" not in text.lower(), n  # 수제(출처 i)만: 닫힌 외부 모델 파생·영감 표기 금지
        d = yaml.safe_load(text)
        assert isinstance(d, dict)
        assert set(d) <= {"name", "description", "provenance", "tags", "theta"}, (n, set(d))
        assert {"name", "description", "provenance", "theta"} <= set(d), n
        assert d["name"] == n
        assert d["provenance"] == "handmade"
        assert isinstance(d["description"], str) and len(d["description"].strip()) >= 10
        assert isinstance(d["theta"], dict) and d["theta"]
        if "tags" in d:
            assert isinstance(d["tags"], list) and all(isinstance(x, str) for x in d["tags"])


def test_each_preset_valid_complete_and_in_bounds(presets):
    for n, p in presets.items():
        t = p.theta
        T.validate_theta(t)
        assert p.provenance == "handmade"
        assert isinstance(p.tags, tuple) and all(isinstance(x, str) for x in p.tags)
        assert set(EXTRA_DEFAULTS) <= set(t.extra), n
        assert type(t.extra["blades"]) is int
        for k, (lo, hi) in HARD.items():
            v = t.extra.get(k) if k in EXTRA_DEFAULTS else getattr(t, k)
            if v is None:
                assert k.startswith("sun_"), (n, k)
                continue
            assert lo <= v <= hi, (n, k, v)
        assert t.lut in LUTS and t.lens_rx in LENSES
        # 노출 시간: 1/4000 s .. 1/8 s
        assert 1 / 4000 <= T.exposure_time_s(t) <= 1 / 8, n
        # 직렬화 왕복
        t2 = T.theta_from_dict(json.loads(json.dumps(T.theta_to_dict(t), allow_nan=False)))
        assert t2.content_hash() == t.content_hash(), n


@pytest.mark.parametrize("name", NAMES)
def test_signature(presets, name):
    t = presets[name].theta
    assert SIGNATURES[name](t), f"{name}: 서명 제약 위반 -> {T.theta_to_dict(t)}"


def test_presets_are_distinct_and_diverse(presets):
    ts = [p.theta for p in presets.values()]
    assert len({t.content_hash() for t in ts}) == 20
    assert len({t.lut for t in ts}) >= 8
    assert len({t.lens_rx for t in ts}) == len(LENSES)  # 렌즈 어휘 5종을 모두 다룬다
    assert len({t.sensor_width_mm for t in ts}) >= 5
    assert len({t.fps for t in ts}) >= 5
    assert len({round(t.aspect, 2) for t in ts}) >= 4
    assert max(t.focal_mm for t in ts) / min(t.focal_mm for t in ts) >= 10
    assert min(t.f_stop for t in ts) <= 1.6 and max(t.f_stop for t in ts) >= 11
    assert min(t.focus_m for t in ts) <= 0.6 and max(t.focus_m for t in ts) >= 8
    assert sum(t.sun_elev_deg is None for t in ts) >= 1
    assert sum(t.sun_elev_deg is not None for t in ts) >= 10
    assert sum(t.grain > 0 for t in ts) >= 8
    assert sum(t.halation > 0 for t in ts) >= 3
    assert sum(t.vignette > 0 for t in ts) >= 5
    assert sum(t.rim > 0 for t in ts) >= 2
    assert sum(t.cpl > 0 for t in ts) >= 2
    assert sum(t.nd_stops > 0 for t in ts) >= 2
    fovs = sorted(t.fov_x_deg() for t in ts)
    assert fovs[0] <= 12 and fovs[-1] >= 80  # 망원부터 광각까지
    assert all(3 <= f <= 130 for f in fovs)


# --------------------------------------------------------------------------------------------- 로더 오류 경로
def _write(tmp_path, body, name="x.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_loader_minimal_file_completes_extra(tmp_path):
    p = _write(tmp_path, """
        name: x
        description: minimal test preset for loader
        provenance: handmade
        theta:
          f_stop: 2.8
          lens_rx: anamorphic_2x
          aspect: 2.39
          extra: {squeeze: 2, blades: 7}
    """)
    pr = T.load_preset(p)
    assert pr.name == "x" and pr.tags == ()
    t = pr.theta
    assert t.f_stop == 2.8 and t.extra["blades"] == 7 and type(t.extra["blades"]) is int
    assert type(t.extra["squeeze"]) is float and t.extra["squeeze"] == 2.0
    assert t.extra["rolloff"] == EXTRA_DEFAULTS["rolloff"]
    assert math.isclose(t.focal_mm, 35.0)


@pytest.mark.parametrize("body", [
    "name: x\ndescription: d d d d d d\nprovenance: handmade\n",  # theta 없음
    "description: d d d d d d\nprovenance: handmade\ntheta: {}\n",  # name 없음
    "name: x\nprovenance: handmade\ntheta: {}\n",  # description 없음
    "name: x\ndescription: d d d d d d\ntheta: {}\n",  # provenance 없음
    "name: x\ndescription: d d d d d d\nprovenance: handmade\ntheta: {bogus: 1}\n",
    "name: x\ndescription: d d d d d d\nprovenance: handmade\ntheta: {f_stop: 100}\n",
    "name: x\ndescription: d d d d d d\nprovenance: handmade\ntheta: {lut: nope}\n",
    "name: x\ndescription: d d d d d d\nprovenance: handmade\ntheta: [1, 2]\n",
    "- just\n- a list\n",
])
def test_loader_rejects_bad_files(tmp_path, body):
    p = _write(tmp_path, body)
    with pytest.raises(ContractError):
        T.load_preset(p)
