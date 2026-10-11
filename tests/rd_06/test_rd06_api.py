"""RD-06 공개 표면·검증·순수성: 데이터클래스 범위 경계, 입력 오류, 불변성, torch 미로드, 전역 RNG 불변."""
import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest

from buc.read import tone_color as T

NAMES = ["LUT_N", "MID_GRAY", "OWN_OK", "OWN_CLIP_LO", "OWN_CLIP_HI", "OWN_LUT_LOST", "FilmCurve", "CDL",
         "ToneParams", "film_curve", "film_curve_inv", "apply_cdl", "invert_cdl", "apply_residual_lut", "invert_residual_lut",
         "forward", "inverse", "ToneInverse", "bake_lut", "theta_fields", "fit_residual_lut", "r_look", "SkyAnchor",
         "TonePriors", "ToneOpts", "estimate_tone_color", "ToneColor"]


def test_surface_and_constants():
    for n in NAMES:
        assert hasattr(T, n), n
    assert T.LUT_N == 17 and T.MID_GRAY == 0.18
    assert (T.OWN_OK, T.OWN_CLIP_LO, T.OWN_CLIP_HI, T.OWN_LUT_LOST) == (0, 1, 2, 4)
    if hasattr(T, "__all__"):
        for n in NAMES:
            assert n in T.__all__, n


def test_defaults_are_identity_and_frozen():
    fc, cdl, p = T.FilmCurve(), T.CDL(), T.ToneParams()
    assert (fc.toe, fc.shoulder, fc.slope, fc.white) == (0.0, 0.0, 1.0, 1.0)
    assert cdl.slope == (1.0, 1.0, 1.0) and cdl.offset == (0.0, 0.0, 0.0) and cdl.power == (1.0, 1.0, 1.0) and cdl.sat == 1.0
    assert cdl.is_identity() and not T.CDL(sat=1.2).is_identity() and not T.CDL(offset=(0.01, 0, 0)).is_identity()
    assert (p.ev, p.wb_kelvin, p.wb_tint, p.lut) == (0.0, 6500.0, 0.0, None)
    assert p.curve == fc and p.cdl == cdl and p.is_identity()
    for bad in (T.ToneParams(ev=0.1), T.ToneParams(wb_kelvin=5000), T.ToneParams(wb_tint=0.1),
                T.ToneParams(curve=T.FilmCurve(toe=0.1)), T.ToneParams(cdl=T.CDL(sat=0.9)),
                T.ToneParams(lut=np.zeros((17, 17, 17, 3), np.float32))):
        assert not bad.is_identity()
    for obj, attr in ((fc, "toe"), (cdl, "sat"), (p, "ev")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, 0.5)
    with pytest.raises(TypeError):
        T.FilmCurve(0.1)            # kw_only
    assert T.FilmCurve(toe=0.2) == T.FilmCurve(toe=0.2)


@pytest.mark.parametrize("cls,field,lo,hi", [
    (T.FilmCurve, "toe", 0.0, 1.0), (T.FilmCurve, "shoulder", 0.0, 1.0), (T.FilmCurve, "slope", 0.5, 2.0),
    (T.FilmCurve, "white", 0.5, 2.0), (T.CDL, "sat", 0.05, 2.0),
    (T.ToneParams, "ev", -5.0, 5.0), (T.ToneParams, "wb_kelvin", 2000.0, 12000.0), (T.ToneParams, "wb_tint", -0.6, 0.6)])
def test_scalar_bounds_inclusive(cls, field, lo, hi):
    cls(**{field: lo})
    cls(**{field: hi})
    eps = 1e-6 * max(1.0, abs(hi))
    for v in (lo - eps, hi + eps, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            cls(**{field: v})
    for v in ("0.5", None, [0.5], True):
        with pytest.raises((TypeError, ValueError)):
            cls(**{field: v})
    assert cls(**{field: np.float32(0.5 * (lo + hi))}) is not None        # numpy 스칼라 허용


@pytest.mark.parametrize("field,lo,hi", [("slope", 0.25, 4.0), ("offset", -0.25, 0.25), ("power", 0.4, 2.5)])
def test_cdl_triplets(field, lo, hi):
    T.CDL(**{field: (lo, hi, lo)})
    for bad in ((lo - 1e-3, 1, 1), (1, hi + 1e-3, 1), (1, 1, float("nan"))):
        bad = tuple(min(max(b, -9), 9) if b == b else b for b in bad)
        with pytest.raises(ValueError):
            T.CDL(**{field: bad})
    for bad in ((1.0, 1.0), (1.0, 1.0, 1.0, 1.0), 1.0):
        with pytest.raises((ValueError, TypeError)):
            T.CDL(**{field: bad})
    assert isinstance(getattr(T.CDL(**{field: [lo, lo, lo]}), field), tuple)      # 리스트도 받아 튜플로


def test_lut_field_validation_and_copy():
    base = np.zeros((17, 17, 17, 3), np.float32) + 0.25
    p = T.ToneParams(lut=base)
    assert p.lut.shape == (17, 17, 17, 3) and p.lut.dtype == np.float32
    base[0, 0, 0, 0] = 0.9                                        # 원본을 바꿔도 복사본은 그대로
    assert p.lut[0, 0, 0, 0] == 0.25
    with pytest.raises(ValueError):
        p.lut[0, 0, 0, 0] = 0.5                                   # 읽기 전용
    assert T.ToneParams(lut=np.full((17, 17, 17, 3), 0.5)).lut.dtype == np.float32     # float64 도 받아 float32
    for bad in (np.zeros((16, 16, 16, 3)), np.zeros((17, 17, 17, 4)), np.zeros((17, 17, 17)),
                np.full((17, 17, 17, 3), np.nan), np.full((17, 17, 17, 3), np.inf)):
        with pytest.raises(ValueError):
            T.ToneParams(lut=bad)
    with pytest.raises((TypeError, ValueError)):
        T.ToneParams(curve=(0.1, 0.1, 1, 1))
    with pytest.raises((TypeError, ValueError)):
        T.ToneParams(cdl={"sat": 1.2})
    q = dataclasses.replace(p, ev=0.5)                            # replace 는 검증을 다시 한다
    assert q.ev == 0.5 and np.array_equal(q.lut, p.lut)
    with pytest.raises(ValueError):
        dataclasses.replace(p, ev=9.0)


def test_priors_opts_anchor_validation():
    T.TonePriors(ev=0.0, wb_kelvin=2000.0, wb_tint=0.6)
    for kw in ({"ev": 6.0}, {"wb_kelvin": 1000.0}, {"wb_tint": 0.7}, {"ev_sigma": 0.0}, {"wb_mired_sigma": -1.0},
               {"wb_tint_sigma": float("nan")}, {"lum_median": 0.0}, {"lum_sigma_stops": -1.0}, {"chroma_log_sigma": 0.0}):
        with pytest.raises(ValueError):
            T.TonePriors(**kw)
    with pytest.raises((TypeError, ValueError)):
        T.TonePriors(sky=np.ones((4, 4), bool))                   # SkyAnchor 로 감싸야 한다
    T.SkyAnchor(mask=np.ones((4, 5), bool), luminance=0.5)
    for kw in ({"mask": np.ones((4, 5), np.uint8), "luminance": 0.5}, {"mask": np.ones(5, bool), "luminance": 0.5},
               {"mask": np.ones((4, 5), bool), "luminance": 0.0}, {"mask": np.ones((4, 5), bool), "luminance": float("nan")}):
        with pytest.raises((TypeError, ValueError)):
            T.SkyAnchor(**kw)
    T.ToneOpts(fit_sat=True, max_samples=256)
    for kw in ({"max_samples": 255}, {"min_range_stops": 0.0}, {"poor_fit_stops": -1.0}, {"clip_dominant_frac": 1.5}):
        with pytest.raises(ValueError):
            T.ToneOpts(**kw)
    with pytest.raises((TypeError, ValueError)):
        T.ToneOpts(max_samples=1000.5)


def test_operator_input_validation():
    p = T.ToneParams()
    good = np.full((2, 3, 3), 0.2, np.float32)
    assert T.forward(good, p).shape == (2, 3, 3) and T.forward(good, p).dtype == np.float32
    assert T.forward(np.array([0.2, 0.3, 0.4], np.float32), p).shape == (3,)
    assert T.forward(good.astype(np.float64), p).dtype == np.float32
    assert T.inverse(good, p).linear.dtype == np.float32 and T.inverse(good, p).status.dtype == np.uint8
    assert T.inverse(good, p).status.shape == (2, 3)
    for fn in (T.forward, T.inverse):
        with pytest.raises(TypeError):
            fn(np.full((2, 3, 3), 3, np.int32), p)
        with pytest.raises(ValueError):
            fn(np.full((2, 3, 2), 0.2, np.float32), p)
        with pytest.raises(ValueError):
            fn(np.full((2, 3, 4), 0.2, np.float32), p)
        with pytest.raises(ValueError):
            fn(np.array(0.2, np.float32), p)
        bad = good.copy()
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            fn(bad, p)
        bad[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            fn(bad, p)
        bad[0, 0, 0] = -0.1
        with pytest.raises(ValueError):
            fn(bad, p)
    with pytest.raises(TypeError):
        T.forward(good, "params")
    with pytest.raises(ValueError):
        T.film_curve(np.array([-0.1]), T.FilmCurve())
    with pytest.raises(ValueError):
        T.film_curve(np.array([np.nan]), T.FilmCurve())
    with pytest.raises(ValueError):
        T.film_curve_inv(np.array([-0.1]), T.FilmCurve())
    with pytest.raises((TypeError, ValueError)):
        T.apply_cdl(np.ones((4, 2)), T.CDL())
    with pytest.raises((TypeError, ValueError)):
        T.apply_residual_lut(np.ones((4, 3)), np.zeros((9, 9, 9, 3), np.float32))
    for size in (1, 0, -3, 2.5, True, "65"):
        with pytest.raises((TypeError, ValueError)):
            T.bake_lut(p, size=size)


def test_inputs_not_mutated():
    rng = np.random.default_rng(0)
    L = rng.uniform(0.01, 0.6, (5, 7, 3)).astype(np.float32)
    keep = L.copy()
    p = T.ToneParams(ev=0.3, wb_kelvin=5200, curve=T.FilmCurve(toe=0.3, shoulder=0.3, slope=1.1),
                     cdl=T.CDL(sat=1.1), lut=np.clip(T.fit_residual_lut(L, L * 0.9) , 0, 1))
    D = T.forward(L, p)
    assert np.array_equal(L, keep)
    Dk = D.copy()
    inv = T.inverse(D, p)
    assert np.array_equal(D, Dk)
    assert not np.shares_memory(inv.linear, D)
    img = T.forward(np.random.default_rng(1).uniform(0.01, 0.7, (16, 16, 3)).astype(np.float32), p)
    ik = img.copy()
    T.estimate_tone_color(img)
    assert np.array_equal(img, ik)


def test_import_is_pure_and_torch_free():
    code = ("import sys, numpy as np\n"
            "np.random.seed(123); a = np.random.get_state()[1].copy(); pos = np.random.get_state()[2]\n"
            "from buc.read import tone_color as T\n"
            "b = np.random.get_state()[1]\n"
            "assert (a == b).all() and np.random.get_state()[2] == pos, 'global RNG touched at import'\n"
            "x = np.random.default_rng(0).uniform(0.02, 0.5, (40, 40, 3)).astype(np.float32)\n"
            "T.estimate_tone_color(x); T.forward(x, T.ToneParams(ev=0.2)); T.bake_lut(T.ToneParams(), 9)\n"
            "T.fit_residual_lut(x, x)\n"
            "assert (np.random.get_state()[1] == a).all() and np.random.get_state()[2] == pos, 'global RNG touched by calls'\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "print('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_theta_fields_roundtrip():
    from buc.core.contracts import Theta
    p = T.ToneParams(ev=0.7, wb_kelvin=4300.0, wb_tint=-0.12, curve=T.FilmCurve(toe=0.3))
    f = T.theta_fields(p)
    assert set(f) == {"ev", "wb_kelvin", "wb_tint"}
    assert f == {"ev": 0.7, "wb_kelvin": 4300.0, "wb_tint": -0.12}
    th = Theta(**f)
    th.validate()
    assert (th.ev, th.wb_kelvin, th.wb_tint) == (0.7, 4300.0, -0.12)
    json.dumps(f, allow_nan=False)
