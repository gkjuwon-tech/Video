"""RD-03 공개 표면·상수·옵션/입력 검증·결정론·전역 RNG·입력 불변·항등 폴백·meta/Theta 이음매·torch 미로드."""
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import rd03_helpers as H
from buc.core import contracts as C
from buc.read import screen_fx as fx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

NAMES = ("ScreenFxError", "ScreenFxOpts", "VignetteFit", "GrainFit", "CAFit", "HaloBlob", "HalationFit", "GhostSpot",
         "GhostFit", "ScreenFx", "estimate_vignette", "estimate_grain", "estimate_ca", "estimate_halation",
         "estimate_ghost", "estimate_screen_fx", "identity_screen_fx")


def test_public_surface_and_constants():
    for n in NAMES:
        assert hasattr(fx, n), n
        assert n in fx.__all__, n
    assert issubclass(fx.ScreenFxError, ValueError)
    # 정본 상수(LNS-04 / LNS-08 / 계약서)
    assert fx.GRAIN_SIGMA_MAX == H.GRAIN_SIGMA_MAX
    assert fx.VIGNETTE_MAX_DROP == H.VIGNETTE_MAX_DROP
    assert fx.HALATION_THRESHOLD == H.HALATION_THRESHOLD
    assert fx.HALATION_SIGMA_FRAC == H.HALATION_SIGMA_FRAC
    assert tuple(fx.HALATION_TINT) == H.HALATION_TINT
    assert fx.HALO_SOURCE_MAX == H.HALO_SOURCE_MAX
    assert fx.HALO_ASSUMED_SOURCE_Y == H.HALO_ASSUMED_SOURCE_Y


def test_constants_follow_lns04():
    """상수는 LNS-04 정본과 같아야 한다(추정 모델이 순방향과 어긋나면 안 된다)."""
    from buc.lens import sensor_film as sf
    assert fx.GRAIN_SIGMA_MAX == sf.GRAIN_SIGMA_MAX
    assert fx.VIGNETTE_MAX_DROP == sf.VIGNETTE_MAX_DROP
    assert fx.HALATION_THRESHOLD == sf.HALATION_THRESHOLD
    assert fx.HALATION_SIGMA_FRAC == sf.HALATION_SIGMA_FRAC
    assert tuple(fx.HALATION_TINT) == tuple(sf.HALATION_TINT)


def test_opts_defaults_and_validation():
    o = fx.ScreenFxOpts()
    assert (o.max_side, o.block, o.sat_level, o.min_blob_px, o.ghost_max) == (1024, 16, 0.98, 30, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.block = 8
    for kw in ({"max_side": 63}, {"block": 7}, {"block": 65}, {"sat_level": 0.5}, {"sat_level": 1.01},
               {"sat_level": float("nan")}, {"min_blob_px": 3}, {"ghost_max": -1}, {"block": True}):
        with pytest.raises(ValueError):
            fx.ScreenFxOpts(**kw)
    with pytest.raises(TypeError):
        fx.ScreenFxOpts(1024)          # kw_only
    for fn in (fx.estimate_vignette, fx.estimate_grain, fx.estimate_ca, fx.estimate_halation, fx.estimate_ghost,
               fx.estimate_screen_fx):
        with pytest.raises(TypeError):
            fn(np.zeros((64, 64, 3), np.float32), opts={"block": 16})


def test_input_validation():
    good = np.full((64, 80, 3), 0.3, np.float32)
    for fn in (fx.estimate_vignette, fx.estimate_grain, fx.estimate_ca, fx.estimate_halation, fx.estimate_ghost,
               fx.estimate_screen_fx):
        with pytest.raises(TypeError):
            fn([[0.1]])
        with pytest.raises(TypeError):
            fn((good * 255).astype(np.uint8))
        with pytest.raises(ValueError):
            fn(good[..., 0])
        with pytest.raises(ValueError):
            fn(np.concatenate([good, good[..., :1]], axis=2))
        bad = good.copy()
        bad[3, 3, 1] = np.nan
        with pytest.raises(ValueError):
            fn(bad)
        bad[3, 3, 1] = np.inf
        with pytest.raises(ValueError):
            fn(bad)
        with pytest.raises(fx.ScreenFxError):
            fn(np.full((31, 80, 3), 0.3, np.float32))
    with pytest.raises(ValueError):
        fx.estimate_grain(good, jpeg_quality=0)
    with pytest.raises(ValueError):
        fx.estimate_grain(good, jpeg_quality=101)
    with pytest.raises(ValueError):
        fx.estimate_grain(good, jpeg_quality=float("nan"))
    with pytest.raises(ValueError):
        fx.estimate_grain(good, jpeg_quality=True)
    with pytest.raises(ValueError):
        fx.estimate_vignette(good, r_norm=0.0)
    with pytest.raises(ValueError):
        fx.estimate_vignette(good, center=(1.0,))
    with pytest.raises(ValueError):
        fx.estimate_vignette(good, center=(float("nan"), 3.0))


def test_determinism_immutability_and_global_rng():
    x = H.mixed_scene(120, 192, seed=3)
    img = H.full_chain(x, H.theta(grain=0.4, vignette=0.5, lut="none"), seed=1)
    snap = img.copy()
    before = np.random.get_state()
    a = fx.estimate_screen_fx(img)
    b = fx.estimate_screen_fx(img)
    after = np.random.get_state()
    assert np.array_equal(snap, img)
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert a == b                                   # 프로즌 데이터클래스 값 동등(배열 필드 없음)
    assert a.to_meta() == b.to_meta()
    f64 = fx.estimate_screen_fx(img.astype(np.float64))
    assert abs(f64.vignette.strength - a.vignette.strength) < 0.03     # float32/64 가 같은 답
    assert abs(f64.grain.strength - a.grain.strength) < 0.03
    assert (a.size) == (120, 192)


def test_frozen_results():
    r = fx.identity_screen_fx(64, 96)
    for obj in (r, r.vignette, r.grain, r.ca, r.halation, r.ghost):
        assert dataclasses.is_dataclass(obj)
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.some_new_attribute = 1
    with pytest.raises(TypeError):                   # kw_only 생성
        fx.GhostSpot(1.0, 2.0, 3.0, 4.0, 5.0)


def test_identity_screen_fx():
    r = fx.identity_screen_fx(48, 80)
    assert r.size == (48, 80)
    assert r.vignette.strength == 0.0 and r.vignette.corner_ev == 0.0
    g = r.vignette.gain()
    assert g.shape == (48, 80) and g.dtype == np.float32 and np.all(g == 1.0)
    assert r.grain.strength == 0.0 and r.grain.floor_sigma == 0.0
    assert np.all(r.grain.sigma_at(np.linspace(0, 1, 5)) == 0.0)
    assert r.ca.shift_corner_px == (0.0, 0.0) and r.ca.eps_r == 0.0 and r.ca.eps_b == 0.0
    assert r.halation.detected is False and r.halation.n_blobs == 0 and r.halation.amount == 0.0
    assert r.ghost.source is None and len(r.ghost.ghosts) == 0
    m = r.ghost.mask(48, 80)
    assert m.shape == (48, 80) and m.dtype == bool and not m.any()
    assert r.theta_fields() == {"vignette": 0.0, "grain": 0.0, "halation": 0.0}
    assert r.reliable is False                       # 측정하지 않은 값이므로 신뢰하지 않는다
    json.dumps(r.to_meta(), allow_nan=False)
    with pytest.raises(ValueError):
        fx.identity_screen_fx(0, 10)


def test_meta_theta_seam():
    """to_meta 는 JSON 가능, 키 계약, theta_fields 는 Theta 필드로 그대로 쓸 수 있다."""
    x = H.mixed_scene(120, 192, seed=1)
    img = H.full_chain(x, H.theta(grain=0.4, vignette=0.5, halation=0.1, lut="none"), seed=2)
    r = fx.estimate_screen_fx(img)
    meta = r.to_meta()
    json.dumps(meta, allow_nan=False)
    assert {"size", "vignette", "grain", "ca", "halation", "ghost", "warnings", "reliable"} <= set(meta)
    assert meta["size"] == [120, 192]
    assert {"strength", "corner_ev", "coeffs", "center", "r_norm", "reliable"} <= set(meta["vignette"])
    assert {"strength", "radius_px", "floor_sigma", "nonphysical", "channel_corr", "reliable"} <= set(meta["grain"])
    assert {"eps_r", "eps_b", "shift_corner_px", "reliable"} <= set(meta["ca"])
    assert {"detected", "n_blobs", "sigma_frac", "tint", "halo_gain", "amount", "amount_min", "reliable"} <= set(meta["halation"])
    assert {"n", "source", "spots"} <= set(meta["ghost"])
    assert meta["warnings"] == sorted(set(meta["warnings"]))
    tf = r.theta_fields()
    assert set(tf) == {"vignette", "grain", "halation"}
    for v in tf.values():
        assert isinstance(v, float) and 0.0 <= v <= 1.0
    th = C.Theta(**tf, lut="none")                   # 계약 검증 통과
    assert th.vignette == tf["vignette"]
    assert tf["vignette"] == r.vignette.strength and tf["grain"] == r.grain.strength


def test_nan_fields_become_none_in_meta():
    gray = np.repeat(H.rect_scene(96, 128, seed=0, colored=False)[..., :1], 3, axis=2).astype(np.float32)
    r = fx.estimate_screen_fx(gray)
    json.dumps(r.to_meta(), allow_nan=False)


def test_no_torch_on_import():
    code = ("import sys; import buc.read.screen_fx as s; "
            "bad=[m for m in ('torch','cv2','skimage') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
