"""RD-04 공개 표면·상수·옵션/입력 검증·결정론·전역 RNG·입력 불변·결과 불변식·meta·torch 미로드."""
import dataclasses
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

import rd04_helpers as H
from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.read import explain_away as ea
from buc.read import screen_fx as fxm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

NAMES = ("ExplainAwayError", "ExplainOpts", "defocus_mtf", "motion_mtf", "defocus_cutoff_nu", "motion_cutoff_nu",
         "BandMap", "observed_band", "wiener_grain", "wiener_band", "AwayResult", "explain_away")
CONSTS = ("OCTAVES", "OCTAVE_HI", "H_MIN", "BAND_TAPER", "WIENER_K", "COC_MIN_PX", "GHOST_MIN_OCTAVE", "R_RATIO_MIN",
          "HDR_NEAR_SAT",          "OWN_OK", "OWN_SENSOR_BLACK", "OWN_SENSOR_SAT", "OWN_LUT_LOST", "OWN_HALO_TAINT", "OWN_GHOST", "OWN_HDR")

HH, WW = H.H0, H.W0


def scene(seed=0):
    return H.texture_scene(HH, WW, seed)


def img(th=None, seed=0):
    th = th or H.theta(ev=0.2, vignette=0.4, lut="neutral_709")
    return H.look_lin(scene(seed), th), th


def test_public_surface_and_constants():
    for n in NAMES + CONSTS:
        assert hasattr(ea, n), n
        assert n in ea.__all__, n
    assert issubclass(ea.ExplainAwayError, ValueError)
    assert ea.OCTAVES == 4
    assert tuple(ea.OCTAVE_HI) == H.OCTAVE_HI
    assert ea.H_MIN == H.H_MIN and ea.BAND_TAPER == H.BAND_TAPER and ea.WIENER_K == H.WIENER_K
    assert ea.COC_MIN_PX == H.COC_MIN_PX and ea.GHOST_MIN_OCTAVE == H.GHOST_MIN_OCTAVE
    assert ea.R_RATIO_MIN == pytest.approx(1.0 / 16.0)
    assert ea.HDR_NEAR_SAT == 0.97


def test_own_bits_follow_lns08():
    """LNS-08 소유 비트는 그대로, RD-04 비트는 겹치지 않는 2의 거듭제곱."""
    assert (ea.OWN_OK, ea.OWN_SENSOR_BLACK, ea.OWN_SENSOR_SAT, ea.OWN_LUT_LOST, ea.OWN_HALO_TAINT) == (
        chain.OWN_OK, chain.OWN_SENSOR_BLACK, chain.OWN_SENSOR_SAT, chain.OWN_LUT_LOST, chain.OWN_HALO_TAINT)
    assert (ea.OWN_GHOST, ea.OWN_HDR) == (16, 32)
    bits = [ea.OWN_SENSOR_BLACK, ea.OWN_SENSOR_SAT, ea.OWN_LUT_LOST, ea.OWN_HALO_TAINT, ea.OWN_GHOST, ea.OWN_HDR]
    assert len(set(bits)) == 6 and all(b & (b - 1) == 0 and 0 < b < 256 for b in bits)


def test_opts_defaults_and_validation():
    o = ea.ExplainOpts()
    assert (o.grain, o.ghost, o.hdr, o.defocus) == (True, True, True, True)
    assert o.wiener_k == H.WIENER_K and o.grain_nsr == 1.0 and o.h_min == H.H_MIN
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.grain = False
    with pytest.raises(TypeError):
        ea.ExplainOpts(True)                        # kw_only
    for bad in ({"grain": 1}, {"ghost": "yes"}, {"hdr": None}, {"defocus": 0}, {"wiener_k": 0.0}, {"wiener_k": -1.0},
                {"wiener_k": float("nan")}, {"wiener_k": True}, {"grain_nsr": 0.0}, {"grain_nsr": float("inf")},
                {"h_min": 0.0}, {"h_min": 1.0}, {"h_min": 1.5}, {"h_min": float("nan")}):
        with pytest.raises(ValueError):
            ea.ExplainOpts(**bad)
    assert ea.ExplainOpts(wiener_k=np.float64(0.05), h_min=np.float32(0.4)).wiener_k == pytest.approx(0.05)


def test_explain_away_input_validation():
    im, th = img()
    fx = fxm.identity_screen_fx(HH, WW)
    ok = dict(theta=th)
    with pytest.raises(TypeError):
        ea.explain_away(im.tolist(), th)
    with pytest.raises(TypeError):
        ea.explain_away((im * 255).astype(np.uint8), th)
    with pytest.raises(TypeError):
        ea.explain_away(im.astype(np.float16), th)
    for bad in (im[..., 0], im[:, :, :2], im[None], np.zeros((0, 4, 3), np.float32)):
        with pytest.raises(ValueError):
            ea.explain_away(bad, th)
    for v in (np.nan, np.inf, -np.inf, -1e-3):
        b = im.copy()
        b[3, 4, 1] = v
        with pytest.raises(ValueError):
            ea.explain_away(b, th)
    with pytest.raises(TypeError):
        ea.explain_away(im, {"lut": "none"})
    with pytest.raises(TypeError):
        ea.explain_away(im, **ok, screen_fx="fx")
    with pytest.raises(ValueError):
        ea.explain_away(im, th, screen_fx=fxm.identity_screen_fx(HH + 1, WW))
    with pytest.raises(TypeError):
        ea.explain_away(im, th, opts={"grain": True})
    with pytest.raises(TypeError):
        ea.explain_away(im, th, spec="spec")
    for bad in (True, "4", [1.0]):
        with pytest.raises(TypeError):
            ea.explain_away(im, th, coc_px=bad)
        with pytest.raises(TypeError):
            ea.explain_away(im, th, motion_len_px=bad)
    for bad in (-1.0, np.nan, np.inf, np.full((HH, WW + 1), 2.0), -np.ones((HH, WW))):
        with pytest.raises(ValueError):
            ea.explain_away(im, th, coc_px=bad)
        with pytest.raises(ValueError):
            ea.explain_away(im, th, motion_len_px=bad)
    for bad in (True, "1", None):
        with pytest.raises(TypeError):
            ea.explain_away(im, th, r_ratio=bad)
    for bad in (0.0, 0.05, 1.01, np.nan, -1.0):
        with pytest.raises(ValueError):
            ea.explain_away(im, th, r_ratio=bad)
    ea.explain_away(im, th, screen_fx=fx, coc_px=np.full((HH, WW), 2.0, np.float32), motion_len_px=np.int64(3),
                    r_ratio=np.float32(0.5))                      # numpy 스칼라·float32 맵 허용


def test_values_above_one_are_clipped():
    """RD-01 은 1 초과 float32 를 허용한다: 1.0 으로 자르고(포화) 오류가 아니다."""
    im, th = img()
    hi = im.copy()
    hi[10:14, 10:14] = 1.0
    over = im.copy()
    over[10:14, 10:14] = 3.7
    a, b = ea.explain_away(hi, th), ea.explain_away(over, th)
    assert np.array_equal(a.linear, b.linear) and np.array_equal(a.status, b.status)


def test_float64_equals_float32_cast():
    im, th = img()
    a = ea.explain_away(im.astype(np.float64), th)
    b = ea.explain_away(im.astype(np.float64).astype(np.float32), th)
    assert np.array_equal(a.linear, b.linear) and np.array_equal(a.status, b.status)
    assert np.array_equal(a.error_bound, b.error_bound)


def test_result_shapes_dtypes_and_invariants():
    im, th = img()
    fx = H.make_fx(HH, WW, grain=H.grain_fit(0.05))
    r = ea.explain_away(im, th, screen_fx=fx, coc_px=3.0)
    assert isinstance(r, ea.AwayResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.linear = r.linear
    assert r.linear.shape == (HH, WW, 3) and r.linear.dtype == np.float32
    assert r.status.shape == (HH, WW) and r.status.dtype == np.uint8
    assert r.error_bound.shape == (HH, WW, 3) and r.error_bound.dtype == np.float32
    assert r.sigma_grain.shape == (HH, WW) and r.sigma_grain.dtype == np.float32
    assert isinstance(r.band, ea.BandMap)
    assert r.band.cutoff_nu.shape == (HH, WW) and r.band.cutoff_nu.dtype == np.float32
    assert r.band.mask.shape == (4, HH, WW) and r.band.mask.dtype == bool
    assert r.band.min_octave.shape == (HH, WW) and r.band.min_octave.dtype == np.uint8
    assert np.all(np.isfinite(r.linear)) and r.linear.min() >= 0.0
    known = ea.OWN_SENSOR_BLACK | ea.OWN_SENSOR_SAT | ea.OWN_LUT_LOST | ea.OWN_HALO_TAINT | ea.OWN_GHOST | ea.OWN_HDR
    assert not np.any(r.status & ~np.uint8(known))
    assert np.array_equal(r.owned, r.status == 0)
    # error_bound: 소유 화소는 유한 양수, 비소유 화소는 +inf (세 채널 모두)
    own = r.owned
    assert np.all(np.isfinite(r.error_bound[own])) and np.all(r.error_bound[own] > 0)
    assert np.all(np.isinf(r.error_bound[~own]))
    assert own.mean() > 0.5
    # 밴드 지도: 단조(옥타브 b 관측이면 더 낮은 주파수 옥타브 b+1 도 관측), min_octave 와 일치
    m = r.band.mask
    assert not np.any(m[:-1] & ~m[1:])
    mo = np.where(m.any(axis=0), m.argmax(axis=0), 4)
    assert np.array_equal(mo.astype(np.uint8), r.band.min_octave)


def test_determinism_and_purity():
    im, th = img()
    fx = H.make_fx(HH, WW, grain=H.grain_fit(0.05))
    coc = np.full((HH, WW), 2.5)
    before = im.copy()
    st_np, st_py = np.random.get_state(), random.getstate()
    a = ea.explain_away(im, th, screen_fx=fx, coc_px=coc, r_ratio=0.75)
    b = ea.explain_away(im, th, screen_fx=fx, coc_px=coc, r_ratio=0.75)
    assert np.array_equal(im, before) and np.array_equal(coc, np.full((HH, WW), 2.5))
    for f in ("linear", "status", "error_bound", "sigma_grain"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), f
    assert a.band.mask.tobytes() == b.band.mask.tobytes()
    assert a.band.cutoff_nu.tobytes() == b.band.cutoff_nu.tobytes()
    assert a.to_meta() == b.to_meta()
    s2 = np.random.get_state()
    assert st_np[0] == s2[0] and np.array_equal(st_np[1], s2[1]) and st_np[2:] == s2[2:]
    assert st_py == random.getstate()
    a.linear[0, 0, 0] = 99.0                                   # 결과를 바꿔도 다음 호출·입력에 영향 없다
    c = ea.explain_away(im, th, screen_fx=fx, coc_px=coc, r_ratio=0.75)
    assert c.linear.tobytes() == b.linear.tobytes()


def test_to_meta_is_json():
    im, th = img()
    fx = H.make_fx(HH, WW, grain=H.grain_fit(0.05))
    m = ea.explain_away(im, th, screen_fx=fx, coc_px=3.0).to_meta()
    json.dumps(m, allow_nan=False)
    for k in ("size", "owned_frac", "ghost_px", "hdr_px", "band_frac", "cutoff_nu_median", "sigma_grain_mean"):
        assert k in m, k
    assert m["size"] == [HH, WW] and len(m["band_frac"]) == 4
    assert 0.0 <= m["owned_frac"] <= 1.0
    assert m["cutoff_nu_median"] == pytest.approx(H.X_DEFOCUS_HALF / (np.pi * 3.0), rel=1e-3)
    assert m["band_frac"] == [0.0, 0.0, 1.0, 1.0]            # c=3: nu_c=0.235 -> 옥타브 2(0.125), 3 만 관측


def test_spec_is_passed_to_the_chain():
    th = H.theta(lut="neutral_709")
    im = H.look_lin(scene(1), th)
    s8 = sf.SensorSpec(bits=8)
    a = ea.explain_away(im, th, spec=s8)
    ref = chain.explain(H.disp(im), th, spec=s8)
    assert np.array_equal(a.error_bound, np.where((a.status != 0)[..., None], np.inf, ref.error_bound))
    assert not np.array_equal(a.error_bound, ea.explain_away(im, th).error_bound)


def test_no_torch_on_import():
    code = ("import sys; import buc.read.explain_away as s; "
            "bad=[m for m in ('torch','cv2','skimage') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
