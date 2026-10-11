"""LNS-08 공개 표면·상수·검증·격리: 이름, 계약 상수, 입력 오류, 입력 불변, 결정론, torch 미로드."""
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from buc.lens import chain
from lns08_helpers import (BLACK_CODE_EPS, HALO_SOURCE_MAX, HALO_TAINT_EPS, LUT_INV_TOL, OWN_HALO_TAINT, OWN_LUT_LOST, OWN_OK,
                           OWN_SENSOR_BLACK, OWN_SENSOR_SAT, SAT_MARGIN_LSB, WB_MIRED_STOPS, WB_REF_KELVIN,
                           WB_TINT_STOPS, flat, scene_hdr, scene_mid, theta)

REPO = Path(__file__).resolve().parents[2]

NAMES = ("CHAIN_ORDER", "WB_REF_KELVIN", "WB_MIRED_STOPS", "WB_TINT_STOPS", "OWN_OK", "OWN_SENSOR_BLACK",
         "OWN_SENSOR_SAT", "OWN_LUT_LOST", "OWN_HALO_TAINT", "SAT_MARGIN_LSB", "LUT_INV_TOL", "HALO_TAINT_EPS",
         "BLACK_CODE_EPS", "HALO_SOURCE_MAX", "wb_gains", "apply_wb", "remove_wb", "look", "look_frames", "grain_sigma", "invert_look",
         "Explained", "explain")


def test_public_names_and_all():
    for n in NAMES:
        assert hasattr(chain, n), n
    assert set(NAMES) <= set(chain.__all__)


def test_constants_match_contract():
    assert tuple(chain.CHAIN_ORDER) == ("wb", "ev", "vignette", "halation", "sensor", "lut", "grain")
    assert chain.WB_REF_KELVIN == WB_REF_KELVIN and chain.WB_MIRED_STOPS == WB_MIRED_STOPS
    assert chain.WB_TINT_STOPS == WB_TINT_STOPS
    assert (chain.OWN_OK, chain.OWN_SENSOR_BLACK, chain.OWN_SENSOR_SAT, chain.OWN_LUT_LOST, chain.OWN_HALO_TAINT) == (
        OWN_OK, OWN_SENSOR_BLACK, OWN_SENSOR_SAT, OWN_LUT_LOST, OWN_HALO_TAINT) == (0, 1, 2, 4, 8)
    assert chain.SAT_MARGIN_LSB == SAT_MARGIN_LSB and chain.LUT_INV_TOL == LUT_INV_TOL
    assert chain.HALO_TAINT_EPS == HALO_TAINT_EPS == 5e-7 and chain.BLACK_CODE_EPS == BLACK_CODE_EPS
    assert chain.HALO_SOURCE_MAX == HALO_SOURCE_MAX == 16.0


# ------------------------------------------------------------------------------------------------ 입력 검증
@pytest.mark.parametrize("fn", ["look", "apply_wb", "remove_wb"])
def test_forward_input_validation(fn):
    f = getattr(chain, fn)
    th = theta()
    x = flat(0.2, 8, 8)
    with pytest.raises(TypeError):
        f(x.astype(np.uint8), th)
    with pytest.raises(TypeError):
        f(x.astype(np.float16), th)
    with pytest.raises(ValueError):
        f(np.where(np.arange(8)[None, :, None] == 3, np.nan, x).astype(np.float32), th)
    with pytest.raises(ValueError):
        f(x[..., 0], th)
    with pytest.raises(ValueError):
        f(x[..., :2], th)
    with pytest.raises(TypeError):
        f(x, "not a theta")


def test_look_rejects_unregistered_lut_and_bad_seed_frame():
    x = flat(0.2, 8, 8)
    with pytest.raises(ValueError):
        chain.look(x, theta(lut="no_such_look"))
    with pytest.raises(TypeError):
        chain.look(x, theta(), seed=True)
    with pytest.raises(TypeError):
        chain.look(x, theta(), frame=1.0)
    with pytest.raises(ValueError):
        chain.look(x, theta(), frame=1 << 32)
    with pytest.raises(TypeError):
        chain.look(x, theta(), 1)  # seed 는 키워드 전용


def test_explain_input_validation():
    th = theta(lut="neutral_709")
    y = flat(0.4, 8, 8)
    with pytest.raises(TypeError):
        chain.explain(y.astype(np.uint8), th)
    with pytest.raises(TypeError):
        chain.explain(y.astype(np.float16), th)
    with pytest.raises(ValueError):
        chain.explain(y[..., 0], th)
    with pytest.raises(ValueError):
        chain.explain(flat(1.2, 8, 8), th)  # 표시값은 [0,1]
    with pytest.raises(ValueError):
        chain.explain(flat(-0.1, 8, 8), th)
    bad = y.copy()
    bad[2, 2, 1] = np.nan
    with pytest.raises(ValueError):
        chain.explain(bad, th)
    with pytest.raises(TypeError):
        chain.explain(y, "theta")
    with pytest.raises(ValueError):
        chain.explain(y, theta(lut="no_such_look"))
    with pytest.raises(TypeError):
        chain.explain(y, th, spec="spec")


def test_invert_look_and_wb_gains_validation():
    y = flat(0.4, 4, 4)
    with pytest.raises(ValueError):
        chain.invert_look(y, "no_such_look")
    with pytest.raises((TypeError, ValueError)):
        chain.invert_look(y, 3)
    with pytest.raises(ValueError):
        chain.invert_look(flat(1.5, 4, 4), "teal_orange")
    with pytest.raises(ValueError):
        chain.wb_gains(0.0, 0.0)
    with pytest.raises(ValueError):
        chain.wb_gains(float("nan"), 0.0)
    with pytest.raises(ValueError):
        chain.wb_gains(6500.0, float("inf"))
    with pytest.raises(TypeError):
        chain.wb_gains("6500", 0.0)
    with pytest.raises(TypeError):
        chain.look_frames(np.zeros((1, 4, 4, 3), np.float32), theta())
    with pytest.raises(ValueError):
        chain.grain_sigma(flat(1.4, 4, 4), theta(grain=0.3))


# ------------------------------------------------------------------------------------------------ 불변·결정론
def test_inputs_are_not_mutated_and_outputs_are_new_float32():
    th = theta(lut="teal_orange", wb_kelvin=4200.0, wb_tint=0.2, ev=0.5, grain=0.2, halation=0.4, vignette=0.3)
    x = scene_hdr()
    keep = x.copy()
    y = chain.look(x, th, seed=3, frame=2)
    assert np.array_equal(x, keep) and y.dtype == np.float32 and y.shape == x.shape
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    for fn in (chain.apply_wb, chain.remove_wb):
        z = fn(x, th)
        assert np.array_equal(x, keep) and z.dtype == np.float32 and z is not x and not np.shares_memory(z, x)
    yk = y.copy()
    r = chain.explain(y, th)
    assert np.array_equal(y, yk)
    assert r.linear.dtype == np.float32 and r.linear.shape == x.shape
    assert r.status.dtype == np.uint8 and r.status.shape == x.shape[:2]
    assert r.error_bound.dtype == np.float32 and r.error_bound.shape == x.shape
    assert r.owned.dtype == np.bool_ and r.owned.shape == x.shape[:2]
    assert np.array_equal(r.owned, r.status == OWN_OK)
    assert np.all(np.isfinite(r.linear))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.linear = x
    y64 = chain.look(x.astype(np.float64), th, seed=3, frame=2)
    assert y64.dtype == np.float32 and np.max(np.abs(y64 - y)) < 1e-5  # float64 입력도 같은 체인


def test_determinism_and_stream_independence():
    th = theta(lut="kodak_warm", grain=0.4, halation=0.3, vignette=0.2, wb_kelvin=3800.0)
    x = scene_hdr()
    a = chain.look(x, th, seed=5, frame=1)
    b = chain.look(x, th, seed=5, frame=1)
    assert a.tobytes() == b.tobytes()
    assert not np.array_equal(a, chain.look(x, th, seed=6, frame=1))
    assert not np.array_equal(a, chain.look(x, th, seed=5, frame=2))
    n1 = chain.look(x, th, seed=5, frame=1, noise=False)
    n2 = chain.look(x, th, seed=99, frame=7, noise=False)
    assert n1.tobytes() == n2.tobytes()  # noise=False 는 시드·프레임과 무관
    r1, r2 = chain.explain(n1, th), chain.explain(n1, th)
    assert r1.linear.tobytes() == r2.linear.tobytes() and r1.status.tobytes() == r2.status.tobytes()
    assert r1.error_bound.tobytes() == r2.error_bound.tobytes()


def test_no_global_rng_or_state_leak():
    th = theta(lut="teal_orange", grain=0.3)
    x = scene_mid()
    np.random.seed(123)
    st = np.random.get_state()
    chain.look(x, th, seed=1)
    chain.explain(chain.look(x, th, noise=False), th)
    st2 = np.random.get_state()
    assert st[0] == st2[0] and np.array_equal(st[1], st2[1]) and st[2:] == st2[2:]


def test_import_isolation():
    code = (
        "import sys, importlib.util\n"
        "from buc.lens import chain\n"
        "assert 'torch' not in sys.modules, 'torch loaded'\n"
        "spec = importlib.util.find_spec('buc.lens')\n"
        "assert spec.submodule_search_locations is not None\n"
        "print('ok')\n"
    )
    p = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code], cwd=REPO, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr[-800:]
    assert not (REPO / "buc" / "lens" / "__init__.py").exists()  # 네임스페이스 유지
