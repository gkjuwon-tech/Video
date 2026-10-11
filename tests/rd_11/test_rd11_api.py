"""RD-11 공개 표면·표(가중치/임계)·검증·순수성: 사양 표를 문자 그대로 대조하고, 입력 오류와 import 위생을 본다."""
import dataclasses
import json
import math
import random
import subprocess
import sys

import numpy as np
import pytest

import rd11_helpers as H
from buc.read import escher_score as E

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
NAMES = ["EscherError", "EscherInputs", "EscComponent", "EscherScore", "MenuHint", "SizeResidual", "PlaneResidual",
         "escher_score", "esc_band", "menu_hint", "size_residual", "plane_residual", "circular_sd_deg",
         "light_residual_deg", "px_at_1080p", "rms_px1080", "light_cue_dirs", "inputs_from_reads", "escher_from_reads",
         "COMPONENTS", "WEIGHTS", "TAU", "GREEN_MAX", "RED_MIN", "CARDS_MIN", "CHEAT_SHARE_MAX", "PX_REF_HEIGHT"]


def test_public_surface_present():
    for n in NAMES:
        assert hasattr(E, n), n


def test_component_order_and_nine_components():
    assert tuple(E.COMPONENTS) == H.ORDER and len(E.COMPONENTS) == 9


def test_weights_and_taus_match_spec_table_exactly():
    for c in H.ORDER:
        assert E.WEIGHTS[c] == pytest.approx(H.W[c], abs=1e-15), c
        assert E.TAU[c] == pytest.approx(H.TAU[c], abs=1e-15), c
    assert set(E.WEIGHTS) == set(H.ORDER) == set(E.TAU)
    assert sum(E.WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)


def test_band_and_misc_constants():
    assert (E.GREEN_MAX, E.RED_MIN, E.CARDS_MIN, E.PX_REF_HEIGHT) == (0.15, 0.35, 0.6, 1080)
    assert E.CHEAT_SHARE_MAX == 0.35
    assert issubclass(E.EscherError, ValueError)


def test_cheat_share_constant_matches_fit_lights():
    from buc.read import fit_lights as F
    assert E.CHEAT_SHARE_MAX == F.CHEAT_SHARE_MAX


# ------------------------------------------------------------------ 입력 검증
@pytest.mark.parametrize("field", ["r_vp_deg", "orphan_frac", "size_sd", "plane_rel", "light_sd_deg", "cheat_share",
                                   "topo_count", "ffg_px1080", "r_dis", "r_refl_px1080"])
@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_inputs_reject_negative_and_nonfinite(field, bad):
    with pytest.raises(E.EscherError):
        E.EscherInputs(**{field: bad})


@pytest.mark.parametrize("field", ["r_vp_deg", "orphan_frac", "r_dis", "topo_count"])
@pytest.mark.parametrize("bad", [True, "0.1", [0.1], object()])
def test_inputs_reject_non_numbers_with_typeerror(field, bad):
    with pytest.raises(TypeError):
        E.EscherInputs(**{field: bad})


def test_fractions_above_one_rejected_and_numpy_scalars_ok():
    with pytest.raises(E.EscherError):
        E.EscherInputs(orphan_frac=1.0001)
    with pytest.raises(E.EscherError):
        E.EscherInputs(cheat_share=1.5)
    ok = E.EscherInputs(orphan_frac=np.float32(0.5), r_vp_deg=np.float64(1.0), topo_count=np.int64(2))
    assert ok.orphan_frac == pytest.approx(0.5) and ok.topo_count == 2


def test_inputs_default_all_none_and_unknown_keyword_rejected():
    e = E.EscherInputs()
    assert all(getattr(e, f.name) is None for f in dataclasses.fields(e))
    with pytest.raises(TypeError):
        E.EscherInputs(r_vp=1.0)
    with pytest.raises(TypeError):
        E.EscherInputs(1.0)          # kw_only


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_tau_scale_validated(bad):
    with pytest.raises(E.EscherError):
        E.escher_score(E.EscherInputs(), tau_scale=bad)


def test_escher_score_requires_inputs_object():
    with pytest.raises(TypeError):
        E.escher_score({"r_dis": 0.1})
    with pytest.raises(TypeError):
        E.escher_score(None)


# ------------------------------------------------------------------ 자료형
def test_dataclasses_frozen():
    s = E.escher_score(H.inputs_at(0.5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.esc = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.components[0].r = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        E.EscherInputs().r_dis = 1.0


def test_score_structure():
    s = E.escher_score(H.inputs_at(0.5))
    assert isinstance(s, E.EscherScore) and len(s.components) == 9
    assert tuple(c.name for c in s.components) == H.ORDER
    assert s.component("r_dis") is s.components[7]
    with pytest.raises(KeyError):
        s.component("nope")
    c = s.component("vp")
    assert (c.tau, c.w, c.unit) == (2.0, 0.20, c.unit) and isinstance(c.unit, str) and c.unit
    assert s.tau_scale == 1.0 and s.inputs == H.inputs_at(0.5)


def test_to_meta_strict_json_and_keys():
    s = E.escher_score(E.EscherInputs(r_vp_deg=1.0, r_dis=0.02))        # 대부분 측정 못 함 -> r=None -> null
    m = s.to_meta()
    txt = json.dumps(m, allow_nan=False)
    back = json.loads(txt)
    for k in ("esc", "band", "components", "pending", "evidence", "dominant", "cards_2_5d"):
        assert k in back, k
    assert set(back["components"]) == set(H.ORDER)
    for c in H.ORDER:
        assert set(back["components"][c]) >= {"r", "tau", "w", "w_eff", "x", "contrib", "measured"}
    assert back["components"]["topo"]["r"] is None and back["components"]["topo"]["measured"] is False
    assert back["components"]["vp"]["r"] == pytest.approx(1.0)
    assert back["esc"] == pytest.approx(s.esc) and back["pending"] == list(s.pending)


# ------------------------------------------------------------------ 순수성·결정론
def test_deterministic_and_inputs_untouched():
    inp = H.inputs_at(0.37)
    snap = dataclasses.asdict(inp)
    a, b = E.escher_score(inp), E.escher_score(inp)
    assert a == b and a.to_meta() == b.to_meta() and dataclasses.asdict(inp) == snap


def test_global_rng_untouched():
    np.random.seed(123)
    st_np, st_py = np.random.get_state(), random.getstate()
    E.escher_score(H.inputs_at(0.4))
    E.plane_residual(H.plane_log_depth(), np.zeros((64, 64), np.int64))
    E.circular_sd_deg([[1, 0, 0], [0, 1, 0]])
    a, b = np.random.get_state(), st_np
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    assert random.getstate() == st_py


def test_import_hygiene_no_torch_scipy_yaml():
    code = ("import sys; import buc.read.escher_score as m; "
            "bad=[k for k in ('torch','scipy','yaml') if k in sys.modules]; assert not bad, bad; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr


def test_nan_never_leaks_into_results():
    s = E.escher_score(E.EscherInputs())
    assert s.esc == 0.0 and math.isfinite(s.evidence)
    for c in s.components:
        assert c.r is None or math.isfinite(c.r)
        assert math.isfinite(c.x) and math.isfinite(c.contrib)
