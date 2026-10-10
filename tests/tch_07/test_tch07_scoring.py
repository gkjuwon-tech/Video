"""상수·사전 등록·z 정규화·ES 가중합·세계당 T4-h (손계산 대조)."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest
from tch07_support import metrics

from buc.teacher import tct as T


def test_constants_match_catalog():
    assert T.CRITERIA == ("K1", "K2", "K3", "K4", "K5", "K6", "K7")
    assert tuple(T.WEIGHTS_L1[k] for k in T.CRITERIA) == pytest.approx((0.10, 0.10, 0.20, 0.20, 0.15, 0.10, 0.15), abs=1e-12)
    assert tuple(T.WEIGHTS_L2B[k] for k in T.CRITERIA) == pytest.approx((0.30, 0.20, 0.05, 0.15, 0.10, 0.10, 0.10), abs=1e-12)
    assert sum(T.WEIGHTS_L1.values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(T.WEIGHTS_L2B.values()) == pytest.approx(1.0, abs=1e-12)
    assert (T.Z_CLIP, T.TIE_PP, T.SWAP_PP, T.NVF_R_ENTRY) == (0.5, 0.02, 0.05, 0.30)
    assert (T.LOCK_DAY, T.GATE_DAY, T.WINDOW_DAYS, T.ENGINE_FREEZE_DAY) == (15, 18, (45, 75), 60)
    assert T.INCUMBENT == "z-image-turbo"
    assert T.T4H_PER_TFU == pytest.approx(1 / 1385, rel=1e-12)


def test_errors_are_value_errors():
    for e in (T.FirewallError, T.PreRegistrationError, T.SelectionError):
        assert issubclass(e, T.TctError)
    assert issubclass(T.TctError, ValueError)


def test_import_is_light_and_never_imports_eval():
    code = ("import sys; import buc.teacher.tct; "
            "bad=[m for m in ('diffusers','transformers','huggingface_hub','lpips','eval') if m in sys.modules]; "
            "print('BAD' if bad else 'OK', bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("OK"), out.stdout
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
        assert not any(x == "eval" or x.startswith("eval.") for x in names)


def test_prereg_defaults_and_key_sensitivity():
    p = T.PREREG
    assert p.validate() is p or p.validate() is None
    assert p.key() == T.PreRegistration().key() and len(p.key()) == 64
    assert p.key() != T.PreRegistration(tie_pp=0.03).key()
    assert p.key() != T.PreRegistration(swap_pp=0.04).key()
    w = list(p.w_l1)
    w[0], w[1] = w[0] + 0.01, w[1] - 0.01
    assert p.key() != T.PreRegistration(w_l1=tuple(w)).key()
    w2 = list(p.w_l2b)
    w2[2], w2[3] = w2[2] + 0.01, w2[3] - 0.01
    assert p.key() != T.PreRegistration(w_l2b=tuple(w2)).key()


@pytest.mark.parametrize("kw", [
    dict(w_l1=(0.2,) * 7),                                   # 합 1.4
    dict(w_l1=(0.5, 0.5, 0.5, -0.5, 0.0, 0.0, 0.0)),         # 음수
    dict(w_l2b=(0.5, 0.5)),                                  # 길이
    dict(tie_pp=0.06),                                       # 동률 폭 < 교체 문턱이어야 한다
    dict(tie_pp=-0.01),
])
def test_prereg_rejects_bad(kw):
    with pytest.raises(T.PreRegistrationError):
        T.PreRegistration(**kw).validate()


def test_normalize_clip_and_abs_denominator():
    ref = {"K1": 2.0, "K2": 4.0, "K3": 5.0, "K4": 10.0, "K5": 1.0, "K6": 0.5, "K7": -2.0}
    m = {"K1": 2.2, "K2": 3.0, "K3": 7.5, "K4": 20.0, "K5": 1.0, "K6": 0.55, "K7": -1.0}
    z = T.normalize(m, ref)
    assert [z[k] for k in T.CRITERIA] == pytest.approx([0.1, -0.25, 0.5, 0.5, 0.0, 0.1, 0.5], abs=1e-12)
    lo = T.normalize({**m, "K2": 0.0, "K5": -5.0}, ref)
    assert lo["K2"] == -0.5 and lo["K5"] == -0.5          # 하한 클립
    assert T.normalize(ref, ref) == {k: 0.0 for k in T.CRITERIA}


def test_engine_score_hand_computed():
    ref = {"K1": 2.0, "K2": 4.0, "K3": 5.0, "K4": 10.0, "K5": 1.0, "K6": 0.5, "K7": -2.0}
    m = {"K1": 2.2, "K2": 3.0, "K3": 7.5, "K4": 20.0, "K5": 1.0, "K6": 0.55, "K7": -1.0}
    assert T.engine_score(m, ref, T.WEIGHTS_L1) == pytest.approx(0.27, abs=1e-12)
    assert T.engine_score(m, ref, T.WEIGHTS_L2B) == pytest.approx(0.14, abs=1e-12)
    assert T.engine_score(ref, ref, T.WEIGHTS_L1) == 0.0
    assert T.engine_score(metrics(0.03), metrics(0.0), T.WEIGHTS_L1) == pytest.approx(0.03, abs=1e-12)


@pytest.mark.parametrize("bad_m,bad_ref", [
    ({"K1": 1.0}, None),                                     # 키 부족
    ({**metrics(), "K8": 1.0}, None),                        # 키 초과
    ({**metrics(), "K3": float("nan")}, None),
    (None, {**metrics(), "K3": 0.0}),                        # 기준 0
    (None, {**metrics(), "K3": float("inf")}),
])
def test_normalize_rejects(bad_m, bad_ref):
    with pytest.raises(T.TctError):
        T.normalize(bad_m or metrics(), bad_ref or metrics())


def test_world_t4h_mix():
    z = {"green": 1.0, "yellow": 1.6, "red": 1.9}
    assert T.world_t4h(z) == pytest.approx(1.345, abs=1e-12)
    assert T.world_t4h({"green": 2.0, "yellow": 2.0, "red": 2.0}) == pytest.approx(2.0, abs=1e-12)
    assert sum(T.WORLD_MIX.values()) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(T.TctError):
        T.world_t4h({"green": 1.0, "yellow": 1.6})
    with pytest.raises(T.TctError):
        T.world_t4h({"green": 1.0, "yellow": -1.6, "red": 1.9})
