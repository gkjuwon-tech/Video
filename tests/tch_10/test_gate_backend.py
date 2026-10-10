"""판정 정밀도 게이트(Spearman ≥0.9, |ΔD| ≤0.05)와 백엔드 부재 시 동작, import 무부작용."""
import subprocess
import sys

import pytest
from tch10_toys import calib

from buc.gov import prov as P
from buc.teacher import quant as Q


def test_spearman_values():
    assert Q.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert Q.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert Q.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(0.8207826816681233, abs=1e-9)  # 동률 평균 순위
    assert Q.spearman([1, 2, 3], [0.1, 0.1, 0.1]) == 0.0
    assert Q.spearman([1, 5, 2, 8], [0.1, 0.9, 0.2, 7.0]) == pytest.approx(1.0)  # 단조 변환 불변
    for a, b in [([1, 2], [1]), ([1], [1]), ([1, float("nan")], [1, 2])]:
        with pytest.raises(ValueError):
            Q.spearman(a, b)


def test_precision_gate():
    ref = [0.1 * i for i in range(20)]
    near = [r + (0.01 if i % 2 else -0.01) for i, r in enumerate(ref)]
    g = Q.precision_gate(ref, near)
    assert isinstance(g, Q.PrecisionGate) and g.ok and g.n == 20
    assert g.mean_abs_delta == pytest.approx(0.01) and g.max_abs_delta == pytest.approx(0.01)
    assert g.rho >= 0.9
    shifted = [r + 0.08 for r in ref]  # 순위는 완벽, 절대값 편차 초과
    g2 = Q.precision_gate(ref, shifted)
    assert g2.rho == pytest.approx(1.0) and not g2.ok
    shuffled = ref[::-1]  # 평균 편차는 크고 순위 역전
    assert not Q.precision_gate(ref, shuffled).ok
    swap = list(ref)
    for i in range(0, 20, 4):
        swap[i], swap[i + 3] = swap[i + 3], swap[i]  # 국소 교환: rho 하락
    gs = Q.precision_gate(ref, swap)
    assert gs.rho < 0.97
    assert Q.precision_gate(ref, swap, min_rho=0.5, max_delta=1.0).ok
    assert not Q.precision_gate([0.1, 0.2], [0.1, 0.2]).ok  # n<3
    one_out = list(ref)
    one_out[3] += 0.5
    g3 = Q.precision_gate(ref, one_out)
    assert g3.max_abs_delta == pytest.approx(0.5) and g3.mean_abs_delta == pytest.approx(0.025)


def test_backend_probes_and_trt_unavailable(tmp_path):
    assert isinstance(Q.tensorrt_available(), bool) and isinstance(Q.nunchaku_available(), bool)
    c = calib(4)
    with pytest.raises(TypeError):
        Q.build_trt_engine(tmp_path / "m.onnx", tmp_path / "m.plan", calib=[1, 2])
    with pytest.raises(Q.QuantSpecError):
        Q.build_trt_engine(tmp_path / "m.onnx", tmp_path / "m.plan", calib=c, precision="int4")
    bad = Q.CalibSet(samples=c.samples, prov=P.ProvRecord(taint=frozenset({"USER"})), digest=c.digest)
    with pytest.raises(P.FirewallError):
        Q.build_trt_engine(tmp_path / "m.onnx", tmp_path / "m.plan", calib=bad)
    if not Q.tensorrt_available():
        with pytest.raises(Q.BackendUnavailable):
            Q.build_trt_engine(tmp_path / "m.onnx", tmp_path / "m.plan", calib=c)
        assert not (tmp_path / "m.plan").exists()
    assert issubclass(Q.BackendUnavailable, Q.QuantError)


def test_import_has_no_heavy_side_effects():
    code = ("import sys; import buc.teacher.quant as q; "
            "bad=[m for m in ('tensorrt','nunchaku','diffusers','onnx') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
