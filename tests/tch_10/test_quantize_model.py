"""quantize_model 전체: 대상·건너뜀, FP32 섬 존중, 정확도, 이중 적용, 키·보고, 통계 수집기."""
import copy

import pytest
import torch
from tch10_toys import K, MLP, calib, mlp_inputs, rel
from torch import nn

from buc.gov import prov as P
from buc.teacher import quant as Q
from buc.teacher.t4_fp16 import Fp32Island, apply_fp32_islands

XT = mlp_inputs(6, 99)


@pytest.mark.parametrize("mode,tol", [("int8", 0.03), ("w4a4", 0.12)])
def test_end_to_end_accuracy_and_report(mode, tol):
    ref_m = MLP()
    ref = ref_m(XT).detach()
    m = copy.deepcopy(ref_m)
    c = calib(16, seed=0)
    rep = Q.quantize_model(m, Q.QuantSpec(mode), c, weight_prov={"ENGINE"}, batch_size=4)
    assert isinstance(rep, Q.QuantReport) and rep.mode == mode
    if mode == "int8":
        assert rep.quantized == ("down", "head", "odd", "up")
        assert rep.skipped == ("unused",)
    else:
        assert rep.quantized == ("down", "head", "up")
        assert rep.skipped == ("odd", "unused")
    for n in rep.quantized:
        assert isinstance(getattr(m, n), Q.QuantLinear)
    for n in rep.skipped:
        assert type(getattr(m, n)) is nn.Linear
    assert m._buc_quant is rep
    assert rep.calib_digest == c.digest
    assert rep.weight_digest == Q.weight_digest(ref_m)
    assert rep.key == Q.quant_key(Q.QuantSpec(mode), c.digest, rep.weight_digest)
    assert isinstance(rep.prov, P.ProvRecord) and P.train_ok(rep.prov)
    y = m(XT)
    assert y.shape == ref.shape and torch.isfinite(y).all()
    e = rel(y, ref)
    assert e <= tol, e
    assert e > 0  # 실제로 양자화됐다


def test_batch_size_does_not_change_result():
    c = calib(12, seed=1)
    outs, scales = [], []
    for bs in (1, 5, 12):
        m = MLP()
        Q.quantize_model(m, Q.QuantSpec("int8"), c, weight_prov={"ENGINE"}, batch_size=bs)
        outs.append(m(XT))
        scales.append(torch.stack([m.up.a_scale, m.down.a_scale, m.head.a_scale]))
    for o, s in zip(outs[1:], scales[1:]):  # 배치 크기에 따른 부동소수 합산 순서 차이만 허용
        torch.testing.assert_close(s, scales[0], rtol=1e-5, atol=0)
        torch.testing.assert_close(o, outs[0], rtol=1e-3, atol=1e-3)
    assert torch.equal(scales[0][0], scales[1][0])  # 첫 층 입력은 보정 데이터 그 자체


def test_calibration_drives_activation_scale():
    """보정 범위 밖 입력은 클램프된다: 보정셋이 실제로 쓰인다는 증거."""
    m1, m2 = MLP(), MLP()
    c = calib(16)
    small = Q.build_calib_set([Q.CalibSample(s.id, s.data * 0.01, s.prov) for s in c.samples], 16)
    Q.quantize_model(m1, Q.QuantSpec("int8"), c, weight_prov={"ENGINE"})
    Q.quantize_model(m2, Q.QuantSpec("int8"), small, weight_prov={"ENGINE"})
    assert m2.up.a_scale.item() < m1.up.a_scale.item() / 5  # α=0.5: √0.01 = 0.1배
    ref = MLP()(XT).detach()
    assert rel(m2(XT), ref) > 3 * rel(m1(XT), ref)


def test_skip_and_unknown_skip_atomic():
    m = MLP()
    rep = Q.quantize_model(m, Q.QuantSpec("int8", skip=("head",)), calib(8), weight_prov={"ENGINE"})
    assert "head" in rep.skipped and "head" not in rep.quantized and type(m.head) is nn.Linear
    m2 = MLP()
    with pytest.raises(Q.QuantSpecError):
        Q.quantize_model(m2, Q.QuantSpec("int8", skip=("nope",)), calib(8), weight_prov={"ENGINE"})
    assert type(m2.up) is nn.Linear and getattr(m2, "_buc_quant", None) is None


def test_double_quantization_refused():
    m = MLP()
    Q.quantize_model(m, Q.QuantSpec("int8"), calib(8), weight_prov={"ENGINE"})
    with pytest.raises(Q.QuantError):
        Q.quantize_model(m, Q.QuantSpec("w4a4"), calib(8), weight_prov={"ENGINE"})


def test_fp32_islands_untouched():
    m = MLP()
    apply_fp32_islands(m, ["head"])
    assert isinstance(m.head, Fp32Island)
    rep = Q.quantize_model(m, Q.QuantSpec("int8"), calib(8), weight_prov={"ENGINE"})
    assert isinstance(m.head, Fp32Island) and type(m.head.inner) is nn.Linear
    assert "head.inner" in rep.skipped and not any(n.startswith("head") for n in rep.quantized)


def test_key_sensitivity():
    c = calib(8)
    wd = Q.weight_digest(MLP())
    base = Q.quant_key(Q.QuantSpec("int8"), c.digest, wd)
    assert base == Q.quant_key(Q.QuantSpec("int8"), c.digest, wd) and len(base) == 64
    others = {
        Q.quant_key(Q.QuantSpec("w4a4"), c.digest, wd),
        Q.quant_key(Q.QuantSpec("int8", alpha=0.8), c.digest, wd),
        Q.quant_key(Q.QuantSpec("int8", smooth=False), c.digest, wd),
        Q.quant_key(Q.QuantSpec("int8", skip=("up",)), c.digest, wd),
        Q.quant_key(Q.QuantSpec("int8"), calib(8, seed=5).digest, wd),
        Q.quant_key(Q.QuantSpec("int8"), c.digest, Q.weight_digest(MLP(seed=1))),
        Q.quant_key(Q.QuantSpec("int8"), c.digest, wd, engine_hash="e" * 64),
    }
    assert base not in others and len(others) == 7


def test_prov_carries_weight_and_calib_lineage():
    m = MLP()
    w = P.ProvRecord(taint=frozenset({"ENGINE"}), roots=("zimage-turbo",))
    cs = calib(24)
    rep = Q.quantize_model(m, Q.QuantSpec("int8"), cs, weight_prov=w)
    assert "zimage-turbo" in rep.prov.roots and "ENGINE" in rep.prov.taint
    assert rep.prov.ns == "train_ok" and P.prov_bits(rep.prov) == 0


def test_collect_act_absmax():
    m = MLP()
    xs = [mlp_inputs(2, 1), mlp_inputs(3, 2)]
    ref = [o.detach().clone() for o in (m(xs[0]), m(xs[1]))]
    n_hooks = sum(len(mod._forward_pre_hooks) + len(mod._forward_hooks) for mod in m.modules())
    st = Q.collect_act_absmax(m, xs)
    assert set(st) == {"up", "down", "head", "odd"}  # unused는 안 불림
    exp_up = torch.maximum(xs[0].reshape(-1, K).abs().amax(0), xs[1].reshape(-1, K).abs().amax(0))
    torch.testing.assert_close(st["up"], exp_up)
    assert st["down"].shape == (512,) and st["odd"].shape == (48,)
    assert sum(len(mod._forward_pre_hooks) + len(mod._forward_hooks) for mod in m.modules()) == n_hooks
    torch.testing.assert_close(m(xs[0]), ref[0])
    st2 = Q.collect_act_absmax(m, xs, names=["up"], run_fn=lambda mod, b: mod(b * 2))
    assert set(st2) == {"up"}
    torch.testing.assert_close(st2["up"], 2 * exp_up)
    with pytest.raises(Q.QuantError):
        Q.collect_act_absmax(m, [torch.full((1, 2, K), float("nan"))])
    assert sum(len(mod._forward_pre_hooks) + len(mod._forward_hooks) for mod in m.modules()) == n_hooks
    with pytest.raises(Q.QuantSpecError):
        Q.collect_act_absmax(m, xs, names=["nope"])
    apply_fp32_islands(m, ["head"])
    assert "head.inner" not in Q.collect_act_absmax(m, xs) and "head" not in Q.collect_act_absmax(m, xs)
