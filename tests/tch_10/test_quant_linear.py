"""QuantLinear: 버퍼 형식, 에뮬레이션 수식, 정확도(SmoothQuant·저랭크 효과) 수치 기준."""
import copy

import pytest
import torch
from tch10_toys import K, O, acts, linear, rel
from torch import nn

from buc.teacher import quant as Q

XC = acts(512, 2)  # 보정
XT = acts(256, 3)  # 시험(같은 분포, 다른 표본)
AM = XC.abs().amax(0)


def _q(lin, **kw):
    return Q.QuantLinear.from_linear(lin, Q.QuantSpec(**kw), AM)


def test_spec_validation():
    s = Q.QuantSpec("int8", skip=("b", "a", "b"))
    assert s.skip == ("a", "b") and s.w_bits == 8 and s.a_bits == 8
    w = Q.QuantSpec("w4a4")
    assert (w.alpha, w.smooth, w.group_size, w.rank, w.w_bits, w.a_bits) == (0.5, True, 64, 32, 4, 4)
    import json
    assert json.loads(json.dumps(w.to_dict()))["mode"] == "w4a4"
    for bad in (dict(mode="fp8"), dict(mode="int8", alpha=1.5), dict(mode="int8", alpha=-0.1),
                dict(mode="w4a4", group_size=0), dict(mode="w4a4", rank=-1)):
        with pytest.raises(Q.QuantSpecError):
            Q.QuantSpec(**bad)
    with pytest.raises(Exception):
        s.mode = "w4a4"  # frozen


def test_int8_buffers_and_formula():
    lin = linear()
    before = copy.deepcopy(lin.state_dict())
    q = _q(lin, mode="int8", alpha=0.5)
    for k, v in lin.state_dict().items():
        assert torch.equal(v, before[k])  # 원본 불변
    assert q.qweight.dtype == torch.int8 and q.qweight.shape == (O, K)
    assert int(q.qweight.abs().max()) <= 127
    assert q.w_scale.shape == (O,) and q.a_scale.dim() == 0
    assert q.lora_up is None and q.lora_down is None
    s = Q.smooth_factors(AM, lin.weight.detach().abs().amax(0), 0.5)
    torch.testing.assert_close(q.inv_smooth, 1 / s)
    torch.testing.assert_close(q.a_scale, (AM / s).max() / 127)
    # 행 absmax 원소는 정확히 ±127
    assert (q.qweight.abs().amax(1) == 127).all()
    # 수식 재현
    xs = XT * q.inv_smooth
    xa = torch.clamp(torch.round(xs / q.a_scale), -127, 127) * q.a_scale
    ref = xa @ (q.qweight.float() * q.w_scale[:, None]).T + lin.bias.detach()
    torch.testing.assert_close(q(XT), ref, rtol=1e-5, atol=1e-4)
    # dequant_weight·inv_smooth ≈ 원 W (채널별 int8 오차 이내)
    W = lin.weight.detach()
    err = (q.dequant_weight() * q.inv_smooth - W).abs()
    assert (err <= (q.w_scale[:, None] * q.inv_smooth[None, :]) / 2 + 1e-6).all()


def test_w4a4_buffers_and_formula():
    lin = linear()
    q = _q(lin, mode="w4a4", alpha=0.5, rank=32, group_size=64)
    assert q.qweight.dtype == torch.int8 and int(q.qweight.abs().max()) <= 7
    assert q.w_scale.shape == (O, K // 64) and q.a_scale is None
    assert q.lora_up.shape == (O, 32) and q.lora_down.shape == (32, K)
    xs = XT * q.inv_smooth
    xa = Q.fake_quant(xs, 4, group_size=64)
    R = Q.dequantize(q.qweight, q.w_scale, group_size=64)
    ref = xa @ R.T + (xs @ q.lora_down.T) @ q.lora_up.T + lin.bias.detach()
    torch.testing.assert_close(q(XT), ref, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(q.dequant_weight(), R + q.lora_up @ q.lora_down)
    W = lin.weight.detach()
    assert rel(q.dequant_weight() * q.inv_smooth, W) < 0.1


def test_no_smooth_means_identity_inv_smooth_and_no_bias():
    lin = nn.Linear(K, 64, bias=False)
    q = Q.QuantLinear.from_linear(lin, Q.QuantSpec("int8", smooth=False), AM)
    assert torch.equal(q.inv_smooth, torch.ones(K)) and q.bias is None
    y = q(XT)
    assert y.shape == (256, 64)


def test_dtype_and_shape_preserved():
    q = _q(linear(), mode="w4a4")
    x = XT[:6].reshape(2, 3, K)
    for dt in (torch.float32, torch.float16, torch.float64):
        y = q(x.to(dt))
        assert y.dtype == dt and y.shape == (2, 3, O)
        assert torch.isfinite(y).all()


def test_from_linear_rejects():
    with pytest.raises(Q.QuantSpecError):
        Q.QuantLinear.from_linear(linear(), Q.QuantSpec("int8"), AM[:10])
    with pytest.raises(Q.QuantSpecError):
        Q.QuantLinear.from_linear(nn.Linear(60, 8), Q.QuantSpec("w4a4", group_size=64), torch.ones(60))
    with pytest.raises(Q.QuantSpecError):
        Q.QuantLinear.from_linear(nn.Conv1d(4, 4, 1), Q.QuantSpec("int8"), torch.ones(4))


# ---- 수치 수용 기준 -------------------------------------------------------------

def test_int8_smoothquant_accuracy_and_gain():
    lin = linear()
    ref = lin(XT).detach()
    e_s = rel(_q(lin, mode="int8", alpha=0.5)(XT), ref)
    e_08 = rel(_q(lin, mode="int8", alpha=0.8)(XT), ref)
    e_n = rel(_q(lin, mode="int8", smooth=False)(XT), ref)
    assert e_s <= 0.03, e_s
    assert e_08 <= 0.04, e_08
    assert e_n >= 2.5 * e_s, (e_n, e_s)


def test_int8_plain_inputs():
    lin = linear()
    xc, xt = acts(512, 5, outliers=()), acts(256, 6, outliers=())
    q = Q.QuantLinear.from_linear(lin, Q.QuantSpec("int8"), xc.abs().amax(0))
    assert rel(q(xt), lin(xt).detach()) <= 0.025


def test_w4a4_lowrank_and_smooth_gain():
    lin = linear()
    ref = lin(XT).detach()
    e32 = rel(_q(lin, mode="w4a4", rank=32)(XT), ref)
    e0 = rel(_q(lin, mode="w4a4", rank=0)(XT), ref)
    e32n = rel(_q(lin, mode="w4a4", rank=32, smooth=False)(XT), ref)
    assert e32 <= 0.06, e32
    assert e0 >= 1.5 * e32, (e0, e32)
    assert e32n >= 1.5 * e32, (e32n, e32)


def test_w4a4_lowrank_absorbs_structured_weight():
    lin = linear(7, lowrank=4)
    xc, xt = acts(512, 5, outliers=()), acts(256, 6, outliers=())
    am = xc.abs().amax(0)
    ref = lin(xt).detach()
    e0 = rel(Q.QuantLinear.from_linear(lin, Q.QuantSpec("w4a4", rank=0), am)(xt), ref)
    e4 = rel(Q.QuantLinear.from_linear(lin, Q.QuantSpec("w4a4", rank=4), am)(xt), ref)
    assert e4 <= 0.5 * e0, (e4, e0)
    assert e4 <= 0.02


def test_deterministic():
    lin = linear()
    a = _q(lin, mode="w4a4")(XT)
    b = _q(copy.deepcopy(lin), mode="w4a4")(XT)
    assert torch.equal(a, b)
