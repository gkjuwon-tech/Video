"""원시 스케일 연산, 계획 검증, 원자성, 이중 적용 금지."""
import pathlib
import subprocess
import sys

import pytest
import torch
from tch09_toys import RMS, make, plan_for
from torch import nn

from buc.teacher import t4_fp16 as R


def _lin(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(4, 6).double()


def test_import_has_no_cuda_or_diffusers_side_effects():
    code = ("import sys; import buc.teacher.t4_fp16 as m; "
            "assert 'diffusers' not in sys.modules; assert m.FP16_MAX == 65504.0; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       cwd=str(pathlib.Path(__file__).resolve().parents[2]))
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "ok"


def test_error_hierarchy():
    assert issubclass(R.RescuePlanError, R.RescueError)
    assert issubclass(R.RescuePlanError, ValueError)
    assert issubclass(R.RescueError, RuntimeError)


def test_scale_output_rows_and_full():
    a = _lin()
    w0, b0 = a.weight.detach().clone(), a.bias.detach().clone()
    out = R.scale_output_(a, 0.25, rows=slice(2, 5))
    assert out is a
    exp_w, exp_b = w0.clone(), b0.clone()
    exp_w[2:5] *= 0.25
    exp_b[2:5] *= 0.25
    assert torch.equal(a.weight, exp_w) and torch.equal(a.bias, exp_b)
    R.scale_output_(a, 2.0)
    assert torch.equal(a.weight, exp_w * 2) and torch.equal(a.bias, exp_b * 2)
    assert not a.weight.requires_grad or a.weight.grad is None


def test_scale_input_keeps_bias():
    a = _lin()
    w0, b0 = a.weight.detach().clone(), a.bias.detach().clone()
    R.scale_input_(a, 8.0)
    assert torch.equal(a.weight, w0 * 8) and torch.equal(a.bias, b0)
    x = torch.randn(3, 4, dtype=torch.float64)
    ref = nn.functional.linear(x, w0, b0)
    assert torch.allclose(a(x / 8), ref, rtol=0, atol=1e-12)


def test_scale_affine_and_eps():
    n = RMS(5, eps=0.5)
    with torch.no_grad():
        n.weight.copy_(torch.arange(1.0, 6.0))
    R.scale_affine_(n, 0.5)
    assert torch.equal(n.weight.detach(), torch.arange(1.0, 6.0) * 0.5)
    R.scale_eps_(n, 1 / 16)
    assert n.eps == pytest.approx(0.5 / 16, rel=0, abs=0)
    ln = nn.LayerNorm(5, eps=0.25)
    with torch.no_grad():
        ln.bias.fill_(3.0)
    R.scale_affine_(ln, 0.5)
    assert torch.equal(ln.bias.detach(), torch.full((5,), 1.5))
    R.scale_eps_(ln, 0.25)
    assert ln.eps == 0.0625

    class HF(nn.Module):
        def __init__(self):
            super().__init__()
            self.variance_epsilon = 1e-2
    h = HF()
    R.scale_eps_(h, 0.25)
    assert h.variance_epsilon == pytest.approx(2.5e-3)


def test_primitive_type_errors():
    with pytest.raises(R.RescuePlanError):
        R.scale_affine_(RMS(4, affine=False), 0.5)
    with pytest.raises(R.RescuePlanError):
        R.scale_eps_(nn.Linear(2, 2), 0.5)
    with pytest.raises(R.RescuePlanError):
        R.scale_output_(RMS(4), 0.5)
    with pytest.raises(R.RescuePlanError):
        R.scale_input_(nn.Conv1d(2, 2, 1), 0.5)


@pytest.mark.parametrize("s", [3.0, 0.5, 0.0, -2.0, float("inf"), float("nan"), 12.0])
def test_plan_validate_rejects_bad_scale(s):
    with pytest.raises(R.RescuePlanError):
        R.RescuePlan(s=s, embed=("embed",)).validate()


def test_plan_validate_rules():
    R.RescuePlan(s=1.0).validate()
    R.RescuePlan(s=2.0 ** 10, embed=("a",)).validate()
    with pytest.raises(R.RescuePlanError):
        R.RescuePlan(s=2.0, embed=("a", "a")).validate()
    with pytest.raises(R.RescuePlanError):
        R.RescuePlan(s=2.0, swiglu=(R.SwiGLUSite("u", "g", "d", 3.0),)).validate()
    with pytest.raises(R.RescuePlanError):
        R.RescuePlan(s=2.0, gate_rows=(("ada", 5, 5),)).validate()
    with pytest.raises(R.RescuePlanError):
        R.RescuePlan(s=2.0, gate_rows=(("ada", -1, 4),)).validate()


def _state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.mark.parametrize("bad", [
    dict(embed=("embed", "nope")),
    dict(residual_out=("blocks.0.attn",)),          # Linear도 norm도 아님
    dict(stream_norms=("blocks.0.mlp.w_up",)),      # eps 없음
    dict(gate_rows=(("blocks.0.ada", 32, 999),)),   # 행 범위 초과
    dict(stream_params=("no_such_param",)),
    dict(fp32_islands=("blocks.9.norm1",)),
])
def test_apply_is_atomic_on_invalid_plan(bad):
    m = make("pre")
    before = _state(m)
    eps_before = [mod.eps for mod in m.modules() if hasattr(mod, "eps")]
    base = dict(s=4.0, embed=("embed",), stream_norms=("blocks.0.norm1",))
    base.update(bad)
    with pytest.raises(R.RescuePlanError):
        R.apply_rescue(m, R.RescuePlan(**base))
    after = _state(m)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert eps_before == [mod.eps for mod in m.modules() if hasattr(mod, "eps")]
    assert not any(isinstance(mod, R.Fp32Island) for mod in m.modules())


def test_double_apply_refused_and_report():
    m = make("pre")
    plan = plan_for(m, 8.0, 4.0, islands=True)
    rep = R.apply_rescue(m, plan)
    assert rep.s == 8.0
    assert list(rep.touched) == sorted(rep.touched)
    assert "embed" in rep.touched and "blocks.1.mlp.w_down" in rep.touched
    assert set(rep.islands) == set(plan.fp32_islands)
    w = m.embed.weight.detach().clone()
    with pytest.raises(R.RescueError):
        R.apply_rescue(m, plan)
    assert torch.equal(m.embed.weight.detach(), w)
