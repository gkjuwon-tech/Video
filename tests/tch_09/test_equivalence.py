"""Rescue는 수학적 동치다(float64): 출력 불변, 스트림 1/s, SwiGLU 중간값 1/t."""
import copy

import pytest
import torch
from tch09_toys import inputs, make, plan_for
from torch import nn

from buc.teacher import t4_fp16 as R


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize("kind", ["pre", "sand"])
@pytest.mark.parametrize("s,t", [(1.0, 1.0), (32.0, 1.0), (32.0, 16.0), (1024.0, 64.0)])
def test_output_invariant_float64(kind, s, t):
    ref = make(kind)
    m = copy.deepcopy(ref)
    x, c = inputs()
    with torch.no_grad():
        y0 = ref(x, c)
        R.apply_rescue(m, plan_for(m, s, t, islands=False))
        y1 = m(x, c)
    assert _rel(y1, y0) <= 1e-10
    if s > 1:
        assert not torch.equal(m.embed.weight, ref.embed.weight)


@pytest.mark.parametrize("kind", ["pre", "sand"])
def test_islands_do_not_change_float64_result(kind):
    ref = make(kind)
    m = copy.deepcopy(ref)
    x, c = inputs(seed=5)
    with torch.no_grad():
        y0 = ref(x, c)
        R.apply_rescue(m, plan_for(m, 16.0, 8.0, islands=True))
        y1 = m(x, c)
    assert y1.dtype == torch.float64
    assert _rel(y1, y0) <= 2e-5  # 섬 안은 float32라 float64 정밀도는 아님
    assert sum(isinstance(mod, R.Fp32Island) for mod in m.modules()) == len(plan_for(m, 1.0, islands=True).fp32_islands)


def test_missing_eps_compensation_would_break():
    """대조군: eps 보상 없이 1/s만 곱하면 큰 eps에서 출력이 눈에 띄게 바뀐다(테스트의 민감도 확인)."""
    ref = make("pre")
    m = copy.deepcopy(ref)
    x, c = inputs()
    p = plan_for(m, 64.0)
    R.apply_rescue(m, R.RescuePlan(s=p.s, embed=p.embed, stream_params=p.stream_params, gate_rows=p.gate_rows,
                                   residual_out=p.residual_out, swiglu=p.swiglu))
    with torch.no_grad():
        assert _rel(m(x, c), ref(x, c)) > 1e-3


def _capture(model, names, pre=False):
    store = {}
    hs = []
    for n in names:
        mod = model.get_submodule(n)
        if pre:
            hs.append(mod.register_forward_pre_hook(lambda _m, a, n=n: store.__setitem__(n, a[0].detach().clone())))
        else:
            hs.append(mod.register_forward_hook(lambda _m, a, o, n=n: store.__setitem__(n, o.detach().clone())))
    return store, hs


@pytest.mark.parametrize("kind", ["pre", "sand"])
def test_stream_and_swiglu_shrink_exactly(kind):
    s, t = 32.0, 8.0
    ref = make(kind)
    m = copy.deepcopy(ref)
    x, c = inputs()
    blocks = [f"blocks.{i}" for i in range(len(ref.blocks))]
    downs = [f"blocks.{i}.mlp.w_down" for i in range(len(ref.blocks))]
    R.apply_rescue(m, plan_for(m, s, t))
    with torch.no_grad():
        s0, h0 = _capture(ref, blocks)
        d0, g0 = _capture(ref, downs, pre=True)
        ref(x, c)
        s1, h1 = _capture(m, blocks)
        d1, g1 = _capture(m, downs, pre=True)
        m(x, c)
    for h in h0 + g0 + h1 + g1:
        h.remove()
    for b in blocks:  # 블록 출력 = 잔차 스트림
        assert torch.allclose(s1[b] * s, s0[b], rtol=1e-11, atol=1e-12)
    for d in downs:  # W_down 입력 = SwiGLU 중간값
        assert torch.allclose(d1[d] * t, d0[d], rtol=1e-11, atol=1e-12)


def test_unscale_direct_readout():
    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(6, 10)
            self.mix = nn.Linear(10, 10)
            self.read = nn.Linear(10, 3)

        def forward(self, x):
            h = self.embed(x)
            h = h + self.mix(torch.tanh(h * 0 + 1.0))  # 스트림에 쓰는 쪽(입력은 상수)
            return self.read(h)

    torch.manual_seed(3)
    ref = Readout().double()
    m = copy.deepcopy(ref)
    x = torch.randn(4, 6, dtype=torch.float64)
    R.apply_rescue(m, R.RescuePlan(s=16.0, embed=("embed",), residual_out=("mix",), unscale=("read",)))
    with torch.no_grad():
        assert torch.equal(m.read.bias, ref.read.bias)
        assert _rel(m(x), ref(x)) <= 1e-12
