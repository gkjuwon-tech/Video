"""FP16 실증: 원본은 FP16에서 터지고, 프로파일→choose_scale→rescue+섬+fp32 SDPA는 FP32 기준에 붙는다."""
import copy

import torch
from tch09_toys import inputs, make, plan_for

from buc.teacher import t4_fp16 as R

NORMS = ("blocks.0.norm1", "blocks.0.norm2", "blocks.1.norm1", "blocks.1.norm2", "final_norm")


def hot_model():
    """스트림(임베딩 출력 ~1e5)과 SwiGLU 곱(~1e5)이 FP16 범위를 넘는 장난감. 가중치 자체는 FP16에 들어간다."""
    m = make("pre", seed=11, dtype=torch.float32)
    g = torch.Generator().manual_seed(12)
    with torch.no_grad():
        m.embed.weight.copy_(torch.randn(m.embed.weight.shape, generator=g) * 1.5e4)
        for b in m.blocks:
            b.mlp.w_up.weight.copy_(torch.randn(b.mlp.w_up.weight.shape, generator=g) * 30)
            b.mlp.w_gate.weight.copy_(torch.randn(b.mlp.w_gate.weight.shape, generator=g) * 30)
            b.mlp.w_down.weight.copy_(torch.randn(b.mlp.w_down.weight.shape, generator=g) * 1e-3)
    return m


def _profile(m, x, c):
    blocks = [f"blocks.{i}" for i in range(len(m.blocks))]
    downs = [f"blocks.{i}.mlp.w_down" for i in range(len(m.blocks))]
    with R.MaxAbsProfiler(m, names=["embed", *blocks, *downs]) as prof, torch.no_grad():
        y = m(x, c)
    stream = {k: prof.out_max[k] for k in ["embed", *blocks]}
    inner = {k: prof.in_max[k] for k in downs}
    return y, stream, inner


def test_fp16_rescue_end_to_end():
    ref = hot_model()
    x, c = inputs(seed=13, dtype=torch.float32)
    y_ref, stream, inner = _profile(ref, x, c)
    assert torch.isfinite(y_ref).all()
    assert max(stream.values()) > R.FP16_MAX and max(inner.values()) > R.FP16_MAX  # 전제: 정말 넘친다

    s = R.choose_scale(stream)
    t = R.choose_scale(inner)
    assert s >= 2 and t >= 2
    assert max(stream.values()) / s <= R.FP16_MAX / 8 and max(inner.values()) / t <= R.FP16_MAX / 8

    # 대조군 1: 섬과 fp32 SDPA만으로는(rescue 없음) 비유한
    plain = copy.deepcopy(ref)
    R.apply_fp32_islands(plain, NORMS)
    R.cast_fp16(plain)
    with torch.no_grad(), R.fp32_sdpa():
        y_plain = plain(x.half(), c.half())
    assert not torch.isfinite(y_plain).all()

    # 대조군 2: 잔차 1/s만 하고 SwiGLU 재균형이 없으면(t=1) 여전히 비유한
    half_fix = copy.deepcopy(ref)
    R.apply_rescue(half_fix, plan_for(half_fix, s, 1.0, islands=True))
    R.cast_fp16(half_fix)
    with torch.no_grad(), R.fp32_sdpa():
        y_half_fix = half_fix(x.half(), c.half())
    assert not torch.isfinite(y_half_fix).all()

    # 본 실험
    m = copy.deepcopy(ref)
    R.apply_rescue(m, plan_for(m, s, t, islands=True))
    out = R.cast_fp16(m)
    assert out is m
    assert m.embed.weight.dtype == torch.float16 and m.blocks[0].mlp.w_down.weight.dtype == torch.float16
    isl = m.get_submodule("blocks.0.norm1")
    assert isinstance(isl, R.Fp32Island) and isl.inner.weight.dtype == torch.float32
    with torch.no_grad(), R.fp32_sdpa():
        y = m(x.half(), c.half())
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
    rel = ((y.float() - y_ref).abs().max() / y_ref.abs().max()).item()
    assert rel <= 3e-2, rel


def test_rescue_is_deterministic():
    ref = hot_model()
    x, c = inputs(seed=13, dtype=torch.float32)
    outs = []
    for _ in range(2):
        m = copy.deepcopy(ref)
        R.apply_rescue(m, plan_for(m, 32.0, 32.0, islands=True))
        R.cast_fp16(m)
        with torch.no_grad(), R.fp32_sdpa():
            outs.append(m(x.half(), c.half()))
    assert torch.equal(outs[0], outs[1])
