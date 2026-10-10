"""FP32 섬과 FP32 누산 SDPA."""
import math

import pytest
import torch
import torch.nn.functional as F
from tch09_toys import RMS
from torch import nn

from buc.teacher import t4_fp16 as R


class Box(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMS(16, eps=1e-6)
        self.lin = nn.Linear(16, 16)

    def forward(self, x):
        return self.lin(self.norm(x))


def test_island_survives_half_and_bf16_and_computes_in_fp32():
    torch.manual_seed(0)
    b = Box()
    names = R.apply_fp32_islands(b, ["norm"])
    assert names == ["norm"]
    assert isinstance(b.norm, R.Fp32Island) and isinstance(b.norm.inner, RMS)
    assert R.apply_fp32_islands(b, ["norm"]) in ([], ["norm"])  # 멱등
    assert isinstance(b.norm.inner, RMS)  # 이중 감싸기 없음
    b.half()
    assert b.lin.weight.dtype == torch.float16 and b.norm.inner.weight.dtype == torch.float32
    x = torch.randn(3, 16) * 1e3  # x² ~ 1e6: FP16 norm이면 넘친다
    y_isl = b.norm(x.half())
    assert y_isl.dtype == torch.float16
    ref = RMS(16, eps=1e-6)(x)
    assert torch.allclose(y_isl.float(), ref, rtol=2e-3, atol=2e-3)
    naive = RMS(16, eps=1e-6).half()(x.half())
    assert not torch.allclose(naive.float(), ref, rtol=1e-1, atol=1e-1)  # 섬 없으면 틀린다(전제 확인)
    b.to(torch.bfloat16)
    assert b.norm.inner.weight.dtype == torch.float32
    assert b.norm(x.bfloat16()).dtype == torch.bfloat16


def test_island_tuple_outputs_and_non_tensor_args():
    class Two(nn.Module):
        def forward(self, a, k, *, flag=True):
            assert a.dtype == torch.float32 and isinstance(k, int) and flag is False
            return a * k, [a.sum()], "tag"

    m = nn.Module()
    m.two = Two()
    R.apply_fp32_islands(m, ["two"])
    a = torch.ones(4, dtype=torch.float16)
    out = m.two(a, 3, flag=False)
    assert out[0].dtype == torch.float16 and torch.equal(out[0], torch.full((4,), 3.0, dtype=torch.float16))
    assert out[1][0].dtype == torch.float16 and out[2] == "tag"


def test_cast_fp16_keeps_buffers_in_islands():
    m = nn.Module()
    m.ln = nn.BatchNorm1d(4)
    m.lin = nn.Linear(4, 4)
    R.apply_fp32_islands(m, ["ln"])
    R.cast_fp16(m)
    assert m.ln.inner.running_mean.dtype == torch.float32
    assert m.lin.weight.dtype == torch.float16


def test_apply_islands_unknown_name_is_atomic():
    b = Box()
    with pytest.raises(R.RescuePlanError):
        R.apply_fp32_islands(b, ["norm", "missing"])
    assert isinstance(b.norm, RMS)


def _ref_attn(q, k, v, mask=None, causal=False, scale=None):
    q, k, v = q.double(), k.double(), v.double()
    sc = scale if scale is not None else 1 / math.sqrt(q.shape[-1])
    a = q @ k.transpose(-1, -2) * sc
    if causal:
        tri = torch.ones(a.shape[-2], a.shape[-1], dtype=torch.bool).tril()
        a = a.masked_fill(~tri, float("-inf"))
    if mask is not None:
        a = a.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else a + mask.double()
    return torch.softmax(a, -1) @ v


@pytest.mark.parametrize("variant", ["plain", "bool_mask", "causal", "scale", "add_mask"])
def test_sdpa_fp32_accuracy_with_huge_logits(variant):
    g = torch.Generator().manual_seed(4)
    q = (torch.randn(2, 3, 9, 64, generator=g) * 40).half()  # 로짓 ~1e5: FP16 누산이면 넘친다
    k = (torch.randn(2, 3, 9, 64, generator=g) * 40).half()
    v = torch.rand(2, 3, 9, 64, generator=g).half() * 2 - 1
    kw, ref_kw = {}, {}
    if variant == "bool_mask":
        mask = torch.rand(9, 9, generator=g) > 0.3
        mask |= torch.eye(9, dtype=torch.bool)
        kw["attn_mask"] = ref_kw["mask"] = mask
    if variant == "add_mask":
        mask = (torch.randn(9, 9, generator=g) * 50).half()
        kw["attn_mask"] = ref_kw["mask"] = mask
    if variant == "causal":
        kw["is_causal"] = ref_kw["causal"] = True
    if variant == "scale":
        kw["scale"] = ref_kw["scale"] = 0.01
    out = R.sdpa_fp32(q, k, v, **kw)
    assert out.dtype == torch.float16 and out.shape == q.shape
    ref = _ref_attn(q, k, v, **ref_kw)
    assert torch.isfinite(out).all()
    assert (out.double() - ref).abs().max().item() <= 2e-3


def test_fp32_sdpa_context_patches_and_restores():
    orig = F.scaled_dot_product_attention
    g = torch.Generator().manual_seed(5)
    q, k, v = (torch.randn(1, 2, 5, 8, generator=g).half() for _ in range(3))
    with R.fp32_sdpa():
        assert F.scaled_dot_product_attention is not orig
        assert torch.nn.functional.scaled_dot_product_attention is not orig
        assert torch.equal(F.scaled_dot_product_attention(q, k, v), R.sdpa_fp32(q, k, v))
    assert F.scaled_dot_product_attention is orig
    with pytest.raises(KeyError), R.fp32_sdpa():
        raise KeyError("x")
    assert F.scaled_dot_product_attention is orig
