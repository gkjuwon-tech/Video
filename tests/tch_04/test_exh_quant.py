"""TCH-04 int8 x̂0 양자화: 반올림 오라클, clamp, 채널별 스케일, 오차 상한, 결정론, 입력 검사."""
import math

import pytest
import torch

from buc.teacher import exhaust as X
import tch04_fakes as F


def test_round_half_even_oracle_and_exact_scale():
    # amax=127 -> scale == 1.0 정확히. 나머지 값은 half-even 반올림.
    vals = [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.4, -3.6, 0.0, 126.5, -126.5]
    x = torch.tensor(vals).reshape(1, 1, 12)
    q, s = X.quantize_x0(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert q.shape == (1, 1, 12) and s.shape == (1,)
    assert s.item() == 1.0
    assert q.flatten().tolist() == [127, 0, 2, 2, 0, -2, -2, 3, -4, 0, 126, -126]


def test_negative_extreme_clamped_to_minus_127_never_minus_128():
    x = torch.tensor([[[-5.0, 5.0, -5.0, 0.1]]])
    q, _ = X.quantize_x0(x)
    assert q.min().item() == -127 and q.max().item() == 127
    q2, _ = X.quantize_x0(-x)
    assert int(q2.min()) >= -127


def test_error_bound_half_scale_per_channel():
    x = F.latent(1, scale=3.0)
    q, s = X.quantize_x0(x)
    d = X.dequantize_x0(q, s)
    assert d.dtype == torch.float32 and not d.requires_grad
    err = (d - x).abs().amax(dim=(1, 2))
    assert torch.all(err <= s / 2 * (1 + 1e-5) + 1e-12)
    amax = x.abs().amax(dim=(1, 2))
    assert torch.allclose(s, amax / 127, rtol=1e-6, atol=0)


def test_relative_rmse_for_gaussian_latent():
    x = F.latent(2, c=16, h=32, w=32)
    q, s = X.quantize_x0(x)
    d = X.dequantize_x0(q, s)
    rel = ((d - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()).item()
    assert rel < 0.015        # 이론값 ~0.009 (255 단계, 4 sigma 범위)


def test_scale_is_per_channel_small_channel_not_crushed():
    x = F.latent(3)
    x[0] *= 1e-3                      # 한 채널만 아주 작다
    x[1] *= 1e3
    q, s = X.quantize_x0(x)
    d = X.dequantize_x0(q, s)
    rel0 = ((d[0] - x[0]).pow(2).mean().sqrt() / x[0].pow(2).mean().sqrt()).item()
    assert rel0 < 0.02                # 전역 스케일이면 ~1e3 배 나빠진다
    assert s[0] < s[1] * 1e-4


def test_zero_channel_and_all_zero_tensor():
    x = F.latent(4)
    x[5] = 0.0
    q, s = X.quantize_x0(x)
    assert s[5].item() == 0.0 and torch.count_nonzero(q[5]) == 0
    d = X.dequantize_x0(q, s)
    assert torch.equal(d[5], torch.zeros_like(d[5])) and torch.isfinite(d).all()
    q0, s0 = X.quantize_x0(torch.zeros(2, 3, 3))
    assert torch.count_nonzero(q0) == 0 and torch.count_nonzero(s0) == 0


def test_deterministic_bitwise_and_input_not_mutated():
    x = F.latent(5)
    before = x.clone()
    a = X.quantize_x0(x)
    b = X.quantize_x0(x.clone())
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(x, before)


def test_requantize_is_idempotent_on_q():
    x = F.latent(6, scale=2.5)
    q, s = X.quantize_x0(x)
    q2, s2 = X.quantize_x0(X.dequantize_x0(q, s))
    assert torch.equal(q, q2)
    assert torch.allclose(s, s2, rtol=1e-5)


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16, torch.float64])
def test_other_float_dtypes_accepted(dt):
    x = F.latent(7).to(dt)
    q, s = X.quantize_x0(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    ref, _ = X.quantize_x0(x.float())
    assert torch.equal(q, ref)


def test_grad_input_is_detached():
    x = F.latent(8).requires_grad_(True)
    q, s = X.quantize_x0(x)
    assert not s.requires_grad
    assert not X.dequantize_x0(q, s).requires_grad


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_rejected(bad):
    x = F.latent(9)
    x[3, 1, 1] = bad
    with pytest.raises(X.ExhaustNumericError):
        X.quantize_x0(x)
    assert issubclass(X.ExhaustNumericError, FloatingPointError)


@pytest.mark.parametrize("shape", [(8, 8), (1, 16, 8, 8), (16,)])
def test_bad_ndim_rejected(shape):
    with pytest.raises(X.ExhaustSpecError):
        X.quantize_x0(torch.zeros(shape))


def test_non_float_input_rejected():
    with pytest.raises(X.ExhaustSpecError):
        X.quantize_x0(torch.zeros(2, 4, 4, dtype=torch.int32))
    assert issubclass(X.ExhaustSpecError, ValueError)


def test_dequantize_matches_formula():
    q = torch.tensor([[[127, -127, 3]], [[1, 2, -4]]], dtype=torch.int8)
    s = torch.tensor([0.5, 0.25], dtype=torch.float32)
    d = X.dequantize_x0(q, s)
    assert d.tolist() == [[[63.5, -63.5, 1.5]], [[0.25, 0.5, -1.0]]]


def test_constants():
    assert X.Q_MAX == 127 and X.QUANT_MODE == "int8-sym-ch-v1"
    assert X.EXT_TTL_S == 72 * 3600
    assert X.PURPOSES == ("resume", "delta", "share", "train")
    assert not math.isnan(X.EXT_TTL_S)
