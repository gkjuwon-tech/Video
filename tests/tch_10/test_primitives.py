"""원시 연산: qmax, (그룹) 양자화 왕복 오차, smoothing 동치, 저랭크 분해."""
import pytest
import torch
from tch10_toys import gen

from buc.teacher import quant as Q


def test_qmax():
    assert Q.qmax(8) == 127 and Q.qmax(4) == 7
    for b in (2, 3, 16, 0):
        with pytest.raises(Q.QuantSpecError):
            Q.qmax(b)
    assert issubclass(Q.QuantSpecError, Q.QuantError) and issubclass(Q.QuantSpecError, ValueError)
    assert Q.MODES == ("int8", "w4a4") and Q.QUANT_TAG == "buc.quant.v1"


@pytest.mark.parametrize("bits", [8, 4])
def test_per_tensor_roundtrip_bound(bits):
    x = torch.randn(64, 96, generator=gen(0)) * 3
    s = Q.absmax_scale(x, bits)
    assert s.dim() == 0 and s.dtype == torch.float32
    assert s.item() == pytest.approx(x.abs().max().item() / Q.qmax(bits), rel=1e-6)
    q = Q.quantize(x, s, bits)
    assert q.dtype == torch.int8 and q.shape == x.shape
    assert q.abs().max().item() == Q.qmax(bits)  # absmax 원소가 정확히 끝값
    d = Q.dequantize(q, s)
    assert d.dtype == torch.float32
    assert (d - x).abs().max().item() <= s.item() / 2 + 1e-6
    torch.testing.assert_close(Q.fake_quant(x, bits), d)


@pytest.mark.parametrize("bits,g", [(4, 64), (8, 32), (4, 16)])
def test_group_roundtrip_bound(bits, g):
    x = torch.randn(5, 7, 128, generator=gen(1))
    x[..., :g] *= 100.0  # 첫 묶음만 크다: 그룹별 scale이면 다른 묶음 오차는 작아야 한다
    s = Q.absmax_scale(x, bits, group_size=g)
    assert s.shape == (5, 7, 128 // g)
    q = Q.quantize(x, s, bits, group_size=g)
    assert q.dtype == torch.int8 and int(q.abs().max()) <= Q.qmax(bits)
    d = Q.dequantize(q, s, group_size=g)
    err = (d - x).abs().reshape(5, 7, 128 // g, g)
    assert (err <= s.unsqueeze(-1) / 2 + 1e-5).all()
    rest = err[..., 1:, :].max().item()
    assert rest <= x[..., g:].abs().max().item() / Q.qmax(bits) / 2 + 1e-5
    pt = (Q.fake_quant(x, bits) - x)[..., g:].abs().max().item()
    assert rest < pt / 10  # 그룹 양자화가 텐서별보다 훨씬 정확


def test_fake_quant_keeps_dtype_and_explicit_scale():
    x = torch.randn(4, 64, generator=gen(2)).half()
    y = Q.fake_quant(x, 8)
    assert y.dtype == torch.float16
    z = Q.fake_quant(torch.tensor([0.26, -1.0, 5.0]), 8, scale=torch.tensor(0.01))
    torch.testing.assert_close(z, torch.tensor([0.26, -1.0, 1.27]))  # 127·0.01에서 클램프


def test_bad_scale_and_group():
    x = torch.randn(3, 64)
    for s in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(Q.QuantSpecError):
            Q.quantize(x, torch.tensor(s), 8)
    with pytest.raises(Q.QuantSpecError):
        Q.absmax_scale(torch.randn(3, 60), 4, group_size=64)
    z = Q.absmax_scale(torch.zeros(8), 8)
    assert z.item() > 0 and torch.isfinite(z)


def test_smooth_factors_formula_and_equivalence():
    g = gen(3)
    a = torch.rand(32, generator=g) * 50
    w = torch.rand(32, generator=g)
    a[0] = 0.0
    for alpha in (0.0, 0.5, 0.8, 1.0):
        s = Q.smooth_factors(a, w, alpha)
        exp = a.clamp_min(1e-5) ** alpha / w.clamp_min(1e-5) ** (1 - alpha)
        torch.testing.assert_close(s, exp, rtol=1e-5, atol=0)
        assert s.dtype == torch.float32 and (s > 0).all()
    # α=0.5: 스무딩 후 활성 absmax와 가중치 열 absmax가 같아진다
    s = Q.smooth_factors(a[1:], w[1:], 0.5)
    torch.testing.assert_close(a[1:] / s, w[1:] * s, rtol=1e-5, atol=1e-6)
    W = torch.randn(16, 32, generator=g, dtype=torch.float64)
    x = torch.randn(10, 32, generator=g, dtype=torch.float64)
    s64 = Q.smooth_factors(a, w, 0.5).double()
    torch.testing.assert_close((x / s64) @ (W * s64).T, x @ W.T, rtol=1e-12, atol=1e-12)
    for bad in (-0.1, 1.5):
        with pytest.raises(Q.QuantSpecError):
            Q.smooth_factors(a, w, bad)
    with pytest.raises(Q.QuantSpecError):
        Q.smooth_factors(a, w[:5], 0.5)


@pytest.mark.parametrize("rank", [0, 1, 8, 200])
def test_lowrank_split(rank):
    W = torch.randn(48, 80, generator=gen(4))
    up, down, R = Q.lowrank_split(W, rank)
    r = min(rank, 48)
    assert up.shape == (48, r) and down.shape == (r, 80) and R.shape == W.shape
    assert up.dtype == down.dtype == R.dtype == torch.float32
    torch.testing.assert_close(up @ down + R, W, rtol=1e-5, atol=1e-5)
    sv = torch.linalg.svdvals(W.double())
    assert R.double().norm().item() == pytest.approx(sv[r:].pow(2).sum().sqrt().item(), rel=1e-4, abs=1e-4)
    if rank == 0:
        torch.testing.assert_close(R, W)
