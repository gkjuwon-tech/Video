"""TCH-02 수식 도우미: add_noise, euler_step."""
import pytest
import torch

from buc.teacher import sample as S


def _rnd(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype)


def test_add_noise_formula_and_dtype():
    x0, e = _rnd(2, 16, 4, 6, seed=1), _rnd(2, 16, 4, 6, seed=2)
    for s in (0.3, 0.5, 1.0):
        z = S.add_noise(x0, s, e)
        assert z.dtype == torch.float32 and z.shape == x0.shape
        torch.testing.assert_close(z, (1 - s) * x0 + s * e, atol=1e-6, rtol=1e-6)
    assert torch.equal(S.add_noise(x0, 1.0, e), e)           # σ=1 -> 순수 노이즈
    torch.testing.assert_close(S.add_noise(x0, 0.0, e), x0, atol=0, rtol=0)   # σ=0 -> 원본
    z16 = S.add_noise(x0.bfloat16(), 0.5, e.bfloat16())
    assert z16.dtype == torch.float32                          # 산술은 float32


def test_add_noise_validation():
    x0 = _rnd(1, 16, 4, 6)
    with pytest.raises(ValueError):
        S.add_noise(x0, 0.5, _rnd(1, 16, 4, 4))
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(S.SigmaError):
            S.add_noise(x0, bad, x0)


def test_euler_step_formula_equals_velocity_euler():
    z, x0 = _rnd(3, 16, 4, 6, seed=3), _rnd(3, 16, 4, 6, seed=4)
    for s, sn in ((1.0, 21 / 22), (0.5, 0.3), (0.3, 0.1), (0.75, 0.0)):
        got = S.euler_step(z, x0, s, sn)
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, (sn / s) * z + (1 - sn / s) * x0, atol=1e-6, rtol=1e-6)
        v = (z - x0) / s                                        # x̂0 = z - σ v  <=>  v = (z - x̂0)/σ
        torch.testing.assert_close(got, z + (sn - s) * v, atol=2e-6, rtol=1e-5)


def test_euler_step_to_zero_is_bit_exact_x0():
    z = _rnd(2, 16, 4, 6, seed=5) * 1e6                          # 커도 정확해야 한다
    x0 = _rnd(2, 16, 4, 6, seed=6)
    out = S.euler_step(z, x0, 0.3, 0.0)
    assert torch.equal(out, x0)


def test_euler_step_recombination_identity():
    # z' = (1-σn) x0 + σn ε̂,  ε̂ = z + (1-σ) v   (x̂0 와 ε̂ 로 다시 합친 것과 같다)
    z, x0 = _rnd(1, 16, 4, 6, seed=7), _rnd(1, 16, 4, 6, seed=8)
    s, sn = 0.642857142857, 0.5
    v = (z - x0) / s
    eps = z + (1 - s) * v
    torch.testing.assert_close(S.euler_step(z, x0, s, sn), (1 - sn) * x0 + sn * eps, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("s,sn", [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.5), (0.5, 0.7), (0.5, -0.1), (1.5, 0.5),
                                  (float("nan"), 0.1), (0.5, float("nan"))])
def test_euler_step_rejects_non_descending(s, sn):
    z = _rnd(1, 16, 4, 6)
    with pytest.raises(S.SigmaError):
        S.euler_step(z, z, s, sn)


def test_euler_step_shape_mismatch():
    with pytest.raises(ValueError):
        S.euler_step(_rnd(1, 16, 4, 6), _rnd(1, 16, 4, 4), 0.5, 0.3)
