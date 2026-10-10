"""옥타브 분해: 정본 정의(Chebyshev r, 하드 마스크), 분할·직교·코사인 위치."""
import numpy as np
import pytest
import torch

import dat02_fakes as F
from buc.data import realify as R


@pytest.mark.parametrize("h,w,n", [(64, 48, 5), (16, 16, 4), (7, 9, 3), (64, 64, 6), (32, 32, 1)])
def test_octave_map_matches_oracle(h, w, n):
    om = R.octave_map(h, w, n)
    assert isinstance(om, torch.Tensor) and tuple(om.shape) == (h, w)
    assert not om.is_floating_point()
    np.testing.assert_array_equal(om.numpy().astype(np.int64), F.oracle_octave_map(h, w, n))


def test_max_octaves():
    assert R.max_octaves(64, 48) == 5
    assert R.max_octaves(16, 16) == 4
    assert R.max_octaves(17, 1000) == 4
    assert R.max_octaves(64, 64) == 6


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-5), (torch.float64, 1e-12)])
def test_split_is_partition(dtype, atol):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 64, 48, generator=g, dtype=torch.float64).to(dtype)
    parts = R.octave_split(x, 5)
    assert tuple(parts.shape) == (5, 2, 3, 64, 48)
    assert parts.dtype == dtype
    torch.testing.assert_close(parts.sum(0), x, atol=atol, rtol=0)


def test_split_orthogonal_energy():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 64, 64, generator=g, dtype=torch.float64)
    parts = R.octave_split(x, 6)
    e = (parts ** 2).sum(dim=(1, 2, 3))
    assert float(e.sum()) == pytest.approx(float((x ** 2).sum()), rel=1e-9)
    for a in range(6):
        for b in range(a + 1, 6):
            ip = float((parts[a] * parts[b]).sum())
            assert abs(ip) < 1e-8 * float((x ** 2).sum())
    # 옥타브마다 오라클 사영과 일치
    for o in range(6):
        valid = [i == o for i in range(6)]
        np.testing.assert_allclose(parts[o].numpy(), F.oracle_band_project(x.numpy(), valid), atol=1e-10)


@pytest.mark.parametrize("period,n_oct,expect", [(3, 5, 0), (4, 5, 1), (8, 5, 2), (16, 5, 3), (48, 5, 4), (48, 4, 3)])
def test_cosine_lands_in_expected_octave(period, n_oct, expect):
    h, w = 64, 48
    xx = torch.arange(w, dtype=torch.float64).view(1, w).expand(h, w)
    x = torch.cos(2 * np.pi * xx / period)
    parts = R.octave_split(x, n_oct)
    for o in range(n_oct):
        if o == expect:
            torch.testing.assert_close(parts[o], x, atol=1e-9, rtol=0)
        else:
            assert float(parts[o].abs().max()) < 1e-9, (o, float(parts[o].abs().max()))


def test_dc_goes_to_last_octave():
    x = torch.full((16, 32), 0.7, dtype=torch.float64)
    parts = R.octave_split(x, 3)
    torch.testing.assert_close(parts[2], x, atol=1e-12, rtol=0)
    assert float(parts[:2].abs().max()) < 1e-12


def test_band_project_semantics():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 32, 32, generator=g)
    n = 5
    torch.testing.assert_close(R.band_project(x, [True] * n), x, atol=2e-5, rtol=0)
    assert float(R.band_project(x, [False] * n).abs().max()) < 1e-6
    valid = (False, True, False, True, True)
    p = R.band_project(x, valid)
    assert p.dtype == x.dtype and tuple(p.shape) == tuple(x.shape)
    np.testing.assert_allclose(p.numpy(), F.oracle_band_project(x.numpy().astype(np.float64), valid), atol=2e-5)
    # 멱등
    torch.testing.assert_close(R.band_project(p, valid), p, atol=2e-5, rtol=0)


def test_octave_errors():
    x = torch.zeros(16, 16)
    with pytest.raises(R.RealifyError):
        R.octave_split(x, 0)
    with pytest.raises(R.RealifyError):
        R.band_project(x, [])
    # n_oct = len(valid): 길이 2 는 합법
    c = torch.full((16, 16), 0.25)
    torch.testing.assert_close(R.band_project(c, [False, True]), c, atol=1e-6, rtol=0)
