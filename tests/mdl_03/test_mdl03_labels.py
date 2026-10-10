"""라벨 해상도 이음매: 옥타브 재매핑(DAT-02 정의와 정합), 라벨 다운샘플 오라클, sem 표."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

import mdl03_helpers as Hp
from buc.data import realify as R
from buc.models.mrh import neck as N
from buc.models.mrh import train as TR


def test_constants():
    assert TR.STRIDE == 4 and TR.MIN_VALID_PX == 16
    assert TR.SEM_TEMP == pytest.approx(0.07)
    assert TR.N_SEM_IDS >= 24  # INV-01 sem id 최대 8+15


@pytest.mark.parametrize("valid,stride,exp", [
    ((True, True, False, True, True, True), 4, (False, True, True, True)),
    ((False, True, True, False, True, True), 2, (True, True, False, True, True)),
    ((True, False), 4, (False,)),
    ((False, False, True), 4, (True,)),
    ((True, False, True), 1, (True, False, True)),
])
def test_band_valid_at_stride(valid, stride, exp):
    assert TR.band_valid_at_stride(valid, stride) == exp


def test_band_valid_at_stride_rejects_non_pow2():
    with pytest.raises(N.MRHSpecError):
        TR.band_valid_at_stride((True, True, True), 3)


@pytest.mark.parametrize("mask_bits", [0b111111, 0b101010, 0b010101, 0b000111, 0b111000, 0b100000])
def test_band_remap_commutes_with_downsample(mask_bits):
    """순수 코사인(주파수 k/64): 전해상 대역 사영 후 4배 평균 풀링 = 풀링 후 1/4 해상 사영(재매핑 마스크)."""
    valid = tuple(bool(mask_bits >> o & 1) for o in range(Hp.N_OCT))
    v4 = TR.band_valid_at_stride(valid, 4)
    assert len(v4) <= R.max_octaves(16, 16)
    i = torch.arange(64, dtype=torch.float64)
    for k in (1, 2, 3, 5, 7):
        x = torch.cos(2 * math.pi * k * i / 64).view(64, 1).expand(64, 64).contiguous()
        lhs = Fn.avg_pool2d(R.band_project(x, valid)[None, None], 4)[0, 0]
        rhs = R.band_project(Fn.avg_pool2d(x[None, None], 4)[0, 0], v4)
        assert torch.allclose(lhs, rhs, atol=1e-9), (k, valid)


def _np_labels(gbuf, sem):
    b, _, h, w = gbuf.shape
    blk = gbuf.reshape(b, 12, h // 4, 4, w // 4, 4).mean(axis=(3, 5))
    mask = (blk[:, 11:12] >= 1 - 1e-6).astype(np.float32)
    ld = blk[:, 0:1] * mask
    n = blk[:, 1:4]
    nn_ = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(nn_ > 1e-6, n / np.maximum(nn_, 1e-12), 0.0) * mask
    alb = blk[:, 4:7] * mask
    return mask, ld, n, alb, sem[:, ::4, ::4].astype(np.int64)


def test_downsample_labels_oracle():
    g = np.random.default_rng(0)
    b, h, w = 2, 32, 48
    gbuf = g.normal(size=(b, 12, h, w)).astype(np.float32)
    lm = np.ones((b, h, w), np.float32)
    lm[0, :8, :8] = 0.0          # 블록 4개 무효
    lm[1, 5, 13] = 0.0           # 부분 블록 1개 무효
    gbuf[:, 11] = lm
    sem = g.integers(0, 30, size=(b, h, w)).astype(np.int64)
    out = TR.downsample_labels(torch.from_numpy(gbuf), torch.from_numpy(sem), 4)
    mask, ld, n, alb, s4 = _np_labels(gbuf, sem)
    assert out["mask"].shape == (b, 1, h // 4, w // 4)
    assert float(out["mask"].sum()) == float(mask.sum()) == b * 8 * 12 - 4 - 1
    np.testing.assert_allclose(out["mask"].numpy(), mask, atol=0)
    np.testing.assert_allclose(out["log_depth"].numpy(), ld, atol=1e-5)
    np.testing.assert_allclose(out["normal"].numpy(), n, atol=1e-5)
    np.testing.assert_allclose(out["albedo"].numpy(), alb, atol=1e-5)
    assert out["sem"].dtype == torch.int64
    np.testing.assert_array_equal(out["sem"].numpy(), s4)


def test_downsample_labels_rejects_bad_size():
    with pytest.raises(TR.MRHDataError):
        TR.downsample_labels(torch.zeros(1, 12, 30, 32), torch.zeros(1, 30, 32, dtype=torch.int64), 4)


def test_default_sem_table():
    torch.manual_seed(3)
    before = torch.get_rng_state().clone()
    a = TR.default_sem_table(64, 16, seed=0)
    b = TR.default_sem_table(64, 16, seed=0)
    c = TR.default_sem_table(64, 16, seed=1)
    assert torch.equal(torch.get_rng_state(), before)
    assert a.shape == (64, 16) and a.dtype == torch.float32
    assert torch.allclose(a.norm(dim=1), torch.ones(64), atol=1e-5)
    assert torch.equal(a, b) and not torch.equal(a, c)
    d = TR.default_sem_table()
    assert d.shape == (TR.N_SEM_IDS, 768)
