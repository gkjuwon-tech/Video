"""TCH-01 결정론 모드와 make_noise(Philox WNF)."""
import os
import random

import numpy as np
import pytest
import torch

from buc.core import det
from buc.teacher import zimage as Z


def test_determinism_mode_sets_all_flags_and_reports(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.backends.cuda.matmul.allow_tf32 = True      # 켜 둔 상태에서 꺼지는지 본다
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    rep = Z.enable_determinism_mode(seed=123)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert rep == {
        "seed": 123, "deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
        "tf32_matmul": False, "tf32_cudnn": False, "bf16_reduced_precision_reduction": False,
        "fp16_reduced_precision_reduction": False, "cublas_workspace_config": ":4096:8",
    }


def test_determinism_mode_keeps_user_cublas_setting(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    rep = Z.enable_determinism_mode()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8" and rep["cublas_workspace_config"] == ":16:8"


def test_determinism_mode_seeds_everything_reproducibly():
    Z.enable_determinism_mode(seed=99)
    a = (torch.rand(5), np.random.rand(3), random.random())
    Z.enable_determinism_mode(seed=99)
    b = (torch.rand(5), np.random.rand(3), random.random())
    assert torch.equal(a[0], b[0]) and (a[1] == b[1]).all() and a[2] == b[2]
    Z.enable_determinism_mode(seed=100)
    assert not torch.equal(a[0], torch.rand(5))


def test_determinism_mode_without_seed_leaves_rng_state():
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    t0, n0, p0 = torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()
    rep = Z.enable_determinism_mode()
    assert rep["seed"] is None
    assert torch.equal(t0, torch.get_rng_state())
    assert (n0 == np.random.get_state()[1]).all() and p0 == random.getstate()


def _oracle(seed, shape, level=0, k=0):
    n = int(np.prod(shape))
    z = det.wnf(seed, level, np.arange(n, dtype=np.int64), k, channels=1, dtype=np.float32)[..., 0]
    return torch.from_numpy(np.ascontiguousarray(z.reshape(shape)))


@pytest.mark.parametrize("seed,shape,level,k", [(7, (2, 16, 8, 8), 0, 0), (2**40 + 5, (1, 16, 4, 6), 3, 9),
                                                (0, (5,), 0, 0)])
def test_make_noise_equals_wnf_oracle_bitwise(seed, shape, level, k):
    got = Z.make_noise(seed, shape, level=level, k=k)
    assert got.dtype == torch.float32 and got.device.type == "cpu" and tuple(got.shape) == shape
    assert torch.equal(got, _oracle(seed, shape, level, k))


def test_make_noise_independent_of_global_rng_and_batch_size():
    a = Z.make_noise(11, (2, 16, 8, 8))
    torch.manual_seed(1)
    torch.rand(100)
    b = Z.make_noise(11, (2, 16, 8, 8))
    assert torch.equal(a, b)
    assert torch.equal(Z.make_noise(11, (1, 16, 8, 8)), a[:1])      # 배치 접두 불변
    assert torch.equal(Z.make_noise(11, (3, 16, 8, 8))[:2], a)


def test_make_noise_varies_with_seed_level_k():
    base = Z.make_noise(1, (1, 16, 8, 8))
    for kw in ({"seed": 2}, {"level": 1}, {"k": 1}):
        args = dict(seed=1, level=0, k=0)
        args.update(kw)
        other = Z.make_noise(args["seed"], (1, 16, 8, 8), level=args["level"], k=args["k"])
        assert not torch.equal(base, other), kw


def test_make_noise_dtype_cast_is_cast_of_float32():
    f = Z.make_noise(3, (1, 16, 8, 8))
    b = Z.make_noise(3, (1, 16, 8, 8), dtype=torch.bfloat16)
    assert b.dtype == torch.bfloat16 and torch.equal(b, f.to(torch.bfloat16))


def test_make_noise_is_standard_normal():
    z = Z.make_noise(42, (1, 16, 64, 64))           # 65,536 표본: 평균 표준오차 0.004, std 0.0028
    assert abs(float(z.mean())) < 0.02 and abs(float(z.std()) - 1.0) < 0.02
    assert torch.isfinite(z).all()
    assert abs(float(((z - z.mean()) ** 3).mean())) < 0.05           # 왜도 ≈ 0


@pytest.mark.parametrize("bad", [dict(seed=-1, shape=(2,)), dict(seed=0, shape=(0, 2)), dict(seed=0, shape=(-1,)),
                                 dict(seed=0, shape=(2,), level=1 << 16), dict(seed=0, shape=(2,), k=-1),
                                 dict(seed=1 << 64, shape=(2,))])
def test_make_noise_rejects_bad_args(bad):
    with pytest.raises(ValueError):
        Z.make_noise(**bad)
