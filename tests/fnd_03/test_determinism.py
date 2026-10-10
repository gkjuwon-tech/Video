"""FND-03 솔버 경로 결정론 설정."""
import os

import torch

from buc.core import det


def test_enable_determinism_sets_flags_and_seeds_torch_and_numpy():
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(), os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    try:
        det.enable_determinism(seed=1234)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        a = torch.randn(8)
        det.enable_determinism(seed=1234)
        assert torch.equal(a, torch.randn(8))
        import numpy as np
        det.enable_determinism(seed=1234)
        n1 = np.random.rand(3)
        det.enable_determinism(seed=1234)
        assert (n1 == np.random.rand(3)).all()
        # seed=None이면 시드를 건드리지 않고 플래그만 설정
        det.enable_determinism()
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev[0], prev[1]
        torch.use_deterministic_algorithms(prev[2])
        if prev[3] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev[3]


def test_module_imports_without_cuda_and_exports_api():
    for name in ("philox4x32", "wnf", "wnf_tile", "cell_id", "derive_seed", "fixed_encode", "fixed_decode",
                 "canonical_order", "segment_reduce", "segment_reduce_int_torch", "hash_bytes", "hash_parts",
                 "hash_array", "merkle_leaf", "merkle_node", "merkle_root", "merkle_root_keyed", "merkle_proof",
                 "merkle_verify", "EMPTY_ROOT", "enable_determinism"):
        assert hasattr(det, name), name
