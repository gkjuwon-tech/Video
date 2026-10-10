"""TCH-01 공용 픽스처: 전역 결정론 상태를 테스트 뒤에 되돌린다."""
import os
import random

import numpy as np
import pytest
import torch


def _snapshot():
    return {
        "det_algo": torch.are_deterministic_algorithms_enabled(),
        "det_warn": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_det": torch.backends.cudnn.deterministic,
        "cudnn_bench": torch.backends.cudnn.benchmark,
        "tf32_mm": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "bf16_red": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_red": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_rng": torch.get_rng_state(),
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
    }


def _restore(s):
    torch.use_deterministic_algorithms(s["det_algo"], warn_only=s["det_warn"])
    torch.backends.cudnn.deterministic = s["cudnn_det"]
    torch.backends.cudnn.benchmark = s["cudnn_bench"]
    torch.backends.cuda.matmul.allow_tf32 = s["tf32_mm"]
    torch.backends.cudnn.allow_tf32 = s["tf32_cudnn"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = s["bf16_red"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = s["fp16_red"]
    if s["cublas"] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = s["cublas"]
    torch.set_rng_state(s["torch_rng"])
    np.random.set_state(s["np_rng"])
    random.setstate(s["py_rng"])


@pytest.fixture(autouse=True)
def restore_global_state():
    s = _snapshot()
    try:
        yield
    finally:
        _restore(s)
