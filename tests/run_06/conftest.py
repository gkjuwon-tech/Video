"""RUN-06 공용 픽스처: 전역 상태(결정론 플래그, 정밀도 플래그, RNG)를 시험마다 격리한다."""
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _isolate_global_state():
    saved = (
        random.getstate(), np.random.get_state(), torch.get_rng_state(),
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(False)
    yield
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])
    torch.use_deterministic_algorithms(saved[3])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = saved[4]
    torch.backends.cuda.matmul.allow_tf32 = saved[5]
    torch.backends.cudnn.allow_tf32 = saved[6]
    torch.backends.cudnn.deterministic = saved[7]
    torch.backends.cudnn.benchmark = saved[8]
