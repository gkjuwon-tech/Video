"""RUN-06 혼합정밀 학습 바닥: fp32 마스터 + autocast(fp16/bf16), 직접 구현한 손실 스케일러, 층별 max-abs 경보.

torch.cuda.amp.GradScaler 를 쓰지 않는 이유: (1) CPU에서도 같은 동작이어야 하고, (2) 모든 dtype에서
비유한 노름이면 스텝을 건너뛰어야 하며, (3) 상태를 파이썬 기본형으로만 저장해야 한다
(``torch.load(weights_only=True)``). CUDA 없이 import 가능하며 import 부작용이 없다.
"""
from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

__all__ = [
    "AmpConfig", "Amp", "StepResult", "set_precision_flags",
    "MaxAbsMonitor", "MaxAbsReport", "Alarm",
]

_DTYPES = ("fp16", "bf16", "fp32")
_DEVICE_TYPES = ("cuda", "cpu")
_F32_MAX = float(torch.finfo(torch.float32).max)


@dataclass(frozen=True)
class AmpConfig:
    dtype: str = "fp16"
    device_type: str = "cuda"
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    def __post_init__(self):
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
        if self.device_type not in _DEVICE_TYPES:
            raise ValueError(f"device_type must be one of {_DEVICE_TYPES}, got {self.device_type!r}")
        if not (isinstance(self.init_scale, (int, float)) and math.isfinite(self.init_scale) and self.init_scale > 0):
            raise ValueError("init_scale must be a positive finite number")
        if not (isinstance(self.growth_factor, (int, float)) and math.isfinite(self.growth_factor)
                and self.growth_factor > 1):
            raise ValueError("growth_factor must be > 1")
        if not (isinstance(self.backoff_factor, (int, float)) and 0 < self.backoff_factor < 1):
            raise ValueError("backoff_factor must be in (0, 1)")
        if isinstance(self.growth_interval, bool) or not isinstance(self.growth_interval, int) \
                or self.growth_interval < 1:
            raise ValueError("growth_interval must be an int >= 1")


def set_precision_flags() -> None:
    """감소 정밀도 경로를 끈다: fp16 감소 정밀도 누적, TF32(matmul·cudnn)."""
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class StepResult(NamedTuple):
    stepped: bool
    grad_norm: float   # 언스케일 후, 클립 전 노름
    scale: float       # 이 스텝의 스케일 갱신 뒤 값


class Amp:
    def __init__(self, cfg: AmpConfig, *, apply_backend_flags: bool = True):
        if not isinstance(cfg, AmpConfig):
            raise TypeError("cfg must be an AmpConfig")
        self.cfg = cfg
        self.scaling = cfg.dtype == "fp16"
        self._scale = float(cfg.init_scale) if self.scaling else 1.0
        self._growth_tracker = 0
        self.skipped_steps = 0
        self.consecutive_skips = 0
        if apply_backend_flags:
            set_precision_flags()

    @property
    def scale_value(self) -> float:
        return self._scale if self.scaling else 1.0

    # ---- 컨텍스트 ------------------------------------------------------------------------------
    def autocast(self):
        if self.cfg.dtype == "fp32":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.cfg.dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.cfg.device_type, dtype=dtype)

    def fp32_region(self):
        """autocast를 끈다(LayerNorm·softmax·손실·FFT용). autocast 바깥에서 불러도 안전하다."""
        return torch.autocast(device_type=self.cfg.device_type, enabled=False)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self._scale if self.scaling else loss

    # ---- 스텝 ----------------------------------------------------------------------------------
    @staticmethod
    def _grads(optimizer) -> list:
        seen, grads = set(), []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and id(p) not in seen:
                    seen.add(id(p))
                    grads.append(p.grad)
        return grads

    @staticmethod
    def _total_norm(grads: list) -> float:
        if not grads:
            return 0.0
        with torch.no_grad():
            norms = [torch.linalg.vector_norm(g.detach(), 2.0, dtype=torch.float32) for g in grads]
            dev = norms[0].device
            total = torch.linalg.vector_norm(torch.stack([n.to(dev) for n in norms]), 2.0)
            return float(total.item())

    def _update_scale(self, ok: bool) -> None:
        if not self.scaling:
            return
        if ok:
            self._growth_tracker += 1
            if self._growth_tracker >= self.cfg.growth_interval:
                grown = self._scale * self.cfg.growth_factor
                if math.isfinite(grown) and grown <= _F32_MAX:
                    self._scale = grown
                self._growth_tracker = 0
        else:
            self._scale *= self.cfg.backoff_factor
            self._growth_tracker = 0

    def step(self, optimizer, *, max_grad_norm: Optional[float] = None) -> StepResult:
        """backward 이후의 전부: 언스케일 -> 노름 -> 클립 -> (유한하면) optimizer.step -> 스케일 갱신."""
        grads = self._grads(optimizer)
        with torch.no_grad():
            if self.scaling:
                inv = 1.0 / self._scale
                for g in grads:
                    g.mul_(inv)
            norm = self._total_norm(grads)
            finite = math.isfinite(norm)
            if finite and max_grad_norm is not None:
                coef = max_grad_norm / (norm + 1e-6)
                if coef < 1.0:
                    for g in grads:
                        g.mul_(coef)
        if finite:
            optimizer.step()
            self.consecutive_skips = 0
        else:
            self.skipped_steps += 1
            self.consecutive_skips += 1
        self._update_scale(finite)
        return StepResult(finite, norm, self.scale_value)

    # ---- 상태 ----------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "v": 1,
            "dtype": self.cfg.dtype,
            "scale": float(self.scale_value),
            "growth_tracker": int(self._growth_tracker),
            "skipped_steps": int(self.skipped_steps),
        }

    def load_state_dict(self, sd: Mapping) -> None:
        try:
            if sd["v"] != 1:
                raise ValueError(f"unsupported amp state version: {sd['v']!r}")
            if sd["dtype"] != self.cfg.dtype:
                raise ValueError(f"amp dtype mismatch: state {sd['dtype']!r} vs config {self.cfg.dtype!r}")
            scale, tracker, skipped = float(sd["scale"]), int(sd["growth_tracker"]), int(sd["skipped_steps"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed amp state: {e!r}") from None
        self._scale = scale if self.scaling else 1.0
        self._growth_tracker = tracker
        self.skipped_steps = skipped
        self.consecutive_skips = 0


# --------------------------------------------------------------------------- max-abs 경보
class Alarm(NamedTuple):
    step: int
    layer: str
    value: float
    threshold: float


@dataclass(frozen=True)
class MaxAbsReport:
    step: int
    values: dict
    alarms: tuple

    @property
    def worst(self) -> Optional[tuple]:
        """비유한 층이 있으면 그중 첫째, 아니면 최댓값 층. 층이 없으면 None."""
        if not self.values:
            return None
        for name, v in self.values.items():
            if not math.isfinite(v):
                return (name, v)
        name = max(self.values, key=lambda k: self.values[k])
        return (name, self.values[name])


def _collect_max(obj, acc: list) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.numel() > 0:
            acc.append(obj.detach().abs().amax().float())
    elif isinstance(obj, Mapping):
        for v in obj.values():
            _collect_max(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_max(v, acc)


class MaxAbsMonitor:
    """무장한 구간(``begin`` ~ ``end``)에서 각 모듈 출력의 |x| 최댓값을 모은다. 출력·기울기는 건드리지 않는다."""

    def __init__(self, model: torch.nn.Module, *, every: int = 100, threshold: float = 2e4,
                 include: Optional[Callable[[str, torch.nn.Module], bool]] = None):
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("every must be an int >= 1")
        if not (isinstance(threshold, (int, float)) and threshold > 0):
            raise ValueError("threshold must be > 0")
        self.every = every
        self.threshold = float(threshold)
        self._armed = False
        self._step = 0
        self._acc: dict[str, torch.Tensor] = {}
        self._handles: list = []
        for name, mod in model.named_modules():
            if include is None:
                if name == "" or next(mod.children(), None) is not None:
                    continue
            elif not include(name, mod):
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, args, output):
            if not self._armed:
                return None
            found: list = []
            _collect_max(output, found)
            if found:
                m = found[0]
                for t in found[1:]:
                    m = torch.maximum(m, t.to(m.device))  # NaN은 전파된다
                prev = self._acc.get(name)
                self._acc[name] = m if prev is None else torch.maximum(prev, m.to(prev.device))
            return None
        return hook

    def begin(self, step: int) -> bool:
        self._acc = {}
        self._step = int(step)
        self._armed = self._step % self.every == 0
        return self._armed

    def end(self) -> Optional[MaxAbsReport]:
        if not self._armed:
            return None
        self._armed = False
        values = {name: float(t.item()) for name, t in self._acc.items()}
        self._acc = {}
        alarms = tuple(
            Alarm(self._step, name, v, self.threshold)
            for name, v in values.items()
            if not math.isfinite(v) or v > self.threshold
        )
        return MaxAbsReport(self._step, values, alarms)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._armed = False
        self._acc = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
