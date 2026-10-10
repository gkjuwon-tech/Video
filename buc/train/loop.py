"""RUN-06 학습 루프: LoopConfig·ModelEMA·Trainer. 14종 모델(MDL-*)이 공유하는 바닥.

재개 정확성이 핵심이다: 멈췄다 이어간 학습은 끝까지 한 번에 돈 학습과 비트 동일해야 한다(CPU fp32).
그래서 (1) 데이터 순서는 무상태 PRP(``sampler.py``), (2) 체크포인트는 파라미터·EMA·옵티마이저·스케줄러·
스케일러·RNG를 전부 담고, (3) 체크포인트는 스텝 경계에서만 만든다. CUDA 없이 import 가능하며
import 부작용이 없다. 로거 이름은 ``"buc.train"``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from buc.core.det import enable_determinism
from buc.rt.hf_sync import validate_run_id
from buc.train.amp import Alarm, Amp, AmpConfig, MaxAbsMonitor
from buc.train.ckpt import CheckpointManager, apply_state, config_hash, make_state
from buc.train.sampler import StatelessSampler

__all__ = ["LoopConfig", "ModelEMA", "to_device", "Trainer", "FitResult", "TrainingDiverged"]

_LOG = logging.getLogger("buc.train")


class TrainingDiverged(RuntimeError):
    """연속으로 건너뛴 스텝이 한도를 넘었다(손실·기울기가 계속 비유한)."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- 설정
@dataclass(frozen=True)
class LoopConfig:
    run_id: str
    ckpt_dir: Path
    num_samples: int
    batch_size: int
    max_steps: int
    seed: int = 0
    grad_accum: int = 1
    grad_clip: Optional[float] = 1.0
    amp: AmpConfig = field(default_factory=AmpConfig)
    ema_decay: Optional[float] = None
    log_every: int = 100
    ckpt_every_steps: int = 1000
    ckpt_every_seconds: float = 900.0
    upload_every_seconds: float = 1800.0
    keep_last: int = 3
    max_abs_every: int = 100
    max_abs_threshold: float = 2e4
    max_consecutive_skips: int = 64
    time_budget_s: Optional[float] = None
    deterministic: bool = True
    git_commit: Optional[str] = None
    hparams: Mapping = field(default_factory=dict)

    def __post_init__(self):
        validate_run_id(self.run_id)
        object.__setattr__(self, "ckpt_dir", Path(self.ckpt_dir))
        for name, lo in (("num_samples", 1), ("batch_size", 1), ("max_steps", 0), ("grad_accum", 1),
                         ("log_every", 1), ("ckpt_every_steps", 1), ("keep_last", 1), ("max_abs_every", 1),
                         ("max_consecutive_skips", 1)):
            v = getattr(self, name)
            if not _is_int(v):
                raise TypeError(f"{name}: integer required")
            if v < lo:
                raise ValueError(f"{name} must be >= {lo}, got {v}")
        if not _is_int(self.seed):
            raise TypeError("seed: integer required")
        if not (0 <= self.seed < (1 << 64)):
            raise ValueError("seed must be in [0, 2^64)")
        if self.batch_size % self.grad_accum != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by grad_accum ({self.grad_accum})")
        if self.grad_clip is not None and not (_is_num(self.grad_clip) and self.grad_clip > 0):
            raise ValueError("grad_clip must be > 0 or None")
        if self.ema_decay is not None and not (_is_num(self.ema_decay) and 0 < self.ema_decay < 1):
            raise ValueError("ema_decay must be in (0, 1) or None")
        for name in ("ckpt_every_seconds", "upload_every_seconds", "max_abs_threshold"):
            v = getattr(self, name)
            if not (_is_num(v) and v > 0):
                raise ValueError(f"{name} must be > 0")
        if self.time_budget_s is not None and not (_is_num(self.time_budget_s) and self.time_budget_s > 0):
            raise ValueError("time_budget_s must be > 0 or None")
        if not isinstance(self.amp, AmpConfig):
            raise TypeError("amp must be an AmpConfig")
        if not isinstance(self.hparams, Mapping):
            raise TypeError("hparams must be a mapping")

    def config_hash(self) -> str:
        """설정 정체성. 일정 항목(max_steps·주기·keep_last·ckpt_dir·time_budget·임계값)은 넣지 않는다."""
        return config_hash({
            "run_id": self.run_id,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "grad_accum": self.grad_accum,
            "grad_clip": self.grad_clip,
            "amp_dtype": self.amp.dtype,
            "ema_decay": self.ema_decay,
            "hparams": dict(self.hparams),
        })


# --------------------------------------------------------------------------- 보조
def to_device(obj, device):
    """텐서·Mapping·list/tuple 을 재귀적으로 ``device`` 로 옮긴다(그 밖의 값은 그대로)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_device(v, device) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


class ModelEMA:
    """fp32 그림자 가중치의 지수이동평균: ``s = d*s + (1-d)*p``."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not (_is_num(decay) and 0 < decay < 1):
            raise ValueError("decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {k: self._shadow_of(v) for k, v in model.state_dict().items()}

    @staticmethod
    def _shadow_of(v: torch.Tensor) -> torch.Tensor:
        v = v.detach()
        return v.to(dtype=torch.float32, copy=True) if v.is_floating_point() else v.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.is_floating_point():
                s.mul_(d).add_(v.detach().to(dtype=s.dtype, device=s.device), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, sd: Mapping) -> None:
        if set(sd) != set(self.shadow):
            raise ValueError("EMA state keys do not match the model")
        for k, v in sd.items():
            self.shadow[k].copy_(v)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)


# --------------------------------------------------------------------------- 결과
@dataclass(frozen=True)
class FitResult:
    step: int
    stop_reason: str                      # "max_steps" | "time_budget"
    skipped_steps: int
    alarms: tuple
    last_log: Optional[dict]
    ckpt_path: Optional[Path]
    resumed_from: Optional[int]


# --------------------------------------------------------------------------- 트레이너
class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer, compute_loss: Callable, batch_fn: Callable,
                 cfg: LoopConfig, *, scheduler=None, hub=None, device="cpu",
                 clock: Callable[[], float] = time.monotonic, log: Optional[Callable[[dict], None]] = None,
                 on_alarm: Optional[Callable[[Alarm], None]] = None):
        for name, p in model.named_parameters():
            if p.is_floating_point() and p.dtype != torch.float32:
                raise ValueError(f"master parameters must be fp32; {name} is {p.dtype}")
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_loss = compute_loss
        self.batch_fn = batch_fn
        self.hub = hub
        self._clock = clock
        self._log_fn = log
        self._on_alarm = on_alarm
        self.amp = Amp(dataclasses.replace(cfg.amp, device_type=self.device.type))
        self.ema = ModelEMA(self.model, cfg.ema_decay) if cfg.ema_decay is not None else None
        self.sampler = StatelessSampler(cfg.num_samples, cfg.batch_size, cfg.seed)
        self.step = 0

    # ---- 보조 ----------------------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        if self._log_fn is not None:
            self._log_fn(event)
        else:
            _LOG.info("%s", json.dumps(event, sort_keys=True, default=str))

    def _seed_everything(self) -> None:
        if self.cfg.deterministic:
            enable_determinism(self.cfg.seed)
            return
        seed = self.cfg.seed
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _state(self, cfg_hash: str) -> dict:
        return make_state(self.step, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                          ema=self.ema, amp=self.amp, config_hash=cfg_hash, git_commit=self.cfg.git_commit)

    def _train_step(self, t: int, want_log: bool):
        """스텝 하나: 누적 조각마다 forward·backward, 그리고 ``amp.step``. (StepResult, 손실, 지표) 반환."""
        cfg = self.cfg
        accum = cfg.grad_accum
        chunks = self.sampler.indices(t).reshape(accum, -1)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        metric_sums: dict[str, float] = {}
        for row in chunks:
            batch = to_device(self.batch_fn(np.ascontiguousarray(row)), self.device)
            with self.amp.autocast():
                out = self.compute_loss(self.model, batch)
            loss, metrics = out if isinstance(out, tuple) else (out, {})
            self.amp.scale(loss / accum if accum > 1 else loss).backward()
            if want_log:
                loss_sum += float(loss.detach().float().item())
                for k, v in metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        res = self.amp.step(self.optimizer, max_grad_norm=cfg.grad_clip)
        return res, loss_sum / accum, {k: v / accum for k, v in metric_sums.items()}

    # ---- 학습 ----------------------------------------------------------------------------------
    def fit(self) -> FitResult:
        cfg = self.cfg
        t_start = self._clock()
        cfg_hash = cfg.config_hash()
        mgr = CheckpointManager(
            cfg.ckpt_dir, cfg.run_id, keep_last=cfg.keep_last, every_steps=cfg.ckpt_every_steps,
            every_seconds=cfg.ckpt_every_seconds, upload_every_seconds=cfg.upload_every_seconds,
            hub=self.hub, clock=self._clock)
        try:
            return self._fit(mgr, cfg_hash, t_start)
        finally:
            mgr.close()

    def _fit(self, mgr: CheckpointManager, cfg_hash: str, t_start: float) -> FitResult:
        cfg = self.cfg
        loaded = mgr.resume(expect_config_hash=cfg_hash)
        if loaded is None:
            self._seed_everything()
            resumed_from = None
        else:
            if cfg.deterministic:
                enable_determinism(None)
            apply_state(loaded.state, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                        ema=self.ema, amp=self.amp)
            self.step = loaded.step
            resumed_from = loaded.step

        alarms: list = []
        last_log: Optional[dict] = None
        stop_reason = "max_steps"

        if self.step < cfg.max_steps:
            self.model.train()
            monitor = MaxAbsMonitor(self.model, every=cfg.max_abs_every, threshold=cfg.max_abs_threshold)
            try:
                while self.step < cfg.max_steps:
                    t = self.step
                    monitor.begin(t)
                    want_log = (t + 1) % cfg.log_every == 0
                    res, loss, metrics = self._train_step(t, want_log)
                    if res.stepped:
                        if self.scheduler is not None:
                            self.scheduler.step()
                        if self.ema is not None:
                            self.ema.update(self.model)
                    self.step = t + 1  # 건너뛴 스텝도 전진한다

                    report = monitor.end()
                    if report is not None:
                        for alarm in report.alarms:
                            _LOG.warning("max-abs alarm: layer=%r step=%d value=%g > threshold=%g",
                                         alarm.layer, alarm.step, alarm.value, alarm.threshold)
                            alarms.append(alarm)
                            if self._on_alarm is not None:
                                self._on_alarm(alarm)
                        worst = report.worst
                        self._emit({"event": "max_abs", "step": report.step,
                                    "worst_layer": worst[0] if worst else None,
                                    "worst": worst[1] if worst else None,
                                    "n_alarms": len(report.alarms)})

                    if want_log:
                        last_log = {
                            "event": "train", "step": self.step, "loss": loss,
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                            "grad_norm": res.grad_norm, "loss_scale": self.amp.scale_value,
                            "skipped_steps": self.amp.skipped_steps, "metrics": metrics,
                        }
                        self._emit(last_log)

                    if self.amp.consecutive_skips > cfg.max_consecutive_skips:
                        raise TrainingDiverged(
                            f"{self.amp.consecutive_skips} consecutive non-finite steps "
                            f"(limit {cfg.max_consecutive_skips}) at step {self.step}")

                    if self.step >= cfg.max_steps:
                        break
                    if mgr.should_save(self.step):
                        mgr.save(self.step, self._state(cfg_hash))
                    if cfg.time_budget_s is not None and self._clock() - t_start >= cfg.time_budget_s:
                        stop_reason = "time_budget"
                        break
            finally:
                monitor.close()

        ckpt_path: Optional[Path] = None
        if self.step > 0:
            path = mgr.path_for(self.step)
            if path.is_file():
                mgr.maybe_upload(self.step, force=True)
            else:
                path = mgr.save(self.step, self._state(cfg_hash), final=True)
            ckpt_path = path
        return FitResult(step=self.step, stop_reason=stop_reason, skipped_steps=self.amp.skipped_steps,
                         alarms=tuple(alarms), last_log=last_log, ckpt_path=ckpt_path,
                         resumed_from=resumed_from)

