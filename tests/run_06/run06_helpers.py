"""RUN-06 시험 보조: 인메모리 가짜 허브(RUN-03 HubClient 이음매), 작은 회귀 문제, Trainer 조립기."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from buc.rt import hf_sync as H
from buc.train.amp import AmpConfig
from buc.train.loop import LoopConfig, Trainer

N = 48  # 샘플 수
RUN_ID = "r06"


class FakeHub:
    """HubClient 프로토콜(list_files/upload_file/download_file/is_private)의 인메모리 구현."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []

    def list_files(self, prefix):
        self.calls.append(("list", prefix))
        return sorted(self.files)

    def upload_file(self, local, remote):
        self.calls.append(("upload", remote))
        self.files[remote] = Path(local).read_bytes()

    def download_file(self, remote, local):
        self.calls.append(("download", remote))
        if remote not in self.files:
            raise H.HubNotFoundError(remote)
        Path(local).write_bytes(self.files[remote])

    def is_private(self):
        return True

    def n(self, op):
        return sum(1 for c in self.calls if c[0] == op)


def make_sync(client):
    return H.HubSync(client, sleep=lambda s: None)


def make_data(n=N, d=4, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d, 1)).astype(np.float32)
    y = (x @ w + 0.1 * rng.standard_normal((n, 1))).astype(np.float32)
    return x, y


X, Y = make_data()


def batch_fn(idx):
    return {"x": torch.from_numpy(X[idx]), "y": torch.from_numpy(Y[idx])}


def make_model(seed=0, dropout=0.2, d=4, h=8):
    g = torch.get_rng_state()
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Dropout(dropout), torch.nn.Linear(h, 1))
    torch.set_rng_state(g)
    return m


def mse(model, batch):
    return F.mse_loss(model(batch["x"]), batch["y"])


def make_trainer(ckpt_dir, *, max_steps, model_seed=0, dropout=0.2, sched=True, hub=None, clock=None,
                 compute_loss=None, opt="adamw", model=None, ema=0.9, lr=1e-2, device="cpu", on_alarm=None, log_fn=None,
                 data_fn=None, **cfg_kw):
    model = model if model is not None else make_model(model_seed, dropout)
    if opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr if lr != 1e-2 else 0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8) if sched else None
    base = dict(run_id=RUN_ID, ckpt_dir=ckpt_dir, num_samples=N, batch_size=8, max_steps=max_steps, seed=7,
                ckpt_every_steps=4, ckpt_every_seconds=1e9, upload_every_seconds=1e9, log_every=1,
                max_abs_every=1000, ema_decay=ema, amp=AmpConfig(dtype="fp32", device_type="cpu"))
    base.update(cfg_kw)
    cfg = LoopConfig(**base)
    logs: list[dict] = []

    def sink(d):
        logs.append(d)
        if log_fn is not None:
            log_fn(d)

    kw = {"clock": clock} if clock is not None else {}
    tr = Trainer(model, optimizer, compute_loss or mse, data_fn or batch_fn, cfg, scheduler=scheduler, hub=hub,
                 log=sink, device=device, on_alarm=on_alarm, **kw)
    return tr, logs


def train_logs(logs):
    return [d for d in logs if d.get("event") == "train"]


def losses(logs):
    return {d["step"]: d["loss"] for d in train_logs(logs)}


def assert_sd_equal(a, b):
    assert list(a) == list(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k
