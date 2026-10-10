"""MDL-03 M3 MRH 조밀 헤드 8종과 MRH 합성 모듈.

헤드(H/4 해상도): log_depth 1, normal 3, albedo 3, rough_metal 2, shading 1, defocus 1, sem sem_dim,
sigma_hat n_sigma_bands. 활성 정의는 계약서 MDL-03 참조. 헤드 이름·채널·활성을 바꾸면 계약서를 올린다.
"""
from __future__ import annotations

import numbers

import torch
import torch.nn.functional as F
from torch import nn

from buc.models.mrh.neck import MRHConfig, MRHNeck, MRHSpecError, NeckOut

__all__ = ["SIGMA_MIN", "HEAD_NAMES", "head_channels", "MRHHeads", "MRH"]

SIGMA_MIN = 1e-3
HEAD_NAMES = ("log_depth", "normal", "albedo", "rough_metal", "shading", "defocus", "sem", "sigma_hat")
_EPS = 1e-6


def head_channels(cfg: MRHConfig) -> dict:
    return {"log_depth": 1, "normal": 3, "albedo": 3, "rough_metal": 2, "shading": 1, "defocus": 1,
            "sem": cfg.sem_dim, "sigma_hat": cfg.n_sigma_bands}


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(_EPS)


class MRHHeads(nn.Module):
    def __init__(self, cfg: MRHConfig):
        super().__init__()
        self.cfg = cfg
        hc = cfg.head_ch
        self.trunk = nn.Conv2d(cfg.neck_ch, hc, 3, padding=1)
        self.branches = nn.ModuleDict({
            name: nn.Sequential(nn.Conv2d(hc, hc, 3, padding=1), nn.ReLU(), nn.Conv2d(hc, c, 1))
            for name, c in head_channels(cfg).items()})

    def forward(self, feat4: torch.Tensor) -> dict:
        if feat4.ndim != 4 or feat4.shape[1] != self.cfg.neck_ch:
            raise MRHSpecError(f"feat4: [B,{self.cfg.neck_ch},h,w]여야 한다: {tuple(feat4.shape)}")
        z = F.relu(self.trunk(feat4))
        raw = {name: self.branches[name](z).float() for name in HEAD_NAMES}
        return {
            "log_depth": raw["log_depth"],
            "normal": _unit(raw["normal"]),
            "albedo": torch.sigmoid(raw["albedo"]),
            "rough_metal": torch.sigmoid(raw["rough_metal"]),
            "shading": F.softplus(raw["shading"]),
            "defocus": F.softplus(raw["defocus"]),
            "sem": _unit(raw["sem"]),
            "sigma_hat": F.softplus(raw["sigma_hat"]) + SIGMA_MIN,
        }


class MRH(nn.Module):
    def __init__(self, cfg: MRHConfig = None, *, seed=None):
        super().__init__()
        cfg = MRHConfig() if cfg is None else cfg
        if not isinstance(cfg, MRHConfig):
            raise MRHSpecError("cfg는 MRHConfig여야 한다")
        if seed is not None and (not isinstance(seed, numbers.Integral) or isinstance(seed, bool)
                                 or not (0 <= int(seed) < (1 << 64))):
            raise MRHSpecError(f"seed: [0, 2^64) 정수 또는 None: {seed!r}")
        self.cfg = cfg
        if seed is None:
            self.neck = MRHNeck(cfg)
            self.heads = MRHHeads(cfg)
        else:
            # CPU 생성기만 잠시 시드하고 복원한다: 같은 seed -> 같은 가중치, 전역 RNG 불변.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(int(seed))
                self.neck = MRHNeck(cfg)
                self.heads = MRHHeads(cfg)

    def features(self, taps, t) -> NeckOut:
        return self.neck(taps, t)

    def forward(self, taps, t) -> dict:
        return self.heads(self.neck(taps, t).feat4)
