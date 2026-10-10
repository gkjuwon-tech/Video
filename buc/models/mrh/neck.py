"""MDL-03 M3 MRH neck: PCA 탭 -> DPT식 재조립 + 융합 neck(FiLM t).

설계 메모
- 탭 3개(층 tap_layers[0..2])를 stride 4/8/16으로 재조립하고 coarse -> fine 융합 3블록으로 합친다.
- 융합 블록마다 ResidualConvUnit(RCU) rcu_per_fusion개(기본 4 -> 총 12). 첫 RCU는 skip 가지에,
  나머지는 합산 + FiLM 뒤에 둔다.
- FiLM: gamma = 1 + dgamma(e(t)), beta(e(t)); 선형층은 영 초기화라 초기 출력은 t와 무관하다.
- BatchNorm 금지(뷰 독립). 업샘플은 결정론 연산만 쓴다(최근접 + 1x1, 재조립은 1x1 + pixel_shuffle):
  RUN-06 deterministic 모드의 CUDA에서 비결정 역전파 연산(bilinear 등)을 피한다.
- import 부작용 없음, CUDA 불필요.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MRHSpecError", "MRHConfig", "ResidualConvUnit", "NeckOut", "MRHNeck"]


class MRHSpecError(ValueError):
    """설정·입력의 모양/값 위반."""


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


@dataclass(frozen=True)
class MRHConfig:
    tap_layers: tuple = (8, 15, 23)
    tap_dim: int = 128
    neck_ch: int = 256
    rcu_per_fusion: int = 4
    head_ch: int = 128
    sem_dim: int = 768
    n_sigma_bands: int = 4
    t_embed_dim: int = 128

    def __post_init__(self):
        try:
            layers = tuple(self.tap_layers)
        except TypeError:
            raise MRHSpecError("tap_layers: 정수 3개의 시퀀스여야 한다") from None
        if len(layers) != 3 or any(not _is_int(l) or int(l) < 0 for l in layers):
            raise MRHSpecError(f"tap_layers: 0 이상 정수 3개여야 한다: {self.tap_layers!r}")
        layers = tuple(int(l) for l in layers)
        if not (layers[0] < layers[1] < layers[2]):
            raise MRHSpecError(f"tap_layers: 엄격 오름차순이어야 한다: {layers!r}")
        object.__setattr__(self, "tap_layers", layers)
        for name in ("tap_dim", "neck_ch", "rcu_per_fusion", "head_ch", "sem_dim", "n_sigma_bands",
                     "t_embed_dim"):
            v = getattr(self, name)
            if not _is_int(v) or int(v) < 1:
                raise MRHSpecError(f"{name}: 양의 정수여야 한다: {v!r}")
            object.__setattr__(self, name, int(v))

    def to_dict(self) -> dict:
        return {"tap_layers": list(self.tap_layers), "tap_dim": self.tap_dim, "neck_ch": self.neck_ch,
                "rcu_per_fusion": self.rcu_per_fusion, "head_ch": self.head_ch, "sem_dim": self.sem_dim,
                "n_sigma_bands": self.n_sigma_bands, "t_embed_dim": self.t_embed_dim}


class ResidualConvUnit(nn.Module):
    """x + conv2(relu(conv1(relu(x)))). 3x3 conv 정확히 2개."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(x))
        y = self.conv2(F.relu(y))
        return x + y


class NeckOut(NamedTuple):
    feat4: torch.Tensor     # [B,C,4gh,4gw]
    feat16: torch.Tensor    # [B,C,gh,gw]  (MDL-04 전역 xfmr 입력)


def _upsample2(x: torch.Tensor) -> torch.Tensor:
    """최근접 2배(결정론 역전파)."""
    b, c, h, w = x.shape
    return x[:, :, :, None, :, None].expand(b, c, h, 2, w, 2).reshape(b, c, 2 * h, 2 * w)


class _Reassemble(nn.Module):
    """tap_dim -> C 1x1, 이어서 stride 4/8은 1x1 + pixel_shuffle(= k=s 전치 합성곱), 16은 항등."""

    def __init__(self, tap_dim: int, ch: int, factor: int):
        super().__init__()
        self.proj = nn.Conv2d(tap_dim, ch, 1)
        self.factor = factor
        if factor > 1:
            self.expand = nn.Conv2d(ch, ch * factor * factor, 1)
            self.shuffle = nn.PixelShuffle(factor)
        else:
            self.expand = None
            self.shuffle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.expand is not None:
            x = self.shuffle(self.expand(x))
        return x


class _FusionBlock(nn.Module):
    def __init__(self, ch: int, n_rcu: int, t_dim: int, has_prev: bool):
        super().__init__()
        self.skip_rcu = ResidualConvUnit(ch)
        self.post = nn.ModuleList(ResidualConvUnit(ch) for _ in range(n_rcu - 1))
        self.up = nn.Conv2d(ch, ch, 1) if has_prev else None   # 이전(coarse) 블록 출력 -> 이 해상도
        self.film = nn.Linear(t_dim, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, skip: torch.Tensor, prev, e: torch.Tensor) -> torch.Tensor:
        x = self.skip_rcu(skip)
        if prev is not None:
            x = x + self.up(_upsample2(prev))
        dg, beta = self.film(e).chunk(2, dim=-1)
        x = x * (1.0 + dg)[:, :, None, None].to(x.dtype) + beta[:, :, None, None].to(x.dtype)
        for r in self.post:
            x = r(x)
        return x


class MRHNeck(nn.Module):
    def __init__(self, cfg: MRHConfig):
        super().__init__()
        if not isinstance(cfg, MRHConfig):
            raise MRHSpecError("cfg는 MRHConfig여야 한다")
        self.cfg = cfg
        c = cfg.neck_ch
        self.reassemble = nn.ModuleList(_Reassemble(cfg.tap_dim, c, f) for f in (4, 2, 1))   # stride 4, 8, 16
        self.t_mlp = nn.Sequential(nn.Linear(cfg.t_embed_dim, cfg.t_embed_dim), nn.SiLU(),
                                   nn.Linear(cfg.t_embed_dim, cfg.t_embed_dim))
        # 융합 순서: coarse(stride 16) -> 8 -> 4
        self.fusion = nn.ModuleList(_FusionBlock(c, cfg.rcu_per_fusion, cfg.t_embed_dim, has_prev=(i > 0))
                                    for i in range(3))

    # ---- 입력 검증 ----
    def _check_taps(self, taps) -> tuple:
        cfg = self.cfg
        if not isinstance(taps, Mapping):
            raise MRHSpecError("taps는 {층: 텐서} 매핑이어야 한다")
        if set(taps) != set(cfg.tap_layers):
            raise MRHSpecError(f"taps 키 {sorted(taps, key=str)} != tap_layers {list(cfg.tap_layers)}")
        xs = [taps[l] for l in cfg.tap_layers]
        for l, x in zip(cfg.tap_layers, xs):
            if not isinstance(x, torch.Tensor):
                raise MRHSpecError(f"taps[{l}]: 텐서여야 한다")
            if not x.is_floating_point():
                raise TypeError(f"taps[{l}]: 부동소수 텐서여야 한다(int8이면 TapSet.features(basis)를 먼저): {x.dtype}")
        for l, x in zip(cfg.tap_layers, xs):
            if x.ndim != 4:
                raise MRHSpecError(f"taps[{l}]: [B,k,gh,gw] 4차원이어야 한다: {tuple(x.shape)}")
            if x.shape[1] != cfg.tap_dim:
                raise MRHSpecError(f"taps[{l}]: 채널 {x.shape[1]} != tap_dim {cfg.tap_dim}")
            if x.shape != xs[0].shape:
                raise MRHSpecError(f"taps[{l}]: 층 간 모양 불일치 {tuple(x.shape)} != {tuple(xs[0].shape)}")
        if xs[0].shape[0] < 1 or xs[0].shape[2] < 1 or xs[0].shape[3] < 1:
            raise MRHSpecError("빈 입력")
        return tuple(xs)

    @staticmethod
    def _check_t(t, b: int, device) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            if t.dtype == torch.bool or t.is_complex():
                raise MRHSpecError("t: 실수 텐서여야 한다")
            tt = t.detach().to(device=device, dtype=torch.float32)
        elif isinstance(t, numbers.Real) and not isinstance(t, bool):
            tt = torch.tensor(float(t), dtype=torch.float32, device=device)
        else:
            raise MRHSpecError(f"t: float 또는 Tensor[B]여야 한다: {type(t).__name__}")
        if tt.numel() == 1:
            tt = tt.reshape(1).expand(b)
        elif tt.ndim != 1 or tt.shape[0] != b:
            raise MRHSpecError(f"t: 길이 {tuple(tt.shape)}가 배치 {b}와 다르다")
        if not bool(torch.isfinite(tt).all()) or not bool(((tt > 0) & (tt <= 1)).all()):
            raise MRHSpecError("t: 유한한 (0,1] 값이어야 한다")
        return tt

    def _t_embed(self, tt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        d = self.cfg.t_embed_dim
        half = d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tt.device, dtype=torch.float32)
                          / max(half, 1))
        ang = (tt * 1000.0)[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < d:
            emb = F.pad(emb, (0, d - emb.shape[-1]))
        return self.t_mlp(emb.to(ref.dtype))

    def forward(self, taps: Mapping, t) -> NeckOut:
        xs = self._check_taps(taps)
        b = xs[0].shape[0]
        tt = self._check_t(t, b, xs[0].device)
        e = self._t_embed(tt, self.t_mlp[0].weight)
        r4, r8, r16 = (rs(x) for rs, x in zip(self.reassemble, xs))    # tap_layers[0]->4, [1]->8, [2]->16
        f16 = self.fusion[0](r16, None, e)
        f8 = self.fusion[1](r8, f16, e)
        f4 = self.fusion[2](r4, f8, e)
        return NeckOut(feat4=f4, feat16=f16)
