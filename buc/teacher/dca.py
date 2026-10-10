"""TCH-06 Teacher-DCA · DR 판정: 조건 2개 x sigma 2개의 denoising 오차 차, DR_rel 합격선.

계약서: ops/harness/contracts/TCH-06.md. 공개 API는 계약서와 1:1이다.
- flow matching: z = (1-s) x0 + s eps, v = eps - x0, x0_hat = z - s v_hat. 그러면 v_hat - v = (x0 - x0_hat)/s.
- 잔차 r = mean_{C,h,w}(x0_hat - x0)^2, 속도 오차 e = r/s^2. DR = sigma 점들에 대한 r 의 평균.
- DCA = mean_s [e(x; c_b) - e(x; c_a)], 양수 = x 가 c_a 에 더 맞는다. 같은 z(공통 난수)를 두 조건에 쓴다.
- 변환기는 TCH-02(TeacherSampler.predict_x0)로만 부른다. import 는 torch·numpy·TCH-01/02 뿐이며 부작용이 없다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from buc.teacher.sample import SigmaError, TeacherSampler, _as_sigma, add_noise
from buc.teacher.zimage import TeacherSpecError

__all__ = [
    "DCA_SIGMAS", "DR_SIGMAS", "DR_REL_MAX", "DR_REL_MAX_OS", "DR_FLOOR", "DCAError", "residual", "velocity_error",
    "DRResult", "DCAResult", "teacher_dr", "teacher_dca", "delta_photo", "dr_rel", "DRVerdict", "dr_verdict",
    "dr_auc", "rank_topk",
]

DCA_SIGMAS = (0.75, 0.5)
DR_SIGMAS = (0.5,)
DR_REL_MAX = 1.2
DR_REL_MAX_OS = 1.1
DR_FLOOR = 1e-12


class DCAError(TeacherSpecError):
    """DCA·DR 입력이 잘못됐다(ValueError 계열)."""


# --------------------------------------------------------------------------- 수식

def _check_pair(x0_hat, x0):
    if not isinstance(x0_hat, torch.Tensor) or not isinstance(x0, torch.Tensor):
        raise DCAError("x0_hat, x0 는 텐서여야 한다")
    if x0_hat.ndim != 4 or x0.ndim != 4:
        raise DCAError(f"[B,C,h,w] 4차원이어야 한다: {tuple(x0_hat.shape)}, {tuple(x0.shape)}")
    if tuple(x0_hat.shape) != tuple(x0.shape):
        raise DCAError(f"모양 불일치: {tuple(x0_hat.shape)} != {tuple(x0.shape)}")
    a = x0_hat.detach().to(torch.float32)
    b = x0.detach().to(torch.float32)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise DCAError("x0_hat 또는 x0 에 NaN/Inf 가 있다")
    return a, b


def residual(x0_hat: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """뷰별 r = mean_{C,h,w}(x0_hat - x0)^2, float32[B]."""
    a, b = _check_pair(x0_hat, x0)
    return ((a - b) ** 2).mean(dim=(1, 2, 3))


def velocity_error(x0_hat: torch.Tensor, x0: torch.Tensor, sigma) -> torch.Tensor:
    """e = r / sigma^2 = mean(v_hat - v)^2, float32[B]. sigma 가 (0, 1] 밖이면 SigmaError."""
    s = _as_sigma(sigma)
    return residual(x0_hat, x0) / (s * s)


# --------------------------------------------------------------------------- 결과 형식

@dataclass(frozen=True)
class DRResult:
    dr: torch.Tensor
    per_sigma: torch.Tensor
    sigmas: tuple
    calls: int
    table_key: str


@dataclass(frozen=True)
class DCAResult:
    dca: torch.Tensor
    err_a: torch.Tensor
    err_b: torch.Tensor
    sigmas: tuple
    calls: int
    table_key: str


# --------------------------------------------------------------------------- 공통 준비

def _prep_sigmas(sampler: TeacherSampler, sigmas) -> tuple:
    """sigma 열 -> 표의 점 튜플(주어진 순서 그대로). 표 밖 SigmaError, 비었거나 중복이면 DCAError."""
    if isinstance(sigmas, (str, bytes)) or not hasattr(sigmas, "__iter__"):
        raise DCAError(f"sigmas 는 실수 시퀀스여야 한다: {sigmas!r}")
    items = list(sigmas)
    if not items:
        raise DCAError("sigmas 가 비었다")
    idx = [sampler.table.index_of(s) for s in items]
    if len(set(idx)) != len(idx):
        raise DCAError(f"sigmas 에 같은 표 점이 중복된다: {items}")
    return tuple(float(sampler.table.points[i]) for i in idx)


def _make_z(sampler: TeacherSampler, x: torch.Tensor, sig: tuple, seed_list: list, hw: tuple, level) -> list:
    zs = []
    h, w = hw
    for j, s in enumerate(sig):
        eps = sampler._noise(seed_list, h, w, level, j).to(x.device)
        zs.append(add_noise(x, s, eps))
    return zs


# --------------------------------------------------------------------------- DR / DCA

def teacher_dr(sampler: TeacherSampler, x0, cond, *, seeds, sigmas=DR_SIGMAS, level=0) -> DRResult:
    with torch.no_grad():
        sig = _prep_sigmas(sampler, sigmas)
        sampler._check_latent(x0, "x0")
        b, _, h, w = x0.shape
        caps = [c.detach() for c in sampler._resolve_cond(cond, b)]
        seed_list = sampler._resolve_seeds(seeds, b)
        x = sampler._to_state(x0).detach()
        zs = _make_z(sampler, x, sig, seed_list, (h, w), level)
        cols = []
        for s, z in zip(sig, zs):
            x0_hat = sampler.predict_x0(z, s, caps)
            cols.append(residual(x0_hat, x))
        per_sigma = torch.stack(cols, dim=1)
        calls = len(sig) * len(sampler._chunks(b))
        return DRResult(dr=per_sigma.mean(dim=1), per_sigma=per_sigma, sigmas=sig, calls=int(calls),
                        table_key=sampler.table.key)


def teacher_dca(sampler: TeacherSampler, x0, cond_a, cond_b, *, seeds, sigmas=DCA_SIGMAS, level=0) -> DCAResult:
    with torch.no_grad():
        sig = _prep_sigmas(sampler, sigmas)
        sampler._check_latent(x0, "x0")
        b, _, h, w = x0.shape
        caps_a = [c.detach() for c in sampler._resolve_cond(cond_a, b)]
        caps_b = [c.detach() for c in sampler._resolve_cond(cond_b, b)]
        seed_list = sampler._resolve_seeds(seeds, b)
        x = sampler._to_state(x0).detach()
        zs = _make_z(sampler, x, sig, seed_list, (h, w), level)
        cols_a, cols_b = [], []
        for s, z in zip(sig, zs):                       # sigma 우선, 같은 sigma 안에서 a -> b
            ha = sampler.predict_x0(z, s, caps_a)
            hb = sampler.predict_x0(z, s, caps_b)
            cols_a.append(velocity_error(ha, x, s))
            cols_b.append(velocity_error(hb, x, s))
        err_a = torch.stack(cols_a, dim=1)
        err_b = torch.stack(cols_b, dim=1)
        calls = 2 * len(sig) * len(sampler._chunks(b))
        return DCAResult(dca=(err_b - err_a).mean(dim=1), err_a=err_a, err_b=err_b, sigmas=sig, calls=int(calls),
                         table_key=sampler.table.key)


def delta_photo(sampler: TeacherSampler, frame, ref, c_photo, c_style, *, seeds, sigmas=DCA_SIGMAS,
                level=0) -> torch.Tensor:
    """Delta_photo = DCA(F; photo, style) - DCA(I; photo, style). 양수면 사진풍으로 끌렸다. 8 fwd."""
    with torch.no_grad():
        sampler._check_latent(frame, "frame")
        sampler._check_latent(ref, "ref")
        if tuple(frame.shape) != tuple(ref.shape):
            raise DCAError(f"frame {tuple(frame.shape)} 와 ref {tuple(ref.shape)} 의 모양이 다르다")
        a = teacher_dca(sampler, frame, c_photo, c_style, seeds=seeds, sigmas=sigmas, level=level)
        b = teacher_dca(sampler, ref, c_photo, c_style, seeds=seeds, sigmas=sigmas, level=level)
        return (a.dca - b.dca).detach()


# --------------------------------------------------------------------------- DR_rel · 판정

def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def dr_rel(dr, dr_ref, *, floor: float = DR_FLOOR) -> torch.Tensor:
    """원소별 DR/DR_ref, float64. dr_ref 는 스칼라(0차원 포함) 또는 같은 길이."""
    try:
        d = torch.as_tensor(dr).detach().to(torch.float64)
    except (TypeError, ValueError, RuntimeError) as e:
        raise DCAError(f"dr 를 텐서로 바꿀 수 없다: {e}") from e
    if _is_num(dr_ref):
        r = torch.tensor(float(dr_ref), dtype=torch.float64)
    elif isinstance(dr_ref, (torch.Tensor, list, tuple, np.ndarray)):
        try:
            r = torch.as_tensor(dr_ref).detach().to(torch.float64)
        except (TypeError, ValueError, RuntimeError) as e:
            raise DCAError(f"dr_ref 를 텐서로 바꿀 수 없다: {e}") from e
    else:
        raise DCAError(f"dr_ref 는 스칼라 또는 텐서여야 한다: {type(dr_ref).__name__}")
    if r.ndim != 0 and tuple(r.shape) != tuple(d.shape):
        raise DCAError(f"dr_ref 모양 {tuple(r.shape)} 가 dr {tuple(d.shape)} 와 맞지 않는다")
    if not bool(torch.isfinite(d).all()) or not bool(torch.isfinite(r).all()):
        raise DCAError("dr 또는 dr_ref 에 NaN/Inf 가 있다")
    if bool((d < 0).any()):
        raise DCAError("dr 에 음수가 있다")
    if bool((r < floor).any()):
        raise DCAError(f"dr_ref 가 바닥 {floor} 보다 작다")
    return d / r


@dataclass(frozen=True)
class DRVerdict:
    rel: torch.Tensor
    accept: torch.Tensor
    max_rel: float


def dr_verdict(dr, dr_ref, *, max_rel: float = DR_REL_MAX) -> DRVerdict:
    """rel <= max_rel 이면 합격(경계 포함)."""
    if not _is_num(max_rel) or not math.isfinite(float(max_rel)) or not float(max_rel) > 0.0:
        raise DCAError(f"max_rel 은 유한 양수여야 한다: {max_rel!r}")
    rel = dr_rel(dr, dr_ref)
    return DRVerdict(rel=rel, accept=rel <= float(max_rel), max_rel=float(max_rel))


def _vec(x, name: str) -> np.ndarray:
    try:
        a = torch.as_tensor(x).detach().to(torch.float64).numpy()
    except (TypeError, ValueError, RuntimeError) as e:
        raise DCAError(f"{name} 를 숫자 배열로 바꿀 수 없다: {e}") from e
    if a.ndim != 1 or a.size < 1:
        raise DCAError(f"{name} 는 비어 있지 않은 1-D 여야 한다: shape {a.shape}")
    if not bool(np.isfinite(a).all()):
        raise DCAError(f"{name} 에 NaN/Inf 가 있다")
    return a


def dr_auc(dr_real, dr_fake) -> float:
    """P(fake > real) + 0.5 P(동률), 전 쌍."""
    real, fake = _vec(dr_real, "dr_real"), _vec(dr_fake, "dr_fake")
    rs = np.sort(real)
    lo = np.searchsorted(rs, fake, side="left").astype(np.int64)      # real < f 의 개수
    hi = np.searchsorted(rs, fake, side="right").astype(np.int64)     # real <= f 의 개수
    twice = int((lo + hi).sum())                                      # 2*(less) + (ties)
    return float(twice / (2 * real.size * fake.size))


def rank_topk(scores, k) -> torch.Tensor:
    """큰 점수 먼저, 동률은 낮은 인덱스 먼저. int64[min(k, n)]."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or int(k) < 1:
        raise DCAError(f"k 는 1 이상의 정수여야 한다: {k!r}")
    a = _vec(scores, "scores")
    t = torch.as_tensor(a)
    order = torch.sort(t, descending=True, stable=True).indices
    return order[: int(k)].to(torch.int64)
