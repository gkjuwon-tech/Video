"""TCH-10 양자화 (T4 부록): INT8(SmoothQuant) · W4A4(SVDQuant식 저랭크) 수치 에뮬레이션, 자체 보정셋 + prov 검사.

CPU에서는 fake-quant 에뮬레이션이 정본이다. TRT 엔진·Nunchaku 커널은 GPU 부록(E0, 미확인)이며,
이 모듈은 tensorrt·nunchaku를 import하지 않는다(`importlib.util.find_spec`로 존재만 확인).
계약서: ops/harness/contracts/TCH-10.md
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from buc.gov import prov as _prov
from buc.gov.prov import FirewallError
from buc.teacher.t4_fp16 import Fp32Island

__all__ = [
    "QUANT_TAG", "MODES", "QuantError", "QuantSpecError", "BackendUnavailable", "FirewallError",
    "qmax", "absmax_scale", "quantize", "dequantize", "fake_quant", "smooth_factors", "lowrank_split",
    "QuantSpec", "QuantLinear", "collect_act_absmax", "CalibSample", "CalibSet", "build_calib_set",
    "assert_not_prequantized", "weight_digest", "quant_key", "QuantReport", "quantize_model",
    "spearman", "PrecisionGate", "precision_gate", "tensorrt_available", "nunchaku_available",
    "build_trt_engine",
]

QUANT_TAG = "buc.quant.v1"
MODES = ("int8", "w4a4")

_EPS = 1e-8
_QUANT_SUFFIXES = frozenset({
    "qweight", "qzeros", "scales", "wscales", "w_scale", "a_scale", "weight_scale", "input_scale",
    "smooth", "smooth_factor", "lora_up", "lora_down", "proj_up", "proj_down", "SCB", "absmax", "quant_map",
})


class QuantError(RuntimeError):
    """양자화 일반 오류."""


class QuantSpecError(QuantError, ValueError):
    """잘못된 양자화 사양·인자."""


class BackendUnavailable(QuantError):
    """tensorrt·nunchaku 같은 GPU 백엔드가 없다."""


# ---------------------------------------------------------------------------------------------
# 원시 연산
# ---------------------------------------------------------------------------------------------

def qmax(bits: int) -> int:
    """대칭·제한 범위 최댓값: 8→127, 4→7."""
    if isinstance(bits, bool) or bits not in (8, 4):
        raise QuantSpecError(f"지원하지 않는 비트 수: {bits!r} (8 또는 4)")
    return 127 if bits == 8 else 7


def _check_group(k: int, group_size: int) -> None:
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise QuantSpecError(f"group_size는 양의 정수여야 한다: {group_size!r}")
    if k % group_size != 0:
        raise QuantSpecError(f"마지막 축 {k}는 group_size {group_size}로 나누어떨어져야 한다")


def absmax_scale(x: torch.Tensor, bits: int, *, group_size: Optional[int] = None) -> torch.Tensor:
    """absmax/qmax 스케일(float32). group_size면 마지막 축을 g씩 묶어 x.shape[:-1]+(K//g,)."""
    qm = qmax(bits)
    xf = x.detach().float()
    if group_size is None:
        amax = xf.abs().max() if xf.numel() else torch.zeros((), dtype=torch.float32, device=xf.device)
        return (amax.clamp_min(_EPS) / qm).to(torch.float32)
    if xf.dim() == 0:
        raise QuantSpecError("group_size에는 1차원 이상 텐서가 필요하다")
    k = xf.shape[-1]
    _check_group(k, group_size)
    g = xf.reshape(*xf.shape[:-1], k // group_size, group_size)
    return (g.abs().amax(-1).clamp_min(_EPS) / qm).to(torch.float32)


def _as_scale(scale: Any, like: torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(scale, dtype=torch.float32, device=like.device)
    if s.numel() and not (bool(torch.isfinite(s).all()) and bool((s > 0).all())):
        raise QuantSpecError("scale은 유한한 양수여야 한다")
    return s


def quantize(x: torch.Tensor, scale: Any, bits: int, *, group_size: Optional[int] = None) -> torch.Tensor:
    """clamp(round(x/scale), ±qmax) → int8."""
    qm = qmax(bits)
    xf = x.detach().float()
    s = _as_scale(scale, xf)
    if group_size is None:
        q = torch.round(xf / s)
    else:
        if xf.dim() == 0:
            raise QuantSpecError("group_size에는 1차원 이상 텐서가 필요하다")
        k = xf.shape[-1]
        _check_group(k, group_size)
        if tuple(s.shape) != tuple(xf.shape[:-1]) + (k // group_size,):
            raise QuantSpecError(f"그룹 scale 모양 불일치: {tuple(s.shape)}")
        g = xf.reshape(*xf.shape[:-1], k // group_size, group_size)
        q = torch.round(g / s.unsqueeze(-1)).reshape(xf.shape)
    return q.clamp_(-qm, qm).to(torch.int8)


def dequantize(q: torch.Tensor, scale: Any, *, group_size: Optional[int] = None) -> torch.Tensor:
    """q*scale → float32."""
    qf = q.float()
    s = torch.as_tensor(scale, dtype=torch.float32, device=qf.device)
    if group_size is None:
        return qf * s
    if qf.dim() == 0:
        raise QuantSpecError("group_size에는 1차원 이상 텐서가 필요하다")
    k = qf.shape[-1]
    _check_group(k, group_size)
    if tuple(s.shape) != tuple(qf.shape[:-1]) + (k // group_size,):
        raise QuantSpecError(f"그룹 scale 모양 불일치: {tuple(s.shape)}")
    g = qf.reshape(*qf.shape[:-1], k // group_size, group_size)
    return (g * s.unsqueeze(-1)).reshape(qf.shape)


def fake_quant(x: torch.Tensor, bits: int, *, group_size: Optional[int] = None,
               scale: Any = None) -> torch.Tensor:
    """dequantize(quantize(x)) — x의 dtype으로 반환."""
    if scale is None:
        scale = absmax_scale(x, bits, group_size=group_size)
    d = dequantize(quantize(x, scale, bits, group_size=group_size), scale, group_size=group_size)
    return d.to(x.dtype)


def smooth_factors(act_absmax: torch.Tensor, w_absmax: torch.Tensor, alpha: float) -> torch.Tensor:
    """s = clamp(a,1e-5)^α / clamp(w,1e-5)^(1-α) (float32)."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not (0.0 <= alpha <= 1.0):
        raise QuantSpecError(f"alpha는 [0,1] 안이어야 한다: {alpha!r}")
    a = torch.as_tensor(act_absmax).detach().double()
    w = torch.as_tensor(w_absmax).detach().double()
    if a.shape != w.shape or a.dim() != 1:
        raise QuantSpecError(f"act_absmax {tuple(a.shape)}와 w_absmax {tuple(w.shape)}는 같은 1차원이어야 한다")
    s = a.clamp_min(1e-5) ** float(alpha) / w.clamp_min(1e-5) ** (1.0 - float(alpha))
    return s.to(torch.float32)


def lowrank_split(w: torch.Tensor, rank: int):
    """w ≈ up@down + resid. float64 SVD 절단(up=U·S, down=Vh). 반환은 float32."""
    if w.dim() != 2:
        raise QuantSpecError("lowrank_split은 2차원 가중치가 필요하다")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise QuantSpecError(f"rank는 0 이상 정수여야 한다: {rank!r}")
    o, k = w.shape
    r = min(rank, o, k)
    w64 = w.detach().double()
    if r == 0:
        up = torch.zeros(o, 0, dtype=torch.float32, device=w.device)
        down = torch.zeros(0, k, dtype=torch.float32, device=w.device)
        return up, down, w.detach().float().clone()
    u, s, vh = torch.linalg.svd(w64, full_matrices=False)
    up64 = u[:, :r] * s[:r]
    down64 = vh[:r]
    resid = (w64 - up64 @ down64).to(torch.float32)
    return up64.to(torch.float32), down64.to(torch.float32), resid


# ---------------------------------------------------------------------------------------------
# 사양
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantSpec:
    mode: str
    alpha: float = 0.5
    smooth: bool = True
    group_size: int = 64
    rank: int = 32
    skip: tuple = ()

    def __post_init__(self):
        if self.mode not in MODES:
            raise QuantSpecError(f"mode는 {MODES} 중 하나: {self.mode!r}")
        a = self.alpha
        if (isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a)
                or not (0.0 <= a <= 1.0)):
            raise QuantSpecError(f"alpha는 [0,1] 안이어야 한다: {a!r}")
        if not isinstance(self.smooth, bool):
            raise QuantSpecError("smooth는 bool이어야 한다")
        for name, lo in (("group_size", 1), ("rank", 0)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise QuantSpecError(f"{name}는 {lo} 이상 정수여야 한다: {v!r}")
        sk = self.skip
        if isinstance(sk, (str, bytes)) or not isinstance(sk, Iterable):
            raise QuantSpecError("skip은 이름의 iterable이어야 한다")
        sk = tuple(sk)
        if not all(isinstance(n, str) and n for n in sk):
            raise QuantSpecError("skip의 원소는 비어 있지 않은 str이어야 한다")
        object.__setattr__(self, "skip", tuple(sorted(set(sk))))

    @property
    def w_bits(self) -> int:
        return 8 if self.mode == "int8" else 4

    @property
    def a_bits(self) -> int:
        return 8 if self.mode == "int8" else 4

    def to_dict(self) -> dict:
        return {"mode": self.mode, "alpha": float(self.alpha), "smooth": self.smooth,
                "group_size": self.group_size, "rank": self.rank, "skip": list(self.skip)}


# ---------------------------------------------------------------------------------------------
# QuantLinear
# ---------------------------------------------------------------------------------------------

class QuantLinear(nn.Module):
    """nn.Linear의 양자화 에뮬레이션. forward는 float32로 계산하고 x.dtype으로 돌려준다."""

    def __init__(self):
        super().__init__()
        self.mode = "int8"
        self.group_size = 64
        self.in_features = 0
        self.out_features = 0
        self.spec: Optional[QuantSpec] = None
        for name in ("inv_smooth", "qweight", "w_scale", "bias", "a_scale", "lora_up", "lora_down"):
            self.register_buffer(name, None)

    @classmethod
    def from_linear(cls, linear: nn.Module, spec: QuantSpec, act_absmax: torch.Tensor) -> "QuantLinear":
        if not isinstance(linear, nn.Linear):
            raise QuantSpecError(f"nn.Linear만 양자화할 수 있다: {type(linear).__name__}")
        if not isinstance(spec, QuantSpec):
            raise TypeError("spec은 QuantSpec이어야 한다")
        o, k = linear.weight.shape
        act = torch.as_tensor(act_absmax).detach().float().to(linear.weight.device)
        if act.shape != (k,):
            raise QuantSpecError(f"act_absmax 모양 {tuple(act.shape)} != ({k},)")
        if not bool(torch.isfinite(act).all()):
            raise QuantSpecError("act_absmax에 비유한 값이 있다")
        act = act.abs()
        if spec.mode == "w4a4":
            _check_group(k, spec.group_size)

        w = linear.weight.detach().float()
        if spec.smooth:
            s = smooth_factors(act, w.abs().amax(0), spec.alpha)
            inv = 1.0 / s
        else:
            s = torch.ones(k, dtype=torch.float32, device=w.device)
            inv = torch.ones(k, dtype=torch.float32, device=w.device)
        wp = w * s  # W' = W·s(열)

        self = cls()
        self.mode, self.spec = spec.mode, spec
        self.group_size = spec.group_size
        self.in_features, self.out_features = k, o
        self.inv_smooth = inv.contiguous()
        self.bias = None if linear.bias is None else linear.bias.detach().float().clone()
        if spec.mode == "int8":
            w_scale = (wp.abs().amax(1).clamp_min(_EPS) / 127).to(torch.float32)
            self.qweight = quantize(wp, w_scale[:, None], 8)
            self.w_scale = w_scale
            self.a_scale = ((act / s).max() / 127).clamp_min(_EPS).to(torch.float32)
        else:
            up, down, resid = lowrank_split(wp, spec.rank)
            g = spec.group_size
            w_scale = absmax_scale(resid, 4, group_size=g)
            self.qweight = quantize(resid, w_scale, 4, group_size=g)
            self.w_scale = w_scale
            self.lora_up, self.lora_down = up.contiguous(), down.contiguous()
        return self

    def dequant_weight(self) -> torch.Tensor:
        """스무딩 공간의 가중치 [O,K]: deq(R)(+up@down). ·inv_smooth면 원 W 근사."""
        if self.mode == "int8":
            return dequantize(self.qweight, self.w_scale[:, None])
        r = dequantize(self.qweight, self.w_scale, group_size=self.group_size)
        if self.lora_up is not None and self.lora_up.shape[1] > 0:
            r = r + self.lora_up @ self.lora_down
        return r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xs = x.float() * self.inv_smooth
        if self.mode == "int8":
            a = self.a_scale
            xa = torch.clamp(torch.round(xs / a), -127, 127) * a
            y = xa @ dequantize(self.qweight, self.w_scale[:, None]).T
        else:
            xa = fake_quant(xs, 4, group_size=self.group_size)
            y = xa @ dequantize(self.qweight, self.w_scale, group_size=self.group_size).T
            if self.lora_up.shape[1] > 0:
                y = y + (xs @ self.lora_down.T) @ self.lora_up.T
        if self.bias is not None:
            y = y + self.bias
        return y.to(dt)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode={self.mode}"


# ---------------------------------------------------------------------------------------------
# 활성 통계 수집
# ---------------------------------------------------------------------------------------------

def _island_names(model: nn.Module) -> list:
    return [n for n, m in model.named_modules() if isinstance(m, Fp32Island)]


def _under(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _linear_targets(model: nn.Module) -> list:
    """Fp32Island 밖의 모든 Linear 이름(정렬 안 함, named_modules 순서)."""
    islands = _island_names(model)
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n and not _under(n, islands)]


def collect_act_absmax(model: nn.Module, batches: Iterable, *, run_fn: Optional[Callable] = None,
                       names: Optional[Sequence[str]] = None) -> dict:
    """Linear 입력의 채널별 |x| 최댓값 누적. 호출되지 않은 모듈은 키가 없다."""
    mods = dict(model.named_modules())
    if names is None:
        names = _linear_targets(model)
    else:
        names = list(names)
        for n in names:
            if n not in mods:
                raise QuantSpecError(f"없는 모듈 이름: {n!r}")
            if not isinstance(mods[n], nn.Linear):
                raise QuantSpecError(f"Linear가 아닌 모듈: {n!r}")
    stats: dict = {}
    handles = []

    def make_hook(name):
        def hook(_mod, args):
            x = args[0]
            xd = x.detach()
            if not bool(torch.isfinite(xd).all()):
                raise QuantError(f"{name}: 비유한 활성이 보정 입력에 있다")
            m = xd.abs().reshape(-1, xd.shape[-1]).amax(0).float()
            prev = stats.get(name)
            stats[name] = m if prev is None else torch.maximum(prev, m)
        return hook

    run = run_fn if run_fn is not None else (lambda mod, b: mod(b))
    try:
        for n in names:
            handles.append(mods[n].register_forward_pre_hook(make_hook(n)))
        with torch.no_grad():
            for b in batches:
                run(model, b)
    finally:
        for h in handles:
            h.remove()
    return {n: stats[n] for n in names if n in stats}


# ---------------------------------------------------------------------------------------------
# 보정셋 + prov
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibSample:
    id: str
    data: torch.Tensor
    prov: Any


def _tensor_hash_update(h, t: torch.Tensor) -> None:
    c = t.detach().cpu().contiguous().reshape(-1)
    h.update(str(c.dtype).encode() + b"|" + repr(tuple(t.shape)).encode() + b"|")
    if c.numel():
        h.update(c.view(torch.uint8).numpy().tobytes())
    h.update(b";")


@dataclass(frozen=True)
class CalibSet:
    samples: tuple
    prov: Any
    digest: str
    rejected: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "rejected", tuple(self.rejected))

    @property
    def ids(self) -> tuple:
        return tuple(s.id for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def batches(self, bs: int) -> list:
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
            raise ValueError(f"배치 크기는 1 이상 정수여야 한다: {bs!r}")
        out = []
        for i in range(0, len(self.samples), bs):
            out.append(torch.stack([s.data for s in self.samples[i:i + bs]]))
        return out


def _calib_digest(samples: Sequence[CalibSample]) -> str:
    h = hashlib.sha256()
    h.update(QUANT_TAG.encode() + b"|calib|")
    for s in sorted(samples, key=lambda s: s.id):
        h.update(s.id.encode() + b"|")
        _tensor_hash_update(h, s.data)
    return h.hexdigest()


def build_calib_set(candidates: Sequence[CalibSample], n: int, *, seed: int = 0,
                    strict: bool = True) -> CalibSet:
    """prov가 train_ok인 후보에서 결정론적으로 n개를 고른다. strict면 하나라도 나쁘면 FirewallError."""
    cands = list(candidates)
    for c in cands:
        if not isinstance(c, CalibSample):
            raise TypeError("candidates의 원소는 CalibSample이어야 한다")
        if not isinstance(c.id, str):
            raise TypeError("CalibSample.id는 str이어야 한다")
        if not isinstance(c.data, torch.Tensor):
            raise TypeError(f"CalibSample {c.id!r}의 data가 텐서가 아니다: {type(c.data).__name__}")
    ids = [c.id for c in cands]
    if len(set(ids)) != len(ids):
        raise ValueError("후보 id가 중복된다")
    if cands:
        shape0 = tuple(cands[0].data.shape)
        for c in cands:
            if tuple(c.data.shape) != shape0:
                raise ValueError(f"모양 불일치: {c.id!r} {tuple(c.data.shape)} != {shape0}")

    good, bad = [], []
    for c in cands:
        why = _prov.why_not_train(c.prov)
        (bad if why else good).append((c, why))
    if bad and strict:
        detail = "; ".join(f"{c.id}: {','.join(w)}" for c, w in sorted(bad, key=lambda t: t[0].id)[:8])
        raise FirewallError(f"보정셋 후보 {len(bad)}개가 train_ok가 아니다 ({detail})")
    avail = sorted((c for c, _ in good), key=lambda c: c.id)
    if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= len(avail)):
        raise ValueError(f"n={n!r}은 [1, {len(avail)}] 안이어야 한다")
    chosen = random.Random(seed).sample(avail, n)
    rec = _prov.derive("quant-calib", [_prov.as_record(c.prov) for c in chosen])
    _prov.assert_train_ok(rec, "quant-calib")
    return CalibSet(samples=tuple(chosen), prov=rec, digest=_calib_digest(chosen),
                    rejected=tuple(sorted(c.id for c, _ in bad)))


def _check_calib_prov(calib: CalibSet, what: str) -> None:
    """소비자 쪽 재검사: 집합 prov와 각 표본 prov가 모두 train_ok여야 한다."""
    _prov.assert_train_ok(calib.prov, f"{what}: 보정셋 prov")
    for s in calib.samples:
        _prov.assert_train_ok(s.prov, f"{what}: 보정 표본 {s.id!r}")


# ---------------------------------------------------------------------------------------------
# 사전 양자화 금지 · 다이제스트 · 키
# ---------------------------------------------------------------------------------------------

def assert_not_prequantized(state_dict: Any) -> None:
    """정수형 텐서(0-d int64 카운터 제외)나 양자화 접미사 키가 있으면 FirewallError."""
    bad = []
    for k, v in state_dict.items():
        if str(k).rsplit(".", 1)[-1] in _QUANT_SUFFIXES:
            bad.append(str(k))
            continue
        if isinstance(v, torch.Tensor) and not (v.is_floating_point() or v.is_complex() or v.dtype == torch.bool):
            if v.dim() == 0 and v.dtype == torch.int64:
                continue
            bad.append(str(k))
    if bad:
        raise FirewallError("제3자 사전 양자화 가중치는 금지: " + ", ".join(sorted(bad)[:8]))


def weight_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    h.update(QUANT_TAG.encode() + b"|weights|")
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode() + b"|")
        if isinstance(v, torch.Tensor):
            _tensor_hash_update(h, v)
    return h.hexdigest()


def quant_key(spec: QuantSpec, calib_digest: str, weight_digest: str = "", engine_hash: str = "") -> str:
    if not isinstance(spec, QuantSpec):
        raise TypeError("spec은 QuantSpec이어야 한다")
    payload = {"tag": QUANT_TAG, "spec": spec.to_dict(), "calib": calib_digest,
               "weights": weight_digest, "engine": engine_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# 모델 양자화
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantReport:
    mode: str
    quantized: tuple
    skipped: tuple
    prov: Any
    calib_digest: str
    weight_digest: str
    key: str


def quantize_model(model: nn.Module, spec: QuantSpec, calib: CalibSet, *, weight_prov: Any,
                   run_fn: Optional[Callable] = None, batch_size: int = 1) -> QuantReport:
    """섬 밖 Linear를 제자리에서 QuantLinear로 교체. 모든 검사는 모델을 바꾸기 전에 한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model은 nn.Module이어야 한다")
    if not isinstance(spec, QuantSpec):
        raise TypeError("spec은 QuantSpec이어야 한다")
    if not isinstance(calib, CalibSet):
        raise TypeError("calib은 CalibSet이어야 한다")
    if getattr(model, "_buc_quant", None) is not None:
        raise QuantError("이미 양자화된 모델이다(이중 적용 금지)")
    if len(calib) == 0:
        raise QuantError("빈 보정셋")

    _check_calib_prov(calib, "quantize")
    out_prov = _prov.derive("quantize", [weight_prov, calib.prov], updates_shared_weights=True)
    _prov.assert_train_ok(out_prov, "quantize")
    assert_not_prequantized(model.state_dict())

    mods = dict(model.named_modules())
    for n in spec.skip:
        if n not in mods:
            raise QuantSpecError(f"skip에 없는 모듈 이름: {n!r}")

    all_linear = sorted(n for n, m in mods.items() if isinstance(m, nn.Linear) and n)
    targets = [n for n in _linear_targets(model) if not _under(n, spec.skip)]
    w_digest = weight_digest(model)

    stats = collect_act_absmax(model, calib.batches(batch_size), run_fn=run_fn, names=targets)
    built = {}
    for n in targets:
        if n not in stats:
            continue  # 보정 중 호출되지 않음 → float 유지
        lin = mods[n]
        if spec.mode == "w4a4" and lin.in_features % spec.group_size != 0:
            continue
        built[n] = QuantLinear.from_linear(lin, spec, stats[n])

    for n, ql in built.items():  # 여기까지 오면 실패 요인이 없다: 한꺼번에 교체
        parent_name, _, leaf = n.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, ql)

    quantized = tuple(sorted(built))
    report = QuantReport(
        mode=spec.mode,
        quantized=quantized,
        skipped=tuple(n for n in all_linear if n not in built),
        prov=out_prov,
        calib_digest=calib.digest,
        weight_digest=w_digest,
        key=quant_key(spec, calib.digest, w_digest),
    )
    model._buc_quant = report
    return report


# ---------------------------------------------------------------------------------------------
# 판정 정밀도 게이트
# ---------------------------------------------------------------------------------------------

def _as_vec(a: Any) -> np.ndarray:
    v = np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("1차원 수열이 필요하다")
    return v


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Any, b: Any) -> float:
    """동률 평균 순위의 Spearman 상관. 분산 0이면 0.0."""
    x, y = _as_vec(a), _as_vec(b)
    if len(x) != len(y):
        raise ValueError(f"길이 불일치: {len(x)} != {len(y)}")
    if len(x) < 2:
        raise ValueError("n >= 2가 필요하다")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("비유한 값이 있다")
    rx, ry = _avg_ranks(x) - (len(x) + 1) / 2.0, _avg_ranks(y) - (len(y) + 1) / 2.0
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if den == 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, float((rx * ry).sum()) / den)))


@dataclass(frozen=True)
class PrecisionGate:
    rho: float
    mean_abs_delta: float
    max_abs_delta: float
    n: int
    ok: bool


def precision_gate(ref: Any, test: Any, *, min_rho: float = 0.9, max_delta: float = 0.05) -> PrecisionGate:
    """판정 D의 순위 상관과 평균 |ΔD|로 양자화 대리를 허용할지 정한다."""
    rho = spearman(ref, test)
    d = np.abs(_as_vec(ref) - _as_vec(test))
    mean_d, max_d, n = float(d.mean()), float(d.max()), int(len(d))
    return PrecisionGate(rho=rho, mean_abs_delta=mean_d, max_abs_delta=max_d, n=n,
                         ok=bool(n >= 3 and rho >= min_rho and mean_d <= max_delta))


# ---------------------------------------------------------------------------------------------
# GPU 백엔드 (E0, 미확인)
# ---------------------------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def tensorrt_available() -> bool:
    return _has_module("tensorrt")


def nunchaku_available() -> bool:
    return _has_module("nunchaku")


def build_trt_engine(onnx_path: Any, out_path: Any, *, calib: CalibSet, precision: str = "int8",
                     workspace_gb: float = 1.0) -> str:
    """ONNX → TRT 엔진(INT8 엔트로피 보정은 우리 보정셋만 쓴다). 반환: 엔진 sha256.

    GPU 부록(E0): 이 경로는 CPU 컨테이너에서 실행·검증되지 않았다.
    """
    if not isinstance(calib, CalibSet):
        raise TypeError("calib은 CalibSet이어야 한다")
    if precision not in ("int8", "fp16"):
        raise QuantSpecError(f"precision은 int8|fp16: {precision!r}")
    _check_calib_prov(calib, "build_trt_engine")
    if not tensorrt_available():
        raise BackendUnavailable("tensorrt가 설치돼 있지 않다")

    import tensorrt as trt  # noqa: PLC0415  (GPU 러너에서만)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(os.fspath(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise QuantError(f"ONNX 파싱 실패: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.INT8)
        batches = calib.batches(1)

        class _Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self._it = iter(batches)
                self._cur = None

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                try:
                    self._cur = next(self._it).cuda().contiguous()
                except StopIteration:
                    return None
                return [int(self._cur.data_ptr())]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                pass

        config.int8_calibrator = _Calibrator()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise QuantError("TRT 엔진 빌드 실패")
    data = bytes(blob)
    out = os.fspath(out_path)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out)
    return hashlib.sha256(data).hexdigest()
