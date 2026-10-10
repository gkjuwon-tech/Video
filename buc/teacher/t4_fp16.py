"""TCH-09 FP16 Rescue: T4(BF16 없음)에서 BF16 교사를 학습 없이 동치로 FP16에서 돌리는 부록.

(i) 잔차 스트림에 쓰는 모든 쓰기에 1/s, 스트림을 읽는 norm의 eps에 1/s²,
(ii) SwiGLU 선형 쪽 W_up 1/t, W_down 입력 쪽 t,
(iii) norm 등은 FP32 섬, SDPA는 FP32 누산,
(iv) max-abs 프로파일로 s, t를 2의 거듭제곱으로 고른다.

import는 torch·numpy·scipy만, CUDA·diffusers 없이 되고 부작용이 없다(scipy는 upper95에서 지연 import).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "FP16_MAX", "RescueError", "RescuePlanError",
    "scale_output_", "scale_input_", "scale_affine_", "scale_eps_",
    "SwiGLUSite", "RescuePlan", "RescueReport", "apply_rescue",
    "Fp32Island", "apply_fp32_islands", "cast_fp16",
    "sdpa_fp32", "fp32_sdpa", "MaxAbsProfiler", "choose_scale",
    "upper95", "AuditReport", "audit_images", "zimage_plan",
]

FP16_MAX = 65504.0

_ORIG_SDPA = F.scaled_dot_product_attention
_MARK = "_buc_fp16_rescue"


class RescueError(RuntimeError):
    """FP16 rescue 일반 오류."""


class RescuePlanError(RescueError, ValueError):
    """계획·이름·타입·범위 오류. 발생 시 모델은 변경되지 않는다."""


# ---------------------------------------------------------------------------
# 원시 연산
# ---------------------------------------------------------------------------

def _num(f, what="f") -> float:
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f):
        raise RescuePlanError(f"{what}는 유한 실수여야 한다: {f!r}")
    return float(f)


def _unwrap(mod):
    return mod.inner if isinstance(mod, Fp32Island) else mod


def _need_linear(lin, what="module") -> nn.Linear:
    if not isinstance(lin, nn.Linear):
        raise RescuePlanError(f"{what}: nn.Linear가 아니다 ({type(lin).__name__})")
    return lin


def scale_output_(linear, f, rows=None):
    """제자리: W[rows] *= f, b[rows] *= f (rows는 slice 또는 None=전체)."""
    _need_linear(linear)
    f = _num(f)
    if rows is None:
        rows = slice(None)
    elif not isinstance(rows, slice):
        raise RescuePlanError(f"rows는 slice 또는 None이어야 한다: {rows!r}")
    with torch.no_grad():
        linear.weight[rows] *= f
        if linear.bias is not None:
            linear.bias[rows] *= f
    return linear


def scale_input_(linear, f):
    """제자리: W *= f, 바이어스는 그대로."""
    _need_linear(linear)
    f = _num(f)
    with torch.no_grad():
        linear.weight *= f
    return linear


def scale_affine_(norm, f):
    """제자리: weight *= f, bias가 있으면 bias *= f. weight가 없으면 RescuePlanError."""
    norm = _unwrap(norm)
    f = _num(f)
    w = getattr(norm, "weight", None)
    if not isinstance(w, torch.Tensor):
        raise RescuePlanError(f"affine weight가 없는 norm이다 ({type(norm).__name__})")
    with torch.no_grad():
        w *= f
        b = getattr(norm, "bias", None)
        if isinstance(b, torch.Tensor):
            b *= f
    return norm


def _eps_attr(norm) -> Optional[str]:
    for name in ("eps", "variance_epsilon"):
        v = getattr(norm, name, None)
        if v is not None and not isinstance(v, bool) and isinstance(v, (int, float)):
            return name
    return None


def scale_eps_(norm, f):
    """제자리: eps(없으면 variance_epsilon) *= f. 둘 다 없거나 None이면 RescuePlanError."""
    norm = _unwrap(norm)
    f = _num(f)
    name = _eps_attr(norm)
    if name is None:
        raise RescuePlanError(f"eps 속성이 없는 모듈이다 ({type(norm).__name__})")
    setattr(norm, name, float(getattr(norm, name)) * f)
    return norm


# ---------------------------------------------------------------------------
# 계획
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwiGLUSite:
    up: str
    gate: str
    down: str
    scale: float


def _is_pow2(x) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    x = float(x)
    if not math.isfinite(x) or x < 1.0:
        return False
    return math.frexp(x)[0] == 0.5


_NAME_FIELDS = ("embed", "stream_params", "residual_out", "stream_norms", "unscale", "fp32_islands")


@dataclass(frozen=True)
class RescuePlan:
    s: float
    embed: tuple = ()
    stream_params: tuple = ()
    gate_rows: tuple = ()
    residual_out: tuple = ()
    stream_norms: tuple = ()
    unscale: tuple = ()
    swiglu: tuple = ()
    fp32_islands: tuple = ()

    def __post_init__(self):
        for f in _NAME_FIELDS + ("swiglu",):
            v = getattr(self, f)
            if not isinstance(v, tuple):
                object.__setattr__(self, f, tuple(v))
        g = tuple(tuple(r) if isinstance(r, (list, tuple)) else r for r in self.gate_rows)
        object.__setattr__(self, "gate_rows", g)

    def validate(self) -> "RescuePlan":
        if not _is_pow2(self.s):
            raise RescuePlanError(f"s는 유한·≥1·2의 거듭제곱이어야 한다: {self.s!r}")
        for f in _NAME_FIELDS:
            names = getattr(self, f)
            for n in names:
                if not isinstance(n, str) or not n:
                    raise RescuePlanError(f"{f}: 이름은 비어 있지 않은 str이어야 한다: {n!r}")
            if len(set(names)) != len(names):
                raise RescuePlanError(f"{f}: 이름 중복")
        for r in self.gate_rows:
            if (not isinstance(r, tuple) or len(r) != 3 or not isinstance(r[0], str) or not r[0]
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in r[1:])):
                raise RescuePlanError(f"gate_rows 항목은 (name, start, stop)이어야 한다: {r!r}")
            if not (0 <= r[1] < r[2]):
                raise RescuePlanError(f"gate_rows: 0<=start<stop 위반: {r!r}")
        if len(set(self.gate_rows)) != len(self.gate_rows):
            raise RescuePlanError("gate_rows: 중복")
        for site in self.swiglu:
            if not isinstance(site, SwiGLUSite):
                raise RescuePlanError(f"swiglu 항목은 SwiGLUSite여야 한다: {site!r}")
            for n in (site.up, site.gate, site.down):
                if not isinstance(n, str) or not n:
                    raise RescuePlanError(f"swiglu: 이름은 비어 있지 않은 str이어야 한다: {n!r}")
            if not _is_pow2(site.scale):
                raise RescuePlanError(f"swiglu t는 유한·≥1·2의 거듭제곱이어야 한다: {site.scale!r}")
        for attr in ("up", "down"):
            names = [getattr(x, attr) for x in self.swiglu]
            if len(set(names)) != len(names):
                raise RescuePlanError(f"swiglu.{attr}: 이름 중복")
        return self


@dataclass(frozen=True)
class RescueReport:
    s: float
    touched: tuple
    islands: tuple


def _get_sub(model, name):
    try:
        return model.get_submodule(name)
    except (AttributeError, TypeError) as e:
        raise RescuePlanError(f"모듈 없음: {name!r}") from e


def _get_param(model, name):
    try:
        return model.get_parameter(name)
    except (AttributeError, TypeError) as e:
        raise RescuePlanError(f"파라미터 없음: {name!r}") from e


def _is_norm_like(mod) -> bool:
    mod = _unwrap(mod)
    return isinstance(getattr(mod, "weight", None), torch.Tensor) and _eps_attr(mod) is not None


def apply_rescue(model, plan: RescuePlan) -> RescueReport:
    """제자리 rescue. 전부 검증한 뒤에만 변경한다. 이중 적용은 RescueError."""
    if getattr(model, _MARK, None) is not None:
        raise RescueError("이미 FP16 rescue가 적용된 모델이다")
    plan.validate()
    s = float(plan.s)
    inv_s, inv_s2 = 1.0 / s, 1.0 / (s * s)
    ops = []  # (callable, arg) 순서 고정
    touched = set()

    def lin(name, what):
        m = _get_sub(model, name)
        if not isinstance(m, nn.Linear):
            raise RescuePlanError(f"{what}: {name!r}는 nn.Linear가 아니다 ({type(m).__name__})")
        return m

    for n in plan.embed:
        m = lin(n, "embed")
        ops.append(lambda m=m: scale_output_(m, inv_s))
        touched.add(n)
    for n in plan.stream_params:
        p = _get_param(model, n)
        if not p.is_floating_point():
            raise RescuePlanError(f"stream_params: {n!r}는 부동소수 파라미터가 아니다")

        def _sp(p=p):
            with torch.no_grad():
                p.mul_(inv_s)
        ops.append(_sp)
        touched.add(n)
    for n, a, b in plan.gate_rows:
        m = lin(n, "gate_rows")
        if b > m.out_features:
            raise RescuePlanError(f"gate_rows: {n!r} 행 범위 {a}:{b} > out_features {m.out_features}")
        ops.append(lambda m=m, a=a, b=b: scale_output_(m, inv_s, rows=slice(a, b)))
        touched.add(n)
    for n in plan.residual_out:
        m = _get_sub(model, n)
        if isinstance(m, nn.Linear):
            ops.append(lambda m=m: scale_output_(m, inv_s))
        elif _is_norm_like(m):
            ops.append(lambda m=m: scale_affine_(m, inv_s))
        else:
            raise RescuePlanError(f"residual_out: {n!r}는 Linear도 (가중치+eps) norm도 아니다")
        touched.add(n)
    for n in plan.stream_norms:
        m = _unwrap(_get_sub(model, n))
        if _eps_attr(m) is None:
            raise RescuePlanError(f"stream_norms: {n!r}에 eps가 없다")
        ops.append(lambda m=m: scale_eps_(m, inv_s2))
        touched.add(n)
    for n in plan.unscale:
        m = lin(n, "unscale")
        ops.append(lambda m=m: scale_input_(m, s))
        touched.add(n)
    for site in plan.swiglu:
        up = lin(site.up, "swiglu.up")
        lin(site.gate, "swiglu.gate")
        down = lin(site.down, "swiglu.down")
        t = float(site.scale)
        ops.append(lambda m=up, t=t: scale_output_(m, 1.0 / t))
        ops.append(lambda m=down, t=t: scale_input_(m, t))
        touched.update((site.up, site.down))
    for n in plan.fp32_islands:
        if not n:
            raise RescuePlanError("fp32_islands: 루트는 섬이 될 수 없다")
        _get_sub(model, n)

    for op in ops:
        op()
    islands = apply_fp32_islands(model, list(plan.fp32_islands)) if plan.fp32_islands else []
    setattr(model, _MARK, s)
    return RescueReport(s=s, touched=tuple(sorted(touched)), islands=tuple(islands))


# ---------------------------------------------------------------------------
# FP32 섬
# ---------------------------------------------------------------------------

def _to32(x):
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() and x.dtype != torch.float32 else x
    if isinstance(x, (list, tuple)) and not hasattr(x, "_fields"):
        return type(x)(_to32(v) for v in x)
    return x


def _first_float(x):
    if isinstance(x, torch.Tensor):
        return x if x.is_floating_point() else None
    if isinstance(x, (list, tuple)):
        for v in x:
            r = _first_float(v)
            if r is not None:
                return r
    elif isinstance(x, dict):
        return _first_float(list(x.values()))
    return None


def _restore(x, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.is_floating_point() and x.dtype != dtype else x
    if isinstance(x, (list, tuple)) and not hasattr(x, "_fields"):
        return type(x)(_restore(v, dtype) for v in x)
    if isinstance(x, dict):
        return {k: _restore(v, dtype) for k, v in x.items()}
    return x


def _no_autocast(device_type: str):
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except (RuntimeError, ValueError):
        return contextlib.nullcontext()


def _keep_float_dtype(fn):
    """dtype만 바꾸는 변환(.half() 등)은 부동소수 텐서에 적용하지 않고, 장치 이동은 따른다."""
    def keep(t):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            probe = fn(torch.empty(0, dtype=t.dtype, device=t.device))
            if probe.dtype != t.dtype:
                return t if probe.device == t.device else t.to(probe.device)
        return fn(t)
    return keep


class Fp32Island(nn.Module):
    """inner를 float32·autocast 끈 채 실행하고 출력을 첫 부동소수 입력 dtype으로 되돌린다."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner.float()

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_float_dtype(fn), *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("__") or name == "inner":
                raise
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def forward(self, *args, **kwargs):
        first = _first_float((args, kwargs))
        a = tuple(_to32(v) for v in args)
        kw = {k: _to32(v) for k, v in kwargs.items()}
        dev = first.device.type if first is not None else "cpu"
        with _no_autocast(dev):
            out = self.inner(*a, **kw)
        return out if first is None else _restore(out, first.dtype)


def apply_fp32_islands(model, names) -> list:
    """각 서브모듈을 Fp32Island로 교체(멱등). 없는 이름은 변경 전 RescuePlanError."""
    names = list(dict.fromkeys(names))
    for n in names:
        if not isinstance(n, str) or not n:
            raise RescuePlanError(f"섬 이름이 잘못됐다: {n!r}")
        _get_sub(model, n)
    # 깊은 것부터 감싸야 바깥을 먼저 감싸도 안쪽 이름이 깨지지 않는다.
    for n in sorted(names, key=lambda x: -x.count(".")):
        try:
            mod = model.get_submodule(n)
        except AttributeError:
            continue
        if isinstance(mod, Fp32Island):
            continue
        parent_name, _, leaf = n.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, Fp32Island(mod))
    return names


def cast_fp16(model):
    """섬 밖은 float16, 섬은 float32(섬이 .half()를 무시한다). 모델 반환."""
    model.half()
    return model


# ---------------------------------------------------------------------------
# FP32 누산 SDPA
# ---------------------------------------------------------------------------

def sdpa_fp32(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """float32로 계산하고 q.dtype으로 반환하는 scaled_dot_product_attention."""
    if attn_mask is not None and attn_mask.is_floating_point():
        attn_mask = attn_mask.float()
    out = _ORIG_SDPA(q.float(), k.float(), v.float(), attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale)
    return out.to(q.dtype)


@contextlib.contextmanager
def fp32_sdpa():
    """컨텍스트 안에서 F.scaled_dot_product_attention을 sdpa_fp32로 바꾸고 나갈 때 복원한다."""
    saved = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa_fp32
    try:
        yield
    finally:
        F.scaled_dot_product_attention = saved


# ---------------------------------------------------------------------------
# 프로파일러 / 스케일 선택
# ---------------------------------------------------------------------------

def _float_tensors(x):
    if isinstance(x, torch.Tensor):
        if x.is_floating_point():
            yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _float_tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _float_tensors(v)


def _stat(tensors):
    """(유한 값 |x| 최댓값, 비유한 포함 여부)."""
    mx, bad = 0.0, False
    for t in tensors:
        if t.numel() == 0:
            continue
        a = t.detach().abs()
        fin = torch.isfinite(a)
        if not bool(fin.all()):
            bad = True
            a = torch.where(fin, a, torch.zeros((), dtype=a.dtype, device=a.device))
        mx = max(mx, float(a.max().item()))
    return mx, bad


class MaxAbsProfiler:
    """forward 훅으로 입·출력 |x| 최댓값을 누적한다. 출력은 바꾸지 않는다."""

    def __init__(self, model, names=None):
        self.model = model
        mods = dict(model.named_modules())
        if names is None:
            names = [n for n, m in mods.items() if n and not any(True for _ in m.children())]
        else:
            names = list(names)
            for n in names:
                if n not in mods or not n:
                    raise RescuePlanError(f"프로파일러: 모듈 없음 {n!r}")
        self.names = names
        self.out_max: dict = {}
        self.in_max: dict = {}
        self.nonfinite: dict = {}
        self.calls: dict = {}
        self._handles: list = []
        self._pending: dict = {}

    def __enter__(self):
        mods = dict(self.model.named_modules())
        for n in self.names:
            self._handles.append(mods[n].register_forward_pre_hook(self._pre(n)))
            self._handles.append(mods[n].register_forward_hook(self._post(n)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _pre(self, name):
        def hook(_mod, args):
            first = next(_first_float_iter(args), None)
            mx, bad = _stat([first] if first is not None else [])
            self.in_max[name] = max(self.in_max.get(name, 0.0), mx)
            self._pending.setdefault(name, []).append(bad)
        return hook

    def _post(self, name):
        def hook(_mod, _args, out):
            mx, bad = _stat(_float_tensors(out))
            stack = self._pending.get(name)
            in_bad = stack.pop() if stack else False
            self.out_max[name] = max(self.out_max.get(name, 0.0), mx)
            self.nonfinite[name] = self.nonfinite.get(name, 0) + int(bad or in_bad)
            self.calls[name] = self.calls.get(name, 0) + 1
        return hook


def _first_float_iter(args):
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            yield a
            return


def choose_scale(max_abs, *, headroom=8.0, fp16_max=FP16_MAX, max_s=2.0 ** 16) -> float:
    """max_abs/s <= fp16_max/headroom 을 만족하는 가장 작은 2의 거듭제곱 s >= 1."""
    if not (isinstance(headroom, (int, float)) and headroom > 0 and math.isfinite(headroom)):
        raise RescuePlanError(f"headroom은 양의 유한수여야 한다: {headroom!r}")
    if isinstance(max_abs, Mapping):
        vals = list(max_abs.values())
    else:
        vals = [max_abs]
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise RescueError(f"max_abs는 유한 비음수여야 한다: {v!r}")
    m = float(max(vals)) if vals else 0.0
    limit = float(fp16_max) / float(headroom)
    s = 1.0
    while m / s > limit:
        s *= 2.0
        if s > max_s:
            raise RescueError(f"필요한 s가 max_s={max_s}를 넘는다 (max_abs={m})")
    return s


# ---------------------------------------------------------------------------
# NaN 감사
# ---------------------------------------------------------------------------

def upper95(k, n) -> float:
    """단측 95% Clopper-Pearson 상한."""
    if n < 1 or k < 0 or k > n:
        raise ValueError(f"upper95: n>=1, 0<=k<=n 이어야 한다 (k={k}, n={n})")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - 0.05 ** (1.0 / n)
    from scipy import stats
    return float(stats.beta.ppf(0.95, k + 1, n - k))


@dataclass(frozen=True)
class AuditReport:
    n_images: int
    n_nonfinite: int
    n_black: int
    upper95_fail: float
    ok: bool


def audit_images(images, *, black_thresh=0.02) -> AuditReport:
    """이미지([B,3,H,W] 또는 [3,H,W], 범위 [-1,1])의 비유한/검은 화면 수를 센다."""
    n = nf = bl = 0
    for img in images:
        x = torch.as_tensor(img)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"이미지는 [B,3,H,W] 또는 [3,H,W]여야 한다: {tuple(x.shape)}")
        x = x.detach().float()
        for i in range(x.shape[0]):
            xi = x[i]
            n += 1
            if not bool(torch.isfinite(xi).all()):
                nf += 1
            elif float(((xi + 1.0) / 2.0).max().item()) <= black_thresh:
                bl += 1
    fail = nf + bl
    up = upper95(fail, n) if n > 0 else 1.0
    return AuditReport(n_images=n, n_nonfinite=nf, n_black=bl, upper95_fail=up, ok=(n > 0 and fail == 0))


# ---------------------------------------------------------------------------
# Z-Image 계획 (diffusers ZImageTransformer2DModel 이름 규약; 실물 정합은 GPU 테스트만 확인)
# ---------------------------------------------------------------------------

_BLOCK_GROUPS = ("noise_refiner", "context_refiner", "layers")


def zimage_plan(transformer, s, *, swiglu_scales=None, islands=True) -> RescuePlan:
    mods = dict(transformer.named_modules())
    layers = getattr(transformer, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        raise RescuePlanError("Z-Image 구조가 아니다: ModuleList 'layers'가 없다")

    def need(name, kind=None):
        m = mods.get(name)
        if m is None:
            raise RescuePlanError(f"Z-Image 구조 불일치: {name!r} 없음")
        if kind is not None and not isinstance(m, kind):
            raise RescuePlanError(f"Z-Image 구조 불일치: {name!r}는 {kind.__name__}가 아니다")
        return m

    # 임베더: x(패치) 임베딩과 캡션 임베딩의 Linear
    embed = []
    if isinstance(mods.get("x_embedder"), nn.Linear):
        embed.append("x_embedder")
    else:
        embed += [n for n, m in mods.items() if n.startswith("all_x_embedder.") and n.count(".") == 1
                  and isinstance(m, nn.Linear)]
    if not embed:
        raise RescuePlanError("Z-Image 구조 불일치: x_embedder(Linear) 없음")
    cap = mods.get("cap_embedder")
    if isinstance(cap, nn.Linear):
        embed.append("cap_embedder")
    elif isinstance(cap, nn.Sequential):
        lin_idx = [n for n, m in cap.named_children() if isinstance(m, nn.Linear)]
        if not lin_idx:
            raise RescuePlanError("Z-Image 구조 불일치: cap_embedder에 Linear 없음")
        embed.append(f"cap_embedder.{lin_idx[-1]}")
    else:
        raise RescuePlanError("Z-Image 구조 불일치: cap_embedder 없음")

    params = dict(transformer.named_parameters())
    stream_params = tuple(n for n in ("x_pad_token", "cap_pad_token") if n in params)

    residual_out, stream_norms, downs = [], [], []
    norm_islands = []
    sites = []
    for grp in _BLOCK_GROUPS:
        blocks = getattr(transformer, grp, None)
        if not isinstance(blocks, nn.ModuleList):
            continue
        for i in range(len(blocks)):
            b = f"{grp}.{i}"
            for n in ("attention_norm1", "ffn_norm1"):
                need(f"{b}.{n}")
                stream_norms.append(f"{b}.{n}")
                norm_islands.append(f"{b}.{n}")
            for n in ("attention_norm2", "ffn_norm2"):
                need(f"{b}.{n}")
                if not _is_norm_like(mods[f"{b}.{n}"]):
                    raise RescuePlanError(f"Z-Image 구조 불일치: {b}.{n}에 weight/eps 없음")
                residual_out.append(f"{b}.{n}")
                norm_islands.append(f"{b}.{n}")
            up, gate, down = (need(f"{b}.feed_forward.{w}", nn.Linear) and f"{b}.feed_forward.{w}"
                              for w in ("w3", "w1", "w2"))
            sites.append((up, gate, down))
            downs.append(down)
            for qk in ("attention.norm_q", "attention.norm_k"):
                if f"{b}.{qk}" in mods and _is_norm_like(mods[f"{b}.{qk}"]):
                    norm_islands.append(f"{b}.{qk}")

    # 최종 층: norm_final(스트림을 읽음) 또는 norm 없이 읽는 Linear
    unscale, finals = [], []
    fl_names = [n for n in mods if n == "final_layer" or (n.startswith("all_final_layer.") and n.count(".") == 1)]
    if not fl_names:
        raise RescuePlanError("Z-Image 구조 불일치: final_layer 없음")
    for fl in fl_names:
        if f"{fl}.norm_final" in mods and _eps_attr(_unwrap(mods[f"{fl}.norm_final"])) is not None:
            finals.append(f"{fl}.norm_final")
            norm_islands.append(f"{fl}.norm_final")
        elif isinstance(mods.get(f"{fl}.linear"), nn.Linear):
            unscale.append(f"{fl}.linear")
        else:
            raise RescuePlanError(f"Z-Image 구조 불일치: {fl}의 norm_final/linear 없음")
    stream_norms += finals

    # 캡션 임베더의 norm도 FP32 섬(스트림은 아님: 원시 캡션을 읽는다)
    if isinstance(cap, nn.Sequential):
        for n, m in cap.named_children():
            if not isinstance(m, nn.Linear) and _is_norm_like(m):
                norm_islands.append(f"cap_embedder.{n}")

    def t_for(down):
        if swiglu_scales is None:
            return 1.0
        if isinstance(swiglu_scales, Mapping):
            return float(swiglu_scales.get(down, 1.0))
        return float(swiglu_scales)

    swiglu = tuple(SwiGLUSite(up=u, gate=g, down=d, scale=t_for(d)) for u, g, d in sites)
    plan = RescuePlan(
        s=s, embed=tuple(embed), stream_params=stream_params, residual_out=tuple(residual_out),
        stream_norms=tuple(stream_norms), unscale=tuple(unscale), swiglu=swiglu,
        fp32_islands=tuple(dict.fromkeys(norm_islands)) if islands else ())
    return plan.validate()
