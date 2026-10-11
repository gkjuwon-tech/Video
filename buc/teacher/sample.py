"""TCH-02 교사 샘플러 v0: sigma 스냅 표, SDEdit, x0 예측 API, 뷰 배치 lockstep.

계약서: ops/harness/contracts/TCH-02.md. 공개 API는 계약서와 1:1이다.
- 모든 교사(Z-Image-Turbo) 호출의 단일 입구. 학습된 sigma 점에서만 믿을 수 있으므로 임의 sigma 요청은 표의 점으로 스냅한다.
- 수식은 flow matching: z_s = (1-s) x0 + s eps, v = eps - x0, x0_hat = z - s v. 변환기는 t = 1-s 를 받고 -v 를 낸다.
- import는 torch·numpy·buc.core.det·buc.teacher.zimage 뿐이며 diffusers 없이 되고 부작용이 없다.
"""
from __future__ import annotations

import contextlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from buc.core import det
from buc.teacher.zimage import TeacherError, TeacherNumericError, TeacherSpecError

__all__ = [
    "FLOW_SHIFT", "N_STEPS", "SigmaError", "SigmaTable", "Snap", "DEFAULT_TABLE", "snap_sigma", "add_noise",
    "euler_step", "StepInfo", "SampleResult", "TeacherSampler",
]

FLOW_SHIFT = 3.0
N_STEPS = 8
_SEED_LIMIT = 1 << 64


class SigmaError(TeacherSpecError):
    """sigma 값·표·스케줄이 잘못됐다(ValueError 계열)."""


# --------------------------------------------------------------------------- 작은 검증 도우미

def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _as_float(v, what: str) -> float:
    """실수 스칼라를 python float으로. 아니면 SigmaError."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise SigmaError(f"{what}는 실수여야 한다: {v!r}")
    return float(v)


def _as_sigma(v, what: str = "sigma") -> float:
    """유한이고 (0, 1] 인 sigma."""
    s = _as_float(v, what)
    if not math.isfinite(s) or not (0.0 < s <= 1.0):
        raise SigmaError(f"{what}는 (0, 1] 의 유한수여야 한다: {v!r}")
    return s


# --------------------------------------------------------------------------- sigma 표

@dataclass(frozen=True)
class Snap:
    index: int
    sigma: float
    requested: float


@dataclass(frozen=True)
class SigmaTable:
    points: tuple
    name: str = "custom"

    def __post_init__(self):
        try:
            pts = tuple(self.points)
        except TypeError as e:
            raise SigmaError(f"points는 실수 열이어야 한다: {self.points!r}") from e
        if len(pts) < 1:
            raise SigmaError("표는 점이 1개 이상이어야 한다")
        fl = tuple(_as_sigma(p, "표의 점") for p in pts)
        for a, b in zip(fl, fl[1:]):
            if not a > b:
                raise SigmaError(f"표의 점은 엄격 내림차순이어야 한다: {fl}")
        if not isinstance(self.name, str):
            raise SigmaError(f"name은 str이어야 한다: {self.name!r}")
        object.__setattr__(self, "points", fl)

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def from_flow_shift(cls, n: int = N_STEPS, shift: float = FLOW_SHIFT, name: Optional[str] = None) -> "SigmaTable":
        if not _is_int(n) or n < 1:
            raise SigmaError(f"n은 양의 정수여야 한다: {n!r}")
        sh = _as_float(shift, "shift")
        if not math.isfinite(sh) or sh <= 0.0:
            raise SigmaError(f"shift는 양의 유한수여야 한다: {shift!r}")
        n = int(n)
        pts = []
        for i in range(n):
            u = 1.0 - i / n
            pts.append(sh * u / (1.0 + (sh - 1.0) * u))
        return cls(tuple(pts), name=f"flow_shift(n={n},shift={sh!r})" if name is None else name)

    @property
    def key(self) -> str:
        return det.hash_parts("buc.tch02.table.v1", self.name, *[struct.pack("<d", p) for p in self.points]).hex()

    def snap(self, sigma) -> Snap:
        """|sigma - 점| 이 최소인 점. 동률은 큰 sigma(앞 인덱스)."""
        s = _as_sigma(sigma)
        best, bd = 0, abs(self.points[0] - s)
        for j in range(1, len(self.points)):
            d = abs(self.points[j] - s)
            if d < bd:
                best, bd = j, d
        return Snap(index=int(best), sigma=float(self.points[best]), requested=s)

    def index_of(self, sigma, tol: float = 1e-9) -> int:
        """표의 점에 (tol 이내로) 정확히 있는 sigma 의 인덱스. 아니면 SigmaError."""
        s = _as_float(sigma, "sigma")
        if not math.isfinite(s):
            raise SigmaError(f"sigma가 유한하지 않다: {sigma!r}")
        best, bd = 0, abs(self.points[0] - s)
        for j in range(1, len(self.points)):
            d = abs(self.points[j] - s)
            if d < bd:
                best, bd = j, d
        if not bd <= tol:
            raise SigmaError(f"sigma={sigma!r}는 표의 점이 아니다(표: {self.points})")
        return int(best)

    def window(self, lo, hi) -> tuple:
        """lo <= sigma <= hi 인 점의 인덱스(내림차순). 비면 중앙값의 스냅 1개."""
        lo_f, hi_f = _as_float(lo, "lo"), _as_float(hi, "hi")
        if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
            raise SigmaError(f"창의 끝이 유한하지 않다: {lo!r}, {hi!r}")
        if lo_f > hi_f:
            raise SigmaError(f"lo <= hi 여야 한다: {lo!r} > {hi!r}")
        idx = tuple(i for i, p in enumerate(self.points) if lo_f <= p <= hi_f)
        if idx:
            return idx
        return (self.snap(0.5 * (lo_f + hi_f)).index,)

    def schedule(self, start, steps) -> tuple:
        """표의 연속 steps 점에 대한 (sigma, sigma_next) 쌍. 표 끝을 넘어가는 sigma_next 는 0.0."""
        if not _is_int(start) or not _is_int(steps):
            raise SigmaError(f"start, steps는 정수여야 한다: {start!r}, {steps!r}")
        start, steps = int(start), int(steps)
        n = len(self.points)
        if steps < 1 or start < 0 or start >= n or start + steps > n:
            raise SigmaError(f"스케줄이 표(길이 {n})를 벗어난다: start={start}, steps={steps}")
        out = []
        for i in range(start, start + steps):
            nxt = self.points[i + 1] if i + 1 < n else 0.0
            out.append((self.points[i], float(nxt)))
        return tuple(out)


DEFAULT_TABLE = SigmaTable.from_flow_shift(N_STEPS, FLOW_SHIFT, name="zimage-turbo-shift3-n8")


def snap_sigma(sigma, table: Optional[SigmaTable] = None) -> Snap:
    return (DEFAULT_TABLE if table is None else table).snap(sigma)


# --------------------------------------------------------------------------- 수식 도우미

def add_noise(x0: torch.Tensor, sigma, noise: torch.Tensor) -> torch.Tensor:
    """z_sigma = (1-sigma) x0 + sigma noise, sigma in [0,1]. float32 산술·출력."""
    s = _as_float(sigma, "sigma")
    if not math.isfinite(s) or not (0.0 <= s <= 1.0):
        raise SigmaError(f"sigma는 [0, 1] 의 유한수여야 한다: {sigma!r}")
    if tuple(x0.shape) != tuple(noise.shape):
        raise ValueError(f"x0 {tuple(x0.shape)} 와 noise {tuple(noise.shape)} 의 모양이 다르다")
    return (1.0 - s) * x0.to(torch.float32) + s * noise.to(torch.float32)


def euler_step(z: torch.Tensor, x0: torch.Tensor, sigma, sigma_next) -> torch.Tensor:
    """z' = (sn/s) z + (1 - sn/s) x0. 0 < s <= 1, 0 <= sn < s. sn == 0 이면 x0 와 비트 동일."""
    s = _as_float(sigma, "sigma")
    sn = _as_float(sigma_next, "sigma_next")
    if not (math.isfinite(s) and math.isfinite(sn)) or not (0.0 < s <= 1.0) or not (0.0 <= sn < s):
        raise SigmaError(f"0 < sigma <= 1 이고 0 <= sigma_next < sigma 여야 한다: {sigma!r}, {sigma_next!r}")
    if tuple(z.shape) != tuple(x0.shape):
        raise ValueError(f"z {tuple(z.shape)} 와 x0 {tuple(x0.shape)} 의 모양이 다르다")
    if sn == 0.0:
        return x0.to(torch.float32, copy=True)
    r = sn / s
    return r * z.to(torch.float32) + (1.0 - r) * x0.to(torch.float32)


# --------------------------------------------------------------------------- 결과 형식

@dataclass(frozen=True)
class StepInfo:
    step: int
    sigma: float
    sigma_next: float
    view_lo: int
    view_hi: int


@dataclass(frozen=True)
class SampleResult:
    x0_hat: torch.Tensor
    z: torch.Tensor
    z_start: torch.Tensor
    sigmas: tuple
    sigma_end: float
    start_index: int
    requested_sigma: float
    steps: int
    calls: int
    table_key: str


# --------------------------------------------------------------------------- 샘플러

class TeacherSampler:
    def __init__(self, teacher, *, table: Optional[SigmaTable] = None, max_views_per_call: Optional[int] = None):
        if table is None:
            table = DEFAULT_TABLE
        if not isinstance(table, SigmaTable):
            raise TypeError(f"table은 SigmaTable이어야 한다: {type(table).__name__}")
        if max_views_per_call is not None and (not _is_int(max_views_per_call) or max_views_per_call < 1):
            raise ValueError(f"max_views_per_call은 None 또는 양의 정수여야 한다: {max_views_per_call!r}")
        self.teacher = teacher
        self.table = table
        self.max_views_per_call = None if max_views_per_call is None else int(max_views_per_call)
        self.numeric_events: list = []      # 비유한 출력이 환경 재시도로 고쳐진 기록(최근 64개)

    # -- 입력 검증 ---------------------------------------------------------
    def _check_latent(self, z, name: str) -> torch.Tensor:
        spec = self.teacher.spec
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"{name}는 텐서여야 한다: {type(z).__name__}")
        if z.ndim != 4:
            raise ValueError(f"{name}는 [B,C,h,w] 4차원이어야 한다: {tuple(z.shape)}")
        if not z.is_floating_point():
            raise TypeError(f"{name}는 부동소수 텐서여야 한다: {z.dtype}")
        b, c, h, w = z.shape
        if b < 1:
            raise ValueError(f"{name}의 배치는 1 이상이어야 한다")
        if c != spec.latent_channels:
            raise ValueError(f"{name}의 채널 {c} != latent_channels {spec.latent_channels}")
        spec.check_image_size(spec.vae_scale * h, spec.vae_scale * w)
        if not bool(torch.isfinite(z.detach()).all()):
            raise ValueError(f"{name}에 NaN/Inf가 있다")
        return z

    def _resolve_cond(self, cond, b: int) -> list:
        """cond -> 길이 b 의 텐서 리스트 [L_i, D] (teacher dtype/device)."""
        if isinstance(cond, (str, bytes, torch.Tensor)) or not isinstance(cond, Sequence):
            raise TypeError("cond는 str 시퀀스 또는 텐서[L,D] 시퀀스여야 한다")
        items = list(cond)
        if not items:
            raise ValueError("cond가 비었다")
        n_str = sum(isinstance(c, str) for c in items)
        n_ten = sum(isinstance(c, torch.Tensor) for c in items)
        if n_str + n_ten != len(items):
            raise TypeError("cond의 원소는 str 또는 텐서여야 한다")
        if n_str and n_ten:
            raise TypeError("cond에 str과 텐서를 섞을 수 없다")
        if len(items) != b and len(items) != 1:
            raise ValueError(f"cond 길이 {len(items)}는 배치 {b} 또는 1이어야 한다")
        t = self.teacher
        if n_str:
            caps = list(t.encode_text(items))
        else:
            d = t.spec.text_dim
            caps = []
            for c in items:
                if c.ndim != 2 or c.shape[0] < 1 or c.shape[-1] != d:
                    raise ValueError(f"조건 임베딩은 [L,{d}] 여야 한다: {tuple(c.shape)}")
                if not c.is_floating_point() or not bool(torch.isfinite(c.detach()).all()):
                    raise ValueError("조건 임베딩은 유한한 부동소수여야 한다")
                caps.append(c.to(device=t.device, dtype=t.dtype))
        if len(caps) != len(items):
            raise TeacherError(f"텍스트 인코더 출력 개수 {len(caps)} != {len(items)}")
        if len(caps) == 1 and b != 1:
            caps = caps * b
        return caps

    @staticmethod
    def _check_seed(s) -> int:
        if not _is_int(s):
            raise TypeError(f"시드는 정수여야 한다: {s!r}")
        s = int(s)
        if not (0 <= s < _SEED_LIMIT):
            raise ValueError(f"시드가 [0, 2^64) 밖이다: {s}")
        return s

    def _resolve_seeds(self, seeds, b: int) -> list:
        if _is_int(seeds):
            base = self._check_seed(seeds)
            return [det.derive_seed(base, i) for i in range(b)]
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (Sequence, np.ndarray)):
            raise TypeError("seeds는 정수 또는 정수 시퀀스여야 한다")
        out = [self._check_seed(s) for s in seeds]
        if len(out) != b:
            raise ValueError(f"seeds 길이 {len(out)} != 배치 {b}")
        return out

    @staticmethod
    def _check_callable(fn, name: str) -> None:
        if fn is not None and not callable(fn):
            raise TypeError(f"{name}는 호출 가능해야 한다")

    def _noise(self, seeds: list, h: int, w: int, level, k) -> torch.Tensor:
        vs = self.teacher.spec.vae_scale
        eps = [self.teacher.make_noise(s, 1, vs * h, vs * w, level=level, k=k)[0] for s in seeds]
        return torch.stack(eps).to(torch.float32)

    # -- 변환기 호출 -------------------------------------------------------
    def _chunks(self, b: int) -> list:
        m = self.max_views_per_call or b
        return [(lo, min(lo + m, b)) for lo in range(0, b, m)]

    def _transformer_v(self, x_list: list, t: torch.Tensor, caps_chunk: list, n: int, want: tuple) -> torch.Tensor:
        """변환기 한 번 호출 -> 속도 v [n,C,h,w] float32 (= -out). 개수·모양 검증만 하고 유한성은 보지 않는다."""
        out = self.teacher.transformer(x_list, t, caps_chunk, return_dict=False)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            out = list(out.unbind(0)) if out.ndim == 5 else out
        if not isinstance(out, (list, tuple)) or len(out) != n:
            raise TeacherError(f"변환기 출력 개수가 입력 {n}과 다르다")
        vs = []
        for o in out:
            if not isinstance(o, torch.Tensor) or tuple(o.shape) != (want[0], 1, want[1], want[2]):
                raise TeacherError(f"변환기 출력 모양이 [C,1,h,w]={(want[0], 1, want[1], want[2])} 가 아니다")
            vs.append(o.to(torch.float32).squeeze(1))
        return -torch.stack(vs)

    def _forward_step(self, z: torch.Tensor, caps: list, step: int, sigma: float, sigma_next: float,
                      observer) -> tuple:
        """z[B,C,h,w] float32 에서 x0_hat[B,C,h,w] float32 를 구한다. (x0_hat, 호출 수)."""
        dev = z.device
        outs = []
        calls = 0
        for lo, hi in self._chunks(z.shape[0]):
            if observer is not None:
                observer(StepInfo(step=int(step), sigma=float(sigma), sigma_next=float(sigma_next),
                                  view_lo=int(lo), view_hi=int(hi)))
            n = hi - lo
            zc = z[lo:hi]
            x_list = [zi.to(dtype=self.teacher.dtype).unsqueeze(1) for zi in zc]   # [C,1,h,w]
            t = torch.full((n,), 1.0 - sigma, dtype=torch.float32, device=dev)
            want = tuple(zc.shape[1:])
            v = self._transformer_v(x_list, t, caps[lo:hi], n, want)
            calls += 1
            if not bool(torch.isfinite(v.detach()).all()):
                v = self._recover_nonfinite(v, x_list, t, caps[lo:hi], n, want, step, sigma, lo, hi)
            outs.append(zc - sigma * v)
        x0_hat = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        return x0_hat, calls

    # -- 비유한 출력: 진단과 환경 재시도 -----------------------------------------
    #: 변환기 출력이 비유한이면 같은 입력으로 (프로브 -> 환경 완화 누적) 다시 불러 본다. 입력·가중치는 건드리지 않는다.
    nonfinite_retry = True

    @staticmethod
    @contextlib.contextmanager
    def _env_fill_off():
        prev = torch.utils.deterministic.fill_uninitialized_memory
        torch.utils.deterministic.fill_uninitialized_memory = False
        try:
            yield
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = prev

    @staticmethod
    @contextlib.contextmanager
    def _env_math_sdpa():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            yield

    @staticmethod
    @contextlib.contextmanager
    def _env_default_gemm():
        m = torch.backends.cuda.matmul
        prev = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
        m.allow_bf16_reduced_precision_reduction = True
        m.allow_fp16_reduced_precision_reduction = True
        try:
            yield
        finally:
            m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = prev

    def _retry_ladder(self) -> list:
        """[(이름, 이 단계에서 누적으로 켜는 컨텍스트 팩토리 목록)]. 첫 단계(probe)는 설정을 바꾸지 않는다."""
        ladder = [("probe", [])]
        mitig = []
        if torch.are_deterministic_algorithms_enabled() and getattr(torch.utils.deterministic, "fill_uninitialized_memory", False):
            mitig.append(self._env_fill_off)
            ladder.append(("no_nan_fill", list(mitig)))
        mitig.append(self._env_math_sdpa)
        ladder.append(("math_sdpa", list(mitig)))
        m = torch.backends.cuda.matmul
        if not (m.allow_bf16_reduced_precision_reduction and m.allow_fp16_reduced_precision_reduction):
            mitig.append(self._env_default_gemm)
            ladder.append(("default_gemm", list(mitig)))
        return ladder

    @staticmethod
    def _first_nonfinite_module(model: torch.nn.Module, fn: Callable) -> tuple:
        """fn() 을 모든 하위 모듈 출력 훅과 함께 실행해 처음으로 비유한 출력을 낸 모듈 이름을 찾는다. (fn 결과, 이름|None)."""
        first: list = []

        def bad(o) -> bool:
            if isinstance(o, torch.Tensor):
                return o.is_floating_point() and not bool(torch.isfinite(o.detach()).all())
            if isinstance(o, (list, tuple)):
                return any(bad(e) for e in o)
            return False

        def mk(name):
            def hook(_m, _a, out):
                if not first and bad(out):
                    first.append(name)
            return hook

        handles = [m.register_forward_hook(mk(nm)) for nm, m in model.named_modules() if nm]
        try:
            res = fn()
        finally:
            for h in handles:
                h.remove()
        return res, (first[0] if first else None)

    def _recover_nonfinite(self, v, x_list, t, caps_chunk, n, want, step, sigma, lo, hi) -> torch.Tensor:
        """비유한 v 를 받았을 때. 환경(프로브 -> 포이즌 끔 -> math SDPA -> 기본 GEMM) 완화를 누적해 재시도하고, 끝내 안 되면
        무슨 일이 있었는지 한 줄에 담아 TeacherNumericError. 고쳐지면 self.numeric_events 에 남기고 그 v 를 돌려준다."""
        fin = torch.isfinite(v.detach()).flatten(1).all(dim=1)
        bad_rows = [int(i) for i in range(n) if not bool(fin[i])]
        absmax = lambda ts: max([float(a.detach().float().abs().max()) for a in ts if a.numel()] or [0.0])  # noqa: E731
        x_ok = all(bool(torch.isfinite(a.detach().float()).all()) for a in x_list)
        c_ok = all(bool(torch.isfinite(c.detach().float()).all()) for c in caps_chunk)
        head = (f"변환기 출력에 NaN/Inf가 있다 | step={step} sigma={sigma:.4g} views={lo}:{hi} "
                f"dtype={str(self.teacher.dtype).replace('torch.', '')} dev={t.device} bad_rows={bad_rows} "
                f"x_finite={x_ok} x_absmax={absmax(x_list):.3g} cap_finite={c_ok} cap_absmax={absmax(caps_chunk):.3g}")
        tried, first_bad = [], None
        if self.nonfinite_retry:
            model = self.teacher.transformer
            for name, makers in self._retry_ladder():
                try:
                    with contextlib.ExitStack() as st:
                        for mk in makers:
                            st.enter_context(mk())
                        if name == "probe":
                            v2, first_bad = self._first_nonfinite_module(
                                model, lambda: self._transformer_v(x_list, t, caps_chunk, n, want))
                        else:
                            v2 = self._transformer_v(x_list, t, caps_chunk, n, want)
                except Exception as e:  # noqa: BLE001  재시도 자체의 실패(OOM, 훅 중복 발화 등)는 진단에만 쓴다
                    tried.append(f"{name}:err({type(e).__name__})")
                    continue
                if bool(torch.isfinite(v2.detach()).all()):
                    self.numeric_events.append({"step": int(step), "sigma": float(sigma), "views": (int(lo), int(hi)),
                                                "healed_by": name, "first_bad": first_bad, "tried": list(tried)})
                    del self.numeric_events[:-64]
                    return v2
                tried.append(f"{name}:nonfinite")
        raise TeacherNumericError(f"{head} first_bad={first_bad} tried={','.join(tried) or 'none'}")

    def _run(self, z: torch.Tensor, caps: list, start_index: int, steps: int, x0_hook, observer, grad: bool,
             requested: float) -> SampleResult:
        pairs = self.table.schedule(start_index, steps)
        zc = z.detach() if not grad else z
        zc = zc.clone()
        z_start = zc
        b = zc.shape[0]
        x0_raw = None
        calls = 0
        sigmas = []
        for step, (s, sn) in enumerate(pairs):
            x0_hat, c = self._forward_step(zc, caps, step, s, sn, observer)
            calls += c
            sigmas.append(float(s))
            x0_raw = x0_hat.clone() if x0_hook is not None else x0_hat
            x0p = x0_hat
            if x0_hook is not None:
                info = StepInfo(step=int(step), sigma=float(s), sigma_next=float(sn), view_lo=0, view_hi=int(b))
                x0p = x0_hook(x0_hat, info)
                if not isinstance(x0p, torch.Tensor):
                    raise TypeError("x0_hook은 텐서를 돌려줘야 한다")
                if tuple(x0p.shape) != tuple(x0_hat.shape):
                    raise ValueError(f"x0_hook 반환 모양 {tuple(x0p.shape)} != {tuple(x0_hat.shape)}")
                x0p = x0p.to(device=x0_hat.device, dtype=torch.float32)
                if not bool(torch.isfinite(x0p.detach()).all()):
                    raise ValueError("x0_hook 반환값에 NaN/Inf가 있다")
            zc = euler_step(zc, x0p, s, sn)
        return SampleResult(x0_hat=x0_raw, z=zc, z_start=z_start, sigmas=tuple(sigmas),
                            sigma_end=float(pairs[-1][1]), start_index=int(start_index), requested_sigma=float(requested),
                            steps=len(pairs), calls=int(calls), table_key=self.table.key)

    def _to_state(self, z: torch.Tensor) -> torch.Tensor:
        return z.to(device=self.teacher.device, dtype=torch.float32)

    # -- 공개 API ----------------------------------------------------------
    def predict_x0(self, z, sigma, cond, *, grad: bool = False, observer: Optional[Callable] = None) -> torch.Tensor:
        """표의 점에 정확히 있는 sigma 에서 forward 1회(청크 단위)로 x0_hat [B,C,h,w] float32."""
        idx = self.table.index_of(sigma)
        self._check_latent(z, "z")
        self._check_callable(observer, "observer")
        caps = self._resolve_cond(cond, z.shape[0])
        s = self.table.points[idx]
        sn = self.table.points[idx + 1] if idx + 1 < len(self.table) else 0.0
        with torch.set_grad_enabled(bool(grad)):
            zc = self._to_state(z)
            if not grad:
                zc = zc.detach()
            x0_hat, _ = self._forward_step(zc, caps, 0, s, float(sn), observer)
        return x0_hat

    def denoise(self, z, cond, *, start_index, steps, x0_hook: Optional[Callable] = None,
                observer: Optional[Callable] = None, grad: bool = False) -> SampleResult:
        """z 는 이미 sigma = table[start_index] 상태. 표의 연속 steps 점을 따라 내려간다."""
        self._check_latent(z, "z")
        self._check_callable(x0_hook, "x0_hook")
        self._check_callable(observer, "observer")
        pairs = self.table.schedule(start_index, steps)               # 범위 검증
        caps = self._resolve_cond(cond, z.shape[0])
        with torch.set_grad_enabled(bool(grad)):
            return self._run(self._to_state(z), caps, int(start_index), len(pairs), x0_hook, observer, bool(grad),
                             self.table.points[int(start_index)])

    def sdedit(self, x0, cond, *, sigma, seeds, steps: int = 2, level: int = 0, k: int = 0,
               x0_hook: Optional[Callable] = None, observer: Optional[Callable] = None,
               grad: bool = False) -> SampleResult:
        """sigma 를 스냅 -> z_start = add_noise(x0, sigma_s, eps) -> 표의 연속 steps 점을 denoise."""
        sn = self.table.snap(sigma)
        self._check_latent(x0, "x0")
        self._check_callable(x0_hook, "x0_hook")
        self._check_callable(observer, "observer")
        self.table.schedule(sn.index, steps)                           # 범위 검증
        b, _, h, w = x0.shape
        caps = self._resolve_cond(cond, b)
        seed_list = self._resolve_seeds(seeds, b)
        with torch.set_grad_enabled(bool(grad)):
            x = self._to_state(x0)
            if not grad:
                x = x.detach()
            eps = self._noise(seed_list, h, w, level, k).to(x.device)
            z_start = add_noise(x, sn.sigma, eps)
            return self._run(z_start, caps, sn.index, int(steps), x0_hook, observer, bool(grad), sn.requested)

    def generate(self, cond, *, seeds, height, width, steps: Optional[int] = None, level: int = 0, k: int = 0,
                 **kw) -> SampleResult:
        """순수 노이즈(sigma=1)에서 표 전체(기본 8 NFE). sdedit(sigma=1.0)과 비트 동일."""
        if isinstance(seeds, (Sequence, np.ndarray)) and not isinstance(seeds, (str, bytes)):
            b = len(seeds)
        else:
            if isinstance(cond, (str, bytes, torch.Tensor)) or not isinstance(cond, Sequence):
                raise TypeError("cond는 str 시퀀스 또는 텐서[L,D] 시퀀스여야 한다")
            b = len(cond)
        shape = self.teacher.spec.latent_shape(height, width, b)      # 모양 검증(ValueError)
        x0 = torch.zeros(shape, dtype=torch.float32, device=self.teacher.device)
        return self.sdedit(x0, cond, sigma=self.table.points[0], seeds=seeds,
                           steps=len(self.table) if steps is None else steps, level=level, k=k, **kw)
