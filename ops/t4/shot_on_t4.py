"""RUN-08: "Shot on T4" 실측 부록. 4K 프레임 시간·메모리를 재고 한국어 부록(.md)과 보고서(.json)를 쓴다.

규칙: T4에서 재지 않은 수치는 T4 수치로 적지 않는다. 못 재면 "미측정".

import 시점에는 표준 라이브러리만 쓴다. torch·numpy·buc.*는 함수 안에서 지연 import한다.
직접 실행: python3 ops/t4/shot_on_t4.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import numbers
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RES_4K = (3840, 2160)
GATE_PRIMS = 3_000_000
GATE_4K_S = 0.6
GIB = 2**30

STATUSES = ("ok", "oom", "error", "unmeasured")
VERDICTS = ("pass", "fail", "unmeasured", "info")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = Path("work") / "t4" / "shot_on_t4.json"


# ---------------------------------------------------------------- 지연 import 도우미
def _ensure_repo_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_torch():
    import torch  # noqa: PLC0415

    return torch


# ---------------------------------------------------------------- 이름·비밀·OOM
_T4_RE = re.compile(r"(?<![0-9A-Za-z])T4(?![0-9A-Za-z])", re.IGNORECASE)


def is_t4_name(name: str) -> bool:
    """앞뒤가 영숫자가 아닌 독립 토큰 "T4"가 있으면 True (대소문자 무시)."""
    if not isinstance(name, str):
        return False
    return _T4_RE.search(name) is not None


# RUN-07 ops/report/hourly.py redact와 같은 규칙.
_TOKEN_RES = (
    re.compile(r"hf_[A-Za-z0-9_]{4,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{8,}"),
)
_KV_RE = re.compile(r"(api_key|token|secret|password)\s*[=:]\s*\S+", re.IGNORECASE)
_LONG_RE = re.compile(r"[A-Za-z0-9]{32,}")


def redact(text) -> str:
    out = str(text)
    for rx in _TOKEN_RES:
        out = rx.sub("***", out)
    out = _KV_RE.sub(lambda m: f"{m.group(1)}=***", out)
    return _LONG_RE.sub("***", out)


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if any(c.__name__ == "OutOfMemoryError" for c in type(exc).__mro__):
        return True
    return "out of memory" in str(exc).lower()


# ---------------------------------------------------------------- 장치
@dataclass(frozen=True, kw_only=True)
class DeviceInfo:
    kind: str
    name: str
    total_mem_gib: float | None
    capability: tuple[int, int] | None
    torch_version: str
    cuda_version: str | None
    is_t4: bool

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "total_mem_gib": self.total_mem_gib,
            "capability": list(self.capability) if self.capability is not None else None,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "is_t4": self.is_t4,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "DeviceInfo":
        cap = d.get("capability")
        return cls(
            kind=d["kind"],
            name=d["name"],
            total_mem_gib=d["total_mem_gib"],
            capability=tuple(cap) if cap is not None else None,
            torch_version=d["torch_version"],
            cuda_version=d["cuda_version"],
            is_t4=bool(d["is_t4"]),
        )


def detect_device(torch_mod=None) -> DeviceInfo:
    if torch_mod is None:
        torch_mod = _import_torch()
    torch_version = str(getattr(torch_mod, "__version__", "unknown"))
    cuda_version = getattr(getattr(torch_mod, "version", None), "cuda", None)
    if not torch_mod.cuda.is_available():
        return DeviceInfo(kind="cpu", name="cpu", total_mem_gib=None, capability=None,
                          torch_version=torch_version, cuda_version=cuda_version, is_t4=False)
    name = str(torch_mod.cuda.get_device_name(0))
    total = float(torch_mod.cuda.get_device_properties(0).total_memory) / GIB
    cap = torch_mod.cuda.get_device_capability(0)
    capability = (int(cap[0]), int(cap[1])) if cap is not None else None
    return DeviceInfo(kind="cuda", name=name, total_mem_gib=total, capability=capability,
                      torch_version=torch_version, cuda_version=cuda_version, is_t4=is_t4_name(name))


def make_sync(device: DeviceInfo, torch_mod=None) -> Callable[[], None]:
    if device.kind == "cuda":
        if torch_mod is None:
            torch_mod = _import_torch()

        def _sync() -> None:
            torch_mod.cuda.synchronize()

        return _sync

    def _noop() -> None:
        return None

    return _noop


# ---------------------------------------------------------------- 시간 측정
@dataclass(frozen=True, kw_only=True)
class Timing:
    n: int
    warmup: int
    samples_s: tuple
    min_s: float
    median_s: float
    mean_s: float
    p95_s: float
    max_s: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "warmup": self.warmup,
            "samples_s": list(self.samples_s),
            "min_s": self.min_s,
            "median_s": self.median_s,
            "mean_s": self.mean_s,
            "p95_s": self.p95_s,
            "max_s": self.max_s,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Timing":
        return cls(
            n=int(d["n"]),
            warmup=int(d["warmup"]),
            samples_s=tuple(float(x) for x in d["samples_s"]),
            min_s=float(d["min_s"]),
            median_s=float(d["median_s"]),
            mean_s=float(d["mean_s"]),
            p95_s=float(d["p95_s"]),
            max_s=float(d["max_s"]),
        )


def _quantile(sorted_vals: list, q: float) -> float:
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def summarize_samples(samples, warmup: int = 0) -> Timing:
    vals: list[float] = []
    for x in samples:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise ValueError(f"표본은 실수여야 한다 ({x!r})")
        f = float(x)
        if not math.isfinite(f) or f < 0.0:
            raise ValueError(f"표본은 유한한 0 이상이어야 한다 ({x!r})")
        vals.append(f)
    if not vals:
        raise ValueError("표본이 비었다")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup 은 0 이상의 정수여야 한다")
    s = sorted(vals)
    return Timing(
        n=len(vals),
        warmup=warmup,
        samples_s=tuple(vals),
        min_s=s[0],
        median_s=_quantile(s, 0.5),
        mean_s=math.fsum(vals) / len(vals),
        p95_s=_quantile(s, 0.95),
        max_s=s[-1],
    )


def time_calls(fn, *, repeats: int = 5, warmup: int = 2, sync=None, clock=time.perf_counter) -> Timing:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats 는 1 이상의 정수여야 한다")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup 은 0 이상의 정수여야 한다")
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()
    samples = []
    for _ in range(repeats):
        if sync is not None:
            sync()
        t0 = clock()
        fn()
        if sync is not None:
            sync()
        t1 = clock()
        samples.append(t1 - t0)
    return summarize_samples(samples, warmup=warmup)


# ---------------------------------------------------------------- 측정 결과
@dataclass(frozen=True, kw_only=True)
class MemStats:
    baseline_gib: float
    peak_allocated_gib: float
    peak_reserved_gib: float

    def to_dict(self) -> dict:
        return {
            "baseline_gib": self.baseline_gib,
            "peak_allocated_gib": self.peak_allocated_gib,
            "peak_reserved_gib": self.peak_reserved_gib,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "MemStats":
        return cls(
            baseline_gib=float(d["baseline_gib"]),
            peak_allocated_gib=float(d["peak_allocated_gib"]),
            peak_reserved_gib=float(d["peak_reserved_gib"]),
        )


@dataclass(frozen=True, kw_only=True)
class Case:
    name: str
    params: Mapping[str, Any]
    build: Callable[[str], Callable[[], Any]]


@dataclass(frozen=True, kw_only=True)
class Measurement:
    name: str
    params: dict
    status: str
    timing: Timing | None
    memory: MemStats | None
    verdict: str
    reason: str | None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"잘못된 status: {self.status!r}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"잘못된 verdict: {self.verdict!r}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": dict(self.params),
            "status": self.status,
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "memory": self.memory.to_dict() if self.memory is not None else None,
            "verdict": self.verdict,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Measurement":
        t = d.get("timing")
        m = d.get("memory")
        return cls(
            name=d["name"],
            params=dict(d["params"]),
            status=d["status"],
            timing=Timing.from_dict(t) if t is not None else None,
            memory=MemStats.from_dict(m) if m is not None else None,
            verdict=d["verdict"],
            reason=d.get("reason"),
        )


def gate_case(params: Mapping) -> bool:
    return (params.get("mode") == "fwd"
            and (params.get("width"), params.get("height")) == RES_4K
            and params.get("n_prims") == GATE_PRIMS)


def verdict_for(params: Mapping, status: str, timing: Timing | None, device: DeviceInfo) -> str:
    if not gate_case(params):
        return "info"
    if not device.is_t4:
        return "unmeasured"
    if status == "oom":
        return "fail"
    if status in ("error", "unmeasured"):
        return "unmeasured"
    if timing is not None and timing.median_s <= GATE_4K_S + 1e-9:
        return "pass"
    return "fail"


def _skipped(case: Case, device: DeviceInfo, reason: str) -> Measurement:
    params = dict(case.params)
    return Measurement(name=case.name, params=params, status="unmeasured", timing=None, memory=None,
                       verdict=verdict_for(params, "unmeasured", None, device), reason=reason)


def _cap_reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {redact(str(exc))}"[:200]


def run_case(case: Case, *, device: DeviceInfo, torch_mod=None, repeats: int = 3, warmup: int = 1,
             clock=time.perf_counter, allow_cpu: bool = False) -> Measurement:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats 는 1 이상의 정수여야 한다")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup 은 0 이상의 정수여야 한다")
    params = dict(case.params)
    if device.kind != "cuda" and not allow_cpu:
        return _skipped(case, device, "no-cuda")

    is_cuda = device.kind == "cuda"
    if is_cuda and torch_mod is None:
        torch_mod = _import_torch()
    sync = make_sync(device, torch_mod)

    status = "ok"
    reason = None
    timing: Timing | None = None
    memory: MemStats | None = None
    try:
        fn = case.build(device.kind)
        for _ in range(warmup):
            fn()
            sync()
        sync()
        baseline = 0
        if is_cuda:
            torch_mod.cuda.reset_peak_memory_stats()
            baseline = torch_mod.cuda.memory_allocated()
        tm = time_calls(fn, repeats=repeats, warmup=0, sync=sync, clock=clock)
        timing = dataclasses.replace(tm, warmup=warmup)
        if is_cuda:
            memory = MemStats(
                baseline_gib=float(baseline) / GIB,
                peak_allocated_gib=float(torch_mod.cuda.max_memory_allocated()) / GIB,
                peak_reserved_gib=float(torch_mod.cuda.max_memory_reserved()) / GIB,
            )
    except Exception as exc:  # noqa: BLE001
        timing = None
        memory = None
        status = "oom" if is_oom(exc) else "error"
        reason = _cap_reason(exc)
    finally:
        if is_cuda:
            try:
                torch_mod.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    return Measurement(name=case.name, params=params, status=status, timing=timing, memory=memory,
                       verdict=verdict_for(params, status, timing, device), reason=reason)


def run_suite(cases, *, device: DeviceInfo, torch_mod=None, repeats: int = 3, warmup: int = 1,
              clock=time.perf_counter, allow_cpu: bool = False, budget_s: float | None = None,
              created_utc: str | None = None) -> "Report":
    start = clock()
    oom_min: dict[tuple, int] = {}
    out: list[Measurement] = []
    for case in cases:
        params = dict(case.params)
        key = (params.get("mode"), params.get("width"), params.get("height"))
        n = params.get("n_prims")
        if budget_s is not None and clock() - start >= budget_s:
            out.append(_skipped(case, device, "time-budget"))
            continue
        if key in oom_min and isinstance(n, int) and oom_min[key] <= n:
            out.append(_skipped(case, device, "skipped-after-oom"))
            continue
        m = run_case(case, device=device, torch_mod=torch_mod, repeats=repeats, warmup=warmup,
                     clock=clock, allow_cpu=allow_cpu)
        if m.status == "oom" and isinstance(n, int):
            oom_min[key] = min(oom_min.get(key, n), n)
        out.append(m)
    return Report(schema=SCHEMA_VERSION, device=device, measurements=tuple(out),
                  created_utc=created_utc, budget_s=budget_s)


# ---------------------------------------------------------------- 보고서
@dataclass(frozen=True, kw_only=True)
class Report:
    schema: int
    device: DeviceInfo
    measurements: tuple
    created_utc: str | None
    budget_s: float | None

    @property
    def shot_on_t4(self) -> bool:
        return bool(self.device.is_t4 and any(m.status == "ok" for m in self.measurements))

    @property
    def gate_verdict(self) -> str:
        gates = [m for m in self.measurements if gate_case(m.params)]
        if not gates:
            return "unmeasured"
        if any(m.verdict == "fail" for m in gates):
            return "fail"
        if all(m.verdict == "pass" for m in gates):
            return "pass"
        return "unmeasured"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "created_utc": self.created_utc,
            "budget_s": self.budget_s,
            "device": self.device.to_dict(),
            "measurements": [m.to_dict() for m in self.measurements],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Report":
        if not isinstance(d, Mapping):
            raise ValueError("보고서는 객체여야 한다")
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"지원하지 않는 schema: {d.get('schema')!r}")
        try:
            return cls(
                schema=SCHEMA_VERSION,
                device=DeviceInfo.from_dict(d["device"]),
                measurements=tuple(Measurement.from_dict(x) for x in d["measurements"]),
                created_utc=d.get("created_utc"),
                budget_s=d.get("budget_s"),
            )
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(f"보고서 형태가 잘못되었다: {exc}") from None


def _atomic_write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def write_report(report: Report, path) -> Path:
    data = json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _atomic_write_text(Path(path), data)


def load_report(path) -> Report:
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"보고서 JSON이 깨졌다: {exc}") from None
    return Report.from_dict(raw)


# ---------------------------------------------------------------- 장면·케이스
def _pos_int(v, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{name}: 1 이상의 정수여야 한다 ({v!r})")
    return v


def make_scene(n_prims, width, height, *, device="cpu", seed=0):
    torch = _import_torch()
    n = _pos_int(n_prims, "n_prims")
    w = _pos_int(width, "width")
    h = _pos_int(height, "height")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed 는 정수여야 한다")
    _ensure_repo_on_path()
    from buc.render.splat2d import Surfels  # noqa: PLC0415

    fx = 0.5 * w / math.tan(math.radians(30.0))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    g = torch.Generator().manual_seed(seed)
    z = 2.0 + 6.0 * torch.rand(n, generator=g, dtype=torch.float64)
    u = w * torch.rand(n, generator=g, dtype=torch.float64)
    v = h * torch.rand(n, generator=g, dtype=torch.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    means = torch.stack([x, y, z], dim=1).to(torch.float32)
    ls = torch.log(1.5 * z / fx).to(torch.float32)
    log_scales = torch.stack([ls, ls], dim=1)
    opacity_logits = torch.ones(n, dtype=torch.float32)
    colors = torch.rand(n, 3, generator=g, dtype=torch.float64).to(torch.float32)
    quats = torch.randn(n, 4, generator=g, dtype=torch.float64).to(torch.float32)

    surf = Surfels(
        means=means.to(device),
        quats=quats.to(device),
        log_scales=log_scales.to(device),
        opacity_logits=opacity_logits.to(device),
        colors=colors.to(device),
    )
    import numpy as np  # noqa: PLC0415

    c2w = np.eye(4, dtype=np.float64)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return surf, {"c2w": c2w, "K": K}


def _build_fwd(n: int, w: int, h: int) -> Callable[[str], Callable[[], Any]]:
    def build(device_kind: str):
        torch = _import_torch()
        _ensure_repo_on_path()
        from buc.render.splat2d import render  # noqa: PLC0415

        surf, cam = make_scene(n, w, h, device=device_kind)

        def fn():
            with torch.no_grad():
                return render(surf, c2w=cam["c2w"], K=cam["K"], width=w, height=h)

        return fn

    return build


def _build_fwdbwd(s: int, n: int) -> Callable[[str], Callable[[], Any]]:
    def build(device_kind: str):
        _ensure_repo_on_path()
        from buc.render.splat2d import render  # noqa: PLC0415

        surf, cam = make_scene(n, s, s, device=device_kind)

        def fn():
            surf.requires_grad_(True)
            for p in surf.params():
                p.grad = None
            out = render(surf, c2w=cam["c2w"], K=cam["K"], width=s, height=s)
            loss = out.rgb.mean()
            loss.backward()
            return float(loss.item())

        return fn

    return build


def default_cases(*, width: int = 3840, height: int = 2160, prims=(500_000, 1_000_000, 3_000_000),
                  train_size: int = 1024, train_prims: int = 1_000_000) -> list[Case]:
    w = _pos_int(width, "width")
    h = _pos_int(height, "height")
    s = _pos_int(train_size, "train_size")
    tp = _pos_int(train_prims, "train_prims")
    try:
        ps = [x for x in prims]
    except TypeError:
        raise ValueError("prims 는 정수 목록이어야 한다") from None
    for p in ps:
        _pos_int(p, "prims")
    if len(set(ps)) != len(ps):
        raise ValueError("prims 에 중복이 있다")
    cases: list[Case] = []
    for n in sorted(ps):
        cases.append(Case(
            name=f"splat_fwd_{w}x{h}_{n}",
            params={"mode": "fwd", "width": w, "height": h, "n_prims": n},
            build=_build_fwd(n, w, h),
        ))
    cases.append(Case(
        name=f"splat_fwdbwd_{s}x{s}_{tp}",
        params={"mode": "fwdbwd", "width": s, "height": s, "n_prims": tp},
        build=_build_fwdbwd(s, tp),
    ))
    return cases


def t4_offer_query(max_dph: float = 0.35):
    if isinstance(max_dph, bool) or not isinstance(max_dph, numbers.Real) \
            or not math.isfinite(float(max_dph)) or float(max_dph) <= 0.0:
        raise ValueError("max_dph 는 0보다 큰 유한한 실수여야 한다")
    _ensure_repo_on_path()
    from buc.rt.vast import OfferQuery  # noqa: PLC0415

    return OfferQuery(gpu_name="Tesla T4", num_gpus=1, max_dph=float(max_dph), min_reliability=0.95,
                      min_cpu_cores=4, min_ram_gb=16, min_disk_gb=60, min_inet_down=100, min_cuda=12.0)


# ---------------------------------------------------------------- 부록
_STATUS_KO = {"ok": "측정", "oom": "OOM", "error": "오류", "unmeasured": "미측정"}
_VERDICT_KO = {"pass": "통과", "fail": "미달", "unmeasured": "미측정", "info": "-"}
_GATE_KO = {"pass": "통과", "fail": "미달", "unmeasured": "미측정"}


def _cell(text: str) -> str:
    return str(text).replace("|", "/").replace("\r", " ").replace("\n", " ")


def render_appendix(report: Report) -> str:
    dev = report.device
    lines = ["# Shot on T4 실측 부록"]
    if dev.is_t4:
        mem = f"{dev.total_mem_gib:.1f} GiB" if dev.total_mem_gib is not None else "n/a"
        lines.append(f"> 측정 장치: {redact(dev.name)} · {mem} · torch {dev.torch_version} · "
                     f"CUDA {dev.cuda_version or 'n/a'}")
    else:
        lines.append(f"> 미측정: 이 장치는 T4가 아니다 ({redact(dev.name)}). T4 수치는 없다.")
    lines.append("")
    lines.append("| 케이스 | 상태 | 중앙값(s) | p95(s) | 최대 메모리(GiB) | 판정 | 비고 |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in report.measurements:
        if m.timing is None:
            med = p95 = "-"
        elif not dev.is_t4:
            med = p95 = "미측정"
        else:
            med = f"{m.timing.median_s:.3f}"
            p95 = f"{m.timing.p95_s:.3f}"
        if m.memory is None:
            mem_cell = "-"
        elif not dev.is_t4:
            mem_cell = "미측정"
        else:
            mem_cell = f"{m.memory.peak_reserved_gib:.2f}"
        reason = redact(m.reason) if m.reason else ""
        lines.append(
            f"| {_cell(m.name)} | {_STATUS_KO.get(m.status, m.status)} | {med} | {p95} | {mem_cell} | "
            f"{_VERDICT_KO.get(m.verdict, m.verdict)} | {_cell(reason)} |"
        )
    lines.append("")
    lines.append(f"게이트: 4K ≤0.6 s @3M 프리미티브 — {_GATE_KO[report.gate_verdict]}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI
def _int_arg(lo: int):
    def conv(s: str) -> int:
        try:
            v = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"정수여야 한다: {s!r}") from None
        if v < lo:
            raise argparse.ArgumentTypeError(f"{lo} 이상이어야 한다: {v}")
        return v

    return conv


def _prims_arg(s: str) -> tuple:
    try:
        return tuple(int(x) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"쉼표로 구분한 정수여야 한다: {s!r}") from None


def _budget_arg(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"실수여야 한다: {s!r}") from None
    if not math.isfinite(v) or v <= 0.0:
        raise argparse.ArgumentTypeError("0보다 큰 유한한 실수여야 한다")
    return v


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shot on T4 실측 부록 생성")
    p.add_argument("--out", default=str(_DEFAULT_OUT))
    p.add_argument("--repeats", type=_int_arg(1), default=3)
    p.add_argument("--warmup", type=_int_arg(0), default=1)
    p.add_argument("--prims", type=_prims_arg, default=(500_000, 1_000_000, 3_000_000))
    p.add_argument("--width", type=_int_arg(1), default=3840)
    p.add_argument("--height", type=_int_arg(1), default=2160)
    p.add_argument("--train-size", type=_int_arg(1), default=1024)
    p.add_argument("--train-prims", type=_int_arg(1), default=1_000_000)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--budget-min", type=_budget_arg, default=None)
    p.add_argument("--created-utc", default=None)
    return p


def main(argv=None, *, torch_mod=None, clock=time.perf_counter, cases=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if cases is None:
        try:
            cases = default_cases(width=args.width, height=args.height, prims=args.prims,
                                  train_size=args.train_size, train_prims=args.train_prims)
        except ValueError as exc:
            parser.error(str(exc))
    created = args.created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    budget_s = args.budget_min * 60.0 if args.budget_min is not None else None
    device = detect_device(torch_mod)
    report = run_suite(cases, device=device, torch_mod=torch_mod, repeats=args.repeats, warmup=args.warmup,
                       clock=clock, allow_cpu=args.allow_cpu, budget_s=budget_s, created_utc=created)
    out = Path(args.out)
    md = out.with_suffix(".md")
    try:
        write_report(report, out)
        _atomic_write_text(md, render_appendix(report))
    except Exception as exc:  # noqa: BLE001
        why = redact(f"{type(exc).__name__}: {exc}")
        print(json.dumps({"ok": False, "why": why}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "json": str(out), "md": str(md), "shot_on_t4": report.shot_on_t4,
                      "gate": report.gate_verdict}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
