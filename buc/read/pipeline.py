"""RD-21 판독 오케스트레이터·패킷 조립 v0 — R0–R13 순서 · 단계 캐시 · 단계별 폴백.

`InputImage` 한 장에 판독 단계 R0–R13 을 정해진 순서로 돌려 `ReadPacket` 을 조립한다. 어떤 단계가 예외를 내거나
계약 밖 출력을 내도 그 단계만 고정 폴백값으로 바꾸고 나머지는 계속 돈다(폴백은 `packet.fallbacks` 에 남는다).
계약서: ops/harness/contracts/RD-21.md.

`import buc.read.pipeline` 은 torch 를 로드하지 않는다(RD-07 `mrh_call` 은 R5 단계 안에서 지연 import).
전역 RNG 불변, 입력 불변, 결정론.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from buc.core import contracts as C

__all__ = [
    "PIPELINE_NAME", "PIPELINE_VERSION", "STEP_IDS", "EXEC_ORDER", "CORE_STEPS", "StepOutputError", "ReadOpts",
    "ReadStep", "ReadState", "StepRecord", "ReadResult", "MemoryReadCache", "default_steps", "run_read",
    "make_read_impl", "read_stage", "plane_depth", "upsample_grid",
]

PIPELINE_NAME = "rd21_read"
PIPELINE_VERSION = "0.1.0"
STEP_IDS = tuple(f"R{i}" for i in range(14))
EXEC_ORDER = ("R0", "R1", "R2", "R4", "R5", "R6", "R3", "R7", "R8", "R9", "R10", "R11", "R12", "R13")
CORE_STEPS = ("R0", "R1", "R5", "R6")

_DEFAULT_FOV_X_DEG = 50.0          # RD-02 DEFAULT_FOV_X_DEG 와 같은 값(geom_cam 을 import 하지 않고 고정)
_F_SOURCES = ("vp3", "vp2", "prior", "default")
_FOCAL_MM_RANGE = (1.0, 2000.0)    # (lo 열림, hi 닫힘) — FND-02 Theta.focal_mm 범위
_ORDER_IDX = {s: i for i, s in enumerate(EXEC_ORDER)}


class StepOutputError(ValueError):
    """단계 출력이 계약 밖이다(폴백 사유)."""


# ---------------------------------------------------------------------------------------------------------------
# 옵션 · 단계 · 상태 · 기록
# ---------------------------------------------------------------------------------------------------------------
def _pos_finite(name: str, v: Any) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: 수여야 한다 ({v!r})")
    f = float(v)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"{name}: 양의 유한수여야 한다 ({v!r})")
    return f


@dataclass(frozen=True, kw_only=True)
class ReadOpts:
    cam_height_m: float = 1.5
    z_near_m: float = 0.5
    z_far_m: float = 50.0
    default_sigma_s: float = 1.5
    depth_min_m: float = 1e-3
    sensor_width_mm: float = 36.0

    def __post_init__(self) -> None:
        for n in ("cam_height_m", "z_near_m", "z_far_m", "default_sigma_s", "depth_min_m", "sensor_width_mm"):
            object.__setattr__(self, n, _pos_finite(n, getattr(self, n)))
        if self.z_near_m > self.z_far_m:
            raise ValueError(f"z_near_m({self.z_near_m}) 은 z_far_m({self.z_far_m}) 이하여야 한다")


@dataclass(frozen=True)
class ReadStep:
    sid: str
    name: str
    version: str
    fn: Callable[..., Any]
    needs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sid, str) or self.sid not in STEP_IDS:
            raise ValueError(f"ReadStep.sid {self.sid!r} 는 {STEP_IDS} 중 하나여야 한다")
        for nm in ("name", "version"):
            v = getattr(self, nm)
            if not isinstance(v, str) or not v:
                raise ValueError(f"ReadStep.{nm} 은 비지 않은 str 이어야 한다 ({v!r})")
        if not callable(self.fn):
            raise ValueError("ReadStep.fn 은 호출 가능해야 한다")
        if isinstance(self.needs, (str, bytes)):
            raise ValueError("ReadStep.needs 는 sid 의 열이어야 한다")
        try:
            needs = tuple(self.needs)
        except TypeError:
            raise ValueError("ReadStep.needs 는 sid 의 열이어야 한다") from None
        if not all(isinstance(n, str) for n in needs):
            raise ValueError("ReadStep.needs 의 원소는 str 이어야 한다")
        object.__setattr__(self, "needs", needs)


class ReadState:
    """단계가 보는 상태. `state[sid]` 는 `needs` 안의 상류 출력(읽기 전용)만 돌려준다."""

    def __init__(self, image: C.InputImage, H: int, W: int, opts: ReadOpts,
                 views: Optional[Mapping[str, Mapping]] = None, statuses: Optional[Mapping[str, str]] = None):
        self.image = image
        self.H = int(H)
        self.W = int(W)
        self.opts = opts
        self._views = dict(views or {})
        self._status = dict(statuses or {})

    def __getitem__(self, sid: str) -> Mapping:
        if sid not in self._views:
            raise KeyError(sid)
        return self._views[sid]

    def status(self, sid: str) -> str:
        if sid not in self._status:
            raise KeyError(sid)
        return self._status[sid]


@dataclass(frozen=True)
class StepRecord:
    sid: str
    name: Optional[str]
    version: Optional[str]
    status: str
    key: Optional[str]
    error: Optional[str]
    wall_s: float


@dataclass(frozen=True)
class ReadResult:
    packet: C.ReadPacket
    records: Tuple[StepRecord, ...]
    outputs: Dict[str, Mapping]


class MemoryReadCache:
    """단계 출력 캐시. put·get 모두 깊은 사본을 주고받는다."""

    def __init__(self) -> None:
        self._d: Dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        v = self._d.get(key)
        return None if v is None else copy.deepcopy(v)

    def put(self, key: str, outputs: Mapping) -> None:
        self._d[key] = copy.deepcopy(dict(outputs))

    def __len__(self) -> int:
        return len(self._d)


# ---------------------------------------------------------------------------------------------------------------
# 수치 도우미
# ---------------------------------------------------------------------------------------------------------------
def _axis_interp(n_in: int, n_out: int):
    """쌍선형 align_corners=False 의 한 축: (i0, i1, 가중치 lam)."""
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    lam = src - i0
    return i0, i1, lam


def _pos_int(name: str, v: Any) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 1:
        raise ValueError(f"{name}: 양의 정수여야 한다 ({v!r})")
    return int(v)


def upsample_grid(a, H: int, W: int) -> np.ndarray:
    """(h,w) 격자를 (H,W) 로 쌍선형(align_corners=False) 업샘플. float32."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"a: (h,w) 2차원이어야 한다 ({a.shape})")
    H, W = _pos_int("H", H), _pos_int("W", W)
    h, w = a.shape
    y0, y1, ly = _axis_interp(h, H)
    x0, x1, lx = _axis_interp(w, W)
    top = a[y0][:, x0] * (1.0 - lx) + a[y0][:, x1] * lx
    bot = a[y1][:, x0] * (1.0 - lx) + a[y1][:, x1] * lx
    out = top * (1.0 - ly)[:, None] + bot * ly[:, None]
    return out.astype(np.float32)


def plane_depth(K, up_cam, H: int, W: int, *, cam_height_m: float, z_near_m: float, z_far_m: float) -> np.ndarray:
    """지면 `up·X = -h` 평면 z-깊이. d=K⁻¹[u,v,1], `-(up·d) > 1e-9` 이면 clip(h/-(up·d), z_near, z_far) 아니면 z_far."""
    K = np.asarray(K, dtype=np.float64)
    up = np.asarray(up_cam, dtype=np.float64)
    if K.shape != (3, 3) or up.shape != (3,):
        raise ValueError("K 는 (3,3), up_cam 은 (3,) 이어야 한다")
    H, W = _pos_int("H", H), _pos_int("W", W)
    Ki = np.linalg.inv(K)
    # den(u,v) = -(up · Ki @ [u,v,1]) 는 (u,v,1) 에 대해 선형이다.
    c = -(up @ Ki)
    uu = np.arange(W, dtype=np.float64)[None, :]
    vv = np.arange(H, dtype=np.float64)[:, None]
    den = c[0] * uu + c[1] * vv + c[2]
    ok = den > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(ok, np.clip(cam_height_m / np.where(ok, den, 1.0), z_near_m, z_far_m), z_far_m)
    return d.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# 출력 검증 · 폴백
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _finite_real(name: str, v: Any) -> float:
    if not _is_real(v) or not math.isfinite(float(v)):
        raise StepOutputError(f"{name}: 유한한 수여야 한다 ({v!r})")
    return float(v)


def _array(name: str, v: Any, shape: Optional[tuple] = None, ndim: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    if v is None:
        raise StepOutputError(f"{name}: 배열이어야 한다 (None)")
    try:
        a = np.asarray(v)
    except Exception as e:  # noqa: BLE001
        raise StepOutputError(f"{name}: 배열로 바꿀 수 없다 ({type(e).__name__})") from None
    if a.dtype.kind not in "biuf" or a.dtype.kind == "b":
        raise StepOutputError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    if shape is not None and a.shape != tuple(shape):
        raise StepOutputError(f"{name}: shape {a.shape} != {tuple(shape)}")
    if ndim is not None and (a.ndim != ndim or a.size == 0):
        raise StepOutputError(f"{name}: {ndim}차원 비어 있지 않은 배열이어야 한다 (shape {a.shape})")
    a = np.array(a, dtype=dtype, copy=True)
    if not np.isfinite(a).all():
        raise StepOutputError(f"{name}: 비유한 값")
    return a


def _check_meta(out: dict) -> None:
    if "meta" not in out:
        raise StepOutputError("meta: 없다")
    if not isinstance(out["meta"], dict):
        raise StepOutputError("meta: dict 이어야 한다")
    try:
        json.dumps(out["meta"], allow_nan=False)
    except (TypeError, ValueError, OverflowError) as e:
        raise StepOutputError(f"meta: JSON 직렬화 불가 ({e})") from None


def _freeze(x: Any) -> None:
    if isinstance(x, np.ndarray):
        x.setflags(write=False)
    elif isinstance(x, dict):
        for v in x.values():
            _freeze(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _freeze(v)


def _validate_step_output(sid: str, raw: Any, state: ReadState) -> dict:
    """계약 밖이면 StepOutputError. 통과하면 정규화된 깊은 사본(배열은 쓰기 금지)을 돌려준다."""
    if not isinstance(raw, dict):
        raise StepOutputError(f"출력은 dict 여야 한다 ({type(raw).__name__})")
    try:
        out = copy.deepcopy(raw)
    except Exception as e:  # noqa: BLE001
        raise StepOutputError(f"출력을 복사할 수 없다 ({type(e).__name__})") from None
    _check_meta(out)
    H, W = state.H, state.W
    try:
        if sid == "R0":
            out["image_lin"] = _array("image_lin", out.get("image_lin"), shape=(H, W, 3))
            if out["image_lin"].min() < 0:
                raise StepOutputError("image_lin: 음수 값")
            fov = out.get("fov_x_prior_deg")
            if fov is not None:
                fov = _finite_real("fov_x_prior_deg", fov)
                if not 0.0 < fov < 180.0:
                    raise StepOutputError(f"fov_x_prior_deg: (0,180) 밖 ({fov})")
            out["fov_x_prior_deg"] = fov
        elif sid == "R1":
            K = _array("K", out.get("K"), shape=(3, 3), dtype=np.float64)
            if np.abs(K[2] - np.array([0.0, 0.0, 1.0])).max() > 1e-9:
                raise StepOutputError("K: 마지막 행이 [0,0,1] 이 아니다")
            if not (K[0, 0] > 0 and K[1, 1] > 0):
                raise StepOutputError("K: fx, fy > 0 이어야 한다")
            focal_mm = float(K[0, 0]) / W * state.opts.sensor_width_mm
            if not (_FOCAL_MM_RANGE[0] < focal_mm <= _FOCAL_MM_RANGE[1]):
                raise StepOutputError(f"K: focal_mm {focal_mm:.4g} 이 범위 밖")
            out["K"] = K
            out["k1"] = _finite_real("k1", out.get("k1"))
            out["k2"] = _finite_real("k2", out.get("k2"))
            up = _array("up_cam", out.get("up_cam"), shape=(3,), dtype=np.float64)
            n = float(np.linalg.norm(up))
            if not 0.99 <= n <= 1.01:
                raise StepOutputError(f"up_cam: 노름 {n:.4g} 이 0.99–1.01 밖")
            out["up_cam"] = up / n
            if out.get("f_source") not in _F_SOURCES:
                raise StepOutputError(f"f_source: {_F_SOURCES} 중 하나여야 한다 ({out.get('f_source')!r})")
        elif sid == "R5":
            ld = _array("log_depth", out.get("log_depth"), ndim=2)
            rd = _array("r_dis_map", out.get("r_dis_map"), shape=ld.shape)
            if rd.min() < 0:
                raise StepOutputError("r_dis_map: 음수 값")
            out["log_depth"], out["r_dis_map"] = ld, rd
        elif sid == "R6":
            s = _finite_real("s", out.get("s"))
            sg = _finite_real("sigma_s", out.get("sigma_s"))
            if not s > 0:
                raise StepOutputError(f"s: 양수여야 한다 ({s})")
            if sg < 0:
                raise StepOutputError(f"sigma_s: 0 이상이어야 한다 ({sg})")
            rel = out.get("reliable")
            if not isinstance(rel, (bool, np.bool_)):
                raise StepOutputError(f"reliable: bool 이어야 한다 ({rel!r})")
            out["s"], out["sigma_s"], out["reliable"] = s, sg, bool(rel)
        elif sid == "R11":
            if "level" in out and (not isinstance(out["level"], str) or out["level"] not in C.LEVELS):
                raise StepOutputError(f"level: {C.LEVELS} 중 하나여야 한다 ({out['level']!r})")
    except StepOutputError:
        raise
    except Exception as e:  # noqa: BLE001
        raise StepOutputError(f"{sid} 출력 검증 중 {type(e).__name__}: {e}") from None
    _freeze(out)
    return out


def _exif_fov_prior(image: C.InputImage, W: int, H: int) -> Optional[float]:
    f35 = image.exif.get("focal_35mm") if isinstance(image.exif, dict) else None
    if not _is_real(f35) or not (math.isfinite(float(f35)) and float(f35) > 0):
        return None
    from buc.read.ingest import fov_x_prior_deg
    try:
        return float(fov_x_prior_deg(float(f35), W, H))
    except ValueError:
        return None


def _classic_lin(image: C.InputImage) -> np.ndarray:
    from buc.pipeline.stubs.read_classic import _image_lin
    return _image_lin(image)


def _fallback_output(sid: str, status: str, image: C.InputImage, H: int, W: int, opts: ReadOpts,
                     outs: Mapping[str, Mapping]) -> dict:
    meta = {"status": status}
    if sid == "R0":
        out = {"image_lin": _classic_lin(image), "fov_x_prior_deg": _exif_fov_prior(image, W, H), "meta": meta}
    elif sid == "R1":
        fov = outs["R0"].get("fov_x_prior_deg") if "R0" in outs else None
        src = "prior"
        f = None
        if fov is not None:
            f = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
            if not (_FOCAL_MM_RANGE[0] < f / W * opts.sensor_width_mm <= _FOCAL_MM_RANGE[1]):
                f = None
        if f is None:
            f = (W / 2.0) / math.tan(math.radians(_DEFAULT_FOV_X_DEG) / 2.0)
            src = "default"
        K = np.array([[f, 0.0, (W - 1) / 2.0], [0.0, f, (H - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        out = {"K": K, "k1": 0.0, "k2": 0.0, "up_cam": np.array([0.0, -1.0, 0.0]), "f_source": src, "meta": meta}
    elif sid == "R5":
        out = {"log_depth": None, "r_dis_map": None, "meta": meta}
    elif sid == "R6":
        out = {"s": 1.0, "sigma_s": float(opts.default_sigma_s), "reliable": False, "meta": meta}
    else:
        out = {"meta": meta}
    _freeze(out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 검증(실행 전)
# ---------------------------------------------------------------------------------------------------------------
def _check_steps(steps: Any) -> Dict[str, ReadStep]:
    if isinstance(steps, (str, bytes)) or not hasattr(steps, "__iter__"):
        raise TypeError("steps: ReadStep 의 열이어야 한다")
    by: Dict[str, ReadStep] = {}
    for st in steps:
        if not isinstance(st, ReadStep):
            raise TypeError(f"steps 원소는 ReadStep 이어야 한다 ({type(st).__name__})")
        if st.sid in by:
            raise ValueError(f"steps: sid {st.sid} 가 중복이다")
        by[st.sid] = st
    for st in by.values():
        for n in st.needs:
            if n not in _ORDER_IDX:
                raise ValueError(f"{st.sid}.needs: 모르는 sid {n!r}")
            if n == st.sid:
                raise ValueError(f"{st.sid}.needs: 자기 자신에 의존한다")
            if _ORDER_IDX[n] >= _ORDER_IDX[st.sid]:
                raise ValueError(f"{st.sid}.needs: {n} 은 실행 순서상 뒤다")
    return by


def _check_params(params: Any) -> Dict[str, dict]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("params: {sid: dict} 매핑이어야 한다")
    out: Dict[str, dict] = {}
    for k, v in params.items():
        if not isinstance(k, str) or k not in STEP_IDS:
            raise ValueError(f"params: 모르는 sid {k!r}")
        if not isinstance(v, Mapping):
            raise ValueError(f"params[{k!r}] 는 dict 여야 한다")
        out[k] = dict(v)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 캐시 키
# ---------------------------------------------------------------------------------------------------------------
def _opts_dict(opts: ReadOpts) -> dict:
    return dataclasses.asdict(opts)


def _step_key(step: ReadStep, params: Mapping, opts: ReadOpts, image_hash: str, eff: Mapping[str, str]) -> str:
    return C.content_hash({
        "pipeline": f"{PIPELINE_NAME}@{PIPELINE_VERSION}", "sid": step.sid, "name": step.name, "version": step.version,
        "params": dict(params), "opts": _opts_dict(opts), "image": image_hash,
        "needs": {n: eff[n] for n in step.needs},
    })


def _fallback_eff_key(sid: str, status: str, opts: ReadOpts, image_hash: str, eff: Mapping[str, str]) -> str:
    body = {"sid": sid, "status": status, "image": image_hash, "opts": _opts_dict(opts)}
    if sid == "R1":
        body["r0"] = eff["R0"]
    return C.content_hash(body)


# ---------------------------------------------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------------------------------------------
def _view_of(out: Mapping) -> Mapping:
    """상류 출력의 읽기 전용 보기: 배열은 공유(쓰기 금지), 나머지는 깊은 사본."""
    return MappingProxyType({k: (v if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in out.items()})


def _err_text(e: BaseException) -> str:
    msg = str(e)
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def run_read(image, *, steps=None, params=None, cache=None, opts=None) -> ReadResult:
    if not isinstance(image, C.InputImage):
        raise TypeError("image: InputImage 여야 한다")
    if opts is None:
        opts = ReadOpts()
    elif not isinstance(opts, ReadOpts):
        raise TypeError("opts: ReadOpts 여야 한다")
    by = _check_steps(default_steps() if steps is None else steps)
    prm = _check_params(params)
    if cache is not None and not (callable(getattr(cache, "get", None)) and callable(getattr(cache, "put", None))):
        raise TypeError("cache: get/put 이 있어야 한다")
    image_hash = None
    if cache is not None:
        image_hash = image.content_hash()
        try:
            for v in prm.values():
                C.content_hash(v)
        except C.ContractError as e:
            raise ValueError(f"params: 해시할 수 없는 값 ({e})") from None

    H, W = int(image.pixels.shape[0]), int(image.pixels.shape[1])
    outs: Dict[str, Mapping] = {}
    st: Dict[str, str] = {}          # ok | fallback | skipped
    eff: Dict[str, str] = {}         # 유효 키(캐시 사용 시)
    errors: Dict[str, str] = {}
    fb_type: Dict[str, str] = {}
    records = []

    for sid in EXEC_ORDER:
        t0 = time.perf_counter()
        step = by.get(sid)
        if step is None:
            outs[sid] = _fallback_output(sid, "skipped", image, H, W, opts, outs)
            st[sid] = "skipped"
            if cache is not None:
                eff[sid] = _fallback_eff_key(sid, "skipped", opts, image_hash, eff)
            records.append(StepRecord(sid, None, None, "skipped", None, None, time.perf_counter() - t0))
            continue

        p = prm.get(sid, {})
        key = None
        if cache is not None:
            key = _step_key(step, p, opts, image_hash, eff)
            try:
                hit = cache.get(key)
            except Exception:  # noqa: BLE001  캐시는 최적화일 뿐이다
                hit = None
            if hit is not None:
                _freeze(hit)
                outs[sid] = hit
                st[sid] = "ok"
                eff[sid] = key
                records.append(StepRecord(sid, step.name, step.version, "hit", key, None, time.perf_counter() - t0))
                continue

        state = ReadState(image, H, W, opts, {n: _view_of(outs[n]) for n in step.needs},
                          {n: st[n] for n in step.needs})
        try:
            raw = step.fn(state, copy.deepcopy(p))
            out = _validate_step_output(sid, raw, state)
        except Exception as e:  # noqa: BLE001  BaseException(KeyboardInterrupt 등)은 전파
            outs[sid] = _fallback_output(sid, "fallback", image, H, W, opts, outs)
            st[sid] = "fallback"
            fb_type[sid] = type(e).__name__
            errors[sid] = _err_text(e)
            if cache is not None:
                eff[sid] = _fallback_eff_key(sid, "fallback", opts, image_hash, eff)
            records.append(StepRecord(sid, step.name, step.version, "fallback", key, errors[sid],
                                      time.perf_counter() - t0))
            continue

        outs[sid] = out
        st[sid] = "ok"
        if cache is not None:
            eff[sid] = key
            try:
                cache.put(key, out)
            except Exception:  # noqa: BLE001
                pass
        records.append(StepRecord(sid, step.name, step.version, "ok", key, None, time.perf_counter() - t0))

    packet = _assemble(image, H, W, opts, by, outs, st, records, fb_type)
    return ReadResult(packet=packet, records=tuple(records), outputs=dict(outs))


# ---------------------------------------------------------------------------------------------------------------
# 패킷 조립
# ---------------------------------------------------------------------------------------------------------------
def _assemble(image, H, W, opts, by, outs, st, records, fb_type) -> C.ReadPacket:
    r0, r1, r5, r6 = outs["R0"], outs["R1"], outs["R5"], outs["R6"]
    K = np.array(r1["K"], dtype=np.float64)
    obs = C.Obs(K=K, R=np.eye(3), t=np.zeros(3), width=W, height=H, k1=r1["k1"], k2=r1["k2"])
    sensor = opts.sensor_width_mm
    theta = C.Theta(aspect=W / H, focal_mm=float(K[0, 0]) / W * sensor, sensor_width_mm=sensor)

    mrh = st["R5"] == "ok" and r5.get("log_depth") is not None
    if mrh:
        ld = upsample_grid(r5["log_depth"], H, W).astype(np.float64)
        depth = np.maximum(float(r6["s"]) * np.exp(ld), opts.depth_min_m).astype(np.float32)
        conf = np.clip(1.0 - upsample_grid(r5["r_dis_map"], H, W).astype(np.float64), 0.0, 1.0).astype(np.float32)
        scale_s, sigma_s, source = float(r6["s"]), float(r6["sigma_s"]), "mrh"
    else:
        depth = plane_depth(K, r1["up_cam"], H, W, cam_height_m=opts.cam_height_m, z_near_m=opts.z_near_m,
                            z_far_m=opts.z_far_m)
        conf = np.zeros((H, W), dtype=np.float32)
        scale_s, sigma_s, source = 1.0, float(opts.default_sigma_s), "plane"

    level = outs["R11"].get("level", "L1")
    fallbacks = []
    for rec in records:
        if rec.status == "fallback":
            fallbacks.append(f"{rec.sid}:{rec.name}@{rec.version}:{fb_type[rec.sid]}")
        elif rec.status == "skipped" and rec.sid in CORE_STEPS:
            fallbacks.append(f"{rec.sid}:skipped")

    steps_meta = {}
    for sid in STEP_IDS:
        s = by.get(sid)
        steps_meta[sid] = {"impl": None if s is None else f"{s.name}@{s.version}", "status": st[sid]}
    meta = {
        "input": copy.deepcopy(r0["meta"]),
        "read": {
            "pipeline": f"{PIPELINE_NAME}@{PIPELINE_VERSION}",
            "depth_source": source,
            "steps": steps_meta,
            "geom": copy.deepcopy(r1["meta"]),
            "mrh": copy.deepcopy(r5["meta"]),
            "scale": copy.deepcopy(r6["meta"]),
            "extra": {sid: copy.deepcopy(outs[sid]["meta"]) for sid in STEP_IDS if sid not in CORE_STEPS},
        },
    }
    return C.ReadPacket(
        image_lin=r0["image_lin"], depth=depth, obs=obs, theta_in=theta, input_hash=image.h_I, prov=image.prov,
        scale_s=scale_s, sigma_s=sigma_s, level=level, depth_conf=conf, normals=None, esc={}, faces=(), lights={},
        dyn={}, fallbacks=tuple(fallbacks), meta=meta)


# ---------------------------------------------------------------------------------------------------------------
# 기본 단계
# ---------------------------------------------------------------------------------------------------------------
def _r0_ingest(state: ReadState, params: dict) -> dict:
    if params:
        raise ValueError(f"R0 ingest: 알 수 없는 params {sorted(params)}")
    from buc.read.ingest import IngestOpts, ingest
    img = state.image
    cs = None if img.color_space == "unknown" else img.color_space
    r = ingest(img.pixels, color_space=cs, prov=img.prov, consent=img.consent,
               opts=IngestOpts(letterbox=False, apply_orientation=False))
    fov = r.fov_x_prior_deg
    if fov is None:
        fov = _exif_fov_prior(img, state.W, state.H)
    return {"image_lin": r.image_lin, "fov_x_prior_deg": fov, "meta": r.meta_input()}


def _r1_geom_cam(state: ReadState, params: dict) -> dict:
    from buc.read.geom_cam import GeomCamOpts, estimate_geom_camera
    r0 = state["R0"]
    g = estimate_geom_camera(r0["image_lin"], fov_prior_deg=r0["fov_x_prior_deg"], opts=GeomCamOpts(**params))
    return {"K": g.K, "k1": g.k1, "k2": g.k2, "up_cam": g.up_cam, "f_source": g.f_source, "meta": g.to_meta()}


def _make_r5(reader) -> Callable:
    def fn(state: ReadState, params: dict) -> dict:
        from buc.read.mrh_call import ReadConfig, read_mrh
        cfg = ReadConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()})
        ro = read_mrh(state["R0"]["image_lin"], reader, prov=state.image.prov, cfg=cfg)
        meta = ro.meta()
        meta["prov"] = ro.prov.to_dict()
        return {"log_depth": ro.outputs["log_depth"][0, 0].detach().cpu().numpy(),
                "r_dis_map": ro.r_dis_map.detach().cpu().numpy(), "meta": meta}
    return fn


def _make_r6(size_obs_fn) -> Callable:
    def fn(state: ReadState, params: dict) -> dict:
        from buc.read.metric_scale import SIGMA_LOG_F, ScaleOpts, estimate_metric_scale
        objs = list(size_obs_fn(state)) if size_obs_fn is not None else []
        ms = estimate_metric_scale(objs, sigma_log_f=SIGMA_LOG_F[state["R1"]["f_source"]], opts=ScaleOpts(**params))
        return {"s": ms.s, "sigma_s": ms.sigma_s, "reliable": ms.reliable, "meta": ms.to_meta()}
    return fn


def _cache_id(obj: Any, what: str) -> str:
    cid = getattr(obj, "cache_id", None)
    if not isinstance(cid, str) or not cid:
        raise ValueError(f"{what}: 비지 않은 str `cache_id` 가 있어야 한다")
    return cid


def default_steps(*, reader=None, size_obs_fn=None) -> Tuple[ReadStep, ...]:
    steps = [ReadStep("R0", "ingest", "0.1.0", _r0_ingest),
             ReadStep("R1", "geom_cam", "0.1.0", _r1_geom_cam, ("R0",))]
    if reader is not None:
        steps.append(ReadStep("R5", "mrh_call", f"0.1.0+{_cache_id(reader, 'reader')}", _make_r5(reader), ("R0",)))
    sid_v = "none" if size_obs_fn is None else _cache_id(size_obs_fn, "size_obs_fn")
    steps.append(ReadStep("R6", "metric_scale", f"0.1.0+{sid_v}", _make_r6(size_obs_fn), ("R1", "R5")))
    return tuple(steps)


# ---------------------------------------------------------------------------------------------------------------
# 스파인 이음매
# ---------------------------------------------------------------------------------------------------------------
def make_read_impl(*, steps=None, cache=None, opts=None) -> Any:  # spine.Impl
    from buc.pipeline import spine as S  # 지연 import: spine 이 det 를 통해 torch 를 끌고 온다
    if opts is None:
        opts = ReadOpts()
    elif not isinstance(opts, ReadOpts):
        raise TypeError("opts: ReadOpts 여야 한다")
    steps = default_steps() if steps is None else tuple(steps)
    by = _check_steps(steps)
    ident = C.content_hash({
        "steps": [[sid, by[sid].name, by[sid].version] for sid in EXEC_ORDER if sid in by],
        "opts": _opts_dict(opts),
    })

    def fn(inputs, ctx):
        params = {k: v for k, v in ctx.params.items() if k != "_seed"}
        res = run_read(inputs["image"], steps=steps, params=params, cache=cache, opts=opts)
        return {"read": res.packet}

    return S.Impl(stage="S-1", kind="real", name=PIPELINE_NAME, version=f"{PIPELINE_VERSION}+{ident[:12]}", fn=fn)


def read_stage(inputs, ctx):
    """기본 단계·옵션으로 도는 S-1 구현 (= `make_read_impl().fn`)."""
    return make_read_impl().fn(inputs, ctx)
