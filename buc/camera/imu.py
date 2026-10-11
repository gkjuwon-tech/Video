"""CAM-06 IMU 파이프 v0: GPMF / 폰·공개 IMU 파서, 200 Hz 균일화, 기기 간 시계 정렬, 증강 x5.5.

학습 0, 순수 수식(numpy / scipy). torch 금지, import 부작용 없음, 전역 RNG 불사용.
SI 단위: rad/s, m/s^2(중력 포함), s, float64. 열 순서는 센서 원본(x, y, z) 그대로.
계약서: ops/harness/contracts/CAM-06.md
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

__all__ = [
    "RATE_HZ", "AUG_FACTOR", "G_STD", "MAX_GAP_S", "WINDOW_LEN", "GRAVITY_LP_HZ", "RIG_KINDS",
    "Series", "RawImu", "ImuClip", "AlignEstimate", "AugmentConfig",
    "parse_gpmf", "parse_phone_csv", "merge_raw", "split_segments", "resample_uniform",
    "rotate_frame", "shift_clip", "estimate_offset", "align_pair", "gravity_estimate",
    "warped_length", "time_warp", "rig_filter", "perturb_sensor",
    "sample_augmentation", "apply_augmentation", "augment_clip", "augment_dataset", "make_windows",
]

RATE_HZ = 200.0
AUG_FACTOR = 5.5
G_STD = 9.80665
MAX_GAP_S = 0.1
WINDOW_LEN = 400
GRAVITY_LP_HZ = 0.5
RIG_KINDS = ("lowpass", "highpass")
_SALT = 0xCA06


# ---------------------------------------------------------------------------- 공용 도우미
def _ro(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _check_meta(meta: Any) -> dict:
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        raise TypeError("meta must be a dict")
    try:
        json.dumps(meta)
    except (TypeError, ValueError) as e:
        raise TypeError(f"meta must be JSON serializable: {e}") from None
    return copy.deepcopy(meta)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _finite_float(x: Any, name: str) -> float:
    if not _is_num(x):
        raise ValueError(f"{name} must be a real number")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _rng_pair(rng_range: Any, name: str) -> tuple:
    try:
        lo, hi = rng_range
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (lo, hi) pair") from None
    lo, hi = _finite_float(lo, name), _finite_float(hi, name)
    if lo > hi:
        raise ValueError(f"{name}: lo must be <= hi")
    return (lo, hi)


# ---------------------------------------------------------------------------- 자료형
@dataclasses.dataclass(frozen=True, eq=False)
class Series:
    """센서 하나의 (시각, 3축 값). t 는 엄격 증가."""
    t: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        t = np.array(self.t, dtype=np.float64)
        v = np.array(self.v, dtype=np.float64)
        if t.ndim != 1 or v.ndim != 2 or v.shape[1] != 3 or v.shape[0] != t.shape[0]:
            raise ValueError("Series needs t (N,) and v (N,3)")
        if t.shape[0] < 2:
            raise ValueError("Series needs N >= 2")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
            raise ValueError("Series must be finite")
        if not np.all(np.diff(t) > 0):
            raise ValueError("Series t must be strictly increasing")
        _set(self, "t", _ro(t))
        _set(self, "v", _ro(v))


@dataclasses.dataclass(frozen=True, eq=False)
class RawImu:
    """센서별 자기 시각을 가진 원시 IMU."""
    gyro: Series | None
    accel: Series | None
    source: str
    device: str = ""
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        for nm in ("gyro", "accel"):
            s = getattr(self, nm)
            if s is not None and not isinstance(s, Series):
                raise TypeError(f"{nm} must be a Series or None")
        if self.gyro is None and self.accel is None:
            raise ValueError("RawImu needs at least one sensor")
        if not isinstance(self.source, str) or not isinstance(self.device, str):
            raise TypeError("source/device must be str")
        _set(self, "meta", _check_meta(self.meta))


@dataclasses.dataclass(frozen=True, eq=False)
class ImuClip:
    """균일 격자 IMU 클립. 표본 i 의 시각 = t0 + i / rate_hz."""
    t0: float
    rate_hz: float
    gyro: np.ndarray
    accel: np.ndarray
    source: str = ""
    device: str = ""
    meta: dict | None = None

    def __post_init__(self):
        t0 = float(self.t0)
        rate = float(self.rate_hz)
        if not math.isfinite(t0):
            raise ValueError("t0 must be finite")
        if not (math.isfinite(rate) and rate > 0):
            raise ValueError("rate_hz must be > 0")
        g = np.array(self.gyro, dtype=np.float64)
        a = np.array(self.accel, dtype=np.float64)
        if g.ndim != 2 or g.shape[1] != 3 or a.shape != g.shape:
            raise ValueError("gyro/accel must both be (N,3)")
        if g.shape[0] < 1:
            raise ValueError("ImuClip needs N >= 1")
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(a))):
            raise ValueError("gyro/accel must be finite")
        if not isinstance(self.source, str) or not isinstance(self.device, str):
            raise TypeError("source/device must be str")
        _set(self, "t0", t0)
        _set(self, "rate_hz", rate)
        _set(self, "gyro", _ro(g))
        _set(self, "accel", _ro(a))
        _set(self, "meta", _check_meta(self.meta))

    @property
    def n(self) -> int:
        return int(self.gyro.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n / self.rate_hz

    @property
    def t(self) -> np.ndarray:
        return self.t0 + np.arange(self.n) / self.rate_hz

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.gyro, self.accel], axis=1)

    def slice(self, i0: int, i1: int) -> "ImuClip":
        i0, i1 = int(i0), int(i1)
        if i0 < 0 or i1 > self.n or i0 >= i1:
            raise ValueError(f"invalid slice [{i0}, {i1}) for n={self.n}")
        return ImuClip(self.t0 + i0 / self.rate_hz, self.rate_hz, self.gyro[i0:i1], self.accel[i0:i1],
                       self.source, self.device, self.meta)


def _clip_like(clip: ImuClip, *, t0=None, gyro=None, accel=None, meta=None) -> ImuClip:
    return ImuClip(clip.t0 if t0 is None else t0, clip.rate_hz, clip.gyro if gyro is None else gyro,
                   clip.accel if accel is None else accel, clip.source, clip.device,
                   clip.meta if meta is None else meta)


def _need_clip(x: Any, name: str) -> None:
    if not isinstance(x, ImuClip):
        raise TypeError(f"{name} must be an ImuClip")


# ---------------------------------------------------------------------------- GPMF
_GPMF_DTYPES = {"b": ">i1", "B": ">u1", "s": ">i2", "S": ">u2", "l": ">i4", "L": ">u4", "j": ">i8", "J": ">u8",
                "f": ">f4", "d": ">f8"}


def _klv_items(buf: bytes, pos: int, end: int):
    """[pos, end) 의 KLV 항목 목록과 '잘림' 여부. 항목: (key, typ, size, repeat, payload_start, payload_end)."""
    items = []
    truncated = False
    while pos + 8 <= end:
        key = buf[pos:pos + 4]
        typ = buf[pos + 4]
        size = buf[pos + 5]
        rep = struct.unpack(">H", buf[pos + 6:pos + 8])[0]
        ps = pos + 8
        pe = ps + size * rep
        if pe > end:
            truncated = True
            break
        items.append((key, typ, size, rep, ps, pe))
        pos = ps + ((size * rep + 3) // 4) * 4
    else:
        if pos < end and any(buf[pos:end]):
            truncated = True
    return items, truncated


def _gpmf_decode(buf: bytes, item) -> np.ndarray | None:
    _key, typ, size, rep, ps, pe = item
    code = _GPMF_DTYPES.get(chr(typ))
    if code is None:
        return None
    isz = np.dtype(code).itemsize
    if size == 0 or size % isz:
        return None
    arr = np.frombuffer(buf, dtype=code, count=(pe - ps) // isz, offset=ps)
    return arr.reshape(rep, size // isz).astype(np.float64)


def _gpmf_assemble(blocks: list, rate_hz: float | None, name: str):
    counts = [b[1].shape[0] for b in blocks]
    stmps = [b[0] for b in blocks]
    has = [s is not None for s in stmps]
    values = np.concatenate([b[1] for b in blocks], axis=0)
    if rate_hz is not None:
        rate = rate_hz
    if any(has) and not all(has):
        raise ValueError(f"{name}: STMP present in only some blocks")
    if all(has):
        if rate_hz is None:
            if len(blocks) < 2:
                raise ValueError(f"{name}: a single block needs rate_hz")
            span = (stmps[-1] - stmps[0]) / 1e6
            if not span > 0:
                raise ValueError(f"{name}: STMP does not advance")
            rate = float(sum(counts[:-1])) / span
        t = np.concatenate([s / 1e6 + np.arange(c) / rate for s, c in zip(stmps, counts)])
    else:
        if rate_hz is None:
            raise ValueError(f"{name}: no STMP, rate_hz is required")
        t = np.arange(values.shape[0]) / rate
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"{name}: invalid rate")
    return Series(t, values), float(rate)


def parse_gpmf(data: bytes, *, rate_hz: float | None = None, device: str = "gopro") -> RawImu:
    """GoPro GPMF 바이너리에서 GYRO / ACCL 을 읽는다(축 치환 없음, SCAL 로 나눔)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    if rate_hz is not None:
        rate_hz = _finite_float(rate_hz, "rate_hz")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
    buf = bytes(data)
    top, truncated = _klv_items(buf, 0, len(buf))
    blocks = {"GYRO": [], "ACCL": []}
    for key, typ, _size, _rep, ps, pe in top:
        if key != b"DEVC" or typ != 0:
            continue
        children, tr = _klv_items(buf, ps, pe)
        truncated = truncated or tr
        for ckey, ctyp, _s, _r, cps, cpe in children:
            if ckey != b"STRM" or ctyp != 0:
                continue
            fields, tr2 = _klv_items(buf, cps, cpe)
            truncated = truncated or tr2
            stmp = None
            scal = None
            sens = None
            skey = None
            for it in fields:
                k = it[0]
                if k == b"STMP":
                    arr = _gpmf_decode(buf, it)
                    if arr is not None and arr.size:
                        stmp = float(arr.ravel()[0])
                elif k == b"SCAL":
                    arr = _gpmf_decode(buf, it)
                    if arr is not None and arr.size:
                        scal = arr.ravel()
                elif k in (b"GYRO", b"ACCL"):
                    arr = _gpmf_decode(buf, it)
                    if arr is not None and arr.shape[1] == 3 and arr.shape[0] > 0:
                        sens, skey = arr, k.decode()
            if sens is None:
                continue
            if scal is not None:
                if scal.size not in (1, 3):
                    raise ValueError("SCAL must have 1 or 3 values")
                if np.any(scal == 0):
                    raise ValueError("SCAL must be non-zero")
                sens = sens / scal
            blocks[skey].append((stmp, sens))
    if not blocks["GYRO"]:
        raise ValueError("no GYRO stream found")
    gyro, rg = _gpmf_assemble(blocks["GYRO"], rate_hz, "GYRO")
    meta = {"truncated": bool(truncated), "gyro_rate_hz": rg}
    accel = None
    if blocks["ACCL"]:
        accel, ra = _gpmf_assemble(blocks["ACCL"], rate_hz, "ACCL")
        meta["accel_rate_hz"] = ra
    return RawImu(gyro, accel, "gpmf", device, meta)


# ---------------------------------------------------------------------------- CSV
_TIME_DIV = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}


def _read_csv_text(source: Any) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8-sig")
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source
        return Path(source).read_text(encoding="utf-8-sig")
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8-sig")
    if hasattr(source, "read"):
        x = source.read()
        return x.decode("utf-8-sig") if isinstance(x, (bytes, bytearray)) else str(x)
    raise TypeError("unsupported CSV source")


def parse_phone_csv(source, *, sensor: str = "both", fmt: str = "auto", time_unit: str | None = None,
                    gyro_unit: str = "rad/s", accel_unit: str = "m/s2", columns: dict | None = None,
                    device: str = "phone") -> RawImu:
    """폰 / TUM VI / 일반 CSV 를 RawImu 로. 구분자는 쉼표."""
    if sensor not in ("both", "gyro", "accel"):
        raise ValueError(f"unknown sensor {sensor!r}")
    if fmt not in ("auto", "tumvi", "generic"):
        raise ValueError(f"unknown fmt {fmt!r}")
    if time_unit is not None and time_unit not in _TIME_DIV:
        raise ValueError(f"unknown time_unit {time_unit!r}")
    if gyro_unit not in ("rad/s", "deg/s"):
        raise ValueError(f"unknown gyro_unit {gyro_unit!r}")
    if accel_unit not in ("m/s2", "g"):
        raise ValueError(f"unknown accel_unit {accel_unit!r}")
    text = _read_csv_text(source)
    lines = text.splitlines()
    k = 0
    while k < len(lines) and not lines[k].strip():
        k += 1
    if k >= len(lines):
        raise ValueError("empty CSV")
    head_raw = lines[k].strip()
    if fmt == "auto":
        fmt = "tumvi" if head_raw.lower().startswith("#timestamp [ns]") else "generic"
    header = [h.strip() for h in head_raw.lstrip("#").split(",")]
    rows = []
    for ln in lines[k + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([c.strip() for c in s.split(",")])
    if len(rows) < 2:
        raise ValueError("CSV needs at least 2 data rows")

    if fmt == "tumvi":
        idx_t, idx_g, idx_a = 0, [1, 2, 3], [4, 5, 6]
        div = 1e9
        source_name = "tumvi"
    else:
        low = [h.lower() for h in header]
        cmap = {"t": None, "gx": "gx", "gy": "gy", "gz": "gz", "ax": "ax", "ay": "ay", "az": "az"}
        if columns:
            for kk, vv in columns.items():
                if kk not in cmap:
                    raise ValueError(f"unknown column key {kk!r}")
                cmap[kk] = str(vv)

        def find(names):
            for nm in names:
                if nm.lower() in low:
                    return low.index(nm.lower())
            return None

        idx_t = find([cmap["t"]] if cmap["t"] else ["t", "time", "timestamp"])
        if idx_t is None:
            raise ValueError("time column not found")
        gi = [find([cmap[c]]) for c in ("gx", "gy", "gz")]
        ai = [find([cmap[c]]) for c in ("ax", "ay", "az")]
        idx_g = gi if all(i is not None for i in gi) else None
        idx_a = ai if all(i is not None for i in ai) else None
        if sensor != "accel" and idx_g is None:
            raise ValueError("gyro columns not found")
        if sensor == "accel" and idx_a is None:
            raise ValueError("accel columns not found")
        if sensor == "both" and idx_a is None and any(i is not None for i in ai):
            raise ValueError("accel columns incomplete")
        div = _TIME_DIV[time_unit or "s"]
        source_name = "csv"

    need = [idx_t] + ([] if idx_g is None else list(idx_g)) + ([] if idx_a is None else list(idx_a))
    if fmt == "tumvi":
        need = list(range(7))
    top = max(need)
    try:
        if any(len(r) <= top for r in rows):
            raise ValueError("row has too few columns")
        if fmt == "tumvi":
            t_raw = []
            for r in rows:
                try:
                    t_raw.append(int(r[idx_t]) / div)
                except ValueError:
                    t_raw.append(float(r[idx_t]) / div)
            t = np.array(t_raw, dtype=np.float64)
        else:
            t = np.array([float(r[idx_t]) for r in rows], dtype=np.float64) / div

        def grab(idxs):
            return np.array([[float(r[i]) for i in idxs] for r in rows], dtype=np.float64)

        gyro_v = grab(idx_g) if (sensor != "accel" and idx_g is not None) else None
        accel_v = grab(idx_a) if (sensor != "gyro" and idx_a is not None) else None
    except ValueError as e:
        raise ValueError(f"bad CSV value: {e}") from None
    if not np.all(np.isfinite(t)):
        raise ValueError("non-finite time")
    for v in (gyro_v, accel_v):
        if v is not None and not np.all(np.isfinite(v)):
            raise ValueError("non-finite sensor value")
    if gyro_v is not None and gyro_unit == "deg/s":
        gyro_v = np.deg2rad(gyro_v)
    if accel_v is not None and accel_unit == "g":
        accel_v = accel_v * G_STD
    gs = Series(t, gyro_v) if gyro_v is not None else None
    ac = Series(t, accel_v) if accel_v is not None else None
    return RawImu(gs, ac, source_name, device, {})


def merge_raw(gyro_raw: RawImu, accel_raw: RawImu) -> RawImu:
    """자이로는 앞의 것, 가속도는 뒤의 것. source/device/meta 는 앞의 것."""
    if not isinstance(gyro_raw, RawImu) or not isinstance(accel_raw, RawImu):
        raise TypeError("merge_raw needs RawImu inputs")
    if gyro_raw.gyro is None:
        raise ValueError("gyro_raw has no gyro")
    if accel_raw.accel is None:
        raise ValueError("accel_raw has no accel")
    return RawImu(gyro_raw.gyro, accel_raw.accel, gyro_raw.source, gyro_raw.device, gyro_raw.meta)


# ---------------------------------------------------------------------------- 균일화
def _need_raw(raw: Any) -> None:
    if not isinstance(raw, RawImu):
        raise TypeError("raw must be a RawImu")


def split_segments(raw: RawImu, max_gap_s: float = MAX_GAP_S) -> list:
    """간격 > max_gap_s 인 틈의 중점에서 두 센서를 함께 자른다. 표본 2개 미만 조각은 버린다."""
    _need_raw(raw)
    max_gap_s = _finite_float(max_gap_s, "max_gap_s")
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be > 0")
    cuts = []
    for s in (raw.gyro, raw.accel):
        if s is None:
            continue
        d = np.diff(s.t)
        for i in np.nonzero(d > max_gap_s)[0]:
            cuts.append(0.5 * (s.t[i] + s.t[i + 1]))
    cuts = np.unique(np.array(cuts, dtype=np.float64))

    def split(s):
        if s is None:
            return [None] * (len(cuts) + 1)
        idx = np.concatenate([[0], np.searchsorted(s.t, cuts, side="left"), [s.t.shape[0]]])
        return [(s.t[idx[j]:idx[j + 1]], s.v[idx[j]:idx[j + 1]]) for j in range(len(idx) - 1)]

    gp, ap = split(raw.gyro), split(raw.accel)
    out = []
    for gj, aj in zip(gp, ap):
        if gj is not None and gj[0].shape[0] < 2:
            continue
        if aj is not None and aj[0].shape[0] < 2:
            continue
        out.append(RawImu(None if gj is None else Series(*gj), None if aj is None else Series(*aj),
                          raw.source, raw.device, raw.meta))
    return out


def _antialias(t: np.ndarray, v: np.ndarray, rate_hz: float) -> np.ndarray:
    med = float(np.median(np.diff(t)))
    orig = 1.0 / med
    if not orig > rate_hz * (1.0 + 1e-4):
        return v
    sos = signal.butter(8, 0.4 * rate_hz, btype="low", fs=orig, output="sos")
    n = v.shape[0]
    padlen = min(n - 1, 3 * (2 * sos.shape[0] + 1))
    if padlen < 1:
        return v
    return signal.sosfiltfilt(sos, v, axis=0, padlen=padlen)


def resample_uniform(raw: RawImu, rate_hz: float = RATE_HZ, *, max_gap_s: float = MAX_GAP_S,
                     t0: float | None = None) -> ImuClip:
    """두 센서의 겹치는 구간을 3차 스플라인으로 균일 격자에 올린다(필요 시 앨리어싱 방지 저역)."""
    _need_raw(raw)
    rate_hz = _finite_float(rate_hz, "rate_hz")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    max_gap_s = _finite_float(max_gap_s, "max_gap_s")
    if raw.gyro is None or raw.accel is None:
        raise ValueError("resample_uniform needs both gyro and accel")
    g, a = raw.gyro, raw.accel
    for s in (g, a):
        if np.max(np.diff(s.t)) > max_gap_s:
            raise ValueError("gap larger than max_gap_s; use split_segments")
    lo = max(g.t[0], a.t[0])
    hi = min(g.t[-1], a.t[-1])
    if hi < lo:
        raise ValueError("gyro and accel do not overlap")
    if t0 is None:
        start = float(lo)
    else:
        start = _finite_float(t0, "t0")
        if start < lo - 1e-12 or start > hi + 1e-12:
            raise ValueError("t0 outside the overlap")
    n = int(math.floor((hi - start) * rate_hz + 1e-9)) + 1
    if n < 1:
        raise ValueError("empty overlap")
    grid = start + np.arange(n) / rate_hz
    outs = []
    for s in (g, a):
        v = _antialias(s.t, s.v, rate_hz)
        cs = CubicSpline(s.t, v, axis=0)
        outs.append(cs(np.clip(grid, s.t[0], s.t[-1])))
    return ImuClip(start, rate_hz, outs[0], outs[1], raw.source, raw.device, raw.meta)


# ---------------------------------------------------------------------------- 정렬
@dataclasses.dataclass(frozen=True)
class AlignEstimate:
    offset_s: float
    score: float
    lag_samples: int


def estimate_offset(ref: ImuClip, other: ImuClip, *, max_lag_s: float = 2.0,
                    min_overlap_s: float = 5.0) -> AlignEstimate:
    """자이로 노름 상호상관으로 t_ref = t_other + offset_s 를 추정한다(부표본)."""
    _need_clip(ref, "ref")
    _need_clip(other, "other")
    if abs(ref.rate_hz - other.rate_hz) > 1e-9 * max(ref.rate_hz, other.rate_hz):
        raise ValueError("clips must share rate_hz; resample first")
    max_lag_s = _finite_float(max_lag_s, "max_lag_s")
    if max_lag_s <= 0:
        raise ValueError("max_lag_s must be > 0")
    min_overlap_s = _finite_float(min_overlap_s, "min_overlap_s")
    if min_overlap_s < 0:
        raise ValueError("min_overlap_s must be >= 0")
    rate = ref.rate_hz
    nr, no = ref.n, other.n
    xr = np.linalg.norm(ref.gyro, axis=1)
    xo = np.linalg.norm(other.gyro, axis=1)
    xr = xr - xr.mean()
    xo = xo - xo.mean()
    dt0 = ref.t0 - other.t0
    lmin = int(math.ceil((-max_lag_s - dt0) * rate - 1e-9))
    lmax = int(math.floor((max_lag_s - dt0) * rate + 1e-9))
    lmin, lmax = max(lmin, -(no - 1)), min(lmax, nr - 1)
    need = max(2, int(math.ceil(min_overlap_s * rate - 1e-9)))
    if lmax < lmin:
        raise ValueError("no lag within max_lag_s")
    ls = np.arange(lmin, lmax + 1)
    i0 = np.maximum(0, -ls)
    i1 = np.minimum(no, nr - ls)
    m = i1 - i0
    ok = m >= need
    if not np.any(ok):
        raise ValueError("overlap shorter than min_overlap_s")
    ls, i0, i1, m = ls[ok], i0[ok], i1[ok], m[ok]
    cc = signal.correlate(xr, xo, mode="full", method="fft")
    sxy = cc[ls + (no - 1)]
    cx = np.concatenate([[0.0], np.cumsum(xo)])
    cxx = np.concatenate([[0.0], np.cumsum(xo * xo)])
    cy = np.concatenate([[0.0], np.cumsum(xr)])
    cyy = np.concatenate([[0.0], np.cumsum(xr * xr)])
    sx, sxx = cx[i1] - cx[i0], cxx[i1] - cxx[i0]
    sy, syy = cy[i1 + ls] - cy[i0 + ls], cyy[i1 + ls] - cyy[i0 + ls]
    cov = sxy - sx * sy / m
    vx = np.maximum(sxx - sx * sx / m, 0.0)
    vy = np.maximum(syy - sy * sy / m, 0.0)
    den = np.sqrt(vx * vy)
    r = np.where(den > 1e-12, cov / np.where(den > 1e-12, den, 1.0), 0.0)
    r = np.clip(r, -1.0, 1.0)
    b = int(np.argmax(r))
    delta = 0.0
    score = float(r[b])
    if 0 < b < len(r) - 1:
        ym, y0, yp = r[b - 1], r[b], r[b + 1]
        d2 = ym - 2.0 * y0 + yp
        if d2 < -1e-15:
            delta = float(np.clip(0.5 * (ym - yp) / d2, -0.5, 0.5))
            score = float(min(1.0, y0 - 0.25 * (ym - yp) * delta))
    lag = int(ls[b])
    return AlignEstimate(float(dt0 + (lag + delta) / rate), score, lag)


def shift_clip(clip: ImuClip, offset_s: float) -> ImuClip:
    _need_clip(clip, "clip")
    return _clip_like(clip, t0=clip.t0 + _finite_float(offset_s, "offset_s"))


def align_pair(ref: ImuClip, other: ImuClip, *, max_lag_s: float = 2.0):
    """other 를 ref 격자에 올리고 겹치는 구간으로 같은 t0/n 으로 자른다."""
    est = estimate_offset(ref, other, max_lag_s=max_lag_s)
    rate = ref.rate_hz
    off = est.offset_s
    o_start = other.t0 + off
    o_end = other.t0 + (other.n - 1) / rate + off
    i0 = max(0, int(math.ceil((o_start - ref.t0) * rate - 1e-6)))
    i1 = min(ref.n - 1, int(math.floor((o_end - ref.t0) * rate + 1e-6)))
    if i1 < i0:
        raise ValueError("clips do not overlap after alignment")
    ref_cut = ref.slice(i0, i1 + 1)
    t_other = np.clip(ref_cut.t - off, other.t0, other.t0 + (other.n - 1) / rate)
    x = other.t0 + np.arange(other.n) / rate
    stack = np.concatenate([other.gyro, other.accel], axis=1)
    out = CubicSpline(x, stack, axis=0)(t_other) if other.n >= 2 else np.repeat(stack, ref_cut.n, axis=0)
    oa = ImuClip(ref_cut.t0, rate, out[:, :3], out[:, 3:], other.source, other.device, other.meta)
    return ref_cut, oa, est


# ---------------------------------------------------------------------------- 증강 기본 연산
def rotate_frame(clip: ImuClip, R) -> ImuClip:
    """자이로·가속도 모두 x @ R.T (x_new = R x)."""
    _need_clip(clip, "clip")
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError("R must be a finite (3,3) matrix")
    if np.max(np.abs(R @ R.T - np.eye(3))) > 1e-6 or abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise ValueError("R is not a rotation")
    return _clip_like(clip, gyro=clip.gyro @ R.T, accel=clip.accel @ R.T)


def gravity_estimate(clip: ImuClip, fc_hz: float = GRAVITY_LP_HZ) -> np.ndarray:
    """가속도의 영위상 2차 저역 통과(중력 추정), (N,3)."""
    _need_clip(clip, "clip")
    fc_hz = _finite_float(fc_hz, "fc_hz")
    if not 0 < fc_hz < 0.5 * clip.rate_hz:
        raise ValueError("fc_hz must be in (0, rate/2)")
    n = clip.n
    if n < 2:
        return clip.accel.copy()
    sos = signal.butter(2, fc_hz, btype="low", fs=clip.rate_hz, output="sos")
    padlen = min(n - 1, int(3 * clip.rate_hz / fc_hz))
    return signal.sosfiltfilt(sos, clip.accel, axis=0, padlen=padlen)


def warped_length(n: int, scale: float) -> int:
    return int(math.floor((n - 1) / scale + 1e-9)) + 1


def time_warp(clip: ImuClip, scale: float) -> ImuClip:
    """운동을 scale 배 빠르게: w'(t)=s w(st), 동적 가속도 s^2 배, 중력 유지."""
    _need_clip(clip, "clip")
    s = _finite_float(scale, "scale")
    if s <= 0:
        raise ValueError("scale must be > 0")
    n = clip.n
    m = warped_length(n, s)
    if n < 2:
        return _clip_like(clip, gyro=clip.gyro * s)
    grav = gravity_estimate(clip)
    stack = np.concatenate([clip.gyro, clip.accel - grav, grav], axis=1)
    u = np.minimum(s * np.arange(m), n - 1)
    out = CubicSpline(np.arange(n, dtype=np.float64), stack, axis=0)(u)
    return _clip_like(clip, gyro=s * out[:, :3], accel=out[:, 6:9] + s * s * out[:, 3:6])


def rig_filter(clip: ImuClip, *, kind: str, fn_hz: float, zeta: float, gain: float = 1.0) -> ImuClip:
    """리그 전달함수(2차계, 쌍선형 변환, 인과적, 첫 표본 직류 정상상태 초기화)."""
    _need_clip(clip, "clip")
    if kind not in RIG_KINDS:
        raise ValueError(f"kind must be one of {RIG_KINDS}")
    fn_hz = _finite_float(fn_hz, "fn_hz")
    zeta = _finite_float(zeta, "zeta")
    gain = _finite_float(gain, "gain")
    fs = clip.rate_hz
    if not 0 < fn_hz < 0.45 * fs:
        raise ValueError("fn_hz must be in (0, 0.45*rate)")
    if zeta <= 0 or gain <= 0:
        raise ValueError("zeta and gain must be > 0")
    wn = 2.0 * fs * math.tan(math.pi * fn_hz / fs)           # 쌍선형 변환의 자연 주파수 사전 보정
    num = [0.0, 0.0, wn * wn] if kind == "lowpass" else [1.0, 0.0, 0.0]
    den = [1.0, 2.0 * zeta * wn, wn * wn]
    b, a = signal.bilinear(num, den, fs)
    zi = signal.lfilter_zi(b, a)

    def run(x):
        y, _ = signal.lfilter(b, a, x, axis=0, zi=zi[:, None] * x[0][None, :])
        return gain * y

    grav = gravity_estimate(clip)
    return _clip_like(clip, gyro=run(clip.gyro), accel=grav + run(clip.accel - grav))


def perturb_sensor(clip: ImuClip, *, seed: int, gyro_sigma: float = 0.0, accel_sigma: float = 0.0,
                   gyro_bias=(0.0, 0.0, 0.0), accel_bias=(0.0, 0.0, 0.0)) -> ImuClip:
    """x + bias + sigma * N(0,1). RNG = default_rng(SeedSequence([seed, 0xCA06])), 자이로 -> 가속도 순."""
    _need_clip(clip, "clip")
    gs = _finite_float(gyro_sigma, "gyro_sigma")
    as_ = _finite_float(accel_sigma, "accel_sigma")
    if gs < 0 or as_ < 0:
        raise ValueError("sigma must be >= 0")
    biases = []
    for nm, bv in (("gyro_bias", gyro_bias), ("accel_bias", accel_bias)):
        arr = np.asarray(bv, dtype=np.float64)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{nm} must be 3 finite numbers")
        biases.append(arr)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be an int >= 0")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SALT]))
    ng = rng.standard_normal(clip.gyro.shape)
    na = rng.standard_normal(clip.accel.shape)
    return _clip_like(clip, gyro=clip.gyro + biases[0] + gs * ng, accel=clip.accel + biases[1] + as_ * na)


# ---------------------------------------------------------------------------- 증강 설정 · 표본
@dataclasses.dataclass(frozen=True, kw_only=True)
class AugmentConfig:
    rot_deg_max: float = 30.0
    time_scale: tuple = (0.85, 1.18)
    rig_prob: float = 0.6
    rig_kinds: tuple = ("lowpass", "highpass")
    rig_fn_hz: tuple = (1.0, 20.0)
    rig_zeta: tuple = (0.3, 1.2)
    rig_gain: tuple = (0.5, 1.5)
    gyro_sigma: tuple = (0.0, 0.02)
    accel_sigma: tuple = (0.0, 0.1)
    gyro_bias_max: float = 0.01
    accel_bias_max: float = 0.05

    def __post_init__(self):
        rd = _finite_float(self.rot_deg_max, "rot_deg_max")
        if rd < 0:
            raise ValueError("rot_deg_max must be >= 0")
        _set(self, "rot_deg_max", rd)
        rp = _finite_float(self.rig_prob, "rig_prob")
        if not 0.0 <= rp <= 1.0:
            raise ValueError("rig_prob must be in [0, 1]")
        _set(self, "rig_prob", rp)
        kinds = tuple(self.rig_kinds)
        if not kinds or any(k not in RIG_KINDS for k in kinds):
            raise ValueError(f"rig_kinds must be a non-empty subset of {RIG_KINDS}")
        _set(self, "rig_kinds", kinds)
        for nm, strict_pos in (("time_scale", True), ("rig_fn_hz", True), ("rig_zeta", True), ("rig_gain", True),
                               ("gyro_sigma", False), ("accel_sigma", False)):
            lo, hi = _rng_pair(getattr(self, nm), nm)
            if strict_pos and lo <= 0:
                raise ValueError(f"{nm} lower bound must be > 0")
            if not strict_pos and lo < 0:
                raise ValueError(f"{nm} must be >= 0")
            _set(self, nm, (lo, hi))
        if self.rig_fn_hz[1] >= 0.45 * RATE_HZ:
            raise ValueError("rig_fn_hz upper bound must be < 0.45 * RATE_HZ")
        for nm in ("gyro_bias_max", "accel_bias_max"):
            v = _finite_float(getattr(self, nm), nm)
            if v < 0:
                raise ValueError(f"{nm} must be >= 0")
            _set(self, nm, v)


def _sample(cfg: AugmentConfig, ss: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(ss)
    d = rng.standard_normal(3)
    nd = float(np.linalg.norm(d))
    d = d / nd if nd > 0 else np.array([1.0, 0.0, 0.0])
    ang = math.radians(cfg.rot_deg_max) * float(rng.random())
    rotvec = [float(x) for x in ang * d]

    def uni(r):
        lo, hi = r
        return float(min(max(lo + (hi - lo) * float(rng.random()), lo), hi))

    ts = uni(cfg.time_scale)
    u_rig = float(rng.random())
    kind = cfg.rig_kinds[int(rng.integers(0, len(cfg.rig_kinds)))]
    rig = {"kind": kind, "fn_hz": uni(cfg.rig_fn_hz), "zeta": uni(cfg.rig_zeta), "gain": uni(cfg.rig_gain)}
    if not u_rig < cfg.rig_prob:
        rig = None
    nseed = int(rng.integers(0, 2 ** 31 - 1))
    gsig, asig = uni(cfg.gyro_sigma), uni(cfg.accel_sigma)
    gb = [float(x) for x in rng.uniform(-cfg.gyro_bias_max, cfg.gyro_bias_max, 3)]
    ab = [float(x) for x in rng.uniform(-cfg.accel_bias_max, cfg.accel_bias_max, 3)]
    return {"rotvec": rotvec, "time_scale": ts, "rig": rig,
            "noise": {"seed": nseed, "gyro_sigma": gsig, "accel_sigma": asig, "gyro_bias": gb, "accel_bias": ab},
            "crop": None}


def sample_augmentation(cfg: AugmentConfig | None = None, *, seed: int = 0) -> dict:
    """JSON 가능한 증강 기록을 결정론적으로 뽑는다."""
    cfg = AugmentConfig() if cfg is None else cfg
    if not isinstance(cfg, AugmentConfig):
        raise TypeError("cfg must be an AugmentConfig")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be an int >= 0")
    return _sample(cfg, np.random.SeedSequence([int(seed), _SALT]))


def _rodrigues(rv: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(rv))
    K = np.array([[0.0, -rv[2], rv[1]], [rv[2], 0.0, -rv[0]], [-rv[1], rv[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + (math.sin(th) / th) * K + ((1.0 - math.cos(th)) / (th * th)) * (K @ K)


_AUG_KEYS = {"rotvec", "time_scale", "rig", "noise", "crop"}
_RIG_KEYS = {"kind", "fn_hz", "zeta", "gain"}
_NOISE_KEYS = {"seed", "gyro_sigma", "accel_sigma", "gyro_bias", "accel_bias"}


def _vec3(x: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be 3 numbers") from None
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be 3 finite numbers")
    return arr


def apply_augmentation(clip: ImuClip, aug: dict) -> ImuClip:
    """회전 -> time_warp -> rig_filter -> perturb_sensor -> crop. 기록(aug)만으로 비트 재현."""
    _need_clip(clip, "clip")
    if not isinstance(aug, dict):
        raise ValueError("aug must be a dict")
    if set(aug) != _AUG_KEYS:
        raise ValueError(f"aug keys must be {sorted(_AUG_KEYS)}")
    rv = _vec3(aug["rotvec"], "rotvec")
    ts = _finite_float(aug["time_scale"], "time_scale")
    if ts <= 0:
        raise ValueError("time_scale must be > 0")
    rig = aug["rig"]
    if rig is not None:
        if not isinstance(rig, dict) or set(rig) != _RIG_KEYS:
            raise ValueError(f"rig must be None or have keys {sorted(_RIG_KEYS)}")
    nz = aug["noise"]
    if not isinstance(nz, dict) or set(nz) != _NOISE_KEYS:
        raise ValueError(f"noise must have keys {sorted(_NOISE_KEYS)}")
    nseed = nz["seed"]
    if isinstance(nseed, bool) or not isinstance(nseed, (int, np.integer)) or nseed < 0:
        raise ValueError("noise seed must be an int >= 0")
    gb, ab = _vec3(nz["gyro_bias"], "gyro_bias"), _vec3(nz["accel_bias"], "accel_bias")
    gsig, asig = _finite_float(nz["gyro_sigma"], "gyro_sigma"), _finite_float(nz["accel_sigma"], "accel_sigma")
    if gsig < 0 or asig < 0:
        raise ValueError("sigma must be >= 0")
    n_after = warped_length(clip.n, ts)
    crop = aug["crop"]
    if crop is not None:
        try:
            st, ln = crop
        except (TypeError, ValueError):
            raise ValueError("crop must be [start, length]") from None
        for x in (st, ln):
            if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
                raise ValueError("crop entries must be ints")
        st, ln = int(st), int(ln)
        if st < 0 or ln < 1 or st + ln > n_after:
            raise ValueError("crop out of range")

    x = rotate_frame(clip, _rodrigues(rv))
    if ts != 1.0:
        x = time_warp(x, ts)
    if rig is not None:
        x = rig_filter(x, kind=rig["kind"], fn_hz=rig["fn_hz"], zeta=rig["zeta"], gain=rig["gain"])
    x = perturb_sensor(x, seed=int(nseed), gyro_sigma=gsig, accel_sigma=asig, gyro_bias=gb, accel_bias=ab)
    if crop is not None:
        x = x.slice(st, st + ln)
    meta = dict(clip.meta)
    meta["aug"] = copy.deepcopy(aug)
    return _clip_like(x, meta=meta)


def augment_clip(clip: ImuClip, cfg: AugmentConfig | None = None, *, seed: int = 0) -> ImuClip:
    return apply_augmentation(clip, sample_augmentation(cfg, seed=seed))


def augment_dataset(clips, *, factor: float = AUG_FACTOR, cfg: AugmentConfig | None = None, seed: int = 0) -> list:
    """원본 전부(순서대로) 다음에 출처별 변형. 출처 i 의 변형 길이 합 = floor((factor-1) n_i + 0.5)."""
    factor = _finite_float(factor, "factor")
    if factor < 1.0:
        raise ValueError("factor must be >= 1")
    cfg = AugmentConfig() if cfg is None else cfg
    if not isinstance(cfg, AugmentConfig):
        raise TypeError("cfg must be an AugmentConfig")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be an int >= 0")
    seed = int(seed)
    clips = list(clips)
    for c in clips:
        _need_clip(c, "clips[i]")
    outs = []
    for i, c in enumerate(clips):
        meta = dict(c.meta)
        meta.update({"origin": i, "variant": 0, "aug": None})
        outs.append(_clip_like(c, meta=meta))
    for i, c in enumerate(clips):
        remaining = int(math.floor((factor - 1.0) * c.n + 0.5))
        v = 0
        while remaining > 0:
            v += 1
            ss = np.random.SeedSequence([seed, i, v, _SALT])
            aug = _sample(cfg, ss)
            wl = warped_length(c.n, aug["time_scale"])
            if wl > remaining:
                crng = np.random.default_rng(np.random.SeedSequence([seed, i, v, _SALT, 1]))
                aug["crop"] = [int(crng.integers(0, wl - remaining + 1)), int(remaining)]
                take = remaining
            else:
                take = wl
            o = apply_augmentation(c, aug)
            meta = dict(o.meta)
            meta.update({"origin": i, "variant": v})
            outs.append(_clip_like(o, meta=meta))
            remaining -= take
    return outs


def make_windows(clip: ImuClip, *, length: int = WINDOW_LEN, hop: int | None = None) -> np.ndarray:
    """(M, length, 6) float32 윈도(자이로 3 | 가속도 3). M = (n-length)//hop + 1, n<length 면 0."""
    _need_clip(clip, "clip")
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("length must be an int >= 1")
    length = int(length)
    if hop is None:
        hop = max(1, length // 2)
    if isinstance(hop, bool) or not isinstance(hop, (int, np.integer)) or hop < 1:
        raise ValueError("hop must be an int >= 1")
    hop = int(hop)
    n = clip.n
    if n < length:
        return np.zeros((0, length, 6), dtype=np.float32)
    m = (n - length) // hop + 1
    idx = hop * np.arange(m)[:, None] + np.arange(length)[None, :]
    return clip.as_array()[idx].astype(np.float32)

