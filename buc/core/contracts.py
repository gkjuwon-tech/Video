"""FND-02 데이터 계약 v0: World / ReadPacket / Obs / Shot / Theta / KPI / StageIO (+ Prov, InputImage, LDILayer,
FarField, Frames).

스테이지 사이 모든 이음매의 단일 원천이다. numpy 와 표준 라이브러리만 쓴다(torch 금지, CPU import).

공통 규약
- 좌표: 세계 오른손·미터. 카메라는 OpenCV(x 오른쪽, y 아래, z 앞). 픽셀 중심이 정수 좌표이므로 영상 중심 주점은
  ((W-1)/2, (H-1)/2).
- 영상은 (H,W,3) float32 장면 선형(Rec.709 원색). 깊이는 광축 z-깊이(m) float32, >0, 하늘은 +inf 허용, NaN 금지.
- 모든 계약 클래스는 ``@dataclass(kw_only=True)``. ``__post_init__`` 에서 배열을 정본 dtype·C-연속으로 바꾼 뒤
  ``validate()`` 를 부르고, 위반은 ``ContractError`` 로 보고한다.
- 해시는 ``content_hash`` 한 곳에서만 만든다(v1에서 FND-03 blake3 로 교체 가능).
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import math
import numbers
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "CONTRACT_VERSION", "LEVELS", "TAINT_BASES", "STAGES", "STAGE_SIGNATURES", "ContractError",
    "ContractVersionError", "Prov", "Theta", "Obs", "InputImage", "ReadPacket", "LDILayer", "FarField", "World",
    "Shot", "Frames", "KPI", "StageIO", "content_hash", "dumps", "loads", "check_stage_io", "make_stage_io",
]

CONTRACT_VERSION = "0.1.0"
LEVELS = ("L1", "L1.5", "L2a", "L2b")
TAINT_BASES = {"USER", "EXTGEN", "CAPTURE", "ASSET", "ENGINE", "SI", "HOLDOUT", "COMP"}
STAGES = ("S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S8.5", "S9-10", "S11")
COLOR_SPACES = ("srgb", "display-p3", "rec2020", "linear-srgb", "adobe-rgb", "unknown")
IMPL_KINDS = ("stub", "real")
DYN_KEYS = {"blur_vel", "support_graph", "matter", "artic_slots"}  # P2 예약
WORLD_SCHEMA = "STRATA-lite-min/v0"

_ROT_ATOL = 1e-6
_WRAP = (TypeError, ValueError, OverflowError, KeyError, AttributeError)


class ContractError(ValueError):
    """계약 위반."""


class ContractVersionError(ContractError):
    """주 버전이 다른 직렬화 데이터."""


# ---------------------------------------------------------------------------------------------------------------
# 정본 해시
# ---------------------------------------------------------------------------------------------------------------
def _p(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "little") + payload


def _le_dtype(a: np.ndarray) -> np.dtype:
    return a.dtype.newbyteorder("<")


def _canon(x: Any) -> bytes:
    """값 -> 모호성 없는 바이트열(타입 태그 + 길이 접두). 배열 본문은 sha256 다이제스트로 대신한다."""
    if x is None:
        return b"N"
    if isinstance(x, (bool, np.bool_)):
        return b"T" if bool(x) else b"F"
    if isinstance(x, (int, np.integer)):
        return _p(b"I", str(int(x)).encode("ascii"))
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if f != f:
            return _p(b"D", b"\x00\x00\x00\x00\x00\x00\xf8\x7f")
        return _p(b"D", struct.pack("<d", f))
    if isinstance(x, str):
        return _p(b"S", x.encode("utf-8"))
    if isinstance(x, (bytes, bytearray)):
        return _p(b"Y", bytes(x))
    if isinstance(x, np.ndarray):
        if x.dtype.kind not in "biuf":
            raise ContractError(f"content_hash: 지원하지 않는 배열 dtype {x.dtype}")
        a = np.ascontiguousarray(x)
        dt = _le_dtype(a)
        if a.dtype != dt:
            a = a.astype(dt)
        hh = hashlib.sha256()
        hh.update(a.reshape(-1).view(np.uint8))
        meta = _canon(dt.str) + _canon(list(a.shape)) + _p(b"B", hh.digest())
        return _p(b"A", meta)
    if isinstance(x, (list, tuple)):
        return _p(b"L", len(x).to_bytes(8, "little") + b"".join(_canon(e) for e in x))
    if isinstance(x, (set, frozenset)):
        return _p(b"E", len(x).to_bytes(8, "little") + b"".join(sorted(_canon(e) for e in x)))
    if isinstance(x, dict):
        pairs = sorted((_canon(k), _canon(v)) for k, v in x.items())
        return _p(b"M", len(pairs).to_bytes(8, "little") + b"".join(k + v for k, v in pairs))
    if isinstance(x, _Contract):
        body = {f.name: getattr(x, f.name) for f in dataclasses.fields(x)}
        return _p(b"C", _canon(type(x).__name__) + _canon(body))
    raise ContractError(f"content_hash: 해시할 수 없는 타입 {type(x).__name__}")


def content_hash(x: Any) -> str:
    """정본 내용 해시(sha256 hex 64자). dict 키 순서·list/tuple 구분·메모리 배치·바이트 순서에 무관하고,
    int/float/bool 은 구분하며, 배열은 dtype(리틀엔디안)+shape+바이트로 비트 단위까지 본다."""
    return hashlib.sha256(_canon(x)).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# JSON 직렬화(배열은 비트 무손실)
# ---------------------------------------------------------------------------------------------------------------
def _enc(v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isfinite(f):
            return f
        return {"__float__": "nan" if f != f else ("inf" if f > 0 else "-inf")}
    if isinstance(v, np.ndarray):
        if v.dtype.kind not in "biuf":
            raise ContractError(f"직렬화할 수 없는 배열 dtype {v.dtype}")
        a = np.ascontiguousarray(v)
        dt = _le_dtype(a)
        if a.dtype != dt:
            a = a.astype(dt)
        return {"__ndarray__": base64.b64encode(a.tobytes()).decode("ascii"), "dtype": dt.str,
                "shape": [int(s) for s in a.shape]}
    if isinstance(v, (list, tuple)):
        return [_enc(e) for e in v]
    if isinstance(v, (set, frozenset)):
        return {"__set__": sorted((_enc(e) for e in v), key=lambda e: json.dumps(e, sort_keys=True))}
    if isinstance(v, dict):
        out = {}
        for k, e in v.items():
            if not isinstance(k, str):
                raise ContractError(f"dict 키는 문자열이어야 한다: {k!r}")
            out[k] = _enc(e)
        return out
    if isinstance(v, _Contract):
        return v.to_dict()
    raise ContractError(f"직렬화할 수 없는 타입 {type(v).__name__}")


def _dec(v: Any) -> Any:
    if isinstance(v, list):
        return [_dec(e) for e in v]
    if not isinstance(v, dict):
        return v
    if "__ndarray__" in v:
        try:
            dt = np.dtype(v["dtype"])
            shape = tuple(int(s) for s in v["shape"])
            raw = base64.b64decode(v["__ndarray__"], validate=True)
        except _WRAP + (binascii.Error,) as e:
            raise ContractError(f"깨진 __ndarray__: {e}") from e
        if dt.kind not in "biuf":
            raise ContractError(f"지원하지 않는 dtype {dt}")
        if len(raw) != int(np.prod(shape, dtype=np.int64)) * dt.itemsize:
            raise ContractError("__ndarray__ 바이트 길이가 shape 와 맞지 않는다")
        return np.frombuffer(raw, dtype=dt).reshape(shape).astype(dt.newbyteorder("="))
    if "__contract__" in v:
        cls = _REGISTRY.get(v["__contract__"])
        if cls is None:
            raise ContractError(f"알 수 없는 계약 타입 {v['__contract__']!r}")
        return cls.from_dict(v)
    if "__set__" in v and len(v) == 1:
        return frozenset(_hashable(_dec(e)) for e in v["__set__"])
    if "__float__" in v and len(v) == 1:
        return float(v["__float__"])
    return {k: _dec(e) for k, e in v.items()}


def _hashable(e: Any) -> Any:
    return tuple(_hashable(i) for i in e) if isinstance(e, list) else e


def _major(ver: Any) -> int:
    try:
        return int(str(ver).split(".")[0])
    except ValueError as e:
        raise ContractError(f"해석할 수 없는 버전 {ver!r}") from e


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _num(name: str, v: Any) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise ContractError(f"{name}: 숫자여야 한다 ({v!r})")
    f = float(v)
    if f != f:
        raise ContractError(f"{name}: NaN 금지")
    return f


def _opt_num(name: str, v: Any) -> Optional[float]:
    return None if v is None else _num(name, v)


def _rng(name: str, f: float, lo: float, hi: float, lo_open: bool = False, hi_open: bool = False) -> None:
    ok_lo = f > lo if lo_open else f >= lo
    ok_hi = f < hi if hi_open else f <= hi
    if not (ok_lo and ok_hi):
        raise ContractError(f"{name}={f} 범위 밖 {'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}")


def _int(name: str, v: Any, minv: Optional[int] = None) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral):
        raise ContractError(f"{name}: 정수여야 한다 ({v!r})")
    i = int(v)
    if minv is not None and i < minv:
        raise ContractError(f"{name}={i} < {minv}")
    return i


def _str(name: str, v: Any, nonempty: bool = False) -> str:
    if not isinstance(v, str):
        raise ContractError(f"{name}: 문자열이어야 한다 ({v!r})")
    if nonempty and not v:
        raise ContractError(f"{name}: 빈 문자열")
    return v


def _dict(name: str, v: Any) -> dict:
    if not isinstance(v, dict):
        raise ContractError(f"{name}: dict 여야 한다")
    return v


def _req(name: str, v: Any, cls: Any) -> Any:
    if not isinstance(v, cls):
        raise ContractError(f"{name}: {getattr(cls, '__name__', cls)} 여야 한다 ({type(v).__name__})")
    return v


def _arr(name: str, v: Any, dtype: Any) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype.kind not in "biuf":
        raise ContractError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.ascontiguousarray(a, dtype=dtype)


def _shape(name: str, a: np.ndarray, shape: Tuple[Optional[int], ...]) -> None:
    ok = a.ndim == len(shape) and all(s is None or s == d for s, d in zip(shape, a.shape))
    if not ok:
        raise ContractError(f"{name}: shape {a.shape} != {shape}")


def _finite(name: str, a: np.ndarray) -> None:
    if a.size and not np.isfinite(a).all():
        raise ContractError(f"{name}: 유한하지 않은 값(NaN/inf)")


def _no_nan(name: str, a: np.ndarray) -> None:
    if a.dtype.kind == "f" and a.size and np.isnan(a).any():
        raise ContractError(f"{name}: NaN 금지")


def _unit_range(name: str, a: np.ndarray) -> None:
    _finite(name, a)
    if a.size and (a.min() < 0 or a.max() > 1):
        raise ContractError(f"{name}: [0,1] 밖의 값")


def _check_rot(name: str, R: np.ndarray) -> None:
    """R: (...,3,3) 직교·det +1."""
    eye = np.eye(3)
    err = np.abs(np.einsum("...ji,...jk->...ik", R, R) - eye).max() if R.size else 0.0
    if err > _ROT_ATOL:
        raise ContractError(f"{name}: 직교가 아니다 (오차 {err:.3g})")
    if R.size and np.abs(np.linalg.det(R) - 1.0).max() > _ROT_ATOL:
        raise ContractError(f"{name}: det != +1")


def _check_K(name: str, K: np.ndarray) -> None:
    _finite(name, K)
    if np.abs(K[..., 2, :] - np.array([0.0, 0.0, 1.0])).max() > 1e-9:
        raise ContractError(f"{name}: 마지막 행이 [0,0,1] 이 아니다")
    if not (K[..., 0, 0] > 0).all() or not (K[..., 1, 1] > 0).all():
        raise ContractError(f"{name}: fx, fy > 0 이어야 한다")


# ---------------------------------------------------------------------------------------------------------------
# 기반 클래스
# ---------------------------------------------------------------------------------------------------------------
_REGISTRY: Dict[str, type] = {}


class _Contract:
    """모든 계약 클래스의 공통 동작."""

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        _REGISTRY[cls.__name__] = cls

    # 하위 클래스가 재정의
    def _coerce(self) -> None:  # pragma: no cover - 기본은 변환 없음
        pass

    def _validate(self) -> None:  # pragma: no cover
        pass

    def __post_init__(self) -> None:
        try:
            self._coerce()
        except ContractError:
            raise
        except _WRAP as e:
            raise ContractError(f"{type(self).__name__}: {e}") from e
        self.validate()

    def validate(self) -> None:
        try:
            self._validate()
        except ContractError:
            raise
        except _WRAP as e:
            raise ContractError(f"{type(self).__name__}: {e}") from e

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {"__contract__": type(self).__name__, "__v__": CONTRACT_VERSION}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            d[f.name] = _enc(getattr(self, f.name))
        return d

    @classmethod
    def from_dict(cls, d: dict):
        if not isinstance(d, dict):
            raise ContractError("from_dict: dict 가 필요하다")
        if d.get("__contract__") != cls.__name__:
            raise ContractError(f"{cls.__name__}.from_dict: 계약 타입 불일치 ({d.get('__contract__')!r})")
        if "__v__" not in d:
            raise ContractError("__v__ 없음")
        if _major(d["__v__"]) != _major(CONTRACT_VERSION):
            raise ContractVersionError(f"주 버전 불일치: {d['__v__']} vs {CONTRACT_VERSION}")
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        kwargs = {k: _dec(v) for k, v in d.items() if k in names}  # 같은 주 버전의 새 필드는 무시한다
        try:
            return cls(**kwargs)
        except ContractError:
            raise
        except TypeError as e:
            raise ContractError(f"{cls.__name__}.from_dict: {e}") from e

    def content_hash(self) -> str:
        return hashlib.sha256(_canon(self)).hexdigest()

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.content_hash() == other.content_hash()

    def __hash__(self) -> int:
        return int(self.content_hash()[:16], 16)


def dumps(obj: _Contract) -> str:
    if not isinstance(obj, _Contract):
        raise ContractError("dumps: 계약 객체가 아니다")
    return json.dumps(obj.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads(s: str) -> _Contract:
    d = json.loads(s)
    if not isinstance(d, dict) or "__contract__" not in d:
        raise ContractError("loads: __contract__ 없음")
    cls = _REGISTRY.get(d["__contract__"])
    if cls is None:
        raise ContractError(f"알 수 없는 계약 타입 {d['__contract__']!r}")
    return cls.from_dict(d)


# ---------------------------------------------------------------------------------------------------------------
# Prov
# ---------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True, eq=False)
class Prov(_Contract):
    """출처 오염(taint) 기록. taint 원소는 ``BASE`` 또는 ``BASE:<한정자>``."""
    taint: frozenset = frozenset()
    mode: str = "engine"
    roots: tuple = ()
    ns: str = "engine"

    def _coerce(self) -> None:
        if isinstance(self.taint, (str, bytes)):
            raise ContractError("taint: 문자열 집합이어야 한다")
        self.taint = frozenset(self.taint)
        if isinstance(self.roots, (str, bytes)):
            raise ContractError("roots: 문자열 시퀀스여야 한다")
        self.roots = tuple(sorted(set(self.roots)))

    def _validate(self) -> None:
        for t in self.taint:
            _str("taint 원소", t, True)
            base, sep, qual = t.partition(":")
            if base not in TAINT_BASES:
                raise ContractError(f"taint 기반 {base!r} 는 {sorted(TAINT_BASES)} 중 하나여야 한다")
            if sep and not qual:
                raise ContractError(f"taint {t!r}: 한정자가 비었다")
        for r in self.roots:
            _str("roots 원소", r)
        _str("mode", self.mode, True)
        _str("ns", self.ns, True)

    def _has(self, base: str) -> bool:
        return any(t.partition(":")[0] == base for t in self.taint)

    @property
    def ext(self) -> bool:
        return self._has("EXTGEN")

    @property
    def comp(self) -> bool:
        return self._has("COMP")

    @property
    def hold(self) -> bool:
        return self._has("HOLDOUT")

    @staticmethod
    def merge(*ps: "Prov") -> "Prov":
        for p in ps:
            _req("merge 인자", p, Prov)
        if not ps:
            return Prov()
        taint = frozenset().union(*(p.taint for p in ps))
        roots = tuple(sorted(set().union(*(p.roots for p in ps))))
        modes = {p.mode for p in ps}
        mode = ps[0].mode if len(modes) == 1 else "mixed"
        ns = "user_vault" if any(p.ns == "user_vault" for p in ps) else ps[0].ns
        return Prov(taint=taint, mode=mode, roots=roots, ns=ns)


# ---------------------------------------------------------------------------------------------------------------
# Theta (촬영 파라미터)
# ---------------------------------------------------------------------------------------------------------------
_INF = float("inf")
# 이름: (lo, hi, lo_open, hi_open)
_THETA_RANGES = {
    "aspect": (0.0, _INF, True, True),
    "focal_mm": (1.0, 2000.0, True, False),
    "sensor_width_mm": (1.0, 100.0, True, False),
    "f_stop": (0.7, 64.0, False, False),
    "focus_m": (0.0, _INF, True, False),
    "shutter_deg": (0.0, 360.0, True, False),
    "fps": (0.0, 240.0, True, False),
    "nd_stops": (0.0, 15.0, False, False),
    "cpl": (0.0, 1.0, False, False),
    "ev": (-10.0, 10.0, False, False),
    "wb_kelvin": (1500.0, 20000.0, False, False),
    "wb_tint": (-1.0, 1.0, False, False),
    "grain": (0.0, 1.0, False, False),
    "halation": (0.0, 1.0, False, False),
    "vignette": (0.0, 1.0, False, False),
    "atmos_beta": (0.0, _INF, False, True),
    "rim": (0.0, 1.0, False, False),
}
_THETA_OPT = {"sun_elev_deg": (-90.0, 90.0, False, False), "sun_azim_deg": (0.0, 360.0, False, True)}


@dataclass(kw_only=True, eq=False)
class Theta(_Contract):
    aspect: float = 16 / 9
    focal_mm: float = 35
    sensor_width_mm: float = 24.89
    f_stop: float = 4
    focus_m: float = 3
    shutter_deg: float = 180
    fps: float = 24
    nd_stops: float = 0
    cpl: float = 0
    ev: float = 0
    wb_kelvin: float = 6500
    wb_tint: float = 0
    lut: str = "none"
    grain: float = 0
    halation: float = 0
    vignette: float = 0
    lens_rx: str = "spherical_clean"
    atmos_beta: float = 0
    sun_elev_deg: Optional[float] = None
    sun_azim_deg: Optional[float] = None
    rim: float = 0
    extra: dict = field(default_factory=dict)

    def _coerce(self) -> None:
        for n in _THETA_RANGES:
            setattr(self, n, _num(n, getattr(self, n)))
        for n in _THETA_OPT:
            setattr(self, n, _opt_num(n, getattr(self, n)))

    def _validate(self) -> None:
        for n, (lo, hi, lo_o, hi_o) in _THETA_RANGES.items():
            _rng(n, getattr(self, n), lo, hi, lo_o, hi_o)
        for n, (lo, hi, lo_o, hi_o) in _THETA_OPT.items():
            v = getattr(self, n)
            if v is not None:
                _rng(n, v, lo, hi, lo_o, hi_o)
        _str("lut", self.lut)
        _str("lens_rx", self.lens_rx)
        _dict("extra", self.extra)

    def fov_x_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_width_mm / (2.0 * self.focal_mm)))

    def intrinsics(self, W: int, H: int) -> np.ndarray:
        """(3,3) float64. fx=fy=focal/sensor*W, 주점은 영상 중심 ((W-1)/2, (H-1)/2)."""
        fx = self.focal_mm / self.sensor_width_mm * W
        return np.array([[fx, 0.0, (W - 1) / 2.0], [0.0, fx, (H - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------
# Obs (관측 카메라)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True, eq=False)
class Obs(_Contract):
    """관측 카메라. R, t 는 세계 -> 카메라(OpenCV). 왜곡은 x_d = x(1 + k1 r^2 + k2 r^4)."""
    K: Any
    R: Any
    t: Any
    width: int
    height: int
    k1: float = 0
    k2: float = 0
    mask: Any = None
    l_obs_hard: float = 0
    l_obs_part: float = 0

    def _coerce(self) -> None:
        self.K = _arr("K", self.K, np.float64)
        self.R = _arr("R", self.R, np.float64)
        self.t = _arr("t", self.t, np.float64)
        self.width = _int("width", self.width, 1)
        self.height = _int("height", self.height, 1)
        for n in ("k1", "k2", "l_obs_hard", "l_obs_part"):
            setattr(self, n, _num(n, getattr(self, n)))
        if self.mask is not None:
            self.mask = _arr("mask", self.mask, bool)

    def _validate(self) -> None:
        _shape("K", self.K, (3, 3))
        _shape("R", self.R, (3, 3))
        _shape("t", self.t, (3,))
        _check_K("K", self.K)
        _finite("R", self.R)
        _finite("t", self.t)
        _check_rot("R", self.R)
        _finite("k1", np.array([self.k1]))
        _finite("k2", np.array([self.k2]))
        if self.mask is not None:
            _shape("mask", self.mask, (self.height, self.width))
        if not (0.0 <= self.l_obs_hard <= self.l_obs_part) or not math.isfinite(self.l_obs_part):
            raise ContractError(f"0 <= l_obs_hard <= l_obs_part 위반 ({self.l_obs_hard}, {self.l_obs_part})")

    def w2c(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def c2w(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R.T
        M[:3, 3] = self.center()
        return M

    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """세계점 (N,3) -> (uv (N,2) 픽셀, z (N,) 카메라 z-깊이). z<=0 인 점의 uv 는 무의미하다."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            x = Xc[:, 0] / z
            y = Xc[:, 1] / z
        r2 = x * x + y * y
        f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        xd, yd = x * f, y * f
        K = self.K
        u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
        v = K[1, 1] * yd + K[1, 2]
        return np.stack([u, v], axis=1), z


# ---------------------------------------------------------------------------------------------------------------
# InputImage / ReadPacket
# ---------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True, eq=False)
class InputImage(_Contract):
    pixels: Any
    color_space: str
    prov: Prov
    exif: dict = field(default_factory=dict)
    consent: bool = True

    def _coerce(self) -> None:
        a = np.asarray(self.pixels)
        if a.dtype.byteorder not in ("=", "|"):
            a = a.astype(a.dtype.newbyteorder("="))
        self.pixels = np.ascontiguousarray(a)

    def _validate(self) -> None:
        a = self.pixels
        if a.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
            raise ContractError(f"pixels dtype {a.dtype}: uint8|uint16|float32 만 허용")
        if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] < 1 or a.shape[1] < 1:
            raise ContractError(f"pixels shape {a.shape}: (H,W,3) 이어야 한다")
        if a.dtype.kind == "f":
            _finite("pixels", a)
        if self.color_space not in COLOR_SPACES:
            raise ContractError(f"color_space {self.color_space!r} 는 {COLOR_SPACES} 중 하나여야 한다")
        _req("prov", self.prov, Prov)
        _dict("exif", self.exif)
        if not isinstance(self.consent, (bool, np.bool_)):
            raise ContractError("consent: bool 이어야 한다")

    @property
    def h_I(self) -> str:
        return content_hash(self.pixels)


@dataclass(kw_only=True, eq=False)
class ReadPacket(_Contract):
    """S-1 출력."""
    image_lin: Any
    depth: Any
    obs: Obs
    theta_in: Theta
    input_hash: str
    prov: Prov
    scale_s: float = 1
    sigma_s: float = 0
    level: str = "L1"
    depth_conf: Any = None
    normals: Any = None
    esc: dict = field(default_factory=dict)
    faces: tuple = ()
    lights: dict = field(default_factory=dict)
    dyn: dict = field(default_factory=dict)
    fallbacks: tuple = ()
    meta: dict = field(default_factory=dict)

    def _coerce(self) -> None:
        self.image_lin = _arr("image_lin", self.image_lin, np.float32)
        self.depth = _arr("depth", self.depth, np.float32)
        if self.depth_conf is not None:
            self.depth_conf = _arr("depth_conf", self.depth_conf, np.float32)
        if self.normals is not None:
            self.normals = _arr("normals", self.normals, np.float32)
        self.scale_s = _num("scale_s", self.scale_s)
        self.sigma_s = _num("sigma_s", self.sigma_s)
        self.faces = tuple(self.faces)
        self.fallbacks = tuple(self.fallbacks)

    def _validate(self) -> None:
        _req("obs", self.obs, Obs)
        _req("theta_in", self.theta_in, Theta)
        _req("prov", self.prov, Prov)
        _str("input_hash", self.input_hash, True)
        H, W = self.obs.height, self.obs.width
        _shape("image_lin", self.image_lin, (H, W, 3))
        _shape("depth", self.depth, (H, W))
        _finite("image_lin", self.image_lin)
        if self.image_lin.size and self.image_lin.min() < 0:
            raise ContractError("image_lin: 음수 값")
        _no_nan("depth", self.depth)
        if not (self.depth > 0).all():
            raise ContractError("depth: > 0 이어야 한다 (하늘은 +inf)")
        if self.depth_conf is not None:
            _shape("depth_conf", self.depth_conf, (H, W))
            _unit_range("depth_conf", self.depth_conf)
        if self.normals is not None:
            _shape("normals", self.normals, (H, W, 3))
            _finite("normals", self.normals)
            n = np.linalg.norm(self.normals.astype(np.float64), axis=-1)
            if np.abs(n - 1.0).max() > 1e-3:
                raise ContractError("normals: 단위벡터가 아니다")
        if not (math.isfinite(self.scale_s) and self.scale_s > 0):
            raise ContractError("scale_s > 0 이어야 한다")
        if not (math.isfinite(self.sigma_s) and self.sigma_s >= 0):
            raise ContractError("sigma_s >= 0 이어야 한다")
        if self.level not in LEVELS:
            raise ContractError(f"level {self.level!r} 는 {LEVELS} 중 하나여야 한다")
        for n in ("esc", "lights", "dyn", "meta"):
            _dict(n, getattr(self, n))
        bad = set(self.dyn) - DYN_KEYS
        if bad:
            raise ContractError(f"dyn 키 {sorted(bad)} 는 예약 목록 {sorted(DYN_KEYS)} 밖이다")
        for f in self.fallbacks:
            _str("fallbacks 원소", f)


# ---------------------------------------------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True, eq=False)
class LDILayer(_Contract):
    """직선(비전치) 알파 RGBA 층과 z-깊이."""
    rgba: Any
    depth: Any

    def _coerce(self) -> None:
        self.rgba = _arr("rgba", self.rgba, np.float32)
        self.depth = _arr("depth", self.depth, np.float32)

    def _validate(self) -> None:
        if self.rgba.ndim != 3 or self.rgba.shape[2] != 4:
            raise ContractError(f"rgba shape {self.rgba.shape}: (H,W,4) 이어야 한다")
        _shape("depth", self.depth, self.rgba.shape[:2])
        _unit_range("rgba", self.rgba)
        _no_nan("depth", self.depth)
        if not (self.depth > 0).all():
            raise ContractError("depth: > 0 이어야 한다")


@dataclass(kw_only=True, eq=False)
class FarField(_Contract):
    """팔면체 매핑 원경 환경(S x S x 3)."""
    rgb: Any

    def _coerce(self) -> None:
        self.rgb = _arr("rgb", self.rgb, np.float32)

    def _validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.shape[0] != self.rgb.shape[1]:
            raise ContractError(f"rgb shape {self.rgb.shape}: (S,S,3) 이어야 한다")
        _finite("rgb", self.rgb)
        if self.rgb.size and self.rgb.min() < 0:
            raise ContractError("rgb: 음수 값")


@dataclass(kw_only=True, eq=False)
class World(_Contract):
    """STRATA-lite-min v0: 관측 시점의 LDI 층 묶음 + 원경."""
    obs: Obs
    layers: tuple
    theta_in: Theta
    level: str
    read_hash: str
    prov: Prov
    far_field: Optional[FarField] = None
    meta: dict = field(default_factory=dict)
    schema: str = WORLD_SCHEMA

    def _coerce(self) -> None:
        self.layers = tuple(self.layers)

    def _validate(self) -> None:
        _req("obs", self.obs, Obs)
        _req("theta_in", self.theta_in, Theta)
        _req("prov", self.prov, Prov)
        if self.far_field is not None:
            _req("far_field", self.far_field, FarField)
        if self.schema != WORLD_SCHEMA:
            raise ContractError(f"schema {self.schema!r} != {WORLD_SCHEMA!r}")
        if self.level not in LEVELS:
            raise ContractError(f"level {self.level!r} 는 {LEVELS} 중 하나여야 한다")
        _str("read_hash", self.read_hash, True)
        _dict("meta", self.meta)
        if len(self.layers) < 1:
            raise ContractError("layers 는 1장 이상이어야 한다")
        hw = (self.obs.height, self.obs.width)
        for i, ly in enumerate(self.layers):
            _req(f"layers[{i}]", ly, LDILayer)
            if ly.rgba.shape[:2] != hw:
                raise ContractError(f"layers[{i}] 크기 {ly.rgba.shape[:2]} != obs {hw}")
        for i in range(len(self.layers) - 1):
            a, b = self.layers[i], self.layers[i + 1]
            both = (a.rgba[..., 3] > 1e-3) & (b.rgba[..., 3] > 1e-3)
            if both.any():
                da = a.depth[both].astype(np.float64)
                db = b.depth[both].astype(np.float64)
                if not (da <= db * (1.0 + 1e-6)).all():
                    raise ContractError(f"layers[{i}] 가 layers[{i + 1}] 보다 뒤에 있다 (앞->뒤 정렬 위반)")


# ---------------------------------------------------------------------------------------------------------------
# Shot / Frames / KPI
# ---------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True, eq=False)
class Shot(_Contract):
    """프레임별 카메라 궤적. c2w 는 OpenCV 카메라 -> 세계."""
    c2w: Any
    t: Any
    K: Any
    focus_m: Any
    width: int
    height: int
    fps: float
    theta: Theta
    move: str = "static"
    rig: str = "locked"
    seed: int = 0

    def _coerce(self) -> None:
        self.c2w = _arr("c2w", self.c2w, np.float64)
        self.t = _arr("t", self.t, np.float64)
        self.K = _arr("K", self.K, np.float64)
        self.focus_m = _arr("focus_m", self.focus_m, np.float64)
        self.width = _int("width", self.width, 1)
        self.height = _int("height", self.height, 1)
        self.fps = _num("fps", self.fps)
        self.seed = _int("seed", self.seed)

    def _validate(self) -> None:
        _req("theta", self.theta, Theta)
        _str("move", self.move)
        _str("rig", self.rig)
        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ContractError("fps > 0 이어야 한다")
        n = self.t.shape[0] if self.t.ndim == 1 else 0
        if n < 1:
            raise ContractError("프레임이 1개 이상이어야 한다")
        _shape("c2w", self.c2w, (n, 4, 4))
        _shape("t", self.t, (n,))
        _shape("K", self.K, (n, 3, 3))
        _shape("focus_m", self.focus_m, (n,))
        _finite("c2w", self.c2w)
        _finite("t", self.t)
        _no_nan("focus_m", self.focus_m)
        if not (self.focus_m > 0).all():
            raise ContractError("focus_m > 0 이어야 한다")
        if abs(self.t[0]) > 1e-9:
            raise ContractError("t[0] 은 0 이어야 한다")
        if n > 1 and np.abs(np.diff(self.t) - 1.0 / self.fps).max() > 1e-9:
            raise ContractError("t 간격이 1/fps 가 아니다")
        _check_rot("c2w 회전", self.c2w[:, :3, :3])
        if np.abs(self.c2w[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-9:
            raise ContractError("c2w 마지막 행이 [0,0,0,1] 이 아니다")
        _check_K("K", self.K)

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def first_pose_matches(self, obs: Obs, rot_tol_deg: float = 0.1, trans_tol_m: float = 1e-3) -> bool:
        """첫 프레임 포즈가 관측 카메라와 (회전 각 오차 deg, 위치 오차 m) 허용 안에서 같은가."""
        T0, To = self.c2w[0], obs.c2w()
        if float(np.linalg.norm(T0[:3, 3] - To[:3, 3])) > trans_tol_m:
            return False
        Rr = T0[:3, :3].T @ To[:3, :3]
        vee = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / 2.0
        ang = math.degrees(math.atan2(float(np.linalg.norm(vee)), (float(np.trace(Rr)) - 1.0) / 2.0))
        return ang <= rot_tol_deg


@dataclass(kw_only=True, eq=False)
class Frames(_Contract):
    """렌더 결과: 표시값 [0,1] RGB 와 AOV."""
    rgb: Any
    fps: float
    aov: dict = field(default_factory=dict)
    shot_hash: str = ""
    world_hash: str = ""

    def _coerce(self) -> None:
        self.rgb = _arr("rgb", self.rgb, np.float32)
        self.fps = _num("fps", self.fps)
        _dict("aov", self.aov)
        self.aov = {k: (_arr(f"aov[{k}]", v, v.dtype if isinstance(v, np.ndarray) else np.float32))
                    for k, v in self.aov.items()}

    def _validate(self) -> None:
        if self.rgb.ndim != 4 or self.rgb.shape[3] != 3 or min(self.rgb.shape) < 1:
            raise ContractError(f"rgb shape {self.rgb.shape}: (N,H,W,3) 이어야 한다")
        _unit_range("rgb", self.rgb)
        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ContractError("fps > 0 이어야 한다")
        n = self.rgb.shape[0]
        for k, a in self.aov.items():
            _str("aov 키", k)
            if a.ndim < 1 or a.shape[0] != n:
                raise ContractError(f"aov[{k}] 첫 축이 N={n} 이 아니다 (shape {a.shape})")
            _no_nan(f"aov[{k}]", a)
        _str("shot_hash", self.shot_hash)
        _str("world_hash", self.world_hash)


# 이름: (lo, hi, lo_open, hi_open)
_KPI_RANGES = {
    "ffd": (0.0, _INF, False, False), "ffg": (0.0, _INF, False, False), "de00": (0.0, _INF, False, False),
    "ir_geo": (0.0, 1.0, False, False), "ir_area": (0.0, 1.0, False, False), "ir_info": (0.0, 1.0, False, False),
    "m_eff": (0.0, _INF, True, False), "esc": (0.0, _INF, False, False),
    "cheat_share": (0.0, 1.0, False, False), "tfu": (0.0, _INF, False, False),
    "gpu_h": (0.0, _INF, False, False), "wall_s": (0.0, _INF, False, False),
}


@dataclass(kw_only=True, eq=False)
class KPI(_Contract):
    ffd: Optional[float] = None
    ffg: Optional[float] = None
    de00: Optional[float] = None
    ir_geo: Optional[float] = None
    ir_area: Optional[float] = None
    ir_info: Optional[float] = None
    m_eff: Optional[float] = None
    esc: Optional[float] = None
    cheat_share: Optional[float] = None
    tfu: Optional[float] = None
    gpu_h: Optional[float] = None
    wall_s: Optional[float] = None
    ladder_step: Optional[str] = None
    proof_of_shot: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def _coerce(self) -> None:
        for n in _KPI_RANGES:
            setattr(self, n, _opt_num(n, getattr(self, n)))

    def _validate(self) -> None:
        for n, (lo, hi, lo_o, hi_o) in _KPI_RANGES.items():
            v = getattr(self, n)
            if v is not None:
                _rng(n, v, lo, hi, lo_o, hi_o)
        for n in ("ladder_step", "proof_of_shot"):
            v = getattr(self, n)
            if v is not None:
                _str(n, v)
        _dict("extra", self.extra)


# ---------------------------------------------------------------------------------------------------------------
# StageIO
# ---------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True, eq=False)
class StageIO(_Contract):
    """스테이지 실행 한 번의 입출력 지문. 캐시 키는 outputs·wall_s·fallback 을 제외한다."""
    stage: str
    impl_kind: str
    impl_name: str
    impl_version: str
    inputs: dict
    outputs: dict
    params: dict = field(default_factory=dict)
    wall_s: float = 0.0
    fallback: Optional[str] = None

    def _coerce(self) -> None:
        self.wall_s = _num("wall_s", self.wall_s)
        _dict("inputs", self.inputs)
        _dict("outputs", self.outputs)
        self.inputs = dict(self.inputs)
        self.outputs = dict(self.outputs)

    def _validate(self) -> None:
        if self.stage not in STAGES:
            raise ContractError(f"stage {self.stage!r} 는 {STAGES} 중 하나여야 한다")
        if self.impl_kind not in IMPL_KINDS:
            raise ContractError(f"impl_kind {self.impl_kind!r} 는 {IMPL_KINDS} 중 하나여야 한다")
        _str("impl_name", self.impl_name)
        _str("impl_version", self.impl_version)
        for nm in ("inputs", "outputs"):
            for k, h in _dict(nm, getattr(self, nm)).items():
                _str(f"{nm} 키", k)
                _str(f"{nm}[{k}]", h)
        _dict("params", self.params)
        _rng("wall_s", self.wall_s, 0.0, _INF)
        if self.fallback is not None:
            _str("fallback", self.fallback)

    @property
    def cache_key(self) -> str:
        return content_hash((CONTRACT_VERSION, self.stage, self.impl_kind, self.impl_name, self.impl_version,
                             self.inputs, self.params))


# ---------------------------------------------------------------------------------------------------------------
# 스테이지 이음매
# ---------------------------------------------------------------------------------------------------------------
STAGE_SIGNATURES: Dict[str, Tuple[Dict[str, type], Dict[str, type]]] = {
    "S-1": ({"image": InputImage}, {"read": ReadPacket}),
    "S2": ({"read": ReadPacket}, {"world": World}),
    "S8": ({"world": World, "read": ReadPacket}, {"shot": Shot}),
    "S8.5": ({"world": World, "read": ReadPacket}, {"world": World}),
    "S9-10": ({"world": World, "shot": Shot}, {"frames": Frames}),
    "S11": ({"frames": Frames, "read": ReadPacket, "world": World, "shot": Shot}, {"kpi": KPI}),
}


def _check_side(stage: str, side: str, got: Any, want: Dict[str, type]) -> None:
    if not isinstance(got, dict):
        raise ContractError(f"{stage} {side}: dict 여야 한다")
    missing, extra = set(want) - set(got), set(got) - set(want)
    if missing or extra:
        raise ContractError(f"{stage} {side}: 누락 {sorted(missing)} 초과 {sorted(extra)}")
    for k, typ in want.items():
        if not isinstance(got[k], typ):
            raise ContractError(f"{stage} {side}[{k}]: {typ.__name__} 여야 한다 ({type(got[k]).__name__})")
        got[k].validate()


def _taint_superset(stage: str, out_prov: Prov, in_prov: Prov, what: str) -> None:
    lost = in_prov.taint - out_prov.taint
    if lost:
        raise ContractError(f"{stage}: {what} 의 taint 가 사라졌다 {sorted(lost)}")


def check_stage_io(stage: str, inputs: dict, outputs: dict) -> None:
    """서명(타입·키)과 이음매 교차 불변식을 검사한다. 위반은 ContractError."""
    if stage not in STAGE_SIGNATURES:
        raise ContractError(f"스테이지 {stage!r} 에는 v0 서명이 없다 ({sorted(STAGE_SIGNATURES)})")
    ins, outs = STAGE_SIGNATURES[stage]
    _check_side(stage, "inputs", inputs, ins)
    _check_side(stage, "outputs", outputs, outs)
    if stage == "S-1":
        image, read = inputs["image"], outputs["read"]
        if read.input_hash != image.h_I:
            raise ContractError("S-1: read.input_hash != image.h_I")
        if (read.obs.height, read.obs.width) != tuple(image.pixels.shape[:2]):
            raise ContractError("S-1: obs 크기가 입력 영상과 다르다")
        _taint_superset(stage, read.prov, image.prov, "read")
    elif stage == "S2":
        read, world = inputs["read"], outputs["world"]
        if world.read_hash != read.content_hash():
            raise ContractError("S2: world.read_hash != read.content_hash()")
        if world.obs.content_hash() != read.obs.content_hash():
            raise ContractError("S2: world.obs != read.obs")
        _taint_superset(stage, world.prov, read.prov, "world")
    elif stage == "S8":
        if not outputs["shot"].first_pose_matches(inputs["read"].obs):
            raise ContractError("S8: shot 첫 프레임이 관측 카메라와 다르다")
    elif stage == "S8.5":
        if outputs["world"].obs.content_hash() != inputs["world"].obs.content_hash():
            raise ContractError("S8.5: 출력 world.obs 가 입력과 다르다")
    elif stage == "S9-10":
        shot, frames = inputs["shot"], outputs["frames"]
        if tuple(frames.rgb.shape[:3]) != (shot.n_frames, shot.height, shot.width):
            raise ContractError(
                f"S9-10: frames {tuple(frames.rgb.shape[:3])} != shot {(shot.n_frames, shot.height, shot.width)}")


def make_stage_io(stage: str, impl_kind: str, impl_name: str, impl_version: str, inputs: dict, outputs: dict,
                  params: Optional[dict] = None, wall_s: float = 0.0, fallback: Optional[str] = None) -> StageIO:
    """check_stage_io 를 먼저 돌린 뒤 각 값의 content_hash 로 StageIO 를 채운다."""
    check_stage_io(stage, inputs, outputs)
    return StageIO(stage=stage, impl_kind=impl_kind, impl_name=impl_name, impl_version=impl_version,
                   inputs={k: v.content_hash() for k, v in inputs.items()},
                   outputs={k: v.content_hash() for k, v in outputs.items()},
                   params={} if params is None else params, wall_s=wall_s, fallback=fallback)
