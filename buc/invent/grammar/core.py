"""INV-01 문법 코어: 장면 프로그램(SceneProgram) · 레이캐스트 · G1-G13 에너지 골격.

규약(계약서 ops/harness/contracts/INV-01.md): 세계 오른손·미터, z 위, 바닥 z=0, 천장 z=shell.height.
슬롯 로컬 +x = 정면, yaw = z축 CCW. 셸 corners 는 CCW, 변 i = corner i -> (i+1)%M, 안쪽 = 진행 방향 왼쪽.
numpy(+scipy 허용)만 쓴다. torch 금지, import 부작용 없음, 전역 RNG 불변. numba 는 선택(첫 호출 때 JIT).
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from buc.core import geom
from buc.core.contracts import content_hash as _content_hash

# --------------------------------------------------------------------------------------------------- 상수
SLOT_CLASSES = ("bed", "sofa", "table", "chair", "desk", "shelf", "cabinet", "wardrobe", "lamp", "plant", "tv",
                "person", "rug", "picture", "stool", "counter")
FURNITURE_CLASSES = ("bed", "sofa", "table", "desk", "shelf", "cabinet", "wardrobe", "counter")
SUPPORTS = ("floor", "wall", "slot", "ceiling")
OPENING_KINDS = ("door", "window")
LIGHT_KINDS = ("window", "lamp", "sun", "other")
MATERIALS = ("wood", "tile", "carpet", "concrete", "plaster", "brick", "paint_white", "wood_panel", "marble",
             "fabric", "leather", "metal", "glass", "stone", "plastic", "ceramic")
SLOT_MATERIALS = MATERIALS + ("default",)
SEM_VOID, SEM_FLOOR, SEM_WALL, SEM_CEILING, SEM_DOOR, SEM_WINDOW = 0, 1, 2, 3, 4, 5
SEM_SLOT_BASE = 8
ENERGY_NAMES = tuple(f"G{i}" for i in range(1, 14))
HAS_NUMBA: bool = importlib.util.find_spec("numba") is not None

_TOL_SLOT = 0.02
_TOL_WALL = 0.05
_EPS_T = 1e-9
_EPS_E = 1e-9


class GrammarError(ValueError):
    """문법 코어 데이터 위반."""


def sem_id(cls: str) -> int:
    """슬롯 클래스 -> 의미 id (8 + 인덱스)."""
    try:
        return SEM_SLOT_BASE + SLOT_CLASSES.index(cls)
    except ValueError:
        raise ValueError(f"모르는 슬롯 클래스 {cls!r}") from None


# --------------------------------------------------------------------------------------------------- 검증 도우미
def _set(obj, name, value):
    object.__setattr__(obj, name, value)


def _farr(name, v, shape):
    try:
        a = np.array(v, dtype=np.float64, order="C")
    except (TypeError, ValueError) as e:
        raise GrammarError(f"{name}: 숫자 배열이 아님 ({e})") from e
    if a.shape != shape:
        raise GrammarError(f"{name}: shape {a.shape}, 기대 {shape}")
    if not np.isfinite(a).all():
        raise GrammarError(f"{name}: 유한하지 않은 값")
    a.setflags(write=False)
    return a


def _fnum(name, v):
    if isinstance(v, (bool, np.bool_)):
        raise GrammarError(f"{name}: bool 은 숫자가 아니다")
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise GrammarError(f"{name}: 숫자가 아님 ({e})") from e
    if not math.isfinite(f):
        raise GrammarError(f"{name}: 유한하지 않은 값")
    return f


def _inum(name, v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise GrammarError(f"{name}: 정수여야 한다 ({v!r})")
    return int(v)


def _choice(name, v, options):
    if not isinstance(v, str) or v not in options:
        raise GrammarError(f"{name}={v!r}: {options} 중 하나여야 한다")
    return v


def _jsonable(x):
    """meta 정규화: 튜플 -> 리스트, numpy -> 파이썬. 비문자열 키 거부."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_jsonable(e) for e in x]
    if isinstance(x, dict):
        out = {}
        for k, e in x.items():
            if not isinstance(k, str):
                raise GrammarError(f"meta 키는 문자열이어야 한다: {k!r}")
            out[k] = _jsonable(e)
        return out
    raise GrammarError(f"meta 에 넣을 수 없는 타입 {type(x).__name__}")


# --------------------------------------------------------------------------------------------------- 데이터
@dataclass(frozen=True, kw_only=True, eq=False)
class Shell:
    corners: Any
    height: Any
    walls_present: Any = None
    floor: bool = True
    ceiling: bool = True

    def __post_init__(self):
        try:
            c = np.array(self.corners, dtype=np.float64, order="C")
        except (TypeError, ValueError) as e:
            raise GrammarError(f"corners: 숫자 배열이 아님 ({e})") from e
        if c.ndim != 2 or c.shape[1] != 2 or not (3 <= c.shape[0] <= 16):
            raise GrammarError(f"corners: (M,2) 3<=M<=16 이어야 한다 (shape {c.shape})")
        if not np.isfinite(c).all():
            raise GrammarError("corners: 유한하지 않은 값")
        x, y = c[:, 0], c[:, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if not area > 0.0:
            raise GrammarError(f"corners 는 CCW 이고 면적>0 이어야 한다 (면적 {area})")
        if (np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1) <= 0.0).any():
            raise GrammarError("길이 0 인 변이 있다")
        c.setflags(write=False)
        _set(self, "corners", c)
        h = _fnum("height", self.height)
        if not h > 0.0:
            raise GrammarError(f"height={h}: 양수여야 한다")
        _set(self, "height", h)
        m = c.shape[0]
        wp = self.walls_present
        if wp is None:
            wp = (True,) * m
        else:
            try:
                wp = tuple(bool(w) for w in wp)
            except TypeError as e:
                raise GrammarError("walls_present: 시퀀스여야 한다") from e
            if len(wp) != m:
                raise GrammarError(f"walls_present 길이 {len(wp)} != 변 수 {m}")
        _set(self, "walls_present", wp)
        _set(self, "floor", bool(self.floor))
        _set(self, "ceiling", bool(self.ceiling))

    @property
    def n_edges(self) -> int:
        return int(self.corners.shape[0])

    def edge_vectors(self) -> np.ndarray:
        return np.roll(self.corners, -1, axis=0) - self.corners

    def edge_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.edge_vectors(), axis=1)

    def perimeter(self) -> float:
        return float(self.edge_lengths().sum())

    def area(self) -> float:
        x, y = self.corners[:, 0], self.corners[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass(frozen=True, kw_only=True, eq=False)
class Opening:
    kind: str
    edge: int
    s0: float
    s1: float
    z0: float
    z1: float

    def __post_init__(self):
        _choice("kind", self.kind, OPENING_KINDS)
        _set(self, "edge", _inum("edge", self.edge))
        for n in ("s0", "s1", "z0", "z1"):
            _set(self, n, _fnum(n, getattr(self, n)))
        if not self.s0 < self.s1:
            raise GrammarError(f"s0={self.s0} < s1={self.s1} 이어야 한다")
        if not self.z0 < self.z1:
            raise GrammarError(f"z0={self.z0} < z1={self.z1} 이어야 한다")


@dataclass(frozen=True, kw_only=True, eq=False)
class Slot:
    id: int
    cls: str
    center: Any
    size: Any
    yaw: float = 0.0
    support: str = "floor"
    support_id: int = -1
    material: str = "default"

    def __post_init__(self):
        _set(self, "id", _inum("id", self.id))
        _choice("cls", self.cls, SLOT_CLASSES)
        _set(self, "center", _farr("center", self.center, (3,)))
        sz = _farr("size", self.size, (3,))
        if not (sz > 0.0).all():
            raise GrammarError(f"size 는 모두 양수여야 한다 ({sz})")
        _set(self, "size", sz)
        _set(self, "yaw", _fnum("yaw", self.yaw))
        _choice("support", self.support, SUPPORTS)
        _set(self, "support_id", _inum("support_id", self.support_id))
        _choice("material", self.material, SLOT_MATERIALS)

    @property
    def z0(self) -> float:
        return float(self.center[2] - self.size[2] / 2.0)

    @property
    def z1(self) -> float:
        return float(self.center[2] + self.size[2] / 2.0)

    def footprint(self) -> np.ndarray:
        """바닥 사각형 4코너 (4,2). 로컬 x=정면, yaw CCW."""
        return _footprint(self.center, self.size, self.yaw)


def _footprint(center, size, yaw) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2.0, size[1] / 2.0
    out = np.empty((4, 2), dtype=np.float64)
    for k, (lx, ly) in enumerate(((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))):
        out[k, 0] = center[0] + c * lx - s * ly
        out[k, 1] = center[1] + s * lx + c * ly
    return out


@dataclass(frozen=True, kw_only=True, eq=False)
class Light:
    id: int
    kind: str
    pos: Any
    rgb: Any
    power: float
    source: Any = None

    def __post_init__(self):
        _set(self, "id", _inum("id", self.id))
        _choice("kind", self.kind, LIGHT_KINDS)
        _set(self, "pos", _farr("pos", self.pos, (3,)))
        rgb = _farr("rgb", self.rgb, (3,))
        if (rgb < 0).any():
            raise GrammarError("rgb 는 0 이상이어야 한다")
        _set(self, "rgb", rgb)
        p = _fnum("power", self.power)
        if p < 0:
            raise GrammarError("power 는 0 이상이어야 한다")
        _set(self, "power", p)
        src = self.source
        if src is not None:
            try:
                k, i = src
            except (TypeError, ValueError) as e:
                raise GrammarError("source 는 None 또는 (종류, 정수) 이어야 한다") from e
            _choice("source[0]", k, ("opening", "slot"))
            src = (k, _inum("source[1]", i))
        _set(self, "source", src)


@dataclass(frozen=True, kw_only=True, eq=False)
class SceneProgram:
    grammar: str
    seed: int
    shell: Shell
    openings: Any = ()
    slots: Any = ()
    lights: Any = ()
    matter: Any = None
    meta: Any = None

    def __post_init__(self):
        if not isinstance(self.grammar, str):
            raise GrammarError("grammar 는 문자열이어야 한다")
        _set(self, "seed", _inum("seed", self.seed))
        if not isinstance(self.shell, Shell):
            raise GrammarError("shell 은 Shell 이어야 한다")
        for name, typ in (("openings", Opening), ("slots", Slot), ("lights", Light)):
            t = tuple(getattr(self, name))
            if not all(isinstance(e, typ) for e in t):
                raise GrammarError(f"{name}: {typ.__name__} 만 들어갈 수 있다")
            _set(self, name, t)
        matter = {} if self.matter is None else self.matter
        if not isinstance(matter, dict):
            raise GrammarError("matter 는 dict 여야 한다")
        _set(self, "matter", dict(matter))
        meta = {} if self.meta is None else self.meta
        if not isinstance(meta, dict):
            raise GrammarError("meta 는 dict 여야 한다")
        _set(self, "meta", _jsonable(meta))
        self.validate()

    def validate(self) -> None:
        m = self.shell.n_edges
        for o in self.openings:
            if not (0 <= o.edge < m):
                raise GrammarError(f"개구부 edge={o.edge} 가 범위 [0,{m}) 밖")
        ids = [s.id for s in self.slots]
        if len(set(ids)) != len(ids):
            raise GrammarError("슬롯 id 중복")
        lids = [lt.id for lt in self.lights]
        if len(set(lids)) != len(lids):
            raise GrammarError("광원 id 중복")
        for k, v in self.matter.items():
            if k not in ("floor", "wall", "ceiling"):
                raise GrammarError(f"matter 키 {k!r}: floor/wall/ceiling 만 허용")
            if v not in MATERIALS:
                raise GrammarError(f"matter[{k!r}]={v!r}: MATERIALS 에 없음")

    def slot_by_id(self, i: int) -> Slot:
        for s in self.slots:
            if s.id == i:
                return s
        raise KeyError(i)

    def to_dict(self) -> dict:
        sh = self.shell
        return {
            "__contract__": "SceneProgram",
            "grammar": self.grammar,
            "seed": int(self.seed),
            "shell": {"corners": sh.corners.tolist(), "height": float(sh.height),
                      "walls_present": [bool(w) for w in sh.walls_present], "floor": bool(sh.floor),
                      "ceiling": bool(sh.ceiling)},
            "openings": [{"kind": o.kind, "edge": int(o.edge), "s0": float(o.s0), "s1": float(o.s1),
                          "z0": float(o.z0), "z1": float(o.z1)} for o in self.openings],
            "slots": [{"id": int(s.id), "cls": s.cls, "center": s.center.tolist(), "size": s.size.tolist(),
                       "yaw": float(s.yaw), "support": s.support, "support_id": int(s.support_id),
                       "material": s.material} for s in self.slots],
            "lights": [{"id": int(lt.id), "kind": lt.kind, "pos": lt.pos.tolist(), "rgb": lt.rgb.tolist(),
                        "power": float(lt.power),
                        "source": None if lt.source is None else [lt.source[0], int(lt.source[1])]}
                       for lt in self.lights],
            "matter": {k: v for k, v in self.matter.items()},
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, d) -> "SceneProgram":
        if not isinstance(d, dict) or d.get("__contract__") != "SceneProgram":
            raise GrammarError("__contract__ 가 SceneProgram 이 아니다")
        try:
            sh = d["shell"]
            shell = Shell(corners=sh["corners"], height=sh["height"], walls_present=sh["walls_present"],
                          floor=sh["floor"], ceiling=sh["ceiling"])
            openings = tuple(Opening(**o) for o in d["openings"])
            slots = tuple(Slot(**s) for s in d["slots"])
            lights = tuple(Light(**lt) for lt in d["lights"])
            return cls(grammar=d["grammar"], seed=d["seed"], shell=shell, openings=openings, slots=slots,
                       lights=lights, matter=d["matter"], meta=d["meta"])
        except GrammarError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise GrammarError(f"from_dict 실패: {e!r}") from e

    def content_hash(self) -> str:
        return _content_hash(self.to_dict())


# --------------------------------------------------------------------------------------------------- 기하 도우미
def _pip_vec(px, py, poly) -> np.ndarray:
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if y0 == y1:
            continue
        cond = (y0 > py) != (y1 > py)
        xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (px < xint)
    return inside


def _pip(x, y, poly) -> bool:
    return bool(_pip_vec(np.array([x], dtype=np.float64), np.array([y], dtype=np.float64), poly)[0])


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _clip_halfplane(poly, d_idx, bound, keep_ge):
    """Sutherland-Hodgman: poly (list of (s,d)); d 좌표 d_idx=1 을 bound 로 자른다."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da, db = a[d_idx] - bound, b[d_idx] - bound
        ina = da >= 0 if keep_ge else da <= 0
        inb = db >= 0 if keep_ge else db <= 0
        if ina:
            out.append(a)
        if ina != inb:
            t = da / (da - db)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _poly_area(pts) -> float:
    if len(pts) < 3:
        return 0.0
    a = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


# --------------------------------------------------------------------------------------------------- 레이캐스트
@dataclass(frozen=True, kw_only=True, eq=False)
class RayCastResult:
    depth: np.ndarray
    sem: np.ndarray
    inst: np.ndarray
    normal: np.ndarray

    def visible_slots(self) -> Dict[int, int]:
        m = self.inst >= 0
        if not m.any():
            return {}
        ids, cnt = np.unique(self.inst[m], return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, cnt)}


def _check_inputs(program, c2w, K, width, height, k1, k2, backend):
    if not isinstance(program, SceneProgram):
        raise TypeError("program 은 SceneProgram 이어야 한다")
    try:
        c = np.asarray(c2w, dtype=np.float64)
        Km = np.asarray(K, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"c2w/K 가 숫자 배열이 아님: {e}") from e
    if c.shape != (4, 4) or not np.isfinite(c).all():
        raise ValueError("c2w: 유한한 (4,4) 여야 한다")
    if Km.shape != (3, 3) or not np.isfinite(Km).all():
        raise ValueError("K: 유한한 (3,3) 여야 한다")
    R = c[:3, :3]
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-5 or abs(np.linalg.det(R) - 1.0) > 1e-5:
        raise ValueError("c2w 의 회전부가 직교(det=+1)가 아니다")
    for n, v in (("width", width), ("height", height)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{n}={v!r}: 1 이상의 정수여야 한다")
    for n, v in (("k1", k1), ("k2", k2)):
        if isinstance(v, (bool, np.bool_)) or not np.isfinite(float(v)):
            raise ValueError(f"{n}: 유한한 숫자여야 한다")
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"backend={backend!r}: auto|numpy|numba")
    if backend == "numba" and not HAS_NUMBA:
        raise RuntimeError("numba 가 설치되어 있지 않다")
    return c, Km, int(width), int(height), float(k1), float(k2)


def raycast(program, *, c2w, K, width, height, k1=0.0, k2=0.0, backend="auto") -> RayCastResult:
    """픽셀 중심 광선(FND-06 unproject, 왜곡 포함)으로 첫 교차의 깊이(광축 z)·의미·슬롯 id·법선을 쏜다."""
    c, Km, W, H, k1, k2 = _check_inputs(program, c2w, K, width, height, k1, k2, backend)
    Xc = geom.unproject(Km, geom.pixel_grid(H, W), 1.0, k1, k2)
    R = c[:3, :3]
    o = c[:3, 3].copy()
    D = np.ascontiguousarray(Xc.reshape(-1, 3) @ R.T)
    if backend == "numba":
        t, sem, inst, nrm = _trace_numba(program, o, D)
    else:
        t, sem, inst, nrm = _trace_numpy(program, o, D)
    return RayCastResult(depth=t.astype(np.float32).reshape(H, W), sem=sem.reshape(H, W),
                         inst=inst.reshape(H, W), normal=nrm.astype(np.float32).reshape(H, W, 3))


def _sorted_slots(program):
    return sorted(program.slots, key=lambda s: s.id)


def _trace_numpy(program, o, D):
    N = D.shape[0]
    sh = program.shell
    poly = sh.corners
    Hh = sh.height
    best_t = np.full(N, np.inf)
    sem = np.zeros(N, np.int16)
    inst = np.full(N, -1, np.int32)
    nrm = np.zeros((N, 3))

    def update(valid, t, s, i, n):
        better = valid & (t < best_t)
        if not better.any():
            return
        best_t[better] = t[better]
        sem[better] = s[better] if isinstance(s, np.ndarray) else s
        inst[better] = i
        nrm[better] = n[better]

    with np.errstate(all="ignore"):
        # 바닥 / 천장
        dz = D[:, 2]
        nz = np.abs(dz) > 1e-12
        sign = np.where(dz < 0, 1.0, -1.0)
        nvec = np.zeros((N, 3))
        nvec[:, 2] = sign
        for z, present, sid in ((0.0, sh.floor, SEM_FLOOR), (Hh, sh.ceiling, SEM_CEILING)):
            if not present:
                continue
            t = (z - o[2]) / np.where(nz, dz, 1.0)
            valid = nz & (t > _EPS_T)
            tt = np.where(valid, t, 0.0)
            px, py = o[0] + tt * D[:, 0], o[1] + tt * D[:, 1]
            valid &= _pip_vec(px, py, poly)
            update(valid, np.where(valid, t, np.inf), sid, -1, nvec)
        # 벽 + 개구부
        M = sh.n_edges
        for i in range(M):
            if not sh.walls_present[i]:
                continue
            A, B = poly[i], poly[(i + 1) % M]
            Lw = float(np.linalg.norm(B - A))
            u = (B - A) / Lw
            n2 = np.array([-u[1], u[0], 0.0])
            den = D @ n2
            num = float((A[0] - o[0]) * n2[0] + (A[1] - o[1]) * n2[1])
            ok = np.abs(den) > 1e-12
            t = num / np.where(ok, den, 1.0)
            valid = ok & (t > _EPS_T)
            tt = np.where(valid, t, 0.0)
            px, py, pz = o[0] + tt * D[:, 0], o[1] + tt * D[:, 1], o[2] + tt * D[:, 2]
            s = (px - A[0]) * u[0] + (py - A[1]) * u[1]
            valid &= (s >= -_EPS_E) & (s <= Lw + _EPS_E) & (pz >= -_EPS_E) & (pz <= Hh + _EPS_E)
            ssem = np.full(N, SEM_WALL, dtype=np.int16)
            for op in reversed([x for x in program.openings if x.edge == i]):
                m = (s >= op.s0) & (s <= op.s1) & (pz >= op.z0) & (pz <= op.z1)
                ssem[m] = SEM_DOOR if op.kind == "door" else SEM_WINDOW
            nsign = np.where(den < 0, 1.0, -1.0)
            update(valid, np.where(valid, t, np.inf), ssem, -1, nsign[:, None] * n2[None, :])
        # 슬롯 OBB (id 순으로 처리 -> 입력 순서 무관)
        for sl in _sorted_slots(program):
            cy, sy = math.cos(sl.yaw), math.sin(sl.yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            ol = Rz.T @ (o - sl.center)
            dl = D @ Rz
            h = sl.size / 2.0
            te = np.full(N, -np.inf)
            tx = np.full(N, np.inf)
            axis = np.full(N, -1, np.int64)
            miss = np.zeros(N, bool)
            for k in range(3):
                par = np.abs(dl[:, k]) < 1e-14
                inv = 1.0 / np.where(par, 1.0, dl[:, k])
                t0, t1 = (-h[k] - ol[k]) * inv, (h[k] - ol[k]) * inv
                lo = np.where(par, -np.inf, np.minimum(t0, t1))
                hi = np.where(par, np.inf, np.maximum(t0, t1))
                miss |= par & (abs(ol[k]) > h[k])
                upd = lo > te
                axis = np.where(upd, k, axis)
                te = np.maximum(te, lo)
                tx = np.minimum(tx, hi)
            valid = ~miss & (axis >= 0) & (te <= tx) & (te > _EPS_T)
            nl = np.zeros((N, 3))
            for k in range(3):
                m = axis == k
                nl[m, k] = -np.sign(dl[m, k])
            wn = nl @ Rz.T
            update(valid, np.where(valid, te, np.inf), sem_id(sl.cls), sl.id, wn)
    return best_t, sem, inst, nrm


# ----- numba 경로: 같은 스칼라 커널. 순수 파이썬으로도 실행 가능(테스트용), numba 가 있으면 첫 호출 때 JIT.
def _build_kernel(jit):
    @jit
    def pip(px, py, poly):
        inside = False
        n = poly.shape[0]
        for i in range(n):
            x0 = poly[i, 0]
            y0 = poly[i, 1]
            x1 = poly[(i + 1) % n, 0]
            y1 = poly[(i + 1) % n, 1]
            if y0 == y1:
                continue
            if (y0 > py) != (y1 > py):
                if px < x0 + (py - y0) * (x1 - x0) / (y1 - y0):
                    inside = not inside
        return inside

    @jit
    def trace(o, D, poly, Hh, has_floor, has_ceil, wp, op_edge, op_s0, op_s1, op_z0, op_z1, op_sem, sl_c, sl_h,
              sl_cs, sl_sn, sl_sem, sl_id, out_t, out_sem, out_inst, out_n):
        N = D.shape[0]
        M = poly.shape[0]
        for r in range(N):
            dx = D[r, 0]
            dy = D[r, 1]
            dz = D[r, 2]
            bt = np.inf
            bs = 0
            bi = -1
            bnx = 0.0
            bny = 0.0
            bnz = 0.0
            for pl in range(2):
                if pl == 0:
                    present = has_floor
                    z = 0.0
                    sid = 1
                else:
                    present = has_ceil
                    z = Hh
                    sid = 3
                if present and abs(dz) > 1e-12:
                    t = (z - o[2]) / dz
                    if t > 1e-9 and t < bt:
                        if pip(o[0] + t * dx, o[1] + t * dy, poly):
                            bt = t
                            bs = sid
                            bi = -1
                            bnx = 0.0
                            bny = 0.0
                            bnz = 1.0 if dz < 0 else -1.0
            for i in range(M):
                if wp[i] == 0:
                    continue
                ax = poly[i, 0]
                ay = poly[i, 1]
                bx = poly[(i + 1) % M, 0]
                by = poly[(i + 1) % M, 1]
                Lw = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
                ux = (bx - ax) / Lw
                uy = (by - ay) / Lw
                nx = -uy
                ny = ux
                den = dx * nx + dy * ny
                if abs(den) > 1e-12:
                    t = ((ax - o[0]) * nx + (ay - o[1]) * ny) / den
                    if t > 1e-9 and t < bt:
                        px = o[0] + t * dx
                        py = o[1] + t * dy
                        pz = o[2] + t * dz
                        s = (px - ax) * ux + (py - ay) * uy
                        if s >= -1e-9 and s <= Lw + 1e-9 and pz >= -1e-9 and pz <= Hh + 1e-9:
                            ssem = 2
                            for k in range(op_edge.shape[0]):
                                if op_edge[k] == i and s >= op_s0[k] and s <= op_s1[k] and pz >= op_z0[k] \
                                        and pz <= op_z1[k]:
                                    ssem = op_sem[k]
                                    break
                            bt = t
                            bs = ssem
                            bi = -1
                            sg = 1.0 if den < 0 else -1.0
                            bnx = sg * nx
                            bny = sg * ny
                            bnz = 0.0
            for j in range(sl_c.shape[0]):
                cs = sl_cs[j]
                sn = sl_sn[j]
                ox = o[0] - sl_c[j, 0]
                oy = o[1] - sl_c[j, 1]
                oz = o[2] - sl_c[j, 2]
                # Rz^T v = (cs*vx + sn*vy, -sn*vx + cs*vy, vz)
                olx = cs * ox + sn * oy
                oly = -sn * ox + cs * oy
                olz = oz
                dlx = cs * dx + sn * dy
                dly = -sn * dx + cs * dy
                dlz = dz
                te = -np.inf
                tx = np.inf
                axis = -1
                miss = False
                for k in range(3):
                    if k == 0:
                        dk = dlx
                        ok_ = olx
                    elif k == 1:
                        dk = dly
                        ok_ = oly
                    else:
                        dk = dlz
                        ok_ = olz
                    hk = sl_h[j, k]
                    if abs(dk) < 1e-14:
                        if abs(ok_) > hk:
                            miss = True
                        continue
                    t0 = (-hk - ok_) / dk
                    t1 = (hk - ok_) / dk
                    lo = min(t0, t1)
                    hi = max(t0, t1)
                    if lo > te:
                        te = lo
                        axis = k
                    if hi < tx:
                        tx = hi
                if (not miss) and axis >= 0 and te <= tx and te > 1e-9 and te < bt:
                    if axis == 0:
                        dk = dlx
                    elif axis == 1:
                        dk = dly
                    else:
                        dk = dlz
                    sgn = 0.0
                    if dk > 0:
                        sgn = -1.0
                    elif dk < 0:
                        sgn = 1.0
                    nlx = sgn if axis == 0 else 0.0
                    nly = sgn if axis == 1 else 0.0
                    nlz = sgn if axis == 2 else 0.0
                    bt = te
                    bs = sl_sem[j]
                    bi = sl_id[j]
                    bnx = cs * nlx - sn * nly
                    bny = sn * nlx + cs * nly
                    bnz = nlz
            out_t[r] = bt
            out_sem[r] = bs
            out_inst[r] = bi
            out_n[r, 0] = bnx
            out_n[r, 1] = bny
            out_n[r, 2] = bnz

    return trace


_KERNEL_CACHE: Dict[str, Callable] = {}


def _get_kernel(use_numba: bool):
    key = "numba" if use_numba else "py"
    if key not in _KERNEL_CACHE:
        if use_numba:
            nb = importlib.import_module("numba")
            _KERNEL_CACHE[key] = _build_kernel(nb.njit(cache=False))
        else:
            _KERNEL_CACHE[key] = _build_kernel(lambda f: f)
    return _KERNEL_CACHE[key]


def _kernel_args(program):
    sh = program.shell
    ops = list(program.openings)
    sls = _sorted_slots(program)
    S = len(sls)
    return dict(
        poly=np.ascontiguousarray(sh.corners), Hh=float(sh.height), has_floor=bool(sh.floor),
        has_ceil=bool(sh.ceiling), wp=np.array(sh.walls_present, dtype=np.uint8),
        op_edge=np.array([x.edge for x in ops], dtype=np.int64),
        op_s0=np.array([x.s0 for x in ops], dtype=np.float64), op_s1=np.array([x.s1 for x in ops], dtype=np.float64),
        op_z0=np.array([x.z0 for x in ops], dtype=np.float64), op_z1=np.array([x.z1 for x in ops], dtype=np.float64),
        op_sem=np.array([SEM_DOOR if x.kind == "door" else SEM_WINDOW for x in ops], dtype=np.int64),
        sl_c=np.array([s.center for s in sls], dtype=np.float64).reshape(S, 3),
        sl_h=np.array([s.size / 2.0 for s in sls], dtype=np.float64).reshape(S, 3),
        sl_cs=np.array([math.cos(s.yaw) for s in sls], dtype=np.float64),
        sl_sn=np.array([math.sin(s.yaw) for s in sls], dtype=np.float64),
        sl_sem=np.array([sem_id(s.cls) for s in sls], dtype=np.int64),
        sl_id=np.array([s.id for s in sls], dtype=np.int64))


def _run_kernel(program, o, D, use_numba):
    N = D.shape[0]
    a = _kernel_args(program)
    out_t = np.empty(N)
    out_sem = np.zeros(N, np.int16)
    out_inst = np.full(N, -1, np.int32)
    out_n = np.zeros((N, 3))
    kern = _get_kernel(use_numba)
    kern(o, D, a["poly"], a["Hh"], a["has_floor"], a["has_ceil"], a["wp"], a["op_edge"], a["op_s0"], a["op_s1"],
         a["op_z0"], a["op_z1"], a["op_sem"], a["sl_c"], a["sl_h"], a["sl_cs"], a["sl_sn"], a["sl_sem"], a["sl_id"],
         out_t, out_sem, out_inst, out_n)
    return out_t, out_sem, out_inst, out_n


def _trace_numba(program, o, D):
    return _run_kernel(program, o, D, True)


# --------------------------------------------------------------------------------------------------- 에너지
@dataclass(frozen=True, kw_only=True, eq=False)
class CameraSpec:
    c2w: Any
    K: Any
    width: int
    height: int

    def __post_init__(self):
        _set(self, "c2w", _farr("c2w", self.c2w, (4, 4)))
        _set(self, "K", _farr("K", self.K, (3, 3)))
        _set(self, "width", _inum("width", self.width))
        _set(self, "height", _inum("height", self.height))
        if self.width < 1 or self.height < 1:
            raise GrammarError("width/height 는 1 이상이어야 한다")


@dataclass(frozen=True, kw_only=True, eq=False)
class EnergyContext:
    cameras: Tuple[CameraSpec, ...] = ()
    allowed_classes: Optional[frozenset] = None
    symmetry: Optional[Tuple[str, float]] = None

    def __post_init__(self):
        cams = tuple(self.cameras)
        if not all(isinstance(c, CameraSpec) for c in cams):
            raise GrammarError("cameras 는 CameraSpec 의 시퀀스여야 한다")
        _set(self, "cameras", cams)
        if self.allowed_classes is not None:
            _set(self, "allowed_classes", frozenset(self.allowed_classes))
        if self.symmetry is not None:
            try:
                ax, a = self.symmetry
            except (TypeError, ValueError) as e:
                raise GrammarError("symmetry 는 (축, 값) 이어야 한다") from e
            _choice("symmetry[0]", ax, ("x", "y"))
            _set(self, "symmetry", (ax, _fnum("symmetry[1]", a)))


@dataclass(frozen=True, kw_only=True, eq=False)
class GrammarEnergy:
    terms: Dict[str, float]
    weights: Dict[str, float]
    total: float

    def vector(self) -> np.ndarray:
        return np.array([self.terms[k] for k in ENERGY_NAMES], dtype=np.float64)


def _g1(p, ctx):
    sh = p.shell
    L = sh.edge_lengths()
    e = sum(float(L[i]) for i in range(sh.n_edges) if not sh.walls_present[i])
    if not sh.floor:
        e += sh.perimeter()
    if not sh.ceiling:
        e += sh.perimeter()
    return float(e)


def _g2(p, ctx):
    L = p.shell.edge_lengths()
    H = p.shell.height
    ndoors = sum(1 for o in p.openings if o.kind == "door")
    e = float(max(0, 1 - ndoors))
    for o in p.openings:
        if o.s0 < 0 or o.s1 > L[o.edge] or o.z0 < 0 or o.z1 > H:
            e += 1.0
    ops = p.openings
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            a, b = ops[i], ops[j]
            if a.edge != b.edge:
                continue
            if min(a.s1, b.s1) - max(a.s0, b.s0) > 1e-9 and min(a.z1, b.z1) - max(a.z0, b.z0) > 1e-9:
                e += 1.0
    return float(e)


def _g3(p, ctx):
    sh = p.shell
    corners = sh.corners
    L = sh.edge_lengths()
    obstacles = [s.footprint() for s in p.slots if s.support == "floor" and s.cls != "rug"]
    total = 0.0
    for o in p.openings:
        if o.kind != "door":
            continue
        A = corners[o.edge]
        B = corners[(o.edge + 1) % sh.n_edges]
        u = (B - A) / L[o.edge]
        n = np.array([-u[1], u[0]])
        slo, shi = max(o.s0, 0.0), min(o.s1, float(L[o.edge]))
        if shi <= slo:
            continue
        ivs = []
        for fp in obstacles:
            q = fp - A
            pts = [(float(r @ u), float(r @ n)) for r in q]
            pts = _clip_halfplane(pts, 1, 0.0, True)
            pts = _clip_halfplane(pts, 1, 1.0, False)
            if _poly_area(pts) <= 1e-12:
                continue
            lo = max(min(x for x, _ in pts), slo)
            hi = min(max(x for x, _ in pts), shi)
            if hi > lo:
                ivs.append((lo, hi))
        ivs.sort()
        cur_lo = cur_hi = None
        for lo, hi in ivs:
            if cur_hi is None or lo > cur_hi:
                if cur_hi is not None:
                    total += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        if cur_hi is not None:
            total += cur_hi - cur_lo
    return float(total)


def _g4(p, ctx):
    sh = p.shell
    H = sh.height
    by_id = {s.id: s for s in p.slots}
    corners = sh.corners
    M = sh.n_edges
    e = 0.0
    for s in p.slots:
        ok = True
        if s.support == "floor":
            ok = abs(s.z0) <= _TOL_SLOT
        elif s.support == "ceiling":
            ok = abs(s.z1 - H) <= _TOL_SLOT
        elif s.support == "slot":
            sup = by_id.get(s.support_id)
            if sup is None or sup.id == s.id:
                ok = False
            else:
                ok = abs(s.z0 - sup.z1) <= _TOL_SLOT and _pip(float(s.center[0]), float(s.center[1]),
                                                              sup.footprint())
        else:  # wall
            fp = s.footprint()
            d = min(_seg_dist(fp[k], corners[i], corners[(i + 1) % M]) for k in range(4) for i in range(M))
            ok = d <= _TOL_WALL
        if not ok:
            e += float(s.size[0] * s.size[1] * s.size[2])
    return float(e)


def _g5(p, ctx):
    by_id = {s.id: s for s in p.slots}
    n = 0
    for lt in p.lights:
        src = lt.source
        ok = False
        if src is not None:
            k, i = src
            if k == "opening":
                ok = 0 <= i < len(p.openings) and p.openings[i].kind == "window"
            else:
                ok = i in by_id and by_id[i].cls == "lamp"
        if not ok:
            n += 1
    return float(n)


def _g6(p, ctx):
    e = 0.0
    for edge in range(p.shell.n_edges):
        cs = sorted((o.s0 + o.s1) / 2.0 for o in p.openings if o.kind == "window" and o.edge == edge)
        if len(cs) < 3:
            continue
        g = np.diff(np.array(cs))
        e += float(np.abs(g - g.mean()).sum())
    return e


def _g7(p, ctx):
    if ctx.symmetry is None or not p.slots:
        return 0.0
    axis, a = ctx.symmetry
    k = 0 if axis == "x" else 1
    e = 0.0
    for s in p.slots:
        m = s.center.copy()
        m[k] = 2.0 * a - m[k]
        best = min(float(np.linalg.norm(t.center - m)) for t in p.slots if t.cls == s.cls)
        e += best
    return float(e)


def _g8(p, ctx):
    if ctx.allowed_classes is None:
        return 0.0
    return float(sum(1 for s in p.slots if s.cls not in ctx.allowed_classes))


def _g_zero(p, ctx):
    return 0.0


def _g13(p, ctx):
    if not ctx.cameras:
        return 0.0
    n = 0
    for s in p.slots:
        if s.cls != "person":
            continue
        head = np.array([s.center[0], s.center[1], s.z1 - 0.12])
        fx_, fy_ = math.cos(s.yaw), math.sin(s.yaw)
        hit = False
        for cam in ctx.cameras:
            Rm = cam.c2w[:3, :3].T
            t = -Rm @ cam.c2w[:3, 3]
            xc = Rm @ head + t
            zc = float(xc[2])
            if not zc > 1e-9:
                continue
            u = cam.K[0, 0] * xc[0] / zc + cam.K[0, 1] * xc[1] / zc + cam.K[0, 2]
            v = cam.K[1, 1] * xc[1] / zc + cam.K[1, 2]
            if not (-0.5 <= u < cam.width - 0.5 and -0.5 <= v < cam.height - 0.5):
                continue
            to_cam = cam.c2w[:2, 3] - s.center[:2]
            if not (fx_ * to_cam[0] + fy_ * to_cam[1] > 0):
                continue
            if 0.22 * cam.K[1, 1] / zc / cam.height > 0.01:
                hit = True
                break
        n += int(hit)
    return float(n)


RULES: Dict[str, Callable[[SceneProgram, EnergyContext], float]] = {
    "G1": _g1, "G2": _g2, "G3": _g3, "G4": _g4, "G5": _g5, "G6": _g6, "G7": _g7, "G8": _g8,
    "G9": _g_zero, "G10": _g_zero, "G11": _g_zero, "G12": _g_zero, "G13": _g13,
}


def grammar_energy(program: SceneProgram, ctx: Optional[EnergyContext] = None, *, weights=None) -> GrammarEnergy:
    """G1-G13 항과 가중 합. total = Σ w·term (기본 w=1)."""
    if ctx is None:
        ctx = EnergyContext()
    w = {k: 1.0 for k in ENERGY_NAMES}
    if weights is not None:
        for k, v in dict(weights).items():
            if k not in w:
                raise ValueError(f"모르는 에너지 가중치 키 {k!r}")
            if isinstance(v, (bool, np.bool_)):
                raise ValueError(f"가중치 {k}: bool 불가")
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"가중치 {k}: 숫자가 아님") from e
            if not math.isfinite(f) or f < 0:
                raise ValueError(f"가중치 {k}={v!r}: 유한한 0 이상이어야 한다")
            w[k] = f
    terms = {k: float(RULES[k](program, ctx)) for k in ENERGY_NAMES}
    total = float(sum(w[k] * terms[k] for k in ENERGY_NAMES))
    return GrammarEnergy(terms=terms, weights=w, total=total)
