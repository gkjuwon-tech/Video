"""WLD-01 세계 컨테이너 STRATA-lite-min v0.

S2 이후 세계 W의 가변 작업 컨테이너다. LDI 층, 원경, meta/prov, 블록 표, 차트, TSQ(SoA) 표를 담고
Merkle 루트와 결정론 직렬화를 제공한다. 바깥으로는 FND-02 ``World`` 로 내보낸다(``to_world``).

바이트 배치(Merkle 섹션, 직렬화)는 계약(ops/harness/contracts/WLD-01.md)이다.
numpy 와 buc.core 만 쓴다. torch 는 쓰지 않고, import 부작용이 없다.
"""
from __future__ import annotations

import copy
import json
import math
import numbers
import os
import re
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from buc.core import contracts as C
from buc.core import det

__all__ = [
    "STRATA_SCHEMA", "H0_M", "BLOCK_CELLS", "LEVEL_RANGE", "MAGIC", "OWNERS", "BLOCK_STATES", "TSQ_FIELDS",
    "StrataError", "cell_size", "texel_pitch", "block_extent", "block_of", "canon_json",
    "Block", "TSQTable", "Strata",
]

STRATA_SCHEMA = C.WORLD_SCHEMA
H0_M = 64.0
BLOCK_CELLS = 16
LEVEL_RANGE = (-8, 24)
MAGIC = b"BUCSTRATA0"
OWNERS = {"NONE": 0, "INPUT": 1, "INPUT_PART": 2, "INPUT_REFL": 3, "ENGINE": 4, "TEXELFM": 5,
          "PROC": 6, "HUMAN": 7, "SOUL": 8}
BLOCK_STATES = ("empty", "bedrock", "proposed", "committed", "dirty")
TSQ_FIELDS = {
    "mu": ("<f4", (3,)),
    "normal": ("<f4", (3,)),
    "s_uv": ("<f4", (2,)),
    "opacity": ("<f4", ()),
    "edge_n": ("<f4", ()),
    "color": ("<f4", (3,)),
    "level": ("<i1", ()),
    "sem": ("<u2", ()),
    "own": ("<u2", ()),
    "unc": ("|u1", ()),
}

_OWNER_VALUES = frozenset(OWNERS.values())
_CHART_ID = re.compile(r"[A-Za-z0-9_.\-]+")
_RESERVED_META = ("strata_root", "strata_seed")
_I32 = 1 << 31


class StrataError(C.ContractError):
    """STRATA 컨테이너의 검증·역직렬화 실패."""


# --------------------------------------------------------------------------- 수치 보조

def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _check_level(l) -> int:
    if not _is_int(l):
        raise StrataError(f"level: 정수여야 한다 ({l!r})")
    l = int(l)
    if not (LEVEL_RANGE[0] <= l <= LEVEL_RANGE[1]):
        raise StrataError(f"level {l} 는 {LEVEL_RANGE} 밖")
    return l


def cell_size(l) -> float:
    """레벨 l 의 셀 한 변(m) = 64 * 2^-l (정확)."""
    return math.ldexp(H0_M, -_check_level(l))


def texel_pitch(l) -> float:
    return cell_size(l) / 8


def block_extent(l) -> float:
    return BLOCK_CELLS * cell_size(l)


def block_of(points, l) -> np.ndarray:
    """세계점 (N,3) -> 블록 키 int64 (N,4) = [l, floor(x/E), floor(y/E), floor(z/E)]."""
    l = _check_level(l)
    E = block_extent(l)
    try:
        p = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise StrataError(f"block_of: 수치 배열이어야 한다 ({e})") from e
    if p.ndim != 2 or p.shape[1] != 3:
        raise StrataError(f"block_of: shape {p.shape} != (N,3)")
    if p.size and not np.isfinite(p).all():
        raise StrataError("block_of: 유한하지 않은 좌표")
    q = np.floor(p / E)  # E 는 2의 거듭제곱이라 나눗셈이 정확하다
    if q.size and not (np.abs(q) < 2.0 ** 62).all():
        raise StrataError("block_of: 블록 인덱스가 int64 범위를 넘는다")
    out = np.empty((p.shape[0], 4), dtype=np.int64)
    out[:, 0] = l
    out[:, 1:] = q.astype(np.int64)
    return out


def _check_json_keys(x, path="$") -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            if not isinstance(k, str):
                raise StrataError(f"canon_json: dict 키는 str 이어야 한다 ({path}: {k!r})")
            _check_json_keys(v, f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _check_json_keys(v, f"{path}[{i}]")


def canon_json(x) -> bytes:
    """정규 JSON 바이트(키 정렬, 공백 없음, NaN/inf 금지)."""
    try:
        _check_json_keys(x)
        return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except StrataError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeEncodeError) as e:
        raise StrataError(f"canon_json: 직렬화할 수 없다 ({e})") from e


# --------------------------------------------------------------------------- Block

@dataclass(frozen=True)
class Block:
    state: str
    own: int
    unc: int = 255
    tsq_start: int = 0
    tsq_stop: int = 0

    def __post_init__(self):
        if not isinstance(self.state, str) or self.state not in BLOCK_STATES:
            raise StrataError(f"Block.state {self.state!r} 는 {BLOCK_STATES} 중 하나여야 한다")
        for name in ("own", "unc", "tsq_start", "tsq_stop"):
            v = getattr(self, name)
            if not _is_int(v):
                raise StrataError(f"Block.{name}: 정수여야 한다 ({v!r})")
            object.__setattr__(self, name, int(v))
        if self.own not in _OWNER_VALUES:
            raise StrataError(f"Block.own {self.own} 는 OWNERS 값이 아니다")
        if not (0 <= self.unc <= 255):
            raise StrataError(f"Block.unc {self.unc} 범위 밖 [0,255]")
        if not (0 <= self.tsq_start <= self.tsq_stop):
            raise StrataError(f"Block.tsq 범위 [{self.tsq_start},{self.tsq_stop}) 위반")

    def digest(self) -> bytes:
        return det.hash_parts(b"buc.strata.block.v0", self.state, self.own, self.unc, self.tsq_start, self.tsq_stop)


# --------------------------------------------------------------------------- TSQ

def _empty_columns() -> dict:
    return {name: np.zeros((0, *tail), dtype=np.dtype(dt)) for name, (dt, tail) in TSQ_FIELDS.items()}


def _coerce_tsq(cols: dict) -> tuple:
    """입력 열을 정본 dtype 으로 바꾸고 검증한다. (정본 열 dict, 행 수) 를 돌려준다."""
    if not isinstance(cols, dict):
        raise StrataError("TSQ: 열 dict 가 필요하다")
    missing = [k for k in TSQ_FIELDS if k not in cols]
    extra = [k for k in cols if k not in TSQ_FIELDS]
    if missing or extra:
        raise StrataError(f"TSQ: 필드 불일치 (누락 {missing}, 초과 {extra})")
    out = {}
    n = None
    for name, (dt, tail) in TSQ_FIELDS.items():
        try:
            a = np.asarray(cols[name])
        except (TypeError, ValueError) as e:
            raise StrataError(f"TSQ.{name}: 배열로 바꿀 수 없다 ({e})") from e
        if a.dtype.kind not in "iuf":
            raise StrataError(f"TSQ.{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
        if a.ndim != 1 + len(tail) or tuple(a.shape[1:]) != tuple(tail):
            raise StrataError(f"TSQ.{name}: shape {a.shape} != (n, {', '.join(map(str, tail))})")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise StrataError(f"TSQ.{name}: 행 수 {a.shape[0]} != {n}")
        target = np.dtype(dt)
        with np.errstate(over="ignore", invalid="ignore"):
            conv = np.ascontiguousarray(a, dtype=target)
        if target.kind in "iu":
            if a.dtype.kind == "f":
                if a.size and not (np.isfinite(a).all() and (np.floor(a) == a).all()):
                    raise StrataError(f"TSQ.{name}: 정수 열에 정수가 아닌 값")
                lossless = bool((conv.astype(np.float64) == a.astype(np.float64)).all()) if a.size else True
            else:
                lossless = bool((conv == a).all()) if a.size else True
            if not lossless:
                raise StrataError(f"TSQ.{name}: {target} 로 바꾸면 값이 변한다")
        out[name] = conv
    return out, int(n)


def _check_tsq_values(cols: dict) -> None:
    for name in ("mu", "normal", "s_uv", "opacity", "edge_n", "color"):
        a = cols[name]
        if a.size and not np.isfinite(a).all():
            raise StrataError(f"TSQ.{name}: 유한하지 않은 값")
    nrm = cols["normal"].astype(np.float64)
    if nrm.size and not (np.abs(np.linalg.norm(nrm, axis=1) - 1.0) <= 1e-3).all():
        raise StrataError("TSQ.normal: 단위 벡터가 아니다(atol 1e-3)")
    op = cols["opacity"]
    if op.size and not ((op >= 0).all() and (op <= 1).all()):
        raise StrataError("TSQ.opacity: [0,1] 밖")
    s = cols["s_uv"]
    if s.size and not (s > 0).all():
        raise StrataError("TSQ.s_uv: > 0 이어야 한다")
    e = cols["edge_n"]
    if e.size and not ((e >= 2).all() and (e <= 64).all()):
        raise StrataError("TSQ.edge_n: [2,64] 밖")
    col = cols["color"]
    if col.size and not (col >= 0).all():
        raise StrataError("TSQ.color: 음수")
    lv = cols["level"]
    if lv.size and not ((lv >= LEVEL_RANGE[0]).all() and (lv <= LEVEL_RANGE[1]).all()):
        raise StrataError(f"TSQ.level: {LEVEL_RANGE} 밖")
    own = cols["own"]
    if own.size and not np.isin(own, sorted(_OWNER_VALUES)).all():
        raise StrataError("TSQ.own: OWNERS 값이 아니다")


class TSQTable:
    """TSQ SoA 표. 열은 정본 dtype 의 C-연속 배열이다."""

    def __init__(self):
        self.columns = _empty_columns()

    @property
    def n(self) -> int:
        return int(self.columns["mu"].shape[0])

    def validate(self) -> None:
        if not isinstance(self.columns, dict) or list(self.columns) != list(TSQ_FIELDS):
            raise StrataError("TSQ: 열 키가 TSQ_FIELDS 와 다르다")
        n = None
        for name, (dt, tail) in TSQ_FIELDS.items():
            a = self.columns[name]
            if not isinstance(a, np.ndarray) or a.dtype != np.dtype(dt):
                raise StrataError(f"TSQ.{name}: dtype 이 {dt} 가 아니다")
            if a.ndim != 1 + len(tail) or tuple(a.shape[1:]) != tuple(tail):
                raise StrataError(f"TSQ.{name}: shape {a.shape}")
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                raise StrataError(f"TSQ.{name}: 행 수 불일치")
        _check_tsq_values(self.columns)

    def append(self, **cols) -> tuple:
        new, m = _coerce_tsq(cols)
        _check_tsq_values(new)
        start = self.n
        merged = {name: np.ascontiguousarray(np.concatenate([self.columns[name], new[name]], axis=0))
                  for name in TSQ_FIELDS}
        self.columns = merged
        return start, start + m


# --------------------------------------------------------------------------- Strata

def _check_block_key(key) -> tuple:
    if not isinstance(key, tuple) or len(key) != 4 or not all(_is_int(v) for v in key):
        raise StrataError(f"block 키는 int 4-튜플이어야 한다 ({key!r})")
    key = tuple(int(v) for v in key)
    if not (LEVEL_RANGE[0] <= key[0] <= LEVEL_RANGE[1]):
        raise StrataError(f"block 키 level {key[0]} 가 {LEVEL_RANGE} 밖")
    if not all(-_I32 <= v < _I32 for v in key[1:]):
        raise StrataError(f"block 키 {key} 의 i,j,k 가 int32 범위 밖")
    return key


def _check_chart_id(cid) -> str:
    if not isinstance(cid, str) or _CHART_ID.fullmatch(cid) is None:
        raise StrataError(f"chart_id {cid!r}: [A-Za-z0-9_.-]+ 이어야 한다")
    return cid


def _check_meta(meta) -> None:
    if not isinstance(meta, dict):
        raise StrataError("meta: dict 여야 한다")
    for k in _RESERVED_META:
        if k in meta:
            raise StrataError(f"meta: 예약 키 {k!r}")
    canon_json(meta)


def _wrap(fn, *a, **kw):
    """ContractError 를 StrataError 로 바꿔 던진다."""
    try:
        return fn(*a, **kw)
    except StrataError:
        raise
    except C.ContractError as e:
        raise StrataError(str(e)) from e


def _check_order(a: "C.LDILayer", b: "C.LDILayer", i: int) -> None:
    both = (a.rgba[..., 3] > 1e-3) & (b.rgba[..., 3] > 1e-3)
    if both.any():
        da = a.depth[both].astype(np.float64)
        db = b.depth[both].astype(np.float64)
        if not (da <= db * (1.0 + 1e-6)).all():
            raise StrataError(f"layers[{i}] 가 layers[{i + 1}] 보다 뒤에 있다 (앞->뒤 정렬 위반)")


class Strata:
    """STRATA-lite-min v0 작업 컨테이너."""

    schema = STRATA_SCHEMA

    def __init__(self, *, obs, theta_in, level, read_hash, prov, seed=0, meta=None, layers=(), far_field=None):
        self.obs = copy.deepcopy(obs)
        self.theta_in = copy.deepcopy(theta_in)
        self.level = level
        self.read_hash = read_hash
        self.prov = copy.deepcopy(prov)
        self.seed = seed
        self.meta = {} if meta is None else copy.deepcopy(meta)
        self.layers = []
        self.far_field = None
        self.blocks = {}
        self.charts = {}
        self.tsq = TSQTable()
        self.validate()
        for ly in layers:
            self.add_layer(ly)
        if far_field is not None:
            self.set_far_field(far_field)

    # ---- 검증
    def _validate_header(self) -> None:
        if not isinstance(self.obs, C.Obs):
            raise StrataError("obs: Obs 여야 한다")
        if not isinstance(self.theta_in, C.Theta):
            raise StrataError("theta_in: Theta 여야 한다")
        if not isinstance(self.prov, C.Prov):
            raise StrataError("prov: Prov 여야 한다")
        _wrap(self.obs.validate)
        _wrap(self.theta_in.validate)
        _wrap(self.prov.validate)
        if not _is_int(self.seed) or not (0 <= int(self.seed) < (1 << 128)):
            raise StrataError(f"seed: [0,2^128) 정수여야 한다 ({self.seed!r})")
        if not isinstance(self.level, str) or self.level not in C.LEVELS:
            raise StrataError(f"level {self.level!r} 는 {C.LEVELS} 중 하나여야 한다")
        if not isinstance(self.read_hash, str) or not self.read_hash:
            raise StrataError("read_hash: 비지 않은 str 이어야 한다")
        _check_meta(self.meta)

    def _check_layer(self, ly) -> None:
        if not isinstance(ly, C.LDILayer):
            raise StrataError(f"층은 LDILayer 여야 한다 ({type(ly).__name__})")
        _wrap(ly.validate)
        hw = (self.obs.height, self.obs.width)
        if tuple(ly.rgba.shape[:2]) != hw:
            raise StrataError(f"층 크기 {ly.rgba.shape[:2]} != obs {hw}")

    def validate(self) -> None:
        self._validate_header()
        if not isinstance(self.layers, list):
            raise StrataError("layers: list 여야 한다")
        for i, ly in enumerate(self.layers):
            self._check_layer(ly)
            if i:
                _check_order(self.layers[i - 1], ly, i - 1)
        if self.far_field is not None:
            if not isinstance(self.far_field, C.FarField):
                raise StrataError("far_field: FarField 여야 한다")
            _wrap(self.far_field.validate)
        if not isinstance(self.tsq, TSQTable):
            raise StrataError("tsq: TSQTable 이어야 한다")
        self.tsq.validate()
        n = self.tsq.n
        for key, b in self.blocks.items():
            _check_block_key(key)
            if not isinstance(b, Block):
                raise StrataError(f"block {key}: Block 이어야 한다")
            if b.tsq_stop > n:
                raise StrataError(f"block {key}: tsq 범위 [{b.tsq_start},{b.tsq_stop}) 가 [0,{n}] 밖")
        for cid, rec in self.charts.items():
            _check_chart_id(cid)
            if not isinstance(rec, dict):
                raise StrataError(f"chart {cid!r}: dict 여야 한다")
            canon_json(rec)

    # ---- 변경
    def add_layer(self, layer) -> int:
        self._check_layer(layer)
        if self.layers:
            _check_order(self.layers[-1], layer, len(self.layers) - 1)
        self.layers.append(layer)
        return len(self.layers) - 1

    def set_far_field(self, ff) -> None:
        if ff is not None:
            if not isinstance(ff, C.FarField):
                raise StrataError(f"far_field: FarField 여야 한다 ({type(ff).__name__})")
            _wrap(ff.validate)
        self.far_field = ff

    def set_block(self, key, block) -> None:
        key = _check_block_key(key)
        if not isinstance(block, Block):
            raise StrataError(f"Block 이어야 한다 ({type(block).__name__})")
        if block.tsq_stop > self.tsq.n:
            raise StrataError(f"block tsq 범위 [{block.tsq_start},{block.tsq_stop}) 가 [0,{self.tsq.n}] 밖")
        self.blocks[key] = block

    def get_block(self, key) -> Optional[Block]:
        return self.blocks.get(_check_block_key(key))

    def set_chart(self, chart_id, rec) -> None:
        _check_chart_id(chart_id)
        if not isinstance(rec, dict):
            raise StrataError("chart rec: dict 여야 한다")
        canon_json(rec)
        self.charts[chart_id] = copy.deepcopy(rec)

    def add_tsq(self, **cols) -> tuple:
        return self.tsq.append(**cols)

    # ---- 합성
    def composite(self) -> tuple:
        h, w = self.obs.height, self.obs.width
        acc = np.zeros((h, w, 3), dtype=np.float64)
        T = np.ones((h, w), dtype=np.float64)
        for ly in self.layers:
            a = ly.rgba[..., 3].astype(np.float64)
            acc += ly.rgba[..., :3].astype(np.float64) * (a * T)[..., None]
            T *= 1.0 - a
        return acc.astype(np.float32), (1.0 - T).astype(np.float32)

    # ---- Merkle
    def section_entries(self) -> dict:
        e = {
            "schema": STRATA_SCHEMA.encode("utf-8"),
            "level": self.level.encode("utf-8"),
            "read_hash": self.read_hash.encode("utf-8"),
            "obs": bytes.fromhex(self.obs.content_hash()),
            "theta_in": bytes.fromhex(self.theta_in.content_hash()),
            "prov": bytes.fromhex(self.prov.content_hash()),
            "seed": int(self.seed).to_bytes(16, "little"),
            "meta": det.hash_parts(b"buc.strata.meta.v0", canon_json(self.meta)),
        }
        for i, ly in enumerate(self.layers):
            e[f"layer/{i:04d}"] = det.hash_parts(b"buc.strata.layer.v0", det.hash_array(ly.rgba),
                                                 det.hash_array(ly.depth))
        if self.far_field is not None:
            e["far_field"] = det.hash_array(self.far_field.rgb)
        for (l, i, j, k), b in self.blocks.items():
            e[f"block/{l}/{i}/{j}/{k}"] = b.digest()
        for cid, rec in self.charts.items():
            e[f"chart/{cid}"] = det.hash_parts(b"buc.strata.chart.v0", canon_json(rec))
        for name in TSQ_FIELDS:
            e[f"tsq/{name}"] = det.hash_array(self.tsq.columns[name])
        return e

    def merkle_root(self) -> bytes:
        return det.merkle_root_keyed(self.section_entries())

    def proof(self, key: str) -> tuple:
        e = self.section_entries()
        if key not in e:
            raise KeyError(key)
        ks = sorted(e, key=lambda k: k.encode("utf-8"))
        leaves = []
        for k in ks:
            kb = k.encode("utf-8")
            leaves.append(struct.pack("<Q", len(kb)) + kb + bytes(e[k]))
        index = ks.index(key)
        return index, len(ks), det.merkle_proof(leaves, index), leaves[index]

    # ---- World 이음매
    def to_world(self) -> "C.World":
        self.validate()
        if len(self.layers) < 1:
            raise StrataError("to_world: 층이 1장 이상 필요하다")
        meta = copy.deepcopy(self.meta)
        meta["strata_root"] = self.merkle_root().hex()
        meta["strata_seed"] = f"{int(self.seed):032x}"
        layers = tuple(C.LDILayer(rgba=ly.rgba.copy(), depth=ly.depth.copy()) for ly in self.layers)
        ff = None if self.far_field is None else C.FarField(rgb=self.far_field.rgb.copy())
        return _wrap(C.World, obs=copy.deepcopy(self.obs), layers=layers, theta_in=copy.deepcopy(self.theta_in),
                     level=self.level, read_hash=self.read_hash, prov=copy.deepcopy(self.prov), far_field=ff,
                     meta=meta)

    @classmethod
    def from_world(cls, world) -> "Strata":
        if not isinstance(world, C.World):
            raise StrataError(f"from_world: World 여야 한다 ({type(world).__name__})")
        meta = copy.deepcopy(world.meta)
        meta.pop("strata_root", None)
        seed_hex = meta.pop("strata_seed", None)
        if seed_hex is None:
            seed = 0
        else:
            if not isinstance(seed_hex, str):
                raise StrataError("from_world: strata_seed 는 hex 문자열이어야 한다")
            try:
                seed = int(seed_hex, 16)
            except ValueError as e:
                raise StrataError(f"from_world: strata_seed 해석 실패 ({e})") from e
        layers = [C.LDILayer(rgba=ly.rgba.copy(), depth=ly.depth.copy()) for ly in world.layers]
        ff = None if world.far_field is None else C.FarField(rgb=world.far_field.rgb.copy())
        return cls(obs=world.obs, theta_in=world.theta_in, level=world.level, read_hash=world.read_hash,
                   prov=world.prov, seed=seed, meta=meta, layers=layers, far_field=ff)

    # ---- 직렬화
    def to_bytes(self) -> bytes:
        """MAGIC | u64le 헤더 길이 | 정규 JSON 헤더 | 배열 본문 | blake3(앞 전부) 32바이트."""
        self.validate()
        header = {
            "v": 0,
            "schema": STRATA_SCHEMA,
            "root": self.merkle_root().hex(),
            "level": self.level,
            "read_hash": self.read_hash,
            "seed": f"{int(self.seed):032x}",
            "meta": self.meta,
            "obs": C.dumps(self.obs),
            "theta_in": C.dumps(self.theta_in),
            "prov": C.dumps(self.prov),
            "layer_shapes": [list(ly.rgba.shape) for ly in self.layers],
            "far_shape": None if self.far_field is None else list(self.far_field.rgb.shape),
            "blocks": [[*k, b.state, b.own, b.unc, b.tsq_start, b.tsq_stop]
                       for k, b in sorted(self.blocks.items())],
            "charts": {cid: self.charts[cid] for cid in sorted(self.charts)},
            "tsq_n": self.tsq.n,
        }
        hb = canon_json(header)
        parts = [MAGIC, struct.pack("<Q", len(hb)), hb]
        for ly in self.layers:
            parts.append(np.ascontiguousarray(ly.rgba, dtype="<f4").tobytes())
            parts.append(np.ascontiguousarray(ly.depth, dtype="<f4").tobytes())
        if self.far_field is not None:
            parts.append(np.ascontiguousarray(self.far_field.rgb, dtype="<f4").tobytes())
        for name, (dt, _tail) in TSQ_FIELDS.items():
            parts.append(np.ascontiguousarray(self.tsq.columns[name], dtype=np.dtype(dt)).tobytes())
        body = b"".join(parts)
        return body + det.hash_bytes(body)

    @classmethod
    def from_bytes(cls, data) -> "Strata":
        try:
            return cls._from_bytes(data)
        except StrataError:
            raise
        except (C.ContractError, ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError,
                struct.error, UnicodeDecodeError, MemoryError) as e:
            raise StrataError(f"from_bytes: 해석 실패 ({type(e).__name__}: {e})") from e

    @classmethod
    def _from_bytes(cls, data) -> "Strata":
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise StrataError("from_bytes: bytes 가 필요하다")
        data = bytes(data)
        if len(data) < len(MAGIC) + 8 + 32 or not data.startswith(MAGIC):
            raise StrataError("from_bytes: MAGIC 불일치 또는 너무 짧다")
        body, tail = data[:-32], data[-32:]
        if det.hash_bytes(body) != tail:
            raise StrataError("from_bytes: 무결성 해시 불일치(변조 또는 잘림)")
        pos = len(MAGIC)
        (hlen,) = struct.unpack_from("<Q", body, pos)
        pos += 8
        if hlen > len(body) - pos:
            raise StrataError("from_bytes: 헤더 길이 오류")
        header = json.loads(body[pos:pos + hlen].decode("utf-8"))
        pos += hlen
        if not isinstance(header, dict) or header.get("v") != 0 or header.get("schema") != STRATA_SCHEMA:
            raise StrataError("from_bytes: 헤더 버전/스키마 불일치")

        def take(dtype, shape):
            nonlocal pos
            if (not isinstance(shape, (list, tuple)) or not all(_is_int(s) and 0 <= s < (1 << 40) for s in shape)):
                raise StrataError("from_bytes: shape 오류")
            count = 1
            for s in shape:
                count *= int(s)
            dt = np.dtype(dtype)
            nbytes = count * dt.itemsize
            if nbytes > len(body) - pos:
                raise StrataError("from_bytes: 본문이 짧다")
            a = np.frombuffer(body, dtype=dt, count=count, offset=pos).reshape(tuple(shape)).copy()
            pos += nbytes
            return a

        obs = C.loads(header["obs"])
        theta_in = C.loads(header["theta_in"])
        prov = C.loads(header["prov"])
        seed_hex = header["seed"]
        if not isinstance(seed_hex, str):
            raise StrataError("from_bytes: seed 형식 오류")
        st = cls(obs=obs, theta_in=theta_in, level=header["level"], read_hash=header["read_hash"], prov=prov,
                 seed=int(seed_hex, 16), meta=header["meta"])
        for shape in header["layer_shapes"]:
            rgba = take("<f4", shape)
            depth = take("<f4", list(shape)[:2])
            st.add_layer(C.LDILayer(rgba=rgba, depth=depth))
        if header["far_shape"] is not None:
            st.set_far_field(C.FarField(rgb=take("<f4", header["far_shape"])))
        n = header["tsq_n"]
        if not _is_int(n) or n < 0:
            raise StrataError("from_bytes: tsq_n 오류")
        cols = {name: take(np.dtype(dt), (n, *tail)) for name, (dt, tail) in TSQ_FIELDS.items()}
        if pos != len(body):
            raise StrataError("from_bytes: 본문 끝에 남는 바이트가 있다")
        st.add_tsq(**cols)
        for rec in header["blocks"]:
            if not isinstance(rec, list) or len(rec) != 9:
                raise StrataError("from_bytes: 블록 레코드 오류")
            st.set_block(tuple(rec[:4]), Block(state=rec[4], own=rec[5], unc=rec[6], tsq_start=rec[7],
                                               tsq_stop=rec[8]))
        for cid, rec in header["charts"].items():
            st.set_chart(cid, rec)
        if st.merkle_root().hex() != header["root"]:
            raise StrataError("from_bytes: 재계산한 Merkle 루트가 저장된 루트와 다르다")
        return st

    def save(self, path) -> None:
        data = self.to_bytes()
        tmp = f"{os.fspath(path)}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> "Strata":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())
