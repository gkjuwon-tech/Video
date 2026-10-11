"""RDR-09 Proof-of-Shot / 왕복 비트 동일 검증 v0.

계약서: ops/harness/contracts/RDR-09.md. 프레임 지문(Merkle 잎) -> 루트 -> ProofOfShot, 재렌더 검증,
정확히 대칭인 왕복 경로의 비트 동일 판정. 영상 인코딩은 범위 밖이며 증거는 Frames 바이트 기준이다.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from buc.core import contracts as C
from buc.core import det
from buc.render import aov as A
from buc.render.quads import RenderOpts

__all__ = [
    "SCHEMA", "RENDERER_PIN", "ROUNDTRIP_KEYS", "PosError", "toolchain_pin", "frame_leaf", "frame_digests",
    "leaf_root", "IRGeo", "ir_geo", "opts_dict", "ProofOfShot", "Verdict", "make_proof", "verify_proof",
    "inclusion_proof", "verify_inclusion", "write_proof", "read_proof", "mirror_index", "out_and_back",
    "is_palindromic_shot", "lsb_diff", "RoundTrip", "roundtrip_check", "verify_roundtrip",
]

SCHEMA = "buc.pos/v1"
RENDERER_PIN = "buc.render.aov/rdr-05+quads/rdr-02"
ROUNDTRIP_KEYS = ("alpha", "atom_id", "depth", "layer_id", "normal")

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class PosError(ValueError):
    pass


def _is_hex64(x) -> bool:
    return isinstance(x, str) and _HEX64.match(x) is not None


def toolchain_pin() -> dict:
    import blake3
    import scipy
    import torch
    return {
        "renderer": RENDERER_PIN,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": str(torch.__version__),
        "blake3": str(getattr(blake3, "__version__", "unknown")),
        "machine": platform.machine() or "unknown",
        "system": platform.system() or "unknown",
    }


# ---------------------------------------------------------------------------------------------------------------
# 지문
# ---------------------------------------------------------------------------------------------------------------
def _resolve_keys(frames, keys):
    if keys is None:
        return sorted(frames.aov)
    ks = sorted(set(keys))
    for k in ks:
        if k not in frames.aov:
            raise ValueError(f"frame_leaf: aov 키 없음 ({k!r})")
    return ks


def _leaf(frames, i, ks) -> str:
    parts = [b"buc.pos.frame.v1", det.hash_array(frames.rgb[i])]
    for k in ks:
        parts += [k, det.hash_array(frames.aov[k][i])]
    return det.hash_parts(*parts).hex()


def frame_leaf(frames, i, *, keys=None) -> str:
    ks = _resolve_keys(frames, keys)
    n = frames.rgb.shape[0]
    i = int(i)
    if not (0 <= i < n):
        raise IndexError(f"frame_leaf: i={i} 범위 밖 (n={n})")
    return _leaf(frames, i, ks)


def frame_digests(frames, *, keys=None) -> tuple:
    ks = _resolve_keys(frames, keys)
    return tuple(_leaf(frames, i, ks) for i in range(frames.rgb.shape[0]))


def leaf_root(leaves) -> str:
    items = []
    for x in leaves:
        if not _is_hex64(x):
            raise ValueError(f"leaf_root: 64자 소문자 hex 가 아니다 ({x!r})")
        items.append(bytes.fromhex(x))
    return det.merkle_root(items).hex()


# ---------------------------------------------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IRGeo:
    per_frame: tuple
    mean: float
    max: float


def ir_geo(atom_id, bits) -> IRGeo:
    a = np.asarray(atom_id)
    if a.ndim != 3:
        raise ValueError(f"ir_geo: atom_id 는 (N,H,W) 이어야 한다 (shape {a.shape})")
    obs = bits.test(a)
    valid = a >= 0
    n = a.shape[0]
    o = obs.reshape(n, -1).sum(axis=1).astype(np.int64)
    v = valid.reshape(n, -1).sum(axis=1).astype(np.int64)
    per = tuple((1.0 - int(o[i]) / int(v[i])) if int(v[i]) > 0 else 0.0 for i in range(n))
    sv = int(v.sum())
    mean = (1.0 - int(o.sum()) / sv) if sv > 0 else 0.0
    mx = max(per) if per else 0.0
    return IRGeo(per_frame=per, mean=float(mean), max=float(mx))


# ---------------------------------------------------------------------------------------------------------------
# 증명
# ---------------------------------------------------------------------------------------------------------------
def opts_dict(opts=None) -> dict:
    o = RenderOpts() if opts is None else opts
    return {"bg": [float(x) for x in o.bg], "edge_px": float(o.edge_px), "z_tol_rel": float(o.z_tol_rel),
            "alpha_eps": float(o.alpha_eps), "near_m": float(o.near_m),
            "max_footprint_px": float(o.max_footprint_px)}


def _check_json(v, path="extra"):
    if v is None or isinstance(v, (bool, str, int)):
        return
    if isinstance(v, float):
        if not math.isfinite(v):
            raise PosError(f"{path}: 유한하지 않은 float")
        return
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                raise PosError(f"{path}: str 아닌 키 {k!r}")
            _check_json(x, f"{path}.{k}")
        return
    if isinstance(v, list):
        for j, x in enumerate(v):
            _check_json(x, f"{path}[{j}]")
        return
    raise PosError(f"{path}: JSON 값이 아니다 ({type(v).__name__})")


_FIELDS = ("schema", "shot_hash", "world_hash", "theta_hash", "toolchain", "opts", "n_frames", "width", "height",
           "fps", "aov_keys", "leaves", "root", "ir", "roundtrip", "extra")


@dataclass(frozen=True, kw_only=True)
class ProofOfShot:
    schema: str
    shot_hash: str
    world_hash: str
    theta_hash: str
    toolchain: dict
    opts: dict
    n_frames: int
    width: int
    height: int
    fps: float
    aov_keys: tuple
    leaves: tuple
    root: str
    ir: dict
    roundtrip: object
    extra: dict

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise PosError(f"schema 불일치 ({self.schema!r})")
        for n in ("shot_hash", "world_hash", "theta_hash", "root"):
            if not _is_hex64(getattr(self, n)):
                raise PosError(f"{n}: 64자 소문자 hex 가 아니다")
        for n in ("n_frames", "width", "height"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int):
                raise PosError(f"{n}: int 가 아니다")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise PosError("fps: 수가 아니다")
        object.__setattr__(self, "fps", float(self.fps))
        for n in ("toolchain", "opts", "ir", "extra"):
            if not isinstance(getattr(self, n), dict):
                raise PosError(f"{n}: dict 가 아니다")
        if self.roundtrip is not None and not isinstance(self.roundtrip, dict):
            raise PosError("roundtrip: dict 또는 None")
        try:
            leaves = tuple(self.leaves)
            keys = tuple(self.aov_keys)
        except TypeError as e:
            raise PosError("leaves/aov_keys: 열거 가능해야 한다") from e
        if not all(isinstance(k, str) for k in keys):
            raise PosError("aov_keys: str 이어야 한다")
        if len(leaves) != self.n_frames:
            raise PosError("len(leaves) != n_frames")
        if not all(_is_hex64(x) for x in leaves):
            raise PosError("leaves: 64자 소문자 hex 가 아닌 잎")
        object.__setattr__(self, "leaves", leaves)
        object.__setattr__(self, "aov_keys", keys)
        for n in ("toolchain", "opts", "ir", "roundtrip", "extra"):
            object.__setattr__(self, n, copy.deepcopy(getattr(self, n)))

    def to_dict(self) -> dict:
        d = {n: copy.deepcopy(getattr(self, n)) for n in _FIELDS}
        d["aov_keys"] = list(self.aov_keys)
        d["leaves"] = list(self.leaves)
        return d

    @classmethod
    def from_dict(cls, d) -> "ProofOfShot":
        if not isinstance(d, dict):
            raise PosError("from_dict: dict 가 아니다")
        if set(d) != set(_FIELDS):
            miss = sorted(set(_FIELDS) - set(d))
            extra = sorted(map(str, set(d) - set(_FIELDS)))
            raise PosError(f"from_dict: 키 불일치 (빠짐 {miss}, 모름 {extra})")
        if not isinstance(d["aov_keys"], (list, tuple)) or not isinstance(d["leaves"], (list, tuple)):
            raise PosError("from_dict: aov_keys/leaves 는 list")
        kw = {n: copy.deepcopy(d[n]) for n in _FIELDS}
        return cls(**kw)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, s) -> "ProofOfShot":
        def _bad(c):
            raise ValueError(f"비유한 상수 {c}")
        try:
            d = json.loads(s, parse_constant=_bad)
        except (ValueError, TypeError) as e:
            raise PosError(f"from_json: JSON 오류 ({e})") from e
        return cls.from_dict(d)

    def content_hash(self) -> str:
        return det.hash_parts(b"buc.pos.v1", self.to_json().encode("utf-8")).hex()


def write_proof(proof, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes((proof.to_json() + "\n").encode("utf-8"))
    os.replace(tmp, p)
    return p


def read_proof(path) -> ProofOfShot:
    p = Path(path)
    data = p.read_bytes()                       # 없음 -> FileNotFoundError
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PosError("read_proof: UTF-8 이 아니다") from e
    return ProofOfShot.from_json(text)


# ---------------------------------------------------------------------------------------------------------------
# 만들기
# ---------------------------------------------------------------------------------------------------------------
def _roundtrip_claim(shot, frames):
    n = frames.rgb.shape[0]
    if n < 2 or not is_palindromic_shot(shot):
        return None
    r = roundtrip_check(frames)
    return {"keys": list(r.keys), "pairs": int(r.pairs), "identical": bool(r.identical),
            "max_abs_diff": float(r.max_abs_diff)}


def _ir_claim(world, frames):
    bits = A.observed_atoms(world)
    r = ir_geo(frames.aov["atom_id"], bits)
    return {"obs_hash": bits.content_hash(), "ir_geo_mean": r.mean, "ir_geo_max": r.max,
            "per_frame": list(r.per_frame)}


def make_proof(world, shot, *, opts=None, frames=None, extra=None) -> ProofOfShot:
    if not isinstance(world, C.World):
        raise TypeError("make_proof: world 는 World")
    if not isinstance(shot, C.Shot):
        raise TypeError("make_proof: shot 은 Shot")
    ex = {} if extra is None else extra
    if not isinstance(ex, dict):
        raise PosError("extra: dict 이어야 한다")
    _check_json(ex)
    ex = copy.deepcopy(ex)
    shot_hash, world_hash = shot.content_hash(), world.content_hash()
    if frames is None:
        frames = A.render_shot_aov(world, shot, opts)
    else:
        if frames.shot_hash != shot_hash:
            raise ValueError("make_proof: frames.shot_hash 가 shot 과 다르다")
        if frames.world_hash != world_hash:
            raise ValueError("make_proof: frames.world_hash 가 world 와 다르다")
    if "atom_id" not in frames.aov:
        raise PosError("make_proof: frames 에 atom_id aov 가 없다")
    leaves = frame_digests(frames)
    return ProofOfShot(
        schema=SCHEMA, shot_hash=shot_hash, world_hash=world_hash, theta_hash=shot.theta.content_hash(),
        toolchain=toolchain_pin(), opts=opts_dict(opts), n_frames=int(frames.rgb.shape[0]),
        width=int(frames.rgb.shape[2]), height=int(frames.rgb.shape[1]), fps=float(frames.fps),
        aov_keys=tuple(sorted(frames.aov)), leaves=leaves, root=leaf_root(leaves),
        ir=_ir_claim(world, frames), roundtrip=_roundtrip_claim(shot, frames), extra=ex)


# ---------------------------------------------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Verdict:
    ok: bool
    world_ok: bool
    shot_ok: bool
    theta_ok: bool
    toolchain_ok: bool
    shape_ok: bool
    root_ok: bool
    frames_ok: bool
    ir_ok: bool
    roundtrip_ok: bool
    bad_frames: tuple
    reason: str


def _opts_from_dict(d) -> RenderOpts:
    return RenderOpts(bg=tuple(d["bg"]), edge_px=d["edge_px"], z_tol_rel=d["z_tol_rel"], alpha_eps=d["alpha_eps"],
                      near_m=d["near_m"], max_footprint_px=d["max_footprint_px"])


def verify_proof(proof, world, shot, *, frames=None) -> Verdict:
    world_ok = world.content_hash() == proof.world_hash
    shot_ok = shot.content_hash() == proof.shot_hash
    theta_ok = shot.theta.content_hash() == proof.theta_hash
    toolchain_ok = proof.toolchain == toolchain_pin()
    root_ok = leaf_root(proof.leaves) == proof.root

    if frames is None:
        try:
            frames = A.render_shot_aov(world, shot, _opts_from_dict(proof.opts))
        except Exception:
            frames = None
    shape_ok = False
    if frames is not None:
        try:
            shape_ok = (frames.rgb.shape[0] == proof.n_frames and frames.rgb.shape[1] == proof.height
                        and frames.rgb.shape[2] == proof.width and float(frames.fps) == proof.fps
                        and tuple(sorted(frames.aov)) == tuple(proof.aov_keys))
        except Exception:
            shape_ok = False
    shape_ok = bool(shape_ok)

    frames_ok = ir_ok = roundtrip_ok = False
    bad = ()
    if shape_ok:
        digs = frame_digests(frames, keys=proof.aov_keys)
        bad = tuple(i for i in range(proof.n_frames) if digs[i] != proof.leaves[i])
        frames_ok = not bad
        try:
            ir_ok = _ir_claim(world, frames) == proof.ir
        except Exception:
            ir_ok = False
        try:
            roundtrip_ok = _roundtrip_claim(shot, frames) == proof.roundtrip
        except Exception:
            roundtrip_ok = False

    checks = (("world_ok", world_ok), ("shot_ok", shot_ok), ("theta_ok", theta_ok), ("shape_ok", shape_ok),
              ("root_ok", root_ok), ("frames_ok", frames_ok), ("ir_ok", ir_ok), ("roundtrip_ok", roundtrip_ok))
    failed = [n for n, v in checks if not v]
    return Verdict(ok=not failed, world_ok=bool(world_ok), shot_ok=bool(shot_ok), theta_ok=bool(theta_ok),
                   toolchain_ok=bool(toolchain_ok), shape_ok=shape_ok, root_ok=bool(root_ok),
                   frames_ok=bool(frames_ok), ir_ok=bool(ir_ok), roundtrip_ok=bool(roundtrip_ok),
                   bad_frames=bad, reason=",".join(failed))


def inclusion_proof(proof, i) -> list:
    items = [bytes.fromhex(x) for x in proof.leaves]
    return [x.hex() for x in det.merkle_proof(items, int(i))]


def verify_inclusion(proof, i, leaf, path) -> bool:
    try:
        if not _is_hex64(leaf):
            return False
        pth = []
        for x in path:
            if not _is_hex64(x):
                return False
            pth.append(bytes.fromhex(x))
        return bool(det.merkle_verify(bytes.fromhex(leaf), int(i), proof.n_frames, pth, bytes.fromhex(proof.root)))
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------
# 왕복
# ---------------------------------------------------------------------------------------------------------------
def mirror_index(n) -> np.ndarray:
    n = int(n)
    if n < 0:
        raise ValueError("mirror_index: n >= 0")
    return (n - 1 - np.arange(n)).astype(np.int64)


def out_and_back(shot) -> C.Shot:
    n = shot.n_frames
    if n < 2:
        raise ValueError("out_and_back: 프레임이 2장 이상이어야 한다")
    idx = np.concatenate([np.arange(n), np.arange(n - 2, -1, -1)]).astype(np.int64)
    m = idx.shape[0]
    return C.Shot(c2w=shot.c2w[idx].copy(), t=np.arange(m) / shot.fps, K=shot.K[idx].copy(),
                  focus_m=shot.focus_m[idx].copy(), width=shot.width, height=shot.height, fps=shot.fps,
                  theta=shot.theta, move=shot.move, rig=shot.rig, seed=shot.seed)


def is_palindromic_shot(shot, *, atol=0.0) -> bool:
    for name in ("c2w", "K", "focus_m"):
        a = np.ascontiguousarray(getattr(shot, name))
        b = np.ascontiguousarray(a[::-1])
        if atol == 0:
            if a.tobytes() != b.tobytes():
                return False
        elif not np.allclose(a, b, rtol=0.0, atol=float(atol)):
            return False
    return True


def lsb_diff(a, b) -> int:
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if x.shape != y.shape:
        raise ValueError(f"lsb_diff: 모양 불일치 {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0
    qx = np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.int64)
    qy = np.rint(np.clip(y, 0.0, 1.0) * 255.0).astype(np.int64)
    return int(np.abs(qx - qy).max())


@dataclass(frozen=True, eq=False)
class RoundTrip:
    n_frames: int
    pairs: int
    identical: bool
    pair_identical: tuple
    first_mismatch: object
    keys: tuple
    max_abs_diff: float
    max_lsb_diff: int
    diff_map: np.ndarray


def roundtrip_check(frames, *, keys=None) -> RoundTrip:
    if keys is None:
        ks = sorted(k for k in ROUNDTRIP_KEYS if k in frames.aov)
    else:
        ks = _resolve_keys(frames, keys)
    n = int(frames.rgb.shape[0])
    pairs = n // 2
    h, w = frames.rgb.shape[1], frames.rgb.shape[2]
    diff_map = np.zeros((pairs, h, w), np.float32)
    pair_ident = []
    lsb = 0
    for i in range(pairs):
        j = n - 1 - i
        a, b = frames.rgb[i], frames.rgb[j]
        same = np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
        for k in ks:
            if not same:
                break
            same = (np.ascontiguousarray(frames.aov[k][i]).tobytes()
                    == np.ascontiguousarray(frames.aov[k][j]).tobytes())
        # aov 비교는 위에서 단락되므로 aov 만 다른 경우도 same=False 로 남는다
        pair_ident.append(bool(same))
        diff_map[i] = np.abs(a - b).max(axis=-1)
        lsb = max(lsb, lsb_diff(a, b))
    first = next((i for i, v in enumerate(pair_ident) if not v), None)
    return RoundTrip(n_frames=n, pairs=pairs, identical=all(pair_ident), pair_identical=tuple(pair_ident),
                     first_mismatch=first, keys=tuple(ks),
                     max_abs_diff=float(diff_map.max()) if diff_map.size else 0.0, max_lsb_diff=int(lsb),
                     diff_map=diff_map)


def verify_roundtrip(world, shot, opts=None) -> RoundTrip:
    if not is_palindromic_shot(shot):
        raise ValueError("verify_roundtrip: 정확히 대칭인 경로가 아니다")
    return roundtrip_check(A.render_shot_aov(world, shot, opts))
