"""FND-03 결정론 유틸: Philox4x32 WNF, int64 고정소수점 segment-reduce, blake3 Merkle.

바이트 배치는 캐시 키·원장·샤드 주소로 디스크에 남으므로 계약(ops/harness/contracts/FND-03.md)이다.
CUDA 없이 import 가능하며 import 부작용이 없다.
"""
from __future__ import annotations

import os
import random
import struct
from collections.abc import Mapping

import numpy as np
import torch

__all__ = [
    "philox4x32", "wnf", "cell_id", "wnf_tile", "derive_seed",
    "fixed_encode", "fixed_decode", "canonical_order", "segment_reduce", "segment_reduce_int_torch",
    "hash_bytes", "hash_parts", "hash_array",
    "merkle_leaf", "merkle_node", "EMPTY_ROOT", "merkle_root", "merkle_proof", "merkle_verify",
    "merkle_root_keyed", "enable_determinism",
]

_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK32 = 0xFFFFFFFF


# --------------------------------------------------------------------------- 해시

def _blake3(data: bytes) -> bytes:
    import blake3
    return blake3.blake3(data).digest()


_BYTES_LIKE = (bytes, bytearray, memoryview)


def hash_bytes(data) -> bytes:
    if not isinstance(data, _BYTES_LIKE):
        raise TypeError("hash_bytes: bytes-like required")
    return _blake3(bytes(data))


def hash_parts(*parts) -> bytes:
    buf = [b"buc.parts.v1", struct.pack("<I", len(parts))]
    for p in parts:
        if isinstance(p, (bool, np.bool_)):
            raise TypeError("hash_parts: bool not allowed")
        if isinstance(p, _BYTES_LIKE):
            tag, payload = b"b", bytes(p)
        elif isinstance(p, str):
            tag, payload = b"s", p.encode("utf-8")
        elif isinstance(p, (int, np.integer)):
            v = int(p)
            if not (-(1 << 63) <= v < (1 << 64)):
                raise OverflowError("hash_parts: int out of range [-2^63, 2^64)")
            tag, payload = b"i", v.to_bytes(16, "little", signed=True)
        else:
            raise TypeError(f"hash_parts: unsupported part type {type(p).__name__}")
        buf.append(tag + struct.pack("<Q", len(payload)) + payload)
    return _blake3(b"".join(buf))


def _to_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def hash_array(a) -> bytes:
    arr = _to_numpy(a)
    if arr.dtype.kind not in "biufc":
        raise TypeError(f"hash_array: unsupported dtype {arr.dtype}")
    le = arr.dtype.newbyteorder("<")
    shape = arr.shape
    arr = np.ascontiguousarray(arr.astype(le, copy=False)).reshape(shape)
    head = b"buc.array.v1" + le.str.encode("ascii") + b"\0" + bytes([arr.ndim])
    head += b"".join(struct.pack("<Q", s) for s in arr.shape)
    return _blake3(head + arr.tobytes())


# --------------------------------------------------------------------------- Merkle

def merkle_leaf(x) -> bytes:
    if not isinstance(x, _BYTES_LIKE):
        raise TypeError("merkle_leaf: bytes-like required")
    return _blake3(b"\x00" + bytes(x))


def merkle_node(left: bytes, right: bytes) -> bytes:
    return _blake3(b"\x01" + bytes(left) + bytes(right))


EMPTY_ROOT = _blake3(b"buc.merkle.empty")


def _levels(items):
    level = [merkle_leaf(x) for x in items]
    levels = [level]
    while len(level) > 1:
        nxt = [merkle_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])  # 복제 없이 승격
        level = nxt
        levels.append(level)
    return levels


def merkle_root(items) -> bytes:
    levels = _levels(list(items))
    if not levels[0]:
        return EMPTY_ROOT
    return levels[-1][0]


def merkle_proof(items, index: int) -> list:
    items = list(items)
    n = len(items)
    if not (0 <= index < n):
        raise IndexError("merkle_proof: index out of range")
    levels = _levels(items)
    proof = []
    idx = index
    for level in levels[:-1]:
        m = len(level)
        if idx == m - 1 and m % 2 == 1:
            pass  # 승격 레벨: 형제 없음
        else:
            proof.append(level[idx ^ 1])
        idx //= 2
    return proof


def merkle_verify(item, index: int, n: int, proof, root) -> bool:
    try:
        if not (0 <= index < n):
            return False
        h = merkle_leaf(item)
        proof = list(proof)
        pos = 0
        idx, m = index, n
        while m > 1:
            if idx == m - 1 and m % 2 == 1:
                pass
            else:
                if pos >= len(proof):
                    return False
                sib = bytes(proof[pos])
                pos += 1
                h = merkle_node(sib, h) if idx % 2 else merkle_node(h, sib)
            idx //= 2
            m = (m + 1) // 2
        return pos == len(proof) and h == bytes(root)
    except Exception:
        return False


def _norm_key(k) -> bytes:
    if isinstance(k, str):
        return k.encode("utf-8")
    if isinstance(k, _BYTES_LIKE):
        return bytes(k)
    raise TypeError("merkle_root_keyed: key must be bytes or str")


def merkle_root_keyed(entries) -> bytes:
    pairs = entries.items() if isinstance(entries, Mapping) else entries
    norm = {}
    for k, v in pairs:
        kb = _norm_key(k)
        if not isinstance(v, _BYTES_LIKE):
            raise TypeError("merkle_root_keyed: value must be bytes-like")
        if kb in norm:
            raise ValueError("merkle_root_keyed: duplicate key")
        norm[kb] = bytes(v)
    leaves = [struct.pack("<Q", len(k)) + k + norm[k] for k in sorted(norm)]
    return merkle_root(leaves)


# --------------------------------------------------------------------------- Philox / WNF

def philox4x32(ctr, key, rounds: int = 10) -> np.ndarray:
    ctr = np.asarray(ctr)
    key = np.asarray(key)
    if ctr.dtype != np.uint32 or key.dtype != np.uint32:
        raise TypeError("philox4x32: uint32 arrays required")
    if ctr.ndim < 1 or ctr.shape[-1] != 4:
        raise ValueError("philox4x32: ctr last axis must be 4")
    if key.ndim < 1 or key.shape[-1] != 2:
        raise ValueError("philox4x32: key last axis must be 2")
    u = np.uint64
    c0, c1, c2, c3 = (ctr[..., i].astype(u) for i in range(4))
    k0, k1 = key[..., 0].astype(u), key[..., 1].astype(u)
    m0, m1, w0, w1, mask, s32 = u(_M0), u(_M1), u(_W0), u(_W1), u(_MASK32), u(32)
    for _ in range(rounds):
        p0 = m0 * c0
        p1 = m1 * c2
        c0, c1, c2, c3 = (p1 >> s32) ^ c1 ^ k0, p1 & mask, (p0 >> s32) ^ c3 ^ k1, p0 & mask
        k0 = (k0 + w0) & mask
        k1 = (k1 + w1) & mask
    shape = np.broadcast(c0, c1, c2, c3, k0, k1).shape
    out = np.empty(shape + (4,), dtype=np.uint32)
    for i, c in enumerate((c0, c1, c2, c3)):
        out[..., i] = c.astype(np.uint32)
    return out


def _check_int(v, name, lo, hi):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{name}: integer required")
    v = int(v)
    if not (lo <= v < hi):
        raise ValueError(f"{name} out of range [{lo}, {hi})")
    return v


def _cell_array(cell) -> np.ndarray:
    c = np.asarray(cell)
    if c.dtype.kind not in "iu":
        raise TypeError("cell: integer array required")
    if c.size:
        if c.dtype.kind == "i" and int(c.min()) < 0:
            raise ValueError("cell out of range [0, 2^63)")
        if c.dtype.kind == "u" and int(c.max()) >= (1 << 63):
            raise ValueError("cell out of range [0, 2^63)")
    return c.astype(np.uint64)


def wnf(seed, level, cell, k=0, channels=4, dtype=np.float32) -> np.ndarray:
    seed = _check_int(seed, "seed", 0, 1 << 64)
    level = _check_int(level, "level", 0, 1 << 16)
    k = _check_int(k, "k", 0, 1 << 32)
    channels = _check_int(channels, "channels", 1, 1 << 18)
    c = _cell_array(cell)
    lo = (c & np.uint64(_MASK32)).astype(np.uint32)
    hi = (c >> np.uint64(32)).astype(np.uint32)
    key = np.array([seed & _MASK32, seed >> 32], dtype=np.uint32)
    nblocks = (channels + 3) // 4
    ctr = np.empty(c.shape + (4,), dtype=np.uint32)
    ctr[..., 0] = lo
    ctr[..., 1] = hi
    ctr[..., 3] = np.uint32(k)
    blocks = []
    two_pi = 2.0 * np.pi
    for b in range(nblocks):
        ctr[..., 2] = np.uint32(level | (b << 16))
        w = philox4x32(ctr, key).astype(np.float64)
        u = (w + 0.5) / 4294967296.0
        r = np.sqrt(-2.0 * np.log(u[..., 0::2]))      # (.., 2): r1, r2
        ang = two_pi * u[..., 1::2]
        z = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)  # (.., 2, 2)
        blocks.append(z.reshape(c.shape + (4,)))
    out = np.concatenate(blocks, axis=-1)[..., :channels]
    return np.ascontiguousarray(out.astype(dtype))


def cell_id(i, j) -> np.int64:
    ia, ja = np.asarray(i), np.asarray(j)
    if ia.dtype.kind not in "iu" or ja.dtype.kind not in "iu":
        raise TypeError("cell_id: integer required")
    ia, ja = ia.astype(np.int64), ja.astype(np.int64)
    if ia.size and (ia.min() < 0 or ia.max() >= (1 << 32)):
        raise ValueError("cell_id: i out of range [0, 2^32)")
    if ja.size and (ja.min() < 0 or ja.max() >= (1 << 31)):
        raise ValueError("cell_id: j out of range [0, 2^31)")
    out = (ia << np.int64(31)) | ja
    return out if out.ndim else np.int64(out)


def wnf_tile(seed, level, i0, j0, h, w, k=0, channels=4, dtype=np.float32) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(i0, i0 + h, dtype=np.int64), np.arange(j0, j0 + w, dtype=np.int64),
                         indexing="ij")
    return wnf(seed, level, cell_id(ii, jj), k=k, channels=channels, dtype=dtype)


def derive_seed(seed, *parts) -> int:
    seed = _check_int(seed, "seed", 0, 1 << 64)
    return int.from_bytes(hash_parts(seed, *parts)[:8], "little")


# --------------------------------------------------------------------------- 고정소수점

_I63 = float(1 << 63)


def fixed_encode(x, frac_bits: int) -> np.ndarray:
    frac_bits = _check_int(frac_bits, "frac_bits", 1, 49)
    a = _to_numpy(x).astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError("fixed_encode: non-finite input")
    with np.errstate(over="ignore"):
        y = np.rint(a * float(1 << frac_bits))  # 2의 거듭제곱 곱셈은 정확, rint는 half-even
    if not (np.abs(y) < _I63).all():
        raise OverflowError("fixed_encode: result out of int64 range")
    return y.astype(np.int64)


def fixed_decode(q, frac_bits: int) -> np.ndarray:
    frac_bits = _check_int(frac_bits, "frac_bits", 1, 49)
    return np.ldexp(_to_numpy(q).astype(np.float64), -frac_bits)


def canonical_order(segment_ids, view_ids=None) -> np.ndarray:
    seg = _to_numpy(segment_ids)
    if view_ids is None:
        return np.argsort(seg, kind="stable").astype(np.int64)
    view = _to_numpy(view_ids)
    if view.shape != seg.shape:
        raise ValueError("canonical_order: length mismatch")
    return np.lexsort((view, seg)).astype(np.int64)


def _seg_sum_exact(q: np.ndarray, seg: np.ndarray, num_segments: int) -> np.ndarray:
    """int64 항을 정확히 세그먼트 합산. 중간 넘침은 무관, 최종 합이 int64 밖이면 OverflowError."""
    tail = q.shape[1:]
    flat = q.reshape(q.shape[0], int(np.prod(tail, dtype=np.int64)))
    lo = torch.from_numpy((flat & np.int64(_MASK32)).astype(np.int64))
    hi = torch.from_numpy((flat >> np.int64(32)).astype(np.int64))  # 산술 시프트
    idx = torch.from_numpy(seg.astype(np.int64))
    lo_s = torch.zeros((num_segments, flat.shape[1]), dtype=torch.int64).index_add_(0, idx, lo).numpy()
    hi_s = torch.zeros((num_segments, flat.shape[1]), dtype=torch.int64).index_add_(0, idx, hi).numpy()
    hi_t = hi_s + (lo_s >> np.int64(32))
    lo_r = lo_s & np.int64(_MASK32)
    if hi_t.size and (hi_t.min() < -(1 << 31) or hi_t.max() >= (1 << 31)):
        raise OverflowError("segment_reduce: final sum out of int64 range")
    out = (hi_t << np.int64(32)) | lo_r
    return out.reshape((num_segments,) + tail)


def segment_reduce(values, segment_ids, num_segments, view_ids=None, weights=None, frac_bits=24) -> np.ndarray:
    v = _to_numpy(values).astype(np.float64)
    seg = _to_numpy(segment_ids)
    num_segments = _check_int(num_segments, "num_segments", 0, 1 << 62)
    if seg.ndim != 1 or v.ndim < 1 or seg.shape[0] != v.shape[0]:
        raise ValueError("segment_reduce: length mismatch")
    n = v.shape[0]
    if seg.dtype.kind not in "iu":
        raise TypeError("segment_reduce: segment_ids must be integer")
    if view_ids is not None and _to_numpy(view_ids).shape != (n,):
        raise ValueError("segment_reduce: view_ids length mismatch")
    if n and (int(seg.min()) < 0 or int(seg.max()) >= num_segments):
        raise ValueError("segment_reduce: segment id out of range")
    if not np.isfinite(v).all():
        raise ValueError("segment_reduce: non-finite values")
    if weights is not None:
        w = _to_numpy(weights).astype(np.float64)
        if w.shape != (n,):
            raise ValueError("segment_reduce: weights length mismatch")
        if not np.isfinite(w).all():
            raise ValueError("segment_reduce: non-finite weights")
        with np.errstate(over="ignore"):
            v = v * w.reshape((n,) + (1,) * (v.ndim - 1))
    q = fixed_encode(v, frac_bits)
    return _seg_sum_exact(q, seg, num_segments)


def segment_reduce_int_torch(q, segment_ids, num_segments) -> torch.Tensor:
    if not isinstance(q, torch.Tensor) or q.dtype != torch.int64:
        raise TypeError("segment_reduce_int_torch: int64 Tensor required")
    seg = torch.as_tensor(segment_ids).to(q.device)
    if seg.dtype.is_floating_point or seg.dtype == torch.bool:
        raise TypeError("segment_reduce_int_torch: integer segment_ids required")
    seg = seg.to(torch.int64)
    if seg.ndim != 1 or q.ndim < 1 or seg.shape[0] != q.shape[0]:
        raise ValueError("segment_reduce_int_torch: length mismatch")
    if seg.numel() and (int(seg.min()) < 0 or int(seg.max()) >= num_segments):
        raise ValueError("segment_reduce_int_torch: segment id out of range")
    out = torch.zeros((num_segments,) + tuple(q.shape[1:]), dtype=torch.int64, device=q.device)
    return out.index_add_(0, seg, q)


# --------------------------------------------------------------------------- 솔버 경로

def enable_determinism(seed=None) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed & _MASK32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
