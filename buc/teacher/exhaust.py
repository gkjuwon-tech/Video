"""TCH-04 Exhaust Ledger v0: blake3 키, int8 x̂0 저장, ext/hold 분리, 세션 재개.

교사 호출 산출(x̂0)을 키로 주소 지정해 int8 로 저장·재사용한다. 키는 방화벽의 일부다:
(ext, comp, hold, ns) 비트가 키에 들어가고, ext 키는 프로젝트 키 없이는 계산·열거할 수 없다.
암호는 구현하지 않는다(Sealer 를 꽂는 쪽의 몫). sealer 없는 ext 쓰기·읽기는 FirewallError.
torch, numpy, blake3 와 buc.core.det / buc.gov.prov / buc.teacher.sample 만 import 하며 부작용이 없다.
"""
from __future__ import annotations

import json
import os
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import blake3
import numpy as np
import torch

from buc.core import det
from buc.gov import prov as _prov
from buc.gov.prov import FirewallError, NAMESPACES, ProvRecord
from buc.teacher.sample import SigmaError, SigmaTable

__all__ = [
    "KEY_TAG", "QUANT_MODE", "Q_MAX", "EXT_TTL_S", "PURPOSES", "ExhaustError", "ExhaustSpecError",
    "ExhaustNumericError", "FirewallError", "quantize_x0", "dequantize_x0", "KeyParts", "compute_key",
    "teacher_id", "Sealer", "ExhaustEntry", "PutResult", "allowed_purposes", "ExhaustLedger",
]

KEY_TAG = "buc.tch04.key.v1"
QUANT_MODE = "int8-sym-ch-v1"
Q_MAX = 127
EXT_TTL_S = 72 * 3600
PURPOSES = ("resume", "delta", "share", "train")

_PROJ_TAG = "buc.tch04.proj.v1"
_MAGIC = b"BUCX1"
_SEAL_MAGIC = b"BUCS1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_KEY_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXT_ZONES = ("user_vault", "ext_vault")


class ExhaustError(RuntimeError):
    """TCH-04 기본 예외."""


class ExhaustSpecError(ExhaustError, ValueError):
    """입력·사양 위반."""


class ExhaustNumericError(ExhaustError, FloatingPointError):
    """비유한 값."""


class _Corrupt(Exception):
    """내부용: 깨진 파일."""


# ----------------------------------------------------------------------------- 양자화

def quantize_x0(x0: torch.Tensor):
    """채널별 대칭 int8. 반환 (q int8[C,h,w], scale float32[C])."""
    if not isinstance(x0, torch.Tensor):
        raise ExhaustSpecError(f"Tensor 여야 한다: {type(x0).__name__}")
    if not x0.is_floating_point():
        raise ExhaustSpecError(f"부동소수 dtype 이어야 한다: {x0.dtype}")
    if x0.ndim != 3:
        raise ExhaustSpecError(f"[C,h,w] 여야 한다: shape={tuple(x0.shape)}")
    if x0.numel() == 0:
        raise ExhaustSpecError("빈 텐서")
    with torch.no_grad():
        x = x0.detach()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not bool(torch.isfinite(x).all()):
            raise ExhaustNumericError("x0 에 비유한 값이 있다")
        scale = (x.abs().amax(dim=(1, 2)) / float(Q_MAX)).to(torch.float32)
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.round(x / safe[:, None, None]).clamp_(-Q_MAX, Q_MAX).to(torch.int8)
        q = torch.where((scale == 0)[:, None, None], torch.zeros_like(q), q)
        return q.contiguous(), scale.contiguous()


def dequantize_x0(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return q.detach().float() * scale.detach().to(torch.float32)[:, None, None]


# ----------------------------------------------------------------------------- 키

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _need_str(name: str, v) -> None:
    if not isinstance(v, str) or not v:
        raise ExhaustSpecError(f"{name}: 비어 있지 않은 str 이어야 한다: {v!r}")


@dataclass(frozen=True)
class KeyParts:
    world: bytes
    view: str
    sigma: float
    seed: int
    teacher_id: str
    caption_hash: str
    table: SigmaTable
    height: int
    width: int
    level: int = 0
    k: int = 0
    precision: str = "bf16"
    quant: str = QUANT_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.world, bytes) or len(self.world) != 32:
            raise ExhaustSpecError("world: 32바이트 bytes 여야 한다")
        _need_str("view", self.view)
        if not isinstance(self.table, SigmaTable):
            raise ExhaustSpecError("table: SigmaTable 이어야 한다")
        self.table.index_of(self.sigma)          # 표의 점이 아니면 SigmaError
        if not _is_int(self.seed) or not (0 <= self.seed < (1 << 64)):
            raise ExhaustSpecError(f"seed: [0, 2^64) 정수여야 한다: {self.seed!r}")
        _need_str("teacher_id", self.teacher_id)
        if not isinstance(self.caption_hash, str) or not _HEX64.match(self.caption_hash):
            raise ExhaustSpecError("caption_hash: 소문자 hex 64자여야 한다")
        for n in ("height", "width"):
            v = getattr(self, n)
            if not _is_int(v) or v <= 0:
                raise ExhaustSpecError(f"{n}: 양의 정수여야 한다: {v!r}")
        for n in ("level", "k"):
            v = getattr(self, n)
            if not _is_int(v) or v < 0:
                raise ExhaustSpecError(f"{n}: 0 이상 정수여야 한다: {v!r}")
        _need_str("precision", self.precision)
        _need_str("quant", self.quant)


def _check_project(project_id, project_key, rec: ProvRecord):
    """ext 에서만 자격을 검사한다. 반환 (proj, key)."""
    if not rec.ext:
        return "", None
    if project_id is None or project_key is None:
        raise FirewallError("ext 항목은 project_id 와 project_key 가 필요하다")
    if not isinstance(project_id, str) or not project_id:
        raise ExhaustSpecError("project_id: 비어 있지 않은 str 이어야 한다")
    if not isinstance(project_key, bytes) or len(project_key) != 32:
        raise ExhaustSpecError("project_key: 32바이트 bytes 여야 한다")
    return project_id, project_key


def compute_key(parts: KeyParts, prov, *, project_id=None, project_key=None) -> str:
    if not isinstance(parts, KeyParts):
        raise ExhaustSpecError("parts: KeyParts 여야 한다")
    rec = _prov.as_record(prov)
    proj, pkey = _check_project(project_id, project_key, rec)
    table_point = parts.table.points[parts.table.index_of(parts.sigma)]
    body = det.hash_parts(
        KEY_TAG, *_prov.merkle_parts(rec), _prov.content_address_kind(rec), proj, parts.world, parts.view,
        struct.pack("<d", table_point), parts.seed, parts.level, parts.k, parts.teacher_id,
        parts.caption_hash, parts.table.key, parts.height, parts.width, parts.precision, parts.quant)
    if rec.ext:
        return blake3.blake3(body, key=pkey).hexdigest()
    return body.hex()


def teacher_id(teacher) -> str:
    fp = getattr(teacher, "fingerprints", None)
    if fp is None:
        raise ExhaustSpecError("teacher.fingerprints 가 없다")
    spec = teacher.spec
    return (f"{spec.repo}@{spec.revision or '-'}|{teacher.dtype}|"
            f"{fp['transformer'].hex()}|{fp['vae'].hex()}")


class Sealer(Protocol):
    def seal(self, plaintext: bytes, *, aad: bytes, project_id: str) -> bytes: ...

    def open(self, blob: bytes, *, aad: bytes, project_id: str) -> bytes: ...


# ----------------------------------------------------------------------------- 항목

@dataclass(frozen=True, eq=False)
class ExhaustEntry:
    key: str
    q: torch.Tensor
    scale: torch.Tensor
    stats: dict
    prov: ProvRecord
    created: float
    file_nbytes: int = 0

    @property
    def x0_hat(self) -> torch.Tensor:
        return dequantize_x0(self.q, self.scale)

    @property
    def nbytes(self) -> int:
        return self.file_nbytes


@dataclass(frozen=True)
class PutResult:
    key: str
    created: bool
    conflict: bool
    nbytes: int


def allowed_purposes(prov) -> tuple:
    rec = _prov.as_record(prov)
    out = ("resume", "delta")
    if rec.bits == 0 and _prov.train_ok(rec):
        out += ("share", "train")
    return out


# ----------------------------------------------------------------------------- 직렬화

def _encode_plain(key: str, rec: ProvRecord, q: torch.Tensor, scale: torch.Tensor, stats: dict,
                  created: float) -> bytes:
    hdr = {"v": 1, "key": key, "prov": rec.to_dict(), "shape": list(q.shape), "stats": stats,
           "created": created, "quant": QUANT_MODE}
    hb = json.dumps(hdr, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body = (_MAGIC + struct.pack("<I", len(hb)) + hb + q.numpy().tobytes()
            + scale.numpy().astype("<f4").tobytes())
    return body + blake3.blake3(body).digest()


def _decode_plain(raw: bytes):
    """평문 BUCX1 파싱·검증. 반환 (hdr, q, scale, rec). 실패 시 _Corrupt."""
    try:
        if len(raw) < 9 + 32 or raw[:5] != _MAGIC:
            raise _Corrupt("매직/길이")
        if blake3.blake3(raw[:-32]).digest() != raw[-32:]:
            raise _Corrupt("체크섬")
        (hl,) = struct.unpack("<I", raw[5:9])
        if 9 + hl > len(raw) - 32:
            raise _Corrupt("헤더 길이")
        hdr = json.loads(raw[9:9 + hl].decode())
        shape = hdr["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or not all(_is_int(s) and s > 0 for s in shape)
                or hdr["quant"] != QUANT_MODE or hdr["v"] != 1):
            raise _Corrupt("헤더")
        c, h, w = shape
        n = c * h * w
        if len(raw) != 9 + hl + n + 4 * c + 32:
            raise _Corrupt("본문 길이")
        off = 9 + hl
        q = torch.from_numpy(np.frombuffer(raw, dtype=np.int8, count=n, offset=off).copy()).reshape(c, h, w)
        sc = torch.from_numpy(np.frombuffer(raw, dtype="<f4", count=c, offset=off + n).astype(np.float32))
        stats = hdr["stats"]
        if not isinstance(stats, dict) or not all(isinstance(v, float) for v in stats.values()):
            raise _Corrupt("stats")
        created = hdr["created"]
        if not isinstance(created, (int, float)) or isinstance(created, bool):
            raise _Corrupt("created")
        rec = ProvRecord.from_dict(hdr["prov"])
        if not isinstance(hdr["key"], str):
            raise _Corrupt("key")
        return hdr, q, sc, rec
    except _Corrupt:
        raise
    except Exception as e:          # noqa: BLE001
        raise _Corrupt(str(e)) from e


def _proj16(project_id: str) -> str:
    return det.hash_parts(_PROJ_TAG, project_id).hex()[:16]


def _check_stats(stats) -> dict:
    if stats is None:
        return {}
    if not isinstance(stats, dict):
        raise ExhaustSpecError("stats: dict 여야 한다")
    out = {}
    for k, v in stats.items():
        if not isinstance(k, str):
            raise ExhaustSpecError(f"stats 키는 str: {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ExhaustSpecError(f"stats 값은 수: {k}={v!r}")
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            raise ExhaustSpecError(f"stats 값이 유한하지 않다: {k}={v!r}")
        out[k] = f
    return out


def _check_purpose(purpose) -> None:
    if not isinstance(purpose, str) or purpose not in PURPOSES:
        raise ExhaustSpecError(f"모르는 purpose: {purpose!r}")


# ----------------------------------------------------------------------------- 장부

class ExhaustLedger:
    def __init__(self, root, *, sealer: Optional[Sealer] = None, clock: Callable[[], float] = time.time,
                 ext_ttl_s: float = EXT_TTL_S):
        self.root = Path(root)
        self._sealer = sealer
        self._clock = clock
        self._ttl = ext_ttl_s
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "writes": 0, "conflicts": 0, "corrupt": 0, "expired": 0,
                       "denied": 0}

    # -- 보조
    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._stats[name] += n

    def key(self, parts: KeyParts, prov, *, project_id=None, project_key=None) -> str:
        return compute_key(parts, prov, project_id=project_id, project_key=project_key)

    def entry_path(self, key: str, prov, *, project_id=None) -> Path:
        rec = _prov.as_record(prov)
        if rec.ext:
            if project_id is None:
                raise FirewallError("ext 경로에는 project_id 가 필요하다")
            if not isinstance(project_id, str) or not project_id:
                raise ExhaustSpecError("project_id: 비어 있지 않은 str 이어야 한다")
            return self.root / rec.ns / _proj16(project_id) / key[:2] / f"{key}.exh"
        return self.root / rec.ns / key[:2] / f"{key}.exh"

    def _need_sealer(self) -> Sealer:
        if self._sealer is None:
            raise FirewallError("ext 항목에는 sealer 가 필요하다")
        return self._sealer

    # -- 읽기
    def _load(self, path: Path, key: str, project_id: Optional[str], ext: bool, *, check_ttl: bool):
        """반환 (entry | None, status) status in {"hit","miss","corrupt","expired"}."""
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None, "miss"
        except OSError:
            return None, "corrupt"
        try:
            if ext:
                if len(raw) < 13 or raw[:5] != _SEAL_MAGIC:
                    raise _Corrupt("봉인 매직")
                (outer,) = struct.unpack("<d", raw[5:13])
                if check_ttl and self._clock() - outer > self._ttl:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    return None, "expired"
                sealer = self._need_sealer()
                try:
                    plain = sealer.open(raw[13:], aad=key.encode() + raw[5:13], project_id=project_id)
                except Exception as e:      # noqa: BLE001
                    raise _Corrupt(f"봉인 열기 실패: {e}") from e
                if not isinstance(plain, (bytes, bytearray)):
                    raise _Corrupt("평문 형식")
                hdr, q, sc, rec = _decode_plain(bytes(plain))
                if hdr["created"] != outer:
                    raise _Corrupt("created 불일치")
            else:
                hdr, q, sc, rec = _decode_plain(raw)
            if hdr["key"] != key:
                raise _Corrupt("key 불일치")
            if rec.ext != ext:
                raise _Corrupt("ext 불일치")
        except _Corrupt:
            return None, "corrupt"
        return ExhaustEntry(key=key, q=q, scale=sc, stats=dict(hdr["stats"]), prov=rec,
                            created=float(hdr["created"]), file_nbytes=len(raw)), "hit"

    def _prepare(self, parts, prov, purpose, project_id, project_key):
        _check_purpose(purpose)
        rec = _prov.as_record(prov)
        if purpose not in allowed_purposes(rec):
            self._bump("denied")
            raise FirewallError(f"purpose {purpose!r} 금지: {sorted(rec.taint)}")
        key = compute_key(parts, rec, project_id=project_id, project_key=project_key)
        if rec.ext:
            self._need_sealer()
        return rec, key

    def _read(self, rec: ProvRecord, key: str, project_id, purpose: str):
        path = self.entry_path(key, rec, project_id=project_id)
        entry, status = self._load(path, key, project_id, rec.ext, check_ttl=True)
        if entry is not None:
            # 저장된 prov 가 같은 키를 공유하지만 더 약한 경우(예: 라이선스 미상) 용도를 다시 검사한다.
            if _prov.merkle_parts(entry.prov) != _prov.merkle_parts(rec):
                return None, "corrupt"
            if purpose not in allowed_purposes(entry.prov):
                self._bump("denied")
                raise FirewallError(f"저장된 prov 가 purpose {purpose!r} 를 허용하지 않는다")
        return entry, status

    def get(self, parts: KeyParts, prov, *, purpose: str, project_id=None, project_key=None):
        rec, key = self._prepare(parts, prov, purpose, project_id, project_key)
        entry, status = self._read(rec, key, project_id, purpose)
        if entry is not None:
            self._bump("hits")
            return entry
        self._bump("misses")
        if status == "corrupt":
            self._bump("corrupt")
        elif status == "expired":
            self._bump("expired")
        return None

    # -- 쓰기
    def _put(self, parts, x0_hat, prov, project_id, project_key, stats):
        rec = _prov.as_record(prov)
        if not isinstance(parts, KeyParts):
            raise ExhaustSpecError("parts: KeyParts 여야 한다")
        key = compute_key(parts, rec, project_id=project_id, project_key=project_key)
        if rec.ext:
            self._need_sealer()
        st = _check_stats(stats)
        if not isinstance(x0_hat, torch.Tensor):
            raise ExhaustSpecError("x0_hat: Tensor 여야 한다")
        if x0_hat.ndim != 3 or tuple(x0_hat.shape[1:]) != (parts.height // 8, parts.width // 8):
            raise ExhaustSpecError(
                f"latent 모양 불일치: {tuple(x0_hat.shape)} vs (*, {parts.height // 8}, {parts.width // 8})")
        q, scale = quantize_x0(x0_hat.detach().cpu())
        path = self.entry_path(key, rec, project_id=project_id)
        created_t = float(self._clock())

        existing, status = self._load(path, key, project_id, rec.ext, check_ttl=False)
        if existing is not None:
            return self._conflict_result(key, existing, q, scale)

        plain = _encode_plain(key, rec, q, scale, st, created_t)
        if rec.ext:
            ct = struct.pack("<d", created_t)
            blob = self._need_sealer().seal(plain, aad=key.encode() + ct, project_id=project_id)
            data = _SEAL_MAGIC + ct + blob
        else:
            data = plain
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{os.urandom(4).hex()}")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            if status == "miss":
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    other, _ = self._load(path, key, project_id, rec.ext, check_ttl=False)
                    if other is not None:
                        return self._conflict_result(key, other, q, scale)
                    os.replace(tmp, path)
                except OSError:
                    os.replace(tmp, path)
            else:
                os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        self._bump("writes")
        return PutResult(key=key, created=True, conflict=False, nbytes=len(data))

    def _conflict_result(self, key, existing: ExhaustEntry, q, scale) -> PutResult:
        same = torch.equal(existing.q, q) and torch.equal(existing.scale, scale)
        if not same:
            self._bump("conflicts")
        return PutResult(key=key, created=False, conflict=not same, nbytes=existing.nbytes)

    def put(self, parts: KeyParts, x0_hat, prov, *, project_id=None, project_key=None, stats=None) -> PutResult:
        return self._put(parts, x0_hat, prov, project_id, project_key, stats)

    def get_or_compute(self, parts: KeyParts, prov, compute, *, purpose: str = "resume", project_id=None,
                       project_key=None) -> ExhaustEntry:
        rec, key = self._prepare(parts, prov, purpose, project_id, project_key)
        entry, status = self._read(rec, key, project_id, purpose)
        if entry is not None:
            self._bump("hits")
            return entry
        self._bump("misses")
        if status == "corrupt":
            self._bump("corrupt")
        elif status == "expired":
            self._bump("expired")
        out = compute()
        stats = None
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ExhaustSpecError("compute: Tensor 또는 (Tensor, stats)")
            out, stats = out
        self._put(parts, out, rec, project_id, project_key, stats)
        stored, _ = self._read(rec, key, project_id, purpose)
        if stored is None:
            raise ExhaustError("저장 직후 읽기 실패")
        return stored

    # -- 열거
    def iter_keys(self, ns: str, project_id=None) -> list:
        if not isinstance(ns, str) or ns not in NAMESPACES:
            raise ExhaustSpecError(f"모르는 ns: {ns!r}")
        if ns in _EXT_ZONES:
            if project_id is None:
                raise FirewallError("ext 구역 열거에는 project_id 가 필요하다")
            if not isinstance(project_id, str) or not project_id:
                raise ExhaustSpecError("project_id: 비어 있지 않은 str 이어야 한다")
            base = self.root / ns / _proj16(project_id)
        else:
            base = self.root / ns
        if not base.is_dir():
            return []
        keys = [p.stem for p in base.glob("??/*.exh") if _KEY_RE.match(p.stem)]
        return sorted(keys)

    def iter_train(self, branch: str = "main"):
        _prov.train_ok(ProvRecord(), branch)         # 모르는 branch -> ValueError (즉시)
        return self._iter_train(branch)

    def _iter_train(self, branch: str):
        base = self.root / "train_ok"
        if not base.is_dir():
            return
        for path in sorted(base.glob("??/*.exh"), key=lambda p: p.stem):
            key = path.stem
            if not _KEY_RE.match(key):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if raw[:5] != _MAGIC:
                continue                              # 봉인 파일 등은 열지 않는다
            try:
                hdr, q, sc, rec = _decode_plain(raw)
            except _Corrupt:
                self._bump("corrupt")
                continue
            if hdr["key"] != key or rec.ns != "train_ok" or rec.bits != 0 or not _prov.train_ok(rec, branch):
                continue
            yield ExhaustEntry(key=key, q=q, scale=sc, stats=dict(hdr["stats"]), prov=rec,
                               created=float(hdr["created"]), file_nbytes=len(raw))

    # -- 수명
    def expire(self) -> int:
        n = 0
        now = self._clock()
        for ns in _EXT_ZONES:
            base = self.root / ns
            if not base.is_dir():
                continue
            for path in base.glob("*/??/*.exh"):
                try:
                    with open(path, "rb") as f:
                        head = f.read(13)
                    if len(head) < 13 or head[:5] != _SEAL_MAGIC:
                        continue
                    (created,) = struct.unpack("<d", head[5:13])
                    if now - created > self._ttl:
                        path.unlink()
                        n += 1
                except OSError:
                    continue
        if n:
            self._bump("expired", n)
        return n

    def purge_project(self, project_id: str) -> int:
        if not isinstance(project_id, str) or not project_id:
            raise ExhaustSpecError("project_id: 비어 있지 않은 str 이어야 한다")
        h = _proj16(project_id)
        n = 0
        for ns in _EXT_ZONES:
            base = self.root / ns / h
            if not base.is_dir():
                continue
            for path in list(base.glob("??/*.exh")):
                try:
                    path.unlink()
                    n += 1
                except OSError:
                    pass
            for d in sorted(base.glob("??"), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
            try:
                base.rmdir()
            except OSError:
                pass
        return n

    def recover(self) -> int:
        n = 0
        if not self.root.is_dir():
            return 0
        for path in self.root.rglob("*.exh.tmp*"):
            if path.is_file():
                try:
                    path.unlink()
                    n += 1
                except OSError:
                    pass
        return n
