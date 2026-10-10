"""DAT-05 샤드 I/O v1: WebDataset 호환 tar · Merkle 내용 주소 · prov(계보) 방화벽.

학습·개발 데이터가 디스크에 쌓이는 유일한 길. 쓰기와 읽기 양쪽에서 방화벽(FND-04)을
통과하지 못한 샘플은 거부된다(닫힌 쪽 실패). `webdataset` 패키지에 의존하지 않는다.
CUDA 없이 import 가능하며 import 부작용이 없다.
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from buc.core import det
from buc.gov import prov as P
from buc.gov.prov import FirewallError

__all__ = [
    "SHARD_TAG", "SAMPLE_TAG", "ShardError", "ShardIntegrityError", "FirewallError", "Sample", "ShardInfo",
    "ShardWriter", "verify_shard", "iter_shard", "iter_shards", "dataset_root", "find_shards_with_root",
    "npy_encode", "npy_decode",
]

SHARD_TAG = "buc.shard.v1"
SAMPLE_TAG = "buc.shard.sample.v1"

_PROV_EXT = "prov.json"
_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_EXT_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_EXT_MAX = 32
_ADDR_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_RE = re.compile(r"([0-9a-f]{64})\.json")
_PURPOSES = ("train", "dev")
_BRANCHES = ("main", "optin-v1")
_CHUNK = 1 << 20
_TAR_ERRORS = (tarfile.TarError, EOFError, UnicodeError)


class ShardError(ValueError):
    """샤드 형식 위반."""


class ShardIntegrityError(ShardError):
    """주소·해시 불일치."""


@dataclass(frozen=True)
class Sample:
    key: str
    fields: dict
    prov: P.ProvRecord


@dataclass(frozen=True)
class ShardInfo:
    root: bytes
    n: int
    keys: tuple
    nbytes: int
    tar_hash: bytes
    roots: tuple
    ns: str
    prov_bits: int
    path: str

    @property
    def address(self) -> str:
        return self.root.hex()

    def to_dict(self) -> dict:
        return {"v": SHARD_TAG, "root": self.root.hex(), "n": self.n, "keys": list(self.keys),
                "nbytes": self.nbytes, "tar_blake3": self.tar_hash.hex(), "roots": list(self.roots),
                "ns": self.ns, "prov_bits": self.prov_bits}


# --------------------------------------------------------------------------- 공통 보조

def _canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("ascii")


def _prov_json(rec: P.ProvRecord) -> bytes:
    return _canon_json(rec.to_dict())


def _sample_digest(key: str, fields: Mapping, pj: bytes) -> bytes:
    exts = sorted(fields)
    parts = [SAMPLE_TAG, key, bytes(pj), len(exts)]
    for e in exts:
        parts += [e, det.hash_bytes(fields[e])]
    return det.hash_parts(*parts)


def _check_purpose(purpose, branch) -> None:
    if purpose not in _PURPOSES:
        raise ValueError(f"purpose 는 {_PURPOSES} 중 하나: {purpose!r}")
    if branch not in _BRANCHES:
        raise ValueError(f"branch 는 {_BRANCHES} 중 하나: {branch!r}")
    if purpose == "dev" and branch != "main":
        raise ValueError("dev 목적은 branch='main' 만 허용한다")


def _check_firewall(key: str, rec: P.ProvRecord, purpose: str, branch: str) -> None:
    """샤드 방화벽: FND-04 판정 + 샤드 불변식(ext|comp|hold 비트 0, ns == train_ok)."""
    if purpose == "train":
        why = list(P.why_not_train(rec, branch))
    else:
        why = list(P.why_not_dev(rec))
    if rec.ext:
        why.append("ext-bit")
    if rec.comp:
        why.append("comp-bit")
    if rec.hold:
        why.append("hold-bit")
    if rec.ns != "train_ok":
        why.append(f"ns={rec.ns}")
    if why:
        raise FirewallError(f"샤드 방화벽 거부 (key={key!r}, purpose={purpose}, branch={branch}): "
                            + ", ".join(why))


def _file_digest(path: str):
    import blake3
    h = blake3.blake3()
    size = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(_CHUNK)
            if not b:
                break
            size += len(b)
            h.update(b)
    return size, h.digest()


def _roots_union(recs) -> tuple:
    out: set = set()
    for r in recs:
        out.update(r.roots)
    return tuple(sorted(out))


# --------------------------------------------------------------------------- 쓰기

class _Open:
    """열린 샤드 하나(임시 파일에 스트리밍)."""

    def __init__(self, out_dir: str, purpose: str):
        os.makedirs(out_dir, exist_ok=True)
        fd, self.tmp = tempfile.mkstemp(dir=out_dir, prefix=".shard-", suffix=".part")
        self.fh = os.fdopen(fd, "wb")
        self.tf = tarfile.open(fileobj=self.fh, mode="w", format=tarfile.USTAR_FORMAT)
        self.digests: list = []
        self.keys: list = []
        self.recs: list = []
        self.payload = 0

    def add(self, key: str, fields: dict, pj: bytes, rec: P.ProvRecord) -> None:
        allf = dict(fields)
        allf[_PROV_EXT] = pj
        for e in sorted(allf):
            data = allf[e]
            ti = tarfile.TarInfo(f"{key}.{e}")
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            self.tf.addfile(ti, io.BytesIO(data))
        self.digests.append(_sample_digest(key, fields, pj))
        self.keys.append(key)
        self.recs.append(rec)
        self.payload += sum(len(v) for v in fields.values())

    def abort(self) -> None:
        try:
            self.tf.close()
        except Exception:
            pass
        try:
            self.fh.close()
        except Exception:
            pass
        _unlink(self.tmp)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class ShardWriter:
    def __init__(self, out_dir, *, max_samples: int = 1000, max_bytes=None, purpose: str = "train",
                 branch: str = "main"):
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples 는 int")
        if max_samples < 1:
            raise ValueError("max_samples 는 1 이상")
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
                raise TypeError("max_bytes 는 int 또는 None")
            if max_bytes < 1:
                raise ValueError("max_bytes 는 1 이상")
        _check_purpose(purpose, branch)
        self._dir = os.fspath(out_dir)
        self._max_samples = max_samples
        self._max_bytes = max_bytes
        self._purpose = purpose
        self._branch = branch
        self._infos: list = []
        self._keys: set = set()
        self._open = None
        self._closed = False

    @property
    def infos(self) -> list:
        return self._infos

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self._abort()
        return False

    def _abort(self) -> None:
        self._closed = True
        if self._open is not None:
            self._open.abort()
            self._open = None

    def write(self, key, fields, prov) -> None:
        if self._closed:
            raise RuntimeError("닫힌 ShardWriter 에는 쓸 수 없다")
        # --- 입력 검증(상태 변경 없음)
        if not isinstance(key, str):
            raise TypeError(f"key 는 str: {type(key).__name__}")
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"key 형식 위반: {key!r}")
        if not isinstance(fields, Mapping):
            raise TypeError(f"fields 는 Mapping: {type(fields).__name__}")
        if len(fields) == 0:
            raise ValueError("fields 가 비었다")
        clean: dict = {}
        for e, v in fields.items():
            if not isinstance(e, str):
                raise TypeError(f"ext 는 str: {type(e).__name__}")
            if e == _PROV_EXT:
                raise ValueError("ext 'prov.json' 은 예약어")
            if len(e) > _EXT_MAX or not _EXT_RE.fullmatch(e):
                raise ValueError(f"ext 형식 위반: {e!r}")
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise TypeError(f"페이로드는 bytes-like: {e!r} -> {type(v).__name__}")
            clean[e] = bytes(v)
        rec = P.as_record(prov)
        if key in self._keys:
            raise ValueError(f"중복 key: {key!r}")
        _check_firewall(key, rec, self._purpose, self._branch)
        pj = _prov_json(rec)
        # --- 여기부터 상태 변경
        try:
            if self._open is None:
                self._open = _Open(self._dir, self._purpose)
            self._open.add(key, clean, pj, rec)
        except BaseException:
            self._abort()
            raise
        self._keys.add(key)
        op = self._open
        if len(op.keys) >= self._max_samples or (self._max_bytes is not None and op.payload >= self._max_bytes):
            self._finish_open()

    def close(self) -> list:
        if not self._closed:
            self._closed = True
            if self._open is not None:
                self._finish_open()
        return self._infos

    def _finish_open(self) -> None:
        op, self._open = self._open, None
        try:
            op.tf.close()
            op.fh.flush()
            op.fh.close()
            root = det.merkle_root(op.digests)
            addr = root.hex()
            nbytes, tar_hash = _file_digest(op.tmp)
            final = os.path.join(self._dir, addr + ".tar")
            if os.path.exists(final):
                if _file_digest(final) != (nbytes, tar_hash):
                    raise ShardIntegrityError(f"같은 주소에 다른 바이트가 이미 있다: {final}")
                _unlink(op.tmp)
            else:
                os.replace(op.tmp, final)
            info = ShardInfo(root=root, n=len(op.keys), keys=tuple(op.keys), nbytes=nbytes, tar_hash=tar_hash,
                             roots=_roots_union(op.recs), ns="train_ok", prov_bits=0, path=final)
            self._write_manifest(info)
        except BaseException:
            op.abort()
            raise
        self._infos.append(info)

    def _write_manifest(self, info: ShardInfo) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=".shard-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(_canon_json(info.to_dict()))
            os.replace(tmp, os.path.join(self._dir, info.address + ".json"))
        except BaseException:
            _unlink(tmp)
            raise


# --------------------------------------------------------------------------- 읽기

def _parse_prov(key: str, pj: bytes) -> P.ProvRecord:
    try:
        d = json.loads(pj)
    except (ValueError, UnicodeError) as e:
        raise ShardError(f"prov 사이드카가 JSON 이 아니다 (key={key!r}): {e}") from None
    if type(d) is not dict:
        raise ShardError(f"prov 사이드카가 객체가 아니다 (key={key!r})")
    try:
        rec = P.ProvRecord.from_dict(d)
    except (ValueError, TypeError) as e:
        raise ShardError(f"prov 사이드카 위반 (key={key!r}): {e}") from None
    if _prov_json(rec) != bytes(pj):
        raise ShardError(f"prov 사이드카가 정규형이 아니다 (key={key!r})")
    return rec


def _scan(path: str) -> Iterator:
    """tar 를 순서대로 읽어 (key, fields, prov_json, rec) 를 샘플마다 내보낸다. 형식 위반은 ShardError."""
    try:
        tf = tarfile.open(path, mode="r:")
    except _TAR_ERRORS as e:
        raise ShardError(f"tar 를 열 수 없다: {path}: {e}") from None
    with tf:
        seen: set = set()
        cur_key = None
        fields: dict = {}
        n_samples = 0

        def finish():
            if cur_key is None:
                return None
            if _PROV_EXT not in fields:
                raise ShardError(f"prov 사이드카 없음 (key={cur_key!r})")
            pj = fields.pop(_PROV_EXT)
            if not fields:
                raise ShardError(f"필드 없는 샘플 (key={cur_key!r})")
            return (cur_key, fields, pj, _parse_prov(cur_key, pj))

        while True:
            try:
                ti = tf.next()
            except _TAR_ERRORS as e:
                raise ShardError(f"tar 손상: {path}: {e}") from None
            if ti is None:
                break
            if not ti.isreg():
                raise ShardError(f"일반 파일이 아닌 멤버: {ti.name!r}")
            key, dot, ext = ti.name.partition(".")
            if (not dot or not _KEY_RE.fullmatch(key) or len(ext) > _EXT_MAX
                    or not _EXT_RE.fullmatch(ext)):
                raise ShardError(f"멤버 이름 위반: {ti.name!r}")
            try:
                data = tf.extractfile(ti).read()
            except _TAR_ERRORS as e:
                raise ShardError(f"tar 잘림: {ti.name!r}: {e}") from None
            if len(data) != ti.size:
                raise ShardError(f"멤버가 잘렸다: {ti.name!r}")
            if key != cur_key:
                out = finish()
                if out is not None:
                    n_samples += 1
                    yield out
                if key in seen:
                    raise ShardError(f"불연속 key: {key!r}")
                seen.add(key)
                cur_key = key
                fields = {}
            if ext in fields:
                raise ShardError(f"중복 ext: {ti.name!r}")
            fields[ext] = data
        out = finish()
        if out is not None:
            n_samples += 1
            yield out
        if n_samples == 0:
            raise ShardError(f"샘플이 없다: {path}")


def _expected_address(path: str, expected_root):
    if expected_root is not None:
        if isinstance(expected_root, (bytes, bytearray, memoryview)):
            return bytes(expected_root)
        if isinstance(expected_root, str) and _ADDR_RE.fullmatch(expected_root):
            return bytes.fromhex(expected_root)
        raise TypeError("expected_root 는 32바이트 root")
    stem = os.path.basename(path)
    if stem.endswith(".tar"):
        stem = stem[:-4]
    if _ADDR_RE.fullmatch(stem):
        return bytes.fromhex(stem)
    return None


def verify_shard(path, *, expected_root=None, purpose: str = "train", branch: str = "main") -> ShardInfo:
    _check_purpose(purpose, branch)
    path = os.fspath(path)
    digests: list = []
    keys: list = []
    recs: list = []
    for key, fields, pj, rec in _scan(path):
        digests.append(_sample_digest(key, fields, pj))
        keys.append(key)
        recs.append(rec)
    root = det.merkle_root(digests)
    want = _expected_address(path, expected_root)
    if want is not None and want != root:
        raise ShardIntegrityError(f"주소 불일치: 기대 {want.hex()} 계산 {root.hex()} ({path})")
    nbytes, tar_hash = _file_digest(path)
    info = ShardInfo(root=root, n=len(keys), keys=tuple(keys), nbytes=nbytes, tar_hash=tar_hash,
                     roots=_roots_union(recs), ns="train_ok", prov_bits=0, path=path)
    man_path = (path[:-4] if path.endswith(".tar") else path) + ".json"
    if os.path.isfile(man_path):
        try:
            with open(man_path, "rb") as fh:
                man = json.loads(fh.read())
        except (ValueError, UnicodeError) as e:
            raise ShardIntegrityError(f"매니페스트를 읽을 수 없다: {man_path}: {e}") from None
        if man != info.to_dict():
            raise ShardIntegrityError(f"매니페스트와 불일치: {man_path}")
    for key, rec in zip(keys, recs):
        _check_firewall(key, rec, purpose, branch)
    return info


def iter_shard(path, *, purpose: str = "train", branch: str = "main", verify: bool = True) -> Iterator[Sample]:
    _check_purpose(purpose, branch)
    path = os.fspath(path)
    if verify:
        verify_shard(path, purpose=purpose, branch=branch)
    for key, fields, _pj, rec in _scan(path):
        _check_firewall(key, rec, purpose, branch)
        yield Sample(key=key, fields=fields, prov=rec)


def iter_shards(paths, **kw) -> Iterator[Sample]:
    for p in paths:
        yield from iter_shard(p, **kw)


def dataset_root(roots: Iterable) -> bytes:
    pairs = []
    for r in roots:
        if not isinstance(r, (bytes, bytearray, memoryview)):
            raise TypeError("root 는 bytes-like")
        b = bytes(r)
        pairs.append((b.hex(), b))
    return det.merkle_root_keyed(pairs)


def find_shards_with_root(directory, root: str) -> list:
    d = os.fspath(directory)
    out = []
    for name in sorted(os.listdir(d)):
        m = _MANIFEST_RE.fullmatch(name)
        if not m:
            continue
        try:
            with open(os.path.join(d, name), "rb") as fh:
                man = json.loads(fh.read())
        except (ValueError, UnicodeError) as e:
            raise ShardError(f"매니페스트 손상: {name}: {e}") from None
        roots = man.get("roots") if isinstance(man, dict) else None
        if not isinstance(roots, list):
            raise ShardError(f"매니페스트 형식 위반: {name}")
        if root in roots:
            out.append(os.path.join(d, m.group(1) + ".tar"))
    return sorted(out)


# --------------------------------------------------------------------------- npy

def npy_encode(a) -> bytes:
    arr = np.asarray(a)
    if arr.dtype.hasobject:
        raise ValueError("object dtype 은 인코딩할 수 없다(pickle 금지)")
    if not arr.flags.c_contiguous:
        arr = arr.copy(order="C")
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()


def npy_decode(b) -> np.ndarray:
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TypeError("npy_decode: bytes-like required")
    try:
        out = np.load(io.BytesIO(bytes(b)), allow_pickle=False)
    except (ValueError, OSError, EOFError) as e:
        raise ValueError(f"npy 디코드 실패: {e}") from None
    if not isinstance(out, np.ndarray):
        raise ValueError("npy 가 아니다")
    return out
