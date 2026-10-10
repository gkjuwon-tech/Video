"""DAT-05 독립 오라클: 계약서의 디스크 배치를 구현과 무관하게 다시 만든다(FND-03/04 기능만 사용)."""
import io
import json
import os
import tarfile

import numpy as np

from buc.core import det
from buc.gov import prov as P

SAMPLE_TAG = "buc.shard.sample.v1"
SHARD_TAG = "buc.shard.v1"


def prov_json(prov) -> bytes:
    return json.dumps(P.as_record(prov).to_dict(), sort_keys=True, separators=(",", ":")).encode("ascii")


def sample_digest(key, fields, pj) -> bytes:
    exts = sorted(fields)
    parts = [SAMPLE_TAG, key, bytes(pj), len(exts)]
    for e in exts:
        parts += [e, det.hash_bytes(bytes(fields[e]))]
    return det.hash_parts(*parts)


def digests(samples):
    return [sample_digest(k, f, prov_json(p)) for k, f, p in samples]


def root_of(samples) -> bytes:
    return det.merkle_root(digests(samples))


def tar_from_members(members) -> bytes:
    """members: [(name, bytes)] 를 계약의 USTAR 메타로 그대로 쓴다."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tf.addfile(ti, io.BytesIO(bytes(data)))
    return bio.getvalue()


def members_of(samples, prov_override=None):
    out = []
    for i, (k, f, p) in enumerate(samples):
        allf = {e: bytes(v) for e, v in f.items()}
        allf["prov.json"] = prov_override[i] if prov_override else prov_json(p)
        for e in sorted(allf):
            out.append((f"{k}.{e}", allf[e]))
    return out


def tar_bytes(samples, prov_override=None) -> bytes:
    return tar_from_members(members_of(samples, prov_override))


def write_raw(dirpath, samples, name=None, prov_override=None):
    """writer를 거치지 않고 샤드를 만든다. 반환 (path, root). name이 None이면 루트 hex 이름."""
    root = det.merkle_root([sample_digest(k, f, prov_override[i] if prov_override else prov_json(p))
                            for i, (k, f, p) in enumerate(samples)])
    path = os.path.join(str(dirpath), (name or root.hex()) + ".tar")
    with open(path, "wb") as fh:
        fh.write(tar_bytes(samples, prov_override))
    return path, root


def npy(a) -> bytes:
    bio = io.BytesIO()
    np.save(bio, np.ascontiguousarray(a), allow_pickle=False)
    return bio.getvalue()


def clean_samples(n, seed=0, prefix="s", prov=("ENGINE",)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        key = f"{prefix}{i:04d}"
        arr = rng.standard_normal((3, 4, 4)).astype(np.float32)
        out.append((key, {"npy": npy(arr), "txt": b"PAYLOAD-MARKER-" + key.encode()}, list(prov)))
    return out
