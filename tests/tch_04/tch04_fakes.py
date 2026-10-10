"""TCH-04 테스트 보조: 가짜 sealer, 기본 KeyParts, 결정론 latent. (비밀 아님: 테스트 전용 고정 바이트)"""
import struct

import blake3
import torch

from buc.teacher import exhaust as X
from buc.teacher.sample import DEFAULT_TABLE

PK = bytes(range(32))            # 테스트용 프로젝트 키(실제 비밀 아님)
PK2 = bytes(range(1, 33))
PROJ = "proj-alpha"
PROJ2 = "proj-beta"
WORLD = bytes(range(100, 132))
CAP = "ab" * 32
C, H8, W8 = 16, 8, 8             # latent [16,8,8] <-> 이미지 64x64


def parts(**over) -> "X.KeyParts":
    base = dict(world=WORLD, view="L2/b7/band3", sigma=DEFAULT_TABLE.points[3], seed=1234,
                teacher_id="zimage@-|torch.bfloat16|aa|bb", caption_hash=CAP, table=DEFAULT_TABLE,
                height=H8 * 8, width=W8 * 8)
    base.update(over)
    return X.KeyParts(**base)


def latent(seed=0, c=C, h=H8, w=W8, scale=1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c, h, w, generator=g) * scale


class FakeSealer:
    """blake3 keyed XOF 스트림 XOR + MAC. 호출을 기록한다. 실제 암호가 아니라 테스트 대역."""

    def __init__(self, secret=b"S" * 32, fail_seal=False):
        self.secret, self.fail_seal = secret, fail_seal
        self.seal_calls, self.open_calls = [], []

    def _ks(self, nonce, n):
        return blake3.blake3(b"ks" + nonce, key=self.secret).digest(length=n)

    def seal(self, plaintext, *, aad, project_id):
        if self.fail_seal:
            raise RuntimeError("seal boom")
        self.seal_calls.append((aad, project_id))
        nonce = blake3.blake3(aad + plaintext, key=self.secret).digest(length=12)
        ct = bytes(a ^ b for a, b in zip(plaintext, self._ks(nonce, len(plaintext))))
        mac = blake3.blake3(b"mac" + aad + project_id.encode() + nonce + ct, key=self.secret).digest(length=16)
        return nonce + ct + mac

    def open(self, blob, *, aad, project_id):
        self.open_calls.append((aad, project_id))
        nonce, ct, mac = blob[:12], blob[12:-16], blob[-16:]
        exp = blake3.blake3(b"mac" + aad + project_id.encode() + nonce + ct, key=self.secret).digest(length=16)
        if exp != mac:
            raise ValueError("bad mac")
        return bytes(a ^ b for a, b in zip(ct, self._ks(nonce, len(ct))))


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


ENGINE = {"ENGINE"}
USER = {"USER"}
USER_D = {"USER-DERIVED"}
EXTGEN = {"EXTGEN:test"}
COMP = {"COMP:vendor"}
HOLD = {"HOLDOUT"}


def ext_kw(project=PROJ, key=PK):
    return dict(project_id=project, project_key=key)


def files_under(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def pack_d(x):
    return struct.pack("<d", x)
