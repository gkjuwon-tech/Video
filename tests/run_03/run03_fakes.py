"""RUN-03 테스트용 인메모리 가짜 허브. 네트워크·huggingface_hub 없이 HubClient 이음매를 흉내 낸다."""
import hashlib
import json
from pathlib import Path

import numpy as np

from buc.rt import hf_sync as H

TOKEN = "hf_" + "Zq7Kx3Lm9Pw2Rt5Vb8Nc4Dy6Gh1Jf0Sa"  # 테스트 전용 가짜 값
SIDE = ".sha256.json"


def blob(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def sidecar_bytes(data: bytes) -> bytes:
    """계약의 사이드카 바이트 배치(독립 오라클)."""
    return json.dumps({"v": 1, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)},
                      sort_keys=True, separators=(",", ":")).encode()


def put_complete(hub, remote: str, data: bytes) -> None:
    hub.files[remote] = data
    hub.files[remote + SIDE] = sidecar_bytes(data)


class FakeHub:
    """HubClient 프로토콜 구현: list_files / upload_file / download_file / is_private."""

    def __init__(self, private: bool = True):
        self.private = private
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []
        self._faults: list[list] = []
        self.mutator = None  # (remote, data, nth_download_of_remote) -> bytes

    # --- 시험 보조
    def add_fault(self, op: str, match: str, exc: Exception, times=1) -> None:
        self._faults.append([op, match, exc, times])

    def count(self, op: str, match: str | None = None, *, data_only: bool = False) -> int:
        n = 0
        for c in self.calls:
            if c[0] != op:
                continue
            if match is not None and match not in (c[1] or ""):
                continue
            if data_only and (c[1] or "").endswith(SIDE):
                continue
            n += 1
        return n

    def _check(self, op: str, remote: str) -> None:
        for f in self._faults:
            if f[0] == op and f[1] in remote and (f[3] is None or f[3] > 0):
                if f[3] is not None:
                    f[3] -= 1
                raise f[2]

    # --- HubClient
    def list_files(self, prefix):
        self.calls.append(("list", prefix))
        self._check("list", prefix)
        return sorted(self.files)  # 일부러 접두어로 거르지 않는다: HubSync가 직접 걸러야 한다

    def upload_file(self, local, remote):
        self.calls.append(("upload", remote))
        self._check("upload", remote)
        self.files[remote] = Path(local).read_bytes()

    def download_file(self, remote, local):
        n = sum(1 for c in self.calls if c[0] == "download" and c[1] == remote) + 1
        self.calls.append(("download", remote, str(local)))
        self._check("download", remote)
        if remote not in self.files:
            raise H.HubNotFoundError(f"missing: {remote}")
        data = self.files[remote]
        if self.mutator is not None and not remote.endswith(SIDE):  # 손상은 데이터에만 건다
            data = self.mutator(remote, data, n)
        Path(local).write_bytes(data)

    def is_private(self):
        self.calls.append(("is_private", None))
        self._check("is_private", "")
        return self.private
