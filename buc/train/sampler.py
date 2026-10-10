"""RUN-06 무상태 샘플러: Feistel PRP(전단사 순열) + 에폭 단위 StatelessSampler.

바이트 계약(ops/harness/contracts/RUN-06.md)이다. 라운드 함수·키 유도·cycle-walking을 바꾸면
이미 저장된 체크포인트의 재개가 깨진다. CUDA 없이 import 가능하며 import 부작용이 없다.
"""
from __future__ import annotations

import numpy as np

from buc.core.det import derive_seed

__all__ = ["FeistelPRP", "StatelessSampler"]

_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MUL1 = 0xBF58476D1CE4E5B9
_MUL2 = 0x94D049BB133111EB
_MAX_N = 1 << 62
_KEY_TAG = "buc.feistel.v1"
_EPOCH_TAG = "buc.train.epoch.v1"


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _check_int(v, name: str, lo: int, hi: int) -> int:
    """lo <= v < hi 인 정수만 허용. 비정수·bool은 TypeError, 범위 밖은 ValueError."""
    if not _is_int(v):
        raise TypeError(f"{name}: integer required, got {type(v).__name__}")
    v = int(v)
    if not (lo <= v < hi):
        raise ValueError(f"{name} out of range [{lo}, {hi}): {v}")
    return v


def _mix(z: int) -> int:
    """splitmix64 마무리(파이썬 정수)."""
    z = (z + _GOLDEN) & _M64
    z = ((z ^ (z >> 30)) * _MUL1) & _M64
    z = ((z ^ (z >> 27)) * _MUL2) & _M64
    return z ^ (z >> 31)


def _mix_np(z: np.ndarray) -> np.ndarray:
    """splitmix64 마무리(uint64 배열, 넘침은 mod 2^64)."""
    u = np.uint64
    z = z + u(_GOLDEN)
    z = (z ^ (z >> u(30))) * u(_MUL1)
    z = (z ^ (z >> u(27))) * u(_MUL2)
    return z ^ (z >> u(31))


class FeistelPRP:
    """[0, n) 위의 키 있는 전단사 순열. 무상태이며 같은 (n, seed, rounds)면 항상 같은 순열."""

    def __init__(self, n, seed, rounds=6):
        self.n = _check_int(n, "n", 1, _MAX_N + 1)
        self.seed = _check_int(seed, "seed", 0, 1 << 64)
        self.rounds = _check_int(rounds, "rounds", 3, 1 << 31)
        self._h = max(1, ((self.n - 1).bit_length() + 1) // 2)
        self._mask = (1 << self._h) - 1
        self._keys = tuple(derive_seed(self.seed, _KEY_TAG, r) for r in range(self.rounds))
        self._keys_np = tuple(np.uint64(k) for k in self._keys)

    def __repr__(self) -> str:
        return f"FeistelPRP(n={self.n}, seed={self.seed}, rounds={self.rounds})"

    # ---- 스칼라 --------------------------------------------------------------------------------
    def _enc(self, x: int) -> int:
        h, mask = self._h, self._mask
        left, right = x >> h, x & mask
        for k in self._keys:
            left, right = right, left ^ (_mix((right ^ k) & _M64) & mask)
        return (left << h) | right

    def _dec(self, y: int) -> int:
        h, mask = self._h, self._mask
        left, right = y >> h, y & mask
        for k in reversed(self._keys):
            left, right = right ^ (_mix((left ^ k) & _M64) & mask), left
        return (left << h) | right

    def __call__(self, i) -> int:
        i = _check_int(i, "i", 0, self.n)
        y = self._enc(i)
        while y >= self.n:  # cycle-walking: 순열의 사이클을 따라 범위 안으로 돌아올 때까지
            y = self._enc(y)
        return y

    def inverse(self, j) -> int:
        j = _check_int(j, "j", 0, self.n)
        y = self._dec(j)
        while y >= self.n:
            y = self._dec(y)
        return y

    # ---- 배열 ----------------------------------------------------------------------------------
    def _enc_np(self, x: np.ndarray) -> np.ndarray:
        h, mask = np.uint64(self._h), np.uint64(self._mask)
        left, right = x >> h, x & mask
        for k in self._keys_np:
            left, right = right, left ^ (_mix_np(right ^ k) & mask)
        return (left << h) | right

    def _dec_np(self, y: np.ndarray) -> np.ndarray:
        h, mask = np.uint64(self._h), np.uint64(self._mask)
        left, right = y >> h, y & mask
        for k in reversed(self._keys_np):
            left, right = right ^ (_mix_np(left ^ k) & mask), left
        return (left << h) | right

    def _apply_array(self, a, step_fn, name: str) -> np.ndarray:
        arr = np.asarray(a)
        if arr.dtype.kind not in "iu":
            raise TypeError(f"{name}: integer array required, got dtype {arr.dtype}")
        shape = arr.shape
        if arr.size == 0:
            return np.empty(shape, dtype=np.int64)
        if int(arr.min()) < 0 or int(arr.max()) >= self.n:
            raise ValueError(f"{name}: value out of range [0, {self.n})")
        n = np.uint64(self.n)
        with np.errstate(over="ignore"):
            y = step_fn(arr.reshape(-1).astype(np.uint64))
            bad = y >= n
            while bad.any():
                y[bad] = step_fn(y[bad])
                bad = y >= n
        return y.astype(np.int64).reshape(shape)

    def permute_array(self, a) -> np.ndarray:
        return self._apply_array(a, self._enc_np, "permute_array")

    def inverse_array(self, a) -> np.ndarray:
        return self._apply_array(a, self._dec_np, "inverse_array")


class StatelessSampler:
    """위치 p = step*B + j 의 인덱스를 (에폭, 오프셋)으로 풀어 에폭별 Feistel 순열로 만든다.

    기록하는 상태가 없다: ``indices(step)`` 은 호출 순서·횟수와 무관하게 같은 값이다.
    (내부 캐시는 같은 입력에 같은 순열을 다시 만드는 비용을 아낄 뿐이다.)
    """

    _CACHE_MAX = 8

    def __init__(self, num_samples, batch_size, seed):
        self.num_samples = _check_int(num_samples, "num_samples", 1, _MAX_N + 1)
        self.batch_size = _check_int(batch_size, "batch_size", 1, 1 << 62)
        self.seed = _check_int(seed, "seed", 0, 1 << 64)
        self._cache: dict[int, FeistelPRP] = {}

    def _prp(self, epoch: int) -> FeistelPRP:
        p = self._cache.get(epoch)
        if p is None:
            p = FeistelPRP(self.num_samples, derive_seed(self.seed, _EPOCH_TAG, epoch))
            if len(self._cache) >= self._CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            self._cache[epoch] = p
        return p

    def index_at(self, pos) -> int:
        pos = _check_int(pos, "pos", 0, 1 << 126)
        epoch, offset = divmod(pos, self.num_samples)
        return self._prp(epoch)(offset)

    def indices(self, step) -> np.ndarray:
        step = _check_int(step, "step", 0, 1 << 100)
        b, n = self.batch_size, self.num_samples
        base = step * b
        out = np.empty(b, dtype=np.int64)
        j = 0
        while j < b:
            epoch, offset = divmod(base + j, n)
            take = min(b - j, n - offset)
            out[j:j + take] = self._prp(epoch).permute_array(np.arange(offset, offset + take, dtype=np.int64))
            j += take
        return out
