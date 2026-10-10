"""TCH-03 Feature-Tap Bus: 교사 층 8/15/23 훅, 고정 PCA-128 int8 사영, 링버퍼, ext 바이패스.

계약서: ops/harness/contracts/TCH-03.md. 공개 API는 계약서와 1:1이다.
- 교사(Z-Image-Turbo) forward 한 번에서 탭 층의 이미지 토큰 은닉(D=3840)을 고정 PCA-k(=128) int8로 즉시 사영한다.
- 학습 탭만 링에 쌓는다. ext(USER 계열·EXTGEN 계열·BIO) 호출은 링에 절대 들어가지 않는다(라이브 임시 탭만).
- 링 진입 규칙 = FND-04 train_ok(prov) AND NOT ext_bit(prov). 플래그가 아니라 prov로 판정한다.
- import는 CUDA·diffusers·transformers 없이 되고 부작용이 없다. torch만 쓴다.
"""
from __future__ import annotations

import dataclasses
import numbers
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from buc.core import det
from buc.gov.prov import FirewallError, as_record, assert_train_ok, ext_bit, merkle_parts, train_ok

__all__ = [
    "TAP_LAYERS", "PCA_DIM", "INT8_MAX", "BASIS_TAG", "TAP_TAG",
    "TapError", "TapSpecError", "TapBasisError", "TapNumericError", "FirewallError",
    "PCABasis", "fit_pca_basis", "TapSet", "TapRing", "TapBus",
]

TAP_LAYERS = (8, 15, 23)
PCA_DIM = 128
INT8_MAX = 127
BASIS_TAG = "buc.tch03.basis.v1"
TAP_TAG = "buc.tch03.tap.v1"

_FIT_CHUNK = 8192


# --------------------------------------------------------------------------- 예외

class TapError(RuntimeError):
    """Tap Bus 오류의 뿌리."""


class TapSpecError(TapError, ValueError):
    """인자·모양·상태가 사양에 맞지 않는다."""


class TapBasisError(TapError):
    """PCA 기저가 변조·불일치·손상됐다."""


class TapNumericError(TapError, FloatingPointError):
    """비유한(NaN/Inf) 값."""


# --------------------------------------------------------------------------- 보조

def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _as_float32(t, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        try:
            t = torch.as_tensor(t)
        except Exception as e:  # noqa: BLE001
            raise TapSpecError(f"{name}: 텐서로 바꿀 수 없다") from e
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _check_layers(layers) -> tuple:
    try:
        out = tuple(layers)
    except TypeError as e:
        raise TapSpecError("layers: 정수 시퀀스여야 한다") from e
    if not out:
        raise TapSpecError("layers: 비어 있다")
    if any(not _is_int(l) or int(l) < 0 for l in out):
        raise TapSpecError("layers: 0 이상 정수여야 한다")
    out = tuple(int(l) for l in out)
    if len(set(out)) != len(out):
        raise TapSpecError("layers: 중복")
    return out


# --------------------------------------------------------------------------- PCA 기저

class PCABasis:
    """층별 고정 PCA 기저. float32로 복사·동결한다. 지문은 mean/comps/scale/층/차원으로만 정해진다(meta 제외)."""

    def __init__(self, layers, mean, comps, scale, meta: Optional[Mapping] = None):
        self.layers = _check_layers(layers)
        L = len(self.layers)
        self.mean = _as_float32(mean, "mean")
        self.comps = _as_float32(comps, "comps")
        self.scale = _as_float32(scale, "scale")
        for t in (self.mean, self.comps, self.scale):
            t.requires_grad_(False)
        if self.mean.ndim != 2 or self.comps.ndim != 3 or self.scale.ndim != 2:
            raise TapSpecError("mean[L,D], comps[L,D,k], scale[L,k] 모양이어야 한다")
        if self.mean.shape[0] != L or self.comps.shape[0] != L or self.scale.shape[0] != L:
            raise TapSpecError("층 개수와 텐서의 첫 축이 다르다")
        D, k = self.comps.shape[1], self.comps.shape[2]
        if self.mean.shape[1] != D or self.scale.shape[1] != k or k < 1 or D < 1:
            raise TapSpecError("mean/comps/scale 모양 불일치")
        for name, t in (("mean", self.mean), ("comps", self.comps), ("scale", self.scale)):
            if not bool(torch.isfinite(t).all()):
                raise TapNumericError(f"{name}: 비유한 값")
        if not bool((self.scale > 0).all()):
            raise TapSpecError("scale > 0 이어야 한다")
        if meta is not None and not isinstance(meta, Mapping):
            raise TapSpecError("meta: Mapping 이어야 한다")
        self.meta = dict(meta) if meta is not None else {}
        self.dim = int(D)
        self.k = int(k)
        self.fingerprint = self._compute_fp()
        self._dev_cache: dict = {}

    # -- 지문
    def _compute_fp(self) -> bytes:
        layers_arr = torch.tensor(self.layers, dtype=torch.int64)
        return det.hash_parts(BASIS_TAG, self.dim, self.k, det.hash_array(layers_arr),
                              det.hash_array(self.mean), det.hash_array(self.comps), det.hash_array(self.scale))

    def verify(self) -> None:
        if self._compute_fp() != self.fingerprint:
            raise TapBasisError("PCA 기저가 변조됐다(지문 불일치)")

    # -- 사영
    def _index(self, layer) -> int:
        if not _is_int(layer) or int(layer) not in self.layers:
            raise TapSpecError(f"기저에 없는 층: {layer!r}")
        return self.layers.index(int(layer))

    def _tensors(self, i: int, device: torch.device):
        if device.type == "cpu":
            return self.mean[i], self.comps[i], self.scale[i]
        key = (i, str(device))
        hit = self._dev_cache.get(key)
        if hit is None:
            hit = (self.mean[i].to(device), self.comps[i].to(device), self.scale[i].to(device))
            self._dev_cache[key] = hit
        return hit

    def project(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """x[..., D] -> int8[..., k] = clamp(round((x.float()-mean)@comps / scale), -127, 127)."""
        i = self._index(layer)
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != self.dim:
            raise TapSpecError(f"x[..., {self.dim}] 텐서여야 한다")
        with torch.no_grad():
            xf = x.detach().float()
            if not bool(torch.isfinite(xf).all()):
                raise TapNumericError("입력에 비유한 값")
            mean, comps, scale = self._tensors(i, xf.device)
            z = ((xf - mean) @ comps) / scale
            if not bool(torch.isfinite(z).all()):
                raise TapNumericError("사영 결과에 비유한 값")
            return torch.round(z).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)

    def dequantize(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        i = self._index(layer)
        if not isinstance(q, torch.Tensor) or q.dtype != torch.int8:
            raise TapSpecError("q: int8 텐서여야 한다")
        if q.ndim < 1 or q.shape[-1] != self.k:
            raise TapSpecError(f"q[..., {self.k}] 이어야 한다")
        _, _, scale = self._tensors(i, q.device)
        return q.float() * scale

    def reconstruct(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        i = self._index(layer)
        deq = self.dequantize(layer, q)
        mean, comps, _ = self._tensors(i, deq.device)
        return deq @ comps.T + mean

    # -- 저장·적재
    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": BASIS_TAG, "layers": list(self.layers),
            "mean": self.mean.clone(), "comps": self.comps.clone(), "scale": self.scale.clone(),
            "meta": dict(self.meta), "fingerprint": self.fingerprint.hex(),
        }
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @classmethod
    def load(cls, path) -> "PCABasis":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise TapBasisError(f"기저 파일을 읽을 수 없다: {p.name}") from e
        if not isinstance(d, dict) or d.get("tag") != BASIS_TAG:
            raise TapBasisError("기저 파일 태그 불일치")
        try:
            basis = cls(d["layers"], d["mean"], d["comps"], d["scale"], d.get("meta") or {})
            stored = bytes.fromhex(d["fingerprint"])
        except (KeyError, TypeError, ValueError, TapError) as e:
            raise TapBasisError("기저 파일 내용이 올바르지 않다") from e
        if basis.fingerprint != stored:
            raise TapBasisError("저장된 지문과 텐서가 일치하지 않는다")
        return basis


def fit_pca_basis(samples: Mapping, *, prov, k: int = PCA_DIM, meta: Optional[Mapping] = None) -> PCABasis:
    """층별 표본 {layer: Tensor[N,D]}에서 PCA-k 기저를 한 번 적합한다(이후 고정).

    prov 키워드 필수. 방화벽(assert_train_ok) 위반이면 FirewallError.
    """
    if prov is None:
        raise TypeError("fit_pca_basis: prov 필수")
    rec = as_record(prov)
    assert_train_ok(rec, "fit_pca_basis")
    if not _is_int(k) or int(k) < 1:
        raise TapSpecError("k >= 1 정수여야 한다")
    k = int(k)
    if not isinstance(samples, Mapping) or not samples:
        raise TapSpecError("samples: 비어 있지 않은 {층: Tensor[N,D]} 이어야 한다")
    layers = _check_layers(sorted(samples))
    D = None
    for l in layers:
        x = samples[l]
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise TapSpecError(f"층 {l}: 2차원 텐서여야 한다")
        if D is None:
            D = int(x.shape[1])
        elif int(x.shape[1]) != D:
            raise TapSpecError("층 간 D가 다르다")
        if x.shape[0] < k + 1:
            raise TapSpecError(f"층 {l}: N >= k+1 필요 (N={x.shape[0]}, k={k})")
    if k > D:
        raise TapSpecError(f"k({k}) <= D({D}) 이어야 한다")

    means, comps, scales, fit_n = [], [], [], {}
    for l in layers:
        x = samples[l].detach().to("cpu")
        if not bool(torch.isfinite(x).all()):
            raise TapNumericError(f"층 {l}: 비유한 표본")
        n = int(x.shape[0])
        mu = torch.zeros(D, dtype=torch.float64)
        for s in range(0, n, _FIT_CHUNK):
            mu += x[s:s + _FIT_CHUNK].double().sum(0)
        mu /= n
        cov = torch.zeros(D, D, dtype=torch.float64)
        for s in range(0, n, _FIT_CHUNK):
            xc = x[s:s + _FIT_CHUNK].double() - mu
            cov += xc.T @ xc
        cov /= (n - 1)
        cov = (cov + cov.T) * 0.5
        w, V = torch.linalg.eigh(cov)
        V = torch.flip(V, dims=[1])[:, :k].contiguous()
        # 부호 규약: 성분마다 절댓값 최대 원소가 양수
        idx = V.abs().argmax(dim=0)
        sign = torch.where(V[idx, torch.arange(k)] < 0, -1.0, 1.0).to(torch.float64)
        V = V * sign
        V32 = V.float()
        mu32 = mu.float()
        zmax = torch.zeros(k, dtype=torch.float64)
        for s in range(0, n, _FIT_CHUNK):
            z = (x[s:s + _FIT_CHUNK].double() - mu32.double()) @ V32.double()
            zmax = torch.maximum(zmax, z.abs().amax(dim=0))
        sc = zmax / INT8_MAX
        sc = torch.where(sc > 0, sc, torch.ones_like(sc))
        means.append(mu32)
        comps.append(V32)
        scales.append(sc.float())
        fit_n[l] = n
    m = dict(meta) if meta is not None else {}
    m["fit_n"] = fit_n
    return PCABasis(layers, torch.stack(means), torch.stack(comps), torch.stack(scales), meta=m)


# --------------------------------------------------------------------------- TapSet

@dataclass(frozen=True, eq=False)
class TapSet:
    """캡처 한 번의 탭. layers[l]은 int8 [B,k,gh,gw]."""
    layers: dict
    grid: tuple
    basis_fp: bytes
    prov: Any
    ephemeral: bool
    key: Optional[str] = None

    @property
    def nbytes(self) -> int:
        return sum(int(q.numel()) for q in self.layers.values())

    def features(self, basis: PCABasis) -> dict:
        if basis.fingerprint != self.basis_fp:
            raise TapBasisError("기저 지문 불일치")
        out = {}
        for l in sorted(self.layers):
            i = basis._index(l)
            q = self.layers[l]
            out[l] = q.float() * basis.scale[i].to(q.device).view(1, -1, 1, 1)
        return out

    def digest(self) -> bytes:
        gh, gw = self.grid
        parts = [TAP_TAG, self.basis_fp, *merkle_parts(self.prov), int(gh), int(gw), self.key or ""]
        for l in sorted(self.layers):
            parts += [int(l), det.hash_array(self.layers[l])]
        return det.hash_parts(*parts)


# --------------------------------------------------------------------------- 링

class TapRing:
    """FIFO 링버퍼. 학습 탭만 받는다(train_ok AND NOT ext). 보관·반환 모두 복사본."""

    def __init__(self, basis_or_fp, *, max_entries: int, max_bytes: Optional[int] = None):
        if isinstance(basis_or_fp, PCABasis):
            fp = basis_or_fp.fingerprint
        elif isinstance(basis_or_fp, (bytes, bytearray)) and len(basis_or_fp) == 32:
            fp = bytes(basis_or_fp)
        else:
            raise TypeError("basis_or_fp: PCABasis 또는 32바이트 지문")
        if not _is_int(max_entries) or int(max_entries) < 1:
            raise TapSpecError("max_entries >= 1 정수여야 한다")
        if max_bytes is not None and (not _is_int(max_bytes) or int(max_bytes) < 1):
            raise TapSpecError("max_bytes: None 또는 1 이상 정수")
        self.basis_fp = fp
        self.max_entries = int(max_entries)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.evicted = 0
        self._items: dict = {}     # seq -> TapSet (삽입 순서 = 오래된 순)
        self._nbytes = 0
        self._next = 0
        self._lock = threading.RLock()

    @staticmethod
    def _copy(ts: TapSet) -> TapSet:
        return dataclasses.replace(ts, layers={l: q.detach().clone() for l, q in ts.layers.items()})

    def push(self, ts) -> int:
        if not isinstance(ts, TapSet):
            raise TypeError("TapSet 이 아니다")
        if ts.ephemeral or ext_bit(ts.prov):
            raise FirewallError("ext/임시 탭은 링에 들어갈 수 없다")
        assert_train_ok(ts.prov, "TapRing.push")
        if ts.basis_fp != self.basis_fp:
            raise TapBasisError("링의 기저 지문과 탭의 기저 지문이 다르다")
        n = ts.nbytes
        if self.max_bytes is not None and n > self.max_bytes:
            raise TapError(f"항목({n}B)이 max_bytes({self.max_bytes}B)보다 크다")
        stored = dataclasses.replace(self._copy(ts), prov=as_record(ts.prov))
        with self._lock:
            while self._items and (len(self._items) >= self.max_entries or
                                   (self.max_bytes is not None and self._nbytes + n > self.max_bytes)):
                old = next(iter(self._items))
                self._nbytes -= self._items.pop(old).nbytes
                self.evicted += 1
            seq = self._next
            self._next += 1
            self._items[seq] = stored
            self._nbytes += n
            return seq

    def get(self, seq: int) -> TapSet:
        with self._lock:
            if seq not in self._items:
                raise KeyError(seq)
            return self._copy(self._items[seq])

    def seqs(self) -> list:
        with self._lock:
            return list(self._items)

    def latest(self, n: int) -> list:
        if not _is_int(n) or n < 0:
            raise ValueError("n >= 0 정수여야 한다")
        with self._lock:
            keys = list(self._items)
            keys = keys[len(keys) - min(n, len(keys)):]
            return [self._copy(self._items[s]) for s in keys]

    def sample(self, n: int, seed: int) -> list:
        """결정론 표본: 오름차순 seq. 해시 순위로 뽑아 플랫폼·버전에 무관하다."""
        with self._lock:
            keys = list(self._items)
        if not _is_int(n) or n < 0 or n > len(keys):
            raise ValueError(f"n 은 0..{len(keys)} 범위여야 한다")
        ranked = sorted(keys, key=lambda s: det.hash_parts("buc.tch03.sample.v1", int(seed), int(s)))
        return sorted(ranked[:n])

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._nbytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._nbytes


# --------------------------------------------------------------------------- 버스

class _CapState:
    __slots__ = ("grid", "off", "taps", "batch")

    def __init__(self, grid, off):
        self.grid = grid
        self.off = off
        self.taps: dict = {}
        self.batch: Optional[int] = None


class _Capture:
    """bus.capture(...)의 컨텍스트 매니저 객체. 정상 종료 뒤에만 .tapset."""

    def __init__(self, bus: "TapBus", prov, grid, off, key, ring):
        self._bus = bus
        self._prov = prov
        self._grid = grid
        self._off = off
        self._key = key
        self._ring = ring
        self._tapset: Optional[TapSet] = None
        self._entered = False

    @property
    def tapset(self) -> TapSet:
        if self._tapset is None:
            raise TapError("캡처가 정상 종료되기 전에는 tapset 이 없다")
        return self._tapset

    def __enter__(self) -> "_Capture":
        bus = self._bus
        if self._entered:
            raise TapError("이 캡처 객체는 이미 사용됐다")
        if not bus.attached:
            raise TapError("버스가 부착되지 않았다")
        if bus._state is not None:
            raise TapError("캡처는 중첩할 수 없다")
        self._entered = True
        bus._state = _CapState(self._grid, self._off)
        return self

    def __exit__(self, exc_type, exc, tb):
        bus = self._bus
        state, bus._state = bus._state, None
        if exc_type is not None:
            return False
        missing = [l for l in bus.layers if l not in state.taps]
        if missing:
            raise TapError(f"탭 층이 발화하지 않았다: {missing}")
        ephemeral = bool(ext_bit(self._prov))
        ts = TapSet(layers={l: state.taps[l] for l in bus.layers}, grid=self._grid,
                    basis_fp=bus.basis.fingerprint, prov=self._prov, ephemeral=ephemeral, key=self._key)
        bus._stats["captures"] += 1
        ring = self._ring
        if ring is not None:
            if ephemeral or not train_ok(self._prov):
                bus._stats["bypassed"] += 1
            else:
                ring.push(ts)
                bus._stats["pushed"] += 1
        self._tapset = ts
        return False


class TapBus:
    """층 훅 3개로 이미지 토큰 은닉을 즉시 PCA-int8로 사영한다. 훅은 출력을 바꾸지 않는다."""

    def __init__(self, basis: PCABasis, layers=TAP_LAYERS):
        if not isinstance(basis, PCABasis):
            raise TypeError("basis: PCABasis")
        self.basis = basis
        self.layers = _check_layers(layers)
        absent = [l for l in self.layers if l not in basis.layers]
        if absent:
            raise TapSpecError(f"기저에 없는 층: {absent}")
        self._handles: list = []
        self._state: Optional[_CapState] = None
        self._stats = {"captures": 0, "pushed": 0, "bypassed": 0}

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    @property
    def hook_count(self) -> int:
        return len(self._handles)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def attach(self, blocks) -> "TapBus":
        if self.attached:
            raise TapError("이미 부착돼 있다")
        self.basis.verify()
        try:
            n = len(blocks)
        except TypeError as e:
            raise TapSpecError("blocks: 길이가 있는 모듈 시퀀스여야 한다") from e
        bad = [l for l in self.layers if not (0 <= l < n)]
        if bad:
            raise TapSpecError(f"층 인덱스 범위 밖: {bad} (블록 {n}개)")
        mods = [blocks[l] for l in self.layers]
        handles = []
        try:
            for l, m in zip(self.layers, mods):
                handles.append(m.register_forward_hook(self._make_hook(l)))
        except Exception:
            for h in handles:
                h.remove()
            raise
        self._handles = handles
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def capture(self, prov, *, grid, token_offset: int = 0, key: Optional[str] = None,
                ring: Optional[TapRing] = None) -> _Capture:
        rec = as_record(prov)          # None·단일 str TypeError, 알 수 없는 태그 ValueError
        if (not isinstance(grid, (tuple, list)) or len(grid) != 2 or any(not _is_int(g) or g < 1 for g in grid)):
            raise TapSpecError("grid: 양의 정수 2-튜플 (gh, gw)")
        if not _is_int(token_offset) or token_offset < 0:
            raise TapSpecError("token_offset: 0 이상 정수")
        if ring is not None and not isinstance(ring, TapRing):
            raise TypeError("ring: TapRing")
        if key is not None and not isinstance(key, str):
            raise TypeError("key: str 또는 None")
        return _Capture(self, rec, (int(grid[0]), int(grid[1])), int(token_offset), key, ring)

    # -- 훅
    def _make_hook(self, layer: int):
        def hook(module, args, out):
            state = self._state
            if state is None:
                return None
            self._on_layer(state, layer, out)
            return None
        return hook

    def _on_layer(self, state: _CapState, layer: int, out) -> None:
        if layer in state.taps:
            raise TapError(f"층 {layer} 이 한 캡처에서 두 번 발화했다")
        h = out
        if isinstance(h, (tuple, list)):
            h = h[0] if h else None
        if not isinstance(h, torch.Tensor):
            raise TapSpecError(f"층 {layer}: 출력이 Tensor 가 아니다")
        if h.ndim != 3:
            raise TapSpecError(f"층 {layer}: 출력은 [B,S,D] 여야 한다 (got {tuple(h.shape)})")
        B, S, D = h.shape
        gh, gw = state.grid
        n_img = gh * gw
        if D != self.basis.dim:
            raise TapSpecError(f"층 {layer}: D={D} != 기저 {self.basis.dim}")
        if S < state.off + n_img:
            raise TapSpecError(f"층 {layer}: S={S} < token_offset+gh*gw={state.off + n_img}")
        if state.batch is not None and state.batch != B:
            raise TapSpecError(f"층 {layer}: 배치 {B} != {state.batch}")
        with torch.no_grad():
            img = h[:, state.off:state.off + n_img]
            q = self.basis.project(layer, img)                       # [B, n, k] int8
            q = q.reshape(B, gh, gw, self.basis.k).permute(0, 3, 1, 2).contiguous()
        state.batch = B
        state.taps[layer] = q
