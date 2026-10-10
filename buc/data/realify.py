"""DAT-02 Realify + 탭 동시 추출 패스: SDEdit 2-3 NFE, 결정 곡선 대역 마스크.

계약서: ops/harness/contracts/DAT-02.md. 공개 API는 계약서와 1:1이다.
- 합성 렌더를 엔진 교사로 SDEdit하는 한 번의 패스에서 사진풍 x'와 그 패스의 교사 탭을 함께 얻는다(추가 NFE 없음).
- 라벨 유효 대역 = {o: D(o, sigma) <= d_thresh}. 저대역 불일치(RMS) > tau 이면 폐기(keep=False).
- import는 torch·numpy·pillow와 buc 내부 모듈뿐이다. CUDA·diffusers를 끌어오지 않고 import 부작용이 없다.
"""
from __future__ import annotations

import io
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

from buc.core import det
from buc.data.shards import npy_encode
from buc.gov import prov as P
from buc.teacher import taps as T

__all__ = [
    "REALIFY_TAG", "CURVE_TAG", "SIGMA_RANGE", "ALLOWED_STEPS", "D_THRESH", "DEFAULT_TAU", "RealifyError",
    "max_octaves", "octave_map", "octave_split", "band_project", "DecisionCurve", "provisional_curve",
    "RealifyItem", "RealifyResult", "item_from_synth", "realify", "write_results",
]

REALIFY_TAG = "buc.dat02.realify.v1"
CURVE_TAG = "buc.dat02.curve.v1"
SIGMA_RANGE = (0.2, 0.7)
ALLOWED_STEPS = (2, 3)
D_THRESH = 0.1
DEFAULT_TAU = 0.05

_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")      # DAT-05 key 정규식과 같다
_LABEL_RE = re.compile(r"[A-Za-z0-9_]+")
_RESERVED_RE = re.compile(r"rgb|meta|tap[0-9]+")
_SEED_LIMIT = 1 << 64


class RealifyError(ValueError):
    """입력·곡선·매개변수가 계약을 어겼다."""


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_real(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


# --------------------------------------------------------------------------- 옥타브

def max_octaves(h: int, w: int) -> int:
    """floor(log2(min(h, w)))."""
    if not _is_int(h) or not _is_int(w) or h < 1 or w < 1:
        raise RealifyError(f"h, w는 양의 정수여야 한다: {h!r}, {w!r}")
    return int(min(int(h), int(w))).bit_length() - 1


def _check_n_oct(n_oct) -> int:
    if not _is_int(n_oct) or n_oct < 1:
        raise RealifyError(f"n_oct는 1 이상 정수여야 한다: {n_oct!r}")
    return int(n_oct)


def octave_map(h: int, w: int, n_oct: int) -> torch.Tensor:
    """FFT 배치 순서의 옥타브 번호 LongTensor[h, w]. r = max(|fy|, |fx|)."""
    n_oct = _check_n_oct(n_oct)
    if not _is_int(h) or not _is_int(w) or h < 1 or w < 1:
        raise RealifyError(f"h, w는 양의 정수여야 한다: {h!r}, {w!r}")
    fy = np.abs(np.fft.fftfreq(int(h)))
    fx = np.abs(np.fft.fftfreq(int(w)))
    r = np.maximum(fy[:, None], fx[None, :])
    out = np.full((int(h), int(w)), n_oct - 1, dtype=np.int64)
    for o in range(n_oct - 1):
        hi = 0.5 * 2.0 ** -o
        lo = 0.5 * 2.0 ** -(o + 1)
        out[(r > lo) & (r <= hi)] = o
    return torch.from_numpy(out)


def _masks(h: int, w: int, n_oct: int) -> torch.Tensor:
    om = octave_map(h, w, n_oct)
    return torch.stack([(om == o) for o in range(n_oct)]).to(torch.float64)


def _check_plane(x) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x는 텐서여야 한다: {type(x).__name__}")
    if x.ndim < 2:
        raise RealifyError(f"x는 [...,H,W] 여야 한다: {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x는 부동소수 텐서여야 한다: {x.dtype}")
    return x


def octave_split(x: torch.Tensor, n_oct: int) -> torch.Tensor:
    """x[...,H,W] -> [n_oct,...,H,W]. 하드 마스크 분할: 합이 x, 서로 직교. float64 계산, 입력 dtype 반환."""
    n_oct = _check_n_oct(n_oct)
    x = _check_plane(x)
    h, w = x.shape[-2:]
    masks = _masks(h, w, n_oct).to(x.device)
    xf = torch.fft.fft2(x.to(torch.float64), dim=(-2, -1))
    parts = [torch.fft.ifft2(xf * masks[o], dim=(-2, -1)).real for o in range(n_oct)]
    return torch.stack(parts).to(x.dtype)


def band_project(x: torch.Tensor, valid: Sequence[bool]) -> torch.Tensor:
    """유효 옥타브(n_oct = len(valid))의 합으로 사영. 입력 dtype 반환."""
    flags = [bool(v) for v in valid]
    if not flags:
        raise RealifyError("valid가 비었다")
    x = _check_plane(x)
    h, w = x.shape[-2:]
    masks = _masks(h, w, len(flags)).to(x.device)
    m = torch.zeros((h, w), dtype=torch.float64, device=x.device)
    for o, v in enumerate(flags):
        if v:
            m = m + masks[o]
    xf = torch.fft.fft2(x.to(torch.float64), dim=(-2, -1))
    return torch.fft.ifft2(xf * m, dim=(-2, -1)).real.to(x.dtype)


# --------------------------------------------------------------------------- 결정 곡선

@dataclass(frozen=True)
class DecisionCurve:
    sigmas: tuple
    D: tuple
    table_key: Optional[str] = None
    name: str = "custom"

    def __post_init__(self):
        try:
            sig = tuple(self.sigmas)
            rows = tuple(tuple(r) for r in self.D)
        except TypeError as e:
            raise RealifyError(f"sigmas·D는 열이어야 한다: {e}") from e
        if not sig:
            raise RealifyError("sigmas가 비었다")
        try:
            sig = tuple(float(s) for s in sig)
            rows = tuple(tuple(float(v) for v in r) for r in rows)
        except (TypeError, ValueError) as e:
            raise RealifyError(f"sigmas·D는 실수여야 한다: {e}") from e
        if not all(math.isfinite(s) and 0.0 < s <= 1.0 for s in sig):
            raise RealifyError(f"sigma는 (0, 1] 의 유한수여야 한다: {sig}")
        if any(not a > b for a, b in zip(sig, sig[1:])):
            raise RealifyError(f"sigmas는 엄격 내림차순이어야 한다: {sig}")
        if len(rows) != len(sig):
            raise RealifyError(f"D의 행 수 {len(rows)} != sigma 수 {len(sig)}")
        n_oct = len(rows[0])
        if n_oct < 1:
            raise RealifyError("n_oct는 1 이상이어야 한다")
        if any(len(r) != n_oct for r in rows):
            raise RealifyError("D의 행 길이가 같지 않다")
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for r in rows for v in r):
            raise RealifyError("D의 값은 유한한 [0, 1] 이어야 한다")
        if self.table_key is not None and not isinstance(self.table_key, str):
            raise RealifyError("table_key는 str 또는 None")
        if not isinstance(self.name, str):
            raise RealifyError("name은 str")
        object.__setattr__(self, "sigmas", sig)
        object.__setattr__(self, "D", rows)

    @property
    def n_octaves(self) -> int:
        return len(self.D[0])

    def _row(self, sigma, tol: float) -> int:
        if not _is_real(sigma) or not math.isfinite(float(sigma)):
            raise RealifyError(f"sigma가 유한한 실수가 아니다: {sigma!r}")
        s = float(sigma)
        best = min(range(len(self.sigmas)), key=lambda j: abs(self.sigmas[j] - s))
        if not abs(self.sigmas[best] - s) <= tol:
            raise RealifyError(f"sigma={sigma!r}는 곡선에 없다(곡선: {self.sigmas})")
        return best

    def at(self, sigma, tol: float = 1e-9) -> tuple:
        return self.D[self._row(sigma, tol)]

    def band_mask(self, sigma, thresh: float = D_THRESH) -> tuple:
        """D(o, sigma) <= thresh (경계 포함)."""
        return tuple(bool(v <= thresh) for v in self.at(sigma))

    @property
    def key(self) -> str:
        vals = list(self.sigmas) + [v for r in self.D for v in r]
        return det.hash_parts(CURVE_TAG, self.name, self.table_key or "", len(self.sigmas), self.n_octaves,
                              *[struct.pack("<d", v) for v in vals]).hex()


def provisional_curve(table, n_oct: int) -> DecisionCurve:
    """TCH-05 전의 자리값: r = ((1-s)/s) 2^(o-2), D = 1/(1+r^2)."""
    n_oct = _check_n_oct(n_oct)
    rows = []
    for s in table.points:
        row = []
        for o in range(n_oct):
            r = ((1.0 - s) / s) * 2.0 ** (o - 2)
            row.append(1.0 / (1.0 + r * r))
        rows.append(tuple(row))
    return DecisionCurve(sigmas=tuple(table.points), D=tuple(rows), table_key=table.key, name="provisional-v0")


# --------------------------------------------------------------------------- 입력·결과

@dataclass(frozen=True, kw_only=True)
class RealifyItem:
    key: str
    image: np.ndarray
    prov: object
    caption: str
    labels: Mapping = field(default_factory=dict)


def item_from_synth(sample, *, key: str, caption: str) -> RealifyItem:
    """DAT-01 SynthSample(덕 타이핑) -> RealifyItem. 라벨은 복사."""
    image = np.clip(sample.rgb_display(), 0, 1).astype(np.float32)
    labels = {k: np.array(getattr(sample, k), copy=True) for k in ("gbuf", "depth", "sem", "inst")}
    return RealifyItem(key=key, image=image, prov=P.as_record(sample.prov), caption=caption, labels=labels)


@dataclass(frozen=True, eq=False)
class RealifyResult:
    key: str
    image: np.ndarray
    taps: T.TapSet
    band_valid: tuple
    lowband_err: float
    keep: bool
    reject_reason: Optional[str]
    sigma: float
    tap_sigma: float
    steps: int
    seed: int
    labels: dict
    prov: P.ProvRecord
    curve_key: str
    table_key: str
    ring_seq: Optional[int] = None
    tau: float = DEFAULT_TAU
    d_thresh: float = D_THRESH

    def shard_fields(self) -> dict:
        """DAT-05 샤드 필드(정본 이름). rgb.png, tap<l>.npy, <label>.npy, meta.json."""
        u8 = np.round(self.image * 255).astype(np.uint8)
        bio = io.BytesIO()
        Image.fromarray(u8, "RGB").save(bio, format="PNG")
        out = {"rgb.png": bio.getvalue()}
        layers = sorted(self.taps.layers)
        for l in layers:
            out[f"tap{l}.npy"] = npy_encode(self.taps.layers[l][0].detach().cpu().numpy())
        for name in sorted(self.labels):
            out[f"{name}.npy"] = npy_encode(self.labels[name])
        meta = {
            "v": REALIFY_TAG, "key": self.key, "sigma": self.sigma, "tap_sigma": self.tap_sigma,
            "steps": self.steps, "seed": self.seed, "band_valid": [bool(v) for v in self.band_valid],
            "n_oct": len(self.band_valid), "lowband_err": self.lowband_err, "tau": self.tau,
            "d_thresh": self.d_thresh, "curve_key": self.curve_key, "table_key": self.table_key,
            "basis_fp": bytes(self.taps.basis_fp).hex(), "grid": list(self.taps.grid), "tap_layers": layers,
            "keep": bool(self.keep),
        }
        out["meta.json"] = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
        return out


# --------------------------------------------------------------------------- 검증

def _validate_items(items, spec) -> tuple:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("items는 RealifyItem 시퀀스여야 한다")
    items = list(items)
    if not items:
        raise RealifyError("items가 비었다")
    for it in items:
        if not isinstance(it, RealifyItem):
            raise TypeError(f"items의 원소는 RealifyItem이어야 한다: {type(it).__name__}")
    seen = set()
    shape = None
    for it in items:
        if not isinstance(it.key, str) or not _KEY_RE.fullmatch(it.key):
            raise RealifyError(f"key 형식 위반: {it.key!r}")
        if it.key in seen:
            raise RealifyError(f"중복 key: {it.key!r}")
        seen.add(it.key)
        if not isinstance(it.caption, str):
            raise TypeError(f"caption은 str이어야 한다: {it.key}")
        img = it.image
        if not isinstance(img, np.ndarray) or not np.issubdtype(img.dtype, np.floating):
            raise TypeError(f"image는 부동소수 ndarray여야 한다: {it.key}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise RealifyError(f"image는 [H,W,3] 이어야 한다: {it.key} {img.shape}")
        if shape is None:
            shape = tuple(img.shape)
            try:
                spec.check_image_size(shape[0], shape[1])
            except ValueError as e:
                raise RealifyError(str(e)) from e
        elif tuple(img.shape) != shape:
            raise RealifyError(f"image 모양이 다르다: {it.key} {tuple(img.shape)} != {shape}")
        if not bool(np.isfinite(img).all()):
            raise RealifyError(f"image에 NaN/Inf가 있다: {it.key}")
        if float(img.min()) < 0.0 or float(img.max()) > 1.0:
            raise RealifyError(f"image가 [0, 1] 밖이다: {it.key}")
        if not isinstance(it.labels, Mapping):
            raise TypeError(f"labels는 Mapping이어야 한다: {it.key}")
        for name, arr in it.labels.items():
            if not isinstance(name, str) or not _LABEL_RE.fullmatch(name) or _RESERVED_RE.fullmatch(name):
                raise RealifyError(f"label 이름 위반: {name!r}")
            if not isinstance(arr, np.ndarray) or arr.dtype.hasobject:
                raise TypeError(f"label은 (object 아닌) ndarray여야 한다: {name!r}")
    for it in items:
        rec = P.as_record(it.prov)
        if P.why_not_train(rec) or rec.bits != 0 or rec.ns != "train_ok":
            raise P.FirewallError(f"학습 불가 계보: {it.key} (why_not_train={P.why_not_train(rec)}, "
                                  f"bits={rec.bits}, ns={rec.ns})")
    return items, shape


class _TapWindow:
    """tap_step 번째 forward 하나만 bus.capture 안에서 돌게 하는 관찰자/훅 쌍. 예외 시에도 상태를 반드시 푼다."""

    def __init__(self, bus, prov, grid, token_offset: int, j: int):
        self.bus, self.prov, self.grid, self.off, self.j = bus, prov, grid, token_offset, j
        self.cap = None
        self.done = False

    def observer(self, info) -> None:
        if info.step == self.j and self.cap is None:
            cap = self.bus.capture(self.prov, grid=self.grid, token_offset=self.off)
            cap.__enter__()
            self.cap = cap

    def hook(self, x0_hat, info):
        if info.step == self.j and self.cap is not None and not self.done:
            self.done = True
            self.cap.__exit__(None, None, None)
        return x0_hat

    def abort(self, exc: BaseException) -> None:
        if self.cap is not None and not self.done:
            self.done = True
            try:
                self.cap.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass

    @property
    def tapset(self) -> T.TapSet:
        if self.cap is None or not self.done:
            raise RealifyError("탭 캡처가 일어나지 않았다")
        return self.cap.tapset


# --------------------------------------------------------------------------- 패스

def realify(sampler, bus, items, *, curve: DecisionCurve, sigma, seed, steps: int = 2, tau: float = DEFAULT_TAU,
            d_thresh: float = D_THRESH, token_offset: int = 0, tap_step: int = -1, ring=None) -> list:
    teacher = sampler.teacher
    table = sampler.table
    # ---- 1. 교사 호출 전 전부 검증
    items, shape = _validate_items(items, teacher.spec)
    h, w = shape[0], shape[1]
    if not isinstance(curve, DecisionCurve):
        raise TypeError("curve는 DecisionCurve여야 한다")
    if not _is_int(seed):
        raise TypeError(f"seed는 정수여야 한다: {seed!r}")
    seed = int(seed)
    if not 0 <= seed < _SEED_LIMIT:
        raise RealifyError(f"seed가 [0, 2^64) 밖이다: {seed}")
    if not _is_int(steps):
        raise TypeError(f"steps는 정수여야 한다: {steps!r}")
    if steps not in ALLOWED_STEPS:
        raise RealifyError(f"steps는 {ALLOWED_STEPS} 중 하나여야 한다: {steps}")
    if not _is_int(tap_step):
        raise TypeError(f"tap_step은 정수여야 한다: {tap_step!r}")
    if not _is_real(tau) or not math.isfinite(float(tau)) or float(tau) < 0.0:
        raise RealifyError(f"tau는 0 이상 유한수여야 한다: {tau!r}")
    if not _is_real(d_thresh) or not (0.0 <= float(d_thresh) <= 1.0):
        raise RealifyError(f"d_thresh는 [0, 1] 이어야 한다: {d_thresh!r}")
    if not _is_int(token_offset) or token_offset < 0:
        raise RealifyError(f"token_offset은 0 이상 정수여야 한다: {token_offset!r}")
    tau, d_thresh = float(tau), float(d_thresh)
    try:
        snap = table.snap(sigma)
    except ValueError as e:
        raise RealifyError(str(e)) from e
    if not (SIGMA_RANGE[0] <= snap.sigma <= SIGMA_RANGE[1]):
        raise RealifyError(f"스냅 sigma={snap.sigma}가 {SIGMA_RANGE} 밖이다")
    n = min(int(steps), len(table) - snap.index)
    if not -n <= tap_step < n:
        raise RealifyError(f"tap_step은 [{-n}, {n}) 이어야 한다: {tap_step}")
    j = int(tap_step) % n
    band_valid = curve.band_mask(snap.sigma, d_thresh)             # 곡선에 σ_s가 없으면 RealifyError
    if curve.table_key is not None and curve.table_key != table.key:
        raise RealifyError("curve.table_key가 sampler.table.key와 다르다")
    if curve.n_octaves > max_octaves(h, w):
        raise RealifyError(f"곡선 옥타브 {curve.n_octaves} > 영상 최대 {max_octaves(h, w)}")
    if not getattr(bus, "attached", False):
        raise RealifyError("bus가 부착되지 않았다")
    if ring is not None:
        if not isinstance(ring, T.TapRing):
            raise TypeError("ring은 TapRing이어야 한다")
        if ring.basis_fp != bus.basis.fingerprint:
            raise T.TapBasisError("링의 기저 지문이 버스의 기저와 다르다")

    # ---- 2. 청크별 SDEdit + 탭
    chunk = sampler.max_views_per_call or len(items)
    tap_sigma = float(table.points[snap.index + j])
    grid = (h // 16, w // 16)
    results = []
    with torch.no_grad():
        for lo in range(0, len(items), chunk):
            its = items[lo:lo + chunk]
            x_in = torch.stack([torch.from_numpy(np.ascontiguousarray(it.image, dtype=np.float32)).permute(2, 0, 1)
                                for it in its]) * 2 - 1
            x0 = teacher.encode_image(x_in)
            seeds = [det.derive_seed(seed, it.key) for it in its]
            chunk_prov = P.derive("realify", [it.prov for it in its])
            win = _TapWindow(bus, chunk_prov, grid, int(token_offset), j)
            try:
                res = sampler.sdedit(x0, [it.caption for it in its], sigma=sigma, seeds=seeds, steps=n,
                                     observer=win.observer, x0_hook=win.hook)
            except BaseException as e:
                win.abort(e)
                raise
            ts = win.tapset
            img = ((teacher.decode_latent(res.x0_hat) + 1) / 2).clamp(0, 1)
            img = img.permute(0, 2, 3, 1).detach().cpu().to(torch.float32).numpy()
            for i, it in enumerate(its):
                xp = np.ascontiguousarray(img[i], dtype=np.float32)
                # ---- 3. 저대역 불일치
                if not any(band_valid):
                    err, keep, why = 0.0, False, "no_band"
                else:
                    d = torch.from_numpy((xp.astype(np.float64) - it.image.astype(np.float64)).transpose(2, 0, 1)
                                         .copy())
                    p = band_project(d, band_valid)
                    err = float(torch.sqrt(torch.mean(p * p)))
                    keep = bool(err <= tau)
                    why = None if keep else "lowband"
                rprov = P.derive("realify", [it.prov])
                tap_i = T.TapSet(layers={l: q[i:i + 1].detach().clone() for l, q in ts.layers.items()},
                                 grid=ts.grid, basis_fp=ts.basis_fp, prov=rprov, ephemeral=False, key=it.key)
                results.append(RealifyResult(
                    key=it.key, image=xp, taps=tap_i, band_valid=tuple(band_valid), lowband_err=err, keep=keep,
                    reject_reason=why, sigma=float(snap.sigma), tap_sigma=tap_sigma, steps=int(n),
                    seed=int(seeds[i]), labels={k: np.array(v, copy=True) for k, v in it.labels.items()},
                    prov=rprov, curve_key=curve.key, table_key=table.key, ring_seq=None, tau=tau,
                    d_thresh=d_thresh))

    # ---- 4. 링: keep 인 것만 항목별로
    if ring is not None:
        out = []
        for r in results:
            if r.keep:
                seq = ring.push(r.taps)
                r = _with_seq(r, seq)
            out.append(r)
        results = out
    return results


def _with_seq(r: RealifyResult, seq: int) -> RealifyResult:
    import dataclasses
    return dataclasses.replace(r, ring_seq=int(seq))


def write_results(writer, results) -> int:
    """keep 인 결과만 샤드에 쓴다. 쓴 개수를 돌려준다."""
    n = 0
    for r in results:
        if r.keep:
            writer.write(r.key, r.shard_fields(), r.prov)
            n += 1
    return n
