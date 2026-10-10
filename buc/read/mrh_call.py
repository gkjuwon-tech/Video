"""RD-07 R5 독심 판독 호출 v0: 전역 σ 2점 + 1024^2 타일 -> 가장자리 혼합 -> r_dis, 클린룸 레인.

계약서: ops/harness/contracts/RD-07.md. 공개 API는 계약서와 1:1이다.
- 순수 수치(global_size, plan_tiles, tile_weights, affine_fit/residual, fuse_global, blend_reads)는 torch CPU 결정론.
- MrhReader: 엔진 교사 sigma-스냅 forward 1회 -> TapBus 탭 -> MRH. TapSet은 보관하지 않는다.
- read_mrh: 전역 뷰(σ 순) -> 타일(행 우선) 호출, 융합·혼합·r_dis, ext 입력은 클린룸 레인 규칙.
- numpy·torch만. CUDA·diffusers 없이 import 되고 import 부작용·전역 RNG 변경이 없다.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from buc.core import color, det
from buc.gov import prov as P
from buc.models.mrh.heads import HEAD_NAMES
from buc.teacher.sample import add_noise
from buc.teacher.taps import TapRing

__all__ = [
    "STRIDE", "UNIT", "TILE_PX", "GLOBAL_PIXELS", "READ_LONG_MAX", "FEATHER_PX", "GLOBAL_SIGMAS", "TILE_SIGMA",
    "DEPTH_FLOOR", "UNC_TAU", "SURVEY_CAPTION", "LANE_ENGINE", "LANE_CLEANROOM",
    "MrhCallError", "MrhCallSpecError", "CleanroomError",
    "global_size", "TileBox", "plan_tiles", "tile_weights", "affine_fit", "affine_residual",
    "FusedGlobal", "fuse_global", "BlendResult", "blend_reads",
    "ViewRead", "MrhReader", "ReadConfig", "lane_for", "MrhReadout", "read_mrh",
]

STRIDE = 4
UNIT = 16
TILE_PX = 1024
GLOBAL_PIXELS = 589_824
READ_LONG_MAX = 3840
FEATHER_PX = 64
GLOBAL_SIGMAS = (0.20, 0.35)
TILE_SIGMA = 0.25
DEPTH_FLOOR = 0.05
UNC_TAU = 0.10
SURVEY_CAPTION = "a photograph of a scene, shown in full detail with its geometry, materials and lighting"
LANE_ENGINE = "engine"
LANE_CLEANROOM = "cleanroom"

_SEED_LIMIT = 1 << 64
_UNIT_KEYS = ("normal", "sem")
_EPS = 1e-6


class MrhCallError(RuntimeError):
    """판독 호출 실패(읽기기 출력 불량 등)."""


class MrhCallSpecError(MrhCallError, ValueError):
    """인자·설정이 계약을 어겼다."""


class CleanroomError(MrhCallError):
    """클린룸 레인 규칙 위반."""


# --------------------------------------------------------------------------- 작은 도우미

def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_real(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(_EPS)


def _round16(v: float) -> int:
    return 16 * int(round(v / 16.0))


# --------------------------------------------------------------------------- 전역 크기·타일 계획

def global_size(h: int, w: int, pixels: int = GLOBAL_PIXELS) -> tuple:
    if not _is_int(h) or not _is_int(w) or h < 1 or w < 1:
        raise ValueError(f"h, w는 양의 정수여야 한다: {h!r}, {w!r}")
    s = min(1.0, math.sqrt(pixels / (h * w)))

    def side(length: int) -> int:
        g = max(UNIT, UNIT * int(round(s * length / UNIT)))
        if g > length and length >= UNIT:
            g = UNIT * (length // UNIT)          # 업스케일 없음
        return g

    return (side(int(h)), side(int(w)))


@dataclass(frozen=True)
class TileBox:
    y0: int
    x0: int
    y1: int
    x1: int


def _axis_starts(length: int, tile: int) -> tuple:
    """(시작 목록, 타일 크기)."""
    if length <= tile:
        return (0,), length
    n = -(-length // tile)
    size = tile
    span = length - size
    starts = [0]
    for i in range(1, n - 1):
        starts.append(_round16(i * span / (n - 1)))
    starts.append(span)
    # 안전 보정: 엄격 증가·틈 없음
    for i in range(1, n - 1):
        starts[i] = max(starts[i], starts[i - 1] + UNIT)
        starts[i] = min(starts[i], starts[i - 1] + size)
    return tuple(starts), size


def plan_tiles(h: int, w: int, tile: int = TILE_PX) -> tuple:
    for name, v in (("h", h), ("w", w), ("tile", tile)):
        if not _is_int(v) or v < UNIT or v % UNIT:
            raise ValueError(f"{name}는 16의 양의 배수여야 한다: {v!r}")
    h, w, tile = int(h), int(w), int(tile)
    if h <= tile and w <= tile:
        return ()
    ys, sh = _axis_starts(h, tile)
    xs, sw = _axis_starts(w, tile)
    return tuple(TileBox(y, x, y + sh, x + sw) for y in ys for x in xs)


def _check_canvas(canvas_hw) -> tuple:
    try:
        h, w = canvas_hw
    except (TypeError, ValueError):
        raise ValueError(f"canvas_hw는 (H, W)여야 한다: {canvas_hw!r}") from None
    if not _is_int(h) or not _is_int(w) or h < UNIT or w < UNIT or h % UNIT or w % UNIT:
        raise ValueError(f"canvas_hw는 16의 배수여야 한다: {canvas_hw!r}")
    return int(h), int(w)


def _check_box(b, canvas_hw) -> None:
    if not isinstance(b, TileBox):
        raise ValueError(f"TileBox가 아니다: {b!r}")
    h, w = canvas_hw
    vals = (b.y0, b.x0, b.y1, b.x1)
    if any(not _is_int(v) or v % UNIT for v in vals) or not (0 <= b.y0 < b.y1 <= h and 0 <= b.x0 < b.x1 <= w):
        raise ValueError(f"타일 박스가 캔버스 {canvas_hw} 안의 16 배수 영역이 아니다: {b}")


def _axis_weight(c0: int, c1: int, n: int, fc: float) -> torch.Tensor:
    i = torch.arange(c0, c1, dtype=torch.float64)
    if fc == 0:
        return torch.ones_like(i)
    d = torch.full_like(i, float("inf"))
    if c0 > 0:
        d = torch.minimum(d, i - c0 + 0.5)
    if c1 < n:
        d = torch.minimum(d, c1 - i - 0.5)
    u = (d / fc).clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def tile_weights(boxes: Sequence, canvas_hw, feather_px: int = FEATHER_PX) -> torch.Tensor:
    ch, cw = _check_canvas(canvas_hw)
    if not _is_real(feather_px) or feather_px < 0 or not math.isfinite(float(feather_px)):
        raise ValueError(f"feather_px는 0 이상의 수여야 한다: {feather_px!r}")
    h4, w4 = ch // STRIDE, cw // STRIDE
    fc = float(feather_px) / STRIDE
    out = torch.zeros(len(boxes), h4, w4, dtype=torch.float64)
    for t, b in enumerate(boxes):
        _check_box(b, (ch, cw))
        y0, y1, x0, x1 = b.y0 // STRIDE, b.y1 // STRIDE, b.x0 // STRIDE, b.x1 // STRIDE
        out[t, y0:y1, x0:x1] = _axis_weight(y0, y1, h4, fc)[:, None] * _axis_weight(x0, x1, w4, fc)[None, :]
    return out.to(torch.float32)


# --------------------------------------------------------------------------- affine

def _prep(src, ref, weight):
    if not isinstance(src, torch.Tensor) or not isinstance(ref, torch.Tensor):
        raise ValueError("src, ref는 텐서여야 한다")
    if tuple(src.shape) != tuple(ref.shape):
        raise ValueError(f"모양 불일치: {tuple(src.shape)} vs {tuple(ref.shape)}")
    if src.numel() == 0:
        raise ValueError("빈 텐서")
    s = src.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    if weight is None:
        w = torch.ones_like(s)
    else:
        if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(src.shape):
            raise ValueError("weight 모양이 src와 다르다")
        w = weight.detach().to(torch.float64)
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError("weight는 유한한 비음수여야 한다")
    tot = float(w.sum())
    if not tot > 0.0:
        raise ValueError("가중치 합이 0이다")
    if not (bool(torch.isfinite(s).all()) and bool(torch.isfinite(r).all())):
        raise ValueError("src/ref에 비유한 값이 있다")
    return s, r, w, tot


def affine_fit(src: torch.Tensor, ref: torch.Tensor, weight: Optional[torch.Tensor] = None) -> tuple:
    """가중 최소제곱 a*src+b ~ ref, a >= 0. 퇴화·음수 기울기면 (0, ref 가중평균)."""
    s, r, w, tot = _prep(src, ref, weight)
    ms = float((w * s).sum()) / tot
    mr = float((w * r).sum()) / tot
    var = float((w * (s - ms) ** 2).sum()) / tot
    if var <= 1e-12:
        return (0.0, mr)
    cov = float((w * (s - ms) * (r - mr)).sum()) / tot
    a = cov / var
    if a < 0.0:
        return (0.0, mr)
    return (a, mr - a * ms)


def affine_residual(src: torch.Tensor, ref: torch.Tensor, weight: Optional[torch.Tensor] = None,
                    floor: float = DEPTH_FLOOR) -> tuple:
    """정렬 뒤 (rel, resid_map). rel = rms / max(ref 가중 표준편차, floor)."""
    a, b = affine_fit(src, ref, weight)
    s, r, w, tot = _prep(src, ref, weight)
    resid = (a * s + b - r).abs()
    rms = math.sqrt(float((w * resid ** 2).sum()) / tot)
    mr = float((w * r).sum()) / tot
    std = math.sqrt(max(0.0, float((w * (r - mr) ** 2).sum()) / tot))
    den = max(std, float(floor))
    return (rms / den, (resid / den).to(torch.float32))


# --------------------------------------------------------------------------- 전역 융합

@dataclass(frozen=True)
class FusedGlobal:
    outputs: dict
    r_dis_sigma: float
    sigma_map: torch.Tensor


def _check_reads(reads: Sequence, what: str) -> list:
    if isinstance(reads, Mapping) or not isinstance(reads, Sequence) or len(reads) < 1:
        raise ValueError(f"{what}: 비어 있지 않은 시퀀스여야 한다")
    reads = list(reads)
    keys = set(reads[0])
    for r in reads:
        if set(r) != keys:
            raise ValueError(f"{what}: 헤드 키가 서로 다르다")
        for k, v in r.items():
            if not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[0] != 1:
                raise ValueError(f"{what}: {k}는 [1,c,h,w] 텐서여야 한다")
            if tuple(v.shape) != tuple(reads[0][k].shape):
                raise ValueError(f"{what}: {k} 모양이 서로 다르다")
    return reads


def fuse_global(reads: Sequence, *, depth_key: str = "log_depth", floor: float = DEPTH_FLOOR) -> FusedGlobal:
    reads = _check_reads(reads, "fuse_global")
    if depth_key not in reads[0]:
        raise ValueError(f"깊이 헤드 {depth_key!r}가 없다")
    h, w = reads[0][depth_key].shape[-2:]
    if len(reads) == 1:
        outs = {k: v.detach().to(torch.float32).clone() for k, v in reads[0].items()}
        return FusedGlobal(outs, 0.0, torch.zeros(h, w, dtype=torch.float32))
    ref = reads[0][depth_key].detach().to(torch.float32)
    aligned = [ref]
    rels, maps = [], []
    for r in reads[1:]:
        d = r[depth_key].detach().to(torch.float32)
        a, b = affine_fit(d, ref)
        aligned.append((a * d.double() + b).to(torch.float32))
        rel, rmap = affine_residual(d, ref, floor=floor)
        rels.append(rel)
        maps.append(rmap[0].mean(0))
    outs = {}
    for k in reads[0]:
        if k == depth_key:
            m = torch.stack([x.double() for x in aligned]).mean(0).to(torch.float32)
        else:
            m = torch.stack([r[k].detach().double() for r in reads]).mean(0).to(torch.float32)
            if k in _UNIT_KEYS:
                m = _unit(m)
        outs[k] = m
    return FusedGlobal(outs, float(sum(rels) / len(rels)), torch.stack(maps).mean(0).to(torch.float32))


# --------------------------------------------------------------------------- 타일 혼합

@dataclass(frozen=True)
class BlendResult:
    outputs: dict
    coverage: torch.Tensor
    weights: torch.Tensor
    affine: tuple
    rel: tuple
    r_dis_tiles: float
    tile_map: torch.Tensor


def blend_reads(glob: Mapping, tiles: Sequence, *, canvas_hw, feather_px: int = FEATHER_PX,
                depth_key: str = "log_depth", floor: float = DEPTH_FLOOR) -> BlendResult:
    ch, cw = _check_canvas(canvas_hw)
    h4, w4 = ch // STRIDE, cw // STRIDE
    g = _check_reads([glob], "blend_reads(glob)")[0]
    if depth_key not in g:
        raise ValueError(f"깊이 헤드 {depth_key!r}가 없다")
    for k, v in g.items():
        if tuple(v.shape[-2:]) != (h4, w4):
            raise ValueError(f"전역 {k} 격자 {tuple(v.shape[-2:])} != 출력 격자 {(h4, w4)}")
    boxes = []
    reads = []
    for item in tiles:
        try:
            b, out = item
        except (TypeError, ValueError):
            raise ValueError("tiles의 원소는 (TileBox, 출력) 쌍이어야 한다") from None
        _check_box(b, (ch, cw))
        if set(out) != set(g):
            raise ValueError("타일과 전역의 헤드 키가 다르다")
        gh_t, gw_t = (b.y1 - b.y0) // STRIDE, (b.x1 - b.x0) // STRIDE
        for k, v in out.items():
            if (not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[0] != 1
                    or v.shape[1] != g[k].shape[1] or tuple(v.shape[-2:]) != (gh_t, gw_t)):
                raise ValueError(f"타일 {k} 모양 {tuple(getattr(v, 'shape', ()))}이 박스 격자와 다르다")
        boxes.append(b)
        reads.append(out)
    gf = {k: v.detach().to(torch.float32) for k, v in g.items()}
    if not boxes:
        return BlendResult({k: v.clone() for k, v in gf.items()}, torch.zeros(h4, w4), torch.zeros(0, h4, w4),
                           (), (), 0.0, torch.zeros(h4, w4))
    w = tile_weights(boxes, (ch, cw), feather_px)
    W = w.sum(0)
    wmax = W.clamp_min(1.0)
    fill = (1.0 - W).clamp_min(0.0)
    num = {k: fill[None, None] * v for k, v in gf.items()}
    tmap = torch.zeros(h4, w4)
    affs, rels = [], []
    for t, (b, out) in enumerate(zip(boxes, reads)):
        y0, y1, x0, x1 = b.y0 // STRIDE, b.y1 // STRIDE, b.x0 // STRIDE, b.x1 // STRIDE
        wt = w[t, y0:y1, x0:x1][None, None]
        den = wmax[y0:y1, x0:x1][None, None]
        gd = gf[depth_key][..., y0:y1, x0:x1]
        td = out[depth_key].detach().to(torch.float32)
        a, bb = affine_fit(td, gd)
        rel, rmap = affine_residual(td, gd, floor=floor)
        affs.append((a, bb))
        rels.append(rel)
        for k in gf:
            v = out[k].detach().to(torch.float32)
            if k == depth_key:
                v = (a * v.double() + bb).to(torch.float32)
            num[k][..., y0:y1, x0:x1] += wt / den * v
        tmap[y0:y1, x0:x1] += w[t, y0:y1, x0:x1] * rmap[0].mean(0)
    outs = {}
    for k, v in num.items():
        outs[k] = _unit(v) if k in _UNIT_KEYS else v
    ok = W >= 1e-6
    tile_map = torch.where(ok, tmap / W.clamp_min(1e-6), torch.zeros_like(tmap))
    return BlendResult(outs, W.clamp(0.0, 1.0), w, tuple(affs), tuple(rels), float(sum(rels) / len(rels)),
                       tile_map)


# --------------------------------------------------------------------------- 읽기기

@dataclass(frozen=True)
class ViewRead:
    outputs: Mapping
    sigma: float
    digest: bytes
    ephemeral: bool


class MrhReader:
    """교사 sigma-스냅 forward 1회 -> 탭 -> MRH. 읽기기 규약: snap(sigma), __call__(image, sigma, *, prov, key)."""

    def __init__(self, sampler, bus, mrh, *, token_offset: int = 0, caption: str = SURVEY_CAPTION, seed: int = 0,
                 ring=None):
        try:
            layers = tuple(mrh.cfg.tap_layers)
            dim = int(mrh.cfg.tap_dim)
        except AttributeError:
            raise MrhCallSpecError("mrh.cfg(tap_layers, tap_dim)가 없다") from None
        if layers != tuple(bus.layers):
            raise MrhCallSpecError(f"mrh.cfg.tap_layers {layers} != bus.layers {tuple(bus.layers)}")
        if dim != int(bus.basis.k):
            raise MrhCallSpecError(f"mrh.cfg.tap_dim {dim} != bus.basis.k {bus.basis.k}")
        if not _is_int(token_offset) or token_offset < 0:
            raise MrhCallSpecError(f"token_offset는 0 이상 정수여야 한다: {token_offset!r}")
        if not _is_int(seed) or not (0 <= int(seed) < _SEED_LIMIT):
            raise MrhCallSpecError(f"seed는 [0, 2^64) 정수여야 한다: {seed!r}")
        if not isinstance(caption, str):
            raise MrhCallSpecError(f"caption은 str이어야 한다: {caption!r}")
        if ring is not None and not isinstance(ring, TapRing):
            raise TypeError("ring은 TapRing 또는 None이어야 한다")
        self.sampler = sampler
        self.bus = bus
        self.mrh = mrh
        self.token_offset = int(token_offset)
        self.caption = caption
        self.seed = int(seed)
        self.ring = ring

    def snap(self, sigma) -> float:
        return float(self.sampler.table.snap(sigma).sigma)

    def _check_image(self, image) -> tuple:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image는 텐서여야 한다: {type(image).__name__}")
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"image는 [1,3,h,w]여야 한다: {tuple(image.shape)}")
        if not image.is_floating_point():
            raise TypeError(f"image는 부동소수여야 한다: {image.dtype}")
        h, w = int(image.shape[2]), int(image.shape[3])
        if h < UNIT or w < UNIT or h % UNIT or w % UNIT:
            raise ValueError(f"h, w는 16의 양의 배수여야 한다: {h}x{w}")
        d = image.detach()
        if not bool(torch.isfinite(d).all()):
            raise ValueError("image에 NaN/Inf가 있다")
        if float(d.min()) < -1.0 or float(d.max()) > 1.0:
            raise ValueError("image는 [-1, 1] 범위여야 한다")
        return h, w

    def __call__(self, image, sigma, *, prov, key) -> ViewRead:
        h, w = self._check_image(image)
        rec = P.as_record(prov)
        if not isinstance(key, str) or not key:
            raise ValueError(f"key는 비어 있지 않은 str이어야 한다: {key!r}")
        if not self.bus.attached:
            raise MrhCallError("TapBus가 부착되지 않았다")
        sig = self.snap(sigma)
        teacher = self.sampler.teacher
        x0 = teacher.encode_image(image.detach().to(torch.float32))
        eps = teacher.make_noise(det.derive_seed(self.seed, key), 1, h, w)
        z = add_noise(x0, sig, eps)
        cap = self.bus.capture(rec, grid=(h // UNIT, w // UNIT), token_offset=self.token_offset, key=key,
                               ring=self.ring)
        with cap:
            self.sampler.predict_x0(z, sig, [self.caption])
        ts = cap.tapset
        feats = ts.features(self.bus.basis)
        with torch.no_grad():
            out = self.mrh(feats, sig)
        outputs = {k: v.detach().to(device="cpu", dtype=torch.float32).clone() for k, v in out.items()}
        return ViewRead(outputs=outputs, sigma=sig, digest=ts.digest(), ephemeral=bool(ts.ephemeral))


# --------------------------------------------------------------------------- 설정

@dataclass(frozen=True, kw_only=True)
class ReadConfig:
    global_sigmas: tuple = GLOBAL_SIGMAS
    tile_sigma: float = TILE_SIGMA
    tiles: bool = True
    tile_px: int = TILE_PX
    global_pixels: int = GLOBAL_PIXELS
    read_long_max: int = READ_LONG_MAX
    feather_px: int = FEATHER_PX
    heads: tuple = HEAD_NAMES
    depth_floor: float = DEPTH_FLOOR
    unc_tau: float = UNC_TAU

    def __post_init__(self):
        def bad(msg):
            raise MrhCallSpecError(msg)

        def sig(v, name):
            if not _is_real(v) or not math.isfinite(float(v)) or not (0.0 < float(v) <= 1.0):
                bad(f"{name}는 (0, 1]의 유한수여야 한다: {v!r}")
            return float(v)

        try:
            gs = tuple(self.global_sigmas)
        except TypeError:
            bad(f"global_sigmas는 열이어야 한다: {self.global_sigmas!r}")
        if not 1 <= len(gs) <= 2:
            bad(f"global_sigmas는 1-2개여야 한다: {gs!r}")
        object.__setattr__(self, "global_sigmas", tuple(sig(s, "global_sigmas") for s in gs))
        object.__setattr__(self, "tile_sigma", sig(self.tile_sigma, "tile_sigma"))
        if not isinstance(self.tiles, (bool, np.bool_)):
            bad(f"tiles는 bool이어야 한다: {self.tiles!r}")
        object.__setattr__(self, "tiles", bool(self.tiles))
        if not _is_int(self.tile_px) or self.tile_px < UNIT or self.tile_px % UNIT:
            bad(f"tile_px는 16의 양의 배수여야 한다: {self.tile_px!r}")
        if not _is_int(self.global_pixels) or self.global_pixels < 256:
            bad(f"global_pixels는 256 이상 정수여야 한다: {self.global_pixels!r}")
        if not _is_int(self.read_long_max) or self.read_long_max < UNIT or self.read_long_max % UNIT:
            bad(f"read_long_max는 16의 양의 배수여야 한다: {self.read_long_max!r}")
        if not _is_int(self.feather_px) or self.feather_px < 0 or self.feather_px % STRIDE:
            bad(f"feather_px는 4의 배수인 0 이상 정수여야 한다: {self.feather_px!r}")
        try:
            heads = tuple(self.heads)
        except TypeError:
            bad(f"heads는 열이어야 한다: {self.heads!r}")
        if (not heads or len(set(heads)) != len(heads) or any(h not in HEAD_NAMES for h in heads)
                or "log_depth" not in heads):
            bad(f"heads는 log_depth를 포함한 HEAD_NAMES의 중복 없는 부분집합이어야 한다: {heads!r}")
        object.__setattr__(self, "heads", heads)
        for name in ("depth_floor", "unc_tau"):
            v = getattr(self, name)
            if not _is_real(v) or not math.isfinite(float(v)) or not float(v) > 0.0:
                bad(f"{name}는 양의 유한수여야 한다: {v!r}")
            object.__setattr__(self, name, float(v))


def lane_for(prov) -> str:
    return LANE_CLEANROOM if P.as_record(prov).ext else LANE_ENGINE


# --------------------------------------------------------------------------- 결과

@dataclass(frozen=True)
class MrhReadout:
    outputs: dict
    depth_global: torch.Tensor
    coverage: torch.Tensor
    r_dis_map: torch.Tensor
    uncertain: torch.Tensor
    r_dis: float
    r_dis_tiles: float
    r_dis_sigma: float
    uncertain_frac: float
    tiles: tuple
    tile_affine: tuple
    global_size: tuple
    global_sigmas: tuple
    tile_sigma: Optional[float]
    sigma_pair_degenerate: bool
    src_hw: tuple
    read_hw: tuple
    canvas_hw: tuple
    n_forward: int
    lane: str
    ext: bool
    ns: str
    prov: Any
    cleanroom_binary_hash: Optional[str]
    key: str
    digests: tuple = field(default=())

    def meta(self) -> dict:
        return {
            "lane": self.lane, "ext": bool(self.ext), "ns": self.ns, "key": self.key,
            "n_forward": int(self.n_forward), "n_tiles": len(self.tiles),
            "global_sigmas": [float(s) for s in self.global_sigmas],
            "tile_sigma": None if self.tile_sigma is None else float(self.tile_sigma),
            "sigma_pair_degenerate": bool(self.sigma_pair_degenerate),
            "r_dis": float(self.r_dis), "r_dis_tiles": float(self.r_dis_tiles),
            "r_dis_sigma": float(self.r_dis_sigma), "uncertain_frac": float(self.uncertain_frac),
            "src_hw": [int(v) for v in self.src_hw], "read_hw": [int(v) for v in self.read_hw],
            "canvas_hw": [int(v) for v in self.canvas_hw], "global_size": [int(v) for v in self.global_size],
            "digests": [d.hex() for d in self.digests],
            "cleanroom_binary_hash": self.cleanroom_binary_hash,
        }


# --------------------------------------------------------------------------- 오케스트레이터

def _validate_image(image_lin) -> np.ndarray:
    if not isinstance(image_lin, np.ndarray):
        raise TypeError(f"image_lin은 numpy 배열이어야 한다: {type(image_lin).__name__}")
    if not np.issubdtype(image_lin.dtype, np.floating):
        raise TypeError(f"image_lin은 실수 dtype이어야 한다: {image_lin.dtype}")
    if image_lin.ndim != 3 or image_lin.shape[2] != 3:
        raise ValueError(f"image_lin은 (H, W, 3)이어야 한다: {image_lin.shape}")
    h, w = image_lin.shape[:2]
    if min(h, w) < UNIT:
        raise ValueError(f"짧은 변은 16 이상이어야 한다: {h}x{w}")
    if not bool(np.isfinite(image_lin).all()):
        raise ValueError("image_lin에 NaN/Inf가 있다")
    if float(image_lin.min()) < 0.0:
        raise ValueError("image_lin은 0 이상이어야 한다")
    return image_lin


def _check_view(view, heads: Sequence, hw4: tuple, what: str) -> dict:
    out = getattr(view, "outputs", None)
    if not isinstance(out, Mapping):
        raise MrhCallError(f"{what}: 읽기기 출력이 매핑이 아니다")
    res = {}
    for k in heads:
        v = out.get(k)
        if not isinstance(v, torch.Tensor):
            raise MrhCallError(f"{what}: 헤드 {k!r}가 없다")
        if v.ndim != 4 or v.shape[0] != 1 or tuple(v.shape[-2:]) != tuple(hw4):
            raise MrhCallError(f"{what}: 헤드 {k!r} 모양 {tuple(v.shape)}이 [1,c,{hw4[0]},{hw4[1]}]가 아니다")
        v = v.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(v).all()):
            raise MrhCallError(f"{what}: 헤드 {k!r}에 NaN/Inf가 있다")
        res[k] = v
    return res


def _resize_grid(t: torch.Tensor, hw: tuple) -> torch.Tensor:
    if tuple(t.shape[-2:]) == tuple(hw):
        return t
    return F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False)


def read_mrh(image_lin, reader, *, prov, cfg: Optional[ReadConfig] = None, key: str = "read",
             lane: Optional[str] = None, cleanroom_binary_hash: Optional[str] = None) -> MrhReadout:
    # ---- 1. 검증(읽기기 호출 전에 전부)
    cfg = ReadConfig() if cfg is None else cfg
    if not isinstance(cfg, ReadConfig):
        raise MrhCallSpecError("cfg는 ReadConfig여야 한다")
    img = _validate_image(image_lin)
    rec = P.as_record(prov)
    if not isinstance(key, str):
        raise TypeError(f"key는 str이어야 한다: {key!r}")
    if not key:
        raise ValueError("key는 비어 있지 않아야 한다")
    if lane is not None and (not isinstance(lane, str) or lane not in (LANE_ENGINE, LANE_CLEANROOM)):
        raise MrhCallSpecError(f"lane은 None/'engine'/'cleanroom'이어야 한다: {lane!r}")
    if cleanroom_binary_hash is not None and not isinstance(cleanroom_binary_hash, str):
        raise MrhCallSpecError("cleanroom_binary_hash는 str 또는 None이어야 한다")
    ext = bool(rec.ext)
    if ext and lane == LANE_ENGINE:
        raise CleanroomError("ext 입력은 엔진 레인으로 읽을 수 없다")
    lane_used = lane if lane is not None else lane_for(rec)

    # ---- 2. 읽기 해상도·캔버스
    src_hw = (int(img.shape[0]), int(img.shape[1]))
    lin = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
    long_side = max(src_hw)
    if long_side > cfg.read_long_max:
        s = cfg.read_long_max / long_side
        read_hw = (max(1, int(round(src_hw[0] * s))), max(1, int(round(src_hw[1] * s))))
        lin = F.interpolate(lin, size=read_hw, mode="area")
    else:
        read_hw = src_hw
    arr = lin[0].permute(1, 2, 0).numpy()
    hc, wc = UNIT * -(-read_hw[0] // UNIT), UNIT * -(-read_hw[1] // UNIT)
    arr = np.pad(arr, ((0, hc - read_hw[0]), (0, wc - read_hw[1]), (0, 0)), mode="edge")
    disp = 2.0 * color.linear_to_display(arr).astype(np.float32) - 1.0
    x = torch.from_numpy(np.ascontiguousarray(disp.transpose(2, 0, 1)))[None].float()
    canvas_hw = (hc, wc)
    h4, w4 = hc // STRIDE, wc // STRIDE

    # ---- 3. 호출 계획과 호출
    gsz = global_size(hc, wc, cfg.global_pixels)
    xg = x if gsz == canvas_hw else F.interpolate(x, size=gsz, mode="area")
    snapped = []
    for s in cfg.global_sigmas:
        v = float(reader.snap(s))
        if v not in snapped:
            snapped.append(v)
    degenerate = len(cfg.global_sigmas) == 2 and len(snapped) < 2
    boxes = plan_tiles(hc, wc, cfg.tile_px) if cfg.tiles else ()
    tile_sigma = float(reader.snap(cfg.tile_sigma)) if boxes else None

    digests = []

    def run(image, sigma, view_key, hw4):
        view = reader(image, sigma, prov=rec, key=view_key)
        if ext and not bool(getattr(view, "ephemeral", False)):
            raise CleanroomError(f"{view_key}: ext 입력의 탭이 임시가 아니다")
        out = _check_view(view, cfg.heads, hw4, view_key)
        digests.append(bytes(view.digest))
        return out

    g_reads = [run(xg, s, f"{key}/g{i}", (gsz[0] // STRIDE, gsz[1] // STRIDE)) for i, s in enumerate(snapped)]
    t_reads = []
    for j, b in enumerate(boxes):
        crop = x[:, :, b.y0:b.y1, b.x0:b.x1]
        t_reads.append((b, run(crop, tile_sigma, f"{key}/t{j}",
                               ((b.y1 - b.y0) // STRIDE, (b.x1 - b.x0) // STRIDE))))

    # ---- 4. 융합·혼합
    fused = fuse_global(g_reads, floor=cfg.depth_floor)
    glob = {}
    for k, v in fused.outputs.items():
        v = _resize_grid(v, (h4, w4))
        glob[k] = _unit(v) if k in _UNIT_KEYS else v
    sigma_map = _resize_grid(fused.sigma_map[None, None], (h4, w4))[0, 0]
    blend = blend_reads(glob, t_reads, canvas_hw=canvas_hw, feather_px=cfg.feather_px, floor=cfg.depth_floor)
    ch4, cw4 = -(-read_hw[0] // STRIDE), -(-read_hw[1] // STRIDE)

    def crop4(t):
        return t[..., :ch4, :cw4].contiguous()

    outputs = {k: crop4(v) for k, v in blend.outputs.items()}

    # ---- 5. r_dis
    r_dis_map = crop4(torch.maximum(blend.tile_map, sigma_map))
    uncertain = r_dis_map > cfg.unc_tau
    r_dis = max(blend.r_dis_tiles, fused.r_dis_sigma)

    # ---- 6. 클린룸 결과 prov
    out_prov = P.derive("mrh_read", [rec])
    return MrhReadout(
        outputs=outputs, depth_global=crop4(glob["log_depth"]), coverage=crop4(blend.coverage),
        r_dis_map=r_dis_map, uncertain=uncertain, r_dis=float(r_dis), r_dis_tiles=float(blend.r_dis_tiles),
        r_dis_sigma=float(fused.r_dis_sigma), uncertain_frac=float(uncertain.float().mean()),
        tiles=tuple(boxes), tile_affine=tuple(blend.affine), global_size=tuple(gsz),
        global_sigmas=tuple(snapped), tile_sigma=tile_sigma, sigma_pair_degenerate=bool(degenerate),
        src_hw=src_hw, read_hw=tuple(read_hw), canvas_hw=canvas_hw, n_forward=len(digests), lane=lane_used,
        ext=ext, ns=out_prov.ns, prov=out_prov, cleanroom_binary_hash=cleanroom_binary_hash, key=key,
        digests=tuple(digests))
