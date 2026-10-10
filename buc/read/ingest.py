"""RD-01 R0 수납: 파일(또는 배열) 한 장을 S-1 판독 입력으로 만든다.

디코드 -> EXIF 방향 -> 색공간 결정 -> 레터박스 -> InputImage -> 장면 선형(Rec.709) -> 실효 해상도 r_eff.
EXIF / C2PA / JPEG 품질은 기록만 하며 분기 조건이 아니다 (taint 도 바꾸지 않는다).

numpy · Pillow · FND-02(contracts) · FND-06(color) 만 쓴다. torch 금지, import 부작용 없음, 순수·결정론, 입력 불변.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from buc.core import color as _color
from buc.core import contracts as _contracts

__all__ = [
    "IngestError", "UnsupportedFormatError", "IngestOpts", "ICCClass", "JpegQuality", "C2paInfo", "REffEstimate",
    "ColorInfo", "IngestResult", "ingest", "sniff_format", "classify_icc", "to_linear_709", "detect_letterbox",
    "estimate_r_eff", "jpeg_quant_tables", "estimate_jpeg_quality", "detect_c2pa", "read_exif", "fov_x_prior_deg",
]


# ---------------------------------------------------------------------------------------------------------------
# 오류 / 옵션 / 결과 타입
# ---------------------------------------------------------------------------------------------------------------
class IngestError(ValueError):
    """수납할 수 없는 입력 (손상, 잘림, 빈 입력, 반투명, CMYK, 픽셀 예산 초과 ...)."""


class UnsupportedFormatError(IngestError):
    """HEIC · EXR · GIF · 미상 형식."""


@dataclass(frozen=True, kw_only=True)
class IngestOpts:
    letterbox: bool = True
    bar_mean: float = 2.0 / 255.0
    bar_std: float = 2.0 / 255.0
    min_bar_px: int = 2
    min_keep_frac: float = 0.5
    apply_orientation: bool = True
    max_pixels: int = 1 << 27

    def __post_init__(self):
        def num(name, v, lo=None, hi=None, lo_open=False):
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v):
                raise ValueError(f"{name}: 유한한 수여야 한다 ({v!r})")
            if lo is not None and (v <= lo if lo_open else v < lo):
                raise ValueError(f"{name}={v!r}: 범위 밖")
            if hi is not None and v > hi:
                raise ValueError(f"{name}={v!r}: 범위 밖")
        if not isinstance(self.letterbox, (bool, np.bool_)):
            raise ValueError("letterbox: bool 이어야 한다")
        if not isinstance(self.apply_orientation, (bool, np.bool_)):
            raise ValueError("apply_orientation: bool 이어야 한다")
        num("bar_mean", self.bar_mean, 0.0)
        num("bar_std", self.bar_std, 0.0)
        num("min_bar_px", self.min_bar_px, 1)
        num("min_keep_frac", self.min_keep_frac, 0.0, 1.0, lo_open=True)
        num("max_pixels", self.max_pixels, 1)


@dataclass(frozen=True, kw_only=True)
class ICCClass:
    space: str
    valid: bool
    description: Optional[str]
    reason: str


@dataclass(frozen=True, kw_only=True)
class JpegQuality:
    quality: float
    exact: bool
    fit_err: float


@dataclass(frozen=True, kw_only=True)
class C2paInfo:
    present: bool
    container: Optional[str]
    manifest_bytes: int
    ai_claim: bool


@dataclass(frozen=True, kw_only=True)
class REffEstimate:
    r_in: int
    r_eff: float
    ratio: float
    reliable: bool


@dataclass(frozen=True, kw_only=True)
class ColorInfo:
    space: str
    source: str
    icc_sha256: Optional[str]


@dataclass(frozen=True, kw_only=True, eq=False)
class IngestResult:
    image: Any
    image_lin: Any
    crop_box: Tuple[int, int, int, int]
    src_size: Tuple[int, int]
    bars: Tuple[int, int, int, int]
    r_in: int
    r_eff: float
    color: ColorInfo
    jpeg: Optional[JpegQuality]
    c2pa: C2paInfo
    exif: Dict[str, Any]
    fov_x_prior_deg: Optional[float]
    fmt: str
    orientation: int
    orientation_applied: bool
    gamut_clip_frac: float
    warnings: Tuple[str, ...]
    raw_sha256: str
    consent: bool = True

    def meta_input(self) -> dict:
        """JSON 가능한 ``meta.input`` (설계 §2.13)."""
        p = self.image.prov
        j = self.jpeg
        c = self.c2pa
        return {
            "h_I": self.image.h_I,
            "raw_sha256": self.raw_sha256,
            "format": self.fmt,
            "r_in": int(self.r_in),
            "r_eff": float(self.r_eff),
            "color_space": self.color.space,
            "color_source": self.color.source,
            "icc_sha256": self.color.icc_sha256,
            "crop_box": [int(v) for v in self.crop_box],
            "src_size": [int(v) for v in self.src_size],
            "orientation": int(self.orientation),
            "orientation_applied": bool(self.orientation_applied),
            "exif": dict(self.exif),
            "fov_x_prior_deg": None if self.fov_x_prior_deg is None else float(self.fov_x_prior_deg),
            "c2pa": {"present": bool(c.present), "container": c.container, "manifest_bytes": int(c.manifest_bytes),
                     "ai_claim": bool(c.ai_claim)},
            "jpeg": None if j is None else {"quality": float(j.quality), "exact": bool(j.exact), "fit_err": float(j.fit_err)},
            "consent": bool(self.consent),
            "gamut_clip_frac": float(self.gamut_clip_frac),
            "warnings": list(self.warnings),
            "prov": {"taint": sorted(p.taint), "mode": p.mode, "roots": list(p.roots), "ns": p.ns},
            "ext": bool(p.ext),
            "comp": bool(p.comp),
            "hold": bool(p.hold),
        }


# ---------------------------------------------------------------------------------------------------------------
# 바이트 도우미 / 형식 판별
# ---------------------------------------------------------------------------------------------------------------
def _bytes_of(data, name="data") -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"{name}: bytes 류여야 한다 ({type(data).__name__})")


def sniff_format(data) -> str:
    """'png' | 'jpeg' | 'webp' | 'heic' | 'exr' | 'unknown' (앞부분 바이트로 판별)."""
    b = _bytes_of(data)[:16]
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[4:8] == b"ftyp" and b[8:12] in (b"heic", b"heix", b"mif1"):
        return "heic"
    if b[:4] == b"\x76\x2f\x31\x01":
        return "exr"
    return "unknown"


# ---------------------------------------------------------------------------------------------------------------
# ICC 분류
# ---------------------------------------------------------------------------------------------------------------
_D50_XYZ = np.array([0.9642, 1.0, 0.8249])
_BFD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
_GRID17 = np.linspace(0.0, 1.0, 17)
_ADOBE_GAMMA = 563.0 / 256.0
_PRIM_SPACES = ("srgb", "display-p3", "adobe-rgb", "rec2020")


def _d50_colorants(space: str) -> np.ndarray:
    m = _color.rgb_to_xyz_matrix(space)
    s = _BFD @ _color.WHITE_XYZ
    d = _BFD @ _D50_XYZ
    return np.linalg.inv(_BFD) @ np.diag(d / s) @ _BFD @ m


_REF_COLORANTS = {k: _d50_colorants(k) for k in _PRIM_SPACES}


class _IccBad(Exception):
    pass


def _icc_tags(p: bytes) -> Dict[bytes, Tuple[int, int]]:
    if len(p) < 132:
        raise _IccBad("too-short")
    size = struct.unpack(">I", p[0:4])[0]
    if size > len(p) or size < 132:
        raise _IccBad("size-mismatch")
    if p[36:40] != b"acsp":
        raise _IccBad("no-acsp-signature")
    n = struct.unpack(">I", p[128:132])[0]
    if n > 4096 or 132 + 12 * n > size:
        raise _IccBad("bad-tag-table")
    tags: Dict[bytes, Tuple[int, int]] = {}
    for i in range(n):
        sig, off, sz = struct.unpack(">4sII", p[132 + 12 * i:144 + 12 * i])
        if off + sz > size or sz < 8:
            raise _IccBad("tag-out-of-range")
        tags.setdefault(sig, (off, sz))
    return tags


def _icc_description(p: bytes, tag) -> Optional[str]:
    if tag is None:
        return None
    off, sz = tag
    t = p[off:off + sz]
    try:
        if t[:4] == b"desc":
            n = struct.unpack(">I", t[8:12])[0]
            return t[12:12 + n].split(b"\0")[0].decode("latin-1")
        if t[:4] == b"mluc":
            cnt, rec = struct.unpack(">II", t[8:16])
            if cnt < 1 or rec < 12:
                return None
            _, _, ln, o = struct.unpack(">4s4sII"[0:0] or ">2s2sII", t[16:28])
            return t[o:o + ln].decode("utf-16-be").split("\0")[0]
    except Exception:
        return None
    return None


def _icc_xyz(p: bytes, tag) -> np.ndarray:
    off, sz = tag
    t = p[off:off + sz]
    if t[:4] != b"XYZ " or len(t) < 20:
        raise _IccBad("bad-xyz-tag")
    return np.array(struct.unpack(">3i", t[8:20]), dtype=np.float64) / 65536.0


def _icc_trc(p: bytes, tag) -> np.ndarray:
    off, sz = tag
    t = p[off:off + sz]
    kind = t[:4]
    x = _GRID17
    if kind == b"curv":
        n = struct.unpack(">I", t[8:12])[0]
        if n == 0:
            return x.copy()
        if n == 1:
            return x ** (struct.unpack(">H", t[12:14])[0] / 256.0)
        if 12 + 2 * n > len(t):
            raise _IccBad("bad-curv")
        tab = np.frombuffer(t[12:12 + 2 * n], dtype=">u2").astype(np.float64) / 65535.0
        return np.interp(x, np.linspace(0.0, 1.0, n), tab)
    if kind == b"para":
        ft = struct.unpack(">H", t[8:10])[0]
        counts = (1, 3, 4, 5, 7)
        if ft > 4 or 12 + 4 * counts[ft] > len(t):
            raise _IccBad("bad-para")
        v = np.array(struct.unpack(f">{counts[ft]}i", t[12:12 + 4 * counts[ft]]), dtype=np.float64) / 65536.0
        g = v[0]
        with np.errstate(all="ignore"):
            if ft == 0:
                y = x ** g
            else:
                a, b = v[1], v[2]
                base = np.maximum(a * x + b, 0.0) ** g
                if ft in (1, 2):
                    thr = -b / a if a != 0 else 0.0
                    c = v[3] if ft == 2 else 0.0
                    y = np.where(x >= thr, base + c, c)
                elif ft == 3:
                    c, d = v[3], v[4]
                    y = np.where(x >= d, base, c * x)
                else:
                    c, d, e, f = v[3], v[4], v[5], v[6]
                    y = np.where(x >= d, base + e, c * x + f)
        if not np.isfinite(y).all():
            raise _IccBad("bad-para")
        return y
    raise _IccBad("unknown-trc-type")


def _close(a, b, tol) -> bool:
    return float(np.max(np.abs(a - b))) <= tol


def classify_icc(profile) -> ICCClass:
    """행렬/TRC 프로파일을 sRGB · Display P3 · Adobe RGB · Rec.2020 · linear-srgb 로 분류한다. 깨진 입력은 valid=False."""
    if not isinstance(profile, (bytes, bytearray, memoryview)):
        raise TypeError("profile: bytes 여야 한다")
    p = bytes(profile)
    try:
        tags = _icc_tags(p)
    except (_IccBad, struct.error):
        return ICCClass(space="unknown", valid=False, description=None, reason="invalid-header")
    except Exception:
        return ICCClass(space="unknown", valid=False, description=None, reason="invalid")
    desc = _icc_description(p, tags.get(b"desc") or tags.get(b"dscm"))

    def unknown(reason):
        return ICCClass(space="unknown", valid=True, description=desc, reason=reason)

    try:
        if p[16:20] != b"RGB ":
            return unknown("not-rgb-data-space")
        need = (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC")
        if any(k not in tags for k in need):
            return unknown("not-matrix-trc")
        col = np.stack([_icc_xyz(p, tags[k]) for k in need[:3]], axis=1)
        trcs = [_icc_trc(p, tags[k]) for k in need[3:]]
    except (_IccBad, struct.error, ValueError):
        return unknown("unparsable-matrix-trc")
    errs = {k: float(np.max(np.abs(col - v))) for k, v in _REF_COLORANTS.items()}
    prim = min(errs, key=errs.get)
    if errs[prim] > 0.01:
        return unknown("primaries-unmatched")
    if max(float(np.max(np.abs(trcs[0] - t))) for t in trcs[1:]) > 0.01:
        return unknown("trc-channel-mismatch")
    t = trcs[0]
    srgb_c = _color.srgb_to_linear(_GRID17)
    space = None
    if prim in ("srgb", "display-p3"):
        if _close(t, srgb_c, 0.02) or _close(t, _GRID17 ** 2.2, 0.02):
            space = prim
        elif prim == "srgb" and _close(t, _GRID17, 0.01):
            space = "linear-srgb"
    elif prim == "adobe-rgb":
        if _close(t, _GRID17 ** _ADOBE_GAMMA, 0.02):
            space = "adobe-rgb"
    else:
        if _close(t, _color.eotf(_GRID17, "rec2020"), 0.03) or _close(t, _GRID17 ** 2.4, 0.03):
            space = "rec2020"
    if space is None:
        return unknown("trc-unmatched")
    return ICCClass(space=space, valid=True, description=desc, reason="matched")


# ---------------------------------------------------------------------------------------------------------------
# 색 변환
# ---------------------------------------------------------------------------------------------------------------
def to_linear_709(pixels, color_space):
    """부호화 영상 -> Rec.709 장면 선형 float32 (>=0), 음수(색역 밖) 화소는 휘도 보존 탈채도. (lin, clip_frac) 반환."""
    raw = _color.decode_to_linear(pixels, color_space)          # float32; 타입·색공간 검증 포함
    lin = np.ascontiguousarray(raw)
    if lin.size == 0:
        return lin, 0.0
    m = lin.min(axis=-1)
    neg = m < 0.0
    if not neg.any():
        return lin, 0.0
    frac = float(np.mean(m < -1e-6))
    c = lin[neg].astype(np.float64)
    mm = m[neg].astype(np.float64)
    y = np.maximum(_color.luminance(c), 0.0)
    t = y / (y - mm)
    out = y[:, None] + t[:, None] * (c - y[:, None])
    lin[neg] = np.maximum(out, 0.0).astype(np.float32)
    return lin, frac


# ---------------------------------------------------------------------------------------------------------------
# 레터박스
# ---------------------------------------------------------------------------------------------------------------
def _unit_scale(img) -> float:
    if img.dtype == np.uint8:
        return 1.0 / 255.0
    if img.dtype == np.uint16:
        return 1.0 / 65535.0
    return 1.0


def detect_letterbox(img, *, opts: Optional[IngestOpts] = None):
    """검은 띠를 뺀 내용 상자 (x0, y0, x1, y1), x1·y1 배타."""
    o = IngestOpts() if opts is None else opts
    a = np.asarray(img)
    if a.dtype not in (np.uint8, np.uint16, np.float32):
        raise TypeError(f"img: uint8/uint16/float32 여야 한다 (dtype {a.dtype})")
    if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"img: (H,W,3) 이어야 한다 (shape {a.shape})")
    h, w = a.shape[:2]
    s = _unit_scale(a)

    def is_bar(line) -> bool:
        v = line.astype(np.float64) * s
        return bool(v.mean() <= o.bar_mean and v.std() <= o.bar_std)

    def run(n, getter):
        k = 0
        while k < n and is_bar(getter(k)):
            k += 1
        return k

    full = (0, 0, w, h)
    top = run(h, lambda i: a[i])
    if top >= h:                                                   # 전부 검다
        return full
    bottom = run(h - top, lambda i: a[h - 1 - i])
    top = top if top >= o.min_bar_px else 0
    bottom = bottom if bottom >= o.min_bar_px else 0
    y0, y1 = top, h - bottom
    sub = a[y0:y1]
    left = run(w, lambda j: sub[:, j])
    if left >= w:
        return full
    right = run(w - left, lambda j: sub[:, w - 1 - j])
    left = left if left >= o.min_bar_px else 0
    right = right if right >= o.min_bar_px else 0
    x0, x1 = left, w - right
    if (y1 - y0) < o.min_keep_frac * h or (x1 - x0) < o.min_keep_frac * w:
        return full
    return (int(x0), int(y0), int(x1), int(y1))


# ---------------------------------------------------------------------------------------------------------------
# 실효 해상도 r_eff
# ---------------------------------------------------------------------------------------------------------------
_REFF_MIN_SIDE = 64
_REFF_CUT = 0.25                    # 모델 대비 전력비 (25 %)


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _unreliable(r_in: int) -> REffEstimate:
    return REffEstimate(r_in=int(r_in), r_eff=float(r_in), ratio=1.0, reliable=False)


def estimate_r_eff(y) -> REffEstimate:
    """부호화 휘도 y (H,W) 의 방사 PSD 를 거듭제곱으로 맞추고, 모델의 25 % 밑으로 지속 하강하는 첫 ν_c 로 r_eff 를 낸다."""
    a = np.asarray(y)
    if a.dtype.kind != "f":
        raise TypeError(f"y: 실수 배열이어야 한다 (dtype {a.dtype})")
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"y: 비어 있지 않은 2차원이어야 한다 (shape {a.shape})")
    h, w = a.shape
    r_in = min(h, w)
    if r_in < _REFF_MIN_SIDE or not np.isfinite(a).all():
        return _unreliable(r_in)
    x = a.astype(np.float64)
    x = x - x.mean()
    if float(x.std()) < 1e-8:
        return _unreliable(r_in)
    spec = np.fft.rfft2(x * np.outer(_hann(h), _hann(w)))
    p = spec.real ** 2 + spec.imag ** 2
    nu = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :])
    sel = (nu > 0) & (nu <= 0.5)
    nu_s, p_s = nu[sel], p[sel]
    dnu = 2.0 / r_in
    nb = int(round(0.5 / dnu))
    idx = np.minimum((nu_s / dnu).astype(np.int64), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    ok = cnt > 0
    pb = np.bincount(idx, weights=p_s, minlength=nb)[ok] / cnt[ok]
    nb_nu = np.bincount(idx, weights=nu_s, minlength=nb)[ok] / cnt[ok]
    if pb.size < 12 or not (pb > 0).all():
        return _unreliable(r_in)
    lo = max(0.01, 3.0 / r_in)
    hi = max(0.04, 2.0 * lo)
    band = (nb_nu >= lo) & (nb_nu <= hi)
    if band.sum() < 3:
        return _unreliable(r_in)
    coef = np.polyfit(np.log(nb_nu[band]), np.log(pb[band]), 1)
    model = np.exp(coef[1] + coef[0] * np.log(nb_nu))
    r = pb / model
    win = max(3, int(round(0.016 / dnu)) | 1)
    pad = win // 2
    rp = np.pad(r, pad, mode="edge")
    cs = np.concatenate([[0.0], np.cumsum(rp)])
    rs = (cs[win:] - cs[:-win]) / win
    span = max(win, int(round(0.04 / dnu)))
    nu_c = 0.5
    start = int(np.searchsorted(nb_nu, hi, side="right"))
    for k in range(max(start, 1), r.size):
        if rs[k] < _REFF_CUT and float(np.median(r[k:k + span])) < _REFF_CUT:
            r0, r1 = rs[k - 1], rs[k]
            f = 0.0 if r0 <= r1 else min(1.0, max(0.0, (r0 - _REFF_CUT) / (r0 - r1)))
            nu_c = float(nb_nu[k - 1] + f * (nb_nu[k] - nb_nu[k - 1]))
            break
    ratio = float(min(1.0, max(1.0 / 16.0, nu_c / 0.5)))
    return REffEstimate(r_in=int(r_in), r_eff=float(ratio * r_in), ratio=ratio, reliable=True)


# ---------------------------------------------------------------------------------------------------------------
# JPEG 양자화표 / 품질
# ---------------------------------------------------------------------------------------------------------------
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100,
    103, 99], dtype=np.int64)
_STD_CHROMA = np.array([17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99,
                        99, 99, 99, 99, 99] + [99] * 32, dtype=np.int64)
_ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def _ijg_table(std: np.ndarray, q: int) -> np.ndarray:
    scale = 5000 // q if q < 50 else 200 - q * 2
    return np.clip((std * scale + 50) // 100, 1, 255).astype(np.int64)


_IJG = {q: (_ijg_table(_STD_LUMA, q), _ijg_table(_STD_CHROMA, q)) for q in range(1, 101)}


def _jpeg_segments(data: bytes):
    """(marker, payload_start, payload_end) 를 SOS 직전까지 낸다. 잘리면 IngestError."""
    n = len(data)
    if data[:2] != b"\xff\xd8":
        raise IngestError("JPEG 이 아니다 (SOI 없음)")
    pos = 2
    while pos < n:
        if data[pos] != 0xFF:
            raise IngestError("JPEG 마커 구조가 깨졌다")
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        mk = data[pos]
        pos += 1
        if mk == 0xD9 or mk == 0xDA:
            return
        if mk == 0x01 or 0xD0 <= mk <= 0xD8 or mk == 0x00:
            continue
        if pos + 2 > n:
            raise IngestError("JPEG 세그먼트가 잘렸다")
        ln = struct.unpack(">H", data[pos:pos + 2])[0]
        if ln < 2 or pos + ln > n:
            raise IngestError("JPEG 세그먼트가 잘렸다")
        yield mk, pos + 2, pos + ln
        pos += ln


def jpeg_quant_tables(data) -> Dict[int, np.ndarray]:
    """DQT 를 직접 파싱해 {id: int64 (64,)} (자연 행 우선 순서) 를 낸다."""
    d = _bytes_of(data)
    tabs: Dict[int, np.ndarray] = {}
    for mk, s, e in _jpeg_segments(d):
        if mk != 0xDB:
            continue
        p = s
        while p < e:
            pq, tq = d[p] >> 4, d[p] & 0x0F
            if pq > 1 or tq > 3:
                raise IngestError("DQT 형식 오류")
            nb = 64 * (pq + 1)
            if p + 1 + nb > e:
                raise IngestError("DQT 가 잘렸다")
            raw = d[p + 1:p + 1 + nb]
            zz = np.frombuffer(raw, dtype=">u2").astype(np.int64) if pq else np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
            nat = np.empty(64, dtype=np.int64)
            nat[_ZIGZAG] = zz
            tabs[tq] = nat
            p += 1 + nb
    if not tabs:
        raise IngestError("DQT 가 없다")
    return {k: tabs[k] for k in sorted(tabs)}


def estimate_jpeg_quality(tables) -> JpegQuality:
    """IJG q=1..100 표와의 최대절댓값 오차가 최소인 q (동률은 낮은 q)."""
    if not isinstance(tables, dict) or 0 not in tables:
        raise IngestError("휘도 양자화표(0)가 없다")
    t0 = np.asarray(tables[0], dtype=np.int64)
    t1 = np.asarray(tables[1], dtype=np.int64) if 1 in tables else None
    if t0.shape != (64,) or (t1 is not None and t1.shape != (64,)):
        raise IngestError("양자화표 모양 오류")
    best_q, best_e = 1, None
    for q in range(1, 101):
        l, c = _IJG[q]
        e = int(np.abs(t0 - l).max())
        if t1 is not None:
            e = max(e, int(np.abs(t1 - c).max()))
        if best_e is None or e < best_e:
            best_q, best_e = q, e
    return JpegQuality(quality=float(best_q), exact=best_e == 0, fit_err=float(best_e))


# ---------------------------------------------------------------------------------------------------------------
# C2PA (기록 전용)
# ---------------------------------------------------------------------------------------------------------------
_C2PA_UUID = bytes.fromhex("63327061001100108000" "00AA00389B71")


def _is_c2pa_jumbf(box: bytes) -> bool:
    """box: JUMBF 슈퍼박스 바이트. 'jumb' 안의 첫 'jumd' 가 c2pa 형식이면 참."""
    if len(box) < 16 or box[4:8] != b"jumb":
        return False
    if box[12:16] != b"jumd":
        return False
    jd = box[16:]
    if len(jd) < 17:
        return False
    label = jd[17:].split(b"\0")[0]
    return label == b"c2pa" or jd[:16] == _C2PA_UUID


def _c2pa_result(manifest: bytes, container: str) -> C2paInfo:
    return C2paInfo(present=True, container=container, manifest_bytes=len(manifest),
                    ai_claim=b"trainedAlgorithmicMedia" in manifest)


_NO_C2PA = C2paInfo(present=False, container=None, manifest_bytes=0, ai_claim=False)


def detect_c2pa(data) -> C2paInfo:
    """C2PA 매니페스트 존재 기록 (JPEG APP11 / PNG caBX / WebP C2PA). 예외를 내지 않는다."""
    try:
        d = _bytes_of(data)
        fmt = sniff_format(d)
        if fmt == "jpeg":
            parts = []
            for mk, s, e in _jpeg_segments(d):
                if mk != 0xEB or d[s:s + 2] != b"JP" or e - s < 8:
                    continue
                z = struct.unpack(">I", d[s + 4:s + 8])[0]
                body = d[s + 8:e]
                if not parts:
                    if _is_c2pa_jumbf(body):
                        parts.append(body)
                elif z > 1:
                    parts.append(body[8:])
            if parts:
                return _c2pa_result(b"".join(parts), "jpeg-app11")
        elif fmt == "png":
            pos = 8
            while pos + 8 <= len(d):
                n = struct.unpack(">I", d[pos:pos + 4])[0]
                typ = d[pos + 4:pos + 8]
                if pos + 12 + n > len(d):
                    break
                if typ == b"caBX" and _is_c2pa_jumbf(d[pos + 8:pos + 8 + n]):
                    return _c2pa_result(d[pos + 8:pos + 8 + n], "png-cabx")
                if typ == b"IEND":
                    break
                pos += 12 + n
        elif fmt == "webp":
            pos = 12
            while pos + 8 <= len(d):
                typ = d[pos:pos + 4]
                n = struct.unpack("<I", d[pos + 4:pos + 8])[0]
                if pos + 8 + n > len(d):
                    break
                if typ == b"C2PA" and _is_c2pa_jumbf(d[pos + 8:pos + 8 + n]):
                    return _c2pa_result(d[pos + 8:pos + 8 + n], "webp-c2pa")
                pos += 8 + n + (n & 1)
    except Exception:
        pass
    return _NO_C2PA


# ---------------------------------------------------------------------------------------------------------------
# EXIF (화이트리스트)
# ---------------------------------------------------------------------------------------------------------------
def _pos_float(v) -> Optional[float]:
    try:
        if isinstance(v, (tuple, list)):
            v = v[0]
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) and f > 0 else None


def _text(v) -> Optional[str]:
    if isinstance(v, (bytes, bytearray)):
        try:
            v = bytes(v).decode("ascii", "ignore")
        except Exception:
            return None
    if not isinstance(v, str):
        return None
    s = v.replace("\0", "").strip()
    return s[:64] if s else None


def _exif_from_image(im) -> dict:
    out: dict = {}
    try:
        ex = im.getexif()
        if not ex:
            return out
        for key, tag in (("make", 0x010F), ("model", 0x0110)):
            t = _text(ex.get(tag))
            if t is not None:
                out[key] = t
        o = ex.get(0x0112)
        if isinstance(o, (int, np.integer)) and not isinstance(o, bool) and 1 <= int(o) <= 8:
            out["orientation"] = int(o)
        try:
            sub = ex.get_ifd(0x8769)
        except Exception:
            sub = {}
        for key, tag in (("focal_mm", 0x920A), ("focal_35mm", 0xA405), ("f_number", 0x829D), ("exposure_s", 0x829A)):
            f = _pos_float(sub.get(tag))
            if f is not None:
                out[key] = f
        iso = sub.get(0x8827)
        if isinstance(iso, (tuple, list)) and iso:
            iso = iso[0]
        f = _pos_float(iso)
        if f is not None and float(int(f)) == f:
            out["iso"] = int(f)
    except Exception:
        return {}
    return out


def read_exif(data) -> dict:
    """화이트리스트 EXIF (make, model, orientation, focal_mm, focal_35mm, f_number, exposure_s, iso). 깨지면 {}."""
    d = _bytes_of(data)
    try:
        with Image.open(io.BytesIO(d)) as im:
            return _exif_from_image(im)
    except Exception:
        return {}


def fov_x_prior_deg(f35, width, height) -> float:
    """35 mm 환산 초점거리와 (방향 보정 후) 픽셀 크기로부터 가로 FOV 사전값(도). 대각선 환산."""
    for name, v in (("f35", f35), ("width", width), ("height", height)):
        try:
            ok = math.isfinite(float(v)) and float(v) > 0
        except Exception:
            ok = False
        if not ok:
            raise ValueError(f"{name}={v!r}: 양의 유한한 수여야 한다")
    f35, width, height = float(f35), float(width), float(height)
    sensor_w = 0.5 * math.hypot(36.0, 24.0) * width / math.hypot(width, height)
    return math.degrees(2.0 * math.atan(sensor_w / f35))


# ---------------------------------------------------------------------------------------------------------------
# 방향
# ---------------------------------------------------------------------------------------------------------------
def _orient(a: np.ndarray, o: int) -> np.ndarray:
    if o == 1:
        return a
    r = {2: lambda: a[:, ::-1], 3: lambda: a[::-1, ::-1], 4: lambda: a[::-1], 5: lambda: a.transpose(1, 0, 2),
         6: lambda: np.rot90(a, -1), 7: lambda: a.transpose(1, 0, 2)[::-1, ::-1], 8: lambda: np.rot90(a, 1)}[o]()
    return np.ascontiguousarray(r)


# ---------------------------------------------------------------------------------------------------------------
# 디코드
# ---------------------------------------------------------------------------------------------------------------
def _decode(data: bytes, opts: IngestOpts):
    """(pixels uint8|uint16 (H,W,3), exif dict, icc bytes|None, png_srgb bool)"""
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        if w < 1 or h < 1:
            raise IngestError("빈 영상")
        if w * h > opts.max_pixels:
            raise IngestError(f"픽셀 수 {w * h} 가 max_pixels={opts.max_pixels} 를 넘는다")
        im.load()
    except IngestError:
        raise
    except Exception as e:                                       # 잘림 / 손상 / 해석 불가
        raise IngestError(f"디코드 실패: {type(e).__name__}: {e}") from e
    try:
        exif = _exif_from_image(im)
        icc = im.info.get("icc_profile")
        icc = bytes(icc) if isinstance(icc, (bytes, bytearray)) and len(icc) else None
        png_srgb = im.info.get("srgb") is not None
        mode = im.mode
        if mode == "CMYK" or mode == "YCbCr" or mode == "LAB" or mode == "HSV":
            raise IngestError(f"색 모드 {mode} 는 수납하지 않는다")
        if mode in ("I;16", "I;16L", "I;16B", "I;16N"):
            g = np.asarray(im).astype(np.uint16)
            pix = np.stack([g, g, g], axis=-1)
        elif mode in ("I", "F"):
            raise IngestError(f"색 모드 {mode} 는 수납하지 않는다")
        else:
            if mode == "P":
                im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            elif mode in ("PA", "La", "RGBa"):
                im = im.convert("RGBA" if mode != "La" else "LA")
            mode = im.mode
            if mode in ("RGBA", "LA"):
                al = np.asarray(im.getchannel("A"))
                if not (al == 255).all():
                    raise IngestError("반투명 알파 채널은 수납하지 않는다")
                im = im.convert("RGB")
            elif mode != "RGB":
                im = im.convert("RGB")
            pix = np.asarray(im)
        pix = np.ascontiguousarray(pix)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"디코드 실패: {type(e).__name__}: {e}") from e
    return pix, exif, icc, png_srgb


# ---------------------------------------------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------------------------------------------
def _default_prov():
    return _contracts.Prov(taint=frozenset({"USER"}), mode="engine", ns="user_vault")


def ingest(source, *, color_space: Optional[str] = None, prov=None, consent: bool = True,
           opts: Optional[IngestOpts] = None) -> IngestResult:
    """파일(bytes / 경로) 또는 (H,W,3) 배열 한 장을 S-1 입력으로 만든다."""
    if opts is None:
        opts = IngestOpts()
    elif not isinstance(opts, IngestOpts):
        raise TypeError("opts: IngestOpts 여야 한다")
    if prov is None:
        prov = _default_prov()
    elif not isinstance(prov, _contracts.Prov):
        raise TypeError("prov: Prov 여야 한다")
    if not isinstance(consent, (bool, np.bool_)):
        raise TypeError("consent: bool 이어야 한다")
    if color_space is not None and (not isinstance(color_space, str) or color_space not in _color.COLOR_SPACES):
        raise ValueError(f"color_space {color_space!r}: {_color.COLOR_SPACES} 중 하나여야 한다")
    consent = bool(consent)

    warnings = set()
    icc_bytes = None
    png_srgb = False
    c2pa = _NO_C2PA
    jpeg: Optional[JpegQuality] = None

    if isinstance(source, np.ndarray):
        a = source
        if a.dtype not in (np.uint8, np.uint16, np.float32):
            raise TypeError(f"ndarray dtype {a.dtype}: uint8/uint16/float32 만 허용")
        if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] < 1 or a.shape[1] < 1:
            raise IngestError(f"ndarray shape {a.shape}: (H,W,3) 이어야 한다")
        if a.shape[0] * a.shape[1] > opts.max_pixels:
            raise IngestError("픽셀 수가 max_pixels 를 넘는다")
        if a.dtype == np.float32 and not np.isfinite(a).all():
            raise IngestError("비유한 값(NaN/inf)이 있다")
        pix = np.array(a, order="C", copy=True)
        raw_sha = hashlib.sha256(pix.tobytes()).hexdigest()
        fmt = "ndarray"
        exif: dict = {}
    else:
        if isinstance(source, (str, os.PathLike)):
            with open(source, "rb") as fh:
                data = fh.read()
        elif isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            raise TypeError(f"source: bytes · 경로 · ndarray 여야 한다 ({type(source).__name__})")
        fmt = sniff_format(data)
        if fmt not in ("png", "jpeg", "webp"):
            raise UnsupportedFormatError(f"지원하지 않는 형식: {fmt}")
        raw_sha = hashlib.sha256(data).hexdigest()
        pix, exif, icc_bytes, png_srgb = _decode(data, opts)
        c2pa = detect_c2pa(data)
        if fmt == "jpeg":
            try:
                jpeg = estimate_jpeg_quality(jpeg_quant_tables(data))
            except IngestError:
                jpeg = None

    # 방향
    orientation = int(exif.get("orientation", 1))
    applied = bool(opts.apply_orientation and orientation != 1)
    if applied:
        pix = _orient(pix, orientation)
    src_size = (int(pix.shape[0]), int(pix.shape[1]))

    # 색공간 결정: caller > 유효한 ICC > PNG sRGB 청크 > 가정 sRGB
    icc_class = None
    icc_hash = None
    if icc_bytes is not None:
        icc_class = classify_icc(icc_bytes)
        if icc_class.valid:
            icc_hash = hashlib.sha256(icc_bytes).hexdigest()
        else:
            warnings.add("icc_invalid")
    if icc_class is not None and icc_class.valid and icc_class.space == "unknown":
        warnings.add("icc_unrecognized")
    if color_space is not None:
        cspace, csource = color_space, "caller"
        if icc_class is not None and icc_class.valid and icc_class.space not in ("unknown", color_space):
            warnings.add("color_space_conflict")
    elif icc_class is not None and icc_class.valid:
        cspace, csource = icc_class.space, "icc"
    elif png_srgb:
        cspace, csource = "srgb", "png-srgb"
    else:
        cspace, csource = "srgb", "assumed"
        warnings.add("color_space_assumed")

    # 레터박스
    h, w = src_size
    if opts.letterbox:
        x0, y0, x1, y1 = detect_letterbox(pix, opts=opts)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    if (x0, y0, x1, y1) != (0, 0, w, h):
        warnings.add("letterbox_cropped")
        pix = np.ascontiguousarray(pix[y0:y1, x0:x1])
    crop_box = (int(x0), int(y0), int(x1), int(y1))
    bars = (int(y0), int(h - y1), int(x0), int(w - x1))

    image = _contracts.InputImage(pixels=pix, color_space=cspace, prov=prov, exif=dict(exif), consent=consent)
    lin, clip_frac = to_linear_709(image.pixels, cspace)
    if clip_frac > 0.0:
        warnings.add("gamut_clipped")

    # r_eff
    yd = _color.linear_to_display(_color.luminance(lin))
    est = estimate_r_eff(yd)
    if not est.reliable:
        warnings.add("r_eff_unreliable")
    ch, cw = image.pixels.shape[:2]
    if not (512 <= max(ch, cw) <= 8192):
        warnings.add("long_side_out_of_range")

    f35 = exif.get("focal_35mm")
    fov = fov_x_prior_deg(f35, src_size[1], src_size[0]) if f35 is not None else None

    return IngestResult(
        image=image, image_lin=lin, crop_box=crop_box, src_size=src_size, bars=bars, r_in=est.r_in, r_eff=est.r_eff,
        color=ColorInfo(space=cspace, source=csource, icc_sha256=icc_hash), jpeg=jpeg, c2pa=c2pa, exif=dict(exif),
        fov_x_prior_deg=fov, fmt=fmt, orientation=orientation, orientation_applied=applied, gamut_clip_frac=float(clip_frac),
        warnings=tuple(sorted(warnings)), raw_sha256=raw_sha, consent=consent)
