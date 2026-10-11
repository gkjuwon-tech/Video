"""LNS-06: 애너모픽 · 빈티지 처방 (Kolb 근사 출사동 커널 · 스월 · 스퀴즈 · 수평 플레어 · 주변부 번짐).

PARADIGM §7.6 "렌즈 처방 -> 출사동 LUT" 을 광선추적 LUT 대신 닫힌 식 근사로 구현한다(학습 0, 순수 수식).
수차 부품(왜곡 · 횡색수차 · cat-eye · 고스트 · 글레어 · 별)은 LNS-05 `buc.lens.optics` 가 이미 주므로 다시 만들지
않고 그 위에 얹는다. 실제 Kolb LUT 가 오더라도 `exit_pupil_kernel` 시그니처를 유지해 갈아 끼운다.

영상 규약은 LNS-05 와 같다: (H,W,3) float32/float64, 화소 중심이 정수 (x=열, y=행), 주점 c=((W-1)/2,(H-1)/2),
r_frac=|p-c|/R, R=hypot(W-1,H-1)/2. 각도 t 는 방향 (cos t, sin t), 도 단위, y 는 아래.
import 부작용·전역 RNG·RuntimeWarning 없음, torch 를 쓰지 않는다. 입력은 바꾸지 않고 새 배열을 돌려준다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import numpy as np
from scipy import fft as _sfft
from scipy import ndimage as _ndi
from scipy.signal import fftconvolve as _fftconvolve

from buc.core import color as _color
from buc.core import contracts as _C
from buc.lens import optics as _optics
from buc.lens import theta as _theta

__all__ = [
    "SWIRL_MAX_TWIST_DEG", "EDGE_SIGMA_FRAC", "EDGE_R_INNER", "SQUEEZE_MAX", "DEFAULT_STREAK_TINT",
    "Prescription", "pupil_open_fraction", "exit_pupil_kernel", "edge_softness", "streak_layer",
    "apply_prescription_bokeh", "RX_TABLE", "prescription_from_theta", "apply_prescription",
    "apply_prescription_frames",
]

SWIRL_MAX_TWIST_DEG = 60.0
EDGE_SIGMA_FRAC = 0.015
EDGE_R_INNER = 0.35
SQUEEZE_MAX = 3.0
DEFAULT_STREAK_TINT = (0.55, 0.75, 1.0)

_STREAK_MATMUL_MAX_W = 1024  # 이 폭 이하는 행렬곱(정확·비음수), 넘으면 FFT


# ---------------------------------------------------------------------------------------------------------------
# 검증 도우미
# ---------------------------------------------------------------------------------------------------------------
def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _num(name, x, lo=None, hi=None, lo_open=False, hi_open=False) -> float:
    """유한 실수(bool 불가) 를 float 로. 범위 [lo,hi] (open 이면 열린 끝) 를 벗어나면 ValueError."""
    if not _is_real(x) or not math.isfinite(float(x)):
        raise ValueError(f"{name}={x!r}: 유한한 실수여야 한다")
    v = float(x)
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ValueError(f"{name}={v}: {'>' if lo_open else '>='} {lo} 이어야 한다")
    if hi is not None and (v >= hi if hi_open else v > hi):
        raise ValueError(f"{name}={v}: {'<' if hi_open else '<='} {hi} 이어야 한다")
    return v


def _img(x, name="rgb") -> np.ndarray:
    """(H,W,3) float32/float64 영상 검사 (그 밖의 dtype 은 TypeError, 모양·비유한은 ValueError)."""
    a = np.asarray(x)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{name}: (H,W,3) 이어야 한다 (shape {a.shape})")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{name}: 빈 영상 (shape {a.shape})")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: 유한하지 않은 값이 있다")
    return a


def _tint3(tint) -> tuple:
    try:
        t = tuple(tint)
    except TypeError as e:
        raise ValueError(f"tint={tint!r}: 길이 3 이어야 한다") from e
    if len(t) != 3:
        raise ValueError(f"tint={tint!r}: 길이 3 이어야 한다")
    return tuple(_num("tint", v, lo=0.0) for v in t)


def _half_diag(h, w) -> float:
    r = math.hypot(w - 1, h - 1) / 2.0
    return r if r > 0.0 else 1.0


# ---------------------------------------------------------------------------------------------------------------
# 처방
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Prescription:
    optics: _optics.OpticsSpec = _optics.OpticsSpec()
    squeeze: float = 1.0
    swirl: float = 0.0
    edge_blur: float = 0.0
    streak: float = 0.0
    streak_len_frac: float = 0.35
    streak_tint: tuple = DEFAULT_STREAK_TINT
    streak_threshold: float = 0.8
    name: str = "custom"

    def __post_init__(self):
        if not isinstance(self.optics, _optics.OpticsSpec):
            raise TypeError(f"optics: OpticsSpec 이어야 한다 ({type(self.optics).__name__})")
        s = object.__setattr__
        s(self, "squeeze", _num("squeeze", self.squeeze, lo=1.0, hi=SQUEEZE_MAX))
        s(self, "swirl", _num("swirl", self.swirl, lo=0.0, hi=1.0))
        s(self, "edge_blur", _num("edge_blur", self.edge_blur, lo=0.0, hi=1.0))
        s(self, "streak", _num("streak", self.streak, lo=0.0, hi=1.0))
        s(self, "streak_len_frac", _num("streak_len_frac", self.streak_len_frac, lo=0.0, hi=1.0, lo_open=True))
        s(self, "streak_tint", _tint3(self.streak_tint))
        s(self, "streak_threshold", _num("streak_threshold", self.streak_threshold, lo=0.0))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name={self.name!r}: 비어 있지 않은 str 이어야 한다")

    @property
    def catseye(self) -> float:
        return self.optics.catseye

    @property
    def blades(self) -> int:
        return self.optics.blades

    @property
    def blade_rot_deg(self) -> float:
        return self.optics.blade_rot_deg


def _check_rx(rx):
    if not isinstance(rx, Prescription):
        raise TypeError(f"rx: Prescription 이어야 한다 ({type(rx).__name__})")
    return rx


# ---------------------------------------------------------------------------------------------------------------
# 출사동 커널
# ---------------------------------------------------------------------------------------------------------------
def pupil_open_fraction(r_frac, catseye) -> float:
    """원형 조리개 cat-eye 개구율 (2/π)(acos q − q√(1−q²)), q = CATSEYE_MAX_SHIFT·r_frac·catseye."""
    rf = _num("r_frac", r_frac, lo=0.0, hi=1.0)
    ce = _num("catseye", catseye, lo=0.0, hi=1.0)
    q = min(_optics.CATSEYE_MAX_SHIFT * rf * ce, 1.0)
    return float((2.0 / math.pi) * (math.acos(q) - q * math.sqrt(max(0.0, 1.0 - q * q))))


@lru_cache(maxsize=4096)
def _kernel_core(d: float, r_frac: float, angle_deg: float, catseye: float, blades: int, rot_deg: float,
                 swirl: float, squeeze: float) -> np.ndarray:
    """읽기 전용 float32 커널 (검증 끝난 인자)."""
    if d <= 1.0:
        k = np.ones((1, 1), np.float32)
        k.flags.writeable = False
        return k
    base = _optics.catseye_kernel(d, r_frac, angle_deg, catseye, blades, rot_deg)
    tau = swirl * SWIRL_MAX_TWIST_DEG * r_frac
    if tau == 0.0 and squeeze == 1.0:
        base.flags.writeable = False
        return base
    sq = math.sqrt(squeeze)
    K = 2 * math.ceil(d * sq / 2.0) + 1
    half = (K - 1) // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(ax, ax)
    qx, qy = X * sq, Y / sq
    if tau != 0.0:
        phi = math.radians(tau) * np.hypot(qx, qy) / (d / 2.0)
        c, s = np.cos(phi), np.sin(phi)
        sx, sy = qx * c + qy * s, -qx * s + qy * c  # q 를 −φ 만큼 되돌린 위치
    else:
        sx, sy = qx, qy
    cb = (base.shape[0] - 1) / 2.0
    out = _ndi.map_coordinates(base.astype(np.float64), [sy + cb, sx + cb], order=1, mode="constant", cval=0.0)
    out = np.maximum(out, 0.0)
    tot = float(out.sum())
    if tot <= 0.0:
        out = np.zeros((K, K))
        out[half, half] = 1.0
    else:
        out = out / tot
    k = out.astype(np.float32)
    k.flags.writeable = False
    return k


def exit_pupil_kernel(diameter_px, r_frac, angle_deg, rx) -> np.ndarray:
    """화소 하나의 산란 커널: LNS-05 cat-eye 바탕 -> 스월(반지름 비례 비틀림) -> 애너모픽 스퀴즈 -> 합 1.
    (K,K) float32, K = 2·ceil(d·√s/2)+1 (d<=1 이면 [[1]])."""
    d = _num("diameter_px", diameter_px, lo=0.0)
    rf = _num("r_frac", r_frac, lo=0.0, hi=1.0)
    ang = _num("angle_deg", angle_deg)
    _check_rx(rx)
    return _kernel_core(d, rf, ang, rx.catseye, rx.blades, rx.blade_rot_deg, rx.swirl, rx.squeeze).copy()


# ---------------------------------------------------------------------------------------------------------------
# 주변부 번짐
# ---------------------------------------------------------------------------------------------------------------
def edge_softness(rgb, amount) -> np.ndarray:
    """out=(1−w)·x + w·G(x), w=clip((r−EDGE_R_INNER)/(1−EDGE_R_INNER),0,1)², G=가우시안(σ=amount·EDGE_SIGMA_FRAC·W)."""
    a = _img(rgb, "rgb")
    amt = _num("amount", amount, lo=0.0, hi=1.0)
    if amt == 0.0:
        return a.copy()
    h, w = a.shape[:2]
    x = a.astype(np.float64)
    sig = amt * EDGE_SIGMA_FRAC * w
    g = _ndi.gaussian_filter(x, sigma=(sig, sig, 0.0), mode="nearest", truncate=4.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0) / _half_diag(h, w)
    wt = np.clip((r - EDGE_R_INNER) / (1.0 - EDGE_R_INNER), 0.0, 1.0) ** 2
    wt = wt[..., None]
    return ((1.0 - wt) * x + wt * g).astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 수평 플레어
# ---------------------------------------------------------------------------------------------------------------
def streak_layer(rgb_lin, *, gain, length_frac=0.35, tint=DEFAULT_STREAK_TINT, threshold=0.8) -> np.ndarray:
    """가산 층. E=max(x−thr,0) 를 행마다 h(k)=exp(−|k|/L)/Z (L=length_frac·W) 와 합성 -> 세로 [.25,.5,.25] -> gain·tint."""
    a = _img(rgb_lin, "rgb_lin")
    g = _num("gain", gain, lo=0.0)
    lf = _num("length_frac", length_frac, lo=0.0, hi=1.0, lo_open=True)
    tn = np.asarray(_tint3(tint), np.float64)
    thr = _num("threshold", threshold, lo=0.0)
    h, w = a.shape[:2]
    E = np.maximum(a.astype(np.float64) - thr, 0.0)
    if g == 0.0 or not E.any():
        return np.zeros(a.shape, a.dtype)
    L = lf * w
    kk = np.arange(-(w - 1), w, dtype=np.float64)
    z = float(np.exp(-np.abs(kk) / L).sum())
    if w <= _STREAK_MATMUL_MAX_W:
        idx = np.arange(w, dtype=np.float64)
        T = np.exp(-np.abs(idx[:, None] - idx[None, :]) / L) / z
        H1 = np.matmul(E.transpose(0, 2, 1), T).transpose(0, 2, 1)
    else:
        hk = np.exp(-np.abs(kk) / L) / z
        H1 = np.maximum(_fftconvolve(E, hk[None, :, None], mode="same", axes=1), 0.0)
    V = 0.5 * H1
    V[1:] += 0.25 * H1[:-1]
    V[:-1] += 0.25 * H1[1:]
    return (g * tn * V).astype(a.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 처방 보케 (화소별 출사동 커널 산란)
# ---------------------------------------------------------------------------------------------------------------
def _quant_d(d: np.ndarray) -> np.ndarray:
    step = np.where(d <= 8.0, 0.25, np.where(d <= 32.0, 0.5, np.where(d <= 96.0, 1.0, 2.0)))
    return np.round(d / step) * step


def apply_prescription_bokeh(rgb, coc_px, rx) -> np.ndarray:
    """단일 층 산란 블러: 화소마다 자기 CoC 와 (r_frac, angle) 의 exit_pupil_kernel 로 퍼뜨린다. CoC<=1 px 는 선명.
    (r_frac 은 1/16, 각도는 7.5°, d 는 _quant_d 단위로 양자화한다. 프레임 경계 안쪽 kernel half 이내는 비규정.)
    swirl=0 이고 squeeze=1 이면 optics.apply_catseye 와 같다."""
    img = _img(rgb, "rgb")
    h, w = img.shape[:2]
    c = np.asarray(coc_px)
    if c.dtype.kind not in "iuf":
        raise TypeError(f"coc_px: 수치 배열이어야 한다 (dtype {c.dtype})")
    if c.shape != (h, w):
        raise ValueError(f"coc_px: (H,W)={(h, w)} 이어야 한다 (shape {c.shape})")
    c = c.astype(np.float64)
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("coc_px: 비음수 유한이어야 한다")
    _check_rx(rx)
    if rx.swirl == 0.0 and rx.squeeze == 1.0:
        return _optics.apply_catseye(img, c, strength=rx.catseye, blades=rx.blades, blade_rot_deg=rx.blade_rot_deg)

    src = img.astype(np.float64)
    dq_all = np.where(c > 1.0, _quant_d(c), 0.0)
    sharp = dq_all <= 1.0
    acc = np.where(sharp[..., None], src, 0.0)
    ys, xs = np.nonzero(~sharp)
    if ys.size == 0:
        return acc.astype(img.dtype)
    d = dq_all[ys, xs]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    R = _half_diag(h, w)
    rf = np.minimum(np.hypot(xs - cx, ys - cy) / R, 1.0)
    ang = np.degrees(np.arctan2(ys - cy, xs - cx)) % 180.0
    ri = np.round(rf * 16.0).astype(np.int64)
    if rx.catseye == 0.0 and rx.swirl == 0.0:
        ri = np.zeros_like(ri)                    # 반지름·각도 모두 커널에 영향이 없다
    ai = np.where((ri > 0) & (rx.catseye > 0.0), np.round(ang / 7.5).astype(np.int64) % 24, 0)
    di = np.round(d * 4.0).astype(np.int64)
    key = (di * 17 + ri) * 24 + ai
    order = np.argsort(key, kind="stable")
    cuts = np.nonzero(np.diff(key[order]))[0] + 1
    groups = np.split(order, cuts)

    kers = []
    for g in groups:
        k0 = int(g[0])
        ker = _kernel_core(float(di[k0]) / 4.0, float(ri[k0]) / 16.0, float(ai[k0]) * 7.5, rx.catseye, rx.blades,
                           rx.blade_rot_deg, rx.swirl, rx.squeeze)
        kers.append(ker)
    Hm = max((k.shape[0] - 1) // 2 for k in kers)
    Hp, Wp = h + 2 * Hm, w + 2 * Hm
    Hf, Wf = _sfft.next_fast_len(Hp, real=True), _sfft.next_fast_len(Wp, real=True)
    K2 = 2 * Hm + 1
    buf = np.zeros((Hp, Wp, 3))
    spec_acc = None
    for g, ker in zip(groups, kers):
        hk = (ker.shape[0] - 1) // 2
        gy, gx = ys[g], xs[g]
        if g.size <= 40:
            kk = ker.astype(np.float64)[:, :, None]
            for y, x in zip(gy.tolist(), gx.tolist()):
                buf[y + Hm - hk:y + Hm + hk + 1, x + Hm - hk:x + Hm + hk + 1] += kk * src[y, x]
            continue
        m = np.zeros((h, w, 3))
        m[gy, gx] = src[gy, gx]
        emb = np.zeros((K2, K2))
        emb[Hm - hk:Hm + hk + 1, Hm - hk:Hm + hk + 1] = ker
        part = _sfft.rfft2(m, s=(Hf, Wf), axes=(0, 1)) * _sfft.rfft2(emb, s=(Hf, Wf))[:, :, None]
        spec_acc = part if spec_acc is None else spec_acc + part
    if spec_acc is not None:
        buf += _sfft.irfft2(spec_acc, s=(Hf, Wf), axes=(0, 1))[:Hp, :Wp]
    return (acc + buf[Hm:Hm + h, Hm:Hm + w]).astype(img.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 렌즈별 처방 표 · θ 사상
# ---------------------------------------------------------------------------------------------------------------
def _row(**kw):
    return MappingProxyType(kw)


# swirl/edge_blur/streak/len_frac 는 해상도 불변 비율. 수차 px 는 LNS-05 spec_from_theta 가 폭에 선형으로 준다.
RX_TABLE = MappingProxyType({
    "spherical_clean": _row(swirl=0.0, edge_blur=0.05, streak=0.0),
    "vintage_petzval": _row(swirl=0.8, edge_blur=0.75, streak=0.0),
    "vintage_double_gauss": _row(swirl=0.25, edge_blur=0.3, streak=0.0),
    "anamorphic_2x": _row(swirl=0.0, edge_blur=0.1, streak=0.45, streak_len_frac=0.4,
                          streak_tint=(0.55, 0.75, 1.0)),
    "soft_single_coat": _row(swirl=0.0, edge_blur=0.35, streak=0.0),
})


def prescription_from_theta(theta, width_px) -> Prescription:
    """θ.lens_rx 표 + LNS-05 spec_from_theta + extra.squeeze -> Prescription."""
    spec = _optics.spec_from_theta(theta, width_px)  # Theta/width/미등록 렌즈 검사 (TypeError/ValueError)
    name = theta.lens_rx
    if name not in _theta.LENS_RX_NAMES or name not in RX_TABLE:
        raise ValueError(f"lens_rx={name!r}: 미등록 렌즈")
    ana = name == "anamorphic_2x"
    extra = theta.extra if isinstance(theta.extra, dict) else {}
    try:
        squeeze = float(extra.get("squeeze", 2.0 if ana else 1.0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"extra.squeeze={extra.get('squeeze')!r}: 실수여야 한다") from e
    if ana != (squeeze == 2.0):
        raise ValueError(f"lens_rx={name!r} 와 squeeze={squeeze} 가 맞지 않는다 (anamorphic_2x <-> squeeze==2.0)")
    return Prescription(optics=spec, squeeze=squeeze, name=name, **RX_TABLE[name])


# ---------------------------------------------------------------------------------------------------------------
# 체인
# ---------------------------------------------------------------------------------------------------------------
def apply_prescription(rgb_lin, rx) -> np.ndarray:
    """out = apply_optics(edge_softness(x), rx.optics) + (streak>0 이면) streak_layer(x). cat-eye·스월·스퀴즈 블러는
    소비하지 않는다 (호출자가 apply_prescription_bokeh)."""
    a = _img(rgb_lin, "rgb_lin")
    _check_rx(rx)
    out = _optics.apply_optics(edge_softness(a, rx.edge_blur), rx.optics)
    if rx.streak > 0.0:
        out = out + streak_layer(a, gain=rx.streak, length_frac=rx.streak_len_frac, tint=rx.streak_tint,
                                 threshold=rx.streak_threshold)
    return out


def _warp_nearest(arr: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """(H,W) AOV 를 왜곡과 같은 사상으로 최근접 표본화."""
    if arr.dtype in (np.float32, np.float64):
        return _optics.apply_distortion(arr, k1, k2, nearest=True)
    return _optics.apply_distortion(arr.astype(np.float64), k1, k2, nearest=True).astype(arr.dtype)


def apply_prescription_frames(frames, theta, *, rx=None):
    """표시값 Frames -> 선형 -> 프레임마다 apply_prescription -> 표시값. (N,H,W) AOV 는 같은 왜곡으로 최근접 사상."""
    if not isinstance(frames, _C.Frames):
        raise TypeError(f"frames: Frames 여야 한다 ({type(frames).__name__})")
    if rx is not None:
        _check_rx(rx)
    rgb = frames.rgb
    n, h, w = rgb.shape[:3]
    if rx is None:
        rx = prescription_from_theta(theta, w)
    out_rgb = np.empty(rgb.shape, np.float32)
    for i in range(n):
        out_rgb[i] = _color.linear_to_display(apply_prescription(_color.srgb_to_linear(rgb[i]), rx))
    aov = {}
    for k, v in frames.aov.items():
        arr = np.asarray(v)
        if arr.ndim == 3 and arr.shape == (n, h, w):
            aov[k] = np.stack([_warp_nearest(arr[i], rx.optics.k1, rx.optics.k2) for i in range(n)])
        else:
            aov[k] = arr.copy()
    return _C.Frames(rgb=out_rgb, fps=frames.fps, aov=aov, shot_hash=frames.shot_hash,
                     world_hash=frames.world_hash)
