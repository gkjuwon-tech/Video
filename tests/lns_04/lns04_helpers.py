"""LNS-04 테스트 공용 도우미. 오라클은 계약서 수식과 numpy/scipy/FND-06 color 만 쓰고 구현 내부를 참조하지 않는다."""
import math

import numpy as np
from scipy import ndimage as ndi

from buc.core import color

# ---- 계약서 상수(정본). 테스트는 이 값으로 기대치를 만들고, 모듈 상수가 같은지도 따로 검사한다.
LUT_SIZE = 65
LOG_MIDGRAY = 0.18
LOG_STOPS = (-8.0, 6.0)
GRAIN_SIGMA_MAX = 0.08
GRAIN_RADIUS_FRAC = 0.0012
GRAIN_RADIUS_MIN_PX = 0.75
HALATION_THRESHOLD = 0.8
HALATION_SIGMA_FRAC = 0.02
HALATION_TINT = (1.0, 0.35, 0.12)
HALATION_GAIN = 1.0
VIGNETTE_MAX_DROP = 0.8
FULL_WELL_PER_UM2 = 1500.0
FULL_WELL_MIN, FULL_WELL_MAX = 1500.0, 100000.0
READ_NOISE_E = 2.5
ADC_BITS = 12

LUTS = ("none", "neutral_709", "teal_orange", "bleach_bypass", "kodak_warm", "fuji_cool", "mono_silver",
        "faded_matte", "neon_night", "cross_process")

# PTC 용 명시 센서
FW, RN, BITS = 20000.0, 3.0, 12


def log_encode_ref(x):
    lo, hi = LOG_STOPS
    e = (np.log2(np.maximum(np.asarray(x, np.float64), 1e-30) / LOG_MIDGRAY) - lo) / (hi - lo)
    return np.clip(e, 0.0, 1.0)


def log_decode_ref(e):
    lo, hi = LOG_STOPS
    return LOG_MIDGRAY * 2.0 ** (lo + np.asarray(e, np.float64) * (hi - lo))


def display_ref(x_lin):
    """none/neutral 의 기준: 장면 선형 -> 클램프 후 sRGB 부호화."""
    return color.linear_to_display(np.asarray(x_lin, np.float32)).astype(np.float64)


def gray_img(v, h=64, w=96):
    return np.full((h, w, 3), v, np.float32)


def scene(h=72, w=104, seed=7):
    """결정론 장면: 매끈한 그라데이션 + 색 블록 + 밝은 광원(HDR). 선형, 최대 > 1."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 0.05 + 0.4 * (xx / w)[..., None] * np.array([1.0, 0.8, 0.6]) + 0.1 * (yy / h)[..., None]
    img = base + 0.02 * rng.random((h, w, 3))
    img[10:22, 14:30] = (0.7, 0.1, 0.1)
    img[40:56, 60:80] = (0.1, 0.5, 0.2)
    img[28:34, 50:56] = (6.0, 5.5, 5.0)  # 광원
    return img.astype(np.float32)


def autocorr(f, lag, axis):
    """영 평균 가정 정규화 자기상관(순환). f: (H,W)."""
    f = f - f.mean()
    g = np.roll(f, -lag, axis=axis)
    return float((f * g).mean() / (f.var() + 1e-30))


def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-30))


def lab_chroma(display_rgb):
    lin = color.srgb_to_linear(np.asarray(display_rgb, np.float32))
    lab = color.rgb_lin_to_lab(lin)
    return np.hypot(lab[..., 1], lab[..., 2])


def lab_L(display_rgb):
    lin = color.srgb_to_linear(np.asarray(display_rgb, np.float32))
    return color.rgb_lin_to_lab(lin)[..., 0]


def halation_oracle(x_lin, amount):
    """계약서 수식: layer_c = amount*GAIN*tint_c * G_sigma * max(Y - T, 0), 영 경계, sigma = FRAC*W."""
    x = np.asarray(x_lin, np.float64)
    Y = color.luminance(x)
    hl = np.maximum(Y - HALATION_THRESHOLD, 0.0)
    b = ndi.gaussian_filter(hl, HALATION_SIGMA_FRAC * x.shape[1], mode="constant", truncate=5.0)
    return amount * HALATION_GAIN * b[..., None] * np.array(HALATION_TINT)


def vignette_ref(h, w, s):
    yy = np.arange(h) - (h - 1) / 2.0
    xx = np.arange(w) - (w - 1) / 2.0
    hd = math.hypot((w - 1) / 2.0, (h - 1) / 2.0)
    return 1.0 - VIGNETTE_MAX_DROP * s * (yy[:, None] ** 2 + xx[None, :] ** 2) / (hd * hd)
