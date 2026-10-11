"""RD-06 테스트 공용 도우미: 독립 오라클(계약서 수식 그대로), 합성 장면, 지표.
RD-06 구현을 오라클에 쓰지 않는다. 난수는 모두 시드 고정 Generator."""
import math

import numpy as np

from buc.core import color
from buc.lens import chain

W709 = np.array([0.2126, 0.7152, 0.0722])


# ------------------------------------------------------------------------------------------------ 오라클
def srgb_enc(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.maximum(v, 1e-300) ** (1 / 2.4) - 0.055)


def srgb_dec(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0.0) + 0.055) / 1.055) ** 2.4)


def o_curve(x, toe=0.0, shoulder=0.0, slope=1.0, white=1.0):
    """필름 곡선 오라클: 피벗 거듭제곱 -> 토(유리식) -> 어깨(일반화 라인하르트 / 하드 클립)."""
    x = np.asarray(x, np.float64)
    p = 0.18 * (x / 0.18) ** slope
    a = 0.02
    b = a * (1.0 + 2.0 * toe)
    t = p * (p + a) / (p + b)
    if shoulder == 0.0:
        return np.minimum(t, white)
    S = 1.0 / shoulder
    r = t / white
    # (1 + r^S)^(1/S) 를 r>1 에서 안정적으로: r * (1 + r^-S)^(1/S)
    big = r > 1.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        d = np.where(big, r * (1.0 + np.where(big, r, 2.0) ** -S) ** (1.0 / S), (1.0 + np.where(big, 0.0, r) ** S) ** (1.0 / S))
    return t / d


def o_cdl(c, slope=(1, 1, 1), offset=(0, 0, 0), power=(1, 1, 1), sat=1.0):
    v = np.maximum(np.asarray(c, np.float64) * np.array(slope) + np.array(offset), 0.0) ** np.array(power)
    Y = (v @ W709)[..., None]
    return Y + sat * (v - Y)


def lab(rgb_lin):
    return color.rgb_lin_to_lab(np.asarray(rgb_lin, np.float64))


def de00(a_lin, b_lin):
    return color.delta_e_2000(lab(a_lin), lab(b_lin))


# ------------------------------------------------------------------------------------------------ 합성 장면
def scene(seed=0, shape=(64, 96), mu=0.18, sigma=1.5, chroma=0.3):
    """앨베도 prior 에 맞춘 장면 선형: log2 휘도 ~ N(log2 mu, sigma), 채도는 로그비 평균 0(회색 세계)."""
    rng = np.random.default_rng(seed)
    lum = 2.0 ** (math.log2(mu) + sigma * rng.standard_normal(shape))
    c = chroma * rng.standard_normal(shape + (2,))
    rgb = np.stack([np.exp(c[..., 0]), np.ones(shape), np.exp(c[..., 1])], -1)
    rgb = rgb / (rgb @ W709)[..., None]
    return (rgb * lum[..., None]).astype(np.float64)


def keyed_scene(seed, mu, sigma=1.5, shape=(80, 120), sky_frac=0.3, chroma=0.25):
    """scene() 와 같은 분포지만 가장 밝은 sky_frac 화소를 위 행에 모은 '하늘' 마스크를 함께 돌려준다."""
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    lum = np.sort(2.0 ** (math.log2(mu) + sigma * rng.standard_normal(n)))[::-1]
    ns = int(sky_frac * n)
    flat = np.concatenate([rng.permutation(lum[:ns]), rng.permutation(lum[ns:])])
    Y = flat.reshape(shape)
    sky = np.zeros(shape, bool)
    sky.reshape(-1)[:ns] = True
    c = chroma * rng.standard_normal(shape + (2,))
    rgb = np.stack([np.exp(c[..., 0]), np.ones(shape), np.exp(c[..., 1])], -1)
    rgb = rgb / (rgb @ W709)[..., None]
    return rgb * Y[..., None], sky


def gains_dev_stops(k1, t1, k2, t2):
    """두 WB 설정의 채널 이득비 최대 편차(stop)."""
    return float(np.max(np.abs(np.log2(chain.wb_gains(k1, t1) / chain.wb_gains(k2, t2)))))


def ramp(lo=-3.0, hi=1.5, n=28):
    """중간 회색 기준 stop 램프 (n,3) float32 중립."""
    return ((0.18 * 2.0 ** np.linspace(lo, hi, n))[:, None] * np.ones((1, 3))).astype(np.float32)


def smooth_lut(rng, amp=0.03, n=17):
    """항등 근처의 매끈한 임의 LUT: 저차 다항 섭동(경계에서 0 이 되도록 c(1-c) 곱)."""
    g = np.linspace(0, 1, n)
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    c = np.stack([r, gg, b], -1)
    env = np.prod(c * (1 - c), axis=-1, keepdims=True) ** (1 / 3) * 2.0
    coef = rng.uniform(-1, 1, (3, 4))
    feats = np.stack([np.ones_like(r), r - 0.5, gg - 0.5, b - 0.5], -1)
    pert = np.einsum("xyzk,ck->xyzc", feats, coef) * amp * env
    return np.clip(c + pert, 0, 1).astype(np.float32)
