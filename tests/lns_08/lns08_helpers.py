"""LNS-08 테스트 공용 도우미. 오라클은 계약서 수식과 numpy/scipy/FND-06 color/LNS-04 공개 조각만 쓰고 chain.py 내부를 참조하지 않는다."""
import numpy as np
from scipy import ndimage as ndi

from buc.core import color
from buc.core import contracts as C
from buc.lens import sensor_film as sf

# ---- 계약서 상수(정본). 테스트는 이 값으로 기대치를 만들고, 모듈 상수가 같은지도 따로 검사한다.
WB_REF_KELVIN = 6500.0
WB_MIRED_STOPS = 0.004
WB_TINT_STOPS = 1.0
OWN_OK, OWN_SENSOR_BLACK, OWN_SENSOR_SAT, OWN_LUT_LOST, OWN_HALO_TAINT = 0, 1, 2, 4, 8
SAT_MARGIN_LSB = 1.5
LUT_INV_TOL = 2e-5
HALO_TAINT_EPS = 5e-7
HALO_SOURCE_MAX = 16.0
BLACK_CODE_EPS = 2e-3

LUTS = ("none", "neutral_709", "teal_orange", "bleach_bypass", "kodak_warm", "fuji_cool", "mono_silver",
        "faded_matte", "neon_night", "cross_process")
# 클립·랭크 결손이 없어 어떤 장면에서도 역변환이 닫혀 있는 룩(5종)
CLEAN_LUTS = ("none", "neutral_709", "bleach_bypass", "faded_matte", "cross_process")
# 출력 클립 때문에 일부 색이 소유되지 않는 룩
CLIPPY_LUTS = ("teal_orange", "kodak_warm", "fuji_cool", "neon_night")
LOG_FLOOR_S = sf.LOG_MIDGRAY * 2.0 ** sf.LOG_STOPS[0]  # 로그 부호화 바닥(선형) = 7.03e-4


def theta(**kw):
    return C.Theta(**kw)


# ------------------------------------------------------------------------------------------------ WB / 체인 오라클
def wb_gains_ref(kelvin, tint):
    m = 1e6 / float(kelvin) - 1e6 / WB_REF_KELVIN
    raw = 2.0 ** np.array([-WB_MIRED_STOPS * m, -WB_TINT_STOPS * float(tint), WB_MIRED_STOPS * m])
    return raw / float(color.luminance(raw))


def halation_k(amount, c):
    """계약서의 할레이션 되먹임 계수 K_c = 1 + a*tint_c / (1 - a*Y_H)."""
    yh = float(color.luminance(np.asarray(sf.HALATION_TINT, np.float64)))
    return 1.0 + amount * sf.HALATION_TINT[c] / (1.0 - yh * amount)


def bound_ref(th, h, w, bits=12):
    """계약서의 error_bound 식(소유 화소). (h,w,3) float64."""
    q = 1.0 / ((1 << bits) - 1)
    g = wb_gains_ref(th.wb_kelvin, th.wb_tint)
    v = sf.vignette_gain(h, w, float(th.vignette)).astype(np.float64)
    scale = v[..., None] * 2.0 ** float(th.ev) * g[None, None, :]
    k = np.array([halation_k(float(th.halation), c) for c in range(3)])
    return (0.5 * q + LUT_INV_TOL) * k[None, None, :] / scale


def exposure_ref(x, th):
    """센서 입력 노출 u = x * gains * 2^ev * V (할레이션 층 제외). (h,w,3) float64."""
    h, w = x.shape[:2]
    g = wb_gains_ref(th.wb_kelvin, th.wb_tint)
    v = sf.vignette_gain(h, w, float(th.vignette)).astype(np.float64)
    return x.astype(np.float64) * g[None, None, :] * 2.0 ** float(th.ev) * v[..., None]


def look_by_pieces(x, th, *, seed=0, frame=0, noise=True):
    """LNS-04 공개 조각의 합성으로 만든 look 기준: WB -> EV -> 비네팅 -> 할레이션 -> 센서 -> 룩 -> 그레인."""
    g32 = wb_gains_ref(th.wb_kelvin, th.wb_tint).astype(np.float32)
    a = x.astype(np.float32) * g32
    a = a * np.float32(2.0 ** float(th.ev))
    a = sf.apply_vignette(a, th.vignette)
    a = sf.apply_halation(a, th.halation)
    spec = sf.sensor_spec_from_theta(th, a.shape[1])
    a = sf.sensor_capture(a, spec, seed=seed, frame=frame, noise=noise)
    y = sf.apply_look(a, th.lut)
    if noise and float(th.grain) > 0.0:
        y = sf.grain_field(y, strength=float(th.grain), radius_px=max(sf.GRAIN_RADIUS_MIN_PX, sf.GRAIN_RADIUS_FRAC * a.shape[1]),
                           seed=seed, frame=frame)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ 장면
def scene_hdr(h=48, w=72, seed=7):
    """결정론 장면: 그라데이션 + 색 블록 + 포화되는 밝은 광원(6.0). 선형, 최대 > 1."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 0.05 + 0.4 * (xx / w)[..., None] * np.array([1.0, 0.8, 0.6]) + 0.1 * (yy / h)[..., None]
    img = base + 0.02 * rng.random((h, w, 3))
    img[h // 5:h // 2, w // 8:w // 4] = (0.7, 0.1, 0.1)
    img[int(h * 0.6):int(h * 0.9), int(w * 0.55):int(w * 0.75)] = (0.1, 0.5, 0.2)
    img[h // 2 - 3:h // 2 + 3, w // 2 - 3:w // 2 + 3] = (6.0, 5.5, 5.0)
    return img.astype(np.float32)


def scene_mid(h=48, w=72, seed=7):
    """포화 없는 중간톤 장면(최대 약 0.4)."""
    return (np.clip(scene_hdr(h, w, seed), 0.0, 0.9) * 0.45).astype(np.float32)


def scene_ramp(h=24, w=96):
    """행마다 다른 색 방향의 로그 램프 1e-4 .. 1.5 (어두운 곳·바닥·포화를 한 장에)."""
    t = np.exp(np.linspace(np.log(1e-4), np.log(1.5), w))
    rows = []
    mixes = [(1, 1, 1), (1, .5, .25), (.25, .5, 1), (.5, 1, .5), (1, .1, .1), (.1, .1, 1)]
    for r in range(h):
        mx = np.array(mixes[r % len(mixes)], np.float64)
        rows.append(t[:, None] * mx[None, :] * (1.0 + 0.0 * r))
    return np.stack(rows).astype(np.float32)


def flat(v, h=32, w=32):
    return np.full((h, w, 3), v, np.float32)


def gauss_blur_ref(m, w_px):
    return ndi.gaussian_filter(m.astype(np.float64), sf.HALATION_SIGMA_FRAC * w_px, mode="constant", truncate=5.0)


def srgb_oetf(lin):
    lin = np.clip(np.asarray(lin, np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
