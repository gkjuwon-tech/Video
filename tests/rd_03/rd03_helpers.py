"""RD-03 테스트 공용 도우미: 합성 장면 생성기와 순방향 오라클.

여기 있는 코드는 RD-03 구현(buc.read.screen_fx)을 참조하지 않는다. numpy, scipy, FND-06 color, LNS-04 공개 순방향
(vignette_gain, grain_field, apply_halation, apply_sensor_film)만 쓴다. 모든 장면은 시드로 결정되는 float64 선형 (H,W,3).
"""
import numpy as np
from scipy import ndimage as ndi

from buc.core import color
from buc.core import contracts as C
from buc.lens import sensor_film as sf

# ---- 계약서 상수(정본). 테스트는 이 값으로 기대치를 만들고, 모듈 상수가 같은지도 따로 검사한다.
GRAIN_SIGMA_MAX = 0.08
VIGNETTE_MAX_DROP = 0.8
HALATION_THRESHOLD = 0.8
HALATION_SIGMA_FRAC = 0.02
HALATION_TINT = (1.0, 0.35, 0.12)
HALO_SOURCE_MAX = 16.0
HALO_ASSUMED_SOURCE_Y = 2.0

H0, W0 = 240, 384          # 기본 시험 해상도 (3:2 근방, 반대각선 ~226 px)


def theta(**kw):
    return C.Theta(**kw)


# ------------------------------------------------------------------------------------------------ 장면 생성기
def field(h, w, seed=0, amp_exp=1.0):
    """진폭 스펙트럼 nu^-amp_exp 인 무작위 위상 장. 평균 0, 표준편차 1. 결정론."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    nu = np.hypot(fx, fy)
    nu[0, 0] = 1.0
    spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) * nu ** (-amp_exp)
    spec[0, 0] = 0.0
    x = np.fft.ifft2(spec).real
    return (x - x.mean()) / x.std()


def rect_scene(h=H0, w=W0, seed=0, n=220, lo=0.05, hi=0.8, soft=0.8, colored=True, base=0.25, base_std=0.004):
    """구간 상수 장면: 매끈한 바탕 위에 무작위 사각형 n 개(수준 lo..hi, 가장자리 σ=soft). 평탄 영역·에지·여러 밝기가 모두 있다."""
    rng = np.random.default_rng(seed)
    bg = base + base_std * field(h, w, seed + 1000)
    img = np.repeat(bg[..., None], 3, axis=2)
    for _ in range(n):
        rw = int(rng.integers(max(8, w // 16), max(9, w // 4)))
        rh = int(rng.integers(max(8, h // 16), max(9, h // 4)))
        x0 = int(rng.integers(-rw // 2, w - rw // 2))
        y0 = int(rng.integers(-rh // 2, h - rh // 2))
        lev = rng.uniform(lo, hi)
        col = lev * (rng.uniform(0.6, 1.4, 3) if colored else np.ones(3))
        img[max(0, y0):max(0, y0 + rh), max(0, x0):max(0, x0 + rw)] = np.clip(col, 0.02, 0.95)
    if soft > 0:
        img = ndi.gaussian_filter(img, (soft, soft, 0), mode="nearest")
    return np.clip(img, 0.02, 0.95)


def flat_scene(h=H0, w=W0, seed=0, n=60, lo=0.05, hi=0.8, soft=0.8):
    """큰 평탄 사각형 위주의 구간 상수 장면(평탄 블록 비율 약 절반 이상). 그레인 시험용: 여러 밝기의 평탄 영역과 몇 개의 에지."""
    rng = np.random.default_rng(seed + 77)
    img = np.repeat((0.25 + 0.002 * field(h, w, seed + 1000))[..., None], 3, axis=2)
    for _ in range(n):
        rw = int(rng.integers(max(16, w // 6), max(17, w // 2)))
        rh = int(rng.integers(max(16, h // 4), max(17, h // 1.6)))
        x0 = int(rng.integers(-rw // 3, w - 2 * rw // 3))
        y0 = int(rng.integers(-rh // 3, h - 2 * rh // 3))
        lev = rng.uniform(lo, hi)
        col = lev * rng.uniform(0.7, 1.3, 3)
        img[max(0, y0):max(0, y0 + rh), max(0, x0):max(0, x0 + rw)] = np.clip(col, 0.02, 0.95)
    return np.clip(ndi.gaussian_filter(img, (soft, soft, 0), mode="nearest"), 0.02, 0.95)


def hf_field(h, w, seed=0, corr_px=4.0, lf_cut_px=20.0):
    """고주파 정상 장: 평균 0, 표준편차 1, σ=lf_cut_px 보다 큰 구조 없음(방사 이득과 혼동되는 저주파 제거)."""
    f = ndi.gaussian_filter(field(h, w, seed, amp_exp=0.5), corr_px / 2.0, mode="wrap")
    f = f - ndi.gaussian_filter(f, lf_cut_px, mode="wrap")
    return (f - f.mean()) / f.std()


def texture_scene(h=H0, w=W0, seed=0, ev_std=0.35, corr_px=4.0, lf_cut_px=20.0):
    """정상(stationary) 질감: 대역통과 로그정규 장(밝기 로그2 표준편차 ev_std). 비네팅 시험용(평탄하지 않다).
    저주파 구조는 뺀다: 그 성분은 방사 이득과 구분되지 않는다(계약서 한계)."""
    f = hf_field(h, w, seed, corr_px, lf_cut_px)
    y = 0.3 * 2.0 ** (ev_std * f)
    tint = 1.0 + 0.08 * field(h, w, seed + 7, amp_exp=1.0)[..., None] * np.array([1.0, 0.0, -1.0])
    return np.clip(y[..., None] * tint, 0.01, 0.97)


def ramp_scene(h=H0, w=W0, lo=0.15, hi=0.6, axis=0, seed=0, ev_std=0.25):
    """하늘-땅 같은 한 방향 밝기 경사(비네팅 아님) + 고주파 질감."""
    t = np.linspace(0.0, 1.0, h if axis == 0 else w)
    prof = lo + (hi - lo) * t
    g = prof[:, None] * np.ones((1, w)) if axis == 0 else np.ones((h, 1)) * prof[None, :]
    tex = 2.0 ** (ev_std * hf_field(h, w, seed))
    return np.clip(np.repeat((g * tex)[..., None], 3, axis=2), 0.02, 0.95)


def mixed_scene(h=H0, w=W0, seed=0, n_patch=40, ev_std=0.35, patch_ev=0.25, soft=1.1):
    """전 스택 시험용: 저주파 없는 정상 질감 바탕(비네팅) 위에 평탄한 색 패치 n_patch 개(그레인·CA).
    패치는 질감의 고주파만 지우고 밝기는 평균 근처(±patch_ev EV)라서 방사 평균 로그 밝기가 거의 정상으로 남는다."""
    rng = np.random.default_rng(seed + 31)
    f = hf_field(h, w, seed)
    ev = ev_std * f
    col = np.ones((h, w, 3))
    for _ in range(n_patch):
        pw = int(rng.integers(40, 101))
        ph = int(rng.integers(40, 101))
        x0 = int(rng.integers(0, w - pw))
        y0 = int(rng.integers(0, h - ph))
        ev[y0:y0 + ph, x0:x0 + pw] = rng.normal(0.0, patch_ev)
        col[y0:y0 + ph, x0:x0 + pw] = rng.uniform(0.75, 1.25, 3)
    img = 0.3 * 2.0 ** ev[..., None] * col
    return np.clip(ndi.gaussian_filter(img, (soft, soft, 0), mode="nearest"), 0.01, 0.97)


def disc_cov(h, w, cx, cy, r):
    """원판 면적 피복률(가장자리 1 px 안티앨리어싱). (h,w) float64 [0,1]."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    return np.clip(r - d + 0.5, 0.0, 1.0)


def disc_scene(h=H0, w=W0, discs=((120.0, 100.0, 14.0, 3.0),), bg=0.06, bg_std=0.002, seed=0):
    """어두운 바탕 + HDR 원판(cx, cy, r, Y). Y>1 이면 센서에서 포화한다. float64 선형 (h,w,3)."""
    base = np.clip(bg * (1.0 + bg_std / bg * field(h, w, seed, amp_exp=1.0)), 0.01, 0.2)
    img = np.repeat(base[..., None], 3, axis=2)
    for (cx, cy, r, y) in discs:
        cov = disc_cov(h, w, cx, cy, r)[..., None]
        img = img * (1.0 - cov) + y * cov
    return img


def saturate(x):
    """센서 포화만: [0,1] 선형 클립(LUT 없음)."""
    return np.clip(np.asarray(x, np.float64), 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ 순방향 오라클
def with_vignette(x_lin, strength):
    """LNS-04 비네팅(선형 영역)."""
    g = sf.vignette_gain(x_lin.shape[0], x_lin.shape[1], strength).astype(np.float64)
    return x_lin * g[..., None]


def with_grain(x_lin, strength, radius_px, seed=0, frame=0):
    """LNS-04 그레인(표시값 영역) 후 선형으로 되돌린다. 입력 선형 [0,1]."""
    d = color.linear_to_display(np.clip(x_lin, 0, 1).astype(np.float32))
    y = sf.grain_field(d, strength=strength, radius_px=radius_px, seed=seed, frame=frame)
    return color.srgb_to_linear(y).astype(np.float64)


def quantize8(x_lin):
    """8비트 sRGB 로 반올림했다가 선형으로(실사 입력의 양자화)."""
    d = color.linear_to_display(np.clip(x_lin, 0, 1).astype(np.float32)).astype(np.float64)
    return color.srgb_to_linear((np.round(d * 255.0) / 255.0).astype(np.float32)).astype(np.float64)


def radial_scale_channel(ch, eps, cx, cy):
    """채널 내용을 중심에서 (1+eps)배 키운다(바깥 이동 = eps*r): out(p) = ch(c + (p-c)/(1+eps)). 3차 보간."""
    h, w = ch.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx = cx + (xx - cx) / (1.0 + eps)
    sy = cy + (yy - cy) / (1.0 + eps)
    return ndi.map_coordinates(ch, [sy, sx], order=3, mode="nearest")


def with_ca(x_lin, eps_r, eps_b, center=None):
    h, w = x_lin.shape[:2]
    cx, cy = ((w - 1) / 2.0, (h - 1) / 2.0) if center is None else center
    out = x_lin.copy()
    out[..., 0] = radial_scale_channel(x_lin[..., 0], eps_r, cx, cy)
    out[..., 2] = radial_scale_channel(x_lin[..., 2], eps_b, cx, cy)
    return out


def half_diag(h, w):
    return float(np.hypot((w - 1) / 2.0, (h - 1) / 2.0))


def with_ghosts(x_lin, source_xy, ks, radius=11.0, amp=0.12, tint=(0.8, 1.0, 0.7), center=None, soft=2.5):
    """광원–중심 직선 c + k(s-c) 위에 연한 원형 반점을 더한다. 반환 (영상, [(x,y), ...])."""
    h, w = x_lin.shape[:2]
    c = np.array(((w - 1) / 2.0, (h - 1) / 2.0) if center is None else center, np.float64)
    s = np.asarray(source_xy, np.float64)
    out = x_lin.copy()
    pos = []
    for k in ks:
        p = c + k * (s - c)
        pos.append((float(p[0]), float(p[1])))
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot(xx - p[0], yy - p[1])
        prof = np.clip((radius - d) / soft + 0.5, 0.0, 1.0)
        out = out + amp * prof[..., None] * np.asarray(tint)[None, None, :]
    return out, pos


def full_chain(x_lin, th, seed=0, frame=0, noise=True):
    """LNS-04 전체 센서·필름 체인 -> 표시값 -> 선형(= RD-01 이 내놓을 image_lin). lut='none' 이면 항등 룩."""
    d = sf.apply_sensor_film(x_lin.astype(np.float32), th, seed=seed, frame=frame, noise=noise)
    return color.srgb_to_linear(d).astype(np.float32)
