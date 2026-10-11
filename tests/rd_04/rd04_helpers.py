"""RD-04 테스트 공용 도우미: 합성 장면, 순방향 오라클, 스펙트럼/톤 측정.

여기 있는 코드는 RD-04 구현(buc.read.explain_away)을 참조하지 않는다. numpy, scipy, FND-06 color, LNS-04/08 공개 순방향
(sensor_film, chain.look)과 RD-03 데이터 형태(screen_fx 데이터클래스)만 쓴다. 모든 장면은 시드로 결정되는 float64 선형 (H,W,3).
"""
import dataclasses

import numpy as np
from scipy import special

from buc.core import color
from buc.core import contracts as C
from buc.lens import chain
from buc.read import screen_fx as fxm

# ---- 계약서 상수(정본). 테스트는 이 값으로 기대치를 만들고, 모듈 상수가 같은지도 따로 검사한다.
OCTAVE_HI = (0.5, 0.25, 0.125, 0.0625)
H_MIN = 0.5
BAND_TAPER = 0.7
WIENER_K = 0.01
COC_MIN_PX = 0.5
GHOST_MIN_OCTAVE = 3
HALO_SOURCE_MAX = 16.0
X_DEFOCUS_HALF = 2.21509        # 2*J1(x)/x = 0.5 의 첫 근 (계약서 근사값; 테스트는 MTF 로 직접 재확인한다)
X_MOTION_HALF = 1.89549         # sin(x)/x = 0.5 의 근

H0, W0 = 96, 128                # 기본 시험 해상도(작게: 전체 스위트 60 s 예산)


def theta(**kw):
    return C.Theta(**kw)


# ------------------------------------------------------------------------------------------------ 오라클: 전달함수
def defocus_otf(nu, c):
    """원판(pillbox) 흐림의 부호 있는 OTF 2 J1(pi c nu)/(pi c nu). c=0 이면 1."""
    nu = np.asarray(nu, np.float64)
    x = np.pi * float(c) * np.abs(nu)
    xs = np.where(x < 1e-9, 1.0, x)
    return np.where(x < 1e-9, 1.0, 2.0 * special.j1(xs) / xs)


def band_gain(nu, c, k=WIENER_K, h_min=H_MIN):
    """계약서의 대역 제한 Wiener 이득 G(nu; c) (DC 정규화, 코사인 테이퍼). nu 배열(주기/px)."""
    nu = np.abs(np.asarray(nu, np.float64))
    h = np.abs(defocus_otf(nu, c))
    gw = (1.0 + k) * h / (h * h + k)
    nc = X_DEFOCUS_HALF / (np.pi * c) if h_min == H_MIN else None
    u = nu / nc
    t = np.where(u <= BAND_TAPER, 1.0, np.where(u >= 1.0, 0.0, 0.5 * (1.0 + np.cos(np.pi * (u - BAND_TAPER) / (1.0 - BAND_TAPER)))))
    return 1.0 + t * (gw - 1.0)


# ------------------------------------------------------------------------------------------------ 장면 생성기
def field(h, w, seed=0, amp_exp=1.0):
    """진폭 스펙트럼 nu^-amp_exp 인 무작위 위상 장(주기적). 평균 0, 표준편차 1. 결정론."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    nu = np.hypot(fx, fy)
    nu[0, 0] = 1.0
    spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) * nu ** (-amp_exp)
    spec[0, 0] = 0.0
    x = np.fft.ifft2(spec).real
    return (x - x.mean()) / x.std()


def texture_scene(h=H0, w=W0, seed=0, ev_std=0.3, amp_exp=1.0, mean=0.3):
    """주기적 로그정규 질감 (h,w,3). 색 기울기는 느린 장."""
    y = mean * 2.0 ** (ev_std * field(h, w, seed, amp_exp))
    tint = 1.0 + 0.05 * field(h, w, seed + 5, 1.0)[..., None] * np.array([1.0, 0.0, -1.0])
    return np.clip(y[..., None] * tint, 0.02, 0.95)


def tone_scene(h, w, nu0, angle=0.0, amp=0.12, mean=0.3, phase=0.4):
    """평균 mean 위의 순수 코사인(주파수 nu0 주기/px, 방향 angle). 반환 (h,w,3) 와 (fx, fy)."""
    fxx, fyy = nu0 * np.cos(angle), nu0 * np.sin(angle)
    yy, xx = np.mgrid[0:h, 0:w]
    s = mean + amp * np.cos(2.0 * np.pi * (fxx * xx + fyy * yy) + phase)
    return np.repeat(s[..., None], 3, axis=2), (fxx, fyy)


def blur_otf_periodic(x, c):
    """주기 경계의 푸리에 영역 원판 흐림(오라클 OTF). x (h,w[,3]). c 는 스칼라."""
    a = np.asarray(x, np.float64)
    h, w = a.shape[:2]
    nu = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    g = defocus_otf(nu, c)
    if a.ndim == 3:
        g = g[..., None]
        return np.fft.ifft2(np.fft.fft2(a, axes=(0, 1)) * g, axes=(0, 1)).real
    return np.fft.ifft2(np.fft.fft2(a) * g).real


def tone_amplitude(img, nu_xy, margin):
    """안쪽 (margin 제외) 영역에서 주어진 주파수 코사인 성분의 진폭(최소제곱). img (h,w) 또는 (h,w,3)(채널 평균)."""
    a = np.asarray(img, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sl = (slice(margin, h - margin), slice(margin, w - margin))
    ph = 2.0 * np.pi * (nu_xy[0] * xx + nu_xy[1] * yy)
    A = np.stack([np.cos(ph)[sl].ravel(), np.sin(ph)[sl].ravel(), np.ones(a[sl].size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, a[sl].ravel(), rcond=None)
    return float(np.hypot(coef[0], coef[1]))


# ------------------------------------------------------------------------------------------------ 순방향
def look_lin(x, th, seed=0, noise=False):
    """LNS-08 look -> 선형(= RD-01 이 내놓을 image_lin). float32."""
    d = chain.look(np.asarray(x, np.float32), th, seed=seed, noise=noise)
    return color.srgb_to_linear(d).astype(np.float32)


def disp(image_lin):
    """RD-04 가 쓰는 표시값 변환(계약서): clip -> float32 -> sRGB 부호화."""
    return color.linear_to_display(np.clip(np.asarray(image_lin), 0.0, 1.0).astype(np.float32))


def grain_fit(sigma_coef=0.08, floor_sigma=0.0, radius_px=0.75, **kw):
    """RD-03 GrainFit (dataclasses.replace 로 만든 손수 제작 값)."""
    base = fxm.identity_screen_fx(8, 8).grain
    return dataclasses.replace(base, sigma_coef=float(sigma_coef), floor_sigma=float(floor_sigma),
                               radius_px=float(radius_px), strength=min(1.0, float(sigma_coef) / 0.16), **kw)


def make_fx(h, w, grain=None, halation=None, ghost=None):
    """손수 제작 ScreenFx: identity 에 필요한 조각만 바꾼다."""
    f = fxm.identity_screen_fx(h, w)
    kw = {}
    if grain is not None:
        kw["grain"] = grain
    if halation is not None:
        kw["halation"] = dataclasses.replace(fxm.identity_screen_fx(h, w).halation, **halation)
    if ghost is not None:
        kw["ghost"] = dataclasses.replace(f.ghost, **ghost)
    return dataclasses.replace(f, **kw)


def disc_cov(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(r - np.hypot(xx - cx, yy - cy) + 0.5, 0.0, 1.0)


def patch_scene(h=H0, w=W0, seed=0, levels=(0.08, 0.2, 0.4, 0.65), band=6):
    """전체가 평탄한 사각 패치(표시값 수준이 다른 4구역)와 뚜렷한 에지. 반환 (선형 장면, 평탄 마스크(에지에서 band px 이상))."""
    x = np.zeros((h, w, 3))
    hh, hw = h // 2, w // 2
    lv = np.asarray(levels, np.float64)
    x[:hh, :hw] = lv[0]
    x[:hh, hw:] = lv[1]
    x[hh:, :hw] = lv[2]
    x[hh:, hw:] = lv[3]
    ys = np.abs(np.arange(h) - hh)
    xs = np.abs(np.arange(w) - hw)
    flat = (ys[:, None] > band) & (xs[None, :] > band)
    flat[:band, :] = False
    flat[-band:, :] = False
    flat[:, :band] = False
    flat[:, -band:] = False
    return x, flat


def rms(a):
    a = np.asarray(a, np.float64)
    return float(np.sqrt(np.mean(a * a)))
