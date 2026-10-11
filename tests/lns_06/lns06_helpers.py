"""LNS-06 테스트 공용 도우미. 오라클은 계약서 수식과 numpy/scipy 만 쓰고 buc.lens.prescriptions 내부를 참조하지 않는다.

좌표 규약(LNS-05 와 같다): 화소 중심이 정수 (x=열, y=행), 각도는 (cos t, sin t) = (x, y) 방향(y 아래), 도 단위.
"""
import math

import numpy as np
from scipy import ndimage

from buc.lens import optics


def half_diag(h, w):
    return math.hypot(w - 1, h - 1) / 2.0


def grid(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return x, y


def smooth_img(h, w, dtype=np.float32):
    x, y = grid(h, w)
    r = 0.5 + 0.18 * np.sin(2 * np.pi * x / 53.0) + 0.17 * np.sin(2 * np.pi * y / 47.0 + 0.5)
    g = 0.5 + 0.18 * np.sin(2 * np.pi * (x + y) / 61.0 + 1.0) + 0.17 * np.cos(2 * np.pi * x / 43.0)
    b = 0.5 + 0.18 * np.cos(2 * np.pi * (x - y) / 57.0) + 0.17 * np.sin(2 * np.pi * y / 41.0 + 2.0)
    return np.stack([r, g, b], axis=-1).astype(dtype)


def hdr_scene(h=72, w=96, seed=3):
    """선형 HDR 장면: 어두운 바탕 + 포화 광원 몇 개 + 매끈한 텍스처."""
    rng = np.random.default_rng(seed)
    img = 0.03 + 0.05 * smooth_img(h, w, np.float64)
    x, y = grid(h, w)
    for _ in range(4):
        x0, y0 = rng.uniform(10, w - 10), rng.uniform(10, h - 10)
        amp = rng.uniform(4, 20)
        img += (amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * 1.5 ** 2)))[..., None] * np.array([1.0, 0.9, 0.8])
    return img.astype(np.float32)


def noise_img(h, w, seed=0, lo=0.1, hi=0.7, dtype=np.float32):
    """백색잡음 영상(고주파 에너지 측정용). 값 < 0.8 이라 고스트·별·스트릭 임계 밖."""
    rng = np.random.default_rng(seed)
    return (lo + (hi - lo) * rng.random((h, w, 3))).astype(dtype)


def impulse(h, w, x, y, val, dtype=np.float32, base=0.0):
    img = np.full((h, w, 3), base, dtype)
    img[int(y), int(x), :] = val
    return img


# ------------------------------------------------------------------------------------------------ 통계
def cov2(k, about=None):
    """(x,y) 2x2 이차 모멘트 (질량 정규화). about 이 None 이면 배열 중심."""
    k = np.asarray(k, np.float64)
    x, y = grid(*k.shape)
    s = k.sum()
    cx, cy = about if about is not None else ((k.shape[1] - 1) / 2.0, (k.shape[0] - 1) / 2.0)
    dx, dy = x - cx, y - cy
    return np.array([[(k * dx * dx).sum(), (k * dx * dy).sum()],
                     [(k * dx * dy).sum(), (k * dy * dy).sum()]]) / s


def centroid(k):
    k = np.asarray(k, np.float64)
    x, y = grid(*k.shape)
    s = k.sum()
    return float((k * x).sum() / s), float((k * y).sum() / s)


def minor_axis_deg(c):
    """공분산의 작은 고유값 방향 (도, [0,180))."""
    _, v = np.linalg.eigh(c)
    vx, vy = v[:, 0]
    return math.degrees(math.atan2(vy, vx)) % 180.0


def ang_diff180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def second_moment(k):
    c = cov2(k)
    return float(c[0, 0] + c[1, 1])


def ring_mass(k):
    """커널 가장자리 한 겹(맨 바깥 행·열)의 질량."""
    k = np.asarray(k, np.float64)
    return float(k[0].sum() + k[-1].sum() + k[1:-1, 0].sum() + k[1:-1, -1].sum())


def vertex_phase_deg(k, n_fold, r_lo=0.75):
    """n각형 커널의 꼭짓점 방향 위상(도): 바깥 고리(반지름 r_lo..1 · k 반지름)에서 질량 가중 n·θ 원형 평균 / n."""
    k = np.asarray(k, np.float64)
    half = (k.shape[0] - 1) / 2.0
    x, y = grid(*k.shape)
    dx, dy = x - half, y - half
    rho = np.hypot(dx, dy)
    m = (rho >= r_lo * half) & (rho <= 1.05 * half)
    th = np.arctan2(dy, dx)
    z = (k[m] * np.exp(1j * n_fold * th[m])).sum()
    return math.degrees(np.angle(z)) / n_fold


# ------------------------------------------------------------------------------------------------ 오라클
def streak_ref(x, gain, length_frac, tint, threshold):
    """계약서 수식: E=max(x-thr,0); 행마다 h(k)=exp(-|k|/L)/Z (k=-(W-1)..W-1, L=length_frac·W) 와 합성,
    세로 3탭 [.25,.5,.25] (밖은 0), 마지막에 gain·tint."""
    x = np.asarray(x, np.float64)
    h, w = x.shape[:2]
    E = np.maximum(x - threshold, 0.0)
    L = length_frac * w
    k = np.arange(-(w - 1), w, dtype=np.float64)
    hk = np.exp(-np.abs(k) / L)
    hk /= hk.sum()
    H1 = np.empty_like(E)
    for yy in range(h):
        for c in range(3):
            H1[yy, :, c] = np.convolve(E[yy, :, c], hk, mode="full")[w - 1:2 * w - 1]
    V = 0.5 * H1
    V[1:] += 0.25 * H1[:-1]
    V[:-1] += 0.25 * H1[1:]
    return gain * np.asarray(tint, np.float64) * V


def edge_ref(x, amount, sigma_frac=0.015, r_inner=0.35):
    """계약서 수식: out=(1-w)x + w·G(x), w=clip((r-r_inner)/(1-r_inner),0,1)², σ=amount·sigma_frac·W,
    G=가우시안(mode 'nearest', truncate 4, 채널 독립)."""
    x = np.asarray(x, np.float64)
    h, w = x.shape[:2]
    R = half_diag(h, w)
    xx, yy = grid(h, w)
    r = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0) / R
    wt = np.clip((r - r_inner) / (1.0 - r_inner), 0.0, 1.0) ** 2
    sig = amount * sigma_frac * w
    g = ndimage.gaussian_filter(x, sigma=(sig, sig, 0), mode="nearest", truncate=4.0)
    return (1.0 - wt[..., None]) * x + wt[..., None] * g


def hf_energy(img, r_lo, r_hi):
    """영상(H,W,3) 의 정규화 반지름 [r_lo, r_hi) 환 안 가로 인접차 제곱 평균(고주파 에너지)."""
    h, w = img.shape[:2]
    R = half_diag(h, w)
    xx, yy = grid(h, w)
    r = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0) / R
    d = np.zeros((h, w))
    d[:, :-1] = ((img[:, 1:, :].astype(np.float64) - img[:, :-1, :]) ** 2).mean(axis=2)
    m = (r >= r_lo) & (r < r_hi)
    m[:, -1] = False
    return float(d[m].mean())


def catseye_area_frac(u):
    """원형 조리개 두 개(지름 d)를 o=0.5·u·d 만큼 어긋나게 겹친 교집합 넓이 / 원 넓이."""
    q = 0.5 * u
    return (2.0 / math.pi) * (math.acos(q) - q * math.sqrt(1.0 - q * q))


def L1(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum())


def patch_kernel(out, x, y, half, ch=0):
    """출력에서 (x,y) 중심 (2half+1)² 패치를 합 1 로 정규화해 돌려준다."""
    p = out[y - half:y + half + 1, x - half:x + half + 1, ch].astype(np.float64)
    return p / p.sum()

