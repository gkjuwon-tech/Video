"""LNS-05 테스트 공용 도우미. 오라클은 계약서 수식과 numpy/scipy 만 쓰고 buc.lens.optics 내부를 참조하지 않는다.

좌표 규약: 픽셀 중심이 정수 (x=열, y=행), 각도는 (cos t, sin t) = (x, y) 방향(y 아래), 도 단위.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


def half_diag(h, w):
    """모서리 화소 중심까지의 거리 R (정규화 반지름 r = 거리 / R)."""
    return math.hypot(w - 1, h - 1) / 2.0


def center(h, w):
    return (w - 1) / 2.0, (h - 1) / 2.0


def grid(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return x, y


# --------------------------------------------------------------------------------------------- 장면
def smooth_fn(x, y):
    """해석적 매끈한 RGB 장면 (주기 >= 40 px, 값 [0.1, 0.9]). 쌍선형 표본 오차 < 5e-3."""
    r = 0.5 + 0.18 * np.sin(2 * np.pi * x / 53.0) + 0.17 * np.sin(2 * np.pi * y / 47.0 + 0.5)
    g = 0.5 + 0.18 * np.sin(2 * np.pi * (x + y) / 61.0 + 1.0) + 0.17 * np.cos(2 * np.pi * x / 43.0)
    b = 0.5 + 0.18 * np.cos(2 * np.pi * (x - y) / 57.0) + 0.17 * np.sin(2 * np.pi * y / 41.0 + 2.0)
    return np.stack([r, g, b], axis=-1)


def smooth_img(h, w, dtype=np.float32):
    x, y = grid(h, w)
    return smooth_fn(x, y).astype(dtype)


def blob(h, w, x0, y0, sigma=2.0, amp=1.0):
    x, y = grid(h, w)
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def impulse(h, w, x, y, val, dtype=np.float32, base=0.0):
    img = np.full((h, w, 3), base, dtype)
    img[int(y), int(x), :] = val
    return img


def hdr_scene(h=72, w=96, seed=3):
    """선형 HDR 장면: 어두운 바탕 + 포화 광원 몇 개 + 매끈한 텍스처."""
    rng = np.random.default_rng(seed)
    img = 0.03 + 0.05 * smooth_img(h, w, np.float64)
    for _ in range(4):
        x0, y0 = rng.uniform(10, w - 10), rng.uniform(10, h - 10)
        img += blob(h, w, x0, y0, sigma=1.5, amp=rng.uniform(4, 20))[..., None] * np.array([1.0, 0.9, 0.8])
    return img.astype(np.float32)


# --------------------------------------------------------------------------------------------- 통계
def centroid(img2d):
    x, y = grid(*img2d.shape)
    s = float(img2d.sum())
    return float((img2d * x).sum() / s), float((img2d * y).sum() / s)


def cov2(img2d, about=None):
    """(x,y) 2x2 이차 모멘트 행렬 (질량 정규화), 중심 기준."""
    x, y = grid(*img2d.shape)
    s = float(img2d.sum())
    cx, cy = about if about is not None else centroid(img2d)
    dx, dy = x - cx, y - cy
    return np.array([[(img2d * dx * dx).sum(), (img2d * dx * dy).sum()],
                     [(img2d * dx * dy).sum(), (img2d * dy * dy).sum()]]) / s


def minor_axis_deg(c):
    """공분산의 작은 고유값 방향 (도, [0,180))."""
    w, v = np.linalg.eigh(c)
    vx, vy = v[:, 0]
    return math.degrees(math.atan2(vy, vx)) % 180.0


def aniso(c):
    w = np.linalg.eigvalsh(c)
    return float(w[1] / w[0])


def ang_diff180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def ang_diff360(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def second_moment(k):
    h = (k.shape[0] - 1) // 2
    ax = np.arange(-h, h + 1, dtype=np.float64)
    r2 = ax[:, None] ** 2 + ax[None, :] ** 2
    return float((k * r2).sum() / k.sum())


# --------------------------------------------------------------------------------------------- 왜곡 오라클
def solve_rd(rq, k1, k2):
    """r_d (1 + k1 r_d^2 + k2 r_d^4) = r_q 를 이분법으로 푼다 (r_d >= 0, 단조 구간)."""
    lo, hi = 0.0, max(2.0 * rq, 1e-9)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * (1 + k1 * mid ** 2 + k2 * mid ** 4) < rq:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ridge_x(img2d, row, x_guess, half=9):
    """행 `row` 에서 x_guess 주변 창의 가중 중심 x."""
    x0 = int(round(x_guess))
    lo, hi = max(0, x0 - half), min(img2d.shape[1], x0 + half + 1)
    seg = img2d[row, lo:hi].astype(np.float64)
    return float((seg * np.arange(lo, hi)).sum() / seg.sum())


# --------------------------------------------------------------------------------------------- 베일링 글레어 오라클
def glare_ref(x, amount, r0_frac=0.03, exponent=1.5):
    """계약서 수식: out = (1-a) x + a (h*x)/(h*1), h(r) = (1+(r/r0)^2)^-1.5, r0 = r0_frac * W (px)."""
    x = np.asarray(x, np.float64)
    h, w = x.shape[:2]
    r0 = r0_frac * w
    yy, xx = np.mgrid[-(h - 1):h, -(w - 1):w].astype(np.float64)
    hk = (1.0 + (xx * xx + yy * yy) / (r0 * r0)) ** (-exponent)
    norm = fftconvolve(np.ones((h, w)), hk, mode="same")
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        out[..., c] = (1 - amount) * x[..., c] + amount * fftconvolve(x[..., c], hk, mode="same") / norm
    return out


# --------------------------------------------------------------------------------------------- 별(스파이크) 분석
def ring_profile(img2d, cx, cy, r0, r1, nang=720):
    """반경 [r0, r1) 띠의 각도 프로필 (r^2 가중 합: 방사 감쇠 보정)."""
    ang = np.arange(nang) * 360.0 / nang
    prof = np.zeros(nang)
    for r in np.arange(r0, r1, 1.0):
        xs = cx + r * np.cos(np.radians(ang))
        ys = cy + r * np.sin(np.radians(ang))
        prof += ndimage.map_coordinates(img2d.astype(np.float64), [ys, xs], order=1) * r ** 2
    return ang, prof


def ring_peaks(ang, prof, sigma_deg=3.0, thr=0.3):
    n = len(prof)
    step = 360.0 / n
    p = ndimage.gaussian_filter1d(prof, sigma_deg / step, mode="wrap")
    p = p / p.max()
    loc = [i for i in range(n) if p[i] >= p[i - 1] and p[i] > p[(i + 1) % n] and p[i] > thr]
    return [float(ang[i]) for i in loc]


def expected_rays(blades, rot_deg):
    """다각형 조리개 회절 별의 광선 각도(도, 정렬): 변 법선 {rot + (2i+1)*180/N} 과 그 반대 방향의 합집합."""
    normals = [(rot_deg + (2 * i + 1) * 180.0 / blades) % 360.0 for i in range(blades)]
    allr = normals + [(a + 180.0) % 360.0 for a in normals]
    out = []
    for a in sorted(allr):
        if not out or ang_diff360(a, out[-1]) > 1e-6:
            out.append(a)
    if len(out) > 1 and ang_diff360(out[0], out[-1]) < 1e-6:
        out.pop()
    return out


def match_rays(found, expected, tol):
    """found 와 expected 가 개수가 같고 각 expected 에 tol(도) 안의 found 가 하나씩 대응하는가."""
    if len(found) != len(expected):
        return False
    return all(min(ang_diff360(f, e) for f in found) <= tol for e in expected)
