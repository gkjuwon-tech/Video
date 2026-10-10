"""RDR-04 테스트 공용 도우미: 결정론 시험 영상, 사인파 이득 측정, 호모그래피 생성.

여기 있는 모든 신호는 numpy 만으로 만든다(구현 모듈 `buc.render.filters` 를 참조하지 않는다)."""
import math

import numpy as np

N = 129                          # 사인파 시험 영상의 한 변(거울 경계와 정합: 주기 2(N-1) = 256)


def freq(m):
    """거울 경계에서 매끄럽게 이어지는 주파수(cy/px): cos(2π f x), f = m / (2(N-1))."""
    return m / (2.0 * (N - 1))


def cos_x(m, rows=8, ch=None, dtype=np.float32):
    """x 방향 cos 줄무늬 (rows, N) 또는 (rows, N, ch). 값 [-1, 1]."""
    row = np.cos(2 * np.pi * freq(m) * np.arange(N))
    a = np.tile(row, (rows, 1))
    if ch is not None:
        a = np.repeat(a[..., None], ch, axis=2)
    return a.astype(dtype)


def cos_y(m, cols=N, dtype=np.float32):
    """y 방향 cos 줄무늬 (N, cols)."""
    col = np.cos(2 * np.pi * freq(m) * np.arange(N))
    return np.tile(col[:, None], (1, cols)).astype(dtype)


def fit_amp(r, xs, f):
    """r(xs) ≈ a cos(2π f xs) + b sin(2π f xs) 의 최소제곱 진폭."""
    r = np.asarray(r, np.float64)
    A = np.stack([np.cos(2 * np.pi * f * xs), np.sin(2 * np.pi * f * xs)], axis=1)
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    return float(np.hypot(c[0], c[1]))


def rms_amp(r):
    """영평균 사인파 진폭 추정: std * sqrt(2)."""
    r = np.asarray(r, np.float64)
    return float(r.std() * math.sqrt(2.0))


def band_image(n=96, seed=3, fc=0.3, ch=3):
    """대역 제한(<= fc cy/px) 1/f 스펙트럼 잡음 영상 (n,n,ch) float32. 평균 0.5, 표준편차 0.15. 클램프 없음."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n, ch))
    F = np.fft.fft2(z, axes=(0, 1))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    r = np.hypot(fx, fy)
    mask = np.where(r <= fc, 1.0 / np.maximum(r * n, 1.0), 0.0) * np.clip((fc - r) / 0.03, 0.0, 1.0)
    im = np.fft.ifft2(F * mask[..., None], axes=(0, 1)).real
    im = (im - im.mean()) / im.std() * 0.15 + 0.5
    return im.astype(np.float32)


def translate_h(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def rot_h(deg, cx, cy, s=1.0):
    """(cx,cy) 둘레 deg 회전 + 등방 배율 s 의 호모그래피 (dst -> src 로 쓴다)."""
    a = math.radians(deg)
    c, sn = math.cos(a) * s, math.sin(a) * s
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    return translate_h(cx, cy) @ R @ translate_h(-cx, -cy)


def interior_rmse(a, b, margin=14):
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64))[margin:-margin, margin:-margin]
    return float(np.sqrt((d ** 2).mean()))


def smooth_tex(w=96, h=64):
    """저주파 RGB 텍스처(파장 >= 40 px), float32 (h,w,3), 값 [0.15, 0.85]."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rgb = np.empty((h, w, 3), np.float64)
    rgb[..., 0] = 0.5 + 0.25 * np.sin(2 * np.pi * u / 52.0) + 0.1 * (v / h - 0.5)
    rgb[..., 1] = 0.5 + 0.25 * np.cos(2 * np.pi * (u / 44.0 + v / 60.0))
    rgb[..., 2] = 0.2 + 0.6 * (u / w) * (0.6 + 0.4 * np.sin(2 * np.pi * v / 48.0))
    return rgb.astype(np.float32)
