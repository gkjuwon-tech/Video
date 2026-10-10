"""RD-05 테스트 공용 도우미: 합성 c(z) 장면(LNS-02 coc_px 오라클), 선 모션 커널, 잡음 텍스처.
RD-05 구현을 오라클에 쓰지 않는다."""
import math

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from buc.lens.dof import coc_px

W = 1024
FOV_X_DEG = 40.0
F_PX = (W / 2) / math.tan(math.radians(FOV_X_DEG) / 2)
F_FF = 0.036 * F_PX / W            # m


def oracle_coc(z, n, zf, w=W, f_ff=F_FF):
    return coc_px(np.asarray(z, dtype=np.float64), focal_mm=f_ff * 1000.0, f_stop=n, focus_m=zf,
                  sensor_width_mm=36.0, width_px=w).astype(np.float64)


def scene(zf, n, *, seed=0, shape=(96, 128), zmin=0.4, zmax=60.0, noise=0.0, outlier=0.0):
    """(coc, depth) 2D 맵. 깊이는 로그 균등. noise=곱 잡음 sd, outlier=균등 이상치 비율."""
    rng = np.random.default_rng(seed)
    z = np.exp(rng.uniform(math.log(zmin), math.log(zmax), shape))
    c = oracle_coc(z, n, zf)
    if noise:
        c = c * np.clip(1.0 + noise * rng.standard_normal(shape), 0.0, None)
    if outlier:
        o = rng.random(shape) < outlier
        c = np.where(o, rng.uniform(0.0, c.max(), shape), c)
    return c, z


def stops(n_hat, n):
    return 2.0 * math.log2(n_hat / n)


def line_kernel(length, theta, ss=8):
    n = int(math.ceil(length)) + 3
    n += (n % 2 == 0)
    k = np.zeros((n, n))
    c = n // 2
    for t in np.linspace(-length / 2, length / 2, int(length * ss) + 1):
        x = c + t * math.cos(theta)
        y = c + t * math.sin(theta)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                k[y0 + dy, x0 + dx] += wy * wx
    return k / k.sum()


def noise_tex(h, w, seed=0, pad=32):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.standard_normal((h + 2 * pad, w + 2 * pad)), 0.7)


def motion_image(length, theta_deg, h=192, w=192, seed=0):
    pad = 32
    n = noise_tex(h, w, seed, pad)
    b = fftconvolve(n, line_kernel(length, math.radians(theta_deg)), mode="same")[pad:-pad, pad:-pad]
    return (b - b.min()) / (b.max() - b.min())


def ang_err_deg(a, b):
    d = abs((a - b) % math.pi)
    return math.degrees(min(d, math.pi - d))
