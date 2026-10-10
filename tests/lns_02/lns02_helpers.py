"""LNS-02 테스트 공용 도우미. 오라클은 계약서 수식과 numpy/scipy만 쓰고 buc.lens.dof 내부를 참조하지 않는다."""
import math

import numpy as np
from scipy.signal import fftconvolve

# 기준 렌즈: 85 mm f/1.4, 풀프레임 폭 36 mm, 영상 폭 128 px
FOCAL, FSTOP, SENSOR = 85.0, 1.4, 36.0
W, H = 128, 64
LENS = dict(focal_mm=FOCAL, f_stop=FSTOP, sensor_width_mm=SENSOR)
Z_FAR = 1000.0  # 먼 평면(무한대 근사)


def coc_ref(z, focus_m, focal_mm=FOCAL, f_stop=FSTOP, sensor_width_mm=SENSOR, width_px=W, signed=False):
    """계약서 수식: c_mm = f^2/(N (z_f - f)) * |z - z_f| / z  (mm 단위 일관), px = c_mm * width_px / sensor."""
    z = np.asarray(z, np.float64)
    f = focal_mm
    zf = focus_m * 1000.0
    k = f * f / (f_stop * (zf - f)) * width_px / sensor_width_mm
    with np.errstate(invalid="ignore"):
        r = np.where(np.isinf(z), 1.0, (z - focus_m) / np.where(np.isinf(z), 1.0, z))
    c = k * r
    return c if signed else np.abs(c)


def focus_for_coc(c_target, z_plane=Z_FAR, **lens):
    """평면(z_plane)이 초점 앞(z_f < z_plane)에 있을 때 CoC=c_target 이 되는 focus_m (이분법)."""
    fl = lens.get("focal_mm", FOCAL)
    lo, hi = fl / 1000.0 * 1.0001, z_plane
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(coc_ref(z_plane, mid, **lens)) > c_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def disk_kernel(d, ss=8):
    """이상적 원형 조리개(지름 d px), 부화소 8x8 적분, 합 1."""
    r = d / 2.0
    h = int(math.ceil(r)) + 1
    o = (np.arange(ss) + 0.5) / ss - 0.5
    y, x = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
    X = x[..., None, None] + o[None, None, None, :]
    Y = y[..., None, None] + o[None, None, :, None]
    k = ((X * X + Y * Y) <= r * r).mean(axis=(2, 3))
    return k / k.sum()


def conv_ref(img, d):
    k = disk_kernel(d)
    return np.stack([fftconvolve(img[..., c], k, mode="same") for c in range(img.shape[2])], -1)


def tex(w=W, h=H, period=(14.0, 11.0, 23.0)):
    """중주파 결정론 텍스처, 선형 rgb [0.1, 0.9]."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty((h, w, 3))
    out[..., 0] = 0.5 + 0.2 * np.sin(2 * np.pi * u / period[0]) + 0.2 * np.sin(2 * np.pi * v / period[1])
    out[..., 1] = 0.5 + 0.4 * np.sin(2 * np.pi * (u + v) / period[2])
    out[..., 2] = 0.5 + 0.4 * np.cos(2 * np.pi * (u - 0.5 * v) / period[1])
    return out.astype(np.float32)


def stripes(w=W, h=H, period=4):
    u = np.arange(w)[None, :].repeat(h, 0)
    g = np.where((u // (period // 2)) % 2 == 0, 0.2, 0.8)
    return np.stack([g, g, g], -1).astype(np.float32)


def second_moment(img2d):
    """질량 중심 기준 E[r^2]."""
    a = np.asarray(img2d, np.float64)
    m = a.sum()
    y, x = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float64)
    cx, cy = (a * x).sum() / m, (a * y).sum() / m
    return float((a * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / m), cx, cy


def grad_energy(img, mask=None):
    g = np.asarray(img, np.float64).mean(-1)
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    e = gx + gy
    if mask is not None:
        e = e[mask[:-1, :-1]]
    return float(e.mean())
