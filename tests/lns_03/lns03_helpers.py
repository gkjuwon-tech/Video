"""LNS-03 테스트 공용 도우미. 오라클은 계약서 수식과 buc.core(geom·pyramid)만 쓰고 buc.lens.shutter 를 참조하지 않는다."""
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from buc.core import geom
from buc.core.pyramid import sample_bilinear

W, H = 96, 64
FX = 120.0
K = np.array([[FX, 0.0, (W - 1) / 2.0], [0.0, FX, (H - 1) / 2.0], [0.0, 0.0, 1.0]])
FPS = 24.0
T3 = np.array([0.0, 1.0, 2.0]) / FPS  # 궤적 시각, 중간 프레임 t_mid = T3[1]


def tex(w=W, h=H, seed=0, sigma=2.0):
    """부드러운 결정론 텍스처 (h,w,3) float32, 값 [0.1, 0.9]."""
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.standard_normal((h, w, 3)), sigma=(sigma, sigma, 0))
    a = (a - a.min()) / (a.max() - a.min())
    return (0.1 + 0.8 * a).astype(np.float32)


def pose_rot(rvec_rate, tau, trans_rate=(0.0, 0.0, 0.0)):
    """해석적 등속 궤적: 회전 rodrigues(rate*tau), 위치 trans_rate*tau."""
    T = np.eye(4)
    T[:3, :3] = geom.rodrigues(np.asarray(rvec_rate, np.float64) * tau)
    T[:3, 3] = np.asarray(trans_rate, np.float64) * tau
    return T


def traj(rvec_rate=(0.0, 0.0, 0.0), trans_rate=(0.0, 0.0, 0.0), t=T3):
    return np.stack([pose_rot(rvec_rate, ti, trans_rate) for ti in t]), np.asarray(t, np.float64)


def exposure(shutter_deg, fps=FPS):
    return shutter_deg / (360.0 * fps)


def window(t_mid, T, centered=True):
    return (t_mid - T / 2, t_mid + T / 2) if centered else (t_mid, t_mid + T)


def homog(K_, c2w_mid, c2w_tau):
    return K_ @ c2w_tau[:3, :3].T @ c2w_mid[:3, :3] @ np.linalg.inv(K_)


def rot_oracle(img, K_, pose_fn, t0, t1, n=256, row_offsets=None, border="edge"):
    """순수 회전 셔터 적분 오라클: out(p) = mean_k img(H(tau_k)^-1 p), tau_k 중점 층화. 행별 시각 오프셋 지원."""
    h, w = img.shape[:2]
    uv = geom.pixel_grid(h, w)
    mid = pose_fn(0.5 * (t0 + t1)) if row_offsets is None else None
    acc = np.zeros(img.shape, np.float64)
    taus = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
    for tau in taus:
        if row_offsets is None:
            Hi = np.linalg.inv(homog(K_, mid, pose_fn(tau)))
            q = geom.apply_homography(Hi, uv)
        else:
            q = np.empty_like(uv)
            for y in range(h):
                Hi = np.linalg.inv(homog(K_, row_offsets[1], pose_fn(tau + row_offsets[0][y])))
                q[y] = geom.apply_homography(Hi, uv[y])
        acc += sample_bilinear(img, q, border=border)
    return (acc / n).astype(np.float32)


def blur_oracle(K_, c2w_mid, pose_fn, t0, t1, depth=None, h=H, w=W, row_off=None):
    """blur_px 오라클: 세계점(깊이 있으면 역투영, 없으면 무한원점)을 창 끝점 자세로 project 한 차."""
    uv = geom.pixel_grid(h, w)
    off = np.zeros(h) if row_off is None else row_off
    out = np.zeros((h, w, 2))
    for y in range(h):
        a, b = pose_fn(t0 + off[y]), pose_fn(t1 + off[y])
        if depth is None:
            d = np.concatenate([uv[y], np.ones((w, 1))], -1) @ np.linalg.inv(K_).T @ c2w_mid[:3, :3].T
            X = c2w_mid[:3, 3] + 1e6 * d  # 사실상 무한원점
        else:
            Xc = geom.unproject(K_, uv[y], depth[y])
            X = Xc @ c2w_mid[:3, :3].T + c2w_mid[:3, 3]
        pa, _ = geom.project(K_, a[:3, :3].T, -a[:3, :3].T @ a[:3, 3], X)
        pb, _ = geom.project(K_, b[:3, :3].T, -b[:3, :3].T @ b[:3, 3], X)
        out[y] = pb - pa
    return out


def moments(img2d):
    """(질량, 중심 (u,v), 분산 (u,v))."""
    m = float(img2d.sum())
    v, u = np.mgrid[0:img2d.shape[0], 0:img2d.shape[1]]
    cu, cv = float((img2d * u).sum() / m), float((img2d * v).sum() / m)
    vu = float((img2d * (u - cu) ** 2).sum() / m)
    vv = float((img2d * (v - cv) ** 2).sum() / m)
    return m, (cu, cv), (vu, vv)


def yaw_len(omega, T, fx=FX):
    """주점 근처 요 블러 길이(px): 2 fx tan(omega T / 2)."""
    return 2.0 * fx * math.tan(omega * T / 2.0)
