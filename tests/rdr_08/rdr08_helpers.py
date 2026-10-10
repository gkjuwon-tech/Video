"""RDR-08 테스트 공용 도우미: 카메라, 쿼터니언, 체적·가닥 생성, 구현 독립 오라클(numpy).

오라클은 numpy·math만 쓰고 RDR-08 구현을 참조하지 않는다."""
import math

import numpy as np
import torch


def K_of(fx, w, h, fy=None, cx=None, cy=None):
    fy = fx if fy is None else fy
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def pose(R=None, t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_axis(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def quat_from_R(R):
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return np.array(q)


def fiber_S(t, r):
    t = np.asarray(t, np.float64)
    t = t / np.linalg.norm(t)
    return np.eye(3) - np.outer(t, t) + r * r * np.outer(t, t)


def surface_S(n, r):
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    return np.outer(n, n) + r * r * (np.eye(3) - np.outer(n, n))


def make_vol(means, rots, scales, density, colors, sggx=None, dtype=torch.float64):
    """rots (N,3,3) 열=축, scales (N,3) m, density (N,) ρ, sggx (N,3,3) 또는 None(=I)."""
    from buc.render.vol import VolGaussians
    means = np.asarray(means, np.float64).reshape(-1, 3)
    n = means.shape[0]
    rots = np.asarray(rots, np.float64).reshape(n, 3, 3)
    quats = np.stack([quat_from_R(r) for r in rots])
    scales = np.asarray(scales, np.float64).reshape(n, 3)
    density = np.asarray(density, np.float64).reshape(n)
    colors = np.asarray(colors, np.float64).reshape(n, 3)
    sggx = np.repeat(np.eye(3)[None], n, 0) if sggx is None else np.asarray(sggx, np.float64).reshape(n, 3, 3)
    t = lambda a: torch.tensor(np.array(a, np.float64), dtype=dtype)  # noqa: E731
    return VolGaussians(means=t(means), quats=t(quats), log_scales=t(np.log(scales)), log_density=t(np.log(density)),
                        colors=t(colors), sggx=t(sggx))


def make_strands(points, radius, opacity, colors, dtype=torch.float64):
    from buc.render.vol import Strands
    t = lambda a: torch.tensor(np.array(a, np.float64), dtype=dtype)  # noqa: E731
    return Strands(points=t(points), radius=t(radius), opacity=t(opacity), colors=t(colors))


def at_pixel(K, u, v, z):
    """항등 카메라에서 화소 (u, v)에 투영되는 깊이 z의 점."""
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def ray(K, c2w, u, v):
    """(C, ω): 화소 (u, v) 광선의 세계 원점과 단위 방향."""
    c2w = np.asarray(c2w, np.float64)
    d = c2w[:3, :3] @ np.linalg.solve(K, np.array([u, v, 1.0]))
    return c2w[:3, 3].copy(), d / np.linalg.norm(d)


def tau_line_oracle(mu, Sigma, S, rho, C, w, n=200001, span=12.0):
    """수치 선적분: ∫ ρ σ_S(ω) exp(-½ (p-μ)ᵀΣ⁻¹(p-μ)) dt, p = C + tω (사다리꼴)."""
    Si = np.linalg.inv(Sigma)
    d = C - mu
    a = w @ Si @ w
    tstar = -(w @ Si @ d) / a
    half = span / math.sqrt(a)
    t = np.linspace(tstar - half, tstar + half, n)
    P = C[None] + t[:, None] * w[None] - mu[None]
    q = np.einsum("ni,ij,nj->n", P, Si, P)
    f = np.exp(-0.5 * q)
    sig = math.sqrt(w @ S @ w)
    integ = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    return rho * sig * integ, tstar


def Phi(x):
    x = np.asarray(x, np.float64)
    from scipy.special import ndtr
    return ndtr(x)


def strand_alpha_oracle(o, d, w, sig=0.5):
    return o * (Phi((d + w) / sig) - Phi((d - w) / sig))


def tau_of(alpha):
    return -np.log1p(-np.asarray(alpha, np.float64))


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, np.float64)
