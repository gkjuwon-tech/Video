"""CAM-07 테스트 공용 도우미. 구현 내부가 아니라 계약서(CAM-07.md)의 정의와 독립 참조식만 쓴다."""
import math

import numpy as np


def cl():
    from buc.camera import clearance
    return clearance


def rot(axis, deg):
    """Rodrigues 회전행렬(도). 구현과 무관한 독립 구현."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def pose(pos, R=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = pos
    return T


def poses(positions, R=None):
    return np.stack([pose(p, R) for p in positions])


def seg_point_dist(p, a, b):
    """점-선분 최단거리(독립 참조식)."""
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    s = 0.0 if ab @ ab == 0 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


def polyline_sphere_clearance(center, radius, pts):
    return min(seg_point_dist(center, pts[i], pts[i + 1]) for i in range(len(pts) - 1)) - radius


def ref_sphere_sdf(c, r, p):
    return np.linalg.norm(np.asarray(p, float) - c, axis=-1) - r


def ref_box_sdf_aabb(h, p):
    """축정렬 상자(중심 원점) 부호거리."""
    q = np.abs(np.asarray(p, float)) - np.asarray(h, float)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def box_surface_samples(center, h, R, n=61):
    """상자 표면의 균일 격자 표본(브루트포스 거리 대조용). 간격 <= 2*max(h)/(n-1)."""
    g = np.linspace(-1.0, 1.0, n)
    A, B = np.meshgrid(g, g, indexing="ij")
    A, B = A.ravel(), B.ravel()
    h = np.asarray(h, float)
    faces = []
    for ax in range(3):
        i, j = [k for k in range(3) if k != ax]
        for s in (-1.0, 1.0):
            P = np.zeros((A.size, 3))
            P[:, ax] = s * h[ax]
            P[:, i] = A * h[i]
            P[:, j] = B * h[j]
            faces.append(P)
    L = np.concatenate(faces)
    return L @ np.asarray(R, float).T + np.asarray(center, float)


def dense_min_sdf(sdf, a, b, n=20001):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float))
    v = sdf(pts)
    return float(v.min()), float(t[int(np.argmin(v)), 0])


def K_pinhole(fx, W, H):
    return np.array([[fx, 0, (W - 1) / 2.0], [0, fx, (H - 1) / 2.0], [0, 0, 1.0]])
