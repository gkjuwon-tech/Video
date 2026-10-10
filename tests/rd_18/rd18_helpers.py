"""RD-18 테스트 공용 도구. numpy·scipy만, 지역 RNG만 쓴다(전역 RNG 불변).

- 합성 장면: 부드러운 원판(상수 F, 선형 기울기 B), 몸통 + 털 가닥, 곧은 경계.
- 오라클: 조밀 매팅 라플라시안(계약서 수식 그대로, 이중 루프), 조밀 하드 제약 α 해,
  F/B 2차 에너지(계약서 수식 그대로).
구현(buc.read.edge_unmix)은 각 테스트에서 import 한다.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BG, UNKNOWN, FG = 0, 128, 255
F_DISC = np.array([0.8, 0.35, 0.1])
F_FUR = np.array([0.75, 0.55, 0.3])


def disc_scene(h: int = 96, w: int = 96, r: float = 30.0, soft: float = 1.0):
    """부드러운 경계 원판. 반환 (I f64 (h,w,3), z (h,w), alpha_true, F_true (3,), B_true (h,w,3))."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.hypot(yy - h / 2 + 0.3, xx - w / 2 + 0.2) - r
    at = np.clip(0.5 - d / (2 * soft), 0.0, 1.0)
    bt = np.stack([0.1 + 0.5 * xx / w, 0.3 + 0.3 * yy / h, 0.6 - 0.3 * xx / w], -1)
    img = at[..., None] * F_DISC + (1 - at[..., None]) * bt
    z = np.where(d < 0, 3.0, 6.0)
    return img, z, at, F_DISC.copy(), bt


def fur_scene(h: int = 72, w: int = 96):
    """왼쪽 몸통(x<30) + 오른쪽으로 뻗은 1 px 남짓 털 가닥(α 최대 0.85).
    반환 (I, z_full, z_coarse4 (body만, stride 4), alpha_true, F_true, B_true, thin_mask, xx)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    at = np.clip(30.5 - xx, 0.0, 1.0)
    for c in range(6, 70, 9):
        d = np.abs(yy - (c + 0.12 * (xx - 30)))
        s = np.clip(1.1 - d, 0.0, 1.0) * 0.85 * (xx >= 29) * (xx < 80)
        at = np.maximum(at, s)
    bt = np.stack([0.05 + 0.2 * xx / w, 0.25 + 0.2 * yy / h, 0.5 - 0.2 * xx / w], -1)
    img = at[..., None] * F_FUR + (1 - at[..., None]) * bt
    z = np.where(at > 0.45, 2.0, 5.0)
    zc = np.where(xx[::4, ::4] < 30, 2.0, 5.0)
    thin = ndi.binary_dilation((at > 0.05) & (xx >= 30), np.ones((3, 3), bool))
    return img, z, zc, at, F_FUR.copy(), bt, thin, xx


def step_scene(h: int = 40, w: int = 48, col: int = 20, soft: float = 0.8):
    """세로 경계(x=col+0.5)의 왼쪽 전경. 좌우 대칭 등변 시험용(EDT 동률 없음)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    at = np.clip(0.5 - (xx - col - 0.5) / (2 * soft), 0.0, 1.0)
    bt = np.stack([0.2 + 0.3 * yy / h, 0.6 - 0.2 * yy / h, 0.3 + 0.1 * xx / w], -1)
    img = at[..., None] * np.array([0.9, 0.2, 0.4]) + (1 - at[..., None]) * bt
    z = np.where(xx <= col, 2.0, 4.0)
    return img, z, at


def dense_laplacian(img: np.ndarray, eps: float = 1e-7, r: int = 1, centers=None) -> np.ndarray:
    """계약서 수식의 직역(느린 루프). 작은 영상 전용."""
    h, w, _ = img.shape
    n = (2 * r + 1) ** 2
    L = np.zeros((h * w, h * w))
    for cy in range(r, h - r):
        for cx in range(r, w - r):
            if centers is not None and not centers[cy, cx]:
                continue
            ids = [(y * w + x) for y in range(cy - r, cy + r + 1) for x in range(cx - r, cx + r + 1)]
            X = np.array([img[i // w, i % w] for i in ids], dtype=np.float64)
            mu = X.mean(0)
            Xc = X - mu
            S = Xc.T @ Xc / n
            Si = np.linalg.inv(S + (eps / n) * np.eye(3))
            for a in range(n):
                for b in range(n):
                    L[ids[a], ids[b]] += (1.0 if a == b else 0.0) - (1.0 + Xc[a] @ Si @ Xc[b]) / n
    return L


def dense_alpha(img: np.ndarray, tri: np.ndarray, eps: float = 1e-7, r: int = 1, ridge: float = 1e-9) -> np.ndarray:
    """전체 라플라시안으로 푼 하드 제약 α(clip 전)."""
    L = dense_laplacian(img, eps, r)
    t = tri.ravel()
    u = np.flatnonzero(t == UNKNOWN)
    k = np.flatnonzero(t != UNKNOWN)
    b = (t == FG).astype(np.float64)
    a = b.copy()
    if len(u):
        a[u] = np.linalg.solve(L[np.ix_(u, u)] + ridge * np.eye(len(u)), -L[np.ix_(u, k)] @ b[k])
    return a.reshape(tri.shape)


def omega(tri: np.ndarray) -> np.ndarray:
    U = tri == UNKNOWN
    return ndi.binary_dilation(U, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))


def fb_energy(img, alpha, tri, F, B, gamma=0.1, delta=1e-6) -> float:
    """계약서의 F/B 에너지(세 채널 합). F,B는 (H,W,3) 전체 배열(고정값 포함)."""
    om = omega(tri)
    fgk = tri == FG
    bgk = tri == BG
    a = alpha[..., None]
    data = ((a * F + (1 - a) * B - img) ** 2)[om].sum()
    sm = 0.0
    for X in (F, B):
        for dy, dx in ((0, 1), (1, 0)):
            h, w = om.shape
            pair = om[: h - dy, : w - dx] & om[dy:, dx:]
            diff = X[: h - dy, : w - dx] - X[dy:, dx:]
            sm += (diff ** 2)[pair].sum()
    rid = ((F - img) ** 2)[om & ~fgk].sum() + ((B - img) ** 2)[om & ~bgk].sum()
    return float(data + gamma * sm + delta * rid)


def composite(alpha, fg, bg):
    a = np.asarray(alpha, np.float64)[..., None]
    return a * np.asarray(fg, np.float64) + (1 - a) * np.asarray(bg, np.float64)
