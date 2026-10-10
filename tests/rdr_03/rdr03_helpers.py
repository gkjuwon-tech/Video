"""RDR-03 테스트 공용 도우미: 카메라, 쿼터니언, TSQ 생성, 독립 numpy 오라클(래스터·바이리니어·BC7 모드 6·BC4).

오라클은 numpy와 FND-06 geom만 쓰고 RDR-03 구현(buc.render.tsq)을 참조하지 않는다. 계약서 RDR-03.md의
'래스터 의미'·'BC 규약'을 그대로 옮긴 것이다."""
import math

import numpy as np
import torch

from buc.core import geom

PAGE = 8
ALPHA_MAX = 0.999
ALPHA_MIN = 1.0 / 255.0
ALPHA_OUT_EPS = 1e-3
DENOM_EPS = 1e-6
BC7_W4 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], np.int64)


# ------------------------------------------------------------------------------------------------ 카메라·회전
def K_of(fx, w, h, fy=None, cx=None, cy=None):
    fy = fx if fy is None else fy
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rot_axis(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    return geom.rodrigues(a * math.radians(deg))


def pose(R=None, t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def quat_from_R(R):
    """회전행렬 -> 단위 쿼터니언 (w, x, y, z). R(q)의 열 = (t_u, t_v, n)."""
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


def logit(p):
    p = np.asarray(p, np.float64)
    return np.log(p / (1 - p))


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------------------------------------ 장면
class Scene:
    """numpy float64 TSQ 장면 (오라클 입력이자 make_tsq 입력)."""

    def __init__(self, means, rots, scales, opac, pages, edge_n):
        self.means = np.asarray(means, np.float64).reshape(-1, 3)
        n = self.means.shape[0]
        self.rots = np.asarray(rots, np.float64).reshape(n, 3, 3)
        self.scales = np.asarray(scales, np.float64).reshape(n, 2)
        self.opac = np.asarray(opac, np.float64).reshape(n)
        self.pages = np.asarray(pages, np.float64).reshape(n, PAGE, PAGE, 3)
        self.edge_n = np.broadcast_to(np.asarray(edge_n, np.float64), (n,)).copy()

    @property
    def n(self):
        return self.means.shape[0]


def make_tsq(sc: Scene, dtype=torch.float64):
    from buc.render.tsq import TSQ
    t = lambda a: torch.tensor(np.asarray(a), dtype=dtype)  # noqa: E731
    quats = np.stack([quat_from_R(r) for r in sc.rots]) if sc.n else np.zeros((0, 4))
    return TSQ(means=t(sc.means), quats=t(quats), log_scales=t(np.log(sc.scales)),
               opacity_logits=t(logit(sc.opac)), edge_n=t(sc.edge_n), pages=t(sc.pages))


def make_surfels_from(sc: Scene, dtype=torch.float64):
    """상수 페이지 장면을 RDR-01 Surfels로 (색 = 페이지[0,0])."""
    from buc.render.splat2d import Surfels
    t = lambda a: torch.tensor(np.asarray(a), dtype=dtype)  # noqa: E731
    quats = np.stack([quat_from_R(r) for r in sc.rots])
    return Surfels(means=t(sc.means), quats=t(quats), log_scales=t(np.log(sc.scales)),
                   opacity_logits=t(logit(sc.opac)), colors=t(sc.pages[:, 0, 0, :]))


def random_scene(n, seed, *, edge_lo=2.0, edge_hi=12.0, const_pages=False, depth=(2.0, 4.0), spread=0.5,
                 scale=(0.08, 0.3), opac=(0.3, 0.95), tilt=60.0):
    rng = np.random.default_rng(seed)
    means = np.c_[rng.uniform(-spread, spread, n), rng.uniform(-0.75 * spread, 0.75 * spread, n),
                  rng.uniform(depth[0], depth[1], n)]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, tilt)) for _ in range(n)])
    if const_pages:
        pages = np.repeat(np.repeat(rng.uniform(0.05, 0.95, (n, 1, 1, 3)), PAGE, 1), PAGE, 2)
    else:
        pages = rng.uniform(0.0, 1.0, (n, PAGE, PAGE, 3))
    return Scene(means, rots, rng.uniform(scale[0], scale[1], (n, 2)), rng.uniform(opac[0], opac[1], n), pages,
                 rng.uniform(edge_lo, edge_hi, n))


# ------------------------------------------------------------------------------------------------ 오라클
def oracle_bilinear(page, u, v):
    """page (8,8,C), u, v 배열 -> (..., C). 열 i <-> u, 행 j <-> v, 클램프 주소."""
    page = np.asarray(page, np.float64)
    x = np.clip(4.0 * (np.asarray(u, np.float64) + 1.0) - 0.5, 0.0, PAGE - 1.0)
    y = np.clip(4.0 * (np.asarray(v, np.float64) + 1.0) - 0.5, 0.0, PAGE - 1.0)
    x0 = np.minimum(np.floor(x), PAGE - 2).astype(np.int64)
    y0 = np.minimum(np.floor(y), PAGE - 2).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    return ((1 - fx) * (1 - fy) * page[y0, x0] + fx * (1 - fy) * page[y0, x0 + 1]
            + (1 - fx) * fy * page[y0 + 1, x0] + fx * fy * page[y0 + 1, x0 + 1])


def oracle_local(sc: Scene, c2w, K, W, H):
    """정렬 전 TSQ별 화소 (u, v, ok, t, d2) 및 카메라 좌표 중심. 각 (N, H*W)."""
    w2c = geom.invert_pose(np.asarray(c2w, np.float64))
    Rw, tw = w2c[:3, :3], w2c[:3, 3]
    pc = sc.means @ Rw.T + tw
    fx, fy, sk, cx, cy = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    yd = (vv - cy) / fy
    xd = (uu - cx - sk * yd) / fx
    D = np.stack([xd, yd, np.ones_like(xd)], 0)
    Rc = Rw[None] @ sc.rots
    tu, tv, nc = Rc[:, :, 0], Rc[:, :, 1], Rc[:, :, 2]
    numer = (nc * pc).sum(-1)
    Dn = nc @ D
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numer[:, None] / np.where(np.abs(Dn) > DENOM_EPS, Dn, 1.0)
    ok = (np.abs(Dn) > DENOM_EPS) & (t > 0)
    ru = t * (tu @ D) - (tu * pc).sum(-1)[:, None]
    rv = t * (tv @ D) - (tv * pc).sum(-1)[:, None]
    u = ru / sc.scales[:, 0:1]
    v = rv / sc.scales[:, 1:2]
    cu = (fx * pc[:, 0] + sk * pc[:, 1]) / pc[:, 2] + cx
    cv = fy * pc[:, 1] / pc[:, 2] + cy
    d2 = (uu[None] - cu[:, None]) ** 2 + (vv[None] - cv[:, None]) ** 2
    return u, v, ok, t, d2, pc, numer


def oracle_render(sc: Scene, c2w, K, W, H, bg=(0.0, 0.0, 0.0), near=0.01, far=1e4):
    """계약서 래스터 의미의 numpy float64 기준 구현. dict(rgb (H,W,3), alpha, depth, normal)."""
    P = W * H
    bg = np.asarray(bg, np.float64)
    rgb = np.zeros((P, 3))
    T = np.ones(P)
    dnum = np.zeros(P)
    nsum = np.zeros((P, 3))
    if sc.n:
        u, v, ok, t, d2, pc, numer = oracle_local(sc, c2w, K, W, H)
        z = pc[:, 2]
        order = [i for i in np.argsort(z, kind="stable") if near <= z[i] <= far]
        w2c = geom.invert_pose(np.asarray(c2w, np.float64))
        for i in order:
            n = max(sc.edge_n[i], 2.0)
            with np.errstate(over="ignore", invalid="ignore"):
                E = 0.5 * (np.abs(u[i]) ** n + np.abs(v[i]) ** n)
            E = np.where(ok[i], E, np.inf)
            A = np.minimum(ALPHA_MAX, sc.opac[i] * np.exp(-np.minimum(E, d2[i])))
            A = np.where(A >= ALPHA_MIN, A, 0.0)
            uu = np.where(ok[i], u[i], 0.0)
            vv = np.where(ok[i], v[i], 0.0)
            col = oracle_bilinear(sc.pages[i], uu, vv)
            w = A * T
            rgb += w[:, None] * col
            zi = np.where(ok[i], t[i], z[i])
            dnum += w * zi
            sign = -1.0 if numer[i] > 0 else 1.0
            nsum += w[:, None] * (sc.rots[i][:, 2] * sign)[None]
            T = T * (1 - A)
        del w2c
    alpha = 1.0 - T
    # alpha = sum w (같은 값, 수치적으로 1 - prod(1-A))
    rgb = rgb + T[:, None] * bg
    low = alpha < ALPHA_OUT_EPS
    depth = np.where(low, np.inf, dnum / np.where(low, 1.0, alpha))
    nn = np.sqrt((nsum ** 2).sum(-1, keepdims=True))
    normal = np.where(low[:, None], 0.0, nsum / np.maximum(nn, 1e-30))
    return dict(rgb=rgb.reshape(H, W, 3), alpha=alpha.reshape(H, W), depth=depth.reshape(H, W),
                normal=normal.reshape(H, W, 3))


def ray_plane_depth(K, c2w, point_w, normal_w, w, h):
    """각 화소 광선과 평면의 교점의 카메라 z-깊이 (h, w). 교점이 뒤면 nan."""
    w2c = geom.invert_pose(np.asarray(c2w, np.float64))
    p = w2c[:3, :3] @ point_w + w2c[:3, 3]
    n = w2c[:3, :3] @ normal_w
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    lam = (n @ p) / (d @ n)
    lam[lam <= 0] = np.nan
    return lam


# ------------------------------------------------------------------------------------------------ BC 오라클
def _bits(b):
    return int.from_bytes(bytes(b), "little")


def bc7_mode6_decode_block(b):
    """16 B -> (16, 4) uint8 (화소 행 우선). 모드 6과 예약(바이트0=0)만. 그 외 None."""
    x = _bits(b)
    if x & 0xFF == 0:
        return np.zeros((16, 4), np.uint8)
    if x & 0x7F != 0x40:
        return None
    pos = 7
    ch = []
    for _ in range(8):
        ch.append((x >> pos) & 0x7F)
        pos += 7
    p0 = (x >> 63) & 1
    p1 = (x >> 64) & 1
    pos = 65
    e0 = np.array([(ch[0] << 1) | p0, (ch[2] << 1) | p0, (ch[4] << 1) | p0, (ch[6] << 1) | p0], np.int64)
    e1 = np.array([(ch[1] << 1) | p1, (ch[3] << 1) | p1, (ch[5] << 1) | p1, (ch[7] << 1) | p1], np.int64)
    out = np.zeros((16, 4), np.uint8)
    for i in range(16):
        nb = 3 if i == 0 else 4
        idx = (x >> pos) & ((1 << nb) - 1)
        pos += nb
        w = BC7_W4[idx]
        out[i] = ((64 - w) * e0 + w * e1 + 32) >> 6
    assert pos == 128
    return out


def bc7_decode_image(data, w, h):
    """오라클: 모드 6 BC7 바이트열 -> (h, w, 4) uint8. 모드 6이 아닌 블록이 있으면 AssertionError."""
    assert len(data) == (w // 4) * (h // 4) * 16
    img = np.zeros((h, w, 4), np.uint8)
    k = 0
    for by in range(h // 4):
        for bx in range(w // 4):
            blk = bc7_mode6_decode_block(data[16 * k:16 * k + 16])
            assert blk is not None, f"블록 {k}: 모드 6 아님 (byte0={data[16 * k]:#x})"
            img[4 * by:4 * by + 4, 4 * bx:4 * bx + 4] = blk.reshape(4, 4, 4)
            k += 1
    return img


def bc4_decode_block(b):
    """8 B -> (16,) float64 [0,1] (D3D 부동 팔레트 / 255)."""
    r0, r1 = int(b[0]), int(b[1])
    if r0 > r1:
        pal = [r0, r1] + [((7 - k) * r0 + k * r1) / 7.0 for k in range(1, 7)]
    else:
        pal = [r0, r1] + [((5 - k) * r0 + k * r1) / 5.0 for k in range(1, 5)] + [0.0, 255.0]
    x = _bits(b[2:8])
    return np.array([pal[(x >> (3 * i)) & 7] for i in range(16)], np.float64) / 255.0


def bc5_decode_image(data, w, h):
    assert len(data) == (w // 4) * (h // 4) * 16
    img = np.zeros((h, w, 2))
    k = 0
    for by in range(h // 4):
        for bx in range(w // 4):
            blk = data[16 * k:16 * k + 16]
            img[4 * by:4 * by + 4, 4 * bx:4 * bx + 4, 0] = bc4_decode_block(blk[0:8]).reshape(4, 4)
            img[4 * by:4 * by + 4, 4 * bx:4 * bx + 4, 1] = bc4_decode_block(blk[8:16]).reshape(4, 4)
            k += 1
    return img


def smooth_image(h, w, c, seed, lo=0, hi=255):
    """저주파 부드러운 영상 (h, w, c) float64 [lo, hi]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros((h, w, c))
    for k in range(c):
        acc = np.zeros((h, w))
        for _ in range(3):
            fx, fy = rng.uniform(0.005, 0.03, 2)
            ph = rng.uniform(0, 2 * np.pi)
            acc += np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
        acc = (acc - acc.min()) / (acc.max() - acc.min())
        out[..., k] = lo + (hi - lo) * acc
    return out


def psnr8(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
