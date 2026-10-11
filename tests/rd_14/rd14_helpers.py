"""RD-14 테스트 공용 도우미: 독립 오라클(계약서 수식 그대로), 장면 생성기, O 집합 전수 오라클.

RD-14 구현(buc.read.obs_map)을 오라클에 쓰지 않는다. numpy·scipy·math 만 쓴다(레벨은 log2 가 아니라 전수 탐색,
ν_obs 는 촘촘한 격자 + 선형 보간, O 는 파이썬 루프). 구현 모듈은 맨 아래 `om` 으로만 가져온다."""
import math
from pathlib import Path

import numpy as np
from scipy.special import j1

ROOT = Path(__file__).resolve().parents[2]

LEVEL_RANGE = (-8, 24)
H0_M = 64.0
BLOCK_CELLS = 16
OCT_HI = (0.5, 0.25, 0.125, 0.0625)
OCT_CENTER = tuple(h / math.sqrt(2.0) for h in OCT_HI)
NO_LEVEL = -128
NOISE_DEFAULT = dict(nsr_ref=0.01, nu_ref=0.25, alpha=2.0, jpeg_gain=0.04)


# ------------------------------------------------------------------ §2.10 식 (전수 탐색 오라클)
def tau_m(l):
    return 8.0 * 2.0 ** (-l)


def phi_in(z, f_px, cos_theta=1.0):
    return np.asarray(z, dtype=np.float64) / (f_px * np.asarray(cos_theta, dtype=np.float64))


def jpeg_factor(q):
    if q is None or q >= 90:
        return 1.0
    return min(4.0, 2.0 ** ((90.0 - q) / 40.0))


def phi_eff(phi, *, r_ratio=1.0, coc=0.0, mot=0.0, qj=1.0, nu_cap=None):
    terms = [1.0, 1.0 / r_ratio, np.asarray(coc, dtype=np.float64) / 1.5, np.asarray(mot, dtype=np.float64) / 1.5, qj]
    if nu_cap is not None:
        terms.append(0.5 / np.asarray(nu_cap, dtype=np.float64))
    f = terms[0]
    for t in terms[1:]:
        f = np.maximum(f, t)
    return np.asarray(phi, dtype=np.float64) * f


def owned_levels(phi, part_extra=0.0):
    """(hard, part): τ_l ≥ 2φ 인 최대 l / τ_l ≥ φ·2^-extra 인 최대 l. 허용 오차 1e-9(동률은 소유)."""
    phi = float(phi)
    hard = part = LEVEL_RANGE[0]
    for l in range(LEVEL_RANGE[0], LEVEL_RANGE[1] + 1):
        if tau_m(l) >= 2.0 * phi * (1 - 1e-9):
            hard = l
        if tau_m(l) >= phi * 2.0 ** (-part_extra) * (1 - 1e-9):
            part = l
    return hard, part


# ------------------------------------------------------------------ w(ν), ν_obs 오라클
def defocus_mtf(nu, c):
    x = np.pi * np.asarray(c, dtype=np.float64) * np.asarray(nu, dtype=np.float64)
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, np.abs(2.0 * j1(safe) / safe))


def motion_mtf(nu, L):
    return np.abs(np.sinc(np.asarray(L, dtype=np.float64) * np.asarray(nu, dtype=np.float64)))


def nsr(nu, qj=1.0, nsr_ref=0.01, nu_ref=0.25, alpha=2.0, jpeg_gain=0.04):
    return (nsr_ref + jpeg_gain * (qj - 1.0)) * (np.asarray(nu, dtype=np.float64) / nu_ref) ** alpha


def w_smooth(nu, c=0.0, m=0.0, qj=1.0, **noise):
    n = dict(NOISE_DEFAULT)
    n.update(noise)
    Hm = defocus_mtf(nu, c) * motion_mtf(nu, m)
    N = nsr(nu, qj, **n)
    return Hm * Hm / (Hm * Hm + N)


def w_full(nu, r_ratio=1.0, c=0.0, m=0.0, qj=1.0, **noise):
    w = w_smooth(nu, c, m, qj, **noise)
    return np.where(np.asarray(nu) > 0.5 * r_ratio * (1 + 1e-12), 0.0, w)


def nu_obs_oracle(r_ratio=1.0, c=0.0, m=0.0, qj=1.0, n=20000, **noise):
    g = np.arange(n + 1) * 0.5 / n
    wg = w_smooth(g, c, m, qj, **noise)
    idx = np.nonzero(wg[1:] < 0.5)[0]
    if len(idx) == 0:
        nw = 0.5
    else:
        i = idx[0] + 1
        nw = g[i - 1] + (0.5 - wg[i - 1]) / (wg[i] - wg[i - 1]) * (g[i] - g[i - 1])
    return min(nw, 0.5 * r_ratio)


def mrh_cap_oracle(flags):
    """flags: 길이 4 bool(옥타브 0=최고 대역). 가장 낮은 대역부터 연속으로 관측인 b*: cap = 0.5·2^-b* (b*=4 → 1/32)."""
    b = 4
    while b > 0 and flags[b - 1]:
        b -= 1
    return 0.5 * 2.0 ** (-b)


# ------------------------------------------------------------------ 장면
def K_from_fov(W, H, fov_x_deg=60.0):
    f = (W / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    return np.array([[f, 0.0, (W - 1) / 2.0], [0.0, f, (H - 1) / 2.0], [0.0, 0.0, 1.0]])


def f_for(W, fov_x_deg=60.0):
    return (W / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rays(K, H, W):
    """단위 시선 (H,W,3) — 화소 중심 정수 좌표, 왜곡 없음."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def world_points(K, R, t, depth, stride=1):
    """오라클: X = R^T (z·[x,y,1] - t), 샘플 (i,j) = (stride·a, stride·b), 화소 중심 좌표 (u=j, v=i)."""
    H, W = depth.shape
    ii, jj = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[ii, jj]
    x = (jj - K[0, 2]) / K[0, 0]
    y = (ii - K[1, 2]) / K[1, 1]
    Xc = np.stack([x * z, y * z, z], axis=-1)
    return (Xc - t) @ R, ii, jj


# ------------------------------------------------------------------ O 집합 전수 오라클
def oracle_obs_set(X, ii, jj, lh, lp, valid, levels):
    """X (a,b,3) 세계점, (ii,jj) 그 표본의 화소 행·열.
    {(l,i,j,k): (hard_idx:set, part_idx:set)} — 셀은 (소유 가능 표본이 모두 경소유일 때만) hard, 아니면 part."""
    out = {}
    cells = {}
    for l in levels:
        cs = H0_M * 2.0 ** (-l)
        for a in range(X.shape[0]):
            for b in range(X.shape[1]):
                i, j = ii[a, b], jj[a, b]
                if not valid[i, j] or l > lp[i, j]:
                    continue
                c = tuple(int(math.floor(v / cs)) for v in X[a, b])
                cells.setdefault((l, c), []).append(bool(l <= lh[i, j]))
    for (l, c), flags in cells.items():
        blk = tuple(v // BLOCK_CELLS for v in c)
        loc = tuple(v % BLOCK_CELLS for v in c)
        idx = loc[0] + 16 * loc[1] + 256 * loc[2]
        hard, part = out.setdefault((l,) + blk, (set(), set()))
        (hard if all(flags) else part).add(idx)
    return out


def decode_bits(words):
    s = set()
    for wi, word in enumerate(np.asarray(words, dtype=np.uint64)):
        word = int(word)
        while word:
            low = word & -word
            s.add(wi * 64 + low.bit_length() - 1)
            word ^= low
    return s


def set_as_dict(obs_set):
    return {k: (decode_bits(b.hard), decode_bits(b.part)) for k, b in obs_set.blocks.items()}


# ------------------------------------------------------------------ 구현 모듈 (구현 전에는 여기서 실패한다: 정상)
from buc.read import obs_map as om  # noqa: E402
