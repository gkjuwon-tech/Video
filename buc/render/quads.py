"""RDR-02 평면 쿼드·LDI 래스터 v0 (빠른 경로: 전치 알파 텐트 스플랫, 소프트 에지).

계약서: ops/harness/contracts/RDR-02.md. 세계 오른손·미터, 카메라 OpenCV, 픽셀 중심이 정수 좌표.
층의 rgb 는 선형 Rec.709, 알파는 직선(비전치). 출력도 선형(표시값 변환은 render_shot 에서만).

구현 요약 (관측 가능한 결과만 계약):
  * 층마다 텍셀 중심과 네 모서리(±0.5 px)를 `world.obs` 로 역투영 -> 목표 카메라로 투영한다.
    모서리 광선은 깊이와 무관하므로 호출당 한 번만 구해 모든 층이 공유한다.
  * 텍셀은 축 정렬 텐트 필터(반경 rx, ry = clip(max(edge_px, 투영 크기), edge_px, max_footprint_px),
    단위 적분)로 면적 Ax 만큼의 알파 질량을 퍼뜨린다. 항등 카메라에서 정확히 항등이다.
  * 전치 알파로 누적한다: 층 색 C = sum(w a c) / sum(w a), 층 알파 A = min(1, sum(w a)).
  * 같은 층 안에서는 화소별 최근접 깊이 z_min 의 (1 + z_tol_rel) 배보다 먼 텍셀을 버린다.
    z_min 은 그 화소에 의미 있게(텐트 값 > _ZMIN_WEIGHT_FLOOR) 걸치는 텍셀만으로 정한다.
  * 층 합성은 앞 -> 뒤 직선 알파 over.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from buc.core import color, geom
from buc.core import contracts as C

__all__ = ["RenderOpts", "QuadRender", "card_layer", "render_ldi", "render_shot"]

# z_min 을 정할 때 화소에 걸친 것으로 치는 최소 텐트 값(피크=1 기준). 수치 잡음 수준의 꼬리가
# 같은 층의 먼 텍셀을 통째로 버리는 것을 막는다.
_ZMIN_WEIGHT_FLOOR = 1e-3
_ALPHA_OUT_EPS = 1e-3


# ---------------------------------------------------------------------------------------------------------------
# 공개 자료형
# ---------------------------------------------------------------------------------------------------------------
def _finite_num(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name}: 유한해야 한다 ({v!r})")
    return f


@dataclass(frozen=True, kw_only=True)
class RenderOpts:
    bg: tuple = (0.0, 0.0, 0.0)      # 모든 층 뒤의 선형 배경색
    edge_px: float = 1.0             # 최소 필터 반경(목표 px) = 소프트 에지 폭
    z_tol_rel: float = 0.1           # 같은 층 안에서 z_min*(1+z_tol_rel) 보다 먼 텍셀은 가려짐
    alpha_eps: float = 1e-3          # 알파 <= eps 텍셀은 건너뜀
    near_m: float = 1e-3             # 목표 카메라 z < near_m 인 텍셀은 건너뜀
    max_footprint_px: float = 8.0    # 텍셀 풋프린트 반경 상한

    def __post_init__(self):
        try:
            bg = tuple(float(x) for x in self.bg)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bg: 길이 3 의 유한한 수열이어야 한다 ({self.bg!r})") from e
        if len(bg) != 3 or not all(math.isfinite(x) for x in bg):
            raise ValueError(f"bg: 길이 3 의 유한한 수열이어야 한다 ({self.bg!r})")
        object.__setattr__(self, "bg", bg)
        edge = _finite_num("edge_px", self.edge_px)
        ztol = _finite_num("z_tol_rel", self.z_tol_rel)
        aeps = _finite_num("alpha_eps", self.alpha_eps)
        near = _finite_num("near_m", self.near_m)
        maxfp = _finite_num("max_footprint_px", self.max_footprint_px)
        if not edge > 0:
            raise ValueError("edge_px > 0 이어야 한다")
        if not ztol >= 0:
            raise ValueError("z_tol_rel >= 0 이어야 한다")
        if not aeps >= 0:
            raise ValueError("alpha_eps >= 0 이어야 한다")
        if not near > 0:
            raise ValueError("near_m > 0 이어야 한다")
        if not maxfp >= 1:
            raise ValueError("max_footprint_px >= 1 이어야 한다")


@dataclass(frozen=True, kw_only=True)
class QuadRender:
    rgb: np.ndarray        # (H,W,3) float32 선형, 배경 위에 합성
    alpha: np.ndarray      # (H,W) float32 [0,1], 1 - prod(1 - A_l)
    depth: np.ndarray      # (H,W) float32 목표 카메라 z-깊이(m), 가중합 < 1e-3 이면 +inf
    layer_id: np.ndarray   # (H,W) int16, 가중치 최대인 층, alpha < 1e-3 이면 -1


def card_layer(rgba, depth_m) -> C.LDILayer:
    """상수 깊이 평면 카드."""
    try:
        d = float(depth_m)
    except (TypeError, ValueError) as e:
        raise ValueError(f"depth_m: 유한·양수여야 한다 ({depth_m!r})") from e
    if not (math.isfinite(d) and d > 0):
        raise ValueError(f"depth_m: 유한·양수여야 한다 ({depth_m!r})")
    rgba = np.asarray(rgba, np.float32)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"rgba shape {rgba.shape}: (H,W,4) 이어야 한다")
    return C.LDILayer(rgba=rgba, depth=np.full(rgba.shape[:2], d, np.float32))


# ---------------------------------------------------------------------------------------------------------------
# 입력 검증
# ---------------------------------------------------------------------------------------------------------------
def _check_size(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError(f"{name}: 정수여야 한다 ({v!r})")
    v = int(v)
    if v < 1:
        raise ValueError(f"{name} >= 1 이어야 한다 ({v})")
    return v


def _check_pose(c2w):
    try:
        T = np.array(c2w, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("c2w: 숫자 (4,4) 배열이어야 한다") from e
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"c2w: 유한한 (4,4) 이어야 한다 (shape {T.shape})")
    if np.abs(T[3] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-6:
        raise ValueError("c2w: 마지막 행이 [0,0,0,1] 이 아니다")
    if not geom.is_rotation(T[:3, :3], atol=1e-5):
        raise ValueError("c2w: 회전부가 직교(det +1)가 아니다")
    return T


def _check_K(K):
    try:
        K = np.array(K, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("K: 숫자 (3,3) 배열이어야 한다") from e
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"K: 유한한 (3,3) 이어야 한다 (shape {K.shape})")
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError("K: fx, fy > 0 이어야 한다")
    return K


# ---------------------------------------------------------------------------------------------------------------
# 래스터
# ---------------------------------------------------------------------------------------------------------------
def _footprints(h, w, obs, A, k1o, k2o):
    """관측 카메라의 텍셀 중심·모서리 광선을 목표 카메라 방향으로 돌려 둔다. 깊이 d 를 곱하면 점이 된다."""
    Ko = obs.K
    uc = geom.pixel_grid(h, w)                                          # (h,w,2)
    lat = np.stack(np.meshgrid(np.arange(w + 1) - 0.5, np.arange(h + 1) - 0.5), -1)   # (h+1,w+1,2) (u,v)
    rc = geom.unproject(Ko, uc, 1.0, k1o, k2o)                          # (h,w,3), z = 1
    rl = geom.unproject(Ko, lat, 1.0, k1o, k2o)                         # (h+1,w+1,3)
    # 모서리 순서(순환): (-,-), (+,-), (+,+), (-,+)
    corners = [rl[:-1, :-1], rl[:-1, 1:], rl[1:, 1:], rl[1:, :-1]]
    Pc = (rc @ A.T).reshape(-1, 3)
    Pk = [(c @ A.T).reshape(-1, 3) for c in corners]
    return Pc, Pk


def _tent_offsets(x, y, rx, ry, k, W, H):
    """한 버킷(K=k)의 모든 오프셋에 대해 (선택 인덱스, 화소 flat 인덱스, 피크 1 텐트값)을 낸다."""
    x0 = np.floor(x + 0.5).astype(np.int64)
    y0 = np.floor(y + 0.5).astype(np.int64)
    for oy in range(-k, k + 1):
        py = y0 + oy
        wy = 1.0 - np.abs(py - y) / ry
        oky = (py >= 0) & (py < H) & (wy > 0)
        for ox in range(-k, k + 1):
            px = x0 + ox
            wx = 1.0 - np.abs(px - x) / rx
            ok = oky & (px >= 0) & (px < W) & (wx > 0)
            sel = np.flatnonzero(ok)
            if sel.size == 0:
                continue
            yield sel, py[sel] * W + px[sel], wx[sel] * wy[sel]


def _raster_layer(layer, Pc, Pk, b, Kt, k1, k2, W, H, opts):
    """한 층 -> (A_l, C_l (H*W,3), z_l) 또는 None."""
    HW = H * W
    a_all = layer.rgba[..., 3].reshape(-1).astype(np.float64)
    d_all = layer.depth.reshape(-1).astype(np.float64)
    cand = np.flatnonzero((a_all > opts.alpha_eps) & np.isfinite(d_all) & (d_all > 0))
    if cand.size == 0:
        return None
    d = d_all[cand]
    # 목표 카메라 z (중심·모서리 모두 near 평면 앞이어야 한다)
    zc = d * Pc[cand, 2] + b[2]
    okz = zc >= opts.near_m
    zk = []
    for P in Pk:
        z = d * P[cand, 2] + b[2]
        zk.append(z)
        okz &= z >= opts.near_m
    keep = np.flatnonzero(okz)
    if keep.size == 0:
        return None
    cand, d, zc = cand[keep], d[keep], zc[keep]

    def proj(P, zz, dd, idx):
        X = dd * P[idx, 0] + b[0]
        Y = dd * P[idx, 1] + b[1]
        xn, yn = X / zz, Y / zz
        r2 = xn * xn + yn * yn
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        xd, yd = xn * f, yn * f
        return Kt[0, 0] * xd + Kt[0, 1] * yd + Kt[0, 2], Kt[1, 1] * yd + Kt[1, 2]

    with np.errstate(all="ignore"):
        x, y = proj(Pc, zc, d, cand)
        cu, cv = [], []
        for P, z in zip(Pk, zk):
            u_, v_ = proj(P, z[keep], d, cand)
            cu.append(u_)
            cv.append(v_)
        cu, cv = np.stack(cu, 0), np.stack(cv, 0)
        ext_x = cu.max(0) - cu.min(0)
        ext_y = cv.max(0) - cv.min(0)
        xs, ys = np.roll(cu, -1, 0), np.roll(cv, -1, 0)
        area = 0.5 * np.abs((cu * ys - xs * cv).sum(0))
        rx = np.clip(np.maximum(opts.edge_px, ext_x), opts.edge_px, opts.max_footprint_px)
        ry = np.clip(np.maximum(opts.edge_px, ext_y), opts.edge_px, opts.max_footprint_px)
        # 거의 축 정렬이면(원근으로 사다리꼴이 된 정도) 경계상자 면적을 질량으로 써서 확대 시 단위분할(구멍 없음)을
        # 정확히 유지한다. 롤 회전 등으로 많이 기울면 실제 면적을 쓴다.
        bbox = ext_x * ext_y
        ratio = area / np.maximum(bbox, 1e-30)
        blend = np.clip((ratio - 0.85) / 0.1, 0.0, 1.0)
        ax = np.minimum(area + (bbox - area) * blend, rx * ry)
        good = (np.isfinite(x) & np.isfinite(y) & np.isfinite(rx) & np.isfinite(ry) & np.isfinite(ax)
                & (x + rx > -0.5) & (x - rx < W - 0.5) & (y + ry > -0.5) & (y - ry < H - 0.5))
    g = np.flatnonzero(good)
    if g.size == 0:
        return None
    cand, d, zc, x, y, rx, ry, ax = cand[g], d[g], zc[g], x[g], y[g], rx[g], ry[g], ax[g]

    # 결정론적 순서: z 내림차순(동률은 원래 순서) -> "마지막 쓰기가 이긴다"로 화소별 최소 z 를 구한다.
    order = np.argsort(-zc, kind="stable")
    cand, zc, x, y, rx, ry, ax = cand[order], zc[order], x[order], y[order], rx[order], ry[order], ax[order]
    a = a_all[cand]
    rgb = layer.rgba[..., :3].reshape(-1, 3)[cand].astype(np.float64)
    wbase = ax * a / (rx * ry)

    kb = np.ceil(np.maximum(rx, ry) + 0.5 - 1e-9).astype(np.int64) - 1
    kb = np.maximum(kb, 1)
    buckets = [(int(k), np.flatnonzero(kb == k)) for k in np.unique(kb)]

    zmin = np.full(HW, np.inf)
    for k, ids in buckets:
        for sel, idx, t in _tent_offsets(x[ids], y[ids], rx[ids], ry[ids], k, W, H):
            sig = t > _ZMIN_WEIGHT_FLOOR
            if not sig.any():
                continue
            s2, i2 = ids[sel[sig]], idx[sig]
            z = zc[s2]
            zmin[i2] = np.minimum(zmin[i2], z)

    S = np.zeros(HW)
    Sc = np.zeros((3, HW))
    Sz = np.zeros(HW)
    tol = 1.0 + opts.z_tol_rel
    for k, ids in buckets:
        for sel, idx, t in _tent_offsets(x[ids], y[ids], rx[ids], ry[ids], k, W, H):
            s2 = ids[sel]
            z = zc[s2]
            kept = z <= tol * zmin[idx]
            if not kept.any():
                continue
            s2, idx, t, z = s2[kept], idx[kept], t[kept], z[kept]
            wgt = t * wbase[s2]
            S += np.bincount(idx, weights=wgt, minlength=HW)
            Sz += np.bincount(idx, weights=wgt * z, minlength=HW)
            wc = wgt[:, None] * rgb[s2]
            for c in range(3):
                Sc[c] += np.bincount(idx, weights=wc[:, c], minlength=HW)

    A = np.minimum(S, 1.0)
    pos = S > 1e-12
    inv = np.zeros(HW)
    inv[pos] = 1.0 / S[pos]
    Cl = (Sc * inv).T
    zl = Sz * inv
    A = np.where(pos, A, 0.0)
    return A, Cl, zl


def render_ldi(world, *, c2w, K, width, height, k1: float = 0.0, k2: float = 0.0, opts=None) -> QuadRender:
    if opts is None:
        opts = RenderOpts()
    T = _check_pose(c2w)
    Kt = _check_K(K)
    W = _check_size("width", width)
    H = _check_size("height", height)
    k1 = _finite_num("k1", k1)
    k2 = _finite_num("k2", k2)

    obs = world.obs
    Rt = T[:3, :3].T
    tt = -Rt @ T[:3, 3]
    A = Rt @ np.asarray(obs.R, np.float64).T                  # 관측 카메라 좌표 -> 목표 카메라 좌표
    b = tt - A @ np.asarray(obs.t, np.float64)

    HW = H * W
    layers = tuple(world.layers)
    bg = np.asarray(opts.bg, np.float64)
    trans = np.ones(HW)
    rgb_acc = np.zeros((HW, 3))
    z_acc = np.zeros(HW)
    w_sum = np.zeros(HW)
    best_w = np.full(HW, -1.0)
    best_id = np.full(HW, -1, np.int16)

    cache = {}
    for li, layer in enumerate(layers):
        lh, lw = layer.rgba.shape[:2]
        if (lh, lw) not in cache:
            cache[(lh, lw)] = _footprints(lh, lw, obs, A, float(obs.k1), float(obs.k2))
        Pc, Pk = cache[(lh, lw)]
        res = _raster_layer(layer, Pc, Pk, b, Kt, k1, k2, W, H, opts)
        if res is None:
            continue
        Al, Cl, zl = res
        wl = Al * trans
        rgb_acc += wl[:, None] * Cl
        z_acc += wl * zl
        w_sum += wl
        upd = wl > best_w + 0.0
        best_w = np.where(upd, wl, best_w)
        best_id = np.where(upd, np.int16(li), best_id)
        trans = trans * (1.0 - Al)

    rgb = rgb_acc + (1.0 - w_sum)[:, None] * bg
    alpha = 1.0 - trans
    has = w_sum >= _ALPHA_OUT_EPS
    depth = np.full(HW, np.inf)
    depth[has] = z_acc[has] / w_sum[has]
    lid = np.where(alpha < _ALPHA_OUT_EPS, np.int16(-1), best_id).astype(np.int16)
    return QuadRender(
        rgb=np.ascontiguousarray(rgb.reshape(H, W, 3), dtype=np.float32),
        alpha=np.ascontiguousarray(np.clip(alpha, 0.0, 1.0).reshape(H, W), dtype=np.float32),
        depth=np.ascontiguousarray(depth.reshape(H, W), dtype=np.float32),
        layer_id=np.ascontiguousarray(lid.reshape(H, W), dtype=np.int16),
    )


def render_shot(world, shot, opts=None) -> C.Frames:
    n = shot.n_frames
    outs = [render_ldi(world, c2w=shot.c2w[i], K=shot.K[i], width=shot.width, height=shot.height, opts=opts)
            for i in range(n)]
    rgb = np.stack([color.linear_to_display(o.rgb) for o in outs], 0).astype(np.float32)
    return C.Frames(
        rgb=rgb,
        fps=shot.fps,
        aov={
            "depth": np.stack([o.depth for o in outs], 0).astype(np.float32),
            "alpha": np.stack([o.alpha for o in outs], 0).astype(np.float32),
            "layer_id": np.stack([o.layer_id for o in outs], 0).astype(np.int16),
        },
        shot_hash=shot.content_hash(),
        world_hash=world.content_hash(),
    )
