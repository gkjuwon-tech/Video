"""RDR-05 AOV (depth·normal·flow·원자 id)·관측 bitset·안정 정렬 v0.

계약서: ops/harness/contracts/RDR-05.md. 렌더 결과(rgb·alpha·depth·layer_id)는 `render_ldi` 를 그대로 호출해 얻으므로
바이트 동일이다. 여기서는 그 위에 지배 원자(화소 -> 텍셀), 법선, 광류, 관측 bitset 을 얹는다.

원자 = LDI 텍셀 (층 l, 행 v, 열 u), atom_id = l*H*W + v*W + u (int64, -1 = 없음).
지배 원자: RDR-02 의 층 내 스플랫과 같은 풋프린트·깊이 컬링을 다시 계산해, `layer_id` 층에서 화소에 걸친 텍셀 중
스플랫 가중치(텐트·알파·면적/(rx·ry))가 최대인 것(동률은 작은 atom_id)을 고른다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from buc.core import color, det, geom
from buc.core import contracts as C
from buc.render import quads
from buc.render.quads import RenderOpts, render_ldi

__all__ = ["Cam", "AOV", "n_atoms", "pack_atom_id", "unpack_atom_id", "atom_points", "render_aov",
           "render_shot_aov", "ObsBitset", "observed_atoms", "atom_pixel_counts", "sort_fragments"]


# ---------------------------------------------------------------------------------------------------------------
# 자료형
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Cam:
    c2w: np.ndarray
    K: np.ndarray
    k1: float = 0.0
    k2: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "c2w", quads._check_pose(self.c2w))
        object.__setattr__(self, "K", quads._check_K(self.K))
        object.__setattr__(self, "k1", quads._finite_num("k1", self.k1))
        object.__setattr__(self, "k2", quads._finite_num("k2", self.k2))


@dataclass(frozen=True, kw_only=True)
class AOV:
    rgb: np.ndarray        # (H,W,3) f32
    alpha: np.ndarray      # (H,W) f32
    depth: np.ndarray      # (H,W) f32
    layer_id: np.ndarray   # (H,W) i16
    normal: np.ndarray     # (H,W,3) f32
    atom_id: np.ndarray    # (H,W) i64
    flow: np.ndarray       # (H,W,2) f32
    flow_valid: np.ndarray  # (H,W) bool


# ---------------------------------------------------------------------------------------------------------------
# 원자 id
# ---------------------------------------------------------------------------------------------------------------
def _int_array(name, x):
    a = np.asarray(x)
    if a.dtype.kind not in "iub":
        raise TypeError(f"{name}: 정수 배열이어야 한다 (dtype {a.dtype})")
    return a.astype(np.int64)


def _hw(world):
    return int(world.obs.height), int(world.obs.width)


def n_atoms(world) -> int:
    h, w = _hw(world)
    return len(world.layers) * h * w


def pack_atom_id(layer, v, u, *, width, height) -> np.ndarray:
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("width, height >= 1 이어야 한다")
    L, V, U = np.broadcast_arrays(_int_array("layer", layer), _int_array("v", v), _int_array("u", u))
    if (L < 0).any() or (V < 0).any() or (V >= height).any() or (U < 0).any() or (U >= width).any():
        raise ValueError("pack_atom_id: 범위 밖 (layer>=0, 0<=v<height, 0<=u<width)")
    return (L * (height * width) + V * width + U).astype(np.int64)


def unpack_atom_id(atom_id, *, width, height):
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("width, height >= 1 이어야 한다")
    a = _int_array("atom_id", atom_id)
    if (a < 0).any():
        raise ValueError("unpack_atom_id: 음수 id")
    hw = height * width
    layer, rem = np.divmod(a, hw)
    v, u = np.divmod(rem, width)
    return layer.astype(np.int64), v.astype(np.int64), u.astype(np.int64)


def atom_points(world, atom_id) -> np.ndarray:
    ids = _int_array("atom_id", atom_id)
    n = n_atoms(world)
    if (ids >= n).any():
        raise ValueError(f"atom_points: id >= n_atoms ({n})")
    out = np.full(ids.shape + (3,), np.nan, np.float64)
    ok = ids >= 0
    if not ok.any():
        return out
    h, w = _hw(world)
    layer, v, u = unpack_atom_id(ids[ok], width=w, height=h)
    obs = world.obs
    depth = np.empty(layer.shape, np.float64)
    for li in np.unique(layer):
        sel = layer == li
        depth[sel] = np.asarray(world.layers[int(li)].depth)[v[sel], u[sel]]
    uv = np.stack([u, v], -1).astype(np.float64)
    Xc = geom.unproject(obs.K, uv, depth, obs.k1, obs.k2)
    R = np.asarray(obs.R, np.float64)
    t = np.asarray(obs.t, np.float64)
    out[ok] = (Xc - t) @ R
    return out


# ---------------------------------------------------------------------------------------------------------------
# 지배 원자
# ---------------------------------------------------------------------------------------------------------------
def _dominant_in_layer(layer, Pc, Pk, b, Kt, k1, k2, W, H, opts, want):
    """한 층에서 want 화소마다 스플랫 가중치가 최대인 텍셀의 층 내 flat 인덱스(없으면 -1) (H*W,).
    quads._raster_layer 의 컬링·풋프린트·z_min 규칙을 그대로 따른다."""
    HW = H * W
    best = np.full(HW, -1, np.int64)
    a_all = layer.rgba[..., 3].reshape(-1).astype(np.float64)
    d_all = layer.depth.reshape(-1).astype(np.float64)
    cand = np.flatnonzero((a_all > opts.alpha_eps) & np.isfinite(d_all) & (d_all > 0))
    if cand.size == 0:
        return best
    d = d_all[cand]
    zc = d * Pc[cand, 2] + b[2]
    okz = zc >= opts.near_m
    zk = []
    for P in Pk:
        z = d * P[cand, 2] + b[2]
        zk.append(z)
        okz &= z >= opts.near_m
    keep = np.flatnonzero(okz)
    if keep.size == 0:
        return best
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
        bbox = ext_x * ext_y
        ratio = area / np.maximum(bbox, 1e-30)
        blend = np.clip((ratio - 0.85) / 0.1, 0.0, 1.0)
        ax = np.minimum(area + (bbox - area) * blend, rx * ry)
        good = (np.isfinite(x) & np.isfinite(y) & np.isfinite(rx) & np.isfinite(ry) & np.isfinite(ax)
                & (x + rx > -0.5) & (x - rx < W - 0.5) & (y + ry > -0.5) & (y - ry < H - 0.5))
    g = np.flatnonzero(good)
    if g.size == 0:
        return best
    cand, d, zc, x, y, rx, ry, ax = cand[g], d[g], zc[g], x[g], y[g], rx[g], ry[g], ax[g]

    order = np.argsort(-zc, kind="stable")
    cand, zc, x, y, rx, ry, ax = cand[order], zc[order], x[order], y[order], rx[order], ry[order], ax[order]
    a = a_all[cand]
    wbase = ax * a / (rx * ry)

    kb = np.ceil(np.maximum(rx, ry) + 0.5 - 1e-9).astype(np.int64) - 1
    kb = np.maximum(kb, 1)
    buckets = [(int(k), np.flatnonzero(kb == k)) for k in np.unique(kb)]

    zmin = np.full(HW, np.inf)
    for k, ids in buckets:
        for sel, idx, t in quads._tent_offsets(x[ids], y[ids], rx[ids], ry[ids], k, W, H):
            sig = t > quads._ZMIN_WEIGHT_FLOOR
            if not sig.any():
                continue
            s2, i2 = ids[sel[sig]], idx[sig]
            zmin[i2] = np.minimum(zmin[i2], zc[s2])

    tol = 1.0 + opts.z_tol_rel
    f_idx, f_w, f_atom = [], [], []
    for k, ids in buckets:
        for sel, idx, t in quads._tent_offsets(x[ids], y[ids], rx[ids], ry[ids], k, W, H):
            s2 = ids[sel]
            kept = (zc[s2] <= tol * zmin[idx]) & want[idx]
            if not kept.any():
                continue
            s2, idx, t = s2[kept], idx[kept], t[kept]
            f_idx.append(idx)
            f_w.append(t * wbase[s2])
            f_atom.append(cand[s2])
    if not f_idx:
        return best
    f_idx = np.concatenate(f_idx)
    f_w = np.concatenate(f_w)
    f_atom = np.concatenate(f_atom)
    o = np.lexsort((f_atom, -f_w, f_idx))           # 화소 -> 가중치 큰 순 -> 작은 atom
    f_idx, f_atom = f_idx[o], f_atom[o]
    first = np.ones(f_idx.size, bool)
    first[1:] = f_idx[1:] != f_idx[:-1]
    best[f_idx[first]] = f_atom[first]
    return best


# ---------------------------------------------------------------------------------------------------------------
# 법선
# ---------------------------------------------------------------------------------------------------------------
def _texel_normals(layer, obs, opts):
    """한 층의 텍셀 법선 (h*w,3), 관측 카메라 좌표, 카메라를 향함."""
    h, w = layer.rgba.shape[:2]
    depth = layer.depth.astype(np.float64)
    valid = (layer.rgba[..., 3].astype(np.float64) > opts.alpha_eps) & np.isfinite(depth) & (depth > 0)
    dsafe = np.where(valid, depth, 1.0)
    P = geom.unproject(obs.K, geom.pixel_grid(h, w), dsafe, obs.k1, obs.k2)       # (h,w,3)
    tol = opts.z_tol_rel

    def axis_diff(axis):
        # 중심 대비 -1/+1 이웃의 유효성
        def shifted(arr, s):
            out = np.zeros_like(arr)
            sl_dst = [slice(None)] * arr.ndim
            sl_src = [slice(None)] * arr.ndim
            if s > 0:       # out[i] = arr[i+1]
                sl_dst[axis], sl_src[axis] = slice(0, -1), slice(1, None)
            else:           # out[i] = arr[i-1]
                sl_dst[axis], sl_src[axis] = slice(1, None), slice(0, -1)
            out[tuple(sl_dst)] = arr[tuple(sl_src)]
            return out
        ok = {}
        Pn = {}
        for s in (-1, 1):
            vn = shifted(valid, s)
            Pn[s] = shifted(P, s)
            zn = shifted(dsafe, s)
            ok[s] = valid & vn & (np.abs(zn - dsafe) <= tol * dsafe)
        dP = np.zeros_like(P)
        both = ok[-1] & ok[1]
        dP = np.where(both[..., None], Pn[1] - Pn[-1], dP)
        only_r = ok[1] & ~ok[-1]
        dP = np.where(only_r[..., None], Pn[1] - P, dP)
        only_l = ok[-1] & ~ok[1]
        dP = np.where(only_l[..., None], P - Pn[-1], dP)
        return dP, ok[-1] | ok[1]

    du, hu = axis_diff(1)
    dv, hv = axis_diff(0)
    n = np.cross(du, dv)
    flip = (n * P).sum(-1) > 0
    n = np.where(flip[..., None], -n, n)
    nn = np.linalg.norm(n, axis=-1)
    pn = np.linalg.norm(P, axis=-1)
    fallback = ~(hu & hv) | ~(nn > 1e-12 * np.maximum(pn, 1e-30) ** 2) | ~np.isfinite(nn)
    n = n / np.where(nn > 0, nn, 1.0)[..., None]
    fb = -P / np.where(pn > 0, pn, 1.0)[..., None]
    n = np.where(fallback[..., None], fb, n)
    return n.reshape(-1, 3)


# ---------------------------------------------------------------------------------------------------------------
# render_aov
# ---------------------------------------------------------------------------------------------------------------
def render_aov(world, *, c2w, K, width, height, k1: float = 0.0, k2: float = 0.0, opts=None,
               flow_to=None) -> AOV:
    if flow_to is not None and not isinstance(flow_to, Cam):
        raise TypeError("flow_to: Cam 또는 None 이어야 한다")
    if opts is None:
        opts = RenderOpts()
    ref = render_ldi(world, c2w=c2w, K=K, width=width, height=height, k1=k1, k2=k2, opts=opts)
    T = quads._check_pose(c2w)
    Kt = quads._check_K(K)
    W = quads._check_size("width", width)
    H = quads._check_size("height", height)
    k1 = quads._finite_num("k1", k1)
    k2 = quads._finite_num("k2", k2)
    HW = H * W

    obs = world.obs
    Rt = T[:3, :3].T
    tt = -Rt @ T[:3, 3]
    A = Rt @ np.asarray(obs.R, np.float64).T
    b = tt - A @ np.asarray(obs.t, np.float64)

    lid = ref.layer_id.reshape(-1)
    atom = np.full(HW, -1, np.int64)
    normal = np.zeros((HW, 3), np.float64)
    ho, wo = _hw(world)
    layers = tuple(world.layers)
    used = np.unique(lid[lid >= 0])
    cache = {}
    for li in used:
        li = int(li)
        ly = layers[li]
        lh, lw = ly.rgba.shape[:2]
        if (lh, lw) not in cache:
            cache[(lh, lw)] = quads._footprints(lh, lw, obs, A, float(obs.k1), float(obs.k2))
        Pc, Pk = cache[(lh, lw)]
        want = lid == li
        best = _dominant_in_layer(ly, Pc, Pk, b, Kt, k1, k2, W, H, opts, want)
        got = want & (best >= 0)
        if not got.any():
            continue
        atom[got] = li * (lh * lw) + best[got]
        tn = _texel_normals(ly, obs, opts)
        normal[got] = tn[best[got]]
    del ho, wo

    nrm = normal @ A.T
    ln = np.linalg.norm(nrm, axis=-1)
    nrm = nrm / np.where(ln > 0, ln, 1.0)[:, None]
    nrm[atom < 0] = 0.0

    flow = np.zeros((HW, 2), np.float64)
    fvalid = np.zeros(HW, bool)
    if flow_to is not None:
        sel = np.flatnonzero(atom >= 0)
        if sel.size:
            X = atom_points(world, atom[sel])
            wa = geom.invert_pose(T)
            wb = geom.invert_pose(flow_to.c2w)
            uv_a, _ = geom.project(Kt, wa[:3, :3], wa[:3, 3], X, k1, k2)
            uv_b, zb = geom.project(flow_to.K, wb[:3, :3], wb[:3, 3], X, flow_to.k1, flow_to.k2)
            f = uv_b - uv_a
            ok = (zb >= opts.near_m) & np.isfinite(f).all(-1)
            flow[sel[ok]] = f[ok]
            fvalid[sel[ok]] = True

    return AOV(
        rgb=ref.rgb, alpha=ref.alpha, depth=ref.depth, layer_id=ref.layer_id,
        normal=np.ascontiguousarray(nrm.reshape(H, W, 3), dtype=np.float32),
        atom_id=np.ascontiguousarray(atom.reshape(H, W), dtype=np.int64),
        flow=np.ascontiguousarray(flow.reshape(H, W, 2), dtype=np.float32),
        flow_valid=np.ascontiguousarray(fvalid.reshape(H, W), dtype=bool),
    )


def render_shot_aov(world, shot, opts=None) -> C.Frames:
    n = shot.n_frames
    outs = []
    for i in range(n):
        nxt = Cam(c2w=shot.c2w[i + 1], K=shot.K[i + 1]) if i + 1 < n else None
        outs.append(render_aov(world, c2w=shot.c2w[i], K=shot.K[i], width=shot.width, height=shot.height,
                               opts=opts, flow_to=nxt))
    rgb = np.stack([color.linear_to_display(o.rgb) for o in outs], 0).astype(np.float32)

    def stk(name, dtype):
        return np.stack([getattr(o, name) for o in outs], 0).astype(dtype)

    return C.Frames(
        rgb=rgb,
        fps=shot.fps,
        aov={
            "depth": stk("depth", np.float32),
            "alpha": stk("alpha", np.float32),
            "layer_id": stk("layer_id", np.int16),
            "normal": stk("normal", np.float32),
            "atom_id": stk("atom_id", np.int64),
            "flow": stk("flow", np.float32),
            "flow_valid": stk("flow_valid", bool),
        },
        shot_hash=shot.content_hash(),
        world_hash=world.content_hash(),
    )


# ---------------------------------------------------------------------------------------------------------------
# 관측 bitset
# ---------------------------------------------------------------------------------------------------------------
def _words_from_bool(flat):
    n = int(flat.size)
    nw = (n + 63) >> 6
    packed = np.packbits(flat, bitorder="little")
    buf = np.zeros(nw * 8, np.uint8)
    buf[: packed.size] = packed
    return buf.view("<u8").astype(np.uint64), n


class ObsBitset:
    """원자 i <-> 워드 i>>6 의 비트 i&63 (LSB=0). n 이후 꼬리 비트는 0."""

    __slots__ = ("words", "n")

    def __init__(self, words, n):
        n = int(n)
        words = np.array(words, dtype=np.uint64)
        if words.ndim != 1 or words.shape[0] != (n + 63) >> 6:
            raise ValueError("ObsBitset: words 길이가 n 과 맞지 않는다")
        if n & 63 and words.size and int(words[-1]) >> (n & 63):
            raise ValueError("ObsBitset: 꼬리 비트는 0 이어야 한다")
        words.setflags(write=False)
        self.words = words
        self.n = n

    @classmethod
    def from_bool(cls, flags):
        a = np.asarray(flags)
        if a.dtype != np.bool_:
            raise TypeError(f"from_bool: bool 배열이어야 한다 (dtype {a.dtype})")
        words, n = _words_from_bool(np.ascontiguousarray(a).reshape(-1))
        return cls(words, n)

    def to_bool(self):
        bits = np.unpackbits(self.words.astype("<u8").view(np.uint8), bitorder="little")
        return bits[: self.n].astype(bool)

    def count(self) -> int:
        return int(self.to_bool().sum())

    def test(self, atom_id):
        ids = _int_array("atom_id", atom_id)
        if (ids >= self.n).any():
            raise ValueError(f"test: id >= n ({self.n})")
        ok = ids >= 0
        safe = np.where(ok, ids, 0)
        if self.n == 0:
            return np.zeros(ids.shape, bool)
        w = self.words[safe >> 6]
        bit = (w >> (safe & 63).astype(np.uint64)) & np.uint64(1)
        return (bit.astype(bool) & ok).reshape(ids.shape)

    def content_hash(self) -> str:
        return det.hash_parts(b"buc.obsbitset.v1", self.n, det.hash_array(self.words)).hex()

    def _check(self, other):
        if not isinstance(other, ObsBitset):
            return NotImplemented
        if other.n != self.n:
            raise ValueError(f"ObsBitset: n 이 다르다 ({self.n} != {other.n})")
        return None

    def __or__(self, other):
        r = self._check(other)
        return r if r is NotImplemented else ObsBitset(self.words | other.words, self.n)

    def __and__(self, other):
        r = self._check(other)
        return r if r is NotImplemented else ObsBitset(self.words & other.words, self.n)

    def __sub__(self, other):
        r = self._check(other)
        return r if r is NotImplemented else ObsBitset(self.words & ~other.words, self.n)

    def __repr__(self):
        return f"ObsBitset(n={self.n}, count={self.count()})"


def observed_atoms(world, *, vis_min=0.5, alpha_eps=1e-3) -> ObsBitset:
    h, w = _hw(world)
    mask = getattr(world.obs, "mask", None)
    mask = None if mask is None else np.asarray(mask, bool)
    trans = np.ones((h, w), np.float64)
    flags = []
    for ly in world.layers:
        al = np.asarray(ly.rgba)[..., 3].astype(np.float64)
        f = (al > alpha_eps) & (trans >= vis_min)
        if mask is not None:
            f &= mask
        flags.append(f)
        trans = trans * (1.0 - al)
    if not flags:
        return ObsBitset.from_bool(np.zeros(0, bool))
    return ObsBitset.from_bool(np.stack(flags, 0))


# ---------------------------------------------------------------------------------------------------------------
# 집계·정렬
# ---------------------------------------------------------------------------------------------------------------
def atom_pixel_counts(atom_id, n) -> np.ndarray:
    ids = _int_array("atom_id", atom_id).reshape(-1)
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError(f"n: 0 이상 정수여야 한다 ({n!r})")
    n = int(n)
    if (ids >= n).any():
        raise ValueError(f"atom_pixel_counts: id >= n ({n})")
    ids = ids[ids >= 0]
    if ids.size == 0 or n == 0:
        return np.zeros(n, np.int64)
    q = det.segment_reduce(np.ones(ids.size, np.float64), ids, n, frac_bits=1)
    return (np.asarray(q, np.int64) >> 1).astype(np.int64)


def sort_fragments(pixel, depth, atom_id) -> np.ndarray:
    pixel, depth, atom_id = np.asarray(pixel), np.asarray(depth), np.asarray(atom_id)
    if pixel.dtype.kind not in "iu" or atom_id.dtype.kind not in "iu":
        raise TypeError("sort_fragments: pixel, atom_id 는 정수 dtype 이어야 한다")
    if depth.dtype.kind != "f":
        raise TypeError("sort_fragments: depth 는 float dtype 이어야 한다")
    if pixel.ndim != 1 or depth.shape != pixel.shape or atom_id.shape != pixel.shape:
        raise ValueError("sort_fragments: 1 차원이고 길이가 같아야 한다")
    if np.isnan(depth).any():
        raise ValueError("sort_fragments: depth 에 NaN")
    return np.lexsort((atom_id, depth, pixel)).astype(np.int64)
