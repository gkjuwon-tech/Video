"""CAM-07: 충돌 / 클리어런스 / 가시성 유틸 (순수 기하 질의, 학습 0).

장애물은 프리미티브 3종(Sphere, Box, HalfSpace)으로 받고, 모두 정확한 부호거리(밖 양수, 안 음수)를
돌려준다. 장면 SDF(각 SDF의 최솟값)는 1-립시츠이므로 경로 클리어런스는 구간 분할로 인증 가능한
하한/상한을 함께 돌려준다. numpy와 표준 라이브러리만 쓴다(torch 금지, 전역 RNG 없음, 입력 불변).

규약: 세계 오른손·미터, 카메라 OpenCV(x 오른쪽, y 아래, z 앞), 포즈 4x4 c2w,
카메라 위치 = c2w[:3, 3]. 영상 픽셀 중심은 정수 좌표(범위 u in [-0.5, W-0.5]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "CAMERA_RADIUS_M", "COMFORT_M", "SURF_TOL", "PATH_TOL_M", "OBSTACLE_TYPES",
    "Sphere", "Box", "HalfSpace", "obstacle_from_dict",
    "PathClearance", "ClearanceReport",
    "camera_positions", "clearance", "path_clearance", "collision_free", "free_distance",
    "segment_visible", "visibility_matrix", "visible_fraction", "depth_visible",
    "clearance_score", "evaluate_path",
]

CAMERA_RADIUS_M = 0.10
COMFORT_M = 0.5
SURF_TOL = 1e-6
PATH_TOL_M = 1e-4
OBSTACLE_TYPES = ("sphere", "box", "halfspace")

_ROT_TOL = 1e-6
_MAX_ROUNDS = 400
_MAX_FRONTIER = 2_000_000


# ---------------------------------------------------------------- 검증 도우미
def _float_array(x: Any, name: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: cannot convert to float array ({e})") from None
    return a


def _vec3(x: Any, name: str) -> np.ndarray:
    a = _float_array(x, name)
    if a.shape != (3,):
        raise ValueError(f"{name}: expected shape (3,), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be finite")
    a = a.copy()
    a.setflags(write=False)
    return a


def _points(x: Any, name: str = "points") -> np.ndarray:
    a = _float_array(x, name)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"{name}: last axis must be 3, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be finite")
    return a


def _finite_scalar(x: Any, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: must be a real number") from None
    if not math.isfinite(v):
        raise ValueError(f"{name}: must be finite")
    return v


def _check_obstacles(obstacles: Any) -> list:
    if obstacles is None:
        return []
    try:
        obs = list(obstacles)
    except TypeError:
        raise ValueError("obstacles: must be a sequence of Sphere/Box/HalfSpace") from None
    for o in obs:
        if not isinstance(o, (Sphere, Box, HalfSpace)):
            raise ValueError(f"obstacles: unsupported obstacle {type(o).__name__}")
    return obs


def _check_rotation(R: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(R)):
        raise ValueError(f"{name}: must be finite")
    if np.abs(R.T @ R - np.eye(3)).max() > _ROT_TOL or abs(np.linalg.det(R) - 1.0) > _ROT_TOL:
        raise ValueError(f"{name}: must be a proper rotation (orthonormal, det=+1)")


# ---------------------------------------------------------------- 장애물
@dataclass(frozen=True, kw_only=True, eq=False)
class Sphere:
    center: Any
    radius: float

    def __post_init__(self):
        c = _vec3(self.center, "center")
        r = _finite_scalar(self.radius, "radius")
        if not r > 0:
            raise ValueError("radius must be > 0")
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "radius", r)

    def signed_distance(self, points):
        p = _points(points)
        return np.linalg.norm(p - self.center, axis=-1) - self.radius

    def _open_interval(self, a, d):
        m = a - self.center
        A = np.sum(d * d, axis=-1)
        B = np.sum(m * d, axis=-1)
        C = np.sum(m * m, axis=-1) - self.radius ** 2
        disc = B * B - A * C
        pos = (A > 0) & (disc > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.sqrt(np.where(pos, disc, 0.0))
            Asafe = np.where(A > 0, A, 1.0)
            r0 = (-B - sq) / Asafe
            r1 = (-B + sq) / Asafe
        const_in = (A <= 0) & (C < 0)
        t0 = np.where(pos, r0, np.where(const_in, -np.inf, np.inf))
        t1 = np.where(pos, r1, np.where(const_in, np.inf, -np.inf))
        return t0, t1

    def blocks_segment(self, a, b):
        return not_visible(self, a, b)

    def to_dict(self) -> dict:
        return {"type": "sphere", "center": [float(v) for v in self.center], "radius": float(self.radius)}


@dataclass(frozen=True, kw_only=True, eq=False)
class Box:
    center: Any
    half_extents: Any
    rotation: Any = None

    def __post_init__(self):
        c = _vec3(self.center, "center")
        h = _vec3(self.half_extents, "half_extents")
        if not np.all(h > 0):
            raise ValueError("half_extents must be > 0")
        if self.rotation is None:
            R = np.eye(3)
        else:
            R = _float_array(self.rotation, "rotation")
            if R.shape != (3, 3):
                raise ValueError(f"rotation: expected shape (3, 3), got {R.shape}")
            _check_rotation(R, "rotation")
            R = R.copy()
        R.setflags(write=False)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "half_extents", h)
        object.__setattr__(self, "rotation", R)

    def signed_distance(self, points):
        p = _points(points)
        q = np.abs((p - self.center) @ self.rotation) - self.half_extents
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)

    def _open_interval(self, a, d):
        ml = (a - self.center) @ self.rotation
        dl = d @ self.rotation
        h = self.half_extents
        zero = dl == 0
        dsafe = np.where(zero, 1.0, dl)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ta = (-h - ml) / dsafe
            tb = (h - ml) / dsafe
        lo = np.minimum(ta, tb)
        hi = np.maximum(ta, tb)
        inside = np.abs(ml) < h
        lo = np.where(zero, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(zero, np.where(inside, np.inf, -np.inf), hi)
        return lo.max(axis=-1), hi.min(axis=-1)

    def blocks_segment(self, a, b):
        return not_visible(self, a, b)

    def to_dict(self) -> dict:
        return {"type": "box", "center": [float(v) for v in self.center],
                "half_extents": [float(v) for v in self.half_extents],
                "rotation": [[float(v) for v in row] for row in self.rotation]}


@dataclass(frozen=True, kw_only=True, eq=False)
class HalfSpace:
    normal: Any
    offset: float

    def __post_init__(self):
        n = _vec3(self.normal, "normal")
        off = _finite_scalar(self.offset, "offset")
        nn = float(np.linalg.norm(n))
        if not nn > 0:
            raise ValueError("normal must be non-zero")
        n = n / nn
        n.setflags(write=False)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "offset", off / nn)

    def signed_distance(self, points):
        p = _points(points)
        return p @ self.normal - self.offset

    def _open_interval(self, a, d):
        f0 = a @ self.normal - self.offset
        nd = d @ self.normal
        nz = nd != 0
        nds = np.where(nz, nd, 1.0)
        ts = -f0 / nds
        const_in = (~nz) & (f0 < 0)
        t0 = np.where(nz, np.where(nd > 0, -np.inf, ts), np.where(const_in, -np.inf, np.inf))
        t1 = np.where(nz, np.where(nd > 0, ts, np.inf), np.where(const_in, np.inf, -np.inf))
        return t0, t1

    def blocks_segment(self, a, b):
        return not_visible(self, a, b)

    def to_dict(self) -> dict:
        return {"type": "halfspace", "normal": [float(v) for v in self.normal], "offset": float(self.offset)}


_FIELDS = {
    "sphere": (("center", "radius"), ()),
    "box": (("center", "half_extents"), ("rotation",)),
    "halfspace": (("normal", "offset"), ()),
}


def obstacle_from_dict(d):
    """`to_dict` 형태(JSON)에서 장애물 복원. 필수 키 누락·모르는 키·모르는 type은 ValueError."""
    if not isinstance(d, dict):
        raise ValueError("obstacle dict expected")
    t = d.get("type")
    if t not in _FIELDS:
        raise ValueError(f"unknown obstacle type: {t!r}")
    req, opt = _FIELDS[t]
    keys = set(d) - {"type"}
    missing = [k for k in req if k not in keys]
    if missing:
        raise ValueError(f"missing keys for {t}: {missing}")
    extra = keys - set(req) - set(opt)
    if extra:
        raise ValueError(f"unknown keys for {t}: {sorted(extra)}")
    kw = {k: d[k] for k in keys}
    if t == "sphere":
        return Sphere(**kw)
    if t == "box":
        return Box(**kw)
    return HalfSpace(**kw)


# ---------------------------------------------------------------- 경로 입력
def camera_positions(x) -> np.ndarray:
    """(N,4,4)·(4,4) c2w 또는 (N,3)·(3,) 위치 -> (N,3) 복사본."""
    a = _float_array(x, "path")
    if not np.all(np.isfinite(a)):
        raise ValueError("path: must be finite")
    if a.ndim == 3 and a.shape[0] >= 1 and a.shape[1:] == (4, 4):
        T = a
    elif a.ndim == 2 and a.shape == (4, 4):
        T = a[None]
    elif a.ndim == 2 and a.shape[0] >= 1 and a.shape[1] == 3:
        return a.copy()
    elif a.ndim == 1 and a.shape == (3,):
        return a[None].copy()
    else:
        raise ValueError(f"path: unsupported shape {a.shape}")
    if np.abs(T[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max() > _ROT_TOL:
        raise ValueError("path: last row of c2w must be [0, 0, 0, 1]")
    return T[:, :3, 3].copy()


def _check_c2w(c2w, single_ok=True) -> np.ndarray:
    T = _float_array(c2w, "c2w")
    if T.shape == (4, 4):
        T = T[None]
    if T.ndim != 3 or T.shape[0] < 1 or T.shape[1:] != (4, 4):
        raise ValueError(f"c2w: expected (N,4,4) or (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("c2w: must be finite")
    if np.abs(T[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max() > _ROT_TOL:
        raise ValueError("c2w: last row must be [0, 0, 0, 1]")
    R = T[:, :3, :3]
    err = np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max()
    if err > _ROT_TOL or np.abs(np.linalg.det(R) - 1.0).max() > _ROT_TOL:
        raise ValueError("c2w: rotation part must be a proper rotation")
    return T


# ---------------------------------------------------------------- clearance
def _scene_sdf(obs: list, p: np.ndarray) -> np.ndarray:
    out = None
    for o in obs:
        v = o.signed_distance(p)
        out = v if out is None else np.minimum(out, v)
    if out is None:
        return np.full(p.shape[:-1], np.inf)
    return out


def clearance(obstacles, points):
    """장면 SDF = 각 장애물 SDF의 최솟값(없으면 +inf). (3,) 입력이면 float."""
    obs = _check_obstacles(obstacles)
    p = _points(points)
    v = _scene_sdf(obs, p)
    if p.ndim == 1:
        return float(v)
    return np.asarray(v, dtype=np.float64)


@dataclass(frozen=True)
class PathClearance:
    min_clearance: float
    lower_bound: float
    at_frame: float
    point: tuple
    tol: float


def path_clearance(obstacles, path, *, tol: float = PATH_TOL_M) -> PathClearance:
    """프레임 점을 잇는 폴리라인 전체의 최소 clearance(상한 `min_clearance`, 인증 하한 `lower_bound`)."""
    obs = _check_obstacles(obstacles)
    tol = _finite_scalar(tol, "tol")
    if not tol > 0:
        raise ValueError("tol must be > 0")
    P = camera_positions(path)
    N = P.shape[0]
    if not obs:
        return PathClearance(math.inf, math.inf, 0.0, tuple(float(v) for v in P[0]), tol)

    fP = _scene_sdf(obs, P)
    k = int(np.argmin(fP))
    U = float(fP[k])
    best_point = P[k].copy()
    best_frame = float(k)
    if N == 1:
        return PathClearance(U, U, best_frame, tuple(float(v) for v in best_point), tol)

    A, B = P[:-1].copy(), P[1:].copy()
    fa, fb = fP[:-1].copy(), fP[1:].copy()
    g0 = np.arange(N - 1, dtype=np.float64)          # 구간 시작의 소수 프레임 인덱스
    g1 = g0 + 1.0
    L = np.linalg.norm(B - A, axis=1)
    settled = math.inf
    lower_final = None

    for _ in range(_MAX_ROUNDS):
        lb = 0.5 * (fa + fb - L)
        active = lb < U - tol
        if (~active).any():
            settled = min(settled, float(lb[~active].min()))
        if not active.any():
            lower_final = settled
            break
        A, B, fa, fb, L, lb = A[active], B[active], fa[active], fb[active], L[active], lb[active]
        g0, g1 = g0[active], g1[active]
        if A.shape[0] > _MAX_FRONTIER:
            lower_final = min(settled, float(lb.min()))
            break
        s = np.clip(0.5 * (L + fb - fa), 0.25 * L, 0.75 * L)   # 하한 최소점 근처, 단 구간 길이는 반드시 줄어든다
        frac = s / np.where(L > 0, L, 1.0)
        M = A + frac[:, None] * (B - A)
        fm = _scene_sdf(obs, M)
        j = int(np.argmin(fm))
        if fm[j] < U:
            U = float(fm[j])
            best_point = M[j].copy()
            best_frame = float(g0[j] + frac[j] * (g1[j] - g0[j]))
        L1 = np.linalg.norm(M - A, axis=1)
        L2 = np.linalg.norm(B - M, axis=1)
        A, B = np.concatenate([A, M]), np.concatenate([M, B])
        fa, fb = np.concatenate([fa, fm]), np.concatenate([fm, fb])
        gm = g0 + frac * (g1 - g0)
        g0, g1 = np.concatenate([g0, gm]), np.concatenate([gm, g1])
        L = np.concatenate([L1, L2])
    if lower_final is None:
        lb = 0.5 * (fa + fb - L)
        lower_final = min(settled, float(lb.min()))

    lower = min(lower_final, U)
    best_frame = float(min(max(best_frame, 0.0), N - 1))
    return PathClearance(U, lower, best_frame, tuple(float(v) for v in best_point), tol)


def _check_nonneg(x, name: str) -> float:
    v = _finite_scalar(x, name)
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def collision_free(obstacles, path, *, camera_radius: float = CAMERA_RADIUS_M, margin: float = 0.0,
                   tol: float = PATH_TOL_M) -> bool:
    """소리(sound): 인증 하한이 camera_radius + margin 이상일 때만 True."""
    rad = _check_nonneg(camera_radius, "camera_radius")
    mar = _check_nonneg(margin, "margin")
    pc = path_clearance(obstacles, path, tol=tol)
    return bool(pc.lower_bound >= rad + mar)


def free_distance(obstacles, origin, direction, *, radius: float = CAMERA_RADIUS_M,
                  max_dist: float = 100.0, tol: float = 1e-6) -> float:
    """광선을 따라 clearance >= radius 를 유지하며 갈 수 있는 거리(구면 추적, 보수적)."""
    obs = _check_obstacles(obstacles)
    o = _vec3(origin, "origin")
    dv = _float_array(direction, "direction")
    if dv.shape != (3,) or not np.all(np.isfinite(dv)):
        raise ValueError("direction: must be a finite 3-vector")
    nrm = float(np.linalg.norm(dv))
    if not nrm > 0:
        raise ValueError("direction: must be non-zero")
    d = dv / nrm
    rad = _check_nonneg(radius, "radius")
    md = _finite_scalar(max_dist, "max_dist")
    if not md > 0:
        raise ValueError("max_dist must be > 0")
    tl = _finite_scalar(tol, "tol")
    if not tl > 0:
        raise ValueError("tol must be > 0")
    if not obs:
        return md
    t = 0.0
    for _ in range(100000):
        f = float(_scene_sdf(obs, o + t * d))
        if t == 0.0 and f < rad:
            return 0.0
        step = f - rad
        if step <= tl:
            return min(t, md)
        t += step
        if t >= md:
            return md
    return min(t, md)


# ---------------------------------------------------------------- 가시성
def segment_visible(obstacles, a, b):
    """a->b 선분이 어떤 고체의 내부도 지나지 않으면 True(양 끝 SURF_TOL 무시)."""
    obs = _check_obstacles(obstacles)
    pa = _points(a, "a")
    pb = _points(b, "b")
    pa, pb = np.broadcast_arrays(pa, pb)
    shape = pa.shape[:-1]
    if not obs:
        blocked = np.zeros(shape, dtype=bool)
    else:
        d = pb - pa
        L = np.linalg.norm(d, axis=-1)
        long = L > 2.0 * SURF_TOL
        Ls = np.where(long, L, 1.0)
        lo = np.where(long, SURF_TOL / Ls, 0.0)
        hi = 1.0 - lo
        mid = 0.5 * (pa + pb)
        blocked = np.zeros(shape, dtype=bool)
        for o in obs:
            t0, t1 = o._open_interval(pa, d)
            overlap = np.minimum(t1, hi) > np.maximum(t0, lo)
            short = o.signed_distance(mid) < 0
            blocked |= np.where(long, overlap, short)
    vis = ~blocked
    if vis.ndim == 0:
        return bool(vis)
    return vis


def not_visible(obstacle, a, b):
    r = segment_visible([obstacle], a, b)
    if isinstance(r, bool):
        return not r
    return ~r


def _check_K(K) -> np.ndarray:
    Km = _float_array(K, "K")
    if Km.shape != (3, 3) or not np.all(np.isfinite(Km)):
        raise ValueError("K: must be a finite 3x3 matrix")
    return Km


def _check_size(size):
    try:
        W, H = size
        W, H = int(W), int(H)
    except (TypeError, ValueError):
        raise ValueError("size: must be (W, H)") from None
    if W < 1 or H < 1:
        raise ValueError("size: must be positive")
    return W, H


def _project(pc: np.ndarray, K: np.ndarray):
    """카메라 좌표 (...,3) -> (u, v, z). z>0 가정 영역에서만 의미."""
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (K[0, 0] * pc[..., 0] + K[0, 1] * pc[..., 1]) / z + K[0, 2]
        v = (K[1, 0] * pc[..., 0] + K[1, 1] * pc[..., 1]) / z + K[1, 2]
    return u, v, z


def visibility_matrix(obstacles, c2w, points, *, K=None, size=None, margin_px: float = 0.0,
                      near: float = 1e-3) -> np.ndarray:
    """(N,M) bool: 프레임 i에서 점 j가 보임 (앞쪽 z>near, 영상 안, 차폐 없음)."""
    obs = _check_obstacles(obstacles)
    T = _check_c2w(c2w)
    pts = _points(points)
    if pts.ndim == 1:
        pts = pts[None]
    if pts.ndim != 2:
        raise ValueError("points: expected (M,3)")
    if (K is None) != (size is None):
        raise ValueError("K and size must be given together")
    mpx = _finite_scalar(margin_px, "margin_px")
    nr = _finite_scalar(near, "near")
    if K is not None:
        Km = _check_K(K)
        W, H = _check_size(size)

    t = T[:, :3, 3]                                   # (N,3)
    R = T[:, :3, :3]                                  # (N,3,3)
    rel = pts[None, :, :] - t[:, None, :]             # (N,M,3)
    pc = np.einsum("nmi,nij->nmj", rel, R)            # R^T (p - t)
    vis = pc[..., 2] > nr
    if K is not None:
        u, v, _ = _project(pc, Km)
        vis &= (u >= -0.5 + mpx) & (u <= W - 0.5 - mpx) & (v >= -0.5 + mpx) & (v <= H - 0.5 - mpx)
    if obs:
        occ = segment_visible(obs, t[:, None, :], pts[None, :, :])
        vis &= np.asarray(occ, dtype=bool)
    return np.asarray(vis, dtype=bool)


def visible_fraction(obstacles, c2w, points, *, K=None, size=None, margin_px: float = 0.0,
                     near: float = 1e-3) -> np.ndarray:
    m = visibility_matrix(obstacles, c2w, points, K=K, size=size, margin_px=margin_px, near=near)
    return m.mean(axis=1)


def depth_visible(points, K, c2w, depth, *, rel_tol: float = 0.01, abs_tol: float = 0.0) -> np.ndarray:
    """렌더 깊이 버퍼 z-버퍼 검사. 단일 포즈 c2w, depth (H,W)."""
    pts = _points(points)
    if pts.ndim == 1:
        pts = pts[None]
    if pts.ndim != 2:
        raise ValueError("points: expected (M,3)")
    Km = _check_K(K)
    T = _check_c2w(c2w)
    if T.shape[0] != 1:
        raise ValueError("c2w: single pose expected")
    D = _float_array(depth, "depth")
    if D.ndim != 2 or D.size == 0:
        raise ValueError("depth: expected (H,W)")
    if np.isnan(D).any():
        raise ValueError("depth: NaN not allowed")
    rt = _finite_scalar(rel_tol, "rel_tol")
    at = _finite_scalar(abs_tol, "abs_tol")
    H, W = D.shape
    pc = (pts - T[0, :3, 3]) @ T[0, :3, :3]
    u, v, z = _project(pc, Km)
    front = z > 0
    ok = front & np.isfinite(u) & np.isfinite(v)
    ui = np.floor(np.where(ok, u, 0.0) + 0.5)
    vi = np.floor(np.where(ok, v, 0.0) + 0.5)
    ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    uc = np.where(ok, ui, 0).astype(np.int64)
    vc = np.where(ok, vi, 0).astype(np.int64)
    d = D[vc, uc]
    with np.errstate(invalid="ignore"):
        lim = d * (1.0 + rt) + at
    return np.asarray(ok & (z <= lim), dtype=bool)


# ---------------------------------------------------------------- 점수·보고서
def clearance_score(min_clearance, *, radius: float = CAMERA_RADIUS_M, comfort_m: float = COMFORT_M):
    """x = clip((c - radius)/comfort_m, 0, 1), 3x^2 - 2x^3 (단조, inf -> 1)."""
    r = _finite_scalar(radius, "radius")
    w = _finite_scalar(comfort_m, "comfort_m")
    if r < 0:
        raise ValueError("radius must be >= 0")
    if not w > 0:
        raise ValueError("comfort_m must be > 0")
    c = _float_array(min_clearance, "min_clearance")
    if np.isnan(c).any():
        raise ValueError("min_clearance: NaN not allowed")
    with np.errstate(invalid="ignore"):
        x = np.clip((c - r) / w, 0.0, 1.0)
    y = 3.0 * x * x - 2.0 * x * x * x
    if y.ndim == 0:
        return float(y)
    return y


@dataclass(frozen=True)
class ClearanceReport:
    ok: bool
    collision: bool
    min_clearance: float
    lower_bound: float
    worst_frame: float
    score: float
    min_visible_fraction: Optional[float]


def evaluate_path(obstacles, c2w, *, camera_radius: float = CAMERA_RADIUS_M, margin: float = 0.0,
                  comfort_m: float = COMFORT_M, subject_points=None, K=None, size=None,
                  min_visible: float = 0.0, tol: float = PATH_TOL_M) -> ClearanceReport:
    rad = _check_nonneg(camera_radius, "camera_radius")
    mar = _check_nonneg(margin, "margin")
    mv = _finite_scalar(min_visible, "min_visible")
    pc = path_clearance(obstacles, c2w, tol=tol)
    collision = not (pc.lower_bound >= rad + mar)
    score = clearance_score(pc.min_clearance, radius=rad, comfort_m=comfort_m)
    mvf = None
    if subject_points is not None:
        fr = visible_fraction(obstacles, c2w, subject_points, K=K, size=size)
        mvf = float(fr.min())
    ok = (not collision) and (mvf is None or mvf >= mv)
    return ClearanceReport(ok=bool(ok), collision=bool(collision), min_clearance=pc.min_clearance,
                           lower_bound=pc.lower_bound, worst_frame=pc.at_frame, score=float(score),
                           min_visible_fraction=mvf)
