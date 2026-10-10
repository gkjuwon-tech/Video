"""RD-02 R1 기하 카메라: LSD 계열 선분 -> plumb-line 왜곡 -> J-Linkage 소실점 -> 맨해튼 프레임 + K 공동 LM -> 수평선.

수납된 영상 한 장(RD-01 ``image_lin``, 크롭 좌표)에서 K(f, 주점), 방사 왜곡 k1·k2, 수평선(roll, pitch), 맨해튼 프레임을
고전 기하로 푼다. 사전값(FOV·수평선·주점)은 초기값과 약한 사전일 뿐이고 소실점(VP) 증거가 있으면 VP가 이긴다.
AI 영상의 원근 불성립(고아 VP)은 맞추지 않고 ``orphan_frac``·``r_vp_deg`` 로 측정해 보고한다.

규약: 픽셀 중심 정수, 주점 기본 ((W-1)/2,(H-1)/2), 카메라 OpenCV(x 오른쪽, y 아래, z 앞), K 는 fx=fy·skew 0.
up_cam = 세계 위쪽의 카메라 좌표 단위벡터, pitch=asin(up[2]), roll=atan2(up[0],-up[1]).

numpy · scipy 만 쓴다(torch · cv2 · skimage 금지). import 부작용 없음, 전역 RNG 불변(지역 default_rng(seed)),
순수·결정론·입력 불변.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage, optimize

from buc.core import geom as _geom

__all__ = [
    "GeomCamError", "DEFAULT_FOV_X_DEG", "GeomCamOpts", "PlumbFit", "GeomCamera",
    "detect_segments", "jlinkage", "segment_vp_residual_deg", "focal_from_vps", "principal_point_from_vps",
    "fit_plumb_line", "up_from_roll_pitch", "roll_pitch_from_up", "horizon_line",
    "estimate_from_segments", "estimate_geom_camera",
]

DEFAULT_FOV_X_DEG = 50.0

_RAD = math.pi / 180.0
_MIN_SIDE = 32
_FOV_RANGE = (8.0, 130.0)        # 후보 프레임이 허용하는 가로 FOV 범위(도)
_JL_MAX_SEGS = 500               # J-Linkage 가설 합의에 쓰는 최장 선분 수
_MAX_SEGS = 3000                 # 선분 상한(길이 내림차순)
_SIGMA_ANG_DEG = 0.5             # 공동 LM 의 선분 각도 잡음(도)
_FRAME_DEV_TOL_DEG = 5.0         # 세 VP 직교성 허용(도, 고정 주점 기준)
_FRAME_DEV_TAU_DEG = 4.0


class GeomCamError(ValueError):
    """너무 작은 영상(짧은 변 < 32) 등 기하 카메라를 풀 수 없는 입력."""


# ---------------------------------------------------------------------------------------------------------------
# 옵션 / 결과 타입
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class GeomCamOpts:
    max_side: int = 1024
    min_seg_frac: float = 0.01
    grad_thresh: float = 0.02
    n_hypotheses: int = 500
    inlier_deg: float = 2.0
    min_cluster: int = 4
    fit_distortion: bool = True
    k_bound: float = 0.3
    min_plumb_lines: int = 3
    orphan_warn: float = 0.10
    r_vp_warn_deg: float = 2.0
    pp_free_max_dist: float = 4.0
    pp_reject_frac: float = 0.2
    prior_sigma_log_f: float = 0.15
    seed: int = 0

    def __post_init__(self):
        for name in ("min_seg_frac", "grad_thresh", "inlier_deg", "k_bound", "orphan_warn", "r_vp_warn_deg",
                     "pp_free_max_dist", "pp_reject_frac", "prior_sigma_log_f"):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name}={v!r}: 숫자여야 한다")
            if not math.isfinite(float(v)):
                raise ValueError(f"{name}={v!r}: 유한해야 한다")
        for name in ("max_side", "n_hypotheses", "min_cluster", "min_plumb_lines", "seed"):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"{name}={v!r}: 정수여야 한다")
        if self.max_side < 64:
            raise ValueError(f"max_side={self.max_side}: 64 이상이어야 한다")
        if not self.grad_thresh > 0:
            raise ValueError(f"grad_thresh={self.grad_thresh}: 양수여야 한다")
        if not self.inlier_deg > 0:
            raise ValueError(f"inlier_deg={self.inlier_deg}: 양수여야 한다")
        if not self.prior_sigma_log_f > 0:
            raise ValueError(f"prior_sigma_log_f={self.prior_sigma_log_f}: 양수여야 한다")
        if self.n_hypotheses < 1:
            raise ValueError("n_hypotheses 는 1 이상이어야 한다")
        if self.min_cluster < 2:
            raise ValueError("min_cluster 는 2 이상이어야 한다")
        if self.k_bound < 0:
            raise ValueError("k_bound 는 0 이상이어야 한다")
        if not 0.0 <= self.orphan_warn <= 1.0:
            raise ValueError("orphan_warn 은 [0,1] 이어야 한다")
        if self.min_seg_frac < 0 or self.r_vp_warn_deg < 0 or self.pp_free_max_dist < 0 or self.pp_reject_frac < 0:
            raise ValueError("음수 옵션")
        if self.min_plumb_lines < 1:
            raise ValueError("min_plumb_lines 는 1 이상이어야 한다")


@dataclass(frozen=True)
class PlumbFit:
    k1: float
    k2: float
    rms_px: float
    n_lines: int


@dataclass(frozen=True, kw_only=True, eq=False)
class GeomCamera:
    W: int
    H: int
    K: np.ndarray
    fov_x_deg: float
    k1: float
    k2: float
    roll_deg: float
    pitch_deg: float
    up_cam: np.ndarray
    R_mc: Optional[np.ndarray]
    vps: np.ndarray
    vp_axis: np.ndarray
    segments: np.ndarray
    labels: np.ndarray
    n_manhattan_vps: int
    f_source: str
    pp_source: str
    horizon_source: str
    r_vp_deg: float
    orphan_frac: float
    plumb_rms_px: float
    reliable: bool
    warnings: Tuple[str, ...]

    def horizon_line(self) -> np.ndarray:
        return horizon_line(self.K, self.up_cam)

    def theta_fields(self, sensor_width_mm: float = 36.0) -> dict:
        s = float(sensor_width_mm)
        return {"focal_mm": float(self.K[0, 0]) / self.W * s, "sensor_width_mm": s, "aspect": self.W / self.H}

    def to_meta(self) -> dict:
        def fl(x):
            x = float(x)
            return x if math.isfinite(x) else None

        return {
            "W": int(self.W), "H": int(self.H),
            "K": [[float(v) for v in row] for row in self.K],
            "fov_x_deg": fl(self.fov_x_deg), "k1": fl(self.k1), "k2": fl(self.k2),
            "roll_deg": fl(self.roll_deg), "pitch_deg": fl(self.pitch_deg),
            "up_cam": [float(v) for v in self.up_cam],
            "n_manhattan_vps": int(self.n_manhattan_vps),
            "f_source": str(self.f_source), "pp_source": str(self.pp_source),
            "horizon_source": str(self.horizon_source),
            "r_vp_deg": fl(self.r_vp_deg), "orphan_frac": fl(self.orphan_frac),
            "plumb_rms_px": fl(self.plumb_rms_px),
            "n_segments": int(len(self.segments)),
            "reliable": bool(self.reliable), "warnings": [str(w) for w in self.warnings],
        }


# ---------------------------------------------------------------------------------------------------------------
# 입력 검사 / 보조
# ---------------------------------------------------------------------------------------------------------------
def _opts(opts) -> GeomCamOpts:
    if opts is None:
        return GeomCamOpts()
    if not isinstance(opts, GeomCamOpts):
        raise TypeError("opts 는 GeomCamOpts 여야 한다")
    return opts


def _f64_array(x, name):
    a = np.asarray(x)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    return a.astype(np.float64)


def _check_segments(segments, name="segments") -> np.ndarray:
    S = _f64_array(segments, name)
    if S.ndim != 2 or S.shape[1] != 4:
        raise ValueError(f"{name}: shape (N,4) 이어야 한다 (got {S.shape})")
    if not np.isfinite(S).all():
        raise ValueError(f"{name}: 비유한 값")
    return S


def _lengths(S):
    return np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])


def _unit_vp(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not n > 0:
        return v
    v = v / n
    # 부호 고정: w >= 0 (유한점이면 양의 w)
    if v[2] < 0 or (v[2] == 0 and (v[0] < 0 or (v[0] == 0 and v[1] < 0))):
        v = -v
    return v


def _is_finite_vp(v, tol=1e-9):
    v = np.asarray(v, dtype=np.float64)
    return bool(np.isfinite(v).all() and abs(v[2]) > tol * max(1e-300, float(np.hypot(v[0], v[1]))))


def _seg_dir_mid(S):
    d = S[:, 2:] - S[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    d = d / np.maximum(L, 1e-300)[:, None]
    m = 0.5 * (S[:, :2] + S[:, 2:])
    return d, m, L


def _sin_resid(v, m, d):
    """VP 동차 v (3,)/(3,n) 와 선분(중점 m, 단위방향 d)의 각 잔차의 sin (부호 있음)."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        u0 = v[0] - m[:, 0] * v[2]
        u1 = v[1] - m[:, 1] * v[2]
    else:
        u0 = v[0] - m[:, 0] * v[2]
        u1 = v[1] - m[:, 1] * v[2]
    nu = np.hypot(u0, u1)
    cr = d[:, 0] * u1 - d[:, 1] * u0
    return np.where(nu > 1e-12, cr / np.maximum(nu, 1e-12), 0.0)


# ---------------------------------------------------------------------------------------------------------------
# 해석 기본 함수
# ---------------------------------------------------------------------------------------------------------------
def segment_vp_residual_deg(segments, vp) -> np.ndarray:
    """선분 방향과 '중점 -> VP' 방향 사이의 각(0–90°). vp: 동차 (3,) (w=0 이면 무한원점) 또는 픽셀 (2,)."""
    S = _check_segments(segments)
    v = _f64_array(vp, "vp").reshape(-1)
    if v.shape[0] == 2:
        v = np.array([v[0], v[1], 1.0])
    if v.shape != (3,):
        raise ValueError("vp: shape (3,) 또는 (2,)")
    d, m, _ = _seg_dir_mid(S)
    s = np.abs(_sin_resid(v, m, d))
    return np.degrees(np.arcsin(np.minimum(1.0, s)))


def _vp_pixel(v):
    """(2,) 픽셀 VP 또는 (3,) 동차 -> 유한 픽셀 (2,) 또는 None."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        return None
    if v.shape[0] == 2:
        return v
    if v.shape[0] == 3:
        if not _is_finite_vp(v):
            return None
        return v[:2] / v[2]
    return None


def focal_from_vps(v1, v2, c) -> Optional[float]:
    """직교 두 VP 로부터 f^2 = -(v1-c)·(v2-c). <=0 · 무한원점 · 비유한이면 None."""
    a = _vp_pixel(v1)
    b = _vp_pixel(v2)
    cc = np.asarray(c, dtype=np.float64).reshape(-1)
    if a is None or b is None or cc.shape != (2,) or not np.isfinite(cc).all():
        return None
    f2 = -float((a - cc) @ (b - cc))
    if not (math.isfinite(f2) and f2 > 0.0):
        return None
    return math.sqrt(f2)


def principal_point_from_vps(v1, v2, v3) -> np.ndarray:
    """세 유한 VP 삼각형의 수심(주점)."""
    pts = [_vp_pixel(v) for v in (v1, v2, v3)]
    if any(p is None for p in pts):
        raise ValueError("세 VP 가 모두 유한해야 한다")
    a, b, c = pts
    # (x-a)·(b-c)=0, (x-b)·(a-c)=0
    A = np.array([b - c, a - c], dtype=np.float64)
    rhs = np.array([a @ (b - c), b @ (a - c)])
    try:
        x = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as e:
        raise ValueError("퇴화한 삼각형") from e
    return x.astype(np.float64)


def up_from_roll_pitch(roll_deg, pitch_deg) -> np.ndarray:
    r = float(roll_deg) * _RAD
    p = float(pitch_deg) * _RAD
    return np.array([math.cos(p) * math.sin(r), -math.cos(p) * math.cos(r), math.sin(p)], dtype=np.float64)


def roll_pitch_from_up(up) -> Tuple[float, float]:
    u = _f64_array(up, "up").reshape(-1)
    if u.shape != (3,) or not np.isfinite(u).all():
        raise ValueError("up: 유한한 (3,)")
    n = np.linalg.norm(u)
    if not n > 0:
        raise ValueError("up: 영벡터")
    u = u / n
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(u[2])))))
    roll = math.degrees(math.atan2(float(u[0]), float(-u[1])))
    return roll, pitch


def horizon_line(K, up) -> np.ndarray:
    """수평선 l = K^{-T} up, ‖l[:2]‖=1 (양의 배율). l·[x,y,1] > 0 이 하늘 쪽."""
    K = _f64_array(K, "K")
    u = _f64_array(up, "up").reshape(-1)
    if K.shape != (3, 3) or u.shape != (3,):
        raise ValueError("K (3,3), up (3,)")
    l = np.linalg.solve(K.T, u)
    n = float(np.hypot(l[0], l[1]))
    if n < 1e-12:
        n = float(np.linalg.norm(l))
    return (l / n) if n > 0 else l


# ---------------------------------------------------------------------------------------------------------------
# 선분 검출 (LSD 계열: 기울기 -> NMS -> 방향 일관 영역 성장 -> 직선성 검사/분할)
# ---------------------------------------------------------------------------------------------------------------
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_REGION_TOL_COS = math.cos(math.radians(22.5))
_STRAIGHT_SIGMA = 0.7


def _edge_pixels(g: np.ndarray, thr: float):
    """NMS 된 서브픽셀 에지점: (pos (E,2) [x,y], theta (E,), mag (E,))."""
    H, W = g.shape
    sm = ndimage.gaussian_filter(g, 1.0, mode="nearest")
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    cand = mag > thr
    cand[0, :] = False
    cand[-1, :] = False
    cand[:, 0] = False
    cand[:, -1] = False
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    m0 = mag[ys, xs]
    nx = gx[ys, xs] / m0
    ny = gy[ys, xs] / m0
    yf = ys.astype(np.float64)
    xf = xs.astype(np.float64)
    mp = ndimage.map_coordinates(mag, [yf + ny, xf + nx], order=1, mode="nearest")
    mm = ndimage.map_coordinates(mag, [yf - ny, xf - nx], order=1, mode="nearest")
    keep = (m0 >= mp) & (m0 > mm)
    ys, xs, m0, nx, ny, mp, mm = ys[keep], xs[keep], m0[keep], nx[keep], ny[keep], mp[keep], mm[keep]
    den = mm - 2.0 * m0 + mp
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den < -1e-12, 0.5 * (mm - mp) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    pos = np.stack([xs + delta * nx, ys + delta * ny], axis=1)
    theta = np.arctan2(ny, nx)
    return pos, theta, m0


def _grow_regions(H: int, W: int, pos: np.ndarray, theta: np.ndarray, mag: np.ndarray, min_pixels: int):
    """방향이 영역 평균과 22.5° 안인 8-이웃 영역 성장(LSD 식). 영역 = 에지점 인덱스 배열의 목록."""
    E = len(theta)
    ys = np.rint(pos[:, 1]).astype(np.int64)
    xs = np.rint(pos[:, 0]).astype(np.int64)
    # 서브픽셀 이동은 +-0.5 안이므로 반올림하면 원래 픽셀이다
    grid = np.full((H + 2, W + 2), -1, dtype=np.int64)
    grid[ys + 1, xs + 1] = np.arange(E)
    nbr = np.stack([grid[ys + 1 + dy, xs + 1 + dx] for dy, dx in _N8], axis=1).tolist()
    cs = np.cos(theta).tolist()
    sn = np.sin(theta).tolist()
    order = np.argsort(-mag, kind="stable").tolist()
    used = [False] * E
    tol = _REGION_TOL_COS
    regions = []
    for s in order:
        if used[s]:
            continue
        used[s] = True
        region = [s]
        sx, sy = cs[s], sn[s]
        mx, my = sx, sy
        i = 0
        while i < len(region):
            p = region[i]
            i += 1
            for q in nbr[p]:
                if q >= 0 and not used[q]:
                    if cs[q] * mx + sn[q] * my >= tol:
                        used[q] = True
                        region.append(q)
                        sx += cs[q]
                        sy += sn[q]
                        h = math.hypot(sx, sy)
                        mx, my = sx / h, sy / h
        if len(region) >= min_pixels:
            regions.append(np.asarray(region, dtype=np.int64))
    return regions


def _pca_line(P: np.ndarray):
    c = P.mean(axis=0)
    Q = P - c
    sxx = float((Q[:, 0] ** 2).mean())
    syy = float((Q[:, 1] ** 2).mean())
    sxy = float((Q[:, 0] * Q[:, 1]).mean())
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    d = np.array([math.cos(ang), math.sin(ang)])
    t = Q @ d
    perp = Q @ np.array([-d[1], d[0]])
    return c, d, t, perp


def _split_straight(P: np.ndarray, min_len: float, out: list, depth: int = 0):
    """점열 P 를 직선성 기준 (표준편차 <= _STRAIGHT_SIGMA) 으로 재귀 분할해 선분 (4,) 을 out 에 쌓는다."""
    n = len(P)
    if n < 4:
        return
    c, d, t, perp = _pca_line(P)
    L = float(t.max() - t.min())
    if L < min_len:
        return
    sig = float(math.sqrt(float((perp ** 2).mean())))
    if sig <= _STRAIGHT_SIGMA or depth >= 7 or n < 12:
        if n >= 0.4 * L and sig <= 2.0 * _STRAIGHT_SIGMA:
            a = c + d * t.min()
            b = c + d * t.max()
            out.append(np.array([a[0], a[1], b[0], b[1]]))
        return
    order = np.argsort(t, kind="stable")
    Ps = P[order]
    k = max(1, n // 20)
    e0 = Ps[:k].mean(axis=0)
    e1 = Ps[-k:].mean(axis=0)
    ch = e1 - e0
    cn = np.hypot(ch[0], ch[1])
    if cn < 1e-9:
        return
    dist = np.abs((Ps[:, 0] - e0[0]) * ch[1] - (Ps[:, 1] - e0[1]) * ch[0]) / cn
    i = int(np.argmax(dist))
    i = min(max(i, 3), n - 3)
    _split_straight(Ps[: i + 1], min_len, out, depth + 1)
    _split_straight(Ps[i:], min_len, out, depth + 1)


def _piece_polyline(P: np.ndarray, c, d, t) -> np.ndarray:
    """직선 조각의 에지점을 주축(t) 순서의 분위 구간 평균 점열로 줄인다."""
    n = len(P)
    order = np.argsort(t, kind="stable")
    Ps = P[order]
    nb = int(min(10, max(3, n // 5)))
    edges = np.linspace(0, n, nb + 1).astype(np.int64)
    return np.asarray([Ps[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:]) if b > a], dtype=np.float64)


def _link_pieces(pieces, diag: float):
    """끝점이 가깝고 방향이 이어지는 직선 조각을 연결해 폴리라인 목록을 만든다(체커 선처럼 극성이 바뀌어 끊긴 선 포함)."""
    R = len(pieces)
    if R == 0:
        return []
    gmax = max(8.0, 0.015 * diag)
    ends = np.zeros((2 * R, 2))
    tang = np.zeros((2 * R, 2))   # 바깥쪽 접선
    for i, (poly, d) in enumerate(pieces):
        ends[2 * i] = poly[0]
        ends[2 * i + 1] = poly[-1]
        tang[2 * i] = -d
        tang[2 * i + 1] = d
    # poly 는 d 방향(t 증가) 순서이므로 poly[0] = 시작(-d 쪽), poly[-1] = 끝(+d 쪽)
    diff = ends[None, :, :] - ends[:, None, :]            # [a,b] = ends[b]-ends[a]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    ia, ib = np.nonzero(np.triu(dist <= gmax, 1))
    keep = (ia // 2) != (ib // 2)
    ia, ib = ia[keep], ib[keep]
    cand = []
    for a, b in zip(ia.tolist(), ib.tolist()):
        v = ends[b] - ends[a]
        g = float(dist[a, b])
        ta, tb = tang[a], tang[b]
        cosd = float(ta @ (-tb))
        if cosd < math.cos(math.radians(10.0)):
            continue
        if g > 1.0:
            u = v / g
            if float(u @ ta) < math.cos(math.radians(35.0)) or float(-u @ tb) < math.cos(math.radians(35.0)):
                continue
            off = max(abs(float(v[0] * ta[1] - v[1] * ta[0])), abs(float(v[0] * tb[1] - v[1] * tb[0])))
            if off > 3.0:
                continue
        cand.append((g + 20.0 * (1.0 - cosd), a, b))
    cand.sort()
    parent = list(range(R))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    link = {}
    for _, a, b in cand:
        if a in link or b in link:
            continue
        ra, rb = find(a // 2), find(b // 2)
        if ra == rb:
            continue
        parent[ra] = rb
        link[a] = b
        link[b] = a
    seen = [False] * R
    chains = []
    for i in range(R):
        if seen[i]:
            continue
        # 사슬의 한쪽 끝 조각을 찾는다: i 에서 반대편 끝 쪽으로 걸어간다
        # 먼저 한 방향 끝까지 이동
        cur, enter = i, 0          # enter 끝(0: poly[0] 쪽)으로 들어왔다고 가정
        guard = 0
        while (2 * cur + enter) in link and guard < R + 2:
            other = link[2 * cur + enter]
            cur, enter = other // 2, other % 2
            guard += 1
        # cur 는 사슬 한 끝; enter 끝은 연결 없음. 이제 반대쪽으로 순회
        pts = []
        guard = 0
        while guard < R + 2:
            seen[cur] = True
            poly = pieces[cur][0]
            pts.append(poly if enter == 0 else poly[::-1])
            exit_end = 1 - enter
            nxt = link.get(2 * cur + exit_end)
            if nxt is None:
                break
            cur, enter = nxt // 2, nxt % 2
            guard += 1
        chains.append(np.vstack(pts))
    return chains


def _limit_bend(ch: np.ndarray, min_len: float, out: list, depth: int = 0):
    """연결된 사슬이 누적해서 크게 휘면(현에서 최대 편차 > 0.06·길이) 가장 먼 점에서 나눈다."""
    n = len(ch)
    if n < 4:
        return
    chord = ch[-1] - ch[0]
    L = float(np.hypot(*chord))
    if L < min_len:
        return
    dev = np.abs((ch[:, 0] - ch[0, 0]) * chord[1] - (ch[:, 1] - ch[0, 1]) * chord[0]) / L
    i = int(np.argmax(dev))
    if float(dev[i]) <= 0.06 * L or depth >= 6:
        out.append(ch)
        return
    i = min(max(i, 2), n - 3)
    _limit_bend(ch[: i + 1], min_len, out, depth + 1)
    _limit_bend(ch[i:], min_len, out, depth + 1)


def _merge_collinear(S: np.ndarray) -> np.ndarray:
    """거의 같은 직선 위에서 이어지거나 겹치는 선분을 병합한다(각 <=1.5°, 수직 거리 <=1 px, 틈 <=max(3, 0.1·짧은 길이))."""
    n = len(S)
    if n < 2:
        return S
    alive = np.ones(n, dtype=bool)
    segs = S.copy()
    out = []
    order = np.argsort(-_lengths(S), kind="stable")
    for idx in order:
        if not alive[idx]:
            continue
        alive[idx] = False
        cur = segs[idx].copy()
        changed = True
        while changed:
            changed = False
            cand = np.nonzero(alive)[0]
            if len(cand) == 0:
                break
            p0 = cur[:2]
            dvec = cur[2:] - cur[:2]
            L = float(np.hypot(*dvec))
            if L < 1e-9:
                break
            d = dvec / L
            nrm = np.array([-d[1], d[0]])
            C = segs[cand]
            cd = C[:, 2:] - C[:, :2]
            cl = np.hypot(cd[:, 0], cd[:, 1])
            cdn = cd / np.maximum(cl, 1e-12)[:, None]
            ang = np.degrees(np.arccos(np.clip(np.abs(cdn @ d), 0.0, 1.0)))
            da = (C[:, :2] - p0) @ nrm
            db = (C[:, 2:] - p0) @ nrm
            ta = (C[:, :2] - p0) @ d
            tb = (C[:, 2:] - p0) @ d
            lo = np.minimum(ta, tb)
            hi = np.maximum(ta, tb)
            gap = np.maximum(lo - L, 0.0 - hi)
            gap = np.maximum(gap, 0.0)
            ok = (ang <= 1.5) & (np.abs(da) <= 1.0) & (np.abs(db) <= 1.0) & (gap <= np.maximum(3.0, 0.1 * np.minimum(cl, L)))
            if ok.any():
                sel = cand[ok]
                alive[sel] = False
                pieces = np.vstack([cur[None, :], segs[sel]])
                pl = _lengths(pieces)
                pts = np.vstack([pieces[:, :2], pieces[:, 2:], 0.5 * (pieces[:, :2] + pieces[:, 2:])])
                w = np.concatenate([pl, pl, 2.0 * pl])
                cc = (pts * w[:, None]).sum(axis=0) / w.sum()
                Q = pts - cc
                sxx = (w * Q[:, 0] ** 2).sum()
                syy = (w * Q[:, 1] ** 2).sum()
                sxy = (w * Q[:, 0] * Q[:, 1]).sum()
                a = 0.5 * math.atan2(2 * sxy, sxx - syy)
                dd = np.array([math.cos(a), math.sin(a)])
                if dd @ d < 0:
                    dd = -dd
                allpts = np.vstack([pieces[:, :2], pieces[:, 2:]])
                tt = (allpts - cc) @ dd
                cur = np.concatenate([cc + dd * tt.min(), cc + dd * tt.max()])
                changed = True
        out.append(cur)
    return np.asarray(out, dtype=np.float64)


def _detect(g: np.ndarray, opts: GeomCamOpts):
    """(segments (N,4) 길이 내림차순, chains list[(n,2)])."""
    H, W = g.shape
    diag = math.hypot(W, H)
    min_len = max(8.0, opts.min_seg_frac * diag)
    if H < 5 or W < 5:
        return np.zeros((0, 4)), []
    pos, theta, mag = _edge_pixels(g, opts.grad_thresh)
    if len(theta) > 400000:
        top = np.argsort(-mag, kind="stable")[:400000]
        top.sort()
        pos, theta, mag = pos[top], theta[top], mag[top]
    if len(theta) == 0:
        return np.zeros((0, 4)), []
    regions = _grow_regions(H, W, pos, theta, mag, max(5, int(0.5 * min_len)))
    segs: list = []
    pieces: list = []
    for r in regions:
        P = pos[r]
        c, d, t, perp = _pca_line(P)
        L = float(t.max() - t.min())
        if L < min_len:
            continue
        sig = float(math.sqrt(float((perp ** 2).mean())))
        if L >= 12.0 and sig <= 0.05 * L and len(r) >= 0.4 * L:
            pieces.append((_piece_polyline(P, c, d, t), d))
        _split_straight(P, min_len, segs)
    chain_min = max(30.0, 0.04 * diag)
    chains = []
    for ch in _link_pieces(pieces, diag):
        _limit_bend(ch, chain_min, chains)
    if not segs:
        return np.zeros((0, 4)), chains
    S = np.asarray(segs, dtype=np.float64)
    if len(S) > 4000:
        S = S[np.argsort(-_lengths(S), kind="stable")[:4000]]
    S = _merge_collinear(S)
    # 영상 영역 안으로 자르기(끝점이 경계 밖이면 선분을 따라 줄임)
    S = _clip_segments(S, W, H)
    L = _lengths(S)
    S = S[L >= min_len - 1e-9]
    L = _lengths(S)
    order = np.lexsort((S[:, 3], S[:, 2], S[:, 1], S[:, 0], -L))
    S = S[order][:_MAX_SEGS]
    chains.sort(key=lambda p: -float(np.hypot(*(p[-1] - p[0]))))
    return S, chains[:300]


def _clip_segments(S: np.ndarray, W: int, H: int) -> np.ndarray:
    lo = np.array([-0.5, -0.5])
    hi = np.array([W - 0.5, H - 0.5])
    S = S.copy()
    for i in range(len(S)):
        a = S[i, :2]
        b = S[i, 2:]
        d = b - a
        t0, t1 = 0.0, 1.0
        for k in range(2):
            if abs(d[k]) < 1e-12:
                continue
            ta = (lo[k] - a[k]) / d[k]
            tb = (hi[k] - a[k]) / d[k]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
        if t1 < t0:
            S[i] = [a[0], a[1], a[0], a[1]]
            continue
        S[i, :2] = a + t0 * d
        S[i, 2:] = a + t1 * d
    return S


def detect_segments(gray, *, opts=None) -> np.ndarray:
    """LSD 계열 선분 검출. (N,4) float64 [x0,y0,x1,y1], 길이 >= max(8, min_seg_frac·대각), 길이 내림차순."""
    opts = _opts(opts)
    g = _f64_array(gray, "gray")
    if g.ndim != 2:
        raise ValueError("gray: (H,W) 이어야 한다")
    if not np.isfinite(g).all():
        raise ValueError("gray: 비유한 값")
    S, _ = _detect(g, opts)
    return S


# ---------------------------------------------------------------------------------------------------------------
# J-Linkage
# ---------------------------------------------------------------------------------------------------------------
def _lines_h(S):
    p0 = np.concatenate([S[:, :2], np.ones((len(S), 1))], axis=1)
    p1 = np.concatenate([S[:, 2:], np.ones((len(S), 1))], axis=1)
    return np.cross(p0, p1)


def _refit_vp(S: np.ndarray, v0: np.ndarray, iters: int = 5) -> np.ndarray:
    """길이 가중 각 잔차의 IRLS 동차 최소제곱. 반환: 단위 동차 VP."""
    if len(S) == 0:
        return _unit_vp(v0)
    lines = _lines_h(S)
    d, m, _ = _seg_dir_mid(S)
    v = _unit_vp(v0)
    for _ in range(iters):
        u0 = v[0] - m[:, 0] * v[2]
        u1 = v[1] - m[:, 1] * v[2]
        nu = np.maximum(np.hypot(u0, u1), 1e-9)
        A = lines / nu[:, None]
        M = A.T @ A
        w, V = np.linalg.eigh(M)
        vn = V[:, 0]
        if not np.isfinite(vn).all():
            break
        v = _unit_vp(vn)
    return v


def jlinkage(segments, *, opts=None):
    """J-Linkage 소실점 군집. 반환: (vps (M,3) 단위 동차, labels (N,)), 군집 크기 >= min_cluster, 총 길이 내림차순."""
    opts = _opts(opts)
    S = _check_segments(segments)
    N = len(S)
    labels = np.full(N, -1, dtype=np.int64)
    L = _lengths(S)
    good = np.nonzero(L > 1e-9)[0]
    if len(good) < opts.min_cluster:
        return np.zeros((0, 3)), labels
    tol = opts.inlier_deg * _RAD
    rng = np.random.default_rng(opts.seed)
    # 가설: 길이 가중으로 뽑은 선분 쌍의 교점
    order = good[np.argsort(-L[good], kind="stable")][:_JL_MAX_SEGS]
    n = len(order)
    Ssub = S[order]
    lines = _lines_h(Ssub)
    p = L[order] / L[order].sum()
    pairs = rng.choice(n, size=(opts.n_hypotheses, 2), p=p)
    same = pairs[:, 0] == pairs[:, 1]
    pairs[same, 1] = (pairs[same, 1] + 1) % n
    hyp = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
    hn = np.linalg.norm(hyp, axis=1)
    hyp = hyp[hn > 1e-9] / hn[hn > 1e-9, None]
    if len(hyp) == 0:
        return np.zeros((0, 3)), labels
    d, m, _ = _seg_dir_mid(Ssub)
    # 잔차 행렬 (n, h)
    u0 = hyp[None, :, 0] - m[:, None, 0] * hyp[None, :, 2]
    u1 = hyp[None, :, 1] - m[:, None, 1] * hyp[None, :, 2]
    nu = np.hypot(u0, u1)
    cr = np.abs(d[:, None, 0] * u1 - d[:, None, 1] * u0)
    res = np.where(nu > 1e-12, cr / np.maximum(nu, 1e-12), 0.0)
    pref = res <= math.sin(tol)
    # 병합 군집화 (Jaccard 거리)
    P = pref.astype(np.float32)
    cnt = P.sum(axis=1)
    members = [[i] for i in range(n)]
    active = np.ones(n, dtype=bool)
    inter = P @ P.T
    while True:
        union = cnt[:, None] + cnt[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.where(union > 0, 1.0 - inter / union, 1.0)
        dist[~active, :] = 1.0
        dist[:, ~active] = 1.0
        np.fill_diagonal(dist, 1.0)
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        if not dist[i, j] < 1.0 - 1e-9:
            break
        if i > j:
            i, j = j, i
        P[i] = np.minimum(P[i], P[j])
        cnt[i] = P[i].sum()
        members[i].extend(members[j])
        active[j] = False
        inter[i, :] = P[i] @ P.T
        inter[:, i] = inter[i, :]
        inter[i, i] = cnt[i]
        inter[j, :] = 0
        inter[:, j] = 0
        cnt[j] = 0
        P[j] = 0
    groups = [members[i] for i in range(n) if active[i] and len(members[i]) >= opts.min_cluster]
    if not groups:
        return np.zeros((0, 3)), labels
    vps = []
    for gidx in groups:
        gi = np.asarray(gidx)
        common = pref[gi].all(axis=0)
        if not common.any():
            common = pref[gi].mean(axis=0) >= 0.5
        hc = np.nonzero(common)[0]
        if len(hc) == 0:
            hc = np.arange(len(hyp))
        score = (res[np.ix_(gi, hc)] ** 2).sum(axis=0)
        v0 = hyp[hc[int(np.argmin(score))]]
        vps.append(_refit_vp(Ssub[gi], v0))
    vps = np.asarray(vps)
    d_all, m_all, L_all = _seg_dir_mid(S)
    sin_tol = math.sin(tol)
    for _ in range(4):
        M = len(vps)
        if M == 0:
            break
        R = np.stack([np.abs(_sin_resid(vps[j], m_all, d_all)) for j in range(M)], axis=1)
        R[L_all <= 1e-9] = np.inf
        lab = np.argmin(R, axis=1)
        best = R[np.arange(N), lab]
        lab = np.where(best <= sin_tol, lab, -1)
        # 중복 군집 병합: 군집 b 의 구성원이 군집 a(더 큰) 의 VP 와 80% 이상 합의하면 a 로
        tot = np.array([L_all[lab == j].sum() for j in range(M)])
        ordr = np.argsort(-tot, kind="stable")
        remap = np.arange(M)
        for ai in range(M):
            a = ordr[ai]
            if remap[a] != a:
                continue
            for bi in range(ai + 1, M):
                b = ordr[bi]
                if remap[b] != b:
                    continue
                mb = lab == b
                if mb.sum() == 0:
                    continue
                if (R[mb, a] <= sin_tol).mean() >= 0.8:
                    remap[b] = a
        lab = np.where(lab >= 0, remap[np.maximum(lab, 0)], -1)
        new_vps = []
        keep_map = {}
        for j in range(M):
            mj = lab == j
            if mj.sum() < opts.min_cluster:
                continue
            keep_map[j] = len(new_vps)
            new_vps.append(_refit_vp(S[mj], vps[j]))
        if not new_vps:
            vps = np.zeros((0, 3))
            lab = np.full(N, -1, dtype=np.int64)
            break
        lab = np.array([keep_map.get(int(x), -1) if x >= 0 else -1 for x in lab], dtype=np.int64)
        vps = np.asarray(new_vps)
    if len(vps) == 0:
        return np.zeros((0, 3)), np.full(N, -1, dtype=np.int64)
    tot = np.array([L_all[lab == j].sum() for j in range(len(vps))])
    ordr = np.argsort(-tot, kind="stable")
    inv = np.empty_like(ordr)
    inv[ordr] = np.arange(len(ordr))
    labels = np.where(lab >= 0, inv[np.maximum(lab, 0)], -1).astype(np.int64)
    return vps[ordr].copy(), labels


# ---------------------------------------------------------------------------------------------------------------
# plumb-line 왜곡
# ---------------------------------------------------------------------------------------------------------------
def _usable_polylines(polylines):
    use = []
    for p in polylines:
        a = np.asarray(p)
        if a.dtype.kind not in "biuf":
            continue
        a = a.astype(np.float64)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or not np.isfinite(a).all():
            continue
        if float(np.hypot(*(a[-1] - a[0]))) < 8.0 and float(np.ptp(a, axis=0).max()) < 8.0:
            continue
        use.append(a)
    return use


def _plumb_residuals(X, idx, cnt, f, k1, k2):
    """정규화 왜곡 좌표 X (n,2) 를 undistort 해 선별 총최소제곱 직선 잔차(px) 를 낸다."""
    xu = _geom.undistort(X, k1, k2) * f
    bad = ~np.isfinite(xu).all(axis=1)
    if bad.any():
        xu = np.where(bad[:, None], 0.0, xu)
    mx = np.bincount(idx, xu[:, 0], minlength=len(cnt)) / cnt
    my = np.bincount(idx, xu[:, 1], minlength=len(cnt)) / cnt
    qx = xu[:, 0] - mx[idx]
    qy = xu[:, 1] - my[idx]
    sxx = np.bincount(idx, qx * qx, minlength=len(cnt)) / cnt
    syy = np.bincount(idx, qy * qy, minlength=len(cnt)) / cnt
    sxy = np.bincount(idx, qx * qy, minlength=len(cnt)) / cnt
    ang = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    nx = -np.sin(ang)
    ny = np.cos(ang)
    r = qx * nx[idx] + qy * ny[idx]
    if bad.any():
        r = np.where(bad, 1e3, r)
    return r


def _plumb(polylines, K, opts: GeomCamOpts):
    """(PlumbFit, rms0) — rms0 는 사용한 선들에서 k=0 일 때의 직선 잔차 RMS."""
    K = _f64_array(K, "K")
    if K.shape != (3, 3) or not np.isfinite(K).all() or not K[0, 0] > 0:
        raise ValueError("K: 유한한 (3,3), fx > 0")
    use = _usable_polylines(polylines)
    if len(use) < opts.min_plumb_lines:
        return PlumbFit(0.0, 0.0, float("nan"), len(use)), float("nan")
    f = float(K[0, 0])
    c = K[:2, 2]
    kb = float(opts.k_bound)
    ridge = np.array([2.0, 4.0])
    k = np.zeros(2)
    keep = np.ones(len(use), dtype=bool)

    def pack(sel):
        pls = [use[i] for i in range(len(use)) if sel[i]]
        X = np.concatenate([(p - c) / f for p in pls], axis=0)
        idx = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in enumerate(pls)])
        cnt = np.array([len(p) for p in pls], dtype=np.float64)
        return X, idx, cnt

    def chain_rms(X, idx, cnt, kk):
        r = _plumb_residuals(X, idx, cnt, f, float(kk[0]), float(kk[1]))
        return np.sqrt(np.bincount(idx, r * r, minlength=len(cnt)) / cnt), r

    def solve(X, idx, cnt, x0, loss):
        def fun(kk):
            r = _plumb_residuals(X, idx, cnt, f, float(kk[0]), float(kk[1]))
            return np.concatenate([r, ridge * kk])
        try:
            sol = optimize.least_squares(fun, x0, bounds=(-kb, kb), loss=loss, f_scale=1.0,
                                         x_scale=np.array([0.05, 0.05]), method="trf", max_nfev=200)
            return np.clip(sol.x, -kb, kb)
        except Exception:
            return np.asarray(x0, dtype=np.float64)

    if kb > 0:
        X, idx, cnt = pack(keep)
        k = solve(X, idx, cnt, np.zeros(2), "cauchy")
        per, _ = chain_rms(X, idx, cnt, k)
        med = float(np.median(per))
        keep = per <= max(3.0 * med, 0.5)
        if keep.sum() < opts.min_plumb_lines:
            keep = np.ones(len(use), dtype=bool)
        X, idx, cnt = pack(keep)
        k = solve(X, idx, cnt, k, "soft_l1")
    else:
        X, idx, cnt = pack(keep)
    n_lines = int(keep.sum())
    rms0 = float(np.sqrt(np.mean(_plumb_residuals(X, idx, cnt, f, 0.0, 0.0) ** 2)))
    rms = float(np.sqrt(np.mean(_plumb_residuals(X, idx, cnt, f, float(k[0]), float(k[1])) ** 2)))
    return PlumbFit(float(k[0]), float(k[1]), rms, n_lines), rms0


def fit_plumb_line(polylines, K, *, opts=None) -> PlumbFit:
    """직선이어야 할 폴리라인들로 방사 왜곡 k1·k2 를 푼다 (|k| <= k_bound, 접선 없음)."""
    opts = _opts(opts)
    fit, _ = _plumb(list(polylines), K, opts)
    return fit


# ---------------------------------------------------------------------------------------------------------------
# 맨해튼 프레임 선택 + 공동 LM
# ---------------------------------------------------------------------------------------------------------------
def _f_from_fov(fov_deg, W):
    return (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def _rays(vps, c, f):
    A = vps[:, :2] - np.asarray(c)[None, :] * vps[:, 2:3]
    r = np.concatenate([A, f * vps[:, 2:3]], axis=1)
    return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-300)


def _pair_dev_deg(vi, vj, c, f):
    ri = _rays(vi[None], c, f)[0]
    rj = _rays(vj[None], c, f)[0]
    return math.degrees(math.asin(min(1.0, abs(float(ri @ rj)))))


def _pair_f(vi, vj, c, f_prior, fmin, fmax):
    """직교 두 VP 의 f. 반환 (f, dev_deg) 또는 None."""
    fi, fj = _is_finite_vp(vi), _is_finite_vp(vj)
    if fi and fj:
        f = focal_from_vps(vi, vj, c)
        if f is None or not (fmin <= f <= fmax):
            return None
        return f, 0.0
    dev = _pair_dev_deg(vi, vj, c, f_prior)
    if dev <= 1.5:
        return f_prior, dev
    return None


def _triple_f(vs, c, f_prior, fmin, fmax):
    """세 VP 의 직교성을 가장 잘 만족하는 f 와 RMS 편차(도)."""
    grid = np.exp(np.linspace(math.log(fmin), math.log(fmax), 400))
    devs = np.zeros((len(grid), 3))
    for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        ra = vs[a]
        rb = vs[b]
        Aa = ra[:2] - c * ra[2]
        Ab = rb[:2] - c * rb[2]
        num = Aa @ Ab + grid ** 2 * ra[2] * rb[2]
        den = np.sqrt((Aa @ Aa + grid ** 2 * ra[2] ** 2) * (Ab @ Ab + grid ** 2 * rb[2] ** 2))
        devs[:, k] = np.degrees(np.arcsin(np.minimum(1.0, np.abs(num) / np.maximum(den, 1e-300))))
    rms = np.sqrt((devs ** 2).mean(axis=1)) + 1e-3 * np.abs(np.log(grid / f_prior))
    i = int(np.argmin(rms))
    return float(grid[i]), float(np.sqrt((devs[i] ** 2).mean()))


def _choose_frame(vps, weights, c, f_prior, W, up_prior):
    """직교 맨해튼 프레임 후보를 점수로 고른다. 반환 dict(clusters, f) 또는 None(단일 수직 VP 후보는 별도)."""
    M = len(vps)
    if M < 2:
        return None
    fmin = _f_from_fov(_FOV_RANGE[1], W)
    fmax = _f_from_fov(_FOV_RANGE[0], W)
    top = list(range(min(M, 8)))
    best = None
    best_score = -1.0
    wprior = lambda f: math.exp(-0.5 * (math.log(f / f_prior) / 0.7) ** 2)
    for a, b in itertools.combinations(top, 2):
        r = _pair_f(vps[a], vps[b], c, f_prior, fmin, fmax)
        if r is None:
            continue
        f, dev = r
        sc = (weights[a] + weights[b]) * wprior(f)
        if sc > best_score * (1 + 1e-9):
            best_score = sc
            best = {"clusters": [a, b], "f": f, "dev": dev}
    for tr in itertools.combinations(top, 3):
        vs = [vps[i] for i in tr]
        f, dev = _triple_f(vs, np.asarray(c, dtype=np.float64), f_prior, fmin, fmax)
        if dev > _FRAME_DEV_TOL_DEG:
            continue
        sc = sum(weights[i] for i in tr) * wprior(f) / (1.0 + (dev / _FRAME_DEV_TAU_DEG) ** 2)
        if sc > best_score * (1 + 1e-9):
            best_score = sc
            best = {"clusters": list(tr), "f": f, "dev": dev}
    return best


def _orthonormal_from_rays(rays: np.ndarray) -> np.ndarray:
    """1~3 개의 대략 직교인 단위 광선(행)에서 우수 회전행렬(열 s = 슬롯 s 의 축)."""
    n = len(rays)
    if n == 3:
        M = rays.T.copy()
        if np.linalg.det(M) < 0:
            M[:, 2] = -M[:, 2]
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, 2] = -U[:, 2]
            R = U @ Vt
        return R
    q0 = rays[0] / np.linalg.norm(rays[0])
    if n == 2:
        q1 = rays[1] - (rays[1] @ q0) * q0
        q1 = q1 / np.linalg.norm(q1)
    else:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(q0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        q1 = tmp - (tmp @ q0) * q0
        q1 = q1 / np.linalg.norm(q1)
    q2 = np.cross(q0, q1)
    return np.stack([q0, q1, q2], axis=1)


def _rot(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3) + np.array([[0, -rv[2], rv[1]], [rv[2], 0, -rv[0]], [-rv[1], rv[0], 0]])
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _joint_lm(su, slots, R0, f0, c0, *, fit_f, fit_pp, f_prior, sigma_f, diag, W, H):
    """맨해튼 프레임 회전(+ log f, 주점) 공동 최소제곱. 반환 (R, f, c)."""
    d, m, L = _seg_dir_mid(su)
    wgt = np.sqrt(L / max(float(np.median(L)), 1e-9))
    wgt = np.clip(wgt, 0.3, 3.0)
    s0 = _SIGMA_ANG_DEG * _RAD
    cdef = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    pp_sigma = 0.25 * diag
    fmin = _f_from_fov(_FOV_RANGE[1], W)
    fmax = _f_from_fov(_FOV_RANGE[0], W)

    def unpack(x):
        R = _rot(x[:3]) @ R0
        k = 3
        f = f0
        c = c0
        if fit_f:
            f = math.exp(x[k])
            k += 1
        if fit_pp:
            c = np.array([x[k], x[k + 1]])
        return R, f, c

    def fun(x):
        R, f, c = unpack(x)
        K = np.array([[f, 0.0, c[0]], [0.0, f, c[1]], [0.0, 0.0, 1.0]])
        v = K @ R[:, slots]
        s = _sin_resid(v, m, d)
        out = [s / s0 * wgt]
        if fit_f:
            out.append(np.array([(math.log(f) - math.log(f_prior)) / sigma_f]))
        if fit_pp:
            out.append((c - cdef) / pp_sigma)
        return np.concatenate(out)

    x0 = [0.0, 0.0, 0.0]
    lo = [-np.inf] * 3
    hi = [np.inf] * 3
    xs = [0.01, 0.01, 0.01]
    if fit_f:
        x0.append(math.log(f0))
        lo.append(math.log(fmin))
        hi.append(math.log(fmax))
        xs.append(0.05)
    if fit_pp:
        x0 += [float(c0[0]), float(c0[1])]
        lo += [cdef[0] - 2.0 * diag, cdef[1] - 2.0 * diag]
        hi += [cdef[0] + 2.0 * diag, cdef[1] + 2.0 * diag]
        xs += [20.0, 20.0]
    x0 = np.asarray(x0, dtype=np.float64)
    x0 = np.clip(x0, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
    try:
        sol = optimize.least_squares(fun, x0, bounds=(np.asarray(lo), np.asarray(hi)), loss="soft_l1", f_scale=3.0,
                                     x_scale=np.asarray(xs), method="trf", max_nfev=200)
        R, f, c = unpack(sol.x)
        if not (np.isfinite(R).all() and math.isfinite(f) and np.isfinite(c).all()):
            raise FloatingPointError
    except Exception:
        R, f, c = R0, f0, c0
    return R, float(f), np.asarray(c, dtype=np.float64)


def _undistort_segments(S, c, f, k1, k2):
    if k1 == 0.0 and k2 == 0.0:
        return S.copy()
    out = np.empty_like(S)
    for a, b in ((0, 2), (2, 4)):
        xy = (S[:, a:b] - c) / f
        out[:, a:b] = _geom.undistort(xy, k1, k2) * f + c
    return out


def _canonical_frame(R, up_prior):
    """LM 슬롯 축 R(열) -> 맨해튼 R_mc(열0=수평 a, 열1=위, 열2=수평 b, 우수) 와 슬롯 -> 열 대응."""
    dots = np.abs(R.T @ up_prior)
    sv = int(np.argmax(dots))
    others = [s for s in range(3) if s != sv]
    up = R[:, sv].copy()
    if up @ up_prior < 0:
        up = -up
    # 수평 두 축: 카메라 x 성분이 더 큰 쪽이 열 0
    h0, h1 = others
    if abs(R[0, h1]) > abs(R[0, h0]) + 1e-12:
        h0, h1 = h1, h0
    x = R[:, h0].copy()
    if x[0] < 0:
        x = -x
    z = np.cross(x, up)
    nz = np.linalg.norm(z)
    z = z / nz if nz > 0 else R[:, h1]
    Rm = np.stack([x, up, z], axis=1)
    slot_to_col = {sv: 1, h0: 0, h1: 2}
    return Rm, slot_to_col


def _solve(segments, W, H, *, fov_prior_deg, horizon_prior, principal_prior, k1, k2, f_undist, plumb_rms, opts):
    S_in = segments
    N = len(S_in)
    Lall = _lengths(S_in)
    valid = Lall > 1e-9
    diag = math.hypot(W, H)
    cdef = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    c_prior = cdef if principal_prior is None else np.asarray(principal_prior, dtype=np.float64)
    if fov_prior_deg is not None:
        f_prior = _f_from_fov(fov_prior_deg, W)
        sigma_f = opts.prior_sigma_log_f
        prior_src = "prior"
    else:
        f_prior = _f_from_fov(DEFAULT_FOV_X_DEG, W)
        sigma_f = 3.0 * opts.prior_sigma_log_f
        prior_src = "default"
    up_prior = up_from_roll_pitch(*horizon_prior) if horizon_prior is not None else np.array([0.0, -1.0, 0.0])
    f_u = f_undist if f_undist is not None else f_prior
    warnings = set()

    idx_valid = np.nonzero(valid)[0]
    S = S_in[idx_valid]
    Su = _undistort_segments(S, c_prior, f_u, k1, k2) if len(S) else S
    if len(S) < 12:
        warnings.add("few_segments")

    vps = np.zeros((0, 3))
    lab_valid = np.full(len(S), -1, dtype=np.int64)
    if len(S) >= opts.min_cluster:
        vps, lab_valid = jlinkage(Su, opts=opts)
    M = len(vps)
    Lu = _lengths(Su) if len(S) else np.zeros(0)
    weights = np.array([Lu[lab_valid == j].sum() for j in range(M)])

    tol = opts.inlier_deg
    frame = _choose_frame(vps, weights, c_prior, f_prior, W, up_prior) if M >= 2 else None

    R_final = None
    f_final = f_prior
    c_final = c_prior.copy()
    f_source = prior_src
    pp_source = "prior" if principal_prior is not None else "center"
    axis_cluster = {}          # 슬롯 -> 군집 인덱스

    if frame is not None:
        cl = frame["clusters"]
        rays = _rays(vps[cl], c_prior, frame["f"])
        R0 = _orthonormal_from_rays(rays)
        slots_of_cluster = {j: s for s, j in enumerate(cl)}
        n_ax = len(cl)
        free_ok = (n_ax == 3 and all(
            _is_finite_vp(vps[j]) and float(np.hypot(*(vps[j][:2] / vps[j][2] - cdef))) <= opts.pp_free_max_dist * diag
            for j in cl))
        f_source = "vp3" if n_ax == 3 else "vp2"
        R, f, c = R0, frame["f"], c_prior.copy()
        for rnd in range(2):
            # 슬롯 배정: 군집 구성원 + 현재 축에 inlier_deg 안인 다른 군집/미배정 선분
            Kc = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
            axv = (Kc @ R).T                              # 슬롯별 VP (3,3)
            dd, mm, _ = _seg_dir_mid(Su)
            Rm = np.stack([np.abs(_sin_resid(axv[s], mm, dd)) for s in range(3)], axis=1)
            best_s = np.argmin(Rm, axis=1)
            best_r = Rm[np.arange(len(Su)), best_s]
            slot = np.full(len(Su), -1, dtype=np.int64)
            for j, s in slots_of_cluster.items():
                slot[lab_valid == j] = s
            if rnd > 0:
                inl = best_r <= math.sin(tol * _RAD)
                slot = np.where(inl, best_s, -1)
            else:
                # 군집 밖 선분 중 축에 합의하는 것을 추가
                extra = (slot < 0) & (lab_valid >= 0) & (best_r <= math.sin(tol * _RAD))
                # 고아 군집 구성원은 축에 합의하는 경우에만
                slot = np.where(extra, best_s, slot)
            msk = slot >= 0
            if msk.sum() < 3:
                break
            R, f, c = _joint_lm(Su[msk], slot[msk], R, f, c, fit_f=True, fit_pp=False, f_prior=f_prior,
                                sigma_f=sigma_f, diag=diag, W=W, H=H)
        # 주점 자유 여부
        if free_ok:
            Kc = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
            axv = (Kc @ R).T
            dd, mm, _ = _seg_dir_mid(Su)
            Rm = np.stack([np.abs(_sin_resid(axv[s], mm, dd)) for s in range(3)], axis=1)
            best_s = np.argmin(Rm, axis=1)
            best_r = Rm[np.arange(len(Su)), best_s]
            inl = best_r <= math.sin(tol * _RAD)
            if inl.sum() >= 6:
                Rp, fp, cp = _joint_lm(Su[inl], best_s[inl], R, f, cdef.copy(), fit_f=True, fit_pp=True,
                                       f_prior=f_prior, sigma_f=sigma_f, diag=diag, W=W, H=H)
                if float(np.hypot(*(cp - cdef))) <= opts.pp_reject_frac * diag:
                    R, f, c = Rp, fp, cp
                    pp_source = "vp3"
                else:
                    warnings.add("pp_rejected")
        R_final, f_final, c_final = R, f, c
        # 슬롯별 군집
        axis_cluster = {s: j for j, s in slots_of_cluster.items()}
    else:
        # 단일 수직 VP 후보
        single = None
        if M >= 1:
            Kp = np.array([[f_prior, 0, c_prior[0]], [0, f_prior, c_prior[1]], [0, 0, 1.0]])
            for j in range(M):
                ray = np.linalg.solve(Kp, vps[j])
                ray = ray / np.linalg.norm(ray)
                cosang = abs(float(ray @ up_prior))
                if cosang >= math.cos(math.radians(45.0)):
                    single = j
                    break
        if single is not None:
            ray = np.linalg.solve(np.array([[f_prior, 0, c_prior[0]], [0, f_prior, c_prior[1]], [0, 0, 1.0]]), vps[single])
            ray = ray / np.linalg.norm(ray)
            R0 = _orthonormal_from_rays(ray[None, :])
            msk = lab_valid == single
            if msk.sum() >= 3:
                R, f, c = _joint_lm(Su[msk], np.zeros(int(msk.sum()), dtype=np.int64), R0, f_prior, c_prior.copy(),
                                    fit_f=False, fit_pp=False, f_prior=f_prior, sigma_f=sigma_f, diag=diag, W=W, H=H)
            else:
                R = R0
            R_final = R
            axis_cluster = {0: single}
            f_final = f_prior
            c_final = c_prior.copy()

    K = np.array([[f_final, 0.0, c_final[0]], [0.0, f_final, c_final[1]], [0.0, 0.0, 1.0]], dtype=np.float64)

    # ----- 프레임 정리 -----
    R_mc = None
    vp_rows: List[np.ndarray] = []
    vp_axis_rows: List[int] = []
    row_of_cluster = {}
    n_man = 0
    horizon_source = "default" if horizon_prior is None else "prior"
    if horizon_prior is not None:
        up_cam = up_from_roll_pitch(*horizon_prior)
    else:
        up_cam = up_from_roll_pitch(0.0, 0.0)

    if R_final is not None and frame is not None:
        Rm, slot_to_col = _canonical_frame(R_final, up_prior)
        R_mc = Rm
        up_cam = Rm[:, 1].copy()
        horizon_source = "vp"
        axes_vp = (K @ Rm)           # 열 = 축 VP
        # 군집 -> 축 대응, 나머지 군집은 광선이 축과 inlier_deg 안이면 중복 병합, 아니면 고아
        cluster_axis = {}
        for s, j in axis_cluster.items():
            cluster_axis[j] = slot_to_col[s]
        rays_all = _rays(vps, c_final, f_final) if M else np.zeros((0, 3))
        for j in range(M):
            if j in cluster_axis:
                continue
            ang = np.degrees(np.arccos(np.clip(np.abs(Rm.T @ rays_all[j]), 0.0, 1.0)))
            a = int(np.argmin(ang))
            if ang[a] <= tol:
                if a in cluster_axis.values():
                    cluster_axis[j] = ("dup", a)
                else:
                    cluster_axis[j] = a
        # 축 -> 주 군집
        primary = {}
        for j, a in cluster_axis.items():
            if not isinstance(a, tuple):
                primary.setdefault(a, j)
        axis_vp = {a: _unit_vp(axes_vp[:, a]) for a in range(3)}
        # 행 구성
        for j in range(M):
            if j in cluster_axis and not isinstance(cluster_axis[j], tuple) and primary.get(cluster_axis[j]) == j:
                a = cluster_axis[j]
                row_of_cluster[j] = len(vp_rows)
                vp_rows.append(axis_vp[a])
                vp_axis_rows.append(a)
            elif j in cluster_axis and isinstance(cluster_axis[j], tuple):
                continue
            elif j in cluster_axis:
                continue
            else:
                row_of_cluster[j] = len(vp_rows)
                vp_rows.append(vps[j])
                vp_axis_rows.append(-1)
        n_man = len(primary)
    elif R_final is not None:
        # 단일 수직 VP
        ray = R_final[:, 0].copy()
        if ray @ up_prior < 0:
            ray = -ray
        up_cam = ray
        horizon_source = "vp"
        n_man = 1
        for j in range(M):
            row_of_cluster[j] = len(vp_rows)
            if j in axis_cluster.values():
                vp_rows.append(_unit_vp(K @ ray))
                vp_axis_rows.append(1)
            else:
                vp_rows.append(vps[j])
                vp_axis_rows.append(-1)
        # 수직 VP 와 비슷한 다른 군집은 고아가 아니라 무시하지 않는다(그대로 -1)
    else:
        for j in range(M):
            row_of_cluster[j] = len(vp_rows)
            vp_rows.append(vps[j])
            vp_axis_rows.append(-1)

    vps_out = np.asarray(vp_rows, dtype=np.float64).reshape(-1, 3)
    vp_axis_out = np.asarray(vp_axis_rows, dtype=np.int64)

    # ----- 선분 최종 배정 (모든 행에 대해 최소 잔차, inlier_deg 안) -----
    labels_full = np.full(N, -1, dtype=np.int64)
    Su_full = np.zeros((N, 4))
    if len(S):
        Su_full[idx_valid] = Su
    if len(vps_out) and len(S):
        dd, mm, _ = _seg_dir_mid(Su)
        Rm_ = np.stack([np.abs(_sin_resid(vps_out[r], mm, dd)) for r in range(len(vps_out))], axis=1)
        lab = np.argmin(Rm_, axis=1)
        bst = Rm_[np.arange(len(Su)), lab]
        lab = np.where(bst <= math.sin(tol * _RAD), lab, -1)
        # 고아 행은 min_cluster 미만이면 제거
        keep_rows = np.ones(len(vps_out), dtype=bool)
        for r in range(len(vps_out)):
            if vp_axis_out[r] < 0 and (lab == r).sum() < opts.min_cluster:
                keep_rows[r] = False
        if not keep_rows.all():
            newidx = -np.ones(len(vps_out), dtype=np.int64)
            newidx[keep_rows] = np.arange(int(keep_rows.sum()))
            lab = np.where(lab >= 0, newidx[np.maximum(lab, 0)], -1)
            vps_out = vps_out[keep_rows]
            vp_axis_out = vp_axis_out[keep_rows]
        labels_full[idx_valid] = lab
        # 행을 배정 총 길이 내림차순으로 정렬(안정)
        if len(vps_out) > 1:
            tot = np.array([Lu[lab == r].sum() for r in range(len(vps_out))])
            ordr = np.argsort(-tot, kind="stable")
            inv = np.empty_like(ordr)
            inv[ordr] = np.arange(len(ordr))
            vps_out = vps_out[ordr]
            vp_axis_out = vp_axis_out[ordr]
            lf = labels_full[idx_valid]
            labels_full[idx_valid] = np.where(lf >= 0, inv[np.maximum(lf, 0)], -1)

    # ----- r_vp, orphan_frac -----
    r_vp = float("nan")
    orphan_frac = 0.0
    lab_v = labels_full[idx_valid] if len(S) else np.zeros(0, dtype=np.int64)
    if n_man >= 1 and len(vps_out) and len(S):
        rr = np.zeros(len(S))
        is_axis = np.zeros(len(S), dtype=bool)
        for r in range(len(vps_out)):
            if vp_axis_out[r] >= 0:
                mk = lab_v == r
                if mk.any():
                    rr[mk] = segment_vp_residual_deg(Su[mk], vps_out[r])
                    is_axis |= mk
        if is_axis.any():
            w = Lu[is_axis]
            r_vp = float(math.sqrt(float((w * rr[is_axis] ** 2).sum() / w.sum())))
        clustered = lab_v >= 0
        if clustered.any():
            orph = clustered & np.array([(vp_axis_out[x] < 0) if x >= 0 else False for x in lab_v])
            orphan_frac = float(Lu[orph].sum() / Lu[clustered].sum())

    if n_man == 0:
        warnings.add("no_vp")
    elif n_man == 1:
        warnings.add("single_vp")
    if orphan_frac > opts.orphan_warn:
        warnings.add("vp_inconsistent")

    reliable = bool(n_man >= 2 and orphan_frac <= opts.orphan_warn
                    and math.isfinite(r_vp) and r_vp <= opts.r_vp_warn_deg)
    fov = _geom.fov_x_deg_from_K(K, W)
    roll, pitch = roll_pitch_from_up(up_cam)
    return GeomCamera(
        W=int(W), H=int(H), K=K, fov_x_deg=float(fov), k1=float(k1), k2=float(k2),
        roll_deg=float(roll), pitch_deg=float(pitch), up_cam=np.asarray(up_cam, dtype=np.float64),
        R_mc=None if R_mc is None or n_man < 2 else R_mc, vps=vps_out, vp_axis=vp_axis_out,
        segments=S_in.copy(), labels=labels_full, n_manhattan_vps=int(n_man),
        f_source=f_source if n_man >= 2 else prior_src, pp_source=pp_source,
        horizon_source=horizon_source, r_vp_deg=float(r_vp), orphan_frac=float(orphan_frac),
        plumb_rms_px=float(plumb_rms), reliable=reliable, warnings=tuple(sorted(warnings)),
    ), f_final


# ---------------------------------------------------------------------------------------------------------------
# 공개 추정기
# ---------------------------------------------------------------------------------------------------------------
def _check_priors(W, H, fov_prior_deg, horizon_prior, principal_prior):
    for n, v in (("W", W), ("H", H)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{n}={v!r}: 1 이상의 정수여야 한다")
    fov = None
    if fov_prior_deg is not None:
        try:
            fov = float(fov_prior_deg)
        except Exception as e:
            raise TypeError("fov_prior_deg: 숫자") from e
        if not (math.isfinite(fov) and 1.0 <= fov < 180.0):
            raise ValueError(f"fov_prior_deg={fov}: [1,180) 밖")
    hp = None
    if horizon_prior is not None:
        a = _f64_array(horizon_prior, "horizon_prior").reshape(-1)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("horizon_prior: 유한한 (roll_deg, pitch_deg)")
        hp = (float(a[0]), float(a[1]))
    pp = None
    if principal_prior is not None:
        a = _f64_array(principal_prior, "principal_prior").reshape(-1)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("principal_prior: 유한한 (cx, cy)")
        pp = a.copy()
    return fov, hp, pp


def estimate_from_segments(segments, W, H, *, fov_prior_deg=None, horizon_prior=None, principal_prior=None,
                           k1=0.0, k2=0.0, opts=None) -> GeomCamera:
    """왜곡된 픽셀 좌표 선분에서 K·수평선·맨해튼 프레임을 푼다(끝점은 (k1,k2) 로 왜곡 제거)."""
    opts = _opts(opts)
    S = _check_segments(segments)
    fov, hp, pp = _check_priors(W, H, fov_prior_deg, horizon_prior, principal_prior)
    k1 = float(k1)
    k2 = float(k2)
    if not (math.isfinite(k1) and math.isfinite(k2)):
        raise ValueError("k1,k2: 유한해야 한다")
    g, _ = _solve(S, int(W), int(H), fov_prior_deg=fov, horizon_prior=hp, principal_prior=pp, k1=k1, k2=k2,
                  f_undist=None, plumb_rms=float("nan"), opts=opts)
    return g


def _to_gray(image) -> np.ndarray:
    a = np.asarray(image)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"image: 수치 배열이어야 한다 (dtype {a.dtype})")
    if a.ndim == 2:
        pass
    elif a.ndim == 3 and a.shape[2] == 3:
        pass
    else:
        raise ValueError(f"image: (H,W) 또는 (H,W,3) 이어야 한다 (got {a.shape})")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError("image: 비유한 값")
    if a.shape[0] < _MIN_SIDE or a.shape[1] < _MIN_SIDE:
        raise GeomCamError(f"영상이 너무 작다: {a.shape[:2]} (짧은 변 >= {_MIN_SIDE})")
    if a.dtype == np.uint8:
        v = a.astype(np.float64) / 255.0
        if v.ndim == 3:
            v = v @ np.array([0.2126, 0.7152, 0.0722])
        return v
    v = np.clip(a.astype(np.float64), 0.0, 1.0)
    if v.ndim == 3:
        v = v @ np.array([0.2126, 0.7152, 0.0722])
    # 장면 선형 휘도 -> 표시 부호화 휘도(sRGB OETF)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def _downsample(g: np.ndarray, max_side: int):
    H, W = g.shape
    long = max(H, W)
    if long <= max_side:
        return g, 1.0, 1.0
    s = max_side / long
    h2 = max(_MIN_SIDE, int(round(H * s)))
    w2 = max(_MIN_SIDE, int(round(W * s)))
    sy = h2 / H
    sx = w2 / W
    pre = ndimage.gaussian_filter(g, sigma=(0.5 / sy, 0.5 / sx), mode="nearest")
    ys = (np.arange(h2) + 0.5) / sy - 0.5
    xs = (np.arange(w2) + 0.5) / sx - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    small = ndimage.map_coordinates(pre, [yy, xx], order=1, mode="nearest")
    return small, sx, sy


def estimate_geom_camera(image, *, fov_prior_deg=None, horizon_prior=None, principal_prior=None, opts=None) -> GeomCamera:
    """영상 한 장에서 K·왜곡·수평선·맨해튼 프레임. 계약서 RD-02 규칙 1 의 순서를 따른다."""
    opts = _opts(opts)
    g = _to_gray(image)
    H, W = g.shape
    fov, hp, pp = _check_priors(W, H, fov_prior_deg, horizon_prior, principal_prior)
    small, sx, sy = _downsample(g, opts.max_side)
    S, chains = _detect(small, opts)
    if len(S):
        S = S.copy()
        S[:, [0, 2]] = (S[:, [0, 2]] + 0.5) / sx - 0.5
        S[:, [1, 3]] = (S[:, [1, 3]] + 0.5) / sy - 0.5
    chains = [np.stack([(c[:, 0] + 0.5) / sx - 0.5, (c[:, 1] + 0.5) / sy - 0.5], axis=1) for c in chains]

    common = dict(fov_prior_deg=fov, horizon_prior=hp, principal_prior=pp, opts=opts)
    g0, _ = _solve(S, W, H, k1=0.0, k2=0.0, f_undist=None, plumb_rms=float("nan"), **common)
    result = g0
    if not opts.fit_distortion or len(chains) < opts.min_plumb_lines:
        return result
    cur_K = g0.K
    f_used = float(g0.K[0, 0])
    extra = set()
    for _ in range(3):
        fit, rms0 = _plumb(chains, cur_K, opts)
        if not (math.isfinite(fit.rms_px) and math.isfinite(rms0)):
            break
        if not (rms0 - fit.rms_px >= max(0.2, 0.2 * rms0)):
            if _ == 0:
                result = dataclasses.replace(g0, plumb_rms_px=float(rms0))
            break
        g1, _f = _solve(S, W, H, k1=fit.k1, k2=fit.k2, f_undist=f_used, plumb_rms=fit.rms_px, **common)
        result = g1
        extra = set()
        kb = opts.k_bound
        if abs(fit.k1) >= kb - 1e-6 or abs(fit.k2) >= kb - 1e-6:
            extra.add("distortion_clipped")
        f_new = float(g1.K[0, 0])
        if abs(f_new / f_used - 1.0) <= 0.03:
            break
        f_used = f_new
        cur_K = g1.K
    if extra:
        result = dataclasses.replace(result, warnings=tuple(sorted(set(result.warnings) | extra)))
    return result
