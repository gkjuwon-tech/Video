"""LNS-03: 셔터 모션블러 (Longuet-Higgins split, 롤링 셔터, 180° 기본).

렌더된 셔터 중간 자세 영상 한 장과 카메라 궤적으로 노출 시간 동안의 블러를 적분한다 (PARADIGM-v2 §7.5).
  * 회전 블러는 호모그래피 H(τ) = K R_τᵀ R_mid K⁻¹ 의 **정확한 역호모그래피** 적분 (gather: q = H(τ)⁻¹ p).
  * 평행이동 블러는 깊이로 화소별 1차 근사: q = H⁻¹ p − d_k(p), d_k = x_full(τ_k) − x_rot(τ_k).
  * 롤링 셔터는 행마다 시각 오프셋 readout·(y/(H-1) − 1/2).

핵심 항등식: 중간 카메라 좌표 X_c = z·K⁻¹p̃ 인 점의 τ 자세 투영은 K(R_τᵀ R_mid K⁻¹ p̃ + R_τᵀ(c_mid − c_τ)/z) 이므로
``g = H(τ) p̃ + e(τ)/z`` (e = K R_τᵀ (c_mid − c_τ)) 의 동차 좌표다. 1/z = 0 이면 무한원점(순수 회전)이 된다.

규약: OpenCV 카메라, c2w 4x4, 픽셀 중심이 정수 좌표 (u,v)=(열,행), 깊이는 광축 z(m), rgb 는 선형 (H,W,3).
결정론(무작위 없음), 입력 불변, import 부작용·전역 RNG·경고 없음. torch 는 쓰지 않는다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from buc.core import color as _color
from buc.core import contracts as _C
from buc.core import geom as _geom
from buc.core import pyramid as _pyr

__all__ = [
    "ShutterOpts", "ShutterResult", "exposure_time", "exposure_window", "sample_times", "row_time_offsets",
    "pose_at", "rotation_homography", "apply_shutter", "apply_shutter_frames",
]

_FAST_PX = 1e-6      # 이 이하의 최대 블러는 비트 복사 경로
_W_EPS = 1e-9        # 동차 좌표 w 가 이보다 작으면 카메라 뒤(또는 무한 먼 곳)로 본다


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _finite(name, x) -> float:
    if not _is_real(x) or not math.isfinite(float(x)):
        raise ValueError(f"{name}={x!r}: 유한한 실수여야 한다")
    return float(x)


def _is_bool(x) -> bool:
    return isinstance(x, (bool, np.bool_))


# ---------------------------------------------------------------------------------------------------------------
# 옵션·결과
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ShutterOpts:
    samples: int = 0
    max_samples: int = 64
    readout_s: float = 0.0
    centered: bool = True
    translation: bool = True
    border: str = "edge"

    def __post_init__(self):
        if not _is_int(self.samples) or self.samples < 0:
            raise ValueError(f"samples={self.samples!r}: 0 이상의 정수여야 한다")
        if not _is_int(self.max_samples) or self.max_samples < 1:
            raise ValueError(f"max_samples={self.max_samples!r}: 1 이상의 정수여야 한다")
        ro = _finite("readout_s", self.readout_s)
        if ro < 0.0:
            raise ValueError(f"readout_s={ro}: 0 이상이어야 한다")
        for n in ("centered", "translation"):
            if not _is_bool(getattr(self, n)):
                raise ValueError(f"{n}={getattr(self, n)!r}: bool 이어야 한다")
        if self.border not in ("edge", "zero"):
            raise ValueError(f"border={self.border!r}: 'edge' 또는 'zero' 여야 한다")


@dataclass(frozen=True, kw_only=True)
class ShutterResult:
    rgb: np.ndarray        # (H,W,3) float32 선형
    blur_px: np.ndarray    # (H,W,2) float32 = x(t_close(y)) - x(t_open(y))
    samples: int           # 실제 쓴 시간 샘플 수


# ---------------------------------------------------------------------------------------------------------------
# 노출 시각
# ---------------------------------------------------------------------------------------------------------------
def exposure_time(shutter_deg, fps) -> float:
    d, f = _finite("shutter_deg", shutter_deg), _finite("fps", fps)
    if not (0.0 < d <= 360.0):
        raise ValueError(f"shutter_deg={d}: (0, 360] 이어야 한다")
    if not f > 0.0:
        raise ValueError(f"fps={f}: 양수여야 한다")
    return d / (360.0 * f)


def exposure_window(t, fps, shutter_deg, *, centered=True) -> tuple:
    t = _finite("t", t)
    T = exposure_time(shutter_deg, fps)
    return (t - T / 2.0, t + T / 2.0) if centered else (t, t + T)


def sample_times(t0, t1, n) -> np.ndarray:
    """중점 층화: t0 + (k+0.5)(t1-t0)/n, k=0..n-1."""
    t0, t1 = _finite("t0", t0), _finite("t1", t1)
    if not _is_int(n) or n < 1:
        raise ValueError(f"n={n!r}: 1 이상의 정수여야 한다")
    n = int(n)
    return t0 + (np.arange(n, dtype=np.float64) + 0.5) * ((t1 - t0) / n)


def row_time_offsets(height, readout_s) -> np.ndarray:
    """(H,) readout·(y/(H-1) − 1/2). H=1 이면 [0]."""
    if not _is_int(height) or height < 1:
        raise ValueError(f"height={height!r}: 1 이상의 정수여야 한다")
    ro = _finite("readout_s", readout_s)
    if ro < 0.0:
        raise ValueError(f"readout_s={ro}: 0 이상이어야 한다")
    h = int(height)
    if h == 1:
        return np.zeros(1, np.float64)
    return ro * (np.arange(h, dtype=np.float64) / (h - 1) - 0.5)


# ---------------------------------------------------------------------------------------------------------------
# 궤적 보간
# ---------------------------------------------------------------------------------------------------------------
def _traj(c2w, t):
    c = np.asarray(c2w)
    tt = np.asarray(t)
    if c.dtype.kind not in "biuf" or tt.dtype.kind not in "biuf":
        raise ValueError("c2w, t: 수치 배열이어야 한다")
    c = c.astype(np.float64, copy=False)
    tt = tt.astype(np.float64, copy=False)
    if c.ndim != 3 or c.shape[1:] != (4, 4) or c.shape[0] < 1:
        raise ValueError(f"c2w shape {c.shape}: (N,4,4) 여야 한다")
    if tt.ndim != 1 or tt.shape[0] != c.shape[0]:
        raise ValueError(f"t shape {tt.shape} 가 c2w N={c.shape[0]} 와 맞지 않는다")
    if not (np.isfinite(c).all() and np.isfinite(tt).all()):
        raise ValueError("c2w, t: 유한하지 않은 값")
    if tt.shape[0] > 1 and not (np.diff(tt) > 0.0).all():
        raise ValueError("t: 엄격 증가여야 한다")
    return c, tt


def _rodrigues_batch(rv):
    """(M,3) 회전벡터 -> (M,3,3). geom.rodrigues 와 같은 식(벡터화)."""
    th = np.linalg.norm(rv, axis=-1)
    safe = np.where(th > 0.0, th, 1.0)
    k = rv / safe[:, None]
    z = np.zeros_like(th)
    Kx = np.stack([np.stack([z, -k[:, 2], k[:, 1]], -1),
                   np.stack([k[:, 2], z, -k[:, 0]], -1),
                   np.stack([-k[:, 1], k[:, 0], z], -1)], axis=-2)
    om = 2.0 * np.sin(th / 2.0) ** 2
    R = np.eye(3)[None] + np.sin(th)[:, None, None] * Kx + om[:, None, None] * (Kx @ Kx)
    return np.where((th > 0.0)[:, None, None], R, np.eye(3)[None])


def _pose_at(c, t, tau):
    """검증된 (N,4,4), (N,) 와 1D tau (M,) -> (M,4,4)."""
    M = tau.shape[0]
    N = c.shape[0]
    if N == 1:
        return np.repeat(c[:1], M, axis=0)
    j = np.clip(np.searchsorted(t, tau, side="right") - 1, 0, N - 2)
    s = (tau - t[j]) / (t[j + 1] - t[j])
    uj = np.unique(j)
    rv_u = np.stack([_geom.rotvec_from_matrix(c[i, :3, :3].T @ c[i + 1, :3, :3]) for i in uj])
    rv = rv_u[np.searchsorted(uj, j)]
    R = c[j, :3, :3] @ _rodrigues_batch(s[:, None] * rv)
    out = np.zeros((M, 4, 4), np.float64)
    out[:, :3, :3] = R
    out[:, :3, 3] = (1.0 - s)[:, None] * c[j, :3, 3] + s[:, None] * c[j + 1, :3, 3]
    out[:, 3, 3] = 1.0
    return out


def pose_at(c2w, t, tau) -> np.ndarray:
    """스칼라 tau -> (4,4), 1D tau -> (M,4,4). 회전 측지·위치 선형, 구간 밖은 끝 구간 외삽."""
    c, tt = _traj(c2w, t)
    ta = np.asarray(tau)
    if ta.dtype.kind not in "biuf" or ta.ndim > 1:
        raise ValueError("tau: 스칼라 또는 1D 수치 배열이어야 한다")
    ta = ta.astype(np.float64)
    if not np.isfinite(ta).all():
        raise ValueError("tau: 유한하지 않은 값")
    if ta.ndim == 0:
        return _pose_at(c, tt, ta.reshape(1))[0]
    return _pose_at(c, tt, ta)


def rotation_homography(K, c2w_mid, c2w_tau) -> np.ndarray:
    """K · R_τᵀ · R_mid · K⁻¹ : 중간 자세 화소 -> τ 자세 화소. c2w_tau (M,4,4) 이면 (M,3,3)."""
    K = np.asarray(K, np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K shape {K.shape}: (3,3) 이어야 한다")
    Rm = np.asarray(c2w_mid, np.float64)
    Rt = np.asarray(c2w_tau, np.float64)
    if Rm.shape != (4, 4):
        raise ValueError(f"c2w_mid shape {Rm.shape}: (4,4) 여야 한다")
    if Rt.ndim not in (2, 3) or Rt.shape[-2:] != (4, 4):
        raise ValueError(f"c2w_tau shape {Rt.shape}: (4,4) 또는 (M,4,4) 여야 한다")
    Rm, Rt = Rm[:3, :3], Rt[..., :3, :3]
    return K @ np.swapaxes(Rt, -1, -2) @ Rm @ np.linalg.inv(K)


# ---------------------------------------------------------------------------------------------------------------
# 적분
# ---------------------------------------------------------------------------------------------------------------
def _check_inputs(rgb, depth, K, c2w, t, t_mid):
    img = np.asarray(rgb)
    if img.dtype.kind != "f" or img.ndim != 3 or img.shape[2] != 3 or min(img.shape) < 1:
        raise ValueError(f"rgb: 부동소수 (H,W,3) 여야 한다 (dtype {img.dtype}, shape {img.shape})")
    if not np.isfinite(img).all():
        raise ValueError("rgb: 유한하지 않은 값")
    h, w = img.shape[:2]
    d = None
    if depth is not None:
        d = np.asarray(depth)
        if d.dtype.kind not in "biuf" or d.shape != (h, w):
            raise ValueError(f"depth: 수치 (H,W)=({h},{w}) 여야 한다 (shape {d.shape})")
        d = d.astype(np.float64)
        if np.isnan(d).any() or not (d > 0.0).all():
            raise ValueError("depth: 양수(또는 +inf)여야 한다")
    Kk = np.asarray(K)
    if Kk.dtype.kind not in "biuf" or Kk.shape != (3, 3) or not np.isfinite(Kk).all():
        raise ValueError("K: 유한한 (3,3) 이어야 한다")
    Kk = Kk.astype(np.float64)
    if not (Kk[0, 0] > 0.0 and Kk[1, 1] > 0.0) or abs(np.linalg.det(Kk)) < 1e-12:
        raise ValueError("K: 초점거리가 양수이고 가역이어야 한다")
    c, tt = _traj(c2w, t)
    R = c[:, :3, :3]
    if (np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max() > 1e-4 or (np.linalg.det(R) <= 0.0).any()
            or np.abs(c[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-6):
        raise ValueError("c2w: 회전 4x4 포즈여야 한다")
    tm = _finite("t_mid", t_mid)
    return img, d, Kk, c, tt, tm


def _apply_rows(P, Hm):
    """P (H,W,3) 동차 좌표에 행별(또는 공통) 3x3 행렬 Hm (R,3,3) 적용 -> (H,W,3)."""
    return np.matmul(P, np.swapaxes(Hm, -1, -2))


def _dehom(g):
    w = g[..., 2]
    ok = w > _W_EPS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xy = g[..., :2] / np.where(ok, w, 1.0)[..., None]
    return xy, ok


def _pose_rows(c, tt, taus):
    """taus (...,) -> (...,4,4)."""
    shp = taus.shape
    return _pose_at(c, tt, taus.reshape(-1)).reshape(shp + (4, 4))


def _blur_vec(P, K, Kinv, Rm, cm, c, tt, t0, t1, inv_z):
    """행별 창 끝점 (t0, t1) 의 투영 차 -> (H,W,2) float64."""
    poses = _pose_rows(c, tt, np.stack([t0, t1]))             # (2,R,4,4)
    Rt = np.swapaxes(poses[..., :3, :3], -1, -2)               # (2,R,3,3)
    Hm = K @ Rt @ Rm @ Kinv                                    # (2,R,3,3)
    xs = []
    oks = []
    for i in range(2):
        g = _apply_rows(P, Hm[i])
        if inv_z is not None:
            e = (K @ Rt[i] @ (cm - poses[i][..., :3, 3])[..., None])[..., 0]    # (R,3)
            g = g + inv_z[..., None] * e[:, None, :]
        xy, ok = _dehom(g)
        xs.append(xy)
        oks.append(ok)
    out = xs[1] - xs[0]
    valid = oks[0] & oks[1] & np.isfinite(out).all(-1)
    return np.where(valid[..., None], out, 0.0)


def apply_shutter(rgb, depth=None, *, K, c2w, t, t_mid, shutter_deg, fps, opts=None) -> ShutterResult:
    if opts is None:
        opts = ShutterOpts()
    elif not isinstance(opts, ShutterOpts):
        raise ValueError("opts: ShutterOpts 여야 한다")
    img, dep, K, c, tt, tm = _check_inputs(rgb, depth, K, c2w, t, t_mid)
    T = exposure_time(shutter_deg, fps)
    h, w = img.shape[:2]
    Kinv = np.linalg.inv(K)

    # 행별 창. 글로벌 셔터면 모든 행이 같은 창이라 R=1 로 계산한다.
    ro = float(opts.readout_s)
    rolling = ro > 0.0 and h > 1
    base = tm - T / 2.0 if opts.centered else tm
    t0 = base + (row_time_offsets(h, ro) if rolling else np.zeros(1))      # (R,)
    t1 = t0 + T

    Pm = _pose_at(c, tt, np.array([tm]))[0]
    Rm, cm = Pm[:3, :3], Pm[:3, 3]

    uv = _geom.pixel_grid(h, w)
    P = np.concatenate([uv, np.ones((h, w, 1))], axis=-1)                  # (H,W,3) 동차 화소
    inv_z = None
    if dep is not None and opts.translation:
        fin = np.isfinite(dep)
        if fin.any():
            inv_z = np.where(fin, 1.0 / np.where(fin, dep, 1.0), 0.0)
            if not np.any(inv_z):
                inv_z = None

    blur = _blur_vec(P, K, Kinv, Rm, cm, c, tt, t0, t1, inv_z)             # (H,W,2)
    blur32 = blur.astype(np.float32)
    mag = np.linalg.norm(blur, axis=-1)
    mx = float(mag.max()) if mag.size else 0.0
    if mx <= _FAST_PX:
        return ShutterResult(rgb=np.array(img, dtype=np.float32, copy=True), blur_px=blur32, samples=1)

    if opts.samples > 0:
        n = int(opts.samples)
    else:
        n = int(min(max(math.ceil(2.0 * mx) + 1, 1), opts.max_samples))

    # 시간 샘플 (n,R) 의 포즈. 중점 층화.
    taus = t0[None, :] + ((np.arange(n, dtype=np.float64) + 0.5) * (T / n))[:, None]
    poses = _pose_rows(c, tt, taus)                                        # (n,R,4,4)
    Rt = np.swapaxes(poses[..., :3, :3], -1, -2)                           # (n,R,3,3)
    Hf = K @ Rt @ Rm @ Kinv                                                # (n,R,3,3) 중간 -> τ
    Hi = np.linalg.inv(Hf)                                                 # (n,R,3,3) τ -> 중간 (정확한 역)
    if inv_z is not None:
        E = (K[None, None] @ Rt @ (cm - poses[..., :3, 3])[..., None])[..., 0]   # (n,R,3)
        trans_mask = inv_z > 0.0

    src = img.astype(np.float64)
    acc = np.zeros((h, w, 3), np.float64)
    for k in range(n):
        qh = _apply_rows(P, Hi[k])
        q, ok = _dehom(qh)
        q = np.where(ok[..., None], q, uv)
        if inv_z is not None:
            g_rot = _apply_rows(P, Hf[k])
            x_rot, ok_r = _dehom(g_rot)
            x_full, ok_f = _dehom(g_rot + inv_z[..., None] * E[k][:, None, :])
            d = np.where((trans_mask & ok_r & ok_f)[..., None], x_full - x_rot, 0.0)
            q = q - d
        q = np.where(np.isfinite(q), q, uv)
        acc += _pyr.sample_bilinear(src, q, border=opts.border)
    out = (acc / n).astype(np.float32)
    return ShutterResult(rgb=out, blur_px=blur32, samples=n)


def apply_shutter_frames(frames, shot, opts=None):
    rgb = frames.rgb
    N = int(shot.t.shape[0])
    if rgb.ndim != 4 or rgb.shape[0] != N or rgb.shape[1] != shot.height or rgb.shape[2] != shot.width:
        raise ValueError(f"Frames rgb {rgb.shape} 가 shot (N={N}, H={shot.height}, W={shot.width}) 과 맞지 않는다")
    depth = frames.aov.get("depth")
    if depth is not None and tuple(depth.shape) != tuple(rgb.shape[:3]):
        raise ValueError(f"depth {depth.shape} 가 rgb {rgb.shape} 과 맞지 않는다")
    th = shot.theta
    H, W = rgb.shape[1:3]
    out_rgb = np.empty((N, H, W, 3), np.float32)
    out_blur = np.empty((N, H, W, 2), np.float32)
    for i in range(N):
        lin = _color.srgb_to_linear(rgb[i])
        r = apply_shutter(lin, None if depth is None else depth[i], K=shot.K[i], c2w=shot.c2w, t=shot.t,
                          t_mid=float(shot.t[i]), shutter_deg=th.shutter_deg, fps=th.fps, opts=opts)
        out_rgb[i] = _color.linear_to_display(r.rgb)
        out_blur[i] = r.blur_px
    aov = dict(frames.aov)
    aov["blur_px"] = out_blur
    return _C.Frames(rgb=out_rgb, fps=frames.fps, aov=aov, shot_hash=frames.shot_hash,
                     world_hash=frames.world_hash)
