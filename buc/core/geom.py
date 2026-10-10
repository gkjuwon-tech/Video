"""FND-06 기하 유틸: K / 회전 / 포즈 / 왜곡 / 투영 / 평면 호모그래피.

규약은 FND-02(``buc.core.contracts``)와 같다: 세계 오른손·미터, 카메라 OpenCV(x 오른쪽, y 아래, z 앞), 픽셀 중심이
정수 좌표(영상 영역 ``[-0.5, W-0.5) x [-0.5, H-0.5)``, 주점 ``((W-1)/2, (H-1)/2)``), 포즈 4x4 는 열벡터 규약.
순수 함수, 입력 불변, 결과는 새 float64 배열. numpy / scipy 만 쓴다(torch 금지), import 부작용 없음.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

__all__ = [
    "fov_from_focal", "focal_from_fov", "intrinsics_from_fov", "intrinsics_from_focal", "fov_x_deg_from_K",
    "scale_intrinsics", "crop_intrinsics",
    "rodrigues", "rotvec_from_matrix", "is_rotation", "project_to_so3", "rotation_angle_deg",
    "make_pose", "invert_pose", "relative_pose", "interp_pose", "look_at",
    "distort", "undistort", "project", "unproject", "unproject_world", "pixel_grid", "visible_mask",
    "plane_homography", "apply_homography",
]


# ---------------------------------------------------------------------------------------------------------------
# 보조
# ---------------------------------------------------------------------------------------------------------------
def _f64(x, name="x"):
    a = np.asarray(x)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    return np.array(a, dtype=np.float64)


def _shape(a, shape, name):
    if a.shape != shape:
        raise ValueError(f"{name}: shape {a.shape} != {shape}")


def _scalar(x, name):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: 숫자여야 한다 ({x!r})")
    return float(x)


def _pos_finite(x, name):
    f = _scalar(x, name)
    if not (math.isfinite(f) and f > 0.0):
        raise ValueError(f"{name}={f}: 양의 유한값이어야 한다")
    return f


def _K(K, name="K"):
    K = _f64(K, name)
    _shape(K, (3, 3), name)
    return K


def _R(R, name="R"):
    R = _f64(R, name)
    _shape(R, (3, 3), name)
    return R


def _T4(T, name="T"):
    T = _f64(T, name)
    _shape(T, (4, 4), name)
    return T


# ---------------------------------------------------------------------------------------------------------------
# 내부행렬
# ---------------------------------------------------------------------------------------------------------------
def fov_from_focal(focal_mm, sensor_width_mm) -> float:
    """수평 화각(도) = 2 atan(s / 2f)."""
    f = _pos_finite(focal_mm, "focal_mm")
    s = _pos_finite(sensor_width_mm, "sensor_width_mm")
    return float(math.degrees(2.0 * math.atan(s / (2.0 * f))))


def focal_from_fov(fov_x_deg, sensor_width_mm) -> float:
    fov = _scalar(fov_x_deg, "fov_x_deg")
    s = _pos_finite(sensor_width_mm, "sensor_width_mm")
    if not (math.isfinite(fov) and 0.0 < fov < 180.0):
        raise ValueError(f"fov_x_deg={fov}: (0, 180) 밖")
    return float(s / (2.0 * math.tan(math.radians(fov) / 2.0)))


def _K_from_fx(fx, W, H):
    return np.array([[fx, 0.0, (W - 1) / 2.0], [0.0, fx, (H - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _size(W, H):
    for n, v in (("W", W), ("H", H)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{n}={v!r}: 1 이상의 정수여야 한다")
    return int(W), int(H)


def intrinsics_from_fov(fov_x_deg, W, H) -> np.ndarray:
    fov = _scalar(fov_x_deg, "fov_x_deg")
    if not (math.isfinite(fov) and 0.0 < fov < 180.0):
        raise ValueError(f"fov_x_deg={fov}: (0, 180) 밖")
    W, H = _size(W, H)
    fx = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
    return _K_from_fx(fx, W, H)


def intrinsics_from_focal(focal_mm, sensor_width_mm, W, H) -> np.ndarray:
    f = _pos_finite(focal_mm, "focal_mm")
    s = _pos_finite(sensor_width_mm, "sensor_width_mm")
    W, H = _size(W, H)
    return _K_from_fx(f / s * W, W, H)


def fov_x_deg_from_K(K, W) -> float:
    K = _K(K)
    if not K[0, 0] > 0:
        raise ValueError("fx > 0 이어야 한다")
    return float(math.degrees(2.0 * math.atan(W / (2.0 * K[0, 0]))))


def scale_intrinsics(K, sx, sy=None) -> np.ndarray:
    """영상을 (sx, sy)배 리샘플할 때의 K. 픽셀 중심이 정수이므로 u' = (u + 0.5) s - 0.5."""
    K = _K(K)
    sx = _pos_finite(sx, "sx")
    sy = sx if sy is None else _pos_finite(sy, "sy")
    out = K.copy()
    out[0, 0] = K[0, 0] * sx
    out[0, 1] = K[0, 1] * sx
    out[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    out[1, 1] = K[1, 1] * sy
    out[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return out


def crop_intrinsics(K, x0, y0) -> np.ndarray:
    K = _K(K)
    out = K.copy()
    out[0, 2] = K[0, 2] - _scalar(x0, "x0")
    out[1, 2] = K[1, 2] - _scalar(y0, "y0")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 회전
# ---------------------------------------------------------------------------------------------------------------
def rodrigues(rvec) -> np.ndarray:
    rv = _f64(rvec, "rvec")
    _shape(rv, (3,), "rvec")
    th = float(np.linalg.norm(rv))
    if th == 0.0:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    one_minus_cos = 2.0 * math.sin(th / 2.0) ** 2
    return np.eye(3) + math.sin(th) * Kx + one_minus_cos * (Kx @ Kx)


def rotvec_from_matrix(R) -> np.ndarray:
    """θ in [0, π]. 0 과 π 근처에서 쿼터니언 경유로 안정."""
    R = _R(R)
    if not np.isfinite(R).all():
        raise ValueError("R: 유한하지 않은 값")
    return np.asarray(_Rotation.from_matrix(R).as_rotvec(), dtype=np.float64)


def is_rotation(R, atol=1e-6) -> bool:
    try:
        R = _f64(R, "R")
    except TypeError:
        return False
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return False
    err = np.abs(R.T @ R - np.eye(3)).max()
    return bool(err <= atol and abs(np.linalg.det(R) - 1.0) <= atol)


def project_to_so3(M) -> np.ndarray:
    M = _R(M, "M")
    if not np.isfinite(M).all():
        raise ValueError("M: 유한하지 않은 값")
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    if d == 0.0:
        d = 1.0
    return (U * np.array([1.0, 1.0, d])) @ Vt


def rotation_angle_deg(Ra, Rb) -> float:
    """측지각(도). atan2(|sin θ|, cos θ) 형이라 작은 각도에서도 정밀하다."""
    Ra, Rb = _R(Ra, "Ra"), _R(Rb, "Rb")
    Rr = Ra.T @ Rb
    v = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) * 0.5
    c = (np.trace(Rr) - 1.0) * 0.5
    return float(math.degrees(math.atan2(float(np.linalg.norm(v)), float(c))))


# ---------------------------------------------------------------------------------------------------------------
# 포즈
# ---------------------------------------------------------------------------------------------------------------
def make_pose(R, t) -> np.ndarray:
    R = _R(R)
    t = _f64(t, "t")
    _shape(t, (3,), "t")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_pose(T) -> np.ndarray:
    T = _T4(T)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def relative_pose(T_w2c_a, T_w2c_b) -> np.ndarray:
    """a 카메라 좌표 -> b 카메라 좌표: T_b · T_a⁻¹."""
    return _T4(T_w2c_b, "T_w2c_b") @ invert_pose(_T4(T_w2c_a, "T_w2c_a"))


def interp_pose(T0, T1, s) -> np.ndarray:
    T0, T1 = _T4(T0, "T0"), _T4(T1, "T1")
    s = _scalar(s, "s")
    R0, R1 = T0[:3, :3], T1[:3, :3]
    rv = rotvec_from_matrix(R0.T @ R1)
    R = R0 @ rodrigues(s * rv)
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = (1.0 - s) * T0[:3, 3] + s * T1[:3, 3]
    return out


def look_at(eye, target, *, up) -> np.ndarray:
    """c2w 4x4. z = 시선, y = -(up 의 시선 수직 성분), x = y × z."""
    eye = _f64(eye, "eye")
    target = _f64(target, "target")
    up = _f64(up, "up")
    for n, a in (("eye", eye), ("target", target), ("up", up)):
        _shape(a, (3,), n)
        if not np.isfinite(a).all():
            raise ValueError(f"{n}: 유한하지 않은 값")
    fwd = target - eye
    nf = np.linalg.norm(fwd)
    if nf == 0.0:
        raise ValueError("eye == target")
    nu = np.linalg.norm(up)
    if nu == 0.0:
        raise ValueError("up 이 영벡터")
    z = fwd / nf
    u = up / nu
    perp = u - np.dot(u, z) * z
    npp = np.linalg.norm(perp)
    if npp < 1e-9:
        raise ValueError("시선이 up 과 평행하다")
    y = -perp / npp
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2] = x, y, z
    out[:3, 3] = eye
    return out


# ---------------------------------------------------------------------------------------------------------------
# 왜곡 / 투영
# ---------------------------------------------------------------------------------------------------------------
def _xy(xy, name):
    a = _f64(xy, name)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise ValueError(f"{name}: 마지막 축이 2 여야 한다 (shape {a.shape})")
    return a


def distort(xy, k1, k2) -> np.ndarray:
    xy = _xy(xy, "xy")
    r2 = (xy ** 2).sum(-1, keepdims=True)
    return xy * (1.0 + float(k1) * r2 + float(k2) * r2 ** 2)


def undistort(xy_d, k1, k2) -> np.ndarray:
    """정규화 좌표 왜곡 제거. 반지름에 대한 뉴턴법(수렴 1e-10 이상, 단조 영역에서)."""
    xy_d = _xy(xy_d, "xy_d")
    k1, k2 = float(k1), float(k2)
    if k1 == 0.0 and k2 == 0.0:
        return xy_d.copy()
    rd = np.sqrt((xy_d ** 2).sum(-1))
    r = rd.copy()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(100):
            r2 = r * r
            g = r * (1.0 + k1 * r2 + k2 * r2 * r2) - rd
            gp = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2
            dr = g / gp
            r = r - dr
            if not np.nanmax(np.abs(dr), initial=0.0) > 1e-15:
                break
        ratio = np.where(rd > 0.0, r / rd, 1.0)
    return xy_d * ratio[..., None]


def project(K, R, t, X, k1=0.0, k2=0.0):
    """세계점 (...,3) -> (uv (...,2), z (...)). z<=0 의 uv 는 무의미."""
    K, R = _K(K), _R(R)
    t = _f64(t, "t")
    _shape(t, (3,), "t")
    X = _f64(X, "X")
    if X.ndim < 1 or X.shape[-1] != 3:
        raise ValueError(f"X: 마지막 축이 3 이어야 한다 (shape {X.shape})")
    Xc = X @ R.T + t
    z = Xc[..., 2].copy()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = Xc[..., 0] / z
        y = Xc[..., 1] / z
        r2 = x * x + y * y
        f = 1.0 + float(k1) * r2 + float(k2) * r2 * r2
        xd, yd = x * f, y * f
        u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
        v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], axis=-1), z


def unproject(K, uv, depth, k1=0.0, k2=0.0) -> np.ndarray:
    """픽셀 + 광축 z-깊이 -> 카메라 좌표 (...,3)."""
    K = _K(K)
    uv = _xy(uv, "uv")
    depth = _f64(depth, "depth")
    try:
        depth = np.broadcast_to(depth, uv.shape[:-1])
    except ValueError as e:
        raise ValueError(f"depth shape {depth.shape} 가 uv {uv.shape} 와 맞지 않는다") from e
    yd = (uv[..., 1] - K[1, 2]) / K[1, 1]
    xd = (uv[..., 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]
    xy = undistort(np.stack([xd, yd], axis=-1), k1, k2)
    return np.stack([xy[..., 0] * depth, xy[..., 1] * depth, depth.copy()], axis=-1)


def unproject_world(K, R, t, uv, depth, k1=0.0, k2=0.0) -> np.ndarray:
    R = _R(R)
    t = _f64(t, "t")
    _shape(t, (3,), "t")
    Xc = unproject(K, uv, depth, k1, k2)
    return (Xc - t) @ R


def pixel_grid(H, W) -> np.ndarray:
    """(H, W, 2) float64, 순서 (u, v) = (열, 행)."""
    W, H = _size(W, H)
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([u, v], axis=-1).astype(np.float64)


def visible_mask(uv, z, W, H) -> np.ndarray:
    """z > 0 이고 영상 영역 [-0.5, W-0.5) x [-0.5, H-0.5) 안. NaN 은 False."""
    uv = _xy(uv, "uv")
    z = _f64(z, "z")
    u, v = uv[..., 0], uv[..., 1]
    with np.errstate(invalid="ignore"):
        return (z > 0.0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)


# ---------------------------------------------------------------------------------------------------------------
# 평면 호모그래피
# ---------------------------------------------------------------------------------------------------------------
def plane_homography(K_src, K_dst, R, t, n, d) -> np.ndarray:
    """평면 n·X = d (src 카메라 좌표), X_dst = R X + t. H = K_dst (R + t nᵀ/d) K_src⁻¹, H[2,2] = 1."""
    Ks, Kd, R = _K(K_src, "K_src"), _K(K_dst, "K_dst"), _R(R)
    t = _f64(t, "t")
    n = _f64(n, "n")
    _shape(t, (3,), "t")
    _shape(n, (3,), "n")
    d = _scalar(d, "d")
    if not (math.isfinite(d) and d != 0.0):
        raise ValueError("d 는 0 이 아닌 유한값이어야 한다")
    if not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1.0) > 1e-6:
        raise ValueError("n 은 단위벡터여야 한다")
    H = Kd @ (R + np.outer(t, n) / d) @ np.linalg.inv(Ks)
    if H[2, 2] == 0.0 or not np.isfinite(H[2, 2]):
        raise ValueError("H[2,2] 가 0 이라 정규화할 수 없다")
    return H / H[2, 2]


def apply_homography(H, uv) -> np.ndarray:
    H = _f64(H, "H")
    _shape(H, (3, 3), "H")
    uv = _xy(uv, "uv")
    p = uv @ H[:, :2].T + H[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[..., :2] / p[..., 2:3]
