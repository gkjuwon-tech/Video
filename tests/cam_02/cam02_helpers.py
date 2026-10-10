"""CAM-02 테스트 공용 도우미. 구현 내부가 아니라 계약서(CAM-02.md)의 공식만 쓴다."""
import math

import numpy as np

DEFAULT_FRAMES = 193          # 8 s * 24 fps + 1 (종점 포함)


def mm():
    from buc.camera import moves
    return moves


def s_mj(tau):
    """min-jerk 정규화 프로파일 10t^3 - 15t^4 + 6t^5 (CAM-01 과 같은 식을 독립적으로 다시 쓴 것)."""
    t = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    return 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5


def s_rt(tau):
    """왕복 프로파일: 앞 절반 s(2t), 뒤 절반 s(2(1-t))."""
    t = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    return s_mj(np.where(t <= 0.5, 2.0 * t, 2.0 * (1.0 - t)))


def taus(n):
    return np.arange(n) / (n - 1)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def T4(R=None, t=None):
    out = np.eye(4)
    if R is not None:
        out[:3, :3] = R
    if t is not None:
        out[:3, 3] = t
    return out


def odd_home():
    """단위행렬이 아닌 C_0 (회전 + 평행이동)."""
    return T4(Ry(0.5) @ Rx(-0.2) @ Rz(0.1), [1.0, 2.0, 3.0])


def is_rot(R, atol=1e-9):
    R = np.asarray(R)
    return (np.abs(R @ R.T - np.eye(3)).max() < atol) and abs(np.linalg.det(R) - 1.0) < atol


def local_poses(c2w, c0):
    """C_0 좌표계 안의 포즈 열 L_k = C_0^-1 C_k."""
    return np.linalg.inv(c0)[None] @ np.asarray(c2w)


def orbit_local(alpha, z):
    """피벗 (0,0,z) 둘레 방위각 alpha 오빗의 C_0 국소 변환: T(P) R_y(a) T(-P)."""
    R = Ry(alpha)
    return T4(R, np.array([0.0, 0.0, z]) + R @ np.array([0.0, 0.0, -z]))


def subject_in_camera(c2w, subject_w):
    """세계 점을 각 프레임 카메라 좌표(OpenCV)로: (N,3)."""
    c2w = np.asarray(c2w)
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    return np.einsum("nji,nj->ni", R, np.asarray(subject_w)[None, :] - t)


def angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cr = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cr, np.sum(a * b, axis=-1)))


def obs_from_c2w(c2w, K, W, H):
    from buc.core.contracts import Obs
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    return Obs(K=K, R=R, t=t, width=W, height=H)
