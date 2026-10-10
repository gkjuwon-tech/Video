"""CAM-01 테스트 공용 도우미. 구현 내부가 아니라 계약서(CAM-01.md)의 공식만 쓴다."""
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

RIG_NAMES = ("tripod_fluid", "dolly", "slider", "crane_jib", "techno_crane", "steadicam", "gimbal_3axis",
             "shoulder_handheld", "cine_drone", "fpv", "cablecam", "car_mount", "motion_control", "underwater")
TIER1 = ("slider", "dolly", "crane_jib")
PARAM_KEYS = ("mass", "damping", "quad_drag", "stiffness", "coulomb", "w2", "kp", "ki", "kd", "ff_gain",
              "u_max", "ou_sigma", "ou_tau")
DT = 1e-3
V_EPS = 1e-3


def rigs():
    from buc.camera import rigs as m
    return m


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


def ref_local(name, q, axis, radius):
    """계약서 FK 표: 홈 카메라 좌표계 안의 국소 변환 L(q)."""
    q = np.asarray(q, dtype=np.float64)
    if name in ("slider", "dolly"):
        return T4(None, q[0] * axis)
    if name == "cablecam":
        return T4(Rz(q[1]), q[0] * axis)
    if name == "car_mount":
        return T4(None, q[0] * axis + np.array([0.0, q[1], 0.0]))
    if name in ("crane_jib", "techno_crane"):
        r = radius + (q[2] if name == "techno_crane" else 0.0)
        rot = Ry(q[0]) @ Rx(q[1])
        return T4(rot, np.array([0.0, 0.0, radius]) + rot @ np.array([0.0, 0.0, -r]))
    if name == "tripod_fluid":
        return T4(Ry(q[0]) @ Rx(q[1]))
    if name == "gimbal_3axis":
        return T4(Ry(q[0]) @ Rx(q[1]) @ Rz(q[2]))
    if name == "steadicam":
        return T4(Ry(q[2]) @ Rz(q[3]), [q[0], 0.0, q[1]])
    if name in ("shoulder_handheld", "fpv", "motion_control"):
        return T4(Ry(q[3]) @ Rx(q[4]) @ Rz(q[5]), q[:3])
    if name in ("cine_drone", "underwater"):
        return T4(Ry(q[3]), q[:3])
    raise AssertionError(name)


def ref_fk(rig, q):
    return rig.home_c2w @ ref_local(rig.name, q, np.asarray(rig.geom["axis"], dtype=np.float64),
                                    float(rig.geom["radius"]))


def odd_home():
    """단위행렬이 아닌 홈 포즈(회전 + 평행이동)."""
    return T4(Ry(0.5) @ Rx(-0.2), [1.0, 2.0, 3.0])


def is_rot(R, atol):
    R = np.asarray(R)
    return (np.abs(R @ R.T - np.eye(3)).max() < atol) and abs(np.linalg.det(R) - 1.0) < atol


def mass_damper_exact(t, m, c, force, q0=0.0, v0=0.0):
    """m q'' = F - c q' 의 해석해 (q, v)."""
    tau = m / c
    e = np.exp(-t / tau)
    v = force / c + (v0 - force / c) * e
    q = q0 + force / c * t + (v0 - force / c) * tau * (1.0 - e)
    return q, v
