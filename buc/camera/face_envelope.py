"""CAM-04 얼굴 봉투 규칙: F0-F3 범주, alpha_sigma, 미관측 얼굴면 노출, 경로 판정.

학습 0, 순수 수식. numpy와 표준 라이브러리만 쓴다(torch 금지, import 부작용·전역 RNG 없음).
규약: 세계 오른손·미터, 포즈는 4x4 c2w float64, 카메라 위치 = c2w[:3, 3].
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GRADES", "CATEGORIES", "EPS_F", "W_F_M", "BACK_YAW_DEG", "H_F1_MAX", "H_F2_MAX",
    "FACE_CAP_HALF_DEG", "HEAD_RADIUS_M", "M_TOL", "ANGLE_TOL_DEG", "UNOBS_TOL",
    "YAW_TABLE_DEG", "PITCH_TABLE_DEG", "M_MAX", "UNOBS_MAX",
    "FaceSpec", "FaceLimits", "FaceVerdict", "FaceReport",
    "alpha_sigma_deg", "face_category", "limits", "face_angles", "unobserved_exposure",
    "evaluate_path", "evaluate_faces", "face_viol_sigma", "orbit_budget_deg",
]

_INF = math.inf

GRADES = ("safe", "extended", "full")
CATEGORIES = ("F0", "F1", "F2", "F3")
EPS_F = 0.03
W_F_M = 0.15
BACK_YAW_DEG = 120
H_F1_MAX = 0.03
H_F2_MAX = 0.15
FACE_CAP_HALF_DEG = 45
HEAD_RADIUS_M = 0.10
M_TOL = 0.02
ANGLE_TOL_DEG = 1e-6
UNOBS_TOL = 1e-9

YAW_TABLE_DEG = {"F0": _INF, "F1": _INF, "F2": 30, "F3": 15}
PITCH_TABLE_DEG = {"F0": _INF, "F1": _INF, "F2": 15, "F3": 10}
M_MAX = {"F0": _INF, "F1": 1.5, "F2": 1.0, "F3": 1.0}
UNOBS_MAX = {"F0": 0.05, "F1": 0.03, "F2": 0.05, "F3": 0.05}

_CAT_GRADES = {
    "F0": GRADES,
    "F1": GRADES,
    "F2": ("safe", "extended"),
    "F3": ("safe",),
}
_VIOLATION_ORDER = ("yaw", "pitch", "push", "unobserved", "grade")
_N_CAP_POINTS = 8192


# ---------------------------------------------------------------- 도우미
def _wrap_deg(x):
    """각도를 [-180, 180) 로 감는다(스칼라·배열)."""
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _vec3(x: Any, name: str) -> np.ndarray:
    try:
        a = np.array(x, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 숫자 (3,) 벡터여야 한다") from e
    if a.shape != (3,):
        raise ValueError(f"{name}: shape (3,) 이어야 한다, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: 유한해야 한다")
    return a


def _scalar(x: Any, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 숫자여야 한다") from e
    return v


def _check_margin(margin_deg: float) -> float:
    m = _scalar(margin_deg, "margin_deg")
    if not (m >= 0.0) or math.isinf(m):
        raise ValueError("margin_deg 는 유한한 0 이상이어야 한다")
    return m


# ---------------------------------------------------------------- FaceSpec
@dataclass(frozen=True, kw_only=True, eq=False)
class FaceSpec:
    center: np.ndarray
    front: np.ndarray
    up: np.ndarray
    h_f: float
    sigma_z: float = 0.0
    head_radius_m: float = HEAD_RADIUS_M

    def __post_init__(self) -> None:
        center = _vec3(self.center, "center")
        front = _vec3(self.front, "front")
        up = _vec3(self.up, "up")
        fn = float(np.linalg.norm(front))
        if fn <= 0.0:
            raise ValueError("front 는 영벡터일 수 없다")
        front = front / fn
        un = float(np.linalg.norm(up))
        if un <= 0.0:
            raise ValueError("up 은 영벡터일 수 없다")
        up_o = up - front * float(front @ up)
        uon = float(np.linalg.norm(up_o))
        if uon / un < 1e-8:
            raise ValueError("up 이 front 와 평행하다")
        up_o = up_o / uon
        h_f = _scalar(self.h_f, "h_f")
        if not (0.0 < h_f <= 1.0):
            raise ValueError("h_f 는 (0, 1] 이어야 한다")
        sigma_z = _scalar(self.sigma_z, "sigma_z")
        if math.isnan(sigma_z) or sigma_z < 0.0:
            raise ValueError("sigma_z 는 0 이상(inf 허용)이어야 한다")
        hr = _scalar(self.head_radius_m, "head_radius_m")
        if not (hr > 0.0) or math.isinf(hr):
            raise ValueError("head_radius_m 은 유한한 양수여야 한다")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "front", front)
        object.__setattr__(self, "up", up_o)
        object.__setattr__(self, "h_f", h_f)
        object.__setattr__(self, "sigma_z", sigma_z)
        object.__setattr__(self, "head_radius_m", hr)

    @property
    def right(self) -> np.ndarray:
        return np.cross(self.up, self.front)

    def to_dict(self) -> dict:
        return {
            "center": [float(v) for v in self.center],
            "front": [float(v) for v in self.front],
            "up": [float(v) for v in self.up],
            "h_f": float(self.h_f),
            "sigma_z": float(self.sigma_z),
            "head_radius_m": float(self.head_radius_m),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "FaceSpec":
        if not isinstance(d, dict):
            raise ValueError("FaceSpec.from_dict 는 dict 를 받는다")
        allowed = {"center", "front", "up", "h_f", "sigma_z", "head_radius_m"}
        required = {"center", "front", "up", "h_f"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"모르는 키: {sorted(map(str, unknown))}")
        missing = required - set(d)
        if missing:
            raise ValueError(f"누락된 키: {sorted(missing)}")
        return cls(**{k: d[k] for k in allowed if k in d})


@dataclass(frozen=True)
class FaceLimits:
    category: str
    yaw_deg: float
    pitch_deg: float
    m_max: float
    unobs_max: float
    grades: tuple
    alpha_sigma_deg: float


@dataclass(frozen=True)
class FaceVerdict:
    ok: bool
    category: str
    yaw_obs_deg: float
    h_f_max: float
    m_peak: float
    yaw_dev_deg: float
    pitch_dev_deg: float
    unobs_max: float
    limits: FaceLimits
    violations: tuple
    allowed_grades: tuple


@dataclass(frozen=True)
class FaceReport:
    ok: bool
    verdicts: tuple
    violations: tuple
    allowed_grades: tuple


# ---------------------------------------------------------------- 규칙 표
def alpha_sigma_deg(sigma_z_m: float, *, eps_f: float = EPS_F, w_f_m: float = W_F_M) -> float:
    """alpha_sigma = degrees(asin(min(1, eps_f * w_f / sigma)))."""
    eps_f = _scalar(eps_f, "eps_f")
    w_f_m = _scalar(w_f_m, "w_f_m")
    if not (eps_f > 0.0) or not (w_f_m > 0.0) or math.isinf(eps_f) or math.isinf(w_f_m):
        raise ValueError("eps_f, w_f_m 은 유한한 양수여야 한다")
    s = _scalar(sigma_z_m, "sigma_z_m")
    if math.isnan(s) or s < 0.0:
        raise ValueError("sigma_z_m 은 0 이상이어야 한다")
    if s == 0.0:
        return 90.0
    if math.isinf(s):
        return 0.0
    return math.degrees(math.asin(min(1.0, eps_f * w_f_m / s)))


def face_category(h_f: float, observed_yaw_deg: float = 0.0, *, has_face: bool = True) -> str:
    if not has_face:
        return "F0"
    h = _scalar(h_f, "h_f")
    if not (0.0 < h <= 1.0):
        raise ValueError("h_f 는 (0, 1] 이어야 한다")
    y = _scalar(observed_yaw_deg, "observed_yaw_deg")
    if math.isnan(y):
        raise ValueError("observed_yaw_deg 가 NaN")
    if abs(float(_wrap_deg(y))) > BACK_YAW_DEG:
        return "F0"
    if h <= H_F1_MAX:
        return "F1"
    if h <= H_F2_MAX:
        return "F2"
    return "F3"


def limits(category: str, sigma_z_m: float = 0.0, *, margin_deg: float = 0.0) -> FaceLimits:
    if not isinstance(category, str) or category not in CATEGORIES:
        raise ValueError(f"알 수 없는 범주: {category!r}")
    margin = _check_margin(margin_deg)
    alpha = alpha_sigma_deg(sigma_z_m)
    if category in ("F2", "F3"):
        yaw = max(0.0, min(float(YAW_TABLE_DEG[category]), alpha) - margin)
    else:
        yaw = _INF
    return FaceLimits(
        category=category,
        yaw_deg=float(yaw),
        pitch_deg=float(PITCH_TABLE_DEG[category]),
        m_max=float(M_MAX[category]),
        unobs_max=float(UNOBS_MAX[category]),
        grades=_CAT_GRADES[category],
        alpha_sigma_deg=float(alpha),
    )


# ---------------------------------------------------------------- 얼굴 각
def _positions(x: Any, name: str) -> tuple:
    try:
        a = np.array(x, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 숫자 배열이어야 한다") from e
    if a.ndim == 1 and a.shape == (3,):
        single = True
        a = a[None, :]
    elif a.ndim == 2 and a.shape[1] == 3:
        single = False
    else:
        raise ValueError(f"{name}: shape (3,) 또는 (N,3) 이어야 한다, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: 유한해야 한다")
    return a, single


def _angles(face: FaceSpec, pts: np.ndarray) -> tuple:
    v = pts - face.center
    x = v @ face.right
    y = v @ face.up
    z = v @ face.front
    dist = np.linalg.norm(v, axis=1)
    if np.any(dist <= face.head_radius_m):
        raise ValueError("카메라가 머리 구 안에 있다")
    yaw = np.degrees(np.arctan2(x, z))
    pitch = np.degrees(np.arctan2(y, np.hypot(x, z)))
    return yaw, pitch, dist


def face_angles(face: FaceSpec, cam_pos: Any) -> tuple:
    """(yaw, pitch, dist): 얼굴 좌표계에서 본 카메라 위치(도, 도, m)."""
    pts, single = _positions(cam_pos, "cam_pos")
    yaw, pitch, dist = _angles(face, pts)
    if single:
        return float(yaw[0]), float(pitch[0]), float(dist[0])
    return yaw, pitch, dist


# ---------------------------------------------------------------- 미관측 노출
@functools.lru_cache(maxsize=1)
def _cap_points() -> np.ndarray:
    """얼굴 기저 (right, up, front) 좌표의 캡 위 결정론 준균일 표본(등면적 피보나치 나선)."""
    n = _N_CAP_POINTS
    i = np.arange(n, dtype=np.float64)
    cmin = math.cos(math.radians(FACE_CAP_HALF_DEG))
    ct = 1.0 - (1.0 - cmin) * (i + 0.5) / n
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    ph = i * golden
    pts = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    pts.setflags(write=False)
    return pts


def _local(face: FaceSpec, p: np.ndarray) -> tuple:
    v = p - face.center
    return float(v @ face.right), float(v @ face.up), float(v @ face.front)


def _seen(face: FaceSpec, p: np.ndarray) -> np.ndarray:
    n = _cap_points()
    x, y, z = _local(face, p)
    return (n[:, 0] * x + n[:, 1] * y + n[:, 2] * z) > face.head_radius_m


def _check_outside(face: FaceSpec, p: np.ndarray) -> None:
    if float(np.linalg.norm(p - face.center)) <= face.head_radius_m:
        raise ValueError("카메라가 머리 구 안에 있다")


def _exposure_one(face: FaceSpec, seen0: np.ndarray, p: np.ndarray) -> float:
    _check_outside(face, p)
    new = _seen(face, p) & ~seen0
    return float(np.count_nonzero(new)) / float(new.size)


def unobserved_exposure(face: FaceSpec, cam0_pos: Any, cam_pos: Any):
    """캡 중 cam_pos 에서는 보이고 cam0_pos 에서는 안 보이는 면적 비율."""
    p0, s0 = _positions(cam0_pos, "cam0_pos")
    if not s0:
        raise ValueError("cam0_pos 는 shape (3,) 이어야 한다")
    pts, single = _positions(cam_pos, "cam_pos")
    _check_outside(face, p0[0])
    seen0 = _seen(face, p0[0])
    out = np.array([_exposure_one(face, seen0, pts[k]) for k in range(pts.shape[0])], dtype=np.float64)
    if single:
        return float(out[0])
    return out


# ---------------------------------------------------------------- 경로 판정
def _check_pose(c, name: str) -> np.ndarray:
    try:
        a = np.array(c, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 숫자 배열이어야 한다") from e
    if a.shape != (4, 4):
        raise ValueError(f"{name}: shape (4,4) 이어야 한다")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: 유한해야 한다")
    if not np.allclose(a[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0):
        raise ValueError(f"{name}: 마지막 행이 [0,0,0,1] 이어야 한다")
    return a


def _check_grade(grade) -> None:
    if grade is not None and (not isinstance(grade, str) or grade not in GRADES):
        raise ValueError(f"알 수 없는 grade: {grade!r}")


def evaluate_path(face: FaceSpec, c2w: Any, *, c0: Any = None, margin_deg: float = 0.0,
                  grade: str | None = None) -> FaceVerdict:
    try:
        poses = np.array(c2w, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("c2w: 숫자 배열이어야 한다") from e
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 1:
        raise ValueError("c2w: shape (N,4,4), N>=1 이어야 한다")
    if not np.all(np.isfinite(poses)):
        raise ValueError("c2w: 유한해야 한다")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0):
        raise ValueError("c2w: 마지막 행이 [0,0,0,1] 이어야 한다")
    pose0 = poses[0] if c0 is None else _check_pose(c0, "c0")
    margin = _check_margin(margin_deg)
    _check_grade(grade)

    pos = poses[:, :3, 3]
    pos0 = pose0[:3, 3]
    yaw0a, pitch0a, d0a = _angles(face, pos0[None, :])
    yaw0, pitch0, d0 = float(yaw0a[0]), float(pitch0a[0]), float(d0a[0])
    yaw, pitch, dist = _angles(face, pos)

    dyaw = np.abs(_wrap_deg(yaw - yaw0))
    dpitch = np.abs(pitch - pitch0)
    m = d0 / dist
    h_k = face.h_f * m
    yaw_dev = float(np.max(dyaw))
    pitch_dev = float(np.max(dpitch))
    m_peak = float(np.max(m))
    h_f_max = float(np.max(h_k))

    category = face_category(min(h_f_max, 1.0), yaw0)
    lim = limits(category, face.sigma_z, margin_deg=margin)

    seen0 = _seen(face, pos0)
    unobs = max(_exposure_one(face, seen0, pos[k]) for k in range(pos.shape[0]))

    viol = []
    if yaw_dev > lim.yaw_deg + ANGLE_TOL_DEG:
        viol.append("yaw")
    if pitch_dev > lim.pitch_deg + ANGLE_TOL_DEG:
        viol.append("pitch")
    if m_peak > lim.m_max + M_TOL:
        viol.append("push")
    if unobs > lim.unobs_max + UNOBS_TOL:
        viol.append("unobserved")
    if grade is not None and grade not in lim.grades:
        viol.append("grade")
    return FaceVerdict(
        ok=not viol,
        category=category,
        yaw_obs_deg=yaw0,
        h_f_max=h_f_max,
        m_peak=m_peak,
        yaw_dev_deg=yaw_dev,
        pitch_dev_deg=pitch_dev,
        unobs_max=float(unobs),
        limits=lim,
        violations=tuple(viol),
        allowed_grades=lim.grades,
    )


def evaluate_faces(faces: Any, c2w: Any, *, c0: Any = None, margin_deg: float = 0.0,
                   grade: str | None = None) -> FaceReport:
    specs = []
    for f in faces:
        if isinstance(f, FaceSpec):
            specs.append(f)
        elif isinstance(f, dict):
            specs.append(FaceSpec.from_dict(f))
        else:
            raise TypeError(f"faces 원소는 FaceSpec 또는 dict 여야 한다, got {type(f).__name__}")
    verdicts = tuple(evaluate_path(s, c2w, c0=c0, margin_deg=margin_deg, grade=grade) for s in specs)
    present = {v for vd in verdicts for v in vd.violations}
    violations = tuple(v for v in _VIOLATION_ORDER if v in present)
    allowed = tuple(g for g in GRADES if all(g in vd.allowed_grades for vd in verdicts))
    return FaceReport(ok=not violations, verdicts=verdicts, violations=violations, allowed_grades=allowed)


def face_viol_sigma(faces: Any, c2w: Any, **kw) -> bool:
    return not evaluate_faces(faces, c2w, **kw).ok


# ---------------------------------------------------------------- 오빗 예산
def orbit_budget_deg(face: FaceSpec, cam0_pos: Any, direction: float, *, margin_deg: float = 0.0,
                     step_deg: float = 0.25) -> float:
    """얼굴 up 축 둘레 수평 오빗에서 yaw 한계·노출 한도를 처음 넘기 직전의 각도(<=180)."""
    if direction not in (1, -1):
        raise ValueError("direction 은 +1 또는 -1 이어야 한다")
    step = _scalar(step_deg, "step_deg")
    if not (step > 0.0) or math.isinf(step):
        raise ValueError("step_deg 는 유한한 양수여야 한다")
    margin = _check_margin(margin_deg)
    p0, single = _positions(cam0_pos, "cam0_pos")
    if not single:
        raise ValueError("cam0_pos 는 shape (3,) 이어야 한다")
    p0 = p0[0]
    yaw0, pitch0, d0 = face_angles(face, p0)
    category = face_category(face.h_f, yaw0)
    lim = limits(category, face.sigma_z, margin_deg=margin)
    seen0 = _seen(face, p0)
    r, u, f = face.right, face.up, face.front
    cp = math.cos(math.radians(pitch0))
    sp = math.sin(math.radians(pitch0))

    n_steps = int(math.floor(180.0 / step + 1e-9))
    last_ok = 0.0
    for k in range(1, n_steps + 1):
        a = k * step
        if a > lim.yaw_deg + ANGLE_TOL_DEG:
            break
        y = math.radians(yaw0 + direction * a)
        p = face.center + d0 * (cp * (math.sin(y) * r + math.cos(y) * f) + sp * u)
        if _exposure_one(face, seen0, p) > lim.unobs_max + UNOBS_TOL:
            break
        last_ok = a
    return float(min(last_ok, 180.0))
