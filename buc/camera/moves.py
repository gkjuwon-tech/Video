"""CAM-02 무브 레시피 · 촬영 메뉴 봉투 Pi_g (푸시인 · 슬라이더 · 오빗 · 랙).

고정 무브 4종의 경로 정의식(벤치 CT-I19 사전 등록 정의식의 원천), 그 경로를 CAM-01 리그로 실현하는 길,
촬영 메뉴 3등급(safe / extended / full)의 기하 봉투와 최소 등급 판정. 학습 0, 순수 수식(numpy).
규약: 세계 오른손·미터, 카메라 OpenCV, 포즈는 4x4 c2w float64. C_0 = 입력 사진의 카메라.
torch 금지, import 부작용·전역 RNG 사용 없음.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from buc.camera.rigs import Intent, make_rig, min_jerk_profile

__all__ = [
    "MOVE_NAMES", "GRADES", "BENCH_INTENTS", "BENCH_DURATION_S", "BENCH_FPS", "M_MAX_SEMANTIC", "M_MAX_MATERIAL",
    "BOUND_TOL", "MoveSpec", "MovePath", "PathMeasure", "Envelope", "recipe", "bench_spec", "frame_times",
    "ease_profile", "ideal_path", "rig_path", "to_shot", "measure", "envelope", "min_grade", "min_grade_path",
]

MOVE_NAMES = ("push_in", "slider", "orbit", "rack_focus")
GRADES = ("safe", "extended", "full")
BENCH_INTENTS = ("I1", "I2", "I3", "I4", "I5")
BENCH_DURATION_S = 8.0
BENCH_FPS = 24.0
M_MAX_SEMANTIC = 1.5
M_MAX_MATERIAL = 4.0
BOUND_TOL = 1e-9

_INF = math.inf
_GRID_TOL = 1e-6


# --------------------------------------------------------------------------------------------------------------
# 검증 도우미
# --------------------------------------------------------------------------------------------------------------
def _f(name, v):
    try:
        x = float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 숫자여야 한다") from e
    return x


def _pos_finite(name, v):
    x = _f(name, v)
    if not (math.isfinite(x) and x > 0.0):
        raise ValueError(f"{name} > 0 이고 유한해야 한다 (받은 값 {v!r})")
    return x


def _check_c0(c0):
    if c0 is None:
        return np.eye(4)
    try:
        a = np.array(c0, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("c0 는 4x4 숫자 배열이어야 한다") from e
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("c0 는 유한한 4x4 여야 한다")
    r = a[:3, :3]
    if (np.abs(a[3] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-9
            or np.abs(r @ r.T - np.eye(3)).max() > 1e-6 or abs(np.linalg.det(r) - 1.0) > 1e-6):
        raise ValueError("c0 는 SE(3) 여야 한다")
    a[3] = (0.0, 0.0, 0.0, 1.0)
    return a


# --------------------------------------------------------------------------------------------------------------
# MoveSpec
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MoveSpec:
    name: str
    z_s: float
    duration_s: float = 8.0
    fps: float = 24.0
    m: float = 1.0
    slide_zs: float = 0.0
    orbit_deg: float = 0.0
    round_trip: bool = False
    focus_from_m: Optional[float] = None
    focus_to_m: Optional[float] = None
    to_blur: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or self.name not in MOVE_NAMES:
            raise ValueError(f"모르는 무브 이름: {self.name!r}")
        sv = object.__setattr__
        sv(self, "z_s", _pos_finite("z_s", self.z_s))
        sv(self, "duration_s", _pos_finite("duration_s", self.duration_s))
        sv(self, "fps", _pos_finite("fps", self.fps))
        sv(self, "m", _pos_finite("m", self.m))
        sv(self, "round_trip", bool(self.round_trip))
        sv(self, "to_blur", bool(self.to_blur))
        steps = self.duration_s * self.fps
        k = round(steps)
        if k < 1 or abs(steps - k) > _GRID_TOL * max(1.0, steps):
            raise ValueError("duration_s * fps 는 1 이상의 정수여야 한다 (종점이 격자에 놓여야 한다)")
        slide = _f("slide_zs", self.slide_zs)
        orb = _f("orbit_deg", self.orbit_deg)
        if not math.isfinite(slide):
            raise ValueError("slide_zs 는 유한해야 한다")
        if not (math.isfinite(orb) and abs(orb) <= 360.0):
            raise ValueError("|orbit_deg| <= 360 이어야 한다")
        sv(self, "slide_zs", slide)
        sv(self, "orbit_deg", orb)
        if self.name != "push_in" and self.m != 1.0:
            raise ValueError("m 은 push_in 에서만 쓴다")
        if self.name != "slider" and slide != 0.0:
            raise ValueError("slide_zs 는 slider 에서만 쓴다")
        if self.name != "orbit" and orb != 0.0:
            raise ValueError("orbit_deg 는 orbit 에서만 쓴다")
        if self.name == "rack_focus":
            if self.focus_from_m is None or self.focus_to_m is None:
                raise ValueError("rack_focus 는 focus_from_m, focus_to_m 이 필요하다")
            sv(self, "focus_from_m", _pos_finite("focus_from_m", self.focus_from_m))
            sv(self, "focus_to_m", _pos_finite("focus_to_m", self.focus_to_m))
        else:
            if self.focus_from_m is not None or self.focus_to_m is not None or self.to_blur:
                raise ValueError("초점 구간·to_blur 는 rack_focus 에서만 쓴다")

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps)) + 1


def recipe(name, z_s, **override) -> MoveSpec:
    """무브 이름과 피사체 깊이로 기본 레시피를 만든다. override 가 기본값을 덮는다."""
    if not isinstance(name, str) or name not in MOVE_NAMES:
        raise ValueError(f"모르는 무브 이름: {name!r}")
    kw = dict(override)
    if name == "push_in":
        kw.setdefault("m", 1.5)
    elif name == "slider":
        kw.setdefault("slide_zs", 0.10)
    elif name == "orbit":
        kw.setdefault("orbit_deg", 20.0)
    else:
        kw.setdefault("focus_from_m", z_s)
    return MoveSpec(name=name, z_s=z_s, **kw)


def bench_spec(intent, z_s) -> MoveSpec:
    """벤치 I1..I5 의 사전 등록 정의식 (8 s, 24 fps)."""
    kw = dict(duration_s=BENCH_DURATION_S, fps=BENCH_FPS)
    if intent == "I1":
        return recipe("push_in", z_s, m=1.5, **kw)
    if intent == "I2":
        return recipe("orbit", z_s, orbit_deg=20.0, **kw)
    if intent == "I3":
        return recipe("orbit", z_s, orbit_deg=20.0, round_trip=True, **kw)
    if intent == "I4":
        return recipe("orbit", z_s, orbit_deg=90.0, **kw)
    if intent == "I5":
        return recipe("orbit", z_s, orbit_deg=180.0, **kw)
    raise ValueError(f"모르는 벤치 의도: {intent!r}")


# --------------------------------------------------------------------------------------------------------------
# 시간축 · 프로파일
# --------------------------------------------------------------------------------------------------------------
def frame_times(spec: MoveSpec) -> np.ndarray:
    return np.arange(spec.n_frames, dtype=np.float64) / spec.fps


def ease_profile(tau, round_trip=False):
    """min-jerk 프로파일 s(tau) (tau 는 [0,1] 클램프). 왕복은 s(2 tau) (tau<=1/2) / s(2 (1-tau))."""
    t = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    if round_trip:
        t = np.where(t <= 0.5, 2.0 * t, 2.0 * (1.0 - t))
    return min_jerk_profile(t)[0]


def _profile(spec: MoveSpec) -> np.ndarray:
    n = spec.n_frames
    return ease_profile(np.arange(n, dtype=np.float64) / (n - 1), spec.round_trip)


# --------------------------------------------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------------------------------------------
@dataclass(eq=False)
class MovePath:
    spec: MoveSpec
    c0: np.ndarray
    t: np.ndarray
    c2w: np.ndarray
    focus_m: np.ndarray
    subject: np.ndarray
    rig: str
    traj: object = None

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    def to_shot(self, theta, width, height, *, seed=0):
        return to_shot(self, theta, width, height, seed=seed)


def _subject(c0, z_s):
    return (c0 @ np.array([0.0, 0.0, z_s, 1.0]))[:3].copy()


def _focus_from_poses(c2w, subject):
    return np.linalg.norm(c2w[:, :3, 3] - subject[None, :], axis=1)


def _local_poses(spec: MoveSpec, s: np.ndarray) -> np.ndarray:
    n = s.shape[0]
    L = np.zeros((n, 4, 4), dtype=np.float64)
    L[:] = np.eye(4)
    z = spec.z_s
    if spec.name == "push_in":
        L[:, 2, 3] = z * (1.0 - 1.0 / spec.m) * s
    elif spec.name == "slider":
        L[:, 0, 3] = spec.slide_zs * z * s
    elif spec.name == "orbit":
        a = math.radians(spec.orbit_deg) * s
        c, sn = np.cos(a), np.sin(a)
        L[:, 0, 0] = c
        L[:, 0, 2] = sn
        L[:, 2, 0] = -sn
        L[:, 2, 2] = c
        # t = P - R P,  P = (0,0,z)
        L[:, 0, 3] = -z * sn
        L[:, 2, 3] = z - z * c
    return L


def _rack_focus(spec: MoveSpec, s: np.ndarray) -> np.ndarray:
    d0 = 1.0 / spec.focus_from_m
    d1 = 1.0 / spec.focus_to_m
    return 1.0 / (d0 + s * (d1 - d0))


def ideal_path(spec: MoveSpec, c0=None) -> MovePath:
    """정의식 그대로의 이상 경로. c2w_k = C_0 L_k."""
    c0 = _check_c0(c0)
    s = _profile(spec)
    subject = _subject(c0, spec.z_s)
    if spec.name == "rack_focus":
        c2w = np.broadcast_to(c0, (s.shape[0], 4, 4)).copy()
        focus = _rack_focus(spec, s)
    else:
        c2w = c0[None] @ _local_poses(spec, s)
        c2w[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
        focus = _focus_from_poses(c2w, subject)
    return MovePath(spec=spec, c0=c0.copy(), t=frame_times(spec), c2w=c2w, focus_m=focus, subject=subject,
                    rig="ideal", traj=None)


def rig_path(spec: MoveSpec, c0=None, *, seed=0) -> MovePath:
    """CAM-01 리그(1 kHz ODE)로 실현한 경로."""
    ideal = ideal_path(spec, c0)
    if spec.name == "rack_focus":
        return replace(ideal, rig="locked")
    c0 = ideal.c0
    T = spec.duration_s
    z = spec.z_s
    if spec.name == "push_in":
        rig_name, geom, end = "dolly", None, [z * (1.0 - 1.0 / spec.m)]
    elif spec.name == "slider":
        rig_name, geom, end = "slider", None, [spec.slide_zs * z]
    else:
        rig_name, geom, end = "crane_jib", {"radius": z}, [math.radians(spec.orbit_deg), 0.0]
    rig = make_rig(rig_name, geom=geom, home_c2w=c0)
    end = np.array(end, dtype=np.float64)
    zero = np.zeros_like(end)
    if spec.round_trip:
        intent = Intent(np.array([0.0, T / 2.0, T]), np.stack([zero, end, zero]), 0.0)
    else:
        intent = Intent(np.array([0.0, T]), np.stack([zero, end]), 0.0)
    traj = rig.simulate(intent, seed=seed)
    c2w = np.array(traj.frame_poses(spec.fps), dtype=np.float64)
    if c2w.shape[0] != spec.n_frames:
        raise ValueError("리그 궤적의 프레임 수가 격자와 다르다")
    c2w[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return MovePath(spec=spec, c0=c0.copy(), t=ideal.t.copy(), c2w=c2w,
                    focus_m=_focus_from_poses(c2w, ideal.subject), subject=ideal.subject.copy(),
                    rig=rig_name, traj=traj)


def to_shot(path: MovePath, theta, width, height, *, seed=0):
    """MovePath -> FND-02 Shot. K 는 전 프레임 동일, 배열은 복사."""
    from buc.core.contracts import Shot

    if abs(float(theta.fps) - path.spec.fps) > 1e-9:
        raise ValueError("theta.fps 가 spec.fps 와 다르다")
    K = np.asarray(theta.intrinsics(width, height), dtype=np.float64)
    n = path.n_frames
    Ks = np.broadcast_to(K, (n, 3, 3)).copy()
    return Shot(c2w=path.c2w.copy(), t=path.t.copy(), K=Ks, focus_m=path.focus_m.copy(), width=width,
                height=height, fps=path.spec.fps, theta=theta, move=path.spec.name, rig=path.rig, seed=seed)


# --------------------------------------------------------------------------------------------------------------
# 봉투 Pi_g
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PathMeasure:
    alpha_deg: float
    yaw_deg: float
    m_peak: float
    disp_m: float


def _angle_deg(a, b):
    cr = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cr, np.sum(a * b, axis=-1)))


def measure(c2w, z_s) -> PathMeasure:
    """경로의 오빗각·조준 오차·배율 최대·카메라 변위 최대를 잰다 (롤은 측정 안 함)."""
    z = _pos_finite("z_s", z_s)
    try:
        c = np.asarray(c2w, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("c2w 는 숫자 배열이어야 한다") from e
    if c.ndim != 3 or c.shape[0] < 1 or c.shape[1:] != (4, 4):
        raise ValueError("c2w 는 (N,4,4) 여야 한다")
    if not np.isfinite(c).all():
        raise ValueError("c2w 가 유한하지 않다")
    R = c[:, :3, :3]
    if (np.abs(R @ np.swapaxes(R, 1, 2) - np.eye(3)).max() > 1e-6 or np.linalg.det(R).min() < 1.0 - 1e-6
            or np.abs(c[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-6):
        raise ValueError("c2w 가 회전 + 평행이동(SE(3))이 아니다")
    p_s = (c[0] @ np.array([0.0, 0.0, z, 1.0]))[:3]
    pos = c[:, :3, 3]
    v = pos - p_s[None, :]
    nv = np.linalg.norm(v, axis=1)
    if nv.min() <= 1e-9:
        raise ValueError("카메라가 피사체 위에 있다")
    f = R[:, :, 2]
    alpha = float(_angle_deg(v, v[0][None, :]).max())
    yaw = float(_angle_deg(f, -v).max())
    m_peak = float((nv[0] / nv).max())
    disp = float(np.linalg.norm(pos - pos[0][None, :], axis=1).max())
    return PathMeasure(alpha_deg=alpha, yaw_deg=yaw, m_peak=m_peak, disp_m=disp)


@dataclass(frozen=True)
class Envelope:
    grade: str
    max_alpha_deg: float
    max_yaw_deg: float
    max_slide_zs: float
    m_max: float
    max_disp_m: float
    ir_cap_mean: Optional[float]
    ir_cap_max: Optional[float]

    def contains(self, ms: PathMeasure) -> bool:
        return bool(ms.alpha_deg <= self.max_alpha_deg + BOUND_TOL
                    and ms.yaw_deg <= self.max_yaw_deg + BOUND_TOL
                    and ms.m_peak <= self.m_max * (1.0 + BOUND_TOL)
                    and ms.disp_m <= self.max_disp_m * (1.0 + BOUND_TOL))

    def contains_path(self, c2w, z_s) -> bool:
        return self.contains(measure(c2w, z_s))

    def contains_spec(self, spec: MoveSpec) -> bool:
        if spec.to_blur and self.grade == "safe":
            return False
        if abs(spec.slide_zs) > self.max_slide_zs * (1.0 + BOUND_TOL):
            return False
        return self.contains(measure(ideal_path(spec).c2w, spec.z_s))


def envelope(grade, *, m_max=M_MAX_SEMANTIC, observed_depth_m=_INF) -> Envelope:
    if not isinstance(grade, str) or grade not in GRADES:
        raise ValueError(f"모르는 등급: {grade!r}")
    mm = _f("m_max", m_max)
    dd = _f("observed_depth_m", observed_depth_m)
    if math.isnan(mm) or mm <= 0.0:
        raise ValueError("m_max > 0 이어야 한다")
    if math.isnan(dd) or dd <= 0.0:
        raise ValueError("observed_depth_m > 0 이어야 한다")
    if grade == "safe":
        return Envelope("safe", 20.0, 20.0, 0.15, mm, dd, 0.20, 0.30)
    if grade == "extended":
        return Envelope("extended", 90.0, 90.0, _INF, mm, dd, 0.65, 0.85)
    return Envelope("full", _INF, _INF, _INF, _INF, _INF, None, None)


def min_grade(spec: MoveSpec, *, m_max=M_MAX_SEMANTIC, observed_depth_m=_INF) -> str:
    """spec 을 포함하는 가장 작은 등급."""
    for g in GRADES:
        if envelope(g, m_max=m_max, observed_depth_m=observed_depth_m).contains_spec(spec):
            return g
    return "full"


def min_grade_path(c2w, z_s, *, m_max=M_MAX_SEMANTIC, observed_depth_m=_INF) -> str:
    """경로 측정을 포함하는 가장 작은 등급."""
    ms = measure(c2w, z_s)
    for g in GRADES:
        if envelope(g, m_max=m_max, observed_depth_m=observed_depth_m).contains(ms):
            return g
    return "full"
