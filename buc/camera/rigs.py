"""CAM-01 카메라 리그 1 kHz ODE: 14종 리그, min-jerk 기준 궤적, OU 외란.

"의도 -> 물리 -> 포즈" 분해의 물리층. 순수 수식(numpy / scipy), 학습 없음, torch 금지.
규약: 세계 오른손·미터, 카메라 OpenCV(x 오른쪽, y 아래, z 앞), 포즈는 4x4 c2w float64.
리그의 일반화 좌표 q (m 또는 rad)만 포즈를 정한다(홀로노믹): 포즈 = home_c2w @ L(q).

자유도마다 독립인 ODE (평형 q=0), RK4, 스텝 1 ms, 한 스텝 동안 힘 u 일정(ZOH):
    m q'' = u - c q' - c2 |q'| q' - k q - f_c tanh(q'/V_EPS) - m w2 sin(q)
import 부작용·전역 RNG 사용 없음.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "DT", "RATE_HZ", "V_EPS", "PARAM_KEYS", "TIER1", "RIG_NAMES",
    "min_jerk_profile", "min_jerk_path", "power_law_speed", "ou_noise", "Intent",
    "Rig", "Trajectory", "make_rig", "default_params",
]

DT = 1e-3
RATE_HZ = 1000
V_EPS = 1e-3
PARAM_KEYS = ("mass", "damping", "quad_drag", "stiffness", "coulomb", "w2", "kp", "ki", "kd", "ff_gain",
              "u_max", "ou_sigma", "ou_tau")
TIER1 = ("slider", "dolly", "crane_jib")
RIG_NAMES = ("tripod_fluid", "dolly", "slider", "crane_jib", "techno_crane", "steadicam", "gimbal_3axis",
             "shoulder_handheld", "cine_drone", "fpv", "cablecam", "car_mount", "motion_control", "underwater")

_OU_SEED_TAG = 0xCA01


# ---------------------------------------------------------------------------------------------------------------
# 순수 함수
# ---------------------------------------------------------------------------------------------------------------
def min_jerk_profile(tau):
    """정규화 시간 tau (스칼라·배열, [0,1] 클램프) -> (s, ds/dtau, d2s/dtau2). s = 10t^3 - 15t^4 + 6t^5."""
    t = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    s = 10.0 * t3 - 15.0 * t3 * t + 6.0 * t3 * t2
    ds = 30.0 * t2 - 60.0 * t3 + 30.0 * t3 * t
    d2s = 60.0 * t - 180.0 * t2 + 120.0 * t3
    return s, ds, d2s


def _check_path(times, waypoints):
    times = np.asarray(times, dtype=np.float64)
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times 는 길이 2 이상의 1차원이어야 한다")
    if waypoints.ndim != 2 or waypoints.shape[0] != times.shape[0] or waypoints.shape[1] < 1:
        raise ValueError("waypoints 는 (K, n) 이고 K == len(times) 여야 한다")
    if not (np.isfinite(times).all() and np.isfinite(waypoints).all()):
        raise ValueError("times / waypoints 가 유한하지 않다")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times 는 엄격 증가여야 한다")
    return times, waypoints


def min_jerk_path(times, waypoints, t):
    """웨이포인트마다 정지하는 구간별 min-jerk. t (M,) -> (M,n) x3, 스칼라 t -> (n,) x3. 범위 밖은 끝점 유지."""
    times, waypoints = _check_path(times, waypoints)
    t_arr = np.asarray(t, dtype=np.float64)
    scalar = t_arr.ndim == 0
    tt = np.atleast_1d(t_arr).reshape(-1)
    k = times.shape[0]
    i = np.clip(np.searchsorted(times, tt, side="right") - 1, 0, k - 2)
    dur = (times[i + 1] - times[i])
    tau = (tt - times[i]) / dur
    s, ds, d2s = min_jerk_profile(tau)
    w0 = waypoints[i]
    delta = waypoints[i + 1] - w0
    x = w0 + delta * s[:, None]
    v = delta * (ds / dur)[:, None]
    a = delta * (d2s / (dur * dur))[:, None]
    # 범위 밖은 정확히 끝점 (v = a = 0)
    lo = tt <= times[0]
    hi = tt >= times[-1]
    if lo.any():
        x[lo] = waypoints[0]
        v[lo] = 0.0
        a[lo] = 0.0
    if hi.any():
        x[hi] = waypoints[-1]
        v[hi] = 0.0
        a[hi] = 0.0
    if scalar:
        return x[0], v[0], a[0]
    return x, v, a


def power_law_speed(curvature, gain=1.0):
    """2/3 멱법칙: v = gain * kappa^(-1/3) (각속도 gain * kappa^(2/3)). kappa <= 0 이 하나라도 있으면 ValueError."""
    k = np.asarray(curvature, dtype=np.float64)
    if not np.all(k > 0.0):
        raise ValueError("curvature 는 모두 0 보다 커야 한다")
    return gain * np.power(k, -1.0 / 3.0)


def ou_noise(n, dt, tau, sigma, rng):
    """정확 이산화 OU: a = exp(-dt/tau), 정상 분산 sigma^2, 첫 표본도 정상분포. sigma = 0 이면 영. (n,) float64."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n >= 1 정수여야 한다")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt > 0 이어야 한다")
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError("tau > 0 이어야 한다")
    if not (np.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma >= 0 이어야 한다")
    n = int(n)
    if sigma == 0.0:
        return np.zeros(n, dtype=np.float64)
    a = math.exp(-dt / tau)
    z = rng.standard_normal(n)
    w = z * (sigma * math.sqrt(1.0 - a * a))
    w[0] = z[0] * sigma                                    # 첫 표본도 정상분포
    return lfilter([1.0], [1.0, -a], w).astype(np.float64)


@dataclass(frozen=True, eq=False)
class Intent:
    """감독의 의도: 웨이포인트(일반화 좌표) 열 + 끝 정지 시간. 웨이포인트마다 정지하는 min-jerk 기준 궤적."""
    times: np.ndarray
    waypoints: np.ndarray
    tail_s: float = 0.0

    def __post_init__(self):
        times, wp = _check_path(self.times, self.waypoints)
        if times[0] != 0.0:
            raise ValueError("times[0] 은 0 이어야 한다")
        tail = float(self.tail_s)
        if not (np.isfinite(tail) and tail >= 0.0):
            raise ValueError("tail_s >= 0 이어야 한다")
        times = np.array(times, dtype=np.float64)
        wp = np.array(wp, dtype=np.float64)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "waypoints", wp)
        object.__setattr__(self, "tail_s", tail)

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] + self.tail_s)

    @property
    def n_q(self) -> int:
        return int(self.waypoints.shape[1])

    def reference(self, t):
        """시각 t (스칼라·(M,)) -> (q, qd, qdd) 기준."""
        return min_jerk_path(self.times, self.waypoints, t)

    @classmethod
    def point_to_point(cls, q0, q1, duration_s, tail_s=0.0):
        q0 = np.atleast_1d(np.asarray(q0, dtype=np.float64))
        q1 = np.atleast_1d(np.asarray(q1, dtype=np.float64))
        if q0.ndim != 1 or q0.shape != q1.shape:
            raise ValueError("q0, q1 은 같은 길이의 1차원이어야 한다")
        d = float(duration_s)
        if not (np.isfinite(d) and d > 0):
            raise ValueError("duration_s > 0 이어야 한다")
        return cls(np.array([0.0, d]), np.stack([q0, q1]), tail_s)


# ---------------------------------------------------------------------------------------------------------------
# 회전·변환 (벡터화)
# ---------------------------------------------------------------------------------------------------------------
def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _se3(rot, trans):
    """rot (...,3,3) / None, trans (...,3) / None -> (...,4,4)."""
    shape = (rot.shape[:-2] if rot is not None else trans.shape[:-1])
    out = np.zeros(shape + (4, 4), dtype=np.float64)
    if rot is None:
        out[..., :3, :3] = np.eye(3)
    else:
        out[..., :3, :3] = rot
    if trans is not None:
        out[..., :3, 3] = trans
    out[..., 3, 3] = 1.0
    return out


def _mm(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = out @ m
    return out


# 리그별 국소 변환 L(q): q (..., n_q), axis (3,), radius -> (...,4,4)
def _L_axis_translate(q, axis, radius):
    return _se3(None, q[..., 0:1] * axis)


def _L_cablecam(q, axis, radius):
    return _se3(_rot_z(q[..., 1]), q[..., 0:1] * axis)


def _L_car_mount(q, axis, radius):
    return _se3(None, q[..., 0:1] * axis + q[..., 1:2] * np.array([0.0, 1.0, 0.0]))


def _L_crane(q, axis, radius):
    r = radius + (q[..., 2] if q.shape[-1] > 2 else 0.0)
    rot = _rot_y(q[..., 0]) @ _rot_x(q[..., 1])
    back = np.stack([np.zeros_like(r), np.zeros_like(r), -r], -1)
    pivot = np.array([0.0, 0.0, radius])
    return _se3(rot, pivot + np.einsum("...ij,...j->...i", rot, back))


def _L_tripod(q, axis, radius):
    return _se3(_rot_y(q[..., 0]) @ _rot_x(q[..., 1]), None)


def _L_gimbal(q, axis, radius):
    return _se3(_mm(_rot_y(q[..., 0]), _rot_x(q[..., 1]), _rot_z(q[..., 2])), None)


def _L_steadicam(q, axis, radius):
    t = np.stack([q[..., 0], np.zeros_like(q[..., 0]), q[..., 1]], -1)
    return _se3(_rot_y(q[..., 2]) @ _rot_z(q[..., 3]), t)


def _L_free6(q, axis, radius):
    return _se3(_mm(_rot_y(q[..., 3]), _rot_x(q[..., 4]), _rot_z(q[..., 5])), q[..., :3])


def _L_free4(q, axis, radius):
    return _se3(_rot_y(q[..., 3]), q[..., :3])


# ---------------------------------------------------------------------------------------------------------------
# 기본 파라미터
# ---------------------------------------------------------------------------------------------------------------
def _vec(x, n):
    a = np.asarray(x, dtype=np.float64)
    return np.broadcast_to(a, (n,)).copy()


def _tuned(n, *, m, c=0.0, c2=0.0, k=0.0, fc=0.0, w2=0.0, wn, zeta=1.0, ki_frac=0.1, ff=1.0, u_max,
           sigma=0.0, tau=0.1, delay=0.0):
    """질량·감쇠로부터 폐루프 고유진동수 wn, 감쇠비 zeta 가 되도록 PD(+I) 이득을 정한다."""
    m, c, wn, zeta = _vec(m, n), _vec(c, n), _vec(wn, n), _vec(zeta, n)
    kp = m * wn * wn
    kd = np.maximum(2.0 * zeta * m * wn - c, 0.0)
    ki = _vec(ki_frac, n) * kp * wn
    out = dict(mass=m, damping=c, quad_drag=_vec(c2, n), stiffness=_vec(k, n), coulomb=_vec(fc, n),
               w2=_vec(w2, n), kp=kp, ki=ki, kd=kd, ff_gain=_vec(ff, n), u_max=_vec(u_max, n),
               ou_sigma=_vec(sigma, n), ou_tau=_vec(tau, n))
    return {k_: tuple(float(x) for x in v) for k_, v in out.items()} | {"delay_s": float(delay)}


# 리그 명세: n_q, q_scale, 기본 기하(axis), L(q), 기본 파라미터 생성기
_SPECS = {
    "tripod_fluid": dict(
        n=2, qs=(1.2, 0.6), axis=(1.0, 0.0, 0.0), L=_L_tripod,
        p=lambda: _tuned(2, m=2.0, c=6.0, fc=0.05, wn=15.0, u_max=500.0)),
    "dolly": dict(
        n=1, qs=(5.0,), axis=(0.0, 0.0, 1.0), L=_L_axis_translate,
        p=lambda: _tuned(1, m=80.0, c=60.0, c2=5.0, wn=10.0, u_max=5000.0)),
    "slider": dict(
        n=1, qs=(1.0,), axis=(1.0, 0.0, 0.0), L=_L_axis_translate,
        p=lambda: _tuned(1, m=2.0, c=4.0, wn=30.0, u_max=500.0)),
    "crane_jib": dict(
        n=2, qs=(0.6, 0.3), axis=(1.0, 0.0, 0.0), L=_L_crane,
        p=lambda: _tuned(2, m=(150.0, 100.0), c=(80.0, 60.0), wn=8.0, u_max=20000.0)),
    "techno_crane": dict(
        n=3, qs=(0.6, 0.3, 1.0), axis=(1.0, 0.0, 0.0), L=_L_crane,
        p=lambda: _tuned(3, m=(200.0, 150.0, 40.0), c=(100.0, 80.0, 50.0), wn=8.0, u_max=30000.0)),
    "steadicam": dict(
        n=4, qs=(1.0, 1.0, 0.6, 0.15), axis=(1.0, 0.0, 0.0), L=_L_steadicam,
        p=lambda: _tuned(4, m=(8.0, 8.0, 1.5, 0.5), c=(12.0, 12.0, 3.0, 0.8), w2=(0.0, 0.0, 0.0, 25.0),
                         wn=(12.0, 12.0, 15.0, 20.0), u_max=500.0)),
    "gimbal_3axis": dict(
        n=3, qs=(1.0, 0.6, 0.3), axis=(1.0, 0.0, 0.0), L=_L_gimbal,
        p=lambda: _tuned(3, m=(0.4, 0.3, 0.2), c=1.0, wn=40.0, u_max=100.0)),
    "shoulder_handheld": dict(
        n=6, qs=(0.3, 0.3, 0.3, 0.2, 0.2, 0.1), axis=(1.0, 0.0, 0.0), L=_L_free6,
        p=lambda: _tuned(6, m=(5.0, 5.0, 5.0, 0.5, 0.5, 0.3), c=(10.0, 10.0, 10.0, 1.0, 1.0, 0.6),
                         wn=2.5, zeta=1.0, ki_frac=0.0, sigma=(0.8, 0.8, 0.8, 0.04, 0.04, 0.02), tau=0.15,
                         delay=0.2, u_max=500.0)),
    "cine_drone": dict(
        n=4, qs=(3.0, 2.0, 2.0, 0.8), axis=(1.0, 0.0, 0.0), L=_L_free4,
        p=lambda: _tuned(4, m=(3.0, 3.0, 3.0, 0.3), c=0.5, c2=(0.8, 0.8, 0.8, 0.02), wn=12.0, u_max=200.0)),
    "fpv": dict(
        n=6, qs=(4.0, 2.0, 4.0, 1.2, 0.8, 1.5), axis=(1.0, 0.0, 0.0), L=_L_free6,
        p=lambda: _tuned(6, m=(0.8, 0.8, 0.8, 0.1, 0.1, 0.1), c=0.3, c2=(0.1, 0.1, 0.1, 0.0, 0.0, 0.0),
                         wn=25.0, u_max=200.0)),
    "cablecam": dict(
        n=2, qs=(10.0, 0.3), axis=(1.0, 0.0, 0.0), L=_L_cablecam,
        p=lambda: _tuned(2, m=(15.0, 0.6), c=(10.0, 0.3), w2=(0.0, 20.0), wn=(6.0, 12.0), u_max=2000.0)),
    "car_mount": dict(
        n=2, qs=(10.0, 0.2), axis=(0.0, 0.0, 1.0), L=_L_car_mount,
        p=lambda: _tuned(2, m=(1200.0, 20.0), c=(400.0, 200.0), k=(0.0, 3000.0), wn=(3.0, 8.0),
                         sigma=(50.0, 15.0), tau=(0.2, 0.05), u_max=20000.0)),
    "motion_control": dict(
        n=6, qs=(1.0, 1.0, 1.0, 0.6, 0.4, 0.3), axis=(1.0, 0.0, 0.0), L=_L_free6,
        p=lambda: _tuned(6, m=(5.0, 5.0, 5.0, 0.5, 0.5, 0.5), c=2.0, wn=60.0, u_max=5000.0)),
    "underwater": dict(
        n=4, qs=(3.0, 2.0, 2.0, 0.8), axis=(1.0, 0.0, 0.0), L=_L_free4,
        p=lambda: _tuned(4, m=(40.0, 40.0, 40.0, 6.0), c=(30.0, 30.0, 30.0, 8.0),
                         c2=(60.0, 60.0, 60.0, 10.0), wn=5.0, u_max=3000.0)),
}
assert tuple(_SPECS) == tuple(sorted(_SPECS, key=RIG_NAMES.index)) and set(_SPECS) == set(RIG_NAMES)

_GEOM_KEYS = ("axis", "radius")
_NONNEG = ("damping", "quad_drag", "stiffness", "coulomb", "w2", "kp", "ki", "kd", "ff_gain", "ou_sigma")


def default_params(name):
    """리그의 기본 파라미터: PARAM_KEYS 별 길이 n_q float 튜플 + delay_s float. 호출마다 새 객체."""
    if name not in _SPECS:
        raise KeyError(name)
    return _SPECS[name]["p"]()


def _build_params(name, n, params):
    base = default_params(name)
    if params is not None:
        allowed = set(PARAM_KEYS) | {"delay_s"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"모르는 파라미터 키: {sorted(unknown)}")
        for k, v in params.items():
            if k == "delay_s":
                base[k] = v
                continue
            try:
                a = np.asarray(v, dtype=np.float64)
                a = np.broadcast_to(a, (n,)) if a.ndim == 0 or a.shape == (n,) else None
            except (TypeError, ValueError):
                a = None
            if a is None:
                raise ValueError(f"{k}: 스칼라 또는 길이 {n} 벡터여야 한다")
            base[k] = tuple(float(x) for x in a)
    out = {}
    for k in PARAM_KEYS:
        a = np.array(base[k], dtype=np.float64)
        if np.isnan(a).any() or (k != "u_max" and not np.isfinite(a).all()):
            raise ValueError(f"{k}: 유한하지 않다")
        if k in ("mass", "ou_tau", "u_max") and not np.all(a > 0):
            raise ValueError(f"{k} > 0 이어야 한다")
        if k in _NONNEG and not np.all(a >= 0):
            raise ValueError(f"{k} >= 0 이어야 한다")
        out[k] = a
    try:
        d = float(base["delay_s"])
    except (TypeError, ValueError) as e:
        raise ValueError("delay_s 는 스칼라여야 한다") from e
    if not (np.isfinite(d) and d >= 0.0):
        raise ValueError("delay_s >= 0 이어야 한다")
    out["delay_s"] = d
    return out


def _build_geom(spec, geom):
    out = {"axis": np.array(spec["axis"], dtype=np.float64), "radius": 3.0}
    if geom is not None:
        unknown = set(geom) - set(_GEOM_KEYS)
        if unknown:
            raise ValueError(f"모르는 geom 키: {sorted(unknown)}")
        if "axis" in geom:
            ax = np.asarray(geom["axis"], dtype=np.float64)
            if ax.shape != (3,) or not np.isfinite(ax).all():
                raise ValueError("axis 는 유한한 3벡터여야 한다")
            out["axis"] = ax.copy()
        if "radius" in geom:
            try:
                out["radius"] = float(geom["radius"])
            except (TypeError, ValueError) as e:
                raise ValueError("radius 는 스칼라여야 한다") from e
    norm = float(np.linalg.norm(out["axis"]))
    if not norm > 0.0:
        raise ValueError("axis 는 0벡터일 수 없다")
    out["axis"] = out["axis"] / norm
    if not (np.isfinite(out["radius"]) and out["radius"] > 0.0):
        raise ValueError("radius > 0 이어야 한다")
    return out


def _check_home(home):
    if home is None:
        return np.eye(4)
    h = np.array(home, dtype=np.float64)
    if h.shape != (4, 4) or not np.isfinite(h).all():
        raise ValueError("home_c2w 는 유한한 4x4 여야 한다")
    r = h[:3, :3]
    if (not np.allclose(h[3], [0, 0, 0, 1], atol=1e-9) or np.abs(r @ r.T - np.eye(3)).max() > 1e-6
            or abs(np.linalg.det(r) - 1.0) > 1e-6):
        raise ValueError("home_c2w 는 SE(3) 여야 한다")
    return h


# ---------------------------------------------------------------------------------------------------------------
# 적분기
# ---------------------------------------------------------------------------------------------------------------
def _integrate_dof(m, c, c2, k, fc, w2, q0, v0, steps, u_open, ctrl):
    """한 자유도 RK4(스텝 DT, u 일정). u_open: 길이 steps 리스트(개루프) 또는 None.
    ctrl: (q_ref, qd_ref, u_ff, kp, ki, kd, u_max, D, d) 리스트/스칼라(폐루프) 또는 None. -> (q, qd, u) 리스트."""
    h = DT
    hh = 0.5 * h
    h6 = h / 6.0
    tanh, sin = math.tanh, math.sin
    inv_m = 1.0 / m
    inv_eps = 1.0 / V_EPS
    qs = [0.0] * (steps + 1)
    vs = [0.0] * (steps + 1)
    us = [0.0] * steps
    q, v = q0, v0
    qs[0], vs[0] = q, v
    integ = 0.0
    if ctrl is not None:
        qr, vr, uff, kp, ki, kd, umax, dly, dist = ctrl
    for i in range(steps):
        if ctrl is None:
            u = u_open[i]
        else:
            j = i - dly if i > dly else 0
            e = qr[j] - qs[j]
            ed = vr[j] - vs[j]
            u = uff[i] + kp * e + kd * ed + ki * integ
            if u > umax:
                u = umax
            elif u < -umax:
                u = -umax
            u += dist[i]
            integ += e * h
        us[i] = u
        a1 = (u - c * v - c2 * abs(v) * v - k * q - fc * tanh(v * inv_eps)) * inv_m - w2 * sin(q)
        q2 = q + hh * v
        v2 = v + hh * a1
        a2 = (u - c * v2 - c2 * abs(v2) * v2 - k * q2 - fc * tanh(v2 * inv_eps)) * inv_m - w2 * sin(q2)
        q3 = q + hh * v2
        v3 = v + hh * a2
        a3 = (u - c * v3 - c2 * abs(v3) * v3 - k * q3 - fc * tanh(v3 * inv_eps)) * inv_m - w2 * sin(q3)
        q4 = q + h * v3
        v4 = v + h * a3
        a4 = (u - c * v4 - c2 * abs(v4) * v4 - k * q4 - fc * tanh(v4 * inv_eps)) * inv_m - w2 * sin(q4)
        q = q + h6 * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v = v + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        qs[i + 1], vs[i + 1] = q, v
    return qs, vs, us


def _disturbance(params, n, steps, seed):
    """자유도 순서로 하나의 rng 에서 뽑는 OU 외란 (steps, n)."""
    d = np.zeros((steps, n), dtype=np.float64)
    if not np.any(params["ou_sigma"] > 0):
        return d
    try:
        ss = np.random.SeedSequence([int(seed), _OU_SEED_TAG])
    except (TypeError, ValueError) as e:
        raise ValueError("seed 는 0 이상의 정수여야 한다") from e
    rng = np.random.default_rng(ss)
    for j in range(n):
        s = float(params["ou_sigma"][j])
        if s > 0:
            d[:, j] = ou_noise(steps, DT, float(params["ou_tau"][j]), s, rng)
    return d


def _vec_state(x, n, name, default):
    if x is None:
        return np.array(default, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: 유한한 길이 {n} 벡터여야 한다")
    return a.copy()


def _num_steps(duration_s):
    try:
        d = float(duration_s)
    except (TypeError, ValueError) as e:
        raise ValueError("duration_s 는 숫자여야 한다") from e
    if not (np.isfinite(d) and d > 0):
        raise ValueError("duration_s > 0 이어야 한다")
    steps = int(round(d / DT))
    if steps < 1:
        raise ValueError("duration_s 가 한 스텝(1 ms)보다 짧다")
    return steps


# ---------------------------------------------------------------------------------------------------------------
# Rig / Trajectory
# ---------------------------------------------------------------------------------------------------------------
@dataclass(eq=False)
class Trajectory:
    """시뮬 결과. q, qd, q_ref, u: (N, n_q); u[k] 는 [t_k, t_{k+1}] 에 인가된 힘(마지막 행은 직전 복사)."""
    rig: "Rig"
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_ref: np.ndarray | None
    u: np.ndarray
    pose: np.ndarray
    dt: float = DT

    def _hermite(self, times):
        tt = np.asarray(times, dtype=np.float64)
        scalar = tt.ndim == 0
        tt = np.atleast_1d(tt).reshape(-1)
        t_end = float(self.t[-1])
        if not np.isfinite(tt).all() or tt.min() < -1e-9 or tt.max() > t_end + 1e-9:
            raise ValueError(f"시각이 [0, {t_end}] 밖이다")
        tt = np.clip(tt, 0.0, t_end)
        i = np.clip(np.floor(tt / self.dt).astype(np.int64), 0, len(self.t) - 2)
        s = (tt - self.t[i]) / self.dt
        s2, s3 = s * s, s * s * s
        h00 = (2 * s3 - 3 * s2 + 1)[:, None]
        h10 = (s3 - 2 * s2 + s)[:, None]
        h01 = (-2 * s3 + 3 * s2)[:, None]
        h11 = (s3 - s2)[:, None]
        q = h00 * self.q[i] + h10 * self.dt * self.qd[i] + h01 * self.q[i + 1] + h11 * self.dt * self.qd[i + 1]
        return q, scalar

    def q_at(self, times):
        """(q, qd) 3차 에르미트 보간한 q. 스칼라 -> (n_q,), (M,) -> (M, n_q). 범위 밖 ValueError."""
        q, scalar = self._hermite(times)
        return q[0] if scalar else q

    def pose_at(self, times):
        """보간한 q 의 FK. 스칼라 -> (4,4), (M,) -> (M,4,4)."""
        q, scalar = self._hermite(times)
        p = self.rig.fk(q)
        return p[0] if scalar else p

    def frame_poses(self, fps):
        """t_k = k/fps, M = floor(T*fps + 1e-9) + 1 개 프레임의 c2w (M,4,4)."""
        f = float(fps)
        if not (np.isfinite(f) and f > 0):
            raise ValueError("fps > 0 이어야 한다")
        m = int(math.floor(float(self.t[-1]) * f + 1e-9)) + 1
        return self.pose_at(np.arange(m) / f)


class Rig:
    """카메라 리그: 일반화 좌표 q -> c2w 포즈(홀로노믹), 1 kHz ODE 시뮬."""

    def __init__(self, name, spec, params, geom, home_c2w):
        self.name = name
        self.n_q = int(spec["n"])
        self.q_scale = np.array(spec["qs"], dtype=np.float64)
        self.params = params
        self.geom = geom
        self.home_c2w = home_c2w
        self._L = spec["L"]

    def __repr__(self):
        return f"Rig({self.name!r}, n_q={self.n_q})"

    def fk(self, q):
        """q (..., n_q) -> c2w (..., 4, 4) = home_c2w @ L(q)."""
        a = np.asarray(q, dtype=np.float64)
        if a.ndim < 1 or a.shape[-1] != self.n_q:
            raise ValueError(f"q 의 마지막 축은 {self.n_q} 여야 한다")
        L = self._L(a, self.geom["axis"], self.geom["radius"])
        out = self.home_c2w @ L
        out[..., 3, :] = (0.0, 0.0, 0.0, 1.0)
        return out

    def _finish(self, qs, vs, us, q_ref):
        q = np.array(qs, dtype=np.float64).T.copy()
        qd = np.array(vs, dtype=np.float64).T.copy()
        u = np.array(us, dtype=np.float64).T
        u = np.concatenate([u, u[-1:]], axis=0)
        n = q.shape[0]
        return Trajectory(rig=self, t=np.arange(n) * DT, q=q, qd=qd, q_ref=q_ref, u=u, pose=self.fk(q), dt=DT)

    def simulate(self, intent, *, seed=0, q0=None, qd0=None):
        """의도(기준 궤적)를 역동역학 ff + 지연 PID 로 추종하는 폐루프 시뮬 -> Trajectory."""
        if not isinstance(intent, Intent):
            raise ValueError("intent 는 Intent 여야 한다")
        n = self.n_q
        if intent.n_q != n:
            raise ValueError(f"intent 자유도 {intent.n_q} != 리그 자유도 {n}")
        q0 = _vec_state(q0, n, "q0", intent.waypoints[0])
        qd0 = _vec_state(qd0, n, "qd0", np.zeros(n))
        steps = _num_steps(intent.duration_s)
        p = self.params
        t = np.arange(steps + 1) * DT
        qr, vr, ar = intent.reference(t)
        uff = (p["mass"] * ar + p["damping"] * vr + p["quad_drag"] * np.abs(vr) * vr + p["stiffness"] * qr
               + p["coulomb"] * np.tanh(vr / V_EPS) + p["mass"] * p["w2"] * np.sin(qr))
        uff = p["ff_gain"] * uff
        dist = _disturbance(p, n, steps, seed)
        dly = int(round(p["delay_s"] / DT))
        qs, vs, us = [], [], []
        for j in range(n):
            ctrl = (qr[:, j].tolist(), vr[:, j].tolist(), uff[:, j].tolist(), float(p["kp"][j]),
                    float(p["ki"][j]), float(p["kd"][j]), float(p["u_max"][j]), dly, dist[:, j].tolist())
            a, b, c = _integrate_dof(float(p["mass"][j]), float(p["damping"][j]), float(p["quad_drag"][j]),
                                     float(p["stiffness"][j]), float(p["coulomb"][j]), float(p["w2"][j]),
                                     float(q0[j]), float(qd0[j]), steps, None, ctrl)
            qs.append(a)
            vs.append(b)
            us.append(c)
        return self._finish(qs, vs, us, np.array(qr, dtype=np.float64))

    def simulate_force(self, force, duration_s, *, q0=None, qd0=None, seed=0, disturbance=False):
        """개루프: 주어진 힘만 인가(제어·ff·지연 없음). force: 스칼라/(n_q,)/콜러블 t -> (n_q,)."""
        n = self.n_q
        steps = _num_steps(duration_s)
        q0 = _vec_state(q0, n, "q0", np.zeros(n))
        qd0 = _vec_state(qd0, n, "qd0", np.zeros(n))
        if callable(force):
            f = np.empty((steps, n), dtype=np.float64)
            for k in range(steps):
                f[k] = np.broadcast_to(np.asarray(force(k * DT), dtype=np.float64), (n,))
        else:
            a = np.asarray(force, dtype=np.float64)
            if a.ndim > 1 or (a.ndim == 1 and a.shape != (n,)):
                raise ValueError(f"force 는 스칼라 또는 길이 {n} 벡터여야 한다")
            f = np.broadcast_to(a, (n,))[None, :].repeat(steps, axis=0)
        if not np.isfinite(f).all():
            raise ValueError("force 가 유한하지 않다")
        if disturbance:
            f = f + _disturbance(self.params, n, steps, seed)
        p = self.params
        qs, vs, us = [], [], []
        for j in range(n):
            a, b, c = _integrate_dof(float(p["mass"][j]), float(p["damping"][j]), float(p["quad_drag"][j]),
                                     float(p["stiffness"][j]), float(p["coulomb"][j]), float(p["w2"][j]),
                                     float(q0[j]), float(qd0[j]), steps, f[:, j].tolist(), None)
            qs.append(a)
            vs.append(b)
            us.append(c)
        return self._finish(qs, vs, us, None)


def make_rig(name, *, params=None, geom=None, home_c2w=None):
    """이름으로 리그 생성. 모르는 이름 KeyError, 잘못된 params / geom / home ValueError."""
    if name not in _SPECS:
        raise KeyError(name)
    spec = _SPECS[name]
    return Rig(name, spec, _build_params(name, spec["n"], params), _build_geom(spec, geom), _check_home(home_c2w))
