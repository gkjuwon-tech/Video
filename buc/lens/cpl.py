"""LNS-07: CPL 편광 필터. Rayleigh 하늘 편광 · 필터 축 · Fresnel 반사 편광 v0.

PARADIGM §7.6: 렌즈 앞 원편광(CPL) 필터를 장면 선형 영상에 곱하는 한 줄 수식.
학습 파라미터 0, 순수 수식, 난수·시각·전역 상태 없음. torch·scipy 를 import 하지 않는다.

    g = (1 - s) + s·τ·(1 + DoP·cos2(φ - φ_f))

s = θ.cpl, φ_f = 필터 축, DoP = 화소의 선형 편광도, cos2φ = 2(E·축)² - 1 (E 는 화소의 편광 방향 단위벡터).
"""
from __future__ import annotations

import math

import numpy as np

from buc.core.contracts import Theta
from buc.lens.theta import EXTRA_DEFAULTS

P_MAX = 0.75  # 하늘 최대 편광도
TAU = 0.40  # 비편광광 투과율 (PARADIGM "T≈0.35–0.45")

_F64 = np.float64
_PARALLEL_EPS = 1e-9  # |ray × sun| 이 이보다 작으면 시선과 태양이 사실상 일치 -> dop=0, E=0
_NORM_TOL = 1e-6  # sun_cam 단위 노름 허용 오차
_ORTHO_TOL = 1e-8  # 회전행렬 허용 오차


# ------------------------------------------------------------------------------------------------ 검증 도우미
def _real_finite(v, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name}: 실수여야 한다 ({v!r})") from e
    if not math.isfinite(f):
        raise ValueError(f"{name}: 유한해야 한다 ({v!r})")
    return f


def _unit_interval(name: str, v) -> float:
    f = _real_finite(v, name)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"{name}: [0,1] 밖 ({f!r})")
    return f


def _tau(v) -> float:
    f = _real_finite(v, "tau")
    if not 0.0 < f <= 1.0:
        raise ValueError(f"tau: (0,1] 밖 ({f!r})")
    return f


def _p_max(v) -> float:
    f = _real_finite(v, "p_max")
    if not 0.0 < f <= 1.0:
        raise ValueError(f"p_max: (0,1] 밖 ({f!r})")
    return f


def _vec3(v, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=_F64)
    if a.shape != (3,):
        raise ValueError(f"{name}: (3,) 이어야 한다 (shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: 유한해야 한다")
    return a


def _check_rgb(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("rgb_lin: numpy 배열이어야 한다")
    if x.dtype != np.float32 and x.dtype != np.float64:
        raise TypeError(f"rgb_lin: float32/float64 만 받는다 ({x.dtype})")
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"rgb_lin: (H,W,3) 이어야 한다 (shape {x.shape})")
    if not np.all(np.isfinite(x)):
        raise ValueError("rgb_lin: 유한해야 한다")
    return x


def _check_rot(R) -> np.ndarray:
    m = np.asarray(R, dtype=_F64)
    if m.shape != (3, 3):
        raise ValueError(f"R_w2c: (3,3) 이어야 한다 (shape {m.shape})")
    if not np.all(np.isfinite(m)):
        raise ValueError("R_w2c: 유한해야 한다")
    if not np.allclose(m @ m.T, np.eye(3), atol=_ORTHO_TOL, rtol=0):
        raise ValueError("R_w2c: 직교행렬이 아니다")
    if abs(float(np.linalg.det(m)) - 1.0) > _ORTHO_TOL:
        raise ValueError("R_w2c: 회전행렬(det=+1)이 아니다")
    return m


def _check_K(K) -> np.ndarray:
    m = np.asarray(K, dtype=_F64)
    if m.shape != (3, 3):
        raise ValueError(f"K: (3,3) 이어야 한다 (shape {m.shape})")
    if not np.all(np.isfinite(m)):
        raise ValueError("K: 유한해야 한다")
    return m


def _check_mask(m, h: int, w: int) -> np.ndarray:
    a = np.asarray(m)
    if a.shape != (h, w):
        raise ValueError(f"sky_mask: ({h},{w}) 이어야 한다 (shape {a.shape})")
    if a.dtype == np.bool_:
        return a.astype(_F64)
    if a.dtype != np.float32 and a.dtype != np.float64:
        raise TypeError(f"sky_mask: bool 또는 float 이어야 한다 ({a.dtype})")
    b = a.astype(_F64)
    if not np.all(np.isfinite(b)) or b.min() < 0.0 or b.max() > 1.0:
        raise ValueError("sky_mask: 유한한 [0,1] 값이어야 한다")
    return b


def _frame(up, north) -> tuple[np.ndarray, np.ndarray]:
    """FarFieldSpec 과 같은 규약: up 정규화, north 는 up 에 수직으로 투영·정규화.
    north=None 이면 up 의 |성분| 이 가장 작은 축을 up 에 수직 투영한다."""
    U = _vec3(up, "up")
    nu = float(np.linalg.norm(U))
    if nu <= 0.0:
        raise ValueError("up: 영벡터")
    U = U / nu
    if north is None:
        N = np.zeros(3, dtype=_F64)
        N[int(np.argmin(np.abs(U)))] = 1.0
        raw_norm = 1.0
    else:
        N = _vec3(north, "north")
        raw_norm = float(np.linalg.norm(N))
    N = N - float(N @ U) * U
    nn = float(np.linalg.norm(N))
    if nn <= 1e-9 * max(1.0, raw_norm):
        raise ValueError("north: up 과 평행하거나 영벡터")
    return U, N / nn


# ------------------------------------------------------------------------------------------------ 편광도 수식
def _dop_from_cos(c: np.ndarray, p: float) -> np.ndarray:
    """p·sin²γ/(1+cos²γ), 입력은 cosγ."""
    c = np.clip(c, -1.0, 1.0)
    c2 = c * c
    return p * (1.0 - c2) / (1.0 + c2)


def rayleigh_dop(gamma_rad, p_max=P_MAX) -> np.ndarray:
    """단일 산란 Rayleigh 편광도 p_max·sin²γ/(1+cos²γ). γ: 시선과 태양 사이 각(rad)."""
    p = _p_max(p_max)
    g = np.asarray(gamma_rad, dtype=_F64)
    if not np.all(np.isfinite(g)):
        raise ValueError("gamma_rad: 유한해야 한다")
    s = np.sin(g)
    c = np.cos(g)
    return np.asarray(p * (s * s) / (1.0 + c * c), dtype=_F64)


def fresnel_dop(theta_i_rad, n=1.33) -> np.ndarray:
    """매끄러운 유전체 표면 반사광의 편광도 |Rs-Rp|/(Rs+Rp). 입사각 ∈ [0, π/2], 굴절률 n > 1."""
    ti = np.asarray(theta_i_rad, dtype=_F64)
    if not np.all(np.isfinite(ti)):
        raise ValueError("theta_i_rad: 유한해야 한다")
    if np.any(ti < 0.0) or np.any(ti > math.pi / 2):
        raise ValueError("theta_i_rad: [0, π/2] 밖")
    nn = _real_finite(n, "n")
    if not nn > 1.0:
        raise ValueError(f"n: 1 보다 커야 한다 ({nn!r})")
    ct = np.cos(ti)
    stt = np.sin(ti) / nn  # <= 1 : 입사각 ≤ π/2 이고 n > 1
    ctt = np.sqrt(np.maximum(1.0 - stt * stt, 0.0))  # cos θ_t
    # 공기(n1=1) -> 매질(n2=n) 진폭 반사율
    rs = ((ct - nn * ctt) / (ct + nn * ctt)) ** 2
    rp = ((nn * ct - ctt) / (nn * ct + ctt)) ** 2
    return np.asarray(np.abs(rs - rp) / (rs + rp), dtype=_F64)


# ------------------------------------------------------------------------------------------------ 기하
def sun_dir_world(elev_deg, azim_deg, *, up, north=None) -> np.ndarray:
    """세계 좌표 태양 방향 단위벡터. FarFieldSpec.sun_dir 와 같은 규약:
    cos e·(cos a·N + sin a·E) + sin e·U, E = N × U."""
    e = math.radians(_real_finite(elev_deg, "sun_elev_deg"))
    a = math.radians(_real_finite(azim_deg, "sun_azim_deg"))
    U, N = _frame(up, north)
    E = np.cross(N, U)
    return np.asarray(math.cos(e) * (math.cos(a) * N + math.sin(a) * E) + math.sin(e) * U, dtype=_F64)


def view_rays(K, h: int, w: int) -> np.ndarray:
    """화소 정수 중심 (u,v) 의 시선 단위벡터 (h,w,3), OpenCV 카메라 좌표 K⁻¹[u,v,1]."""
    Km = _check_K(K)
    hh, ww = int(h), int(w)
    if hh < 1 or ww < 1:
        raise ValueError("view_rays: h, w 는 1 이상")
    try:
        Ki = np.linalg.inv(Km)
    except np.linalg.LinAlgError as e:
        raise ValueError("K: 역행렬이 없다") from e
    u, v = np.meshgrid(np.arange(ww, dtype=_F64), np.arange(hh, dtype=_F64))
    p = np.stack([u, v, np.ones_like(u)], axis=-1)
    r = p @ Ki.T
    return np.asarray(r / np.linalg.norm(r, axis=-1, keepdims=True), dtype=_F64)


def sky_polarization(rays, sun_cam, *, p_max=P_MAX):
    """하늘 편광 장. rays (...,3) 시선 단위벡터, sun_cam (3,) 카메라 좌표 태양 단위벡터.
    반환 (dop (...), evec (...,3)): evec = normalize(ray × sun_cam) 은 산란면에 수직인 E벡터.
    ray ∥ sun 이면 dop=0, evec=0."""
    p = _p_max(p_max)
    R = np.asarray(rays, dtype=_F64)
    if R.ndim < 1 or R.shape[-1] != 3:
        raise ValueError(f"rays: (...,3) 이어야 한다 (shape {R.shape})")
    if not np.all(np.isfinite(R)):
        raise ValueError("rays: 유한해야 한다")
    S = _vec3(sun_cam, "sun_cam")
    if abs(float(np.linalg.norm(S)) - 1.0) > _NORM_TOL:
        raise ValueError("sun_cam: 단위벡터여야 한다")
    rn = np.linalg.norm(R, axis=-1, keepdims=True)
    if np.any(rn <= 0.0):
        raise ValueError("rays: 영벡터가 있다")
    Ru = R / rn
    c = Ru @ S
    dop = _dop_from_cos(c, p)
    cross = np.cross(Ru, S)
    cn = np.linalg.norm(cross, axis=-1)
    ok = cn > _PARALLEL_EPS
    safe = np.where(ok, cn, 1.0)
    evec = np.where(ok[..., None], cross / safe[..., None], 0.0)
    dop = np.where(ok, dop, 0.0)
    return np.asarray(dop, dtype=_F64), np.asarray(evec, dtype=_F64)


def polarizer_axis(angle_deg) -> np.ndarray:
    """필터 축 (cos a, sin a, 0). 영상 +x 에서 +y(아래)로 잰다."""
    a = math.radians(_real_finite(angle_deg, "angle_deg"))
    return np.array([math.cos(a), math.sin(a), 0.0], dtype=_F64)


# ------------------------------------------------------------------------------------------------ 이득
def polarizer_gain(dop, evec, angle_deg, *, strength=1.0, tau=TAU) -> np.ndarray:
    """화소 이득 (H,W): (1-s) + s·τ·(1 + dop·(2(evec·축)² - 1))."""
    d = np.asarray(dop, dtype=_F64)
    if not np.all(np.isfinite(d)):
        raise ValueError("dop: 유한해야 한다")
    if d.size and (d.min() < 0.0 or d.max() > 1.0):
        raise ValueError("dop: [0,1] 밖")
    e = np.asarray(evec, dtype=_F64)
    if e.shape != d.shape + (3,):
        raise ValueError(f"evec: dop.shape + (3,) 이어야 한다 ({e.shape} vs {d.shape})")
    if not np.all(np.isfinite(e)):
        raise ValueError("evec: 유한해야 한다")
    ax = polarizer_axis(angle_deg)
    s = _unit_interval("strength", strength)
    t = _tau(tau)
    proj = e @ ax
    cos2 = 2.0 * proj * proj - 1.0
    return np.asarray((1.0 - s) + s * t * (1.0 + d * cos2), dtype=_F64)


def apply_polarizer(rgb_lin, dop, evec, angle_deg, *, strength=1.0, tau=TAU) -> np.ndarray:
    """일반 진입점: 임의의 편광도·E벡터 장(수면·유리 반사 등)을 선형 영상에 곱한다."""
    x = _check_rgb(rgb_lin)
    g = polarizer_gain(dop, evec, angle_deg, strength=strength, tau=tau)
    if g.shape != x.shape[:2]:
        raise ValueError(f"dop/evec: 영상 크기 {x.shape[:2]} 와 맞아야 한다 (shape {g.shape})")
    if _unit_interval("strength", strength) == 0.0:
        return x.copy()
    return np.asarray(x.astype(_F64) * g[..., None], dtype=x.dtype)


def apply_cpl(rgb_lin, theta, *, R_w2c, up, north=None, K=None, sky_mask=None) -> np.ndarray:
    """하늘 편광 진입점. s = θ.cpl, 축 = θ.extra['cpl_angle_deg'], 태양 = θ.sun_elev/azim.
    태양이 없으면 dop=0 (균일 이득 (1-s)+sτ). sky_mask 는 dop 에 곱해진다(None = 전 화면 하늘)."""
    if not isinstance(theta, Theta):
        raise TypeError("theta: Theta 여야 한다")
    x = _check_rgb(rgb_lin)
    H, W = x.shape[:2]
    R = _check_rot(R_w2c)
    U, N = _frame(up, north)
    Kc = theta.intrinsics(W, H) if K is None else _check_K(K)
    mask = None if sky_mask is None else _check_mask(sky_mask, H, W)
    s = _unit_interval("cpl", theta.cpl)
    ang = _real_finite(theta.extra.get("cpl_angle_deg", EXTRA_DEFAULTS["cpl_angle_deg"]), "cpl_angle_deg")
    if s == 0.0:
        return x.copy()
    if theta.sun_elev_deg is None or theta.sun_azim_deg is None:
        dop = np.zeros((H, W), dtype=_F64)
        ev = np.zeros((H, W, 3), dtype=_F64)
    else:
        sun_cam = R @ sun_dir_world(theta.sun_elev_deg, theta.sun_azim_deg, up=up, north=north)
        dop, ev = sky_polarization(view_rays(Kc, H, W), sun_cam)
        if mask is not None:
            dop = dop * mask
    g = polarizer_gain(dop, ev, ang, strength=s)
    return np.asarray(x.astype(_F64) * g[..., None], dtype=x.dtype)


def compensation_ev(strength, tau=TAU) -> float:
    """필터가 어둡게 하는 만큼의 노출 보상(EV, >= 0). 비편광 하늘 이득 (1-s)+sτ 의 역수의 log2."""
    s = _unit_interval("strength", strength)
    t = _tau(tau)
    return 0.0 - math.log2((1.0 - s) + s * t)
