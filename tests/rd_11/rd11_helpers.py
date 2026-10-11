"""RD-11 테스트 공용 도우미: 파이프라인 §2.9.2 표를 그대로 옮긴 독립 오라클.
RD-11 구현을 오라클에 쓰지 않는다. 표의 값은 여기에 문자 그대로 적고, 구현의 상수와 대조한다."""
import math
from types import SimpleNamespace

import numpy as np

from buc.read import escher_score as E  # noqa: F401  (구현 전에는 여기서 실패한다: 정상)

# ------------------------------------------------------------------------------------------ 사양 표 (문자 그대로)
ORDER = ("vp", "multi_vp", "size", "plane", "light", "topo", "reconcile", "r_dis", "r_refl")
W = {"vp": 0.20, "multi_vp": 0.15, "size": 0.15, "plane": 0.05, "light": 0.15, "topo": 0.10,
     "reconcile": 0.05, "r_dis": 0.10, "r_refl": 0.05}
TAU = {"vp": 2.0, "multi_vp": 0.10, "size": 0.25, "plane": 0.05, "light": 15.0, "topo": 2.0,
       "reconcile": 4.0, "r_dis": 0.05, "r_refl": 4.0}

# 성분 -> EscherInputs 필드 (빛은 합성이라 따로)
FIELD = {"vp": "r_vp_deg", "multi_vp": "orphan_frac", "size": "size_sd", "plane": "plane_rel", "light": "light_sd_deg",
         "topo": "topo_count", "reconcile": "ffg_px1080", "r_dis": "r_dis", "r_refl": "r_refl_px1080"}


def inputs_at(frac, *, skip=(), **extra):
    """모든 성분 잔차를 τ_c * frac 로 둔 입력(빛은 SD만, 치트 몫 0). skip 에 든 성분은 None."""
    kw = {}
    for c in ORDER:
        if c in skip:
            continue
        kw[FIELD[c]] = TAU[c] * frac
    if "light" not in skip:
        kw["cheat_share"] = 0.0
    kw.update(extra)
    return E.EscherInputs(**kw)


def oracle_esc(r, refl_absent=None):
    """r: {성분: 잔차|None}. 계약서 규칙 1·2 그대로. 반환 (esc, {성분: contrib})."""
    if refl_absent is None:
        refl_absent = r.get("r_refl") is None
    w = dict(W)
    if refl_absent:
        w["vp"] += w["r_refl"]
        w["r_refl"] = 0.0
    contrib = {}
    for c in ORDER:
        if r.get(c) is None:
            contrib[c] = 0.0
        else:
            contrib[c] = w[c] * min(1.0, r[c] / TAU[c])
    return sum(contrib.values()), contrib


def csd(dirs, weights=None):
    """원형 표준편차 오라클(규칙 6)."""
    d = np.asarray(dirs, np.float64)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, np.float64)
    if len(d) < 2:
        return 0.0
    rr = np.linalg.norm((w[:, None] * d).sum(0)) / w.sum()
    if rr < 1e-12:
        return 180.0
    return min(180.0, math.degrees(math.sqrt(max(0.0, -2.0 * math.log(min(rr, 1.0))))))


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def rot_about(axis, deg, v):
    """Rodrigues 회전: v 를 axis 둘레로 deg 만큼."""
    k = unit(axis)
    t = math.radians(deg)
    v = np.asarray(v, np.float64)
    return v * math.cos(t) + np.cross(k, v) * math.sin(t) + k * np.dot(k, v) * (1 - math.cos(t))


def angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def fit_obj(status="used", name="window_residential", resid=0.0, cls=None):
    """RD-08 ObsFit 의 덕 타이핑 대역(이 단위가 읽는 필드만)."""
    return SimpleNamespace(status=status, name=name, cls=cls or (name or "x"), log_size_resid=resid)


def mad_sd(x):
    x = np.asarray(x, np.float64)
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


# ------------------------------------------------------------------------------------------ 평면 장면
H = W_ = 64
_YY, _XX = np.mgrid[0:H, 0:W_].astype(np.float64)
COL = _XX - (W_ - 1) / 2
ROW = _YY - (H - 1) / 2


def plane_log_depth(a=0.004, b=-0.003, c=0.5):
    """핀홀의 평면: 역깊이가 화소 좌표의 아핀. 반환 log z (H,W)."""
    w = c + a * COL + b * ROW
    assert w.min() > 0
    return -np.log(w)


def indep_plane_rel(ld, mask):
    """독립 오라클: 마스크 영역을 w=1/z 아핀 LS 로 맞춘 화소별 상대 오차 (규칙 5)."""
    z = np.exp(ld - ld.max())
    w = (1.0 / z)[mask]
    cols, rows = COL[mask], ROW[mask]
    A = np.stack([cols, rows, np.ones_like(cols)], 1)
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    what = A @ coef
    return np.abs(w - what) / np.maximum(np.abs(what), 1e-3 * np.median(w))


def ball(h=32, w=32, r=0.92):
    """화면을 꽉 채운 구의 카메라 쪽 반구 법선(단위, z<0)과 유효 원판."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (xx - (w - 1) / 2) / ((w - 1) / 2)
    y = (yy - (h - 1) / 2) / ((h - 1) / 2)
    rho2 = x * x + y * y
    z = -np.sqrt(np.maximum(1.0 - rho2, 1e-6))
    n = np.stack([x, y, z], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True), rho2 <= r * r
