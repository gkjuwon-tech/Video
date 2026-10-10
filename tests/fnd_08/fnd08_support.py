"""FND-08 테스트 공용 도우미.

도구 모듈은 파일 경로로 적재한다(tools/는 패키지가 아니다). 이 파일은 계약서의 장면 표를 **독립적으로** 다시 구현한
해석적 렌더러(`expected`)를 담는다 - 도구의 코드를 쓰지 않으므로 정답 대조가 의미를 가진다.
"""
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "make_fixtures.py"
FIXDIR = ROOT / "tests" / "fixtures"

NAMES = ("plane_frontal", "plane_tilted", "ground_horizon", "sphere_lit", "box_pair", "corridor",
         "two_layer_card", "two_view_parallax", "distortion_dots", "color_chart")
W, H = 96, 64
LIGHT = np.array([-0.3, -0.7, -0.6]) / np.linalg.norm([-0.3, -0.7, -0.6])
SKY = np.array([0.25, 0.40, 0.75])
CHECK_A, CHECK_B, CHECK_M = np.array([0.8, 0.8, 0.8]), np.array([0.1, 0.1, 0.1]), 0.25
CHART_CODES = np.array([
    [0, 0, 0], [51, 51, 51], [102, 102, 102], [153, 153, 153], [204, 204, 204], [255, 255, 255],
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0],
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46]], dtype=np.uint8)


@lru_cache(maxsize=1)
def tool():
    spec = importlib.util.spec_from_file_location("make_fixtures_fnd08", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass가 모듈을 찾을 수 있게
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def built():
    return tool().build_all()


# ---------------------------------------------------------------------------------------------------------------
# 독립 카메라·광선
# ---------------------------------------------------------------------------------------------------------------
def rot_c2w(yaw_deg):
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def intr(fov_deg):
    fx = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[fx, 0.0, (W - 1) / 2.0], [0.0, fx, (H - 1) / 2.0], [0.0, 0.0, 1.0]])


# 이름 -> 뷰별 (fov, yaw, eye, k1, k2)
CAMS = {n: [(60.0, 0.0, (0, 0, 0), 0.0, 0.0)] for n in NAMES}
CAMS["ground_horizon"] = [(70.0, 0.0, (0, 0, 0), 0.0, 0.0)]
CAMS["corridor"] = [(70.0, 0.0, (0, 0, 0), 0.0, 0.0)]
CAMS["sphere_lit"] = [(50.0, 0.0, (0, 0, 0), 0.0, 0.0)]
CAMS["box_pair"] = [(60.0, -8.0, (0.4, -0.1, -0.5), 0.0, 0.0)]
CAMS["two_view_parallax"] = [(60.0, 0.0, (0, 0, 0), 0.0, 0.0), (60.0, -3.0, (0.3, 0.0, 0.0), 0.0, 0.0)]
CAMS["distortion_dots"] = [(60.0, 0.0, (0, 0, 0), -0.2, 0.05)]


def cam_pose(name, vi=0):
    """(K, R_w2c, t, center, k1, k2) - 계약서 표에서 직접 계산."""
    fov, yaw, eye, k1, k2 = CAMS[name][vi]
    Rc = rot_c2w(yaw)
    R = Rc.T
    eye = np.asarray(eye, float)
    return intr(fov), R, -R @ eye, eye, k1, k2


def rays(name, vi=0):
    """세계 광선 (원점 (3,), 방향 (H,W,3)); 방향의 카메라 z 성분 = 1."""
    from buc.core import geom
    K, R, t, eye, k1, k2 = cam_pose(name, vi)
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    if k1 or k2:
        xy = geom.undistort(np.stack([x, y], -1), k1, k2)
        x, y = xy[..., 0], xy[..., 1]
    dc = np.stack([x, y, np.ones_like(x)], -1)
    return eye, dc @ R, dc  # dc @ R == (R^T dc)^T


# ---------------------------------------------------------------------------------------------------------------
# 독립 해석 렌더러
# ---------------------------------------------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def plane(n, d, pid, rect=None, albedo=None):
    return dict(kind="plane", n=_unit(n), d=float(d), id=pid, rect=rect, albedo=albedo)


def prims_for(name):
    from buc.core import color
    if name == "plane_frontal":
        return [plane((0, 0, -1), -4.0, 1)]
    if name == "plane_tilted":
        n = _unit((0.3, -0.2, -1.0))
        return [plane(n, n @ np.array([0, 0, 4.0]), 1)]
    if name == "ground_horizon":
        return [plane((0, -1, 0), -1.5, 1)]
    if name == "sphere_lit":
        return [dict(kind="sphere", c=np.array([0.1, -0.05, 3.0]), r=0.9, id=1, albedo=np.array([0.7, 0.5, 0.3]))]
    if name == "box_pair":
        return [plane((0, 0, -1), -6.0, 1),
                dict(kind="box", lo=np.array([-0.9, -0.4, 2.5]), hi=np.array([-0.1, 0.6, 3.3]), id=2, albedo=np.array([0.2, 0.5, 0.8])),
                dict(kind="box", lo=np.array([0.3, 0.0, 3.5]), hi=np.array([1.3, 0.8, 4.5]), id=3, albedo=np.array([0.8, 0.6, 0.2]))]
    if name == "corridor":
        return [plane((0, -1, 0), -1.0, 1), plane((0, 1, 0), -1.0, 2), plane((1, 0, 0), -1.5, 3),
                plane((-1, 0, 0), -1.5, 4), plane((0, 0, -1), -10.0, 5)]
    if name == "two_layer_card":
        return [plane((0, 0, -1), -5.0, 1),
                plane((0, 0, -1), -2.0, 2, rect=(np.array([-0.6, -0.4, 2.0]), np.array([0.6, 0.4, 2.0])), albedo=np.array([0.8, 0.2, 0.2]))]
    if name == "two_view_parallax":
        return [plane((0, 0, -1), -6.0, 1),
                dict(kind="sphere", c=np.array([-0.2, 0.1, 3.0]), r=0.6, id=2, albedo=np.array([0.7, 0.5, 0.3]))]
    if name == "distortion_dots":
        ps = [plane((0, 0, -1), -3.0, 1, albedo=np.array([0.02, 0.02, 0.02]))]
        for k, c in enumerate(dots_world()):
            ps.append(dict(kind="disc", c=c, r=0.1, id=2 + k, albedo=np.ones(3)))
        return ps
    if name == "color_chart":
        lin = color.srgb8_to_linear(CHART_CODES).astype(np.float64)
        bg = color.srgb8_to_linear(np.array([40, 40, 40], np.uint8)).astype(np.float64)
        Z = 2.5
        ps = [plane((0, 0, -1), -Z, 25, albedo=bg)]
        for r in range(4):
            for c in range(6):
                lo = np.array([-1.2 + 0.4 * c, -0.8 + 0.4 * r, Z])
                ps.append(plane((0, 0, -1), -Z, 6 * r + c + 1, rect=(lo, lo + np.array([0.4, 0.4, 0.0])), albedo=lin[6 * r + c]))
        return ps
    raise KeyError(name)


def dots_world():
    return np.array([[0.4 * i, 0.4 * j, 3.0] for j in range(-2, 3) for i in range(-3, 4)])


EMISSIVE = {"plane_frontal", "distortion_dots", "color_chart"}


def expected(name, vi=0):
    """계약서 표에서 직접 계산한 정답: dict(depth, inst, nw (세계 법선), img (선형), hit)."""
    o, d, _ = rays(name, vi)
    best = np.full((H, W), np.inf)
    inst = np.zeros((H, W), np.int32)
    nw = np.zeros((H, W, 3))
    alb = np.zeros((H, W, 3))
    for p in prims_for(name):
        t = np.full((H, W), np.inf)
        n = np.zeros((H, W, 3))
        a = np.zeros((H, W, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            if p["kind"] in ("plane", "disc"):
                nn = p["n"] if p["kind"] == "plane" else np.array([0.0, 0.0, -1.0])
                dd = p["d"] if p["kind"] == "plane" else -p["c"][2]
                den = d @ nn
                tt = (dd - o @ nn) / den
                ok = (den < 0) & (tt > 0)
                P = o + tt[..., None] * d
                if p["kind"] == "disc":
                    ok &= ((P[..., 0] - p["c"][0]) ** 2 + (P[..., 1] - p["c"][1]) ** 2) <= p["r"] ** 2
                elif p["rect"] is not None:
                    lo, hi = p["rect"]
                    ok &= np.all((P >= lo - 1e-9) & (P <= hi + 1e-9), axis=-1)
                t = np.where(ok, tt, np.inf)
                n[:] = nn
                if p["albedo"] is not None:
                    a[:] = p["albedo"]
                else:
                    ax = [i for i in range(3) if i != int(np.argmax(np.abs(nn)))]
                    par = (np.floor(P[..., ax[0]] / CHECK_M) + np.floor(P[..., ax[1]] / CHECK_M)) % 2
                    a = np.where((par == 0)[..., None], CHECK_A, CHECK_B)
            elif p["kind"] == "sphere":
                oc = o - p["c"]
                qa = np.sum(d * d, -1)
                qb = d @ oc
                qc = oc @ oc - p["r"] ** 2
                disc = qb * qb - qa * qc
                tt = (-qb - np.sqrt(disc)) / qa
                ok = (disc > 0) & (tt > 0)
                t = np.where(ok, tt, np.inf)
                P = o + np.where(ok, tt, 0.0)[..., None] * d
                n = (P - p["c"]) / p["r"]
                a[:] = p["albedo"]
            elif p["kind"] == "box":
                t1 = (p["lo"] - o) / d
                t2 = (p["hi"] - o) / d
                near = np.minimum(t1, t2)
                far = np.maximum(t1, t2)
                te, tx = near.max(-1), far.min(-1)
                ok = (te < tx) & (te > 0)
                t = np.where(ok, te, np.inf)
                axis = np.argmax(near, -1)
                sg = -np.sign(np.take_along_axis(d, axis[..., None], -1)[..., 0])
                n = np.zeros((H, W, 3))
                np.put_along_axis(n, axis[..., None], sg[..., None], -1)
                a[:] = p["albedo"]
        better = (t <= best) & np.isfinite(t)
        best = np.where(better, t, best)
        inst = np.where(better, p["id"], inst).astype(np.int32)
        nw = np.where(better[..., None], n, nw)
        alb = np.where(better[..., None], a, alb)
    hit = np.isfinite(best)
    rad = alb if name in EMISSIVE else alb * (0.2 + 0.8 * np.maximum(0.0, nw @ LIGHT))[..., None]
    img = np.where(hit[..., None], rad, SKY)
    return dict(depth=best, inst=inst, nw=nw, img=img, hit=hit)
