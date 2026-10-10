#!/usr/bin/env python3
"""FND-08 골든 픽스처: 정답(깊이·법선·인스턴스·카메라·색)을 해석적으로 아는 소형 합성 장면 10종.

무작위 없이 화소 중심 광선을 해석적 교차로만 계산한다(전역 RNG 미사용, 같은 입력 -> 같은 바이트).
torch를 쓰지 않으며 numpy와 buc.core(geom·color·contracts)만 쓴다.

CLI:
  python3 tools/make_fixtures.py [--out DIR]   # 기본 tests/fixtures 에 생성
  python3 tools/make_fixtures.py --check [--out DIR]   # 쓰지 않고 일치 검사(일치 0, 불일치 1)
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # 스크립트로 직접 실행해도 buc 를 import 할 수 있게
    sys.path.insert(0, str(_ROOT))

from buc.core import color, contracts, geom  # noqa: E402

FIXTURE_VERSION = "1"
FIX_W, FIX_H = 96, 64
SCENE_NAMES = (
    "plane_frontal", "plane_tilted", "ground_horizon", "sphere_lit", "box_pair",
    "corridor", "two_layer_card", "two_view_parallax", "distortion_dots", "color_chart",
)
LIGHT_DIR = np.array([-0.3, -0.7, -0.6]) / np.linalg.norm([-0.3, -0.7, -0.6])
AMBIENT = 0.2
SKY_LIN = (0.25, 0.40, 0.75)
CHECK_A = (0.8, 0.8, 0.8)
CHECK_B = (0.1, 0.1, 0.1)
CHECK_M = 0.25

FIXDIR = _ROOT / "tests" / "fixtures"

_CHART_CODES = np.array([
    [0, 0, 0], [51, 51, 51], [102, 102, 102], [153, 153, 153], [204, 204, 204], [255, 255, 255],
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0],
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
], dtype=np.uint8)

_PROV_FACTORY = lambda: contracts.Prov(taint=frozenset({"ENGINE"}))  # noqa: E731


# ---------------------------------------------------------------------------------------------------------------
# 공개 데이터 타입
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True, eq=False)
class View:
    obs: contracts.Obs
    image_lin: np.ndarray  # (H,W,3) float32 선형
    depth: np.ndarray  # (H,W) float32 z-깊이, 하늘 +inf
    normals: np.ndarray  # (H,W,3) float32 카메라 좌표 단위 법선
    instance: np.ndarray  # (H,W) int32, 0 = 하늘


@dataclass(frozen=True, kw_only=True, eq=False)
class Scene:
    name: str
    views: tuple
    gt: dict

    def content_hash(self) -> str:
        return contracts.content_hash({
            "name": self.name,
            "gt": dict(self.gt),
            "views": [
                {"obs": v.obs.content_hash(), "image_lin": v.image_lin, "depth": v.depth,
                 "normals": v.normals, "instance": v.instance}
                for v in self.views
            ],
        })


# ---------------------------------------------------------------------------------------------------------------
# 카메라·광선
# ---------------------------------------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _rot_c2w(yaw_deg: float) -> np.ndarray:
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _camera(spec):
    """spec = (fov_x_deg, yaw_deg, eye, k1, k2) -> (K, R_w2c, t, eye)."""
    fov, yaw, eye, _k1, _k2 = spec
    R = np.ascontiguousarray(_rot_c2w(yaw).T)
    e = np.asarray(eye, dtype=np.float64)
    t = -(R @ e)
    K = geom.intrinsics_from_fov(float(fov), FIX_W, FIX_H)
    return K, R, t, e


def _ray_dirs(K, R, k1, k2) -> np.ndarray:
    """화소 중심 (u,v) 마다 세계 방향 (H,W,3). 카메라 z 성분은 정확히 1."""
    v, u = np.mgrid[0:FIX_H, 0:FIX_W].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    if k1 != 0.0 or k2 != 0.0:
        xy = geom.undistort(np.stack([x, y], axis=-1), k1, k2)
        x, y = xy[..., 0], xy[..., 1]
    dc = np.stack([x, y, np.ones_like(x)], axis=-1)
    return dc @ R  # == (R^T dc)^T


# ---------------------------------------------------------------------------------------------------------------
# 기본체와 교차
# ---------------------------------------------------------------------------------------------------------------
def _plane(n, d, pid, albedo=None, rect=None, disc=None) -> dict:
    return {"kind": "plane", "n": _unit(n), "d": float(d), "id": int(pid),
            "albedo": None if albedo is None else np.asarray(albedo, dtype=np.float64),
            "rect": rect, "disc": disc}


def _sphere(c, r, pid, albedo) -> dict:
    return {"kind": "sphere", "c": np.asarray(c, dtype=np.float64), "r": float(r), "id": int(pid),
            "albedo": np.asarray(albedo, dtype=np.float64)}


def _box(lo, hi, pid, albedo) -> dict:
    return {"kind": "box", "lo": np.asarray(lo, dtype=np.float64), "hi": np.asarray(hi, dtype=np.float64),
            "id": int(pid), "albedo": np.asarray(albedo, dtype=np.float64)}


def _checker(P: np.ndarray, n: np.ndarray) -> np.ndarray:
    ax = [i for i in range(3) if i != int(np.argmax(np.abs(n)))]
    with np.errstate(invalid="ignore"):
        par = (np.floor(P[..., ax[0]] / CHECK_M) + np.floor(P[..., ax[1]] / CHECK_M)) % 2
    return np.where((par == 0)[..., None], np.asarray(CHECK_A), np.asarray(CHECK_B))


def _hit_plane(p, o, D):
    n, d = p["n"], p["d"]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = D @ n
        t = (d - o @ n) / den
        P = o + t[..., None] * D
        ok = (den < 0) & (t > 0)
        if p["rect"] is not None:
            lo, hi = p["rect"]
            ok &= np.all((P >= lo - 1e-9) & (P <= hi + 1e-9), axis=-1)
        if p["disc"] is not None:
            cx, cy, r = p["disc"]
            ok &= (P[..., 0] - cx) ** 2 + (P[..., 1] - cy) ** 2 <= r * r
        t = np.where(ok, t, np.inf)
        if p["albedo"] is None:
            alb = _checker(P, n)
        else:
            alb = np.broadcast_to(p["albedo"], D.shape).copy()
    return t, np.broadcast_to(n, D.shape).copy(), alb


def _hit_sphere(p, o, D):
    c, r = p["c"], p["r"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        oc = o - c
        qa = np.sum(D * D, axis=-1)
        qb = D @ oc
        qc = oc @ oc - r * r
        disc = qb * qb - qa * qc
        tt = (-qb - np.sqrt(disc)) / qa
        ok = (disc > 0) & (tt > 0)
        P = o + np.where(ok, tt, 0.0)[..., None] * D
        n = (P - c) / r
    t = np.where(ok, tt, np.inf)
    return t, n, np.broadcast_to(p["albedo"], D.shape).copy()


def _hit_box(p, o, D):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (p["lo"] - o) / D
        t2 = (p["hi"] - o) / D
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)
        te = near.max(axis=-1)
        tx = far.min(axis=-1)
        ok = (te < tx) & (te > 0)
        axis = np.argmax(near, axis=-1)
        s = -np.sign(np.take_along_axis(D, axis[..., None], axis=-1)[..., 0])
    t = np.where(ok, te, np.inf)
    n = np.zeros(D.shape, dtype=np.float64)
    np.put_along_axis(n, axis[..., None], s[..., None], -1)
    return t, n, np.broadcast_to(p["albedo"], D.shape).copy()


_HIT = {"plane": _hit_plane, "sphere": _hit_sphere, "box": _hit_box}


def _trace(prims, o, D):
    """가장 가까운 양의 교점. 같은 깊이면 목록에서 뒤에 있는 기본체가 이긴다(패치·원판이 평면을 덮는다)."""
    shp = D.shape[:2]
    best = np.full(shp, np.inf)
    inst = np.zeros(shp, dtype=np.int32)
    nw = np.zeros(D.shape, dtype=np.float64)
    alb = np.zeros(D.shape, dtype=np.float64)
    for p in prims:
        t, n, a = _HIT[p["kind"]](p, o, D)
        better = (t <= best) & np.isfinite(t)
        best = np.where(better, t, best)
        inst = np.where(better, p["id"], inst).astype(np.int32)
        nw = np.where(better[..., None], n, nw)
        alb = np.where(better[..., None], a, alb)
    return best, inst, nw, alb


def _shade_rgb(nw, alb, hit, emissive: bool) -> np.ndarray:
    if emissive:
        rad = alb
    else:
        lam = nw[..., 0] * LIGHT_DIR[0] + nw[..., 1] * LIGHT_DIR[1] + nw[..., 2] * LIGHT_DIR[2]
        rad = alb * (AMBIENT + 0.8 * np.maximum(0.0, lam))[..., None]
    return np.where(hit[..., None], rad, np.asarray(SKY_LIN, dtype=np.float64))


def _render_view(prims, spec, emissive: bool) -> View:
    K, R, t, e = _camera(spec)
    k1, k2 = float(spec[3]), float(spec[4])
    obs = contracts.Obs(K=K, R=R, t=t, width=FIX_W, height=FIX_H, k1=k1, k2=k2)
    D = _ray_dirs(K, R, k1, k2)
    best, inst, nw, alb = _trace(prims, e, D)
    hit = np.isfinite(best)
    img = _shade_rgb(nw, alb, hit, emissive)
    depth = np.where(hit, best, np.inf)
    nc = nw @ R.T
    normals = np.where(hit[..., None], nc, np.array([0.0, 0.0, -1.0]))
    return View(
        obs=obs,
        image_lin=np.ascontiguousarray(img.astype(np.float32)),
        depth=np.ascontiguousarray(depth.astype(np.float32)),
        normals=np.ascontiguousarray(normals.astype(np.float32)),
        instance=np.ascontiguousarray(np.where(hit, inst, 0).astype(np.int32)),
    )


def _plane_gt(prims) -> dict:
    base = [p for p in prims if p["kind"] == "plane" and p["rect"] is None and p["disc"] is None]
    if not base:
        return {}
    return {
        "plane_n": np.array([p["n"] for p in base], dtype=np.float64).reshape(-1, 3),
        "plane_d": np.array([p["d"] for p in base], dtype=np.float64),
    }


def _scene(name, cams, prims, emissive, extra=None) -> Scene:
    views = tuple(_render_view(prims, c, emissive) for c in cams)
    gt = {"shading": "emissive" if emissive else "lit"}
    gt.update(_plane_gt(prims))
    if extra is not None:
        gt.update(extra(views, prims, cams))
    return Scene(name=name, views=views, gt=gt)


# ---------------------------------------------------------------------------------------------------------------
# 장면 10종
# ---------------------------------------------------------------------------------------------------------------
_C60 = (60.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0)


def _b_plane_frontal() -> Scene:
    return _scene("plane_frontal", [_C60], [_plane((0, 0, -1), -4.0, 1)], emissive=True)


def _b_plane_tilted() -> Scene:
    n = _unit((0.3, -0.2, -1.0))
    return _scene("plane_tilted", [_C60], [_plane(n, n @ np.array([0.0, 0.0, 4.0]), 1)], emissive=False)


def _b_ground_horizon() -> Scene:
    return _scene("ground_horizon", [(70.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0)],
                  [_plane((0, -1, 0), -1.5, 1)], emissive=False)


def _b_sphere_lit() -> Scene:
    c = np.array([0.1, -0.05, 3.0])
    prims = [_sphere(c, 0.9, 1, (0.7, 0.5, 0.3))]
    return _scene("sphere_lit", [(50.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0)], prims, emissive=False,
                  extra=lambda vs, ps, cs: {"sphere_c": c.copy(), "sphere_r": 0.9})


def _b_box_pair() -> Scene:
    lo1, hi1 = np.array([-0.9, -0.4, 2.5]), np.array([-0.1, 0.6, 3.3])
    lo2, hi2 = np.array([0.3, 0.0, 3.5]), np.array([1.3, 0.8, 4.5])
    prims = [_plane((0, 0, -1), -6.0, 1), _box(lo1, hi1, 2, (0.2, 0.5, 0.8)), _box(lo2, hi2, 3, (0.8, 0.6, 0.2))]
    return _scene("box_pair", [(60.0, -8.0, (0.4, -0.1, -0.5), 0.0, 0.0)], prims, emissive=False,
                  extra=lambda vs, ps, cs: {"box_min": np.stack([lo1, lo2]), "box_max": np.stack([hi1, hi2])})


def _b_corridor() -> Scene:
    prims = [
        _plane((0, -1, 0), -1.0, 1),   # 바닥 y=1
        _plane((0, 1, 0), -1.0, 2),    # 천장 y=-1
        _plane((1, 0, 0), -1.5, 3),    # 왼벽 x=-1.5
        _plane((-1, 0, 0), -1.5, 4),   # 오른벽 x=1.5
        _plane((0, 0, -1), -10.0, 5),  # 끝벽 z=10
    ]
    return _scene("corridor", [(70.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0)], prims, emissive=False)


def _b_two_layer_card() -> Scene:
    lo, hi = np.array([-0.6, -0.4, 2.0]), np.array([0.6, 0.4, 2.0])
    bg = _plane((0, 0, -1), -5.0, 1)
    card = _plane((0, 0, -1), -2.0, 2, albedo=(0.8, 0.2, 0.2), rect=(lo, hi))
    prims = [bg, card]

    def extra(views, ps, cams):
        v = views[0]
        K, R, t, e = _camera(cams[0])
        D = _ray_dirs(K, R, 0.0, 0.0)
        bbest, _, bnw, balb = _trace([bg], e, D)
        bhit = np.isfinite(bbest)
        bgimg = _shade_rgb(bnw, balb, bhit, False).astype(np.float32)
        bgdep = bbest.astype(np.float32)
        c = v.instance == 2
        rgba = np.zeros((2, FIX_H, FIX_W, 4), dtype=np.float32)
        rgba[0, ..., :3] = np.where(c[..., None], v.image_lin, np.float32(0.0))
        rgba[0, ..., 3] = c.astype(np.float32)
        rgba[1, ..., :3] = bgimg
        rgba[1, ..., 3] = 1.0
        dep = np.empty((2, FIX_H, FIX_W), dtype=np.float32)
        dep[0] = np.where(c, v.depth, bgdep)
        dep[1] = bgdep
        return {"card_min": lo.copy(), "card_max": hi.copy(),
                "ldi_rgba": np.ascontiguousarray(rgba), "ldi_depth": np.ascontiguousarray(dep)}

    return _scene("two_layer_card", [_C60], prims, emissive=False, extra=extra)


def _b_two_view_parallax() -> Scene:
    c = np.array([-0.2, 0.1, 3.0])
    prims = [_plane((0, 0, -1), -6.0, 1), _sphere(c, 0.6, 2, (0.7, 0.5, 0.3))]
    cams = [_C60, (60.0, -3.0, (0.3, 0.0, 0.0), 0.0, 0.0)]

    def extra(views, ps, cs):
        T = views[1].obs.w2c() @ views[0].obs.c2w()
        return {"T_01": np.ascontiguousarray(T.astype(np.float64)), "sphere_c": c.copy(), "sphere_r": 0.6}

    return _scene("two_view_parallax", cams, prims, emissive=False, extra=extra)


def _dots_world() -> np.ndarray:
    return np.array([[0.4 * i, 0.4 * j, 3.0] for j in range(-2, 3) for i in range(-3, 4)], dtype=np.float64)


def _b_distortion_dots() -> Scene:
    prims = [_plane((0, 0, -1), -3.0, 1, albedo=(0.02, 0.02, 0.02))]
    dots = _dots_world()
    for k, c in enumerate(dots):
        prims.append(_plane((0, 0, -1), -3.0, 2 + k, albedo=(1.0, 1.0, 1.0), disc=(c[0], c[1], 0.1)))

    def extra(views, ps, cs):
        o = views[0].obs
        uv, _z = o.project(dots)
        return {"dots_world": dots.copy(), "dots_uv": np.ascontiguousarray(uv.astype(np.float64)),
                "dot_radius": 0.1}

    return _scene("distortion_dots", [(60.0, 0.0, (0.0, 0.0, 0.0), -0.2, 0.05)], prims, emissive=True, extra=extra)


def _b_color_chart() -> Scene:
    Z = 2.5
    bg_lin = color.srgb8_to_linear(np.array([40, 40, 40], dtype=np.uint8)).astype(np.float64)
    lin = color.srgb8_to_linear(_CHART_CODES).astype(np.float64)
    prims = [_plane((0, 0, -1), -Z, 25, albedo=bg_lin)]
    rects = []
    for r in range(4):
        for c in range(6):
            k = 6 * r + c
            x0, x1 = -1.2 + 0.4 * c, -0.8 + 0.4 * c
            y0, y1 = -0.8 + 0.4 * r, -0.4 + 0.4 * r
            prims.append(_plane((0, 0, -1), -Z, k + 1, albedo=lin[k],
                                rect=(np.array([x0, y0, Z]), np.array([x1, y1, Z]))))
            rects.append((x0, x1, y0, y1))

    def extra(views, ps, cs):
        K = geom.intrinsics_from_fov(60.0, FIX_W, FIX_H)
        pr = np.array([[K[0, 0] * x0 / Z + K[0, 2], K[0, 0] * x1 / Z + K[0, 2],
                        K[1, 1] * y0 / Z + K[1, 2], K[1, 1] * y1 / Z + K[1, 2]] for (x0, x1, y0, y1) in rects],
                      dtype=np.float64)
        return {
            "patch_codes": _CHART_CODES.copy(),
            "patch_lin": color.srgb8_to_linear(_CHART_CODES),
            "patch_rects": pr,
            "chart_bg_code": 40,
        }

    return _scene("color_chart", [(60.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0)], prims, emissive=True, extra=extra)


_BUILDERS = {
    "plane_frontal": _b_plane_frontal,
    "plane_tilted": _b_plane_tilted,
    "ground_horizon": _b_ground_horizon,
    "sphere_lit": _b_sphere_lit,
    "box_pair": _b_box_pair,
    "corridor": _b_corridor,
    "two_layer_card": _b_two_layer_card,
    "two_view_parallax": _b_two_view_parallax,
    "distortion_dots": _b_distortion_dots,
    "color_chart": _b_color_chart,
}


def build_scene(name) -> Scene:
    if not isinstance(name, str) or name not in _BUILDERS:
        raise KeyError(name)
    return _BUILDERS[name]()


def build_all() -> dict:
    return {n: build_scene(n) for n in SCENE_NAMES}


# ---------------------------------------------------------------------------------------------------------------
# 이음매: InputImage / ReadPacket
# ---------------------------------------------------------------------------------------------------------------
def to_input_image(scene: Scene, view: int = 0) -> contracts.InputImage:
    v = scene.views[view]
    return contracts.InputImage(pixels=color.linear_to_srgb8(v.image_lin), color_space="srgb", prov=_PROV_FACTORY())


def to_read_packet(scene: Scene, view: int = 0) -> contracts.ReadPacket:
    v = scene.views[view]
    ii = to_input_image(scene, view)
    W, H = v.obs.width, v.obs.height
    fov = geom.fov_x_deg_from_K(v.obs.K, W)
    theta = contracts.Theta(aspect=W / H, focal_mm=geom.focal_from_fov(fov, 36.0), sensor_width_mm=36.0)
    return contracts.ReadPacket(
        image_lin=v.image_lin,
        depth=v.depth,
        obs=v.obs,
        theta_in=theta,
        input_hash=ii.h_I,
        prov=_PROV_FACTORY(),
        normals=v.normals,
    )


# ---------------------------------------------------------------------------------------------------------------
# 저장·적재 (결정론적 npz: 고정 날짜 zip 항목, 무손실)
# ---------------------------------------------------------------------------------------------------------------
def _gt_meta(key: str, val):
    if isinstance(val, np.ndarray):
        return {"t": "nd"}
    if isinstance(val, str):
        return {"t": "str", "v": val}
    if isinstance(val, (bool, np.bool_)):
        return {"t": "bool", "v": bool(val)}
    if isinstance(val, (int, np.integer)):
        return {"t": "int", "v": int(val)}
    if isinstance(val, (float, np.floating)):
        return {"t": "float", "v": float(val)}
    raise TypeError(f"gt[{key!r}]: 지원하지 않는 타입 {type(val).__name__}")


def _scene_arrays(sc: Scene) -> dict:
    arrays: dict = {}
    meta = {"name": sc.name, "version": FIXTURE_VERSION, "gt": {}, "views": []}
    for k in sorted(sc.gt):
        val = sc.gt[k]
        meta["gt"][k] = _gt_meta(k, val)
        if isinstance(val, np.ndarray):
            arrays[f"g_{k}"] = np.ascontiguousarray(val)
    for i, v in enumerate(sc.views):
        o = v.obs
        meta["views"].append({"width": o.width, "height": o.height, "k1": o.k1, "k2": o.k2,
                              "l_obs_hard": o.l_obs_hard, "l_obs_part": o.l_obs_part})
        arrays[f"v{i}_K"] = o.K
        arrays[f"v{i}_R"] = o.R
        arrays[f"v{i}_t"] = o.t
        arrays[f"v{i}_image_lin"] = v.image_lin
        arrays[f"v{i}_depth"] = v.depth
        arrays[f"v{i}_normals"] = v.normals
        arrays[f"v{i}_instance"] = v.instance
    arrays["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8).copy()
    return arrays


def _npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload.getvalue())
    return buf.getvalue()


def write_fixtures(out_dir) -> dict:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    scenes = build_all()
    man = {"version": FIXTURE_VERSION, "scenes": {}}
    for n in SCENE_NAMES:
        sc = scenes[n]
        (d / f"{n}.npz").write_bytes(_npz_bytes(_scene_arrays(sc)))
        man["scenes"][n] = {"file": f"{n}.npz", "hash": sc.content_hash()}
    (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return man


def load_scene(name, root=None) -> Scene:
    if not isinstance(name, str) or name not in SCENE_NAMES:
        raise KeyError(name)
    base = FIXDIR if root is None else Path(root)
    with np.load(base / f"{name}.npz", allow_pickle=False) as z:
        meta = json.loads(z["__meta__"].tobytes().decode("utf-8"))
        arr = {k: z[k] for k in z.files if k != "__meta__"}
    gt = {}
    for k, m in meta["gt"].items():
        if m["t"] == "nd":
            gt[k] = arr[f"g_{k}"]
        else:
            gt[k] = m["v"]
    views = []
    for i, vm in enumerate(meta["views"]):
        obs = contracts.Obs(K=arr[f"v{i}_K"], R=arr[f"v{i}_R"], t=arr[f"v{i}_t"], width=vm["width"],
                            height=vm["height"], k1=vm["k1"], k2=vm["k2"],
                            l_obs_hard=vm["l_obs_hard"], l_obs_part=vm["l_obs_part"])
        views.append(View(obs=obs, image_lin=arr[f"v{i}_image_lin"], depth=arr[f"v{i}_depth"],
                          normals=arr[f"v{i}_normals"], instance=arr[f"v{i}_instance"]))
    return Scene(name=meta["name"], views=tuple(views), gt=gt)


def check_fixtures(out_dir) -> bool:
    """재생성 해시와 manifest·파일이 모두 일치하면 True. 파일 없음·불일치는 False(예외를 내지 않는다)."""
    try:
        d = Path(out_dir)
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if man.get("version") != FIXTURE_VERSION:
            return False
        scenes = man.get("scenes", {})
        if set(scenes) != set(SCENE_NAMES):
            return False
        fresh = build_all()
        for n in SCENE_NAMES:
            ent = scenes[n]
            want = fresh[n].content_hash()
            if ent.get("file") != f"{n}.npz" or ent.get("hash") != want:
                return False
            if not (d / ent["file"]).is_file():
                return False
            if load_scene(n, d).content_hash() != want:
                return False
        return True
    except Exception:  # noqa: BLE001 - 검사기는 예외 대신 False
        return False


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FND-08 골든 픽스처 생성·검사")
    ap.add_argument("--out", default=None, help="출력 디렉터리(기본 tests/fixtures)")
    ap.add_argument("--check", action="store_true", help="쓰지 않고 일치만 검사")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else FIXDIR
    if a.check:
        ok = check_fixtures(out)
        print(json.dumps({"ok": ok, "dir": str(out)}))
        return 0 if ok else 1
    man = write_fixtures(out)
    print(json.dumps({"ok": True, "dir": str(out), "scenes": len(man["scenes"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
