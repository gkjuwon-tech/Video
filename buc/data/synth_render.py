"""DAT-01 절차 장면 GT 렌더 파이프 (G-buffer 12ch, 디노이저 OFF).

SceneProgram(= 라벨)을 카메라로 찍어 선형 HDR rgb, G-buffer 12채널, 깊이, sem, inst, 광류를 만든다.
셰이딩은 해석식(점광원 람베르트, 그림자·간접광·반사 없음)이며 어떤 화소 간 필터도 쓰지 않는다.
네임스페이스 패키지(buc/data 에 __init__.py 없음). numpy 와 buc 의 FND/INV 모듈만 쓴다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from buc.core import color, geom
from buc.gov import prov as _prov  # noqa: F401  (계보 모듈 의존 명시)
from buc.core.contracts import Prov
from buc.invent.grammar import core as G
from buc.invent.grammar.room import generate_room

__all__ = ["N_GBUF", "GBUF_CHANNELS", "SEM_GROUP_NAMES", "ALBEDO", "DEFAULT_MATERIAL_BY_CLASS", "sem_group",
           "SynthCamera", "SynthOpts", "SynthSample", "render_gbuffer", "default_camera", "translate_camera",
           "render_room"]

N_GBUF = 12
GBUF_CHANNELS = ("log_depth", "normal_x", "normal_y", "normal_z", "albedo_r", "albedo_g", "albedo_b",
                 "sem_structure", "sem_opening", "sem_furniture", "sem_accessory", "level_mask")
SEM_GROUP_NAMES = ("structure", "opening", "furniture", "accessory")
RENDERER = "buc.data.synth_render.v1"

# 선형 Rec.709 알베도 (각 성분 [0.02, 0.95])
ALBEDO: dict = {
    "wood": (0.30, 0.19, 0.10),
    "tile": (0.62, 0.60, 0.56),
    "carpet": (0.28, 0.24, 0.30),
    "concrete": (0.40, 0.40, 0.39),
    "plaster": (0.72, 0.69, 0.63),
    "brick": (0.42, 0.17, 0.12),
    "paint_white": (0.85, 0.85, 0.83),
    "wood_panel": (0.22, 0.13, 0.07),
    "marble": (0.80, 0.79, 0.76),
    "fabric": (0.30, 0.34, 0.46),
    "leather": (0.20, 0.10, 0.06),
    "metal": (0.55, 0.56, 0.58),
    "glass": (0.30, 0.38, 0.42),
    "stone": (0.46, 0.44, 0.40),
    "plastic": (0.60, 0.20, 0.18),
    "ceramic": (0.88, 0.87, 0.84),
    "door": (0.34, 0.22, 0.12),
    "window": (0.50, 0.62, 0.72),
}

DEFAULT_MATERIAL_BY_CLASS: dict = {
    "bed": "fabric", "sofa": "fabric", "table": "wood", "chair": "wood", "desk": "wood_panel",
    "shelf": "wood_panel", "cabinet": "wood_panel", "wardrobe": "wood_panel", "lamp": "metal",
    "plant": "ceramic", "tv": "plastic", "person": "fabric", "rug": "carpet", "picture": "paint_white",
    "stool": "wood", "counter": "stone",
}


def sem_group(sem) -> int:
    """sem id -> 그룹: VOID -1, 셸 0, 개구부 1, 가구 슬롯 2, 그 밖 슬롯 3. 모르는 id 는 ValueError."""
    try:
        s = int(sem)
    except (TypeError, ValueError) as e:
        raise ValueError(f"sem: 정수여야 한다 ({sem!r})") from e
    if s == G.SEM_VOID:
        return -1
    if s in (G.SEM_FLOOR, G.SEM_WALL, G.SEM_CEILING):
        return 0
    if s in (G.SEM_DOOR, G.SEM_WINDOW):
        return 1
    k = s - G.SEM_SLOT_BASE
    if 0 <= k < len(G.SLOT_CLASSES):
        return 2 if G.SLOT_CLASSES[k] in G.FURNITURE_CLASSES else 3
    raise ValueError(f"모르는 sem id: {s}")


# --------------------------------------------------------------------------- 자료형
def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


@dataclass(frozen=True, kw_only=True, eq=False)
class SynthCamera:
    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0

    def __post_init__(self):
        try:
            c2w = np.array(self.c2w, dtype=np.float64)
            K = np.array(self.K, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"c2w/K 를 수치 배열로 바꿀 수 없다: {e}") from e
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError("c2w: 유한한 (4,4) 여야 한다")
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("K: 유한한 (3,3) 여야 한다")
        if not geom.is_rotation(c2w[:3, :3]):
            raise ValueError("c2w: 회전 부분이 직교(det=+1)가 아니다")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K: fx, fy 는 양수여야 한다")
        for nm in ("width", "height"):
            v = getattr(self, nm)
            if not _is_int(v) or int(v) < 1:
                raise ValueError(f"{nm}: 1 이상의 정수여야 한다 ({v!r})")
        try:
            k1, k2 = float(self.k1), float(self.k2)
        except (TypeError, ValueError) as e:
            raise ValueError("k1, k2: 실수여야 한다") from e
        if not (math.isfinite(k1) and math.isfinite(k2)):
            raise ValueError("k1, k2: 유한해야 한다")
        object.__setattr__(self, "c2w", c2w)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "k2", k2)


@dataclass(frozen=True, kw_only=True)
class SynthOpts:
    spp: int = 1
    ambient: float = 0.05
    exposure: float = 0.05
    bg: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not _is_int(self.spp) or int(self.spp) not in (1, 4, 9, 16):
            raise ValueError(f"spp: 1, 4, 9, 16 중 하나 ({self.spp!r})")
        object.__setattr__(self, "spp", int(self.spp))
        for nm in ("ambient", "exposure"):
            try:
                v = float(getattr(self, nm))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{nm}: 실수여야 한다") from e
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{nm}: 유한하고 0 이상이어야 한다 ({v!r})")
            object.__setattr__(self, nm, v)
        try:
            bg = tuple(float(x) for x in self.bg)
        except (TypeError, ValueError) as e:
            raise ValueError("bg: 길이 3 의 실수열") from e
        if len(bg) != 3 or not all(math.isfinite(x) and x >= 0 for x in bg):
            raise ValueError("bg: 길이 3, 유한, 0 이상이어야 한다")
        object.__setattr__(self, "bg", bg)


@dataclass(frozen=True, kw_only=True, eq=False)
class SynthSample:
    rgb: np.ndarray
    gbuf: np.ndarray
    depth: np.ndarray
    sem: np.ndarray
    inst: np.ndarray
    flow: np.ndarray
    flow_valid: np.ndarray
    camera: SynthCamera
    flow_camera: SynthCamera | None
    prov: Prov
    meta: dict

    @property
    def log_depth(self):
        return self.gbuf[0]

    @property
    def normal(self):
        return self.gbuf[1:4]

    @property
    def albedo(self):
        return self.gbuf[4:7]

    @property
    def sem4(self):
        return self.gbuf[7:11]

    @property
    def level_mask(self):
        return self.gbuf[11]

    def rgb_display(self) -> np.ndarray:
        return color.linear_to_display(self.rgb)

    def to_arrays(self) -> dict:
        return {
            "rgb": self.rgb, "gbuf": self.gbuf, "depth": self.depth, "sem": self.sem, "inst": self.inst,
            "flow": self.flow, "flow_valid": self.flow_valid,
            "c2w": np.asarray(self.camera.c2w, np.float64), "K": np.asarray(self.camera.K, np.float64),
        }

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.to_arrays().items()):
            a = np.ascontiguousarray(v)
            h.update(k.encode())
            h.update(str(a.dtype).encode())
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
        h.update(json.dumps(self.meta, sort_keys=True, separators=(",", ":")).encode())
        return h.hexdigest()


# --------------------------------------------------------------------------- 렌더 핵심
def _trace(program, cam: SynthCamera, dx: float = 0.0, dy: float = 0.0):
    K = cam.K.copy()
    if dx or dy:
        K[0, 2] -= dx
        K[1, 2] -= dy
    rc = G.raycast(program, c2w=cam.c2w, K=K, width=cam.width, height=cam.height, k1=cam.k1, k2=cam.k2)
    Xc = geom.unproject(K, geom.pixel_grid(cam.height, cam.width), 1.0, cam.k1, cam.k2)
    D = Xc @ cam.c2w[:3, :3].T
    return rc, D


def _albedo_image(program, sem, inst):
    H, W = sem.shape
    out = np.zeros((H, W, 3), np.float64)
    matter = dict(program.matter)
    shell_mat = {G.SEM_FLOOR: matter.get("floor", "wood"), G.SEM_WALL: matter.get("wall", "paint_white"),
                 G.SEM_CEILING: matter.get("ceiling", "paint_white")}
    for sv, mat in shell_mat.items():
        out[sem == sv] = ALBEDO[mat]
    out[sem == G.SEM_DOOR] = ALBEDO["door"]
    out[sem == G.SEM_WINDOW] = ALBEDO["window"]
    for sl in program.slots:
        m = inst == sl.id
        if m.any():
            mat = sl.material if sl.material != "default" else DEFAULT_MATERIAL_BY_CLASS[sl.cls]
            out[m] = ALBEDO[mat]
    return out


def _shade(program, cam, rc, D, opts):
    """한 샘플 광선 묶음의 rgb (H,W,3) float64. 미스는 bg."""
    depth = rc.depth
    hit = np.isfinite(depth)
    H, W = depth.shape
    out = np.empty((H, W, 3), np.float64)
    out[:] = np.asarray(opts.bg, np.float64)
    if not hit.any():
        return out
    alb = _albedo_image(program, rc.sem, rc.inst)
    d64 = np.where(hit, depth, 0.0).astype(np.float64)
    P = cam.c2w[:3, 3] + d64[..., None] * D
    nw = rc.normal.astype(np.float64)
    E = np.zeros((H, W, 3), np.float64)
    for L in sorted(program.lights, key=lambda l: l.id):
        v = np.asarray(L.pos, np.float64) - P
        dist = np.sqrt(np.sum(v * v, axis=-1))
        d = np.maximum(dist, 0.5)
        # 동결 테스트 오라클과 동일: 코사인 분모도 하한이 적용된 d (|v|>=0.5 에서는 정확한 cos)
        c = np.maximum(0.0, np.sum(nw * v, axis=-1) / d)
        w = c * (1.0 / (4.0 * math.pi ** 2 * d ** 2))
        E = E + (np.asarray(L.rgb, np.float64) * float(L.power))[None, None, :] * w[..., None]
    shaded = alb * (opts.ambient + opts.exposure * E)
    out[hit] = shaded[hit]
    return out


def _sem4_image(sem, hit):
    H, W = sem.shape
    out = np.zeros((4, H, W), np.float32)
    for sv in np.unique(sem[hit]):
        g = sem_group(int(sv))
        if g < 0:
            continue
        out[g][hit & (sem == sv)] = 1.0
    return out


def render_gbuffer(program, camera, *, flow_to=None, opts=None) -> SynthSample:
    if not isinstance(program, G.SceneProgram):
        raise TypeError("program: SceneProgram 이어야 한다")
    if not isinstance(camera, SynthCamera):
        raise TypeError("camera: SynthCamera 이어야 한다")
    if opts is None:
        opts = SynthOpts()
    elif not isinstance(opts, SynthOpts):
        raise TypeError("opts: SynthOpts 이어야 한다")
    if flow_to is not None:
        if not isinstance(flow_to, SynthCamera):
            raise TypeError("flow_to: SynthCamera 또는 None")
        if (flow_to.width, flow_to.height) != (camera.width, camera.height):
            raise ValueError("flow_to: 영상 크기가 camera 와 같아야 한다")
    H, W = camera.height, camera.width

    rc, D = _trace(program, camera)
    depth = rc.depth
    hit = np.isfinite(depth)
    sem = np.where(hit, rc.sem, 0).astype(np.int16)
    inst = np.where(hit, rc.inst, -1).astype(np.int32)

    # --- G-buffer
    gbuf = np.zeros((N_GBUF, H, W), np.float32)
    d64 = np.where(hit, depth, 1.0).astype(np.float64)
    gbuf[0] = np.where(hit, np.log(d64), 0.0).astype(np.float32)
    nc = rc.normal.astype(np.float64) @ camera.c2w[:3, :3]
    nrm = np.linalg.norm(nc, axis=-1, keepdims=True)
    nc = nc / np.where(nrm > 0, nrm, 1.0)
    nc[~hit] = 0.0
    gbuf[1:4] = np.moveaxis(nc, -1, 0).astype(np.float32)
    alb = _albedo_image(program, sem, inst)
    alb[~hit] = 0.0
    gbuf[4:7] = np.moveaxis(alb, -1, 0).astype(np.float32)
    gbuf[7:11] = _sem4_image(sem, hit)
    gbuf[11] = hit.astype(np.float32)

    # --- rgb
    if opts.spp == 1:
        rgb = _shade(program, camera, rc, D, opts)
    else:
        n = int(round(math.sqrt(opts.spp)))
        offs = [(i + 0.5) / n - 0.5 for i in range(n)]
        acc = np.zeros((H, W, 3), np.float64)
        for oy in offs:
            for ox in offs:
                rcs, Ds = _trace(program, camera, ox, oy)
                acc = acc + _shade(program, camera, rcs, Ds, opts)
        rgb = acc / opts.spp
    rgb = rgb.astype(np.float32)

    # --- 광류
    flow = np.zeros((H, W, 2), np.float32)
    flow_valid = np.zeros((H, W), bool)
    if flow_to is not None and hit.any():
        Xc = geom.unproject(camera.K, geom.pixel_grid(H, W), d64, camera.k1, camera.k2)
        X = Xc @ camera.c2w[:3, :3].T + camera.c2w[:3, 3]
        Rb = flow_to.c2w[:3, :3].T
        tb = -Rb @ flow_to.c2w[:3, 3]
        with np.errstate(all="ignore"):
            uvb, zb = geom.project(flow_to.K, Rb, tb, X, flow_to.k1, flow_to.k2)
            f = uvb - geom.pixel_grid(H, W)
        flow_valid = hit & (zb > 1e-3) & np.isfinite(f).all(axis=-1)
        flow = np.where(flow_valid[..., None], f, 0.0).astype(np.float32)

    meta = {"renderer": RENDERER, "denoiser": "off", "spp": int(opts.spp),
            "program_hash": program.content_hash(), "seed": int(program.seed), "grammar": str(program.grammar)}
    return SynthSample(rgb=rgb, gbuf=gbuf, depth=depth.astype(np.float32), sem=sem, inst=inst, flow=flow,
                       flow_valid=flow_valid, camera=camera, flow_camera=flow_to, prov=Prov(), meta=meta)


# --------------------------------------------------------------------------- 카메라
def translate_camera(camera, d_cam) -> SynthCamera:
    if not isinstance(camera, SynthCamera):
        raise TypeError("camera: SynthCamera 이어야 한다")
    d = np.asarray(d_cam, np.float64)
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError("d_cam: 유한한 길이 3")
    c2w = camera.c2w.copy()
    c2w[:3, 3] = c2w[:3, 3] + c2w[:3, :3] @ d
    return SynthCamera(c2w=c2w, K=camera.K, width=camera.width, height=camera.height, k1=camera.k1, k2=camera.k2)


def _pip(pts, poly):
    x, y = pts[:, 0], pts[:, 1]
    ins = np.zeros(len(pts), bool)
    n = len(poly)
    with np.errstate(all="ignore"):
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            cond = (y0 > y) != (y1 > y)
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            ins ^= cond & (x < xint)
    return ins


def _dist_boundary(pts, poly):
    best = np.full(len(pts), np.inf)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(pts - proj, axis=1))
    return best


_LADDER = (0.5, 0.4, 0.3, 0.2)
_N_CAND = 16
_N_TRY = 400


def default_camera(program, seed: int, *, width: int = 256, height: int = 256, fov_x_deg: float = 75.0) -> SynthCamera:
    if not isinstance(program, G.SceneProgram):
        raise TypeError("program: SceneProgram 이어야 한다")
    if not _is_int(seed) or int(seed) < 0:
        raise ValueError("seed: 음이 아닌 정수여야 한다")
    if not _is_int(width) or not _is_int(height) or width < 1 or height < 1:
        raise ValueError("width, height: 1 이상의 정수")
    fov = float(fov_x_deg)
    if not (math.isfinite(fov) and 0.0 < fov < 180.0):
        raise ValueError("fov_x_deg: (0, 180) 안")
    K = geom.intrinsics_from_fov(fov, int(width), int(height))
    rng = np.random.default_rng([int(seed), int(program.content_hash()[:12], 16)])

    poly = np.asarray(program.shell.corners, np.float64)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    fps = [np.asarray(s.footprint(), np.float64) for s in program.slots if s.cls != "rug"]
    zmax = min(1.8, float(program.shell.height) - 0.5)
    zmin = 1.0
    zmax = max(zmax, zmin)
    Kp = geom.intrinsics_from_fov(fov, 32, 32)

    def clearance(pts):
        ok = _pip(pts, poly)
        clr = np.where(ok, _dist_boundary(pts, poly), -1.0)
        for fp in fps:
            inside = _pip(pts, fp)
            clr = np.minimum(clr, np.where(inside, -1.0, _dist_boundary(pts, fp)))
        return clr

    best = None
    best_ratio = -1.0
    for _ in range(_N_CAND):
        pts = lo + rng.random((_N_TRY, 2)) * (hi - lo)
        clr = clearance(pts)
        chosen = None
        for m in _LADDER:
            idx = np.flatnonzero(clr >= m)
            if len(idx):
                chosen = pts[idx[0]]
                break
        if chosen is None:
            chosen = pts[int(np.argmax(clr))]
        z = zmin + rng.random() * (zmax - zmin)
        yaw = rng.random() * 2.0 * math.pi
        pitch = math.radians((rng.random() * 2.0 - 1.0) * 12.0)
        f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, f
        c2w[:3, 3] = (chosen[0], chosen[1], z)
        rc = G.raycast(program, c2w=c2w, K=Kp, width=32, height=32)
        ratio = float((rc.inst >= 0).mean())
        if ratio > best_ratio:
            best, best_ratio = c2w, ratio
        if ratio >= 0.05:
            best = c2w
            break
    return SynthCamera(c2w=best, K=K, width=int(width), height=int(height))


def render_room(seed, *, width=256, height=256, fov_x_deg=75.0, room_cfg=None, opts=None, flow_dx_m=0.05) -> SynthSample:
    prog = generate_room(seed, room_cfg)
    cam = default_camera(prog, int(seed), width=width, height=height, fov_x_deg=fov_x_deg)
    flow_cam = None if flow_dx_m is None else translate_camera(cam, (float(flow_dx_m), 0.0, 0.0))
    return render_gbuffer(prog, cam, flow_to=flow_cam, opts=opts)
