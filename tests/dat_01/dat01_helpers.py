"""DAT-01 테스트 공용 도우미: 손으로 만든 상자 방, 카메라, 구현과 독립인 해석 오라클.

오라클은 FND-06 geom(검증된 모듈)과 numpy 만 쓰고 buc.data.synth_render 구현은 참조하지 않는다.
세계: 오른손, z 위, 바닥 z=0, 미터. 상자 방 = [0,4]x[0,4]x[0,3]. 변 0: y=0, 변 1: x=4, 변 2: y=4, 변 3: x=0.
"""
import math

import numpy as np

from buc.core import geom
from buc.invent.grammar import core as G

BOX = (4.0, 4.0, 3.0)
FACE_SEM = {  # (축, 쪽) -> 의미 id. 쪽 0 = 낮은 평면, 1 = 높은 평면
    (0, 0): G.SEM_WALL, (0, 1): G.SEM_WALL, (1, 0): G.SEM_WALL, (1, 1): G.SEM_WALL,
    (2, 0): G.SEM_FLOOR, (2, 1): G.SEM_CEILING,
}
FACE_NORMAL_W = {  # 안쪽을 향하는 세계 법선
    (0, 0): (1.0, 0.0, 0.0), (0, 1): (-1.0, 0.0, 0.0),
    (1, 0): (0.0, 1.0, 0.0), (1, 1): (0.0, -1.0, 0.0),
    (2, 0): (0.0, 0.0, 1.0), (2, 1): (0.0, 0.0, -1.0),
}
WALL_EDGE_OF_FACE = {(1, 0): 0, (0, 1): 1, (1, 1): 2, (0, 0): 3}


# --------------------------------------------------------------------------- 장면
def box_program(*, walls=(True, True, True, True), slots=(), lights=(), openings=(), matter=None, seed=0):
    shell = G.Shell(corners=[[0, 0], [4, 0], [4, 4], [0, 4]], height=BOX[2], walls_present=walls)
    return G.SceneProgram(grammar="test", seed=seed, shell=shell, openings=tuple(openings), slots=tuple(slots),
                          lights=tuple(lights), matter=matter or {})


def lamp(i, pos, rgb=(1.0, 0.9, 0.8), power=500.0, kind="lamp"):
    return G.Light(id=i, kind=kind, pos=pos, rgb=rgb, power=power, source=None)


def two_lights():
    return (lamp(0, (1.0, 1.5, 2.5), (1.0, 0.8, 0.6), 500.0), lamp(1, (3.0, 3.0, 1.0), (0.5, 0.6, 1.0), 300.0, "window"))


def furnished_program(lights=None):
    """카메라 앞(y=2.75..3.25)에 가구 둘, 뒷벽(y=4)에 문. 재질이 서로 다르다."""
    slots = (
        G.Slot(id=3, cls="table", center=(2.0, 3.0, 0.5), size=(1.0, 0.5, 1.0), material="marble"),
        G.Slot(id=5, cls="plant", center=(0.9, 3.0, 0.4), size=(0.4, 0.4, 0.8)),          # material=default
        G.Slot(id=8, cls="bed", center=(3.2, 3.0, 0.3), size=(0.8, 0.8, 0.6), material="leather"),
    )
    door = G.Opening(kind="door", edge=2, s0=1.5, s1=2.5, z0=0.0, z1=2.0)  # 변 2 = (4,4)->(0,4): x 2.5..1.5
    window = G.Opening(kind="window", edge=2, s0=0.3, s1=0.9, z0=1.0, z1=2.0)  # x 3.7..3.1 (벽 위)
    return box_program(slots=slots, openings=(door, window), lights=two_lights() if lights is None else lights,
                       matter={"floor": "tile", "wall": "brick", "ceiling": "plaster"})


# --------------------------------------------------------------------------- 카메라
def pose_look(eye, target):
    return geom.look_at(np.asarray(eye, float), np.asarray(target, float), up=np.array([0.0, 0.0, 1.0]))


def cam_center(w=64, h=48, fov=70.0, eye=(2.0, 2.0, 1.5), target=(2.0, 3.0, 1.5), k1=0.0, k2=0.0):
    from buc.data import synth_render as S
    return S.SynthCamera(c2w=pose_look(eye, target), K=geom.intrinsics_from_fov(fov, w, h), width=w, height=h,
                         k1=k1, k2=k2)


def shifted(cam, dx_cam):
    """카메라 자신의 x(오른쪽) 방향으로 평행이동. 오라클 쪽 정의."""
    from buc.data import synth_render as S
    c2w = cam.c2w.copy()
    c2w[:3, 3] = c2w[:3, 3] + c2w[:3, :3] @ np.asarray(dx_cam, float)
    return S.SynthCamera(c2w=c2w, K=cam.K, width=cam.width, height=cam.height, k1=cam.k1, k2=cam.k2)


def rotated_yaw(cam, deg):
    """세계 z축 둘레로 카메라 회전(위치 고정)."""
    from buc.data import synth_render as S
    a = math.radians(deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    c2w = cam.c2w.copy()
    c2w[:3, :3] = Rz @ c2w[:3, :3]
    return S.SynthCamera(c2w=c2w, K=cam.K, width=cam.width, height=cam.height, k1=cam.k1, k2=cam.k2)


def ray_dirs(cam):
    """세계 광선 방향 (H,W,3), 카메라 z=1 정규화(그래서 매개변수 t = z-깊이)."""
    Xc = geom.unproject(cam.K, geom.pixel_grid(cam.height, cam.width), 1.0, cam.k1, cam.k2)
    return Xc @ cam.c2w[:3, :3].T


# --------------------------------------------------------------------------- 상자 오라클
def box_oracle(cam, walls=(True, True, True, True)):
    """상자 안쪽 카메라의 첫 교차: depth(H,W) f64(미스 inf), face_axis, face_side, P(H,W,3)."""
    D = ray_dirs(cam)
    o = cam.c2w[:3, 3]
    H, W = cam.height, cam.width
    best = np.full((H, W), np.inf)
    ax_best = np.full((H, W), -1)
    side_best = np.full((H, W), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for a in range(3):
            d = D[..., a]
            side = (d > 0).astype(int)
            target = np.where(side == 1, BOX[a], 0.0)
            t = (target - o[a]) / d
            t = np.where(d == 0, np.inf, t)
            better = t < best
            best = np.where(better, t, best)
            ax_best = np.where(better, a, ax_best)
            side_best = np.where(better, side, side_best)
    P = o + best[..., None] * D
    removed = np.zeros((H, W), bool)
    for (a, s), e in WALL_EDGE_OF_FACE.items():
        if not walls[e]:
            removed |= (ax_best == a) & (side_best == s)
    depth = np.where(removed, np.inf, best)
    return depth, ax_best, side_best, P


def face_arrays(ax, side):
    sem = np.zeros(ax.shape, np.int16)
    nw = np.zeros(ax.shape + (3,))
    for (a, s), v in FACE_SEM.items():
        m = (ax == a) & (side == s)
        sem[m] = v
        nw[m] = FACE_NORMAL_W[(a, s)]
    return sem, nw


def normal_cam(nw, cam):
    return nw @ cam.c2w[:3, :3]  # R^T n  (행벡터 형태)


# --------------------------------------------------------------------------- 셰이딩·광류 오라클
def shade_oracle(P, nw, albedo, lights, ambient, exposure):
    """rgb = albedo * (ambient + exposure * sum_i rgb_i power_i max(0, n.w_i) / (4 pi^2 d_i^2))."""
    E = np.zeros(P.shape)
    for L in sorted(lights, key=lambda l: l.id):
        v = np.asarray(L.pos, float) - P
        d = np.maximum(np.linalg.norm(v, axis=-1), 0.5)
        c = np.maximum(0.0, np.sum(nw * v, axis=-1) / d)
        E = E + (np.asarray(L.rgb) * L.power)[None, None, :] * (c / (4.0 * math.pi ** 2 * d ** 2))[..., None]
    return albedo * (ambient + exposure * E)


def flow_oracle(depth, cam, cam_b):
    """depth(H,W) f64 (유한), A 의 각 화소 중심에서 B 로의 광류 (H,W,2) 와 B 의 z (H,W)."""
    uv = geom.pixel_grid(cam.height, cam.width)
    Xc = geom.unproject(cam.K, uv, depth, cam.k1, cam.k2)
    X = Xc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]
    Rb = cam_b.c2w[:3, :3].T
    tb = -Rb @ cam_b.c2w[:3, 3]
    uvb, zb = geom.project(cam_b.K, Rb, tb, X, cam_b.k1, cam_b.k2)
    return uvb - uv, zb


# --------------------------------------------------------------------------- 기하 도우미 (방 테스트용)
def point_in_poly(p, poly):
    x, y = p
    ins = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            ins = not ins
    return ins


def dist_to_boundary(p, poly):
    p = np.asarray(p, float)
    best = np.inf
    n = len(poly)
    for i in range(n):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % n], float)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best
