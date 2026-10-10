"""INV-01 테스트 공용 도우미: 손으로 만든 장면, 카메라, 그리고 구현과 독립인 느린 오라클 레이캐스터.

오라클은 FND-06 geom(검증된 모듈)과 numpy만 쓰고 buc.invent 구현은 참조하지 않는다.
세계: 오른손, z 위, 바닥 z=0, 미터. 슬롯 로컬 +x가 정면(front), yaw는 z축 CCW 회전."""
import math

import numpy as np

from buc.core import geom

INF = float("inf")


# --------------------------------------------------------------------------- 카메라
def make_K(w, h, fov=70.0):
    return geom.intrinsics_from_fov(fov, w, h)


def look(eye, target, up=(0.0, 0.0, 1.0)):
    return geom.look_at(np.asarray(eye, float), np.asarray(target, float), up=np.asarray(up, float))


def w2c_Rt(c2w):
    R = np.asarray(c2w, float)[:3, :3].T
    t = -R @ np.asarray(c2w, float)[:3, 3]
    return R, t


# --------------------------------------------------------------------------- 기하 도우미 (테스트 쪽 정의)
def footprint(center, size, yaw):
    """슬롯 바닥 사각형 4코너 (4,2). 로컬 x=정면, yaw CCW."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2.0, size[1] / 2.0
    out = []
    for lx, ly in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
        out.append((center[0] + c * lx - s * ly, center[1] + s * lx + c * ly))
    return np.array(out)


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
    best = INF
    n = len(poly)
    for i in range(n):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % n], float)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def obb_overlap_depth(fa, fb):
    """두 볼록 사각형의 분리축 최소 겹침 깊이(m). <=0 이면 분리."""
    depth = INF
    for poly in (fa, fb):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            ax = np.array([-e[1], e[0]])
            ax = ax / np.linalg.norm(ax)
            pa, pb = fa @ ax, fb @ ax
            ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            depth = min(depth, ov)
    return depth


def free_point(prog, clearance=0.4, step=0.25):
    """다각형 안쪽, 벽과 슬롯에서 clearance 이상 떨어진 점 중 무게중심에 가장 가까운 점."""
    poly = np.asarray(prog.shell.corners, float)
    cen = poly.mean(0)
    best = None
    xs = np.arange(poly[:, 0].min(), poly[:, 0].max() + 1e-9, step)
    ys = np.arange(poly[:, 1].min(), poly[:, 1].max() + 1e-9, step)
    for x in xs:
        for y in ys:
            p = (x, y)
            if not point_in_poly(p, poly) or dist_to_boundary(p, poly) < clearance:
                continue
            ok = True
            for s in prog.slots:
                if s.cls in ("rug", "lamp"):
                    continue
                fp = footprint(s.center, s.size, s.yaw)
                if point_in_poly(p, fp) or dist_to_boundary(p, fp) < clearance:
                    ok = False
                    break
            if ok:
                d = (x - cen[0]) ** 2 + (y - cen[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, (x, y))
    return best[1] if best else None


# --------------------------------------------------------------------------- 오라클 레이캐스터
def _mt(o, d, p0, p1, p2):
    """Moller-Trumbore. 교차 t 또는 None (양면)."""
    e1, e2 = p1 - p0, p2 - p0
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    if abs(a) < 1e-14:
        return None
    f = 1.0 / a
    s = o - p0
    u = f * float(np.dot(s, h))
    if u < -1e-9 or u > 1 + 1e-9:
        return None
    q = np.cross(s, e1)
    v = f * float(np.dot(d, q))
    if v < -1e-9 or u + v > 1 + 1e-9:
        return None
    t = f * float(np.dot(e2, q))
    return t


def oracle_trace(prog, o, d):
    """한 광선의 첫 교차: (t, sem, inst, normal). 상수는 계약서 4절과 같다."""
    from buc.invent.grammar import core as G  # 상수만 가져온다
    sh = prog.shell
    poly = np.asarray(sh.corners, float)
    H = sh.height
    best = [INF, 0, -1, np.zeros(3)]

    def cand(t, sem, inst, n):
        if t is not None and 1e-9 < t < best[0]:
            n = np.asarray(n, float)
            if np.dot(n, d) > 0:
                n = -n
            best[:] = [t, sem, inst, n]

    for z, flag, sem in ((0.0, sh.floor, G.SEM_FLOOR), (H, sh.ceiling, G.SEM_CEILING)):
        if flag and abs(d[2]) > 1e-12:
            t = (z - o[2]) / d[2]
            p = o + t * d
            if t > 1e-9 and point_in_poly((p[0], p[1]), poly):
                cand(t, sem, -1, (0, 0, 1))
    M = len(poly)
    for i in range(M):
        if not sh.walls_present[i]:
            continue
        A, B = poly[i], poly[(i + 1) % M]
        L = np.linalg.norm(B - A)
        u = (B - A) / L
        P0, P1 = np.array([A[0], A[1], 0.0]), np.array([B[0], B[1], 0.0])
        P2, P3 = np.array([B[0], B[1], H]), np.array([A[0], A[1], H])
        ts = [x for x in (_mt(o, d, P0, P1, P2), _mt(o, d, P0, P2, P3)) if x is not None]
        if not ts:
            continue
        t = min(ts)
        p = o + t * d
        s = (p[0] - A[0]) * u[0] + (p[1] - A[1]) * u[1]
        sem = G.SEM_WALL
        for op in prog.openings:
            if op.edge == i and op.s0 <= s <= op.s1 and op.z0 <= p[2] <= op.z1:
                sem = G.SEM_DOOR if op.kind == "door" else G.SEM_WINDOW
                break
        cand(t, sem, -1, (-u[1], u[0], 0.0))
    for sl in prog.slots:
        c, sn = math.cos(sl.yaw), math.sin(sl.yaw)
        Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        ol = Rz.T @ (o - np.asarray(sl.center, float))
        dl = Rz.T @ d
        h = np.asarray(sl.size, float) / 2.0
        te, tx, axis = -INF, INF, -1
        ok = True
        for k in range(3):
            if abs(dl[k]) < 1e-14:
                if abs(ol[k]) > h[k]:
                    ok = False
                    break
                continue
            t0, t1 = (-h[k] - ol[k]) / dl[k], (h[k] - ol[k]) / dl[k]
            lo, hi = min(t0, t1), max(t0, t1)
            if lo > te:
                te, axis = lo, k
            tx = min(tx, hi)
        if not ok or axis < 0 or te > tx or te <= 1e-9:
            continue
        nl = np.zeros(3)
        nl[axis] = -np.sign(dl[axis])
        cand(te, G.sem_id(sl.cls), sl.id, Rz @ nl)
    return best


def oracle_raycast(prog, c2w, K, W, H, k1=0.0, k2=0.0):
    uv = geom.pixel_grid(H, W)
    Xc = geom.unproject(K, uv, 1.0, k1, k2)
    R = np.asarray(c2w, float)[:3, :3]
    o = np.asarray(c2w, float)[:3, 3]
    depth = np.full((H, W), INF)
    sem = np.zeros((H, W), np.int16)
    inst = np.full((H, W), -1, np.int32)
    nrm = np.zeros((H, W, 3))
    for v in range(H):
        for u in range(W):
            d = R @ Xc[v, u]
            t, s_, i_, n_ = oracle_trace(prog, o, d)
            depth[v, u], sem[v, u], inst[v, u], nrm[v, u] = t, s_, i_, n_
    return depth, sem, inst, nrm


# --------------------------------------------------------------------------- 손으로 만든 장면
def base_program():
    """5 x 4 x 2.6 m 직사각 방. 모든 에너지 항이 0이어야 하는 기준 장면.
    변 0: y=0 (0,0)->(5,0), 변 1: x=5, 변 2: y=4 (5,4)->(0,4), 변 3: x=0. 안쪽 = 진행 방향 왼쪽."""
    from buc.invent.grammar import core as G
    shell = G.Shell(corners=[[0, 0], [5, 0], [5, 4], [0, 4]], height=2.6)
    openings = (
        G.Opening(kind="door", edge=0, s0=3.5, s1=4.4, z0=0.0, z1=2.1),       # x 3.5..4.4, 문 앞 띠 y 0..1
        G.Opening(kind="window", edge=2, s0=1.5, s1=2.7, z0=0.9, z1=2.1),
    )
    slots = (
        G.Slot(id=0, cls="bed", center=(1.1, 3.0, 0.25), size=(2.0, 1.6, 0.5), yaw=0.0),
        G.Slot(id=1, cls="cabinet", center=(4.7, 2.0, 0.45), size=(0.5, 1.2, 0.9), yaw=0.0),
        G.Slot(id=2, cls="tv", center=(4.7, 2.0, 1.2), size=(0.1, 0.8, 0.6), yaw=0.0, support="slot", support_id=1),
        G.Slot(id=3, cls="lamp", center=(2.5, 2.0, 2.55), size=(0.4, 0.4, 0.1), yaw=0.0, support="ceiling"),
        G.Slot(id=4, cls="picture", center=(0.02, 2.0, 1.6), size=(0.04, 0.8, 0.6), yaw=0.0, support="wall"),
    )
    lights = (
        G.Light(id=0, kind="lamp", pos=(2.5, 2.0, 2.45), rgb=(1.0, 0.85, 0.7), power=800.0, source=("slot", 3)),
        G.Light(id=1, kind="window", pos=(2.1, 3.7, 1.5), rgb=(0.9, 0.95, 1.0), power=3000.0, source=("opening", 1)),
    )
    return G.SceneProgram(grammar="room", seed=0, shell=shell, openings=openings, slots=slots, lights=lights,
                          matter={"floor": "wood", "wall": "plaster", "ceiling": "paint_white"}, meta={})


def box_room(slots=(), openings=(), walls_present=None, floor=True, ceiling=True):
    """4 x 6 x 2.8 m, 중심 원점: 모서리 (-2,-3),(2,-3),(2,3),(-2,3). 변 2 = y=3 (s는 x=2에서 -x 방향)."""
    from buc.invent.grammar import core as G
    shell = G.Shell(corners=[[-2, -3], [2, -3], [2, 3], [-2, 3]], height=2.8, walls_present=walls_present,
                    floor=floor, ceiling=ceiling)
    return G.SceneProgram(grammar="room", seed=0, shell=shell, openings=tuple(openings), slots=tuple(slots))
