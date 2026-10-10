"""INV-01 RoomGen v0: 시드 하나로 문법(G1-G13)을 모두 만족하는 방 장면 프로그램을 만든다.

난수는 전부 지역 np.random.Generator(시드 고정)에서 나온다. 전역 RNG·set 순회에 의존하지 않는다.
배치 원칙: 가구는 벽에 등을 대고 안쪽을 본다(front = 안쪽 법선), 문 앞 띠(폭 + 0.15 m 여유, 깊이 1.15 m)는 비운다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from buc.invent.grammar.core import (FURNITURE_CLASSES, Light, Opening, SceneProgram, Shell, Slot, _footprint,
                                     _pip, _pip_vec, _seg_dist)

__all__ = ["RoomGenConfig", "generate_room"]


def _pair(name, v, typ, lo_min, lo_name=None):
    try:
        a, b = v
    except (TypeError, ValueError):
        raise ValueError(f"{name}: (lo, hi) 쌍이어야 한다") from None
    for x in (a, b):
        if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not np.isfinite(x):
            raise ValueError(f"{name}: 유한한 숫자여야 한다")
    a, b = typ(a), typ(b)
    if a > b:
        raise ValueError(f"{name}: lo<=hi 여야 한다 ({a}, {b})")
    if a < lo_min:
        raise ValueError(f"{name}: lo>={lo_min} 여야 한다 ({a})")
    return (a, b)


@dataclass(frozen=True, kw_only=True)
class RoomGenConfig:
    area_range: Tuple[float, float] = (9.0, 40.0)
    aspect_range: Tuple[float, float] = (1.0, 2.2)
    height_range: Tuple[float, float] = (2.4, 3.2)
    l_shape_prob: float = 0.25
    furniture_range: Tuple[int, int] = (3, 9)
    accessory_range: Tuple[int, int] = (0, 6)
    n_doors_range: Tuple[int, int] = (1, 2)
    n_windows_range: Tuple[int, int] = (0, 4)
    people: int = 0

    def __post_init__(self):
        s = object.__setattr__
        s(self, "area_range", _pair("area_range", self.area_range, float, 9.0))
        s(self, "aspect_range", _pair("aspect_range", self.aspect_range, float, 1.0))
        s(self, "height_range", _pair("height_range", self.height_range, float, 2.4))
        s(self, "furniture_range", _pair("furniture_range", self.furniture_range, int, 0))
        s(self, "accessory_range", _pair("accessory_range", self.accessory_range, int, 0))
        s(self, "n_doors_range", _pair("n_doors_range", self.n_doors_range, int, 1))
        s(self, "n_windows_range", _pair("n_windows_range", self.n_windows_range, int, 0))
        p = self.l_shape_prob
        if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)) or not (0.0 <= p <= 1.0):
            raise ValueError("l_shape_prob 는 [0,1] 의 숫자여야 한다")
        s(self, "l_shape_prob", float(p))
        n = self.people
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("people 는 0 이상의 정수여야 한다")
        s(self, "people", int(n))


# ------------------------------------------------------------------------------------------ 기하 도우미
def _sep(fa, fb) -> float:
    """분리축 최대 간격(m). >=0 이면 분리, 음수면 관통 깊이."""
    best = -np.inf
    for poly in (fa, fb):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            ax = np.array([-e[1], e[0]])
            ax = ax / np.linalg.norm(ax)
            pa, pb = fa @ ax, fb @ ax
            best = max(best, max(pb.min() - pa.max(), pa.min() - pb.max()))
    return float(best)


def _bdist(pt, poly) -> float:
    n = len(poly)
    return min(_seg_dist(np.asarray(pt, float), poly[i], poly[(i + 1) % n]) for i in range(n))


def _fp_inside(fp, poly, margin=0.0) -> bool:
    """발자국이 다각형 안(경계 여유 margin 이상)."""
    pts = []
    for i in range(4):
        a, b = fp[i], fp[(i + 1) % 4]
        for t in (0.0, 0.25, 0.5, 0.75):
            pts.append(a + t * (b - a))
    pts = np.array(pts)
    if not _pip_vec(pts[:, 0], pts[:, 1], poly).all():
        return False
    for v in poly:                                       # 오목 모서리가 발자국 안에 들어오면 안 된다
        if _pip(float(v[0]), float(v[1]), fp):
            return False
    if margin > 0.0:
        return min(_bdist(p, poly) for p in pts) >= margin
    return True


# ------------------------------------------------------------------------------------------ 가구 사양
# (depth sx, width sy, height sz) 범위. 로컬 +x = 정면.
_DIMS = {
    "bed": ((1.95, 2.15), (1.3, 1.8), (0.45, 0.6)),
    "sofa": ((0.8, 0.95), (1.5, 2.2), (0.75, 0.9)),
    "table": ((0.8, 1.0), (1.1, 1.7), (0.72, 0.76)),
    "desk": ((0.55, 0.75), (1.0, 1.5), (0.73, 0.77)),
    "shelf": ((0.28, 0.4), (0.7, 1.2), (1.2, 2.0)),
    "cabinet": ((0.4, 0.5), (0.8, 1.4), (0.6, 0.9)),
    "wardrobe": ((0.55, 0.65), (0.9, 1.5), (1.9, 2.2)),
    "counter": ((0.55, 0.65), (1.2, 2.0), (0.88, 0.92)),
}
_WEIGHTS = {"bed": 1.0, "sofa": 1.5, "table": 1.4, "desk": 1.0, "shelf": 1.4, "cabinet": 1.5, "wardrobe": 1.0,
            "counter": 0.7}
_FLOAT_CLASSES = ("table",)
_SMALL = ("cabinet", "shelf", "counter", "desk")
_MATS_FLOOR = ("wood", "tile", "carpet", "concrete")
_MATS_WALL = ("plaster", "brick", "paint_white", "wood_panel")
_MATS_CEIL = ("paint_white", "plaster")
_SLOT_MAT = {"bed": "fabric", "sofa": "fabric", "table": "wood", "desk": "wood", "shelf": "wood",
             "cabinet": "wood", "wardrobe": "wood", "counter": "stone", "chair": "wood", "stool": "wood",
             "tv": "plastic", "rug": "fabric", "plant": "default", "lamp": "metal", "picture": "default",
             "person": "default"}


def _urange(rng, r):
    return float(rng.uniform(r[0], r[1])) if r[1] > r[0] else float(r[0])


class _Room:
    def __init__(self, rng, shell, doors, windows):
        self.rng = rng
        self.shell = shell
        self.poly = shell.corners
        self.M = shell.n_edges
        self.H = shell.height
        self.edge_L = shell.edge_lengths()
        ev = shell.edge_vectors()
        self.u = ev / self.edge_L[:, None]
        self.n = np.stack([-self.u[:, 1], self.u[:, 0]], axis=1)
        self.openings = doors + windows
        self.slots: List[Slot] = []
        self.next_id = 0
        self.floor_fps: List[np.ndarray] = []
        self.strips = []
        for o in doors:
            A, u, n = self.poly[o.edge], self.u[o.edge], self.n[o.edge]
            p = 0.15
            s0, s1 = o.s0 - p, o.s1 + p
            q = np.array([A + u * s0, A + u * s1, A + u * s1 + n * 1.15, A + u * s0 + n * 1.15])
            self.strips.append(q)

    # --- 공통
    def add(self, **kw) -> Slot:
        s = Slot(id=self.next_id, **kw)
        self.next_id += 1
        self.slots.append(s)
        return s

    def free_floor(self, fp, gap, wall_margin=0.0) -> bool:
        if not _fp_inside(fp, self.poly, wall_margin):
            return False
        for q in self.strips:
            if _sep(fp, q) < 0.0:
                return False
        for other in self.floor_fps:
            if _sep(fp, other) < gap:
                return False
        return True

    def add_floor(self, cls, size, center_xy, yaw, material=None):
        size = np.array(size, float)
        c = (float(center_xy[0]), float(center_xy[1]), float(size[2] / 2.0))
        s = self.add(cls=cls, center=c, size=size, yaw=float(yaw), support="floor",
                     material=material or _SLOT_MAT.get(cls, "default"))
        self.floor_fps.append(s.footprint())
        return s

    def wall_item_pose(self, i, s, sx, sy, gap):
        c = self.poly[i] + self.u[i] * s + self.n[i] * (sx / 2.0 + gap)
        return c, math.atan2(self.n[i][1], self.n[i][0])

    def pick_edge(self, min_len):
        cand = [i for i in range(self.M) if self.edge_L[i] >= min_len]
        if not cand:
            return -1
        w = np.array([self.edge_L[i] for i in cand])
        return int(cand[int(self.rng.choice(len(cand), p=w / w.sum()))])

    # --- 가구
    def try_furniture(self, cls, gap, attempts, small=False):
        rng = self.rng
        dims = _DIMS[cls]
        for _ in range(attempts):
            sx, sy, sz = (_urange(rng, r) for r in dims)
            sz = min(sz, self.H - 0.3)
            if small and attempts:
                sx, sy = dims[0][0], dims[1][0]
            if cls in _FLOAT_CLASSES and self.shell.area() >= 16.0 and rng.random() < 0.7:
                x0, y0 = self.poly.min(0)
                x1, y1 = self.poly.max(0)
                cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
                yaw = float(rng.integers(0, 4)) * math.pi / 2.0
                fp = _footprint((cx, cy), (sx, sy, sz), yaw)
                if self.free_floor(fp, gap, wall_margin=0.7):
                    return self.add_floor(cls, (sx, sy, sz), (cx, cy), yaw)
                continue
            i = self.pick_edge(sy + 0.3)
            if i < 0:
                continue
            L = float(self.edge_L[i])
            s = float(rng.uniform(sy / 2.0 + 0.1, L - sy / 2.0 - 0.1))
            g = float(rng.uniform(0.01, 0.06))
            c, yaw = self.wall_item_pose(i, s, sx, sy, g)
            fp = _footprint(c, (sx, sy, sz), yaw)
            if self.free_floor(fp, gap):
                return self.add_floor(cls, (sx, sy, sz), c, yaw)
        return None

    # --- 임의 바닥 위치
    def try_free_spot(self, cls, size, gap, attempts, wall_margin, yaw_mode="rand", near=None):
        rng = self.rng
        x0, y0 = self.poly.min(0)
        x1, y1 = self.poly.max(0)
        for k in range(attempts):
            if near is not None and k < attempts // 2:
                ctr = near.center[:2]
                ang = float(rng.uniform(0, 2 * math.pi))
                r = float(rng.uniform(0.6, 1.0))
                cx, cy = ctr[0] + r * math.cos(ang), ctr[1] + r * math.sin(ang)
                yaw = math.atan2(ctr[1] - cy, ctr[0] - cx)
            else:
                cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
                yaw = float(rng.uniform(-math.pi, math.pi))
            if yaw_mode == "wall":
                best = min(range(self.M), key=lambda i: _seg_dist(np.array([cx, cy]), self.poly[i],
                                                                   self.poly[(i + 1) % self.M]))
                a, b = self.poly[best], self.poly[(best + 1) % self.M]
                ab = b - a
                t = float(np.clip(np.dot(np.array([cx, cy]) - a, ab) / np.dot(ab, ab), 0, 1))
                tgt = a + t * ab
                yaw = math.atan2(tgt[1] - cy, tgt[0] - cx)
            fp = _footprint((cx, cy), size, yaw)
            if self.free_floor(fp, gap, wall_margin=wall_margin):
                return self.add_floor(cls, size, (cx, cy), yaw)
        return None

    # --- 벽걸이
    def try_picture(self, attempts=40):
        rng = self.rng
        for _ in range(attempts):
            i = self.pick_edge(1.4)
            if i < 0:
                return None
            L = float(self.edge_L[i])
            w = float(rng.uniform(0.4, 0.9))
            h = float(rng.uniform(0.4, 0.8))
            if w > L - 0.6:
                continue
            s = float(rng.uniform(w / 2.0 + 0.3, L - w / 2.0 - 0.3))
            zc = float(rng.uniform(1.3, 1.7))
            if zc + h / 2.0 > self.H - 0.2:
                continue
            ok = True
            for o in self.openings:
                if o.edge == i and min(o.s1, s + w / 2.0) - max(o.s0, s - w / 2.0) > -0.2:
                    ok = False
                    break
            if not ok:
                continue
            for q in self.slots:                       # 키 큰 가구 뒤에 걸지 않는다
                if q.support == "floor" and q.cls != "rug" and q.z1 > zc - h / 2.0:
                    fq = q.footprint()
                    if min(_seg_dist(fq[k], self.poly[i], self.poly[(i + 1) % self.M]) for k in range(4)) < 0.3:
                        ok = False
                        break
            if not ok:
                continue
            c = self.poly[i] + self.u[i] * s + self.n[i] * 0.02
            return self.add(cls="picture", center=(float(c[0]), float(c[1]), zc), size=(0.04, w, h),
                            yaw=math.atan2(self.n[i][1], self.n[i][0]), support="wall")
        return None


# ------------------------------------------------------------------------------------------ 셸 / 개구부
def _rect(W, D):
    return np.array([[0.0, 0.0], [W, 0.0], [W, D], [0.0, D]])


def _make_shell(rng, cfg):
    A = _urange(rng, cfg.area_range)
    height = _urange(rng, cfg.height_range)
    want_l = bool(rng.random() < cfg.l_shape_prob)
    if want_l:
        for _ in range(400):
            aspect = _urange(rng, cfg.aspect_range)
            fx, fy = float(rng.uniform(0.3, 0.55)), float(rng.uniform(0.3, 0.55))
            tot = A / (1.0 - fx * fy)
            W, D = math.sqrt(tot * aspect), math.sqrt(tot / aspect)
            if rng.random() < 0.5:
                W, D = D, W
            cw, ch = fx * W, fy * D
            if min(W, D, cw, ch, W - cw, D - ch) < 1.0:
                continue
            pts = np.array([[0, 0], [W, 0], [W, D - ch], [W - cw, D - ch], [W - cw, D], [0, D]], float)
            for _r in range(int(rng.integers(0, 4))):
                pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
            pts = pts - pts.min(0)
            return Shell(corners=pts, height=height), True
    aspect = _urange(rng, cfg.aspect_range)
    W, D = math.sqrt(A * aspect), math.sqrt(A / aspect)
    if rng.random() < 0.5:
        W, D = D, W
    return Shell(corners=_rect(W, D), height=height), False


def _make_doors(rng, shell, nd):
    L = shell.edge_lengths()
    cand = [i for i in range(shell.n_edges) if L[i] >= 1.6]
    order = [cand[j] for j in rng.permutation(len(cand))]
    doors = []
    for i in order[:nd]:
        w = min(float(rng.uniform(0.8, 1.05)), float(L[i]) - 0.8)
        w = max(w, 0.7)
        s0 = float(rng.uniform(0.4, L[i] - 0.4 - w))
        doors.append(Opening(kind="door", edge=i, s0=s0, s1=s0 + w, z0=0.0, z1=float(rng.uniform(2.0, 2.2))))
    return doors


def _make_windows(rng, shell, door_edges, nw, lo_req):
    if nw <= 0:
        return []
    L = shell.edge_lengths()
    cand = [i for i in range(shell.n_edges) if i not in door_edges and L[i] >= 2.0]
    if not cand:
        return []
    cand.sort(key=lambda i: -L[i])
    i = cand[int(rng.integers(0, min(2, len(cand))))]
    ww = float(rng.uniform(0.8, 1.3))
    nmax = int((L[i] - 0.3) // (ww + 0.3))
    n = min(nw, nmax)
    if lo_req >= 1:
        n = max(n, 1)
    if n < 1:
        return []
    z0 = float(rng.uniform(0.8, 1.1))
    z1 = z0 + float(rng.uniform(0.8, 1.1))
    z1 = min(z1, shell.height - 0.2)
    out = []
    for k in range(n):
        c = float(L[i]) * (k + 0.5) / n
        out.append(Opening(kind="window", edge=i, s0=c - ww / 2.0, s1=c + ww / 2.0, z0=z0, z1=z1))
    return out


# ------------------------------------------------------------------------------------------ 본체
def generate_room(seed: int, cfg: RoomGenConfig | None = None) -> SceneProgram:
    """(seed, cfg) -> SceneProgram. 같은 입력이면 같은 content_hash."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed 는 음이 아닌 정수여야 한다: {seed!r}")
    seed = int(seed)
    cfg = RoomGenConfig() if cfg is None else cfg
    if not isinstance(cfg, RoomGenConfig):
        raise TypeError("cfg 는 RoomGenConfig 여야 한다")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x524F4F4D]))

    shell, is_l = _make_shell(rng, cfg)
    area = shell.area()
    nd = int(rng.integers(cfg.n_doors_range[0], cfg.n_doors_range[1] + 1))
    doors = _make_doors(rng, shell, nd)
    nw = int(rng.integers(cfg.n_windows_range[0], cfg.n_windows_range[1] + 1))
    windows = _make_windows(rng, shell, {d.edge for d in doors}, nw, cfg.n_windows_range[0])
    room = _Room(rng, shell, doors, windows)
    big = area >= 25.0

    # 가구
    k = int(rng.integers(cfg.furniture_range[0], cfg.furniture_range[1] + 1))
    placed = 0
    foot = 0.0
    has_bed = False
    for _ in range(k):
        if not big and foot > 0.35 * area:
            break
        w = np.array([_WEIGHTS[c] * (0.0 if (c == "bed" and has_bed) else 1.0) for c in FURNITURE_CLASSES])
        cls = FURNITURE_CLASSES[int(rng.choice(len(w), p=w / w.sum()))]
        s = room.try_furniture(cls, 0.1, 150)
        if s is None:
            s = room.try_furniture(cls, 0.0, 150)
        if s is None:
            for alt in _SMALL:
                s = room.try_furniture(alt, 0.0, 100, small=True)
                if s is not None:
                    break
        if s is None and big:
            for _t in range(5):
                s = room.try_furniture("cabinet", 0.0, 200, small=True)
                if s is not None:
                    break
        if s is None:
            continue
        placed += 1
        has_bed = has_bed or s.cls == "bed"
        foot += float(s.size[0] * s.size[1])

    # 부속
    m = int(rng.integers(cfg.accessory_range[0], cfg.accessory_range[1] + 1))
    pool = ("chair", "plant", "stool", "tv", "picture", "rug")
    for _ in range(m):
        cls = pool[int(rng.integers(0, len(pool)))]
        if cls == "chair":
            tab = [s for s in room.slots if s.cls in ("table", "desk")]
            near = tab[int(rng.integers(0, len(tab)))] if tab else None
            room.try_free_spot("chair", (0.45, 0.45, 0.9), 0.05, 60, 0.15, near=near)
        elif cls == "stool":
            room.try_free_spot("stool", (0.35, 0.35, 0.45), 0.05, 60, 0.15)
        elif cls == "plant":
            room.try_free_spot("plant", (0.4, 0.4, float(rng.uniform(0.6, 1.4))), 0.05, 60, 0.1)
        elif cls == "rug":
            sx, sy = float(rng.uniform(1.2, 2.2)), float(rng.uniform(0.8, 1.6))
            for _t in range(40):
                x0, y0 = room.poly.min(0)
                x1, y1 = room.poly.max(0)
                cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
                yaw = float(rng.integers(0, 4)) * math.pi / 2.0
                fp = _footprint((cx, cy), (sx, sy, 0.01), yaw)
                if _fp_inside(fp, room.poly, 0.1):
                    room.add(cls="rug", center=(cx, cy, 0.005), size=(sx, sy, 0.01), yaw=yaw, support="floor",
                             material="fabric")
                    break
        elif cls == "picture":
            room.try_picture()
        else:  # tv: 낮은 가구 위
            used = {s.support_id for s in room.slots if s.cls == "tv"}
            sup = [s for s in room.slots if s.cls in ("cabinet", "table", "desk", "counter") and s.id not in used
                   and s.z1 <= 1.0 and s.size[1] >= 0.8 and s.z1 + 0.5 <= shell.height - 0.2]
            if sup:
                b = sup[int(rng.integers(0, len(sup)))]
                room.add(cls="tv", center=(float(b.center[0]), float(b.center[1]), b.z1 + 0.25),
                         size=(0.08, 0.7, 0.5), yaw=b.yaw, support="slot", support_id=b.id, material="plastic")

    # 램프(천장) 1-2개
    nl = 1 if area < 20.0 else 2
    x0, y0 = room.poly.min(0)
    x1, y1 = room.poly.max(0)
    lamp_xy = []
    for _ in range(nl):
        best, bd = None, -1.0
        for _t in range(200):
            p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if not _pip(float(p[0]), float(p[1]), room.poly):
                continue
            d = _bdist(p, room.poly)
            if lamp_xy:
                d = min(d, 0.5 * min(float(np.linalg.norm(p - q)) for q in lamp_xy))
            if d > bd:
                best, bd = p, d
        if best is None:
            best = room.poly.mean(0)
        lamp_xy.append(best)
    lamps = []
    for p in lamp_xy:
        lamps.append(room.add(cls="lamp", center=(float(p[0]), float(p[1]), shell.height - 0.05),
                              size=(0.4, 0.4, 0.1), yaw=0.0, support="ceiling", material="metal"))

    # 사람(요청 시)
    for _ in range(cfg.people):
        s = room.try_free_spot("person", (0.5, 0.5, 1.7), 0.3, 200, 0.5, yaw_mode="wall")
        if s is None and big:
            room.try_free_spot("person", (0.5, 0.5, 1.7), 0.1, 400, 0.4, yaw_mode="wall")

    # 광원
    lights = []
    lid = 0
    for lp in lamps:
        lights.append(Light(id=lid, kind="lamp", pos=(float(lp.center[0]), float(lp.center[1]), shell.height - 0.15),
                            rgb=(1.0, 0.85, 0.7), power=800.0, source=("slot", lp.id)))
        lid += 1
    base = len(doors)
    for wi, w in enumerate(windows):
        A, u, n = room.poly[w.edge], room.u[w.edge], room.n[w.edge]
        c = A + u * (w.s0 + w.s1) / 2.0 + n * 0.15
        lights.append(Light(id=lid, kind="window", pos=(float(c[0]), float(c[1]), (w.z0 + w.z1) / 2.0),
                            rgb=(0.9, 0.95, 1.0), power=3000.0, source=("opening", base + wi)))
        lid += 1

    matter = {"floor": _MATS_FLOOR[int(rng.integers(0, len(_MATS_FLOOR)))],
              "wall": _MATS_WALL[int(rng.integers(0, len(_MATS_WALL)))],
              "ceiling": _MATS_CEIL[int(rng.integers(0, len(_MATS_CEIL)))]}
    meta = {"roomgen": "v0", "area": float(area), "l_shape": bool(is_l), "n_furniture": int(placed)}
    return SceneProgram(grammar="room", seed=seed, shell=shell, openings=tuple(doors) + tuple(windows),
                        slots=tuple(room.slots), lights=tuple(lights), matter=matter, meta=meta)
