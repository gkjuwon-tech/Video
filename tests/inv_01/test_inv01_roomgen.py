"""INV-01 RoomGen v0: 결정론, 구성 가능성, 문법 만족(G1-G13 = 0), 기하 건전성, 다양성, 속도."""
import dataclasses
import math
import time

import numpy as np
import pytest

from inv01_helpers import dist_to_boundary, footprint, obb_overlap_depth, point_in_poly

FAST = dict(furniture_range=(0, 0), accessory_range=(0, 0))


def _R():
    from buc.invent.grammar import room
    return room


def _G():
    from buc.invent.grammar import core
    return core


def test_config_defaults_and_validation():
    R = _R()
    c = R.RoomGenConfig()
    assert c.area_range == (9.0, 40.0) and c.aspect_range == (1.0, 2.2) and c.height_range == (2.4, 3.2)
    assert c.l_shape_prob == 0.25 and c.furniture_range == (3, 9) and c.accessory_range == (0, 6)
    assert c.n_doors_range == (1, 2) and c.n_windows_range == (0, 4) and c.people == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.people = 2
    with pytest.raises(TypeError):
        R.RoomGenConfig((9, 40))
    for bad in (dict(area_range=(5.0, 10.0)), dict(area_range=(20.0, 10.0)), dict(l_shape_prob=1.5),
                dict(l_shape_prob=-0.1), dict(people=-1), dict(n_doors_range=(0, 1)), dict(height_range=(2.0, 3.0)),
                dict(furniture_range=(5, 2)), dict(aspect_range=(0.5, 2.0)), dict(n_windows_range=(-1, 2))):
        with pytest.raises(ValueError):
            R.RoomGenConfig(**bad)
    for bad_seed in (-1, 1.5, True, "3", None):
        with pytest.raises((ValueError, TypeError)):
            R.generate_room(bad_seed)


def test_determinism_and_seed_sensitivity():
    R = _R()
    a, b = R.generate_room(11), R.generate_room(11)
    assert a.content_hash() == b.content_hash()
    assert R.generate_room(np.int64(11)).content_hash() == a.content_hash()
    assert R.generate_room(12).content_hash() != a.content_hash()
    cfg = R.RoomGenConfig(l_shape_prob=0.0)
    assert R.generate_room(11, cfg).content_hash() == R.generate_room(11, cfg).content_hash()
    assert a.seed == 11 and a.grammar == "room"
    assert R.generate_room(0).seed == 0


def test_every_room_satisfies_the_grammar_exactly(rooms):
    """G1-G13 모두 0 (카메라 없음). 방 30개."""
    G = _G()
    for p in rooms:
        p.validate()
        e = G.grammar_energy(p)
        for k, v in e.terms.items():
            assert v == pytest.approx(0.0, abs=1e-6), (p.seed, k, v)


def test_room_geometry_is_sound(rooms):
    R = _R()
    cfg = R.RoomGenConfig()
    for p in rooms:
        sh = p.shell
        assert sh.n_edges in (4, 6)
        assert cfg.area_range[0] - 1e-6 <= sh.area() <= cfg.area_range[1] + 1e-6, (p.seed, sh.area())
        assert cfg.height_range[0] - 1e-9 <= sh.height <= cfg.height_range[1] + 1e-9
        assert sh.edge_lengths().min() >= 1.0, (p.seed, sh.edge_lengths())
        assert all(sh.walls_present) and sh.floor and sh.ceiling
        poly = sh.corners
        floor_fps = []
        for s in p.slots:
            fp = footprint(s.center, s.size, s.yaw)
            for i in range(4):                                       # 발자국 경계가 모두 방 안(또는 벽 위)
                for t in np.linspace(0, 1, 7):
                    pt = fp[i] + t * (fp[(i + 1) % 4] - fp[i])
                    assert point_in_poly(pt, poly) or dist_to_boundary(pt, poly) < 1e-6, (p.seed, s.id, s.cls)
            assert s.z0 >= -1e-6 and s.z1 <= sh.height + 1e-6, (p.seed, s.cls)
            if s.support == "floor" and s.cls != "rug":
                floor_fps.append((s, fp))
        for i in range(len(floor_fps)):                              # 바닥 가구끼리 관통 없음
            for j in range(i + 1, len(floor_fps)):
                assert obb_overlap_depth(floor_fps[i][1], floor_fps[j][1]) <= 1e-6, (p.seed, floor_fps[i][0].cls,
                                                                                    floor_fps[j][0].cls)
        ids = [s.id for s in p.slots]
        assert len(set(ids)) == len(ids)


def test_openings_and_lights_structure(rooms):
    for p in rooms:
        doors = [o for o in p.openings if o.kind == "door"]
        wins = [o for o in p.openings if o.kind == "window"]
        assert 1 <= len(doors) <= 2 and len(wins) <= 4
        assert len({o.edge for o in doors}) == len(doors)                       # 한 변에 문 하나
        for d in doors:
            assert d.z0 == pytest.approx(0.0) and 1.9 <= d.z1 <= 2.4 and 0.7 <= d.s1 - d.s0 <= 1.2
        win_edges = {o.edge for o in wins}
        assert not (win_edges & {d.edge for d in doors})                         # 창과 문은 다른 변
        assert len(win_edges) <= 1
        assert len(p.lights) >= 1
        lamps = [s for s in p.slots if s.cls == "lamp"]
        assert len(lamps) >= 1 and all(s.support == "ceiling" for s in lamps)
        assert len(p.lights) == len(lamps) + len(wins)                         # 램프 하나당, 창 하나당 광원 하나
        assert set(p.matter) == {"floor", "wall", "ceiling"}


def test_config_controls_are_honored():
    R = _R()
    G = _G()
    # L자 확률 0/1
    for s in range(12):
        assert R.generate_room(s, R.RoomGenConfig(l_shape_prob=0.0, **FAST)).shell.n_edges == 4
        assert R.generate_room(s, R.RoomGenConfig(l_shape_prob=1.0, **FAST)).shell.n_edges == 6
    # 면적/높이 범위
    cfg = R.RoomGenConfig(area_range=(30.0, 31.0), height_range=(3.0, 3.0), **FAST)
    for s in range(10):
        p = R.generate_room(s, cfg)
        assert 30.0 - 1e-6 <= p.shell.area() <= 31.0 + 1e-6 and p.shell.height == pytest.approx(3.0)
    # 가구 0개: 바닥 가구 클래스가 없다
    for s in range(10):
        p = R.generate_room(s, R.RoomGenConfig(**FAST))
        assert not [x for x in p.slots if x.cls in G.FURNITURE_CLASSES]
    # 큰 방 + 가구 6 -> 정확히 6 (재시도로 반드시 채운다)
    big = R.RoomGenConfig(area_range=(25.0, 40.0), furniture_range=(6, 6), accessory_range=(0, 0))
    for s in range(15):
        p = R.generate_room(s, big)
        n = sum(1 for x in p.slots if x.support == "floor" and x.cls in G.FURNITURE_CLASSES)
        assert n == 6, (s, n)
    # 문 정확히 2, 창 없음
    c2 = R.RoomGenConfig(area_range=(14.0, 40.0), l_shape_prob=0.0, n_doors_range=(2, 2), n_windows_range=(0, 0), **FAST)
    for s in range(10):
        p = R.generate_room(s, c2)
        assert sum(o.kind == "door" for o in p.openings) == 2 and not any(o.kind == "window" for o in p.openings)
    # 창을 요구하면 (넓은 방) 대부분 창이 생긴다
    c3 = R.RoomGenConfig(area_range=(25.0, 40.0), l_shape_prob=0.0, n_windows_range=(3, 3), **FAST)
    counts = [sum(o.kind == "window" for o in R.generate_room(s, c3).openings) for s in range(10)]
    assert min(counts) >= 1 and max(counts) <= 3
    # 사람: 요청한 수 이하, 큰 방에서는 정확히. 그리고 문법을 깨지 않는다
    c4 = R.RoomGenConfig(area_range=(25.0, 40.0), furniture_range=(2, 2), accessory_range=(0, 0), people=2)
    for s in range(6):
        p = R.generate_room(s, c4)
        ppl = [x for x in p.slots if x.cls == "person"]
        assert len(ppl) == 2
        assert all(abs(x.z0) < 1e-9 and x.size[2] == pytest.approx(1.7, abs=0.3) for x in ppl)
        assert G.grammar_energy(p).total == pytest.approx(0.0, abs=1e-6)


def test_people_are_not_placed_by_default(rooms):
    assert not any(s.cls == "person" for p in rooms for s in p.slots)


def test_diversity_over_seeds(rooms):
    G = _G()
    hashes = {p.content_hash() for p in rooms}
    assert len(hashes) == len(rooms)
    areas = np.array([p.shell.area() for p in rooms])
    assert areas.min() < 16.0 and areas.max() > 26.0 and areas.std() > 4.0
    classes = {s.cls for p in rooms for s in p.slots}
    assert len(classes) >= 9, classes
    assert {"bed", "sofa", "table", "lamp"} & classes
    counts = [len(p.slots) for p in rooms]
    assert min(counts) >= 1 and max(counts) >= 8 and len(set(counts)) >= 5
    heights = np.array([p.shell.height for p in rooms])
    assert heights.max() - heights.min() > 0.3
    mats = {p.matter["floor"] for p in rooms}
    assert len(mats) >= 3 and mats <= set(G.MATERIALS)
    assert sum(1 for p in rooms if any(o.kind == "window" for o in p.openings)) >= 10


def test_l_shape_rate_matches_probability():
    R = _R()
    n = 200
    frac = np.mean([R.generate_room(s, R.RoomGenConfig(l_shape_prob=0.25, **FAST)).shell.n_edges == 6
                    for s in range(n)])
    assert 0.15 <= frac <= 0.35, frac                          # 이항 표준편차 약 0.031, 약 3.2 sigma
    frac = np.mean([R.generate_room(s, R.RoomGenConfig(l_shape_prob=0.5, **FAST)).shell.n_edges == 6
                    for s in range(n)])
    assert 0.39 <= frac <= 0.61, frac


def test_l_shaped_rooms_are_simple_ccw_polygons():
    R = _R()
    for s in range(20):
        p = R.generate_room(s, R.RoomGenConfig(l_shape_prob=1.0, **FAST))
        c = p.shell.corners
        assert c.shape == (6, 2) and p.shell.area() > 0
        # 직각 다각형: 모든 변이 축 평행이고 안쪽 모서리(오목) 정확히 1개
        ev = p.shell.edge_vectors()
        assert np.all((np.abs(ev[:, 0]) < 1e-9) | (np.abs(ev[:, 1]) < 1e-9))
        cross = ev[:, 0] * np.roll(ev[:, 1], -1) - ev[:, 1] * np.roll(ev[:, 0], -1)
        assert (cross > 1e-9).sum() == 5 and (cross < -1e-9).sum() == 1


def test_generation_speed():
    """장면당 목표는 수십 ms. CPU 컨테이너에서 느슨하게 평균 0.4 s 이하."""
    R = _R()
    G = _G()
    t0 = time.perf_counter()
    ps = [R.generate_room(1000 + i) for i in range(20)]
    dt = (time.perf_counter() - t0) / 20
    assert dt < 0.4, f"{dt * 1000:.0f} ms/room"
    t0 = time.perf_counter()
    for p in ps:
        G.grammar_energy(p)
    dt = (time.perf_counter() - t0) / 20
    assert dt < 0.4, f"{dt * 1000:.0f} ms/energy"
