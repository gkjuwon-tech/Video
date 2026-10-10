"""INV-01 레이캐스트: 해석해, 독립 오라클 대조, 왜곡, 가림, 열린 벽, 결정론, 속도."""
import math
import time

import numpy as np
import pytest

from buc.core import geom
from inv01_helpers import (INF, base_program, box_room, free_point, look, make_K, oracle_raycast, w2c_Rt)


def _G():
    from buc.invent.grammar import core
    return core


def _rc(prog, c2w, w, h, fov=90.0, **kw):
    G = _G()
    return G.raycast(prog, c2w=c2w, K=make_K(w, h, fov), width=w, height=h, **kw)


def test_result_types_and_shapes():
    G = _G()
    prog = box_room()
    r = _rc(prog, look((0, 0, 1.5), (0, 1, 1.5)), 20, 12)
    assert isinstance(r, G.RayCastResult)
    assert r.depth.shape == (12, 20) and r.depth.dtype == np.float32
    assert r.sem.shape == (12, 20) and r.sem.dtype == np.int16
    assert r.inst.shape == (12, 20) and r.inst.dtype == np.int32
    assert r.normal.shape == (12, 20, 3) and r.normal.dtype == np.float32
    assert np.all(np.isfinite(r.depth)) and np.all(r.depth > 0)
    assert r.visible_slots() == {}


def test_empty_room_analytic_depths_sem_normals():
    """4x6x2.8 방, 눈 (0,0,1.5)에서 +y를 본다. FOV 90, 33x21: 주점 = 픽셀 (16,10), fx=fy=(W/2)/tan(45)=16.5 (FND-06 intrinsics_from_fov)."""
    prog = box_room()
    r = _rc(prog, look((0, 0, 1.5), (0, 1, 1.5)), 33, 21, 90.0)
    G = _G()
    # 중앙: 앞 벽 y=3 까지 정확히 3.0 (광축 z-깊이)
    assert r.depth[10, 16] == pytest.approx(3.0, abs=1e-5)
    assert r.sem[10, 16] == G.SEM_WALL and r.inst[10, 16] == -1
    np.testing.assert_allclose(r.normal[10, 16], [0, -1, 0], atol=1e-6)
    # 맨 아랫줄 v=20: y_n=10/16.5 -> 바닥, t = 1.5*16.5/10 = 2.475
    assert r.depth[20, 16] == pytest.approx(2.475, abs=1e-5)
    assert r.sem[20, 16] == G.SEM_FLOOR
    np.testing.assert_allclose(r.normal[20, 16], [0, 0, 1], atol=1e-6)
    # 맨 윗줄 v=0: y_n=-10/16.5 -> 천장, t = 1.3*16.5/10 = 2.145
    assert r.depth[0, 16] == pytest.approx(2.145, abs=1e-5)
    assert r.sem[0, 16] == G.SEM_CEILING
    np.testing.assert_allclose(r.normal[0, 16], [0, 0, -1], atol=1e-6)
    # 오른쪽 끝 열 u=32: x_n=16/16.5 -> 카메라 오른쪽은 세계 +x (눈이 +y, up=+z 일 때). 벽 x=2 까지 t=2*16.5/16
    assert r.depth[10, 32] == pytest.approx(2.0625, abs=1e-5)
    np.testing.assert_allclose(r.normal[10, 32], [-1, 0, 0], atol=1e-6)
    # 모든 법선은 단위 길이이고 광선을 마주 본다
    n = r.normal.reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


def test_box_slot_hit_and_rotation():
    G = _G()
    sl = G.Slot(id=7, cls="wardrobe", center=(0, 2, 1.0), size=(1, 1, 2.0), yaw=0.0)
    prog = box_room(slots=(sl,))
    r = _rc(prog, look((0, 0, 1.5), (0, 1, 1.5)), 33, 21)
    assert r.depth[10, 16] == pytest.approx(1.5, abs=1e-5)
    assert r.sem[10, 16] == G.sem_id("wardrobe") and r.inst[10, 16] == 7
    np.testing.assert_allclose(r.normal[10, 16], [0, -1, 0], atol=1e-6)
    assert r.visible_slots().get(7, 0) > 0
    # 45도 회전: 눈을 x=+0.1 로 옮기면 중앙 광선은 법선 (+1,-1)/sqrt2 인 면을 맞힌다.
    sl45 = G.Slot(id=7, cls="wardrobe", center=(0, 2, 1.0), size=(1, 1, 2.0), yaw=math.pi / 4)
    prog = box_room(slots=(sl45,))
    r = _rc(prog, look((0.1, 0, 1.5), (0.1, 1, 1.5)), 33, 21)
    assert r.depth[10, 16] == pytest.approx(2.0 - (math.sqrt(0.5) - 0.1), abs=1e-5)
    np.testing.assert_allclose(r.normal[10, 16], [math.sqrt(0.5), -math.sqrt(0.5), 0], atol=1e-5)
    assert r.inst[10, 16] == 7


def test_occlusion_and_visible_slot_ids():
    G = _G()
    a = G.Slot(id=1, cls="wardrobe", center=(0, 1.2, 1.0), size=(0.6, 0.6, 2.0), yaw=0.0)
    b = G.Slot(id=2, cls="wardrobe", center=(0, 2.5, 1.0), size=(0.6, 0.6, 2.0), yaw=0.0)   # a 에 완전히 가려짐
    c = G.Slot(id=3, cls="cabinet", center=(1.2, 2.2, 0.4), size=(0.5, 0.5, 0.8), yaw=0.0)   # 옆, 보임
    prog = box_room(slots=(b, a, c))                                                     # 순서와 무관해야 한다
    r = _rc(prog, look((0, 0, 1.5), (0, 1, 1.5)), 48, 32, 70.0)
    vis = r.visible_slots()
    assert 1 in vis and vis[1] > 20
    assert 2 not in vis
    assert 3 in vis
    assert sum(vis.values()) == int((r.inst >= 0).sum())
    assert r.inst[16, 24] == 1
    # 순서를 바꿔도 결과 바이트가 같다
    r2 = _rc(box_room(slots=(a, c, b)), look((0, 0, 1.5), (0, 1, 1.5)), 48, 32, 70.0)
    assert r.depth.tobytes() == r2.depth.tobytes() and r.inst.tobytes() == r2.inst.tobytes()


def test_missing_wall_gives_inf_void_and_zero_normal():
    G = _G()
    prog = box_room(walls_present=(True, True, True, False))        # 변 3 (x=-2)이 없다
    r = _rc(prog, look((1, 0, 1.5), (0, 0, 1.5)), 33, 21)
    assert np.isinf(r.depth[10, 16]) and r.depth[10, 16] > 0
    assert r.sem[10, 16] == G.SEM_VOID and r.inst[10, 16] == -1
    np.testing.assert_array_equal(r.normal[10, 16], [0, 0, 0])
    # 바닥/천장은 다각형 안에서만 존재한다: 열린 쪽으로 나간 광선은 바닥을 맞히지 않는다
    assert np.isinf(r.depth).any() and np.isfinite(r.depth).any()
    # 바닥이 없으면 아래 광선이 빠진다
    r = _rc(box_room(floor=False), look((0, 0, 1.5), (0, 0.001, 0), up=(0, 1, 0)), 9, 9, 40.0)
    assert np.isinf(r.depth[4, 4]) and r.sem[4, 4] == G.SEM_VOID


def test_door_and_window_are_painted_on_wall_plane():
    G = _G()
    # 변 2 = y=3, 모서리 (2,3)->(-2,3). s 는 (2,3)에서 -x 방향. x=0 은 s=2.
    door = G.Opening(kind="door", edge=2, s0=1.5, s1=2.5, z0=0.0, z1=2.1)
    win = G.Opening(kind="window", edge=2, s0=0.0, s1=0.8, z0=1.0, z1=2.0)       # x in [1.2, 2.0]
    prog = box_room(openings=(door, win))
    r = _rc(prog, look((0, 0, 1.5), (0, 1, 1.5)), 33, 21, 70.0)
    assert r.sem[10, 16] == G.SEM_DOOR and r.depth[10, 16] == pytest.approx(3.0, abs=1e-5)
    np.testing.assert_allclose(r.normal[10, 16], [0, -1, 0], atol=1e-6)         # 깊이·법선은 벽과 같다
    fx = 16.5 / math.tan(math.radians(35.0))
    u_win = 16 + fx * (1.6 / 3.0)                                                # 창 중심 x=1.6
    assert r.sem[10, int(round(u_win))] == G.SEM_WINDOW
    # 겹치면 먼저 나온 개구부가 이긴다
    over = G.Opening(kind="window", edge=2, s0=1.0, s1=3.0, z0=0.0, z1=2.5)
    r2 = _rc(box_room(openings=(door, over)), look((0, 0, 1.5), (0, 1, 1.5)), 33, 21, 70.0)
    assert r2.sem[10, 16] == G.SEM_DOOR
    r3 = _rc(box_room(openings=(over, door)), look((0, 0, 1.5), (0, 1, 1.5)), 33, 21, 70.0)
    assert r3.sem[10, 16] == G.SEM_WINDOW
    # 개구부가 없는 변의 픽셀은 벽
    assert (r.sem == G.SEM_WALL).any()


def test_lens_distortion_matches_fnd06_and_lands_on_surfaces():
    G = _G()
    prog = box_room()
    c2w = look((0.2, -0.3, 1.4), (0.3, 1.0, 1.2))
    K = make_K(40, 30, 80.0)
    k1, k2 = 0.15, -0.05
    r = G.raycast(prog, c2w=c2w, K=K, width=40, height=30, k1=k1, k2=k2)
    r0 = G.raycast(prog, c2w=c2w, K=K, width=40, height=30)
    assert np.abs(r.depth - r0.depth).max() > 1e-3                              # 왜곡이 실제로 적용됨
    R, t = w2c_Rt(c2w)
    uv = geom.pixel_grid(30, 40)
    X = geom.unproject_world(K, R, t, uv, r.depth.astype(np.float64), k1, k2)   # FND-06 규약으로 3D 복원
    d = np.minimum.reduce([np.abs(X[..., 0] - 2), np.abs(X[..., 0] + 2), np.abs(X[..., 1] - 3),
                           np.abs(X[..., 1] + 3), np.abs(X[..., 2]), np.abs(X[..., 2] - 2.8)])
    assert d.max() < 5e-4                                                       # 모든 점이 어떤 면 위에 있다
    assert (np.abs(X[..., 0]) <= 2 + 5e-4).all() and (np.abs(X[..., 1]) <= 3 + 5e-4).all()
    assert ((X[..., 2] >= -5e-4) & (X[..., 2] <= 2.8 + 5e-4)).all()


def _compare(prog, c2w, w, h, fov, k1=0.0, k2=0.0):
    G = _G()
    K = make_K(w, h, fov)
    r = G.raycast(prog, c2w=c2w, K=K, width=w, height=h, k1=k1, k2=k2)
    od, os_, oi, on = oracle_raycast(prog, c2w, K, w, h, k1, k2)
    fin = np.isfinite(od)
    assert (np.isfinite(r.depth) == fin).mean() >= 0.998
    both = fin & np.isfinite(r.depth)
    rel = np.abs(r.depth[both].astype(np.float64) - od[both]) / od[both]
    # 경계 픽셀 몇 개를 제외하면 깊이가 같다
    assert (rel < 2e-5).mean() >= 0.995, rel.max()
    assert (r.sem == os_).mean() >= 0.995
    assert (r.inst == oi).mean() >= 0.995
    same = both & (r.sem == os_) & (r.inst == oi)
    dots = (r.normal[same].astype(np.float64) * on[same]).sum(-1)
    assert (dots > 0.999).mean() >= 0.995
    return r


def test_matches_independent_oracle_on_handmade_scene():
    prog = base_program()
    for seed, (w, h, fov, k1) in enumerate([(16, 12, 70.0, 0.0), (15, 11, 100.0, 0.12), (12, 12, 60.0, -0.08)]):
        rng = np.random.default_rng(seed)
        eye = np.array([rng.uniform(1.5, 3.5), rng.uniform(1.0, 3.0), rng.uniform(1.2, 1.8)])
        tgt = eye + np.array([rng.normal(), rng.normal(), rng.normal() * 0.2])
        _compare(prog, look(eye, tgt), w, h, fov, k1=k1, k2=0.02 * k1)


def test_matches_independent_oracle_on_roomgen_rooms(rooms):
    count = 0
    for i, p in enumerate(rooms[:6]):
        pt = free_point(p)
        if pt is None:
            continue
        rng = np.random.default_rng(100 + i)
        yaw = rng.uniform(0, 2 * np.pi)
        eye = np.array([pt[0], pt[1], 1.5])
        tgt = eye + np.array([math.cos(yaw), math.sin(yaw), -0.15])
        _compare(p, look(eye, tgt), 14, 10, 75.0, k1=0.05 * (i % 2))
        count += 1
    assert count >= 4


def test_roomgen_rooms_are_closed_from_a_free_point(rooms):
    """방은 닫혀 있다: 자유 지점에서 큐브 6면(FOV 90)으로 찍으면 모든 광선이 맞는다."""
    G = _G()
    tested = 0
    for p in rooms[:10]:
        pt = free_point(p)
        if pt is None:
            continue
        eye = np.array([pt[0], pt[1], 1.5])
        for tgt_off, up in (((1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)),
                            ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (1, 0, 0))):
            r = G.raycast(p, c2w=look(eye, eye + np.array(tgt_off), up=up), K=make_K(16, 16, 90.0), width=16,
                          height=16)
            assert np.isfinite(r.depth).all() and (r.depth > 0).all()
            assert (r.sem != G.SEM_VOID).all()
        tested += 1
    assert tested >= 7


def test_no_runtime_warnings_even_with_axis_parallel_and_missing_surfaces():
    import warnings
    G = _G()
    prog = box_room(slots=(G.Slot(id=1, cls="cabinet", center=(0, 2, 0.5), size=(1, 1, 1), yaw=0.0),),
                    walls_present=(True, False, True, True), floor=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 광축이 축과 정확히 평행한 카메라(주점 광선의 성분이 0)와 일반 카메라
        _rc(prog, look((0, 0, 0.5), (0, 1, 0.5)), 9, 9, 60.0)
        _rc(prog, look((0, 0, 0.5), (1, 0, 0.5)), 9, 9, 60.0)
        _rc(prog, look((0, 0, 1.5), (0.3, 1, 0.2)), 17, 11, 80.0, k1=0.1)


def test_deterministic_and_input_unchanged():
    G = _G()
    prog = base_program()
    h = prog.content_hash()
    c2w = look((2.5, 1.0, 1.5), (2.5, 4.0, 1.2))
    a = _rc(prog, c2w, 24, 16, 70.0)
    b = _rc(prog, c2w, 24, 16, 70.0)
    for f in ("depth", "sem", "inst", "normal"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes()
    assert prog.content_hash() == h
    c2w32 = np.asarray(c2w, np.float32)
    r = G.raycast(prog, c2w=c2w32.tolist(), K=make_K(24, 16, 70.0).tolist(), width=24, height=16)
    assert np.isfinite(r.depth).all()


def test_input_validation():
    G = _G()
    prog = base_program()
    K = make_K(8, 6)
    c2w = look((2.5, 1.0, 1.5), (2.5, 4.0, 1.5))
    with pytest.raises(TypeError):
        G.raycast(prog, c2w, K, 8, 6)
    bad = c2w.copy()
    bad[:3, :3] *= 1.5
    with pytest.raises(ValueError):
        G.raycast(prog, c2w=bad, K=K, width=8, height=6)
    with pytest.raises(ValueError):
        G.raycast(prog, c2w=c2w, K=K, width=0, height=6)
    with pytest.raises(ValueError):
        G.raycast(prog, c2w=c2w, K=K, width=8.5, height=6)
    with pytest.raises(ValueError):
        G.raycast(prog, c2w=c2w, K=K[:2], width=8, height=6)
    nan = c2w.copy()
    nan[0, 3] = np.nan
    with pytest.raises(ValueError):
        G.raycast(prog, c2w=nan, K=K, width=8, height=6)
    with pytest.raises(ValueError):
        G.raycast(prog, c2w=c2w, K=K, width=8, height=6, backend="cuda")


def test_backends():
    G = _G()
    prog = base_program()
    c2w = look((2.5, 1.0, 1.5), (2.0, 4.0, 1.2))
    K = make_K(32, 24, 80.0)
    a = G.raycast(prog, c2w=c2w, K=K, width=32, height=24, backend="numpy")
    d = G.raycast(prog, c2w=c2w, K=K, width=32, height=24)             # auto
    assert a.depth.tobytes() == d.depth.tobytes() and a.sem.tobytes() == d.sem.tobytes()
    if G.HAS_NUMBA:
        n = G.raycast(prog, c2w=c2w, K=K, width=32, height=24, backend="numba")
        np.testing.assert_allclose(n.depth, a.depth, rtol=1e-6, atol=1e-6)
        assert (n.sem == a.sem).mean() > 0.999 and (n.inst == a.inst).mean() > 0.999
    else:
        with pytest.raises(RuntimeError):
            G.raycast(prog, c2w=c2w, K=K, width=32, height=24, backend="numba")


def test_speed_64x64(rooms):
    """장면당 레이캐스트는 CPU에서 수십 ms 급이어야 한다. 느슨한 상한만 본다."""
    G = _G()
    p = rooms[1]
    pt = free_point(p) or tuple(p.shell.corners.mean(0))
    eye = np.array([pt[0], pt[1], 1.5])
    c2w = look(eye, eye + np.array([1.0, 0.3, -0.1]))
    K = make_K(64, 64, 70.0)
    G.raycast(p, c2w=c2w, K=K, width=64, height=64)                     # 워밍업(JIT 포함)
    t0 = time.perf_counter()
    for _ in range(20):
        G.raycast(p, c2w=c2w, K=K, width=64, height=64)
    dt = (time.perf_counter() - t0) / 20
    assert dt < 0.5, f"{dt * 1000:.0f} ms"
