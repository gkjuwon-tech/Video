"""INV-01 에너지 골격 G1-G13: 기준 장면은 0, 위반은 계약서 4.3절 공식대로 정확히 센다."""
import dataclasses
import math

import numpy as np
import pytest

from inv01_helpers import base_program, look, make_K


def _G():
    from buc.invent.grammar import core
    return core


def E(prog, ctx=None, **kw):
    return _G().grammar_energy(prog, ctx, **kw)


def rep(p, **kw):
    return dataclasses.replace(p, **kw)


def slot(**kw):
    G = _G()
    args = dict(id=50, cls="wardrobe", center=(0.0, 0.0, 1.0), size=(0.6, 0.6, 2.0), yaw=0.0)
    args.update(kw)
    return G.Slot(**args)


def test_result_structure_and_base_scene_is_zero():
    G = _G()
    e = E(base_program())
    assert isinstance(e, G.GrammarEnergy)
    assert tuple(e.terms) == G.ENERGY_NAMES and len(e.terms) == 13
    assert all(isinstance(v, float) and math.isfinite(v) and v >= 0 for v in e.terms.values())
    for k, v in e.terms.items():
        assert v == pytest.approx(0.0, abs=1e-9), k
    assert e.total == pytest.approx(0.0, abs=1e-9)
    assert e.vector().shape == (13,) and e.vector().dtype == np.float64
    assert set(G.RULES) == set(G.ENERGY_NAMES)


def test_weights_and_total():
    p = rep(base_program(), openings=base_program().openings[1:])              # 문 없음 -> G2 = 1
    e = E(p)
    assert e.terms["G2"] == pytest.approx(1.0)
    assert e.total == pytest.approx(sum(e.terms.values()))
    w = E(p, weights={"G2": 2.5})
    assert w.total == pytest.approx(2.5 * w.terms["G2"] + sum(v for k, v in w.terms.items() if k != "G2"))
    assert w.terms == e.terms and w.weights["G2"] == 2.5 and w.weights["G1"] == 1.0
    z = E(p, weights={"G2": 0.0})
    assert z.total == pytest.approx(sum(v for k, v in z.terms.items() if k != "G2"))
    with pytest.raises(ValueError):
        E(p, weights={"G99": 1.0})
    with pytest.raises(ValueError):
        E(p, weights={"G1": -1.0})
    with pytest.raises(ValueError):
        E(p, weights={"G1": float("nan")})


def test_g1_closure_open_edge_length():
    p = base_program()
    sh = dataclasses.replace(p.shell, walls_present=(True, False, True, True))      # 변 1 길이 4.0
    assert E(rep(p, shell=sh)).terms["G1"] == pytest.approx(4.0, abs=1e-9)
    sh = dataclasses.replace(p.shell, walls_present=(False, False, True, True))      # 5 + 4
    assert E(rep(p, shell=sh)).terms["G1"] == pytest.approx(9.0, abs=1e-9)
    assert E(rep(p, shell=dataclasses.replace(p.shell, floor=False))).terms["G1"] == pytest.approx(18.0, abs=1e-9)
    assert E(rep(p, shell=dataclasses.replace(p.shell, ceiling=False))).terms["G1"] == pytest.approx(18.0, abs=1e-9)
    sh = dataclasses.replace(p.shell, walls_present=(True, False, True, True), floor=False, ceiling=False)
    assert E(rep(p, shell=sh)).terms["G1"] == pytest.approx(4.0 + 36.0, abs=1e-9)


def test_g2_doors_and_openings_inside_walls():
    G = _G()
    p = base_program()
    d0, w0 = p.openings
    assert E(rep(p, openings=(w0,))).terms["G2"] == pytest.approx(1.0)                  # 문 없음
    assert E(rep(p, openings=())).terms["G2"] == pytest.approx(1.0)
    out_s = G.Opening(kind="door", edge=0, s0=4.6, s1=5.2, z0=0.0, z1=2.1)             # 변 길이 5.0 초과
    assert E(rep(p, openings=(d0, w0, out_s))).terms["G2"] == pytest.approx(1.0)
    out_z = G.Opening(kind="window", edge=1, s0=1.0, s1=2.0, z0=1.0, z1=2.7)           # 높이 2.6 초과
    assert E(rep(p, openings=(d0, w0, out_z))).terms["G2"] == pytest.approx(1.0)
    neg = G.Opening(kind="window", edge=1, s0=-0.2, s1=1.0, z0=1.0, z1=2.0)
    assert E(rep(p, openings=(d0, w0, neg))).terms["G2"] == pytest.approx(1.0)
    both = rep(p, openings=(d0, w0, out_s, out_z))
    assert E(both).terms["G2"] == pytest.approx(2.0)
    ov = G.Opening(kind="window", edge=2, s0=2.4, s1=3.4, z0=0.9, z1=2.1)               # w0(1.5..2.7)와 겹침
    assert E(rep(p, openings=(d0, w0, ov))).terms["G2"] == pytest.approx(1.0)
    touch = G.Opening(kind="window", edge=2, s0=2.7, s1=3.4, z0=0.9, z1=2.1)           # 맞닿기만: 겹침 아님
    assert E(rep(p, openings=(d0, w0, touch))).terms["G2"] == pytest.approx(0.0)
    other_edge = G.Opening(kind="window", edge=1, s0=1.5, s1=2.7, z0=0.9, z1=2.1)      # 다른 변: 겹침 아님
    assert E(rep(p, openings=(d0, w0, other_edge))).terms["G2"] == pytest.approx(0.0)


def test_g3_door_front_clearance_blocked_length():
    """문(변 0, x 3.5..4.4, 폭 0.9) 앞 1 m 띠에서 막힌 가로 길이(m). 허용 오차 0.03."""
    p = base_program()
    assert E(p).terms["G3"] == pytest.approx(0.0, abs=1e-9)
    w = slot(center=(3.8, 0.7, 1.0), size=(0.6, 0.6, 2.0))                  # x 3.5..4.1, y 0.4..1.0
    assert E(rep(p, slots=p.slots + (w,))).terms["G3"] == pytest.approx(0.6, abs=0.03)
    w = slot(center=(3.0, 0.5, 1.0), size=(1.0, 1.0, 2.0))                  # x 2.5..3.5: 띠 밖(맞닿음)
    assert E(rep(p, slots=p.slots + (w,))).terms["G3"] == pytest.approx(0.0, abs=0.03)
    w = slot(center=(3.95, 0.5, 1.0), size=(1.2, 1.8, 2.0))                 # 폭 전체를 덮음 -> 0.9
    assert E(rep(p, slots=p.slots + (w,))).terms["G3"] == pytest.approx(0.9, abs=0.03)
    w = slot(center=(3.95, 1.8, 1.0), size=(1.2, 0.6, 2.0))                 # y 1.5..2.1: 깊이 1 m 밖
    assert E(rep(p, slots=p.slots + (w,))).terms["G3"] == pytest.approx(0.0, abs=0.03)
    rug = slot(cls="rug", center=(3.95, 0.5, 0.005), size=(1.2, 0.8, 0.01))
    assert E(rep(p, slots=p.slots + (rug,))).terms["G3"] == pytest.approx(0.0, abs=0.03)    # 러그는 장애물 아님
    wall_item = slot(cls="picture", center=(4.0, 0.02, 1.5), size=(0.04, 0.8, 0.6), support="wall")
    assert E(rep(p, slots=p.slots + (wall_item,))).terms["G3"] == pytest.approx(0.0, abs=0.03)  # 바닥 지지 아님
    two = (slot(id=51, center=(3.7, 0.7, 1.0), size=(0.4, 0.4, 2.0)), slot(id=52, center=(4.2, 0.7, 1.0), size=(0.3, 0.4, 2.0)))
    assert E(rep(p, slots=p.slots + two)).terms["G3"] == pytest.approx(0.4 + 0.3, abs=0.04)   # 합집합 길이
    # 문이 둘이면 합산: 변 3(x=0 벽, 모서리 (0,4)->(0,0), s=0.5..1.3 은 y=3.5..2.7)의 문은 침대(y 2.2..3.8, x 0.1..2.1)가
    # 폭 전체를 막는다 -> 0.8. 첫 문 앞 옷장(0.6)과 합산.
    G = _G()
    d2 = G.Opening(kind="door", edge=3, s0=0.5, s1=1.3, z0=0.0, z1=2.1)
    assert E(rep(p, openings=p.openings + (d2,))).terms["G3"] == pytest.approx(0.8, abs=0.03)
    wd = slot(center=(3.8, 0.7, 1.0), size=(0.6, 0.6, 2.0))
    q = rep(p, openings=p.openings + (d2,), slots=p.slots + (wd,))
    assert E(q).terms["G3"] == pytest.approx(0.8 + 0.6, abs=0.05)
    assert E(rep(p, openings=(p.openings[1],))).terms["G3"] == pytest.approx(0.0, abs=1e-9)                 # 문 없음


def test_g4_support_floating_volume():
    p = base_program()
    bed = p.slots[0]
    vol = 2.0 * 1.6 * 0.5
    up = dataclasses.replace(bed, center=np.array([1.1, 3.0, 0.75]))                  # 바닥에서 0.5 m 뜸
    assert E(rep(p, slots=(up,) + p.slots[1:])).terms["G4"] == pytest.approx(vol, abs=1e-9)
    ok = dataclasses.replace(bed, center=np.array([1.1, 3.0, 0.26]))                  # 0.01 m: 허용(<=0.02)
    assert E(rep(p, slots=(ok,) + p.slots[1:])).terms["G4"] == pytest.approx(0.0, abs=1e-9)
    tv = p.slots[2]
    nosup = dataclasses.replace(tv, support_id=99)
    assert E(rep(p, slots=p.slots[:2] + (nosup,) + p.slots[3:])).terms["G4"] == pytest.approx(0.1 * 0.8 * 0.6, abs=1e-9)
    hover = dataclasses.replace(tv, center=np.array([4.7, 2.0, 1.4]))                  # 캐비닛 윗면(0.9) 위 0.2 m
    assert E(rep(p, slots=p.slots[:2] + (hover,) + p.slots[3:])).terms["G4"] == pytest.approx(0.048, abs=1e-9)
    off = dataclasses.replace(tv, center=np.array([4.7, 3.2, 1.2]))                    # 캐비닛 발자국 밖
    assert E(rep(p, slots=p.slots[:2] + (off,) + p.slots[3:])).terms["G4"] == pytest.approx(0.048, abs=1e-9)
    selfsup = dataclasses.replace(tv, support_id=2)
    assert E(rep(p, slots=p.slots[:2] + (selfsup,) + p.slots[3:])).terms["G4"] == pytest.approx(0.048, abs=1e-9)
    pic = p.slots[4]
    far = dataclasses.replace(pic, center=np.array([0.5, 2.0, 1.6]))                    # 벽에서 0.48 m
    assert E(rep(p, slots=p.slots[:4] + (far,))).terms["G4"] == pytest.approx(0.04 * 0.8 * 0.6, abs=1e-9)
    lamp = p.slots[3]
    low = dataclasses.replace(lamp, center=np.array([2.5, 2.0, 2.0]))
    assert E(rep(p, slots=p.slots[:3] + (low,) + p.slots[4:])).terms["G4"] == pytest.approx(0.4 * 0.4 * 0.1, abs=1e-9)
    # 위반 부피는 합산
    both = rep(p, slots=(up, p.slots[1], nosup, low, far))
    assert E(both).terms["G4"] == pytest.approx(vol + 0.048 + 0.016 + 0.0192, abs=1e-9)


def test_g5_light_causality_counts_unmotivated_lights():
    G = _G()
    p = base_program()
    base = p.lights

    def L(i, src):
        return G.Light(id=i, kind="other", pos=(1, 1, 1), rgb=(1, 1, 1), power=10.0, source=src)

    assert E(rep(p, lights=base + (L(9, None),))).terms["G5"] == pytest.approx(1.0)
    assert E(rep(p, lights=base + (L(9, ("opening", 0)),))).terms["G5"] == pytest.approx(1.0)     # 문은 광원 출처 아님
    assert E(rep(p, lights=base + (L(9, ("opening", 7)),))).terms["G5"] == pytest.approx(1.0)
    assert E(rep(p, lights=base + (L(9, ("slot", 1)),))).terms["G5"] == pytest.approx(1.0)        # 캐비닛은 램프 아님
    assert E(rep(p, lights=base + (L(9, ("slot", 99)),))).terms["G5"] == pytest.approx(1.0)
    assert E(rep(p, lights=base + (L(9, ("opening", 1)), L(10, ("slot", 3))))).terms["G5"] == pytest.approx(0.0)
    many = base + (L(9, None), L(10, ("slot", 1)), L(11, ("opening", 0)), L(12, ("slot", 3)))
    assert E(rep(p, lights=many)).terms["G5"] == pytest.approx(3.0)
    assert E(rep(p, lights=())).terms["G5"] == pytest.approx(0.0)


def test_g6_repeat_spacing_residual():
    G = _G()
    p = base_program()
    door = p.openings[0]

    def wins(centers, edge=2, hw=0.3):
        return tuple(G.Opening(kind="window", edge=edge, s0=c - hw, s1=c + hw, z0=0.9, z1=2.1) for c in centers)

    assert E(rep(p, openings=(door,) + wins([1.0, 2.0, 3.0]))).terms["G6"] == pytest.approx(0.0, abs=1e-9)
    # 간격 0.8, 1.2 -> 평균 1.0 -> 잔차 0.2 + 0.2
    assert E(rep(p, openings=(door,) + wins([1.0, 1.8, 3.0]))).terms["G6"] == pytest.approx(0.4, abs=1e-9)
    # 네 개: 간격 1.0, 1.0, 1.6 -> 평균 1.2 -> 잔차 0.2 + 0.2 + 0.4 = 0.8 (중앙값/첫 간격 기준이 아니라 평균 기준)
    assert E(rep(p, openings=(door,) + wins([0.5, 1.5, 2.5, 4.1]))).terms["G6"] == pytest.approx(0.8, abs=1e-9)
    # 입력 순서와 무관
    assert E(rep(p, openings=(door,) + wins([3.0, 1.0, 1.8]))).terms["G6"] == pytest.approx(0.4, abs=1e-9)
    # 창 2개는 패턴이 아니다
    assert E(rep(p, openings=(door,) + wins([1.0, 2.3]))).terms["G6"] == pytest.approx(0.0, abs=1e-9)
    # 변이 다르면 합치지 않는다 (변 2에 2개 + 변 1에 1개)
    mixed = wins([1.0, 2.3]) + wins([2.0], edge=1)
    assert E(rep(p, openings=(door,) + mixed)).terms["G6"] == pytest.approx(0.0, abs=1e-9)
    # 변마다 따로 잔차를 더한다
    two_edges = wins([1.0, 1.8, 3.0]) + wins([0.8, 1.6, 3.2], edge=1)           # 0.4 + (0.8, 1.6 -> 0.8, 1.6; 평균 1.2 -> 0.8)
    expect = 0.4 + (abs(0.8 - 1.2) + abs(1.6 - 1.2))
    assert E(rep(p, openings=(door,) + two_edges)).terms["G6"] == pytest.approx(expect, abs=1e-9)
    # 문은 G6 대상이 아니다(창만 센다)
    doors = tuple(G.Opening(kind="door", edge=2, s0=c - 0.3, s1=c + 0.3, z0=0.0, z1=2.1) for c in (1.0, 1.8, 3.0))
    assert E(rep(p, openings=(door,) + doors)).terms["G6"] == pytest.approx(0.0, abs=1e-9)


def test_g7_symmetry_residual():
    from buc.invent.grammar.core import EnergyContext
    p = base_program()

    def chairs(xs, y=2.0, ids=(60, 61, 62)):
        return tuple(slot(id=i, cls="chair", center=(x, y, 0.45), size=(0.4, 0.4, 0.9)) for i, x in zip(ids, xs))

    base = rep(p, slots=chairs([1.0, -1.0]))
    ctx = EnergyContext(symmetry=("x", 0.0))
    assert E(base, ctx).terms["G7"] == pytest.approx(0.0, abs=1e-9)
    assert E(base).terms["G7"] == pytest.approx(0.0, abs=1e-9)                              # 축이 없으면 0
    shifted = rep(p, slots=chairs([1.0, -0.7]))
    assert E(shifted, ctx).terms["G7"] == pytest.approx(0.6, abs=1e-9)                       # 0.3 + 0.3
    single = rep(p, slots=chairs([1.0]))
    assert E(single, ctx).terms["G7"] == pytest.approx(2.0, abs=1e-9)                        # 자기 자신까지 2.0
    on_axis = rep(p, slots=chairs([0.0]))
    assert E(on_axis, ctx).terms["G7"] == pytest.approx(0.0, abs=1e-9)
    # 다른 클래스는 대칭 짝이 될 수 없다
    mixed = rep(p, slots=(chairs([1.0])[0], slot(id=61, cls="plant", center=(-1.0, 2.0, 0.6), size=(0.4, 0.4, 1.2))))
    assert E(mixed, ctx).terms["G7"] == pytest.approx(2.0 + 2.0, abs=1e-9)
    # 축 선택과 오프셋: y = 2 에 대한 거울
    ys = rep(p, slots=tuple(slot(id=70 + i, cls="chair", center=(1.0, y, 0.45), size=(0.4, 0.4, 0.9))
                            for i, y in enumerate([1.0, 3.0])))
    assert E(ys, EnergyContext(symmetry=("y", 2.0))).terms["G7"] == pytest.approx(0.0, abs=1e-9)
    assert E(ys, EnergyContext(symmetry=("x", 1.0))).terms["G7"] == pytest.approx(0.0, abs=1e-9)   # x=1 평면 위의 의자 둘
    assert E(ys, EnergyContext(symmetry=("y", 1.0))).terms["G7"] > 0.5                              # 잘못된 축
    assert E(rep(p, slots=()), ctx).terms["G7"] == pytest.approx(0.0, abs=1e-9)


def test_g8_style_cluster_and_g9_g12_are_inert_in_v0():
    from buc.invent.grammar.core import EnergyContext
    p = base_program()
    allowed = frozenset({"bed", "cabinet", "tv", "lamp", "picture"})
    ctx = EnergyContext(allowed_classes=allowed)
    assert E(p, ctx).terms["G8"] == pytest.approx(0.0)
    sofa = slot(id=80, cls="sofa", center=(2.0, 1.5, 0.4), size=(0.9, 1.8, 0.8))
    desk = slot(id=81, cls="desk", center=(3.0, 3.0, 0.4), size=(0.6, 1.2, 0.76))
    assert E(rep(p, slots=p.slots + (sofa,)), ctx).terms["G8"] == pytest.approx(1.0)
    assert E(rep(p, slots=p.slots + (sofa, desk)), ctx).terms["G8"] == pytest.approx(2.0)
    assert E(rep(p, slots=p.slots + (sofa, desk))).terms["G8"] == pytest.approx(0.0)         # allowed=None -> 0
    e = E(rep(p, slots=p.slots + (sofa, desk)), ctx)
    for k in ("G9", "G10", "G11", "G12"):                                                    # v0: 방에는 해당 없음
        assert e.terms[k] == 0.0


def _persp(w=640, h=360, fov=60.0):
    return make_K(w, h, fov)


def _cam(eye, target, w=640, h=360, fov=60.0):
    from buc.invent.grammar.core import CameraSpec
    return CameraSpec(c2w=look(eye, target), K=_persp(w, h, fov), width=w, height=h)


def _person(y, yaw, x=0.0, i=90):
    return slot(id=i, cls="person", center=(x, y, 0.85), size=(0.5, 0.5, 1.7), yaw=yaw)


def test_g13_invented_people_face_the_camera_rule():
    from buc.invent.grammar.core import EnergyContext
    p0 = rep(base_program(), slots=())
    cam = _cam((0, 0, 1.5), (0, 1, 1.5))
    ctx = EnergyContext(cameras=(cam,))
    fy = cam.K[1, 1]
    # 카메라를 향한 사람(front = -y, 카메라는 y=0): 3 m 거리 -> h_f = 0.22*fy/3/360 ~ 0.11 > 1%
    assert 0.22 * fy / 3.0 / 360 > 0.05
    toward = rep(p0, slots=(_person(3.0, -math.pi / 2),))
    away = rep(p0, slots=(_person(3.0, math.pi / 2),))
    assert E(toward, ctx).terms["G13"] == pytest.approx(1.0)
    assert E(away, ctx).terms["G13"] == pytest.approx(0.0)                                    # 등지면 OK
    assert E(toward).terms["G13"] == pytest.approx(0.0)                                       # 카메라 없음 -> 0
    # 멀면 얼굴이 1% 이하: 20 m -> 0.0169 (위반), 40 m -> 0.0085 (OK)
    assert E(rep(p0, slots=(_person(20.0, -math.pi / 2),)), ctx).terms["G13"] == pytest.approx(1.0)
    assert 0.22 * fy / 40.0 / 360 < 0.01
    assert E(rep(p0, slots=(_person(40.0, -math.pi / 2),)), ctx).terms["G13"] == pytest.approx(0.0)
    # 카메라 뒤의 사람, 화면 밖의 사람은 세지 않는다
    assert E(rep(p0, slots=(_person(-3.0, math.pi / 2),)), ctx).terms["G13"] == pytest.approx(0.0)
    assert E(rep(p0, slots=(_person(3.0, -math.pi / 2, x=30.0),)), ctx).terms["G13"] == pytest.approx(0.0)
    # 사람이 아닌 슬롯은 무시
    assert E(rep(p0, slots=(slot(id=91, cls="plant", center=(0, 3, 0.6), size=(0.4, 0.4, 1.2), yaw=-math.pi / 2),)),
             ctx).terms["G13"] == pytest.approx(0.0)
    # 위반 슬롯 수: 두 사람이면 2, 같은 사람이 카메라 두 대에서 위반해도 1
    two = rep(p0, slots=(_person(3.0, -math.pi / 2, x=-0.5, i=90), _person(3.5, -math.pi / 2, x=0.5, i=91)))
    assert E(two, ctx).terms["G13"] == pytest.approx(2.0)
    cam2 = _cam((1.0, 0, 1.5), (0, 3, 1.5))
    assert E(toward, EnergyContext(cameras=(cam, cam2))).terms["G13"] == pytest.approx(1.0)
    # 한 대에서만 위반: 등진 카메라 + 마주 보는 카메라 -> 1
    cam_front = _cam((0, 6, 1.5), (0, 3, 1.5))
    assert E(away, EnergyContext(cameras=(cam_front,))).terms["G13"] == pytest.approx(1.0)
    assert E(away, EnergyContext(cameras=(cam, cam_front))).terms["G13"] == pytest.approx(1.0)


def test_energy_invariances_on_generated_rooms(rooms):
    """슬롯 순서 불변, 평행이동 불변, 입력 불변, 음수 없음."""
    G = _G()
    for p in rooms[:12]:
        h = p.content_hash()
        e = E(p)
        assert p.content_hash() == h
        assert all(v >= 0 and math.isfinite(v) for v in e.terms.values())
        e_rev = E(rep(p, slots=p.slots[::-1], openings=p.openings, lights=p.lights[::-1]))
        for k in G.ENERGY_NAMES:
            assert e_rev.terms[k] == pytest.approx(e.terms[k], abs=1e-9), k
        dx = np.array([13.7, -4.2])
        shell = dataclasses.replace(p.shell, corners=p.shell.corners + dx)
        slots = tuple(dataclasses.replace(s, center=s.center + np.array([dx[0], dx[1], 0.0])) for s in p.slots)
        lights = tuple(dataclasses.replace(l, pos=l.pos + np.array([dx[0], dx[1], 0.0])) for l in p.lights)
        e_sh = E(rep(p, shell=shell, slots=slots, lights=lights))
        for k in G.ENERGY_NAMES:
            assert e_sh.terms[k] == pytest.approx(e.terms[k], abs=1e-7), k


def test_energy_detects_damage_to_generated_rooms(rooms):
    """생성기가 만든 방을 망가뜨리면 해당 항이 반응한다 (센서가 살아 있다는 증거)."""
    G = _G()
    hit = {"G1": 0, "G2": 0, "G4": 0, "G5": 0}
    for p in rooms[:10]:
        assert E(p).total == pytest.approx(0.0, abs=1e-9)
        if E(rep(p, shell=dataclasses.replace(p.shell, walls_present=(False,) + p.shell.walls_present[1:]))
             ).terms["G1"] > 0:
            hit["G1"] += 1
        if E(rep(p, openings=tuple(o for o in p.openings if o.kind != "door"))).terms["G2"] >= 1:
            hit["G2"] += 1
        s0 = p.slots[0]
        lifted = dataclasses.replace(s0, center=s0.center + np.array([0, 0, 0.5]))
        if E(rep(p, slots=(lifted,) + p.slots[1:])).terms["G4"] > 0:
            hit["G4"] += 1
        if E(rep(p, lights=p.lights + (G.Light(id=99, kind="other", pos=(0, 0, 1), rgb=(1, 1, 1), power=1.0),))
             ).terms["G5"] == 1:
            hit["G5"] += 1
    assert all(v == 10 for v in hit.values()), hit
