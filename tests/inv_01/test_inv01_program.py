"""INV-01 장면 프로그램: 상수, 데이터 형태, 검증, 직렬화/해시, 부작용 없음."""
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from inv01_helpers import base_program

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _G():
    from buc.invent.grammar import core
    return core


def test_constants_are_frozen_by_contract():
    G = _G()
    assert G.SLOT_CLASSES[:16] == ("bed", "sofa", "table", "chair", "desk", "shelf", "cabinet", "wardrobe", "lamp",
                                   "plant", "tv", "person", "rug", "picture", "stool", "counter")
    assert len(set(G.SLOT_CLASSES)) == len(G.SLOT_CLASSES)
    assert G.FURNITURE_CLASSES == ("bed", "sofa", "table", "desk", "shelf", "cabinet", "wardrobe", "counter")
    assert (G.SEM_VOID, G.SEM_FLOOR, G.SEM_WALL, G.SEM_CEILING, G.SEM_DOOR, G.SEM_WINDOW) == (0, 1, 2, 3, 4, 5)
    assert G.SEM_SLOT_BASE == 8
    assert G.sem_id("bed") == 8 and G.sem_id("counter") == 8 + 15
    with pytest.raises(ValueError):
        G.sem_id("nonexistent")
    assert G.ENERGY_NAMES == tuple(f"G{i}" for i in range(1, 14))
    assert issubclass(G.GrammarError, ValueError)
    assert isinstance(G.HAS_NUMBA, bool)
    assert set(G.SUPPORTS) == {"floor", "wall", "slot", "ceiling"}


def test_base_program_is_valid_and_arrays_canonical():
    G = _G()
    p = base_program()
    p.validate()
    assert p.shell.corners.dtype == np.float64 and p.shell.corners.shape == (4, 2)
    assert p.shell.corners.flags["C_CONTIGUOUS"]
    for s in p.slots:
        assert s.center.dtype == np.float64 and s.center.shape == (3,)
        assert s.size.dtype == np.float64 and s.size.shape == (3,)
    assert p.shell.n_edges == 4
    np.testing.assert_allclose(p.shell.edge_lengths(), [5, 4, 5, 4], atol=1e-12)
    assert p.shell.area() == pytest.approx(20.0, abs=1e-12)
    assert p.shell.perimeter() == pytest.approx(18.0, abs=1e-12)
    assert p.slot_by_id(3).cls == "lamp"
    with pytest.raises(KeyError):
        p.slot_by_id(99)
    s = p.slot_by_id(0)
    assert s.z0 == pytest.approx(0.0, abs=1e-12) and s.z1 == pytest.approx(0.5, abs=1e-12)
    fp = s.footprint()
    assert fp.shape == (4, 2)
    assert fp[:, 0].min() == pytest.approx(0.1, abs=1e-12) and fp[:, 0].max() == pytest.approx(2.1, abs=1e-12)
    assert fp[:, 1].min() == pytest.approx(2.2, abs=1e-12) and fp[:, 1].max() == pytest.approx(3.8, abs=1e-12)
    assert isinstance(G.SceneProgram, type)


def test_footprint_yaw_convention_local_x_is_front():
    G = _G()
    s = G.Slot(id=0, cls="sofa", center=(0, 0, 0.4), size=(1.0, 2.0, 0.8), yaw=np.pi / 2)
    fp = s.footprint()
    # yaw 90도: 로컬 x(두께 1.0)가 세계 y로, 로컬 y(폭 2.0)가 세계 -x로 간다.
    assert np.ptp(fp[:, 0]) == pytest.approx(2.0, abs=1e-12)
    assert np.ptp(fp[:, 1]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kw", [
    dict(size=(0.0, 1, 1)), dict(size=(1, -1, 1)), dict(center=(np.nan, 0, 0)), dict(center=(0, 0)),
    dict(cls="dragon"), dict(support="ceilling"), dict(yaw=float("inf")), dict(material="unobtainium"),
])
def test_slot_validation_rejects(kw):
    G = _G()
    args = dict(id=0, cls="bed", center=(0, 0, 0.3), size=(1, 1, 0.6), yaw=0.0)
    args.update(kw)
    with pytest.raises(G.GrammarError):
        G.Slot(**args)


def test_shell_validation():
    G = _G()
    ok = [[0, 0], [4, 0], [4, 3], [0, 3]]
    G.Shell(corners=ok, height=2.5)
    with pytest.raises(G.GrammarError):
        G.Shell(corners=ok[::-1], height=2.5)          # 시계 방향
    with pytest.raises(G.GrammarError):
        G.Shell(corners=ok[:2], height=2.5)
    with pytest.raises(G.GrammarError):
        G.Shell(corners=ok, height=0.0)
    with pytest.raises(G.GrammarError):
        G.Shell(corners=ok, height=2.5, walls_present=(True, True))
    with pytest.raises(G.GrammarError):
        G.Shell(corners=[[0, 0], [np.nan, 0], [4, 3]], height=2.5)


def test_opening_light_program_validation():
    G = _G()
    with pytest.raises(G.GrammarError):
        G.Opening(kind="gate", edge=0, s0=0, s1=1, z0=0, z1=2)
    with pytest.raises(G.GrammarError):
        G.Opening(kind="door", edge=0, s0=1, s1=1, z0=0, z1=2)
    with pytest.raises(G.GrammarError):
        G.Opening(kind="door", edge=0, s0=0, s1=1, z0=2, z1=1)
    with pytest.raises(G.GrammarError):
        G.Light(id=0, kind="lamp", pos=(0, 0, 1), rgb=(-1, 0, 0), power=1.0)
    p = base_program()
    with pytest.raises(G.GrammarError):                      # 존재하지 않는 변
        dataclasses.replace(p, openings=p.openings + (G.Opening(kind="door", edge=4, s0=0, s1=1, z0=0, z1=2),))
    with pytest.raises(G.GrammarError):                      # 슬롯 id 중복
        dataclasses.replace(p, slots=p.slots + (p.slots[0],))
    with pytest.raises(G.GrammarError):
        dataclasses.replace(p, matter={"floor": "lava"})
    # 규칙 위반(범위 밖 개구부, 출처 없는 광원)은 구성 가능해야 한다: 에너지가 점수를 매긴다.
    bad = dataclasses.replace(p, openings=p.openings + (G.Opening(kind="door", edge=0, s0=4.6, s1=5.5, z0=0, z1=2),))
    assert len(bad.openings) == 3


def test_kw_only_frozen_no_positional():
    G = _G()
    with pytest.raises(TypeError):
        G.Slot(0, "bed", (0, 0, 0.3), (1, 1, 0.6))
    p = base_program()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.seed = 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.slots[0].yaw = 1.0


def test_dict_roundtrip_hash_and_json():
    G = _G()
    p = base_program()
    d = p.to_dict()
    assert d["__contract__"] == "SceneProgram"
    s = json.dumps(d)
    q = G.SceneProgram.from_dict(json.loads(s))
    assert q.content_hash() == p.content_hash()
    assert len(p.content_hash()) == 64 and all(c in "0123456789abcdef" for c in p.content_hash())
    assert q.to_dict() == d
    for a, b in zip(p.slots, q.slots):
        np.testing.assert_array_equal(a.center, b.center)       # 비트 무손실
    bad = dict(d)
    bad["__contract__"] = "Other"
    with pytest.raises(G.GrammarError):
        G.SceneProgram.from_dict(bad)


def test_hash_is_sensitive_to_one_ulp_and_to_content():
    G = _G()
    p = base_program()
    s0 = p.slots[0]
    c = s0.center.copy()
    c[0] = np.nextafter(c[0], np.inf)
    p2 = dataclasses.replace(p, slots=(dataclasses.replace(s0, center=c),) + p.slots[1:])
    assert p2.content_hash() != p.content_hash()
    p3 = dataclasses.replace(p, seed=1)
    assert p3.content_hash() != p.content_hash()
    p4 = dataclasses.replace(p, matter={"floor": "tile", "wall": "plaster", "ceiling": "paint_white"})
    assert p4.content_hash() != p.content_hash()
    assert dataclasses.replace(p).content_hash() == p.content_hash()


def test_import_is_light_and_deterministic_across_hash_seeds():
    """torch 비로드, numba JIT 없이 import, PYTHONHASHSEED가 달라도 같은 방 해시."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from buc.invent.grammar.room import generate_room\n"
        "from buc.invent.grammar import core\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "print(generate_room(7).content_hash(), generate_room(8).content_hash())\n" % ROOT
    )
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT,
                           timeout=50)
        assert r.returncode == 0, r.stderr[-800:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] and len(outs[0].split()) == 2
    from buc.invent.grammar.room import generate_room
    assert outs[0].split()[0] == generate_room(7).content_hash()


def test_no_global_rng_or_input_mutation():
    import random
    G = _G()
    from buc.invent.grammar.room import generate_room
    from buc.invent.grammar.core import grammar_energy, raycast
    from inv01_helpers import look, make_K
    np_state = np.random.get_state()
    py_state = random.getstate()
    p = generate_room(3)
    h = p.content_hash()
    c2w = look((p.shell.corners[:, 0].mean(), p.shell.corners[:, 1].mean(), 1.5),
               (p.shell.corners[0, 0], p.shell.corners[0, 1], 1.5))
    raycast(p, c2w=c2w, K=make_K(16, 12), width=16, height=12)
    grammar_energy(p)
    assert p.content_hash() == h
    a, b = np.random.get_state(), np_state
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    assert random.getstate() == py_state
    assert G.SceneProgram.from_dict(p.to_dict()).content_hash() == h
