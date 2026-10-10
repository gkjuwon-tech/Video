"""스테이지 이음매: STAGE_SIGNATURES·check_stage_io·make_stage_io·cache_key."""
import numpy as np
import pytest

from buc.core import contracts as C
from fnd02_fixtures import make_image, make_obs, make_prov, make_read, make_shot, make_world


def test_signatures_shape():
    sig = C.STAGE_SIGNATURES
    assert sig["S-1"] == ({"image": C.InputImage}, {"read": C.ReadPacket})
    assert sig["S2"] == ({"read": C.ReadPacket}, {"world": C.World})
    assert sig["S8"] == ({"world": C.World, "read": C.ReadPacket}, {"shot": C.Shot})
    assert sig["S8.5"] == ({"world": C.World, "read": C.ReadPacket}, {"world": C.World})
    assert sig["S9-10"] == ({"world": C.World, "shot": C.Shot}, {"frames": C.Frames})
    assert sig["S11"] == ({"frames": C.Frames, "read": C.ReadPacket, "world": C.World, "shot": C.Shot},
                          {"kpi": C.KPI})
    assert all(s in C.STAGES for s in sig)


def test_full_chain_passes(chain):
    c = chain
    C.check_stage_io("S-1", {"image": c["image"]}, {"read": c["read"]})
    C.check_stage_io("S2", {"read": c["read"]}, {"world": c["world"]})
    C.check_stage_io("S8", {"world": c["world"], "read": c["read"]}, {"shot": c["shot"]})
    C.check_stage_io("S8.5", {"world": c["world"], "read": c["read"]}, {"world": c["world"]})
    C.check_stage_io("S9-10", {"world": c["world"], "shot": c["shot"]}, {"frames": c["frames"]})
    C.check_stage_io("S11", {k: c[k] for k in ("frames", "read", "world", "shot")}, {"kpi": c["kpi"]})


def test_structural_rejects(chain):
    c = chain
    with pytest.raises(C.ContractError):
        C.check_stage_io("S3", {"read": c["read"]}, {"world": c["world"]})  # 서명 없음
    with pytest.raises(C.ContractError):
        C.check_stage_io("S2", {}, {"world": c["world"]})  # 누락
    with pytest.raises(C.ContractError):
        C.check_stage_io("S2", {"read": c["read"], "x": 1}, {"world": c["world"]})  # 초과
    with pytest.raises(C.ContractError):
        C.check_stage_io("S2", {"read": c["world"]}, {"world": c["world"]})  # 타입


def test_cross_checks_s_minus_1(chain):
    img = chain["image"]
    with pytest.raises(C.ContractError):
        C.check_stage_io("S-1", {"image": img}, {"read": make_read(img, input_hash="0" * 64)})
    with pytest.raises(C.ContractError):  # taint 상실
        C.check_stage_io("S-1", {"image": img}, {"read": make_read(img, prov=make_prov("ENGINE"))})
    other = make_image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(C.ContractError):  # obs 크기 != image 크기
        C.check_stage_io("S-1", {"image": other}, {"read": make_read(other)})


def test_cross_checks_s2_s8_s85_s910(chain):
    c = chain
    read2 = make_read(scale_s=2.0)
    with pytest.raises(C.ContractError):  # read_hash 불일치
        C.check_stage_io("S2", {"read": read2}, {"world": c["world"]})
    with pytest.raises(C.ContractError):  # world.obs != read.obs
        C.check_stage_io("S2", {"read": c["read"]}, {"world": make_world(c["read"], obs=make_obs(k1=0.02))})
    with pytest.raises(C.ContractError):  # taint 상실
        C.check_stage_io("S2", {"read": c["read"]}, {"world": make_world(c["read"], prov=make_prov("ENGINE"))})
    moved = c["shot"].c2w.copy()
    moved[0, 1, 3] += 0.05
    with pytest.raises(C.ContractError):  # 첫 프레임 = 관측 카메라
        C.check_stage_io("S8", {"world": c["world"], "read": c["read"]}, {"shot": make_shot(c["read"].obs, c2w=moved)})
    with pytest.raises(C.ContractError):  # S8.5는 obs를 바꾸지 못한다
        C.check_stage_io("S8.5", {"world": c["world"], "read": c["read"]},
                         {"world": make_world(c["read"], obs=make_obs(t=np.array([0.0, 0.0, 0.0])))})
    from fnd02_fixtures import make_frames
    short = make_frames(make_shot(c["read"].obs, n=3))
    with pytest.raises(C.ContractError):  # 프레임 수 != shot
        C.check_stage_io("S9-10", {"world": c["world"], "shot": c["shot"]}, {"frames": short})


def test_make_stage_io_and_cache_key(chain):
    c = chain
    io = C.make_stage_io("S2", "stub", "flat_world", "0.0.1", {"read": c["read"]}, {"world": c["world"]},
                         params={"cards": 3}, wall_s=1.5)
    assert io.inputs == {"read": c["read"].content_hash()} and io.outputs == {"world": c["world"].content_hash()}
    assert io.impl_kind == "stub" and len(io.cache_key) == 64
    expect = C.content_hash((C.CONTRACT_VERSION, "S2", "stub", "flat_world", "0.0.1", io.inputs, {"cards": 3}))
    assert io.cache_key == expect
    # outputs·wall_s·fallback은 캐시 키에 들어가지 않는다.
    io2 = C.StageIO(stage="S2", impl_kind="stub", impl_name="flat_world", impl_version="0.0.1", inputs=io.inputs,
                    outputs={"world": "f" * 64}, params={"cards": 3}, wall_s=99.0, fallback="R3")
    assert io2.cache_key == io.cache_key and io2.content_hash() != io.content_hash()
    keys = {C.StageIO(stage="S2", impl_kind=k, impl_name=n, impl_version=v, inputs=i, outputs={}, params=p).cache_key
            for k, n, v, i, p in [("real", "flat_world", "0.0.1", io.inputs, {"cards": 3}),
                                  ("stub", "other", "0.0.1", io.inputs, {"cards": 3}),
                                  ("stub", "flat_world", "0.0.2", io.inputs, {"cards": 3}),
                                  ("stub", "flat_world", "0.0.1", {"read": "0" * 64}, {"cards": 3}),
                                  ("stub", "flat_world", "0.0.1", io.inputs, {"cards": 4})]}
    assert io.cache_key not in keys and len(keys) == 5
    with pytest.raises(C.ContractError):
        C.StageIO(stage="S2", impl_kind="mock", impl_name="x", impl_version="0", inputs={}, outputs={})
    with pytest.raises(C.ContractError):
        C.StageIO(stage="S99", impl_kind="stub", impl_name="x", impl_version="0", inputs={}, outputs={})
    with pytest.raises(C.ContractError):  # make_stage_io는 check_stage_io를 먼저 돈다
        C.make_stage_io("S2", "stub", "flat_world", "0.0.1", {"read": make_read(scale_s=3.0)},
                        {"world": c["world"]})
