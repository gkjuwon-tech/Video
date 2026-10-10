"""PRD-01: DAG 계획, 레지스트리, 키·시드·root 공식."""
import random

import pytest

from buc.core import contracts as C
from buc.core import det
from buc.pipeline import spine as S
from prd01_helpers import STUB_FNS, stub_registry


def test_spine_stages_derived_from_contracts():
    assert S.SPINE_STAGES == ("S-1", "S2", "S8", "S8.5", "S9-10", "S11")
    assert S.SPINE_STAGES == tuple(s for s in C.STAGES if s in C.STAGE_SIGNATURES)


def test_plan_resolves_producers():
    sp = S.Spine(stub_registry())
    nodes = sp.plan()
    assert [n.stage for n in nodes] == list(S.SPINE_STAGES)
    by = {n.stage: n for n in nodes}
    assert by["S-1"].inputs == {"image": "input"}
    assert by["S2"].inputs == {"read": "S-1"}
    assert by["S8"].inputs == {"world": "S2", "read": "S-1"}
    assert by["S8.5"].inputs == {"world": "S2", "read": "S-1"}
    assert by["S9-10"].inputs == {"world": "S8.5", "shot": "S8"}
    assert by["S11"].inputs == {"frames": "S9-10", "read": "S-1", "world": "S8.5", "shot": "S8"}
    assert by["S9-10"].outputs == ("frames",)
    for n in nodes:
        assert n.impl is not None and n.impl.kind == "stub" and n.impl.stage == n.stage


def test_plan_skip_and_until():
    sp = S.Spine(stub_registry())
    by = {n.stage: n for n in sp.plan(skip=("S8.5",))}
    assert "S8.5" not in by
    assert by["S9-10"].inputs["world"] == "S2"
    assert by["S11"].inputs["world"] == "S2"
    assert [n.stage for n in sp.plan(until="S8")] == ["S-1", "S2", "S8"]
    assert [n.stage for n in sp.plan(skip=("S11",))] == ["S-1", "S2", "S8", "S8.5", "S9-10"]


def test_plan_rejects_orphans_and_unknown():
    sp = S.Spine(stub_registry())
    with pytest.raises(S.SpineError):
        sp.plan(skip=("S2",))          # S8 의 world 생산자가 없다
    with pytest.raises(S.SpineError):
        sp.plan(skip=("S-1",))
    with pytest.raises(ValueError):
        sp.plan(until="S5")            # 서명 없는 스테이지
    with pytest.raises(ValueError):
        sp.plan(skip=("S99",))


def test_impl_validation():
    f = STUB_FNS["S2"]
    with pytest.raises(ValueError):
        S.Impl(stage="S4", kind="stub", name="x", version="0", fn=f)     # v0 서명 없음
    with pytest.raises(ValueError):
        S.Impl(stage="S2", kind="fake", name="x", version="0", fn=f)
    with pytest.raises(ValueError):
        S.Impl(stage="S2", kind="stub", name="", version="0", fn=f)
    with pytest.raises(ValueError):
        S.Impl(stage="S2", kind="stub", name="x", version="", fn=f)
    with pytest.raises(ValueError):
        S.Impl(stage="S2", kind="stub", name="x", version="0", fn=None)
    impl = S.Impl(stage="S2", kind="real", name="wld", version="1.2", fn=f)
    assert impl.ref == "real:wld@1.2"


def test_registry_register_and_resolve():
    reg = S.Registry()
    stub = S.Impl(stage="S2", kind="stub", name="flat", version="0", fn=STUB_FNS["S2"])
    real = S.Impl(stage="S2", kind="real", name="depth", version="1", fn=STUB_FNS["S2"])
    assert reg.register(stub) is stub
    with pytest.raises(ValueError):
        reg.register(S.Impl(stage="S2", kind="stub", name="other", version="0", fn=STUB_FNS["S2"]))
    assert reg.resolve("S2") is stub                 # real 없으면 stub
    assert reg.resolve("S2", prefer="stub") is stub
    reg.register(real)
    assert reg.resolve("S2") is real
    assert reg.resolve("S2", prefer="stub") is stub
    assert reg.get("S2", "real") is real and reg.get("S8", "stub") is None
    stub2 = S.Impl(stage="S2", kind="stub", name="flat", version="1", fn=STUB_FNS["S2"])
    reg.register(stub2, replace=True)
    assert reg.get("S2", "stub") is stub2
    with pytest.raises(S.SpineError):
        reg.resolve("S8")
    with pytest.raises(ValueError):
        reg.resolve("S2", prefer="best")


def test_plan_missing_impl_is_spine_error():
    reg = stub_registry()
    sp = S.Spine(S.Registry())
    with pytest.raises(S.SpineError):
        sp.plan()
    assert len(S.Spine(reg).plan()) == 6


def test_stage_seed_and_params():
    assert S.stage_seed(7, "S2") == det.derive_seed(7, "buc.spine.v1", "S2")
    assert S.stage_seed(7, "S2") != S.stage_seed(7, "S8")
    assert S.stage_seed(7, "S2") != S.stage_seed(8, "S2")
    p = S.stage_params("S8", 3, {"S8": {"n": 5}, "S2": {"gain": 2.0}})
    assert p == {"n": 5, "_seed": S.stage_seed(3, "S8")}
    assert S.stage_params("S2", 3) == {"_seed": S.stage_seed(3, "S2")}
    with pytest.raises(ValueError):
        S.stage_params("S8", 0, {"S8": {"_seed": 1}})


def test_stage_cache_key_equals_stageio_formula():
    impl = S.Impl(stage="S8", kind="real", name="camfm", version="2", fn=STUB_FNS["S8"])
    ih = {"world": "a" * 64, "read": "b" * 64}
    p = {"n": 4, "_seed": 11}
    key = S.stage_cache_key("S8", impl, ih, p)
    sio = C.StageIO(stage="S8", impl_kind="real", impl_name="camfm", impl_version="2", inputs=ih, outputs={},
                    params=p)
    assert key == sio.cache_key and len(key) == 64
    impl2 = S.Impl(stage="S8", kind="real", name="camfm", version="3", fn=STUB_FNS["S8"])
    assert S.stage_cache_key("S8", impl2, ih, p) != key
    assert S.stage_cache_key("S8", impl, ih, {"n": 4.0, "_seed": 11}) != key       # int ≠ float
    assert S.stage_cache_key("S8", impl, {"read": "b" * 64, "world": "a" * 64}, p) == key


def _sio(stage, out):
    return C.StageIO(stage=stage, impl_kind="stub", impl_name="n", impl_version="0", inputs={"x": "1" * 64},
                     outputs=out, params={})


def test_run_root_oracle_and_order_independence():
    ios = {s: _sio(s, {"o": format(i, "064x")}) for i, s in enumerate(S.SPINE_STAGES)}
    want = det.merkle_root_keyed(
        {s: det.hash_parts(io.cache_key, C.content_hash(io.outputs)) for s, io in ios.items()}).hex()
    assert S.run_root(ios) == want
    items = list(ios.items())
    for k in range(4):
        random.Random(k).shuffle(items)
        assert S.run_root(dict(items)) == want
    assert S.run_root({}) == det.EMPTY_ROOT.hex()
    ios2 = dict(ios)
    ios2["S8"] = _sio("S8", {"o": "f" * 64})
    assert S.run_root(ios2) != want
