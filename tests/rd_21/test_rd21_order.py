"""RD-21 실행 순서·needs 가시성·미등록 단계."""
import numpy as np
import pytest

from buc.core import contracts as C
from buc.read import pipeline as P
from rd21_helpers import ALL_SIDS, classic_lin, fake_outputs, fake_steps, make_image, status_map


def test_all_steps_run_in_exec_order():
    calls = []
    img = make_image()
    res = P.run_read(img, steps=fake_steps(calls))
    assert calls == list(P.EXEC_ORDER)
    assert tuple(r.sid for r in res.records) == P.EXEC_ORDER
    assert all(r.status == "ok" for r in res.records)
    assert all(r.error is None for r in res.records)
    assert res.packet.fallbacks == ()
    assert isinstance(res.packet, C.ReadPacket)
    C.check_stage_io("S-1", {"image": img}, {"read": res.packet})


def test_step_registration_order_is_irrelevant():
    calls = []
    img = make_image()
    steps = tuple(reversed(fake_steps(calls)))
    res = P.run_read(img, steps=steps)
    assert calls == list(P.EXEC_ORDER)
    base = P.run_read(img, steps=fake_steps())
    assert res.packet.content_hash() == base.packet.content_hash()


def test_unregistered_steps_are_skipped():
    calls = []
    seen = {}

    def r6(state, params):
        seen["R1"] = state.status("R1")
        seen["R5"] = state.status("R5")
        seen["R1_K"] = np.array(state["R1"]["K"])
        return fake_outputs("R6", state)

    img = make_image()
    res = P.run_read(img, steps=fake_steps(calls, sids=("R0", "R6", "R2"), override={"R6": r6}))
    assert calls == ["R0", "R2", "R6"]
    st = status_map(res)
    assert st["R0"] == "ok" and st["R6"] == "ok" and st["R2"] == "ok"
    for sid in ALL_SIDS:
        if sid not in ("R0", "R6", "R2"):
            assert st[sid] == "skipped"
            rec = [r for r in res.records if r.sid == sid][0]
            assert rec.name is None and rec.version is None and rec.key is None
    assert seen["R1"] == "skipped" and seen["R5"] == "skipped"
    assert seen["R1_K"].shape == (3, 3)  # skipped 단계도 폴백값을 낸다
    # 핵심 단계만 fallbacks 에 남는다(R2 등 선택 단계는 meta 에만)
    assert res.packet.fallbacks == ("R1:skipped", "R5:skipped")
    steps_meta = res.packet.meta["read"]["steps"]
    assert set(steps_meta) == set(ALL_SIDS)
    assert steps_meta["R3"] == {"impl": None, "status": "skipped"}
    assert steps_meta["R6"] == {"impl": "fake_R6@1", "status": "ok"}


def test_step_sees_only_its_needs():
    def r6(state, params):
        _ = state["R0"]  # R6 needs = (R1, R5)
        return fake_outputs("R6", state)

    res = P.run_read(make_image(), steps=fake_steps(override={"R6": r6}))
    assert status_map(res)["R6"] == "fallback"
    assert "R6:fake_R6@1:KeyError" in res.packet.fallbacks


def test_needs_outputs_are_the_upstream_values():
    got = {}

    def r1(state, params):
        got["lin"] = np.array(state["R0"]["image_lin"])
        got["status"] = state.status("R0")
        got["params"] = params
        return fake_outputs("R1", state)

    img = make_image()
    P.run_read(img, steps=fake_steps(override={"R1": r1}), params={"R1": {"a": [1, 2]}})
    np.testing.assert_array_equal(got["lin"], classic_lin(img))
    assert got["status"] == "ok"
    assert got["params"] == {"a": [1, 2]}


def test_params_reach_only_their_step():
    got = {}

    def mk(sid):
        def fn(state, params):
            got[sid] = dict(params)
            return fake_outputs(sid, state)
        return fn

    P.run_read(make_image(), steps=fake_steps(override={s: mk(s) for s in ALL_SIDS}),
               params={"R4": {"x": 1}, "R9": {"y": 2}})
    assert got["R4"] == {"x": 1} and got["R9"] == {"y": 2}
    assert all(got[s] == {} for s in ALL_SIDS if s not in ("R4", "R9"))


@pytest.mark.parametrize("dtype,cs", [("uint8", "srgb"), ("float32", "linear-srgb")])
def test_input_image_not_mutated(dtype, cs):
    img = make_image(dtype=dtype, cs=cs)
    before = img.content_hash()
    P.run_read(img, steps=fake_steps())
    assert img.content_hash() == before
