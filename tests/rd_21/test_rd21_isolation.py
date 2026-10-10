"""RD-21 실패 격리: 예외·계약 밖 출력은 그 단계만 폴백시키고 나머지는 계속 돈다."""
import numpy as np
import pytest

from buc.core import contracts as C
from buc.read import pipeline as P
from rd21_helpers import ALL_SIDS, fake_outputs, fake_steps, make_image, status_map

# 패킷 핵심 필드에 영향이 없는 단계(그 단계 실패 시 핵심 필드는 정상 실행과 같아야 한다)
NON_PACKET = tuple(s for s in ALL_SIDS if s not in ("R0", "R1", "R5", "R6", "R11"))


def _raise(exc):
    def fn(state, params):
        raise exc
    return fn


@pytest.mark.parametrize("sid", ALL_SIDS)
def test_exception_in_one_step_is_isolated(sid):
    calls = []
    img = make_image()
    base = P.run_read(img, steps=fake_steps())
    res = P.run_read(img, steps=fake_steps(calls, override={sid: _raise(RuntimeError("boom"))}))
    assert calls == list(P.EXEC_ORDER)  # 모두 호출됨
    st = status_map(res)
    assert st[sid] == "fallback"
    assert all(st[s] == "ok" for s in ALL_SIDS if s != sid)
    assert res.packet.fallbacks == (f"{sid}:fake_{sid}@1:RuntimeError",)
    rec = [r for r in res.records if r.sid == sid][0]
    assert rec.error is not None and "RuntimeError" in rec.error and "boom" in rec.error
    assert res.packet.meta["read"]["steps"][sid] == {"impl": f"fake_{sid}@1", "status": "fallback"}
    assert res.outputs[sid]["meta"] == {"status": "fallback"}
    C.check_stage_io("S-1", {"image": img}, {"read": res.packet})
    if sid in NON_PACKET:
        a, b = res.packet, base.packet
        np.testing.assert_array_equal(a.depth, b.depth)
        np.testing.assert_array_equal(a.image_lin, b.image_lin)
        np.testing.assert_array_equal(a.depth_conf, b.depth_conf)
        assert a.obs.content_hash() == b.obs.content_hash()
        assert (a.scale_s, a.sigma_s, a.level) == (b.scale_s, b.sigma_s, b.level)


def test_multiple_failures_listed_in_exec_order():
    ov = {s: _raise(ValueError("x")) for s in ("R13", "R3", "R1", "R6")}
    res = P.run_read(make_image(), steps=fake_steps(override=ov))
    assert res.packet.fallbacks == ("R1:fake_R1@1:ValueError", "R6:fake_R6@1:ValueError",
                                    "R3:fake_R3@1:ValueError", "R13:fake_R13@1:ValueError")


def _with(sid, **upd):
    def fn(state, params):
        out = dict(fake_outputs(sid, state))
        for k, v in upd.items():
            if v is _DROP:
                out.pop(k)
            else:
                out[k] = v(state) if callable(v) else v
        return out
    return fn


_DROP = object()

BAD_OUTPUTS = [
    ("R0", dict(image_lin=lambda s: np.zeros((s.H, s.W - 1, 3), np.float32))),
    ("R0", dict(image_lin=lambda s: np.full((s.H, s.W, 3), np.nan, np.float32))),
    ("R0", dict(image_lin=lambda s: np.full((s.H, s.W, 3), -0.5, np.float32))),
    ("R0", dict(fov_x_prior_deg=200.0)),
    ("R0", dict(image_lin=_DROP)),
    ("R1", dict(K=lambda s: np.diag([-100.0, 100.0, 1.0]))),
    ("R1", dict(K=lambda s: np.array([[100.0, 0, 5], [0, 100.0, 5], [0, 0.1, 1.0]]))),
    ("R1", dict(K=lambda s: np.diag([1e9, 1e9, 1.0]))),  # focal_mm 범위 밖
    ("R1", dict(up_cam=np.array([0.0, -0.5, 0.0]))),
    ("R1", dict(f_source="magic")),
    ("R1", dict(k1=float("nan"))),
    ("R5", dict(log_depth=lambda s: np.full((s.H // 4, s.W // 4), np.nan, np.float32))),
    ("R5", dict(r_dis_map=lambda s: np.zeros((s.H // 4 + 1, s.W // 4), np.float32))),
    ("R5", dict(r_dis_map=lambda s: -np.ones((s.H // 4, s.W // 4), np.float32))),
    ("R5", dict(log_depth=lambda s: np.zeros((2, 3, 4), np.float32))),
    ("R6", dict(s=0.0)),
    ("R6", dict(sigma_s=float("nan"))),
    ("R6", dict(reliable="yes")),
    ("R11", dict(level="L9")),
    ("R7", dict(meta={"bad": {1, 2}})),
    ("R8", dict(meta={"x": float("nan")})),
    ("R9", dict(meta=_DROP)),
]


@pytest.mark.parametrize("sid,upd", BAD_OUTPUTS, ids=[f"{s}-{i}" for i, (s, _) in enumerate(BAD_OUTPUTS)])
def test_invalid_output_triggers_fallback(sid, upd):
    img = make_image()
    res = P.run_read(img, steps=fake_steps(override={sid: _with(sid, **upd)}))
    assert status_map(res)[sid] == "fallback"
    assert res.packet.fallbacks == (f"{sid}:fake_{sid}@1:StepOutputError",)
    C.check_stage_io("S-1", {"image": img}, {"read": res.packet})


@pytest.mark.parametrize("ret", [None, 3, ["meta"]])
def test_non_dict_return_is_output_error(ret):
    res = P.run_read(make_image(), steps=fake_steps(override={"R2": lambda s, p: ret}))
    assert res.packet.fallbacks == ("R2:fake_R2@1:StepOutputError",)


def test_extra_output_keys_allowed_and_visible_downstream():
    seen = {}

    def r5(state, params):
        out = dict(fake_outputs("R5", state))
        out["extra_head"] = 7
        return out

    def r6(state, params):
        seen["x"] = state["R5"]["extra_head"]
        return fake_outputs("R6", state)

    res = P.run_read(make_image(), steps=fake_steps(override={"R5": r5, "R6": r6}))
    assert res.packet.fallbacks == ()
    assert seen["x"] == 7


def test_up_cam_is_normalized():
    up = np.array([0.0, -1.005, 0.0])
    res = P.run_read(make_image(), steps=fake_steps(override={"R1": _with("R1", up_cam=up)}))
    assert res.packet.fallbacks == ()
    np.testing.assert_allclose(np.linalg.norm(res.outputs["R1"]["up_cam"]), 1.0, atol=1e-12)


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        P.run_read(make_image(), steps=fake_steps(override={"R4": _raise(KeyboardInterrupt())}))


def test_upstream_arrays_are_read_only():
    def r1(state, params):
        state["R0"]["image_lin"][0, 0, 0] = 5.0
        return fake_outputs("R1", state)

    img = make_image()
    base = P.run_read(img, steps=fake_steps())
    res = P.run_read(img, steps=fake_steps(override={"R1": r1}))
    assert status_map(res)["R1"] == "fallback"
    assert res.packet.fallbacks[0].startswith("R1:fake_R1@1:")
    np.testing.assert_array_equal(res.packet.image_lin, base.packet.image_lin)


def test_downstream_of_failed_step_receives_fallback_values():
    seen = {}

    def r6(state, params):
        seen["st"] = state.status("R5")
        seen["ld"] = state["R5"]["log_depth"]
        return fake_outputs("R6", state)

    res = P.run_read(make_image(), steps=fake_steps(override={"R5": _raise(MemoryError()), "R6": r6}))
    assert seen == {"st": "fallback", "ld": None}
    assert res.packet.meta["read"]["depth_source"] == "plane"
    assert res.packet.fallbacks == ("R5:fake_R5@1:MemoryError",)
