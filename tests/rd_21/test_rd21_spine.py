"""RD-21 x PRD-01: S-1 실물 Impl 로 스파인에 끼운다 — 실행, hit, 잘못된 params 의 스텁 폴백."""
import re

import pytest

from buc.core import contracts as C
from buc.pipeline import spine as S
from buc.pipeline.stubs.pack import register_stubs
from buc.read import pipeline as P
from rd21_helpers import fake_steps, make_image, smooth_image


def test_make_read_impl_identity():
    a = P.make_read_impl()
    assert isinstance(a, S.Impl)
    assert (a.stage, a.kind, a.name) == ("S-1", "real", "rd21_read")
    assert re.fullmatch(r"0\.1\.0\+[0-9a-f]{12}", a.version)
    assert P.make_read_impl().version == a.version
    assert P.make_read_impl(steps=fake_steps()).version != a.version
    assert P.make_read_impl(opts=P.ReadOpts(cam_height_m=1.6)).version != a.version
    assert P.make_read_impl(steps=fake_steps(version="2")).version != P.make_read_impl(steps=fake_steps()).version


def test_read_stage_direct_call_matches_run_read():
    img = smooth_image()
    impl = P.make_read_impl()
    ctx = S.StageContext(stage="S-1", seed=3, params={"_seed": 99}, impl=impl)
    out = P.read_stage({"image": img}, ctx)
    assert set(out) == {"read"}
    ref = P.run_read(img, steps=P.default_steps()).packet
    assert out["read"].content_hash() == ref.content_hash()


def _registry(impl):
    reg = S.Registry()
    register_stubs(reg)
    reg.register(impl)
    return reg


def test_spine_runs_real_read_and_hits():
    img = smooth_image()
    sp = S.Spine(_registry(P.make_read_impl()))
    r1 = sp.run(img, until="S2")
    assert r1.status["S-1"] == "run"
    read = r1.artifacts["read"]
    assert read.meta["read"]["pipeline"] == "rd21_read@0.1.0"
    assert read.fallbacks == ("R5:skipped",)
    assert r1.stage_ios["S-1"].impl_kind == "real"
    r2 = sp.run(img, until="S2")
    assert r2.status["S-1"] == "hit" and r2.root == r1.root


def test_spine_params_reach_steps():
    img = smooth_image()
    calls, got = [], {}

    def r9(state, params):
        got["p"] = dict(params)
        return {"meta": {}}

    sp = S.Spine(_registry(P.make_read_impl(steps=fake_steps(calls, override={"R9": r9}))))
    r = sp.run(img, until="S-1", params={"S-1": {"R9": {"q": 1}}})
    assert r.status["S-1"] == "run"
    assert len(calls) == 14
    assert got["p"] == {"q": 1}  # _seed 는 단계로 가지 않는다


def test_bad_params_raise_from_impl():
    """인자 위반만 예외(→ 스파인이 다룬다). 단계 실패는 예외가 아니다."""
    img = make_image(h=40, w=56)
    impl = P.make_read_impl(steps=fake_steps())
    for bad in ({"R9": 5}, {"R99": {}}):
        ctx = S.StageContext(stage="S-1", seed=0, params={**bad, "_seed": 1}, impl=impl)
        with pytest.raises(ValueError):
            impl.fn({"image": img}, ctx)
    sp = S.Spine(_registry(impl))
    with pytest.raises(S.StageFailed):  # 스텁도 같은 params 를 거부한다
        sp.run(img, until="S-1", params={"S-1": {"R9": 5}})


def test_spine_cache_passthrough_to_step_cache():
    img = smooth_image()
    cache = P.MemoryReadCache()
    calls = []
    impl = P.make_read_impl(steps=fake_steps(calls), cache=cache)
    sp1 = S.Spine(_registry(impl))
    sp1.run(img, until="S-1")
    assert len(calls) == 14
    calls.clear()
    sp2 = S.Spine(_registry(impl))  # 새 스파인 캐시 → S-1 다시 실행, 하지만 단계 캐시는 hit
    r = sp2.run(img, until="S-1", params={"S-1": {"R13": {"x": 1}}})
    assert r.status["S-1"] == "run"
    assert calls == ["R13"]
    C.check_stage_io("S-1", {"image": img}, {"read": r.artifacts["read"]})
