"""PRD-01: 스텁↔실물 교체, 폴백, 계약 위반·입력 변이 거부(미캐시)."""
import numpy as np
import pytest

from buc.core import contracts as C
from buc.pipeline import spine as S
from prd01_helpers import STUB_FNS, Counter, hashes, make_image, stub_registry


def brighter_world(inputs, ctx):
    out = STUB_FNS["S2"](inputs, ctx)
    w = out["world"]
    rgba = w.layers[0].rgba.copy()
    rgba[..., :3] = np.clip(rgba[..., :3] * 0.5 + 0.25, 0, 1)
    layer = C.LDILayer(rgba=rgba, depth=w.layers[0].depth)
    return {"world": C.World(obs=w.obs, layers=(layer,), theta_in=w.theta_in, level=w.level, read_hash=w.read_hash,
                             prov=w.prov)}


def test_swap_stub_to_real_and_back():
    cnt = Counter()
    reg = stub_registry(cnt)
    sp = S.Spine(reg)
    img = make_image()
    r_stub = sp.run(img)
    reg.register(S.Impl(stage="S2", kind="real", name="wld01", version="0.1", fn=cnt.wrap("S2real", brighter_world)))
    r_real = sp.run(img)                                   # 기본 prefer="real"
    assert r_real.stage_ios["S2"].impl_kind == "real" and r_real.stage_ios["S2"].impl_name == "wld01"
    assert r_real.status["S-1"] == "hit"
    assert set(r_real.ran) == {"S2", "S8", "S8.5", "S9-10", "S11"}
    assert r_real.root != r_stub.root
    assert r_real.artifacts["world"].content_hash() != r_stub.artifacts["world"].content_hash()
    # 스파인 코드 변경 없이 되돌림: 원래 캐시 전부 hit, 원래 root
    n_before = cnt.total()
    r_back = sp.run(img, prefer={"S2": "stub"})
    assert r_back.ran == () and cnt.total() == n_before
    assert r_back.root == r_stub.root and hashes(r_back.artifacts) == hashes(r_stub.artifacts)
    r_all_stub = S.Spine(reg, cache=sp.cache, prefer="stub").run(img)
    assert r_all_stub.ran == () and r_all_stub.root == r_stub.root


def test_spine_level_prefer_mapping():
    reg = stub_registry()
    reg.register(S.Impl(stage="S2", kind="real", name="wld01", version="0.1", fn=brighter_world))
    r = S.Spine(reg, prefer={"S2": "stub"}).run(make_image())
    assert r.stage_ios["S2"].impl_kind == "stub"
    r = S.Spine(reg, prefer="stub").run(make_image(), prefer={"S2": "real"})
    assert r.stage_ios["S2"].impl_kind == "real"


def boom(inputs, ctx):
    raise RuntimeError("gpu fell over")


def test_fallback_to_stub_records_reason():
    cnt = Counter()
    reg = stub_registry(cnt)
    reg.register(S.Impl(stage="S8", kind="real", name="camfm", version="1", fn=cnt.wrap("S8real", boom)))
    sp = S.Spine(reg)
    img = make_image()
    r = sp.run(img)
    assert r.status["S8"] == "fallback" and "S8" in r.ran
    io = r.stage_ios["S8"]
    assert io.impl_kind == "stub" and io.impl_name == "t_S8"
    assert io.fallback is not None and "real:camfm@1" in io.fallback and "RuntimeError" in io.fallback
    assert cnt.n["S8real"] == 1 and cnt.n["S8"] == 1
    ref = S.Spine(stub_registry()).run(img)
    assert hashes(r.artifacts) == hashes(ref.artifacts)
    # 실물은 다음 실행에서 다시 시도된다(실패 키는 캐시되지 않음)
    r2 = sp.run(img)
    assert cnt.n["S8real"] == 2 and cnt.n["S8"] == 1         # stub 은 캐시 hit
    assert r2.status["S8"] == "fallback" and "real:camfm@1" in r2.stage_ios["S8"].fallback
    assert r2.root == r.root
    # 폴백 표시는 그 실행에만: stub 을 직접 고르면 평범한 hit
    r3 = sp.run(img, prefer={"S8": "stub"})
    assert r3.status["S8"] == "hit" and r3.stage_ios["S8"].fallback is None and cnt.n["S8real"] == 2
    real = reg.get("S8", "real")
    key = S.stage_cache_key("S8", real, r.stage_ios["S8"].inputs, r.stage_ios["S8"].params)
    assert key not in sp.cache


def test_no_fallback_raises_stage_failed():
    reg = stub_registry()
    reg.register(S.Impl(stage="S8", kind="real", name="camfm", version="1", fn=boom))
    sp = S.Spine(reg, allow_fallback=False)
    with pytest.raises(S.StageFailed) as ei:
        sp.run(make_image())
    assert ei.value.stage == "S8" and ei.value.impl == "real:camfm@1"
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert isinstance(ei.value, S.SpineError)


def bad_first_pose(inputs, ctx):
    out = STUB_FNS["S8"](inputs, ctx)
    s = out["shot"]
    c2w = s.c2w.copy()
    c2w[:, 0, 3] += 0.5                                     # 첫 포즈가 관측 카메라와 다르다
    return {"shot": C.Shot(c2w=c2w, t=s.t, K=s.K, focus_m=s.focus_m, width=s.width, height=s.height, fps=s.fps,
                           theta=s.theta)}


def test_contract_violation_rejected_and_not_cached():
    reg = S.Registry()
    for st, fn in STUB_FNS.items():
        kind = "real" if st == "S8" else "stub"
        reg.register(S.Impl(stage=st, kind=kind, name=f"t_{st}", version="0", fn=bad_first_pose if st == "S8" else fn))
    sp = S.Spine(reg)                                       # S8 stub 없음 → 폴백 불가
    with pytest.raises(S.StageFailed) as ei:
        sp.run(make_image())
    assert ei.value.stage == "S8"
    assert isinstance(ei.value.__cause__, C.ContractError)
    # 같은 위반 구현에 stub 이 생기면 폴백
    reg.register(S.Impl(stage="S8", kind="stub", name="t_S8", version="0", fn=STUB_FNS["S8"]))
    r = sp.run(make_image())
    assert r.status["S8"] == "fallback" and "ContractError" in r.stage_ios["S8"].fallback


def wrong_keys(inputs, ctx):
    out = STUB_FNS["S9-10"](inputs, ctx)
    out["extra"] = out["frames"]
    return out


def mutating(inputs, ctx):
    out = STUB_FNS["S8.5"](inputs, ctx)
    inputs["read"].meta["touched"] = True                  # 입력 객체 변이
    return out


@pytest.mark.parametrize("stage,fn", [("S9-10", wrong_keys), ("S8.5", mutating)])
def test_bad_stub_fails_without_cache(stage, fn):
    reg = stub_registry()
    reg.register(S.Impl(stage=stage, kind="stub", name="bad", version="0", fn=fn), replace=True)
    cache = S.MemoryCache()
    sp = S.Spine(reg, cache=cache)
    with pytest.raises(S.StageFailed) as ei:
        sp.run(make_image())
    assert ei.value.stage == stage and ei.value.impl == "stub:bad@0"
    # 실패 스테이지 이전 것만 캐시에 있다
    ok = S.Spine(stub_registry(), cache=cache).run(make_image(), until=S.SPINE_STAGES[S.SPINE_STAGES.index(stage) - 1])
    assert ok.ran == ()
