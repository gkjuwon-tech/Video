"""PRD-03 x PRD-01: 스텁 팩이 스파인에서 돈다 — 실행, 캐시, 부분 재실행, 실물 실패 폴백, 렌더 이음매."""
import time

import numpy as np
import pytest

from buc.core import contracts as C
from buc.pipeline import spine as S
from prd03_helpers import STUB_STAGES, make_image

H, W = 18, 32


def make_spine(**kw):
    from buc.pipeline import stubs  # noqa: F401  (namespace 또는 일반 패키지 모두 허용)
    from buc.pipeline.stubs import pack
    reg = pack.register_stubs(S.Registry())
    return S.Spine(reg, **kw), reg


def hashes(art):
    return {k: v.content_hash() for k, v in art.items()}


def test_run_to_polish_and_plan():
    sp, _ = make_spine()
    plan = sp.plan(until="S8.5")
    assert [n.stage for n in plan] == list(STUB_STAGES)
    by = {n.stage: n for n in plan}
    assert by["S-1"].inputs == {"image": "input"} and by["S2"].inputs == {"read": "S-1"}
    assert by["S8"].inputs == {"world": "S2", "read": "S-1"} and by["S8.5"].inputs == {"world": "S2", "read": "S-1"}
    assert all(n.impl.kind == "stub" for n in plan)
    r = sp.run(make_image(H, W, seed=1), until="S8.5")
    assert r.ran == STUB_STAGES and r.hits == () and set(r.status.values()) == {"run"}
    assert set(r.artifacts) >= {"read", "world", "shot"}
    assert list(r.stage_ios) == list(STUB_STAGES)
    for st, io in r.stage_ios.items():
        assert io.impl_kind == "stub" and io.fallback is None
    assert r.stage_ios["S-1"].impl_name == "classic_read"
    assert r.stage_ios["S2"].impl_name == "flat_world" and r.stage_ios["S8"].impl_name == "fixed_move"
    assert r.stage_ios["S8.5"].impl_name == "identity_polish"
    # S8.5 는 world 를 그대로 통과 → 출력 해시가 S2 와 같다
    assert r.stage_ios["S8.5"].outputs["world"] == r.stage_ios["S2"].outputs["world"]
    assert len(r.root) == 64


def test_second_run_all_hit_same_root_and_fresh_spine_same_root():
    img = make_image(H, W, seed=2)
    sp, _ = make_spine()
    r1 = sp.run(img, seed=7, until="S8.5")
    r2 = sp.run(img, seed=7, until="S8.5")
    assert r2.ran == () and set(r2.status.values()) == {"hit"} and r2.root == r1.root
    assert hashes(r2.artifacts) == hashes(r1.artifacts)
    sp3, _ = make_spine()
    r3 = sp3.run(img, seed=7, until="S8.5")
    assert r3.ran == STUB_STAGES and r3.root == r1.root and hashes(r3.artifacts) == hashes(r1.artifacts)


def test_seed_changes_only_shot_seed():
    img = make_image(H, W, seed=3)
    a = make_spine()[0].run(img, seed=1, until="S8.5")
    b = make_spine()[0].run(img, seed=2, until="S8.5")
    assert a.artifacts["shot"].seed != b.artifacts["shot"].seed
    assert a.artifacts["read"].content_hash() == b.artifacts["read"].content_hash()
    assert a.artifacts["world"].content_hash() == b.artifacts["world"].content_hash()
    for k in ("c2w", "t", "K", "focus_m"):
        assert np.array_equal(getattr(a.artifacts["shot"], k), getattr(b.artifacts["shot"], k))


def test_disk_cache_reopen_all_hit(tmp_path):
    img = make_image(H, W, seed=4)
    sp1, _ = make_spine(cache=S.DiskCache(tmp_path / "c"))
    r1 = sp1.run(img, until="S8.5")
    sp2, _ = make_spine(cache=S.DiskCache(tmp_path / "c"))
    r2 = sp2.run(img, until="S8.5")
    assert r2.ran == () and r2.root == r1.root and hashes(r2.artifacts) == hashes(r1.artifacts)


def test_partial_rerun_sets():
    img = make_image(H, W, seed=5)
    sp, _ = make_spine()
    base = sp.run(img, until="S8.5")
    r = sp.run(img, until="S8.5", params={"S8": {"push_frac": 0.2}})
    assert r.ran == ("S8",) and set(r.hits) == {"S-1", "S2", "S8.5"}
    assert r.artifacts["shot"].content_hash() != base.artifacts["shot"].content_hash()
    assert r.artifacts["world"].content_hash() == base.artifacts["world"].content_hash()
    assert r.root != base.root
    r = sp.run(img, until="S8.5", params={"S2": {"n_layers": 5}})
    assert r.ran == ("S2", "S8", "S8.5") and r.hits == ("S-1",)            # S8 은 world 를 입력으로 받는다
    assert len(r.artifacts["world"].layers) == 5
    assert r.artifacts["shot"].content_hash() == base.artifacts["shot"].content_hash()   # 그러나 샷 내용은 같다
    r = sp.run(img, until="S8.5", params={"S-1": {"cam_height_m": 2.0}})
    assert r.ran == STUB_STAGES
    assert r.artifacts["read"].content_hash() != base.artifacts["read"].content_hash()
    r = sp.run(img, until="S8.5", force=("S8",))
    assert r.ran == ("S8",) and r.root == base.root                        # 강제 재실행도 같은 결과


def _boom(inputs, ctx):
    raise RuntimeError("real impl exploded")


@pytest.mark.parametrize("stage", ["S-1", "S2", "S8"])
def test_real_failure_falls_back_to_stub_with_same_root(stage):
    img = make_image(H, W, seed=6)
    ref, _ = make_spine()
    r_stub = ref.run(img, until="S8.5")
    sp, reg = make_spine()
    reg.register(S.Impl(stage=stage, kind="real", name="broken", version="9", fn=_boom))
    r = sp.run(img, until="S8.5")
    assert r.status[stage] == "fallback" and stage in r.ran
    io = r.stage_ios[stage]
    assert io.impl_kind == "stub" and io.fallback == "real:broken@9: RuntimeError"
    assert r.root == r_stub.root and hashes(r.artifacts) == hashes(r_stub.artifacts)
    for other in STUB_STAGES:
        if other != stage:
            assert r.stage_ios[other].fallback is None
    # 폴백을 끄면 스텁이 구해 주지 않는다
    sp2, reg2 = make_spine(allow_fallback=False)
    reg2.register(S.Impl(stage=stage, kind="real", name="broken", version="9", fn=_boom))
    with pytest.raises(S.StageFailed) as ei:
        sp2.run(img, until="S8.5")
    assert ei.value.stage == stage and ei.value.impl == "real:broken@9"


def test_real_swap_in_leaves_stub_chain_valid():
    """실물이 S2 만 대체해도(여기서는 층 5장 평면 세계) 나머지 스텁과 맞물린다."""
    from buc.pipeline.stubs import world_flat
    img = make_image(H, W, seed=7)
    sp, reg = make_spine()

    def five(inputs, ctx):
        ctx2 = S.StageContext(stage=ctx.stage, seed=ctx.seed, params={**ctx.params, "n_layers": 5}, impl=ctx.impl)
        return world_flat.flat_world(inputs, ctx2)

    reg.register(S.Impl(stage="S2", kind="real", name="five_cards", version="1", fn=five))
    r = sp.run(img, until="S8.5")
    assert r.stage_ios["S2"].impl_kind == "real" and len(r.artifacts["world"].layers) == 5
    assert r.stage_ios["S-1"].impl_kind == "stub" and r.stage_ios["S8"].impl_kind == "stub"
    back = sp.run(img, until="S8.5", prefer={"S2": "stub"})
    assert len(back.artifacts["world"].layers) == 3


def _quads():
    return pytest.importorskip("buc.render.quads")


def test_render_seam_first_frame_reproduces_input_and_shot_renders():
    quads = _quads()
    img = make_image(H, W, seed=8)
    r = make_spine()[0].run(img, until="S8.5")
    read, world, shot = r.artifacts["read"], r.artifacts["world"], r.artifacts["shot"]
    t0 = time.time()
    o = quads.render_ldi(world, c2w=shot.c2w[0], K=shot.K[0], width=shot.width, height=shot.height)
    assert np.allclose(o.rgb, read.image_lin, atol=1e-4)                  # 관측 포즈에서 입력 재현
    assert np.allclose(o.alpha, 1.0, atol=1e-4)
    frames = quads.render_shot(world, shot)
    frames.validate()
    assert frames.rgb.shape == (25, H, W, 3) and frames.fps == shot.fps
    assert np.isfinite(frames.rgb).all() and frames.rgb.min() >= 0.0 and frames.rgb.max() <= 1.0
    assert frames.shot_hash == shot.content_hash() and frames.world_hash == world.content_hash()
    C.check_stage_io("S9-10", {"world": world, "shot": shot}, {"frames": frames})
    move = np.abs(frames.rgb[-1] - frames.rgb[0]).mean()
    assert move > 1e-3                                                     # 카메라가 실제로 움직였다(시차)
    assert time.time() - t0 < 40


def test_full_skeleton_to_s11_with_local_render_and_report():
    """걷는 해골: 스텁 4개 + (실물) RDR-02 렌더 + 보고 스텁 → 같은 root 로 재현."""
    quads = _quads()

    def render(inputs, ctx):
        return {"frames": quads.render_shot(inputs["world"], inputs["shot"])}

    def report(inputs, ctx):
        return {"kpi": C.KPI(ffd=float(inputs["frames"].rgb.mean()), wall_s=0.0)}

    def build():
        sp, reg = make_spine()
        reg.register(S.Impl(stage="S9-10", kind="real", name="rdr02", version="t", fn=render))
        reg.register(S.Impl(stage="S11", kind="stub", name="t_report", version="t", fn=report))
        return sp

    img = make_image(H, W, seed=9)
    r1 = build().run(img, seed=3)
    assert r1.ran == S.SPINE_STAGES and isinstance(r1.artifacts["kpi"], C.KPI)
    r2 = build().run(img, seed=3)
    assert r2.root == r1.root and hashes(r2.artifacts) == hashes(r1.artifacts)
