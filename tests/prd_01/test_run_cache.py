"""PRD-01: 전 구간 실행, Merkle 캐시 hit, 부분 재실행, 조기 차단, force, seed·config."""
import pytest

from buc.core import contracts as C
from buc.core.config import Config
from buc.pipeline import spine as S
from prd01_helpers import STUB_FNS, Counter, hashes, make_image, stub_registry


def test_full_run_records_and_types():
    cnt = Counter()
    sp = S.Spine(stub_registry(cnt))
    img = make_image()
    r = sp.run(img, seed=3)
    assert tuple(r.stage_ios) == S.SPINE_STAGES
    assert r.status == {s: "run" for s in S.SPINE_STAGES}
    assert r.ran == S.SPINE_STAGES and r.hits == ()
    assert cnt.n == {s: 1 for s in S.SPINE_STAGES}
    a = r.artifacts
    assert set(a) == {"image", "read", "world", "shot", "frames", "kpi"}
    assert isinstance(a["read"], C.ReadPacket) and isinstance(a["world"], C.World)
    assert isinstance(a["shot"], C.Shot) and isinstance(a["frames"], C.Frames) and isinstance(a["kpi"], C.KPI)
    assert a["image"] is img
    assert r.seed == 3 and r.config_hash is None
    io = r.stage_ios
    assert io["S-1"].inputs == {"image": img.content_hash()}
    assert io["S2"].inputs == {"read": io["S-1"].outputs["read"]}
    assert io["S9-10"].inputs == {"world": io["S8.5"].outputs["world"], "shot": io["S8"].outputs["shot"]}
    assert io["S11"].outputs == {"kpi": a["kpi"].content_hash()}
    assert io["S9-10"].outputs == {"frames": a["frames"].content_hash()}
    for s, sio in io.items():
        assert isinstance(sio, C.StageIO)
        assert sio.stage == s and sio.impl_kind == "stub" and sio.impl_name == f"t_{s}" and sio.impl_version == "0"
        assert sio.fallback is None and sio.wall_s >= 0.0
        assert sio.params == S.stage_params(s, 3)
        impl = S.Impl(stage=s, kind="stub", name=f"t_{s}", version="0", fn=STUB_FNS[s])
        assert sio.cache_key == S.stage_cache_key(s, impl, sio.inputs, sio.params)
    assert r.root == S.run_root(io) and len(r.root) == 64


def test_second_run_all_hits_zero_calls():
    cnt = Counter()
    sp = S.Spine(stub_registry(cnt))
    img = make_image()
    r1 = sp.run(img, seed=1)
    before = dict(cnt.n)
    r2 = sp.run(make_image(), seed=1)          # 같은 내용의 새 객체
    assert cnt.n == before
    assert r2.status == {s: "hit" for s in S.SPINE_STAGES}
    assert r2.ran == () and r2.hits == S.SPINE_STAGES
    assert r2.root == r1.root
    assert hashes(r2.artifacts) == hashes(r1.artifacts)
    assert {s: io.cache_key for s, io in r2.stage_ios.items()} == {s: io.cache_key for s, io in r1.stage_ios.items()}


def test_fresh_spines_deterministic():
    ra = S.Spine(stub_registry()).run(make_image(), seed=9)
    rb = S.Spine(stub_registry()).run(make_image(), seed=9)
    assert ra.root == rb.root
    assert hashes(ra.artifacts) == hashes(rb.artifacts)


def test_param_change_reruns_exact_downstream():
    cnt = Counter()
    sp = S.Spine(stub_registry(cnt))
    img = make_image()
    r1 = sp.run(img, params={"S8": {"n": 3}})
    r2 = sp.run(img, params={"S8": {"n": 4}})
    assert set(r2.ran) == {"S8", "S9-10", "S11"}           # S8.5 는 shot 을 읽지 않는다
    assert set(r2.hits) == {"S-1", "S2", "S8.5"}
    assert r2.artifacts["shot"].n_frames == 4 and r2.artifacts["frames"].rgb.shape[0] == 4
    assert r2.root != r1.root
    r3 = sp.run(img, params={"S8": {"n": 3}})              # 원래 params → 전부 hit, 원래 root
    assert r3.ran == () and r3.root == r1.root


def test_upstream_param_change_propagates():
    sp = S.Spine(stub_registry())
    img = make_image()
    sp.run(img)
    r = sp.run(img, params={"S-1": {"depth_scale": 2.0}})
    assert r.ran == S.SPINE_STAGES


def test_early_cutoff_same_bytes_keeps_downstream_cached():
    cnt = Counter()
    reg = stub_registry(cnt)
    sp = S.Spine(reg)
    img = make_image()
    r1 = sp.run(img)
    # 버전만 올라가고 출력 바이트는 같은 S2 → S2만 다시 돈다
    reg.register(S.Impl(stage="S2", kind="stub", name="t_S2", version="1", fn=cnt.wrap("S2b", STUB_FNS["S2"])),
                 replace=True)
    r2 = sp.run(img)
    assert r2.ran == ("S2",)
    assert cnt.n.get("S2b") == 1
    assert r2.stage_ios["S2"].impl_version == "1"
    assert r2.stage_ios["S2"].outputs == r1.stage_ios["S2"].outputs
    assert r2.root != r1.root                              # 구현 버전은 root 에 남는다
    assert hashes(r2.artifacts) == hashes(r1.artifacts)


def test_force_recomputes_only_forced_when_deterministic():
    cnt = Counter()
    sp = S.Spine(stub_registry(cnt))
    img = make_image()
    r1 = sp.run(img)
    r2 = sp.run(img, force=("S2",))
    assert r2.status["S2"] == "run" and cnt.n["S2"] == 2
    assert set(r2.hits) == set(S.SPINE_STAGES) - {"S2"}
    assert r2.root == r1.root


def test_seed_changes_every_key_and_ctx_seed():
    seen = {}

    def spy(inputs, ctx):
        seen[ctx.stage] = (ctx.seed, dict(ctx.params), ctx.impl.ref)
        return STUB_FNS["S9-10"](inputs, ctx)

    reg = stub_registry()
    reg.register(S.Impl(stage="S9-10", kind="stub", name="t_S9-10", version="0", fn=spy), replace=True)
    sp = S.Spine(reg)
    img = make_image()
    r0 = sp.run(img, seed=0, params={"S9-10": {"q": 1}})
    assert seen["S9-10"] == (S.stage_seed(0, "S9-10"), {"q": 1, "_seed": S.stage_seed(0, "S9-10")},
                             "stub:t_S9-10@0")
    r5 = sp.run(img, seed=5, params={"S9-10": {"q": 1}})
    assert r5.ran == S.SPINE_STAGES
    for s in S.SPINE_STAGES:
        assert r5.stage_ios[s].cache_key != r0.stage_ios[s].cache_key
    # render 스텁은 ctx.seed 로 잡음을 만든다 → 프레임이 다르다
    assert r5.artifacts["frames"].content_hash() != r0.artifacts["frames"].content_hash()
    with pytest.raises(ValueError):
        sp.run(img, params={"S2": {"_seed": 3}})


def test_config_seed_and_hash():
    sp = S.Spine(stub_registry())
    img = make_image()
    cfg = Config(seed=5)
    rc = sp.run(img, config=cfg)
    rs = sp.run(img, seed=5)
    assert rc.seed == 5 and rc.root == rs.root
    assert rc.config_hash == cfg.hash()
    assert sp.run(img, config=cfg, seed=5).root == rc.root
    with pytest.raises(ValueError):
        sp.run(img, config=cfg, seed=6)


def test_until_and_skip_runs():
    cnt = Counter()
    sp = S.Spine(stub_registry(cnt))
    img = make_image()
    r = sp.run(img, until="S8")
    assert tuple(r.stage_ios) == ("S-1", "S2", "S8")
    assert "shot" in r.artifacts and "frames" not in r.artifacts
    assert r.root == S.run_root(r.stage_ios)
    rs = sp.run(img, skip=("S8.5",))
    assert "S8.5" not in rs.stage_ios
    assert rs.status["S-1"] == "hit" and rs.status["S2"] == "hit" and rs.status["S8"] == "hit"
    assert rs.stage_ios["S9-10"].inputs["world"] == rs.stage_ios["S2"].outputs["world"]
    with pytest.raises(S.SpineError):
        sp.run(img, skip=("S2",))
