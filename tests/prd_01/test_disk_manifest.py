"""PRD-01: DiskCache 재오픈·손상 복구·파일 배치, MemoryCache 사본, manifest 안정성, GPU 재현."""
import json
import os

import numpy as np
import pytest

from buc.core import contracts as C
from buc.core.config import Config
from buc.core.manifest import Manifest
from buc.pipeline import spine as S
from prd01_helpers import Counter, hashes, make_image, stub_registry


def _files(root):
    out = []
    for d, _, fs in os.walk(root):
        out += [os.path.join(d, f) for f in fs]
    return sorted(out)


def test_disk_cache_reopen_all_hits(tmp_path):
    c1 = Counter()
    r1 = S.Spine(stub_registry(c1), cache=S.DiskCache(tmp_path)).run(make_image(), seed=2)
    assert c1.total() == 6
    for s, io in r1.stage_ios.items():
        p = tmp_path / io.cache_key[:2] / f"{io.cache_key}.json"
        assert p.is_file()
        d = json.loads(p.read_text())
        assert C.StageIO.from_dict(d["stage_io"]).cache_key == io.cache_key
        assert set(d["outputs"]) == set(io.outputs)
    assert all(f.endswith(".json") for f in _files(tmp_path)) and len(_files(tmp_path)) == 6
    c2 = Counter()
    r2 = S.Spine(stub_registry(c2), cache=S.DiskCache(tmp_path)).run(make_image(), seed=2)
    assert c2.total() == 0 and r2.ran == ()
    assert r2.root == r1.root and hashes(r2.artifacts) == hashes(r1.artifacts)
    np.testing.assert_array_equal(r2.artifacts["frames"].rgb, r1.artifacts["frames"].rgb)


def test_disk_cache_get_put_roundtrip(tmp_path):
    cache = S.DiskCache(tmp_path)
    r = S.Spine(stub_registry(), cache=cache).run(make_image())
    io = r.stage_ios["S8"]
    assert io.cache_key in cache
    got = cache.get(io.cache_key)
    assert got is not None
    gio, gout = got
    assert gio.cache_key == io.cache_key and gio.outputs == io.outputs
    assert gout["shot"].content_hash() == r.artifacts["shot"].content_hash()
    assert cache.get("0" * 64) is None and ("0" * 64) not in cache


def test_disk_cache_corruption_is_miss_and_repaired(tmp_path):
    img = make_image()
    r1 = S.Spine(stub_registry(), cache=S.DiskCache(tmp_path)).run(img)
    k8 = r1.stage_ios["S8"].cache_key
    k2 = r1.stage_ios["S2"].cache_key
    (tmp_path / k8[:2] / f"{k8}.json").write_text("{not json")
    p2 = tmp_path / k2[:2] / f"{k2}.json"
    d = json.loads(p2.read_text())
    d["stage_io"]["outputs"]["world"] = "f" * 64              # 출력 해시 위조
    p2.write_text(json.dumps(d))
    cache = S.DiskCache(tmp_path)
    assert cache.get(k8) is None and cache.get(k2) is None
    cnt = Counter()
    r2 = S.Spine(stub_registry(cnt), cache=cache).run(img)
    assert set(r2.ran) == {"S2", "S8"}                        # 다시 계산된 출력이 같으니 하류는 hit
    assert r2.root == r1.root
    assert S.DiskCache(tmp_path).get(k8) is not None        # 덮어써서 복구
    assert all(f.endswith(".json") for f in _files(tmp_path))


def test_memory_cache_isolated_from_mutation():
    cache = S.MemoryCache()
    sp = S.Spine(stub_registry(), cache=cache)
    img = make_image()
    r1 = sp.run(img)
    r1.artifacts["read"].meta["poison"] = 1                  # 결과 객체를 바꿔도
    r1.artifacts["kpi"].extra["poison"] = 1
    cnt = Counter()
    r2 = S.Spine(stub_registry(cnt), cache=cache).run(img)
    assert r2.ran == () and cnt.total() == 0                  # 캐시는 오염되지 않는다
    assert r2.root == r1.root
    assert "poison" not in r2.artifacts["read"].meta
    assert r2.artifacts["read"].content_hash() == r1.stage_ios["S-1"].outputs["read"]


def test_manifest_stable_across_cached_rerun(tmp_path):
    cfg = Config(seed=4)
    img = make_image()
    sp = S.Spine(stub_registry(), cache=S.DiskCache(tmp_path))
    r1 = sp.run(img, config=cfg)
    m1 = r1.manifest("prd01-test", cfg, git_commit="unknown", created_utc="2026-10-10T00:00:00Z")
    assert isinstance(m1, Manifest)
    assert m1.inputs == {"image": img.content_hash()}
    assert m1.params["root"] == r1.root
    assert m1.params["impls"] == {s: f"stub:t_{s}@0" for s in S.SPINE_STAGES}
    assert m1.seed == 4 and m1.config_hash == cfg.hash()
    r2 = S.Spine(stub_registry(), cache=S.DiskCache(tmp_path)).run(img, config=cfg)
    m2 = r2.manifest("prd01-test", cfg, git_commit="unknown", created_utc="2026-10-11T12:00:00Z")
    assert m2.manifest_id == m1.manifest_id
    r3 = sp.run(img, config=cfg, params={"S8": {"n": 5}})
    m3 = r3.manifest("prd01-test", cfg, git_commit="unknown", created_utc="2026-10-10T00:00:00Z")
    assert m3.manifest_id != m1.manifest_id


@pytest.mark.gpu
def test_gpu_real_render_reproducible():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")

    def gpu_render(inputs, ctx):
        world, shot = inputs["world"], inputs["shot"]
        g = torch.Generator(device="cuda").manual_seed(ctx.seed % (2**63))
        base = torch.as_tensor(world.layers[0].rgba[..., :3], device="cuda")
        noise = torch.rand((shot.n_frames, shot.height, shot.width, 3), generator=g, device="cuda") * 0.01
        rgb = torch.clamp(base[None] * 0.9 + noise, 0, 1).float().cpu().numpy()
        return {"frames": C.Frames(rgb=rgb, fps=shot.fps, shot_hash=shot.content_hash(),
                                   world_hash=world.content_hash())}

    roots = []
    for _ in range(2):
        reg = stub_registry()
        reg.register(S.Impl(stage="S9-10", kind="real", name="cuda_render", version="0", fn=gpu_render))
        r = S.Spine(reg).run(make_image(), seed=1)
        assert r.status["S9-10"] == "run" and r.stage_ios["S9-10"].impl_kind == "real"
        roots.append(r.root)
    assert roots[0] == roots[1]
