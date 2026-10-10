"""RD-07 클린룸 레인(ext 비트)과 끝-끝 판독: read_mrh + MrhReader(진짜 TapBus·소형 MRH·가짜 DiT)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import rd07_helpers as H
from buc.gov import prov as P
from buc.read import mrh_call as M
from buc.teacher import taps as T

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG = dict(tile_px=64, feather_px=16, global_pixels=3072)
E2E_HW = (96, 128)


def _img():
    return H.make_image(*E2E_HW, seed=41)


# ----------------------------------------------------------------------------------------- lane_for · 스텁 읽기기

@pytest.mark.parametrize("tags,lane", [
    (["ENGINE"], "engine"), (["CAPTURE"], "engine"), (["HOLDOUT"], "engine"), (["COMP"], "engine"),
    (["USER"], "cleanroom"), (["USER-DERIVED"], "cleanroom"), (["EXTGEN:test"], "cleanroom"),
    (["EXTGEN-DERIVED:soul"], "cleanroom"), (["BIO"], "cleanroom"), (["ENGINE", "USER"], "cleanroom"),
])
def test_lane_for_follows_ext_bit(tags, lane):
    assert M.lane_for(P.as_record(tags)) == lane
    assert M.lane_for(tags) == lane                              # as_record가 받는 형태면 된다


def test_ext_input_runs_in_cleanroom_lane_with_derived_prov_and_vault_ns():
    stub = H.StubReader()
    r = M.read_mrh(_img(), stub, prov=H.USER, cfg=M.ReadConfig(**CFG), key="k", cleanroom_binary_hash="ab" * 32)
    assert r.lane == "cleanroom" and r.ext is True and r.ns == "user_vault"
    assert r.prov == P.derive("mrh_read", [H.USER]) and "USER-DERIVED" in r.prov.taint
    assert r.prov.ns == "user_vault" and r.cleanroom_binary_hash == "ab" * 32
    m = r.meta()
    json.dumps(m, allow_nan=False)
    assert m["lane"] == "cleanroom" and m["ext"] is True and m["ns"] == "user_vault"
    assert m["cleanroom_binary_hash"] == "ab" * 32
    assert all(c.prov == P.as_record(H.USER) for c in stub.calls)            # 읽기기에는 원본 prov가 간다


def test_engine_input_keeps_engine_lane_and_train_ok_prov():
    r = M.read_mrh(_img(), H.StubReader(), prov=H.ENGINE, cfg=M.ReadConfig(**CFG), key="k")
    assert r.lane == "engine" and r.ext is False and r.ns == "train_ok" and r.cleanroom_binary_hash is None
    assert r.prov == P.derive("mrh_read", [H.ENGINE]) and P.train_ok(r.prov)


def test_explicit_lane_rules():
    img = _img()
    cfg = M.ReadConfig(**CFG)
    ok = M.read_mrh(img, H.StubReader(), prov=H.USER, cfg=cfg, key="k", lane="cleanroom")
    assert ok.lane == "cleanroom"
    stub = H.StubReader()
    with pytest.raises(M.CleanroomError):
        M.read_mrh(img, stub, prov=H.USER, cfg=cfg, key="k", lane="engine")
    with pytest.raises(M.CleanroomError):
        M.read_mrh(img, stub, prov=P.as_record(["EXTGEN:test"]), cfg=cfg, key="k", lane="engine")
    assert stub.calls == []
    with pytest.raises((M.MrhCallSpecError, ValueError)):
        M.read_mrh(img, H.StubReader(), prov=H.ENGINE, cfg=cfg, key="k", lane="bogus")


def test_ext_lane_fails_closed_when_taps_are_not_ephemeral():
    stub = H.StubReader(ephemeral=False)                         # 읽기기가 ext인데 임시 탭이 아니라고 주장
    with pytest.raises(M.CleanroomError):
        M.read_mrh(_img(), stub, prov=H.USER, cfg=M.ReadConfig(**CFG), key="k")
    assert len(stub.calls) == 1, "첫 위반 즉시 중단"
    # 엔진 레인은 임시 여부를 따지지 않는다
    M.read_mrh(_img(), H.StubReader(ephemeral=False), prov=H.ENGINE, cfg=M.ReadConfig(**CFG), key="k")


# ----------------------------------------------------------------------------------------- 끝-끝(MrhReader)

def _e2e(stack, prov=H.ENGINE, key="e2e", **kw):
    return M.read_mrh(_img(), stack.reader, prov=prov, cfg=M.ReadConfig(**CFG), key=key, **kw)


def test_end_to_end_forward_budget_sigmas_shapes_and_heads(tmp_path):
    s = H.make_stack(tmp_path)
    h0 = H.weights_hash(s.mrh)
    r = _e2e(s)
    calls = s.dit.calls
    # 잠정 표에서 0.2/0.35/0.25는 모두 0.3으로 붙는다 → 전역 1회 + 타일 4회
    assert r.sigma_pair_degenerate is True and r.n_forward == 5 == len(calls) == len(r.digests)
    assert tuple(r.global_sigmas) == pytest.approx((0.3,)) and r.tile_sigma == pytest.approx(0.3)
    assert [c.x_shapes[0] for c in calls] == [(16, 1, 6, 8)] + [(16, 1, 8, 8)] * 4      # 전역 48x64, 타일 64x64
    assert all(c.t.tolist() == pytest.approx([0.7]) for c in calls)
    assert [float(c.t) for c in s.spy.calls] == pytest.approx([0.3] * 5)
    assert r.global_size == (48, 64) and len(r.tiles) == 4 and len(set(r.digests)) == 5
    o = r.outputs
    assert set(o) == {"log_depth", "normal", "albedo", "rough_metal", "shading", "defocus", "sem", "sigma_hat"}
    assert {k: tuple(v.shape) for k, v in o.items()} == {
        "log_depth": (1, 1, 24, 32), "normal": (1, 3, 24, 32), "albedo": (1, 3, 24, 32),
        "rough_metal": (1, 2, 24, 32), "shading": (1, 1, 24, 32), "defocus": (1, 1, 24, 32),
        "sem": (1, 8, 24, 32), "sigma_hat": (1, 4, 24, 32)}
    assert all(bool(torch.isfinite(v).all()) and v.dtype == torch.float32 and not v.requires_grad for v in o.values())
    assert float((o["normal"].norm(dim=1) - 1).abs().max()) < 1e-5
    assert float((o["sem"].norm(dim=1) - 1).abs().max()) < 1e-5
    assert float(o["albedo"].min()) >= 0.0 and float(o["albedo"].max()) <= 1.0
    assert float(o["defocus"].min()) >= 0.0 and float(o["sigma_hat"].min()) >= 1e-3 - 1e-9
    assert np.isfinite(r.r_dis) and r.r_dis >= 0.0 and r.r_dis_sigma == 0.0
    assert r.depth_global.shape == o["log_depth"].shape
    assert H.weights_hash(s.mrh) == h0 and all(p.grad is None for p in s.mrh.parameters())
    json.dumps(r.meta(), allow_nan=False)


def test_end_to_end_with_distinct_sigma_table_calls_both_global_sigmas(tmp_path):
    from buc.teacher.sample import SigmaTable
    s = H.make_stack(tmp_path, table=SigmaTable((1.0, 0.75, 0.5, 0.35, 0.25, 0.2), name="rd07-test"))
    r = _e2e(s)
    assert r.sigma_pair_degenerate is False and r.n_forward == 6
    assert tuple(r.global_sigmas) == pytest.approx((0.2, 0.35)) and r.tile_sigma == pytest.approx(0.25)
    assert [c.t.tolist()[0] for c in s.dit.calls] == pytest.approx([0.8, 0.65] + [0.75] * 4)
    assert r.r_dis_sigma >= 0.0 and np.isfinite(r.r_dis_sigma)
    assert r.r_dis == pytest.approx(max(r.r_dis_tiles, r.r_dis_sigma), rel=1e-6)


def test_end_to_end_is_deterministic_and_independent_of_lane_for_values(tmp_path):
    s = H.make_stack(tmp_path)
    a = _e2e(s)
    b = _e2e(s)
    c = _e2e(s, prov=H.USER)                                      # ext 레인: 값은 같다, digest는 prov 비트 때문에 다르다
    for k in a.outputs:
        assert torch.equal(a.outputs[k], b.outputs[k]) and torch.equal(a.outputs[k], c.outputs[k]), k
    assert torch.equal(a.r_dis_map, c.r_dis_map) and a.r_dis == c.r_dis
    assert a.digests == b.digests and a.digests != c.digests
    d = _e2e(s, key="other")                                      # 키가 다르면 노이즈가 달라 판독이 달라진다
    assert not torch.equal(a.outputs["log_depth"], d.outputs["log_depth"])


def test_end_to_end_ext_never_reaches_ring_engine_does(tmp_path):
    ring = T.TapRing(H.make_basis(), max_entries=64)
    s = H.make_stack(tmp_path, ring=ring)
    r = _e2e(s, prov=H.USER)
    assert r.lane == "cleanroom" and len(ring) == 0
    assert s.bus.stats["pushed"] == 0 and s.bus.stats["bypassed"] == 5 and s.bus.stats["captures"] == 5
    r2 = _e2e(s, prov=H.ENGINE, key="eng")
    assert r2.lane == "engine" and len(ring) == 5 and s.bus.stats["pushed"] == 5
    assert all(not ts.ephemeral for ts in ring.latest(5))
    r3 = _e2e(s, prov=P.as_record(["EXTGEN:test"]), key="x")
    assert len(ring) == 5 and r3.ext is True


# ----------------------------------------------------------------------------------------- import 위생

def test_import_is_light_and_side_effect_free():
    code = (
        "import sys, torch\n"
        "st = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,\n"
        "      torch.backends.cudnn.benchmark, torch.initial_seed())\n"
        "import buc.read.mrh_call as m\n"
        "assert st == (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,\n"
        "              torch.backends.cudnn.benchmark, torch.initial_seed()), 'global state changed'\n"
        "bad = [x for x in ('diffusers', 'transformers', 'huggingface_hub', 'safetensors') if x in sys.modules]\n"
        "assert not bad, bad\n"
        "assert not torch.cuda.is_initialized()\n"
        "for n in ('read_mrh', 'MrhReader', 'ReadConfig', 'MrhReadout', 'ViewRead', 'TileBox', 'plan_tiles',\n"
        "          'tile_weights', 'blend_reads', 'fuse_global', 'affine_fit', 'affine_residual', 'global_size',\n"
        "          'lane_for', 'MrhCallError', 'MrhCallSpecError', 'CleanroomError'):\n"
        "    assert hasattr(m, n), n\n"
    )
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
