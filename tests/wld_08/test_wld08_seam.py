"""WLD-08 이음매: 실제 TeacherSampler(TCH-02) + 가짜 변환기, S2 스테이지(spine), WLD-03 입력."""
import importlib.util
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from buc.core import contracts as C
from buc.pipeline import spine
from buc.teacher.sample import TeacherSampler
from buc.world.depth_world import DepthWorldConfig, build_depth_world

import wld08_helpers as h

S = h.mod()

_FAKES = pathlib.Path(__file__).resolve().parents[1] / "tch_02" / "tch02_fakes.py"


def _load_tch02_fakes():
    spec = importlib.util.spec_from_file_location("tch02_fakes_for_wld08", _FAKES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PoolCodec:
    """손실 코덱: 8x8 평균 풀링을 앞 3채널에 담고, 디코드는 쌍선형 업샘플. VaeCodec 와 같은 시그니처."""

    def encode(self, x, *, out_dtype=torch.float32):
        b, _, hh, ww = x.shape
        z = torch.zeros(b, 16, hh // 8, ww // 8)
        z[:, :3] = F.avg_pool2d(x.float(), 8)
        return z.to(out_dtype)

    def decode(self, z, *, clamp=True, out_dtype=torch.float32):
        y = F.interpolate(z[:, :3].float(), scale_factor=8, mode="bilinear", align_corners=False)
        return (y.clamp(-1.0, 1.0) if clamp else y).to(out_dtype)


def _real_sampler(tmp_path):
    fakes = _load_tch02_fakes()
    teacher = fakes.make_teacher(tmp_path, mode="affine")
    teacher.vae = PoolCodec()
    return teacher, TeacherSampler(teacher)


def test_real_teacher_sampler_seam(tmp_path):
    teacher, sampler = _real_sampler(tmp_path / "a")
    _, dw = h.build()
    res = S.sdedit_avg_fill(dw, sampler, seed=5)
    calls = teacher.transformer.calls
    assert len(calls) == 2, "σ=0.5 에서 표 끝까지 2 step, step 마다 전 뷰를 한 번에"
    assert [c.n for c in calls] == [4, 4]
    pts = sampler.table.points
    assert calls[0].t.tolist() == pytest.approx([1.0 - pts[6]] * 4, abs=1e-6)
    assert calls[1].t.tolist() == pytest.approx([1.0 - pts[7]] * 4, abs=1e-6)
    assert all(c.t_dtype == torch.float32 for c in calls)
    assert all(s == (16, 1, h.H // 8, h.W // 8) for c in calls for s in c.x_shapes)
    # 결과 세계는 유효하고 관측 영역은 비트 동일
    a, b = dw.world, res.world
    b.validate()
    assert h.layer_bytes_equal(a.layers[0], b.layers[0])
    assert np.array_equal(a.layers[1].rgba[~dw.hidden], b.layers[1].rgba[~dw.hidden])
    assert res.filled.sum() >= 0.6 * dw.hidden.sum()
    px = b.layers[1].rgba[res.filled][:, :3]
    assert np.isfinite(px).all() and px.min() >= 0 and px.max() <= 1
    assert np.any(b.layers[1].rgba[res.filled] != a.layers[1].rgba[res.filled])
    # 같은 시드 → 같은 세계, 다른 시드 → 다른 색
    t2, s2 = _real_sampler(tmp_path / "b")
    again = S.sdedit_avg_fill(dw, s2, seed=5)
    assert again.strata.merkle_root() == res.strata.merkle_root()
    t3, s3 = _real_sampler(tmp_path / "c")
    other = S.sdedit_avg_fill(dw, s3, seed=6)
    assert other.strata.merkle_root() != res.strata.merkle_root()
    assert np.array_equal(other.filled, res.filled)


def _const(imgs):
    return torch.full_like(imgs, -0.2)


def test_s2_stage_impl_matches_direct_call_and_passes_stage_io():
    read, _ = h.build()
    sam = h.FakeSampler(_const)
    impl = S.make_world_impl(sam)
    assert isinstance(impl, spine.Impl)
    assert (impl.stage, impl.kind, impl.name, impl.version) == ("S2", "real", "sdedit_avg", "0.1.0")
    ctx = spine.StageContext(stage="S2", seed=1, params={"_seed": 5, "max_angle_deg": 15.0, "n_per_side": 1}, impl=impl)
    out = impl.fn({"read": read}, ctx)
    assert set(out) == {"world"}
    C.check_stage_io("S2", {"read": read}, out)
    assert len(sam.calls) == 1 and sam.calls[0].seeds == 5 and sam.calls[0].x0_shape[0] == 2
    ref = S.sdedit_avg_fill(build_depth_world(read), h.FakeSampler(_const),
                            cfg=S.SDEditAvgConfig(max_angle_deg=15.0, n_per_side=1), seed=5).world
    assert out["world"].content_hash() == ref.content_hash()
    assert out["world"].meta["sdedit_avg"]["n_views"] == 2
    assert "depth_world" in out["world"].meta


def test_s2_stage_depth_cfg_and_bad_params():
    read, _ = h.build()
    sam = h.FakeSampler(_const)
    impl = S.make_world_impl(sam, depth_cfg=DepthWorldConfig(ext_px=8.0, tsq_stride=3))
    ctx = spine.StageContext(stage="S2", seed=1, params={"_seed": 2}, impl=impl)
    out = impl.fn({"read": read}, ctx)["world"]
    ref = S.sdedit_avg_fill(build_depth_world(read, cfg=DepthWorldConfig(ext_px=8.0, tsq_stride=3)),
                            h.FakeSampler(_const), seed=2).world
    assert out.content_hash() == ref.content_hash()
    bad = spine.StageContext(stage="S2", seed=1, params={"_seed": 2, "bogus": 1}, impl=impl)
    with pytest.raises(ValueError):
        impl.fn({"read": read}, bad)
    worse = spine.StageContext(stage="S2", seed=1, params={"_seed": 2, "n_per_side": 0}, impl=impl)
    with pytest.raises(ValueError):
        impl.fn({"read": read}, worse)
    with pytest.raises(TypeError):
        S.make_world_impl(sam, depth_cfg={"ext_px": 8.0})


def test_stage_world_flows_into_registry():
    read, _ = h.build()
    reg = spine.Registry()
    impl = S.make_world_impl(h.FakeSampler(_const))
    reg.register(impl)
    assert reg.resolve("S2").ref == "real:sdedit_avg@0.1.0"
