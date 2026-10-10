"""RD-21 기본 단계: 실제 RD-01/RD-02/RD-08 과 가짜 읽기기로 돈 실제 RD-07 read_mrh."""
import math

import numpy as np
import pytest

from buc.core import contracts as C
from buc.read import pipeline as P
from buc.read.geom_cam import estimate_geom_camera
from buc.read.ingest import IngestOpts, ingest
from buc.read.metric_scale import SIGMA_LOG_F, SizeObs, estimate_metric_scale
from rd21_helpers import USER_PROV, make_image, smooth_image, status_map, up_torch

LD_CONST = 0.7


class FakeReader:
    """RD-07 읽기기 규약: snap(sigma), __call__(image, sigma, *, prov, key) -> ViewRead."""
    cache_id = "fake-reader-v1"

    def __init__(self):
        self.n = 0

    def snap(self, sigma):
        return float(sigma)

    def __call__(self, image, sigma, *, prov, key):
        import torch
        from buc.read.mrh_call import ViewRead
        self.n += 1
        h, w = image.shape[-2:]
        return ViewRead(outputs={"log_depth": torch.full((1, 1, h // 4, w // 4), LD_CONST)}, sigma=sigma,
                        digest=bytes(32), ephemeral=True)


class SizeFn:
    cache_id = "sizes-v1"

    def __init__(self, objs):
        self.objs = objs

    def __call__(self, state):
        return list(self.objs)


def test_default_steps_shape():
    st = P.default_steps()
    assert tuple(s.sid for s in st) == ("R0", "R1", "R6")
    by = {s.sid: s for s in st}
    assert (by["R0"].name, by["R0"].version, by["R0"].needs) == ("ingest", "0.1.0", ())
    assert (by["R1"].name, by["R1"].version, by["R1"].needs) == ("geom_cam", "0.1.0", ("R0",))
    assert (by["R6"].name, by["R6"].version) == ("metric_scale", "0.1.0+none")
    assert set(by["R6"].needs) == {"R1", "R5"}
    st2 = {s.sid: s for s in P.default_steps(reader=FakeReader(), size_obs_fn=SizeFn([]))}
    assert tuple(st2) == ("R0", "R1", "R5", "R6")
    assert (st2["R5"].name, st2["R5"].version, st2["R5"].needs) == ("mrh_call", "0.1.0+fake-reader-v1", ("R0",))
    assert st2["R6"].version == "0.1.0+sizes-v1"


def test_default_steps_require_cache_ids():
    class NoId:
        def snap(self, s):
            return s

        def __call__(self, *a, **k):
            raise AssertionError

    with pytest.raises(ValueError):
        P.default_steps(reader=NoId())
    r = FakeReader()
    r.cache_id = ""
    with pytest.raises(ValueError):
        P.default_steps(reader=r)
    with pytest.raises(ValueError):
        P.default_steps(size_obs_fn=lambda s: [])


def test_default_cpu_read_matches_modules():
    img = smooth_image()
    res = P.run_read(img)
    pk = res.packet
    ing = ingest(img.pixels, color_space="srgb", prov=img.prov, consent=img.consent,
                 opts=IngestOpts(letterbox=False, apply_orientation=False))
    np.testing.assert_array_equal(pk.image_lin, ing.image_lin)
    assert pk.meta["input"] == ing.meta_input()
    g = estimate_geom_camera(ing.image_lin, fov_prior_deg=ing.fov_x_prior_deg)
    assert pk.meta["read"]["geom"] == g.to_meta()
    np.testing.assert_allclose(pk.obs.K, g.K, atol=1e-9)
    assert res.outputs["R1"]["f_source"] == g.f_source
    ms = estimate_metric_scale([], sigma_log_f=SIGMA_LOG_F[g.f_source])
    assert pk.meta["read"]["scale"] == ms.to_meta()
    assert pk.fallbacks == ("R5:skipped",)
    assert pk.meta["read"]["depth_source"] == "plane"
    assert pk.scale_s == 1.0 and pk.sigma_s == 1.5
    st = status_map(res)
    assert st["R0"] == st["R1"] == st["R6"] == "ok"
    C.check_stage_io("S-1", {"image": img}, {"read": pk})


def test_unknown_color_space_is_assumed_srgb():
    img = smooth_image(cs="unknown")
    res = P.run_read(img)
    assert status_map(res)["R0"] == "ok"
    ref = P.run_read(smooth_image(cs="srgb")).packet
    np.testing.assert_array_equal(res.packet.image_lin, ref.image_lin)


def test_tiny_image_geom_fallback():
    img = make_image(h=24, w=24)
    res = P.run_read(img)
    pk = res.packet
    assert pk.fallbacks == ("R1:geom_cam@0.1.0:GeomCamError", "R5:skipped")
    C.check_stage_io("S-1", {"image": img}, {"read": pk})


def test_bad_step_params_fall_back_not_raise():
    res = P.run_read(smooth_image(), params={"R1": {"no_such_opt": 1}})
    assert res.packet.fallbacks[0].startswith("R1:geom_cam@0.1.0:")
    assert status_map(res)["R0"] == "ok"


def test_r0_rejects_unknown_params():
    res = P.run_read(smooth_image(), params={"R0": {"letterbox": True}})
    assert res.packet.fallbacks[0].startswith("R0:ingest@0.1.0:")


def test_default_r5_with_real_read_mrh():
    img = make_image(h=64, w=96, prov=USER_PROV)
    reader = FakeReader()
    res = P.run_read(img, steps=P.default_steps(reader=reader),
                     params={"R5": {"heads": ["log_depth"], "tiles": False}})
    pk = res.packet
    assert pk.fallbacks == ()
    assert reader.n >= 1
    assert pk.meta["read"]["depth_source"] == "mrh"
    np.testing.assert_allclose(pk.depth, math.exp(LD_CONST) * pk.scale_s, rtol=1e-5)
    np.testing.assert_allclose(pk.depth_conf, 1.0, atol=1e-6)
    mrh = pk.meta["read"]["mrh"]
    assert mrh["lane"] == "cleanroom" and mrh["ext"] is True
    assert "USER-DERIVED" in mrh["prov"]["taint"]
    assert pk.prov.content_hash() == img.prov.content_hash()
    C.check_stage_io("S-1", {"image": img}, {"read": pk})


def test_size_obs_fn_drives_scale():
    objs = [SizeObs(cls="chair", h_rel=0.4, sigma_h=0.05), SizeObs(cls="chair", h_rel=0.45, sigma_h=0.05),
            SizeObs(cls="door", h_rel=1.0, sigma_h=0.05)]
    img = make_image(h=64, w=96)
    res = P.run_read(img, steps=P.default_steps(reader=FakeReader(), size_obs_fn=SizeFn(objs)),
                     params={"R5": {"heads": ["log_depth"], "tiles": False}})
    pk = res.packet
    f_source = res.outputs["R1"]["f_source"]
    ms = estimate_metric_scale(objs, sigma_log_f=SIGMA_LOG_F[f_source])
    assert math.isclose(pk.scale_s, ms.s, rel_tol=1e-12)
    assert math.isclose(pk.sigma_s, ms.sigma_s, rel_tol=1e-12)
    assert pk.meta["read"]["scale"] == ms.to_meta()
    np.testing.assert_allclose(pk.depth, np.maximum(ms.s * np.exp(up_torch(np.full((16, 24), LD_CONST), 64, 96)),
                                                    1e-3), rtol=1e-5)


def test_reader_failure_isolated():
    class Boom(FakeReader):
        def __call__(self, *a, **k):
            raise RuntimeError("gpu gone")

    img = make_image(h=64, w=96)
    res = P.run_read(img, steps=P.default_steps(reader=Boom()))
    assert res.packet.fallbacks == ("R5:mrh_call@0.1.0+fake-reader-v1:RuntimeError",)
    assert res.packet.meta["read"]["depth_source"] == "plane"
    assert status_map(res)["R6"] == "ok"
