"""RD-21 고정 폴백값 오라클: R0 전달함수, R1 K, 평면 깊이, R6 기본 스케일."""
import math

import numpy as np
import pytest

from buc.core import contracts as C
from buc.read import pipeline as P
from buc.read.ingest import fov_x_prior_deg
from rd21_helpers import (FAKE_FOV, classic_lin, fake_outputs, fake_steps, k_from_fov, make_image, plane_oracle,
                          status_map, up_torch)


def _raise(state, params):
    raise RuntimeError("fail")


@pytest.mark.parametrize("dtype,cs", [("uint8", "srgb"), ("uint16", "adobe-rgb"), ("float32", "linear-srgb"),
                                      ("uint8", "display-p3"), ("uint16", "unknown"), ("float32", "rec2020")])
def test_r0_fallback_matches_classic_transfer(dtype, cs):
    img = make_image(dtype=dtype, cs=cs)
    res = P.run_read(img, steps=fake_steps(override={"R0": _raise}))
    assert status_map(res)["R0"] == "fallback"
    assert res.packet.image_lin.dtype == np.float32
    np.testing.assert_allclose(res.packet.image_lin, classic_lin(img), rtol=1e-6, atol=1e-7)
    assert res.outputs["R0"]["fov_x_prior_deg"] is None


def test_r0_fallback_matches_prd03_stub():
    from buc.pipeline.stubs.pack import stub_impls
    from buc.pipeline import spine as S
    img = make_image(h=30, w=44, dtype="uint16", cs="adobe-rgb")
    impl = stub_impls()["S-1"]
    ctx = S.StageContext(stage="S-1", seed=0, params={"_seed": 0}, impl=impl)
    stub = impl.fn({"image": img}, ctx)["read"]
    res = P.run_read(img, steps=fake_steps(override={"R0": _raise}))
    np.testing.assert_allclose(res.packet.image_lin, stub.image_lin, atol=1e-7)


def test_r1_fallback_default_fov():
    img = make_image(h=48, w=64)
    res = P.run_read(img, steps=fake_steps(override={"R1": _raise}))
    out = res.outputs["R1"]
    np.testing.assert_allclose(out["K"], k_from_fov(50.0, 64, 48), atol=1e-9)
    assert out["f_source"] == "default"
    assert out["k1"] == 0.0 and out["k2"] == 0.0
    np.testing.assert_allclose(out["up_cam"], [0.0, -1.0, 0.0])
    pk = res.packet
    np.testing.assert_allclose(pk.obs.K, k_from_fov(50.0, 64, 48), atol=1e-9)
    assert pk.obs.k1 == 0.0 and pk.obs.k2 == 0.0
    assert pk.theta_in.sensor_width_mm == 36.0
    assert math.isclose(pk.theta_in.fov_x_deg(), 50.0, abs_tol=1e-9)
    assert math.isclose(pk.theta_in.aspect, 64 / 48, rel_tol=1e-12)


def test_r1_fallback_uses_exif_fov_prior_after_r0_fallback():
    img = make_image(h=40, w=60, exif={"focal_35mm": 28.0})
    res = P.run_read(img, steps=fake_steps(override={"R0": _raise, "R1": _raise}))
    fov = fov_x_prior_deg(28.0, 60, 40)
    assert math.isclose(res.outputs["R0"]["fov_x_prior_deg"], fov, rel_tol=1e-12)
    np.testing.assert_allclose(res.outputs["R1"]["K"], k_from_fov(fov, 60, 40), atol=1e-9)
    assert res.outputs["R1"]["f_source"] == "prior"


@pytest.mark.parametrize("f35", [0.0, -5.0, "x", None])
def test_bad_exif_focal_ignored(f35):
    img = make_image(h=40, w=60, exif={"focal_35mm": f35})
    res = P.run_read(img, steps=fake_steps(override={"R0": _raise, "R1": _raise}))
    assert res.outputs["R0"]["fov_x_prior_deg"] is None
    assert res.outputs["R1"]["f_source"] == "default"


def test_r1_skipped_uses_r0_prior():
    def r0(state, params):
        out = dict(fake_outputs("R0", state))
        out["fov_x_prior_deg"] = 72.0
        return out

    res = P.run_read(make_image(), steps=fake_steps(sids=("R0", "R5", "R6"), override={"R0": r0}))
    np.testing.assert_allclose(res.packet.obs.K, k_from_fov(72.0, 64, 48), atol=1e-9)
    assert res.outputs["R1"]["meta"] == {"status": "skipped"}
    assert res.packet.fallbacks == ("R1:skipped",)


def test_r5_fallback_plane_depth_with_r1_geometry():
    pitch, roll = math.radians(-8.0), math.radians(3.0)
    up = np.array([math.cos(pitch) * math.sin(roll), -math.cos(pitch) * math.cos(roll), math.sin(pitch)])

    def r1(state, params):
        out = dict(fake_outputs("R1", state))
        out["up_cam"] = up
        return out

    img = make_image(h=36, w=52)
    res = P.run_read(img, steps=fake_steps(override={"R5": _raise, "R1": r1}))
    pk = res.packet
    exp = plane_oracle(k_from_fov(FAKE_FOV, 52, 36), up, 36, 52)
    np.testing.assert_allclose(pk.depth, exp, rtol=1e-5)
    np.testing.assert_array_equal(pk.depth_conf, np.zeros((36, 52), np.float32))
    assert pk.scale_s == 1.0 and pk.sigma_s == 1.5
    assert pk.meta["read"]["depth_source"] == "plane"
    assert pk.fallbacks == ("R5:fake_R5@1:RuntimeError",)


def test_plane_depth_respects_opts():
    opts = P.ReadOpts(cam_height_m=2.0, z_near_m=0.2, z_far_m=30.0, default_sigma_s=0.9)
    res = P.run_read(make_image(), steps=fake_steps(sids=("R0", "R1", "R6")), opts=opts)
    pk = res.packet
    exp = plane_oracle(k_from_fov(FAKE_FOV, 64, 48), [0, -1, 0], 48, 64, h=2.0, zn=0.2, zf=30.0)
    np.testing.assert_allclose(pk.depth, exp, rtol=1e-5)
    assert pk.sigma_s == 0.9 and pk.scale_s == 1.0
    assert pk.fallbacks == ("R5:skipped",)


def test_r6_fallback_with_mrh_depth():
    img = make_image()
    res = P.run_read(img, steps=fake_steps(override={"R6": _raise}))
    pk = res.packet
    out6 = res.outputs["R6"]
    assert out6["s"] == 1.0 and out6["sigma_s"] == 1.5 and out6["reliable"] is False
    ld = fake_outputs("R5", type("S", (), {"H": 48, "W": 64})())["log_depth"]
    exp = np.maximum(np.exp(up_torch(ld, 48, 64)), 1e-3)
    np.testing.assert_allclose(pk.depth, exp, rtol=1e-5)
    assert pk.scale_s == 1.0 and pk.sigma_s == 1.5
    assert pk.meta["read"]["depth_source"] == "mrh"
    C.check_stage_io("S-1", {"image": img}, {"read": pk})


def test_skipped_meta_and_fallback_meta():
    res = P.run_read(make_image(), steps=fake_steps(sids=("R0", "R1", "R5", "R6", "R7"),
                                                    override={"R7": _raise}))
    assert res.outputs["R2"]["meta"] == {"status": "skipped"}
    assert res.outputs["R7"]["meta"] == {"status": "fallback"}
    assert res.outputs["R11"]["meta"] == {"status": "skipped"}
    assert res.packet.level == "L1"
    assert res.packet.fallbacks == ("R7:fake_R7@1:RuntimeError",)


def test_everything_failing_still_yields_valid_packet():
    from rd21_helpers import ALL_SIDS
    img = make_image(h=33, w=47, dtype="uint16", cs="display-p3")
    res = P.run_read(img, steps=fake_steps(override={s: _raise for s in ALL_SIDS}))
    pk = res.packet
    assert len(pk.fallbacks) == 14
    C.check_stage_io("S-1", {"image": img}, {"read": pk})
    np.testing.assert_allclose(pk.image_lin, classic_lin(img), atol=1e-7)
    np.testing.assert_allclose(pk.depth, plane_oracle(k_from_fov(50.0, 47, 33), [0, -1, 0], 33, 47), rtol=1e-5)
    assert pk.level == "L1" and pk.normals is None
    assert pk.esc == {} and pk.lights == {} and pk.faces == () and pk.dyn == {}
