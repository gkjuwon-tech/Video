"""RD-21 패킷 조립 수식: 깊이·신뢰도·스케일·카메라·메타."""
import json

import numpy as np

from buc.core import contracts as C
from buc.read import pipeline as P
from rd21_helpers import (FAKE_FOV, FAKE_K1, FAKE_S, FAKE_SIGMA, EXTGEN_PROV, fake_ld, fake_outputs, fake_rdis,
                          fake_steps, k_from_fov, make_image, up_torch)

H, W = 48, 64


def test_full_packet_formulas():
    img = make_image(h=H, w=W)
    res = P.run_read(img, steps=fake_steps())
    pk = res.packet
    ld, rd = fake_ld(H // 4, W // 4), fake_rdis(H // 4, W // 4)
    np.testing.assert_allclose(pk.depth, np.maximum(FAKE_S * np.exp(up_torch(ld, H, W)), 1e-3), rtol=1e-5)
    np.testing.assert_allclose(pk.depth_conf, np.clip(1.0 - up_torch(rd, H, W), 0, 1), atol=1e-5)
    assert pk.depth.dtype == np.float32 and pk.depth_conf.dtype == np.float32
    assert pk.scale_s == FAKE_S and pk.sigma_s == FAKE_SIGMA
    K = k_from_fov(FAKE_FOV, W, H)
    np.testing.assert_allclose(pk.obs.K, K, atol=1e-9)
    np.testing.assert_allclose(pk.obs.R, np.eye(3))
    np.testing.assert_allclose(pk.obs.t, np.zeros(3))
    assert (pk.obs.width, pk.obs.height) == (W, H)
    assert pk.obs.k1 == FAKE_K1 and pk.obs.k2 == 0.0
    assert abs(pk.theta_in.focal_mm - K[0, 0] / W * 36.0) < 1e-9
    assert pk.theta_in.sensor_width_mm == 36.0
    assert abs(pk.theta_in.fov_x_deg() - FAKE_FOV) < 1e-9
    assert pk.level == "L1.5"
    assert pk.normals is None
    assert pk.input_hash == img.h_I
    assert pk.prov.content_hash() == img.prov.content_hash()
    assert pk.fallbacks == ()
    C.check_stage_io("S-1", {"image": img}, {"read": pk})


def test_meta_structure_and_json():
    img = make_image(h=H, w=W)
    pk = P.run_read(img, steps=fake_steps()).packet
    m = pk.meta
    assert set(m) == {"input", "read"}
    assert m["input"] == {"fake": "R0"}
    r = m["read"]
    assert r["pipeline"] == "rd21_read@0.1.0"
    assert r["depth_source"] == "mrh"
    assert r["geom"] == {"fake": "R1"}
    assert r["mrh"] == {"fake": "R5"}
    assert r["scale"] == {"fake": "R6"}
    assert set(r["extra"]) == {"R2", "R3", "R4", "R7", "R8", "R9", "R10", "R11", "R12", "R13"}
    assert r["extra"]["R9"] == {"fake": "R9"}
    assert len(r["steps"]) == 14
    assert all(v == {"impl": f"fake_{k}@1", "status": "ok"} for k, v in r["steps"].items())
    json.dumps(pk.to_dict(), allow_nan=False)
    back = C.loads(C.dumps(pk))
    assert back.content_hash() == pk.content_hash()


def test_depth_floor_and_conf_clip():
    def r5(state, params):
        out = dict(fake_outputs("R5", state))
        out["log_depth"] = np.full((12, 16), -30.0, np.float32)
        out["r_dis_map"] = np.full((12, 16), 3.0, np.float32)
        return out

    def r6(state, params):
        return {"s": 1.0, "sigma_s": 0.2, "reliable": False, "meta": {}}

    pk = P.run_read(make_image(h=H, w=W), steps=fake_steps(override={"R5": r5, "R6": r6})).packet
    np.testing.assert_allclose(pk.depth, 1e-3, rtol=1e-6)
    np.testing.assert_array_equal(pk.depth_conf, np.zeros((H, W), np.float32))


def test_depth_from_coarse_grid_of_any_size():
    def r5(state, params):
        out = dict(fake_outputs("R5", state))
        out["log_depth"] = fake_ld(5, 7)
        out["r_dis_map"] = np.zeros((5, 7), np.float32)
        return out

    pk = P.run_read(make_image(h=H, w=W), steps=fake_steps(override={"R5": r5})).packet
    np.testing.assert_allclose(pk.depth, FAKE_S * np.exp(up_torch(fake_ld(5, 7), H, W)), rtol=1e-5)
    np.testing.assert_allclose(pk.depth_conf, 1.0, atol=1e-6)


def test_ext_prov_passes_through_unchanged():
    img = make_image(h=H, w=W, prov=EXTGEN_PROV)
    pk = P.run_read(img, steps=fake_steps()).packet
    assert pk.prov.content_hash() == EXTGEN_PROV.content_hash()
    C.check_stage_io("S-1", {"image": img}, {"read": pk})


def test_packet_deterministic():
    img = make_image(h=H, w=W, seed=4)
    a = P.run_read(img, steps=fake_steps()).packet
    b = P.run_read(img, steps=fake_steps()).packet
    assert a.content_hash() == b.content_hash()
