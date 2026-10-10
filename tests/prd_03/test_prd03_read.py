"""PRD-03 classic_read: 전달함수·평면 깊이 오라클, 고정 필드, 계약 검사, params."""
import copy

import numpy as np
import pytest

from buc.core import contracts as C
from prd03_helpers import call, ground_depth, make_ctx, make_image, srgb_eotf


def read_of(img, params=None, seed=12345):
    return call("S-1", {"image": img}, params, seed)["read"]


def test_output_contract_and_fixed_fields():
    img = make_image(24, 40, seed=1)
    out = call("S-1", {"image": img})
    assert set(out) == {"read"}
    r = out["read"]
    assert isinstance(r, C.ReadPacket)
    r.validate()
    C.check_stage_io("S-1", {"image": img}, out)
    assert r.input_hash == img.h_I
    assert r.prov == img.prov and r.prov.taint >= img.prov.taint
    assert r.level == "L1" and r.scale_s == 1.0 and r.sigma_s == 1.0
    assert r.fallbacks == ("stub:classic_read",)
    assert r.meta == {"stub": "classic_read"}
    assert r.depth_conf is not None and r.depth_conf.shape == (24, 40) and not r.depth_conf.any()
    assert r.normals is None and r.esc == {} and r.faces == () and r.lights == {} and r.dyn == {}
    assert r.image_lin.dtype == np.float32 and r.image_lin.shape == (24, 40, 3)
    assert r.depth.dtype == np.float32 and r.depth.shape == (24, 40)


def test_default_theta_and_obs():
    h, w = 24, 40
    r = read_of(make_image(h, w))
    want = C.Theta(aspect=w / h, focal_mm=35.0)
    assert r.theta_in.content_hash() == want.content_hash()
    assert r.theta_in.focal_mm == 35.0 and r.theta_in.fps == 24 and r.theta_in.f_stop == 4
    assert (r.obs.width, r.obs.height) == (w, h)
    assert np.allclose(r.obs.K, want.intrinsics(w, h), atol=1e-9)
    assert np.allclose(r.obs.K[:2, 2], [(w - 1) / 2, (h - 1) / 2], atol=1e-12)
    assert np.array_equal(r.obs.R, np.eye(3)) and not r.obs.t.any()
    assert r.obs.k1 == 0 and r.obs.k2 == 0 and r.obs.mask is None


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("cs", ["srgb", "display-p3", "rec2020", "unknown"])
def test_srgb_transfer_family(dtype, cs):
    img = make_image(12, 20, seed=3, dtype=dtype, cs=cs)
    px = img.pixels.astype(np.float64)
    px = px / {"uint8": 255.0, "uint16": 65535.0, "float32": 1.0}[dtype]
    r = read_of(img)
    assert np.allclose(r.image_lin, srgb_eotf(px), atol=2e-6, rtol=1e-5)


def test_linear_and_adobe_transfer():
    img = make_image(12, 20, seed=4, cs="linear-srgb")
    assert np.allclose(read_of(img).image_lin, img.pixels / 255.0, atol=2e-7)
    img = make_image(12, 20, seed=5, cs="adobe-rgb")
    assert np.allclose(read_of(img).image_lin, (img.pixels / 255.0) ** 2.2, atol=2e-6)


def test_transfer_known_points_and_negative_float_clipped():
    px = np.zeros((2, 2, 3), np.float32)
    px[0, 0] = 1.0
    px[0, 1] = 0.5
    px[1, 0] = -0.25
    px[1, 1] = 0.04
    img = C.InputImage(pixels=px, color_space="srgb", prov=make_image().prov)
    r = read_of(img)
    assert np.allclose(r.image_lin[0, 0], 1.0, atol=1e-6)
    assert np.allclose(r.image_lin[0, 1], 0.21404114, atol=1e-6)       # sRGB(0.5)
    assert np.all(r.image_lin[1, 0] == 0.0)
    assert np.allclose(r.image_lin[1, 1], 0.04 / 12.92, atol=1e-8)


@pytest.mark.parametrize("h,w", [(24, 40), (25, 33), (8, 12), (1, 5)])
def test_ground_plane_depth_oracle(h, w):
    img = make_image(h, w, seed=h)
    r = read_of(img)
    fy = float(r.obs.K[1, 1])
    want = ground_depth(h, w, fy)
    assert np.allclose(r.depth, want, rtol=2e-6, atol=0)
    assert np.isfinite(r.depth).all() and (r.depth > 0).all()
    assert (r.depth == r.depth[:, :1]).all()                             # 열마다 같다
    cy = (h - 1) / 2.0
    for v in range(h):
        if v <= cy:
            assert r.depth[v, 0] == np.float32(50.0)                     # 수평선 이상 = z_far (행 cy 포함)
    below = r.depth[:, 0][np.arange(h) > cy]
    assert np.all(np.diff(below) <= 1e-6)                                 # 아래로 갈수록 가까워진다


def test_depth_values_hand_checked():
    # 24x40: fy = 40*35/24.89, cy = 11.5. 마지막 행 v=23 -> d=11.5 -> z = fy*1.5/11.5
    r = read_of(make_image(24, 40))
    fy = 40 * 35.0 / 24.89
    assert r.depth[23, 5] == pytest.approx(fy * 1.5 / 11.5, rel=1e-6)
    assert r.depth[12, 5] == pytest.approx(min(fy * 1.5 / 0.5, 50.0), rel=1e-6)   # 수평선 바로 아래: 상한에 걸림
    assert r.depth[11, 5] == pytest.approx(50.0)


def test_params_override():
    img = make_image(24, 40)
    base = read_of(img)
    big = read_of(img, {"cam_height_m": 3.0, "z_far_m": 500.0, "z_near_m": 0.1})
    fy = float(base.obs.K[1, 1])
    assert np.allclose(big.depth, ground_depth(24, 40, fy, 3.0, 0.1, 500.0), rtol=2e-6)
    # 하한이 걸리는 경우: 가장 아래 행이 z_near_m 으로 잘린다
    near = read_of(img, {"z_near_m": 30.0, "z_far_m": 60.0})
    assert near.depth[:, 0].min() == np.float32(30.0)
    tele = read_of(img, {"focal_mm": 70.0})
    assert tele.theta_in.focal_mm == 70.0 and tele.obs.K[0, 0] == pytest.approx(2 * base.obs.K[0, 0], rel=1e-12)
    assert tele.depth.dtype == np.float32


@pytest.mark.parametrize("params", [
    {"bogus": 1.0}, {"focal_mm": 0.0}, {"cam_height_m": -1.0}, {"z_near_m": 10.0, "z_far_m": 5.0},
    {"z_far_m": float("inf")}, {"z_near_m": 0.0}, {"focal_mm": float("nan")}])
def test_bad_params_rejected(params):
    with pytest.raises(ValueError):
        read_of(make_image(8, 12), params)


def test_inputs_not_mutated_and_deterministic():
    img = make_image(16, 24, seed=9)
    before = img.content_hash()
    a = read_of(img, seed=1)
    b = read_of(copy.deepcopy(img), seed=999)                             # _seed·ctx.seed 와 무관
    assert img.content_hash() == before
    assert a.content_hash() == b.content_hash()
    c = read_of(img, {"_seed": 5})
    assert c.content_hash() == a.content_hash()


def test_prov_passthrough_keeps_all_taint():
    prov = C.Prov(taint=frozenset({"USER", "EXTGEN:soul"}), mode="mixed", roots=("r1",), ns="user_vault")
    img = make_image(8, 12, prov=prov)
    r = read_of(img)
    assert r.prov == prov and r.prov.ext


def test_make_ctx_helper_matches_spine_params_shape():
    ctx = make_ctx("S-1", {"focal_mm": 50.0})
    assert ctx.params["_seed"] == ctx.seed and ctx.impl.kind == "stub"
