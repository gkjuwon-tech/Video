"""FND-08 카탈로그·불변식: 10종 이름, 형식, FND-02 계약 통과, 결정론, 해시, 이음매(InputImage/ReadPacket/StageIO)."""
import dataclasses

import numpy as np
import pytest

from buc.core import color, contracts, geom

import fnd08_support as S

NAMES = S.NAMES


def test_catalog_names_and_constants():
    t = S.tool()
    assert tuple(t.SCENE_NAMES) == NAMES and len(set(NAMES)) == 10
    assert t.FIXTURE_VERSION == "1"
    assert (t.FIX_W, t.FIX_H) == (96, 64)
    np.testing.assert_allclose(t.LIGHT_DIR, S.LIGHT, atol=1e-15)
    assert t.AMBIENT == 0.2 and tuple(t.SKY_LIN) == (0.25, 0.40, 0.75) and t.CHECK_M == 0.25
    assert tuple(t.CHECK_A) == (0.8, 0.8, 0.8) and tuple(t.CHECK_B) == (0.1, 0.1, 0.1)
    b = S.built()
    assert list(b) == list(NAMES)
    for n, sc in b.items():
        assert sc.name == n
        assert len(sc.views) == (2 if n == "two_view_parallax" else 1)
    with pytest.raises(KeyError):
        t.build_scene("no_such_scene")


@pytest.mark.parametrize("name", NAMES)
def test_array_formats_and_value_ranges(name):
    sc = S.built()[name]
    for v in sc.views:
        assert isinstance(v.obs, contracts.Obs)
        assert (v.obs.width, v.obs.height) == (96, 64)
        for arr, dt, shp in [(v.image_lin, np.float32, (64, 96, 3)), (v.depth, np.float32, (64, 96)),
                             (v.normals, np.float32, (64, 96, 3)), (v.instance, np.int32, (64, 96))]:
            assert arr.dtype == dt and arr.shape == shp and arr.flags["C_CONTIGUOUS"]
        assert np.isfinite(v.image_lin).all() and v.image_lin.min() >= 0
        assert not np.isnan(v.depth).any() and (v.depth > 0).all()
        sky = ~np.isfinite(v.depth)
        np.testing.assert_array_equal(sky, v.instance == 0)
        assert np.isfinite(v.normals).all()
        np.testing.assert_allclose(np.linalg.norm(v.normals.astype(np.float64), axis=-1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(v.normals[sky], np.broadcast_to(np.float32([0, 0, -1]), (int(sky.sum()), 3)))
        assert v.instance.min() >= 0 and (~sky).mean() > 0.3          # 장면이 화면을 충분히 채운다


@pytest.mark.parametrize("name", NAMES)
def test_cameras_match_spec_and_normals_face_camera(name):
    sc = S.built()[name]
    for vi, v in enumerate(sc.views):
        K, R, t, eye, k1, k2 = S.cam_pose(name, vi)
        o = v.obs
        np.testing.assert_allclose(o.K, K, rtol=0, atol=1e-9)
        assert o.K[0, 0] == pytest.approx(o.K[1, 1], abs=1e-12) and o.K[0, 1] == 0
        np.testing.assert_allclose([o.K[0, 2], o.K[1, 2]], [47.5, 31.5], atol=1e-12)
        np.testing.assert_allclose(o.R, R, atol=1e-12)
        np.testing.assert_allclose(o.t, t, atol=1e-12)
        np.testing.assert_allclose(o.center(), eye, atol=1e-12)
        assert (o.k1, o.k2) == (k1, k2)
        assert geom.is_rotation(o.R, atol=1e-9)
        _, _, dc = S.rays(name, vi)
        hit = np.isfinite(v.depth)
        dots = np.einsum("hwc,hwc->hw", v.normals.astype(np.float64), dc)
        assert (dots[hit] < 0).all()


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_and_rng_free(name):
    t = S.tool()
    np.random.seed(123)
    a = t.build_scene(name)
    np.random.seed(456)
    b = t.build_scene(name)
    assert a.content_hash() == b.content_hash() == S.built()[name].content_hash()
    assert len(a.content_hash()) == 64
    for va, vb in zip(a.views, b.views):
        for f in ("image_lin", "depth", "normals", "instance"):
            assert getattr(va, f).tobytes() == getattr(vb, f).tobytes()


def test_hashes_distinct_and_sensitive():
    b = S.built()
    hs = {n: sc.content_hash() for n, sc in b.items()}
    assert len(set(hs.values())) == 10
    sc = b["sphere_lit"]
    v = sc.views[0]
    img = v.image_lin.copy()
    img[10, 10, 1] = np.nextafter(img[10, 10, 1], np.float32(10.0))   # 1 ulp
    assert dataclasses.replace(sc, views=(dataclasses.replace(v, image_lin=img),)).content_hash() != hs["sphere_lit"]
    dep = v.depth.copy()
    dep[0, 0] = 7.0
    assert dataclasses.replace(sc, views=(dataclasses.replace(v, depth=dep),)).content_hash() != hs["sphere_lit"]
    obs2 = contracts.Obs(K=v.obs.K, R=v.obs.R, t=v.obs.t + np.array([1e-9, 0, 0]), width=96, height=64)
    assert dataclasses.replace(sc, views=(dataclasses.replace(v, obs=obs2),)).content_hash() != hs["sphere_lit"]
    gt2 = dict(sc.gt)
    gt2["sphere_r"] = float(gt2["sphere_r"]) + 1e-9
    assert dataclasses.replace(sc, gt=gt2).content_hash() != hs["sphere_lit"]


@pytest.mark.parametrize("name", NAMES)
def test_input_image_and_read_packet_seam(name):
    t = S.tool()
    sc = S.built()[name]
    for vi, v in enumerate(sc.views):
        ii = t.to_input_image(sc, vi)
        assert isinstance(ii, contracts.InputImage) and ii.color_space == "srgb" and ii.pixels.dtype == np.uint8
        np.testing.assert_array_equal(ii.pixels, color.linear_to_srgb8(v.image_lin))
        assert "ENGINE" in ii.prov.taint
        # 8비트 양자화 한도 안에서 선형 영상과 같다(표시 영역 반 코드)
        disp = color.linear_to_display(v.image_lin).astype(np.float64)
        assert np.abs(ii.pixels / 255.0 - disp).max() <= 0.5 / 255 + 1e-6
        rp = t.to_read_packet(sc, vi)
        assert isinstance(rp, contracts.ReadPacket)
        rp.validate()
        assert rp.input_hash == ii.h_I
        np.testing.assert_array_equal(rp.depth, v.depth)
        np.testing.assert_array_equal(rp.image_lin, v.image_lin)
        np.testing.assert_array_equal(rp.normals, v.normals)
        assert rp.obs.content_hash() == v.obs.content_hash()
        assert rp.theta_in.sensor_width_mm == 36.0
        np.testing.assert_allclose(rp.theta_in.intrinsics(96, 64), v.obs.K, rtol=0, atol=1e-9)
        contracts.check_stage_io("S-1", {"image": ii}, {"read": rp})
        # 왕복: 직렬화해도 같은 패킷
        assert contracts.loads(contracts.dumps(rp)).content_hash() == rp.content_hash()
