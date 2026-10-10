"""WLD-17: FarFieldSpec 검증·좌표 규약, build() 오라클·결정론, meta 왕복, 세계(Obs/RD-02) 이음매, STRATA 이음매."""
import dataclasses
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from buc.core import contracts as C
from buc.read import geom_cam
from buc.world import farfield as F
from buc.world import strata as ST
from wld17_helpers import (elev_az_dir, make_layer, make_obs, make_prov, rand_dirs, rot_x, rot_y, rot_z)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def spec(**kw):
    base = dict(up=(0.0, 0.0, 1.0), north=(1.0, 0.0, 0.0), sun_elev_deg=45.0, sun_azim_deg=0.0, turbidity=2.5, albedo=0.3, size=64)
    base.update(kw)
    return F.FarFieldSpec(**base)


# ---- 상수·검증 ----------------------------------------------------------------------------------------------------------
def test_constants_and_error_type():
    assert F.FARFIELD_VERSION == "farfield/v0"
    assert (F.MIN_SIZE, F.MAX_SIZE, F.DEFAULT_SIZE) == (8, 4096, 256)
    assert F.TURBIDITY_RANGE == (1.0, 10.0)
    assert (F.DEFAULT_SUN_ELEV_DEG, F.DEFAULT_SUN_AZIM_DEG) == (45.0, 180.0)
    assert issubclass(F.FarFieldError, C.ContractError) and issubclass(F.FarFieldError, ValueError)
    s = F.FarFieldSpec(up=(0, 0, 1))
    assert (s.size, s.turbidity, s.albedo, s.exposure, s.camera_height_m, s.horizon_blend_deg, s.include_sun) == \
        (256, 2.5, 0.3, 1.0, 1.6, 1.5, False)
    assert s.sun_elev_deg == 45.0 and s.sun_azim_deg == 180.0 and s.ground_albedo_rgb is None


@pytest.mark.parametrize("kw", [
    dict(turbidity=0.5), dict(turbidity=10.5), dict(turbidity=float("nan")), dict(albedo=-0.1), dict(albedo=1.5),
    dict(exposure=0.0), dict(exposure=-1.0), dict(exposure=float("inf")), dict(sun_elev_deg=95.0), dict(sun_elev_deg=-91.0),
    dict(sun_azim_deg=float("nan")), dict(sun_azim_deg=float("inf")), dict(size=4), dict(size=8192), dict(size=64.5), dict(size=True),
    dict(camera_height_m=0.0), dict(camera_height_m=-1.0), dict(horizon_blend_deg=0.0), dict(horizon_blend_deg=float("nan")),
    dict(up=(0, 0, 0)), dict(up=(0, float("nan"), 1)), dict(up=(0, 1)), dict(north=(0, 0, 1)), dict(north=(0, 0, 3)),
    dict(ground_albedo_rgb=(2.0, 0.0, 0.0)), dict(ground_albedo_rgb=(0.1, 0.2)), dict(include_sun=1), dict(turbidity="2.5"),
])
def test_invalid_specs_are_rejected(kw):
    with pytest.raises(F.FarFieldError):
        spec(**kw)


def test_spec_normalizes_and_is_frozen():
    s = F.FarFieldSpec(up=(0.0, 0.0, 2.0), north=(1.0, 0.0, 1.0), sun_azim_deg=370.0)
    np.testing.assert_allclose(s.up, (0, 0, 1), atol=1e-15)
    np.testing.assert_allclose(s.north, (1, 0, 0), atol=1e-15)  # up 에 직교화
    assert s.sun_azim_deg == pytest.approx(10.0, abs=1e-12)
    assert F.FarFieldSpec(up=(0, 0, 1), sun_azim_deg=-30.0).sun_azim_deg == pytest.approx(330.0, abs=1e-12)
    assert isinstance(s.up, tuple) and isinstance(s.north, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.turbidity = 3.0
    hash(s)


def test_default_north_rule():
    # north=None: |up_k| 가 가장 작은 축(동률은 낮은 k)을 up 에 직교화
    np.testing.assert_allclose(F.FarFieldSpec(up=(0, 0, 1)).north, (1, 0, 0), atol=1e-15)
    np.testing.assert_allclose(F.FarFieldSpec(up=(0, -1, 0)).north, (1, 0, 0), atol=1e-15)
    np.testing.assert_allclose(F.FarFieldSpec(up=(1, 0, 0)).north, (0, 1, 0), atol=1e-15)


def test_sun_direction_convention_clockwise_from_above():
    s = F.FarFieldSpec(up=(0, 0, 1), north=(1, 0, 0), sun_elev_deg=30.0, sun_azim_deg=0.0)
    np.testing.assert_allclose(s.sun_dir(), (math.cos(math.radians(30)), 0, 0.5), atol=1e-12)
    e90 = F.FarFieldSpec(up=(0, 0, 1), north=(1, 0, 0), sun_elev_deg=0.0, sun_azim_deg=90.0).sun_dir()
    np.testing.assert_allclose(e90, (0, -1, 0), atol=1e-12)  # 위(+z)에서 내려다볼 때 x -> -y 가 시계방향
    ey = F.FarFieldSpec(up=(0, -1, 0), north=(0, 0, 1), sun_elev_deg=0.0, sun_azim_deg=90.0).sun_dir()
    np.testing.assert_allclose(ey, (1, 0, 0), atol=1e-12)  # y 아래 세계, 북쪽=+z(전방) -> 동쪽=+x(카메라 오른쪽)
    for el, az in ((10.0, 33.0), (-5.0, 200.0), (80.0, 359.0)):
        sd = F.FarFieldSpec(up=(0, 0, 1), north=(1, 0, 0), sun_elev_deg=el, sun_azim_deg=az).sun_dir()
        np.testing.assert_allclose(sd, elev_az_dir(spec(), el, az), atol=1e-12)
        assert np.linalg.norm(sd) == pytest.approx(1.0, abs=1e-12)


# ---- build ------------------------------------------------------------------------------------------------------------
def test_build_is_the_environment_at_texel_centers():
    for s in (spec(), spec(sun_elev_deg=12.0, sun_azim_deg=75.0, turbidity=5.0, up=(0.0, -1.0, 0.0), north=(0.0, 0.0, 1.0)),
              spec(exposure=3.0, ground_albedo_rgb=(0.2, 0.3, 0.1))):
        ff = s.build()
        assert isinstance(ff, C.FarField)
        ff.validate()
        assert ff.rgb.shape == (s.size, s.size, 3) and ff.rgb.dtype == np.float32 and ff.rgb.flags["C_CONTIGUOUS"]
        assert np.isfinite(ff.rgb).all() and ff.rgb.min() >= 0
        want = np.asarray(F.environment_radiance(s, F.texel_dirs(s.size)))
        np.testing.assert_allclose(ff.rgb, want.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_environment_is_finite_nonnegative_over_the_parameter_grid():
    for el in (-20.0, -1.0, 0.0, 3.0, 25.0, 90.0):
        for t in (1.0, 2.5, 10.0):
            for al in (0.0, 1.0):
                ff = spec(sun_elev_deg=el, turbidity=t, albedo=al, size=32).build()
                assert np.isfinite(ff.rgb).all() and ff.rgb.min() >= 0.0, (el, t, al)


def test_build_deterministic_pure_and_independent_of_global_rng():
    s = spec(size=128)
    np.random.seed(1)
    a = s.build().rgb.tobytes()
    np.random.seed(999)
    b = s.build().rgb.tobytes()
    assert a == b
    assert spec(size=128).build().rgb.tobytes() == a  # 같은 값의 다른 인스턴스
    x, y = s.build(), s.build()
    assert x.rgb is not y.rgb  # 호출마다 새 배열
    x.rgb[:] = 0
    assert s.build().rgb.tobytes() == a
    assert spec(size=128, turbidity=2.6).build().rgb.tobytes() != a


def test_build_seam_free_at_back_pole_and_map_edges():
    """뒤통수(-z)·네 변 근처에서 맵 표본이 해석식과 일치한다: 이음매는 보이지 않는다."""
    s = spec(size=128)
    ff = s.build()
    d = np.concatenate([rand_dirs(1, 0) * 0 + np.array([[0.0, 0.0, -1.0]])] +
                       [_near(c, 8.0, 800, i) for i, c in enumerate(((0, 0, -1), (1, 0, -0.3), (-1, 0, -0.3), (0, 1, -0.3), (0, -1, -0.3)))])
    got = F.sample(ff, d)
    want = np.asarray(F.environment_radiance(s, d))
    rel = np.abs(got - want) / np.maximum(want.max(-1, keepdims=True), 1e-9)
    assert rel.max() < 0.02, rel.max()
    # 전 구면에서도(해 주변 8도, 지평선 ±2도 제외)
    dd = rand_dirs(6000, 5)
    up = np.asarray(s.up)
    keep = (np.degrees(np.arccos(np.clip(dd @ s.sun_dir(), -1, 1))) > 8.0) & (np.abs(np.degrees(np.arcsin(dd @ up))) > 2.0)
    dd = dd[keep]
    rel = np.abs(F.sample(ff, dd) - np.asarray(F.environment_radiance(s, dd))) / np.asarray(F.environment_radiance(s, dd)).max(-1, keepdims=True)
    assert np.percentile(rel, 99.5) < 0.03


def _near(c, deg, n, seed):
    from wld17_helpers import dirs_near
    return dirs_near(c, deg, n, seed)


# ---- meta -------------------------------------------------------------------------------------------------------------
def test_meta_roundtrip_is_json_and_rebuilds_identically():
    s = spec(sun_elev_deg=22.5, sun_azim_deg=301.0, turbidity=4.2, albedo=0.55, exposure=1.3, camera_height_m=1.2,
             horizon_blend_deg=2.5, ground_albedo_rgb=(0.2, 0.25, 0.3), include_sun=True, size=32)
    m = s.to_meta()
    assert m["version"] == F.FARFIELD_VERSION
    m2 = json.loads(json.dumps(m, allow_nan=False))
    s2 = F.FarFieldSpec.from_meta(m2)
    assert s2 == s
    assert s2.build().rgb.tobytes() == s.build().rgb.tobytes()
    assert {"up", "north", "sun_elev_deg", "sun_azim_deg", "turbidity", "albedo", "exposure", "camera_height_m",
            "horizon_blend_deg", "ground_albedo_rgb", "size", "include_sun"} <= set(m)
    with pytest.raises(F.FarFieldError):
        F.FarFieldSpec.from_meta({**m, "version": "farfield/v9"})
    with pytest.raises(F.FarFieldError):
        F.FarFieldSpec.from_meta({**m, "bogus": 1})
    bad = dict(m)
    del bad["version"]
    with pytest.raises(F.FarFieldError):
        F.FarFieldSpec.from_meta(bad)


# ---- 세계 이음매: Obs · RD-02 up_cam · Theta ----------------------------------------------------------------------------
def _world_case(R, roll, pitch, **theta_kw):
    obs = make_obs(R)
    up_cam = geom_cam.up_from_roll_pitch(roll, pitch)
    return obs, up_cam, C.Theta(**theta_kw)


def test_up_world_from_obs_uses_R_transpose():
    R = rot_y(25.0) @ rot_x(15.0)
    obs, up_cam, th = _world_case(R, 7.0, -12.0)
    np.testing.assert_allclose(F.up_world_from_obs(obs, up_cam), R.T @ up_cam, atol=1e-12)
    np.testing.assert_allclose(F.up_world_from_obs(obs, up_cam * 3.0), R.T @ up_cam, atol=1e-12)
    with pytest.raises(F.FarFieldError):
        F.up_world_from_obs(obs, np.zeros(3))


def test_spec_for_world_frame_from_camera_heading():
    R = rot_y(25.0) @ rot_x(15.0)
    obs, up_cam, th = _world_case(R, 7.0, -12.0)
    s = F.spec_for_world(obs, up_cam, th, size=32)
    up = R.T @ up_cam
    np.testing.assert_allclose(s.up, up, atol=1e-12)
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    north = np.asarray(s.north)
    assert abs(north @ np.asarray(s.up)) < 1e-12 and np.linalg.norm(north) == pytest.approx(1.0, abs=1e-12)
    assert north @ fwd > 0
    assert abs(np.linalg.det(np.stack([north, np.asarray(s.up), fwd]))) < 1e-9  # 광축이 (north, up) 평면 안
    # 수평 카메라: north == 광축
    obs0, up0, _ = _world_case(rot_y(40.0), 0.0, 0.0)
    s0 = F.spec_for_world(obs0, up0, C.Theta(), size=32)
    np.testing.assert_allclose(s0.north, rot_y(40.0).T @ np.array([0.0, 0.0, 1.0]), atol=1e-9)
    # 광축이 up 과 평행(위를 봄): 유한·정규직교로 폴백
    obsu, upu, _ = _world_case(np.eye(3), 0.0, 90.0)
    su = F.spec_for_world(obsu, upu, C.Theta(), size=32)
    assert np.isfinite(su.north).all() and abs(np.dot(su.north, su.up)) < 1e-9


def test_spec_for_world_sun_from_theta_or_defaults():
    obs, up_cam, _ = _world_case(rot_y(30.0), 0.0, 0.0)
    d = F.spec_for_world(obs, up_cam, C.Theta(), size=32)
    assert (d.sun_elev_deg, d.sun_azim_deg) == (F.DEFAULT_SUN_ELEV_DEG, F.DEFAULT_SUN_AZIM_DEG)
    s = F.spec_for_world(obs, up_cam, C.Theta(sun_elev_deg=30.0, sun_azim_deg=0.0), size=32)
    assert (s.sun_elev_deg, s.sun_azim_deg) == (30.0, 0.0)
    fwd = rot_y(30.0).T @ np.array([0.0, 0.0, 1.0])
    assert s.sun_dir() @ fwd > 0.8  # 방위 0 = 카메라 전방
    back = F.spec_for_world(obs, up_cam, C.Theta(sun_elev_deg=30.0, sun_azim_deg=180.0), size=32)
    assert back.sun_dir() @ fwd < -0.8
    right = F.spec_for_world(obs, up_cam, C.Theta(sun_elev_deg=0.0, sun_azim_deg=90.0), size=32)
    cam_right = rot_y(30.0).T @ np.array([1.0, 0.0, 0.0])
    assert right.sun_dir() @ cam_right > 0.99  # 방위 90 = 카메라 오른쪽(시계방향)
    o = F.spec_for_world(obs, up_cam, C.Theta(), size=32, turbidity=4.0, albedo=0.5)
    assert (o.turbidity, o.albedo, o.size) == (4.0, 0.5, 32)


def test_horizon_line_of_rd02_agrees_with_far_field_up():
    """RD-02 수평선 l=K^-T up_cam 의 '하늘 쪽(+)' 은 far-field 가 하늘로 그리는 방향(d·up_world>0)과 일치한다."""
    W, H = 96, 64
    for roll, pitch, R in ((7.0, -12.0, rot_y(25.0) @ rot_x(15.0)), (-9.0, 8.0, rot_z(30.0) @ rot_y(-70.0)), (0.0, 0.0, np.eye(3))):
        obs, up_cam, th = _world_case(R, roll, pitch)
        s = F.spec_for_world(obs, up_cam, th, size=128)
        l = geom_cam.horizon_line(obs.K, up_cam)
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        p = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
        dc = (np.linalg.inv(obs.K) @ p.T).T
        dw = (obs.R.T @ dc.T).T
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        side = p @ l  # 부호 있는 픽셀 거리
        sel = np.abs(side) > 2.0
        assert sel.sum() > 1000
        np.testing.assert_array_equal(np.sign(dw[sel] @ np.asarray(s.up)), np.sign(side[sel]))
        # far-field 로 렌더한 하늘 쪽 픽셀은 해석 하늘과 일치, 지면 쪽은 조명된 지면/헤이즈(하늘과 다름)
        ff = s.build()
        got = F.sample(ff, dw[sel])
        want = np.asarray(F.environment_radiance(s, dw[sel]))
        keep = np.degrees(np.arccos(np.clip(dw[sel] @ s.sun_dir(), -1, 1))) > 10.0
        elev = np.degrees(np.arcsin(np.clip(dw[sel] @ np.asarray(s.up), -1, 1)))
        keep &= np.abs(elev) > 3.0
        rel = np.abs(got - want)[keep] / want[keep].max(-1, keepdims=True)
        assert rel.max() < 0.04, rel.max()


# ---- STRATA 이음매 ------------------------------------------------------------------------------------------------------
def _strata(ff, meta=None):
    obs = make_obs(rot_y(5.0))
    st = ST.Strata(obs=obs, theta_in=C.Theta(focal_mm=35.0), level="L1", read_hash="cd" * 32, prov=make_prov(), seed=7,
                   meta=meta if meta is not None else {})
    st.add_layer(make_layer(obs.height, obs.width, 2.0, 1.0, seed=1))
    st.set_far_field(ff)
    return st


def test_far_field_flows_through_strata_and_world():
    s = spec(size=32)
    st = _strata(s.build(), {"far_field": s.to_meta()})
    st.validate()
    w = st.to_world()
    assert w.far_field is not None and w.far_field.rgb.tobytes() == s.build().rgb.tobytes()
    st2 = ST.Strata.from_bytes(st.to_bytes())
    assert st2.merkle_root() == st.merkle_root()
    assert st2.far_field.rgb.tobytes() == s.build().rgb.tobytes()
    assert F.FarFieldSpec.from_meta(st2.meta["far_field"]) == s
    other = _strata(spec(size=32, turbidity=3.0).build(), {"far_field": spec(size=32, turbidity=3.0).to_meta()})
    assert other.merkle_root() != st.merkle_root()  # far-field 값이 루트를 바꾼다
    sib = _strata(s.build(), {"far_field": s.to_meta()})
    assert sib.merkle_root() == st.merkle_root()


# ---- 격리 -------------------------------------------------------------------------------------------------------------
def test_import_is_torch_free_and_side_effect_free():
    code = ("import sys, random, numpy as np; s0 = np.random.get_state()[1].copy(); r0 = random.getstate();"
            "import buc.world.farfield as f;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "s = f.FarFieldSpec(up=(0,0,1), size=16); s.build();"
            "assert 'torch' not in sys.modules;"
            "assert (np.random.get_state()[1] == s0).all() and random.getstate() == r0")
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=55)
    assert p.returncode == 0, p.stderr[-800:]


def test_inputs_are_not_mutated():
    obs, up_cam, th = _world_case(rot_y(10.0), 3.0, -4.0, sun_elev_deg=20.0, sun_azim_deg=40.0)
    R0, K0, u0 = obs.R.copy(), obs.K.copy(), up_cam.copy()
    h0 = th.content_hash()
    F.spec_for_world(obs, up_cam, th, size=16).build()
    np.testing.assert_array_equal(obs.R, R0)
    np.testing.assert_array_equal(obs.K, K0)
    np.testing.assert_array_equal(up_cam, u0)
    assert th.content_hash() == h0
    d = rand_dirs(10, 1)
    d0 = d.copy()
    F.environment_radiance(spec(), d)
    F.sample(spec().build(), d)
    np.testing.assert_array_equal(d, d0)
