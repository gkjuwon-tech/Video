"""수치 계약(Theta·Obs·Shot)과 불변식 위반 거부."""
import math

import numpy as np
import pytest

from buc.core import contracts as C
from fnd02_fixtures import (H, W, make_frames, make_image, make_kpi, make_layers, make_obs, make_prov, make_read,
                            make_shot, make_world, rot_y)


# ---------- Theta ----------
def test_theta_defaults_and_intrinsics():
    th = C.Theta()
    assert th.aspect == pytest.approx(16 / 9) and th.focal_mm == 35 and th.fps == 24 and th.lut == "none"
    assert th.sun_elev_deg is None and th.extra == {}
    K = th.intrinsics(1920, 1080)
    assert K.shape == (3, 3) and K.dtype == np.float64
    fx = 35 / 24.89 * 1920
    np.testing.assert_allclose(K, [[fx, 0, 959.5], [0, fx, 539.5], [0, 0, 1]], rtol=0, atol=1e-9)
    assert th.fov_x_deg() == pytest.approx(math.degrees(2 * math.atan(24.89 / 70)), abs=1e-9)
    assert C.Theta(focal_mm=12.0).fov_x_deg() > C.Theta(focal_mm=85.0).fov_x_deg()


@pytest.mark.parametrize("kw", [dict(focal_mm=1.0), dict(focal_mm=2001.0), dict(sensor_width_mm=0.5),
                                dict(f_stop=0.5), dict(f_stop=65.0), dict(focus_m=0.0), dict(shutter_deg=0.0),
                                dict(shutter_deg=361.0), dict(fps=0.0), dict(fps=241.0), dict(nd_stops=-1.0),
                                dict(ev=10.5), dict(wb_kelvin=1000.0), dict(wb_tint=1.5), dict(grain=1.2),
                                dict(vignette=-0.1), dict(atmos_beta=-0.01), dict(sun_elev_deg=91.0),
                                dict(sun_azim_deg=360.0), dict(focal_mm=float("nan"))])
def test_theta_rejects_out_of_range(kw):
    with pytest.raises(C.ContractError):
        C.Theta(**kw)


def test_theta_accepts_edges():
    C.Theta(f_stop=0.7, shutter_deg=360.0, fps=240.0, focus_m=float("inf"), wb_tint=-1.0, sun_elev_deg=-90.0,
            sun_azim_deg=0.0, nd_stops=15.0, ev=-10.0)


# ---------- Obs ----------
def test_obs_projection_numeric():
    th = C.Theta()
    K = th.intrinsics(W, H)
    R = rot_y(10.0)
    t = np.array([0.1, -0.2, 0.3])
    k1, k2 = 0.05, -0.01
    obs = make_obs(K=K, R=R, t=t, k1=k1, k2=k2)
    rng = np.random.default_rng(3)
    Xw = rng.uniform(-1, 1, (20, 3)) + np.array([0, 0, 4.0])
    uv, z = obs.project(Xw)
    assert uv.shape == (20, 2) and z.shape == (20,)
    Xc = Xw @ R.T + t
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2
    u = K[0, 0] * x * f + K[0, 2]
    v = K[1, 1] * y * f + K[1, 2]
    np.testing.assert_allclose(uv, np.stack([u, v], 1), rtol=0, atol=1e-9)
    np.testing.assert_allclose(z, Xc[:, 2], rtol=0, atol=1e-12)


def test_obs_pose_algebra():
    obs = make_obs()
    w2c, c2w = obs.w2c(), obs.c2w()
    assert w2c.shape == c2w.shape == (4, 4)
    np.testing.assert_allclose(w2c @ c2w, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(w2c[3], [0, 0, 0, 1], atol=0)
    np.testing.assert_allclose(obs.center(), -obs.R.T @ obs.t, atol=1e-12)
    np.testing.assert_allclose(c2w[:3, 3], obs.center(), atol=1e-12)
    # 광학 중심은 카메라 좌표 원점으로 간다; 주점을 지나는 점은 영상 중심에 찍힌다.
    p = obs.center() + 2.0 * obs.R.T @ np.array([0, 0, 1.0])
    uv, z = obs.project(p[None])
    np.testing.assert_allclose(uv[0], [(W - 1) / 2, (H - 1) / 2], atol=1e-9)
    assert z[0] == pytest.approx(2.0, abs=1e-12)


def test_obs_coerces_dtype_and_contiguity():
    obs = make_obs(K=C.Theta().intrinsics(W, H).astype(np.float32), t=[0.1, -0.2, 0.3])
    assert obs.K.dtype == np.float64 and obs.t.dtype == np.float64 and obs.t.shape == (3,)
    assert obs.R.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("bad", [
    dict(R=np.diag([1.0, 1.0, -1.0])),                      # det -1
    dict(R=np.eye(3) * 1.01),                               # 비직교
    dict(K=np.array([[10.0, 0, 5], [0, 10, 4], [0, 0, 2]])),  # 마지막 행
    dict(K=np.array([[-10.0, 0, 5], [0, 10, 4], [0, 0, 1]])),  # fx<0
    dict(t=np.zeros(4)),
    dict(mask=np.ones((H + 1, W), dtype=bool)),
    dict(l_obs_hard=2.0, l_obs_part=1.0),
    dict(l_obs_hard=-0.1),
])
def test_obs_rejects(bad):
    with pytest.raises(C.ContractError):
        make_obs(**bad)


# ---------- Prov ----------
def test_prov_rules():
    p = make_prov("USER", "EXTGEN:soul", roots=("b", "a", "b"))
    assert p.roots == ("a", "b") and p.ext and not p.comp and not p.hold
    assert make_prov("COMP").comp and make_prov("HOLDOUT:eval").hold
    with pytest.raises(C.ContractError):
        make_prov("BOGUS")
    with pytest.raises(C.ContractError):
        make_prov("bogus:USER")
    m = C.Prov.merge(make_prov("USER", ns="user_vault", mode="user", roots=("r1",)),
                     make_prov("ENGINE", roots=("r0",)))
    assert m.taint == frozenset({"USER", "ENGINE"}) and m.roots == ("r0", "r1")
    assert m.mode == "mixed" and m.ns == "user_vault"
    m2 = C.Prov.merge(make_prov("ENGINE", ns="a"), make_prov("SI", ns="b"))
    assert m2.mode == "engine" and m2.ns == "a"


# ---------- InputImage / ReadPacket ----------
def test_input_image_hash_and_rejects():
    img = make_image()
    assert img.h_I == C.content_hash(img.pixels)
    with pytest.raises(C.ContractError):
        make_image(color_space="cmyk")
    with pytest.raises(C.ContractError):
        make_image(pixels=np.zeros((H, W, 4), dtype=np.uint8))
    with pytest.raises(C.ContractError):
        make_image(pixels=np.zeros((H, W, 3), dtype=np.int32))


@pytest.mark.parametrize("bad", [
    lambda: dict(depth=np.full((H, W), np.nan, dtype=np.float32)),
    lambda: dict(depth=np.zeros((H, W), dtype=np.float32)),
    lambda: dict(depth=np.ones((H, W - 1), dtype=np.float32)),
    lambda: dict(image_lin=-np.ones((H, W, 3), dtype=np.float32)),
    lambda: dict(image_lin=np.full((H, W, 3), np.inf, dtype=np.float32)),
    lambda: dict(scale_s=0.0),
    lambda: dict(level="L3"),
    lambda: dict(depth_conf=np.full((H, W), 1.5, dtype=np.float32)),
    lambda: dict(normals=np.full((H, W, 3), 1.0, dtype=np.float32)),
    lambda: dict(dyn={"wind": 1}),
])
def test_read_packet_rejects(bad):
    with pytest.raises(C.ContractError):
        make_read(**bad())


def test_read_packet_accepts_optional_fields_and_coerces():
    n = np.zeros((H, W, 3), dtype=np.float64)
    n[..., 2] = 1.0
    r = make_read(depth=np.ones((H, W), dtype=np.float64), normals=n,
                  depth_conf=np.full((H, W), 0.5, dtype=np.float32), dyn={"blur_vel": 0.0})
    assert r.depth.dtype == np.float32 and r.level in C.LEVELS


# ---------- World ----------
def test_world_layer_rules():
    w = make_world()
    assert w.schema == "STRATA-lite-min/v0" and len(w.layers) == 2
    with pytest.raises(C.ContractError):
        make_world(layers=())
    with pytest.raises(C.ContractError):
        make_world(layers=tuple(reversed(make_layers())))  # 뒤→앞 정렬 위반
    with pytest.raises(C.ContractError):
        C.LDILayer(rgba=np.full((H, W, 4), 1.2, dtype=np.float32), depth=np.ones((H, W), dtype=np.float32))
    with pytest.raises(C.ContractError):
        make_world(layers=(C.LDILayer(rgba=np.ones((H, W + 1, 4), dtype=np.float32),
                                      depth=np.ones((H, W + 1), dtype=np.float32)),))
    # 투명 픽셀에서는 순서를 따지지 않는다.
    a, b = make_layers()
    rgba = a.rgba.copy()
    rgba[..., 3] = 0.0
    far = C.LDILayer(rgba=rgba, depth=np.full((H, W), 9.0, dtype=np.float32))
    make_world(layers=(far, b))
    with pytest.raises(C.ContractError):
        C.FarField(rgb=np.full((4, 4, 3), np.nan, dtype=np.float32))


# ---------- Shot / Frames / KPI ----------
def test_shot_timing_and_first_pose():
    obs = make_obs()
    s = make_shot(obs, n=48, fps=24.0)
    assert s.n_frames == 48 and s.duration_s == pytest.approx(2.0, abs=1e-12)
    assert s.first_pose_matches(obs)
    c2w = s.c2w.copy()
    c2w[0, 0, 3] += 2e-3
    assert not make_shot(obs, n=48, c2w=c2w).first_pose_matches(obs)
    c2w = s.c2w.copy()
    c2w[0, :3, :3] = c2w[0, :3, :3] @ rot_y(0.2)
    assert not make_shot(obs, n=48, c2w=c2w).first_pose_matches(obs)
    assert make_shot(obs, n=48, c2w=c2w).first_pose_matches(obs, rot_tol_deg=0.3)


@pytest.mark.parametrize("bad", [
    lambda s: dict(t=s.t + 0.01),                     # t[0] != 0
    lambda s: dict(t=s.t * 1.001),                    # 간격 != 1/fps
    lambda s: dict(focus_m=np.zeros(s.n_frames)),
    lambda s: dict(K=s.K[:-1]),                       # N 불일치
    lambda s: dict(c2w=np.concatenate([s.c2w[:, :3], np.ones((s.n_frames, 1, 4))], 1)),
    lambda s: dict(c2w=s.c2w * np.array([1, 1, -1, 1.0])[None, :, None]),  # det -1
])
def test_shot_rejects(bad):
    s = make_shot()
    with pytest.raises(C.ContractError):
        make_shot(**bad(s))


def test_frames_and_kpi_rules():
    s = make_shot()
    with pytest.raises(C.ContractError):
        make_frames(s, rgb=np.full((s.n_frames, H, W, 3), 1.5, dtype=np.float32))
    with pytest.raises(C.ContractError):
        make_frames(s, aov={"depth": np.ones((s.n_frames + 1, H, W), dtype=np.float32)})
    assert make_kpi(ffd=None).ffd is None
    for kw in [dict(ir_geo=1.1), dict(cheat_share=-0.1), dict(m_eff=0.0), dict(ffd=-1.0), dict(de00=float("nan"))]:
        with pytest.raises(C.ContractError):
            make_kpi(**kw)


def test_keyword_only_construction():
    with pytest.raises(TypeError):
        C.KPI(0.1)
    with pytest.raises(TypeError):
        C.Theta(1.5)
