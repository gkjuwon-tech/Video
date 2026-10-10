"""끝-끝: 렌더한 상자 방 영상 -> estimate_geom_camera. FOV·수평선·시간·왜곡 보정."""
import math
import time

import numpy as np
import pytest

import rd02_helpers as H
from buc.read import geom_cam as gc

W, HH = 640, 480
FOV, YAW, PITCH, ROLL = 60.0, 30.0, 8.0, -4.0
K_GT = H.K_from_fov(FOV, W, HH)
R_GT = H.camera_R(YAW, PITCH, ROLL)


@pytest.fixture(scope="module")
def clean():
    img = H.render_room(K_GT, R_GT, W, HH)
    t0 = time.perf_counter()
    g = gc.estimate_geom_camera(img)
    return img, g, time.perf_counter() - t0


@pytest.fixture(scope="module")
def distorted():
    img = H.render_room(K_GT, R_GT, W, HH, k1=-0.12)
    return gc.estimate_geom_camera(img)


def test_room_fov_and_horizon(clean):
    _, g, _ = clean
    assert g.W == W and g.H == HH
    assert g.n_manhattan_vps >= 2 and g.f_source in ("vp2", "vp3")
    assert abs(g.fov_x_deg - FOV) / FOV <= 0.04
    assert g.roll_deg == pytest.approx(ROLL, abs=0.75)
    assert g.pitch_deg == pytest.approx(PITCH, abs=0.75)
    assert "vp_inconsistent" not in g.warnings and g.orphan_frac <= 0.10
    assert len(g.segments) >= 30 and g.labels.shape == (len(g.segments),)
    # 왜곡 없는 영상에서 추정 왜곡은 작다
    assert H.max_displacement_px(K_GT, W, HH, g.k1, g.k2, 0.0, 0.0) <= 2.0


def test_room_time_budget(clean):
    _, _, dt = clean
    assert dt <= 15.0


def test_rgb_and_uint8_inputs_agree(clean):
    img, g, _ = clean
    small = img[::2, ::2]
    rgb = np.repeat(small[..., None], 3, axis=-1).astype(np.float32)
    u8 = np.clip(np.round(small * 255), 0, 255).astype(np.uint8)
    a = gc.estimate_geom_camera(rgb)
    b = gc.estimate_geom_camera(u8)
    for x in (a, b):
        assert x.W == small.shape[1] and x.H == small.shape[0]
        assert abs(x.fov_x_deg - FOV) / FOV <= 0.06
        assert x.roll_deg == pytest.approx(ROLL, abs=1.0) and x.pitch_deg == pytest.approx(PITCH, abs=1.0)


def test_distortion_is_reduced(distorted):
    g = distorted
    base = H.max_displacement_px(K_GT, W, HH, 0.0, 0.0, -0.12, 0.0)
    err = H.max_displacement_px(K_GT, W, HH, g.k1, g.k2, -0.12, 0.0)
    assert g.k1 < 0
    assert err <= 0.4 * base
    assert abs(g.fov_x_deg - FOV) / FOV <= 0.05
    assert g.roll_deg == pytest.approx(ROLL, abs=1.0) and g.pitch_deg == pytest.approx(PITCH, abs=1.0)
    assert math.isfinite(g.plumb_rms_px)


def test_featureless_image_falls_back_to_prior():
    rng = np.random.default_rng(1)
    img = 0.5 + 0.005 * rng.standard_normal((240, 320))
    g = gc.estimate_geom_camera(img, fov_prior_deg=48.0, horizon_prior=(2.0, -3.0))
    assert g.f_source == "prior" and g.fov_x_deg == pytest.approx(48.0, rel=1e-6)
    assert g.horizon_source == "prior"
    assert g.roll_deg == pytest.approx(2.0, abs=1e-9) and g.pitch_deg == pytest.approx(-3.0, abs=1e-9)
    assert not g.reliable and g.k1 == 0.0 and g.k2 == 0.0
