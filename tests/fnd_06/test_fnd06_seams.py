"""FND-06 이음매: FND-02 계약(Theta·Obs·Frames·InputImage)과 같은 규약을 쓰는지, 부작용 없는지, 결정론인지."""
import os
import subprocess
import sys

import numpy as np
import pytest

from buc.core import color, contracts, geom, pyramid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(99)


def test_intrinsics_equal_theta_intrinsics_and_fov():
    for kw in [dict(), dict(focal_mm=24.0), dict(focal_mm=85.0, sensor_width_mm=36.0), dict(focal_mm=12.0, sensor_width_mm=9.6)]:
        th = contracts.Theta(**kw)
        for W, H in [(1920, 1080), (101, 51), (640, 640)]:
            np.testing.assert_allclose(geom.intrinsics_from_focal(th.focal_mm, th.sensor_width_mm, W, H), th.intrinsics(W, H), rtol=0, atol=1e-9)
        assert geom.fov_from_focal(th.focal_mm, th.sensor_width_mm) == pytest.approx(th.fov_x_deg(), abs=1e-9)


def test_project_and_pose_agree_with_obs():
    W, H = 320, 240
    c2w = geom.look_at(np.array([1.0, -2.0, 0.5]), np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    w2c = geom.invert_pose(c2w)
    K = geom.intrinsics_from_fov(55.0, W, H)
    for k1, k2 in [(0.0, 0.0), (-0.1, 0.02)]:
        obs = contracts.Obs(K=K, R=w2c[:3, :3], t=w2c[:3, 3], width=W, height=H, k1=k1, k2=k2)    # 계약 검증을 통과해야 한다
        np.testing.assert_allclose(obs.c2w(), c2w, atol=1e-12)
        np.testing.assert_allclose(obs.center(), c2w[:3, 3], atol=1e-12)
        X = RNG.normal(size=(64, 3)) * 0.7
        uv_c, z_c = obs.project(X)
        uv_g, z_g = geom.project(obs.K, obs.R, obs.t, X, obs.k1, obs.k2)
        np.testing.assert_allclose(uv_g, uv_c, atol=1e-9)
        np.testing.assert_allclose(z_g, z_c, atol=1e-12)
        vis = geom.visible_mask(uv_g, z_g, W, H)
        assert 0 < vis.sum() <= 64


def test_unproject_world_inverts_obs_project_for_render_depth():
    """RDR 의 AOV 깊이(광축 z-깊이)에서 세계점을 복원해 다시 투영하면 같은 픽셀로 돌아온다."""
    W, H = 64, 48
    K = geom.intrinsics_from_fov(60.0, W, H)
    R = geom.rodrigues(np.array([0.1, -0.2, 0.05]))
    t = np.array([0.0, 0.1, 0.2])
    obs = contracts.Obs(K=K, R=R, t=t, width=W, height=H)
    depth = RNG.uniform(1.0, 6.0, (H, W))
    Xw = geom.unproject_world(K, R, t, geom.pixel_grid(H, W), depth)
    uv, z = obs.project(Xw.reshape(-1, 3))
    np.testing.assert_allclose(uv.reshape(H, W, 2), geom.pixel_grid(H, W), atol=1e-9)
    np.testing.assert_allclose(z.reshape(H, W), depth, atol=1e-9)


def test_decode_accepts_every_inputimage_color_space_and_feeds_readpacket_domain():
    for cs in contracts.COLOR_SPACES:
        for dt, hi in [(np.uint8, 255), (np.uint16, 65535)]:
            px = RNG.integers(0, hi + 1, size=(6, 8, 3)).astype(dt)
            img = contracts.InputImage(pixels=px, color_space=cs, prov=contracts.Prov())
            lin = color.decode_to_linear(img.pixels, img.color_space)
            assert lin.dtype == np.float32 and lin.shape == (6, 8, 3) and np.isfinite(lin).all()
        fpx = RNG.uniform(0, 1, (6, 8, 3)).astype(np.float32)
        assert color.decode_to_linear(fpx, cs).shape == (6, 8, 3)
    # srgb 입력의 선형화 결과는 ReadPacket.image_lin 의 정의역(>=0, 유한)에 들어간다
    px = RNG.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    assert color.decode_to_linear(px, "srgb").min() >= 0.0


def test_frames_display_values_pass_contract():
    lin = RNG.uniform(0, 3.0, (2, 6, 8, 3)).astype(np.float32)           # HDR 선형
    rgb = color.linear_to_display(lin)
    fr = contracts.Frames(rgb=rgb, fps=24)                               # [0,1] 표시값 계약
    assert fr.rgb.shape == (2, 6, 8, 3)


def test_modules_are_torch_free_and_side_effect_free():
    code = ("import sys, numpy as np; s0 = np.random.get_state()[1].copy();"
            "import buc.core.geom, buc.core.color, buc.core.pyramid;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "assert (np.random.get_state()[1] == s0).all()")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert p.returncode == 0, p.stderr[-800:]


def test_bitwise_deterministic_and_inputs_untouched():
    img = RNG.uniform(0, 1, (33, 41, 3)).astype(np.float32)
    keep = img.copy()
    uv = RNG.uniform(-1, 42, (500, 2))
    for fn in [lambda: pyramid.gaussian_blur(img, 1.3), lambda: pyramid.pyr_down(img), lambda: pyramid.pyr_up(img),
               lambda: pyramid.resize(img, (50, 21), "lanczos"), lambda: pyramid.resize(img, (50, 21), "area"),
               lambda: pyramid.sample_bilinear(img, uv), lambda: color.srgb_to_linear(img), lambda: color.linear_to_display(img),
               lambda: color.rgb_lin_to_lab(img), lambda: pyramid.laplacian_pyramid(img, 2)[1]]:
        a, b = fn(), fn()
        assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(img, keep)
    assert img.flags.writeable
