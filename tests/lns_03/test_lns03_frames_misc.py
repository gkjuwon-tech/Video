"""LNS-03 Frames 이음매(180° 기본, θ 노출), 입력 검증, 결정론·입력 불변, 속도."""
import time

import numpy as np
import pytest

from buc.core import color
from buc.core import contracts as C
from buc.lens import shutter
from lns03_helpers import FPS, H, K, T3, W, pose_rot, tex, traj


def _shot(n=3, rate=(0.0, 4.8, 0.0), trate=(0.5, 0.0, 0.0), theta=None):
    t = np.arange(n) / FPS
    c2w = np.stack([pose_rot(rate, ti, trate) for ti in t])
    th = theta or C.Theta(shutter_deg=180.0, fps=FPS)
    return C.Shot(c2w=c2w, t=t, K=np.repeat(K[None], n, 0), focus_m=np.full(n, 3.0), width=W, height=H, fps=FPS,
                  theta=th)


def _frames(n=3):
    lin = np.stack([tex(seed=i) for i in range(n)])
    depth = np.full((n, H, W), 3.0, np.float32)
    depth[:, :, W // 2:] = 6.0
    return C.Frames(rgb=color.linear_to_display(lin), fps=FPS,
                    aov={"depth": depth, "alpha": np.ones((n, H, W), np.float32),
                         "layer_id": np.zeros((n, H, W), np.int16)}, shot_hash="s" * 8, world_hash="w" * 8)


@pytest.mark.parametrize("deg", [180.0, 90.0])
def test_frames_seam(deg):
    shot = _shot(theta=C.Theta(shutter_deg=deg, fps=FPS))
    fr = _frames()
    rgb0 = fr.rgb.copy()
    out = shutter.apply_shutter_frames(fr, shot)
    assert isinstance(out, C.Frames)
    out.validate()
    assert out.rgb.shape == fr.rgb.shape and out.rgb.dtype == np.float32
    assert out.fps == fr.fps and out.shot_hash == fr.shot_hash and out.world_hash == fr.world_hash
    assert set(out.aov) >= {"depth", "alpha", "layer_id", "blur_px"}
    for k in ("depth", "alpha", "layer_id"):
        np.testing.assert_array_equal(out.aov[k], fr.aov[k])
    assert out.aov["blur_px"].shape == (3, H, W, 2) and out.aov["blur_px"].dtype == np.float32
    for i in range(3):  # 프레임 0 은 창이 t<0 으로 나가므로 외삽이 필요하다
        ref = shutter.apply_shutter(color.srgb_to_linear(fr.rgb[i]), fr.aov["depth"][i], K=shot.K[i], c2w=shot.c2w,
                                    t=shot.t, t_mid=shot.t[i], shutter_deg=deg, fps=FPS)
        np.testing.assert_allclose(out.rgb[i], color.linear_to_display(ref.rgb), atol=1e-6)
        np.testing.assert_allclose(out.aov["blur_px"][i], ref.blur_px, atol=1e-6)
    np.testing.assert_array_equal(fr.rgb, rgb0)
    L = np.abs(out.aov["blur_px"][1, H // 2, W // 2, 0])
    assert L == pytest.approx(2 * 120.0 * np.tan(4.8 * deg / (360 * FPS) / 2) + 120 * 0.5 * deg / (360 * FPS) / 6.0,
                              rel=0.03)


def test_frames_without_depth_and_static():
    shot = _shot(rate=(0, 0, 0), trate=(0, 0, 0))
    fr = _frames()
    out = shutter.apply_shutter_frames(C.Frames(rgb=fr.rgb, fps=FPS), shot)
    np.testing.assert_allclose(out.rgb, fr.rgb, atol=2e-6)
    assert np.abs(out.aov["blur_px"]).max() <= 1e-6


def test_frames_shape_mismatch():
    fr = _frames(2)
    with pytest.raises(ValueError):
        shutter.apply_shutter_frames(fr, _shot(3))
    with pytest.raises(ValueError):
        shutter.apply_shutter_frames(C.Frames(rgb=np.zeros((3, H, W + 1, 3), np.float32), fps=FPS), _shot(3))


def test_input_validation():
    img = tex()
    c2w, t = traj((0, 1, 0))
    kw = dict(K=K, c2w=c2w, t=t, t_mid=T3[1], shutter_deg=180.0, fps=FPS)
    with pytest.raises(TypeError):
        shutter.apply_shutter(img, None, K, c2w, t, T3[1], 180.0, FPS)
    bad_imgs = [img[..., :2], img[0], np.where(np.arange(W)[None, :, None] == 3, np.nan, img).astype(np.float32)]
    for b in bad_imgs:
        with pytest.raises(ValueError):
            shutter.apply_shutter(b, None, **kw)
    for d in (np.ones((H, W + 1), np.float32), np.zeros((H, W), np.float32), np.full((H, W), np.nan, np.float32),
              -np.ones((H, W), np.float32)):
        with pytest.raises(ValueError):
            shutter.apply_shutter(img, d, **kw)
    for over in (dict(shutter_deg=0.0), dict(shutter_deg=400.0), dict(fps=0.0), dict(t=np.array([0.0, 0.1, 0.05])),
                 dict(t=np.array([0.0, 0.1])), dict(c2w=c2w[:, :3]), dict(K=np.eye(3) * 0.0),
                 dict(t_mid=float("nan"))):
        with pytest.raises(ValueError):
            shutter.apply_shutter(img, None, **{**kw, **over})


def test_determinism_and_input_immutability():
    img = tex()
    depth = np.full((H, W), 3.0, np.float32)
    depth[:, :20] = 1.5
    img0, d0 = img.copy(), depth.copy()
    c2w, t = traj((0.5, 3.0, -1.0), (1.0, 0.2, 2.0))
    c0 = c2w.copy()
    kw = dict(K=K, c2w=c2w, t=t, t_mid=T3[1], shutter_deg=270.0, fps=FPS, opts=shutter.ShutterOpts(readout_s=0.01))
    a = shutter.apply_shutter(img, depth, **kw)
    b = shutter.apply_shutter(img, depth, **kw)
    assert a.rgb.tobytes() == b.rgb.tobytes() and a.blur_px.tobytes() == b.blur_px.tobytes() and a.samples == b.samples
    np.testing.assert_array_equal(img, img0)
    np.testing.assert_array_equal(depth, d0)
    np.testing.assert_array_equal(c2w, c0)
    assert np.isfinite(a.rgb).all() and np.isfinite(a.blur_px).all()
    assert (a.rgb >= img.min(axis=(0, 1)) - 1e-5).all() and (a.rgb <= img.max(axis=(0, 1)) + 1e-5).all()


def test_speed_320x180():
    w, h = 320, 180
    Kb = np.array([[300.0, 0, (w - 1) / 2], [0, 300.0, (h - 1) / 2], [0, 0, 1]])
    img = tex(w, h)
    depth = np.full((h, w), 4.0, np.float32)
    omega = 30.0 / (300.0 / 48)  # 약 30 px
    c2w, t = traj((0.0, omega, 0.0), (1.0, 0.0, 0.0))
    t0 = time.perf_counter()
    res = shutter.apply_shutter(img, depth, K=Kb, c2w=c2w, t=t, t_mid=T3[1], shutter_deg=180.0, fps=FPS)
    dt = time.perf_counter() - t0
    assert np.linalg.norm(res.blur_px, axis=-1).max() > 25
    assert dt < 4.0, dt
