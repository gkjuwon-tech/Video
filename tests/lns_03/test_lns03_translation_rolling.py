"""LNS-03 평행이동 블러(깊이 의존, 달리 인 FOE)와 롤링 셔터."""
import math

import numpy as np
import pytest

from buc.core import geom
from buc.lens import shutter
from lns03_helpers import FPS, FX, H, K, T3, W, blur_oracle, exposure, moments, pose_rot, tex, traj, window

T = exposure(180.0)


def _run(img, depth, rate=(0.0, 0.0, 0.0), trate=(0.0, 0.0, 0.0), opts=None, shutter_deg=180.0):
    c2w, t = traj(rate, trate)
    return shutter.apply_shutter(img, depth, K=K, c2w=c2w, t=t, t_mid=T3[1], shutter_deg=shutter_deg, fps=FPS, opts=opts)


def test_truck_streak_length_depends_on_depth():
    Z = 4.0
    v = 10.0 * Z / (FX * T)  # Z 에서 10 px
    img = np.zeros((H, W, 3), np.float32)
    img[20, 30] = 1.0
    img[44, 66] = 1.0
    depth = np.full((H, W), Z, np.float32)
    depth[32:] = 2 * Z  # 아래 절반은 두 배 멀다
    res = _run(img, depth, trate=(v, 0.0, 0.0))
    top, bot = res.rgb[:32, :, 0].astype(np.float64), res.rgb[32:, :, 0].astype(np.float64)
    mt, (cut, _), (vt, vvt) = moments(top)
    mb, (cub, _), (vb, _) = moments(bot)
    assert mt == pytest.approx(1.0, abs=0.01) and mb == pytest.approx(1.0, abs=0.01)
    assert abs(cut - 30) < 0.25 and abs(cub - 66) < 0.25
    assert abs(vt - 100 / 12) < 0.15 * 100 / 12 + 0.1, vt
    assert abs(vb - 25 / 12) < 0.15 * 25 / 12 + 0.1, vb
    assert vvt < 0.3
    # blur_px: project 오라클과 일치, 길이 fx v T / Z
    fn = lambda tau: pose_rot((0, 0, 0), tau, (v, 0, 0))  # noqa: E731
    bref = blur_oracle(K, fn(T3[1]), fn, *window(T3[1], T), depth=depth.astype(np.float64))
    np.testing.assert_allclose(res.blur_px, bref, atol=1e-3)
    np.testing.assert_allclose(np.abs(res.blur_px[10, 40, 0]), 10.0, rtol=0.05)
    np.testing.assert_allclose(np.abs(res.blur_px[50, 40, 0]), 5.0, rtol=0.05)
    assert res.blur_px[10, 40, 0] < 0  # 카메라가 +x 로 가면 장면은 왼쪽(-u)으로


def test_translation_off_or_no_depth_is_copy():
    img = tex()
    depth = np.full((H, W), 4.0, np.float32)
    for d, o in ((depth, shutter.ShutterOpts(translation=False)), (None, None),
                 (np.full((H, W), np.inf, np.float32), None)):
        res = _run(img, d, trate=(30.0, 0.0, 0.0), opts=o)
        np.testing.assert_array_equal(res.rgb, img)
        assert np.abs(res.blur_px).max() <= 1e-6
        assert res.samples == 1


def test_dolly_in_radial_foe():
    Z = 5.0
    v = 0.25 * Z / T  # 노출 동안 Z 의 25 % 전진
    img = tex()
    depth = np.full((H, W), Z, np.float32)
    res = _run(img, depth, trate=(0.0, 0.0, v))
    fn = lambda tau: pose_rot((0, 0, 0), tau, (0, 0, v))  # noqa: E731
    bref = blur_oracle(K, fn(T3[1]), fn, *window(T3[1], T), depth=depth.astype(np.float64))
    np.testing.assert_allclose(res.blur_px, bref, atol=1e-3)
    uv = geom.pixel_grid(H, W)
    rel = uv - np.array([K[0, 2], K[1, 2]])
    r = np.linalg.norm(rel, axis=-1)
    b = res.blur_px.astype(np.float64)
    sel = r > 5
    mag = np.linalg.norm(b, axis=-1)
    cos = (b[sel] * rel[sel]).sum(-1) / (mag[sel] * r[sel])
    assert cos.min() > 0.99  # 바깥쪽 방사형(FOE = 주점)
    d = v * T
    np.testing.assert_allclose(mag[sel], r[sel] * Z * d / (Z * Z - d * d / 4), rtol=0.05)
    # 영상: 중심은 거의 그대로, 가장자리는 크게 블러
    c = np.abs(res.rgb - img)[29:35, 45:51].mean()
    e = np.abs(res.rgb - img)[4:12, 4:20].mean()
    assert c < 0.01 and e > 3 * c and e > 0.01


def test_rolling_shutter_skew_matches_projection():
    omega, readout = 4.8, 0.02
    img = np.zeros((H, W, 3), np.float32)
    u0 = 47
    img[:, u0] = 1.0
    c2w, t = traj((0.0, omega, 0.0))
    res = shutter.apply_shutter(img, None, K=K, c2w=c2w, t=t, t_mid=T3[1], shutter_deg=1.0, fps=FPS,
                                opts=shutter.ShutterOpts(readout_s=readout))
    off = readout * (np.arange(H) / (H - 1) - 0.5)
    mid = pose_rot((0, omega, 0), T3[1])
    shifts = []
    for y in range(4, H - 4):
        row = res.rgb[y, :, 0].astype(np.float64)
        cu = float((row * np.arange(W)).sum() / row.sum())
        Pt = pose_rot((0, omega, 0), T3[1] + off[y])
        d = np.array([u0, y, 1.0]) @ np.linalg.inv(K).T @ mid[:3, :3].T
        ref, _ = geom.project(K, Pt[:3, :3].T, np.zeros(3), 1e7 * d)
        assert cu == pytest.approx(ref[0], abs=0.15), (y, cu, ref[0])
        shifts.append(cu - u0)
    s = np.array(shifts)
    assert s[0] > 3.0 and s[-1] < -3.0 and (np.diff(s) < 0).all()  # 위 행은 오른쪽, 아래 행은 왼쪽
    # blur_px 는 행별 창으로 계산된다(짧은 셔터라 작다)
    fn = lambda tau: pose_rot((0, omega, 0), tau)  # noqa: E731
    bref = blur_oracle(K, mid, fn, *window(T3[1], exposure(1.0)), row_off=off)
    np.testing.assert_allclose(res.blur_px, bref, atol=1e-3)


def test_rolling_static_camera_is_copy():
    img = tex()
    c2w = np.repeat(pose_rot((0.1, 0.2, 0.3), 0.5, (1, 2, 3))[None], 3, 0)
    res = shutter.apply_shutter(img, np.full((H, W), 2.0, np.float32), K=K, c2w=c2w, t=T3, t_mid=T3[1],
                                shutter_deg=270.0, fps=FPS, opts=shutter.ShutterOpts(readout_s=0.03))
    np.testing.assert_array_equal(res.rgb, img)
    assert res.samples == 1 and np.abs(res.blur_px).max() <= 1e-6
    assert math.isfinite(W)
