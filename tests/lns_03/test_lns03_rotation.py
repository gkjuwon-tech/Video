"""LNS-03 회전 블러(LH split의 정확한 호모그래피 적분): 오라클 대조, 임펄스 통계, 셔터 각 비례, 롤, 자동 샘플 수."""
import math
import warnings

import numpy as np
import pytest

from buc.lens import shutter
from lns03_helpers import FPS, FX, H, K, T3, W, blur_oracle, exposure, moments, pose_rot, rot_oracle, tex, traj, window, yaw_len

OMEGA = 4.8  # rad/s, 180°@24fps 에서 주점 근처 블러 약 12 px


def _run(img, depth=None, rate=(0.0, OMEGA, 0.0), shutter_deg=180.0, opts=None):
    c2w, t = traj(rate)
    return shutter.apply_shutter(img, depth, K=K, c2w=c2w, t=t, t_mid=T3[1], shutter_deg=shutter_deg, fps=FPS, opts=opts)


@pytest.mark.parametrize("rate", [(0.0, OMEGA, 0.0), (0.0, 0.0, 19.2), (2.0, -3.0, 6.0)])
@pytest.mark.parametrize("with_depth", [False, True])
def test_pure_rotation_matches_inverse_homography_oracle(rate, with_depth):
    img = tex()
    depth = np.full((H, W), 3.0, np.float32) if with_depth else None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _run(img, depth, rate=rate)
    assert isinstance(res, shutter.ShutterResult)
    assert res.rgb.shape == (H, W, 3) and res.rgb.dtype == np.float32
    assert res.blur_px.shape == (H, W, 2) and res.blur_px.dtype == np.float32
    T = exposure(180.0)
    t0, t1 = window(T3[1], T)
    fn = lambda tau: pose_rot(rate, tau)  # noqa: E731
    ref = rot_oracle(img, K, fn, t0, t1, n=256)
    bref = blur_oracle(K, fn(T3[1]), fn, t0, t1, depth=None if depth is None else depth.astype(np.float64))
    np.testing.assert_allclose(res.blur_px, bref, atol=1e-3)
    m = int(math.ceil(np.abs(bref).max())) + 3
    err = np.abs(res.rgb - ref)[m:H - m, m:W - m]
    assert err.mean() < 2.5e-4, err.mean()  # 1차 근사 2p - Hp 는 4e-4 ~ 4e-3 으로 실패한다
    assert err.max() < 2e-3, err.max()
    assert np.abs(res.rgb - img)[m:H - m, m:W - m].mean() > 0.01  # 실제로 블러가 생겼다
    # 볼록성
    assert (res.rgb >= img.min(axis=(0, 1)) - 1e-5).all() and (res.rgb <= img.max(axis=(0, 1)) + 1e-5).all()


@pytest.mark.parametrize("deg", [90.0, 180.0, 360.0])
def test_yaw_impulse_streak(deg):
    img = np.zeros((H, W, 3), np.float32)
    u0, v0 = 47, 31
    img[v0, u0] = 1.0
    res = _run(img, shutter_deg=deg)
    L = yaw_len(OMEGA, exposure(deg))
    for c in range(3):
        m, (cu, cv), (vu, vv) = moments(res.rgb[..., c].astype(np.float64))
        assert m == pytest.approx(1.0, abs=0.01)
        assert abs(cu - u0) < 0.25 and abs(cv - v0) < 0.25
        assert abs(vu - L * L / 12) < 0.15 * L * L / 12 + 0.1, (vu, L)
        assert vv < 0.3  # 수평 줄무늬
    assert (res.rgb >= -1e-6).all() and (res.rgb <= 1 + 1e-6).all()


def test_streak_variance_scales_with_shutter_angle():
    img = np.zeros((H, W, 3), np.float32)
    img[31, 47] = 1.0
    v = [moments(_run(img, shutter_deg=d).rgb[..., 0].astype(np.float64))[2][0] for d in (45.0, 90.0, 180.0, 360.0)]
    assert v[0] < v[1] < v[2] < v[3]
    assert v[3] / v[2] == pytest.approx(4.0, rel=0.15)
    assert v[2] / v[1] == pytest.approx(4.0, rel=0.2)


def test_open_shutter_phase_shifts_centroid():
    img = np.zeros((H, W, 3), np.float32)
    u0 = 47
    img[31, u0] = 1.0
    res = _run(img, opts=shutter.ShutterOpts(centered=False))
    L = yaw_len(OMEGA, exposure(180.0))
    _, (cu, _), (vu, _) = moments(res.rgb[..., 1].astype(np.float64))
    # 카메라가 +y(오른쪽)로 돌면 장면은 왼쪽으로 움직인다: 노출 [t, t+T] 의 궤적은 u0 -> u0 - L
    assert cu == pytest.approx(u0 - L / 2, abs=0.3)
    assert abs(vu - L * L / 12) < 0.15 * L * L / 12 + 0.1


def test_roll_tangential_blur():
    rate = (0.0, 0.0, 19.2)  # 180°@24 -> 0.4 rad
    res = _run(tex(), rate=rate)
    dth = 19.2 * exposure(180.0)
    uv = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).astype(np.float64)
    rel = uv - np.array([K[0, 2], K[1, 2]])
    r = np.linalg.norm(rel, axis=-1)
    b = res.blur_px.astype(np.float64)
    mag = np.linalg.norm(b, axis=-1)
    sel = r > 5
    np.testing.assert_allclose(mag[sel], 2 * r[sel] * math.sin(dth / 2), rtol=0.05)
    cosr = np.abs((b[sel] * rel[sel]).sum(-1)) / (mag[sel] * r[sel])
    assert cosr.max() < 0.05  # 접선 방향
    assert mag[r < 1].max() < 2 * 1.0 * math.sin(dth / 2) + 1e-3  # 주점 근처는 거의 0


def test_auto_and_explicit_samples():
    img = tex()
    res = _run(img)
    mx = float(np.linalg.norm(res.blur_px.astype(np.float64), axis=-1).max())
    exp = min(max(math.ceil(2 * mx) + 1, 1), 64)
    assert abs(res.samples - exp) <= 1
    assert _run(img, opts=shutter.ShutterOpts(samples=5)).samples == 5
    assert _run(img, opts=shutter.ShutterOpts(max_samples=4)).samples == 4
    # 샘플 1개(centered)는 중간 자세 = 원본
    one = _run(img, opts=shutter.ShutterOpts(samples=1))
    np.testing.assert_allclose(one.rgb, img, atol=1e-5)
    # 샘플이 많을수록 오라클에 수렴
    fn = lambda tau: pose_rot((0.0, OMEGA, 0.0), tau)  # noqa: E731
    ref = rot_oracle(img, K, fn, *window(T3[1], exposure(180.0)), n=256)
    e4 = np.abs(_run(img, opts=shutter.ShutterOpts(samples=4)).rgb - ref)[16:-16, 16:-16].mean()
    e32 = np.abs(_run(img, opts=shutter.ShutterOpts(samples=32)).rgb - ref)[16:-16, 16:-16].mean()
    assert e32 < e4 and e32 < 4e-3


def test_zero_border_darkens_edges_only():
    img = np.full((H, W, 3), 0.5, np.float32)
    e = _run(img).rgb
    z = _run(img, opts=shutter.ShutterOpts(border="zero")).rgb
    np.testing.assert_allclose(e, 0.5, atol=1e-5)  # edge: 상수 유지
    assert z[:, :2].mean() < 0.49 or z[:, -2:].mean() < 0.49
    np.testing.assert_allclose(z[6:H - 6, 20:W - 20], 0.5, atol=1e-5)
    assert FX > 0
