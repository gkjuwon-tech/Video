"""LNS-03 공개 API, 옵션 검증, 노출 창·샘플 시각·롤링 행 오프셋, 궤적 보간(pose_at), 회전 호모그래피."""
import dataclasses
import inspect
import math

import numpy as np
import pytest

from buc.core import geom
from buc.lens import shutter
from lns03_helpers import FPS, K, W, homog, pose_rot, traj


def test_public_names_and_kwonly():
    for n in ("ShutterOpts", "ShutterResult", "exposure_time", "exposure_window", "sample_times",
              "row_time_offsets", "pose_at", "rotation_homography", "apply_shutter", "apply_shutter_frames"):
        assert hasattr(shutter, n), n
    p = inspect.signature(shutter.apply_shutter).parameters
    for n in ("K", "c2w", "t", "t_mid", "shutter_deg", "fps", "opts"):
        assert p[n].kind is inspect.Parameter.KEYWORD_ONLY, n
    assert "centered" in inspect.signature(shutter.exposure_window).parameters
    assert inspect.signature(shutter.exposure_window).parameters["centered"].kind is inspect.Parameter.KEYWORD_ONLY


def test_opts_defaults_and_validation():
    o = shutter.ShutterOpts()
    assert (o.samples, o.max_samples, o.readout_s, o.centered, o.translation, o.border) == (0, 64, 0.0, True, True, "edge")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.samples = 3
    for bad in (dict(samples=-1), dict(samples=2.5), dict(samples=True), dict(max_samples=0), dict(max_samples=3.0),
                dict(readout_s=-0.01), dict(readout_s=float("nan")), dict(readout_s=float("inf")),
                dict(border="wrap")):
        with pytest.raises(ValueError):
            shutter.ShutterOpts(**bad)
    shutter.ShutterOpts(samples=8, max_samples=8, readout_s=0.03, centered=False, translation=False, border="zero")


def test_exposure_time_and_window():
    assert shutter.exposure_time(180.0, 24.0) == pytest.approx(1 / 48, rel=1e-12)
    assert shutter.exposure_time(360.0, 25.0) == pytest.approx(1 / 25, rel=1e-12)
    assert shutter.exposure_time(172.8, 24.0) == pytest.approx(172.8 / (360 * 24), rel=1e-12)
    for d, f in ((0.0, 24.0), (-10.0, 24.0), (361.0, 24.0), (180.0, 0.0), (180.0, -24.0), (float("nan"), 24.0),
                 (180.0, float("inf"))):
        with pytest.raises(ValueError):
            shutter.exposure_time(d, f)
    a, b = shutter.exposure_window(1.0, 24.0, 180.0)
    assert (a, b) == pytest.approx((1.0 - 1 / 96, 1.0 + 1 / 96), abs=1e-12)
    a, b = shutter.exposure_window(1.0, 24.0, 180.0, centered=False)
    assert (a, b) == pytest.approx((1.0, 1.0 + 1 / 48), abs=1e-12)


def test_sample_times_and_row_offsets():
    s = shutter.sample_times(0.0, 1.0, 4)
    assert s.dtype == np.float64
    np.testing.assert_allclose(s, [0.125, 0.375, 0.625, 0.875], atol=1e-15)
    np.testing.assert_allclose(shutter.sample_times(2.0, 2.0, 3), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(shutter.sample_times(-1.0, 1.0, 1), [0.0], atol=1e-15)
    for bad in (0, -2, 2.5):
        with pytest.raises(ValueError):
            shutter.sample_times(0.0, 1.0, bad)
    r = shutter.row_time_offsets(5, 0.04)
    assert r.dtype == np.float64 and r.shape == (5,)
    np.testing.assert_allclose(r, [-0.02, -0.01, 0.0, 0.01, 0.02], atol=1e-15)
    np.testing.assert_array_equal(shutter.row_time_offsets(1, 0.04), [0.0])
    np.testing.assert_array_equal(shutter.row_time_offsets(7, 0.0), np.zeros(7))
    for h, ro in ((0, 0.01), (4, -0.01), (4, float("nan"))):
        with pytest.raises(ValueError):
            shutter.row_time_offsets(h, ro)


def test_pose_at_knots_interp_extrap():
    rate, trate = (0.3, 4.8, -0.7), (1.0, -0.5, 2.0)
    c2w, t = traj(rate, trate)
    for j in range(3):
        np.testing.assert_allclose(shutter.pose_at(c2w, t, t[j]), c2w[j], atol=1e-12)
    # 등속 궤적: 구간 안·밖(외삽) 모두 해석해와 일치
    taus = np.array([-0.02, 0.005, 1 / 48, 0.06, 2 / 24 + 0.015])
    P = shutter.pose_at(c2w, t, taus)
    assert P.shape == (5, 4, 4)
    for k, tau in enumerate(taus):
        np.testing.assert_allclose(P[k], pose_rot(rate, tau, trate), atol=1e-9)
        np.testing.assert_allclose(shutter.pose_at(c2w, t, float(tau)), P[k], atol=1e-12)
    # 일반 구간: geom.interp_pose 와 동일
    rng = np.random.default_rng(3)
    A = geom.make_pose(geom.rodrigues(rng.normal(size=3) * 0.4), rng.normal(size=3))
    B = geom.make_pose(geom.rodrigues(rng.normal(size=3) * 0.4), rng.normal(size=3))
    tt = np.array([0.0, 0.5])
    np.testing.assert_allclose(shutter.pose_at(np.stack([A, B]), tt, 0.2), geom.interp_pose(A, B, 0.4), atol=1e-12)
    # N=1 은 상수
    np.testing.assert_allclose(shutter.pose_at(A[None], np.array([0.0]), 3.7), A, atol=0)
    with pytest.raises(ValueError):
        shutter.pose_at(c2w, np.array([0.0, 0.1, 0.1]), 0.05)
    with pytest.raises(ValueError):
        shutter.pose_at(c2w, np.array([0.0, 0.1]), 0.05)


def test_rotation_homography():
    A = pose_rot((0.0, 4.8, 0.0), 1 / 24)
    np.testing.assert_allclose(shutter.rotation_homography(K, A, A), np.eye(3), atol=1e-12)
    rng = np.random.default_rng(5)
    for _ in range(3):
        M = geom.make_pose(geom.rodrigues(rng.normal(size=3) * 0.3), rng.normal(size=3))
        Tq = geom.make_pose(geom.rodrigues(rng.normal(size=3) * 0.3), rng.normal(size=3))
        Hm = shutter.rotation_homography(K, M, Tq)
        np.testing.assert_allclose(Hm, homog(K, M, Tq), atol=1e-10)
        # 무한원점 투영 일치(관례 확인): 중간 자세 화소 -> tau 자세 화소
        uv = np.array([[10.0, 5.0], [50.0, 40.0], [80.0, 12.0]])
        d = np.concatenate([uv, np.ones((3, 1))], 1) @ np.linalg.inv(K).T @ M[:3, :3].T
        X = M[:3, 3] + 1e7 * d
        ref, z = geom.project(K, Tq[:3, :3].T, -Tq[:3, :3].T @ Tq[:3, 3], X)
        ok = z > 0
        np.testing.assert_allclose(geom.apply_homography(Hm, uv)[ok], ref[ok], atol=1e-3)
    Ms = np.stack([pose_rot((0, 4.8, 0), s) for s in (0.0, 0.01, 0.02)])
    Hb = shutter.rotation_homography(K, A, Ms)
    assert Hb.shape == (3, 3, 3)
    for k in range(3):
        np.testing.assert_allclose(Hb[k], homog(K, A, Ms[k]), atol=1e-12)
    assert math.isfinite(float(FPS)) and W > 0
