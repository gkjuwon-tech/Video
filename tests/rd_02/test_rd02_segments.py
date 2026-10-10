"""선분 입력에서의 VP·K·수평선 추정: 정확도, 사전값의 약함, 주점, 고아 VP(AI 불성립), VP 없음."""
import math

import numpy as np
import pytest

import rd02_helpers as H
from buc.read import geom_cam as gc

W, HH = 1280, 720
DIAG = math.hypot(W, HH)


def _scene(fov=60, yaw=30, pitch=6, roll=-3, pp=None, outlier_frac=0.15, seed=0, noise=0.3):
    K = H.K_from_fov(fov, W, HH, pp)
    R = H.camera_R(yaw, pitch, roll)
    segs, labs = H.manhattan_segments(K, R, W, HH, noise_px=noise, seed=seed)
    n_out = int(round(outlier_frac * len(segs) / (1 - outlier_frac)))
    if n_out:
        segs = np.vstack([segs, H.random_segments(n_out, W, HH, seed=seed + 100)])
        labs = np.r_[labs, -np.ones(n_out, int)]
    return K, R, segs, labs


def _fov_err(g, fov):
    return abs(g.fov_x_deg - fov) / fov


@pytest.mark.parametrize("fov,yaw,pitch,roll,seed", [(60, 30, 6, -3, 0), (45, -40, 12, 4, 1), (75, 20, -8, 2, 2)])
def test_accuracy_two_and_three_point_perspective(fov, yaw, pitch, roll, seed):
    K, R, segs, _ = _scene(fov, yaw, pitch, roll, seed=seed)
    g = gc.estimate_from_segments(segs, W, HH)
    assert g.n_manhattan_vps >= 2
    assert g.f_source in ("vp2", "vp3")
    assert _fov_err(g, fov) <= 0.03
    assert g.roll_deg == pytest.approx(roll, abs=0.5)
    assert g.pitch_deg == pytest.approx(pitch, abs=0.5)
    assert g.horizon_source == "vp"
    np.testing.assert_allclose(g.up_cam, H.up_cam(R), atol=math.radians(0.7))


def test_clean_scene_is_reliable_with_small_residual_and_no_orphans():
    _, _, segs, _ = _scene(outlier_frac=0.0)
    g = gc.estimate_from_segments(segs, W, HH)
    assert g.reliable
    assert g.orphan_frac <= 0.05
    assert g.r_vp_deg <= 1.0
    assert "vp_inconsistent" not in g.warnings
    # 프레임 축에 배정된 VP 는 R_mc 열과 inlier_deg 안
    opts = gc.GeomCamOpts()
    for i, ax in enumerate(g.vp_axis):
        if ax >= 0:
            assert H.ray_angle_deg(g.K, g.vps[i], g.R_mc[:, ax]) <= opts.inlier_deg
    # 축 배정 선분의 대부분이 GT 축과 일치하는 VP 에 묶인다
    assert (g.labels >= 0).mean() >= 0.9


def test_weak_prior_loses_to_vp_evidence():
    _, _, segs, _ = _scene(fov=60, seed=3)
    g = gc.estimate_from_segments(segs, W, HH, fov_prior_deg=75.0, horizon_prior=(5.0, -6.0))
    assert _fov_err(g, 60) <= 0.03
    assert g.roll_deg == pytest.approx(-3, abs=0.5) and g.pitch_deg == pytest.approx(6, abs=0.5)
    assert g.f_source in ("vp2", "vp3") and g.horizon_source == "vp"


def test_principal_point_from_three_finite_vps():
    pp = ((W - 1) / 2 + 40, (HH - 1) / 2 - 25)
    _, _, segs, _ = _scene(60, 35, 22, 3, pp=pp, outlier_frac=0.1, seed=4)
    g = gc.estimate_from_segments(segs, W, HH)
    assert g.n_manhattan_vps == 3 and g.f_source == "vp3" and g.pp_source == "vp3"
    assert math.hypot(g.K[0, 2] - pp[0], g.K[1, 2] - pp[1]) <= 0.03 * DIAG
    assert _fov_err(g, 60) <= 0.03
    assert g.pitch_deg == pytest.approx(22, abs=0.5) and g.roll_deg == pytest.approx(3, abs=0.5)


def test_principal_point_fixed_when_vertical_vp_far_and_prior_is_used():
    pp = ((W - 1) / 2 - 60.0, (HH - 1) / 2 + 30.0)  # 크롭된 벤치 입력: 원 주점을 크롭 좌표로
    _, _, segs, _ = _scene(60, 30, 4, -2, pp=pp, seed=5)
    g = gc.estimate_from_segments(segs, W, HH, principal_prior=pp)
    assert g.pp_source == "prior"
    np.testing.assert_allclose(g.K[:2, 2], pp, atol=1e-9)
    assert _fov_err(g, 60) <= 0.03
    g0 = gc.estimate_from_segments(segs, W, HH)
    assert g0.pp_source == "center"
    np.testing.assert_allclose(g0.K[:2, 2], [(W - 1) / 2, (HH - 1) / 2], atol=1e-9)
    assert g0.K[0, 0] == g0.K[1, 1] and g0.K[0, 1] == 0.0


def test_single_vertical_vp_uses_prior_focal_and_vertical_for_horizon():
    K = H.K_from_fov(55, W, HH)
    R = H.camera_R(10, 15, 4)
    segs, _ = H.manhattan_segments(K, R, W, HH, n_per_axis=(80,), axes=(1,), seed=6)
    g = gc.estimate_from_segments(segs, W, HH, fov_prior_deg=55.0)
    assert g.f_source == "prior"
    assert g.fov_x_deg == pytest.approx(55.0, rel=1e-6)
    assert g.horizon_source == "vp"
    assert g.roll_deg == pytest.approx(4, abs=0.5) and g.pitch_deg == pytest.approx(15, abs=0.5)
    assert "single_vp" in g.warnings and not g.reliable


def test_no_segments_uses_priors_then_defaults():
    empty = np.zeros((0, 4))
    g = gc.estimate_from_segments(empty, W, HH, fov_prior_deg=42.0, horizon_prior=(1.5, -2.5))
    assert g.f_source == "prior" and g.horizon_source == "prior"
    assert g.fov_x_deg == pytest.approx(42.0, rel=1e-9)
    assert g.roll_deg == pytest.approx(1.5, abs=1e-9) and g.pitch_deg == pytest.approx(-2.5, abs=1e-9)
    assert g.n_manhattan_vps == 0 and g.R_mc is None and not g.reliable and "no_vp" in g.warnings
    assert math.isnan(g.r_vp_deg) and g.orphan_frac == 0.0
    d = gc.estimate_from_segments(empty, W, HH)
    assert d.f_source == "default" and d.horizon_source == "default" and d.pp_source == "center"
    assert d.fov_x_deg == pytest.approx(gc.DEFAULT_FOV_X_DEG, rel=1e-9)
    assert d.roll_deg == 0.0 and d.pitch_deg == 0.0


def test_random_segments_do_not_crash():
    segs = H.random_segments(40, W, HH, seed=7)
    g = gc.estimate_from_segments(segs, W, HH, fov_prior_deg=50.0)
    assert np.isfinite(g.K).all() and 10.0 < g.fov_x_deg < 140.0
    assert 0.0 <= g.orphan_frac <= 1.0


def test_pasted_perspective_is_measured_not_fitted():
    """AI 불성립 함정: 왼쪽 반은 카메라 A, 오른쪽 반은 yaw 가 12° 다른 카메라 B(같은 수직)."""
    K = H.K_from_fov(60, W, HH)
    RA = H.camera_R(30, 6, -3)
    RB = H.camera_R(42, 6, -3)
    sa, _ = H.manhattan_segments(K, RA, W, HH, seed=8, x_range=(0, W / 2))
    sb, _ = H.manhattan_segments(K, RB, W, HH, seed=9, x_range=(W / 2, W - 1))
    segs = np.vstack([sa, sb])
    g = gc.estimate_from_segments(segs, W, HH, fov_prior_deg=60.0)
    assert np.isfinite(g.K).all() and np.isfinite(g.roll_deg) and np.isfinite(g.pitch_deg)
    assert g.orphan_frac >= 0.15
    assert "vp_inconsistent" in g.warnings
    assert not g.reliable
    assert (g.vp_axis == -1).any()
    # 수직은 두 반쪽이 공유하므로 수평선은 여전히 맞다
    assert g.roll_deg == pytest.approx(-3, abs=1.0) and g.pitch_deg == pytest.approx(6, abs=1.0)
    clean = gc.estimate_from_segments(sa, W, HH, fov_prior_deg=60.0)
    assert g.orphan_frac > clean.orphan_frac + 0.1


def test_jlinkage_clusters_families():
    K, R, segs, labs = _scene(outlier_frac=0.2, seed=10)
    vps, lab = gc.jlinkage(segs)
    assert vps.ndim == 2 and vps.shape[1] == 3 and lab.shape == (len(segs),)
    np.testing.assert_allclose(np.linalg.norm(vps, axis=1), 1.0, atol=1e-9)
    assert lab.min() >= -1 and lab.max() < len(vps)
    lengths = np.hypot(segs[:, 0] - segs[:, 2], segs[:, 1] - segs[:, 3])
    tot = [lengths[lab == j].sum() for j in range(len(vps))]
    assert all(tot[j] >= tot[j + 1] for j in range(len(tot) - 1))  # 총 길이 내림차순
    for ax in range(3):
        fam = lab[labs == ax]
        mode = np.bincount(fam[fam >= 0]).argmax()
        assert (fam == mode).mean() >= 0.9  # 재현율
        assert (labs[lab == mode] == ax).mean() >= 0.85  # 정밀도
        v_gt = K @ (R @ H.AXES[ax])
        assert H.ray_angle_deg(K, vps[mode], np.linalg.solve(K, v_gt)) <= 1.0
