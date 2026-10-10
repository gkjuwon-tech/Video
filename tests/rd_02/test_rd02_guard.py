"""오류·옵션 검증, 결정론, 전역 RNG·입력 불변, torch 미로드, 하류 이음매(Theta·geom·to_meta)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import rd02_helpers as H
from buc.core import contracts, geom
from buc.read import geom_cam as gc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W, HH = 1280, 720


def _segs(seed=0):
    K = H.K_from_fov(60, W, HH)
    R = H.camera_R(30, 6, -3)
    s, _ = H.manhattan_segments(K, R, W, HH, seed=seed)
    return np.vstack([s, H.random_segments(20, W, HH, seed=seed + 50)])


@pytest.mark.parametrize("kw", [dict(inlier_deg=0.0), dict(inlier_deg=-1.0), dict(n_hypotheses=0), dict(min_cluster=1),
                                dict(k_bound=-0.1), dict(orphan_warn=1.5), dict(max_side=16), dict(grad_thresh=-0.1),
                                dict(prior_sigma_log_f=0.0), dict(inlier_deg=float("nan"))])
def test_opts_validation(kw):
    with pytest.raises(ValueError):
        gc.GeomCamOpts(**kw)


def test_opts_defaults_match_contract():
    o = gc.GeomCamOpts()
    assert (o.max_side, o.inlier_deg, o.min_cluster, o.k_bound, o.orphan_warn, o.r_vp_warn_deg, o.seed) == (
        1024, 2.0, 4, 0.3, 0.10, 2.0, 0)
    assert gc.DEFAULT_FOV_X_DEG == 50.0
    assert issubclass(gc.GeomCamError, ValueError)


def test_image_input_errors():
    with pytest.raises(ValueError):  # GeomCamError 는 ValueError 의 하위
        gc.estimate_geom_camera(np.zeros((20, 20)))
    with pytest.raises(TypeError):
        gc.estimate_geom_camera(np.full((64, 64), "a"))
    bad = np.zeros((64, 64))
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        gc.estimate_geom_camera(bad)
    with pytest.raises(ValueError):
        gc.estimate_geom_camera(np.zeros((64, 64, 3, 2)))
    with pytest.raises(ValueError):
        gc.estimate_from_segments(np.zeros((5, 3)), W, HH)


def test_too_small_raises_geomcam_error():
    with pytest.raises(gc.GeomCamError):
        gc.estimate_geom_camera(np.zeros((20, 200)))


def test_determinism_rng_and_inputs_untouched():
    segs = _segs(1)
    keep = segs.copy()
    np.random.seed(123)
    st = np.random.get_state()[1].copy()
    a = gc.estimate_from_segments(segs, W, HH, fov_prior_deg=55.0)
    b = gc.estimate_from_segments(segs, W, HH, fov_prior_deg=55.0)
    assert np.array_equal(np.random.get_state()[1], st)
    assert np.array_equal(segs, keep)
    for f in ("K", "vps", "vp_axis", "labels", "up_cam"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert (a.k1, a.k2, a.roll_deg, a.pitch_deg, a.r_vp_deg, a.orphan_frac, a.warnings) == (
        b.k1, b.k2, b.roll_deg, b.pitch_deg, b.r_vp_deg, b.orphan_frac, b.warnings)
    v1, l1 = gc.jlinkage(segs)
    v2, l2 = gc.jlinkage(segs)
    assert np.array_equal(v1, v2) and np.array_equal(l1, l2)


def test_image_input_untouched_and_rng():
    K = H.K_from_fov(60, 160, 120)
    img = H.render_room(K, H.camera_R(30, 8, -4), 160, 120)
    keep = img.copy()
    np.random.seed(7)
    st = np.random.get_state()[1].copy()
    a = gc.estimate_geom_camera(img)
    b = gc.estimate_geom_camera(img)
    assert np.array_equal(img, keep) and np.array_equal(np.random.get_state()[1], st)
    assert np.array_equal(a.K, b.K) and np.array_equal(a.segments, b.segments)


def test_result_is_frozen_and_consistent():
    g = gc.estimate_from_segments(_segs(2), W, HH)
    with pytest.raises(Exception):
        g.k1 = 1.0
    assert g.K.shape == (3, 3) and g.K.dtype == np.float64
    assert g.K[0, 0] == g.K[1, 1] and g.K[0, 1] == 0.0 and tuple(g.K[2]) == (0.0, 0.0, 1.0)
    assert g.fov_x_deg == pytest.approx(geom.fov_x_deg_from_K(g.K, W), abs=1e-9)
    assert geom.is_rotation(g.R_mc)
    np.testing.assert_allclose(g.R_mc[:, 1], g.up_cam, atol=1e-9)
    r, p = gc.roll_pitch_from_up(g.up_cam)
    assert r == pytest.approx(g.roll_deg, abs=1e-9) and p == pytest.approx(g.pitch_deg, abs=1e-9)
    np.testing.assert_allclose(g.horizon_line(), gc.horizon_line(g.K, g.up_cam), atol=1e-12)
    assert g.vps.shape == (len(g.vp_axis), 3) and g.segments.shape == (len(g.labels), 4)
    assert set(np.unique(g.vp_axis)) <= {-1, 0, 1, 2}
    assert g.labels.max() < len(g.vps) and g.labels.min() >= -1
    assert int((g.vp_axis >= 0).sum()) == g.n_manhattan_vps
    assert list(g.warnings) == sorted(set(g.warnings))


def test_theta_seam():
    g = gc.estimate_from_segments(_segs(3), W, HH)
    tf = g.theta_fields()
    assert set(tf) == {"focal_mm", "sensor_width_mm", "aspect"}
    assert tf["sensor_width_mm"] == 36.0 and tf["aspect"] == pytest.approx(W / HH)
    th = contracts.Theta(**tf)
    assert th.fov_x_deg() == pytest.approx(g.fov_x_deg, abs=1e-9)
    assert th.intrinsics(W, HH)[0, 0] == pytest.approx(g.K[0, 0], rel=1e-12)
    tf2 = g.theta_fields(sensor_width_mm=24.89)
    assert tf2["focal_mm"] == pytest.approx(g.K[0, 0] / W * 24.89, rel=1e-12)


def test_to_meta_is_strict_json():
    for g in (gc.estimate_from_segments(_segs(4), W, HH), gc.estimate_from_segments(np.zeros((0, 4)), W, HH)):
        m = g.to_meta()
        s = json.dumps(m, allow_nan=False)
        back = json.loads(s)
        for k in ("W", "H", "K", "fov_x_deg", "k1", "k2", "roll_deg", "pitch_deg", "n_manhattan_vps", "f_source",
                  "pp_source", "horizon_source", "r_vp_deg", "orphan_frac", "plumb_rms_px", "reliable", "warnings"):
            assert k in back, k
        np.testing.assert_allclose(np.array(back["K"]), g.K, rtol=1e-12)
        assert isinstance(back["warnings"], list)
        if math.isnan(g.r_vp_deg):
            assert back["r_vp_deg"] is None


def test_crop_seam_with_geom_crop_intrinsics():
    """RD-01 크롭: 원 영상 K 의 주점을 crop_intrinsics 로 옮겨 principal_prior 로 준다."""
    Wf, Hf = 1440, 810
    x0, y0 = 40, 70  # 비대칭 크롭: 주점이 크롭 중심에서 벗어난다
    K_full = H.K_from_fov(64.0, Wf, Hf)
    K_crop = geom.crop_intrinsics(K_full, x0, y0)
    R = H.camera_R(28, 5, 1)
    s, _ = H.manhattan_segments(K_crop, R, W, HH, seed=11)
    assert abs(K_crop[0, 2] - (W - 1) / 2) > 30 and abs(K_crop[1, 2] - (HH - 1) / 2) > 20
    g = gc.estimate_from_segments(s, W, HH, principal_prior=tuple(K_crop[:2, 2]))
    assert g.pp_source == "prior"
    np.testing.assert_allclose(g.K[:2, 2], K_crop[:2, 2], atol=1e-9)
    assert g.K[0, 0] == pytest.approx(K_full[0, 0], rel=0.03)


def test_no_torch_or_cv2_on_import():
    code = ("import sys; import buc.read.geom_cam as g; "
            "bad=[m for m in ('torch','cv2','skimage') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
