"""RD-05 모션블러 방향장: 선 커널 복원, 등방·평탄 거부, 일관성 수식, read_blur 판정."""
import json
import math

import numpy as np
import pytest
from scipy import ndimage

import rd05_helpers as H
from buc.read import blur_est as B


@pytest.mark.parametrize("length", [7, 11, 15, 21])
@pytest.mark.parametrize("theta_deg", [0, 30, 90, 135])
def test_line_blur_direction_and_length(length, theta_deg):
    img = H.motion_image(length, theta_deg, seed=length + theta_deg)
    m = B.estimate_motion_field(img)
    assert m.theta.shape == m.length_px.shape == m.anisotropy.shape == m.is_motion.shape == (3, 3)
    assert m.tile_px == 64
    th = math.radians(theta_deg)
    tol_l = max(1.5, 0.15 * length)
    ok = [(m.is_motion[i, j] and H.ang_err_deg(m.theta[i, j], th) <= 8.0 and abs(m.length_px[i, j] - length) <= tol_l)
          for i in range(3) for j in range(3)]
    assert sum(ok) >= 8                     # 타일 ≥80 %
    assert m.frac_motion >= 0.8
    assert 0.0 <= m.theta_global < math.pi and H.ang_err_deg(m.theta_global, th) <= 8.0
    assert abs(m.length_global - length) <= tol_l
    assert np.all((m.anisotropy >= 0) & (m.anisotropy <= 1))
    vals = m.theta[np.isfinite(m.theta)]
    assert np.all((vals >= 0) & (vals < math.pi))


def test_rgb_equals_luminance_and_tile_grid():
    g = H.motion_image(11, 60, h=150, w=200, seed=3)
    rgb = np.stack([g, g, g], -1)
    a = B.estimate_motion_field(g)
    b = B.estimate_motion_field(rgb)
    assert a.theta.shape == (2, 3)                       # 150//64, 200//64 (나머지 버림)
    assert np.allclose(np.nan_to_num(a.theta), np.nan_to_num(b.theta), atol=1e-6)
    m = B.estimate_motion_field(g, opts=B.BlurOpts(tile_px=32))
    assert m.theta.shape == (4, 6) and m.tile_px == 32
    with pytest.raises(ValueError):
        B.estimate_motion_field(g[:40, :40])             # 타일 0개
    bad = g.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        B.estimate_motion_field(bad)


def test_isotropic_blur_and_flat_are_not_motion():
    n = H.noise_tex(192, 192, seed=4, pad=0)
    iso = ndimage.gaussian_filter(n, 3.0)
    m = B.estimate_motion_field(iso)
    assert m.is_motion.sum() <= 1 and m.frac_motion <= 0.12
    sharp = B.estimate_motion_field(H.noise_tex(192, 192, seed=5, pad=0))
    assert sharp.is_motion.sum() == 0 and sharp.theta_global is None and sharp.length_global is None
    flat = B.estimate_motion_field(np.full((128, 128), 0.3))
    assert flat.is_motion.sum() == 0 and np.all(np.isnan(flat.theta)) and np.all(flat.anisotropy == 0)
    assert np.all(np.isnan(flat.length_px))
    json.dumps(flat.to_meta(), allow_nan=False)


def test_mixed_field_left_right():
    left = H.motion_image(13, 0, h=128, w=128, seed=6)
    right = H.motion_image(13, 90, h=128, w=128, seed=7)
    m = B.estimate_motion_field(np.concatenate([left, right], 1))
    assert m.theta.shape == (2, 4)
    for i in range(2):
        for j in range(2):
            assert H.ang_err_deg(m.theta[i, j], 0.0) <= 10.0
            assert H.ang_err_deg(m.theta[i, j + 2], math.pi / 2) <= 10.0


def test_motion_depth_consistency_formula():
    z = np.geomspace(1.0, 30.0, 40)
    L = 2.0 + 30.0 / z
    a, b, rel = B.motion_depth_consistency(L, z)
    assert abs(a - 2.0) <= 1e-6 and abs(b - 30.0) <= 1e-6 and rel <= 1e-9
    # 비음수 제약: 길이가 깊이와 함께 늘면 b=0으로 막히고 잔차가 남는다
    a2, b2, rel2 = B.motion_depth_consistency(1.0 + 0.5 * z, z)
    assert b2 == 0.0 and a2 >= 0 and rel2 > 0.3
    # 가중 오라클(상수 모델)
    Lc = np.array([4.0, 10.0, 4.0, 10.0])
    zc = np.full(4, 5.0)
    w = np.array([1.0, 1.0, 3.0, 3.0])
    a3, b3, rel3 = B.motion_depth_consistency(Lc, zc, w)
    pred = a3 + b3 / 5.0
    assert abs(pred - 7.0) <= 1e-6
    assert abs(rel3 - 3.0 / 7.0) <= 1e-6
    # NaN 길이(모션 아님)는 빠지고, 점이 2개 미만이면 nan
    a4, b4, rel4 = B.motion_depth_consistency(np.array([5.0, np.nan]), np.array([2.0, 3.0]))
    assert math.isnan(a4) and math.isnan(b4) and math.isnan(rel4)


def test_read_blur_physical_and_nonphysical_motion():
    c, z = H.scene(3.0, 2.0, seed=30, shape=(48, 64), noise=0.02)
    img = H.motion_image(11, 0, h=192, w=256, seed=31)
    r = B.read_blur(defocus=c, depth_m=z, f_px=H.F_PX, width_px=H.W, image=img)
    assert r.defocus.status == "fit" and abs(r.defocus.z_f_m / 3.0 - 1) <= 0.1
    assert r.motion is not None and r.motion.theta.shape == (3, 4)
    assert r.motion_consistency is not None
    assert r.motion_consistency[2] <= 0.3 and "motion_nonphysical" not in r.warnings
    meta = r.to_meta()
    json.dumps(meta, allow_nan=False)
    assert set(meta) >= {"defocus", "motion", "motion_consistency", "warnings"}
    # 같은 깊이에서 길이가 반반 다르다 → 깊이로 설명 불가 → 비물리
    left = H.motion_image(5, 0, h=192, w=128, seed=32)
    right = H.motion_image(19, 0, h=192, w=128, seed=33)
    zc = np.full((48, 64), 6.0)
    cc = H.oracle_coc(zc, 4.0, 3.0)
    r2 = B.read_blur(defocus=cc, depth_m=zc, f_px=H.F_PX, width_px=H.W, image=np.concatenate([left, right], 1))
    assert r2.motion_consistency is not None and r2.motion_consistency[2] > 0.3
    assert "motion_nonphysical" in r2.warnings
    assert set(r2.defocus.warnings) <= set(r2.warnings)
    assert list(r2.warnings) == sorted(set(r2.warnings))


def test_read_blur_without_image_and_few_motion_tiles():
    c, z = H.scene(3.0, 2.0, seed=34, shape=(48, 64))
    r = B.read_blur(defocus=c, depth_m=z, f_px=H.F_PX, width_px=H.W)
    assert r.motion is None and r.motion_consistency is None
    assert r.warnings == r.defocus.warnings
    json.dumps(r.to_meta(), allow_nan=False)
    sharp = H.noise_tex(192, 256, seed=35, pad=0)
    r2 = B.read_blur(defocus=c, depth_m=z, f_px=H.F_PX, width_px=H.W, image=sharp)
    assert r2.motion is not None and r2.motion_consistency is None
