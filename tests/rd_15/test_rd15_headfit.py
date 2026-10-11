"""RD-15 머리 모형(타원체 + 코 타원체) 전방 렌더와 LM 피팅, σ_z.

정답 렌더는 helpers.render_oracle(계약서 수식 그대로, 구현과 독립)이다. 합성 머리는 모형 안의 모양이라
복원 허용 오차는 '모형이 맞을 때의 능력'을 잰다. 실제 얼굴과의 격차는 미검증이다(계약서)."""
import math

import numpy as np
import pytest

import rd15_helpers as H

HW = (480, 640)
PP = H.pp_of(*HW)
GRID = (HW[0] // 4, HW[1] // 4)


def truth(yaw, pitch, dist, ox=0.0, oy=0.0, width=0.15):
    cen = np.array([ox, oy, dist])
    front, up, (yw, pt) = H.pose_from_yaw_pitch(cen, yaw, pitch)
    return cen, front, up, yw, pt, width


def crop_inputs(cen, front, up, width, *, noise_d=0.0, noise_n=0.0, seed=0, s_true=H.S_TRUE):
    """머리가 차지하는 격자 부분만 잘라 (depth_rel, normal, mask, origin) 로 만든다."""
    z, n, m = H.render_oracle(cen, front, up, width, H.F_PX, GRID, 4.0, PP)
    rng = np.random.default_rng(seed)
    zn = z + (rng.normal(0, noise_d, z.shape) if noise_d else 0.0)
    nn = n.copy()
    if noise_n:
        nn = n + rng.normal(0, math.tan(math.radians(noise_n)), n.shape) * m[..., None]
        nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-9)
    ys, xs = np.nonzero(m)
    sl = (slice(max(ys.min() - 1, 0), ys.max() + 2), slice(max(xs.min() - 1, 0), xs.max() + 2))
    d = np.nan_to_num(zn[sl] / s_true, nan=1.0)
    return d, nn[sl], m[sl], (4.0 * sl[0].start, 4.0 * sl[1].start)


def do_fit(cen, front, up, width, *, scale, **kw):
    fm = H.fm()
    d, n, m, org = crop_inputs(cen, front, up, width, **kw)
    return fm.fit_head(d, n, m, f_px=H.F_PX, pp=PP, cell_px=4.0, origin=org, scale=scale), int(m.sum())


# ----------------------------------------------------------------------------- 전방 렌더
@pytest.mark.parametrize("yaw,pitch,dist,width", [(0, 0, 1.5, 0.15), (30, -10, 1.2, 0.15), (-20, 15, 2.0, 0.17),
                                                  (60, 0, 1.0, 0.13)])
def test_render_head_matches_oracle(yaw, pitch, dist, width):
    fm = H.fm()
    cen, front, up, yw, pt, _ = truth(yaw, pitch, dist, 0.05, -0.03, width)
    r = fm.render_head(cen, front, up, width, f_px=H.F_PX, grid_hw=(60, 80), pp=PP, cell_px=8.0, origin=(0.0, 0.0))
    z, n, m = H.render_oracle(cen, front, up, width, H.F_PX, (60, 80), 8.0, PP)
    assert r.mask.shape == (60, 80) and r.depth.shape == (60, 80) and r.normal.shape == (60, 80, 3)
    assert (r.mask != m).mean() <= 0.01
    both = r.mask & m
    assert both.sum() > 25
    np.testing.assert_allclose(np.asarray(r.depth, np.float64)[both], z[both], atol=1e-4)
    cosv = (np.asarray(r.normal, np.float64)[both] * n[both]).sum(-1)
    assert float(np.degrees(np.arccos(np.clip(cosv, -1, 1))).max()) < 0.5
    assert np.isnan(np.asarray(r.depth)[~r.mask]).all()
    assert (np.abs(np.asarray(r.normal)[~r.mask]).sum() == 0)


def test_render_head_geometry_facts():
    fm = H.fm()
    cen, front, up, *_ = truth(0, 0, 1.5)
    r = fm.render_head(cen, front, up, 0.15, f_px=H.F_PX, grid_hw=GRID, pp=PP)
    m = r.mask
    d = np.asarray(r.depth, np.float64)
    nrm = np.asarray(r.normal, np.float64)
    assert float(np.nanmin(d)) == pytest.approx(1.5 - 0.13, abs=0.004)               # 코끝
    assert float(np.nanmax(d)) < 1.5 + 0.0                                          # 보이는 면은 중심보다 앞
    assert np.allclose(np.linalg.norm(nrm[m], axis=-1), 1.0, atol=1e-5)
    assert (nrm[m][:, 2] < 0).all()                                                  # 카메라를 향한다(FND-08)
    ys, xs = np.nonzero(m)
    h_px = (ys.max() - ys.min() + 1) * 4
    w_px = (xs.max() - xs.min() + 1) * 4
    assert h_px == pytest.approx(H.F_PX * 0.23 / 1.5, rel=0.08) and w_px == pytest.approx(H.F_PX * 0.15 / 1.5, rel=0.1)
    # 가까우면 커지고, 폭 인자는 모형 전체를 비례 확대한다
    near = fm.render_head(cen * np.array([1, 1, 0.7]), front, up, 0.15, f_px=H.F_PX, grid_hw=GRID, pp=PP).mask
    assert near.sum() > 1.7 * m.sum()
    wide = fm.render_head(cen, front, up, 0.30, f_px=H.F_PX, grid_hw=GRID, pp=PP).mask
    assert 3.2 < wide.sum() / m.sum() < 4.8
    assert fm.HEAD_WIDTH_REF_M == 0.15 and tuple(fm.HEAD_CRANIUM_AXES_M) == (0.075, 0.115, 0.095)
    assert tuple(fm.HEAD_NOSE_CENTER_M) == (0.0, -0.015, 0.100) and tuple(fm.HEAD_NOSE_AXES_M) == (0.015, 0.035, 0.030)


def test_render_head_is_mirror_symmetric_for_a_frontal_face():
    """정면(얼굴 좌우가 영상 좌우와 평행)이면 깊이·마스크가 좌우 반사에 대칭이다."""
    fm = H.fm()
    cen = np.array([0.0, 0.0, 1.5])
    front, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])
    pp = (PP[0], (HW[1] - 1) / 2)
    r = fm.render_head(cen, front, up, 0.15, f_px=H.F_PX, grid_hw=GRID, pp=pp)
    d = np.nan_to_num(np.asarray(r.depth, np.float64), nan=0.0)
    assert np.abs(d - d[:, ::-1]).max() < 1e-6
    assert (r.mask == r.mask[:, ::-1]).all()


# ----------------------------------------------------------------------------- 복원 (무잡음)
POSES = [(20, 0, 1.5), (-30, 10, 1.5), (35, -10, 2.0)]


@pytest.mark.parametrize("yaw,pitch,dist", POSES)
def test_noiseless_recovery_with_known_scale(yaw, pitch, dist):
    fm = H.fm()
    cen, front, up, yw, pt, w = truth(yaw, pitch, dist, 0.04, -0.02)
    fit, n = do_fit(cen, front, up, w, scale=H.S_TRUE)
    assert fit.ok and fit.n_cells == n and fit.scale_source == "ms" and fit.scale == pytest.approx(H.S_TRUE)
    assert abs(fit.yaw_deg - yw) <= 3.0 and abs(fit.pitch_deg - pt) <= 3.0, (fit.yaw_deg, yw, fit.pitch_deg, pt)
    assert np.linalg.norm(np.asarray(fit.center) - cen) <= 0.02
    assert fit.width_m == pytest.approx(0.15, rel=0.05)
    assert fm.WIDTH_RANGE_M[0] <= fit.width_m <= fm.WIDTH_RANGE_M[1]
    assert fit.rms_depth_m <= 0.005 and fit.rms_normal_deg <= 10.0 and fit.iou >= 0.9
    f, u = np.asarray(fit.front), np.asarray(fit.up)
    assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-6) and np.linalg.norm(u) == pytest.approx(1.0, abs=1e-6)
    assert abs(f @ u) < 1e-6
    # yaw/pitch 는 CAM-04 face_angles(원점 카메라) 정의 그대로다
    from buc.camera.face_envelope import FaceSpec, face_angles
    yaw_c, pitch_c, _ = face_angles(FaceSpec(center=fit.center, front=fit.front, up=fit.up, h_f=0.1), np.zeros(3))
    assert fit.yaw_deg == pytest.approx(yaw_c, abs=1e-6) and fit.pitch_deg == pytest.approx(pitch_c, abs=1e-6)


@pytest.mark.parametrize("yaw,pitch,dist", [(0, 0, 1.5), (-30, 10, 1.5)])
def test_noiseless_recovery_with_head_prior_scale(yaw, pitch, dist):
    """스케일을 모르면 머리 폭 15 cm 가 스케일을 정한다(폭 고정, 깊이 스케일을 맞춤)."""
    cen, front, up, yw, pt, w = truth(yaw, pitch, dist)
    fit, _ = do_fit(cen, front, up, w, scale=None)
    assert fit.ok and fit.scale_source == "head_prior"
    assert fit.scale == pytest.approx(H.S_TRUE, rel=0.04)
    assert fit.width_m == pytest.approx(0.15, abs=1e-9)
    assert abs(fit.yaw_deg - yw) <= 3.0 and abs(fit.pitch_deg - pt) <= 3.0
    assert np.linalg.norm(np.asarray(fit.center) - cen) <= 0.03
    assert fit.rms_depth_m <= 0.005


def test_prior_mode_is_invariant_to_the_unknown_depth_gauge():
    fm = H.fm()
    cen, front, up, yw, pt, w = truth(15, 5, 1.5)
    d, n, m, org = crop_inputs(cen, front, up, w)
    a = fm.fit_head(d, n, m, f_px=H.F_PX, pp=PP, origin=org)
    b = fm.fit_head(d * 2.5, n, m, f_px=H.F_PX, pp=PP, origin=org)
    assert b.scale == pytest.approx(a.scale / 2.5, rel=0.03)
    assert np.linalg.norm(np.asarray(a.center) - np.asarray(b.center)) <= 0.01
    assert abs(a.yaw_deg - b.yaw_deg) <= 1.0 and abs(a.pitch_deg - b.pitch_deg) <= 1.0


def test_fit_is_deterministic_and_leaves_inputs_and_rng_alone():
    fm = H.fm()
    cen, front, up, yw, pt, w = truth(10, 0, 2.0)
    d, n, m, org = crop_inputs(cen, front, up, w)
    keep = (d.copy(), n.copy(), m.copy())
    st = np.random.get_state()[1].copy()
    a = fm.fit_head(d, n, m, f_px=H.F_PX, pp=PP, origin=org, scale=H.S_TRUE)
    b = fm.fit_head(d, n, m, f_px=H.F_PX, pp=PP, origin=org, scale=H.S_TRUE)
    np.testing.assert_array_equal(np.asarray(a.center), np.asarray(b.center))
    assert (a.yaw_deg, a.pitch_deg, a.rms_depth_m, a.width_m) == (b.yaw_deg, b.pitch_deg, b.rms_depth_m, b.width_m)
    for x, y in zip((d, n, m), keep):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(np.random.get_state()[1], st)


# ----------------------------------------------------------------------------- 잡음, σ_z 의 의미
def test_noisy_recovery_and_rms_tracks_injected_depth_noise():
    fm = H.fm()
    cen, front, up, yw, pt, w = truth(20, 5, 1.5)
    rms = []
    for noise in (0.01, 0.02):
        fit, _ = do_fit(cen, front, up, w, scale=H.S_TRUE, noise_d=noise, noise_n=6.0, seed=3)
        assert fit.ok
        assert abs(fit.yaw_deg - yw) <= 7.0 and abs(fit.pitch_deg - pt) <= 7.0, (noise, fit.yaw_deg, yw)
        rms.append(fit.rms_depth_m)
    assert 0.007 <= rms[0] <= 0.014, rms
    assert 0.014 <= rms[1] <= 0.028, rms
    assert rms[0] < rms[1]
    # 잔차가 곧 σ_z 로 이어진다(σ̂ 항 없음)
    assert 0.007 <= fm.combine_sigma_z(rms[0]) <= 0.014


def test_a_flat_photo_does_not_pass_as_a_head():
    """얼굴 크기의 평평한 판(사진·그림): 깊이 기복이 없다 -> 머리 모형과 어긋난다."""
    fm = H.fm()
    cen, front, up, yw, pt, w = truth(0, 0, 1.7)
    d, n, m, org = crop_inputs(cen, front, up, w)
    flat_d = np.full_like(d, float(np.median(d[m])))
    flat_n = np.zeros_like(n)
    flat_n[..., 2] = -1.0
    fit = fm.fit_head(flat_d, flat_n, m, f_px=H.F_PX, pp=PP, origin=org, scale=H.S_TRUE)
    assert (not fit.ok) or fit.rms_depth_m >= 0.010 or fit.rms_normal_deg >= 15.0, (fit.rms_depth_m, fit.rms_normal_deg)


# ----------------------------------------------------------------------------- 검증
def test_fit_head_validation():
    fm = H.fm()
    cen, front, up, yw, pt, w = truth(0, 0, 1.5)
    d, n, m, org = crop_inputs(cen, front, up, w)
    kw = dict(f_px=H.F_PX, pp=PP, origin=org, scale=H.S_TRUE)
    with pytest.raises(ValueError):
        fm.fit_head(d[:-1], n, m, **kw)                                  # 모양 불일치
    with pytest.raises(ValueError):
        fm.fit_head(d, n[..., :2], m, **kw)
    with pytest.raises(ValueError):
        fm.fit_head(d, n, m[:, :-1], **kw)
    tiny = np.zeros_like(m)
    tiny[:3, :3] = True
    with pytest.raises(ValueError):                                       # 칸이 MIN_FIT_CELLS 미만
        fm.fit_head(d, n, tiny, **kw)
    assert fm.MIN_FIT_CELLS == 24
    bad = d.copy()
    bad[np.nonzero(m)[0][0], np.nonzero(m)[1][0]] = np.nan
    with pytest.raises(ValueError):
        fm.fit_head(bad, n, m, **kw)
    neg = d.copy()
    neg[m] *= -1
    with pytest.raises(ValueError):
        fm.fit_head(neg, n, m, **kw)
    for k, v in (("f_px", 0.0), ("f_px", -1.0), ("scale", 0.0), ("scale", -2.0), ("cell_px", 0.0)):
        with pytest.raises(ValueError):
            fm.fit_head(d, n, m, **{**kw, k: v})
