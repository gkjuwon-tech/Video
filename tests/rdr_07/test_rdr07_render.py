"""RDR-07 단일 거울 렌더: 등가식(R ⊙ fliplr(render_ldi(clip, 가상))), 점 착지, 깊이 AOV, 가림, 누출, 뒷면."""
import numpy as np
import pytest

from buc.core import geom
from buc.render.mirror import Mirror, MirrorOpts, render_mirrors, virtual_camera
from buc.render.quads import RenderOpts, render_ldi
from rdr07_helpers import (H, I4, W, K_default, clip_oracle, const_rgba, layer, make_world, pixel_rays,
                           project_pt, smooth_tex, translate)

R_FLOOR = (0.7, 0.8, 0.9)
FLOOR = Mirror(center=(0, 1.0, 0), normal=(0, -1, 0), reflectance=R_FLOOR)   # y 아래(OpenCV) -> 바닥 y=1


def _floor_world():
    return make_world([layer(smooth_tex(), 4.0)])


@pytest.mark.parametrize("c2w", [I4, translate(I4, (0.05, -0.1, 0.2))])
def test_floor_mirror_equals_flipped_virtual_render_of_clipped_world(c2w):
    w = _floor_world()
    q = RenderOpts(bg=(0.05, 0.1, 0.15))
    K = K_default()
    r = render_mirrors(w, [FLOOR], c2w=c2w, K=K, width=W, height=H, opts=MirrorOpts(quad=q))
    prim = render_ldi(w, c2w=c2w, K=K, width=W, height=H, opts=q)
    c2w_v, K_v = virtual_camera(c2w, K, W, FLOOR)
    side = 1 if (np.asarray(FLOOR.normal) @ c2w[:3, 3] - FLOOR.d) > 0 else -1
    refl = render_ldi(clip_oracle(w, FLOOR.normal, FLOOR.d, side=side), c2w=c2w_v, K=K_v, width=W, height=H,
                      opts=q)
    m = r.mirror_id == 0
    assert 0.1 < m.mean() < 0.6, m.mean()
    exp = np.asarray(R_FLOOR, np.float32) * refl.rgb[:, ::-1]
    np.testing.assert_allclose(r.rgb[m], exp[m], atol=1e-4)
    np.testing.assert_allclose(r.virtual_depth[m], refl.depth[:, ::-1][m], rtol=1e-5)
    assert (r.bounce[m] == 1).all() and (r.alpha[m] == 1.0).all() and (r.layer_id[m] == -1).all()
    # 거울이 아닌 화소는 1차 렌더 그대로
    nm = ~m
    for k in ("rgb", "alpha", "depth", "layer_id"):
        np.testing.assert_array_equal(getattr(r, k)[nm], getattr(prim, k)[nm])
    assert (r.bounce[nm] == 0).all()
    np.testing.assert_array_equal(r.virtual_depth[nm], r.depth[nm])
    # 거울 화소 판정 = 화소 중심 적중 & (1차 알파 < .5 또는 1차 깊이 > 적중 깊이)
    eye, dw, dc = pixel_rays(c2w, K, W, H)
    t = (1.0 - eye[1]) / dw[..., 1]
    hit = (dw[..., 1] > 0) & (t >= 1e-3)
    vis = hit & ((prim.alpha < 0.5) | (prim.depth > t))
    border = np.abs(prim.depth - t) < 1e-3
    assert (m == vis)[~border].all()
    np.testing.assert_allclose(r.depth[m], t[m], rtol=1e-5)               # 1차 깊이 AOV = 바닥 적중 z


@pytest.mark.parametrize("k1", [0.0, -0.05])
def test_point_landing_primary_and_reflection(k1):
    """밝은 텍셀 하나: 1차는 project(X), 반사는 project(M X) 에 0.35 px 이내로 착지."""
    rgba = const_rgba((0.05, 0.05, 0.05), 1.0)
    u0, v0 = 58, 50
    rgba[v0, u0, :3] = (1.0, 0.9, 0.8)
    w = make_world([layer(rgba, 4.0)])
    c2w = translate(I4, (0.05, -0.1, 0.2))
    K = K_default()
    r = render_mirrors(w, [FLOOR], c2w=c2w, K=K, width=W, height=H, k1=k1)
    X = geom.unproject_world(w.obs.K, w.obs.R, w.obs.t, np.array([[u0, v0]], float), np.array([4.0]))[0]
    assert X[1] < 1.0
    MX = X.copy()
    MX[1] = 2.0 - X[1]
    for P, want_mirror in ((X, False), (MX, True)):
        uv, z = project_pt(c2w, K, P, k1)
        ui, vi = int(round(uv[0])), int(round(uv[1]))
        assert 3 <= ui < W - 3 and 3 <= vi < H - 3
        mid = 0 if want_mirror else -1
        assert r.mirror_id[vi, ui] == mid
        sl = np.s_[vi - 3:vi + 4, ui - 3:ui + 4]
        same = r.mirror_id[sl] == mid                                       # 같은 표면 화소만(거울 경계 제외)
        lum = r.rgb[sl][..., 0].astype(np.float64)
        base = np.median(r.rgb[..., 0][r.mirror_id == mid])
        wgt = np.clip(lum - base, 0, None) * same
        assert wgt.sum() > 0.3
        vv, uu = np.mgrid[vi - 3:vi + 4, ui - 3:ui + 4]
        cu, cv = (wgt * uu).sum() / wgt.sum(), (wgt * vv).sum() / wgt.sum()
        assert abs(cu - uv[0]) < 0.35 and abs(cv - uv[1]) < 0.35, (want_mirror, cu, cv, uv)
        if want_mirror:
            assert r.virtual_depth[vi, ui] == pytest.approx(z, abs=0.03)
            assert r.depth[vi, ui] < z - 0.2                               # 거울면이 허상보다 가깝다


def test_virtual_depth_follows_virtual_image_not_surface():
    """반사 문신 방지의 핵심: 반사 화소의 허상 깊이는 거울 면 깊이가 아니라 반사된 평면의 깊이."""
    w = _floor_world()
    c2w = translate(I4, (0.0, -0.1, 0.2))
    r = render_mirrors(w, [FLOOR], c2w=c2w, K=K_default(), width=W, height=H)
    m = (r.mirror_id == 0) & np.isfinite(r.virtual_depth)
    assert m.sum() > 200
    vd = r.virtual_depth[m]
    assert abs(np.median(vd) - 3.8) < 0.02                                 # 카드 z=4, 카메라 z=0.2
    assert (np.abs(vd - 3.8) < 0.05).mean() > 0.9
    assert (r.depth[m] < r.virtual_depth[m]).all()


def test_foreground_in_front_of_mirror_occludes_it():
    fg = np.zeros((H, W, 4), np.float32)
    fg[50:, 10:30] = (0.9, 0.1, 0.1, 1.0)                                  # z=2 앞의 빨간 카드 (바닥보다 가깝다)
    w = make_world([layer(fg, 2.0), layer(smooth_tex(), 4.0)])
    r = render_mirrors(w, [FLOOR], c2w=I4, K=K_default(), width=W, height=H)
    prim = render_ldi(w, c2w=I4, K=K_default(), width=W, height=H)
    blk = np.zeros((H, W), bool)
    blk[52:62, 12:28] = True
    assert (prim.alpha[blk] > 0.99).all()
    assert (r.mirror_id[blk] == -1).all() and (r.layer_id[blk] == 0).all()
    np.testing.assert_array_equal(r.rgb[blk], prim.rgb[blk])
    assert (r.mirror_id[54:62, 40:90] == 0).all()                          # 같은 행, 전경 밖은 거울


def _front_mirror(normal_z=-1.0, two_sided=False):
    return Mirror(center=(0, 0, 3.0), normal=(0, 0, normal_z), u_axis=(1, 0, 0), half_size=(0.6, 0.6),
                  reflectance=(0.9, 0.9, 0.9), two_sided=two_sided)


def _leak_world():
    red = np.zeros((H, W, 4), np.float32)
    v, u = np.mgrid[0:H, 0:W]
    X, Y = (u - (W - 1) / 2) / 83.1, (v - (H - 1) / 2) / 83.1             # 대략 x/z, y/z
    sel = (X > -0.5) & (X < -0.12) & (Y > -0.45) & (Y < -0.12)
    red[sel] = (1.0, 0.0, 0.0, 1.0)
    blue = const_rgba((0.0, 0.0, 1.0), 1.0)
    return make_world([layer(red, 2.0), layer(blue, 5.0)])                 # 파랑은 거울 평면(z=3) 뒤


def test_geometry_behind_mirror_never_appears_in_reflection():
    w = _leak_world()
    r = render_mirrors(w, [_front_mirror()], c2w=I4, K=K_default(), width=W, height=H)
    m = r.mirror_id == 0
    assert m.sum() > 300
    assert r.rgb[..., 2][m].max() < 1e-3, "거울 뒤 파랑이 반사에 샜다"
    assert (r.rgb[..., 0][m] > 0.85).sum() > 10                            # 앞쪽 빨강은 반사된다(0.9 x 1)


def test_back_face_is_invisible_and_two_sided_reflects_camera_side():
    w = _leak_world()
    ref = render_ldi(w, c2w=I4, K=K_default(), width=W, height=H)
    r = render_mirrors(w, [_front_mirror(normal_z=1.0)], c2w=I4, K=K_default(), width=W, height=H)
    assert (r.mirror_id == -1).all()
    assert r.rgb.tobytes() == ref.rgb.tobytes()
    r2 = render_mirrors(w, [_front_mirror(normal_z=1.0, two_sided=True)], c2w=I4, K=K_default(), width=W, height=H)
    r1 = render_mirrors(w, [_front_mirror()], c2w=I4, K=K_default(), width=W, height=H)
    np.testing.assert_array_equal(r2.mirror_id, r1.mirror_id)
    np.testing.assert_allclose(r2.rgb, r1.rgb, atol=1e-6)
    assert r2.rgb[..., 2][r2.mirror_id == 0].max() < 1e-3


def test_max_depth_zero_paints_terminal_times_reflectance():
    w = _floor_world()
    r = render_mirrors(w, [FLOOR], c2w=I4, K=K_default(), width=W, height=H,
                       opts=MirrorOpts(max_depth=0, terminal_rgb=(1.0, 0.5, 0.25)))
    m = r.mirror_id == 0
    assert m.sum() > 100
    np.testing.assert_allclose(r.rgb[m], np.broadcast_to(np.float32([0.7, 0.4, 0.225]), r.rgb[m].shape), atol=1e-6)
    assert (r.bounce[m] == 1).all() and np.isinf(r.virtual_depth[m]).all()
    assert r.n_renders == 1


def test_mirror_behind_reflecting_plane_is_ignored_in_reflection():
    """거울 A 뒤(평면 반대편)에 있는, 가상 카메라 쪽을 향한 거울 C 는 A 의 반사에 나타나면 안 된다(교점 클립)."""
    w = _leak_world()
    C_hidden = Mirror(center=(0, 0, 4.5), normal=(0, 0, 1), u_axis=(1, 0, 0), half_size=(2.0, 2.0),
                      reflectance=(0.5, 0.5, 0.5))
    a = render_mirrors(w, [_front_mirror()], c2w=I4, K=K_default(), width=W, height=H)
    b = render_mirrors(w, [_front_mirror(), C_hidden], c2w=I4, K=K_default(), width=W, height=H)
    for k in ("rgb", "alpha", "depth", "layer_id", "mirror_id", "bounce", "virtual_depth"):
        np.testing.assert_array_equal(getattr(b, k), getattr(a, k), err_msg=k)
    assert b.n_renders == a.n_renders == 2
