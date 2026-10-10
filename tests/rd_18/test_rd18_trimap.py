"""RD-18 trimap: 깊이 불연속 검출, 띠 폭, 근/원 라벨, thin 합류, 깊이 업샘플."""
import numpy as np
import pytest

import rd18_helpers as H


def _eu():
    from buc.read import edge_unmix as M
    return M


def test_codes_and_config_defaults():
    M = _eu()
    assert (M.BG, M.UNKNOWN, M.FG) == (0, 128, 255)
    c = M.UnmixConfig()
    assert (c.rel_jump, c.band_px, c.eps, c.win_r) == (0.10, 3, 1e-7, 1)
    assert (c.ridge, c.gamma, c.delta, c.clip_tau, c.recon_tau) == (1e-9, 0.1, 1e-6, 0.2, 0.02)
    assert issubclass(M.EdgeUnmixError, ValueError)


@pytest.mark.parametrize("kw", [dict(rel_jump=0.0), dict(band_px=0), dict(eps=-1.0), dict(win_r=0),
                                dict(gamma=0.0), dict(delta=0.0), dict(ridge=0.0), dict(clip_tau=-0.1),
                                dict(recon_tau=-1.0), dict(rel_jump=float("nan"))])
def test_config_validation(kw):
    M = _eu()
    with pytest.raises(ValueError):
        M.UnmixConfig(**kw)


def test_depth_edges_near_far_and_threshold():
    M = _eu()
    z = np.full((6, 8), 2.0)
    z[:, 4:] = 2.21          # 10.5 % 점프 -> 에지
    near, far = M.depth_edges(z)
    assert near.dtype == bool and far.dtype == bool
    exp_near = np.zeros((6, 8), bool); exp_near[:, 3] = True
    exp_far = np.zeros((6, 8), bool); exp_far[:, 4] = True
    assert np.array_equal(near, exp_near) and np.array_equal(far, exp_far)
    z[:, 4:] = 2.19          # 9.5 % -> 에지 아님
    near, far = M.depth_edges(z)
    assert not near.any() and not far.any()
    # 문턱 인자
    near, far = M.depth_edges(z, rel_jump=0.05)
    assert near[:, 3].all() and far[:, 4].all()


def test_depth_edges_pixel_both_near_and_far_and_vertical():
    M = _eu()
    z = np.full((5, 5), 2.0)
    z[:, 2] = 3.0
    z[:, 3:] = 5.0
    near, far = M.depth_edges(z)
    assert near[:, 2].all() and far[:, 2].all()        # 왼쪽보다 멀고 오른쪽보다 가깝다
    assert near[:, 1].all() and not far[:, 1].any()
    assert far[:, 3].all() and not near[:, 3].any()
    zt = z.T.copy()
    n2, f2 = M.depth_edges(zt)
    assert np.array_equal(n2, near.T) and np.array_equal(f2, far.T)


def test_depth_edges_ignores_invalid():
    M = _eu()
    z = np.full((4, 6), 2.0)
    z[:, 3] = np.nan
    z[:, 4:] = 9.0
    z[0, 0] = -1.0
    near, far = M.depth_edges(z)
    assert not near.any() and not far.any()


def test_trimap_straight_edge_band_width_and_sides():
    M = _eu()
    h, w, c = 10, 30, 14
    z = np.where(np.arange(w)[None, :] <= c, 2.0, 4.0) * np.ones((h, 1))
    t = M.make_trimap(z)
    assert t.dtype == np.uint8 and t.shape == (h, w)
    exp = np.full(w, M.FG, np.uint8)
    exp[c - 2: c + 4] = M.UNKNOWN          # 경계 양쪽 3 px씩, 폭 6
    exp[c + 4:] = M.BG
    assert np.array_equal(t, np.tile(exp, (h, 1)))
    t5 = M.make_trimap(z, band_px=5)
    assert (t5[0] == M.UNKNOWN).sum() == 10
    t1 = M.make_trimap(z, band_px=1)
    assert np.flatnonzero(t1[0] == M.UNKNOWN).tolist() == [c, c + 1]


def test_trimap_far_side_left_is_background():
    M = _eu()
    z = np.where(np.arange(30)[None, :] <= 14, 7.0, 2.0) * np.ones((6, 1))
    t = M.make_trimap(z)
    assert (t[:, :12] == M.BG).all() and (t[:, 18:] == M.FG).all()


def test_trimap_disc_labels_inside_fg_outside_bg():
    M = _eu()
    img, z, at, _, _ = H.disc_scene()
    t = M.make_trimap(z)
    U = t == M.UNKNOWN
    assert (t[(at > 0.999) & ~U] == M.FG).all()
    assert (t[(at < 1e-3) & ~U] == M.BG).all()
    assert U[(at > 0.01) & (at < 0.99)].all()      # 혼합 화소는 모두 미지 띠 안
    yy, xx = np.mgrid[0:96, 0:96]
    d = np.hypot(yy - 48 + 0.3, xx - 48 + 0.2) - 30
    assert not U[np.abs(d) > 4.5].any()             # 띠는 경계에서 멀리 가지 않는다


def test_trimap_no_edges_all_fg_and_thin_union():
    M = _eu()
    z = np.full((12, 12), 3.0)
    t = M.make_trimap(z)
    assert (t == M.FG).all()
    thin = np.zeros((12, 12), bool); thin[3:5, 2:9] = True
    t = M.make_trimap(z, thin=thin)
    assert np.array_equal(t == M.UNKNOWN, thin)
    assert (t[~thin] == M.FG).all()
    with pytest.raises(ValueError):
        M.make_trimap(z, thin=thin[:-1])
    with pytest.raises(TypeError):
        M.make_trimap(z, thin=thin.astype(np.float32))


@pytest.mark.parametrize("s", [1, 2, 4])
def test_upsample_depth_nearest_and_stride(s):
    M = _eu()
    Hh, Ww = 37, 50
    h, w = -(-Hh // s), -(-Ww // s)
    rng = np.random.default_rng(3)
    zc = rng.uniform(1, 5, (h, w))
    z, st = M.upsample_depth(zc, (Hh, Ww))
    assert st == s and z.shape == (Hh, Ww) and z.dtype == np.float64
    assert np.array_equal(z, np.repeat(np.repeat(zc, s, 0), s, 1)[:Hh, :Ww])


def test_upsample_depth_bad_shape():
    M = _eu()
    with pytest.raises(ValueError):
        M.upsample_depth(np.ones((10, 7)), (40, 40))
    with pytest.raises(ValueError):
        M.upsample_depth(np.ones((50, 50)), (40, 40))
