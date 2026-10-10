"""RD-18 α 풀이(하드 제약)와 F/B 분리(2차 에너지)."""
import numpy as np
import pytest

import rd18_helpers as H


def _eu():
    from buc.read import edge_unmix as M
    return M


def _small(seed=0):
    img, z, at = H.step_scene(h=10, w=14, col=6)
    img = img + np.random.default_rng(seed).normal(0, 0.01, img.shape)
    img = np.clip(img, 0, None)
    return img, z, at


def test_solve_alpha_matches_dense_full_laplacian():
    M = _eu()
    img, z, _ = _small()
    tri = M.make_trimap(z)
    assert (tri == M.UNKNOWN).any()
    a, raw = M.solve_alpha(img, tri)
    ref = H.dense_alpha(img, tri)
    np.testing.assert_allclose(raw, ref, atol=1e-6)
    np.testing.assert_array_equal(a, np.clip(raw, 0, 1))
    K = tri != M.UNKNOWN
    assert np.array_equal(raw[K], (tri[K] == M.FG).astype(np.float64))   # 알려진 화소는 정확히


def test_solve_alpha_win_r2_and_eps():
    M = _eu()
    img, z, _ = _small(1)
    tri = M.make_trimap(z)
    _, raw = M.solve_alpha(img, tri, eps=1e-4, win_r=2)
    np.testing.assert_allclose(raw, H.dense_alpha(img, tri, eps=1e-4, r=2), atol=1e-6)


def test_solve_alpha_edge_cases():
    M = _eu()
    img, z, _ = _small()
    tri = np.full(z.shape, M.FG, np.uint8); tri[:, 9:] = M.BG
    a, raw = M.solve_alpha(img, tri)
    assert np.array_equal(a, (tri == M.FG).astype(np.float64))
    with pytest.raises(M.EdgeUnmixError):
        M.solve_alpha(img, np.full(z.shape, M.UNKNOWN, np.uint8))
    bad = tri.copy(); bad[0, 0] = 7
    with pytest.raises(ValueError):
        M.solve_alpha(img, bad)


def test_fg_bg_is_energy_stationary_point():
    M = _eu()
    img, z, at = _small(2)
    tri = M.make_trimap(z)
    F, B = M.solve_fg_bg(img, at, tri)
    assert F.shape == img.shape and B.shape == img.shape and (F >= 0).all() and (B >= 0).all()
    om = H.omega(tri)
    fgk, bgk = tri == M.FG, tri == M.BG
    # 고정값·Ω 밖 규칙
    assert np.array_equal(F[fgk], img[fgk]) and np.array_equal(B[bgk], img[bgk])
    assert (F[~om & ~fgk] == 0).all() and (B[~om & ~bgk] == 0).all()
    assert (F[om & ~fgk] > 1e-3).all() and (B[om & ~bgk] > 1e-3).all()   # 클램프가 걸리지 않는 장면
    e0 = H.fb_energy(img, at, tri, F, B)
    rng = np.random.default_rng(7)
    for _ in range(4):
        hF = np.where((om & ~fgk)[..., None], rng.normal(0, 1, img.shape), 0) * 1e-3
        hB = np.where((om & ~bgk)[..., None], rng.normal(0, 1, img.shape), 0) * 1e-3
        ep = H.fb_energy(img, at, tri, F + hF, B + hB)
        em = H.fb_energy(img, at, tri, F - hF, B - hB)
        curv = ep + em - 2 * e0
        assert curv > 0
        assert abs(ep - em) <= 1e-4 * curv + 1e-12     # 기울기 ≈ 0
        assert ep >= e0 - 1e-12 and em >= e0 - 1e-12


def test_fg_bg_recovers_constant_colors_with_true_alpha():
    M = _eu()
    h, w = 20, 24
    xx = np.arange(w)[None, :] * np.ones((h, 1))
    at = np.clip(0.5 - (xx - 11.5) / 3.0, 0, 1)
    Ft, Bt = np.array([0.7, 0.2, 0.5]), np.array([0.1, 0.6, 0.3])
    img = at[..., None] * Ft + (1 - at[..., None]) * Bt
    z = np.where(xx <= 11, 2.0, 5.0)
    tri = M.make_trimap(z)
    F, B = M.solve_fg_bg(img, at, tri)
    U = tri == M.UNKNOWN
    np.testing.assert_allclose(F[U & (at > 0.05)], np.broadcast_to(Ft, F[U & (at > 0.05)].shape), atol=1e-4)
    np.testing.assert_allclose(B[U & (at < 0.95)], np.broadcast_to(Bt, B[U & (at < 0.95)].shape), atol=1e-4)
    with pytest.raises(ValueError):
        M.solve_fg_bg(img, at * 1.5, tri)
