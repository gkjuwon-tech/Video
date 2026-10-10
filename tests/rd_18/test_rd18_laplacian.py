"""RD-18 Levin 매팅 라플라시안: 조밀 오라클, 대칭·L1=0·PSD, color-line 에너지."""
import numpy as np
import scipy.sparse as sp

import rd18_helpers as H


def _eu():
    from buc.read import edge_unmix as M
    return M


def _img(h, w, seed):
    return np.random.default_rng(seed).uniform(0.0, 1.0, (h, w, 3))


def test_laplacian_matches_dense_oracle():
    M = _eu()
    img = _img(6, 7, 0)
    L = M.matting_laplacian(img)
    assert sp.issparse(L) and L.shape == (42, 42) and L.dtype == np.float64
    np.testing.assert_allclose(L.toarray(), H.dense_laplacian(img), rtol=0, atol=1e-10)


def test_laplacian_oracle_win_r2_eps_and_centers():
    M = _eu()
    img = _img(7, 8, 1)
    L = M.matting_laplacian(img, eps=1e-3, win_r=2)
    np.testing.assert_allclose(L.toarray(), H.dense_laplacian(img, 1e-3, 2), atol=1e-10)
    cen = np.zeros((7, 8), bool); cen[2:4, 3:7] = True; cen[0, 0] = True   # 경계 밖 중심은 무시
    L = M.matting_laplacian(img, centers=cen)
    np.testing.assert_allclose(L.toarray(), H.dense_laplacian(img, centers=cen), atol=1e-10)


def test_laplacian_symmetric_zero_rowsum_psd():
    M = _eu()
    img = _img(9, 8, 2)
    L = M.matting_laplacian(img).toarray()
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    np.testing.assert_allclose(L @ np.ones(72), 0.0, atol=1e-9)
    ev = np.linalg.eigvalsh(L)
    assert ev.min() > -1e-8


def test_color_line_alpha_has_near_zero_energy():
    M = _eu()
    img = _img(16, 16, 4)
    L = M.matting_laplacian(img)
    a = (0.3 * img[..., 0] + 0.5 * img[..., 1] - 0.2 * img[..., 2] + 0.1).ravel()
    e_lin = a @ (L @ a) / (a @ a)
    rnd = np.random.default_rng(5).uniform(0, 1, a.size)
    e_rnd = rnd @ (L @ rnd) / (rnd @ rnd)
    assert e_lin < 1e-5
    assert e_rnd > 1e3 * e_lin
