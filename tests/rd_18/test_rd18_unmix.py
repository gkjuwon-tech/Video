"""RD-18 종단: edge_unmix 정확도(원판·stride 4·털 가닥), 불변식, 등변, 결정론, meta, 검증, 속도, 격리."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import rd18_helpers as H



def _eu():
    from buc.read import edge_unmix as M
    return M


def test_disc_alpha_fg_bg_and_reconstruction():
    M = _eu()
    img, z, at, Ft, Bt = H.disc_scene()
    img32 = img.astype(np.float32)
    r = M.edge_unmix(img32, z)
    assert r.alpha.dtype == np.float32 and r.alpha.shape == (96, 96)
    assert r.fg.dtype == np.float32 and r.fg.shape == (96, 96, 3) and r.bg.shape == (96, 96, 3)
    assert r.trimap.dtype == np.uint8 and r.band.dtype == bool and r.edges.dtype == bool
    assert r.depth_stride == 1 and r.n_unknown == int(r.band.sum()) > 0
    assert np.array_equal(r.band, r.trimap == M.UNKNOWN)
    assert 0.0 <= r.alpha.min() and r.alpha.max() <= 1.0
    U = r.band
    err = np.abs(r.alpha.astype(np.float64) - at)
    assert err[U].mean() <= 0.01 and err.max() <= 0.05
    m = U & (at > 0.1)
    f_err = np.abs(r.fg.astype(np.float64) - Ft).max(-1)[m].mean()
    naive = np.abs(img - Ft).max(-1)[m].mean()          # 혼합 화소를 전경에 몰아넣은 경우
    assert f_err <= 0.01 and naive >= 0.05
    m2 = U & (at < 0.9)
    assert np.abs(r.bg.astype(np.float64) - Bt).max(-1)[m2].mean() <= 0.01
    rec = H.composite(r.alpha, r.fg, r.bg)
    assert np.abs(rec - img32)[U].max() <= 2e-3
    assert r.recon_err == pytest.approx(np.abs(rec - img32.astype(np.float64))[U].max(), abs=1e-6)
    # 띠 밖: α ∈ {0,1}, 합성이 입력과 비트 동일
    out = ~U
    assert np.isin(r.alpha[out], (0.0, 1.0)).all()
    a = r.alpha[..., None]
    rec32 = a * r.fg + (1 - a) * r.bg
    assert np.array_equal(rec32[out], img32[out])
    assert not r.low_conf and r.clip_frac <= 0.05


def test_stride4_depth_like_mrh_output():
    M = _eu()
    img, z, at, _, _ = H.disc_scene()
    zc = z[2::4, 2::4]
    assert zc.shape == (24, 24)
    r = M.edge_unmix(img, zc)
    assert r.depth_stride == 4
    assert np.abs(r.alpha.astype(np.float64) - at).mean() <= 0.005
    assert np.abs(r.alpha.astype(np.float64) - at).max() <= 0.05


def test_fur_strands_need_thin_mask_with_coarse_depth():
    M = _eu()
    img, z, zc, at, Ft, Bt, thin, xx = H.fur_scene()
    m = (at > 0.3) & (xx > 36)
    assert m.sum() > 100
    r0 = M.edge_unmix(img, zc)                         # 거친 깊이에는 가닥이 없다
    r1 = M.edge_unmix(img, zc, thin=thin)
    e0 = np.abs(r0.alpha.astype(np.float64) - at)[m].mean()
    e1 = np.abs(r1.alpha.astype(np.float64) - at)[m].mean()
    assert e0 >= 0.3
    assert e1 <= 0.01
    assert np.array_equal(r1.thin, thin) and not r0.thin.any()
    assert r1.band[thin].all()
    f_err = np.abs(r1.fg.astype(np.float64) - Ft).max(-1)[m].mean()
    assert f_err <= 0.02 and np.abs(img - Ft).max(-1)[m].mean() >= 0.1
    assert r1.meta()["thin_frac"] == pytest.approx(thin.mean(), abs=1e-9)


def test_full_res_fur_depth_without_thin():
    M = _eu()
    img, z, zc, at, *_ = H.fur_scene()
    r = M.edge_unmix(img, z)
    assert np.abs(r.alpha.astype(np.float64) - at)[r.band].mean() <= 0.01


def test_mirror_equivariance():
    M = _eu()
    img, z, at = H.step_scene()
    r = M.edge_unmix(img, z)
    rf = M.edge_unmix(img[:, ::-1].copy(), z[:, ::-1].copy())
    np.testing.assert_array_equal(rf.trimap, r.trimap[:, ::-1])
    np.testing.assert_allclose(rf.alpha, r.alpha[:, ::-1], atol=1e-5)
    np.testing.assert_allclose(rf.fg, r.fg[:, ::-1], atol=1e-5)
    rv = M.edge_unmix(img[::-1].copy(), z[::-1].copy())
    np.testing.assert_allclose(rv.alpha, r.alpha[::-1], atol=1e-5)
    assert np.abs(r.alpha.astype(np.float64) - at).mean() <= 0.01


def test_determinism_input_immutability_rng_and_meta():
    M = _eu()
    img, z, at, _, _ = H.disc_scene(64, 64, r=18)
    i0, z0 = img.copy(), z.copy()
    st = np.random.get_state()
    a = M.edge_unmix(img, z)
    b = M.edge_unmix(img, z)
    for k in ("alpha", "alpha_raw", "fg", "bg", "trimap", "band", "edges"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert np.array_equal(img, i0) and np.array_equal(z, z0)
    s2 = np.random.get_state()
    assert st[0] == s2[0] and np.array_equal(st[1], s2[1]) and st[2:] == s2[2:]
    meta = a.meta()
    js = json.dumps(meta, allow_nan=False)
    assert json.loads(js) == meta
    for key in ("n_unknown", "band_frac", "clip_frac", "recon_err", "low_conf", "depth_stride",
                "thin_frac", "rel_jump", "band_px", "eps", "win_r"):
        assert key in meta, key
    assert meta["n_unknown"] == a.n_unknown and meta["band_frac"] == pytest.approx(a.band.mean())
    raw = a.alpha_raw.astype(np.float64)[a.band]
    assert a.clip_frac == pytest.approx(((raw < -0.05) | (raw > 1.05)).mean(), abs=1e-9)
    cfg = M.UnmixConfig()
    assert a.low_conf == (a.clip_frac > cfg.clip_tau or a.recon_err > cfg.recon_tau)


def test_config_band_px_changes_band_and_no_edges_case():
    M = _eu()
    img, z, at, _, _ = H.disc_scene(64, 64, r=18)
    r3 = M.edge_unmix(img, z)
    r5 = M.edge_unmix(img, z, cfg=M.UnmixConfig(band_px=5))
    assert r5.n_unknown > r3.n_unknown and r5.meta()["band_px"] == 5
    flat = M.edge_unmix(img, np.full((64, 64), 4.0))
    assert flat.n_unknown == 0 and (flat.alpha == 1).all() and not flat.edges.any()
    assert flat.clip_frac == 0.0 and flat.recon_err == 0.0
    assert np.array_equal(flat.fg, img.astype(np.float32))


def test_validation():
    M = _eu()
    img, z, *_ = H.disc_scene(32, 32, r=10)
    bad = img.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.edge_unmix(bad, z)
    with pytest.raises(ValueError):
        M.edge_unmix(-img, z)
    with pytest.raises(ValueError):
        M.edge_unmix(img[..., :2], z)
    with pytest.raises(ValueError):
        M.edge_unmix(img, z[:, :-3])
    with pytest.raises(ValueError):
        M.edge_unmix(img[:2, :2], z[:2, :2])
    with pytest.raises(TypeError):
        M.edge_unmix(img.astype(np.complex128), z)
    with pytest.raises(TypeError):
        M.edge_unmix(img, z, thin=np.zeros((32, 32), np.float32))
    with pytest.raises(ValueError):
        M.edge_unmix(img, z, thin=np.zeros((31, 32), bool))
    allthin = np.ones((32, 32), bool)
    with pytest.raises(M.EdgeUnmixError):
        M.edge_unmix(img, z, thin=allthin)


def test_speed_512():
    M = _eu()
    img, z, at, _, _ = H.disc_scene(512, 512, r=170)
    rng = np.random.default_rng(11)
    for _ in range(25):                                     # 작은 전경 조각 여럿(띠를 늘린다)
        cy, cx, rr = rng.integers(20, 492), rng.integers(20, 492), rng.uniform(4, 12)
        yy, xx = np.mgrid[0:512, 0:512]
        d = np.hypot(yy - cy, xx - cx) - rr
        z = np.where(d < 0, 1.5, z)
    t0 = time.perf_counter()
    r = M.edge_unmix(img.astype(np.float32), z)
    dt = time.perf_counter() - t0
    assert r.n_unknown > 8000
    assert dt < 15.0, dt


def test_import_isolation_no_torch():
    import buc
    root = os.path.dirname(os.path.dirname(os.path.abspath(buc.__file__)))
    code = ("import sys; import buc.read.edge_unmix as M; "
            "assert 'torch' not in sys.modules, 'torch loaded'; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr
