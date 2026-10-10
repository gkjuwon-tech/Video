"""RD-05 c(z) 회귀: 복원·강건성·불변식·판정·σ 전파·직렬화."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

import rd05_helpers as H
from buc.read import blur_est as B


def fit(c, z, **kw):
    return B.estimate_defocus(c, z, f_px=H.F_PX, width_px=H.W, **kw)


@pytest.mark.parametrize("zf,n", [(3.0, 2.0), (1.2, 1.4), (10.0, 4.0), (2.0, 8.0)])
def test_noise_free_recovery(zf, n):
    c, z = H.scene(zf, n, seed=1)
    r = fit(c, z)
    assert r.status == "fit"
    assert abs(r.z_f_m / zf - 1) <= 1e-3
    assert abs(r.n_eq / n - 1) <= 1e-3
    assert abs(r.f_ff_mm - 1000 * H.F_FF) <= 1e-6 * 1000 * H.F_FF
    assert r.inlier_frac >= 0.99 and r.rel_resid <= 1e-3
    assert r.reliable and r.warnings == ()
    assert r.sigma_log_zf <= 0.01 and r.sigma_log_n <= 0.01


@pytest.mark.parametrize("zf,n,seed", [(3.0, 2.0, 2), (1.2, 1.4, 3), (10.0, 4.0, 4), (2.0, 8.0, 5), (0.8, 2.8, 6)])
def test_noisy_with_outliers_within_spec(zf, n, seed):
    c, z = H.scene(zf, n, seed=seed, noise=0.05, outlier=0.10)
    r = fit(c, z)
    assert r.status == "fit"
    assert abs(r.z_f_m / zf - 1) <= 0.10
    assert abs(H.stops(r.n_eq, n)) <= 0.5
    assert r.inlier_frac >= 0.7 and "depth_inconsistent" not in r.warnings
    assert 0.0 < r.sigma_log_zf < 0.2


def test_focus_at_infinity():
    c, z = H.scene(math.inf, 2.8, seed=7, noise=0.03)
    r = fit(c, z)
    assert r.status == "fit"
    assert r.u_f <= 0.01
    assert (r.z_f_m == math.inf) == (r.u_f <= B.U_INF)
    if r.z_f_m == math.inf:
        assert "focus_at_infinity" in r.warnings
    assert abs(H.stops(r.n_eq, 2.8)) <= 0.5


def test_one_sided_depth_behind_focus():
    # 모든 점이 초점 뒤(z>z_f): V의 한쪽 가지만 보인다. 절편이 u_f를 정한다.
    c, z = H.scene(1.0, 2.0, seed=8, zmin=1.5, zmax=40.0)
    r = fit(c, z)
    assert abs(r.z_f_m / 1.0 - 1) <= 0.02 and abs(H.stops(r.n_eq, 2.0)) <= 0.1


def test_poppy_foreground_rerender_matches_input_bokeh():
    """D15: 전경 양귀비(0.6 m, 크게 녹음) + 피사체 3 m(선명) + 배경 25 m. 판독 θ로 다시 만든 CoC가 입력과 맞고,
    랙 포커스(초점을 0.6 m로)하면 양귀비가 선명해진다."""
    rng = np.random.default_rng(11)
    z = np.empty((90, 120))
    z[:, :40] = 0.6 * np.exp(0.05 * rng.standard_normal((90, 40)))
    z[:, 40:80] = 3.0 * np.exp(0.05 * rng.standard_normal((90, 40)))
    z[:, 80:] = 25.0 * np.exp(0.2 * rng.standard_normal((90, 40)))
    c = H.oracle_coc(z, 1.8, 3.0) * np.clip(1 + 0.03 * rng.standard_normal(z.shape), 0, None)
    r = fit(c, z)
    assert abs(r.z_f_m / 3.0 - 1) <= 0.10 and abs(H.stops(r.n_eq, 1.8)) <= 0.5
    th = r.theta_fields()
    poppy = z[:, :40]
    re = H.coc_px(poppy, focal_mm=th["focal_mm"], f_stop=th["f_stop"], focus_m=th["focus_m"],
                  sensor_width_mm=th["sensor_width_mm"], width_px=H.W)
    truth = H.oracle_coc(poppy, 1.8, 3.0)
    assert np.median(np.abs(re / truth - 1)) <= 0.10
    assert truth.mean() > 20.0
    racked = H.coc_px(np.array([0.6]), focal_mm=th["focal_mm"], f_stop=th["f_stop"], focus_m=0.6,
                      sensor_width_mm=36.0, width_px=H.W)
    assert racked[0] < 0.5
    # 예측 맵은 전경과 배경에서 부호가 갈린다
    sp = r.coc_pred(np.array([0.6, 25.0]), signed=True)
    assert sp[0] < 0 < sp[1]


def test_coc_pred_and_theta_fields_roundtrip_lns02():
    c, z = H.scene(4.0, 2.8, seed=12, noise=0.02)
    r = fit(c, z)
    th = r.theta_fields()
    assert set(th) >= {"focal_mm", "sensor_width_mm", "f_stop", "focus_m"} and th["sensor_width_mm"] == 36.0
    zz = np.geomspace(0.3, 100, 50)
    ref = H.coc_px(zz, focal_mm=th["focal_mm"], f_stop=th["f_stop"], focus_m=th["focus_m"],
                   sensor_width_mm=36.0, width_px=H.W).astype(np.float64)
    pred = r.coc_pred(zz)
    assert pred.dtype == np.float64
    assert np.allclose(pred, ref, rtol=1e-4, atol=1e-4)
    # 오라클 수식: D|1/z - u_f|
    assert np.allclose(pred, r.d_px_m * np.abs(1 / zz - r.u_f), rtol=1e-9)
    s = r.coc_pred(zz, signed=True)
    assert np.allclose(np.abs(s), pred) and np.all(np.sign(s[zz > r.z_f_m * 1.01]) > 0)
    assert np.allclose(r.residual(c, z), c - r.coc_pred(z), atol=1e-9)
    # N_eq 정의식
    n_def = H.F_FF ** 2 * (H.W / 0.036) / ((1 - H.F_FF * r.u_f) * r.d_px_m)
    assert abs(r.n_eq / n_def - 1) <= 1e-9


def test_scale_equivariance():
    c, z = H.scene(3.0, 2.0, seed=13, noise=0.03)
    a = fit(c, z)
    b = fit(c, 2.5 * z)
    assert abs(b.z_f_m / (2.5 * a.z_f_m) - 1) <= 1e-4
    assert abs(b.d_px_m / (2.5 * a.d_px_m) - 1) <= 1e-4
    # 미니어처 신호: 장면이 커지면 같은 블러에 필요한 N은 작아진다
    assert b.n_eq < a.n_eq


def test_zero_weight_points_ignored_and_permutation_invariant():
    c, z = H.scene(3.0, 2.0, seed=14, noise=0.03, shape=(60, 80))
    w = np.ones_like(c)
    w[:, :20] = 0.0
    c2 = c.copy()
    c2[:, :20] = 500.0
    z2 = z.copy()
    z2[:, :20] = 0.01
    a = fit(c, z, weight=w)
    b = fit(c2, z2, weight=w)
    assert abs(a.z_f_m - b.z_f_m) <= 1e-9 * a.z_f_m and abs(a.n_eq - b.n_eq) <= 1e-9 * a.n_eq
    assert a.n_used == b.n_used == 60 * 60
    p = np.random.default_rng(0).permutation(c.size)
    d = fit(c.ravel()[p].reshape(c.shape), z.ravel()[p].reshape(z.shape), weight=w.ravel()[p].reshape(w.shape))
    assert abs(d.z_f_m / a.z_f_m - 1) <= 1e-6 and abs(d.n_eq / a.n_eq - 1) <= 1e-6


def test_pan_focus_and_insufficient():
    c, z = H.scene(3.0, 22.0, seed=15, zmin=2.0, zmax=6.0)
    assert np.percentile(c, 95) < 1.0
    r = fit(c, z)
    assert r.status == "pan_focus" and "pan_focus" in r.warnings and not r.reliable
    assert r.z_f_m is None and r.n_eq == 11.0 and r.d_px_m == 0.0
    assert np.all(r.coc_pred(np.array([1.0, 50.0])) == 0)
    th = r.theta_fields()
    f = H.F_FF * 1000
    assert abs(th["focus_m"] - (f * f / (11.0 * 0.03) + f) / 1000) <= 1e-6
    json.dumps(r.to_meta(), allow_nan=False)
    # 점 부족
    c, z = H.scene(3.0, 2.0, seed=16, shape=(5, 5))
    r = fit(c, z)
    assert r.status == "insufficient" and "insufficient" in r.warnings and not r.reliable
    # 깊이 폭 부족(모든 점이 같은 깊이 근처)
    c, z = H.scene(3.0, 2.0, seed=17, zmin=8.0, zmax=8.2)
    r = fit(c, z)
    assert r.status == "insufficient"


def test_depth_inconsistent_bokeh_flagged():
    """AI 이미지: 배경 일부만 녹고 나머지는 깊이와 무관하게 선명(F-M7)."""
    c, z = H.scene(2.0, 1.8, seed=18, zmin=0.5, zmax=50.0)
    rng = np.random.default_rng(18)
    paint = (rng.random(c.shape) < 0.45) & (c >= 5.0)
    c2 = np.where(paint, 0.0, c)
    r = fit(c2, z)
    assert "depth_inconsistent" in r.warnings and not r.reliable
    assert r.inlier_frac < 0.7
    clean = fit(c, z)
    assert clean.inlier_frac > 0.95 and "depth_inconsistent" not in clean.warnings


def test_sigma_propagation_and_noise_monotone():
    c, z = H.scene(3.0, 2.0, seed=19)
    r0 = fit(c, z, sigma_s=0.2)
    assert 0.2 <= r0.sigma_log_zf <= 0.25 and 0.2 <= r0.sigma_log_n <= 0.25 and r0.reliable
    r1 = fit(c, z, sigma_s=0.5)
    assert "scale_unreliable" in r1.warnings and not r1.reliable and r1.sigma_log_zf >= 0.5
    lo = fit(*H.scene(3.0, 2.0, seed=20, noise=0.03))
    hi = fit(*H.scene(3.0, 2.0, seed=20, noise=0.30))
    assert hi.sigma_log_zf > lo.sigma_log_zf and hi.sigma_log_n > lo.sigma_log_n


def test_n_out_of_range_warning():
    c, z = H.scene(3.0, 0.8, seed=21, noise=0.02)
    r = fit(c, z)
    assert "n_out_of_range" in r.warnings


def test_torch_input_accepted():
    torch = pytest.importorskip("torch")
    c, z = H.scene(3.0, 2.0, seed=22)
    a = fit(c, z)
    b = fit(torch.from_numpy(c.astype(np.float32))[None, None], torch.from_numpy(z.astype(np.float32))[None, None])
    assert abs(b.z_f_m / a.z_f_m - 1) <= 1e-3 and abs(b.n_eq / a.n_eq - 1) <= 1e-3


def test_determinism_json_and_input_immutability():
    c, z = H.scene(3.0, 2.0, seed=23, noise=0.05, outlier=0.1)
    w = np.ones_like(c)
    cc, zc, wc = c.copy(), z.copy(), w.copy()
    a = fit(c, z, weight=w, sigma_s=0.1)
    b = fit(c, z, weight=w, sigma_s=0.1)
    assert np.array_equal(c, cc) and np.array_equal(z, zc) and np.array_equal(w, wc)
    ma, mb = a.to_meta(), b.to_meta()
    s = json.dumps(ma, allow_nan=False, sort_keys=True)
    assert s == json.dumps(mb, allow_nan=False, sort_keys=True)
    assert set(ma) >= {"status", "z_f_m", "n_eq", "f_ff_mm", "d_px_m", "u_f", "sigma_log_zf", "sigma_log_n",
                       "rel_resid", "inlier_frac", "n_used", "reliable", "warnings"}
    assert list(a.warnings) == sorted(set(a.warnings))
    inf = fit(*H.scene(math.inf, 2.0, seed=24))
    json.dumps(inf.to_meta(), allow_nan=False)


@pytest.mark.parametrize("bad", ["nan_c", "neg_c", "zero_z", "nan_z", "shape", "neg_w", "fpx", "width", "sigma", "ndim"])
def test_validation_errors(bad):
    c, z = H.scene(3.0, 2.0, seed=25, shape=(20, 20))
    kw = dict(f_px=H.F_PX, width_px=H.W)
    w = None
    if bad == "nan_c":
        c[0, 0] = np.nan
    elif bad == "neg_c":
        c[0, 0] = -1
    elif bad == "zero_z":
        z[0, 0] = 0
    elif bad == "nan_z":
        z[0, 0] = np.nan
    elif bad == "shape":
        z = z[:, :10]
    elif bad == "neg_w":
        w = np.ones_like(c)
        w[0, 0] = -1
    elif bad == "fpx":
        kw["f_px"] = 0.0
    elif bad == "width":
        kw["width_px"] = True
    elif bad == "sigma":
        kw["sigma_s"] = -0.1
    elif bad == "ndim":
        c, z = c.ravel(), z.ravel()
    with pytest.raises(ValueError):
        B.estimate_defocus(c, z, weight=w, **kw)


def test_inf_depth_allowed_and_opts_validation():
    c, z = H.scene(3.0, 2.0, seed=26)
    z = z.copy()
    z[:3, :] = np.inf
    c = c.copy()
    c[:3, :] = H.oracle_coc(np.array([np.inf]), 2.0, 3.0)[0]
    r = fit(c, z)
    assert abs(r.z_f_m / 3.0 - 1) <= 1e-3
    for kw in ({"tile_px": 8}, {"pan_focus_px": -1.0}, {"n_range": (32.0, 1.0)}, {"min_inlier_frac": float("nan")}):
        with pytest.raises(ValueError):
            B.BlurOpts(**kw)


def test_import_does_not_load_torch():
    code = "import sys; import buc.read.blur_est; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=H.__file__.rsplit("/tests/", 1)[0])
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
