"""RD-14 w(ν)=|H|²/(|H|²+NSR), ν_obs(첫 w=0.5 교차), MTF 닫힌형, MRH.obs 거부권 캡."""
import math

import numpy as np
import pytest

import rd14_helpers as H

om = H.om


# ------------------------------------------------------------------ MTF 닫힌형
def test_defocus_and_motion_mtf_closed_forms():
    nu = np.linspace(0.0, 0.5, 101)
    for c in (0.0, 0.5, 1.0, 3.0, 8.0):
        got = om.defocus_mtf(nu, c)
        assert np.allclose(got, H.defocus_mtf(nu, c), atol=1e-12) and got[0] == 1.0
        assert (got >= 0).all() and (got <= 1 + 1e-12).all()
    for L in (0.0, 1.0, 2.5, 6.0):
        got = om.motion_mtf(nu, L)
        assert np.allclose(got, H.motion_mtf(nu, L), atol=1e-12) and got[0] == 1.0
    # 알려진 점: 원판 MTF 의 첫 0 은 c·ν = 1.21967, 모션 sinc 의 첫 0 은 L·ν = 1
    assert om.defocus_mtf(1.21966989 / 4.0, 4.0) == pytest.approx(0.0, abs=1e-6)
    assert om.motion_mtf(0.25, 4.0) == pytest.approx(0.0, abs=1e-12)
    # 방송: ν (n,1), c (m,) -> (n, m)
    assert om.defocus_mtf(nu[:, None], np.array([1.0, 2.0, 3.0])).shape == (101, 3)
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError):
            om.defocus_mtf(0.1, bad)
        with pytest.raises(ValueError):
            om.motion_mtf(0.1, bad)


def test_noise_model_formula_and_validation():
    nm = om.NoiseModel()
    assert (nm.nsr_ref, nm.nu_ref, nm.alpha, nm.jpeg_gain) == (0.01, 0.25, 2.0, 0.04)
    nu = np.array([0.0, 0.05, 0.25, 0.5])
    assert np.allclose(nm.nsr(nu), H.nsr(nu), rtol=1e-12) and nm.nsr(0.25) == pytest.approx(0.01)
    assert np.allclose(nm.nsr(nu, 3.0), H.nsr(nu, 3.0), rtol=1e-12)          # JPEG 가 NSR 을 올린다
    assert nm.nsr(0.5, 3.0) > nm.nsr(0.5, 1.0) > nm.nsr(0.25, 1.0) > nm.nsr(0.1, 1.0) > nm.nsr(0.0, 1.0) == 0.0
    for kw in (dict(nsr_ref=0.0), dict(nsr_ref=-1.0), dict(nu_ref=0.0), dict(alpha=-0.1), dict(jpeg_gain=-1.0),
               dict(nsr_ref=float("nan")), dict(alpha=float("inf"))):
        with pytest.raises(ValueError):
            om.NoiseModel(**kw)
    with pytest.raises(Exception):
        nm.nsr_ref = 1.0                                                      # frozen


# ------------------------------------------------------------------ w(ν)
def test_w_curve_matches_oracle_and_properties():
    nu = np.linspace(0.0, 0.5, 257)
    cases = [dict(), dict(coc_px=3.0), dict(motion_px=4.0), dict(r_ratio=0.5), dict(q_jpeg=3.0, coc_px=2.0),
             dict(coc_px=5.0, motion_px=2.0, r_ratio=0.75, q_jpeg=2.0)]
    for kw in cases:
        got = om.w_of_nu(nu, **kw)
        exp = H.w_full(nu, r_ratio=kw.get("r_ratio", 1.0), c=kw.get("coc_px", 0.0), m=kw.get("motion_px", 0.0),
                       qj=kw.get("q_jpeg", 1.0))
        assert np.allclose(got, exp, atol=1e-10), kw
        assert got[0] == 1.0 and (got >= 0).all() and (got <= 1).all()
    # r_eff 가 자르는 위: w = 0 (경계 ν_c 자체는 포함)
    w = om.w_of_nu(np.array([0.2499999, 0.25, 0.2500001, 0.4]), r_ratio=0.5)
    assert w[0] > 0 and w[1] > 0 and w[2] == 0 and w[3] == 0
    # 잡음이 늘면(JPEG, NSR) w 는 점별로 줄어든다
    nu2 = np.linspace(0.01, 0.5, 60)
    assert (om.w_of_nu(nu2, q_jpeg=4.0) <= om.w_of_nu(nu2, q_jpeg=1.0) + 1e-12).all()
    assert (om.w_of_nu(nu2, coc_px=2.0, q_jpeg=3.0) <= om.w_of_nu(nu2, coc_px=2.0) + 1e-12).all()
    big = om.NoiseModel(nsr_ref=0.5)
    assert (om.w_of_nu(nu2, coc_px=2.0, noise=big) <= om.w_of_nu(nu2, coc_px=2.0) + 1e-12).all()
    # 깨끗한 영상은 나이퀴스트까지 신뢰: w(0.5) > 0.9 (기본 잡음)
    assert float(om.w_of_nu(0.5)) > 0.9
    # 방송
    assert om.w_of_nu(nu[:, None], coc_px=np.array([0.0, 2.0, 4.0])).shape == (257, 3)
    for bad in (-0.01, 0.5001, float("nan")):
        with pytest.raises(ValueError):
            om.w_of_nu(bad)


# ------------------------------------------------------------------ ν_obs
def test_nu_obs_matches_fine_grid_oracle_sweep():
    rng = np.random.default_rng(141)
    worst = 0.0
    for _ in range(60):
        c = float(rng.choice([0.0, rng.uniform(0, 9)]))
        m = float(rng.choice([0.0, rng.uniform(0, 9)]))
        q = float(rng.uniform(1, 4))
        r = float(rng.choice([1.0, rng.uniform(1 / 16, 1)]))
        got = om.nu_obs_cutoff(r_ratio=r, coc_px=c, motion_px=m, q_jpeg=q)
        exp = H.nu_obs_oracle(r, c, m, q)
        worst = max(worst, abs(got - exp))
        assert isinstance(got, float)
        assert abs(got - exp) <= 1e-3, (r, c, m, q, got, exp)
    assert worst <= 1e-3


def test_nu_obs_reference_points():
    # 해석 지식: 깨끗하면 Nyquist, r_eff 만 있으면 정확히 0.5·ratio(선형 보간 없이)
    assert om.nu_obs_cutoff() == 0.5
    for r in (1.0, 0.75, 0.5, 0.25, 1 / 16):
        assert om.nu_obs_cutoff(r_ratio=r) == pytest.approx(0.5 * r, abs=1e-12)
    # 프로토타입 값(기본 잡음): c=3 -> ~0.348, c=4 -> ~0.270, c=8 -> ~0.143, m=4 -> ~0.229
    for kw, exp in (({"coc_px": 3.0}, 0.348), ({"coc_px": 4.0}, 0.270), ({"coc_px": 8.0}, 0.143),
                    ({"motion_px": 4.0}, 0.229)):
        assert om.nu_obs_cutoff(**kw) == pytest.approx(exp, abs=2e-3)
    # 상한: ν_obs ≤ min(0.5·ratio, 0.5)
    assert om.nu_obs_cutoff(r_ratio=0.1, coc_px=1.0) <= 0.05 + 1e-12


def test_nu_obs_monotone_and_broadcast():
    cs = np.linspace(0.0, 10.0, 41)
    v = om.nu_obs_cutoff(coc_px=cs)
    assert v.shape == cs.shape and (np.diff(v) <= 1e-9).all() and v[0] == 0.5 and v[-1] < v[len(v) // 2]
    ms_ = np.linspace(0.0, 10.0, 41)
    vm = om.nu_obs_cutoff(motion_px=ms_)
    assert (np.diff(vm) <= 1e-9).all()
    grid = om.nu_obs_cutoff(coc_px=cs[:, None], motion_px=ms_[None, :])
    assert grid.shape == (41, 41) and (np.diff(grid, axis=0) <= 1e-9).all() and (np.diff(grid, axis=1) <= 1e-9).all()
    assert grid[20, 0] == pytest.approx(om.nu_obs_cutoff(coc_px=cs[20]), abs=1e-12)
    # 잡음이 세면 더 일찍 신뢰를 잃는다 / JPEG 도
    base = om.nu_obs_cutoff(coc_px=3.0)
    assert om.nu_obs_cutoff(coc_px=3.0, noise=om.NoiseModel(nsr_ref=0.3)) < base
    assert om.nu_obs_cutoff(coc_px=3.0, q_jpeg=4.0) <= base + 1e-12


def test_w_is_half_exactly_at_nu_obs_when_defocus_limited():
    for c in (3.0, 5.0, 7.0):
        nu = om.nu_obs_cutoff(coc_px=c)
        assert float(om.w_of_nu(nu, coc_px=c)) == pytest.approx(0.5, abs=2e-3)
        assert float(om.w_of_nu(nu * 0.95, coc_px=c)) > 0.5 > float(om.w_of_nu(min(0.5, nu * 1.05), coc_px=c))


# ------------------------------------------------------------------ MRH.obs 캡 (소프트 의존: 거부권만)
def test_mrh_nu_cap_chain_rule():
    tau = 0.5
    combos = [(0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 1, 1), (1, 1, 0, 1),
              (0, 1, 0, 1), (1, 1, 1, 0), (0, 0, 0, 0)]
    obs = np.zeros((4, len(combos), 1), dtype=np.float32)
    for k, flags in enumerate(combos):
        for b in range(4):
            obs[b, k, 0] = 0.9 if flags[b] else 0.1
    cap = om.mrh_nu_cap(obs, tau=tau)
    assert cap.shape == (len(combos), 1)
    for k, flags in enumerate(combos):
        assert cap[k, 0] == pytest.approx(H.mrh_cap_oracle([bool(f) for f in flags])), flags
    # 가장 낮은 대역도 관측이 아니면 1/32
    assert cap[0, 0] == pytest.approx(1 / 32) and cap[1, 0] == 0.5 and cap[2, 0] == 0.25
    # 임계는 엄격 부등식 (p > tau)
    edge = np.full((4, 1, 1), tau, dtype=np.float32)
    assert om.mrh_nu_cap(edge, tau=tau)[0, 0] == pytest.approx(1 / 32)
    edge2 = np.full((4, 1, 1), np.nextafter(np.float32(tau), np.float32(1)), dtype=np.float32)
    assert om.mrh_nu_cap(edge2, tau=tau)[0, 0] == 0.5


@pytest.mark.parametrize("bad", [
    np.zeros((3, 4, 4)), np.zeros((4, 4)), np.full((4, 2, 2), 1.5), np.full((4, 2, 2), -0.1),
    np.full((4, 2, 2), np.nan), np.zeros((4, 0, 2)),
])
def test_mrh_nu_cap_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        om.mrh_nu_cap(bad)
    with pytest.raises(TypeError):
        om.mrh_nu_cap("nope")


def test_oct_center_constants():
    assert tuple(om.OCT_CENTER) == pytest.approx(H.OCT_CENTER, rel=1e-12)
    assert om.NU_GRID_N == 512 and om.W_TRUST == 0.5
