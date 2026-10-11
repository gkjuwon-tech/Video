"""RD-14 §2.10 식: φ_in, φ_eff, l_obs 경/부분, 소유 판정, PARADIGM 표(1K–8K)."""
import math

import numpy as np
import pytest

import rd14_helpers as H

om = H.om


# ------------------------------------------------------------------ 상수·격자 이음매
def test_constants_and_texel_pitch():
    assert om.LEVEL_RANGE == (-8, 24) and om.H0_M == 64.0 and om.BLOCK_CELLS == 16
    assert tuple(om.OCT_HI) == H.OCT_HI and om.NO_LEVEL == H.NO_LEVEL
    assert (om.OWN_INPUT, om.OWN_INPUT_PART, om.OWN_INPUT_REFL) == (1, 2, 3)
    for l in range(-8, 25):
        assert om.texel_pitch_m(l) == 8.0 * 2.0 ** (-l)          # 정확(2의 거듭제곱)
        assert om.cell_size_m(l) == 64.0 * 2.0 ** (-l)
        assert om.block_extent_m(l) == 16 * om.cell_size_m(l)
    for bad in (-9, 25):
        with pytest.raises(ValueError):
            om.texel_pitch_m(bad)


@pytest.mark.parametrize("W,exp_phi_mm,exp_levels", [
    (1024, 3.4, (10, 11)), (2048, 1.7, (11, 12)), (3840, 0.9, (12, 13)), (7680, 0.45, (13, 14)),
])
def test_paradigm_table_1k_to_8k(W, exp_phi_mm, exp_levels):
    """PARADIGM §2.10 표: 60° 화각, 정면, z = 3 m. φ_in 은 3 자리 근사, 레벨은 정확히 일치."""
    f = H.f_for(W, 60.0)
    phi = om.footprint_m(3.0, f)
    assert abs(phi * 1000 - exp_phi_mm) / exp_phi_mm < 0.02
    hard, part = om.levels_from_footprint(phi)
    assert (int(hard), int(part)) == exp_levels
    # 텍셀 피치: 경 / 부분 (표의 7.8/3.9 mm, 3.9/1.95 mm ...)
    assert om.texel_pitch_m(int(hard)) >= 2 * phi and om.texel_pitch_m(int(part)) >= phi
    assert om.texel_pitch_m(int(hard) + 1) < 2 * phi and om.texel_pitch_m(int(part) + 1) < phi


# ------------------------------------------------------------------ φ_in
def test_footprint_formula_and_scale_invariance():
    z = np.array([[1.0, 2.0], [3.0, 6.0]])
    out = om.footprint_m(z, 500.0)
    assert out.shape == z.shape and np.allclose(out, z / 500.0, rtol=1e-12)
    cos = np.array([[1.0, 0.5], [0.25, 0.1]])
    assert np.allclose(om.footprint_m(z, 500.0, cos), z / (500.0 * cos), rtol=1e-12)
    # 깊이와 초점거리를 같이 c배 해도 φ 는 같다; 깊이만 2배면 φ 도 2배
    assert math.isclose(float(om.footprint_m(3.0, 1000.0)), float(om.footprint_m(6.0, 2000.0)), rel_tol=1e-12)
    assert math.isclose(float(om.footprint_m(6.0, 1000.0)), 2 * float(om.footprint_m(3.0, 1000.0)), rel_tol=1e-12)


def test_footprint_invalid_depth_is_nan_not_error():
    out = om.footprint_m(np.array([1.0, 0.0, -2.0, np.nan, np.inf]), 100.0)
    assert out[0] == 0.01 and np.isnan(out[1:]).all()


@pytest.mark.parametrize("kw", [dict(f_px=0.0), dict(f_px=-1.0), dict(f_px=float("nan")), dict(f_px=float("inf")),
                                dict(f_px=100.0, cos_theta=0.0), dict(f_px=100.0, cos_theta=-0.5),
                                dict(f_px=100.0, cos_theta=1.01), dict(f_px=100.0, cos_theta=float("nan"))])
def test_footprint_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        om.footprint_m(2.0, **kw)


# ------------------------------------------------------------------ JPEG 계수
def test_jpeg_band_factor():
    assert om.jpeg_band_factor(None) == 1.0
    for q in (90, 91, 95, 100):
        assert om.jpeg_band_factor(q) == 1.0
    for q in (1, 10, 50, 75, 89.5):
        assert math.isclose(om.jpeg_band_factor(q), H.jpeg_factor(q), rel_tol=1e-12)
    assert math.isclose(om.jpeg_band_factor(50), 2.0, rel_tol=1e-12) and om.jpeg_band_factor(1) == 4.0
    qs = np.arange(1, 101)
    vals = [om.jpeg_band_factor(float(q)) for q in qs]
    assert all(a >= b for a, b in zip(vals, vals[1:])) and min(vals) >= 1.0 and max(vals) <= 4.0
    for bad in (0, 0.5, 101, float("nan")):
        with pytest.raises(ValueError):
            om.jpeg_band_factor(bad)
    with pytest.raises(TypeError):
        om.jpeg_band_factor("90")


# ------------------------------------------------------------------ φ_eff
def test_effective_footprint_each_term_and_max():
    phi = np.full((2, 2), 0.01)
    f = lambda **kw: om.effective_footprint(phi, **kw)       # noqa: E731
    assert np.allclose(f(), phi)                                # 열화 없음 -> φ_in
    assert np.allclose(f(r_ratio=0.5), 0.02)                    # r_in / r_eff = 2
    assert np.allclose(f(coc_px=3.0), 0.02)                     # c/1.5
    assert np.allclose(f(coc_px=1.5), 0.01) and np.allclose(f(coc_px=1.0), 0.01)   # 1.5 px 까지는 무손실
    assert np.allclose(f(motion_px=4.5), 0.03)
    assert np.allclose(f(q_jpeg=2.5), 0.025)
    assert np.allclose(f(nu_cap=0.125), 0.04)                   # 0.5 / ν_cap
    assert np.allclose(f(nu_cap=0.5), 0.01)
    # 합이 아니라 최대: 가장 큰 항 하나만 작용한다
    assert np.allclose(f(r_ratio=0.5, coc_px=3.0, motion_px=3.0, q_jpeg=2.0), 0.02)
    assert np.allclose(f(r_ratio=0.5, coc_px=6.0, motion_px=3.0, q_jpeg=2.0), 0.04)
    # 화소별 지도 방송
    coc = np.array([[0.0, 3.0], [6.0, 1.0]])
    assert np.allclose(f(coc_px=coc), 0.01 * np.array([[1, 2], [4, 1]]))


def test_effective_footprint_nan_propagates_and_validates():
    out = om.effective_footprint(np.array([np.nan, 0.01]), coc_px=3.0)
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.02)
    for kw in (dict(r_ratio=1 / 17), dict(r_ratio=1.01), dict(r_ratio=0.0), dict(coc_px=-0.1), dict(motion_px=-1.0),
               dict(q_jpeg=0.99), dict(nu_cap=0.0), dict(nu_cap=0.51), dict(coc_px=float("nan")),
               dict(motion_px=float("inf"))):
        with pytest.raises(ValueError):
            om.effective_footprint(0.01, **kw)
    with pytest.raises(ValueError):
        om.effective_footprint(0.0)
    with pytest.raises(ValueError):
        om.effective_footprint(-0.01)


# ------------------------------------------------------------------ 레벨
def test_levels_match_exhaustive_oracle_random_sweep():
    rng = np.random.default_rng(14)
    phis = np.exp(rng.uniform(math.log(1e-6), math.log(40.0), size=3000))
    hard, part = om.levels_from_footprint(phis)
    assert hard.shape == phis.shape and np.issubdtype(hard.dtype, np.integer)
    for p, h, q in zip(phis, hard, part):
        oh, op = H.owned_levels(p)
        assert (int(h), int(q)) == (oh, op), p


def test_part_is_hard_plus_one_and_ownership_predicate_agrees():
    rng = np.random.default_rng(15)
    for p in np.exp(rng.uniform(math.log(1e-4), math.log(5.0), size=200)):
        hard, part = om.levels_from_footprint(p)
        hard, part = int(hard), int(part)
        assert part == hard + 1
        for l in range(-8, 25):
            own = int(om.owner_of_level(l, p))
            assert own == (1 if l <= hard else 2 if l <= part else 0), (p, l)
            tau = 8.0 * 2.0 ** (-l)
            assert (own == 1) == (tau >= 2 * p * (1 - 1e-9))
            assert (own >= 1) == (tau >= p * (1 - 1e-9))


def test_levels_tie_goes_to_owned_and_edges():
    # τ_l = 2φ 정확히: 경소유. φ = 4·2^-l 은 2 의 거듭제곱이라 log2 가 정확하다.
    for l in (-3, 0, 5, 11, 20):
        phi = 4.0 * 2.0 ** (-l)
        assert int(om.levels_from_footprint(phi)[0]) == l
        assert int(om.levels_from_footprint(phi * (1 + 1e-6))[0]) == l - 1      # 조금 넘으면 한 단 내려간다
        assert int(om.levels_from_footprint(phi * (1 - 1e-6))[0]) == l
        assert int(om.levels_from_footprint(phi)[1]) == l + 1
    # LEVEL_RANGE 로 자른다
    lo = om.levels_from_footprint(1e6)
    hi = om.levels_from_footprint(1e-12)
    assert (int(lo[0]), int(lo[1])) == (-8, -8)
    assert (int(hi[0]), int(hi[1])) == (24, 24)


def test_part_extra_raises_partial_ceiling_only():
    """part = floor(a + 1 + extra) (a = log2(4/φ)): 반 옥타브 상향 폴백. 경소유는 안 변한다."""
    phi = 4.0 * 2.0 ** (-6) * 1.3        # a = 6 - log2(1.3) = 5.6215
    for extra, exp_part in ((0.0, 6), (0.30, 6), (0.37, 6), (0.40, 7), (0.5, 7), (0.9, 7), (1.0, 7)):
        h, p = (int(v) for v in om.levels_from_footprint(phi, part_extra=extra))
        assert (h, p) == (5, exp_part), extra
        assert (h, p) == H.owned_levels(phi, extra)
    for pe in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            om.levels_from_footprint(phi, part_extra=pe)
    # 소유 판정도 같은 규칙: 레벨 7 은 extra=0.5 일 때만 부분 소유
    assert int(om.owner_of_level(7, phi)) == 0 and int(om.owner_of_level(7, phi, part_extra=0.5)) == 2


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_levels_reject_bad_footprint(bad):
    with pytest.raises(ValueError):
        om.levels_from_footprint(bad)
    with pytest.raises(ValueError):
        om.levels_from_footprint(np.array([0.01, bad]))


def test_degradations_lower_levels_by_log2_factor():
    """열화 계수 c 배는 경소유 레벨을 floor(a - log2 c) 로 낮춘다(a = log2(4/φ_in))."""
    f = H.f_for(2048)
    phi = float(om.footprint_m(3.0, f))
    a = math.log2(4.0 / phi)
    for kw, fac in (({"r_ratio": 0.5}, 2.0), ({"coc_px": 6.0}, 4.0), ({"motion_px": 3.0}, 2.0), ({"q_jpeg": 3.0}, 3.0)):
        pe = float(om.effective_footprint(phi, **kw))
        assert math.isclose(pe, phi * fac, rel_tol=1e-12)
        h, p = om.levels_from_footprint(pe)
        assert int(h) == math.floor(a - math.log2(fac)) and int(p) == int(h) + 1


def test_returns_python_ints_for_scalars():
    h, p = om.levels_from_footprint(0.01)
    assert isinstance(h, (int, np.integer)) and isinstance(p, (int, np.integer))
    assert np.ndim(h) == 0 and np.ndim(p) == 0
