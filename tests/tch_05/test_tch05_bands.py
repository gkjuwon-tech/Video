"""TCH-05 옥타브 분해·결정 D·씨앗 퍼짐·가장자리 보간의 수치 계약 (라플라시안 피라미드 = FND-06)."""
import math

import numpy as np
import pytest

from buc.core.pyramid import laplacian_pyramid
from buc.teacher import decision_curve as dc
from tch05_support import natural, ref_edge

L = 4


def _pair(x, y, levels=L, seeds=2):
    return dc.band_curve([x], [[y] * seeds], levels)


@pytest.fixture(scope="module")
def x():
    return natural(64, 64, 1)


def test_public_names():
    for n in ("DecisionCurveError", "band_curve", "edge_octave", "decision_width", "valid_mask", "DecisionCurve",
              "interpret", "run_x1_slim", "TeacherCodec", "poverty_line", "initial_o_s_mm", "octave_pitch_mm",
              "forward_count", "max_abs_delta"):
        assert n in dc.__all__ and hasattr(dc, n), n
    assert issubclass(dc.DecisionCurveError, ValueError)
    assert (dc.VALID_TAU, dc.DECIDED_TAU, dc.WIDTH_MAX, dc.SEP_TARGET, dc.SEP_TOL) == (0.1, 0.5, 1.5, 1.0, 0.5)


def test_identity_gives_zero_and_negation_gives_two(x):
    c = _pair(x, x)
    assert c.decision.shape == (L,) and np.all(c.decision == 0.0)
    assert np.allclose(c.spread, 0.0, atol=1e-15)
    c2 = _pair(x, -x)
    assert np.allclose(c2.decision, 2.0, atol=1e-12)


@pytest.mark.parametrize("g", [0.5, 2.0, 0.25])
def test_gain_closed_form(x, g):
    c = _pair(x, g * x)
    assert np.allclose(c.decision, (1 - g) ** 2 / (1 + g ** 2), rtol=1e-10)


def test_zero_output_is_one_and_spread_undefined(x):
    c = _pair(x, np.zeros_like(x))
    assert np.allclose(c.decision, 1.0, atol=1e-12)
    assert np.all(np.isnan(c.spread))          # 출력 에너지 0 -> 정의되지 않음(조용히 0으로 바꾸지 않는다)


def test_scale_invariance_and_symmetry(x):
    y = natural(64, 64, 2)
    base = dc.band_curve([x], [[y, y]], L).decision
    scaled = dc.band_curve([7.5 * x], [[7.5 * y, 7.5 * y]], L).decision
    assert np.allclose(base, scaled, rtol=1e-9)
    swapped = dc.band_curve([y], [[x, x]], L).decision
    assert np.allclose(base, swapped, rtol=1e-10)


def test_independent_noise_is_about_one():
    g = np.random.default_rng(0)
    a, b = g.standard_normal((128, 128, 3)), g.standard_normal((128, 128, 3))
    d = _pair(a, b).decision
    assert np.all(np.abs(d - 1.0) < 0.15), d


def test_band_localization_finest_octave_only():
    h = 64
    yy, xx = np.mgrid[0:h, 0:h]
    x = np.cos(2 * np.pi * xx / 32)[..., None] * np.ones(3)
    chk = ((-1.0) ** (xx + yy))[..., None] * np.ones(3)
    d = _pair(x, x + 0.5 * chk).decision
    assert d[0] > 0.99                      # 주기 2 px 체커는 옥타브 0에만 들어간다
    assert np.all(d[1:] < 1e-3), d


def test_input_energy_matches_fnd06_laplacian(x):
    y = natural(64, 64, 3)
    c = dc.band_curve([x, y], [[y, y], [x, x]], L)
    exp = np.zeros(L)
    for img in (x, y):
        lp = laplacian_pyramid(img, L)
        for o in range(L):
            exp[o] += float(np.sum(lp[o].astype(np.float64) ** 2))
    assert np.allclose(c.input_energy, exp, rtol=1e-11)
    assert (c.n_images, c.n_seeds, c.levels) == (2, 2, L)


def test_pooling_is_energy_weighted_ratio_of_sums():
    a, b = natural(64, 64, 4), 3.0 * natural(64, 64, 5)
    ea = dc.band_curve([a], [[a, a]], L).input_energy
    eb = dc.band_curve([b], [[b, b]], L).input_energy
    pooled = dc.band_curve([a, b], [[0.5 * a, 0.5 * a], [0 * b, 0 * b]], L).decision
    num = 2 * 0.25 * ea + 2 * eb
    den = 2 * 1.25 * ea + 2 * eb
    assert np.allclose(pooled, num / den, rtol=1e-10)


def test_spread_closed_forms(x):
    u = natural(64, 64, 6)
    assert np.allclose(dc.band_curve([x], [[u, -u]], L).spread, 1.0, atol=1e-12)
    assert np.allclose(dc.band_curve([x], [[u, u]], L).spread, 0.0, atol=1e-12)
    assert np.allclose(dc.band_curve([x], [[u, u, -u]], L).spread, 8.0 / 9.0, rtol=1e-10)
    s = dc.band_curve([x], [[natural(64, 64, 10 + i) for i in range(4)]], L).spread
    assert np.all((s >= 0) & (s <= 1.0 + 1e-12))
    assert np.all(np.abs(s - 0.75) < 0.12), s         # 독립 4시드: 1 - 1/4 근처


def test_constant_images_are_nan_not_zero():
    c = np.full((64, 64, 3), 0.3)
    r = dc.band_curve([c], [[c, c]], L)
    assert np.all(np.isnan(r.decision)) and np.all(np.isnan(r.spread))


def test_channels_and_2d_inputs_ok(x):
    g2 = x[..., 0]
    c = dc.band_curve([g2], [[g2, 2 * g2]], L)
    assert c.decision.shape == (L,) and np.all(np.isfinite(c.decision))


def test_pure_deterministic_and_inputs_untouched(x):
    y = natural(64, 64, 7)
    x0, y0 = x.copy(), y.copy()
    st = np.random.get_state()[1].copy()
    a = dc.band_curve([x], [[y, 2 * y]], L)
    b = dc.band_curve([x], [[y, 2 * y]], L)
    assert np.array_equal(a.decision, b.decision) and np.array_equal(a.spread, b.spread)
    assert np.array_equal(x, x0) and np.array_equal(y, y0)
    assert np.array_equal(np.random.get_state()[1], st)


def test_float32_inputs_accepted_same_result_as_float64(x):
    y = natural(64, 64, 8)
    a = dc.band_curve([x.astype(np.float32)], [[y.astype(np.float32)] * 2], L).decision
    b = dc.band_curve([x.astype(np.float32).astype(np.float64)], [[y.astype(np.float32).astype(np.float64)] * 2], L).decision
    assert np.allclose(a, b, rtol=1e-12)          # 내부 계산은 float64


@pytest.mark.parametrize("bad", ["one_seed", "count_mismatch", "ragged", "shape", "nan", "tiny", "levels0",
                                 "levels_bool", "levels_float", "ndim", "empty"])
def test_band_curve_rejects_bad_input(x, bad):
    y = natural(64, 64, 9)
    cases = {
        "one_seed": lambda: dc.band_curve([x], [[y]], L),
        "count_mismatch": lambda: dc.band_curve([x, x], [[y, y]], L),
        "ragged": lambda: dc.band_curve([x, x], [[y, y], [y, y, y]], L),
        "shape": lambda: dc.band_curve([x], [[y[:32], y[:32]]], L),
        "nan": lambda: dc.band_curve([x], [[np.where(y > 5, y, np.nan), y]], L),
        "tiny": lambda: dc.band_curve([x[:16, :16]], [[y[:16, :16]] * 2], L),     # 16 < 2^(L+1)
        "levels0": lambda: dc.band_curve([x], [[y, y]], 0),
        "levels_bool": lambda: dc.band_curve([x], [[y, y]], True),
        "levels_float": lambda: dc.band_curve([x], [[y, y]], 4.0),
        "ndim": lambda: dc.band_curve([x[0]], [[y[0], y[0]]], L),
        "empty": lambda: dc.band_curve([], [], L),
    }
    with pytest.raises(dc.DecisionCurveError):
        cases[bad]()


def test_min_size_boundary(x):
    ok = x[:32, :32]
    assert dc.band_curve([ok], [[ok, ok]], 4).decision.shape == (4,)       # 32 == 2^(4+1) 허용
    with pytest.raises(dc.DecisionCurveError):
        dc.band_curve([ok], [[ok, ok]], 5)                                  # 32 < 2^6


# ----------------------------------------------------------------------------- 가장자리 / 폭 / 마스크

D5 = np.array([0.9, 0.7, 0.5, 0.2, 0.05])


def test_edge_hand_values():
    assert dc.edge_octave(D5, 0.1) == pytest.approx(3 + 2 / 3, abs=1e-12)
    assert dc.edge_octave(D5, 0.5) == pytest.approx(2.0, abs=1e-12)
    assert dc.edge_octave(D5, 0.7) == pytest.approx(1.0, abs=1e-12)
    assert dc.decision_width(D5) == pytest.approx(3 + 2 / 3 - 2.0, abs=1e-12)
    step = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert dc.edge_octave(step, 0.1) == pytest.approx(2.9, abs=1e-12)
    assert dc.edge_octave(step, 0.5) == pytest.approx(2.5, abs=1e-12)
    assert dc.decision_width(step) == pytest.approx(0.4, abs=1e-12)


def test_edge_scans_from_coarse_and_ignores_fine_side_dips():
    d = np.array([0.05, 0.9, 0.05, 0.05])
    assert dc.edge_octave(d, 0.1) == pytest.approx(1 + 0.8 / 0.85, abs=1e-12)
    assert np.array_equal(dc.valid_mask(d, 0.1), [False, False, True, True])


def test_edge_infinities():
    assert dc.edge_octave([0.9, 0.8, 0.7], 0.1) == math.inf          # 가장 굵은 옥타브도 교사가 정함
    assert dc.edge_octave([0.05, 0.02, 0.01], 0.1) == -math.inf      # 가장 미세한 옥타브까지 보존
    assert math.isnan(dc.decision_width([0.9, 0.8, 0.7]))
    assert math.isnan(dc.decision_width([0.05, 0.02, 0.01]))


def test_edge_matches_independent_oracle_on_random_curves():
    g = np.random.default_rng(3)
    for _ in range(25):
        d = np.sort(g.uniform(0.0, 1.2, 7))[::-1] * g.uniform(0.9, 1.1, 7)     # 대체로 감소, 약간의 요철
        for tau in (0.1, 0.5):
            got, ref = dc.edge_octave(d, tau), ref_edge(d, tau)
            if math.isinf(ref):
                assert got == ref
            else:
                assert abs(got - ref) < 2e-4, (d, tau, got, ref)


def test_width_nonnegative_and_orders_thresholds():
    g = np.random.default_rng(4)
    for _ in range(25):
        d = np.sort(g.uniform(0.0, 1.0, 8))[::-1]
        w = dc.decision_width(d)
        assert math.isnan(w) or w >= 0.0
    with pytest.raises(dc.DecisionCurveError):
        dc.decision_width(D5, lo=0.5, hi=0.1)
    with pytest.raises(dc.DecisionCurveError):
        dc.decision_width(D5, lo=0.3, hi=0.3)


def test_mask_is_contiguous_coarse_suffix_and_matches_ceil_of_edge():
    assert np.array_equal(dc.valid_mask(D5, 0.1), [False] * 4 + [True])
    assert np.array_equal(dc.valid_mask(D5, 0.5), [False, False, True, True, True])      # D == tau 는 유효
    g = np.random.default_rng(5)
    for _ in range(25):
        d = np.sort(g.uniform(0.0, 1.0, 7))[::-1]
        m = dc.valid_mask(d, 0.1)
        p = dc.edge_octave(d, 0.1)
        assert m.dtype == bool and m.shape == (7,)
        if math.isinf(p):
            assert m.all() if p < 0 else not m.any()
        else:
            first = int(np.argmax(m))
            assert m[first:].all() and not m[:first].any() and first == math.ceil(p)


@pytest.mark.parametrize("fn", ["edge", "width", "mask"])
def test_edge_functions_reject_nan_and_bad_args(fn):
    bad = np.array([0.9, np.nan, 0.1])
    call = {"edge": lambda a, **k: dc.edge_octave(a, 0.1), "width": lambda a, **k: dc.decision_width(a),
            "mask": lambda a, **k: dc.valid_mask(a, 0.1)}[fn]
    with pytest.raises(dc.DecisionCurveError):
        call(bad)
    with pytest.raises(dc.DecisionCurveError):
        call(np.zeros((2, 3)))
    with pytest.raises(dc.DecisionCurveError):
        call(np.array([]))
    with pytest.raises(dc.DecisionCurveError):
        dc.edge_octave(D5, 0.0)
