"""TCH-05 DecisionCurve 컨테이너, 직렬화, 해석 규칙(σ 채택)."""
import json
import math

import numpy as np
import pytest

from buc.teacher import decision_curve as dc
from tch05_support import TABLE, logistic_rows, make_curve, ref_edge

A_EDGES = [6 - 0.5 * j for j in range(7)]          # 두 step마다 정확히 1 옥타브씩 이동(정수 평행이동)


def test_container_validation():
    d = logistic_rows([5, 4], 4, 6)
    z = np.zeros_like(d)
    ok = dict(indices=(0, 1), sigmas=(TABLE.points[0], TABLE.points[1]), decision=d, spread=z,
              input_energy=np.ones(6), levels=6, steps=2, n_images=1, n_seeds=2, table_key=TABLE.key)
    dc.DecisionCurve(**ok)
    bad = [dict(ok, indices=(1, 0)), dict(ok, indices=(0, 0)), dict(ok, sigmas=(0.9,)), dict(ok, levels=5),
           dict(ok, spread=z[:1]), dict(ok, input_energy=np.ones(5)), dict(ok, indices=(), sigmas=())]
    for b in bad:
        with pytest.raises(dc.DecisionCurveError):
            dc.DecisionCurve(**b)


def test_edges_and_widths_match_independent_oracle():
    c = make_curve(A_EDGES, 4, 8)
    for tau in (0.1, 0.5):
        for row, e in zip(c.decision, c.edges(tau)):
            assert abs(e - ref_edge(row, tau)) < 2e-4
    for w, a, b in zip(c.widths(), c.edges(0.1), c.edges(0.5)):
        assert w == pytest.approx(a - b, abs=1e-12)
    # 해상 수치: 로지스틱 폭은 2 ln9 / k 근처(조각 선형 근사) 이고 1.5 옥타브보다 좁다
    assert all(0.5 < w < 1.0 for w in c.widths())


def test_valid_mask_by_exact_sigma_only():
    c = make_curve(A_EDGES, 4, 8)
    s = c.sigmas[2]
    m = c.valid_mask(s)
    assert m.dtype == bool and m.shape == (8,)
    assert np.array_equal(m, dc.valid_mask(c.decision[2], 0.1))
    assert np.array_equal(c.valid_mask(s, tau=0.5), dc.valid_mask(c.decision[2], 0.5))
    assert c.valid_mask(s + 5e-10).shape == (8,)               # 1e-9 이내는 같은 점
    with pytest.raises(dc.DecisionCurveError):
        c.valid_mask(0.77)                                      # 조용히 스냅하지 않는다
    # σ가 클수록(앞 인덱스) 보존되는 굵은 옥타브의 수가 적다
    counts = [int(c.valid_mask(s).sum()) for s in c.sigmas]
    assert all(a <= b for a, b in zip(counts, counts[1:])) and counts[0] < counts[-1]


def test_json_round_trip_with_nan():
    c = make_curve(A_EDGES, 4, 8, spread=np.full((7, 8), 0.25))
    c.decision[3, 5] = math.nan
    c.spread[1, 0] = math.nan
    s = json.dumps(c.to_dict(), allow_nan=False)               # NaN 은 null 로
    r = dc.DecisionCurve.from_dict(json.loads(s))
    assert r.indices == c.indices and r.sigmas == c.sigmas and r.table_key == c.table_key
    assert (r.levels, r.steps, r.n_images, r.n_seeds) == (c.levels, c.steps, c.n_images, c.n_seeds)
    for a, b in ((r.decision, c.decision), (r.spread, c.spread), (r.input_energy, c.input_energy)):
        assert np.array_equal(a, b, equal_nan=True)


def test_max_abs_delta():
    a = make_curve(A_EDGES, 4, 8)
    b = make_curve(A_EDGES, 4, 8)
    assert dc.max_abs_delta(a, b) == 0.0
    b.decision[2, 3] += 0.04
    b.decision[5, 7] -= 0.01
    assert dc.max_abs_delta(a, b) == pytest.approx(0.04, abs=1e-12)
    with pytest.raises(dc.DecisionCurveError):
        dc.max_abs_delta(a, make_curve(A_EDGES[:5], 4, 8))
    with pytest.raises(dc.DecisionCurveError):
        dc.max_abs_delta(a, make_curve(A_EDGES, 4, 6))
    b.decision[0, 0] = math.nan
    with pytest.raises(dc.DecisionCurveError):
        dc.max_abs_delta(a, b)


# ----------------------------------------------------------------------------- interpret

def test_interpret_clean_curve_adopts_all_separating_windows():
    c = make_curve(A_EDGES, 4, 8)
    it = dc.interpret(c)
    assert [v.index for v in it.verdicts] == list(range(7))
    assert all(v.usable and 0.5 < v.width < 1.0 for v in it.verdicts)
    assert np.allclose(it.shifts[:5], 1.0, atol=1e-9) and all(math.isnan(s) for s in it.shifts[5:])
    assert it.adopted == (0, 1, 2, 3, 4)
    assert it.monotone is True and it.e2a_pass is True and it.fallback == "none"
    for v, e1, e5, w in zip(it.verdicts, c.edges(0.1), c.edges(0.5), c.widths()):
        assert (v.edge_valid, v.edge_decided, v.width) == (pytest.approx(e1), pytest.approx(e5), pytest.approx(w))
    assert [v.sigma for v in it.verdicts] == list(c.sigmas)


def test_interpret_wide_transition_triggers_fallback():
    c = make_curve(A_EDGES, 1.2, 8)                             # 폭 약 1.9 옥타브 > 1.5
    it = dc.interpret(c)
    assert not any(v.usable for v in it.verdicts)
    assert math.isnan(it.verdicts[0].width) and it.verdicts[2].width > 1.5      # 굵은 쪽이 범위 밖이면 폭은 NaN
    assert it.adopted == () and it.e2a_pass is False and it.fallback == "widen_level_gap_or_sr_lora"


def test_interpret_too_coarse_windows_not_separating():
    c = make_curve([9 - 1.2 * j for j in range(7)], 4, 10)      # 두 step당 약 2.4 옥타브 이동
    it = dc.interpret(c)
    assert it.verdicts[2].usable
    assert all(s > 2.0 for s in it.shifts[:5])
    assert it.adopted == () and it.fallback == "widen_level_gap_or_sr_lora"


def test_interpret_partial_adoption_uses_shift_window_and_usable():
    c = make_curve([6.0, 5.2, 4.0, 3.3, 3.0, 2.0, 1.2], 4, 8)
    it = dc.interpret(c)
    assert it.shifts[0] == pytest.approx(2.0, abs=1e-9) and it.shifts[2] == pytest.approx(1.0, abs=1e-9)
    assert 1.7 < it.shifts[1] < 2.1
    assert 1.2 < it.shifts[3] < 1.5 and 1.6 < it.shifts[4] < 2.0
    assert it.adopted == (2, 3) and it.e2a_pass is True and it.fallback == "none"


def test_interpret_parameter_overrides():
    c = make_curve(A_EDGES, 4, 8)
    assert dc.interpret(c, sep_target=2.0).adopted == ()                  # 2 옥타브 분리를 요구
    assert dc.interpret(c, sep_target=2.0, sep_tol=1.1).adopted == (0, 1, 2, 3, 4)
    assert dc.interpret(c, width_max=0.5).adopted == ()                    # 폭 한도를 줄이면 전부 탈락


def test_interpret_requires_two_adopted_for_e2a():
    c = make_curve([6.0, 5.5, 5.0], 4, 8, indices=(0, 1, 2))     # j=0 만 파트너(2)가 있다
    it = dc.interpret(c)
    assert it.adopted == (0,) and it.e2a_pass is False


def test_interpret_noncontiguous_indices_partner_lookup():
    c = make_curve([6, 5, 4, 3], 4, 8, indices=(0, 2, 4, 6))
    it = dc.interpret(c)
    assert it.shifts[:3] == pytest.approx((1.0, 1.0, 1.0), abs=1e-9) and math.isnan(it.shifts[3])
    assert it.adopted == (0, 2, 4)
    c2 = make_curve([6.0, 5.4, 4.2], 4, 8, indices=(0, 1, 3))    # 0 -> 2 없음, 1 -> 3 있음, 3 -> 5 없음
    s2 = dc.interpret(c2).shifts
    assert math.isnan(s2[0]) and math.isfinite(s2[1]) and math.isnan(s2[2])


def test_interpret_monotone_flag():
    up = make_curve([3.0, 3.6, 4.2, 4.8, 5.4], 4, 8)             # 인덱스가 커질수록 결정 경계가 굵어짐: 이상
    it = dc.interpret(up)
    assert it.monotone is False and it.adopted == ()
    jitter = make_curve([5.0, 5.3, 4.0, 3.5, 3.0], 4, 8)         # 0.3 옥타브 역행은 mono_tol(0.5) 안
    assert dc.interpret(jitter).monotone is True
    assert dc.interpret(jitter, mono_tol=0.1).monotone is False


@pytest.mark.parametrize("e", [20.0, -20.0])
def test_interpret_inconclusive_when_edges_infinite(e):
    c = make_curve([e] * 7, 4, 8)                                # 전부 결정됨(+inf) 또는 전부 보존(-inf)
    it = dc.interpret(c)
    assert all(math.isnan(v.width) and not v.usable for v in it.verdicts)
    assert it.adopted == () and it.e2a_pass is False and it.fallback == "inconclusive"


def test_interpret_rejects_nan_and_bad_taus():
    c = make_curve(A_EDGES, 4, 8)
    c.decision[2, 4] = math.nan
    with pytest.raises(dc.DecisionCurveError):
        dc.interpret(c)
    good = make_curve(A_EDGES, 4, 8)
    for kw in (dict(valid_tau=0.5, decided_tau=0.1), dict(valid_tau=0.3, decided_tau=0.3), dict(width_max=0.0),
               dict(sep_tol=-0.1)):
        with pytest.raises(dc.DecisionCurveError):
            dc.interpret(good, **kw)


def test_interpret_does_not_mutate_curve():
    c = make_curve(A_EDGES, 4, 8)
    before = c.decision.copy()
    dc.interpret(c)
    assert np.array_equal(c.decision, before)
