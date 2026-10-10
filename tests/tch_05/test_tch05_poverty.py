"""TCH-05 가난선 프로브: 옥타브별 DR 비 + 2AFC -> 학생 소유 옥타브 수 -> mm 경계, 초기 o_s."""
import math

import numpy as np
import pytest

from buc.teacher import decision_curve as dc


def test_octave_pitch_mm():
    assert dc.octave_pitch_mm(0, 0.5) == 0.5
    assert dc.octave_pitch_mm(3, 0.5) == 4.0
    assert dc.octave_pitch_mm(2, 1.8) == pytest.approx(7.2, abs=1e-12)


def test_prefix_rule_dr_only():
    # 비 = 0.9, 1.1, 1.2, 1.0  ->  [T, T, F, T]; 소유는 가장 미세한 옥타브에서 이어진 접두만
    p = dc.poverty_line([0.9, 1.1, 1.2, 1.0], [1.0] * 4)
    assert p.passes == (True, True, False, True) and p.n_octaves == 2
    assert p.ratio == pytest.approx((0.9, 1.1, 1.2, 1.0))
    assert p.boundary_mm(0.5) == pytest.approx(2.0)           # 0.5 mm * 2^2
    assert p.teacher_pixel_fraction == pytest.approx(1 / 16)


def test_ratio_threshold_is_inclusive_and_configurable():
    assert dc.poverty_line([1.1], [1.0]).n_octaves == 1                       # 비 == 1.1 은 통과
    assert dc.poverty_line([1.1000001], [1.0]).n_octaves == 0
    assert dc.poverty_line([1.15, 1.15], [1.0, 1.0], ratio_max=1.2).n_octaves == 2
    assert dc.poverty_line([2.0], [4.0]).ratio == pytest.approx((0.5,))       # 학생/교사


def test_afc_gate():
    st, te = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    assert dc.poverty_line(st, te, afc=[0.50, 0.55, 0.60]).passes == (True, True, False)     # 0.55 포함
    assert dc.poverty_line(st, te, afc=[0.50, 0.50, 0.50]).n_octaves == 3
    assert dc.poverty_line(st, te, afc=[0.6, 0.5, 0.5]).n_octaves == 0                        # 둘 다 만족해야 통과
    assert dc.poverty_line([1.2, 1.0], [1.0, 1.0], afc=[0.5, 0.5]).passes == (False, True)
    assert dc.poverty_line(st, te, afc=[0.7, 0.7, 0.7], afc_max=0.75).n_octaves == 3


def test_zero_octaves_and_all_octaves():
    z = dc.poverty_line([5.0, 5.0], [1.0, 1.0])
    assert z.n_octaves == 0 and z.boundary_mm(0.5) == 0.5 and z.teacher_pixel_fraction == 1.0
    a = dc.poverty_line([1.0] * 5, [1.0] * 5)
    assert a.n_octaves == 5 and a.boundary_mm(1.0) == 32.0


def test_each_extra_octave_cuts_teacher_pixels_to_a_quarter():
    fr = [dc.poverty_line([1.0] * n + [9.0] * (6 - n), [1.0] * 6).teacher_pixel_fraction for n in range(7)]
    assert fr == pytest.approx([0.25 ** n for n in range(7)])


@pytest.mark.parametrize("bad", ["len", "nan_s", "inf_t", "zero_t", "neg_t", "neg_s", "afc_len", "afc_range", "empty",
                                 "ndim"])
def test_poverty_rejects_bad_input(bad):
    s, t, kw = [1.0, 1.0], [1.0, 1.0], {}
    if bad == "len":
        t = [1.0]
    elif bad == "nan_s":
        s = [1.0, math.nan]
    elif bad == "inf_t":
        t = [1.0, math.inf]
    elif bad == "zero_t":
        t = [1.0, 0.0]
    elif bad == "neg_t":
        t = [1.0, -1.0]
    elif bad == "neg_s":
        s = [1.0, -0.1]
    elif bad == "afc_len":
        kw["afc"] = [0.5]
    elif bad == "afc_range":
        kw["afc"] = [0.5, 1.2]
    elif bad == "empty":
        s, t = [], []
    elif bad == "ndim":
        s, t = np.ones((2, 2)), np.ones((2, 2))
    with pytest.raises(dc.DecisionCurveError):
        dc.poverty_line(s, t, **kw)


def test_inputs_not_mutated_and_numpy_ok():
    s, t = np.array([0.9, 1.3]), np.array([1.0, 1.0])
    s0 = s.copy()
    p = dc.poverty_line(s, t)
    assert np.array_equal(s, s0) and p.n_octaves == 1 and isinstance(p.n_octaves, int)


def test_initial_o_s_is_max_of_judge_boundary_and_measurement():
    assert dc.initial_o_s_mm(1.8, 1.0) == 1.8
    assert dc.initial_o_s_mm(1.8, 2.0) == 2.0
    assert dc.initial_o_s_mm(2, 2) == 2.0
    with pytest.raises(dc.DecisionCurveError):
        dc.initial_o_s_mm(0.0, 1.0)
    with pytest.raises(dc.DecisionCurveError):
        dc.initial_o_s_mm(1.8, math.nan)
