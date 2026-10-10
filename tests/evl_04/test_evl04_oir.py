"""EVL-04 OIR: 사람 2AFC 정답률 집계(Wilson 95% 구간, 정확 이항검정)와 목표표."""
import json

import numpy as np
import pytest
from scipy import stats

from evl04_support import ff


def test_counts_rate_wilson_and_exact_p_against_scipy_oracle():
    m = ff()
    r = m.oir_counts(55, 100)
    assert (r.n, r.k) == (100, 55) and type(r.rate) is float
    assert r.rate == pytest.approx(0.55)
    ref = stats.binomtest(55, 100).proportion_ci(confidence_level=0.95, method="wilson")
    assert r.ci_low == pytest.approx(ref.low, abs=1e-9) and r.ci_high == pytest.approx(ref.high, abs=1e-9)
    assert (r.ci_low, r.ci_high) == (pytest.approx(0.4524, abs=2e-4), pytest.approx(0.6439, abs=2e-4))
    assert r.p_value == pytest.approx(0.3682, abs=1e-3)  # 양측, 기대 50%
    assert r.p_value == pytest.approx(stats.binomtest(55, 100).pvalue, abs=1e-9)


def test_edges_and_symmetry():
    m = ff()
    assert m.oir_counts(50, 100).p_value == pytest.approx(1.0)
    z = m.oir_counts(0, 20)
    assert z.rate == 0.0 and z.ci_low == 0.0 and 0.0 < z.ci_high < 0.2
    o = m.oir_counts(20, 20)
    assert o.rate == 1.0 and o.ci_high == pytest.approx(1.0) and o.p_value < 1e-4
    a, b = m.oir_counts(30, 100), m.oir_counts(70, 100)
    assert a.p_value == pytest.approx(b.p_value, abs=1e-12)
    assert a.ci_low == pytest.approx(1 - b.ci_high, abs=1e-12)


def test_from_responses_equals_counts():
    m = ff()
    resp = [True] * 33 + [False] * 17
    r1 = m.oir(resp)
    r2 = m.oir_counts(33, 50)
    assert r1 == r2
    assert m.oir(np.array(resp)) == r2
    assert m.oir([1, 0, 1, 1]) == m.oir_counts(3, 4)  # 0/1 정수 허용


def test_to_dict_is_json_ready():
    m = ff()
    d = m.oir_counts(12, 20).as_dict()
    assert set(d) == {"n", "k", "rate", "ci_low", "ci_high", "p_value"}
    assert json.loads(json.dumps(d)) == d


def test_oir_validation():
    m = ff()
    for bad in ([], [2, 0], ["a"], [0.5], np.zeros((2, 2), bool), None):
        with pytest.raises((ValueError, TypeError)):
            m.oir(bad)
    for k, n in ((-1, 5), (6, 5), (0, 0), (1.5, 5)):
        with pytest.raises((ValueError, TypeError)):
            m.oir_counts(k, n)


# ---------------------------------------------------------------------------------------- 목표표 (§5.5)
def test_target_table_is_the_spec_table():
    m = ff()
    T = m.TARGETS
    assert set(T) == {"L1_photo_green", "L1_ai_yellow_safe", "L1_ai_yellow_ext_full", "escher_red_safe", "L2_painting_safe"}
    got = {k: (v.ffd, v.ffg, v.ffg_sal, v.de00, v.oir) for k, v in T.items()}
    assert got == {
        "L1_photo_green": (0.05, 0.5, 0.5, 1.5, 0.55),
        "L1_ai_yellow_safe": (0.06, 1.0, 0.5, 2.0, 0.60),
        "L1_ai_yellow_ext_full": (0.10, 2.5, 1.0, 2.0, 0.65),
        "escher_red_safe": (0.08, 1.5, 0.75, 2.5, None),
        "L2_painting_safe": (0.10, 1.5, 0.75, 3.0, None),
    }
    assert m.FFD_FLOOR_MAX == 0.015 and m.DE00_FLOOR_MAX == 0.5


def test_evaluate_targets():
    m = ff()
    ok = {"ffd": 0.05, "ffg": 0.5, "ffg_sal": 0.5, "de00": 1.5, "oir": 0.55}  # 경계값은 통과(<=)
    v = m.evaluate_targets(ok, "L1_photo_green")
    assert v.ok is True and v.checks == {"ffd": True, "ffg": True, "ffg_sal": True, "de00": True, "oir": True}
    bad = dict(ok, ffd=0.0501, de00=1.6)
    v2 = m.evaluate_targets(bad, "L1_photo_green")
    assert v2.ok is False and v2.checks["ffd"] is False and v2.checks["de00"] is False and v2.checks["ffg"] is True
    # 없는 측정·목표 없음("보고")은 실패가 아니라 None
    v3 = m.evaluate_targets({"ffd": 0.07, "ffg": 1.4}, "escher_red_safe")
    assert v3.ok is True and v3.checks["ffd"] is True and v3.checks["ffg"] is True
    assert v3.checks["ffg_sal"] is None and v3.checks["oir"] is None and v3.checks["de00"] is None
    v4 = m.evaluate_targets({"ffd": 0.11, "oir": 0.9}, "L2_painting_safe")
    assert v4.ok is False and v4.checks["oir"] is None  # OIR 목표 없음 -> 보고만
    assert m.evaluate_targets({"ffd": float("nan")}, "L1_photo_green").checks["ffd"] is False  # NaN 은 통과 못 한다
    with pytest.raises(ValueError):
        m.evaluate_targets(ok, "nope")
    with pytest.raises(ValueError):
        m.evaluate_targets({"bogus": 1.0}, "L1_photo_green")
