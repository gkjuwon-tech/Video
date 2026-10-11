"""CAM-04: 규칙 표 — 상수, alpha_sigma, 범주 F0-F3 경계, 범주별 한계표, 포함 관계."""
import math

import pytest

from cam04_helpers import fe


# ---------------------------------------------------------------- 상수
def test_constants_match_contract():
    m = fe()
    assert m.GRADES == ("safe", "extended", "full")
    assert m.CATEGORIES == ("F0", "F1", "F2", "F3")
    assert m.EPS_F == pytest.approx(0.03) and m.W_F_M == pytest.approx(0.15)
    assert m.BACK_YAW_DEG == pytest.approx(120.0)
    assert m.H_F1_MAX == pytest.approx(0.03) and m.H_F2_MAX == pytest.approx(0.15)
    assert m.FACE_CAP_HALF_DEG == pytest.approx(45.0) and m.HEAD_RADIUS_M == pytest.approx(0.10)
    assert m.M_TOL == pytest.approx(0.02)
    assert m.ANGLE_TOL_DEG == pytest.approx(1e-6) and m.UNOBS_TOL == pytest.approx(1e-9)


# ---------------------------------------------------------------- alpha_sigma
@pytest.mark.parametrize("sigma,expect", [(0.01, 26.74), (0.02, 13.0), (0.03, 8.63)])
def test_alpha_sigma_matches_published_table(sigma, expect):
    # 파이프라인 §10 표: 1 cm -> 27 도, 2 cm -> 13 도, 3 cm -> 8.6 도.
    assert fe().alpha_sigma_deg(sigma) == pytest.approx(expect, abs=0.02)


@pytest.mark.parametrize("sigma", [0.004, 0.0045, 0.0051, 0.007, 0.0123, 0.05, 0.2])
def test_alpha_sigma_is_asin_formula(sigma):
    # 독립 재계산: asin(eps_f * w_f / sigma) = asin(4.5 mm / sigma).
    expect = math.degrees(math.asin(min(1.0, 0.03 * 0.15 / sigma)))
    assert fe().alpha_sigma_deg(sigma) == pytest.approx(expect, abs=1e-9)


def test_alpha_sigma_saturates_at_90_and_zero_sigma():
    m = fe()
    assert m.alpha_sigma_deg(0.0) == 90.0
    assert m.alpha_sigma_deg(0.0045) == pytest.approx(90.0)     # eps*w/sigma = 1
    assert m.alpha_sigma_deg(0.001) == 90.0
    assert m.alpha_sigma_deg(math.inf) == pytest.approx(0.0, abs=1e-12)


def test_alpha_sigma_monotone_and_overrides():
    m = fe()
    sig = [0.0046, 0.006, 0.01, 0.02, 0.05, 0.5]
    vals = [m.alpha_sigma_deg(s) for s in sig]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    # eps_f, w_f 를 바꾸면 곱만 중요하다.
    assert m.alpha_sigma_deg(0.02, eps_f=0.06, w_f_m=0.15) == pytest.approx(math.degrees(math.asin(0.45)), abs=1e-9)
    assert m.alpha_sigma_deg(0.02, eps_f=0.03, w_f_m=0.30) == pytest.approx(math.degrees(math.asin(0.45)), abs=1e-9)


@pytest.mark.parametrize("bad", [-0.01, float("nan")])
def test_alpha_sigma_rejects_bad_sigma(bad):
    with pytest.raises(ValueError):
        fe().alpha_sigma_deg(bad)


@pytest.mark.parametrize("kw", [dict(eps_f=0.0), dict(eps_f=-1.0), dict(w_f_m=0.0), dict(w_f_m=-0.1)])
def test_alpha_sigma_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        fe().alpha_sigma_deg(0.02, **kw)


# ---------------------------------------------------------------- 범주
def test_category_h_f_boundaries():
    c = fe().face_category
    assert c(0.001) == "F1" and c(0.03) == "F1"           # h_f <= 3%
    assert c(0.0300001) == "F2" and c(0.15) == "F2"       # 3% < h_f <= 15%
    assert c(0.1500001) == "F3" and c(1.0) == "F3"        # h_f > 15%


def test_category_back_view_and_no_face_are_f0():
    c = fe().face_category
    assert c(0.1, has_face=False) == "F0"
    assert c(0.1, 120.0) == "F2"                          # 정확히 120 도는 뒷모습이 아니다
    assert c(0.1, 120.001) == "F0" and c(0.1, -120.001) == "F0"
    assert c(0.1, 180.0) == "F0" and c(0.5, 170.0) == "F0" and c(0.01, 150.0) == "F0"
    assert c(0.1, -119.9) == "F2"
    # yaw 는 감아서 비교한다: 200 도 = -160 도(뒷모습), 250 도 = -110 도(옆얼굴), 360 도 = 0 도.
    assert c(0.1, 200.0) == "F0" and c(0.1, 250.0) == "F2" and c(0.1, 360.0) == "F2"


@pytest.mark.parametrize("h", [0.0, -0.1, 1.0001, float("nan")])
def test_category_rejects_bad_h_f(h):
    with pytest.raises(ValueError):
        fe().face_category(h)


# ---------------------------------------------------------------- 한계표
def test_limits_table_without_uncertainty():
    L = fe().limits
    f0, f1, f2, f3 = L("F0"), L("F1"), L("F2"), L("F3")
    assert (f0.category, f1.category, f2.category, f3.category) == ("F0", "F1", "F2", "F3")
    assert math.isinf(f0.yaw_deg) and math.isinf(f0.pitch_deg) and math.isinf(f0.m_max)
    assert math.isinf(f1.yaw_deg) and math.isinf(f1.pitch_deg) and f1.m_max == pytest.approx(1.5)
    assert f2.yaw_deg == pytest.approx(30.0) and f2.pitch_deg == pytest.approx(15.0) and f2.m_max == pytest.approx(1.0)
    assert f3.yaw_deg == pytest.approx(15.0) and f3.pitch_deg == pytest.approx(10.0) and f3.m_max == pytest.approx(1.0)
    assert f0.unobs_max == pytest.approx(0.05) and f1.unobs_max == pytest.approx(0.03)
    assert f2.unobs_max == pytest.approx(0.05) and f3.unobs_max == pytest.approx(0.05)
    assert f0.grades == ("safe", "extended", "full") and f1.grades == ("safe", "extended", "full")
    assert f2.grades == ("safe", "extended") and f3.grades == ("safe",)
    assert f2.alpha_sigma_deg == 90.0


def test_limits_yaw_is_min_of_table_and_alpha_sigma():
    L = fe().limits
    assert L("F2", 0.005).yaw_deg == pytest.approx(30.0)                 # alpha(0.5cm)=64 도 -> 표가 이긴다
    assert L("F2", 0.01).yaw_deg == pytest.approx(26.74, abs=0.02)       # alpha 가 이긴다
    assert L("F2", 0.02).yaw_deg == pytest.approx(13.0, abs=0.02)
    assert L("F3", 0.01).yaw_deg == pytest.approx(15.0)                  # 표(15)가 alpha(26.7)보다 작다
    assert L("F3", 0.02).yaw_deg == pytest.approx(13.0, abs=0.02)
    assert L("F3", 0.03).yaw_deg == pytest.approx(8.63, abs=0.02)
    assert L("F2", 0.02).alpha_sigma_deg == pytest.approx(13.0, abs=0.02)
    # pitch/m/unobs/grades 는 sigma 와 무관, F0/F1 은 sigma 가 있어도 무제한.
    assert L("F2", 0.03).pitch_deg == pytest.approx(15.0) and L("F2", 0.03).m_max == pytest.approx(1.0)
    assert math.isinf(L("F1", 0.03).yaw_deg) and math.isinf(L("F0", 0.03).yaw_deg)


def test_limits_margin_shrinks_yaw_only_and_clamps_at_zero():
    L = fe().limits
    assert L("F2", 0.0, margin_deg=24.0).yaw_deg == pytest.approx(6.0)           # 2 x MAE(12 도)
    assert L("F2", 0.0, margin_deg=24.0).pitch_deg == pytest.approx(15.0)
    assert L("F2", 0.02, margin_deg=5.0).yaw_deg == pytest.approx(13.0 - 5.0, abs=0.02)
    assert L("F3", 0.0, margin_deg=40.0).yaw_deg == 0.0
    assert math.isinf(L("F1", 0.0, margin_deg=24.0).yaw_deg)
    with pytest.raises(ValueError):
        L("F2", 0.0, margin_deg=-1.0)


def test_limits_rejects_bad_input():
    L = fe().limits
    with pytest.raises(ValueError):
        L("F4")
    with pytest.raises(ValueError):
        L("f2")
    with pytest.raises(ValueError):
        L("F2", -0.01)
    with pytest.raises(ValueError):
        L("F2", float("nan"))


def test_limits_are_nested_f3_in_f2_in_f1_f0():
    L = fe().limits
    for sigma in (0.0, 0.006, 0.01, 0.02, 0.05):
        f0, f1, f2, f3 = (L(c, sigma) for c in ("F0", "F1", "F2", "F3"))
        assert f3.yaw_deg <= f2.yaw_deg <= f1.yaw_deg <= f0.yaw_deg
        assert f3.pitch_deg <= f2.pitch_deg <= f1.pitch_deg <= f0.pitch_deg
        assert f3.m_max <= f2.m_max <= f1.m_max <= f0.m_max
        assert set(f3.grades) <= set(f2.grades) <= set(f1.grades) <= set(f0.grades)
        assert f1.unobs_max <= f2.unobs_max            # F1 의 3% 가 가장 엄격
    for cat in ("F2", "F3"):
        ys = [L(cat, s).yaw_deg for s in (0.0, 0.006, 0.01, 0.02, 0.03, 0.1)]
        assert all(a >= b for a, b in zip(ys, ys[1:]))     # sigma 가 클수록 봉투는 줄기만 한다
        assert all(y <= L(cat, 0.0).yaw_deg + 1e-12 for y in ys)


def test_limits_is_frozen_dataclass():
    import dataclasses
    lim = fe().limits("F2")
    assert dataclasses.is_dataclass(lim)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lim.yaw_deg = 99.0
