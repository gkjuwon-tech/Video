"""판정선(녹·황·적), BT 가산성, NVF-obs·z_style, Wilson, Bradley-Terry."""
import math

import pytest
from tch07_support import l2b

from buc.teacher import tct as T


# ------------------------------------------------------------------ 엔진 등급 (models.upg §2.5)
@pytest.mark.parametrize("hc,obs,exp", [
    (0.60, 0.95, "green"), (0.75, 1.00, "green"),
    (0.5999, 0.99, "yellow"), (0.55, 0.99, "yellow"),
    (0.5499, 0.99, "red"), (0.30, 1.00, "red"),
    (0.65, 0.9499, "yellow"),                     # NVF-obs 미달은 녹색을 황색으로 내린다(적색으로는 안 내린다)
    (0.50, 0.50, "red"),
])
def test_grade_engine(hc, obs, exp):
    assert T.grade_engine(hc, obs) == exp


@pytest.mark.parametrize("bad", [(-0.1, 0.9), (1.1, 0.9), (0.6, 1.2), (float("nan"), 0.9)])
def test_grade_engine_rejects_bad_rates(bad):
    with pytest.raises(T.TctError):
        T.grade_engine(*bad)


@pytest.mark.parametrize("ev,exp", [
    (l2b(0.55, 1.5, 0.2, 0.2), "green"),
    (l2b(0.5499, 1.0), "yellow"), (l2b(0.50, 1.0), "yellow"), (l2b(0.4999, 1.0), "red"),
    (l2b(0.70, 1.51), "yellow"),                  # 스타일 z 초과
    (l2b(0.70, 1.0, dphoto=0.2001, tau=0.2), "yellow"),   # 사진풍 드리프트 초과
    (l2b(0.45, 5.0, dphoto=9.0, tau=0.2), "red"),
])
def test_grade_l2b(ev, exp):
    assert T.grade_l2b(ev) == exp


def test_grade_l2a():
    assert T.grade_l2a(z_style=1.5, outline_drift_px=1.0, palette_emd=0.3, palette_tau=0.3) == "green"
    assert T.grade_l2a(z_style=1.6, outline_drift_px=0.5, palette_emd=0.1, palette_tau=0.3) == "yellow"
    assert T.grade_l2a(z_style=1.0, outline_drift_px=1.2, palette_emd=0.1, palette_tau=0.3) == "yellow"
    assert T.grade_l2a(z_style=1.0, outline_drift_px=0.5, palette_emd=0.31, palette_tau=0.3) == "yellow"


def test_sp_i2v_logit_additivity():
    # logit(SP) = logit(NVF) + logit(TP); TP = 1 - NVF 이면 SP = 0.5
    for hc in (0.55, 0.60, 0.35, 0.8):
        assert T.required_tp(hc) == pytest.approx(1 - hc, abs=1e-12)
        assert T.sp_i2v(hc, 1 - hc) == pytest.approx(0.5, abs=1e-12)
    assert T.sp_i2v(0.6, 0.5) == pytest.approx(0.6, abs=1e-12)          # logit(0.5)=0
    got = T.sp_i2v(0.7, 0.4)
    want = 1 / (1 + math.exp(-(math.log(0.7 / 0.3) + math.log(0.4 / 0.6))))
    assert got == pytest.approx(want, abs=1e-12)
    assert T.sp_i2v(0.6, 0.45) > 0.5 > T.sp_i2v(0.6, 0.39)             # 문턱 TP ≥ 0.4
    for bad in ((0.0, 0.5), (0.5, 1.0), (1.2, 0.5)):
        with pytest.raises(T.TctError):
            T.sp_i2v(*bad)


# ------------------------------------------------------------------ 자동 지표
def test_nvf_obs_fraction():
    views = [(1.9, 0.5), (2.0, 1.0), (2.01, 0.5), (1.0, 1.01), (float("nan"), 0.0)]
    assert T.nvf_obs(views) == pytest.approx(2 / 5, abs=1e-12)       # 경계 포함, NaN은 불합격
    assert T.nvf_obs([(0.1, 0.1)] * 20) == 1.0
    with pytest.raises(T.TctError):
        T.nvf_obs([])


def test_z_style_and_self_tau():
    self_d = [1.0, 2.0, 3.0, 4.0, 5.0]                # mean 3, std(ddof=1) = sqrt(2.5)
    sd = math.sqrt(2.5)
    assert T.z_style(3.0 + 1.5 * sd, self_d) == pytest.approx(1.5, abs=1e-12)
    assert T.z_style(3.0, self_d) == pytest.approx(0.0, abs=1e-12)
    assert T.self_tau(self_d) == pytest.approx(2 * sd, abs=1e-12)
    for bad in ([1.0], [2.0, 2.0, 2.0]):
        with pytest.raises(T.TctError):
            T.z_style(1.0, bad)
        with pytest.raises(T.TctError):
            T.self_tau(bad)


# ------------------------------------------------------------------ Wilson
def test_wilson_known_values():
    r = T.wilson(8, 10)
    assert (r.k, r.n) == (8, 10) and r.p == pytest.approx(0.8, abs=1e-12)
    assert r.lo == pytest.approx(0.4902, abs=1e-3) and r.hi == pytest.approx(0.9433, abs=1e-3)
    r = T.wilson(0, 20)
    assert r.p == 0.0 and r.lo == 0.0 and 0.1 < r.hi < 0.2
    r = T.wilson(50, 100)
    assert r.lo == pytest.approx(0.4038, abs=1e-3) and r.hi == pytest.approx(0.5962, abs=1e-3)
    assert r.lo < r.p < r.hi
    for k, n in ((-1, 5), (6, 5), (0, 0)):
        with pytest.raises(T.TctError):
            T.wilson(k, n)


# ------------------------------------------------------------------ Bradley-Terry
def test_bt_two_items_closed_form():
    res = [("a", "b")] * 7 + [("b", "a")] * 3
    s = T.bradley_terry(["a", "b"], res)
    assert s["a"] - s["b"] == pytest.approx(math.log(7 / 3), abs=1e-6)
    assert s["a"] + s["b"] == pytest.approx(0.0, abs=1e-9)           # 평균 0 고정
    assert T.bt_prob(s, "a", "b") == pytest.approx(0.7, abs=1e-6)
    assert T.bt_prob(s, "a", "b") + T.bt_prob(s, "b", "a") == pytest.approx(1.0, abs=1e-12)


def test_bt_prior_shrinks():
    res = [("a", "b")] * 7 + [("b", "a")] * 3
    s = T.bradley_terry(["a", "b"], res, prior=1.0)                   # 비교된 쌍마다 양방향 0.5승 가상 관측
    assert T.bt_prob(s, "a", "b") == pytest.approx(7.5 / 11.0, abs=1e-6)


def test_bt_recovers_three_item_truth_and_is_order_free():
    true = {"x": 0.8, "y": 0.0, "z": -0.8}
    items = ["x", "y", "z"]
    res = []
    for a, b in (("x", "y"), ("y", "z"), ("x", "z")):
        p = 1 / (1 + math.exp(-(true[a] - true[b])))
        w = round(2000 * p)
        res += [(a, b)] * w + [(b, a)] * (2000 - w)
    s = T.bradley_terry(items, res)
    for i in items:
        assert s[i] == pytest.approx(true[i], abs=0.02)
    s2 = T.bradley_terry(items, list(reversed(res)))
    assert all(s[i] == pytest.approx(s2[i], abs=1e-9) for i in items)


def test_bt_errors():
    with pytest.raises(T.TctError):
        T.bradley_terry(["a", "b"], [("a", "b")] * 5)                 # b가 한 번도 못 이김: 식별 불가
    with pytest.raises(T.TctError):
        T.bradley_terry(["a", "b"], [("a", "c"), ("b", "a")])         # 모르는 항목
    with pytest.raises(T.TctError):
        T.bradley_terry(["a", "b"], [("a", "a"), ("b", "a")])         # 자기 자신과의 비교
    with pytest.raises(T.TctError):
        T.bradley_terry(["a", "b", "c"], [("a", "b"), ("b", "a")])    # c는 비교가 없다
    with pytest.raises(T.TctError):
        T.bradley_terry(["a", "a"], [("a", "a")])
