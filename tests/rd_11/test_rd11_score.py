"""RD-11 점수 합: Σ w·min(1, r/τ) 오라클, 단조성, 반사 가중 이월, 측정 못 한 성분, 띠·메뉴·카드 폴백."""
import math

import numpy as np
import pytest

import rd11_helpers as H
from buc.read import escher_score as E

ALL = H.ORDER


def esc_of(**kw):
    return E.escher_score(E.EscherInputs(**kw))


# ------------------------------------------------------------------ 닫힌 형태
def test_all_zero_is_zero_green_all_measured():
    s = E.escher_score(H.inputs_at(0.0))
    assert s.esc == 0.0 and s.band == "green" and s.dominant is None and s.pending == ()
    assert s.evidence == pytest.approx(1.0) and not s.refl_redirected and not s.cards_2_5d


def test_all_saturated_is_exactly_one_red():
    s = E.escher_score(H.inputs_at(1.0))
    assert s.esc == pytest.approx(1.0, abs=1e-12) and s.band == "red" and s.cards_2_5d
    s2 = E.escher_score(H.inputs_at(7.0))                       # 포화: 더 커져도 1
    assert s2.esc == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("frac", [0.05, 0.1, 0.25, 0.5, 0.9, 1.0])
def test_uniform_fraction_gives_that_fraction(frac):
    """전 성분이 τ·f 이고 모두 측정되면 가중치 합이 1 이므로 Esc = f."""
    assert E.escher_score(H.inputs_at(frac)).esc == pytest.approx(frac, abs=1e-12)


@pytest.mark.parametrize("c", ALL)
def test_single_component_saturation_equals_its_weight(c):
    inp = H.inputs_at(0.0)
    kw = {f: getattr(inp, f) for f in ("r_vp_deg", "orphan_frac", "size_sd", "plane_rel", "light_sd_deg", "cheat_share",
                                       "topo_count", "ffg_px1080", "r_dis", "r_refl_px1080")}
    kw[H.FIELD[c]] = H.TAU[c] * 3.0
    s = E.escher_score(E.EscherInputs(**kw))
    assert s.esc == pytest.approx(H.W[c], abs=1e-12), c
    assert s.dominant == c
    assert s.component(c).x == 1.0 and s.component(c).contrib == pytest.approx(H.W[c])
    for o in ALL:
        if o != c:
            assert s.component(o).contrib == 0.0


@pytest.mark.parametrize("c", ALL)
def test_linear_below_tau_and_clipped_above(c):
    base = {H.FIELD[k]: 0.0 for k in ALL}
    base["cheat_share"] = 0.0
    for f in (0.0, 0.2, 0.5, 0.99):
        kw = dict(base)
        kw[H.FIELD[c]] = H.TAU[c] * f
        assert E.escher_score(E.EscherInputs(**kw)).esc == pytest.approx(H.W[c] * f, abs=1e-12), (c, f)
    kw = dict(base)
    kw[H.FIELD[c]] = H.TAU[c] * 1.0000001
    assert E.escher_score(E.EscherInputs(**kw)).esc == pytest.approx(H.W[c], abs=1e-9)


# ------------------------------------------------------------------ 손으로 계산한 시나리오 3종
def test_scenario_clean_scene():
    inp = E.EscherInputs(r_vp_deg=0.3, orphan_frac=0.01, size_sd=0.02, plane_rel=0.005, light_sd_deg=2.0,
                         cheat_share=0.1, topo_count=0, r_dis=0.005)         # 반사 없음, 화해 전
    s = E.escher_score(inp)
    # vp (w .25: 반사 이월) .25*.15 + multi .15*.1 + size .15*.08 + plane .05*.1 + light .15*(2/15) + r_dis .1*.1
    assert s.esc == pytest.approx(0.0375 + 0.015 + 0.012 + 0.005 + 0.02 + 0.01, abs=1e-12)
    assert s.esc == pytest.approx(0.0995, abs=1e-12) and s.band == "green"
    assert s.pending == ("reconcile", "r_refl") and s.refl_redirected
    assert s.dominant == "vp"
    assert s.evidence == pytest.approx(0.95, abs=1e-12)            # 측정된 w_eff 합: 화해·반사 빼고 (VP 에 0.05 이월)


def test_scenario_half_broken_scene_is_yellow():
    inp = E.EscherInputs(r_vp_deg=1.0, orphan_frac=0.03, size_sd=0.08, plane_rel=0.02, light_sd_deg=3.0,
                         cheat_share=0.2, topo_count=0, r_dis=0.01, ffg_px1080=1.0, r_refl_px1080=1.0)
    s = E.escher_score(inp)
    want = (0.20 * 0.5 + 0.15 * 0.3 + 0.15 * 0.32 + 0.05 * 0.4 + 0.15 * 0.2 + 0.10 * 0.0 + 0.05 * 0.25 + 0.10 * 0.2
            + 0.05 * 0.25)
    assert want == pytest.approx(0.288, abs=1e-12)
    assert s.esc == pytest.approx(want, abs=1e-12) and s.band == "yellow"
    assert not s.refl_redirected and s.pending == () and s.dominant == "vp"


def test_scenario_escher_picture_is_red_and_triggers_cards():
    inp = E.EscherInputs(r_vp_deg=5.0, orphan_frac=0.25, size_sd=0.5, plane_rel=0.02, light_sd_deg=20.0,
                         cheat_share=0.5, topo_count=3, r_dis=0.12, ffg_px1080=8.0, r_refl_px1080=6.0)
    s = E.escher_score(inp)
    assert s.esc == pytest.approx(1.0 - 0.05 * 0.6, abs=1e-12)           # 평면만 40 %
    assert s.band == "red" and s.cards_2_5d
    assert s.component("plane").x == pytest.approx(0.4)


def test_random_inputs_match_independent_oracle():
    rng = np.random.default_rng(11)
    for _ in range(60):
        r = {}
        for c in ALL:
            r[c] = None if rng.random() < 0.25 else float(rng.random() * H.TAU[c] * 1.6)
        kw = {}
        for c in ALL:
            if r[c] is None or c == "light":
                continue
            kw[H.FIELD[c]] = r[c]
        if r["light"] is not None:
            kw["light_sd_deg"] = r["light"]
            kw["cheat_share"] = 0.0
        if r["multi_vp"] is not None:
            kw["orphan_frac"] = min(1.0, r["multi_vp"])
            r["multi_vp"] = kw["orphan_frac"]
        want, contrib = H.oracle_esc(r)
        s = E.escher_score(E.EscherInputs(**kw))
        assert s.esc == pytest.approx(want, abs=1e-12)
        for c in ALL:
            assert s.component(c).contrib == pytest.approx(contrib[c], abs=1e-12), c
        assert 0.0 <= s.esc <= 1.0


# ------------------------------------------------------------------ 규칙 1: 단조
def test_monotone_non_decreasing_in_every_component():
    rng = np.random.default_rng(5)
    for _ in range(40):
        base = {c: float(rng.random() * H.TAU[c] * 1.3) for c in ALL}
        base["multi_vp"] = min(base["multi_vp"], 1.0)
        for c in ALL:
            prev = -1.0
            for f in np.linspace(0.0, 2.0, 9):
                kw = {H.FIELD[k]: v for k, v in base.items()}
                kw["cheat_share"] = 0.0
                kw[H.FIELD[c]] = H.TAU[c] * f if c != "multi_vp" else min(1.0, H.TAU[c] * f)
                e = E.escher_score(E.EscherInputs(**kw)).esc
                assert e >= prev - 1e-15, (c, f)
                prev = e


def test_adding_a_measurement_never_lowers_esc_when_refl_present():
    """측정 못 한 성분은 0 으로 센다: 값을 더하면 Esc 는 줄지 않는다(하한 의미). 반사 이월이 없는 경로."""
    full = dict(r_vp_deg=1.0, orphan_frac=0.05, size_sd=0.1, plane_rel=0.02, light_sd_deg=5.0, cheat_share=0.0,
                topo_count=1, r_dis=0.02, ffg_px1080=1.0, r_refl_px1080=1.0)
    ref = E.escher_score(E.EscherInputs(**full)).esc
    for drop in ("size_sd", "plane_rel", "topo_count", "ffg_px1080", "r_dis", "orphan_frac"):
        part = {k: v for k, v in full.items() if k != drop}
        sp = E.escher_score(E.EscherInputs(**part))
        assert sp.esc <= ref + 1e-15 and len(sp.pending) == 1, drop


# ------------------------------------------------------------------ 측정 못 한 성분
def test_unmeasured_components_are_zero_and_listed_in_order():
    s = esc_of(r_vp_deg=2.0, r_dis=0.05, r_refl_px1080=4.0)
    assert s.pending == ("multi_vp", "size", "plane", "light", "topo", "reconcile")
    for n in s.pending:
        c = s.component(n)
        assert c.r is None and c.measured is False and c.w_eff == 0.0 and c.contrib == 0.0 and c.x == 0.0
    assert s.esc == pytest.approx(0.20 + 0.10 + 0.05, abs=1e-12)
    assert s.evidence == pytest.approx(0.20 + 0.10 + 0.05, abs=1e-12)


def test_nothing_measured_is_zero_with_everything_pending():
    s = esc_of()
    assert s.esc == 0.0 and s.pending == ALL and s.evidence == 0.0 and s.band == "green" and s.dominant is None


def test_light_measured_if_either_sd_or_cheat_given_missing_side_counts_zero():
    a = esc_of(light_sd_deg=7.5)
    b = esc_of(cheat_share=0.7)
    assert "light" not in a.pending and "light" not in b.pending
    assert a.component("light").r == pytest.approx(7.5)
    assert b.component("light").r == pytest.approx(15.0)                      # 15 * (0.7-0.35)/0.35
    assert b.esc == pytest.approx(0.15 * 1.0, abs=1e-12)


def test_light_composite_sd_plus_cheat_excess():
    for sd, share, want in ((4.0, 0.0, 4.0), (4.0, 0.35, 4.0), (4.0, 0.525, 4.0 + 7.5), (0.0, 1.0, 15.0 * 0.65 / 0.35)):
        s = esc_of(light_sd_deg=sd, cheat_share=share)
        assert s.component("light").r == pytest.approx(want, abs=1e-9)
        assert s.esc == pytest.approx(0.15 * min(1.0, want / 15.0), abs=1e-12)


# ------------------------------------------------------------------ 규칙 2: 반사 가중 이월
def test_no_reflection_region_moves_weight_to_vp():
    s = esc_of(r_vp_deg=2.0)
    assert s.refl_redirected and s.component("vp").w_eff == pytest.approx(0.25)
    assert s.component("r_refl").w_eff == 0.0 and s.component("r_refl").measured is False
    assert s.esc == pytest.approx(0.25, abs=1e-12)
    t = esc_of(r_vp_deg=2.0, r_refl_px1080=0.0)                                  # 반사 영역이 있고 잔차 0
    assert not t.refl_redirected and t.component("vp").w_eff == pytest.approx(0.20)
    assert t.esc == pytest.approx(0.20, abs=1e-12) and t.component("r_refl").measured is True


def test_redirected_weight_is_lost_when_vp_unmeasured():
    s = esc_of(r_dis=0.05)                       # 반사 없음, VP 도 없음
    assert s.refl_redirected and s.component("vp").w_eff == 0.0 and s.esc == pytest.approx(0.10, abs=1e-12)
    assert s.evidence == pytest.approx(0.10, abs=1e-12)


def test_vp_residual_counts_more_without_reflection_than_with():
    a = esc_of(r_vp_deg=1.0)
    b = esc_of(r_vp_deg=1.0, r_refl_px1080=0.0)
    assert a.esc == pytest.approx(0.125) and b.esc == pytest.approx(0.10)


# ------------------------------------------------------------------ 화해 비용(RD-12 이후)
def test_with_reconcile_adds_exactly_its_term():
    pre = E.escher_score(H.inputs_at(0.5, skip=("reconcile",)))
    assert pre.pending == ("reconcile",)
    for ffg in (0.0, 1.0, 4.0, 9.0):
        post = pre.with_reconcile(ffg)
        assert post.pending == () and post.component("reconcile").r == pytest.approx(ffg)
        assert post.esc == pytest.approx(pre.esc + 0.05 * min(1.0, ffg / 4.0), abs=1e-12)
        assert post.inputs.ffg_px1080 == pytest.approx(ffg)
    assert pre.pending == ("reconcile",) and pre.inputs.ffg_px1080 is None      # 원본 불변


def test_with_reconcile_replaces_previous_value_and_keeps_tau_scale():
    s = E.escher_score(H.inputs_at(0.0), tau_scale=2.0)
    t = s.with_reconcile(8.0)
    assert t.tau_scale == 2.0 and t.esc == pytest.approx(0.05 * 1.0, abs=1e-12)  # 8/(4*2)=1


def test_with_reconcile_validates():
    s = esc_of(r_dis=0.0)
    with pytest.raises(E.EscherError):
        s.with_reconcile(-1.0)
    with pytest.raises(E.EscherError):
        s.with_reconcile(float("nan"))


# ------------------------------------------------------------------ tau_scale (X-ESC 보정 훅)
def test_tau_scale_scales_every_threshold():
    inp = H.inputs_at(0.5)
    assert E.escher_score(inp, tau_scale=1.0).esc == pytest.approx(0.5)
    assert E.escher_score(inp, tau_scale=2.0).esc == pytest.approx(0.25, abs=1e-12)
    assert E.escher_score(inp, tau_scale=0.5).esc == pytest.approx(1.0, abs=1e-12)
    assert E.escher_score(inp, tau_scale=0.25).esc == pytest.approx(1.0, abs=1e-12)
    s = E.escher_score(inp, tau_scale=2.0)
    assert s.tau_scale == 2.0 and s.component("vp").tau == pytest.approx(4.0)


def test_tau_scale_monotone_decreasing():
    inp = H.inputs_at(0.8)
    vals = [E.escher_score(inp, tau_scale=k).esc for k in (0.5, 0.8, 1.0, 1.5, 3.0, 10.0)]
    assert all(a >= b - 1e-15 for a, b in zip(vals, vals[1:])) and vals[0] > vals[-1]


# ------------------------------------------------------------------ dominant
def test_dominant_is_largest_contribution_ties_go_first():
    s = esc_of(r_vp_deg=1.0, r_dis=0.05, size_sd=0.25)                    # contrib: vp .125, r_dis .10, size .15
    assert s.dominant == "size"
    t = esc_of(orphan_frac=1.0, size_sd=1.0)                             # 둘 다 정확히 .15 -> 앞선 multi_vp
    assert t.component("size").contrib == t.component("multi_vp").contrib and t.dominant == "multi_vp"


# ------------------------------------------------------------------ 띠·메뉴·카드 폴백
def test_band_edges():
    assert E.esc_band(0.0) == "green"
    assert E.esc_band(math.nextafter(0.15, 0.0)) == "green"
    assert E.esc_band(0.15) == "yellow" and E.esc_band(0.35) == "yellow"
    assert E.esc_band(math.nextafter(0.35, 1.0)) == "red" and E.esc_band(1.0) == "red"


@pytest.mark.parametrize("bad", [-0.001, 1.001, float("nan"), float("inf")])
def test_band_and_menu_reject_out_of_range(bad):
    with pytest.raises(E.EscherError):
        E.esc_band(bad)
    with pytest.raises(E.EscherError):
        E.menu_hint(bad)


def test_score_band_follows_esc():
    for f, b in ((0.05, "green"), (0.2, "yellow"), (0.35, "yellow"), (0.5, "red"), (0.9, "red")):
        s = E.escher_score(H.inputs_at(f))
        assert s.band == b == E.esc_band(s.esc), f


def test_menu_hint_levels():
    g = E.menu_hint(0.10)
    assert (g.band, g.safe, g.ext, g.full, g.full_warn, g.cards_2_5d) == ("green", True, True, True, False, False)
    y = E.menu_hint(0.20)
    assert (y.band, y.safe, y.ext, y.full, y.full_warn) == ("yellow", True, True, True, True)
    edge = E.menu_hint(0.35)
    assert edge.ext and edge.full and edge.full_warn and edge.band == "yellow"
    r = E.menu_hint(math.nextafter(0.35, 1.0))
    assert (r.band, r.safe, r.ext, r.full, r.full_warn, r.cards_2_5d) == ("red", True, False, False, False, False)
    assert E.menu_hint(0.15).full_warn is True and E.menu_hint(math.nextafter(0.15, 0.0)).full_warn is False


def test_cards_fallback_threshold_is_strictly_above_0_6():
    assert E.menu_hint(0.6).cards_2_5d is False
    assert E.menu_hint(math.nextafter(0.6, 1.0)).cards_2_5d is True
    assert E.escher_score(H.inputs_at(0.6)).cards_2_5d is False
    assert E.escher_score(H.inputs_at(0.61)).cards_2_5d is True
    assert E.menu_hint(1.0).safe is True and E.menu_hint(1.0).ext is False


def test_severity_family_orders_bands_end_to_end():
    """심각도 f 를 모든 잔차에 곱하면 Esc 가 f 로 올라가고 띠가 녹→황→적으로 바뀐다."""
    prev_band_rank, rank = -1, {"green": 0, "yellow": 1, "red": 2}
    prev = -1.0
    for f in np.linspace(0.0, 1.2, 25):
        s = E.escher_score(H.inputs_at(float(f)))
        assert s.esc >= prev - 1e-15 and rank[s.band] >= prev_band_rank
        assert s.esc == pytest.approx(min(f, 1.0), abs=1e-12)
        prev, prev_band_rank = s.esc, rank[s.band]
    assert prev_band_rank == 2
