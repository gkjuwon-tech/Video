"""SelectTeachers_v3: 엔진 잠금(비율 argmax + 2%p 동률 + 빠른 쪽), O1-O4 참조 경로, D15 Z 유지 경로, 결정론."""
import itertools
import json

import pytest
from tch07_support import KLEIN, Z, ZT4H, cand, gates, l2b, ring

from buc.teacher import tct as T
from buc.teacher.ref_slot import choose_l2b_path


def pick(cands, rr=None, **kw):
    rr = rr if rr is not None else ring([c.id for c in cands], nvf=0.60)     # 기본: Ring-R 녹색 -> O1
    return T.select_teachers(cands, rr, **kw)


# ------------------------------------------------------------------ 엔진 잠금
def test_only_incumbent_keeps_z():
    s = pick([cand(Z)])
    assert (s.engine, s.ref, s.outcome, s.path) == (Z, None, "O1", "ring_r")
    assert s.eligible == (Z,) and s.es_l1[Z] == 0.0


def test_unverified_klein_license_keeps_z_even_if_better():
    k = cand(KLEIN, d=0.30, t4h=0.8, license_ok=False)
    s = pick([cand(Z), k])
    assert s.engine == Z and s.eligible == (Z,)
    assert s.excluded[KLEIN] == ("license",)


def test_gate_failures_are_reported_per_candidate():
    k = cand(KLEIN, d=0.30, t4h=0.8, nvf_r=0.29, fp16_nan=2)
    s = pick([cand(Z), k])
    assert s.engine == Z and s.excluded[KLEIN] == ("fp16", "nvf_r_entry")


@pytest.mark.parametrize("d,t4h,engine", [
    (0.01, 1.20, KLEIN),     # 비율 1위 = klein, 2%p 안에서 더 빠르다
    (0.01, 1.50, Z),         # 비율 1위 = klein, 그러나 2%p 안의 Z가 더 빠르다
    (-0.02, 1.20, KLEIN),    # 경계 포함(|ΔES| <= 2%p): 더 빠른 klein
    (-0.0201, 1.20, Z),
    (-0.025, 1.00, Z),
    (0.03, 2.00, KLEIN),     # 2%p 밖: 느려도 ES가 크면 klein
    (0.03, 1.20, KLEIN),
])
def test_two_way_lock(d, t4h, engine):
    s = pick([cand(Z), cand(KLEIN, d=d, t4h=t4h)])
    assert s.engine == engine
    assert s.es_l1[KLEIN] == pytest.approx(d, abs=1e-12)


def test_ratio_argmax_not_max_es():
    # alt: ES .10 / 3.0 = .033, klein: .05 / 1.0 = .05 -> 비율 1위 klein, 밴드(.05차) 밖이라 klein
    s = pick([cand(Z), cand("alt", d=0.10, t4h=3.0), cand(KLEIN, d=0.05, t4h=1.0)])
    assert s.engine == KLEIN
    # alt를 1.8로 줄이면 .0556 > .05 -> alt
    s = pick([cand(Z), cand("alt", d=0.10, t4h=1.8), cand(KLEIN, d=0.05, t4h=1.0)])
    assert s.engine == "alt"
    assert s.score["alt"] == pytest.approx(0.10 / 1.8, abs=1e-12)


def test_fastest_in_band_of_ratio_winner():
    # tstar(비율 1위) 밴드 안의 eng가 더 빠르다
    s = pick([cand(Z), cand("tstar", 0.10, 1.0), cand("eng", 0.085, 0.99)])
    assert s.engine == "eng" and set(s.tie_band) == {"tstar", "eng"}


def test_no_positive_es_picks_max_es_then_band():
    # 전부 ES<=0: 비율은 의미가 없으므로 ES 최대(Z=0)를 기준으로 밴드
    s = pick([cand(Z), cand("a", -0.05, 0.5), cand("b", -0.015, 1.0)])
    assert s.engine == "b"                         # Z 밴드 안(−1.5%p)에서 가장 빠른 쪽
    s = pick([cand(Z), cand("a", -0.05, 0.5)])
    assert s.engine == Z


def test_z_fails_its_own_gate_then_best_other_wins():
    s = pick([cand(Z, nvf_r=0.2), cand(KLEIN, d=-0.30, t4h=1.0)])
    assert s.engine == KLEIN and s.excluded[Z] == ("nvf_r_entry",)


def test_nobody_eligible_or_missing_incumbent():
    with pytest.raises(T.SelectionError):
        pick([cand(Z, license_ok=False), cand(KLEIN, license_ok=False)])
    with pytest.raises(T.SelectionError):
        pick([cand(KLEIN, 0.1, 1.0)])
    with pytest.raises(T.TctError):
        pick([cand(Z), cand(Z)])
    with pytest.raises(T.TctError):
        pick([])
    with pytest.raises(T.TctError):
        pick([cand(Z, t4h=0.0)])


def test_order_invariance_and_determinism():
    cs = [cand(Z), cand("alt", 0.10, 1.8), cand(KLEIN, 0.05, 1.0), cand("eng", 0.085, 0.99)]
    base = pick(cs).to_dict()
    assert json.dumps(base, sort_keys=True) == json.dumps(pick(cs).to_dict(), sort_keys=True)
    for perm in itertools.permutations(cs):
        assert pick(list(perm)).to_dict() == base


# ------------------------------------------------------------------ 방화벽·사전 등록
@pytest.mark.parametrize("tag", ["COMP", "O-photo-home", "S-closed", "nonsense"])
def test_firewall_blocks_selection(tag):
    with pytest.raises(T.FirewallError):
        pick([cand(Z), cand(KLEIN, 0.1, 1.0, sources=("R-real", tag))])


def test_prereg_key_must_match():
    cs = [cand(Z)]
    pick(cs, expected_prereg_key=T.PREREG.key())
    other = T.PreRegistration(w_l1=(0.10, 0.10, 0.20, 0.20, 0.10, 0.15, 0.15))
    with pytest.raises(T.PreRegistrationError):
        pick(cs, prereg=other, expected_prereg_key=T.PREREG.key())
    s = pick(cs, prereg=other)                      # 키를 안 주면 통과하되 보고서에 키가 남는다
    assert s.prereg_key == other.key() != T.PREREG.key()
    assert pick(cs).prereg_key == T.PREREG.key()
    with pytest.raises(T.PreRegistrationError):
        pick(cs, prereg=T.PreRegistration(w_l1=(0.2,) * 7))


def test_l2b_scores_use_l2b_weights():
    # K1=+10%, 나머지 0 -> ES_L1 = 0.01, ES_L2b = 0.03
    m = {k: 1.0 for k in T.CRITERIA}
    m["K1"] = 1.1
    k = T.Candidate(id=KLEIN, metrics=m, t4h_world=1.2, gates=gates(), sources=("R-real",))
    s = pick([cand(Z), k])
    assert s.es_l1[KLEIN] == pytest.approx(0.01, abs=1e-12)
    assert s.es_l2b[KLEIN] == pytest.approx(0.03, abs=1e-12)


# ------------------------------------------------------------------ O1-O4
def refc(cid, d, t4h, nvf, cost, **kw):
    return cand(cid, d, t4h, ref=True, swap=True, nvf_s=nvf, ref_cost=cost, **kw)


def test_o1_ring_r_green_needs_no_reference():
    s = pick([cand(Z), refc(KLEIN, -0.03, 1.2, 0.9, 0.3)], ring([Z, KLEIN], nvf=0.56))
    assert (s.engine, s.ref, s.outcome, s.path) == (Z, None, "O1", "ring_r")


def test_o2_swappable_reference_slot():
    s = pick([cand(Z), refc(KLEIN, -0.03, 1.2, 0.56, 0.3)], ring([Z, KLEIN], nvf=0.50))
    assert (s.engine, s.ref, s.outcome, s.path) == (Z, KLEIN, "O2", "klein_ref")
    assert s.path == choose_l2b_path("L2b", ring_r_green=False, klein_ok=True, adapter_ok=True)


@pytest.mark.parametrize("nvf_s,ok", [(0.55, True), (0.549, False)])
def test_reference_threshold_is_plus_5pp_inclusive(nvf_s, ok):
    s = pick([cand(Z), refc(KLEIN, -0.03, 1.2, nvf_s, 0.3)], ring([Z, KLEIN], nvf=0.50))
    assert (s.ref == KLEIN) is ok
    assert s.outcome == ("O2" if ok else "O4")


@pytest.mark.parametrize("kw", [dict(swap=False), dict(ref=False), dict(nvf_s=None), dict(license_ok=False)])
def test_reference_must_satisfy_contract(kw):
    base = dict(ref=True, swap=True, nvf_s=0.90, ref_cost=0.3)
    base.update({k: v for k, v in kw.items() if k in base})
    gk = {k: v for k, v in kw.items() if k not in base}
    s = pick([cand(Z), cand(KLEIN, -0.03, 1.2, **base, **gk)], ring([Z, KLEIN], nvf=0.40))
    assert s.ref is None and s.outcome == "O4"


def test_o4_ref_adapter_and_cards_fallback():
    s = pick([cand(Z)], ring([Z], nvf=0.40))
    assert (s.outcome, s.path, s.ref) == ("O4", "ref_adapter", None)
    s = pick([cand(Z)], ring([Z], nvf=0.40), adapter_ok=False)
    assert (s.outcome, s.path) == ("O4", "cards_2_5d")
    assert s.path == choose_l2b_path("L2b", ring_r_green=False, klein_ok=False, adapter_ok=False)


def test_best_reference_by_gain_per_cost():
    cs = [cand(Z), refc("r1", -0.03, 1.2, 0.56, 0.3), refc("r2", -0.03, 1.2, 0.60, 1.0)]
    s = pick(cs, ring([Z, "r1", "r2"], nvf=0.50))
    assert s.ref == "r1"                              # .06/.3 = .2 > .10/1.0 = .1
    cs[1] = refc("r1", -0.03, 1.2, 0.56, 0.7)
    assert pick(cs, ring([Z, "r1", "r2"], nvf=0.50)).ref == "r2"


def three_way(es_ref, t4h_ref=0.9):
    return [cand(Z), cand("tstar", 0.10, 1.0), cand("eng", 0.085, 0.99),
            refc("refc", es_ref, t4h_ref, 0.60, 0.3)]


def test_o3_single_teacher_swap_when_reference_is_within_tie_and_faster():
    s = pick(three_way(0.066), ring([Z, "tstar", "eng", "refc"], nvf=0.40))
    assert (s.engine, s.ref, s.outcome, s.path) == ("refc", None, "O3", "klein_ref")


def test_o2_when_reference_is_outside_tie_of_engine():
    s = pick(three_way(0.0649), ring([Z, "tstar", "eng", "refc"], nvf=0.40))
    assert (s.engine, s.ref, s.outcome) == ("eng", "refc", "O2")


def test_o2_when_reference_is_not_faster_than_engine():
    s = pick(three_way(0.066, t4h_ref=1.05), ring([Z, "tstar", "eng", "refc"], nvf=0.40))
    assert (s.engine, s.ref, s.outcome) == ("eng", "refc", "O2")


def test_ring_r_evidence_required_for_engine():
    with pytest.raises(T.TctError):
        pick([cand(Z)], {})
    with pytest.raises(T.TctError):
        pick([cand(Z)], {Z: "green"})


def test_ref_cost_must_be_positive_when_used():
    with pytest.raises(T.TctError):
        pick([cand(Z), refc(KLEIN, -0.03, 1.2, 0.60, 0.0)], ring([Z, KLEIN], nvf=0.40))


# ------------------------------------------------------------------ 잠금
def test_lock_is_frozen_and_stable():
    s = pick([cand(Z), refc(KLEIN, -0.03, 1.2, 0.60, 0.3)], ring([Z, KLEIN], nvf=0.40))
    a, b = s.lock(), s.lock()
    assert a == b and a.key == b.key and len(a.key) == 64
    assert (a.engine, a.ref, a.outcome) == (Z, KLEIN, "O2")
    with pytest.raises(Exception):
        a.engine = KLEIN
    s2 = pick([cand(Z)], ring([Z], nvf=0.40))
    assert s2.lock().key != a.key
    assert json.dumps(s.to_dict())                      # JSON 직렬화 가능
