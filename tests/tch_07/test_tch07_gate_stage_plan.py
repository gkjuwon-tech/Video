"""COMP 천장 게이트(잠금 뒤에만), 등록 실패 편향 처리, 교체 창, 시험 계획·비용(models.upg §2.3, §2.6)."""
import pytest
from tch07_support import Z, cand, ring

from buc.teacher import tct as T
from buc.teacher.sample import DEFAULT_TABLE
from buc.teacher.ref_slot import KLEIN_TABLE


def locked():
    return T.select_teachers([cand(Z)], ring([Z], nvf=0.60)).lock()


def recs(spec):
    """spec: [(registered, wins, losses, grade)]"""
    out, i = [], 0
    for registered, w, l, g in spec:
        for win in [True] * w + [False] * l:
            out.append(T.CompRecord(input_id=f"i{i}", grade=g, registered=registered, win=win))
            i += 1
    return out


# ------------------------------------------------------------------ COMP 게이트
def test_comp_requires_locked_config():
    with pytest.raises(T.TctError):
        T.comp_gate("z-image-turbo", recs([(True, 6, 4, "green")]), nvf_obs_rate=0.97)
    with pytest.raises(T.TctError):
        T.comp_gate(locked(), [], nvf_obs_rate=0.97)


def test_comp_primary_vs_fallback_and_conservative_grade():
    # 등록 성공 10: 6.2승이 필요하지만 정수로 62%에 가까운 25 중 16승 = 0.64, 폴백 포함 40 중 24승 = 0.60 -> 0.60
    r = T.comp_gate(locked(), recs([(True, 16, 9, "green"), (False, 8, 7, "yellow")]), nvf_obs_rate=0.97)
    assert r.n == 40 and r.n_registered == 25
    assert r.nvf_hc_primary == pytest.approx(16 / 25, abs=1e-12)
    assert r.nvf_hc_fallback == pytest.approx(24 / 40, abs=1e-12)
    assert r.nvf_hc == pytest.approx(0.60, abs=1e-12) and r.grade == "green"
    assert r.reg_fail_rate == pytest.approx(15 / 40, abs=1e-12) and r.reg_fail_flag is True      # > 20%


def test_comp_failed_registrations_can_downgrade():
    # 등록 성공만 보면 0.62(녹색)이지만 폴백 포함은 0.55 보다 낮아 적색이면 보수적 등급을 따른다
    r = T.comp_gate(locked(), recs([(True, 31, 19, "green"), (False, 5, 25, "red")]), nvf_obs_rate=0.97)
    assert r.nvf_hc_primary == pytest.approx(0.62, abs=1e-12)
    assert r.nvf_hc_fallback == pytest.approx(36 / 80, abs=1e-12)
    assert r.grade == "red" and r.needs_human_q3 is True


def test_comp_small_failure_rate_not_flagged_and_obs_downgrade():
    r = T.comp_gate(locked(), recs([(True, 30, 20, "green"), (False, 6, 4, "green")]), nvf_obs_rate=0.97)
    assert r.reg_fail_rate == pytest.approx(10 / 60, abs=1e-12) and r.reg_fail_flag is False
    assert r.nvf_hc == pytest.approx(min(30 / 50, 36 / 60), abs=1e-12) and r.grade == "green"
    r2 = T.comp_gate(locked(), recs([(True, 30, 20, "green"), (False, 6, 4, "green")]), nvf_obs_rate=0.90)
    assert r2.grade == "yellow" and r2.needs_human_q3 is False


def test_comp_all_unregistered_has_no_primary():
    r = T.comp_gate(locked(), recs([(False, 7, 3, "green")]), nvf_obs_rate=0.99)
    assert r.nvf_hc_primary is None and r.nvf_hc == pytest.approx(0.7, abs=1e-12) and r.reg_fail_flag is True


def test_comp_by_grade_table_is_published():
    r = T.comp_gate(locked(), recs([(True, 9, 1, "green"), (True, 5, 5, "yellow"), (True, 2, 8, "red")]),
                    nvf_obs_rate=0.99)
    assert set(r.by_grade) == {"green", "yellow", "red"}
    assert r.by_grade["green"] == pytest.approx(0.9, abs=1e-12)
    assert r.by_grade["red"] == pytest.approx(0.2, abs=1e-12)
    assert r.to_dict()["engine"] == Z


def test_comp_record_validation():
    with pytest.raises(T.TctError):
        T.CompRecord(input_id="x", grade="purple", registered=True, win=True).validate()
    with pytest.raises(T.TctError):
        T.comp_gate(locked(), [T.CompRecord(input_id="x", grade="blue", registered=True, win=True)], nvf_obs_rate=0.9)


# ------------------------------------------------------------------ 교체 창
@pytest.mark.parametrize("day,kind,delta,ok", [
    (45, "engine", 0.05, True), (45, "engine", 0.0499, False), (45, "engine", 0.20, True),
    (75, "engine", 0.50, False), (60, "engine", 0.50, False), (89, "engine", 0.50, False), (15, "engine", 0.5, False),
    (45, "ref", 0.05, True), (75, "ref", 0.05, True), (75, "ref", 0.049, False),
    (60, "ref", 0.5, False), (30, "ref", 0.5, False), (90, "ref", 0.5, False),
])
def test_can_replace(day, kind, delta, ok):
    assert T.can_replace(day, kind, delta) is ok


def test_can_replace_validation():
    for args in ((45, "adapter", 0.1), (45.0, "engine", 0.1), (True, "engine", 0.1), (45, "engine", float("nan"))):
        with pytest.raises(T.TctError):
            T.can_replace(*args)


# ------------------------------------------------------------------ 시험 계획·비용
def test_arm_registry():
    assert set(T.FILL_ARMS) == {"Z-plain", "Z-Ring", "klein-plain", "klein-ref", "Z-Ring+theta"}
    assert "Z-RingR" in T.ARMS and "Z-RingR" not in T.FILL_ARMS
    assert {a: T.arm_seeds(a) for a in T.ARMS} == {"Z-plain": 1, "Z-Ring": 2, "klein-plain": 1, "klein-ref": 2,
                                                  "Z-Ring+theta": 1, "Z-RingR": 1}
    # NFE는 TCH-02/TCH-08의 σ 표 길이에서 온다(Ring-R은 8 -> 10)
    assert T.ARMS["Z-plain"].nfe == len(DEFAULT_TABLE.points) == 8
    assert T.ARMS["klein-plain"].nfe == len(KLEIN_TABLE.points) == 4
    assert T.ARMS["Z-RingR"].nfe == 10
    # 포즈당 TFU = NFE x 1 fwd TFU (표 값과 0.05 이내)
    assert T.ARMS["Z-plain"].tfu == pytest.approx(8.0, abs=1e-9)
    assert T.ARMS["Z-Ring"].tfu == pytest.approx(9.1, abs=1e-9)
    assert T.ARMS["klein-plain"].tfu == pytest.approx(2.6, abs=1e-9)
    assert T.ARMS["klein-ref"].tfu == pytest.approx(4.0, abs=1e-9)
    assert T.ARMS["Z-RingR"].tfu == pytest.approx(11.4, abs=1e-9)
    assert T.ARMS["Z-Ring"].nfe * 1.14 == pytest.approx(T.ARMS["Z-Ring"].tfu, abs=0.05)
    assert T.ARMS["klein-ref"].nfe * 1.0 == pytest.approx(T.ARMS["klein-ref"].tfu, abs=0.05)
    with pytest.raises(T.TctError):
        T.arm_seeds("nope")


def test_fill_cost_matches_catalog_numbers():
    assert T.N_INPUTS == 330 and T.N_POSES_PER_INPUT == 3
    n = T.N_INPUTS * T.N_POSES_PER_INPUT
    assert n == 990
    raw = T.fill_cost_tfu(n, w4a4=False)
    assert raw == pytest.approx(990 * (2 * 9.1 + 8.0 + 2 * 4.0 + 2.6 + 9.1), abs=1e-6)     # 45,441 TFU
    c = T.fill_cost_tfu(n)
    assert c == pytest.approx(40_059.5, abs=1.0)                                            # "4.0만 TFU"
    assert T.tfu_to_t4h(c) == pytest.approx(28.92, abs=0.05)                                # "약 29 T4-h"
    assert T.W4A4_FACTOR == 0.846
    assert T.fill_cost_tfu(0) == 0.0
    with pytest.raises(T.TctError):
        T.fill_cost_tfu(-1)


def test_lite_plan_constants():
    assert (T.LITE_INPUTS, T.LITE_POSES, T.LITE_SEEDS) == (160, 3, 1)
    assert T.LITE_ARMS == ("Z-Ring", "klein-ref")
