"""네 게이트(라이선스·FP16·메모리·NVF-R 입장), klein 라이선스 연동(TCH-08), 선정 입력 방화벽."""
import math

import pytest
from tch07_support import gates

from buc.teacher import tct as T
from buc.teacher.ref_slot import KleinSpec


def test_all_pass():
    r = T.gate_report(gates())
    assert r.ok is True and r.failed == ()


@pytest.mark.parametrize("kw,name", [
    (dict(license_ok=False), "license"),
    (dict(fp16_nan=1), "fp16"),
    (dict(fp16_ref_nan=1), "fp16"),
    (dict(fp16_trials=9_999), "fp16"),          # 시행 수 부족도 실패(미측정을 통과로 치지 않는다)
    (dict(fp16_ref_trials=1_999), "fp16"),
    (dict(mem_peak_gib=15.01), "memory"),
    (dict(host_completed=False), "memory"),
    (dict(nvf_r=0.2999), "nvf_r_entry"),
    (dict(nvf_r=float("nan")), "nvf_r_entry"),
    (dict(mem_peak_gib=float("nan")), "memory"),
])
def test_each_gate_fails_alone(kw, name):
    r = T.gate_report(gates(**kw))
    assert r.ok is False and r.failed == (name,)


def test_gate_boundaries_inclusive():
    assert T.gate_report(gates(nvf_r=0.30)).ok
    assert T.gate_report(gates(mem_peak_gib=15.0)).ok
    assert T.gate_report(gates(fp16_trials=10_000, fp16_ref_trials=2_000)).ok


def test_failed_names_are_ordered():
    r = T.gate_report(gates(nvf_r=0.1, license_ok=False, mem_peak_gib=20.0, fp16_nan=3))
    assert r.failed == ("license", "fp16", "memory", "nvf_r_entry")


def test_klein_license_gate_uses_tch08_spec():
    assert T.klein_license_gate(KleinSpec()).ok is False                       # 사람 확인 플래그 없음
    g = T.klein_license_gate(KleinSpec(license_verified=True))
    assert g.ok is True
    for repo in ("black-forest-labs/FLUX.2-klein-9B", "someone/flux.2-KLEIN-9b-int8"):
        r = T.klein_license_gate(KleinSpec(repo=repo, license_verified=True, unofficial_ok=True))
        assert r.ok is False and isinstance(r.reason, str) and r.reason
    r = T.klein_license_gate(KleinSpec(repo="someone/klein-copy", license_verified=True))
    assert r.ok is False                                                       # 제3자 파생물


@pytest.mark.parametrize("tags", [
    ("R-real",), ("R-real", "R-syn", "O-photo-dev", "O-style-2a", "O-style-2b", "E-escher"), (),
])
def test_firewall_allows_selection_sets(tags):
    T.check_selection_inputs(tags)


@pytest.mark.parametrize("tag", ["COMP", "O-photo-home", "O-style-home", "S-closed", "HOLDOUT", "USER", "mystery"])
def test_firewall_blocks_everything_else(tag):
    with pytest.raises(T.FirewallError):
        T.check_selection_inputs(("R-real", tag))


def test_firewall_tag_sets_are_disjoint_and_complete():
    assert not (set(T.SELECTION_SETS) & set(T.FORBIDDEN_SELECTION_SETS))
    assert {"COMP", "O-photo-home", "O-style-home", "S-closed"} <= set(T.FORBIDDEN_SELECTION_SETS)
    assert {"R-real", "R-syn", "O-photo-dev", "O-style-2a", "O-style-2b", "E-escher"} <= set(T.SELECTION_SETS)


def test_home_away_gap_and_seam():
    gap, flag = T.home_away_gap(0.58, 0.52)
    assert gap == pytest.approx(0.06, abs=1e-12) and flag is True
    gap, flag = T.home_away_gap(0.55, 0.50)
    assert gap == pytest.approx(0.05, abs=1e-12) and flag is False            # 5%p 정확히는 경고 아님(>5%p)
    assert T.home_away_gap(0.5, 0.56)[0] == pytest.approx(0.06, abs=1e-12)     # 절댓값
    assert T.seam_ok(1.3, 1.0) is True and T.seam_ok(1.31, 1.0) is False
    with pytest.raises(T.TctError):
        T.seam_ok(1.0, 0.0)
    assert math.isfinite(T.SEAM_RATIO_MAX) and T.SEAM_RATIO_MAX == 1.3
