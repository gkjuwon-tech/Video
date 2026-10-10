"""예산 캡: 임대 전 거절, 여유(margin) 경계에서의 정지, 다음 UTC일 재개, 누적 잠금, gpu_h 캡."""
import pytest
from run01_fakes import DAY0, H, raw_offer

from buc.core.config import Config
from buc.core.ledger import BudgetCheck
from buc.rt import vast as V

TOL = 1e-9


def _rent_running(orch, api):
    iid = orch.rent()
    api.set_status(iid, actual="running")
    return iid


def test_rent_refused_by_daily_cap_before_create(api, make):
    with pytest.raises(V.BudgetRefused) as ei:
        make(daily=0.3).rent()  # 0.40 × 1 h 예약 > 0.30
    assert ei.value.reason == "daily"
    assert isinstance(ei.value.check, BudgetCheck) and ei.value.check.ok is False
    assert api.n("create_instance") == 0
    iid = make(daily=0.3, reserve_hours=0.5).rent()  # 0.20 ≤ 0.30
    assert iid in api.instances


def test_rent_refused_by_total_cap(api, make):
    with pytest.raises(V.BudgetRefused) as ei:
        make(total=0.39).rent()
    assert ei.value.reason == "total"
    assert api.n("create_instance") == 0


def test_rent_refused_by_gpu_hour_cap(api, make):
    cfg = Config(budget_daily_usd=100.0, budget_total_usd=1000.0, budget_total_gpu_h=0.5)
    with pytest.raises(V.BudgetRefused) as ei:
        make(config=cfg).rent()
    assert ei.value.reason == "total"
    assert api.n("create_instance") == 0


@pytest.mark.parametrize("hours,stop", [(2.2, False), (2.3, True)])
def test_total_cap_margin_boundary(api, ledger, make, clock, hours, stop):
    # 누적 캡 1.00, 0.40 $/h, 여유 0.25 h → 0.10 $. 2.2 h: 0.88+0.10=0.98 (유지) / 2.3 h: 0.92+0.10=1.02 (정지)
    orch = make(total=1.0, cap_margin_h=0.25)
    t = clock()
    iid = _rent_running(orch, api)
    clock.set(t + 1 * H)
    orch.tick()
    clock.set(t + hours * H)
    rep = orch.tick()
    assert rep.total_usd == pytest.approx(0.40 * hours, abs=TOL)
    if stop:
        assert rep.actions == [V.Action("stop", iid, "cap_total")]
        assert api.instances[iid]["intended_status"] == "stopped"
        assert rep.locked is False
        with pytest.raises(V.BudgetRefused) as ei:
            orch.start(iid)  # 0.92 + 0.40 > 1.00
        assert ei.value.reason == "total"
        assert api.instances[iid]["intended_status"] == "stopped"
    else:
        assert rep.actions == []
        assert api.instances[iid]["intended_status"] == "running"


def test_daily_cap_stops_then_next_day_resumes(api, ledger, make, clock):
    clock.set(DAY0 + 20 * H)
    orch = make(daily=1.0, total=100.0, cap_margin_h=0.25)
    iid = _rent_running(orch, api)
    clock.set(DAY0 + 22.3 * H)
    rep = orch.tick()
    assert rep.daily_usd == pytest.approx(0.92, abs=TOL)
    assert rep.actions == [V.Action("stop", iid, "cap_daily")]
    with pytest.raises(V.BudgetRefused) as ei:
        orch.start(iid)
    assert ei.value.reason == "daily"
    n_set = api.n("set_state")
    clock.set(DAY0 + 24.5 * H)
    rep = orch.tick()
    # 22.3 h → 24.5 h 정지 요율 0.02: 전날 1.7 h, 오늘 0.5 h
    assert ledger.daily_total("usd", ts=DAY0 + 1) == pytest.approx(0.92 + 1.7 * 0.02, abs=TOL)
    assert rep.daily_usd == pytest.approx(0.5 * 0.02, abs=TOL)
    assert rep.actions == []
    orch.start(iid)
    assert api.n("set_state") == n_set + 1
    assert api.instances[iid]["intended_status"] == "running"


def test_total_exhausted_locks_everything(api, ledger, make, clock):
    orch = make(total=0.5)
    t = clock()
    iid = _rent_running(orch, api)
    clock.set(t + 2 * H)  # 드문 tick: 실제 지출 0.80은 그대로 기록한다(현실을 숨기지 않는다)
    rep = orch.tick()
    assert ledger.total("usd") == pytest.approx(0.80, abs=TOL)
    assert rep.actions == [V.Action("stop", iid, "cap_total")]
    assert rep.locked is True
    n_set, n_search = api.n("set_state"), api.n("search_offers")
    with pytest.raises(V.BudgetRefused) as ei:
        orch.start(iid)
    assert ei.value.reason == "locked"
    with pytest.raises(V.BudgetRefused) as ei:
        orch.rent()
    assert ei.value.reason == "locked"
    assert api.n("set_state") == n_set and api.n("create_instance") == 1 and api.n("search_offers") == n_search
    clock.set(t + 30 * H)  # 날이 바뀌어도 누적 잠금은 풀리지 않는다
    assert orch.tick().locked is True
    with pytest.raises(V.BudgetRefused):
        orch.start(iid)


def test_cap_stops_all_running_managed(api, ledger, make, clock):
    api.offers = [raw_offer(1, dph=0.40), raw_offer(2, dph=0.45)]
    orch = make(total=2.0, max_instances=2, cap_margin_h=0.25)
    t = clock()
    a = _rent_running(orch, api)
    b = _rent_running(orch, api)
    assert {api.instances[a]["dph_total"], api.instances[b]["dph_total"]} == {0.40, 0.45}
    clock.set(t + 2.2 * H)  # 0.85 × 2.2 = 1.87; +0.85×0.25 = 2.0825 > 2.0
    rep = orch.tick()
    assert rep.burn_usd_per_h == pytest.approx(0.85, abs=TOL)
    assert sorted(rep.actions, key=lambda x: x.instance_id) == sorted(
        [V.Action("stop", a, "cap_total"), V.Action("stop", b, "cap_total")], key=lambda x: x.instance_id)
    assert api.instances[a]["intended_status"] == api.instances[b]["intended_status"] == "stopped"


def test_both_caps_reports_total(api, make, clock):
    orch = make(daily=0.5, total=0.5)
    t = clock()
    iid = _rent_running(orch, api)
    clock.set(t + 1.2 * H)
    rep = orch.tick()
    assert rep.actions == [V.Action("stop", iid, "cap_total")]
