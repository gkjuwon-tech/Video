"""계량: 금액, 멱등, 재시작, UTC 자정 분할, 정지 요율, 외부 소멸, 미관리 인스턴스."""
import pytest
from run01_fakes import DAY0, H, raw_offer

from buc.core.ledger import utc_day_bounds
from buc.rt import vast as V

TOL = 1e-9


def _rent_running(orch, api, clock):
    iid = orch.rent()
    api.set_status(iid, actual="running")
    return iid


def test_running_billing_amounts_and_gpu_hours(api, ledger, make, clock):
    orch = make()
    t = clock()
    iid = _rent_running(orch, api, clock)
    clock.set(t + 1800)
    rep = orch.tick()
    assert isinstance(rep, V.TickReport)
    assert ledger.total("usd") == pytest.approx(0.20, abs=TOL)
    assert ledger.total("gpu_h") == pytest.approx(0.5, abs=TOL)
    assert rep.metered_usd == pytest.approx(0.20, abs=TOL)
    assert rep.total_usd == pytest.approx(0.20, abs=TOL)
    assert rep.burn_usd_per_h == pytest.approx(0.40, abs=TOL)
    assert rep.actions == [] and rep.errors == [] and rep.locked is False
    clock.set(t + 3600)
    orch.tick()
    assert ledger.total("usd") == pytest.approx(0.40, abs=TOL)
    assert ledger.total("usd", run_id=f"vast:{iid}") == pytest.approx(0.40, abs=TOL)
    seg = [e for e in ledger.events(kind="usd", run_id=f"vast:{iid}") if e.amount > 0]
    assert [e.ts for e in seg] == pytest.approx([t, t + 1800])
    assert all(e.idem_key and e.idem_key.startswith(f"vast:{iid}:") for e in seg)
    assert seg[0].meta["t0"] == pytest.approx(t) and seg[0].meta["t1"] == pytest.approx(t + 1800)


def test_tick_is_idempotent_at_same_time(api, ledger, make, clock):
    orch = make()
    t = clock()
    _rent_running(orch, api, clock)
    clock.set(t + 2 * H)
    orch.tick()
    n = ledger.count()
    total = ledger.total("usd")
    rep = orch.tick()
    orch.tick(now=t + 2 * H)
    assert ledger.count() == n
    assert ledger.total("usd") == total
    assert rep.metered_usd == pytest.approx(0.0, abs=TOL)


def test_restart_does_not_double_bill(api, ledger, make, clock):
    orch = make()
    t = clock()
    iid = _rent_running(orch, api, clock)
    clock.set(t + H)
    orch.tick()
    orch2 = make()  # 프로세스 재시작: 같은 장부, 새 객체(메모리 상태 없음)
    clock.set(t + 2 * H)
    orch2.tick()
    orch2.tick()
    assert ledger.total("usd", run_id=f"vast:{iid}") == pytest.approx(0.80, abs=TOL)
    orch3 = make()
    clock.set(t + 2.5 * H)
    orch3.tick()
    assert ledger.total("usd") == pytest.approx(1.00, abs=TOL)


def test_segment_split_at_utc_midnight(api, ledger, make, clock):
    api.offers = [raw_offer(1, dph=1.0)]
    clock.set(DAY0 + 23.5 * H)
    orch = make()
    iid = orch.rent(query=V.OfferQuery(max_dph=2.0))
    api.set_status(iid, actual="running")
    clock.set(DAY0 + 24.5 * H)
    orch.tick()
    assert ledger.daily_total("usd", ts=DAY0 + 12 * H) == pytest.approx(0.5, abs=TOL)
    assert ledger.daily_total("usd", ts=DAY0 + 30 * H) == pytest.approx(0.5, abs=TOL)
    assert ledger.total("usd") == pytest.approx(1.0, abs=TOL)
    pos = [e for e in ledger.events(kind="usd") if e.amount > 0]
    assert len(pos) == 2
    for e in pos:  # 조각은 하루를 넘지 않는다
        lo, hi = utc_day_bounds(e.meta["t0"])
        assert lo <= e.meta["t0"] and e.meta["t1"] <= hi + TOL


def test_stopped_rate_after_stop(api, ledger, make, clock):
    orch = make()
    t = clock()
    iid = _rent_running(orch, api, clock)
    clock.set(t + H)
    orch.tick()
    orch.stop(iid)
    assert api.instances[iid]["intended_status"] == "stopped"
    clock.set(t + 3 * H)
    orch.tick()
    # 1 h × 0.40 + 2 h × 0.02 (디스크)
    assert ledger.total("usd") == pytest.approx(0.44, abs=TOL)
    assert ledger.total("gpu_h") == pytest.approx(1.0, abs=TOL)
    rep = orch.tick()
    assert rep.burn_usd_per_h == pytest.approx(0.0, abs=TOL)


def test_stop_meters_up_to_now(api, ledger, make, clock):
    orch = make()
    t = clock()
    _iid = _rent_running(orch, api, clock)
    clock.set(t + 1.5 * H)
    orch.stop(_iid)  # tick 없이 바로 정지해도 그때까지 full 요율로 과금
    assert ledger.total("usd") == pytest.approx(0.60, abs=TOL)


def test_externally_destroyed_instance_billed_until_seen_gone(api, ledger, make, clock):
    orch = make()
    t = clock()
    iid = _rent_running(orch, api, clock)
    clock.set(t + H)
    orch.tick()
    del api.instances[iid]
    clock.set(t + 1.5 * H)
    orch.tick()
    assert ledger.total("usd") == pytest.approx(0.60, abs=TOL)
    clock.set(t + 5 * H)
    orch.tick()
    make().tick()
    assert ledger.total("usd") == pytest.approx(0.60, abs=TOL)
    assert any(e.meta.get("final") for e in ledger.events(run_id=f"vast:{iid}"))


def test_destroy_is_final(api, ledger, make, clock):
    orch = make()
    t = clock()
    iid = _rent_running(orch, api, clock)
    clock.set(t + H)
    orch.destroy(iid)
    assert iid not in api.instances and api.n("destroy_instance") == 1
    assert ledger.total("usd") == pytest.approx(0.40, abs=TOL)
    clock.set(t + 9 * H)
    orch.tick()
    assert ledger.total("usd") == pytest.approx(0.40, abs=TOL)


def test_unmanaged_instances_untouched(api, ledger, make, clock):
    api.instances[42] = {"id": 42, "machine_id": 1, "label": "founder-manual", "gpu_name": "RTX 5090",
                         "num_gpus": 1, "dph_total": 0.9, "actual_status": "running", "intended_status": "running"}
    api.instances[43] = {"id": 43, "machine_id": 2, "label": None, "gpu_name": "RTX 5090",
                         "num_gpus": 1, "dph_total": 0.9, "actual_status": "running", "intended_status": "running"}
    orch = make(total=0.1, idle_timeout_s=60, boot_timeout_s=60)
    clock.set(clock() + 10 * H)
    rep = orch.tick()
    assert rep.actions == []
    assert api.n("set_state") == 0 and api.n("destroy_instance") == 0
    assert ledger.events(run_id="vast:42") == [] and ledger.events(run_id="vast:43") == []
    assert api.instances[42]["intended_status"] == "running"


def test_missing_intended_status_is_billed_conservatively(api, ledger, make, clock):
    orch = make()
    t = clock()
    iid = _rent_running(orch, api, clock)
    del api.instances[iid]["intended_status"]
    clock.set(t + H)
    orch.tick()
    assert ledger.total("usd") == pytest.approx(0.40, abs=TOL)
    inst = V.Instance.from_api({"id": 1, "label": "buc-x", "dph_total": 0.3, "cur_state": "stopped"})
    assert inst.intended_status == "stopped"
    assert V.Instance.from_api({"id": 1, "label": "buc-x", "dph_total": 0.3}).intended_status == "running"
