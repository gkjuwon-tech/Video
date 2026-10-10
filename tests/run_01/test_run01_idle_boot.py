"""유휴 자동 종료·부팅 기한·행동 우선순위·실패 격리."""
import pytest
from run01_fakes import raw_offer

from buc.rt import vast as V


def _observe_running(orch, api, clock, iid, dt=60):
    api.set_status(iid, actual="running")
    clock.set(clock() + dt)
    return orch.tick()


def test_idle_stop_boundary(api, make, clock):
    orch = make(idle_timeout_s=1800)
    t = clock()
    iid = orch.rent()
    assert _observe_running(orch, api, clock, iid).actions == []
    clock.set(t + 1799)
    assert orch.tick().actions == []
    clock.set(t + 1800)
    rep = orch.tick()
    assert rep.actions == [V.Action("stop", iid, "idle")]
    assert api.instances[iid]["intended_status"] == "stopped"
    clock.set(t + 9000)
    assert orch.tick().actions == []  # 이미 정지된 것은 다시 건드리지 않는다


def test_mark_active_extends(api, make, clock):
    orch = make(idle_timeout_s=1800)
    t = clock()
    iid = orch.rent()
    _observe_running(orch, api, clock, iid)
    orch.mark_active(iid, t + 1000)
    clock.set(t + 2799)
    assert orch.tick().actions == []
    clock.set(t + 2800)
    assert orch.tick().actions == [V.Action("stop", iid, "idle")]


def test_activity_source_extends(api, make, clock):
    t = clock()
    seen = {}
    orch = make(idle_timeout_s=1800, activity=lambda i: seen.get(i))
    iid = orch.rent()
    _observe_running(orch, api, clock, iid)
    seen[iid] = t + 1500
    clock.set(t + 3299)
    assert orch.tick().actions == []
    clock.set(t + 3300)
    assert orch.tick().actions == [V.Action("stop", iid, "idle")]


def test_idle_destroy_action(api, ledger, make, clock):
    orch = make(idle_timeout_s=600, idle_action="destroy")
    t = clock()
    iid = orch.rent()
    _observe_running(orch, api, clock, iid)
    clock.set(t + 600)
    assert orch.tick().actions == [V.Action("destroy", iid, "idle")]
    assert iid not in api.instances
    assert any(e.meta.get("final") for e in ledger.events(run_id=f"vast:{iid}"))


def test_idle_restarts_from_start(api, make, clock):
    orch = make(idle_timeout_s=1800)
    t = clock()
    iid = orch.rent()
    _observe_running(orch, api, clock, iid)
    clock.set(t + 1800)
    orch.tick()
    clock.set(t + 5000)
    orch.start(iid)  # 재개 시각이 새 기준점
    _observe_running(orch, api, clock, iid, dt=10)
    clock.set(t + 5000 + 1799)
    assert orch.tick().actions == []
    clock.set(t + 5000 + 1800)
    assert orch.tick().actions == [V.Action("stop", iid, "idle")]


def test_boot_timeout_destroys_and_excludes_machine(api, make, clock):
    bad_offer = raw_offer(1, dph=0.40, machine_id=111)
    api.offers = [dict(bad_offer), raw_offer(2, dph=0.45, machine_id=222)]
    orch = make(boot_timeout_s=3000, idle_timeout_s=60)
    t = clock()
    iid = orch.rent()
    assert api.instances[iid]["machine_id"] == 111
    clock.set(t + 2999)
    assert orch.tick().actions == []  # 한 번도 running이 아니면 유휴 규칙은 적용되지 않는다
    clock.set(t + 3000)
    assert orch.tick().actions == [V.Action("destroy", iid, "boot_timeout")]
    assert iid not in api.instances
    assert 111 in orch.bad_machines
    api.offers.append(dict(bad_offer))  # 같은 기계가 다시 싸게 나와도
    iid2 = orch.rent()
    assert api.instances[iid2]["machine_id"] == 222


def test_boot_timer_restarts_on_start(api, make, clock):
    orch = make(boot_timeout_s=3000)
    t = clock()
    iid = orch.rent()
    _observe_running(orch, api, clock, iid)
    orch.stop(iid)
    clock.set(t + 10000)
    orch.start(iid)
    api.set_status(iid, actual="loading")
    clock.set(t + 10000 + 2999)
    assert orch.tick().actions == []
    clock.set(t + 10000 + 3000)
    assert orch.tick().actions == [V.Action("destroy", iid, "boot_timeout")]


def test_boot_timeout_survives_restart(api, make, clock):
    t = clock()
    iid = make(boot_timeout_s=3000).rent()
    clock.set(t + 3000)
    assert make(boot_timeout_s=3000).tick().actions == [V.Action("destroy", iid, "boot_timeout")]


def test_one_action_per_instance_boot_beats_cap(api, make, clock):
    orch = make(total=0.5, boot_timeout_s=3000)
    t = clock()
    iid = orch.rent()
    clock.set(t + 4000)  # 0.444 + 0.1 > 0.5 이고 부팅 기한도 넘음
    rep = orch.tick()
    assert rep.actions == [V.Action("destroy", iid, "boot_timeout")]
    assert api.n("set_state") == 0


def test_one_action_per_instance_cap_beats_idle(api, make, clock):
    orch = make(total=0.5, idle_timeout_s=1800)
    t = clock()
    iid = orch.rent()
    _observe_running(orch, api, clock, iid)
    clock.set(t + 4000)
    assert orch.tick().actions == [V.Action("stop", iid, "cap_total")]
    assert api.n("set_state") == 1


def test_action_failure_is_isolated(api, make, clock):
    api.offers = [raw_offer(1, dph=0.40), raw_offer(2, dph=0.45)]
    orch = make(total=2.0, max_instances=2)
    t = clock()
    a = orch.rent()
    b = orch.rent()
    api.set_status(a, actual="running")
    api.set_status(b, actual="running")
    api.fail.add(("set_state", a))
    clock.set(t + 2.2 * H_)
    rep = orch.tick()
    assert len(rep.errors) == 1 and isinstance(rep.errors[0], str)
    assert api.instances[b]["intended_status"] == "stopped"
    assert api.instances[a]["intended_status"] == "running"
    api.fail.clear()
    clock.set(t + 2.3 * H_)
    rep = orch.tick()  # 다음 tick에서 다시 시도
    assert V.Action("stop", a, "cap_total") in rep.actions
    assert api.instances[a]["intended_status"] == "stopped"


def test_list_failure_propagates(api, make, clock):
    orch = make()

    def boom():
        raise V.VastAPIError("down")
    api.list_instances = boom
    with pytest.raises(V.VastError):
        orch.tick()


H_ = 3600.0
