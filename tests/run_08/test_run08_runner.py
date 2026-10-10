"""RUN-08: run_case / run_suite / 판정 / 보고서 왕복."""
import json
import os

import pytest
from run08_helpers import (GATE_PARAMS, Clock, OutOfMemoryError, fake_5090, fake_cpu, fake_t4,
                           make_case)

SECRET = "hf_" + "B" * 24


def dev(t4, tt):
    return t4.detect_device(tt)


# ------------------------------------------------------------------ verdict_for
def test_verdict_for_table(t4):
    t4dev = dev(t4, fake_t4())
    other = dev(t4, fake_5090())
    fast = t4.summarize_samples([0.55])
    slow = t4.summarize_samples([0.65])
    edge = t4.summarize_samples([0.6])
    over = t4.summarize_samples([0.6000011])
    nongate = {"mode": "fwd", "width": 3840, "height": 2160, "n_prims": 1_000_000}
    assert t4.gate_case(GATE_PARAMS) and not t4.gate_case(nongate)
    assert not t4.gate_case({**GATE_PARAMS, "mode": "fwdbwd"})
    assert not t4.gate_case({**GATE_PARAMS, "width": 1920})
    assert t4.verdict_for(GATE_PARAMS, "ok", fast, t4dev) == "pass"
    assert t4.verdict_for(GATE_PARAMS, "ok", edge, t4dev) == "pass"        # 경계 포함
    assert t4.verdict_for(GATE_PARAMS, "ok", over, t4dev) == "fail"
    assert t4.verdict_for(GATE_PARAMS, "ok", slow, t4dev) == "fail"
    assert t4.verdict_for(GATE_PARAMS, "oom", None, t4dev) == "fail"
    assert t4.verdict_for(GATE_PARAMS, "error", None, t4dev) == "unmeasured"
    assert t4.verdict_for(GATE_PARAMS, "unmeasured", None, t4dev) == "unmeasured"
    # T4가 아니면 빨라도 통과를 주장하지 않는다
    assert t4.verdict_for(GATE_PARAMS, "ok", fast, other) == "unmeasured"
    assert t4.verdict_for(GATE_PARAMS, "oom", None, other) == "unmeasured"
    # 게이트 케이스가 아니면 항상 info
    for st in ("ok", "oom", "error", "unmeasured"):
        assert t4.verdict_for(nongate, st, fast if st == "ok" else None, t4dev) == "info"


# --------------------------------------------------------------------- run_case
def test_run_case_no_cuda_skips_build(t4):
    case, state, builds = make_case(t4, params=GATE_PARAMS)
    m = t4.run_case(case, device=dev(t4, fake_cpu()), torch_mod=fake_cpu())
    assert m.status == "unmeasured" and m.reason == "no-cuda"
    assert m.timing is None and m.memory is None
    assert m.verdict == "unmeasured"
    assert builds == [] and state["calls"] == 0


def test_run_case_non_gate_unmeasured_verdict_is_info(t4):
    case, _, builds = make_case(t4)
    m = t4.run_case(case, device=dev(t4, fake_cpu()), torch_mod=fake_cpu())
    assert m.verdict == "info" and builds == []


def test_run_case_ok_on_t4_orders_calls_and_converts_gib(t4):
    tt = fake_t4()
    clock = Clock(100.0, tt.events)
    case, state, builds = make_case(t4, clock=clock, dur=0.5, params=GATE_PARAMS)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=3, warmup=2, clock=clock)
    assert builds == ["cuda"]
    assert state["calls"] == 5
    assert m.status == "ok" and m.reason is None and m.verdict == "pass"      # 0.5 s <= 0.6 s
    assert m.timing.n == 3 and m.timing.warmup == 2
    assert m.timing.samples_s == (0.5, 0.5, 0.5)
    assert m.memory.baseline_gib == pytest.approx(1.0, abs=1e-12)
    assert m.memory.peak_allocated_gib == pytest.approx(6.0, abs=1e-12)
    assert m.memory.peak_reserved_gib == pytest.approx(7.5, abs=1e-12)
    ev = tt.events
    # 워밍업 뒤, 첫 측정 전에 피크를 리셋하고 기준선을 읽는다
    i_reset = ev.index("reset_peak")
    assert ev.count("reset_peak") == 1
    assert ev[i_reset + 1] == "memory_allocated"
    assert ev[:i_reset].count("clock") == 0          # 워밍업에는 시계를 읽지 않는다
    assert ev[i_reset:].count("clock") == 6          # 측정 3회 x (t0, t1)
    # 측정이 끝난 뒤 피크를 읽고, 마지막에 캐시를 비운다
    assert ev.index("max_memory_allocated") > max(i for i, e in enumerate(ev) if e == "clock")
    assert ev.index("max_memory_reserved") > max(i for i, e in enumerate(ev) if e == "clock")
    assert ev[-1] == "empty_cache" and ev.count("empty_cache") == 1


def test_run_case_gate_pass_and_fail_on_t4(t4):
    for dur, want in ((0.5, "pass"), (0.7, "fail")):
        tt = fake_t4()
        clock = Clock(100.0)
        case, _, _ = make_case(t4, clock=clock, dur=dur, params=GATE_PARAMS)
        m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=3, warmup=1, clock=clock)
        assert m.status == "ok" and m.verdict == want, dur


def test_run_case_ok_on_5090_is_not_a_t4_pass(t4):
    tt = fake_5090()
    clock = Clock(100.0)
    case, _, builds = make_case(t4, clock=clock, dur=0.125, params=GATE_PARAMS)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=2, warmup=1, clock=clock)
    assert m.status == "ok" and builds == ["cuda"]
    assert m.timing.median_s == 0.125
    assert m.verdict == "unmeasured"


def test_run_case_oom_in_fn(t4):
    tt = fake_t4()
    clock = Clock(100.0)

    def behavior(i):
        if i >= 1:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 3.00 GiB")

    case, state, _ = make_case(t4, clock=clock, behavior=behavior, params=GATE_PARAMS)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=3, warmup=2, clock=clock)
    assert m.status == "oom" and m.timing is None and m.memory is None
    assert m.verdict == "fail"
    assert m.reason.startswith("OutOfMemoryError: ") and len(m.reason) <= 200
    assert tt.cuda.calls.count("empty_cache") == 1


def test_run_case_oom_in_build(t4):
    tt = fake_t4()

    def build(kind):
        raise OutOfMemoryError("CUDA out of memory")

    case = t4.Case(name="b", params=dict(GATE_PARAMS), build=build)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt)
    assert m.status == "oom" and m.verdict == "fail"
    assert tt.cuda.calls.count("empty_cache") == 1


def test_run_case_error_is_redacted_and_not_oom(t4):
    tt = fake_t4()

    def behavior(i):
        raise ValueError(f"bad token={SECRET} here")

    case, _, _ = make_case(t4, behavior=behavior, params=GATE_PARAMS)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=1, warmup=0)
    assert m.status == "error" and m.verdict == "unmeasured"
    assert m.reason.startswith("ValueError: ")
    assert SECRET not in m.reason and "***" in m.reason
    assert tt.cuda.calls.count("empty_cache") == 1


def test_run_case_reason_is_capped_at_200(t4):
    tt = fake_t4()

    def behavior(i):
        raise RuntimeError("boom " + "word " * 200)

    case, _, _ = make_case(t4, behavior=behavior)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=1, warmup=0)
    assert m.status == "error" and len(m.reason) <= 200


def test_run_case_keyboard_interrupt_propagates(t4):
    tt = fake_t4()

    def behavior(i):
        raise KeyboardInterrupt

    case, _, _ = make_case(t4, behavior=behavior)
    with pytest.raises(KeyboardInterrupt):
        t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=1, warmup=0)


def test_run_case_cpu_allowed_runs_without_memory(t4):
    tt = fake_cpu()
    clock = Clock(100.0)
    case, state, builds = make_case(t4, clock=clock, dur=0.25, params=GATE_PARAMS)
    m = t4.run_case(case, device=dev(t4, tt), torch_mod=tt, repeats=2, warmup=1, clock=clock, allow_cpu=True)
    assert builds == ["cpu"] and state["calls"] == 3
    assert m.status == "ok" and m.memory is None
    assert m.timing.samples_s == (0.25, 0.25)
    assert m.verdict == "unmeasured"                # CPU 수치는 T4 게이트가 아니다
    assert tt.cuda.calls == []                      # CPU에서는 cuda API를 건드리지 않는다


# -------------------------------------------------------------------- run_suite
def ladder(t4, ns, clock, behaviors=None, w=3840, h=2160, dur=0.25, prefix="k"):
    out = []
    for n in ns:
        beh = (behaviors or {}).get(n)
        c, st, b = make_case(t4, name=f"{prefix}{n}", clock=clock, dur=dur, behavior=beh,
                             params={"mode": "fwd", "width": w, "height": h, "n_prims": n})
        out.append((c, st, b))
    return out


def test_run_suite_skips_after_oom_without_building(t4):
    tt = fake_t4()
    clock = Clock(100.0)

    def oom(i):
        raise OutOfMemoryError("CUDA out of memory")

    a = ladder(t4, [500, 1000, 3000], clock, behaviors={1000: oom})
    other = ladder(t4, [3000], clock, w=1920, h=1080, prefix="o")            # 다른 해상도는 영향 없음
    fwdbwd, st_fb, b_fb = make_case(t4, name="fb", clock=clock,
                                    params={"mode": "fwdbwd", "width": 3840, "height": 2160, "n_prims": 4000})
    cases = [c for c, _, _ in a] + [other[0][0], fwdbwd]
    rep = t4.run_suite(cases, device=dev(t4, tt), torch_mod=tt, repeats=2, warmup=1, clock=clock)
    assert [m.name for m in rep.measurements] == ["k500", "k1000", "k3000", "o3000", "fb"]
    assert rep.measurements[0].status == "ok"
    assert rep.measurements[1].status == "oom"
    assert rep.measurements[2].status == "unmeasured" and rep.measurements[2].reason == "skipped-after-oom"
    assert a[2][2] == [] and a[2][1]["calls"] == 0                 # 스킵된 케이스는 build도 안 한다
    assert rep.measurements[3].status == "ok"                      # 1920x1080
    assert rep.measurements[4].status == "ok" and b_fb == ["cuda"]  # 다른 mode


def test_run_suite_smaller_n_after_oom_is_not_skipped(t4):
    tt = fake_t4()
    clock = Clock(100.0)

    def oom(i):
        raise OutOfMemoryError("CUDA out of memory")

    a = ladder(t4, [3000, 500], clock, behaviors={3000: oom})     # 입력 순서가 내림차순이면 작은 쪽은 그대로 실행
    rep = t4.run_suite([c for c, _, _ in a], device=dev(t4, tt), torch_mod=tt, repeats=1, warmup=0, clock=clock)
    assert [m.status for m in rep.measurements] == ["oom", "ok"]


def test_run_suite_error_does_not_stop_next(t4):
    tt = fake_t4()
    clock = Clock(100.0)

    def bad(i):
        raise RuntimeError("kaput")

    bad_case, _, _ = make_case(t4, name="bad", clock=clock, behavior=bad,
                               params={"mode": "fwd", "width": 64, "height": 48, "n_prims": 5})
    good_case, st, _ = make_case(t4, name="good", clock=clock,
                                 params={"mode": "fwd", "width": 32, "height": 32, "n_prims": 5})
    rep = t4.run_suite([bad_case, good_case], device=dev(t4, tt), torch_mod=tt, repeats=1, warmup=0, clock=clock)
    assert [m.status for m in rep.measurements] == ["error", "ok"]
    assert st["calls"] == 1


def test_run_suite_time_budget(t4):
    tt = fake_t4()
    clock = Clock(100.0)
    a = ladder(t4, [100, 200, 300], clock, dur=1.0)
    # 케이스 하나가 (워밍업 1 + 반복 2) x 1.0 = 3.0 초를 쓴다. 예산 5초 -> 둘째 케이스까지 시작, 셋째는 예산 초과.
    rep = t4.run_suite([c for c, _, _ in a], device=dev(t4, tt), torch_mod=tt, repeats=2, warmup=1, clock=clock,
                       budget_s=5.0)
    st = [m.status for m in rep.measurements]
    assert st == ["ok", "ok", "unmeasured"]
    assert rep.measurements[2].reason == "time-budget"
    assert a[2][2] == [] and a[2][1]["calls"] == 0
    assert rep.budget_s == 5.0


def test_run_suite_budget_boundary_is_inclusive_and_none_runs_all(t4):
    tt = fake_t4()
    clock = Clock(100.0)
    a = ladder(t4, [100, 200], clock, dur=1.0)
    # 첫 케이스가 정확히 3.0초 -> budget_s=3.0 이면 clock()-시작 >= budget 이므로 둘째는 건너뛴다
    rep = t4.run_suite([c for c, _, _ in a], device=dev(t4, tt), torch_mod=tt, repeats=2, warmup=1, clock=clock,
                       budget_s=3.0)
    assert rep.measurements[1].reason == "time-budget"
    clock2 = Clock(100.0)
    b = ladder(t4, [100, 200], clock2, dur=1.0)
    rep2 = t4.run_suite([c for c, _, _ in b], device=dev(t4, fake_t4()), torch_mod=fake_t4(), repeats=2, warmup=1,
                        clock=clock2, budget_s=None)
    assert [m.status for m in rep2.measurements] == ["ok", "ok"]


def test_run_suite_cpu_marks_everything_unmeasured(t4):
    tt = fake_cpu()
    a = ladder(t4, [100, 200], Clock(0.0))
    gate, _, gb = make_case(t4, name="g", params=GATE_PARAMS)
    rep = t4.run_suite([c for c, _, _ in a] + [gate], device=dev(t4, tt), torch_mod=tt, created_utc="2026-10-24T00:00:00Z")
    assert all(m.status == "unmeasured" and m.reason == "no-cuda" for m in rep.measurements)
    assert rep.shot_on_t4 is False
    assert rep.gate_verdict == "unmeasured"
    assert gb == [] and all(b == [] for _, _, b in a)
    assert rep.created_utc == "2026-10-24T00:00:00Z" and rep.budget_s is None


# ----------------------------------------------------------------------- Report
def meas(t4, name, status, params, verdict, secs=None):
    tm = t4.summarize_samples([secs], warmup=1) if secs is not None else None
    mem = t4.MemStats(baseline_gib=1.0, peak_allocated_gib=2.0, peak_reserved_gib=3.0) if secs is not None else None
    return t4.Measurement(name=name, params=dict(params), status=status, timing=tm, memory=mem, verdict=verdict,
                          reason=None)


def report(t4, tt, ms, **kw):
    return t4.Report(schema=1, device=dev(t4, tt), measurements=tuple(ms), created_utc=kw.get("created_utc"),
                     budget_s=kw.get("budget_s"))


def test_report_shot_on_t4_and_gate_verdict(t4):
    ok_gate = meas(t4, "g", "ok", GATE_PARAMS, "pass", 0.5)
    fail_gate = meas(t4, "g", "oom", GATE_PARAMS, "fail")
    unm_gate = meas(t4, "g", "unmeasured", GATE_PARAMS, "unmeasured")
    info = meas(t4, "i", "ok", {"mode": "fwdbwd", "width": 1024, "height": 1024, "n_prims": 10}, "info", 0.1)
    assert report(t4, fake_t4(), [ok_gate, info]).shot_on_t4 is True
    assert report(t4, fake_t4(), [ok_gate, info]).gate_verdict == "pass"
    assert report(t4, fake_t4(), [fail_gate, info]).gate_verdict == "fail"
    assert report(t4, fake_t4(), [ok_gate, fail_gate]).gate_verdict == "fail"
    assert report(t4, fake_t4(), [ok_gate, unm_gate]).gate_verdict == "unmeasured"
    assert report(t4, fake_t4(), [info]).gate_verdict == "unmeasured"           # 게이트 케이스 없음
    assert report(t4, fake_t4(), []).gate_verdict == "unmeasured"
    assert report(t4, fake_t4(), [unm_gate]).shot_on_t4 is False                 # T4인데 아무것도 못 쟀다
    assert report(t4, fake_5090(), [ok_gate, info]).shot_on_t4 is False          # 측정은 했지만 T4가 아니다
    assert report(t4, fake_cpu(), [unm_gate]).shot_on_t4 is False


def test_measurement_validates_status_and_verdict(t4):
    with pytest.raises(ValueError):
        t4.Measurement(name="x", params={}, status="weird", timing=None, memory=None, verdict="info", reason=None)
    with pytest.raises(ValueError):
        t4.Measurement(name="x", params={}, status="ok", timing=None, memory=None, verdict="maybe", reason=None)
    assert t4.STATUSES == ("ok", "oom", "error", "unmeasured")
    assert t4.VERDICTS == ("pass", "fail", "unmeasured", "info")


def full_report(t4):
    tt = fake_t4()
    clock = Clock(100.0)

    def oom(i):
        raise OutOfMemoryError("CUDA out of memory")

    cs = [c for c, _, _ in ladder(t4, [500, 3_000_000], clock, behaviors={3_000_000: oom}, dur=0.25)]
    return t4.run_suite(cs, device=dev(t4, tt), torch_mod=tt, repeats=2, warmup=1, clock=clock,
                        created_utc="2026-10-24T01:02:03Z", budget_s=3600.0)


def test_report_dict_roundtrip(t4):
    rep = full_report(t4)
    d = rep.to_dict()
    assert set(d) == {"schema", "created_utc", "budget_s", "device", "measurements"}
    assert d["schema"] == 1
    blob = json.dumps(d)
    back = t4.Report.from_dict(json.loads(blob))
    assert back == rep
    assert back.to_dict() == d
    bad = dict(d, schema=2)
    with pytest.raises(ValueError):
        t4.Report.from_dict(bad)


def test_write_and_load_report(t4, tmp_path):
    rep = full_report(t4)
    p = t4.write_report(rep, tmp_path / "deep" / "dir" / "r.json")
    assert p.exists() and p.name == "r.json"
    raw = p.read_bytes()
    assert raw.endswith(b"\n")
    text = raw.decode("utf-8")
    assert text == json.dumps(rep.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert sorted(os.listdir(p.parent)) == ["r.json"]                 # 잔여 임시 파일 없음
    assert t4.load_report(p) == rep
    p2 = t4.write_report(rep, p)                                       # 덮어쓰기도 같은 바이트
    assert p2.read_bytes() == raw
    assert sorted(os.listdir(p.parent)) == ["r.json"]


def test_load_report_errors(t4, tmp_path):
    with pytest.raises(FileNotFoundError):
        t4.load_report(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        t4.load_report(bad)


def test_report_has_no_environment_leak(t4, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("VAST_API_KEY", "k" * 40)
    blob = json.dumps(full_report(t4).to_dict())
    assert SECRET not in blob and "k" * 40 not in blob
