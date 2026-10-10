"""FND-07 ledger.py: 추가 전용 SQLite 장부, 합계 정확성, 멱등 키, 예산 캡(원자적), 영속."""
import math
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone

import pytest

from buc.core import ledger as L
from buc.core.ledger import BudgetCheck, BudgetExceeded, LedgerError, LedgerEvent, open_ledger, utc_day_bounds

DAY = 86400
D0 = 20000 * DAY  # 임의의 UTC 자정(2024-10-04)
COLS = {"id", "ts", "kind", "amount", "run_id", "stage", "note", "meta", "idem_key"}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clk():
    return Clock()


@pytest.fixture
def led(tmp_path, clk):
    with open_ledger(tmp_path / "ledger.sqlite", clock=clk) as lg:
        yield lg


def rec(lg, kind="usd", amount=1.0, run_id="r1", **kw):
    return lg.record(LedgerEvent(kind=kind, amount=amount, run_id=run_id, **kw))


def rwb(lg, kind="usd", amount=1.0, run_id="r1", daily_cap=None, total_cap=None, **kw):
    return lg.record_within_budget(LedgerEvent(kind=kind, amount=amount, run_id=run_id, **kw),
                                   daily_cap=daily_cap, total_cap=total_cap)


# ---------------------------------------------------------------- 상수와 헬퍼
def test_constants_and_price_table():
    assert L.KINDS == ("tfu", "gpu_h", "usd")
    assert L.TFU_FLOPS == 6.2e13
    assert L.TFU_PER_FORWARD == {256: 0.1, 512: 0.25, 768: 0.56, 1024: 1.0}


def test_tfu_helpers():
    assert L.tfu_for_forward(512) == 0.25
    assert L.tfu_for_forward(1024) == 1.0
    assert L.tfu_for_forward(768, 3) == pytest.approx(1.68, abs=1e-12)
    assert L.tfu_for_forward(256, 10) == pytest.approx(1.0, abs=1e-12)
    assert L.tfu_for_forward(512, 0) == 0.0
    for bad in [(640,), (512, -1), (512, True), (0,)]:
        with pytest.raises(ValueError):
            L.tfu_for_forward(*bad)
    assert L.tfu_from_flops(6.2e13) == pytest.approx(1.0, abs=1e-12)
    assert L.tfu_from_flops(3.1e13) == pytest.approx(0.5, abs=1e-12)
    assert L.tfu_from_flops(0) == 0.0
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            L.tfu_from_flops(bad)


def test_gpu_helpers():
    assert L.gpu_hours(7200) == pytest.approx(2.0, abs=1e-12)
    assert L.gpu_hours(0) == 0.0
    assert L.gpu_cost_usd(1800, 0.5) == pytest.approx(0.25, abs=1e-12)
    assert L.gpu_cost_usd(3600, 0.236) == pytest.approx(0.236, abs=1e-12)
    for args in [(-1,), (float("nan"),)]:
        with pytest.raises(ValueError):
            L.gpu_hours(*args)
    for args in [(-1, 0.5), (10, -0.5), (float("inf"), 0.5), (10, float("nan"))]:
        with pytest.raises(ValueError):
            L.gpu_cost_usd(*args)


# ---------------------------------------------------------------- 기록과 합계
def test_record_returns_monotonic_ids_and_events_roundtrip(led, clk):
    clk.t = 1234.5
    i1 = rec(led, "usd", 0.25, "runA", stage="S4", note="n1", meta={"b": 1, "a": [2]})
    clk.t = 1300.0
    i2 = rec(led, "tfu", 700, "runA")
    i3 = rec(led, "gpu_h", 0.5, "runB", ts=42.0)
    assert (i1, i2, i3) == (1, 2, 3)
    evs = led.events()
    assert [e.id for e in evs] == [1, 2, 3]
    e1 = evs[0]
    assert (e1.kind, e1.amount, e1.run_id, e1.stage, e1.note, e1.meta, e1.ts) == \
           ("usd", 0.25, "runA", "S4", "n1", {"b": 1, "a": [2]}, 1234.5)
    assert evs[1].amount == 700.0 and isinstance(evs[1].amount, float) and evs[1].ts == 1300.0
    assert evs[2].ts == 42.0 and evs[2].stage == "" and evs[2].note == "" and evs[2].meta == {}
    assert led.count() == 3


def test_default_clock_is_wall_time(tmp_path):
    with open_ledger(tmp_path / "w.sqlite") as lg:
        rec(lg)
        assert abs(lg.events()[0].ts - time.time()) < 60


def test_totals_by_kind_and_run(led):
    rec(led, "usd", 1.5, "a")
    rec(led, "usd", 2.0, "b")
    rec(led, "usd", 0.5, "a")
    rec(led, "tfu", 100, "a")
    rec(led, "gpu_h", 2, "b")
    assert led.total("usd") == 4.0
    assert led.total("usd", run_id="a") == 2.0
    assert led.total("tfu") == 100.0 and led.total("gpu_h", run_id="a") == 0.0
    assert led.totals_by_run("usd") == {"a": 2.0, "b": 2.0}
    assert led.totals_by_run("gpu_h") == {"b": 2.0}
    assert led.totals_by_run("tfu") == {"a": 100.0}
    with pytest.raises(ValueError):
        led.total("eur")


def test_empty_total_is_float_zero(led):
    t = led.total("usd")
    assert t == 0.0 and isinstance(t, float)
    assert led.totals_by_run("usd") == {} and led.events() == [] and led.count() == 0


def test_zero_amount_is_allowed(led):
    rec(led, amount=0)
    assert led.total("usd") == 0.0 and led.count() == 1


@pytest.mark.parametrize("kw", [
    dict(kind="eur"), dict(kind=""), dict(amount=-1), dict(amount=-1e-9), dict(amount=float("nan")),
    dict(amount=float("inf")), dict(amount=True), dict(amount="1"), dict(amount=None), dict(run_id=""),
    dict(run_id=5), dict(meta={"a": object()}), dict(meta=[1]), dict(meta={"a": float("nan")}),
    dict(idem_key=""), dict(stage=3), dict(note=None),
])
def test_invalid_events_are_rejected_and_not_written(led, kw):
    with pytest.raises((TypeError, ValueError)):
        rec(led, **kw)
    assert led.count() == 0


def test_range_is_half_open_and_accepts_datetimes(led):
    rec(led, amount=1, ts=100)
    rec(led, amount=2, ts=200)
    rec(led, amount=4, ts=300)
    assert led.total("usd", since=100) == 7.0
    assert led.total("usd", since=101) == 6.0
    assert led.total("usd", until=200) == 1.0  # 200 제외
    assert led.total("usd", since=100, until=300) == 3.0
    assert led.total("usd", since=300, until=300) == 0.0
    assert led.total("usd", since=datetime.fromtimestamp(200, timezone.utc)) == 6.0
    assert led.total("usd", until=datetime.fromtimestamp(300, timezone.utc)) == 3.0
    with pytest.raises(ValueError):
        led.total("usd", since=datetime(2026, 1, 1))  # naive
    rec(led, amount=8, ts=250, run_id="other")
    assert led.total("usd", since=200, until=300, run_id="other") == 8.0
    ev = led.events(since=200, until=300)
    assert [e.amount for e in ev] == [2.0, 8.0]


def test_events_filters_order_and_limit(led):
    ids = [rec(led, "usd", 1, "a"), rec(led, "tfu", 2, "a"), rec(led, "usd", 3, "b"), rec(led, "usd", 4, "a", ts=5.0)]
    assert [e.id for e in led.events()] == ids
    assert [e.amount for e in led.events(kind="usd")] == [1.0, 3.0, 4.0]
    assert [e.amount for e in led.events(kind="usd", run_id="a")] == [1.0, 4.0]
    assert [e.id for e in led.events(limit=2)] == ids[:2]
    assert led.events(run_id="zzz") == []


def test_utc_day_bounds_and_daily_total(led, clk):
    assert utc_day_bounds(D0 + 5) == (D0, D0 + DAY)
    assert utc_day_bounds(D0) == (D0, D0 + DAY)
    assert utc_day_bounds(D0 - 1) == (D0 - DAY, D0)
    rec(led, amount=1, ts=D0 - 1)
    rec(led, amount=2, ts=D0)
    rec(led, amount=4, ts=D0 + DAY - 1)
    rec(led, amount=8, ts=D0 + DAY)
    rec(led, "tfu", amount=16, ts=D0 + 10)
    assert led.daily_total("usd", ts=D0 + 5) == 6.0
    assert led.daily_total("usd", ts=D0 - 100) == 1.0
    assert led.daily_total("usd", ts=D0 + DAY) == 8.0
    assert led.daily_total("tfu", ts=D0 + 5) == 16.0
    clk.t = D0 + 3
    assert led.daily_total("usd") == 6.0  # 기본 ts = clock()


def test_total_is_bit_identical_under_any_record_order(tmp_path):
    rng = random.Random(7)
    amounts = [rng.uniform(0, 1) * 10 ** rng.randint(-9, 6) for _ in range(150)]
    amounts += [0.1] * 30 + [1e16, 1.0, -0.0 + 0.0, 1e-16]
    runs = [f"r{i % 5}" for i in range(len(amounts))]
    want = math.fsum(amounts)
    got = []
    for k in range(8):
        order = list(range(len(amounts)))
        random.Random(k).shuffle(order)
        with open_ledger(":memory:", clock=Clock()) as lg:
            for i in order:
                rec(lg, "usd", amounts[i], runs[i], ts=5.0)  # ts 동일: 저장 순서가 곧 삽입 순서
            got.append(lg.total("usd"))
            by_run = lg.totals_by_run("usd")
        for r in set(runs):
            assert by_run[r] == math.fsum(a for a, rr in zip(amounts, runs) if rr == r)
    assert all(g == want for g in got), (got, want)


# ---------------------------------------------------------------- 멱등 키
def test_idempotency_key(led):
    a = rec(led, "usd", 2.0, "job1", stage="S4", idem_key="job1#1", note="first", ts=10.0)
    b = rec(led, "usd", 2.0, "job1", stage="S4", idem_key="job1#1", note="retry", ts=99.0)
    assert a == b and led.count() == 1 and led.total("usd") == 2.0
    c = rec(led, "usd", 2.0, "job1", stage="S4", idem_key="job1#2")
    assert c != a and led.count() == 2
    for kw in (dict(amount=3.0), dict(kind="tfu"), dict(run_id="job2"), dict(stage="S5")):
        base = dict(kind="usd", amount=2.0, run_id="job1", stage="S4", idem_key="job1#1")
        base.update(kw)
        with pytest.raises(LedgerError):
            rec(led, **base)
    assert led.count() == 2
    rec(led, amount=1.0)
    rec(led, amount=1.0)  # 키 없는 이벤트는 얼마든지 반복 가능
    assert led.count() == 4


# ---------------------------------------------------------------- 저장소(추가 전용, 하류가 SELECT)
def test_append_only_raw_schema_and_select(tmp_path, clk):
    path = tmp_path / "x.sqlite"
    with open_ledger(path, clock=clk) as lg:
        rec(lg, "usd", 1.25, "r", meta={"b": 1, "a": [2]})
        rec(lg, "tfu", 5, "r")
        raw = sqlite3.connect(str(path))
        try:
            cols = {r[1] for r in raw.execute("PRAGMA table_info(events)")}
            assert cols == COLS
            assert raw.execute("PRAGMA user_version").fetchone()[0] == 1
            assert raw.execute("SELECT SUM(amount) FROM events WHERE kind='usd'").fetchone()[0] == 1.25
            assert raw.execute("SELECT meta FROM events WHERE id=1").fetchone()[0] == '{"a":[2],"b":1}'
            for sql in ("UPDATE events SET amount=0", "UPDATE events SET kind='usd' WHERE id=2",
                        "DELETE FROM events", "DELETE FROM events WHERE id=1"):
                with pytest.raises(sqlite3.DatabaseError):
                    raw.execute(sql)
                    raw.commit()
                raw.rollback()
        finally:
            raw.close()
        assert lg.count() == 2 and lg.total("usd") == 1.25 and lg.total("tfu") == 5.0


# ---------------------------------------------------------------- 열기·닫기·영속
def test_persists_across_reopen_and_ids_continue(tmp_path, clk):
    path = str(tmp_path / "deep" / "er" / "l.sqlite")
    with open_ledger(path, clock=clk) as lg:
        rec(lg, "usd", 1.5, "r", idem_key="k1", meta={"m": 1})
        rec(lg, "tfu", 3, "r")
    with open_ledger(path, clock=clk) as lg2:
        assert lg2.count() == 2 and lg2.total("usd") == 1.5
        assert lg2.events()[0].meta == {"m": 1}
        assert rec(lg2, "usd", 1.5, "r", idem_key="k1") == 1  # 멱등 키도 영속
        assert rec(lg2, "usd", 1.0, "r") == 3


def test_close_is_idempotent_and_use_after_close_fails(tmp_path):
    lg = open_ledger(tmp_path / "c.sqlite")
    rec(lg)
    lg.close()
    lg.close()
    for call in (lambda: rec(lg), lambda: lg.total("usd"), lambda: lg.events(), lambda: lg.count(),
                 lambda: lg.daily_total("usd"), lambda: check_budget_on(lg), lambda: rwb(lg, daily_cap=5)):
        with pytest.raises(LedgerError):
            call()


def check_budget_on(lg):
    return lg.check_budget("usd", 1.0, daily_cap=5)


def test_newer_schema_version_is_refused(tmp_path):
    path = tmp_path / "v.sqlite"
    with open_ledger(path):
        pass
    raw = sqlite3.connect(str(path))
    raw.execute("PRAGMA user_version=99")
    raw.commit()
    raw.close()
    with pytest.raises(LedgerError):
        open_ledger(path)


def test_memory_ledger_works():
    with open_ledger(":memory:", clock=Clock()) as lg:
        rec(lg, amount=2)
        assert lg.total("usd") == 2.0


# ---------------------------------------------------------------- 예산 검사(읽기 전용)
def test_check_budget_boundaries(led):
    t = D0 + 100
    c = led.check_budget("usd", 5.0, daily_cap=5.0, total_cap=None, ts=t)
    assert isinstance(c, BudgetCheck) and c.ok and c.reason == ""  # 정확히 캡: 통과
    assert (c.kind, c.amount, c.daily_spent, c.total_spent) == ("usd", 5.0, 0.0, 0.0)
    assert c.daily_remaining == 5.0 and c.total_remaining is None
    assert led.check_budget("usd", 5.0 + 1e-12, daily_cap=5.0, ts=t).ok  # 1e-9 허용
    c = led.check_budget("usd", 5.0 + 1e-6, daily_cap=5.0, ts=t)
    assert not c.ok and c.reason == "daily"
    assert led.check_budget("usd", 1e9).ok  # 캡 없음 = 무제한
    assert led.count() == 0  # 읽기 전용


def test_check_budget_daily_vs_total(led):
    rec(led, amount=3.0, ts=D0 + 10)
    rec(led, "tfu", amount=999.0, ts=D0 + 11)  # 다른 종류는 안 센다
    day2 = D0 + DAY + 5
    c = led.check_budget("usd", 5.0, daily_cap=5.0, total_cap=8.0, ts=day2)
    assert c.ok and (c.daily_spent, c.total_spent) == (0.0, 3.0)
    assert (c.daily_remaining, c.total_remaining) == (5.0, 5.0)
    c = led.check_budget("usd", 5.5, daily_cap=6.0, total_cap=8.0, ts=day2)
    assert not c.ok and c.reason == "total"
    c = led.check_budget("usd", 6.0, daily_cap=5.0, total_cap=8.0, ts=day2)
    assert not c.ok and c.reason == "daily"  # 둘 다 넘으면 daily
    c = led.check_budget("usd", 3.0, daily_cap=4.0, ts=D0 + 20)  # 같은 날: 3+3 > 4
    assert not c.ok and c.reason == "daily" and c.daily_spent == 3.0 and c.daily_remaining == 1.0
    c = led.check_budget("usd", 0.5, daily_cap=2.0, total_cap=2.0, ts=D0 + 20)  # 이미 넘은 뒤: remaining은 0 하한
    assert not c.ok and c.reason == "daily" and c.daily_remaining == 0.0 and c.total_remaining == 0.0
    assert led.check_budget("tfu", 1.0, daily_cap=1000.0, ts=D0 + 12).daily_spent == 999.0


def test_remaining_is_clamped_at_zero(led):
    rec(led, amount=9.0, ts=D0 + 1)
    c = led.check_budget("usd", 0.0, daily_cap=4.0, total_cap=5.0, ts=D0 + 2)
    assert (c.daily_remaining, c.total_remaining) == (0.0, 0.0)
    assert not led.check_budget("usd", 0.5, daily_cap=4.0, ts=D0 + 2).ok


def test_check_budget_validates_inputs(led):
    for args in [("eur", 1.0), ("usd", -1.0), ("usd", float("nan")), ("usd", True)]:
        with pytest.raises((TypeError, ValueError)):
            led.check_budget(*args, daily_cap=5)
    for kw in (dict(daily_cap=-1), dict(total_cap=float("nan")), dict(daily_cap="5")):
        with pytest.raises((TypeError, ValueError)):
            led.check_budget("usd", 1.0, **kw)


# ---------------------------------------------------------------- 원자적 예산 기록
def test_record_within_budget_ok_and_exceeded(led):
    i = rwb(led, amount=4.0, daily_cap=5.0, total_cap=10.0, ts=D0 + 1)
    assert i == 1 and led.total("usd") == 4.0
    with pytest.raises(BudgetExceeded) as ei:
        rwb(led, amount=1.5, daily_cap=5.0, total_cap=10.0, ts=D0 + 2)
    assert isinstance(ei.value, LedgerError)
    assert ei.value.reason == "daily" and ei.value.check.ok is False and ei.value.check.daily_spent == 4.0
    assert led.count() == 1 and led.total("usd") == 4.0  # 아무것도 안 썼다
    assert rwb(led, amount=1.0, daily_cap=5.0, total_cap=10.0, ts=D0 + 3) == 2  # 정확히 캡
    with pytest.raises(BudgetExceeded):
        rwb(led, amount=1e-6, daily_cap=5.0, ts=D0 + 4)
    assert led.count() == 2


def test_budget_days_use_event_ts_and_total_spans_days(led):
    caps = dict(daily_cap=5.0, total_cap=10.0)
    rwb(led, amount=4.0, ts=D0 + 10, **caps)
    rwb(led, amount=4.0, ts=D0 + DAY + 10, **caps)  # 다음 날: 일 캡 새로 시작
    with pytest.raises(BudgetExceeded) as e1:
        rwb(led, amount=2.0, ts=D0 + 20, **caps)  # 같은 날(첫째 날) 4+2 > 5
    assert e1.value.reason == "daily"
    with pytest.raises(BudgetExceeded) as e2:
        rwb(led, amount=4.0, ts=D0 + 2 * DAY + 10, **caps)  # 셋째 날: 일은 OK, 누적 12 > 10
    assert e2.value.reason == "total"
    rwb(led, amount=2.0, ts=D0 + 2 * DAY + 10, **caps)  # 누적 10 = 캡
    assert led.count() == 3 and led.total("usd") == 10.0


def test_budget_default_ts_uses_clock_and_kinds_are_independent(led, clk):
    clk.t = D0 + 50
    rwb(led, "usd", 4.0, daily_cap=5.0)
    rwb(led, "tfu", 500.0, daily_cap=600.0)
    rwb(led, "gpu_h", 1.0, daily_cap=1.0)
    with pytest.raises(BudgetExceeded):
        rwb(led, "usd", 2.0, daily_cap=5.0)
    clk.t = D0 + DAY + 50
    rwb(led, "usd", 5.0, daily_cap=5.0)
    assert led.total("usd") == 9.0


def test_replay_with_same_idem_key_skips_budget_check(led):
    first = rwb(led, amount=5.0, daily_cap=5.0, total_cap=5.0, ts=D0 + 1, idem_key="job#1")
    again = rwb(led, amount=5.0, daily_cap=5.0, total_cap=5.0, ts=D0 + 2, idem_key="job#1")
    assert again == first and led.count() == 1 and led.total("usd") == 5.0
    with pytest.raises(BudgetExceeded):
        rwb(led, amount=0.5, daily_cap=5.0, ts=D0 + 3, idem_key="job#2")
    with pytest.raises(LedgerError):  # 같은 키, 다른 내용
        rwb(led, amount=4.0, daily_cap=50.0, ts=D0 + 3, idem_key="job#1")
    assert led.count() == 1


def test_zero_amount_at_the_cap_is_allowed_and_no_caps_means_unlimited(led):
    rwb(led, amount=5.0, daily_cap=5.0, ts=D0 + 1)
    rwb(led, amount=0.0, daily_cap=5.0, ts=D0 + 2)
    rwb(led, amount=1e9, ts=D0 + 3)
    assert led.count() == 3


def test_record_within_budget_validates_caps(led):
    with pytest.raises((TypeError, ValueError)):
        rwb(led, daily_cap=-1.0)
    with pytest.raises((TypeError, ValueError)):
        rwb(led, total_cap=float("nan"))
    assert led.count() == 0


# ---------------------------------------------------------------- 동시성: 같은 파일, 여러 핸들
def test_two_handles_see_each_other(tmp_path, clk):
    path = tmp_path / "shared.sqlite"
    with open_ledger(path, clock=clk) as a, open_ledger(path, clock=clk) as b:
        rwb(a, amount=3.0, daily_cap=5.0, ts=D0 + 1)
        assert b.total("usd") == 3.0
        with pytest.raises(BudgetExceeded):
            rwb(b, amount=3.0, daily_cap=5.0, ts=D0 + 2)
        rwb(b, amount=2.0, daily_cap=5.0, ts=D0 + 3)
        assert a.total("usd") == 5.0 and a.count() == 2


def test_concurrent_writers_never_overshoot_the_cap(tmp_path):
    path = tmp_path / "race.sqlite"
    with open_ledger(path):
        pass  # 스키마를 먼저 만든다
    n_threads, tries = 8, 3
    barrier = threading.Barrier(n_threads)
    results = [[] for _ in range(n_threads)]

    def worker(k):
        try:
            with open_ledger(path, clock=Clock(D0 + 5.0)) as lg:
                barrier.wait(timeout=30)
                for j in range(tries):
                    try:
                        rwb(lg, amount=1.0, run_id=f"t{k}", daily_cap=None, total_cap=5.0, ts=D0 + 5.0,
                            idem_key=f"t{k}#{j}")
                        results[k].append("ok")
                    except BudgetExceeded:
                        results[k].append("over")
        except BaseException as e:  # noqa: BLE001
            results[k].append(f"error:{type(e).__name__}:{e}")

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=45)
    assert not any(t.is_alive() for t in threads)
    flat = [x for r in results for x in r]
    assert not [x for x in flat if x.startswith("error")], flat
    assert flat.count("ok") == 5 and flat.count("over") == n_threads * tries - 5
    with open_ledger(path) as lg:
        assert lg.count() == 5 and lg.total("usd") == 5.0
