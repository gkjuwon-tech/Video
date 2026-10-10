"""RUN-07 데이터 수집: 구간 경계, 시간 칸, 단위 실소요, 비밀 가림, 읽기 전용."""
import copy
import json
import math

import pytest
from run07_helpers import NOW, SECRET, H, item, make_items

from buc.core.ledger import LedgerEvent, open_ledger

approx = pytest.approx


def _ev(ts, amt, kind="usd"):
    return LedgerEvent(kind=kind, amount=amt, run_id="r", ts=ts)


def test_hourly_bins_half_open_edges(hr):
    evs = [_ev(NOW - 3600, 1.0), _ev(NOW - 1, 2.0), _ev(NOW, 99.0), _ev(NOW + 5, 99.0),
           _ev(NOW - 24 * H, 4.0), _ev(NOW - 24 * H - 1, 99.0)]
    b = hr.hourly_bins(evs, NOW)
    assert len(b) == 24
    assert b[23] == approx(3.0, abs=1e-12)      # [now-1h, now): 시작 포함, now 제외
    assert b[0] == approx(4.0, abs=1e-12)       # 가장 오래된 칸의 시작 포함, 1초 더 오래된 건 제외
    assert sum(b) == approx(7.0, abs=1e-12)
    assert all(v == 0.0 for v in b[1:23])


def test_hourly_bins_order_and_length(hr):
    evs = [_ev(NOW - 3 * H, 1.0), _ev(NOW - 7000, 0.5), _ev(NOW - 1, 0.25)]
    b = hr.hourly_bins(evs, NOW, hours=3)
    assert b == approx([1.0, 0.5, 0.25], abs=1e-12)           # 오래된 것부터
    assert hr.hourly_bins([], NOW, hours=5) == [0.0] * 5


def test_hourly_bins_sum_is_order_independent(hr):
    amounts = [0.1] * 7 + [1e-9, 3.3, 0.7]
    evs = [_ev(NOW - 100 - i, a) for i, a in enumerate(amounts)]
    a = hr.hourly_bins(evs, NOW)
    b = hr.hourly_bins(list(reversed(evs)), NOW)
    assert a == b
    assert a[23] == math.fsum(amounts)


@pytest.mark.parametrize("kw", [dict(hours=0), dict(hours=-1)])
def test_hourly_bins_rejects_bad_hours(hr, kw):
    with pytest.raises(ValueError):
        hr.hourly_bins([], NOW, **kw)


def test_collect_spend_numbers(hr, data):
    assert data.ledger_available is True
    assert data.spend_window["usd"] == approx(5.5, abs=1e-9)    # 3.0(창 시작) + 0.5 + 2.0
    assert data.spend_today["usd"] == approx(2.0, abs=1e-9)     # UTC 00:00 이후, now 미만
    assert data.spend_total["usd"] == approx(6.5, abs=1e-9)     # ts==now(4.0)·미래(9.0) 제외
    assert data.spend_window["gpu_h"] == approx(0.25, abs=1e-12)
    assert data.spend_window["tfu"] == approx(12.5, abs=1e-12)
    assert data.spend_today["gpu_h"] == approx(0.25, abs=1e-12)
    for k in ("tfu", "gpu_h", "usd"):
        assert k in data.spend_total and k in data.hourly and len(data.hourly[k]) == 24


def test_collect_hourly_and_total_invariant(hr, data):
    u = data.hourly["usd"]
    assert u[23] == approx(5.5, abs=1e-9)
    assert u[22] == approx(1.0, abs=1e-9)
    assert sum(u[:22]) == 0.0
    assert math.fsum(u) == approx(data.spend_total["usd"], abs=1e-9)   # 모든 이벤트가 24시간 안


def test_collect_window_edges(hr, items, ledger):
    d = hr.collect(now=NOW, items=items, ledger=ledger, window_h=2.0)
    assert d.window_start == NOW - 2 * H and d.window_end == NOW and d.window_h == 2.0
    assert d.spend_window["usd"] == approx(6.5, abs=1e-9)      # now-7200 이벤트가 시작 경계로 포함
    d_small = hr.collect(now=NOW, items=items, ledger=ledger, window_h=0.5)
    assert d_small.spend_window["usd"] == approx(2.0, abs=1e-9)   # 1800초 창: 2.0만


def test_collect_caps(hr, data, items, ledger):
    assert data.caps["usd"] == (10.0, 35.0)
    assert data.caps["gpu_h"] == (None, None) and data.caps["tfu"] == (None, None)
    d = hr.collect(now=NOW, items=items, ledger=ledger)
    assert all(v == (None, None) for v in d.caps.values())


def test_collect_counts_and_units(hr, data):
    assert data.counts == {"todo": 2, "doing": 1, "done": 3, "blocked": 1}
    assert data.total_units == 7
    assert data.done_in_window == ["A", "C"]                  # B는 4000초 전이라 1시간 창 밖
    assert data.durations["A"] == approx(2.0, abs=1e-9)
    assert data.durations["C"] == approx(0.5, abs=1e-9)
    assert set(data.durations) == {"A", "C"}


def test_collect_wider_window_includes_multi_attempt_unit(hr, items, ledger):
    d = hr.collect(now=NOW, items=items, ledger=ledger, window_h=2.0)
    assert d.done_in_window == ["A", "B", "C"]
    assert d.durations["B"] == approx(16000 / 3600, abs=1e-6)   # 첫 claimed → 마지막 done (blocked 구간 포함)


def test_done_without_claimed_has_no_duration(hr, ledger):
    its = [item("Z", "done", [{"ts": NOW - 60, "event": "done"}])]
    d = hr.collect(now=NOW, items=its, ledger=ledger)
    assert d.done_in_window == ["Z"] and "Z" not in d.durations


def test_done_window_edges_half_open(hr, ledger):
    its = [item("S", "done", [{"ts": NOW - 7200, "event": "claimed"}, {"ts": NOW - 3600, "event": "done"}]),
           item("T", "done", [{"ts": NOW - 7200, "event": "claimed"}, {"ts": NOW, "event": "done"}]),
           item("U", "done", [{"ts": NOW - 7200, "event": "claimed"}, {"ts": NOW - 3601, "event": "done"}])]
    d = hr.collect(now=NOW, items=its, ledger=ledger)
    assert d.done_in_window == ["S"]                         # 시작 경계 포함, now 제외, 1초 밖 제외
    assert d.counts["done"] == 3


def test_last_done_event_decides_window(hr, ledger):
    its = [item("R", "done", [{"ts": NOW - 90000, "event": "claimed"}, {"ts": NOW - 80000, "event": "done"},
                              {"ts": NOW - 500, "event": "done"}])]
    d = hr.collect(now=NOW, items=its, ledger=ledger)
    assert d.done_in_window == ["R"] and d.durations["R"] == approx((90000 - 500) / 3600, abs=1e-9)


def test_status_done_required_for_window(hr, ledger):
    its = [item("Y", "blocked", [{"ts": NOW - 900, "event": "claimed"}, {"ts": NOW - 60, "event": "done"}])]
    d = hr.collect(now=NOW, items=its, ledger=ledger)
    assert d.done_in_window == []


def test_blocked_note_is_redacted_and_short(hr, data):
    (uid, note), = data.blocked
    assert uid == "E"
    assert SECRET not in note and "***" in note
    assert "게이트 3회 실패" in note
    assert len(note) <= 80


def test_blocked_note_truncated_and_missing(hr, ledger):
    its = [item("L", "blocked", [{"ts": NOW - 5, "event": "blocked", "note": "가" * 300}]),
           item("M", "blocked", [{"ts": NOW - 5, "event": "claimed"}])]
    d = hr.collect(now=NOW, items=its, ledger=ledger)
    assert dict(d.blocked)["M"] == "" and len(dict(d.blocked)["L"]) <= 80
    assert [b[0] for b in d.blocked] == ["L", "M"]


def test_ledger_unavailable_is_not_zero(hr, items):
    d = hr.collect(now=NOW, items=items, ledger=None)
    assert d.ledger_available is False
    assert d.spend_window == {} and d.spend_today == {} and d.spend_total == {} and d.hourly == {}
    assert d.counts["done"] == 3


def test_empty_backlog(hr):
    d = hr.collect(now=NOW, items=[], ledger=None)
    assert d.total_units == 0 and d.counts == {"todo": 0, "doing": 0, "done": 0, "blocked": 0}
    assert d.done_in_window == [] and d.blocked == []


@pytest.mark.parametrize("bad", [dict(now=float("nan")), dict(now=float("inf")), dict(hours=0),
                                 dict(window_h=0), dict(window_h=-1.0)])
def test_collect_validation(hr, bad):
    kw = dict(now=NOW, items=[], ledger=None)
    kw.update(bad)
    with pytest.raises(ValueError):
        hr.collect(**kw)


def test_unknown_status_rejected(hr):
    with pytest.raises(ValueError):
        hr.collect(now=NOW, items=[item("Q", "weird")], ledger=None)


def test_collect_is_read_only(hr, items, ledger, cfg):
    before_items = copy.deepcopy(items)
    n = ledger.count()
    hr.collect(now=NOW, items=items, ledger=ledger, config=cfg)
    assert items == before_items and ledger.count() == n


def test_report_data_to_dict_is_json(hr, data):
    s = json.dumps(data.to_dict(), allow_nan=False, ensure_ascii=False)
    back = json.loads(s)
    assert back["total_units"] == 7 and back["ledger_available"] is True
    assert back["spend_today"]["usd"] == approx(2.0, abs=1e-9)
    assert SECRET not in s


def test_alerts_boundary(hr, items):
    from buc.core.config import Config
    led = open_ledger(":memory:")
    try:
        led.record(LedgerEvent(kind="usd", amount=8.0, run_id="r", ts=NOW - 60))
        d = hr.collect(now=NOW, items=items, ledger=led, config=Config(budget_daily_usd=10.0, budget_total_usd=1000.0))
        assert d.alerts == ["경고: 일 예산 80% 사용"]          # 8/10 = 경계 포함
        d = hr.collect(now=NOW, items=items, ledger=led, config=Config(budget_daily_usd=10.01, budget_total_usd=10.0))
        assert d.alerts == ["경고: 누적 예산 80% 사용"]          # 일 7.99/10.01 미만, 누적 8/10
        d = hr.collect(now=NOW, items=items, ledger=led, config=Config(budget_daily_usd=10.5, budget_total_usd=1000.0))
        assert d.alerts == []                                    # 76% < 80%
        d = hr.collect(now=NOW, items=items, ledger=led, config=Config(budget_daily_usd=0.0, budget_total_usd=0.0))
        assert d.alerts == []                                    # 캡 0은 사용률을 정의하지 않는다
    finally:
        led.close()


@pytest.mark.parametrize("raw,must_not", [
    ("토큰 hf_" + "a1B2c3D4e5" * 2, "hf_a1B2"),
    ("키 sk-" + "x9" * 12, "sk-x9"),
    ("push ghp_" + "Z" * 20, "ghp_ZZZ"),
    ("HF_TOKEN=abcd1234 실패", "abcd1234"),
    ("api_key: s3cr3tvalue 끝", "s3cr3tvalue"),
    ("Password = hunter2!!", "hunter2"),
    ("해시 " + "0123456789abcdef" * 3, "0123456789abcdef0123456789abcdef"),
])
def test_redact_masks_secrets(hr, raw, must_not):
    out = hr.redact(raw)
    assert must_not not in out and "***" in out


@pytest.mark.parametrize("plain", [
    "pytest 157 passed, 0 failed", "FND-07 계약서 작성 완료", "게이트 3회 실패: 경로 오류", "", "a" * 31,
])
def test_redact_keeps_plain_text(hr, plain):
    assert hr.redact(plain) == plain


def test_load_backlog_real_file_shape(hr, tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"version": 1, "items": make_items()}, ensure_ascii=False), encoding="utf-8")
    its = hr.load_backlog(p)
    assert [i["id"] for i in its] == list("ABCDEFG")
    with pytest.raises(FileNotFoundError):
        hr.load_backlog(tmp_path / "none.json")
    for bad in ("{not json", "[]", '{"items": 3}', '{"foo": []}'):
        q = tmp_path / "bad.json"
        q.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            hr.load_backlog(q)
