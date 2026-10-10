"""RUN-07 한국어 요약: 정확한 줄 형식, 경고, 장부 없음, 비밀 비노출."""
from run07_helpers import HEADER, NOW, SECRET, item

EXPECTED = [
    HEADER,
    "진행: 완료 3/7 (42.9%) · 진행 중 1 · 막힘 1 · 대기 2",
    "지난 1시간 완료: 2건 — A, C",
    "실소요: A 2.00h, C 0.50h",
]


def test_full_summary_lines(hr, data):
    text = hr.summarize_ko(data)
    lines = text.split("\n")
    assert text.endswith("\n") and not text.endswith("\n\n")
    lines = lines[:-1]
    assert lines[:4] == EXPECTED
    assert lines[4].startswith("막힘: E — 게이트 3회 실패 ") and "***" in lines[4]
    assert lines[5:] == [
        "지출(지난 1시간): $5.50 · GPU 0.25h · TFU 12.5",
        "지출(오늘 UTC): $2.00 / 일 캡 $10.00 (20%)",
        "지출(누적): $6.50 / 누적 캡 $35.00 (19%)",
    ]
    assert len(lines) <= 30


def test_summary_never_leaks_secret(hr, data):
    assert SECRET not in hr.summarize_ko(data)


def test_summary_is_deterministic(hr, data, items, ledger, cfg):
    again = hr.collect(now=NOW, items=items, ledger=ledger, config=cfg)
    assert hr.summarize_ko(data) == hr.summarize_ko(again)


def test_summary_without_caps(hr, items, ledger):
    text = hr.summarize_ko(hr.collect(now=NOW, items=items, ledger=ledger))
    assert "지출(오늘 UTC): $2.00 / 일 캡 없음" in text.split("\n")
    assert "지출(누적): $6.50 / 누적 캡 없음" in text.split("\n")


def test_summary_zero_completions_and_no_blocked(hr, ledger, cfg):
    its = [item("A", "done", [{"ts": NOW - 90000, "event": "claimed"}, {"ts": NOW - 80000, "event": "done"}]),
           item("B", "todo")]
    lines = hr.summarize_ko(hr.collect(now=NOW, items=its, ledger=ledger, config=cfg)).split("\n")
    assert "진행: 완료 1/2 (50.0%) · 진행 중 0 · 막힘 0 · 대기 1" in lines
    assert "지난 1시간 완료: 0건" in lines
    assert not any(ln.startswith(("실소요", "막힘", "경고")) for ln in lines)


def test_summary_window_label_follows_window(hr, items, ledger, cfg):
    lines = hr.summarize_ko(hr.collect(now=NOW, items=items, ledger=ledger, config=cfg, window_h=2.0)).split("\n")
    assert "지난 2시간 완료: 3건 — A, B, C" in lines
    assert "지출(지난 2시간): $6.50 · GPU 0.25h · TFU 12.5" in lines


def test_summary_blocked_without_note(hr, ledger):
    its = [item("M", "blocked", [{"ts": NOW - 5, "event": "claimed"}]),
           item("L", "blocked", [{"ts": NOW - 5, "event": "blocked", "note": "원인 불명"}])]
    lines = hr.summarize_ko(hr.collect(now=NOW, items=its, ledger=ledger)).split("\n")
    assert "막힘: L — 원인 불명" in lines and "막힘: M" in lines
    assert lines.index("막힘: L — 원인 불명") < lines.index("막힘: M")


def test_summary_ledger_missing_does_not_invent_zero(hr, items):
    lines = hr.summarize_ko(hr.collect(now=NOW, items=items, ledger=None)).split("\n")
    assert "지출: 장부 없음 — 기록을 읽지 못했다" in lines
    assert not any(ln.startswith("지출(") for ln in lines)
    assert "$0.00" not in "\n".join(lines)


def test_summary_empty_backlog_percent(hr):
    lines = hr.summarize_ko(hr.collect(now=NOW, items=[], ledger=None)).split("\n")
    assert "진행: 완료 0/0 (0.0%) · 진행 중 0 · 막힘 0 · 대기 0" in lines


def test_summary_warning_line(hr, items):
    from buc.core.config import Config
    from buc.core.ledger import LedgerEvent, open_ledger
    led = open_ledger(":memory:")
    try:
        led.record(LedgerEvent(kind="usd", amount=2.0, run_id="r", ts=NOW - 60))
        d = hr.collect(now=NOW, items=items, ledger=led, config=Config(budget_daily_usd=2.4, budget_total_usd=35.0))
        lines = hr.summarize_ko(d).split("\n")
        assert lines[-2] == "경고: 일 예산 83% 사용"
        assert "지출(오늘 UTC): $2.00 / 일 캡 $2.40 (83%)" in lines
    finally:
        led.close()


def test_summary_header_uses_kst_and_utc(hr):
    import calendar
    t = calendar.timegm((2026, 12, 31, 20, 5, 59))      # KST로는 다음 해 1월 1일 05:05
    first = hr.summarize_ko(hr.collect(now=t, items=[], ledger=None)).split("\n")[0]
    assert first == "[BUC 시간별 보고] 2027-01-01 05:05 KST (2026-12-31 20:05Z)"
