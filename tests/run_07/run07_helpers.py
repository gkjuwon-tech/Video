"""RUN-07 테스트 공용: 모듈 적재(파일 경로), 고정 시각, 합성 backlog·장부."""
import calendar
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ops" / "report" / "hourly.py"

# 2026-10-11 00:30:00Z = 09:30 KST. 직전 UTC일 마지막 30분은 now-1800 이전.
NOW = calendar.timegm((2026, 10, 11, 0, 30, 0))
H = 3600
DAY_START = calendar.timegm((2026, 10, 11, 0, 0, 0))
HEADER = "[BUC 시간별 보고] 2026-10-11 09:30 KST (2026-10-11 00:30Z)"
SECRET = "hf_" + "A" * 24


def load_module():
    name = "buc_run07_hourly"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def item(uid, status, log=()):
    return {"id": uid, "status": status, "attempts": 1, "log": [dict(e) for e in log]}


def make_items():
    """done 3(A: 창 안·2.00h, B: 창 밖 4000s 전·4.44h, C: 창 안·0.50h), doing 1, blocked 1(E, 비밀 포함 사유), todo 2."""
    return [
        item("A", "done", [{"ts": NOW - 9000, "event": "claimed"}, {"ts": NOW - 1800, "event": "done"}]),
        item("B", "done", [{"ts": NOW - 20000, "event": "claimed"}, {"ts": NOW - 15000, "event": "blocked", "note": "x"},
                           {"ts": NOW - 4000, "event": "done"}]),
        item("C", "done", [{"ts": NOW - 1900, "event": "claimed"}, {"ts": NOW - 100, "event": "done"}]),
        item("D", "doing", [{"ts": NOW - 300, "event": "claimed"}]),
        item("E", "blocked", [{"ts": NOW - 7000, "event": "claimed"},
                              {"ts": NOW - 600, "event": "blocked", "note": f"게이트 3회 실패 {SECRET} 로그 확인"}]),
        item("F", "todo"), item("G", "todo"),
    ]


def fill_ledger(led):
    """USD: now-7200:1.00(전날) · now-3600:3.00(전날, 창 시작 포함) · now-3000:0.50(전날) · now-1200:2.00(오늘)
    · now:4.00(제외) · now+600:9.00(제외). GPU 0.25h·TFU 12.5는 now-600."""
    from buc.core.ledger import LedgerEvent
    for i, (dt, amt) in enumerate([(-7200, 1.0), (-3600, 3.0), (-3000, 0.5), (-1200, 2.0), (0, 4.0), (600, 9.0)]):
        led.record(LedgerEvent(kind="usd", amount=amt, run_id=f"r{i}", ts=NOW + dt))
    led.record(LedgerEvent(kind="gpu_h", amount=0.25, run_id="g", ts=NOW - 600))
    led.record(LedgerEvent(kind="tfu", amount=12.5, run_id="g", ts=NOW - 600))
