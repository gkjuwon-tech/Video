"""RUN-07 시간별 보고 생성기: 하네스 진행·최근 구간 지출·단위 실소요를 PNG 한 장과 한국어 요약(.md)으로 쓴다.

읽기 전용: 장부·backlog를 바꾸지 않고, 없는 장부 파일을 만들지 않는다. numpy·torch·matplotlib 미사용(PIL + 표준 라이브러리).
사용: python3 ops/report/hourly.py --out work/reports
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import io
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__" and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from buc.core.config import ConfigError, load_config  # noqa: E402
from buc.core.ledger import KINDS, LedgerError, open_ledger, utc_day_bounds  # noqa: E402

SCHEMA_VERSION = 1
HOURS = 24
WARN_FRAC = 0.8
KST_OFFSET_S = 32400
CANVAS = (1280, 720)
CHART_BOX = (80, 150, 1200, 430)
PROGRESS_BOX = (80, 540, 1200, 580)
COLOR_BG = (15, 17, 21)
COLOR_USD = (122, 162, 247)
COLOR_DONE = (158, 206, 106)
COLOR_TRACK = (42, 46, 54)
COLOR_GRID = (34, 38, 46)
COLOR_TEXT = (205, 210, 220)
COLOR_DIM = (130, 136, 148)
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
)
STATUSES = ("todo", "doing", "done", "blocked")
_NOTE_MAX = 80
_BLOCKED_SHOWN = 12


# ---------------------------------------------------------------- 공통 검사·가림

def _finite(name: str, x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name}: 숫자여야 합니다")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name}: 유한한 값이어야 합니다")
    return v


def _check_hours(hours) -> int:
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
        raise ValueError("hours: 1 이상의 정수여야 합니다")
    return hours


def _valid_ts(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


_TOKEN_RES = (
    re.compile(r"hf_[A-Za-z0-9_]{4,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{8,}"),
)
_KV_RE = re.compile(r"(api_key|token|secret|password)\s*[=:]\s*\S+", re.IGNORECASE)
_LONG_RE = re.compile(r"[A-Za-z0-9]{32,}")


def redact(text: str) -> str:
    """토큰 형태·키=값 비밀·32자 이상 영숫자 연속을 ***로 가린다."""
    out = str(text)
    for rx in _TOKEN_RES:
        out = rx.sub("***", out)
    out = _KV_RE.sub(lambda m: f"{m.group(1)}=***", out)
    return _LONG_RE.sub("***", out)


def load_backlog(path) -> list[dict]:
    """{"items": [...]} 형태의 backlog를 읽는다. 없으면 FileNotFoundError, 형태·JSON 오류는 ValueError."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"backlog JSON 오류: {e}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        raise ValueError("backlog 형태 오류: {'items': [...]} 여야 합니다")
    items = obj["items"]
    if not all(isinstance(i, dict) for i in items):
        raise ValueError("backlog 형태 오류: items의 각 항목은 객체여야 합니다")
    return items


# ---------------------------------------------------------------- 집계

def hourly_bins(events, now, hours: int = HOURS) -> list[float]:
    """한 kind의 LedgerEvent들을 시간 칸으로 나눈다. 칸 i = [now-(hours-i)h, now-(hours-i-1)h), 오래된 칸부터."""
    now = _finite("now", now)
    hours = _check_hours(hours)
    start = now - hours * 3600.0
    buckets: list[list[float]] = [[] for _ in range(hours)]
    for ev in events:
        ts = ev.ts
        if ts is None or not (start <= ts < now):
            continue
        idx = int((ts - start) // 3600.0)
        idx = min(max(idx, 0), hours - 1)
        buckets[idx].append(float(ev.amount))
    return [math.fsum(b) for b in buckets]


@dataclass(frozen=True, kw_only=True)
class ReportData:
    schema: int
    now: float
    window_start: float
    window_end: float
    window_h: float
    hours: int
    ledger_available: bool
    spend_window: dict
    spend_today: dict
    spend_total: dict
    hourly: dict
    caps: dict
    counts: dict
    total_units: int
    done_in_window: list
    durations: dict
    blocked: list
    alerts: list

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "now": self.now,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_h": self.window_h,
            "hours": self.hours,
            "ledger_available": self.ledger_available,
            "spend_window": copy.deepcopy(self.spend_window),
            "spend_today": copy.deepcopy(self.spend_today),
            "spend_total": copy.deepcopy(self.spend_total),
            "hourly": {k: list(v) for k, v in self.hourly.items()},
            "caps": {k: [v[0], v[1]] for k, v in self.caps.items()},
            "counts": dict(self.counts),
            "total_units": self.total_units,
            "done_in_window": list(self.done_in_window),
            "durations": dict(self.durations),
            "blocked": [[uid, note] for uid, note in self.blocked],
            "alerts": list(self.alerts),
        }


def _log_before(item: dict, now: float) -> list[dict]:
    """ts < now인 log 항목만 (ts==now와 미래 이벤트는 어디에도 세지 않는다)."""
    log = item.get("log") or []
    if not isinstance(log, list):
        return []
    return [e for e in log if isinstance(e, dict) and _valid_ts(e.get("ts")) and float(e["ts"]) < now]


def collect(*, now, items, ledger=None, config=None, window_h: float = 1.0, hours: int = HOURS) -> ReportData:
    now = _finite("now", now)
    window_h = _finite("window_h", window_h)
    if window_h <= 0:
        raise ValueError("window_h: 0보다 커야 합니다")
    hours = _check_hours(hours)
    w_start = now - window_h * 3600.0
    day_start, _ = utc_day_bounds(now)

    counts = {s: 0 for s in STATUSES}
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("items의 각 항목은 객체여야 합니다")
        st = it.get("status")
        if st not in counts:
            raise ValueError(f"알 수 없는 status: {st!r}")
        counts[st] += 1

    done_in_window: list[str] = []
    durations: dict[str, float] = {}
    blocked: list[tuple[str, str]] = []
    for it in items:
        uid = str(it.get("id"))
        status = it.get("status")
        log = _log_before(it, now)
        if status == "done":
            done_ts = [float(e["ts"]) for e in log if e.get("event") == "done"]
            if done_ts:
                last = max(done_ts)
                if w_start <= last < now:
                    done_in_window.append(uid)
                    claimed = [float(e["ts"]) for e in log if e.get("event") == "claimed"]
                    if claimed and min(claimed) <= last:
                        durations[uid] = (last - min(claimed)) / 3600.0
        elif status == "blocked":
            notes = [e for e in log if e.get("event") == "blocked"]
            note = ""
            if notes:
                note = str(max(notes, key=lambda e: float(e["ts"])).get("note") or "")
            blocked.append((uid, redact(note)[:_NOTE_MAX]))
    done_in_window.sort()
    blocked.sort(key=lambda x: x[0])
    durations = {k: durations[k] for k in sorted(durations)}

    ledger_available = ledger is not None
    spend_window: dict = {}
    spend_today: dict = {}
    spend_total: dict = {}
    hourly: dict = {}
    if ledger is not None:
        for k in KINDS:
            spend_window[k] = ledger.total(k, since=w_start, until=now)
            spend_today[k] = ledger.total(k, since=day_start, until=now)
            spend_total[k] = ledger.total(k, until=now)
            evs = ledger.events(kind=k, since=now - hours * 3600.0, until=now)
            hourly[k] = hourly_bins(evs, now, hours)

    caps = {k: (config.caps(k) if config is not None else (None, None)) for k in KINDS}

    alerts: list[str] = []
    if ledger is not None:
        dcap, tcap = caps["usd"]
        for name, spent, cap in (
            ("일 예산", spend_today.get("usd", 0.0), dcap),
            ("누적 예산", spend_total.get("usd", 0.0), tcap),
        ):
            if cap is not None and cap > 0 and spent / cap >= WARN_FRAC:
                alerts.append(f"경고: {name} {spent / cap * 100:.0f}% 사용")

    return ReportData(
        schema=SCHEMA_VERSION,
        now=now,
        window_start=w_start,
        window_end=now,
        window_h=window_h,
        hours=hours,
        ledger_available=ledger_available,
        spend_window=spend_window,
        spend_today=spend_today,
        spend_total=spend_total,
        hourly=hourly,
        caps=caps,
        counts=counts,
        total_units=len(items),
        done_in_window=done_in_window,
        durations=durations,
        blocked=blocked,
        alerts=alerts,
    )


# ---------------------------------------------------------------- 요약 (한국어, 결정론)

def _num(x: float) -> str:
    """소수 둘째 자리까지, 뒤의 0 제거."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _stamp(now: float) -> str:
    kst = _dt.datetime.fromtimestamp(now + KST_OFFSET_S, _dt.timezone.utc)
    utc = _dt.datetime.fromtimestamp(now, _dt.timezone.utc)
    return f"{kst:%Y-%m-%d %H:%M} KST ({utc:%Y-%m-%d %H:%M}Z)"


def _cap_line(label: str, spent: float, cap, capname: str) -> str:
    if cap is None:
        return f"{label}: ${spent:.2f} / {capname} 없음"
    if cap == 0:
        return f"{label}: ${spent:.2f} / {capname} ${cap:.2f}"
    return f"{label}: ${spent:.2f} / {capname} ${cap:.2f} ({spent / cap * 100:.0f}%)"


def summarize_ko(d: ReportData) -> str:
    """결정론적 한국어 요약. 개행으로 끝나고 30줄 이하, 비밀을 담지 않는다."""
    w = _num(d.window_h)
    total = d.total_units
    done = d.counts["done"]
    pct = (done / total * 100.0) if total else 0.0
    lines = [
        f"[BUC 시간별 보고] {_stamp(d.now)}",
        f"진행: 완료 {done}/{total} ({pct:.1f}%) · 진행 중 {d.counts['doing']} · 막힘 {d.counts['blocked']} · 대기 {d.counts['todo']}",
    ]
    if d.done_in_window:
        lines.append(f"지난 {w}시간 완료: {len(d.done_in_window)}건 — {', '.join(d.done_in_window)}")
    else:
        lines.append(f"지난 {w}시간 완료: 0건")
    dur = [uid for uid in d.done_in_window if uid in d.durations]
    if dur:
        lines.append("실소요: " + ", ".join(f"{uid} {d.durations[uid]:.2f}h" for uid in dur))
    for uid, note in d.blocked[:_BLOCKED_SHOWN]:
        lines.append(f"막힘: {uid} — {note}" if note else f"막힘: {uid}")
    if len(d.blocked) > _BLOCKED_SHOWN:
        lines.append(f"막힘: 외 {len(d.blocked) - _BLOCKED_SHOWN}건")
    if not d.ledger_available:
        lines.append("지출: 장부 없음 — 기록을 읽지 못했다")
    else:
        lines.append(
            f"지출(지난 {w}시간): ${d.spend_window.get('usd', 0.0):.2f}"
            f" · GPU {_num(d.spend_window.get('gpu_h', 0.0))}h"
            f" · TFU {_num(d.spend_window.get('tfu', 0.0))}"
        )
        dcap, tcap = d.caps["usd"]
        lines.append(_cap_line("지출(오늘 UTC)", d.spend_today.get("usd", 0.0), dcap, "일 캡"))
        lines.append(_cap_line("지출(누적)", d.spend_total.get("usd", 0.0), tcap, "누적 캡"))
    lines.extend(d.alerts)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 그림

def bar_rects(values, box, vmax=None) -> list[tuple[int, int, int, int]]:
    """막대 사각형 (xa, ya, xb, yb), x1·y1 배타. 높이는 vmax(기본 max(values)) 기준."""
    x0, y0, x1, y1 = box
    if not (x1 > x0 and y1 > y0):
        raise ValueError("box: 퇴화된 박스입니다")
    vals = [float(v) for v in values]
    for v in vals:
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"막대 값은 유한한 0 이상이어야 합니다: {v!r}")
    if not vals:
        return []
    if vmax is None:
        vm = max(vals) or 1.0
    else:
        vm = float(vmax)
        if not math.isfinite(vm) or vm <= 0:
            raise ValueError("vmax: 0보다 큰 유한한 값이어야 합니다")
    n = len(vals)
    s = (x1 - x0) / n
    rects = []
    for i, v in enumerate(vals):
        h = min(round((y1 - y0) * v / vm), y1 - y0)
        xa = round(x0 + i * s + 0.15 * s)
        xb = round(x0 + (i + 1) * s - 0.15 * s)
        rects.append((xa, y1 - h, xb, y1))
    return rects


def _font(size: int):
    for cand in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_chart(draw: ImageDraw.ImageDraw, d: ReportData) -> None:
    x0, y0, x1, y1 = CHART_BOX
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = y1 - round((y1 - y0) * frac)
        draw.line([(x0, y), (x1 - 1, y)], fill=COLOR_GRID)
    vals = d.hourly.get("usd") if d.ledger_available else None
    if not vals:
        return
    for xa, ya, xb, yb in bar_rects(vals, CHART_BOX):
        if xb > xa and yb > ya:
            draw.rectangle([xa, ya, xb - 1, yb - 1], fill=COLOR_USD)


def _draw_progress(draw: ImageDraw.ImageDraw, d: ReportData) -> None:
    x0, y0, x1, y1 = PROGRESS_BOX
    total = d.total_units
    done = d.counts["done"]
    fw = round((x1 - x0) * done / total) if total else 0
    if fw > 0:
        draw.rectangle([x0, y0, x0 + fw - 1, y1 - 1], fill=COLOR_DONE)
    if fw < x1 - x0:
        draw.rectangle([x0 + fw, y0, x1 - 1, y1 - 1], fill=COLOR_TRACK)


def _draw(d: ReportData) -> Image.Image:
    img = Image.new("RGB", CANVAS, COLOR_BG)
    draw = ImageDraw.Draw(img)
    f_big, f_body, f_small = _font(24), _font(16), _font(14)
    lines = summarize_ko(d).split("\n")[:-1]

    draw.text((80, 22), lines[0], font=f_big, fill=COLOR_TEXT)
    cap = (f"시간별 USD 지출 (최근 {d.hours}시간)" if d.ledger_available
           else "시간별 USD 지출 — 장부 없음, 그림 생략")
    draw.text((80, 118), cap, font=f_small, fill=COLOR_DIM)
    _draw_chart(draw, d)

    draw.text((80, 512), lines[1], font=f_body, fill=COLOR_TEXT)
    _draw_progress(draw, d)

    rest = lines[2:]
    capacity = 10
    if len(rest) > capacity:
        rest = rest[: capacity - 1] + ["… 전체 내용은 .md 파일 참조"]
    for i, ln in enumerate(rest):
        cx = 80 if i < 5 else 660
        cy = 598 + (i % 5) * 22
        draw.text((cx, cy), ln, font=f_body, fill=COLOR_TEXT)
    return img


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def render_png(d: ReportData, path) -> Path:
    """CANVAS 크기 RGB PNG. 같은 입력이면 같은 바이트, 임시 파일 후 os.replace."""
    path = Path(path)
    buf = io.BytesIO()
    _draw(d).save(buf, format="PNG")
    _atomic_write(path, buf.getvalue())
    return path


def write_report(out_dir, d: ReportData) -> tuple[Path, Path]:
    """(png, md). 파일명은 d.now의 UTC 시각 기준 hourly-YYYYMMDDTHHMMZ."""
    out = Path(out_dir)
    stamp = _dt.datetime.fromtimestamp(d.now, _dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
    png = render_png(d, out / f"hourly-{stamp}.png")
    md = out / f"hourly-{stamp}.md"
    _atomic_write(md, summarize_ko(d).encode("utf-8"))
    return png, md


# ---------------------------------------------------------------- CLI

def _fail(why: str) -> int:
    print(json.dumps({"ok": False, "why": redact(why)}, ensure_ascii=False))
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BUC 시간별 보고 (PNG + 한국어 요약)")
    ap.add_argument("--out", default=str(ROOT / "work" / "reports"))
    ap.add_argument("--backlog", default=str(ROOT / "ops" / "harness" / "backlog.json"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--now", type=float, default=None)
    ap.add_argument("--hours", type=int, default=HOURS)
    ap.add_argument("--window-h", type=float, default=1.0)
    a = ap.parse_args(argv)

    now = time.time() if a.now is None else a.now
    try:
        cfg = load_config()
    except (ConfigError, OSError) as e:
        return _fail(f"설정 읽기 실패: {type(e).__name__}")
    try:
        items = load_backlog(a.backlog)
    except (OSError, ValueError) as e:
        return _fail(f"backlog 읽기 실패: {e}")

    ledger_path = a.ledger if a.ledger is not None else cfg.ledger_path
    ledger = None
    if os.path.exists(ledger_path):
        try:
            ledger = open_ledger(ledger_path)
        except (LedgerError, sqlite3.Error, OSError, ValueError):
            ledger = None
    try:
        data = collect(now=now, items=items, ledger=ledger, config=cfg, window_h=a.window_h, hours=a.hours)
        png, md = write_report(a.out, data)
    except (ValueError, OSError) as e:
        return _fail(f"보고 생성 실패: {e}")
    finally:
        if ledger is not None:
            ledger.close()
    print(json.dumps({"ok": True, "png": str(png), "md": str(md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
