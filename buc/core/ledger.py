"""FND-07 장부: 추가 전용 SQLite 이벤트 로그(TFU·GPU-h·USD), 정확한 합계, 멱등 키, 원자적 예산 캡.

계약: ops/harness/contracts/FND-07.md. numpy/torch를 import하지 않는다.
동시성: 파일 DB는 WAL + busy timeout 10 s. 예산 기록은 BEGIN IMMEDIATE 안에서 검사와 쓰기를 한다.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from buc.core.config import canonical_json

__all__ = [
    "KINDS", "TFU_FLOPS", "TFU_PER_FORWARD", "tfu_for_forward", "tfu_from_flops", "gpu_hours",
    "gpu_cost_usd", "LedgerError", "BudgetExceeded", "LedgerEvent", "BudgetCheck", "Ledger",
    "open_ledger", "utc_day_bounds",
]

KINDS = ("tfu", "gpu_h", "usd")
TFU_FLOPS = 6.2e13
TFU_PER_FORWARD = {256: 0.1, 512: 0.25, 768: 0.56, 1024: 1.0}
SCHEMA_VERSION = 1
_DAY = 86400.0
_BUSY_MS = 10_000
_TOL = 1e-9


class LedgerError(Exception):
    """장부 오류(닫힌 뒤 사용, 스키마 버전 불일치, 멱등 키 충돌 등)."""


class BudgetExceeded(LedgerError):
    def __init__(self, check: "BudgetCheck"):
        self.check = check
        self.reason = check.reason
        super().__init__(f"budget exceeded ({check.reason}): kind={check.kind} amount={check.amount}")


# --------------------------------------------------------------------------- 헬퍼

def _real(name, x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name}: 숫자여야 합니다 (받은 타입 {type(x).__name__})")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name}: 유한한 값이어야 합니다")
    return v


def _nonneg(name, x) -> float:
    v = _real(name, x)
    if v < 0:
        raise ValueError(f"{name}: 0 이상이어야 합니다 (받은 값 {x!r})")
    return v


def _to_ts(x, name="ts") -> float:
    if isinstance(x, datetime):
        if x.tzinfo is None:
            raise ValueError(f"{name}: tz-aware datetime이어야 합니다")
        return x.timestamp()
    return _real(name, x)


def utc_day_bounds(ts) -> tuple:
    """ts가 속한 UTC 달력일의 [start, end) (유닉스 초, float)."""
    t = _to_ts(ts)
    start = math.floor(t / _DAY) * _DAY
    return (start, start + _DAY)


def tfu_for_forward(resolution, n=1) -> float:
    if isinstance(resolution, bool) or resolution not in TFU_PER_FORWARD:
        raise ValueError(f"해상도는 {tuple(TFU_PER_FORWARD)} 중 하나여야 합니다 (받은 값 {resolution!r})")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n은 0 이상의 정수여야 합니다 (받은 값 {n!r})")
    return TFU_PER_FORWARD[resolution] * n


def tfu_from_flops(flops) -> float:
    return _nonneg("flops", flops) / TFU_FLOPS


def gpu_hours(seconds) -> float:
    return _nonneg("seconds", seconds) / 3600.0


def gpu_cost_usd(seconds, usd_per_hour) -> float:
    return gpu_hours(seconds) * _nonneg("usd_per_hour", usd_per_hour)


def _check_kind(kind):
    if not isinstance(kind, str):
        raise TypeError("kind는 문자열이어야 합니다")
    if kind not in KINDS:
        raise ValueError(f"kind는 {KINDS} 중 하나여야 합니다 (받은 값 {kind!r})")
    return kind


def _check_cap(name, cap):
    if cap is None:
        return None
    return _nonneg(name, cap)


# --------------------------------------------------------------------------- 값 객체

@dataclass(frozen=True, kw_only=True)
class LedgerEvent:
    kind: str
    amount: float
    run_id: str
    stage: str = ""
    note: str = ""
    meta: dict = field(default_factory=dict)
    idem_key: str | None = None
    ts: float | None = None
    id: int | None = None

    def __post_init__(self):
        set_ = object.__setattr__
        _check_kind(self.kind)
        set_(self, "amount", _nonneg("amount", self.amount))
        if not isinstance(self.run_id, str):
            raise TypeError("run_id는 문자열이어야 합니다")
        if self.run_id == "":
            raise ValueError("run_id는 비어 있으면 안 됩니다")
        if not isinstance(self.stage, str) or not isinstance(self.note, str):
            raise TypeError("stage·note는 문자열이어야 합니다")
        if not isinstance(self.meta, dict):
            raise TypeError("meta는 dict여야 합니다")
        set_(self, "meta", json.loads(canonical_json(self.meta)))
        if self.idem_key is not None:
            if not isinstance(self.idem_key, str):
                raise TypeError("idem_key는 문자열이어야 합니다")
            if self.idem_key == "":
                raise ValueError("idem_key는 비어 있으면 안 됩니다")
        if self.ts is not None:
            set_(self, "ts", _real("ts", self.ts))


@dataclass(frozen=True, kw_only=True)
class BudgetCheck:
    ok: bool
    reason: str
    kind: str
    amount: float
    daily_spent: float
    total_spent: float
    daily_remaining: float | None
    total_remaining: float | None


_SCHEMA_STATEMENTS = (
    """CREATE TABLE events (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        ts       REAL NOT NULL,
        kind     TEXT NOT NULL,
        amount   REAL NOT NULL,
        run_id   TEXT NOT NULL,
        stage    TEXT NOT NULL DEFAULT '',
        note     TEXT NOT NULL DEFAULT '',
        meta     TEXT NOT NULL DEFAULT '{}',
        idem_key TEXT UNIQUE
    )""",
    "CREATE INDEX events_kind_ts ON events(kind, ts)",
    "CREATE INDEX events_run_kind ON events(run_id, kind)",
    "CREATE TRIGGER events_no_update BEFORE UPDATE ON events "
    "BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END",
    "CREATE TRIGGER events_no_delete BEFORE DELETE ON events "
    "BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END",
)


def _row_to_event(row) -> LedgerEvent:
    id_, ts, kind, amount, run_id, stage, note, meta, idem_key = row
    return LedgerEvent(kind=kind, amount=amount, run_id=run_id, stage=stage, note=note,
                       meta=json.loads(meta), idem_key=idem_key, ts=ts, id=id_)


class Ledger:
    """open_ledger로 만든다. 핸들은 여러 스레드에서 써도 되며 내부 락으로 직렬화한다."""

    def __init__(self, conn: sqlite3.Connection, clock):
        self._conn = conn
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False

    # -- 생명주기
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def _open(self) -> sqlite3.Connection:
        if self._closed:
            raise LedgerError("닫힌 장부는 쓸 수 없습니다")
        return self._conn

    @contextmanager
    def _tx(self):
        conn = self._open()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def _now(self, ts):
        return self._clock() if ts is None else ts

    # -- 쓰기
    def _insert(self, conn, ev: LedgerEvent) -> int:
        if ev.idem_key is not None:
            row = conn.execute("SELECT id, kind, amount, run_id, stage FROM events WHERE idem_key=?",
                               (ev.idem_key,)).fetchone()
            if row is not None:
                if (row[1], row[2], row[3], row[4]) != (ev.kind, ev.amount, ev.run_id, ev.stage):
                    raise LedgerError(f"idem_key {ev.idem_key!r}가 다른 내용으로 이미 쓰였습니다")
                return row[0]
        ts = self._now(ev.ts) if ev.ts is None else ev.ts
        cur = conn.execute(
            "INSERT INTO events(ts, kind, amount, run_id, stage, note, meta, idem_key) VALUES (?,?,?,?,?,?,?,?)",
            (float(ts), ev.kind, ev.amount, ev.run_id, ev.stage, ev.note,
             canonical_json(ev.meta).decode("utf-8"), ev.idem_key))
        return int(cur.lastrowid)

    def record(self, ev: LedgerEvent) -> int:
        if not isinstance(ev, LedgerEvent):
            raise TypeError("LedgerEvent여야 합니다")
        with self._lock, self._tx() as conn:
            if ev.ts is None:
                ev = replace(ev, ts=self._now(None))
            return self._insert(conn, ev)

    # -- 읽기
    def _sum_rows(self, kind, since=None, until=None, run_id=None):
        sql = "SELECT run_id, amount, ts FROM events WHERE kind=?"
        args: list = [kind]
        if since is not None:
            sql += " AND ts >= ?"
            args.append(_to_ts(since, "since"))
        if until is not None:
            sql += " AND ts < ?"
            args.append(_to_ts(until, "until"))
        if run_id is not None:
            sql += " AND run_id = ?"
            args.append(run_id)
        return self._open().execute(sql, args).fetchall()

    def total(self, kind, since=None, until=None, run_id=None) -> float:
        _check_kind(kind)
        with self._lock:
            return math.fsum(r[1] for r in self._sum_rows(kind, since, until, run_id))

    def daily_total(self, kind, ts=None) -> float:
        _check_kind(kind)
        start, end = utc_day_bounds(self._now(ts))
        with self._lock:
            return math.fsum(r[1] for r in self._sum_rows(kind, start, end))

    def totals_by_run(self, kind) -> dict:
        _check_kind(kind)
        groups: dict = {}
        with self._lock:
            for run_id, amount, _ in self._sum_rows(kind):
                groups.setdefault(run_id, []).append(amount)
        return {r: math.fsum(v) for r, v in groups.items()}

    def events(self, kind=None, run_id=None, since=None, until=None, limit=None) -> list:
        sql = ("SELECT id, ts, kind, amount, run_id, stage, note, meta, idem_key FROM events WHERE 1=1")
        args: list = []
        if kind is not None:
            sql += " AND kind=?"
            args.append(_check_kind(kind))
        if run_id is not None:
            sql += " AND run_id=?"
            args.append(run_id)
        if since is not None:
            sql += " AND ts >= ?"
            args.append(_to_ts(since, "since"))
        if until is not None:
            sql += " AND ts < ?"
            args.append(_to_ts(until, "until"))
        sql += " ORDER BY id"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("limit은 0 이상의 정수여야 합니다")
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            return [_row_to_event(r) for r in self._open().execute(sql, args).fetchall()]

    def count(self) -> int:
        with self._lock:
            return int(self._open().execute("SELECT COUNT(*) FROM events").fetchone()[0])

    # -- 예산
    def _make_check(self, kind, amount, daily_cap, total_cap, ts) -> BudgetCheck:
        start, end = utc_day_bounds(ts)
        daily_spent = math.fsum(r[1] for r in self._sum_rows(kind, start, end))
        total_spent = math.fsum(r[1] for r in self._sum_rows(kind))
        daily_ok = daily_cap is None or daily_spent + amount <= daily_cap + _TOL
        total_ok = total_cap is None or total_spent + amount <= total_cap + _TOL
        if not daily_ok:
            reason = "daily"
        elif not total_ok:
            reason = "total"
        else:
            reason = ""
        return BudgetCheck(
            ok=daily_ok and total_ok, reason=reason, kind=kind, amount=amount,
            daily_spent=daily_spent, total_spent=total_spent,
            daily_remaining=None if daily_cap is None else max(0.0, daily_cap - daily_spent),
            total_remaining=None if total_cap is None else max(0.0, total_cap - total_spent),
        )

    def check_budget(self, kind, amount, *, daily_cap=None, total_cap=None, ts=None) -> BudgetCheck:
        _check_kind(kind)
        amount = _nonneg("amount", amount)
        daily_cap = _check_cap("daily_cap", daily_cap)
        total_cap = _check_cap("total_cap", total_cap)
        with self._lock:
            self._open()
            return self._make_check(kind, amount, daily_cap, total_cap, self._now(ts))

    def record_within_budget(self, ev: LedgerEvent, *, daily_cap=None, total_cap=None) -> int:
        if not isinstance(ev, LedgerEvent):
            raise TypeError("LedgerEvent여야 합니다")
        daily_cap = _check_cap("daily_cap", daily_cap)
        total_cap = _check_cap("total_cap", total_cap)
        with self._lock, self._tx() as conn:
            if ev.ts is None:
                ev = replace(ev, ts=self._now(None))
            if ev.idem_key is not None:
                row = conn.execute("SELECT id FROM events WHERE idem_key=?", (ev.idem_key,)).fetchone()
                if row is not None:
                    return self._insert(conn, ev)  # 재시도: 예산 검사 없이 기존 id
            check = self._make_check(ev.kind, ev.amount, daily_cap, total_cap, ev.ts)
            if not check.ok:
                raise BudgetExceeded(check)
            return self._insert(conn, ev)


# --------------------------------------------------------------------------- 열기

def open_ledger(path, *, clock=time.time) -> Ledger:
    if isinstance(path, str) and path == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    else:
        p = Path(os.fspath(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=_BUSY_MS / 1000, check_same_thread=False, isolation_level=None)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_MS}")
        if path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                    for stmt in _SCHEMA_STATEMENTS:
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        if ver > SCHEMA_VERSION:
            raise LedgerError(f"장부 스키마 버전 {ver}은 이 코드({SCHEMA_VERSION})보다 새 것입니다")
        return Ledger(conn, clock)
    except BaseException:
        conn.close()
        raise
