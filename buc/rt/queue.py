"""RUN-04 잡 큐·GPU 예산 집행. SQLite 한 파일, 장부(FND-07)는 읽기만 한다."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from buc.core.config import Config, canonical_json
from buc.core.ledger import Ledger

SCHEMA_VERSION = 1
CLASSES = ("ab", "mrh", "train", "n")
STATES = ("queued", "running", "done", "failed", "cancelled")
PRIORITY_LIMIT = 500_000
_BUSY_MS = 10_000
_TOL = 1e-9
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_KINDS = ("gpu_h", "usd")


class QueueError(RuntimeError):
    pass


class JobNotFound(QueueError, KeyError):
    def __str__(self):
        return RuntimeError.__str__(self)


class LeaseLost(QueueError):
    pass


class QueueBudgetError(QueueError):
    def __init__(self, message, *, reason, kind):
        super().__init__(message)
        self.reason = reason
        self.kind = kind


@dataclass(frozen=True, kw_only=True)
class Job:
    id: str
    cls: str
    priority: int
    est_gpu_h: float
    gpu: bool
    payload: dict
    max_attempts: int
    not_before: float | None
    state: str
    attempt: int
    worker_id: str | None
    lease_until: float | None
    enqueued_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None
    result: dict | None
    seq: int

    @property
    def run_id(self) -> str:
        return f"job:{self.id}"

    @property
    def rank(self) -> int:
        return CLASSES.index(self.cls)

    @property
    def effective_priority(self) -> int:
        return (len(CLASSES) - 1 - self.rank) * 1_000_000 + self.priority


# --------------------------------------------------------------------------- 검증

def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _real(name, x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name}: 숫자여야 합니다")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name}: 유한한 수여야 합니다")
    return x


def _check_id(job_id):
    if not isinstance(job_id, str):
        raise TypeError("job_id: str이어야 합니다")
    if _ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"job_id 형식 오류: {job_id!r}")


def _check_kind(kind):
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


def _dump_payload(payload) -> str:
    if not isinstance(payload, dict):
        raise TypeError("payload: dict여야 합니다")
    try:
        return canonical_json(payload).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"payload: JSON 직렬화 불가: {e}") from e


_COLS = ("seq, id, cls, priority, est_gpu_h, gpu, payload, max_attempts, not_before, state, attempt, "
         "worker_id, lease_until, enqueued_at, started_at, finished_at, error, result")

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS jobs (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        id           TEXT NOT NULL UNIQUE,
        cls          TEXT NOT NULL,
        priority     INTEGER NOT NULL,
        est_gpu_h    REAL NOT NULL,
        gpu          INTEGER NOT NULL,
        payload      TEXT NOT NULL,
        max_attempts INTEGER NOT NULL,
        not_before   REAL,
        state        TEXT NOT NULL,
        attempt      INTEGER NOT NULL DEFAULT 0,
        worker_id    TEXT,
        lease_until  REAL,
        enqueued_at  REAL NOT NULL,
        started_at   REAL,
        finished_at  REAL,
        error        TEXT,
        result       TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state)",
)


def _row_to_job(r) -> Job:
    return Job(
        seq=r[0], id=r[1], cls=r[2], priority=r[3], est_gpu_h=float(r[4]), gpu=bool(r[5]),
        payload=json.loads(r[6]), max_attempts=r[7], not_before=r[8], state=r[9], attempt=r[10],
        worker_id=r[11], lease_until=r[12], enqueued_at=r[13], started_at=r[14], finished_at=r[15],
        error=r[16], result=None if r[17] is None else json.loads(r[17]))


def _dispatch_key(j: Job):
    return (j.rank, -j.priority, j.seq)


class JobQueue:
    def __init__(self, conn, *, ledger, config, clock, usd_per_h, max_running):
        self._conn = conn
        self._ledger = ledger
        self._config = config
        self._clock = clock
        self._usd_per_h = usd_per_h
        self._max_running = max_running
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

    def _db(self):
        if self._closed:
            raise QueueError("닫힌 큐는 쓸 수 없습니다")
        return self._conn

    def _now(self, now):
        return float(self._clock()) if now is None else _real("now", now)

    @contextmanager
    def _tx(self):
        conn = self._db()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- 내부 조회
    def _fetch(self, conn, job_id) -> Job:
        r = conn.execute(f"SELECT {_COLS} FROM jobs WHERE id=?", (job_id,)).fetchone()
        if r is None:
            raise JobNotFound(f"no such job: {job_id!r}")
        return _row_to_job(r)

    def _need(self, kind, est) -> float:
        return est if kind == "gpu_h" else est * self._usd_per_h

    def _held_locked(self, conn, kind) -> float:
        parts = []
        for r in conn.execute(f"SELECT {_COLS} FROM jobs WHERE state='running' AND gpu=1 ORDER BY seq"):
            j = _row_to_job(r)
            need = self._need(kind, j.est_gpu_h)
            parts.append(max(0.0, need - self._ledger.total(kind, run_id=j.run_id)))
        return math.fsum(parts)

    def _reap_locked(self, conn, now) -> list:
        out = []
        rows = conn.execute("SELECT id, attempt, max_attempts FROM jobs WHERE state='running' "
                            "AND lease_until IS NOT NULL AND lease_until <= ? ORDER BY seq", (now,)).fetchall()
        for jid, attempt, max_attempts in rows:
            if attempt < max_attempts:
                conn.execute("UPDATE jobs SET state='queued', worker_id=NULL, lease_until=NULL, "
                             "error='lease_expired' WHERE id=?", (jid,))
            else:
                conn.execute("UPDATE jobs SET state='failed', worker_id=NULL, lease_until=NULL, "
                             "finished_at=?, error='lease_expired' WHERE id=?", (now, jid))
            out.append(jid)
        return out

    # -- 공개 API
    def enqueue(self, job_id, *, cls="train", priority=0, est_gpu_h=0.0, gpu=True, payload=None,
                max_attempts=3, not_before=None, now=None) -> Job:
        _check_id(job_id)
        if not isinstance(cls, str) or cls not in CLASSES:
            raise ValueError(f"cls must be one of {CLASSES}")
        if not _is_int(priority):
            raise TypeError("priority: int여야 합니다")
        if abs(priority) >= PRIORITY_LIMIT:
            raise ValueError(f"|priority| < {PRIORITY_LIMIT} 여야 합니다")
        est = _real("est_gpu_h", est_gpu_h)
        if not isinstance(gpu, bool):
            raise TypeError("gpu: bool이어야 합니다")
        if gpu and est <= 0:
            raise ValueError("gpu 잡은 est_gpu_h > 0 이어야 합니다")
        if not gpu and est != 0:
            raise ValueError("gpu=False 잡은 est_gpu_h == 0 이어야 합니다")
        if not _is_int(max_attempts):
            raise TypeError("max_attempts: int여야 합니다")
        if max_attempts < 1:
            raise ValueError("max_attempts >= 1")
        nb = None if not_before is None else _real("not_before", not_before)
        ptxt = _dump_payload({} if payload is None else payload)
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                r = conn.execute(f"SELECT {_COLS} FROM jobs WHERE id=?", (job_id,)).fetchone()
                if r is not None:
                    ex = _row_to_job(r)
                    same = (ex.cls == cls and ex.priority == priority and ex.est_gpu_h == est
                            and ex.gpu == gpu and ex.max_attempts == max_attempts and ex.not_before == nb
                            and _dump_payload(ex.payload) == ptxt)
                    if not same:
                        raise QueueError(f"job {job_id!r} already exists with different arguments")
                    return ex
                if gpu:
                    for kind in _KINDS:
                        if kind == "usd" and not self._usd_per_h > 0:
                            continue
                        need = self._need(kind, est)
                        daily, total = self._config.caps(kind)
                        if daily is not None and need > daily + _TOL:
                            raise QueueBudgetError(f"{kind} need {need} exceeds daily cap {daily}",
                                                   reason="daily", kind=kind)
                        if total is not None and need > total + _TOL:
                            raise QueueBudgetError(f"{kind} need {need} exceeds total cap {total}",
                                                   reason="total", kind=kind)
                conn.execute(
                    "INSERT INTO jobs(id, cls, priority, est_gpu_h, gpu, payload, max_attempts, not_before, "
                    "state, attempt, enqueued_at) VALUES (?,?,?,?,?,?,?,?, 'queued', 0, ?)",
                    (job_id, cls, priority, est, int(gpu), ptxt, max_attempts, nb, ts))
                return self._fetch(conn, job_id)

    def claim(self, worker_id, *, lease_s=900.0, now=None) -> Job | None:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id: 비어 있지 않은 str")
        lease_s = _real("lease_s", lease_s)
        if lease_s <= 0:
            raise ValueError("lease_s > 0")
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                self._reap_locked(conn, ts)
                rows = conn.execute(
                    f"SELECT {_COLS} FROM jobs WHERE state='queued' AND (not_before IS NULL OR not_before <= ?)",
                    (ts,)).fetchall()
                cands = sorted((_row_to_job(r) for r in rows), key=_dispatch_key)
                if not cands:
                    return None
                running_gpu = conn.execute("SELECT COUNT(*) FROM jobs WHERE state='running' AND gpu=1").fetchone()[0]
                held = {}
                chosen = None
                for j in cands:
                    if j.gpu:
                        if running_gpu >= self._max_running:
                            continue
                        ok = True
                        for kind in _KINDS:
                            if kind == "usd" and not self._usd_per_h > 0:
                                continue
                            if kind not in held:
                                held[kind] = self._held_locked(conn, kind)
                            daily, total = self._config.caps(kind)
                            chk = self._ledger.check_budget(kind, held[kind] + self._need(kind, j.est_gpu_h),
                                                            daily_cap=daily, total_cap=total, ts=ts)
                            if not chk.ok:
                                ok = False
                                break
                        if not ok:
                            continue
                    chosen = j
                    break
                if chosen is None:
                    return None
                conn.execute("UPDATE jobs SET state='running', attempt=attempt+1, worker_id=?, lease_until=?, "
                             "started_at=?, error=NULL WHERE id=?", (worker_id, ts + lease_s, ts, chosen.id))
                return self._fetch(conn, chosen.id)

    def reserved(self, kind, *, now=None) -> float:
        _check_kind(kind)
        with self._lock:
            self._db()
            return self._held_locked(self._conn, kind)

    def _owned_running(self, conn, job_id, worker_id) -> Job:
        j = self._fetch(conn, job_id)
        if j.state != "running" or j.worker_id != worker_id:
            raise LeaseLost(f"lease on {job_id!r} not held by {worker_id!r} (state={j.state})")
        return j

    def heartbeat(self, job_id, worker_id, *, lease_s=900.0, now=None) -> Job:
        lease_s = _real("lease_s", lease_s)
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                self._owned_running(conn, job_id, worker_id)
                conn.execute("UPDATE jobs SET lease_until=? WHERE id=?", (ts + lease_s, job_id))
                return self._fetch(conn, job_id)

    def complete(self, job_id, worker_id, *, result=None, now=None) -> Job:
        rtxt = None if result is None else _dump_payload(result)
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                j = self._fetch(conn, job_id)
                if j.state == "done" and j.worker_id == worker_id:
                    return j
                self._owned_running(conn, job_id, worker_id)
                conn.execute("UPDATE jobs SET state='done', finished_at=?, result=?, lease_until=NULL WHERE id=?",
                             (ts, rtxt, job_id))
                return self._fetch(conn, job_id)

    def fail(self, job_id, worker_id, error="", *, retry=True, retry_delay_s=0.0, now=None) -> Job:
        delay = _real("retry_delay_s", retry_delay_s)
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                j = self._owned_running(conn, job_id, worker_id)
                if retry and j.attempt < j.max_attempts:
                    conn.execute("UPDATE jobs SET state='queued', not_before=?, worker_id=NULL, lease_until=NULL, "
                                 "error=? WHERE id=?", (ts + delay if delay else None, str(error), job_id))
                else:
                    conn.execute("UPDATE jobs SET state='failed', finished_at=?, lease_until=NULL, error=? "
                                 "WHERE id=?", (ts, str(error), job_id))
                return self._fetch(conn, job_id)

    def reap(self, *, now=None) -> list:
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                return self._reap_locked(conn, ts)

    def cancel(self, job_id, *, now=None) -> Job:
        with self._lock:
            ts = self._now(now)
            with self._tx() as conn:
                j = self._fetch(conn, job_id)
                if j.state not in ("queued", "running"):
                    return j
                conn.execute("UPDATE jobs SET state='cancelled', finished_at=?, lease_until=NULL WHERE id=?",
                             (ts, job_id))
                return self._fetch(conn, job_id)

    def get(self, job_id) -> Job:
        with self._lock:
            return self._fetch(self._db(), job_id)

    def list(self, state=None) -> list:
        if state is not None and state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        with self._lock:
            conn = self._db()
            if state is None:
                rows = conn.execute(f"SELECT {_COLS} FROM jobs ORDER BY seq").fetchall()
            else:
                rows = conn.execute(f"SELECT {_COLS} FROM jobs WHERE state=? ORDER BY seq", (state,)).fetchall()
        jobs = [_row_to_job(r) for r in rows]
        if state == "queued":
            jobs.sort(key=_dispatch_key)
        return jobs

    def counts(self) -> dict:
        with self._lock:
            rows = self._db().execute("SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall()
        out = {s: 0 for s in STATES}
        out.update({s: n for s, n in rows})
        return out


def open_queue(path, *, ledger, config, clock=time.time, usd_per_h=0.0, max_running=1) -> JobQueue:
    if not _is_int(max_running) or max_running < 1:
        raise ValueError("max_running: 1 이상의 int")
    usd = _real("usd_per_h", usd_per_h)
    if usd < 0:
        raise ValueError("usd_per_h >= 0")
    if not isinstance(ledger, Ledger):
        raise TypeError("ledger: Ledger여야 합니다")
    if not isinstance(config, Config):
        raise TypeError("config: Config여야 합니다")
    memory = isinstance(path, str) and path == ":memory:"
    if memory:
        conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    else:
        p = Path(os.fspath(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=_BUSY_MS / 1000, check_same_thread=False, isolation_level=None)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_MS}")
        if not memory:
            conn.execute("PRAGMA journal_mode = WAL")
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        if ver > SCHEMA_VERSION:
            raise QueueError(f"queue schema version {ver} is newer than supported {SCHEMA_VERSION}")
        if ver == 0:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                    for stmt in _SCHEMA:
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
    except BaseException:
        conn.close()
        raise
    return JobQueue(conn, ledger=ledger, config=config, clock=clock, usd_per_h=usd, max_running=max_running)
