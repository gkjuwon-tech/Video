"""RUN-01 Vast.ai 오케스트레이터: 오퍼 검색, 임대, 유휴·부팅 기한 자동 종료, 일/누적 예산 캡.

돈이 새는 코드이므로 모든 지출은 FND-07 장부에 멱등 키와 함께 남기고, 캡은 넘기기 전에 집행한다.
계약서: ops/harness/contracts/RUN-01.md. 표준 라이브러리 + buc.core만 쓰며 import 시 부작용·네트워크가 없다.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from buc.core.config import Config
from buc.core.ledger import Ledger, LedgerEvent, utc_day_bounds

__all__ = [
    "API_BASE", "KEY_ENV", "DEFAULT_KEY_PATH", "DEFAULT_IMAGE",
    "VastError", "VastAuthError", "VastAPIError", "VastNotFound", "NoOfferError", "RentLimitError",
    "BudgetRefused", "VastAPI", "HttpVastAPI", "resolve_api_key", "country_code", "Offer", "OfferQuery",
    "select_offers", "Instance", "VastPolicy", "Action", "TickReport", "VastOrchestrator",
]

API_BASE = "https://console.vast.ai/api/v0"
KEY_ENV = "VAST_API_KEY"
DEFAULT_KEY_PATH = "~/.config/vastai/vast_api_key"
DEFAULT_IMAGE = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"

_EPS = 1e-9
_HOUR = 3600.0


# --------------------------------------------------------------------------- 예외

class VastError(RuntimeError):
    """Vast 오케스트레이터 예외의 뿌리."""


class VastAuthError(VastError):
    pass


class VastAPIError(VastError):
    def __init__(self, message: str = "", status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class VastNotFound(VastError):
    pass


class NoOfferError(VastError):
    pass


class RentLimitError(VastError):
    pass


class BudgetRefused(VastError):
    def __init__(self, reason: str, check=None, message: Optional[str] = None):
        self.reason = reason
        self.check = check
        super().__init__(message or f"budget refused ({reason})")


# --------------------------------------------------------------------------- 키 위생

_SCRUB_RES = (
    re.compile(r"(api_key=)[^&\s\"']+", re.I),
    re.compile(r"(Bearer\s+)[^\s\"']+", re.I),
)


def _scrub(text, secrets=()) -> str:
    s = str(text)
    for sec in secrets:
        if sec:
            s = s.replace(sec, "***")
    env_key = os.environ.get(KEY_ENV)
    if env_key and env_key.strip():
        s = s.replace(env_key.strip(), "***")
    for rx in _SCRUB_RES:
        s = rx.sub(r"\1***", s)
    return s


# --------------------------------------------------------------------------- 키 해석

def resolve_api_key(env=None, key_path=None) -> Optional[str]:
    source = os.environ if env is None else env
    v = source.get(KEY_ENV) if hasattr(source, "get") else None
    if isinstance(v, str) and v.strip():
        return v.strip()
    p = Path(os.path.expanduser(str(DEFAULT_KEY_PATH if key_path is None else key_path)))
    try:
        text = p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return text or None


# --------------------------------------------------------------------------- API 이음매

class VastAPI(Protocol):
    def search_offers(self, query: dict) -> list: ...

    def create_instance(self, offer_id: int, config: dict) -> int: ...

    def show_instance(self, iid: int) -> dict: ...

    def list_instances(self) -> list: ...

    def set_state(self, iid: int, state: str) -> None: ...

    def destroy_instance(self, iid: int) -> None: ...


class HttpVastAPI:
    """Vast REST 어댑터. 키는 Authorization 헤더로만 보낸다."""

    def __init__(self, api_key: str, *, base_url: str = API_BASE, timeout: float = 30.0,
                 urlopen=urllib.request.urlopen):
        if not isinstance(api_key, str) or not api_key.strip():
            raise VastAuthError("Vast API 키가 없습니다")
        self._key = api_key.strip()
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._urlopen = urlopen

    def __repr__(self) -> str:
        return f"HttpVastAPI(base_url={self._base!r}, timeout={self._timeout!r})"

    @classmethod
    def from_env(cls, env=None, key_path=None, **kw) -> "HttpVastAPI":
        key = resolve_api_key(env, key_path)
        if not key:
            raise VastAuthError(f"Vast API 키를 찾을 수 없습니다 ({KEY_ENV} 또는 키 파일)")
        return cls(key, **kw)

    # -- 저수준
    def _request(self, method: str, path: str, body=None):
        url = self._base + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", "Bearer " + self._key)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "buc-vast/0")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        where = f"{method} {path}"
        try:
            with self._urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            code = int(e.code)
            try:
                snippet = _scrub(e.read().decode("utf-8", "replace"), (self._key,))[:200]
            except Exception:
                snippet = ""
            msg = f"HTTP {code} on {where}" + (f": {snippet}" if snippet else "")
            if code in (401, 403):
                raise VastAuthError(msg) from None
            if code == 404:
                raise VastNotFound(msg) from None
            raise VastAPIError(msg, status=code) from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise VastAPIError(f"network error on {where}: {_scrub(type(e).__name__, (self._key,))}") from None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if not raw or not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            raise VastAPIError(f"invalid JSON on {where}") from None

    # -- 이음매
    def search_offers(self, query: dict) -> list:
        out = self._request("POST", "/bundles/", query)
        offers = out.get("offers") if isinstance(out, dict) else out
        return [o for o in (offers or []) if isinstance(o, dict)]

    def create_instance(self, offer_id: int, config: dict) -> int:
        body = {"image": config.get("image"), "disk": config.get("disk"), "label": config.get("label"),
                "env": config.get("env") or {}, "runtype": "ssh"}
        if config.get("onstart"):
            body["onstart"] = config["onstart"]
        out = self._request("PUT", f"/asks/{int(offer_id)}/", body)
        if not isinstance(out, dict) or out.get("success") is False or out.get("new_contract") is None:
            raise VastAPIError("create_instance: 응답에 new_contract 없음")
        try:
            return int(out["new_contract"])
        except (TypeError, ValueError):
            raise VastAPIError("create_instance: new_contract가 정수가 아님") from None

    def show_instance(self, iid: int) -> dict:
        out = self._request("GET", f"/instances/{int(iid)}/")
        inst = out.get("instances", out) if isinstance(out, dict) else out
        if isinstance(inst, list):
            inst = inst[0] if inst else None
        if not isinstance(inst, dict) or not inst:
            raise VastNotFound(f"instance {iid}")
        return inst

    def list_instances(self) -> list:
        out = self._request("GET", "/instances/")
        insts = out.get("instances") if isinstance(out, dict) else out
        if isinstance(insts, dict):
            insts = [insts]
        return [i for i in (insts or []) if isinstance(i, dict)]

    def set_state(self, iid: int, state: str) -> None:
        if state not in ("running", "stopped"):
            raise ValueError(f"state는 running|stopped: {state!r}")
        self._request("PUT", f"/instances/{int(iid)}/", {"state": state})

    def destroy_instance(self, iid: int) -> None:
        self._request("DELETE", f"/instances/{int(iid)}/")


# --------------------------------------------------------------------------- 값 객체

def country_code(geo) -> Optional[str]:
    if not isinstance(geo, str):
        return None
    piece = geo.rsplit(",", 1)[-1].strip()
    if not piece:
        return None
    if piece.lower() == "china":
        return "CN"
    if len(piece) == 2 and piece.isascii() and piece.isalpha():
        return piece.upper()
    return None


def _fnum(d: dict, *keys, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return default


@dataclass(frozen=True)
class Offer:
    id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    dph_total: float
    reliability: float
    geolocation: str
    country: Optional[str]
    cpu_cores: float
    cpu_ram_gb: float
    disk_gb: float
    inet_down: float
    cuda_vers: float
    verified: bool
    rentable: bool

    @classmethod
    def from_api(cls, d: dict) -> "Offer":
        if not isinstance(d, dict):
            raise ValueError("offer는 dict여야 합니다")
        try:
            oid = int(d["id"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("offer에 id가 없거나 정수가 아닙니다") from None
        try:
            dph = float(d["dph_total"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("offer에 dph_total이 없거나 숫자가 아닙니다") from None
        if not math.isfinite(dph):
            raise ValueError("dph_total이 유한하지 않습니다")
        geo = d.get("geolocation")
        geo = geo if isinstance(geo, str) else ""
        rel = _fnum(d, "reliability2", "reliability")
        return cls(
            id=oid,
            machine_id=int(_fnum(d, "machine_id")),
            gpu_name=str(d.get("gpu_name") or ""),
            num_gpus=int(_fnum(d, "num_gpus")),
            dph_total=dph,
            reliability=rel,
            geolocation=geo,
            country=country_code(geo),
            cpu_cores=_fnum(d, "cpu_cores_effective", "cpu_cores"),
            cpu_ram_gb=_fnum(d, "cpu_ram") / 1024.0,
            disk_gb=_fnum(d, "disk_space"),
            inet_down=_fnum(d, "inet_down"),
            cuda_vers=_fnum(d, "cuda_max_good"),
            verified=bool(d.get("verified") is True or d.get("verification") == "verified"),
            rentable=bool(d.get("rentable", True)),
        )


def _norm_gpu(name) -> str:
    return str(name).lower().replace("_", "").replace(" ", "")


@dataclass(frozen=True)
class OfferQuery:
    gpu_name: str = "RTX 5090"
    num_gpus: int = 1
    max_dph: float = 0.50
    min_reliability: float = 0.98
    min_cpu_cores: float = 12
    min_ram_gb: float = 30
    min_disk_gb: float = 100
    min_inet_down: float = 300
    min_cuda: float = 12.8
    verified_only: bool = True
    exclude_countries: tuple = ("CN",)
    exclude_machines: frozenset = frozenset()

    def accepts(self, o: Offer) -> bool:
        if _norm_gpu(o.gpu_name) != _norm_gpu(self.gpu_name):
            return False
        if o.num_gpus != self.num_gpus:
            return False
        if not (0.0 < o.dph_total <= self.max_dph):
            return False
        if not o.rentable:
            return False
        if self.verified_only and not o.verified:
            return False
        if o.reliability < self.min_reliability:
            return False
        if o.cpu_cores < self.min_cpu_cores or o.cpu_ram_gb < self.min_ram_gb:
            return False
        if o.disk_gb < self.min_disk_gb or o.inet_down < self.min_inet_down or o.cuda_vers < self.min_cuda:
            return False
        excl = {str(c).upper() for c in self.exclude_countries}
        if o.country is not None and o.country in excl:
            return False
        if o.machine_id in self.exclude_machines:
            return False
        return True

    def to_api_query(self) -> dict:
        q = {
            "gpu_name": {"eq": self.gpu_name},
            "num_gpus": {"eq": self.num_gpus},
            "dph_total": {"lte": self.max_dph},
            "rentable": {"eq": True},
            "reliability2": {"gte": self.min_reliability},
            "cpu_cores_effective": {"gte": self.min_cpu_cores},
            "cpu_ram": {"gte": self.min_ram_gb * 1024.0},
            "disk_space": {"gte": self.min_disk_gb},
            "inet_down": {"gte": self.min_inet_down},
            "cuda_max_good": {"gte": self.min_cuda},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
        }
        if self.verified_only:
            q["verified"] = {"eq": True}
        return q


def select_offers(offers, query: OfferQuery) -> list:
    ok = [o for o in offers if query.accepts(o)]
    return sorted(ok, key=lambda o: (o.dph_total, -o.reliability, o.id))


@dataclass(frozen=True)
class Instance:
    id: int
    machine_id: Optional[int]
    label: Optional[str]
    actual_status: Optional[str]
    intended_status: str
    dph_total: float
    num_gpus: int

    @classmethod
    def from_api(cls, d: dict) -> "Instance":
        try:
            iid = int(d["id"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("instance에 id가 없습니다") from None
        mid = d.get("machine_id")
        try:
            mid = None if mid is None else int(mid)
        except (TypeError, ValueError):
            mid = None
        label = d.get("label")
        intended = d.get("intended_status") or d.get("cur_state") or "running"
        actual = d.get("actual_status")
        return cls(
            id=iid, machine_id=mid, label=label if isinstance(label, str) else None,
            actual_status=str(actual).lower() if actual else None,
            intended_status=str(intended).strip().lower(),
            dph_total=_fnum(d, "dph_total"),
            num_gpus=max(1, int(_fnum(d, "num_gpus", default=1.0))),
        )


@dataclass(frozen=True)
class VastPolicy:
    idle_timeout_s: float = 1800
    boot_timeout_s: float = 3000
    idle_action: str = "stop"
    cap_margin_h: float = 0.25
    reserve_hours: float = 1.0
    max_instances: int = 1
    stopped_usd_per_h: float = 0.02
    label_prefix: str = "buc-"

    def __post_init__(self):
        for name in ("idle_timeout_s", "boot_timeout_s", "cap_margin_h", "reserve_hours", "stopped_usd_per_h"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name}: 유한하고 0 이상이어야 합니다 (받은 값 {v!r})")
        if self.idle_action not in ("stop", "destroy"):
            raise ValueError(f"idle_action은 stop|destroy (받은 값 {self.idle_action!r})")
        if isinstance(self.max_instances, bool) or not isinstance(self.max_instances, int) or self.max_instances < 0:
            raise ValueError("max_instances는 0 이상의 정수")
        if not isinstance(self.label_prefix, str) or not self.label_prefix:
            raise ValueError("label_prefix는 비어 있지 않은 문자열")


@dataclass(frozen=True)
class Action:
    kind: str
    instance_id: int
    reason: str


@dataclass
class TickReport:
    ts: float
    actions: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    metered_usd: float = 0.0
    daily_usd: float = 0.0
    total_usd: float = 0.0
    burn_usd_per_h: float = 0.0
    locked: bool = False


# --------------------------------------------------------------------------- 오케스트레이터

class VastOrchestrator:
    def __init__(self, api, ledger: Ledger, config: Config, *, policy: VastPolicy = VastPolicy(),
                 clock: Callable[[], float] = time.time, activity: Optional[Callable] = None, log=None):
        self.api = api
        self.ledger = ledger
        self.config = config
        self.policy = policy
        self._clock = clock
        self._activity_fn = activity
        self._log = log
        self._lock = threading.RLock()
        self._active: dict = {}
        self.bad_machines: set = set()

    # -- 작은 도우미
    def _say(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(_scrub(msg))
            except Exception:
                pass

    def _managed(self, label) -> bool:
        return isinstance(label, str) and label.startswith(self.policy.label_prefix)

    def mark_active(self, iid: int, ts: Optional[float] = None) -> None:
        self._active[int(iid)] = float(self._clock() if ts is None else ts)

    def activity(self, iid: int) -> Optional[float]:
        if self._activity_fn is None:
            return None
        try:
            v = self._activity_fn(iid)
        except Exception:
            return None
        if v is None or isinstance(v, bool):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def is_locked(self) -> bool:
        _, total_cap = self.config.caps("usd")
        if total_cap is None:
            return False
        return self.ledger.total("usd") >= total_cap - _EPS

    def _list_managed(self) -> list:
        out = []
        for d in self.api.list_instances():
            if not isinstance(d, dict) or not self._managed(d.get("label")):
                continue
            try:
                out.append(Instance.from_api(d))
            except ValueError:
                continue
        return out

    def _running_managed_count(self, exclude: Optional[int] = None) -> int:
        return sum(1 for i in self._list_managed() if i.intended_status == "running" and i.id != exclude)

    def _check_reserve(self, dph: float, num_gpus: int, now: float) -> None:
        daily, total = self.config.caps("usd")
        amt = dph * self.policy.reserve_hours
        chk = self.ledger.check_budget("usd", amt, daily_cap=daily, total_cap=total, ts=now)
        if not chk.ok:
            raise BudgetRefused(chk.reason, chk, f"usd 예산 초과 ({chk.reason}): 예약 {amt:.4f}")
        gd, gt = self.config.caps("gpu_h")
        if gd is not None or gt is not None:
            g_amt = num_gpus * self.policy.reserve_hours
            gchk = self.ledger.check_budget("gpu_h", g_amt, daily_cap=gd, total_cap=gt, ts=now)
            if not gchk.ok:
                raise BudgetRefused(gchk.reason, gchk, f"gpu_h 예산 초과 ({gchk.reason}): 예약 {g_amt:.4f}")

    # -- 장부
    @staticmethod
    def _rid(iid: int) -> str:
        return f"vast:{int(iid)}"

    def _events(self, iid: int) -> list:
        return self.ledger.events(run_id=self._rid(iid))

    @staticmethod
    def _t_meter(evs) -> Optional[float]:
        ts = [e.meta["t1"] for e in evs if isinstance(e.meta.get("t1"), (int, float))]
        return max(ts) if ts else None

    @staticmethod
    def _last_meta(evs, key, default=None):
        for e in reversed(evs):
            if key in e.meta:
                return e.meta[key]
        return default

    @staticmethod
    def _t_start(evs) -> Optional[float]:
        for e in reversed(evs):
            if e.meta.get("event") in ("rent", "start", "adopt"):
                return e.meta.get("t1")
        return None

    @staticmethod
    def _is_final(evs) -> bool:
        return any(e.meta.get("final") for e in evs)

    @staticmethod
    def _seen_running(evs, t_start) -> bool:
        lo = (t_start if t_start is not None else -math.inf) - _EPS
        for e in evs:
            m = e.meta
            if m.get("event") is None and m.get("status") == "running" and m.get("t0", -math.inf) >= lo:
                return True
        return False

    def _lifecycle(self, iid: int, event: str, now: float, intended: str, evs_count: int,
                   dph: float, num_gpus: int, machine_id, final: bool = False, **extra) -> None:
        meta = {"event": event, "t1": now, "intended": intended, "dph": dph, "num_gpus": num_gpus,
                "machine_id": machine_id}
        if final:
            meta["final"] = True
        meta.update(extra)
        self.ledger.record(LedgerEvent(
            kind="usd", amount=0.0, run_id=self._rid(iid), stage="vast", meta=meta, ts=now,
            idem_key=f"vast:{iid}:{event}:{now:.3f}:{evs_count}"))

    def _meter(self, iid: int, evs: list, now: float, cur_intended: str, dph: float, num_gpus: int,
               status: Optional[str], machine_id=None) -> float:
        """[t_meter, now)를 UTC 자정에서 나눠 장부에 남긴다. 반환: 이번에 쓴 usd 합."""
        t0 = self._t_meter(evs)
        if t0 is None:
            self._lifecycle(iid, "adopt", now, cur_intended, 0, dph, num_gpus, machine_id)
            return 0.0
        if not now > t0:
            return 0.0
        last_intended = self._last_meta(evs, "intended", cur_intended)
        full = last_intended != "stopped" or cur_intended != "stopped"
        rate = dph if full else self.policy.stopped_usd_per_h
        total = 0.0
        cur = t0
        while cur < now:
            end = min(now, utc_day_bounds(cur)[1])
            if end <= cur:
                break
            dur = end - cur
            meta = {"t0": cur, "t1": end, "intended": cur_intended, "status": status, "rate": rate,
                    "dph": dph, "num_gpus": num_gpus}
            key = f"vast:{iid}:{cur:.3f}:{end:.3f}"
            amount = rate * dur / _HOUR
            self.ledger.record(LedgerEvent(kind="usd", amount=amount, run_id=self._rid(iid), stage="vast",
                                           meta=meta, ts=cur, idem_key=key))
            total += amount
            if full:
                self.ledger.record(LedgerEvent(kind="gpu_h", amount=num_gpus * dur / _HOUR,
                                               run_id=self._rid(iid), stage="vast", meta=meta, ts=cur,
                                               idem_key=key + ":gpu_h"))
            cur = end
        return total

    def _meter_inst(self, inst: Instance, now: float) -> float:
        evs = self._events(inst.id)
        return self._meter(inst.id, evs, now, inst.intended_status, inst.dph_total, inst.num_gpus,
                           inst.actual_status, inst.machine_id)

    def _finish_gone(self, iid: int, now: float) -> float:
        """목록에서 사라진 인스턴스: 직전 요율로 now까지 과금하고 final."""
        evs = self._events(iid)
        dph = float(self._last_meta(evs, "dph", 0.0) or 0.0)
        ng = int(self._last_meta(evs, "num_gpus", 1) or 1)
        last_intended = self._last_meta(evs, "intended", "running")
        m = self._meter(iid, evs, now, last_intended, dph, ng, "gone", self._last_meta(evs, "machine_id"))
        evs = self._events(iid)
        self._lifecycle(iid, "destroy", now, "destroyed", len(evs), dph, ng,
                        self._last_meta(evs, "machine_id"), final=True, reason="gone")
        return m

    def _show(self, iid: int) -> Instance:
        inst = Instance.from_api(self.api.show_instance(int(iid)))
        if not self._managed(inst.label):
            raise ValueError(f"인스턴스 {iid}은 관리 대상이 아닙니다 (label 접두사 {self.policy.label_prefix!r})")
        return inst

    # -- 임대·시작·정지·삭제
    def rent(self, *, query: Optional[OfferQuery] = None, image: str = DEFAULT_IMAGE, disk_gb=90,
             label: Optional[str] = None, onstart: Optional[str] = None, env: Optional[dict] = None) -> int:
        with self._lock:
            label = self.policy.label_prefix + "gpu" if label is None else label
            if not self._managed(label):
                raise ValueError(f"label은 {self.policy.label_prefix!r}로 시작해야 합니다 (받은 값 {label!r})")
            query = query if query is not None else OfferQuery()
            now = self._clock()
            if self.is_locked():
                raise BudgetRefused("locked", None, "누적 예산 소진으로 잠김")
            if self._running_managed_count() >= self.policy.max_instances:
                raise RentLimitError(f"동시 인스턴스 한도 {self.policy.max_instances}")
            raws = self.api.search_offers(query.to_api_query())
            offers = []
            for r in raws:
                try:
                    offers.append(Offer.from_api(r))
                except ValueError:
                    continue
            cands = [o for o in select_offers(offers, query) if o.machine_id not in self.bad_machines]
            if not cands:
                raise NoOfferError("조건에 맞는 오퍼가 없습니다")
            best = cands[0]
            self._check_reserve(best.dph_total, best.num_gpus, now)
            env = dict(env) if env else {}
            iid = int(self.api.create_instance(best.id, {
                "image": image, "disk": disk_gb, "label": label, "onstart": onstart, "env": env}))
            self._lifecycle(iid, "rent", now, "running", len(self._events(iid)), best.dph_total, best.num_gpus,
                            best.machine_id, offer=best.id, env_keys=sorted(env))
            self._say(f"rent iid={iid} offer={best.id} dph={best.dph_total}")
            return iid

    def start(self, iid: int) -> None:
        with self._lock:
            now = self._clock()
            if self.is_locked():
                raise BudgetRefused("locked", None, "누적 예산 소진으로 잠김")
            inst = self._show(iid)
            if inst.intended_status == "running":
                return
            if self._running_managed_count(exclude=inst.id) >= self.policy.max_instances:
                raise RentLimitError(f"동시 인스턴스 한도 {self.policy.max_instances}")
            self._check_reserve(inst.dph_total, inst.num_gpus, now)
            self._meter_inst(inst, now)
            self.api.set_state(inst.id, "running")
            self._lifecycle(inst.id, "start", now, "running", len(self._events(inst.id)), inst.dph_total,
                            inst.num_gpus, inst.machine_id)

    def _stop_core(self, inst: Instance, now: float) -> None:
        self._meter_inst(inst, now)
        self.api.set_state(inst.id, "stopped")
        self._lifecycle(inst.id, "stop", now, "stopped", len(self._events(inst.id)), inst.dph_total,
                        inst.num_gpus, inst.machine_id)

    def stop(self, iid: int) -> None:
        with self._lock:
            now = self._clock()
            inst = self._show(iid)
            if inst.intended_status == "stopped":
                self._meter_inst(inst, now)
                return
            self._stop_core(inst, now)

    def _destroy_core(self, inst: Instance, now: float, reason: str) -> None:
        self._meter_inst(inst, now)
        try:
            self.api.destroy_instance(inst.id)
        except VastNotFound:
            pass
        self._lifecycle(inst.id, "destroy", now, "destroyed", len(self._events(inst.id)), inst.dph_total,
                        inst.num_gpus, inst.machine_id, final=True, reason=reason)

    def destroy(self, iid: int, reason: str = "manual") -> None:
        with self._lock:
            now = self._clock()
            try:
                inst = self._show(iid)
            except VastNotFound:
                if self._events(iid) and not self._is_final(self._events(iid)):
                    self._finish_gone(int(iid), now)
                return
            self._destroy_core(inst, now, reason)

    # -- tick
    def tick(self, now: Optional[float] = None) -> TickReport:
        with self._lock:
            return self._tick(self._clock() if now is None else float(now))

    def _tick(self, now: float) -> TickReport:
        pol = self.policy
        rep = TickReport(ts=now)
        raw = self.api.list_instances()  # 실패는 전파
        listed_ids = set()
        managed: dict = {}
        for d in raw:
            if not isinstance(d, dict):
                continue
            try:
                inst = Instance.from_api(d)
            except ValueError:
                continue
            listed_ids.add(inst.id)
            if self._managed(inst.label):
                managed[inst.id] = inst

        metered = 0.0
        # ② 계량
        finals: set = set()
        for iid, inst in sorted(managed.items()):
            try:
                evs = self._events(iid)
                if self._is_final(evs):
                    finals.add(iid)
                    continue
                metered += self._meter_inst(inst, now)
            except Exception as e:  # 한 인스턴스의 장부 오류가 나머지를 막지 않는다
                rep.errors.append(f"meter {iid}: {_scrub(e)}")
        # 사라진 미종결 iid
        try:
            for rid in sorted(self.ledger.totals_by_run("usd")):
                if not rid.startswith("vast:"):
                    continue
                try:
                    iid = int(rid[5:])
                except ValueError:
                    continue
                if iid in listed_ids:
                    continue
                try:
                    if self._is_final(self._events(iid)):
                        continue
                    metered += self._finish_gone(iid, now)
                except Exception as e:
                    rep.errors.append(f"gone {iid}: {_scrub(e)}")
        except Exception as e:
            rep.errors.append(f"ledger scan: {_scrub(e)}")
        rep.metered_usd = metered

        # ③ 행동 (인스턴스당 최대 1개)
        handled: set = set(finals)
        states: dict = {}
        for iid, inst in managed.items():
            if iid in finals:
                continue
            try:
                evs = self._events(iid)
            except Exception as e:
                rep.errors.append(f"state {iid}: {_scrub(e)}")
                handled.add(iid)
                continue
            t_start = self._t_start(evs)
            seen = self._seen_running(evs, t_start) or inst.actual_status == "running"
            states[iid] = (t_start, seen)

        # boot
        for iid in sorted(states):
            inst = managed[iid]
            t_start, seen = states[iid]
            if inst.intended_status != "running" or seen or t_start is None:
                continue
            if now - t_start >= pol.boot_timeout_s:
                handled.add(iid)
                try:
                    self._destroy_core(inst, now, "boot_timeout")
                    if inst.machine_id is not None:
                        self.bad_machines.add(inst.machine_id)
                    rep.actions.append(Action("destroy", iid, "boot_timeout"))
                except Exception as e:
                    rep.errors.append(f"destroy {iid}: {_scrub(e)}")

        # cap
        running = [managed[i] for i in sorted(states) if managed[i].intended_status == "running"
                   and not (i in handled and any(a.instance_id == i for a in rep.actions))]
        burn = math.fsum(i.dph_total for i in running)
        rep.burn_usd_per_h = burn
        daily_cap, total_cap = self.config.caps("usd")
        spent_d = self.ledger.daily_total("usd", ts=now)
        spent_t = self.ledger.total("usd")
        margin = burn * pol.cap_margin_h
        d_hit = daily_cap is not None and spent_d + margin > daily_cap + _EPS
        t_hit = total_cap is not None and spent_t + margin > total_cap + _EPS
        if d_hit or t_hit:
            reason = "cap_total" if t_hit else "cap_daily"
            for inst in running:
                if inst.id in handled:
                    continue
                handled.add(inst.id)
                try:
                    self._stop_core(inst, now)
                    rep.actions.append(Action("stop", inst.id, reason))
                except Exception as e:
                    rep.errors.append(f"stop {inst.id}: {_scrub(e)}")

        # idle
        for iid in sorted(states):
            inst = managed[iid]
            if iid in handled or inst.intended_status != "running":
                continue
            t_start, seen = states[iid]
            if not seen or t_start is None:
                continue
            cands = [t_start]
            for v in (self._active.get(iid), self.activity(iid)):
                if v is not None and math.isfinite(v):
                    cands.append(v)
            if now - max(cands) >= pol.idle_timeout_s:
                handled.add(iid)
                try:
                    if pol.idle_action == "destroy":
                        self._destroy_core(inst, now, "idle")
                        rep.actions.append(Action("destroy", iid, "idle"))
                    else:
                        self._stop_core(inst, now)
                        rep.actions.append(Action("stop", iid, "idle"))
                except Exception as e:
                    rep.errors.append(f"{pol.idle_action} {iid}: {_scrub(e)}")

        rep.daily_usd = self.ledger.daily_total("usd", ts=now)
        rep.total_usd = self.ledger.total("usd")
        rep.locked = total_cap is not None and rep.total_usd >= total_cap - _EPS
        for a in rep.actions:
            self._say(f"action {a.kind} {a.instance_id} {a.reason}")
        return rep
