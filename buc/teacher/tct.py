"""TCH-07 교사 선정 시험: TCT-v3 NVF · ES_v3 결정식 (Z 현직 vs klein 도전자).

계약서: ops/harness/contracts/TCH-07.md. 공개 API는 계약서와 1:1이다.
- models.upg §1.2-1.4(CT-M1), §2(TCT-v3)의 판정 논리를 사전 등록 가능한 순수 함수로 고정한다.
- 사람 평가·GPU 생성·경쟁 API는 이 모듈 밖이다. 측정값을 받기만 한다(``eval/`` import 금지).
- 기본 경로는 Z 유지다. klein은 증거가 게이트를 통과하고 결정식이 이길 때만 바뀐다.
- 모든 비율·문턱 비교는 1e-12 허용 오차의 경계 포함이다.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from buc.core import det
from buc.teacher.ref_slot import KLEIN_TABLE, KleinSpec, choose_l2b_path
from buc.teacher.sample import DEFAULT_TABLE

__all__ = [
    "INCUMBENT", "CHALLENGER", "CRITERIA", "WEIGHTS_L1", "WEIGHTS_L2B", "Z_CLIP", "TIE_PP", "SWAP_PP",
    "NVF_R_ENTRY", "WORLD_MIX", "WINDOW_DAYS", "ENGINE_FREEZE_DAY", "LOCK_DAY", "GATE_DAY", "SEAM_RATIO_MAX",
    "W4A4_FACTOR", "T4H_PER_TFU", "SELECTION_SETS", "FORBIDDEN_SELECTION_SETS",
    "TctError", "FirewallError", "PreRegistrationError", "SelectionError",
    "PreRegistration", "PREREG", "normalize", "engine_score", "world_t4h",
    "GateInputs", "GateReport", "gate_report", "LicenseGate", "klein_license_gate",
    "check_selection_inputs", "home_away_gap", "seam_ok",
    "grade_engine", "L2bEvidence", "grade_l2b", "grade_l2a", "sp_i2v", "required_tp", "nvf_obs", "z_style",
    "self_tau", "Rate", "wilson", "bradley_terry", "bt_prob",
    "Candidate", "Selection", "LockedConfig", "select_teachers",
    "CompRecord", "CompReport", "comp_gate", "can_replace",
    "Arm", "ARMS", "FILL_ARMS", "arm_seeds", "fill_cost_tfu", "tfu_to_t4h",
    "N_INPUTS", "N_POSES_PER_INPUT", "LITE_INPUTS", "LITE_POSES", "LITE_SEEDS", "LITE_ARMS",
]

# --------------------------------------------------------------------------- 상수

INCUMBENT = "z-image-turbo"
CHALLENGER = "flux2-klein-4b"
CRITERIA = ("K1", "K2", "K3", "K4", "K5", "K6", "K7")
WEIGHTS_L1 = dict(zip(CRITERIA, (0.10, 0.10, 0.20, 0.20, 0.15, 0.10, 0.15)))
WEIGHTS_L2B = dict(zip(CRITERIA, (0.30, 0.20, 0.05, 0.15, 0.10, 0.10, 0.10)))
Z_CLIP = 0.5
TIE_PP = 0.02
SWAP_PP = 0.05
NVF_R_ENTRY = 0.30
WORLD_MIX = {"green": 0.5, "yellow": 0.35, "red": 0.15}
WINDOW_DAYS = (45, 75)
ENGINE_FREEZE_DAY = 60
LOCK_DAY = 15
GATE_DAY = 18
SEAM_RATIO_MAX = 1.3
W4A4_FACTOR = 0.846
T4H_PER_TFU = 1.0 / 1385.0
SELECTION_SETS = ("R-real", "R-syn", "O-photo-dev", "O-style-2a", "O-style-2b", "E-escher")
FORBIDDEN_SELECTION_SETS = ("COMP", "O-photo-home", "O-style-home", "S-closed", "HOLDOUT", "USER")

_EPS = 1e-12
_GRADES = ("green", "yellow", "red")


class TctError(ValueError):
    pass


class FirewallError(TctError):
    pass


class PreRegistrationError(TctError):
    pass


class SelectionError(TctError):
    pass


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))


def _hex(*parts) -> str:
    return det.hash_parts(*parts).hex()


# --------------------------------------------------------------------------- 사전 등록

_W1_DEFAULT = tuple(WEIGHTS_L1[k] for k in CRITERIA)
_W2_DEFAULT = tuple(WEIGHTS_L2B[k] for k in CRITERIA)


def _check_weights(name, w):
    try:
        t = tuple(float(x) for x in w)
    except (TypeError, ValueError) as e:
        raise PreRegistrationError(f"{name}는 실수 7개여야 한다: {w!r}") from e
    if len(t) != len(CRITERIA):
        raise PreRegistrationError(f"{name}는 길이 {len(CRITERIA)}여야 한다: {len(t)}")
    if not all(math.isfinite(x) and x >= 0.0 for x in t):
        raise PreRegistrationError(f"{name}는 유한한 비음수여야 한다: {t}")
    if abs(sum(t) - 1.0) > 1e-9:
        raise PreRegistrationError(f"{name}의 합은 1이어야 한다: {sum(t)}")
    return t


@dataclass(frozen=True)
class PreRegistration:
    w_l1: tuple = _W1_DEFAULT
    w_l2b: tuple = _W2_DEFAULT
    tie_pp: float = TIE_PP
    swap_pp: float = SWAP_PP
    nvf_r_entry: float = NVF_R_ENTRY
    mix: tuple = tuple(sorted(WORLD_MIX.items()))

    def validate(self) -> "PreRegistration":
        _check_weights("w_l1", self.w_l1)
        _check_weights("w_l2b", self.w_l2b)
        for n in ("tie_pp", "swap_pp", "nvf_r_entry"):
            if not _finite(getattr(self, n)):
                raise PreRegistrationError(f"{n}는 유한한 실수여야 한다")
        if self.tie_pp < 0 or not self.tie_pp < self.swap_pp:
            raise PreRegistrationError(f"0 <= tie_pp < swap_pp 이어야 한다: {self.tie_pp}, {self.swap_pp}")
        if not (0.0 < self.nvf_r_entry < 1.0):
            raise PreRegistrationError(f"nvf_r_entry는 (0,1)이어야 한다: {self.nvf_r_entry}")
        try:
            mix = dict(self.mix)
        except (TypeError, ValueError) as e:
            raise PreRegistrationError("mix는 (등급, 비율) 쌍이어야 한다") from e
        if set(mix) != set(_GRADES) or not all(_finite(v) and v >= 0 for v in mix.values()) \
                or abs(sum(mix.values()) - 1.0) > 1e-9:
            raise PreRegistrationError(f"mix는 green/yellow/red 비음수 합 1이어야 한다: {mix}")
        return self

    def key(self) -> str:
        doc = {"w_l1": [float(x) for x in self.w_l1], "w_l2b": [float(x) for x in self.w_l2b],
               "tie_pp": float(self.tie_pp), "swap_pp": float(self.swap_pp),
               "nvf_r_entry": float(self.nvf_r_entry),
               "mix": sorted((str(k), float(v)) for k, v in dict(self.mix).items())}
        return _hex("tct.prereg.v1", json.dumps(doc, sort_keys=True, separators=(",", ":")))


PREREG = PreRegistration()


# --------------------------------------------------------------------------- 점수

def _check_metrics(m, name):
    if not isinstance(m, Mapping) or set(m) != set(CRITERIA):
        raise TctError(f"{name}의 키는 정확히 {CRITERIA}여야 한다")
    for k in CRITERIA:
        if not _finite(m[k]):
            raise TctError(f"{name}[{k}]는 유한한 실수여야 한다: {m[k]!r}")


def normalize(m, m_ref) -> dict:
    _check_metrics(m, "m")
    _check_metrics(m_ref, "m_ref")
    out = {}
    for k in CRITERIA:
        r = float(m_ref[k])
        if r == 0.0:
            raise TctError(f"기준 {k}가 0이다")
        z = (float(m[k]) - r) / abs(r)
        out[k] = min(Z_CLIP, max(-Z_CLIP, z))
    return out


def engine_score(m, m_ref, weights) -> float:
    z = normalize(m, m_ref)
    if isinstance(weights, Mapping):
        w = [float(weights[k]) for k in CRITERIA]
    else:
        w = [float(x) for x in weights]
        if len(w) != len(CRITERIA):
            raise TctError("가중치 길이가 7이 아니다")
    return float(sum(wi * z[k] for wi, k in zip(w, CRITERIA)))


def world_t4h(per_grade, mix=None) -> float:
    mix = dict(WORLD_MIX if mix is None else mix)
    if not isinstance(per_grade, Mapping) or set(per_grade) != set(_GRADES):
        raise TctError("per_grade는 green/yellow/red 키가 모두 필요하다")
    tot = 0.0
    for g in _GRADES:
        v = per_grade[g]
        if not _finite(v) or v <= 0:
            raise TctError(f"T4h[{g}]는 양의 유한 실수여야 한다: {v!r}")
        tot += float(mix[g]) * float(v)
    return float(tot)


# --------------------------------------------------------------------------- 게이트

@dataclass(frozen=True)
class GateInputs:
    license_ok: bool
    fp16_nan: int
    fp16_trials: int
    fp16_ref_nan: int
    fp16_ref_trials: int
    mem_peak_gib: float
    host_completed: bool
    nvf_r: float


@dataclass(frozen=True)
class GateReport:
    ok: bool
    failed: tuple


def gate_report(g: GateInputs, nvf_r_entry: float = NVF_R_ENTRY) -> GateReport:
    failed = []
    if g.license_ok is not True:
        failed.append("license")
    if not (g.fp16_nan == 0 and g.fp16_ref_nan == 0 and g.fp16_trials >= 10_000 and g.fp16_ref_trials >= 2_000):
        failed.append("fp16")
    if not (g.host_completed is True and _finite(g.mem_peak_gib) and g.mem_peak_gib <= 15.0 + _EPS):
        failed.append("memory")
    if not (_finite(g.nvf_r) and g.nvf_r >= nvf_r_entry - _EPS):
        failed.append("nvf_r_entry")
    return GateReport(ok=not failed, failed=tuple(failed))


@dataclass(frozen=True)
class LicenseGate:
    ok: bool
    reason: str = ""


def klein_license_gate(spec: KleinSpec) -> LicenseGate:
    try:
        spec.validate()
    except Exception as e:  # noqa: BLE001 - TCH-08의 어떤 검증 예외도 불통과다
        return LicenseGate(False, f"{type(e).__name__}: {e}")
    return LicenseGate(True, "")


# --------------------------------------------------------------------------- 방화벽

def check_selection_inputs(tags) -> None:
    for t in tags:
        if not isinstance(t, str) or t not in SELECTION_SETS:
            raise FirewallError(f"선정에 쓸 수 없는 입력 집합이다: {t!r}")


def home_away_gap(home: float, away: float):
    gap = abs(float(home) - float(away))
    return gap, bool(gap > 0.05 + _EPS)


def seam_ok(seam: float, inner: float) -> bool:
    if not _finite(seam) or not _finite(inner) or inner <= 0:
        raise TctError(f"inner는 양수여야 한다: {inner!r}")
    return bool(float(seam) / float(inner) <= SEAM_RATIO_MAX + _EPS)


# --------------------------------------------------------------------------- 등급

def _rate(x, name):
    if not _finite(x) or not (0.0 <= x <= 1.0):
        raise TctError(f"{name}는 [0,1]의 유한한 비율이어야 한다: {x!r}")
    return float(x)


def grade_engine(nvf_hc: float, nvf_obs_rate: float) -> str:
    hc = _rate(nvf_hc, "nvf_hc")
    ob = _rate(nvf_obs_rate, "nvf_obs")
    if hc >= 0.60 - _EPS and ob >= 0.95 - _EPS:
        return "green"
    if hc >= 0.55 - _EPS:
        return "yellow"
    return "red"


@dataclass(frozen=True)
class L2bEvidence:
    nvf: float
    z_style: float
    delta_photo: float
    tau_photo: float


def grade_l2b(ev: L2bEvidence) -> str:
    if not isinstance(ev, L2bEvidence):
        raise TctError("L2bEvidence가 필요하다")
    nvf = _rate(ev.nvf, "nvf")
    for n in ("z_style", "delta_photo", "tau_photo"):
        if not _finite(getattr(ev, n)):
            raise TctError(f"{n}는 유한해야 한다")
    if nvf >= 0.55 - _EPS and ev.z_style <= 1.5 + _EPS and ev.delta_photo <= ev.tau_photo + _EPS:
        return "green"
    if nvf >= 0.50 - _EPS:
        return "yellow"
    return "red"


def grade_l2a(z_style: float, outline_drift_px: float, palette_emd: float, palette_tau: float) -> str:
    for v in (z_style, outline_drift_px, palette_emd, palette_tau):
        if not _finite(v):
            raise TctError("L2a 지표는 유한해야 한다")
    if z_style <= 1.5 + _EPS and outline_drift_px <= 1.0 + _EPS and palette_emd <= palette_tau + _EPS:
        return "green"
    return "yellow"


def _open_unit(x, name):
    if not _finite(x) or not (0.0 < x < 1.0):
        raise TctError(f"{name}는 (0,1)이어야 한다: {x!r}")
    return float(x)


def sp_i2v(nvf_hc: float, tp: float) -> float:
    h = _open_unit(nvf_hc, "nvf_hc")
    t = _open_unit(tp, "tp")
    s = math.log(h / (1 - h)) + math.log(t / (1 - t))
    return 1.0 / (1.0 + math.exp(-s))


def required_tp(nvf_hc: float) -> float:
    return 1.0 - _open_unit(nvf_hc, "nvf_hc")


def nvf_obs(views) -> float:
    views = list(views)
    if not views:
        raise TctError("views가 비었다")
    ok = 0
    for de, drift in views:
        de, drift = float(de), float(drift)
        if math.isfinite(de) and math.isfinite(drift) and de <= 2.0 + _EPS and drift <= 1.0 + _EPS:
            ok += 1
    return ok / len(views)


def _self_stats(self_d):
    a = np.asarray(list(self_d), dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise TctError("자기 거리는 유한한 값 2개 이상이어야 한다")
    sd = float(np.std(a, ddof=1))
    if not sd > 0.0:
        raise TctError("자기 거리의 표준편차가 0이다")
    return float(np.mean(a)), sd


def z_style(d: float, self_d) -> float:
    mu, sd = _self_stats(self_d)
    if not _finite(d):
        raise TctError("d는 유한해야 한다")
    return (float(d) - mu) / sd


def self_tau(v) -> float:
    return 2.0 * _self_stats(v)[1]


@dataclass(frozen=True)
class Rate:
    k: int
    n: int
    p: float
    lo: float
    hi: float


def wilson(k: int, n: int, z: float = 1.96) -> Rate:
    if not (isinstance(k, (int, np.integer)) and isinstance(n, (int, np.integer))) or n <= 0 or k < 0 or k > n:
        raise TctError(f"0 <= k <= n, n > 0 이어야 한다: {k}, {n}")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, c - h)
    hi = 1.0 if k == n else min(1.0, c + h)
    return Rate(int(k), int(n), p, lo, hi)


# --------------------------------------------------------------------------- Bradley-Terry

def _reach(adj, start):
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def bradley_terry(items, results, prior: float = 0.0) -> dict:
    items = list(items)
    if len(set(items)) != len(items) or len(items) < 2:
        raise TctError("항목은 서로 다른 2개 이상이어야 한다")
    if not _finite(prior) or prior < 0:
        raise TctError("prior는 비음수여야 한다")
    idx = {it: i for i, it in enumerate(items)}
    m = len(items)
    wins = np.zeros((m, m))
    for r in results:
        w, l = r
        if w not in idx or l not in idx:
            raise TctError(f"모르는 항목이다: {r!r}")
        if w == l:
            raise TctError(f"자기 자신과의 비교다: {r!r}")
        wins[idx[w], idx[l]] += 1.0
    compared = (wins + wins.T) > 0
    if prior > 0:
        wins = wins + 0.5 * prior * compared
    adj_dir = [[j for j in range(m) if wins[i, j] > 0] for i in range(m)]
    adj_und = [[j for j in range(m) if compared[i, j]] for i in range(m)]
    if any(not adj_und[i] for i in range(m)):
        raise TctError("비교가 없는 항목이 있다")
    if len(_reach(adj_und, 0)) != m:
        raise TctError("비교 그래프가 연결되지 않았다")
    if prior == 0:
        radj = [[j for j in range(m) if wins[j, i] > 0] for i in range(m)]
        if len(_reach(adj_dir, 0)) != m or len(_reach(radj, 0)) != m:
            raise TctError("모든 항목이 이기고 지는 사슬로 이어져야 한다(식별 불가)")
    n_ij = wins + wins.T
    w_i = wins.sum(axis=1)
    p = np.ones(m)
    for _ in range(200_000):
        denom = (n_ij / (p[:, None] + p[None, :])).sum(axis=1)
        new = w_i / denom
        new = new / np.exp(np.mean(np.log(new)))
        done = float(np.max(np.abs(np.log(new) - np.log(p)))) < 1e-14
        p = new
        if done:
            break
    s = np.log(p)
    s = s - s.mean()
    return {it: float(s[idx[it]]) for it in items}


def bt_prob(scores, a, b) -> float:
    if a not in scores or b not in scores:
        raise TctError("모르는 항목이다")
    return 1.0 / (1.0 + math.exp(-(scores[a] - scores[b])))


# --------------------------------------------------------------------------- 선정

@dataclass(frozen=True)
class Candidate:
    id: str
    metrics: Mapping
    t4h_world: float
    gates: GateInputs
    sources: tuple = ()
    ref_contract: bool = False
    swap_ok: bool = False
    nvf_s: Optional[float] = None
    ref_t4h_world: Optional[float] = None


@dataclass(frozen=True)
class LockedConfig:
    engine: str
    ref: Optional[str]
    outcome: str
    key: str


@dataclass(frozen=True)
class Selection:
    engine: str
    ref: Optional[str]
    outcome: str
    path: str
    es_l1: Mapping
    es_l2b: Mapping
    score: Mapping
    eligible: tuple
    excluded: Mapping
    tie_band: tuple
    prereg_key: str

    def to_dict(self) -> dict:
        return {
            "engine": self.engine, "ref": self.ref, "outcome": self.outcome, "path": self.path,
            "es_l1": {k: self.es_l1[k] for k in sorted(self.es_l1)},
            "es_l2b": {k: self.es_l2b[k] for k in sorted(self.es_l2b)},
            "score": {k: self.score[k] for k in sorted(self.score)},
            "eligible": list(self.eligible),
            "excluded": {k: list(self.excluded[k]) for k in sorted(self.excluded)},
            "tie_band": list(self.tie_band),
            "prereg_key": self.prereg_key,
        }

    def lock(self) -> LockedConfig:
        key = _hex("tct.lock.v1", self.engine, self.ref or "", self.outcome, self.path, self.prereg_key)
        return LockedConfig(engine=self.engine, ref=self.ref, outcome=self.outcome, key=key)


def _validate_candidates(candidates):
    cands = list(candidates)
    if not cands:
        raise TctError("후보가 비었다")
    ids = [c.id for c in cands]
    if len(set(ids)) != len(ids):
        raise TctError("후보 id가 중복됐다")
    for c in cands:
        if not isinstance(c.id, str) or not c.id:
            raise TctError("후보 id는 비지 않은 str이어야 한다")
        if not _finite(c.t4h_world) or c.t4h_world <= 0:
            raise TctError(f"{c.id}: t4h_world는 양의 유한 실수여야 한다: {c.t4h_world!r}")
        check_selection_inputs(tuple(c.sources))
    return sorted(cands, key=lambda c: c.id)


def select_teachers(candidates, ring_r: Mapping, *, prereg: PreRegistration = PREREG,
                    expected_prereg_key: Optional[str] = None, adapter_ok: bool = True) -> Selection:
    prereg.validate()
    pkey = prereg.key()
    if expected_prereg_key is not None and expected_prereg_key != pkey:
        raise PreRegistrationError("사전 등록 키가 기대값과 다르다")
    cands = _validate_candidates(candidates)
    by_id = {c.id: c for c in cands}
    if INCUMBENT not in by_id:
        raise SelectionError("현직 Z가 후보에 없다")
    z = by_id[INCUMBENT]
    es1 = {c.id: engine_score(c.metrics, z.metrics, prereg.w_l1) for c in cands}
    es2 = {c.id: engine_score(c.metrics, z.metrics, prereg.w_l2b) for c in cands}

    excluded, eligible = {}, []
    for c in cands:
        rep = gate_report(c.gates, prereg.nvf_r_entry)
        if rep.ok:
            eligible.append(c.id)
        else:
            excluded[c.id] = rep.failed
    if not eligible:
        raise SelectionError("게이트를 통과한 후보가 없다")
    t4h = {i: float(by_id[i].t4h_world) for i in eligible}
    score = {i: es1[i] / t4h[i] for i in eligible}

    def order_key(i):                       # 동률 처리: ES 큰 쪽 -> Z -> id
        return (-es1[i], 0 if i == INCUMBENT else 1, i)

    positive = [i for i in eligible if es1[i] > 0.0]
    if positive:
        tstar = min(positive, key=lambda i: (-score[i], order_key(i)))
    else:
        tstar = min(eligible, key=order_key)
    band = [i for i in eligible if abs(es1[i] - es1[tstar]) <= prereg.tie_pp + _EPS]
    engine = min(band, key=lambda i: (t4h[i], order_key(i)))

    ev = ring_r.get(engine) if isinstance(ring_r, Mapping) else None
    if not isinstance(ev, L2bEvidence):
        raise TctError(f"엔진 {engine}의 Ring-R 증거(L2bEvidence)가 필요하다")
    ref, outcome = None, None
    if grade_l2b(ev) == "green":
        outcome, path = "O1", choose_l2b_path("L2b", ring_r_green=True, klein_ok=False, adapter_ok=adapter_ok)
    else:
        rset = []
        for i in eligible:
            c = by_id[i]
            if i == engine or not (c.ref_contract and c.swap_ok) or c.nvf_s is None:
                continue
            if _finite(c.nvf_s) and c.nvf_s - ev.nvf >= prereg.swap_pp - _EPS:
                rset.append(i)
        if not rset:
            outcome = "O4"
            path = choose_l2b_path("L2b", ring_r_green=False, klein_ok=False, adapter_ok=adapter_ok)
        else:
            gain = {}
            for i in rset:
                cost = by_id[i].ref_t4h_world
                if not _finite(cost) or cost <= 0:
                    raise TctError(f"{i}: ref_t4h_world는 양의 유한 실수여야 한다: {cost!r}")
                gain[i] = (by_id[i].nvf_s - ev.nvf) / float(cost)
            tref = min(rset, key=lambda i: (-gain[i], i))
            path = choose_l2b_path("L2b", ring_r_green=False, klein_ok=True, adapter_ok=adapter_ok)
            if abs(es1[tref] - es1[engine]) <= prereg.tie_pp + _EPS and t4h[tref] < t4h[engine]:
                outcome, engine = "O3", tref
            else:
                outcome, ref = "O2", tref
    return Selection(engine=engine, ref=ref, outcome=outcome, path=path, es_l1=es1, es_l2b=es2, score=score,
                     eligible=tuple(sorted(eligible)), excluded=excluded, tie_band=tuple(sorted(band)),
                     prereg_key=pkey)


# --------------------------------------------------------------------------- COMP 게이트 (잠금 뒤에만)

@dataclass(frozen=True)
class CompRecord:
    input_id: str
    grade: str
    registered: bool
    win: bool

    def validate(self) -> "CompRecord":
        if self.grade not in _GRADES:
            raise TctError(f"등급은 {_GRADES} 중 하나여야 한다: {self.grade!r}")
        if not isinstance(self.registered, (bool, np.bool_)) or not isinstance(self.win, (bool, np.bool_)):
            raise TctError("registered, win은 bool이어야 한다")
        return self


@dataclass(frozen=True)
class CompReport:
    engine: str
    n: int
    n_registered: int
    nvf_hc_primary: Optional[float]
    nvf_hc_fallback: float
    nvf_hc: float
    nvf_obs_rate: float
    reg_fail_rate: float
    reg_fail_flag: bool
    grade: str
    needs_human_q3: bool
    by_grade: Mapping = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine, "n": self.n, "n_registered": self.n_registered,
            "nvf_hc_primary": self.nvf_hc_primary, "nvf_hc_fallback": self.nvf_hc_fallback,
            "nvf_hc": self.nvf_hc, "nvf_obs_rate": self.nvf_obs_rate, "reg_fail_rate": self.reg_fail_rate,
            "reg_fail_flag": self.reg_fail_flag, "grade": self.grade, "needs_human_q3": self.needs_human_q3,
            "by_grade": {k: self.by_grade[k] for k in sorted(self.by_grade)},
        }


def comp_gate(locked, records, *, nvf_obs_rate: float) -> CompReport:
    if not isinstance(locked, LockedConfig):
        raise TctError("COMP 게이트는 LockedConfig가 있어야 돈다(잠금 뒤에만)")
    recs = list(records)
    if not recs:
        raise TctError("기록이 비었다")
    for r in recs:
        if not isinstance(r, CompRecord):
            raise TctError("CompRecord가 필요하다")
        r.validate()
    _rate(nvf_obs_rate, "nvf_obs_rate")
    n = len(recs)
    reg = [r for r in recs if r.registered]
    primary = (sum(1 for r in reg if r.win) / len(reg)) if reg else None
    fallback = sum(1 for r in recs if r.win) / n
    hc = fallback if primary is None else min(primary, fallback)
    fail_rate = (n - len(reg)) / n
    grade = grade_engine(hc, nvf_obs_rate)
    by_grade = {}
    for g in _GRADES:
        sel = [r for r in recs if r.grade == g]
        if sel:
            by_grade[g] = sum(1 for r in sel if r.win) / len(sel)
    return CompReport(engine=locked.engine, n=n, n_registered=len(reg), nvf_hc_primary=primary,
                      nvf_hc_fallback=fallback, nvf_hc=hc, nvf_obs_rate=float(nvf_obs_rate),
                      reg_fail_rate=fail_rate, reg_fail_flag=bool(fail_rate > 0.20 + _EPS), grade=grade,
                      needs_human_q3=(grade == "red"), by_grade=by_grade)


# --------------------------------------------------------------------------- 교체 창

def can_replace(day, kind, delta) -> bool:
    if not isinstance(day, (int, np.integer)) or isinstance(day, (bool, np.bool_)):
        raise TctError(f"day는 정수여야 한다: {day!r}")
    if kind not in ("engine", "ref"):
        raise TctError(f"kind는 engine/ref여야 한다: {kind!r}")
    if not _finite(delta):
        raise TctError(f"delta는 유한해야 한다: {delta!r}")
    days = (WINDOW_DAYS[0],) if kind == "engine" else WINDOW_DAYS
    return bool(int(day) in days and float(delta) >= SWAP_PP - _EPS)


# --------------------------------------------------------------------------- 시험 계획·비용

@dataclass(frozen=True)
class Arm:
    engine: str
    seeds: int
    nfe: int
    tfu: float


_NFE_Z = len(DEFAULT_TABLE.points)
_NFE_K = len(KLEIN_TABLE.points)

ARMS = {
    "Z-plain": Arm(INCUMBENT, 1, _NFE_Z, 8.0),
    "Z-Ring": Arm(INCUMBENT, 2, _NFE_Z, 9.1),
    "klein-plain": Arm(CHALLENGER, 1, _NFE_K, 2.6),
    "klein-ref": Arm(CHALLENGER, 2, _NFE_K, 4.0),
    "Z-Ring+theta": Arm(INCUMBENT, 1, _NFE_Z, 9.1),
    "Z-RingR": Arm(INCUMBENT, 1, _NFE_Z + 2, 11.4),
}
FILL_ARMS = ("Z-plain", "Z-Ring", "klein-plain", "klein-ref", "Z-Ring+theta")
N_INPUTS = 330
N_POSES_PER_INPUT = 3
LITE_INPUTS = 160
LITE_POSES = 3
LITE_SEEDS = 1
LITE_ARMS = ("Z-Ring", "klein-ref")


def arm_seeds(arm: str) -> int:
    if arm not in ARMS:
        raise TctError(f"모르는 팔이다: {arm!r}")
    return ARMS[arm].seeds


def fill_cost_tfu(n_poses: int, *, w4a4: bool = True) -> float:
    if not isinstance(n_poses, (int, np.integer)) or isinstance(n_poses, bool) or n_poses < 0:
        raise TctError(f"n_poses는 비음수 정수여야 한다: {n_poses!r}")
    per = 0.0
    for name in FILL_ARMS:
        a = ARMS[name]
        c = a.seeds * a.tfu
        if w4a4 and a.engine == INCUMBENT:      # W4A4 절감은 Z 팔에만 적용된다
            c *= W4A4_FACTOR
        per += c
    return float(n_poses) * per


def tfu_to_t4h(tfu: float) -> float:
    if not _finite(tfu) or tfu < 0:
        raise TctError(f"tfu는 비음수여야 한다: {tfu!r}")
    return float(tfu) * T4H_PER_TFU
