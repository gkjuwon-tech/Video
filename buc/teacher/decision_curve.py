"""TCH-05 결정 곡선 X1-slim · 가난선 프로브 v0 (계약서: ops/harness/contracts/TCH-05.md).

옥타브 o = 라플라시안 피라미드(FND-06)의 o번째 밴드(o=0 이 최미세). D(o,σ) 는 SDEdit 출력이 입력과 얼마나 다른지를
옥타브별로 잰 풀링 비율이다. 순수·결정론, import 부작용 없음(diffusers/transformers 를 import 하지 않는다).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from buc.core.pyramid import laplacian_pyramid

__all__ = [
    "VALID_TAU", "DECIDED_TAU", "WIDTH_MAX", "SEP_TARGET", "SEP_TOL", "MONO_TOL", "LEVELS",
    "DecisionCurveError", "BandCurve", "band_curve", "edge_octave", "decision_width", "valid_mask",
    "DecisionCurve", "max_abs_delta", "TeacherCodec", "run_x1_slim", "forward_count",
    "SigmaVerdict", "Interpretation", "interpret",
    "PovertyLine", "poverty_line", "octave_pitch_mm", "initial_o_s_mm",
]

VALID_TAU = 0.1
DECIDED_TAU = 0.5
WIDTH_MAX = 1.5
SEP_TARGET = 1.0
SEP_TOL = 0.5
MONO_TOL = 0.5
LEVELS = 4

_EPS = 1e-18


class DecisionCurveError(ValueError):
    pass


# ----------------------------------------------------------------------------- 작은 검증 도구

def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _pos_int(v, name):
    if not _is_int(v) or v < 1:
        raise DecisionCurveError(f"{name}: 양의 정수여야 한다: {v!r}")
    return int(v)


def _real(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise DecisionCurveError(f"{name}: 실수여야 한다: {v!r}")
    return float(v)


def _pos_finite(v, name):
    x = _real(v, name)
    if not math.isfinite(x) or x <= 0.0:
        raise DecisionCurveError(f"{name}: 양의 유한수여야 한다: {v!r}")
    return x


def _nonneg_finite(v, name):
    x = _real(v, name)
    if not math.isfinite(x) or x < 0.0:
        raise DecisionCurveError(f"{name}: 0 이상의 유한수여야 한다: {v!r}")
    return x


def _levels(v):
    return _pos_int(v, "levels")


# ----------------------------------------------------------------------------- 옥타브별 D / spread

@dataclass(frozen=True, eq=False)
class BandCurve:
    levels: int
    decision: np.ndarray
    spread: np.ndarray
    input_energy: np.ndarray
    n_images: int
    n_seeds: int


def _as_image(a, name):
    try:
        arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise DecisionCurveError(f"{name}: 수치 배열이어야 한다") from e
    if arr.ndim not in (2, 3):
        raise DecisionCurveError(f"{name}: (H,W) 또는 (H,W,C) 여야 한다 (shape {arr.shape})")
    if arr.ndim == 3 and arr.shape[2] < 1:
        raise DecisionCurveError(f"{name}: 채널이 없다")
    if not np.all(np.isfinite(arr)):
        raise DecisionCurveError(f"{name}: 유한하지 않은 값")
    return arr


def band_curve(inputs: Sequence[np.ndarray], outputs: Sequence[Sequence[np.ndarray]], levels: int = LEVELS) -> BandCurve:
    L = _levels(levels)
    try:
        ins = list(inputs)
        outs = [list(o) for o in outputs]
    except TypeError as e:
        raise DecisionCurveError("inputs/outputs 는 시퀀스여야 한다") from e
    n = len(ins)
    if n < 1:
        raise DecisionCurveError("입력이 비어 있다")
    if len(outs) != n:
        raise DecisionCurveError(f"입력 {n} 장과 출력 묶음 {len(outs)} 개가 다르다")
    s = len(outs[0])
    if s < 2:
        raise DecisionCurveError("시드는 2개 이상이어야 한다")
    if any(len(o) != s for o in outs):
        raise DecisionCurveError("이미지마다 시드 수가 같아야 한다")
    xs = [_as_image(x, f"inputs[{i}]") for i, x in enumerate(ins)]
    ys = [[_as_image(y, f"outputs[{i}][{k}]") for k, y in enumerate(o)] for i, o in enumerate(outs)]
    shape = xs[0].shape
    for i, x in enumerate(xs):
        if x.shape != shape:
            raise DecisionCurveError("모든 입력의 모양이 같아야 한다")
        for y in ys[i]:
            if y.shape != shape:
                raise DecisionCurveError("출력 모양이 입력과 같아야 한다")
    if min(shape[0], shape[1]) < 2 ** (L + 1):
        raise DecisionCurveError(f"영상이 너무 작다: min(H,W) >= 2^(levels+1) = {2 ** (L + 1)} 필요, shape {shape}")

    num = np.zeros(L)
    den = np.zeros(L)
    ein = np.zeros(L)
    snum = np.zeros(L)
    sden = np.zeros(L)
    for i, x in enumerate(xs):
        bx = [np.asarray(b, dtype=np.float64) for b in laplacian_pyramid(x, L)[:L]]
        by_all = [[np.asarray(b, dtype=np.float64) for b in laplacian_pyramid(y, L)[:L]] for y in ys[i]]
        for o in range(L):
            ex = float(np.sum(bx[o] ** 2))
            ein[o] += ex
            mean = by_all[0][o].copy()
            for k in range(1, s):
                mean = mean + by_all[k][o]
            mean = mean / s
            for k in range(s):
                by = by_all[k][o]
                num[o] += float(np.sum((bx[o] - by) ** 2))
                den[o] += ex + float(np.sum(by ** 2))
                snum[o] += float(np.sum((by - mean) ** 2))
                sden[o] += float(np.sum(by ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        dec = np.where(den > _EPS, num / np.where(den > _EPS, den, 1.0), np.nan)
        spr = np.where(sden > _EPS, snum / np.where(sden > _EPS, sden, 1.0), np.nan)
    return BandCurve(L, dec, spr, ein, n, s)


# ----------------------------------------------------------------------------- 가장자리 / 폭 / 마스크

def _curve_1d(d, name="d"):
    try:
        a = np.asarray(d, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise DecisionCurveError(f"{name}: 수치 1차원 배열이어야 한다") from e
    if a.ndim != 1 or a.size < 1:
        raise DecisionCurveError(f"{name}: 비어 있지 않은 1차원이어야 한다 (shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise DecisionCurveError(f"{name}: NaN/inf 가 있다")
    return a


def _tau(tau):
    return _pos_finite(tau, "tau")


def edge_octave(d, tau: float) -> float:
    a = _curve_1d(d)
    t = _tau(tau)
    n = a.size
    if a[-1] > t:
        return math.inf
    over = np.nonzero(a > t)[0]
    if over.size == 0:
        return -math.inf
    i = int(over[-1])           # 굵은 쪽에서 처음 넘는 칸; d[i+1] <= tau
    return float(i + (a[i] - t) / (a[i] - a[i + 1]))


def decision_width(d, lo: float = VALID_TAU, hi: float = DECIDED_TAU) -> float:
    lo = _tau(lo)
    hi = _tau(hi)
    if not lo < hi:
        raise DecisionCurveError(f"lo < hi 여야 한다: {lo}, {hi}")
    a = edge_octave(d, lo)
    b = edge_octave(d, hi)
    if math.isinf(a) or math.isinf(b):
        return math.nan
    return max(0.0, a - b)


def valid_mask(d, tau: float = VALID_TAU) -> np.ndarray:
    a = _curve_1d(d)
    t = _tau(tau)
    m = np.zeros(a.size, dtype=bool)
    for o in range(a.size - 1, -1, -1):
        if a[o] <= t:
            m[o] = True
        else:
            break
    return m


# ----------------------------------------------------------------------------- 컨테이너

def _f64_2d(a, name, shape):
    try:
        arr = np.array(a, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise DecisionCurveError(f"{name}: 수치 배열이어야 한다") from e
    if arr.shape != shape:
        raise DecisionCurveError(f"{name}: 모양 {arr.shape} != {shape}")
    return arr


@dataclass(frozen=True, eq=False)
class DecisionCurve:
    indices: tuple
    sigmas: tuple
    decision: np.ndarray
    spread: np.ndarray
    input_energy: np.ndarray
    levels: int
    steps: int
    n_images: int
    n_seeds: int
    table_key: str

    def __post_init__(self):
        try:
            idx = tuple(self.indices)
            sig = tuple(self.sigmas)
        except TypeError as e:
            raise DecisionCurveError("indices/sigmas 는 열이어야 한다") from e
        if len(idx) < 1:
            raise DecisionCurveError("indices 가 비어 있다")
        if not all(_is_int(i) and i >= 0 for i in idx):
            raise DecisionCurveError(f"indices 는 0 이상 정수여야 한다: {idx}")
        idx = tuple(int(i) for i in idx)
        if any(not a < b for a, b in zip(idx, idx[1:])):
            raise DecisionCurveError(f"indices 는 엄격 오름차순이어야 한다: {idx}")
        if len(sig) != len(idx):
            raise DecisionCurveError("sigmas 와 indices 길이가 다르다")
        sig = tuple(_pos_finite(s, "sigma") for s in sig)
        L = _levels(self.levels)
        _pos_int(self.steps, "steps")
        _pos_int(self.n_images, "n_images")
        if not _is_int(self.n_seeds) or self.n_seeds < 2:
            raise DecisionCurveError(f"n_seeds: 2 이상 정수여야 한다: {self.n_seeds!r}")
        if not isinstance(self.table_key, str):
            raise DecisionCurveError("table_key 는 str")
        n = len(idx)
        dec = _f64_2d(self.decision, "decision", (n, L))
        spr = _f64_2d(self.spread, "spread", (n, L))
        ein = _f64_2d(self.input_energy, "input_energy", (L,))
        object.__setattr__(self, "indices", idx)
        object.__setattr__(self, "sigmas", sig)
        object.__setattr__(self, "decision", dec)
        object.__setattr__(self, "spread", spr)
        object.__setattr__(self, "input_energy", ein)
        object.__setattr__(self, "levels", int(L))
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "n_images", int(self.n_images))
        object.__setattr__(self, "n_seeds", int(self.n_seeds))

    def edges(self, tau: float) -> tuple:
        return tuple(edge_octave(row, tau) for row in self.decision)

    def widths(self, lo: float = VALID_TAU, hi: float = DECIDED_TAU) -> tuple:
        return tuple(decision_width(row, lo, hi) for row in self.decision)

    def valid_mask(self, sigma: float, tau: float = VALID_TAU) -> np.ndarray:
        s = _real(sigma, "sigma")
        if not math.isfinite(s):
            raise DecisionCurveError(f"sigma 가 유한하지 않다: {sigma!r}")
        best, bd = -1, math.inf
        for r, p in enumerate(self.sigmas):
            if abs(p - s) < bd:
                best, bd = r, abs(p - s)
        if not bd <= 1e-9:
            raise DecisionCurveError(f"sigma={sigma!r} 는 곡선의 σ 점이 아니다 (스냅하지 않는다)")
        return valid_mask(self.decision[best], tau)

    def to_dict(self) -> dict:
        def cl(a):
            return [[None if not math.isfinite(v) else float(v) for v in row] for row in a]
        return {
            "indices": list(self.indices),
            "sigmas": [float(s) for s in self.sigmas],
            "decision": cl(self.decision),
            "spread": cl(self.spread),
            "input_energy": [None if not math.isfinite(v) else float(v) for v in self.input_energy],
            "levels": self.levels, "steps": self.steps, "n_images": self.n_images, "n_seeds": self.n_seeds,
            "table_key": self.table_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionCurve":
        try:
            def un(a):
                return np.array([[math.nan if v is None else v for v in row] for row in a], dtype=np.float64)
            return cls(
                indices=tuple(d["indices"]), sigmas=tuple(d["sigmas"]),
                decision=un(d["decision"]), spread=un(d["spread"]),
                input_energy=np.array([math.nan if v is None else v for v in d["input_energy"]], dtype=np.float64),
                levels=d["levels"], steps=d["steps"], n_images=d["n_images"], n_seeds=d["n_seeds"],
                table_key=d["table_key"],
            )
        except (KeyError, TypeError) as e:
            raise DecisionCurveError(f"from_dict: 형식 오류: {e!r}") from e


def max_abs_delta(a: DecisionCurve, b: DecisionCurve) -> float:
    if not isinstance(a, DecisionCurve) or not isinstance(b, DecisionCurve):
        raise DecisionCurveError("DecisionCurve 두 개가 필요하다")
    if a.indices != b.indices or a.levels != b.levels:
        raise DecisionCurveError("indices/levels 가 다른 곡선은 비교할 수 없다")
    if np.isnan(a.decision).any() or np.isnan(b.decision).any():
        raise DecisionCurveError("NaN 이 있는 곡선은 비교할 수 없다")
    return float(np.max(np.abs(a.decision - b.decision)))


# ----------------------------------------------------------------------------- 실행기

class TeacherCodec:
    def __init__(self, teacher):
        self.teacher = teacher

    def encode(self, x):
        return self.teacher.encode_image(x)

    def decode(self, z):
        return self.teacher.decode_latent(z)


def forward_count(n_images: int, n_seeds: int, n_indices: int, steps: int) -> int:
    return (_pos_int(n_images, "n_images") * _pos_int(n_seeds, "n_seeds")
            * _pos_int(n_indices, "n_indices") * _pos_int(steps, "steps"))


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def run_x1_slim(sampler, codec, images, cond, *, seeds, levels: int = LEVELS, steps: int = 2,
                indices: Optional[Sequence[int]] = None, level: int = 0, k: int = 0) -> DecisionCurve:
    # --- 검증 (sampler/codec 호출 전)
    L = _levels(levels)
    st = _pos_int(steps, "steps")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise DecisionCurveError("images: [N,3,H,W] 텐서여야 한다")
    if images.shape[1] != 3 or images.shape[0] < 1:
        raise DecisionCurveError(f"images: 3채널 N>=1 이어야 한다 (shape {tuple(images.shape)})")
    if not (images.is_floating_point() and bool(torch.isfinite(images).all())):
        raise DecisionCurveError("images: 유한한 부동소수 텐서여야 한다")
    if min(images.shape[2], images.shape[3]) < 2 ** (L + 1):
        raise DecisionCurveError(f"images 가 너무 작다: min(H,W) >= {2 ** (L + 1)}")
    try:
        seed_list = list(seeds)
    except TypeError as e:
        raise DecisionCurveError("seeds 는 시퀀스여야 한다") from e
    if len(seed_list) < 2:
        raise DecisionCurveError("시드는 2개 이상이어야 한다")
    if not all(_is_int(s) for s in seed_list):
        raise DecisionCurveError(f"seeds 는 int 여야 한다: {seed_list}")
    seed_list = [int(s) for s in seed_list]
    if len(set(seed_list)) != len(seed_list):
        raise DecisionCurveError("seeds 는 서로 달라야 한다")
    table = sampler.table
    nt = len(table)
    if st > nt:
        raise DecisionCurveError(f"steps={st} > 표 길이 {nt}")
    if indices is None:
        idx = tuple(range(nt - st + 1))
    else:
        try:
            idx = tuple(indices)
        except TypeError as e:
            raise DecisionCurveError("indices 는 시퀀스여야 한다") from e
    if len(idx) < 1:
        raise DecisionCurveError("indices 가 비어 있다")
    if not all(_is_int(j) for j in idx):
        raise DecisionCurveError(f"indices 는 정수여야 한다: {idx}")
    idx = tuple(int(j) for j in idx)
    if idx[0] < 0 or any(not a < b for a, b in zip(idx, idx[1:])) or idx[-1] + st > nt:
        raise DecisionCurveError(f"indices 는 0 <= j, 엄격 오름차순, j+steps <= {nt} 여야 한다: {idx}")

    # --- 실행
    lat = codec.encode(images)
    ref = codec.decode(lat)
    if not isinstance(ref, torch.Tensor) or ref.ndim != 4 or ref.shape[1] != 3 or ref.shape[0] != images.shape[0]:
        raise DecisionCurveError("codec.decode 는 [N,3,H,W] 를 내야 한다")
    refs = _to_np(ref)
    n = refs.shape[0]
    inputs = [refs[i] for i in range(n)]
    dec_rows, spr_rows = [], []
    ein = None
    for j in idx:
        sigma = table.points[j]
        ys = []
        for sd in seed_list:
            res = sampler.sdedit(lat, cond, sigma=sigma, seeds=sd, steps=st, level=level, k=k)
            x0 = res.x0_hat
            if not isinstance(x0, torch.Tensor) or tuple(x0.shape) != tuple(lat.shape):
                raise DecisionCurveError(f"x0_hat 모양이 잠재와 다르다: {getattr(x0, 'shape', None)} != {tuple(lat.shape)}")
            ys.append(_to_np(codec.decode(x0)))
        bc = band_curve(inputs, [[ys[s][i] for s in range(len(seed_list))] for i in range(n)], L)
        dec_rows.append(bc.decision)
        spr_rows.append(bc.spread)
        ein = bc.input_energy
    return DecisionCurve(
        indices=idx, sigmas=tuple(float(table.points[j]) for j in idx),
        decision=np.stack(dec_rows), spread=np.stack(spr_rows), input_energy=ein,
        levels=L, steps=st, n_images=n, n_seeds=len(seed_list), table_key=table.key,
    )


# ----------------------------------------------------------------------------- 해석

@dataclass(frozen=True)
class SigmaVerdict:
    index: int
    sigma: float
    edge_valid: float
    edge_decided: float
    width: float
    usable: bool


@dataclass(frozen=True)
class Interpretation:
    verdicts: tuple
    shifts: tuple
    adopted: tuple
    monotone: bool
    e2a_pass: bool
    fallback: str


def interpret(curve: DecisionCurve, *, width_max: float = WIDTH_MAX, sep_target: float = SEP_TARGET,
              sep_tol: float = SEP_TOL, mono_tol: float = MONO_TOL, valid_tau: float = VALID_TAU,
              decided_tau: float = DECIDED_TAU) -> Interpretation:
    if not isinstance(curve, DecisionCurve):
        raise DecisionCurveError("DecisionCurve 가 필요하다")
    vt = _pos_finite(valid_tau, "valid_tau")
    dt = _pos_finite(decided_tau, "decided_tau")
    if not vt < dt:
        raise DecisionCurveError(f"0 < valid_tau < decided_tau 여야 한다: {vt}, {dt}")
    wm = _pos_finite(width_max, "width_max")
    stg = _pos_finite(sep_target, "sep_target")
    stol = _nonneg_finite(sep_tol, "sep_tol")
    mtol = _nonneg_finite(mono_tol, "mono_tol")
    if np.isnan(curve.decision).any():
        raise DecisionCurveError("곡선에 NaN 이 있어 해석할 수 없다")

    verdicts, e_dec = [], []
    for r, j in enumerate(curve.indices):
        row = curve.decision[r]
        ev = edge_octave(row, vt)
        ed = edge_octave(row, dt)
        if math.isinf(ev) or math.isinf(ed):
            w = math.nan
        else:
            w = max(0.0, ev - ed)
        usable = bool(math.isfinite(w) and w <= wm)
        verdicts.append(SigmaVerdict(int(j), float(curve.sigmas[r]), ev, ed, w, usable))
        e_dec.append(ed)
    pos = {j: r for r, j in enumerate(curve.indices)}
    shifts = []
    for r, j in enumerate(curve.indices):
        p = pos.get(j + curve.steps)
        if p is None or math.isinf(e_dec[r]) or math.isinf(e_dec[p]):
            shifts.append(math.nan)
        else:
            shifts.append(float(e_dec[r] - e_dec[p]))
    adopted = tuple(
        v.index for v, s in zip(verdicts, shifts)
        if v.usable and math.isfinite(s) and abs(s - stg) <= stol + 1e-12
    )
    fin = [e for e in e_dec if math.isfinite(e)]
    monotone = bool(all(b - a < mtol for a, b in zip(fin, fin[1:])))
    e2a = len(adopted) >= 2
    if e2a:
        fb = "none"
    elif sum(1 for v in verdicts if math.isfinite(v.width)) < 2:
        fb = "inconclusive"
    else:
        fb = "widen_level_gap_or_sr_lora"
    return Interpretation(tuple(verdicts), tuple(shifts), adopted, monotone, bool(e2a), fb)


# ----------------------------------------------------------------------------- 가난선 프로브

@dataclass(frozen=True)
class PovertyLine:
    n_octaves: int
    passes: tuple
    ratio: tuple

    def boundary_mm(self, footprint_mm: float) -> float:
        return _pos_finite(footprint_mm, "footprint_mm") * 2.0 ** self.n_octaves

    @property
    def teacher_pixel_fraction(self) -> float:
        return 0.25 ** self.n_octaves


def _vec(a, name):
    try:
        v = np.array(a, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise DecisionCurveError(f"{name}: 수치 1차원이어야 한다") from e
    if v.ndim != 1 or v.size < 1:
        raise DecisionCurveError(f"{name}: 비어 있지 않은 1차원이어야 한다")
    if not np.all(np.isfinite(v)):
        raise DecisionCurveError(f"{name}: 유한하지 않은 값")
    return v


def poverty_line(student_dr, teacher_dr, *, afc=None, ratio_max: float = 1.1, afc_max: float = 0.55) -> PovertyLine:
    s = _vec(student_dr, "student_dr")
    t = _vec(teacher_dr, "teacher_dr")
    if s.shape != t.shape:
        raise DecisionCurveError("student_dr 와 teacher_dr 길이가 다르다")
    if np.any(t <= 0.0):
        raise DecisionCurveError("teacher_dr 는 양수여야 한다")
    if np.any(s < 0.0):
        raise DecisionCurveError("student_dr 는 0 이상이어야 한다")
    rm = _pos_finite(ratio_max, "ratio_max")
    am = _nonneg_finite(afc_max, "afc_max")
    ratio = s / t
    ok = ratio <= rm
    if afc is not None:
        a = _vec(afc, "afc")
        if a.shape != s.shape:
            raise DecisionCurveError("afc 길이가 다르다")
        if np.any(a < 0.0) or np.any(a > 1.0):
            raise DecisionCurveError("afc 는 [0,1] 이어야 한다")
        ok = ok & (a <= am)
    n = 0
    for v in ok:
        if not v:
            break
        n += 1
    return PovertyLine(int(n), tuple(bool(v) for v in ok), tuple(float(r) for r in ratio))


def octave_pitch_mm(o: int, footprint_mm: float) -> float:
    if not _is_int(o) or o < 0:
        raise DecisionCurveError(f"o: 0 이상 정수여야 한다: {o!r}")
    return _pos_finite(footprint_mm, "footprint_mm") * 2.0 ** int(o)


def initial_o_s_mm(judge_mm: float, measured_mm: float) -> float:
    return max(_pos_finite(judge_mm, "judge_mm"), _pos_finite(measured_mm, "measured_mm"))
