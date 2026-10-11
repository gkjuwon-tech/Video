"""LNS-08: θ 사슬 합성기 + ExplainAway 역사슬.

순방향 ``look`` 은 화면 고정 스택(WB -> EV -> 비네팅 -> 할레이션 -> 센서 -> 65^3 LUT -> 그레인)을 θ 하나로 합성한다
(WB 만 이 모듈 소유, 나머지는 LNS-04 ``sensor_film`` 의 공개 조각). 역방향 ``explain`` 은 노이즈 없는 ``look`` 의 수치
역사슬이며 ``explain(look(x)) ~ x`` 가 소유 화소에서 ``error_bound`` 이내로 성립한다. 역변환이 닫히지 않는 화소(센서 바닥·포화,
룩의 클립/랭크 결손, 포화 광원의 헤일로가 번지는 범위)는 ``status`` 비트로 가려 소유하지 않는다.

학습 파라미터 0, 순수 수식. torch 를 import 하지 않는다. 전역 RNG·시각 없음. 입력 배열은 바꾸지 않고 새 배열을 돌려준다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as _ndi
from scipy.spatial import cKDTree as _KDTree

from buc.core import color as _color
from buc.core import contracts as _C
from buc.lens import sensor_film as _sf
from buc.lens.theta import LUT_NAMES as _LUT_NAMES

__all__ = [
    "CHAIN_ORDER", "WB_REF_KELVIN", "WB_MIRED_STOPS", "WB_TINT_STOPS", "OWN_OK", "OWN_SENSOR_BLACK", "OWN_SENSOR_SAT",
    "OWN_LUT_LOST", "OWN_HALO_TAINT", "SAT_MARGIN_LSB", "LUT_INV_TOL", "HALO_TAINT_EPS", "BLACK_CODE_EPS",
    "HALO_SOURCE_MAX", "wb_gains", "apply_wb", "remove_wb", "look", "look_frames", "grain_sigma", "invert_look",
    "Explained", "explain",
]

CHAIN_ORDER = ("wb", "ev", "vignette", "halation", "sensor", "lut", "grain")
WB_REF_KELVIN = 6500.0
WB_MIRED_STOPS = 0.004
WB_TINT_STOPS = 1.0
OWN_OK = 0
OWN_SENSOR_BLACK = 1
OWN_SENSOR_SAT = 2
OWN_LUT_LOST = 4
OWN_HALO_TAINT = 8
SAT_MARGIN_LSB = 1.5
LUT_INV_TOL = 2e-5
HALO_TAINT_EPS = 5e-7
BLACK_CODE_EPS = 2e-3
HALO_SOURCE_MAX = 16.0

# ---- 내부 상수(공개하지 않음)
_CLIP_EPS = 1e-6          # LUT 출력이 0/1 에서 이만큼 안이면 클립으로 본다
_NEWTON_TOL = 1e-11       # 뉴턴 목표 잔차(f64 평가)
_CONV_TOL = 2e-7          # 이 잔차 이하면 y 에 도달한 것으로 본다(y 의 f32 반올림 여유)
_Y_ERR = 6e-8             # 입력 표시값 y 의 f32 반올림 오차 추정
_E_ERR = 6e-8             # 코드값 e 의 f32 반올림 오차 추정
_COND_FRAC = 0.4          # 예측 센서 오차가 LUT_INV_TOL 의 이 비율을 넘으면 소유하지 않는다
_FOLD_DET_MIN = 0.05     # 셀 모서리 정규화 행렬식 하한
_FOLD_KAPPA = 0.99       # 셀 단사 인증 문턱
_START_CANDIDATES = 8   # 뉴턴 시작점 후보 수(가까운 거친 노드 순)
_MAX_NEWTON = 40
_MAX_HALO_ITER = 80
_E_SAT = (math.log2(1.0 / _sf.LOG_MIDGRAY) - _sf.LOG_STOPS[0]) / (_sf.LOG_STOPS[1] - _sf.LOG_STOPS[0])
_E_MAX = _E_SAT + 2e-6      # 센서 출력은 1.0 을 넘지 못하므로 코드값 상한(그 위는 LUT 고원이라 뉴턴이 움직이지 못한다)
_LN2_SPAN = math.log(2.0) * (_sf.LOG_STOPS[1] - _sf.LOG_STOPS[0])   # d ln s / d e


# ---------------------------------------------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------------------------------------------
def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _float_array(x, name) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN/inf 가 있다")
    return a


def _img3(x, name) -> np.ndarray:
    a = _float_array(x, name)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape) < 1:
        raise ValueError(f"{name}: (H,W,3) 이어야 한다 (shape {a.shape})")
    return a


def _display_array(x, name, *, min_ndim) -> np.ndarray:
    a = _float_array(x, name)
    if a.ndim < min_ndim or a.shape[-1] != 3:
        raise ValueError(f"{name}: 마지막 축이 3 인 배열이어야 한다 (shape {a.shape})")
    if a.size and (float(a.min()) < 0.0 or float(a.max()) > 1.0):
        raise ValueError(f"{name}: [0,1] 표시값이어야 한다")
    return a


def _check_theta(theta):
    if not isinstance(theta, _C.Theta):
        raise TypeError("theta: Theta 여야 한다")


def _check_lut(name) -> str:
    if not isinstance(name, str):
        raise TypeError(f"lut: 이름(str) 이어야 한다 ({type(name).__name__})")
    if name not in _LUT_NAMES:
        raise ValueError(f"lut {name!r}: {tuple(_LUT_NAMES)} 중 하나여야 한다")
    return name


def _check_spec(spec):
    if spec is not None and not isinstance(spec, _sf.SensorSpec):
        raise TypeError("spec: SensorSpec 또는 None 이어야 한다")


# ---------------------------------------------------------------------------------------------------------------
# 화이트밸런스
# ---------------------------------------------------------------------------------------------------------------
def wb_gains(wb_kelvin, wb_tint) -> np.ndarray:
    """채널 이득 (3,) float64. m=1e6/K-1e6/6500, raw=2**[-0.004 m, -1.0 tint, +0.004 m], 이득=raw/휘도(raw)."""
    for n, v in (("wb_kelvin", wb_kelvin), ("wb_tint", wb_tint)):
        if not _is_real(v):
            raise TypeError(f"{n}: 실수여야 한다 ({type(v).__name__})")
        if not math.isfinite(float(v)):
            raise ValueError(f"{n}={v!r}: 유한값이어야 한다")
    k = float(wb_kelvin)
    if k <= 0.0:
        raise ValueError(f"wb_kelvin={k}: 양수여야 한다")
    m = 1e6 / k - 1e6 / WB_REF_KELVIN
    raw = 2.0 ** np.array([-WB_MIRED_STOPS * m, -WB_TINT_STOPS * float(wb_tint), WB_MIRED_STOPS * m], np.float64)
    return raw / float(_color.luminance(raw))


def _gains32(theta) -> np.ndarray:
    return wb_gains(theta.wb_kelvin, theta.wb_tint).astype(np.float32)


def apply_wb(rgb_lin, theta) -> np.ndarray:
    """WB 이득을 float32 로 내려 곱한다. HDR 비클램프. 기본 θ 면 비트 동일(새 배열)."""
    a = _img3(rgb_lin, "rgb_lin")
    _check_theta(theta)
    return (a.astype(np.float32) * _gains32(theta)).astype(np.float32, copy=False)


def remove_wb(rgb_lin, theta) -> np.ndarray:
    """``apply_wb`` 의 역: 같은 float32 이득으로 나눈다."""
    a = _img3(rgb_lin, "rgb_lin")
    _check_theta(theta)
    return (a.astype(np.float32) / _gains32(theta)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------------------------
# 순방향
# ---------------------------------------------------------------------------------------------------------------
def look(rgb_lin, theta, *, seed=0, frame=0, spec=None, noise=True) -> np.ndarray:
    """WB 후 LNS-04 체인: ``sf.apply_sensor_film(apply_wb(x, θ), θ, ...)`` 와 바이트 동일. (H,W,3) float32 표시값."""
    a = _img3(rgb_lin, "rgb_lin")
    _check_theta(theta)
    x = apply_wb(a, theta)
    return _sf.apply_sensor_film(x, theta, seed=seed, frame=frame, spec=spec, noise=noise)


def look_frames(frames, theta, *, seed=0, spec=None, noise=True):
    """Frames(표시값) -> sRGB 복호 -> i번째 프레임을 ``frame=i`` 로 ``look`` -> 새 Frames(fps·해시 유지, aov 복사)."""
    if not isinstance(frames, _C.Frames):
        raise TypeError("frames: Frames 여야 한다")
    _check_theta(theta)
    rgb = frames.rgb
    out = np.empty(rgb.shape, np.float32)
    for i in range(rgb.shape[0]):
        lin = _color.srgb_to_linear(rgb[i])
        out[i] = look(lin, theta, seed=seed, frame=i, spec=spec, noise=noise)
    return _C.Frames(rgb=out, fps=frames.fps, aov={k: np.array(v, copy=True) for k, v in frames.aov.items()},
                     shot_hash=frames.shot_hash, world_hash=frames.world_hash)


def grain_sigma(display, theta) -> np.ndarray:
    """표시 영역 화소 표준편차 ``strength*GRAIN_SIGMA_MAX*2*sqrt(u(1-u))`` (float32, 입력과 같은 모양)."""
    u = _display_array(display, "display", min_ndim=0)
    _check_theta(theta)
    s = float(theta.grain)
    v = u.astype(np.float64)
    return (s * _sf.GRAIN_SIGMA_MAX * 2.0 * np.sqrt(np.maximum(v * (1.0 - v), 0.0))).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# 룩 역변환 (로그 부호화 + 65^3 삼선형 LUT 의 역)
# ---------------------------------------------------------------------------------------------------------------
class _LutState:
    __slots__ = ("flat", "size", "offs", "tree", "coarse_e", "gray", "fold_ok")


_LUT_STATES: dict = {}


def _fold_free_cells(table):
    """셀별 '접힘 없음' 판정(셀 번호 상한은 센서 도달 범위). 한 셀과 그 이웃 26셀 모두가 건전해야 True.

    삼선형 LUT 는 클립된 노드가 끼면 접혀(같은 출력의 원상이 둘) 이웃 셀과 겹칠 수 있다. 접힘은 야코비안 행렬식의 부호가
    바뀌는 곳에서 생기므로, 셀의 8 모서리에서 모서리 변 차분으로 만든 행렬식(열 노름으로 정규화)이 양수로 충분히 큰지 본다.
    """
    S = table.shape[0]
    n = min(int(math.floor(_E_MAX * (S - 1))) + 1, S - 1)             # 센서가 닿을 수 있는 셀 수(축당)
    dr = table[1:] - table[:-1]                                       # (S-1, S, S, 3)
    dg = table[:, 1:] - table[:, :-1]                                 # (S, S-1, S, 3)
    db = table[:, :, 1:] - table[:, :, :-1]                           # (S, S, S-1, 3)
    ok = np.ones((n, n, n), bool)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                cr = dr[:n, b:b + n, c:c + n]
                cg = dg[a:a + n, :n, c:c + n]
                cb = db[a:a + n, b:b + n, :n]
                det = np.einsum("...i,...i->...", cr, np.cross(cg, cb))
                nrm = np.linalg.norm(cr, axis=-1) * np.linalg.norm(cg, axis=-1) * np.linalg.norm(cb, axis=-1)
                ok &= det > _FOLD_DET_MIN * nrm
    # 이웃 26셀까지 건전해야 한다(상자 밖은 영향 없음으로 본다)
    return _ndi.minimum_filter(ok, size=3, mode="constant", cval=True)


def _lut_state(name) -> _LutState:
    st = _LUT_STATES.get(name)
    if st is not None:
        return st
    table = np.asarray(_sf.make_lut(name), np.float64)
    size = table.shape[0]
    st = _LutState()
    st.size = size
    st.flat = np.ascontiguousarray(table.reshape(-1, 3))
    st.offs = np.array([a * size * size + b * size + c for a in (0, 1) for b in (0, 1) for c in (0, 1)], np.intp)
    idx = np.arange(0, size, 2)
    coarse = table[np.ix_(idx, idx, idx)].reshape(-1, 3)
    g = idx / (size - 1.0)
    er, eg, eb = np.meshgrid(g, g, g, indexing="ij")
    coarse_e = np.stack([er, eg, eb], axis=-1).reshape(-1, 3)
    # 시작점 후보에서 클립(0/1) 노드는 뺀다: 평탄한 고원에서는 뉴턴이 움직이지 못한다.
    keep = ~((coarse <= _CLIP_EPS) | (coarse >= 1.0 - _CLIP_EPS)).any(axis=1) & (coarse_e <= _E_MAX + 1.0 / (size - 1)).all(axis=1)
    if keep.sum() < 8:
        keep[:] = True
    st.coarse_e = coarse_e[keep]
    st.tree = _KDTree(coarse[keep])
    st.gray = bool(np.array_equal(table[..., 0], table[..., 1]) and np.array_equal(table[..., 0], table[..., 2]))
    st.fold_ok = None if st.gray else _fold_free_cells(table)
    _LUT_STATES[name] = st
    return st


def _weights(f):
    return np.stack([1.0 - f, f], axis=1)


def _eval(st, e, want_j=True):
    """코드 좌표 e (N,3) in [0,1] 에서 삼선형 LUT 값 (N,3) 과 야코비안 J[n,out,in] (N,3,3)."""
    S = st.size
    p = e * (S - 1)
    i0 = np.minimum(np.floor(p).astype(np.intp), S - 2)
    f = p - i0
    base = (i0[:, 0] * S + i0[:, 1]) * S + i0[:, 2]
    c = st.flat[base[:, None] + st.offs[None, :]]                     # (N,8,3), 코너 k = 4a+2b+c
    wr, wg, wb = _weights(f[:, 0]), _weights(f[:, 1]), _weights(f[:, 2])
    n = e.shape[0]

    def comb(x, y, z):
        return (x[:, :, None, None] * y[:, None, :, None] * z[:, None, None, :]).reshape(n, 8)

    val = np.einsum("nk,nkc->nc", comb(wr, wg, wb), c)
    if not want_j:
        return val
    d = np.array([-1.0, 1.0])[None, :]
    dr, dg, db = np.broadcast_to(d, wr.shape), np.broadcast_to(d, wg.shape), np.broadcast_to(d, wb.shape)
    jr = np.einsum("nk,nkc->nc", comb(dr, wg, wb), c)
    jg = np.einsum("nk,nkc->nc", comb(wr, dg, wb), c)
    jb = np.einsum("nk,nkc->nc", comb(wr, wg, db), c)
    return val, (S - 1.0) * np.stack([jr, jg, jb], axis=2)


def _fold_kappa(st, e, inv):
    """셀 안 야코비안 변동으로 본 접힘 지표. kappa = |J*^-1|_inf * max_행 sum_j |dJ|; kappa < 1 이면 그 셀에서 단사(해가 유일).

    삼선형 보간의 dF/df_j 는 나머지 두 좌표에 쌍선형이라, 셀 안 변동은 혼합 2계 차분과 3계 차분으로 위에서 막힌다.
    클립된 노드가 낀 셀은 이 값이 커져 접힘(거짓 소유의 근원)을 가려낸다.
    """
    S = st.size
    p = e * (S - 1)
    i0 = np.minimum(np.floor(p).astype(np.intp), S - 2)
    base = (i0[:, 0] * S + i0[:, 1]) * S + i0[:, 2]
    c = st.flat[base[:, None] + st.offs[None, :]].reshape(-1, 2, 2, 2, 3)
    d01 = c[:, 1, 1] - c[:, 1, 0] - c[:, 0, 1] + c[:, 0, 0]            # (N, b, ch)
    d02 = c[:, 1, :, 1] - c[:, 1, :, 0] - c[:, 0, :, 1] + c[:, 0, :, 0]  # (N, g, ch)
    d12 = c[:, :, 1, 1] - c[:, :, 1, 0] - c[:, :, 0, 1] + c[:, :, 0, 0]  # (N, r, ch)
    t3 = np.abs(d01[:, 1] - d01[:, 0])
    m01, m02, m12 = (np.abs(x).max(axis=1) for x in (d01, d02, d12))
    dj = (S - 1.0) * np.stack([m01 + m02 + t3, m01 + m12 + t3, m02 + m12 + t3], axis=2)   # (N, ch, j)
    return np.abs(inv).sum(axis=2).max(axis=1) * dj.sum(axis=2).max(axis=1)


def _inv3(J):
    """배치 3x3 역행렬과 '특이에 가까운지' 마스크. 특이 화소의 역은 0 으로 채운다."""
    a, b, c = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
    d, e, f = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
    g, h, i = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
    c00, c01, c02 = e * i - f * h, c * h - b * i, b * f - c * e
    c10, c11, c12 = f * g - d * i, a * i - c * g, c * d - a * f
    c20, c21, c22 = d * h - e * g, b * g - a * h, a * e - b * d
    det = a * c00 + b * c10 + c * c20
    had = np.sqrt((a * a + b * b + c * c) * (d * d + e * e + f * f) * (g * g + h * h + i * i))
    sing = ~(np.abs(det) > 1e-9 * had)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(sing, 0.0, 1.0 / np.where(sing, 1.0, det))
    inv = np.stack([np.stack([c00, c01, c02], 1), np.stack([c10, c11, c12], 1), np.stack([c20, c21, c22], 1)], 1)
    return inv * k[:, None, None], sing


def _newton_step(J, r):
    inv, sing = _inv3(J)
    step = -np.einsum("nij,nj->ni", inv, r)
    if sing.any():                                                    # 특이: 감쇠 최소제곱(랭크 결손·클립 평탄부)
        Js, rs = J[sing], r[sing]
        A = np.einsum("nki,nkj->nij", Js, Js) + 1e-6 * np.eye(3)[None]
        step[sing] = -np.linalg.solve(A, np.einsum("nki,nk->ni", Js, rs)[..., None])[..., 0]
    return step


def _newton(st, y, e):
    """시작점 e 에서 LUT(e)=y 를 푼다(뉴턴 + 백트래킹, 상자 [0,1]^3 투영). (e, 잔차, 야코비안)."""
    e = e.copy()
    val, J = _eval(st, e)
    r = val - y
    rn = np.abs(r).max(axis=1)
    active = rn > _NEWTON_TOL
    for _ in range(_MAX_NEWTON):
        ia = np.flatnonzero(active)
        if ia.size == 0:
            break
        step = _newton_step(J[ia], r[ia])
        e0, y0, rn0 = e[ia], y[ia], rn[ia]
        m = ia.size
        alpha = np.ones(m)
        new_e, new_v, new_J, new_rn = e0.copy(), val[ia].copy(), J[ia].copy(), rn0.copy()
        pending = np.ones(m, bool)
        for _bt in range(10):
            idx = np.flatnonzero(pending)
            if idx.size == 0:
                break
            en = np.clip(e0[idx] + alpha[idx, None] * step[idx], 0.0, _E_MAX)
            vn, Jn = _eval(st, en)
            rr = np.abs(vn - y0[idx]).max(axis=1)
            good = (rr < rn0[idx] * (1.0 - 1e-6)) | (rr <= _NEWTON_TOL)
            gi = idx[good]
            new_e[gi], new_v[gi], new_J[gi], new_rn[gi] = en[good], vn[good], Jn[good], rr[good]
            pending[gi] = False
            alpha[idx[~good]] *= 0.5
        e[ia], val[ia], J[ia], rn[ia] = new_e, new_v, new_J, new_rn
        r[ia] = new_v - y0
        active[ia] = (new_rn > _NEWTON_TOL) & ~pending
    return e, rn, J


def _solve_codes(st, y):
    """삼선형 LUT 의 역: LUT(e)=y 를 만족하는 e (N,3) in [0,1], 잔차, 그리고 최종 야코비안.

    시작점은 출력 공간에서 가까운 거친 노드(KD 트리). 평탄부(클립 고원)에서 시작해 막히면 다음 후보 노드로 다시 시도한다.
    """
    k = min(_START_CANDIDATES, st.coarse_e.shape[0])
    _, nn = st.tree.query(y, k=k)
    nn = nn.reshape(y.shape[0], k)
    e, rn, J = _newton(st, y, st.coarse_e[nn[:, 0]])
    for j in range(1, k):
        bad = np.flatnonzero(rn > _CONV_TOL)
        if bad.size == 0:
            break
        e2, rn2, J2 = _newton(st, y[bad], st.coarse_e[nn[bad, j]])
        better = rn2 < rn[bad]
        bi = bad[better]
        e[bi], rn[bi], J[bi] = e2[better], rn2[better], J2[better]
    return e, rn, J


def _solve_gray(st, y):
    """랭크 1(출력이 회색) 룩: 회색축 (t,t,t) 에서 단조 이분법. 회색 최선 추정."""
    tgt = y.mean(axis=1)
    lo, hi = np.zeros(y.shape[0]), np.full(y.shape[0], _E_MAX)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        v = _eval(st, np.repeat(mid[:, None], 3, axis=1), want_j=False)[:, 0]
        up = v < tgt
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.repeat((0.5 * (lo + hi))[:, None], 3, axis=1)


def _log_decode64(e):
    lo, hi = _sf.LOG_STOPS
    return _sf.LOG_MIDGRAY * 2.0 ** (lo + e * (hi - lo))


def invert_look(display, lut):
    """``sf.apply_look(·, lut)`` 의 역: (센서 출력 선형값 f32 (...,3), status u8 (...)). 비트는 BLACK/LUT_LOST 만.

    ``none`` 은 sRGB 복호. 나머지는 로그 부호화 + 65^3 삼선형 보간 자체를 뉴턴으로 푼다(소유 색에서 오차 <= LUT_INV_TOL).
    출력 클립 채널·랭크 결손·수렴 실패·조건 불량은 LUT_LOST(``linear`` 는 최선 추정), 코드값 < BLACK_CODE_EPS 는 BLACK.
    """
    d = _display_array(display, "display", min_ndim=1)
    name = _check_lut(lut)
    shape = d.shape[:-1]
    y = d.reshape(-1, 3).astype(np.float64)
    n = y.shape[0]
    status = np.zeros(n, np.uint8)
    if name == "none":
        s = _color.srgb_to_linear(y).astype(np.float32)
        return s.reshape(d.shape), status.reshape(shape)
    if n == 0:
        return np.zeros(d.shape, np.float32), status.reshape(shape)
    st = _lut_state(name)
    if st.gray:
        e = _solve_gray(st, y)
        lost = np.ones(n, bool)
    else:
        e, rn, J = _solve_codes(st, y)
        clipped = ((y <= _CLIP_EPS) | (y >= 1.0 - _CLIP_EPS)).any(axis=1)
        inv, sing = _inv3(J)
        s64 = _log_decode64(e)
        rowsum = np.abs(inv).sum(axis=2)                              # [in] = sum_out |J^-1[in,out]|
        pred = s64 * _LN2_SPAN * (rowsum * _Y_ERR + _E_ERR)           # 예측 센서 오차(채널별)
        ci = np.minimum(np.floor(e * (st.size - 1)).astype(np.intp), st.size - 2)
        n_ok = st.fold_ok.shape[0]
        in_box = (ci < n_ok).all(axis=1)
        fold_ok = st.fold_ok[tuple(np.minimum(ci, n_ok - 1).T)] | ~in_box
        ill = (sing | ~np.all(pred <= _COND_FRAC * LUT_INV_TOL, axis=1) | ~(_fold_kappa(st, e, inv) < _FOLD_KAPPA)
               | ~fold_ok)
        lost = clipped | ill | ~(rn <= _CONV_TOL)
    status[e.min(axis=1) < BLACK_CODE_EPS] |= OWN_SENSOR_BLACK
    status[lost] |= OWN_LUT_LOST
    s = _log_decode64(e).astype(np.float32)
    return s.reshape(d.shape), status.reshape(shape)


# ---------------------------------------------------------------------------------------------------------------
# 역사슬
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Explained:
    """``explain`` 결과. linear: WB 이전 장면 선형, status: 소유 상태 비트, error_bound: 소유 화소의 채널별 오차 한계."""
    linear: np.ndarray
    status: np.ndarray
    error_bound: np.ndarray

    @property
    def owned(self) -> np.ndarray:
        return self.status == OWN_OK


def _halo_indicator(sat, lost, u):
    """헤일로 광원이 될 수 있는 미확정 화소: 포화, 그리고 노출 휘도가 클 수 있는 LUT_LOST."""
    return sat | (lost & (_color.luminance(u) >= 0.5 * _sf.HALATION_THRESHOLD))


def explain(display, theta, *, spec=None) -> Explained:
    """노이즈 없는 ``look`` 의 역사슬: LUT^-1 -> 센서 -> 할레이션 고정점 -> EV/비네팅/WB 제거."""
    d = _display_array(display, "display", min_ndim=3)
    if d.ndim != 3 or min(d.shape) < 1:
        raise ValueError(f"display: (H,W,3) 이어야 한다 (shape {d.shape})")
    _check_theta(theta)
    _check_lut(theta.lut)
    _check_spec(spec)
    H, W = d.shape[:2]
    if spec is None:
        spec = _sf.sensor_spec_from_theta(theta, W)
    q = 1.0 / float((1 << int(spec.bits)) - 1)

    s32, status = invert_look(d, theta.lut)
    status = status.copy()
    s = s32.astype(np.float64)
    sat = (s32 >= 1.0 - SAT_MARGIN_LSB * q).any(axis=-1)
    status[sat] |= OWN_SENSOR_SAT

    a_halo = float(theta.halation)
    v32 = _sf.vignette_gain(H, W, float(theta.vignette)).astype(np.float64)
    ev32 = float(np.float32(2.0 ** float(theta.ev)))
    g32 = _gains32(theta).astype(np.float64)
    scale = v32[..., None] * ev32 * g32[None, None, :]

    u = s
    if a_halo > 0.0:
        # 할레이션 고정점 u = max(s - layer(u), 0). 수축률 <= a*Y_H < 1 이라 기하급수로 수렴한다.
        u = s.copy()
        for _ in range(_MAX_HALO_ITER):
            layer = _sf.halation_layer(u.astype(np.float32), a_halo).astype(np.float64)
            un = np.maximum(s - layer, 0.0)
            done = float(np.max(np.abs(un - u))) <= 1e-9
            u = un
            if done:
                break
        # 포화 광원(과 휘도가 클 수 있는 룩 손실 화소)의 미지 휘도가 헤일로로 번지는 범위를 가린다.
        lost = (status & OWN_LUT_LOST) != 0
        ind = _halo_indicator(sat, lost, u)
        if ind.any():
            blur = _ndi.gaussian_filter(ind.astype(np.float64), _sf.HALATION_SIGMA_FRAC * W, mode="constant",
                                        truncate=5.0)
            status[blur > HALO_TAINT_EPS] |= OWN_HALO_TAINT

    linear = (u / scale).astype(np.float32)

    # 오차 한계(계약서 식). 이득은 float64 정의값을 쓴다.
    g64 = wb_gains(theta.wb_kelvin, theta.wb_tint)
    y_h = float(_color.luminance(np.asarray(_sf.HALATION_TINT, np.float64)))
    tint = np.asarray(_sf.HALATION_TINT, np.float64)
    k = 1.0 + a_halo * tint / (1.0 - a_halo * y_h)
    scale_b = v32[..., None] * (2.0 ** float(theta.ev)) * g64[None, None, :]
    bound = (0.5 * q + LUT_INV_TOL) * k[None, None, :] / scale_b
    owned = status == OWN_OK
    err = np.where(owned[..., None], bound, np.inf).astype(np.float32)
    return Explained(linear=linear, status=status, error_bound=err)
