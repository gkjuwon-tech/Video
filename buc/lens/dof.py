"""LNS-02: 물리 피사계 심도 (박막 렌즈 CoC), Defocus Pyramid, 포커스 브리딩, 디옵터 랙 스케줄.

구조(칼선 방지의 핵심):
  * 부호 CoC(앞=음, 뒤=양)는 깊이에 단조이므로, 이산 CoC 매듭(knot) 위의 "겹침 없는 선형 분할"로 장면을 깊이 슬라이스로 나눈다.
    화소의 CoC 가 매듭 k_n, k_{n+1} 사이면 두 슬라이스의 커널 결과를 선형 보간한다(커널 크기의 이산 보간).
  * 슬라이스 n 의 over 알파는 a_n = clip((k_{n+1}-c)/(k_{n+1}-k_n), 0, 1): 자기보다 가까운(c<=k_n) 화소는 1 (가리는 쪽),
    먼 화소는 0. 가려져 알 수 없는 색은 pull-push 로 채워 두 번 세지 않는다(어두운 링 없음).
  * 슬라이스마다 전치 알파(rgb·a, a)를 조리개 커널로 퍼뜨리고 먼 것부터 가까운 것 순서로 over 합성한다.
  * CoC 가 임계값을 넘으면 위상 보존 텐트 피라미드(FND-06 pyr_down/pyr_up)에서 흐리고, 텐트 필터가 더한 분산을
    커널 E[r^2] 에서 보상한다.
import 부작용·전역 RNG·경고 없음. torch 는 쓰지 않는다.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve

from buc.core import color as _color
from buc.core import contracts as _C
from buc.core import pyramid as _pyr

__all__ = [
    "DofOpts", "DofResult", "coc_px", "coc_from_theta", "defocus_level", "aperture_kernel", "apply_dof",
    "apply_dof_theta", "rack_schedule", "breathing_scale", "breathing_K", "apply_breathing", "apply_dof_frames",
]


# ---------------------------------------------------------------------------------------------------------------
# 옵션·결과
# ---------------------------------------------------------------------------------------------------------------
def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _finite(name, x):
    if not _is_real(x) or not math.isfinite(float(x)):
        raise ValueError(f"{name}={x!r}: 유한한 실수여야 한다")
    return float(x)


@dataclass(frozen=True, kw_only=True)
class DofOpts:
    blades: int = 0
    blade_rot_deg: float = 0.0
    pyramid: bool = True
    pyramid_threshold_px: float = 8.0
    max_coc_px: float = 64.0
    in_focus_px: float = 0.5
    breathing_beta: float = 0.0

    def __post_init__(self):
        b = self.blades
        if isinstance(b, (bool, np.bool_)) or not isinstance(b, numbers.Integral):
            raise ValueError(f"blades={b!r}: 정수여야 한다")
        if not (b == 0 or 3 <= b <= 18):
            raise ValueError(f"blades={b}: 0 또는 3..18 이어야 한다")
        _finite("blade_rot_deg", self.blade_rot_deg)
        if not isinstance(self.pyramid, (bool, np.bool_)):
            raise ValueError(f"pyramid={self.pyramid!r}: bool 이어야 한다")
        if _finite("pyramid_threshold_px", self.pyramid_threshold_px) < 2.0:
            raise ValueError("pyramid_threshold_px >= 2 이어야 한다")
        if _finite("max_coc_px", self.max_coc_px) < 1.0:
            raise ValueError("max_coc_px >= 1 이어야 한다")
        if not 0.0 <= _finite("in_focus_px", self.in_focus_px) < 2.0:
            raise ValueError("in_focus_px 는 [0, 2) 이어야 한다")
        if not -1.0 <= _finite("breathing_beta", self.breathing_beta) <= 1.0:
            raise ValueError("breathing_beta 는 [-1, 1] 이어야 한다")


@dataclass(frozen=True, kw_only=True)
class DofResult:
    rgb: np.ndarray
    coc_px: np.ndarray


# ---------------------------------------------------------------------------------------------------------------
# CoC
# ---------------------------------------------------------------------------------------------------------------
def _pos_scalar(name, x):
    v = _finite(name, x)
    if v <= 0.0:
        raise ValueError(f"{name}={v}: 양수여야 한다")
    return v


def _check_lens(focal_mm, f_stop, focus_m, sensor_width_mm, width_px):
    f = _pos_scalar("focal_mm", focal_mm)
    n = _pos_scalar("f_stop", f_stop)
    s = _pos_scalar("sensor_width_mm", sensor_width_mm)
    if not _is_real(width_px) or not math.isfinite(float(width_px)) or float(width_px) < 1:
        raise ValueError(f"width_px={width_px!r}: >= 1 이어야 한다")
    if not _is_real(focus_m) or math.isnan(float(focus_m)):
        raise ValueError(f"focus_m={focus_m!r}: 유효한 거리여야 한다")
    zf = float(focus_m)
    if not zf * 1000.0 > f:
        raise ValueError(f"focus_m={zf}: 초점거리({f} mm)보다 멀어야 한다")
    return f, n, zf, s, float(width_px)


def _check_depth(depth):
    z = np.asarray(depth)
    if z.dtype.kind not in "iuf":
        raise ValueError(f"depth: 실수 배열이어야 한다 (dtype {z.dtype})")
    z = z.astype(np.float64)
    if np.isnan(z).any():
        raise ValueError("depth: NaN 이 있다")
    if not (z > 0.0).all():
        raise ValueError("depth: 양수여야 한다 (+inf 허용)")
    return z


def _coc64(z, f, n, zf, s, w, signed):
    """부호 있는 CoC(px, float64). 디옵터 형: c_mm = f²/N · (1/z - 1/z_f) · z_f/(z_f - f)."""
    inv_z = 1.0 / (z * 1000.0)
    if math.isinf(zf):
        inv_zf, gain = 0.0, 1.0
    else:
        zf_mm = zf * 1000.0
        inv_zf, gain = 1.0 / zf_mm, zf_mm / (zf_mm - f)
    # 앞(z<z_f)은 1/z > 1/z_f 이므로 음수, 뒤는 양수.
    c = (f * f / n) * gain * (inv_zf - inv_z) * (w / s)
    return c if signed else np.abs(c)


def coc_px(depth, *, focal_mm, f_stop, focus_m, sensor_width_mm, width_px: int, signed=False) -> np.ndarray:
    f, n, zf, s, w = _check_lens(focal_mm, f_stop, focus_m, sensor_width_mm, width_px)
    z = _check_depth(depth)
    # c_signed = k (z - z_f)/z ; (1/z_f - 1/z) = (z - z_f)/(z z_f) 와 같은 부호(앞 음, 뒤 양)
    return np.asarray(_coc64(z, f, n, zf, s, w, bool(signed)), dtype=np.float64).astype(np.float32)


def coc_from_theta(depth, theta, width_px, *, focus_m=None, signed=False) -> np.ndarray:
    return coc_px(depth, focal_mm=theta.focal_mm, f_stop=theta.f_stop,
                  focus_m=theta.focus_m if focus_m is None else focus_m,
                  sensor_width_mm=theta.sensor_width_mm, width_px=width_px, signed=signed)


def defocus_level(coc, threshold_px=8.0) -> np.ndarray:
    thr = _pos_scalar("threshold_px", threshold_px)
    a = np.abs(np.asarray(coc, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.ceil(np.log2(np.where(a > thr, a / thr, 1.0)))
    return np.where(a > thr, lv, 0.0).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------
# 조리개 커널
# ---------------------------------------------------------------------------------------------------------------
_SS = 8


def _shape_q(blades: int) -> float:
    """반지름 R 짜리 (내접) 조리개의 연속 E[r²] = q·R²."""
    if blades == 0:
        return 0.5
    return (2.0 + math.cos(2.0 * math.pi / blades)) / 6.0


def _coverage(R: float, blades: int, rot_deg: float, half: int) -> np.ndarray:
    o = (np.arange(_SS) + 0.5) / _SS - 0.5
    ax = np.arange(-half, half + 1, dtype=np.float64)
    # 축 순서: (row=y, col=x, sy, sx)
    Xg = np.broadcast_to(ax[None, :, None, None] + o[None, None, None, :], (2 * half + 1, 2 * half + 1, _SS, _SS))
    Yg = np.broadcast_to(ax[:, None, None, None] + o[None, None, :, None], (2 * half + 1, 2 * half + 1, _SS, _SS))
    if blades == 0:
        inside = (Xg * Xg + Yg * Yg) <= R * R
    else:
        apothem = R * math.cos(math.pi / blades)
        rot = math.radians(rot_deg)
        m = None
        for i in range(blades):
            phi = rot + (2 * i + 1) * math.pi / blades
            proj = Xg * math.cos(phi) + Yg * math.sin(phi)
            m = proj if m is None else np.maximum(m, proj)
        inside = m <= apothem
    return inside.mean(axis=(2, 3))


def _disc_m2(k: np.ndarray) -> float:
    half = (k.shape[0] - 1) // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    r2 = ax[:, None] ** 2 + ax[None, :] ** 2
    return float((k * r2).sum() / k.sum())


@lru_cache(maxsize=1024)
def _kernel_m2(m2_target: float, blades: int, rot_deg: float, half: int) -> np.ndarray:
    """이산 E[r²] 가 m2_target 이 되도록 크기를 맞춘 면적 적분 조리개 커널(합 1, float64, 읽기 전용)."""
    q = _shape_q(blades)
    rmax = half + 0.5
    if m2_target <= 1e-9:
        k = np.zeros((2 * half + 1,) * 2)
        k[half, half] = 1.0
        k.flags.writeable = False
        return k
    r2 = (m2_target + 1.0 / 6.0) / q
    best = None
    for _ in range(6):
        R = min(max(math.sqrt(max(r2, 0.25)), 0.5), rmax)
        cov = _coverage(R, blades, rot_deg, half)
        if cov.sum() <= 0:
            cov = np.zeros_like(cov)
            cov[half, half] = 1.0
        k = cov / cov.sum()
        err = m2_target - _disc_m2(k)
        best = k
        if abs(err) <= 2e-3 * max(m2_target, 0.05) or (R >= rmax and err > 0):
            break
        r2 = R * R + err / q
    best = np.array(best, dtype=np.float64)
    best.flags.writeable = False
    return best


def _check_blades(blades):
    if isinstance(blades, (bool, np.bool_)) or not isinstance(blades, numbers.Integral):
        raise ValueError(f"blades={blades!r}: 정수여야 한다")
    if not (blades == 0 or 3 <= blades <= 18):
        raise ValueError(f"blades={blades}: 0 또는 3..18 이어야 한다")
    return int(blades)


def _rot_key(rotation_deg, blades):
    if blades == 0:
        return 0.0
    period = 360.0 / blades
    return round(float(rotation_deg) % period, 9)


def aperture_kernel(diameter_px, blades=0, rotation_deg=0.0) -> np.ndarray:
    if not _is_real(diameter_px) or not math.isfinite(float(diameter_px)) or float(diameter_px) < 0:
        raise ValueError(f"diameter_px={diameter_px!r}: 0 이상의 유한 실수여야 한다")
    b = _check_blades(blades)
    _finite("rotation_deg", rotation_deg)
    d = float(diameter_px)
    if d <= 1.0:
        return np.ones((1, 1), np.float32)
    half = int(math.ceil(d / 2.0))
    m2 = _shape_q(b) * (d / 2.0) ** 2
    return _kernel_m2(round(m2, 9), b, _rot_key(rotation_deg, b), half).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# 컨볼루션 · pull-push · 슬라이스 렌더
# ---------------------------------------------------------------------------------------------------------------
def _conv_edge(a: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """(H,W,C) float64 를 커널로 컨볼루션(가장자리 복제). 커널은 합 1 이므로 상수는 보존된다."""
    kh, kw = ker.shape
    if kh == 1 and kw == 1:
        return a
    ry, rx = kh // 2, kw // 2
    p = np.pad(a, ((ry, ry), (rx, rx), (0, 0)), mode="edge")
    if kh * kw <= 25:
        H, W = a.shape[:2]
        out = np.zeros_like(a)
        for i in range(kh):
            for j in range(kw):
                w = ker[2 * ry - i, 2 * rx - j]  # 진짜 컨볼루션(커널 뒤집기)
                if w != 0.0:
                    out += w * p[i:i + H, j:j + W]
        return out
    return fftconvolve(p, ker[:, :, None], mode="valid")


def _fill(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    """pull-push: known 이 아닌 화소의 색을 known 화소들로부터 부드럽게 채운다. known 화소는 그대로."""
    if known.all():
        return rgb
    w0 = known.astype(np.float64)
    v = np.concatenate([rgb * w0[..., None], w0[..., None]], axis=-1)
    levels = []
    while True:
        levels.append(v)
        if float(v[..., 3].min()) > 1e-6 or v.shape[:2] == (1, 1):
            break
        h, w = v.shape[:2]
        vp = np.pad(v, ((0, max(2, h + (h & 1)) - h), (0, max(2, w + (w & 1)) - w), (0, 0)))
        v = _pyr.pyr_down(vp)
    top = levels[-1]
    fc = top[..., :3] / np.maximum(top[..., 3:4], 1e-12)
    for lv in range(len(levels) - 2, -1, -1):
        v = levels[lv]
        h, w = v.shape[:2]
        up = _pyr.pyr_up(fc)[:h, :w]
        fc = v[..., :3] + (1.0 - v[..., 3:4]) * up
    return fc


def _knot_grid(max_c: float) -> np.ndarray:
    """부호 CoC 매듭: |c|<=16 은 1 px, 16..32 는 2 px, 32..64 는 4 px, ... (옥타브마다 간격 2배). ±max 포함."""
    pos = [0.0]
    step = 1.0
    nxt = 16.0
    while pos[-1] < max_c - 1e-9:
        v = pos[-1] + step
        pos.append(v)
        if v >= nxt - 1e-9:
            step *= 2.0
            nxt *= 2.0
    pos = [p for p in pos if p < max_c - 1e-9] + [float(max_c)]
    p = np.array(pos, dtype=np.float64)
    return np.concatenate([-p[:0:-1], p])


_MAX_OVERHEAD = 0.2


def _level_for(d: float, m2_t: float, opts: DofOpts) -> int:
    if not opts.pyramid or d <= opts.pyramid_threshold_px:
        return 0
    lv = int(defocus_level(np.array([d]), opts.pyramid_threshold_px)[0])
    # 텐트 필터가 더하는 분산이 목표 E[r²] 의 _MAX_OVERHEAD 를 넘으면 한 단계 낮춰 모양 오차를 줄인다
    while lv > 0 and (4 ** lv - 1) > _MAX_OVERHEAD * m2_t:
        lv -= 1
    return lv


def _blur_stack(stack: np.ndarray, d: float, opts: DofOpts) -> np.ndarray:
    """(H,W,C) 전치 스택을 지름 d(px) 조리개로 흐린다. 필요하면 Defocus Pyramid 를 쓴다."""
    if d <= 1.0:
        return stack
    b = opts.blades
    rot = _rot_key(opts.blade_rot_deg, b)
    m2_t = _shape_q(b) * (d / 2.0) ** 2
    lv = _level_for(d, m2_t, opts)
    if lv == 0:
        half = int(math.ceil(d / 2.0))
        return _conv_edge(stack, _kernel_m2(round(m2_t, 9), b, rot, half))
    s = 2 ** lv
    H, W = stack.shape[:2]
    ph, pw = (-H) % s, (-W) % s
    v = np.pad(stack, ((0, ph), (0, pw), (0, 0)), mode="edge") if (ph or pw) else stack
    for _ in range(lv):
        v = _pyr.pyr_down(v)
    # 텐트 down/up 이 더한 분산 (4^L - 1) 을 E[r²] 에서 보상한다 (원 해상도 단위)
    m2_c = max(m2_t - (4 ** lv - 1), 0.0) / (4 ** lv)
    if m2_c > 1e-9:
        # 기준 크기: 보상 후의 반지름
        half = int(math.ceil(math.sqrt(m2_c / _shape_q(b)))) + 1
        v = _conv_edge(v, _kernel_m2(round(m2_c, 9), b, rot, half))
    for _ in range(lv):
        v = _pyr.pyr_up(v)
    return v[:H, :W]


def _signed_clip(c, opts):
    c = np.clip(c, -opts.max_coc_px, opts.max_coc_px)
    return np.where(np.abs(c) <= opts.in_focus_px, 0.0, c)


def _render(rgb: np.ndarray, c: np.ndarray, opts: DofOpts) -> np.ndarray:
    """rgb (H,W,3) float64 선형, c (H,W) 부호 CoC(px, 이미 클립·in_focus 처리). 합성 결과 float64."""
    grid = _knot_grid(opts.max_coc_px)
    # 매듭에 가까우면 붙인다
    j = np.clip(np.searchsorted(grid, c), 1, len(grid) - 1)
    nearest = np.where(np.abs(c - grid[j - 1]) <= np.abs(c - grid[j]), j - 1, j)
    snap = np.abs(c - grid[nearest]) <= 1e-3
    c = np.where(snap, grid[nearest], c)
    lo = np.clip(np.searchsorted(grid, c, side="right") - 1, 0, len(grid) - 1)
    hi_needed = c > grid[lo]
    used = np.zeros(len(grid), bool)
    used[lo] = True
    used[np.minimum(lo[hi_needed] + 1, len(grid) - 1)] = True
    K = grid[used]
    m = len(K)
    acc = None
    for n in range(m - 1, -1, -1):
        d = abs(float(K[n]))
        if n == m - 1:
            a = np.ones(c.shape, np.float64)
        else:
            a = np.clip((K[n + 1] - c) / (K[n + 1] - K[n]), 0.0, 1.0)
        known = np.ones(c.shape, bool) if n == 0 else (c > K[n - 1])
        col = _fill(rgb, known)
        full = bool((a >= 1.0).all())
        stack = np.concatenate([col * a[..., None], a[..., None]], axis=-1) if not full else \
            np.concatenate([col, np.ones(c.shape + (1,), np.float64)], axis=-1)
        if full:
            blurred = _blur_stack(col, d, opts)
            ba = None
            premult = blurred
        else:
            blurred = _blur_stack(stack, d, opts)
            premult, ba = blurred[..., :3], np.clip(blurred[..., 3], 0.0, 1.0)
        acc = premult if acc is None else premult + (1.0 - ba)[..., None] * acc
    return acc


# ---------------------------------------------------------------------------------------------------------------
# apply_dof
# ---------------------------------------------------------------------------------------------------------------
def apply_dof(rgb, depth, *, focal_mm, f_stop, focus_m, sensor_width_mm, opts=None) -> DofResult:
    opts = DofOpts() if opts is None else opts
    if not isinstance(opts, DofOpts):
        raise ValueError("opts 는 DofOpts 여야 한다")
    img = np.asarray(rgb)
    if img.dtype.kind not in "iuf":
        raise ValueError(f"rgb: 실수 배열이어야 한다 (dtype {img.dtype})")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"rgb: (H,W,3) 이어야 한다 (shape {img.shape})")
    z = np.asarray(depth)
    if z.ndim != 2 or z.shape != img.shape[:2]:
        raise ValueError(f"depth: (H,W)={img.shape[:2]} 이어야 한다 (shape {z.shape})")
    img64 = img.astype(np.float64)
    if not np.isfinite(img64).all():
        raise ValueError("rgb: 유한하지 않은 값이 있다")
    f, n, zf, s, w = _check_lens(focal_mm, f_stop, focus_m, sensor_width_mm, img.shape[1])
    z64 = _check_depth(z)
    c = _coc64(z64, f, n, zf, s, w, True)
    ac = np.minimum(np.abs(c), opts.max_coc_px)
    coc_out = ac.astype(np.float32)
    if float(ac.max()) <= opts.in_focus_px:
        return DofResult(rgb=img.astype(np.float32, copy=True), coc_px=coc_out)
    cs = _signed_clip(np.clip(c, -opts.max_coc_px, opts.max_coc_px), opts)
    out = _render(img64, cs, opts)
    return DofResult(rgb=out.astype(np.float32), coc_px=coc_out)


def apply_dof_theta(rgb, depth, theta, *, focus_m=None, opts=None) -> DofResult:
    if opts is None:
        opts = DofOpts(blades=int(theta.extra.get("blades", 0)))
    return apply_dof(rgb, depth, focal_mm=theta.focal_mm, f_stop=theta.f_stop,
                     focus_m=theta.focus_m if focus_m is None else focus_m,
                     sensor_width_mm=theta.sensor_width_mm, opts=opts)


# ---------------------------------------------------------------------------------------------------------------
# 랙 스케줄
# ---------------------------------------------------------------------------------------------------------------
def _focus_val(name, x):
    if not _is_real(x) or math.isnan(float(x)) or float(x) <= 0.0:
        raise ValueError(f"{name}={x!r}: 양수(또는 inf)여야 한다")
    return float(x)


def rack_schedule(n_frames, fps, focus_from_m, focus_to_m, *, start_s=0.0, duration_s=None,
                  ease="smooth") -> np.ndarray:
    if isinstance(n_frames, (bool, np.bool_)) or not isinstance(n_frames, numbers.Integral) or n_frames < 1:
        raise ValueError(f"n_frames={n_frames!r}: 1 이상의 정수여야 한다")
    n = int(n_frames)
    fps = _pos_scalar("fps", fps)
    z0, z1 = _focus_val("focus_from_m", focus_from_m), _focus_val("focus_to_m", focus_to_m)
    if ease not in ("linear", "smooth"):
        raise ValueError(f"ease={ease!r}: 'linear' 또는 'smooth'")
    start = _finite("start_s", start_s)
    if start < 0.0:
        raise ValueError("start_s >= 0 이어야 한다")
    if duration_s is not None and _pos_scalar("duration_s", duration_s) <= 0.0:
        raise ValueError("duration_s > 0 이어야 한다")
    if n == 1:
        return np.array([z0], dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / fps
    dur = float(duration_s) if duration_s is not None else (n - 1) / fps - start
    if dur <= 0.0:
        return np.full(n, z0, dtype=np.float64)
    u = np.clip((t - start) / dur, 0.0, 1.0)
    e = u if ease == "linear" else u * u * (3.0 - 2.0 * u)
    i0 = 0.0 if math.isinf(z0) else 1.0 / z0
    i1 = 0.0 if math.isinf(z1) else 1.0 / z1
    inv = i0 + (i1 - i0) * e
    with np.errstate(divide="ignore"):
        z = 1.0 / inv
    z = np.where(u <= 0.0, z0, np.where(u >= 1.0, z1, z))
    return z.astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------
# 포커스 브리딩
# ---------------------------------------------------------------------------------------------------------------
def _magnif(f_m, z):
    z = np.asarray(z, dtype=np.float64)
    if np.isnan(z).any() or not (z > f_m).all():
        raise ValueError("focus_m 은 초점거리보다 멀어야 한다 (NaN 불가)")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isinf(z), 0.0, f_m / (z - f_m))


def breathing_scale(focal_mm, focus_m, beta, ref_focus_m=math.inf):
    f = _pos_scalar("focal_mm", focal_mm) / 1000.0
    b = _finite("beta", beta)
    if abs(b) > 1.0:
        raise ValueError("|beta| <= 1 이어야 한다")
    m = _magnif(f, focus_m)
    mr = _magnif(f, ref_focus_m)
    s = (1.0 + b * m) / (1.0 + b * mr)
    return float(s) if np.ndim(s) == 0 else s


def breathing_K(K, scale) -> np.ndarray:
    k = np.array(K, dtype=np.float64, copy=True)
    s = float(scale)
    k[..., 0, 0] *= s
    k[..., 1, 1] *= s
    return k


def apply_breathing(img, scale, K, *, nearest=False) -> np.ndarray:
    a = np.asarray(img)
    s = _pos_scalar("scale", scale)
    if a.ndim not in (2, 3):
        raise ValueError(f"img: (H,W) 또는 (H,W,C) 여야 한다 (shape {a.shape})")
    out_dtype = a.dtype if (nearest or a.dtype in (np.float32, np.float64)) else np.float32
    if s == 1.0:
        return a.astype(out_dtype, copy=True)
    Km = np.asarray(K, dtype=np.float64)
    cx, cy = float(Km[0, 2]), float(Km[1, 2])
    H, W = a.shape[:2]
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    su, sv = cx + (u - cx) / s, cy + (v - cy) / s
    if nearest:
        iu = np.clip(np.floor(su + 0.5), 0, W - 1).astype(np.int64)
        iv = np.clip(np.floor(sv + 0.5), 0, H - 1).astype(np.int64)
        return a[iv, iu].astype(out_dtype, copy=False)
    src = a if a.dtype in (np.float32, np.float64) else a.astype(np.float32)
    res = _pyr.sample_bilinear(src, np.stack([su, sv], axis=-1), border="edge")
    return np.asarray(res).astype(out_dtype)


# ---------------------------------------------------------------------------------------------------------------
# Frames 이음매
# ---------------------------------------------------------------------------------------------------------------
def apply_dof_frames(frames, shot, opts=None):
    th = shot.theta
    if "depth" not in frames.aov:
        raise ValueError("frames.aov['depth'] 가 필요하다")
    depth = frames.aov["depth"]
    rgb = frames.rgb
    N = int(shot.focus_m.shape[0])
    if rgb.ndim != 4 or rgb.shape[0] != N or depth.shape != rgb.shape[:3]:
        raise ValueError(f"Frames {rgb.shape}/depth {depth.shape} 가 shot(N={N}) 과 맞지 않는다")
    if rgb.shape[1] != shot.height or rgb.shape[2] != shot.width:
        raise ValueError(f"Frames {rgb.shape[1:3]} 가 shot {(shot.height, shot.width)} 와 다르다")
    if opts is None:
        opts = DofOpts(blades=int(th.extra.get("blades", 0)))
    beta = opts.breathing_beta
    H, W = rgb.shape[1:3]
    out_rgb = np.empty((N, H, W, 3), np.float32)
    out_depth = np.empty((N, H, W), depth.dtype if depth.dtype in (np.float32, np.float64) else np.float32)
    out_coc = np.empty((N, H, W), np.float32)
    f0 = float(shot.focus_m[0])
    for i in range(N):
        lin = _color.srgb_to_linear(rgb[i])
        d = depth[i]
        fm = float(shot.focus_m[i])
        if beta != 0.0:
            s = breathing_scale(th.focal_mm, fm, beta, ref_focus_m=f0)
            lin = apply_breathing(lin, s, shot.K[i])
            d = apply_breathing(d, s, shot.K[i], nearest=True)
        r = apply_dof(lin, d, focal_mm=th.focal_mm, f_stop=th.f_stop, focus_m=fm,
                      sensor_width_mm=th.sensor_width_mm, opts=opts)
        out_rgb[i] = _color.linear_to_display(r.rgb)
        out_depth[i] = d
        out_coc[i] = r.coc_px
    aov = dict(frames.aov)
    aov["depth"] = out_depth
    aov["coc_px"] = out_coc
    return _C.Frames(rgb=out_rgb, fps=frames.fps, aov=aov, shot_hash=frames.shot_hash,
                     world_hash=frames.world_hash)
