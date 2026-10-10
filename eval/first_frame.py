"""EVL-04 첫 프레임 KPI v0: FFD, FFG, dE00, dG, FP(t), OIR (R-EVAL).

평가 전용 LPIPS 는 별도 워커 프로세스에서만 돈다. 이 모듈의 모듈 수준 import 에는 torch/lpips 가 없다.
워커 실행: ``python eval/first_frame.py --worker --backend B``.
"""
from __future__ import annotations

import json
import math
import os
import select
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

if __name__ == "__main__":  # 워커로 실행될 때만 레포 루트를 경로에 넣는다(import 부작용 없음)
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import numpy as np
from scipy import ndimage

from buc.core.color import delta_e_2000, rgb_lin_to_lab, srgb_to_linear
from buc.core.pyramid import gaussian_blur, pyr_down, pyr_up, resize

SCHEMA_VERSION = 1
R_EVAL = True
REF_SHORT = 1080
GRAIN_BLUR_SIGMA = 0.7
PATCH = 32
SAL_FRAC = 0.10
GRAIN_F_LO = 1 / 16
GRAIN_F_HI = 0.5
FP_MIN_COVIS = 0.05
FFD_FLOOR_MAX = 0.015
DE00_FLOOR_MAX = 0.5

_BACKENDS = ("lpips", "proxy")
_REC709 = np.array([0.2126, 0.7152, 0.0722], np.float64)


class EvalWorkerError(RuntimeError):
    """평가 워커 프로세스의 사망·타임아웃·프로토콜 오류."""


class EvalModelUnavailable(EvalWorkerError):
    """진짜 LPIPS 패키지/가중치/장치를 쓸 수 없다. 조용히 proxy 로 바꾸지 않는다."""


# ------------------------------------------------------------------------------------------- 입력 규약
def as_display(img, name="img"):
    """표시값 (H,W,3) float32 [0,1]. uint8 은 /255."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / np.float32(255.0)
        shape_ok = True
    elif a.dtype in (np.float32, np.float64):
        shape_ok = True
    else:
        raise TypeError(f"{name}: dtype {a.dtype} 는 지원하지 않는다(uint8/float32/float64)")
    if not shape_ok or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{name}: (H,W,3) 이어야 한다, 받은 shape={a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name}: 빈 영상")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: NaN/Inf")
    if a.min() < -1e-6 or a.max() > 1.0 + 1e-6:
        raise ValueError(f"{name}: [0,1] 범위 밖")
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _pair(inp, f0):
    a = as_display(inp, "inp")
    b = as_display(f0, "f0")
    if a.shape != b.shape:
        raise ValueError(f"크기가 다르다: {a.shape} vs {b.shape}")
    return a, b


def _mask(mask, shape, name="mask"):
    m = np.asarray(mask)
    if m.dtype != np.bool_ or m.shape != tuple(shape[:2]):
        raise ValueError(f"{name}: bool {tuple(shape[:2])} 이어야 한다")
    return m


def _lum(img):
    return np.asarray(img, np.float64) @ _REC709


# ------------------------------------------------------------------------------------------- Psi, B
def prepare(img, *, r_eff=None, short_side=REF_SHORT, sigma=GRAIN_BLUR_SIGMA):
    a = as_display(img)
    if short_side < 1:
        raise ValueError("short_side >= 1")
    target = int(short_side)
    if r_eff is not None:
        if int(r_eff) < 1:
            raise ValueError("r_eff >= 1")
        target = min(target, int(r_eff))
    h, w = a.shape[:2]
    s = min(h, w)
    if s > target:
        if h <= w:
            nh, nw = target, (2 * w * target + s) // (2 * s)
        else:
            nw, nh = target, (2 * h * target + s) // (2 * s)
        a = resize(a, (int(nh), int(nw)), "bilinear")
    return np.asarray(gaussian_blur(a, sigma), np.float32)


def ffd(inp, f0, client, *, r_eff=None):
    a, b = _pair(inp, f0)
    return float(client.distance(prepare(a, r_eff=r_eff), prepare(b, r_eff=r_eff)))


# ------------------------------------------------------------------------------------------- dE00
def delta_e00(inp, f0, *, mask=None, patch=PATCH, min_cover=0.5):
    a, b = _pair(inp, f0)
    return _de00(a, b, mask, patch, min_cover, strict=True)


def _de00(a, b, mask, patch, min_cover, strict):
    if int(patch) != patch or patch < 1:
        raise ValueError("patch >= 1 정수")
    patch = int(patch)
    h, w = a.shape[:2]
    if h < patch or w < patch:
        raise ValueError("영상이 패치보다 작다")
    ny, nx = h // patch, w // patch
    if mask is not None:
        mask = _mask(mask, a.shape)
        wgt = mask[: ny * patch, : nx * patch].reshape(ny, patch, nx, patch).astype(np.float64)
    else:
        wgt = np.ones((ny, patch, nx, patch), np.float64)
    cover = wgt.mean(axis=(1, 3))
    use = (cover >= min_cover - 1e-12) & (cover > 0)
    if not use.any():
        if strict:
            raise ValueError("조건을 만족하는 패치가 없다")
        return float("nan")
    cnt = wgt.sum(axis=(1, 3))
    cnt = np.where(cnt > 0, cnt, 1.0)
    labs = []
    for img in (a, b):
        lin = srgb_to_linear(np.asarray(img, np.float64))[: ny * patch, : nx * patch]
        lin = lin.reshape(ny, patch, nx, patch, 3)
        mean = (lin * wgt[..., None]).sum(axis=(1, 3)) / cnt[..., None]
        labs.append(rgb_lin_to_lab(mean))
    de = delta_e_2000(labs[0], labs[1])
    return float(np.mean(np.asarray(de)[use]))


# ------------------------------------------------------------------------------------------- 광류 / FFG
def estimate_flow(a, b):
    """피라미드 Lucas-Kanade. 채널 (dx,dy), a(p) ~ b(p + flow(p))."""
    a, b = _pair(a, b)
    la, lb = _lum(a), _lum(b)
    h, w = la.shape
    levels = int(max(0, min(4, math.floor(math.log2(max(min(h, w), 1) / 16.0))))) if min(h, w) >= 16 else 0
    pa, pb = [la], [lb]
    for _ in range(levels):
        pa.append(pyr_down(pa[-1]))
        pb.append(pyr_down(pb[-1]))
    flow = np.zeros(pa[-1].shape + (2,), np.float64)
    for lv in range(levels, -1, -1):
        A, B = pa[lv], pb[lv]
        if flow.shape[:2] != A.shape:
            flow = 2.0 * pyr_up(flow, A.shape)
        iters = 8 if lv > 0 else 6
        yy, xx = np.mgrid[0 : A.shape[0], 0 : A.shape[1]].astype(np.float64)
        for _ in range(iters):
            coords = np.stack([yy + flow[..., 1], xx + flow[..., 0]])
            bw = ndimage.map_coordinates(B, coords, order=3, mode="nearest")
            valid = ((coords[0] >= 0) & (coords[0] <= A.shape[0] - 1) & (coords[1] >= 0)
                     & (coords[1] <= A.shape[1] - 1)).astype(np.float64)
            gy = ndimage.gaussian_filter(bw, 1.0, order=(1, 0), mode="nearest")
            gx = ndimage.gaussian_filter(bw, 1.0, order=(0, 1), mode="nearest")
            gx, gy = gx * valid, gy * valid
            r = (A - bw) * valid
            sig = 2.5
            sxx = ndimage.gaussian_filter(gx * gx, sig, mode="nearest")
            syy = ndimage.gaussian_filter(gy * gy, sig, mode="nearest")
            sxy = ndimage.gaussian_filter(gx * gy, sig, mode="nearest")
            bx = ndimage.gaussian_filter(gx * r, sig, mode="nearest")
            by = ndimage.gaussian_filter(gy * r, sig, mode="nearest")
            ridge = 1e-2 * (sxx + syy) + 0.1 * float((sxx + syy).mean()) + 1e-9
            sxx, syy = sxx + ridge, syy + ridge
            det = sxx * syy - sxy * sxy
            dx = (syy * bx - sxy * by) / det
            dy = (sxx * by - sxy * bx) / det
            support = ndimage.gaussian_filter(valid, sig, mode="nearest") > 0.5  # 대응이 영상 밖이면 갱신 안 함
            flow[..., 0] += np.where(support, np.clip(dx, -1.5, 1.5), 0.0)
            flow[..., 1] += np.where(support, np.clip(dy, -1.5, 1.5), 0.0)
        flow[..., 0] = ndimage.median_filter(flow[..., 0], size=3, mode="nearest")
        flow[..., 1] = ndimage.median_filter(flow[..., 1], size=3, mode="nearest")
    return flow.astype(np.float32)


@dataclass(frozen=True)
class FFGResult:
    median: float
    p95: float
    n: int


def ffg(inp, f0, *, mask=None, ref_short=REF_SHORT):
    a, b = _pair(inp, f0)
    if not ref_short > 0:
        raise ValueError("ref_short > 0")
    sel = None
    if mask is not None:
        sel = _mask(mask, a.shape)
        if not sel.any():
            raise ValueError("빈 마스크")
    fl = estimate_flow(a, b).astype(np.float64)
    mag = np.hypot(fl[..., 0], fl[..., 1]) * (float(ref_short) / min(a.shape[:2]))
    vals = mag.ravel() if sel is None else mag[sel]
    return FFGResult(float(np.median(vals)), float(np.percentile(vals, 95)), int(vals.size))


def ffg_sal(inp, f0, region_mask, *, ref_short=REF_SHORT):
    return ffg(inp, f0, mask=region_mask, ref_short=ref_short).p95


def top_fraction_mask(score, frac=SAL_FRAC):
    s = np.asarray(score, np.float64)
    if s.ndim != 2 or s.size == 0:
        raise ValueError("score 는 비어 있지 않은 2차원")
    if not np.isfinite(s).all():
        raise ValueError("score 에 NaN/Inf")
    if not (0.0 < frac <= 1.0):
        raise ValueError("0 < frac <= 1")
    k = min(s.size, max(1, int(math.ceil(frac * s.size - 1e-9))))
    order = np.argsort(-s.ravel(), kind="stable")  # 동점은 래스터 순서가 앞선 쪽
    out = np.zeros(s.size, bool)
    out[order[:k]] = True
    return out.reshape(s.shape)


# ------------------------------------------------------------------------------------------- dG
def grain_band_ratios(inp, f0, *, n_bands=6):
    a, b = _pair(inp, f0)
    if int(n_bands) != n_bands or n_bands < 1:
        raise ValueError("n_bands >= 1 정수")
    n_bands = int(n_bands)
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rad = np.hypot(fy, fx)
    edges = np.exp(np.linspace(math.log(GRAIN_F_LO), math.log(GRAIN_F_HI), n_bands + 1))
    edges[0], edges[-1] = GRAIN_F_LO, GRAIN_F_HI
    bands = []
    for i in range(n_bands):
        hi_ok = (rad <= edges[i + 1]) if i == n_bands - 1 else (rad < edges[i + 1])
        sel = (rad >= edges[i]) & hi_ok
        if not sel.any():
            raise ValueError("빈 주파수 대역: 영상이 너무 작거나 대역이 너무 많다")
        bands.append(sel)
    psd = []
    for img in (a, b):
        lum = _lum(img)
        lum = lum - lum.mean()
        p = np.abs(np.fft.fft2(lum)) ** 2 / (h * w)
        psd.append(np.array([max(float(p[s].mean()), 1e-12) for s in bands]))
    return np.log2(psd[1] / psd[0])


def delta_g(inp, f0, *, n_bands=6):
    return float(np.mean(np.abs(grain_band_ratios(inp, f0, n_bands=n_bands))))


# ------------------------------------------------------------------------------------------- proxy LPIPS
def _proxy_feats(x):
    """한 스케일의 특징: 휘도, 기울기, 색 대립 채널."""
    lum = x @ _REC709
    c1 = x[..., 0] - x[..., 1]
    c2 = x[..., 2] - 0.5 * (x[..., 0] + x[..., 1])
    return [
        ndimage.gaussian_filter(lum, 1.0, mode="reflect"),
        2.0 * ndimage.gaussian_filter(lum, 1.0, order=(1, 0), mode="reflect"),
        2.0 * ndimage.gaussian_filter(lum, 1.0, order=(0, 1), mode="reflect"),
        ndimage.gaussian_filter(c1, 1.0, mode="reflect"),
        ndimage.gaussian_filter(c2, 1.0, mode="reflect"),
    ]


def _pool2(x):
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    if h < 8 or w < 8:
        return None
    x = x[:h, :w]
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


def proxy_lpips(a, b):
    """numpy 전용 LPIPS 대역(다중 스케일 특징 거리). 지각 지표가 아니다."""
    a, b = _pair(a, b)
    xa, xb = a.astype(np.float64), b.astype(np.float64)
    total, wsum = 0.0, 0.0
    for s in range(3):
        if xa is None:
            break
        fa, fb = _proxy_feats(xa), _proxy_feats(xb)
        total += sum(float(np.mean((p - q) ** 2)) for p, q in zip(fa, fb)) * (0.5 ** s)
        wsum += 0.5 ** s
        xa, xb = _pool2(xa), _pool2(xb)
        if xa is None or xb is None:
            break
    return float(math.sqrt(total / wsum))


# ------------------------------------------------------------------------------------------- 워커 프로토콜
_HDR = struct.Struct("<II")


def _pack(header, body=b""):
    h = json.dumps(header).encode("utf-8")
    return _HDR.pack(len(h), len(body)) + h + body


def _worker_main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--backend", default="proxy")
    ap.add_argument("--net", default="alex")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    inp = os.fdopen(os.dup(0), "rb", buffering=0)
    sys.stdout = sys.stderr  # 지나가는 print 가 프로토콜을 깨지 않게

    def send(header, body=b""):
        out.write(_pack(header, body))

    def read_exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = inp.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def info():
        return {"pid": os.getpid(), "backend": args.backend, "torch_loaded": "torch" in sys.modules,
                "lpips_loaded": "lpips" in sys.modules}

    dist = None
    if args.backend == "proxy":
        dist = proxy_lpips
    elif args.backend == "lpips":
        try:
            import torch
            import lpips

            net = lpips.LPIPS(net=args.net, verbose=False).to(args.device).eval()
        except Exception as e:  # noqa: BLE001 - 어떤 실패든 '사용 불가'로 보고
            send({"ok": False, "kind": "unavailable", "msg": f"{type(e).__name__}: {e}"})
            return 0

        def dist(a, b):
            def t(x):
                return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() * 2 - 1).to(args.device)

            with torch.no_grad():
                return float(net(t(a), t(b)).reshape(-1)[0].item())
    else:
        send({"ok": False, "kind": "unavailable", "msg": f"알 수 없는 백엔드 {args.backend}"})
        return 0
    send({"ok": True, "info": info()})
    while True:
        head = read_exact(_HDR.size)
        if head is None:
            return 0
        hl, bl = _HDR.unpack(head)
        hb, body = read_exact(hl), read_exact(bl) if bl else b""
        if hb is None or body is None:
            return 0
        msg = json.loads(hb.decode("utf-8"))
        op = msg.get("op")
        try:
            if op == "shutdown":
                send({"ok": True})
                return 0
            if op == "info":
                send({"ok": True, "info": info()})
            elif op == "distance":
                shape = tuple(msg["shape"])
                n = int(np.prod(shape)) * 4
                a = np.frombuffer(body[:n], np.float32).reshape(shape)
                b = np.frombuffer(body[n:2 * n], np.float32).reshape(shape)
                send({"ok": True, "value": float(dist(a, b))})
            else:
                send({"ok": False, "kind": "protocol", "msg": f"알 수 없는 op {op}"})
        except Exception as e:  # noqa: BLE001
            send({"ok": False, "kind": "error", "msg": f"{type(e).__name__}: {e}"})


class LpipsClient:
    """워커 프로세스 1개를 소유한다. 부모 프로세스는 torch 를 만지지 않는다."""

    def __init__(self, backend="lpips", *, net="alex", device="cpu", timeout_s=120.0):
        if backend not in _BACKENDS:
            raise ValueError(f"알 수 없는 백엔드: {backend!r}")
        self.backend = backend
        self.is_perceptual = backend == "lpips"
        self.timeout_s = float(timeout_s)
        self._proc = None
        self.worker_pid = -1
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--backend", backend,
               "--net", str(net), "--device", str(device)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, cwd="/")
        self.worker_pid = self._proc.pid
        try:
            head, _ = self._recv()
        except EvalWorkerError:
            self._reap()
            raise
        if not head.get("ok"):
            self._reap()
            if head.get("kind") == "unavailable":
                raise EvalModelUnavailable(head.get("msg", "백엔드를 쓸 수 없다"))
            raise EvalWorkerError(head.get("msg", "워커 시작 실패"))

    # ----- 전송
    def _send(self, data):
        fd = self._proc.stdin.fileno()
        view = memoryview(data)
        deadline = _now() + self.timeout_s
        try:
            while view:
                rem = deadline - _now()
                if rem <= 0:
                    raise EvalWorkerError("워커 쓰기 타임아웃")
                _, w, _ = select.select([], [fd], [], rem)
                if not w:
                    continue
                n = os.write(fd, view[:65536])
                view = view[n:]
        except (BrokenPipeError, OSError) as e:
            self._reap()
            raise EvalWorkerError(f"워커 연결 끊김: {e}") from e
        except EvalWorkerError:
            self._reap()
            raise

    def _read(self, n, deadline):
        fd = self._proc.stdout.fileno()
        buf = bytearray()
        while len(buf) < n:
            rem = deadline - _now()
            if rem <= 0:
                self._reap()
                raise EvalWorkerError("워커 응답 타임아웃")
            r, _, _ = select.select([fd], [], [], rem)
            if not r:
                continue
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                self._reap()
                raise EvalWorkerError("워커가 응답 없이 종료되었다")
            buf += chunk
        return bytes(buf)

    def _recv(self):
        deadline = _now() + self.timeout_s
        hl, bl = _HDR.unpack(self._read(_HDR.size, deadline))
        head = json.loads(self._read(hl, deadline).decode("utf-8"))
        body = self._read(bl, deadline) if bl else b""
        return head, body

    def _call(self, header, body=b""):
        if self._proc is None:
            raise EvalWorkerError("클라이언트가 닫혔다")
        self._send(_pack(header, body))
        head, _ = self._recv()
        if not head.get("ok"):
            raise EvalWorkerError(head.get("msg", "워커 오류"))
        return head

    # ----- 공개
    def distance(self, a, b):
        a, b = _pair(a, b)  # 검증은 부모에서 먼저
        body = np.ascontiguousarray(a, np.float32).tobytes() + np.ascontiguousarray(b, np.float32).tobytes()
        return float(self._call({"op": "distance", "shape": list(a.shape)}, body)["value"])

    def info(self):
        return dict(self._call({"op": "info"})["info"])

    def _reap(self):
        p, self._proc = self._proc, None
        if p is None:
            return
        for s in (p.stdin, p.stdout):
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        if p.poll() is None:
            try:
                p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                p.kill()
        p.wait()

    def close(self):
        p = self._proc
        if p is None:
            return
        try:
            if p.poll() is None:
                self._send(_pack({"op": "shutdown"}))
        except EvalWorkerError:
            pass
        self._reap()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self._reap()
        except Exception:  # noqa: BLE001
            pass


def _now():
    import time

    return time.monotonic()


# ------------------------------------------------------------------------------------------- FP(t)
@dataclass(frozen=True, eq=False)
class FPCurve:
    times: np.ndarray
    lpips: np.ndarray
    de00: np.ndarray
    covis: np.ndarray
    slope_lpips: float
    slope_de00: float
    backend: str


def _slope(t, y):
    ok = np.isfinite(y) & np.isfinite(t)
    if ok.sum() < 2:
        return float("nan")
    tt, yy = t[ok], y[ok]
    dt = tt - tt.mean()
    den = float((dt * dt).sum())
    if den == 0.0:
        return float("nan")
    return float((dt * (yy - yy.mean())).sum() / den)


def fp_curve(frames, warped_inputs, covis_masks, client, *, times=None, r_eff=None):
    fr = np.asarray(frames)
    wi = np.asarray(warped_inputs)
    cv = np.asarray(covis_masks)
    if fr.ndim != 4 or fr.shape[0] == 0:
        raise ValueError("frames 는 비어 있지 않은 (N,H,W,3)")
    if wi.shape != fr.shape:
        raise ValueError("warped_inputs 크기가 frames 와 다르다")
    if cv.dtype != np.bool_ or cv.shape != fr.shape[:3]:
        raise ValueError("covis_masks 는 bool (N,H,W)")
    n = fr.shape[0]
    if times is None:
        t = np.arange(n, dtype=np.float64)
    else:
        t = np.asarray(times, np.float64)
        if t.shape != (n,) or not np.isfinite(t).all():
            raise ValueError("times 는 유한한 길이 N 벡터")
    lp = np.full(n, np.nan)
    de = np.full(n, np.nan)
    cov = np.zeros(n)
    for i in range(n):
        f = as_display(fr[i], f"frames[{i}]")
        w = as_display(wi[i], f"warped_inputs[{i}]")
        m = cv[i]
        cov[i] = float(m.mean())
        if cov[i] < FP_MIN_COVIS:
            continue
        filled = np.where(m[..., None], w, f)
        lp[i] = float(client.distance(prepare(filled, r_eff=r_eff), prepare(f, r_eff=r_eff)))
        de[i] = _de00(w, f, m, PATCH, 0.5, strict=False)
    return FPCurve(t, lp, de, cov, _slope(t, lp), _slope(t, de), str(getattr(client, "backend", "unknown")))


# ------------------------------------------------------------------------------------------- OIR
@dataclass(frozen=True)
class OIRResult:
    n: int
    k: int
    rate: float
    ci_low: float
    ci_high: float
    p_value: float

    def as_dict(self):
        return {"n": self.n, "k": self.k, "rate": self.rate, "ci_low": self.ci_low,
                "ci_high": self.ci_high, "p_value": self.p_value}


def oir_counts(k, n):
    for v in (k, n):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError("k, n 은 정수")
    k, n = int(k), int(n)
    if n < 1 or k < 0 or k > n:
        raise ValueError("n >= 1, 0 <= k <= n")
    from scipy import stats

    z = float(stats.norm.ppf(0.975))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    pv = float(stats.binomtest(k, n, 0.5).pvalue)
    return OIRResult(n, k, float(p), float(lo), float(hi), pv)


def oir(correct):
    a = np.asarray(correct)
    if a.ndim != 1:
        raise ValueError("1차원이어야 한다")
    if a.size == 0:
        raise ValueError("응답이 없다")
    if a.dtype == np.bool_:
        b = a
    elif np.issubdtype(a.dtype, np.integer):
        if not np.isin(a, (0, 1)).all():
            raise ValueError("정수 응답은 0/1 만")
        b = a.astype(bool)
    else:
        raise TypeError("bool 또는 0/1 정수 응답이어야 한다")
    return oir_counts(int(b.sum()), int(b.size))


# ------------------------------------------------------------------------------------------- 목표표
@dataclass(frozen=True)
class Targets:
    ffd: Optional[float]
    ffg: Optional[float]
    ffg_sal: Optional[float]
    de00: Optional[float]
    oir: Optional[float]


TARGETS = {
    "L1_photo_green": Targets(0.05, 0.5, 0.5, 1.5, 0.55),
    "L1_ai_yellow_safe": Targets(0.06, 1.0, 0.5, 2.0, 0.60),
    "L1_ai_yellow_ext_full": Targets(0.10, 2.5, 1.0, 2.0, 0.65),
    "escher_red_safe": Targets(0.08, 1.5, 0.75, 2.5, None),
    "L2_painting_safe": Targets(0.10, 1.5, 0.75, 3.0, None),
}
_KPI_KEYS = ("ffd", "ffg", "ffg_sal", "de00", "oir")


@dataclass(frozen=True)
class TargetVerdict:
    ok: bool
    checks: dict


def evaluate_targets(metrics: Mapping, klass: str) -> TargetVerdict:
    if klass not in TARGETS:
        raise ValueError(f"알 수 없는 klass: {klass!r}")
    bad = [k for k in metrics if k not in _KPI_KEYS]
    if bad:
        raise ValueError(f"알 수 없는 지표 키: {bad}")
    tgt = TARGETS[klass]
    checks = {}
    for key in _KPI_KEYS:
        limit, val = getattr(tgt, key), metrics.get(key)
        if limit is None or val is None:
            checks[key] = None
        elif isinstance(val, float) and math.isnan(val):
            checks[key] = False
        else:
            checks[key] = bool(val <= limit)
    return TargetVerdict(all(c is not False for c in checks.values()), checks)


# ------------------------------------------------------------------------------------------- 보고서
@dataclass(frozen=True)
class FirstFrameReport:
    schema: int
    eval_only: bool
    backend: str
    is_perceptual: bool
    r_eff: Optional[int]
    ffd: float
    ffg_median: float
    ffg_p95: float
    ffg_sal_p95: Optional[float]
    de00: float
    dg: float
    oir: Optional[float]
    klass: Optional[str]
    verdict: Optional[TargetVerdict]

    def as_dict(self):
        d = asdict(self)
        d["verdict"] = None if self.verdict is None else {"ok": self.verdict.ok, "checks": dict(self.verdict.checks)}
        return d

    def to_json(self):
        return json.dumps(self.as_dict(), sort_keys=True, allow_nan=False)

    def kpi_fields(self):
        return {"ffd": self.ffd, "ffg": self.ffg_median, "de00": self.de00}


def first_frame_report(inp, f0, client, *, r_eff=None, sal_mask=None, oir_responses=None, klass=None):
    if klass is not None and klass not in TARGETS:
        raise ValueError(f"알 수 없는 klass: {klass!r}")
    a, b = _pair(inp, f0)
    d_ffd = ffd(a, b, client, r_eff=r_eff)
    g = ffg(a, b)
    sal = ffg_sal(a, b, sal_mask) if sal_mask is not None else None
    d_de = delta_e00(a, b)
    d_dg = delta_g(a, b)
    rate = oir(oir_responses).rate if oir_responses is not None else None
    verdict = None
    if klass is not None:
        verdict = evaluate_targets({"ffd": d_ffd, "ffg": g.median, "ffg_sal": sal, "de00": d_de, "oir": rate}, klass)
    return FirstFrameReport(SCHEMA_VERSION, True, str(client.backend), bool(client.is_perceptual),
                            None if r_eff is None else int(r_eff), d_ffd, g.median, g.p95, sal, d_de, d_dg,
                            rate, klass, verdict)


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        sys.exit(_worker_main(sys.argv[1:]))
    sys.exit("usage: python eval/first_frame.py --worker --backend {proxy,lpips}")
