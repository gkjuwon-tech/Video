#!/usr/bin/env python3
"""Day1 job 2: M2 LatentDown/Up-S (latent resolution lift between two VAE levels).

Down: Z[C,H,W] -> Z[C,H/2,W/2]  (= what the teacher VAE emits for the half-size image)
Up:   the inverse. Both are ~0.9M-parameter residual conv nets that only learn a correction
on top of a trivial resample (Down: area, Up: bilinear), so they start exactly at the baseline.

Run on the GPU box (torch 2.9.1 + CUDA), next to the M6 label job (Z-Image resident, ~21 GB):

    HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=1791723600 \
    PYTHONPATH=$N1_ROOT/repo:$N1_ROOT/code PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python m2_latent_lift.py --root $N1_ROOT --finish-margin-s 600 --mem-gb 2.9

Inputs (night1 layout under N1_ROOT; fetched from the HF vault when absent):
    img/{i:08d}.webp, lat/{i:08d}.npy  (float16 (C,H/8,W/8), (z-shift)*scale)
    tae/ckpt_best.pt                   M1 TAE-T v0 (frozen decoder for the image-space loss)
Outputs under N1_ROOT/day1/m2 (uploaded to HF ckpt/day1/m2/):
    zd/{i:08d}.npy        cached Enc(down2(x)) in the same normalisation (float16)
    ckpt_last.pt          resume state (both models + optimizers + step)
    ckpt_best.pt          best Down and best Up (selected independently by eval latent MSE)
    down_best.pt, up_best.pt, metrics.jsonl, samples/{down,up}_step_{s}.jpg

Markers (one JSON per line, <= 400 chars, flushed): D1_M2, D1_M2_CACHE, D1_M2_EVAL, D1_M2_DONE.
Rerun after any crash: the zd cache is skipped where valid and training resumes from ckpt_last.pt.
The HF token is read from the environment only and never printed or stored.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import random
import re
import shutil
import signal
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

MARK = "D1_M2"
DONE_FILE = "DONE.json"
MARKER_LIMIT = 400
REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
REPO_TYPE = "model"
REMOTE_DIR = "ckpt/day1/m2"
TAE_REMOTE = "ckpt/night1/tae_t/ckpt_best.pt"
CORPUS_PREFIX = "corpus/night1/"
ZIMAGE_ID = "Tongyi-MAI/Z-Image-Turbo"

PARAM_MIN = 700_000
PARAM_MAX = 1_500_000
TOTAL_STEPS = 30_000
MAX_CONSEC_ERRORS = 100         # spec: 100 failures in a row abort the run
MEM_CAP_GB = 2.9                # spec: <= 3 GB of VRAM (the GPU is shared with Z-Image)
SIDES = ("down", "up")
_UNSET = object()
_EMPTY = object()

OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

# ---------------------------------------------------------------------------
# logging / secrets
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")


def scrub(text: object, token: Optional[str] = None) -> str:
    """Remove anything that looks like an HF token from a message."""
    s = str(text)
    tok = token if token is not None else os.environ.get("HF_TOKEN")
    if tok:
        s = s.replace(tok, "***")
    return _TOKEN_RE.sub("hf_***", s)


def _round_floats(v, nd: int):
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _round_floats(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round_floats(x, nd) for x in v]
    return v


def fit_marker(marker: str, payload: dict, limit: int = MARKER_LIMIT) -> str:
    """One marker line that is always <= ``limit`` chars and free of token-like strings."""
    def render(p) -> str:
        return scrub(f"{marker} {json.dumps(p, default=str, separators=(',', ':'))}")

    line = render(payload)
    if len(line) <= limit:
        return line
    p = _round_floats(payload, 4)
    p = {k: (v[:100] if isinstance(v, str) else v) for k, v in p.items()}
    line = render(p)
    if len(line) <= limit:
        return line
    keys = list(p)
    while len(keys) > 1:
        keys.pop()
        q = {k: p[k] for k in keys}
        q["cut"] = len(p) - len(keys)
        line = render(q)
        if len(line) <= limit:
            return line
    return line[:limit]


def log(payload: dict, marker: str = MARK) -> None:
    print(fit_marker(marker, payload), flush=True)


def warn(what: str, **kw) -> None:
    d = {"warn": what}
    for k, v in kw.items():
        d[k] = scrub(v)[:200] if isinstance(v, str) else v
    log(d)


def err_str(e: BaseException, n: int = 200) -> str:
    return scrub(f"{type(e).__name__}: {e}")[:n]


def parse_deadline(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        warn("bad_deadline", value=s)
        return None
    return f if f > 0 else None


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


def area_down(z: torch.Tensor) -> torch.Tensor:
    """2x2 area average (even H and W)."""
    return F.avg_pool2d(z, 2)


def bilinear_up(z: torch.Tensor) -> torch.Tensor:
    return F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)


def _aa_bilinear_down(z: torch.Tensor) -> torch.Tensor:
    return F.interpolate(z, scale_factor=0.5, mode="bilinear", antialias=True, align_corners=False)


def _bicubic_up(z: torch.Tensor) -> torch.Tensor:
    return F.interpolate(z, scale_factor=2, mode="bicubic", align_corners=False)


# the first entry of each dict is the model's own base; the others are extra reference lines
BASELINES: Dict[str, Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {
    "down": {"area": area_down, "aa_bilinear": _aa_bilinear_down},
    "up": {"bilinear": bilinear_up, "bicubic": _bicubic_up},
}


class _Res(nn.Module):
    """x + conv3x3(silu(groupnorm(x)))."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = _gn(c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.silu(self.norm(x)))


def _check_even(z: torch.Tensor, what: str) -> None:
    if z.ndim != 4 or (what == "down" and (z.shape[-1] % 2 or z.shape[-2] % 2)):
        raise ValueError(f"{what}: bad input shape {tuple(z.shape)}")


class LatentDown(nn.Module):
    """Z[C,H,W] -> Z[C,H/2,W/2]: stem, 6 x 3x3 conv layers at ``width`` channels (layer 2 has stride 2),
    zero-initialised head, plus the area-resampled input (the model starts exactly at the baseline)."""

    def __init__(self, latent_channels: int = 16, width: int = 128):
        super().__init__()
        self.latent_channels, self.width = int(latent_channels), int(width)
        self.stem = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.l1 = _Res(width)
        self.n2 = _gn(width)
        self.l2 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.body = nn.Sequential(*[_Res(width) for _ in range(4)])       # layers 3..6
        self.nh = _gn(width)
        self.head = nn.Conv2d(width, latent_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        _check_even(z, "down")
        h = self.l1(self.stem(z))
        h = self.l2(F.silu(self.n2(h))) + F.avg_pool2d(h, 2)
        h = self.body(h)
        return area_down(z) + self.head(F.silu(self.nh(h)))


class LatentUp(nn.Module):
    """Z[C,H,W] -> Z[C,2H,2W]: stem, 6 x 3x3 conv layers (layer 2 = nearest 2x upsample, then conv),
    zero-initialised head, plus the bilinearly upsampled input."""

    def __init__(self, latent_channels: int = 16, width: int = 128):
        super().__init__()
        self.latent_channels, self.width = int(latent_channels), int(width)
        self.stem = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.l1 = _Res(width)
        self.n2 = _gn(width)
        self.l2 = nn.Conv2d(width, width, 3, padding=1)
        self.body = nn.Sequential(*[_Res(width) for _ in range(4)])       # layers 3..6
        self.nh = _gn(width)
        self.head = nn.Conv2d(width, latent_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        _check_even(z, "up")
        h = self.l1(self.stem(z))
        hu = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.l2(F.silu(self.n2(hu))) + hu
        h = self.body(h)
        return bilinear_up(z) + self.head(F.silu(self.nh(h)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_models(latent_channels: int, width: int = 128) -> Tuple[LatentDown, LatentUp]:
    return LatentDown(latent_channels, width), LatentUp(latent_channels, width)


# ---------------------------------------------------------------------------
# losses / schedule
# ---------------------------------------------------------------------------

def pick_window(h: int, w: int, win: int, rng: random.Random) -> Tuple[int, int, int, int]:
    """Random (oy, ox, wh, ww) window of at most ``win`` x ``win`` inside an h x w latent grid."""
    wh, ww = min(win, h), min(win, w)
    return rng.randint(0, h - wh), rng.randint(0, w - ww), wh, ww


def compute_losses(model: nn.Module, inp: torch.Tensor, tgt: torch.Tensor,
                   decoder: Optional[nn.Module], window: Optional[Tuple[int, int, int, int]],
                   img_weight: float = 0.1, amp: bool = False,
                   device_type: str = "cpu") -> Dict[str, torch.Tensor]:
    """latent MSE + img_weight * L1(TAE-D(pred), TAE-D(target)) on a latent window (decoder frozen)."""
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        pred = model(inp)
    pred = pred.float()
    l_lat = F.mse_loss(pred, tgt)
    if decoder is None or window is None or img_weight <= 0:
        l_img = torch.zeros((), device=pred.device)
        return {"loss": l_lat, "l_lat": l_lat.detach(), "l_img": l_img}
    oy, ox, wh, ww = window
    p = pred[:, :, oy:oy + wh, ox:ox + ww]
    t = tgt[:, :, oy:oy + wh, ox:ox + ww]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        yp = decoder(p)
        with torch.no_grad():
            yt = decoder(t)
    l_img = F.l1_loss(yp.float(), yt.float())
    return {"loss": l_lat + img_weight * l_img, "l_lat": l_lat.detach(), "l_img": l_img.detach()}


def lr_at(step: int, base_lr: float = 1e-3, warmup: int = 500, total: int = TOTAL_STEPS,
          min_frac: float = 0.01) -> float:
    """Linear warmup, then cosine over the constant total length."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


def clip_and_step(model: nn.Module, opt: torch.optim.Optimizer, clip: float = 1.0) -> bool:
    """Clip the accumulated gradients and step. False (weights untouched) when the norm is non-finite."""
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not bool(torch.isfinite(gn)):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


def psnr(y: torch.Tensor, x: torch.Tensor) -> float:
    """PSNR in dB for tensors in [-1, 1] (both clamped first)."""
    a = (y.float().clamp(-1, 1) + 1) / 2
    b = (x.float().clamp(-1, 1) + 1) / 2
    mse = float(F.mse_loss(a, b))
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


# ---------------------------------------------------------------------------
# data layout
# ---------------------------------------------------------------------------

def idx_str(i: int) -> str:
    return f"{i:08d}"


def pair_paths(root: str, i: int) -> Dict[str, str]:
    s = idx_str(i)
    return {"img": os.path.join(root, "img", f"{s}.webp"),
            "lat": os.path.join(root, "lat", f"{s}.npy")}


def zd_path(out_dir: str, i: int) -> str:
    return os.path.join(out_dir, "zd", f"{idx_str(i)}.npy")


def _indices_in(d: str, ext: str) -> set:
    out = set()
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return out
    for n in names:
        if n.endswith(ext):
            stem = n[: -len(ext)]
            if stem.isdigit():
                out.add(int(stem))
    return out


def list_pairs(root: str) -> List[int]:
    """Sorted indices having both img and lat (a ``.tmp`` never matches the extension)."""
    return sorted(_indices_in(os.path.join(root, "img"), ".webp")
                  & _indices_in(os.path.join(root, "lat"), ".npy"))


def list_zd(out_dir: str) -> set:
    return _indices_in(os.path.join(out_dir, "zd"), ".npy")


def npy_shape(path: str) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(np.load(path, mmap_mode="r").shape)
    except Exception:  # noqa: BLE001 - unreadable file == absent
        return None


def infer_latent_channels(root: str, pairs: Sequence[int]) -> int:
    for i in pairs:
        s = npy_shape(pair_paths(root, i)["lat"])
        if s is not None and len(s) == 3:
            return int(s[0])
    raise RuntimeError("cannot infer latent channels from any lat file")


def choose_eval_indices(avail: Sequence[int], n: int) -> List[int]:
    """``n`` indices evenly strided over the sorted pairs (deterministic)."""
    s = sorted(avail)
    if n <= 0 or not s:
        return []
    if n >= len(s):
        return s
    if n == 1:
        return [s[len(s) // 2]]
    return sorted({s[int(round(j * (len(s) - 1) / (n - 1)))] for j in range(n)})


def load_or_make_eval_indices(out_dir: str, avail: Sequence[int], n: int) -> List[int]:
    """Freeze the evaluation set on first use so it never drifts if the corpus changes."""
    path = os.path.join(out_dir, "eval_indices.json")
    have = set(avail)
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = [int(i) for i in json.load(f)]
        valid = [i for i in saved if i in have]
        if valid:
            if len(valid) != len(saved):
                warn("eval_indices_missing", missing=len(saved) - len(valid))
            return valid
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        pass
    idx = choose_eval_indices(avail, n)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    os.replace(tmp, path)
    return idx


def save_npy_atomic(arr: np.ndarray, path: str) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:           # file object: np.save must not append ".npy" to the temp name
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# teacher VAE wrappers (diffusers AutoencoderKL; anything with encode/decode/config works)
# ---------------------------------------------------------------------------

def vae_consts(vae) -> Tuple[float, float]:
    cfg = getattr(vae, "config", None)
    scale = float(getattr(cfg, "scaling_factor", 1.0) or 1.0)
    shift = getattr(cfg, "shift_factor", 0.0)
    return scale, (0.0 if shift is None else float(shift))


def _vae_place(vae) -> Tuple[object, object]:
    try:
        p = next(vae.parameters())
        return p.device, p.dtype
    except Exception:  # noqa: BLE001
        return torch.device("cpu"), torch.float32


def down2(x: torch.Tensor) -> torch.Tensor:
    """Antialiased 2x down-sampling by exact area averaging (``F.interpolate(mode="area")``)."""
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h // 2, w // 2), mode="area")


def encode_normalized(vae, x: torch.Tensor) -> torch.Tensor:
    """x (B,3,H,W) in [-1,1] float -> normalised latent ``(mean - shift) * scale``, float32."""
    scale, shift = vae_consts(vae)
    device, dtype = _vae_place(vae)
    with torch.inference_mode():
        z = vae.encode(x.to(device=device, dtype=dtype)).latent_dist.mean
        return ((z.float() - shift) * scale).clone()


TILE_SAMPLE_PX = 512      # diffusers' default tile is 1024 px (128 latent): a 128x128 latent never tiles
TILE_LATENT = 64          # so tile at 64 latent / 512 px; 128x128 latents then really split into tiles


def vae_tiled(vae) -> bool:
    """True once this module switched the teacher VAE to (small-tile) tiled decoding."""
    return bool(getattr(vae, "_m2_tiled", False))


def enable_small_tiles(vae) -> bool:
    """Tiled decode with 64-latent tiles (idempotent). False when the VAE cannot tile.

    The Z-Image VAE has ``sample_size=1024`` -> ``tile_latent_min_size=128`` and diffusers only tiles a
    latent whose side is *greater* than that, so the stock ``enable_tiling()`` leaves every 1024 px image
    (128x128 latent) untiled and saves no memory. Shrinking the tile makes the retry actually help.
    """
    if vae_tiled(vae):
        return True
    enable = getattr(vae, "enable_tiling", None)
    if enable is None:
        return False
    for name, val in (("tile_sample_min_size", TILE_SAMPLE_PX), ("tile_latent_min_size", TILE_LATENT),
                      ("tile_sample_min_height", TILE_SAMPLE_PX), ("tile_sample_min_width", TILE_SAMPLE_PX),
                      ("tile_latent_min_height", TILE_LATENT), ("tile_latent_min_width", TILE_LATENT)):
        if hasattr(vae, name):
            setattr(vae, name, val)
    enable()
    setattr(vae, "_m2_tiled", True)
    return True


def decode_latents(vae, z_norm: torch.Tensor) -> torch.Tensor:
    """Normalised latent (B,C,h,w) -> image (B,3,8h,8w) float32 in [-1,1]; small-tile retry on OOM."""
    scale, shift = vae_consts(vae)
    device, dtype = _vae_place(vae)

    def run() -> torch.Tensor:
        z = (z_norm.float() / scale + shift).to(device=device, dtype=dtype)
        with torch.inference_mode():
            out = vae.decode(z)
        img = out.sample if hasattr(out, "sample") else out[0]
        return img.float().clamp(-1, 1).clone()

    try:
        return run()
    except RuntimeError as e:
        if not is_oom(e):
            raise
        free_cuda_cache()
        if not enable_small_tiles(vae):
            raise
        return run()


class VaeDecode:
    """``decode_fn`` for :func:`evaluate`; exposes whether the VAE is tiled so an item is decoded uniformly."""

    def __init__(self, vae):
        self.vae = vae

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return decode_latents(self.vae, z)

    def tiled(self) -> bool:
        return vae_tiled(self.vae)


def load_u8(root: str, i: int) -> np.ndarray:
    with Image.open(pair_paths(root, i)["img"]) as im:
        return np.asarray(im.convert("RGB"))


# ---------------------------------------------------------------------------
# zd cache: Enc(down2(x)) for every corpus image
# ---------------------------------------------------------------------------

def expected_zd_shape(lat_shape: Sequence[int]) -> Optional[Tuple[int, int, int]]:
    c, h, w = lat_shape
    if h % 2 or w % 2:
        return None
    return int(c), int(h) // 2, int(w) // 2


def zd_valid(root: str, out_dir: str, i: int) -> bool:
    ls = npy_shape(pair_paths(root, i)["lat"])
    zs = npy_shape(zd_path(out_dir, i))
    if ls is None or zs is None or len(ls) != 3:
        return False
    return expected_zd_shape(ls) == zs


def _encode_images(vae, imgs: Sequence[np.ndarray], device) -> np.ndarray:
    """uint8 HWC images of one size -> float16 (B,C,H/16,W/16) cached latents."""
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device).float().div_(127.5).sub_(1.0)
    z = encode_normalized(vae, down2(x))
    return z.to(torch.float16).cpu().numpy()


def build_zd_cache(root: str, out_dir: str, indices: Sequence[int], vae, device, batch: int = 4,
                   now: Callable[[], float] = time.time,
                   should_stop: Callable[[], bool] = lambda: False,
                   progress: Optional[Callable[[dict], None]] = None,
                   progress_every_s: float = 60.0, load_workers: int = 4) -> dict:
    """Fill ``out_dir/zd`` for ``indices`` (in order). Existing valid files are kept; bad items are skipped."""
    os.makedirs(os.path.join(out_dir, "zd"), exist_ok=True)
    stats = {"built": 0, "have": 0, "bad": 0, "stopped": False, "total": len(indices)}
    todo: List[Tuple[int, Tuple[int, int, int]]] = []
    for i in indices:
        if zd_valid(root, out_dir, i):
            stats["have"] += 1
            continue
        ls = npy_shape(pair_paths(root, i)["lat"])
        if ls is None or len(ls) != 3 or expected_zd_shape(ls) is None:
            stats["bad"] += 1
            warn("zd_skip_bad_lat", index=i, shape=list(ls) if ls else None)
            continue
        todo.append((i, tuple(int(v) for v in ls)))
    if not todo:
        return stats
    if vae is None:
        stats["bad"] += len(todo)
        warn("zd_no_vae", missing=len(todo))
        return stats

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    chunks: List[List[int]] = []
    for i, ls in todo:                      # same-size items share a batch; first-come order is kept
        b = buckets.setdefault(ls, [])
        b.append(i)
        if len(b) >= batch:
            chunks.append(b)
            buckets[ls] = []
    chunks.extend(b for b in buckets.values() if b)
    chunks.sort(key=lambda c: c[0])

    last = now()
    t0 = last
    with ThreadPoolExecutor(max_workers=max(1, load_workers)) as pool:
        for chunk in chunks:
            if should_stop():
                stats["stopped"] = True
                break
            loaded: List[Tuple[int, np.ndarray]] = []
            for i, res in zip(chunk, pool.map(_try_load, [(root, j) for j in chunk])):
                if isinstance(res, Exception):
                    stats["bad"] += 1
                    warn("zd_load_failed", index=i, err=err_str(res))
                else:
                    loaded.append((i, res))
            for group in _split_for_encode(loaded, vae, device):
                for i, arr in group:
                    try:
                        _save_zd(root, out_dir, i, arr)
                        stats["built"] += 1
                    except Exception as e:  # noqa: BLE001
                        stats["bad"] += 1
                        warn("zd_save_failed", index=i, err=err_str(e))
            t = now()
            if progress is not None and t - last >= progress_every_s:
                last = t
                progress({**stats, "s": round(t - t0, 1)})
    return stats


def _try_load(args):
    root, i = args
    try:
        return load_u8(root, i)
    except Exception as e:  # noqa: BLE001
        return e


def _split_for_encode(loaded: List[Tuple[int, np.ndarray]], vae, device):
    """Encode ``loaded`` as one batch; on any failure retry item by item. Yields lists of (i, latent)."""
    if not loaded:
        return
    try:
        z = _encode_images(vae, [a for _, a in loaded], device)
        yield [(i, z[k]) for k, (i, _) in enumerate(loaded)]
        return
    except Exception as e:  # noqa: BLE001 - OOM or one odd image: go one by one
        warn("zd_batch_failed", n=len(loaded), err=err_str(e))
        free_cuda_cache()
    for i, a in loaded:
        try:
            yield [(i, _encode_images(vae, [a], device)[0])]
        except Exception as e:  # noqa: BLE001
            warn("zd_item_failed", index=i, err=err_str(e))
            free_cuda_cache()


def _save_zd(root: str, out_dir: str, i: int, z: np.ndarray) -> None:
    ls = npy_shape(pair_paths(root, i)["lat"])
    if ls is None or expected_zd_shape(ls) != tuple(z.shape):
        raise ValueError(f"zd shape {tuple(z.shape)} does not match lat {ls}")
    if not np.isfinite(z.astype(np.float32)).all():
        raise ValueError("non-finite zd")
    save_npy_atomic(np.ascontiguousarray(z, dtype=np.float16), zd_path(out_dir, i))


# ---------------------------------------------------------------------------
# training batches: aligned crops, homogeneous in shape, deterministic per batch number
# ---------------------------------------------------------------------------

def crop_shape(hd: int, wd: int, crop: int) -> Tuple[int, int]:
    """Crop size on the low-res (zd) grid: ``crop`` squared, clipped to the grid (wide images are short)."""
    return min(crop, hd), min(crop, wd)


def crop_origin(hd: int, wd: int, ch: int, cw: int, rng: random.Random) -> Tuple[int, int]:
    return rng.randint(0, hd - ch), rng.randint(0, wd - cw)


def crop_pair(z: np.ndarray, zd: np.ndarray, a: int, b: int, ch: int, cw: int
              ) -> Tuple[np.ndarray, np.ndarray]:
    """z (C,H,W), zd (C,H/2,W/2): the zd window at (a,b) of size ch x cw and the 2x window of z over it."""
    zc = z[:, 2 * a:2 * (a + ch), 2 * b:2 * (b + cw)]
    zdc = zd[:, a:a + ch, b:b + cw]
    return np.ascontiguousarray(zc, dtype=np.float16), np.ascontiguousarray(zdc, dtype=np.float16)


def load_crop(root: str, out_dir: str, i: int, latent_channels: int, a: int, b: int, ch: int, cw: int):
    z = np.load(pair_paths(root, i)["lat"], mmap_mode="r")
    zd = np.load(zd_path(out_dir, i), mmap_mode="r")
    if z.ndim != 3 or z.shape[0] != latent_channels:
        raise ValueError(f"lat shape {tuple(z.shape)}")
    if expected_zd_shape(z.shape) != tuple(zd.shape):
        raise ValueError(f"zd {tuple(zd.shape)} vs lat {tuple(z.shape)}")
    return crop_pair(z, zd, a, b, ch, cw)


class BatchSource:
    """``make(k)`` is a pure function of (seed, k): resumable and testable. Every batch holds crops of
    one shape class (1024^2 images and 1344x576 images have different crop shapes)."""

    def __init__(self, root: str, out_dir: str, indices: Sequence[int], latent_channels: int,
                 crop: int = 64, batch: int = 32, seed: int = 0):
        self.root, self.out_dir, self.C = root, out_dir, int(latent_channels)
        self.crop, self.batch, self.seed = int(crop), int(batch), int(seed)
        self.classes: Dict[Tuple[int, int], List[int]] = {}
        self.latshape: Dict[int, Tuple[int, int]] = {}
        for i in indices:
            s = npy_shape(pair_paths(root, i)["lat"])
            if s is None or len(s) != 3 or s[0] != self.C or expected_zd_shape(s) is None:
                continue
            hd, wd = s[1] // 2, s[2] // 2
            self.latshape[i] = (hd, wd)
            self.classes.setdefault(crop_shape(hd, wd, self.crop), []).append(i)
        self.keys = sorted(self.classes)
        self.weights = [len(self.classes[k]) for k in self.keys]
        self.bad: set = set()
        self._warned = 0

    def __len__(self) -> int:
        return sum(self.weights)

    def make(self, k: int):
        if not self.keys:
            return None
        rng = random.Random(f"{self.seed}-b-{k}")
        key = rng.choices(self.keys, weights=self.weights)[0]
        pool = self.classes[key]
        ch, cw = key
        zs, zds = [], []
        for _ in range(self.batch * 3):
            if len(zs) >= self.batch:
                break
            i = pool[rng.randrange(len(pool))]
            if i in self.bad:
                continue
            hd, wd = self.latshape[i]
            a, b = crop_origin(hd, wd, ch, cw, rng)
            try:
                zc, zdc = load_crop(self.root, self.out_dir, i, self.C, a, b, ch, cw)
            except Exception as e:  # noqa: BLE001 - one bad item must not stop the run
                self.bad.add(i)
                if self._warned < 20:
                    self._warned += 1
                    warn("bad_item", index=i, err=err_str(e))
                continue
            zs.append(zc)
            zds.append(zdc)
        if len(zs) < max(1, self.batch // 2):
            return None
        return torch.from_numpy(np.stack(zs)), torch.from_numpy(np.stack(zds))


class Prefetcher:
    """Background threads producing ``fn(k)`` for k = start, start+1, ... (None = failed batch)."""

    def __init__(self, fn: Callable[[int], object], start: int = 0, depth: int = 6, threads: int = 2):
        self.fn, self.q = fn, queue.Queue(maxsize=max(1, depth))
        self._halt = threading.Event()
        self._next, self._lock = int(start), threading.Lock()
        self._threads = [threading.Thread(target=self._work, daemon=True) for _ in range(max(1, threads))]
        for t in self._threads:
            t.start()

    def _work(self) -> None:
        while not self._halt.is_set():
            with self._lock:
                k = self._next
                self._next += 1
            try:
                item = self.fn(k)
            except Exception as e:  # noqa: BLE001
                warn("batch_failed", k=k, err=err_str(e))
                item = None
            while not self._halt.is_set():
                try:
                    self.q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def get(self, timeout: float = 1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return _EMPTY

    def close(self) -> None:
        self._halt.set()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

class EvalSet:
    """Frozen evaluation items, loaded once and kept in RAM as float16 (400 x ~0.7 MB)."""

    def __init__(self, root: str, out_dir: str, indices: Sequence[int], latent_channels: int):
        self.root, self.out_dir, self.C = root, out_dir, int(latent_channels)
        self.indices = list(indices)
        self._cache: Dict[int, tuple] = {}
        self._bad: set = set()

    def items(self) -> List[tuple]:
        out = []
        for i in self.indices:
            if i in self._bad:
                continue
            if i not in self._cache:
                try:
                    z = np.load(pair_paths(self.root, i)["lat"])
                    zd = np.load(zd_path(self.out_dir, i))
                    if z.ndim != 3 or z.shape[0] != self.C or expected_zd_shape(z.shape) != zd.shape:
                        raise ValueError(f"shapes lat {z.shape} zd {zd.shape}")
                    self._cache[i] = (torch.from_numpy(np.ascontiguousarray(z, dtype=np.float16)),
                                      torch.from_numpy(np.ascontiguousarray(zd, dtype=np.float16)))
                except Exception as e:  # noqa: BLE001
                    self._bad.add(i)
                    warn("bad_eval_item", index=i, err=err_str(e))
                    continue
            out.append((i,) + self._cache[i])
        return out

    def image(self, i: int) -> np.ndarray:
        return load_u8(self.root, i)


def sample_positions(n: int, k: int) -> List[int]:
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({int(round(j * (n - 1) / (k - 1))) for j in range(k)})


def _to_u8_hwc(y: torch.Tensor) -> np.ndarray:
    a = ((y.float().clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)
    return a[0].permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def _mean(v: Sequence[float]) -> float:
    return float(sum(v) / len(v)) if v else float("nan")


def summarize_side(acc: dict) -> Optional[dict]:
    """Per-direction metrics from accumulated per-image values."""
    if not acc["mse"]:
        return None
    mse = _mean(acc["mse"])
    b1, b2 = _mean(acc["b1"]), _mean(acc["b2"])
    out = {"lat_mse": mse, "base_mse": b1, "base2_mse": b2,
           "impr": 1.0 - mse / max(b1, 1e-12), "impr_best": 1.0 - mse / max(min(b1, b2), 1e-12)}
    if acc["psnr"]:
        out.update({"psnr": _mean(acc["psnr"]), "base_psnr": _mean(acc["psnr_b1"]),
                    "base2_psnr": _mean(acc["psnr_b2"]), "n_dec": len(acc["psnr"])})
        out["psnr_gain"] = out["psnr"] - out["base_psnr"]
    return out


def _decode_item(decode_fn, tgt, pred, base1, base2):
    """Decode (truth, prediction, baseline 1, baseline 2) the same way. If an OOM flips the VAE to tiled
    decoding part-way through, all four are redone tiled so the PSNR compares like with like."""
    tiled = getattr(decode_fn, "tiled", None)
    was = bool(tiled()) if callable(tiled) else None
    ys = tuple(decode_fn(t) for t in (tgt, pred, base1, base2))
    if was is False and tiled():
        del ys
        free_cuda_cache()
        ys = tuple(decode_fn(t) for t in (tgt, pred, base1, base2))
    return ys


@torch.no_grad()
def evaluate(down: nn.Module, up: nn.Module, evalset: EvalSet, device,
             decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
             amp: bool = False, decode_n: int = 40, n_figs: int = 4, budget_s: float = 240.0,
             now: Callable[[], float] = time.time):
    """Latent MSE on every eval item, teacher-decode PSNR (prediction vs ground-truth latent decoded)
    on ``decode_n`` evenly spaced items, each against the resample baselines.

    Returns (metrics | None, samples) with samples = {"down": [(orig, base, ours)], "up": [...]}.
    """
    was = (down.training, up.training)
    down.eval()
    up.eval()
    items = evalset.items()
    dec_pos = set(sample_positions(len(items), decode_n)) if decode_fn is not None else set()
    fig_pos = set()
    if dec_pos:
        order = sorted(dec_pos)
        fig_pos = {order[j] for j in sample_positions(len(order), n_figs)}
    acc = {s: {"mse": [], "b1": [], "b2": [], "psnr": [], "psnr_b1": [], "psnr_b2": []} for s in SIDES}
    samples: Dict[str, list] = {s: [] for s in SIDES}
    t0 = now()
    dev_type = torch.device(device).type
    models = {"down": down, "up": up}
    try:
        for k, (i, zt, zdt) in enumerate(items):
            z = zt[None].to(device).float()
            zd = zdt[None].to(device).float()
            for side, inp, tgt in (("down", z, zd), ("up", zd, z)):
                try:
                    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                        pred = models[side](inp)
                    pred = pred.float()
                    names = list(BASELINES[side])
                    bases = [BASELINES[side][n](inp.float()) for n in names]
                    a = acc[side]
                    a["mse"].append(float(F.mse_loss(pred, tgt)))
                    a["b1"].append(float(F.mse_loss(bases[0], tgt)))
                    a["b2"].append(float(F.mse_loss(bases[1], tgt)))
                    if k in dec_pos and now() - t0 <= budget_s:
                        g, yo, yb1, yb2 = _decode_item(decode_fn, tgt, pred, bases[0], bases[1])
                        a["psnr"].append(psnr(yo, g))
                        a["psnr_b1"].append(psnr(yb1, g))
                        a["psnr_b2"].append(psnr(yb2, g))
                        if k in fig_pos:
                            samples[side].append((i, _to_u8_hwc(yb1), _to_u8_hwc(yo)))
                        del g, yo, yb1, yb2
                except Exception as e:  # noqa: BLE001 - skip this item/side
                    warn("eval_item_failed", index=i, side=side, err=err_str(e))
                    free_cuda_cache()
    finally:
        down.train(was[0])
        up.train(was[1])
    metrics = {"n": len(items)}
    for side in SIDES:
        m = summarize_side(acc[side])
        if m is not None:
            metrics[side] = m
    # originals for the comparison figure (lazy: only for the rows we kept)
    for side in SIDES:
        rows = []
        for i, base_u8, ours_u8 in samples[side]:
            try:
                x = evalset.image(i)
                if side == "down":
                    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
                    x = x[:h, :w].astype(np.float32).reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
                    x = x.round().clip(0, 255).astype(np.uint8)
                if x.shape != ours_u8.shape:
                    x = x[:ours_u8.shape[0], :ours_u8.shape[1]]
                rows.append((x, base_u8, ours_u8))
            except Exception as e:  # noqa: BLE001
                warn("figure_row_failed", index=i, err=err_str(e))
        samples[side] = rows
    if len(metrics) <= 1:
        return None, samples
    return metrics, samples


def make_compare_grid(rows: Sequence[tuple], crop_px: int = 256) -> Image.Image:
    """Rows of (original | baseline resample decoded | ours decoded): centre crops at native pixels."""
    head = 18
    labels = ("original", "baseline resample", "ours")
    sizes = [(min(crop_px, r[0].shape[0]), min(crop_px, r[0].shape[1])) for r in rows] or [(crop_px, crop_px)]
    ch = max(s[0] for s in sizes)
    cw = max(s[1] for s in sizes)
    canvas = Image.new("RGB", (3 * cw, head + max(1, len(rows)) * ch), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(labels):
        draw.text((c * cw + 4, 3), name, fill=(255, 255, 255))
    for r, trio in enumerate(rows):
        for c, arr in enumerate(trio):
            h, w = arr.shape[:2]
            y0, x0 = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
            tile = np.ascontiguousarray(arr[y0:y0 + ch, x0:x0 + cw])
            canvas.paste(Image.fromarray(tile), (c * cw, head + r * ch))
    return canvas


# ---------------------------------------------------------------------------
# checkpoints (atomic) and upload
# ---------------------------------------------------------------------------

def save_atomic(obj: dict, path: str) -> None:
    """Write to a temp file, fsync, then rename over ``path``."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_ckpt(path: str, need: Sequence[str] = ("step",)) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - our own file; retry permissive before giving up
            obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or any(k not in obj for k in need):
            raise ValueError("missing keys")
        return obj
    except Exception as e:  # noqa: BLE001
        warn("ckpt_unreadable", path=os.path.basename(path), err=err_str(e))
        return None


def clean_tmp(out_dir: str) -> int:
    n = 0
    for d in (out_dir, os.path.join(out_dir, "samples"), os.path.join(out_dir, "zd")):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".tmp") or name.endswith(".part"):
                try:
                    os.remove(os.path.join(d, name))
                    n += 1
                except OSError:
                    pass
    return n


def latest_sample(samples_dir: str, side: str) -> Optional[str]:
    best, best_step = None, -1
    try:
        names = os.listdir(samples_dir)
    except FileNotFoundError:
        return None
    for n in names:
        m = re.fullmatch(rf"{side}_step_(\d+)\.jpg", n)
        if m and int(m.group(1)) > best_step:
            best, best_step = n, int(m.group(1))
    return os.path.join(samples_dir, best) if best else None


def stage_upload(out_dir: str) -> Tuple[str, List[str]]:
    """Copy the files to publish into one folder so a single commit uploads them."""
    stage = os.path.join(out_dir, "upload_stage")
    os.makedirs(stage, exist_ok=True)
    srcs = [(os.path.join(out_dir, n), n) for n in
            ("ckpt_last.pt", "ckpt_best.pt", "down_best.pt", "up_best.pt", "metrics.jsonl")]
    for side in SIDES:
        ls = latest_sample(os.path.join(out_dir, "samples"), side)
        if ls:
            srcs.append((ls, f"{side}_latest.jpg"))
    names = []
    for src, name in srcs:
        if not os.path.isfile(src):
            continue
        dst = os.path.join(stage, name)
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
        names.append(name)
    return stage, names


def make_uploader(token: Optional[str], repo_id: str = REPO_ID) -> Optional[Callable[[str, str], None]]:
    """``upload(folder, remote_dir)`` using huggingface_hub (imported lazily), or None without a token."""
    if not token:
        return None

    def upload(folder: str, remote_dir: str) -> None:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_folder(
            folder_path=folder, path_in_repo=remote_dir, repo_id=repo_id, repo_type=REPO_TYPE,
            commit_message="day1 m2 latent lift checkpoint")

    return upload


def upload_with_retry(uploader: Callable[[str, str], None], folder: str, remote: str,
                      tries: int = 3, sleep: Callable[[float], None] = time.sleep) -> bool:
    for k in range(tries):
        try:
            uploader(folder, remote)
            return True
        except Exception as e:  # noqa: BLE001 - upload trouble is a warning, never fatal
            warn("upload_failed", attempt=k + 1, err=err_str(e, 300))
            if k + 1 < tries:
                sleep(5 * (3 ** k))
    return False


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def vram_gb() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def free_cuda_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def is_oom(e: BaseException) -> bool:
    """CUDA OOM, including the cuDNN/cuBLAS workspace failures that surface as plain RuntimeError."""
    if isinstance(e, OOM_ERROR):
        return True
    m = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in m or "alloc_failed" in m)


def mem_fraction(mem_gb: float, total_bytes: int) -> float:
    if mem_gb <= 0 or total_bytes <= 0:
        return 1.0
    return min(1.0, (mem_gb * 2**30) / float(total_bytes))


def apply_cuda_limits(device: torch.device, mem_gb: float) -> Optional[float]:
    """Cap this process's caching allocator (the GPU is shared with Z-Image). Returns the cap in GB."""
    if device.type != "cuda" or mem_gb <= 0:
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory
        frac = mem_fraction(mem_gb, total)
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        return round(frac * total / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        warn("mem_cap_failed", err=err_str(e, 300))
        return None


# ---------------------------------------------------------------------------
# assets: night1 corpus + M1 TAE-T checkpoint (HF vault fallback), teacher VAE, TAE decoder
# ---------------------------------------------------------------------------

_SHARD_MEMBER = re.compile(r"(img/\d{8}\.webp|lat/\d{8}\.npy|meta/\d{8}\.json)")


def extract_shard(tar_path: str, root: str) -> int:
    """Extract img/lat/meta members of a corpus shard into ``root`` (existing files kept). Returns new files."""
    n = 0
    with tarfile.open(tar_path, "r") as t:
        for m in t:
            if not m.isfile() or not _SHARD_MEMBER.fullmatch(m.name):
                continue
            dst = os.path.join(root, m.name)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = t.extractfile(m)
            if src is None:
                continue
            tmp = dst + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f)
                os.replace(tmp, dst)
                n += 1
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    return n


def make_hf_fetchers(token: Optional[str], root: str, repo_id: str = REPO_ID):
    """(lister(prefix) -> [remote names], fetcher(remote) -> local path); (None, None) without a token."""
    if not token:
        return None, None

    def lister(prefix: str) -> List[str]:
        from huggingface_hub import HfApi
        return [n for n in HfApi(token=token).list_repo_files(repo_id, repo_type=REPO_TYPE)
                if n.startswith(prefix)]

    def fetcher(remote: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id, remote, repo_type=REPO_TYPE, token=token,
                               local_dir=os.path.join(root, "_hf_dl"))

    return lister, fetcher


def ensure_assets(root: str, tae_path: str, min_pairs: int, lister, fetcher) -> dict:
    """Fetch the M1 checkpoint and the corpus shards from the vault when the box does not have them."""
    res = {"tae": os.path.isfile(tae_path), "pairs": len(list_pairs(root)), "shards": 0}
    if not res["tae"] and fetcher is not None:
        try:
            src = fetcher(TAE_REMOTE)
            os.makedirs(os.path.dirname(tae_path), exist_ok=True)
            shutil.copyfile(src, tae_path + ".part")
            os.replace(tae_path + ".part", tae_path)
            res["tae"] = True
        except Exception as e:  # noqa: BLE001
            warn("tae_fetch_failed", err=err_str(e, 300))
    if res["pairs"] < min_pairs and lister is not None and fetcher is not None:
        try:
            names = sorted(n for n in lister(CORPUS_PREFIX)
                           if re.fullmatch(re.escape(CORPUS_PREFIX) + r"gen_\d{4,}\.tar", n))
        except Exception as e:  # noqa: BLE001
            warn("corpus_list_failed", err=err_str(e, 300))
            names = []
        for name in names:
            try:
                p = fetcher(name)
                added = extract_shard(p, root)
                res["shards"] += 1
                try:
                    os.remove(p)
                except OSError:
                    pass
                log({"shard": os.path.basename(name), "new_files": added}, marker=MARK + "_CACHE")
            except Exception as e:  # noqa: BLE001
                warn("shard_fetch_failed", name=os.path.basename(name), err=err_str(e, 300))
        res["pairs"] = len(list_pairs(root))
    return res


def import_tae_module(extra_dirs: Sequence[str] = ()):
    """The night1 ``tae_train`` module (model classes), found next to us, in night1/, or under N1_ROOT."""
    mod = sys.modules.get("tae_train")
    if mod is not None and hasattr(mod, "TAE"):
        return mod
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("N1_ROOT", "/workspace/night1")
    cands = list(extra_dirs) + [here, os.path.join(here, "..", "night1"), os.path.join(root, "code"),
                                os.path.join(root, "repo", "ops", "gpu", "night1")]
    for d in cands:
        p = os.path.join(d, "tae_train.py")
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("tae_train", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["tae_train"] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop("tae_train", None)
                raise
            return mod
    try:
        import tae_train  # noqa: F401  (PYTHONPATH)
        return sys.modules["tae_train"]
    except ImportError as e:
        raise ImportError("tae_train.py not found (put it next to this file or on PYTHONPATH)") from e


def load_tae_decoder(path: str, device) -> Tuple[nn.Module, int]:
    """Frozen M1 TAE-T decoder from ``ckpt_best.pt``. Returns (decoder, latent_channels)."""
    ck = load_ckpt(path, need=("model", "config"))
    if ck is None:
        raise FileNotFoundError(f"no usable TAE checkpoint at {path}")
    mod = import_tae_module()
    cfg = ck["config"]
    tae = mod.TAE(int(cfg["latent_channels"]), tuple(int(w) for w in cfg["widths"]),
                  int(cfg.get("blocks", getattr(mod, "BLOCKS_PER_STAGE", 2))))
    tae.load_state_dict(ck["model"])
    dec = tae.dec.to(device).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec, int(cfg["latent_channels"])


def load_teacher_vae(device, vae_path: Optional[str] = None):
    """diffusers AutoencoderKL from the Z-Image snapshot's ``vae`` subfolder, bf16 (lazy imports)."""
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download

    path = vae_path or snapshot_download(ZIMAGE_ID, allow_patterns=["vae/*"])
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class Trainer:
    def __init__(self, args, uploader=_UNSET, now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 stop_event: Optional[threading.Event] = None, deps: Optional[dict] = None):
        self.a = args
        self.deps = deps or {}
        self.now, self.sleep = now, sleep
        self.stop = stop_event or threading.Event()
        self.root = args.root
        self.out = args.out or os.path.join(self.root, "day1", "m2")
        self.samples_dir = os.path.join(self.out, "samples")
        self.deadline = parse_deadline(args.deadline)
        self.uploader = uploader
        self.upload_thread: Optional[threading.Thread] = None
        self.step = 0
        self.skipped = 0
        self.bad_run = 0
        self.fail_streak = 0
        self.down: Optional[LatentDown] = None
        self.up: Optional[LatentUp] = None
        self.opt: Dict[str, torch.optim.Optimizer] = {}
        self.best = {s: math.inf for s in SIDES}                   # best eval latent MSE per model
        self.best_state: Dict[str, Optional[dict]] = {s: None for s in SIDES}
        self.vae = None
        self.decoder: Optional[nn.Module] = None
        self.evalset: Optional[EvalSet] = None
        self.device = None
        self.amp = False
        self.C = 0
        self.last_loss = {s: float("nan") for s in SIDES}
        self.micro_batch = int(args.micro_batch or args.batch)
        self.mem_cap_gb: Optional[float] = None
        self.first_step_logged = False
        self.acc: Optional[torch.Tensor] = None
        self.acc_n = 0
        self.samples_seen = 0
        self.log_t = 0.0

    # -- helpers ----------------------------------------------------------
    def _stop_reason(self) -> Optional[str]:
        if self.stop.is_set():
            return "signal"
        if self.deadline is not None and self.now() >= self.deadline - self.a.finish_margin_s:
            return "deadline"
        if self.a.max_steps is not None and self.step >= self.a.max_steps:
            return "max_steps"
        if self.step >= self.a.total_steps:
            return "total_steps"
        return None

    def _config(self) -> dict:
        return {"latent_channels": self.C, "width": self.a.width}

    def _ckpt_obj(self) -> dict:
        return {"version": 1, "step": self.step, "config": self._config(),
                "down": self.down.state_dict(), "up": self.up.state_dict(),
                "opt_down": self.opt["down"].state_dict(), "opt_up": self.opt["up"].state_dict(),
                "best": dict(self.best), "micro_batch": self.micro_batch, "saved_at": time.time()}

    def _best_obj(self) -> dict:
        obj = {"version": 1, "step": self.step, "config": self._config()}
        for s in SIDES:
            if self.best_state[s] is not None:
                obj[s] = {"model": self.best_state[s]["model"], "step": self.best_state[s]["step"],
                          "lat_mse": self.best[s]}
        return obj

    def _load_resume(self) -> Optional[dict]:
        ck = load_ckpt(os.path.join(self.out, "ckpt_last.pt"), need=("step", "down", "up", "config"))
        if ck is None:
            return None
        cfg = ck["config"]
        if int(cfg.get("latent_channels", -1)) != self.C or int(cfg.get("width", -1)) != self.a.width:
            warn("ckpt_config_mismatch", ckpt=cfg, want=self._config())
            return None
        return ck

    def _restore(self, ck: dict) -> bool:
        try:
            self.down.load_state_dict(ck["down"])
            self.up.load_state_dict(ck["up"])
            for s, key in (("down", "opt_down"), ("up", "opt_up")):
                if key in ck:
                    self.opt[s].load_state_dict(ck[key])
            self.step = int(ck["step"])
            for s in SIDES:
                v = (ck.get("best") or {}).get(s)
                if v is not None:
                    self.best[s] = float(v)
            mb = ck.get("micro_batch")
            if mb and not self.a.micro_batch:
                self.micro_batch = max(1, min(self.micro_batch, int(mb)))   # never grow back past an OOM size
            return True
        except Exception as e:  # noqa: BLE001
            warn("ckpt_restore_failed", err=err_str(e))
            return False

    def _restore_best(self) -> None:
        ck = load_ckpt(os.path.join(self.out, "ckpt_best.pt"), need=("config",))
        if ck is None:
            return
        cfg = ck["config"]
        if int(cfg.get("latent_channels", -1)) != self.C or int(cfg.get("width", -1)) != self.a.width:
            return
        for s in SIDES:
            if s in ck and "model" in ck[s]:
                self.best_state[s] = {"model": ck[s]["model"], "step": int(ck[s].get("step", 0))}
                self.best[s] = min(self.best[s], float(ck[s].get("lat_mse", math.inf)))

    # -- checkpoint + eval + upload ------------------------------------------
    def _evaluate(self, decode: bool = True):
        decode_fn = None
        if decode and self.vae is not None and self.a.decode_n > 0:
            decode_fn = VaeDecode(self.vae)
        try:
            return evaluate(self.down, self.up, self.evalset, self.device, decode_fn, self.amp,
                            self.a.decode_n, 4, self.a.eval_budget_s, self.now)
        finally:
            free_cuda_cache()

    def _save_last(self) -> None:
        try:
            save_atomic(self._ckpt_obj(), os.path.join(self.out, "ckpt_last.pt"))
        except Exception as e:  # noqa: BLE001
            warn("ckpt_save_failed", err=err_str(e, 300))

    def checkpoint_and_eval(self, final: bool = False, reason: Optional[str] = None) -> Optional[dict]:
        metrics = None
        improved = {s: False for s in SIDES}
        self._save_last()          # before the eval: a kill or CUDA fault during the eval must not lose the steps
        try:
            metrics, samples = self._evaluate(decode=reason != "signal")   # SIGTERM: no slow decodes
            if metrics is not None:
                for s in SIDES:
                    m = metrics.get(s)
                    if m is None:
                        continue
                    self.last_loss[s] = m["lat_mse"]
                    if m["lat_mse"] < self.best[s]:
                        self.best[s] = m["lat_mse"]
                        improved[s] = True
                        model = self.down if s == "down" else self.up
                        self.best_state[s] = {"model": cpu_state(model), "step": self.step}
                    rows = samples.get(s) or []
                    if rows:
                        try:
                            path = os.path.join(self.samples_dir, f"{s}_step_{self.step}.jpg")
                            make_compare_grid(rows).save(path + ".tmp", format="JPEG", quality=90)
                            os.replace(path + ".tmp", path)
                        except Exception as e:  # noqa: BLE001
                            warn("sample_save_failed", err=err_str(e))
                rec = {"step": self.step, "time": round(time.time(), 1), "final": final,
                       "improved": improved, "best": dict(self.best), **metrics}
                with open(os.path.join(self.out, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(_round_floats(rec, 6)) + "\n")
                log(self._eval_marker(metrics, improved, final), marker=MARK + "_EVAL")
        except Exception as e:  # noqa: BLE001 - eval trouble must not lose the checkpoint
            warn("eval_failed", err=err_str(e, 300))
        try:
            save_atomic(self._ckpt_obj(), os.path.join(self.out, "ckpt_last.pt"))
            if any(improved.values()) or (any(self.best_state[s] is not None for s in SIDES)
                                          and not os.path.isfile(os.path.join(self.out, "ckpt_best.pt"))):
                save_atomic(self._best_obj(), os.path.join(self.out, "ckpt_best.pt"))
                for s in SIDES:
                    if improved[s] and self.best_state[s] is not None:
                        save_atomic({"version": 1, "model": self.best_state[s]["model"],
                                     "config": self._config(), "step": self.best_state[s]["step"],
                                     "lat_mse": self.best[s], "kind": s},
                                    os.path.join(self.out, f"{s}_best.pt"))
        except Exception as e:  # noqa: BLE001
            warn("ckpt_save_failed", err=err_str(e, 300))
        self._start_upload(final)
        return metrics

    def _eval_marker(self, m: dict, improved: dict, final: bool) -> dict:
        out = {"step": self.step, "n": m["n"], "final": final}
        for s, p in (("down", "d"), ("up", "u")):
            v = m.get(s)
            if not v:
                continue
            out.update({f"{p}_mse": v["lat_mse"], f"{p}_base": v["base_mse"], f"{p}_impr": v["impr"],
                        f"{p}_best": v["impr_best"]})
            if "psnr" in v:
                out.update({f"{p}_psnr": v["psnr"], f"{p}_pbase": v["base_psnr"], f"{p}_gain": v["psnr_gain"]})
            if improved[s]:
                out[f"{p}_new_best"] = True
        if "down" in m and "n_dec" in m["down"]:
            out["n_dec"] = m["down"]["n_dec"]
        return out

    def _upload_job(self, stage: str) -> None:
        if upload_with_retry(self.uploader, stage, REMOTE_DIR, sleep=self.sleep):
            log({"uploaded": REMOTE_DIR, "step": self.step})

    def _start_upload(self, final: bool) -> None:
        if self.uploader is None:
            return
        th = self.upload_thread
        if th is not None and th.is_alive():
            if not final:
                warn("upload_still_running_skip")
                return
            th.join(timeout=self.a.final_upload_timeout_s)
        try:
            stage, _ = stage_upload(self.out)
        except Exception as e:  # noqa: BLE001
            warn("stage_failed", err=err_str(e, 300))
            return
        self.upload_thread = threading.Thread(target=self._upload_job, args=(stage,), daemon=True)
        self.upload_thread.start()

    def _finish_upload(self) -> None:
        th = self.upload_thread
        if th is not None and th.is_alive():
            th.join(timeout=self.a.final_upload_timeout_s)
            if th.is_alive():
                warn("final_upload_timeout")

    # -- training ----------------------------------------------------------
    def _accumulate(self, zc: torch.Tensor, zdc: torch.Tensor, micro: int) -> torch.Tensor:
        """Backward over micro-batches (gradient accumulation). Returns the batch means of
        [loss_down, lat_down, img_down, loss_up, lat_up, img_up]."""
        for o in self.opt.values():
            o.zero_grad(set_to_none=True)
        n = zc.shape[0]
        wrng = random.Random(f"{self.a.seed}-w-{self.step}")
        tot = None
        dt = self.device.type
        for s in range(0, n, micro):
            zs, zds = zc[s:s + micro], zdc[s:s + micro]
            w = zs.shape[0] / n
            win_d = pick_window(zds.shape[2], zds.shape[3], self.a.img_win, wrng)      # on the Down output grid
            win_u = pick_window(zs.shape[2], zs.shape[3], self.a.img_win, wrng)        # on the Up output grid
            ld = compute_losses(self.down, zs, zds, self.decoder, win_d, self.a.img_weight, self.amp, dt)
            lu = compute_losses(self.up, zds, zs, self.decoder, win_u, self.a.img_weight, self.amp, dt)
            ((ld["loss"] + lu["loss"]) * w).backward()
            part = torch.stack([ld["loss"].detach(), ld["l_lat"], ld["l_img"],
                                lu["loss"].detach(), lu["l_lat"], lu["l_img"]]) * w
            tot = part if tot is None else tot + part
            del ld, lu
        return tot

    def _shrink_micro_batch(self, n: int) -> bool:
        cur = max(1, min(self.micro_batch, n))
        if cur <= 1:
            return False
        self.micro_batch = max(1, cur // 2)
        free_cuda_cache()
        warn("oom_backoff", step=self.step, micro_batch=self.micro_batch,
             accum_steps=-(-n // self.micro_batch), vram_GB=vram_gb())
        return True

    def _train_step(self, batch) -> bool:
        zb, zdb = batch
        zc = zb.to(self.device, non_blocking=True).float()
        zdc = zdb.to(self.device, non_blocking=True).float()
        lr = lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps)
        for o in self.opt.values():
            for g in o.param_groups:
                g["lr"] = lr
        while True:
            parts, oom = None, False
            try:
                parts = self._accumulate(zc, zdc, max(1, min(self.micro_batch, zc.shape[0])))
            except RuntimeError as e:        # torch.OutOfMemoryError is a RuntimeError
                if not is_oom(e):
                    raise
                oom = True
            if not oom:
                break
            for o in self.opt.values():
                o.zero_grad(set_to_none=True)
            if not self._shrink_micro_batch(zc.shape[0]):
                raise OOM_ERROR("out of memory at micro_batch=1")
        ok_d = clip_and_step(self.down, self.opt["down"], 1.0)
        ok_u = clip_and_step(self.up, self.opt["up"], 1.0)
        self.skipped += (not ok_d) + (not ok_u)
        if not (ok_d or ok_u):
            self.bad_run += 1
            return False
        self.bad_run = 0
        self.acc += parts
        self.acc_n += 1
        self.samples_seen += zc.shape[0]
        self.step += 1
        return True

    def _log_progress(self, t: float) -> None:
        dt = max(1e-6, t - self.log_t)
        if self.acc_n:
            v = [x / self.acc_n for x in self.acc.tolist()]
            self.last_loss.update({"down": v[1], "up": v[4]})
        else:
            v = [float("nan")] * 6
        log({"step": self.step, "ld": v[0], "ld_lat": v[1], "ld_img": v[2],
             "lu": v[3], "lu_lat": v[4], "lu_img": v[5],
             "ips": round(self.samples_seen / dt, 1),
             "lr": lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps),
             "skip": self.skipped, "vram_GB": vram_gb(), "mb": self.micro_batch})
        self.acc.zero_()
        self.acc_n = 0
        self.samples_seen = 0
        self.log_t = t

    def _setup_device(self) -> None:
        a = self.a
        dev_name = self.deps.get("device") or a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(dev_name).startswith("cuda") and not (os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                                                     or os.environ.get("PYTORCH_ALLOC_CONF")):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before first CUDA use
        self.device = torch.device(dev_name)
        self.amp = {"on": True, "off": False}.get(a.amp, self.device.type == "cuda")
        if self.device.type == "cuda":
            self.mem_cap_gb = apply_cuda_limits(self.device, a.mem_gb)
        torch.manual_seed(a.seed)

    def _load_teachers(self, need_vae: bool) -> None:
        """TAE-T decoder (image-space loss) and teacher VAE; each one degrades gracefully when missing."""
        a = self.a
        dec = self.deps.get("decoder", _UNSET)
        if dec is _UNSET:
            path = a.tae_ckpt or os.path.join(self.root, "tae", "ckpt_best.pt")
            try:
                dec, c = load_tae_decoder(path, self.device)
                if c != self.C:
                    warn("tae_channels_mismatch", tae=c, data=self.C)
                    dec = None
            except Exception as e:  # noqa: BLE001
                warn("tae_unavailable_img_loss_off", err=err_str(e, 300))
                dec = None
        elif dec is not None:
            dec = dec.to(self.device).eval()
            for p in dec.parameters():
                p.requires_grad_(False)
        self.decoder = dec
        vae = self.deps.get("vae", _UNSET)
        if vae is _UNSET:
            vae = None
            if need_vae:
                try:
                    vae = load_teacher_vae(self.device, a.vae_path)
                except Exception as e:  # noqa: BLE001
                    warn("vae_unavailable", err=err_str(e, 300))
        self.vae = vae

    def _target_step(self) -> int:
        t = self.a.total_steps
        return t if self.a.max_steps is None else min(t, self.a.max_steps)

    def _already_done(self) -> Optional[int]:
        """Step of a finished run (DONE.json + ckpt_last.pt at/after the target), else None."""
        try:
            if not os.path.isfile(os.path.join(self.out, DONE_FILE)):
                return None
            ck = load_ckpt(os.path.join(self.out, "ckpt_last.pt"), need=("step", "config"))
            if ck is None or int(ck["step"]) < self._target_step():
                return None
            cfg = ck["config"]
            if int(cfg.get("width", -1)) != self.a.width:
                return None                    # other model size: a fresh run, not a relaunch
            if self.a.latent_channels and int(cfg.get("latent_channels", -1)) != self.a.latent_channels:
                return None
            return int(ck["step"])
        except Exception as e:  # noqa: BLE001
            warn("done_check_failed", err=err_str(e))
        return None

    def _final_row_exists(self, step: int) -> bool:
        try:
            with open(os.path.join(self.out, "metrics.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and r.get("final") and r.get("step") == step:
                        return True
        except OSError:
            pass
        return False

    def _write_done(self, summary: dict) -> None:
        path = os.path.join(self.out, DONE_FILE)
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(summary, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(path + ".tmp", path)
        except Exception as e:  # noqa: BLE001
            warn("done_write_failed", err=err_str(e))

    def _probe_vae_tiling(self) -> Optional[bool]:
        """Decide tiled vs untiled decoding once, before any eval: trial-decode one 128x128 latent
        (every 1024 px image). An OOM switches the VAE to small tiles for the whole run."""
        if self.vae is None or self.a.decode_n <= 0 or self.C <= 0:
            return None
        try:
            decode_latents(self.vae, torch.zeros(1, self.C, 128, 128))
        except Exception as e:  # noqa: BLE001
            warn("vae_probe_failed", err=err_str(e, 300))
        finally:
            free_cuda_cache()
        return vae_tiled(self.vae)

    def run(self) -> dict:
        a = self.a
        os.makedirs(self.samples_dir, exist_ok=True)
        done_step = self._already_done()
        if done_step is not None:      # relaunched by a keep-alive loop after a normal finish: do nothing
            log({"status": "already_done", "step": done_step}, marker=MARK + "_DONE")
            return {"status": "already_done", "step": done_step}
        os.makedirs(os.path.join(self.out, "zd"), exist_ok=True)
        clean_tmp(self.out)
        token = os.environ.get("HF_TOKEN")
        if self.uploader is _UNSET:
            self.uploader = None if a.no_upload else make_uploader(token, a.repo_id)
        log({"start": True, "root": self.root, "hf_token": "yes" if token else "no",
             "upload": bool(self.uploader), "deadline": self.deadline, "torch": torch.__version__})

        tae_path = a.tae_ckpt or os.path.join(self.root, "tae", "ckpt_best.pt")
        if not a.no_fetch:
            lister, fetcher = self.deps.get("fetchers") or make_hf_fetchers(token, self.root, a.repo_id)
            if lister is not None and (not os.path.isfile(tae_path) or len(list_pairs(self.root)) < a.min_pairs):
                res = ensure_assets(self.root, tae_path, a.min_pairs, lister, fetcher)
                log({"assets": res})
        pairs = list_pairs(self.root)
        if len(pairs) < max(2, a.min_train):
            log({"fatal": "no_data", "pairs": len(pairs)}, marker=MARK + "_DONE")
            return {"status": "no_data", "step": 0}

        self._setup_device()
        self.C = a.latent_channels or infer_latent_channels(self.root, pairs)
        n_eval = min(a.eval_n, len(pairs) // 2)
        eval_idx = load_or_make_eval_indices(self.out, pairs, n_eval)
        eval_set = set(eval_idx)
        have_zd = list_zd(self.out)
        need_vae = any(not zd_valid(self.root, self.out, i) for i in pairs) or a.decode_n > 0
        self._load_teachers(need_vae)
        vae_tiled_at_load = self._probe_vae_tiling()

        # 1) zd cache: evaluation items first, then the rest
        order = eval_idx + [i for i in pairs if i not in eval_set]
        stats = build_zd_cache(
            self.root, self.out, order, self.vae, self.device, a.cache_batch, self.now,
            should_stop=lambda: self._stop_reason() is not None,
            progress=lambda s: log({"cache": s}, marker=MARK + "_CACHE"))
        log({"cache_done": stats, "had": len(have_zd)}, marker=MARK + "_CACHE")
        free_cuda_cache()

        usable = [i for i in pairs if zd_valid(self.root, self.out, i)]
        train_idx = [i for i in usable if i not in eval_set]
        if len(train_idx) < 1:
            reason = self._stop_reason() or "no_train_data"
            log({"status": reason, "usable": len(usable), "cache": stats}, marker=MARK + "_DONE")
            return {"status": reason, "step": 0}

        self.down, self.up = (m.to(self.device) for m in build_models(self.C, a.width))
        self.opt = {s: torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
                    for s, m in (("down", self.down), ("up", self.up))}
        n_d, n_u = count_params(self.down), count_params(self.up)
        ck = self._load_resume()
        resumed = ck is not None and self._restore(ck)
        self._restore_best()
        start_step = self.step
        self.evalset = EvalSet(self.root, self.out, eval_idx, self.C)
        src = BatchSource(self.root, self.out, train_idx, self.C, a.crop, a.batch, a.seed)
        log({"vae_tiled": vae_tiled_at_load, "params_down": n_d, "params_up": n_u,
             "in_range": PARAM_MIN <= n_d <= PARAM_MAX and PARAM_MIN <= n_u <= PARAM_MAX,
             "C": self.C, "width": a.width, "dev": str(self.device), "amp": self.amp,
             "resumed": bool(resumed), "step": self.step, "img_loss": self.decoder is not None,
             "vae": self.vae is not None, "mb": self.micro_batch, "batch": a.batch,
             "cap_GB": self.mem_cap_gb, "train": len(src), "eval": len(eval_idx),
             "classes": [list(k) for k in src.keys]})

        self.acc = torch.zeros(6, device=self.device)
        t0 = self.now()
        self.log_t = t0
        next_log = t0 + a.log_every_s
        next_eval = t0 + min(a.first_eval_s, a.eval_every_s)
        reason: Optional[str] = None
        status_extra: dict = {}
        pf = Prefetcher(src.make, self.step, a.prefetch_depth, a.prefetch_threads)
        self.down.train()
        self.up.train()
        try:
            while True:
                reason = self._stop_reason()
                if reason:
                    break
                batch = pf.get(1.0)
                if batch is _EMPTY:
                    continue
                if batch is None:
                    self.fail_streak += 1
                    warn("empty_batch", n=self.fail_streak)
                    if self.fail_streak >= a.max_consec_errors:
                        reason = "fatal"
                        status_extra["fatal"] = "no_batches"
                        break
                    continue
                try:
                    if self._train_step(batch):
                        self.fail_streak = 0
                    else:
                        self.fail_streak += 1
                    if not self.first_step_logged and self.step > start_step:
                        self.first_step_logged = True
                        log({"first_step": self.step, "vram_GB": vram_gb(), "mb": self.micro_batch})
                    if self.bad_run >= 25:
                        warn("nonfinite_streak", step=self.step)
                        ck2 = self._load_resume()
                        if ck2 is not None:
                            self._restore(ck2)
                        self.bad_run = 0
                except Exception as e:  # noqa: BLE001 - OOM or anything else: record, skip the step
                    self.fail_streak += 1
                    warn("step_failed", step=self.step, n=self.fail_streak, err=err_str(e))
                    for o in self.opt.values():
                        o.zero_grad(set_to_none=True)
                    free_cuda_cache()
                    if self.fail_streak < a.max_consec_errors:
                        self.sleep(min(30.0, float(self.fail_streak)))
                if self.fail_streak >= a.max_consec_errors:
                    reason = "fatal"
                    status_extra["fatal"] = "too_many_step_errors"
                    break
                t = self.now()
                if t >= next_log:
                    self._log_progress(t)
                    next_log = t + a.log_every_s
                if t >= next_eval:
                    self.checkpoint_and_eval()
                    next_eval = self.now() + a.eval_every_s
        finally:
            pf.close()

        if self.acc_n:
            self._log_progress(self.now())
        if self.step == start_step and self._final_row_exists(self.step):
            log({"final_skipped": "no_progress", "step": self.step})      # relaunch that trained nothing
        else:
            self.checkpoint_and_eval(final=True, reason=reason)    # last save + eval + upload, also after a fatal stop
            self._finish_upload()
        summary = {"status": reason, "start_step": start_step, "step": self.step,
                   "best": {s: (None if math.isinf(v) else round(v, 6)) for s, v in self.best.items()},
                   "params": [n_d, n_u], "skipped": self.skipped, **status_extra}
        if reason in ("total_steps", "max_steps"):
            self._write_done(summary)      # a relaunch (keep loop) then exits at once, see _already_done
        log(summary, marker=MARK + "_DONE")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Day1 job 2: M2 LatentDown/Up-S")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", "/workspace/night1"))
    ap.add_argument("--out", default=None, help="default: <root>/day1/m2")
    ap.add_argument("--deadline", default=os.environ.get("N1_DEADLINE"),
                    help="unix seconds (default: env N1_DEADLINE)")
    ap.add_argument("--finish-margin-s", type=float, default=600.0,
                    help="start the final save/eval/upload this many seconds before the deadline")
    ap.add_argument("--tae-ckpt", default=None, help="default: <root>/tae/ckpt_best.pt")
    ap.add_argument("--vae-path", default=None, help="Z-Image snapshot dir (default: HF cache download of vae/*)")
    ap.add_argument("--min-pairs", type=int, default=1000,
                    help="fetch corpus shards from the vault when fewer pairs exist")
    ap.add_argument("--min-train", type=int, default=2)
    ap.add_argument("--no-fetch", action="store_true", help="never download assets from the vault")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--decode-n", type=int, default=40, help="eval items that also get the teacher-decode PSNR")
    ap.add_argument("--eval-budget-s", type=float, default=240.0, help="time cap for the decode part of an eval")
    ap.add_argument("--cache-batch", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="samples per forward/backward (0 = whole batch; halves itself on CUDA OOM)")
    ap.add_argument("--crop", type=int, default=64, help="crop on the low-res latent grid (Up input 64, Down input 128)")
    ap.add_argument("--img-win", type=int, default=32,
                    help="latent window decoded by the TAE-T for the image-space L1 (32 -> 256 px)")
    ap.add_argument("--img-weight", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--latent-channels", type=int, default=None, help="default: read from the lat files")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--max-steps", type=int, default=None, help="stop at this absolute step (tests)")
    ap.add_argument("--log-every-s", type=float, default=240.0)
    ap.add_argument("--eval-every-s", type=float, default=900.0)
    ap.add_argument("--first-eval-s", type=float, default=300.0,
                    help="first eval/checkpoint comes early so problems show up in the logs")
    ap.add_argument("--prefetch-depth", type=int, default=6)
    ap.add_argument("--prefetch-threads", type=int, default=2)
    ap.add_argument("--max-consec-errors", type=int, default=MAX_CONSEC_ERRORS)
    ap.add_argument("--mem-gb", type=float, default=MEM_CAP_GB,
                    help="per-process CUDA memory cap in GB (0 = off); the GPU is shared with Z-Image")
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final-upload-timeout-s", type=float, default=240.0)
    return ap


def main(argv: Optional[Sequence[str]] = None, deps: Optional[dict] = None) -> int:
    args = build_parser().parse_args(argv)
    deps = deps or {}
    stop = deps.get("stop_event") or threading.Event()

    if threading.current_thread() is threading.main_thread():
        def _on_signal(signum, _frame):
            log({"signal": int(signum)})
            stop.set()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass

    trainer = Trainer(args, uploader=deps.get("uploader", _UNSET), now=deps.get("now", time.time),
                      sleep=deps.get("sleep", time.sleep), stop_event=stop, deps=deps)
    try:
        res = trainer.run()
    except Exception as e:  # noqa: BLE001 - last resort: save what we have and say why
        log({"fatal": err_str(e, 300)}, marker=MARK + "_DONE")
        try:
            if trainer.down is not None and trainer.up is not None and trainer.opt:
                save_atomic(trainer._ckpt_obj(), os.path.join(trainer.out, "ckpt_last.pt"))
        except Exception:  # noqa: BLE001
            pass
        return 2
    return 2 if res.get("status") in ("fatal",) else 0


if __name__ == "__main__":
    sys.exit(main())
