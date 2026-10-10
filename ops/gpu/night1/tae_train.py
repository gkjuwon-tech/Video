#!/usr/bin/env python3
"""Job B: M1 TAE-T spike training (tiny autoencoder that mimics the teacher VAE).

Run on the GPU box (torch 2.9.1 + CUDA), next to job A (corpus_gen.py):

    HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=<unix seconds> \
        python tae_train.py

Reads the pairs job A writes under N1_ROOT:
    img/{i:08d}.webp         RGB image
    lat/{i:08d}.npy          float16 (C, H/8, W/8), already (z - shift) * scale
Writes under N1_ROOT/tae:
    ckpt_last.pt             model + optimizer + step (atomic: tmp file, then rename)
    ckpt_best.pt             best eval PSNR(D(E(x)), x)
    samples/step_{s}.jpg     rows: original | D(teacher latent) | D(E(x))
    metrics.jsonl            one line per evaluation
    eval_indices.json        the frozen evaluation pairs (never trained on)
and uploads last/best/latest sample/metrics to HF ckpt/night1/tae_t/ every eval.

Markers on stdout (one JSON per line, flushed): N1_TAE, N1_TAE_EVAL, N1_TAE_DONE.
Run it again after any crash: it resumes from ckpt_last.pt.
The HF token comes from the environment only and is never printed or stored.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MARK = "N1_TAE"
REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
REPO_TYPE = "model"
REMOTE_DIR = "ckpt/night1/tae_t"

DEFAULT_WIDTHS: Tuple[int, ...] = (32, 64, 96, 128)
BLOCKS_PER_STAGE = 2
PARAM_MIN = 2_000_000
PARAM_MAX = 3_200_000
PARAM_TARGET = 2_500_000
TOTAL_STEPS = 200_000          # schedule length is a constant, not derived from the deadline
MAX_CONSEC_ERRORS = 200
MEM_CAP_GB = 6.0               # per-process VRAM ceiling: the GPU is shared with Z-Image (~20 GB resident)
LOADER_TIMEOUT_S = 120.0       # a stuck DataLoader worker must not block the stop/deadline checks forever
OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
_UNSET = object()

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


def log(payload: dict, marker: str = MARK) -> None:
    """One JSON line on stdout, flushed (vastai logs is the only window)."""
    print(f"{marker} {json.dumps(payload, ensure_ascii=False, default=str)}", flush=True)


def warn(what: str, **kw) -> None:
    d = {"warn": what}
    for k, v in kw.items():
        d[k] = scrub(v) if isinstance(v, str) else v
    log(d)


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
# model
# ---------------------------------------------------------------------------

def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.n1 = _gn(c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = _gn(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Encoder(nn.Module):
    """RGB -> latent (C channels, 1/8 resolution)."""

    def __init__(self, latent_channels: int, widths: Sequence[int] = DEFAULT_WIDTHS,
                 blocks: int = BLOCKS_PER_STAGE):
        super().__init__()
        w = list(widths)
        self.stem = nn.Conv2d(3, w[0], 3, padding=1)
        layers: List[nn.Module] = []
        for i in range(1, len(w)):
            layers.append(nn.Conv2d(w[i - 1], w[i], 3, stride=2, padding=1))
            layers.extend(ResBlock(w[i]) for _ in range(blocks))
        self.body = nn.Sequential(*layers)
        self.norm = _gn(w[-1])
        self.head = nn.Conv2d(w[-1], latent_channels, 1)

    def forward(self, x):
        return self.head(F.silu(self.norm(self.body(self.stem(x)))))


class Decoder(nn.Module):
    """latent -> RGB (no output activation; clamp only at evaluation)."""

    def __init__(self, latent_channels: int, widths: Sequence[int] = DEFAULT_WIDTHS,
                 blocks: int = BLOCKS_PER_STAGE):
        super().__init__()
        w = list(widths)
        self.stem = nn.Conv2d(latent_channels, w[-1], 1)
        layers: List[nn.Module] = []
        for i in range(len(w) - 1, 0, -1):
            layers.extend(ResBlock(w[i]) for _ in range(blocks))
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(nn.Conv2d(w[i], w[i - 1], 3, padding=1))
        self.body = nn.Sequential(*layers)
        self.norm = _gn(w[0])
        self.head = nn.Conv2d(w[0], 3, 3, padding=1)

    def forward(self, z):
        return self.head(F.silu(self.norm(self.body(self.stem(z)))))


class TAE(nn.Module):
    def __init__(self, latent_channels: int, widths: Sequence[int] = DEFAULT_WIDTHS,
                 blocks: int = BLOCKS_PER_STAGE):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.widths = tuple(int(x) for x in widths)
        self.enc = Encoder(latent_channels, widths, blocks)
        self.dec = Decoder(latent_channels, widths, blocks)

    def config(self) -> dict:
        return {"latent_channels": self.latent_channels, "widths": list(self.widths),
                "blocks": BLOCKS_PER_STAGE}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fit_widths(latent_channels: int, base: Sequence[int] = DEFAULT_WIDTHS,
               lo: int = PARAM_MIN, hi: int = PARAM_MAX) -> Tuple[Tuple[int, ...], int]:
    """Widths (multiples of 8) whose total parameter count lies in [lo, hi].

    Keeps the recommended widths when they already fit, else scales them
    toward the ~2.5M target.
    """
    def widths_for(s: float) -> Tuple[int, ...]:
        return tuple(max(8, int(round(b * s / 8.0)) * 8) for b in base)

    seen = set()
    scales = [1.0] + [round(1.0 + sgn * 0.05 * k, 4) for k in range(1, 40) for sgn in (-1, 1)]
    for s in scales:      # ordered by closeness to the recommended widths
        if s <= 0.1:
            continue
        w = widths_for(s)
        if w in seen:
            continue
        seen.add(w)
        n = count_params(TAE(latent_channels, w))
        if lo <= n <= hi:
            return w, n
    raise RuntimeError(f"no widths fit {lo}..{hi} params for C={latent_channels}")


# ---------------------------------------------------------------------------
# losses / schedule
# ---------------------------------------------------------------------------

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def sobel(x: torch.Tensor) -> torch.Tensor:
    """Per-channel Sobel gradients ``(B, 2C, H, W)`` ordered [c0x, c0y, c1x, ...]."""
    c = x.shape[1]
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype)
    k = torch.stack([kx, kx.t()])[:, None].repeat(c, 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=c)


def compute_losses(model: TAE, x: torch.Tensor, z: torch.Tensor, amp: bool = False,
                   device_type: str = "cpu") -> Dict[str, torch.Tensor]:
    """L_enc = MSE(E(x), z_t); L_dec = L1(D(z_t),x) + 0.5 L1(D(E(x)),x) + 0.1 L1(grad D(z_t), grad x)."""
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        ze = model.enc(x)
        both = model.dec(torch.cat([z, ze.float()], dim=0))
    ze = ze.float()
    y_z, y_e = both.float().chunk(2, dim=0)
    l_enc = F.mse_loss(ze, z)
    l_rec = F.l1_loss(y_z, x)
    l_e2e = F.l1_loss(y_e, x)
    with torch.no_grad():
        gx = sobel(x)
    l_grad = F.l1_loss(sobel(y_z), gx)
    l_dec = l_rec + 0.5 * l_e2e + 0.1 * l_grad
    return {"loss": l_enc + l_dec, "l_enc": l_enc, "l_dec": l_dec,
            "l_rec": l_rec, "l_e2e": l_e2e, "l_grad": l_grad}


def lr_at(step: int, base_lr: float = 1e-3, warmup: int = 1000, total: int = TOTAL_STEPS,
          min_frac: float = 0.01) -> float:
    """Linear warmup, then cosine over the constant total length."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


def optimizer_step(model: nn.Module, opt: torch.optim.Optimizer, loss: torch.Tensor,
                   clip: float = 1.0) -> bool:
    """Backward + clip + step. Returns False (and leaves weights alone) on a non-finite step."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    return clip_and_step(model, opt, clip)


def clip_and_step(model: nn.Module, opt: torch.optim.Optimizer, clip: float = 1.0) -> bool:
    """Clip the accumulated gradients and step. False (weights untouched) when the norm is non-finite."""
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not bool(torch.isfinite(gn)):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


def psnr(y: torch.Tensor, x: torch.Tensor) -> float:
    """PSNR in dB for tensors in [-1, 1] (prediction is clamped first)."""
    a = (y.float().clamp(-1, 1) + 1) / 2
    b = (x.float().clamp(-1, 1) + 1) / 2
    mse = float(F.mse_loss(a, b))
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


# ---------------------------------------------------------------------------
# data: file layout, crops, dataset
# ---------------------------------------------------------------------------

def idx_str(i: int) -> str:
    return f"{i:08d}"


def pair_paths(root: str, i: int) -> Dict[str, str]:
    s = idx_str(i)
    return {"img": os.path.join(root, "img", f"{s}.webp"),
            "lat": os.path.join(root, "lat", f"{s}.npy")}


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


def choose_eval_indices(avail: Sequence[int], n: int) -> List[int]:
    """The last ``n`` pairs by index (deterministic)."""
    s = sorted(avail)
    return s[-n:] if n > 0 else []


def load_or_make_eval_indices(tae_dir: str, avail: Sequence[int], n: int) -> List[int]:
    """Freeze the evaluation set on first use so it never drifts as the corpus grows."""
    path = os.path.join(tae_dir, "eval_indices.json")
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


def infer_latent_channels(root: str, pairs: Sequence[int]) -> int:
    for i in pairs:
        try:
            lat = np.load(pair_paths(root, i)["lat"], mmap_mode="r")
            if lat.ndim == 3:
                return int(lat.shape[0])
        except Exception as e:  # noqa: BLE001 - skip a bad file
            warn("lat_probe_failed", index=i, err=f"{type(e).__name__}: {e}")
    raise RuntimeError("cannot infer latent channels from any lat file")


def random_crop_origin(h: int, w: int, crop: int, rng: random.Random) -> Tuple[int, int]:
    """Top-left corner of a crop x crop window, aligned to the 8 px latent grid."""
    if crop % 8:
        raise ValueError("crop must be a multiple of 8")
    if h < crop or w < crop:
        raise ValueError(f"image {w}x{h} smaller than crop {crop}")
    return rng.randint(0, (h - crop) // 8) * 8, rng.randint(0, (w - crop) // 8) * 8


def crop_pair(arr: np.ndarray, lat: np.ndarray, y: int, x: int, crop: int,
              flip: bool) -> Tuple[np.ndarray, np.ndarray]:
    """arr (H,W,3) uint8, lat (C,H/8,W/8) -> (3,crop,crop) uint8, (C,crop/8,crop/8) float16."""
    if y % 8 or x % 8:
        raise ValueError("crop origin not aligned to 8")
    img = arr[y:y + crop, x:x + crop]
    z = lat[:, y // 8:(y + crop) // 8, x // 8:(x + crop) // 8]
    if flip:
        img = img[:, ::-1]
        z = z[:, :, ::-1]
    return (np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.uint8),
            np.ascontiguousarray(z, dtype=np.float16))


def load_train_item(root: str, i: int, crop: int, n: int, rng: random.Random,
                    latent_channels: Optional[int] = None, flip: bool = True):
    p = pair_paths(root, i)
    lat = np.load(p["lat"], mmap_mode="r")
    if lat.ndim != 3:
        raise ValueError(f"lat ndim {lat.ndim}")
    if latent_channels is not None and lat.shape[0] != latent_channels:
        raise ValueError(f"lat channels {lat.shape[0]} != {latent_channels}")
    with Image.open(p["img"]) as im:
        w, h = im.size
        if lat.shape[1] != h // 8 or lat.shape[2] != w // 8:
            raise ValueError(f"lat {tuple(lat.shape)} does not match image {w}x{h}")
        if h < crop or w < crop:
            raise ValueError(f"image {w}x{h} smaller than crop {crop}")
        arr = np.asarray(im.convert("RGB"))
    xs, zs = [], []
    for _ in range(n):
        y0, x0 = random_crop_origin(h, w, crop, rng)
        xc, zc = crop_pair(arr, lat, y0, x0, crop, flip and rng.random() < 0.5)
        xs.append(xc)
        zs.append(zc)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(zs))


_ITEM_WARNS = [0]


class PairDataset(torch.utils.data.Dataset):
    """Random aligned crops of (image, latent) pairs. Bad items are logged and replaced."""

    def __init__(self, root: str, seq: Sequence[int], crop: int = 256, crops_per_image: int = 1,
                 latent_channels: Optional[int] = None, seed: int = 0, flip: bool = True):
        self.root, self.seq = root, list(seq)
        self.crop, self.n = int(crop), max(1, int(crops_per_image))
        self.latent_channels, self.seed, self.flip = latent_channels, int(seed), flip

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, k: int):
        if not self.seq:
            return None
        for attempt in range(5):
            i = self.seq[(k + attempt) % len(self.seq)]
            try:
                rng = random.Random(f"{self.seed}-{k}-{attempt}")
                return load_train_item(self.root, i, self.crop, self.n, rng,
                                       self.latent_channels, self.flip)
            except Exception as e:  # noqa: BLE001 - one bad image must not stop the night
                if _ITEM_WARNS[0] < 20:
                    _ITEM_WARNS[0] += 1
                    warn("bad_item", index=i, err=f"{type(e).__name__}: {e}")
        return None


def collate(items):
    items = [it for it in items if it is not None]
    if not items:
        return None
    return torch.cat([a for a, _ in items]), torch.cat([b for _, b in items])


def build_epoch_sequence(indices: Sequence[int], batch_items: int, rng: random.Random,
                         min_steps: int = 300, max_steps: int = 2000) -> List[int]:
    """Shuffled index sequence for one 'epoch' (a full pass, clamped to [min,max] steps)."""
    n = len(indices)
    if n == 0 or batch_items <= 0:
        return []
    target = max(min_steps * batch_items, min(n, max_steps * batch_items))
    target = max(batch_items, target - target % batch_items)
    seq: List[int] = []
    while len(seq) < target:
        p = list(indices)
        rng.shuffle(p)
        seq.extend(p)
    return seq[:target]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def load_full_pair(root: str, i: int, latent_channels: Optional[int] = None):
    """Full-resolution (3,H8,W8) uint8 image and (C,H8/8,W8/8) float16 latent."""
    p = pair_paths(root, i)
    lat = np.load(p["lat"])
    with Image.open(p["img"]) as im:
        arr = np.asarray(im.convert("RGB"))
    h8, w8 = arr.shape[0] // 8 * 8, arr.shape[1] // 8 * 8
    if lat.ndim != 3 or lat.shape[1] != h8 // 8 or lat.shape[2] != w8 // 8:
        raise ValueError(f"lat {tuple(lat.shape)} does not match image {arr.shape[1]}x{arr.shape[0]}")
    if latent_channels is not None and lat.shape[0] != latent_channels:
        raise ValueError(f"lat channels {lat.shape[0]} != {latent_channels}")
    x = np.ascontiguousarray(arr[:h8, :w8].transpose(2, 0, 1))
    return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(lat, dtype=np.float16))


class EvalSet:
    """Frozen evaluation pairs, decoded once and kept in RAM (64 x ~3 MB)."""

    def __init__(self, root: str, indices: Sequence[int], latent_channels: Optional[int] = None):
        self.root, self.indices, self.C = root, list(indices), latent_channels
        self._cache: Dict[int, tuple] = {}
        self._bad: set = set()

    def items(self) -> List[tuple]:
        out = []
        for i in self.indices:
            if i in self._bad:
                continue
            if i not in self._cache:
                try:
                    self._cache[i] = load_full_pair(self.root, i, self.C)
                except Exception as e:  # noqa: BLE001
                    self._bad.add(i)
                    warn("bad_eval_item", index=i, err=f"{type(e).__name__}: {e}")
                    continue
            out.append((i,) + self._cache[i])
        return out


def sample_positions(n: int, k: int = 4) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({int(round(j * (n - 1) / (k - 1))) for j in range(k)})


def _to_u8_hwc(y: torch.Tensor) -> np.ndarray:
    a = ((y.float().clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)
    return a[0].permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@torch.no_grad()
def evaluate(model: TAE, evalset: EvalSet, device, n_samples: int = 4):
    """Full-resolution metrics on the frozen set. Returns (metrics | None, samples)."""
    was_training = model.training
    model.eval()
    items = evalset.items()
    pos = set(sample_positions(len(items), n_samples))
    ps_zt, ps_ee, lat_mse, samples = [], [], [], []
    try:
        for k, (i, xu8, zt) in enumerate(items):
            try:
                x = xu8[None].to(device).float() / 127.5 - 1.0
                z = zt[None].to(device).float()
                ze = model.enc(x)
                y_z = model.dec(z)
                y_e = model.dec(ze)
                ps_zt.append(psnr(y_z, x))
                ps_ee.append(psnr(y_e, x))
                lat_mse.append(float(F.mse_loss(ze, z)))
                if k in pos:
                    samples.append((_to_u8_hwc(x), _to_u8_hwc(y_z), _to_u8_hwc(y_e)))
            except Exception as e:  # noqa: BLE001 - skip this eval image
                warn("eval_item_failed", index=i, err=f"{type(e).__name__}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        model.train(was_training)
    if not ps_ee:
        return None, samples
    n = len(ps_ee)
    return {"n": n, "psnr_dec_zt": sum(ps_zt) / n, "psnr_dec_enc": sum(ps_ee) / n,
            "latent_mse": sum(lat_mse) / n}, samples


def make_sample_grid(samples: Sequence[tuple], tile: int = 384) -> Image.Image:
    """Rows of (original | teacher latent -> our decoder | our encoder -> our decoder)."""
    rows = max(1, len(samples))
    canvas = Image.new("RGB", (3 * tile, rows * tile), (0, 0, 0))
    for r, trio in enumerate(samples):
        for c, arr in enumerate(trio):
            im = Image.fromarray(arr)
            s = min(tile / im.width, tile / im.height)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            canvas.paste(im, (c * tile + (tile - im.width) // 2, r * tile + (tile - im.height) // 2))
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


def load_ckpt(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - our own file; retry permissive before giving up
            obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "model" not in obj or "step" not in obj:
            raise ValueError("missing keys")
        return obj
    except Exception as e:  # noqa: BLE001
        warn("ckpt_unreadable", path=os.path.basename(path), err=f"{type(e).__name__}: {e}")
        return None


def clean_tmp(tae_dir: str) -> int:
    n = 0
    for d in (tae_dir, os.path.join(tae_dir, "samples")):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(d, name))
                    n += 1
                except OSError:
                    pass
    return n


def latest_sample(samples_dir: str) -> Optional[str]:
    best, best_step = None, -1
    try:
        names = os.listdir(samples_dir)
    except FileNotFoundError:
        return None
    for n in names:
        m = re.fullmatch(r"step_(\d+)\.jpg", n)
        if m and int(m.group(1)) > best_step:
            best, best_step = n, int(m.group(1))
    return os.path.join(samples_dir, best) if best else None


def stage_upload(tae_dir: str) -> Tuple[str, List[str]]:
    """Copy last/best/latest sample/metrics into one folder so a single commit uploads them."""
    stage = os.path.join(tae_dir, "upload_stage")
    os.makedirs(stage, exist_ok=True)
    srcs = [(os.path.join(tae_dir, "ckpt_last.pt"), "ckpt_last.pt"),
            (os.path.join(tae_dir, "ckpt_best.pt"), "ckpt_best.pt"),
            (os.path.join(tae_dir, "metrics.jsonl"), "metrics.jsonl")]
    ls = latest_sample(os.path.join(tae_dir, "samples"))
    if ls:
        srcs.append((ls, "sample_latest.jpg"))
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
            commit_message="night1 tae_t checkpoint")

    return upload


def upload_with_retry(uploader: Callable[[str, str], None], folder: str, remote: str,
                      tries: int = 3, sleep: Callable[[float], None] = time.sleep) -> bool:
    for k in range(tries):
        try:
            uploader(folder, remote)
            return True
        except Exception as e:  # noqa: BLE001 - upload trouble is a warning, never fatal
            warn("upload_failed", attempt=k + 1, err=f"{type(e).__name__}: {e}"[:300])
            if k + 1 < tries:
                sleep(5 * (3 ** k))
    return False


def vram_gb() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def is_oom(e: BaseException) -> bool:
    """CUDA OOM, including the cuDNN/cuBLAS workspace failures that surface as plain RuntimeError."""
    if isinstance(e, OOM_ERROR):
        return True
    m = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in m or "alloc_failed" in m)


def vram_reserved_gb() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_reserved() / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def mem_fraction(mem_gb: float, total_bytes: int) -> float:
    """Fraction of the device to allow this process (clamped to (0, 1])."""
    if mem_gb <= 0 or total_bytes <= 0:
        return 1.0
    return min(1.0, (mem_gb * 2**30) / float(total_bytes))


def apply_cuda_limits(device: torch.device, mem_gb: float) -> Optional[float]:
    """Cap this process's caching allocator so the co-resident Z-Image job keeps its headroom.

    Returns the cap in GB, or None when no cap was applied.
    """
    if device.type != "cuda" or mem_gb <= 0:
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory
        frac = mem_fraction(mem_gb, total)
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        return round(frac * total / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        warn("mem_cap_failed", err=f"{type(e).__name__}: {e}"[:300])
        return None


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

def wait_for_pairs(root: str, need: int, poll: float, deadline: Optional[float],
                   now: Callable[[], float], sleep: Callable[[float], None],
                   stop: threading.Event, log_every: float = 300.0) -> Optional[List[int]]:
    """Block until ``need`` pairs exist (polling). None when the deadline or a stop signal comes first."""
    last = None
    while True:
        pairs = list_pairs(root)
        if len(pairs) >= need:
            return pairs
        t = now()
        if stop.is_set() or (deadline is not None and t >= deadline):
            return None
        if last is None or t - last >= log_every:
            log({"waiting_for_pairs": len(pairs), "need": need})
            last = t
        sleep(poll if deadline is None else max(0.0, min(poll, deadline - t)))


class Trainer:
    def __init__(self, args, uploader=_UNSET, now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 stop_event: Optional[threading.Event] = None):
        self.a = args
        self.now, self.sleep = now, sleep
        self.stop = stop_event or threading.Event()
        self.root = args.root
        self.tae_dir = os.path.join(self.root, "tae")
        self.samples_dir = os.path.join(self.tae_dir, "samples")
        self.deadline = parse_deadline(args.deadline)
        self.uploader = uploader
        self.upload_thread: Optional[threading.Thread] = None
        self.best_psnr = -math.inf
        self.step = 0
        self.skipped = 0
        self.bad_run = 0
        self.model: Optional[TAE] = None
        self.opt = None
        self.evalset: Optional[EvalSet] = None
        self.device = None
        self.amp = False
        self.last_loss = float("nan")
        self.micro_batch = int(getattr(args, "micro_batch", 0) or args.batch)
        self.mem_cap_gb: Optional[float] = None
        self.first_step_logged = False

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

    def _ckpt_obj(self) -> dict:
        return {"version": 1, "step": self.step, "model": self.model.state_dict(),
                "opt": self.opt.state_dict(), "config": self.model.config(),
                "best_psnr": self.best_psnr, "micro_batch": self.micro_batch,
                "saved_at": time.time()}

    def _load_resume(self, latent_channels: int) -> Optional[dict]:
        for name in ("ckpt_last.pt", "ckpt_best.pt"):
            ck = load_ckpt(os.path.join(self.tae_dir, name))
            if ck is None:
                continue
            cfg = ck.get("config") or {}
            if int(cfg.get("latent_channels", -1)) != latent_channels or not cfg.get("widths"):
                warn("ckpt_config_mismatch", path=name, ckpt_C=cfg.get("latent_channels"),
                     data_C=latent_channels)
                continue
            return ck
        return None

    def _restore(self, ck: dict) -> bool:
        try:
            self.model.load_state_dict(ck["model"])
            if "opt" in ck:
                self.opt.load_state_dict(ck["opt"])
            self.step = int(ck["step"])
            bp = ck.get("best_psnr")
            self.best_psnr = float(bp) if bp is not None else -math.inf
            mb = ck.get("micro_batch")
            if mb and not getattr(self.a, "micro_batch", 0):
                # never grow back past a size that already hit OOM on this card
                self.micro_batch = max(1, min(self.micro_batch, int(mb)))
            return True
        except Exception as e:  # noqa: BLE001
            warn("ckpt_restore_failed", err=f"{type(e).__name__}: {e}")
            return False

    # -- checkpoint + eval + upload ------------------------------------------
    def _evaluate_guarded(self):
        """Full-resolution eval without leaving a bigger footprint behind on the shared GPU.

        cudnn.benchmark is off while evaluating (it trial-allocates large workspaces for every new
        resolution) and the cache is released afterwards.
        """
        cuda = self.device is not None and self.device.type == "cuda"
        prev = torch.backends.cudnn.benchmark
        if cuda:
            torch.backends.cudnn.benchmark = False
        try:
            return evaluate(self.model, self.evalset, self.device)
        finally:
            if cuda:
                torch.backends.cudnn.benchmark = prev
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

    def checkpoint_and_eval(self, final: bool = False) -> Optional[dict]:
        metrics = None
        improved = False
        try:
            metrics, samples = self._evaluate_guarded()
            if metrics is not None:
                if samples:
                    try:
                        path = os.path.join(self.samples_dir, f"step_{self.step}.jpg")
                        make_sample_grid(samples).save(path + ".tmp", format="JPEG", quality=90)
                        os.replace(path + ".tmp", path)
                    except Exception as e:  # noqa: BLE001
                        warn("sample_save_failed", err=f"{type(e).__name__}: {e}")
                improved = metrics["psnr_dec_enc"] > self.best_psnr
                if improved:
                    self.best_psnr = metrics["psnr_dec_enc"]
                rec = {"step": self.step, "time": round(time.time(), 1), "loss": self.last_loss,
                       "best": improved, "best_psnr": self.best_psnr, **metrics}
                with open(os.path.join(self.tae_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                log({"step": self.step, **{k: (round(v, 5) if isinstance(v, float) else v)
                                           for k, v in metrics.items()},
                     "best_psnr": round(self.best_psnr, 4), "improved": improved,
                     "final": final}, marker=MARK + "_EVAL")
        except Exception as e:  # noqa: BLE001 - eval trouble must not lose the checkpoint
            warn("eval_failed", err=f"{type(e).__name__}: {e}"[:300])
        try:
            obj = self._ckpt_obj()
            save_atomic(obj, os.path.join(self.tae_dir, "ckpt_last.pt"))
            if improved:
                save_atomic(obj, os.path.join(self.tae_dir, "ckpt_best.pt"))
        except Exception as e:  # noqa: BLE001
            warn("ckpt_save_failed", err=f"{type(e).__name__}: {e}"[:300])
        self._start_upload(final)
        return metrics

    def _upload_job(self, stage: str) -> None:
        ok = upload_with_retry(self.uploader, stage, REMOTE_DIR, sleep=self.sleep)
        if ok:
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
            stage, _ = stage_upload(self.tae_dir)
        except Exception as e:  # noqa: BLE001
            warn("stage_failed", err=f"{type(e).__name__}: {e}"[:300])
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
    def _make_loader(self, seq: List[int], seed: int, workers: int):
        crops = max(1, self.a.crops_per_image)
        ds = PairDataset(self.root, seq, self.a.crop, crops, self._C, seed)
        batch_items = max(1, self.a.batch // crops)
        kw = {}
        if workers > 0:
            kw["prefetch_factor"] = 4
            if self.a.loader_timeout_s > 0:
                kw["timeout"] = self.a.loader_timeout_s     # raises -> run() rebuilds the loader
            if self.a.mp_context != "default":
                # forkserver children start clean, so no lock held by the upload thread leaks in
                kw["multiprocessing_context"] = self.a.mp_context
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_items, shuffle=False, num_workers=workers, drop_last=True,
            collate_fn=collate, pin_memory=(self.device.type == "cuda"), **kw)

    def _accumulate(self, x: torch.Tensor, z: torch.Tensor, micro: int) -> torch.Tensor:
        """Backward over micro-batches (gradient accumulation). Returns [loss, l_enc, l_dec] for the full batch."""
        self.opt.zero_grad(set_to_none=True)
        n = x.shape[0]
        tot = None
        for s in range(0, n, micro):
            xs, zs = x[s:s + micro], z[s:s + micro]
            w = xs.shape[0] / n
            out = compute_losses(self.model, xs, zs, self.amp, self.device.type)
            (out["loss"] * w).backward()
            part = torch.stack([out["loss"].detach(), out["l_enc"].detach(), out["l_dec"].detach()]) * w
            tot = part if tot is None else tot + part
            del out
        return tot

    def _shrink_micro_batch(self, n: int) -> bool:
        """Halve the micro-batch after an OOM (effective batch stays ``n`` via accumulation)."""
        cur = max(1, min(self.micro_batch, n))
        if cur <= 1:
            return False
        self.micro_batch = max(1, cur // 2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        warn("oom_backoff", step=self.step, micro_batch=self.micro_batch,
             accum_steps=-(-n // self.micro_batch), vram_GB=vram_gb())
        return True

    def _train_step(self, batch) -> bool:
        xb, zb = batch
        x = xb.to(self.device, non_blocking=True).float().div_(127.5).sub_(1.0)
        z = zb.to(self.device, non_blocking=True).float()
        lr = lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr
        while True:
            parts, oom = None, False
            try:
                parts = self._accumulate(x, z, max(1, min(self.micro_batch, x.shape[0])))
            except RuntimeError as e:        # torch.OutOfMemoryError is a RuntimeError
                if not is_oom(e):
                    raise
                oom = True
            if not oom:
                break
            self.opt.zero_grad(set_to_none=True)
            if not self._shrink_micro_batch(x.shape[0]):
                raise OOM_ERROR("out of memory at micro_batch=1")
        if not clip_and_step(self.model, self.opt, 1.0):
            self.skipped += 1
            self.bad_run += 1
            return False
        self.bad_run = 0
        self.acc += parts
        self.acc_n += 1
        self.crops_seen += x.shape[0]
        self.step += 1
        return True

    def _log_progress(self, t: float) -> None:
        dt = max(1e-6, t - self.log_t)
        if self.acc_n:
            loss, l_enc, l_dec = [v / self.acc_n for v in self.acc.tolist()]
            self.last_loss = loss
        else:
            loss = l_enc = l_dec = float("nan")
        log({"step": self.step, "loss": round(loss, 5), "l_enc": round(l_enc, 5),
             "l_dec": round(l_dec, 5), "img_per_s": round(self.crops_seen / dt, 1),
             "lr": round(lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps), 6),
             "skipped": self.skipped, "vram_GB": vram_gb(), "micro_batch": self.micro_batch})
        self.acc.zero_()
        self.acc_n = 0
        self.crops_seen = 0
        self.log_t = t

    def run(self) -> dict:
        a = self.a
        os.makedirs(self.samples_dir, exist_ok=True)
        clean_tmp(self.tae_dir)
        if self.uploader is _UNSET:
            self.uploader = None if a.no_upload else make_uploader(os.environ.get("HF_TOKEN"), a.repo_id)
        log({"start": True, "root": self.root, "hf_token": "yes" if self.uploader else "no",
             "deadline": self.deadline, "torch": torch.__version__})

        pairs = wait_for_pairs(self.root, a.min_pairs, a.poll_s, self.deadline, self.now,
                               self.sleep, self.stop)
        if pairs is None:
            log({"stop": "deadline_no_data" if not self.stop.is_set() else "signal_no_data",
                 "pairs": len(list_pairs(self.root))}, marker=MARK + "_DONE")
            return {"status": "no_data", "step": 0}

        dev_name = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev_name.startswith("cuda") and not (os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                                                or os.environ.get("PYTORCH_ALLOC_CONF")):
            # must be set before the first CUDA allocation; lets empty_cache() hand memory back
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device = torch.device(dev_name)
        self.amp = {"on": True, "off": False}.get(a.amp, self.device.type == "cuda")
        if self.device.type == "cuda":
            self.mem_cap_gb = apply_cuda_limits(self.device, a.mem_gb)
            torch.backends.cudnn.benchmark = True      # training crops only; _evaluate_guarded turns it off
        torch.manual_seed(a.seed)

        n_eval = min(a.eval_n, len(pairs) // 2)
        eval_idx = load_or_make_eval_indices(self.tae_dir, pairs, n_eval)
        eval_set = set(eval_idx)
        self._C = a.latent_channels or infer_latent_channels(self.root, pairs)
        self.evalset = EvalSet(self.root, eval_idx, self._C)

        ck = self._load_resume(self._C)
        if ck is not None:
            widths, n_params = tuple(ck["config"]["widths"]), None
        else:
            widths, n_params = fit_widths(self._C)
        self.model = TAE(self._C, widths).to(self.device)
        n_params = count_params(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=a.wd,
                                     fused=(self.device.type == "cuda"))
        resumed = ck is not None and self._restore(ck)
        start_step = self.step
        log({"params": n_params, "params_M": round(n_params / 1e6, 3), "in_range": PARAM_MIN <= n_params <= PARAM_MAX,
             "widths": list(widths), "latent_channels": self._C, "device": str(self.device),
             "amp": self.amp, "resumed": bool(resumed), "step": self.step,
             "micro_batch": self.micro_batch, "batch": a.batch, "mem_cap_GB": self.mem_cap_gb,
             "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"),
             "eval_pairs": len(eval_idx), "pairs": len(pairs)})

        self.acc = torch.zeros(3, device=self.device)
        self.acc_n = 0
        self.crops_seen = 0
        t0 = self.now()
        self.log_t = t0
        next_log = t0 + a.log_every_s
        next_eval = t0 + min(a.first_eval_s, a.eval_every_s)
        workers = a.workers
        loader_errors = 0
        consec_err = 0
        epoch = 0
        reason: Optional[str] = None
        self.model.train()

        while reason is None:
            reason = self._stop_reason()
            if reason:
                break
            train_idx = [i for i in list_pairs(self.root) if i not in eval_set]
            if not train_idx:
                self.sleep(a.poll_s)
                continue
            seed = a.seed * 1_000_003 + self.step * 31 + epoch
            epoch += 1
            seq = build_epoch_sequence(train_idx, max(1, a.batch // max(1, a.crops_per_image)),
                                       random.Random(seed), a.epoch_min_steps, a.epoch_max_steps)
            try:
                it = iter(self._make_loader(seq, seed, workers))
            except Exception as e:  # noqa: BLE001
                warn("loader_build_failed", err=f"{type(e).__name__}: {e}"[:300])
                workers = 0
                self.sleep(2)
                continue
            while True:
                reason = self._stop_reason()
                if reason:
                    break
                try:
                    batch = next(it)
                except StopIteration:
                    loader_errors = 0       # a whole epoch went through: earlier hiccups were transient
                    break
                except Exception as e:  # noqa: BLE001 - worker died / timed out: rebuild the loader
                    loader_errors += 1
                    warn("loader_error", n=loader_errors, err=f"{type(e).__name__}: {e}"[:300])
                    if loader_errors >= 3:
                        workers = 0
                    self.sleep(1)
                    break
                if batch is None:
                    continue
                try:
                    self._train_step(batch)
                    consec_err = 0
                    if not self.first_step_logged and self.step > start_step:
                        self.first_step_logged = True
                        log({"first_step": self.step, "vram_GB": vram_gb(),
                             "vram_reserved_GB": vram_reserved_gb(), "micro_batch": self.micro_batch,
                             "batch": a.batch, "mem_cap_GB": self.mem_cap_gb})
                    if self.bad_run >= 25:
                        warn("nonfinite_streak", step=self.step)
                        ck2 = self._load_resume(self._C)
                        if ck2 is not None:
                            self._restore(ck2)
                        self.bad_run = 0
                except Exception as e:  # noqa: BLE001 - OOM or anything else: skip the step
                    consec_err += 1
                    warn("step_failed", step=self.step, n=consec_err, err=f"{type(e).__name__}: {e}"[:300])
                    self.opt.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if consec_err >= MAX_CONSEC_ERRORS:
                        log({"fatal": "too_many_step_errors"}, marker=MARK + "_DONE")
                        return {"status": "fatal", "step": self.step}
                    self.sleep(min(30.0, float(consec_err)))
                    continue
                t = self.now()
                if t >= next_log:
                    self._log_progress(t)
                    next_log = t + a.log_every_s
                if t >= next_eval:
                    self.checkpoint_and_eval()
                    next_eval = self.now() + a.eval_every_s
            del it

        # cleanup: final save/eval/upload, then a normal exit
        if self.acc_n:
            self._log_progress(self.now())
        self.checkpoint_and_eval(final=True)
        self._finish_upload()
        summary = {"status": reason, "start_step": start_step, "step": self.step,
                   "best_psnr": None if self.best_psnr == -math.inf else round(self.best_psnr, 4),
                   "params": n_params, "skipped": self.skipped}
        log(summary, marker=MARK + "_DONE")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Job B: M1 TAE-T spike training")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", "/workspace/night1"))
    ap.add_argument("--deadline", default=os.environ.get("N1_DEADLINE"),
                    help="unix seconds (default: env N1_DEADLINE)")
    ap.add_argument("--finish-margin-s", type=float, default=0.0,
                    help="start the final save/eval/upload this many seconds before the deadline")
    ap.add_argument("--min-pairs", type=int, default=1000, help="wait until this many pairs exist")
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32, help="crops per step")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops-per-image", type=int, default=1,
                    help="crops cut from each decoded image (raise it if data loading is the bottleneck)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="crops per forward/backward (0 = whole batch; halves itself on CUDA OOM, "
                         "gradient accumulation keeps the effective batch)")
    ap.add_argument("--mem-gb", type=float, default=MEM_CAP_GB,
                    help="per-process CUDA memory cap in GB (0 = off); the GPU is shared with Z-Image")
    ap.add_argument("--loader-timeout-s", type=float, default=LOADER_TIMEOUT_S,
                    help="DataLoader worker timeout (0 = wait forever)")
    ap.add_argument("--mp-context", choices=["forkserver", "spawn", "fork", "default"],
                    default="forkserver", help="DataLoader worker start method")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--max-steps", type=int, default=None, help="stop at this absolute step (tests)")
    ap.add_argument("--log-every-s", type=float, default=300.0)
    ap.add_argument("--eval-every-s", type=float, default=900.0)
    ap.add_argument("--first-eval-s", type=float, default=300.0,
                    help="first eval/checkpoint comes early so problems show up in the logs")
    ap.add_argument("--epoch-min-steps", type=int, default=300)
    ap.add_argument("--epoch-max-steps", type=int, default=2000)
    ap.add_argument("--latent-channels", type=int, default=None, help="default: read from the lat files")
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--final-upload-timeout-s", type=float, default=300.0)
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
                      sleep=deps.get("sleep", time.sleep), stop_event=stop)
    try:
        res = trainer.run()
    except Exception as e:  # noqa: BLE001 - last resort: save what we have and say why
        log({"fatal": f"{type(e).__name__}: {scrub(e)}"[:500]}, marker=MARK + "_DONE")
        try:
            if trainer.model is not None and trainer.opt is not None:
                save_atomic(trainer._ckpt_obj(), os.path.join(trainer.tae_dir, "ckpt_last.pt"))
        except Exception:  # noqa: BLE001
            pass
        return 2
    return 2 if res.get("status") == "fatal" else 0


if __name__ == "__main__":
    sys.exit(main())
