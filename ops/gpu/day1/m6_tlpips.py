#!/usr/bin/env python3
"""Day1 job 3: M6 T-LPIPS-S, a 4M perceptual-distance ruler distilled from Z-Image internals.

Run on the GPU box (torch 2.9.1 + CUDA), after night1 left its corpus/TAE under N1_ROOT:

    HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=1791723600 \
        PYTHONPATH=$N1_ROOT/repo:$N1_ROOT/repo/ops/gpu/night1 \
        python ops/gpu/day1/m6_tlpips.py

Phases (``--phase all`` runs A+B, frees Z-Image, then runs C):
  A  build distorted pairs from 256^2 corpus crops (CPU pool; the M1 TAE-T and teacher-VAE
     round trips run on the GPU) and store them as ``pairs/{split}_{chunk}.npy``
  B  teacher labels: both images -> teacher VAE -> same noise at sigma=0.3 -> ONE transformer
     forward with hooks on blocks 8/15/23 -> token-wise normalised squared difference
     -> ``labels_{split}_{chunk}.npz`` (resumable, one file per chunk)
  C  train the student (5 conv stages, learned non-negative channel weights) on the labels

Output under ``$N1_ROOT/day1/m6``: manifest.json, labels_*.npz, pairs/, ckpt_last.pt,
ckpt_best.pt, metrics.jsonl, scatter.png; uploaded to HF ``ckpt/day1/m6/``.
Markers (one JSON line, <= 400 chars, flushed): D1_M6, D1_M6_EVAL, D1_M6_DONE.
Re-running resumes at chunk / checkpoint granularity. HF_TOKEN comes from the environment
only and is never printed or stored.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
import math
import multiprocessing as mp
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
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

MARK = "D1_M6"
MARKER_LIMIT = 400
REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
REPO_TYPE = "model"
REMOTE_DIR = "ckpt/day1/m6"
TAE_REMOTE = "ckpt/night1/tae_t/ckpt_best.pt"
CORPUS_PREFIX = "corpus/night1/"
ZIMAGE_ID = "Tongyi-MAI/Z-Image-Turbo"

CROP = 256
SIGMA = 0.3                       # flow-matching time: z_t = (1 - sigma) z + sigma eps
BLOCKS = (8, 15, 23)              # indices into transformer.layers (30 blocks, dim 3840)
PATCH = 2
N_TRAIN, N_VAL, N_TRIP = 60_000, 3_000, 2_000
CHUNK = 480                       # pairs per stored chunk (15 teacher batches of 32)
LABEL_BATCH = 32                  # pairs per teacher forward (= 64 images)
PARAM_MIN, PARAM_MAX = 3_000_000, 5_000_000
TOTAL_STEPS = 20_000
MAX_CONSEC_ERRORS = 100           # spec: 100 failures in a row abort the run
FINISH_MARGIN_S = 600.0           # spec: final save/eval/upload 600 s before the deadline
TRAIN_RESERVE_S = 5400.0          # labelling stops this long before the deadline so C always gets time
LOG_EPS = 1e-4
MAX_DEADLINE_AHEAD = 48 * 3600.0
ID_OFFSET = {"train": 0, "val": 10_000_000, "trip": 20_000_000}
GPU_OPS = ("tae", "vae")
CPU_OPS = ("jpeg", "blur", "noise", "resample", "unsharp", "color", "geom")
OP_WEIGHTS = {"jpeg": 1.0, "blur": 1.0, "noise": 1.0, "resample": 1.0, "unsharp": 0.8,
              "color": 0.8, "geom": 0.8, "tae": 0.7, "vae": 0.7}
_UNSET = object()
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
        warn("bad_deadline", value=s[:40])
        return None
    return f if (f > 0 and math.isfinite(f)) else None


def is_oom(e: BaseException) -> bool:
    if isinstance(e, OOM_ERROR):
        return True
    m = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in m or "alloc_failed" in m)


def free_cuda_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def vram_gb() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def free_disk_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 2**30
    except OSError:
        return float("inf")


# ---------------------------------------------------------------------------
# step A · distortions (pure numpy / cv2 / PIL; uint8 HxWx3 in, uint8 HxWx3 out)
# ---------------------------------------------------------------------------

_INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
           "nearest": cv2.INTER_NEAREST}


def _clip_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def op_jpeg(img: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB")).copy()


def op_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101)


def op_noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Additive Gaussian noise; ``sigma`` is on the 0..1 intensity scale (e.g. 8/255)."""
    rng = np.random.default_rng(int(seed))
    return _clip_u8(img.astype(np.float32) + rng.normal(0.0, float(sigma) * 255.0, img.shape))


def op_resample(img: np.ndarray, scale: float, down: str = "area", up: str = "linear") -> np.ndarray:
    h, w = img.shape[:2]
    sh, sw = max(2, int(round(h * scale))), max(2, int(round(w * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=_INTERP[down])
    return cv2.resize(small, (w, h), interpolation=_INTERP[up])


def op_unsharp(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    f = img.astype(np.float32)
    b = cv2.GaussianBlur(f, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101)
    return _clip_u8(f + float(amount) * (f - b))


def op_color(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    f = f * (1.0 + brightness)
    gray = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])[..., None]
    f = (f - gray.mean()) * (1.0 + contrast) + gray.mean()
    gray = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])[..., None]
    f = gray + (f - gray) * (1.0 + saturation)
    return _clip_u8(f * 255.0)


def op_geom(img: np.ndarray, dx: float, dy: float, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), float(angle), 1.0)
    m[0, 2] += float(dx)
    m[1, 2] += float(dy)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def sample_op(name: str, rng: random.Random) -> dict:
    """Random strength for one op, ranges from the spec."""
    if name == "jpeg":
        return {"op": name, "q": rng.randint(10, 90)}
    if name == "blur":
        return {"op": name, "sigma": round(math.exp(rng.uniform(math.log(0.5), math.log(3.0))), 4)}
    if name == "noise":
        return {"op": name, "sigma": round(rng.uniform(2.0, 20.0) / 255.0, 6), "seed": rng.getrandbits(32)}
    if name == "resample":
        return {"op": name, "scale": round(rng.uniform(0.25, 0.75), 4),
                "down": rng.choice(["area", "linear"]), "up": rng.choice(["linear", "cubic"])}
    if name == "unsharp":
        return {"op": name, "amount": round(rng.uniform(1.0, 5.0), 3), "sigma": round(rng.uniform(0.8, 2.5), 3)}
    if name == "color":
        return {"op": name, "brightness": round(rng.uniform(-0.15, 0.15), 4),
                "contrast": round(rng.uniform(-0.15, 0.15), 4), "saturation": round(rng.uniform(-0.15, 0.15), 4)}
    if name == "geom":
        mag, ang = rng.uniform(0.5, 3.0), rng.uniform(0.0, 2.0 * math.pi)
        return {"op": name, "dx": round(mag * math.cos(ang), 3), "dy": round(mag * math.sin(ang), 3),
                "angle": round(rng.uniform(-2.0, 2.0), 3)}
    if name in GPU_OPS:
        return {"op": name}
    raise ValueError(f"unknown op {name}")


def sample_ops(rng: random.Random, gpu_ops: Sequence[str] = GPU_OPS) -> List[dict]:
    """1-2 distinct ops (never both round trips); GPU ops are ordered first."""
    names = [n for n in list(CPU_OPS) + list(gpu_ops) if n in OP_WEIGHTS]
    k = 1 if rng.random() < 0.55 else 2
    chosen: List[str] = []
    pool = list(names)
    while len(chosen) < k and pool:
        w = [OP_WEIGHTS[n] for n in pool]
        pick = rng.choices(pool, weights=w, k=1)[0]
        pool.remove(pick)
        if pick in GPU_OPS and any(c in GPU_OPS for c in chosen):
            continue
        chosen.append(pick)
    chosen.sort(key=lambda n: 0 if n in GPU_OPS else 1)
    return [sample_op(n, rng) for n in chosen]


def apply_op(img: np.ndarray, op: dict) -> np.ndarray:
    n = op["op"]
    if n == "jpeg":
        return op_jpeg(img, op["q"])
    if n == "blur":
        return op_blur(img, op["sigma"])
    if n == "noise":
        return op_noise(img, op["sigma"], op["seed"])
    if n == "resample":
        return op_resample(img, op["scale"], op.get("down", "area"), op.get("up", "linear"))
    if n == "unsharp":
        return op_unsharp(img, op["amount"], op["sigma"])
    if n == "color":
        return op_color(img, op["brightness"], op["contrast"], op["saturation"])
    if n == "geom":
        return op_geom(img, op["dx"], op["dy"], op["angle"])
    raise ValueError(f"cannot apply op {n} on the CPU")


def apply_cpu_ops(img: np.ndarray, ops: Sequence[dict]) -> np.ndarray:
    """Apply the CPU ops in order (GPU ops are skipped: they were applied before)."""
    out = img
    for op in ops:
        if op["op"] in CPU_OPS:
            out = apply_op(out, op)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# step A · deterministic pair specs, corpus crops, chunk building
# ---------------------------------------------------------------------------

def idx_str(i: int) -> str:
    return f"{i:08d}"


def corpus_img_path(root: str, i: int) -> str:
    return os.path.join(root, "img", f"{idx_str(i)}.webp")


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


def list_corpus(root: str) -> List[int]:
    """Sorted corpus indices that have an image (a ``.tmp`` never matches the extension)."""
    return sorted(_indices_in(os.path.join(root, "img"), ".webp"))


def split_corpus(indices: Sequence[int], val_pct: int = 10) -> Tuple[List[int], List[int]]:
    """Deterministic train / held-out split by a hash of the index; val and triplets draw only from the held-out part."""
    tr, va = [], []
    for i in sorted(indices):
        (va if zlib.crc32(f"m6-split-{i}".encode()) % 100 < val_pct else tr).append(i)
    if not tr or not va:       # tiny corpus: share it rather than fail
        return list(sorted(indices)), list(sorted(indices))
    return tr, va


def make_spec(seed: int, split: str, k: int, pool: Sequence[int], gpu_ops: Sequence[str] = GPU_OPS) -> dict:
    """Everything needed to rebuild pair/triplet ``k`` of ``split`` bit-for-bit (apart from GPU round trips)."""
    rng = random.Random(f"m6|{seed}|{split}|{k}")
    spec = {"split": split, "k": int(k), "src": int(pool[rng.randrange(len(pool))]),
            "alts": [int(pool[rng.randrange(len(pool))]) for _ in range(3)],
            "u": rng.random(), "v": rng.random()}
    if split == "trip":
        spec["ops_a"] = sample_ops(rng, gpu_ops)
        spec["ops_b"] = sample_ops(rng, gpu_ops)
    else:
        spec["ops"] = sample_ops(rng, gpu_ops)
    return spec


def crop_origin(h: int, w: int, crop: int, u: float, v: float) -> Tuple[int, int]:
    """Top-left (y, x) of the crop from the spec fractions."""
    return (min(h - crop, int(v * (h - crop + 1))), min(w - crop, int(u * (w - crop + 1))))


def load_crop(root: str, spec: dict, crop: int = CROP) -> Optional[np.ndarray]:
    """The clean crop ``x`` (crop, crop, 3) uint8; falls back to the spec's alternates if an image is bad."""
    for src in [spec["src"]] + list(spec.get("alts", [])):
        try:
            with Image.open(corpus_img_path(root, src)) as im:
                w, h = im.size
                if h < crop or w < crop:
                    continue
                arr = np.asarray(im.convert("RGB"))
            y0, x0 = crop_origin(h, w, crop, spec["u"], spec["v"])
            return np.ascontiguousarray(arr[y0:y0 + crop, x0:x0 + crop])
        except Exception:  # noqa: BLE001 - one bad image must not stop the run
            continue
    return None


def _w_init() -> None:
    try:
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _w_load(args):
    root, spec, crop = args
    try:
        return load_crop(root, spec, crop)
    except Exception:  # noqa: BLE001
        return None


def _w_apply(args):
    img, ops = args
    try:
        return apply_cpu_ops(img, ops)
    except Exception:  # noqa: BLE001
        return None


def make_pool(workers: int, ctx_name: str = "forkserver"):
    """Process pool for the CPU ops, or None for serial execution."""
    if workers <= 1:
        return None
    try:
        return mp.get_context(ctx_name).Pool(workers, initializer=_w_init)
    except Exception as e:  # noqa: BLE001
        warn("pool_failed", err=err_str(e))
        return None


def pmap(pool, fn, items: Sequence, timeout: float = 600.0) -> list:
    """Map on the pool; a hung / dead pool falls back to serial execution."""
    items = list(items)
    if pool is not None and len(items) > 1:
        try:
            return pool.map_async(fn, items, chunksize=max(1, len(items) // (4 * 8))).get(timeout)
        except Exception as e:  # noqa: BLE001
            warn("pool_map_failed", err=err_str(e))
    return [fn(it) for it in items]


def distort_many(xs: Sequence[Optional[np.ndarray]], ops_lists: Sequence[Sequence[dict]], pool=None,
                 gpu_fns: Optional[Dict[str, Callable]] = None, gpu_batch: int = 32,
                 timeout: float = 600.0) -> List[Optional[np.ndarray]]:
    """y_i = ops_i(x_i). GPU ops (always first) run batched in this process, the rest on the pool."""
    gpu_fns = gpu_fns or {}
    n = len(xs)
    base: List[Optional[np.ndarray]] = [x for x in xs]
    for name in GPU_OPS:
        ids = [i for i in range(n) if xs[i] is not None and ops_lists[i] and ops_lists[i][0]["op"] == name]
        if not ids:
            continue
        fn = gpu_fns.get(name)
        for s in range(0, len(ids), gpu_batch):
            part = ids[s:s + gpu_batch]
            try:
                if fn is None:
                    raise RuntimeError(f"no {name} round-trip function")
                out = np.asarray(fn(np.stack([xs[i] for i in part])))
                if out.shape != (len(part),) + xs[part[0]].shape:
                    raise ValueError(f"{name} round trip returned {out.shape}")
                for j, i in enumerate(part):
                    base[i] = out[j]
            except Exception as e:  # noqa: BLE001 - these pairs are dropped, the run goes on
                warn("gpu_op_failed", op=name, n=len(part), err=err_str(e))
                for i in part:
                    base[i] = None
                if is_oom(e):
                    free_cuda_cache()
    jobs, job_ids = [], []
    ys: List[Optional[np.ndarray]] = [None] * n
    for i in range(n):
        if base[i] is None:
            continue
        cpu_ops = [o for o in ops_lists[i] if o["op"] in CPU_OPS]
        if cpu_ops:
            jobs.append((base[i], cpu_ops))
            job_ids.append(i)
        else:
            ys[i] = np.ascontiguousarray(base[i])
    for i, y in zip(job_ids, pmap(pool, _w_apply, jobs, timeout)):
        ys[i] = y
    return ys


def build_chunk(specs: Sequence[dict], root: str, crop: int, pool=None, gpu_fns=None,
                timeout: float = 600.0) -> Tuple[np.ndarray, np.ndarray]:
    """(arr, ok): arr (n, 2|3, crop, crop, 3) uint8 = [x, y] or [ref, a, b]; ok (n,) bool."""
    n = len(specs)
    trip = bool(specs) and specs[0]["split"] == "trip"
    xs = pmap(pool, _w_load, [(root, s, crop) for s in specs], timeout)
    if trip:
        ya = distort_many(xs, [s["ops_a"] for s in specs], pool, gpu_fns, timeout=timeout)
        yb = distort_many(xs, [s["ops_b"] for s in specs], pool, gpu_fns, timeout=timeout)
        cols = [xs, ya, yb]
    else:
        cols = [xs, distort_many(xs, [s["ops"] for s in specs], pool, gpu_fns, timeout=timeout)]
    arr = np.zeros((n, len(cols), crop, crop, 3), np.uint8)
    ok = np.ones(n, bool)
    for i in range(n):
        for c, col in enumerate(cols):
            im = col[i]
            if im is None or im.shape != (crop, crop, 3):
                ok[i] = False
                break
            arr[i, c] = im
    arr[~ok] = 0
    return arr, ok


# ---------------------------------------------------------------------------
# on-disk layout
# ---------------------------------------------------------------------------

def n_for(split: str, n_train: int, n_val: int, n_trip: int) -> int:
    return {"train": n_train, "val": n_val, "trip": n_trip}[split]


def n_chunks(total: int, chunk: int) -> int:
    return (total + chunk - 1) // chunk if total > 0 else 0


def chunk_ids(split: str, c: int, chunk: int, total: int) -> List[int]:
    return list(range(c * chunk, min(total, (c + 1) * chunk)))


def pairs_path(out: str, split: str, c: int) -> str:
    return os.path.join(out, "pairs", f"{split}_{c:05d}.npy")


def ok_path(out: str, split: str, c: int) -> str:
    return os.path.join(out, "pairs", f"{split}_{c:05d}.ok.npy")


def labels_path(out: str, split: str, c: int) -> str:
    return os.path.join(out, f"labels_{split}_{c:05d}.npz")


def _atomic(path: str, writer: Callable) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_npy_atomic(arr: np.ndarray, path: str) -> None:
    _atomic(path, lambda f: np.save(f, arr))


def save_npz_atomic(path: str, **arrays) -> None:
    _atomic(path, lambda f: np.savez(f, **arrays))


def save_pairs(out: str, split: str, c: int, arr: np.ndarray, ok: np.ndarray) -> None:
    os.makedirs(os.path.join(out, "pairs"), exist_ok=True)
    save_npy_atomic(ok.astype(bool), ok_path(out, split, c))      # sidecar first: pairs present => ok present
    save_npy_atomic(arr, pairs_path(out, split, c))


def load_pairs_if_valid(out: str, split: str, c: int, n: int, k: int, crop: int):
    """(mmap array, ok) of a stored chunk, or None when absent / the wrong shape."""
    try:
        arr = np.load(pairs_path(out, split, c), mmap_mode="r")
        ok = np.load(ok_path(out, split, c))
    except Exception:  # noqa: BLE001
        return None
    if arr.shape != (n, k, crop, crop, 3) or arr.dtype != np.uint8 or ok.shape != (n,):
        return None
    return arr, ok.astype(bool)


def cfg_hash(cfg: dict) -> int:
    return zlib.crc32(json.dumps(cfg, sort_keys=True, default=str).encode())


def read_labels(path: str) -> Optional[dict]:
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception:  # noqa: BLE001
        return None


def labels_valid(path: str, h: int, n: int) -> bool:
    z = read_labels(path)
    return z is not None and int(z.get("cfg", -1)) == h and len(z.get("ids", [])) == n


def clean_tmp(out: str) -> int:
    n = 0
    for d in (out, os.path.join(out, "pairs"), os.path.join(out, "upload_stage")):
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


def load_manifest(out: str) -> Optional[dict]:
    try:
        with open(os.path.join(out, "manifest.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_manifest(out: str, man: dict) -> None:
    _atomic(os.path.join(out, "manifest.json"), lambda f: f.write(json.dumps(man, indent=1).encode()))


def manifest_mismatch(old: dict, new: dict) -> List[str]:
    return [k for k in new if k in old and old[k] != new[k]]


# ---------------------------------------------------------------------------
# step B · teacher features (Z-Image transformer, forward hooks)
# ---------------------------------------------------------------------------

class _Captured(Exception):
    """Raised from the last hook to skip the rest of the transformer (blocks after the deepest tap)."""


class TooManyFailures(RuntimeError):
    """MAX_CONSEC_ERRORS failures in a row: the run stops (spec)."""


def vae_consts(vae) -> Tuple[float, float]:
    cfg = getattr(vae, "config", None)
    scale = float(getattr(cfg, "scaling_factor", 1.0) or 1.0)
    shift = getattr(cfg, "shift_factor", 0.0)
    return scale, (0.0 if shift is None else float(shift))


def _place(module) -> Tuple[object, object]:
    try:
        p = next(module.parameters())
        return p.device, p.dtype
    except Exception:  # noqa: BLE001
        return torch.device("cpu"), torch.float32


def u8_to_tensor(a: np.ndarray, device=None, dtype=torch.float32) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(a))
    if device is not None:
        t = t.to(device)
    return (t.permute(0, 3, 1, 2).float() / 127.5 - 1.0).to(dtype)


def tensor_to_u8(t: torch.Tensor) -> np.ndarray:
    a = ((t.float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    return a.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()


def token_distance_map(fa: torch.Tensor, fb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(B,Ht,Wt,D) x2 -> (B,Ht,Wt): squared difference of channel-unit-normalised token vectors, summed over D."""
    fa, fb = fa.float(), fb.float()
    na = fa / fa.norm(dim=-1, keepdim=True).clamp_min(eps)
    nb = fb / fb.norm(dim=-1, keepdim=True).clamp_min(eps)
    return ((na - nb) ** 2).sum(-1)


def noise_for(seed_base: int, pid: int, shape: Sequence[int]) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed_base) + int(pid))
    return torch.randn(tuple(shape), generator=g)


class TeacherLabeler:
    """Teacher perceptual distance from Z-Image transformer features.

    ``transformer`` follows diffusers' ZImageTransformer2DModel: ``.layers`` (the main blocks) and
    ``transformer(list_of_(C,1,h,w), t, list_of_(L,cap_dim), return_dict=False)`` where ``t = 1 - sigma``
    (the pipeline feeds ``(1000 - timestep)/1000``). Block outputs are ``(B, tokens, dim)`` with the
    image tokens first in raster order ((h/2) x (w/2) for patch 2).
    """

    def __init__(self, transformer, vae, cap_feats: torch.Tensor, blocks: Sequence[int] = BLOCKS,
                 sigma: float = SIGMA, patch: int = PATCH, seed: int = 0, early_exit: bool = True,
                 enc_batch: int = 32):
        self.transformer, self.vae = transformer, vae
        self.blocks = tuple(int(b) for b in blocks)
        n_layers = len(transformer.layers)
        bad = [b for b in self.blocks if not 0 <= b < n_layers]
        if bad:
            raise ValueError(f"blocks {bad} outside transformer.layers (len {n_layers})")
        self.sigma, self.patch, self.seed = float(sigma), int(patch), int(seed)
        self.early_exit, self.enc_batch = early_exit, int(enc_batch)
        self.t_dtype = _place(transformer)[1]
        self.t_device = _place(transformer)[0]
        self.cap = cap_feats.to(device=self.t_device, dtype=self.t_dtype)
        self.store: Dict[int, torch.Tensor] = {}
        self.capturing = False
        self.handles = []
        for b in self.blocks:
            self.handles.append(transformer.layers[b].register_forward_hook(self._make_hook(b)))

    def _make_hook(self, b: int):
        last = max(self.blocks)

        def hook(_mod, _inp, out):
            if not self.capturing:
                return None
            self.store[b] = out[0] if isinstance(out, (tuple, list)) else out
            if self.early_exit and b == last:
                raise _Captured()
            return None

        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.store.clear()

    # -- VAE ------------------------------------------------------------------
    def encode_u8(self, x_u8: np.ndarray) -> torch.Tensor:
        """(N,H,W,3) uint8 -> normalised latent (N,C,h,w) float32 on the transformer device."""
        scale, shift = vae_consts(self.vae)
        dev, dt = _place(self.vae)
        outs = []
        with torch.inference_mode():
            for s in range(0, len(x_u8), self.enc_batch):
                x = u8_to_tensor(x_u8[s:s + self.enc_batch], dev, dt)
                z = self.vae.encode(x).latent_dist.mean
                outs.append(((z.float() - shift) * scale).to(self.t_device))
        return torch.cat(outs).clone()

    def vae_roundtrip(self, x_u8: np.ndarray) -> np.ndarray:
        """Teacher VAE encode (mean) -> decode, as uint8 images (the 'teacher VAE round trip' distortion)."""
        scale, shift = vae_consts(self.vae)
        dev, dt = _place(self.vae)
        outs = []
        with torch.inference_mode():
            for s in range(0, len(x_u8), self.enc_batch):
                x = u8_to_tensor(x_u8[s:s + self.enc_batch], dev, dt)
                z = self.vae.encode(x).latent_dist.mean
                dec = self.vae.decode(z)
                img = dec.sample if hasattr(dec, "sample") else dec[0]
                outs.append(tensor_to_u8(img))
        return np.concatenate(outs)

    # -- transformer ---------------------------------------------------------------
    def features(self, z_t: torch.Tensor) -> List[torch.Tensor]:
        """One transformer forward at t = 1 - sigma; returns the image-token grids (B,Ht,Wt,D) of each tapped block."""
        b, _, h, w = z_t.shape
        ht, wt = h // self.patch, w // self.patch
        xs = [z.to(self.t_dtype).unsqueeze(1) for z in z_t]
        t = torch.full((b,), 1.0 - self.sigma, dtype=torch.float32, device=z_t.device)
        caps = [self.cap] * b
        self.store.clear()
        self.capturing = True
        try:
            with torch.inference_mode():
                self.transformer(xs, t, caps, return_dict=False)
        except _Captured:
            pass
        finally:
            self.capturing = False
        out = []
        for blk in self.blocks:
            if blk not in self.store:
                raise RuntimeError(f"hook on block {blk} did not fire")
            hs = self.store[blk][:, : ht * wt]
            out.append(hs.float().reshape(b, ht, wt, -1))
        self.store.clear()
        return out

    def label_pairs(self, x_u8: np.ndarray, y_u8: np.ndarray, ids: Sequence[int]) -> Dict[str, np.ndarray]:
        """Teacher distance of each (x_i, y_i). Same noise (seeded by the pair id) and sigma for both images."""
        n = len(ids)
        z = self.encode_u8(np.concatenate([x_u8, y_u8]))                       # (2n,C,h,w)
        eps = torch.stack([noise_for(self.seed, pid, z.shape[1:]) for pid in ids]).to(z.device)
        eps = torch.cat([eps, eps])
        z_t = (1.0 - self.sigma) * z + self.sigma * eps
        feats = self.features(z_t)
        maps = [token_distance_map(f[:n], f[n:]) for f in feats]               # each (n,Ht,Wt)
        tmap = torch.stack(maps).mean(0)
        dl = torch.stack([m.mean((1, 2)) for m in maps], 1)
        return {"d": tmap.mean((1, 2)).cpu().numpy().astype(np.float32),
                "dl": dl.cpu().numpy().astype(np.float32),
                "map": tmap.cpu().numpy().astype(np.float16)}


def label_resilient(labeler, x_u8: np.ndarray, y_u8: np.ndarray, ids: Sequence[int],
                    state: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Label a batch; on failure bisect so a single bad pair is isolated. Returns (arrays, ok)."""
    state = state if state is not None else {"fail_run": 0, "errors": 0}
    n = len(ids)
    try:
        res = labeler.label_pairs(x_u8, y_u8, ids)
        if not all(np.isfinite(res["d"])):
            raise FloatingPointError("non-finite teacher distance")
        state["fail_run"] = 0
        return res, np.ones(n, bool)
    except Exception as e:  # noqa: BLE001
        state["errors"] += 1
        if is_oom(e):
            free_cuda_cache()
        if n == 1:
            state["fail_run"] += 1
            warn("label_pair_failed", id=int(ids[0]), run=state["fail_run"], err=err_str(e))
            return ({"d": np.full(1, np.nan, np.float32), "dl": np.full((1, 3), np.nan, np.float32),
                     "map": np.zeros((1, 1, 1), np.float16)}, np.zeros(1, bool))
        if state["fail_run"] >= MAX_CONSEC_ERRORS:
            raise TooManyFailures(f"{state['fail_run']} consecutive label failures") from e
        h = n // 2
        ra, oa = label_resilient(labeler, x_u8[:h], y_u8[:h], ids[:h], state)
        rb, ob = label_resilient(labeler, x_u8[h:], y_u8[h:], ids[h:], state)
        ma, mb = ra["map"], rb["map"]
        if ma.shape[1:] != mb.shape[1:]:           # a failed single pair carries a dummy map
            tgt = ma.shape[1:] if oa.any() else mb.shape[1:]
            ma = ma if ma.shape[1:] == tgt else np.zeros((len(ma),) + tgt, np.float16)
            mb = mb if mb.shape[1:] == tgt else np.zeros((len(mb),) + tgt, np.float16)
        return ({"d": np.concatenate([ra["d"], rb["d"]]), "dl": np.concatenate([ra["dl"], rb["dl"]]),
                 "map": np.concatenate([ma, mb])}, np.concatenate([oa, ob]))


# ---------------------------------------------------------------------------
# step A+B driver
# ---------------------------------------------------------------------------

class LabelJob:
    """Builds the pair chunks and labels them with the teacher; chunk-granular and resumable."""

    def __init__(self, args, out: str, root: str, train_pool: Sequence[int], val_pool: Sequence[int],
                 labeler, gpu_fns=None, pool=None, now: Callable[[], float] = time.time,
                 stop: Optional[threading.Event] = None, label_deadline: Optional[float] = None):
        self.a, self.out, self.root = args, out, root
        self.train_pool, self.val_pool = list(train_pool), list(val_pool)
        self.labeler, self.gpu_fns, self.pool = labeler, gpu_fns or {}, pool
        self.now = now
        self.stop = stop or threading.Event()
        self.label_deadline = label_deadline
        self.gpu_ops = tuple(g for g in args.gpu_ops.split(",") if g)
        self.cfg = {"sigma": args.sigma, "blocks": list(args.blocks), "seed": args.seed, "crop": args.crop,
                    "patch": PATCH, "teacher": ZIMAGE_ID}
        self.h = cfg_hash(self.cfg)
        self.state = {"fail_run": 0, "errors": 0}
        self.done_pairs = 0
        self.t0 = now()
        self.last_progress = self.t0
        self.stats = {"chunks": 0, "skipped": 0, "pairs_ok": 0, "pairs_bad": 0}

    def tasks(self) -> List[Tuple[str, int]]:
        a = self.a
        out: List[Tuple[str, int]] = []
        for split in ("val", "trip", "train"):         # val first so evaluation is possible with a partial train set
            total = n_for(split, a.n_train, a.n_val, a.n_trip)
            out += [(split, c) for c in range(n_chunks(total, a.chunk))]
        return out

    def _stop_reason(self) -> Optional[str]:
        if self.stop.is_set():
            return "signal"
        if self.label_deadline is not None and self.now() >= self.label_deadline:
            return "label_budget"
        return None

    def _progress(self, split: str, c: int, force: bool = False) -> None:
        t = self.now()
        if force or t - self.last_progress >= self.a.progress_every_s:
            dt = max(1e-6, t - self.t0)
            log({"phase": "label", "split": split, "chunk": c, "pairs": self.done_pairs,
                 "pairs_per_s": round(self.done_pairs / dt, 2), "errors": self.state["errors"],
                 "vram_GB": vram_gb(), "free_GB": round(free_disk_gb(self.out), 1)})
            self.last_progress = t

    def process_chunk(self, split: str, c: int) -> str:
        """'done' | 'skipped' | 'stop' | 'disk' for one chunk."""
        a = self.a
        total = n_for(split, a.n_train, a.n_val, a.n_trip)
        ids = chunk_ids(split, c, a.chunk, total)
        n, k = len(ids), (3 if split == "trip" else 2)
        lp = labels_path(self.out, split, c)
        if labels_valid(lp, self.h, n):
            self.stats["skipped"] += 1
            return "skipped"
        stored = load_pairs_if_valid(self.out, split, c, n, k, a.crop)
        if stored is None:
            if free_disk_gb(self.out) < a.min_free_gb:
                log({"stop": "disk", "free_GB": round(free_disk_gb(self.out), 1)})
                return "disk"
            pool = self.val_pool if split in ("val", "trip") else self.train_pool
            specs = [make_spec(a.seed, split, i, pool, self.gpu_ops) for i in ids]
            arr, ok = build_chunk(specs, self.root, a.crop, self.pool, self.gpu_fns)
            save_pairs(self.out, split, c, arr, ok)
            stored = (arr, ok)
        arr, ok = stored
        pids = np.array([ID_OFFSET[split] + i for i in ids], np.int64)
        res = self._label_chunk(split, c, arr, ok, pids)
        if res is None:
            return "stop"
        d, dl, tmap, ok2 = res
        arrays = {"ids": pids, "ok": ok2, "d": d, "dl": dl, "cfg": np.array(self.h, np.int64)}
        if tmap is not None:
            arrays["map"] = tmap
        save_npz_atomic(lp, **arrays)
        self.stats["chunks"] += 1
        self.stats["pairs_ok"] += int(ok2.sum())
        self.stats["pairs_bad"] += int((~ok2).sum())
        return "done"

    def _label_chunk(self, split, c, arr, ok, pids):
        a = self.a
        n = len(pids)
        trip = split == "trip"
        d = np.full((n, 2) if trip else (n,), np.nan, np.float32)
        dl = np.full((n, 2, 3) if trip else (n, 3), np.nan, np.float32)
        msz = a.crop // 16
        tmap = None if trip else np.zeros((n, msz, msz), np.float16)
        ok2 = ok.copy()
        for s in range(0, n, a.label_batch):
            if self._stop_reason():
                return None
            sel = np.arange(s, min(n, s + a.label_batch))
            sel = sel[ok[sel]]
            if len(sel):
                ref = np.ascontiguousarray(arr[sel, 0])
                if trip:
                    ra, oa = label_resilient(self.labeler, ref, np.ascontiguousarray(arr[sel, 1]), pids[sel], self.state)
                    rb, ob = label_resilient(self.labeler, ref, np.ascontiguousarray(arr[sel, 2]), pids[sel], self.state)
                    d[sel, 0], d[sel, 1] = ra["d"], rb["d"]
                    dl[sel, 0], dl[sel, 1] = ra["dl"], rb["dl"]
                    ok2[sel] = oa & ob
                else:
                    res, o = label_resilient(self.labeler, ref, np.ascontiguousarray(arr[sel, 1]), pids[sel], self.state)
                    d[sel], dl[sel] = res["d"], res["dl"]
                    good = sel[o]
                    if len(good):
                        tmap[good] = res["map"][o]
                    ok2[sel] = o
                self.done_pairs += int(len(sel))
            self._progress(split, c)
            if self.state["fail_run"] >= MAX_CONSEC_ERRORS:
                raise TooManyFailures(f"{self.state['fail_run']} consecutive label failures")
        return d, dl, tmap, ok2

    def run(self) -> dict:
        reason = "complete"
        chunk_fail = 0
        for split, c in self.tasks():
            r = self._stop_reason()
            if r:
                reason = r
                break
            try:
                status = self.process_chunk(split, c)
            except TooManyFailures as e:
                reason = "fatal"
                log({"fatal": err_str(e)}, marker=MARK + "_DONE")
                break
            except Exception as e:  # noqa: BLE001
                warn("chunk_failed", split=split, chunk=c, err=err_str(e))
                chunk_fail += 1
                if chunk_fail >= MAX_CONSEC_ERRORS:
                    reason = "fatal"
                    break
                continue
            chunk_fail = 0
            if status == "stop":
                reason = self._stop_reason() or "stop"
                break
            if status == "disk":
                reason = "disk"
                break
            self._progress(split, c)
        self._progress("-", -1, force=True)
        return {"reason": reason, **self.stats, "errors": self.state["errors"]}


# ---------------------------------------------------------------------------
# step C · the student
# ---------------------------------------------------------------------------

DEFAULT_WIDTHS = (32, 64, 128, 192, 256)
DEFAULT_EXTRA = (1, 2, 2, 3, 3)


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


class TLPIPSStudent(nn.Module):
    """Five stride-2 conv stages; per stage: unit-normalise channels, squared difference, learned
    non-negative channel weights (softplus), spatial mean. Distance = sum over stages; the distance
    map is the sum of the per-stage maps resampled to ``crop/16``. Symmetric, zero on identical inputs."""

    def __init__(self, widths: Sequence[int] = DEFAULT_WIDTHS, extra: Sequence[int] = DEFAULT_EXTRA,
                 map_div: int = 16):
        super().__init__()
        self.widths, self.extra, self.map_div = tuple(widths), tuple(extra), int(map_div)
        stages, cin = [], 3
        for w, e in zip(self.widths, self.extra):
            layers: List[nn.Module] = [nn.Conv2d(cin, w, 3, stride=2, padding=1), _gn(w), nn.SiLU()]
            for _ in range(e):
                layers += [nn.Conv2d(w, w, 3, padding=1), _gn(w), nn.SiLU()]
            stages.append(nn.Sequential(*layers))
            cin = w
        self.stages = nn.ModuleList(stages)
        self.lin = nn.ParameterList([nn.Parameter(torch.full((w,), _inv_softplus(1.0 / w))) for w in self.widths])

    def config(self) -> dict:
        return {"widths": list(self.widths), "extra": list(self.extra), "map_div": self.map_div}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = x.shape[0]
        h = torch.cat([x, y])
        msz = x.shape[-1] // self.map_div
        scalars, total_map = [], 0.0
        for stage, lin in zip(self.stages, self.lin):
            h = stage(h)
            f = h.float()
            f = f / (f.norm(dim=1, keepdim=True) + 1e-6)
            diff = (f[:n] - f[n:]) ** 2                                   # (n,C,h,w)
            m = torch.einsum("nchw,c->nhw", diff, F.softplus(lin))          # (n,h,w)
            scalars.append(m.mean((1, 2)))
            m4 = m[:, None]
            if m4.shape[-1] > msz:
                m4 = F.adaptive_avg_pool2d(m4, msz)
            elif m4.shape[-1] < msz:
                m4 = F.interpolate(m4, size=(msz, msz), mode="bilinear", align_corners=False)
            total_map = total_map + m4[:, 0]
        st = torch.stack(scalars, 1)
        return {"d": st.sum(1), "stage": st, "map": total_map}


def rank_loss(ls: torch.Tensor, lt: torch.Tensor, margin: float = 0.1, min_gap: float = 0.2) -> torch.Tensor:
    """Margin ranking over all pairs-of-pairs in the batch (log-distance domain), teacher order as target."""
    dt = lt[:, None] - lt[None, :]
    ds = ls[:, None] - ls[None, :]
    mask = (dt.abs() > min_gap) & torch.triu(torch.ones_like(dt, dtype=torch.bool), diagonal=1)
    if not bool(mask.any()):
        return ls.sum() * 0.0
    return F.relu(margin - torch.sign(dt) * ds)[mask].mean()


def compute_loss(out: Dict[str, torch.Tensor], d_t: torch.Tensor, map_t: Optional[torch.Tensor],
                 rank_w: float = 0.5, map_w: float = 0.0) -> Dict[str, torch.Tensor]:
    ls = torch.log(out["d"].float() + LOG_EPS)
    lt = torch.log(d_t.float() + LOG_EPS)
    l_mse = F.mse_loss(ls, lt)
    l_rank = rank_loss(ls, lt)
    loss = l_mse + rank_w * l_rank
    l_map = torch.zeros((), device=ls.device)
    if map_w > 0 and map_t is not None and out["map"].shape == map_t.shape:
        l_map = F.mse_loss(torch.log(out["map"].float() + LOG_EPS), torch.log(map_t.float() + LOG_EPS))
        loss = loss + map_w * l_map
    return {"loss": loss, "l_mse": l_mse, "l_rank": l_rank, "l_map": l_map}


def lr_at(step: int, base_lr: float = 1e-3, warmup: int = 500, total: int = TOTAL_STEPS,
          min_frac: float = 0.01) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


def clip_and_step(model: nn.Module, opt: torch.optim.Optimizer, clip: float = 1.0) -> bool:
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not bool(torch.isfinite(gn)):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


# ---------------------------------------------------------------------------
# step C · label store, batches, baselines, evaluation
# ---------------------------------------------------------------------------

class SplitData:
    """All labelled, usable items of one split (labels in RAM, images through mmap)."""

    def __init__(self, out: str, split: str, h: int, crop: int):
        self.out, self.split, self.crop = out, split, crop
        self.trip = split == "trip"
        self.k = 3 if self.trip else 2
        cs, rows, d, maps, ids = [], [], [], [], []
        pat = re.compile(rf"labels_{split}_(\d+)\.npz")
        try:
            found = sorted(int(m.group(1)) for m in (pat.fullmatch(n) for n in os.listdir(out)) if m)
        except FileNotFoundError:
            found = []
        for c in found:
            z = read_labels(labels_path(out, split, c))
            if z is None or int(z.get("cfg", -1)) != h:
                continue
            ok = z["ok"].astype(bool) & np.all(np.isfinite(z["d"]).reshape(len(z["ok"]), -1), axis=1)
            sel = np.nonzero(ok)[0]
            cs.append(np.full(len(sel), c, np.int32))
            rows.append(sel.astype(np.int32))
            d.append(z["d"][sel])
            ids.append(z["ids"][sel])
            if not self.trip and "map" in z:
                maps.append(z["map"][sel])
        self.chunk = np.concatenate(cs) if cs else np.zeros(0, np.int32)
        self.row = np.concatenate(rows) if rows else np.zeros(0, np.int32)
        self.d = np.concatenate(d) if d else np.zeros((0, 2) if self.trip else (0,), np.float32)
        self.ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        self.map = np.concatenate(maps) if maps else None
        self._mm: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.row)

    def _arr(self, c: int):
        mm = self._mm.get(c)
        if mm is None:
            mm = np.load(pairs_path(self.out, self.split, c), mmap_mode="r")
            self._mm[c] = mm
        return mm

    def images(self, idx: Sequence[int]) -> np.ndarray:
        """(len(idx), k, crop, crop, 3) uint8 copies."""
        out = np.empty((len(idx), self.k, self.crop, self.crop, 3), np.uint8)
        for j, i in enumerate(idx):
            out[j] = self._arr(int(self.chunk[i]))[int(self.row[i])]
        return out

    def drop_mmaps(self) -> None:
        self._mm.clear()


class BatchLoader:
    """Background thread that assembles random training batches (uint8) from the labelled chunks."""

    def __init__(self, data: SplitData, batch: int, seed: int, prefetch: int = 6):
        self.data, self.batch, self.seed = data, batch, seed
        self.q: "queue.Queue" = queue.Queue(maxsize=prefetch)
        self.stop = threading.Event()
        self.errors = 0
        self.epoch = 0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self) -> None:
        n = len(self.data)
        while not self.stop.is_set():
            rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch)
            perm = rng.permutation(n)
            self.epoch += 1
            starts = list(range(0, n - self.batch + 1, self.batch)) if n >= self.batch else [0]
            for s in starts:
                if self.stop.is_set():
                    return
                idx = perm[s:s + self.batch] if n >= self.batch else rng.choice(n, self.batch, replace=True)
                try:
                    item = (self.data.images(idx), self.data.d[idx].copy(),
                            None if self.data.map is None else self.data.map[idx].copy())
                except Exception as e:  # noqa: BLE001 - a bad chunk file: skip this batch
                    self.errors += 1
                    if self.errors <= 10:
                        warn("batch_failed", err=err_str(e))
                    continue
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue

    def next(self, timeout: float = 120.0):
        return self.q.get(timeout=timeout)

    def close(self) -> None:
        self.stop.set()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3:
        return float("nan")

    def ranks(v):
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        return (csum - (cnt - 1) / 2.0)[inv]

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pixel_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared difference on [0,1] intensities (inputs in [-1,1])."""
    return (((x - y) / 2.0) ** 2).mean((1, 2, 3))


def ssim_dist(x: torch.Tensor, y: torch.Tensor, win: int = 7) -> torch.Tensor:
    """1 - mean SSIM (uniform window, luma), a cheap SSIM approximation."""
    def gray(t):
        t = (t + 1.0) / 2.0
        return (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2])[:, None]

    gx, gy = gray(x), gray(y)
    pool = lambda t: F.avg_pool2d(t, win, stride=1, padding=win // 2, count_include_pad=False)  # noqa: E731
    mx, my = pool(gx), pool(gy)
    vx, vy, cxy = pool(gx * gx) - mx * mx, pool(gy * gy) - my * my, pool(gx * gy) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return 1.0 - s.mean((1, 2, 3))


@torch.no_grad()
def distances_pairs(fn: Callable, data: SplitData, device, batch: int = 64, amp: bool = False) -> np.ndarray:
    out = []
    for s in range(0, len(data), batch):
        im = data.images(range(s, min(len(data), s + batch)))
        x = u8_to_tensor(im[:, 0], device)
        y = u8_to_tensor(im[:, 1], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(fn(x, y).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


@torch.no_grad()
def distances_trips(fn: Callable, data: SplitData, device, batch: int = 32, amp: bool = False) -> np.ndarray:
    out = []
    for s in range(0, len(data), batch):
        im = data.images(range(s, min(len(data), s + batch)))
        r, a, b = (u8_to_tensor(im[:, j], device) for j in range(3))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(torch.stack([fn(r, a).float(), fn(r, b).float()], 1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2), np.float32)


def afc_agreement(ds: np.ndarray, dt: np.ndarray, clear_gap: float = 0.2) -> Tuple[float, float]:
    """2AFC: fraction of triplets where the student picks the same closer image as the teacher.
    Second value restricts to triplets whose teacher log-distances differ by more than ``clear_gap``."""
    if len(dt) == 0:
        return float("nan"), float("nan")
    same = (ds[:, 0] < ds[:, 1]) == (dt[:, 0] < dt[:, 1])
    gap = np.abs(np.log(dt[:, 0] + LOG_EPS) - np.log(dt[:, 1] + LOG_EPS))
    clear = gap > clear_gap
    return float(same.mean()), (float(same[clear].mean()) if clear.any() else float("nan"))


def map_corr(sm: np.ndarray, tm: np.ndarray) -> float:
    a = np.log(np.asarray(sm, np.float64).reshape(len(sm), -1) + LOG_EPS)
    b = np.log(np.asarray(tm, np.float64).reshape(len(tm), -1) + LOG_EPS)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = den > 1e-12
    return float(((a * b).sum(1)[ok] / den[ok]).mean()) if ok.any() else float("nan")


def compute_baselines(val: SplitData, trip: SplitData, device) -> dict:
    res = {}
    for name, fn in (("l2", pixel_l2), ("ssim", ssim_dist)):
        if len(val):
            res[f"{name}_sp"] = spearman(distances_pairs(fn, val, device), val.d)
        if len(trip):
            dt = distances_trips(fn, trip, device)
            res[f"{name}_afc"] = afc_agreement(dt, trip.d)[0]
    res["n_val"], res["n_trip"] = len(val), len(trip)
    return res


@torch.no_grad()
def evaluate(model: nn.Module, val: SplitData, trip: SplitData, device, amp: bool = False,
             baselines: Optional[dict] = None, map_n: int = 300) -> Tuple[Optional[dict], Optional[tuple]]:
    """Metrics against the teacher + (teacher_d, student_d) for the scatter plot."""
    was = model.training
    model.eval()
    m: dict = {}
    scatter = None
    try:
        if len(val):
            ds = distances_pairs(lambda x, y: model(x, y)["d"], val, device, amp=amp)
            m["spearman"] = spearman(ds, val.d)
            m["log_pearson"] = (float(np.corrcoef(np.log(ds + LOG_EPS), np.log(val.d + LOG_EPS))[0, 1])
                                if len(ds) > 2 else float("nan"))
            m["n_val"] = int(len(val))
            scatter = (val.d.copy(), ds)
            if val.map is not None and len(val):
                k = min(map_n, len(val))
                im = val.images(range(k))
                x, y = u8_to_tensor(im[:, 0], device), u8_to_tensor(im[:, 1], device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    sm = model(x, y)["map"].float().cpu().numpy()
                if sm.shape[1:] == val.map.shape[1:]:
                    m["map_corr"] = map_corr(sm, val.map[:k])
        if len(trip):
            dts = distances_trips(lambda x, y: model(x, y)["d"], trip, device, amp=amp)
            m["afc"], m["afc_clear"] = afc_agreement(dts, trip.d)
            m["n_trip"] = int(len(trip))
    finally:
        model.train(was)
    if baselines:
        for k, v in baselines.items():
            if k.startswith(("l2_", "ssim_")):
                m["base_" + k] = v
    return (m or None), scatter


def draw_scatter(dt: np.ndarray, ds: np.ndarray, path: str, size: int = 640) -> None:
    """Teacher distance (x) vs student distance (y), log-log, with the y = x diagonal."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    pad = 40
    a = np.log10(np.maximum(np.asarray(dt, np.float64), 1e-6))
    b = np.log10(np.maximum(np.asarray(ds, np.float64), 1e-6))
    lo = float(min(a.min(), b.min())) if len(a) else -4.0
    hi = float(max(a.max(), b.max())) if len(a) else 0.0
    if hi - lo < 1e-6:
        hi = lo + 1.0

    def px(v):
        return pad + (v - lo) / (hi - lo) * (size - 2 * pad)

    for t in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
        if lo <= t <= hi:
            p = px(t)
            dr.line([(p, pad), (p, size - pad)], fill=(230, 230, 230))
            dr.line([(pad, size - p), (size - pad, size - p)], fill=(230, 230, 230))
            dr.text((p - 10, size - pad + 6), f"1e{t}", fill=(90, 90, 90))
    dr.rectangle([pad, pad, size - pad, size - pad], outline=(120, 120, 120))
    dr.line([(px(lo), size - px(lo)), (px(hi), size - px(hi))], fill=(200, 60, 60))
    step = max(1, len(a) // 4000)
    for x, y in zip(a[::step], b[::step]):
        cx, cy = px(float(x)), size - px(float(y))
        dr.ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill=(40, 90, 200))
    dr.text((pad, 10), "teacher distance (x) vs student distance (y), log10", fill=(0, 0, 0))
    tmp = path + ".tmp"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# checkpoints and upload
# ---------------------------------------------------------------------------

def save_atomic(obj: dict, path: str) -> None:
    _atomic(path, lambda f: torch.save(obj, f))


def load_ckpt(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - our own file
            obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "model" not in obj or "step" not in obj:
            raise ValueError("missing keys")
        return obj
    except Exception as e:  # noqa: BLE001
        warn("ckpt_unreadable", path=os.path.basename(path), err=err_str(e))
        return None


def stage_upload(out: str) -> Tuple[str, List[str]]:
    stage = os.path.join(out, "upload_stage")
    os.makedirs(stage, exist_ok=True)
    names = []
    for name in ("ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl", "scatter.png", "baselines.json", "manifest.json"):
        src = os.path.join(out, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(stage, name)
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
        names.append(name)
    return stage, names


def make_uploader(token: Optional[str], repo_id: str = REPO_ID) -> Optional[Callable]:
    """``upload(folder, remote_dir, allow=None)`` using huggingface_hub (lazy import), or None without a token."""
    if not token:
        return None

    def upload(folder: str, remote_dir: str, allow: Optional[Sequence[str]] = None) -> None:
        from huggingface_hub import HfApi
        kw = {"allow_patterns": list(allow)} if allow else {}
        HfApi(token=token).upload_folder(folder_path=folder, path_in_repo=remote_dir, repo_id=repo_id,
                                         repo_type=REPO_TYPE, commit_message="day1 m6 t-lpips-s", **kw)

    return upload


def upload_with_retry(uploader: Callable, folder: str, remote: str, allow=None, tries: int = 3,
                      sleep: Callable[[float], None] = time.sleep) -> bool:
    for k in range(tries):
        try:
            if allow:
                uploader(folder, remote, allow)
            else:
                uploader(folder, remote)
            return True
        except Exception as e:  # noqa: BLE001 - upload trouble is a warning, never fatal
            warn("upload_failed", attempt=k + 1, err=err_str(e, 300))
            if k + 1 < tries:
                sleep(5 * (3 ** k))
    return False


# ---------------------------------------------------------------------------
# assets: night1 corpus + TAE checkpoint (HF vault fallback), TAE module, teacher
# ---------------------------------------------------------------------------

_SHARD_MEMBER = re.compile(r"(img/\d{8}\.webp|lat/\d{8}\.npy|meta/\d{8}\.json)")


def extract_shard(tar_path: str, root: str) -> int:
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
    if not token:
        return None, None

    def lister(prefix: str) -> List[str]:
        from huggingface_hub import HfApi
        return [n for n in HfApi(token=token).list_repo_files(repo_id, repo_type=REPO_TYPE) if n.startswith(prefix)]

    def fetcher(remote: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id, remote, repo_type=REPO_TYPE, token=token,
                               local_dir=os.path.join(root, "_hf_dl"))

    return lister, fetcher


def ensure_assets(root: str, tae_path: str, min_images: int, lister, fetcher, need_tae: bool = True) -> dict:
    """Fetch the M1 checkpoint and the corpus shards from the vault when the box does not have them."""
    res = {"tae": os.path.isfile(tae_path), "images": len(list_corpus(root)), "shards": 0}
    if need_tae and not res["tae"] and fetcher is not None:
        try:
            src = fetcher(TAE_REMOTE)
            os.makedirs(os.path.dirname(tae_path), exist_ok=True)
            shutil.copyfile(src, tae_path + ".part")
            os.replace(tae_path + ".part", tae_path)
            res["tae"] = True
        except Exception as e:  # noqa: BLE001
            warn("tae_fetch_failed", err=err_str(e, 300))
    if res["images"] < min_images and lister is not None and fetcher is not None:
        try:
            names = sorted(n for n in lister(CORPUS_PREFIX)
                           if re.fullmatch(re.escape(CORPUS_PREFIX) + r"gen_\d{4,}\.tar", n))
        except Exception as e:  # noqa: BLE001
            warn("corpus_list_failed", err=err_str(e, 300))
            names = []
        for name in names:
            try:
                p = fetcher(name)
                extract_shard(p, root)
                res["shards"] += 1
                try:
                    os.remove(p)
                except OSError:
                    pass
            except Exception as e:  # noqa: BLE001
                warn("shard_fetch_failed", name=os.path.basename(name), err=err_str(e, 300))
        res["images"] = len(list_corpus(root))
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
    import tae_train  # noqa: F401  (PYTHONPATH)
    return sys.modules["tae_train"]


def load_tae(path: str, device) -> nn.Module:
    """Frozen M1 TAE-T (encoder + decoder) from ``ckpt_best.pt``."""
    ck = load_ckpt(path)
    if ck is None or "config" not in ck:
        raise FileNotFoundError(f"no usable TAE checkpoint at {path}")
    mod = import_tae_module()
    cfg = ck["config"]
    tae = mod.TAE(int(cfg["latent_channels"]), tuple(int(w) for w in cfg["widths"]),
                  int(cfg.get("blocks", getattr(mod, "BLOCKS_PER_STAGE", 2))))
    tae.load_state_dict(ck["model"])
    tae = tae.to(device).eval()
    for p in tae.parameters():
        p.requires_grad_(False)
    return tae


def make_tae_roundtrip(tae: nn.Module, device, batch: int = 32) -> Callable[[np.ndarray], np.ndarray]:
    """uint8 (N,H,W,3) -> D(E(x)) as uint8; the M1 'compressor' distortion."""
    def fn(x_u8: np.ndarray) -> np.ndarray:
        outs = []
        with torch.inference_mode():
            for s in range(0, len(x_u8), batch):
                x = u8_to_tensor(x_u8[s:s + batch], device)
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                    enabled=torch.device(device).type == "cuda"):
                    y = tae.dec(tae.enc(x))
                outs.append(tensor_to_u8(y))
        return np.concatenate(outs)

    return fn


def load_teacher(device: str, cache_path: str, blocks: Sequence[int], sigma: float, seed: int,
                 model_id: str = ZIMAGE_ID):
    """Z-Image transformer + VAE (bf16) and the cached empty-prompt embedding. Lazy imports (diffusers).

    The first run loads the whole pipeline to encode the empty prompt once; the text encoder is then
    dropped (and the embedding cached on disk so later runs skip it entirely).
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from huggingface_hub import snapshot_download

    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    path = snapshot_download(model_id, ignore_patterns=["assets/*"])
    cap = None
    if os.path.isfile(cache_path):
        try:
            cap = torch.load(cache_path, map_location="cpu", weights_only=True)["cap"]
        except Exception as e:  # noqa: BLE001
            warn("cap_cache_unreadable", err=err_str(e))
    if cap is None:
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=False)
        pipe.to(device)
        with torch.inference_mode():
            embeds, _ = pipe.encode_prompt("", device=torch.device(device), do_classifier_free_guidance=False)
        cap = embeds[0].detach().to("cpu", dtype).clone()
        save_atomic({"cap": cap}, cache_path)
        transformer, vae = pipe.transformer, pipe.vae
        pipe.text_encoder = None
        del pipe
    else:
        from diffusers import AutoencoderKL
        from diffusers.models import ZImageTransformer2DModel
        transformer = ZImageTransformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype).to(device)
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype).to(device)
    transformer.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    gc.collect()
    free_cuda_cache()
    return TeacherLabeler(transformer, vae, cap, blocks, sigma, PATCH, seed)


# ---------------------------------------------------------------------------
# step C · trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args, out: str, h: int, uploader=None, now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, stop: Optional[threading.Event] = None,
                 deadline: Optional[float] = None, device=None):
        self.a, self.out, self.h = args, out, h
        self.uploader, self.now, self.sleep = uploader, now, sleep
        self.stop = stop or threading.Event()
        self.deadline = deadline
        self.device = device if device is not None else torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = {"on": True, "off": False}.get(args.amp, self.device.type == "cuda")
        self.step = 0
        self.best = -math.inf
        self.skipped = 0
        self.bad_run = 0
        self.model: Optional[TLPIPSStudent] = None
        self.opt = None
        self.upload_thread: Optional[threading.Thread] = None
        self.baselines: Optional[dict] = None
        self.last_loss = float("nan")

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
        return {"version": 1, "step": self.step, "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "config": self.model.config(), "best_spearman": self.best, "cfg_hash": self.h,
                "saved_at": time.time()}

    def _restore(self, ck: dict) -> bool:
        try:
            self.model.load_state_dict(ck["model"])
            if "opt" in ck:
                self.opt.load_state_dict(ck["opt"])
            self.step = int(ck["step"])
            bs = ck.get("best_spearman")
            self.best = float(bs) if bs is not None and math.isfinite(float(bs)) else -math.inf
            return True
        except Exception as e:  # noqa: BLE001
            warn("ckpt_restore_failed", err=err_str(e))
            return False

    def _load_baselines(self, val: SplitData, trip: SplitData) -> Optional[dict]:
        path = os.path.join(self.out, "baselines.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                b = json.load(f)
            if b.get("n_val") == len(val) and b.get("n_trip") == len(trip):
                return b
        except (OSError, ValueError):
            pass
        try:
            b = compute_baselines(val, trip, self.device)
            _atomic(path, lambda f: f.write(json.dumps(_round_floats(b, 5)).encode()))
            return b
        except Exception as e:  # noqa: BLE001
            warn("baselines_failed", err=err_str(e))
            return None

    def checkpoint_and_eval(self, val: SplitData, trip: SplitData, final: bool = False) -> Optional[dict]:
        metrics, improved = None, False
        try:
            if self.baselines is None and (len(val) or len(trip)):
                self.baselines = self._load_baselines(val, trip)
            metrics, scatter = evaluate(self.model, val, trip, self.device, self.amp, self.baselines)
            if metrics is not None:
                sp = metrics.get("spearman", float("nan"))
                improved = bool(math.isfinite(sp) and sp > self.best)
                if improved:
                    self.best = sp
                if scatter is not None:
                    try:
                        draw_scatter(scatter[0], scatter[1], os.path.join(self.out, "scatter.png"))
                    except Exception as e:  # noqa: BLE001
                        warn("scatter_failed", err=err_str(e))
                rec = {"step": self.step, "time": round(time.time(), 1), "loss": self.last_loss,
                       "best": improved, "best_spearman": self.best, **metrics}
                with open(os.path.join(self.out, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(_round_floats(rec, 5)) + "\n")
                log({"step": self.step, **{k: v for k, v in metrics.items() if not k.startswith("n_")},
                     "best": round(self.best, 4), "improved": improved, "final": final}, marker=MARK + "_EVAL")
        except Exception as e:  # noqa: BLE001 - eval trouble must not lose the checkpoint
            warn("eval_failed", err=err_str(e, 300))
        try:
            obj = self._ckpt_obj()
            save_atomic(obj, os.path.join(self.out, "ckpt_last.pt"))
            if improved or (final and not os.path.exists(os.path.join(self.out, "ckpt_best.pt"))):
                save_atomic(obj, os.path.join(self.out, "ckpt_best.pt"))
        except Exception as e:  # noqa: BLE001
            warn("ckpt_save_failed", err=err_str(e, 300))
        self._start_upload(final)
        return metrics

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

    def _train_step(self, batch) -> bool:
        imgs, d_t, map_t = batch
        dev = self.device
        im = torch.from_numpy(imgs).to(dev, non_blocking=True)                       # (B,2,H,W,3) uint8
        x = (im[:, 0].permute(0, 3, 1, 2).float() / 127.5 - 1.0)
        y = (im[:, 1].permute(0, 3, 1, 2).float() / 127.5 - 1.0)
        d = torch.from_numpy(d_t).to(dev).float()
        mp_ = None if map_t is None else torch.from_numpy(map_t).to(dev).float()
        g = torch.Generator(device="cpu").manual_seed(self.a.seed * 7919 + self.step)
        flip = torch.rand(x.shape[0], 2, generator=g) < 0.5                            # (hflip, vflip) per pair
        swap = (torch.rand(x.shape[0], generator=g) < 0.5).to(dev)
        fh, fv = flip[:, 0].to(dev), flip[:, 1].to(dev)
        x = torch.where(fh[:, None, None, None], x.flip(3), x)
        y = torch.where(fh[:, None, None, None], y.flip(3), y)
        x = torch.where(fv[:, None, None, None], x.flip(2), x)
        y = torch.where(fv[:, None, None, None], y.flip(2), y)
        if mp_ is not None:
            mp_ = torch.where(fh[:, None, None], mp_.flip(2), mp_)
            mp_ = torch.where(fv[:, None, None], mp_.flip(1), mp_)
        x, y = torch.where(swap[:, None, None, None], y, x), torch.where(swap[:, None, None, None], x, y)
        lr = lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps)
        for grp in self.opt.param_groups:
            grp["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(x, y)
        losses = compute_loss(out, d, mp_, self.a.rank_w, self.a.map_w)
        if not bool(torch.isfinite(losses["loss"])):
            self.skipped += 1
            self.bad_run += 1
            return False
        losses["loss"].backward()
        if not clip_and_step(self.model, self.opt, 1.0):
            self.skipped += 1
            self.bad_run += 1
            return False
        self.bad_run = 0
        self.acc += torch.stack([losses["loss"].detach(), losses["l_mse"].detach(), losses["l_rank"].detach()])
        self.acc_n += 1
        self.seen += x.shape[0]
        self.step += 1
        return True

    def _log_progress(self, t: float) -> None:
        dt = max(1e-6, t - self.log_t)
        if self.acc_n:
            loss, l_mse, l_rank = [v / self.acc_n for v in self.acc.tolist()]
            self.last_loss = loss
        else:
            loss = l_mse = l_rank = float("nan")
        log({"phase": "train", "step": self.step, "loss": round(loss, 5), "mse": round(l_mse, 5),
             "rank": round(l_rank, 5), "pairs_per_s": round(self.seen / dt, 1),
             "lr": round(lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps), 6),
             "skipped": self.skipped, "vram_GB": vram_gb()})
        self.acc.zero_()
        self.acc_n = 0
        self.seen = 0
        self.log_t = t

    def run(self) -> dict:
        a = self.a
        train = SplitData(self.out, "train", self.h, a.crop)
        val = SplitData(self.out, "val", self.h, a.crop)
        trip = SplitData(self.out, "trip", self.h, a.crop)
        log({"phase": "train_data", "train": len(train), "val": len(val), "trip": len(trip)})
        if len(train) == 0:
            log({"stop": "no_train_labels"}, marker=MARK + "_DONE")
            return {"status": "no_data", "step": 0}
        torch.manual_seed(a.seed)
        self.model = TLPIPSStudent().to(self.device)
        n_params = count_params(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=a.wd)
        resumed = False
        for name in ("ckpt_last.pt", "ckpt_best.pt"):
            ck = load_ckpt(os.path.join(self.out, name))
            if ck is not None and ck.get("config", {}).get("widths") == list(self.model.widths):
                resumed = self._restore(ck)
                if resumed:
                    break
        start_step = self.step
        log({"params": n_params, "in_range": PARAM_MIN <= n_params <= PARAM_MAX, "resumed": resumed,
             "step": self.step, "device": str(self.device), "amp": self.amp, "batch": a.batch})
        self.acc = torch.zeros(3, device=self.device)
        self.acc_n = 0
        self.seen = 0
        t0 = self.now()
        self.log_t = t0
        next_log = t0 + a.log_every_s
        next_eval = t0 + min(a.first_eval_s, a.eval_every_s)
        loader = BatchLoader(train, a.batch, a.seed * 1009 + self.step)
        consec = 0
        reason: Optional[str] = None
        self.model.train()
        try:
            while True:
                reason = self._stop_reason()
                if reason:
                    break
                try:
                    batch = loader.next(timeout=a.loader_timeout_s)
                except queue.Empty:
                    consec += 1
                    warn("loader_stalled", n=consec)
                    if consec >= 3:
                        reason = "loader_stalled"
                        break
                    continue
                try:
                    self._train_step(batch)
                    consec = 0
                    if self.bad_run >= 25:
                        warn("nonfinite_streak", step=self.step)
                        ck2 = load_ckpt(os.path.join(self.out, "ckpt_last.pt"))
                        if ck2 is not None:
                            self._restore(ck2)
                        self.bad_run = 0
                except Exception as e:  # noqa: BLE001 - skip the step, keep going
                    consec += 1
                    warn("step_failed", step=self.step, n=consec, err=err_str(e, 300))
                    self.opt.zero_grad(set_to_none=True)
                    free_cuda_cache()
                    if consec >= MAX_CONSEC_ERRORS:
                        log({"fatal": "too_many_step_errors"}, marker=MARK + "_DONE")
                        return {"status": "fatal", "step": self.step}
                    self.sleep(min(10.0, float(consec)))
                    continue
                t = self.now()
                if t >= next_log:
                    self._log_progress(t)
                    next_log = t + a.log_every_s
                if t >= next_eval:
                    self.checkpoint_and_eval(val, trip)
                    next_eval = self.now() + a.eval_every_s
        finally:
            loader.close()
        if self.acc_n:
            self._log_progress(self.now())
        self.checkpoint_and_eval(val, trip, final=True)
        self._finish_upload()
        summary = {"status": reason, "start_step": start_step, "step": self.step,
                   "best_spearman": None if self.best == -math.inf else round(self.best, 4),
                   "params": n_params, "skipped": self.skipped}
        log(summary, marker=MARK + "_DONE")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Day1 job 3: M6 T-LPIPS-S")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", "/workspace/night1"))
    ap.add_argument("--out", default=None, help="default: <root>/day1/m6")
    ap.add_argument("--deadline", default=os.environ.get("N1_DEADLINE"), help="unix seconds (default: env N1_DEADLINE)")
    ap.add_argument("--finish-margin-s", type=float, default=FINISH_MARGIN_S)
    ap.add_argument("--train-reserve-s", type=float, default=TRAIN_RESERVE_S,
                    help="labelling stops this long before the deadline (phase all) so training gets time")
    ap.add_argument("--phase", choices=["all", "label", "train"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--n-trip", type=int, default=N_TRIP)
    ap.add_argument("--gpu-ops", default=",".join(GPU_OPS), help="GPU round-trip distortions to include (tae,vae)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--mp-context", choices=["forkserver", "spawn", "fork"], default="forkserver")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--blocks", default=",".join(str(b) for b in BLOCKS))
    ap.add_argument("--label-batch", type=int, default=LABEL_BATCH, help="pairs per teacher forward")
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="stop writing pair chunks below this free disk")
    ap.add_argument("--progress-every-s", type=float, default=120.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--mem-gb", type=float, default=0.0, help="per-process CUDA cap for training (0 = off)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--max-steps", type=int, default=None, help="stop at this absolute step (tests)")
    ap.add_argument("--rank-w", type=float, default=0.5)
    ap.add_argument("--map-w", type=float, default=0.0, help="optional weight of a distance-map loss")
    ap.add_argument("--log-every-s", type=float, default=240.0)
    ap.add_argument("--eval-every-s", type=float, default=900.0)
    ap.add_argument("--first-eval-s", type=float, default=300.0)
    ap.add_argument("--loader-timeout-s", type=float, default=120.0)
    ap.add_argument("--final-upload-timeout-s", type=float, default=300.0)
    ap.add_argument("--min-images", type=int, default=1000, help="fetch corpus shards when fewer images exist")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--no-upload", action="store_true")
    return ap


def parse_blocks(s) -> Tuple[int, ...]:
    if isinstance(s, (list, tuple)):
        return tuple(int(b) for b in s)
    return tuple(int(b) for b in str(s).split(",") if str(b).strip())


def _fatal(msg: str, rc: int = 2, **kw) -> int:
    log({"fatal": scrub(msg)[:300], **kw}, marker=MARK + "_DONE")
    return rc


def main(argv: Optional[Sequence[str]] = None, deps: Optional[dict] = None) -> int:
    """``deps`` (tests only): labeler, gpu_fns, uploader, now, sleep, device, pool, stop_event."""
    args = build_parser().parse_args(argv)
    args.blocks = parse_blocks(args.blocks)
    deps = dict(deps or {})
    now = deps.get("now", time.time)
    sleep = deps.get("sleep", time.sleep)
    token = os.environ.get("HF_TOKEN") or None
    if args.no_upload:
        uploader = None
    elif "uploader" in deps:
        uploader = deps["uploader"]
    else:
        uploader = make_uploader(token, args.repo_id)
    root = args.root
    out = args.out or os.path.join(root, "day1", "m6")
    deadline = parse_deadline(args.deadline)
    log({"start": True, "phase": args.phase, "root": root, "hf_token": "yes" if token else "no",
         "upload": uploader is not None, "deadline": deadline, "torch": torch.__version__})
    if args.deadline not in (None, "") and deadline is None:
        return _fatal("deadline_unparsed", 1)
    if deadline is not None:
        left = deadline - now()
        if left <= 0:
            log({"stop": "deadline_in_past", "seconds_left": round(left, 1)}, marker=MARK + "_DONE")
            return 1
        if left > MAX_DEADLINE_AHEAD:
            return _fatal("deadline_too_far (N1_DEADLINE must be unix SECONDS)", 1, seconds_left=round(left))
    else:
        warn("no_deadline", note="running until the work is done or a signal")

    stop = deps.get("stop_event") or threading.Event()
    if threading.current_thread() is threading.main_thread() and "stop_event" not in deps:
        def _on_signal(signum, _frame):
            log({"signal": int(signum)})
            stop.set()
        for sg in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sg, _on_signal)
            except (ValueError, OSError):
                pass

    os.makedirs(os.path.join(out, "pairs"), exist_ok=True)
    clean_tmp(out)
    h = cfg_hash({"sigma": args.sigma, "blocks": list(args.blocks), "seed": args.seed, "crop": args.crop,
                  "patch": PATCH, "teacher": ZIMAGE_ID})
    if args.phase in ("all", "label"):
        rc = _label_phase(args, root, out, deps, token, uploader, deadline, stop, now, sleep)
        if rc:
            return rc
    if args.phase in ("all", "train") and not stop.is_set():
        if args.mem_gb > 0 and torch.cuda.is_available():
            try:
                frac = min(1.0, args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
                torch.cuda.set_per_process_memory_fraction(frac)
            except Exception as e:  # noqa: BLE001
                warn("mem_cap_failed", err=err_str(e))
        tr = Trainer(args, out, h, uploader=uploader, now=now, sleep=sleep, stop=stop, deadline=deadline,
                     device=deps.get("device"))
        try:
            res = tr.run()
        except Exception as e:  # noqa: BLE001 - last resort: save what we have and say why
            log({"fatal": err_str(e, 400)}, marker=MARK + "_DONE")
            try:
                if tr.model is not None and tr.opt is not None:
                    save_atomic(tr._ckpt_obj(), os.path.join(out, "ckpt_last.pt"))
            except Exception:  # noqa: BLE001
                pass
            return 2
        return 2 if res.get("status") == "fatal" else 0
    return 0


def _label_phase(args, root, out, deps, token, uploader, deadline, stop, now, sleep) -> int:
    """Steps A+B. Returns a process exit code (0 = continue)."""
    tae_path = os.path.join(root, "tae", "ckpt_best.pt")
    gpu_ops = tuple(g for g in args.gpu_ops.split(",") if g)
    lister, fetcher = make_hf_fetchers(token, root, args.repo_id)
    assets = ensure_assets(root, tae_path, args.min_images, lister, fetcher, need_tae="tae" in gpu_ops)
    corpus = list_corpus(root)
    log({"assets": assets, "corpus": len(corpus)})
    if not corpus:
        return _fatal("no corpus images under " + root)
    train_pool, val_pool = split_corpus(corpus)

    man = {"version": 1, "seed": args.seed, "crop": args.crop, "chunk": args.chunk, "n_train": args.n_train,
           "n_val": args.n_val, "n_trip": args.n_trip, "gpu_ops": list(gpu_ops), "sigma": args.sigma,
           "blocks": list(args.blocks)}
    old = load_manifest(out)
    if old is not None:
        bad = manifest_mismatch(old, man)
        if bad:
            return _fatal("manifest mismatch (use a new --out): " + ",".join(bad))
    else:
        write_manifest(out, man)

    pool = deps.get("pool")
    own_pool = False
    if pool is None:
        pool = make_pool(args.workers, args.mp_context)
        own_pool = pool is not None
    labeler = None
    try:
        labeler = deps.get("labeler")
        gpu_fns = dict(deps.get("gpu_fns") or {})
        dev_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if labeler is None:
            try:
                labeler = load_teacher(dev_name, os.path.join(out, "empty_prompt_embed.pt"), args.blocks,
                                       args.sigma, args.seed * 1_000_003)
                log({"teacher": "loaded", "blocks": list(args.blocks), "vram_GB": vram_gb()})
            except Exception as e:  # noqa: BLE001
                return _fatal("teacher_load_failed: " + err_str(e, 250))
        if "vae" in gpu_ops and "vae" not in gpu_fns:
            gpu_fns["vae"] = labeler.vae_roundtrip
        if "tae" in gpu_ops and "tae" not in gpu_fns:
            try:
                dev = torch.device(dev_name)
                gpu_fns["tae"] = make_tae_roundtrip(load_tae(tae_path, dev), dev)
            except Exception as e:  # noqa: BLE001 - TAE pairs get dropped (ok=False), the rest goes on
                warn("tae_unavailable", err=err_str(e))
        label_deadline = None
        if deadline is not None:
            label_deadline = deadline - args.finish_margin_s
            if args.phase == "all":
                label_deadline = min(label_deadline, deadline - args.train_reserve_s)
        job = LabelJob(args, out, root, train_pool, val_pool, labeler, gpu_fns, pool, now, stop, label_deadline)
        res = job.run()
        log({"phase": "label_done", **res})
        if uploader is not None:
            upload_with_retry(uploader, out, REMOTE_DIR + "/labels", allow=["labels_*.npz", "manifest.json"],
                              sleep=sleep)
        return 2 if res["reason"] == "fatal" else 0
    finally:
        if own_pool and pool is not None:
            try:
                pool.terminate()
            except Exception:  # noqa: BLE001
                pass
        if labeler is not None:                      # Z-Image comes down before the training phase starts
            try:
                labeler.close()
            except Exception:  # noqa: BLE001
                pass
            labeler = None
            gc.collect()
            free_cuda_cache()
            log({"teacher": "released", "vram_GB": vram_gb()})


if __name__ == "__main__":
    sys.exit(main())
