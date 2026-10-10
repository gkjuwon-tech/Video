#!/usr/bin/env python3
"""Job A: teacher corpus generator (Z-Image-Turbo images + paired VAE latents).

Run on the GPU box (torch 2.9.1 + CUDA):

    HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=<unix seconds> \
        python corpus_gen.py

Layout under N1_ROOT (index i is zero padded to 8 digits):
    img/{i}.webp          quality 92
    lat/{i}.npy           float16, (C, H/8, W/8), (z - shift) * scale
    meta/{i}.json         prompt, seed, h, w, category, t_gen   (written LAST)
    shards/gen_{k:04d}.tar  every 500 items, uploaded to HF corpus/night1/

An item counts as done only when img, lat and meta all exist, so a crash at
any point is repaired by simply running the script again.

Heavy dependencies (diffusers, huggingface_hub upload, torch.cuda) are
imported lazily, so the pure functions can be unit tested on a CPU box.
The HF token is read from the environment only and is never printed.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
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

MARK = "N1_GEN"
REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
REPO_TYPE = "model"
REMOTE_DIR = "corpus/night1"
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
SHARD_SIZE = 500
ENCODE_BATCH = 8
MIN_FREE_BYTES = 8 * 2**30
NUM_STEPS = 9
GUIDANCE = 0.0
WEBP_QUALITY = 92

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


# ---------------------------------------------------------------------------
# prompt grammar (deterministic: everything is a function of the index)
# ---------------------------------------------------------------------------

CATEGORIES: List[Tuple[str, int]] = [
    ("indoor", 30),
    ("landscape", 20),
    ("hair", 20),
    ("material", 15),
    ("surreal", 15),
]
SIZES: List[Tuple[Tuple[int, int], int]] = [((1024, 1024), 70), ((576, 1344), 30)]  # (h, w)

_INDOOR_PLACES = [
    "a crowded Seoul grilled-meat restaurant", "a small late-night Seoul cafe",
    "a narrow Seoul apartment kitchen", "a cluttered Seoul living room",
    "a cramped Seoul woodworking workshop", "a Seoul subway station platform",
    "a tiny Seoul ceramics studio", "a steamy Seoul noodle shop",
    "a cozy Seoul record store", "a one-room studio apartment in Seoul",
]
_INDOOR_PROPS = [
    "steaming pots and stacked ceramic bowls", "glass jars and shelves packed with small objects",
    "mirrors and glossy tiles full of reflections", "hanging lamps reflected in a wet metal counter",
    "tangled cables, potted plants and mismatched chairs", "rows of bottles catching the light",
    "stainless steel surfaces, utensils and condensation on the windows",
    "piles of books, cups and folded blankets", "polished floor reflecting colorful lights",
    "tools on pegboards, wood shavings and a half-finished chair",
    "glass doors and a rain-streaked window with blurry city lights",
]
_LAND_PLACES = [
    "a snow-capped mountain range", "a vast green meadow with wildflowers",
    "a rocky seashore with long waves", "a deep misty pine forest",
    "a distant city skyline seen across a wide valley", "a terraced hillside above a river",
    "a lonely lake between ridges", "a windswept coastal cliff",
]
_LAND_FEATURES = [
    "layers of blue haze receding toward the horizon", "strong atmospheric perspective",
    "tiny details in the foreground and a soft distant ridge line",
    "rolling clouds casting moving shadows on the land", "fog pooling in the valleys",
    "a thin road winding into the distance",
]
_HAIR_SUBJECTS = [
    "a shaggy yak", "a highland cow with a long fringe", "a woolly sheep",
    "a long-haired dog running through grass", "tall reeds swaying in the wind",
    "a bundle of loose feathers", "a handmade tassel ornament with fine threads",
    "a long-haired goat", "a fluffy owl with ruffled plumage",
]
_HAIR_DETAILS = [
    "backlit so every strand glows with rim light", "thousands of fine hairs catching the sun",
    "wispy individual fibers against a dark background",
    "wind-blown strands with sharp detail", "delicate translucent edges lit from behind",
    "thin structures crossing a bright sky",
]
_MATERIAL_SUBJECTS = [
    "weathered wood grain", "woven linen fabric", "rough granite stone",
    "rusted corrugated metal", "a glass bottle with a thick base",
    "water droplets on a green leaf", "cracked old leather", "hand-knitted wool",
    "wet river pebbles", "frosted glass with condensation",
]
_MATERIAL_SHOTS = [
    "extreme macro close-up", "tight close-up with shallow focus", "detailed texture study",
    "raking light revealing every surface imperfection", "close-up with specular highlights",
]
_SURREAL_GIANTS = [
    "a colossal yak", "a gigantic whale drifting through the sky",
    "an enormous heron wading", "a mountain-sized teapot",
    "a towering moss-covered elephant", "a giant glowing jellyfish",
    "a monumental stone fox", "an immense paper crane",
]
_SURREAL_SCENES = [
    "above a quiet valley", "over a salt flat at dusk", "in a flooded city street",
    "beside a calm harbor", "across a snowy plain", "in a field of tall grass",
]
_SURREAL_SMALL = [
    "a tiny person standing in the foreground for scale",
    "two very small figures walking below", "a lone small traveler looking up",
    "a minuscule person holding a lantern",
]
_TIMES = ["at dawn", "in the early morning", "at midday", "in the late afternoon",
          "at sunset", "at blue hour", "at night", "just after sunrise"]
_WEATHER = ["on a clear day", "under overcast skies", "after the rain", "in light fog",
            "in falling snow", "on a humid summer day", "in drifting mist", "under broken clouds"]
_LIGHTS = ["dramatic backlight", "soft overcast light", "neon glow", "golden hour light",
           "warm tungsten practicals", "cool moonlight", "hard side light", "diffuse window light"]
_TONES = ["photographic realism", "cinematic film still", "natural color photograph",
          "35mm film look with subtle grain", "high dynamic range photograph",
          "moody cinematic still"]
_LENSES = [24, 35, 50, 85]

_CAT_NAMES = [c for c, _ in CATEGORIES]


@functools.lru_cache(maxsize=4096)
def _block_perm(kind: str, block: int) -> Tuple[int, ...]:
    """Deterministic shuffle of one block of 100 indices."""
    if kind == "cat":
        pool = [n for n, (_, cnt) in enumerate(CATEGORIES) for _ in range(cnt)]
        rng = random.Random(1_000_003 * block + 17)
    else:
        pool = [n for n, (_, cnt) in enumerate(SIZES) for _ in range(cnt)]
        rng = random.Random(2_000_029 * block + 31)
    rng.shuffle(pool)
    return tuple(pool)


def category_for(i: int) -> str:
    return CATEGORIES[_block_perm("cat", i // 100)[i % 100]][0]


def size_for(i: int) -> Tuple[int, int]:
    """(height, width)."""
    return SIZES[_block_perm("size", i // 100)[i % 100]][0]


def _pick(rng: random.Random, seq: Sequence):
    return seq[rng.randrange(len(seq))]


def _lens_phrase(rng: random.Random, close: bool = False) -> str:
    mm = _pick(rng, _LENSES)
    shallow = rng.random() < (0.75 if close else 0.4)
    dof = "shallow depth of field" if shallow else "deep focus"
    return f"shot on a {mm}mm lens, {dof}"


def build_prompt(i: int) -> dict:
    """Everything about item i: prompt, seed (== i), h, w, category."""
    cat = category_for(i)
    h, w = size_for(i)
    rng = random.Random(7_919 * i + 13)
    wide = w > h
    if cat == "indoor":
        core = f"{_pick(rng, _INDOOR_PLACES)}, with {_pick(rng, _INDOOR_PROPS)}"
        mods = [_pick(rng, _TIMES), _pick(rng, _LIGHTS), _lens_phrase(rng)]
    elif cat == "landscape":
        core = f"{_pick(rng, _LAND_PLACES)}, {_pick(rng, _LAND_FEATURES)}"
        mods = [_pick(rng, _TIMES), _pick(rng, _WEATHER), _pick(rng, _LIGHTS), _lens_phrase(rng)]
    elif cat == "hair":
        core = f"{_pick(rng, _HAIR_SUBJECTS)}, {_pick(rng, _HAIR_DETAILS)}"
        mods = [_pick(rng, _WEATHER), "strong backlight", _lens_phrase(rng, close=True)]
    elif cat == "material":
        core = f"{_pick(rng, _MATERIAL_SHOTS)} of {_pick(rng, _MATERIAL_SUBJECTS)}"
        mods = [_pick(rng, _LIGHTS), _lens_phrase(rng, close=True)]
    else:  # surreal
        core = (f"{_pick(rng, _SURREAL_GIANTS)} {_pick(rng, _SURREAL_SCENES)}, "
                f"{_pick(rng, _SURREAL_SMALL)}")
        mods = [_pick(rng, _TIMES), _pick(rng, _WEATHER), _pick(rng, _LIGHTS),
                _lens_phrase(rng), "epic sense of scale"]
    if wide or cat == "surreal":
        mods.append("anamorphic cinemascope composition")
    mods.append(_pick(rng, _TONES))
    prompt = core + ", " + ", ".join(mods)
    return {"index": i, "prompt": prompt, "seed": i, "h": h, "w": w, "category": cat}


# ---------------------------------------------------------------------------
# paths / done-set / atomic writes
# ---------------------------------------------------------------------------

def idx_str(i: int) -> str:
    return f"{i:08d}"


def item_paths(root: str, i: int) -> Dict[str, str]:
    s = idx_str(i)
    return {
        "img": os.path.join(root, "img", f"{s}.webp"),
        "lat": os.path.join(root, "lat", f"{s}.npy"),
        "meta": os.path.join(root, "meta", f"{s}.json"),
    }


def ensure_dirs(root: str) -> None:
    for d in ("img", "lat", "meta", "shards"):
        os.makedirs(os.path.join(root, d), exist_ok=True)


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


def done_set(root: str) -> set:
    """Indices whose img, lat and meta all exist."""
    return (_indices_in(os.path.join(root, "img"), ".webp")
            & _indices_in(os.path.join(root, "lat"), ".npy")
            & _indices_in(os.path.join(root, "meta"), ".json"))


def clean_tmp(root: str) -> int:
    n = 0
    for d in ("img", "lat", "meta", "shards"):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for name in os.listdir(p):
            if name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(p, name))
                    n += 1
                except OSError:
                    pass
    return n


def write_item(root: str, spec: dict, img, lat: np.ndarray, t_gen: float) -> None:
    """Write webp, npy, then meta (meta last = completion marker). Atomic per file."""
    p = item_paths(root, spec["index"])
    tmps = {k: v + ".tmp" for k, v in p.items()}
    try:
        with open(tmps["img"], "wb") as f:
            img.convert("RGB").save(f, format="WEBP", quality=WEBP_QUALITY)
        with open(tmps["lat"], "wb") as f:
            np.save(f, np.ascontiguousarray(lat, dtype=np.float16))
        meta = {
            "index": spec["index"], "prompt": spec["prompt"], "seed": spec["seed"],
            "h": spec["h"], "w": spec["w"], "category": spec["category"],
            "t_gen": round(float(t_gen), 4),
        }
        with open(tmps["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmps["img"], p["img"])
        os.replace(tmps["lat"], p["lat"])
        os.replace(tmps["meta"], p["meta"])
    finally:
        for t in tmps.values():
            if os.path.exists(t):
                try:
                    os.remove(t)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------

def encode_images(vae, imgs: Sequence, max_batch: int = ENCODE_BATCH,
                  device=None, dtype=None) -> np.ndarray:
    """Encode same-size PIL images to normalized latents, float16 ``(B,C,h/8,w/8)``.

    ``z = (mean - shift_factor) * scaling_factor`` with config defaults 0.0 / 1.0.
    Runs in batches of at most ``max_batch``.
    """
    import torch

    if device is None or dtype is None:
        try:
            p = next(vae.parameters())
            device = device if device is not None else p.device
            dtype = dtype if dtype is not None else p.dtype
        except Exception:
            device = device if device is not None else "cpu"
            dtype = dtype if dtype is not None else torch.float32
    cfg = getattr(vae, "config", None)
    scale = float(getattr(cfg, "scaling_factor", 1.0) or 1.0)
    shift = getattr(cfg, "shift_factor", 0.0)
    shift = 0.0 if shift is None else float(shift)

    outs: List[np.ndarray] = []
    for s in range(0, len(imgs), max_batch):
        chunk = imgs[s: s + max_batch]
        arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in chunk])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        x = x.to(device=device, dtype=dtype)
        with torch.inference_mode():
            z = vae.encode(x).latent_dist.mean
            z = (z.float() - shift) * scale
        outs.append(z.to(torch.float16).cpu().numpy())
    return np.concatenate(outs, axis=0)


def check_latent(lat: np.ndarray, img, vae_channels: Optional[int] = None) -> Optional[str]:
    """Return an error string if the latent is unusable, else None."""
    if lat.ndim != 3:
        return f"bad_ndim:{lat.ndim}"
    if not np.isfinite(lat).all():
        return "non_finite"
    if vae_channels is not None and lat.shape[0] != vae_channels:
        return f"channels:{lat.shape[0]}!={vae_channels}"
    if lat.shape[1] != img.height // 8 or lat.shape[2] != img.width // 8:
        return f"spatial:{tuple(lat.shape[1:])} vs {(img.height // 8, img.width // 8)}"
    return None


# ---------------------------------------------------------------------------
# shards + upload
# ---------------------------------------------------------------------------

def shard_name(k: int) -> str:
    return f"gen_{k:04d}.tar"


def partial_shard_name(k: int) -> str:
    return f"gen_{k:04d}_partial.tar"


def _tar_count(path: str) -> int:
    try:
        with tarfile.open(path, "r") as t:
            return len(t.getnames()) // 3
    except Exception:
        return -1


def pack_shard(root: str, name: str, indices: Sequence[int]) -> str:
    """Write shards/<name> atomically; keep the source files in place."""
    out = os.path.join(root, "shards", name)
    tmp = out + ".tmp"
    try:
        with tarfile.open(tmp, "w") as t:
            for i in sorted(indices):
                p = item_paths(root, i)
                s = idx_str(i)
                t.add(p["img"], arcname=f"img/{s}.webp")
                t.add(p["lat"], arcname=f"lat/{s}.npy")
                t.add(p["meta"], arcname=f"meta/{s}.json")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return out


def uploaded_marker(root: str, name: str) -> str:
    return os.path.join(root, "shards", name + ".uploaded")


def make_uploader(token: Optional[str], repo_id: str = REPO_ID) -> Optional[Callable[[str, str], None]]:
    """Return ``upload(local_path, path_in_repo)`` or None when there is no token."""
    if not token:
        return None

    def upload(local: str, remote: str) -> None:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=local, path_in_repo=remote,
            repo_id=repo_id, repo_type=REPO_TYPE)

    return upload


def upload_with_retry(uploader: Callable[[str, str], None], local: str, remote: str,
                      tries: int = 3, backoff: float = 20.0,
                      sleep: Callable[[float], None] = time.sleep,
                      token: Optional[str] = None) -> bool:
    for a in range(1, tries + 1):
        try:
            uploader(local, remote)
            return True
        except Exception as e:  # noqa: BLE001 - upload must never kill the job
            warn("upload_failed", remote=remote, attempt=a, err=scrub(f"{type(e).__name__}: {e}", token)[:300])
            if a < tries:
                sleep(backoff * a)
    return False


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------

def vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_reserved() / 2**30, 2)
    except Exception:
        pass
    return 0.0


def _free_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class CorpusJob:
    def __init__(self, root: str, gen_fn: Callable[[str, int, int, int], object],
                 encode_fn: Callable[[Sequence], np.ndarray], *,
                 uploader: Optional[Callable[[str, str], None]] = None,
                 start: int = 0, max_images: int = 1_000_000,
                 deadline: Optional[float] = None, margin: float = 90.0,
                 batch: int = ENCODE_BATCH, shard_size: int = SHARD_SIZE,
                 min_free_bytes: int = MIN_FREE_BYTES, progress_every: float = 600.0,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 free_fn: Optional[Callable[[str], int]] = None,
                 upload_threads: bool = True, upload_backoff: float = 20.0,
                 final_upload_timeout: float = 900.0, vae_channels: Optional[int] = None,
                 max_consecutive_fail: int = 100):
        self.root = root
        self.gen_fn = gen_fn
        self.encode_fn = encode_fn
        self.uploader = uploader
        self.start = max(0, start)
        self.end = self.start + max_images
        self.deadline = deadline
        self.margin = margin
        self.batch = max(1, batch)
        self.S = shard_size
        self.min_free = min_free_bytes
        self.progress_every = progress_every
        self.now = now
        self.sleep = sleep
        self.free_fn = free_fn or (lambda p: shutil.disk_usage(p).free)
        self.upload_threads = upload_threads
        self.upload_backoff = upload_backoff
        self.final_upload_timeout = final_upload_timeout
        self.vae_channels = vae_channels
        self.max_consec = max_consecutive_fail

        self.stop_event = threading.Event()
        self.done: set = set()
        self.n_new = 0
        self.n_fail = 0
        self.consec_fail = 0
        self.t0 = now()
        self.last_progress = self.t0
        self.last_shard: Optional[str] = None
        self.stop_reason = "complete"
        self._pool: Optional[ThreadPoolExecutor] = None
        self._futures: List = []
        self._inflight: set = set()
        self._lock = threading.Lock()

    # -- control -----------------------------------------------------------
    def request_stop(self) -> None:
        self.stop_event.set()

    def _past_deadline(self) -> bool:
        return self.deadline is not None and self.now() >= self.deadline - self.margin

    def _should_stop(self) -> Optional[str]:
        if self.stop_event.is_set():
            return "signal"
        if self._past_deadline():
            return "deadline"
        try:
            free = self.free_fn(self.root)
        except Exception:
            free = None
        if free is not None and free < self.min_free:
            return "disk"
        return None

    # -- progress ----------------------------------------------------------
    def progress(self, force: bool = False) -> None:
        t = self.now()
        if not force and t - self.last_progress < self.progress_every:
            return
        self.last_progress = t
        el = max(t - self.t0, 1e-6)
        log({"n": len(self.done), "new": self.n_new, "failed": self.n_fail,
             "rate_img_per_min": round(self.n_new / el * 60.0, 2),
             "vram_GB": vram_gb(), "last_shard": self.last_shard})

    # -- shards ------------------------------------------------------------
    def _members(self, k: int) -> List[int]:
        lo, hi = k * self.S, (k + 1) * self.S
        return [i for i in range(lo, hi) if i in self.done]

    def _ensure_shard(self, k: int, partial: bool = False) -> Optional[str]:
        members = self._members(k)
        if not members:
            return None
        name = partial_shard_name(k) if partial else shard_name(k)
        path = os.path.join(self.root, "shards", name)
        try:
            if not (os.path.exists(path) and _tar_count(path) == len(members)):
                pack_shard(self.root, name, members)
                m = uploaded_marker(self.root, name)
                if os.path.exists(m):
                    os.remove(m)
        except Exception as e:  # noqa: BLE001
            warn("pack_failed", shard=name, err=f"{type(e).__name__}: {e}"[:300])
            return None
        return name

    def _upload_task(self, name: str) -> None:
        path = os.path.join(self.root, "shards", name)
        remote = f"{REMOTE_DIR}/{name}"
        try:
            ok = upload_with_retry(self.uploader, path, remote, backoff=self.upload_backoff,
                                   sleep=self.sleep)
            if ok:
                with open(uploaded_marker(self.root, name), "w") as f:
                    f.write(str(int(self.now())))
                self.last_shard = name
                log({"shard_uploaded": name, "n": len(self.done)})
        except Exception as e:  # noqa: BLE001
            warn("upload_task_error", shard=name, err=f"{type(e).__name__}: {e}"[:300])
        finally:
            with self._lock:
                self._inflight.discard(name)

    def _submit_upload(self, name: str) -> None:
        if self.uploader is None:
            return
        if os.path.exists(uploaded_marker(self.root, name)):
            return
        with self._lock:
            if name in self._inflight:
                return
            self._inflight.add(name)
        if not self.upload_threads:
            self._upload_task(name)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="n1_upload")
        self._futures.append(self._pool.submit(self._upload_task, name))

    def _finish_shard(self, k: int, partial: bool = False) -> None:
        name = self._ensure_shard(k, partial=partial)
        if name is None:
            return
        if not partial:
            self.last_shard = name
        self._submit_upload(name)

    def _drain_uploads(self) -> None:
        if self._pool is None:
            return
        end = time.time() + self.final_upload_timeout
        for f in self._futures:
            left = end - time.time()
            try:
                f.result(timeout=max(left, 0.1))
            except Exception as e:  # noqa: BLE001
                warn("upload_wait", err=f"{type(e).__name__}: {e}"[:200])
                break
        self._pool.shutdown(wait=False)
        self._pool = None
        self._futures = []

    # -- one chunk ---------------------------------------------------------
    def _process_chunk(self, idxs: List[int]) -> Optional[str]:
        """Generate, encode and store ``idxs``. Returns a stop reason or None."""
        generated: List[Tuple[dict, object, float]] = []
        stop: Optional[str] = None
        for i in idxs:
            stop = self._should_stop()
            if stop:
                break
            spec = build_prompt(i)
            t = time.time()
            try:
                img = self.gen_fn(spec["prompt"], spec["h"], spec["w"], spec["seed"])
            except Exception as e:  # noqa: BLE001
                self._item_failed(i, "gen_failed", e)
                stop = self._maybe_abort()
                if stop:
                    break
                continue
            generated.append((spec, img, time.time() - t))
        self._encode_and_store(generated)
        return stop

    def _maybe_abort(self) -> Optional[str]:
        if self.consec_fail >= self.max_consec:
            return "errors"
        if self.consec_fail >= 20:
            self.sleep(30.0)
        return None

    def _item_failed(self, i: int, what: str, e: BaseException) -> None:
        self.n_fail += 1
        self.consec_fail += 1
        warn(what, i=i, err=f"{type(e).__name__}: {e}"[:300])
        if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
            _free_cuda_cache()

    def _encode_and_store(self, generated: List[Tuple[dict, object, float]]) -> None:
        groups: Dict[Tuple[int, int], List[Tuple[dict, object, float]]] = {}
        for g in generated:
            im = g[1]
            groups.setdefault((im.height, im.width), []).append(g)
        for _, items in groups.items():
            for s in range(0, len(items), self.batch):
                sub = items[s: s + self.batch]
                lats: List[Optional[np.ndarray]]
                try:
                    arr = self.encode_fn([g[1] for g in sub])
                    lats = [arr[n] for n in range(len(sub))]
                except Exception as e:  # noqa: BLE001
                    warn("encode_batch_failed", err=f"{type(e).__name__}: {e}"[:300])
                    _free_cuda_cache()
                    lats = []
                    for g in sub:  # retry one by one so a single bad item is isolated
                        try:
                            lats.append(self.encode_fn([g[1]])[0])
                        except Exception as e2:  # noqa: BLE001
                            self._item_failed(g[0]["index"], "encode_failed", e2)
                            lats.append(None)
                for g, lat in zip(sub, lats):
                    spec, img, t_gen = g
                    if lat is None:
                        continue
                    bad = check_latent(np.asarray(lat), img, self.vae_channels)
                    if bad:
                        self._item_failed(spec["index"], "bad_latent", RuntimeError(bad))
                        continue
                    try:
                        write_item(self.root, spec, img, np.asarray(lat), t_gen)
                    except Exception as e:  # noqa: BLE001
                        self._item_failed(spec["index"], "write_failed", e)
                        continue
                    self.done.add(spec["index"])
                    self.n_new += 1
                    self.consec_fail = 0

    # -- main loop ---------------------------------------------------------
    def run(self) -> dict:
        ensure_dirs(self.root)
        clean_tmp(self.root)
        self.done = done_set(self.root)
        log({"start": True, "root": self.root, "already_done": len(self.done),
             "hf_token": "yes" if self.uploader else "no", "deadline": self.deadline})
        i = self.start
        try:
            while i < self.end:
                reason = self._should_stop()
                if reason:
                    self.stop_reason = reason
                    if reason == "disk":
                        log({"stop": "disk"})
                    break
                k = i // self.S
                chunk_end = min(i + self.batch, (k + 1) * self.S, self.end)
                idxs = [j for j in range(i, chunk_end) if j not in self.done]
                if idxs:
                    reason = self._process_chunk(idxs)
                    if reason:
                        self.stop_reason = reason
                        if reason == "disk":
                            log({"stop": "disk"})
                        if reason == "errors":
                            log({"stop": "errors", "consecutive_failures": self.consec_fail})
                        break
                i = chunk_end
                if i % self.S == 0:
                    self._finish_shard(k)
                self.progress()
            else:
                self.stop_reason = "complete"
        except Exception as e:  # noqa: BLE001 - never die with files unsaved
            warn("loop_error", err=f"{type(e).__name__}: {e}"[:300])
            self.stop_reason = "error"
        finally:
            self._finalize(i)
        return self.summary()

    def _finalize(self, i: int) -> None:
        try:
            if self.stop_reason == "complete":
                k = (i - 1) // self.S if i > self.start else None
                need = k is not None and i % self.S != 0
            else:
                k, need = i // self.S, True
            if need:
                full = len(self._members(k)) >= self.S
                self._finish_shard(k, partial=not full)
            self._drain_uploads()
        except Exception as e:  # noqa: BLE001
            warn("finalize_error", err=f"{type(e).__name__}: {e}"[:300])
        self.progress(force=True)
        log({"done": True, "reason": self.stop_reason, "n": len(self.done),
             "new": self.n_new, "failed": self.n_fail, "last_shard": self.last_shard})

    def summary(self) -> dict:
        return {"reason": self.stop_reason, "n": len(self.done), "new": self.n_new,
                "failed": self.n_fail, "last_shard": self.last_shard}


# ---------------------------------------------------------------------------
# GPU wiring (lazy imports; not exercised by CPU tests)
# ---------------------------------------------------------------------------

def load_pipeline(device: str = "cuda"):
    import torch
    from diffusers import ZImagePipeline
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, ignore_patterns=["assets/*"])
    pipe = ZImagePipeline.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    pipe.to(device)
    return pipe


def make_gen_fn(pipe, device: str = "cuda"):
    import torch

    def gen(prompt: str, h: int, w: int, seed: int):
        g = torch.Generator(device).manual_seed(int(seed))
        with torch.inference_mode():
            out = pipe(prompt=prompt, height=h, width=w, num_inference_steps=NUM_STEPS,
                       guidance_scale=GUIDANCE, generator=g)
        return out.images[0]

    return gen


def make_encode_fn(vae, batch: int = ENCODE_BATCH):
    def enc(imgs):
        return encode_images(vae, imgs, max_batch=batch)

    return enc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_deadline(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Night1 job A: Z-Image teacher corpus (images + VAE latents)")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", "/workspace/night1"))
    ap.add_argument("--start", type=int, default=0, help="first image index")
    ap.add_argument("--max-images", type=int, default=1_000_000, help="how many indices to cover")
    ap.add_argument("--deadline", default=os.environ.get("N1_DEADLINE"),
                    help="unix seconds (default: env N1_DEADLINE)")
    ap.add_argument("--margin", type=float, default=90.0, help="stop starting new images this many seconds before the deadline")
    ap.add_argument("--batch", type=int, default=ENCODE_BATCH, help="VAE encode batch (max 8)")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--progress-every", type=float, default=600.0)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--print-prompts", type=int, default=0, metavar="N",
                    help="print the first N prompts as JSON lines and exit (no GPU)")
    return ap


def main(argv: Optional[Sequence[str]] = None, deps: Optional[dict] = None) -> int:
    """``deps`` (tests only) may supply gen_fn, encode_fn, uploader, vae_channels, now, sleep, free_fn."""
    args = build_parser().parse_args(argv)
    if args.print_prompts:
        for i in range(args.start, args.start + args.print_prompts):
            print(json.dumps(build_prompt(i), ensure_ascii=False))
        return 0
    deps = dict(deps or {})
    token = os.environ.get("HF_TOKEN") or None
    uploader = None
    if not args.no_upload:
        uploader = deps.pop("uploader", None) or make_uploader(token, args.repo_id)
    deps.pop("uploader", None)

    vae_channels = deps.pop("vae_channels", None)
    gen_fn = deps.pop("gen_fn", None)
    encode_fn = deps.pop("encode_fn", None)
    if gen_fn is None or encode_fn is None:
        try:
            pipe = load_pipeline()
            gen_fn = make_gen_fn(pipe)
            encode_fn = make_encode_fn(pipe.vae, min(args.batch, ENCODE_BATCH))
            vae_channels = getattr(pipe.vae.config, "latent_channels", None)
            log({"pipeline": "loaded", "latent_channels": vae_channels, "vram_GB": vram_gb()})
        except Exception as e:  # noqa: BLE001
            log({"stop": "pipeline_load_failed", "err": scrub(f"{type(e).__name__}: {e}")[:500]})
            return 1

    job = CorpusJob(
        args.root, gen_fn, encode_fn, uploader=uploader, start=args.start,
        max_images=args.max_images, deadline=parse_deadline(args.deadline),
        margin=args.margin, batch=min(args.batch, ENCODE_BATCH), shard_size=args.shard_size,
        min_free_bytes=int(args.min_free_gb * 2**30), progress_every=args.progress_every,
        vae_channels=vae_channels, **deps)

    def _on_signal(signum, _frame):
        job.request_stop()

    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(s, _on_signal)
            except (ValueError, OSError):
                pass
    res = job.run()
    return 2 if res["reason"] in ("errors", "error") else 0


if __name__ == "__main__":
    sys.exit(main())
