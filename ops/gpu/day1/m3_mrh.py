#!/usr/bin/env python3
"""Day1 job 4: M3 "MRH" spike (the mind reader: Z-Image-Turbo feature taps -> depth / normal heads).

Spec: ops/gpu/day1/M3_SPEC.md.  One launcher process (``--stage all``) supervises sub-processes:

    diode    CPU + network  stream DIODE val/train tarballs (never stored whole), keep the selected crops
    render   CPU x6         RoomGen / DAT-01 synthetic renders (+ tone map)
    dav2     GPU <= 3 GiB   Depth-Anything-V2-Small baseline numbers (evaluation only)
    teacher  GPU <= 22 GiB  F0 smoke, F1 PCA-128 basis, F2a eval taps, F2b realify, F2c DIODE taps, F3 release
    train    GPU <= 10 GiB  MRH training + evaluation + previews + uploads

Run on the GPU box (torch 2.9.1 + CUDA) next to M2 / M6 / pipe_yak::

    HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=1791723600 \\
    PYTHONPATH=$N1_ROOT/repo_m3:$N1_ROOT/code PYTHONUNBUFFERED=1 \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python -u m3_mrh.py --root $N1_ROOT --stage all

Outputs live under ``$N1_ROOT/day1/m3`` (see ``Paths``).  Everything is restartable: each stage writes
``stages/<name>.done.json`` last, per-sample stages keep ``progress_<stage>.jsonl``, training resumes from
``ckpt_last.pt``.  ``DONE.json`` is written only when training reached ``total_steps`` (keep-alive semantics: a
relaunch after that exits at once).  A deadline stop exits 0 without DONE.json so night2 can continue.

Data rules (docs/DATA-PLAN.md): R1 commercial-training sources only; R2 no neural-net labels (ground truth comes
from the DIODE laser scans and from analytic RoomGen renders only; Depth-Anything output is evaluation-only and
never leaves ``eval/``); the founder's yak image is read from the private vault, only ever lives under
``<out>/private`` and ``<out>/preview`` and is never written to a repository directory.  DIODE is MIT licensed and
the FND-04 firewall (``buc.gov.prov.TRAIN_LICENSES``) does not accept MIT yet: its taps are extracted but parked in
``taps_pending/`` until the firewall check passes (``promote_pending``).  Provenance tags are never relabelled.

Markers (one JSON object per line, <= 400 chars, flushed): D1_M3, D1_M3_DATA, D1_M3_EVAL, D1_M3_DONE.  The HF token
is read from the environment only and never printed.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import http.client
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
import ssl
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ROOT = "/workspace/night1"


def setup_paths(root: Optional[str] = None) -> None:
    """Put the repo checkout (``<root>/repo_m3``, ``<root>/repo``, the checkout this file lives in) on sys.path."""
    r = root or os.environ.get("N1_ROOT") or DEFAULT_ROOT
    for c in (Path(r) / "repo_m3", Path(r) / "repo", Path(__file__).resolve().parents[3]):
        if (c / "buc").is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


setup_paths()


def ensure_blake3() -> bool:
    """buc.core.det needs ``blake3``.  Install a hashlib-backed shim when the box lacks it (pipe_yak style).
    True = shim in use (shard addresses are then not canonical blake3 values; recorded in every sample meta)."""
    try:
        import blake3  # noqa: F401
        return False
    except ImportError:
        pass
    import types
    mod = types.ModuleType("blake3")

    class _B3:
        def __init__(self, data=b""):
            self._h = hashlib.blake2b(bytes(data), digest_size=32)

        def digest(self):
            return self._h.digest()

        def hexdigest(self):
            return self._h.hexdigest()

        def update(self, data):
            self._h.update(bytes(data))

    mod.blake3 = _B3
    mod.__version__ = "shim-blake2b"
    sys.modules["blake3"] = mod
    return True


BLAKE3_SHIM = ensure_blake3()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from buc.core import det  # noqa: E402
from buc.data import realify as R  # noqa: E402
from buc.data import shards as SH  # noqa: E402
from buc.data import synth_render as SR  # noqa: E402
from buc.gov import prov as P  # noqa: E402
from buc.models.mrh import heads as MH  # noqa: E402
from buc.models.mrh import neck as MN  # noqa: E402
from buc.models.mrh import train as MT  # noqa: E402
from buc.read.mrh_call import SURVEY_CAPTION, global_size  # noqa: E402
from buc.teacher import sample as TS  # noqa: E402
from buc.teacher import taps as T  # noqa: E402
from buc.teacher import zimage as Z  # noqa: E402

MARK_DEFAULT = "D1_M3"
MARKER_LIMIT = 400
DONE_FILE = "DONE.json"
REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
REPO_TYPE = "model"
REMOTE_CKPT = "ckpt/day1/m3"
REMOTE_PREVIEW = "ckpt/day1/m3/preview"
REMOTE_EVAL_TAPS = "ckpt/day1/m3/eval_taps"
REMOTE_TAPS = "corpus/day1/m3_taps"
REMOTE_DIODE = "corpus/day1/diode"
REMOTE_PREFIXES = (REMOTE_CKPT, REMOTE_TAPS, REMOTE_DIODE)
CORPUS_PREFIX = "corpus/night1/"
YAK_REMOTE = "demo/d15_hero_yak.webp"
DAV2_ID = "depth-anything/Depth-Anything-V2-Small-hf"

TAP_LAYERS = (8, 15, 23)
TAP_SIGMA = 0.3                  # snapped to the sampler table (index 7)
PCA_K = 128
SAMPLE_TAG = "buc.d1.m3.sample.v1"
STRIDE = MT.STRIDE               # label stride (H/4)
SEED = 0                         # reader noise seed (det.derive_seed(SEED, key))
TOKENS_PER_IMG = 96              # PCA fit tokens per image and layer
BASIS_N_CORPUS, BASIS_N_RENDER = 600, 400
DEPTH_LO, DEPTH_HI = 0.25, 400.0  # label clamp (metres)
EVAL_LO, EVAL_HI = 0.25, 350.0    # evaluation mask range (metres)
DIODE_BASE = "https://diode-dataset.s3.amazonaws.com"
DIODE_EXPECTED_PNG = {"train": 25458, "val": 771}
DIODE_DEV_SCENES = ("00004", "00009", "00012")
DIODE_HW = (768, 1024)
SYNTH_TRAIN_SEEDS = (100000, 103000)     # [lo, hi)
SYNTH_DEV_SEEDS = (0, 200)
SYNTH_VIEWS = 2
SYNTH_WH = (512, 384)
REALIFY_SEED = 7
GATE_BYTES = 24e9
TEACHER_CAP_GIB = 22.0
TRAIN_CAP_GIB = 10.0
DAV2_CAP_GIB = 3.0
MIN_VALID_PX = MT.MIN_VALID_PX
LOSS_W = {"depth": 1.0, "scale": 0.15, "normal": 0.5}
STATUS_RC = {"done": 0, "deadline": 0, "signal": 0, "fatal": 2}
RC_GATE_DEADLINE = 10            # teacher child: gate never opened in time (parent maps it to rc 0)
RC_INCOMPLETE = 3                # teacher child: stopped before F3 (retry)

_MARK = MARK_DEFAULT
_PRINT_LOCK = threading.Lock()


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


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fit_marker(marker: str, payload: dict, limit: int = MARKER_LIMIT) -> str:
    """One marker line that is always <= ``limit`` chars and free of token-like strings."""
    def render(p) -> str:
        return scrub(f"{marker} {json.dumps(p, default=_json_default, separators=(',', ':'))}")

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


def set_marker_prefix(prefix: str) -> None:
    global _MARK
    _MARK = prefix or MARK_DEFAULT


def marker_prefix() -> str:
    return _MARK


def log(payload: dict, suffix: str = "") -> None:
    """Print one marker.  ``suffix`` is "", "_DATA", "_EVAL" or "_DONE"."""
    line = fit_marker(_MARK + suffix, payload)
    with _PRINT_LOCK:
        print(line, flush=True)


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


def gb(nbytes: float) -> float:
    return round(float(nbytes) / 1e9, 2)


# ---------------------------------------------------------------------------
# atomic IO, zip helpers
# ---------------------------------------------------------------------------

def write_atomic(path: str, data: bytes) -> None:
    """``.part`` -> fsync -> ``os.replace``.  The target appears complete or not at all."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.remove(tmp)


def write_json_atomic(path: str, obj) -> None:
    write_atomic(path, json.dumps(obj, default=_json_default, sort_keys=True).encode("utf-8"))


def read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_atomic(obj, path: str) -> None:
    """torch.save to ``.tmp`` -> fsync -> rename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.remove(tmp)


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


_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def zip_bytes(members: Dict[str, bytes]) -> bytes:
    """Deterministic zip (fixed timestamps, sorted members, deflate)."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, members[name])
    return bio.getvalue()


def npy_bytes(a: np.ndarray) -> bytes:
    return SH.npy_encode(a)


def npz_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` (readable with ``np.load``)."""
    return zip_bytes({f"{k}.npy": npy_bytes(np.asarray(v)) for k, v in arrays.items()})


def load_npz_bytes(b: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def stable_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=_json_default).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def key_parity(key: str) -> int:
    """Stable 0/1 per sample key (realify mode: even -> sigma 0.5 / 2 NFE, odd -> sigma 0.3 / 1 NFE)."""
    return hashlib.sha256(key.encode("utf-8")).digest()[0] & 1


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

class UnsafePathError(RuntimeError):
    """A write was refused because it would land in a repository directory or outside the job folder."""


class Paths:
    """Everything this job writes lives under ``<root>/day1/m3`` (``out``)."""

    def __init__(self, root: str, out: Optional[str] = None):
        self.root = os.path.abspath(root)
        self.out = os.path.abspath(out or os.path.join(self.root, "day1", "m3"))
        j = lambda *a: os.path.join(self.out, *a)  # noqa: E731
        self.data = j("data")
        self.diode = j("data", "diode")
        self.diode_train = j("data", "diode", "train")
        self.diode_val = j("data", "diode", "val")
        self.synth = j("data", "synth")
        self.synth_train = j("data", "synth", "train")
        self.synth_dev = j("data", "synth", "dev")
        self.stages = j("stages")
        self.shards = j("shards")
        self.shards_train = j("shards", "train")
        self.shards_dev = j("shards", "dev")
        self.pending = j("taps_pending", "diode")
        self.eval = j("eval")
        self.val_taps = j("eval", "taps", "val")
        self.dav2 = j("eval", "dav2")
        self.private = j("private")
        self.preview = j("preview")
        self.pids = j("pids")
        self.upload_stage = j("upload_stage")
        self.basis_samples = j("basis_samples")
        self.te_cache = j("te_cache")
        self.basis = j("basis.pt")
        self.cap_embed = j("cap_embed.pt")
        self.backend = j("backend.json")
        self.plan = j("plan.json")
        self.tap_std = j("tap_std.pt")
        self.ckpt_last = j("ckpt_last.pt")
        self.ckpt_best = j("ckpt_best.pt")
        self.metrics = j("metrics.jsonl")
        self.eval_json = j("eval_table.json")
        self.eval_md = j("eval_table.md")
        self.stages_json = j("stages.json")
        self.source_diode = j("SOURCE_DIODE.json")
        self.done = j(DONE_FILE)
        self.yak_taps = j("private", "yak_taps.npz")
        self.yak_dav2 = j("private", "yak_dav2.npy")
        self.logs = os.path.join(self.root, "logs")

    def progress(self, stage: str) -> str:
        return os.path.join(self.out, f"progress_{stage}.jsonl")

    def stage_file(self, name: str) -> str:
        return os.path.join(self.stages, f"{name}.done.json")

    def ensure(self) -> None:
        for d in (self.out, self.stages, self.diode_train, self.diode_val, self.synth_train, self.synth_dev,
                  self.shards_train, self.shards_dev, self.pending, self.val_taps, self.dav2, self.private,
                  self.preview, self.pids):
            os.makedirs(d, exist_ok=True)


def repo_roots(root: str) -> List[Path]:
    """Directories a write must never land in: every checkout the box (or this file) uses."""
    here = Path(__file__).resolve().parents[3]
    c = [here] + [Path(root) / n for n in ("repo", "repo_m3", "repo_pipe", "code")]
    out = []
    for p in c:
        with contextlib.suppress(OSError, RuntimeError):
            out.append(p.resolve())
    return out


def safe_out_path(path, paths: Paths) -> str:
    """Resolve ``path`` and refuse it when it is inside a repository checkout or outside the job folder."""
    rp = Path(path).resolve()
    out = Path(paths.out).resolve()
    for bad in repo_roots(paths.root):
        if rp == bad or bad in rp.parents:
            raise UnsafePathError(f"refusing to write into a repository directory: {rp}")
    if not (rp == out or out in rp.parents):
        raise UnsafePathError(f"refusing to write outside the job folder: {rp}")
    return str(rp)


def clean_tmp(paths: Paths) -> int:
    """Remove half-written ``.tmp`` / ``.part`` files left by a kill (never touches finished files)."""
    n = 0
    for d, _dirs, files in os.walk(paths.out):
        if os.path.basename(d) in ("hf_cache", ".cache"):
            continue
        for name in files:
            if name.endswith(".tmp") or name.endswith(".part"):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(d, name))
                    n += 1
    return n


# ---------------------------------------------------------------------------
# stage markers (idempotence)
# ---------------------------------------------------------------------------

def stage_record(paths: Paths, name: str) -> Optional[dict]:
    d = read_json(paths.stage_file(name))
    return d if isinstance(d, dict) and d.get("stage") == name else None


def stage_done(paths: Paths, name: str, config_hash: Optional[str] = None) -> bool:
    """True when ``stages/<name>.done.json`` exists (and, if given, was written for the same config hash)."""
    d = stage_record(paths, name)
    if d is None:
        return False
    return config_hash is None or d.get("config_hash") == config_hash


def write_stage_done(paths: Paths, name: str, config_hash: str, counts: Optional[dict] = None) -> None:
    write_json_atomic(paths.stage_file(name), {"stage": name, "config_hash": config_hash,
                                               "finished_at": time.time(), "counts": counts or {}})


def run_stage(paths: Paths, name: str, config_hash: str, fn: Callable[[], Optional[dict]]) -> Tuple[bool, Optional[dict]]:
    """Run ``fn`` unless the stage is already done for this config.  ``fn`` returns the counts to record, or
    None for a partial run (no done marker is written).  Returns (ran, counts)."""
    if stage_done(paths, name, config_hash):
        return False, (stage_record(paths, name) or {}).get("counts")
    counts = fn()
    if counts is not None and not counts.get("partial"):
        write_stage_done(paths, name, config_hash, counts)
    return True, counts


def write_stages_json(paths: Paths) -> dict:
    allst = {}
    if os.path.isdir(paths.stages):
        for n in sorted(os.listdir(paths.stages)):
            if n.endswith(".done.json"):
                d = read_json(os.path.join(paths.stages, n))
                if isinstance(d, dict):
                    allst[d.get("stage", n)] = {k: d.get(k) for k in ("finished_at", "counts", "config_hash")}
    write_json_atomic(paths.stages_json, allst)
    return allst


def append_jsonl(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_json_default, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> List[dict]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                  # a half-written last line after a kill
                if isinstance(r, dict):
                    out.append(r)
    except OSError:
        pass
    return out


def progress_keys(path: str) -> set:
    return {r["key"] for r in read_jsonl(path) if "key" in r}


# ---------------------------------------------------------------------------
# process bookkeeping (duplicate guard), disk
# ---------------------------------------------------------------------------

def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def pid_alive(pid: int, needle: str = "m3_mrh.py", cmdline_fn: Callable[[int], str] = _proc_cmdline) -> bool:
    """pid file check: the process exists AND its command line mentions this job (pid reuse safe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return needle in cmdline_fn(pid)


def claim_pid(paths: Paths, name: str, pid: Optional[int] = None,
              cmdline_fn: Callable[[int], str] = _proc_cmdline) -> bool:
    """Take the single-instance slot ``name``.  False when another live m3_mrh.py process holds it."""
    os.makedirs(paths.pids, exist_ok=True)
    pf = os.path.join(paths.pids, f"{name}.pid")
    me = int(os.getpid() if pid is None else pid)
    try:
        with open(pf, "r") as f:
            old = int(f.read().strip() or 0)
    except (OSError, ValueError):
        old = 0
    if old and old != me and pid_alive(old, "m3_mrh.py", cmdline_fn):
        return False
    write_atomic(pf, str(me).encode())
    return True


def release_pid(paths: Paths, name: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(os.path.join(paths.pids, f"{name}.pid"))


def free_disk_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def disk_policy(free_gb: float) -> str:
    """"ok" | "double" (>= 8 GB, < 25 GB free: double the DIODE selection strides) | "stop" (< 8 GB free)."""
    if free_gb < 8.0:
        return "stop"
    if free_gb < 25.0:
        return "double"
    return "ok"


# ---------------------------------------------------------------------------
# provenance (FND-04).  Tags are assigned per source and never relabelled.
# ---------------------------------------------------------------------------

def prov_roomgen(seed: int) -> "P.ProvRecord":
    return P.ProvRecord(taint=frozenset({"ENGINE"}), mode="none", roots=(f"roomgen:{int(seed)}",))


def prov_night1(i: int) -> "P.ProvRecord":
    return P.ProvRecord(taint=frozenset({"ENGINE"}), mode="none", roots=(f"night1:{int(i)}",))


def prov_diode_train(scene: str, scan: str) -> "P.ProvRecord":
    return P.ProvRecord(taint=frozenset({"ASSET:MIT"}), mode="none", roots=(f"diode:{scene}/{scan}",))


def prov_diode_val(scene: str, scan: str) -> "P.ProvRecord":
    return P.ProvRecord(taint=frozenset({"ASSET:MIT", "HOLDOUT:diode_val"}), mode="none",
                        roots=(f"diode:{scene}/{scan}",))


def prov_yak() -> "P.ProvRecord":
    """Same lane as pipe_yak.make_input_image: USER + HOLDOUT:r_eval in the private user vault namespace."""
    return P.ProvRecord(taint=frozenset({"USER", "HOLDOUT:r_eval"}), mode="none", roots=("yak",),
                        ns="user_vault")


def diode_firewall() -> dict:
    """Current FND-04 verdict for DIODE train crops.  ``ok`` False = park taps in ``taps_pending/``."""
    why = tuple(P.why_not_train(prov_diode_train("00000", "00000")))
    return {"ok": not why, "why": list(why)}


def batch_prov(provs: Sequence) -> "P.ProvRecord":
    return P.derive("m3_read", list(provs))


# ---------------------------------------------------------------------------
# HF helpers (token from the environment only)
# ---------------------------------------------------------------------------

class Uploader:
    """Thin HfApi wrapper: ``folder(local_dir, remote_dir)``, ``files([(local, remote), ...])``."""

    def __init__(self, token: str, repo_id: str = REPO_ID):
        self._token = token
        self.repo_id = repo_id

    def folder(self, local_dir: str, remote_dir: str) -> None:
        from huggingface_hub import HfApi
        HfApi(token=self._token).upload_folder(folder_path=local_dir, path_in_repo=remote_dir,
                                               repo_id=self.repo_id, repo_type=REPO_TYPE,
                                               commit_message="day1 m3 spike")

    def files(self, pairs: Sequence[Tuple[str, str]]) -> None:
        from huggingface_hub import CommitOperationAdd, HfApi
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r in pairs]
        HfApi(token=self._token).create_commit(repo_id=self.repo_id, repo_type=REPO_TYPE, operations=ops,
                                               commit_message="day1 m3 spike")


def make_uploader(token: Optional[str], repo_id: str = REPO_ID) -> Optional[Uploader]:
    return Uploader(token, repo_id) if token else None


def check_remote(remote: str) -> str:
    """Uploads go only to the m3 prefixes of the private vault."""
    if ".." in remote.split("/") or not any(remote == p or remote.startswith(p + "/") for p in REMOTE_PREFIXES):
        raise ValueError(f"remote path outside the m3 prefixes: {remote}")
    return remote


def upload_with_retry(fn: Callable[[], None], tries: int = 3, sleep: Callable[[float], None] = time.sleep) -> bool:
    for k in range(tries):
        try:
            fn()
            return True
        except Exception as e:  # noqa: BLE001 - upload trouble is a warning, never fatal
            warn("upload_failed", attempt=k + 1, err=err_str(e, 300))
            if k + 1 < tries:
                sleep(5 * (3 ** k))
    return False


def make_hf_fetchers(token: Optional[str], root: str, repo_id: str = REPO_ID):
    """(lister(prefix) -> [remote names], fetcher(remote, local_dir) -> local path); (None, None) without a token."""
    if not token:
        return None, None

    def lister(prefix: str) -> List[str]:
        from huggingface_hub import HfApi
        return [n for n in HfApi(token=token).list_repo_files(repo_id, repo_type=REPO_TYPE) if n.startswith(prefix)]

    def fetcher(remote: str, local_dir: Optional[str] = None) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id, remote, repo_type=REPO_TYPE, token=token,
                               local_dir=local_dir or os.path.join(root, "_hf_dl"))

    return lister, fetcher


# ---------------------------------------------------------------------------
# DIODE: member names, selection, ground-truth reduction
# ---------------------------------------------------------------------------

MEMBER_RE = re.compile(
    r"(?P<split>train|val)/(?P<env>indoors|outdoor)/scene_(?P<scene>\d{5})/scan_(?P<scan>\d{5})/"
    r"(?P=scene)_(?P=scan)_(?P=env)_(?P<yaw>\d{3})_(?P<pitch>\d{3})(?P<suffix>_depth_mask|_depth)?\.(?P<ext>png|npy)")


@dataclasses.dataclass(frozen=True)
class Member:
    split: str
    env: str            # "indoors" | "outdoor"
    scene: str
    scan: str
    yaw: int            # file name code, 0..350 step 10
    pitch: int          # file name code, 0..50 step 10
    kind: str           # "rgb" | "depth" | "mask"

    @property
    def yaw_i(self) -> int:
        return self.yaw // 10

    @property
    def pitch_i(self) -> int:
        return self.pitch // 10

    @property
    def stem(self) -> str:
        return f"{self.scene}_{self.scan}_{self.env}_{self.yaw:03d}_{self.pitch:03d}"

    @property
    def scan_key(self) -> tuple:
        return (self.split, self.env, self.scene, self.scan)

    @property
    def rel_dir(self) -> str:
        return os.path.join(self.env, f"scene_{self.scene}", f"scan_{self.scan}")


def parse_member(name) -> Optional[Member]:
    """Tar member name -> Member, or None for directories, junk, absolute / ``..`` paths and odd spellings.
    Output paths are later built from the matched digit / fixed-word groups only."""
    if not isinstance(name, str) or not name or "\0" in name or "\\" in name:
        return None
    n = name[2:] if name.startswith("./") else name
    if n.startswith("/") or ".." in n.split("/"):
        return None
    m = MEMBER_RE.fullmatch(n)
    if m is None:
        return None
    suffix, ext = m.group("suffix"), m.group("ext")
    if (ext == "png") != (suffix is None):
        return None
    kind = "rgb" if suffix is None else ("depth" if suffix == "_depth" else "mask")
    return Member(m.group("split"), m.group("env"), m.group("scene"), m.group("scan"),
                  int(m.group("yaw")), int(m.group("pitch")), kind)


@dataclasses.dataclass(frozen=True)
class Selection:
    """Which DIODE crops to keep ("spike" profile).  ``keep`` looks at the file name only (a pure function)."""
    profile: str = "spike"
    pitch_idx: tuple = (1, 2, 3)         # -10 / 0 / +10 degrees
    stride_outdoor: int = 3
    stride_indoor: int = 6
    mult: int = 1                        # disk policy: 2 doubles both strides

    def keep(self, m: Member) -> bool:
        if m.split == "val":
            return True                  # the whole validation split is evaluation data
        stride = (self.stride_indoor if m.env == "indoors" else self.stride_outdoor) * self.mult
        return m.pitch_i in self.pitch_idx and m.yaw_i % stride == int(m.scan) % stride

    def params(self) -> dict:
        return {"profile": self.profile, "pitch_idx": list(self.pitch_idx), "stride_outdoor": self.stride_outdoor,
                "stride_indoor": self.stride_indoor, "mult": self.mult}

    @property
    def sel_hash(self) -> str:
        return stable_hash(self.params())[:16]


def role_of(m: Member) -> str:
    """"val" (evaluation only) | "dev" (checkpoint selection, scenes held out of training) | "train"."""
    if m.split == "val":
        return "val"
    return "dev" if m.scene in DIODE_DEV_SCENES else "train"


def _depth_mask_arrays(depth, mask) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth)
    m = np.asarray(mask)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if d.ndim != 2 or d.shape != m.shape:
        raise ValueError(f"bad depth/mask shapes {d.shape} {m.shape}")
    df = d.astype(np.float64)
    valid = (m.astype(np.float64) > 0.5) & np.isfinite(df) & (df > 0)
    return df, valid


def reduce_gt_half(depth, mask, out_dtype=np.float16) -> Tuple[np.ndarray, np.ndarray]:
    """Full-resolution laser depth -> half resolution: ``depth_m`` f16 (H/2, W/2) = geometric mean of the valid
    pixels of each 2x2 cell (0 if none) and ``vcnt`` u8 = number of valid pixels (0..4)."""
    df, valid = _depth_mask_arrays(depth, mask)
    h, w = df.shape
    if h % 2 or w % 2:
        raise ValueError("odd depth size")
    ld = np.where(valid, np.log(np.where(valid, df, 1.0)), 0.0)
    cnt = valid.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    s = ld.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    gm = np.where(cnt > 0, np.exp(s / np.maximum(cnt, 1)), 0.0)
    return np.clip(gm, 0.0, 60000.0).astype(out_dtype), cnt.astype(np.uint8)


def pack_val_gt(depth, mask) -> Dict[str, np.ndarray]:
    """Validation ground truth at full resolution: ``depth_m`` f16 (0 where invalid) + bit-packed ``valid``."""
    df, valid = _depth_mask_arrays(depth, mask)
    return {"depth_m": np.clip(np.where(valid, df, 0.0), 0.0, 60000.0).astype(np.float16),
            "valid": np.packbits(valid, axis=-1)}


def unpack_val_valid(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1)[:, :width].astype(bool)


def lab4_from_half(depth_m: np.ndarray, vcnt: np.ndarray, lo: float = DEPTH_LO, hi: float = DEPTH_HI):
    """Half-resolution GT -> H/4 labels with the same rule as ``downsample_labels(stride=4)``: a block is valid only
    when all its 16 full-resolution pixels are (sum of ``vcnt`` over 2x2 == 16); log depth is the block mean of ln d
    (the half-resolution value is already the geometric mean of its valid pixels).  Depth is clamped to [lo, hi]."""
    h, w = depth_m.shape
    if h % 2 or w % 2 or vcnt.shape != depth_m.shape:
        raise ValueError("bad half-resolution GT")
    c = vcnt.astype(np.int32).reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    mask = c == 16
    ld = np.log(np.clip(depth_m.astype(np.float32), lo, hi))
    ld4 = ld.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return (ld4 * mask).astype(np.float32), mask.astype(np.uint8)


def half_ln_stats(depth_m: np.ndarray, vcnt: np.ndarray) -> Tuple[float, float]:
    """(sum of ln d over valid full-res pixels, number of valid pixels) of one half-resolution GT."""
    d = depth_m.astype(np.float64)
    ok = (vcnt > 0) & (d > 0)
    return float((np.log(np.where(ok, d, 1.0)) * vcnt).sum()), float(vcnt[ok].sum())


# ---------------------------------------------------------------------------
# streaming download with Range + If-Match resume
# ---------------------------------------------------------------------------

class HttpFatal(RuntimeError):
    """Not worth retrying (412 / 416 / 403, range ignored, too many retries)."""


class _Retry(Exception):
    pass


_NET_ERRORS = (OSError, http.client.HTTPException, ssl.SSLError, TimeoutError)


class ResumableHTTP(io.RawIOBase):
    """Read-only raw stream over one URL.  A dropped TCP connection resumes at the same compressed offset with
    ``Range: bytes=<pos>-`` and ``If-Match: <etag>``; 412 / 416 / 403 are fatal; 429 and 5xx retry; at most
    ``max_retries`` consecutive failures with back-off min(60, 2*2^k) seconds.  The md5 of the bytes handed out is
    observed on the fly (contiguous across reconnects)."""

    def __init__(self, url: str, *, max_retries: int = 8, sleep: Callable[[float], None] = time.sleep,
                 opener: Optional[Callable] = None, timeout: float = 60.0):
        super().__init__()
        self.url = url
        self.max_retries = int(max_retries)
        self.sleep = sleep
        self.timeout = timeout
        self._open_fn = opener or urllib.request.urlopen
        self.pos = 0
        self.size: Optional[int] = None
        self.etag: Optional[str] = None
        self.reconn = 0
        self.md5 = hashlib.md5()
        self._resp = None
        self._fails = 0

    def readable(self) -> bool:
        return True

    def _close_resp(self) -> None:
        if self._resp is not None:
            with contextlib.suppress(Exception):
                self._resp.close()
        self._resp = None

    def close(self) -> None:
        self._close_resp()
        super().close()

    def _open(self) -> None:
        headers = {"User-Agent": "buc-m3-spike/1"}
        if self.pos > 0:
            headers["Range"] = f"bytes={self.pos}-"
            if self.etag:
                headers["If-Match"] = self.etag
        req = urllib.request.Request(self.url, headers=headers)
        try:
            resp = self._open_fn(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (412, 416, 403):
                raise HttpFatal(f"http {e.code}") from None
            if e.code == 429 or e.code >= 500:
                raise _Retry(f"http {e.code}") from None
            raise HttpFatal(f"http {e.code}") from None
        except _NET_ERRORS as e:
            raise _Retry(err_str(e)) from None
        status = getattr(resp, "status", None) or resp.getcode()
        hdr = resp.headers
        if self.pos > 0:
            cr = hdr.get("Content-Range") or ""
            if status != 206 or not cr.startswith(f"bytes {self.pos}-"):
                with contextlib.suppress(Exception):
                    resp.close()
                raise HttpFatal("server ignored the range request")
            et = hdr.get("ETag")
            if self.etag and et and et != self.etag:
                with contextlib.suppress(Exception):
                    resp.close()
                raise HttpFatal("etag changed")
        else:
            cl = hdr.get("Content-Length")
            self.size = int(cl) if cl and cl.isdigit() else None
            self.etag = hdr.get("ETag")
        self._resp = resp

    def _again(self, why) -> None:
        self._close_resp()
        self._fails += 1
        self.reconn += 1
        if self._fails > self.max_retries:
            raise HttpFatal(f"too many retries ({scrub(why)[:100]})")
        self.sleep(min(60.0, 2.0 * 2 ** (self._fails - 1)))

    def readinto(self, b) -> int:
        if self.size is not None and self.pos >= self.size:
            return 0
        n = len(b)
        while True:
            try:
                if self._resp is None:
                    self._open()
                data = self._resp.read(n)
                if not data:
                    if self.size is not None and self.pos < self.size:
                        raise _Retry("premature end of stream")
                    return 0
            except _Retry as e:
                self._again(e)
                continue
            except _NET_ERRORS as e:
                self._again(err_str(e))
                continue
            b[:len(data)] = data
            self.pos += len(data)
            self.md5.update(data)
            self._fails = 0
            return len(data)


# ---------------------------------------------------------------------------
# DIODE ingest
# ---------------------------------------------------------------------------

class DiskLowError(RuntimeError):
    pass


def diode_license_text() -> str:
    return (
        "DIODE: A Dense Indoor and Outdoor DEpth Dataset (https://diode-dataset.org)\n"
        "Vasiljevic et al., arXiv:1908.00463.  Released under the MIT License.\n\n"
        "MIT License.  Copyright (c) the DIODE dataset authors.\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this dataset and associated "
        "documentation files (the \"Dataset\"), to deal in the Dataset without restriction, including without "
        "limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of "
        "the Dataset, and to permit persons to whom the Dataset is furnished to do so, subject to the following "
        "condition: the above copyright notice and this permission notice shall be included in all copies or "
        "substantial portions of the Dataset.\n\n"
        "THE DATASET IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT "
        "LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n\n"
        "The authoritative licence text is the one published with the dataset; this file records the terms the "
        "job relied on.  Local use: val = evaluation only (tag HOLDOUT:diode_val); train taps are held back by the "
        "FND-04 firewall until MIT is added to TRAIN_LICENSES.\n")


def write_diode_docs(paths: Paths) -> None:
    os.makedirs(paths.diode_val, exist_ok=True)
    p = os.path.join(paths.diode, "LICENSE_DIODE.txt")
    if not os.path.isfile(p):
        write_atomic(p, diode_license_text().encode("utf-8"))
    q = os.path.join(paths.diode_val, "EVAL_ONLY.txt")
    if not os.path.isfile(q):
        write_atomic(q, b"DIODE val: evaluation only (HOLDOUT:diode_val). Never used for training, "
                        b"PCA fitting or checkpoint selection.\n")


class DiodeIngest:
    """Consumes tar members of one split.  The three files of a crop (png, depth, depth_mask) arrive in arbitrary
    order inside their scan folder; the png is written at once, depth and mask are buffered until both are there,
    then ``<stem>.gt.npz`` is written LAST (it is the completion marker; a stem with a ``.gt.npz`` is skipped)."""

    def __init__(self, paths: Paths, split: str, selection: Selection, *, expected_png: Optional[int] = None,
                 max_member_bytes: int = 96 << 20, disk_check_every: int = 200,
                 free_disk_fn: Callable[[str], float] = free_disk_gb):
        self.paths = paths
        self.split = split
        self.sel = selection
        self.dest = paths.diode_train if split == "train" else paths.diode_val
        self.expected_png = expected_png
        self.max_member_bytes = int(max_member_bytes)
        self.stats = {"seen": 0, "kept": 0, "written": 0, "skip": 0, "bad": 0}
        self.pending: Dict[str, dict] = {}
        self.bad_stems: set = set()
        self.cur_scan = None
        self._disk_every = int(disk_check_every)
        self._free_disk = free_disk_fn
        self._since_disk = 0

    def stem_dir(self, m: Member) -> str:
        return os.path.join(self.dest, m.rel_dir)

    def gt_path(self, m: Member) -> str:
        return os.path.join(self.stem_dir(m), m.stem + ".gt.npz")

    def png_path(self, m: Member) -> str:
        return os.path.join(self.stem_dir(m), m.stem + ".png")

    def _evict(self) -> None:
        if self.pending:
            self.stats["bad"] += len(self.pending)
            self.pending.clear()

    def _mark_bad(self, m: Member) -> None:
        if m.stem not in self.bad_stems:
            self.bad_stems.add(m.stem)
            self.stats["bad"] += 1
        self.pending.pop(m.stem, None)

    def handle(self, name: str, size: int, read: Callable[[], bytes]) -> None:
        m = parse_member(name)
        if m is None or m.split != self.split:
            return
        if m.scan_key != self.cur_scan:
            self._evict()
            self.cur_scan = m.scan_key
        if m.kind == "rgb":
            self.stats["seen"] += 1
        if not self.sel.keep(m):
            return
        if m.kind == "rgb":
            self.stats["kept"] += 1
        if m.stem in self.bad_stems:
            return
        if os.path.exists(self.gt_path(m)):
            if m.kind == "rgb":
                self.stats["skip"] += 1
            return
        if size > self.max_member_bytes:
            self._mark_bad(m)
            return
        data = read()
        st = self.pending.setdefault(m.stem, {"m": m})
        try:
            if m.kind == "rgb":
                os.makedirs(self.stem_dir(m), exist_ok=True)
                write_atomic(self.png_path(m), data)
                st["rgb"] = True
            else:
                st[m.kind] = np.load(io.BytesIO(data), allow_pickle=False)
            if all(k in st for k in ("rgb", "depth", "mask")):
                self._finish(m, st)
        except DiskLowError:
            raise
        except Exception as e:  # noqa: BLE001 - one odd crop must not stop the stream
            warn("diode_member_bad", stem=m.stem, err=err_str(e))
            self._mark_bad(m)

    def _finish(self, m: Member, st: dict) -> None:
        if self.split == "train":
            dm, vc = reduce_gt_half(st["depth"], st["mask"])
            blob = npz_bytes({"depth_m": dm, "vcnt": vc})
        else:
            blob = npz_bytes(pack_val_gt(st["depth"], st["mask"]))
        write_atomic(self.gt_path(m), blob)          # last: the completion marker
        self.pending.pop(m.stem, None)
        self.stats["written"] += 1
        self._since_disk += 1
        if self._since_disk >= self._disk_every:
            self._since_disk = 0
            if disk_policy(self._free_disk(self.dest)) == "stop":
                raise DiskLowError("free disk below 8 GB: DIODE writing stopped")

    def finish(self) -> None:
        self._evict()

    def complete(self) -> bool:
        return self.expected_png is None or self.stats["seen"] == self.expected_png


def stream_diode(ingest: DiodeIngest, raw: "ResumableHTTP", *, now: Callable[[], float] = time.time,
                 progress_every_s: float = 60.0, stop: Callable[[], bool] = lambda: False,
                 bufsize: int = 4 << 20) -> str:
    """Stream ``raw`` (a .tar.gz byte stream) through the ingest.  Returns "eof" or "stopped"."""
    br = io.BufferedReader(raw, buffer_size=bufsize)
    t0 = last = now()
    last_pos = 0
    with tarfile.open(fileobj=br, mode="r|gz", bufsize=bufsize) as tar:
        for ti in tar:
            if stop():
                return "stopped"
            if not ti.isfile():
                continue
            ingest.handle(ti.name, ti.size, lambda ti=ti: tar.extractfile(ti).read())
            t = now()
            if t - last >= progress_every_s:
                mbps = (raw.pos - last_pos) / max(t - last, 1e-6) / 1e6
                log({"split": ingest.split, **ingest.stats, "pos_gb": gb(raw.pos), "mbps": round(mbps, 1),
                     "reconn": raw.reconn}, "_DATA")
                last, last_pos = t, raw.pos
    ingest.finish()
    log({"split": ingest.split, **ingest.stats, "pos_gb": gb(raw.pos), "reconn": raw.reconn, "eof": True,
         "t": round(now() - t0, 1)}, "_DATA")
    return "eof"


def choose_selection(paths: Paths, free_gb: Optional[float] = None) -> Selection:
    """The selection is decided once (disk policy) and persisted in SOURCE_DIODE.json so restarts agree."""
    src = read_json(paths.source_diode, {}) or {}
    sel = src.get("selection")
    if isinstance(sel, dict) and sel.get("profile") == "spike":
        return Selection(mult=int(sel.get("mult", 1)))
    fg = free_disk_gb(paths.out) if free_gb is None else free_gb
    pol = disk_policy(fg)
    if pol == "stop":
        raise DiskLowError(f"only {fg:.1f} GB free")
    s = Selection(mult=2 if pol == "double" else 1)
    src["selection"] = s.params()
    src["sel_hash"] = s.sel_hash
    write_json_atomic(paths.source_diode, src)
    return s


def update_source(paths: Paths, split: str, info: dict) -> None:
    src = read_json(paths.source_diode, {}) or {}
    src[split] = info
    src.setdefault("selection", info.get("selection"))
    src.setdefault("sel_hash", info.get("sel_hash"))
    write_json_atomic(paths.source_diode, src)


def run_diode_split(paths: Paths, split: str, *, url: Optional[str] = None, selection: Optional[Selection] = None,
                    expected_png: Optional[int] = None, raw_factory: Optional[Callable] = None,
                    now: Callable[[], float] = time.time, stop: Callable[[], bool] = lambda: False,
                    sleep: Callable[[float], None] = time.sleep, progress_every_s: float = 60.0) -> dict:
    """Stream one DIODE split.  Done marker ``stages/diode_<split>.done.json`` only after the whole stream was read
    and the PNG member count matches (train 25,458 / val 771)."""
    name = f"diode_{split}"
    sel = selection or choose_selection(paths)
    exp = DIODE_EXPECTED_PNG[split] if expected_png is None else expected_png
    cfg = stable_hash(name, sel.params(), exp)
    rec = stage_record(paths, name)
    if rec is not None and rec.get("config_hash") == cfg:
        return {"status": "done", **(rec.get("counts") or {})}
    write_diode_docs(paths)
    url = url or f"{DIODE_BASE}/{split}.tar.gz"
    ingest = DiodeIngest(paths, split, sel, expected_png=exp)
    raw = raw_factory(url) if raw_factory else ResumableHTTP(url, sleep=sleep)
    status = stream_diode(ingest, raw, now=now, progress_every_s=progress_every_s, stop=stop)
    out = {"status": status, **ingest.stats, "reconn": raw.reconn, "pos_gb": gb(raw.pos)}
    if status == "eof" and ingest.complete():
        info = {"url": url, "etag": raw.etag, "size": raw.size, "md5_observed": raw.md5.hexdigest(),
                "sel_hash": sel.sel_hash, "selection": sel.params(), "png_members": ingest.stats["seen"],
                "written": ingest.stats["written"], "bad": ingest.stats["bad"]}
        update_source(paths, split, info)
        write_stage_done(paths, name, cfg, {**ingest.stats, "md5_observed": raw.md5.hexdigest()})
        out["status"] = "done"
    elif status == "eof":
        out["status"] = "member_count_mismatch"
        warn("diode_count_mismatch", split=split, seen=ingest.stats["seen"], expected=exp)
    return out


def list_diode_samples(paths: Paths, split: str) -> List[Tuple[Member, str]]:
    """Finished crops (``.gt.npz`` present) under data/diode/<split>, sorted by stem path."""
    root = paths.diode_train if split == "train" else paths.diode_val
    out = []
    for d, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".gt.npz"):
                continue
            stem = f[:-len(".gt.npz")]
            rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
            m = parse_member(f"{split}/{rel[:-len('.gt.npz')]}.png")
            if m is None or m.stem != stem:
                continue
            if os.path.isfile(os.path.join(d, stem + ".png")):
                out.append((m, os.path.join(d, stem)))
    out.sort(key=lambda t: (t[0].env, t[0].scene, t[0].scan, t[0].yaw, t[0].pitch))
    return out


def balanced_prefix(cap: Optional[int], samples: Sequence[Tuple[Member, str]]) -> List[Tuple[Member, str]]:
    """Scene-balanced prefix: round-robin over scenes (sorted) so a cap keeps every scene represented."""
    if cap is None or cap >= len(samples):
        return list(samples)
    by: Dict[str, List[Tuple[Member, str]]] = {}
    for it in samples:
        by.setdefault(it[0].scene, []).append(it)
    scenes = sorted(by)
    out: List[Tuple[Member, str]] = []
    i = 0
    while len(out) < cap and any(i < len(by[s]) for s in scenes):
        for s in scenes:
            if i < len(by[s]) and len(out) < cap:
                out.append(by[s][i])
        i += 1
    return sorted(out, key=lambda t: (t[0].env, t[0].scene, t[0].scan, t[0].yaw, t[0].pitch))


# ---------------------------------------------------------------------------
# synthetic renders (RoomGen / DAT-01) - CPU
# ---------------------------------------------------------------------------

SYNTH_KEY_RE = re.compile(r"r(\d{6})v(\d)")


def synth_key(seed: int, view: int) -> str:
    return f"r{int(seed):06d}v{int(view)}"


def parse_synth_key(key: str) -> Optional[Tuple[int, int]]:
    m = SYNTH_KEY_RE.fullmatch(key) if isinstance(key, str) else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def tone_map(rgb_lin: np.ndarray, hit: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-view auto exposure + Reinhard: ``x = rgb * 0.35 / p70(lum)``, ``display(x / (1 + x))``.
    (``rgb_display()`` is not used: it blows out 31% of the views.)  Returns float32 [0, 1]."""
    from buc.core import color
    rgb = np.asarray(rgb_lin, dtype=np.float64)
    lum = np.asarray(color.luminance(rgb))
    ref = lum[hit] if hit is not None and np.any(hit) else lum.reshape(-1)
    p70 = max(float(np.percentile(ref, 70)), 1e-6)
    x = rgb * (0.35 / p70)
    return color.linear_to_display(x / (1.0 + x))


def render_views(seed: int, views: Sequence[int] = tuple(range(SYNTH_VIEWS)), w: int = SYNTH_WH[0],
                 h: int = SYNTH_WH[1], max_attempts: int = 3) -> Optional[List[dict]]:
    """Render the views of one RoomGen seed.  None when ``generate_room`` rejects the seed (ValueError, ~0.37% of
    seeds).  A view whose camera sees almost nothing ((inst>=0).mean() < 0.02) is retried with another camera seed
    (at most ``max_attempts`` tries), then dropped."""
    from buc.invent.grammar.room import generate_room
    try:
        prog = generate_room(int(seed))
    except ValueError:
        return None
    out = []
    for v in views:
        got = None
        for att in range(max_attempts):
            vs = int(v) + 2 * att
            rng = np.random.default_rng([int(seed), vs])
            fov = float(rng.uniform(55.0, 90.0))
            cam = SR.default_camera(prog, vs, width=w, height=h, fov_x_deg=fov)
            s = SR.render_gbuffer(prog, cam)
            if float((s.inst >= 0).mean()) >= 0.02:
                got = (s, vs, fov, att)
                break
        if got is None:
            out.append({"view": int(v), "empty": True})
            continue
        s, vs, fov, att = got
        disp = tone_map(s.rgb, s.inst >= 0)
        out.append({"view": int(v), "empty": False, "gbuf": s.gbuf.astype(np.float16),
                    "sem": s.sem.astype(np.int16), "image": np.round(disp * 255.0).astype(np.uint8),
                    "meta": {"seed": int(seed), "view": int(v), "cam_seed": vs, "attempt": att, "fov_x_deg": round(fov, 4),
                             "program_hash": s.meta["program_hash"], "w": w, "h": h}})
    return out


def _png_bytes(u8: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(u8, "RGB").save(bio, format="PNG", compress_level=6)
    return bio.getvalue()


def synth_file(dirpath: str, key: str, ext: str) -> str:
    return os.path.join(dirpath, key + ext)


def synth_done(dirpath: str, key: str) -> bool:
    return os.path.isfile(synth_file(dirpath, key, ".json"))


def write_synth_sample(dirpath: str, key: str, rec: dict) -> None:
    """npz, png, then json LAST (the completion marker); every file is written atomically."""
    os.makedirs(dirpath, exist_ok=True)
    write_atomic(synth_file(dirpath, key, ".npz"), npz_bytes({"gbuf": rec["gbuf"], "sem": rec["sem"]}))
    write_atomic(synth_file(dirpath, key, ".png"), _png_bytes(rec["image"]))
    write_atomic(synth_file(dirpath, key, ".json"),
                 json.dumps(rec["meta"], sort_keys=True, separators=(",", ":")).encode("utf-8"))


def load_synth_sample(dirpath: str, key: str) -> dict:
    z = load_npz(synth_file(dirpath, key, ".npz"))
    with Image.open(synth_file(dirpath, key, ".png")) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.uint8)
    meta = read_json(synth_file(dirpath, key, ".json"), {})
    return {"gbuf": z["gbuf"], "sem": z["sem"], "image": img.astype(np.float32) / 255.0, "meta": meta}


def list_synth_keys(dirpath: str) -> List[str]:
    try:
        names = os.listdir(dirpath)
    except OSError:
        return []
    return sorted(n[:-5] for n in names if n.endswith(".json") and SYNTH_KEY_RE.fullmatch(n[:-5]))


def _render_job(job: tuple) -> dict:
    dirpath, seed, views, w, h = job
    try:
        recs = render_views(seed, views, w, h)
    except Exception as e:  # noqa: BLE001
        return {"seed": seed, "status": "error", "err": err_str(e)}
    if recs is None:
        return {"seed": seed, "status": "skip_seed"}
    n, empty = 0, []
    for r in recs:
        key = synth_key(seed, r["view"])
        if r["empty"]:
            empty.append(key)
            continue
        write_synth_sample(dirpath, key, r)
        n += 1
    return {"seed": seed, "status": "ok", "n": n, "empty": empty}


def _render_worker_init() -> None:
    with contextlib.suppress(Exception):
        os.nice(10)
    with contextlib.suppress(Exception):
        torch.set_num_threads(1)


def render_jobs(paths: Paths, train_range: Tuple[int, int], dev_range: Tuple[int, int], wh: Tuple[int, int],
                skip_seeds: Iterable[int] = ()) -> List[tuple]:
    """Ordered jobs: the first 200 train seeds (they feed the PCA fit), then dev, then the rest of train."""
    skip = set(skip_seeds)
    w, h = wh
    train = [s for s in range(*train_range) if s not in skip]
    dev = [s for s in range(*dev_range) if s not in skip]
    jobs = []
    for s in train[:200]:
        jobs.append((paths.synth_train, s, tuple(range(SYNTH_VIEWS)), w, h))
    for s in dev:
        jobs.append((paths.synth_dev, s, (0,), w, h))
    for s in train[200:]:
        jobs.append((paths.synth_train, s, tuple(range(SYNTH_VIEWS)), w, h))
    out = []
    for d, s, views, w_, h_ in jobs:
        todo = tuple(v for v in views if not synth_done(d, synth_key(s, v)))
        if todo:
            out.append((d, s, todo, w_, h_))
    return out


def run_render(paths: Paths, *, procs: int = 6, train_range: Tuple[int, int] = SYNTH_TRAIN_SEEDS,
               dev_range: Tuple[int, int] = SYNTH_DEV_SEEDS, wh: Tuple[int, int] = SYNTH_WH,
               stop: Callable[[], bool] = lambda: False, now: Callable[[], float] = time.time,
               progress_every_s: float = 60.0) -> dict:
    """Render all synthetic samples (resumable: finished samples and rejected seeds are skipped)."""
    cfg = stable_hash("render", list(train_range), list(dev_range), list(wh), SYNTH_VIEWS)
    rec = stage_record(paths, "render")
    if rec is not None and rec.get("config_hash") == cfg:
        return {"status": "done", **(rec.get("counts") or {})}
    pg = paths.progress("render")
    prior = read_jsonl(pg)
    skip_seeds = {r["seed"] for r in prior if r.get("status") == "skip_seed"}
    empty = {k for r in prior for k in (r.get("empty") or [])}
    jobs = [j for j in render_jobs(paths, train_range, dev_range, wh, skip_seeds)]
    jobs = [(d, s, tuple(v for v in vs if synth_key(s, v) not in empty), w, h) for d, s, vs, w, h in jobs]
    jobs = [j for j in jobs if j[2]]
    total_new = len(jobs)
    done, bad, t0 = 0, 0, now()
    last = t0
    status = "done"

    def consume(res: dict) -> None:
        nonlocal done, bad
        done += 1
        if res["status"] == "ok":
            if res.get("empty"):
                append_jsonl(pg, {"seed": res["seed"], "status": "ok", "empty": res["empty"]})
        elif res["status"] == "skip_seed":
            append_jsonl(pg, {"seed": res["seed"], "status": "skip_seed"})
        else:
            bad += 1
            warn("render_failed", seed=res["seed"], err=res.get("err", ""))

    if procs <= 1:
        for j in jobs:
            if stop():
                status = "stopped"
                break
            consume(_render_job(j))
            t = now()
            if t - last >= progress_every_s:
                last = t
                log({"synth": "render", "done": done, "of": total_new, "rate": round(done / max(t - t0, 1e-6), 2)}, "_DATA")
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(procs, initializer=_render_worker_init)
        try:
            for res in pool.imap(_render_job, jobs, chunksize=2):
                consume(res)
                t = now()
                if t - last >= progress_every_s:
                    last = t
                    log({"synth": "render", "done": done, "of": total_new,
                         "rate": round(done / max(t - t0, 1e-6), 2)}, "_DATA")
                if stop():
                    status = "stopped"
                    break
        finally:
            pool.terminate()
            pool.join()
    counts = {"train": len(list_synth_keys(paths.synth_train)), "dev": len(list_synth_keys(paths.synth_dev)),
              "skip_seeds": len(skip_seeds | {r["seed"] for r in read_jsonl(pg) if r.get("status") == "skip_seed"}),
              "bad": bad}
    log({"synth": "render_done", **counts, "status": status, "t": round(now() - t0, 1)}, "_DATA")
    if status == "done" and bad == 0:
        write_stage_done(paths, "render", cfg, counts)
    return {"status": status if bad == 0 else "partial", **counts}


# ---------------------------------------------------------------------------
# GPU gate, plan tiers, release verification
# ---------------------------------------------------------------------------

def mem_get_info_default() -> Tuple[int, int]:
    return torch.cuda.mem_get_info()


def m6_state(root: str) -> str:
    """Reference only (the gate decision always uses ``mem_get_info``): did M6 log that it released the teacher?"""
    p = os.path.join(root, "logs", "m6.run.log")
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 400_000))
            txt = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return "no_log"
    return "released" if "teacher released" in txt else "running"


def wait_for_gate(mem_fn: Callable[[], Tuple[int, int]], *, need: float = GATE_BYTES, now: Callable[[], float] = time.time,
                  sleep: Callable[[float], None] = time.sleep, deadline: Optional[float] = None,
                  poll_s: float = 60.0, report_s: float = 300.0, m6_fn: Callable[[], str] = lambda: "?",
                  min_remaining_s: float = 1.25 * 3600) -> Optional[dict]:
    """Poll the free VRAM every ``poll_s`` seconds until ``free >= need``.  A ``gpu_wait`` marker every ``report_s``.
    None when the deadline gets closer than ``min_remaining_s`` before the gate opened (nothing fits any more)."""
    last_report = None
    while True:
        free = float(mem_fn()[0])
        if free >= need:
            return {"free_gb": gb(free)}
        t = now()
        if deadline is not None and deadline - t < min_remaining_s:
            return None
        if last_report is None or t - last_report >= report_s:
            log({"stage": "gpu_wait", "free_gb": gb(free), "m6": m6_fn()})
            last_report = t
        sleep(poll_s)


def plan_for(remaining_s: float) -> Optional[dict]:
    """Time-budget tier, decided when the gate opens (R = deadline - now)."""
    h = remaining_s / 3600.0
    if h >= 4.5:
        return {"tier": "full", "f2b_cap": None, "f2c_cap": None, "train_budget_s": None}
    if h >= 2.5:
        return {"tier": "mid", "f2b_cap": 3000, "f2c_cap": 2000, "train_budget_s": None}
    if h >= 1.25:
        return {"tier": "short", "f2b_cap": 1500, "f2c_cap": 800, "train_budget_s": 2400.0}
    return None


def release_report(alloc: int, reserved: int, free_loaded: float, free_after: float,
                   min_recover: float = 10e9) -> dict:
    """After the teacher is dropped: nothing allocated / reserved by this process and >= 10 GB came back."""
    rec = float(free_after) - float(free_loaded)
    ok = int(alloc) == 0 and int(reserved) == 0 and rec >= min_recover
    return {"released": bool(ok), "alloc": int(alloc), "reserved": int(reserved), "recovered_gb": gb(rec)}


def cuda_ready() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def sync_cuda() -> None:
    if cuda_ready():
        torch.cuda.synchronize()


def free_cuda_cache() -> None:
    gc.collect()
    if cuda_ready():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def apply_vram_cap(gib: float) -> Optional[float]:
    """``torch.cuda.set_per_process_memory_fraction`` so M2 + M3 + the rest fit next to each other."""
    if not cuda_ready() or gib <= 0:
        return None
    try:
        idx = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory
        frac = min(1.0, gib * 2 ** 30 / float(total))
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        return round(frac * total / 2 ** 30, 2)
    except Exception as e:  # noqa: BLE001
        warn("mem_cap_failed", err=err_str(e, 300))
        return None


def vram_gb() -> float:
    try:
        if cuda_ready():
            return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def is_oom(e: BaseException) -> bool:
    if isinstance(e, getattr(torch, "OutOfMemoryError", ())):
        return True
    m = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in m or "alloc_failed" in m)


# ---------------------------------------------------------------------------
# teacher loading (two backends), caption embedding
# ---------------------------------------------------------------------------

def hub_cache_candidates() -> List[str]:
    c: List[str] = []
    if os.environ.get("HF_HUB_CACHE"):
        c.append(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        c.append(str(Path(os.environ["HF_HOME"]) / "hub"))
    with contextlib.suppress(Exception):
        from huggingface_hub import constants as hc
        c.append(str(hc.HF_HUB_CACHE))
    c += [str(Path.home() / ".cache" / "huggingface" / "hub"), "/workspace/buc/hf/hub", "/workspace/hf/hub",
          "/root/.cache/huggingface/hub"]
    root = os.environ.get("N1_ROOT") or DEFAULT_ROOT
    c += [str(Path(root) / "hf" / "hub"), str(Path(root) / "hf_cache")]
    seen, out = set(), []
    for x in c:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_teacher_cache(snapshot_dir_fn: Callable, spec, required=("transformer", "vae", "text_encoder", "tokenizer")):
    """-> (hub_cache, revision | None): first cache whose ``refs/main`` snapshot is complete, else the newest complete
    snapshot folder found in any candidate (revision pinned explicitly).  (pipe_yak logic.)"""
    last = None
    cands = hub_cache_candidates()
    for c in cands:
        try:
            snapshot_dir_fn(spec, hub_cache=c)
            return c, None
        except Exception as e:  # noqa: BLE001
            last = e
    repo_dir = "models--" + spec.repo.replace("/", "--")
    for c in cands:
        snaps = Path(c) / repo_dir / "snapshots"
        try:
            dirs = sorted((d for d in snaps.iterdir() if d.is_dir() and all((d / r).is_dir() for r in required)),
                          key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            continue
        if dirs:
            return c, dirs[0].name
    raise FileNotFoundError(f"Z-Image-Turbo snapshot not found in any HF cache ({err_str(last) if last else '-'})")


class PlainDiffusersTeacher:
    """``ZImageTeacher`` duck type built from plain diffusers components (the M6 loader): no determinism mode, no
    TCH-01 wrappers.  The text encoder is gone after construction: only the cached survey-caption embedding exists."""

    backend = "diffusers"

    def __init__(self, transformer, vae, cap_cpu: torch.Tensor, *, spec, device, dtype, snapshot=None):
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = dtype
        self.transformer = transformer
        self.layers = transformer.layers
        cfg = vae.config
        self.vae = Z.VaeCodec(vae, scaling_factor=float(cfg.scaling_factor),
                              shift_factor=0.0 if getattr(cfg, "shift_factor", None) is None else float(cfg.shift_factor),
                              vae_scale=spec.vae_scale, latent_channels=spec.latent_channels)
        self.snapshot = snapshot
        self._cap = cap_cpu.detach().to("cpu")

    def encode_text(self, prompts) -> list:
        prompts = list(prompts)
        if any(p != SURVEY_CAPTION for p in prompts):
            raise Z.TEUnloadedError("plain backend only holds the survey caption embedding")
        return [self._cap.to(device=self.device, dtype=self.dtype) for _ in prompts]

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    def decode_latent(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        return self.vae.decode(z, clamp=clamp)

    def make_noise(self, seed, batch, height, width, *, level=0, k=0) -> torch.Tensor:
        shape = self.spec.latent_shape(height, width, batch)
        g = torch.Generator("cpu")
        g.manual_seed(int(seed) % (1 << 63))
        return torch.randn(shape, generator=g, dtype=torch.float32).to(self.device)

    def release_text_encoder(self) -> None:
        return None


def read_backend(paths: Paths) -> Optional[dict]:
    d = read_json(paths.backend)
    return d if isinstance(d, dict) and d.get("backend") in ("buc", "diffusers") else None


def write_backend(paths: Paths, name: str, why: str, **extra) -> None:
    write_json_atomic(paths.backend, {"backend": name, "why": why, "det": False, "noise": "wnf" if name == "buc" else "torch_cpu",
                                      **extra})


def load_cap_file(paths: Paths) -> Optional[torch.Tensor]:
    ck = load_ckpt(paths.cap_embed, need=("cap", "caption"))
    if ck is None or ck.get("caption") != SURVEY_CAPTION or not isinstance(ck["cap"], torch.Tensor):
        return None
    return ck["cap"]


def save_cap_file(paths: Paths, cap: torch.Tensor, backend: str) -> None:
    save_atomic({"caption": SURVEY_CAPTION, "cap": cap.detach().to("cpu").clone(), "backend": backend}, paths.cap_embed)


def load_teacher_real(paths: Paths, backend: str, device: str = "cuda"):
    """-> teacher object.  ``backend`` "buc" = ``buc.teacher.zimage.load_teacher`` (non-deterministic fast mode);
    "diffusers" = plain ZImageTransformer2DModel / AutoencoderKL, caption embedding from ZImagePipeline."""
    spec = Z.TeacherSpec()
    hub, rev = find_teacher_cache(Z.snapshot_dir, spec)
    if rev:
        spec = Z.TeacherSpec(revision=rev)
    if backend == "buc":
        os.makedirs(paths.te_cache, exist_ok=True)
        return Z.load_teacher(spec, device=device, te_device=device, hub_cache=hub, seed=0, deterministic=False,
                              te_cache_dir=paths.te_cache)
    snap = Z.snapshot_dir(spec, hub_cache=hub)
    dtype = torch.bfloat16
    cap = load_cap_file(paths)
    from diffusers import AutoencoderKL, ZImageTransformer2DModel
    if cap is None:
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(str(snap), torch_dtype=dtype, low_cpu_mem_usage=False)
        pipe.to(device)
        with torch.inference_mode():
            embeds, _ = pipe.encode_prompt(SURVEY_CAPTION, device=torch.device(device), do_classifier_free_guidance=False)
        cap = embeds[0].detach().to("cpu", dtype).clone()
        transformer, vae = pipe.transformer, pipe.vae
        pipe.text_encoder = None
        del pipe
    else:
        transformer = ZImageTransformer2DModel.from_pretrained(str(snap), subfolder="transformer", torch_dtype=dtype).to(device)
        vae = AutoencoderKL.from_pretrained(str(snap), subfolder="vae", torch_dtype=dtype).to(device)
    transformer.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    gc.collect()
    free_cuda_cache()
    return PlainDiffusersTeacher(transformer, vae, cap, spec=spec, device=device, dtype=dtype, snapshot=snap)


def caption_embedding(teacher, paths: Paths, backend: str) -> torch.Tensor:
    """The survey-caption embedding, on the teacher's device / dtype.  Encoded once, cached in cap_embed.pt (a
    restart with the cache never loads the text encoder), then the text encoder is released."""
    cached = load_cap_file(paths)
    if cached is None:
        cap = teacher.encode_text([SURVEY_CAPTION])[0]
        save_cap_file(paths, cap, backend)
    else:
        cap = cached
    with contextlib.suppress(Exception):
        teacher.release_text_encoder()
    free_cuda_cache()
    return cap.to(device=teacher.device, dtype=teacher.dtype)


# ---------------------------------------------------------------------------
# reading images through the teacher (RD-07 MrhReader procedure, one forward pass, early exit after block 23)
# ---------------------------------------------------------------------------

class _StopAfterTap(Exception):
    """Raised by the second hook on the last tapped block: blocks after it are skipped (about 20% faster)."""


class RawTap:
    """Forward hooks that keep the full hidden state ``[B, S, D]`` of the tapped blocks (F0 smoke, F1 PCA fit)."""

    def __init__(self, layers: Sequence[int] = TAP_LAYERS):
        self.layers = tuple(layers)
        self.store: Dict[int, torch.Tensor] = {}
        self.calls: Dict[int, int] = {l: 0 for l in self.layers}
        self.handles: list = []

    def attach(self, blocks) -> "RawTap":
        for l in self.layers:
            self.handles.append(blocks[l].register_forward_hook(self._make(l)))
        return self

    def _make(self, l: int):
        def hook(_m, _a, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            self.calls[l] += 1
            self.store[l] = h.detach()
            return None
        return hook

    def begin(self) -> None:
        self.store.clear()
        self.calls = {l: 0 for l in self.layers}

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.store.clear()


class TeacherReader:
    """Image -> int8 PCA taps (``TapBus``) or raw tokens, with the fixed read condition: sigma 0.3 (sampler table
    index 7), the single survey caption, token_offset 0, one transformer forward per batch."""

    def __init__(self, teacher, sampler, cap: torch.Tensor, *, layers: Sequence[int] = TAP_LAYERS, seed: int = SEED,
                 early_exit: bool = True, backend: str = "buc"):
        self.teacher = teacher
        self.sampler = sampler
        self.cap = cap
        self.layers = tuple(layers)
        self.seed = int(seed)
        self.sigma = float(sampler.table.snap(TAP_SIGMA).sigma)
        self.early_exit = bool(early_exit)
        self.backend = backend
        self.bus: Optional[T.TapBus] = None
        self.raw: Optional[RawTap] = None
        self._stop_handle = None
        self._armed = False

    # -- hooks --------------------------------------------------------------
    def _stop_hook(self, _m, _a, _out):
        if self._armed:
            raise _StopAfterTap()
        return None

    def _attach_stop(self) -> None:
        if self._stop_handle is not None:
            self._stop_handle.remove()
        self._stop_handle = self.teacher.layers[max(self.layers)].register_forward_hook(self._stop_hook)

    def use_bus(self, basis: T.PCABasis) -> "TeacherReader":
        self.close_hooks()
        self.bus = T.TapBus(basis, self.layers).attach(self.teacher.layers)
        self._attach_stop()                 # registered after the bus hooks: the tap is taken first
        return self

    def use_raw(self) -> "TeacherReader":
        self.close_hooks()
        self.raw = RawTap(self.layers).attach(self.teacher.layers)
        self._attach_stop()
        return self

    def close_hooks(self) -> None:
        if self.bus is not None:
            self.bus.detach()
            self.bus = None
        if self.raw is not None:
            self.raw.detach()
            self.raw = None
        if self._stop_handle is not None:
            self._stop_handle.remove()
            self._stop_handle = None

    # -- reading ------------------------------------------------------------
    def _prepare(self, x01: torch.Tensor, keys: Sequence[str]):
        if x01.ndim != 4 or x01.shape[1] != 3:
            raise ValueError(f"x01 must be [B,3,H,W]: {tuple(x01.shape)}")
        b, _, h, w = x01.shape
        if h % 16 or w % 16:
            raise ValueError(f"H and W must be multiples of 16: {h}x{w}")
        if len(keys) != b:
            raise ValueError("one key per image")
        x = x01.to(torch.float32) * 2.0 - 1.0
        x0 = self.teacher.encode_image(x.to(self.teacher.device))
        eps = torch.cat([self.teacher.make_noise(det.derive_seed(self.seed, k), 1, h, w) for k in keys]).to(x0.device)
        return TS.add_noise(x0, self.sigma, eps), h, w

    def _forward(self, z: torch.Tensor):
        self._armed = self.early_exit
        try:
            return self.sampler.predict_x0(z, self.sigma, [self.cap])
        except _StopAfterTap:
            return None
        finally:
            self._armed = False

    def read_tapset(self, x01: torch.Tensor, keys: Sequence[str], prov) -> "T.TapSet":
        if self.bus is None:
            raise RuntimeError("use_bus() first")
        with torch.no_grad():
            z, h, w = self._prepare(x01, keys)
            cap = self.bus.capture(prov, grid=(h // 16, w // 16), token_offset=0, key=str(keys[0]), ring=None)
            with cap:
                self._forward(z)
        return cap.tapset

    def read_int8(self, x01: torch.Tensor, keys: Sequence[str], prov) -> List[Dict[int, torch.Tensor]]:
        """Per image ``{layer: int8 [k, gh, gw]}`` (cpu copies)."""
        ts = self.read_tapset(x01, keys, prov)
        return [{l: ts.layers[l][i].detach().cpu().clone() for l in ts.layers} for i in range(len(keys))]

    def read_raw(self, x01: torch.Tensor, keys: Sequence[str], ids: Sequence[str], n_tokens: int = TOKENS_PER_IMG,
                 fit_seed: int = 1234) -> List[Dict[int, torch.Tensor]]:
        """Per image ``{layer: float32 [n_tokens, D]}``: ``n_tokens`` random image tokens (seeded per image id)."""
        if self.raw is None:
            raise RuntimeError("use_raw() first")
        with torch.no_grad():
            z, h, w = self._prepare(x01, keys)
            self.raw.begin()
            self._forward(z)
            n_img = (h // 16) * (w // 16)
            out = []
            for i, ident in enumerate(ids):
                g = torch.Generator("cpu")
                g.manual_seed(det.derive_seed(fit_seed, ident) % (1 << 63))
                idx = torch.randperm(n_img, generator=g)[:min(n_tokens, n_img)]
                out.append({l: self.raw.store[l][i, :n_img].index_select(0, idx.to(self.raw.store[l].device))
                            .float().cpu().clone() for l in self.layers})
        return out


def fixed_gradient_image(h: int = 512, w: int = 512) -> np.ndarray:
    """Smooth colour gradient used by the F0 smoke test (float32 HxWx3 in [0, 1])."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    img = np.stack([0.15 + 0.7 * xx, 0.2 + 0.6 * yy, 0.8 - 0.5 * (xx + yy) / 2], axis=-1)
    return np.clip(img, 0, 1).astype(np.float32)


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR (dB) of two tensors in [-1, 1]."""
    mse = float(((a.float() - b.float()) ** 2).mean()) / 4.0
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def pad32(n: int) -> int:
    return -(-int(n) // 32) * 32


class SmokeError(RuntimeError):
    """F0 failed: the job stops (token layout / numerics assumptions do not hold on the real weights)."""


class SmokeNonFinite(SmokeError):
    """NaN / Inf in the teacher output: the signature of the known buc/teacher/zimage.py bug."""


def f0_smoke(reader: TeacherReader, images: Dict[str, np.ndarray], *, psnr_min: float = 15.0,
             mismatch_max: float = 0.001) -> dict:
    """F0: on a gradient, a corpus image and a DIODE val image check (a) hook output shape (1, n_img + pad32(L_cap),
    D) with S - n_img in {32, 64}, (b) finite outputs, (c) x0-hat PSNR of the gradient, (d) every tapped block runs
    exactly once per forward, (e) early exit leaves the tapped tokens unchanged (mismatch share <= 0.1%), plus
    forward time and peak memory per resolution."""
    teacher = reader.teacher
    spec = teacher.spec
    raw = RawTap(reader.layers).attach(teacher.layers)
    reader._attach_stop()                   # after the raw hooks: the tokens are kept before the early exit
    rep: dict = {"images": {}}
    was = reader.early_exit
    try:
        for name, img in images.items():
            h, w = img.shape[:2]
            n_img = (h // 16) * (w // 16)
            x01 = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
            key = f"smoke_{name}"
            with torch.no_grad():
                z, _, _ = reader._prepare(x01, [key])
                raw.begin()
                reader.early_exit = False
                try:
                    x0 = reader.sampler.predict_x0(z, reader.sigma, [reader.cap])
                except (Z.TeacherNumericError, FloatingPointError) as e:
                    raise SmokeNonFinite(err_str(e, 300)) from None
                if not bool(torch.isfinite(x0).all()):
                    raise SmokeNonFinite("x0-hat has NaN/Inf")
                shapes = {}
                for l in reader.layers:
                    hs = raw.store[l]
                    if hs.ndim != 3 or hs.shape[0] != 1 or hs.shape[2] != spec.hidden_dim:
                        raise SmokeError(f"layer {l} hidden shape {tuple(hs.shape)}")
                    extra = int(hs.shape[1]) - n_img
                    if extra not in (32, 64) and extra != pad32(int(reader.cap.shape[0])):
                        raise SmokeError(f"layer {l}: S - n_img = {extra}, expected 32 or 64 (token layout assumption)")
                    if not bool(torch.isfinite(hs.float()).all()):
                        raise SmokeNonFinite(f"layer {l} hidden state not finite")
                    if raw.calls[l] != 1:
                        raise SmokeError(f"layer {l} ran {raw.calls[l]} times in one forward")
                    shapes[l] = list(hs.shape)
                full = {l: raw.store[l][:, :n_img].float().cpu().clone() for l in reader.layers}
                row = {"hw": [h, w], "S": shapes[reader.layers[0]][1], "extra": shapes[reader.layers[0]][1] - n_img}
                if name == "gradient":
                    dec = teacher.decode_latent(x0)
                    row["psnr_x0"] = round(psnr01(dec.cpu(), (x01 * 2 - 1)), 2)
                    if row["psnr_x0"] < psnr_min:
                        raise SmokeError(f"x0-hat PSNR {row['psnr_x0']} dB < {psnr_min} on the gradient image")
                # early exit pass (timed, second run is warm)
                reader.early_exit = True
                for rep_i in range(2):
                    raw.begin()
                    if cuda_ready():
                        torch.cuda.reset_peak_memory_stats()
                    sync_cuda()
                    t0 = time.perf_counter()
                    reader._forward(z)
                    sync_cuda()
                    dt = time.perf_counter() - t0
                if any(raw.calls[l] != 1 for l in reader.layers):
                    raise SmokeError("early-exit pass: a tapped block did not run exactly once")
                a = full[reader.layers[-1]]
                b = raw.store[reader.layers[-1]][:, :n_img].float().cpu()
                tol = 1e-3 * max(float(a.std()), 1e-6)
                mism = float(((a - b).abs() > tol).float().mean())
                row.update({"early_exit_mismatch": round(mism, 6), "fwd_s": round(dt, 3),
                            "peak_gb": vram_gb()})
                if mism > mismatch_max:
                    raise SmokeError(f"early exit changes the layer-{reader.layers[-1]} tokens ({mism:.4f})")
            rep["images"][name] = row
    finally:
        reader.early_exit = was
        raw.detach()
        reader.close_hooks()
    return rep


# ---------------------------------------------------------------------------
# PCA-128 basis: what goes in (never DIODE / yak / eval images), fitting, diagnostics
# ---------------------------------------------------------------------------

IP_META_KEYS = ("ip", "ip_flag", "ipflag", "ip-flag")


@dataclasses.dataclass(frozen=True)
class FitItem:
    kind: str            # "corpus" | "render"
    ident: str           # "night1:<i>" | render key
    path: str
    prov: object


def corpus_img_path(root: str, i: int) -> str:
    return os.path.join(root, "img", f"{int(i):08d}.webp")


def corpus_ids(root: str) -> List[int]:
    try:
        names = os.listdir(os.path.join(root, "img"))
    except OSError:
        return []
    return sorted(int(n[:-5]) for n in names if n.endswith(".webp") and n[:-5].isdigit())


def corpus_ip_flagged(root: str, i: int) -> bool:
    m = read_json(os.path.join(root, "meta", f"{int(i):08d}.json"), {}) or {}
    return any(bool(m.get(k)) for k in IP_META_KEYS)


def choose_evenly(ids: Sequence, n: int) -> list:
    s = sorted(ids)
    if n <= 0 or not s:
        return []
    if n >= len(s):
        return s
    if n == 1:
        return [s[len(s) // 2]]
    return sorted({s[int(round(j * (len(s) - 1) / (n - 1)))] for j in range(n)})


def check_fit_paths(items: Sequence[FitItem], paths: Paths) -> None:
    """The PCA fit may read only night1 corpus images and synthetic TRAIN renders.  Anything else (DIODE, yak,
    eval, synthetic dev) raises."""
    img_root = os.path.realpath(os.path.join(paths.root, "img"))
    synth_root = os.path.realpath(paths.synth_train)
    for it in items:
        rp = os.path.realpath(it.path)
        if it.kind == "corpus":
            ok = os.path.dirname(rp) == img_root and rp.endswith(".webp")
        elif it.kind == "render":
            sk = parse_synth_key(it.ident)
            ok = (os.path.dirname(rp) == synth_root and sk is not None
                  and SYNTH_TRAIN_SEEDS[0] <= sk[0] < SYNTH_TRAIN_SEEDS[1])
        else:
            ok = False
        if not ok:
            raise ValueError(f"not allowed in the PCA fit set: {it.kind} {it.ident} {rp}")


def fit_set(paths: Paths, corpus: Sequence[int], render_keys: Sequence[str], *, n_corpus: int = BASIS_N_CORPUS,
            n_render: int = BASIS_N_RENDER, ip_flagged: Optional[Callable[[int], bool]] = None) -> List[FitItem]:
    """600 night1 corpus images (evenly spread, IP-flagged ones dropped) + the first 400 synthetic TRAIN renders.
    Takes corpus indices and render keys only: DIODE, the yak image and every eval image have no way in."""
    flagged = ip_flagged or (lambda i: corpus_ip_flagged(paths.root, i))
    cands = [int(i) for i in corpus if not flagged(int(i))]
    items = [FitItem("corpus", f"night1:{i}", corpus_img_path(paths.root, i), prov_night1(i))
             for i in choose_evenly(cands, n_corpus)]
    for k in sorted(render_keys)[:n_render]:
        sk = parse_synth_key(k)
        if sk is None or not (SYNTH_TRAIN_SEEDS[0] <= sk[0] < SYNTH_TRAIN_SEEDS[1]):
            raise ValueError(f"render key outside the synthetic train range: {k}")
        items.append(FitItem("render", k, synth_file(paths.synth_train, k, ".png"), prov_roomgen(sk[0])))
    check_fit_paths(items, paths)
    return items


def effective_hw(h: int, w: int) -> Tuple[int, int]:
    """Shape a corpus image is read at: 1024x1024 -> centre 768x1024; everything else floored to multiples of 16
    (1344x576 stays)."""
    if (h, w) == (1024, 1024):
        return 768, 1024
    return (h // 16) * 16, (w // 16) * 16


def prepare_image(arr_u8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 [0,1] cropped to ``effective_hw`` (centre crop)."""
    h, w = arr_u8.shape[:2]
    eh, ew = effective_hw(h, w)
    y0, x0 = (h - eh) // 2, (w - ew) // 2
    return np.ascontiguousarray(arr_u8[y0:y0 + eh, x0:x0 + ew, :3]).astype(np.float32) / 255.0


def load_image_float(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return prepare_image(np.asarray(im.convert("RGB"), dtype=np.uint8))


def image_hw(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return effective_hw(h, w)


def batch_size_for(h: int, w: int) -> int:
    return 2 if h * w >= 384 * 512 * 3 else 4


def group_by_shape(entries: Sequence[tuple], hw_fn: Callable[[tuple], Tuple[int, int]]) -> List[Tuple[Tuple[int, int], List[tuple]]]:
    """[(hw, [entries...])] keeping first-seen order of shapes and the order inside each shape."""
    groups: Dict[Tuple[int, int], List[tuple]] = {}
    for e in entries:
        groups.setdefault(hw_fn(e), []).append(e)
    return list(groups.items())


def plan_chunks(items: Sequence[FitItem], chunk_images: int = 20,
                hw_fn: Callable[[FitItem], Tuple[int, int]] = lambda it: image_hw(it.path)) -> List[Tuple[str, List[FitItem]]]:
    """Deterministic chunk plan ``[(chunk_id, items)]``: same-shape images, <= chunk_images per chunk."""
    out = []
    for hw, its in group_by_shape(items, hw_fn):
        for s in range(0, len(its), chunk_images):
            part = its[s:s + chunk_images]
            cid = hashlib.sha256(("|".join(i.ident for i in part) + f"@{hw}").encode()).hexdigest()[:16]
            out.append((cid, part))
    return out


def chunk_path(paths: Paths, cid: str) -> str:
    return os.path.join(paths.basis_samples, f"chunk_{cid}.pt")


def collect_basis_samples(reader: TeacherReader, items: Sequence[FitItem], paths: Paths, *, chunk_images: int = 20,
                          n_tokens: int = TOKENS_PER_IMG, load_fn: Callable[[str], np.ndarray] = load_image_float,
                          hw_fn: Callable[[FitItem], Tuple[int, int]] = lambda it: image_hw(it.path),
                          stop: Callable[[], bool] = lambda: False) -> dict:
    """Read raw hidden tokens (``n_tokens`` per image and layer) in resumable chunk files (bf16, lossless: the
    transformer runs in bf16)."""
    os.makedirs(paths.basis_samples, exist_ok=True)
    plan = plan_chunks(items, chunk_images, hw_fn)
    done = built = 0
    for cid, part in plan:
        if stop():
            return {"chunks": len(plan), "done": done, "built": built, "stopped": True}
        cp = chunk_path(paths, cid)
        if os.path.isfile(cp):
            done += 1
            continue
        toks: Dict[int, List[torch.Tensor]] = {l: [] for l in reader.layers}
        imgs = [(it, load_fn(it.path)) for it in part]
        bs = batch_size_for(*imgs[0][1].shape[:2])
        for s in range(0, len(imgs), bs):
            sub = imgs[s:s + bs]
            x01 = torch.from_numpy(np.stack([a for _, a in sub])).permute(0, 3, 1, 2).contiguous()
            keys = [f"fit_{it.ident.replace(':', '_')}" for it, _ in sub]
            res = reader.read_raw(x01, keys, [it.ident for it, _ in sub], n_tokens)
            for r in res:
                for l in reader.layers:
                    toks[l].append(r[l])
        save_atomic({"ids": [it.ident for it, _ in part],
                     "tokens": {l: torch.cat(v).to(torch.bfloat16) for l, v in toks.items()}}, cp)
        built += 1
        done += 1
        free_cuda_cache()
    return {"chunks": len(plan), "done": done, "built": built, "stopped": False}


def load_basis_samples(paths: Paths, items: Sequence[FitItem], chunk_images: int = 20,
                       hw_fn: Callable[[FitItem], Tuple[int, int]] = lambda it: image_hw(it.path)):
    """-> ({layer: float32 [N, D]}, [prov of every image that contributed])."""
    acc: Dict[int, List[torch.Tensor]] = {}
    provs = []
    by_id = {it.ident: it for it in items}
    for cid, part in plan_chunks(items, chunk_images, hw_fn):
        ck = load_ckpt(chunk_path(paths, cid), need=("ids", "tokens"))
        if ck is None:
            raise FileNotFoundError(f"missing basis chunk {cid}")
        for l, t in ck["tokens"].items():
            acc.setdefault(int(l), []).append(t.float())
        provs.extend(by_id[i].prov for i in ck["ids"])
    return {l: torch.cat(v) for l, v in acc.items()}, provs


def trim_by_norm(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Drop the ``frac`` share of tokens with the largest L2 norm (giant 'massive activation' tokens)."""
    n = x.shape[0]
    drop = int(round(n * float(frac)))
    if drop <= 0:
        return x
    order = torch.argsort(x.norm(dim=1), descending=True)
    keep = torch.ones(n, dtype=torch.bool)
    keep[order[:drop]] = False
    return x[keep]


def int8_diagnostics(basis: T.PCABasis, layer: int, held: torch.Tensor, norm_cut: Optional[float] = None) -> dict:
    """int8 projection vs float projection on held-out tokens: relative error (tokens inside the fit norm range),
    share of tokens with a saturated component, worst per-component saturation share, explained variance."""
    i = basis._index(layer)
    mean, comps, scale = basis.mean[i], basis.comps[i], basis.scale[i]
    xc = held.float() - mean
    zf = xc @ comps
    q = torch.round(zf / scale).clamp(-T.INT8_MAX, T.INT8_MAX)
    sat = (q.abs() == T.INT8_MAX)
    inside = torch.ones(held.shape[0], dtype=torch.bool) if norm_cut is None else (held.norm(dim=1) <= norm_cut)
    err = (q * scale - zf)[inside]
    rel = float(err.norm() / zf[inside].norm().clamp_min(1e-12)) if bool(inside.any()) else float("nan")
    ev = float(zf.var(dim=0).sum() / xc.var(dim=0).sum().clamp_min(1e-12))
    return {"rel_err": rel, "sat_tok": float(sat.any(dim=1).float().mean()), "sat_comp_max": float(sat.float().mean(0).max()),
            "ev": ev}


def fit_basis_from_samples(samples: Dict[int, torch.Tensor], provs: Sequence, *, k: int = PCA_K,
                           meta: Optional[dict] = None, trims: Sequence[float] = (0.001, 0.01, 0.03),
                           holdout_frac: float = 0.1, rel_tol: float = 0.10, seed: int = 1234):
    """Fit the PCA-k basis (once, then frozen).  Per layer the top ``trims[0]`` share of tokens by norm is dropped;
    10% of the tokens are held out for the int8 diagnostics.  When the relative int8 error exceeds ``rel_tol`` the fit
    is repeated with the next, larger trim (at most ``len(trims) - 1`` refits); the best attempt is kept.
    ``provs`` are the provenance records of every contributing image: one that is not train_ok raises FirewallError
    (``T.fit_pca_basis`` asserts the derived record)."""
    prov = P.derive("pca_fit", list(provs))
    layers = sorted(samples)
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    split = {}
    for l in layers:
        x = samples[l]
        perm = torch.randperm(x.shape[0], generator=g)
        nh = max(1, int(x.shape[0] * holdout_frac))
        split[l] = (x[perm[nh:]], x[perm[:nh]])
    best = None
    tries = []
    for ti, frac in enumerate(trims):
        fit = {l: trim_by_norm(split[l][0], frac) for l in layers}
        basis = T.fit_pca_basis(fit, prov=prov, k=k, meta={**(meta or {}), "trim_frac": float(frac), "refit": ti})
        diags = {}
        for l in layers:
            cut = float(fit[l].norm(dim=1).max())
            diags[l] = int8_diagnostics(basis, l, split[l][1], norm_cut=cut)
        worst = max(d["rel_err"] for d in diags.values())
        tries.append({"trim": frac, "rel_err": round(worst, 4)})
        if best is None or worst < best[0]:
            best = (worst, basis, diags, frac)
        if worst <= rel_tol:
            break
    worst, basis, diags, frac = best
    diag = {"trim_frac": frac, "rel_err_max": round(worst, 4), "tries": tries,
            "per_layer": {int(l): {k2: round(v, 4) for k2, v in d.items()} for l, d in diags.items()}}
    return basis, diag


def invalidate_after_refit(paths: Paths) -> str:
    """A re-fitted basis invalidates every shard, eval tap, pending tap and checkpoint: move them to ``stale_<ts>``."""
    dst = os.path.join(paths.out, f"stale_{int(time.time())}")
    os.makedirs(dst, exist_ok=True)
    for p in (paths.shards, os.path.join(paths.eval, "taps"), os.path.dirname(paths.pending), paths.ckpt_last,
              paths.ckpt_best, paths.tap_std, paths.progress("f2b"), paths.progress("f2c"), paths.progress("f2a"),
              paths.stage_file("f2a"), paths.stage_file("f2b"), paths.stage_file("f2c"), paths.yak_taps):
        if os.path.exists(p):
            with contextlib.suppress(OSError):
                shutil.move(p, os.path.join(dst, os.path.basename(p)))
    for d in (paths.shards_train, paths.shards_dev, paths.val_taps, paths.pending):
        os.makedirs(d, exist_ok=True)
    return dst


def run_f1(paths: Paths, reader: TeacherReader, args, *, stop: Callable[[], bool] = lambda: False,
           corpus: Optional[Sequence[int]] = None, load_fn: Callable[[str], np.ndarray] = load_image_float,
           hw_fn: Callable[[FitItem], Tuple[int, int]] = lambda it: image_hw(it.path),
           wait_fn: Optional[Callable[[Callable[[], bool], float], bool]] = None) -> Optional[T.PCABasis]:
    """F1: fit and freeze the basis.  ``basis.pt`` present -> skipped unless ``--refit-basis``."""
    if os.path.isfile(paths.basis) and not args.refit_basis:
        b = T.PCABasis.load(paths.basis)
        log({"basis": "frozen", "fp": b.fingerprint.hex()[:16], "k": b.k})
        return b
    ids = list(corpus) if corpus is not None else corpus_ids(paths.root)
    need = min(BASIS_N_RENDER, 2 * (SYNTH_TRAIN_SEEDS[1] - SYNTH_TRAIN_SEEDS[0]))
    if wait_fn is not None:
        wait_fn(lambda: len(list_synth_keys(paths.synth_train)) >= need or stage_done(paths, "render"), 3600.0)
    items = fit_set(paths, ids, list_synth_keys(paths.synth_train), n_corpus=args.basis_corpus, n_render=args.basis_render)
    n_c = sum(1 for i in items if i.kind == "corpus")
    log({"stage": "f1", "corpus": n_c, "render": len(items) - n_c})
    if not items or n_c == 0 and len(items) < 20:
        raise SmokeError("F1: not enough images for the PCA fit")
    reader.use_raw()
    try:
        st = collect_basis_samples(reader, items, paths, n_tokens=args.basis_tokens, load_fn=load_fn, hw_fn=hw_fn, stop=stop)
    finally:
        reader.close_hooks()
    if st.get("stopped"):
        return None
    samples, provs = load_basis_samples(paths, items, hw_fn=hw_fn)
    meta = {"sigma": reader.sigma, "caption": SURVEY_CAPTION, "backend": reader.backend, "det": False,
            "teacher": getattr(getattr(reader.teacher, "snapshot", None), "name", "?"),
            "tokens_per_img": args.basis_tokens, "n_corpus": n_c, "n_render": len(items) - n_c,
            "res": ["768x1024", "512x384", "1344x576"], "blake3": "shim" if BLAKE3_SHIM else "real"}
    basis, diag = fit_basis_from_samples(samples, provs, k=args.pca_k, meta=meta)
    stale = invalidate_after_refit(paths) if (args.refit_basis and os.path.isfile(paths.basis)) else None
    basis.save(paths.basis)
    write_json_atomic(os.path.join(paths.out, "basis_diag.json"), {**diag, "fp": basis.fingerprint.hex()})
    log({"basis": "fitted", "fp": basis.fingerprint.hex()[:16], "k": basis.k, "rel_err": diag["rel_err_max"],
         "trim": diag["trim_frac"], "ev": [diag["per_layer"][l]["ev"] for l in sorted(diag["per_layer"])],
         "stale": bool(stale)})
    return basis


# ---------------------------------------------------------------------------
# compact samples: fields, shards (resumable writer), pending taps, promotion
# ---------------------------------------------------------------------------

def synth_lab4(gbuf: np.ndarray, sem: np.ndarray, lo: float = DEPTH_LO, hi: float = DEPTH_HI) -> Dict[str, np.ndarray]:
    """H/4 labels of a synthetic view with the MDL-03 rule (``downsample_labels``, stride 4): a block counts only if
    all 16 pixels are valid; log depth is the block mean (clamped to [lo, hi] m); normals are averaged and
    re-normalised."""
    g = torch.from_numpy(np.ascontiguousarray(gbuf, dtype=np.float32))[None]
    s = torch.from_numpy(np.ascontiguousarray(sem).astype(np.int64))[None]
    lab = MT.downsample_labels(g, s, STRIDE)
    mask = lab["mask"][0, 0].numpy()
    ld = np.clip(lab["log_depth"][0, 0].numpy(), math.log(lo), math.log(hi)) * mask
    return {"log_depth": ld.astype(np.float32), "mask": (mask > 0.5).astype(np.uint8),
            "normal": lab["normal"][0].numpy().astype(np.float16)}


def make_meta(*, key: str, domain: str, env: str, scene: str, basis_fp_hex: str, band_valid, keep: bool,
              lowband_err: float, realify: str, backend: str, grid: Sequence[int], role: str,
              tap_sigma: float = TAP_SIGMA, extra: Optional[dict] = None) -> dict:
    m = {"v": SAMPLE_TAG, "key": key, "domain": domain, "env": env, "scene": scene, "basis_fp": basis_fp_hex,
         "tap_layers": list(TAP_LAYERS), "tap_sigma": float(tap_sigma), "band_valid": [bool(v) for v in band_valid],
         "keep": bool(keep), "lowband_err": float(lowband_err), "realify": realify, "backend": backend, "det": False,
         "blake3": "shim" if BLAKE3_SHIM else "real", "grid": [int(grid[0]), int(grid[1])], "role": role}
    if extra:
        m.update(extra)
    return m


def sample_fields(taps: Dict[int, np.ndarray], lab: Dict[str, np.ndarray], meta: dict) -> Dict[str, bytes]:
    f = {f"tap{l}.npy": npy_bytes(np.ascontiguousarray(np.asarray(taps[l], dtype=np.int8))) for l in sorted(taps)}
    f["lab4.npz"] = npz_bytes({k: v for k, v in lab.items() if v is not None})
    f["meta.json"] = json.dumps(meta, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return f


class ShardSink:
    """``ShardWriter`` + durable progress: a key is appended to the progress file only after the shard that holds it
    was closed (a kill loses at most the open shard, whose samples are then redone)."""

    def __init__(self, out_dir: str, purpose: str, progress_path: str, max_samples: int = 500,
                 on_closed: Optional[Callable[[List[str]], None]] = None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.writer = SH.ShardWriter(out_dir, max_samples=max_samples, purpose=purpose)
        self.progress_path = progress_path
        self.on_closed = on_closed
        self._records: Dict[str, dict] = {}
        self._seen = 0
        self.written = 0

    def write(self, key: str, fields: Dict[str, bytes], prov, record: Optional[dict] = None) -> None:
        self.writer.write(key, fields, prov)
        self._records[key] = {"key": key, "keep": True, **(record or {})}
        self.written += 1
        self._flush()

    def _flush(self) -> None:
        infos = self.writer.infos
        while self._seen < len(infos):
            info = infos[self._seen]
            self._seen += 1
            keys = list(info.keys)
            for k in keys:
                append_jsonl(self.progress_path, {**self._records.pop(k, {"key": k, "keep": True}), "shard": info.address[:16]})
            if self.on_closed:
                self.on_closed(keys)

    def close(self) -> None:
        self.writer.close()
        self._flush()

    def abort(self) -> None:
        """Drop the open shard (its keys stay un-recorded and are redone)."""
        with contextlib.suppress(Exception):
            self.writer.__exit__(RuntimeError, RuntimeError("abort"), None)


def pending_path(paths: Paths, key: str) -> str:
    return os.path.join(paths.pending, f"{key}.zip")


def write_pending(paths: Paths, key: str, fields: Dict[str, bytes], prov, role: str) -> None:
    """Park a sample the firewall refuses (``taps_pending/diode``; no loader reads this directory)."""
    members = dict(fields)
    members["prov.json"] = json.dumps(P.as_record(prov).to_dict(), sort_keys=True).encode("utf-8")
    members["role.txt"] = role.encode("utf-8")
    write_atomic(pending_path(paths, key), zip_bytes(members))


def list_pending(paths: Paths) -> List[str]:
    try:
        return sorted(n[:-4] for n in os.listdir(paths.pending) if n.endswith(".zip"))
    except OSError:
        return []


def read_pending(paths: Paths, key: str) -> Tuple[Dict[str, bytes], "P.ProvRecord", str]:
    with zipfile.ZipFile(pending_path(paths, key)) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    prov = P.ProvRecord.from_dict(json.loads(members.pop("prov.json")))
    role = members.pop("role.txt").decode("utf-8")
    return members, prov, role


def promote_pending(paths: Paths, *, max_samples: int = 500) -> dict:
    """If the FND-04 check passes now, move parked DIODE taps into the train / dev shards through ``ShardWriter``
    (which re-checks the firewall).  Called right before training starts and on every restart."""
    fw = diode_firewall()
    keys = list_pending(paths)
    if not fw["ok"]:
        return {"promoted": 0, "pending": len(keys), "blocked": True, "why": fw["why"]}
    sinks: Dict[str, ShardSink] = {}

    def on_closed(ks: List[str]) -> None:
        for k in ks:
            with contextlib.suppress(OSError):
                os.remove(pending_path(paths, k))

    n = 0
    try:
        for key in keys:
            fields, prov, role = read_pending(paths, key)
            if role not in sinks:
                d = paths.shards_dev if role == "dev" else paths.shards_train
                sinks[role] = ShardSink(os.path.join(d, "diode"), "dev" if role == "dev" else "train",
                                        paths.progress("promote"), max_samples, on_closed)
            sinks[role].write(key, fields, prov, {"promoted": True})
            n += 1
    finally:
        for s in sinks.values():
            s.close()
    return {"promoted": n, "pending": len(list_pending(paths)), "blocked": False, "why": []}


REALIFY_MODES = ({"name": "s0.5n2", "sigma": 0.5, "steps": 2}, {"name": "s0.3n1", "sigma": 0.3, "steps": 1})


def realify_mode(key: str) -> dict:
    """Even key hash -> SDEdit from sigma 0.5 with 2 NFE; odd -> sigma 0.3 with 1 NFE.  Both tap at sigma 0.3."""
    return REALIFY_MODES[key_parity(key)]


@contextlib.contextmanager
def allow_one_step_realify():
    """Local workaround: ``buc.data.realify.realify`` only accepts steps in (2, 3) while the spec reads half of the
    renders with a single NFE from sigma 0.3 (tap sigma 0.3 either way).  Patch the module constant for the call."""
    old = R.ALLOWED_STEPS
    R.ALLOWED_STEPS = tuple(sorted(set(old) | {1}))
    try:
        yield
    finally:
        R.ALLOWED_STEPS = old


def dev_key(ident: str) -> str:
    return f"dv_{ident}"


def stats_from_progress(path: str) -> dict:
    out: Dict[str, dict] = {}
    for r in read_jsonl(path):
        m = r.get("mode")
        if not m:
            continue
        s = out.setdefault(m, {"n": 0, "kept": 0, "lb_sum": 0.0})
        s["n"] += 1
        s["kept"] += 1 if r.get("keep") else 0
        s["lb_sum"] += float(r.get("lowband_err", 0.0))
    return {m: {"n": s["n"], "keep": round(s["kept"] / max(s["n"], 1), 3), "lowband_err": round(s["lb_sum"] / max(s["n"], 1), 4)}
            for m, s in out.items()}


def run_f2b(paths: Paths, reader: TeacherReader, basis: T.PCABasis, args, *, cap: Optional[int] = None,
            stop: Callable[[], bool] = lambda: False, wh: Tuple[int, int] = SYNTH_WH,
            batch: int = 4, max_samples: int = 500) -> dict:
    """F2b: SDEdit-realify the synthetic renders and tap them in the same pass (sigma 0.5 / 2 NFE or sigma 0.3 /
    1 NFE, tap sigma 0.3).  Train samples whose low-band mismatch exceeds tau=0.05 are dropped (keep=False); the
    200 dev samples use tau=1e9 so every one is kept.  Output: compact shards + progress file (resumable)."""
    pg = paths.progress("f2b")
    done = progress_keys(pg)
    train_keys = list_synth_keys(paths.synth_train)
    if cap is not None:
        train_keys = train_keys[:cap]
    dev_keys = list_synth_keys(paths.synth_dev)
    h, w = wh[1], wh[0]
    curve = R.provisional_curve(reader.sampler.table, R.max_octaves(h, w))
    sinks = {"train": ShardSink(os.path.join(paths.shards_train, "synth"), "train", pg, max_samples),
             "dev": ShardSink(os.path.join(paths.shards_dev, "synth"), "dev", pg, max_samples)}
    reader.use_bus(basis)
    stats = {"n": 0, "kept": 0, "dropped": 0, "stopped": False}
    fp = basis.fingerprint.hex()
    try:
        for role, keys, tau in (("dev", dev_keys, 1e9), ("train", train_keys, R.DEFAULT_TAU)):
            src_dir = paths.synth_dev if role == "dev" else paths.synth_train
            ikey = dev_key if role == "dev" else (lambda k: k)
            todo = [k for k in keys if ikey(k) not in done]
            for parity in (0, 1):
                mode = REALIFY_MODES[parity]
                ks = [k for k in todo if key_parity(ikey(k)) == parity]
                for s in range(0, len(ks), batch):
                    if stop():
                        stats["stopped"] = True
                        return _f2b_finish(sinks, stats, pg)
                    sub = ks[s:s + batch]
                    items = []
                    for k in sub:
                        smp = load_synth_sample(src_dir, k)
                        items.append(R.RealifyItem(key=ikey(k), image=smp["image"], prov=prov_roomgen(parse_synth_key(k)[0]),
                                                   caption=SURVEY_CAPTION, labels={"gbuf": smp["gbuf"], "sem": smp["sem"]}))
                    with allow_one_step_realify(), torch.no_grad():
                        results = R.realify(reader.sampler, reader.bus, items, curve=curve, sigma=mode["sigma"],
                                            seed=REALIFY_SEED, steps=mode["steps"], tau=tau, token_offset=0,
                                            tap_step=-1, ring=None)
                    for res, k in zip(results, sub):
                        stats["n"] += 1
                        rec = {"mode": mode["name"], "lowband_err": round(res.lowband_err, 5)}
                        if not res.keep:
                            stats["dropped"] += 1
                            append_jsonl(pg, {"key": res.key, "keep": False, "reason": res.reject_reason, **rec})
                            continue
                        lab = synth_lab4(res.labels["gbuf"], res.labels["sem"])
                        taps = {l: res.taps.layers[l][0].cpu().numpy() for l in sorted(res.taps.layers)}
                        meta = make_meta(key=res.key, domain="synth", env="room", scene=k[:7], basis_fp_hex=fp,
                                         band_valid=res.band_valid, keep=True, lowband_err=res.lowband_err,
                                         realify=mode["name"], backend=reader.backend, grid=res.taps.grid, role=role,
                                         tap_sigma=res.tap_sigma, extra={"sigma": res.sigma, "steps": res.steps})
                        sinks[role].write(res.key, sample_fields(taps, lab, meta), res.prov, rec)
                        stats["kept"] += 1
                    free_cuda_cache()
    except BaseException:
        for sk in sinks.values():
            sk.abort()
        raise
    finally:
        reader.close_hooks()
    return _f2b_finish(sinks, stats, pg)


def _f2b_finish(sinks: Dict[str, ShardSink], stats: dict, pg: str) -> dict:
    for sk in sinks.values():
        sk.close()
    stats["modes"] = stats_from_progress(pg)
    log({"stage": "f2b", **{k: v for k, v in stats.items() if k != "modes"}, "modes": stats["modes"]})
    return stats


# ---------------------------------------------------------------------------
# F2a eval taps (DIODE val + the yak), F2c DIODE train / dev taps
# ---------------------------------------------------------------------------

def wait_until(cond: Callable[[], bool], timeout_s: float, *, now: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep, poll_s: float = 10.0,
               stop: Callable[[], bool] = lambda: False) -> bool:
    t0 = now()
    while not cond():
        if stop() or now() - t0 >= timeout_s:
            return False
        sleep(poll_s)
    return True


def save_tap_npz(path: str, taps: Dict[int, "torch.Tensor | np.ndarray"], meta: dict) -> None:
    arrays = {f"tap{l}": np.ascontiguousarray(np.asarray(taps[l].cpu() if isinstance(taps[l], torch.Tensor) else taps[l],
                                                         dtype=np.int8)) for l in sorted(taps)}
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True, default=_json_default).encode("utf-8"), dtype=np.uint8)
    write_atomic(path, npz_bytes(arrays))


def load_tap_npz(path: str) -> Tuple[Dict[int, np.ndarray], dict]:
    z = load_npz(path)
    meta = json.loads(bytes(z.pop("meta")).decode("utf-8"))
    return {int(k[3:]): v for k, v in z.items() if k.startswith("tap")}, meta


def val_tap_path(paths: Paths, stem: str) -> str:
    return os.path.join(paths.val_taps, stem + ".npz")


def run_f2a(paths: Paths, reader: TeacherReader, basis: T.PCABasis, *, stop: Callable[[], bool] = lambda: False,
            batch: int = 2) -> dict:
    """F2a: int8 taps of the 771 DIODE val images (prov ASSET:MIT + HOLDOUT:diode_val; never a shard)."""
    samples = list_diode_samples(paths, "val")
    todo = [(m, b) for m, b in samples if not os.path.isfile(val_tap_path(paths, m.stem))]
    reader.use_bus(basis)
    n = 0
    try:
        for s in range(0, len(todo), batch):
            if stop():
                return {"n": n, "total": len(samples), "partial": True}
            sub = todo[s:s + batch]
            x01 = torch.from_numpy(np.stack([load_image_float(b + ".png") for _, b in sub])).permute(0, 3, 1, 2).contiguous()
            prov = batch_prov([prov_diode_val(m.scene, m.scan) for m, _ in sub])
            res = reader.read_int8(x01, [f"val_{m.stem}" for m, _ in sub], prov)
            for (m, _b), taps in zip(sub, res):
                meta = {"stem": m.stem, "env": m.env, "scene": m.scene, "basis_fp": basis.fingerprint.hex(),
                        "backend": reader.backend, "hw": list(x01.shape[2:]), "sigma": reader.sigma}
                save_tap_npz(val_tap_path(paths, m.stem), taps, meta)
                n += 1
            free_cuda_cache()
    finally:
        reader.close_hooks()
    return {"n": n, "total": len(samples), "partial": False}


def default_download(repo_id: str, filename: str, token: str, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=REPO_TYPE, token=token, cache_dir=cache_dir)


def fetch_yak(paths: Paths, token: Optional[str], download_fn: Optional[Callable] = None) -> Optional[str]:
    """The founder's yak image comes from the private vault straight into ``<out>/private`` (never anywhere else)."""
    if not token:
        return None
    cache = safe_out_path(os.path.join(paths.private, "hf_cache"), paths)
    os.makedirs(cache, exist_ok=True)
    return (download_fn or default_download)(REPO_ID, YAK_REMOTE, token, cache)


def yak_size(h: int, w: int) -> Tuple[int, int]:
    """About 589,824 px (RD-07 global size), aspect kept, both sides multiples of 16 -> (H, W)."""
    return global_size(h, w)


def load_yak_float(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        gh, gw = yak_size(h, w)
        im = im.resize((gw, gh), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0


def run_yak_taps(paths: Paths, reader: TeacherReader, basis: T.PCABasis, token: Optional[str],
                 download_fn: Optional[Callable] = None) -> dict:
    """Taps of the yak image: ``ring=None``, a temporary file under private/ - not a shard, never in the fit set."""
    out = safe_out_path(paths.yak_taps, paths)
    if os.path.isfile(out):
        return {"status": "have"}
    src = fetch_yak(paths, token, download_fn)
    if src is None:
        return {"status": "no_token"}
    img = load_yak_float(src)
    write_atomic(safe_out_path(os.path.join(paths.private, "yak_input.png"), paths), _png_bytes((img * 255.0).round().astype(np.uint8)))
    reader.use_bus(basis)
    try:
        x01 = torch.from_numpy(img).permute(2, 0, 1)[None].contiguous()
        taps = reader.read_int8(x01, ["yak"], prov_yak())[0]
    finally:
        reader.close_hooks()
    save_tap_npz(out, taps, {"basis_fp": basis.fingerprint.hex(), "hw": list(img.shape[:2]), "sigma": reader.sigma,
                             "backend": reader.backend})
    return {"status": "ok", "hw": list(img.shape[:2])}


def diode_key(m: Member) -> str:
    return (dev_key(m.stem) if role_of(m) == "dev" else "d" + m.stem)


def run_f2c(paths: Paths, reader: TeacherReader, basis: T.PCABasis, args, *, cap: Optional[int] = None,
            stop: Callable[[], bool] = lambda: False, now: Callable[[], float] = time.time,
            sleep: Callable[[float], None] = time.sleep, batch: int = 2, idle_timeout_s: float = 900.0,
            poll_s: float = 20.0, max_samples: int = 500) -> dict:
    """F2c: taps of the DIODE train + dev crops (concurrent with the stream), labels from the half-resolution laser
    GT.  The firewall verdict decides where a sample goes: train / dev shards (verdict ok) or ``taps_pending/``."""
    pg = paths.progress("f2c")
    done = progress_keys(pg)
    sinks: Dict[str, ShardSink] = {}
    fp = basis.fingerprint.hex()
    stats = {"n": 0, "shards": 0, "pending": 0, "partial": False}
    reader.use_bus(basis)
    last_new = now()
    try:
        while True:
            if stop():
                stats["partial"] = True
                break
            stream_done = stage_done(paths, "diode_train")
            samples = list_diode_samples(paths, "train")
            if cap is not None and not stream_done:
                samples = []                       # a scene-balanced prefix needs the complete list
            elif cap is not None:
                samples = balanced_prefix(cap, samples)
            todo = [(m, b) for m, b in samples if diode_key(m) not in done]
            if not todo:
                if stream_done:
                    break
                if now() - last_new >= idle_timeout_s:
                    stats["partial"] = True
                    break
                sleep(poll_s)
                continue
            last_new = now()
            for s in range(0, len(todo), batch):
                if stop():
                    stats["partial"] = True
                    break
                sub = todo[s:s + batch]
                try:
                    x01 = torch.from_numpy(np.stack([load_image_float(b + ".png") for _, b in sub])).permute(0, 3, 1, 2).contiguous()
                    provs = [prov_diode_train(m.scene, m.scan) for m, _ in sub]
                    res = reader.read_int8(x01, [diode_key(m) for m, _ in sub], batch_prov(provs))
                except Exception as e:  # noqa: BLE001
                    if is_oom(e) or isinstance(e, (Z.TeacherNumericError, SmokeError)):
                        raise
                    warn("f2c_batch_failed", err=err_str(e))
                    for m, _b in sub:
                        done.add(diode_key(m))
                        append_jsonl(pg, {"key": diode_key(m), "keep": False, "reason": "read_failed"})
                    continue
                for (m, b), prov, taps in zip(sub, provs, res):
                    key, role = diode_key(m), role_of(m)
                    gt = load_npz(b + ".gt.npz")
                    ld, mk = lab4_from_half(gt["depth_m"], gt["vcnt"])
                    gh, gw = taps[reader.layers[0]].shape[1:]
                    nbands = R.max_octaves(x01.shape[2], x01.shape[3])
                    meta = make_meta(key=key, domain="diode", env=m.env, scene=m.scene, basis_fp_hex=fp,
                                     band_valid=(True,) * nbands, keep=True, lowband_err=0.0, realify="none",
                                     backend=reader.backend, grid=(gh, gw), role=role, extra={"scan": m.scan})
                    fields = sample_fields({l: t.numpy() for l, t in taps.items()}, {"log_depth": ld, "mask": mk}, meta)
                    fw = diode_firewall()
                    if fw["ok"]:
                        if role not in sinks:
                            d = paths.shards_dev if role == "dev" else paths.shards_train
                            sinks[role] = ShardSink(os.path.join(d, "diode"), "dev" if role == "dev" else "train", pg, max_samples)
                        sinks[role].write(key, fields, prov, {"role": role})
                        stats["shards"] += 1
                    else:
                        write_pending(paths, key, fields, prov, role)
                        append_jsonl(pg, {"key": key, "keep": True, "pending": True, "role": role})
                        stats["pending"] += 1
                    done.add(key)
                    stats["n"] += 1
                free_cuda_cache()
            if cap is not None and not stats["partial"]:
                continue
    except BaseException:
        for sk in sinks.values():
            sk.abort()
        reader.close_hooks()
        raise
    for sk in sinks.values():
        sk.close()
    reader.close_hooks()
    log({"stage": "f2c", **stats, "fw": "ok" if diode_firewall()["ok"] else "blocked_fnd04"})
    return stats


# ---------------------------------------------------------------------------
# teacher stage (sub-process --stage teacher): gate -> F0 -> F1 -> F2a -> F2b -> F2c -> F3 release
# ---------------------------------------------------------------------------

def rig_hash(backend: str) -> str:
    return stable_hash("f0", backend, SURVEY_CAPTION, list(TAP_LAYERS))


def teacher_needed(paths: Paths, args) -> bool:
    """False when there is nothing left for the GPU teacher to do (all F stages finished)."""
    if args.refit_basis or not os.path.isfile(paths.basis):
        return True
    try:
        fp = T.PCABasis.load(paths.basis).fingerprint.hex()
    except Exception:  # noqa: BLE001
        return True
    need = [("f2a", stable_hash("f2a", fp)), ("f2b", stable_hash("f2b", fp, REALIFY_SEED)),
            ("f2c", stable_hash("f2c", fp))]
    return not all(stage_done(paths, n, h) for n, h in need)


def build_rig(paths: Paths, args, deps: dict, backend: str) -> dict:
    factory = deps.get("teacher_factory")
    teacher = factory(backend) if factory else load_teacher_real(paths, backend, args.device or "cuda")
    cap = caption_embedding(teacher, paths, backend)
    sampler = TS.TeacherSampler(teacher, max_views_per_call=None)
    reader = TeacherReader(teacher, sampler, cap, backend=backend)
    return {"teacher": teacher, "sampler": sampler, "reader": reader, "cap": cap, "backend": backend}


def teardown_rig(rig: dict, mem_fn: Optional[Callable[[], Tuple[int, int]]] = None,
                 free_loaded: float = 0.0) -> dict:
    """F3: detach the hooks, drop every reference, empty the cache, verify the memory came back."""
    reader = rig.pop("reader", None)
    if reader is not None:
        reader.close_hooks()
    rig.clear()
    del reader
    free_cuda_cache()
    alloc = torch.cuda.memory_allocated() if cuda_ready() else 0
    reserved = torch.cuda.memory_reserved() if cuda_ready() else 0
    free_after = float(mem_fn()[0]) if mem_fn is not None else free_loaded
    rep = release_report(alloc, reserved, free_loaded, free_after)
    if rep["released"]:
        log({"teacher": "released", **rep})
    else:
        warn("teacher_release_check", **rep)
    return rep


def smoke_images(paths: Paths, deps: dict) -> Dict[str, np.ndarray]:
    if "smoke_images" in deps:
        return deps["smoke_images"]
    imgs = {"gradient": fixed_gradient_image(512, 512)}
    ids = corpus_ids(paths.root)
    if ids:
        with contextlib.suppress(Exception):
            imgs["corpus"] = load_image_float(corpus_img_path(paths.root, ids[len(ids) // 2]))
    vs = list_diode_samples(paths, "val")
    if vs:
        with contextlib.suppress(Exception):
            imgs["diode"] = load_image_float(vs[0][1] + ".png")
    return imgs


def establish_backend(paths: Paths, args, deps: dict) -> Tuple[dict, Optional[dict]]:
    """Load the teacher and run F0.  The ``buc`` backend first; NaN / TeacherNumericError in the smoke test (the
    known zimage.py bug) -> plain diffusers loader (M6 style).  The choice is recorded in backend.json and every
    later sample / basis / checkpoint meta, and a restart keeps it."""
    saved = read_backend(paths)
    order = [saved["backend"]] if saved else ["buc", "diffusers"]
    last: Optional[BaseException] = None
    for name in order:
        rig = build_rig(paths, args, deps, name)
        try:
            if stage_done(paths, "f0", rig_hash(name)):
                return rig, None
            rep = f0_smoke(rig["reader"], smoke_images(paths, deps))
            write_stage_done(paths, "f0", rig_hash(name), {"backend": name, **rep})
            write_backend(paths, name, "smoke_ok" if saved is None and name == "buc" else (saved or {}).get("why", "smoke_ok"))
            log({"stage": "f0", "backend": name, "images": rep["images"]})
            return rig, rep
        except (SmokeNonFinite, Z.TeacherNumericError, FloatingPointError) as e:
            last = e
            warn("teacher_nan", backend=name, err=err_str(e, 160))
            teardown_rig(rig)
            if name == "buc" and saved is None:
                write_backend(paths, "diffusers", "buc_nan:" + err_str(e, 100))
                saved = read_backend(paths)
                continue
            break
        except BaseException:
            teardown_rig(rig)
            raise
    raise SmokeError(f"no teacher backend passed the smoke test ({err_str(last) if last else '-'})")


def cap_satisfied(done_cap: Optional[int], want: Optional[int]) -> bool:
    """A finished F2b/F2c with cap ``done_cap`` (None = everything) covers a request for ``want`` samples."""
    return done_cap is None or (want is not None and done_cap >= want)


def _teacher_work(paths: Paths, args, reader: TeacherReader, plan: dict, deps: dict, now, sleep, stopper) -> int:
    """F1 -> F2a -> F2b -> F2c (each skipped when its stage marker is current).  Everything is local to this function
    so no teacher reference survives it (F3 then verifies the memory came back)."""
    rc = 0
    basis = run_f1(paths, reader, args, stop=stopper, corpus=deps.get("corpus"),
                   wait_fn=lambda c, t: wait_until(c, t, now=now, sleep=sleep, stop=stopper))
    if basis is None:
        return RC_INCOMPLETE
    try:
        fp = basis.fingerprint.hex()
        wait_until(lambda: stage_done(paths, "diode_val"), args.wait_diode_s, now=now, sleep=sleep, stop=stopper)
        h = stable_hash("f2a", fp)
        if not stage_done(paths, "f2a", h):
            st = run_f2a(paths, reader, basis, stop=stopper)
            yk = run_yak_taps(paths, reader, basis, os.environ.get("HF_TOKEN"), deps.get("yak_download"))
            log({"stage": "f2a", **st, "yak": yk["status"]})
            if not st["partial"] and stage_done(paths, "diode_val"):
                write_stage_done(paths, "f2a", h, {**st, "yak": yk["status"]})
            else:
                rc = RC_INCOMPLETE
        h = stable_hash("f2b", fp, REALIFY_SEED)
        rec = stage_record(paths, "f2b")
        ok = rec is not None and rec.get("config_hash") == h and cap_satisfied((rec.get("counts") or {}).get("cap"),
                                                                              plan["f2b_cap"])
        if not ok:
            wait_until(lambda: stage_done(paths, "render"), args.wait_render_s, now=now, sleep=sleep, stop=stopper)
            st = run_f2b(paths, reader, basis, args, cap=plan["f2b_cap"], stop=stopper)
            if not st["stopped"] and stage_done(paths, "render"):
                write_stage_done(paths, "f2b", h, {**{k: v for k, v in st.items() if k != "modes"}, "cap": plan["f2b_cap"],
                                                  "modes": st["modes"]})
            else:
                rc = RC_INCOMPLETE
        h = stable_hash("f2c", fp)
        rec = stage_record(paths, "f2c")
        ok = rec is not None and rec.get("config_hash") == h and cap_satisfied((rec.get("counts") or {}).get("cap"),
                                                                              plan["f2c_cap"])
        if not ok:
            st = run_f2c(paths, reader, basis, args, cap=plan["f2c_cap"], stop=stopper, now=now, sleep=sleep)
            if not st["partial"]:
                write_stage_done(paths, "f2c", h, {**st, "cap": plan["f2c_cap"]})
            else:
                rc = RC_INCOMPLETE
    finally:
        getattr(basis, "_dev_cache", {}).clear()           # GPU copies of the basis tensors
    return rc


def run_teacher_stage(paths: Paths, args, deps: Optional[dict] = None) -> int:
    """The GPU extraction sub-process.  Returns 0 (finished / skipped), RC_GATE_DEADLINE (gate never opened in
    time), RC_INCOMPLETE (stopped early) or 2 (fatal: F0 failed)."""
    deps = deps or {}
    now = deps.get("now", time.time)
    sleep = deps.get("sleep", time.sleep)
    mem_fn = deps.get("mem_get_info", mem_get_info_default)
    deadline = parse_deadline(args.deadline)
    stop_event = deps.get("stop_event") or threading.Event()

    def stopper() -> bool:
        return stop_event.is_set() or (deadline is not None and now() >= deadline - 1800.0)

    if not claim_pid(paths, "teacher", cmdline_fn=deps.get("cmdline_fn", _proc_cmdline)):
        log({"dup": "teacher"})
        return 0
    try:
        if not teacher_needed(paths, args):
            log({"stage": "teacher", "skip": "all_done"})
            return 0
        gate = wait_for_gate(mem_fn, need=args.gate_gb * 1e9, now=now, sleep=sleep, deadline=deadline,
                             m6_fn=lambda: m6_state(paths.root))
        remaining = None if deadline is None else deadline - now()
        plan = plan_for(remaining if remaining is not None else 1e9) if gate is not None else None
        if plan is None:
            log({"status": "deadline", "stage": "gpu_wait"}, "_DONE")
            return RC_GATE_DEADLINE
        write_json_atomic(paths.plan, {**plan, "gate_at": now(), "remaining_s": remaining, "free_gb": gate["free_gb"]})
        log({"stage": "gate", "free_gb": gate["free_gb"], "remaining_h": None if remaining is None else round(remaining / 3600, 2),
             **plan})
        cap_gib = apply_vram_cap(args.teacher_vram_gib)
        free_before = float(mem_fn()[0])
        try:
            rig, _smoke = establish_backend(paths, args, deps)
        except SmokeError as e:
            log({"status": "fatal", "stage": "f0", "err": err_str(e, 250)}, "_DONE")
            return 2
        free_loaded = float(mem_fn()[0])
        log({"stage": "teacher_loaded", "backend": rig["backend"], "cap_gib": cap_gib,
             "loaded_gb": gb(free_before - free_loaded), "vram_gb": vram_gb()})
        try:
            rc = _teacher_work(paths, args, rig["reader"], plan, deps, now, sleep, stopper)
        finally:
            rep = teardown_rig(rig, mem_fn, free_loaded)
        write_stages_json(paths)
        log({"stage": "f3", "rc": rc, **rep})
        return rc
    finally:
        release_pid(paths, "teacher")


# ---------------------------------------------------------------------------
# lazy shard dataset, deterministic batch plan, ordered prefetch
# ---------------------------------------------------------------------------

class ShardDataError(ValueError):
    """A sample in a shard does not match what training expects (meta, keep flag, layers)."""


def _pread_all(fd: int, size: int, off: int) -> bytes:
    out = []
    got = 0
    while got < size:
        b = os.pread(fd, size - got, off + got)
        if not b:
            raise ShardDataError("short read from shard")
        out.append(b)
        got += len(b)
    return b"".join(out)


class LazyShardDataset:
    """Compact samples (``tap<l>.npy`` int8 [k, gh, gw], ``lab4.npz``, ``meta.json``) read on demand with ``os.pread``.

    * ``allowed_root`` is the folder this dataset may read from (``shards/train`` for training): a shard directory
      outside it stops the job with an AssertionError (nothing from eval/, data/, private/ can reach the loader).
    * Every shard is verified once (``verify_shard`` with the purpose's firewall) and every meta is checked at init:
      format tag, ``basis_fp`` (TapBasisError), ``keep``, tap layers.
    * Duplicate keys (a kill between closing a shard and recording it) are indexed once.
    """

    def __init__(self, shard_dir: str, *, purpose: str, basis_fp_hex: str, allowed_root: str,
                 tap_layers: Sequence[int] = TAP_LAYERS, verify: bool = True):
        self.dir = os.path.abspath(shard_dir)
        root = os.path.abspath(allowed_root)
        if not (self.dir == root or self.dir.startswith(root + os.sep)):
            raise AssertionError(f"training loader may only read under {root}, not {self.dir}")
        self.purpose = purpose
        self.layers = tuple(int(l) for l in tap_layers)
        self.basis_fp_hex = basis_fp_hex
        self._fds: List[int] = []
        self.items: Dict[str, dict] = {}
        self.shards: List[str] = []
        if os.path.isdir(self.dir):
            for n in sorted(os.listdir(self.dir)):
                if n.endswith(".tar") and not n.startswith("."):
                    self._index(os.path.join(self.dir, n), verify)
        self.keys: List[str] = sorted(self.items)

    def _index(self, path: str, verify: bool) -> None:
        if verify:
            SH.verify_shard(path, purpose=self.purpose)
        members: Dict[str, Dict[str, Tuple[int, int]]] = {}
        with tarfile.open(path, "r:") as tf:
            for ti in tf:
                if ti.isreg():
                    key, _, ext = ti.name.partition(".")
                    members.setdefault(key, {})[ext] = (ti.offset_data, ti.size)
        fd = os.open(path, os.O_RDONLY)
        fi = len(self._fds)
        self._fds.append(fd)
        self.shards.append(path)
        need = ["meta.json", "lab4.npz"] + [f"tap{l}.npy" for l in self.layers]
        for key, ext_map in members.items():
            if key in self.items:
                continue
            miss = [e for e in need if e not in ext_map]
            if miss:
                raise ShardDataError(f"{key}: missing fields {miss}")
            meta = json.loads(_pread_all(fd, ext_map["meta.json"][1], ext_map["meta.json"][0]))
            self._check_meta(key, meta)
            self.items[key] = {"fd": fi, "ext": ext_map, "meta": meta}

    def _check_meta(self, key: str, meta: dict) -> None:
        if meta.get("v") != SAMPLE_TAG:
            raise ShardDataError(f"{key}: meta.v={meta.get('v')!r}")
        if meta.get("basis_fp") != self.basis_fp_hex:
            raise T.TapBasisError(f"{key}: PCA basis fingerprint mismatch")
        if not meta.get("keep"):
            raise ShardDataError(f"{key}: keep=False sample in a shard")
        if tuple(meta.get("tap_layers") or ()) != self.layers:
            raise ShardDataError(f"{key}: tap_layers {meta.get('tap_layers')}")
        if not (0.0 < float(meta.get("tap_sigma", 0.0)) <= 1.0) or not meta.get("band_valid"):
            raise ShardDataError(f"{key}: tap_sigma / band_valid")

    def __len__(self) -> int:
        return len(self.keys)

    def meta(self, key: str) -> dict:
        return self.items[key]["meta"]

    def get(self, key: str) -> dict:
        it = self.items[key]
        fd, ext = self._fds[it["fd"]], it["ext"]
        taps = [SH.npy_decode(_pread_all(fd, ext[f"tap{l}.npy"][1], ext[f"tap{l}.npy"][0])) for l in self.layers]
        lab = load_npz_bytes(_pread_all(fd, ext["lab4.npz"][1], ext["lab4.npz"][0]))
        return {"key": key, "meta": it["meta"], "taps": taps, "log_depth": lab["log_depth"],
                "mask": lab["mask"], "normal": lab.get("normal")}

    def close(self) -> None:
        for fd in self._fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []


class StepPlan:
    """Which domain each global step draws from: ``random.Random(f"{seed}-dom-{step}")``, weights fixed at start.
    ``locate(step)`` -> (domain, number of earlier steps with the same domain)."""

    def __init__(self, seed: int, weights: Dict[str, float]):
        self.seed = int(seed)
        self.names = sorted(weights)
        tot = float(sum(weights.values()))
        self.cum = []
        c = 0.0
        for n in self.names:
            c += weights[n] / tot
            self.cum.append(c)
        self._doms: List[str] = []
        self._counts: Dict[str, int] = {n: 0 for n in self.names}
        self._before: List[int] = []
        self._lock = threading.Lock()

    def _extend(self, upto: int) -> None:
        while len(self._doms) <= upto:
            step = len(self._doms)
            u = random.Random(f"{self.seed}-dom-{step}").random()
            dom = next((n for n, c in zip(self.names, self.cum) if u < c), self.names[-1])
            self._before.append(self._counts[dom])
            self._counts[dom] += 1
            self._doms.append(dom)

    def locate(self, step: int) -> Tuple[str, int]:
        with self._lock:
            self._extend(step)
            return self._doms[step], self._before[step]


class BatchMaker:
    """``make(step)`` is a pure function of (seed, step): within a domain the order follows a ``(seed, domain,
    epoch)`` permutation, so a resumed run sees exactly the same data."""

    def __init__(self, datasets: Dict[str, LazyShardDataset], batch: Dict[str, int], weights: Dict[str, float],
                 seed: int, tap_sigma: float = TAP_SIGMA):
        self.ds = datasets
        self.batch = batch
        self.seed = int(seed)
        self.plan = StepPlan(seed, {d: weights[d] for d in datasets})
        self._perms: Dict[Tuple[str, int], List[int]] = {}
        self._lock = threading.Lock()
        self.t = float(tap_sigma)

    def _perm(self, dom: str, epoch: int, n: int) -> List[int]:
        with self._lock:
            p = self._perms.get((dom, epoch))
            if p is None:
                p = list(range(n))
                random.Random(f"{self.seed}-{dom}-{epoch}").shuffle(p)
                if len(self._perms) > 8:
                    self._perms.pop(next(iter(self._perms)))
                self._perms[(dom, epoch)] = p
            return p

    def keys_for(self, step: int) -> Tuple[str, List[str]]:
        dom, c = self.plan.locate(step)
        ds = self.ds[dom]
        n, b = len(ds), self.batch[dom]
        keys = []
        for j in range(b):
            epoch, off = divmod(c * b + j, n)
            keys.append(ds.keys[self._perm(dom, epoch, n)[off]])
        return dom, keys

    def make(self, step: int) -> dict:
        dom, keys = self.keys_for(step)
        return collate_samples([self.ds[dom].get(k) for k in keys], dom, self.t)


def collate_samples(samples: Sequence[dict], domain: str, t: float = TAP_SIGMA) -> dict:
    """Stack samples of one domain (same resolution) into CPU tensors; taps stay int8 (converted on the GPU)."""
    s0 = samples[0]
    nl = len(s0["taps"])
    taps = [torch.from_numpy(np.stack([s["taps"][i] for s in samples])) for i in range(nl)]
    ld = torch.from_numpy(np.stack([s["log_depth"] for s in samples])).float()
    mk = torch.from_numpy(np.stack([s["mask"] for s in samples])).float()
    has_n = all(s["normal"] is not None for s in samples)
    nrm = torch.from_numpy(np.stack([s["normal"] for s in samples]).astype(np.float32)) if has_n else None
    return {"domain": domain, "keys": [s["key"] for s in samples], "taps": taps, "log_depth": ld, "mask": mk,
            "normal": nrm, "band_valid": [tuple(s["meta"]["band_valid"]) for s in samples],
            "t": torch.full((len(samples),), float(t), dtype=torch.float32)}


def slice_batch(b: dict, lo: int, hi: int) -> dict:
    return {"domain": b["domain"], "keys": b["keys"][lo:hi], "taps": [t[lo:hi] for t in b["taps"]],
            "log_depth": b["log_depth"][lo:hi], "mask": b["mask"][lo:hi],
            "normal": None if b["normal"] is None else b["normal"][lo:hi], "band_valid": b["band_valid"][lo:hi],
            "t": b["t"][lo:hi]}


class OrderedPrefetcher:
    """Worker threads build ``fn(k)`` for k = start, start+1, ... and ``get(k)`` hands them out strictly in order
    (the step -> batch mapping must survive the threading)."""

    def __init__(self, fn: Callable[[int], object], start: int = 0, depth: int = 8, threads: int = 4):
        self.fn = fn
        self.depth = max(1, int(depth))
        self._next_claim = int(start)
        self._consume = int(start)
        self._ready: Dict[int, object] = {}
        self._cv = threading.Condition()
        self._halt = False
        self._threads = [threading.Thread(target=self._work, daemon=True) for _ in range(max(1, threads))]
        for t in self._threads:
            t.start()

    def _work(self) -> None:
        while True:
            with self._cv:
                while not self._halt and self._next_claim >= self._consume + self.depth:
                    self._cv.wait(0.2)
                if self._halt:
                    return
                k = self._next_claim
                self._next_claim += 1
            try:
                item = self.fn(k)
            except Exception as e:  # noqa: BLE001
                warn("batch_failed", k=k, err=err_str(e))
                item = None
            with self._cv:
                self._ready[k] = item
                self._cv.notify_all()

    def get(self, k: int, timeout: float = 5.0):
        with self._cv:
            end = time.time() + timeout
            while k not in self._ready:
                rem = end - time.time()
                if rem <= 0 or self._halt:
                    return _MISSING
                self._cv.wait(min(rem, 0.2))
            item = self._ready.pop(k)
            self._consume = k + 1
            self._cv.notify_all()
            return item

    def close(self) -> None:
        with self._cv:
            self._halt = True
            self._cv.notify_all()


_MISSING = object()


# ---------------------------------------------------------------------------
# model input scaling, forward, loss
# ---------------------------------------------------------------------------

def compute_tap_std(ds_list: Sequence[LazyShardDataset], basis: T.PCABasis, n: int = 256,
                    layers: Sequence[int] = TAP_LAYERS) -> Dict[int, torch.Tensor]:
    """Per layer, per component std of the dequantised taps (``q * scale``) over the first ``n`` sorted training keys
    (min 1e-6).  Computed once and frozen."""
    pairs = sorted((k, d) for d in ds_list for k in d.keys)[:n]
    if not pairs:
        raise ShardDataError("no training samples for tap_std")
    s1 = {l: torch.zeros(basis.k, dtype=torch.float64) for l in layers}
    s2 = {l: torch.zeros(basis.k, dtype=torch.float64) for l in layers}
    cnt = 0
    for key, d in pairs:
        smp = d.get(key)
        for l, q in zip(layers, smp["taps"]):
            x = torch.from_numpy(q).double() * basis.scale[basis._index(l)].double().view(-1, 1, 1)
            s1[l] += x.sum(dim=(1, 2))
            s2[l] += (x * x).sum(dim=(1, 2))
        cnt += smp["taps"][0].shape[1] * smp["taps"][0].shape[2]
    out = {}
    for l in layers:
        m = s1[l] / cnt
        out[l] = torch.sqrt((s2[l] / cnt - m * m).clamp_min(0.0)).clamp_min(1e-6).float()
    return out


def tap_mult(basis: T.PCABasis, tap_std: Dict[int, torch.Tensor], layers: Sequence[int] = TAP_LAYERS) -> Dict[int, torch.Tensor]:
    """Fixed per-component multiplier ``scale_l / tap_std_l``: feature = q (int8) * mult.  (Foldable into
    ``reassemble[i].proj.weight`` when exporting.)"""
    return {l: (basis.scale[basis._index(l)] / tap_std[l]).float() for l in layers}


def tap_std_hash(tap_std: Dict[int, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for l in sorted(tap_std):
        h.update(str(l).encode())
        h.update(tap_std[l].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


def make_features(taps_int8: Sequence[torch.Tensor], mult: Dict[int, torch.Tensor], device,
                  layers: Sequence[int] = TAP_LAYERS) -> Dict[int, torch.Tensor]:
    return {l: q.to(device, non_blocking=True).float() * mult[l].to(device).view(1, -1, 1, 1)
            for l, q in zip(layers, taps_int8)}


def mrh_forward_dn(model: "MH.MRH", feats: Dict[int, torch.Tensor], t: torch.Tensor, *, need_normal: bool,
                   amp: bool = False, device_type: str = "cpu") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Neck under bf16 autocast, depth (and normal) heads in fp32.  Same maths as ``MRH.forward`` for these two heads,
    without the other six (albedo, sem, ... are not trained here)."""
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        f4 = model.neck(feats, t).feat4
    with torch.autocast(device_type=device_type, enabled=False):
        z = F.relu(model.heads.trunk(f4.float()))
        depth = model.heads.branches["log_depth"](z).float()
        normal = MH._unit(model.heads.branches["normal"](z).float()) if need_normal else None
    return depth, normal


def _shift_only_depth_term(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, valid4: Sequence[bool]) -> torch.Tensor:
    """Shift-invariant band-limited log-depth loss, exponent fixed at 1 (no power distortion, no inversion):
    Bp, Bm, Bg = band_project(m*p, m, m*g); b = <Bm, Bg - Bp> / (<Bm, Bm> + 1e-8); r = Bp + b*Bm - Bg; <r, r> / n."""
    m = mask.double()
    stack = torch.stack([m * pred.double(), m, m * gt.double()])
    if not all(valid4):
        stack = R.band_project(stack, valid4)
    bp, bm, bg = stack.reshape(3, -1)
    b = (bm @ (bg - bp)) / (bm @ bm + 1e-8)
    r = bp + b * bm - bg
    return (r @ r) / torch.clamp(m.sum(), min=1.0)


def m3_loss(depth: torch.Tensor, normal: Optional[torch.Tensor], batch: dict, weights: Optional[dict] = None):
    """total = 1.0 L_d + 0.15 L_s + 0.5 L_n.

    L_d  shift-only band-limited log depth (above).  L_s = (sum m (p - g) / sum m)^2 keeps the metre scale weakly
    (GT is ln metres for both domains).  L_n = MDL-03 band-projected vector term on samples that have >= 16 pixels
    with a unit normal GT (synthetic only; DIODE batches never touch the normal head).  Samples with fewer than
    ``MIN_VALID_PX`` valid label pixels are left out of the depth terms."""
    w = {**LOSS_W, **(weights or {})}
    gt_ld = batch["log_depth"].to(depth.device)
    mask = batch["mask"].to(depth.device)
    gt_n = batch["normal"].to(depth.device) if (batch.get("normal") is not None and normal is not None) else None
    d_terms, s_terms, n_terms = [], [], []
    for i in range(depth.shape[0]):
        m = mask[i]
        n_px = float(m.sum())
        if n_px < MIN_VALID_PX:
            continue
        v4 = MT.band_valid_at_stride(batch["band_valid"][i], STRIDE)
        d_terms.append(_shift_only_depth_term(depth[i, 0], gt_ld[i], m, v4))
        s_terms.append(((m * (depth[i, 0] - gt_ld[i])).sum() / n_px) ** 2)
        if gt_n is not None:
            nm = m * (gt_n[i].norm(dim=0) > 0.5).float()
            if float(nm.sum()) >= MIN_VALID_PX:
                n_terms.append(MT._vector_term(normal[i], gt_n[i].float(), nm, v4))
    l_d = MT._mean_or_zero(d_terms, depth)
    l_s = MT._mean_or_zero([x.double() for x in s_terms], depth)
    ref_n = normal if normal is not None else depth
    l_n = MT._mean_or_zero(n_terms, ref_n)
    total = w["depth"] * l_d.float() + w["scale"] * l_s.float() + w["normal"] * l_n.float()
    parts = {"l_d": float(l_d.detach()), "l_s": float(l_s.detach()), "l_n": float(l_n.detach())}
    return total, parts


def assert_normal_grad_zero(model: "MH.MRH") -> None:
    """On a DIODE batch the normal head must receive exactly zero gradient (no normal GT there)."""
    for name, p in model.heads.branches["normal"].named_parameters():
        if p.grad is not None and bool((p.grad != 0).any()):
            raise AssertionError(f"normal head got gradient on a DIODE batch: {name}")


def lr_at(step: int, base_lr: float = 3e-4, warmup: int = 300, total: int = 6000, min_frac: float = 0.1) -> float:
    """Linear warm-up, then cosine down to ``min_frac * base_lr`` at ``total``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


class EMA:
    """Exponential moving average of the weights; warm-up decay ``min(decay, (1 + s) / (10 + s))``."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def decay_at(self, step: int) -> float:
        return min(self.decay, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self.decay_at(step)
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p.detach(), 1.0 - d)


# ---------------------------------------------------------------------------
# evaluation: alignments, metrics, bootstrap, baselines, tables
# ---------------------------------------------------------------------------

PRED_LO, PRED_HI = 1e-3, 1e4
ALIGNS = ("raw", "med", "ssd", "ssp")
ALIGN_NAMES = {"raw": "raw metres", "med": "median scale (headline)", "ssd": "scale+shift (depth)",
               "ssp": "scale+shift (disparity)"}
METRICS = ("absrel", "d1", "rmse_log")


def _lstsq_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Closed-form least squares ``s*x + t ~ y``; degenerate x -> (0, mean(y)) (the mean predictor)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = x.mean(), y.mean()
    xc = x - mx
    var = float((xc * xc).mean())
    if not var > 1e-12 * max(1.0, mx * mx):
        return 0.0, float(my)
    s = float((xc * (y - my)).mean()) / var
    return s, float(my - s * mx)


def align_median(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Scale only: ``pred * median(gt / pred)`` - fixes the scale, not the shape (exponent, shift)."""
    return pred * float(np.median(gt / pred))


def align_ss_depth(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Least squares ``s*pred + t`` in depth space (Marigold convention)."""
    s, t = _lstsq_affine(pred, gt)
    return s * pred + t


def align_ss_disp(pred_disp: np.ndarray, gt: np.ndarray, min_depth: float = 1.0 / EVAL_HI) -> np.ndarray:
    """Least squares ``s*disp + t`` against the GT disparity (MiDaS convention); returns depth.
    ``pred_disp`` is a disparity (larger = closer); the aligned disparity is floored at 1/EVAL_HI."""
    s, t = _lstsq_affine(pred_disp, 1.0 / gt)
    return 1.0 / np.maximum(s * pred_disp + t, 1.0 / EVAL_HI)


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """AbsRel, delta1 (< 1.25) and RMSE-log of positive depth vectors."""
    p = np.clip(pred.astype(np.float64), PRED_LO, PRED_HI)
    g = gt.astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {"absrel": float(np.mean(np.abs(p - g) / g)), "d1": float(np.mean(ratio < 1.25)),
            "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))}


def eval_mask(gt_depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return valid & np.isfinite(gt_depth) & (gt_depth >= EVAL_LO) & (gt_depth <= EVAL_HI)


def eval_image(pred_log: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray) -> Optional[dict]:
    """All four alignments for one image.  ``pred_log`` is ln(metres) at GT resolution.  None for an empty mask."""
    mask = eval_mask(gt_depth, valid)
    if int(mask.sum()) < 16:
        return None
    raw_p = np.exp(pred_log[mask].astype(np.float64))
    clamped = float(np.mean((raw_p < PRED_LO) | (raw_p > PRED_HI)))
    p = np.clip(raw_p, PRED_LO, PRED_HI)
    g = gt_depth[mask].astype(np.float64)
    out = {"raw": depth_metrics(p, g), "med": depth_metrics(align_median(p, g), g),
           "ssd": depth_metrics(align_ss_depth(p, g), g), "ssp": depth_metrics(align_ss_disp(1.0 / p, g), g),
           "n_px": int(mask.sum()), "clamped": clamped}
    return out


def eval_image_disp(disp: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray) -> Optional[dict]:
    """Relative-disparity predictor (Depth-Anything): only the disparity-space alignment is defined."""
    mask = eval_mask(gt_depth, valid)
    if int(mask.sum()) < 16:
        return None
    d = disp[mask].astype(np.float64)
    g = gt_depth[mask].astype(np.float64)
    return {"ssp": depth_metrics(align_ss_disp(d, g), g), "n_px": int(mask.sum())}


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Image-level bootstrap interval of the mean (deterministic: ``default_rng(seed)``)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def aggregate_records(records: Sequence[dict], aligns: Sequence[str] = ALIGNS) -> dict:
    """Per-image records ``{"env": ..., align: {metric: v}}`` -> mean of the per-image values for all / indoors /
    outdoor, with N."""
    out: Dict[str, dict] = {}
    for grp, sel in (("all", lambda e: True), ("indoors", lambda e: e == "indoors"), ("outdoor", lambda e: e == "outdoor")):
        recs = [r for r in records if sel(r.get("env"))]
        g: Dict[str, dict] = {"n": len(recs)}
        for a in aligns:
            vals = [r[a] for r in recs if a in r]
            if vals:
                g[a] = {m: float(np.mean([v[m] for v in vals])) for m in METRICS}
        out[grp] = g
    return out


def load_val_gt(base: str) -> Tuple[np.ndarray, np.ndarray]:
    z = load_npz(base + ".gt.npz")
    d = z["depth_m"].astype(np.float32)
    return d, unpack_val_valid(z["valid"], d.shape[1]) & (d > 0)


def upsample_log(pred: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """H/4 log depth -> full resolution, bilinear in log space (``align_corners=False``)."""
    return F.interpolate(pred, size=hw, mode="bilinear", align_corners=False)


def _stack_taps(npz_list: Sequence[Dict[int, np.ndarray]], layers: Sequence[int] = TAP_LAYERS) -> List[torch.Tensor]:
    return [torch.from_numpy(np.stack([t[l] for t in npz_list])) for l in layers]


@torch.no_grad()
def predict_log_depth(model: "MH.MRH", taps_int8: Sequence[torch.Tensor], mult: Dict[int, torch.Tensor], device,
                      amp: bool = False) -> torch.Tensor:
    """int8 taps [B,k,gh,gw] x3 -> H/4 log depth [B,1,4gh,4gw] (cpu float32)."""
    model.eval()
    feats = make_features(taps_int8, mult, device)
    t = torch.full((feats[TAP_LAYERS[0]].shape[0],), TAP_SIGMA, dtype=torch.float32, device=device)
    d, _ = mrh_forward_dn(model, feats, t, need_normal=False, amp=amp, device_type=torch.device(device).type)
    return d.float().cpu()


def eval_diode_val(model: "MH.MRH", paths: Paths, mult: Dict[int, torch.Tensor], device, *, amp: bool = False,
                   batch: int = 8, basis_fp: Optional[str] = None, limit: Optional[int] = None,
                   stems: Optional[Sequence[str]] = None) -> Tuple[List[dict], dict]:
    """Evaluate on the DIODE val images (771): H/4 log depth -> bilinear upsample in log space -> four alignments.
    Reads only ``eval/taps/val`` and the val GT.  Returns (per-image records, extra stats)."""
    samples = [(m, b) for m, b in list_diode_samples(paths, "val") if os.path.isfile(val_tap_path(paths, m.stem))]
    if stems is not None:
        want = set(stems)
        samples = [(m, b) for m, b in samples if m.stem in want]
    if limit is not None:
        samples = samples[:limit]
    recs: List[dict] = []
    clamped = []
    for s in range(0, len(samples), batch):
        sub = samples[s:s + batch]
        loaded = []
        for m, _b in sub:
            taps, meta = load_tap_npz(val_tap_path(paths, m.stem))
            if basis_fp is not None and meta.get("basis_fp") != basis_fp:
                raise T.TapBasisError(f"val tap {m.stem}: basis fingerprint mismatch")
            loaded.append(taps)
        by_hw: Dict[tuple, List[int]] = {}
        for i, t in enumerate(loaded):
            by_hw.setdefault(tuple(t[TAP_LAYERS[0]].shape), []).append(i)
        for idxs in by_hw.values():
            pred = predict_log_depth(model, _stack_taps([loaded[i] for i in idxs]), mult, device, amp)
            for j, i in enumerate(idxs):
                m, b = sub[i]
                gt, valid = load_val_gt(b)
                up = upsample_log(pred[j:j + 1], gt.shape)[0, 0].numpy()
                r = eval_image(up, gt, valid)
                if r is None:
                    continue
                clamped.append(r["clamped"])
                recs.append({"stem": m.stem, "env": m.env, **{a: r[a] for a in ALIGNS}})
    return recs, {"clamped": float(np.mean(clamped)) if clamped else 0.0, "n": len(recs)}


def const_log_depth(paths: Paths) -> Optional[float]:
    """ln of the constant raw-metre baseline: mean ln d over the valid pixels of the selected DIODE train crops
    (cached in eval/const_stats.json)."""
    p = os.path.join(paths.eval, "const_stats.json")
    c = read_json(p)
    if isinstance(c, dict) and "mean_ln" in c:
        return float(c["mean_ln"])
    s, n = 0.0, 0.0
    for _m, base in list_diode_samples(paths, "train"):
        z = load_npz(base + ".gt.npz")
        a, b = half_ln_stats(z["depth_m"], z["vcnt"])
        s += a
        n += b
    if n <= 0:
        return None
    write_json_atomic(p, {"mean_ln": s / n, "n_px": n})
    return s / n


def eval_const_baseline(paths: Paths, ln_c: float, stems: Optional[Sequence[str]] = None) -> List[dict]:
    recs = []
    for m, b in list_diode_samples(paths, "val"):
        if stems is not None and m.stem not in set(stems):
            continue
        gt, valid = load_val_gt(b)
        r = eval_image(np.full(gt.shape, ln_c, np.float32), gt, valid)
        if r is not None:
            recs.append({"stem": m.stem, "env": m.env, **{a: r[a] for a in ALIGNS}})
    return recs


def eval_lab4_depth(pred_ld: np.ndarray, gt_ld: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    """Median-scale-aligned AbsRel / delta1 at label resolution (dev sets: model selection, not reporting)."""
    m = mask > 0.5
    if int(m.sum()) < MIN_VALID_PX:
        return None
    p = np.exp(pred_ld[m].astype(np.float64))
    g = np.exp(gt_ld[m].astype(np.float64))
    return depth_metrics(align_median(p, g), g)


def normal_angle_stats(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    """Angle (degrees) between predicted and GT unit normals over valid pixels (GT normalised first)."""
    gn = np.linalg.norm(gt.astype(np.float64), axis=0)
    m = (mask > 0.5) & (gn > 0.5)
    if int(m.sum()) < MIN_VALID_PX:
        return None
    g = gt.astype(np.float64) / np.maximum(gn, 1e-12)
    p = pred.astype(np.float64)
    p = p / np.maximum(np.linalg.norm(p, axis=0), 1e-12)
    ang = np.degrees(np.arccos(np.clip((p * g).sum(0)[m], -1.0, 1.0)))
    return {"ang": ang}


def summarize_angles(chunks: Sequence[np.ndarray]) -> dict:
    if not chunks:
        return {}
    a = np.concatenate(chunks)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "lt11": float((a < 11.25).mean()),
            "lt22": float((a < 22.5).mean()), "lt30": float((a < 30.0).mean()), "n_px": int(a.size)}


@torch.no_grad()
def eval_dev_set(model: "MH.MRH", ds: Optional[LazyShardDataset], mult: Dict[int, torch.Tensor], device, *,
                 amp: bool = False, batch: int = 8, with_normal: bool = False, limit: Optional[int] = None) -> Optional[dict]:
    """Median-aligned AbsRel / delta1 (+ normal angle stats) on a dev shard set at H/4 resolution."""
    if ds is None or len(ds) == 0:
        return None
    model.eval()
    keys = ds.keys[:limit] if limit else ds.keys
    recs, angs = [], []
    groups: Dict[tuple, List[str]] = {}
    for k in keys:
        groups.setdefault(tuple(ds.items[k]["meta"]["grid"]), []).append(k)
    for ks in groups.values():
        for s in range(0, len(ks), batch):
            b = collate_samples([ds.get(k) for k in ks[s:s + batch]], "dev")
            feats = make_features(b["taps"], mult, device)
            d, n = mrh_forward_dn(model, feats, b["t"].to(device), need_normal=with_normal, amp=amp,
                                  device_type=torch.device(device).type)
            d = d.cpu().numpy()
            n = None if n is None else n.cpu().numpy()
            for i in range(d.shape[0]):
                r = eval_lab4_depth(d[i, 0], b["log_depth"][i].numpy(), b["mask"][i].numpy())
                if r is not None:
                    recs.append(r)
                if with_normal and n is not None and b["normal"] is not None:
                    a = normal_angle_stats(n[i], b["normal"][i].numpy(), b["mask"][i].numpy())
                    if a is not None:
                        angs.append(a["ang"])
    if not recs:
        return None
    out = {"n": len(recs), "absrel": float(np.mean([r["absrel"] for r in recs])), "d1": float(np.mean([r["d1"] for r in recs])),
           "rmse_log": float(np.mean([r["rmse_log"] for r in recs]))}
    if with_normal:
        out["normal"] = summarize_angles(angs)
    return out


def _fmt(v, nd=3) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) and math.isfinite(v) else str(v))


def build_table(step: int, mrh: dict, const: Optional[dict], dav2: Optional[dict], ci: Optional[dict],
                extra: Optional[dict] = None) -> dict:
    """The evaluation table (json): rows mrh / const / dav2, groups all / indoors / outdoor, four alignments."""
    return {"step": step, "schema": "d1_m3_eval/v1", "rows": {"mrh": mrh, "const": const, "dav2": dav2}, "ci95": ci,
            "headline": "DIODE val median-scale-aligned AbsRel / delta1 (indoors / outdoor)", **(extra or {})}


def table_markdown(tab: dict) -> str:
    rows = tab["rows"]
    L = [f"# M3 MRH spike - evaluation (step {tab['step']})", "",
         "Headline: DIODE val, median-scale-aligned AbsRel / delta1 (scale fixed per image, shape not touched).",
         "Raw metres and scale+shift columns are supporting numbers; the disparity-space alignment exists so that "
         "Depth-Anything-V2 (relative disparity) can be compared apples to apples.", ""]
    for grp in ("all", "indoors", "outdoor"):
        n = (rows["mrh"] or {}).get(grp, {}).get("n")
        L += [f"## {grp} (N={n})", "", "| model | raw AbsRel | raw d1 | med AbsRel | med d1 | ss-depth AbsRel | ss-depth d1 "
              "| ss-disp AbsRel | ss-disp d1 |", "|---|---|---|---|---|---|---|---|---|"]
        for name, key in (("MRH (EMA)", "mrh"), ("constant depth", "const"), ("Depth-Anything-V2-Small", "dav2")):
            r = rows.get(key)
            g = (r or {}).get(grp) or {}
            cells = []
            for a in ALIGNS:
                v = g.get(a)
                cells += [_fmt(v["absrel"]) if v else "n/a", _fmt(v["d1"]) if v else "n/a"]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    if tab.get("ci95"):
        c = tab["ci95"]
        L += [f"95% bootstrap CI (1000 resamples, seed 0, images): median AbsRel {_fmt(c['med_absrel'][0])}..{_fmt(c['med_absrel'][1])}, "
              f"median d1 {_fmt(c['med_d1'][0])}..{_fmt(c['med_d1'][1])}.", ""]
    for k in ("synth_dev", "diode_dev", "clamped"):
        if tab.get(k) is not None:
            L.append(f"{k}: `{json.dumps(tab[k], default=_json_default)}`")
    L += ["", "Notes:", "- constant row: raw = exp(mean ln d over the selected DIODE train crops); median-aligned = the "
          "per-image GT median (it knows the true scale: a floor, not a model); scale+shift is degenerate (mean).",
          "- Depth-Anything-V2-Small (Apache-2.0) is an evaluation reference only; whether DIODE was in its training "
          "data was not checked (it is described as zero-shot). Its output never enters any training path.",
          "- A bad raw-metre column is reported as is."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Depth-Anything-V2-Small baseline (B0, evaluation only)
# ---------------------------------------------------------------------------

class DepthAnythingSmall:
    """``__call__(rgb_u8 HxWx3) -> disparity (H, W) float32`` through transformers (lazy)."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(DAV2_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(DAV2_ID).to(device).eval()

    @torch.no_grad()
    def __call__(self, rgb_u8: np.ndarray) -> np.ndarray:
        h, w = rgb_u8.shape[:2]
        inp = self.proc(images=rgb_u8, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        pd = self.model(**inp).predicted_depth
        pd = F.interpolate(pd[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)[0, 0]
        return pd.cpu().numpy().astype(np.float32)


def pick_strip_stems(samples: Sequence[Tuple[Member, str]], n_each: int = 3) -> List[str]:
    """Fixed validation images for the strip: ``n_each`` indoor + ``n_each`` outdoor, evenly spread over the sorted list."""
    out = []
    for env in ("indoors", "outdoor"):
        stems = sorted(m.stem for m, _ in samples if m.env == env)
        out += [stems[i] for i in choose_idx(len(stems), n_each)]
    return out


def choose_idx(n: int, k: int) -> List[int]:
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted({int(round((j + 0.5) * n / k - 0.5)) for j in range(k)})


def dav2_paths(paths: Paths) -> Dict[str, str]:
    return {"metrics": os.path.join(paths.dav2, "metrics.json"), "strip": os.path.join(paths.eval, "strip_stems.json")}


def run_dav2_stage(paths: Paths, args, deps: Optional[dict] = None) -> dict:
    """B0: Depth-Anything-V2-Small on the DIODE val images (disparity-space alignment only) + the maps needed for
    the preview strip and the yak.  Everything lands in eval/dav2 (the yak map in private/); nothing is a label."""
    deps = deps or {}
    now, sleep = deps.get("now", time.time), deps.get("sleep", time.sleep)
    if stage_done(paths, "dav2"):
        return {"status": "done"}
    ok = wait_until(lambda: stage_done(paths, "diode_val"), args.wait_diode_s, now=now, sleep=sleep)
    if not ok:
        return {"status": "no_val"}
    apply_vram_cap(args.dav2_vram_gib)
    factory = deps.get("dav2_factory")
    pred_fn = factory() if factory else DepthAnythingSmall(args.device or "cuda")
    samples = list_diode_samples(paths, "val")
    strip = pick_strip_stems(samples)
    os.makedirs(paths.dav2, exist_ok=True)
    per: Dict[str, dict] = {}
    for m, base in samples:
        with Image.open(base + ".png") as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        disp = pred_fn(rgb)
        gt, valid = load_val_gt(base)
        r = eval_image_disp(disp, gt, valid)
        if r is None:
            continue
        per[m.stem] = {"env": m.env, **r["ssp"], "n_px": r["n_px"]}
        if m.stem in strip:
            write_atomic(os.path.join(paths.dav2, f"preview_{m.stem}.npz"), npz_bytes({"disp": disp.astype(np.float16)}))
    write_json_atomic(dav2_paths(paths)["metrics"], per)
    write_json_atomic(dav2_paths(paths)["strip"], strip)
    yak = "no_token"
    src = fetch_yak(paths, os.environ.get("HF_TOKEN"), deps.get("yak_download"))
    if src is not None:
        img = (load_yak_float(src) * 255.0).round().astype(np.uint8)
        np_path = safe_out_path(paths.yak_dav2, paths)
        write_atomic(np_path, npy_bytes(pred_fn(img).astype(np.float16)))
        yak = "ok"
    del pred_fn
    free_cuda_cache()
    write_stage_done(paths, "dav2", "dav2", {"n": len(per), "yak": yak})
    log({"stage": "dav2", "n": len(per), "yak": yak})
    return {"status": "done", "n": len(per), "yak": yak}


def dav2_records(paths: Paths) -> Optional[List[dict]]:
    d = read_json(dav2_paths(paths)["metrics"])
    if not isinstance(d, dict) or not d:
        return None
    return [{"stem": s, "env": v["env"], "ssp": {m: v[m] for m in METRICS}} for s, v in sorted(d.items())]


# ---------------------------------------------------------------------------
# previews: turbo colour map, depth panels, point cloud splats (numpy only)
# ---------------------------------------------------------------------------

_TURBO_R4 = (0.13572138, 4.61539260, -42.66032258, 132.13108234)
_TURBO_R2 = (-152.94239396, 59.28637943)
_TURBO_G4 = (0.09140261, 2.19418839, 4.84296658, -14.18503333)
_TURBO_G2 = (4.27729857, 2.82956604)
_TURBO_B4 = (0.10667330, 12.64194608, -60.58204836, 110.36276771)
_TURBO_B2 = (-89.90310912, 27.34824973)


def turbo(t: np.ndarray) -> np.ndarray:
    """Turbo colour map (polynomial approximation), t in [0, 1] -> uint8 [..., 3]."""
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    v4 = np.stack([np.ones_like(x), x, x * x, x ** 3], axis=-1)
    v2 = np.stack([x ** 4, x ** 5], axis=-1)
    ch = []
    for a4, a2 in ((_TURBO_R4, _TURBO_R2), (_TURBO_G4, _TURBO_G2), (_TURBO_B4, _TURBO_B2)):
        ch.append((v4 * np.asarray(a4)).sum(-1) + (v2 * np.asarray(a2)).sum(-1))
    return np.round(np.clip(np.stack(ch, axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)


def depth_vis(depth: np.ndarray, near_is_large: bool = False) -> np.ndarray:
    """Median-normalised depth (or disparity when ``near_is_large``) -> turbo; nearer = warmer."""
    d = np.asarray(depth, dtype=np.float64)
    med = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else 1.0
    x = np.nan_to_num(d / (med if abs(med) > 1e-9 else 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    t = np.clip(x / 2.0, 0.0, 1.0) if near_is_large else 1.0 - np.clip(x / 2.0, 0.0, 1.0)
    return turbo(t)


def unproject_depth(depth: np.ndarray, hfov_deg: float = 60.0) -> np.ndarray:
    """z-depth map (h, w) -> points (h*w, 3) in camera space (x right, y down, z forward), HFOV assumed."""
    h, w = depth.shape
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = depth.astype(np.float64)
    return np.stack([(u - w / 2.0) / fx * z, (v - h / 2.0) / fx * z, z], axis=-1).reshape(-1, 3)


def splat_view(pts: np.ndarray, colors: np.ndarray, yaw_deg: float, dolly: float, hfov_deg: float,
               out_hw: Tuple[int, int], pivot_z: float) -> np.ndarray:
    """z-buffer splat (one pixel per point) of the points seen from a camera yawed around the vertical axis through
    (0, 0, pivot_z) and moved ``dolly`` forward.  Returns uint8 (H, W, 3); holes are dark grey."""
    H, W = out_hw
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    p = pts - np.array([0.0, 0.0, pivot_z])
    x = c * p[:, 0] + s * p[:, 2]
    z = -s * p[:, 0] + c * p[:, 2]
    z = z + pivot_z - dolly
    y = p[:, 1]
    fx = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    ok = z > 1e-3
    u = np.round(x[ok] / z[ok] * fx + W / 2.0 - 0.5).astype(np.int64)
    v = np.round(y[ok] / z[ok] * fx + H / 2.0 - 0.5).astype(np.int64)
    zz, cc = z[ok], colors[ok]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zz, cc = u[inb], v[inb], zz[inb], cc[inb]
    img = np.full((H, W, 3), 40, np.uint8)
    if u.size == 0:
        return img
    lin = v * W + u
    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, lin, zz)
    win = zz <= zbuf[lin]
    img.reshape(-1, 3)[lin[win]] = cc[win]
    return img


def pcd_strip(depth: np.ndarray, rgb: np.ndarray, width: int = 768, hfov_deg: float = 60.0) -> np.ndarray:
    """Three views (yaw -12 deg, 0 deg with a dolly-in of 0.3 x median depth, +12 deg) side by side, total ``width`` px."""
    h, w = depth.shape
    pts = unproject_depth(depth, hfov_deg)
    cols = rgb.reshape(-1, 3)
    med = float(np.median(depth))
    vw = width // 3
    vh = int(round(vw * h / w))
    views = [splat_view(pts, cols, yaw, dolly * med, hfov_deg, (vh, vw), med) for yaw, dolly in ((-12.0, 0.0), (0.0, 0.3), (12.0, 0.0))]
    return np.concatenate(views, axis=1)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 14], fill=(0, 0, 0))
    d.text((4, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def _fit_cell(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def compose_grid(cells: Sequence[Sequence[np.ndarray]], cell_wh: Tuple[int, int]) -> np.ndarray:
    rows = [np.concatenate([_fit_cell(c, cell_wh) for c in row], axis=1) for row in cells]
    return np.concatenate(rows, axis=0)


def yak_panel(rgb: np.ndarray, mrh_log_depth: np.ndarray, dav2_disp: Optional[np.ndarray], cell_w: int = 384) -> Tuple[np.ndarray, np.ndarray]:
    """2x2 panel (input | MRH depth ; DA-V2 reference | point cloud) and the 3-view point cloud strip (width 768)."""
    h, w = mrh_log_depth.shape
    big = np.asarray(Image.fromarray(np.round(np.clip(rgb, 0, 255)).astype(np.uint8)).resize((w * 2, h * 2), Image.BILINEAR))
    d_up = F.interpolate(torch.from_numpy(mrh_log_depth.astype(np.float32))[None, None], size=(h * 2, w * 2),
                         mode="bilinear", align_corners=False)[0, 0].numpy()
    depth = np.exp(d_up)
    strip = pcd_strip(depth, big)
    mrh_vis = depth_vis(depth)
    if dav2_disp is not None:
        dv = np.asarray(Image.fromarray(dav2_disp.astype(np.float32)).resize((w * 2, h * 2), Image.BILINEAR))
        ref = _label(depth_vis(dv, near_is_large=True), "DA-V2-S (eval reference)")
    else:
        ref = _label(np.full_like(mrh_vis, 60), "DA-V2-S: n/a")
    ch = int(round(cell_w * h / w))
    pcd_cell = strip
    panel = compose_grid([[_label(big, "input"), _label(mrh_vis, "MRH depth (median-normalised)")],
                          [ref, _label(pcd_cell, "point cloud (-12 / 0+dolly / +12 deg)")]], (cell_w, ch))
    return panel, strip


def save_png(path: str, arr: np.ndarray, paths: Paths) -> str:
    p = safe_out_path(path, paths)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    write_atomic(p, bio.getvalue())
    return p


def make_yak_preview(paths: Paths, model: "MH.MRH", mult: Dict[int, torch.Tensor], device, step: int,
                     amp: bool = False) -> Optional[List[str]]:
    """yak_step{step:06d}.png (+ yak_latest.png, yak_pcd_step{step:06d}.png).  Needs private/yak_taps.npz and the
    reading copy private/yak_input.png; the output only ever goes to <out>/preview and the private vault."""
    tp, inp = safe_out_path(paths.yak_taps, paths), safe_out_path(os.path.join(paths.private, "yak_input.png"), paths)
    if not (os.path.isfile(tp) and os.path.isfile(inp)):
        return None
    taps, _meta = load_tap_npz(tp)
    pred = predict_log_depth(model, _stack_taps([taps]), mult, device, amp)[0, 0].numpy()
    with Image.open(inp) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    dav2 = None
    if os.path.isfile(paths.yak_dav2):
        dav2 = np.load(paths.yak_dav2, allow_pickle=False).astype(np.float32)
    panel, strip = yak_panel(rgb, pred, dav2)
    out = [save_png(os.path.join(paths.preview, f"yak_step{step:06d}.png"), panel, paths),
           save_png(os.path.join(paths.preview, "yak_latest.png"), panel, paths),
           save_png(os.path.join(paths.preview, f"yak_pcd_step{step:06d}.png"), strip, paths)]
    return out


def make_val_strip(paths: Paths, model: "MH.MRH", mult: Dict[int, torch.Tensor], device, amp: bool = False,
                   cell_wh: Tuple[int, int] = (256, 192)) -> Optional[str]:
    """diode_val_strip.png: six fixed val images, columns RGB | GT | MRH | DA-V2 (reference)."""
    stems = read_json(dav2_paths(paths)["strip"])
    if not stems:
        stems = pick_strip_stems(list_diode_samples(paths, "val"))
    by = {m.stem: b for m, b in list_diode_samples(paths, "val")}
    rows = []
    for stem in stems:
        if stem not in by or not os.path.isfile(val_tap_path(paths, stem)):
            continue
        taps, _ = load_tap_npz(val_tap_path(paths, stem))
        pred = predict_log_depth(model, _stack_taps([taps]), mult, device, amp)
        with Image.open(by[stem] + ".png") as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        gt, valid = load_val_gt(by[stem])
        up = upsample_log(pred, gt.shape)[0, 0].numpy()
        gvis = depth_vis(np.where(valid, gt, np.nan))
        gvis[~valid] = 0
        dpath = os.path.join(paths.dav2, f"preview_{stem}.npz")
        dv = depth_vis(load_npz(dpath)["disp"].astype(np.float32), near_is_large=True) if os.path.isfile(dpath) \
            else np.full_like(rgb, 60)
        rows.append([rgb, gvis, depth_vis(np.exp(up)), dv])
    if not rows:
        return None
    return save_png(os.path.join(paths.preview, "diode_val_strip.png"), compose_grid(rows, cell_wh), paths)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

RUN_MIX, RUN_SYN = "m3d1-mix", "m3d1-syn"
DOMAIN_WEIGHTS = {RUN_MIX: {"diode": 0.6, "synth": 0.4}, RUN_SYN: {"synth": 1.0}}
GLOBAL_BATCH = {"diode": 8, "synth": 16}
MICRO_BATCH = {"diode": 4, "synth": 16}


def decide_run_id(fw_ok: bool, n_diode_train: int, ckpt_run_id: Optional[str], ckpt_run: Optional[str],
                  run_name: str) -> str:
    """The domain mix is fixed when a run starts.  A restart of the same run keeps its run_id even if the firewall
    amendment arrives meanwhile; a new run (night2) starts ``m3d1-mix`` as soon as DIODE is trainable."""
    if ckpt_run_id in DOMAIN_WEIGHTS and ckpt_run == run_name:
        return ckpt_run_id
    return RUN_MIX if (fw_ok and n_diode_train > 0) else RUN_SYN


def mrh_config(deps: Optional[dict] = None) -> "MN.MRHConfig":
    return (deps or {}).get("mrh_cfg") or MN.MRHConfig()


def build_train_config(args, run_id: str, basis_fp: str, tap_hash: str, cfg: "MN.MRHConfig") -> dict:
    return {"total_steps": args.total_steps, "warmup": args.warmup, "lr": args.lr, "wd": args.wd,
            "loss": LOSS_W, "run_id": run_id, "basis_fp": basis_fp, "tap_std": tap_hash,
            "weights": DOMAIN_WEIGHTS[run_id], "batch": GLOBAL_BATCH, "mrh": cfg.to_dict(), "ema": args.ema_decay,
            "seed": args.seed}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class FatalTrain(RuntimeError):
    pass


class M3Trainer:
    def __init__(self, args, paths: Paths, deps: Optional[dict] = None, uploader=None):
        self.a = args
        self.paths = paths
        self.deps = deps or {}
        self.now: Callable[[], float] = self.deps.get("now", time.time)
        self.sleep: Callable[[float], None] = self.deps.get("sleep", time.sleep)
        self.stop = self.deps.get("stop_event") or threading.Event()
        self.uploader = uploader
        self.deadline = parse_deadline(args.deadline)
        self.step = 0
        self.data_k = 0
        self.mb = dict(MICRO_BATCH)
        self.best = math.inf
        self.skipped = 0
        self.bad_run = 0
        self.fail_streak = 0
        self.oom_events = 0
        self.model: Optional["MH.MRH"] = None
        self.ema: Optional[EMA] = None
        self.opt = None
        self.device = None
        self.amp = False
        self.basis: Optional[T.PCABasis] = None
        self.basis_fp = ""
        self.tap_std: Dict[int, torch.Tensor] = {}
        self.mult: Dict[int, torch.Tensor] = {}
        self.run_id = RUN_SYN
        self.cfg_hash = ""
        self.ds: Dict[str, LazyShardDataset] = {}
        self.dev: Dict[str, Optional[LazyShardDataset]] = {}
        self.upload_thread: Optional[threading.Thread] = None
        self.last_eval: dict = {}
        self._diode_checked = False
        self._last_eval_step = -1
        self._acc = {"loss": 0.0, "l_d": 0.0, "l_s": 0.0, "l_n": 0.0, "n": 0}
        self.backend = (read_backend(paths) or {}).get("backend", "?")
        self.mrh_cfg = mrh_config(self.deps)
        self.t_start = self.now()
        self.budget_s = None

    # -- stop / budget -------------------------------------------------------
    def stop_reason(self) -> Optional[str]:
        if self.stop.is_set():
            return "signal"
        if self.deadline is not None and self.now() >= self.deadline - self.a.finish_margin_s:
            return "deadline"
        if self.budget_s is not None and self.now() - self.t_start >= self.budget_s:
            return "budget"
        if self.a.max_steps is not None and self.step >= self.a.max_steps:
            return "max_steps"
        if self.step >= self.a.total_steps:
            return "total_steps"
        return None

    # -- done flag -----------------------------------------------------------
    def already_done(self) -> Optional[dict]:
        d = read_json(self.paths.done)
        if isinstance(d, dict) and d.get("status") == "done" and d.get("run") == self.a.run \
                and int(d.get("step", -1)) >= int(self.a.total_steps) and int(d.get("total_steps", -1)) == int(self.a.total_steps):
            return d
        return None

    def write_done(self, summary: dict) -> None:
        write_json_atomic(self.paths.done, {**summary, "status": "done", "run": self.a.run, "total_steps": self.a.total_steps,
                                            "finished_at": time.time()})

    # -- setup ---------------------------------------------------------------
    def _setup_device(self) -> None:
        a = self.a
        name = self.deps.get("device") or a.device or ("cuda" if cuda_ready() else "cpu")
        self.device = torch.device(name)
        self.amp = {"on": True, "off": False}.get(a.amp, self.device.type == "cuda")
        if self.device.type == "cuda":
            apply_vram_cap(a.train_vram_gib)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(a.seed)

    def _open_datasets(self) -> None:
        p, fp = self.paths, self.basis_fp
        def mk(base, dom, purpose, root):
            d = os.path.join(base, dom)
            return LazyShardDataset(d, purpose=purpose, basis_fp_hex=fp, allowed_root=root)
        self.ds = {"synth": mk(p.shards_train, "synth", "train", p.shards_train),
                   "diode": mk(p.shards_train, "diode", "train", p.shards_train)}
        self.dev = {"synth": mk(p.shards_dev, "synth", "dev", p.shards_dev),
                    "diode": mk(p.shards_dev, "diode", "dev", p.shards_dev)}
        for d in list(self.dev):
            if len(self.dev[d]) == 0:
                self.dev[d] = None

    def _load_or_make_tap_std(self, ck: Optional[dict]) -> None:
        f = load_ckpt(self.paths.tap_std, need=("std",))
        if f is not None and all(l in f["std"] for l in TAP_LAYERS) and f.get("basis_fp") == self.basis_fp:
            self.tap_std = {int(l): v.float() for l, v in f["std"].items()}
        elif ck is not None and ck.get("tap_std") and ck.get("basis_fp") == self.basis_fp:
            self.tap_std = {int(l): v.float() for l, v in ck["tap_std"].items()}
            save_atomic({"std": self.tap_std, "basis_fp": self.basis_fp}, self.paths.tap_std)
        else:
            use = [d for d in self.ds.values() if len(d)]
            self.tap_std = compute_tap_std(use, self.basis, 256, TAP_LAYERS)
            save_atomic({"std": self.tap_std, "basis_fp": self.basis_fp}, self.paths.tap_std)
        self.mult = tap_mult(self.basis, self.tap_std)

    # -- checkpoints ---------------------------------------------------------
    def _ckpt_obj(self) -> dict:
        return {"version": 1, "step": self.step, "config": build_train_config(self.a, self.run_id, self.basis_fp,
                                                                              tap_std_hash(self.tap_std), self.mrh_cfg),
                "config_hash": self.cfg_hash, "model": self.model.state_dict(), "ema": self.ema.model.state_dict(),
                "optimizer": self.opt.state_dict(), "scheduler": {"lr": lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps)},
                "micro_batch": dict(self.mb), "rng": {"data_k": self.data_k, "torch": torch.get_rng_state()},
                "best": None if math.isinf(self.best) else self.best, "tap_std": self.tap_std, "basis_fp": self.basis_fp,
                "run_id": self.run_id, "run": self.a.run, "backend": self.backend, "skipped": self.skipped,
                "saved_at": time.time()}

    def save_last(self) -> None:
        try:
            save_atomic(self._ckpt_obj(), self.paths.ckpt_last)
        except Exception as e:  # noqa: BLE001
            warn("ckpt_save_failed", err=err_str(e, 300))

    def _save_best(self, metric: float) -> None:
        try:
            save_atomic({"version": 1, "step": self.step, "ema": self.ema.model.state_dict(),
                         "config": build_train_config(self.a, self.run_id, self.basis_fp, tap_std_hash(self.tap_std), self.mrh_cfg),
                         "config_hash": self.cfg_hash, "tap_std": self.tap_std, "basis_fp": self.basis_fp,
                         "sel_metric": metric, "run_id": self.run_id, "backend": self.backend}, self.paths.ckpt_best)
        except Exception as e:  # noqa: BLE001
            warn("ckpt_best_failed", err=err_str(e, 300))

    def _restore(self, ck: dict, warm: bool = False) -> bool:
        try:
            self.model.load_state_dict(ck["model"])
            self.ema.model.load_state_dict(ck["ema"])
            if warm:
                return True
            self.opt.load_state_dict(ck["optimizer"])
            self.step = int(ck["step"])
            self.data_k = int((ck.get("rng") or {}).get("data_k", self.step))
            b = ck.get("best")
            self.best = math.inf if b is None else float(b)
            mb = ck.get("micro_batch") or {}
            for d in self.mb:
                if d in mb:
                    self.mb[d] = max(1, min(self.mb[d], int(mb[d])))     # never grow back past an OOM size
            self.skipped = int(ck.get("skipped", 0))
            return True
        except Exception as e:  # noqa: BLE001
            warn("ckpt_restore_failed", err=err_str(e))
            return False

    def _resume(self) -> Optional[dict]:
        ck = load_ckpt(self.paths.ckpt_last, need=("step", "config_hash", "model", "ema", "optimizer"))
        if ck is None:
            return None
        return ck

    # -- one optimisation step ---------------------------------------------------
    def _loss(self, sub: dict):
        feats = make_features(sub["taps"], self.mult, self.device)
        depth, normal = mrh_forward_dn(self.model, feats, sub["t"].to(self.device), need_normal=sub["normal"] is not None,
                                       amp=self.amp, device_type=self.device.type)
        return m3_loss(depth, normal, sub)

    def train_step(self, batch: dict) -> bool:
        dom, B = batch["domain"], len(batch["keys"])
        lr = lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr
        while True:
            mb = max(1, min(self.mb[dom], B))
            acc = {"loss": 0.0, "l_d": 0.0, "l_s": 0.0, "l_n": 0.0}
            try:
                self.opt.zero_grad(set_to_none=True)
                for lo in range(0, B, mb):
                    sub = slice_batch(batch, lo, lo + mb)
                    w = len(sub["keys"]) / B
                    loss, parts = self._loss(sub)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("non-finite loss")
                    (loss * w).backward()
                    acc["loss"] += float(loss.detach()) * w
                    for k in ("l_d", "l_s", "l_n"):
                        acc[k] += parts[k] * w
                break
            except FloatingPointError:
                self.opt.zero_grad(set_to_none=True)
                self.skipped += 1
                self.bad_run += 1
                return False
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                self.opt.zero_grad(set_to_none=True)
                free_cuda_cache()
                if mb <= 1:
                    raise
                self.mb[dom] = max(1, mb // 2)
                self.oom_events += 1
                warn("oom_backoff", step=self.step, domain=dom, mb=self.mb[dom], accum=-(-B // self.mb[dom]), vram_gb=vram_gb())
                self.save_last()
        if dom == "diode" and not self._diode_checked:
            assert_normal_grad_zero(self.model)
            self._diode_checked = True
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if not bool(torch.isfinite(gn)):
            self.opt.zero_grad(set_to_none=True)
            self.skipped += 1
            self.bad_run += 1
            return False
        self.opt.step()
        self.ema.update(self.model, self.step)
        self.step += 1
        self.bad_run = 0
        for k in ("loss", "l_d", "l_s", "l_n"):
            self._acc[k] += acc[k]
        self._acc["n"] += 1
        return True

    # -- evaluation --------------------------------------------------------------
    def _const_agg(self, n_val: int) -> Optional[dict]:
        p = os.path.join(self.paths.eval, "const_agg.json")
        c = read_json(p)
        if isinstance(c, dict) and c.get("n_val") == n_val and c.get("agg"):
            return c["agg"]
        ln_c = const_log_depth(self.paths)
        if ln_c is None:
            return None
        agg = aggregate_records(eval_const_baseline(self.paths, ln_c))
        write_json_atomic(p, {"n_val": n_val, "ln_c": ln_c, "agg": agg})
        return agg

    def evaluate(self, final: bool = False) -> dict:
        """EMA weights: DIODE val (reporting), DIODE dev / synthetic dev (model selection), tables, previews."""
        p, ema = self.paths, self.ema.model
        sel_name = "diode" if (self.run_id == RUN_MIX and self.dev.get("diode") is not None) else "synth"
        out: dict = {"step": self.step}
        dev_d = eval_dev_set(ema, self.dev.get("diode"), self.mult, self.device, amp=self.amp)
        dev_s = eval_dev_set(ema, self.dev.get("synth"), self.mult, self.device, amp=self.amp, with_normal=True)
        out["diode_dev"], out["synth_dev"] = dev_d, dev_s
        sel = {"diode": dev_d, "synth": dev_s}[sel_name]
        out["dev_absrel"] = None if sel is None else sel["absrel"]
        recs, ex = [], {"clamped": 0.0}
        n_val_taps = len(os.listdir(p.val_taps)) if os.path.isdir(p.val_taps) else 0
        if n_val_taps:
            recs, ex = eval_diode_val(ema, p, self.mult, self.device, amp=self.amp, basis_fp=self.basis_fp)
        agg = aggregate_records(recs)
        ci = None
        if recs:
            ci = {"med_absrel": bootstrap_ci([r["med"]["absrel"] for r in recs]),
                  "med_d1": bootstrap_ci([r["med"]["d1"] for r in recs])}
        const = self._const_agg(len(recs)) if recs else None
        dr = dav2_records(p)
        dav2 = aggregate_records(dr, aligns=("ssp",)) if dr else None
        tab = build_table(self.step, agg, const, dav2, ci, {"run_id": self.run_id, "backend": self.backend,
                                                            "synth_dev": dev_s, "diode_dev": dev_d, "clamped": ex["clamped"],
                                                            "sel": sel_name, "n_val": len(recs)})
        try:
            write_json_atomic(p.eval_json, tab)
            write_atomic(p.eval_md, table_markdown(tab).encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            warn("eval_table_failed", err=err_str(e))
        al = agg.get("all", {})
        previews: List[str] = []
        for fn in (lambda: make_yak_preview(self.paths, ema, self.mult, self.device, self.step, self.amp),
                   lambda: [make_val_strip(self.paths, ema, self.mult, self.device, self.amp)]):
            try:
                got = fn()
                previews += [os.path.basename(x) for x in (got or []) if x]
            except Exception as e:  # noqa: BLE001 - previews never stop the run
                warn("preview_failed", err=err_str(e, 200))
        out.update({"val": agg, "previews": previews})
        marker = {"step": self.step, "val_med_absrel": (al.get("med") or {}).get("absrel"),
                  "val_med_d1": (al.get("med") or {}).get("d1"),
                  "in_absrel": ((agg.get("indoors") or {}).get("med") or {}).get("absrel"),
                  "out_absrel": ((agg.get("outdoor") or {}).get("med") or {}).get("absrel"),
                  "raw_absrel": (al.get("raw") or {}).get("absrel"), "ss_absrel": (al.get("ssd") or {}).get("absrel"),
                  "ssp_absrel": (al.get("ssp") or {}).get("absrel"),
                  "dav2_ssd_absrel": (((dav2 or {}).get("all") or {}).get("ssp") or {}).get("absrel"),
                  "const_med_absrel": (((const or {}).get("all") or {}).get("med") or {}).get("absrel"),
                  "dev_absrel": out["dev_absrel"], "n": len(recs),
                  "preview": f"{REMOTE_PREVIEW}/yak_step{self.step:06d}.png" if any(x.startswith("yak_step") for x in previews) else None}
        log(marker, "_EVAL")
        rec = {"step": self.step, "time": round(time.time(), 1), "final": final, "run_id": self.run_id, "loss": self._mean_loss(),
               "dev_absrel": out["dev_absrel"], "diode_dev": dev_d, "synth_dev": dev_s, "val": agg, "ci95": ci,
               "clamped": ex["clamped"]}
        try:
            append_jsonl(p.metrics, _round_floats(rec, 6))
        except Exception as e:  # noqa: BLE001
            warn("metrics_write_failed", err=err_str(e))
        improved = out["dev_absrel"] is not None and out["dev_absrel"] < self.best
        if improved:
            self.best = out["dev_absrel"]
            self._save_best(out["dev_absrel"])
        elif not os.path.isfile(p.ckpt_best) and out["dev_absrel"] is None:
            self._save_best(float("nan"))
        out["improved"] = improved
        self.last_eval = out
        free_cuda_cache()
        return out

    def _mean_loss(self) -> Optional[float]:
        return self._acc["loss"] / self._acc["n"] if self._acc["n"] else None

    # -- upload ----------------------------------------------------------------
    def stage_upload(self) -> Tuple[str, List[str], str, List[str]]:
        p = self.paths
        write_stages_json(p)
        st, pv = os.path.join(p.upload_stage, "ckpt"), os.path.join(p.upload_stage, "preview")
        shutil.rmtree(p.upload_stage, ignore_errors=True)
        os.makedirs(st, exist_ok=True)
        os.makedirs(pv, exist_ok=True)
        names = []
        for src in (p.ckpt_last, p.ckpt_best, p.basis, p.metrics, p.eval_json, p.eval_md, p.stages_json, p.source_diode):
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(st, os.path.basename(src)))
                names.append(os.path.basename(src))
        pnames = []
        if os.path.isdir(p.preview):
            for n in sorted(os.listdir(p.preview)):
                if n.endswith(".png"):
                    shutil.copyfile(os.path.join(p.preview, n), os.path.join(pv, n))
                    pnames.append(n)
        return st, names, pv, pnames

    def _upload_job(self, st: str, pv: str, has_prev: bool) -> None:
        sl = self.sleep
        ok = upload_with_retry(lambda: self.uploader.folder(st, check_remote(REMOTE_CKPT)), sleep=sl)
        if has_prev:
            ok = upload_with_retry(lambda: self.uploader.folder(pv, check_remote(REMOTE_PREVIEW)), sleep=sl) and ok
        if ok:
            log({"uploaded": REMOTE_CKPT, "step": self.step})

    def start_upload(self, final: bool) -> None:
        if self.uploader is None:
            return
        th = self.upload_thread
        if th is not None and th.is_alive():
            if not final:
                warn("upload_still_running_skip")
                return
            th.join(timeout=self.a.final_upload_timeout_s)
        try:
            st, _names, pv, pnames = self.stage_upload()
        except Exception as e:  # noqa: BLE001
            warn("stage_failed", err=err_str(e, 300))
            return
        self.upload_thread = threading.Thread(target=self._upload_job, args=(st, pv, bool(pnames)), daemon=True)
        self.upload_thread.start()

    def finish_upload(self) -> None:
        th = self.upload_thread
        if th is not None and th.is_alive():
            th.join(timeout=self.a.final_upload_timeout_s)
            if th.is_alive():
                warn("final_upload_timeout")

    # -- main loop -------------------------------------------------------------
    def log_progress(self, t: float, t_prev: float, steps_prev: int) -> None:
        n = max(self._acc["n"], 1)
        free = None
        with contextlib.suppress(Exception):
            if cuda_ready():
                free = gb(torch.cuda.mem_get_info()[0])
        log({"stage": "train", "step": self.step, "loss": self._acc["loss"] / n, "l_d": self._acc["l_d"] / n,
             "l_s": self._acc["l_s"] / n, "l_n": self._acc["l_n"] / n,
             "lr": lr_at(self.step, self.a.lr, self.a.warmup, self.a.total_steps),
             "sps": round((self.step - steps_prev) / max(t - t_prev, 1e-6), 2), "mb": self.mb["diode"] if self.run_id == RUN_MIX else self.mb["synth"],
             "vram_gb": vram_gb(), "free_gb": free, "skip": self.skipped})
        self._acc = {"loss": 0.0, "l_d": 0.0, "l_s": 0.0, "l_n": 0.0, "n": 0}

    def eval_due(self) -> bool:
        a = self.a
        return self.step == a.eval_first_step or (self.step > 0 and self.step % a.eval_every == 0)

    def eval_now(self) -> bool:
        """Once per step number, and never at the step a resumed run starts from."""
        if self.step == self._last_eval_step or not self.eval_due():
            return False
        self._last_eval_step = self.step
        return True

    def run(self) -> dict:
        a, p = self.a, self.paths
        p.ensure()
        clean_tmp(p)
        token = os.environ.get("HF_TOKEN")
        log({"start": True, "root": p.root, "hf_token": "yes" if token else "no", "deadline": self.deadline,
             "torch": torch.__version__, "stage": "train", "run": a.run})
        done = self.already_done()
        if done is not None:
            log({"status": "already_done", "step": done["step"]}, "_DONE")
            return {"status": "already_done", "step": int(done["step"])}
        plan = read_json(p.plan, {}) or {}
        self.budget_s = plan.get("train_budget_s")
        if not os.path.isfile(p.basis):
            raise FatalTrain("basis.pt missing")
        self.basis = T.PCABasis.load(p.basis)
        self.basis_fp = self.basis.fingerprint.hex()
        pr = promote_pending(p)
        fw = diode_firewall()
        log({"firewall": "ok" if fw["ok"] else "blocked_fnd04", **{k: v for k, v in pr.items() if k != "why"}})
        self._open_datasets()
        ck = self._resume()
        n_diode = len(self.ds["diode"])
        self.run_id = decide_run_id(fw["ok"], n_diode, (ck or {}).get("run_id"), (ck or {}).get("run"), a.run)
        if self.run_id == RUN_SYN:
            log({"diode_train": "blocked_fnd04" if not fw["ok"] else "no_shards", "run_id": self.run_id})
        if len(self.ds["synth"]) == 0:
            raise FatalTrain("no synthetic training shards")
        if self.run_id == RUN_MIX and n_diode == 0:
            raise FatalTrain("run is m3d1-mix but there are no DIODE train shards")
        self._setup_device()
        self._load_or_make_tap_std(ck)
        self.model = MH.MRH(self.mrh_cfg, seed=0).to(self.device)
        self.ema = EMA(self.model, a.ema_decay)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
        cfg = build_train_config(a, self.run_id, self.basis_fp, tap_std_hash(self.tap_std), self.mrh_cfg)
        self.cfg_hash = stable_hash(cfg)[:16]
        resumed = False
        if ck is not None:
            if ck["config_hash"] == self.cfg_hash:
                resumed = self._restore(ck)
            else:
                warn("ckpt_config_mismatch", ckpt=ck.get("run_id"), want=self.run_id)
                if a.warm_start and ck.get("run_id") == RUN_SYN and self.run_id == RUN_MIX:
                    resumed = self._restore(ck, warm=True)
                    log({"warm_start": True, "from": ck.get("run_id"), "to": self.run_id})
        n_par = count_params(self.model)
        log({"stage": "train_setup", "run_id": self.run_id, "params": n_par, "dev": str(self.device), "amp": self.amp,
             "resumed": bool(resumed), "step": self.step, "train": {d: len(s) for d, s in self.ds.items()},
             "dev_sets": {d: (len(s) if s else 0) for d, s in self.dev.items()}, "backend": self.backend,
             "cfg": self.cfg_hash})
        self._last_eval_step = self.step
        weights = DOMAIN_WEIGHTS[self.run_id]
        used = {d: self.ds[d] for d in weights}
        maker = BatchMaker(used, GLOBAL_BATCH, weights, a.seed)
        pf = OrderedPrefetcher(maker.make, self.data_k, a.prefetch_depth, a.prefetch_threads)
        t0 = self.now()
        t_log, steps_log = t0, self.step
        next_log = t0 + a.log_every_s
        t_ck, steps_ck = t0, self.step
        start_step = self.step
        reason: Optional[str] = None
        extra: dict = {}
        first_logged = False
        try:
            while True:
                reason = self.stop_reason()
                if reason:
                    break
                batch = pf.get(self.data_k, 5.0)
                if batch is _MISSING:
                    continue
                self.data_k += 1
                if batch is None:
                    self.fail_streak += 1
                    warn("empty_batch", n=self.fail_streak)
                else:
                    try:
                        ts = self.now()
                        okstep = self.train_step(batch)
                        self.fail_streak = 0 if okstep else self.fail_streak + 1
                        if okstep and not first_logged and self.step - start_step >= min(5, max(1, a.total_steps)):
                            first_logged = True
                            log({"first_steps": self.step - start_step, "max_gb": vram_gb(), "step_s": round(self.now() - ts, 3)})
                        if self.bad_run >= 25:
                            warn("nonfinite_streak", step=self.step)
                            ck2 = self._resume()
                            if ck2 is not None:
                                self._restore(ck2)
                            self.bad_run = 0
                    except FatalTrain:
                        raise
                    except Exception as e:  # noqa: BLE001
                        self.fail_streak += 1
                        warn("step_failed", step=self.step, n=self.fail_streak, err=err_str(e))
                        self.opt.zero_grad(set_to_none=True)
                        free_cuda_cache()
                        if self.fail_streak < a.max_consec_errors:
                            self.sleep(min(30.0, float(self.fail_streak)))
                if self.fail_streak >= a.max_consec_errors:
                    reason = "fatal"
                    extra["fatal"] = "too_many_failures"
                    break
                t = self.now()
                if t >= next_log:
                    self.log_progress(t, t_log, steps_log)
                    t_log, steps_log, next_log = t, self.step, t + a.log_every_s
                if self.step - steps_ck >= a.ckpt_every_steps or t - t_ck >= a.ckpt_every_s:
                    self.save_last()
                    t_ck, steps_ck = self.now(), self.step
                if self.eval_now():
                    self.save_last()
                    self.evaluate()
                    self.start_upload(False)
                    steps_ck, t_ck = self.step, self.now()
        finally:
            pf.close()
        if self._acc["n"]:
            self.log_progress(self.now(), t_log, steps_log)
        self.save_last()
        if self.step > start_step or not os.path.isfile(p.eval_json):
            try:
                self.evaluate(final=True)
            except Exception as e:  # noqa: BLE001 - the checkpoint is already safe
                warn("final_eval_failed", err=err_str(e, 300))
        self.start_upload(True)
        self.finish_upload()
        status = {"total_steps": "done", "max_steps": "max_steps", "deadline": "deadline", "budget": "deadline",
                  "signal": "signal", "fatal": "fatal"}.get(reason or "", "deadline")
        summary = {"status": status, "step": self.step, "start_step": start_step, "run_id": self.run_id,
                   "best": None if math.isinf(self.best) else round(self.best, 5), "skipped": self.skipped,
                   "oom": self.oom_events, "backend": self.backend, **extra}
        if status == "done":
            self.write_done({k: v for k, v in summary.items() if k != "status"})
        ev = self.last_eval.get("val", {}).get("all", {}) if self.last_eval else {}
        log({**summary, "val_med_absrel": (ev.get("med") or {}).get("absrel"), "val_med_d1": (ev.get("med") or {}).get("d1")}, "_DONE")
        for d in list(self.ds.values()) + [x for x in self.dev.values() if x]:
            d.close()
        return summary


# ---------------------------------------------------------------------------
# vault: restore on a fresh box (night2), deferred uploads, DIODE subset bundles
# ---------------------------------------------------------------------------

def _pull(fetcher: Callable, remote: str, dst: str) -> bool:
    try:
        src = fetcher(remote)
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
        with contextlib.suppress(OSError):
            if os.path.abspath(src) != os.path.abspath(dst):
                os.remove(src)
        return True
    except Exception as e:  # noqa: BLE001
        warn("restore_failed", remote=remote, err=err_str(e, 200))
        return False


def _has_tars(d: str) -> bool:
    try:
        return any(n.endswith(".tar") for n in os.listdir(d))
    except OSError:
        return False


def restore_missing(paths: Paths, lister: Optional[Callable], fetcher: Optional[Callable], *,
                    want_ckpt: bool = True) -> dict:
    """On a box without the local files (night2), pull basis, ckpt_last, shards and eval taps from the vault.
    Only what is missing locally is fetched."""
    got: Dict[str, int] = {}
    if lister is None or fetcher is None:
        return got
    if not os.path.isfile(paths.basis) and _pull(fetcher, f"{REMOTE_CKPT}/basis.pt", paths.basis):
        got["basis"] = 1
    if want_ckpt and not os.path.isfile(paths.ckpt_last) and _pull(fetcher, f"{REMOTE_CKPT}/ckpt_last.pt", paths.ckpt_last):
        got["ckpt_last"] = 1
    for role in ("train", "dev"):
        for dom in ("diode", "synth"):
            d = os.path.join(paths.shards, role, dom)
            if _has_tars(d):
                continue
            try:
                names = [n for n in lister(f"{REMOTE_TAPS}/{role}/{dom}/") if n.endswith((".tar", ".json"))]
            except Exception as e:  # noqa: BLE001
                warn("restore_list_failed", err=err_str(e))
                continue
            n = sum(1 for r in names if _pull(fetcher, r, os.path.join(d, os.path.basename(r))))
            if n:
                got[f"shards_{role}_{dom}"] = n
    if not (os.path.isdir(paths.val_taps) and os.listdir(paths.val_taps)):
        try:
            names = [n for n in lister(f"{REMOTE_EVAL_TAPS}/") if n.endswith(".npz")]
        except Exception as e:  # noqa: BLE001
            warn("restore_list_failed", err=err_str(e))
            names = []
        n = 0
        for r in names:
            base = os.path.basename(r)
            dst = paths.yak_taps if base == "yak_taps.npz" else os.path.join(paths.val_taps, base)
            n += int(_pull(fetcher, r, dst))
        if n:
            got["eval_taps"] = n
    return got


def bundle_diode_subset(paths: Paths, split: str, out_dir: str, target_bytes: int = 1_000_000_000) -> List[str]:
    """Pack the kept DIODE crops (png + gt.npz) of a split into ~1 GB tars so a later night does not stream the 87 GB
    again.  Shard 0 carries the licence and a MANIFEST.json."""
    os.makedirs(out_dir, exist_ok=True)
    samples = list_diode_samples(paths, split)
    out: List[str] = []
    tf = None
    size = 0
    idx = 0

    def open_next():
        nonlocal tf, size, idx
        if tf is not None:
            tf.close()
        pth = os.path.join(out_dir, f"{split}_{idx:03d}.tar")
        tf = tarfile.open(pth, "w")
        out.append(pth)
        size = 0
        if idx == 0:
            for name, data in (("LICENSE_DIODE.txt", diode_license_text().encode("utf-8")),
                               ("MANIFEST.json", json.dumps({"split": split, "n_samples": len(samples),
                                                             "sel_hash": (read_json(paths.source_diode, {}) or {}).get("sel_hash")}).encode())):
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                ti.mtime = 0
                tf.addfile(ti, io.BytesIO(data))
                size += len(data)
        idx += 1

    open_next()
    for m, base in samples:
        for ext in (".png", ".gt.npz"):
            fp = base + ext
            sz = os.path.getsize(fp)
            if size + sz > target_bytes and size > 0:
                open_next()
            ti = tarfile.TarInfo(f"{split}/{m.rel_dir.replace(os.sep, '/')}/{m.stem}{ext}")
            ti.size = sz
            ti.mtime = 0
            with open(fp, "rb") as f:
                tf.addfile(ti, f)
            size += sz
    if tf is not None:
        tf.close()
    return out


def restore_diode_subset(paths: Paths, split: str, lister: Callable, fetcher: Callable) -> int:
    """Unpack ``corpus/day1/diode/<split>_NNN.tar`` bundles into data/diode/<split> (only member names that
    ``parse_member`` style checks accept).  Returns the number of ``.gt.npz`` restored."""
    root = paths.diode_train if split == "train" else paths.diode_val
    names = sorted(n for n in lister(f"{REMOTE_DIODE}/{split}_") if re.fullmatch(rf"{re.escape(REMOTE_DIODE)}/{split}_\d{{3}}\.tar", n))
    n_gt = 0
    pat = re.compile(rf"{split}/(?:indoors|outdoor)/scene_\d{{5}}/scan_\d{{5}}/\d{{5}}_\d{{5}}_(?:indoors|outdoor)_\d{{3}}_\d{{3}}\.(?:png|gt\.npz)")
    for r in names:
        tmp = os.path.join(paths.out, f"_restore_{os.path.basename(r)}")
        if not _pull(fetcher, r, tmp):
            continue
        with tarfile.open(tmp, "r") as tf:
            for ti in tf:
                if ti.name == "LICENSE_DIODE.txt":
                    dst = os.path.join(paths.diode, "LICENSE_DIODE.txt")
                elif pat.fullmatch(ti.name):
                    dst = os.path.join(paths.diode, ti.name)
                else:
                    continue
                if ti.isreg() and not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    write_atomic(dst, tf.extractfile(ti).read())
                    n_gt += int(ti.name.endswith(".gt.npz"))
        with contextlib.suppress(OSError):
            os.remove(tmp)
    return n_gt


def folder_signature(d: str) -> str:
    try:
        items = sorted((n, os.path.getsize(os.path.join(d, n))) for n in os.listdir(d) if not n.startswith("."))
    except OSError:
        items = []
    return stable_hash(items)[:16]


def run_upload_stage(paths: Paths, args, deps: Optional[dict] = None) -> dict:
    """Lowest priority, after the teacher stage: shards, eval taps (private vault only) and DIODE subset bundles."""
    deps = deps or {}
    up = deps.get("uploader") or make_uploader(os.environ.get("HF_TOKEN"), args.repo_id)
    if up is None or args.no_upload:
        return {"status": "no_token"}
    sleep = deps.get("sleep", time.sleep)
    done = {}
    tasks: List[Tuple[str, str, str]] = []
    for role in ("train", "dev"):
        for dom in ("diode", "synth"):
            d = os.path.join(paths.shards, role, dom)
            if _has_tars(d):
                tasks.append((f"shards_{role}_{dom}", d, f"{REMOTE_TAPS}/{role}/{dom}"))
    if os.path.isdir(paths.val_taps) and os.listdir(paths.val_taps):
        tasks.append(("eval_taps", paths.val_taps, REMOTE_EVAL_TAPS))
    for name, d, remote in tasks:
        sig = folder_signature(d)
        if stage_done(paths, f"upload_{name}", sig):
            continue
        ok = upload_with_retry(lambda d=d, remote=remote: up.folder(d, check_remote(remote)), sleep=sleep)
        done[name] = ok
        if ok:
            write_stage_done(paths, f"upload_{name}", sig, {"remote": remote})
            log({"uploaded": remote})
    if os.path.isfile(paths.yak_taps) and not stage_done(paths, "upload_yak_taps", folder_signature(paths.private)):
        sig = folder_signature(paths.private)
        if upload_with_retry(lambda: up.files([(paths.yak_taps, check_remote(f"{REMOTE_EVAL_TAPS}/yak_taps.npz"))]), sleep=sleep):
            write_stage_done(paths, "upload_yak_taps", sig, {})
    for split in ("val", "train"):
        if not stage_done(paths, f"diode_{split}") or stage_done(paths, f"upload_diode_{split}"):
            continue
        bdir = os.path.join(paths.out, "bundles")
        tars = bundle_diode_subset(paths, split, bdir)
        ok = True
        for t in tars:
            ok = upload_with_retry(lambda t=t: up.files([(t, check_remote(f"{REMOTE_DIODE}/{os.path.basename(t)}"))]), sleep=sleep) and ok
            with contextlib.suppress(OSError):
                os.remove(t)
        if ok:
            write_stage_done(paths, f"upload_diode_{split}", "bundle", {"tars": len(tars)})
            log({"uploaded": f"{REMOTE_DIODE}/{split}", "tars": len(tars)})
        done[f"diode_{split}"] = ok
    return {"status": "done", **done}


# ---------------------------------------------------------------------------
# supervisor (--stage all)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Day1 job 4: M3 MRH spike")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", DEFAULT_ROOT))
    ap.add_argument("--out", default=None, help="default: <root>/day1/m3")
    ap.add_argument("--stage", default="all", choices=["all", "diode", "render", "dav2", "teacher", "train", "upload"])
    ap.add_argument("--run", default="day1", help="run name (night2 continues the work of day1)")
    ap.add_argument("--marker-prefix", default=MARK_DEFAULT)
    ap.add_argument("--deadline", default=os.environ.get("N1_DEADLINE"), help="unix seconds (default: env N1_DEADLINE)")
    ap.add_argument("--finish-margin-s", type=float, default=600.0)
    ap.add_argument("--refit-basis", action="store_true", help="re-fit the frozen PCA basis (invalidates shards / ckpts)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="never restore from the vault")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log-every-s", type=float, default=240.0)
    ap.add_argument("--ckpt-every-steps", type=int, default=500)
    ap.add_argument("--ckpt-every-s", type=float, default=600.0)
    ap.add_argument("--eval-first-step", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--prefetch-depth", type=int, default=8)
    ap.add_argument("--prefetch-threads", type=int, default=4)
    ap.add_argument("--max-consec-errors", type=int, default=100)
    ap.add_argument("--final-upload-timeout-s", type=float, default=240.0)
    ap.add_argument("--gate-gb", type=float, default=GATE_BYTES / 1e9)
    ap.add_argument("--teacher-vram-gib", type=float, default=TEACHER_CAP_GIB)
    ap.add_argument("--train-vram-gib", type=float, default=TRAIN_CAP_GIB)
    ap.add_argument("--dav2-vram-gib", type=float, default=DAV2_CAP_GIB)
    ap.add_argument("--pca-k", type=int, default=PCA_K)
    ap.add_argument("--basis-corpus", type=int, default=BASIS_N_CORPUS)
    ap.add_argument("--basis-render", type=int, default=BASIS_N_RENDER)
    ap.add_argument("--basis-tokens", type=int, default=TOKENS_PER_IMG)
    ap.add_argument("--procs", type=int, default=6, help="render processes")
    ap.add_argument("--wait-diode-s", type=float, default=7200.0)
    ap.add_argument("--wait-render-s", type=float, default=7200.0)
    ap.add_argument("--upload-wait-s", type=float, default=3600.0)
    ap.add_argument("--max-child-restarts", type=int, default=3)
    return ap


def child_argv(args, stage: str, parser: Optional[argparse.ArgumentParser] = None) -> List[str]:
    """Command line of a sub-process: every option that differs from its default, with ``--stage`` replaced."""
    parser = parser or build_parser()
    out = [sys.executable, "-u", os.path.abspath(__file__)]
    for act in parser._actions:
        if not act.option_strings or act.dest in ("help", "stage"):
            continue
        val = getattr(args, act.dest, act.default)
        if val == act.default:
            continue
        flag = act.option_strings[0]
        if isinstance(act, argparse.BooleanOptionalAction):
            out.append(flag if val else act.option_strings[1])
        elif isinstance(act, argparse._StoreTrueAction):
            if val:
                out.append(flag)
        elif val is not None:
            out += [flag, str(val)]
    out += ["--stage", stage]
    return out


def default_spawn(argv: List[str], env: Optional[dict] = None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.Popen(argv, env=e, stdin=subprocess.DEVNULL)


def gpu_free_after_exit() -> Optional[float]:
    """Free VRAM in bytes via nvidia-smi (the supervisor never creates a CUDA context itself)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20)
        return float(r.stdout.strip().splitlines()[0]) * 2 ** 20
    except Exception:  # noqa: BLE001
        return None


CPU_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


class Supervisor:
    """Starts and watches the CPU sub-processes (diode, render, dav2) and the GPU ones (teacher, train)."""

    def __init__(self, args, paths: Paths, deps: dict):
        self.a, self.paths, self.deps = args, paths, deps
        self.spawn = deps.get("spawn", default_spawn)
        self.now = deps.get("now", time.time)
        self.sleep = deps.get("sleep", time.sleep)
        self.procs: Dict[str, object] = {}
        self.restarts: Dict[str, int] = {}
        self.rcs: Dict[str, int] = {}
        self.parser = build_parser()

    def start(self, name: str) -> None:
        env = CPU_ENV if name in ("diode", "render", "upload") else None
        argv = child_argv(self.a, name, self.parser)
        self.procs[name] = self.spawn(argv, env)
        log({"spawn": name, "restarts": self.restarts.get(name, 0)})

    def running(self, name: str) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def reap(self, name: str) -> Optional[int]:
        p = self.procs.get(name)
        if p is None:
            return None
        rc = p.poll()
        if rc is not None:
            self.rcs[name] = rc
            self.procs[name] = None
        return rc

    def cpu_done(self, name: str) -> bool:
        p = self.paths
        if name == "diode":
            return stage_done(p, "diode_val") and stage_done(p, "diode_train")
        if name == "render":
            return stage_done(p, "render")
        if name == "dav2":
            return stage_done(p, "dav2")
        return False

    def maintain_cpu(self) -> None:
        """(Re)start the CPU sub-processes whose stage is not finished; dav2 waits for DIODE val."""
        for name in ("diode", "render", "dav2"):
            if self.running(name):
                continue
            rc = self.reap(name)
            if self.cpu_done(name):
                continue
            if name == "dav2" and not stage_done(self.paths, "diode_val"):
                continue
            if rc is not None and self.restarts.get(name, 0) >= self.a.max_child_restarts:
                continue
            if rc is not None:
                self.restarts[name] = self.restarts.get(name, 0) + 1
                warn("child_exit", child=name, rc=rc)
                self.sleep(5.0)
            self.start(name)

    def terminate_all(self, wait_s: float = 60.0) -> None:
        for n, p in list(self.procs.items()):
            if p is not None and p.poll() is None:
                with contextlib.suppress(Exception):
                    p.terminate()
        t0 = self.now()
        for n, p in list(self.procs.items()):
            if p is None:
                continue
            while p.poll() is None and self.now() - t0 < wait_s:
                self.sleep(1.0)
            if p.poll() is None:
                with contextlib.suppress(Exception):
                    p.kill()


def verify_plan_prereqs(paths: Paths) -> bool:
    """Training needs the frozen basis and synthetic train shards."""
    return os.path.isfile(paths.basis) and _has_tars(os.path.join(paths.shards_train, "synth"))


def run_all(args, deps: Optional[dict] = None) -> int:
    """The launcher: P0 checks, CPU sub-processes, the GPU gate / teacher sub-process, then training."""
    deps = deps or {}
    paths = Paths(args.root, args.out)
    paths.ensure()
    now, sleep = deps.get("now", time.time), deps.get("sleep", time.sleep)
    deadline = parse_deadline(args.deadline)
    stop = deps.get("stop_event") or threading.Event()
    token = os.environ.get("HF_TOKEN")
    cleaned = clean_tmp(paths)
    fw = diode_firewall()
    free = free_disk_gb(paths.out)
    log({"start": True, "root": paths.root, "hf_token": "yes" if token else "no", "deadline": deadline,
         "torch": torch.__version__, "stage": "p0", "run": args.run, "disk_free_gb": round(free, 1),
         "cleaned": cleaned, "fw": "ok" if fw["ok"] else "blocked_fnd04", "blake3": "shim" if BLAKE3_SHIM else "real"})
    if not args.no_fetch and token:
        lister, fetcher = deps.get("fetchers") or make_hf_fetchers(token, paths.root, args.repo_id)
        got = restore_missing(paths, lister, fetcher)
        if got:
            log({"restored": got})
    d = read_json(paths.done)
    if isinstance(d, dict) and d.get("status") == "done" and d.get("run") == args.run \
            and int(d.get("step", -1)) >= args.total_steps and int(d.get("total_steps", -1)) == args.total_steps:
        log({"status": "already_done", "step": d.get("step")}, "_DONE")
        return 0
    if deadline is not None and deadline - now() < 1.25 * 3600 and not os.path.isfile(paths.ckpt_last):
        log({"status": "deadline", "stage": "gpu_wait"}, "_DONE")
        return 0
    if disk_policy(free) == "stop":
        log({"status": "fatal", "stage": "p0", "err": "disk below 8 GB"}, "_DONE")
        return 2
    sup = Supervisor(args, paths, deps)

    def on_signal(signum, _frame=None):
        log({"signal": int(signum)})
        stop.set()

    if threading.current_thread() is threading.main_thread() and "now" not in deps:
        for sg in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sg, on_signal)

    last_hb = now()

    def heartbeat() -> None:
        nonlocal last_hb
        t = now()
        if t - last_hb >= args.log_every_s:
            last_hb = t
            log({"stage": "supervisor", "alive": [n for n in sup.procs if sup.running(n)], "rcs": sup.rcs,
                 "restarts": sup.restarts})

    # ---- GPU extraction sub-process (it waits for the VRAM gate itself)
    teacher_rc: Optional[int] = None
    attempts = 0
    if teacher_needed(paths, args):
        sup.start("teacher")
        while True:
            sup.maintain_cpu()
            heartbeat()
            if stop.is_set():
                sup.terminate_all()
                log({"status": "signal", "step": 0}, "_DONE")
                return 0
            rc = sup.reap("teacher")
            if rc is None:
                if deadline is not None and now() >= deadline - 300:
                    sup.terminate_all()
                    log({"status": "deadline", "stage": "teacher"}, "_DONE")
                    return 0
                sleep(5.0)
                continue
            if rc == RC_GATE_DEADLINE:
                sup.terminate_all()
                return 0                          # the child already printed D1_M3_DONE (deadline, gpu_wait)
            if rc == 2:
                sup.terminate_all()
                return 2
            free_after = gpu_free_after_exit()
            log({"teacher": "exited", "rc": rc, "free_gb": None if free_after is None else gb(free_after)})
            if rc == 0 and not teacher_needed(paths, args):
                teacher_rc = 0
                break
            if deadline is not None and now() >= deadline - 1800:
                teacher_rc = rc
                break
            attempts += 1
            if attempts > args.max_child_restarts:
                teacher_rc = rc
                break
            sleep(30.0)
            sup.start("teacher")
    else:
        teacher_rc = 0
    # ---- training
    if not verify_plan_prereqs(paths) or (deadline is not None and now() >= deadline - 900):
        sup.terminate_all()
        log({"status": "deadline" if deadline is not None and now() >= deadline - 900 else "fatal", "stage": "pre_train",
             "teacher_rc": teacher_rc}, "_DONE")
        return 0 if deadline is not None and now() >= deadline - 900 else 2
    sup.start("train")
    rc = None
    while True:
        sup.maintain_cpu()
        heartbeat()
        rc = sup.reap("train")
        if rc is not None:
            break
        if stop.is_set():
            with contextlib.suppress(Exception):
                sup.procs["train"].terminate()
            stop_wait_until = now() + 300
            while sup.running("train") and now() < stop_wait_until:
                sleep(2.0)
            rc = sup.reap("train")
            if rc is None:
                sup.terminate_all()
                rc = 0
            break
        sleep(5.0)
    # ---- shards / eval taps / DIODE bundles go to the vault in the background of the remaining time
    if rc in (0, None) and not args.no_upload and token:
        sup.start("upload")
        t0 = now()
        while sup.running("upload") and now() - t0 < args.upload_wait_s and not (deadline and now() > deadline - 300):
            sleep(5.0)
    sup.terminate_all(30.0)
    write_stages_json(paths)
    return 0 if rc in (0, None) else int(rc)


# ---------------------------------------------------------------------------
# stage entry points (sub-processes) and CLI
# ---------------------------------------------------------------------------

def _install_stop(stop: threading.Event) -> None:
    def handler(signum, _frame=None):
        log({"signal": int(signum)})
        stop.set()
    if threading.current_thread() is threading.main_thread():
        for sg in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sg, handler)


def stage_diode(args, paths: Paths, deps: dict) -> int:
    if not claim_pid(paths, "diode", cmdline_fn=deps.get("cmdline_fn", _proc_cmdline)):
        log({"dup": "diode"})
        return 0
    stop = deps.get("stop_event") or threading.Event()
    _install_stop(stop)
    with contextlib.suppress(Exception):
        os.nice(10)
    try:
        if os.environ.get("HF_TOKEN") and not args.no_fetch and "raw_factory" not in deps:
            lister, fetcher = deps.get("fetchers") or make_hf_fetchers(os.environ.get("HF_TOKEN"), paths.root, args.repo_id)
            sel = choose_selection(paths)
            for split in ("val", "train"):
                if not stage_done(paths, f"diode_{split}") and not list_diode_samples(paths, split) and lister is not None:
                    n = restore_diode_subset(paths, split, lister, fetcher)
                    if n:
                        log({"restored_diode": split, "gt": n})
        ok = True
        for split in ("val", "train"):
            if stop.is_set():
                return 1
            res = run_diode_split(paths, split, url=(deps.get("urls") or {}).get(split), raw_factory=deps.get("raw_factory"),
                                  expected_png=(deps.get("expected_png") or {}).get(split), stop=stop.is_set,
                                  now=deps.get("now", time.time), sleep=deps.get("sleep", time.sleep))
            ok = ok and res.get("status") == "done"
        return 0 if ok else 1
    except DiskLowError as e:
        log({"status": "fatal", "stage": "diode", "err": err_str(e)}, "_DONE")
        return 1
    finally:
        release_pid(paths, "diode")


def stage_render(args, paths: Paths, deps: dict) -> int:
    if not claim_pid(paths, "render", cmdline_fn=deps.get("cmdline_fn", _proc_cmdline)):
        log({"dup": "render"})
        return 0
    stop = deps.get("stop_event") or threading.Event()
    _install_stop(stop)
    try:
        res = run_render(paths, procs=args.procs, stop=stop.is_set, now=deps.get("now", time.time),
                         **{k: deps[k] for k in ("train_range", "dev_range", "wh") if k in deps})
        return 0 if res.get("status") == "done" else 1
    finally:
        release_pid(paths, "render")


def stage_dav2(args, paths: Paths, deps: dict) -> int:
    if not claim_pid(paths, "dav2", cmdline_fn=deps.get("cmdline_fn", _proc_cmdline)):
        log({"dup": "dav2"})
        return 0
    try:
        res = run_dav2_stage(paths, args, deps)
        return 0 if res.get("status") == "done" else 1
    finally:
        release_pid(paths, "dav2")


def stage_train(args, paths: Paths, deps: dict) -> int:
    if not claim_pid(paths, "train", cmdline_fn=deps.get("cmdline_fn", _proc_cmdline)):
        log({"dup": "train"})
        return 0
    stop = deps.get("stop_event") or threading.Event()
    _install_stop(stop)
    d = dict(deps)
    d["stop_event"] = stop
    token = os.environ.get("HF_TOKEN")
    uploader = d.get("uploader") if "uploader" in d else (None if args.no_upload else make_uploader(token, args.repo_id))
    tr = M3Trainer(args, paths, d, uploader)
    try:
        try:
            res = tr.run()
        except (FatalTrain, SmokeError, T.TapBasisError, ShardDataError) as e:
            log({"status": "fatal", "stage": "train", "err": err_str(e, 250)}, "_DONE")
            return 2
        return STATUS_RC.get(res.get("status", "done"), 0)
    finally:
        release_pid(paths, "train")


def main(argv: Optional[Sequence[str]] = None, deps: Optional[dict] = None) -> int:
    args = build_parser().parse_args(argv)
    deps = deps or {}
    set_marker_prefix(args.marker_prefix)
    setup_paths(args.root)
    paths = Paths(args.root, args.out)
    paths.ensure()
    handlers = {"all": lambda: run_all(args, deps), "diode": lambda: stage_diode(args, paths, deps),
                "render": lambda: stage_render(args, paths, deps), "dav2": lambda: stage_dav2(args, paths, deps),
                "teacher": lambda: run_teacher_stage(paths, args, deps), "train": lambda: stage_train(args, paths, deps),
                "upload": lambda: (run_upload_stage(paths, args, deps), 0)[1]}
    try:
        return int(handlers[args.stage]())
    except Exception as e:  # noqa: BLE001 - last resort: say why, never print a secret
        import traceback
        print(scrub(traceback.format_exc())[-1500:], file=sys.stderr, flush=True)
        log({"status": "fatal", "stage": args.stage, "err": err_str(e, 250)}, "_DONE")
        return 2 if args.stage in ("all", "train", "teacher") else 1


if __name__ == "__main__":
    sys.exit(main())
