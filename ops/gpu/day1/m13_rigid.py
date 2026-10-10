#!/usr/bin/env python3
"""Day1 job 1: M13 Rig-ID-S (which of the 14 camera rigs shot this pose sequence?).

Run on the GPU box (torch 2.9.1 + CUDA), CPU cores for the synthetic data, then a short GPU training:

    PYTHONPATH=$N1_ROOT/repo HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=1791723600 \
        python3 m13_rigid.py

Data (all synthetic, deterministic per sample index, cached so a rerun never rebuilds it):
    for each of the 14 rigs of ``buc.camera.rigs``: random intent (2-5 waypoints, 3-10 s) and random
    parameters around ``default_params`` -> ``simulate`` -> ``frame_poses(24)`` c2w poses -> structure-
    from-motion style noise (unknown global scale, rotation/translation jitter, slow OU drift, 5 % missing
    frames interpolated, start pose = origin) -> 240-frame features (relative rot6d, relative translation,
    cumulative path) + padding mask.
Outputs under $N1_ROOT/day1/m13/:
    cache/<key>/{train,val}.npz   merged cache (cache/<key>/<split>/unit_*.npz are the resumable pieces)
    ckpt_last.pt, ckpt_best.pt    atomic (tmp file, then rename)
    metrics.jsonl                 one line per epoch: epoch, step, loss, val_acc, top2, ...
    confusion.png / confusion.json  14x14 validation confusion matrix (rig names on both axes)
and uploads them to the HF vault under ckpt/day1/m13/.

Markers on stdout (one JSON per line, <= 400 chars, flushed): D1_M13, D1_M13_DONE.
Run it again after any crash: data units already built are skipped, training resumes from ckpt_last.pt.
The HF token comes from the environment only and is never printed or stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MARK = "D1_M13"
REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
REPO_TYPE = "model"
REMOTE_DIR = "ckpt/day1/m13"
MARKER_LIMIT = 400                 # vastai logs cuts a line at ~489 chars

RIG_NAMES_FALLBACK: Tuple[str, ...] = (
    "tripod_fluid", "dolly", "slider", "crane_jib", "techno_crane", "steadicam", "gimbal_3axis",
    "shoulder_handheld", "cine_drone", "fpv", "cablecam", "car_mount", "motion_control", "underwater")
N_RIGS = 14
FPS = 24
MAX_LEN = 240                      # frames fed to the model (longer sequences are cut, shorter padded)
FEAT_DIM = 10                      # rot6d(rel rotation) 6 + rel translation 3 + cumulative path 1
ROT_SCALE = 20.0                   # per-frame relative rot6d is ~0.01-0.05 -> bring it to O(0.1-1)
UNIT = 200                         # samples per cache unit (resumable work item)
MAX_ATTEMPTS = 4                   # re-draws of a sample whose simulation fails or diverges
MAX_CONSEC_ERRORS = 100            # consecutive failed training steps -> stop
MEM_CAP_GB = 3.0                   # per-process VRAM ceiling: the GPU is shared with M6 (Z-Image) and M2
PARAM_MIN = 2_000_000
PARAM_MAX = 4_000_000
CACHE_VERSION = 1
SPLIT_TAG = {"train": 1, "val": 2}  # seed streams of the two splits never overlap
TRAIN_SEED_BASE = 0
_UNSET = object()

# random-parameter multipliers (log-uniform in [lo, hi]); the normalised position in [0, 1] is the
# auxiliary regression target. "gain" scales kp/ki/kd together, "duration" is the sequence length.
AUX_RANGES: Dict[str, Tuple[float, float]] = {
    "mass": (0.6, 1.7), "damping": (0.6, 1.7), "quad_drag": (0.6, 1.7), "stiffness": (0.6, 1.7),
    "coulomb": (0.6, 1.7), "w2": (0.7, 1.4), "ou_sigma": (0.6, 1.7), "delay_s": (0.7, 1.4),
    "gain": (0.7, 1.4)}
AUX_KEYS: Tuple[str, ...] = tuple(AUX_RANGES) + ("duration",)
N_AUX = len(AUX_KEYS)
DUR_MIN, DUR_MAX = 3.0, 10.0

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


def format_marker(payload: dict, marker: str = MARK, limit: int = MARKER_LIMIT) -> str:
    """``MARKER {json}`` that never exceeds ``limit`` characters (trailing keys are dropped first)."""
    d = dict(payload)

    def line(x: dict) -> str:
        return scrub(f"{marker} {json.dumps(x, ensure_ascii=False, default=str, separators=(',', ':'))}")

    s = line(d)
    if len(s) <= limit:
        return s
    d["trunc"] = 1
    while len(s) > limit and len(d) > 2:
        keys = [k for k in d if k != "trunc"]
        d.pop(keys[-1])
        s = line(d)
    if len(s) > limit:
        s = f"{marker} " + json.dumps({"trunc": 1}, separators=(",", ":"))
    return s


def log(payload: dict, marker: str = MARK) -> None:
    """One JSON line on stdout, flushed (vastai logs is the only window)."""
    print(format_marker(payload, marker), flush=True)


def warn(what: str, **kw) -> None:
    d = {"warn": what}
    for k, v in kw.items():
        d[k] = scrub(v)[:160] if isinstance(v, str) else v
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


def rnd(x, n: int = 4):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


# ---------------------------------------------------------------------------
# buc.camera.rigs (read-only import; the runner puts the repo tarball on PYTHONPATH)
# ---------------------------------------------------------------------------

def _rigs():
    try:
        from buc.camera import rigs
    except ImportError:
        root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        if root not in sys.path:
            sys.path.append(root)
        from buc.camera import rigs
    return rigs


def rig_names() -> Tuple[str, ...]:
    try:
        names = tuple(_rigs().RIG_NAMES)
    except Exception:  # noqa: BLE001 - names are only labels for the confusion plot
        names = RIG_NAMES_FALLBACK
    return names if len(names) == N_RIGS else RIG_NAMES_FALLBACK


# ---------------------------------------------------------------------------
# synthetic sample: intent, parameters, geometry
# ---------------------------------------------------------------------------

def sample_rng(split: str, rig_idx: int, k: int, attempt: int = 0,
               base_seed: int = TRAIN_SEED_BASE) -> np.random.Generator:
    """Independent stream per (seed, split, rig, k, attempt): train/val streams are disjoint."""
    return np.random.default_rng(np.random.SeedSequence(
        [int(base_seed), 0x13D1, SPLIT_TAG[split], int(rig_idx), int(k), int(attempt)]))


def make_intent(q_scale: np.ndarray, rng: np.random.Generator):
    """2-5 waypoints, total length 3-10 s (waypoint times plus a short rest tail)."""
    rigs = _rigs()
    n_q = len(q_scale)
    k = int(rng.integers(2, 6))
    total = float(rng.uniform(DUR_MIN, DUR_MAX))
    tail = float(rng.uniform(0.0, min(0.4, 0.1 * total)))
    body = total - tail
    seg_min = 0.6
    seg = seg_min + (body - seg_min * (k - 1)) * rng.dirichlet(np.ones(k - 1))
    times = np.concatenate([[0.0], np.cumsum(seg)])
    amp = float(rng.uniform(0.3, 1.0))
    wp = np.zeros((k, n_q))
    wp[0] = q_scale * rng.uniform(-0.3, 0.3, n_q)
    moved = False
    for i in range(1, k):
        active = rng.random(n_q) < 0.75
        target = q_scale * amp * rng.uniform(-1.0, 1.0, n_q)
        wp[i] = np.where(active, target, wp[i - 1])
        moved = moved or bool(np.any(wp[i] != wp[i - 1]))
    if not moved:
        wp[-1, 0] = wp[0, 0] + q_scale[0] * amp * (1.0 if rng.random() < 0.5 else -1.0)
    return rigs.Intent(times, wp, tail_s=tail), total


def draw_params(name: str, rng: np.random.Generator, shrink: float = 1.0):
    """Random parameters around ``default_params``.

    Returns (params for make_rig, aux (N_AUX,) in [0, 1] with the duration slot left 0, mask (N_AUX,) bool).
    ``shrink`` < 1 pulls every multiplier toward 1 (used when a draw made the simulation diverge).
    """
    base = _rigs().default_params(name)
    u = rng.random(len(AUX_RANGES))            # always drawn, so the rng stream does not depend on the rig
    aux = np.zeros(N_AUX, dtype=np.float64)
    mask = np.zeros(N_AUX, dtype=bool)
    params: dict = {}
    for j, (key, (lo, hi)) in enumerate(AUX_RANGES.items()):
        ue = 0.5 + (float(u[j]) - 0.5) * shrink
        mult = math.exp(math.log(lo) + ue * (math.log(hi) - math.log(lo)))
        if key == "gain":
            targets, present = ("kp", "ki", "kd"), True
        elif key == "delay_s":
            targets, present = (), float(base["delay_s"]) > 0.0
            if present:
                params["delay_s"] = float(base["delay_s"]) * mult
        else:
            targets, present = (key,), any(float(v) != 0.0 for v in base[key])
        if present:
            for t in targets:
                params[t] = tuple(float(v) * mult for v in base[t])
        aux[j], mask[j] = ue, present
    return params, aux, mask


def draw_geom(name: str, rng: np.random.Generator) -> Optional[dict]:
    """Crane radius 2-5 m; slight tilt of the translation axis for the rail-like rigs."""
    r1 = float(rng.uniform(2.0, 5.0))
    jitter = rng.normal(0.0, 0.08, 3)
    if name in ("crane_jib", "techno_crane"):
        return {"radius": r1}
    if name in ("slider", "dolly", "cablecam", "car_mount"):
        axis = np.asarray(_rigs().make_rig(name).geom["axis"], dtype=np.float64) + jitter
        return {"axis": tuple(float(v) for v in axis)}
    return None


# ---------------------------------------------------------------------------
# structure-from-motion style noise, normalisation, features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseCfg:
    rot_deg: Tuple[float, float] = (0.1, 0.5)          # per-axis rotation jitter / drift std, degrees
    trans_frac: Tuple[float, float] = (0.005, 0.02)    # translation error as a fraction of the path length
    jitter_ratio: float = 0.2                          # iid translation jitter std = ratio * trans_frac * path
    drift_tau_s: Tuple[float, float] = (1.0, 4.0)      # OU time constant of the slow drift
    missing_frac: float = 0.05                         # frames lost by the extractor (re-interpolated)
    min_ref_len: float = 0.05                          # path-length floor (m) for pure rotations
    fps: int = FPS


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of ``R (...,3,3)`` flattened column after column -> (...,6)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


_ID6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def normalize_poses(poses: np.ndarray) -> np.ndarray:
    """Start pose -> origin (expressed in the first camera frame); translations / path length (unknown scale)."""
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim != 3 or P.shape[1:] != (4, 4) or P.shape[0] < 2:
        raise ValueError(f"poses must be (M>=2,4,4), got {P.shape}")
    R0, p0 = P[0, :3, :3], P[0, :3, 3]
    R = np.einsum("ji,tjk->tik", R0, P[:, :3, :3])
    p = (P[:, :3, 3] - p0) @ R0
    p = p / max(path_length(p), 1e-9)
    out = np.tile(np.eye(4), (P.shape[0], 1, 1))
    out[:, :3, :3], out[:, :3, 3] = R, p
    return out


def interpolate_missing(R: np.ndarray, p: np.ndarray, missing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the frames flagged in ``missing`` (never the first/last) by slerp / linear interpolation."""
    from scipy.spatial.transform import Rotation, Slerp
    if not missing.any():
        return R, p
    t = np.arange(len(R), dtype=np.float64)
    ok = ~missing
    R2, p2 = R.copy(), p.copy()
    slerp = Slerp(t[ok], Rotation.from_matrix(R[ok]))
    R2[missing] = slerp(t[missing]).as_matrix()
    for c in range(3):
        p2[missing, c] = np.interp(t[missing], t[ok], p[ok, c])
    return R2, p2


def add_sfm_noise(poses: np.ndarray, rng: np.random.Generator, cfg: NoiseCfg = NoiseCfg()) -> np.ndarray:
    """Imitate a structure-from-motion extractor on true c2w poses ``(M,4,4)``.

    iid rotation jitter + slow OU rotation drift, iid translation jitter + slow OU translation drift
    (scaled by the path length, floored so pure rotations still get a tiny noisy "path"), ``missing_frac``
    of the frames re-interpolated, then start pose -> origin and translations / path length.
    """
    from scipy.spatial.transform import Rotation
    ou_noise = _rigs().ou_noise
    P = np.asarray(poses, dtype=np.float64)
    M = P.shape[0]
    R, p = P[:, :3, :3], P[:, :3, 3]
    ref = max(path_length(p), cfg.min_ref_len)
    sig_rot = math.radians(float(rng.uniform(*cfg.rot_deg)))
    frac = float(rng.uniform(*cfg.trans_frac))
    tau = float(rng.uniform(*cfg.drift_tau_s))
    dt = 1.0 / cfg.fps
    rot_eps = rng.normal(0.0, sig_rot, (M, 3))
    t_eps = rng.normal(0.0, frac * ref * cfg.jitter_ratio, (M, 3))
    for ax in range(3):
        rot_eps[:, ax] += ou_noise(M, dt, tau, sig_rot, rng)
        t_eps[:, ax] += ou_noise(M, dt, tau, frac * ref, rng)
    Rn = (Rotation.from_matrix(R) * Rotation.from_rotvec(rot_eps)).as_matrix()
    pn = p + t_eps
    missing = np.zeros(M, dtype=bool)
    n_miss = int(round(cfg.missing_frac * M))
    if M >= 4 and n_miss > 0:
        missing[rng.choice(np.arange(1, M - 1), size=min(n_miss, M - 2), replace=False)] = True
    Rn, pn = interpolate_missing(Rn, pn, missing)
    out = np.tile(np.eye(4), (M, 1, 1))
    out[:, :3, :3], out[:, :3, 3] = Rn, pn
    return normalize_poses(out)


def pose_features(norm_poses: np.ndarray, max_len: int = MAX_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised poses ``(M,4,4)`` -> (feat (max_len, FEAT_DIM) float32, mask (max_len,) bool).

    per frame: rot6d of the relative rotation R_{t-1}^T R_t minus identity (x ROT_SCALE), relative
    translation R_{t-1}^T (p_t - p_{t-1}) x (M-1) (mean step norm 1), cumulative path (ends at 1).
    Frame 0 is all zeros. Longer than ``max_len`` is cut, shorter is zero-padded with mask False.
    """
    P = np.asarray(norm_poses, dtype=np.float64)
    M = P.shape[0]
    R, p = P[:, :3, :3], P[:, :3, 3]
    rel_r = np.tile(np.eye(3), (M, 1, 1))
    rel_r[1:] = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    d = np.zeros((M, 3))
    d[1:] = p[1:] - p[:-1]
    rel_t = np.zeros((M, 3))
    rel_t[1:] = np.einsum("tji,tj->ti", R[:-1], d[1:])
    cum = np.cumsum(np.linalg.norm(d, axis=1))
    f = np.concatenate([(rot6d(rel_r) - _ID6) * ROT_SCALE, rel_t * (M - 1), cum[:, None]], axis=1)
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0)
    n = min(M, max_len)
    feat = np.zeros((max_len, FEAT_DIM), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    feat[:n], mask[:n] = f[:n], True
    return feat, mask


def featurize(poses: np.ndarray, max_len: int = MAX_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """Raw extracted c2w poses (any start, any scale) -> model input. Use this at inference time."""
    return pose_features(normalize_poses(poses), max_len)


def gen_sample(split: str, idx: int, base_seed: int = TRAIN_SEED_BASE, cfg: NoiseCfg = NoiseCfg()):
    """Sample ``idx`` of a split: rig = idx % 14, k = idx // 14. Returns a dict or None if every attempt failed."""
    rigs = _rigs()
    names = rigs.RIG_NAMES
    rig_idx, k = idx % N_RIGS, idx // N_RIGS
    name = names[rig_idx]
    for attempt in range(MAX_ATTEMPTS):
        rng = sample_rng(split, rig_idx, k, attempt, base_seed)
        shrink = 1.0 if attempt == 0 else (0.5 ** attempt if attempt < MAX_ATTEMPTS - 1 else 0.0)
        try:
            params, aux, mask = draw_params(name, rng, shrink)
            geom = draw_geom(name, rng)
            rig = rigs.make_rig(name, params=params, geom=geom)
            intent, total = make_intent(rig.q_scale, rng)
            traj = rig.simulate(intent, seed=int(rng.integers(0, 2**31 - 1)))
            if not (np.isfinite(traj.q).all() and np.isfinite(traj.pose).all()):
                continue
            if float(np.max(np.abs(traj.q - traj.q_ref) / rig.q_scale)) > 3.0:     # diverged / unstable draw
                continue
            poses = traj.frame_poses(FPS)
            noisy = add_sfm_noise(poses, rng, cfg)
            feat, fmask = pose_features(noisy)
        except Exception:  # noqa: BLE001 - a bad draw is re-drawn, never fatal
            continue
        aux = aux.copy()
        aux[-1] = (total - DUR_MIN) / (DUR_MAX - DUR_MIN)
        mask = mask.copy()
        mask[-1] = True
        return {"x": feat, "len": int(fmask.sum()), "y": rig_idx, "aux": aux, "aux_mask": mask}
    return None


def gen_unit(split: str, start: int, stop: int, base_seed: int = TRAIN_SEED_BASE,
             cfg: NoiseCfg = NoiseCfg()) -> Dict[str, np.ndarray]:
    """Samples ``[start, stop)`` of a split packed as arrays (failed samples are left out)."""
    xs, lens, ys, auxs, ams, idxs = [], [], [], [], [], []
    for i in range(start, stop):
        s = gen_sample(split, i, base_seed, cfg)
        if s is None:
            continue
        idxs.append(i)
        xs.append(s["x"].astype(np.float16))
        lens.append(s["len"])
        ys.append(s["y"])
        auxs.append(s["aux"].astype(np.float32))
        ams.append(s["aux_mask"].astype(np.uint8))
    n = len(idxs)
    return {"idx": np.asarray(idxs, dtype=np.int32),
            "x": np.stack(xs) if n else np.zeros((0, MAX_LEN, FEAT_DIM), dtype=np.float16),
            "len": np.asarray(lens, dtype=np.int16), "y": np.asarray(ys, dtype=np.int8),
            "aux": np.stack(auxs) if n else np.zeros((0, N_AUX), dtype=np.float32),
            "aux_mask": np.stack(ams) if n else np.zeros((0, N_AUX), dtype=np.uint8),
            "n_failed": np.asarray(stop - start - n, dtype=np.int32)}


# ---------------------------------------------------------------------------
# cache: resumable units, merged split files
# ---------------------------------------------------------------------------

def cache_key(base_seed: int, cfg: NoiseCfg) -> str:
    blob = json.dumps({"v": CACHE_VERSION, "seed": int(base_seed), "cfg": asdict(cfg), "fps": FPS,
                       "len": MAX_LEN, "feat": FEAT_DIM, "aux": list(AUX_KEYS)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def savez_atomic(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def unit_bounds(u: int, n_total: int, unit: int = UNIT) -> Tuple[int, int]:
    return u * unit, min(n_total, (u + 1) * unit)


def _unit_task(args):
    split, u, start, stop, base_seed, cfg = args
    try:
        return u, gen_unit(split, start, stop, base_seed, cfg), None
    except Exception as e:  # noqa: BLE001 - one bad unit must not kill the run
        return u, None, f"{type(e).__name__}: {e}"


def merge_units(sdir: str, n_units: int, n_total: int, out_path: str) -> Dict[str, int]:
    parts = []
    for u in range(n_units):
        p = os.path.join(sdir, f"unit_{u:05d}.npz")
        if not os.path.isfile(p):
            continue
        try:
            with np.load(p) as z:
                parts.append({k: z[k] for k in z.files})
        except Exception as e:  # noqa: BLE001 - corrupt unit: drop it, it is rebuilt next run
            warn("bad_unit", unit=u, err=f"{type(e).__name__}: {e}")
            try:
                os.remove(p)
            except OSError:
                pass
    keys = ("idx", "x", "len", "y", "aux", "aux_mask")
    merged = {k: (np.concatenate([q[k] for q in parts]) if parts else None) for k in keys}
    if not parts:
        merged = {"idx": np.zeros(0, np.int32), "x": np.zeros((0, MAX_LEN, FEAT_DIM), np.float16),
                  "len": np.zeros(0, np.int16), "y": np.zeros(0, np.int8),
                  "aux": np.zeros((0, N_AUX), np.float32), "aux_mask": np.zeros((0, N_AUX), np.uint8)}
    order = np.argsort(merged["idx"], kind="stable")
    out = {k: merged[k][order] for k in keys}
    complete = len(parts) == n_units
    out["n_expected"] = np.asarray(n_total, dtype=np.int64)
    out["complete"] = np.asarray(int(complete), dtype=np.int8)
    savez_atomic(out_path, out)
    return {"n": int(len(out["y"])), "units": len(parts), "complete": int(complete)}


def merged_is_complete(path: str, n_total: int) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path) as z:
            return bool(int(z["complete"])) and int(z["n_expected"]) == n_total
    except Exception:  # noqa: BLE001
        return False


def ensure_split(cdir: str, split: str, per_rig: int, base_seed: int, cfg: NoiseCfg, procs: int,
                 should_stop: Callable[[], bool], now: Callable[[], float] = time.time,
                 log_every_s: float = 300.0, unit: int = UNIT) -> Optional[str]:
    """Build (or finish) the cache of one split. Returns the merged ``.npz`` path, None if stopped early."""
    n_total = per_rig * N_RIGS
    n_units = -(-n_total // unit)
    sdir = os.path.join(cdir, split)
    os.makedirs(sdir, exist_ok=True)
    merged = os.path.join(cdir, f"{split}.npz")
    if merged_is_complete(merged, n_total):
        return merged
    for n in os.listdir(sdir):
        if n.endswith(".tmp"):
            try:
                os.remove(os.path.join(sdir, n))
            except OSError:
                pass
    todo = [u for u in range(n_units) if not os.path.isfile(os.path.join(sdir, f"unit_{u:05d}.npz"))]
    t0 = now()
    last_log = t0
    done = failed_units = made = 0
    log({"phase": "gen", "split": split, "units_todo": len(todo), "units": n_units, "procs": procs})

    def handle(u: int, arrays, err) -> None:
        nonlocal done, failed_units, made, last_log
        if arrays is None:
            failed_units += 1
            warn("unit_failed", split=split, unit=u, err=err or "?")
        else:
            savez_atomic(os.path.join(sdir, f"unit_{u:05d}.npz"), arrays)
            made += int(len(arrays["y"]))
        done += 1
        t = now()
        if t - last_log >= log_every_s:
            last_log = t
            log({"phase": "gen", "split": split, "units_done": done, "units_todo": len(todo),
                 "samples": made, "per_min": rnd(made / max(1e-6, (t - t0) / 60.0), 1)})

    tasks = [(split, u, *unit_bounds(u, n_total, unit), base_seed, cfg) for u in todo]
    stopped = False
    if todo and procs > 1:
        ctx = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods()
                                          else "spawn")
        pool = ctx.Pool(min(procs, len(todo)))
        try:
            for u, arrays, err in pool.imap_unordered(_unit_task, tasks):
                handle(u, arrays, err)
                if should_stop():
                    stopped = True
                    break
        finally:
            pool.terminate()
            pool.join()
    else:
        for t in tasks:
            if should_stop():
                stopped = True
                break
            handle(*_unit_task(t))
    if stopped:
        return None
    info = merge_units(sdir, n_units, n_total, merged)
    log({"phase": "gen_done", "split": split, "n": info["n"], "units": info["units"], "of": n_units,
         "failed_units": failed_units, "s": rnd(now() - t0, 1)})
    return merged


def load_split(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in ("x", "len", "y", "aux", "aux_mask", "idx")}


# ---------------------------------------------------------------------------
# model (~3M): 3 x conv1d(k=5) -> 4-layer transformer (d=192) with [CLS]
# ---------------------------------------------------------------------------

class RigIDNet(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, n_cls: int = N_RIGS, n_aux: int = N_AUX, d: int = 192,
                 conv_ch: Sequence[int] = (128, 192, 192), layers: int = 4, heads: int = 6,
                 ff: int = 1280, max_len: int = MAX_LEN, dropout: float = 0.1):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "n_cls": n_cls, "n_aux": n_aux, "d": d, "conv_ch": list(conv_ch),
                    "layers": layers, "heads": heads, "ff": ff, "max_len": max_len, "dropout": dropout}
        chans = [in_dim] + list(conv_ch)
        self.convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], 5, padding=2)
                                   for i in range(len(conv_ch)))
        self.proj = nn.Identity() if chans[-1] == d else nn.Linear(chans[-1], d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.head = nn.Linear(d, n_cls)
        self.aux_head = nn.Linear(d, n_aux)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x (B,T,F), mask (B,T) bool (True = real frame) -> (logits (B,n_cls), aux (B,n_aux) in [0,1])."""
        m = mask.to(x.dtype).unsqueeze(1)
        h = x.transpose(1, 2) * m
        for conv in self.convs:
            h = F.gelu(conv(h)) * m
        h = self.proj(h.transpose(1, 2))
        B, T, _ = h.shape
        h = torch.cat([self.cls.expand(B, -1, -1).to(h.dtype), h], dim=1) + self.pos[:, : T + 1].to(h.dtype)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask.bool()], dim=1)
        h = self.enc(h, src_key_padding_mask=~keep)
        c = h[:, 0]
        return self.head(c), torch.sigmoid(self.aux_head(c))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lens_to_mask(lens: torch.Tensor, T: int = MAX_LEN) -> torch.Tensor:
    return torch.arange(T, device=lens.device)[None, :] < lens.long()[:, None]


def compute_loss(model: RigIDNet, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, aux: torch.Tensor,
                 aux_mask: torch.Tensor, aux_w: float = 0.1, label_smoothing: float = 0.05,
                 amp: bool = False, device_type: str = "cpu") -> Dict[str, torch.Tensor]:
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        logits, pred_aux = model(x, mask)
    logits, pred_aux = logits.float(), pred_aux.float()
    ce = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
    am = aux_mask.float()
    l_aux = (((pred_aux - aux) ** 2) * am).sum() / am.sum().clamp(min=1.0)
    return {"loss": ce + aux_w * l_aux, "ce": ce, "aux": l_aux, "logits": logits, "aux_pred": pred_aux}


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_frac: float = 0.02) -> float:
    """Linear warmup, then cosine down to ``min_frac * base_lr``."""
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


# ---------------------------------------------------------------------------
# evaluation, confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(pred: np.ndarray, y: np.ndarray, n: int = N_RIGS) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)), 1)
    return cm


@torch.no_grad()
def evaluate(model: RigIDNet, data: Dict[str, np.ndarray], device, bs: int = 512, amp: bool = False,
             aux_w: float = 0.1) -> dict:
    """Validation accuracy, top-2, loss, per-rig accuracy and the confusion matrix."""
    was_training = model.training
    model.eval()
    n = len(data["y"])
    preds, top2, losses, aux_err, aux_cnt = [], 0, 0.0, 0.0, 0.0
    try:
        for s in range(0, n, bs):
            x = torch.from_numpy(data["x"][s:s + bs]).to(device).float()
            m = lens_to_mask(torch.from_numpy(data["len"][s:s + bs].astype(np.int64)).to(device))
            y = torch.from_numpy(data["y"][s:s + bs].astype(np.int64)).to(device)
            a = torch.from_numpy(data["aux"][s:s + bs]).to(device)
            am = torch.from_numpy(data["aux_mask"][s:s + bs]).to(device)
            out = compute_loss(model, x, m, y, a, am, aux_w, 0.0, amp, torch.device(device).type)
            losses += float(out["ce"]) * len(y)
            logits = out["logits"]
            preds.append(logits.argmax(1).cpu().numpy())
            top2 += int((logits.topk(min(2, logits.shape[1]), dim=1).indices == y[:, None]).any(1).sum())
            aux_err += float(((out["aux_pred"] - a).abs() * am.float()).sum())
            aux_cnt += float(am.float().sum())
    finally:
        model.train(was_training)
    pred = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
    y_all = data["y"].astype(np.int64)
    cm = confusion_matrix(pred, y_all)
    per_rig = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan") for i in range(cm.shape[0])]
    return {"n": int(n), "val_acc": float((pred == y_all).mean()) if n else float("nan"),
            "top2": top2 / n if n else float("nan"), "val_loss": losses / n if n else float("nan"),
            "aux_mae": aux_err / aux_cnt if aux_cnt else float("nan"), "per_rig_acc": per_rig,
            "confusion": cm}


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_confusion(cm: np.ndarray, names: Sequence[str], title: str = "", cell: int = 46):
    """14x14 heat map (row-normalised) with counts; rows = true rig, columns = predicted rig."""
    from PIL import Image, ImageDraw
    n = cm.shape[0]
    names = list(names)[:n] + [str(i) for i in range(len(names), n)]
    f_small, f_big = _font(12), _font(16)
    left, top = 160, 150
    W, H = left + n * cell + 10, top + n * cell + 10
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.text((10, 8), title or "rig identification: validation confusion matrix", fill=(0, 0, 0), font=f_big)
    dr.text((10, 32), "rows = true rig, columns = predicted rig (colour = row-normalised)", fill=(60, 60, 60),
            font=f_small)
    rows = cm.sum(1, keepdims=True).clip(min=1)
    frac = cm / rows
    for i in range(n):
        label = names[i]
        w = dr.textlength(label, font=f_small)
        dr.text((left - 6 - w, top + i * cell + cell // 2 - 6), label, fill=(0, 0, 0), font=f_small)
        tile = Image.new("RGBA", (int(dr.textlength(label, font=f_small)) + 8, 18), (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((2, 2), label, fill=(0, 0, 0, 255), font=f_small)
        tile = tile.rotate(90, expand=True)
        img.paste(tile, (left + i * cell + cell // 2 - 9, top - 6 - tile.height), tile)
        for j in range(n):
            v = float(frac[i, j])
            if i == j:
                col = (int(255 - 215 * v), int(255 - 120 * v), int(255 - 215 * v))   # green diagonal
            else:
                col = (255, int(255 - 205 * v), int(255 - 205 * v))                  # red off-diagonal
            x0, y0 = left + j * cell, top + i * cell
            dr.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col, outline=(200, 200, 200))
            if cm[i, j]:
                s = str(int(cm[i, j]))
                dr.text((x0 + (cell - dr.textlength(s, font=f_small)) / 2, y0 + cell // 2 - 6), s,
                        fill=(0, 0, 0), font=f_small)
    return img


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


def write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


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


def clean_tmp(out_dir: str) -> int:
    n = 0
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.endswith(".tmp") or name.endswith(".part"):
                try:
                    os.remove(os.path.join(out_dir, name))
                    n += 1
                except OSError:
                    pass
    return n


UPLOAD_FILES = ("ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl", "confusion.png", "confusion.json")


def stage_upload(out_dir: str) -> Tuple[str, List[str]]:
    """Copy the upload files into one folder so a single commit carries them."""
    stage = os.path.join(out_dir, "upload_stage")
    os.makedirs(stage, exist_ok=True)
    names = []
    for name in UPLOAD_FILES:
        src = os.path.join(out_dir, name)
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
            commit_message="day1 m13 rig-id checkpoint")

    return upload


def upload_with_retry(uploader: Callable[[str, str], None], folder: str, remote: str, tries: int = 3,
                      sleep: Callable[[float], None] = time.sleep) -> bool:
    for k in range(tries):
        try:
            uploader(folder, remote)
            return True
        except Exception as e:  # noqa: BLE001 - upload trouble is a warning, never fatal
            warn("upload_failed", attempt=k + 1, err=f"{type(e).__name__}: {e}")
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
    oom_cls = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", ())
    if oom_cls and isinstance(e, oom_cls):
        return True
    m = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in m or "alloc_failed" in m)


def apply_cuda_limits(device: torch.device, mem_gb: float) -> Optional[float]:
    """Cap this process's caching allocator (the GPU is shared with M6's Z-Image). Returns the cap in GB."""
    if device.type != "cuda" or mem_gb <= 0:
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory
        frac = min(1.0, (mem_gb * 2**30) / float(total))
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        return round(frac * total / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        warn("mem_cap_failed", err=f"{type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args, uploader=_UNSET, now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, stop_event: Optional[threading.Event] = None):
        self.a = args
        self.now, self.sleep = now, sleep
        self.stop = stop_event or threading.Event()
        self.out = args.out or os.path.join(args.root, "day1", "m13")
        self.deadline = parse_deadline(args.deadline)
        self.uploader = uploader
        self.upload_thread: Optional[threading.Thread] = None
        self.last_upload = -math.inf
        self.noise = NoiseCfg()
        self.model: Optional[RigIDNet] = None
        self.opt = None
        self.device = None
        self.amp = False
        self.step = 0
        self.epoch = 0                       # fully completed epochs
        self.best_acc = -math.inf
        self.best_loss = math.inf
        self.skipped = 0
        self.bad_run = 0
        self.micro_batch = int(args.micro_batch or args.batch)
        self.last_loss = float("nan")
        self.names = rig_names()

    # -- helpers ------------------------------------------------------------
    def _stop_reason(self) -> Optional[str]:
        if self.stop.is_set():
            return "signal"
        if self.deadline is not None and self.now() >= self.deadline - self.a.finish_margin_s:
            return "deadline"
        if self.a.max_steps is not None and self.step >= self.a.max_steps:
            return "max_steps"
        return None

    def _ckpt_obj(self) -> dict:
        return {"version": 1, "step": self.step, "epoch": self.epoch, "model": self.model.state_dict(),
                "opt": self.opt.state_dict(), "config": self.model.cfg, "best_acc": self.best_acc,
                "best_loss": self.best_loss, "rig_names": list(self.names), "micro_batch": self.micro_batch,
                "saved_at": time.time()}

    def _load_resume(self) -> Optional[dict]:
        for name in ("ckpt_last.pt", "ckpt_best.pt"):
            ck = load_ckpt(os.path.join(self.out, name))
            if ck is None:
                continue
            cfg = ck.get("config") or {}
            if int(cfg.get("n_cls", -1)) != N_RIGS or int(cfg.get("in_dim", -1)) != FEAT_DIM:
                warn("ckpt_config_mismatch", path=name)
                continue
            return ck
        return None

    def _restore(self, ck: dict) -> bool:
        try:
            self.model.load_state_dict(ck["model"])
            if "opt" in ck:
                self.opt.load_state_dict(ck["opt"])
            self.step = int(ck["step"])
            self.epoch = int(ck.get("epoch", 0))
            ba = ck.get("best_acc")
            self.best_acc = float(ba) if ba is not None else -math.inf
            bl = ck.get("best_loss")
            self.best_loss = float(bl) if bl is not None else math.inf
            mb = ck.get("micro_batch")
            if mb and not self.a.micro_batch:
                self.micro_batch = max(1, min(self.micro_batch, int(mb)))    # never grow back past an OOM size
            return True
        except Exception as e:  # noqa: BLE001
            warn("ckpt_restore_failed", err=f"{type(e).__name__}: {e}")
            return False

    # -- upload ---------------------------------------------------------------
    def _upload_job(self, stage: str) -> None:
        if upload_with_retry(self.uploader, stage, REMOTE_DIR, sleep=self.sleep):
            log({"uploaded": REMOTE_DIR, "step": self.step})

    def _start_upload(self, final: bool = False) -> None:
        if self.uploader is None:
            return
        t = self.now()
        if not final and t - self.last_upload < self.a.upload_every_s:
            return
        th = self.upload_thread
        if th is not None and th.is_alive():
            if not final:
                return
            th.join(timeout=self.a.final_upload_timeout_s)
        try:
            stage, _ = stage_upload(self.out)
        except Exception as e:  # noqa: BLE001
            warn("stage_failed", err=f"{type(e).__name__}: {e}")
            return
        self.last_upload = t
        self.upload_thread = threading.Thread(target=self._upload_job, args=(stage,), daemon=True)
        self.upload_thread.start()

    def _finish_upload(self) -> None:
        th = self.upload_thread
        if th is not None and th.is_alive():
            th.join(timeout=self.a.final_upload_timeout_s)
            if th.is_alive():
                warn("final_upload_timeout")

    # -- epoch bookkeeping ------------------------------------------------------
    def _batch_tensors(self, data: Dict[str, np.ndarray], idx: np.ndarray):
        x = torch.from_numpy(data["x"][idx]).to(self.device, non_blocking=True).float()
        ln = torch.from_numpy(data["len"][idx].astype(np.int64)).to(self.device)
        y = torch.from_numpy(data["y"][idx].astype(np.int64)).to(self.device)
        a = torch.from_numpy(data["aux"][idx]).to(self.device)
        am = torch.from_numpy(data["aux_mask"][idx]).to(self.device)
        return x, lens_to_mask(ln, x.shape[1]), y, a, am

    def _accumulate(self, batch, micro: int) -> torch.Tensor:
        """Backward over micro-batches (gradient accumulation). Returns [loss, ce, aux] of the full batch."""
        x, m, y, a, am = batch
        self.opt.zero_grad(set_to_none=True)
        n = x.shape[0]
        tot = None
        for s in range(0, n, micro):
            sl = slice(s, s + micro)
            w = x[sl].shape[0] / n
            out = compute_loss(self.model, x[sl], m[sl], y[sl], a[sl], am[sl], self.a.aux_weight,
                               self.a.label_smoothing, self.amp, self.device.type)
            (out["loss"] * w).backward()
            part = torch.stack([out["loss"].detach(), out["ce"].detach(), out["aux"].detach()]) * w
            tot = part if tot is None else tot + part
            del out
        return tot

    def _train_step(self, batch, total_steps: int) -> bool:
        lr = lr_at(self.step, self.a.lr, self.a.warmup_steps, total_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr
        n = batch[0].shape[0]
        while True:
            oom, parts = False, None
            try:
                parts = self._accumulate(batch, max(1, min(self.micro_batch, n)))
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                oom = True
            if not oom:
                break
            self.opt.zero_grad(set_to_none=True)
            cur = max(1, min(self.micro_batch, n))
            if cur <= 1:
                raise RuntimeError("out of memory at micro_batch=1")
            self.micro_batch = max(1, cur // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            warn("oom_backoff", step=self.step, micro_batch=self.micro_batch, vram_GB=vram_gb())
        if not clip_and_step(self.model, self.opt, self.a.clip):
            self.skipped += 1
            self.bad_run += 1
            return False
        self.bad_run = 0
        self.acc += parts
        self.acc_n += 1
        self.step += 1
        return True

    def _take_loss(self, reset: bool = True) -> Tuple[float, float, float]:
        """Mean [loss, ce, aux] over the steps since the last reset (the epoch so far)."""
        if not self.acc_n:
            return float("nan"), float("nan"), float("nan")
        v = [x / self.acc_n for x in self.acc.tolist()]
        if reset:
            self.acc.zero_()
            self.acc_n = 0
        return v[0], v[1], v[2]

    def _write_eval(self, metrics: dict, loss: float, partial: bool) -> bool:
        """Append the metrics line, refresh the confusion files, save ckpts. Returns whether it is a new best."""
        improved = (metrics["val_acc"] > self.best_acc + 1e-12 or
                    (abs(metrics["val_acc"] - self.best_acc) <= 1e-12 and metrics["val_loss"] < self.best_loss))
        if improved:
            self.best_acc, self.best_loss = metrics["val_acc"], metrics["val_loss"]
        epoch_f = self.epoch if not partial else self.epoch + 0.5
        rec = {"epoch": epoch_f, "step": self.step, "loss": rnd(loss, 5), "val_acc": rnd(metrics["val_acc"], 5),
               "top2": rnd(metrics["top2"], 5), "val_loss": rnd(metrics["val_loss"], 5),
               "aux_mae": rnd(metrics["aux_mae"], 5), "best": improved, "partial": partial,
               "per_rig_acc": [rnd(v, 3) for v in metrics["per_rig_acc"]], "time": round(time.time(), 1)}
        try:
            with open(os.path.join(self.out, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            warn("metrics_write_failed", err=str(e))
        try:
            cm = metrics["confusion"]
            title = f"epoch {epoch_f}  val_acc {metrics['val_acc']:.4f}  top2 {metrics['top2']:.4f}"
            import io
            buf = io.BytesIO()
            draw_confusion(cm, self.names, title).save(buf, format="PNG")
            write_atomic(os.path.join(self.out, "confusion.png"), buf.getvalue())
            write_atomic(os.path.join(self.out, "confusion.json"), json.dumps(
                {"names": list(self.names), "epoch": epoch_f, "confusion": cm.tolist()}).encode())
        except Exception as e:  # noqa: BLE001 - a plotting problem must not lose the checkpoint
            warn("confusion_failed", err=f"{type(e).__name__}: {e}")
        try:
            obj = self._ckpt_obj()
            save_atomic(obj, os.path.join(self.out, "ckpt_last.pt"))
            if improved:
                save_atomic(obj, os.path.join(self.out, "ckpt_best.pt"))
        except Exception as e:  # noqa: BLE001
            warn("ckpt_save_failed", err=f"{type(e).__name__}: {e}")
        log({"epoch": epoch_f, "step": self.step, "loss": rnd(loss, 4), "val_acc": rnd(metrics["val_acc"], 4),
             "top2": rnd(metrics["top2"], 4), "best": rnd(self.best_acc, 4), "partial": partial,
             "vram_GB": vram_gb()})
        return improved

    # -- main -------------------------------------------------------------------
    def run(self) -> dict:
        a = self.a
        os.makedirs(self.out, exist_ok=True)
        clean_tmp(self.out)
        if self.uploader is _UNSET:
            self.uploader = None if a.no_upload else make_uploader(os.environ.get("HF_TOKEN"), a.repo_id)
        log({"start": True, "out": self.out, "hf_token": "yes" if self.uploader else "no",
             "deadline": self.deadline, "torch": torch.__version__})

        # ---- data ----
        procs = a.procs if a.procs > 0 else max(1, (os.cpu_count() or 4) - 2)
        cdir = os.path.join(self.out, "cache", cache_key(a.seed, self.noise))
        stop_fn = lambda: self._stop_reason() in ("signal", "deadline")  # noqa: E731
        paths = {}
        for split, per_rig in (("val", a.val_per_rig), ("train", a.train_per_rig)):
            try:
                paths[split] = ensure_split(cdir, split, per_rig, a.seed + (0 if split == "train" else 7919),
                                            self.noise, procs, stop_fn, self.now, a.log_every_s)
            except Exception as e:  # noqa: BLE001
                log({"fatal": f"datagen {type(e).__name__}: {scrub(e)}"[:300]}, marker=MARK + "_DONE")
                return {"status": "fatal", "step": self.step}
            if paths[split] is None:
                log({"status": "deadline_in_datagen" if not self.stop.is_set() else "signal_in_datagen",
                     "split": split}, marker=MARK + "_DONE")
                return {"status": "no_data", "step": 0}
        if a.only_gen:
            log({"status": "gen_only"}, marker=MARK + "_DONE")
            return {"status": "gen_only", "step": 0}
        train, val = load_split(paths["train"]), load_split(paths["val"])
        n_train, n_val = len(train["y"]), len(val["y"])
        if n_train == 0 or n_val == 0:
            log({"fatal": "empty_dataset", "train": n_train, "val": n_val}, marker=MARK + "_DONE")
            return {"status": "fatal", "step": 0}

        # ---- model ----
        dev_name = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_name)
        self.amp = {"on": True, "off": False}.get(a.amp, self.device.type == "cuda")
        mem_cap = apply_cuda_limits(self.device, a.mem_gb) if self.device.type == "cuda" else None
        torch.manual_seed(a.seed)
        ck = self._load_resume()
        model_cfg = json.loads(a.model_cfg) if a.model_cfg else {}
        if ck is not None:
            model_cfg = dict(ck["config"])                   # the checkpoint decides the architecture
        self.model = RigIDNet(**model_cfg).to(self.device)
        n_params = count_params(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=a.wd,
                                     fused=(self.device.type == "cuda"))
        resumed = ck is not None and self._restore(ck)
        batch = max(1, min(a.batch, n_train))
        spe = max(1, n_train // batch)                       # steps per epoch (drop_last)
        total_steps = a.epochs * spe
        start_step = self.step
        log({"params": n_params, "in_range": PARAM_MIN <= n_params <= PARAM_MAX, "train": n_train, "val": n_val,
             "device": str(self.device), "amp": self.amp, "resumed": bool(resumed), "epoch": self.epoch,
             "step": self.step, "batch": batch, "micro": self.micro_batch, "mem_cap_GB": mem_cap})

        self.acc = torch.zeros(3, device=self.device)
        self.acc_n = 0
        t0 = self.now()
        next_log = t0 + a.log_every_s
        reason: Optional[str] = None
        consec_err = 0
        self.model.train()
        if self.epoch >= a.epochs:
            reason = "complete"

        while reason is None and self.epoch < a.epochs:
            perm = np.random.default_rng([a.seed, 0xE90C, self.epoch]).permutation(n_train)
            interrupted = False
            for s in range(spe):
                reason = self._stop_reason()
                if reason:
                    interrupted = True
                    break
                idx = np.sort(perm[s * batch:(s + 1) * batch])
                try:
                    ok = self._train_step(self._batch_tensors(train, idx), total_steps)
                    consec_err = 0
                    if self.bad_run >= 25:
                        warn("nonfinite_streak", step=self.step)
                        ck2 = self._load_resume()
                        if ck2 is not None:
                            self._restore(ck2)
                        self.bad_run = 0
                except Exception as e:  # noqa: BLE001 - one bad step is recorded and skipped
                    ok = False
                    consec_err += 1
                    warn("step_failed", step=self.step, n=consec_err, err=f"{type(e).__name__}: {e}")
                    self.opt.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if consec_err >= MAX_CONSEC_ERRORS:
                        log({"fatal": "too_many_step_errors", "step": self.step}, marker=MARK + "_DONE")
                        return {"status": "fatal", "step": self.step}
                t = self.now()
                if t >= next_log:
                    l, _, _ = self._take_loss(reset=False)
                    log({"progress": 1, "epoch": self.epoch, "step": self.step, "of": total_steps,
                         "loss": rnd(l, 4), "lr": rnd(lr_at(self.step, a.lr, a.warmup_steps, total_steps), 6),
                         "skipped": self.skipped, "vram_GB": vram_gb()})
                    next_log = t + a.log_every_s
            l, _, _ = self._take_loss()
            if not math.isnan(l):
                self.last_loss = l
            if interrupted:
                break
            self.epoch += 1
            try:
                metrics = evaluate(self.model, val, self.device, amp=self.amp, aux_w=a.aux_weight)
                self._write_eval(metrics, self.last_loss, partial=False)
            except Exception as e:  # noqa: BLE001
                warn("eval_failed", err=f"{type(e).__name__}: {e}")
                try:
                    save_atomic(self._ckpt_obj(), os.path.join(self.out, "ckpt_last.pt"))
                except Exception as e2:  # noqa: BLE001
                    warn("ckpt_save_failed", err=f"{type(e2).__name__}: {e2}")
            self._start_upload(False)
            next_log = self.now() + a.log_every_s

        # ---- cleanup: final eval, save, upload, normal exit ----
        if reason not in (None, "complete"):
            try:
                metrics = evaluate(self.model, val, self.device, amp=self.amp, aux_w=a.aux_weight)
                self._write_eval(metrics, self.last_loss, partial=True)
            except Exception as e:  # noqa: BLE001
                warn("eval_failed", err=f"{type(e).__name__}: {e}")
                try:
                    save_atomic(self._ckpt_obj(), os.path.join(self.out, "ckpt_last.pt"))
                except Exception:  # noqa: BLE001
                    pass
        elif reason == "complete" and not os.path.isfile(os.path.join(self.out, "confusion.png")):
            try:
                metrics = evaluate(self.model, val, self.device, amp=self.amp, aux_w=a.aux_weight)
                self._write_eval(metrics, self.last_loss, partial=False)
            except Exception as e:  # noqa: BLE001
                warn("eval_failed", err=f"{type(e).__name__}: {e}")
        self._start_upload(True)
        self._finish_upload()
        summary = {"status": reason or "epochs_done", "start_step": start_step, "step": self.step,
                   "epochs_done": self.epoch,
                   "best_val_acc": None if self.best_acc == -math.inf else rnd(self.best_acc, 4),
                   "params": n_params, "skipped": self.skipped, "s": rnd(self.now() - t0, 1)}
        log(summary, marker=MARK + "_DONE")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Day1 job 1: M13 Rig-ID-S")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", "/workspace/night1"))
    ap.add_argument("--out", default=None, help="default: <root>/day1/m13")
    ap.add_argument("--deadline", default=os.environ.get("N1_DEADLINE"),
                    help="unix seconds (default: env N1_DEADLINE)")
    ap.add_argument("--finish-margin-s", type=float, default=600.0,
                    help="final save/eval/upload starts this many seconds before the deadline")
    ap.add_argument("--train-per-rig", type=int, default=2857, help="x14 rigs = 39,998 training sequences")
    ap.add_argument("--val-per-rig", type=int, default=285, help="x14 rigs = 3,990 validation sequences")
    ap.add_argument("--procs", type=int, default=0, help="data workers (0 = cores - 2)")
    ap.add_argument("--only-gen", action="store_true", help="build the data cache and stop")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=128,
                    help="sequences per forward/backward (halves itself on CUDA OOM; accumulation keeps the batch)")
    ap.add_argument("--mem-gb", type=float, default=MEM_CAP_GB, help="per-process CUDA memory cap (0 = off)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=150)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=None, help="stop at this absolute step (tests)")
    ap.add_argument("--log-every-s", type=float, default=240.0, help="progress marker period (>=1 per 5 min)")
    ap.add_argument("--upload-every-s", type=float, default=300.0)
    ap.add_argument("--final-upload-timeout-s", type=float, default=300.0)
    ap.add_argument("--model-cfg", default=None, help="JSON overrides of RigIDNet hyper-parameters (tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--repo-id", default=REPO_ID)
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
        log({"fatal": f"{type(e).__name__}: {scrub(e)}"[:300]}, marker=MARK + "_DONE")
        try:
            if trainer.model is not None and trainer.opt is not None:
                save_atomic(trainer._ckpt_obj(), os.path.join(trainer.out, "ckpt_last.pt"))
        except Exception:  # noqa: BLE001
            pass
        return 2
    return 2 if res.get("status") == "fatal" else 0


if __name__ == "__main__":
    sys.exit(main())
