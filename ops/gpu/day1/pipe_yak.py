#!/usr/bin/env python3
"""Day1 job 4: the FIRST yak video rendered by OUR pipeline code (buc/), D15 demo image (docs/DEMO-D15.md).

Run on the GPU box (torch 2.9.1 + CUDA; repo tarball is unpacked to $N1_ROOT/repo and put on PYTHONPATH):

    HF_TOKEN=... N1_ROOT=/workspace/night1 N1_DEADLINE=1791723600 \
        python ops/gpu/day1/pipe_yak.py [--image PATH] [--modes skeleton,scaffold] [--shots S1,S2,...]

Two modes, one script:

  (A) skeleton  PRD-01 ``Spine.run`` with ONLY the PRD-03 stub pack registered (S-1 classic read, S2 flat world,
                S8 fixed push-in, S8.5 identity) -> RDR-02 render -> ``skeleton_S1.mp4``.  Works even when every
                real stage is broken (render and encode have their own cv2/imageio fallbacks).
  (B) scaffold  The REAL stages wherever they exist, one spine run per shot S1..S6:
                  S-1  RD-21 orchestrator (R0 ingest, R1 geom_cam, R5 mrh_call, R6 metric_scale)
                       R5's reader is a TEMPORARY Depth-Anything-V2-Small (Apache-2.0) because M3 MRH is not
                       trained yet.  R-EVAL lane: eval-only, every frame is labelled
                       "SCAFFOLD: temp depth (eval-only), not final", prov carries HOLDOUT:r_eval, no depth
                       map or intermediate is ever written to disk or uploaded.
                  S2   RD-18 edge_unmix -> WLD-03 depth_world -> WLD-08 sdedit_avg (Z-Image teacher, GPU)
                  S8   CAM-02 recipes realised on CAM-01 rigs (+ local crane / dolly-zoom, logged)
                  S8.5 identity polish (PRD-03 stub, no real exists)
                  S9-10 RDR-02 render_ldi (+ LNS-02 physical DoF for the rack focus), RDR-06 mp4 encoder
                Each real stage falls back to its stub / next-best alternative on failure and the choice is
                logged in ``stages_report.json`` (that report matters more than pretty frames).

Outputs under ``$N1_ROOT/day1/pipe``: skeleton_S1.mp4, scaffold_S1..S6.mp4, stages_report.json,
contact_sheet.jpg; uploaded to HF vault ``video/pipe_d1/``.
Markers (one JSON line, <= 400 chars, flushed): ``D1_PIPE {...}`` (+ ``D1_PIPE_DONE``).  HF_TOKEN comes from the
environment only and is never printed or stored.  One failing shot never stops the run.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MARK = "D1_PIPE"
MARKER_LIMIT = 400
REPORT_SCHEMA = "d1_pipe_report/v1"
HF_REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
HF_REPO_TYPE = "model"
HF_IMAGE_PATH = "demo/d15_hero_yak.webp"
HF_UPLOAD_PREFIX = "video/pipe_d1"
IMAGE_SHA256 = "7c1fc8dcebcfb485e93cff2b206405dd4af155acf3085213df82128876362bda"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
ZIMAGE_ID = "Tongyi-MAI/Z-Image-Turbo"

DEFAULT_ROOT = "/workspace/night1"
OUT_SUBDIR = ("day1", "pipe")
FPS = 24
OUT_W = 1920
REF_SIZE = (2000, 861)               # the image as delivered (docs/DEMO-D15.md); boxes below are in these pixels
PERSON_BOX = (900, 640, 1130, 840)   # two people under the chin (night1 baseline annotation)
FAR_LIMIT, NEAR_LIMIT, FAR_PERCENTILE = 40.0, 0.1, 2.0   # relative depth, person = 1.0
FINISH_MARGIN_S = 600.0              # stop rendering this long before N1_DEADLINE; save, upload, exit
HEARTBEAT_S = 120.0
FRAME_TIMEOUT_S = 900.0              # a pool worker silent this long -> the pool is dropped, serial takes over

SCAFFOLD_LABEL = "SCAFFOLD: temp depth (eval-only), not final"
SCAFFOLD_LABEL_NODEPTH = "SCAFFOLD: temp depth FAILED, plane fallback (eval-only), not final"
SKELETON_LABEL = "SKELETON: PRD-03 stub pack, not final"

R5_CFG = {"heads": ("log_depth",), "tiles": False, "global_sigmas": (0.2,)}
S5_STOPS = (0.35, 1.0, 1.6)          # focus distance in units of the people's depth (poppy, people, yak eye)
S5_F_STOP = 2.0
S5_LENS_DEPTH_SCALE = 0.15           # lens sees the scene shrunk by this factor so the rack is visible (logged)
S5_MAX_COC_PX = 48.0

# S1: the people sit at y=640..840 of 861, far below the frame centre.  A push along the optical axis (CAM-02 default
# m=1.5) would throw them off the bottom edge, so the push runs along the ray through the people-box centre and m is
# 1.25 (box centre keeps its pixel, the box grows 1.25x about it; frame 0 is still the observed pose).
PUSH_IN_M = 1.25

# name -> (move, seconds, kwargs).  docs/DEMO-D15.md section 3; amplitudes follow the "safe first" rule.
SHOT_DEFS: Dict[str, Dict[str, Any]] = {
    "S1": {"move": "push_in", "dur": 8.0, "kw": {"m": PUSH_IN_M}, "doc": "push-in (dolly in) toward the two people"},
    "S2": {"move": "slider", "dur": 8.0, "kw": {"slide_zs": 0.12}, "doc": "slider (left to right truck)"},
    "S3": {"move": "orbit", "dur": 10.0, "kw": {"orbit_deg": 12.0}, "doc": "arc around the two people"},
    "S4": {"move": "crane", "dur": 8.0, "kw": {"elev_deg": 8.0}, "doc": "crane up"},
    "S5": {"move": "rack_focus", "dur": 6.0, "kw": {"stops": S5_STOPS}, "doc": "rack focus, camera locked"},
    "S6": {"move": "dolly_zoom", "dur": 6.0, "kw": {"back_frac": 0.4}, "doc": "dolly zoom (vertigo)"},
}
SHOT_ORDER = tuple(SHOT_DEFS)

# unit names used in the report (one line per pipeline unit)
U_R0, U_R1, U_R5, U_R6 = "S-1.R0 ingest", "S-1.R1 geom_cam", "S-1.R5 mrh_call [temp DA-V2-S reader]", "S-1.R6 metric_scale"
U_MATTE = "RD-18 edge_unmix"
U_W3, U_W8 = "S2 WLD-03 depth_world", "S2 WLD-08 sdedit_avg"
U_CAM = "S8 CAM-02 moves/CAM-01 rigs"
U_POLISH = "S8.5 polish"
U_RENDER = "S9-10 RDR-02 render_ldi"
U_DOF = "LNS-02 dof (S5)"
U_ENC = "RDR-06 video encode"
S1_UNITS = (U_R0, U_R1, U_R5, U_R6)
UNITS_EXPECTED_REAL = [U_R0, U_R1, U_R5, U_R6, U_MATTE, U_W3, U_W8, U_CAM, U_RENDER, U_DOF, U_ENC]


# ---------------------------------------------------------------------------------------------------------------
# paths / imports
# ---------------------------------------------------------------------------------------------------------------
def setup_paths(root: Optional[str] = None) -> None:
    """Put the repo (``$N1_ROOT/repo`` first, then the checkout this file lives in) on sys.path."""
    cands = []
    r = root or os.environ.get("N1_ROOT") or DEFAULT_ROOT
    cands.append(Path(r) / "repo")
    cands.append(Path(__file__).resolve().parents[3])
    for c in cands:
        if (c / "buc").is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


setup_paths()


def ensure_blake3() -> bool:
    """buc.core.det needs ``blake3``; the GPU image may lack it.  Install a hashlib-backed shim in that case
    (only seeds and cache keys depend on it; roots are then not the canonical blake3 values).  True = shim."""
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


# ---------------------------------------------------------------------------------------------------------------
# logging / secrets
# ---------------------------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")


def scrub(text: object, token: Optional[str] = None) -> str:
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
    """One marker line, always <= ``limit`` chars and free of token-like strings."""
    def render(p) -> str:
        return scrub(f"{marker} {json.dumps(p, default=str, separators=(',', ':'))}")

    line = render(payload)
    if len(line) <= limit:
        return line
    p = _round_floats(payload, 3)
    p = {k: (v[:90] if isinstance(v, str) else v) for k, v in p.items()}
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


_PRINT_LOCK = threading.Lock()


def log(payload: dict, marker: str = MARK) -> None:
    with _PRINT_LOCK:
        print(fit_marker(marker, payload), flush=True)


def err_text(e: BaseException, n: int = 300) -> str:
    return scrub(f"{type(e).__name__}: {e}")[:n]


class Phase:
    """Current phase name for the heartbeat thread (a marker at least every 5 minutes, spec)."""
    name = "init"
    t0 = time.time()


def start_heartbeat(period_s: float = HEARTBEAT_S) -> threading.Event:
    stop = threading.Event()

    def run():
        while not stop.wait(period_s):
            log({"ev": "hb", "phase": Phase.name, "t": round(time.time() - Phase.t0)})

    threading.Thread(target=run, name="d1-pipe-heartbeat", daemon=True).start()
    return stop


# ---------------------------------------------------------------------------------------------------------------
# deadline
# ---------------------------------------------------------------------------------------------------------------
class Deadline:
    def __init__(self, ts: Optional[float], margin_s: float = FINISH_MARGIN_S,
                 clock: Callable[[], float] = time.time):
        self.ts = ts
        self.margin_s = margin_s
        self.clock = clock

    @classmethod
    def from_env(cls, clock: Callable[[], float] = time.time) -> "Deadline":
        raw = os.environ.get("N1_DEADLINE", "").strip()
        ts = None
        if raw:
            try:
                ts = float(raw)
            except ValueError:
                log({"warn": "bad N1_DEADLINE ignored"})
        return cls(ts, clock=clock)

    def left(self) -> Optional[float]:
        return None if self.ts is None else self.ts - self.clock()

    def stop_now(self) -> bool:
        """True once less than ``margin_s`` remain: no new work, wrap up."""
        left = self.left()
        return left is not None and left < self.margin_s

    def budget_s(self, frac: float = 0.6) -> Optional[float]:
        left = self.left()
        return None if left is None else max(0.0, (left - self.margin_s) * frac)


# ---------------------------------------------------------------------------------------------------------------
# the ledger: which units were real / temp / stub, per shot (the report is the point of this job)
# ---------------------------------------------------------------------------------------------------------------
# real = real stage ran; temp = temporary substitute ran (DA-V2 depth); rejected = real stage ran but its own quality
# gate refused the output; fallback = a stub / lesser alternative was used; unavailable = cannot run on this box;
# stub_only = no real implementation exists yet.
STATUS_OK = ("real", "temp")
STATUS_BAD = ("fallback", "failed", "skipped", "unavailable", "rejected")


class Ledger:
    def __init__(self) -> None:
        self.units: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, unit: str, status: str, detail: Optional[str] = None, shot: Optional[str] = None) -> None:
        with self._lock:
            u = self.units.setdefault(unit, {"counts": {}, "events": []})
            u["counts"][status] = u["counts"].get(status, 0) + 1
            if detail or status in STATUS_BAD:
                u["events"].append({"shot": shot, "status": status, "detail": (detail or "")[:300]})
                u["events"] = u["events"][-8:]

    def verdict(self, unit: str) -> str:
        u = self.units.get(unit)
        if not u:
            return "not_run"
        c = u["counts"]
        good = sum(c.get(s, 0) for s in STATUS_OK)
        bad = sum(c.get(s, 0) for s in STATUS_BAD)
        if good and not bad:
            return "temp" if c.get("temp") and not c.get("real") else "real"
        if good and bad:
            return "mixed"
        if c.get("stub_only") and not bad:
            return "stub_only"
        for st in ("fallback", "rejected", "unavailable", "skipped"):
            if c.get(st):
                return st
        return "failed"

    def summary(self) -> dict:
        return {k: {"verdict": self.verdict(k), "counts": v["counts"], "events": v["events"]}
                for k, v in self.units.items()}


# ---------------------------------------------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------------------------------------------
def scale_box(box, src_size, dst_size) -> Tuple[int, int, int, int]:
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    x0, y0, x1, y1 = box
    x0, y0 = int(round(x0 * sx)), int(round(y0 * sy))
    x1, y1 = int(round(x1 * sx)), int(round(y1 * sy))
    x0 = min(max(x0, 0), dst_size[0] - 1)
    y0 = min(max(y0, 0), dst_size[1] - 1)
    return (x0, y0, min(max(x1, x0 + 1), dst_size[0]), min(max(y1, y0 + 1), dst_size[1]))


def out_size_for(width: int, ref: Tuple[int, int] = REF_SIZE) -> Tuple[int, int]:
    """(W, H) with even sides and the reference aspect (2000x861 -> 1920x826)."""
    w = int(width) // 2 * 2
    h = int(round(w * ref[1] / ref[0] / 2.0)) * 2
    return w, max(2, h)


def resize_rgb(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    from PIL import Image
    if (arr.shape[1], arr.shape[0]) == tuple(size):
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.asarray(Image.fromarray(arr).resize(tuple(size), Image.LANCZOS)))


def load_image_array(path) -> np.ndarray:
    from PIL import Image
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))


def make_input_image(pixels_u8: np.ndarray, source: str):
    """FND-02 InputImage in the R-EVAL lane: USER + HOLDOUT:r_eval so BLOCK_TRAIN / BLOCK_DEV both refuse it."""
    from buc.core import contracts as C
    prov = C.Prov(taint=frozenset({"USER", "HOLDOUT:r_eval"}), mode="none", roots=(source,), ns="user_vault")
    return C.InputImage(pixels=pixels_u8, color_space="srgb", prov=prov)


def acquire_image(args, root: Path, token: Optional[str], download_fn: Optional[Callable] = None) -> dict:
    """-> {"array": HxWx3 uint8 at original size, "source": str, "sha256": str, "matches_doc": bool|None}"""
    if args.image:
        path, source = Path(args.image), "local"
    else:
        if not token:
            raise RuntimeError("no --image and no HF_TOKEN: cannot fetch the yak")
        fn = download_fn or default_download
        path, source = Path(fn(HF_REPO_ID, HF_IMAGE_PATH, token, str(root / "hf_cache"))), "hf_vault"
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    return {"array": load_image_array(path), "source": source, "sha256": sha,
            "matches_doc": (sha == IMAGE_SHA256) if source == "hf_vault" else None}


def default_download(repo_id: str, filename: str, token: str, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=HF_REPO_TYPE, token=token,
                           cache_dir=cache_dir)


# ---------------------------------------------------------------------------------------------------------------
# temporary depth source (R-EVAL lane): Depth-Anything-V2-Small  ->  R5 "reader"
# ---------------------------------------------------------------------------------------------------------------
def disp_to_depth(disp: np.ndarray, box, far: float = FAR_LIMIT, near: float = NEAR_LIMIT,
                  far_percentile: float = FAR_PERCENTILE) -> np.ndarray:
    """Relative inverse depth (larger = closer) -> relative depth with the people's median = 1.0 and the
    2nd percentile of disparity = ``far``:  depth = 1 / (a*disp + b)  (same calibration as the night1 baseline)."""
    d = np.asarray(disp, dtype=np.float64)
    x0, y0, x1, y1 = box
    dp = float(np.median(d[y0:y1, x0:x1]))
    dlo = float(np.percentile(d, far_percentile))
    gap = dp - dlo
    eps = 1e-6 * max(1.0, abs(dp))
    if gap < eps:
        gap = max(eps, 1e-3 * max(1.0, abs(dp)))
    a = (1.0 - 1.0 / far) / gap
    b = 1.0 - a * dp
    inv = np.clip(a * d + b, 1.0 / far, 1.0 / near)
    return (1.0 / inv).astype(np.float32)


class DepthAnythingTemp:
    """Depth-Anything-V2-Small through transformers (lazy).  ``__call__(rgb_u8) -> disparity (h, w) float32``.
    TEMPORARY eval-only depth source; nothing it produces is ever cached on disk."""

    def __init__(self, model_id: str = DEPTH_MODEL, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._m = None

    def _load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(dev).eval()
        self._m = (proc, model, dev)

    def __call__(self, rgb_u8: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        if self._m is None:
            self._load()
        proc, model, dev = self._m
        h, w = rgb_u8.shape[:2]
        inputs = proc(images=rgb_u8, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.inference_mode():
            pd = model(**inputs).predicted_depth
        pd = F.interpolate(pd[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)[0, 0]
        return pd.detach().cpu().numpy().astype(np.float32)

    def release(self) -> None:
        self._m = None
        gc.collect()
        with contextlib.suppress(Exception):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class TempDepthReader:
    """RD-07 reader protocol (``snap`` / ``__call__ -> ViewRead``) backed by a disparity function.  Emits only the
    ``log_depth`` head (people = 0, i.e. relative depth 1.0); R6 then scales it to metres."""
    cache_id = "temp-da2s-v1"

    def __init__(self, disparity_fn: Callable[[np.ndarray], np.ndarray], box_ref=PERSON_BOX, ref_size=REF_SIZE):
        self.disparity_fn = disparity_fn
        self.box_ref = box_ref
        self.ref_size = ref_size
        self.calls = 0

    def snap(self, sigma) -> float:
        return float(sigma)

    def __call__(self, image, sigma, *, prov, key):
        import cv2
        import torch
        from buc.read.mrh_call import ViewRead
        self.calls += 1
        x = image.detach().float().clamp(-1.0, 1.0)[0]                      # [-1, 1] display values
        arr = np.ascontiguousarray(((x + 1.0) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
        h, w = arr.shape[:2]
        disp = np.asarray(self.disparity_fn(arr), dtype=np.float32)
        if disp.shape != (h, w):
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
        h4, w4 = max(1, h // 4), max(1, w // 4)
        small = cv2.resize(disp, (w4, h4), interpolation=cv2.INTER_AREA)
        depth_rel = disp_to_depth(small, scale_box(self.box_ref, self.ref_size, (w4, h4)))
        ld = np.log(depth_rel.astype(np.float64)).astype(np.float32)
        if not np.isfinite(ld).all():
            raise ValueError("temp depth: non-finite log depth")
        out = torch.from_numpy(np.ascontiguousarray(ld))[None, None]
        return ViewRead(outputs={"log_depth": out}, sigma=float(sigma),
                        digest=hashlib.sha256(arr.tobytes()).digest()[:16], ephemeral=True)


def make_size_obs_fn(box_ref=PERSON_BOX, ref_size=REF_SIZE) -> Callable:
    """R6 size observations: the two people (hand-annotated box, eval-only) at relative depth 1.0."""
    def fn(state):
        from buc.read.metric_scale import make_obs
        W, H = state.W, state.H
        x0, y0, x1, y1 = scale_box(box_ref, ref_size, (W, H))
        f_px = float(np.asarray(state["R1"]["K"])[0, 0])
        ph = float(max(4, y1 - y0))
        return [make_obs("person", extent_px=ph, depth_rel=1.0, f_px=f_px, sigma_extent=0.15, sigma_depth=0.15,
                         id=f"person{i}") for i in (1, 2)]
    fn.cache_id = "d15-two-people-v1"
    return fn


def build_read_steps(reader, size_obs_fn):
    """RD-21 default steps with R5 pinned to: log_depth head only, one global read, no tiles."""
    from buc.read import pipeline as RP
    steps = list(RP.default_steps(reader=reader, size_obs_fn=size_obs_fn))
    for i, st in enumerate(steps):
        if st.sid == "R5":
            base = st.fn

            def r5(state, params, _b=base):
                return _b(state, {**R5_CFG, **dict(params)})

            steps[i] = RP.ReadStep("R5", st.name, st.version + "+cfg1", r5, st.needs)
    return tuple(steps)


# ---------------------------------------------------------------------------------------------------------------
# teacher (WLD-08) provider
# ---------------------------------------------------------------------------------------------------------------
def hub_cache_candidates() -> List[str]:
    c: List[str] = []
    for k in ("HF_HUB_CACHE",):
        if os.environ.get(k):
            c.append(os.environ[k])
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
    """-> (hub_cache, revision|None).  First candidate cache whose ``refs/main`` snapshot is complete; else the
    newest complete snapshot folder found in any candidate (revision pinned explicitly)."""
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
    raise FileNotFoundError(f"Z-Image-Turbo snapshot not found in any HF cache ({err_text(last) if last else '-'})")


class TeacherUnavailable(RuntimeError):
    """No CUDA / no weights / no diffusers: WLD-08 cannot run on this box (not a bug in WLD-08)."""


class TeacherHandle:
    """Lazy Z-Image teacher + TCH-02 sampler for WLD-08.  ``get()`` raises TeacherUnavailable if CUDA, the HF-cache
    weights or diffusers are missing."""

    def __init__(self) -> None:
        self.sampler = None
        self.teacher = None

    def get(self):
        if self.sampler is not None:
            return self.sampler
        import torch
        from buc.teacher import sample, zimage
        if not torch.cuda.is_available():
            raise TeacherUnavailable("no CUDA: Z-Image teacher not loaded")
        spec = zimage.TeacherSpec()
        try:
            hub, rev = find_teacher_cache(zimage.snapshot_dir, spec)
        except FileNotFoundError as e:
            raise TeacherUnavailable(str(e)) from e
        if rev:
            spec = zimage.TeacherSpec(revision=rev)
        try:
            try:
                teacher = zimage.load_teacher(spec, device="cuda", te_device="cuda", hub_cache=hub)
            except (zimage.TeacherDependencyError, zimage.TeacherWeightsMissing):
                raise
            except Exception as e:  # noqa: BLE001  determinism mode can trip on exotic ops: retry without it once
                log({"warn": "teacher load retry without determinism", "err": err_text(e, 160)})
                teacher = zimage.load_teacher(spec, device="cuda", te_device="cuda", hub_cache=hub,
                                              deterministic=False)
        except (zimage.TeacherDependencyError, zimage.TeacherWeightsMissing) as e:
            raise TeacherUnavailable(err_text(e)) from e
        self.teacher = teacher
        self.sampler = sample.TeacherSampler(teacher, max_views_per_call=2)
        return self.sampler

    def release(self) -> None:
        self.sampler = None
        self.teacher = None
        gc.collect()
        with contextlib.suppress(Exception):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------
# process isolation with timeout (RD-18 closed-form matting is pure scipy and can be slow)
# ---------------------------------------------------------------------------------------------------------------
def _iso_child(conn, fn, args):
    try:
        conn.send(("ok", fn(*args)))
    except BaseException as e:  # noqa: BLE001
        with contextlib.suppress(Exception):
            conn.send(("err", f"{type(e).__name__}: {e}"[:400]))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def run_isolated(fn: Callable, args: tuple, timeout_s: float):
    """Run ``fn(*args)`` in a forked child; raise TimeoutError / RuntimeError on timeout / child failure."""
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_iso_child, args=(child, fn, args), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout_s):
            raise TimeoutError(f"isolated job exceeded {timeout_s:.0f}s")
        kind, val = parent.recv()
    except EOFError:
        raise RuntimeError("isolated job died without a result") from None
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=10)
        parent.close()
    if kind != "ok":
        raise RuntimeError(val)
    return val


def _unmix_job(image_lin, depth):
    from buc.read.edge_unmix import edge_unmix
    return edge_unmix(image_lin, depth)


# ---------------------------------------------------------------------------------------------------------------
# memoised stage functions (a real impl that fails must not be retried for every shot)
# ---------------------------------------------------------------------------------------------------------------
class Memo:
    """Memoise a stage fn (successes and failures).  ``on_error(exc, tb_text)`` runs once, at the first real failure,
    so the exception's message (which the spine's StageIO.fallback deliberately omits) reaches the report."""

    def __init__(self, fn: Callable, on_error: Optional[Callable[[BaseException, str], None]] = None) -> None:
        self.fn = fn
        self.on_error = on_error
        self.store: Dict[str, Tuple[bool, Any]] = {}

    def __call__(self, inputs: dict, ctx) -> dict:
        from buc.core import contracts as C
        key = C.content_hash({"i": {k: v.content_hash() for k, v in inputs.items()},
                              "p": json.dumps({k: v for k, v in dict(ctx.params).items() if k != "_seed"},
                                              sort_keys=True, default=str)})
        if key in self.store:
            ok, val = self.store[key]
            if ok:
                return dict(val)
            raise val
        try:
            out = self.fn(inputs, ctx)
        except Exception as e:  # noqa: BLE001
            self.store[key] = (False, e)
            if self.on_error is not None:
                with contextlib.suppress(Exception):
                    self.on_error(e, scrub(traceback.format_exc()))
            raise
        self.store[key] = (True, out)
        return dict(out)


# ---------------------------------------------------------------------------------------------------------------
# S-1 / S2 / S8 real implementations (spine Impls)
# ---------------------------------------------------------------------------------------------------------------
def subject_depth(read, box_ref=PERSON_BOX, ref_size=REF_SIZE) -> float:
    d = np.asarray(read.depth, dtype=np.float64)
    H, W = d.shape
    x0, y0, x1, y1 = scale_box(box_ref, ref_size, (W, H))
    reg = d[y0:y1, x0:x1]
    reg = reg[np.isfinite(reg)]
    if reg.size == 0:
        reg = d[np.isfinite(d)]
    return float(np.median(reg)) if reg.size else 1.0


def summarize_read(packet) -> dict:
    """What the RD-21 orchestrator actually did (R-step statuses, fallbacks, depth source, scale)."""
    m = packet.meta.get("read", {})
    steps = {sid: v for sid, v in m.get("steps", {}).items() if v.get("impl") is not None}
    return {
        "depth_source": m.get("depth_source"),
        "steps": steps,
        "fallbacks": list(packet.fallbacks),
        "scale_s": float(packet.scale_s),
        "sigma_s": float(packet.sigma_s),
        "f_source": m.get("geom", {}).get("f_source"),
        "fov_x_deg": m.get("geom", {}).get("fov_x_deg"),
        "scale_reliable": m.get("scale", {}).get("reliable"),
        "depth_m": [float(np.min(packet.depth)), float(np.median(packet.depth)), float(np.max(packet.depth))],
    }


def record_read_ledger(ledger: Ledger, summary: dict, shot: Optional[str]) -> None:
    steps = summary["steps"]
    for sid, unit in (("R0", U_R0), ("R1", U_R1), ("R5", U_R5), ("R6", U_R6)):
        st = steps.get(sid, {}).get("status")
        if st == "ok":
            ledger.add(unit, "temp" if sid == "R5" else "real", shot=shot)
        elif st in ("fallback", "skipped"):
            ledger.add(unit, "fallback", f"RD-21 step {sid} {st}", shot=shot)
        else:
            ledger.add(unit, "unavailable", f"no status for {sid}", shot=shot)


TB_TAIL = 1200


def record_stage_error(trace: dict, ledger: Optional[Ledger], stage: str, units: Sequence[str], e: BaseException,
                       tb_text: str, shot: Optional[str] = None) -> dict:
    """A real stage raised: keep the exception text + traceback tail in ``trace["errors"][stage]`` and one ledger
    'fallback' event per unit carrying the message (the spine itself only records the exception type)."""
    rec = {"error": err_text(e, 400), "tb": scrub(tb_text)[-TB_TAIL:], "units": list(units), "shot": shot}
    trace.setdefault("errors", {})[stage] = rec
    if ledger is not None:
        for u in units:
            ledger.add(u, "fallback", f"{stage} raised {err_text(e, 240)}", shot)
    return rec


def make_s1_impl(reader, size_obs_fn, trace: dict, ledger: Ledger):
    from buc.pipeline import spine as S
    from buc.read import pipeline as RP
    base = RP.make_read_impl(steps=build_read_steps(reader, size_obs_fn))
    memo = Memo(base.fn, on_error=lambda e, tb: record_stage_error(trace, ledger, "S-1", S1_UNITS, e, tb))

    def fn(inputs, ctx):
        out = memo(inputs, ctx)
        if "S-1" not in trace:
            trace["S-1"] = summarize_read(out["read"])
            record_read_ledger(ledger, trace["S-1"], None)
        return out

    return S.Impl(stage="S-1", kind="real", name=base.name, version=base.version, fn=fn)


def build_world_chain(read, *, teacher: Optional[TeacherHandle], cfg: dict, trace: list, ledger: Ledger,
                      isolate: Callable = run_isolated):
    """RD-18 matte (if confident) -> WLD-03 depth world -> WLD-08 SDEdit-average fill (if the teacher is up).
    Each optional sub-stage falls back to the one before it; WLD-03 failing propagates (spine then uses the stub)."""
    from buc.world.depth_world import build_depth_world
    matte = None
    if cfg.get("use_matte", True):
        try:
            um = isolate(_unmix_job, (np.asarray(read.image_lin), np.asarray(read.depth)), cfg.get("matte_timeout_s", 240.0))
            if um.low_conf and not cfg.get("force_matte"):
                trace.append({"unit": U_MATTE, "status": "rejected", "detail": f"low_conf {um.meta()}"[:300]})
                ledger.add(U_MATTE, "rejected", "matte low_conf: not used", None)
            else:
                matte = um
                trace.append({"unit": U_MATTE, "status": "real", "detail": um.meta()})
                ledger.add(U_MATTE, "real")
        except Exception as e:  # noqa: BLE001
            trace.append({"unit": U_MATTE, "status": "fallback", "detail": err_text(e)})
            ledger.add(U_MATTE, "fallback", err_text(e))
    else:
        ledger.add(U_MATTE, "skipped", "disabled")
    try:
        dw = build_depth_world(read, matte=matte)
    except Exception as e:  # noqa: BLE001
        if matte is None:
            trace.append({"unit": U_W3, "status": "failed", "detail": err_text(e, 400)})
            raise
        trace.append({"unit": U_W3, "status": "fallback", "detail": "with matte: " + err_text(e)})
        matte = None
        try:
            dw = build_depth_world(read)
        except Exception as e2:  # noqa: BLE001
            trace.append({"unit": U_W3, "status": "failed", "detail": "without matte: " + err_text(e2, 400)})
            raise
    ledger.add(U_W3, "real", f"matte={'yes' if matte is not None else 'no'}")
    trace.append({"unit": U_W3, "status": "real", "detail": dw.meta()})
    world = dw.world
    if teacher is None:
        ledger.add(U_W8, "unavailable", "teacher disabled")
        trace.append({"unit": U_W8, "status": "unavailable", "detail": "teacher disabled"})
        return world
    try:
        from buc.world.sdedit_avg import sdedit_avg_fill
        sampler = teacher.get()
        res = sdedit_avg_fill(dw, sampler, seed=int(cfg.get("seed", 0)))
        world = res.world
        ledger.add(U_W8, "real")
        trace.append({"unit": U_W8, "status": "real", "detail": res.meta()})
    except Exception as e:  # noqa: BLE001
        unavailable = isinstance(e, TeacherUnavailable)
        ledger.add(U_W8, "unavailable" if unavailable else "fallback", err_text(e))
        trace.append({"unit": U_W8, "status": "unavailable" if unavailable else "fallback", "detail": err_text(e)})
    finally:
        teacher.release()
    return world


def make_s2_impl(teacher: Optional[TeacherHandle], cfg: dict, trace: dict, ledger: Ledger):
    from buc.pipeline import spine as S

    def inner(inputs, ctx):
        tr: list = trace.setdefault("S2", [])
        return {"world": build_world_chain(inputs["read"], teacher=teacher, cfg=cfg, trace=tr, ledger=ledger)}

    memo = Memo(inner, on_error=lambda e, tb: record_stage_error(trace, ledger, "S2", (U_W3,), e, tb))
    return S.Impl(stage="S2", kind="real", name="world_chain", version="0.1.0+wld03-wld08-rd18", fn=memo)


def _c0_of(read):
    return np.asarray(read.obs.c2w(), dtype=np.float64)


def _shot_from_poses(read, c2w, t, focus_m, K, move, rig, seed, theta=None):
    from buc.core import contracts as C
    obs = read.obs
    return C.Shot(c2w=c2w, t=t, K=K, focus_m=focus_m, width=int(obs.width), height=int(obs.height), fps=float(FPS),
                  theta=read.theta_in if theta is None else theta, move=move, rig=rig, seed=int(seed))


def people_ray(read, box_ref=PERSON_BOX, ref_size=REF_SIZE) -> dict:
    """The people-box centre as a ray in the observed camera frame: pixel (u, v) and (dx, dy) = (u-cx)/fx, (v-cy)/fy,
    so the 3D point at depth ``z`` is ``z * (dx, dy, 1)``."""
    obs = read.obs
    W, H = int(obs.width), int(obs.height)
    x0, y0, x1, y1 = scale_box(box_ref, ref_size, (W, H))
    K = np.asarray(obs.K, dtype=np.float64)
    u, v = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return {"box": (x0, y0, x1, y1), "uv": (u, v), "dx": (u - K[0, 2]) / K[0, 0], "dy": (v - K[1, 2]) / K[1, 1],
            "W": W, "H": H, "K": K}


def aim_push_at_people(read, c0: np.ndarray, c2w: np.ndarray, z: float, m: float, info: dict):
    """CAM-02 push_in moves along the optical axis; here the same advance (``adv_k`` along the camera's forward axis)
    is taken along the ray through the people-box centre instead: the camera translates by adv_k * (dx, dy, 1) in
    its own observed frame, rotation unchanged.  Frame 0 stays the observed pose (adv_0 = 0), the people-box centre
    keeps its pixel, and the box grows ~m about it.  -> (c2w', focus_m' = distance to the people point)."""
    ray = people_ray(read)
    R0, fwd = c0[:3, :3], c0[:3, 2]
    out = np.array(c2w, dtype=np.float64, copy=True)
    adv = (out[:, :3, 3] - c0[:3, 3][None, :]) @ fwd
    local = adv[:, None] * np.array([ray["dx"], ray["dy"], 0.0])[None, :]
    out[:, :3, 3] += local @ R0.T
    target = (c0 @ np.array([z * ray["dx"], z * ray["dy"], z, 1.0]))[:3]
    focus = np.linalg.norm(out[:, :3, 3] - target[None, :], axis=1)
    # where the people box ends up in the last frame (box grows ~m about its fixed centre)
    x0, y0, x1, y1 = ray["box"]
    u, v = ray["uv"]
    ex = (u + (x0 - u) * m, v + (y0 - v) * m, u + (x1 - u) * m, v + (y1 - v) * m)
    W, H = ray["W"], ray["H"]
    vis = (min(ex[2], W) - max(ex[0], 0.0)) * (min(ex[3], H) - max(ex[1], 0.0))
    info["push_target"] = {"what": "people-box centre (ray, not optical axis)", "px": [round(u, 1), round(v, 1)],
                           "offset_from_centre_px": [round(u - W / 2.0, 1), round(v - H / 2.0, 1)],
                           "m": m, "people_box_end_px": [round(c, 1) for c in ex],
                           "people_box_visible_frac_end": round(max(0.0, vis) / max(1e-9, (ex[2] - ex[0]) * (ex[3] - ex[1])), 3)}
    return out, focus


def orbit_pivot_note(read, z: float) -> dict:
    """S3 arc: CAM-02 pivots about the optical-axis point at the people's depth (not about the people).  Logged."""
    ray = people_ray(read)
    return {"where": "optical axis at the people depth (CAM-02 orbit pivot), NOT the people point",
            "people_lateral_offset_px": round(ray["uv"][0] - ray["W"] / 2.0, 1),
            "people_lateral_offset_m": round(ray["dx"] * z, 4)}


def build_shot(read, name: str, dur_s: float, seed: int, info: dict):
    """Real S8: CAM-02 recipes realised on CAM-01 rigs.  Fills ``info`` (recipe actually used, fallbacks)."""
    from buc.camera import moves, rigs
    d = SHOT_DEFS[name]
    mv, kw = d["move"], d["kw"]
    z = subject_depth(read)
    c0 = _c0_of(read)
    K0 = np.asarray(read.obs.K, dtype=np.float64)
    info.update({"requested": mv, "z_s": z, "dur_s": dur_s, "kw": {k: (list(v) if isinstance(v, tuple) else v)
                                                                       for k, v in kw.items()}})

    def finish_cam02(spec):
        try:
            path = moves.rig_path(spec, c0, seed=seed)
            info["rig"] = path.rig
        except Exception as e:  # noqa: BLE001
            info["rig_fallback"] = err_text(e)
            path = moves.ideal_path(spec, c0)
            info["rig"] = "ideal"
        n = path.n_frames
        c2w, focus = path.c2w, path.focus_m
        if spec.name == "push_in":
            c2w, focus = aim_push_at_people(read, c0, c2w, z, spec.m, info)
        K = np.broadcast_to(K0, (n, 3, 3)).copy()
        return _shot_from_poses(read, c2w, path.t, focus, K, spec.name, path.rig, seed)

    if mv in ("push_in", "slider", "orbit"):
        spec = moves.recipe(mv, z, duration_s=dur_s, fps=float(FPS), **kw)
        info["source"] = "CAM-02"
        if mv == "push_in":
            info["aim"] = "dolly advance taken along the ray through the people-box centre (not the optical axis)"
        info["used"] = mv
        if mv == "orbit":
            info["note"] = "arc runs 0 -> +%.0f deg: S8 contract needs frame 0 at the observed pose (docs: +-12)" % kw["orbit_deg"]
            info["pivot"] = orbit_pivot_note(read, z)
        return finish_cam02(spec)

    if mv == "rack_focus":
        stops = [z * s for s in kw["stops"]]
        half = max(1, int(round(dur_s * FPS / 2.0)))
        seg = half / float(FPS)
        segs = []
        for a, b in zip(stops[:-1], stops[1:]):
            spec = moves.recipe("rack_focus", z, duration_s=seg, fps=float(FPS), focus_from_m=a, focus_to_m=b)
            segs.append(moves.ideal_path(spec, c0))
        focus = np.concatenate([segs[0].focus_m] + [s.focus_m[1:] for s in segs[1:]])
        c2w = np.broadcast_to(c0, (focus.shape[0], 4, 4)).copy()
        t = np.arange(focus.shape[0], dtype=np.float64) / FPS
        K = np.broadcast_to(K0, (focus.shape[0], 3, 3)).copy()
        info.update({"source": "CAM-02", "used": "rack_focus x2 (poppy->people->eye)", "focus_stops_m": stops})
        theta = dataclasses.replace(read.theta_in, f_stop=S5_F_STOP, focus_m=float(stops[0]))
        return _shot_from_poses(read, c2w, t, focus, K, "rack_focus", "locked", seed, theta=theta)

    if mv == "crane":
        elev = math.radians(float(kw["elev_deg"]))
        n_expect = int(round(dur_s * FPS)) + 1
        poses = None
        for sign in (-1.0, 1.0):
            rig = rigs.make_rig("crane_jib", geom={"radius": z}, home_c2w=c0)
            intent = rigs.Intent(np.array([0.0, dur_s]), np.array([[0.0, 0.0], [0.0, sign * elev]]), 0.0)
            traj = rig.simulate(intent, seed=seed)
            p = np.array(traj.frame_poses(float(FPS)), dtype=np.float64)
            p[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
            if p[-1, 1, 3] < p[0, 1, 3]:                      # OpenCV y points down: up means y decreases
                poses = p
                break
        if poses is None or poses.shape[0] != n_expect:
            raise RuntimeError(f"crane rig produced {None if poses is None else poses.shape[0]} frames, want {n_expect}")
        subject = (c0 @ np.array([0.0, 0.0, z, 1.0]))[:3]
        focus = np.linalg.norm(poses[:, :3, 3] - subject[None, :], axis=1)
        K = np.broadcast_to(K0, (n_expect, 3, 3)).copy()
        info.update({"source": "CAM-01 crane_jib + local intent (CAM-02 has no crane recipe)", "used": "crane_up"})
        return _shot_from_poses(read, poses, np.arange(n_expect) / FPS, focus, K, "crane_up", "crane_jib", seed)

    if mv == "dolly_zoom":
        n = int(round(dur_s * FPS)) + 1
        s = moves.ease_profile(np.arange(n, dtype=np.float64) / (n - 1))
        back = float(kw["back_frac"]) * z * s
        c2w = np.repeat(c0[None], n, axis=0)
        c2w[:, :3, 3] = c0[:3, 3][None, :] - back[:, None] * c0[:3, 2][None, :]
        K = np.repeat(K0[None], n, axis=0)
        ratio = (z + back) / z                                 # people keep their pixel size
        K[:, 0, 0] *= ratio
        K[:, 1, 1] *= ratio
        info.update({"source": "local composition (CAM-02 has no dolly zoom): push_in(m<1) path + K scaled",
                     "used": "dolly_zoom"})
        return _shot_from_poses(read, c2w, np.arange(n) / FPS, z + back, K, "dolly_zoom", "ideal", seed)

    raise ValueError(f"unknown move {mv!r}")


def make_s8_impl(shot_name: str, dur_s: float, seed: int, trace: dict, ledger: Optional[Ledger] = None):
    from buc.pipeline import spine as S
    tr = trace.setdefault("S8", {})

    def inner(inputs, ctx):
        info: dict = {}
        tr[shot_name] = info
        try:
            shot = build_shot(inputs["read"], shot_name, dur_s, seed, info)
        except Exception as e:  # noqa: BLE001
            info["error"] = err_text(e, 400)
            info["tb"] = scrub(traceback.format_exc())[-TB_TAIL:]
            trace.setdefault("errors", {})[f"S8:{shot_name}"] = {"error": info["error"], "tb": info["tb"],
                                                                 "units": [U_CAM], "shot": shot_name}
            if ledger is not None:
                ledger.add(U_CAM, "fallback", f"S8 raised {err_text(e, 240)}", shot_name)
            raise
        return {"shot": shot}

    version = "0.1.0+" + hashlib.sha1(json.dumps([shot_name, dur_s, SHOT_DEFS[shot_name]], default=str,
                                                 sort_keys=True).encode()).hexdigest()[:10]
    return S.Impl(stage="S8", kind="real", name="cam02_moves", version=version, fn=inner)


# ---------------------------------------------------------------------------------------------------------------
# frame production: overlay, stub (homography) renderer, DoF fallback
# ---------------------------------------------------------------------------------------------------------------
def overlay_label(frame_u8: np.ndarray, text: str) -> np.ndarray:
    import cv2
    h, w = frame_u8.shape[:2]
    scale = max(0.35, 0.8 * w / 1920.0)
    th = max(1, int(round(2 * w / 1920.0)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    pad = max(3, int(round(10 * w / 1920.0)))
    x0, y0 = pad, pad
    x1, y1 = min(w, x0 + tw + 2 * pad), min(h, y0 + tht + base + 2 * pad)
    roi = frame_u8[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * 0.35).astype(np.uint8)
    cv2.putText(frame_u8, text, (x0 + pad, y0 + pad + tht), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th,
                cv2.LINE_AA)
    return frame_u8


def to_display_u8(rgb_lin: np.ndarray) -> np.ndarray:
    from buc.core import color
    d = color.linear_to_display(np.asarray(rgb_lin, dtype=np.float32))
    return np.ascontiguousarray(np.rint(np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8))


def homography_frame(src_u8: np.ndarray, obs, c2w_i: np.ndarray, K_i: np.ndarray, z0: float) -> np.ndarray:
    """Stub renderer: the image as a fronto-parallel plane at depth ``z0`` seen from pose ``c2w_i`` (cv2 only)."""
    import cv2
    h, w = src_u8.shape[:2]
    Ko = np.asarray(obs.K, dtype=np.float64)
    T = np.linalg.inv(np.asarray(obs.c2w(), dtype=np.float64)) @ np.asarray(c2w_i, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    n = np.array([[0.0, 0.0, 1.0]])
    M = np.linalg.inv(np.eye(3) - (t[:, None] @ n) / z0)
    Hm = Ko @ M @ R @ np.linalg.inv(np.asarray(K_i, dtype=np.float64))     # target pixel -> source pixel
    return cv2.warpPerspective(src_u8, Hm, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def simple_dof(rgb_lin: np.ndarray, depth: np.ndarray, focus_m: float, max_r: float = 24.0, k: float = 8.0,
               layers: int = 8) -> np.ndarray:
    """DoF fallback: depth layers blurred with a Gaussian of radius k*|1/z - 1/zf|, composited back to front."""
    import cv2
    z = np.where(np.isfinite(depth), depth, 1e6).astype(np.float64)
    inv = 1.0 / np.maximum(z, 1e-3)
    r = np.minimum(max_r, k * np.abs(inv - 1.0 / focus_m) * focus_m)
    edges = np.linspace(0.0, max_r, layers + 1)
    out = np.zeros_like(rgb_lin, dtype=np.float32)
    wsum = np.zeros(rgb_lin.shape[:2], dtype=np.float32)
    for j in range(layers):
        m = ((r >= edges[j]) & (r < edges[j + 1] if j < layers - 1 else r <= edges[j + 1])).astype(np.float32)
        if not m.any():
            continue
        sig = 0.5 * (edges[j] + edges[j + 1])
        b = rgb_lin * m[..., None]
        if sig > 0.3:
            b = cv2.GaussianBlur(b, (0, 0), sig)
            m = cv2.GaussianBlur(m, (0, 0), sig)
        out += b
        wsum += m
    return (out / np.maximum(wsum, 1e-6)[..., None]).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# mp4 writer with fallbacks
# ---------------------------------------------------------------------------------------------------------------
def _ffmpeg_exe() -> Optional[str]:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    with contextlib.suppress(Exception):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    return None


class OutVideo:
    """RDR-06 ``VideoWriter`` first, then imageio, then cv2.  ``impl`` tells which one produced the file."""

    def __init__(self, path: Path, width: int, height: int, fps: float = FPS, prefer: Sequence[str] = ("rdr06", "imageio", "cv2")):
        self.path, self.w, self.h, self.fps = Path(path), int(width), int(height), fps
        self.impl = None
        self.errors: List[str] = []
        self._w = None
        self.n = 0
        for kind in prefer:
            try:
                self._w = getattr(self, "_open_" + kind)()
                self.impl = kind
                break
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{kind}: {err_text(e, 160)}")
        if self._w is None:
            raise RuntimeError("no video writer available: " + " | ".join(self.errors))

    def _open_rdr06(self):
        from buc.render import video
        return video.VideoWriter(self.path, width=self.w, height=self.h, fps=self.fps, ffmpeg=_ffmpeg_exe())

    def _open_imageio(self):
        import imageio
        return imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=None, macro_block_size=None,
                                  ffmpeg_params=["-crf", "18"])

    def _open_cv2(self):
        import cv2
        vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), float(self.fps), (self.w, self.h))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        return vw

    def write(self, frame_u8: np.ndarray) -> None:
        if self.impl == "rdr06":
            self._w.write(frame_u8)
        elif self.impl == "imageio":
            self._w.append_data(frame_u8)
        else:
            self._w.write(np.ascontiguousarray(frame_u8[..., ::-1]))
        self.n += 1

    def close(self) -> None:
        if self.impl == "rdr06":
            self._w.close()
        elif self.impl == "imageio":
            self._w.close()
        else:
            self._w.release()

    def abort(self) -> None:
        with contextlib.suppress(Exception):
            if self.impl == "rdr06":
                self._w.__exit__(RuntimeError, RuntimeError("abort"), None)
            elif self.impl == "imageio":
                self._w.close()
            else:
                self._w.release()
        with contextlib.suppress(Exception):
            self.path.unlink()


# ---------------------------------------------------------------------------------------------------------------
# shot rendering (S9-10) with a fork pool
# ---------------------------------------------------------------------------------------------------------------
_JOB: Dict[str, Any] = {}


def _pool_init() -> None:
    with contextlib.suppress(Exception):
        import cv2
        cv2.setNumThreads(0)


def _pool_job(i: int):
    fn = _JOB["frame_fn"]
    try:
        frame, note = fn(i)
        return i, "ok", frame, note
    except Exception as e:  # noqa: BLE001
        return i, "err", None, err_text(e)


def make_frame_fns(world, read, shot, *, label: str, lens: Optional[dict], src_u8: np.ndarray, z0: float):
    """-> (real(i) -> (uint8 frame, note), stub(i) -> (uint8 frame, note), prime()).

    ``real`` is RDR-02 ``render_ldi``; with ``lens`` (locked camera, rack focus) the world is rendered once in
    ``prime()`` and every frame gets LNS-02 ``apply_dof`` (a layered-blur fallback on error, reported in ``note``).
    ``stub`` is the plain homography renderer."""
    from buc.render import quads
    obs = world.obs
    W, H = int(shot.width), int(shot.height)
    k1, k2 = float(obs.k1), float(obs.k2)
    base: Dict[str, Any] = {}

    def render_lin(i):
        return quads.render_ldi(world, c2w=shot.c2w[i], K=shot.K[i], width=W, height=H, k1=k1, k2=k2)

    def finish(f):
        return overlay_label(f, label) if label else f

    def prime():
        if lens is None:
            return None
        try:
            base["o"] = render_lin(0)
            base["note"] = None
        except Exception as e:  # noqa: BLE001
            from buc.core import color
            lin = color.srgb_to_linear(src_u8.astype(np.float32) / 255.0).astype(np.float32)
            base["o"] = type("O", (), {"rgb": lin, "depth": np.asarray(read.depth, np.float32)})()
            base["note"] = "base render failed -> source image + read depth: " + err_text(e, 120)
        return base["note"]

    def real(i):
        if lens is None:
            return finish(to_display_u8(render_lin(i).rgb)), None
        if "o" not in base:
            prime()
        o = base["o"]
        sc = lens["depth_scale"]
        depth = np.where(np.isfinite(o.depth), np.asarray(o.depth) * sc, np.inf).astype(np.float32)
        th = shot.theta
        focus = float(shot.focus_m[i]) * sc
        try:
            from buc.lens import dof
            r = dof.apply_dof(o.rgb, depth, focal_mm=th.focal_mm, f_stop=th.f_stop, focus_m=focus,
                              sensor_width_mm=th.sensor_width_mm, opts=dof.DofOpts(max_coc_px=lens["max_coc_px"]))
            return finish(to_display_u8(r.rgb)), base.get("note")
        except Exception as e:  # noqa: BLE001
            note = "dof fallback: " + err_text(e, 160)
            return finish(to_display_u8(simple_dof(np.asarray(o.rgb, np.float32), depth, focus,
                                                   max_r=lens["max_coc_px"] * 0.5))), note

    def stub(i):
        return finish(homography_frame(src_u8, obs, shot.c2w[i], shot.K[i], z0)), "stub homography"

    return real, stub, prime


def render_shot_video(world, read, shot, out_path: Path, *, label: str, lens: Optional[dict], src_u8: np.ndarray,
                      z0: float, workers: int, deadline: Deadline, thumbs: Optional[list] = None,
                      progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Render every frame of ``shot`` (real RDR-02, per-frame stub fallback), stream them into an mp4.
    Returns a dict with counts; raises on deadline / writer failure (the caller records the shot as failed)."""
    n = shot.n_frames
    real, stub, prime = make_frame_fns(world, read, shot, label=label, lens=lens, src_u8=src_u8, z0=z0)
    out = OutVideo(out_path, int(shot.width), int(shot.height), float(shot.fps))
    stats = {"frames": 0, "render_real": 0, "render_stub": 0, "render_errors": [], "notes": [], "pool": False,
             "encoder": out.impl, "encoder_errors": list(out.errors)}
    t0 = time.time()
    last_prog = t0
    keep = {0, n // 2, n - 1}
    ok = False

    def note(msg):
        if msg and len(stats["notes"]) < 5 and msg not in stats["notes"]:
            stats["notes"].append(msg)

    def emit(i, frame):
        nonlocal last_prog
        out.write(frame)
        stats["frames"] += 1
        if thumbs is not None and i in keep:
            import cv2
            tw = 480
            th_ = max(2, int(round(frame.shape[0] * tw / frame.shape[1])))
            thumbs.append((i, cv2.resize(frame, (tw, th_), interpolation=cv2.INTER_AREA)))
        now = time.time()
        if progress and now - last_prog > 60:
            progress(stats["frames"], n)
            last_prog = now

    def one(i):
        try:
            f, nt = real(i)
            stats["render_real"] += 1
            note(nt)
            return f
        except Exception as e:  # noqa: BLE001
            if len(stats["render_errors"]) < 5:
                stats["render_errors"].append(err_text(e, 200))
            stats["render_stub"] += 1
            return stub(i)[0]

    try:
        pn = prime()
        note(pn)
        done = 0
        if workers > 1 and n > 2:
            _JOB["frame_fn"] = real
            pool = None
            try:
                ctx = mp.get_context("fork")
                pool = ctx.Pool(workers, initializer=_pool_init)
                stats["pool"] = True
                it = pool.imap(_pool_job, range(n), chunksize=1)
                while done < n:
                    if deadline.stop_now():
                        raise TimeoutError("deadline reached while rendering")
                    i, kind, frame, nt = it.next(timeout=FRAME_TIMEOUT_S)
                    if kind == "ok":
                        stats["render_real"] += 1
                        note(nt)
                    else:
                        if len(stats["render_errors"]) < 5:
                            stats["render_errors"].append(nt)
                        stats["render_stub"] += 1
                        frame = stub(i)[0]
                    emit(i, frame)
                    done += 1
            except mp.TimeoutError:
                stats["pool_dropped"] = f"no frame for {FRAME_TIMEOUT_S:.0f}s at frame {done}"
            finally:
                if pool is not None:
                    pool.terminate()
                    pool.join()
                _JOB.pop("frame_fn", None)
        while done < n:
            if deadline.stop_now():
                raise TimeoutError("deadline reached while rendering")
            emit(done, one(done))
            done += 1
        out.close()
        ok = True
    finally:
        if not ok:
            out.abort()
    stats["wall_s"] = round(time.time() - t0, 1)
    stats["bytes"] = Path(out_path).stat().st_size if Path(out_path).exists() else 0
    return stats


def plan_width(widths: Sequence[int], s_per_px: float, frame_equiv: float, workers: int,
               budget_s: Optional[float]) -> int:
    """Largest width whose estimated render time fits ``budget_s`` (None = unlimited -> the largest)."""
    ws = sorted(set(int(w) for w in widths), reverse=True)
    if budget_s is None:
        return ws[0]
    for w in ws:
        h = out_size_for(w)[1]
        est = s_per_px * w * h * frame_equiv / max(1, workers)
        if est <= budget_s:
            return w
    return ws[-1]


def bench_s_per_px(trial_w: int = 320) -> float:
    """Seconds per frame per pixel of render_ldi on a tiny 3-layer world (used only to pick the width)."""
    from buc.pipeline import spine as S
    from buc.pipeline.stubs import pack
    from buc.render import quads
    w, h = out_size_for(trial_w)
    rs = np.random.RandomState(0)
    img = make_input_image((rs.rand(h, w, 3) * 255).astype(np.uint8), "bench")
    reg = S.Registry()
    pack.register_stubs(reg)
    r = S.Spine(reg, prefer="stub").run(img, until="S8.5", params={"S8": {"n_frames": 3}})
    shot, world = r.artifacts["shot"], r.artifacts["world"]
    t = time.time()
    quads.render_ldi(world, c2w=shot.c2w[1], K=shot.K[1], width=w, height=h)
    return (time.time() - t) / float(w * h)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def stage_rows(res, first_run: Optional[dict] = None) -> dict:
    rows = {}
    for st, io in res.stage_ios.items():
        status = res.status[st]
        row = {"impl": f"{io.impl_kind}:{io.impl_name}@{io.impl_version}", "kind": io.impl_kind,
               "status": status, "fallback": io.fallback, "wall_s": round(float(io.wall_s), 2)}
        if status == "hit" and first_run and st in first_run:
            row["first_run"] = first_run[st]
        rows[st] = row
    return rows


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def host_info() -> dict:
    d: Dict[str, Any] = {"python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    with contextlib.suppress(Exception):
        import torch
        d["torch"] = torch.__version__
        d["cuda"] = bool(torch.cuda.is_available())
        if d["cuda"]:
            d["gpu"] = torch.cuda.get_device_name(0)
    return d


REAL_STAGE_MODULES = ("buc.read.pipeline", "buc.world.depth_world", "buc.world.sdedit_avg", "buc.render.quads",
                      "buc.camera.moves", "buc.lens.dof", "buc.read.edge_unmix", "buc.teacher.zimage")


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def repo_info(root: Optional[str] = None) -> dict:
    """Which copy of the repo this run really imported, so a stale-code fallback (stage module missing from an old
    tarball) can be told apart from a real stage failure: buc location, git/tarball id when known, short content
    hashes of the real-stage files, and whether each real-stage module is importable at all."""
    import importlib.util
    r = root or os.environ.get("N1_ROOT") or DEFAULT_ROOT
    d: Dict[str, Any] = {"script": str(Path(__file__).resolve()), "script_sha256": _sha256_file(Path(__file__)),
                         "n1_repo_dir": str(Path(r) / "repo")}
    for k in ("N1_REPO_SHA", "N1_REPO_COMMIT", "N1_REPO_TARBALL_SHA"):
        if os.environ.get(k):
            d[k.lower()] = scrub(os.environ[k])[:80]
    buc_dir = None
    with contextlib.suppress(Exception):
        spec = importlib.util.find_spec("buc")
        if spec and spec.submodule_search_locations:
            buc_dir = Path(list(spec.submodule_search_locations)[0])
    if buc_dir is None:
        d["buc"] = None
        return d
    d["buc"] = str(buc_dir)
    repo_dir = buc_dir.parent
    d["buc_in_n1_repo"] = str(repo_dir.resolve()) == str((Path(r) / "repo").resolve())
    with contextlib.suppress(Exception):
        import subprocess
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            d["git_head"] = out.stdout.strip()[:40]
    for name in (".repo_sha", "REPO_SHA", ".git_sha"):
        t = repo_dir / name
        if t.is_file():
            with contextlib.suppress(OSError):
                d["repo_sha_file"] = t.read_text().strip()[:80]
    d["file_sha"] = {rel: _sha256_file(repo_dir / rel) for rel in (
        "buc/read/pipeline.py", "buc/world/depth_world.py", "buc/world/sdedit_avg.py", "buc/camera/moves.py")}
    mods: Dict[str, bool] = {}
    for m in REAL_STAGE_MODULES:
        try:
            mods[m] = importlib.util.find_spec(m) is not None
        except Exception:  # noqa: BLE001
            mods[m] = False
    d["stage_modules"] = mods
    d["stale_code_suspected"] = not all(mods.values())
    return d


def new_report(args, size, image_info: dict, shim: bool) -> dict:
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job": "pipe_yak (D15 hero yak through buc/ pipeline)",
        "lane": "R-EVAL (eval-only): temp depth, prov HOLDOUT:r_eval, nothing written to training-data paths",
        "host": {**host_info(), "repo": repo_info(getattr(args, "root", None))},
        "blake3_shim": bool(shim),
        "image": {k: v for k, v in image_info.items() if k != "array"},
        "config": {"width": size[0], "height": size[1], "fps": FPS, "modes": args.modes, "shots": args.shots,
                   "dur_scale": args.dur_scale, "workers": args.workers, "depth_model": DEPTH_MODEL},
        "skeleton": {"status": "not_run"},
        "scaffold": {"status": "not_run", "shots": {}},
        "units": {},
        "errors": [],
        "outputs": [],
        "no_depth_artifacts_written": True,
    }


# ---------------------------------------------------------------------------------------------------------------
# contact sheet / upload
# ---------------------------------------------------------------------------------------------------------------
def make_contact_sheet(thumbs: Dict[str, list], out_path: Path) -> bool:
    import cv2
    rows = [(k, sorted(v, key=lambda t: t[0])) for k, v in thumbs.items() if v]
    if not rows:
        return False
    tw = 480
    th = max(t[1].shape[0] for _, v in rows for t in v)
    lab_h = 22
    canvas = np.zeros((len(rows) * (th + lab_h), 3 * tw, 3), np.uint8)
    for r, (name, v) in enumerate(rows):
        y = r * (th + lab_h)
        cv2.putText(canvas, name, (6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for c, (_, img) in enumerate(v[:3]):
            hh, ww = img.shape[:2]
            canvas[y + lab_h:y + lab_h + hh, c * tw:c * tw + ww] = img
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return bool(ok)


def upload_outputs(paths: Sequence[Path], token: Optional[str], api=None, repo_id: str = HF_REPO_ID,
                   prefix: str = HF_UPLOAD_PREFIX) -> dict:
    """Upload each file to the vault; failures are warnings.  Only files in ``paths`` (videos, report, sheet)."""
    if not token:
        log({"upload": "skipped", "hf_token": "no"})
        return {"ok": 0, "failed": 0, "skipped": True}
    if api is None:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        except Exception as e:  # noqa: BLE001
            log({"warn": "hf api unavailable", "err": err_text(e, 200)})
            return {"ok": 0, "failed": len(paths), "skipped": False, "uploaded": []}
    ok = failed = 0
    uploaded: List[str] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{prefix}/{p.name}", repo_id=repo_id,
                            repo_type=HF_REPO_TYPE)
            ok += 1
            uploaded.append(p.name)
            log({"uploaded": p.name})
        except Exception as e:  # noqa: BLE001
            failed += 1
            log({"warn": "upload failed", "file": p.name, "err": err_text(e, 200)})
    return {"ok": ok, "failed": failed, "skipped": False, "uploaded": uploaded}


class Uploader:
    """Incremental vault upload: every finished mp4 goes up right away together with a fresh stages_report.json, so a
    killed run (OOM, runner restart, hard stop) still leaves the finished videos and the report in the vault.
    Cumulative ok / failed counts end up in ``report["upload"]``."""

    def __init__(self, token: Optional[str], api=None, enabled: bool = True) -> None:
        self.token, self.api, self.enabled = token, api, enabled
        self.ok = self.failed = 0
        self.done: List[str] = []
        self.skipped = False
        self._tried = False

    def _api(self):
        if self.api is None and not self._tried and self.token:
            self._tried = True
            try:
                from huggingface_hub import HfApi
                self.api = HfApi(token=self.token)
            except Exception as e:  # noqa: BLE001
                log({"warn": "hf api unavailable", "err": err_text(e, 200)})
        return self.api

    def push(self, paths: Sequence[Path]) -> dict:
        if not self.enabled:
            return {"ok": 0, "failed": 0, "skipped": True, "uploaded": []}
        r = upload_outputs(paths, self.token, api=self._api())
        self.skipped = bool(r.get("skipped"))
        self.ok += r["ok"]
        self.failed += r["failed"]
        self.done += r.get("uploaded", [])
        return r

    def stats(self) -> dict:
        return {"ok": self.ok, "failed": self.failed, "skipped": self.skipped or not self.enabled,
                "files": sorted(set(self.done))}


def safe_out_dir(root: Path) -> Path:
    """Outputs only ever go to ``<root>/day1/pipe`` (never next to img/ lat/ meta/ corpus)."""
    d = Path(root).joinpath(*OUT_SUBDIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------
# mode A: skeleton
# ---------------------------------------------------------------------------------------------------------------
def run_skeleton(image_u8: np.ndarray, out_dir: Path, report: dict, ledger: Ledger, deadline: Deadline,
                 thumbs: Dict[str, list], *, workers: int, dur_scale: float = 1.0) -> Optional[Path]:
    """PRD-01 spine with ONLY the PRD-03 stub pack -> RDR-02 render -> skeleton_S1.mp4."""
    from buc.pipeline import spine as S
    from buc.pipeline.stubs import pack
    rec = report["skeleton"] = {"status": "running"}
    Phase.name = "skeleton"
    t0 = time.time()
    try:
        img = make_input_image(image_u8, "demo/d15_hero_yak.webp")
        reg = S.Registry()
        pack.register_stubs(reg)
        spine = S.Spine(reg, prefer="stub", allow_fallback=False)
        dur = max(4.0 / FPS, round(SHOT_DEFS["S1"]["dur"] * dur_scale * FPS) / FPS)
        n = int(round(dur * FPS)) + 1
        res = spine.run(img, seed=0, until="S8.5", params={"S8": {"n_frames": n, "fps": float(FPS), "push_frac": 0.25}})
        rec["stages"] = stage_rows(res)
        rec["spine_root"] = res.root
        read, world, shot = res.artifacts["read"], res.artifacts["world"], res.artifacts["shot"]
        z0 = subject_depth(read)
        out_path = out_dir / "skeleton_S1.mp4"
        Phase.name = "skeleton-render"
        st = render_shot_video(world, read, shot, out_path, label=SKELETON_LABEL, lens=None, src_u8=image_u8, z0=z0,
                               workers=workers, deadline=deadline, thumbs=thumbs.setdefault("skeleton_S1", []),
                               progress=lambda a, b: log({"ev": "progress", "shot": "skeleton_S1", "frames": a, "of": b}))
        rec["render"] = {"impl": "real:rdr02_render_ldi (per-frame stub homography on error)", **{
            k: st[k] for k in ("frames", "render_real", "render_stub", "render_errors", "notes", "pool", "wall_s")}}
        rec["encode"] = {"impl": st["encoder"], "errors": st["encoder_errors"], "bytes": st["bytes"]}
        ledger.add(U_RENDER, "real" if st["render_stub"] == 0 else "fallback", "skeleton", "skeleton")
        ledger.add(U_ENC, "real" if st["encoder"] == "rdr06" else "fallback", st["encoder"], "skeleton")
        rec.update({"status": "ok", "video": out_path.name, "wall_s": round(time.time() - t0, 1),
                    "move": shot.move, "n_frames": shot.n_frames})
        log({"ev": "skeleton_done", "frames": st["frames"], "s": rec["wall_s"], "real": st["render_real"],
             "stub": st["render_stub"], "enc": st["encoder"]})
        return out_path
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "failed", "error": err_text(e, 400), "trace": scrub(traceback.format_exc())[-1500:],
                    "wall_s": round(time.time() - t0, 1)})
        report["errors"].append({"where": "skeleton", "error": err_text(e, 300)})
        log({"ev": "skeleton_failed", "err": err_text(e, 200)})
        return None


# ---------------------------------------------------------------------------------------------------------------
# mode B: scaffold
# ---------------------------------------------------------------------------------------------------------------
def copy_trace(rec: dict, trace: dict) -> None:
    """Copy what the real stages said into the report - also when they fell back (message + traceback tail)."""
    rec["world_build"] = list(trace.get("S2", []))
    if trace.get("errors"):
        rec["stage_errors"] = {k: dict(v) for k, v in trace["errors"].items()}
    if "S-1" in trace:
        rec.setdefault("read_summary", trace["S-1"])


def build_scaffold_registry(image_hw, *, disparity_fn, teacher: Optional[TeacherHandle], cfg: dict, trace: dict,
                            ledger: Ledger):
    from buc.pipeline import spine as S
    from buc.pipeline.stubs import pack
    reg = S.Registry()
    pack.register_stubs(reg)
    reader = TempDepthReader(disparity_fn)
    reg.register(make_s1_impl(reader, make_size_obs_fn(), trace, ledger))
    reg.register(make_s2_impl(teacher, cfg, trace, ledger))
    return reg, reader


def run_scaffold(image_u8: np.ndarray, out_dir: Path, report: dict, ledger: Ledger, deadline: Deadline,
                 thumbs: Dict[str, list], *, shots: Sequence[str], workers: int, dur_scale: float,
                 disparity_fn: Callable, teacher: Optional[TeacherHandle], cfg: dict,
                 on_output: Optional[Callable[[Path], None]] = None) -> List[Path]:
    """``on_output(path)`` runs right after each shot's mp4 is finished and the report refreshed (incremental upload)."""
    from buc.pipeline import spine as S
    rec = report["scaffold"]
    rec["status"] = "running"
    Phase.name = "scaffold-setup"
    trace: dict = {}
    rec["temp_depth"] = {"model": DEPTH_MODEL, "license": "Apache-2.0", "lane": "R-EVAL eval-only",
                         "label": SCAFFOLD_LABEL, "calibration": {"people_box_ref": list(PERSON_BOX),
                                                                   "ref_size": list(REF_SIZE), "people_depth_rel": 1.0,
                                                                   "far_rel": FAR_LIMIT, "near_rel": NEAR_LIMIT},
                         "r5_cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in R5_CFG.items()},
                         "s6_scale": "RD-08 estimate_metric_scale from two hand-annotated people (person_adult prior)"}
    rec["lens_knobs"] = {"S5_f_stop": S5_F_STOP, "S5_scene_scale": S5_LENS_DEPTH_SCALE, "S5_stops_x_people_depth": list(S5_STOPS)}
    img = make_input_image(image_u8, "demo/d15_hero_yak.webp")
    reg, reader = build_scaffold_registry(image_u8.shape[:2], disparity_fn=disparity_fn, teacher=teacher, cfg=cfg,
                                          trace=trace, ledger=ledger)
    spine = S.Spine(reg, prefer="real", allow_fallback=True)
    first_run: Dict[str, dict] = {}
    first_run_ledgered: set = set()
    produced: List[Path] = []
    for name in shots:
        srec = rec["shots"].setdefault(name, {})
        if deadline.stop_now():
            srec.update({"status": "skipped_deadline"})
            ledger.add(U_CAM, "skipped", "deadline", name)
            log({"ev": "shot_skipped", "shot": name, "why": "deadline"})
            continue
        t0 = time.time()
        Phase.name = f"scaffold-{name}"
        d = SHOT_DEFS[name]
        dur = max(4.0 / FPS, round(d["dur"] * dur_scale * FPS) / FPS)
        srec.update({"status": "running", "doc": d["doc"], "move_requested": d["move"], "dur_s": dur})
        try:
            reg.register(make_s8_impl(name, dur, 0, trace, ledger), replace=True)
            n = int(round(dur * FPS)) + 1
            res = spine.run(img, seed=0, until="S8.5", params={"S8": {"n_frames": n, "fps": float(FPS)}})
            stages = stage_rows(res, first_run)
            for st, row in stages.items():
                if row["status"] in ("run", "fallback"):
                    first_run.setdefault(st, {k: row[k] for k in ("impl", "status", "fallback")})
            srec["stages"] = stages
            srec["spine_root"] = res.root
            s8 = stages.get("S8", {})
            if not (s8.get("kind") != "real" and f"S8:{name}" in trace.get("errors", {})):     # impl already ledgered it
                ledger.add(U_CAM, "real" if s8.get("kind") == "real" else "fallback", s8.get("fallback") or "", name)
            ledger.add(U_POLISH, "stub_only", "identity_polish (PRD-03 stub; no real S8.5 exists)", name)
            srec["camera"] = dict(trace.get("S8", {}).get(name, {}))
            read, world, shot = res.artifacts["read"], res.artifacts["world"], res.artifacts["shot"]
            if "read_summary" not in rec:
                rec["read_summary"] = trace.get("S-1") or summarize_read(read)
            depth_ok = rec["read_summary"].get("depth_source") == "mrh"
            label = SCAFFOLD_LABEL if depth_ok else SCAFFOLD_LABEL_NODEPTH
            lens = None
            if d["move"] == "rack_focus":
                lens = {"depth_scale": S5_LENS_DEPTH_SCALE, "max_coc_px": S5_MAX_COC_PX * shot.width / 1920.0,
                        "f_stop": S5_F_STOP}
            z0 = subject_depth(read)
            out_path = out_dir / f"scaffold_{name}.mp4"
            st = render_shot_video(world, read, shot, out_path, label=label, lens=lens, src_u8=image_u8, z0=z0,
                                   workers=workers, deadline=deadline, thumbs=thumbs.setdefault(f"scaffold_{name}", []),
                                   progress=lambda a, b, nm=name: log({"ev": "progress", "shot": nm, "frames": a, "of": b}))
            srec["render"] = {k: st[k] for k in ("frames", "render_real", "render_stub", "render_errors", "notes", "pool",
                                                  "wall_s")}
            srec["encode"] = {"impl": st["encoder"], "errors": st["encoder_errors"], "bytes": st["bytes"]}
            ledger.add(U_RENDER, "real" if st["render_stub"] == 0 else "fallback", "; ".join(st["render_errors"][:1]), name)
            for stg, units in (("S-1", (U_R0, U_R1, U_R5, U_R6)), ("S2", (U_W3, U_W8))):
                row = stages.get(stg, {})
                if row.get("kind") == "stub" and stg not in first_run_ledgered:
                    first_run_ledgered.add(stg)
                    err = trace.get("errors", {}).get(stg)
                    done_units = set(err["units"]) if err else set()          # the impl wrapper ledgered these
                    for u in units:
                        if u in done_units:
                            continue
                        why = f"; {err['error']}" if err else ""
                        ledger.add(u, "fallback", f"whole {stg} fell back to the PRD-03 stub: {row.get('fallback')}{why}", name)
            ledger.add(U_ENC, "real" if st["encoder"] == "rdr06" else "fallback", st["encoder"], name)
            if lens is not None:
                dof_err = [x for x in st["notes"] if x.startswith(("dof fallback", "base render"))]
                srec["lens"] = {"impl": "LNS-02 apply_dof" if not dof_err else "simple layered blur fallback",
                                "f_stop": S5_F_STOP, "scene_scale": S5_LENS_DEPTH_SCALE, "notes": dof_err[:3]}
                ledger.add(U_DOF, "real" if not dof_err else "fallback", "; ".join(dof_err[:1]), name)
            srec.update({"status": "ok", "video": out_path.name, "n_frames": shot.n_frames,
                         "move_used": srec["camera"].get("used") if s8.get("kind") == "real" else f"stub:{shot.move}",
                         "wall_s": round(time.time() - t0, 1)})
            produced.append(out_path)
            kinds = {s: r["kind"] for s, r in stages.items()}
            log({"ev": "shot_done", "shot": name, "frames": shot.n_frames, "s": srec["wall_s"], "kinds": kinds,
                 "stub_fr": st["render_stub"]})
        except Exception as e:  # noqa: BLE001
            srec.update({"status": "failed", "error": err_text(e, 400), "trace": scrub(traceback.format_exc())[-1500:],
                         "wall_s": round(time.time() - t0, 1)})
            report["errors"].append({"where": f"scaffold {name}", "error": err_text(e, 300)})
            ledger.add(U_RENDER, "failed", err_text(e, 160), name)
            log({"ev": "shot_failed", "shot": name, "err": err_text(e, 180)})
        finally:
            gc.collect()
        copy_trace(rec, trace)
        report["units"] = ledger.summary()
        atomic_write_json(out_dir / "stages_report.json", report)
        if on_output is not None and produced and srec.get("status") == "ok" and produced[-1].name == f"scaffold_{name}.mp4":
            with contextlib.suppress(Exception):
                on_output(produced[-1])
    ok = [n for n, s in rec["shots"].items() if s.get("status") == "ok"]
    rec["status"] = "ok" if len(ok) == len(shots) else ("partial" if ok else "failed")
    rec["temp_depth_reader_calls"] = reader.calls
    return produced


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="D15 yak through the buc/ pipeline: skeleton + scaffold")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", DEFAULT_ROOT))
    ap.add_argument("--image", default=None, help="local yak image instead of the HF vault copy")
    ap.add_argument("--modes", default="skeleton,scaffold")
    ap.add_argument("--shots", default=",".join(SHOT_ORDER))
    ap.add_argument("--width", type=int, default=OUT_W, help="max output width (auto-reduced if the deadline is tight)")
    ap.add_argument("--workers", type=int, default=0, help="render processes (0 = auto)")
    ap.add_argument("--dur-scale", type=float, default=1.0, dest="dur_scale", help="scale all shot durations (smoke runs)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-teacher", action="store_true", help="skip WLD-08 (Z-Image)")
    ap.add_argument("--no-matte", action="store_true", help="skip RD-18 edge_unmix")
    ap.add_argument("--force-matte", action="store_true", help="use the matte even when RD-18 flags low_conf")
    ap.add_argument("--fixed-width", action="store_true", help="never auto-reduce the width")
    a = ap.parse_args(argv)
    a.modes = [m for m in a.modes.split(",") if m]
    a.shots = [s for s in a.shots.split(",") if s]
    bad = [s for s in a.shots if s not in SHOT_DEFS] + [m for m in a.modes if m not in ("skeleton", "scaffold")]
    if bad:
        ap.error(f"unknown shots/modes: {bad}")
    return a


def default_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(6, n - 1))


def main(argv=None, *, disparity_fn: Optional[Callable] = None, teacher: Optional[TeacherHandle] = None,
         download_fn: Optional[Callable] = None, upload_api=None, deadline: Optional[Deadline] = None,
         image_array: Optional[np.ndarray] = None, bench_fn: Optional[Callable] = None) -> int:
    args = parse_args(argv)
    shim = ensure_blake3()
    token = os.environ.get("HF_TOKEN") or None
    root = Path(args.root)
    setup_paths(str(root))
    out_dir = safe_out_dir(root)
    deadline = deadline or Deadline.from_env()
    Phase.t0 = time.time()
    hb = start_heartbeat()
    ledger = Ledger()
    uploader = Uploader(token, api=upload_api, enabled=not args.no_upload)
    thumbs: Dict[str, list] = {}
    workers = args.workers or default_workers()
    log({"ev": "start", "hf_token": "yes" if token else "no", "modes": args.modes, "shots": args.shots,
         "workers": workers, "left_s": None if deadline.left() is None else round(deadline.left())})
    report = new_report(args, out_size_for(args.width), {"source": "pending"}, shim)
    report_path = out_dir / "stages_report.json"
    code = 0
    try:
        # ---- image
        Phase.name = "image"
        if image_array is not None:
            info = {"array": image_array, "source": "injected", "sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
                    "matches_doc": None}
        else:
            info = acquire_image(args, root, token, download_fn)
        report["image"] = {k: v for k, v in info.items() if k != "array"}
        report["image"]["original_size"] = [int(info["array"].shape[1]), int(info["array"].shape[0])]

        # ---- width plan
        width = args.width
        frame_equiv = 0.0
        if "skeleton" in args.modes:
            frame_equiv += 1.5 * (SHOT_DEFS["S1"]["dur"] * args.dur_scale * FPS + 1)
        if "scaffold" in args.modes:
            for s in args.shots:
                n = SHOT_DEFS[s]["dur"] * args.dur_scale * FPS + 1
                frame_equiv += n * (3.0 if s == "S5" else 1.0)      # a DoF frame costs about three render frames
        if not args.fixed_width and deadline.ts is not None:
            try:
                spp = (bench_fn or bench_s_per_px)()
                cands = [w for w in (args.width, 1440, 1280, 960, 640, 480) if w <= args.width]
                width = plan_width(cands, spp, frame_equiv, workers, deadline.budget_s(0.6))
            except Exception as e:  # noqa: BLE001
                report["errors"].append({"where": "bench", "error": err_text(e)})
        size = out_size_for(width)
        report["config"].update({"width": size[0], "height": size[1], "requested_width": args.width,
                                 "frame_equiv_estimate": round(frame_equiv)})
        arr = resize_rgb(info["array"], size)
        log({"ev": "image", "src": info["source"], "orig": report["image"]["original_size"], "work": list(size),
             "sha_ok": info["matches_doc"]})
        atomic_write_json(report_path, report)

        outputs: List[Path] = []

        def checkpoint(new_video: Optional[Path] = None) -> None:
            """Fresh report on disk, then (right now, not at the end) the new mp4 + that report to the vault."""
            report["units"] = ledger.summary()
            report["real_vs_stub"] = real_vs_stub(report)
            report["headline"] = headline(report)
            report["outputs"] = [p.name for p in outputs] + ["stages_report.json"]
            report["upload"] = uploader.stats()
            atomic_write_json(report_path, report)
            if not uploader.enabled:
                return
            r = uploader.push(([new_video] if new_video else []) + [report_path])
            if new_video is not None and new_video.name in r.get("uploaded", []):
                log({"ev": "video_uploaded", "file": new_video.name, "hf": f"{HF_REPO_ID}/{HF_UPLOAD_PREFIX}/{new_video.name}"})

        # ---- A
        if "skeleton" in args.modes:
            p = run_skeleton(arr, out_dir, report, ledger, deadline, thumbs, workers=workers, dur_scale=args.dur_scale)
            if p:
                outputs.append(p)
            checkpoint(p)

        def scaffold_output(path: Path) -> None:
            outputs.append(path)
            checkpoint(path)
        # ---- B
        if "scaffold" in args.modes:
            if disparity_fn is None:
                da = DepthAnythingTemp()
                disparity_fn = da
            if teacher is None and not args.no_teacher:
                teacher = TeacherHandle()
            cfg = {"use_matte": not args.no_matte, "force_matte": args.force_matte, "matte_timeout_s": 240.0, "seed": 0}
            try:
                made = run_scaffold(arr, out_dir, report, ledger, deadline, thumbs, shots=args.shots, workers=workers,
                                    dur_scale=args.dur_scale, disparity_fn=disparity_fn,
                                    teacher=None if args.no_teacher else teacher, cfg=cfg, on_output=scaffold_output)
                outputs += [p for p in made if p not in outputs]
            except Exception as e:  # noqa: BLE001
                report["scaffold"].update({"status": "failed", "error": err_text(e, 400)})
                report["errors"].append({"where": "scaffold", "error": err_text(e, 300)})
                log({"ev": "scaffold_failed", "err": err_text(e, 200)})

        # ---- sheet, report, upload
        Phase.name = "finish"
        sheet = out_dir / "contact_sheet.jpg"
        order = ["skeleton_S1"] + [f"scaffold_{s}" for s in SHOT_ORDER]
        sheet_ok = False
        with contextlib.suppress(Exception):
            sheet_ok = make_contact_sheet({k: thumbs[k] for k in order if k in thumbs}, sheet)
        report["units"] = ledger.summary()
        report["real_vs_stub"] = real_vs_stub(report)
        report["headline"] = headline(report)
        report["outputs"] = [p.name for p in outputs] + (["contact_sheet.jpg"] if sheet_ok else []) + ["stages_report.json"]
        atomic_write_json(report_path, report)
        if uploader.enabled:
            # the mp4s are already up (incremental); the final upload is the contact sheet and the final report
            uploader.push([sheet] if sheet_ok else [])
        report["upload"] = uploader.stats()
        atomic_write_json(report_path, report)
        if uploader.enabled:
            uploader.push([report_path])
        n_ok = len([p for p in outputs])
        code = 0 if n_ok else 1
        log({"ev": "done", "videos": n_ok, "skeleton": report["skeleton"].get("status"),
             "scaffold": report["scaffold"].get("status")}, marker=MARK + "_DONE")
    except Exception as e:  # noqa: BLE001
        code = 2
        report["errors"].append({"where": "main", "error": err_text(e, 300)})
        report["fatal"] = err_text(e, 400)
        with contextlib.suppress(Exception):
            report["units"] = ledger.summary()
            atomic_write_json(report_path, report)
        log({"ev": "fatal", "err": err_text(e, 200)}, marker=MARK + "_DONE")
    finally:
        hb.set()
    return code


def headline(report: dict) -> dict:
    """Two-line answer for a human: what ran, which units were real / temp / not."""
    units = report.get("units", {})
    by: Dict[str, List[str]] = {}
    for name, u in units.items():
        by.setdefault(u["verdict"], []).append(name)
    shots = report.get("scaffold", {}).get("shots", {})
    return {
        "skeleton": report.get("skeleton", {}).get("status"),
        "scaffold_shots_ok": sorted(n for n, s in shots.items() if s.get("status") == "ok"),
        "scaffold_shots_not_ok": {n: s.get("status") for n, s in shots.items() if s.get("status") != "ok"},
        "units_by_verdict": by,
        "depth_source": report.get("scaffold", {}).get("read_summary", {}).get("depth_source"),
    }


def real_vs_stub(report: dict) -> dict:
    """Compact answer to 'which stages were real and which were stubs' for the scaffold shots."""
    out: Dict[str, Any] = {}
    for name, s in report.get("scaffold", {}).get("shots", {}).items():
        if s.get("status") != "ok":
            out[name] = s.get("status")
            continue
        out[name] = {st: ("real" if row["kind"] == "real" else "stub") for st, row in s.get("stages", {}).items()}
    sk = report.get("skeleton", {})
    if sk.get("stages"):
        out["skeleton"] = {st: row["kind"] for st, row in sk["stages"].items()}
    return out


if __name__ == "__main__":
    sys.exit(main())
