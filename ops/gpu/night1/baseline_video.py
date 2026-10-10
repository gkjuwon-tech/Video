#!/usr/bin/env python3
"""Night1 job C: naive 2.5D depth-warp BASELINE video of the D15 hero yak.

This is NOT our model.  It is the baseline we have to beat.  Six shots
(S1..S6, same names as docs/DEMO-D15.md) are rendered from a single image plus
a monocular depth estimate:

  image + Depth-Anything-V2-Small -> calibrated metric-ish depth (person = 1.0)
  -> pinhole point cloud (hfov 50 deg) -> re-projection with a z-buffer
  (torch scatter_reduce amin) -> cv2.inpaint TELEA for holes -> H.264 mp4.

Usage (GPU box):
  N1_ROOT=/workspace/night1 N1_DEADLINE=<unix seconds> HF_TOKEN=... \
      python baseline_video.py [--image PATH] [--shots S1,S2,...] [--no-upload]

Log markers (one JSON line, flush=True):  N1_VID {...}  and  N1_VID_DONE {...}.
HF_TOKEN is never printed or written anywhere.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

TAG = "N1_VID"
HF_REPO_ID = "sdsdsdsdsdsdsd123/buc-artifacts"
HF_REPO_TYPE = "model"
HF_IMAGE_PATH = "demo/d15_hero_yak.webp"
HF_UPLOAD_PREFIX = "video/night1"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

DEFAULT_ROOT = "/workspace/night1"
OUT_W, OUT_H = 1920, 826
FPS = 24
HFOV_DEG = 50.0
LABEL = "BASELINE (naive depth warp)"

REF_SIZE = (2000, 861)  # the person box below is given in these image coordinates
PERSON_BOX = (900, 640, 1130, 840)  # x0, y0, x1, y1
FAR_LIMIT = 40.0  # sky / far mountains are clamped to this distance
NEAR_LIMIT = 0.1  # sanity clamp, avoids negative / absurd depth
FAR_PERCENTILE = 2.0
DISC_RATIO = 1.15

# shot name -> seconds (docs/DEMO-D15.md)
SHOTS: dict[str, float] = {"S1": 8, "S2": 8, "S3": 10, "S4": 8, "S5": 6, "S6": 6}
SHOT_TITLES = {
    "S1": "push-in",
    "S2": "slider",
    "S3": "arc",
    "S4": "crane up",
    "S5": "rack focus",
    "S6": "dolly zoom",
}


# ----------------------------------------------------------------------------
# logging / secrets / deadline
# ----------------------------------------------------------------------------
def marker(tag: str, payload: dict) -> None:
    print(f"{tag} {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}", flush=True)


def log(payload: dict) -> None:
    marker(TAG, payload)


def scrub(text: str, token: Optional[str]) -> str:
    """Remove the token value from any text before it is printed."""
    text = str(text)
    if token:
        text = text.replace(token, "***")
    return text


class DeadlineReached(Exception):
    pass


class Deadline:
    def __init__(self, ts: Optional[float], clock: Callable[[], float] = time.time):
        self.ts = ts
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
        return cls(ts, clock)

    def reached(self) -> bool:
        return self.ts is not None and self.clock() >= self.ts

    def left(self) -> Optional[float]:
        return None if self.ts is None else self.ts - self.clock()

    def check(self) -> None:
        if self.reached():
            raise DeadlineReached()


# ----------------------------------------------------------------------------
# depth calibration
# ----------------------------------------------------------------------------
def scale_box(box, src_size, dst_size) -> tuple[int, int, int, int]:
    """Scale (x0,y0,x1,y1) given in src_size=(w,h) coordinates to dst_size."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    x0, y0, x1, y1 = box
    out = (int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy)))
    x0, y0, x1, y1 = out
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return (x0, y0, min(x1, dst_size[0]), min(y1, dst_size[1]))


def disp_to_depth(
    disp: np.ndarray,
    box: tuple[int, int, int, int],
    far: float = FAR_LIMIT,
    near: float = NEAR_LIMIT,
    far_percentile: float = FAR_PERCENTILE,
) -> np.ndarray:
    """Relative inverse depth (larger = closer) -> depth with person median = 1.0.

    depth = 1 / (a*disp + b), fitted on two points: the median disparity of the
    people box -> 1.0 and the 2nd percentile of disparity -> `far` (40).
    """
    disp = np.asarray(disp, dtype=np.float64)
    x0, y0, x1, y1 = box
    disp_p = float(np.median(disp[y0:y1, x0:x1]))
    disp_lo = float(np.percentile(disp, far_percentile))
    gap = disp_p - disp_lo
    eps = 1e-6 * max(1.0, abs(disp_p))
    if gap < eps:  # degenerate: person is as far as the far plane; keep monotonic anyway
        gap = max(eps, 1e-3 * max(1.0, abs(disp_p)))
    a = (1.0 - 1.0 / far) / gap
    b = 1.0 - a * disp_p
    inv = a * disp + b
    inv = np.clip(inv, 1.0 / far, 1.0 / near)
    return (1.0 / inv).astype(np.float32)


def discontinuity_mask(depth: np.ndarray, ratio: float = DISC_RATIO) -> np.ndarray:
    """True where a pixel has a right/down neighbour whose depth ratio exceeds `ratio`."""
    d = np.asarray(depth, dtype=np.float32)
    m = np.zeros(d.shape, dtype=bool)
    r = np.maximum(d[:, 1:], d[:, :-1]) / np.maximum(np.minimum(d[:, 1:], d[:, :-1]), 1e-6)
    m[:, :-1] |= r > ratio
    r = np.maximum(d[1:, :], d[:-1, :]) / np.maximum(np.minimum(d[1:, :], d[:-1, :]), 1e-6)
    m[:-1, :] |= r > ratio
    return m


# ----------------------------------------------------------------------------
# camera trajectories (world frame = source camera frame: x right, y DOWN, z forward)
# ----------------------------------------------------------------------------
@dataclass
class Pose:
    pos: np.ndarray  # camera position in world
    R: np.ndarray  # camera -> world rotation (3x3)
    fscale: float = 1.0  # focal length multiplier relative to the source camera
    focus: Optional[float] = None  # focus distance (z) for rack focus


def smoothstep(t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def focal_px(width: int, hfov_deg: float = HFOV_DEG) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def person_point(width: int, height: int, hfov_deg: float = HFOV_DEG, box=PERSON_BOX,
                 ref_size=REF_SIZE, dist: float = 1.0) -> np.ndarray:
    """3D position (source camera frame) of the people, at z = dist."""
    x0, y0, x1, y1 = scale_box(box, ref_size, (width, height))
    u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    f = focal_px(width, hfov_deg)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([(u - cx) / f * dist, (v - cy) / f * dist, dist], dtype=np.float64)


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix R with R @ a == b (a, b need not be unit)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate pi about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def keep_person_fixed(pos: np.ndarray, person: np.ndarray) -> np.ndarray:
    """Camera rotation that keeps the person at its ORIGINAL pixel from `pos`."""
    d0 = person / np.linalg.norm(person)  # direction in camera frame (as seen from the origin)
    d_new = person - pos
    return rot_between(d0, d_new)


def focus_at(t: float) -> float:
    """S5 focus distance: 0.35 (poppies) -> 1.0 (people) -> 1.6 (yak eye); interpolated in 1/z."""
    keys = [(0.0, 0.35), (0.15, 0.35), (0.50, 1.0), (0.62, 1.0), (0.92, 1.6), (1.0, 1.6)]
    t = min(max(float(t), 0.0), 1.0)
    for (t0, z0), (t1, z1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            if t1 == t0 or z0 == z1:
                return z1 if t >= t1 else z0
            e = smoothstep((t - t0) / (t1 - t0))
            return 1.0 / lerp(1.0 / z0, 1.0 / z1, e)
    return keys[-1][1]


def shot_pose(shot: str, t: float, person: Optional[np.ndarray] = None) -> Pose:
    """Camera pose at normalised time t in [0, 1]; distance to the people d = 1.0."""
    if person is None:
        person = person_point(OUT_W, OUT_H)
    e = smoothstep(t)
    I3 = np.eye(3)
    if shot == "S1":  # push-in
        return Pose(np.array([0.0, 0.0, 0.30 * e]), I3)
    if shot == "S2":  # slider, slightly below
        return Pose(np.array([lerp(-0.06, 0.06, e), 0.01, 0.0]), I3)
    if shot == "S3":  # arc around the people, yaw -12 -> +12 deg
        th = math.radians(lerp(-12.0, 12.0, e))
        c, s = math.cos(th), math.sin(th)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        pos = person + Ry @ (-person)
        return Pose(pos, keep_person_fixed(pos, person))
    if shot == "S4":  # crane up (y is down: +0.03 low -> -0.10 high), tilt to keep people
        pos = np.array([0.0, lerp(0.03, -0.10, e), 0.0])
        return Pose(pos, keep_person_fixed(pos, person))
    if shot == "S5":  # locked camera, focus pull
        return Pose(np.zeros(3), I3, 1.0, focus_at(t))
    if shot == "S6":  # dolly zoom, people keep their size
        back = lerp(0.0, 0.4, float(min(max(t, 0.0), 1.0)))
        return Pose(np.array([0.0, 0.0, -back]), I3, 1.0 + back)
    raise ValueError(f"unknown shot {shot!r}")


def frame_count(shot: str, fps: int = FPS, max_frames: Optional[int] = None) -> int:
    n = int(round(SHOTS[shot] * fps))
    if max_frames is not None:
        n = max(1, min(n, int(max_frames)))
    return n


def project_point(P, pose: Pose, width: int, height: int, hfov_deg: float = HFOV_DEG):
    """Project world point(s) P (...,3) -> (u, v, z) in the pose's camera (numpy)."""
    P = np.asarray(P, dtype=np.float64)
    pc = (P - pose.pos) @ pose.R
    f = focal_px(width, hfov_deg) * pose.fscale
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    z = pc[..., 2]
    return f * pc[..., 0] / z + cx, f * pc[..., 1] / z + cy, z


# ----------------------------------------------------------------------------
# z-buffer splatting renderer (torch)
# ----------------------------------------------------------------------------
def splat_zbuffer(u, v, z, colors, width: int, height: int, valid=None, bias: float = 2e-3):
    """Forward-splat points into a (height, width) z-buffer.

    Every point is written to the 4 pixels around its sub-pixel position; the
    farther candidate pixels get a tiny depth bias so that the nearest pixel of
    the point wins ties (exact identity for an unmoved camera) while still
    closing sub-pixel cracks when the view magnifies.  `scatter_reduce(amin)`
    resolves visibility; colour comes from the z-buffer winner.

    u, v, z: float tensors (N,); colors: uint8 tensor (N,3).
    Returns (rgb uint8 (H,W,3), hole bool (H,W), zbuf float (H,W)).
    """
    import torch

    dev = u.device
    N = u.shape[0]
    if valid is None:
        valid = torch.ones(N, dtype=torch.bool, device=dev)
    valid = valid & torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z)
    # keep floor() safe for absurd projections
    u = torch.where(valid, u, torch.zeros_like(u))
    v = torch.where(valid, v, torch.zeros_like(v))
    fu = torch.floor(u)
    fv = torch.floor(v)
    idx_l, z_l, src_l = [], [], []
    for dx in (0, 1):
        for dy in (0, 1):
            iu = fu + dx
            iv = fv + dy
            ok = valid & (iu >= 0) & (iu < width) & (iv >= 0) & (iv < height)
            src = torch.nonzero(ok, as_tuple=False).squeeze(1)
            if src.numel() == 0:
                continue
            d = (u[src] - iu[src]).abs() + (v[src] - iv[src]).abs()
            idx_l.append((iv[src].long() * width + iu[src].long()))
            z_l.append(z[src] * (1.0 + bias * d))
            src_l.append(src)
    zbuf = torch.full((height * width,), float("inf"), dtype=z.dtype, device=dev)
    out = torch.zeros((height * width, 3), dtype=torch.uint8, device=dev)
    if idx_l:
        idx = torch.cat(idx_l)
        zz = torch.cat(z_l)
        src = torch.cat(src_l)
        zbuf.scatter_reduce_(0, idx, zz, reduce="amin", include_self=True)
        win = zz <= zbuf[idx]
        out[idx[win]] = colors[src[win]]
    hole = torch.isinf(zbuf)
    return out.view(height, width, 3), hole.view(height, width), zbuf.view(height, width)


class Renderer:
    """Holds the source point cloud; renders it from any Pose."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, hfov_deg: float = HFOV_DEG, device=None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.h, self.w = rgb.shape[:2]
        self.hfov = hfov_deg
        self.f0 = focal_px(self.w, hfov_deg)
        self.cx = (self.w - 1) / 2.0
        self.cy = (self.h - 1) / 2.0
        d = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)).to(self.device).reshape(-1)
        vv, uu = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32, device=self.device),
            torch.arange(self.w, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        uu, vv = uu.reshape(-1), vv.reshape(-1)
        self.P = torch.stack([(uu - self.cx) / self.f0 * d, (vv - self.cy) / self.f0 * d, d], dim=1)
        self.colors = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.uint8)).to(self.device).reshape(-1, 3)

    def render(self, pose: Pose):
        """-> (rgb uint8 (H,W,3) with holes black, hole bool (H,W))."""
        torch = self.torch
        pos = torch.as_tensor(pose.pos, dtype=torch.float32, device=self.device)
        R = torch.as_tensor(pose.R, dtype=torch.float32, device=self.device)
        pc = (self.P - pos) @ R
        z = pc[:, 2]
        valid = z > 0.02
        zs = torch.where(valid, z, torch.ones_like(z))
        f = self.f0 * pose.fscale
        u = f * pc[:, 0] / zs + self.cx
        v = f * pc[:, 1] / zs + self.cy
        rgb, hole, _ = splat_zbuffer(u, v, z, self.colors, self.w, self.h, valid)
        return rgb.cpu().numpy(), hole.cpu().numpy()


def fill_holes(img: np.ndarray, hole: np.ndarray, radius: int = 3) -> np.ndarray:
    """cv2.inpaint TELEA (radius 3).  Big holes are first filled on a 1/4 scale
    pyramid level (TELEA at full res is slow and streaky on huge areas)."""
    import cv2

    hole = np.asarray(hole, dtype=bool)
    if not hole.any() or hole.all():
        return img
    out = np.ascontiguousarray(img).copy()
    hole_u8 = hole.astype(np.uint8)
    big = cv2.morphologyEx(hole_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    big_sel = np.zeros(hole.shape, dtype=bool)
    if big.any():
        big = cv2.dilate(big, np.ones((3, 3), np.uint8)).astype(bool) & hole
        h, w = hole.shape
        cw, ch = max(w // 4, 1), max(h // 4, 1)
        valid = (1 - hole_u8).astype(np.float32)
        num = cv2.resize(out.astype(np.float32) * valid[..., None], (cw, ch), interpolation=cv2.INTER_AREA)
        den = cv2.resize(valid, (cw, ch), interpolation=cv2.INTER_AREA)
        coarse = np.clip(num / np.maximum(den, 1e-6)[..., None], 0, 255).astype(np.uint8)
        chole = (den < 0.5).astype(np.uint8) * 255
        if chole.any() and not chole.all():
            coarse = cv2.inpaint(coarse, chole, radius, cv2.INPAINT_TELEA)
        up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        out[big] = up[big]
        big_sel = big
    rest = hole & ~big_sel
    if rest.any():
        out = cv2.inpaint(out, rest.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return out


# ----------------------------------------------------------------------------
# rack focus (layered gaussian blur)
# ----------------------------------------------------------------------------
class RackFocus:
    """Depth-layered defocus.  CoC radius r = k*|1/z - 1/z_f| (<= max_r px)."""

    def __init__(self, img: np.ndarray, depth: np.ndarray, layers: int = 8, k: float = 8.0, max_r: float = 18.0):
        self.img = np.ascontiguousarray(img, dtype=np.uint8)
        self.imgf = self.img.astype(np.float32)
        self.k = float(k)
        self.max_r = float(max_r)
        invz = (1.0 / np.maximum(depth.astype(np.float64), 1e-6)).astype(np.float32)
        self.invz = invz
        lo, hi = float(invz.min()), float(invz.max())
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        b = np.clip(((invz - lo) / (hi - lo) * layers).astype(np.int32), 0, layers - 1)
        self.masks, self.mean_invz = [], []
        for j in range(layers):  # j = 0 is the farthest layer (smallest 1/z)
            m = (b == j)
            if not m.any():
                continue
            self.masks.append(m.astype(np.float32))
            self.mean_invz.append(float(invz[m].mean()))
        self._key = None
        self._val = None

    def radii(self, focus_z: float) -> list[float]:
        inv_f = 1.0 / focus_z
        return [min(self.max_r, self.k * abs(c - inv_f)) for c in self.mean_invz]

    def __call__(self, focus_z: float) -> np.ndarray:
        import cv2

        radii = self.radii(focus_z)
        key = tuple(round(r, 2) for r in radii)
        if key == self._key:
            return self._val.copy()

        def blur(a, r):
            sigma = r / 2.0
            if sigma < 0.3:
                return a
            return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

        out = blur(self.imgf, radii[0])  # start from a defocused version of the whole image
        for m, r in zip(self.masks, radii):
            pm = np.concatenate([self.imgf * m[..., None], m[..., None]], axis=2)
            pm = blur(pm, r)
            a = np.clip(pm[..., 3:4], 0.0, 1.0)
            out = out * (1.0 - a) + pm[..., :3]
        res = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        self._key, self._val = key, res
        return res.copy()


# ----------------------------------------------------------------------------
# frames, overlay, video IO
# ----------------------------------------------------------------------------
def overlay_label(frame: np.ndarray, text: str = LABEL) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(frame).copy()
    h, w = out.shape[:2]
    scale = max(0.3, 0.65 * w / 1920.0)
    th = max(1, int(round(scale * 2)))
    org = (max(4, int(14 * w / 1920.0)), max(12, int(36 * w / 1920.0)))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return out


@dataclass
class Scene:
    rgb: np.ndarray  # (H,W,3) uint8 at output resolution
    depth: np.ndarray  # (H,W) float32, person median = 1.0
    hfov: float = HFOV_DEG
    device: Optional[str] = None
    person: Optional[np.ndarray] = None
    _renderer: Optional[Renderer] = None
    _rack: Optional[RackFocus] = None

    def __post_init__(self):
        h, w = self.rgb.shape[:2]
        if self.person is None:
            self.person = person_point(w, h, self.hfov)

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer(self.rgb, self.depth, self.hfov, self.device)
        return self._renderer

    @property
    def rack(self) -> RackFocus:
        if self._rack is None:
            self._rack = RackFocus(self.rgb, self.depth)
        return self._rack


def make_frame_fn(shot: str, scene: Scene) -> Callable[[float], np.ndarray]:
    """t in [0,1] -> final uint8 frame (label overlaid)."""

    def fn(t: float) -> np.ndarray:
        pose = shot_pose(shot, t, scene.person)
        if shot == "S5":
            frame = scene.rack(pose.focus)
        else:
            rgb, hole = scene.renderer.render(pose)
            frame = fill_holes(rgb, hole, 3)
        return overlay_label(frame)

    return fn


def open_writer(path, fps: int = FPS):
    import imageio

    return imageio.get_writer(
        str(path), format="FFMPEG", mode="I", fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, quality=None, output_params=["-crf", "18", "-preset", "medium"],
    )


def render_shot(
    shot: str,
    frame_fn: Callable[[float], np.ndarray],
    n_frames: int,
    out_path: Path,
    fps: int = FPS,
    deadline: Optional[Deadline] = None,
    progress_every_s: float = 60.0,
    writer_factory: Callable = open_writer,
) -> float:
    """Render one shot to `out_path` (atomic: temp file then rename).  Returns seconds."""
    out_path = Path(out_path)
    part = out_path.with_name(out_path.stem + ".part.mp4")
    t0 = time.time()
    last = t0
    writer = writer_factory(part, fps)
    try:
        for i in range(n_frames):
            if deadline is not None:
                deadline.check()
            t = i / (n_frames - 1) if n_frames > 1 else 0.0
            writer.append_data(frame_fn(t))
            if time.time() - last >= progress_every_s:
                last = time.time()
                log({"shot": shot, "progress": f"{i + 1}/{n_frames}", "s": round(last - t0, 1)})
        writer.close()
        writer = None
        os.replace(part, out_path)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return time.time() - t0


def concat_videos(paths: list[Path], out_path: Path) -> None:
    """Lossless concat of same-codec mp4 files with the ffmpeg concat demuxer."""
    import imageio_ffmpeg

    out_path = Path(out_path)
    part = out_path.with_name(out_path.stem + ".part.mp4")
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="concat_", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            for p in paths:
                f.write("file '{}'\n".format(str(Path(p).resolve()).replace("'", "'\\''")))
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", "-movflags", "+faststart", str(part)],
            check=True, timeout=600,
        )
        os.replace(part, out_path)
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
        try:
            part.unlink()
        except OSError:
            pass


def read_frames_at(path, wanted: list[int]) -> dict[int, np.ndarray]:
    """Decode a video and return {index: frame} for the wanted indices; index -1 means the last."""
    import imageio

    got: dict[int, np.ndarray] = {}
    last = None
    reader = imageio.get_reader(str(path), format="FFMPEG")
    try:
        want = set(w for w in wanted if w >= 0)
        for i, fr in enumerate(reader):
            if i in want:
                got[i] = np.asarray(fr)
            last = fr
    finally:
        reader.close()
    if -1 in wanted and last is not None:
        got[-1] = np.asarray(last)
    return got


def make_sheet(shot_paths: dict[str, Path], counts: dict[str, int], out_path: Path, thumb_w: int = 640) -> None:
    """Contact sheet: one row per shot, columns = first / middle / last frame."""
    import cv2

    cells_rows = []
    cell_h = None
    for shot in SHOTS:
        p = shot_paths.get(shot)
        row = []
        if p is not None and Path(p).exists():
            n = counts.get(shot, 0)
            mid = (n - 1) // 2 if n else 0
            frames = read_frames_at(p, [0, mid, -1])
            for key, name in ((0, "start"), (mid, "mid"), (-1, "end")):
                fr = frames.get(key)
                if fr is None:
                    fr = frames.get(0)
                if fr is None:
                    continue
                h, w = fr.shape[:2]
                th = max(1, int(round(h * thumb_w / w)))
                cell = cv2.resize(fr, (thumb_w, th), interpolation=cv2.INTER_AREA)
                cell = np.ascontiguousarray(cell)
                label = f"{shot} {SHOT_TITLES[shot]} {name}"
                cv2.putText(cell, label, (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(cell, label, (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                row.append(cell)
                cell_h = th
        cells_rows.append(row)
    if cell_h is None:
        raise RuntimeError("no frames available for the contact sheet")
    blank = np.zeros((cell_h, thumb_w, 3), dtype=np.uint8)
    rows = []
    for row in cells_rows:
        row = list(row) + [blank] * (3 - len(row))
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_path = Path(out_path)
    part = out_path.with_name(out_path.stem + ".part.jpg")
    cv2.imwrite(str(part), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    os.replace(part, out_path)


# ----------------------------------------------------------------------------
# image, depth, upload
# ----------------------------------------------------------------------------
def load_image(path, size: Optional[tuple[int, int]] = None):
    """-> (PIL RGB image at original size, RGB uint8 array resized to `size`=(w,h))."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img if size is None or img.size == tuple(size) else img.resize(tuple(size), Image.LANCZOS))
    return img, np.ascontiguousarray(arr)


def make_depth_fn() -> Callable:
    """Depth-Anything-V2-Small via transformers (lazy import).  Returns relative
    inverse depth (larger = closer) at the input image's resolution."""

    def build():
        import torch
        from transformers import pipeline

        dev = 0 if torch.cuda.is_available() else -1
        return pipeline("depth-estimation", model=DEPTH_MODEL, device=dev)

    state: dict = {}

    def fn(pil_image) -> np.ndarray:
        import cv2

        if "pipe" not in state:
            state["pipe"] = build()
        out = state["pipe"](pil_image)
        pd = out.get("predicted_depth") if isinstance(out, dict) else None
        if pd is not None and hasattr(pd, "detach"):
            arr = pd.detach().float().cpu().numpy()
        elif pd is not None:
            arr = np.asarray(pd, dtype=np.float32)
        else:
            arr = np.asarray(out["depth"], dtype=np.float32)
        arr = np.squeeze(arr).astype(np.float32)
        w, h = pil_image.size
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
        return arr

    return fn


def estimate_disparity(pil_image, depth_fn: Callable, cache_dir: Path) -> np.ndarray:
    """Run the depth model once; cache the result so a restart skips it."""
    key = hashlib.sha1(pil_image.tobytes()).hexdigest()[:12]
    cache = Path(cache_dir) / f"disp_{key}.npy"
    h, w = pil_image.size[1], pil_image.size[0]
    if cache.exists():
        try:
            d = np.load(cache)
            if d.shape == (h, w):
                return d.astype(np.float32)
        except Exception:
            pass
    d = np.asarray(depth_fn(pil_image), dtype=np.float32)
    d = np.squeeze(d)
    if d.shape != (h, w):
        import cv2

        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp.npy")
    np.save(tmp, d)
    os.replace(tmp, cache)
    return d


def build_depth(disp: np.ndarray, out_size: tuple[int, int], box=PERSON_BOX, ref_size=REF_SIZE) -> np.ndarray:
    import cv2

    w, h = out_size
    if disp.shape != (h, w):
        disp = cv2.resize(disp.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return disp_to_depth(disp, scale_box(box, ref_size, (w, h)))


def default_download(repo_id: str, filename: str, token: str, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=HF_REPO_TYPE, token=token,
                           cache_dir=cache_dir)


def upload_outputs(paths: list[Path], token: Optional[str], api=None, repo_id: str = HF_REPO_ID,
                   prefix: str = HF_UPLOAD_PREFIX) -> int:
    """Upload each file to the vault; a failure is a warning, never fatal.  Returns #ok."""
    if not token:
        log({"upload": "skipped", "hf_token": "no", "note": "hf_token=no"})
        return 0
    if api is None:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        except Exception as e:  # pragma: no cover - needs real hub lib
            log({"warn": "hf api unavailable", "err": scrub(f"{type(e).__name__}: {e}", token)[:300]})
            return 0
    ok = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{prefix}/{p.name}",
                            repo_id=repo_id, repo_type=HF_REPO_TYPE)
            ok += 1
            log({"uploaded": p.name})
        except Exception as e:
            log({"warn": "upload failed", "file": p.name, "err": scrub(f"{type(e).__name__}: {e}", token)[:300]})
    return ok


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_person_box(s: str) -> tuple[int, int, int, int]:
    vals = tuple(int(x) for x in s.split(","))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("person box must be x0,y0,x1,y1")
    return vals  # type: ignore[return-value]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Night1 job C: naive depth-warp baseline video (yak).")
    ap.add_argument("--root", default=os.environ.get("N1_ROOT", DEFAULT_ROOT), help="work folder (env N1_ROOT)")
    ap.add_argument("--out", default=None, help="output folder (default ROOT/video)")
    ap.add_argument("--image", default=None, help="local input image (used instead of the vault download)")
    ap.add_argument("--shots", default=",".join(SHOTS), help="comma list of shots, default all")
    ap.add_argument("--width", type=int, default=OUT_W)
    ap.add_argument("--height", type=int, default=OUT_H)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--hfov", type=float, default=HFOV_DEG)
    ap.add_argument("--person-box", type=parse_person_box, default=PERSON_BOX,
                    help="x0,y0,x1,y1 of the people in 2000x861 image coordinates")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per shot (smoke tests)")
    ap.add_argument("--device", default=None, help="torch device for the renderer (default cuda if available)")
    ap.add_argument("--no-upload", action="store_true", help="do not upload to the HF vault")
    ap.add_argument("--force", action="store_true", help="re-render even if outputs exist")
    return ap.parse_args(argv)


def main(argv=None, *, depth_fn: Optional[Callable] = None, download_fn: Optional[Callable] = None,
         upload_api=None, clock: Callable[[], float] = time.time) -> int:
    args = parse_args(argv)
    t_start = time.time()
    token = (os.environ.get("HF_TOKEN") or "").strip() or None
    deadline = Deadline.from_env(clock)
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = [s.strip().upper() for s in args.shots.split(",") if s.strip()]
    bad = [s for s in shots if s not in SHOTS]
    if bad:
        log({"error": f"unknown shots {bad}"})
        return 2
    shot_path = {s: out_dir / f"baseline_yak_{s}.mp4" for s in SHOTS}
    all_path = out_dir / "baseline_yak_all.mp4"
    sheet_path = out_dir / "baseline_yak_sheet.jpg"
    counts = {s: frame_count(s, args.fps, args.max_frames) for s in SHOTS}

    log({"start": True, "root": str(root), "shots": shots, "hf_token": "yes" if token else "no",
         "deadline_in_s": None if deadline.left() is None else round(deadline.left())})
    if not token:
        log({"note": "hf_token=no", "hf_token": "no"})

    done, failed, skipped = [], [], []
    todo = [s for s in shots if args.force or not shot_path[s].exists()]
    skipped = [s for s in shots if s not in todo]
    for s in skipped:
        log({"shot": s, "frames": counts[s], "skip": "exists"})
    hit_deadline = deadline.reached()
    rendered_any = False
    fatal = None

    if todo and not hit_deadline:
        try:
            scene = prepare_scene(args, token, depth_fn, download_fn, out_dir, root)
        except Exception as e:
            fatal = f"{type(e).__name__}: {e}"
            log({"error": "prepare failed", "err": scrub(fatal, token)[:400]})
            print(scrub(traceback.format_exc(), token), file=sys.stderr, flush=True)
            scene = None
        if scene is not None:
            for s in todo:
                if deadline.reached():
                    hit_deadline = True
                    break
                t0 = time.time()
                try:
                    secs = render_shot(s, make_frame_fn(s, scene), counts[s], shot_path[s], args.fps, deadline)
                    done.append(s)
                    rendered_any = True
                    log({"shot": s, "frames": counts[s], "s": round(secs, 1)})
                except DeadlineReached:
                    hit_deadline = True
                    log({"shot": s, "aborted": "deadline"})
                    break
                except Exception as e:
                    failed.append(s)
                    log({"shot": s, "error": scrub(f"{type(e).__name__}: {e}", token)[:400],
                         "s": round(time.time() - t0, 1)})
                    print(scrub(traceback.format_exc(), token), file=sys.stderr, flush=True)
                    gc.collect()
    elif todo:
        log({"note": "deadline reached before start"})

    have = [s for s in SHOTS if shot_path[s].exists()]
    selected_have = [s for s in shots if s in have]

    # all-in-one (only when all six shots exist)
    made_all = False
    if len(have) == len(SHOTS) and (rendered_any or args.force or not all_path.exists()):
        try:
            concat_videos([shot_path[s] for s in SHOTS], all_path)
            made_all = True
            log({"all": all_path.name})
        except Exception as e:
            log({"error": "concat failed", "err": scrub(f"{type(e).__name__}: {e}", token)[:300]})

    # contact sheet from whatever exists
    made_sheet = False
    if have and (rendered_any or args.force or not sheet_path.exists()):
        try:
            make_sheet({s: shot_path[s] for s in have}, counts, sheet_path)
            made_sheet = True
            log({"sheet": sheet_path.name})
        except Exception as e:
            log({"error": "sheet failed", "err": scrub(f"{type(e).__name__}: {e}", token)[:300]})

    uploaded = 0
    if not args.no_upload:
        files = [shot_path[s] for s in have] + [p for p in (all_path, sheet_path) if p.exists()]
        try:
            uploaded = upload_outputs(files, token, upload_api)
        except Exception as e:  # never kill the night
            log({"warn": "upload crashed", "err": scrub(f"{type(e).__name__}: {e}", token)[:300]})

    marker("N1_VID_DONE", {
        "done": done, "skipped": skipped, "failed": failed, "have": have,
        "all": all_path.exists(), "sheet": sheet_path.exists(), "uploaded": uploaded,
        "deadline": hit_deadline, "fatal": bool(fatal), "s": round(time.time() - t_start, 1),
    })
    if fatal:
        return 1
    if not selected_have and not hit_deadline:
        return 1
    return 0


def prepare_scene(args, token, depth_fn, download_fn, out_dir: Path, root: Path) -> Scene:
    """Fetch the image, estimate + calibrate depth, build the Scene."""
    t0 = time.time()
    if args.image:
        image_path = args.image
    elif token:
        dl = download_fn or default_download
        image_path = dl(HF_REPO_ID, HF_IMAGE_PATH, token, str(root / "hf_cache"))
    else:
        raise RuntimeError("no HF_TOKEN and no --image: nothing to render")
    size = (args.width, args.height)
    pil, rgb = load_image(image_path, size)
    log({"image": Path(str(image_path)).name, "src_size": list(pil.size), "out_size": list(size)})
    fn = depth_fn or make_depth_fn()
    disp = estimate_disparity(pil, fn, out_dir / "cache")
    del fn
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    box = args.person_box
    depth = build_depth(disp, size, box, REF_SIZE)
    tear = float(discontinuity_mask(depth).mean())
    log({"depth": "ok", "z_min": round(float(depth.min()), 3), "z_max": round(float(depth.max()), 2),
         "tear_frac": round(tear, 4), "s": round(time.time() - t0, 1)})
    return Scene(rgb, depth, args.hfov, args.device)


if __name__ == "__main__":
    sys.exit(main())
