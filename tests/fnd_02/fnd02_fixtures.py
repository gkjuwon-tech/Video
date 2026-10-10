"""FND-02 테스트 공용 픽스처: 작은 결정론 장면(8x12)으로 7+3 계약 타입을 만든다."""
import math

import numpy as np
import pytest

from buc.core import contracts as C

H, W = 8, 12


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_prov(*taint, ns="engine", mode="engine", roots=()):
    return C.Prov(taint=frozenset(taint), mode=mode, roots=tuple(roots), ns=ns)


def make_theta(**kw):
    return C.Theta(**kw)


def make_obs(**kw):
    K = C.Theta().intrinsics(W, H)
    args = dict(K=K, R=rot_y(10.0), t=np.array([0.1, -0.2, 0.3]), width=W, height=H, k1=0.0, k2=0.0)
    args.update(kw)
    return C.Obs(**args)


def make_image(**kw):
    rng = np.random.default_rng(0)
    args = dict(pixels=rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8), color_space="srgb",
                prov=make_prov("USER", ns="user_vault", mode="user"), exif={"Model": "test"}, consent=True)
    args.update(kw)
    return C.InputImage(**args)


def make_read(image=None, **kw):
    image = image if image is not None else make_image()
    rng = np.random.default_rng(1)
    depth = np.linspace(1.0, 5.0, H * W, dtype=np.float32).reshape(H, W)
    depth[0, 0] = np.inf  # 하늘
    args = dict(image_lin=rng.random((H, W, 3)).astype(np.float32), depth=depth, obs=make_obs(),
                theta_in=C.Theta(focal_mm=50.0), input_hash=image.h_I, prov=image.prov,
                scale_s=1.25, sigma_s=0.1, level="L1", esc={"total": 0.2}, faces=({"bbox": [0, 0, 2, 2]},),
                fallbacks=("R3",), meta={"note": "fixture"})
    args.update(kw)
    return C.ReadPacket(**args)


def make_layers(n=2):
    layers = []
    for i in range(n):
        rgba = np.full((H, W, 4), 0.5, dtype=np.float32)
        rgba[..., 3] = 1.0 if i == n - 1 else 0.6
        layers.append(C.LDILayer(rgba=rgba, depth=np.full((H, W), 1.0 + 2.0 * i, dtype=np.float32)))
    return tuple(layers)


def make_world(read=None, **kw):
    read = read if read is not None else make_read()
    args = dict(obs=read.obs, layers=make_layers(), theta_in=read.theta_in, level=read.level,
                read_hash=read.content_hash(), prov=C.Prov.merge(read.prov, make_prov("ENGINE")),
                far_field=C.FarField(rgb=np.ones((4, 4, 3), dtype=np.float32)), meta={"lot": "near"})
    args.update(kw)
    return C.World(**args)


def make_shot(obs=None, n=5, fps=24.0, width=W, height=H, **kw):
    obs = obs if obs is not None else make_obs()
    c2w = np.repeat(obs.c2w()[None], n, axis=0).copy()
    c2w[:, 2, 3] += np.arange(n) * 0.05  # 천천히 전진
    K = np.repeat(C.Theta().intrinsics(width, height)[None], n, axis=0)
    args = dict(c2w=c2w, t=np.arange(n) / fps, K=K, focus_m=np.full(n, 3.0), width=width, height=height,
                fps=fps, theta=C.Theta(), move="push_in", rig="slider", seed=7)
    args.update(kw)
    return C.Shot(**args)


def make_frames(shot=None, **kw):
    shot = shot if shot is not None else make_shot()
    n = shot.n_frames
    args = dict(rgb=np.full((n, shot.height, shot.width, 3), 0.25, dtype=np.float32), fps=shot.fps,
                aov={"depth": np.ones((n, shot.height, shot.width), dtype=np.float32)},
                shot_hash=shot.content_hash(), world_hash="")
    args.update(kw)
    return C.Frames(**args)


def make_kpi(**kw):
    args = dict(ffd=0.02, ffg=0.01, de00=1.5, ir_geo=0.3, ir_area=0.4, ir_info=0.2, m_eff=1.1, esc=0.3,
                cheat_share=0.1, tfu=12.0, gpu_h=0.5, wall_s=30.0, ladder_step="safe", proof_of_shot="ab" * 32)
    args.update(kw)
    return C.KPI(**args)


@pytest.fixture
def chain():
    """S-1 → S2 → S8 → S9-10 → S11 한 줄."""
    image = make_image()
    read = make_read(image)
    world = make_world(read)
    shot = make_shot(read.obs)
    frames = make_frames(shot, world_hash=world.content_hash())
    kpi = make_kpi()
    return dict(image=image, read=read, world=world, shot=shot, frames=frames, kpi=kpi)
