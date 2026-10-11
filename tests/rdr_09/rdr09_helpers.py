"""RDR-09 테스트 공용 도우미: 아주 작은 결정론 LDI 월드와 카메라 경로.

구현(buc.render.pos)은 참조하지 않는다. 오라클은 FND-03 det 와 numpy 로만 쓴다."""
import math

import numpy as np

from buc.core import contracts as C
from buc.core import geom

W, H = 48, 32
FOV = 60.0
I4 = np.eye(4)


def FX():
    return (W / 2.0) / math.tan(math.radians(FOV) / 2.0)


def K_default():
    return geom.intrinsics_from_fov(FOV, W, H)


def make_obs(mask=None):
    w2c = np.eye(4)
    return C.Obs(K=K_default(), R=w2c[:3, :3], t=w2c[:3, 3], width=W, height=H, mask=mask)


def rand_tex(seed, alpha=1.0):
    rng = np.random.default_rng(seed)
    rgba = np.empty((H, W, 4), np.float32)
    rgba[..., :3] = rng.uniform(0.1, 0.9, (H, W, 3))
    rgba[..., 3] = alpha
    return rgba


def smooth_tex():
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rgba = np.empty((H, W, 4), np.float64)
    rgba[..., 0] = 0.5 + 0.25 * np.sin(2 * np.pi * u / 31.0)
    rgba[..., 1] = 0.5 + 0.25 * np.cos(2 * np.pi * (u / 27.0 + v / 33.0))
    rgba[..., 2] = 0.2 + 0.6 * (u / W)
    rgba[..., 3] = 1.0
    return rgba.astype(np.float32)


def layer(rgba, depth):
    rgba = np.asarray(rgba, np.float32)
    d = np.broadcast_to(np.asarray(depth, np.float32), rgba.shape[:2]).copy()
    return C.LDILayer(rgba=rgba, depth=d)


def make_world(layers, obs=None):
    obs = make_obs() if obs is None else obs
    return C.World(obs=obs, layers=tuple(layers), theta_in=C.Theta(), level="L1", read_hash="r" * 8, prov=C.Prov())


def two_layer_world(seed=40, mask=None):
    """앞 층: 가운데 창만 불투명인 카드(깊이 2.5), 뒤 층: 매끈한 카드(깊이 5)."""
    front = rand_tex(seed, alpha=0.0)
    front[6:26, 10:38, 3] = 1.0
    return make_world([layer(front, 2.5), layer(smooth_tex(), 5.0)], make_obs(mask=mask))


def translate(c2w, d):
    T = np.array(c2w, np.float64, copy=True)
    T[:3, 3] = T[:3, 3] + np.asarray(d, np.float64)
    return T


def yaw(c2w, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.array(c2w, np.float64, copy=True)
    T[:3, :3] = T[:3, :3] @ Ry
    return T


def forward_shot(n=3, step=0.06, deg=1.5, fps=24.0):
    """앞으로 걸어 들어가며 살짝 도는 n 프레임 경로(프레임마다 K 도 약간 다르다)."""
    c2w = np.stack([yaw(translate(I4, (step * i, 0.01 * i, 0.05 * i)), deg * i) for i in range(n)])
    K = np.stack([K_default() for _ in range(n)])
    for i in range(n):
        K[i][0, 0] *= 1 + 0.03 * i
        K[i][1, 1] *= 1 + 0.03 * i
    return C.Shot(c2w=c2w, t=np.arange(n) / fps, K=K, focus_m=np.full(n, 3.0), width=W, height=H, fps=fps,
                  theta=C.Theta(), move="push-in")


def shot_from_arrays(c2w, K, focus_m, like):
    n = len(c2w)
    return C.Shot(c2w=c2w, t=np.arange(n) / like.fps, K=K, focus_m=focus_m, width=like.width, height=like.height,
                  fps=like.fps, theta=like.theta, move=like.move, rig=like.rig, seed=like.seed)


def synth_frames(n=4, h=6, w=5, seed=0, with_aov=True):
    """렌더 없이 만든 합성 Frames(해시·왕복 비교 단위 테스트용)."""
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(0.0, 1.0, (n, h, w, 3)).astype(np.float32)
    aov = {}
    if with_aov:
        aov = {
            "depth": rng.uniform(1.0, 5.0, (n, h, w)).astype(np.float32),
            "alpha": rng.uniform(0.0, 1.0, (n, h, w)).astype(np.float32),
            "layer_id": rng.integers(-1, 3, (n, h, w)).astype(np.int16),
            "normal": rng.uniform(-1.0, 1.0, (n, h, w, 3)).astype(np.float32),
            "atom_id": rng.integers(-1, 90, (n, h, w)).astype(np.int64),
            "flow": rng.uniform(-3.0, 3.0, (n, h, w, 2)).astype(np.float32),
            "flow_valid": rng.integers(0, 2, (n, h, w)).astype(bool),
        }
    return C.Frames(rgb=rgb, fps=24.0, aov=aov, shot_hash="s" * 8, world_hash="w" * 8)
