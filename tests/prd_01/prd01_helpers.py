"""PRD-01 테스트 공용: 작은 결정론 스텁 구현(8x12)과 호출 계수기. PRD-03 스텁 팩에 의존하지 않는다."""
import numpy as np

from buc.core import contracts as C
from buc.pipeline import spine as S

H, W = 8, 12


def make_image(seed=0):
    rng = np.random.default_rng(seed)
    return C.InputImage(pixels=rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8), color_space="srgb",
                        prov=C.Prov(taint=frozenset({"USER"}), mode="user", ns="user_vault"))


def read_stub(inputs, ctx):
    img = inputs["image"]
    lin = (img.pixels.astype(np.float32) / 255.0) ** 2
    h, w = img.pixels.shape[:2]
    obs = C.Obs(K=C.Theta().intrinsics(w, h), R=np.eye(3), t=np.zeros(3), width=w, height=h)
    depth = np.linspace(1.0, 4.0, h * w, dtype=np.float32).reshape(h, w) * float(ctx.params.get("depth_scale", 1.0))
    read = C.ReadPacket(image_lin=lin.astype(np.float32), depth=depth, obs=obs, theta_in=C.Theta(),
                        input_hash=img.h_I, prov=img.prov)
    return {"read": read}


def world_stub(inputs, ctx):
    read = inputs["read"]
    h, w = read.depth.shape
    rgba = np.concatenate([np.clip(read.image_lin, 0, 1), np.ones((h, w, 1), np.float32)], axis=-1)
    gain = float(ctx.params.get("gain", 1.0))
    rgba[..., :3] = np.clip(rgba[..., :3] * gain, 0, 1)
    layer = C.LDILayer(rgba=rgba.astype(np.float32), depth=read.depth.copy())
    world = C.World(obs=read.obs, layers=(layer,), theta_in=read.theta_in, level=read.level,
                    read_hash=read.content_hash(), prov=C.Prov.merge(read.prov, C.Prov(taint=frozenset({"ENGINE"}))))
    return {"world": world}


def shot_stub(inputs, ctx):
    obs = inputs["read"].obs
    n = int(ctx.params.get("n", 3))
    dz = float(ctx.params.get("dz", 0.05))
    fps = 24.0
    c2w = np.repeat(obs.c2w()[None], n, axis=0).copy()
    c2w[:, 2, 3] += np.arange(n) * dz
    shot = C.Shot(c2w=c2w, t=np.arange(n) / fps, K=np.repeat(obs.K[None], n, axis=0), focus_m=np.full(n, 3.0),
                  width=obs.width, height=obs.height, fps=fps, theta=C.Theta(), move="push_in")
    return {"shot": shot}


def polish_stub(inputs, ctx):
    return {"world": inputs["world"]}


def render_stub(inputs, ctx):
    world, shot = inputs["world"], inputs["shot"]
    base = world.layers[0].rgba[..., :3]
    rng = np.random.default_rng(ctx.seed)
    noise = rng.random((shot.n_frames, shot.height, shot.width, 3)).astype(np.float32) * 0.01
    rgb = np.clip(base[None] * 0.9 + noise, 0.0, 1.0).astype(np.float32)
    return {"frames": C.Frames(rgb=rgb, fps=shot.fps, shot_hash=shot.content_hash(),
                               world_hash=world.content_hash())}


def report_stub(inputs, ctx):
    return {"kpi": C.KPI(ffd=float(inputs["frames"].rgb.mean()), wall_s=0.0)}


STUB_FNS = {"S-1": read_stub, "S2": world_stub, "S8": shot_stub, "S8.5": polish_stub, "S9-10": render_stub,
            "S11": report_stub}


class Counter:
    """스테이지별 fn 호출 횟수."""

    def __init__(self):
        self.n = {}

    def wrap(self, label, fn):
        def f(inputs, ctx):
            self.n[label] = self.n.get(label, 0) + 1
            return fn(inputs, ctx)
        return f

    def total(self):
        return sum(self.n.values())


def stub_registry(counter=None, version="0"):
    reg = S.Registry()
    for st, fn in STUB_FNS.items():
        f = counter.wrap(st, fn) if counter is not None else fn
        reg.register(S.Impl(stage=st, kind="stub", name=f"t_{st}", version=version, fn=f))
    return reg


def hashes(artifacts):
    return {k: v.content_hash() for k, v in artifacts.items()}
