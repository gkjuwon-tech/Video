"""PRD-03 테스트 공용: 입력 생성, 컨텍스트, 계약서의 공식을 독립적으로 다시 쓴 오라클."""
import numpy as np

from buc.core import contracts as C
from buc.pipeline import spine as S

NAME_FOR = {"S-1": "classic_read", "S2": "flat_world", "S8": "fixed_move", "S8.5": "identity_polish"}
STUB_STAGES = ("S-1", "S2", "S8", "S8.5")
USER_PROV = C.Prov(taint=frozenset({"USER"}), mode="user", ns="user_vault")


def make_image(h=24, w=40, seed=0, dtype="uint8", cs="srgb", prov=USER_PROV):
    rng = np.random.default_rng(seed)
    if dtype == "uint8":
        px = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    elif dtype == "uint16":
        px = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
    else:
        px = rng.random((h, w, 3)).astype(np.float32)
    return C.InputImage(pixels=px, color_space=cs, prov=prov)


def impls():
    from buc.pipeline.stubs import pack
    return pack.stub_impls()


def make_ctx(stage, params=None, seed=12345):
    p = dict(params or {})
    p["_seed"] = seed
    return S.StageContext(stage=stage, seed=seed, params=p, impl=impls()[stage])


def call(stage, inputs, params=None, seed=12345):
    return impls()[stage].fn(inputs, make_ctx(stage, params, seed))


def srgb_eotf(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def ground_depth(h, w, fy, cam_h=1.5, zn=0.5, zf=50.0):
    """계약서의 수평선 가정 평면 깊이 (H,W) float64."""
    cy = (h - 1) / 2.0
    out = np.empty((h, w))
    for v in range(h):
        d = v - cy
        out[v, :] = np.clip(fy * cam_h / d, zn, zf) if d > 0 else zf
    return out


def manual_read(depth, image_lin=None, obs=None, theta=None, prov=USER_PROV, seed=0):
    """판독 스텁을 거치지 않고 원하는 깊이로 ReadPacket 을 직접 만든다."""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    theta = theta or C.Theta(aspect=w / h)
    if obs is None:
        obs = C.Obs(K=theta.intrinsics(w, h), R=np.eye(3), t=np.zeros(3), width=w, height=h)
    if image_lin is None:
        image_lin = np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)
    return C.ReadPacket(image_lin=image_lin, depth=depth, obs=obs, theta_in=theta, input_hash="a" * 64, prov=prov)


def composite(world):
    """앞→뒤 직선 알파 over, float64 -> (H,W,3)."""
    h, w = world.layers[0].rgba.shape[:2]
    rgb = np.zeros((h, w, 3))
    trans = np.ones((h, w, 1))
    for ly in world.layers:
        a = ly.rgba[..., 3:4].astype(np.float64)
        rgb += ly.rgba[..., :3].astype(np.float64) * a * trans
        trans = trans * (1 - a)
    return rgb


def tilted_obs(w, h):
    """회전·이동이 있는 비단위 관측 카메라."""
    th = C.Theta(aspect=w / h)
    a = np.radians(12.0)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return C.Obs(K=th.intrinsics(w, h), R=R, t=np.array([0.3, -0.2, 1.1]), width=w, height=h)
