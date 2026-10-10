"""PRD-03 고전 판독 스텁(S-1 ``{image}→{read}``): 전달함수 + 수평선 가정 평면 깊이.

계약서: ops/harness/contracts/PRD-03.md. 실물 판독이 없거나 실패할 때의 폴백이다. 색역 변환은 하지 않는다.
"""
import numpy as np

from buc.core import contracts as C
from buc.pipeline.stubs._common import check_keys, finite_float

NAME = "classic_read"
VERSION = "0.1.0"
_PARAMS = ("focal_mm", "cam_height_m", "z_near_m", "z_far_m")


def _eotf(color_space: str, v: np.ndarray) -> np.ndarray:
    """v >= 0 (float64). linear-srgb 항등, adobe-rgb v^2.2, 그 밖은 sRGB EOTF."""
    if color_space == "linear-srgb":
        return v
    if color_space == "adobe-rgb":
        return v ** 2.2
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _image_lin(img: C.InputImage) -> np.ndarray:
    px = np.asarray(img.pixels)
    if px.dtype == np.uint8:
        v = px.astype(np.float64) / 255.0
    elif px.dtype == np.uint16:
        v = px.astype(np.float64) / 65535.0
    else:
        v = px.astype(np.float64)
    v = np.maximum(v, 0.0)
    return _eotf(img.color_space, v).astype(np.float32)


def _ground_depth(H: int, W: int, fy: float, cam_h: float, zn: float, zf: float) -> np.ndarray:
    """수평선(주점 행 cy=(H-1)/2) 가정 평면 깊이. 행 v>cy 는 clip(fy·h/(v-cy)), 그 밖은 zf. 열마다 같다."""
    cy = (H - 1) / 2.0
    d = np.arange(H, dtype=np.float64) - cy
    row = np.full(H, zf, dtype=np.float64)
    below = d > 0.0
    row[below] = np.clip(fy * cam_h / d[below], zn, zf)
    return np.repeat(row[:, None], W, axis=1).astype(np.float32)


def classic_read(inputs: dict, ctx) -> dict:
    img = inputs["image"]
    p = check_keys(ctx.params, _PARAMS)
    focal = finite_float("focal_mm", p.get("focal_mm", 35.0), positive=True)
    cam_h = finite_float("cam_height_m", p.get("cam_height_m", 1.5), positive=True)
    zn = finite_float("z_near_m", p.get("z_near_m", 0.5), positive=True)
    zf = finite_float("z_far_m", p.get("z_far_m", 50.0), positive=True)
    if zn > zf:
        raise ValueError(f"z_near_m({zn}) 은 z_far_m({zf}) 이하여야 한다")

    H, W = int(img.pixels.shape[0]), int(img.pixels.shape[1])
    theta = C.Theta(aspect=W / H, focal_mm=focal)
    K = theta.intrinsics(W, H)
    obs = C.Obs(K=K, R=np.eye(3), t=np.zeros(3), width=W, height=H)
    depth = _ground_depth(H, W, float(K[1, 1]), cam_h, zn, zf)
    read = C.ReadPacket(
        image_lin=_image_lin(img),
        depth=depth,
        obs=obs,
        theta_in=theta,
        input_hash=img.h_I,
        prov=img.prov,
        scale_s=1.0,
        sigma_s=1.0,
        level="L1",
        depth_conf=np.zeros((H, W), dtype=np.float32),
        fallbacks=("stub:classic_read",),
        meta={"stub": NAME},
    )
    return {"read": read}
