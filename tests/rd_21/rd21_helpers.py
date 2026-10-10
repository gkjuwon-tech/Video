"""RD-21 테스트 공용: 입력, 가짜 단계, 계약서 공식을 독립적으로 다시 쓴 오라클."""
import math

import numpy as np

from buc.core import contracts as C

USER_PROV = C.Prov(taint=frozenset({"USER"}), mode="engine", ns="user_vault")
EXTGEN_PROV = C.Prov(taint=frozenset({"EXTGEN:soul"}), mode="engine", ns="ext_vault")
ALL_SIDS = tuple(f"R{i}" for i in range(14))
FAKE_NEEDS = {"R1": ("R0",), "R5": ("R0",), "R6": ("R1", "R5"), "R3": ("R5", "R6")}
FAKE_FOV = 60.0
FAKE_S = 2.0
FAKE_SIGMA = 0.1
FAKE_K1 = 0.01


def make_image(h=48, w=64, seed=0, dtype="uint8", cs="srgb", exif=None, prov=USER_PROV):
    rng = np.random.default_rng(seed)
    if dtype == "uint8":
        px = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    elif dtype == "uint16":
        px = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
    else:
        px = rng.random((h, w, 3)).astype(np.float32)
    return C.InputImage(pixels=px, color_space=cs, prov=prov, exif=dict(exif or {}))


def smooth_image(h=96, w=128, cs="srgb"):
    yy, xx = np.mgrid[0:h, 0:w]
    px = (np.stack([xx / w, yy / h, 0.5 + 0 * xx], -1) * 255).astype(np.uint8)
    px[h // 3:2 * h // 3, w // 3:2 * w // 3] = 200
    return C.InputImage(pixels=px, color_space=cs, prov=USER_PROV)


def srgb_eotf(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def classic_lin(image):
    """PRD-03 classic_read 전달함수 오라클."""
    p = image.pixels
    if p.dtype == np.uint8:
        v = p.astype(np.float64) / 255.0
    elif p.dtype == np.uint16:
        v = p.astype(np.float64) / 65535.0
    else:
        v = p.astype(np.float64)
    v = np.maximum(v, 0.0)
    cs = image.color_space
    if cs == "linear-srgb":
        out = v
    elif cs == "adobe-rgb":
        out = v ** 2.2
    else:
        out = srgb_eotf(v)
    return out.astype(np.float32)


def k_from_fov(fov_deg, W, H):
    f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, (W - 1) / 2.0], [0, f, (H - 1) / 2.0], [0, 0, 1.0]])


def plane_oracle(K, up, H, W, h=1.5, zn=0.5, zf=50.0):
    up = np.asarray(up, dtype=np.float64)
    Ki = np.linalg.inv(np.asarray(K, dtype=np.float64))
    out = np.empty((H, W))
    for v in range(H):
        for u in range(W):
            d = Ki @ np.array([u, v, 1.0])
            den = -(up @ d)
            out[v, u] = np.clip(h / den, zn, zf) if den > 1e-9 else zf
    return out


def up_torch(a, H, W):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.asarray(a, dtype=np.float64))[None, None]
    return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()


def fake_ld(h4, w4):
    yy, xx = np.mgrid[0:h4, 0:w4]
    return (0.3 + 0.05 * xx - 0.03 * yy).astype(np.float32)


def fake_rdis(h4, w4):
    yy, xx = np.mgrid[0:h4, 0:w4]
    return (0.04 * xx + 0.02 * yy).astype(np.float32)


def fake_outputs(sid, state):
    H, W = state.H, state.W
    meta = {"fake": sid}
    if sid == "R0":
        return {"image_lin": classic_lin(state.image), "fov_x_prior_deg": None, "meta": meta}
    if sid == "R1":
        return {"K": k_from_fov(FAKE_FOV, W, H), "k1": FAKE_K1, "k2": 0.0, "up_cam": np.array([0.0, -1.0, 0.0]),
                "f_source": "vp3", "meta": meta}
    if sid == "R5":
        h4, w4 = max(1, H // 4), max(1, W // 4)
        return {"log_depth": fake_ld(h4, w4), "r_dis_map": fake_rdis(h4, w4), "meta": meta}
    if sid == "R6":
        return {"s": FAKE_S, "sigma_s": FAKE_SIGMA, "reliable": True, "meta": meta}
    if sid == "R11":
        return {"level": "L1.5", "meta": meta}
    return {"meta": meta}


def fake_steps(calls=None, sids=ALL_SIDS, override=None, version="1", needs=None):
    """sid마다 유효한 출력을 내는 가짜 단계. override[sid](state, params) 가 있으면 그것을 쓴다."""
    from buc.read import pipeline as P
    override = dict(override or {})
    needs = dict(FAKE_NEEDS if needs is None else needs)
    out = []
    for sid in sids:
        def fn(state, params, _sid=sid):
            if calls is not None:
                calls.append(_sid)
            if _sid in override:
                return override[_sid](state, params)
            return fake_outputs(_sid, state)
        v = version.get(sid, "1") if isinstance(version, dict) else version
        out.append(P.ReadStep(sid, f"fake_{sid}", v, fn, tuple(needs.get(sid, ()))))
    return tuple(out)


def status_map(res):
    return {r.sid: r.status for r in res.records}
