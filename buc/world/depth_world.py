"""WLD-03 깊이 세계 빌더 v0: 미터 깊이 -> 불연속 찢기 -> 2층 LDI -> 2DGS(TSQ 행) 초기화.

S2 실물. ``ReadPacket`` 으로 ``Strata``(WLD-01) 와 ``World`` 를 짓는다. 고무 시트를 막는 세 장치:
  (1) 깊이 불연속에서 찢는다(깊이 평활·보간 없음, 화소 깊이 그대로),
  (2) 전경 뒤로 배경을 ``ext_px`` 만큼 잇는 숨은 층(LDI 층 1; 색은 임시 최근접 복사, WLD-08 이 덮는다),
  (3) 매팅(RD-18)이 주어지면 띠 화소를 ``alpha*F``(전경 깊이) 와 ``B``(배경 깊이)로 분해한다.
관측 시점 합성은 입력과 같다. 계약서: ops/harness/contracts/WLD-03.md.

numpy · scipy 만 모듈 수준에서 쓴다. torch 와 buc.render.*, buc.pipeline.spine 은 함수 안에서 늦게 불러온다.
import 부작용·전역 RNG 변경 없음, 입력 불변, 결정론.
"""
from __future__ import annotations

import dataclasses
import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy import ndimage as ndi

from buc.core import contracts as C
from buc.core import geom
from buc.world.strata import LEVEL_RANGE, OWNERS, Strata

__all__ = [
    "NAME", "VERSION", "N_LAYERS", "DepthWorldError", "DepthWorldConfig", "DepthWorld",
    "build_depth_world", "tsq_to_surfels", "depth_world_stage", "make_world_impl",
]

NAME = "depth_world"
VERSION = "0.1.0"
N_LAYERS = 2


class DepthWorldError(C.ContractError):
    """깊이 세계 빌드 중 내부 불변식 위반."""


# --------------------------------------------------------------------------- 설정

def _real(name: str, v: Any) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    return float(v)


@dataclass(frozen=True, kw_only=True)
class DepthWorldConfig:
    rel_jump: float = 0.10
    ext_px: float = 16.0
    near_m: float = 1e-3
    far_m: float = 1e4
    tsq_stride: int = 2
    alpha_eps: float = 1e-3
    opacity: float = 0.99
    scale_mult: float = 0.5
    edge_n: float = 2.0

    def __post_init__(self) -> None:
        rj = _real("rel_jump", self.rel_jump)
        if not (math.isfinite(rj) and rj > 0):
            raise ValueError(f"rel_jump: 유한한 양수여야 한다 ({self.rel_jump!r})")
        ext = _real("ext_px", self.ext_px)
        if not (math.isfinite(ext) and ext >= 0):
            raise ValueError(f"ext_px: 유한한 0 이상이어야 한다 ({self.ext_px!r})")
        nr = _real("near_m", self.near_m)
        if not (math.isfinite(nr) and nr > 0):
            raise ValueError(f"near_m: 유한한 양수여야 한다 ({self.near_m!r})")
        fr = _real("far_m", self.far_m)
        if not (math.isfinite(fr) and fr > nr):
            raise ValueError(f"far_m: near_m 보다 큰 유한값이어야 한다 ({self.far_m!r})")
        st = self.tsq_stride
        if isinstance(st, (bool, np.bool_)) or not isinstance(st, numbers.Integral) or int(st) < 1:
            raise ValueError(f"tsq_stride: 1 이상의 정수여야 한다 ({st!r})")
        ae = _real("alpha_eps", self.alpha_eps)
        if not (math.isfinite(ae) and 0.0 <= ae < 0.5):
            raise ValueError(f"alpha_eps: [0,0.5) 여야 한다 ({self.alpha_eps!r})")
        op = _real("opacity", self.opacity)
        if not (math.isfinite(op) and 0.0 < op <= 1.0):
            raise ValueError(f"opacity: (0,1] 여야 한다 ({self.opacity!r})")
        sm = _real("scale_mult", self.scale_mult)
        if not (math.isfinite(sm) and sm > 0):
            raise ValueError(f"scale_mult: 유한한 양수여야 한다 ({self.scale_mult!r})")
        en = _real("edge_n", self.edge_n)
        if not (math.isfinite(en) and 2.0 <= en <= 64.0):
            raise ValueError(f"edge_n: [2,64] 여야 한다 ({self.edge_n!r})")
        object.__setattr__(self, "tsq_stride", int(st))


def _meta_dict(cfg: DepthWorldConfig, n_tsq: int, n_tear_pairs: int, hidden_frac: float, matted_frac: float,
               sky_frac: float, has_matte: bool, has_far_field: bool) -> dict:
    return {
        "name": NAME,
        "version": VERSION,
        "cfg": {k: (int(v) if k == "tsq_stride" else float(v)) for k, v in dataclasses.asdict(cfg).items()},
        "n_layers": N_LAYERS,
        "n_tsq": int(n_tsq),
        "n_tear_pairs": int(n_tear_pairs),
        "hidden_frac": float(hidden_frac),
        "matted_frac": float(matted_frac),
        "sky_frac": float(sky_frac),
        "matte": bool(has_matte),
        "far_field": bool(has_far_field),
    }


# --------------------------------------------------------------------------- 결과

@dataclass(frozen=True, kw_only=True, eq=False)
class DepthWorld:
    strata: Strata
    world: C.World
    z: np.ndarray
    sky: np.ndarray
    near: np.ndarray
    far: np.ndarray
    hidden: np.ndarray
    matted: np.ndarray
    tear_x: np.ndarray
    tear_y: np.ndarray
    n_tsq: int
    tsq_layer: np.ndarray
    has_matte: bool = False
    has_far_field: bool = False
    cfg: DepthWorldConfig = dataclasses.field(default_factory=DepthWorldConfig)

    def meta(self) -> dict:
        return _meta_dict(self.cfg, self.n_tsq, int(self.tear_x.sum() + self.tear_y.sum()),
                          float(self.hidden.mean()), float(self.matted.mean()), float(self.sky.mean()),
                          self.has_matte, self.has_far_field)


# --------------------------------------------------------------------------- 매트 검증

def _check_matte(matte: Any, h: int, w: int) -> tuple:
    for name in ("alpha", "fg", "bg", "band"):
        if not hasattr(matte, name):
            raise TypeError(f"matte: 속성 {name!r} 가 없다")

    def arr(name, dtype, shape):
        try:
            a = np.asarray(getattr(matte, name), dtype=dtype)
        except (TypeError, ValueError) as e:
            raise ValueError(f"matte.{name}: 배열로 바꿀 수 없다 ({e})") from e
        if a.shape != shape:
            raise ValueError(f"matte.{name}: 모양 {a.shape} != {shape}")
        return a

    alpha = arr("alpha", np.float64, (h, w))
    fg = arr("fg", np.float64, (h, w, 3))
    bg = arr("bg", np.float64, (h, w, 3))
    band_raw = np.asarray(getattr(matte, "band"))
    if band_raw.dtype != np.bool_:
        raise ValueError(f"matte.band: bool 배열이어야 한다 (dtype {band_raw.dtype})")
    band = arr("band", bool, (h, w))
    for name, a in (("alpha", alpha), ("fg", fg), ("bg", bg)):
        if not np.isfinite(a).all():
            raise ValueError(f"matte.{name}: 비유한 값")
    if alpha.size and (alpha.min() < 0.0 or alpha.max() > 1.0):
        raise ValueError("matte.alpha: [0,1] 밖")
    if fg.size and fg.min() < 0.0:
        raise ValueError("matte.fg: 음수")
    if bg.size and bg.min() < 0.0:
        raise ValueError("matte.bg: 음수")
    return alpha, fg, bg, band


# --------------------------------------------------------------------------- 빌더

def _tears(z: np.ndarray, rel: float) -> tuple:
    h, w = z.shape
    ax, bx = z[:, :-1], z[:, 1:]
    tx = np.abs(ax - bx) > rel * np.minimum(ax, bx)
    ay, by = z[:-1, :], z[1:, :]
    ty = np.abs(ay - by) > rel * np.minimum(ay, by)
    near = np.zeros((h, w), bool)
    far = np.zeros((h, w), bool)
    near[:, :-1] |= tx & (ax < bx)
    near[:, 1:] |= tx & (bx < ax)
    far[:, :-1] |= tx & (ax > bx)
    far[:, 1:] |= tx & (bx > ax)
    near[:-1, :] |= ty & (ay < by)
    near[1:, :] |= ty & (by < ay)
    far[:-1, :] |= ty & (ay > by)
    far[1:, :] |= ty & (by > ay)
    return tx, ty, near, far


def _nearest(mask: np.ndarray) -> tuple:
    """mask 화소까지의 EDT 거리와 최근접 mask 화소 인덱스(행, 열)."""
    dist, idx = ndi.distance_transform_edt(~mask, return_indices=True)
    return dist, idx[0], idx[1]


def build_depth_world(read, *, cfg=None, matte=None, far_field=None) -> DepthWorld:
    if not isinstance(read, C.ReadPacket):
        raise TypeError(f"read: ReadPacket 이어야 한다 ({type(read).__name__})")
    if cfg is None:
        cfg = DepthWorldConfig()
    elif not isinstance(cfg, DepthWorldConfig):
        raise TypeError(f"cfg: DepthWorldConfig 또는 None 이어야 한다 ({type(cfg).__name__})")
    if far_field is not None and not isinstance(far_field, C.FarField):
        raise TypeError(f"far_field: FarField 또는 None 이어야 한다 ({type(far_field).__name__})")

    obs = read.obs
    H, W = int(obs.height), int(obs.width)
    I = np.asarray(read.image_lin, np.float64)
    D = np.asarray(read.depth, np.float64)
    rel = float(cfg.rel_jump)
    ext = float(cfg.ext_px)

    mt = None
    if matte is not None:
        mt = _check_matte(matte, H, W)

    # 1. 깊이
    with np.errstate(invalid="ignore"):
        sky = ~np.isfinite(D) | (D >= cfg.far_m)
        z = np.where(sky, float(cfg.far_m), np.maximum(D, float(cfg.near_m)))

    # 2. 찢기
    tear_x, tear_y, near, far = _tears(z, rel)

    # 3. 매팅
    matted = np.zeros((H, W), bool)
    zf = zb = None
    if mt is not None and mt[3].any() and near.any() and far.any():
        _, nr, nc = _nearest(near)
        _, fr, fc = _nearest(far)
        zf = z[nr, nc]
        zb = z[fr, fc]
        matted = mt[3] & (zb > zf * (1.0 + rel))

    # 4. 숨은 층
    hidden = np.zeros((H, W), bool)
    c_h = np.zeros((H, W, 3))
    z_h = z
    if far.any():
        d, qr, qc = _nearest(far)
        zq = z[qr, qc]
        hidden = (d <= ext) & (zq > z * (1.0 + rel)) & ~matted
        if mt is not None:
            use_b = mt[3][qr, qc]
            c_h = np.where(use_b[..., None], mt[2][qr, qc], I[qr, qc])
        else:
            c_h = I[qr, qc]
        z_h = zq

    # 5. 층
    raw0 = I.copy()
    a0 = np.ones((H, W))
    d0 = z.copy()
    raw1 = np.zeros((H, W, 3))
    a1 = np.zeros((H, W))
    d1 = z.copy()
    if matted.any():
        alpha, fg, bg, _band = mt
        am = np.where(alpha > cfg.alpha_eps, alpha, 0.0)
        raw0[matted] = fg[matted]
        a0[matted] = am[matted]
        d0[matted] = zf[matted]
        raw1[matted] = bg[matted]
        a1[matted] = 1.0
        d1[matted] = zb[matted]
    if hidden.any():
        raw1[hidden] = c_h[hidden]
        a1[hidden] = 1.0
        d1[hidden] = z_h[hidden]

    def layer(raw, a, d):
        rgba = np.concatenate([np.clip(raw, 0.0, 1.0), a[..., None]], axis=-1).astype(np.float32)
        return C.LDILayer(rgba=rgba, depth=d.astype(np.float32))

    layers = (layer(raw0, a0, d0), layer(raw1, a1, d1))

    # 6. TSQ 행 (층 저장값에서 계산해 World 와 일치시킨다)
    cols, tsq_layer = _tsq_rows(read, cfg, layers, (raw0, raw1), matted, hidden)
    n_tsq = int(tsq_layer.shape[0])

    # 7. Strata / World
    sky_frac, hidden_frac, matted_frac = float(sky.mean()), float(hidden.mean()), float(matted.mean())
    n_tear = int(tear_x.sum() + tear_y.sum())
    meta = _meta_dict(cfg, n_tsq, n_tear, hidden_frac, matted_frac, sky_frac, matte is not None,
                      far_field is not None)
    strata = Strata(obs=obs, theta_in=read.theta_in, level=read.level, read_hash=read.content_hash(),
                    prov=read.prov, seed=0, meta={"depth_world": meta})
    for ly in layers:
        strata.add_layer(ly)
    if far_field is not None:
        strata.set_far_field(far_field)
    if n_tsq:
        strata.add_tsq(**cols)
    world = strata.to_world()

    return DepthWorld(strata=strata, world=world, z=z, sky=sky, near=near, far=far, hidden=hidden,
                      matted=matted, tear_x=tear_x, tear_y=tear_y, n_tsq=n_tsq, tsq_layer=tsq_layer,
                      has_matte=matte is not None, has_far_field=far_field is not None, cfg=cfg)


def _tsq_rows(read, cfg: DepthWorldConfig, layers: tuple, raws: tuple, matted: np.ndarray,
              hidden: np.ndarray) -> tuple:
    obs = read.obs
    H, W = int(obs.height), int(obs.width)
    s = int(cfg.tsq_stride)
    K = np.asarray(obs.K, np.float64)
    R = np.asarray(obs.R, np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    vv, uu = np.mgrid[0:H:s, 0:W:s]
    if read.depth_conf is None:
        unc0 = np.full((H, W), 255, np.uint8)
    else:
        conf = np.asarray(read.depth_conf, np.float64)
        unc0 = np.rint(255.0 * (1.0 - conf)).astype(np.uint8)

    parts = {k: [] for k in ("u", "v", "z", "a", "color", "own", "unc", "layer")}
    for li, (ly, raw) in enumerate(zip(layers, raws)):
        a = ly.rgba[..., 3][vv, uu].astype(np.float64)
        ok = a > cfg.alpha_eps
        if not ok.any():
            continue
        pv, pu = vv[ok], uu[ok]
        parts["u"].append(pu.astype(np.float64))
        parts["v"].append(pv.astype(np.float64))
        parts["z"].append(ly.depth[pv, pu].astype(np.float64))
        parts["a"].append(a[ok])
        parts["color"].append(np.maximum(raw[pv, pu], 0.0))
        m = matted[pv, pu]
        if li == 0:
            own = np.where(m, OWNERS["INPUT_PART"], OWNERS["INPUT"])
            unc = unc0[pv, pu]
        else:
            own = np.where(m, OWNERS["INPUT_PART"], np.where(hidden[pv, pu], OWNERS["ENGINE"], OWNERS["NONE"]))
            unc = np.full(pv.shape, 255, np.uint8)
        parts["own"].append(own.astype(np.uint16))
        parts["unc"].append(unc.astype(np.uint8))
        parts["layer"].append(np.full(pv.shape, li, np.int8))
    if not parts["u"]:
        return {}, np.zeros(0, np.int8)

    cat = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    n = cat["u"].shape[0]
    u, v, zz, a = cat["u"], cat["v"], cat["z"], cat["a"]
    mu = geom.unproject_world(K, R, np.asarray(obs.t, np.float64), np.stack([u, v], axis=1), zz,
                              float(obs.k1), float(obs.k2))
    normal = np.tile(-R[2, :], (n, 1))
    s_uv = np.stack([cfg.scale_mult * s * zz / fx, cfg.scale_mult * s * zz / fy], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.clip(np.rint(np.log2(8.0 * fx / (s * zz))), LEVEL_RANGE[0], LEVEL_RANGE[1])
    cols = {
        "mu": mu.astype(np.float32),
        "normal": normal.astype(np.float32),
        "s_uv": s_uv.astype(np.float32),
        "opacity": (cfg.opacity * a).astype(np.float32),
        "edge_n": np.full(n, cfg.edge_n, np.float32),
        "color": cat["color"].astype(np.float32),
        "level": lv.astype(np.int8),
        "sem": np.zeros(n, np.uint16),
        "own": cat["own"],
        "unc": cat["unc"],
    }
    return cols, cat["layer"]


# --------------------------------------------------------------------------- RDR-01 이음매

def _quat_wxyz(Rm: np.ndarray) -> np.ndarray:
    """회전행렬 (N,3,3) -> 쿼터니언 (N,4) wxyz (Shepperd, 결정론)."""
    n = Rm.shape[0]
    q = np.empty((n, 4))
    m00, m11, m22 = Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]
    tr = m00 + m11 + m22
    for i in range(n):
        m = Rm[i]
        if tr[i] > 0:
            sq = math.sqrt(tr[i] + 1.0) * 2.0
            q[i] = [0.25 * sq, (m[2, 1] - m[1, 2]) / sq, (m[0, 2] - m[2, 0]) / sq, (m[1, 0] - m[0, 1]) / sq]
        elif m00[i] > m11[i] and m00[i] > m22[i]:
            sq = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[i] = [(m[2, 1] - m[1, 2]) / sq, 0.25 * sq, (m[0, 1] + m[1, 0]) / sq, (m[0, 2] + m[2, 0]) / sq]
        elif m11[i] > m22[i]:
            sq = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[i] = [(m[0, 2] - m[2, 0]) / sq, (m[0, 1] + m[1, 0]) / sq, 0.25 * sq, (m[1, 2] + m[2, 1]) / sq]
        else:
            sq = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[i] = [(m[1, 0] - m[0, 1]) / sq, (m[0, 2] + m[2, 0]) / sq, (m[1, 2] + m[2, 1]) / sq, 0.25 * sq]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def tsq_to_surfels(cols: Mapping[str, np.ndarray], obs, *, dtype=None):
    """TSQ 열 -> RDR-01 ``Surfels``. torch 는 여기서 늦게 불러온다."""
    import torch

    from buc.render import splat2d

    if not isinstance(obs, C.Obs):
        raise TypeError(f"obs: Obs 여야 한다 ({type(obs).__name__})")
    dt = torch.float32 if dtype is None else dtype
    for k in ("mu", "normal", "s_uv", "opacity", "color"):
        if k not in cols:
            raise ValueError(f"cols: 열 {k!r} 가 없다")
    mu = np.asarray(cols["mu"], np.float64)
    n = mu.shape[0]
    if n == 0:
        return splat2d.Surfels.empty(dt)
    nrm = np.asarray(cols["normal"], np.float64)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    R0 = np.asarray(obs.R, np.float64)[0, :]
    tu = R0[None, :] - np.sum(R0[None, :] * nrm, axis=1, keepdims=True) * nrm
    tu = tu / np.linalg.norm(tu, axis=1, keepdims=True)
    tv = np.cross(nrm, tu)
    rot = np.stack([tu, tv, nrm], axis=2)  # 열 = (t_u, t_v, n)
    quats = _quat_wxyz(rot)
    s_uv = np.asarray(cols["s_uv"], np.float64)
    op = np.clip(np.asarray(cols["opacity"], np.float64), 1e-6, 1.0 - 1e-6)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)  # noqa: E731
    return splat2d.Surfels(means=t(mu), quats=t(quats), log_scales=t(np.log(s_uv)),
                           opacity_logits=t(np.log(op / (1.0 - op))),
                           colors=t(np.asarray(cols["color"], np.float64)))


# --------------------------------------------------------------------------- S2 스테이지

def depth_world_stage(inputs: dict, ctx) -> dict:
    params = {k: v for k, v in dict(ctx.params).items() if k != "_seed"}
    known = {f.name for f in dataclasses.fields(DepthWorldConfig)}
    bad = sorted(set(params) - known)
    if bad:
        raise ValueError(f"depth_world: 모르는 파라미터 {bad}")
    cfg = DepthWorldConfig(**params)
    res = build_depth_world(inputs["read"], cfg=cfg)
    return {"world": res.world}


def make_world_impl():
    from buc.pipeline import spine
    return spine.Impl("S2", "real", NAME, VERSION, depth_world_stage)
