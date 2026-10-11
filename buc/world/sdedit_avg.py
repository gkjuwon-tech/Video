"""WLD-08 뷰별 SDEdit + 평균 역투영 폴백: 실제 화면에 나가는 발명 채움 (S2 확장).

WLD-03 ``DepthWorld`` 의 숨은 층(LDI 층 1, ``hidden`` 텍셀; 색은 임시 최근접 복사)을 교사 발명으로 덮는다.
관측 시점에서 +-20 도 이내 가상 카메라 몇 대로 세계를 찍고(RDR-02 ``render_ldi``), 한 번의 lockstep SDEdit(sigma~0.5)
으로 다시 그리게 한 뒤, 각 숨은 텍셀을 그 텍셀이 보이는 뷰들에 역투영해 선형광 산술평균으로 색을 정한다.
관측 영역(층 0, 층 1 의 비숨김)과 층 알파·깊이는 건드리지 않는다.

계약서: ops/harness/contracts/WLD-08.md.
교사 샘플러는 덕 타입이다(TCH-02 ``TeacherSampler`` 와 호환). ``diffusers``·``transformers`` 는 부르지 않고, torch 는
함수 안에서 늦게 불러온다. 전역 RNG 불변, 입력 불변, 결정론.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import numbers
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import ndimage as ndi

from buc.core import color
from buc.core import contracts as C
from buc.core import geom
from buc.world.depth_world import DepthWorld, DepthWorldConfig, build_depth_world
from buc.world.strata import OWNERS, Strata

__all__ = [
    "NAME", "VERSION", "SDEditAvgError", "SDEditAvgConfig", "ViewSpec", "SDEditAvgResult",
    "auto_pivot_m", "plan_views", "sdedit_avg_fill", "make_world_impl",
]

NAME = "sdedit_avg"
VERSION = "0.1.0"

_SOLID_ALPHA = 0.99          # 교사에게 주는 영상에서 "채워진" 화소의 알파 하한
_Z_MIN = 1e-3                # 투표에 필요한 최소 카메라 z (m)
_SIGMA_END_TOL = 1e-9


class SDEditAvgError(C.ContractError):
    """SDEdit 평균 역투영 중 교사 출력·내부 불변식 위반."""


# --------------------------------------------------------------------------- 검증 도우미

def _is_int(v: Any) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _real(name: str, v: Any) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    return float(v)


def _int_in(name: str, v: Any, lo: int, hi: Optional[int] = None) -> int:
    if not _is_int(v):
        raise ValueError(f"{name}: 정수여야 한다 ({v!r})")
    x = int(v)
    if x < lo or (hi is not None and x > hi):
        rng = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ValueError(f"{name}: {rng} 여야 한다 ({v!r})")
    return x


# --------------------------------------------------------------------------- 설정

@dataclass(frozen=True, kw_only=True)
class SDEditAvgConfig:
    sigma: float = 0.5
    max_angle_deg: float = 20.0
    n_per_side: int = 2
    pivot_m: Optional[float] = None
    erode_px: int = 2
    depth_tol_rel: float = 0.05
    min_votes: int = 1
    prompt: str = "a photograph"

    def __post_init__(self) -> None:
        s = _real("sigma", self.sigma)
        if not (math.isfinite(s) and 0.0 < s <= 1.0):
            raise ValueError(f"sigma: (0,1] 의 유한값이어야 한다 ({self.sigma!r})")
        a = _real("max_angle_deg", self.max_angle_deg)
        if not (math.isfinite(a) and 0.0 < a <= 45.0):
            raise ValueError(f"max_angle_deg: (0,45] 여야 한다 ({self.max_angle_deg!r})")
        n = _int_in("n_per_side", self.n_per_side, 1, 8)
        pv = self.pivot_m
        if pv is not None:
            pv = _real("pivot_m", pv)
            if not (math.isfinite(pv) and pv > 0.0):
                raise ValueError(f"pivot_m: None 또는 유한한 양수여야 한다 ({self.pivot_m!r})")
        er = _int_in("erode_px", self.erode_px, 0, 8)
        tol = _real("depth_tol_rel", self.depth_tol_rel)
        if not (math.isfinite(tol) and 0.0 < tol <= 0.5):
            raise ValueError(f"depth_tol_rel: (0,0.5] 여야 한다 ({self.depth_tol_rel!r})")
        mv = _int_in("min_votes", self.min_votes, 1)
        if not isinstance(self.prompt, str):
            raise ValueError(f"prompt: str 이어야 한다 ({self.prompt!r})")
        object.__setattr__(self, "sigma", s)
        object.__setattr__(self, "max_angle_deg", a)
        object.__setattr__(self, "n_per_side", n)
        object.__setattr__(self, "pivot_m", pv)
        object.__setattr__(self, "erode_px", er)
        object.__setattr__(self, "depth_tol_rel", tol)
        object.__setattr__(self, "min_votes", mv)


# --------------------------------------------------------------------------- 뷰

@dataclass(frozen=True, kw_only=True, eq=False)
class ViewSpec:
    c2w: Any
    K: Any
    width: int
    height: int
    angle_deg: float
    k1: float = 0.0
    k2: float = 0.0

    def __post_init__(self) -> None:
        try:
            c2w = np.array(self.c2w, dtype=np.float64)
            K = np.array(self.K, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"ViewSpec: c2w/K 를 배열로 바꿀 수 없다 ({e})") from e
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError(f"ViewSpec.c2w: 유한한 (4,4) 여야 한다 (shape {c2w.shape})")
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"ViewSpec.K: 유한한 (3,3) 이어야 한다 (shape {K.shape})")
        w = _int_in("width", self.width, 1)
        h = _int_in("height", self.height, 1)
        ang = _real("angle_deg", self.angle_deg)
        k1 = _real("k1", self.k1)
        k2 = _real("k2", self.k2)
        if not all(math.isfinite(x) for x in (ang, k1, k2)):
            raise ValueError("ViewSpec: angle_deg/k1/k2 는 유한해야 한다")
        c2w.setflags(write=False)
        K.setflags(write=False)
        object.__setattr__(self, "c2w", c2w)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "width", w)
        object.__setattr__(self, "height", h)
        object.__setattr__(self, "angle_deg", ang)
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "k2", k2)


def _check_dw(dw: Any) -> None:
    if not isinstance(dw, DepthWorld):
        raise TypeError(f"dw: DepthWorld 이어야 한다 ({type(dw).__name__})")


def _check_cfg(cfg: Any) -> SDEditAvgConfig:
    if cfg is None:
        return SDEditAvgConfig()
    if not isinstance(cfg, SDEditAvgConfig):
        raise TypeError(f"cfg: SDEditAvgConfig 또는 None 이어야 한다 ({type(cfg).__name__})")
    return cfg


def auto_pivot_m(dw: DepthWorld) -> float:
    """피벗 거리: 전경 경계(near) 깊이의 중앙값. near 가 없으면 전체 깊이의 중앙값."""
    _check_dw(dw)
    z = np.asarray(dw.z, np.float64)
    near = np.asarray(dw.near, bool)
    return float(np.median(z[near])) if near.any() else float(np.median(z))


def plan_views(dw: DepthWorld, cfg: Optional[SDEditAvgConfig] = None) -> tuple:
    """관측 카메라 좌표계의 피벗 P=(0,0,d) 를 도는 y축 오빗 뷰. theta 증가순, 0 제외."""
    _check_dw(dw)
    cfg = _check_cfg(cfg)
    obs = dw.world.obs
    d = float(cfg.pivot_m) if cfg.pivot_m is not None else auto_pivot_m(dw)
    c2w_obs = geom.invert_pose(geom.make_pose(obs.R, obs.t))
    n, amax = cfg.n_per_side, cfg.max_angle_deg
    angles = [-amax * j / n for j in range(n, 0, -1)] + [amax * j / n for j in range(1, n + 1)]
    out = []
    for ang in angles:
        th = math.radians(ang)
        s, c = math.sin(th), math.cos(th)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        T[:3, 3] = [-d * s, 0.0, d * (1.0 - c)]
        out.append(ViewSpec(c2w=c2w_obs @ T, K=obs.K, width=int(obs.width), height=int(obs.height),
                            angle_deg=float(ang), k1=float(obs.k1), k2=float(obs.k2)))
    return tuple(out)


# --------------------------------------------------------------------------- 결과

def _meta_dict(cfg: SDEditAvgConfig, views: tuple, view_disocc_frac: np.ndarray, view_hole_frac: np.ndarray,
               steps: int, sigma_snapped: float, hidden_px: int, filled_px: int, unfilled_px: int) -> dict:
    return {
        "name": NAME,
        "version": VERSION,
        "cfg": {k: (None if v is None else (v if isinstance(v, str) else (int(v) if _is_int(v) else float(v))))
                for k, v in dataclasses.asdict(cfg).items()},
        "n_views": int(len(views)),
        "angles_deg": [float(v.angle_deg) for v in views],
        "sigma_requested": float(cfg.sigma),
        "sigma_snapped": float(sigma_snapped),
        "steps": int(steps),
        "hidden_px": int(hidden_px),
        "filled_px": int(filled_px),
        "unfilled_px": int(unfilled_px),
        "filled_frac": float(filled_px / hidden_px) if hidden_px else 0.0,
        "view_disocc_frac": [float(x) for x in view_disocc_frac],
        "view_hole_frac": [float(x) for x in view_hole_frac],
    }


@dataclass(frozen=True, kw_only=True, eq=False)
class SDEditAvgResult:
    strata: Strata
    world: C.World
    filled: np.ndarray
    votes: np.ndarray
    unfilled: np.ndarray
    views: tuple
    view_disocc_frac: np.ndarray
    view_hole_frac: np.ndarray
    cfg: SDEditAvgConfig
    steps: int
    sigma_snapped: float

    def meta(self) -> dict:
        f, u = int(self.filled.sum()), int(self.unfilled.sum())
        return _meta_dict(self.cfg, self.views, self.view_disocc_frac, self.view_hole_frac, self.steps,
                          self.sigma_snapped, f + u, f, u)


# --------------------------------------------------------------------------- 샘플러 규약

def _check_sampler(sampler: Any) -> None:
    teacher = getattr(sampler, "teacher", None)
    table = getattr(sampler, "table", None)
    if teacher is None:
        raise TypeError("sampler: 속성 'teacher' 가 없다")
    if table is None:
        raise TypeError("sampler: 속성 'table' 가 없다")
    if not callable(getattr(sampler, "sdedit", None)):
        raise TypeError("sampler: 메서드 'sdedit' 가 없다")
    for name in ("encode_image", "decode_latent"):
        if not callable(getattr(teacher, name, None)):
            raise TypeError(f"sampler.teacher: 메서드 {name!r} 가 없다")
    spec = getattr(teacher, "spec", None)
    if spec is None or not callable(getattr(spec, "check_image_size", None)):
        raise TypeError("sampler.teacher.spec.check_image_size 가 없다")
    if not callable(getattr(table, "snap", None)):
        raise TypeError("sampler.table: 메서드 'snap' 이 없다")
    try:
        len(table)
    except TypeError as e:
        raise TypeError("sampler.table: len() 을 지원해야 한다") from e


def _check_views(views: Any, obs: C.Obs) -> tuple:
    try:
        vs = tuple(views)
    except TypeError as e:
        raise TypeError(f"views: ViewSpec 시퀀스여야 한다 ({type(views).__name__})") from e
    if not vs:
        raise ValueError("views: 비어 있다")
    for i, v in enumerate(vs):
        if not isinstance(v, ViewSpec):
            raise TypeError(f"views[{i}]: ViewSpec 이어야 한다 ({type(v).__name__})")
        if (v.width, v.height) != (int(obs.width), int(obs.height)):
            raise ValueError(f"views[{i}]: 크기 {(v.width, v.height)} != 관측 {(int(obs.width), int(obs.height))}")
    return vs


# --------------------------------------------------------------------------- 렌더 / 투표

def _render_view(world: C.World, v: ViewSpec):
    from buc.render.quads import render_ldi
    return render_ldi(world, c2w=v.c2w, K=v.K, width=v.width, height=v.height, k1=v.k1, k2=v.k2)


def _teacher_image_lin(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha<0.99 화소를 가장 가까운 alpha>=0.99 화소의 색으로 대체한 선형 RGB (H,W,3) float64."""
    rgb = np.asarray(rgb, np.float64)
    solid = np.asarray(alpha) >= _SOLID_ALPHA
    if solid.all() or not solid.any():
        return rgb
    idx = ndi.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    return rgb[idx[0], idx[1]]


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """img (H,W,3) 의 쌍선형 표본(화소 중심 = 정수 좌표, 가장자리 클램프). u,v: (N,)"""
    H, W = img.shape[:2]
    u = np.clip(u, 0.0, W - 1.0)
    v = np.clip(v, 0.0, H - 1.0)
    x0 = np.minimum(np.floor(u).astype(np.int64), W - 1)
    y0 = np.minimum(np.floor(v).astype(np.int64), H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def _core_mask(layer_id: np.ndarray, erode_px: int) -> np.ndarray:
    m = layer_id == 1
    if erode_px <= 0:
        return m
    return ndi.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=int(erode_px), border_value=0)


# --------------------------------------------------------------------------- 본체

def sdedit_avg_fill(dw: DepthWorld, sampler, *, cfg: Optional[SDEditAvgConfig] = None, views=None,
                    seed: int = 0) -> SDEditAvgResult:
    _check_dw(dw)
    cfg = _check_cfg(cfg)
    _check_sampler(sampler)
    obs = dw.world.obs
    H, W = int(obs.height), int(obs.width)
    if views is None:
        vs = plan_views(dw, cfg)
    else:
        vs = _check_views(views, obs)
    if not _is_int(seed) or int(seed) < 0:
        raise ValueError(f"seed: 0 이상의 정수여야 한다 ({seed!r})")
    seed = int(seed)

    table = sampler.table
    sn = table.snap(cfg.sigma)
    steps = int(len(table) - int(sn.index))
    sigma_snapped = float(sn.sigma)
    n = len(vs)

    hidden = np.asarray(dw.hidden, bool)
    renders = [_render_view(dw.world, v) for v in vs]
    disocc = np.array([float((r.layer_id == 1).mean()) for r in renders], np.float64)
    holes = np.array([float(((r.alpha < 0.5) & (r.layer_id != 1)).mean()) for r in renders], np.float64)

    votes = np.zeros((H, W), np.int16)
    filled = np.zeros((H, W), bool)
    new_rgb = np.asarray(dw.world.layers[1].rgba[..., :3], np.float32).copy()

    if hidden.any():
        y_imgs = _invent(dw, sampler, cfg, vs, renders, seed, steps, sn)
        rows, cols = np.nonzero(hidden)
        z1 = np.asarray(dw.world.layers[1].depth, np.float64)[rows, cols]
        uv_src = np.stack([cols, rows], axis=1).astype(np.float64)
        X = geom.unproject_world(obs.K, obs.R, obs.t, uv_src, z1, float(obs.k1), float(obs.k2))
        acc = np.zeros((rows.shape[0], 3), np.float64)
        cnt = np.zeros(rows.shape[0], np.int64)
        for v, r, y in zip(vs, renders, y_imgs):
            w2c = geom.invert_pose(v.c2w)
            uv, z = geom.project(v.K, w2c[:3, :3], w2c[:3, 3], X, v.k1, v.k2)
            with np.errstate(invalid="ignore"):
                ok = (z > _Z_MIN) & np.isfinite(uv).all(axis=1)
                px = np.rint(np.where(ok[:, None], uv, 0.0)).astype(np.int64)
                ok &= (px[:, 0] >= 0) & (px[:, 0] < v.width) & (px[:, 1] >= 0) & (px[:, 1] < v.height)
            if not ok.any():
                continue
            core = _core_mask(r.layer_id, cfg.erode_px)
            depth_i = np.asarray(r.depth, np.float64)
            pxc = np.clip(px[:, 0], 0, v.width - 1)
            pyc = np.clip(px[:, 1], 0, v.height - 1)
            ok &= core[pyc, pxc]
            with np.errstate(invalid="ignore"):
                ok &= np.abs(depth_i[pyc, pxc] - z) <= cfg.depth_tol_rel * z
            if not ok.any():
                continue
            samp = _bilinear(y, uv[ok, 0], uv[ok, 1])
            acc[ok] += samp
            cnt[ok] += 1
        votes[rows, cols] = np.minimum(cnt, np.iinfo(np.int16).max).astype(np.int16)
        take = cnt >= cfg.min_votes
        if take.any():
            mean = acc[take] / cnt[take][:, None]
            new_rgb[rows[take], cols[take]] = np.clip(mean, 0.0, 1.0).astype(np.float32)
            filled[rows[take], cols[take]] = True
    unfilled = hidden & ~filled

    meta = _meta_dict(cfg, vs, disocc, holes, steps, sigma_snapped, int(hidden.sum()), int(filled.sum()),
                      int(unfilled.sum()))
    strata = _build_strata(dw, new_rgb, hidden, filled, meta)
    world = strata.to_world()
    return SDEditAvgResult(strata=strata, world=world, filled=filled, votes=votes, unfilled=unfilled,
                           views=vs, view_disocc_frac=disocc, view_hole_frac=holes, cfg=cfg, steps=steps,
                           sigma_snapped=sigma_snapped)


def _invent(dw: DepthWorld, sampler, cfg: SDEditAvgConfig, vs: tuple, renders: list, seed: int, steps: int,
            sn) -> list:
    """뷰 렌더 -> 교사 입력 -> lockstep SDEdit 1회 -> 선형 영상 (H,W,3) float64 목록."""
    import torch

    obs = dw.world.obs
    H, W = int(obs.height), int(obs.width)
    teacher = sampler.teacher
    H16, W16 = -(-H // 16) * 16, -(-W // 16) * 16
    teacher.spec.check_image_size(H16, W16)

    xs = []
    for r in renders:
        lin = _teacher_image_lin(r.rgb, r.alpha)
        x = 2.0 * color.linear_to_srgb(np.clip(lin, 0.0, 1.0)) - 1.0
        if (H16, W16) != (H, W):
            x = np.pad(x, ((0, H16 - H), (0, W16 - W), (0, 0)), mode="edge")
        xs.append(np.moveaxis(x, -1, 0))
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(xs, axis=0), dtype=np.float32))
    n = len(vs)

    with torch.no_grad():
        z0 = teacher.encode_image(batch)
        res = sampler.sdedit(z0, [cfg.prompt], sigma=cfg.sigma, seeds=seed, steps=steps)
        sigma_end = getattr(res, "sigma_end", None)
        z = getattr(res, "z", None)
        if sigma_end is None or z is None:
            raise SDEditAvgError("sampler.sdedit 결과에 z/sigma_end 가 없다")
        if not abs(float(sigma_end)) <= _SIGMA_END_TOL:
            raise SDEditAvgError(f"SDEdit 가 표 끝(sigma=0)까지 가지 않았다 (sigma_end={sigma_end!r})")
        if not isinstance(z, torch.Tensor) or tuple(z.shape) != tuple(z0.shape):
            got = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise SDEditAvgError(f"교사 잠재 모양 {got} != 입력 잠재 {tuple(z0.shape)}")
        dec = teacher.decode_latent(z)
    if not isinstance(dec, torch.Tensor) or tuple(dec.shape) != (n, 3, H16, W16):
        got = tuple(dec.shape) if isinstance(dec, torch.Tensor) else type(dec).__name__
        raise SDEditAvgError(f"교사 디코드 모양 {got} != {(n, 3, H16, W16)}")
    arr = dec.detach().to("cpu", torch.float32).numpy().astype(np.float64)
    if not np.isfinite(arr).all():
        raise SDEditAvgError("교사 디코드 출력이 유한하지 않다")
    disp = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)[:, :, :H, :W]
    lin = color.srgb_to_linear(np.moveaxis(disp, 1, -1))
    return [np.ascontiguousarray(lin[i], dtype=np.float64) for i in range(n)]


def _build_strata(dw: DepthWorld, new_rgb: np.ndarray, hidden: np.ndarray, filled: np.ndarray,
                  meta: dict) -> Strata:
    strata = copy.deepcopy(dw.strata)
    old1 = strata.layers[1]
    rgba = np.array(old1.rgba, dtype=np.float32, copy=True)
    rgba[..., :3] = np.where(filled[..., None], new_rgb, rgba[..., :3])
    strata.layers[1] = C.LDILayer(rgba=rgba, depth=np.array(old1.depth, dtype=np.float32, copy=True))

    cols = strata.tsq.columns
    eng = np.nonzero(cols["own"] == OWNERS["ENGINE"])[0]
    if eng.size:
        obs = dw.world.obs
        H, W = int(obs.height), int(obs.width)
        uv, _ = geom.project(obs.K, obs.R, obs.t, np.asarray(cols["mu"][eng], np.float64), float(obs.k1),
                             float(obs.k2))
        px = np.rint(uv).astype(np.int64)
        pc = np.clip(px[:, 0], 0, W - 1)
        pr = np.clip(px[:, 1], 0, H - 1)
        sel = hidden[pr, pc]
        color_col = np.array(cols["color"], copy=True)
        color_col[eng[sel]] = rgba[pr[sel], pc[sel], :3]
        cols["color"] = np.ascontiguousarray(color_col)

    strata.meta[NAME] = meta
    return strata


# --------------------------------------------------------------------------- S2 스테이지

def make_world_impl(sampler, *, depth_cfg: Optional[DepthWorldConfig] = None):
    if depth_cfg is not None and not isinstance(depth_cfg, DepthWorldConfig):
        raise TypeError(f"depth_cfg: DepthWorldConfig 또는 None 이어야 한다 ({type(depth_cfg).__name__})")
    from buc.pipeline import spine

    known = {f.name for f in dataclasses.fields(SDEditAvgConfig)}

    def fn(inputs: dict, ctx) -> dict:
        params = dict(ctx.params)
        seed = params.pop("_seed", ctx.seed)
        bad = sorted(set(params) - known)
        if bad:
            raise ValueError(f"sdedit_avg: 모르는 파라미터 {bad}")
        cfg = SDEditAvgConfig(**params)
        dw = build_depth_world(inputs["read"], cfg=depth_cfg)
        res = sdedit_avg_fill(dw, sampler, cfg=cfg, seed=seed)
        return {"world": res.world}

    return spine.Impl("S2", "real", NAME, VERSION, fn)
