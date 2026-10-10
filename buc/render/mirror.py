"""RDR-07 평면 재귀 반사 · secondary 광선 (Warp BVH 폴백).

계약서: ops/harness/contracts/RDR-07.md. 세계 오른손·미터, 카메라 OpenCV, 픽셀 중심이 정수 좌표.
출력 rgb 는 선형(표시값 변환은 render_mirror_shot 에서만).

구현 요약 (관측 가능한 결과만 계약):
  * 평면 거울 = 가상 카메라 재귀. 거울 i 를 보는 화소는 `R_i ⊙ fliplr(L(가상 카메라))` 로 칠한다.
    가상 카메라 c2w_v = M c2w F (M: 하우스홀더, F = diag(-1,1,1,1)), K_v 는 스큐·주점을 좌우 반전한다.
  * 가상 시점마다 월드를 거울 평면으로 잘라(평면 뒤 텍셀 알파 0) 거울 뒤 물체가 반사에 새지 않게 한다.
  * 거울 적중은 화소 중심 광선 vs 사각형(또는 무한 평면). 직전 거울과 clip 평면 뒤의 교점은 제외한다.
  * 순회는 깊이 우선, 거울 번호 오름차순. 자식 렌더가 필요한 화소가 있을 때만 렌더한다.
  * secondary 광선 `trace_closest`: numpy 전수(블록) Möller-Trumbore 기준 구현, 가능하면 Warp `wp.Mesh`.
    `warp` 는 지연 import 한다(모듈 import 시 sys.modules 에 올리지 않는다).
"""
from __future__ import annotations

import dataclasses
import math
import numbers
from dataclasses import dataclass

import numpy as np

from buc.core import color, geom
from buc.core import contracts as C
from buc.render import quads as _Q
from buc.render.quads import RenderOpts, render_ldi

__all__ = [
    "Mirror", "MirrorOpts", "MirrorRender", "reflection_matrix", "virtual_camera", "ray_mirror_hits",
    "clip_world", "render_mirrors", "render_mirror_shot", "BackendUnavailable", "RayHits",
    "available_backends", "trace_closest", "reflect_dirs", "fresnel_schlick",
]

_T_EPS = 1e-9          # 거울 광선 t 의 하한(자기 교차 방지)
_ALPHA_MIRROR = 0.5    # 1차 알파가 이보다 작으면 거울이 보인다


# ---------------------------------------------------------------------------------------------------------------
# 검증 도우미
# ---------------------------------------------------------------------------------------------------------------
def _vec3(name, v):
    try:
        a = np.array(v, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: 길이 3 의 숫자 수열이어야 한다 ({v!r})") from e
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: 유한한 길이 3 이어야 한다 (shape {a.shape})")
    return a


def _unit(name, v):
    a = _vec3(name, v)
    n = float(np.linalg.norm(a))
    if not n > 0.0:
        raise ValueError(f"{name}: 0벡터일 수 없다")
    return a / n, n


def _finite_num(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name}: 실수여야 한다 ({v!r})")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name}: 유한해야 한다 ({v!r})")
    return f


def _ro(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


# ---------------------------------------------------------------------------------------------------------------
# 공개 자료형
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True, eq=False)
class Mirror:
    center: object
    normal: object
    u_axis: object = None
    half_size: object = None
    reflectance: tuple = (0.9, 0.9, 0.9)
    two_sided: bool = False

    def __post_init__(self):
        c = _vec3("center", self.center)
        n, _ = _unit("normal", self.normal)
        u = None
        if self.u_axis is not None:
            u0 = _vec3("u_axis", self.u_axis)
            n0 = float(np.linalg.norm(u0))
            up = u0 - float(u0 @ n) * n
            if not (n0 > 0.0 and float(np.linalg.norm(up)) > 1e-9 * n0):
                raise ValueError("u_axis: 0벡터이거나 법선과 평행하다")
            u = up / np.linalg.norm(up)
        hs = None
        if self.half_size is not None:
            try:
                hs = tuple(float(x) for x in self.half_size)
            except (TypeError, ValueError) as e:
                raise ValueError(f"half_size: 길이 2 의 양수 수열이어야 한다 ({self.half_size!r})") from e
            if len(hs) != 2 or not all(math.isfinite(x) and x > 0.0 for x in hs):
                raise ValueError(f"half_size: 길이 2 의 유한한 양수여야 한다 ({self.half_size!r})")
            if u is None:
                raise ValueError("half_size 가 있으면 u_axis 가 필요하다")
        try:
            r = tuple(float(x) for x in self.reflectance)
        except (TypeError, ValueError) as e:
            raise ValueError(f"reflectance: 길이 3 이어야 한다 ({self.reflectance!r})") from e
        if len(r) != 3 or not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in r):
            raise ValueError(f"reflectance: 각 성분 [0,1] 의 길이 3 이어야 한다 ({self.reflectance!r})")
        object.__setattr__(self, "center", _ro(c))
        object.__setattr__(self, "normal", _ro(n))
        object.__setattr__(self, "u_axis", None if u is None else _ro(u))
        object.__setattr__(self, "half_size", hs)
        object.__setattr__(self, "reflectance", r)
        object.__setattr__(self, "two_sided", bool(self.two_sided))

    @property
    def d(self) -> float:
        """평면 오프셋: normal·X = d."""
        return float(np.asarray(self.normal) @ np.asarray(self.center))

    @property
    def v_axis(self):
        """v = normal x u (u_axis 가 없으면 None)."""
        if self.u_axis is None:
            return None
        return np.cross(np.asarray(self.normal), np.asarray(self.u_axis))


@dataclass(frozen=True, kw_only=True)
class MirrorOpts:
    max_depth: int = 2
    max_renders: int = 64
    min_weight: float = 1e-3
    terminal_rgb: tuple = (0, 0, 0)
    clip_eps_m: float = 1e-4
    quad: RenderOpts | None = None

    def __post_init__(self):
        md, mr = self.max_depth, self.max_renders
        if isinstance(md, bool) or not isinstance(md, numbers.Integral) or not 0 <= int(md) <= 16:
            raise ValueError(f"max_depth: 0..16 정수여야 한다 ({md!r})")
        if isinstance(mr, bool) or not isinstance(mr, numbers.Integral) or int(mr) < 1:
            raise ValueError(f"max_renders >= 1 정수여야 한다 ({mr!r})")
        mw = _finite_num("min_weight", self.min_weight)
        if not 0.0 <= mw < 1.0:
            raise ValueError("min_weight: [0,1) 이어야 한다")
        ce = _finite_num("clip_eps_m", self.clip_eps_m)
        if ce < 0.0:
            raise ValueError("clip_eps_m >= 0 이어야 한다")
        try:
            tr = tuple(float(x) for x in self.terminal_rgb)
        except (TypeError, ValueError) as e:
            raise ValueError(f"terminal_rgb: 길이 3 의 유한한 수열이어야 한다 ({self.terminal_rgb!r})") from e
        if len(tr) != 3 or not all(math.isfinite(x) for x in tr):
            raise ValueError(f"terminal_rgb: 길이 3 의 유한한 수열이어야 한다 ({self.terminal_rgb!r})")
        if self.quad is not None and not isinstance(self.quad, RenderOpts):
            raise ValueError("quad: RenderOpts 또는 None 이어야 한다")
        object.__setattr__(self, "max_depth", int(md))
        object.__setattr__(self, "max_renders", int(mr))
        object.__setattr__(self, "min_weight", mw)
        object.__setattr__(self, "clip_eps_m", ce)
        object.__setattr__(self, "terminal_rgb", tr)


@dataclass(frozen=True, kw_only=True)
class MirrorRender:
    rgb: np.ndarray            # (H,W,3) float32 선형
    alpha: np.ndarray          # (H,W) float32
    depth: np.ndarray          # (H,W) float32 1차 z-깊이
    layer_id: np.ndarray       # (H,W) int16
    mirror_id: np.ndarray      # (H,W) int16, 첫 거울(없으면 -1)
    bounce: np.ndarray         # (H,W) int8, 경로가 맞은 거울 수
    virtual_depth: np.ndarray  # (H,W) float32, 허상의 1차 카메라 z-깊이(없으면 +inf)
    n_renders: int


def _check_mirrors(mirrors):
    if isinstance(mirrors, (str, bytes)) or isinstance(mirrors, Mirror):
        raise TypeError("mirrors: Mirror 의 수열이어야 한다")
    try:
        ms = list(mirrors)
    except TypeError as e:
        raise TypeError("mirrors: Mirror 의 수열이어야 한다") from e
    for i, m in enumerate(ms):
        if not isinstance(m, Mirror):
            raise TypeError(f"mirrors[{i}]: Mirror 여야 한다 ({type(m).__name__})")
    return ms


# ---------------------------------------------------------------------------------------------------------------
# 기하
# ---------------------------------------------------------------------------------------------------------------
def reflection_matrix(normal, d) -> np.ndarray:
    """(4,4) 하우스홀더: X' = X - 2 (n·X - d) n  (n 은 단위화, d 는 그대로)."""
    n, _ = _unit("normal", normal)
    d = _finite_num("d", d)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] -= 2.0 * np.outer(n, n)
    M[:3, 3] = 2.0 * d * n
    return M


_FLIP = np.diag([-1.0, 1.0, 1.0, 1.0])


def _virtual(T, Kt, W, normal, d):
    M = reflection_matrix(normal, d)
    Tv = M @ T @ _FLIP
    Kv = Kt.copy()
    Kv[0, 1] = -Kt[0, 1]
    Kv[0, 2] = (W - 1) - Kt[0, 2]
    return Tv, Kv


def virtual_camera(c2w, K, width, mirror) -> tuple[np.ndarray, np.ndarray]:
    """거울 `mirror` 의 가상 카메라 (c2w_v, K_v). 가상 영상을 좌우로 뒤집으면 원 카메라에서 본 거울상."""
    T = _Q._check_pose(c2w)
    Kt = _Q._check_K(K)
    W = _Q._check_size("width", width)
    if not isinstance(mirror, Mirror):
        raise TypeError("mirror: Mirror 여야 한다")
    return _virtual(T, Kt, W, mirror.normal, mirror.d)


def _pixel_rays(T, Kt, W, H, k1, k2):
    """화소 중심 광선: (eye (3,), dirs_world (H*W,3), z=1 정규화 카메라 방향이므로 t 가 곧 z-깊이)."""
    uv = geom.pixel_grid(H, W).reshape(-1, 2)
    yd = (uv[:, 1] - Kt[1, 2]) / Kt[1, 1]
    xd = (uv[:, 0] - Kt[0, 2] - Kt[0, 1] * yd) / Kt[0, 0]
    xy = np.stack([xd, yd], -1)
    if k1 != 0.0 or k2 != 0.0:
        xy = geom.undistort(xy, k1, k2)
    dc = np.concatenate([xy, np.ones((xy.shape[0], 1))], -1)
    dw = dc @ T[:3, :3].T
    return T[:3, 3].copy(), dw


def _mirror_hits(mirrors, T, Kt, W, H, k1, k2, near_m, exclude=None, clip=None):
    """가장 가까운 거울 적중. clip=(n, d, side, eps) 면 clip 평면 뒤의 교점은 제외한다."""
    N = H * W
    best_t = np.full(N, np.inf)
    best_i = np.full(N, -1, np.int16)
    if not mirrors:
        return best_i.reshape(H, W), best_t.reshape(H, W)
    eye, dw = _pixel_rays(T, Kt, W, H, k1, k2)
    for i, m in enumerate(mirrors):
        if exclude is not None and i == exclude:
            continue
        n = np.asarray(m.normal)
        dn = dw @ n
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            t = (m.d - float(eye @ n)) / dn
            ok = np.isfinite(t) & (t > _T_EPS) & (t >= near_m)
            ok &= (dn != 0.0) if m.two_sided else (dn < 0.0)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        P = eye + t[idx, None] * dw[idx]
        keep = np.ones(idx.size, bool)
        if m.half_size is not None:
            rel = P - np.asarray(m.center)
            u = np.asarray(m.u_axis)
            v = np.cross(n, u)
            keep &= (np.abs(rel @ u) <= m.half_size[0]) & (np.abs(rel @ v) <= m.half_size[1])
        if clip is not None:
            cn, cd, cs, ce = clip
            keep &= cs * (P @ cn - cd) > ce
        idx = idx[keep]
        if idx.size == 0:
            continue
        ti = t[idx]
        upd = ti < best_t[idx]
        sel = idx[upd]
        best_t[sel] = ti[upd]
        best_i[sel] = i
    return best_i.reshape(H, W), best_t.reshape(H, W)


def ray_mirror_hits(mirrors, *, c2w, K, width, height, k1=0.0, k2=0.0, near_m=1e-3):
    """화소 중심 광선의 가장 가까운 거울 적중: (mirror_id int16 (H,W), z float64 (H,W) 미스 +inf)."""
    ms = _check_mirrors(mirrors)
    T = _Q._check_pose(c2w)
    Kt = _Q._check_K(K)
    W = _Q._check_size("width", width)
    H = _Q._check_size("height", height)
    k1 = _finite_num("k1", k1)
    k2 = _finite_num("k2", k2)
    near_m = _finite_num("near_m", near_m)
    return _mirror_hits(ms, T, Kt, W, H, k1, k2, near_m)


def _texel_points(world):
    """층마다 텍셀을 월드 좌표로 역투영한 (h*w, 3) float64."""
    obs = world.obs
    out = []
    for ly in world.layers:
        h, w = ly.rgba.shape[:2]
        uv = geom.pixel_grid(h, w).reshape(-1, 2)
        X = geom.unproject_world(obs.K, obs.R, obs.t, uv, ly.depth.reshape(-1).astype(np.float64),
                                 obs.k1, obs.k2)
        out.append(np.asarray(X, np.float64).reshape(-1, 3))
    return out


def _clip_unit(world, pts, n, d, side, eps):
    layers = []
    for ly, X in zip(world.layers, pts):
        h, w = ly.rgba.shape[:2]
        s = side * (X @ n - d)
        rgba = ly.rgba.copy()
        a = rgba[..., 3].reshape(-1)
        a[s <= eps] = 0.0
        rgba[..., 3] = a.reshape(h, w)
        layers.append(C.LDILayer(rgba=rgba, depth=ly.depth.copy()))
    return dataclasses.replace(world, layers=tuple(layers))


def clip_world(world, normal, d, *, side=1, eps=1e-4):
    """평면 (n·X = d) 의 side 쪽 반대편(= side*(n·X-d) <= eps) 텍셀의 알파를 0 으로. 새 World."""
    n, nrm = _unit("normal", normal)
    d = _finite_num("d", d) / nrm
    if isinstance(side, bool) or side not in (1, -1):
        raise ValueError(f"side: 1 또는 -1 이어야 한다 ({side!r})")
    eps = _finite_num("eps", eps)
    return _clip_unit(world, _texel_points(world), n, d, int(side), eps)


# ---------------------------------------------------------------------------------------------------------------
# 재귀 렌더
# ---------------------------------------------------------------------------------------------------------------
class _Ctx:
    __slots__ = ("world", "mirrors", "W", "H", "k1", "k2", "opts", "quad", "n", "pts", "refl", "term")


def _level(ctx, T, Kt, clip, last, depth_left, w):
    """한 시점 렌더 + 거울 화소 채움. 반환: dict(rgb, alpha, depth, layer_id, mirror_id, bounce, vdepth)."""
    opts = ctx.opts
    if clip is None:
        wc = ctx.world
    else:
        if ctx.pts is None:
            ctx.pts = _texel_points(ctx.world)
        wc = _clip_unit(ctx.world, ctx.pts, clip[0], clip[1], clip[2], clip[3])
    prim = render_ldi(wc, c2w=T, K=Kt, width=ctx.W, height=ctx.H, k1=ctx.k1, k2=ctx.k2, opts=ctx.quad)
    ctx.n += 1
    H, W = ctx.H, ctx.W
    ids, z = _mirror_hits(ctx.mirrors, T, Kt, W, H, ctx.k1, ctx.k2, 1e-3, exclude=last, clip=clip)
    pz = prim.depth.astype(np.float64)
    with np.errstate(invalid="ignore"):
        mpx = (ids >= 0) & ((prim.alpha < _ALPHA_MIRROR) | (pz > z))
    rgb = prim.rgb.copy()
    alpha = prim.alpha.copy()
    depth = prim.depth.copy()
    layer_id = prim.layer_id.copy()
    mirror_id = np.full((H, W), -1, np.int16)
    bounce = np.zeros((H, W), np.int8)
    vdepth = prim.depth.copy()
    if mpx.any():
        for i in np.unique(ids[mpx]):
            i = int(i)
            sel = mpx & (ids == i)
            m = ctx.mirrors[i]
            R = ctx.refl[i]
            wn = w * max(m.reflectance)
            alpha[sel] = 1.0
            depth[sel] = z[sel].astype(np.float32)
            layer_id[sel] = -1
            mirror_id[sel] = i
            if depth_left > 0 and wn >= opts.min_weight and ctx.n < opts.max_renders:
                n_i = np.asarray(m.normal)
                side = 1 if float(n_i @ T[:3, 3]) - m.d > 0.0 else -1
                Tv, Kv = _virtual(T, Kt, W, n_i, m.d)
                ch = _level(ctx, Tv, Kv, (n_i, m.d, side, opts.clip_eps_m), i, depth_left - 1, wn)
                crgb = ch["rgb"][:, ::-1]
                rgb[sel] = R * crgb[sel]
                bounce[sel] = (1 + ch["bounce"][:, ::-1][sel].astype(np.int16)).astype(np.int8)
                vdepth[sel] = ch["vdepth"][:, ::-1][sel]
            else:
                rgb[sel] = R * ctx.term
                bounce[sel] = 1
                vdepth[sel] = np.inf
    return dict(rgb=rgb, alpha=alpha, depth=depth, layer_id=layer_id, mirror_id=mirror_id, bounce=bounce,
                vdepth=vdepth)


def render_mirrors(world, mirrors, *, c2w, K, width, height, k1=0.0, k2=0.0, opts=None) -> MirrorRender:
    ms = _check_mirrors(mirrors)
    if opts is None:
        opts = MirrorOpts()
    elif not isinstance(opts, MirrorOpts):
        raise TypeError("opts: MirrorOpts 여야 한다")
    T = _Q._check_pose(c2w)
    Kt = _Q._check_K(K)
    W = _Q._check_size("width", width)
    H = _Q._check_size("height", height)
    k1 = _finite_num("k1", k1)
    k2 = _finite_num("k2", k2)
    ctx = _Ctx()
    ctx.world, ctx.mirrors, ctx.W, ctx.H, ctx.k1, ctx.k2 = world, ms, W, H, k1, k2
    ctx.opts, ctx.quad, ctx.n, ctx.pts = opts, opts.quad, 0, None
    ctx.refl = [np.asarray(m.reflectance, np.float32) for m in ms]
    ctx.term = np.asarray(opts.terminal_rgb, np.float32)
    out = _level(ctx, T, Kt, None, None, opts.max_depth, 1.0)
    return MirrorRender(
        rgb=np.ascontiguousarray(out["rgb"], np.float32),
        alpha=np.ascontiguousarray(out["alpha"], np.float32),
        depth=np.ascontiguousarray(out["depth"], np.float32),
        layer_id=np.ascontiguousarray(out["layer_id"], np.int16),
        mirror_id=np.ascontiguousarray(out["mirror_id"], np.int16),
        bounce=np.ascontiguousarray(out["bounce"], np.int8),
        virtual_depth=np.ascontiguousarray(out["vdepth"], np.float32),
        n_renders=int(ctx.n),
    )


def render_mirror_shot(world, mirrors, shot, opts=None) -> C.Frames:
    ms = _check_mirrors(mirrors)
    n = shot.n_frames
    outs = [render_mirrors(world, ms, c2w=shot.c2w[i], K=shot.K[i], width=shot.width, height=shot.height,
                           opts=opts) for i in range(n)]
    rgb = np.stack([color.linear_to_display(o.rgb) for o in outs], 0).astype(np.float32)
    aov = {
        "depth": np.stack([o.depth for o in outs], 0).astype(np.float32),
        "alpha": np.stack([o.alpha for o in outs], 0).astype(np.float32),
        "layer_id": np.stack([o.layer_id for o in outs], 0).astype(np.int16),
        "mirror_id": np.stack([o.mirror_id for o in outs], 0).astype(np.int16),
        "bounce": np.stack([o.bounce for o in outs], 0).astype(np.int8),
        "virtual_depth": np.stack([o.virtual_depth for o in outs], 0).astype(np.float32),
    }
    return C.Frames(rgb=rgb, fps=shot.fps, aov=aov, shot_hash=shot.content_hash(),
                    world_hash=world.content_hash())


# ---------------------------------------------------------------------------------------------------------------
# secondary 광선
# ---------------------------------------------------------------------------------------------------------------
class BackendUnavailable(RuntimeError):
    """요청한 광선 추적 백엔드를 쓸 수 없다."""


@dataclass(frozen=True, kw_only=True)
class RayHits:
    t: np.ndarray      # (N,) float64, 미스 +inf
    tri: np.ndarray    # (N,) int64, 미스 -1
    bary: np.ndarray   # (N,2) float64 (b1,b2), 미스 0


_WARP = {"checked": False, "ok": False}


def _warp_module():
    """warp 를 지연 import 하고 CUDA 장치가 있으면 모듈을 돌려준다. 아니면 None."""
    if not _WARP["checked"]:
        _WARP["checked"] = True
        try:
            import warp as wp  # noqa: PLC0415 (지연 import)
            wp.config.quiet = True
            wp.init()
            _WARP["ok"] = bool(wp.get_cuda_device_count() > 0)
        except Exception:  # noqa: BLE001 (warp/CUDA 없음 -> 폴백)
            _WARP["ok"] = False
    if not _WARP["ok"]:
        return None
    import warp as wp  # noqa: PLC0415
    return wp


def available_backends() -> tuple[str, ...]:
    out = ["numpy"]
    if _warp_module() is not None:
        out.append("warp")
    return tuple(out)


def _check_rays(origins, dirs, tris, t_min, t_max):
    try:
        o = np.array(origins, dtype=np.float64)
        d = np.array(dirs, dtype=np.float64)
        tr = np.array(tris, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("origins/dirs/tris: 숫자 배열이어야 한다") from e
    if o.ndim != 2 or o.shape[1] != 3:
        raise ValueError(f"origins: (N,3) 이어야 한다 (shape {o.shape})")
    if d.shape != o.shape:
        raise ValueError(f"dirs shape {d.shape} != origins shape {o.shape}")
    if tr.ndim != 3 or tr.shape[1:] != (3, 3):
        raise ValueError(f"tris: (M,3,3) 이어야 한다 (shape {tr.shape})")
    if not (np.isfinite(o).all() and np.isfinite(d).all() and np.isfinite(tr).all()):
        raise ValueError("origins/dirs/tris: 유한해야 한다")
    t_min = _finite_num("t_min", t_min)
    t_max = float(t_max)
    if math.isnan(t_max):
        raise ValueError("t_max: NaN 일 수 없다")
    nrm = np.linalg.norm(d, axis=1)
    if not (nrm > 0.0).all():
        raise ValueError("dirs: 0벡터가 있다")
    return o, d / nrm[:, None], tr, t_min, t_max


def _trace_numpy(o, d, tris, t_min, t_max):
    """블록 단위 전수 Möller-Trumbore(양면). 동률은 작은 삼각형 번호. 결정론."""
    N, M = o.shape[0], tris.shape[0]
    best_t = np.full(N, np.inf)
    best_i = np.full(N, -1, np.int64)
    best_b = np.zeros((N, 2))
    if N == 0 or M == 0:
        return best_t, best_i, best_b
    ox, oy, oz = (o[:, k][None, :] for k in range(3))
    dx, dy, dz = (d[:, k][None, :] for k in range(3))
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    blk = max(1, int(300_000 // N))
    ar = np.arange(N)
    for s0 in range(0, M, blk):
        sl = slice(s0, min(M, s0 + blk))
        a0 = [v0[sl, k][:, None] for k in range(3)]
        a1 = [e1[sl, k][:, None] for k in range(3)]
        a2 = [e2[sl, k][:, None] for k in range(3)]
        with np.errstate(all="ignore"):
            px = dy * a2[2] - dz * a2[1]
            py = dz * a2[0] - dx * a2[2]
            pz = dx * a2[1] - dy * a2[0]
            det = px * a1[0] + py * a1[1] + pz * a1[2]
            inv = 1.0 / det
            sx, sy, sz = ox - a0[0], oy - a0[1], oz - a0[2]
            b1 = (sx * px + sy * py + sz * pz) * inv
            qx = sy * a1[2] - sz * a1[1]
            qy = sz * a1[0] - sx * a1[2]
            qz = sx * a1[1] - sy * a1[0]
            b2 = (dx * qx + dy * qy + dz * qz) * inv
            t = (qx * a2[0] + qy * a2[1] + qz * a2[2]) * inv
            ok = ((np.abs(det) > 1e-12) & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
                  & (t > t_min) & (t <= t_max))
        tm = np.where(ok, t, np.inf)
        j = np.argmin(tm, axis=0)                   # 동률은 블록 안 가장 작은 번호
        tb = tm[j, ar]
        upd = tb < best_t                           # 엄격: 앞 블록(작은 번호)이 동률을 이긴다
        if upd.any():
            best_t[upd] = tb[upd]
            best_i[upd] = s0 + j[upd]
            best_b[upd, 0] = b1[j, ar][upd]
            best_b[upd, 1] = b2[j, ar][upd]
    return best_t, best_i, best_b


_WARP_KERNEL = {}


def _warp_kernel(wp):
    if "k" not in _WARP_KERNEL:
        @wp.kernel
        def _closest(mesh: wp.uint64, org: wp.array(dtype=wp.vec3), dr: wp.array(dtype=wp.vec3),
                     t_max: float, out_t: wp.array(dtype=float), out_f: wp.array(dtype=int)):
            i = wp.tid()
            q = wp.mesh_query_ray(mesh, org[i], dr[i], t_max)
            if q.result:
                out_t[i] = q.t
                out_f[i] = q.face
            else:
                out_t[i] = -1.0
                out_f[i] = -1
        _WARP_KERNEL["k"] = _closest
    return _WARP_KERNEL["k"]


def _trace_warp(wp, o, d, tris, t_min, t_max):
    """Warp `wp.Mesh` BVH 폴백(GPU). t_min 은 원점을 t_min 만큼 전진시켜 근사한다.

    무게중심은 Warp 가 주는 (u, v) 규약에 기대지 않고, 찾은 (광선, 삼각형) 쌍에 대해 호스트에서 다시 푼다."""
    N, M = o.shape[0], tris.shape[0]
    best_t = np.full(N, np.inf)
    best_i = np.full(N, -1, np.int64)
    best_b = np.zeros((N, 2))
    if N == 0 or M == 0:
        return best_t, best_i, best_b
    dev = "cuda:0"
    pts = tris.reshape(-1, 3).astype(np.float32)
    idx = np.arange(3 * M, dtype=np.int32)
    mesh = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=dev), indices=wp.array(idx, dtype=int, device=dev))
    o2 = (o + t_min * d).astype(np.float32)
    tmax_eff = float(min(t_max - t_min, 1e30)) if np.isfinite(t_max) else 1e30
    org = wp.array(o2, dtype=wp.vec3, device=dev)
    dr = wp.array(d.astype(np.float32), dtype=wp.vec3, device=dev)
    ot = wp.zeros(N, dtype=float, device=dev)
    of = wp.zeros(N, dtype=int, device=dev)
    wp.launch(_warp_kernel(wp), dim=N, inputs=[mesh.id, org, dr, tmax_eff, ot, of], device=dev)
    wp.synchronize_device(dev)
    t = ot.numpy().astype(np.float64)
    f = of.numpy().astype(np.int64)
    hit = f >= 0
    if hit.any():
        h = np.flatnonzero(hit)
        v = tris[f[h]]
        e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        p = np.cross(d[h], e2)
        det = (p * e1).sum(1)
        with np.errstate(all="ignore"):
            inv = 1.0 / det
            s = o[h] - v[:, 0]
            b1 = (s * p).sum(1) * inv
            b2 = (d[h] * np.cross(s, e1)).sum(1) * inv
        good = np.isfinite(b1) & np.isfinite(b2)
        hh = h[good]
        best_t[hh] = t[hh] + t_min
        best_i[hh] = f[hh]
        best_b[hh, 0] = b1[good]
        best_b[hh, 1] = b2[good]
    return best_t, best_i, best_b


def trace_closest(origins, dirs, tris, *, t_min=1e-6, t_max=math.inf, backend="auto") -> RayHits:
    """광선별 가장 가까운 삼각형 적중(양면). dirs 는 내부 단위화, t 는 미터."""
    if not isinstance(backend, str) or backend not in ("auto", "numpy", "warp", "optix"):
        raise ValueError(f"backend: 'auto'|'numpy'|'warp' 이어야 한다 ({backend!r})")
    o, d, tr, t_min, t_max = _check_rays(origins, dirs, tris, t_min, t_max)
    if backend == "optix":
        raise BackendUnavailable("optix 백엔드는 구현되지 않았다(이름만 예약)")
    wp = None
    if backend in ("auto", "warp"):
        wp = _warp_module()
        if wp is None and backend == "warp":
            raise BackendUnavailable("warp/CUDA 를 쓸 수 없다")
    t = i = b = None
    if wp is not None:
        try:
            t, i, b = _trace_warp(wp, o, d, tr, t_min, t_max)
        except Exception as e:  # noqa: BLE001
            if backend == "warp":
                raise BackendUnavailable(f"warp 추적 실패: {e}") from e
            t = None                                  # auto: numpy 로 폴백
    if t is None:
        t, i, b = _trace_numpy(o, d, tr, t_min, t_max)
    return RayHits(t=t, tri=i, bary=b)


def reflect_dirs(d, n) -> np.ndarray:
    """d - 2 (d·n̂) n̂ (n 은 단위화, |d| 보존)."""
    d = np.asarray(d, np.float64)
    n = np.asarray(n, np.float64)
    nrm = np.linalg.norm(n, axis=-1, keepdims=True)
    if not (nrm > 0.0).all() or not np.isfinite(nrm).all():
        raise ValueError("n: 0벡터이거나 비유한이다")
    nu = n / nrm
    return d - 2.0 * (d * nu).sum(-1, keepdims=True) * nu


def fresnel_schlick(cos_theta, f0) -> np.ndarray:
    """f0 + (1 - f0) (1 - clip(cos, 0, 1))^5 (브로드캐스트)."""
    c = np.clip(np.asarray(cos_theta, np.float64), 0.0, 1.0)
    f0 = np.asarray(f0, np.float64)
    out = f0 + (1.0 - f0) * (1.0 - c) ** 5
    return out[()] if isinstance(out, np.ndarray) and out.ndim == 0 else out
