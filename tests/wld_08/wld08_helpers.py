"""WLD-08 테스트 공용: 합성 장면(전경 상자 2 m / 텍스처 배경 6 m), 정답 세계(GT), 가짜 교사·샘플러.

GT 오라클은 RDR-02 ``render_ldi`` 와 FND-06/WLD-03 의 동결 모듈만 쓰고 WLD-08 구현은 참조하지 않는다.
가짜 교사의 코덱은 "손잡이(handle)" 방식이다: encode 는 영상을 보관하고 잠재의 [b,0,0,0] 칸에 번호를 적고,
decode 는 그 번호로 영상을 돌려준다. 그래서 SDEdit 이후 영상을 테스트가 정확히 통제한다(코덱 충실도는 GPU/V 몫)."""
from types import SimpleNamespace

import numpy as np
import torch

from buc.core import color
from buc.core import contracts as C
from buc.core import geom
from buc.render.quads import render_ldi
from buc.teacher.sample import DEFAULT_TABLE
from buc.teacher.zimage import TeacherSpec
from buc.world.depth_world import DepthWorldConfig, build_depth_world

USER_PROV = C.Prov(taint=frozenset({"USER"}), mode="user", ns="user_vault")

W, H = 96, 64
FX = 100.0
Y0, Y1, X0, X1 = 20, 44, 40, 64          # 전경 사각형(반열린)
Z_FG, Z_BG = 2.0, 6.0
BOX = np.array([0.9, 0.1, 0.1])


def K_of(w=W, h=H, f=FX):
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])


def obs_of(w=W, h=H, f=FX):
    return C.Obs(K=K_of(w, h, f), R=np.eye(3), t=np.zeros(3), width=w, height=h)


def bg_tex(w=W, h=H):
    """저주파 배경 텍스처(파장 >= 44 px). 선형 RGB, 값 [0.15, 0.85]. 최근접 복사로는 틀리고 연속 외삽으로는 맞는다."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.empty((h, w, 3))
    img[..., 0] = 0.5 + 0.30 * np.sin(2 * np.pi * u / 52.0)
    img[..., 1] = 0.5 + 0.25 * np.cos(2 * np.pi * (u / 44.0 + v / 60.0))
    img[..., 2] = 0.2 + 0.6 * (u / w) * (0.6 + 0.4 * np.sin(2 * np.pi * v / 48.0))
    return np.clip(img, 0.0, 1.0)


def box_mask(w=W, h=H, y0=Y0, y1=Y1, x0=X0, x1=X1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def scene(w=W, h=H):
    """(image_lin f32, depth f32): 텍스처 배경(6 m) 위의 빨간 상자(2 m)."""
    img = bg_tex(w, h)
    m = box_mask(w, h)
    img[m] = BOX
    depth = np.where(m, Z_FG, Z_BG)
    return img.astype(np.float32), depth.astype(np.float32)


def make_read(image, depth, obs=None):
    image = np.asarray(image, np.float32)
    depth = np.asarray(depth, np.float32)
    h, w = depth.shape
    return C.ReadPacket(image_lin=image, depth=depth, obs=obs or obs_of(w, h), theta_in=C.Theta(aspect=w / h),
                        input_hash="b" * 64, prov=USER_PROV, level="L1")


def build(w=W, h=H, cfg=None):
    img, depth = scene(w, h)
    read = make_read(img, depth)
    return read, build_depth_world(read, cfg=cfg or DepthWorldConfig())


def build_flat(w=W, h=H):
    """깊이 불연속이 없는 장면(숨은 층 비어 있음)."""
    read = make_read(bg_tex(w, h).astype(np.float32), np.full((h, w), Z_BG, np.float32))
    return read, build_depth_world(read)


def gt_world(obs):
    """정답 세계: 층 0 = 상자(알파 하드, 2 m), 층 1 = 상자 뒤까지 이어진 배경(6 m, 알파 1)."""
    h, w = obs.height, obs.width
    rgba0 = np.zeros((h, w, 4), np.float32)
    m = box_mask(w, h)
    rgba0[m, :3] = BOX
    rgba0[m, 3] = 1.0
    rgba1 = np.concatenate([bg_tex(w, h), np.ones((h, w, 1))], axis=-1).astype(np.float32)
    layers = (C.LDILayer(rgba=rgba0, depth=np.full((h, w), Z_FG, np.float32)),
              C.LDILayer(rgba=rgba1, depth=np.full((h, w), Z_BG, np.float32)))
    return C.World(obs=obs, layers=layers, theta_in=C.Theta(aspect=w / h), level="L1", read_hash="g" * 8,
                   prov=USER_PROV)


def gt_views_display(obs, views, box_rgb=None):
    """정답 세계를 각 뷰에서 찍은 표시 sRGB [0,1] 영상 목록. box_rgb 로 상자 색만 바꿀 수 있다(누출 검사)."""
    gw = gt_world(obs)
    if box_rgb is not None:
        rgba = gw.layers[0].rgba.copy()
        rgba[..., :3] = np.where(rgba[..., 3:4] > 0, np.asarray(box_rgb, np.float32), 0.0)
        gw = C.World(obs=obs, layers=(C.LDILayer(rgba=rgba, depth=gw.layers[0].depth), gw.layers[1]),
                     theta_in=gw.theta_in, level="L1", read_hash="g" * 8, prov=USER_PROV)
    out = []
    for v in views:
        r = render_ldi(gw, c2w=v.c2w, K=v.K, width=v.width, height=v.height, k1=v.k1, k2=v.k2)
        out.append(color.linear_to_srgb(np.clip(r.rgb.astype(np.float64), 0.0, 1.0)))
    return out


# --------------------------------------------------------------------------- 가짜 교사 / 샘플러

class HandleTeacher:
    """encode_image/decode_latent 규약(TCH-01 ZImageTeacher 와 같은 시그니처)을 손잡이 코덱으로 흉내 낸다."""

    def __init__(self):
        self.spec = TeacherSpec()
        self.store = {}
        self.next_id = 1
        self.encoded = []        # encode_image 에 들어온 영상 [B,3,H,W] (기록)

    def encode_image(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3
        assert torch.isfinite(x).all(), "교사 입력에 NaN/Inf"
        self.spec.check_image_size(x.shape[2], x.shape[3])
        self.encoded.append(x.detach().clone())
        b, _, h, w = x.shape
        z = torch.zeros(b, self.spec.latent_channels, h // 8, w // 8, dtype=torch.float32)
        for i in range(b):
            self.store[self.next_id] = x[i].detach().clone().float()
            z[i, 0, 0, 0] = float(self.next_id)
            self.next_id += 1
        return z

    def decode_latent(self, z, clamp=True):
        assert isinstance(z, torch.Tensor) and z.ndim == 4 and z.shape[1] == self.spec.latent_channels
        ys = [self.store[int(round(float(z[i, 0, 0, 0])))] for i in range(z.shape[0])]
        y = torch.stack(ys)
        return y.clamp(-1.0, 1.0) if clamp else y


class FakeSampler:
    """TeacherSampler.sdedit 규약을 지키는 가짜. transform(images[B,3,H,W] in [-1,1]) -> 같은 모양 영상.

    규약(WLD-08 계약서): ``sampler.teacher``, ``sampler.table``, ``sampler.sdedit(x0, cond, *, sigma, seeds, steps,
    level, k)`` -> 객체(.z 잠재 [B,C,h,w], .sigma_end)."""

    def __init__(self, transform, teacher=None, table=DEFAULT_TABLE):
        self.teacher = teacher or HandleTeacher()
        self.table = table
        self.transform = transform
        self.calls = []

    def sdedit(self, x0, cond, *, sigma, seeds, steps=2, level=0, k=0, **kw):
        imgs = self.teacher.decode_latent(x0)
        self.calls.append(SimpleNamespace(x0_shape=tuple(x0.shape), images=imgs.clone(), cond=cond, sigma=sigma,
                                          seeds=seeds, steps=steps, level=level, k=k, kw=kw))
        out = self.transform(imgs)
        assert out.shape == imgs.shape
        z = self.teacher.encode_image(out)
        return SimpleNamespace(z=z, x0_hat=z, z_start=x0, sigma_end=0.0, steps=steps)


def const_transform(values):
    """뷰 i 의 출력을 표시값 values[i] (스칼라, 회색) 상수 영상으로."""
    def f(imgs):
        out = torch.zeros_like(imgs)
        for i in range(imgs.shape[0]):
            out[i] = 2.0 * float(values[i]) - 1.0
        return out
    return f


def oracle_transform(gt_display_list, pad_to=None):
    """뷰 i 의 출력을 정답 영상(표시 sRGB [0,1], H x W x 3)으로. 교사 입력 모양에 맞춰 edge 패딩."""
    def f(imgs):
        out = torch.zeros_like(imgs)
        for i, g in enumerate(gt_display_list):
            a = np.asarray(g, np.float32)
            hh, ww = imgs.shape[2], imgs.shape[3]
            a = np.pad(a, ((0, hh - a.shape[0]), (0, ww - a.shape[1]), (0, 0)), mode="edge")
            out[i] = torch.from_numpy(2.0 * a - 1.0).permute(2, 0, 1)
        return out
    return f


def hidden_gt_linear(dw):
    """숨은 층 텍셀(소스 화소 p)의 정답 선형색 = 배경 텍스처[p] (배경 평면은 정면, 소스 화소와 같은 위치)."""
    return bg_tex(dw.world.obs.width, dw.world.obs.height)


def project_row_pixels(obs, mu):
    """TSQ mu(세계) -> 소스 화소 (행, 열) 반올림."""
    uv, _ = geom.project(obs.K, obs.R, obs.t, np.asarray(mu, np.float64), obs.k1, obs.k2)
    px = np.rint(uv).astype(int)
    return px[:, 1], px[:, 0]


def box_visibility(dw, views, margin_m=0.1, border_px=4, rim=4):
    """해석적 가림 오라클(구현과 독립). 숨은 텍셀(상자 화소 위치의 6 m 배경점)을 각 뷰 카메라 중심에서 볼 때
    광선이 상자 평면(2 m)을 상자 사각형 안에서 지나면 가려진다. 반환:
    hid(행,열 인덱스), clear_vis (n_views, N) bool, clear_occ (n_views, N) bool.
    ``margin_m`` 안쪽 띠와, 숨은 영역 가장자리 ``rim`` 화소(층 0 배경과 같은 평면에서 맞닿아, 뷰에서 어느 층이 화소를
    차지하는지가 동률인 곳)는 어느 쪽도 아니다(구현의 부분화소·침식 선택에 맡긴다)."""
    obs = dw.world.obs
    K = np.asarray(obs.K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = np.nonzero(dw.hidden)
    X = np.stack([(uu - cx) / fx * Z_BG, (vv - cy) / fy * Z_BG, np.full(uu.shape, Z_BG)], axis=1).astype(np.float64)
    xl, xr = (X0 - 0.5 - cx) / fx * Z_FG, (X1 - 0.5 - cx) / fx * Z_FG
    yt, yb = (Y0 - 0.5 - cy) / fy * Z_FG, (Y1 - 0.5 - cy) / fy * Z_FG
    from scipy import ndimage as ndi
    depth_in = ndi.distance_transform_edt(dw.hidden)[vv, uu]
    tie = depth_in < rim
    vis, occ = [], []
    for v in views:
        c = np.asarray(v.c2w, np.float64)[:3, 3]
        s = (Z_FG - c[2]) / (X[:, 2] - c[2])
        q = c[None, :] + s[:, None] * (X - c[None, :])
        m = np.minimum.reduce([q[:, 0] - xl, xr - q[:, 0], q[:, 1] - yt, yb - q[:, 1]])
        w2c = geom.invert_pose(np.asarray(v.c2w, np.float64))
        uv, z = geom.project(v.K, w2c[:3, :3], w2c[:3, 3], X, v.k1, v.k2)
        inb = ((z > 0.1) & (uv[:, 0] >= border_px) & (uv[:, 0] <= v.width - 1 - border_px)
               & (uv[:, 1] >= border_px) & (uv[:, 1] <= v.height - 1 - border_px))
        vis.append((m < -margin_m) & inb & ~tie)
        occ.append((m > margin_m) & ~tie)
    return (vv, uu), np.array(vis), np.array(occ)


# --------------------------------------------------------------------------- 캐시된 표준 실행

import importlib  # noqa: E402
from functools import lru_cache  # noqa: E402


def mod():
    return importlib.import_module("buc.world.sdedit_avg")


@lru_cache(maxsize=None)
def oracle_run():
    """정답 오라클 교사로 한 번 채운 표준 실행(테스트들이 읽기만 한다)."""
    S = mod()
    read, dw = build()
    cfg = S.SDEditAvgConfig()
    views = S.plan_views(dw, cfg)
    sampler = FakeSampler(oracle_transform(gt_views_display(dw.world.obs, views)))
    res = S.sdedit_avg_fill(dw, sampler, cfg=cfg, views=views, seed=11)
    return SimpleNamespace(read=read, dw=dw, cfg=cfg, views=views, sampler=sampler, res=res)


def layer_bytes_equal(a, b):
    return np.array_equal(a.rgba, b.rgba) and np.array_equal(a.depth, b.depth)
