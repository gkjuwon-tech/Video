"""RD-15 테스트 공용 도구. 결정론: 지역 default_rng 만 쓴다(전역 RNG 불변).

- 합성 얼굴 영상: 피부 타원 + 눈 한 쌍 + (선택) 머리카락. 배경은 피부색이 아닌 하늘/잎/회색 벽.
- 머리 모형 오라클: 계약서의 두통(타원체 + 코 타원체) 광선 투사. 구현의 render_head 와 독립으로 짠 것이다.
- 가짜 판독(readout): RD-07 MrhReadout 의 덕 타이핑 대역(.outputs, .src_hw, .read_hw, .canvas_hw).
- StubReader: RD-07 읽기기 프로토콜(snap, __call__)의 대역. 호출을 기록하고 sem 헤드를 주입할 수 있다.
RD-15 구현(buc.read.faces)은 테스트 함수 안에서 지연 import 한다.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from buc.core import color
from buc.gov import prov as P

USER = P.as_record(["USER"])
ENGINE = P.as_record(["ENGINE"])

# ----------------------------------------------------------------------------- 피부 오라클 (계약서 상수)
SKIN_CENTER = (102.0, 153.0)
SKIN_AXES = (26.0, 20.0)


def skin_oracle(image_lin: np.ndarray) -> np.ndarray:
    """BT.601 풀레인지 YCbCr(sRGB 부호화 8비트 환산)의 Cb,Cr 타원 안쪽 거리 -> clip(1-d2,0,1)."""
    e = color.linear_to_display(np.asarray(image_lin, np.float64)).astype(np.float64) * 255.0
    r, g, b = e[..., 0], e[..., 1], e[..., 2]
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    d2 = ((cb - SKIN_CENTER[0]) / SKIN_AXES[0]) ** 2 + ((cr - SKIN_CENTER[1]) / SKIN_AXES[1]) ** 2
    return np.clip(1.0 - d2, 0.0, 1.0)


# ----------------------------------------------------------------------------- 합성 얼굴 영상
TONES = {  # sRGB 8비트 (피부색 넷)
    "light": (240, 200, 170), "medium": (224, 172, 140), "tan": (198, 134, 96), "deep": (141, 85, 36),
}
NON_SKIN = {"blue": (60, 90, 200), "green": (60, 160, 70), "gray": (150, 150, 150)}   # 같은 구도, 피부색이 아닌 얼굴판
_ALL_TONES = {**TONES, **NON_SKIN}
EYE_RGB = (40, 28, 24)
HAIR_RGB = (50, 36, 26)


def background(h: int, w: int, kind: str = "sky", seed: int = 0) -> np.ndarray:
    """sRGB 0..1 (h,w,3) 비피부색 배경."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, h)[:, None, None]
    if kind == "sky":
        top, bot = np.array([0.30, 0.50, 0.85]), np.array([0.70, 0.82, 0.95])
        img = top * (1 - yy) + bot * yy + np.zeros((h, w, 3))
    elif kind == "foliage":
        n = gaussian_filter(rng.normal(size=(h, w)), 4)
        n = (n - n.min()) / (n.max() - n.min())
        img = np.stack([0.10 + 0.10 * n, 0.35 + 0.30 * n, 0.08 + 0.08 * n], -1)
    elif kind == "gray":
        img = np.full((h, w, 3), 0.45) + 0.05 * yy
    elif kind == "dark":
        img = np.full((h, w, 3), 0.06)
    else:
        raise ValueError(kind)
    return np.clip(img, 0, 1)


def _cov(sd: np.ndarray) -> np.ndarray:
    """부호 거리(px, 안쪽 양수) -> 안티앨리어싱 덮임 [0,1]."""
    return np.clip(sd + 0.5, 0.0, 1.0)


def _ellipse_sd(yy, xx, cy, cx, ry, rx):
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    return (1.0 - r) * min(ry, rx)


def draw_face(img: np.ndarray, cy: float, cx: float, fh: float, *, tone="medium", eyes="pair", hair=False,
              mouth=False, shade=0.08) -> tuple:
    """img(sRGB 0..1, 제자리 수정)에 얼굴을 그린다. 반환: 정답 상자 (y0,x0,y1,x1).

    비율: 높이 fh, 폭 0.75 fh. 눈 중심은 (cy-0.10fh, cx±0.19fh), 반지름 0.055fh.
    eyes: "pair"(두 눈) | "one"(왼쪽 한 개) | "none".
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ry, rx = fh / 2.0, 0.375 * fh
    skin = np.array(_ALL_TONES[tone], np.float64) / 255.0
    if hair:
        cov = _cov(_ellipse_sd(yy, xx, cy - 0.06 * fh, cx, 0.56 * fh, 0.43 * fh)) * (yy < cy - 0.20 * fh)
        img[:] = img * (1 - cov[..., None]) + (np.array(HAIR_RGB) / 255.0) * cov[..., None]
    cov = _cov(_ellipse_sd(yy, xx, cy, cx, ry, rx))
    if hair:
        cov = cov * (yy >= cy - 0.32 * fh)
    shading = 1.0 + shade * (xx - cx) / fh
    face = np.clip(skin * shading[..., None], 0, 1)
    img[:] = img * (1 - cov[..., None]) + face * cov[..., None]
    er = max(0.055 * fh, 1.4)
    spots = {"pair": [(-0.10, -0.19, 1.0), (-0.10, 0.19, 1.0)], "one": [(-0.10, -0.19, 1.0)],
             "none": []}[eyes]
    for dy, dx, k in spots:
        c = _cov(k * er - np.hypot(yy - (cy + dy * fh), xx - (cx + dx * fh)))
        img[:] = img * (1 - c[..., None]) + (np.array(EYE_RGB) / 255.0) * c[..., None]
    if mouth and eyes in ("pair", "one"):
        c = _cov(_ellipse_sd(yy, xx, cy + 0.26 * fh, cx, max(0.025 * fh, 1.0), 0.12 * fh))
        img[:] = img * (1 - c[..., None]) + np.array([0.62, 0.28, 0.27]) * c[..., None]
    return (cy - ry, cx - rx, cy + ry, cx + rx)


def to_linear(img_srgb: np.ndarray, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """sRGB 0..1 -> 선형 float32 (+ 센서 잡음 sRGB 값에 noise)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(img_srgb, np.float64)
    if noise > 0:
        x = np.clip(x + rng.normal(0, noise, x.shape), 0, 1)
    return np.ascontiguousarray(color.srgb_to_linear(x).astype(np.float32))


def scene(h=240, w=320, faces=(), bg="sky", seed=0, noise=0.01):
    """faces: [(cy,cx,fh,kwargs)] -> (image_lin, gt_boxes)."""
    img = background(h, w, bg, seed)
    boxes = []
    for cy, cx, fh, kw in faces:
        boxes.append(draw_face(img, cy, cx, fh, **kw))
    return to_linear(img, noise, seed + 1), boxes


def iou(a, b) -> float:
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, y1 - y0) * max(0.0, x1 - x0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------- 머리 모형 오라클
CRANIUM_AXES = np.array([0.075, 0.115, 0.095])      # 반축 (x 오른쪽, y 위, z 앞) m, 폭 0.15 m 기준
NOSE_CENTER = np.array([0.0, -0.015, 0.100])
NOSE_AXES = np.array([0.015, 0.035, 0.030])
W_REF = 0.15


def head_frame(front, up):
    """CAM-04 규약: front 정규화, up 은 직교화, right = cross(up, front). 열 = (right, up, front)."""
    f = np.asarray(front, np.float64)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, np.float64)
    u = u - f * (f @ u)
    u = u / np.linalg.norm(u)
    r = np.cross(u, f)
    return np.stack([r, u, f], axis=1), f, u


def _hit(o, d, c, ax):
    """광선 o + t d (head 좌표) 와 중심 c 반축 ax 타원체의 가까운 양의 교점 t (없으면 inf)."""
    oc = (o - c) / ax
    dd = d / ax
    a = (dd ** 2).sum(-1)
    b = 2.0 * (oc * dd).sum(-1)
    cc = (oc ** 2).sum(-1) - 1.0
    disc = b * b - 4 * a * cc
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    t = np.where(t0 > 1e-9, t0, np.where(t1 > 1e-9, t1, np.inf))
    return np.where(ok, t, np.inf)


def render_oracle(center, front, up, width_m, f_px, grid_hw, cell_px=4.0, pp=None, origin=(0.0, 0.0)):
    """격자 칸 (i,j) 의 중심 화소 (origin + cell_px*(i,j) + (cell_px-1)/2) 에서 광선을 쏴 깊이·법선을 낸다.

    반환 depth (h,w) float64(카메라 z, 밖은 nan), normal (h,w,3)(카메라 좌표, 단위, 밖은 0), mask (h,w).
    주점 pp=(cy,cx) 기본 없음 -> 호출자가 준다."""
    h, w = grid_hw
    R, f, u = head_frame(front, up)
    s = width_m / W_REF
    cen = np.asarray(center, np.float64)
    ii, jj = np.mgrid[0:h, 0:w].astype(np.float64)
    py = origin[0] + cell_px * ii + (cell_px - 1) / 2.0
    px = origin[1] + cell_px * jj + (cell_px - 1) / 2.0
    d_cam = np.stack([(px - pp[1]) / f_px, (py - pp[0]) / f_px, np.ones_like(px)], -1)
    d_h = d_cam @ R                      # R^T d (행벡터 규약)
    o_h = (-cen) @ R                     # 카메라 원점의 head 좌표
    o_h = np.broadcast_to(o_h, d_h.shape)
    t1 = _hit(o_h, d_h, np.zeros(3), CRANIUM_AXES * s)
    t2 = _hit(o_h, d_h, NOSE_CENTER * s, NOSE_AXES * s)
    use_nose = t2 < t1
    t = np.where(use_nose, t2, t1)
    mask = np.isfinite(t)
    tt = np.where(mask, t, 0.0)
    p_h = o_h + tt[..., None] * d_h
    n_c = (p_h / (CRANIUM_AXES * s) ** 2)
    n_n = ((p_h - NOSE_CENTER * s) / (NOSE_AXES * s) ** 2)
    n_h = np.where(use_nose[..., None], n_n, n_c)
    n_h = n_h / np.maximum(np.linalg.norm(n_h, axis=-1, keepdims=True), 1e-12)
    n_cam = n_h @ R.T
    n_cam = np.where(mask[..., None], n_cam, 0.0)
    depth = np.where(mask, tt, np.nan)
    return depth, n_cam, mask


def fm():
    """구현 모듈을 지연 import 한다(테스트 수집은 구현 없이도 가능)."""
    import importlib
    return importlib.import_module("buc.read.faces")


def pose_from_yaw_pitch(center, yaw_deg, pitch_deg):
    """CAM-04 규약의 (front, up) 와 정답 (yaw, pitch) 를 만든다.

    원점의 카메라를 얼굴 좌표계 (right, up, front) 에서 본 단위 방향이 c = (sin y cos p, sin p, cos y cos p) 가
    되도록 회전을 잡는다(기저 R 의 열 = right, up, front; R @ c = 머리->카메라 단위벡터 w).
    반환 (front, up, (yaw, pitch))는 CAM-04 face_angles 로 다시 계산한 값이다."""
    from scipy.spatial.transform import Rotation as Rot

    from buc.camera.face_envelope import FaceSpec, face_angles
    cen = np.asarray(center, np.float64)
    w = -cen / np.linalg.norm(cen)
    y_up = np.array([0.0, -1.0, 0.0])                   # OpenCV 의 위쪽
    right0 = np.cross(y_up - w * (w @ y_up), w)
    right0 /= np.linalg.norm(right0)
    up0 = np.cross(w, right0)
    base = np.stack([right0, up0, w], axis=1)           # 열 (right, up, front), front = w (정면 응시)
    ya, pa = np.radians(yaw_deg), np.radians(pitch_deg)
    c_loc = np.array([np.sin(ya) * np.cos(pa), np.sin(pa), np.cos(ya) * np.cos(pa)])
    ez = np.array([0.0, 0.0, 1.0])
    ax = np.cross(c_loc, ez)
    sn = np.linalg.norm(ax)
    q = np.eye(3) if sn < 1e-12 else Rot.from_rotvec(ax / sn * np.arctan2(sn, c_loc @ ez)).as_matrix()
    r = base @ q                                        # r @ c_loc = w
    front, up = r[:, 2], r[:, 1]
    yaw, pitch, _ = face_angles(FaceSpec(center=cen, front=front, up=up, h_f=0.1), np.zeros(3))
    return front, up, (float(yaw), float(pitch))


# ----------------------------------------------------------------------------- 가짜 판독·읽기기
def make_readout(src_hw, *, log_depth=None, normal=None, sem=None, sigma_hat=None, read_hw=None):
    """RD-07 MrhReadout 대역. 격자 = ceil(read/4). 주어진 헤드만 outputs 에 넣는다([1,c,h,w] f32)."""
    read_hw = tuple(read_hw or src_hw)
    h4, w4 = -(-read_hw[0] // 4), -(-read_hw[1] // 4)
    out = {}
    for k, v in (("log_depth", log_depth), ("normal", normal), ("sem", sem), ("sigma_hat", sigma_hat)):
        if v is None:
            continue
        t = torch.as_tensor(np.asarray(v), dtype=torch.float32)
        if t.ndim == 2:
            t = t[None]
        if t.ndim == 3:
            t = t[None]
        assert t.shape[-2:] == (h4, w4), (k, tuple(t.shape), (h4, w4))
        out[k] = t.contiguous()
    return SimpleNamespace(outputs=out, src_hw=tuple(src_hw), read_hw=read_hw,
                           canvas_hw=(16 * -(-read_hw[0] // 16), 16 * -(-read_hw[1] // 16)))


def unit_sem(h4, w4, cls, n_cls=6, dim=16, seed=0):
    """sem 대역: 칸마다 table[cls] 방향의 단위 벡터. 반환 (sem [1,D,h,w] np, table [N,D] np)."""
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(n_cls, dim))
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    cls = np.broadcast_to(np.asarray(cls), (h4, w4))
    sem = table[cls].transpose(2, 0, 1)[None].astype(np.float32)
    return sem, table.astype(np.float32)


class StubReader:
    """RD-07 읽기기 프로토콜 대역(snap, __call__). 입력 영상의 4x4 평균 휘도에서 8개 헤드를 낸다.

    sem_fn(image[1,3,h,w], key) -> [1,D,h/4,w/4] 로 sem 을 덮어쓸 수 있다."""

    def __init__(self, sem_fn=None, points=(1.0, 0.75, 0.5, 0.35, 0.25, 0.2)):
        self.points = tuple(points)
        self.sem_fn = sem_fn
        self.calls = []

    def snap(self, sigma) -> float:
        return float(min(self.points, key=lambda p: (abs(p - float(sigma)), -p)))

    def __call__(self, image, sigma, *, prov, key):
        from buc.read.mrh_call import ViewRead
        import torch.nn.functional as F
        self.calls.append(SimpleNamespace(shape=tuple(image.shape), sigma=float(sigma), key=key))
        g4 = F.avg_pool2d(image.mean(dim=1, keepdim=True), 4)
        nrm = torch.cat([g4 * 0, g4 * 0, -torch.ones_like(g4)], 1)
        sem = torch.ones_like(g4).repeat(1, 4, 1, 1) / 2.0 if self.sem_fn is None else self.sem_fn(image, key)
        out = {
            "log_depth": 0.1 * g4, "normal": nrm, "albedo": torch.sigmoid(g4.repeat(1, 3, 1, 1)),
            "rough_metal": torch.sigmoid(g4.repeat(1, 2, 1, 1)), "shading": F.softplus(g4), "defocus": F.softplus(-g4),
            "sem": sem, "sigma_hat": F.softplus(g4.repeat(1, 4, 1, 1)) + 1e-3,
        }
        return ViewRead(outputs=out, sigma=self.snap(sigma), digest=bytes(32), ephemeral=bool(P.ext_bit(prov)))


# ----------------------------------------------------------------------------- 종단 장면 (영상 + 판독)
F_PX = 600.0
S_TRUE = 1.7          # 판독 게이지: log_depth = ln(z_미터 / S_TRUE)  (스케일 전용 게이지)
FACE_CLS, BG_CLS = 2, 0


def pp_of(h, w):
    return ((h - 1) / 2.0, (w - 1) / 2.0)


def head_geometry(yaw, pitch, fh, cy, cx, hw=(240, 320), width=0.15, f_px=F_PX):
    """영상 (cy,cx)에 높이 약 fh 화소로 보이는 머리의 정답: (center, front, up, (yaw, pitch))."""
    h, w = hw
    pp = pp_of(h, w)
    z = f_px * 0.23 * (width / W_REF) / fh
    cen = np.array([(cx - pp[1]) / f_px * z, (cy - pp[0]) / f_px * z, z])
    front, up, yp = pose_from_yaw_pitch(cen, yaw, pitch)
    return cen, front, up, yp


def head_readout(yaw=20.0, pitch=5.0, fh=80, cy=110, cx=150, hw=(240, 320), width=0.15, *, noise_d=0.0, noise_n=0.0,
                 seed=0, read_div=1, sem_face_p=None, with_geometry=True, with_sigma_hat=None):
    """머리 모형 정답 깊이·법선·sem 을 담은 MrhReadout 대역과 정답.

    read_div=2 면 읽기 해상도가 원본의 절반이다(격자도 절반, 한 칸 = 4*read_div 원본 화소).
    반환 (readout, table, truth) ; truth = dict(center, front, up, yaw, pitch, width, mask, grid_hw)."""
    h, w = hw
    rh, rw = h // read_div, w // read_div
    h4, w4 = -(-rh // 4), -(-rw // 4)
    cell_px = 4.0 * read_div
    pp = pp_of(h, w)
    cen, front, up, (yw, pt) = head_geometry(yaw, pitch, fh, cy, cx, hw, width)
    z, n, m = render_oracle(cen, front, up, width, F_PX, (h4, w4), cell_px, pp)
    rng = np.random.default_rng(seed)
    zz = np.where(m, z + (rng.normal(0, noise_d, z.shape) if noise_d else 0.0), 4.0)
    nn = n.copy()
    if noise_n:
        nn = n + rng.normal(0, np.tan(np.radians(noise_n)), n.shape) * m[..., None]
        nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-9)
    nn = np.where(m[..., None], nn, np.array([0.0, 0.0, -1.0]))
    sem, table = unit_sem(h4, w4, np.where(m, FACE_CLS, BG_CLS))
    if sem_face_p is not None:      # 얼굴 칸의 얼굴 확률을 sem_face_p 로 맞춘다(2클래스 혼합, table 의 FACE/BG 행)
        sem = mix_sem(table, m, sem_face_p)
    kw = dict(sem=sem)
    if with_geometry:
        kw.update(log_depth=np.log(zz / S_TRUE)[None], normal=nn.transpose(2, 0, 1))
    if with_sigma_hat is not None:
        kw["sigma_hat"] = np.full((4, h4, w4), with_sigma_hat)
    ro = make_readout((h, w), read_hw=(rh, rw), **kw)
    truth = dict(center=cen, front=front, up=up, yaw=yw, pitch=pt, width=width, mask=m, grid_hw=(h4, w4),
                 cell_px=cell_px)
    return ro, table, truth


def mix_sem(table, mask, p, temp=0.07):
    """얼굴 칸에서 softmax(<s,t>/temp) 의 FACE 클래스 확률이 p 가 되도록 table[FACE]·table[BG] 의 혼합 단위벡터를 둔다."""
    from scipy.optimize import brentq
    tf, tb = table[FACE_CLS].astype(np.float64), table[BG_CLS].astype(np.float64)
    # 직교화한 기저 (e1=tf, e2) 에서 s = cos(a) e1 + sin(a) e2: 다른 클래스까지 포함한 정확한 확률을 풀이로 맞춘다.
    e1 = tf / np.linalg.norm(tf)
    e2 = tb - e1 * (e1 @ tb)
    e2 /= np.linalg.norm(e2)
    t = table.astype(np.float64)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)

    def prob(a):
        s = np.cos(a) * e1 + np.sin(a) * e2
        lg = t @ s / temp
        lg -= lg.max()
        e = np.exp(lg)
        return e[FACE_CLS] / e.sum()
    a = brentq(lambda x: prob(x) - p, 0.0, np.pi / 2 + 0.6)
    s_face = np.cos(a) * e1 + np.sin(a) * e2
    h4, w4 = mask.shape
    out = np.where(mask[None], s_face[:, None, None], t[BG_CLS][:, None, None])
    return out[None].astype(np.float32)


def head_image(fh=80, cy=110, cx=150, hw=(240, 320), tone="medium", bg="sky", seed=0, **kw):
    """head_readout 과 같은 자리에 얼굴(피부 타원 + 눈)을 그린 선형 영상과 정답 상자."""
    img, boxes = scene(hw[0], hw[1], [(cy, cx, fh, dict(tone=tone, **kw))], bg=bg, seed=seed)
    return img, boxes[0]


def face_sem_fn(table):
    """StubReader.sem_fn: 모든 칸이 FACE 클래스 방향이다."""
    t = torch.as_tensor(np.asarray(table), dtype=torch.float32)[FACE_CLS]
    t = t / t.norm()

    def fn(image, key):
        _, _, hh, ww = image.shape
        return t.view(1, -1, 1, 1).expand(1, t.numel(), hh // 4, ww // 4).contiguous()
    return fn


def blank_image(h=240, w=320, kind="sky"):
    return to_linear(background(h, w, kind, 0), 0.0, 0)


def sem_readout_from_boxes(boxes, hw=(240, 320), n_cls=6, dim=16, seed=0, geometry=False):
    """sem 만 있는 판독: 원본 화소 상자들의 칸이 FACE 클래스, 나머지 BG. 반환 (readout, table)."""
    h4, w4 = hw[0] // 4, hw[1] // 4
    cls = np.full((h4, w4), BG_CLS)
    for (y0, x0, y1, x1) in boxes:
        cls[int(y0 // 4):int(np.ceil(y1 / 4)), int(x0 // 4):int(np.ceil(x1 / 4))] = FACE_CLS
    sem, table = unit_sem(h4, w4, cls, n_cls, dim, seed)
    return make_readout(hw, sem=sem), table
