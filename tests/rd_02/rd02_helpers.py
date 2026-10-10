"""RD-02 테스트 오라클: 정답 카메라(K, R, 왜곡)로 맨해튼 선분·곡선 폴리라인·상자 방 영상을 합성한다.
RD-02 구현을 import 하지 않는다. numpy 와 buc.core.geom(FND-06)만 쓴다. 모든 난수는 지역 Generator.

규약(계약서 RD-02):
  카메라 OpenCV(x 오른쪽, y 아래, z 앞). up_cam = 세계 위쪽 방향을 카메라 좌표로 본 단위벡터.
  pitch = asin(up[2])  (위를 보면 +),  roll = atan2(up[0], -up[1])  (영상에서 수평선 오른쪽이 내려가면 +).
"""
import math

import numpy as np

from buc.core import geom

WORLD_UP = np.array([0.0, 1.0, 0.0])
AXES = np.eye(3)  # 세계 맨해튼 축 (x, y=위, z)


def K_from_fov(fov_x_deg, W, H, pp=None):
    f = (W / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    cx, cy = ((W - 1) / 2.0, (H - 1) / 2.0) if pp is None else pp
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def rot_axis(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def camera_R(yaw_deg, pitch_deg, roll_deg):
    """세계->카메라 회전. 수평 카메라(세계 +z를 봄)에서 yaw(세계 위축), 카메라 x축 pitch, 카메라 z축 roll 순."""
    R0 = np.diag([-1.0, -1.0, 1.0])  # 세계 y 위 -> 카메라 y 아래 (det +1)
    Ry = rot_axis((0, 1, 0), yaw_deg)
    Rp = rot_axis((1, 0, 0), -pitch_deg)  # 카메라 x축 둘레, 위를 보면 +pitch (아래 검사로 확정)
    Rr = rot_axis((0, 0, 1), roll_deg)
    return Rr @ Rp @ R0 @ Ry


def up_cam(R):
    return R @ WORLD_UP


def roll_pitch(up):
    up = np.asarray(up, float) / np.linalg.norm(up)
    return math.degrees(math.atan2(up[0], -up[1])), math.degrees(math.asin(up[2]))


def vp_of(K, R, axis):
    v = K @ (R @ AXES[axis])
    return v / np.linalg.norm(v)


def ray_angle_deg(K, v, d_cam):
    """동차 VP v 의 광선 K^-1 v 와 카메라 방향 d_cam 사이 각(부호 무시)."""
    r = np.linalg.solve(K, np.asarray(v, float))
    c = abs(r @ d_cam) / (np.linalg.norm(r) * np.linalg.norm(d_cam))
    return math.degrees(math.acos(min(1.0, c)))


def manhattan_segments(K, R, W, H, n_per_axis=(60, 60, 60), noise_px=0.3, seed=0, min_len=40.0,
                       axes=(0, 1, 2), x_range=None, k1=0.0, k2=0.0):
    """세계 축 방향 3D 선분을 투영한 (N,4) 선분과 축 라벨 (N,). 끝점 잡음 noise_px.
    k1/k2 가 있으면 끝점만 왜곡한다(선분은 직선으로 남는다; 왜곡된 끝점 좌표).
    x_range=(lo,hi) 이면 두 끝점의 x 가 그 범위 안인 선분만 쓴다."""
    rng = np.random.default_rng(seed)
    segs, labs = [], []
    Kinv = np.linalg.inv(K)
    for ai, n in zip(axes, n_per_axis):
        d_w = AXES[ai]
        got, tries = 0, 0
        while got < n and tries < 200000:
            tries += 1
            u = rng.uniform(0, W - 1)
            v = rng.uniform(0, H - 1)
            depth = rng.uniform(2.0, 10.0)
            Pc = depth * (Kinv @ np.array([u, v, 1.0]))
            Pw = R.T @ Pc
            L = rng.uniform(0.4, 3.0)
            A, B = Pw - 0.5 * L * d_w, Pw + 0.5 * L * d_w
            Ac, Bc = R @ A, R @ B
            if Ac[2] < 0.3 or Bc[2] < 0.3:
                continue
            a = Ac[:2] / Ac[2]
            b = Bc[:2] / Bc[2]
            if k1 or k2:
                a = geom.distort(a, k1, k2)
                b = geom.distort(b, k1, k2)
            pa = K[:2, :2] @ a + K[:2, 2]
            pb = K[:2, :2] @ b + K[:2, 2]
            if not (0 <= pa[0] <= W - 1 and 0 <= pb[0] <= W - 1 and 0 <= pa[1] <= H - 1 and 0 <= pb[1] <= H - 1):
                continue
            if x_range is not None and not (x_range[0] <= min(pa[0], pb[0]) and max(pa[0], pb[0]) <= x_range[1]):
                continue
            if np.hypot(*(pa - pb)) < min_len:
                continue
            segs.append(np.r_[pa, pb] + rng.normal(0, noise_px, 4))
            labs.append(ai)
            got += 1
    return np.array(segs, dtype=np.float64), np.array(labs)


def random_segments(n, W, H, seed=0, min_len=40.0, max_len=200.0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = rng.uniform([0, 0], [W - 1, H - 1])
        a = rng.uniform(0, math.pi)
        L = rng.uniform(min_len, max_len)
        q = p + L * np.array([math.cos(a), math.sin(a)])
        if 0 <= q[0] <= W - 1 and 0 <= q[1] <= H - 1:
            out.append(np.r_[p, q])
    return np.array(out, dtype=np.float64)


def distorted_polylines(K, R, W, H, k1, k2, n_lines=24, n_pts=40, noise_px=0.2, seed=0, wiggle_px=0.0):
    """세계 직선을 투영·왜곡한 폴리라인 목록 (각 (n,2) 픽셀). wiggle_px>0 이면 법선 방향 사인 휨(에셔 '휜 직선')."""
    rng = np.random.default_rng(seed)
    Kinv = np.linalg.inv(K)
    out = []
    tries = 0
    while len(out) < n_lines and tries < 100000:
        tries += 1
        ai = len(out) % 3
        u, v = rng.uniform(0.02 * W, 0.98 * W), rng.uniform(0.02 * H, 0.98 * H)
        if math.hypot((u - K[0, 2]) / W, (v - K[1, 2]) / H) < 0.3:
            continue  # 곡률이 보이는 바깥쪽 선을 쓴다
        depth = rng.uniform(2.0, 6.0)
        Pw = R.T @ (depth * (Kinv @ np.array([u, v, 1.0])))
        L = rng.uniform(4.0, 10.0)
        s = np.linspace(-0.5 * L, 0.5 * L, 400)
        P = Pw[None, :] + s[:, None] * AXES[ai][None, :]
        Pc = P @ R.T
        if Pc[:, 2].min() < 0.3:
            continue
        xy = Pc[:, :2] / Pc[:, 2:3]
        xyd = geom.distort(xy, k1, k2)
        px = xyd * K[0, 0] + K[:2, 2]
        inside = (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
        if inside.sum() < 200 or not inside[np.argmax(inside):][: inside.sum()].all():
            continue
        px = px[inside]
        if np.hypot(*(px[-1] - px[0])) < 0.4 * min(W, H):
            continue
        idx = np.linspace(0, len(px) - 1, n_pts).round().astype(int)
        pts = px[idx].copy()
        if wiggle_px:
            d = pts[-1] - pts[0]
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            t = np.linspace(0, 1, n_pts)
            pts += wiggle_px * np.sin(2 * math.pi * 1.5 * t)[:, None] * nrm[None, :]
        pts += rng.normal(0, noise_px, pts.shape)
        out.append(pts)
    return out


def max_displacement_px(K, W, H, k1a, k2a, k1b, k2b):
    """두 방사 왜곡 모델의 화면 안 최대 변위 차(px). r 은 정규화 반경, 모서리까지."""
    cx, cy, f = K[0, 2], K[1, 2], K[0, 0]
    rmax = max(math.hypot(x - cx, y - cy) for x in (0, W - 1) for y in (0, H - 1)) / f
    r = np.linspace(0, rmax, 2001)
    da = r * (k1a * r ** 2 + k2a * r ** 4)
    db = r * (k1b * r ** 2 + k2b * r ** 4)
    return float(np.max(np.abs(da - db)) * f)


def render_room(K, R, W, H, k1=0.0, k2=0.0, ss=2, period=0.45):
    """카메라가 세계 원점에 있는 상자 방(체커 무늬 벽·바닥·천장)의 회색 영상 (H,W) float64 [0,1].
    왜곡 k1,k2 는 영상 쪽(픽셀 -> 정규화 왜곡 좌표 -> undistort -> 광선)."""
    lo = np.array([-3.0, -1.5, -4.0])
    hi = np.array([3.4, 2.3, 4.6])
    acc = np.zeros((H, W))
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    shade = np.array([0.0, 0.08, -0.06])
    for oy in offs:
        for ox in offs:
            xd = np.stack([(uu + ox - K[0, 2]) / K[0, 0], (vv + oy - K[1, 2]) / K[1, 1]], -1)
            xy = geom.undistort(xd, k1, k2) if (k1 or k2) else xd
            dc = np.concatenate([xy, np.ones((H, W, 1))], -1)
            dw = dc @ R  # = (R^T dc)
            with np.errstate(divide="ignore", invalid="ignore"):
                tpos = np.where(dw > 0, hi / dw, np.inf)
                tneg = np.where(dw < 0, lo / dw, np.inf)
            t = np.minimum(tpos, tneg)
            face = np.argmin(t, axis=-1)
            tt = np.take_along_axis(t, face[..., None], -1)[..., 0]
            P = dw * tt[..., None]
            val = np.zeros((H, W))
            for ax in range(3):
                m = face == ax
                o = [i for i in range(3) if i != ax]
                a = np.floor(P[..., o[0]] / period) + np.floor(P[..., o[1]] / period)
                c = np.where(np.mod(a, 2) == 0, 0.25, 0.75) + shade[ax]
                val = np.where(m, c, val)
            acc += val
    return acc / (ss * ss)


def render_polygon(W, H, corners, inside=0.2, outside=0.8, ss=4):
    """볼록 다각형(시계/반시계 무관)을 안티앨리어싱해 그린 (H,W) float64."""
    corners = np.asarray(corners, float)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    acc = np.zeros((H, W))
    n = len(corners)
    # 2D 외적의 z 성분. numpy 2는 2차원 벡터의 np.cross를 거부한다.
    e0, e1 = corners[1] - corners[0], corners[2] - corners[1]
    sgn = np.sign(e0[0] * e1[1] - e0[1] * e1[0])
    for oy in offs:
        for ox in offs:
            x, y = uu + ox, vv + oy
            ok = np.ones((H, W), bool)
            for i in range(n):
                a, b = corners[i], corners[(i + 1) % n]
                cr = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
                ok &= sgn * cr >= 0
            acc += np.where(ok, inside, outside)
    return acc / (ss * ss)


def point_line_dist(p, a, b):
    d = b - a
    return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / np.linalg.norm(d)
