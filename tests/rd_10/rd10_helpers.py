"""RD-10 테스트 공용 도우미: 합성 장면(구·돔·두 법선), 독립 오라클(계약서 수식 그대로), 지표.
RD-10 구현을 오라클에 쓰지 않는다(공개 dataclass 생성자 제외). 난수는 모두 시드 고정 Generator."""
import math

import numpy as np


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


# ------------------------------------------------------------------------------------------------ 합성 장면
KEY_DIR = unit([-0.5, -0.5, -0.7])        # 표면 -> 광원(카메라 좌표, OpenCV: z<0 = 카메라 쪽, y<0 = 위)
RIM_DIR = unit([0.75, -0.2, 0.63])        # 역광 림(z>0 = 카메라 반대쪽): 구의 가장자리 초승달만 비춘다
DOME_DIR = unit([-0.4, -0.5, -0.77])
AMB = 0.15


def ball(h=32, w=32, r=0.92):
    """화면을 꽉 채운 구의 카메라 쪽 반구 법선(단위, z<0)과 유효 원판 마스크."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (xx - (w - 1) / 2) / ((w - 1) / 2)
    y = (yy - (h - 1) / 2) / ((h - 1) / 2)
    rho2 = x * x + y * y
    valid = rho2 <= r * r
    z = -np.sqrt(np.maximum(1.0 - rho2, 1e-6))
    n = np.stack([x, y, z], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True), valid


def dome(h=24, w=32, depth_amp=18.0, sigma_frac=0.22):
    """카메라 쪽으로 솟은 가우시안 범프의 법선(단위, 카메라를 향함)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    s = sigma_frac * w
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
    z = 50.0 - depth_amp * g
    zy, zx = np.gradient(z)
    n = np.stack([zx, zy, -np.ones_like(z)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def two_normal_scene(h=24, w=32):
    """바닥(위 향함)과 벽(카메라 향함) 두 평면. 영상은 두 값뿐이라 방향이 결정되지 않는다(사전분포 시험용)."""
    n = np.zeros((h, w, 3))
    n[: h // 2] = (0.0, -1.0, 0.0)
    n[h // 2:] = (0.0, 0.0, -1.0)
    return n


def lam(n, d):
    """max(0, n·ω) (오라클)."""
    return np.maximum(np.asarray(n, np.float64) @ np.asarray(d, np.float64), 0.0)


def shading_from(n, lights, ambient, vis=None, cover=None):
    """lights: [(dir, intensity)], vis: [(H,W)|None], cover: [(H,W)|None] (플래그 감쇠 = 1-coverage)."""
    s = np.full(n.shape[:2], float(ambient))
    for j, (d, k) in enumerate(lights):
        t = lam(n, d)
        if vis is not None and vis[j] is not None:
            t = t * vis[j]
        if cover is not None and cover[j] is not None:
            t = t * (1.0 - cover[j])
        s = s + k * t
    return s


def energy_oracle(n, lights, ambient, roles, weight):
    """E_j = Σ w I_j T_j / Σ w, 점유 = E/(a+ΣE), 치트 몫 = Σ_{cheat} 점유."""
    w = np.asarray(weight, np.float64)
    e = [float(np.sum(w * k * lam(n, d)) / w.sum()) for d, k in lights]
    tot = ambient + sum(e)
    sh = [x / tot for x in e]
    cs = sum(s for s, r in zip(sh, roles) if r == "cheat")
    return e, sh, cs


# ------------------------------------------------------------------------------------------------ 합성 가시성(그림자) 함수
def make_shadow_vis(h=24, w=32, base=(16.0, 8.0), length=14.0, rad=2.6):
    """방향 -> (H,W) 가시성. 기둥(밑 base=(행,열))이 광원 반대쪽으로 끌리는 선분 그림자. 방향에 매끈하게 의존."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    def vis(d):
        d = np.asarray(d, np.float64)
        q = d[:2] / (abs(d[2]) + 0.2)
        nq = np.linalg.norm(q)
        t = -q / (nq + 1e-9) * length * min(1.0, nq)
        p0 = np.array([base[1], base[0]])
        seg = t
        l2 = max(float(seg @ seg), 1e-9)
        u = np.clip(((xx - p0[0]) * seg[0] + (yy - p0[1]) * seg[1]) / l2, 0.0, 1.0)
        dist = np.hypot(xx - (p0[0] + u * seg[0]), yy - (p0[1] + u * seg[1]))
        return 1.0 - np.clip(rad + 0.5 - dist, 0.0, 1.0)

    return vis
