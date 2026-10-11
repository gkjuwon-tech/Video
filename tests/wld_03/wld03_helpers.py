"""WLD-03 테스트 공용: 합성 장면, ReadPacket 생성, 계약서 식의 독립 오라클."""
import numpy as np

from buc.core import contracts as C

USER_PROV = C.Prov(taint=frozenset({"USER"}), mode="user", ns="user_vault")

# 기본 장면: 96x64, fx=fy=100, 전경 사각형(행 20..43, 열 40..63) 2 m, 배경 6 m
W, H = 96, 64
FX = 100.0
Y0, Y1, X0, X1 = 20, 44, 40, 64  # 반열린 구간
Z_FG, Z_BG = 2.0, 6.0
BASE = 0.3  # 옆 이동(m)
RED = np.array([0.9, 0.1, 0.1])
BLUE = np.array([0.1, 0.2, 0.8])


def K_of(w=W, h=H, f=FX):
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])


def obs_of(w=W, h=H, f=FX, R=None, t=None, k1=0.0, k2=0.0):
    return C.Obs(K=K_of(w, h, f), R=np.eye(3) if R is None else R, t=np.zeros(3) if t is None else t,
                 width=w, height=h, k1=k1, k2=k2)


def make_read(image, depth, obs=None, depth_conf=None, level="L1", prov=USER_PROV):
    image = np.asarray(image, np.float32)
    depth = np.asarray(depth, np.float32)
    h, w = depth.shape
    obs = obs or obs_of(w, h)
    return C.ReadPacket(image_lin=image, depth=depth, obs=obs, theta_in=C.Theta(aspect=w / h),
                        input_hash="b" * 64, prov=prov, level=level,
                        depth_conf=None if depth_conf is None else np.asarray(depth_conf, np.float32))


def bg_gradient(w=W, h=H):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.empty((h, w, 3))
    img[..., 0] = 0.1 + 0.3 * u / w
    img[..., 1] = 0.5
    img[..., 2] = 0.2 + 0.4 * v / h
    return img


def fg_mask(w=W, h=H):
    m = np.zeros((h, w), bool)
    m[Y0:Y1, X0:X1] = True
    return m


def box_scene():
    """하드 에지 전경 사각형(빨강, 2 m) + 매끈한 배경 그라디언트(6 m)."""
    img = bg_gradient()
    m = fg_mask()
    img[m] = RED
    depth = np.where(m, Z_FG, Z_BG)
    return img.astype(np.float32), depth.astype(np.float32)


def ring_mask():
    """전경 사각형 바로 바깥 Chebyshev 거리 1 링."""
    big = np.zeros((H, W), bool)
    big[Y0 - 1:Y1 + 1, X0 - 1:X1 + 1] = True
    return big & ~fg_mask()


def halo_scene():
    """혼합 화소 링(0.5 빨강 + 0.5 파랑)이 배경 깊이로 잘못 붙은 장면 + 정답 매트."""
    m = fg_mask()
    ring = ring_mask()
    img = np.empty((H, W, 3))
    img[:] = BLUE
    img[m] = RED
    img[ring] = 0.5 * RED + 0.5 * BLUE
    depth = np.where(m, Z_FG, Z_BG)
    alpha = np.where(m, 1.0, np.where(ring, 0.5, 0.0))
    band = np.zeros((H, W), bool)
    band[Y0 - 2:Y1 + 2, X0 - 2:X1 + 2] = True
    band[Y0 + 2:Y1 - 2, X0 + 2:X1 - 2] = False
    fg = np.empty((H, W, 3))
    fg[:] = RED
    bg = np.empty((H, W, 3))
    bg[:] = BLUE
    return img.astype(np.float32), depth.astype(np.float32), Matte(alpha, fg, bg, band)


class Matte:
    def __init__(self, alpha, fg, bg, band):
        self.alpha, self.fg, self.bg, self.band = alpha, fg, bg, band


def shifted_c2w(dx=BASE):
    c2w = np.eye(4)
    c2w[0, 3] = dx
    return c2w


def composite(world):
    """앞→뒤 직선 알파 over (float64)."""
    h, w = world.layers[0].rgba.shape[:2]
    rgb = np.zeros((h, w, 3))
    trans = np.ones((h, w, 1))
    for ly in world.layers:
        a = ly.rgba[..., 3:4].astype(np.float64)
        rgb += ly.rgba[..., :3].astype(np.float64) * a * trans
        trans = trans * (1 - a)
    return rgb


def tear_oracle(z, rel):
    z = np.asarray(z, np.float64)
    dx = np.abs(z[:, 1:] - z[:, :-1]) > rel * np.minimum(z[:, 1:], z[:, :-1])
    dy = np.abs(z[1:, :] - z[:-1, :]) > rel * np.minimum(z[1:, :], z[:-1, :])
    near = np.zeros(z.shape, bool)
    far = np.zeros(z.shape, bool)
    a, b = z[:, :-1], z[:, 1:]
    near[:, :-1] |= dx & (a < b)
    near[:, 1:] |= dx & (b < a)
    far[:, :-1] |= dx & (a > b)
    far[:, 1:] |= dx & (b > a)
    a, b = z[:-1, :], z[1:, :]
    near[:-1, :] |= dy & (a < b)
    near[1:, :] |= dy & (b < a)
    far[:-1, :] |= dy & (a > b)
    far[1:, :] |= dy & (b > a)
    return dx, dy, near, far


def cheb_dist_to_rect_boundary(y0, y1, x0, x1, h=H, w=W):
    """각 화소에서 사각형 [y0,y1)x[x0,x1) 경계까지의 대략적 Chebyshev 거리(안·밖 모두)."""
    v, u = np.mgrid[0:h, 0:w]
    inside = (v >= y0) & (v < y1) & (u >= x0) & (u < x1)
    dy_out = np.maximum(np.maximum(y0 - v, v - (y1 - 1)), 0)
    dx_out = np.maximum(np.maximum(x0 - u, u - (x1 - 1)), 0)
    out_d = np.maximum(dy_out, dx_out)
    in_d = np.minimum.reduce([v - y0, (y1 - 1) - v, u - x0, (x1 - 1) - u]) + 1
    return np.where(inside, in_d, out_d)
