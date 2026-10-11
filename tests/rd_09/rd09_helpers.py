"""RD-09 테스트 공용 도우미: 독립 오라클(계약서 에너지 그대로), 합성 장면(알려진 알베도·법선·조명), 지표.
RD-09 구현을 오라클에 쓰지 않는다(공개 dataclass 생성자 제외). 난수는 모두 시드 고정 Generator."""
import math

import numpy as np

W709 = np.array([0.2126, 0.7152, 0.0722])
Y_FLOOR = 1e-6      # 계약: 휘도·채널 로그 하한
S_MIN = 1e-4        # 계약: 사전 로그 하한
NORMAL_MIN = 0.5


# ------------------------------------------------------------------------------------------------ 합성 장면
def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


LIGHT_DIR = unit([-0.4, -0.5, -0.77])      # 표면 -> 광원(카메라 좌표, 카메라 쪽 = z<0, 위 = y<0)
LIGHT2_DIR = unit([0.6, -0.3, -0.74])
AMB, KEY = 0.25, 0.75


def dome_normals(h=24, w=32, depth_amp=18.0, sigma_frac=0.22):
    """카메라 쪽으로 솟은 가우시안 범프의 법선(카메라 좌표, 카메라를 향함, 단위). 기울기 최대 약 55도."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    s = sigma_frac * w
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
    z = 50.0 - depth_amp * g                        # z-깊이(작을수록 가까움)
    zy, zx = np.gradient(z)
    n = np.stack([zx, zy, -np.ones_like(z)], -1)    # 평면이면 (0,0,-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def geo_shading(normal, direction=LIGHT_DIR, amb=AMB, key=KEY):
    return amb + key * np.maximum(normal @ np.asarray(direction), 0.0)


def two_patch_albedo(h=24, w=32, a1=(0.70, 0.40, 0.20), a2=(0.20, 0.50, 0.70)):
    a = np.empty((h, w, 3))
    a[:, : w // 2] = a1
    a[:, w // 2:] = a2
    return a


def gray_patch_albedo(h=24, w=32, lo=0.30, hi=0.60):
    a = np.empty((h, w, 3))
    a[:, : w // 2] = lo
    a[:, w // 2:] = hi
    return a


def scene(h=24, w=32, albedo=None, direction=LIGHT_DIR):
    n = dome_normals(h, w)
    a = two_patch_albedo(h, w) if albedo is None else albedo
    s = geo_shading(n, direction)
    return {"normal": n, "albedo": a, "shading": s, "I": a * s[..., None], "light": direction}


def smooth_field(h, w, seed, amp=1.0):
    """저주파 매끈한 임의 장(평균 0 근처)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros((h, w))
    for _ in range(3):
        fx, fy = rng.uniform(0.5, 1.5) * 2 * math.pi / w, rng.uniform(0.5, 1.5) * 2 * math.pi / h
        out += rng.uniform(-1, 1) * np.sin(fx * xx + fy * yy + rng.uniform(0, 6.28))
    return amp * out / 3.0


def angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


# ------------------------------------------------------------------------------------------------ 에너지 오라클
def energy(s, I, *, normal=None, geo=None, geo_weight=None, shading_head=None, albedo_head=None, exclude=None, opts):
    """계약서 에너지 E(s) (s: (H,W) ln 음영). opts: alpha,beta,gamma,eta,delta,tau_chroma,tau_normal 속성."""
    I = np.asarray(I, np.float64)
    h, w = I.shape[:2]
    valid = np.ones((h, w), bool) if exclude is None else ~exclude
    Y = np.maximum(I @ W709, Y_FLOOR)
    l = np.log(Y)
    csum = np.maximum(I.sum(-1), 1e-6)
    chrom = I[..., :2] / csum[..., None]
    tot = 0.0
    for dy, dx in ((0, 1), (1, 0)):
        sl_a = (slice(0, h - dy), slice(0, w - dx))
        sl_b = (slice(dy, h), slice(dx, w))
        ok = valid[sl_a] & valid[sl_b]
        om = np.exp(-((np.linalg.norm(chrom[sl_a] - chrom[sl_b], axis=-1) / opts.tau_chroma) ** 2))
        if normal is None:
            bn = np.ones_like(om)
        else:
            bn = np.exp(-((np.linalg.norm(normal[sl_a] - normal[sl_b], axis=-1) / opts.tau_normal) ** 2))
        ds = s[sl_a] - s[sl_b]
        dl = l[sl_a] - l[sl_b]
        tot += np.sum(ok * (opts.alpha * om * (dl - ds) ** 2 + opts.beta * bn * ds ** 2))
    if geo is not None:
        gw = np.ones((h, w)) if geo_weight is None else geo_weight
        if normal is not None and not np.isscalar(geo):
            pass
        tot += np.sum(valid * opts.gamma * gw * (s - np.log(np.maximum(geo, S_MIN))) ** 2)
    if shading_head is not None:
        tot += np.sum(valid * opts.eta * (s - np.log(np.maximum(shading_head, S_MIN))) ** 2)
    if albedo_head is not None:
        for c in range(3):
            tot += np.sum(valid * opts.delta * (np.log(np.maximum(I[..., c], Y_FLOOR)) - s
                                                - np.log(np.maximum(albedo_head[..., c], Y_FLOOR))) ** 2)
    return float(tot)


def grad_fd(f, s, h=1e-3):
    """중앙 차분 기울기(f 가 2차식이면 거의 정확)."""
    g = np.zeros(s.size)
    flat = s.reshape(-1)
    for i in range(s.size):
        p = flat.copy(); p[i] += h
        m = flat.copy(); m[i] -= h
        g[i] = (f(p.reshape(s.shape)) - f(m.reshape(s.shape))) / (2 * h)
    return g.reshape(s.shape)


def luminance(rgb):
    return np.asarray(rgb, np.float64) @ W709


def rel_err(a, b, floor=1e-12):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


# ------------------------------------------------------------------------------------------------ 조명 적합 오라클
def fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    ph = math.pi * (3 - math.sqrt(5)) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def light_objective(n, S, w, direction, amb, key):
    r = S - amb - key * np.maximum(n @ np.asarray(direction), 0.0)
    return float(np.sum(w * r * r))


def brute_light(n, S, w, ndirs=3000):
    """ω 를 구면 격자로 훑고 각 ω 마다 (amb,k>=0) 2변수 비음수 최소제곱(scipy.optimize.nnls)로 푼 최솟값."""
    from scipy.optimize import nnls
    sw = np.sqrt(w)
    best = (math.inf, None, 0.0, 0.0)
    for d in fib_sphere(ndirs):
        t = np.maximum(n @ d, 0.0)
        A = np.stack([np.ones_like(t), t], 1) * sw[:, None]
        x, _ = nnls(A, S * sw)
        o = light_objective(n, S, w, d, x[0], x[1])
        if o < best[0]:
            best = (o, d, float(x[0]), float(x[1]))
    return best
