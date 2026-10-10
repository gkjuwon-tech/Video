"""RDR-07 재귀: 거울 복도 광선 오라클(bounce, ∏R·bg / ∏R·terminal), max_depth·min_weight·예산 등가, 초록 터널, 속도."""
import functools
import time

import numpy as np
import pytest

from buc.render.mirror import Mirror, MirrorOpts, render_mirrors
from buc.render.quads import RenderOpts
from rdr07_helpers import (H, I4, W, K_default, const_rgba, layer, make_obs, make_world, pixel_rays, rect_hit,
                           smooth_tex, translate, yaw)

RA = (0.80, 0.90, 0.85)
RB = (0.85, 0.90, 0.80)
A = Mirror(center=(0, 0, 3.0), normal=(0, 0, -1), u_axis=(1, 0, 0), half_size=(1.2, 0.9), reflectance=RA)
B = Mirror(center=(0, 0, -1.0), normal=(0, 0, 1), u_axis=(1, 0, 0), half_size=(1.0, 0.8), reflectance=RB)
CAM = translate(yaw(I4, 8.0), (0.1, -0.05, 0.0))
BG = (1.0, 1.0, 1.0)
TERM = (0.2, 0.1, 0.05)


def _empty_world():
    return make_world([layer(const_rgba((0.5, 0.5, 0.5), 0.0), 1.0)])     # 알파 0: 모든 것이 배경


@functools.lru_cache(maxsize=None)
def _render(max_depth, min_weight=0.0, max_renders=64):
    opts = MirrorOpts(max_depth=max_depth, min_weight=min_weight, max_renders=max_renders, terminal_rgb=TERM,
                      quad=RenderOpts(bg=BG))
    return render_mirrors(_empty_world(), [A, B], c2w=CAM, K=K_default(), width=W, height=H, opts=opts)


@functools.lru_cache(maxsize=None)
def _oracle(max_depth):
    """화소 중심 광선을 거울 사이에서 직접 튕긴다. 반환 (bounce, rgb, first_id, first_z)."""
    eye, dw, dc = pixel_rays(CAM, K_default(), W, H)
    n = H * W
    o = np.broadcast_to(eye, (n, 3)).copy()
    d = dw.reshape(-1, 3).copy()
    ms = [A, B]
    alive = np.ones(n, bool)
    last = np.full(n, -1)
    bounce = np.zeros(n, int)
    w = np.ones((n, 3))
    term = np.zeros(n, bool)
    first_id = np.full(n, -1)
    first_z = np.full(n, np.inf)
    for k in range(max_depth + 1):
        ts = []
        for i, m in enumerate(ms):
            t = rect_hit(o, d, m.center, m.normal, m.u_axis, m.half_size[0], m.half_size[1])
            if k == 0:
                t = np.where(t >= 1e-3, t, np.inf)                         # 1차 z-깊이 near
            t = np.where(last == i, np.inf, t)
            ts.append(t)
        ts = np.stack(ts, 1)
        idx = np.argmin(ts, 1)
        tmin = ts[np.arange(n), idx]
        hit = alive & np.isfinite(tmin)
        if k == 0:
            first_id[hit] = idx[hit]
            first_z[hit] = tmin[hit]
        R = np.array([RA, RB])[idx]
        w[hit] *= R[hit]
        bounce[hit] += 1
        if k == max_depth:
            term |= hit
        P = o + tmin[:, None] * d
        nn = np.array([m.normal for m in ms])[idx]
        dr = d - 2 * (d * nn).sum(1, keepdims=True) * nn
        o = np.where(hit[:, None], P, o)
        d = np.where(hit[:, None], dr, d)
        last = np.where(hit, idx, last)
        alive &= hit
    rgb = w * np.where(term[:, None], np.asarray(TERM), np.asarray(BG))
    return bounce.reshape(H, W), rgb.reshape(H, W, 3), first_id.reshape(H, W), first_z.reshape(H, W)


@pytest.mark.parametrize("D", [1, 3, 12])
def test_corridor_matches_ray_bounce_oracle(D):
    r = _render(D)
    ob, orgb, ofid, oz = _oracle(D)
    agree = (r.bounce == ob) & (r.mirror_id == ofid)
    assert agree.mean() >= 0.99, agree.mean()
    np.testing.assert_allclose(r.rgb[agree], orgb[agree].astype(np.float32), atol=1e-5)
    assert r.bounce.max() == D + 1                                          # 종단 거울까지 센다
    assert len(np.unique(ob)) >= min(D + 2, 5)                              # 오라클이 실제로 여러 단계를 만든다
    m = r.mirror_id >= 0
    np.testing.assert_allclose(r.depth[m & agree], oz[m & agree], rtol=1e-5)
    assert (r.alpha[m] == 1.0).all() and (r.layer_id[m] == -1).all()
    assert np.isinf(r.virtual_depth[m]).all()                               # 모두 배경/종단으로 끝난다
    assert r.n_renders == D + 1                                             # 사슬: 단계마다 자식 하나


def test_green_tunnel_tint_accumulates_per_bounce():
    r = _render(12)
    ratios = []
    for k in range(1, 8):
        sel = (r.bounce == k)
        if sel.sum() < 5:
            continue
        g, red = r.rgb[..., 1][sel].mean(), r.rgb[..., 0][sel].mean()
        ratios.append((k, g / red))
    assert len(ratios) >= 4
    ks, vals = zip(*ratios)
    assert all(b > a for a, b in zip(vals, vals[1:])), ratios               # 깊을수록 초록
    # ∏R 은 A,B 가 번갈아 -> 짝수 k 에서 G/R = (0.9/0.8 * 0.9/0.85)^(k/2)
    for k, v in ratios:
        if k % 2 == 0:
            assert v == pytest.approx((0.9 / 0.8 * 0.9 / 0.85) ** (k // 2), rel=1e-4)


def test_min_weight_is_equivalent_to_depth_cut():
    """가중 w = ∏max(R) = 0.9^k. min_weight=0.5 -> 반사 6회까지 따라감 = max_depth 6."""
    a = _render(12, min_weight=0.5)
    b = _render(6)
    for k in ("rgb", "bounce", "mirror_id", "depth"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
    assert a.n_renders == b.n_renders == 7


def test_render_budget_is_equivalent_to_depth_cut():
    a = _render(12, max_renders=4)
    b = _render(3)
    for k in ("rgb", "bounce", "mirror_id", "depth"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
    assert a.n_renders == 4


def test_depth12_corridor_with_texture_is_fast():
    tex = smooth_tex()
    tex[16:, :, 3] = 0.0                                                    # 위쪽 띠만 남겨 복도 중심을 비운다
    w = make_world([layer(tex, 1.5)])
    opts = MirrorOpts(max_depth=12, quad=RenderOpts(bg=BG))
    t0 = time.perf_counter()
    r = render_mirrors(w, [A, B], c2w=CAM, K=K_default(), width=W, height=H, opts=opts)
    dt = time.perf_counter() - t0
    assert dt < 10.0, dt
    assert r.bounce.max() >= 6
    assert np.isfinite(r.virtual_depth[r.bounce >= 2]).any()                 # 반사 속에 텍스처 카드의 허상
    K2 = K_default(192, 128)
    t0 = time.perf_counter()
    render_mirrors(_empty_world_sized(192, 128), [A, B], c2w=CAM, K=K2, width=192, height=128, opts=opts)
    assert time.perf_counter() - t0 < 15.0


def _empty_world_sized(w, h):
    return make_world([layer(const_rgba((0.5, 0.5, 0.5), 0.0, w, h), 1.0)], make_obs(w=w, h=h))
