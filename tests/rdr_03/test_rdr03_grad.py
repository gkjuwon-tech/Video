"""RDR-03 미분: edge_falloff·sample_page 수식과 기울기, render 중앙 차분(6개 파라미터, float64), 경계 유한성,
텍스처·기하 복원 피팅, 속도."""
import time

import numpy as np
import pytest
import torch

from buc.render.tsq import TSQOpts, edge_falloff, render, sample_page
from rdr03_helpers import K_of, Scene, make_tsq, oracle_bilinear, pose, random_scene, rot_axis, to_np

FIELDS = ("means", "quats", "log_scales", "opacity_logits", "pages", "edge_n")


def test_edge_falloff_values_and_boundary_grads():
    u = torch.tensor([0.0, 0.3, -0.7, 1.0, 1.5, -2.0], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([0.0, -0.2, 0.0, 1.0, 0.1, 0.5], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([2.0, 3.0, 8.0, 16.0, 5.5, 2.0], dtype=torch.float64, requires_grad=True)
    g = edge_falloff(u, v, n)
    un, vn, nn = to_np(u), to_np(v), to_np(n)
    np.testing.assert_allclose(to_np(g), np.exp(-0.5 * (np.abs(un) ** nn + np.abs(vn) ** nn)), rtol=1e-12)
    g.sum().backward()
    for x in (u, v, n):
        assert torch.isfinite(x.grad).all()
    assert float(u.grad[0]) == 0.0 and float(n.grad[0]) == 0.0         # u=v=0: 기울기 0
    # 해석 기울기: dg/du = -½·n·|u|^(n-1)·sign(u)·g
    k = 1
    exp_du = -0.5 * nn[k] * abs(un[k]) ** (nn[k] - 1) * np.sign(un[k]) * to_np(g)[k]
    np.testing.assert_allclose(float(u.grad[k]), exp_du, rtol=1e-10)
    exp_dn = -0.5 * (abs(un[k]) ** nn[k] * np.log(abs(un[k])) + abs(vn[k]) ** nn[k] * np.log(abs(vn[k]))) * to_np(g)[k]
    np.testing.assert_allclose(float(n.grad[k]), exp_dn, rtol=1e-10)
    # float32, 큰 n·큰 |u|: 오버플로해도 값 0, 기울기 유한
    u32 = torch.tensor([5.0, 30.0, 0.0], requires_grad=True)
    v32 = torch.tensor([0.0, 3.0, 40.0], requires_grad=True)
    n32 = torch.tensor([64.0, 64.0, 128.0], requires_grad=True)
    g32 = edge_falloff(u32, v32, n32)
    assert torch.all(g32 == 0.0)
    g32.sum().backward()
    for x in (u32, v32, n32):
        assert torch.isfinite(x.grad).all()


def test_sample_page_matches_oracle_and_weights():
    rng = np.random.default_rng(0)
    pages = torch.tensor(rng.uniform(size=(3, 8, 8, 4)), requires_grad=True)
    M = 200
    idx = torch.tensor(rng.integers(0, 3, M))
    uv = torch.tensor(rng.uniform(-1.4, 1.4, (M, 2)), requires_grad=True)
    out = sample_page(pages, idx, uv)
    assert out.shape == (M, 4)
    ref = np.stack([oracle_bilinear(to_np(pages)[i], a, b) for i, (a, b) in zip(idx.numpy(), to_np(uv))])
    np.testing.assert_allclose(to_np(out), ref, atol=1e-12)
    # 클램프: (1.5, -2) -> page[0, 7]; 텍셀 중심 -> 정확한 텍셀
    p0 = to_np(pages)[0]
    c = sample_page(pages, torch.tensor([0, 0]), torch.tensor([[1.5, -2.0], [-1 + 5 / 8, -1 + 3 / 8]],
                                                             dtype=torch.float64))
    np.testing.assert_allclose(to_np(c[0]), p0[0, 7], atol=1e-12)
    np.testing.assert_allclose(to_np(c[1]), p0[1, 2], atol=1e-12)
    # 페이지 기울기 = 바이리니어 가중치 (채널마다 합 1)
    out[:, 0].sum().backward()
    assert abs(float(pages.grad[..., 0].sum()) - M) < 1e-9
    assert float(pages.grad[..., 1:].abs().max()) == 0.0
    # uv 기울기: 중앙 차분
    uv2 = torch.tensor([[0.13, -0.41], [0.77, 0.52], [-0.9, 0.05]], dtype=torch.float64, requires_grad=True)
    pg = pages.detach()
    ii = torch.tensor([0, 1, 2])
    sample_page(pg, ii, uv2).sum().backward()
    eps = 1e-6
    for r in range(3):
        for c_ in range(2):
            d = torch.zeros_like(uv2)
            d[r, c_] = eps
            fd = (sample_page(pg, ii, uv2.detach() + d).sum() - sample_page(pg, ii, uv2.detach() - d).sum()) / (2 * eps)
            np.testing.assert_allclose(float(uv2.grad[r, c_]), float(fd), rtol=1e-6, atol=1e-8)
    # 클램프 영역 밖에서 uv 기울기 0
    uv3 = torch.tensor([[1.7, 1.8]], dtype=torch.float64, requires_grad=True)
    sample_page(pg, torch.tensor([0]), uv3).sum().backward()
    assert float(uv3.grad.abs().max()) == 0.0


def _loss(t, wr, wa, c2w, K, Wd, Hd):
    r = render(t, c2w=c2w, K=K, width=Wd, height=Hd, opts=TSQOpts(bg=(0.2, 0.3, 0.1)))
    return (r.rgb * wr).sum() + (r.alpha * wa).sum()


def test_render_gradients_match_central_differences():
    Wd, Hd = 16, 12
    K = K_of(14.0, Wd, Hd)
    c2w = pose(rot_axis([0.1, 1, 0.2], 6), [0.02, 0.01, -0.03])
    sc = random_scene(3, 31, edge_lo=2.5, edge_hi=9.0, scale=(0.25, 0.45), opac=(0.4, 0.85), spread=0.3,
                      depth=(2.2, 3.2), tilt=40.0)
    rng = np.random.default_rng(5)
    wr = torch.tensor(rng.normal(size=(Hd, Wd, 3)))
    wa = torch.tensor(rng.normal(size=(Hd, Wd)))
    t = make_tsq(sc).requires_grad_(True)
    _loss(t, wr, wa, c2w, K, Wd, Hd).backward()
    grads = {f: getattr(t, f).grad.clone() for f in FIELDS}
    base = make_tsq(sc)
    eps = 1e-6
    for f in FIELDS:
        g = grads[f]
        assert torch.isfinite(g).all(), f
        flat = g.reshape(-1)
        picks = rng.choice(flat.numel(), size=min(6, flat.numel()), replace=False)
        if f == "pages":   # 기울기가 실제로 흐르는 텍셀도 포함
            picks = np.r_[picks, np.argsort(-flat.abs().numpy())[:3]]
        for k in picks:
            vals = []
            for sgn in (1.0, -1.0):
                b = base.clone()
                getattr(b, f).reshape(-1)[k] += sgn * eps
                with torch.no_grad():
                    vals.append(float(_loss(b, wr, wa, c2w, K, Wd, Hd)))
            fd = (vals[0] - vals[1]) / (2 * eps)
            assert abs(float(flat[k]) - fd) <= 1e-5 + 1e-3 * abs(fd), (f, int(k), float(flat[k]), fd)
        assert float(g.abs().max()) > 1e-4, f    # 기울기가 실제로 흐른다


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_finite_at_center_pixel_and_large_n(dtype):
    """중심이 화소 중심에 정확히 놓여 u=v=0 인 화소, n=64 (float32 |u|^n 오버플로) 에서도 기울기 유한."""
    K = K_of(30.0, 24, 20, cx=12.0, cy=10.0)
    sc = Scene([[0.0, 0.0, 3.0], [0.05, 0.02, 3.5]], np.stack([np.eye(3), rot_axis([1, 1, 0], 30)]),
               [[0.3, 0.3], [0.5, 0.4]], [0.9, 0.7], np.random.default_rng(1).uniform(size=(2, 8, 8, 3)),
               [2.0, 64.0])
    t = make_tsq(sc, dtype=dtype).requires_grad_(True)
    r = render(t, c2w=np.eye(4), K=K, width=24, height=20)
    (r.rgb.sum() + r.alpha.sum() + r.normal.sum()).backward()
    for f in FIELDS:
        assert torch.isfinite(getattr(t, f).grad).all(), f


def test_texture_recovery_fit():
    """페이지만 학습: 정면 하드 쿼드의 8×8 페이지를 렌더 손실로 복원 (200회, RMSE < 0.02)."""
    Kc = K_of(48.0, 40, 40, cx=19.5, cy=19.5)
    z, s = 3.0, 0.75                       # σ_px = 12
    rng = np.random.default_rng(3)
    gt_pages = rng.uniform(0.1, 0.9, (1, 8, 8, 3))
    sc = Scene([[0.0, 0.0, z]], np.eye(3)[None], [[s, s]], [0.99], gt_pages, [12.0])
    target = render(make_tsq(sc, torch.float32), c2w=np.eye(4), K=Kc, width=40, height=40).rgb.detach()
    sc0 = Scene(sc.means, sc.rots, sc.scales, sc.opac, np.full((1, 8, 8, 3), 0.5), sc.edge_n)
    t = make_tsq(sc0, torch.float32)
    t.pages.requires_grad_(True)
    opt = torch.optim.Adam([t.pages], lr=0.03)
    for _ in range(200):
        opt.zero_grad()
        loss = ((render(t, c2w=np.eye(4), K=Kc, width=40, height=40).rgb - target) ** 2).mean()
        loss.backward()
        opt.step()
    rmse = float(np.sqrt(((to_np(t.pages) - gt_pages) ** 2).mean()))
    assert rmse < 0.02, rmse


def test_geometry_recovery_fit():
    """위치·크기만 학습: 3 cm 어긋난 TSQ(n=6)를 목표 렌더로 되돌린다 (150회, 위치 오차 < 4 mm, 크기 2%)."""
    Kc = K_of(40.0, 40, 32)
    rng = np.random.default_rng(4)
    pages = rng.uniform(0.2, 0.9, (1, 8, 8, 3))
    R = rot_axis([0.3, 1, 0], 20)
    gt = Scene([[0.02, -0.01, 3.0]], R[None], [[0.45, 0.35]], [0.9], pages, [6.0])
    target = render(make_tsq(gt, torch.float64), c2w=np.eye(4), K=Kc, width=40, height=32)
    init = Scene([[0.05, 0.01, 3.0]], R[None], [[0.5, 0.32]], [0.9], pages, [6.0])
    t = make_tsq(init, torch.float64)
    t.means.requires_grad_(True)
    t.log_scales.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [t.means], "lr": 2e-3}, {"params": [t.log_scales], "lr": 5e-3}])
    for _ in range(150):
        opt.zero_grad()
        r = render(t, c2w=np.eye(4), K=Kc, width=40, height=32)
        loss = ((r.rgb - target.rgb) ** 2).mean() + ((r.alpha - target.alpha) ** 2).mean()
        loss.backward()
        opt.step()
    err_xy = np.abs(to_np(t.means)[0, :2] - gt.means[0, :2]).max()
    assert err_xy < 4e-3, err_xy
    np.testing.assert_allclose(np.exp(to_np(t.log_scales)[0]), gt.scales[0], rtol=0.02)


def test_speed_forward_backward_cpu():
    Wd, Hd = 64, 48
    K = K_of(50.0, Wd, Hd)
    sc = random_scene(300, 9, scale=(0.03, 0.12), spread=1.0, depth=(2.5, 5.0))
    t = make_tsq(sc, torch.float32).requires_grad_(True)
    t0 = time.perf_counter()
    r = render(t, c2w=np.eye(4), K=K, width=Wd, height=Hd)
    (r.rgb.sum() + r.alpha.sum()).backward()
    dt = time.perf_counter() - t0
    assert dt < 8.0, dt
    assert torch.isfinite(t.pages.grad).all() and float(t.pages.grad.abs().sum()) > 0
