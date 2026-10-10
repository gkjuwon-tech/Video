"""RDR-01 fit.py: COLMAP points3D txt/bin 리더, 점 초기화(스케일·노멀·색·불투명도), rgbd 초기화 항등 재현,
피팅 수렴·결정론·입력 불변·가지치기·정규화, COLMAP -> 초기화 -> 피팅 왕복."""
import math
import struct

import numpy as np
import pytest
import torch

from buc.core import color, geom
from buc.render.fit import (ColmapPoints, FitOpts, FitResult, FitView, fit, init_from_points, init_from_rgbd, psnr,
                            read_colmap_points3d)
from buc.render.splat2d import Surfels, render
from rdr01_helpers import K_of, make_surfels, pose, rot_axis, to_np

PTS = [  # id, xyz, rgb, err, track
    (7, (1.0, 2.0, 3.0), (255, 0, 10), 0.5, [(1, 2), (3, 4)]),
    (2, (-0.5, 0.25, 4.125), (12, 34, 56), 1.25, [(2, 9)]),
    (5, (0.0, -1.5, 2.0), (200, 100, 0), 0.0, [(1, 0), (2, 1), (4, 7)]),
]


def _write_txt(p):
    lines = ["# 3D point list with one line of data per point:",
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
             "# Number of points: 3, mean track length: 2", ""]
    for pid, xyz, rgb, err, tr in PTS:
        lines.append(" ".join(map(str, [pid, *xyz, *rgb, err] + [x for pair in tr for x in pair])))
    p.write_text("\n".join(lines) + "\n")


def _write_bin(p):
    b = struct.pack("<Q", len(PTS))
    for pid, xyz, rgb, err, tr in PTS:
        b += struct.pack("<Q3d3BdQ", pid, *xyz, *rgb, err, len(tr))
        for im, idx in tr:
            b += struct.pack("<ii", im, idx)
    p.write_bytes(b)


def _check_pts(cp):
    assert isinstance(cp, ColmapPoints)
    assert cp.ids.dtype == np.int64 and list(cp.ids) == [2, 5, 7]
    assert cp.xyz.dtype == np.float64 and cp.xyz.shape == (3, 3)
    assert np.array_equal(cp.xyz, [[-0.5, 0.25, 4.125], [0.0, -1.5, 2.0], [1.0, 2.0, 3.0]])
    assert cp.rgb.dtype == np.uint8 and np.array_equal(cp.rgb, [[12, 34, 56], [200, 100, 0], [255, 0, 10]])
    assert np.array_equal(cp.error, [1.25, 0.0, 0.5])
    assert cp.track_len.dtype == np.int64 and list(cp.track_len) == [1, 3, 2]


def test_colmap_txt_bin_equivalent(tmp_path):
    t, b = tmp_path / "points3D.txt", tmp_path / "points3D.bin"
    _write_txt(t)
    _write_bin(b)
    _check_pts(read_colmap_points3d(t))
    _check_pts(read_colmap_points3d(str(b)))
    _check_pts(read_colmap_points3d(tmp_path))          # 디렉터리: .bin 우선
    (tmp_path / "points3D.bin").unlink()
    _check_pts(read_colmap_points3d(tmp_path))


def test_colmap_errors_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_colmap_points3d(tmp_path / "nope.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("1 0.0 1.0 2.0 255 255\n")
    with pytest.raises(ValueError):
        read_colmap_points3d(bad)
    trunc = tmp_path / "points3D.bin"
    _write_bin(trunc)
    trunc.write_bytes(trunc.read_bytes()[:-5])
    with pytest.raises(ValueError):
        read_colmap_points3d(trunc)
    empty = tmp_path / "e" / "points3D.txt"
    empty.parent.mkdir()
    empty.write_text("# Number of points: 0\n")
    cp = read_colmap_points3d(empty)
    assert cp.xyz.shape == (0, 3) and cp.rgb.shape == (0, 3) and cp.ids.shape == (0,)


def _plane_grid(nx=7, ny=6, d=0.05):
    n0 = np.array([0.3, 0.2, 1.0])
    n0 /= np.linalg.norm(n0)
    t1 = np.cross(n0, [1.0, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n0, t1)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.array([0.0, 0.0, 3.0]) + (i.reshape(-1, 1) * d) * t1 + (j.reshape(-1, 1) * d) * t2
    return P, n0, (i.reshape(-1), j.reshape(-1))


def test_init_from_points_scales_normals_colors():
    P, n0, (i, j) = _plane_grid()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (len(P), 3)).astype(np.uint8)
    s = init_from_points(P, rgb, k_nn=3, scale_mult=0.8, opacity=0.1, cam_centers=[[0.0, 0.0, 0.0]])
    assert isinstance(s, Surfels) and s.n == len(P) and s.means.dtype == torch.float32
    assert torch.equal(s.means, torch.tensor(P, dtype=torch.float32))
    sc = to_np(s.scales())
    corner = ((i == 0) | (i == 6)) & ((j == 0) | (j == 5))
    assert np.abs(sc[~corner] - 0.8 * 0.05).max() < 1e-6
    assert np.abs(sc[corner] - 0.8 * 0.05 * (2 + math.sqrt(2)) / 3).max() < 1e-6
    n = to_np(s.normals())
    assert (np.abs(n @ n0) > 0.999).all()
    assert ((n * (0 - P)).sum(1) >= 0).all()           # 카메라(원점) 쪽
    R = to_np(s.rotations())
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5)
    assert np.allclose(to_np(s.opacities()), 0.1, atol=1e-6)
    assert np.allclose(to_np(s.colors), color.srgb8_to_linear(rgb), atol=1e-7)
    # float rgb 는 선형 그대로, None 은 0.5
    lin = rng.uniform(0, 1, (len(P), 3))
    assert np.allclose(to_np(init_from_points(P, lin).colors), lin, atol=1e-7)
    assert np.allclose(to_np(init_from_points(P).colors), 0.5)
    s64 = init_from_points(P, dtype=torch.float64)
    assert s64.means.dtype == torch.float64
    # 결정론
    s2 = init_from_points(P, rgb, k_nn=3, scale_mult=0.8, opacity=0.1, cam_centers=[[0.0, 0.0, 0.0]])
    for a, b in zip(s.params(), s2.params()):
        assert torch.equal(a, b)


def test_init_from_points_edge_cases():
    P, _, _ = _plane_grid(4, 4)
    dup = np.vstack([P, P[:3]])
    s = init_from_points(dup, min_scale=1e-4)
    sc = to_np(s.scales())
    assert np.isfinite(sc).all() and (sc >= 1e-4 * (1 - 1e-6)).all()
    assert all(bool(torch.isfinite(p).all()) for p in s.params())
    s2 = init_from_points(P[:2])
    assert s2.n == 2 and bool(torch.isfinite(s2.log_scales).all())
    with pytest.raises(ValueError):
        init_from_points(P[:1])
    bad = P.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        init_from_points(bad)
    with pytest.raises(ValueError):
        init_from_points(P[:, :2])


def test_init_from_rgbd_identity_reproduces():
    W, H = 48, 40
    K = K_of(40.0, W, H)
    c2w = pose(rot_axis([0.1, 1.0, 0.0], 12), [0.2, -0.1, 0.3])
    img = np.empty((H, W, 3))
    img[:, :24] = [0.2, 0.3, 0.4]
    img[:, 24:] = [0.8, 0.6, 0.1]
    depth = np.full((H, W), 2.5)
    depth[:, 30:] = 3.5
    s = init_from_rgbd(img, depth, K=K, c2w=c2w)
    assert s.n == H * W
    # 중심 = geom.unproject_world, 접선 = 카메라 축
    w2c = geom.invert_pose(c2w)
    X = geom.unproject_world(K, w2c[:3, :3], w2c[:3, 3], np.array([[5.0, 7.0]]), np.array([2.5]))
    assert np.abs(to_np(s.means[7 * W + 5]) - X[0]).max() < 1e-5
    assert np.abs(to_np(s.rotations()[0]) - c2w[:3, :3]).max() < 1e-5
    assert np.allclose(to_np(s.scales()[0]), 0.5 * 2.5 / 40.0, rtol=1e-5)
    r = render(s, c2w=c2w, K=K, width=W, height=H)
    a, rgb, d = to_np(r.alpha), to_np(r.rgb), to_np(r.depth)
    inner = (slice(2, H - 2), slice(2, W - 2))
    assert a[inner].min() >= 0.98
    assert np.abs(rgb[inner] - img[inner]).mean() <= 0.02
    m = a > 0.5
    assert np.abs(d[m] / depth[m] - 1).mean() < 0.01
    flat = (slice(2, H - 2), slice(2, 27))
    assert np.abs(d[flat] / 2.5 - 1).max() < 1e-3
    # stride·alpha·무효 깊이
    al = np.ones((H, W))
    al[:10] = 0.0
    dep = depth.copy()
    dep[20, :] = np.inf
    dep[21, :] = -1.0
    s2 = init_from_rgbd(img, dep, K=K, c2w=c2w, stride=2, alpha=al)
    rows = [v for v in range(0, H, 2) if v >= 10 and v not in (20, 21)]
    assert s2.n == len(rows) * len(range(0, W, 2))
    assert np.allclose(to_np(s2.scales()[0]), 0.5 * 2 * 2.5 / 40.0, rtol=1e-5)


# ---------- 피팅 ----------
FW, FH = 32, 24
FK = K_of(30.0, FW, FH)
CAMS = [pose(None, [0, 0, 0]), pose(rot_axis([0, 1, 0], 4), [0.15, 0, 0]),
        pose(rot_axis([1, 0, 0], -3), [-0.1, 0.08, 0.05])]
HOLD = pose(rot_axis([0, 1, 0], -2), [-0.06, -0.04, 0.0])


def _gt():
    rng = np.random.default_rng(5)
    n = 30
    means = np.c_[rng.uniform(-0.8, 0.8, n), rng.uniform(-0.55, 0.55, n), rng.uniform(2.8, 3.2, n)]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, 25)) for _ in range(n)])
    return make_surfels(means, rots, rng.uniform(0.12, 0.2, (n, 2)), np.full(n, 0.9),
                        rng.uniform(0.05, 0.95, (n, 3)))


def _views(gt, cams=CAMS):
    out = []
    for c in cams:
        img = render(gt, c2w=c, K=FK, width=FW, height=FH).rgb.detach().numpy()
        out.append(FitView(image=img, c2w=c, K=FK))
    return out


def _perturbed(gt, seed=1):
    g = torch.Generator().manual_seed(seed)
    s = gt.clone()
    s.means.add_(0.02 * torch.randn(s.means.shape, generator=g))
    s.colors.fill_(0.5)
    s.opacity_logits.fill_(0.0)
    s.log_scales.add_(0.2)
    return s


OPTS = dict(lr_means=1e-3, lr_quats=5e-3, lr_scales=1e-2, lr_opacity=5e-2, lr_colors=2e-2)


def test_psnr():
    a = np.zeros((4, 4, 3))
    assert psnr(a, a) == math.inf
    assert abs(psnr(a, a + 0.1) - 20.0) < 1e-9
    m = np.zeros((4, 4), bool)
    m[0, 0] = True
    b = a.copy()
    b[1:] = 0.5
    assert psnr(a, b, mask=m) == math.inf


def test_fit_converges_and_keeps_input():
    gt = _gt()
    views = _views(gt)
    init = _perturbed(gt)
    before = [p.clone() for p in init.params()]
    res = fit(init, views, FitOpts(iters=150, **OPTS))
    assert isinstance(res, FitResult) and len(res.losses) == 150
    assert all(math.isfinite(x) for x in res.losses)
    assert res.losses[-1] <= 0.1 * res.losses[0], (res.losses[0], res.losses[-1])
    assert len(res.psnr) == 3 and min(res.psnr) >= 35.0, res.psnr
    for p in res.surfels.params():
        assert not p.requires_grad
    for p, q in zip(before, init.params()):
        assert torch.equal(p, q)
    # 보고된 psnr 은 결과 surfel 로 다시 렌더한 값
    r = render(res.surfels, c2w=CAMS[1], K=FK, width=FW, height=FH)
    assert abs(psnr(r.rgb.numpy(), views[1].image) - res.psnr[1]) < 1e-3


def test_fit_deterministic():
    gt = _gt()
    views = _views(gt)
    a = fit(_perturbed(gt), views, FitOpts(iters=15, seed=3, **OPTS))
    b = fit(_perturbed(gt), views, FitOpts(iters=15, seed=3, **OPTS))
    assert a.losses == b.losses
    for p, q in zip(a.surfels.params(), b.surfels.params()):
        assert p.numpy().tobytes() == q.numpy().tobytes()


def test_fit_prune_and_regularizers():
    gt = _gt()
    views = _views(gt)
    ghosts = make_surfels(np.c_[np.zeros(5), np.zeros(5), -np.arange(1, 6.0)], np.repeat(np.eye(3)[None], 5, 0),
                          np.full((5, 2), 0.1), np.full(5, 0.01), np.full((5, 3), 0.5))
    init = Surfels.cat([_perturbed(gt), ghosts])
    res = fit(init, views, FitOpts(iters=40, prune_opacity=0.05, lambda_dist=100.0, lambda_normal=0.05, **OPTS))
    assert all(math.isfinite(x) for x in res.losses)
    assert res.surfels.n <= init.n - 5
    assert float(res.surfels.opacities().min()) >= 0.05
    assert res.losses[-1] < res.losses[0]


def test_fit_batch_views_and_mask():
    gt = _gt()
    views = _views(gt)
    mask = np.ones((FH, FW), bool)
    mask[:, :8] = False
    mv = [FitView(image=v.image, c2w=v.c2w, K=v.K, mask=mask) for v in views]
    res = fit(_perturbed(gt), mv, FitOpts(iters=30, batch_views=1, **OPTS))
    assert len(res.losses) == 30 and res.losses[-1] < res.losses[0]


def test_colmap_to_fit_roundtrip(tmp_path):
    """GT 장면 -> 표면 위 점 + 잡음을 points3D.txt 로 -> 읽기 -> 점 초기화 -> 피팅 -> 학습·홀드아웃 PSNR."""
    gt = _gt()
    rng = np.random.default_rng(9)
    R = to_np(gt.rotations())
    sc = to_np(gt.scales())
    mu = to_np(gt.means)
    lin = to_np(gt.colors)
    pts, cols = [], []
    for k in range(gt.n):
        for _ in range(6):
            uv = rng.normal(size=2) * 0.7
            pts.append(mu[k] + R[k][:, 0] * sc[k, 0] * uv[0] + R[k][:, 1] * sc[k, 1] * uv[1] + rng.normal(size=3) * 0.003)
            cols.append(color.linear_to_srgb8(lin[k]))
    p = tmp_path / "points3D.txt"
    p.write_text("\n".join(f"{i + 1} {float(x[0])!r} {float(x[1])!r} {float(x[2])!r} {c[0]} {c[1]} {c[2]} 0.5 1 {i}"
                           for i, (x, c) in enumerate(zip(pts, cols))) + "\n")
    cp = read_colmap_points3d(p)
    assert cp.xyz.shape == (gt.n * 6, 3)
    init = init_from_points(cp.xyz, cp.rgb, opacity=0.5, cam_centers=[c[:3, 3] for c in CAMS])
    views = _views(gt)
    res = fit(init, views, FitOpts(iters=150, **OPTS))
    assert min(res.psnr) >= 32.0, res.psnr
    hold = render(gt, c2w=HOLD, K=FK, width=FW, height=FH).rgb.numpy()
    got = render(res.surfels, c2w=HOLD, K=FK, width=FW, height=FH).rgb.numpy()
    assert psnr(got, hold) >= 28.0
    init_img = render(init, c2w=HOLD, K=FK, width=FW, height=FH).rgb.numpy()
    assert psnr(got, hold) > psnr(init_img, hold) + 6.0
