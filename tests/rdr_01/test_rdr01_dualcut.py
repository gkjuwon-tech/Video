"""RDR-01 dual-cut LOD: lod_lambda 공식, 컷 블렌드·스냅·클램프, λ-연속(정수 λ 양쪽 동일), 16×16 타일 평균,
우세 컷 depth, 통계(n_tiles_double, cuts_rendered), 입력 검증."""
import math

import numpy as np
import pytest
import torch

from buc.render.splat2d import DualCutRender, SplatOpts, lod_lambda, render, render_dual_cut
from rdr01_helpers import K_of, make_surfels, rot_axis

W, H = 64, 40            # 타일 4 x 3 (마지막 행 타일은 8 px 높이)
K = K_of(40.0, W, H)
I4 = np.eye(4)
OPTS = SplatOpts(bg=(0.05, 0.05, 0.05))


def _cut(level):
    """레벨마다 색·깊이·개수가 다른 완전한 세계(거친 -> 고운)."""
    rng = np.random.default_rng(100 + level)
    n = 6 * (level + 1)
    means = np.c_[rng.uniform(-1.0, 1.0, n), rng.uniform(-0.6, 0.6, n), rng.uniform(2.0, 3.0, n) + 0.5 * level]
    rots = np.stack([rot_axis(rng.normal(size=3), rng.uniform(0, 30)) for _ in range(n)])
    col = np.clip(rng.uniform(0, 1, (n, 3)) * 0.5 + 0.25 * level, 0, 1)
    return make_surfels(means, rots, rng.uniform(0.25, 0.5, (n, 2)) / (level + 1) ** 0.5, rng.uniform(0.6, 0.9, n),
                        col)


CUTS = [_cut(0), _cut(1), _cut(2)]


def _ref(level):
    return render(CUTS[level], c2w=I4, K=K, width=W, height=H, opts=OPTS)


def _dc(lam, cuts=CUTS, **kw):
    return render_dual_cut(cuts, lam, c2w=I4, K=K, width=W, height=H, opts=OPTS, **kw)


def _close(a, b, atol=1e-6):
    return bool(torch.allclose(a, b, atol=atol))


def test_lod_lambda_formula():
    d = np.array([[2.0, 4.0], [1.0, np.inf]])
    lam = lod_lambda(d, K, h0=0.2)
    assert isinstance(lam, np.ndarray)
    assert abs(lam[0, 0] - 2.0) < 1e-12 and abs(lam[0, 1] - 1.0) < 1e-12 and abs(lam[1, 0] - 3.0) < 1e-12
    assert lam[1, 1] == -np.inf
    lam2 = lod_lambda(torch.tensor(d), K, h0=0.2, kappa=2.0)
    assert isinstance(lam2, torch.Tensor)
    assert abs(float(lam2[0, 0]) - 1.0) < 1e-12


def test_single_cut_equals_plain_render():
    r = _dc(3.7, cuts=CUTS[:1])
    ref = _ref(0)
    assert isinstance(r, DualCutRender)
    for k in ("rgb", "alpha", "normal"):
        assert _close(getattr(r, k), getattr(ref, k)), k
    assert torch.equal(torch.isinf(r.depth), torch.isinf(ref.depth))
    assert r.cuts_rendered == (0,) and r.n_tiles == 12 and r.n_tiles_double == 0
    assert torch.allclose(r.level, torch.zeros(H, W, dtype=r.level.dtype))


def test_half_blend():
    r = _dc(0.5)
    a, b = _ref(0), _ref(1)
    assert _close(r.rgb, 0.5 * a.rgb + 0.5 * b.rgb, 1e-5)
    assert _close(r.alpha, 0.5 * a.alpha + 0.5 * b.alpha, 1e-5)
    assert r.n_tiles == 12 and r.n_tiles_double == 12 and r.cuts_rendered == (0, 1)
    assert torch.allclose(r.level, torch.full((H, W), 0.5, dtype=r.level.dtype))
    r3 = _dc(1.25)
    b1, c2 = _ref(1), _ref(2)
    assert _close(r3.rgb, 0.75 * b1.rgb + 0.25 * c2.rgb, 1e-5)
    # depth 는 우세 컷(f<0.5 -> l_a)
    fin = torch.isfinite(b1.depth)
    assert torch.equal(torch.isfinite(r3.depth), fin)
    assert _close(r3.depth[fin], b1.depth[fin], 1e-5)
    r4 = _dc(1.75)
    fin = torch.isfinite(c2.depth)
    assert _close(r4.depth[fin], c2.depth[fin], 1e-5)


def test_snap_band_and_clamp():
    for lam, lev, cut, used in ((0.02, 0.0, 0, (0,)), (0.97, 1.0, 1, (1,)), (-3.0, 0.0, 0, (0,)),
                                (7.0, 2.0, 2, (2,)), (2.0, 2.0, 2, (2,)), (1.04, 1.0, 1, (1,))):
        r = _dc(lam)
        ref = _ref(cut)
        assert _close(r.rgb, ref.rgb), lam
        assert _close(r.alpha, ref.alpha), lam
        assert r.n_tiles_double == 0, lam
        assert r.cuts_rendered == used, (lam, r.cuts_rendered)
        assert torch.allclose(r.level, torch.full((H, W), lev, dtype=r.level.dtype)), lam
    # 띠 바로 안쪽은 블렌드
    r = _dc(0.06)
    a, b = _ref(0), _ref(1)
    assert _close(r.rgb, 0.94 * a.rgb + 0.06 * b.rgb, 1e-5) and r.n_tiles_double == 12


def test_lambda_continuity_at_integer():
    for k in (1.0, 2.0):
        lo, hi = _dc(k - 1e-4), _dc(k + 1e-4)
        assert _close(lo.rgb, hi.rgb, 1e-6) and _close(lo.alpha, hi.alpha, 1e-6)
    # 띠 안에서는 λ 에 대해 립시츠: |ΔI| <= |Δλ|·|I1-I0| (+오차)
    a, b = _ref(0), _ref(1)
    diff = (b.rgb - a.rgb).abs()
    r1, r2 = _dc(0.40), _dc(0.41)
    assert bool(((r2.rgb - r1.rgb).abs() <= 0.01 * diff + 1e-5).all())


def test_per_tile_lambda_map_and_mean_pooling():
    lam = np.zeros((H, W))
    lam[:, 32:] = 2.0                        # 오른쪽 2열 타일 -> 가장 고운 컷
    lam[0:16, 0:16] = 0.0
    lam[0:16, 0:16][::2, :] = 1.0            # 타일 (0,0): 0/1 반반 -> 평균 0.5 -> 블렌드
    lam[16:32, 16:32] = 0.98                 # 타일 (1,1): 스냅 -> 컷 1
    lam[33, 40] = np.nan
    with pytest.raises(ValueError):
        _dc(lam)
    lam[33, 40] = 2.0
    lam[32:40, 0:16] = -np.inf               # 유한 값 없는 타일 -> λ=0
    r = _dc(torch.tensor(lam))
    a, b, c = _ref(0), _ref(1), _ref(2)
    t00 = (slice(0, 16), slice(0, 16))
    assert _close(r.rgb[t00], 0.5 * a.rgb[t00] + 0.5 * b.rgb[t00], 1e-5)
    t11 = (slice(16, 32), slice(16, 32))
    assert _close(r.rgb[t11], b.rgb[t11])
    right = (slice(0, H), slice(32, W))
    assert _close(r.rgb[right], c.rgb[right])
    t20 = (slice(32, 40), slice(0, 16))
    assert _close(r.rgb[t20], a.rgb[t20])
    t01 = (slice(0, 16), slice(16, 32))
    assert _close(r.rgb[t01], a.rgb[t01])
    assert r.n_tiles == 12 and r.n_tiles_double == 1
    assert r.cuts_rendered == (0, 1, 2)
    lv = r.level.detach().cpu().numpy()
    assert abs(lv[5, 5] - 0.5) < 1e-6 and abs(lv[20, 20] - 1.0) < 1e-6 and abs(lv[3, 50] - 2.0) < 1e-6
    assert abs(lv[35, 3]) < 1e-6
    # 타일이 정렬되지 않은 경계: tile=8 이면 같은 맵에서 타일 (0,0)..은 0 또는 1 반반 행 -> 여전히 평균 0.5
    r8 = _dc(lam, tile=8)
    assert r8.n_tiles == 5 * 8
    assert _close(r8.rgb[0:8, 0:8], 0.5 * a.rgb[0:8, 0:8] + 0.5 * b.rgb[0:8, 0:8], 1e-5)


def test_dual_cut_from_depth_driven_lambda():
    """lod_lambda(깊이)로 만든 맵이 그대로 들어간다(이음매)."""
    d = _ref(0).depth.detach().cpu().numpy()
    lam = lod_lambda(d, K, h0=0.1)
    r = _dc(lam)
    assert bool(torch.isfinite(r.rgb).all())
    lv = r.level.detach().cpu().numpy()
    assert lv.min() >= 0 and lv.max() <= 2


def test_validation():
    with pytest.raises(ValueError):
        _dc(0.5, cuts=[])
    with pytest.raises(ValueError):
        _dc(np.zeros((H, W + 1)))
    with pytest.raises(ValueError):
        _dc(float("nan"))
    with pytest.raises(ValueError):
        _dc(0.5, tile=0)
    assert math.isfinite(float(_dc(np.float32(0.3)).rgb.sum()))
