"""R2 할레이션(적색 번짐 헤일로: 반경·색·세기)과 플레어 고스트(광원–중심 직선 위 반점)."""
from functools import lru_cache

import numpy as np
import pytest
from scipy import ndimage as ndi

import rd03_helpers as H
from buc.lens import sensor_film as sf
from buc.read import screen_fx as fx

HH, WW = H.H0, H.W0
DISCS = ((120.0, 100.0, 14.0), (290.0, 160.0, 10.0))


def halo_image(a, y_src, seed=0):
    x = H.disc_scene(discs=tuple((cx, cy, r, y_src) for cx, cy, r in DISCS), seed=seed)
    return H.saturate(sf.apply_halation(x.astype(np.float32), a)).astype(np.float32)


@lru_cache(maxsize=None)
def halo_fit(a, y_src, seed=0):
    return fx.estimate_halation(halo_image(a, y_src, seed))


@pytest.mark.parametrize("a,y_src,seed", [(0.15, 3.0, 0), (0.3, 3.0, 1)])
def test_halation_gain_radius_tint(a, y_src, seed):
    f = halo_fit(a, y_src, seed)
    p_true = a * (y_src - H.HALATION_THRESHOLD)                   # amount·(Y-0.8)
    assert f.detected is True and f.reliable is True
    assert f.n_blobs == 2 and len(f.blobs) == 2
    assert 0.75 * p_true <= f.halo_gain <= 1.25 * p_true
    assert 0.016 <= f.sigma_frac <= 0.024                           # LNS-04: 0.02·W
    assert abs(f.sigma_px - f.sigma_frac * WW) < 1e-6
    assert f.tint[0] == 1.0                                        # 적색 기준 정규화
    assert 0.25 <= f.tint[1] <= 0.45 and 0.04 <= f.tint[2] <= 0.20      # (1, 0.35, 0.12)
    # 광원 위치(면적 큰 순)
    got = sorted((b.x, b.y) for b in f.blobs)
    want = sorted((cx, cy) for cx, cy, _ in DISCS)
    for (gx, gy), (wx, wy) in zip(got, want):
        assert abs(gx - wx) <= 2.0 and abs(gy - wy) <= 2.0
    areas = sorted(b.area_px for b in f.blobs)
    assert abs(areas[1] - np.pi * 14.0 ** 2) <= 0.25 * np.pi * 14.0 ** 2
    # 세기-휘도 식별성 한계: 곱 P 는 식별, 분리는 가정으로
    assert f.amount_min <= a                                        # 하한은 사실을 넘지 않는다(건전)
    assert abs(f.amount_min - f.halo_gain / (H.HALO_SOURCE_MAX - H.HALATION_THRESHOLD)) < 1e-9
    assert abs(f.amount - min(1.0, f.halo_gain / (H.HALO_ASSUMED_SOURCE_Y - H.HALATION_THRESHOLD))) < 1e-9
    # 참 amount 를 알면 헤일로 에너지로 포화 광원의 HDR 휘도를 역산한다
    assert abs(f.source_luminance(a) / y_src - 1.0) <= 0.25


def test_hdr_inversion_with_assumed_source_matches():
    """광원 휘도가 가정(2.0)과 같으면 amount 가 참값에 가깝다."""
    f = halo_fit(0.4, 2.0, 2)
    assert f.detected is True
    assert abs(f.amount - 0.4) <= 0.4 * 0.25
    for bad in (0.0, -0.1, 1.01, float("nan")):
        with pytest.raises(ValueError):
            f.source_luminance(bad)


def test_no_halation_is_not_detected():
    f = halo_fit(0.0, 3.0, 3)
    assert f.detected is False and f.amount == 0.0
    assert f.halo_gain <= 0.05
    assert f.n_blobs == 2                                           # 포화 광원 자체는 센다


def test_white_glow_is_not_halation():
    x = H.disc_scene(discs=((120.0, 100.0, 14.0, 3.0),), seed=1)
    glow = ndi.gaussian_filter(H.disc_cov(HH, WW, 120.0, 100.0, 14.0), 0.02 * WW)[..., None] * 0.3
    f = fx.estimate_halation(H.saturate(x + glow).astype(np.float32))      # 중성(무채색) 번짐
    assert f.detected is False and f.amount == 0.0
    assert "neutral_halo" in f.warnings


def test_no_saturated_source():
    f = fx.estimate_halation(H.rect_scene(seed=0).astype(np.float32))
    assert f.n_blobs == 0 and f.detected is False and f.amount == 0.0
    assert "no_saturation" in f.warnings and f.reliable is False
    # 최소 면적 미만의 포화 점은 광원이 아니다
    tiny = H.disc_scene(discs=((100.0, 100.0, 2.0, 3.0),)).astype(np.float32)
    assert fx.estimate_halation(H.saturate(tiny).astype(np.float32)).n_blobs == 0
    big = fx.estimate_halation(H.saturate(tiny).astype(np.float32), opts=fx.ScreenFxOpts(min_blob_px=4))
    assert big.n_blobs == 1


# ------------------------------------------------------------------------------------------------ 고스트
SRC = (290.0, 60.0)


def ghost_image(seed=0, ks=(-0.45, -1.0), center=None, radius=11.0):
    x = H.disc_scene(discs=((SRC[0], SRC[1], 8.0, 3.0),), bg=0.1, seed=seed)
    x = x + 0.01 * H.hf_field(HH, WW, seed)[..., None]
    y, pos = H.with_ghosts(x, SRC, ks, radius=radius, center=center)
    return H.saturate(y).astype(np.float32), pos


@pytest.mark.parametrize("seed", [0, 1])
def test_ghost_positions(seed):
    img, pos = ghost_image(seed)
    g = fx.estimate_ghost(img)
    assert g.source is not None and abs(g.source[0] - SRC[0]) <= 2.0 and abs(g.source[1] - SRC[1]) <= 2.0
    assert len(g.ghosts) == 2
    c = np.array([(WW - 1) / 2.0, (HH - 1) / 2.0])
    assert tuple(np.round(g.center, 6)) == tuple(np.round(c, 6))
    for (px, py), k_true in zip(sorted(pos, key=lambda p: p[0]), (-1.0, -0.45)):
        best = min(g.ghosts, key=lambda s: np.hypot(s.x - px, s.y - py))
        assert np.hypot(best.x - px, best.y - py) <= 5.0                  # 목표 ±5 px
        assert abs(best.k - k_true) <= 0.08
        assert 5.5 <= best.radius_px <= 22.0
        assert best.z >= 6.0
    assert len({(round(s.x), round(s.y)) for s in g.ghosts}) == 2


def test_ghost_mask_covers_and_stays_small():
    img, pos = ghost_image(1)
    g = fx.estimate_ghost(img)
    m = g.mask(HH, WW)
    assert m.shape == (HH, WW) and m.dtype == bool
    yy, xx = np.mgrid[0:HH, 0:WW]
    truth = np.zeros((HH, WW), bool)
    for px, py in pos:
        truth |= np.hypot(xx - px, yy - py) <= 11.0
    assert (m & truth).sum() >= 0.85 * truth.sum()
    assert m.sum() <= 3.0 * truth.sum()
    with pytest.raises(ValueError):
        g.mask(0, 10)


def test_ghost_negatives():
    # 광원만(고스트 없음)
    x = H.disc_scene(discs=((SRC[0], SRC[1], 8.0, 3.0),), bg=0.1)
    g = fx.estimate_ghost(H.saturate(x).astype(np.float32))
    assert len(g.ghosts) == 0 and g.source is not None
    # 광원–중심 직선 밖의 반점은 고스트가 아니다(광학 중심을 다른 곳으로 놓고 만든 반점)
    y, pos2 = H.with_ghosts(x, SRC, (0.6,), center=(250.0, 200.0))
    assert 0 < pos2[0][0] < WW and 0 < pos2[0][1] < HH
    g2 = fx.estimate_ghost(H.saturate(y).astype(np.float32))
    assert len(g2.ghosts) == 0
    # 포화 광원 없음
    g3 = fx.estimate_ghost(H.rect_scene(seed=2).astype(np.float32))
    assert g3.source is None and len(g3.ghosts) == 0


def test_ghost_center_and_max_options():
    c = (150.0, 80.0)
    img, pos = ghost_image(2, ks=(-0.4, -0.8), center=c)
    g = fx.estimate_ghost(img, center=c)
    assert len(g.ghosts) == 2 and tuple(g.center) == c
    for px, py in pos:
        assert min(np.hypot(s.x - px, s.y - py) for s in g.ghosts) <= 5.0
    g1 = fx.estimate_ghost(img, center=c, opts=fx.ScreenFxOpts(ghost_max=1))
    assert len(g1.ghosts) == 1
    assert g1.ghosts[0].z >= max(s.z for s in g.ghosts) - 1e-9           # 가장 강한 것을 남긴다
    g0 = fx.estimate_ghost(img, center=c, opts=fx.ScreenFxOpts(ghost_max=0))
    assert len(g0.ghosts) == 0
