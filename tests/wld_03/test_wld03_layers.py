"""WLD-03 깊이 클립·찢기·숨은 층·매팅 층 공식·합성 항등."""
import buc.world.depth_world as dw
import numpy as np
import pytest
from scipy import ndimage as ndi
from wld03_helpers import (
    BLUE,
    RED,
    X0,
    X1,
    Y0,
    Y1,
    Z_BG,
    Z_FG,
    H,
    W,
    box_scene,
    composite,
    fg_mask,
    halo_scene,
    make_read,
    ring_mask,
    tear_oracle,
)


def _cfg(**kw):
    kw.setdefault("tsq_stride", 4)
    return dw.DepthWorldConfig(**kw)


def test_depth_clip_and_sky():
    img = np.full((8, 10, 3), 0.5, np.float32)
    d = np.full((8, 10), 3.0, np.float32)
    d[0, :] = np.inf
    d[1, :3] = 2e4      # far_m 이상 -> 하늘
    d[2, :3] = 1e-5     # near_m 미만 -> near_m
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(far_m=1e4, near_m=1e-3))
    sky = np.zeros((8, 10), bool)
    sky[0, :] = True
    sky[1, :3] = True
    assert np.array_equal(res.sky, sky)
    assert np.all(res.z[sky] == 1e4)
    assert np.allclose(res.z[2, :3], 1e-3)
    assert np.allclose(res.z[3:], 3.0)
    assert res.meta()["sky_frac"] == pytest.approx(sky.mean())
    l0 = res.world.layers[0]
    assert np.allclose(l0.depth, res.z.astype(np.float32))
    assert np.all(l0.rgba[..., 3] == 1.0)


@pytest.mark.parametrize("jump,expect", [(1.09, False), (1.11, True)])
def test_tear_threshold_both_sides(jump, expect):
    img = np.full((6, 12, 3), 0.4, np.float32)
    d = np.full((6, 12), 2.0)
    d[:, 6:] = 2.0 * jump
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(rel_jump=0.10))
    assert bool(res.tear_x[:, 5].all()) is expect
    assert not res.tear_x[:, :5].any() and not res.tear_x[:, 6:].any()
    assert not res.tear_y.any()
    assert bool(res.near[:, 5].all()) is expect and bool(res.far[:, 6].all()) is expect


def test_tear_oracle_random_scene():
    rng = np.random.default_rng(3)
    h, w = 24, 32
    lab = rng.integers(0, 4, size=(h // 4, w // 4))
    d = np.repeat(np.repeat(np.array([1.5, 1.6, 3.0, 7.0])[lab], 4, 0), 4, 1)
    d = d * (1.0 + 0.02 * rng.random((h, w)))
    img = rng.random((h, w, 3)).astype(np.float32)
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(rel_jump=0.10))
    z = np.asarray(d, np.float32).astype(np.float64)
    tx, ty, near, far = tear_oracle(z, 0.10)
    assert np.array_equal(res.tear_x, tx)
    assert np.array_equal(res.tear_y, ty)
    assert np.array_equal(res.near, near)
    assert np.array_equal(res.far, far)
    assert res.meta()["n_tear_pairs"] == int(tx.sum() + ty.sum())


def test_smooth_slanted_surface_has_no_tears():
    """인접 비 1.05 경사(전체 범위는 수십 배) -> 찢김 0, 숨은 층 비어 있음."""
    h, w = 16, 64
    u = np.arange(w)
    d = np.tile(1.0 * 1.05 ** u, (h, 1))
    img = np.full((h, w, 3), 0.3, np.float32)
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(ext_px=16.0))
    assert not res.tear_x.any() and not res.tear_y.any()
    assert not res.hidden.any()
    assert np.all(res.world.layers[1].rgba[..., 3] == 0.0)
    # 층 0 깊이는 입력 깊이 그대로(평활 없음)
    assert np.array_equal(res.world.layers[0].depth, np.asarray(d, np.float32))


def test_box_layers_formula():
    img, d = box_scene()
    ext = 6.0
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(ext_px=ext))
    z = d.astype(np.float64)
    _, _, _near, far = tear_oracle(z, 0.10)
    dist, idx = ndi.distance_transform_edt(~far, return_indices=True)
    qz = z[idx[0], idx[1]]
    hidden = (dist <= ext) & (qz > z * 1.1)
    assert np.array_equal(res.hidden, hidden)
    assert hidden.sum() > 0 and not (hidden & ~fg_mask()).any()
    assert not res.matted.any()
    l0, l1 = res.world.layers
    assert np.allclose(l0.rgba[..., :3], np.clip(img, 0, 1))
    assert np.all(l0.rgba[..., 3] == 1.0)
    assert np.array_equal(l0.depth, d)
    a1 = l1.rgba[..., 3]
    assert np.array_equal(a1 > 0, hidden)
    assert np.all(a1[hidden] == 1.0)
    assert np.allclose(l1.depth[hidden], Z_BG)
    qc = img[idx[0], idx[1]]
    assert np.allclose(l1.rgba[..., :3][hidden], qc[hidden], atol=1e-6)
    # 층 깊이는 {2, 6} 값만(중간 깊이 없음)
    vals = np.unique(np.concatenate([l0.depth.ravel(), l1.depth[hidden]]))
    assert set(np.round(vals, 5)) <= {Z_FG, Z_BG}
    # 관측 시점 합성 = 입력
    assert np.allclose(composite(res.world), np.clip(img, 0, 1), atol=1e-6)
    assert res.strata.composite()[1].min() == pytest.approx(1.0)


def test_ext_px_zero_gives_empty_hidden():
    img, d = box_scene()
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(ext_px=0.0))
    assert not res.hidden.any()
    assert np.all(res.world.layers[1].rgba[..., 3] == 0.0)


def test_hidden_extent_scales_with_ext_px():
    img, d = box_scene()
    n = [int(dw.build_depth_world(make_read(img, d), cfg=_cfg(ext_px=e)).hidden.sum()) for e in (2.0, 5.0, 30.0)]
    assert n[0] < n[1] < n[2]
    assert n[2] == int(fg_mask().sum())  # 30 px 이면 전경 전체 뒤가 배경으로 이어진다


def test_hdr_input_clipped_in_layers_only():
    img, d = box_scene()
    img = img.copy()
    img[0:4, 0:4] = 3.0
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(tsq_stride=1))
    assert res.world.layers[0].rgba[..., :3].max() <= 1.0
    col = res.strata.tsq.columns["color"]
    assert col.max() == pytest.approx(3.0)


def test_matte_layers_formula_and_composite():
    img, d, m = halo_scene()
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(ext_px=16.0), matte=m)
    z = d.astype(np.float64)
    _, _, near, far = tear_oracle(z, 0.10)
    _, in_ = ndi.distance_transform_edt(~near, return_indices=True)
    _, if_ = ndi.distance_transform_edt(~far, return_indices=True)
    zf = z[in_[0], in_[1]]
    zb = z[if_[0], if_[1]]
    matted = m.band & (zb > zf * 1.1)
    assert np.array_equal(res.matted, matted)
    assert matted.sum() == m.band.sum()
    assert not (res.hidden & res.matted).any()
    l0, l1 = res.world.layers
    a0 = l0.rgba[..., 3].astype(np.float64)
    exp_a0 = np.where(matted, np.where(m.alpha > 1e-3, m.alpha, 0.0), 1.0)
    assert np.allclose(a0, exp_a0, atol=1e-7)
    assert np.allclose(l0.depth[matted], zf[matted])
    assert np.allclose(l1.depth[matted], zb[matted])
    assert np.all(l1.rgba[..., 3][matted] == 1.0)
    assert np.allclose(l1.rgba[..., :3][matted], BLUE, atol=1e-6)
    ring = ring_mask()
    assert np.allclose(l0.rgba[..., :3][ring], RED, atol=1e-6)
    assert np.allclose(l0.depth[ring], Z_FG)
    # 숨은 층 색은 band 안 최근접 far 화소의 B(파랑) — 혼합 보라가 아니다
    hid = res.hidden
    assert hid.sum() > 0
    assert np.allclose(l1.rgba[..., :3][hid], BLUE, atol=1e-6)
    # 합성 = αF + (1-α)B = I
    assert np.allclose(composite(res.world), img, atol=1e-6)
    assert res.meta()["matte"] is True
    assert res.meta()["matted_frac"] == pytest.approx(matted.mean())


def test_matte_ignored_without_discontinuity():
    """찢김이 없으면 band 가 있어도 매팅을 적용하지 않는다(zf, zb 정의 불가)."""
    h, w = 12, 12
    img = np.full((h, w, 3), 0.5, np.float32)
    d = np.full((h, w), 3.0)
    band = np.zeros((h, w), bool)
    band[4:8, 4:8] = True
    m = type("M", (), {})()
    m.alpha = np.full((h, w), 0.5)
    m.fg = np.full((h, w, 3), 0.9)
    m.bg = np.full((h, w, 3), 0.1)
    m.band = band
    res = dw.build_depth_world(make_read(img, d), cfg=_cfg(), matte=m)
    assert not res.matted.any()
    assert np.allclose(composite(res.world), img, atol=1e-6)


def test_world_layer_order_valid_everywhere():
    """층 0 ≤ 층 1 깊이(둘 다 보이는 화소)는 World 검증이 통과하는 것으로 확인."""
    img, d, m = halo_scene()
    for matte in (None, m):
        res = dw.build_depth_world(make_read(img, d), cfg=_cfg(ext_px=40.0), matte=matte)
        l0, l1 = res.world.layers
        both = (l0.rgba[..., 3] > 1e-3) & (l1.rgba[..., 3] > 1e-3)
        assert both.any()
        assert np.all(l0.depth[both] < l1.depth[both])
        res.world.validate()


def test_box_geometry_sanity():
    # 장면 상수 자체가 맞는지(오라클 전제)
    m = fg_mask()
    assert m.sum() == (Y1 - Y0) * (X1 - X0)
    assert W == 96 and H == 64
