"""PRD-03 flat_world: 깊이 구간 분할, 합성 항등, 층 순서, params, 극단 깊이."""
import numpy as np
import pytest

from buc.core import contracts as C
from prd03_helpers import call, composite, make_image, manual_read

# 경계 근처를 피한 깊이 값: n=3, dmin=1, dmax=8 -> 경계 1, 2, 4, 8
DEPTH = np.array([[1.0, 1.5, 3.0, 6.0],
                  [8.0, 1.2, 3.9, 7.0],
                  [2.5, 5.0, 1.1, 6.5]], np.float32)


def world_of(read, params=None, seed=777):
    return call("S2", {"read": read}, params, seed)["world"]


def test_split_matches_oracle_and_contract_fields():
    read = manual_read(DEPTH, seed=1)
    out = call("S2", {"read": read})
    assert set(out) == {"world"}
    w = out["world"]
    assert isinstance(w, C.World)
    w.validate()
    C.check_stage_io("S2", {"read": read}, out)
    assert len(w.layers) == 3
    assert w.obs.content_hash() == read.obs.content_hash()
    assert w.theta_in.content_hash() == read.theta_in.content_hash()
    assert w.level == read.level and w.read_hash == read.content_hash()
    assert w.prov == read.prov and w.far_field is None and w.schema == C.WORLD_SCHEMA
    edges = [1.0 * 8.0 ** (j / 3) for j in range(4)]
    assert w.meta["stub"] == "flat_world" and w.meta["n_layers"] == 3
    assert np.allclose(w.meta["bin_edges"], edges, rtol=1e-9)
    want_bin = np.array([[0, 0, 1, 2], [2, 0, 1, 2], [1, 2, 0, 2]])
    for j, ly in enumerate(w.layers):
        a = ly.rgba[..., 3]
        assert set(np.unique(a)) <= {0.0, 1.0}
        assert np.array_equal(a == 1.0, want_bin == j), j
        z = np.sqrt(edges[j] * edges[j + 1])
        assert ly.depth.dtype == np.float32 and ly.depth.shape == DEPTH.shape
        assert np.allclose(ly.depth, z, rtol=1e-6) and (ly.depth == ly.depth.flat[0]).all()
        assert np.allclose(ly.rgba[..., :3], np.clip(read.image_lin, 0, 1), atol=0)


def test_layer_depth_lies_in_its_bin_and_is_ordered():
    w = world_of(manual_read(DEPTH, seed=2))
    zs = [float(ly.depth.flat[0]) for ly in w.layers]
    assert zs == sorted(zs) and len(set(zs)) == 3
    e = w.meta["bin_edges"]
    for j, z in enumerate(zs):
        assert e[j] < z < e[j + 1]


def test_alpha_partitions_each_pixel_and_composite_equals_image():
    img = make_image(24, 40, seed=5)
    read = call("S-1", {"image": img})["read"]
    w = world_of(read)
    alpha_sum = sum(ly.rgba[..., 3].astype(np.float64) for ly in w.layers)
    assert (alpha_sum == 1.0).all()
    assert np.allclose(composite(w), np.clip(read.image_lin, 0, 1), atol=1e-6)
    # 평면 판독의 층은 수평 띠 3개: 열 방향으로 같다
    for ly in w.layers:
        a = ly.rgba[..., 3]
        assert (a == a[:, :1]).all()
    assert [bool(ly.rgba[..., 3].any()) for ly in w.layers] == [True, True, True]


def test_n_layers_param():
    read = manual_read(DEPTH, seed=3)
    for n in (1, 2, 5, 16):
        w = world_of(read, {"n_layers": n})
        assert len(w.layers) == n and w.meta["n_layers"] == n and len(w.meta["bin_edges"]) == n + 1
        assert np.allclose(composite(w), np.clip(read.image_lin, 0, 1), atol=1e-6)
        assert sum(ly.rgba[..., 3].astype(np.float64) for ly in w.layers).min() == 1.0
    w1 = world_of(read, {"n_layers": 1})
    assert w1.layers[0].depth.flat[0] == pytest.approx(np.sqrt(1.0 * 8.0), rel=1e-6)


def test_constant_depth_puts_everything_in_last_layer():
    read = manual_read(np.full((4, 6), 5.0, np.float32), seed=4)
    w = world_of(read)
    assert len(w.layers) == 3
    assert [float(ly.rgba[..., 3].sum()) for ly in w.layers] == [0.0, 0.0, 24.0]
    assert all(np.allclose(ly.depth, 5.0, rtol=1e-6) for ly in w.layers)
    assert np.allclose(composite(w), np.clip(read.image_lin, 0, 1), atol=1e-6)


def test_infinite_depth_goes_to_last_layer():
    d = DEPTH.copy()
    d[0, :] = np.inf
    read = manual_read(d, seed=5)
    w = world_of(read)
    w.validate()
    assert (w.layers[2].rgba[0, :, 3] == 1.0).all()
    assert np.isfinite([ly.depth for ly in w.layers]).all()
    assert np.allclose(composite(w), np.clip(read.image_lin, 0, 1), atol=1e-6)
    only_inf = manual_read(np.full((3, 3), np.inf, np.float32), seed=6)
    w2 = world_of(only_inf)
    w2.validate()
    assert (w2.layers[2].rgba[..., 3] == 1.0).all() and (w2.layers[0].rgba[..., 3] == 0.0).all()


def test_hdr_float_image_is_clipped_not_rejected():
    rgb = np.random.default_rng(8).random((3, 4, 3)).astype(np.float32) * 4.0
    read = manual_read(DEPTH, image_lin=rgb)
    w = world_of(read)
    w.validate()
    assert np.allclose(composite(w), np.clip(rgb, 0, 1), atol=1e-6)


def test_does_not_mutate_input_and_ignores_seed():
    read = manual_read(DEPTH, seed=9)
    h0 = read.content_hash()
    a = world_of(read, seed=1)
    b = world_of(read, {"_seed": 3}, seed=424242)
    assert read.content_hash() == h0
    assert a.content_hash() == b.content_hash()
    for ly in a.layers:                                  # 서로 다른 버퍼
        assert not np.shares_memory(ly.rgba, read.image_lin)
    assert not np.shares_memory(a.layers[0].rgba, a.layers[1].rgba)


def test_taint_preserved():
    prov = C.Prov(taint=frozenset({"USER", "EXTGEN:soul"}), mode="mixed", roots=("r1",), ns="user_vault")
    read = manual_read(DEPTH, prov=prov)
    w = world_of(read)
    assert w.prov == prov and w.prov.ext


@pytest.mark.parametrize("params", [{"n_layers": 0}, {"n_layers": 17}, {"n_layers": 2.5}, {"n_layers": -1},
                                    {"layers": 3}, {"n_layers": True}])
def test_bad_params_rejected(params):
    with pytest.raises(ValueError):
        world_of(manual_read(DEPTH), params)
