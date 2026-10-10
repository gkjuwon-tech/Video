"""WLD-01: 컨테이너 불변식·실패 시 불변·composite·Block·TSQ·chart."""
import numpy as np
import pytest

from buc.core import contracts as C
from buc.world import strata as S
from wld01_fixtures import H, W, make_far_field, make_layer, make_obs, make_prov, make_strata, make_tsq_cols, populate


def test_construct_defaults():
    st = S.Strata(obs=make_obs(), theta_in=C.Theta(), level="L2a", read_hash="x", prov=C.Prov())
    assert st.seed == 0 and st.meta == {} and list(st.layers) == [] and st.far_field is None
    assert st.blocks == {} and st.charts == {} and st.tsq.n == 0
    assert st.schema == C.WORLD_SCHEMA
    st.validate()


@pytest.mark.parametrize("kw", [dict(level="L3"), dict(read_hash=""), dict(seed=-1), dict(seed=2 ** 128),
                                dict(seed=1.0), dict(meta={"a": float("nan")}), dict(meta={"strata_root": "x"}),
                                dict(meta={"strata_seed": "0"}), dict(meta={3: "x"})])
def test_construct_rejects(kw):
    args = dict(obs=make_obs(), theta_in=C.Theta(), level="L1", read_hash="h", prov=C.Prov())
    args.update(kw)
    with pytest.raises(S.StrataError):
        S.Strata(**args)


def test_seed_max_ok():
    st = S.Strata(obs=make_obs(), theta_in=C.Theta(), level="L1", read_hash="h", prov=C.Prov(), seed=2 ** 128 - 1)
    assert st.seed == 2 ** 128 - 1


def test_meta_deep_copied():
    m = {"lot": {"near": [1, 2]}}
    st = make_strata(S, meta=m)
    m["lot"]["near"].append(3)
    assert st.meta == {"lot": {"near": [1, 2]}}


def test_add_layer_order_and_size():
    st = make_strata(S, layers=False)
    assert st.add_layer(make_layer(1.0, 0.5, 1)) == 0
    assert st.add_layer(make_layer(2.0, 1.0, 2)) == 1
    with pytest.raises(S.StrataError):  # 뒤 층보다 앞에 놓인 층(앞->뒤 위반)
        st.add_layer(make_layer(1.5, 1.0, 3))
    assert len(st.layers) == 2
    with pytest.raises(S.StrataError):  # obs 크기 불일치
        st.add_layer(make_layer(3.0, 1.0, 4, h=H + 1))
    with pytest.raises(S.StrataError):
        st.add_layer("not a layer")
    assert len(st.layers) == 2
    # 알파가 거의 0인 픽셀은 순서 검사에서 제외된다
    st.add_layer(make_layer(0.5, 0.0, 5))
    assert len(st.layers) == 3


def test_far_field_type():
    st = make_strata(S, far=False)
    st.set_far_field(make_far_field())
    assert isinstance(st.far_field, C.FarField)
    st.set_far_field(None)
    assert st.far_field is None
    with pytest.raises(S.StrataError):
        st.set_far_field(np.ones((4, 4, 3), dtype=np.float32))


def test_composite_two_layers():
    st = make_strata(S, layers=False)
    a = make_layer(1.0, 0.6, 1)
    b = make_layer(2.0, 1.0, 2)
    st.add_layer(a)
    st.add_layer(b)
    rgb, alpha = st.composite()
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.float32
    assert alpha.shape == (H, W) and alpha.dtype == np.float32
    exp = 0.6 * a.rgba[..., :3].astype(np.float64) + 0.4 * b.rgba[..., :3].astype(np.float64)
    np.testing.assert_allclose(rgb, exp, atol=1e-6)
    np.testing.assert_allclose(alpha, 1.0, atol=1e-6)


def test_composite_random_oracle():
    st = make_strata(S, layers=False)
    rng = np.random.default_rng(9)
    layers = []
    for i in range(3):
        rgba = rng.random((H, W, 4)).astype(np.float32)
        ly = C.LDILayer(rgba=rgba, depth=np.full((H, W), 1.0 + i, dtype=np.float32))
        st.add_layer(ly)
        layers.append(ly)
    acc = np.zeros((H, W, 3))
    T = np.ones((H, W))
    for ly in layers:
        a = ly.rgba[..., 3].astype(np.float64)
        acc += ly.rgba[..., :3] * (a * T)[..., None]
        T *= 1.0 - a
    rgb, alpha = st.composite()
    np.testing.assert_allclose(rgb, acc, atol=1e-6)
    np.testing.assert_allclose(alpha, 1.0 - T, atol=1e-6)


def test_composite_empty():
    st = make_strata(S, layers=False)
    rgb, alpha = st.composite()
    assert rgb.shape == (H, W, 3) and not rgb.any() and not alpha.any()


@pytest.mark.parametrize("kw", [dict(state="done", own=1), dict(state="committed", own=99),
                                dict(state="committed", own=1, unc=256), dict(state="committed", own=1, unc=-1),
                                dict(state="committed", own=1, tsq_start=3, tsq_stop=2),
                                dict(state="committed", own=1, tsq_start=-1, tsq_stop=2),
                                dict(state="committed", own=True)])
def test_block_rejects(kw):
    with pytest.raises(S.StrataError):
        S.Block(**kw)


def test_block_frozen_and_digest():
    b = S.Block(state="committed", own=S.OWNERS["INPUT"], unc=3, tsq_start=0, tsq_stop=0)
    with pytest.raises(Exception):
        b.state = "dirty"
    from buc.core import det
    assert b.digest() == det.hash_parts(b"buc.strata.block.v0", "committed", 1, 3, 0, 0)
    assert S.Block(state="empty", own=0).unc == 255


def test_set_block_keys_and_ranges():
    st = make_strata(S)
    st.add_tsq(**make_tsq_cols(4))
    st.set_block((10, -1, 0, 2), S.Block(state="committed", own=1, tsq_start=1, tsq_stop=4))
    assert st.get_block((10, -1, 0, 2)).tsq_stop == 4
    assert st.get_block((10, 0, 0, 0)) is None
    bad_keys = [(30, 0, 0, 0), (-9, 0, 0, 0), (10, 0, 0), (10, 0.0, 0, 0), (10, 2 ** 31, 0, 0), (10, 0, -2 ** 31 - 1, 0)]
    for key in bad_keys:
        with pytest.raises(S.StrataError):
            st.set_block(key, S.Block(state="empty", own=0))
    with pytest.raises(S.StrataError):  # tsq 범위 밖
        st.set_block((10, 0, 0, 0), S.Block(state="committed", own=1, tsq_start=2, tsq_stop=5))
    with pytest.raises(S.StrataError):
        st.set_block((10, 0, 0, 0), {"state": "empty"})
    assert len(st.blocks) == 1
    st.set_block((10, 0, -2 ** 31, 2 ** 31 - 1), S.Block(state="empty", own=0))
    assert len(st.blocks) == 2


def test_tsq_append_and_dtypes():
    st = make_strata(S)
    assert st.add_tsq(**make_tsq_cols(5, seed=1)) == (0, 5)
    assert st.add_tsq(**make_tsq_cols(3, seed=2)) == (5, 8)
    assert st.tsq.n == 8
    for name, (dt, tail) in S.TSQ_FIELDS.items():
        col = st.tsq.columns[name]
        assert col.dtype == np.dtype(dt), name
        assert col.shape == (8, *tail), name
        assert col.flags["C_CONTIGUOUS"]
    ref = np.concatenate([make_tsq_cols(5, seed=1)["mu"], make_tsq_cols(3, seed=2)["mu"]])
    np.testing.assert_array_equal(st.tsq.columns["mu"], ref)
    # float64 입력도 정본 dtype으로 변환된다
    cols = make_tsq_cols(2, seed=3)
    cols["mu"] = cols["mu"].astype(np.float64)
    assert st.add_tsq(**cols) == (8, 10)
    assert st.tsq.columns["mu"].dtype == np.float32


def _bad(field, value_fn):
    cols = make_tsq_cols(4, seed=6)
    cols[field] = value_fn(cols[field])
    return cols


@pytest.mark.parametrize("cols", [
    _bad("normal", lambda a: a * 1.01),
    _bad("opacity", lambda a: a + 1.5),
    _bad("edge_n", lambda a: a * 0 + 1.0),
    _bad("edge_n", lambda a: a * 0 + 65.0),
    _bad("s_uv", lambda a: a * 0),
    _bad("color", lambda a: a - 2.0),
    _bad("mu", lambda a: np.where(np.arange(4)[:, None] == 2, np.nan, a)),
    _bad("level", lambda a: np.full(4, 200, dtype=np.int64)),
    _bad("level", lambda a: np.full(4, 30, dtype=np.int8)),
    _bad("own", lambda a: np.full(4, 99, dtype=np.uint16)),
    _bad("mu", lambda a: a[:3]),
    _bad("s_uv", lambda a: np.ones((4, 3), dtype=np.float32)),
])
def test_tsq_rejects_and_unchanged(cols):
    st = make_strata(S)
    st.add_tsq(**make_tsq_cols(2, seed=0))
    before = {k: v.copy() for k, v in st.tsq.columns.items()}
    with pytest.raises(S.StrataError):
        st.add_tsq(**cols)
    assert st.tsq.n == 2
    for k, v in before.items():
        np.testing.assert_array_equal(st.tsq.columns[k], v)


def test_tsq_missing_and_extra_fields():
    st = make_strata(S)
    cols = make_tsq_cols(2)
    del cols["sem"]
    with pytest.raises(S.StrataError):
        st.add_tsq(**cols)
    cols = make_tsq_cols(2)
    cols["z_slot"] = np.zeros((2, 16), dtype=np.int8)
    with pytest.raises(S.StrataError):
        st.add_tsq(**cols)
    assert st.tsq.n == 0


def test_charts():
    st = make_strata(S)
    rec = {"axis": "+z", "uv": [0, 1]}
    st.set_chart("c-0.a_b", rec)
    rec["uv"].append(2)
    assert st.charts["c-0.a_b"] == {"axis": "+z", "uv": [0, 1]}
    for bad in ["a/b", "", "c 0", "é"]:
        with pytest.raises(S.StrataError):
            st.set_chart(bad, {})
    with pytest.raises(S.StrataError):
        st.set_chart("c1", {"x": float("nan")})
    assert set(st.charts) == {"c-0.a_b"}


def test_validate_full():
    st = populate(S, make_strata(S))
    st.validate()
    assert make_prov() == st.prov
