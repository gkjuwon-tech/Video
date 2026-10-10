"""WLD-01: 상수·레벨 수치·block_of·canon_json."""
import numpy as np
import pytest

from buc.core import contracts as C
from buc.world import strata as S


def test_constants():
    assert S.STRATA_SCHEMA == C.WORLD_SCHEMA
    assert S.H0_M == 64.0
    assert S.BLOCK_CELLS == 16
    assert tuple(S.LEVEL_RANGE) == (-8, 24)
    assert S.MAGIC == b"BUCSTRATA0"
    assert S.OWNERS == {"NONE": 0, "INPUT": 1, "INPUT_PART": 2, "INPUT_REFL": 3, "ENGINE": 4, "TEXELFM": 5,
                        "PROC": 6, "HUMAN": 7, "SOUL": 8}
    assert tuple(S.BLOCK_STATES) == ("empty", "bedrock", "proposed", "committed", "dirty")
    assert list(S.TSQ_FIELDS) == ["mu", "normal", "s_uv", "opacity", "edge_n", "color", "level", "sem", "own", "unc"]
    spec = {k: (np.dtype(v[0]), tuple(v[1])) for k, v in S.TSQ_FIELDS.items()}
    assert spec["mu"] == (np.dtype("<f4"), (3,))
    assert spec["s_uv"] == (np.dtype("<f4"), (2,))
    assert spec["opacity"] == (np.dtype("<f4"), ())
    assert spec["level"] == (np.dtype("i1"), ())
    assert spec["sem"] == (np.dtype("<u2"), ())
    assert spec["unc"] == (np.dtype("u1"), ())
    assert issubclass(S.StrataError, C.ContractError)


@pytest.mark.parametrize("l,cell", [(0, 64.0), (6, 1.0), (10, 0.0625), (15, 64.0 / 32768), (-2, 256.0),
                                    (24, 64.0 * 2.0 ** -24), (-8, 64.0 * 256)])
def test_cell_size_exact(l, cell):
    assert S.cell_size(l) == cell
    assert S.texel_pitch(l) == cell / 8
    assert S.block_extent(l) == cell * 16


@pytest.mark.parametrize("bad", [25, -9, 1.5, "3", None])
def test_cell_size_rejects(bad):
    with pytest.raises(S.StrataError):
        S.cell_size(bad)


def test_block_of_values():
    pts = np.array([[-0.5, 0.0, 2.3], [0.999999, -1e-9, 1.0], [-1.0, -1.0, -1.0], [15.9, 16.0, -16.1]])
    got = S.block_of(pts, 10)  # 블록 폭 1 m
    assert got.dtype == np.int64 and got.shape == (4, 4)
    np.testing.assert_array_equal(got, [[10, -1, 0, 2], [10, 0, -1, 1], [10, -1, -1, -1], [10, 15, 16, -17]])
    got6 = S.block_of(pts, 6)  # 블록 폭 16 m
    np.testing.assert_array_equal(got6[:, 1:], [[-1, 0, 0], [0, -1, 0], [-1, -1, -1], [0, 1, -2]])
    assert (got6[:, 0] == 6).all()


def test_block_of_oracle_random():
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=40.0, size=(500, 3))
    for l in (-3, 0, 7, 12):
        E = 64.0 * 2.0 ** -l * 16
        exp = np.floor(pts / E).astype(np.int64)
        got = S.block_of(pts, l)
        np.testing.assert_array_equal(got[:, 1:], exp)
        assert (got[:, 0] == l).all()


def test_block_of_rejects_nonfinite():
    with pytest.raises(S.StrataError):
        S.block_of(np.array([[0.0, np.nan, 1.0]]), 10)
    with pytest.raises(S.StrataError):
        S.block_of(np.array([[0.0, np.inf, 1.0]]), 10)


def test_canon_json_bytes():
    assert S.canon_json({"b": 1, "a": [1.5, "é"]}) == '{"a":[1.5,"é"],"b":1}'.encode("utf-8")
    assert S.canon_json({"z": {"y": True, "x": None}}) == b'{"z":{"x":null,"y":true}}'


@pytest.mark.parametrize("bad", [{"a": float("nan")}, {"a": float("inf")}, {1: 2}, {"a": {2: 1}}, {"a": object()}])
def test_canon_json_rejects(bad):
    with pytest.raises(S.StrataError):
        S.canon_json(bad)
