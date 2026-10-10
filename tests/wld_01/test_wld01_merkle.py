"""WLD-01: Merkle 섹션 바이트 배치 오라클·순서 무관·1 ulp 민감·증명."""
import struct

import numpy as np
import pytest

from buc.core import contracts as C
from buc.core import det
from buc.world import strata as S
from wld01_fixtures import make_strata, make_tsq_cols, oracle_entries, populate


def test_section_entries_match_oracle():
    st = populate(S, make_strata(S))
    got = st.section_entries()
    exp = oracle_entries(st)
    assert set(got) == set(exp)
    for k in exp:
        assert bytes(got[k]) == exp[k], k
    assert st.merkle_root() == det.merkle_root_keyed(exp)
    assert len(st.merkle_root()) == 32


def test_entries_minimal_and_empty_tsq():
    st = S.Strata(obs=make_strata(S).obs, theta_in=C.Theta(), level="L1", read_hash="h", prov=C.Prov())
    e = st.section_entries()
    assert "far_field" not in e
    assert not any(k.startswith(("layer/", "block/", "chart/")) for k in e)
    assert e["tsq/mu"] == det.hash_array(np.zeros((0, 3), dtype="<f4"))
    assert e["tsq/unc"] == det.hash_array(np.zeros((0,), dtype="u1"))
    assert e["seed"] == bytes(16)
    assert e == oracle_entries(st)


def test_block_key_format_negative():
    st = populate(S, make_strata(S))
    e = st.section_entries()
    assert "block/10/-1/0/2" in e and "block/6/3/-2/0" in e
    assert "layer/0000" in e and "layer/0001" in e and "chart/wall.1" in e


def test_insertion_order_independent():
    a = populate(S, make_strata(S, meta={"a": 1, "b": {"x": 1, "y": 2}}), order=0)
    b = populate(S, make_strata(S, meta={"b": {"y": 2, "x": 1}, "a": 1}), order=1)
    assert a.merkle_root() == b.merkle_root()
    assert a.to_bytes() == b.to_bytes()


def _root_after(mut):
    st = populate(S, make_strata(S))
    r0 = st.merkle_root()
    e0 = st.section_entries()
    mut(st)
    return r0, e0, st.merkle_root(), st.section_entries()


def _ulp_tsq(st):
    st.tsq.columns["mu"][4, 1] = np.nextafter(st.tsq.columns["mu"][4, 1], np.float32(np.inf))


def test_ulp_tsq_changes_root_only_that_section():
    r0, e0, r1, e1 = _root_after(_ulp_tsq)
    assert r0 != r1
    diff = {k for k in e0 if e0[k] != e1[k]}
    assert diff == {"tsq/mu"}


def test_ulp_layer_depth_changes_root():
    def mut(st):
        d = st.layers[1].depth
        d[0, 0] = np.nextafter(d[0, 0], np.float32(np.inf))
    r0, e0, r1, e1 = _root_after(mut)
    assert r0 != r1 and {k for k in e0 if e0[k] != e1[k]} == {"layer/0001"}


def test_block_change_isolated():
    def mut(st):
        st.set_block((6, 3, -2, 0), S.Block(state="bedrock", own=6, unc=254))
    r0, e0, r1, e1 = _root_after(mut)
    assert r0 != r1 and {k for k in e0 if e0[k] != e1[k]} == {"block/6/3/-2/0"}


@pytest.mark.parametrize("field", ["seed", "meta", "prov", "level", "far_field"])
def test_scalar_sections_sensitive(field):
    base = populate(S, make_strata(S))
    kw = {}
    if field == "seed":
        other = populate(S, make_strata(S, seed=12346))
    elif field == "meta":
        other = populate(S, make_strata(S, meta={"lot": "near", "n": 3.0}))  # int != float
    else:
        other = populate(S, make_strata(S, **kw))
        if field == "prov":
            other.prov = C.Prov(taint=frozenset({"USER"}))
        elif field == "level":
            other.level = "L2a"
        else:
            ff = other.far_field.rgb.copy()
            ff[0, 0, 0] = np.nextafter(ff[0, 0, 0], np.float32(0))
            other.set_far_field(C.FarField(rgb=ff))
    eb, eo = base.section_entries(), other.section_entries()
    assert {k for k in eb if eb[k] != eo.get(k)} == {field}
    assert base.merkle_root() != other.merkle_root()


def test_proofs_all_keys():
    st = populate(S, make_strata(S))
    root = st.merkle_root()
    e = st.section_entries()
    keys = sorted(e, key=lambda k: k.encode("utf-8"))
    for pos, key in enumerate(keys):
        idx, n, proof, item = st.proof(key)
        kb = key.encode("utf-8")
        assert idx == pos and n == len(keys)
        assert item == struct.pack("<Q", len(kb)) + kb + bytes(e[key])
        assert det.merkle_verify(item, idx, n, proof, root)
        assert not det.merkle_verify(item + b"\x00", idx, n, proof, root)
    with pytest.raises(KeyError):
        st.proof("block/0/0/0/0")


def test_root_stable_across_calls_and_tsq_append_changes():
    st = populate(S, make_strata(S))
    assert st.merkle_root() == st.merkle_root()
    r = st.merkle_root()
    st.add_tsq(**make_tsq_cols(1, seed=11))
    assert st.merkle_root() != r
