"""check_ledger: 정상 원장, 1% 경계, 위반 종류별 탐지, 결정론."""
import copy

import pytest

from fnd05_helpers import D, E, SCHEMA, checks, doc, load

INS = ["world.tfu.safe", "world.tfu.mid", "world.tfu.full"]
BUCKETS = ["bucket.escher", "bucket.ff", "bucket.ir", "bucket.mass", "bucket.sty", "bucket.x6"]


def good_entries():
    return [
        E("world.tfu.safe", 179), E("world.tfu.mid", 393), E("world.tfu.full", 544),
        D("world.tfu.mean3", 372, "(world.tfu.safe+world.tfu.mid+world.tfu.full)/3", INS),
        E("bucket.escher", 30, "T4-h", "bench"), E("bucket.ff", 40, "T4-h", "bench"),
        E("bucket.ir", 35, "T4-h", "bench"), E("bucket.mass", 10, "T4-h", "bench"),
        E("bucket.sty", 25, "T4-h", "bench"), E("bucket.x6", 30, "T4-h", "bench"),
        D("bucket.sum", 170, "sum(" + ",".join(BUCKETS) + ")", BUCKETS, "T4-h", "bench"),
        E("x.hours", 2.5, "h", "catalog"), E("x.rate", 60, "TFU/h", "catalog"),
        D("x.tfu", 150, "x.hours*x.rate", ["x.hours", "x.rate"], "TFU", "catalog"),
    ]


def test_good_ledger_is_clean(lc):
    d = doc(*good_entries())
    snap = copy.deepcopy(d)
    r = lc.check_ledger(d)
    assert r.ok is True and r.violations == () and isinstance(r.violations, tuple)
    assert r.checked == len(good_entries())
    assert set(r.computed) == {"world.tfu.mean3", "bucket.sum", "x.tfu"}
    assert r.computed["world.tfu.mean3"] == pytest.approx(372.0)
    assert r.computed["bucket.sum"] == pytest.approx(170.0)
    assert r.computed["x.tfu"] == pytest.approx(150.0)
    assert d == snap, "입력 문서를 바꾸면 안 된다"


def one_derived(value, tol=None, **top):
    d = doc(E("x", 100), D("y", value, "x", ["x"]), **top)
    L = load()
    return L.check_ledger(d) if tol is None else L.check_ledger(d, tol=tol)


@pytest.mark.parametrize("value,ok", [
    (100, True), (101, True),          # 정확히 +1.000% : 통과
    (99, True),                        # -1.000% : 통과
    (101.01, False),                   # +1.01%
    (98.9, False),                     # -1.1%
    (110, False), (50, False), (0, False), (-100, False),
])
def test_one_percent_boundary(value, ok):
    r = one_derived(value)
    assert r.ok is ok
    if not ok:
        vs = checks(r, "mismatch")
        assert [v.id for v in vs] == ["y"] and vs[0].why
    assert r.computed["y"] == pytest.approx(100.0)


def test_relative_to_computed_not_declared():
    # computed=100, value=101 -> 1.0% 통과. computed=101, value=100 -> 0.990% 통과. computed=100,value=98.5 -> 1.5% 실패
    assert one_derived(101).ok
    d = doc(E("x", 101), D("y", 100, "x", ["x"]))
    assert load().check_ledger(d).ok
    assert not one_derived(98.5).ok


def test_tolerance_priority():
    assert not one_derived(105).ok
    assert one_derived(105, tol=0.05).ok                   # 인자
    assert one_derived(105, tolerance=0.06).ok             # 문서
    assert not one_derived(105, tol=0.01, tolerance=0.5).ok  # 인자 > 문서
    assert one_derived(105, tol=0.5, tolerance=0.001).ok
    assert one_derived(100, tol=0).ok and not one_derived(100.0000001, tol=0).ok
    assert one_derived(100, tolerance=0).ok


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2, True, "x", float("nan"), []])
def test_bad_tol_argument_raises(lc, bad):
    with pytest.raises(lc.LedgerFormatError):
        lc.check_ledger(doc(E("x", 1)), tol=bad)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "a", float("nan")])
def test_bad_doc_tolerance_is_violation_not_exception(lc, bad):
    r = lc.check_ledger(doc(E("x", 1), tolerance=bad))
    assert not r.ok and checks(r, "schema")


def test_zero_computed():
    base = [E("x", 3), E("w", 3)]
    ok = doc(*base, D("z", 0, "x-w", ["x", "w"]))
    assert load().check_ledger(ok).ok
    bad = doc(*base, D("z", 0.001, "x-w", ["x", "w"]))
    r = load().check_ledger(bad)
    assert [v.id for v in checks(r, "mismatch")] == ["z"]
    assert load().check_ledger(bad, tol=0.99).ok is False   # 0 으로 나누는 상대오차는 허용 불가


def test_negative_and_sign():
    ok = doc(E("x", -5), D("y", -10.05, "x*2", ["x"]))
    assert load().check_ledger(ok).ok
    bad = doc(E("x", -5), D("y", 10, "x*2", ["x"]))
    assert [v.id for v in checks(load().check_ledger(bad), "mismatch")] == ["y"]


def test_error_is_local_not_propagated():
    d = doc(E("a", 10), D("b", 25, "a*2", ["a"]), D("c", 30, "b+5", ["b"]))
    r = load().check_ledger(d)
    assert {v.id for v in r.violations} == {"b"} and {v.check for v in r.violations} == {"mismatch"}
    assert r.computed["b"] == pytest.approx(20.0) and r.computed["c"] == pytest.approx(30.0)


def test_unit_check_additive_only():
    L = load()
    base = [E("a", 100, "TFU"), E("b", 50, "T4-h")]
    bad = doc(*base, D("t", 150, "a+b", ["a", "b"], "TFU"))
    assert [v.id for v in checks(L.check_ledger(bad), "unit")] == ["t"]
    bad2 = doc(*base, D("t", 100, "max(a, b)", ["a", "b"], "TFU"))
    assert [v.id for v in checks(L.check_ledger(bad2), "unit")] == ["t"]
    bad3 = doc(*base, D("t", 100, "a", ["a"], "T4-h"))     # 출력 단위가 다르다
    assert [v.id for v in checks(L.check_ledger(bad3), "unit")] == ["t"]
    ok_lit = doc(*base, D("t", 105, "a + 5", ["a"], "TFU"))
    assert L.check_ledger(ok_lit).ok
    ok_neg = doc(*base, D("t", -100, "-a", ["a"], "TFU"))
    assert L.check_ledger(ok_neg).ok
    mult = doc(*base, D("t", 5000, "a*b", ["a", "b"], "TFU*T4-h"))
    assert L.check_ledger(mult).ok
    div = doc(*base, D("t", 2, "a/b", ["a", "b"], "ratio"))
    assert L.check_ledger(div).ok
    mixed = doc(*base, D("t", 250, "a*2 + b", ["a", "b"], "TFU"))   # '*' 가 있으면 단위 검사 안 함
    assert L.check_ledger(mixed).ok


def test_duplicate_id_one_violation_per_id():
    L = load()
    r = L.check_ledger(doc(E("a", 1), E("a", 1), E("a", 1), E("b", 2), E("b", 2), E("c", 3)))
    vs = checks(r, "duplicate_id")
    assert sorted(v.id for v in vs) == ["a", "b"] and not r.ok


def test_owners_list():
    L = load()
    d = doc(E("a", 1, owner="pipeline"), E("b", 1, owner="legal"), owners=["pipeline", "bench"])
    r = L.check_ledger(d)
    assert [v.id for v in checks(r, "owner")] == ["b"] and not r.ok
    assert L.check_ledger(doc(E("b", 1, owner="legal"))).ok     # owners 없으면 자유
    assert checks(L.check_ledger(doc(E("a", 1), owners="pipeline")), "schema")
    assert checks(L.check_ledger(doc(E("a", 1), owners=[1, 2])), "schema")


def test_formula_errors_are_violations_not_exceptions():
    L = load()
    d = doc(E("a", 5), E("z", 0),
            D("syn", 1, "a +", ["a"]),
            D("div", 1, "a/z", ["a", "z"]),
            D("junk", 1, "a $ 2", ["a"]),
            D("ok", 5, "a", ["a"]))
    r = L.check_ledger(d)
    assert sorted(v.id for v in checks(r, "formula")) == ["div", "junk", "syn"]
    assert not checks(r, "mismatch")
    assert "ok" in r.computed and not ({"syn", "div", "junk"} & set(r.computed))


def test_inputs_mismatch_variants():
    L = load()
    base = [E("a", 1), E("b", 2)]
    cases = {
        "missing": D("t", 3, "a+b", ["a"]),
        "extra": D("t", 1, "a", ["a", "b"]),
        "dup": D("t", 3, "a+b", ["a", "b", "b"]),
    }
    for name, ent in cases.items():
        r = L.check_ledger(doc(*base, ent))
        assert [v.id for v in checks(r, "inputs_mismatch")] == ["t"], name
        assert not r.ok
    no_inputs = {"id": "t", "value": 3, "unit": "TFU", "owner": "pipeline", "formula": "a+b"}
    assert [v.id for v in checks(L.check_ledger(doc(*base, no_inputs)), "inputs_mismatch")] == ["t"]
    leaf_with_inputs = E("t", 3, inputs=["a"])
    assert [v.id for v in checks(L.check_ledger(doc(*base, leaf_with_inputs)), "inputs_mismatch")] == ["t"]
    # 순서가 달라도, 빈 inputs 의 잎도, formula: null 의 잎도 정상
    assert L.check_ledger(doc(*base, D("t", 3, "a+b", ["b", "a"]))).ok
    assert L.check_ledger(doc(*base, E("t", 3, inputs=[]), E("u", 3, formula=None))).ok
    assert L.check_ledger(doc(D("k", 6, "2*3", []))).ok         # 참조 없는 상수식


def test_unknown_input():
    L = load()
    r = L.check_ledger(doc(E("a", 1), D("t", 2, "a+ghost", ["a", "ghost"])))
    vs = checks(r, "unknown_input")
    assert [v.id for v in vs] == ["t"] and "ghost" in vs[0].why and not r.ok
    assert not checks(r, "mismatch")


def test_cycles():
    L = load()
    selfloop = L.check_ledger(doc(D("a", 5, "a+1", ["a"])))
    assert [v.id for v in checks(selfloop, "cycle")] == ["a"]
    two = L.check_ledger(doc(E("leaf", 1), D("a", 2, "b+leaf", ["b", "leaf"]), D("b", 2, "a", ["a"]),
                             D("c", 2, "a", ["a"])))
    assert sorted(v.id for v in checks(two, "cycle")) == ["a", "b"]
    three = L.check_ledger(doc(D("a", 1, "c", ["c"]), D("b", 1, "a", ["a"]), D("c", 1, "b", ["b"]), E("z", 1)))
    assert sorted(v.id for v in checks(three, "cycle")) == ["a", "b", "c"]
    assert not three.ok
    acyclic = L.check_ledger(doc(E("a", 1), D("b", 1, "a", ["a"]), D("c", 1, "a+b-1", ["a", "b"]),
                                 D("d", 2, "b+c", ["b", "c"])))
    assert acyclic.ok


def test_deep_chain_no_recursion_problem():
    L = load()
    ents = [E("n0", 1.0)] + [D(f"n{i}", 1.0, f"n{i-1}", [f"n{i-1}"]) for i in range(1, 400)]
    assert L.check_ledger(doc(*ents)).ok
    ents[0] = D("n0", 1.0, f"n{399}", ["n399"])           # 긴 순환
    assert len(checks(L.check_ledger(doc(*ents)), "cycle")) == 400


@pytest.mark.parametrize("bad", [None, [], [1], "x", 3, {"schema": "nope", "entries": [E("a", 1)]},
                                 {"entries": [E("a", 1)]}, {"schema": SCHEMA}, {"schema": SCHEMA, "entries": []},
                                 {"schema": SCHEMA, "entries": {"a": 1}}, {"schema": SCHEMA, "entries": None}])
def test_document_level_schema(lc, bad):
    r = lc.check_ledger(bad)
    assert r.ok is False and checks(r, "schema") and r.checked >= 0
    assert all(v.check == "schema" for v in r.violations)


def test_non_mapping_entry_does_not_hide_others():
    L = load()
    r = L.check_ledger(doc(5, E("x", 100), D("y", 150, "x", ["x"]), "junk", None))
    assert r.checked == 5
    assert len(checks(r, "schema")) == 3
    assert [v.id for v in checks(r, "mismatch")] == ["y"]


@pytest.mark.parametrize("ent", [
    E("a-b", 1), E("1a", 1), E("", 1), E("a..b", 1), E("a.", 1), E("a b", 1), E("a\n", 1), E(".a", 1), E("한글", 1),
    E(None, 1), E(5, 1),
    E("a", True), E("a", "5"), E("a", None), E("a", float("nan")), E("a", float("inf")), E("a", float("-inf")),
    E("a", 1, unit=""), E("a", 1, unit=None), E("a", 1, unit=3),
    E("a", 1, owner=""), E("a", 1, owner=None),
    {"id": "a", "unit": "TFU", "owner": "p"}, {"id": "a", "value": 1, "owner": "p"},
    {"id": "a", "value": 1, "unit": "TFU"}, {"value": 1, "unit": "TFU", "owner": "p"},
])
def test_entry_schema_violations(lc, ent):
    r = lc.check_ledger(doc(E("good", 1), ent))
    assert r.ok is False and checks(r, "schema"), ent
    assert r.checked == 2


def test_extra_keys_and_numeric_types_ok(lc):
    r = lc.check_ledger(doc(E("a", 1, note="메모", src="v2.1 §3"), E("b", 2.5), E("c", -3), E("d", 0)))
    assert r.ok


def test_deterministic_and_order_independent(lc):
    ents = [E("a", 10), D("b", 25, "a*2", ["a"]), D("c", 1, "a+ghost", ["a", "ghost"]),
            D("d", 5, "d", ["d"]), E("u1", 1, "TFU"), E("u2", 1, "h"), D("e", 2, "u1+u2", ["u1", "u2"]),
            E("o", 1, owner="legal")]
    d1 = doc(*ents, owners=["pipeline"])
    d2 = doc(*reversed(ents), owners=["pipeline"])
    r1, r2, r3 = lc.check_ledger(d1), lc.check_ledger(d1), lc.check_ledger(d2)
    assert r1 == r2 == r3 and not r1.ok
    keys = [(v.id, v.check, v.why) for v in r1.violations]
    assert keys == sorted(keys)
    assert {v.check for v in r1.violations} >= {"mismatch", "unknown_input", "cycle", "unit", "owner"}
    assert r1.ok == (len(r1.violations) == 0)


def test_dataclasses_are_frozen(lc):
    r = lc.check_ledger(doc(E("a", 1)))
    with pytest.raises(Exception):
        r.ok = False
    v = lc.Violation("mismatch", "a", "x")
    assert (v.check, v.id, v.why) == ("mismatch", "a", "x")
    with pytest.raises(Exception):
        v.id = "b"
    assert lc.SCHEMA == SCHEMA and lc.DEFAULT_TOL == 0.01 and lc.DEFAULT_LEDGER == "docs/number-ledger.yaml"
