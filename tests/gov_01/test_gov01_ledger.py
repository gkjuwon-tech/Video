"""GOV-01 원장: 조상 DAG 해시(순서 무관·민감), JSON 왕복·변조 거부, 세탁 거부, 추가 규칙."""
import json
import random

import pytest

from buc.core import det
from buc.gov import lineage as L
from buc.gov import prov as P

H0 = "0" * 64
H1 = "1" * 64


def _prov_json(rec):
    return json.dumps(rec.to_dict(), sort_keys=True, separators=(",", ":"))


def _diamond(order=0, parent_order=0, h_src=H0):
    lg = L.LineageLedger()
    srcs = [("cap", ["CAPTURE"]), ("eng", ["ENGINE"])]
    if order:
        srcs.reverse()
    for sid, taint in srcs:
        lg.add(sid, kind="source", prov=P.ProvRecord(taint=frozenset(taint)),
               content_hash=h_src if sid == "cap" else None)
    ps = ["cap", "eng"] if parent_order == 0 else ["eng", "cap", "cap"]
    lg.add("ds", kind="data", parents=ps)
    lg.add("shard1", kind="shard", parents=["ds"], content_hash=H1)
    lg.add("ckpt", kind="checkpoint", parents=["shard1", "eng"])
    return lg


def test_node_fields_and_parents_sorted():
    lg = _diamond(parent_order=1)
    n = lg.node("ds")
    assert isinstance(n, L.Node)
    assert n.parents == ("cap", "eng")
    assert n.kind == "data"
    assert isinstance(n.prov, P.ProvRecord)
    assert n.prov.taint == frozenset({"CAPTURE", "ENGINE"})
    assert list(lg.ids()) == ["cap", "eng", "ds", "shard1", "ckpt"]
    assert len(lg) == 5 and "ckpt" in lg and "nope" not in lg
    assert lg.ancestors("ckpt") == ("cap", "ds", "eng", "shard1")
    assert lg.ancestors("cap") == ()
    with pytest.raises(KeyError):
        lg.node("nope")
    with pytest.raises(KeyError):
        lg.ancestors("nope")


def test_node_hash_matches_spec_formula():
    lg = _diamond()
    src = lg.node("cap")
    exp_cap = det.hash_parts(L.LINEAGE_TAG, "cap", "source", _prov_json(src.prov), H0, 0)
    assert lg.node_hash("cap") == exp_cap
    eng = lg.node("eng")
    exp_eng = det.hash_parts(L.LINEAGE_TAG, "eng", "source", _prov_json(eng.prov), "", 0)
    assert lg.node_hash("eng") == exp_eng
    ds = lg.node("ds")
    exp_ds = det.hash_parts(L.LINEAGE_TAG, "ds", "data", _prov_json(ds.prov), "", 2, exp_cap, exp_eng)
    assert lg.node_hash("ds") == exp_ds
    assert len(lg.root()) == 32
    exp_root = det.merkle_root_keyed({i: lg.node_hash(i) for i in lg.ids()})
    assert lg.root() == exp_root


def test_derived_prov_is_fnd04_derive():
    lg = _diamond()
    exp = P.derive("lineage:data", [lg.node("cap").prov, lg.node("eng").prov])
    assert lg.node("ds").prov == exp


def test_hash_order_independent():
    a, b, c = _diamond(0, 0), _diamond(1, 0), _diamond(0, 1)
    assert a.root() == b.root() == c.root()
    for i in a.ids():
        assert a.node_hash(i) == b.node_hash(i) == c.node_hash(i)


def test_hash_sensitive_to_ancestor_change():
    a, b = _diamond(), _diamond(h_src="2" * 64)
    assert a.node_hash("eng") == b.node_hash("eng")  # 무관한 노드
    for i in ("cap", "ds", "shard1", "ckpt"):
        assert a.node_hash(i) != b.node_hash(i), i
    assert a.root() != b.root()
    # 소스 taint 변화도 후손 해시를 바꾼다
    c = L.LineageLedger()
    c.add("cap", kind="source", prov=P.ProvRecord(taint=frozenset({"CAPTURE", "HUMAN"})), content_hash=H0)
    c.add("eng", kind="source", prov=P.ProvRecord(taint=frozenset({"ENGINE"})))
    c.add("ds", kind="data", parents=["cap", "eng"])
    assert c.node_hash("ds") != a.node_hash("ds")


def test_empty_ledger_root_defined():
    assert L.LineageLedger().root() == det.merkle_root_keyed({})


def test_json_roundtrip_and_tamper():
    lg = _diamond()
    s = lg.to_json()
    d = json.loads(s)
    assert d["v"] == L.LINEAGE_TAG
    assert d["root"] == lg.root().hex()
    assert [n["id"] for n in d["nodes"]] == list(lg.ids())
    assert d["nodes"][2]["hash"] == lg.node_hash("ds").hex()
    lg2 = L.LineageLedger.from_json(s)
    assert lg2.root() == lg.root()
    assert lg2.node("ckpt") == lg.node("ckpt")

    def tampered(fn):
        dd = json.loads(s)
        fn(dd)
        return json.dumps(dd)

    bad = [
        tampered(lambda dd: dd.update(v="buc.lineage.v0")),
        tampered(lambda dd: dd.update(root="00" * 32)),
        tampered(lambda dd: dd["nodes"][0].update(content_hash="3" * 64)),
        tampered(lambda dd: dd["nodes"][0]["prov"].update(taint=["ENGINE"])),
        tampered(lambda dd: dd["nodes"][3].update(hash="ab" * 32)),
        # 부모가 자식보다 뒤에 오는 순서
        tampered(lambda dd: dd.update(nodes=list(reversed(dd["nodes"])))),
        # USER 소스를 ENGINE으로 세탁: 해시가 맞지 않는다
        tampered(lambda dd: dd["nodes"][1]["prov"].update(taint=["USER"], ns="user_vault")),
        s[:-7],
    ]
    for b in bad:
        with pytest.raises(L.LineageError):
            L.LineageLedger.from_json(b)


def test_add_rules():
    lg = L.LineageLedger()
    with pytest.raises(L.LineageError):
        lg.add("s", kind="source")  # 소스 prov 필수
    with pytest.raises(L.LineageError):
        lg.add("x", kind="data", parents=["missing"])
    with pytest.raises(L.LineageError):
        lg.add("s", kind="weird", prov=["ENGINE"])
    for bad_id in ("", " a", "a b", "-a", "a" * 129, "é"):
        with pytest.raises(L.LineageError):
            lg.add(bad_id, kind="source", prov=["ENGINE"])
    for bad_h in ("A" * 64, "0" * 63, "g" * 64):
        with pytest.raises(L.LineageError):
            lg.add("s", kind="source", prov=["ENGINE"], content_hash=bad_h)
    assert len(lg) == 0
    n1 = lg.add("s", kind="source", prov=["ENGINE"], content_hash=H0)
    assert lg.add("s", kind="source", prov=["ENGINE"], content_hash=H0) == n1  # 멱등
    with pytest.raises(L.LineageError):
        lg.add("s", kind="source", prov=["ENGINE"], content_hash=H1)
    with pytest.raises(L.LineageError):
        lg.add("s", kind="data", prov=["ENGINE"], content_hash=H0)
    assert len(lg) == 1
    lg.add("MRH/v0@step100", kind="checkpoint", parents=["s"])  # 정본 ID 문자 허용


def test_laundering_rejected():
    lg = L.LineageLedger()
    lg.add("u", kind="source", prov=["USER:t1"])
    lg.add("e", kind="source", prov=["ENGINE"])
    # 선언 prov로 USER-DERIVED를 지우려 함
    with pytest.raises(L.LineageError):
        lg.add("w", kind="data", parents=["u", "e"], prov=["ENGINE"])
    # taint는 덮지만 ns를 느슨하게: ProvRecord 자체가 막거나 원장이 막는다
    with pytest.raises((L.LineageError, P.ProvError)):
        lg.add("w", kind="data", parents=["u"],
               prov=P.ProvRecord(taint=frozenset({"USER-DERIVED:t1"}), ns="train_ok"))
    # 상위집합 선언은 허용(더 엄격)
    n = lg.add("w", kind="data", parents=["u", "e"],
               prov=P.ProvRecord(taint=frozenset({"USER-DERIVED:t1", "ENGINE", "NC"})))
    assert "NC" in n.prov.taint
    assert "w" in lg


def test_random_order_invariance_seeded():
    rng = random.Random(1234)
    specs = []
    for k in range(40):
        if k < 5:
            specs.append((f"s{k}", None, rng.choice(["ENGINE", "CAPTURE", "SI", "ASSET:CC0"])))
        else:
            ps = rng.sample([s[0] for s in specs], rng.randint(1, min(3, len(specs))))
            specs.append((f"n{k}", ps, None))

    def build(shuffle_parents):
        lg = L.LineageLedger()
        r = random.Random(7 + shuffle_parents)
        for nid, ps, tag in specs:
            if ps is None:
                lg.add(nid, kind="source", prov=[tag])
            else:
                ps = list(ps)
                if shuffle_parents:
                    r.shuffle(ps)
                    ps = ps + ps[:1]
                lg.add(nid, kind="data", parents=ps)
        return lg

    a, b = build(0), build(1)
    assert a.root() == b.root()
    assert L.LineageLedger.from_json(b.to_json()).root() == a.root()
