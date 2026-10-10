"""FND-04 ProvRecord·ns 엄격도·비트·Merkle 배치·FND-02 변환·JSON 왕복."""
import dataclasses
import json

import pytest

from buc.gov import prov as P

R = P.ProvRecord


@pytest.mark.parametrize("tags,ext,comp,hold,bits,ns", [
    ([], False, False, False, 0, "train_ok"),
    (["ENGINE", "CAPTURE"], False, False, False, 0, "train_ok"),
    (["USER"], True, False, False, 1, "user_vault"),
    (["BIO"], True, False, False, 1, "user_vault"),
    (["EXTGEN:soul"], True, False, False, 1, "ext_vault"),
    (["EXTGEN-INSPIRED:soul"], True, False, False, 1, "ext_vault"),
    (["COMP"], False, True, False, 2, "comp_vault"),
    (["EVAL-ONLY"], False, False, False, 0, "comp_vault"),
    (["HOLDOUT"], False, False, True, 4, "pub_bench"),
    (["PUB-SEALED"], False, False, True, 4, "pub_bench"),
    (["COMP:veo", "HOLDOUT"], False, True, True, 6, "comp_vault"),
    (["USER", "COMP", "HOLDOUT"], True, True, True, 7, "user_vault"),
    (["EXTGEN:gpt-image", "HOLDOUT"], True, False, True, 5, "ext_vault"),
    (["TEL-CAGG", "D3", "NC"], False, False, False, 0, "train_ok"),
])
def test_bits_and_ns(tags, ext, comp, hold, bits, ns):
    r = R(taint=frozenset(tags))
    assert (r.ext, r.comp, r.hold) == (ext, comp, hold)
    assert (P.ext_bit(tags), P.comp_bit(tags), P.hold_bit(tags)) == (ext, comp, hold)
    assert P.prov_bits(tags) == bits == r.bits
    assert r.ns == ns == P.implied_ns(frozenset(tags))
    assert P.merkle_parts(r) == ("buc.prov.v1", bits, ns)
    assert P.content_address_kind(r) == ("hmac" if ext else "sha256")


def test_most_restrictive():
    assert P.most_restrictive("train_ok", "pub_bench") == "pub_bench"
    assert P.most_restrictive("pub_bench", "comp_vault", "train_ok") == "comp_vault"
    assert P.most_restrictive("ext_vault", "user_vault") == "user_vault"
    assert P.most_restrictive("train_ok") == "train_ok"
    with pytest.raises(ValueError):
        P.most_restrictive()
    with pytest.raises(ValueError):
        P.most_restrictive("engine")


def test_record_validation():
    with pytest.raises(P.ProvError):
        R(taint=frozenset({"USER"}), ns="train_ok")  # 명시 ns가 implied보다 느슨
    with pytest.raises(P.ProvError):
        R(taint=frozenset({"HOLDOUT"}), ns="train_ok")
    assert R(taint=frozenset({"ENGINE"}), ns="user_vault").ns == "user_vault"  # 더 엄격은 허용
    with pytest.raises(P.ProvError):
        R(mode="engine")
    with pytest.raises(P.ProvError):
        R(ns="nowhere")
    with pytest.raises(P.TagError):
        R(taint=frozenset({"BOGUS"}))


def test_record_normalizes_and_is_value_object():
    a = R(taint=frozenset({"SOUL", "ENGINE"}), roots=("b", "a", "b"))
    b = R(taint=frozenset({"EXTGEN:soul", "ENGINE"}), roots=("a", "b"))
    assert a == b and hash(a) == hash(b)
    assert a.taint == frozenset({"EXTGEN:soul", "ENGINE"})
    assert a.roots == ("a", "b")
    assert a.mode == "none" and a.ns == "ext_vault"
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.mode = "derived"  # frozen


def test_json_roundtrip():
    r = R(taint=frozenset({"USER:soul", "COMP", "OPEN-GEN:z:h:train"}), mode="derived",
          roots=("u1", "a9"), ns="user_vault")
    d = r.to_dict()
    s = json.dumps(d)
    assert d["v"] == "buc.prov.v1"
    assert d["taint"] == sorted(d["taint"])
    assert R.from_dict(json.loads(s)) == r
    bad = dict(d, v="buc.prov.v0")
    with pytest.raises(P.ProvError):
        R.from_dict(bad)


def test_as_record_mapping_without_v():
    r = P.as_record({"taint": ["HOLDOUT"], "mode": "none", "roots": [], "ns": "pub_bench"})
    assert r == R(taint=frozenset({"HOLDOUT"}))


def test_as_record_from_fnd02_prov():
    from buc.core.contracts import Prov
    p = Prov(taint=frozenset({"USER", "COMP:kling"}), mode="engine", roots=("u2", "u1"), ns="engine")
    r = P.as_record(p)
    assert r.taint == frozenset({"USER", "COMP:kling"})
    assert r.mode == "none"
    assert r.roots == ("u1", "u2")
    assert r.ns == "user_vault"  # engine→train_ok, 그러나 implied가 더 엄격
    assert r.ext and r.comp and not r.hold
    # FND-02 Prov.ext는 USER를 빼지만 FND-04 ext_bit는 USER를 센다.
    assert P.ext_bit(p) is True
    assert not P.train_ok(p) and not P.dev_ok(p)
    clean = Prov(taint=frozenset({"ENGINE", "CAPTURE"}), mode="mixed", ns="engine")
    rc = P.as_record(clean)
    assert rc.ns == "train_ok" and rc.mode == "derived"
    assert P.train_ok(clean)


def test_as_record_object_duck_typing():
    class Duck:
        taint = frozenset({"HOLDOUT"})
        mode = "authored"
        roots = ("r",)
        ns = "user_vault"
    r = P.as_record(Duck())
    assert r.ns == "user_vault" and r.mode == "authored" and r.hold
