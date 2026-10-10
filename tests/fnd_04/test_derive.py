"""FND-04 derive 전파: 태그 변환, mode, ns, FirewallError, 단조성·순서 무관(시드 고정 무작위)."""
import random

import pytest

from buc.gov import prov as P

R = P.ProvRecord


def rec(*tags, mode="none", roots=(), ns=None):
    return R(taint=frozenset(tags), mode=mode, roots=tuple(roots), ns=ns)


def test_user_becomes_user_derived():
    w = P.derive("world", [rec("USER:soul", roots=("up1",))])
    assert w.taint == frozenset({"USER-DERIVED:soul"})
    assert w.roots == ("up1",)
    assert w.ns == "user_vault" and w.ext


def test_extgen_becomes_derived():
    w = P.derive("world", [rec("EXTGEN:soul", "CAPTURE")])
    assert w.taint == frozenset({"EXTGEN-DERIVED:soul", "CAPTURE"})
    assert w.ns == "ext_vault"


def test_human_ref_makes_inspired():
    for t in ("EXTGEN:soul", "EXTGEN-DERIVED:soul", "EXTGEN-INSPIRED:soul"):
        x = P.derive("preset", [rec(t, mode="authored")], human_ref=True)
        assert x.taint == frozenset({"EXTGEN-INSPIRED:soul"})
        assert x.mode == "inspired"
        assert x.ext
        assert not P.train_ok(x) and not P.dev_ok(x)
        assert "Q17" in P.why_not_train(x)


def test_human_ref_never_launders_user():
    x = P.derive("preset", [rec("USER")], human_ref=True)
    assert x.taint == frozenset({"USER-DERIVED"})
    assert not any(t.startswith("EXTGEN-INSPIRED") for t in x.taint)
    assert x.mode == "inspired" and x.ns == "user_vault"


def test_human_ref_on_clean_parent_keeps_mode():
    x = P.derive("preset", [rec("CAPTURE", mode="derived")], human_ref=True)
    assert x.mode == "derived" and x.taint == frozenset({"CAPTURE"})
    assert P.train_ok(x)


def test_other_tags_carried():
    x = P.derive("op", [rec("HOLDOUT", "COMP:kling", "IP-FLAG", "TWIN:c2", "OPEN-GEN:z:h:train")])
    assert x.taint == frozenset({"HOLDOUT", "COMP:kling", "IP-FLAG", "TWIN:c2", "OPEN-GEN:z:h:train"})
    assert x.ns == "comp_vault" and x.comp and x.hold


def test_mode_is_max():
    x = P.derive("op", [rec("ENGINE", mode="none"), rec("ENGINE", mode="authored"), rec("SI", mode="derived")])
    assert x.mode == "authored"


def test_ns_propagates_from_parent_even_if_clean():
    x = P.derive("op", [rec("ENGINE", ns="user_vault"), rec("CAPTURE")])
    assert x.ns == "user_vault"


def test_shared_weight_update_firewall():
    with pytest.raises(P.FirewallError):
        P.derive("ttt", [rec("USER")], updates_shared_weights=True)
    with pytest.raises(P.FirewallError):
        P.derive("lora", [rec("ENGINE"), rec("EXTGEN:soul")], updates_shared_weights=True)
    ok = P.derive("train", [rec("ENGINE"), rec("COMP")], updates_shared_weights=True)  # ext 아님
    assert ok.comp


def test_bad_args():
    with pytest.raises(ValueError):
        P.derive("op", [])
    with pytest.raises(ValueError):
        P.derive("", [rec("ENGINE")])
    with pytest.raises(TypeError):
        P.derive("op", "USER")


def test_accepts_prov_like_parents():
    x = P.derive("op", [["USER"], {"taint": ["COMP"], "mode": "none", "roots": ["r"], "ns": "comp_vault"}])
    assert x.taint == frozenset({"USER-DERIVED", "COMP"}) and x.roots == ("r",)


# ---- 시드 고정 무작위 성질 검사 ----
POOL = [
    "ENGINE", "CAPTURE", "SI", "HUMAN", "TWIN:c1", "TEL-SYS", "ASSET:CC0", "ASSET:CC-BY-NC", "ASSET",
    "OPEN-GEN:z:h:train", "OPEN-GEN:g3b:h:dev-eval", "OPEN-GEN:g4:h:bench", "COMMISSION:train",
    "COMMISSION:eval", "COMMISSION:eval+train", "IP-FLAG", "IP-FLAG:masked", "USER", "USER:mj",
    "USER-DERIVED", "BIO", "EXTGEN:soul", "EXTGEN-DERIVED:gpt-image", "EXTGEN-INSPIRED:soul", "COMP",
    "HOLDOUT", "PUB-SEALED", "EVAL-ONLY", "TEL-CAGG", "OPTIN", "D3", "NC",
]
NS_ORDER = {n: i for i, n in enumerate(P.NAMESPACES)}


def rand_rec(rng):
    tags = rng.sample(POOL, rng.randint(0, 3))
    mode = rng.choice(P.MODES)
    roots = tuple(rng.sample(["r1", "r2", "r3", "r4"], rng.randint(0, 2)))
    base = R(taint=frozenset(tags), mode=mode, roots=roots)
    ns = rng.choice([n for n in P.NAMESPACES if NS_ORDER[n] <= NS_ORDER[base.ns]])
    return R(taint=frozenset(tags), mode=mode, roots=roots, ns=ns)


def test_random_monotonic_and_order_free():
    rng = random.Random(20261010)
    for _ in range(600):
        ps = [rand_rec(rng) for _ in range(rng.randint(1, 4))]
        hr = rng.random() < 0.3
        x = P.derive("op", ps, human_ref=hr)
        if P.train_ok(x):
            assert all(P.train_ok(p) for p in ps), (ps, x)
        if P.dev_ok(x):
            assert all(P.dev_ok(p) for p in ps), (ps, x)
        assert x.ext == any(p.ext for p in ps)
        assert x.comp == any(p.comp for p in ps)
        assert x.hold == any(p.hold for p in ps)
        assert all(NS_ORDER[x.ns] <= NS_ORDER[p.ns] for p in ps)
        assert set(x.roots) == set().union(*(p.roots for p in ps))
        assert P.prov_bits(x) == (x.ext * 1) | (x.comp * 2) | (x.hold * 4)
        shuffled = list(ps)
        rng.shuffle(shuffled)
        assert P.derive("op", shuffled + [ps[0]], human_ref=hr) == x
        if not hr:
            assert P.derive("op", [x]).taint == x.taint
