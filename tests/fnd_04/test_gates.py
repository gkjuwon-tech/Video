"""FND-04 train_ok/dev_ok/signal_class 진리표(권리 렌즈 §1.2 표)와 조건 규칙."""
import pytest

from buc.gov import prov as P

# (태그, train_ok(main), dev_ok)
TRUTH = [
    ("ENGINE", True, True),
    ("CAPTURE", True, True),
    ("SI", True, True),
    ("HUMAN", True, True),
    ("TWIN:c3", True, True),
    ("TEL-SYS", True, True),
    ("ASSET:CC0", True, True),
    ("ASSET:PD", True, True),
    ("ASSET:CC-BY", True, True),
    ("ASSET:Apache-2.0", True, True),
    ("ASSET:CC-BY-NC", False, True),
    ("ASSET", False, True),
    ("OPEN-GEN:z:h1:train", True, True),
    ("OPEN-GEN:g3b:h2:dev-eval", False, True),
    ("OPEN-GEN:g4:h3:bench", False, False),
    ("COMMISSION:train", True, False),
    ("COMMISSION:eval", False, True),
    ("COMMISSION:eval+train", True, True),
    ("COMMISSION:publish", False, False),
    ("IP-FLAG", False, False),
    ("IP-FLAG:masked", True, True),
    ("USER", False, False),
    ("USER:soul", False, False),
    ("USER-DERIVED", False, False),
    ("BIO", False, False),
    ("EXTGEN:soul", False, False),
    ("EXTGEN-DERIVED:soul", False, False),
    ("EXTGEN-INSPIRED:soul", False, False),
    ("COMP", False, False),
    ("COMP:seedance", False, False),
    ("HOLDOUT", False, False),
    ("PUB-SEALED", False, False),
    ("EVAL-ONLY", False, False),
    ("TEL-CAGG", False, False),
    ("OPTIN", False, False),
    ("D3", False, True),
    ("NC", False, True),
]


@pytest.mark.parametrize("tag,tr,dv", TRUTH)
def test_truth_table(tag, tr, dv):
    x = [tag]
    assert P.train_ok(x) is tr
    assert P.dev_ok(x) is dv
    assert P.signal_class(x) == ("S" if dv else "J")
    assert (len(P.why_not_train(x)) == 0) is tr
    assert (len(P.why_not_dev(x)) == 0) is dv


@pytest.mark.parametrize("tag,tr,dv", TRUTH)
def test_contamination_is_absorbing(tag, tr, dv):
    # 깨끗한 태그를 더해도 금지는 풀리지 않는다.
    x = [tag, "ENGINE", "CAPTURE", "ASSET:CC0"]
    assert P.train_ok(x) is tr
    assert P.dev_ok(x) is dv


def test_empty_is_clean():
    assert P.train_ok([]) and P.dev_ok([]) and P.signal_class([]) == "S"
    assert P.why_not_train([]) == () and P.why_not_dev([]) == ()


def test_canary_c1_c5():
    for c in (["USER"], ["EXTGEN:test"], ["ENGINE", "EXTGEN:test"], ["CAPTURE", "USER:x"]):
        assert not P.train_ok(c)
        assert not P.dev_ok(c)
        with pytest.raises(P.FirewallError):
            P.assert_train_ok(c, what="canary")
        with pytest.raises(P.FirewallError):
            P.assert_dev_ok(c, what="canary")


def test_assert_ok_passes_and_message_names_reason():
    P.assert_train_ok(["ENGINE", "CAPTURE"])
    P.assert_dev_ok(["TWIN:c1"])
    with pytest.raises(P.FirewallError) as e:
        P.assert_train_ok(["ENGINE", "COMP:kling"], what="shard-7")
    assert "COMP" in str(e.value)


def test_why_not_sorted_and_names_offenders():
    w = P.why_not_train(["HOLDOUT", "ENGINE", "COMP", "EXTGEN:soul"])
    assert list(w) == sorted(w)
    joined = " ".join(w)
    for t in ("HOLDOUT", "COMP", "EXTGEN:soul"):
        assert t in joined
    assert "ENGINE" not in joined


def test_q17_inspired_mode():
    clean_inspired = P.ProvRecord(taint=frozenset({"ENGINE"}), mode="inspired")
    assert P.train_ok(clean_inspired)  # ext 조상 없는 inspired는 Q17 대상 아님
    r = P.ProvRecord(taint=frozenset({"EXTGEN-INSPIRED:soul"}), mode="inspired")
    assert "Q17" in P.why_not_train(r)
    assert not P.dev_ok(r)


def test_optin_branch():
    assert P.train_ok(["CAPTURE", "OPTIN", "HUMAN"], branch="optin-v1")
    assert P.train_ok(["ENGINE", "ASSET:CC0", "OPTIN"], branch="optin-v1")
    assert not P.train_ok(["CAPTURE", "OPTIN"])  # main 브랜치는 OPTIN 금지
    for bad in (["OPTIN", "USER"], ["OPTIN", "SI"], ["OPTIN", "TWIN:a"], ["OPTIN", "IP-FLAG:masked"]):
        assert not P.train_ok(bad, branch="optin-v1")
    with pytest.raises(ValueError):
        P.train_ok(["ENGINE"], branch="dev")


def test_record_vs_iterable_equivalent():
    tags = ["COMP:seedance", "ENGINE"]
    assert P.train_ok(tags) == P.train_ok(P.as_record(tags)) == P.train_ok(frozenset(tags))
    assert P.why_not_dev(tags) == P.why_not_dev(P.as_record(tags))


def test_single_string_is_not_tag_set():
    for bad in ("USER", b"USER", None):
        with pytest.raises(TypeError):
            P.train_ok(bad)
        with pytest.raises(TypeError):
            P.as_record(bad)


def test_unknown_tag_fails_closed():
    with pytest.raises(P.TagError):
        P.train_ok(["ENGINE", "MYSTERY"])
