"""FND-04 태그 문법·별칭·정규화·거부."""
import pytest

from buc.gov import prov as P


@pytest.mark.parametrize("s", [
    "ENGINE", "CAPTURE", "ASSET", "ASSET:CC0", "HUMAN", "SI", "TWIN:c12", "USER", "USER:soul",
    "USER-DERIVED", "BIO", "EXTGEN:soul", "EXTGEN-DERIVED:gpt-image", "EXTGEN-INSPIRED:mj", "COMP",
    "COMP:seedance", "HOLDOUT", "PUB-SEALED", "EVAL-ONLY", "IP-FLAG", "IP-FLAG:masked", "OPTIN",
    "TEL-SYS", "TEL-CAGG", "D3", "NC", "OPEN-GEN:z-image:ab12:train", "OPEN-GEN:g3b:ff00:dev-eval",
    "OPEN-GEN:g4:0a:bench", "COMMISSION:eval", "COMMISSION:eval+train+publish",
])
def test_valid_tags_roundtrip_idempotent(s):
    n = P.normalize_tag(s)
    assert n == s
    assert P.normalize_tag(n) == n
    base, qual = P.parse_tag(s)
    assert base in P.KNOWN_BASES
    if ":" in s:
        assert qual == s.split(":", 1)[1]
    else:
        assert qual is None


@pytest.mark.parametrize("old,new", [
    ("SOUL", "EXTGEN:soul"),
    ("SOUL-DERIVED", "EXTGEN-DERIVED:soul"),
    ("SOUL-INSPIRED", "EXTGEN-INSPIRED:soul"),
    ("COMMISSION:publish+eval", "COMMISSION:eval+publish"),
    ("COMMISSION:train+eval", "COMMISSION:eval+train"),
])
def test_aliases(old, new):
    assert P.normalize_tag(old) == new
    assert P.as_record([old]) == P.as_record([new])


@pytest.mark.parametrize("s", [
    "", "FOO", "user", "PARTNER", "EXTGEN", "EXTGEN:", "EXTGEN-DERIVED", "EXTGEN-INSPIRED", "TWIN",
    "OPEN-GEN", "OPEN-GEN:z:train", "OPEN-GEN:z:h:prod", "OPEN-GEN:z::train", "OPEN-GEN:z:h:train:x",
    "COMMISSION", "COMMISSION:sell", "COMMISSION:eval+eval", "COMMISSION:", "IP-FLAG:hidden",
    "ENGINE:", "EXTGEN:so ul", " ENGINE", "ENGINE ",
])
def test_bad_tags_rejected(s):
    with pytest.raises(P.TagError):
        P.normalize_tag(s)


@pytest.mark.parametrize("s", [None, 3, b"ENGINE", ("ENGINE",)])
def test_non_str_tag_typeerror(s):
    with pytest.raises(TypeError):
        P.normalize_tag(s)


def test_errors_hierarchy():
    assert issubclass(P.TagError, ValueError)
    assert issubclass(P.ProvError, ValueError)
    assert issubclass(P.FirewallError, RuntimeError)


def test_constants_exact():
    ext = {"USER", "USER-DERIVED", "BIO", "EXTGEN", "EXTGEN-DERIVED", "EXTGEN-INSPIRED"}
    assert set(P.EXT_BASES) == ext
    assert set(P.BLOCK_TRAIN) == ext | {"COMP", "HOLDOUT", "PUB-SEALED", "EVAL-ONLY", "TEL-CAGG", "OPTIN", "D3", "NC"}
    assert set(P.BLOCK_DEV) == ext | {"COMP", "HOLDOUT", "PUB-SEALED", "EVAL-ONLY", "TEL-CAGG", "OPTIN"}
    assert set(P.BLOCK_DEV) <= set(P.BLOCK_TRAIN)
    assert set(P.TRAIN_LICENSES) == {"CC0", "PD", "CC-BY", "Apache-2.0"}
    assert tuple(P.MODES) == ("none", "inspired", "derived", "authored")
    assert tuple(P.NAMESPACES) == ("user_vault", "ext_vault", "comp_vault", "pub_bench", "train_ok")
    assert P.PROV_TAG == "buc.prov.v1"
    assert set(P.BLOCK_TRAIN) <= set(P.KNOWN_BASES)
