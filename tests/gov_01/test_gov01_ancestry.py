"""GOV-01 Lineage CI 규칙 1·8: 조상 폐포의 BLOCK_TRAIN/BLOCK_DEV, 카나리아 C1·C5, 금지목록, 무작위 DAG."""
import random

import pytest

from buc.gov import lineage as L
from buc.gov import prov as P


def _chain(src_tag, depth=6):
    """ENGINE 소스 + 오염 소스 하나가 깊이 depth 사슬의 중간에 섞인 DAG."""
    lg = L.LineageLedger()
    lg.add("eng", kind="source", prov=["ENGINE"])
    lg.add("cap", kind="source", prov=["CAPTURE"])
    lg.add("bad", kind="source", prov=[src_tag])
    prev = "eng"
    for k in range(depth):
        ps = [prev, "cap"] if k != 2 else [prev, "bad"]
        lg.add(f"op{k}", kind="op", parents=ps)
        prev = f"op{k}"
    lg.add("shard", kind="shard", parents=[prev], content_hash="a" * 64)
    lg.add("ckpt", kind="checkpoint", parents=["shard", "eng"])
    lg.add("clean_shard", kind="shard", parents=["eng", "cap"])
    return lg


def test_clean_artifact_passes():
    lg = _chain("ENGINE")
    assert lg.why_not("ckpt") == ()
    assert lg.why_not("ckpt", purpose="dev") == ()
    lg.assert_ok("ckpt")
    assert lg.check_all() == {}


@pytest.mark.parametrize("tag", ["USER", "EXTGEN:test", "USER:soul", "SOUL", "COMP:vendor",
                                 "HOLDOUT", "D3", "NC", "ASSET:CC-BY-NC", "IP-FLAG", "OPEN-GEN:m:h:bench"])
def test_canary_deep_ancestor_blocks_training(tag):
    lg = _chain(tag)
    why = lg.why_not("ckpt")
    assert why, tag
    assert list(why) == sorted(why)
    norm = P.normalize_tag(tag)
    assert f"bad:{norm}" in why  # 소스 자체가 사유로 나온다
    # 사유는 모두 "<폐포 노드>:<사유>" 형식이고 노드는 폐포 안
    closure = set(lg.ancestors("ckpt")) | {"ckpt"}
    for w in why:
        assert w.split(":", 1)[0] in closure
    with pytest.raises(P.FirewallError) as ei:
        lg.assert_ok("ckpt")
    assert norm in str(ei.value)
    # 무관한 형제 아티팩트는 통과
    assert lg.why_not("clean_shard") == ()
    bad_arts = lg.check_all()
    assert set(bad_arts) == {"shard", "ckpt"}
    assert bad_arts["ckpt"] == why


def test_c1_user_propagates_as_user_derived():
    lg = _chain("USER")
    why = lg.why_not("ckpt")
    assert "bad:USER" in why
    assert "op2:USER-DERIVED" in why and "ckpt:USER-DERIVED" in why
    assert lg.node("ckpt").prov.ns == "user_vault"


def test_c5_extgen_blocks_dev_purpose():
    lg = _chain("EXTGEN:test")
    why = lg.why_not("ckpt", purpose="dev")
    assert "bad:EXTGEN:test" in why
    assert "ckpt:EXTGEN-DERIVED:test" in why
    with pytest.raises(P.FirewallError):
        lg.assert_ok("ckpt", purpose="dev")


def test_dev_vs_train_split():
    # COMMISSION:train은 학습 가능, 개발 금지 / OPEN-GEN dev-eval은 반대
    lg = _chain("COMMISSION:train")
    assert lg.why_not("ckpt") == ()
    assert "bad:COMMISSION:train" in lg.why_not("ckpt", purpose="dev")
    lg = _chain("OPEN-GEN:m:h:dev-eval")
    assert lg.why_not("ckpt", purpose="dev") == ()
    assert "bad:OPEN-GEN:m:h:dev-eval" in lg.why_not("ckpt")
    with pytest.raises(ValueError):
        lg.why_not("ckpt", purpose="eval")
    with pytest.raises(ValueError):
        lg.why_not("ckpt", purpose="dev", branch="optin-v1")
    with pytest.raises(KeyError):
        lg.why_not("nope")


def test_optin_branch():
    lg = _chain("OPTIN")
    assert "bad:OPTIN" in lg.why_not("ckpt")
    assert lg.why_not("ckpt", branch="optin-v1") == ()
    lg = _chain("SI")  # SI는 optin-v1 허용 목록 밖
    assert lg.why_not("ckpt") == ()
    assert "bad:SI" in lg.why_not("ckpt", branch="optin-v1")
    with pytest.raises(ValueError):
        lg.why_not("ckpt", branch="soul-s1")


def test_human_ref_inspired_q17():
    lg = L.LineageLedger()
    lg.add("soul", kind="source", prov=["EXTGEN:soul"])
    insp = P.derive("look", [lg.node("soul").prov], human_ref=True)
    lg.add("preset", kind="data", parents=["soul"], prov=insp)
    why = lg.why_not("preset")
    assert "preset:Q17" in why and "preset:EXTGEN-INSPIRED:soul" in why


def test_denylist_on_content_hash():
    lg = L.LineageLedger()
    tv = "ff5fadd5" + "0" * 56
    lg.add("raft_init", kind="checkpoint", prov=["ENGINE"], content_hash=tv)
    lg.add("raft_ft", kind="checkpoint", parents=["raft_init"], content_hash="c" * 64)
    dl = L.Denylist({"ff5fadd5": "torchvision raft_large C_T_SKHT_V2"})
    assert lg.why_not("raft_ft") == ()  # 금지목록 없이는 태그만
    why = lg.why_not("raft_ft", denylist=dl)
    assert why == ("raft_init:DENYLIST:torchvision raft_large C_T_SKHT_V2",)
    assert set(lg.check_all(denylist=dl)) == {"raft_init", "raft_ft"}
    with pytest.raises(P.FirewallError):
        lg.assert_ok("raft_ft", denylist=dl)


TAGS = ["ENGINE", "CAPTURE", "SI", "HUMAN", "ASSET:CC0", "OPEN-GEN:z:h1:train", "COMMISSION:eval+train",
        "TEL-SYS", "IP-FLAG:masked",
        "USER", "EXTGEN:test", "COMP", "HOLDOUT", "D3", "NC", "ASSET", "IP-FLAG", "OPEN-GEN:g:h:dev-eval",
        "COMMISSION:eval", "OPTIN", "TEL-CAGG", "EVAL-ONLY", "BIO", "PUB-SEALED"]
CLEAN = TAGS[:9]


def test_random_dags_oracle_300():
    rng = random.Random(20261010)
    for trial in range(300):
        lg = L.LineageLedger()
        ids = []
        n_src = rng.randint(1, 4)
        for s in range(n_src):
            pool = CLEAN if rng.random() < 0.7 else TAGS
            tags = rng.sample(pool, rng.randint(1, 2))
            lg.add(f"s{s}", kind="source", prov=tags)
            ids.append(f"s{s}")
        for k in range(rng.randint(1, 8)):
            ps = rng.sample(ids, rng.randint(1, min(3, len(ids))))
            kind = rng.choice(["op", "data", "shard", "checkpoint", "calib"])
            lg.add(f"n{k}", kind=kind, parents=ps)
            ids.append(f"n{k}")
        target = ids[-1]
        closure = set(lg.ancestors(target)) | {target}
        for purpose, fn in (("train", P.train_ok), ("dev", P.dev_ok)):
            oracle_ok = all(fn(lg.node(n).prov) for n in closure)
            got = lg.why_not(target, purpose=purpose)
            assert (got == ()) == oracle_ok, (trial, purpose, got)
            # 소스 수준 사유는 FND-04가 낸 사유와 정확히 같다
            for s in closure & {i for i in ids if i.startswith("s")}:
                exp = P.why_not_train(lg.node(s).prov) if purpose == "train" else P.why_not_dev(lg.node(s).prov)
                assert {w.split(":", 1)[1] for w in got if w.split(":", 1)[0] == s} == set(exp)
            # 단조성: 아티팩트가 통과하면 모든 소스 조상도 통과
            if not got:
                assert all(fn(lg.node(s).prov) for s in closure if s.startswith("s"))
