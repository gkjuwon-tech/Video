"""DAT-05 방화벽: 쓰기·읽기 양쪽에서 닫힌 쪽 실패. 카나리아 C1(USER)·C5(EXTGEN:test)."""
import glob
import os

import numpy as np
import pytest

import dat05_oracle as O
from buc.data import shards as S
from buc.gov import prov as P

# (태그 집합, train 허용, dev 허용) — 권리 렌즈 §1.2 표(FND-04 진리표)를 손으로 옮긴 것
TABLE = [
    (["ENGINE"], True, True), (["CAPTURE"], True, True), (["SI"], True, True), (["HUMAN"], True, True),
    (["TWIN:c3"], True, True), (["TEL-SYS"], True, True), (["ASSET:CC0"], True, True), (["ASSET:PD"], True, True),
    (["ASSET:CC-BY"], True, True), (["ASSET:Apache-2.0"], True, True), (["OPEN-GEN:m:h:train"], True, True),
    (["COMMISSION:eval+train"], True, True), (["IP-FLAG:masked"], True, True),
    (["ENGINE", "CAPTURE", "ASSET:CC0"], True, True),
    (["COMMISSION:train"], True, False),
    (["D3"], False, True), (["NC"], False, True), (["ASSET"], False, True), (["ASSET:CC-BY-NC"], False, True),
    (["OPEN-GEN:m:h:dev-eval"], False, True), (["COMMISSION:eval"], False, True),
    (["USER"], False, False), (["EXTGEN:test"], False, False), (["SOUL"], False, False), (["SOUL-DERIVED"], False, False),
    (["SOUL-INSPIRED"], False, False), (["USER-DERIVED"], False, False), (["BIO"], False, False),
    (["EXTGEN-DERIVED:g"], False, False), (["COMP:seedance"], False, False), (["HOLDOUT"], False, False),
    (["PUB-SEALED"], False, False), (["EVAL-ONLY"], False, False), (["TEL-CAGG"], False, False),
    (["OPTIN"], False, False), (["IP-FLAG"], False, False), (["OPEN-GEN:m:h:bench"], False, False),
    (["COMMISSION:publish"], False, False), (["ENGINE", "USER"], False, False), (["CAPTURE", "EXTGEN:test"], False, False),
]


def one(tmp_path, tags, purpose="train", branch="main"):
    """샘플 하나를 쓴다. 성공하면 ShardInfo, 방화벽이 막으면 FirewallError를 올린다."""
    out = tmp_path / f"o{len(os.listdir(tmp_path))}"
    with S.ShardWriter(out, purpose=purpose, branch=branch) as w:
        w.write("victim", {"txt": b"payload"}, tags)
    return w.infos[0] if w.infos else None


def test_table_matches_prov_module():
    for tags, tr, dv in TABLE:
        assert P.train_ok(tags) is tr and P.dev_ok(tags) is dv, tags


@pytest.mark.parametrize("tags,train,dev", TABLE, ids=lambda x: "+".join(x) if isinstance(x, list) else str(x))
def test_writer_truth_table(tmp_path, tags, train, dev):
    for purpose, allowed in (("train", train), ("dev", dev)):
        if allowed:
            info = one(tmp_path, tags, purpose)
            assert info.n == 1
        else:
            with pytest.raises(P.FirewallError, match="victim"):
                one(tmp_path, tags, purpose)
    # 막힌 쓰기는 어떤 파일도 남기지 않는다
    for d in glob.glob(str(tmp_path / "o*")):
        for f in os.listdir(d):
            assert f.endswith((".tar", ".json"))


def test_canaries_c1_and_c5_are_rejected_with_reason_in_message(tmp_path):
    with pytest.raises(P.FirewallError, match="USER"):
        one(tmp_path, ["USER"])
    with pytest.raises(P.FirewallError, match="EXTGEN:test"):
        one(tmp_path, ["EXTGEN:test"])
    with pytest.raises(P.FirewallError, match="EXTGEN:soul"):
        one(tmp_path, ["SOUL"])  # 별칭도 정규화돼 걸린다
    assert not glob.glob(str(tmp_path / "*" / "*"))


def test_firewall_error_is_not_swallowed_by_context_manager_and_discards_open_shard(tmp_path):
    with pytest.raises(P.FirewallError):
        with S.ShardWriter(tmp_path / "w") as w:
            w.write("good", {"txt": b"1"}, ["ENGINE"])
            w.write("bad", {"txt": b"2"}, ["USER"])  # 호출자가 잡지 않으면 with를 깨고 열린 샤드는 버려진다
    assert w.infos == [] and not glob.glob(str(tmp_path / "w" / "*"))


def test_rejected_write_does_not_change_writer_state(tmp_path):
    good = O.clean_samples(3, seed=12)
    with S.ShardWriter(tmp_path / "a") as w:
        w.write(*good[0])
        with pytest.raises(P.FirewallError):
            w.write("evil", {"txt": b"x"}, ["USER"])
        w.write(*good[1])
        with pytest.raises(P.FirewallError):
            w.write("evil", {"txt": b"y"}, ["EXTGEN:test"])  # 같은 key를 다시 써도 '중복'이 아니다
        w.write("evil", {"txt": b"ok"}, ["ENGINE"])
        w.write(*good[2])
    (info,) = w.infos
    assert info.keys == (good[0][0], good[1][0], "evil", good[2][0])
    assert [s.key for s in S.iter_shard(info.path)] == list(info.keys)


def test_derive_propagation_cannot_launder_into_a_shard(tmp_path):
    clean = P.derive("crop", [["ENGINE"], ["CAPTURE"]])
    assert one(tmp_path, clean).n == 1
    mixed = P.derive("mix", [["ENGINE"], ["USER"]])
    with pytest.raises(P.FirewallError):
        one(tmp_path, mixed)
    deep = P.derive("grade", [P.derive("crop", [["EXTGEN:g"]])])
    with pytest.raises(P.FirewallError):
        one(tmp_path, deep)
    with pytest.raises(P.FirewallError):
        one(tmp_path, P.derive("look", [["EXTGEN:g"]], human_ref=True))  # EXTGEN-INSPIRED도 막힌다
    # 부모 roots는 샤드 매니페스트까지 전파된다
    a = P.ProvRecord(taint=frozenset({"ENGINE"}), roots=("img:1",))
    b = P.ProvRecord(taint=frozenset({"CAPTURE"}), roots=("img:2",))
    info = one(tmp_path, P.derive("blend", [a, b]))
    assert info.roots == ("img:1", "img:2")


def test_ns_must_be_train_ok_even_if_taint_is_clean(tmp_path):
    # FND-04의 train_ok는 ns를 판정에 쓰지 않는다. 샤드 쓰기는 ns까지 닫는다.
    for ns in ("user_vault", "ext_vault", "comp_vault", "pub_bench"):
        rec = P.ProvRecord(taint=frozenset({"ENGINE"}), ns=ns)
        assert P.train_ok(rec) and P.dev_ok(rec)
        for purpose in ("train", "dev"):
            with pytest.raises(P.FirewallError):
                one(tmp_path, rec, purpose)
    assert one(tmp_path, P.ProvRecord(taint=frozenset({"ENGINE"}), ns="train_ok")).ns == "train_ok"
    # 매핑 형태도 같은 길을 탄다
    d = P.ProvRecord(taint=frozenset({"ENGINE"}), ns="comp_vault").to_dict()
    with pytest.raises(P.FirewallError):
        one(tmp_path, d)


def test_branch_optin_v1(tmp_path):
    assert one(tmp_path, ["OPTIN"], branch="optin-v1").n == 1
    assert one(tmp_path, ["CAPTURE", "ENGINE"], branch="optin-v1").n == 1
    for tags in (["SI"], ["TWIN:c3"], ["USER"], ["EXTGEN:g"], ["TEL-SYS"]):
        with pytest.raises(P.FirewallError):
            one(tmp_path, tags, branch="optin-v1")
    with pytest.raises(P.FirewallError):
        one(tmp_path, ["OPTIN"], branch="main")
    # 읽기도 branch를 따른다
    info = one(tmp_path, ["OPTIN"], branch="optin-v1")
    assert [s.key for s in S.iter_shard(info.path, branch="optin-v1")] == ["victim"]
    with pytest.raises(P.FirewallError):
        list(S.iter_shard(info.path, branch="main"))
    with pytest.raises(ValueError):
        list(S.iter_shard(info.path, purpose="dev", branch="optin-v1"))


def test_random_600_prov_sets_writer_accepts_iff_allowed(tmp_path):
    pool = [t for tags, _, _ in TABLE for t in tags]
    rng = np.random.default_rng(20260610)
    for purpose in ("train", "dev"):
        out = tmp_path / purpose
        accepted = []
        with S.ShardWriter(out, purpose=purpose, max_samples=64) as w:
            for i in range(300):
                tags = sorted({pool[j] for j in rng.integers(0, len(pool), size=int(rng.integers(1, 4)))})
                try:
                    P.as_record(tags)
                except (P.TagError, P.ProvError):
                    continue
                ok = P.train_ok(tags) if purpose == "train" else P.dev_ok(tags)
                key = f"r{i:04d}"
                if ok:
                    w.write(key, {"txt": key.encode()}, tags)
                    accepted.append((key, tags))
                else:
                    with pytest.raises(P.FirewallError):
                        w.write(key, {"txt": key.encode()}, tags)
        assert len(accepted) > 30 and len(accepted) < 300
        back = [(s.key, s.prov) for info in w.infos for s in S.iter_shard(info.path, purpose=purpose)]
        assert [k for k, _ in back] == [k for k, _ in accepted]
        for (_, rec), (_, tags) in zip(back, accepted):
            assert rec == P.as_record(tags)
            assert not rec.ext and not rec.comp and not rec.hold and rec.ns == "train_ok"


# ----- 읽기 쪽: 쓰기 경로를 우회해 만든 샤드(주소·루트가 전부 일관됨)도 막아야 한다 -----

def poisoned(tmp_path, bad_tags, pos=2, n=5):
    s = O.clean_samples(n, seed=21)
    s[pos] = (s[pos][0], s[pos][1], bad_tags)
    path, root = O.write_raw(tmp_path, s)
    return path, root, s


@pytest.mark.parametrize("bad", [["USER"], ["EXTGEN:test"], ["COMP:seedance"], ["HOLDOUT"], ["ASSET"], ["OPEN-GEN:m:h:bench"]])
def test_reader_rejects_bypassed_poisoned_shard_before_yielding_anything(tmp_path, bad):
    path, root, _ = poisoned(tmp_path, bad)
    got = []
    with pytest.raises(P.FirewallError):
        for smp in S.iter_shard(path):
            got.append(smp.key)
    assert got == []  # verify=True: 한 샘플도 내보내지 않는다
    with pytest.raises(P.FirewallError):
        S.verify_shard(path)
    with pytest.raises(P.FirewallError):
        S.verify_shard(path, expected_root=root)  # 주소가 맞아도 방화벽이 먼저 풀리지 않는다


def test_reader_without_verify_still_never_yields_a_violating_sample(tmp_path):
    path, _, s = poisoned(tmp_path, ["USER"], pos=2)
    got = []
    with pytest.raises(P.FirewallError):
        for smp in S.iter_shard(path, verify=False):
            got.append(smp.key)
    assert got == [s[0][0], s[1][0]]


def test_iter_shards_stops_at_first_poisoned_shard(tmp_path):
    good = O.clean_samples(3, seed=31, prefix="g")
    p1, _ = O.write_raw(tmp_path, good)
    bad = O.clean_samples(3, seed=32, prefix="h")
    bad[0] = (bad[0][0], bad[0][1], ["EXTGEN:test"])
    p2, _ = O.write_raw(tmp_path, bad)
    got = []
    with pytest.raises(P.FirewallError):
        for smp in S.iter_shards([p1, p2]):
            got.append(smp.key)
    assert got == [g[0] for g in good]


def test_reader_checks_ns_and_hidden_ext_in_prov_sidecar(tmp_path):
    s = O.clean_samples(2, seed=41)
    vault = P.ProvRecord(taint=frozenset({"ENGINE"}), ns="user_vault")
    s[1] = (s[1][0], s[1][1], vault)
    path, _ = O.write_raw(tmp_path, s)
    with pytest.raises(P.FirewallError):
        list(S.iter_shard(path))


def test_purpose_separation_on_read(tmp_path):
    # dev 목적으로 쓴 샤드(train 금지 태그 포함)는 train으로 읽을 수 없고, dev로는 읽힌다
    with S.ShardWriter(tmp_path / "d", purpose="dev") as w:
        w.write("a", {"txt": b"1"}, ["ENGINE"])
        w.write("b", {"txt": b"2"}, ["ASSET:CC-BY-NC"])
        w.write("c", {"txt": b"3"}, ["OPEN-GEN:m:h:dev-eval"])
    (info,) = w.infos
    assert [s.key for s in S.iter_shard(info.path, purpose="dev")] == ["a", "b", "c"]
    with pytest.raises(P.FirewallError):
        list(S.iter_shard(info.path))
    with pytest.raises(P.FirewallError):
        S.verify_shard(info.path, purpose="train")
    assert S.verify_shard(info.path, purpose="dev") == info
    # 반대로 train 허용인데 dev 금지인 샘플
    with S.ShardWriter(tmp_path / "t") as w2:
        w2.write("x", {"txt": b"1"}, ["COMMISSION:train"])
    assert [s.key for s in S.iter_shard(w2.infos[0].path)] == ["x"]
    with pytest.raises(P.FirewallError):
        list(S.iter_shard(w2.infos[0].path, purpose="dev"))
    for bad in ("vault", "x", None):
        with pytest.raises(ValueError):
            list(S.iter_shard(w2.infos[0].path, purpose=bad))
