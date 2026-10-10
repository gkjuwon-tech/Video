"""DAT-05 무결성: 변조·손상·주소 불일치 검출, 매니페스트, 삭제용 루트 검색."""
import json
import os
import shutil

import pytest

import dat05_oracle as O
from buc.data import shards as S
from buc.gov import prov as P


def make(tmp_path, n=6, seed=50, **kw):
    with S.ShardWriter(tmp_path, **kw) as w:
        for k, f, p in O.clean_samples(n, seed=seed):
            w.write(k, f, p)
    return w.infos


def test_verify_returns_info_equal_to_writer_info(tmp_path):
    (info,) = make(tmp_path)
    v = S.verify_shard(info.path)
    assert v == info and v.root == O.root_of(O.clean_samples(6, seed=50))
    assert S.verify_shard(info.path, expected_root=info.root) == info


def test_flipped_payload_byte_is_detected(tmp_path):
    (info,) = make(tmp_path)
    raw = bytearray(open(info.path, "rb").read())
    marker = b"PAYLOAD-MARKER-s0003"
    at = bytes(raw).index(marker)
    for off in (0, len(marker) - 1):
        bad = bytearray(raw)
        bad[at + off] ^= 0x01
        open(info.path, "wb").write(bytes(bad))
        with pytest.raises(S.ShardIntegrityError):
            S.verify_shard(info.path)
        with pytest.raises(S.ShardIntegrityError):
            list(S.iter_shard(info.path))
    open(info.path, "wb").write(bytes(raw))
    assert len(list(S.iter_shard(info.path))) == 6


def test_every_sample_payload_is_covered(tmp_path):
    (info,) = make(tmp_path, n=5)
    raw = open(info.path, "rb").read()
    for i in range(5):
        at = raw.index(b"PAYLOAD-MARKER-s%04d" % i)
        bad = bytearray(raw)
        bad[at + 3] ^= 0x20
        open(info.path, "wb").write(bytes(bad))
        with pytest.raises(S.ShardIntegrityError):
            S.verify_shard(info.path)
    open(info.path, "wb").write(raw)


@pytest.mark.parametrize("frac", [0.0, 0.3, 0.5, 0.85, 0.999])
def test_truncated_file_is_a_shard_error(tmp_path, frac):
    (info,) = make(tmp_path)
    raw = open(info.path, "rb").read()
    data_end = raw.rindex(b"PAYLOAD-MARKER-s0005") + 512  # 마지막 페이로드 블록 끝(그 뒤는 tar 끝 표지·패딩)
    open(info.path, "wb").write(raw[: int(data_end * frac)])
    with pytest.raises(S.ShardError):
        S.verify_shard(info.path)
    with pytest.raises(S.ShardError):
        list(S.iter_shard(info.path))


def test_dropping_the_last_sample_cleanly_is_detected_by_address(tmp_path):
    s = O.clean_samples(4, seed=60)
    path, root = O.write_raw(tmp_path, s)
    short = tmp_path / "short"
    short.mkdir()
    # 샘플 3개만 든 올바른 tar를 원래 주소 이름으로 놓는다(루트 불일치)
    open(short / (root.hex() + ".tar"), "wb").write(O.tar_bytes(s[:3]))
    with pytest.raises(S.ShardIntegrityError):
        S.verify_shard(str(short / (root.hex() + ".tar")))
    assert S.verify_shard(path).n == 4


def test_garbage_and_empty_files(tmp_path):
    for i, blob in enumerate((b"", b"hello world", b"\x00" * 1024, b"PK\x03\x04" + b"x" * 600)):
        p = tmp_path / f"g{i}.tar"
        p.write_bytes(blob)
        with pytest.raises(S.ShardError):
            S.verify_shard(str(p))


def test_renamed_shard_fails_address_check_but_expected_root_can_override(tmp_path):
    (info,) = make(tmp_path)
    other = tmp_path / ("0" * 64 + ".tar")
    shutil.copy(info.path, other)
    with pytest.raises(S.ShardIntegrityError):
        S.verify_shard(str(other))
    with pytest.raises(S.ShardIntegrityError):
        list(S.iter_shard(str(other)))
    assert S.verify_shard(str(other), expected_root=info.root).root == info.root
    # 주소 이름이 아니면 비교하지 않는다(그래도 내용은 검증)
    plain = tmp_path / "mydata.tar"
    shutil.copy(info.path, plain)
    assert S.verify_shard(str(plain)).root == info.root
    assert [s.key for s in S.iter_shard(str(plain))] == list(info.keys)
    # 틀린 expected_root는 파일명이 맞아도 실패
    with pytest.raises(S.ShardIntegrityError):
        S.verify_shard(info.path, expected_root=b"\x00" * 32)


def test_uppercase_hex_name_is_not_an_address(tmp_path):
    (info,) = make(tmp_path)
    p = tmp_path / (info.root.hex().upper() + ".tar")
    shutil.copy(info.path, p)
    assert S.verify_shard(str(p)).root == info.root  # 소문자 hex만 주소로 취급


def raw_check(tmp_path, members, name="x"):
    p = tmp_path / f"{name}.tar"
    p.write_bytes(O.tar_from_members(members))
    return str(p)


def good_members():
    return O.members_of(O.clean_samples(2, seed=70))


def test_baseline_member_helper_is_valid(tmp_path):
    p = raw_check(tmp_path, good_members(), "base")
    assert S.verify_shard(p).n == 2


def test_missing_prov_sidecar_is_a_shard_error(tmp_path):
    m = [x for x in good_members() if x[0] != "s0001.prov.json"]
    p = raw_check(tmp_path, m)
    with pytest.raises(S.ShardError):
        S.verify_shard(p)
    with pytest.raises(S.ShardError):
        list(S.iter_shard(p, verify=False))


def test_non_canonical_prov_json_is_rejected_even_if_root_is_consistent(tmp_path):
    s = O.clean_samples(3, seed=71)
    pj = [O.prov_json(p) for _, _, p in s]
    pretty = json.dumps(json.loads(pj[1]), indent=2, sort_keys=True).encode()
    assert pretty != pj[1]
    pj[1] = pretty
    path, root = O.write_raw(tmp_path, s, prov_override=pj)
    assert os.path.basename(path) == root.hex() + ".tar"
    with pytest.raises(S.ShardError):
        S.verify_shard(path)
    with pytest.raises(S.ShardError):
        list(S.iter_shard(path))


def test_prov_json_with_wrong_version_or_garbage_is_rejected(tmp_path):
    s = O.clean_samples(2, seed=72)
    blobs = (b"not json", b'{"v":"buc.prov.v0","taint":["ENGINE"]}', b"[]", b'{"v":"buc.prov.v1","taint":"ENGINE"}')
    for i, blob in enumerate(blobs):
        pj = [O.prov_json(p) for _, _, p in s]
        pj[0] = blob
        path, _ = O.write_raw(tmp_path, s, name="n%d" % i, prov_override=pj)
        with pytest.raises((S.ShardError, P.ProvError, P.TagError)):
            S.verify_shard(path)


def test_duplicate_keys_duplicate_exts_and_empty_samples_are_rejected(tmp_path):
    a = O.members_of(O.clean_samples(1, seed=73, prefix="a"))
    b = O.members_of(O.clean_samples(1, seed=74, prefix="b"))
    cases = {
        "dup_key_far": a + b + a,
        "dup_key_adjacent": a + a,
        "dup_ext": a + [a[0]],
        "prov_only": [x for x in a if x[0].endswith("prov.json")],
        "interleaved": [a[0], b[0], a[1], b[1], a[2], b[2]],
    }
    for name, members in cases.items():
        p = raw_check(tmp_path, members, name)
        with pytest.raises(S.ShardError):
            S.verify_shard(p)


def test_unsafe_member_names_and_types_are_rejected(tmp_path):
    base = good_members()
    for name, bad in {
        "slash": base + [("sub/s9999.txt", b"x"), ("sub/s9999.prov.json", O.prov_json(["ENGINE"]))],
        "dotdot": base + [("../evil.txt", b"x")],
        "nodot": base + [("README", b"x")],
        "dotfile": base + [(".hidden", b"x")],
    }.items():
        p = raw_check(tmp_path, bad, name)
        with pytest.raises(S.ShardError):
            S.verify_shard(p)
    import io
    import tarfile
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in base:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        link = tarfile.TarInfo("s0009.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    p = tmp_path / "link.tar"
    p.write_bytes(bio.getvalue())
    with pytest.raises(S.ShardError):
        S.verify_shard(str(p))


def test_sidecar_manifest_must_agree_if_present(tmp_path):
    (info,) = make(tmp_path)
    man = tmp_path / (info.root.hex() + ".json")
    good = man.read_bytes()
    assert json.loads(good) == info.to_dict()
    assert S.verify_shard(info.path) == info
    man.unlink()
    assert S.verify_shard(info.path) == info  # 매니페스트는 선택
    for field, value in (("n", info.n + 1), ("root", "0" * 64)):
        d = info.to_dict()
        d[field] = value
        man.write_bytes(json.dumps(d, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(S.ShardIntegrityError):
            S.verify_shard(info.path)
    man.write_bytes(good)
    assert S.verify_shard(info.path) == info


def write_with_roots(tmp_path, tag, roots_per_sample):
    with S.ShardWriter(tmp_path) as w:
        for i, roots in enumerate(roots_per_sample):
            rec = P.ProvRecord(taint=frozenset({"ENGINE"}), roots=tuple(roots))
            w.write(f"{tag}{i}", {"txt": f"{tag}{i}".encode()}, rec)
    return w.infos[0]


def test_find_shards_with_root_for_deletion_jobs(tmp_path):
    s1 = write_with_roots(tmp_path, "p", [("img:a",), ("img:b",)])
    s2 = write_with_roots(tmp_path, "q", [("img:b", "img:c")])
    s3 = write_with_roots(tmp_path, "r", [()])
    (tmp_path / "notes.json").write_text('{"hello": 1}')
    (tmp_path / "config.json").write_text("not even json")
    found_b = S.find_shards_with_root(tmp_path, "img:b")
    assert found_b == sorted([s1.path, s2.path])
    assert S.find_shards_with_root(tmp_path, "img:a") == [s1.path]
    assert S.find_shards_with_root(tmp_path, "img:c") == [s2.path]
    assert S.find_shards_with_root(tmp_path, "img:zzz") == []
    assert S.find_shards_with_root(tmp_path, "") == []
    assert s3.path not in found_b


def test_find_shards_with_root_rejects_corrupt_manifest(tmp_path):
    s1 = write_with_roots(tmp_path, "p", [("img:a",)])
    (tmp_path / (s1.root.hex() + ".json")).write_text("{ broken")
    with pytest.raises(S.ShardError):
        S.find_shards_with_root(tmp_path, "img:a")
