"""DAT-05 디스크 배치 계약: 독립 오라클(dat05_oracle)과 비트 동일해야 한다."""
import glob
import json
import os
import tarfile

import numpy as np
import pytest

import dat05_oracle as O
from buc.core import det
from buc.data import shards as S


def write_all(out, samples, **kw):
    with S.ShardWriter(out, **kw) as w:
        for k, f, p in samples:
            w.write(k, f, p)
    return w.infos


def test_writer_output_equals_independent_oracle(tmp_path):
    samples = [
        ("a0", {"png": b"\x89PNG-fake", "npy": O.npy(np.arange(6, dtype=np.int16).reshape(2, 3)), "json": b'{"x":1}'}, ["ENGINE"]),
        ("a1", {"txt": b"zz"}, ["CAPTURE", "ASSET:CC0"]),
        ("b-2_x", {"seg.png": b"\x00", "txt": b""}, P_with_roots(("img:7", "img:3"))),
    ]
    infos = write_all(tmp_path, samples)
    assert len(infos) == 1
    info = infos[0]
    root = O.root_of(samples)
    expected_tar = O.tar_bytes(samples)
    path = tmp_path / (root.hex() + ".tar")
    assert info.root == root and info.address == root.hex() and info.path == str(path)
    assert sorted(os.listdir(tmp_path)) == [root.hex() + ".json", root.hex() + ".tar"]
    assert path.read_bytes() == expected_tar
    assert info.n == 3 and info.keys == ("a0", "a1", "b-2_x")
    assert info.nbytes == len(expected_tar) and info.tar_hash == det.hash_bytes(expected_tar)
    assert info.roots == ("img:3", "img:7") and info.ns == "train_ok" and info.prov_bits == 0
    man = {"v": O.SHARD_TAG, "root": root.hex(), "n": 3, "keys": ["a0", "a1", "b-2_x"], "nbytes": len(expected_tar),
           "tar_blake3": det.hash_bytes(expected_tar).hex(), "roots": ["img:3", "img:7"], "ns": "train_ok", "prov_bits": 0}
    assert info.to_dict() == man
    assert (tmp_path / (root.hex() + ".json")).read_bytes() == json.dumps(man, sort_keys=True, separators=(",", ":")).encode()


def P_with_roots(roots):
    from buc.gov import prov as P
    return P.ProvRecord(taint=frozenset({"ENGINE"}), roots=tuple(roots))


def test_tar_member_metadata_and_webdataset_grouping(tmp_path):
    samples = O.clean_samples(4, seed=1)
    (info,) = write_all(tmp_path, samples)
    with open(info.path, "rb") as fh:
        assert fh.read(512)[257:265] == b"ustar\x0000"  # USTAR 헤더(GNU/PAX 아님)
    with tarfile.open(info.path, "r:") as tf:
        names = []
        for ti in tf.getmembers():
            assert ti.isreg() and ti.mtime == 0 and ti.mode == 0o644
            assert ti.uid == 0 and ti.gid == 0 and ti.uname == "" and ti.gname == ""
            names.append(ti.name)
        # WebDataset 규칙: 첫 '.' 앞이 key, 샘플은 연속, 샘플 안은 ext 오름차순
        groups = []
        for n in names:
            key, ext = n.split(".", 1)
            if not groups or groups[-1][0] != key:
                groups.append((key, []))
            groups[-1][1].append(ext)
        assert [g[0] for g in groups] == [s[0] for s in samples]
        for _, exts in groups:
            assert exts == sorted(exts) == ["npy", "prov.json", "txt"]
        for ti in tf.getmembers():
            if ti.name.endswith(".prov.json"):
                assert tf.extractfile(ti).read() == O.prov_json(["ENGINE"])
        # prov 사이드카는 정규 JSON(공백 없음, 키 정렬, ASCII)
        assert O.prov_json(["ENGINE"]) == b'{"mode":"none","ns":"train_ok","roots":[],"taint":["ENGINE"],"v":"buc.prov.v1"}'


def test_golden_sample_digests_and_root():
    s = [("s0", {"txt": b"hello"}, ["ENGINE"]), ("s1", {"npy": b"\x00\x01", "txt": b"x"}, ["CAPTURE", "ASSET:CC0"])]
    d = O.digests(s)
    assert d[0].hex() == "8362a8842da09a9c129e631df9a33b3d17b560b582a7d5c136d27d4de091b76e"
    assert d[1].hex() == "80740b4af2be840b83fa7f7d3a2ae5a5c2ba8a842cc437315ecd9442a0fd859d"
    assert O.root_of(s).hex() == "1277ccac11af93714d09fd76d62e406cd25100308da4e46bcd3e3d2ce1c50078"


def test_writer_root_matches_golden(tmp_path):
    s = [("s0", {"txt": b"hello"}, ["ENGINE"]), ("s1", {"npy": b"\x00\x01", "txt": b"x"}, ["CAPTURE", "ASSET:CC0"])]
    (info,) = write_all(tmp_path, s)
    assert info.root.hex() == "1277ccac11af93714d09fd76d62e406cd25100308da4e46bcd3e3d2ce1c50078"
    assert os.path.basename(info.path) == info.root.hex() + ".tar"


def test_same_input_same_bytes_in_two_directories(tmp_path):
    samples = O.clean_samples(9, seed=3)
    d1, d2 = tmp_path / "x", tmp_path / "y"
    (i1,) = write_all(d1, samples)
    (i2,) = write_all(d2, samples)
    assert os.path.basename(i1.path) == os.path.basename(i2.path)
    with open(i1.path, "rb") as a, open(i2.path, "rb") as b:
        assert a.read() == b.read()
    assert i1.root == i2.root and i1.tar_hash == i2.tar_hash


def test_out_dir_is_created(tmp_path):
    out = tmp_path / "a" / "b" / "c"
    (info,) = write_all(out, O.clean_samples(2))
    assert os.path.isfile(info.path)


def test_root_is_sensitive_to_every_component(tmp_path):
    base = [("k0", {"txt": b"aaa", "npy": b"n"}, ["ENGINE"]), ("k1", {"txt": b"bbb"}, ["ENGINE"])]

    def root(s):
        out = tmp_path / ("r%d" % len(os.listdir(tmp_path)))
        (info,) = write_all(out, s)
        return info.root

    r0 = root(base)
    variants = {
        "payload": [("k0", {"txt": b"aab", "npy": b"n"}, ["ENGINE"]), base[1]],
        "key": [("k9", {"txt": b"aaa", "npy": b"n"}, ["ENGINE"]), base[1]],
        "ext": [("k0", {"txt2": b"aaa", "npy": b"n"}, ["ENGINE"]), base[1]],
        "prov": [("k0", {"txt": b"aaa", "npy": b"n"}, ["CAPTURE"]), base[1]],
        "extra_field": [("k0", {"txt": b"aaa", "npy": b"n", "bin": b""}, ["ENGINE"]), base[1]],
        "order": [base[1], base[0]],
        "drop": [base[0]],
        "field_move": [("k0", {"txt": b"aaan"}, ["ENGINE"]), base[1]],
    }
    roots = {name: root(s) for name, s in variants.items()}
    assert len({r0, *roots.values()}) == len(variants) + 1
    for name, s in variants.items():
        assert roots[name] == O.root_of(s), name


def test_field_insertion_order_does_not_matter(tmp_path):
    a = [("k", {"txt": b"1", "npy": b"2", "bin": b"3"}, ["ENGINE"])]
    b = [("k", {"bin": b"3", "npy": b"2", "txt": b"1"}, ["ENGINE"])]
    (ia,) = write_all(tmp_path / "a", a)
    (ib,) = write_all(tmp_path / "b", b)
    assert ia.root == ib.root
    with open(ia.path, "rb") as f1, open(ib.path, "rb") as f2:
        assert f1.read() == f2.read()


def test_prov_alias_and_container_forms_give_same_shard(tmp_path):
    from buc.gov import prov as P
    forms = [["CAPTURE", "ASSET:CC0"], ("ASSET:CC0", "CAPTURE"), {"CAPTURE", "ASSET:CC0"},
             P.as_record(["CAPTURE", "ASSET:CC0"]), P.as_record(["CAPTURE", "ASSET:CC0"]).to_dict()]
    roots = set()
    for i, p in enumerate(forms):
        (info,) = write_all(tmp_path / str(i), [("k", {"txt": b"v"}, p)])
        roots.add(info.root)
    assert len(roots) == 1


def test_npy_helpers_are_bit_exact_and_canonical():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 4, 3)).astype(np.float32)
    b = npy_encode_roundtrip(a)
    assert b.dtype == a.dtype and b.shape == a.shape and a.tobytes() == b.tobytes()
    enc = S.npy_encode(a)
    assert enc[:6] == b"\x93NUMPY" and enc == O.npy(a) and S.npy_encode(a) == enc
    fa = np.asfortranarray(a)
    assert S.npy_encode(fa) == enc  # 메모리 배치와 무관한 C순서 정규형
    for arr in (np.arange(7, dtype=np.uint8), np.float64(3.5) * np.ones((2, 2)), np.array(5, dtype=np.int64),
                np.zeros((0, 3), dtype=np.float16)):
        out = S.npy_decode(S.npy_encode(arr))
        assert out.dtype == arr.dtype and out.shape == arr.shape and out.tobytes() == arr.tobytes()
    with pytest.raises(ValueError):
        S.npy_encode(np.array([object(), 1], dtype=object))


def npy_encode_roundtrip(a):
    return S.npy_decode(S.npy_encode(a))


def test_dataset_root_is_a_keyed_merkle_set_and_order_free():
    rs = [det.hash_bytes(bytes([i])) for i in range(9)]
    base = S.dataset_root(rs)
    assert base == det.merkle_root_keyed({r.hex(): r for r in rs})
    rng = np.random.default_rng(5)
    for _ in range(8):
        perm = list(rng.permutation(len(rs)))
        assert S.dataset_root([rs[i] for i in perm]) == base
    assert S.dataset_root(rs[:-1]) != base
    assert S.dataset_root([]) == det.EMPTY_ROOT
    with pytest.raises(ValueError):
        S.dataset_root(rs + [rs[0]])
    assert S.dataset_root(iter(rs)) == base  # 반복자도 받는다


def test_glob_sanity_no_stray_files(tmp_path):
    write_all(tmp_path, O.clean_samples(3), max_samples=2)
    assert not glob.glob(str(tmp_path / "*.tmp*")) and not glob.glob(str(tmp_path / ".*"))
    assert len(glob.glob(str(tmp_path / "*.tar"))) == 2 and len(glob.glob(str(tmp_path / "*.json"))) == 2
