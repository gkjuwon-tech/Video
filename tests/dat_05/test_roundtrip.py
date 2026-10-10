"""DAT-05 쓰기·읽기 왕복, 롤오버, 중단, 입력 검증."""
import glob
import os

import numpy as np
import pytest

import dat05_oracle as O
from buc.core import det
from buc.data import shards as S
from buc.gov import prov as P


def tars(d):
    return sorted(glob.glob(os.path.join(str(d), "*.tar")))


def test_roundtrip_bit_exact(tmp_path):
    rng = np.random.default_rng(11)
    samples = []
    for i in range(12):
        arr = rng.standard_normal((3, 8, 8)).astype(np.float32)
        img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        prov = [["ENGINE"], ["CAPTURE", "ASSET:CC0"], ["SI"], P.ProvRecord(taint=frozenset({"HUMAN"}), roots=(f"r{i}",))][i % 4]
        samples.append((f"smp{i:03d}", {"npy": S.npy_encode(arr), "u8": S.npy_encode(img), "json": b'{"i":%d}' % i}, prov))
    with S.ShardWriter(tmp_path) as w:
        for k, f, p in samples:
            w.write(k, f, p)
    (info,) = w.infos
    got = list(S.iter_shard(info.path))
    assert [g.key for g in got] == [s[0] for s in samples]
    for g, (k, f, p) in zip(got, samples):
        assert isinstance(g, S.Sample) and g.prov == P.as_record(p)
        assert set(g.fields) == set(f) and "prov.json" not in g.fields
        for e in f:
            assert isinstance(g.fields[e], bytes) and g.fields[e] == f[e]
    a = S.npy_decode(got[5].fields["npy"])
    assert a.dtype == np.float32 and a.tobytes() == S.npy_decode(samples[5][1]["npy"]).tobytes()
    assert list(S.iter_shard(info.path, verify=False)) == got
    assert S.verify_shard(info.path) == info


def test_rollover_by_sample_count(tmp_path):
    samples = O.clean_samples(25, seed=2)
    with S.ShardWriter(tmp_path, max_samples=10) as w:
        for k, f, p in samples:
            w.write(k, f, p)
        assert len(w.infos) == 2  # 10, 10 은 이미 닫혔다
    assert [i.n for i in w.infos] == [10, 10, 5]
    assert len(tars(tmp_path)) == 3
    keys = [s.key for i in w.infos for s in S.iter_shard(i.path)]
    assert keys == [s[0] for s in samples]
    for i, chunk in zip(w.infos, (samples[:10], samples[10:20], samples[20:])):
        assert i.root == O.root_of(chunk) and i.keys == tuple(c[0] for c in chunk)
    assert len({i.root for i in w.infos}) == 3
    assert [s.key for s in S.iter_shards([i.path for i in w.infos])] == keys


def test_exact_multiple_has_no_empty_trailing_shard(tmp_path):
    with S.ShardWriter(tmp_path, max_samples=5) as w:
        for k, f, p in O.clean_samples(10):
            w.write(k, f, p)
    assert [i.n for i in w.infos] == [5, 5] and len(tars(tmp_path)) == 2


def test_rollover_by_payload_bytes(tmp_path):
    out = []
    for i in range(5):
        out.append((f"b{i}", {"bin": bytes([i]) * 60}, ["ENGINE"]))
    with S.ShardWriter(tmp_path, max_bytes=100) as w:
        for k, f, p in out:
            w.write(k, f, p)
    assert [i.n for i in w.infos] == [2, 2, 1]  # 60 < 100 <= 120
    with S.ShardWriter(tmp_path / "exact", max_bytes=120) as w2:
        for k, f, p in out:
            w2.write(k, f, p)
    assert [i.n for i in w2.infos] == [2, 2, 1]  # 합이 정확히 max_bytes면 닫는다
    with S.ShardWriter(tmp_path / "multi", max_bytes=120, max_samples=1) as w3:
        for k, f, p in out[:3]:
            w3.write(k, f, p)
    assert [i.n for i in w3.infos] == [1, 1, 1]


def test_empty_writer_writes_nothing(tmp_path):
    with S.ShardWriter(tmp_path / "e") as w:
        pass
    assert w.infos == []
    assert not glob.glob(str(tmp_path / "e" / "*"))
    w2 = S.ShardWriter(tmp_path / "e2")
    assert w2.close() == []


def test_close_returns_infos_and_blocks_further_writes(tmp_path):
    w = S.ShardWriter(tmp_path)
    for k, f, p in O.clean_samples(3):
        w.write(k, f, p)
    infos = w.close()
    assert infos == w.infos and len(infos) == 1 and infos[0].n == 3
    with pytest.raises(RuntimeError):
        w.write("late", {"txt": b"x"}, ["ENGINE"])


def test_exception_in_with_discards_open_shard_only(tmp_path):
    samples = O.clean_samples(5, seed=4)
    with pytest.raises(KeyError):
        with S.ShardWriter(tmp_path, max_samples=2) as w:
            for k, f, p in samples[:3]:
                w.write(k, f, p)  # 처음 2개는 닫힌 샤드, 3번째는 열린 샤드
            raise KeyError("boom")
    left = sorted(os.listdir(tmp_path))
    root = O.root_of(samples[:2])
    assert left == [root.hex() + ".json", root.hex() + ".tar"]
    assert [i.n for i in w.infos] == [2]
    # 닫힌 샤드는 온전하다
    assert S.verify_shard(os.path.join(tmp_path, root.hex() + ".tar")).root == root


def test_exception_before_any_shard_leaves_directory_empty(tmp_path):
    with pytest.raises(RuntimeError):
        with S.ShardWriter(tmp_path / "z") as w:
            w.write("k", {"txt": b"x"}, ["ENGINE"])
            raise RuntimeError("x")
    assert not glob.glob(str(tmp_path / "z" / "*"))


def test_rewriting_same_shard_is_idempotent_but_conflict_is_detected(tmp_path):
    samples = O.clean_samples(4, seed=8)
    (i1,) = _write(tmp_path, samples)
    before = open(i1.path, "rb").read()
    (i2,) = _write(tmp_path, samples)
    assert i2 == i1 and open(i1.path, "rb").read() == before
    # 같은 이름에 다른 바이트가 이미 있으면 덮어쓰지 않고 실패
    other = tmp_path / "o"
    other.mkdir()
    (other / (i1.root.hex() + ".tar")).write_bytes(b"not the same")
    with pytest.raises(S.ShardIntegrityError):
        _write(other, samples)
    assert (other / (i1.root.hex() + ".tar")).read_bytes() == b"not the same"


def _write(d, samples, **kw):
    with S.ShardWriter(d, **kw) as w:
        for k, f, p in samples:
            w.write(k, f, p)
    return w.infos


@pytest.mark.parametrize("key,exc", [
    ("", ValueError), ("a.b", ValueError), ("a/b", ValueError), ("-lead", ValueError), ("sp ace", ValueError),
    ("é", ValueError), ("a\n", ValueError), ("k" * 65, ValueError), (b"bytes", TypeError), (7, TypeError), (None, TypeError),
])
def test_bad_keys_rejected(tmp_path, key, exc):
    w = S.ShardWriter(tmp_path)
    with pytest.raises(exc):
        w.write(key, {"txt": b"x"}, ["ENGINE"])
    assert w.close() == []


def test_good_key_boundaries(tmp_path):
    keys = ["a", "A9", "0x", "a_b-c", "k" * 64]
    infos = _write(tmp_path, [(k, {"txt": b"x"}, ["ENGINE"]) for k in keys])
    assert infos[0].keys == tuple(keys)


@pytest.mark.parametrize("ext", ["", ".png", "a b", "a/b", "prov.json", "a..b", "x" * 33, "é", "png\n"])
def test_bad_exts_rejected(tmp_path, ext):
    w = S.ShardWriter(tmp_path)
    with pytest.raises(ValueError):
        w.write("k", {ext: b"x"}, ["ENGINE"])
    with pytest.raises(ValueError):
        w.write("k", {"txt": b"x", ext: b"y"}, ["ENGINE"])
    assert w.close() == []


def test_good_ext_forms(tmp_path):
    infos = _write(tmp_path, [("k", {"seg.png": b"1", "a_b": b"2", "x" * 32: b"3", "prov": b"4", "json": b"5"}, ["ENGINE"])])
    (s,) = list(S.iter_shard(infos[0].path))
    assert set(s.fields) == {"seg.png", "a_b", "x" * 32, "prov", "json"}


def test_bad_field_payloads_and_containers(tmp_path):
    w = S.ShardWriter(tmp_path)
    for bad in ("text", 5, None, np.zeros(3), [1, 2]):
        with pytest.raises(TypeError):
            w.write("k", {"txt": bad}, ["ENGINE"])
    with pytest.raises(ValueError):
        w.write("k", {}, ["ENGINE"])
    for bad in (None, [("txt", b"x")], b"x"):
        with pytest.raises(TypeError):
            w.write("k", bad, ["ENGINE"])
    assert w.close() == []


def test_bytes_like_payloads_are_accepted_and_returned_as_bytes(tmp_path):
    (info,) = _write(tmp_path, [("k", {"a": bytearray(b"abc"), "b": memoryview(b"def"), "c": b""}, ["ENGINE"])])
    (s,) = list(S.iter_shard(info.path))
    assert s.fields == {"a": b"abc", "b": b"def", "c": b""}
    assert all(type(v) is bytes for v in s.fields.values())


def test_prov_is_required_and_has_no_default(tmp_path):
    w = S.ShardWriter(tmp_path)
    for bad in (None, "ENGINE", b"ENGINE", 3):
        with pytest.raises(TypeError):
            w.write("k", {"txt": b"x"}, bad)
    with pytest.raises(TypeError):
        w.write("k", {"txt": b"x"})
    with pytest.raises((P.TagError, P.ProvError)):
        w.write("k", {"txt": b"x"}, ["NOT-A-TAG"])
    assert w.close() == []


def test_duplicate_keys_rejected_for_the_whole_writer_lifetime(tmp_path):
    w = S.ShardWriter(tmp_path, max_samples=2)
    w.write("a", {"txt": b"1"}, ["ENGINE"])
    with pytest.raises(ValueError):
        w.write("a", {"txt": b"2"}, ["ENGINE"])
    w.write("b", {"txt": b"1"}, ["ENGINE"])  # 여기서 샤드가 닫힌다
    with pytest.raises(ValueError):
        w.write("a", {"txt": b"3"}, ["ENGINE"])  # 롤오버 뒤에도 중복
    w.write("c", {"txt": b"1"}, ["ENGINE"])
    assert [i.keys for i in w.close()] == [("a", "b"), ("c",)]


def test_rejected_writes_leave_no_trace(tmp_path):
    good = O.clean_samples(3, seed=6)
    (ref,) = _write(tmp_path / "ref", good)
    w = S.ShardWriter(tmp_path / "x")
    w.write(*good[0])
    for bad in [("k", {}, ["ENGINE"]), ("a.b", {"txt": b"1"}, ["ENGINE"]), ("kk", {"txt": "no"}, ["ENGINE"]),
                ("kk", {"txt": b"1"}, None)]:
        with pytest.raises((ValueError, TypeError)):
            w.write(*bad)
    w.write(*good[1])
    w.write(*good[2])
    (info,) = w.close()
    assert info.root == ref.root and open(info.path, "rb").read() == open(ref.path, "rb").read()


@pytest.mark.parametrize("kw", [
    {"max_samples": 0}, {"max_samples": -1}, {"max_bytes": 0}, {"max_bytes": -5}, {"purpose": "vault"}, {"purpose": "x"},
    {"branch": "dev"}, {"purpose": "dev", "branch": "optin-v1"},
])
def test_bad_writer_arguments(tmp_path, kw):
    with pytest.raises(ValueError):
        S.ShardWriter(tmp_path, **kw)


def test_shardinfo_roots_union_and_invariants(tmp_path):
    recs = [P.ProvRecord(taint=frozenset({"ENGINE"}), roots=("z", "a")), P.ProvRecord(taint=frozenset({"CAPTURE"}), roots=("a", "m")),
            ["SI"]]
    (info,) = _write(tmp_path, [(f"k{i}", {"txt": b"x"}, r) for i, r in enumerate(recs)])
    assert info.roots == ("a", "m", "z") and info.prov_bits == 0 and info.ns == "train_ok"
    assert S.verify_shard(info.path).roots == info.roots


def test_iter_shards_is_ordered_and_chains(tmp_path):
    infos = _write(tmp_path, O.clean_samples(7, seed=9), max_samples=3)
    paths = [i.path for i in infos]
    fwd = [s.key for s in S.iter_shards(paths)]
    rev = [s.key for s in S.iter_shards(paths[::-1])]
    assert sorted(fwd) == sorted(rev) and fwd != rev
    assert fwd == [k for i in infos for k in i.keys]
    assert list(S.iter_shards([])) == []
    assert len(S.dataset_root([i.root for i in infos])) == 32
    assert det.merkle_root([]) == det.EMPTY_ROOT
