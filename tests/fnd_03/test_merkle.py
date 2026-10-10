"""FND-03 blake3 해시·Merkle 검증."""
import itertools

import numpy as np
import pytest
import torch

from buc.core import det


def b3(data):
    import blake3
    return blake3.blake3(data).digest()


def test_blake3_dependency_present_and_known_vectors():
    assert det.hash_bytes(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    assert det.hash_bytes(b"abc").hex() == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
    assert len(det.hash_bytes(b"x")) == 32 and isinstance(det.hash_bytes(b"x"), bytes)
    assert det.hash_bytes(memoryview(b"abc")) == det.hash_bytes(b"abc")


def test_hash_parts_has_unambiguous_framing():
    assert det.hash_parts(b"ab", b"c") != det.hash_parts(b"a", b"bc")
    assert det.hash_parts("ab", "c") != det.hash_parts("a", "bc")
    assert det.hash_parts(b"a") != det.hash_parts("a")  # bytes와 str은 다른 타입 태그
    assert det.hash_parts(1) != det.hash_parts("1") and det.hash_parts(1) != det.hash_parts(b"\x01")
    assert det.hash_parts() != det.hash_parts(b"")
    assert det.hash_parts(1, 2, b"x") == det.hash_parts(1, 2, b"x")
    assert det.hash_parts(-1) != det.hash_parts(1)
    with pytest.raises(TypeError):
        det.hash_parts(1.5)
    with pytest.raises(OverflowError):
        det.hash_parts(1 << 64)  # int는 int64/uint64 범위


def test_hash_array_canonical_and_sensitive():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    h = det.hash_array(a)
    assert len(h) == 32
    assert h == det.hash_array(np.ascontiguousarray(a))
    assert h == det.hash_array(np.asfortranarray(a))  # 메모리 배치가 아니라 논리 값이 기준
    assert h == det.hash_array(torch.from_numpy(a))
    assert h == det.hash_array(a.astype(">f4"))  # 바이트 순서와 무관
    assert h != det.hash_array(a.reshape(6, 4))  # shape 포함
    assert h != det.hash_array(a.astype(np.float64))  # dtype 포함
    assert h != det.hash_array(a.reshape(-1).view(np.int32).reshape(4, 6))
    b = a.copy()
    b[3, 5] += 1
    assert h != det.hash_array(b)
    assert det.hash_array(a[:, ::2]) == det.hash_array(np.ascontiguousarray(a[:, ::2]))
    assert det.hash_array(np.zeros((0, 3), np.float32)) != det.hash_array(np.zeros((3, 0), np.float32))
    assert det.hash_array(np.float32(1)) == det.hash_array(np.array(1, np.float32))


def test_hash_array_rejects_object_dtype():
    with pytest.raises(TypeError):
        det.hash_array(np.array([object()], dtype=object))


def test_merkle_small_cases_pinned_by_domain_separation():
    assert det.EMPTY_ROOT == b3(b"buc.merkle.empty") and det.merkle_root([]) == det.EMPTY_ROOT
    x, y, z = b"x", b"y", b"z"
    lx, ly, lz = (b3(b"\x00" + t) for t in (x, y, z))
    assert det.merkle_root([x]) == lx
    assert det.merkle_root([x, y]) == b3(b"\x01" + lx + ly)
    # 홀수 노드는 복제가 아니라 승격된다
    assert det.merkle_root([x, y, z]) == b3(b"\x01" + b3(b"\x01" + lx + ly) + lz)
    assert det.merkle_root([x, y, z]) != det.merkle_root([x, y, z, z])  # CVE-2012-2459형 모호성 없음
    assert det.merkle_leaf(x) == lx and det.merkle_node(lx, ly) == b3(b"\x01" + lx + ly)


def test_merkle_is_order_sensitive_and_second_preimage_safe():
    items = [bytes([i]) * 3 for i in range(9)]
    assert det.merkle_root(items) != det.merkle_root(items[::-1])
    r2 = det.merkle_root(items[:2])
    # 내부 노드 해시를 잎으로 위장할 수 없다
    assert det.merkle_root([r2]) != r2
    assert det.merkle_root([det.merkle_leaf(items[0]) + det.merkle_leaf(items[1])]) != r2
    with pytest.raises(TypeError):
        det.merkle_root(["not bytes"])


@pytest.mark.parametrize("n", list(range(1, 18)) + [31, 32, 33, 100])
def test_merkle_proofs_verify_for_all_indices(n):
    items = [f"item-{i}".encode() for i in range(n)]
    root = det.merkle_root(items)
    for i in range(n):
        proof = det.merkle_proof(items, i)
        assert all(isinstance(p, bytes) and len(p) == 32 for p in proof)
        assert len(proof) <= max(1, int(np.ceil(np.log2(max(n, 2)))))
        assert det.merkle_verify(items[i], i, n, proof, root)
        assert not det.merkle_verify(b"tampered", i, n, proof, root)
        assert not det.merkle_verify(items[i], i, n, proof, b3(b"other root"))
        if n > 1:
            assert not det.merkle_verify(items[i], (i + 1) % n, n, proof, root) or items[i] == items[(i + 1) % n]
            if proof:
                bad = [proof[0][:-1] + bytes([proof[0][-1] ^ 1])] + list(proof[1:])
                assert not det.merkle_verify(items[i], i, n, bad, root)
    with pytest.raises(IndexError):
        det.merkle_proof(items, n)


def test_merkle_root_keyed_is_order_independent_over_8_shuffles():
    """GOV-04 이음매: 수요 순서를 8번 섞어도 루트가 같다."""
    rng = np.random.default_rng(42)
    entries = {f"block/{i:04d}".encode(): det.hash_array(rng.normal(size=(4,)).astype(np.float32)) for i in range(257)}
    root = det.merkle_root_keyed(entries)
    assert len(root) == 32
    keys = list(entries)
    for s in range(8):
        order = rng.permutation(len(keys))
        shuffled = {keys[i]: entries[keys[i]] for i in order}
        assert list(shuffled) != keys
        assert det.merkle_root_keyed(shuffled) == root
        assert det.merkle_root_keyed(list(shuffled.items())) == root  # (key, value) 쌍 목록도 허용
    # 값 하나, 키 하나만 바뀌어도 루트가 변한다
    k0 = keys[0]
    v2 = dict(entries)
    v2[k0] = b3(b"changed")
    assert det.merkle_root_keyed(v2) != root
    k2 = dict(entries)
    k2[b"block/extra"] = k2.pop(k0)
    assert det.merkle_root_keyed(k2) != root
    # str 키는 utf-8 bytes로 취급
    assert det.merkle_root_keyed({"a": b"1", "b": b"2"}) == det.merkle_root_keyed({b"b": b"2", b"a": b"1"})
    assert det.merkle_root_keyed({}) == det.EMPTY_ROOT


def test_merkle_root_keyed_key_value_framing_and_duplicates():
    assert det.merkle_root_keyed({b"ab": b"c"}) != det.merkle_root_keyed({b"a": b"bc"})
    with pytest.raises(ValueError):
        det.merkle_root_keyed([(b"k", b"1"), (b"k", b"2")])  # 같은 키 중복은 모호하므로 금지
    with pytest.raises(ValueError):
        det.merkle_root_keyed([(b"k", b"1"), ("k", b"1")])  # str/bytes로 정규화하면 같은 키


def test_all_permutations_small_keyed():
    pairs = [(b"a", b"1"), (b"b", b"2"), (b"c", b"3"), (b"d", b"4")]
    roots = {det.merkle_root_keyed(list(p)) for p in itertools.permutations(pairs)}
    assert len(roots) == 1
