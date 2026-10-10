"""FND-03 바이트 배치 고정. 해시 키는 디스크·캐시·원장에 남으므로 구현 상세가 아니라 계약이다.
각 테스트는 계약서의 배치를 독립적으로 다시 짜서 det의 결과와 비교한다."""
import struct

import numpy as np
import pytest
import torch

from buc.core import det

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK32 = 0xFFFFFFFF


def ref_philox(ctr, key, rounds=10):
    """Random123 philox4x32 순수 파이썬 오라클(test_philox와 독립 복제: 테스트 파일끼리 import하지 않는다)."""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for _ in range(rounds):
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & MASK32, p1 & MASK32, ((p0 >> 32) ^ c3 ^ k1) & MASK32, p0 & MASK32
        k0, k1 = (k0 + W0) & MASK32, (k1 + W1) & MASK32
    return (c0, c1, c2, c3)


def b3(data):
    import blake3
    return blake3.blake3(data).digest()


def u64(n):
    return struct.pack("<Q", n)


def test_hash_parts_layout_is_pinned():
    expect = b3(b"buc.parts.v1" + struct.pack("<I", 4)
                + b"b" + u64(3) + b"abc"
                + b"s" + u64(3) + "한".encode("utf-8")
                + b"i" + u64(16) + (-5).to_bytes(16, "little", signed=True)
                + b"i" + u64(16) + ((1 << 64) - 1).to_bytes(16, "little", signed=True))
    assert det.hash_parts(b"abc", "한", -5, (1 << 64) - 1) == expect
    assert det.hash_parts(bytearray(b"abc"), "한", np.int64(-5), np.uint64((1 << 64) - 1)) == expect
    assert det.hash_parts() == b3(b"buc.parts.v1" + struct.pack("<I", 0))
    assert det.hash_parts(-(1 << 63)) != det.hash_parts(1 << 63)  # 양끝 범위 구분
    with pytest.raises(OverflowError):
        det.hash_parts(-(1 << 63) - 1)
    with pytest.raises(TypeError):
        det.hash_parts(True)  # bool은 int와 모호하므로 금지
    with pytest.raises(TypeError):
        det.hash_parts(None)


def test_derive_seed_layout_is_pinned():
    expect = int.from_bytes(det.hash_parts(77, b"x", "y")[:8], "little")
    assert det.derive_seed(77, b"x", "y") == expect
    assert det.derive_seed(77) == int.from_bytes(det.hash_parts(77)[:8], "little")
    with pytest.raises(ValueError):
        det.derive_seed(-1, b"x")
    with pytest.raises(ValueError):
        det.derive_seed(1 << 64, b"x")


def test_hash_array_layout_is_pinned():
    a = np.arange(6, dtype="<i2").reshape(2, 3)
    expect = b3(b"buc.array.v1" + b"<i2" + b"\0" + bytes([2]) + u64(2) + u64(3) + a.tobytes())
    assert det.hash_array(a) == expect
    assert det.hash_array(a.astype(">i2")) == expect
    u8 = np.array([1, 2, 3], dtype=np.uint8)
    assert det.hash_array(u8) == b3(b"buc.array.v1" + b"|u1" + b"\0" + bytes([1]) + u64(3) + b"\x01\x02\x03")
    s = np.float32(2.5)
    assert det.hash_array(s) == b3(b"buc.array.v1" + b"<f4" + b"\0" + bytes([0]) + struct.pack("<f", 2.5))
    bo = np.array([True, False])
    assert det.hash_array(bo) == b3(b"buc.array.v1" + b"|b1" + b"\0" + bytes([1]) + u64(2) + b"\x01\x00")
    # 토치 텐서는 CPU numpy와 같은 해시, 연산 그래프에 묶인 텐서도 허용
    t = torch.arange(6, dtype=torch.int32).reshape(2, 3).float().requires_grad_(True)
    assert det.hash_array(t) == det.hash_array(t.detach().numpy())


def test_merkle_root_keyed_leaf_layout_is_pinned():
    entries = {b"zz": b"\x01" * 32, b"a": b"\x02" * 32, "mid": b"\x03" * 32}
    leaves = [u64(len(k)) + k + v for k, v in sorted([(b"zz", b"\x01" * 32), (b"a", b"\x02" * 32),
                                                      (b"mid", b"\x03" * 32)])]
    assert det.merkle_root_keyed(entries) == det.merkle_root(leaves)
    # 정렬 기준은 키 바이트 사전식(길이가 아님): b"a" < b"mid" < b"zz"
    order = [k for k, _ in sorted([(b"zz", 0), (b"a", 0), (b"mid", 0)])]
    assert order == [b"a", b"mid", b"zz"]
    assert det.merkle_root_keyed([(b"a", b"")]) == det.merkle_root([u64(1) + b"a"])
    with pytest.raises(TypeError):
        det.merkle_root_keyed({b"a": "str value"})
    with pytest.raises(TypeError):
        det.merkle_root_keyed({1: b"v"})


def test_merkle_root_accepts_bytes_like_and_iterators():
    items = [b"a", bytearray(b"b"), memoryview(b"c"), np.frombuffer(b"d", dtype=np.uint8).tobytes()]
    assert det.merkle_root(iter(items)) == det.merkle_root([bytes(x) for x in items])


def test_wnf_extreme_cell_and_level_and_seed_boundaries():
    cell = (1 << 63) - 1
    out = det.wnf((1 << 64) - 1, 0xFFFF, np.int64(cell), k=(1 << 32) - 1, channels=4, dtype=np.float64)
    key = (MASK32, MASK32)
    w = ref_philox((cell & MASK32, cell >> 32, 0xFFFF, MASK32), key)
    expect = []
    for a, b in ((w[0], w[1]), (w[2], w[3])):
        u1, u2 = (a + 0.5) / 2.0**32, (b + 0.5) / 2.0**32
        r = np.sqrt(-2.0 * np.log(u1))
        expect += [r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)]
    np.testing.assert_allclose(out, np.array(expect), rtol=0, atol=1e-9)
    # 블록 번호는 카운터 3번째 워드 상위 16비트: 채널 인덱스 8..11은 3번째 블록(blk=2)
    many = det.wnf(1, 7, np.int64(5), k=2, channels=12, dtype=np.float64)
    w2 = ref_philox((5, 0, 7 | (2 << 16), 2), (1, 0))
    u1 = (w2[0] + 0.5) / 2.0**32
    u2 = (w2[1] + 0.5) / 2.0**32
    r = np.sqrt(-2.0 * np.log(u1))
    assert abs(many[8] - r * np.cos(2 * np.pi * u2)) < 1e-9


def test_wnf_is_pure_function_of_arguments_and_does_not_touch_global_rng():
    np.random.seed(5)
    before = np.random.get_state()[1].copy()
    torch.manual_seed(5)
    t_before = torch.get_rng_state().clone()
    a = det.wnf(5, 1, np.arange(100, dtype=np.int64), k=1)
    assert (np.random.get_state()[1] == before).all()
    assert torch.equal(torch.get_rng_state(), t_before)
    assert a.tobytes() == det.wnf(5, 1, np.arange(100, dtype=np.int64), k=1).tobytes()
    # 입력 배열의 dtype(int32/uint32/int64)이 달라도 같은 정수면 같은 값
    c = np.arange(100)
    assert det.wnf(5, 1, c.astype(np.int32), k=1).tobytes() == a.tobytes()
    assert det.wnf(5, 1, c.astype(np.uint32), k=1).tobytes() == a.tobytes()


def test_philox_requires_uint32():
    with pytest.raises(TypeError):
        det.philox4x32(np.zeros(4, dtype=np.int64), np.zeros(2, dtype=np.uint32))
    with pytest.raises(TypeError):
        det.philox4x32(np.zeros(4, dtype=np.uint32), np.zeros(2, dtype=np.int32))


def test_segment_reduce_int_torch_one_dim_and_no_float_involved():
    q = torch.tensor([5, -3, 7, 1 << 60, 2], dtype=torch.int64)
    seg = torch.tensor([1, 1, 0, 2, 2], dtype=torch.int64)
    out = det.segment_reduce_int_torch(q, seg, 4)
    assert out.dtype == torch.int64 and out.tolist() == [7, 2, (1 << 60) + 2, 0]
    with pytest.raises(TypeError):
        det.segment_reduce_int_torch(q.float(), seg, 4)
    with pytest.raises(ValueError):
        det.segment_reduce_int_torch(q, torch.tensor([0, 1, 2, 3, 4]), 4)  # 범위 밖


def test_segment_reduce_float64_values_and_weights_dtype_paths():
    rng = np.random.default_rng(8)
    n, nseg = 3000, 11
    seg = rng.integers(0, nseg, n)
    v64 = rng.normal(size=(n, 2))
    w64 = rng.uniform(size=n)
    a = det.segment_reduce(v64, seg, nseg, weights=w64, frac_bits=30)
    q = det.fixed_encode(v64 * w64[:, None], 30)
    ref = np.zeros((nseg, 2), dtype=np.int64)
    for i in range(n):
        ref[seg[i]] += q[i]
    assert a.tobytes() == ref.tobytes()
    # frac_bits=1에서 값 1.0은 정수 2: 개수 세기와 일치
    ones = det.segment_reduce(np.ones(n, dtype=np.float32), seg, nseg, frac_bits=1)
    assert (ones == 2 * np.bincount(seg, minlength=nseg)).all()
