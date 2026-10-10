"""FND-03 Philox4x32-10 / WNF 검증. 독립 순수 파이썬 오라클과 Random123 KAT로 판정한다."""
import numpy as np
import pytest
from scipy import stats

from buc.core import det

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def ref_philox(ctr, key, rounds=10):
    """Random123 philox4x32 순수 파이썬 참조 구현(오라클)."""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for _ in range(rounds):
        p0 = M0 * c0
        p1 = M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & MASK, p1 & MASK, ((p0 >> 32) ^ c3 ^ k1) & MASK, p0 & MASK
        k0 = (k0 + W0) & MASK
        k1 = (k1 + W1) & MASK
    return (c0, c1, c2, c3)


# Random123 kat_vectors (philox4x32 10)
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((MASK,) * 4, (MASK, MASK), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr,key,out", KAT)
def test_oracle_matches_random123_kat(ctr, key, out):
    assert ref_philox(ctr, key) == out


@pytest.mark.parametrize("ctr,key,out", KAT)
def test_philox4x32_kat(ctr, key, out):
    got = det.philox4x32(np.array(ctr, dtype=np.uint32), np.array(key, dtype=np.uint32))
    assert got.dtype == np.uint32 and got.shape == (4,)
    assert tuple(int(x) for x in got) == out


def test_philox4x32_vectorized_matches_oracle_and_broadcasts_key():
    rng = np.random.default_rng(1234)
    ctr = rng.integers(0, 2**32, size=(500, 4), dtype=np.uint64).astype(np.uint32)
    key = rng.integers(0, 2**32, size=(2,), dtype=np.uint64).astype(np.uint32)
    got = det.philox4x32(ctr, key)
    assert got.shape == (500, 4)
    for i in range(0, 500, 7):
        assert tuple(int(x) for x in got[i]) == ref_philox([int(x) for x in ctr[i]], [int(x) for x in key])
    # 키가 행별로 다른 경우
    keys = rng.integers(0, 2**32, size=(500, 2), dtype=np.uint64).astype(np.uint32)
    got2 = det.philox4x32(ctr, keys)
    for i in range(0, 500, 11):
        assert tuple(int(x) for x in got2[i]) == ref_philox([int(x) for x in ctr[i]], [int(x) for x in keys[i]])


def test_philox4x32_rounds_and_input_not_mutated():
    ctr = np.array([1, 2, 3, 4], dtype=np.uint32)
    key = np.array([5, 6], dtype=np.uint32)
    c0, k0 = ctr.copy(), key.copy()
    r7 = det.philox4x32(ctr, key, rounds=7)
    assert tuple(int(x) for x in r7) == ref_philox((1, 2, 3, 4), (5, 6), rounds=7)
    assert (ctr == c0).all() and (key == k0).all()
    assert tuple(int(x) for x in r7) != tuple(int(x) for x in det.philox4x32(ctr, key))


def test_philox4x32_rejects_bad_shapes():
    with pytest.raises(ValueError):
        det.philox4x32(np.zeros((3,), dtype=np.uint32), np.zeros(2, dtype=np.uint32))
    with pytest.raises(ValueError):
        det.philox4x32(np.zeros((4,), dtype=np.uint32), np.zeros(3, dtype=np.uint32))


# ---------------------------------------------------------------- WNF

def test_wnf_shape_dtype_and_run_to_run_identity():
    cell = np.arange(1000, dtype=np.int64).reshape(10, 100)
    a = det.wnf(7, 3, cell, k=2, channels=6)
    b = det.wnf(7, 3, cell, k=2, channels=6)
    assert a.shape == (10, 100, 6) and a.dtype == np.float32
    assert a.tobytes() == b.tobytes()
    d = det.wnf(7, 3, cell, k=2, channels=6, dtype=np.float64)
    assert d.dtype == np.float64
    np.testing.assert_allclose(d.astype(np.float32), a, rtol=0, atol=1e-6)
    assert np.isfinite(a).all()


def test_wnf_rho_one_is_layout_independent():
    """ρ=1: 값은 (seed, level, cell, k, 채널)만의 함수다. 배치 크기·순서·분할과 무관해야 한다."""
    big = np.arange(5000, dtype=np.int64)
    full = det.wnf(99, 5, big, k=1, channels=8)
    rng = np.random.default_rng(0)
    pick = rng.permutation(5000)[:333]
    sub = det.wnf(99, 5, big[pick], k=1, channels=8)
    assert sub.tobytes() == full[pick].tobytes()
    one = det.wnf(99, 5, np.int64(4321), k=1, channels=8)
    assert one.shape == (8,)
    assert one.tobytes() == full[4321].tobytes()
    # 채널 수를 줄여도 앞쪽 채널은 같다
    few = det.wnf(99, 5, big[:50], k=1, channels=3)
    assert few.tobytes() == full[:50, :3].tobytes()


def test_wnf_depends_on_every_argument():
    cell = np.arange(64, dtype=np.int64)
    base = det.wnf(1, 2, cell, k=3, channels=4)
    for other in (det.wnf(2, 2, cell, k=3, channels=4), det.wnf(1, 3, cell, k=3, channels=4),
                  det.wnf(1, 2, cell, k=4, channels=4), det.wnf(1, 2, cell + 1, k=3, channels=4)):
        assert other.tobytes() != base.tobytes()
        assert not np.allclose(other, base)
    # 64비트 seed 상·하위 32비트 모두 쓰인다, 64비트 cell 상위 비트도 쓰인다
    s_hi = det.wnf(1 + (1 << 40), 2, cell, k=3)
    assert s_hi.tobytes() != det.wnf(1, 2, cell, k=3).tobytes()
    c_hi = det.wnf(1, 2, cell + (1 << 40), k=3)
    assert c_hi.tobytes() != det.wnf(1, 2, cell, k=3).tobytes()


def test_wnf_counter_layout_is_pinned_to_philox_box_muller():
    """계약서의 카운터 배치와 Box-Muller를 독립 재구현으로 고정한다(이음매: 다른 단위가 같은 노이즈를 재현해야 함)."""
    seed, level, k, cell = 0x1234_5678_9ABC_DEF0, 11, 5, 0x1_0000_0002
    out = det.wnf(seed, level, np.int64(cell), k=k, channels=8, dtype=np.float64)
    key = (seed & MASK, seed >> 32)
    expect = []
    for blk in range(2):
        w = ref_philox((cell & MASK, cell >> 32, (level & 0xFFFF) | (blk << 16), k), key)
        for a, b in ((w[0], w[1]), (w[2], w[3])):
            u1, u2 = (a + 0.5) / 2.0**32, (b + 0.5) / 2.0**32
            r = np.sqrt(-2.0 * np.log(u1))
            expect += [r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)]
    np.testing.assert_allclose(out, np.array(expect), rtol=0, atol=1e-9)


def test_wnf_is_standard_normal():
    cell = np.arange(100_000, dtype=np.int64)
    z = det.wnf(2024, 4, cell, k=0, channels=4, dtype=np.float64)
    flat = z.reshape(-1)
    n = flat.size
    assert abs(flat.mean()) < 5 / np.sqrt(n)
    assert abs(flat.var() - 1.0) < 0.01
    assert abs(stats.skew(flat)) < 0.02
    assert abs(stats.kurtosis(flat)) < 0.05
    assert stats.kstest(flat, "norm").pvalue > 1e-4
    # 인접 셀·채널 간 무상관
    assert abs(np.corrcoef(z[:-1, 0], z[1:, 0])[0, 1]) < 0.01
    assert abs(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) < 0.01
    # 다른 k(스텝)의 노이즈도 무상관
    z2 = det.wnf(2024, 4, cell, k=1, channels=1, dtype=np.float64)[:, 0]
    assert abs(np.corrcoef(z[:, 0], z2)[0, 1]) < 0.01


def test_wnf_argument_validation():
    cell = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError):
        det.wnf(-1, 0, cell)
    with pytest.raises(ValueError):
        det.wnf(1 << 64, 0, cell)
    with pytest.raises(ValueError):
        det.wnf(0, 1 << 16, cell)
    with pytest.raises(ValueError):
        det.wnf(0, -1, cell)
    with pytest.raises(ValueError):
        det.wnf(0, 0, cell, k=1 << 32)
    with pytest.raises(ValueError):
        det.wnf(0, 0, cell, k=-1)
    with pytest.raises(ValueError):
        det.wnf(0, 0, np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError):
        det.wnf(0, 0, cell, channels=0)
    with pytest.raises(TypeError):
        det.wnf(0, 0, np.array([0.5]))


def test_cell_id_and_wnf_tile_overlap_has_rho_one():
    ids = det.cell_id(np.array([0, 1, 5, 7]), np.array([0, 2, 3, (1 << 31) - 1]))
    assert ids.dtype == np.int64
    assert list(ids) == [0, (1 << 31) + 2, (5 << 31) + 3, (7 << 31) + (1 << 31) - 1]
    with pytest.raises(ValueError):
        det.cell_id(-1, 0)
    with pytest.raises(ValueError):
        det.cell_id(0, 1 << 31)
    # apron이 겹치는 두 타일은 겹친 셀에서 비트 동일
    a = det.wnf_tile(3, 6, i0=10, j0=20, h=40, w=48, k=2, channels=4)
    b = det.wnf_tile(3, 6, i0=30, j0=44, h=40, w=48, k=2, channels=4)
    assert a.shape == (40, 48, 4)
    assert a[20:, 24:].tobytes() == b[:20, :24].tobytes()
    full = det.wnf(3, 6, det.cell_id(*np.meshgrid(np.arange(10, 50), np.arange(20, 68), indexing="ij")), k=2)
    assert a.tobytes() == full.tobytes()


def test_derive_seed():
    s1 = det.derive_seed(5, b"abc")
    assert isinstance(s1, int) and 0 <= s1 < 1 << 64
    assert s1 == det.derive_seed(5, b"abc")
    assert s1 != det.derive_seed(6, b"abc") and s1 != det.derive_seed(5, b"abd")
    # 구분자 모호성이 없다
    assert det.derive_seed(5, "ab", "c") != det.derive_seed(5, "a", "bc")
    h_i = det.hash_bytes(b"input image bytes")
    assert 0 <= det.derive_seed(123, h_i) < 1 << 64
