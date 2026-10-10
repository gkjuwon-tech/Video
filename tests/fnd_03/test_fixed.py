"""FND-03 int64 고정소수점 segment-reduce 검증: 순서 불변(비트 동일)과 정확성."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from buc.core import det


def py_encode(x, f):
    # 반올림 반짝수(half-even) 정확 오라클
    return round(Fraction(float(x)) * (1 << f))


def test_fixed_encode_decode_exact_oracle():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(size=2000) * 10, [0.0, -0.0, 0.5 / (1 << 24), 1.5 / (1 << 24), 2.5 / (1 << 24)]])
    q = det.fixed_encode(x.astype(np.float32), 24)
    assert q.dtype == np.int64
    assert [int(v) for v in q] == [py_encode(np.float32(v), 24) for v in x]
    back = det.fixed_decode(q, 24)
    assert back.dtype == np.float64
    assert np.abs(back - x.astype(np.float32).astype(np.float64)).max() <= 0.5 / (1 << 24) + 1e-15
    # 2의 거듭제곱 격자 값은 정확히 왕복
    g = np.arange(-100, 100) / 256.0
    assert (det.fixed_decode(det.fixed_encode(g, 24), 24) == g).all()


def test_fixed_encode_rejects_nonfinite_and_overflow():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            det.fixed_encode(np.array([1.0, bad]), 24)
    with pytest.raises(OverflowError):
        det.fixed_encode(np.array([2.0**40]), 24)  # 2^64 > int64
    with pytest.raises(ValueError):
        det.fixed_encode(np.array([1.0]), 0)
    with pytest.raises(ValueError):
        det.fixed_encode(np.array([1.0]), 49)


def make_problem(n=60_000, nseg=500, d=3, seed=0):
    rng = np.random.default_rng(seed)
    seg = rng.integers(0, nseg, size=n).astype(np.int64)
    view = rng.integers(0, 17, size=n).astype(np.int64)
    # 크기 차이가 큰 값들: 부동소수점 합은 순서에 민감해진다
    val = (rng.normal(size=(n, d)) * np.exp(rng.normal(size=(n, d)) * 3)).astype(np.float32)
    own = rng.uniform(0, 1, size=n).astype(np.float32)
    return seg, view, val, own, nseg


def test_segment_reduce_matches_exact_python_integer_reference():
    seg, view, val, own, nseg = make_problem(n=4000, nseg=37, d=2, seed=1)
    out = det.segment_reduce(val, seg, nseg, view_ids=view, weights=own, frac_bits=20)
    assert out.dtype == np.int64 and out.shape == (nseg, 2)
    ref = [[0, 0] for _ in range(nseg)]
    for i in range(len(seg)):
        for c in range(2):
            ref[seg[i]][c] += py_encode(np.float64(own[i]) * np.float64(val[i, c]), 20)
    assert out.tolist() == ref


def test_segment_reduce_is_order_invariant_bit_identical_over_8_shuffles():
    seg, view, val, own, nseg = make_problem()
    base = det.segment_reduce(val, seg, nseg, view_ids=view, weights=own)
    rng = np.random.default_rng(99)
    for _ in range(8):
        p = rng.permutation(len(seg))
        out = det.segment_reduce(val[p], seg[p], nseg, view_ids=view[p], weights=own[p])
        assert out.tobytes() == base.tobytes()
    # view_ids를 주지 않아도 결과는 같다(정수 합은 결합법칙이 성립)
    assert det.segment_reduce(val, seg, nseg, weights=own).tobytes() == base.tobytes()


def test_naive_float32_sum_is_order_dependent_but_ours_is_not():
    """동기 확인: 부동소수점 누산은 순서에 따라 비트가 달라진다."""
    seg, view, val, own, nseg = make_problem(n=60_000, nseg=3, d=1, seed=3)
    rng = np.random.default_rng(1)
    sums = set()
    for _ in range(4):
        p = rng.permutation(len(seg))
        acc = np.zeros(nseg, dtype=np.float32)
        for s, v in zip(seg[p][:20000], val[p][:20000, 0]):
            acc[s] += v
        sums.add(acc.tobytes())
    assert len(sums) > 1
    outs = {det.segment_reduce(val[p], seg[p], nseg).tobytes() for p in (rng.permutation(len(seg)) for _ in range(4))}
    assert len(outs) == 1


def test_segment_reduce_empty_segments_zero_and_scalar_values():
    out = det.segment_reduce(np.array([1.0, 2.0, 4.0], dtype=np.float32), np.array([0, 3, 3]), 5, frac_bits=10)
    assert out.shape == (5,)
    assert out.tolist() == [1024, 0, 0, 6144, 0]
    empty = det.segment_reduce(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64), 3)
    assert empty.shape == (3, 4) and not empty.any()


def test_segment_reduce_validation():
    v = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        det.segment_reduce(v, np.array([0, 1, 5]), 3)  # 범위 밖
    with pytest.raises(ValueError):
        det.segment_reduce(v, np.array([0, -1, 1]), 3)
    with pytest.raises(ValueError):
        det.segment_reduce(v, np.array([0, 1]), 3)  # 길이 불일치
    with pytest.raises(ValueError):
        det.segment_reduce(v, np.array([0, 1, 1]), 3, view_ids=np.array([0, 1]))
    with pytest.raises(ValueError):
        det.segment_reduce(v, np.array([0, 1, 1]), 3, weights=np.ones(2))
    # 항마다는 안전(2^36 * 2^24 = 2^60)하지만 합이 int64를 넘는 경우: 감김 없이 OverflowError
    big = np.full((16, 1), 2.0**36, dtype=np.float64)
    with pytest.raises(OverflowError):
        det.segment_reduce(big, np.zeros(16, dtype=np.int64), 1, frac_bits=24)
    with pytest.raises(OverflowError):  # 정확히 2^63
        det.segment_reduce(big[:8], np.zeros(8, dtype=np.int64), 1, frac_bits=24)
    with pytest.raises(OverflowError):  # 음수 방향 (-2^63 - 2^60)
        det.segment_reduce(-big[:9], np.zeros(9, dtype=np.int64), 1, frac_bits=24)
    # int64 안에 들어가는 큰 합(2^62, 2^63 - 2^60)은 정확히 돌려준다
    ok4 = det.segment_reduce(big[:4], np.zeros(4, dtype=np.int64), 1, frac_bits=24)
    assert ok4.tolist() == [[1 << 62]]
    ok7 = det.segment_reduce(big[:7], np.zeros(7, dtype=np.int64), 1, frac_bits=24)
    assert ok7.tolist() == [[7 << 60]]
    okn = det.segment_reduce(-big[:8], np.zeros(8, dtype=np.int64), 1, frac_bits=24)
    assert okn.tolist() == [[-(1 << 63)]]
    # 중간 부분합이 넘쳐도 최종 합이 범위 안이면 정확해야 한다(정수 합은 결합법칙이 성립)
    # 순서가 달라도 같은 결과여야 하므로(순서 불변), 부분합의 중간 넘침은 오류가 아니다.
    mix = np.concatenate([big[:8], -big[:8], np.ones((1, 1))])
    for p in (np.arange(17), np.arange(17)[::-1], np.random.default_rng(3).permutation(17)):
        assert det.segment_reduce(mix[p], np.zeros(17, dtype=np.int64), 1, frac_bits=24).tolist() == [[1 << 24]]
    with pytest.raises(ValueError):
        det.segment_reduce(np.array([np.nan], dtype=np.float32), np.array([0]), 1)


def test_canonical_order_is_stable_lexsort_on_segment_then_view():
    seg = np.array([2, 1, 2, 1, 2, 1])
    view = np.array([0, 5, 0, 3, 0, 5])
    o = det.canonical_order(seg, view)
    assert o.dtype == np.int64
    assert list(o) == [3, 1, 5, 0, 2, 4]  # (1,3) (1,5)a (1,5)b (2,0)a (2,0)b (2,0)c, 동률은 원래 순서
    # view 없으면 seg만, 안정 정렬
    assert list(det.canonical_order(seg)) == [1, 3, 5, 0, 2, 4]


def test_torch_input_and_torch_integer_path_agree_with_numpy():
    seg, view, val, own, nseg = make_problem(n=20_000, nseg=100, d=3, seed=7)
    ref = det.segment_reduce(val, seg, nseg, view_ids=view, weights=own)
    out = det.segment_reduce(torch.from_numpy(val), torch.from_numpy(seg), nseg,
                             view_ids=torch.from_numpy(view), weights=torch.from_numpy(own))
    assert isinstance(out, np.ndarray) and out.tobytes() == ref.tobytes()
    # 정수 항을 이미 가진 경우 GPU에서도 쓸 수 있는 torch 경로(index_add_ on int64)
    q = torch.from_numpy(det.fixed_encode(own[:, None].astype(np.float64) * val.astype(np.float64), 24))
    t = det.segment_reduce_int_torch(q, torch.from_numpy(seg), nseg)
    assert t.dtype == torch.int64 and t.shape == (nseg, 3)
    assert t.numpy().tobytes() == ref.tobytes()
    perm = torch.from_numpy(np.random.default_rng(2).permutation(len(seg)))
    t2 = det.segment_reduce_int_torch(q[perm], torch.from_numpy(seg)[perm], nseg)
    assert torch.equal(t, t2)


@pytest.mark.gpu
def test_segment_reduce_int_torch_cuda_matches_cpu():
    assert torch.cuda.is_available()
    seg, view, val, own, nseg = make_problem(n=200_000, nseg=1000, d=4, seed=11)
    q = torch.from_numpy(det.fixed_encode(val.astype(np.float64), 24))
    cpu = det.segment_reduce_int_torch(q, torch.from_numpy(seg), nseg)
    gpu = det.segment_reduce_int_torch(q.cuda(), torch.from_numpy(seg).cuda(), nseg)
    assert torch.equal(cpu, gpu.cpu())


def test_decode_of_reduction_is_close_to_float64_sum():
    seg, view, val, own, nseg = make_problem(n=5000, nseg=20, d=1, seed=4)
    val = np.clip(val, -50, 50)
    out = det.fixed_decode(det.segment_reduce(val, seg, nseg, frac_bits=24), 24)[:, 0]
    ref = np.zeros(nseg)
    np.add.at(ref, seg, val[:, 0].astype(np.float64))
    # 항당 최대 0.5 ulp(2^-25) 양자화 오차
    counts = np.bincount(seg, minlength=nseg)
    assert (np.abs(out - ref) <= counts * 0.5 / (1 << 24) + 1e-9).all()
