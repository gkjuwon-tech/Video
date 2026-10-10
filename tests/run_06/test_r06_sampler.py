"""Feistel PRP + 무상태 샘플러: 전단사, 역함수, 결정론, 독립 오라클 대조, 통계, 재개 성질."""
import time

import numpy as np
import pytest

from buc.core.det import derive_seed
from buc.train.sampler import FeistelPRP, StatelessSampler

M64 = (1 << 64) - 1


# ---- 계약서의 알고리즘을 그대로 옮긴 독립 오라클 -------------------------------------------------
def _mix(z):
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def oracle_prp(n, seed, rounds=6):
    h = max(1, ((n - 1).bit_length() + 1) // 2)
    mask = (1 << h) - 1
    keys = [derive_seed(seed, "buc.feistel.v1", r) for r in range(rounds)]

    def enc(x):
        left, right = x >> h, x & mask
        for k in keys:
            left, right = right, left ^ (_mix((right ^ k) & M64) & mask)
        return (left << h) | right

    def perm(i):
        y = enc(i)
        while y >= n:
            y = enc(y)
        return y

    return perm


# 오라클이 계약 그대로인지 못 박는 고정 값(독립 계산). 알고리즘이 바뀌면 재개가 깨지므로 여기서 잡는다.
GOLDEN_10_12345 = [7, 1, 4, 2, 5, 6, 8, 9, 3, 0]


def test_golden_vector_pins_the_algorithm():
    p = FeistelPRP(10, 12345)
    assert [p(i) for i in range(10)] == GOLDEN_10_12345
    assert [oracle_prp(10, 12345)(i) for i in range(10)] == GOLDEN_10_12345


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 16, 17, 100, 255, 256, 257, 1000, 4097])
def test_bijection_and_inverse_match_oracle(n):
    p = FeistelPRP(n, 2024)
    img = [p(i) for i in range(n)]
    assert sorted(img) == list(range(n))
    assert img == [oracle_prp(n, 2024)(i) for i in range(n)]
    assert [p.inverse(j) for j in img] == list(range(n))


def test_default_rounds_and_attributes():
    p = FeistelPRP(10, 3)
    assert (p.n, p.seed, p.rounds) == (10, 3, 6)
    q = FeistelPRP(10, 3, rounds=3)
    assert sorted(q(i) for i in range(10)) == list(range(10))
    assert [q(i) for i in range(10)] == [oracle_prp(10, 3, 3)(i) for i in range(10)]


def test_deterministic_and_seed_sensitive():
    a = [FeistelPRP(500, 1)(i) for i in range(500)]
    assert a == [FeistelPRP(500, 1)(i) for i in range(500)]
    b = [FeistelPRP(500, 2)(i) for i in range(500)]
    assert a != b
    assert sum(x == y for x, y in zip(a, b)) < 20  # 서로 다른 시드는 거의 다른 순열


def test_array_api_matches_scalar():
    p = FeistelPRP(1234, 99)
    idx = np.arange(1234)
    out = p.permute_array(idx)
    assert out.dtype == np.int64 and out.shape == (1234,)
    assert out.tolist() == [p(int(i)) for i in idx]
    assert p.inverse_array(out).tolist() == idx.tolist()
    m = np.array([[3, 1], [0, 1233]])
    assert p.permute_array(m).shape == (2, 2)
    assert p.permute_array(np.array([], dtype=np.int64)).shape == (0,)


def test_input_validation():
    p = FeistelPRP(10, 1)
    for bad in (-1, 10):
        with pytest.raises(ValueError):
            p(bad)
        with pytest.raises(ValueError):
            p.inverse(bad)
    with pytest.raises(ValueError):
        p.permute_array(np.array([0, 10]))
    with pytest.raises(TypeError):
        p(1.5)
    with pytest.raises(TypeError):
        p(True)
    for n in (0, -3):
        with pytest.raises(ValueError):
            FeistelPRP(n, 1)
    with pytest.raises(ValueError):
        FeistelPRP((1 << 62) + 1, 1)
    for seed in (-1, 1 << 64):
        with pytest.raises(ValueError):
            FeistelPRP(10, seed)
    with pytest.raises(ValueError):
        FeistelPRP(10, 1, rounds=2)
    with pytest.raises(TypeError):
        FeistelPRP(10.0, 1)
    with pytest.raises(TypeError):
        FeistelPRP(10, True)


def test_n_equals_one_is_identity():
    p = FeistelPRP(1, 5)
    assert p(0) == 0 and p.inverse(0) == 0


def test_mixing_statistics():
    n = 1000
    p = FeistelPRP(n, 31337)
    img = np.array([p(i) for i in range(n)])
    # 임의의 순열에서 E|pi(i+1)-pi(i)| ~ n/3
    adj = np.abs(np.diff(img)).mean()
    assert 0.8 * n / 3 < adj < 1.2 * n / 3
    # 위치와 값의 상관이 거의 없다
    assert abs(np.corrcoef(np.arange(n), img)[0, 1]) < 0.12
    # 고정점은 평균 1개 정도
    assert (img == np.arange(n)).sum() <= 8
    # 시드를 바꿔도 pi(0)는 [0,n)에 골고루 퍼진다
    firsts = np.array([FeistelPRP(n, s)(0) for s in range(300)])
    assert abs(firsts.mean() - (n - 1) / 2) < 70
    assert firsts.min() < 0.1 * n and firsts.max() > 0.9 * n


def test_huge_domain_spot_checks():
    n = (1 << 40) + 12345
    p = FeistelPRP(n, 77)
    t0 = time.time()
    xs = [0, 1, 2, n - 1, n // 2, 123456789, 987654321012]
    xs += list(np.random.default_rng(0).integers(0, n, 3000))
    ys = [p(int(x)) for x in xs]
    assert all(0 <= y < n for y in ys)
    assert [p.inverse(y) for y in ys] == [int(x) for x in xs]
    # 서로 다른 입력은 서로 다른 출력(전단사의 표본 검증)
    uniq = {int(x): y for x, y in zip(xs, ys)}
    assert len(set(uniq.values())) == len(uniq)
    assert ys[:3] == [oracle_prp(n, 77)(x) for x in xs[:3]]
    assert time.time() - t0 < 10
    top = FeistelPRP(1 << 62, 1)
    assert 0 <= top((1 << 62) - 1) < (1 << 62)


# ---- StatelessSampler --------------------------------------------------------------------------
def epoch_oracle(n, seed, epoch):
    return oracle_prp(n, derive_seed(seed, "buc.train.epoch.v1", epoch))


def test_sampler_positions_follow_the_epoch_formula():
    n, b, seed = 10, 4, 5
    s = StatelessSampler(n, b, seed)
    for pos in range(0, 45):
        e, o = divmod(pos, n)
        assert s.index_at(pos) == epoch_oracle(n, seed, e)(o)
    for step in range(11):
        got = s.indices(step)
        assert got.dtype == np.int64 and got.shape == (b,)
        assert got.tolist() == [s.index_at(step * b + j) for j in range(b)]


def test_each_epoch_is_a_permutation_and_epochs_differ():
    n, b = 20, 4  # 한 에폭 = 5스텝
    s = StatelessSampler(n, b, 11)
    e0 = np.concatenate([s.indices(t) for t in range(5)])
    e1 = np.concatenate([s.indices(t) for t in range(5, 10)])
    assert sorted(e0.tolist()) == list(range(n))
    assert sorted(e1.tolist()) == list(range(n))
    assert e0.tolist() != e1.tolist()


def test_batches_straddling_an_epoch_boundary():
    n, b = 10, 4
    s = StatelessSampler(n, b, 3)
    flat = np.concatenate([s.indices(t) for t in range(5)])  # 20개 = 두 에폭
    assert sorted(flat[:10].tolist()) == list(range(10))
    assert sorted(flat[10:].tolist()) == list(range(10))
    assert len(set(s.indices(2).tolist())) == b  # 경계를 걸친 배치 안에서도 중복 없음(n>=b)


def test_stateless_history_independence():
    s1 = StatelessSampler(37, 5, 42)
    seq = [s1.indices(t).tolist() for t in range(30)]
    s2 = StatelessSampler(37, 5, 42)
    order = list(range(30))
    np.random.default_rng(0).shuffle(order)
    for t in order:
        assert s2.indices(t).tolist() == seq[t]
    # 새 인스턴스(= 재개한 프로세스)가 중간 스텝부터 같은 열을 만든다
    s3 = StatelessSampler(37, 5, 42)
    assert [s3.indices(t).tolist() for t in range(17, 30)] == seq[17:]


def test_sampler_seed_sensitivity_and_large_step():
    a = StatelessSampler(100, 8, 1)
    b = StatelessSampler(100, 8, 2)
    assert a.indices(0).tolist() != b.indices(0).tolist()
    t0 = time.time()
    big = a.indices(10 ** 9)
    assert big.shape == (8,) and big.min() >= 0 and big.max() < 100
    assert time.time() - t0 < 5
    assert a.indices(10 ** 9).tolist() == big.tolist()


def test_sampler_validation():
    for kw in (dict(num_samples=0, batch_size=1, seed=0), dict(num_samples=5, batch_size=0, seed=0),
               dict(num_samples=5, batch_size=1, seed=-1)):
        with pytest.raises(ValueError):
            StatelessSampler(**kw)
    s = StatelessSampler(5, 2, 0)
    with pytest.raises(ValueError):
        s.indices(-1)
    with pytest.raises(ValueError):
        s.index_at(-1)
    with pytest.raises(TypeError):
        s.indices(1.0)
