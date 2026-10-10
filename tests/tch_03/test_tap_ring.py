"""TCH-03 링버퍼: FIFO 축출, 바이트 예산, 복사 격리, 결정론 표본, 기저 지문 고정."""
import dataclasses

import pytest
import torch

import tch03_fakes as F
from buc.teacher import taps as T

PER = 3 * 2 * F.K * F.GRID[0] * F.GRID[1]        # 배치 2 탭 세트 하나의 바이트 수


def make_tapset(bus, model, seed, key=None, batch=2, prov=F.ENGINE):
    with bus.capture(prov, grid=F.GRID, token_offset=F.OFF, key=key) as cap:
        with torch.no_grad():
            model(F.make_input(seed, batch=batch))
    return cap.tapset


def test_ring_ctor_validation(basis):
    for kw in ({"max_entries": 0}, {"max_entries": -1}, {"max_entries": 2.0}, {"max_entries": True},
               {"max_entries": 2, "max_bytes": 0}):
        with pytest.raises(T.TapSpecError):
            T.TapRing(basis, **kw)
    assert T.TapRing(basis, max_entries=2).basis_fp == basis.fingerprint
    assert T.TapRing(basis.fingerprint, max_entries=2).basis_fp == basis.fingerprint


def test_fifo_eviction_and_monotonic_seq(bus, basis, model):
    ring = T.TapRing(basis, max_entries=3)
    seqs = [ring.push(make_tapset(bus, model, s, key=f"k{s}")) for s in range(5)]
    assert seqs == [0, 1, 2, 3, 4]                          # seq는 단조 증가, 재사용 없음
    assert len(ring) == 3 and ring.seqs() == [2, 3, 4] and ring.evicted == 2
    assert ring.nbytes == 3 * PER
    for gone in (0, 1, 99):
        with pytest.raises(KeyError):
            ring.get(gone)
    assert [ts.key for ts in ring.latest(2)] == ["k3", "k4"]          # 오래된 것 먼저
    assert [ts.key for ts in ring.latest(10)] == ["k2", "k3", "k4"]
    assert ring.latest(0) == []
    ring.clear()
    assert len(ring) == 0 and ring.nbytes == 0
    assert ring.push(make_tapset(bus, model, 9)) == 5               # clear 뒤에도 seq 계속 증가


def test_byte_budget_evicts_oldest_until_fit_and_rejects_oversize(bus, basis, model):
    ring = T.TapRing(basis, max_entries=100, max_bytes=2 * PER + PER // 2)     # 2.5개 분량
    for s in range(3):
        ring.push(make_tapset(bus, model, s, key=f"k{s}"))
    assert len(ring) == 2 and ring.nbytes == 2 * PER and ring.seqs() == [1, 2]
    big = make_tapset(bus, model, 7, batch=2)
    huge = dataclasses.replace(big, layers={l: torch.cat([q, q, q], dim=0) for l, q in big.layers.items()})
    assert huge.nbytes == 3 * PER > ring.max_bytes
    before = (ring.seqs(), ring.nbytes, ring.evicted)
    with pytest.raises(T.TapError):
        ring.push(huge)
    assert (ring.seqs(), ring.nbytes, ring.evicted) == before         # 거부된 push는 아무것도 바꾸지 않는다
    # 작은 항목 여러 개 뒤에 큰 항목 하나: 필요한 만큼만 연달아 축출
    small = make_tapset(bus, model, 8, batch=1)
    assert small.nbytes == PER // 2
    ring2 = T.TapRing(basis, max_entries=100, max_bytes=2 * PER)
    for s in range(4):
        ring2.push(make_tapset(bus, model, s, batch=1))
    assert len(ring2) == 4 and ring2.nbytes == 2 * PER
    ring2.push(make_tapset(bus, model, 20, batch=2))
    assert ring2.nbytes <= 2 * PER and ring2.seqs()[-1] == 4 and len(ring2) == 3 and ring2.evicted == 2


def test_ring_stores_independent_copies(bus, basis, model):
    ring = T.TapRing(basis, max_entries=4)
    ts = make_tapset(bus, model, 1)
    d0 = ts.digest()
    s = ring.push(ts)
    for q in ts.layers.values():
        q.zero_()                                           # 원본을 망가뜨려도
    got = ring.get(s)
    assert got.digest() == d0                               # 링 안의 값은 그대로
    for q in got.layers.values():
        q.fill_(5)                                          # 꺼낸 것을 바꿔도
    assert ring.get(s).digest() == d0                       # 링은 그대로
    assert ring.get(s).prov == F.ENGINE and ring.get(s).grid == F.GRID


def test_ring_rejects_foreign_basis_and_wrong_types(bus, basis, model):
    ring = T.TapRing(basis, max_entries=4)
    ts = make_tapset(bus, model, 1)
    other = F.make_basis(seed=50)
    foreign = dataclasses.replace(ts, basis_fp=other.fingerprint)
    with pytest.raises(T.TapBasisError):
        ring.push(foreign)
    with pytest.raises(TypeError):
        ring.push({"layers": ts.layers})
    assert len(ring) == 0 and ring.nbytes == 0
    assert ring.push(ts) == 0                               # 실패한 push는 seq를 소모하지 않는다


def test_sample_is_deterministic_sorted_subset(bus, basis, model):
    ring = T.TapRing(basis, max_entries=30)
    for s in range(20):
        ring.push(make_tapset(bus, model, s, batch=1))
    a = ring.sample(5, seed=3)
    assert a == ring.sample(5, seed=3)
    assert a == sorted(a) and len(set(a)) == 5 and set(a) <= set(ring.seqs())
    assert len({tuple(ring.sample(5, seed=s)) for s in range(10)}) >= 3     # 시드에 따라 달라진다
    assert ring.sample(0, seed=1) == []
    assert ring.sample(20, seed=1) == ring.seqs()
    with pytest.raises(ValueError):
        ring.sample(21, seed=1)
    with pytest.raises(ValueError):
        ring.sample(-1, seed=1)
