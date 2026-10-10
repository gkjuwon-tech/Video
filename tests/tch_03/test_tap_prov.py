"""TCH-03 prov 검사: ext 바이패스, 카나리아 C1/C5, 링 진입 규칙 = FND-04 train_ok AND NOT ext."""
import dataclasses
import gc
import itertools
import weakref

import pytest
import torch

import tch03_fakes as F
from buc.gov import prov as P
from buc.teacher import taps as T


def run(bus, model, prov, seed=1, ring=None, key=None):
    with bus.capture(prov, grid=F.GRID, token_offset=F.OFF, key=key, ring=ring) as cap:
        with torch.no_grad():
            model(F.make_input(seed))
    return cap.tapset


def test_prov_is_mandatory_and_validated(bus, model):
    with pytest.raises(TypeError):
        with bus.capture(grid=F.GRID):                                   # prov 없음
            pass
    for bad in (None, "USER", b"USER"):                                   # 단일 str·bytes·None
        with pytest.raises(TypeError):
            with bus.capture(bad, grid=F.GRID):
                pass
    with pytest.raises(ValueError):
        with bus.capture(["NOT-A-TAG"], grid=F.GRID):                    # 알 수 없는 태그: 닫힌 쪽 실패
            pass
    ts = run(bus, model, ["CAPTURE", "ENGINE"])
    assert ts.prov == P.as_record(["ENGINE", "CAPTURE"])                  # 정규 레코드로 보관


def test_ext_capture_is_ephemeral_but_taps_are_identical(bus, model):
    clean = run(bus, model, ["ENGINE"], seed=4)
    for tags in (["USER"], ["USER-DERIVED"], ["EXTGEN:test"], ["EXTGEN-DERIVED:g"], ["BIO"], ["ENGINE", "USER"]):
        ts = run(bus, model, tags, seed=4)
        assert ts.ephemeral is True and P.ext_bit(ts.prov)
        for l in T.TAP_LAYERS:                                            # 라이브 판독(MRH)에는 같은 탭이 필요하다
            assert torch.equal(ts.layers[l], clean.layers[l])
        assert ts.digest() != clean.digest()                              # 그러나 prov가 다르면 다이제스트도 다르다
    assert clean.ephemeral is False
    # HOLDOUT은 ext가 아니다: 임시 표시는 아니지만 링에는 못 들어간다 (아래 테스트)
    assert run(bus, model, ["HOLDOUT"]).ephemeral is False


@pytest.mark.parametrize("tags", [["USER"], ["EXTGEN:test"], ["USER-DERIVED"], ["EXTGEN-DERIVED:g"],
                                  ["EXTGEN-INSPIRED:g"], ["BIO"], ["HOLDOUT"], ["COMP"], ["PUB-SEALED"],
                                  ["EVAL-ONLY"], ["NC"], ["D3"], ["ENGINE", "USER"], ["ASSET"], ["ASSET:GPL"]])
def test_ring_push_blocks_forbidden_lineage_canaries(bus, basis, model, tags):
    """카나리아 C1(USER)·C5(EXTGEN:test) 포함: 실패하는 것이 정상이다."""
    ring = T.TapRing(basis, max_entries=4)
    ts = run(bus, model, ["ENGINE"])
    bad = dataclasses.replace(ts, prov=P.as_record(tags), ephemeral=False)   # 플래그를 믿지 않고 prov로 판정해야 한다
    with pytest.raises(P.FirewallError):
        ring.push(bad)
    assert len(ring) == 0 and ring.nbytes == 0
    assert ring.push(ts) == 0                                                # 링은 멀쩡하다


def test_ring_push_blocks_ephemeral_flag_even_with_clean_looking_prov(bus, basis, model):
    ring = T.TapRing(basis, max_entries=4)
    ts = run(bus, model, ["ENGINE"])
    with pytest.raises(P.FirewallError):
        ring.push(dataclasses.replace(ts, ephemeral=True))
    ext = run(bus, model, ["USER"])
    with pytest.raises(P.FirewallError):
        ring.push(ext)
    assert len(ring) == 0


def test_derived_user_lineage_cannot_enter(bus, basis, model):
    ring = T.TapRing(basis, max_entries=4)
    ts = run(bus, model, ["ENGINE"])
    child = P.derive("tap", [P.as_record(["ENGINE"]), P.as_record(["USER"])])
    with pytest.raises(P.FirewallError):
        ring.push(dataclasses.replace(ts, prov=child, ephemeral=False))


def test_ring_rule_equals_train_ok_and_not_ext_over_tag_sets(bus, basis, model):
    pool = ["ENGINE", "CAPTURE", "HUMAN", "ASSET:CC0", "ASSET:GPL", "USER", "EXTGEN:x", "HOLDOUT", "COMP", "NC",
            "IP-FLAG", "IP-FLAG:masked", "OPEN-GEN:m:h:train", "OPEN-GEN:m:h:bench", "COMMISSION:eval",
            "COMMISSION:eval+train", "OPTIN"]
    base = run(bus, model, ["ENGINE"])
    n_ok = n_bad = 0
    for r in (1, 2):
        for tags in itertools.combinations(pool, r):
            rec = P.as_record(list(tags))
            ring = T.TapRing(basis, max_entries=2)
            ts = dataclasses.replace(base, prov=rec, ephemeral=False)
            should = P.train_ok(rec) and not P.ext_bit(rec)
            if should:
                assert ring.push(ts) == 0, tags
                n_ok += 1
            else:
                with pytest.raises(P.FirewallError):
                    ring.push(ts)
                assert len(ring) == 0, tags
                n_bad += 1
    assert n_ok >= 20 and n_bad >= 100                                   # 표본이 양쪽을 모두 덮는다


def test_auto_wired_ring_bypasses_instead_of_raising(bus, basis, model):
    ring = T.TapRing(basis, max_entries=8)
    run(bus, model, ["ENGINE"], ring=ring, key="e")
    assert len(ring) == 1 and bus.stats == {"captures": 1, "pushed": 1, "bypassed": 0}
    for tags in (["USER"], ["EXTGEN:test"], ["HOLDOUT"], ["COMP"], ["ENGINE", "USER-DERIVED"]):
        ts = run(bus, model, tags, ring=ring)                           # 예외 없이 지나간다
        assert ts.layers[8].shape == (2, F.K, *F.GRID)                  # 호출자는 라이브 탭을 받는다
    assert len(ring) == 1 and ring.nbytes == 3 * 2 * F.K * 12
    assert bus.stats == {"captures": 6, "pushed": 1, "bypassed": 5}
    assert [t.key for t in ring.latest(5)] == ["e"]


def test_bus_retains_no_reference_after_capture(bus, basis, model):
    ring = T.TapRing(basis, max_entries=8)
    for tags, use_ring in ((["USER"], True), (["USER"], False), (["ENGINE"], False), (["HOLDOUT"], True)):
        ts = run(bus, model, tags, ring=ring if use_ring else None)
        wr_set = weakref.ref(ts)
        wr_q = weakref.ref(ts.layers[8])
        del ts
        gc.collect()
        assert wr_set() is None and wr_q() is None, tags                 # 버스·링(미입력 시)이 탭을 붙잡지 않는다
    assert len(ring) == 0


def test_nothing_is_recorded_after_ext_capture_exits(bus, model):
    run(bus, model, ["USER"])
    n = bus.stats["captures"]
    model(F.make_input(2))                                               # 캡처 밖 forward
    assert bus.stats["captures"] == n
