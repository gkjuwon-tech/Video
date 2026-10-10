"""RD-21 단계 캐시: hit, 최소 재실행, 폴백 미캐시, 사본."""
from collections import Counter

import numpy as np

from buc.read import pipeline as P
from rd21_helpers import ALL_SIDS, fake_outputs, fake_steps, make_image, status_map


def test_second_run_all_hits_zero_calls_same_packet():
    img = make_image()
    cache = P.MemoryReadCache()
    c1, c2 = [], []
    r1 = P.run_read(img, steps=fake_steps(c1), cache=cache)
    assert len(c1) == 14 and len(cache) == 14
    assert all(r.key is not None and len(r.key) == 64 for r in r1.records)
    r2 = P.run_read(img, steps=fake_steps(c2), cache=cache)
    assert c2 == []
    assert all(r.status == "hit" for r in r2.records)
    assert [r.key for r in r1.records] == [r.key for r in r2.records]
    assert r2.packet.content_hash() == r1.packet.content_hash()
    nocache = P.run_read(img, steps=fake_steps())
    assert nocache.packet.content_hash() == r1.packet.content_hash()
    assert r2.packet.meta["read"]["steps"]["R5"]["status"] == "ok"  # hit 는 패킷에 드러나지 않는다


def test_param_change_reruns_step_and_dependents_only():
    img = make_image()
    cache = P.MemoryReadCache()
    P.run_read(img, steps=fake_steps(), cache=cache)
    calls = []
    res = P.run_read(img, steps=fake_steps(calls), cache=cache, params={"R6": {"k": 1}})
    assert sorted(calls) == ["R3", "R6"]  # R3 needs R6
    st = status_map(res)
    assert st["R6"] == "ok" and st["R3"] == "ok"
    assert all(st[s] == "hit" for s in ALL_SIDS if s not in ("R3", "R6"))
    calls.clear()
    P.run_read(img, steps=fake_steps(calls), cache=cache, params={"R0": {"z": 0}})
    # R0 의 하류: R1, R5 → R6 → R3
    assert sorted(calls) == sorted(["R0", "R1", "R5", "R6", "R3"])


def test_version_change_reruns_step_and_dependents():
    img = make_image()
    cache = P.MemoryReadCache()
    P.run_read(img, steps=fake_steps(), cache=cache)
    calls = []
    P.run_read(img, steps=fake_steps(calls, version={"R5": "2"}), cache=cache)
    assert sorted(calls) == ["R3", "R5", "R6"]


def test_opts_and_image_are_in_key():
    img = make_image()
    cache = P.MemoryReadCache()
    P.run_read(img, steps=fake_steps(), cache=cache)
    calls = []
    P.run_read(make_image(seed=1), steps=fake_steps(calls), cache=cache)
    assert len(calls) == 14
    calls.clear()
    P.run_read(img, steps=fake_steps(calls), cache=cache, opts=P.ReadOpts(cam_height_m=1.7))
    assert len(calls) == 14
    calls.clear()
    P.run_read(make_image(exif={"focal_35mm": 35.0}), steps=fake_steps(calls), cache=cache)
    assert len(calls) == 14  # exif 는 image.content_hash 에 들어간다


def test_fallback_not_cached_but_dependents_hit():
    img = make_image()
    cache = P.MemoryReadCache()
    calls = []

    def bad(state, params):
        raise RuntimeError("x")

    r1 = P.run_read(img, steps=fake_steps(calls, override={"R1": bad}), cache=cache)
    assert len(cache) == 13
    calls.clear()
    r2 = P.run_read(img, steps=fake_steps(calls, override={"R1": bad}), cache=cache)
    assert calls == ["R1"]  # 폴백은 다시 시도, R6(needs R1)는 같은 폴백 유효 키로 hit
    assert status_map(r2)["R1"] == "fallback" and status_map(r2)["R6"] == "hit"
    assert r2.packet.content_hash() == r1.packet.content_hash()


def test_recovered_upstream_invalidates_dependents():
    img = make_image()
    cache = P.MemoryReadCache()
    state = {"fail": True}

    def flaky(st, params):
        if state["fail"]:
            raise RuntimeError("x")
        return fake_outputs("R1", st)

    P.run_read(img, steps=fake_steps(override={"R1": flaky}), cache=cache)
    state["fail"] = False
    calls = []
    res = P.run_read(img, steps=fake_steps(calls, override={"R1": flaky}), cache=cache)
    assert sorted(calls) == ["R1", "R3", "R6"]
    assert res.packet.fallbacks == ()
    assert res.packet.content_hash() == P.run_read(img, steps=fake_steps()).packet.content_hash()


def test_skipped_upstream_key_stable():
    img = make_image()
    cache = P.MemoryReadCache()
    sids = ("R0", "R6")
    P.run_read(img, steps=fake_steps(sids=sids), cache=cache)
    calls = []
    res = P.run_read(img, steps=fake_steps(calls, sids=sids), cache=cache)
    assert calls == []
    assert status_map(res)["R6"] == "hit"
    # 이제 R1 이 실제로 등록되면 R6 키가 바뀐다
    calls.clear()
    P.run_read(img, steps=fake_steps(calls, sids=("R0", "R1", "R6")), cache=cache)
    assert sorted(calls) == ["R1", "R6"]


def test_cache_hit_outputs_read_only_and_cache_holds_copies():
    cache = P.MemoryReadCache()
    arr = np.zeros(3)
    cache.put("k" * 64, {"a": arr, "m": {"x": 1}})
    arr[0] = 7.0
    got = cache.get("k" * 64)
    assert got["a"][0] == 0.0
    got["m"]["x"] = 99
    got["new"] = 1
    again = cache.get("k" * 64)
    assert again["m"]["x"] == 1 and "new" not in again
    assert cache.get("z" * 64) is None

    img = make_image()
    c2 = P.MemoryReadCache()
    P.run_read(img, steps=fake_steps(), cache=c2)
    seen = Counter()

    def r1(state, params):
        try:
            state["R0"]["image_lin"][0, 0, 0] = 1.0
        except ValueError:
            seen["ro"] += 1
        return fake_outputs("R1", state)

    P.run_read(img, steps=fake_steps(override={"R1": r1}, version={"R1": "9"}), cache=c2)
    assert seen["ro"] == 1  # 캐시에서 온 R0 출력도 쓰기 금지
