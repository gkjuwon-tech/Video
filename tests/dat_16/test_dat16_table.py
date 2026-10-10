"""DAT-16 실제 표(data/size_prior.yaml)의 수용 기준: 구조, 규모, 도메인 앵커, 순서 상식, 분포 위생."""
import math
import statistics

import pytest

import dat16_anchors as A
import dat16_support as S

T = S.tool()


@pytest.fixture(scope="module")
def raw():
    return S.real_raw()


@pytest.fixture(scope="module")
def table():
    return T.load_table(S.DATA)


def test_checker_passes_on_real_table_with_default_limits(raw):
    assert T.check_raw(raw) == []
    assert T.main([]) == 0


def test_scale_of_table(raw):
    classes = raw["classes"]
    assert 480 <= len(classes) <= 600
    assert sum(1 for e in classes.values() if e["tier"] == 1) >= 150
    groups = {e["group"] for e in classes.values()}
    assert len(groups) >= 12
    for g in ("architecture", "furniture", "person", "animal", "vehicle", "street", "plant", "tableware", "electronics"):
        assert sum(1 for e in classes.values() if e["group"] == g) >= 8, g


def test_domain_anchors_present_in_range_and_core(raw):
    classes = raw["classes"]
    bad = []
    for name, (dim, lo, hi, s_lo, s_hi, group) in A.ANCHORS.items():
        e = classes.get(name)
        if e is None:
            bad.append(f"{name}: 없음")
            continue
        if e["dim"] != dim:
            bad.append(f"{name}: dim {e['dim']} != {dim}")
        if e["group"] != group:
            bad.append(f"{name}: group {e['group']} != {group}")
        if e["tier"] != 1:
            bad.append(f"{name}: tier != 1")
        if not (lo <= e["median_m"] <= hi):
            bad.append(f"{name}: median {e['median_m']} 밖 [{lo},{hi}]")
        if not (s_lo <= e["sigma_log"] <= s_hi):
            bad.append(f"{name}: sigma {e['sigma_log']} 밖 [{s_lo},{s_hi}]")
    assert not bad, "\n".join(bad)


def test_task_statement_ranges(table):
    # 사양(PARADIGM-v2 / pipeline.upg 2.6): 문 2.0-2.1 m, 좌석 높이 0.42-0.48 m, 계단 단 0.15-0.19 m, 성인 키 1.5-1.9 m.
    # 95% 구간(range_m)이 사양 범위 주변 여유(약 10%) 안에 있어야 한다.
    d = table["door_interior"]
    assert d.range_m[0] >= 1.8 and d.range_m[1] <= 2.4
    cs = table["chair_seat"]
    assert cs.range_m[0] >= 0.36 and cs.range_m[1] <= 0.52
    sr = table["stair_riser"]
    assert sr.range_m[0] >= 0.13 and sr.range_m[1] <= 0.21
    p = table["person_adult"]
    assert p.range_m[0] >= 1.45 and p.range_m[1] <= 1.95
    # 95% 구간의 로그 정규 분위수와 range_m이 서로 맞는다(±15%)
    for c in (d, cs, sr, p):
        assert c.quantile(0.025) == pytest.approx(c.range_m[0], rel=0.15)
        assert c.quantile(0.975) == pytest.approx(c.range_m[1], rel=0.15)


def test_size_ordering_chain(raw):
    med = {n: raw["classes"][n]["median_m"] for n in A.CHAIN}
    for a, b in zip(A.CHAIN, A.CHAIN[1:]):
        assert med[b] >= med[a] * A.CHAIN_MIN_RATIO, f"{a}({med[a]}) -> {b}({med[b]})"


def test_pairwise_ratio_sanity(raw):
    m = {n: e["median_m"] for n, e in raw["classes"].items()}
    assert 1.1 <= m["door_interior"] / m["person_adult"] <= 1.35
    assert 1.6 <= m["person_adult"] / m["chair_seat"] <= 4.5
    assert 0.35 <= m["person_child"] / m["person_adult"] <= 0.85
    assert 1.5 <= m["chair"] / m["chair_seat"] <= 2.5
    assert m["desk"] / m["dining_table"] == pytest.approx(1.0, abs=0.12)
    assert 1.5 <= m["bus"] / m["car"] <= 3.0
    assert m["horse"] >= m["cow"]
    assert m["elephant_african"] > m["horse"] > m["dog"]


def test_sigma_regimes_standardized_vs_variable(raw):
    c = raw["classes"]
    # 규격품은 좁고 변동이 큰 것은 넓다(복붙한 σ 방지)
    for name in ("door_interior", "dining_table", "soda_can", "credit_card", "basketball", "golf_ball"):
        assert c[name]["sigma_log"] <= 0.08, name
    for name in ("tree", "dog", "street_lamp", "television", "house"):
        assert c[name]["sigma_log"] >= 0.15, name
    sig = [e["sigma_log"] for e in c.values()]
    assert 0.07 <= statistics.median(sig) <= 0.35
    assert len({round(s, 3) for s in sig}) >= 12


def test_table_is_not_lazy_copy_paste(raw):
    meds = [e["median_m"] for e in raw["classes"].values()]
    assert len({round(m, 4) for m in meds}) >= 0.45 * len(meds)
    # 단위 감각: 로그 중앙값의 분포가 몇 자릿수에 걸쳐 있다
    logs = [math.log10(m) for m in meds]
    assert max(logs) - min(logs) >= 2.5
    # 같은 (dim, median, sigma) 삼중 중복 금지
    trip = [(e["dim"], e["median_m"], e["sigma_log"]) for e in raw["classes"].values()]
    assert len(set(trip)) >= len(trip) - 3


def test_aliases_resolve(table):
    for alias, canon in A.REQUIRED_ALIASES.items():
        assert table.resolve(alias).name == canon, alias
    for name in A.REQUIRED_CANONICAL:
        assert table.resolve(name).name == name


def test_every_alias_unique_across_table(raw):
    seen = {}
    for name, e in raw["classes"].items():
        for key in (name, *e.get("aliases", [])):
            assert key not in seen or seen[key] == name, (key, seen.get(key), name)
            seen[key] = name


def test_load_is_fast_and_deterministic():
    import time
    t0 = time.perf_counter()
    a, b = T.load_table(S.DATA), T.load_table(S.DATA)
    assert time.perf_counter() - t0 < 5.0
    assert a == b
    assert a.names() == b.names() == tuple(sorted(a.names()))
