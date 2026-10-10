"""DAT-16 적재 API: ClassPrior 수학, SizePriorTable, load_table, 그리고 하류(RD-08) 로그 가중 최소제곱 이음매."""
import dataclasses
import math

import numpy as np
import pytest
import yaml

import dat16_support as S

T = S.tool()


@pytest.fixture()
def small_table(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(S.small_raw(), sort_keys=True, allow_unicode=True), encoding="utf-8")
    return T.load_table(p, limits=S.small_limits())


def test_load_table_basic_shape_and_values(small_table):
    t = small_table
    raw = S.small_raw()["classes"]
    assert len(t) == 12
    assert list(t) == sorted(raw)
    assert "door_interior" in t and "interior_door" not in t and "nope" not in t
    for name, e in raw.items():
        c = t[name]
        assert isinstance(c, T.ClassPrior)
        assert (c.name, c.dim, c.group, c.tier) == (name, e["dim"], e["group"], e["tier"])
        assert c.median_m == pytest.approx(e["median_m"], rel=0, abs=1e-12)
        assert c.sigma_log == pytest.approx(e["sigma_log"], rel=0, abs=1e-12)
        assert c.range_m == tuple(e["range_m"])
        assert c.aliases == tuple(e.get("aliases", ()))
    with pytest.raises(KeyError):
        t["interior_door"]  # 별칭은 resolve로만
    with pytest.raises(dataclasses.FrozenInstanceError):
        t["car"].median_m = 3.0


def test_names_core_groups(small_table):
    t = small_table
    assert t.names() == tuple(sorted(S.small_raw()["classes"]))
    assert t.names(tier=1) == t.core() == ("car", "chair_seat", "dining_table", "door_interior", "person_adult", "stair_riser")
    assert t.names(group="person") == ("person_adult",)
    assert t.names(group="furniture", tier=1) == ("chair_seat", "dining_table")
    assert t.groups() == tuple(sorted({e["group"] for e in S.small_raw()["classes"].values()}))
    assert t.names(group="no_such_group") == ()


def test_resolve_normalizes_text_and_aliases(small_table):
    t = small_table
    for q in ("door_interior", "Door Interior", "  DOOR-interior ", "interior_door", "Interior-Door", "door", "DOOR"):
        assert t.resolve(q).name == "door_interior", q
    assert t.resolve("human").name == "person_adult"
    assert t.resolve("car") is t["car"]
    for bad in ("nope", "", "   ", "door interior door"):
        with pytest.raises(KeyError):
            t.resolve(bad)


def test_lognormal_math(small_table):
    c = small_table["door_interior"]
    s = c.sigma_log
    assert c.mu_log == pytest.approx(math.log(2.03), abs=1e-12)
    assert c.quantile(0.5) == pytest.approx(2.03, rel=1e-9)
    assert c.quantile(0.975) == pytest.approx(2.03 * math.exp(1.959963984540054 * s), rel=1e-6)
    assert c.quantile(0.841344746068543) == pytest.approx(2.03 * math.exp(s), rel=1e-6)
    assert c.quantile(0.025) * c.quantile(0.975) == pytest.approx(2.03 ** 2, rel=1e-9)
    ps = np.linspace(0.01, 0.99, 25)
    q = [c.quantile(float(p)) for p in ps]
    assert all(b > a for a, b in zip(q, q[1:]))
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            c.quantile(bad)
    assert c.z(2.03) == pytest.approx(0.0, abs=1e-12)
    assert c.z(2.03 * math.exp(s)) == pytest.approx(1.0, abs=1e-9)
    assert c.z(2.03 * math.exp(-2 * s)) == pytest.approx(-2.0, abs=1e-9)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            c.z(bad)
    assert c.contains(2.03 * math.exp(1.9 * s)) and not c.contains(2.03 * math.exp(2.1 * s))
    assert c.contains(2.03 * math.exp(2.9 * s), k=3.0) and not c.contains(2.03 * math.exp(3.1 * s), k=3.0)


def test_arrays_for_downstream_ls(small_table):
    t = small_table
    mu, sg = t.arrays(["door_interior", "person", "car"])  # 별칭 허용
    assert mu.dtype == np.float64 and sg.dtype == np.float64 and mu.shape == sg.shape == (3,)
    np.testing.assert_allclose(mu, np.log([2.03, 1.70, 1.5]), atol=1e-12)
    np.testing.assert_allclose(sg, [0.04, 0.06, 0.06], atol=1e-12)
    m0, s0 = t.arrays([])
    assert m0.shape == s0.shape == (0,)
    with pytest.raises(KeyError):
        t.arrays(["car", "nope"])


def test_load_table_is_deterministic_and_does_not_touch_file(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(S.small_raw(), sort_keys=True), encoding="utf-8")
    before = p.read_bytes()
    a = T.load_table(p, limits=S.small_limits())
    b = T.load_table(p, limits=S.small_limits())
    assert a == b and a is not b
    assert p.read_bytes() == before


def test_load_table_rejects_invalid_with_all_problems(tmp_path):
    raw = S.small_raw()
    raw["classes"]["dog"]["sigma_log"] = 0.0
    raw["classes"]["tree"]["dim"] = "volume"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(T.SizePriorError) as ei:
        T.load_table(p, limits=S.small_limits())
    probs = ei.value.problems
    assert any(x.startswith("E_VALUE: dog") for x in probs) and any(x.startswith("E_ENUM: tree") for x in probs)
    assert isinstance(ei.value, ValueError)
    assert "E_VALUE: dog" in str(ei.value)
    with pytest.raises(T.SizePriorError):
        T.load_table(tmp_path / "missing.yaml", limits=S.small_limits())


# ---- 하류 이음매: RD-08의 로그 영역 가중 최소제곱 (pipeline.upg.md 2.6) ----
SEAM_CLASSES = ["door_interior", "chair_seat", "person_adult", "dining_table", "stair_riser", "building_story",
                "car", "coffee_mug", "wine_bottle", "window_residential"]


def solve_scale(mu, sigma, log_h):
    w = 1.0 / sigma ** 2
    log_s = float(np.sum(w * (mu - log_h)) / np.sum(w))
    return log_s, float(np.sum(w)) ** -0.5


def test_rd08_log_weighted_ls_seam_on_real_table():
    t = T.load_table(S.DATA)
    mu, sg = t.arrays(SEAM_CLASSES)
    assert np.all(np.isfinite(mu)) and np.all(sg > 0)
    s_true = 0.37  # m / 상대 단위
    # 모든 물체가 정확히 중앙값: 스케일을 비트 수준으로 복원
    log_h = mu - math.log(s_true)
    log_s, sig_s = solve_scale(mu, sg, log_h)
    assert math.exp(log_s) == pytest.approx(s_true, rel=1e-12)
    assert sig_s < float(sg.min())  # 증거를 모으면 어떤 단일 클래스보다 정밀
    # 몬테카를로 보정: 표에서 뽑은 로그정규 크기로 2σ_s 포함률이 N(0,1)의 95.45%에 맞는다
    rng = np.random.default_rng(20240607)
    hits, n = 0, 4000
    for _ in range(n):
        log_size = mu + sg * rng.standard_normal(len(mu))
        ls, ss = solve_scale(mu, sg, log_size - math.log(s_true))
        hits += abs(ls - math.log(s_true)) <= 2 * ss
    assert 0.93 <= hits / n <= 0.975


def test_size_lie_is_detectable_with_real_table():
    t = T.load_table(S.DATA)
    for name in ("door_interior", "person_adult", "soda_can", "wine_bottle"):
        c = t[name]
        for lie in (0.5, 2.0):  # 모델 학습 '크기 거짓말' ×0.5–2
            assert not c.contains(c.median_m * lie, k=3.0), (name, lie)
        assert c.contains(c.median_m * 1.02, k=3.0), name
