"""DAT-16 검사기(tools/size_prior_check.py) 계약: 오류 코드별 변조 탐지, 결정론, 입력 불변, CLI."""
import copy
import math
import subprocess
import sys

import pytest
import yaml

import dat16_support as S

T = S.tool()
LIM = S.small_limits()


def check(raw, limits=LIM):
    return T.check_raw(raw, limits=limits)


def mutated(fn, limits=LIM):
    raw = S.small_raw()
    fn(raw["classes"])
    return check(raw, limits)


def has(problems, code, where=None):
    return any(p.startswith(f"{code}: " + (f"{where}" if where else "")) for p in problems)


def scale_entry(e, k):
    e["median_m"] = e["median_m"] * k
    e["range_m"] = [e["range_m"][0] * k, e["range_m"][1] * k]


def test_valid_fixture_has_no_problems_and_ints_are_accepted():
    assert check(S.small_raw()) == []
    raw = S.small_raw()
    raw["classes"]["tree"]["median_m"] = 15  # YAML 정수
    raw["classes"]["tree"]["range_m"] = [6, 36]
    raw["classes"]["tree"]["sigma_log"] = math.log(36 / 6) / 4
    assert check(raw) == []


def test_checker_is_pure_sorted_and_deterministic():
    raw = S.small_raw()
    raw["classes"]["door_interior"]["median_m"] = -1.0
    raw["classes"]["dog"]["tier"] = 3
    raw["classes"]["Tree"] = raw["classes"].pop("tree")
    before = copy.deepcopy(raw)
    a = check(raw)
    b = check(raw)
    assert raw == before
    assert a == b and a == sorted(set(a)) and len(a) >= 3
    assert all(isinstance(p, str) and ": " in p for p in a)


@pytest.mark.parametrize("name,fn", [
    ("negative_median", lambda c: c["door_interior"].__setitem__("median_m", -2.03)),
    ("zero_sigma", lambda c: c["door_interior"].__setitem__("sigma_log", 0.0)),
    ("huge_sigma", lambda c: c["door_interior"].__setitem__("sigma_log", 5.0)),
    ("tiny_sigma", lambda c: c["door_interior"].__setitem__("sigma_log", 0.001)),
    ("nan_sigma", lambda c: c["door_interior"].__setitem__("sigma_log", float("nan"))),
    ("inf_median", lambda c: c["door_interior"].__setitem__("median_m", float("inf"))),
    ("range_inverted", lambda c: c["door_interior"].__setitem__("range_m", [2.2, 1.88])),
    ("range_nonpositive", lambda c: c["door_interior"].__setitem__("range_m", [0.0, 2.2])),
    ("median_outside_range", lambda c: c["door_interior"].__setitem__("median_m", 3.5)),
])
def test_value_errors_are_flagged_on_the_right_class(name, fn):
    p = mutated(fn)
    assert has(p, "E_VALUE", "door_interior"), (name, p)
    # 다른 클래스에는 영향 없음
    assert not any("stair_riser" in x for x in p)


def test_range_vs_median_sigma_consistency():
    # σ와 range가 어긋남(σ=0.04인데 range는 ±2σ가 아니라 훨씬 넓음)
    p = mutated(lambda c: c["door_interior"].__setitem__("range_m", [0.5, 8.0]))
    assert has(p, "E_CONSIST", "door_interior"), p
    # 중앙값이 range의 기하 중심에서 멀다(범위 안이지만 치우침)
    p = mutated(lambda c: c["door_interior"].update(range_m=[1.2, 2.5], sigma_log=math.log(2.5 / 1.2) / 4))
    assert has(p, "E_CONSIST", "door_interior"), p


@pytest.mark.parametrize("cls,k", [("person_adult", 100.0), ("coffee_mug", 100.0), ("door_interior", 100.0)])
def test_centimeter_slip_is_caught_by_group_bounds(cls, k):
    def fn(c):
        scale_entry(c[cls], k)
    p = mutated(fn)
    assert has(p, "E_UNIT", cls), p
    assert not has(p, "E_CONSIST", cls), "일관되게 ×100한 항목은 단위 오류로만 잡혀야 한다"


def test_group_bounds_table_covers_all_groups_and_is_sane():
    assert set(T.GROUP_BOUNDS) == set(T.GROUPS)
    for g, (lo, hi) in T.GROUP_BOUNDS.items():
        assert 0 < lo < hi <= 200, g
    assert T.GROUP_BOUNDS["person"][1] <= 2.5 < T.GROUP_BOUNDS["architecture"][1]
    assert set(T.DIMS) == {"height", "width", "length", "diameter", "depth"}


def test_name_rules():
    def rename(c):
        c["Door-Interior"] = c.pop("door_interior")
    assert has(mutated(rename), "E_NAME", "Door-Interior")

    def toolong(c):
        c["a" * 41] = c.pop("dog")
    assert has(mutated(toolong), "E_NAME")

    def digit_first(c):
        c["9lives"] = c.pop("dog")
    assert has(mutated(digit_first), "E_NAME")

    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", ["Hound Dog"])), "E_NAME", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", [""])), "E_NAME", "dog")


def test_alias_collisions():
    # 다른 클래스의 정본 이름과 충돌
    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", ["car"])), "E_DUP")
    # 두 클래스가 같은 별칭
    def same(c):
        c["dog"]["aliases"] = ["pet"]
        c["tree"]["aliases"] = ["pet"]
    assert has(mutated(same), "E_DUP")
    # 자기 이름과 같은 별칭, 같은 클래스 안 중복
    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", ["dog"])), "E_DUP")
    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", ["pet", "pet"])), "E_DUP")


def test_enum_violations():
    assert has(mutated(lambda c: c["dog"].__setitem__("dim", "volume")), "E_ENUM", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("group", "toys")), "E_ENUM", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("tier", 3)), "E_ENUM", "dog")


def test_schema_violations():
    def drop(key):
        return lambda c: c["dog"].pop(key)
    for key in ("median_m", "sigma_log", "range_m", "dim", "group", "tier"):
        assert has(mutated(drop(key)), "E_SCHEMA", "dog"), key
    assert has(mutated(lambda c: c["dog"].__setitem__("color", "brown")), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("median_m", "0.5")), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("median_m", True)), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("tier", True)), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("tier", 1.0)), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("range_m", [0.3, 0.5, 0.8])), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("range_m", [0.3, "x"])), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", "pet")), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("aliases", [3])), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c.__setitem__("dog", None)), "E_SCHEMA", "dog")
    assert has(mutated(lambda c: c["dog"].__setitem__("note", 5)), "E_SCHEMA", "dog")
    # note(문자열)는 선택 키로 허용
    assert mutated(lambda c: c["dog"].__setitem__("note", "어깨높이")) == []


def test_top_level_schema_violations():
    for fn in (
        lambda r: r.__setitem__("schema", 2),
        lambda r: r.__setitem__("unit", "cm"),
        lambda r: r.pop("schema"),
        lambda r: r.pop("unit"),
        lambda r: r.pop("classes"),
        lambda r: r.__setitem__("classes", [1, 2]),
        lambda r: r.__setitem__("extra", 1),
    ):
        raw = S.small_raw()
        fn(raw)
        assert has(check(raw), "E_SCHEMA"), raw.keys()
    for garbage in (None, 5, "x", [], [1], {}, {"schema": 1}, {"schema": 1, "unit": "m", "classes": None}):
        p = check(garbage)
        assert isinstance(p, list) and p and all(isinstance(x, str) for x in p)


def test_count_limits():
    # 기본 한도(최소 480)에서는 12개 표가 거부된다
    assert has(T.check_raw(S.small_raw()), "E_COUNT")
    assert T.DEFAULT_LIMITS == T.Limits(min_classes=480, max_classes=600, min_core=150)
    # 상한
    assert has(check(S.small_raw(), T.Limits(min_classes=1, max_classes=11, min_core=1)), "E_COUNT")
    # 핵심(tier 1) 개수
    raw = S.small_raw()
    for e in raw["classes"].values():
        e["tier"] = 2
    assert has(check(raw), "E_COUNT")
    raw = S.small_raw()
    raw["classes"] = {}
    assert has(check(raw), "E_COUNT")
    with pytest.raises(ValueError):
        T.Limits(min_classes=10, max_classes=5, min_core=1)
    with pytest.raises(ValueError):
        T.Limits(min_classes=10, max_classes=50, min_core=11)


def test_all_problems_are_reported_not_just_the_first():
    def many(c):
        c["door_interior"]["median_m"] = -1.0
        c["dog"]["dim"] = "volume"
        c["tree"]["aliases"] = ["car"]
        scale_entry(c["person_adult"], 100.0)
    p = mutated(many)
    assert has(p, "E_VALUE", "door_interior") and has(p, "E_ENUM", "dog")
    assert has(p, "E_DUP") and has(p, "E_UNIT", "person_adult")


# ---------------- CLI ----------------
def write_yaml(path, raw):
    path.write_text(yaml.safe_dump(raw, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return str(path)


SMALL_ARGS = ["--min-classes", "10", "--max-classes", "50", "--min-core", "5"]


def test_cli_exit_codes_and_output(tmp_path, capsys):
    good = write_yaml(tmp_path / "good.yaml", S.small_raw())
    assert T.main(["--path", good, *SMALL_ARGS]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "12" in out

    raw = S.small_raw()
    raw["classes"]["dog"]["sigma_log"] = 0.0
    bad = write_yaml(tmp_path / "bad.yaml", raw)
    assert T.main(["--path", bad, *SMALL_ARGS]) == 1
    out = capsys.readouterr().out
    assert "E_VALUE: dog" in out and "OK" not in out

    assert T.main(["--path", str(tmp_path / "missing.yaml"), *SMALL_ARGS]) == 2
    broken = tmp_path / "broken.yaml"
    broken.write_text("classes: [unclosed\n  - x: : :\n", encoding="utf-8")
    assert T.main(["--path", str(broken), *SMALL_ARGS]) == 2
    assert "E_PARSE" in capsys.readouterr().out


def test_cli_catches_duplicate_yaml_keys(tmp_path, capsys):
    text = yaml.safe_dump(S.small_raw(), sort_keys=False)  # schema, unit, classes 순 -> classes가 맨 끝
    # 같은 클래스 키를 한 번 더 쓴다: 평범한 yaml.safe_load는 조용히 덮어쓴다.
    text += "  dog:\n    dim: height\n    median_m: 9.0\n    sigma_log: 0.1\n    range_m: [7.0, 11.0]\n    group: animal\n    tier: 2\n"
    p = tmp_path / "dup.yaml"
    p.write_text(text, encoding="utf-8")
    assert yaml.safe_load(text)["classes"]["dog"]["median_m"] == 9.0  # 함정이 실제로 있음을 확인
    assert T.main(["--path", str(p), *SMALL_ARGS]) in (1, 2)
    assert "E_DUP" in capsys.readouterr().out
    with pytest.raises(T.SizePriorError) as ei:
        T.read_raw(p)
    assert any(x.startswith("E_DUP") for x in ei.value.problems)


def test_script_runs_standalone_from_any_cwd(tmp_path):
    good = write_yaml(tmp_path / "good.yaml", S.small_raw())
    r = subprocess.run([sys.executable, str(S.TOOL), "--path", good, *SMALL_ARGS], cwd=tmp_path,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(S.TOOL), "--path", good], cwd=tmp_path,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "E_COUNT" in r.stdout  # 기본 한도에서는 소형 표 거부


def test_import_has_no_heavy_deps_and_no_side_effects():
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('m',{str(S.TOOL)!r});m=importlib.util.module_from_spec(s);"
        "sys.modules['m']=m;s.loader.exec_module(m);"
        "print([x for x in ('torch','cv2','skimage','buc') if x in sys.modules])"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, cwd=S.ROOT.parent)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
