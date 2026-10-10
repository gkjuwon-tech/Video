"""식 파서: 우선순위·결합·함수·오류. eval 없이 자체 파서여야 한다."""
import math
import pathlib
import re

import pytest


@pytest.mark.parametrize("src,want", [
    ("1+2*3", 7.0), ("(1+2)*3", 9.0), ("-2*3", -6.0), ("--2", 2.0), ("+3", 3.0), ("-(1+2)", -3.0),
    ("10/4", 2.5), ("8-3-2", 3.0), ("2*3/4", 1.5), ("100/10/5", 2.0), ("1.5e1+.5", 15.5), ("2e3", 2000.0),
    ("sum(1,2,3)", 6.0), ("sum(2)", 2.0), ("min(4,2,9)", 2.0), ("max(4, 2, 9)", 9.0),
    ("max(1, sum(2,3)*2)", 10.0), ("  1 +  1 ", 2.0), ("2*-3", -6.0), ("1-(-1)", 2.0),
    ("min(sum(1,2), 2.5)", 2.5), ("((((7))))", 7.0),
])
def test_eval_constants(lc, src, want):
    got = lc.eval_formula(src, {})
    assert isinstance(got, float)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_eval_with_env_dotted_ids_and_assoc(lc):
    env = {"world.tfu.a": 179, "world.tfu.b": 10.5, "x": 100.0, "y": 5, "z": 2}
    assert lc.eval_formula("world.tfu.a + 2*world.tfu.b", env) == pytest.approx(200.0)
    assert lc.eval_formula("x-y-z", env) == pytest.approx(93.0)      # 좌결합
    assert lc.eval_formula("x/y/z", env) == pytest.approx(10.0)      # 좌결합
    assert lc.eval_formula("x-y*z", env) == pytest.approx(90.0)
    assert lc.eval_formula("sum(x,y,z)/3", env) == pytest.approx(107 / 3)
    assert lc.eval_formula("x-y", {"x": 3, "y": 3.0}) == 0.0
    # 하이픈은 id 문자가 아니라 뺄셈이다
    assert lc.eval_formula("x-y", env) == pytest.approx(95.0)
    assert lc.eval_formula("x-z", env) == pytest.approx(98.0)


@pytest.mark.parametrize("src", [
    "", "   ", "1+", "+", "*2", "(1+2", "1+2)", "()", "1 2", "foo(1)", "sin(1)", "abs(-1)", "sum()", "min(1,)",
    "sum(1 2)", "1/0", "1 $ 2", "__import__('os').getcwd()", "a.b.", ".a", "1..2", "2**3", "1e999", "1,2", "x y",
    "sum(1,2", "max)1(", "1+*2", "'a'", "[1]", "1;2",
])
def test_eval_rejects_malformed(lc, src):
    with pytest.raises(lc.FormulaError):
        lc.eval_formula(src, {})


def test_eval_unknown_id_and_zero_division_with_env(lc):
    with pytest.raises(lc.FormulaError):
        lc.eval_formula("a + b", {"a": 1})
    with pytest.raises(lc.FormulaError):
        lc.eval_formula("a/(b-b)", {"a": 1, "b": 2})
    with pytest.raises(lc.FormulaError):
        lc.eval_formula("a*b", {"a": 1e200, "b": 1e200})  # inf
    assert issubclass(lc.FormulaError, ValueError)
    assert issubclass(lc.LedgerFormatError, ValueError)


def test_eval_does_not_mutate_env_and_is_pure(lc):
    env = {"a": 2, "b": 3}
    snap = dict(env)
    r1 = lc.eval_formula("a*b+a", env)
    r2 = lc.eval_formula("a*b+a", env)
    assert env == snap and r1 == r2 == 8.0
    assert not math.isnan(r1)


def test_formula_refs_order_dedup_and_functions(lc):
    assert lc.formula_refs("a + b*a - sum(c.d, a)") == ("a", "b", "c.d")
    assert lc.formula_refs("2*3") == ()
    assert lc.formula_refs("sum(x)") == ("x",)          # 함수 이름은 참조가 아니다
    assert lc.formula_refs("max(min(p.q, 1), r_1)") == ("p.q", "r_1")
    assert lc.formula_refs("world.tfu.a-world.tfu.b") == ("world.tfu.a", "world.tfu.b")
    assert isinstance(lc.formula_refs("a"), tuple)
    with pytest.raises(lc.FormulaError):
        lc.formula_refs("a +")
    with pytest.raises(lc.FormulaError):
        lc.formula_refs("a $ b")


def test_source_has_no_eval_or_exec():
    src = (pathlib.Path(__file__).resolve().parents[2] / "tools" / "ledger_check.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(eval|exec)\s*\(", src), "식은 자체 파서로 계산해야 한다"
    assert "literal_eval" not in src
