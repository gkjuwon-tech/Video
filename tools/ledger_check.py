"""수치 원장(docs/number-ledger.yaml) 재계산 검사기 (FND-05).

모든 숫자는 원장 한 파일에 둔다. 파생 숫자는 식으로 적고, 이 도구가 식을 다시 계산해
선언값과 허용 오차(기본 1%) 넘게 어긋나면 위반으로 보고한다.
표준 라이브러리와 PyYAML만 쓴다. buc 패키지는 import 하지 않는다.

CLI: python tools/ledger_check.py [--ledger PATH] [--tol X]
종료 코드: 0 위반 없음, 1 위반(YAML 오류 포함), 2 사용법 오류·파일 없음.
stdout 마지막 줄은 JSON 한 줄이다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import re
import sys
from collections.abc import Iterator, Mapping
from typing import Any

import yaml

SCHEMA = "buc.number-ledger/v1"
DEFAULT_TOL = 0.01
DEFAULT_LEDGER = "docs/number-ledger.yaml"
ROOT = pathlib.Path(__file__).resolve().parent.parent

_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*")
_WS_RE = re.compile(r"\s*")
_TOKEN_RE = re.compile(
    r"(?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<id>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)"
    r"|(?P<op>[-+*/(),])"
)
_FUNCS = frozenset({"sum", "min", "max"})


class FormulaError(ValueError):
    """식 문법 오류, 모르는 함수·id, 0 나눗셈, 비유한 결과."""


class LedgerFormatError(ValueError):
    """호출자가 잘못 준 인자(tol 등). 데이터 문제는 예외가 아니라 위반이다."""


@dataclasses.dataclass(frozen=True)
class Violation:
    check: str
    id: str
    why: str


@dataclasses.dataclass(frozen=True)
class Report:
    ok: bool
    violations: tuple[Violation, ...]
    computed: dict[str, float]
    checked: int


# ---------------------------------------------------------------- 식 파서
# 문법: expr = term {(+|-) term}, term = unary {(*|/) unary},
#       unary = (-|+) unary | atom, atom = 숫자 | id | 함수(expr, ...) | (expr)
# 결과 노드: ("num", float) | ("ref", id) | ("neg", node) | ("bin", op, l, r) | ("call", name, [args])


def _tokenize(src: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    pos = 0
    n = len(src)
    while True:
        pos = _WS_RE.match(src, pos).end()
        if pos >= n:
            break
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            raise FormulaError(f"알 수 없는 문자 {src[pos]!r} (위치 {pos})")
        pos = m.end()
        if m.group("num") is not None:
            toks.append(("num", m.group("num")))
        elif m.group("id") is not None:
            toks.append(("id", m.group("id")))
        else:
            toks.append(("op", m.group("op")))
    return toks


class _Parser:
    def __init__(self, toks: list[tuple[str, str]]) -> None:
        self.toks = toks
        self.i = 0

    def _peek_op(self) -> str | None:
        if self.i < len(self.toks) and self.toks[self.i][0] == "op":
            return self.toks[self.i][1]
        return None

    def _take(self) -> tuple[str, str]:
        if self.i >= len(self.toks):
            raise FormulaError("식이 중간에 끝났다")
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect_close(self) -> None:
        if self._peek_op() != ")":
            raise FormulaError("닫는 괄호 ')' 가 필요하다")
        self.i += 1

    def expr(self) -> tuple:
        node = self.term()
        while self._peek_op() in ("+", "-"):
            op = self._take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self) -> tuple:
        node = self.unary()
        while self._peek_op() in ("*", "/"):
            op = self._take()[1]
            node = ("bin", op, node, self.unary())
        return node

    def unary(self) -> tuple:
        op = self._peek_op()
        if op == "-":
            self.i += 1
            return ("neg", self.unary())
        if op == "+":
            self.i += 1
            return self.unary()
        return self.atom()

    def atom(self) -> tuple:
        kind, val = self._take()
        if kind == "num":
            x = float(val)
            if not math.isfinite(x):
                raise FormulaError(f"숫자가 범위를 벗어났다: {val}")
            return ("num", x)
        if kind == "id":
            if self._peek_op() == "(":
                if val not in _FUNCS:
                    raise FormulaError(f"모르는 함수: {val}")
                self.i += 1
                args = [self.expr()]
                while self._peek_op() == ",":
                    self.i += 1
                    args.append(self.expr())
                self._expect_close()
                return ("call", val, args)
            return ("ref", val)
        if kind == "op" and val == "(":
            node = self.expr()
            self._expect_close()
            return node
        raise FormulaError(f"예상치 못한 토큰: {val!r}")


def _parse(formula: Any) -> tuple:
    if not isinstance(formula, str):
        raise FormulaError("식은 문자열이어야 한다")
    try:
        toks = _tokenize(formula)
        if not toks:
            raise FormulaError("빈 식")
        p = _Parser(toks)
        node = p.expr()
        if p.i != len(toks):
            raise FormulaError(f"식 뒤에 남는 토큰: {toks[p.i][1]!r}")
    except RecursionError:
        raise FormulaError("식이 너무 깊다") from None
    return node


def _nodes(root: tuple) -> Iterator[tuple]:
    """노드를 원문 순서(전위, 왼쪽 우선)로 돈다. 재귀 없음."""
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        k = n[0]
        if k == "neg":
            stack.append(n[1])
        elif k == "bin":
            stack.append(n[3])
            stack.append(n[2])
        elif k == "call":
            stack.extend(reversed(n[2]))


def _refs(node: tuple) -> tuple[str, ...]:
    out: dict[str, None] = {}
    for n in _nodes(node):
        if n[0] == "ref":
            out.setdefault(n[1], None)
    return tuple(out)


def _is_additive(node: tuple) -> bool:
    return not any(n[0] == "bin" and n[1] in ("*", "/") for n in _nodes(node))


def _fin(x: float, what: str) -> float:
    if not math.isfinite(x):
        raise FormulaError(f"비유한 결과 ({what})")
    return x


def _ev(node: tuple, env: Mapping[str, Any]) -> float:
    k = node[0]
    if k == "num":
        return node[1]
    if k == "ref":
        name = node[1]
        if name not in env:
            raise FormulaError(f"식의 입력이 없다: {name}")
        try:
            v = float(env[name])
        except (TypeError, ValueError):
            raise FormulaError(f"입력 값이 수치가 아니다: {name}") from None
        return _fin(v, name)
    if k == "neg":
        return -_ev(node[1], env)
    if k == "bin":
        a = _ev(node[2], env)
        b = _ev(node[3], env)
        op = node[1]
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        else:
            if b == 0:
                raise FormulaError("0 으로 나눌 수 없다")
            r = a / b
        return _fin(r, op)
    vals = [_ev(a, env) for a in node[2]]
    name = node[1]
    if name == "sum":
        r = sum(vals)
    elif name == "min":
        r = min(vals)
    else:
        r = max(vals)
    return _fin(float(r), name)


def formula_refs(formula: str) -> tuple[str, ...]:
    """식이 참조하는 id 를 첫 등장 순서로, 중복 없이 돌려준다. 문법 오류는 FormulaError."""
    return _refs(_parse(formula))


def eval_formula(formula: str, env: Mapping[str, float]) -> float:
    """env 의 값으로 식을 계산한다. env 에 없는 id, 0 나눗셈, 비유한 결과는 FormulaError."""
    return float(_ev(_parse(formula), env))


# ---------------------------------------------------------------- 검사


def _is_real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _valid_tol(x: Any) -> bool:
    return _is_real(x) and math.isfinite(x) and 0 <= x < 1


def _tol_arg(tol: Any) -> float | None:
    if tol is None:
        return None
    if not _valid_tol(tol):
        raise LedgerFormatError(f"tol 은 [0, 1) 의 유한 수치여야 한다: {tol!r}")
    return float(tol)


def _cyclic_ids(graph: dict[str, list[str]]) -> set[str]:
    """반복형 Tarjan SCC. 자기 루프 포함, 순환에 속한 모든 노드를 돌려준다."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: set[str] = set()
    counter = 0
    for root in sorted(graph):
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(graph.get(root, ())))]
        while work:
            v, it = work[-1]
            descended = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(graph.get(w, ()))))
                    descended = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1 or v in graph.get(v, ()):
                    out.update(comp)
    return out


def _report(found: list[Violation], computed: dict[str, float], checked: int) -> Report:
    violations = tuple(sorted(found, key=lambda v: (v.id, v.check, v.why)))
    return Report(ok=not violations, violations=violations, computed=computed, checked=checked)


def check_ledger(doc: Any, *, tol: float | None = None) -> Report:
    """원장 문서(이미 읽은 매핑)를 검사한다. 데이터 문제는 위반으로 보고하고 예외를 던지지 않는다.

    tol 인자 > 문서 tolerance > DEFAULT_TOL 순으로 허용 오차를 정한다.
    """
    tol_arg = _tol_arg(tol)
    found: list[Violation] = []

    def add(check: str, id_: str, why: str) -> None:
        found.append(Violation(check, id_, why))

    if not isinstance(doc, Mapping):
        add("schema", "", "최상위가 매핑이 아니다")
        return _report(found, {}, 0)

    if doc.get("schema") != SCHEMA:
        add("schema", "", f"schema 는 {SCHEMA} 이어야 한다 (받음: {doc.get('schema')!r})")

    doc_tol: float | None = None
    raw_tol = doc.get("tolerance")
    if raw_tol is not None:
        if _valid_tol(raw_tol):
            doc_tol = float(raw_tol)
        else:
            add("schema", "", f"tolerance 는 [0, 1) 의 유한 수치여야 한다: {raw_tol!r}")
    if tol_arg is not None:
        eff_tol = tol_arg
    elif doc_tol is not None:
        eff_tol = doc_tol
    else:
        eff_tol = DEFAULT_TOL

    owners: frozenset[str] | None = None
    raw_owners = doc.get("owners")
    if raw_owners is not None:
        if isinstance(raw_owners, list) and all(isinstance(o, str) and o.strip() for o in raw_owners):
            owners = frozenset(raw_owners)
        else:
            add("schema", "", "owners 는 문자열 목록이어야 한다")

    entries = doc.get("entries")
    if not isinstance(entries, list) or not entries:
        add("schema", "", "entries 는 비어 있지 않은 목록이어야 한다")
        return _report(found, {}, 0)
    checked = len(entries)

    # 중복 id: id 마다 1건
    counts: dict[str, int] = {}
    for e in entries:
        if isinstance(e, Mapping) and isinstance(e.get("id"), str):
            counts[e["id"]] = counts.get(e["id"], 0) + 1
    for rid in sorted(counts):
        if counts[rid] > 1:
            add("duplicate_id", rid, f"id 가 {counts[rid]}번 나온다")

    # 1단계: entry 형식 검사. 형식이 맞는 entry 만 rows 에 남긴다.
    rows: list[dict[str, Any]] = []
    for idx, e in enumerate(entries):
        if not isinstance(e, Mapping):
            add("schema", "", f"entries[{idx}] 가 매핑이 아니다")
            continue
        raw_id = e.get("id")
        eid = raw_id if isinstance(raw_id, str) else ""
        good = True
        if not (isinstance(raw_id, str) and _ID_RE.fullmatch(raw_id)):
            add("schema", eid, f"id 형식 위반: {raw_id!r}")
            good = False
        value = e.get("value")
        if not _is_real(value) or not math.isfinite(value):
            add("schema", eid, f"value 는 유한 수치여야 한다: {value!r}")
            good = False
        for label in ("unit", "owner"):
            v = e.get(label)
            if not (isinstance(v, str) and v.strip()):
                add("schema", eid, f"{label} 는 비어 있지 않은 문자열이어야 한다")
                good = False
        formula = e.get("formula")
        if formula is not None and not isinstance(formula, str):
            add("schema", eid, f"formula 는 문자열 또는 null 이어야 한다: {formula!r}")
            good = False
        if not good:
            continue
        rows.append({
            "id": raw_id,
            "value": float(value),
            "unit": e["unit"],
            "owner": e["owner"],
            "formula": formula,
            "inputs": e.get("inputs"),
        })

    # 같은 id 가 여럿이면 첫 번째 유효 entry 를 계산에 쓴다.
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        table.setdefault(row["id"], row)

    computed: dict[str, float] = {}
    deps: dict[str, list[str]] = {}

    for row in rows:
        rid = row["id"]
        is_first = table[rid] is row

        if owners is not None and row["owner"] not in owners:
            add("owner", rid, f"owner {row['owner']!r} 는 owners 목록 밖이다")

        if row["formula"] is None:
            if row["inputs"] not in (None, []):
                add("inputs_mismatch", rid, f"잎 entry 는 inputs 를 비우거나 생략해야 한다: {row['inputs']!r}")
            continue

        try:
            node = _parse(row["formula"])
        except FormulaError as ex:
            add("formula", rid, str(ex))
            continue
        refs = _refs(node)
        if is_first:
            deps[rid] = [r for r in refs if r in table]

        declared = row["inputs"] if row["inputs"] is not None else []
        if (
            not isinstance(declared, list)
            or not all(isinstance(x, str) for x in declared)
            or len(set(declared)) != len(declared)
            or set(declared) != set(refs)
        ):
            add("inputs_mismatch", rid, f"inputs {declared!r} 가 식의 참조 {list(refs)!r} 와 정확히 같지 않다")

        unknown = [r for r in refs if r not in table]
        if unknown:
            add("unknown_input", rid, f"원장에 없는 입력: {unknown}")
            continue

        if _is_additive(node):
            bad_units = [r for r in refs if table[r]["unit"] != row["unit"]]
            if bad_units:
                add("unit", rid, f"가산 식의 입력 단위가 출력 단위 {row['unit']!r} 와 다르다: {bad_units}")

        if not is_first:
            continue

        env = {r: table[r]["value"] for r in refs}
        try:
            comp = _ev(node, env)
        except FormulaError as ex:
            add("formula", rid, str(ex))
            continue
        computed[rid] = comp
        value = row["value"]
        if comp == 0:
            if value != 0:
                add("mismatch", rid, f"재계산값이 0 인데 선언값은 {value:g} 이다 (0 기준 상대오차는 정의되지 않는다)")
        else:
            rel = abs(value - comp) / abs(comp)
            if rel > eff_tol + 1e-12:
                add(
                    "mismatch",
                    rid,
                    f"선언값 {value:g} 과 재계산 {comp:.6g} 의 상대오차 {rel * 100:.3f}% 가 "
                    f"허용 {eff_tol * 100:g}% 를 넘는다",
                )

    for rid in sorted(_cyclic_ids(deps)):
        add("cycle", rid, "파생 식의 의존 그래프에서 순환에 속한다")

    return _report(found, computed, checked)


def _schema_only(msg: str) -> Report:
    return Report(ok=False, violations=(Violation("schema", "", msg),), computed={}, checked=0)


def check_file(path: str | pathlib.Path, *, tol: float | None = None) -> Report:
    """원장 파일을 읽어 검사한다. 파일이 없으면 FileNotFoundError, YAML 오류는 schema 위반이다."""
    tol_arg = _tol_arg(tol)
    raw = pathlib.Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as ex:
        return _schema_only(f"UTF-8 디코딩 실패: {ex}")
    try:
        data = yaml.safe_load(text)
    except (yaml.YAMLError, ValueError, RecursionError) as ex:
        return _schema_only(f"YAML 구문 오류: {ex}")
    return check_ledger(data, tol=tol_arg)


# ---------------------------------------------------------------- CLI


def _tol_cli(s: str) -> float:
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"수치가 아니다: {s!r}") from None
    if not (math.isfinite(x) and 0 <= x < 1):
        raise argparse.ArgumentTypeError(f"[0, 1) 범위 밖이다: {s}")
    return x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ledger_check", description="수치 원장 재계산 검사 (FND-05)")
    ap.add_argument("--ledger", default=None, help=f"원장 경로 (기본: 저장소 루트 기준 {DEFAULT_LEDGER})")
    ap.add_argument("--tol", type=_tol_cli, default=None, help="허용 상대오차 [0, 1) (기본: 문서 값 또는 0.01)")
    ns = ap.parse_args(argv)

    path = pathlib.Path(ns.ledger) if ns.ledger else ROOT / DEFAULT_LEDGER
    if not path.is_file():
        print(f"원장 파일이 없다: {path}", file=sys.stderr)
        return 2
    try:
        rep = check_file(path, tol=ns.tol)
    except OSError as ex:
        print(f"원장을 읽을 수 없다: {ex}", file=sys.stderr)
        return 2

    if rep.ok:
        print(f"OK: 원장 entry {rep.checked}건 검사, 위반 없음 ({path})")
    else:
        print(f"위반 {len(rep.violations)}건 ({path}):")
        for v in rep.violations:
            print(f"  [{v.check}] {v.id or '-'}: {v.why}")
    payload = {
        "ok": rep.ok,
        "checked": rep.checked,
        "violations": [{"check": v.check, "id": v.id, "why": v.why} for v in rep.violations],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
