"""GOV-03 렌더 경로 린트 세트 v1 (buc.lints.v1).

I6(Read-Only Camera)와 비회귀 가드 R-NR1·R-NR11·(e)를 기계 검사로 만든다.
  1. no_input_pixels_in_render_path  렌더 경로는 입력 버퍼를 import·참조하지 않는다
  2. canvas_extension_lint           교사를 입력 크롭으로 초기화하거나 캔버스를 넓히지 않는다
  3. refl_bake_lint                  반사 영역을 입력 픽셀로 덮지 않고, 잔차 흡수는 M_refl 을 뺀다
  4. preset_lineage_check            공유 프리셋은 ENGINE/CAPTURE/ASSET 에서만, θ_in 은 프리셋이 되지 않는다

모든 판정은 정적(AST, 실행하지 않음)·결정론·닫힌 쪽이다(파싱 실패 = 위반 SYNTAX).
표준 라이브러리 + buc.gov.prov 만 모듈 수준에서 import 한다(torch·numpy 미적재).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from buc.gov import prov as _prov

__all__ = [
    "LINT_TAG", "LINTS", "REPO_ROOT", "RENDER_PATH_DIRS", "CANVAS_DIRS", "REFL_DIRS", "PRESET_SCAN_DIRS",
    "PRESET_EXEMPT_DIRS", "INPUT_MODULES", "INPUT_IDENTS", "INPUT_ATTRS", "INPUT_TYPES", "IMAGE_READ_FUNCS",
    "OUTPAINT_RE", "INIT_KWARGS", "PAD_FUNCS", "REFL_RE", "MASKED_FUNCS", "BLEND_FUNCS",
    "PRESET_ALLOWED_BASES", "PRESET_PROVENANCE", "Violation", "PresetEntry", "scan_render_source",
    "check_render_path", "scan_canvas_source", "scan_refl_source", "scan_preset_source",
    "preset_provenance_record", "why_not_preset", "check_presets", "check_preset_dir", "scan_paths",
    "run_all", "main",
]

LINT_TAG = "buc.lints.v1"
LINTS = ("no_input_pixels_in_render_path", "canvas_extension_lint", "refl_bake_lint", "preset_lineage_check")
_L_RENDER, _L_CANVAS, _L_REFL, _L_PRESET = LINTS

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_PATH_DIRS = ("buc/render", "buc/lens", "buc/camera")
CANVAS_DIRS = ("buc/teacher", "buc/world", "buc/invent", "buc/render", "buc/lens", "buc/camera", "buc/models")
REFL_DIRS = CANVAS_DIRS
PRESET_SCAN_DIRS = ("buc",)
PRESET_EXEMPT_DIRS = ("buc/gov",)

INPUT_MODULES = ("buc.read",)
INPUT_IDENTS = frozenset({"image_lin", "input_image", "input_pixels", "input_rgb", "input_buffer", "I_w"})
INPUT_ATTRS = INPUT_IDENTS | {"pixels"}
INPUT_TYPES = frozenset({"InputImage", "ReadPacket"})
IMAGE_READ_FUNCS = frozenset({"imread", "read_image", "decode_image", "decode_jpeg", "decode_png"})
OUTPAINT_RE = re.compile(
    r"(?i)(out_?paint|un_?crop|extend_?canvas|canvas_?exten(d|sion)|expand_?canvas|border_?extend|extend_?border)")
INIT_KWARGS = frozenset({"init", "init_image", "init_latent", "init_latents", "x0", "x_init", "z0", "z_init",
                         "image", "latents", "start_image", "source_image", "init_rgb"})
PAD_FUNCS = frozenset({"pad", "copyMakeBorder"})
REFL_RE = re.compile(r"(?i)^(m_?refl\w*|refl(ection)?_?mask\w*|mask_?refl\w*)$")
MASKED_FUNCS = frozenset({"masked_scatter", "masked_scatter_", "masked_fill", "masked_fill_", "putmask", "place",
                          "copyto", "index_put", "index_put_"})
BLEND_FUNCS = frozenset({"lerp", "mix", "blend", "composite", "alpha_blend", "over"})
PRESET_ALLOWED_BASES = frozenset({"ENGINE", "CAPTURE", "ASSET"})
PRESET_PROVENANCE = {"handmade": "ENGINE", "engine": "ENGINE", "capture": "CAPTURE"}

_INPUT_NAMES = INPUT_IDENTS | INPUT_TYPES
_INPUT_ATTR_NAMES = INPUT_ATTRS | INPUT_TYPES
_IMPORT_FUNCS = frozenset({"import_module", "__import__"})
_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError)


# ------------------------------------------------------------------------------------------------ 공통
@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    lint: str
    rule: str
    detail: str

    def to_dict(self) -> dict:
        return {"lint": self.lint, "path": self.path, "line": self.line, "rule": self.rule,
                "detail": self.detail}


def _finish(vs: Iterable[Violation]) -> list:
    return sorted(set(vs))


def _line(node: ast.AST) -> int:
    return int(getattr(node, "lineno", 1) or 1)


def _parse(src: str, path: str, lint: str):
    """(tree, None) 또는 (None, [SYNTAX 위반])."""
    try:
        return ast.parse(src), None
    except SyntaxError as e:
        return None, [Violation(path, int(e.lineno or 1), lint, "SYNTAX", str(e.msg))]
    except _PARSE_ERRORS as e:
        return None, [Violation(path, 1, lint, "SYNTAX", type(e).__name__)]


def _end_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dotted(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return None if inner is None else inner + "." + node.attr
    return None


def _has_input_ref(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in _INPUT_NAMES:
            return True
        if isinstance(n, ast.Attribute) and n.attr in _INPUT_ATTR_NAMES:
            return True
    return False


def _is_refl_ident(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return REFL_RE.match(node.id) is not None
    if isinstance(node, ast.Attribute):
        return REFL_RE.match(node.attr) is not None
    return False


def _has_refl_ident(node: ast.AST) -> bool:
    return any(_is_refl_ident(n) for n in ast.walk(node))


def _is_input_module(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in INPUT_MODULES)


def _module_from_rel(rel: str) -> str:
    parts = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(level: int, name: Optional[str], module: Optional[str], is_pkg: bool) -> Optional[str]:
    if module is None:
        return None
    pkg = module if is_pkg else module.rpartition(".")[0]
    parts = pkg.split(".") if pkg else []
    up = level - 1
    if up >= len(parts):
        return None
    parts = parts[:len(parts) - up]
    if name:
        parts = parts + [name]
    return ".".join(parts) if parts else None


def _from_base(node: ast.ImportFrom, module: Optional[str], is_pkg: bool) -> Optional[str]:
    if node.level:
        return _resolve_relative(node.level, node.module, module, is_pkg)
    return node.module


def _literal_import_target(call: ast.Call, module: Optional[str], is_pkg: bool):
    """import_module/__import__ 호출이면 ("lit", 이름|None) / ("dyn", None) / None(해당 없음)."""
    if _end_name(call.func) not in _IMPORT_FUNCS:
        return None
    arg = call.args[0] if call.args else None
    if arg is None:
        for kw in call.keywords:
            if kw.arg == "name":
                arg = kw.value
    if arg is None:
        return None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        s = arg.value
        if s.startswith("."):
            stripped = s.lstrip(".")
            return ("lit", _resolve_relative(len(s) - len(stripped), stripped or None, module, is_pkg))
        return ("lit", s)
    return ("dyn", None)


def _alias_map(tree: ast.AST, module: Optional[str], is_pkg: bool) -> dict:
    m: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    m[a.asname] = a.name
                else:
                    root = a.name.split(".")[0]
                    m[root] = root
        elif isinstance(n, ast.ImportFrom):
            base = _from_base(n, module, is_pkg)
            if base is None:
                continue
            for a in n.names:
                if a.name != "*":
                    m[a.asname or a.name] = base + "." + a.name
    return m


def _resolved_dotted(func: ast.AST, aliases: dict) -> Optional[str]:
    d = _dotted(func)
    if d is None:
        return None
    head, sep, rest = d.partition(".")
    if head in aliases:
        return aliases[head] + (sep + rest if sep else "")
    return d


def _is_pkg_path(path: str) -> bool:
    return os.path.basename(path) == "__init__.py"


# ------------------------------------------------------------------------------------------------ lint 1
def scan_render_source(src: str, path: str = "<src>", module: Optional[str] = None) -> list:
    L = _L_RENDER
    tree, err = _parse(src, path, L)
    if err:
        return _finish(err)
    is_pkg = _is_pkg_path(path)
    aliases = _alias_map(tree, module, is_pkg)
    out: list = []

    def add(node, rule, detail):
        out.append(Violation(path, _line(node), L, rule, detail))

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if _is_input_module(a.name):
                    add(n, "INPUT_IMPORT", a.name)
        elif isinstance(n, ast.ImportFrom):
            base = _from_base(n, module, is_pkg)
            for a in n.names:
                if a.name in INPUT_TYPES:
                    add(n, "INPUT_TYPE", a.name)
                if base is None:
                    continue
                for cand in (base, base + "." + a.name):
                    if _is_input_module(cand):
                        add(n, "INPUT_IMPORT", cand)
                        break
        elif isinstance(n, ast.Name):
            if n.id in INPUT_IDENTS:
                add(n, "INPUT_NAME", n.id)
            if n.id in INPUT_TYPES:
                add(n, "INPUT_TYPE", n.id)
        elif isinstance(n, ast.Attribute):
            if n.attr in INPUT_ATTRS:
                add(n, "INPUT_NAME", n.attr)
            if n.attr in INPUT_TYPES:
                add(n, "INPUT_TYPE", n.attr)
        elif isinstance(n, ast.arg):
            if n.arg in INPUT_IDENTS:
                add(n, "INPUT_NAME", n.arg)
        elif isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg in INPUT_IDENTS:
                    add(kw, "INPUT_NAME", kw.arg)
            tgt = _literal_import_target(n, module, is_pkg)
            if tgt is not None:
                kind, name = tgt
                if kind == "dyn":
                    add(n, "DYNAMIC_IMPORT", _end_name(n.func) or "")
                elif name and _is_input_module(name):
                    add(n, "INPUT_IMPORT", name)
            end = _end_name(n.func)
            if end in IMAGE_READ_FUNCS:
                add(n, "IMAGE_READ", end)
            elif _resolved_dotted(n.func, aliases) == "PIL.Image.open":
                add(n, "IMAGE_READ", "PIL.Image.open")
    return _finish(out)


def _iter_py(root: Path, dirs: Iterable[str], exempt: Iterable[str] = ()):
    """(상대 POSIX 경로, 절대 경로) 정렬·중복 제거."""
    seen: dict = {}
    ex = tuple(e.strip("/") for e in exempt)
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts or not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if any(rel == e or rel.startswith(e + "/") for e in ex):
                continue
            seen[rel] = p
    return sorted(seen.items())


def _read(p: Path):
    try:
        return p.read_text(encoding="utf-8"), None
    except (OSError, UnicodeDecodeError) as e:
        return None, type(e).__name__


class _ImportGraph:
    """root 아래 buc.* 정적 import 그래프(지연 구성, 파일당 한 번만 파싱)."""

    def __init__(self, root: Path):
        self.root = root
        self._edges: dict = {}
        self._exists: dict = {}

    def _file(self, name: str):
        base = self.root.joinpath(*name.split("."))
        for cand in (Path(str(base) + ".py"), base / "__init__.py"):
            if cand.is_file():
                return cand
        return None

    def valid(self, name: str) -> bool:
        if _is_input_module(name):
            return True
        if name not in self._exists:
            base = self.root.joinpath(*name.split("."))
            self._exists[name] = self._file(name) is not None or base.is_dir()
        return self._exists[name]

    def edges(self, name: str) -> dict:
        """모듈 name 의 {대상 모듈: 첫 줄}."""
        if name in self._edges:
            return self._edges[name]
        out: dict = {}
        self._edges[name] = out
        f = self._file(name)
        if f is None or _is_input_module(name):
            return out
        text, _ = _read(f)
        if text is None:
            return out
        try:
            tree = ast.parse(text)
        except _PARSE_ERRORS:
            return out
        is_pkg = f.name == "__init__.py"

        def put(target, line):
            if target and self.valid(target) and (target not in out or line < out[target]):
                out[target] = line

        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    put(a.name, _line(n))
            elif isinstance(n, ast.ImportFrom):
                base = _from_base(n, name, is_pkg)
                if base is None:
                    continue
                for a in n.names:
                    cand = base + "." + a.name if a.name != "*" else None
                    put(cand if cand and self.valid(cand) else base, _line(n))
            elif isinstance(n, ast.Call):
                tgt = _literal_import_target(n, name, is_pkg)
                if tgt is not None and tgt[0] == "lit":
                    put(tgt[1], _line(n))
        return out

    def chain_to_input(self, start: str):
        """start 에서 처음 닿는 입력 모듈까지의 최단 사슬([start, ..., input]) 또는 None."""
        parent: dict = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in sorted(self.edges(cur)):
                if nxt in parent:
                    continue
                parent[nxt] = cur
                if _is_input_module(nxt):
                    chain = [nxt]
                    while parent[chain[-1]] is not None:
                        chain.append(parent[chain[-1]])
                    return chain[::-1]
                q.append(nxt)
        return None


def check_render_path(root=REPO_ROOT, dirs: Sequence[str] = RENDER_PATH_DIRS) -> list:
    root = Path(root)
    L = _L_RENDER
    graph = _ImportGraph(root)
    out: list = []
    for rel, p in _iter_py(root, dirs):
        module = _module_from_rel(rel)
        text, err = _read(p)
        if text is None:
            out.append(Violation(rel, 1, L, "SYNTAX", err or "unreadable"))
            continue
        out.extend(scan_render_source(text, rel, module))
        chain = graph.chain_to_input(module)
        if chain is not None and len(chain) >= 3:
            out.append(Violation(rel, graph.edges(module)[chain[1]], L, "TRANSITIVE_IMPORT",
                                 " -> ".join(chain)))
    return _finish(out)


# ------------------------------------------------------------------------------------------------ lint 2
def scan_canvas_source(src: str, path: str, module: Optional[str] = None) -> list:
    L = _L_CANVAS
    tree, err = _parse(src, path, L)
    if err:
        return _finish(err)
    aliases = _alias_map(tree, module, _is_pkg_path(path))
    out: list = []

    def add(node, rule, detail):
        out.append(Violation(path, _line(node), L, rule, detail))

    def outpaint(node, name):
        if isinstance(name, str) and OUTPAINT_RE.search(name):
            add(node, "OUTPAINT_API", name)

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            outpaint(n, n.name)
        elif isinstance(n, ast.Name):
            outpaint(n, n.id)
        elif isinstance(n, ast.Attribute):
            outpaint(n, n.attr)
        elif isinstance(n, ast.Import):
            for a in n.names:
                outpaint(n, a.name)
                outpaint(n, a.asname)
        elif isinstance(n, ast.ImportFrom):
            outpaint(n, n.module)
            for a in n.names:
                outpaint(n, a.name)
                outpaint(n, a.asname)
        elif isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg in INIT_KWARGS and _has_input_ref(kw.value):
                    add(n, "INPUT_INIT", kw.arg)
            end = _end_name(n.func)
            dotted = _dotted(n.func) or ""
            resolved = _resolved_dotted(n.func, aliases) or ""
            is_pad = end in PAD_FUNCS
            is_expand = dotted.endswith("ImageOps.expand") or resolved.endswith("ImageOps.expand")
            if is_pad or is_expand:
                vals = list(n.args) + [kw.value for kw in n.keywords]
                if any(_has_input_ref(v) for v in vals):
                    add(n, "PAD_INPUT", end if is_pad else "ImageOps.expand")
    return _finish(out)


# ------------------------------------------------------------------------------------------------ lint 3
def _strip_subscript(node: ast.AST) -> ast.AST:
    while isinstance(node, ast.Subscript):
        node = node.value
    return node


def _is_absorb_name(name: Optional[str]) -> bool:
    if not name:
        return False
    low = name.lower()
    return "absorb" in low and any(k in low for k in ("resid", "lowband", "low_band"))


def _all_params(args: ast.arguments) -> list:
    ps = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        ps.append(args.vararg)
    if args.kwarg:
        ps.append(args.kwarg)
    return [p.arg for p in ps]


def scan_refl_source(src: str, path: str, module: Optional[str] = None) -> list:
    L = _L_REFL
    tree, err = _parse(src, path, L)
    if err:
        return _finish(err)
    out: list = []

    def add(node, rule, detail):
        out.append(Violation(path, _line(node), L, rule, detail))

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            end = _end_name(n.func)
            if end == "where":
                if (n.args and _has_refl_ident(n.args[0])
                        and any(_has_input_ref(a) for a in n.args[1:3])):
                    add(n, "REFL_INPUT", "where")
            elif end in MASKED_FUNCS or end in BLEND_FUNCS:
                vals = list(n.args) + [kw.value for kw in n.keywords]
                if any(_has_refl_ident(v) for v in vals) and any(_has_input_ref(v) for v in vals):
                    add(n, "REFL_INPUT", "masked" if end in MASKED_FUNCS else "blend")
            if _is_absorb_name(end):
                ok = any(_has_refl_ident(a) for a in n.args)
                for kw in n.keywords:
                    if ok:
                        break
                    if kw.arg is not None and REFL_RE.match(kw.arg):
                        v = kw.value
                        if not (isinstance(v, ast.Constant) and (v.value is None or v.value is False)):
                            ok = True
                    elif _has_refl_ident(kw.value):
                        ok = True
                if not ok:
                    add(n, "ABSORB_NO_REFL", end)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            value = getattr(n, "value", None)
            if value is not None and _has_input_ref(value):
                if any(isinstance(s, ast.Subscript) and _has_refl_ident(s.slice)
                       for t in targets for s in ast.walk(t)):
                    add(n, "REFL_INPUT", "assign")
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            for a, b in ((n.left, n.right), (n.right, n.left)):
                if _is_refl_ident(_strip_subscript(a)) and _has_input_ref(b):
                    add(n, "REFL_INPUT", "mult")
                    break
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_absorb_name(n.name):
            refl_params = {p for p in _all_params(n.args) if REFL_RE.match(p)}
            used = {m.id for stmt in n.body for m in ast.walk(stmt)
                    if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)}
            if not (refl_params & used):
                add(n, "ABSORB_NO_REFL", n.name)
    return _finish(out)


# ------------------------------------------------------------------------------------------------ lint 4
def _names_in(node: ast.AST) -> list:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.Attribute):
            out.append(n.attr)
    return out


def _mentions_preset(node: ast.AST) -> bool:
    return any("preset" in s.lower() for s in _names_in(node))


def _mentions_theta_in(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == "theta_in":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "theta_in":
            return True
    return False


def scan_preset_source(src: str, path: str, module: Optional[str] = None) -> list:
    L = _L_PRESET
    tree, err = _parse(src, path, L)
    if err:
        return _finish(err)
    out: list = []

    def add(node):
        out.append(Violation(path, _line(node), L, "THETA_IN_TO_PRESET", "theta_in"))

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if _mentions_preset(n.func):
                vals = list(n.args) + [kw.value for kw in n.keywords]
                if any(_mentions_theta_in(v) for v in vals):
                    add(n)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            value = getattr(n, "value", None)
            if value is not None and _mentions_theta_in(value):
                if any(isinstance(s, ast.Subscript) and _mentions_preset(s.value)
                       for t in targets for s in ast.walk(t)):
                    add(n)
    return _finish(out)


def preset_provenance_record(s) -> "_prov.ProvRecord":
    if not isinstance(s, str):
        raise ValueError(f"프리셋 provenance 는 str 이어야 한다: {type(s).__name__}")
    if s in PRESET_PROVENANCE:
        return _prov.ProvRecord(taint=frozenset({PRESET_PROVENANCE[s]}))
    if s.startswith("asset:"):
        lic = s[len("asset:"):]
        if lic:
            try:
                return _prov.ProvRecord(taint=frozenset({"ASSET:" + lic}))
            except ValueError as e:
                raise ValueError(f"프리셋 provenance 위반: {s!r}") from e
    raise ValueError(f"모르는 프리셋 provenance: {s!r}")


def why_not_preset(prov) -> tuple:
    try:
        rec = _prov.as_record(prov)
    except Exception:  # noqa: BLE001 - 읽을 수 없으면 닫힌 쪽
        return ("UNREADABLE",)
    reasons = set()
    if not rec.taint:
        reasons.add("NO_TAINT")
    for tag in rec.taint:
        if tag.partition(":")[0] not in PRESET_ALLOWED_BASES:
            reasons.add(f"TAINT:{tag}")
    if rec.ns != "train_ok":
        reasons.add(f"NS:{rec.ns}")
    return tuple(sorted(reasons))


@dataclass(frozen=True)
class PresetEntry:
    name: str
    theta: Any
    prov: Any


def _theta_hash(theta) -> str:
    from buc.lens.theta import complete_extra  # 지연 import: 모듈 import 시 torch·numpy 미적재
    return complete_extra(theta).content_hash()


def check_presets(entries: Iterable, *, project_thetas: Iterable = ()) -> list:
    L = _L_PRESET
    out: list = []
    project_hashes = None
    for e in entries:
        if isinstance(e, PresetEntry):
            name, theta, prov = e.name, e.theta, e.prov
        else:
            name, theta = e.name, e.theta
            try:
                prov = preset_provenance_record(e.provenance)
            except ValueError:
                out.append(Violation(str(name), 0, L, "PROVENANCE", str(e.provenance)))
                prov = None
        if prov is not None:
            for reason in why_not_preset(prov):
                rule, _, detail = reason.partition(":")
                out.append(Violation(str(name), 0, L, rule, detail))
        if project_hashes is None:
            project_hashes = set()
            for t in project_thetas:
                try:
                    project_hashes.add(_theta_hash(t))
                except Exception:  # noqa: BLE001
                    continue
        if project_hashes:
            try:
                h = _theta_hash(theta)
            except Exception:  # noqa: BLE001
                out.append(Violation(str(name), 0, L, "UNREADABLE", "theta"))
                continue
            if h in project_hashes:
                out.append(Violation(str(name), 0, L, "THETA_IN_COPY", h[:16]))
    return _finish(out)


def check_preset_dir(presets_dir=None, *, project_thetas: Iterable = ()) -> list:
    from buc.lens import theta as _theta  # 지연 import
    d = Path(_theta.PRESETS_DIR if presets_dir is None else presets_dir)
    L = _L_PRESET
    out: list = []
    entries: list = []
    files = sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))) if d.is_dir() else []
    for p in files:
        try:
            entries.append(_theta.load_preset(p))
        except Exception as e:  # noqa: BLE001 - 읽을 수 없는 프리셋 = 위반
            out.append(Violation(p.name, 0, L, "UNREADABLE", type(e).__name__))
    out.extend(check_presets(entries, project_thetas=tuple(project_thetas)))
    return _finish(out)


# ------------------------------------------------------------------------------------------------ 묶음
_SCANNERS: dict = {
    _L_CANVAS: (scan_canvas_source, CANVAS_DIRS),
    _L_REFL: (scan_refl_source, REFL_DIRS),
    _L_PRESET: (scan_preset_source, PRESET_SCAN_DIRS),
}


def scan_paths(lint: str, root=REPO_ROOT, dirs: Optional[Sequence[str]] = None, exempt: Sequence[str] = ()) -> list:
    if lint not in LINTS:
        raise ValueError(f"모르는 lint: {lint!r}")
    root = Path(root)
    if lint == _L_RENDER:
        return check_render_path(root, RENDER_PATH_DIRS if dirs is None else tuple(dirs))
    scanner, default_dirs = _SCANNERS[lint]
    ex = tuple(exempt) + (PRESET_EXEMPT_DIRS if lint == _L_PRESET else ())
    out: list = []
    for rel, p in _iter_py(root, default_dirs if dirs is None else tuple(dirs), ex):
        text, err = _read(p)
        if text is None:
            out.append(Violation(rel, 1, lint, "SYNTAX", err or "unreadable"))
            continue
        out.extend(scanner(text, rel, _module_from_rel(rel)))
    return _finish(out)


def run_all(root=REPO_ROOT, lints: Sequence[str] = LINTS) -> dict:
    lints = tuple(lints)
    for lint in lints:
        if lint not in LINTS:
            raise ValueError(f"모르는 lint: {lint!r}")
    root = Path(root)
    res: dict = {}
    for lint in LINTS:
        if lint not in lints:
            continue
        vs = scan_paths(lint, root)
        if lint == _L_PRESET and (root / "presets").is_dir():
            vs = _finish(vs + check_preset_dir(root / "presets"))
        res[lint] = vs
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m buc.gov.lints", description="GOV-03 렌더 경로 린트 세트")
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--lint", action="append", choices=LINTS, default=None)
    args = ap.parse_args(argv)
    res = run_all(Path(args.root), tuple(args.lint) if args.lint else LINTS)
    vs = sorted(v for lst in res.values() for v in lst)
    for v in vs:
        print(f"{v.path}:{v.line}: {v.lint} {v.rule} {v.detail}", file=sys.stderr)
    print(json.dumps({"ok": not vs, "violations": [v.to_dict() for v in vs]}, ensure_ascii=False,
                     sort_keys=True))
    return 1 if vs else 0


if __name__ == "__main__":
    sys.exit(main())
