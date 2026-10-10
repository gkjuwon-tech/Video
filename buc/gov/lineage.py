"""GOV-01 Lineage Ledger CI + vault 구조 v1.

조상 DAG 해시, 조상 폐포의 BLOCK_TRAIN/BLOCK_DEV 검사, weights=None AST 린트,
체크포인트 해시 금지목록, vault 경로·마운트 규칙. 모든 판정은 닫힌 쪽 실패.

표준 라이브러리 + buc.gov.prov + buc.core.det 만 import 한다. torch 는 체크포인트
함수 안에서만 지연 import 하며 import 부작용이 없다.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from buc.core.det import hash_parts, merkle_root_keyed
from buc.gov.prov import (
    FirewallError, ProvRecord, as_record, derive, most_restrictive, parse_tag, why_not_dev, why_not_train,
)

__all__ = [
    "LINEAGE_TAG", "DENYLIST_TAG", "SD_TAG", "NODE_KINDS", "ARTIFACT_KINDS", "VAULT_NAMES",
    "FORBIDDEN_TRAIN_MOUNTS", "STUDENT_DIRS", "LineageError", "Node", "LineageLedger", "Denylist",
    "file_sha256", "state_dict_hash", "check_checkpoint", "assert_from_scratch", "Violation",
    "scan_source", "scan_paths", "vault_relpath", "check_mounts", "FirewallError",
]

LINEAGE_TAG = "buc.lineage.v1"
DENYLIST_TAG = "buc.denylist.v1"
SD_TAG = b"buc.sd.v1"
NODE_KINDS = ("source", "data", "op", "shard", "checkpoint", "calib", "cache", "eval")
ARTIFACT_KINDS = ("shard", "checkpoint", "calib", "cache")
VAULT_NAMES = ("user_vault", "ext_vault", "comp_vault", "pub_bench", "pub_bench_sealed",
               "hold_vault", "train_ok")
FORBIDDEN_TRAIN_MOUNTS = tuple(n for n in VAULT_NAMES if n != "train_ok") + ("soul_vault",)
STUDENT_DIRS = ("buc/models", "buc/train")

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_DENY_KEY_RE = re.compile(r"[0-9a-f]{8,64}")
_VAULT_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_EXT_QUAL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_PURPOSES = ("train", "dev")
_BRANCHES = ("main", "optin-v1")


class LineageError(ValueError):
    """원장·금지목록 구조 위반(닫힌 쪽 실패)."""


# --------------------------------------------------------------------------- 원장

@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    prov: ProvRecord
    parents: tuple = ()
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _ID_RE.fullmatch(self.id) is None:
            raise LineageError(f"id 형식 위반: {self.id!r}")
        if self.kind not in NODE_KINDS:
            raise LineageError(f"모르는 kind: {self.kind!r}")
        if not isinstance(self.prov, ProvRecord):
            raise LineageError("prov 는 ProvRecord 이어야 한다")
        if isinstance(self.parents, (str, bytes)):
            raise LineageError("parents 는 id 시퀀스여야 한다")
        try:
            parents = tuple(sorted(set(self.parents)))
        except TypeError:
            raise LineageError("parents 원소를 정렬할 수 없다") from None
        for p in parents:
            if not isinstance(p, str) or _ID_RE.fullmatch(p) is None:
                raise LineageError(f"부모 id 형식 위반: {p!r}")
        ch = self.content_hash
        if ch is not None and (not isinstance(ch, str) or _HASH_RE.fullmatch(ch) is None):
            raise LineageError(f"content_hash 는 소문자 hex 64자 또는 None: {ch!r}")
        object.__setattr__(self, "parents", parents)


def _prov_json(prov: ProvRecord) -> str:
    return json.dumps(prov.to_dict(), sort_keys=True, separators=(",", ":"))


class LineageLedger:
    """추가 전용 계보 DAG. 부모는 항상 먼저 존재하므로 비순환이다."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._hashes: dict[str, bytes] = {}

    # -- 추가 --------------------------------------------------------------
    def add(self, id: str, *, kind: str, prov=None, parents: Iterable[str] = (),
            content_hash: str | None = None) -> Node:
        if isinstance(parents, (str, bytes)):
            raise LineageError("parents 는 id 시퀀스여야 한다")
        if kind not in NODE_KINDS:
            raise LineageError(f"모르는 kind: {kind!r}")
        try:
            plist = list(parents)
        except TypeError:
            raise LineageError("parents 는 id 시퀀스여야 한다") from None
        for p in plist:
            if not isinstance(p, str) or p not in self._nodes:
                raise LineageError(f"부모가 원장에 없다: {p!r}")
        parent_provs = [self._nodes[p].prov for p in sorted(set(plist))]
        if prov is None:
            if not parent_provs:
                raise LineageError("소스 노드는 prov 가 필수다")
            rec = derive("lineage:" + kind, parent_provs)
        else:
            rec = as_record(prov)
            if parent_provs:
                self._check_no_laundering(rec, parent_provs, kind)
        node = Node(id=id, kind=kind, prov=rec, parents=tuple(plist), content_hash=content_hash)
        old = self._nodes.get(node.id)
        if old is not None:
            if old == node:
                return old
            raise LineageError(f"같은 id 가 다른 내용으로 이미 있다: {node.id!r}")
        self._nodes[node.id] = node
        self._hashes[node.id] = self._compute_hash(node)
        return node

    @staticmethod
    def _check_no_laundering(rec: ProvRecord, parent_provs: Sequence[ProvRecord], kind: str) -> None:
        op = "lineage:" + kind
        ok = False
        for human_ref in (False, True):
            floor = derive(op, parent_provs, human_ref=human_ref)
            if floor.taint <= rec.taint and most_restrictive(rec.ns, floor.ns) == rec.ns:
                ok = True
                break
        if not ok:
            raise LineageError("선언한 prov 가 부모 계보를 덮지 못한다(세탁 금지)")

    def _compute_hash(self, node: Node) -> bytes:
        return hash_parts(LINEAGE_TAG, node.id, node.kind, _prov_json(node.prov),
                          node.content_hash or "", len(node.parents),
                          *[self._hashes[p] for p in node.parents])

    # -- 조회 --------------------------------------------------------------
    def node(self, id: str) -> Node:
        return self._nodes[id]

    def __contains__(self, id: object) -> bool:
        return id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    def ancestors(self, id: str) -> tuple[str, ...]:
        if id not in self._nodes:
            raise KeyError(id)
        seen: set = set()
        stack = list(self._nodes[id].parents)
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(self._nodes[p].parents)
        return tuple(sorted(seen))

    # -- 해시 --------------------------------------------------------------
    def node_hash(self, id: str) -> bytes:
        return self._hashes[id]

    def root(self) -> bytes:
        return merkle_root_keyed(dict(self._hashes))

    # -- 직렬화 ------------------------------------------------------------
    def to_json(self) -> str:
        nodes = []
        for nid, n in self._nodes.items():
            nodes.append({"id": nid, "kind": n.kind, "prov": n.prov.to_dict(),
                          "parents": list(n.parents), "content_hash": n.content_hash,
                          "hash": self._hashes[nid].hex()})
        return json.dumps({"v": LINEAGE_TAG, "nodes": nodes, "root": self.root().hex()},
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "LineageLedger":
        def pairs(items):
            d = {}
            for k, v in items:
                if k in d:
                    raise LineageError(f"JSON 키 중복: {k!r}")
                d[k] = v
            return d

        def no_const(c):
            raise LineageError(f"JSON 상수 금지: {c}")

        if not isinstance(s, str):
            raise LineageError("from_json: str 이어야 한다")
        try:
            d = json.loads(s, object_pairs_hook=pairs, parse_constant=no_const)
        except LineageError:
            raise
        except (ValueError, RecursionError) as e:
            raise LineageError(f"깨진 JSON: {e}") from None
        if not isinstance(d, dict) or set(d) != {"v", "nodes", "root"}:
            raise LineageError("원장 JSON 최상위 형식 위반")
        if d["v"] != LINEAGE_TAG:
            raise LineageError(f"원장 버전 불일치: {d['v']!r}")
        if not isinstance(d["nodes"], list) or not isinstance(d["root"], str):
            raise LineageError("원장 JSON 형식 위반")
        lg = cls()
        for e in d["nodes"]:
            if not isinstance(e, dict) or set(e) != {"id", "kind", "prov", "parents", "content_hash", "hash"}:
                raise LineageError("노드 형식 위반")
            nid, parents = e["id"], e["parents"]
            if not isinstance(nid, str) or nid in lg:
                raise LineageError(f"노드 id 위반 또는 중복: {nid!r}")
            if (not isinstance(parents, list) or any(not isinstance(p, str) for p in parents)
                    or parents != sorted(set(parents))):
                raise LineageError(f"부모 목록은 정렬·중복 없는 id 목록이어야 한다: {nid!r}")
            if not isinstance(e["kind"], str) or not isinstance(e["hash"], str):
                raise LineageError(f"노드 형식 위반: {nid!r}")
            try:
                prov = ProvRecord.from_dict(e["prov"])
            except (ValueError, TypeError) as ex:
                raise LineageError(f"prov 위반 ({nid!r}): {ex}") from None
            try:
                lg.add(nid, kind=e["kind"], prov=prov, parents=parents, content_hash=e["content_hash"])
            except LineageError:
                raise
            except (ValueError, TypeError) as ex:
                raise LineageError(f"노드 위반 ({nid!r}): {ex}") from None
            if lg.node_hash(nid).hex() != e["hash"]:
                raise LineageError(f"노드 해시 불일치: {nid!r}")
        if lg.root().hex() != d["root"]:
            raise LineageError("root 불일치")
        return lg

    # -- 판정 --------------------------------------------------------------
    def why_not(self, id: str, purpose: str = "train", branch: str = "main",
                denylist: "Denylist | None" = None) -> tuple[str, ...]:
        if purpose not in _PURPOSES:
            raise ValueError(f"모르는 purpose: {purpose!r}")
        if branch not in _BRANCHES:
            raise ValueError(f"모르는 branch: {branch!r}")
        if purpose == "dev" and branch != "main":
            raise ValueError("dev 는 branch 'main' 만 가능하다")
        if id not in self._nodes:
            raise KeyError(id)
        out: set = set()
        for nid in (id, *self.ancestors(id)):
            n = self._nodes[nid]
            reasons = why_not_train(n.prov, branch) if purpose == "train" else why_not_dev(n.prov)
            out.update(f"{nid}:{r}" for r in reasons)
            if denylist is not None and n.content_hash is not None:
                name = denylist.match(n.content_hash)
                if name is not None:
                    out.add(f"{nid}:DENYLIST:{name}")
        return tuple(sorted(out))

    def assert_ok(self, id: str, purpose: str = "train", branch: str = "main",
                  denylist: "Denylist | None" = None) -> None:
        why = self.why_not(id, purpose, branch, denylist)
        if why:
            raise FirewallError(f"{purpose} 금지 ({id}): {', '.join(why)}")

    def check_all(self, purpose: str = "train", branch: str = "main",
                  denylist: "Denylist | None" = None) -> dict[str, tuple[str, ...]]:
        out: dict = {}
        for nid, n in self._nodes.items():
            if n.kind in ARTIFACT_KINDS:
                why = self.why_not(nid, purpose, branch, denylist)
                if why:
                    out[nid] = why
        return out


# --------------------------------------------------------------------------- 금지목록

class Denylist:
    """체크포인트 sha256(접두사 8–64 hex 자) → 이름."""

    def __init__(self, entries: Mapping[str, str] | Iterable = ()) -> None:
        if isinstance(entries, Mapping):
            pairs = list(entries.items())
        else:
            try:
                pairs = [tuple(p) for p in entries]
            except TypeError:
                raise LineageError("Denylist: Mapping 또는 (키, 이름) 쌍이어야 한다") from None
        self._entries: dict[str, str] = {}
        for pair in pairs:
            if len(pair) != 2:
                raise LineageError("Denylist: (키, 이름) 쌍이어야 한다")
            k, name = pair
            if not isinstance(k, str) or _DENY_KEY_RE.fullmatch(k.lower()) is None:
                raise LineageError(f"금지목록 키는 hex 8–64자: {k!r}")
            if not isinstance(name, str) or not name:
                raise LineageError(f"금지목록 이름은 비어 있지 않은 str: {name!r}")
            k = k.lower()
            if self._entries.get(k, name) != name:
                raise LineageError(f"금지목록 키 중복(이름 상이): {k!r}")
            self._entries[k] = name

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, h: str) -> str | None:
        if not isinstance(h, str):
            return None
        h = h.lower()
        best = None
        for k in self._entries:
            if h.startswith(k) and (best is None or len(k) > len(best)):
                best = k
        return None if best is None else self._entries[best]

    def to_json(self) -> str:
        ents = [{"sha256": k, "name": self._entries[k]} for k in sorted(self._entries)]
        return json.dumps({"v": DENYLIST_TAG, "entries": ents}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def load(cls, path) -> "Denylist":
        try:
            with open(os.fspath(path), "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError, RecursionError) as e:
            raise LineageError(f"금지목록을 읽을 수 없다: {e}") from None
        if not isinstance(d, dict) or set(d) != {"v", "entries"} or d["v"] != DENYLIST_TAG:
            raise LineageError("금지목록 형식·버전 위반")
        if not isinstance(d["entries"], list):
            raise LineageError("금지목록 entries 는 리스트")
        pairs = []
        for e in d["entries"]:
            if not isinstance(e, dict) or set(e) != {"sha256", "name"}:
                raise LineageError("금지목록 항목 형식 위반")
            pairs.append((e["sha256"], e["name"]))
        return cls(pairs)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(os.fspath(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_hash(sd: Mapping[str, Any]) -> str:
    """직렬화 형식·dict 순서·requires_grad 와 무관한 state dict 내용 해시(sha256 hex)."""
    import torch  # 지연 import

    if not isinstance(sd, Mapping):
        raise LineageError("state_dict_hash: Mapping 이어야 한다")
    h = hashlib.sha256(SD_TAG)
    for k in sorted(sd):
        t = sd[k]
        if not isinstance(k, str) or not isinstance(t, torch.Tensor):
            raise LineageError(f"state dict 항목은 str -> Tensor 이어야 한다: {k!r}")
        t = t.detach().cpu().contiguous()
        kb = k.encode("utf-8")
        db = str(t.dtype).encode("utf-8")
        h.update(struct.pack("<I", len(kb)) + kb)
        h.update(struct.pack("<I", len(db)) + db)
        h.update(struct.pack("<I", t.dim()))
        for d in t.shape:
            h.update(struct.pack("<Q", d))
        if t.numel():
            raw_t = torch.view_as_real(t) if t.is_complex() else t
            raw = raw_t.reshape(-1).view(torch.uint8).numpy()
            h.update(struct.pack("<Q", raw.nbytes))
            h.update(memoryview(raw))
        else:
            h.update(struct.pack("<Q", 0))
    return h.hexdigest()


def check_checkpoint(path, denylist: Denylist) -> tuple[str, ...]:
    """파일 해시·state dict 해시를 금지목록과 대조. 코드를 실행하지 않는다(weights_only)."""
    out: set = set()
    p = os.fspath(path)
    try:
        name = denylist.match(file_sha256(p))
    except OSError:
        return ("UNREADABLE",)
    if name is not None:
        out.add(f"FILE:{name}")
    try:
        import torch  # 지연 import

        obj = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(obj, Mapping) and isinstance(obj.get("state_dict"), Mapping):
            obj = obj["state_dict"]
        if not isinstance(obj, Mapping) or not all(
                isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()):
            raise LineageError("텐서 dict 아님")
        name = denylist.match(state_dict_hash(obj))
    except Exception:  # 읽기 실패는 어떤 형태든 닫힌 쪽
        out.add("UNREADABLE")
    else:
        if name is not None:
            out.add(f"STATE_DICT:{name}")
    return tuple(sorted(out))


def assert_from_scratch(module_or_sd, denylist: Denylist) -> None:
    sd = module_or_sd
    if not isinstance(sd, Mapping):
        fn = getattr(module_or_sd, "state_dict", None)
        if not callable(fn):
            raise TypeError("assert_from_scratch: nn.Module 또는 state dict 이어야 한다")
        sd = fn()
    name = denylist.match(state_dict_hash(sd))
    if name is not None:
        raise FirewallError(f"from-scratch 위반: 금지목록 가중치와 일치 ({name})")


# --------------------------------------------------------------------------- weights=None 린트

@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    detail: str = ""


_WATCHED = ("torchvision", "timm", "transformers", "diffusers", "open_clip",
            "segmentation_models_pytorch", "torch.hub")
_HUB_FUNCS = frozenset({"torch.hub.load", "torch.hub.load_state_dict_from_url",
                        "torch.utils.model_zoo.load_url"})
_WEIGHT_KW = ("weights", "weights_backbone")
_PRETRAINED_KW = ("pretrained", "pretrained_backbone")


def _is_watched(dotted: str) -> bool:
    return any(dotted == m or dotted.startswith(m + ".") for m in _WATCHED)


def _collect_aliases(tree: ast.AST) -> dict[str, set]:
    al: dict[str, set] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    al.setdefault(a.asname, set()).add(a.name)
                else:
                    top = a.name.split(".")[0]
                    al.setdefault(top, set()).add(top)
        elif isinstance(n, ast.ImportFrom):
            if n.level or not n.module:
                continue
            for a in n.names:
                if a.name != "*":
                    al.setdefault(a.asname or a.name, set()).add(f"{n.module}.{a.name}")
    return al


def _resolve(node: ast.AST, al: Mapping[str, set]) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return tuple(sorted(al.get(node.id, {node.id})))
    if isinstance(node, ast.Attribute):
        return tuple(sorted(f"{b}.{node.attr}" for b in _resolve(node.value, al)))
    return ()


def _is_none(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and n.value is None


def _is_false_or_none(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and (n.value is None or n.value is False)


def _scan_tree(tree: ast.AST, path: str) -> list[Violation]:
    al = _collect_aliases(tree)
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        line = getattr(node, "lineno", 1)
        names = _resolve(node.func, al)
        last = {nm.rsplit(".", 1)[-1] for nm in names}
        if isinstance(node.func, ast.Attribute):
            last.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            last.add(node.func.id)
        if any(_is_watched(nm) for nm in names):
            for kw in node.keywords:
                if kw.arg is None:
                    out.append(Violation(path, line, "KWARGS", "**kwargs 로는 weights 를 검증할 수 없다"))
                elif kw.arg in _WEIGHT_KW and not _is_none(kw.value):
                    out.append(Violation(path, line, "WEIGHTS", f"{kw.arg} 는 리터럴 None 이어야 한다"))
                elif kw.arg in _PRETRAINED_KW and not _is_false_or_none(kw.value):
                    out.append(Violation(path, line, "PRETRAINED", f"{kw.arg} 는 리터럴 False/None 이어야 한다"))
        if "from_pretrained" in last:
            out.append(Violation(path, line, "FROM_PRETRAINED", "from_pretrained 호출 금지"))
        if any(nm in _HUB_FUNCS for nm in names) or "load_state_dict_from_url" in last:
            out.append(Violation(path, line, "HUB", "사전학습 가중치 다운로드 호출 금지"))
    return out


def _sorted_violations(vs: Iterable[Violation]) -> list[Violation]:
    return sorted(set(vs), key=lambda v: (v.path, v.line, v.rule, v.detail))


def _scan(src, path: str) -> list[Violation]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        line = getattr(e, "lineno", None)
        line = line if isinstance(line, int) and line >= 1 else 1
        return [Violation(path, line, "SYNTAX", f"파싱 실패: {type(e).__name__}")]
    return _scan_tree(tree, path)


def scan_source(src: str, path: str = "<src>") -> list[Violation]:
    return _sorted_violations(_scan(src, path))


def scan_paths(paths: Iterable) -> list[Violation]:
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    files: list[str] = []
    out: list[Violation] = []
    for p in paths:
        p = os.fspath(p)
        if os.path.isdir(p):
            for dp, dns, fns in os.walk(p):
                dns.sort()
                files.extend(os.path.join(dp, fn) for fn in sorted(fns) if fn.endswith(".py"))
        elif os.path.isfile(p):
            files.append(p)
        else:
            out.append(Violation(p, 1, "SYNTAX", "경로를 읽을 수 없다"))
    for f in files:
        try:
            with open(f, "rb") as fh:
                data = fh.read()
        except OSError:
            out.append(Violation(f, 1, "SYNTAX", "파일을 읽을 수 없다"))
            continue
        out.extend(_scan(data, f))
    return _sorted_violations(out)


# --------------------------------------------------------------------------- vault

def vault_relpath(x, *, tenant: str | None = None, project: str | None = None) -> str:
    rec = as_record(x)
    ns = rec.ns
    if ns == "user_vault":
        for label, v in (("tenant", tenant), ("project", project)):
            if not isinstance(v, str) or _VAULT_PART_RE.fullmatch(v) is None:
                raise ValueError(f"user_vault 경로에는 안전한 {label} 가 필요하다: {v!r}")
        return f"user_vault/{tenant}/{project}/"
    if ns == "ext_vault":
        quals = set()
        for t in rec.taint:
            base, qual = parse_tag(t)
            if base in ("EXTGEN", "EXTGEN-DERIVED", "EXTGEN-INSPIRED"):
                quals.add(qual)
        if len(quals) == 1:
            g = next(iter(quals))
            if isinstance(g, str) and _EXT_QUAL_RE.fullmatch(g):
                return f"ext_vault/{g}/"
        return "ext_vault/_mixed/"
    if ns == "comp_vault":
        return "comp_vault/"
    if ns == "pub_bench":
        sealed = any(parse_tag(t)[0] == "PUB-SEALED" for t in rec.taint)
        return "pub_bench_sealed/" if sealed else "pub_bench/"
    if ns == "train_ok":
        return "train_ok/"
    raise ValueError(f"모르는 ns: {ns!r}")


def check_mounts(mounts: Iterable, purpose: str = "train") -> tuple[str, ...]:
    if purpose not in _PURPOSES:
        raise ValueError(f"모르는 purpose: {purpose!r}")
    if isinstance(mounts, (str, bytes, os.PathLike)):
        mounts = [mounts]
    bad = set()
    for m in mounts:
        s = os.fspath(m)
        if not isinstance(s, str):
            raise TypeError("마운트는 str 이어야 한다")
        low = s.lower()
        if any(name in low for name in FORBIDDEN_TRAIN_MOUNTS):
            bad.add(s)
    return tuple(sorted(bad))

