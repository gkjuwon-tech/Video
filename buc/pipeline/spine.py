"""PRD-01 파이프라인 스파인 v0 — 스테이지 DAG, Merkle 캐시, 부분 재실행, 스텁↔실물 교체.

FND-02 ``STAGE_SIGNATURES`` 가 정한 이음매 위에서 스테이지 구현(Impl)을 갈아 끼우며 돌리고, 캐시 키(계약 버전,
스테이지, 구현 kind·name·version, 입력 내용 해시, 실효 params)가 같은 스테이지는 다시 계산하지 않는다.
numpy 와 표준 라이브러리만 쓴다(torch 금지, import 부작용 없음). 계약서: ops/harness/contracts/PRD-01.md.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from buc.core import contracts, det
from buc.core.manifest import Manifest, make_manifest

__all__ = [
    "SPINE_STAGES", "SpineError", "StageFailed", "Impl", "StageContext", "Registry", "stage_seed", "stage_params",
    "stage_cache_key", "run_root", "MemoryCache", "DiskCache", "PlanNode", "Spine", "RunResult",
]

SEED_TAG = "buc.spine.v1"
SPINE_STAGES: Tuple[str, ...] = tuple(s for s in contracts.STAGES if s in contracts.STAGE_SIGNATURES)
_KINDS = ("stub", "real")
_KEY_RE = re.compile(r"[0-9a-f]{64}")


class SpineError(RuntimeError):
    """스파인 계획·실행 오류."""


class StageFailed(SpineError):
    """스테이지 구현이 실패했다. 원인은 ``__cause__``."""

    def __init__(self, stage: str, impl: str, message: Optional[str] = None):
        self.stage = stage
        self.impl = impl
        super().__init__(message if message is not None else f"스테이지 {stage} 실패 ({impl})")


# ---------------------------------------------------------------------------------------------------------------
# 구현 · 문맥 · 레지스트리
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Impl:
    """스테이지 구현 하나. ``fn(inputs, ctx) -> outputs`` (둘 다 서명 이름 → 계약 객체 dict)."""
    stage: str
    kind: str
    name: str
    version: str
    fn: Callable[..., Any]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or self.stage not in contracts.STAGE_SIGNATURES:
            raise ValueError(f"Impl.stage {self.stage!r} 는 서명이 있는 스테이지여야 한다 {sorted(contracts.STAGE_SIGNATURES)}")
        if self.kind not in _KINDS:
            raise ValueError(f"Impl.kind {self.kind!r} 는 {_KINDS} 중 하나여야 한다")
        for nm in ("name", "version"):
            v = getattr(self, nm)
            if not isinstance(v, str) or not v:
                raise ValueError(f"Impl.{nm} 은 비지 않은 str 이어야 한다 ({v!r})")
        if not callable(self.fn):
            raise ValueError("Impl.fn 은 호출 가능해야 한다")

    @property
    def ref(self) -> str:
        return f"{self.kind}:{self.name}@{self.version}"


@dataclass(frozen=True)
class StageContext:
    stage: str
    seed: int
    params: dict
    impl: Impl


class Registry:
    """(스테이지, kind) → Impl."""

    def __init__(self) -> None:
        self._impls: Dict[Tuple[str, str], Impl] = {}

    def register(self, impl: Impl, *, replace: bool = False) -> Impl:
        if not isinstance(impl, Impl):
            raise TypeError("register: Impl 이 필요하다")
        k = (impl.stage, impl.kind)
        if k in self._impls and not replace:
            raise ValueError(f"이미 등록됨: {impl.stage} {impl.kind} ({self._impls[k].ref})")
        self._impls[k] = impl
        return impl

    def get(self, stage: str, kind: str) -> Optional[Impl]:
        return self._impls.get((stage, kind))

    def resolve(self, stage: str, prefer: str = "real") -> Impl:
        if prefer not in _KINDS:
            raise ValueError(f"prefer {prefer!r} 는 {_KINDS} 중 하나여야 한다")
        other = "stub" if prefer == "real" else "real"
        impl = self._impls.get((stage, prefer)) or self._impls.get((stage, other))
        if impl is None:
            raise SpineError(f"스테이지 {stage} 에 등록된 구현이 없다")
        return impl


# ---------------------------------------------------------------------------------------------------------------
# 시드 · params · 키 · root
# ---------------------------------------------------------------------------------------------------------------
def stage_seed(seed: int, stage: str) -> int:
    return det.derive_seed(seed, SEED_TAG, stage)


def stage_params(stage: str, seed: int, params: Optional[Mapping] = None) -> dict:
    own = {} if params is None else params.get(stage, {})
    if own is None:
        own = {}
    if not isinstance(own, Mapping):
        raise ValueError(f"params[{stage!r}] 는 dict 여야 한다")
    if "_seed" in own:
        raise ValueError("params 에 '_seed' 키를 쓸 수 없다(스파인이 채운다)")
    out = copy.deepcopy(dict(own))
    out["_seed"] = stage_seed(seed, stage)
    return out


def stage_cache_key(stage: str, impl: Impl, input_hashes: Mapping[str, str], params: Mapping) -> str:
    return contracts.StageIO(stage=stage, impl_kind=impl.kind, impl_name=impl.name, impl_version=impl.version,
                             inputs=dict(input_hashes), outputs={}, params=params).cache_key


def run_root(stage_ios: Mapping[str, "contracts.StageIO"]) -> str:
    leaves = {s: det.hash_parts(io.cache_key, contracts.content_hash(io.outputs)) for s, io in stage_ios.items()}
    return det.merkle_root_keyed(leaves).hex()


# ---------------------------------------------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------------------------------------------
def _check_put(key: str, stage_io: "contracts.StageIO", outputs: Mapping) -> None:
    if stage_io.cache_key != key:
        raise ValueError("cache.put: key 가 stage_io.cache_key 와 다르다")
    if set(outputs) != set(stage_io.outputs):
        raise ValueError("cache.put: outputs 이름이 stage_io.outputs 와 다르다")


class MemoryCache:
    """프로세스 안 캐시. put·get 모두 to_dict/from_dict 사본을 주고받는다."""

    def __init__(self) -> None:
        self._d: Dict[str, Tuple[dict, Dict[str, dict]]] = {}

    def put(self, key: str, stage_io: "contracts.StageIO", outputs: Mapping) -> None:
        _check_put(key, stage_io, outputs)
        self._d[key] = (stage_io.to_dict(), {n: o.to_dict() for n, o in outputs.items()})

    def get(self, key: str):
        ent = self._d.get(key)
        if ent is None:
            return None
        try:
            io = contracts.StageIO.from_dict(ent[0])
            outs = {n: contracts.loads(json.dumps(d)) for n, d in ent[1].items()}
        except Exception:
            return None
        return io, outs

    def __contains__(self, key: object) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)


class DiskCache:
    """``<path>/<key[:2]>/<key>.json`` = {"stage_io":…, "outputs":{이름: 계약 dict}}. 손상은 miss."""

    def __init__(self, path) -> None:
        self.path = Path(path)

    def _file(self, key: str) -> Path:
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
            raise ValueError(f"캐시 키는 64자 소문자 hex 여야 한다 ({key!r})")
        return self.path / key[:2] / f"{key}.json"

    def put(self, key: str, stage_io: "contracts.StageIO", outputs: Mapping) -> None:
        _check_put(key, stage_io, outputs)
        p = self._file(key)
        text = json.dumps({"stage_io": stage_io.to_dict(), "outputs": {n: o.to_dict() for n, o in outputs.items()}},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get(self, key: str):
        try:
            p = self._file(key)
            if not p.is_file():
                return None
            d = json.loads(p.read_text(encoding="utf-8"))
            io = contracts.StageIO.from_dict(d["stage_io"])
            if io.cache_key != key:
                return None
            raw = d["outputs"]
            if not isinstance(raw, dict) or set(raw) != set(io.outputs):
                return None
            outs = {}
            for n, od in raw.items():
                obj = contracts.loads(json.dumps(od))
                if obj.content_hash() != io.outputs[n]:
                    return None
                outs[n] = obj
            return io, outs
        except Exception:
            return None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None


# ---------------------------------------------------------------------------------------------------------------
# 계획
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanNode:
    stage: str
    impl: Impl
    inputs: dict
    outputs: Tuple[str, ...]


def _stage_tuple(name: str, v) -> Tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str):
        v = (v,)
    t = tuple(v)
    for s in t:
        if s not in SPINE_STAGES:
            raise ValueError(f"{name}: 알 수 없는 스테이지 {s!r} (스파인 스테이지 {SPINE_STAGES})")
    return t


def _norm_prefer(prefer, what: str = "prefer"):
    """None | 'real'|'stub' | {stage: kind} → 검증된 값."""
    if prefer is None:
        return None
    if isinstance(prefer, str):
        if prefer not in _KINDS:
            raise ValueError(f"{what} {prefer!r} 는 {_KINDS} 중 하나여야 한다")
        return prefer
    if isinstance(prefer, Mapping):
        out = {}
        for s, k in prefer.items():
            if s not in SPINE_STAGES:
                raise ValueError(f"{what}: 알 수 없는 스테이지 {s!r}")
            if k not in _KINDS:
                raise ValueError(f"{what}[{s!r}] {k!r} 는 {_KINDS} 중 하나여야 한다")
            out[s] = k
        return out
    raise ValueError(f"{what} 는 str 또는 dict 여야 한다")


@dataclass
class RunResult:
    artifacts: Dict[str, Any]
    stage_ios: Dict[str, "contracts.StageIO"]
    status: Dict[str, str]
    root: str
    seed: int
    config_hash: Optional[str] = None

    @property
    def ran(self) -> Tuple[str, ...]:
        return tuple(s for s, v in self.status.items() if v in ("run", "fallback"))

    @property
    def hits(self) -> Tuple[str, ...]:
        return tuple(s for s, v in self.status.items() if v == "hit")

    def manifest(self, name, config, *, git_commit=None, created_utc=None) -> Manifest:
        if getattr(config, "seed", None) != self.seed:
            raise ValueError("manifest: config.seed 가 실행 seed 와 다르다")
        s1 = self.stage_ios.get("S-1")
        image_hash = s1.inputs["image"] if s1 is not None else self.artifacts["image"].content_hash()
        impls = {s: f"{io.impl_kind}:{io.impl_name}@{io.impl_version}" for s, io in self.stage_ios.items()}
        return make_manifest(name, config, inputs={"image": image_hash}, params={"root": self.root, "impls": impls},
                             git_commit=git_commit, created_utc=created_utc)


# ---------------------------------------------------------------------------------------------------------------
# 스파인
# ---------------------------------------------------------------------------------------------------------------
class Spine:
    def __init__(self, registry: Registry, *, cache=None, prefer="real", allow_fallback: bool = True):
        if not isinstance(registry, Registry):
            raise TypeError("registry 는 Registry 여야 한다")
        self.registry = registry
        self.cache = MemoryCache() if cache is None else cache
        p = _norm_prefer(prefer)
        self.prefer = "real" if p is None else p
        self.allow_fallback = bool(allow_fallback)

    # -- 선호 해석
    def _prefer_for(self, stage: str, run_prefer) -> str:
        if isinstance(run_prefer, str):
            return run_prefer
        if isinstance(run_prefer, dict) and stage in run_prefer:
            return run_prefer[stage]
        if isinstance(self.prefer, str):
            return self.prefer
        return self.prefer.get(stage, "real")

    # -- 계획
    def plan(self, *, skip=(), until=None, prefer=None) -> List[PlanNode]:
        skip_t = _stage_tuple("skip", skip)
        if until is not None and until not in SPINE_STAGES:
            raise ValueError(f"until: 알 수 없는 스테이지 {until!r} (스파인 스테이지 {SPINE_STAGES})")
        run_prefer = _norm_prefer(prefer)
        stages = SPINE_STAGES if until is None else SPINE_STAGES[:SPINE_STAGES.index(until) + 1]
        producers = {"image": "input"}
        nodes: List[PlanNode] = []
        for st in stages:
            if st in skip_t:
                continue
            ins, outs = contracts.STAGE_SIGNATURES[st]
            missing = [n for n in ins if n not in producers]
            if missing:
                raise SpineError(f"스테이지 {st}: 입력 {missing} 의 생산자가 없다 (skip={skip_t})")
            impl = self.registry.resolve(st, self._prefer_for(st, run_prefer))
            nodes.append(PlanNode(stage=st, impl=impl, inputs={n: producers[n] for n in ins}, outputs=tuple(outs)))
            for n in outs:
                producers[n] = st
        return nodes

    # -- 실행
    def run(self, image, *, seed=None, config=None, params=None, skip=(), until=None, force=(),
            prefer=None) -> RunResult:
        if not isinstance(image, contracts.InputImage):
            raise TypeError("image 는 InputImage 여야 한다")
        if config is not None:
            if seed is not None and seed != config.seed:
                raise ValueError(f"seed={seed} 가 config.seed={config.seed} 와 다르다")
            seed = config.seed
        elif seed is None:
            seed = 0
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed 는 정수여야 한다")
        if params is not None and not isinstance(params, Mapping):
            raise ValueError("params 는 {스테이지: {…}} 여야 한다")
        force_t = _stage_tuple("force", force)
        nodes = self.plan(skip=skip, until=until, prefer=prefer)
        run_prefer = _norm_prefer(prefer)
        eff = {n.stage: stage_params(n.stage, seed, params) for n in nodes}   # 실행 전에 검증

        artifacts: Dict[str, Any] = {"image": image}
        stage_ios: Dict[str, contracts.StageIO] = {}
        status: Dict[str, str] = {}
        for node in nodes:
            st = node.stage
            ins_names = contracts.STAGE_SIGNATURES[st][0]
            inputs = {k: artifacts[k] for k in ins_names}
            p = eff[st]
            forced = st in force_t
            io, outs, how, err = self._attempt(node.impl, inputs, p, seed, forced)
            if err is None:
                if how == "hit" and io.fallback is not None:
                    io = dataclasses.replace(io, fallback=None)
            else:
                stub = self.registry.get(st, "stub")
                if node.impl.kind == "real" and self.allow_fallback and stub is not None:
                    fb = f"{node.impl.ref}: {type(err).__name__}"
                    io, outs, how, err2 = self._attempt(stub, inputs, p, seed, forced)
                    if err2 is not None:
                        raise StageFailed(st, stub.ref, f"스테이지 {st} 실패 ({stub.ref}): {err2}") from err2
                    io = dataclasses.replace(io, fallback=fb)
                    how = "fallback"
                else:
                    raise StageFailed(st, node.impl.ref, f"스테이지 {st} 실패 ({node.impl.ref}): {err}") from err
            stage_ios[st] = io
            status[st] = how
            artifacts.update(outs)
        return RunResult(artifacts=artifacts, stage_ios=stage_ios, status=status, root=run_root(stage_ios),
                         seed=seed, config_hash=None if config is None else config.hash())

    def _attempt(self, impl: Impl, inputs: Dict[str, Any], p: dict, seed: int, forced: bool):
        """구현 하나로 1–3 단계. (io, outputs, 'hit'|'run', None) 또는 (None, None, None, 예외)."""
        st = impl.stage
        input_hashes = {k: v.content_hash() for k, v in inputs.items()}
        key = stage_cache_key(st, impl, input_hashes, p)
        if not forced:
            got = self.cache.get(key)
            if got is not None:
                io, outs = got
                try:
                    if (io.stage != st or io.cache_key != key or set(outs) != set(io.outputs)):
                        raise contracts.ContractError("캐시 항목이 맞지 않는다")
                    contracts.check_stage_io(st, inputs, outs)
                except Exception:
                    pass        # 불량 캐시 항목 = miss
                else:
                    return io, outs, "hit", None
        ctx = StageContext(stage=st, seed=stage_seed(seed, st), params=copy.deepcopy(p), impl=impl)
        try:
            t0 = time.perf_counter()
            outs = impl.fn(dict(inputs), ctx)
            wall = time.perf_counter() - t0
            after = {k: v.content_hash() for k, v in inputs.items()}
            if after != input_hashes:
                changed = sorted(k for k in after if after[k] != input_hashes[k])
                raise contracts.ContractError(f"{st}: 구현이 입력을 변이했다 {changed}")
            io = contracts.make_stage_io(st, impl.kind, impl.name, impl.version, inputs, outs,
                                         params=copy.deepcopy(p), wall_s=wall)
            outs = dict(outs)
            if io.cache_key != key:
                raise SpineError(f"{st}: StageIO.cache_key 와 계산한 키가 다르다")
        except Exception as e:
            return None, None, None, e
        self.cache.put(key, io, outs)
        return io, outs, "run", None
