# PRD-01 파이프라인 스파인 v0 — 스테이지 DAG · Merkle 캐시 · 부분 재실행 · 스텁↔실물 교체

**목적.** FND-02 `STAGE_SIGNATURES`가 정한 이음매 위에서 스테이지 구현을 갈아 끼우며(스텁↔실물) 돌리고, 입력이 같은 스테이지는 다시 계산하지 않는다. 걷는 해골(PRD-03)과 D7 슬라이스의 단일 실행 경로.
**소유.** `buc/pipeline/spine.py`, 테스트 `tests/prd_01/`(동결). `buc/pipeline/__init__.py`는 없어도 namespace로 동작(필요하면 PRD 단위가 만든다).
**의존.** `buc.core.contracts`(FND-02), `buc.core.det`(FND-03), `buc.core.config`·`manifest`(FND-07), numpy, 표준 라이브러리. torch import 금지(스테이지 구현이 알아서). import 부작용 없음.

## 공개 API (`buc.pipeline.spine`)
- `SPINE_STAGES: tuple[str,...]` = `contracts.STAGES` 순서 중 `STAGE_SIGNATURES`에 있는 것 → v0 `("S-1","S2","S8","S8.5","S9-10","S11")`. 하드코딩하지 않고 계산한다(FND-02에 서명이 늘면 자동 확장).
- `class SpineError(RuntimeError)`, `class StageFailed(SpineError)`: 속성 `stage: str`, `impl: str`(Impl.ref), 원인은 `__cause__`.
- `@dataclass(frozen=True) class Impl(stage, kind, name, version, fn)`. `stage ∈ STAGE_SIGNATURES`, `kind ∈ {"stub","real"}`, name·version 비지 않은 str, fn 호출 가능. 위반 `ValueError`. 속성 `ref = f"{kind}:{name}@{version}"`. `fn(inputs: dict[str, 계약객체], ctx: StageContext) -> dict[str, 계약객체]`.
- `@dataclass(frozen=True) class StageContext(stage: str, seed: int, params: dict, impl: Impl)`. `params`는 실효 파라미터(아래, `_seed` 포함) 깊은 복사.
- `class Registry`: `register(impl, *, replace=False) -> Impl`(같은 (stage, kind) 중복은 `ValueError`, replace=True면 교체), `get(stage, kind) -> Impl|None`, `resolve(stage, prefer="real") -> Impl`(prefer 쪽이 있으면 그것, 없으면 다른 쪽, 둘 다 없으면 `SpineError`; prefer∉{"real","stub"} `ValueError`).
- `stage_seed(seed, stage) -> int` = `det.derive_seed(seed, "buc.spine.v1", stage)`.
- `stage_params(stage, seed, params=None) -> dict` = `{**params.get(stage, {}), "_seed": stage_seed(seed, stage)}`. 사용자 params에 `_seed` 키가 있으면 `ValueError`.
- `stage_cache_key(stage, impl, input_hashes, params) -> str` = `StageIO(stage, impl.kind, impl.name, impl.version, inputs=input_hashes, outputs={}, params=params).cache_key` (FND-02 공식 그대로, 64 hex).
- `run_root(stage_ios: Mapping[str, StageIO]) -> str` = `det.merkle_root_keyed({s: det.hash_parts(io.cache_key, contracts.content_hash(io.outputs))}).hex()`. 순서 무관, 빈 입력은 `det.EMPTY_ROOT.hex()`.
- 캐시: `get(key) -> (StageIO, dict[str,계약객체]) | None`, `put(key, stage_io, outputs) -> None`, `key in cache`.
  - `MemoryCache()`: put·get 모두 **사본**(to_dict/from_dict 왕복)을 주고받는다 → 결과 객체를 나중에 바꿔도 캐시 불변.
  - `DiskCache(path)`: 파일 `<path>/<key[:2]>/<key>.json` = `{"stage_io": io.to_dict(), "outputs": {이름: obj.to_dict()}}`. 임시 파일 + `os.replace`(잔여 임시 파일 없음). `get`은 파싱 실패·계약 검증 실패·`io.cache_key != key`·출력 content_hash ≠ `io.outputs[이름]`이면 **예외 없이 None**(손상 = miss, 재계산 후 덮어씀).
- `@dataclass(frozen=True) class PlanNode(stage, impl: Impl, inputs: dict[str, str], outputs: tuple[str,...])`. impl은 prefer로 resolve한 것(없으면 plan이 `SpineError`). inputs 값 = 그 이름을 마지막으로 만든 앞 스테이지, 또는 `"input"`(image).
- `class Spine(registry, *, cache=None, prefer="real", allow_fallback=True)`. cache None → `MemoryCache()`; 속성 `spine.cache`·`spine.registry`로 노출. prefer는 str 또는 {stage: "real"|"stub"}(빠진 스테이지는 "real").
  - `plan(*, skip=(), until=None, prefer=None) -> list[PlanNode]`: SPINE_STAGES 순서, `until` 포함까지, `skip` 제외. 모르는 스테이지 `ValueError`. 어떤 입력이 생산자 없이 남으면 `SpineError`(실행 전).
  - `run(image, *, seed=None, config=None, params=None, skip=(), until=None, force=(), prefer=None) -> RunResult`. config가 있으면 seed=config.seed(seed가 주어졌는데 다르면 `ValueError`), 둘 다 없으면 0.
- `@dataclass class RunResult`: `artifacts: dict[str, 계약객체]`(최종값; S8.5가 world를 덮어씀), `stage_ios: dict[str, StageIO]`(실행 순서), `status: dict[str, "run"|"hit"|"fallback"]`, `root: str`(= run_root(stage_ios)), `seed: int`, `config_hash: str|None`. 속성 `ran`(run+fallback 스테이지 tuple), `hits`. `manifest(name, config, *, git_commit=None, created_utc=None) -> Manifest` = `make_manifest(name, config, inputs={"image": image.content_hash()}, params={"root": root, "impls": {s: io의 "kind:name@version"}}, ...)`.

## 실행 의미 (스테이지마다)
1. impl = resolve(stage, prefer). 입력 = 현재 artifacts에서 서명 이름들. `input_hashes = {k: v.content_hash()}`, `p = stage_params(...)`, `key = stage_cache_key(...)`.
2. stage ∉ force 이고 캐시 hit → 저장된 출력·StageIO 사용, fn 호출 0회, status "hit".
3. 아니면 fn(입력 얕은 사본, ctx) 호출 → `make_stage_io(stage, kind, name, version, inputs, outputs, params=p, wall_s=측정, fallback)`(= check_stage_io). 호출 뒤 입력 content_hash가 바뀌었으면 실패(입력 변이 금지). 성공 시 `cache.put(key, io, outputs)`, status "run".
4. 실패(예외·계약 위반·입력 변이): impl이 real이고 allow_fallback이고 stub이 있으면 stub으로 1–3을 다시(stub 키로 캐시 조회 포함), StageIO.fallback = `f"{real.ref}: {예외 타입명}"`, status "fallback"(stub이 캐시 hit여도). stub도 실패하면 `StageFailed(stage, stub.ref)`. 그 밖은 `StageFailed(stage, impl.ref)` from 원인. 결과의 `StageIO.fallback`은 status가 "fallback"일 때만 non-None(캐시에서 온 io도 그 실행 기준으로 덮어씀; cache_key·root에는 안 들어감). 실패한 키는 캐시에 넣지 않는다.

## 불변식
- 캐시 키는 (계약 버전, 스테이지, 구현 kind·name·version, 입력 내용 해시, 실효 params)만의 함수 → 하류 재실행은 **입력 내용이 바뀐 스테이지만**(조기 차단: 다시 돈 상류가 같은 바이트를 내면 하류는 hit).
- 같은 (image, 구현, seed, params) → 같은 root·같은 artifacts 해시(메모리/디스크 캐시, 새 프로세스 무관). root는 실행 경로(hit/run)와 무관.
- 모든 스테이지 출력은 check_stage_io를 통과한 것만 artifacts·캐시에 들어간다. taint는 이음매에서 줄지 않는다(FND-02).

## 수용 기준
`tests/prd_01/` 전부 통과(CPU, <60 s, 동결 `PRD-01.tests.json`): DAG 계획(생산자 해석·skip/until·고아 입력 거부), 레지스트리 검증·선호 해석, 키 공식 = StageIO.cache_key, root 오라클 대조, 두 번째 실행 전부 hit·호출 0회, params/구현 교체 시 정확한 재실행 집합, 조기 차단, force, 스텁↔실물 교체와 되돌림 시 원래 root, 폴백 기록·비허용 시 StageFailed, 계약 위반·입력 변이 거부와 미캐시, seed 파생·config 연동, DiskCache 재오픈 hit·손상 복구·파일 배치, manifest 안정성. `@pytest.mark.gpu`: CUDA 구현이 들어간 스파인의 root 재현.

## 하류 이음매
PRD-03 스텁은 `Impl(kind="stub")`으로 `Registry`에 등록하고 FND-02 서명만 맞춘다. 실물 단위(RD-21 판독, WLD-01 세계, CAM-02 샷, RDR-02/06 렌더, PRD-05 S11)는 `Impl(kind="real")`로 끼운다 — 스파인 코드는 바꾸지 않는다. PRD-02 CLI는 `Spine.run`/`RunResult.manifest`를 감싼다. EVL-02/EVL-11/GOV-04는 `root`를 비교한다(no-op 변경 → root 불변). 키 공식·root 배치 변경은 호환 파괴이므로 태그(`buc.spine.v1`)와 이 계약서를 함께 올린다.
