# GOV-01 Lineage Ledger CI + vault 구조 v1 — 조상 DAG 해시 · BLOCK_TRAIN · weights=None · 체크포인트 금지목록

**목적.** 하위 모델이 쓴 코드가 방화벽 원칙을 어기지 못하게 하는 기계 검사. 권리 렌즈 v3 §1.5 Lineage CI 규칙 1(학습 아티팩트 조상 `train_ok`), 2(학습 잡 마운트에 vault 금지), 6·8(조상 ext / dev_ok), §1.8 vault 네임스페이스, PARADIGM-v2 §5.1 "`weights=None` 강제 + 체크포인트 해시 금지목록"을 코드로 만든다. 모든 판정은 **닫힌 쪽 실패**: 읽을 수 없는 것·모르는 것은 위반.
**소유.** `buc/gov/lineage.py`(표준 라이브러리 + `buc.gov.prov` + `buc.core.det`; torch는 체크포인트 함수 안에서만 지연 import, import 부작용 없음), `ci/lineage_check.py`(CLI), 테스트 `tests/gov_01/`(동결). 의존: FND-04 `buc.gov.prov`(`as_record`·`derive`·`why_not_train`·`why_not_dev`·`implied_ns`·`FirewallError`·`ProvRecord`), FND-03 `buc.core.det`(`hash_parts`·`merkle_root_keyed`).
**범위 밖.** 실제 암호화·KMS·삭제 잡·카나리아 주입(GOV-02)·사전학습 import 금지(GOV-05)·렌더 경로 린트(GOV-03). 금지목록에 실제 공개 가중치 해시를 채우는 일은 운영 작업(해시를 지어내지 않는다).

## 공개 API (`buc.gov.lineage`)
상수: `LINEAGE_TAG="buc.lineage.v1"`, `DENYLIST_TAG="buc.denylist.v1"`, `SD_TAG=b"buc.sd.v1"`, `NODE_KINDS=("source","data","op","shard","checkpoint","calib","cache","eval")`, `ARTIFACT_KINDS=("shard","checkpoint","calib","cache")`, `VAULT_NAMES=("user_vault","ext_vault","comp_vault","pub_bench","pub_bench_sealed","hold_vault","train_ok")`, `FORBIDDEN_TRAIN_MOUNTS = VAULT_NAMES − {"train_ok"} ∪ {"soul_vault"}`, `STUDENT_DIRS=("buc/models","buc/train")`. 예외 `LineageError(ValueError)`; 방화벽 위반은 `buc.gov.prov.FirewallError`.

**원장.** `@dataclass(frozen=True) Node(id:str, kind:str, prov:ProvRecord, parents:tuple[str,...], content_hash:str|None)` (parents 정렬·중복 제거). id 정규식 `^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$`, content_hash는 소문자 hex 64자 또는 None. 어기면 `LineageError`.
`class LineageLedger()` — 추가 전용, DAG는 구성상 비순환:
- `add(id, *, kind, prov=None, parents=(), content_hash=None) -> Node`. 부모는 이미 있어야 한다(없으면 `LineageError`). prov=None: 부모가 있으면 `derive("lineage:"+kind, [부모 prov…])`, 없으면 `LineageError`(소스는 prov 필수). prov가 주어지고 부모가 있으면 **세탁 금지**: 선언 taint가 `derive(…, human_ref=False)` 또는 `derive(…, human_ref=True)` 결과 taint 중 하나를 포함하고, 선언 ns가 그 결과보다 느슨하지 않아야 한다. 아니면 `LineageError`. 같은 id 재추가: 모든 필드가 같으면 멱등(기존 Node 반환), 다르면 `LineageError`. kind∉NODE_KINDS `LineageError`.
- `node(id)`, `__contains__`, `__len__`, `ids()`(추가 순서), `ancestors(id) -> tuple[str,...]`(자기 제외 전이 폐포, 정렬). 모르는 id `KeyError`.
- `node_hash(id) -> bytes`(32B) = `hash_parts(LINEAGE_TAG, id, kind, prov_json, content_hash or "", len(parents), *[node_hash(p) for p in parents(정렬된 id 순)])`, `prov_json = json.dumps(prov.to_dict(), sort_keys=True, separators=(",",":"))`. 조상 하나라도 바뀌면 후손 해시가 바뀐다.
- `root() -> bytes` = `merkle_root_keyed({id: node_hash(id)})`(추가 순서 무관). 빈 원장도 정의됨.
- `to_json() -> str`: `{"v":LINEAGE_TAG,"nodes":[{"id","kind","prov","parents","content_hash","hash"(hex)}…추가 순서],"root":hex}`(sort_keys). `LineageLedger.from_json(s)`: 버전·부모 순서·각 `hash`·`root` 재계산 일치 검사, 불일치·깨진 JSON `LineageError`.
- `why_not(id, purpose="train", branch="main", denylist=None) -> tuple[str,...]` 정렬: {id} ∪ ancestors의 각 노드 n에 대해 `f"{n}:{r}"` (r ∈ purpose "train"이면 `why_not_train(n.prov, branch)`, "dev"면 `why_not_dev(n.prov)`; dev는 branch "main"만). denylist가 있고 n.content_hash가 걸리면 `f"{n}:DENYLIST:{name}"`. purpose 그 밖 `ValueError`. `assert_ok(id, ...)` 위반 시 `FirewallError`(메시지에 사유 포함). `check_all(purpose, branch, denylist) -> dict[id, reasons]`: ARTIFACT_KINDS 노드 중 위반만.

**금지목록.** `class Denylist(entries: Mapping[str,str]=())` 키 = 소문자 hex 8–64자(접두사 허용), 값 = 이름. `match(h:str) -> str|None`(h가 키로 시작하면 이름, 대소문자 무시). `Denylist.load(path)` / `to_json()`: `{"v":DENYLIST_TAG,"entries":[{"sha256":hex,"name":str}…]}`, 형식 위반 `LineageError`.
`file_sha256(path) -> str`(hex). `state_dict_hash(sd: Mapping[str, Tensor]) -> str` = sha256 hex of `SD_TAG` ‖ 키 정렬 순으로 각 (u32le len‖utf8 key, u32le len‖`str(dtype)`, u32le ndim‖각 dim u64le, u64le nbytes‖`t.detach().cpu().contiguous()` 원시 바이트). 직렬화 형식·dict 순서·requires_grad와 무관.
`check_checkpoint(path, denylist) -> tuple[str,...]`: `FILE:<name>`(파일 해시 적중), `STATE_DICT:<name>`(`torch.load(map_location="cpu", weights_only=True)` 결과의 state dict; 최상위가 `{"state_dict": …}`면 그 안), 읽기 실패·텐서 dict 아님이면 `UNREADABLE`. `assert_from_scratch(module_or_sd, denylist)` 적중 시 `FirewallError`.

**weights=None 린트.** `@dataclass(frozen=True) Violation(path:str, line:int, rule:str, detail:str)`. `scan_source(src, path="<src>") -> list[Violation]`(AST, (line, rule) 정렬). 감시 모듈 = `torchvision`, `timm`, `transformers`, `diffusers`, `open_clip`, `segmentation_models_pytorch`, `torch.hub` 와 그 하위. import 별칭 추적(`import torchvision.models as tvm`, `from torchvision.models import raft_large`, `from torchvision import models`). 규칙:
- `WEIGHTS`: 감시 모듈 호출의 키워드 `weights`·`weights_backbone`이 리터럴 `None`이 아님.
- `PRETRAINED`: 감시 모듈 호출의 키워드 `pretrained`·`pretrained_backbone`이 리터럴 `False`/`None`이 아님.
- `KWARGS`: 감시 모듈 호출에 `**kw`(검증 불가).
- `FROM_PRETRAINED`: 모듈 무관, 어떤 `.from_pretrained(...)`·`from_pretrained(...)` 호출.
- `HUB`: `torch.hub.load`·`load_state_dict_from_url`·`torch.utils.model_zoo.load_url` 호출(별칭 포함).
- `SYNTAX`: 파싱 실패(닫힌 쪽).
`scan_paths(paths) -> list[Violation]`: 파일·디렉터리 재귀 `*.py`, (path, line, rule) 정렬.

**vault.** `vault_relpath(x, *, tenant=None, project=None) -> str`(x는 ProvLike, ns = `as_record(x).ns`): user_vault → `user_vault/<tenant>/<project>/`(둘 다 `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` 필수, 아니면 `ValueError`), ext_vault → `ext_vault/<g>/`(EXTGEN 계열 한정자 g가 하나면 그것, 둘 이상이면 `ext_vault/_mixed/`), comp_vault → `comp_vault/`, pub_bench → PUB-SEALED 있으면 `pub_bench_sealed/` 아니면 `pub_bench/`, train_ok → `train_ok/`. `check_mounts(mounts, purpose="train") -> tuple[str,...]`: 소문자화한 경로에 FORBIDDEN_TRAIN_MOUNTS 이름이 부분 문자열로 든 마운트(입력 문자열 그대로, 정렬·중복 제거). purpose ∈ {"train","dev"}.

## CLI (`ci/lineage_check.py`, `main(argv:list[str]|None=None) -> int`)
옵션: `--ledger P` + `--artifact ID`(반복, 없으면 `check_all`), `--purpose train|dev`, `--branch main|optin-v1`, `--denylist P`, `--ckpt P`(반복), `--src P`(반복), `--mount S`(반복), `--root DIR`(기본 저장소 루트). 아무 검사 옵션도 없으면 `--src`로 `<root>/STUDENT_DIRS` 중 있는 것. stdout 마지막 줄 JSON `{"ok":bool,"violations":[{"check","where","why"}…]}`, (check, where, why) 정렬. 항목: `ancestry`(where=아티팩트 id, why=`why_not` 원소 그대로), `denylist`(where=`--ckpt` 인자 그대로, why=`check_checkpoint` 원소), `pretrained`(where=`f"{path}:{line}"`, path는 `scan_paths`가 낸 그대로, why=규칙명), `mount`(where=마운트 문자열, why=`"FORBIDDEN_MOUNT"`), `ledger`(원장·금지목록 파일을 못 읽음/변조, where=파일 경로, why=오류 메시지). 반환 0 = 위반 없음, 1 = 위반(깨진·변조 원장·금지목록 포함, check "ledger"), 2 = argparse 사용법 오류. `python3 ci/lineage_check.py …`로도 실행(저장소 루트를 sys.path에 넣는다).

## 불변식
1. 아티팩트가 train(dev) 통과 ⇒ 폐포의 모든 노드가 FND-04 `train_ok`(`dev_ok`) ∧ 금지목록 미적중. 선언 prov로 부모 taint를 지울 수 없다.
2. `node_hash`·`root`는 부모 나열 순서·독립 노드 추가 순서와 무관, 조상의 prov·content_hash·id·kind 변화에 민감. 바꾸면 `LINEAGE_TAG`를 올린다.
3. `to_json`→`from_json` 왕복은 같은 root, 한 바이트 변조는 거부.
4. 린트·체크포인트 판정은 실행하지 않고(AST·weights_only) 닫힌 쪽.

## 수용 기준
`tests/gov_01/` 전부 통과(CPU, <60 s): 원장 해시 순서 무관·민감도, JSON 왕복·변조 거부, 세탁 거부, 깊은 DAG의 USER/EXTGEN:test(카나리아 C1·C5) 적발, 시드 고정 무작위 DAG 300건에서 `why_not==()` ⇔ 독립 계산(폐포 모두 train_ok), dev·optin 분기, 금지목록(파일·state dict·접두사) 적중, `state_dict_hash` 불변·민감, 린트 진리표, vault 경로·마운트, CLI 종료 코드와 JSON, 저장소 `STUDENT_DIRS` 기본 검사 통과.

## 하류 이음매
DAT-05 샤드 → `add(kind="shard", content_hash=root_hex)`. MDL/TCH 학습 잡 시작 전 `assert_ok(ckpt_id)`·`check_mounts`·`assert_from_scratch(model)`. GOV-02 카나리아는 이 CLI가 1을 내는지 본다. GOV-05는 import 금지를 따로 맡는다. **요청:** CI 워크플로(.github)에 `python3 ci/lineage_check.py` 추가, 금지목록 파일 위치(`ci/checkpoint_denylist.json`) 생성은 운영 단위가 실제 해시로.
