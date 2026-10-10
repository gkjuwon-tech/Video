# FND-05 CI(GitHub Actions) + 수치 원장 재계산 v0

**목적.** 렌즈 간 숫자가 따로 놀지 않게(F-M1, CT-M10) 모든 숫자를 **수치 원장(YAML 한 파일)** 에 두고, CI가 식을 다시 계산해 **1% 넘게 어긋나면 실패**시킨다. 같은 워크플로가 린트·CPU 테스트를 돌린다.
**소유.** `.github/workflows/ci.yml`, `tools/ledger_check.py`, 테스트 `tests/fnd_05/`(동결). 의존: 표준 라이브러리 + `yaml`(PyYAML). `buc`·numpy·torch를 import하지 않는다. import 부작용 없음(파일은 함수 호출 때만 읽는다). 비밀 없음.
**소유 밖 요청.** pyproject `dev` extras에 `pyyaml>=6` 추가(FND 단위). 원장 파일 `docs/number-ledger.yaml` 자체는 이 단위가 만들지 않는다(카탈로그 렌즈 소유, 내용은 하류).

## 원장 형식 (`SCHEMA = "buc.number-ledger/v1"`)
```yaml
schema: buc.number-ledger/v1
tolerance: 0.01          # 선택. 기본 0.01, [0,1)
owners: [pipeline, bench, rights, catalog]   # 선택. 있으면 entry.owner 는 이 안이어야 함
entries:
  - {id: world.tfu.a, value: 179, unit: TFU, owner: pipeline}          # 잎: 식 없음
  - {id: world.tfu.sum, value: 400, unit: TFU, owner: pipeline,
     formula: "world.tfu.a + 2*world.tfu.b", inputs: [world.tfu.a, world.tfu.b]}
```
- `id`: `^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*$`(fullmatch, 하이픈 불가), 원장 안에서 유일. `value`: 유한 int/float(bool·문자열 불가). `unit`·`owner`: 비어 있지 않은 str. 모르는 키는 허용(주석용).
- 잎(`formula` 없음/null): `inputs`는 없거나 `[]`. 파생(`formula` 있음): `inputs`는 **식이 참조하는 id 집합과 정확히 같아야** 한다(순서 무관, 중복 불가).

## 식 문법 (안전한 자체 파서, `eval`·`ast.literal_eval` 금지)
`expr = term {(+|-) term}`, `term = unary {(*|/) unary}`, `unary = (-|+) unary | atom`, `atom = 숫자 | id | 함수'('expr {, expr}')' | '('expr')'`. 숫자 `12`, `1.5`, `.5`, `2e3`, 함수 `sum`·`min`·`max`(인자 ≥1). 공백 자유. 그 밖의 문자·모르는 함수·빈 식·괄호 불일치·0 나눗셈·비유한 결과는 `FormulaError`.

## 공개 API (`tools/ledger_check.py`, 파일 경로로 import 가능해야 함)
- 상수 `SCHEMA`, `DEFAULT_TOL = 0.01`, `DEFAULT_LEDGER = "docs/number-ledger.yaml"`(저장소 루트 기준).
- `class FormulaError(ValueError)`, `class LedgerFormatError(ValueError)`.
- `formula_refs(formula: str) -> tuple[str, ...]`: 참조 id, 첫 등장 순서, 중복 제거(문법 오류 `FormulaError`).
- `eval_formula(formula: str, env: Mapping[str, float]) -> float`: env에 없는 id는 `FormulaError`.
- `@dataclass(frozen=True) Violation(check: str, id: str, why: str)`; `@dataclass(frozen=True) Report(ok: bool, violations: tuple[Violation, ...], computed: dict[str, float], checked: int)`. `checked`=**entries 리스트 길이**(깨진 entry 포함, 리스트가 아니면 0). `computed`: 식 계산에 성공한 파생 entry의 id → 재계산값(불일치여도 포함, 식 오류·미지 입력은 제외).
- `check_ledger(doc: Mapping, *, tol: float | None = None) -> Report`: 데이터 문제로 **예외를 던지지 않는다**. `tol` 인자 > 문서 `tolerance` > 기본값. `tol`이 [0,1) 밖이거나 bool·비수치면 `LedgerFormatError`. `violations`는 `(id, check, why)` 사전순 정렬, `ok == (violations 없음)`.
- `check_file(path, *, tol=None) -> Report`: 파일이 없으면 `FileNotFoundError`; YAML 오류·최상위가 매핑 아님은 `schema` 위반 Report.
- `main(argv=None) -> int`. CLI: `python tools/ledger_check.py [--ledger PATH] [--tol X]`(기본 경로는 cwd가 아니라 저장소 루트 기준). 종료 코드 **0** 위반 없음, **1** 위반(YAML 오류 포함), **2** 사용법 오류·파일 없음·`--tol` 비수치/범위 밖(argparse 종료도 2). stdout **마지막 줄**은 JSON `{"ok": bool, "checked": int, "violations": [{"check","id","why"}...]}`(`ensure_ascii=False`, 앞 줄은 사람이 읽는 설명).

## 검사(`Violation.check` 값)
- `schema`: 문서가 매핑 아님·`schema` 불일치·`entries`가 리스트 아님/비어 있음·entry가 매핑 아님·id/value/unit/owner 형식 위반·`tolerance` 범위 밖(entry 쪽은 해당 id, 모르면 `""`). 한 entry가 깨져도 나머지는 계속 검사한다.
- `duplicate_id`(중복 id마다 1건), `owner`(`owners` 목록 밖), `formula`(문법·0 나눗셈·비유한), `inputs_mismatch`(선언 inputs ≠ 참조 집합, 잎에 inputs가 있음), `unknown_input`(참조 id가 원장에 없음), `cycle`(파생 entry 의존 그래프의 순환에 속한 모든 id에 각 1건), `unit`, `mismatch`.
- `mismatch`: 파생 entry마다 **입력의 선언값**(재귀 재계산 아님, 오류 전파 방지)으로 식을 계산해 `rel = |value - computed| / |computed|`; `rel > tol + 1e-12`이면 위반(정확히 1%는 통과). `computed == 0`이면 `value == 0`일 때만 통과.
- `unit`: 식이 **가산형**(`*`·`/` 없음, 함수는 sum/min/max만)이면 참조한 모든 entry의 `unit`이 자기 `unit`과 같아야 한다(숫자 리터럴은 무시). 곱·나눗셈 식은 검사하지 않는다.

## CI 워크플로 (`.github/workflows/ci.yml`)
`name: CI`; 트리거 `push`(main)와 `pull_request`; 최상위 `permissions: {contents: read}`; `pull_request_target`·`secrets.`·`self-hosted`·`continue-on-error: true`·`|| true` 없음. `python-version`은 **따옴표 친 문자열**(`"3.11"`; `3.10`을 맨몸으로 쓰면 YAML이 float 3.1로 읽는다). `runs-on: ubuntu-latest`, `timeout-minutes` ≤ 30. 단계: `actions/checkout@v4`, `actions/setup-python@v5`(python 3.10 이상, pip 캐시 선택), CPU torch 설치(`--index-url https://download.pytorch.org/whl/cpu`) 후 `pip install -e ".[dev]" pyyaml blake3`, `ruff check --select E9,F63,F7,F82 buc ops tools ci`, `python -m pytest -m "not gpu" tests`, `python tools/ledger_check.py`(단계 `if: hashFiles('docs/number-ledger.yaml') != ''`로 원장이 생기기 전엔 건너뜀). GPU 테스트는 돌리지 않는다.

중복 id는 `duplicate_id` 1건을 내고, 식 계산에는 첫 번째 유효 entry를 쓴다. 의존 그래프 순회는 반복형(깊이 400 사슬·순환에서 재귀 한도에 걸리면 안 됨).

## 불변식
같은 원장 → 같은 Report(순서·실행 횟수 무관) · 데이터 문제는 예외가 아니라 위반 · 위반이 하나라도 있으면 종료 코드 1 · 입력 파일은 수정하지 않는다.

## 수용 기준
`tests/fnd_05/` 전부 통과(CPU, <60 s): 식 파서(우선순위·단항·함수·오류), 1% 경계(1.0%/1.01%), 위반 종류별 탐지와 정상 원장 무위반, 순환·중복·단위·inputs 불일치, 실제 F-M1식 합계(버킷 4,320) 시나리오, CLI 종료 코드와 마지막 줄 JSON, ci.yml 구조 검사(트리거·권한·단계·보호 규칙·경로 존재).

## 하류 이음매
카탈로그 렌즈가 `docs/number-ledger.yaml`을 채우면 CI가 자동으로 켜진다(별도 수정 불필요). 하류 단위는 `check_file(path).ok`로 원장을 프로그램에서 검증할 수 있다. 위반 `check` 문자열과 스키마 문자열은 호환 파괴 시 `SCHEMA`와 이 계약서를 함께 올린다. ci.yml에 단계를 더하려는 단위는 요청으로 올린다.
