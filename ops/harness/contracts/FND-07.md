# FND-07 설정 · 실험 매니페스트 · 장부(SQLite) v0

**목적.** 모든 실행이 "어떤 설정·코드·시드로 돌았고 TFU·GPU-h·달러를 얼마 썼는가"를 한 곳에 남기고, 예산 캡을 **원자적으로** 집행하게 한다(RUN-01 일/누적 캡, RUN-04 잡 큐, RUN-07 보고, PRD-01 캐시 키가 읽는다).
**소유.** `buc/core/config.py`, `buc/core/manifest.py`, `buc/core/ledger.py`, 테스트 `tests/fnd_07/`(동결). 의존: 표준 라이브러리(`sqlite3`, `json`, `math`), `blake3`. **config·ledger는 numpy/torch를 import하지 않는다.** import 부작용 없음(파일·DB는 함수 호출 때만 만든다). 비밀(키·토큰)은 어디에도 두지 않는다.
**소유 밖 요청.** pyproject 의존성에 `blake3>=1.0`(FND-03과 동일 요청).

## config.py
- `class ConfigError(ValueError)`. `canonical_json(obj) -> bytes`: `sort_keys`, 구분자 `(",",":")`, `ensure_ascii=False`, UTF-8, `allow_nan=False`, tuple→list, 키가 str 아니거나 JSON 밖 타입이면 `TypeError`, NaN/Inf `ValueError`. `1`과 `1.0`, `True`는 서로 다른 바이트(정수/실수 구분 유지).
- `@dataclass(frozen=True, kw_only=True) class Config` 필드와 기본값: `seed:int=0`([0,2^64)), `runs_dir:str="runs"`, `ledger_path:str="runs/ledger.sqlite"`, `budget_daily_usd:float=10.0`, `budget_total_usd:float=35.0`, `budget_daily_gpu_h/budget_total_gpu_h/budget_daily_tfu/budget_total_tfu: float|None=None`(None=무제한), `extra: dict=<빈 dict>`(JSON 값만; 하류 단위의 자유 설정칸; 생성 시 깊은 복사하므로 원본을 나중에 바꿔도 해시 불변).
  숫자 필드는 유한·≥0, int는 float로 정규화(10과 10.0은 같은 해시), bool·그 밖 타입은 `ConfigError`, 문자열은 비어 있으면 안 됨. 위반은 전부 `ConfigError`(생성자 포함).
- `to_dict() -> dict`(깊은 복사: 바꿔도 객체 불변), `Config.from_dict(d)`(모르는 키 `ConfigError`, 빠진 키는 기본값), `replace(**kw) -> Config`(검증 다시, 모르는 키 `ConfigError`), `caps(kind) -> (daily|None, total|None)`(kind∈`usd|gpu_h|tfu`, 아니면 `ValueError`), `hash() -> str` = `blake3(canonical_json(to_dict())).hexdigest()`(64 hex).
- `load_config(path=None, *, env=None, overrides=None) -> Config`. 우선순위 **기본값 < JSON 파일 < 환경변수 `BUC_<FIELD대문자>` < overrides**. `env=None`이면 `os.environ`. 파일의 모르는 키·JSON이 아니거나 객체 아님·overrides의 모르는 키는 `ConfigError`, 경로가 있는데 파일이 없으면 `FileNotFoundError`. 모르는 `BUC_*` 환경변수는 무시. 환경값은 문자열을 `int/float`로 해석(실패 `ConfigError`), None 허용 필드는 `""`/`"none"`(대소문자 무관)이 None. `extra`는 환경변수로 못 바꾼다.

## manifest.py
- `SCHEMA="buc.manifest/v1"`, `class ManifestError(ValueError)`. `@dataclass(frozen=True, kw_only=True) class Manifest`: `schema, name, seed:int, config:dict, config_hash:str, git_commit:str, inputs:dict[str,str], params:dict, created_utc:str`.
  검증: `name`은 `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`(fullmatch: 끝의 개행도 불가); `git_commit`은 40자 소문자 hex 또는 `"unknown"`; `inputs` 값은 64자 소문자 hex(입력 해시); `created_utc`는 실존하는 `YYYY-MM-DDTHH:MM:SSZ`; `config_hash` == `Config.from_dict(config).hash()`; `seed`∈[0,2^64). 위반은 `ManifestError`.
- `config`·`params`는 JSON 직렬화 가능한 dict(아니면 `ManifestError`), `inputs` 키는 비어 있지 않은 str. 생성 시 dict를 깊은 복사한다.
- `manifest_id -> str` = `blake3(canonical_json(to_dict() 에서 created_utc 제외)).hexdigest()`. 시각만 다르면 같은 id, 나머지 어느 필드가 달라도 다른 id, dict 삽입 순서와 무관.
- `to_dict()`(깊은 복사, `manifest_id` 미포함), `Manifest.from_dict(d)`(모르는 키·누락 `ManifestError`).
- `make_manifest(name, config: Config, *, inputs=None, params=None, git_commit=None, created_utc=None) -> Manifest`: `seed=config.seed`, `config=config.to_dict()`, `config_hash=config.hash()`. `git_commit=None`이면 `read_git_commit()`, `created_utc=None`이면 현재 UTC(초 단위).
- `read_git_commit(repo_dir=None) -> str`: `git rev-parse HEAD`(타임아웃 있음), 실패·git 없음·저장소 아님이면 `"unknown"`. 예외를 던지지 않는다.
- `write_manifest(m, path) -> Path`: 임시 파일에 쓴 뒤 `os.replace`(원자적, 부모 디렉터리 생성, 잔여 임시 파일 없음). 파일 = `canonical_json(to_dict()+{"manifest_id": id}) + b"\n"`(같은 매니페스트 → 같은 바이트). `read_manifest(path) -> Manifest`: `manifest_id` 누락·불일치(변조), 모르는 최상위 키, 스키마 주 버전 불일치는 `ManifestError`.

## ledger.py
- 상수 `KINDS=("tfu","gpu_h","usd")`, `TFU_FLOPS=6.2e13`, `TFU_PER_FORWARD={256:0.1, 512:0.25, 768:0.56, 1024:1.0}`(PARADIGM §1.4). 헬퍼 `tfu_for_forward(resolution, n=1) -> float`(표 밖 해상도·음수·bool `ValueError`), `tfu_from_flops(flops) -> float`, `gpu_hours(seconds) -> float`, `gpu_cost_usd(seconds, usd_per_hour) -> float`(음수·비유한 `ValueError`).
- 예외: `LedgerError(Exception)`, `BudgetExceeded(LedgerError)`(속성 `check: BudgetCheck`, `reason`).
- `@dataclass(frozen=True, kw_only=True) class LedgerEvent`: `kind, amount, run_id, stage="", note="", meta:dict={}, idem_key:str|None=None, ts:float|None=None, id:int|None=None`. `amount`는 유한 ≥0 실수(bool 불가, TypeError/ValueError), `kind`∈KINDS, `stage`·`note`는 str, `run_id` 비어 있으면 `ValueError`(관례: `run_id = manifest_id`), `meta`는 JSON 직렬화 가능한 dict.
- `open_ledger(path, *, clock=time.time) -> Ledger`. `path`는 str/PathLike 또는 `":memory:"`. 부모 디렉터리 생성, 파일 DB는 WAL·busy timeout ≥10 s. 스키마 버전은 `PRAGMA user_version=1`; 더 높은 버전이면 `LedgerError`. 컨텍스트 매니저, `close()` 멱등, 닫힌 뒤 사용은 `LedgerError`.
- **저장소(읽기 전용 접근은 하류가 직접 SELECT해도 된다):** 테이블 `events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, amount REAL, run_id TEXT, stage TEXT, note TEXT, meta TEXT(canonical JSON), idem_key TEXT UNIQUE)`. **추가 전용:** `UPDATE`·`DELETE`는 트리거가 `sqlite3.DatabaseError`로 막는다.
- `Ledger.record(ev) -> int`(id ≥1, 단조 증가; `ts=None`이면 `clock()`). `idem_key`가 이미 있으면 새 행 없이 기존 id 반환(kind·amount·run_id·stage가 같을 때), 다르면 `LedgerError`.
- `total(kind, since=None, until=None, run_id=None) -> float`: 구간 **[since, until)** (유닉스 초 또는 tz-aware `datetime`, naive는 `ValueError`), 합은 `math.fsum`이라 **기록 순서와 무관하게 비트 동일**, 없으면 `0.0`. `Ledger.daily_total(kind, ts=None) -> float`: `ts`가 속한 **UTC 달력일**. 모듈 함수 `utc_day_bounds(ts) -> (start, end)`(float 초, [start,end)).
  `events(kind=None, run_id=None, since=None, until=None, limit=None) -> list[LedgerEvent]`(id 오름차순), `totals_by_run(kind) -> dict[str,float]`, `count() -> int`.
- `BudgetCheck`(frozen): `ok, reason("" | "daily" | "total"; 둘 다 넘으면 "daily"), kind, amount, daily_spent, total_spent, daily_remaining, total_remaining`(`*_spent`는 요청 amount를 더하기 **전** 기존 합, 캡 None이면 remaining도 None, 아니면 `max(0, cap-spent)`). `Ledger.check_budget(kind, amount, *, daily_cap=None, total_cap=None, ts=None) -> BudgetCheck`: **`spent + amount <= cap + 1e-9`이면 통과**(읽기 전용; 금액·캡이 음수·비유한·bool·비수치면 `ValueError`/`TypeError`).
- `Ledger.record_within_budget(ev, *, daily_cap=None, total_cap=None) -> int`: 검사와 기록을 **한 `BEGIN IMMEDIATE` 트랜잭션**으로. 초과면 `BudgetExceeded`를 던지고 **아무것도 쓰지 않는다**. 일 합은 `ev.ts`(없으면 `clock()`)가 속한 UTC일 기준. 이미 있는 `idem_key`의 재시도는 예산 검사 없이 기존 id 반환(자기 지출 때문에 재시도가 실패하면 안 된다). 같은 파일을 연 여러 핸들·스레드가 동시에 시도해도 캡을 넘기지 않는다. (처음 만드는 순간의 동시 `open_ledger`는 보장하지 않는다: 한쪽이 먼저 열어 스키마를 만든 뒤 나머지가 연다.)

## 불변식
장부는 추가 전용 · 합계는 기록 순서·재시도와 무관 · 캡 초과 기록은 0건 · 같은 (설정, 코드, 시드, 입력)은 같은 `manifest_id` · 매니페스트 변조는 읽을 때 드러난다 · config 해시는 정수/실수 표기 차이를 흡수한다.

## 수용 기준
`tests/fnd_07/` 전부 통과(CPU, <60 s): 설정 우선순위·검증·해시 안정성, `canonical_json` 바이트 고정과 blake3 오라클 대조, 매니페스트 id 민감도/불변성·원자적 쓰기·변조 탐지, 장부 합계(구간 반열림·UTC일 경계·순서 셔플 비트 동일), 추가 전용 트리거, 멱등 키, 예산 검사 경계, 8스레드 경쟁에서 캡 정확히 준수, 재오픈 영속, config·ledger의 torch/numpy 미로드, TFU 가격표 값.

## 하류 이음매
RUN-01은 `Config.caps("usd")`와 `record_within_budget`/`check_budget`으로 임대 전 캡을 검사한다. RUN-02는 `idem_key`(잡 id+시도)로 재개 시 이중 계상을 막는다. RUN-04는 같은 `ledger_path`를 여러 프로세스가 열어도 안전하다는 점에 기댄다. RUN-07은 `events()`·`totals_by_run()` 또는 `events` 테이블 SELECT로 읽는다. PRD-01은 `manifest_id`와 `Config.hash()`를 캐시 키 재료로 쓴다(`det.hash_parts`에 str로 넣는다). 스키마·필드 변경은 호환 파괴이므로 `user_version`/`SCHEMA`와 이 계약서를 함께 올린다.
