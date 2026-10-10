# RUN-01 Vast.ai 오케스트레이터 v0

**목적.** RTX 5090을 **필요할 때만** 빌리고, 놀거나 못 뜨거나 예산에 닿으면 **스스로 끈다**. 돈이 새는 코드이므로 모든 지출을 FND-07 장부에 원자적·멱등으로 남기고, 캡은 넘기기 **전에** 집행한다. (기존 `ops/gpu/ensure_up.py`·`ops/vast_watch.py`의 규칙: 5090, 검증됨, 신뢰도 ≥0.98, ≤$0.50/h, 중국 제외를 이 모듈로 옮긴다.)
**소유.** `buc/rt/vast.py`, 테스트 `tests/run_01/`(동결). 의존: 표준 라이브러리, `buc.core.config`, `buc.core.ledger`. import 시 torch·numpy·requests 금지, 부작용 없음. 네트워크는 `HttpVastAPI` 메서드 호출 때만. 시계는 주입(`clock`).

## 상수·예외
`API_BASE="https://console.vast.ai/api/v0"`, `KEY_ENV="VAST_API_KEY"`, `DEFAULT_KEY_PATH="~/.config/vastai/vast_api_key"`, `DEFAULT_IMAGE="pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"`.
`VastError(RuntimeError)` ⊃ `VastAuthError`, `VastAPIError`(속성 `status:int|None`), `VastNotFound`, `NoOfferError`, `RentLimitError`, `BudgetRefused`(속성 `reason: "daily"|"total"|"locked"`, `check: BudgetCheck|None`).

## API 이음매 (`Protocol VastAPI`, 원시 dict)
`search_offers(query: dict) -> list[dict]` · `create_instance(offer_id: int, config: dict) -> int`(config 키: `image, disk, label, onstart, env`) · `show_instance(iid) -> dict`(없으면 `VastNotFound`) · `list_instances() -> list[dict]` · `set_state(iid, state: "running"|"stopped")` · `destroy_instance(iid)`.
`HttpVastAPI(api_key, *, base_url=API_BASE, timeout=30.0, urlopen=urllib.request.urlopen)`: `urlopen(req, timeout=...)`. 키는 **`Authorization: Bearer` 헤더에만**(URL·repr·예외 메시지에 없음). HTTP 401/403 → `VastAuthError`, 404 → `VastNotFound`, 그 밖 → `VastAPIError(status)`. 응답 `{"offers":[...]}`/`{"instances":...}`/`{"new_contract":id}`를 푼다. `set_state` = `PUT .../instances/<id>/` 본문 `{"state":...}`, `destroy_instance` = `DELETE .../instances/<id>/`. `HttpVastAPI.from_env(env=None, key_path=None)`: 키 없으면 `VastAuthError`.
`resolve_api_key(env=None, key_path=None) -> str|None`: `env[VAST_API_KEY]`(공백 제거, 비면 다음) → 키 파일 → `None`.

## 값 객체
- `country_code(geo) -> str|None`: 마지막 `,` 조각을 strip; 영문 2글자면 대문자, `"china"`(대소문자 무관)면 `"CN"`, 아니면 `None`.
- `Offer`(frozen): `id, machine_id, gpu_name, num_gpus, dph_total, reliability, geolocation, country, cpu_cores, cpu_ram_gb, disk_gb, inet_down, cuda_vers, verified, rentable`. `Offer.from_api(d)`: `reliability2`→없으면 `reliability`, `cpu_cores_effective`→`cpu_cores`, `cpu_ram`(MB)/1024, `disk_space`, `cuda_max_good`→`cuda_vers`, `verified = d.get("verified") is True or d.get("verification")=="verified"`, `rentable` 기본 True, 빠진 수치는 0. `id`·`dph_total`이 없거나 비유한이면 `ValueError`.
- `OfferQuery`(frozen): `gpu_name="RTX 5090", num_gpus=1, max_dph=0.50, min_reliability=0.98, min_cpu_cores=12, min_ram_gb=30, min_disk_gb=100, min_inet_down=300, min_cuda=12.8, verified_only=True, exclude_countries=("CN",), exclude_machines=frozenset()`. `accepts(o)`: 전부 만족(gpu 이름은 대소문자·`_`/공백 무시, `num_gpus` 일치, `0<dph≤max_dph`, rentable, 국가 제외). `to_api_query()`: 최소 `gpu_name:{eq}`, `num_gpus:{eq}`, `dph_total:{lte}`, `rentable:{eq:True}`, `order:[["dph_total","asc"]]`.
- `select_offers(offers, query) -> list[Offer]`: accepts 통과분을 `(dph_total, -reliability, id)` 오름차순. 결정론적.
- `Instance.from_api(d)`(frozen): `id, machine_id, label, actual_status, intended_status, dph_total, num_gpus`. `intended_status` 없으면 `cur_state`, 그것도 없으면 `"running"`(보수적 과금).
- `VastPolicy`(frozen): `idle_timeout_s=1800, boot_timeout_s=3000, idle_action="stop"|"destroy", cap_margin_h=0.25, reserve_hours=1.0, max_instances=1, stopped_usd_per_h=0.02, label_prefix="buc-"`. 음수·비유한·잘못된 action은 `ValueError`.
- `Action(kind: "stop"|"destroy", instance_id, reason: "idle"|"boot_timeout"|"cap_daily"|"cap_total")`: frozen dataclass, 위치 인자 이 순서, 값 비교. `TickReport(ts, actions: list[Action], errors: list[str], metered_usd, daily_usd, total_usd, burn_usd_per_h, locked)`: `metered_usd`=이번 tick 계량 합, `daily_usd`·`total_usd`=행동 후 장부 합(now의 UTC일), `burn_usd_per_h`=아래 cap 판정에 쓴 burn(행동 전), 성공한 행동만 `actions`에.

## `VastOrchestrator(api, ledger, config, *, policy=VastPolicy(), clock=time.time, activity=None, log=None)`
캡은 `config.caps("usd")`, `config.caps("gpu_h")`. 관리 대상 = `label`이 `label_prefix`로 시작하는 인스턴스(그 밖은 건드리지도 과금하지도 않는다).
- **장부 형식.** `run_id=f"vast:{iid}"`. 계량 조각 `[t0,t1)`: `usd` 금액 = `rate·(t1−t0)/3600`, `ts=t0`, `idem_key=f"vast:{iid}:{t0:.3f}:{t1:.3f}"`, `meta`에 `t0,t1,intended,status,rate`. full 조각이면 `gpu_h` = `num_gpus·시간`(같은 키+`":gpu_h"`). 조각은 **UTC 자정에서 나눈다**. 생명주기 이벤트(`rent`/`start`/`stop`/`destroy`)는 `usd` 0.0, `meta.event`, `meta.t1=now`, `meta.intended`, destroy는 `meta.final=True`; rent는 `meta.offer, machine_id, dph, env_keys`. env는 **키 이름만** 기록(값 금지).
- **요율.** 직전(장부 마지막 meta `intended`) 또는 현재 `intended_status`가 `"stopped"`가 아니면 `dph_total`, 둘 다 stopped면 `stopped_usd_per_h`. 계량 시작점 = 그 iid의 장부 최대 `meta.t1`. 길이 0 조각은 기록 안 함.
- `rent(*, query=None, image=DEFAULT_IMAGE, disk_gb=90, label=None, onstart=None, env=None) -> int`: label 기본 `prefix+"gpu"`, prefix로 안 시작하면 `ValueError`. 순서: 잠금이면 `BudgetRefused("locked")` → intended running 관리 인스턴스 수 ≥ `max_instances`면 `RentLimitError` → `search_offers(query.to_api_query())` → `select_offers`(+`bad_machines` 제외) 빈 목록이면 `NoOfferError` → 최저가 `dph·reserve_hours`로 `ledger.check_budget("usd", …, caps)`(gpu_h 캡 있으면 `num_gpus·reserve_hours`도) 불가면 `BudgetRefused(reason)` → `create_instance` **1회** → rent 이벤트. 거절 경로에서는 create 0회.
- `start(iid)`: 같은 잠금·한도·예산 검사(그 인스턴스 `dph·reserve_hours`) → 계량 → `set_state(running)` → start 이벤트. `stop(iid)`: 계량 → `set_state(stopped)` → stop 이벤트. `destroy(iid, reason="manual")`: 계량 → `destroy_instance` → final 이벤트.
- 속성 `bad_machines: set[int]`(부팅 실패 기계; rent에서 제외). `mark_active(iid, ts=None)`(메모리); `activity(iid) -> float|None`(주입, 예: 러너 하트비트).
- `tick(now=None) -> TickReport`: ① `list_instances()`(실패는 전파) ② 관리 인스턴스 계량, 목록에서 사라진 미종결 iid는 직전 요율로 now까지 과금 후 final ③ 인스턴스당 **최대 1개** 행동, 우선순위 boot_timeout > cap > idle:
  - **boot:** `t_start`=장부의 마지막 rent/start `meta.t1`(재시작해도 유지). 그 이후 계량 meta `status`나 현재 관측에서 `actual_status=="running"`을 한 번도 못 봤고 `now−t_start ≥ boot_timeout_s` → destroy, `machine_id`를 `bad_machines`에 추가.
  - **cap:** `burn` = boot 처리 뒤 남은 intended running 관리 인스턴스 `dph` 합. `spent`=장부 usd 합(일: now의 UTC일). 캡 c(일/누적)에 대해 `spent + burn·cap_margin_h > c + 1e-9`면 intended running 전부 stop(`cap_daily`/`cap_total`; 둘 다면 total).
  - **idle:** running을 본 적 있고 intended running인 인스턴스, `last=max(t_start, mark_active, activity)`, `now−last ≥ idle_timeout_s` → `idle_action`.
  행동 실패는 `errors`에 (키 가림) 적고 나머지를 계속한다. `locked` = 누적 usd ≥ 누적 캡 − 1e-9.

## 불변식
캡을 넘기는 임대·재개는 0건 · 같은 시각 재실행·프로세스 재시작은 이중 과금 0 · 지출은 장부에만 쌓이고 순서와 무관 · 키는 어디에도 출력되지 않음 · 미관리 인스턴스 불간섭.

## 수용 기준
`tests/run_01/` 전부 통과(CPU, <60 s, 가짜 API·주입 시계·`:memory:` 장부): 오퍼 매핑·필터·정렬, 임대 거절 경로(무오퍼·일/누적/잠금·한도·label), 계량 금액(허용 1e-9)·멱등·재시작·자정 분할·정지 요율·외부 소멸, 캡 여유 경계·다음 UTC일 재개·잠금, 유휴·부팅 기한 경계, 행동 1개·실패 격리, HTTP 어댑터 헤더·오류 매핑·키 위생, import 순수성. `@pytest.mark.gpu`: 실 API 오퍼 검색만(키 없으면 skip, 임대 금지).

## 하류 이음매
RUN-02 러너: 하트비트를 `activity`로 공급. RUN-05: `rent(onstart=부트스트랩)`. RUN-07: 장부 `run_id LIKE 'vast:%'` 읽기. 크론/오케스트레이터가 ≤15분마다 `tick()`(데드맨은 `ops/gpu/heartbeat.json`이 별도로 담당).
