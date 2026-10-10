# RUN-04 잡 큐·GPU 예산 집행 v0

**목적.** GPU는 1장이다. 학습 잡·교사 호출·야간 A/B가 한 장을 나눠 쓰므로 **우선순위 순서로 내보내되, 장부(FND-07)의 GPU-h·USD 캡을 넘길 잡은 시작시키지 않는다.** 큐는 SQLite 한 파일이며 여러 스레드·프로세스가 같은 파일을 열어도 안전하다. 큐는 장부에 **쓰지 않는다**(지출은 RUN-01/RUN-02가 기록한다). 읽기만 한다.
**소유.** `buc/rt/queue.py`, 테스트 `tests/run_04/`(동결). 의존: 표준 라이브러리(`sqlite3`, `json`, `re`, `threading`), `buc.core.config`(`Config.caps`, `canonical_json`), `buc.core.ledger`(`Ledger`, 읽기 API만). **import 시 torch·numpy·yaml 금지, 파일·DB 부작용 금지.** 시계는 주입. 비밀 금지.

## 상수·예외
`SCHEMA_VERSION=1`(`PRAGMA user_version`; 더 높으면 `QueueError`), `CLASSES=("ab","mrh","train","n")`(앞이 먼저: A/B > MRH > 나머지 학습 > N 클래스), `STATES=("queued","running","done","failed","cancelled")`, `PRIORITY_LIMIT=500_000`.
`QueueError(RuntimeError)` ⊃ `JobNotFound`(`KeyError`도 상속), `LeaseLost`, `QueueBudgetError`(속성 `reason:"daily"|"total"`, `kind`). 인자 검증 실패는 `ValueError`/`TypeError`(bool은 숫자가 아니다).

## `Job` (frozen dataclass, kw_only)
`id, cls, priority:int, est_gpu_h:float, gpu:bool, payload:dict, max_attempts:int, not_before:float|None, state, attempt:int, worker_id:str|None, lease_until:float|None, enqueued_at:float, started_at:float|None, finished_at:float|None, error:str|None, result:dict|None, seq:int`(삽입 순서, 1부터 단조).
속성: `run_id = f"job:{id}"`(RUN-02 장부 `run_id`와 같다), `rank = CLASSES.index(cls)`, `effective_priority = (len(CLASSES)-1-rank)*1_000_000 + priority`(클수록 먼저; RUN-02 `JobSpec.priority`에 그대로 넣으면 같은 순서가 된다). `payload`·`result`는 깊은 복사본.

## `open_queue(path, *, ledger, config, clock=time.time, usd_per_h=0.0, max_running=1) -> JobQueue`
`path`는 str/PathLike 또는 `":memory:"`. 부모 디렉터리 생성, 파일 DB는 WAL·busy timeout ≥10 s. 테이블 `jobs`(위 필드 + `seq INTEGER PRIMARY KEY AUTOINCREMENT`, `id TEXT UNIQUE`; 읽기 전용 SELECT는 하류가 해도 된다). `max_running≥1`(동시에 `running`인 **gpu 잡** 수 상한), `usd_per_h≥0`. 컨텍스트 매니저, `close()` 멱등(**주입된 장부는 닫지 않는다**), 닫힌 뒤 사용은 `QueueError`. 핸들은 스레드 안전.

## `JobQueue` 메서드 (`now=None`이면 `clock()`)
- `enqueue(job_id, *, cls="train", priority=0, est_gpu_h=0.0, gpu=True, payload=None, max_attempts=3, not_before=None) -> Job`. `job_id`는 `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` 전체 일치(끝 개행 불가). `cls`∈CLASSES, `|priority|<PRIORITY_LIMIT`, `max_attempts≥1`, `payload`는 JSON dict. **gpu 잡은 `est_gpu_h>0`(유한), gpu=False는 `est_gpu_h==0`.** 같은 id를 같은 인자로 다시 부르면 기존 행 그대로 반환(멱등, 상태 불변), 인자가 다르면 `QueueError`. 단독으로 절대 시작할 수 없는 잡(필요량 > 일 캡 또는 > 누적 캡)은 쓰지 않고 `QueueBudgetError`. 필요량: gpu_h = `est_gpu_h`, usd = `est_gpu_h*usd_per_h`(`usd_per_h>0`일 때만 검사).
- `claim(worker_id, *, lease_s=900.0, now=None) -> Job|None`. 한 `BEGIN IMMEDIATE` 안에서: ① 만료 리스 정리(`reap`) ② 후보 = `queued`이고 `not_before`가 없거나 ≤ now ③ **디스패치 순서 `(rank, -priority, seq)`** 로 훑어 **처음으로 통과하는 잡**을 `running`으로(attempt+1, `worker_id`, `lease_until=now+lease_s`, `started_at=now`, `error=None`). 통과 조건(gpu 잡만; gpu=False는 슬롯·예산을 쓰지 않는다): 실행 중 gpu 잡 수 < `max_running`, 그리고 종류 K∈{gpu_h, usd}마다 `ledger.check_budget(K, held_K + need_K, daily_cap, total_cap, ts=now).ok`. 못 통과한 잡은 `queued`로 남기고 다음 후보로 넘어간다(예산 백필: 큰 잡이 오늘 안 들어가면 작은 잡이 먼저 나간다). 없으면 `None`.
- **예약(held).** 실행 중(`state=="running"`) gpu 잡 j의 `held_gpu_h = max(0, est_gpu_h - ledger.total("gpu_h", run_id=j.run_id))`, `held_usd = max(0, est_gpu_h*usd_per_h - ledger.total("usd", run_id=j.run_id))`. 즉 러너가 이미 장부에 쓴 몫은 이중 계산하지 않는다. `reserved(kind, *, now=None) -> float`(kind∈gpu_h|usd, 아니면 `ValueError`, `fsum`)가 합을 돌려준다. 캡 경계는 장부와 같다: `spent + held + need <= cap + 1e-9`면 통과.
- `heartbeat(job_id, worker_id, *, lease_s=900.0, now=None) -> Job`: `running`이고 `worker_id`가 같으면 `lease_until=now+lease_s`(만료됐어도 아직 아무도 reap하지 않았으면 가능). 아니면 `LeaseLost`.
- `complete(job_id, worker_id, *, result=None, now=None) -> Job`: `running`+같은 worker → `done`(`finished_at`, `result`, 리스 비움). **같은 worker의 반복 호출은 멱등**(기존 `done` 반환). 그 밖은 `LeaseLost`.
- `fail(job_id, worker_id, error="", *, retry=True, retry_delay_s=0.0, now=None) -> Job`: `retry`이고 `attempt<max_attempts`면 `queued`(`not_before=now+retry_delay_s`, 0이면 None, worker·리스 비움), 아니면 `failed`. `error` 저장. `running`+같은 worker가 아니면 `LeaseLost`.
- `reap(*, now=None) -> list[str]`: `running`이고 `lease_until <= now`인 잡을 정리하고 id 목록(seq 순)을 돌려준다. `attempt<max_attempts`면 `queued`, 아니면 `failed`, 둘 다 `error="lease_expired"`.
- `cancel(job_id, *, now=None) -> Job`: `queued`/`running` → `cancelled`(예약 해제; 그 worker의 이후 `complete`/`heartbeat`는 `LeaseLost`). 종결 상태면 변화 없이 반환. 없는 id는 `JobNotFound`.
- `get(job_id) -> Job`(`JobNotFound`), `list(state=None) -> list[Job]`(`queued`는 디스패치 순서, 그 밖은 seq 순; 모르는 state는 `ValueError`), `counts() -> dict[str,int]`(STATES 전부 키), `close()`.

## 불변식
**캡 보장:** 어느 시점에도 `ledger 합 + reserved ≤ 캡 + 1e-9`가 큐의 claim 때문에 깨지지 않는다(실제 지출이 추정을 넘는 것은 큐 책임 밖) · 같은 잡을 동시에 둘이 claim하지 못한다(8스레드·여러 프로세스) · gpu 잡 동시 실행 ≤ `max_running` · 디스패치 순서는 입력 삽입 순서와 무관하게 `(rank,-priority,seq)` · 종결 상태는 되돌아가지 않는다 · 재시작(재오픈)해도 상태·리스·시도 횟수가 보존 · 큐는 장부에 쓰지 않는다.

## 수용 기준
`tests/run_04/` 전부 통과(CPU, <60 s, 주입 시계·인메모리/임시 파일 DB·실제 장부): 디스패치 순서 오라클(난수 입력), not_before, 예산 경계(일/누적/gpu_h/usd, 허용 오차), UTC일 경계 재개, 백필, held 공식, 슬롯 상한, 리스·재시도·소진, 취소, 멱등 enqueue/complete, 입력 검증, 영속, 8스레드·3프로세스 경쟁에서 정확히 N건만 claim, 난수 시뮬레이션 불변식, import 순수성.

## 하류 이음매
RUN-02 러너는 `Job.payload`에 `JobSpec.to_dict()`를 담아 쓰고 `effective_priority`로 `jobs/*.yaml`의 `priority`를 정한다. 러너는 장부에 `run_id=job.run_id`로 `gpu_h`·`usd`를 기록한 뒤 `complete`/`fail`을 부른다(순서 무관해도 안전: 보수적으로 이중 계산되지 않는다). RUN-01은 같은 장부 캡으로 임대하므로 큐와 캡을 공유한다. RUN-07은 `list()`/`counts()` 또는 `jobs` 테이블 SELECT로 읽는다. 스키마 변경은 `SCHEMA_VERSION`과 이 계약서를 함께 올린다.
