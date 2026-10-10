# RUN-08 T4 실측 부록 v0 ("Shot on T4": 4K 프레임 시간·메모리)

**목적.** D14에 T4를 몇 시간 임대해 PARADIGM §1.7 항목 6(gsplat/2DGS 4K fwd, 1024² fwd+bwd; 합격선 **4K ≤0.6 s @3M 프리미티브**)을 재고, 시간·메모리 표를 낸다. 핵심 규칙: **T4에서 재지 않은 수치는 T4 수치로 적지 않는다.** 못 재면 "미측정"(타임박스 폴백). 성공을 지어내지 않는다.
**소유.** `ops/t4/shot_on_t4.py`, 테스트 `tests/run_08/`(동결). 표준 라이브러리만 import 시점에 쓴다. torch·numpy·`buc.*`는 함수 안에서 지연 import(import 부작용 0, `sys.path` 불변, CUDA 초기화 금지). `buc`가 필요하면 레포 루트를 그때 `sys.path`에 넣는다. 직접 실행 `python3 ops/t4/shot_on_t4.py` 가능. 의존: RDR-01 `buc.render.splat2d`(`Surfels`, `render`), RUN-01 `buc.rt.vast.OfferQuery`. 임대·과금은 RUN-01 몫이고 이 스크립트는 지출을 기록하지 않는다.

## 상수
`SCHEMA_VERSION=1`, `RES_4K=(3840,2160)`, `GATE_PRIMS=3_000_000`, `GATE_4K_S=0.6`, `GIB=2**30`. 메모리 수치는 모두 GiB(바이트/`GIB`).

## 공개 API
```python
def is_t4_name(name: str) -> bool        # 대소문자 무시, 앞뒤가 영숫자가 아닌 토큰 "T4". "Tesla T4"/"NVIDIA T4 16GB" True, "RTX 5090"/"Quadro T400"/"cpu" False
def redact(text) -> str                   # RUN-07 hourly.redact와 같은 규칙(hf_·sk-·ghp_ 토큰, api_key|token|secret|password 키=값 → 이름=***, 영숫자 32자+ → ***)
def is_oom(exc: BaseException) -> bool    # MRO에 이름이 "OutOfMemoryError"인 클래스가 있거나 MemoryError이거나 str에 "out of memory"(대소문자 무시)
@dataclass(frozen=True, kw_only=True) class DeviceInfo: kind("cuda"|"cpu"), name, total_mem_gib: float|None, capability: tuple[int,int]|None, torch_version: str, cuda_version: str|None, is_t4: bool   # +to_dict/from_dict
def detect_device(torch_mod=None) -> DeviceInfo   # None이면 torch를 지연 import. cuda.is_available() 거짓 → kind "cpu", name "cpu", is_t4 False. 참 → 장치 0의 get_device_name/get_device_properties(0).total_memory/get_device_capability(0), torch_mod.version.cuda. is_t4 = kind=="cuda" and is_t4_name(name)
def make_sync(device: DeviceInfo, torch_mod=None) -> Callable[[], None]   # cuda면 torch_mod.cuda.synchronize(), cpu면 아무것도 안 하는 함수(torch 안 건드림)
@dataclass(frozen=True, kw_only=True) class Timing: n, warmup, samples_s: tuple[float,...], min_s, median_s, mean_s, p95_s, max_s
def summarize_samples(samples, warmup=0) -> Timing   # 빈 목록·음수·비유한·bool → ValueError. mean=math.fsum/n. 분위수(중앙값=0.5, p95=0.95) = 정렬 후 pos=q*(n-1)의 선형 보간
def time_calls(fn, *, repeats=5, warmup=2, sync=None, clock=time.perf_counter) -> Timing
```
`time_calls`: 워밍업마다 `fn(); sync()`. 측정 반복마다 정확히 `sync(); t0=clock(); fn(); sync(); t1=clock()`, 표본 `t1-t0`. `sync=None`이면 호출 안 함. `repeats<1`, `warmup<0` → ValueError. `fn`의 예외는 그대로 전파.
```python
@dataclass(frozen=True, kw_only=True) class MemStats: baseline_gib, peak_allocated_gib, peak_reserved_gib
@dataclass(frozen=True, kw_only=True) class Case: name: str; params: Mapping[str,Any]; build: Callable[[str], Callable[[], Any]]   # build(device_kind)->fn. 준비는 build 안(측정 밖)
STATUSES=("ok","oom","error","unmeasured"); VERDICTS=("pass","fail","unmeasured","info")
@dataclass(frozen=True, kw_only=True) class Measurement: name, params: dict, status, timing: Timing|None, memory: MemStats|None, verdict, reason: str|None   # 잘못된 status/verdict → ValueError; to_dict/from_dict
def gate_case(params) -> bool                 # mode=="fwd" and (width,height)==RES_4K and n_prims==GATE_PRIMS
def verdict_for(params, status, timing, device) -> str
```
`verdict_for`: 게이트 케이스가 아니면 `"info"`. 게이트 케이스인데 `device.is_t4`가 거짓 → `"unmeasured"`(status와 무관). T4에서 status `oom` → `"fail"`, `error`·`unmeasured` → `"unmeasured"`, `ok` → `timing.median_s <= GATE_4K_S + 1e-9`면 `"pass"` 아니면 `"fail"`.

`run_case(case, *, device, torch_mod=None, repeats=3, warmup=1, clock=time.perf_counter, allow_cpu=False) -> Measurement`
1. `device.kind!="cuda"`이고 `not allow_cpu` → status `unmeasured`, reason `"no-cuda"`, **build 호출 0회**.
2. `fn=case.build(device.kind)` 후 워밍업 → `sync()` → (cuda면) `torch.cuda.reset_peak_memory_stats()`, `baseline=memory_allocated()` → `time_calls(fn, repeats=repeats, warmup=0, ...)`(결과 Timing의 `warmup`은 실제 워밍업 수로 바꾼다) → 끝나면 `max_memory_allocated()`, `max_memory_reserved()`로 MemStats. CPU면 memory None. 성공은 status `ok`.
3. build·fn이 `is_oom` 예외 → status `oom`, timing·memory None. 그 밖의 `Exception` → `error`. reason = `f"{타입명}: {redact(메시지)}"` 200자 이하. `KeyboardInterrupt` 등 BaseException은 전파. cuda 장치면 **어떤 결과든** 끝에 `torch.cuda.empty_cache()` 1회.
4. verdict = `verdict_for(...)`.

`run_suite(cases, *, device, torch_mod=None, repeats=3, warmup=1, clock=time.perf_counter, allow_cpu=False, budget_s=None, created_utc=None) -> Report`: 입력 순서대로. 각 케이스 앞에서 ① `budget_s` 있고 `clock()-시작 >= budget_s` → `unmeasured`, reason `"time-budget"` ② 앞서 `oom`이 난 케이스와 같은 (mode,width,height)이고 그 `n_prims <=` 이번 `n_prims` → `unmeasured`, reason `"skipped-after-oom"` ③ 아니면 `run_case`. ①②는 build 호출 0회, verdict는 `verdict_for`. 한 케이스의 오류는 다음 케이스를 막지 않는다.

`Report`(frozen, kw_only): `schema, device, measurements: tuple[Measurement,...], created_utc: str|None, budget_s: float|None`. 속성 `shot_on_t4` = `device.is_t4 and any(m.status=="ok")`. `gate_verdict`: 게이트 케이스가 없으면 `"unmeasured"`; 하나라도 `fail`이면 `"fail"`; 전부 `pass`면 `"pass"`; 그 외 `"unmeasured"`. `to_dict()`(JSON 직렬화 가능, 키 `schema, created_utc, budget_s, device, measurements`)·`Report.from_dict(d)`(schema≠1 → ValueError). 환경변수·호스트명·경로는 어디에도 넣지 않는다.
`write_report(report, path) -> Path`: 임시 파일 + `os.replace`, 부모 생성, 잔여 임시 파일 없음, `json.dumps(indent=2, sort_keys=True, ensure_ascii=False)+"\n"`(같은 입력 → 같은 바이트). `load_report(path) -> Report`(없으면 FileNotFoundError, 깨짐 ValueError).

## 케이스 (RDR-01 래스터)
```python
def make_scene(n_prims, width, height, *, device="cpu", seed=0) -> tuple[Surfels, dict]   # (surfels, {"c2w": (4,4) float64 단위행렬, "K": (3,3) float64})
def default_cases(*, width=3840, height=2160, prims=(500_000,1_000_000,3_000_000), train_size=1024, train_prims=1_000_000) -> list[Case]
def t4_offer_query(max_dph=0.35) -> "OfferQuery"
```
`make_scene`: `torch.Generator().manual_seed(seed)`(CPU에서 생성 후 `.to(device)`, 전역 RNG 불변), float32. `fx=fy=0.5*width/tan(30°)`, `cx=width/2`, `cy=height/2`. 점마다 깊이 z~U[2,8], 화소 (u,v)~U[0,W]×U[0,H]에서 `x=(u-cx)z/fx`, `y=(v-cy)z/fy`. `log_scales=log(1.5·z/fx)`(두 축), `opacity_logits=1.0`, 색 U[0,1], 쿼터니언 정규분포 4원. 같은 인자 → 비트 동일. n_prims<1·width/height<1·정수 아님 → ValueError.
`default_cases`: `prims`(양의 int, 중복 없음)와 `width, height, train_size, train_prims`(양의 int)를 검증해 어기면 ValueError, 통과하면 prims 오름차순으로 `splat_fwd_{W}x{H}_{n}`(params `{"mode":"fwd","width","height","n_prims"}`)들을 먼저, 그다음 `splat_fwdbwd_{S}x{S}_{train_prims}`(mode `"fwdbwd"`, width=height=S). 반환된 fn: `fwd`는 `torch.no_grad()` 아래 `render(...)`의 `SplatRender` 반환. `fwdbwd`는 매 호출 grad를 비우고(`.grad=None`) 파라미터에 `requires_grad`를 켠 뒤 `render(...).rgb.mean()`을 `backward()`하고 손실 `float`을 반환(두 번 호출해도 같은 값).
`t4_offer_query(max_dph)`: `OfferQuery(gpu_name="Tesla T4", num_gpus=1, max_dph=max_dph, min_reliability=0.95, min_cpu_cores=4, min_ram_gb=16, min_disk_gb=60, min_inet_down=100, min_cuda=12.0)`(나머지 기본). `max_dph`가 비유한·≤0 → ValueError.

## 부록 (`render_appendix(report) -> str`, 결정론, 개행 종료, 한국어)
```
# Shot on T4 실측 부록
> 측정 장치: {name} · {total_mem_gib:.1f} GiB · torch {torch_version} · CUDA {cuda_version or "n/a"}     (is_t4일 때)
> 미측정: 이 장치는 T4가 아니다 ({name}). T4 수치는 없다.                                              (아니면; CPU 포함)

| 케이스 | 상태 | 중앙값(s) | p95(s) | 최대 메모리(GiB) | 판정 | 비고 |
|---|---|---|---|---|---|---|
| {name} | {상태} | {median:.3f} | {p95:.3f} | {peak_reserved:.2f} | {판정} | {reason or ""} |

게이트: 4K ≤0.6 s @3M 프리미티브 — {통과|미달|미측정}
```
상태 ok→`측정`, oom→`OOM`, error→`오류`, unmeasured→`미측정`. 판정 pass→`통과`, fail→`미달`, unmeasured→`미측정`, info→`-`. 수치 칸: timing·memory가 None이면 `-`, **T4가 아닌 장치면 값이 있어도 `미측정`**(원시 값은 JSON에만). 비고의 `|`는 `/`, 개행은 공백으로 바꾸고 `redact`한다. 케이스 순서 유지.

## CLI
`main(argv=None, *, torch_mod=None, clock=time.perf_counter, cases=None) -> int`: `--out`(기본 `work/t4/shot_on_t4.json`; 부록은 같은 이름 `.md`) `--repeats 3` `--warmup 1` `--prims 500000,1000000,3000000` `--width 3840` `--height 2160` `--train-size 1024` `--train-prims 1000000` `--allow-cpu` `--budget-min FLOAT`(0보다 큰 유한 실수, 분 → `budget_s=분*60`) `--created-utc STR`(없으면 현재 UTC `%Y-%m-%dT%H:%M:%SZ`). `cases`가 주어지면 `default_cases` 대신 쓴다. 성공: stdout JSON 한 줄(그 한 줄만) `{"ok":true,"json":str(Path(--out)),"md":그 경로의 .with_suffix(".md") 문자열,"shot_on_t4":bool,"gate":str}`, 반환 0 — **CUDA가 없어 전부 미측정이어도 0**(보고서는 "미측정"). 쓰기 실패: `{"ok":false,"why"}`(redact), 반환 1. 잘못된 인자(`--repeats 0`, `--warmup -1`, 숫자 아님, `--width/--height 0`, `default_cases`가 거부하는 `--prims`(0·중복 등), 비양수·비유한 `--budget-min`, 모르는 옵션): 파일을 쓰기 전에 `SystemExit(2)`.

## 불변식
T4 아닌 장치의 수치는 부록에 숫자로 나오지 않음 · 게이트 `pass`는 T4에서 실제로 잰 게이트 케이스가 있을 때만 · OOM은 숨기지 않고 `fail`로 기록 · 한 케이스 실패가 나머지를 막지 않음 · 비밀은 JSON·부록·stdout에 없음 · 같은 입력(시계 주입) → 같은 바이트.

## 수용 기준
`tests/run_08/` 전부 통과(CPU, <60 s, 가짜 torch·주입 시계): 장치 판별, 분위수·시간 측정 호출 순서, 메모리 GiB 환산과 리셋 순서, OOM·오류 분류와 비밀 가림, 래더 스킵·시간 예산·CPU 허용, 판정 경계(0.6 s), 보고서 왕복·원자 쓰기, 부록 정확 일치와 비T4 숫자 은닉, 장면 결정론, 실제 CPU 래스터 소형 실행, 임대 쿼리, import 위생, CLI. `@pytest.mark.gpu`: 실 CUDA에서 소형 케이스 `ok`(T4면 `shot_on_t4`).

## 하류 이음매
D14 운영자는 `ops/t4/shot_on_t4.py`를 임대한 T4에서 돌려 `.json`·`.md`를 회수한다(PRD-05/06 정직 리포트가 `.md` 인용, T4 임대는 `t4_offer_query()`를 RUN-01 `VastOrchestrator.rent(query=...)`에 넘김). 임대 못 하면 CPU에서 돌려도 "미측정" 부록이 나온다. 소유 밖 요청 없음.
