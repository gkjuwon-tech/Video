# BUC 15일 스프린트 작업 단위 전수 목록 (r-inventory)

작성: 작업 분해 담당 (Sonnet 5.5). 근거: `docs/PARADIGM-v2.md` §4·§5·§6·§7·§8·§11, `work/wf3/lens/pipeline.upg.md`·`models.upg.md`, `work/wf3/day15-forecast.json`, `work/wf3/inputs/07-founder-decision-i2v.md`, `ops/STATE.md`.

## 0. 한 장 요약

- 코드로 만들 작업 단위 **166개**, 예상 코드 **약 11.5만 줄**(테스트 포함 아님이 아니라 포함), 이론 작업량 **약 131 에이전트-일**이다. 15일 동안 평균 8.7명이 쉬지 않고 일하는 양이라, 워크플로 하나(동시 2명)로는 못 하고 **워크플로·형제 세션 5개 이상**이 필요하다.
- 소형 모델 14종은 **전원 포함**했다(MDL-01~22, 모델 14종 = 단위 22개). 범위는 그대로고 순서만 S/P/M/N 등급으로 매겼다(§5).
- D6–7 수직 슬라이스의 **최소 집합은 45개**다. 폐포를 코드로 검증했다(슬라이스 단위의 하드 의존이 전부 슬라이스 안에 있다). 이론상 가장 빠른 완료일은 **D6.9**이고, 이 날은 **여유 0일**이다. 즉 D7에 닫히려면 모든 게 맞아야 한다. 안 맞으면 스텁(PRD-03)으로 D7 클립을 먼저 내고 MRH 실물은 D8에 끼운다.
- 임계 경로(슬라이스): `FND-01 → FND-07 → RUN-01 → RUN-05 → TCH-01 → TCH-03 → DAT-02 → MDL-03 → RD-07 → RD-08 → WLD-03 → WLD-08`. 병목은 GPU 요금이 아니라 **교사 기동 → 탭 → Realify → MRH 학습 → 판독 → 깊이 세계**로 이어지는 직렬 사슬이다.
- 구현 등급: Haiku 31개(1.2만 줄), Sonnet 131개(9.6만 줄), **Opus 4개**(0.6만 줄: RDR-03, RD-12, WLD-07, WLD-09). Fable은 구현에 0개, 설계·검토 게이트에 29개 단위.
- 돈보다 시계가 문제다. 구현+검토+설계 토큰 비용은 전부 합쳐 수백 달러대로 추정된다(§9). 막는 것은 동시 2명 제한과 GPU 반복 횟수다.

## 1. 읽는 법과 전제

### 1.1 표의 열

| 열 | 뜻 |
|---|---|
| ID | `서브시스템-번호`. ★는 D6–7 수직 슬라이스 필수 |
| 산출물 | 레포에 생길 파일·모듈. 경로는 **제안**이다(`buc/` 패키지, 코드 0줄 출발). 설계 담당이 FND-01에서 확정한다 |
| 의존 | **하드 의존**만 적었다. 소프트 의존(있으면 좋아지는 것)은 이유 칸에 적었다 |
| 난/위 | 난이도 1–5 / 위험 1–5. 위험은 틀렸을 때 며칠을 날리는지, 이음매를 깨는지를 본 것이다 |
| 화면 | ● 화면 경로에 직접(Day 15 영상 품질에 직접 영향) / ◐ 간접(다른 단위를 통해, 또는 폴백이 있음) / ○ 없음 |
| 검증 | U 단위 테스트로 가능 / G GPU 필요 / V 사람의 시각 판정 필요. 섞이면 `U+G+V` |
| LOC | 예상 코드 규모(천 줄, 테스트 포함). 내 추정이며 ±50% |
| 구현 | H Haiku 5.5 / S Sonnet 5.5 / O Opus 5.5. 뒤의 m·h는 effort(medium·high) |
| 설계→검토 | 화살표 왼쪽이 설계(인터페이스·수락 테스트·함정 명세) 담당, 오른쪽이 완료 검토 담당. S·O·F(Fable 5.1). 모든 단위는 이와 별개로 CI 자동 검사를 받는다 |
| 소요 | 에이전트-일(검토 포함). 모델 학습 단위는 **데이터 첫 샤드가 준비된 뒤**의 학습·평가 시간이다. 코드 작성은 데이터 생성과 병행한다고 본다 |
| D | **이론 최단 완료일**. 의존만 보고 계산한 값이라 인력·GPU 대기를 뺐다. 실제는 1.3–1.5배로 본다 |
| 분류 | S 슬라이스 필수 / P D8–9 합류 / M D10–14 주력 / N 야간·보조(주의력 분류 대상) / G 안전망·운영(D0–3 병행 설치) |

### 1.2 전제 (기준 문서와 다른 점)

1. **컴퓨트는 T4×2가 아니라 Vast.ai RTX 5090이다.** 기준 문서와 예측 패널 수치는 T4 전제다. `ops/STATE.md`(2026-10-10)가 5090을 필요할 때만 켜기로 확정했다. 그래서 BF16 학습이 되고, T4용 우회(FP16 Rescue, INT8·W4A4·TRT)는 **범위에서 빼지 않되 순서를 최후(N)로** 미뤘다(TCH-09, TCH-10, RUN-08). 대신 5090 쪽 위험이 새로 생겼다. sm_120 빌드(gsplat JIT, flash-attention)가 RUN-05, RDR-01의 숨은 지뢰다.
2. **GPU에는 SSH가 막혀** 있어 GPU 서버가 GitHub에서 잡을 당겨 가고 결과를 푸시한다(RUN-02가 유일한 GPU 경로다). 체크포인트는 HF 비공개 저장소(RUN-03).
3. **코드 0줄 출발.** 레포에는 문서와 `ops/`뿐이다. 모든 단위가 새로 만드는 것이다.
4. **사전학습 깊이·포즈 모델은 시스템 경로에서 금지**다(PARADIGM §6.6). 예측 패널이 "이번 15일 창에서 가장 비싼 제약"이라 한 것이 이것이다. 평가 전용 모델은 R-EVAL로 격리해 허용한다(EVL-04, EVL-06, GOV-05). 이 제약을 창업자가 완화할지는 §11 질문 1이다.
5. **소형 모델 14종은 gen0**이다. 아키텍처는 전량, 학습량은 PARADIGM 1세대의 일부(§5의 제안). 예측 패널 기준으로 체크포인트 9–12종이 나오고 화면에 기여하는 것은 4–6종이다. 그래서 순서를 정한다.
6. 엔진 교사는 Z-Image-Turbo(Apache-2.0 [검증]) 유지. 교체 후보 klein 4B는 [2차] 정보라 라이선스 원문 확인 전까지 N 클래스에 둔다(TCH-07, TCH-08).
7. 입력 24장 중 Soul·GPT 등 외부 생성기 출력은 `EXTGEN` 태그다. 내부 비교·데모 전용이며 학습·보정·회귀셋에 못 들어간다(FND-04, GOV-01). Seedance 비교 클립은 `COMP` 태그, 평가 전용이다(EVL-03).
8. **범위 밖**은 §10에 이유와 함께 적었다(Phase 2, Hero FX, DIEL, 8K·웹뷰어, Soul 레인 등).

## 2. 구현·설계·검토 등급 배정 규칙

창업자 원칙은 "구현은 하위(Sonnet 5.5, Haiku 5.5), 설계와 검사는 상위(Opus 5.5, Fable 5.1)"다. 이를 규칙 8개로 풀었다.

| # | 규칙 | 근거 |
|---|---|---|
| R1 | 기본 구현자는 **Sonnet 5.5, effort medium**. 수학이 빡빡하거나 CUDA·계약이 미묘하면 high | 공식 권고가 에이전트 코딩은 medium부터. Sonnet이 $2/$10으로 Opus의 절반 |
| R2 | **Haiku 5.5는 완결 스펙 단위만**: 입력·출력이 테스트로 정해졌고, 새 알고리즘이 없고, 프롬프트가 100K 토큰 미만 | 100K 초과 프롬프트는 $0.50/$2.50로 5배 오른다. 스키마·파서·스크립트·테스트·집계가 해당 |
| R3 | **Opus 구현은 난이도 5 · 위험 5 · 수식 경계조건이 많은 단위만** (RDR-03, RD-12, WLD-07, WLD-09). Sonnet 타임박스 1일 후 에스컬레이션도 허용 | Sonnet이 세 번 틀리는 비용이 Opus 한 번보다 비싸다 |
| R4 | **설계는 기본 Opus.** 설계 산출물은 인터페이스 시그니처, 수락 테스트, 알려진 함정이다. 스펙이 이미 문서에 있는 단순 단위는 Sonnet이 쓰고 Opus가 훑는다 | 설계가 틀리면 구현을 하위 모델이 해도 헛수고다 |
| R5 | **Fable 5.1은 게이트에서만**: 계약 동결, 소유권·비회귀 수학, 방화벽, 독창 핵심(WLD-07, WLD-09), 선별 편향 검토. 구현에는 쓰지 않는다 | $10/$50에 한 턴이 수 분. 강제 tool_choice 불가라 자동 루프에 박기 어렵다 |
| R6 | **검토는 계단식**: CI(테스트·린트) → Haiku 1차 리뷰 → 표의 검토 등급(Sonnet 교차 또는 Opus) → Fable 게이트 | 비싼 눈은 싼 눈이 걸러 낸 뒤에 쓴다 |
| R7 | **단위 하나의 구현 루프는 한 모델로 끝낸다.** 모델은 검토 단계에서만 바꾼다 | 프롬프트 캐시가 모델별 네임스페이스라 중간 교체는 캐시를 버린다 |
| R8 | **Advisor 패턴**: 난이도 4 이상 단위에서 Sonnet 실행자가 막히면 Opus 조언자를 부른다. 그래도 안 되면 R3으로 에스컬레이션 | 공식 도구가 있다. 전면 Opus보다 싸다 |

공식 비용 원칙("캐스케이드를 짓기 전에 가장 강한 모델을 낮은 effort로 돌려 보라")은 **D0 파일럿**으로 이행한다. FND-02와 RD-02 두 단위를 (a) Sonnet-medium 구현 + Opus 검토, (b) Opus-low 구현으로 같은 스펙에 돌려 **완료된 작업당 비용과 시간**을 잰다. 결과로 R3의 승격 기준을 조정한다. 파일럿 비용은 몇 달러다.

## 3. 작업 단위 목록 (166개)

서브시스템 14개로 나눴다. 각 표의 `D`는 이론 최단 완료일(의존만 보고 계산, 인력·GPU 대기 제외)이다. 분류 S·P·M·N·G의 뜻은 §1.1, 슬라이스 전개는 §7에 있다.


### 3.1 FND 기반·계약

모든 레인이 기대는 계약, 결정론, 계보, CI. 여기가 틀리면 이음매가 틀린다(예측 패널: 진짜 적은 이음매). 단위 8개, 코드 4.6k줄, 3.2 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **FND-01★** | 레포 골격·툴체인 락·GPU 이미지 | `pyproject.toml, uv.lock, docker/gpu.Dockerfile, Makefile, buc/__init__.py` | - | 2/3 | ○ | U+G | 0.4k | S·m | S→O | 0.25 | 0.2 | S | 5090(sm_120) 빌드 핀이 함정이나 정형 작업 |
| **FND-02★** | 데이터 계약 v0 (World·ReadPacket·Obs·Shot·Theta·KPI·StageIO) | `buc/core/contracts.py, docs/contracts.md` | FND-01 | 3/5 | ◐ | U | 0.9k | S·h | O→F | 0.75 | 1.0 | S | 모든 이음매의 원천. 틀리면 전 레인 재작업이라 동결 검토만 Fable |
| **FND-03★** | 결정론 유틸 (Philox WNF·int64 고정소수점 segment-reduce·blake3 Merkle) | `buc/core/det.py` | FND-01 | 3/3 | ◐ | U+G | 0.5k | S·m | O→O | 0.4 | 0.7 | S | 수식이 명세에 완결돼 있고 테스트로 판정됨 |
| **FND-04★** | 계보 태그·prov 비트 (taint·ext/comp/hold·BLOCK_TRAIN/DEV) | `buc/gov/prov.py` | FND-01 | 3/4 | ○ | U | 0.4k | S·m | O→F | 0.4 | 0.7 | S | 방화벽의 어휘. 규칙이 미묘해 동결 검토는 Fable |
| **FND-05** | CI(GitHub Actions)+수치 원장 재계산 | `.github/workflows/ci.yml, tools/ledger_check.py` | FND-01 | 1/1 | ○ | U | 0.3k | H·m | S→O | 0.25 | 0.5 | G | 정형 yaml과 합계 검사 |
| **FND-06★** | 기하·색·필터 수학 유틸 (K·포즈·투영, sRGB↔선형, 피라미드, 리샘플) | `buc/core/geom.py, color.py, pyramid.py` | FND-01 | 2/2 | ◐ | U | 1k | S·m | S→O | 0.5 | 0.8 | S | 전 단위가 쓰는 공용 수학. 수치 테스트로 판정 |
| **FND-07★** | 설정·실험 매니페스트·장부 (SQLite: TFU·GPU-h·비용) | `buc/core/config.py, manifest.py, ledger.py` | FND-01 | 1/2 | ○ | U | 0.6k | H·m | S→O | 0.25 | 0.5 | S | CRUD 수준, 스키마만 정하면 됨 |
| **FND-08** | 골든 픽스처·기하 정답 합성 장면 10종 | `tests/fixtures/*, tools/make_fixtures.py` | FND-02, FND-06 | 2/2 | ○ | U | 0.5k | H·m | S→O | 0.4 | 1.4 | G | 정답을 아는 소형 장면. 단위 테스트의 땅 |

### 3.2 RUN 러너·GPU 운영

Vast.ai 임대, pull 러너, 체크포인트, 학습 하네스, 시간별 보고. 단위 8개, 코드 4.4k줄, 4.0 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RUN-01★** | Vast.ai 오케스트레이터 (오퍼 검색·임대·유휴 자동 종료·일/누적 예산 캡) | `buc/rt/vast.py` | FND-01, FND-07 | 3/4 | ○ | U+G | 0.6k | S·m | O→O | 0.5 | 1.0 | S | 돈이 새는 코드. 모킹 테스트+캡 로직은 Opus가 검토 |
| **RUN-02★** | pull 러너 (GitHub 잡 당김→실행→지표·산출 푸시, 락·멱등·재개) | `buc/rt/runner.py, jobs/*.yaml` | FND-01, FND-07, RUN-01 | 4/4 | ○ | U+G | 0.8k | S·h | O→O | 0.75 | 1.8 | S | SSH 불가라 유일한 GPU 경로. 멱등·재개가 핵심 |
| **RUN-03★** | 체크포인트·HF Hub 비공개 저장소 동기 | `buc/rt/hf_sync.py` | FND-01 | 2/3 | ○ | U+G | 0.4k | S·m | S→O | 0.4 | 0.7 | S | 정형 I/O |
| **RUN-04** | 잡 큐·GPU 예산 집행 | `buc/rt/queue.py` | FND-07 | 2/2 | ○ | U | 0.5k | S·m | S→O | 0.4 | 0.9 | G | SQLite, 명세 완결 |
| **RUN-05★** | GPU 서버 부트스트랩 (가중치 prefetch·gsplat 캐시 빌드) | `ops/gpu/bootstrap.sh` | RUN-01 | 3/4 | ○ | G | 0.3k | S·m | S→O | 0.4 | 1.4 | S | Blackwell JIT 빌드 시행착오 |
| **RUN-06★** | 학습 하네스 공용 (AMP·원자 체크포인트·Feistel PRP·max-abs 경보) | `buc/train/loop.py, amp.py, ckpt.py, sampler.py` | FND-03, RUN-05, RUN-03 | 3/3 | ○ | U+G | 1k | S·h | O→O | 0.75 | 2.1 | S | 14종 전체의 기반. 한 번 틀리면 14번 틀림 |
| **RUN-07** | 시간별 보고 생성기 (시각화 PNG+한국어 요약) | `ops/report/hourly.py` | FND-07 | 1/1 | ○ | V | 0.5k | H·m | S→S | 0.4 | 0.9 | G | 템플릿 작업. 창업자 보고 형식 |
| **RUN-08** | T4 실측 부록 ("Shot on T4": 4K 프레임 시간·메모리) | `ops/t4/shot_on_t4.py` | RUN-01, PRD-01, RDR-01 | 2/2 | ○ | G | 0.3k | H·m | S→O | 0.4 | 3.0 | N | D14. T4 몇 시간 임대 측정 스크립트 |

### 3.3 TCH 엔진 교사 (Z-Image-Turbo)

교사 기동, σ 스냅, 탭, Exhaust. T4 부록(TCH-09·10)은 범위 유지·순서 최후. 단위 11개, 코드 6.5k줄, 8.9 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TCH-01★** | 교사 로더 (Z-Image-Turbo BF16·TE encode-only 캐시·VAE·결정론 모드) | `buc/teacher/zimage.py` | RUN-05, FND-03 | 3/3 | ◐ | G | 0.7k | S·h | O→O | 0.75 | 2.1 | S | 공식 모델 래핑. 메모리·결정론 시행착오 |
| **TCH-02★** | σ 스냅 스케줄·SDEdit·x̂0 API (뷰 배치 lockstep) | `buc/teacher/sample.py` | TCH-01 | 3/3 | ◐ | G | 0.5k | S·h | O→O | 0.5 | 2.6 | S | 모든 교사 호출의 단일 입구. 시그니처는 Opus가 확정 |
| **TCH-03★** | Feature-Tap Bus (층 8/15/23 훅·고정 PCA-128 int8·링버퍼·ext 바이패스) | `buc/teacher/taps.py` | TCH-01, FND-04 | 3/3 | ◐ | U+G | 0.6k | S·h | O→O | 0.75 | 2.9 | S | MRH 학습 입력의 원천. prov 검사 포함 |
| **TCH-04** | Exhaust Ledger (blake3 키·int8 x̂0 저장·ext/hold 분리·세션 재개) | `buc/teacher/exhaust.py` | TCH-02, FND-03, FND-04 | 3/3 | ○ | U | 0.5k | S·m | O→O | 0.4 | 3.0 | M | 키 설계가 방화벽과 닿아 있음 |
| **TCH-05** | 결정 곡선 X1-slim (σ별 옥타브 분리)+가난선 프로브 (E2) | `buc/teacher/decision_curve.py` | TCH-02, FND-06 | 3/3 | ◐ | G+V | 0.5k | S·m | O→F | 0.75 | 3.4 | M | 결과 해석이 σ 채택을 정함. 해석은 Fable |
| **TCH-06** | Teacher-DCA·DR 판정 | `buc/teacher/dca.py` | TCH-02 | 2/3 | ○ | G | 0.3k | S·m | O→O | 0.4 | 3.0 | M | 수식 명세 완결 |
| **TCH-07** | 교사 선정 시험 (TCT-v3 NVF·ES_v3, Z 현직 vs klein 도전자) | `buc/teacher/tct.py` | TCH-02, TCH-08, EVL-04 | 3/3 | ○ | G+V | 0.7k | S·m | F→F | 1.5 | 4.9 | N | 교사 잠금 판정은 최상위. D15 경로는 Z 유지 |
| **TCH-08** | 참조 교사 슬롯 (klein 4B 로더·Reference Ring/Ring-R) | `buc/teacher/ref_slot.py` | TCH-01 | 4/4 | ○ | G+V | 0.8k | S·h | O→O | 1.25 | 3.4 | N | P1-확장(L2b). 그림체 입력용, 15일 경로 밖 |
| **TCH-09** | FP16 Rescue (T4 부록: 잔차 1/s·SwiGLU 재균형·FP32 섬) | `buc/teacher/t4_fp16.py` | TCH-01 | 4/4 | ○ | G | 0.6k | S·h | O→O | 0.75 | 2.9 | N | T4 전용. 5090 BF16에서는 불필요, 범위 유지·순서 최후 |
| **TCH-10** | 양자화 (W4A4·INT8·TRT, 자체 보정셋+prov 검사) | `buc/teacher/quant.py` | TCH-09, FND-04 | 5/5 | ○ | G | 1k | S·h | O→O | 1.5 | 4.4 | N | T4 전용 부록. 어려우나 화면 경로 밖 |
| **TCH-11** | 입구 생성 Entry-Gen (Z-Image T2I 8 NFE로 AI풍 dev 입력 생산) | `buc/teacher/entry_gen.py` | TCH-02 | 1/1 | ○ | G | 0.3k | H·m | S→O | 0.4 | 3.0 | M | dev·학습 입력 공급용 단순 루프 |

### 3.4 RDR 렌더러

슬라이스는 평면 쿼드·LDI(RDR-02)로 닫고, 2DGS(RDR-01)와 TSQ(RDR-03)는 뒤에서 올라탄다. 단위 9개, 코드 7.0k줄, 8.2 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RDR-01** | gsplat 2DGS 래퍼 (미분 래스터·dual-cut LOD·COLMAP 점 초기화 피팅) | `buc/render/splat2d.py, fit.py` | RUN-05, FND-06 | 4/4 | ● | G+V | 0.9k | S·h | O→O | 1.25 | 2.6 | M | primary 래스터. sm_120 JIT 위험 |
| **RDR-02★** | 평면 쿼드·LDI 래스터 (빠른 경로: 알파·소프트 에지) | `buc/render/quads.py` | FND-02, FND-06 | 3/3 | ● | G+V | 0.8k | S·m | O→O | 0.75 | 1.8 | S | 슬라이스용 렌더. 단순하고 빠르게 돌아야 함 |
| **RDR-03** | TSQ 텍스처 쿼드 커널 (8×8 페이지 바이리니어·n≥2 에지·미분·BC 압축) | `buc/render/tsq_kernel.cu, tsq.py` | RDR-01 | 5/5 | ◐ | G+V | 1.8k | O·h | O→F | 2 | 4.7 | M | CUDA 커널. Sonnet 1일 타임박스 후 실패 시 Opus. 폴백은 2DGS |
| **RDR-04** | EWA 비등방 필터·C_0 정합 렌더 필터 (카디널 B-스플라인 사전필터) | `buc/render/filters.py` | RDR-02 | 3/3 | ● | U+G | 0.5k | S·m | O→O | 0.5 | 2.2 | P | 첫 프레임 충실도(FFD 바닥)를 정함 |
| **RDR-05★** | AOV (depth·normal·flow·원자 id)·관측 bitset·안정 정렬 | `buc/render/aov.py` | RDR-02, FND-03 | 3/3 | ◐ | U+G | 0.5k | S·m | S→O | 0.5 | 2.2 | S | IR·halo 지표·마스크의 원천 |
| **RDR-06★** | 영상 인코더·타임라인 (ffmpeg·NVENC·24fps·1080p/4K) | `buc/render/video.py` | FND-02 | 1/1 | ● | U | 0.3k | H·m | S→S | 0.25 | 1.2 | S | 정형 래퍼 |
| **RDR-07** | 평면 재귀 반사·secondary (Warp BVH 폴백) | `buc/render/mirror.py` | RDR-02 | 4/4 | ● | G+V | 0.8k | S·h | O→O | 1.25 | 3.0 | N | 거울 입력은 메뉴에서 안전 등급으로 내림(예측 0.15) |
| **RDR-08** | STRAND/SGGX 체적 (털·잎·솜털) | `buc/render/vol.py` | RDR-01 | 4/4 | ◐ | G+V | 1k | S·h | O→O | 1.25 | 3.9 | N | 매팅 α를 싣는 그릇. 15일 경로에서는 α 카드로 대체 |
| **RDR-09** | Proof-of-Shot·왕복 비트 동일 검증 | `buc/render/pos.py` | RDR-05, FND-03 | 3/3 | ◐ | U+G | 0.4k | S·m | O→O | 0.4 | 2.6 | M | 결정론 증거. 헤드라인 주장 하나를 받침 |

### 3.5 LNS 렌즈·센서·필름 스택

순수 수식 영역. 에이전트가 가장 잘하고 일반 시청자가 가장 먼저 느끼는 장점(레버 8). 단위 8개, 코드 5.3k줄, 5.9 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **LNS-01★** | θ 스키마·사전분포 샘플러·수제 프리셋 20 | `buc/lens/theta.py, presets/*.yaml` | FND-02 | 2/2 | ● | U | 0.5k | S·m | O→O | 0.4 | 1.4 | S | 물리 범위는 Opus가 확정 |
| **LNS-02★** | DOF·Defocus Pyramid·포커스 브리딩·랙 스케줄 | `buc/lens/dof.py` | RDR-02 | 4/3 | ● | G+V | 0.8k | S·h | O→O | 1 | 2.8 | S | 랙 포커스가 데모의 얼굴. 칼선 방지가 관건 |
| **LNS-03★** | 셔터 (Longuet-Higgins split·롤링·180°) | `buc/lens/shutter.py` | RDR-02 | 4/3 | ● | U+G+V | 0.6k | S·h | O→O | 0.75 | 2.5 | S | 호모그래피 적분 수식. 단위 테스트로 검증 |

(표 계속)

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **LNS-04★** | 센서·필름 (Poisson-Gauss·ADC·Newson 그레인·65³ LUT·할레이션·비네팅) | `buc/lens/sensor_film.py` | FND-06, LNS-01 | 3/2 | ● | U+G+V | 0.9k | S·m | O→O | 0.75 | 2.1 | S | 순수 수식. 가장 싸고 가장 먼저 눈에 띔 |
| **LNS-05** | 렌즈 수차 (왜곡·CA·cat-eye·Hullin 고스트·veiling glare·회절 스타) | `buc/lens/optics.py` | LNS-02 | 4/3 | ◐ | G+V | 1k | S·h | O→O | 1 | 3.8 | M | 근사식으로 충분 |
| **LNS-06** | 애너모픽·빈티지 처방 (Kolb 근사 출사동 LUT) | `buc/lens/prescriptions.py` | LNS-05 | 4/4 | ◐ | G+V | 0.8k | S·h | O→O | 1.25 | 5.0 | N | 예측 L1–L2. 근사식 대체가 기본 |
| **LNS-07** | CPL 편광 필터 | `buc/lens/cpl.py` | LNS-04 | 1/1 | ◐ | U | 0.2k | H·m | S→S | 0.25 | 2.4 | N | 한 줄 수식 |
| **LNS-08** | θ 사슬 합성기+ExplainAway 역사슬 왕복 검증 | `buc/lens/chain.py, tests/test_roundtrip.py` | LNS-01, LNS-04 | 3/4 | ● | U+G | 0.5k | S·h | O→F | 0.5 | 2.6 | P | look∘explain≈id 성질이 관측 소유권의 토대 |

### 3.6 CAM 카메라

리그·무브·DP-0·얼굴 봉투. 24장 A/B는 무브가 고정이라 DP-0는 주력이 아니다. 단위 7개, 코드 4.8k줄, 5.3 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CAM-01★** | 리그 1 kHz ODE (14종: 질량·감쇠·홀로노믹·진자·PID, min-jerk) | `buc/camera/rigs.py` | FND-06 | 3/3 | ● | U+V | 0.9k | S·m | O→O | 1 | 1.8 | S | CPU 수식. 슬라이더·달리·오빗 3종부터 |
| **CAM-02★** | 무브 레시피·촬영 메뉴 봉투 Π_g (푸시인·슬라이더·±20° 오빗·랙) | `buc/camera/moves.py` | CAM-01 | 3/3 | ● | U+V | 0.6k | S·m | O→O | 0.5 | 2.2 | S | 벤치 경로 정의식의 원천 |
| **CAM-03** | DP-0 v0 탐색 (후보 1000→프리뷰 50, PR·Wipe·FL·FiF·IR 캡) | `buc/camera/dp0.py` | CAM-02, CAM-04, CAM-07, EVL-05 | 4/3 | ◐ | U+G+V | 1.2k | S·h | O→F | 1.5 | 4.2 | M | 24장 A/B는 고정 경로라 우회 가능. 보상 설계는 Fable |
| **CAM-04** | 얼굴 봉투 규칙 (F0–F3·α_σ·미관측 얼굴면 5%) | `buc/camera/face_envelope.py` | FND-02 | 2/3 | ● | U | 0.4k | S·m | O→O | 0.4 | 1.4 | P | 인물 입력의 하드 제약 |
| **CAM-05** | Gaffer CEM (조명 ≤8, ΔFFD≤0.01 제약) | `buc/camera/gaffer.py` | WLD-05, RDR-02 | 3/3 | ◐ | G+V | 0.6k | S·m | O→O | 0.75 | 7.7 | M | CEM 루프, 제약식 명세 완결 |
| **CAM-06** | IMU 파이프 (GPMF·폰 IMU 파서·정렬·증강 ×5.5) | `buc/camera/imu.py` | FND-06 | 3/3 | ○ | U | 0.7k | S·m | S→O | 0.75 | 1.5 | M | 창업자 촬영 데이터 도착 후 가동 |
| **CAM-07** | 충돌·클리어런스·가시성 유틸 | `buc/camera/clearance.py` | FND-02 | 2/2 | ◐ | U | 0.4k | S·m | S→O | 0.4 | 1.4 | M | 기하 질의 |

### 3.7 RD 입력 판독 (R0–R13, InverseLook, 에셔, 경계 분해)

입력은 관측이다. 설명할 수 있는 것은 세계에 넣지 않는다. 단위 21개, 코드 16.7k줄, 18.2 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RD-01★** | R0 수납 (색관리·ICC·EXIF/C2PA·DQT·레터박스·r_eff) | `buc/read/ingest.py` | FND-02, FND-06 | 2/2 | ● | U | 0.6k | S·m | S→O | 0.5 | 1.5 | S | 정형 입출력과 통계 추정 |
| **RD-02★** | R1 기하 카메라 (LSD→J-Linkage VP→K 공동 LM·plumb-line·수평선) | `buc/read/geom_cam.py` | RD-01 | 4/4 | ● | U+V | 1.1k | S·h | O→O | 1 | 2.5 | S | 고전 기하. AI 이미지 불성립 사례가 함정 |
| **RD-03** | R2 화면 고정 효과 추정기 (비네팅·그레인 PSD·CA·할레이션·고스트) | `buc/read/screen_fx.py` | RD-01, LNS-04 | 4/4 | ● | U+V | 1k | S·h | O→O | 1 | 3.1 | P | 비네팅·그레인 문신의 해독제(레버 4) |
| **RD-04** | ExplainAway 연산자 (역사슬·대역 마스크·Wiener 대역 제한) | `buc/read/explain_away.py` | RD-03, LNS-08 | 3/4 | ● | U+V | 0.6k | S·h | O→F | 0.5 | 3.6 | P | "설명할 수 있는 것은 세계에 넣지 않는다"의 구현 |
| **RD-05** | R3 광학 블러 추정 (c(z) 회귀→z_f·N_eq·모션블러 방향장) | `buc/read/blur_est.py` | RD-07, RD-08 | 4/3 | ● | U+G | 0.7k | S·m | O→O | 0.75 | 5.9 | P | 이중 보케 방지의 전제 |
| **RD-06** | R4 톤·색 (4파라미터 필름 곡선·WB·CDL·룩 잔차 LUT 17³ TTO) | `buc/read/tone_color.py` | RD-01, LNS-08 | 4/3 | ● | U+V | 0.9k | S·h | O→O | 1 | 3.6 | P | 명제 L2(선형 오귀속 상쇄)에 의존 |
| **RD-07★** | R5 독심 판독 호출 (전역 σ 2점+1024² 타일·혼합·r_dis·클린룸 레인) | `buc/read/mrh_call.py` | TCH-02, TCH-03, MDL-03, RD-01 | 3/3 | ● | G | 0.5k | S·m | O→O | 0.5 | 4.9 | S | MRH 연결부. 타일 경계 혼합만 주의 |
| **RD-08★** | R6 미터 스케일 (크기 사전 로그 가중 LS+FOV+EXIF) | `buc/read/metric_scale.py` | RD-02, RD-07, DAT-16 | 2/3 | ● | U | 0.3k | S·m | S→O | 0.25 | 5.2 | S | DOF·레벨이 미터를 요구 |
| **RD-09** | R7 내재 분해 (Retinex+기하 음영) | `buc/read/intrinsics.py` | RD-07, RD-06 | 4/3 | ◐ | U+G | 0.7k | S·h | O→O | 0.75 | 5.7 | M | 재조명 용도. 첫 프레임은 L_d를 직접 소유 |
| **RD-10** | R8 FitLights+치트 라이트 (NNLS·CMA-ES·그림자 에지·CL1–CL6) | `buc/read/fit_lights.py` | RD-09, RD-08 | 4/4 | ● | U+G+V | 1.2k | S·h | O→O | 1.5 | 7.2 | M | 예측 L1. 기본은 복사휘도 굽기로 폴백 |
| **RD-11** | Esc 7성분+r_dis+r_refl (에셔 점수) | `buc/read/escher_score.py` | RD-02, RD-07, RD-08, RD-10 | 3/3 | ◐ | U | 0.8k | S·m | O→O | 0.75 | 7.9 | M | 파라미터 0 잔차 합. 메뉴 등급의 근거 |
| **RD-12** | 기하 화해 솔버 (IRLS×λ 6점·평면 스냅·강체 가중 ARAP 워프·FFG) | `buc/read/reconcile.py` | RD-11 | 5/5 | ● | U+G+V | 1.5k | O·m | O→F | 2 | 9.9 | M | 수식·경계조건 과다. Sonnet 반복 비용이 Opus 1회보다 큼 |
| **RD-13** | 변형 브랜치 D_g (케이지 ARAP-3D·CoW 재오픈) | `buc/read/deform_branch.py` | RD-12, WLD-03 | 5/5 | ◐ | U+G+V | 0.9k | S·h | O→F | 1.25 | 11.2 | N | 등급 전환 재사용용. 15일 경로 밖 |
| **RD-14** | R10 관측 대역 지도 (l_obs 경/부분·w(ν)·O bitset) | `buc/read/obs_map.py` | RD-01, RD-08 | 3/4 | ● | U | 0.5k | S·m | O→O | 0.5 | 5.7 | P | 소유권 수학의 핵심. MRH.obs는 소프트 의존 |
| **RD-15** | R11a 얼굴 맵 (템플릿 검출+MRH sem OR·머리 모형 LM·σ_z) | `buc/read/faces.py` | RD-07, RD-08, MDL-03 | 4/4 | ● | U+V | 1k | S·h | O→O | 1.25 | 6.4 | M | 실제 얼굴 재현율 미측정(F-M8) |
| **RD-16** | R11b 글자 런·SYMBOL 재구성 (OFL 폰트 매칭·MSDF) | `buc/read/glyph_runs.py, render/symbol_msdf.py` | RD-01, MDL-20 | 4/3 | ● | U+V | 0.9k | S·h | O→O | 1 | 4.2 | P | 한글 메뉴판 베스트 클립의 핵심 |
| **RD-17** | R12 반사 판독 (M_refl·평면 RANSAC·C_0^R·프레넬) | `buc/read/reflection.py` | RD-07, RD-02 | 4/4 | ◐ | U+V | 0.7k | S·h | O→O | 1 | 5.9 | N | 거울 입력은 안전 등급 강등 |
| **RD-18** | R13 경계 분해 (깊이 불연속 trimap·Levin 닫힌형 매팅) | `buc/read/edge_unmix.py` | RD-07, RD-01 | 4/4 | ● | U+V | 0.8k | S·h | O→O | 0.75 | 5.7 | P | 예측 아티팩트 1순위(고무 시트)의 해독제. 슬라이스는 LDI 찢기로 버티고 D8에 합류 |
| **RD-19** | PCI 레벨 판정·문법 커버리지 분류 | `buc/read/pci.py` | RD-07 | 2/2 | ◐ | U | 0.3k | H·m | S→O | 0.25 | 5.2 | M | 규칙 기반 분류 |
| **RD-20** | θ_in 전역 최적화 (InverseLook TTO·r_look·자기일관 검사) | `buc/read/inverse_look.py` | RD-02, RD-03, RD-04, RD-05, RD-06 | 4/4 | ● | U+G+V | 1k | S·h | O→F | 1 | 6.9 | M | 최종 룩 일치. 설계 검토는 Fable |
| **RD-21★** | 판독 오케스트레이터·패킷 조립 (R0–R13 순서·캐시·단계별 폴백) | `buc/read/pipeline.py` | RD-01, RD-02, RD-07, RD-08 | 3/4 | ● | U | 0.7k | S·m | O→O | 0.75 | 5.9 | S | 실패 격리와 폴백값이 설계의 전부 |

### 3.8 WLD 세계 솔버 (WSSC)

가장 새롭고 가장 위험한 구역. 출하 경로는 WLD-08(뷰별 SDEdit+평균 역투영)이고 나머지는 이것을 이기러 온다. 단위 21개, 코드 17.7k줄, 21.5 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WLD-01★** | 세계 컨테이너 STRATA-lite-min (blocks·charts·SoA·meta/prov·Merkle·직렬화) | `buc/world/strata.py` | FND-02, FND-03 | 4/4 | ◐ | U | 1.2k | S·h | O→F | 1 | 2.0 | S | v0은 LDI 층+far-field+meta 부분집합만 |
| **WLD-02** | 입력 역투영 TABP-입력 (경/부분 소유·카디널 사전필터) | `buc/world/input_bp.py` | WLD-01, RD-14, RDR-04 | 4/4 | ● | U+G+V | 0.9k | S·h | O→O | 1 | 6.7 | P | "프레임 0 = 입력"의 구현 |
| **WLD-03★** | 깊이 세계 빌더 (MRH 깊이→불연속 찢기→LDI→2DGS 초기화; α 매팅은 RD-18이 나중에 합류) | `buc/world/depth_world.py` | WLD-01, RD-07, RD-08, RDR-02 | 4/5 | ● | G+V | 1k | S·h | O→O | 1 | 6.2 | S | 고무 시트 방지 핵심. 시각 판정 필요 |
| **WLD-04** | xatlas 세계축 차트·Fold 그래프·Fold 뷰 | `buc/world/charts.py` | WLD-01 | 4/4 | ◐ | U+V | 0.8k | S·m | O→O | 0.75 | 2.8 | M | OTBI의 전제 |
| **WLD-05** | L0 조명 bake (Warp/OptiX 경로추적·동기+치트) | `buc/world/bake.py` | WLD-03 | 3/3 | ◐ | G+V | 0.8k | S·m | O→O | 0.75 | 6.9 | M | 발명 영역 조명 일치용 |
| **WLD-06** | 입력 견본 Swatch Bank (정면 패치 정류·탈조명·퀼팅 synth·세계 로컬) | `buc/world/swatch.py` | WLD-02 | 3/3 | ● | U+V | 0.6k | S·m | O→O | 0.75 | 7.4 | M | Texel Bank은 엔진 세계 전용, 입력 견본은 전역 금지 |
| **WLD-07** | 의미 패스 lockstep (48 cine 뷰 동시 디노이즈+TSQ 피팅+관측항+gauge) | `buc/world/semantic_pass.py` | WLD-03, WLD-05, WLD-04, TCH-02, RDR-01, MDL-03 | 5/5 | ● | G+V | 1.6k | O·h | F→F | 2.5 | 9.4 | M | 가장 새롭고 위험. 예측 0.35. 설계·검토 Fable, 구현 Opus |
| **WLD-08★** | 뷰별 SDEdit+평균 역투영 폴백 (실제 화면에 나가는 발명 채움) | `buc/world/sdedit_avg.py` | WLD-03, TCH-02, RDR-02 | 3/3 | ● | G+V | 0.6k | S·h | O→O | 0.75 | 6.9 | S | 예측상 실제 출하 경로. 먼저 만들고 나머지가 이것을 이긴다 |

(표 계속)

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WLD-09** | Grid-Locked OTBI+TABP 닫힌형 sync+Context Ring | `buc/world/otbi_tabp.py` | WLD-04, WLD-07, RDR-03, MDL-02 | 5/5 | ◐ | G+V | 1.5k | O·h | F→F | 2.5 | 11.9 | M | 독창 핵심, 예측 0.35. 타임박스 D12, 실패 시 WLD-08 유지 |
| **WLD-10** | graph cut+multiband commit (maxflow·Laplacian 저대역 교체·VAE 1회) | `buc/world/commit.py` | WLD-04, WLD-08 | 3/3 | ● | U+G+V | 0.8k | S·m | O→O | 0.75 | 7.7 | P | 이음매 제거의 기계 |
| **WLD-11** | 이음매 지우기 (입력 gauge 저대역 고정·apron 40%·Ref Ring 띠·발명 영역 θ_in) | `buc/world/seam.py` | WLD-02, WLD-08 | 4/4 | ● | G+V | 0.7k | S·h | O→F | 0.75 | 7.7 | P | 확장·풀 +5–8%p 레버. 진짜 적은 이음매 |
| **WLD-12** | Router+Spec-commit+가난선 o_s | `buc/world/router.py` | WLD-09, MDL-08, MDL-10, TCH-06 | 5/5 | ○ | G | 0.8k | S·h | O→F | 1.25 | 13.2 | N | 예측 0.10. 주의력 분류로 밤 잡 처리 |
| **WLD-13** | 관할 이양 래칫 루프 | `buc/world/ratchet.py` | WLD-12, MDL-08 | 4/4 | ○ | G | 0.5k | S·m | O→F | 0.75 | 13.9 | N | 예측 0.10. 세대 한 번 돌 시간 없음 |
| **WLD-14** | InputPolish (S8.5 첫 프레임 게이트: FFD>τ→CoW 1 Jacobi) | `buc/world/input_polish.py` | WLD-02, RD-14, WLD-08, MDL-11, EVL-04 | 4/3 | ● | G+V | 0.6k | S·h | O→O | 0.75 | 7.7 | P | FFD 게이트 |
| **WLD-15** | CoW·Merkle 델타 솔브·세션 재개 | `buc/world/cow.py` | WLD-01, FND-03, TCH-04 | 4/4 | ◐ | U+G | 0.8k | S·h | O→F | 1 | 4.0 | M | 재현성과 InputPolish·등급 전환의 토대 |
| **WLD-16** | 감사 S7 (적대 카메라 CMA-ES)+레이아웃 거부권 S2d | `buc/world/audit.py` | WLD-08, TCH-06 | 3/3 | ○ | G | 0.6k | S·m | O→O | 0.75 | 7.7 | N | 품질 보증. 화면 경로 밖 |
| **WLD-17★** | far-field (팔면체 HDR·Hosek–Wilkie 하늘·지평선·지면 연장) | `buc/world/farfield.py` | WLD-01, RD-02 | 3/3 | ● | U+V | 0.6k | S·m | O→O | 0.5 | 3.0 | S | 뒤통수 하늘 이음매 방지. 해석식 |
| **WLD-18** | 중첩 lot 3겹 결합 (스케일 DAG·E_couple·best-of-16×16) | `buc/world/lots.py` | MDL-13, MDL-14, WLD-17 | 3/4 | ◐ | U+G | 0.6k | S·m | O→O | 1 | 6.5 | M | 실외 피사체 확장용 |
| **WLD-19** | 턴어라운드 링 (3패널 동시 디노이즈) | `buc/world/turnaround_ring.py` | WLD-07, TCH-02, INV-08 | 4/4 | ◐ | G+V | 0.7k | S·h | O→F | 1 | 10.4 | M | 뒷면 정체성. 비회귀 조항 위반 유혹 지점 |
| **WLD-20** | 레이어 카드 세계 (2.5D)+NPR 렌더+스트로크 밉 (L2) | `buc/world/layer_cards.py, npr.py` | WLD-03, RD-19 | 3/4 | ● | U+G+V | 1.3k | S·m | O→O | 1.25 | 7.4 | N | 그림체 입력 D15 기본값. 예측상 의외로 볼 만함 |
| **WLD-21** | Demand (Ray-Bundle+Corman–Shannon·IR 맵·입력 소유 B_I) | `buc/world/demand.py` | WLD-01, CAM-02, LNS-01 | 3/3 | ◐ | U | 0.7k | S·m | O→O | 0.75 | 3.0 | M | OTBI·TexelFM의 수요 입력 |

### 3.9 INV 프레임 밖 발명

문법, 부분 lift, 텍스트 소유권, 뒷면. 라벨 공장이기도 하다. 단위 9개, 코드 7.5k줄, 7.0 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **INV-01★** | 문법 코어+RoomGen v0 (장면 프로그램·numba 레이캐스트·G1–G13 에너지 골격) | `buc/invent/grammar/core.py, room.py` | FND-02 | 3/3 | ◐ | U | 1.4k | S·m | O→O | 1 | 2.0 | S | MRH 합성 데이터의 원천. v0 RoomGen을 먼저 |
| **INV-02** | 장르 문법 (한옥·을지로·바·거실·거리) | `buc/invent/grammar/genres.py` | INV-01 | 3/3 | ◐ | U+V | 1.2k | S·m | O→O | 1 | 3.0 | M | 규칙 코드. 도메인 지식은 Opus 설계 |
| **INV-03** | OOG 문법 6종 (스튜디오·사구·해안·계곡·숲 바닥·광장) | `buc/invent/grammar/oog.py` | INV-01 | 3/3 | ○ | U+V | 1.4k | S·m | O→O | 1.5 | 3.5 | N | 문법 밖 장면 예측 0.15. 범위 유지·순서 후순위 |
| **INV-04** | 부분 점유 lift·카빙·Render-Hide-Complete 샘플러 | `buc/invent/lift.py` | INV-01, FND-06 | 3/3 | ◐ | U | 0.8k | S·m | O→O | 0.75 | 2.8 | M | MassingFM 학습 라벨 공장 |
| **INV-05** | 사용자 텍스트 파서 (규칙 한·영·슬롯 사전 800·재질 사전 150) | `buc/invent/parse_text.py` | FND-02 | 3/3 | ○ | U | 0.7k | S·m | S→O | 0.75 | 1.8 | N | 24장 A/B는 텍스트 없이 가능. 사전 초안은 Haiku |
| **INV-06** | CaptionComposer+ReadCaption (TE 사전 최근접) | `buc/invent/caption.py` | INV-01, TCH-01 | 2/2 | ◐ | U | 0.4k | H·m | S→O | 0.4 | 2.5 | M | 템플릿 조합 |
| **INV-07** | best-of-N 선택 에너지 (E_gram·E_light·E_text·E_cont·E_obs·E_face) | `buc/invent/select.py` | INV-01, INV-04 | 3/3 | ◐ | U | 0.5k | S·m | O→O | 0.5 | 3.2 | M | 문법 에너지 합 |
| **INV-08** | 대칭 완성+CC0 에셋 정렬 ICP (뒷면 초기값) | `buc/invent/backside.py` | WLD-01, INV-09 | 3/3 | ◐ | U+V | 0.7k | S·m | O→O | 0.75 | 2.8 | M | 초기값일 뿐 소유권 없음 |
| **INV-09** | CC0 에셋 수집·색인 (Poly Haven·ambientCG, 라이선스 manifest) | `tools/assets/fetch_cc0.py` | FND-01 | 1/2 | ○ | U | 0.4k | H·m | S→O | 0.4 | 0.7 | M | 스크립트. 네트워크 허용 확인 필요 |

### 3.10 MDL 소형 모델 14종 (M1–M14)

모델 14종 = 단위 22개. M3 MRH 4개, M5 TexelFM 3개, M7 MassingFM 2개, M14 Glyph-S 2개로 쪼갰다. 대응표는 §5. 단위 22개, 코드 14.6k줄, 22.6 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MDL-01** | M1 TAE-T (2.5M f8 AE, 교사 VAE 쌍 증류, T-LPIPS 항) | `buc/models/tae_t/model.py, train.py, eval.py` | TCH-01, DAT-05, RUN-06 | 2/2 | ◐ | G+V | 0.6k | S·m | O→O | 1 | 3.1 | P | TAESD 선례. 곡면 경로·프리뷰용 |
| **MDL-02** | M2 LatentDown/Up-S (각 1M) | `buc/models/latent_ud/*.py` | MDL-01, RUN-06 | 1/1 | ○ | G+U | 0.3k | H·m | S→O | 0.4 | 3.5 | M | 6층 conv. 가장 작은 모델 |
| **MDL-03★** | M3 MRH 본체 (PCA 탭→DPT neck→깊이·법선·알베도·rough/metal·shading·defocus·sem·σ̂) | `buc/models/mrh/neck.py, heads.py, train.py` | RUN-06, DAT-02, DAT-05, TCH-03 | 4/4 | ● | G | 1.5k | S·h | O→F | 1 | 4.4 | S | D15 화면의 심장(기여 0.60). 헤드 정의 검토는 Fable |
| **MDL-04** | M3 MRH 전역 헤드 28+NPR 6+비물리 플래그 3 | `buc/models/mrh/glob_heads.py` | MDL-03, DAT-03 | 3/3 | ● | G | 0.5k | S·m | O→O | 0.75 | 5.2 | P | InverseLook 초기값. 고전 추정이 폴백 |
| **MDL-05** | M3 MRH.obs 관측 대역 헤드+정밀화 (H/2–H) | `buc/models/mrh/obs_head.py` | MDL-03, DAT-10 | 4/3 | ● | G+V | 0.6k | S·m | O→O | 1 | 5.4 | P | 흐림과 매끈함 구분. 얼굴 경계에도 사용 |
| **MDL-06** | M3 MRH 실사 MVS 차선·홀드아웃 평가 (GT 비율 ≥55% 규칙) | `buc/models/mrh/real_lane.py` | MDL-03, DAT-08, DAT-04 | 3/4 | ● | G+U | 0.6k | S·m | O→O | 1 | 5.4 | P | 합성만으로는 일반화가 나쁨. 실사 차선 필수 |
| **MDL-07** | M4 G-Adapter (25M, zero-init 5곳, 동결 교사 역전파; E3가 생략 판정) | `buc/models/g_adapter/*.py` | MDL-03, DAT-01, TCH-02 | 4/4 | ◐ | G | 0.8k | S·h | O→F | 2 | 6.4 | N | 조건부. SDEdit만으로 충분하면 생략하나 코드는 만든다 |
| **MDL-08** | M5 TexelFM-35 본체 (s2d·conv-res·window attn·견본 cross-attn·2×2 평균0 사영) | `buc/models/texelfm/model.py, train.py` | RUN-06, DAT-06, DAT-15 | 4/4 | ◐ | G+V | 1.4k | S·h | O→F | 2 | 5.5 | M | 푸시인 끝 구간 품질. 평균0 사영이 효도 정확성의 열쇠 |
| **MDL-09** | M5 TexelFM 입력 부모 모드·견본 로더 (열화 부모 30%) | `buc/models/texelfm/data.py` | MDL-08, DAT-10 | 3/3 | ◐ | U+G | 0.5k | S·m | O→O | 0.75 | 6.2 | M | 푸시인 역설 대응 |
| **MDL-10** | M5 TexelFM native 추론 (WNF 고정 타일 격자·T8 비트 동일) | `buc/models/texelfm/native.py` | MDL-08, FND-03 | 3/3 | ◐ | U+G | 0.5k | S·m | O→O | 0.75 | 6.2 | M | 결정론 검사로 판정 |
| **MDL-11** | M6 T-LPIPS-S (4M, 교사 특징 거리 라벨, θ 불변 헤드) | `buc/models/t_lpips/*.py` | RUN-06, DAT-10, TCH-02 | 3/3 | ◐ | G | 0.5k | S·m | O→O | 1 | 3.9 | P | 내부 FFD 자. 공개 FFD는 평가 전용 LPIPS |
| **MDL-12** | M7 MassingFM 본체 (DiT 12L d384·FM+masked·3D patchify·3상태 조건) | `buc/models/massing/model.py, train.py` | RUN-06, INV-04, INV-01 | 4/4 | ◐ | G | 1.3k | S·h | O→F | 2 | 4.8 | M | 프레임 밖 레이아웃. 기여 0.45 |
| **MDL-13** | M7 MassingFM 샘플러 (관측 고정 집합 inpainting·best-of-N 연결) | `buc/models/massing/sample.py` | MDL-12, INV-07 | 3/3 | ◐ | U+G | 0.5k | S·m | O→O | 0.75 | 5.5 | M | 집합 공간 inpainting. 비회귀 조항 안 |
| **MDL-14** | M8 BRP (42M 3D U-Net) | `buc/models/brp/*.py` | RUN-06, INV-03, INV-04 | 3/3 | ○ | G | 0.8k | S·m | O→O | 1.5 | 5.0 | N | 실외 지형용. 예측상 화면 기여 0.20 |
| **MDL-15** | M9 IPM-S (8M ConvNeXt식) | `buc/models/ipm/*.py` | RUN-06, INV-01 | 2/2 | ○ | G | 0.5k | H·m | S→S | 0.75 | 2.9 | N | 작고 명세 완결. 기여 0.20 |
| **MDL-16** | M10 CamFM-Intent (20M DiT 8L) | `buc/models/camfm/intent.py` | RUN-06, DAT-13 | 3/3 | ○ | G | 0.9k | S·m | O→O | 1.5 | 4.0 | N | 벤치 경로 고정이라 덜 중요(0.20) |
| **MDL-17** | M11 CamFM-Hand (8M causal xfmr·μ-law 256-bin) | `buc/models/camfm/hand.py` | RUN-06, CAM-06 | 3/3 | ● | G+V | 0.6k | S·m | O→O | 1 | 3.1 | M | 손떨림이 "진짜 카메라" 느낌을 만듦(0.55). IMU 데이터 의존 |
| **MDL-18** | M12 CamFM-CRM (12M, 해석 통계) | `buc/models/camfm/crm.py` | RUN-06, DAT-14 | 3/3 | ○ | G | 0.6k | S·m | O→O | 1 | 3.9 | N | 선호 쌍 5k가 15일에 안 모임(0.10–0.15) |
| **MDL-19** | M13 Rig-ID-S (3M 1D conv) | `buc/models/rig_id/*.py` | RUN-06, CAM-01 | 1/1 | ○ | G | 0.3k | H·m | S→S | 0.4 | 2.5 | N | 돌아가지만 클립 기여 약 0.05 |
| **MDL-20** | M14 Glyph-S 검출기 (4M U-Net-lite) | `buc/models/glyph/detector.py` | RUN-06, DAT-12 | 3/3 | ● | G+V | 0.6k | S·m | O→O | 1 | 3.1 | P | 한글 SYMBOL 교체의 입구 |
| **MDL-21** | 모델 레지스트리·export 추론 API | `buc/models/registry.py` | FND-02 | 1/1 | ◐ | U | 0.3k | H·m | S→S | 0.25 | 1.2 | M | 체크포인트 해시·weights=None 검사 연동 |
| **MDL-22** | M14 Glyph-S CTC 인식기 (2.8M) | `buc/models/glyph/ctc.py` | MDL-20 | 3/3 | ○ | G | 0.4k | S·m | O→O | 0.75 | 3.9 | N | CTC·SYMBOL 교체 기여 0.25–0.30 |

### 3.11 DAT 데이터 주조소

기하는 코드에서, 외형은 교사에서, 라벨은 구성에서. 실사 쪽은 창업자 촬영에 달렸다. 단위 16개, 코드 10.0k줄, 9.9 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DAT-01★** | 절차 장면 GT 렌더 파이프 (G-buffer 12ch·디노이저 OFF) | `buc/data/synth_render.py` | INV-01, RDR-02, RDR-05 | 3/3 | ● | G | 0.7k | S·m | O→O | 0.5 | 2.8 | S | MRH 학습 GT의 원천 |
| **DAT-02★** | Realify+탭 동시 추출 패스 (SDEdit 2–3 NFE·결정 곡선 대역 마스크) | `buc/data/realify.py` | DAT-01, TCH-02, TCH-03 | 3/4 | ● | G | 0.6k | S·m | O→O | 0.5 | 3.4 | S | 한 번의 교사 호출로 라벨과 학습 특징을 동시에 |
| **DAT-03** | θ-Realify (θ 적용본+ExplainAway 통과본) | `buc/data/theta_realify.py` | DAT-02, LNS-01, LNS-08 | 3/3 | ● | G | 0.4k | S·m | O→O | 0.4 | 3.8 | P | MRH가 추론 때 보는 분포에 맞춤 |
| **DAT-04** | 실사 노이즈 탭 패스 (RAW+MVS GT→VAE enc→1 NFE) | `buc/data/real_tap.py` | TCH-03, DAT-08 | 3/3 | ● | G | 0.4k | S·m | O→O | 0.5 | 3.4 | P | 생성 없음, 탭만 추출 |
| **DAT-05★** | 샤드 I/O (WebDataset·Merkle 주소·prov 검사) | `buc/data/shards.py` | FND-03, FND-04 | 2/3 | ○ | U | 0.7k | S·m | O→O | 0.5 | 1.1 | S | 계보 검사가 여기서 걸림 |
| **DAT-06** | CC0 PBR 옥타브 피라미드 | `buc/data/pbr_pyramid.py` | FND-01, INV-09 | 2/2 | ○ | U | 0.5k | H·m | S→O | 0.5 | 1.1 | M | 다운샘플 스크립트 |
| **DAT-07** | RAW 매크로 처리 (고전 디모자이크·무작위 현상·크롭 샤드) | `buc/data/raw_macro.py` | DAT-05, FND-06 | 3/3 | ○ | U | 0.7k | S·m | S→O | 0.75 | 1.9 | M | 창업자 촬영 의존. 예측상 계획의 1–10%만 모임 |
| **DAT-08** | 실사 MVS 파이프 (COLMAP SIFT·GLOMAP·PatchMatch·ArUco 스케일) | `buc/data/mvs.py` | FND-01, DAT-05 | 4/4 | ● | G | 0.8k | S·h | O→O | 1.25 | 2.4 | P | MRH 실사 차선의 유일한 씨앗. 빌드·옵션 시행착오 |
| **DAT-09** | I2V 학습쌍 생성기 (Degrade∘Escher∘Style∘Look∘Render+라벨) | `buc/data/i2v_pairs.py` | DAT-01, DAT-10, DAT-11, LNS-08, TCH-11 | 3/3 | ◐ | U+G | 0.9k | S·m | O→O | 0.75 | 4.2 | M | 라벨이 구성상 정확하다는 것이 해자 |
| **DAT-10** | 열화·AI 아티팩트 라이브러리 (JPEG·리샘플 체인·샤프닝·VAE 왕복·AI 업스케일) | `buc/data/degrade.py` | FND-06, TCH-01 | 3/3 | ◐ | U | 0.8k | S·m | S→O | 0.75 | 2.9 | P | 열화 파라미터가 곧 MRH.obs 라벨 |
| **DAT-11** | 에셔 합성 9계열 | `buc/data/escher_synth.py` | DAT-01, INV-01 | 3/3 | ◐ | U+G+V | 0.9k | S·m | O→O | 0.75 | 3.5 | M | 판독기 국소 진실 라벨(CT-M14) |
| **DAT-12** | 글리프 합성 (OFL 폰트·한글 11,172자·가짜 글리프 음성) | `buc/data/glyph_synth.py` | FND-06 | 2/2 | ○ | U | 0.5k | H·m | S→O | 0.5 | 1.2 | P | 스크립트 |
| **DAT-13** | 카메라 궤적 코퍼스 (Toric 문법·Blender 카메라 추출·GLOMAP) | `buc/data/traj.py` | CAM-01 | 3/3 | ○ | U | 0.8k | S·m | O→O | 0.75 | 2.5 | N | CamFM-Intent용 |
| **DAT-14** | CRM 합성 선호쌍 (지터·충돌·수평 기울기·전경 레이어링 제거) | `buc/data/crm_pairs.py` | CAM-01, DAT-13 | 2/2 | ○ | U | 0.4k | H·m | S→O | 0.4 | 2.9 | N | 열화 규칙 적용 |
| **DAT-15** | band-iso 타일·견본 학습셋 공장 | `buc/data/bandiso.py` | TCH-02, DAT-01 | 3/3 | ○ | G | 0.6k | S·m | O→O | 0.75 | 3.5 | M | TexelFM 학습 데이터 |
| **DAT-16★** | 크기 사전 테이블 (~500 클래스 μ·σ) | `data/size_prior.yaml, tools/size_prior_check.py` | FND-01 | 2/3 | ● | U | 0.3k | H·m | O→O | 0.4 | 0.7 | S | Haiku가 초안, Opus가 수치 검수(도메인 지식) |

### 3.12 EVL 평가·벤치·KPI

고정 입력 24장 매일 A/B가 최대 레버다. 평가 모델은 R-EVAL 방화벽 뒤에 둔다. 단위 14개, 코드 8.5k줄, 9.6 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EVL-01★** | 고정 입력 24장 매니페스트 (출처 태그·크롭 규약·커버리지 매트릭스) | `eval/inputs24/manifest.yaml, tools/inputs_qa.py` | FND-01 | 2/4 | ● | U+V | 0.2k | H·m | O→F | 0.4 | 0.7 | S | 선별 편향이 승률 해석을 오염. 매트릭스는 Fable이 적대 검토 |
| **EVL-02★** | 일일 A/B 러너 (24장 고정 무브→전 구간→mp4→전일 대비 시트, 야간 잡) | `eval/daily_ab.py` | PRD-01, EVL-01, RDR-06 | 3/3 | ● | G+V | 0.8k | S·m | O→O | 0.75 | 2.8 | S | 최대 레버(반복 횟수)의 엔진 |
| **EVL-03** | 경쟁 클립 수집 도구 (COMP 격리·프레임 정렬·해상도 정규화) | `eval/comp_ingest.py` | FND-04 | 2/2 | ○ | U | 0.3k | H·m | S→O | 0.4 | 1.1 | P | Seedance 클립은 창업자가 생성 |
| **EVL-04★** | FFD·FFG·ΔE00·ΔG·FP(t)·OIR (평가 전용 LPIPS, 별도 프로세스) | `eval/first_frame.py` | FND-06 | 3/3 | ● | U+G | 0.8k | S·m | O→O | 1 | 1.8 | S | R-EVAL 격리가 핵심 |
| **EVL-05** | IR 계산기 (IR_geo·area·info·C·m_eff) | `eval/ir.py` | RDR-05 | 3/3 | ◐ | U | 0.6k | S·m | O→O | 0.5 | 2.8 | P | 렌더러 AOV에서 공짜로 나옴 |
| **EVL-06** | GeCo 재구현 (평가 전용, R-EVAL) | `eval/geco.py` | RUN-05 | 4/4 | ○ | G | 0.9k | S·h | O→F | 1.5 | 2.9 | M | 평가 모델은 시스템 경로 밖에서만 허용 |
| **EVL-07** | halo 지수·반사 문신 지수·seam 2AFC 자동 지표 | `eval/artifact_metrics.py` | EVL-04, RDR-05 | 3/3 | ● | U+G | 0.6k | S·m | O→O | 0.75 | 3.0 | P | 예측 아티팩트 1순위를 수치로 추적 |
| **EVL-08** | 블라인드 패널 페이지 (2AFC 수집, 정적 HTML) | `eval/panel/index.html, collect.py` | FND-01 | 2/2 | ● | V | 0.6k | H·m | S→O | 0.5 | 0.8 | M | D9·D12 패널용. 눈 적응 방지 |
| **EVL-09** | BT 집계·KPI 카드·일일 한국어 리포트 | `eval/report.py` | EVL-04, EVL-08 | 2/2 | ○ | U | 0.5k | H·m | S→O | 0.4 | 2.1 | M | 집계와 렌더링 |
| **EVL-10** | 벤치 정의식·벤치 모드 계약 (경로 고정·β=0·사다리 단계·홈/어웨이) | `eval/bench_def.py` | CAM-02, FND-04 | 3/4 | ○ | U | 0.5k | S·m | O→F | 0.5 | 2.8 | M | 공정성 규칙. 홈 경기 편향 방지 |
| **EVL-11** | 골든 회귀·결정론 스모크 (no-op PR에서 Merkle 불변) | `tests/golden/*` | FND-03, PRD-01 | 2/2 | ◐ | U+G | 0.4k | H·m | S→O | 0.4 | 2.4 | G | 야간 A/B와 별도의 빠른 안전망 |
| **EVL-12** | 실험 패키지 A (X-FF0 항등 루프·X-FF·X-INV) | `eval/exp_a.py` | EVL-04, WLD-02, RDR-04 | 3/3 | ● | G+V | 0.7k | S·m | O→O | 0.75 | 7.4 | M | 첫 프레임 바닥 FFD ≤0.015 확인 |
| **EVL-13** | 실험 패키지 B (X-ESC·X-ESC-H·MRH 게이트 E3 (a)(d)) | `eval/exp_b.py` | RD-11, RD-12, EVL-04, MDL-04 | 3/3 | ◐ | G+V | 0.7k | S·m | O→O | 0.75 | 10.7 | M | 판독기 정확도 게이트 |
| **EVL-14** | 실험 패키지 C (X-REFL·X-EDGE·X-LOT·X-TURN·X-FACE-Z) | `eval/exp_c.py` | EVL-07, RD-17, RD-18, WLD-19, CAM-04 | 3/3 | ◐ | G+V | 0.9k | S·m | O→O | 1 | 11.4 | N | 확장 레버 검증. 15일 후반·밤 잡 |

### 3.13 GOV 방화벽·계보·린트

하위 모델이 쓴 코드가 상위 원칙을 어기지 못하게 하는 자동 검사 계층. 단위 6개, 코드 3.3k줄, 3.1 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GOV-01★** | Lineage Ledger CI+vault 구조 (조상 DAG 해시·BLOCK_TRAIN 검사·weights=None·체크포인트 해시 금지목록) | `buc/gov/lineage.py, ci/lineage_check.py` | FND-04 | 4/4 | ○ | U | 1k | S·h | O→F | 0.75 | 1.4 | S | 하위 모델이 쓴 코드가 원칙을 어기지 못하게 하는 방화벽 |
| **GOV-02** | 카나리아 (가짜 USER/EXTGEN 샘플 주입→CI 실패가 정상) | `tests/canary/*` | GOV-01 | 2/3 | ○ | U | 0.3k | H·m | S→O | 0.4 | 1.8 | G | 테스트 작성은 Haiku 적합 |
| **GOV-03★** | 렌더 경로 린트 세트 (no_input_pixels_in_render_path·canvas_extension_lint·refl_bake_lint·preset_lineage_check) | `buc/gov/lints.py` | FND-01, FND-02 | 4/3 | ◐ | U | 0.8k | S·h | O→F | 1 | 2.0 | S | I6(Read-Only Camera)를 코드로 강제. AST 규칙은 Fable이 설계 검토 |
| **GOV-04** | 순서 순도 체크 (셔플 8회 Merkle/OIE·branch/ring order) | `buc/gov/order_purity.py` | FND-03, PRD-01 | 3/3 | ○ | U+G | 0.5k | S·m | O→O | 0.5 | 2.5 | G | 비회귀 조항의 자동 증명 |
| **GOV-05** | 사전학습 import 금지 린트·R-EVAL 격리 검사 | `buc/gov/import_ban.py` | FND-01 | 2/2 | ○ | U | 0.4k | H·m | S→O | 0.25 | 0.5 | G | 금지 목록 AST 검사 |
| **GOV-06** | 홈/어웨이·HOLDOUT·벤치 모드 검사 | `buc/gov/bench_guard.py` | FND-04, EVL-10 | 2/2 | ○ | U | 0.3k | H·m | S→O | 0.25 | 3.0 | G | 꼬리표 검사 |

### 3.14 PRD 파이프라인 스파인·제품 표면

스테이지 DAG와 스텁↔실물 교체. 걷는 해골을 가능하게 하는 뼈대. 단위 6개, 코드 3.7k줄, 3.0 에이전트-일.

| ID | 이름 | 산출물 | 의존 | 난/위 | 화면 | 검증 | LOC | 구현 | 설계→검토 | 소요 | D | 분류 | 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PRD-01★** | 파이프라인 스파인 (S−1…S11 스테이지 DAG·Merkle 캐시·부분 재실행·스텁↔실물 교체) | `buc/pipeline/spine.py` | FND-02, FND-03, FND-07 | 3/5 | ● | U+G | 1k | S·h | O→F | 1 | 2.0 | S | 이음매를 지배. 교체 가능성이 설계의 전부 |
| **PRD-02** | CLI/API (buc world·shoot·report) | `buc/cli.py` | PRD-01 | 1/1 | ○ | U | 0.4k | H·m | S→O | 0.4 | 2.4 | M | 얇은 래퍼 |
| **PRD-03★** | 스텁 팩 (고전 판독 스텁·평면 세계 스텁·고정 무브 스텁) | `buc/pipeline/stubs/*.py` | PRD-01, FND-02 | 2/3 | ● | U | 0.8k | H·m | O→O | 0.5 | 2.5 | S | 걷는 해골. 계약만 지키면 되는 단순 코드 |
| **PRD-04** | 촬영 메뉴+등급 추천 (IR_pred·경고·m_eff) | `buc/pipeline/menu.py` | CAM-02, CAM-04, EVL-05, RD-21 | 3/3 | ◐ | U+V | 0.6k | S·m | O→O | 0.5 | 6.4 | P | 정직한 메뉴가 확장·풀 승률을 지킴 |
| **PRD-05** | 납품 패키징+S11 리포트 (KPI·사다리 단계·치트 몫·Proof-of-Shot) | `buc/pipeline/deliver.py` | EVL-04, EVL-05, RD-21, PRD-01 | 2/2 | ◐ | U | 0.5k | H·m | S→O | 0.4 | 6.3 | M | 집계와 직렬화 |
| **PRD-06** | 비교 시트 생성기 (vs Seedance A/B 토글·크롭 확대·FFD 오버레이) | `buc/pipeline/compare_sheet.py` | EVL-02, EVL-03 | 2/2 | ● | V | 0.4k | H·m | S→S | 0.25 | 3.0 | P | 창업자 판정용 화면. 시각 자료 |

## 4. 집계

### 4.1 서브시스템별

| 서브시스템 | 단위 | LOC(k) | 에이전트-일 | 난이도≥4 | 위험≥4 | 화면● | S | P | M | N | G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FND | 8 | 4.6 | 3.2 | 0 | 2 | 0 | 6 | 0 | 0 | 0 | 2 |
| RUN | 8 | 4.4 | 4.0 | 1 | 3 | 0 | 5 | 0 | 0 | 1 | 2 |
| TCH | 11 | 6.5 | 8.9 | 3 | 3 | 0 | 3 | 0 | 4 | 4 | 0 |
| RDR | 9 | 7.0 | 8.2 | 4 | 4 | 5 | 3 | 1 | 3 | 2 | 0 |
| LNS | 8 | 5.3 | 5.9 | 4 | 2 | 5 | 4 | 1 | 1 | 2 | 0 |
| CAM | 7 | 4.8 | 5.3 | 1 | 0 | 3 | 2 | 1 | 4 | 0 | 0 |
| RD | 21 | 16.7 | 18.2 | 13 | 12 | 16 | 5 | 7 | 7 | 2 | 0 |
| WLD | 21 | 17.7 | 21.5 | 12 | 13 | 10 | 4 | 4 | 9 | 4 | 0 |
| INV | 9 | 7.5 | 7.0 | 0 | 0 | 0 | 1 | 0 | 6 | 2 | 0 |
| MDL | 22 | 14.6 | 22.6 | 5 | 5 | 6 | 1 | 6 | 8 | 7 | 0 |
| DAT | 16 | 10.0 | 9.9 | 1 | 2 | 6 | 4 | 5 | 5 | 2 | 0 |
| EVL | 14 | 8.5 | 9.6 | 1 | 3 | 6 | 3 | 3 | 6 | 1 | 1 |
| GOV | 6 | 3.3 | 3.1 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 4 |
| PRD | 6 | 3.7 | 3.0 | 0 | 1 | 3 | 2 | 2 | 2 | 0 | 0 |
| **합계** | **166** | **114.6** | **130.6** | 47 | 51 | 60 | 45 | 30 | 55 | 27 | 9 |

### 4.2 구현 등급별 (R1–R3)

| 구현 등급 | 단위 | LOC(k) | LOC 비중 | 에이전트-일 | 비고 |
|---|---|---|---|---|---|
| Haiku 5.5 (medium) | 31 | 12.3 | 10.7% | 11.8 | 스키마·스크립트·파서·집계·테스트 작성. 완결 스펙만 |
| Sonnet 5.5 | 131 | 95.9 | 83.7% | 109.8 | medium 83개, high 48개. 판독·렌더·솔버·모델 대부분 |
| Opus 5.5 | 4 | 6.4 | 5.6% | 9.0 | RDR-03(TSQ CUDA), RD-12(기하 화해), WLD-07(의미 패스), WLD-09(OTBI+TABP) |

### 4.3 설계·검토 등급별

| 등급 | 설계 담당 단위 | 검토 담당 단위 | 쓰는 곳 |
|---|---|---|---|
| Opus 5.5 | 122 | 130 | 기본 설계자·기본 검토자 |
| Fable 5.1 | 3 | 29 | 게이트 전용(R5). 아래 목록 |
| Sonnet 5.5 | 41 | 7 | 스펙이 문서에 이미 있는 단순 단위의 설계, Haiku 구현물의 교차 검토 |

Fable이 들어가는 단위 (29개): FND-02, FND-04, TCH-05, TCH-07, RDR-03, LNS-08, CAM-03, RD-04, RD-12, RD-13, RD-20, WLD-01, WLD-07, WLD-09, WLD-11, WLD-12, WLD-13, WLD-15, WLD-19, MDL-03, MDL-07, MDL-08, MDL-12, EVL-01, EVL-06, EVL-10, GOV-01, GOV-03, PRD-01

### 4.4 검증 방법·위험·화면

- 단위 테스트로 판정 가능(U 포함) 104개, 그중 U만으로 끝나는 것 49개. GPU 필요(G 포함) 96개. **사람의 시각 판정 필요(V 포함) 61개**.
- V가 붙은 61개는 창업자 눈과 블라인드 패널(EVL-08)의 시간을 먹는다. 예측 패널 경고: 창업자 눈은 사흘이면 판지 말에 적응한다. 패널은 D9·D12 두 번 돌린다.
- 화면 경로: ● 직접 60개, ◐ 간접 55개, ○ 없음 51개.
- 위험 5: FND-02, TCH-10, RDR-03, RD-12, RD-13, WLD-03, WLD-07, WLD-09, WLD-12, PRD-01. 난이도 5: TCH-10, RDR-03, RD-12, RD-13, WLD-07, WLD-09, WLD-12.
- 위험 5이면서 화면 ●인 단위는 WLD-03, PRD-01, WLD-07, RD-12다. 핵심 에이전트와 창업자의 눈은 여기에 먼저 간다. 그중 WLD-03(깊이 세계 빌더)는 슬라이스 임계 경로 위에 있고 시각 판정(V)이 필요하다. 나머지 셋은 설계 검토가 Fable이거나 타임박스 대상이다.

## 5. 소형 모델 14종 대응표 (하나도 빼지 않았다)

PARADIGM §5.3 Phase 1 카탈로그 M1–M14 전원이다. 단위 22개로 나뉘었다(M3은 4개, M5는 3개, M7은 2개, M14는 2개). **범위는 그대로고 순서만 정한다**: 분류가 S·P면 먼저, M이면 D10–14 주력 시간대, N이면 밤 잡·보조 에이전트(예측 패널의 '주의력 분류'). 모두 D15까지 gen0 체크포인트를 낸다.

| M# | 모델 | 크기 | 단위(분류, 이론 D) | 문서 학습량 T4-h | 5090-h 추정 | gen0 학습량 제안 | 화면 기여 확률(예측) | 선행 |
|---|---|---|---|---|---|---|---|---|
| M1 | TAE-T | 2.5M | MDL-01(P, 3.1) | 27 | 4.5 | 교사 VAE 쌍 크롭 100만(문서 400만의 25%) | 0.60 | TCH-01 교사 VAE, DAT-05 |
| M2 | LatentDown/Up-S | 2×1M | MDL-02(M, 3.5) | 3 | 0.5 | 100%(원래 작음) | 0.45 | M1의 Enc 쌍 재사용 |
| M3 | MRH (독심 헤드) | 26M | MDL-03(S, 4.4), MDL-04(P, 5.2), MDL-05(P, 5.4), MDL-06(P, 5.4) | 8–14 (+특징 공장 80) | 1.5–2.5 (+13) | 합성 3–5만 장 + 실사 MVS 가능한 만큼. v0은 1일 학습 후 P 단계에서 재학습 | 0.60 | DAT-02 Realify+탭, DAT-08 MVS, TCH-03 |
| M4 | G-Adapter (조건부) | 25M | MDL-07(N, 6.4) | 36–110 | 6–18 | E3 판정(SDEdit만으로 충분하면 생략). 코드는 만든다 | 0.15 | MRH 이후, DAT-01, TCH-02 |
| M5 | TexelFM-35 | 35M | MDL-08(M, 5.5), MDL-09(M, 6.2), MDL-10(M, 6.2) | 90 | 15 | 문서 1,000만 샘플의 10–20%(100–200만). RAW 매크로는 계획의 1–10%만 모임 | 0.35 | DAT-06 CC0 피라미드, DAT-07 RAW, DAT-15 교사 타일 |
| M6 | T-LPIPS-S | 4M | MDL-11(P, 3.9) | 4 | 0.7 | 라벨 쌍 20만 중 8만 이상(40%) | 0.50 (화면에 안 보임) | DAT-10, TCH-02 |
| M7 | MassingFM | 28M | MDL-12(M, 4.8), MDL-13(M, 5.5) | 32–50 | 5–8 | Render-Hide-Complete 장면 120–180만(문서 600만의 20–30%) | 0.45 | INV-01·04 문법 장면 |
| M8 | BRP | 42M | MDL-14(N, 5.0) | 9–18 | 1.5–3 | ProcGen 4만(문서 20만의 20%) | 0.20 | INV-03 OOG, INV-04 |
| M9 | IPM-S | 8M | MDL-15(N, 2.9) | 2–3 | 0.4–0.5 | 100% | 0.20 | INV-01 절차 렌더(라벨 공짜) |
| M10 | CamFM-Intent | 20M | MDL-16(N, 4.0) | 13 | 2 | 궤적 40–100만(문서 200–500만의 20%) | 0.20 | DAT-13 궤적 코퍼스 |
| M11 | CamFM-Hand | 8M | MDL-17(M, 3.1) | 3 | 0.5 | IMU 40–60 기기-시간(문서 300의 13–20%, 예측 패널 값) | 0.55 | CAM-06, 창업자 IMU 산책 |
| M12 | CamFM-CRM | 12M | MDL-18(N, 3.9) | 1–2 (+프리뷰 렌더 18) | 0.3 | 합성 선호 200만 + 사람 쌍 수백(5k는 15일에 안 모임) | 0.10–0.15 | DAT-14 |
| M13 | Rig-ID-S | 3M | MDL-19(N, 2.5) | 0.2 | 0.03 | 100% | 0.05 | CAM-01 리그 시뮬 |
| M14 | Glyph-S | 4M (+CTC 2.8M) | MDL-20(P, 3.1), MDL-22(N, 3.9) | 3–4 | 0.6 | 100%. 검출기 먼저, CTC는 야간 | 검출기 0.45, CTC·SYMBOL 0.25–0.30 | DAT-12 글리프 합성 |

- 합계(문서 기준): P1 코어 1세대 195–231 T4-h + G-Adapter 36–110. 5090으로는 T4-h ÷ 4–8(중심 6, **추정**, D1 RUN-05·TCH-01 실측으로 교정)이라 모델 학습 약 35–45 5090-h, 데이터 공장(PARADIGM v3 장부 350–407 T4-h)이 약 55–70 5090-h, 합쳐 **약 90–115 5090-h**다.
- 이 값은 `ops/STATE.md`의 'Phase 1 완주 2–4만원(현실치 3–7만원)'과 같은 자릿수다. 단 단가 의존이 크다. 한국 호스트 1장 $0.213/h면 약 $19–25(3만원 안팎), 2×5090 $0.83/h로 돌리면 약 $75–95(10–13만원)다. 충전 5만원이면 1장 기준 약 170시간, 2×5090 기준 약 43시간이다. 그래서 학습은 1장 위주로, 매일 야간 A/B는 같은 장에서 큐(RUN-04)로 돌리고, 슬라이스 이후(D5 이후)에만 두 번째 임대를 검토한다.
- 순서 규칙: D3–4 학습 착수 = M3(MRH 본체) > M1 > M14 검출기 > M11 > M6 > M2. D4–6 = M7 > M5 > M13. D6–9 = M9, M12, M10, M8, M4(E3 판정). 이론 완료일은 위 표의 괄호 값이다.
- 예측 패널 기대: 체크포인트 9–12종이 나오고 영상에 의미 있게 기여하는 것은 4–6종(MRH, CamFM-Hand, TAE-T, LatentDown/Up, MassingFM, TexelFM). 나머지는 '출석률 100%에 활약률 35%'로 보고하되, 만든다.

## 6. 임계 경로 D0–D15

D 값은 의존만 보고 계산한 이론 최단일이다(인력·GPU 대기 제외). 현실은 ×1.3–1.5로 읽는다. 아래 사슬은 의존 그래프(166노드)에서 코드로 계산했고 순환이 없음을 확인했다.

### 6.1 세 개의 사슬

| # | 사슬 | 이론 D | 현실(×1.3–1.5) | 성격 |
|---|---|---|---|---|
| ① | **슬라이스 사슬(여유 0)**: FND-01 → FND-07 → RUN-01 → RUN-05 → TCH-01 → TCH-03 → DAT-02 → MDL-03 → RD-07 → RD-08 → WLD-03 → WLD-08 | 6.9 | D9–10 | 교사 기동 → 탭 → Realify → MRH 학습 → 판독 → 깊이 세계 → 발명 채움. 이 열두 개가 15일 전체의 척추다 |
| ② | **판독 정밀 사슬**: RD-07 → RD-09 → RD-10 → RD-11 → RD-12 (기하 화해) | 9.9 | D13–15 | 확장·풀 등급과 에셔 입력용. 화면 영향은 안전 등급에서 작다. **D11 타임박스** |
| ③ | **의미 패스 사슬**: WLD-03 → WLD-05 → WLD-07 → WLD-09 (→ WLD-12 → WLD-13) | WLD-07 9.4, WLD-09 11.9, WLD-12 14.9, WLD-13 15.7 | WLD-09 D15–18 | 가장 새롭고 위험(예측 0.35). WLD-09는 현실 보정하면 D15를 넘는다. **D12 채택 판정 하드 타임박스.** WLD-12·13은 이론상 D15에 겨우 닿으므로 밤 잡으로만 돌린다(N) |

사슬 ③의 함의: 의미 패스와 OTBI가 D15 영상에 **못 들어가도 정상**이다. 출하 경로는 WLD-08(뷰별 SDEdit+평균 역투영)이고, 예측 패널 p50이 이미 "의미 패스는 반쯤 돈다, Spec-commit과 래칫은 이름만 있다"다.

### 6.2 일자별 (이론 D 기준으로 닫히는 단위)

| 날 | 닫히는 단위 (이론 D) | 임계 경로 위 | 같이 도는 것 | 게이트·사람 |
|---|---|---|---|---|
| **D0** (확정일) | 파일럿 R3 비교, FND-01 | FND-01 | FND-02 설계 초안, EVL-01 후보 80장 수집, GOV-01·FND-04 착수 | 창업자: 결제·Vast 키·HF 토큰, 촬영 개시. 가중치·라이선스 접근 확인 |
| **D1** | FND-01·02·03·04·06·07, RUN-01·03, DAT-16, EVL-01 (≤1.0) | FND-07 → RUN-01 | 계약 동결 | **G0 (Fable)**: FND-02·FND-04·PRD-01 시그니처 동결 |
| **D2** | RUN-05 1.4, GOV-01 1.4, LNS-01 1.4, RD-01 1.5, RDR-06 1.3, RDR-02 1.8, CAM-01 1.8, RUN-02 1.8, EVL-04 1.8, INV-01 2.0, PRD-01 2.0, WLD-01 2.0 | RUN-05 | 첫 Vast 임대·스모크, 예산 캡 시험 | 5090에서 gsplat·FA 빌드 가능 여부 판정 |
| **D3** | TCH-01 2.2, RUN-06 2.2, LNS-04 2.2, RDR-05 2.3, CAM-02 2.3, RD-02 2.5, PRD-03 2.5, LNS-03 2.5, TCH-02 2.7, DAT-01 2.8, EVL-02 2.8, LNS-02 2.8, TCH-03 2.9, WLD-17 3.0 | TCH-01 → TCH-03 | **걷는 해골 클립**(7.3) | **G1 (Fable)**: 해골 클립과 계약 테스트 |
| **D4** | DAT-02 3.4, RD-03 3.2, MDL-01·17·20 3.2, RD-04·06 3.7 | DAT-02 | **MRH 학습 착수(밤)**, TAE-T·Hand·Glyph 학습, ExplainAway 최소판 | 창업자 촬영 D1–4 종료, IMU·방 MVS·RAW 매크로 업로드 |
| **D5** | MDL-03 4.4, MDL-12 4.8, RD-07 4.9 | MDL-03 → RD-07 | DAT-08 MVS 투입, MassingFM 학습, TexelFM 착수 | MRH v0 체크포인트 육안 확인 |
| **D6** | RD-08 5.2, MDL-04 5.2, MDL-05·06 5.4, MDL-08·13 5.5, RD-14·18 5.7, RD-21 5.9, RD-05 5.9 | RD-08 | WLD-03 착수, 경계 처리 | |
| **D7** | WLD-03 6.2, RD-15 6.4, WLD-02 6.7, WLD-05·08 6.9, RD-20 6.9 | WLD-03 → WLD-08 | | **슬라이스 첫 클립 + 24장 첫 야간 A/B. G2 (Fable)**: FFD·FFG 수치와 눈 판정 |
| **D8–9** | RD-10 7.2, EVL-12 7.4, WLD-10·11·14 7.7 (P 합류) | WLD-10/11 (이음매) | 깊이 에지 수술(레버 2), 마스크·매팅, 이음매 지우기(레버 5) | **D9 패널 #1**(EVL-08), 창업자 눈 |
| **D10–12** | WLD-07 9.4, RD-12 9.9, WLD-09 11.9 (타임박스 대상) | WLD-07 | TexelFM gen0·MassingFM 합류, InputPolish, FitLights | **D12 패널 #2**, **G4 (Fable)**: 기능 동결, 의미 패스·OTBI 채택/폐기 |
| **D13** | | | 확장 무브(90° 오빗) 2차 대결, N 클래스 체크포인트 수집 | 실패 분석 |
| **D14** | | | 최종 후보 선정, 릴 편집, T4 실측 몇 시간(RUN-08) | **G5 (Fable)**: 선별 편향 캡션 검토 |
| **D15** | | | 최종 릴 3–6편 + 정직 리포트(PRD-05·06), 시간별 보고 마무리 | 창업자 최종 미감 판정 |

### 6.3 타임박스 (실패해도 폴백이 있는 것은 정해진 날 자른다)

| 단위 | 시작 | 자르는 날 | 채택 기준 | 폴백 |
|---|---|---|---|---|
| RDR-01 gsplat (sm_120) | D2 | D3 오전 | 5090에서 빌드·미분 통과 | 순수 PyTorch/Warp 2DGS, 또는 쿼드 유지 |
| MDL-03 MRH v0 | D4 | D5 | 24장에서 깊이가 plausible(수동 확인) | RD-02 평면 깊이 스텁으로 D7 클립을 내고 MRH는 D8 합류 |
| RDR-03 TSQ 커널 | D6 | D8 (Sonnet 1일 → Opus 에스컬레이션) | 스침각 바늘 스플랫 제거 | 2DGS·쿼드 유지(예측 0.40이라 기대하지 않는다) |
| RD-12 기하 화해 | D8 | D11 | 에셔 황색 입력의 FFG ≤1 px | λ 최소 고정 + 안전 등급만 개방 |
| WLD-07 의미 패스 | D8 | D12 | 24장 A/B에서 WLD-08 대비 +5%p 이상 | WLD-08 유지 |
| WLD-09 OTBI+TABP | D9 | D12 | 같은 기준 | WLD-08 유지 |
| TCH-09·10, RUN-08 (T4 부록) | D13 | D14 | T4에서 4K 프레임 시간 실측 | 측정 못 하면 보고서에 '미측정' 표기 |

### 6.4 매일의 리듬

`밤: EVL-02가 24장 × 고정 안전 무브를 끝까지 돌린다 → 아침: PRD-06 비교 시트 + EVL-04·07 수치 + 시간별 보고(RUN-07) → 창업자 눈과 점수 → 낮: 가장 큰 아티팩트 하나를 고친다 → 다시 밤`. 예측 패널 레버 1의 효과(전 구간 반복 20–40회 → 50–80회)가 이 리듬에서 나온다. 반복이 곧 품질이다.

## 7. D6–7 수직 슬라이스 최소 집합

### 7.1 정의

**슬라이스** = 입력 사진 1장이 `판독(R0·R1·R5·R6) → MRH 깊이 → 경계 처리(LDI 찢기) → 평면 쿼드 세계 → 발명 채움(SDEdit+far-field) → 카메라 리그 → 렌즈·센서 스택 → mp4`로 **끝까지 도는 것**이다. 못생겨도 된다(예측 패널 레버 1). 그다음 고정 입력 24장으로 매일 밤 전 구간을 돈다(EVL-02). 이 정의에서 단위 45개가 하드 의존 폐포다. 슬라이스 단위의 하드 의존은 전부 슬라이스 안에 있음을 스크립트로 확인했다.

- 이론 최단 완료일 **D6.9**. 그 날 여유 0일인 사슬이 아래 12개다(임계 경로). 현실 보정(×1.3–1.5)을 하면 D9–10이고, 예측 패널의 첫 종단 픽셀 p50 D10, 80% 구간 D8–13과 같은 곳에 떨어진다.
- **임계 사슬(여유 0):** FND-01 → FND-07 → RUN-01 → RUN-05 → TCH-01 → TCH-03 → DAT-02 → MDL-03 → RD-07 → RD-08 → WLD-03 → WLD-08
- 그래서 D6–7 슬라이스를 지키려면 **두 장치**를 둔다. (1) D3까지 모든 스테이지를 스텁으로 이어 붙인 '걷는 해골'(7.3)을 먼저 낸다. 스텁↔실물 교체 계약이 PRD-01·FND-02에 있다. (2) MRH(MDL-03)는 D4.4에 나오는 1일짜리 v0으로 시작하고, 실패하면 RD-02의 고전 평면 깊이 스텁으로 D7 클립을 내고 MRH는 D8에 끼운다.

### 7.2 최소 집합 (분류 S, 45개)

| 서브시스템 | ID | 이름 | D | 여유일 | v0 최소판 |
|---|---|---|---|---|---|
| FND | **FND-01★** | 레포 골격·툴체인 락·GPU 이미지 | 0.2 | 0.0 | 명세 전량 |
| FND | **FND-02★** | 데이터 계약 v0 | 1.0 | 0.1 | World·ReadPacket·Shot·Theta 4종만 먼저 동결, 나머지는 확장 가능 필드로 |
| FND | **FND-03★** | 결정론 유틸 | 0.7 | 0.8 | 명세 전량 |
| FND | **FND-04★** | 계보 태그·prov 비트 | 0.7 | 1.5 | taint·ext/hold 비트와 BLOCK_TRAIN 검사만 |
| FND | **FND-06★** | 기하·색·필터 수학 유틸 | 0.8 | 0.4 | 명세 전량 |
| FND | **FND-07★** | 설정·실험 매니페스트·장부 | 0.5 | 0.0 | 명세 전량 |
| RUN | **RUN-01★** | Vast.ai 오케스트레이터 | 1.0 | 0.0 | 명세 전량 |
| RUN | **RUN-02★** | pull 러너 | 1.8 | 5.2 | 잡 1종(train)·1종(render)만 지원 |
| RUN | **RUN-03★** | 체크포인트·HF Hub 비공개 저장소 동기 | 0.7 | 2.0 | 명세 전량 |
| RUN | **RUN-05★** | GPU 서버 부트스트랩 | 1.4 | 0.0 | Z-Image·Qwen3 가중치 prefetch와 gsplat 빌드는 별도 잡으로 분리 |
| RUN | **RUN-06★** | 학습 하네스 공용 | 2.1 | 1.2 | 명세 전량 |
| TCH | **TCH-01★** | 교사 로더 | 2.1 | 0.0 | 명세 전량 |
| TCH | **TCH-02★** | σ 스냅 스케줄·SDEdit·x̂0 API | 2.6 | 0.2 | Turbo 스케줄 8점 스냅, SDEdit t=0.3–0.6, 단일 뷰부터 |
| TCH | **TCH-03★** | Feature-Tap Bus | 2.9 | 0.0 | 층 8/15/23 탭 + PCA-128 적합 1회, ext 바이패스 플래그는 껍데기 |
| RDR | **RDR-02★** | 평면 쿼드·LDI 래스터 | 1.8 | 0.1 | 깊이 레이어 카드 3–8장, 알파·소프트 에지, torch로 먼저 |
| RDR | **RDR-05★** | AOV | 2.2 | 0.1 | depth·instance id만. 관측 bitset은 P 단계 |
| RDR | **RDR-06★** | 영상 인코더·타임라인 | 1.2 | 4.9 | ffmpeg 24fps 1080p H.264만 |
| LNS | **LNS-01★** | θ 스키마·사전분포 샘플러·수제 프리셋 20 | 1.4 | 4.8 | θ 필드 전체 스키마, 프리셋 3개만 |
| LNS | **LNS-02★** | DOF·Defocus Pyramid·포커스 브리딩·랙 스케줄 | 2.8 | 4.2 | CoC(z) 레이어 블러, 랙 스케줄. Defocus Pyramid는 CoC>8px 층만 |
| LNS | **LNS-03★** | 셔터 | 2.5 | 4.4 | 180° 셔터 + 회전 블러만. 롤링은 P |
| LNS | **LNS-04★** | 센서·필름 | 2.1 | 4.8 | 그레인(Newson)·비네팅·65³ LUT 적용·할레이션 약하게 |
| CAM | **CAM-01★** | 리그 1 kHz ODE | 1.8 | 4.7 | 슬라이더·달리·오빗 3종, 핸드헬드는 OU 잡음 근사 |
| CAM | **CAM-02★** | 무브 레시피·촬영 메뉴 봉투 Π_g | 2.2 | 4.7 | 푸시인·슬라이더·±20° 오빗·랙 4종 고정 경로 |
| RD | **RD-01★** | R0 수납 | 1.5 | 2.4 | 색관리·r_eff·레터박스만. DQT는 P |
| RD | **RD-02★** | R1 기하 카메라 | 2.5 | 2.4 | 수평선·FOV 초기값까지. plumb-line은 P |
| RD | **RD-07★** | R5 독심 판독 호출 | 4.9 | 0.0 | 전역 σ 1점 + 타일은 P. 클린룸 레인은 플래그만 |
| RD | **RD-08★** | R6 미터 스케일 | 5.2 | 0.0 | 크기 사전 로그 LS. EXIF 앵커는 있을 때만 |
| RD | **RD-21★** | 판독 오케스트레이터·패킷 조립 | 5.9 | 1.0 | R0·R1·R5·R6만 연결. 나머지는 플러그인 자리(소프트 의존) |
| WLD | **WLD-01★** | 세계 컨테이너 STRATA-lite-min | 2.0 | 3.1 | LDI 층 + far-field + meta 부분집합 |
| WLD | **WLD-03★** | 깊이 세계 빌더 | 6.2 | 0.0 | MRH 깊이→불연속 찢기→LDI. 매팅 α는 RD-18(P)이 나중에 합류 |
| WLD | **WLD-08★** | 뷰별 SDEdit+평균 역투영 폴백 | 6.9 | 0.0 | 디스오클루전 영역만 SDEdit(t≈0.5)→역투영. ±20° 범위 |
| WLD | **WLD-17★** | far-field | 3.0 | 3.9 | 하늘 해석식 + 지면 연장 평면 |
| INV | **INV-01★** | 문법 코어+RoomGen v0 | 2.0 | 0.4 | RoomGen v0 + 장면 프로그램 + 레이캐스트. 장르 문법은 M |
| MDL | **MDL-03★** | M3 MRH 본체 | 4.4 | 0.0 | 깊이·법선·sem만 학습한 v0. 알베도·rough 등은 P |
| DAT | **DAT-01★** | 절차 장면 GT 렌더 파이프 | 2.8 | 0.1 | RoomGen 방 2,000개, 1024² G-buffer |
| DAT | **DAT-02★** | Realify+탭 동시 추출 패스 | 3.4 | 0.0 | 첫 샤드 5천 장(t∈{0.3,0.5}), 대량은 야간 병행 |
| DAT | **DAT-05★** | 샤드 I/O | 1.1 | 2.2 | 샤드 쓰기·읽기 + taint 검사 |
| DAT | **DAT-16★** | 크기 사전 테이블 | 0.7 | 4.2 | 상위 150 클래스만, 나머지는 P |
| EVL | **EVL-01★** | 고정 입력 24장 매니페스트 | 0.7 | 5.5 | 24장 선정 완료 + 출처 태그. 입력 사진은 창업자 촬영(D1–4)에 의존 |
| EVL | **EVL-02★** | 일일 A/B 러너 | 2.8 | 4.2 | 24장 × 안전 무브 1개, 야간 큐, 전일 대비 시트 |
| EVL | **EVL-04★** | FFD·FFG·ΔE00·ΔG·FP(t)·OIR | 1.8 | 5.2 | FFD(LPIPS 평가 전용 프로세스)·FFG만. OIR·FP(t)는 P |
| GOV | **GOV-01★** | Lineage Ledger CI+vault 구조 | 1.4 | 5.5 | 조상 태그 검사와 weights=None 검사만 |
| GOV | **GOV-03★** | 렌더 경로 린트 세트 | 2.0 | 4.9 | no_input_pixels_in_render_path 린트만 |
| PRD | **PRD-01★** | 파이프라인 스파인 | 2.0 | 4.2 | 스테이지 6개(판독·세계·발명·카메라·렌즈·인코딩) 스텁↔실물 교체 |
| PRD | **PRD-03★** | 스텁 팩 | 2.5 | 4.4 | 고전 스텁: 평면 깊이, 평면 세계, 고정 무브, 기본 θ |

여유일이 큰 단위(EVL-01, GOV-01, RUN-02, EVL-04 등 4일 이상)는 임계 사슬을 건드리지 않고 병행하다가 늦게 끝내도 된다. 반대로 임계 사슬 12개는 하루 늦으면 슬라이스가 하루 늦는다. 레인을 나눌 때 이 12개에 **가장 안정적인 에이전트와 가장 짧은 검토 대기**를 줘야 한다.

### 7.3 걷는 해골 (D3 목표, 전부 스텁)

목표는 D3 안에 `입력 → (스텁 판독) → (평면 스텁 세계) → 카메라 → 렌즈 → mp4`가 돌고 24장 야간 A/B 껍데기가 도는 것이다. 이론 최단 D2.8.
- 필요 단위: FND-01·02·03·06·07, PRD-01, **PRD-03(스텁 팩)**, RDR-02, RDR-06, LNS-01·02·03·04, CAM-01·02, EVL-01·02.
- 스텁: 판독 = 수평선 가정 평면 깊이 + 기본 θ, 세계 = 깊이 레이어 카드 3장, 발명 = 가장자리 복제, 무브 = 고정 경로. 못생긴 클립이 나와야 정상이다. 이 클립이 **계약이 맞물린다는 증거**이고 Fable 게이트 G1의 심사 대상이다.

### 7.4 슬라이스에서 일부러 뺀 것과 대체물

| 뺀 단위 | 슬라이스에서의 대체 | 합류 시점 |
|---|---|---|
| RD-18 경계 분해(매팅) | WLD-03의 깊이 불연속 LDI 찢기(고무 시트가 그대로 남는다) | D8–9 (P) |
| RD-03·04 ExplainAway, LNS-08 | 기본 θ 프리셋 적용. Soul풍 입력의 비네팅·그레인 문신이 남는다 | D5 최소판, D8 합류 (P) |
| MDL-04 MRH 전역 헤드 | RD-02 고전 FOV·수평선 | D8–9 (P) |
| WLD-02 입력 역투영(2단 소유) | WLD-03 안의 단순 레이어 텍스처 투영 | D8–9 (P) |
| MDL-05 MRH.obs | l_obs를 입력 Nyquist에서 직접 계산 | D8–9 (P) |
| MDL-08~10 TexelFM | 쌍선형 업샘플. 푸시인 끝 구간이 흐릿하다 | D10–12 (M) |
| MDL-12~13 MassingFM, INV-04 | WLD-17 far-field + 지면 연장. 프레임 밖은 ±20° 이내라 거의 드러나지 않는다 | D10–12 (M) |
| RDR-01 2DGS, RDR-03 TSQ | 평면 쿼드(RDR-02). 스침각 문제는 ±20°에서 작다 | D10 이후 (M) |
| MDL-17 CamFM-Hand | CAM-01의 OU 잡음 근사 핸드헬드 | D8–10 (M) |
| DP-0(CAM-03) | 고정 경로 4종 | D10 이후 (M) |
| WLD-05 L0 조명 bake, RD-10 FitLights | 입력 복사휘도를 그대로 소유(조명 모델 없음) | D10 이후 (M) |
| TCH-04 Exhaust, WLD-15 CoW | 캐시 없이 매번 재계산 | D10 이후 (M) |

### 7.5 D8–9 즉시 합류 (분류 P, 30개)

슬라이스가 닫힌 뒤 같은 사슬의 약점을 바로 메운다. 모두 이론 완료일이 슬라이스 전이거나 직후라 인력만 맞으면 D8–9에 합류한다.

EVL-03(1.1), DAT-12(1.2), CAM-04(1.4), RDR-04(2.2), DAT-08(2.4), LNS-08(2.6), EVL-05(2.8), DAT-10(2.9), EVL-07(3.0), PRD-06(3.0), RD-03(3.1), MDL-01(3.1), MDL-20(3.1), DAT-04(3.4), RD-04(3.6), RD-06(3.6), DAT-03(3.8), MDL-11(3.9), RD-16(4.2), MDL-04(5.2), MDL-05(5.4), MDL-06(5.4), RD-14(5.7), RD-18(5.7), RD-05(5.9), PRD-04(6.4), WLD-02(6.7), WLD-10(7.7), WLD-11(7.7), WLD-14(7.7)

## 8. 병렬 레인 배치 제안 (워크플로 동시 2명 제한 대응)

이 컨테이너는 CPU 4개라 워크플로 하나의 동시 에이전트가 2명이다. 131 에이전트-일을 D1–D14 13영업일에 넣으려면 동시 약 10명이 필요하다. 형제 클라우드 세션(각자 별도 컨테이너)을 레인 단위로 6개 띄우는 구성을 제안한다. 세션당 2명 × 13일 = 26 에이전트-일 용량이다.

| 레인 | 서브시스템 | 에이전트-일 | 용량 대비 | 임계 사슬 ① 소유 단위 | 비고 |
|---|---|---|---|---|---|
| A 기반·교사 | FND, RUN, TCH, GOV | 19.2 | 74% | FND-01, FND-07, RUN-01, RUN-05, TCH-01, TCH-03 | D0–3에 최우선. 여기가 늦으면 모두 늦는다 |
| B 렌더·렌즈·카메라 | RDR, LNS, CAM | 19.4 | 75% | (없음, 여유 4일+) | 슬라이스에 필요한 건 RDR-02·05·06, LNS-01~04, CAM-01·02 뿐. 나머지는 D8 이후 |
| C 판독·발명 | RD, INV | 25.2 | 97% | RD-07, RD-08 | 가장 빡빡하다. INV-03·05(N)와 RD-13·17(N)은 밤 잡·Haiku로 밀어 낸다 |
| D 세계 솔버 | WLD | 21.5 | 83% | WLD-03, WLD-08 | Opus 구현 2개(WLD-07·09)가 여기. 설계·검토 Fable |
| E 소형 모델 | MDL | 22.6 | 87% | MDL-03 | 학습은 GPU 큐(RUN-04)로. 코드는 데이터 도착 전에 끝낸다 |
| F 데이터·평가·제품 | DAT, EVL, PRD | 22.5 | 87% | DAT-02 | DAT-02는 E·A·C와 맞물리는 교차점. EVL-02는 D3 해골부터 돈다 |

- **임계 사슬 ①은 레인 5개를 가로지른다**(A→F→E→C→D). 인수인계마다 하루가 샌다. 오케스트레이터 세션에 **체인 캡틴**(Opus, 항상 ①만 본다)을 두고 매시간 12개 단위의 진행·막힘을 확인하는 것을 권한다.
- **레인 간 계약은 FND-02 하나**다. 레인이 서로의 코드를 읽지 않고 계약과 골든 픽스처(FND-08)로만 만나게 한다. 이음매가 진짜 적이라는 예측 패널의 결론에 대한 대응이다.
- **GPU는 1장이 기본**이다. 학습 잡·교사 호출·야간 A/B가 한 장을 나눠 쓰므로 RUN-04 큐로 우선순위를 준다(A/B > MRH > 나머지 학습 > N 클래스). 슬라이스 이후(D5–6)에만 두 번째 임대를 검토한다.
- 레인당 에이전트 구성: 구현 Sonnet 1–2, Haiku 1(테스트·스크립트·데이터), 검토는 Opus가 요청 순으로 돈다. Fable은 레인에 상주하지 않고 게이트 때만 호출한다.

## 9. 토큰 비용 추정과 시간 제약

**추정이다. 단위당 ±2–3배 오차를 본다.** 공식 가격(2026-10-06 캐시): Fable 5.1 $10/$50, Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 (MTok, 입력/출력), 캐시 읽기 Fable $0.25·Opus $0.20·Sonnet $0.20. Haiku는 캐시 읽기 가격이 없어 전부 신규 입력으로 계산했다(그래도 Sonnet 캐시 읽기보다 싸다).

가정: 구현 1k줄당 출력 2.5만 토큰, 신규 입력 12.5만, 캐시 읽기 100만 토큰(에이전트 루프 수십 턴). 재시도 계수 2배(낙관)–4배(비관). 검토는 단위당 Opus 입력 6만+출력 6천, Fable 입력 15만+출력 1.5만, 전 단위 Haiku 1차 리뷰 포함. 설계는 단위당 Opus 입력 5만+출력 8천. Fable 게이트 6회(입력 20만+출력 2만).

| 항목 | 낙관 | 비관 | 메모 |
|---|---|---|---|
| 구현 (H 3.1–6.2 + S 134–269 + O 15–31) | $153 | $305 | 이 계획 |
| 검토 (Opus 130, Fable 29, Sonnet 7, Haiku 전수) | $170 | $284 | 재검토 포함 |
| 설계 (Opus 122, Sonnet 41, Fable 3) | $54 | $81 | |
| Fable 게이트 6회 | $18 | $27 | G0–G5 |
| **합계** | **$395** | **$697** | 약 54–96만원(1$≈1,380원) |
| 대안 A: 구현 전부 Opus | $275 | $550 | 구현비만, 이 계획의 구현비 대비 1.8배 |
| 대안 B: 구현 전부 Fable | $630 | $1261 | 구현비만, 4.1배, 한 턴이 수 분 |
| 대안 C: 구현 전부 Sonnet | $160 | $321 | 구현비만, Opus 4개 대신 Sonnet 시도 후 에스컬레이션이 현실적 |

읽을 점 세 가지.

1. **지출의 대부분은 GPU가 아니라 에이전트 토큰이다.** 이 계획의 토큰 비용은 위 표 합계(약 $400–700)로, GPU 요금($20–95, §5)의 5–30배다. 결제 한도를 GPU 5만원이 아니라 토큰 수십만~백만 원 규모로 잡아야 한다. 여기에는 **매일 A/B 해석, 시간별 보고, 오케스트레이터 세션, GPU 오류 로그 디버깅**이 들어 있지 않다. 그것들이 +50–100% 더할 수 있다.
2. **시간이 진짜 병목이다.** 구현을 전부 Opus로 하면 구현비가 1.8배, Fable로 하면 4.1배이고 한 턴이 수 분이라 처리량이 직접 깎인다. 131 에이전트-일을 15일에 넣으려면 평균 8.7명이 필요한데 워크플로 하나는 2명이다. §8의 6레인(6세션 × 2명)이 최소 구성이다. 이 컨테이너는 CPU 4개라 한 세션 안에서 워크플로를 더 늘려도 소용없다.
3. **공식 원칙 확인 절차.** 완료된 작업당 비용으로 판단하라는 원칙에 따라 D0 파일럿(FND-02, RD-02)에서 Sonnet 캐스케이드와 Opus-low 단독을 직접 재서, R3의 Opus 승격 기준(난이도 5·위험 5)이 맞는지 정한다. Opus-low 단독이 더 싸게 끝나는 단위가 나오면 그 계열을 승격한다.

## 10. 이 목록에 넣지 않은 것 (이유와 함께)

범위를 슬쩍 줄인 것이 아니라 **기준 문서가 15일 스프린트 밖으로 두었거나, I2V 결정으로 사라졌거나, 조건이 발동해야 생기는 것**이다. 소형 모델 14종과 §4·§5·§6·§7·§8의 Phase 1 파이프라인·렌더·카메라·벤치 부품은 전부 위 166개 안에 있다.

| 항목 | 이유 |
|---|---|
| Phase 2 모델 N1–N10(EPD-S, MoPri-S, Feas-S, HeroProxy-S, ChronoProxy-S, MAN-S, F_φ-S, Reader-S, Denoiser-S, Expr-S, TTS-S) 및 P2-E2~E5 | Phase 2. 사용자 범위("Phase 1 소형 모델 14종")에 없음 |
| 쇼케이스 부품: Hero FX(Warp FLIP/MPM, Voronoi), DIEL SDF, Powers of Ten, 8K120 데모, 웹 뷰어 export, Corman 슬라이더, 체호프의 컵 | 기준 문서 일정이 Day 45–90. D15 목표(같은 사진으로 Seedance I2V보다 눈으로 좋은 영상)에 불필요 |
| Soul 레인(Higgsfield 메일, soul_vault 웹 생성 480회, X-LA, 룩 프리셋 Hybrid-A/A+/S, S 승인 해금) | 07 결정으로 핵심 의존성에서 선택적 파트너가 됨. 일반화된 사용자 콘텐츠 방화벽만 GOV-01·FND-04에 남김 |
| LookPrior-S, hero search(S1 히어로 오디션) | v3에서 삭제(CT-M6, 입력 이미지가 자기 룩과 hero) |
| 조건부 모델·어댑터: Ref-Adapter A3(O4), MRH.esc 0.5M(오지목 >20%일 때), Survey-LoRA A1, SR-LoRA A2, M2-k | 트리거 결과(E1·E3·X-ESC)가 나온 뒤에만 생김. 트리거가 켜지면 RUN-06 위에 단위 1개씩 추가 |
| FREEZE-TAG 전체 벤치(240 프롬프트 × 의도 12), 공개 Kaggle 재현 | 스프린트의 벤치는 고정 입력 24장 미니 벤치(EVL-10)로 한정. Kaggle은 약관 때문에 주력에서 제외 |
| Failure Twin 회귀셋, 사용자 업로드 삭제(crypto-shredding) 구현, 법무·약관 | 제품화 단계. 스프린트에는 vault 구조와 태그 규칙(GOV-01)만 |
| Sensor Purity Test 하드웨어(LED 카운터+피에조 보드) | 하드웨어 조달. IMU 대기 시간 보정이 필요해지면 창업자 작업 목록에 추가 |
| 얼굴 본격 오빗, 캐릭터(identity LoRA, Expr-S, Dot-Face) | Phase 2c. 15일에는 얼굴 봉투(CAM-04, RD-15)로 막는다 |

## 11. 가정과 열린 질문

### 11.1 내가 가정한 것 (틀리면 이 목록이 바뀐다)

1. 소요일·LOC·화면 기여는 **내 추정**이다. 소요는 에이전트가 하루 8시간 일하고 검토 대기를 포함한 값이다. 처음 3일 실측으로 계수를 교정한다(RUN-07 보고에 단위별 실제 소요를 넣는다).
2. 5090 환산 ÷6은 추정이다. D2(TCH-01·RUN-05)에서 교사 1024² 1회 시간, 소형 모델 한 스텝 시간을 재서 §5 표를 고친다.
3. 산출물 경로(`buc/…`)는 제안이다. 레포 구조를 바꿔도 단위는 그대로다.
4. 교사 가중치(Z-Image-Turbo, Qwen3-4B)를 HF에서 받을 수 있다고 가정한다(Apache-2.0 [검증]). 막히면 TCH-01이 전 사슬을 멈춘다.
5. 학습 데이터에 쓰는 CC0·자체 촬영만으로 gen0이 돈다고 가정한다. 외부 생성기 출력(Soul·GPT)과 Seedance 클립은 학습·보정·회귀에 들어가지 않는다.
6. 모델별 gen0 학습량(§5)은 **제안**이다. 설계 담당(Opus)이 D2에 확정한다.

### 11.2 창업자·오케스트레이터가 정해야 할 것

1. **사전학습 깊이·포즈 모델 금지를 15일 동안 유지하는가?** 기준 문서 규칙이지만 예측 패널이 이번 창에서 가장 비싼 제약이라 했다. 유지하면 MRH가 D4.4 안에 plausible하게 나와야 하고, 실패 시 폴백은 RD-02 고전 평면 깊이라 화면 품질이 크게 떨어진다. 완화하더라도 **시스템 경로가 아니라 평가 전용 기준선**(EVL-04·06, R-EVAL 격리)에 한정하는 것을 권한다.
2. 레인 6개를 형제 세션으로 띄울 수 있는가(§8). 못 띄우면 D15 영상 품질 대신 범위가 줄어야 하는데, 그 순서는 N → M → P 순이고 S는 건드릴 수 없다.
3. Fable 게이트 6회(G0–G5)와 Fable 검토 29개 단위의 예산. 합쳐 약 $50–80 추정(§9).
4. 토큰 지출 한도. 위 추정으로 $400–700이고 미포함 항목이 있다(§9).
5. 입력 24장의 구성(EVL-01). 예측 패널이 실사 안전 칸 승률 0.40, 그림체 풀 0.04로 칸마다 천지 차이다. 구성을 이기는 칸 위주로 짜면 선별 편향이다. 커버리지 매트릭스는 Fable이 적대 검토하고, 데모 릴은 선별했음을 캡션에 적는다.

## 12. 창업자가 해야 하는 일 (코드 단위가 의존하는 것)

| 일 | 마감 | 의존하는 단위 |
|---|---|---|
| 결제, Vast 가입·충전, `VAST_API_KEY`·`HF_TOKEN`을 Network secrets에 | D0 | RUN-01, RUN-03 |
| 입력 24장 선정과 촬영(실사 12 안팎: 미러리스 2K, 간판·글자 있는 실내 포함. 외부 생성기 8 안팎. 그림 4 안팎) | D0 선정 / D1–4 촬영 | EVL-01, EVL-02 |
| 실사 방 30–50개 촬영(MVS용, ArUco/ChArUco 포함, 방당 40–100장) | D1–4 | DAT-08, MDL-06 |
| IMU 산책(폰 4–6대 동시 기록, 오퍼레이터 3인 이상 권장, 40–60 기기-시간) | D1–4 | CAM-06, MDL-17 |
| RAW 매크로 촬영(교차편광, 소재 다양) | D1–4 (가능한 만큼) | DAT-07, MDL-08 |
| Seedance I2V 비교 클립: 24장 각각 같은 사진·같은 무브 문장으로 Higgsfield에서 생성 | D6까지 | EVL-03, PRD-06 |
| 블라인드 패널 모집(지인 20–30명) | D9, D12 | EVL-08 |
| 매일 아침 시트 판정(눈 적응 방지를 위해 D9·D12 패널로 교차 확인) | D7–D15 | EVL-02, EVL-09 |
| T4 몇 시간 임대 허락 | D13 | RUN-08 |
