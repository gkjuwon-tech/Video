# BUC 15일 스프린트 에이전트 하네스 설계안 — 비용·처리량 우선판 (design-cost)

작성: 하네스 설계 담당(Opus 5.5), 2026-10-10. 입력: `r-models.md`, `r-practice.md`, `r-inventory.md`, `ops/STATE.md`, `ops/RESUME.md`, `ops/persist.sh`.
관점: **같은 품질을 가장 싸고 빠르게.** Haiku·Sonnet이 일의 부피를 맡고, Opus·Fable 호출은 게이트에 몰아 넣는다. 품질 바닥(아래 §0.3)은 비용 때문에 내리지 않는다.

---

## 0. 한 장 요약

### 0.1 결론 일곱 줄
1. **판단은 스크립트가, 부피는 하위 모델이, 서명은 상위 모델이 한다.** 승격·강등·재시도·예산 차단 같은 운영 판단은 결정적 코드(`ops/harness/lib/*.py` + 워크플로 스크립트)에 박는다. 모델은 "무엇을 만들까(Opus)", "만든다(Sonnet/Haiku)", "통과시킬까(기계 → Opus → Fable)"에만 쓴다.
2. **구현의 기본 말은 Sonnet 5.5 medium**, 정답이 좁은 일(스키마·CRUD·스크립트·픽스처·로그 분류·보고 데이터 수집·스쿼드 디스패처)은 **Haiku 5.5**. Opus 5.5는 계약 작성·리뷰·진단·4개 지정 단위 구현. **Fable 5.1은 15일 동안 호출 상한 약 30회**(마일스톤 게이트 6회 + 위험 5 단위 심사 12회 + 막다른 골목 디버깅 8회 + 예비).
3. **검사 순서는 싼 눈 → 비싼 눈.** 기계(GitHub Actions 무료) → Haiku 체크리스트 → Opus 리뷰 → (지정된 것만) Fable → 창업자. 위험 2 이하 카드는 Opus 리뷰를 **표본 1/3**로 줄이되, 표본 밖에서 사고가 나면 그 유형 전체를 20장 동안 전수 리뷰로 되돌린다.
4. **토폴로지: 허브 1 + 구현 스쿼드 4(필요 시 6) + 리뷰 스쿼드 1.** 스쿼드 리더는 Haiku 디스패처(판단은 스크립트가 하므로)다. 리뷰를 한 세션에 모아 같은 접두부로 몰아 처리해 Opus 캐시를 살린다. 스쿼드 수는 "준비된 카드 대기열 길이"로 늘리고 줄인다.
5. **버스는 GitHub(지시·코드, 공개)와 HF 비공개(결과·체크포인트·사람 데이터) 둘로 나눈다.** GPU 기계에는 GitHub 쓰기 토큰을 주지 않는다. 결과는 HF로만 올리고 허브가 지표 숫자만 레포에 옮긴다.
6. **15일 토큰 예산 기준안 약 $650(약 90만원), 하드 상한 $1,000.** 이 중 운영비(허브·틱·스쿼드 리더)가 20–30%를 먹는다. "전부 Opus" 하네스는 같은 일에 약 $1,100–1,500이다. 아낀 돈의 절반 이상은 구현이 아니라 **운영 루프와 캐시**에서 나온다(§6.4). Opus와 Sonnet의 단가 차이는 2배뿐이라 재시도 두 번이면 이득이 사라지므로, 진짜 레버는 시간·Fable 절제·Haiku 부피·캐시다.
7. **D0 반나절 최소 가동**: 카드 1장이 `계약 → 점유 → 구현 → CI → 리뷰 → 병합 → 장부 기록`을 한 바퀴 도는 것. 같은 오후에 **기준선 파일럿**(Opus-low 단독 vs Sonnet-medium+Opus 리뷰 vs Haiku-high+Opus 리뷰)을 돌려 완료 카드당 비용표를 실측으로 덮어쓴다.

### 0.2 입력 사이의 불일치 (설계에 반영한 방식)
| # | 불일치 | 이 설계의 처리 |
|---|---|---|
| 1 | 과제문은 GPU를 **Vast.ai 5090**이라 하지만, 최신 `ops/STATE.md`는 창업자가 **RunPod RTX 4090 커뮤니티 $0.34/h**로 바꿨다고 적었다(Vast가 한국 전화번호 인증 불가). 시크릿도 `RUNPOD_API_KEY`, `HF_TOKEN`. | 러너는 **업체 중립**(`provider_runpod.py` 기본, `provider_vast.py` 예비)으로 짓는다. 스모크 기준은 24GB VRAM(4090)에 맞춘다. 덤: 4090은 sm_89라 5090(sm_120)의 gsplat·flash-attention JIT 지뢰가 대부분 사라진다(인벤토리의 RUN-05·RDR-01 위험 하향). |
| 2 | Sonnet 5.5 캐시 읽기: 확정 표 $0.20 vs 공식 개요·Haiku 발표 $0.10. Haiku 캐시 읽기: 확정 표 "—" vs 공식 $0.01. | 지시대로 **확정 표($0.20)로 계산**하고 Haiku는 공식 $0.01을 쓴다(없는 값을 메우는 것뿐이라 결론 불변). `prices.yaml`에 두 값을 다 적고 실측 청구서로 교정. |
| 3 | 인벤토리는 Fable 검토 단위 29개, 6레인 상시를 제안. | 비용 우선: Fable 단위 심사를 **12개로 압축**(계약 동결 3 + 위험 5·화면 직결 9), 나머지 17개는 Opus high 적대 리뷰 + 기계 검사로 대체. 레인은 **4개로 시작해 대기열이 길 때만 6개**. |
| 4 | 인벤토리 R8(Advisor 패턴)은 "Sonnet이 막히면 Opus 조언자를 부른다". 그러나 공식 측정에서 Sonnet/Haiku 실행자가 조언자를 **한 번도 안 불렀다**(r-models §3). | 자발적 상담에 기대지 않는다. **실패 횟수로 강제 진단 단계를 박는다**(§4). |

### 0.3 품질 바닥 (비용이 아무리 급해도 안 내린다)
1. 임계 사슬 12개 단위(FND-01 → FND-07 → RUN-01 → RUN-05 → TCH-01 → TCH-03 → DAT-02 → MDL-03 → RD-07 → RD-08 → WLD-03 → WLD-08)와 위험 4 이상 단위는 **Opus 리뷰 전수**, 표본 감축 금지.
2. 수용 테스트는 **상위 모델이 구현 전에 쓰고 잠근다.** 구현자는 못 고친다(훅 + CI 이중 차단).
3. "완료"는 말이 아니라 **CI 종료 코드와 재실행된 증거**로만 성립한다.
4. 계약(FND-02·FND-04·PRD-01) 변경은 Opus 작성 + Fable 서명 없이는 병합 불가.
5. 매일 밤 24장 A/B는 예산 초과 상황에서도 **마지막까지 끄지 않는 작업**이다. 반복 횟수가 곧 품질이라는 예측 패널 결론 때문이다.
6. 비밀(키)은 채팅·레포에 절대 없음. 사람 데이터(사진·IMU·방 스캔)는 공개 레포에 절대 없음.

---

## 1. 계층과 역할

### 1.1 네 등급의 성격 (하네스가 쓰는 관점으로만)
| 등급 | 단가(입/출, $/MTok) | 캐시 읽기 | 속도 | 하네스에서의 자리 | 쓰지 말아야 할 곳 |
|---|---|---|---|---|---|
| Fable 5.1 | 10 / 50 | 0.25 | 느림(어려운 턴 수 분) | 마일스톤 심판, 계약 동결 서명, 막다른 골목 근본 원인 | 루프 안쪽 전부. tool_choice 강제 불가라 구조화 출력이 필요한 자동 단계에도 부적합 |
| Opus 5.5 | 4 / 20 | 0.20 | 보통 | 계약(카드) 작성, 리뷰, 테스트 감사, 진단, 체인 캡틴, 지정 4단위 구현 | 대량 구현, 로그 읽기, 보고 문장 다듬기 |
| Sonnet 5.5 | 2 / 10 | 0.20(공식 0.10) | 빠름 | 구현 기본, 1차 디버깅, 시각 1차 판정, 보고서 본문, 조사 | max effort(서브에이전트 폭주·타임아웃 보고), 계약 서명 |
| Haiku 5.5 | 0.10 / 0.50 (100K 초과 0.50 / 2.50) | 0.01 | 가장 빠름 | 완결 스펙 구현, 테스트 보일러플레이트·픽스처, 로그 분류, 체크리스트 리뷰, 스쿼드 디스패처, 데이터 라벨, 틱 데이터 수집 | 복잡한 에이전트 코딩(공식 경고), 100K 넘는 컨텍스트, 수학 검증 |

**비용 감각 한 줄:** 구현 1천 줄을 한 번 시도하는 값(출력 2.5만 + 신규 입력 12.5만 + 캐시 읽기 100만 토큰 가정)이 Haiku $0.035, Sonnet $0.70, Opus $1.20, Fable $2.75다. Sonnet과 Opus는 1.7배 차이뿐이고 Haiku는 20배 싸다. 그래서 **Sonnet↔Opus 경계는 "재시도율"로, Haiku 경계는 "스펙 완결성"으로** 정한다.

### 1.2 역할 × 모델 × effort × 컨텍스트 규칙
| # | 역할 | 기본 모델·effort | 올림 조건 → 올린 값 | 컨텍스트 규칙(입력 상한) | 출력 규칙 |
|---|---|---|---|---|---|
| R01 | 총괄 설계(아키텍트): 계약 v0/v1, 레인 경계, 타임박스 판정 초안 | Opus high | 동결 대상(FND-02·04, PRD-01)은 Fable high 서명 | PARADIGM 발췌 + 인벤토리 + 계약 문서, ≤300K | 계약 파일 + 결정록 1쪽 |
| R02 | 모듈 설계(카드 작성자) | 위험 ≥3: Opus medium / 위험 ≥4·임계 사슬: Opus high | — | 고정 접두(역할 프롬프트·CLAUDE.md·contracts.md) + 해당 서브시스템 문서 발췌, ≤120K | 카드 YAML + 실패 상태 수용 테스트 |
| R02' | 단순 카드 초안(위험 ≤2, 정형 작업) | Sonnet medium 초안 → **Opus low 서명**(G0) | 서명 거부 2회 → Opus medium이 직접 작성 | ≤60K | 카드 YAML |
| R03 | 구현(기본) | Sonnet medium | 난이도 ≥4 또는 1회 실패 후: Sonnet high | 카드 + 허용 파일 + 인터페이스 스텁 + 계약 발췌, **시작 ≤60K**, 루프 중 ≤250K(넘으면 요약 후 새 시도) | 코드 + 단위 테스트 + 증거 파일(`runs/<ID>/a<n>.json`) + 요약 ≤1.5K 토큰 |
| R04 | 구현(완결 스펙) | Haiku medium | 1회 실패 시 Haiku high, 2회 실패 시 Sonnet medium | **≤40K 시작, 절대 100K 미만**(단가 5배 구간 진입 금지, 스크립트가 측정해 차단) | 위와 같음 |
| R05 | 구현(지정·승격) | Opus medium(RDR-03, RD-12, WLD-07, WLD-09는 high) | 2회 실패 → Fable 진단 | ≤300K | 위와 같음 |
| R06 | 수용 테스트 작성 | 카드 작성자와 같은 Opus 호출(한 번에 같이 씀) | — | — | `tests/acceptance/<ID>/` 잠금 |
| R07 | 단위 테스트 작성 | 구현자 본인(같은 세션, 캐시 유지) | — | — | 변경 줄 커버리지 ≥80% |
| R08 | 테스트 보일러플레이트·픽스처·속성 테스트 틀 | Haiku medium | — | ≤40K | 픽스처는 해시 잠금 대상 |
| R09 | 체크리스트 리뷰(1차) | Haiku medium + 기계 신호 | — | diff + 카드 + 기계 리포트, ≤40K | JSON(항목별 pass/fail/의심) |
| R10 | 리뷰(2차, 서명) | Opus medium(위험 ≤3) / Opus high(위험 ≥4) | 의견 분쟁·재리뷰 2회 → Fable(상한 내) | 고정 접두 + diff + 카드 + Haiku 리포트, ≤80K | "정확성·요구사항 위반·범위 이탈·테스트 약함"만. 스타일 금지 |
| R11 | 테스트 감사(G2 상위 감사) | R10과 같은 호출 안에서 | — | 테스트 diff + 변이 점수 | 약한 단언 목록 |
| R12 | 판정(게이트 분쟁·타임박스 채택) | Opus high | 마일스톤·계약 → Fable high | 지표 JSON + 시트 이미지 + 결정 규칙 | 채택/폐기/연장 + 근거 |
| R13 | 조사(웹·논문·라이선스) | Sonnet medium | 의사결정 직결 종합만 Opus medium | 결과 ≤2K 요약 | 출처 등급 표기 |
| R14 | 로그 분류 | **스크립트 사전 필터**(꼬리·중복 제거·오류 시그니처) → Haiku low | 미분류 2회 → Sonnet medium | ≤20K | 범주 + 시그니처 + 담당 레인 |
| R15 | 디버깅 | Sonnet high | 2회 실패 → Opus xhigh 진단, 그다음 Fable high(상한 내) | 재현 명령 + 로그 발췌 | 원인 + 수정 카드 패치 |
| R16 | 시각 1차 판정(깨진 프레임·깜빡임·찢김) | 기계 지표 → Sonnet medium(비전) | — | 클립당 프레임 8장 + 지표 | 클립별 플래그 |
| R17 | 시각 순위 판정(A/B) | Opus high | D7·D12·D14만 Fable high | 접촉 시트 24장 + 지표 + 전일 결과 | 순위·회귀 후보·한 줄 이유 |
| R18 | 보고 작성(한국어, 웃기고 길게) | 데이터 수집 Haiku low + 그래프 스크립트 → 본문 Sonnet medium | 일일 결산만 Opus medium 검수 | 틱 JSON ≤30K | 한국어 본문 + PNG |
| R19 | 스쿼드 디스패처(리더 세션) | **Haiku medium** | 스크립트가 `ANOMALY`를 내면 허브에 send_message(리더가 판단 안 함) | 세션은 매일 새로(컨텍스트 ≤60K 유지) | 워크플로 실행·결과 커밋 |
| R20 | 허브(창업자 대화·병합 열차·체인 캡틴 호출) | Opus medium(세션 모델) | — | 세션 컨텍스트 ≤200K, 넘으면 STATE 갱신 후 새 세션 | 창업자 응답, 결정 기록 |
| R21 | 체인 캡틴(임계 사슬 12개 감시) | 기계 규칙 → 막힘 발견 시에만 Opus medium | — | 보드 JSON ≤20K | 막힘 원인·처방 1개 |
| R22 | 데이터 라벨·캡션·품질 분류 | Haiku medium(API 키 있으면 Batch) | 표본 5–10%를 Sonnet이 재채점, 불일치 >8%면 그 클래스 Sonnet으로 | 항목당 작게 | JSONL |

### 1.3 "구현은 하위, 설계·검사는 상위"의 정확한 경계
**상위(Opus·Fable)만 할 수 있는 것**
- 공개 인터페이스(시그니처, 데이터클래스 필드, 파일 포맷, 단위·좌표 규약) 정의와 변경.
- 수용 테스트·불변식·금지 사항·예산의 작성(= 카드).
- 병합 서명: 위험 ≥3 카드의 리뷰 통과, 계약 변경, 타임박스 채택/폐기, 마일스톤 게이트.

**하위(Sonnet·Haiku)만 하는 것**
- 함수 본문, 내부 헬퍼, 단위 테스트, 모듈 문서, 픽스처 생성 스크립트, CI 설정의 정형 부분.
- 대량 처리: 로그 분류, 라벨, 보고 데이터 수집, 디스패치.

**하위가 해서는 안 되는 것(위반 = F-SCOPE)**
- `contracts.md`·`buc/core/contracts.py`의 공개 타입 변경, `tests/acceptance/**`·`tests/fixtures/**`·`ops/harness/**`·`.github/**` 수정, `pyproject.toml` 의존성 추가(카드에 `deps_allowed`가 있을 때만), 카드의 `allowed_paths` 밖 쓰기.

**예외 다섯 개 (정해진 문으로만 넘는다)**
| 예외 | 조건 | 넘는 방향 | 안전장치 |
|---|---|---|---|
| E1 승격 구현 | 같은 카드에서 하위 2회 실패(§4) 또는 지정 4단위 | 상위가 구현 | 리뷰는 **다른 컨텍스트**의 Opus high 적대 리뷰(위험 5는 Fable) |
| E2 메인 빨간불 핫픽스 | main CI 실패, 병합 동결 상태 | Sonnet이 리뷰 전 병합 | 2시간 안 Opus 사후 리뷰, 실패 시 되돌림 |
| E3 명세 이의 | 하위가 `spec_issue`(모순·불가능) 제출 | 하위 → 상위로 반송 | 하위 실패 횟수에 안 셈. 테스트 우회 대신 멈추라는 정식 출구 |
| E4 단순 카드 초안 | 위험 ≤2 정형 카드 | Sonnet이 카드 초안 | Opus low 서명 없으면 G0 불통 |
| E5 상위의 유휴 | 상위가 놀아도 | **대량 구현 금지** | 상위 유휴는 다음날 카드 선발행(버퍼)으로 돌린다 |

---
## 2. 작업 계약 (Task Contract, "카드")

카드는 **상위가 쓰고 하위가 이행하는 유일한 문서**다. 서브에이전트는 대화를 못 보므로 카드가 곧 전부다. 파일 하나 = 카드 하나 = 기능 하나(diff를 한 문장으로 설명할 수 있는 크기, 대략 0.2–1.2k줄).

### 2.1 파일 위치와 불변성
- 계약: `ops/harness/cards/<ID>.yaml` (main에만 존재, G0 통과 후 **불변**. 고치려면 `contract-change` 라벨 PR을 상위가 연다. 버전은 `rev` 증가)
- 수용 테스트: `tests/acceptance/<ID>/test_*.py` (카드와 같은 PR에 들어가며 잠금 목록 `tests/acceptance/LOCK.sha256`에 등록)
- 상태는 카드에 쓰지 않는다. 상태는 시도 기록·브랜치·PR 라벨에서 **유도**한다(§7). 여러 세션이 같은 파일을 쓰다 충돌하는 일을 원천 차단하기 위해서다.

### 2.2 스키마 (YAML 예시: 실제 FND-07 카드 모양)
```yaml
schema: buc.card/v1
id: FND-07
rev: 1
title: 설정·실험 매니페스트·장부(SQLite)
lane: A                       # 소유 스쿼드
type: crud-store              # 라우팅 통계·강등 판단용 작업 유형
class: S                      # S 슬라이스 / P / M / N / G
critical_chain: true          # 임계 사슬 12개 여부 → 리뷰 표본 감축 금지
risk: 2                       # 1–5 (틀리면 며칠 날리나, 이음매를 깨나)
difficulty: 1                 # 1–5
goal: >
  실험 매니페스트(설정 해시·커밋·시드)와 비용·GPU-h 장부를 SQLite에 기록하고 질의한다.
context_refs:                 # 읽을 것(경로#줄범위). 이것 말고는 읽지 않아도 일이 되게 쓴다
  - docs/contracts.md#L1-L120
  - docs/PARADIGM-v2.md#L880-L930
inputs:
  - "buc.core.contracts.Manifest (FND-02 v0)"
outputs:
  - path: buc/core/config.py
  - path: buc/core/manifest.py
  - path: buc/core/ledger.py
interfaces:                   # 상위가 확정. 하위는 시그니처를 못 바꾼다
  - "def open_ledger(path: Path) -> Ledger"
  - "class Ledger: def record(self, ev: LedgerEvent) -> int; def total(self, kind: str, since: datetime|None=None) -> float"
allowed_paths: ["buc/core/config.py", "buc/core/manifest.py", "buc/core/ledger.py", "tests/unit/core/test_ledger*.py"]
read_paths: ["buc/core/**", "docs/contracts.md"]
forbidden:
  - 수용 테스트·픽스처·CI·ops/harness 수정
  - 새 외부 의존성(표준 라이브러리 sqlite3만)
  - 입력 파일명·해시에 분기하는 코드
  - 예외 삼키기(except: pass), type: ignore 신규
deps_allowed: []
acceptance:
  tests: ["tests/acceptance/FND-07/"]       # 카드 PR 시점에 '빨간불' 확인됨
  commands: ["pytest -q tests/acceptance/FND-07 tests/unit/core"]
  metrics: []                               # GPU·수치 지표가 있으면 임계값
unit_tests:
  min_changed_line_coverage: 0.80
  mutation_min: null                        # 위험 ≥3이면 0.6
  property_tests: ["ledger.total은 record 순서와 무관"]
evidence_required: [commands_with_exit_codes, test_count_before_after, diff_stat]
deps: [FND-01, FND-02]                       # 하드 의존(카드 ID)
deps_soft: []
tier:
  design: opus/medium
  implement: haiku/medium                    # 라우터가 통계로 바꿀 수 있음(§4)
  review: opus/medium
  review_sampling: full                      # full | sample_1_in_3 (critical_chain이면 강제 full)
  fable_gate: false
gpu: {needed: false}
budget:
  usd: 0.60                                  # 구현+리뷰 합계 목표. 3배 넘으면 강제 정지·승격
  max_attempts: 4                            # 사다리 전체 합
  wall_minutes: 90
timebox: null                                # {cut_day: D5, fallback: "RD-02 평면 깊이 스텁"}
contract_version: "contracts@v0"
designer: {model: opus-5.5, run: "wf-design-20261011-01"}
approved_by: {gate: G0, model: opus-5.5/low, at: "2026-10-11T03:12Z"}
```

### 2.3 필드 규칙 (G0에서 기계가 검사)
| 필드 | 규칙 | 위반 시 |
|---|---|---|
| `id`·`deps` | 의존 그래프 순환 없음, 존재하는 카드만 | G0 불통 |
| `allowed_paths` | 진행 중인 다른 카드와 **쓰기 경로 겹침 0** | G0 불통(레인 간 충돌 예방의 핵심) |
| `interfaces` | 파이썬 시그니처로 파싱 가능, `contracts.py`와 타입명 일치 | G0 불통 |
| `acceptance.tests` | 존재하고 **현재 실패(빨간불)** 해야 함. 이미 초록이면 테스트가 아무것도 안 잰다는 뜻 | G0 불통 |
| `budget.usd` | `kLOC × 등급 단가 × 2.5 + 리뷰 몫` 공식의 0.5–2배 범위 | 경고(허브 보드에 표시) |
| `tier.implement` | Haiku는 R2 조건(완결 스펙·새 알고리즘 없음·컨텍스트 ≤40K 추정)을 만족해야 함 | Sonnet으로 자동 상향 |
| `critical_chain` / `risk ≥4` | `review_sampling`은 반드시 `full` | 강제 교정 |
| `forbidden` | 표준 4항(테스트·CI·하네스 수정 금지, 의존성, 입력 분기, 예외 삼키기) 포함 | 자동 추가 |

### 2.4 시도 기록 스키마 (`ops/harness/runs/<ID>/a<n>.json`, 시도마다 새 파일)
```json
{"schema":"buc.attempt/v1","card":"FND-07","card_rev":1,"attempt":2,
 "model":"sonnet-5.5","effort":"medium","squad":"A","session":"session_…","workflow_run":"…",
 "branch":"w/A/FND-07-a2","head":"3f2c…","started":"…","ended":"…",
 "tokens":{"in":182000,"cache_read":940000,"cache_write":61000,"out":27000},"usd":0.71,
 "gates":{"G1":"pass","G2":"fail","G3":null},
 "failure":{"type":"F-TEST","signature":"sha1:9be1…","summary":"ledger.total 시간대 경계"},
 "claims":{"tests_passed":true},"verified":{"tests_passed":false},
 "evidence":["runs/FND-07/a2.log.txt"]}
```
`claims`(구현자가 말한 것)와 `verified`(하네스가 재실행해 본 것)를 나란히 둔다. 둘이 다르면 **F-FAKE**다(§4.3).

### 2.5 하위에게 주는 프롬프트 골격 (캐시용 고정 접두 순서)
1. `ops/harness/prompts/implementer.md` (역할·금지·증거 형식·`spec_issue` 출구) — **바이트 고정**
2. `CLAUDE.md` 핵심(≤150줄) — 고정
3. `docs/contracts.md` 현재 동결본 — 하루 1회 이하로만 바뀜
4. ── 여기까지 공통 접두 ──
5. 카드 YAML, 허용 파일 현재 내용, 직전 시도의 실패 요약(있으면)

같은 등급·같은 역할의 에이전트는 1–4가 바이트 단위로 같아야 캐시가 붙는다. 그래서 **역할 프롬프트 파일을 하루 중간에 고치지 않는다**(고칠 일은 모아서 밤에).

---

## 3. 게이트 G0–G6

게이트는 **싼 순서**로 선다. 기계로 거를 수 있는 것에 모델 토큰을 쓰지 않는다. 각 게이트는 결과를 PR 라벨(`g1-ok` 등)과 시도 기록에 남긴다.

### 3.1 한눈에
| 게이트 | 무엇을 | 누가(모델) | 통과 기준 | 실패 시 | 1회 비용 감각 |
|---|---|---|---|---|---|
| G0 계약 검토 | 카드 + 수용 테스트 | 기계(스키마·그래프·빨간불) → Haiku 일관성 대조 → 위험 ≥4·임계 사슬은 다른 Opus 호출의 "모호성 찾기", 동결 계약은 Fable | §2.3 전부 + 모호성 0건 | 작성자에게 반송, 2회 넘으면 허브 | 기계 $0, Haiku $0.005, Opus $0.2, Fable $3 |
| G1 정적 검사 | lint·type·경계·비밀 | 기계(GitHub Actions + 로컬 훅) | ruff·pyright 오류 0, import-linter 계층 위반 0, gitleaks 0, 금지 패턴 신규 0 | 같은 세션에서 구현자가 수정(시도 횟수 미차감). 같은 G1 오류 3연속이면 시도 1회 실패로 계상 | $0 |
| G2 단위 테스트 | 수용 + 단위 테스트, 테스트 조작 | 기계 → Opus 테스트 감사(리뷰 안에서) | 전부 통과, 변경 줄 커버리지 ≥80%, 위험 ≥3은 변이 점수 ≥0.6, 테스트 수 비감소, skip/xfail 신규 0, 잠금 파일 해시 일치 | 사다리(§4) | 기계 $0, 감사는 리뷰 몫 |
| G3 통합·CI | main 위에서 전체 | 기계(GitHub Actions, 병합 열차) | 전체 CPU 테스트, 계약 왕복 테스트, **걷는 해골 초소형 종단 실행**(64×64, 8프레임 → mp4), 계보 린트 | 충돌은 소유 스쿼드가 리베이스, main 빨간불이면 병합 동결 + 핫픽스(E2) | $0 |
| G4 GPU 스모크 | GPU가 필요한 카드 | 기계(러너가 실행, 임계값 비교) → 실패 로그는 Haiku 분류 | 종료 0, NaN 0, 최대 VRAM ≤21GB(4090 기준), 시간 ≤ 예산×1.5, 결정론 커널은 재실행 해시 일치, 지표가 마지막 정상값 대비 허용폭 안 | Sonnet 수정, 반복 시 사다리 | GPU 몇 분 ≈ $0.02–0.05 |
| G5 시각·A/B | 매일 밤 24장 | 기계 지표 → Sonnet 비전 1차 플래그 → Opus high 순위 판정 (D7·D12·D14는 Fable) | 24장 중 6장 이상에서 주 지표 5% 넘게 악화 없음 + 새 "깨짐" 플래그 0 | 회귀 후보 PR 표시 → 아침 창업자 판정 → 되돌림 또는 유지 | 밤당 약 $1–2 |
| G6 창업자 | 아침 시트, D9·D12 패널, 최종 | 사람 | 창업자 판정이 지표를 이긴다 | 창업자가 고른 "가장 큰 아티팩트"가 그날 최우선 카드가 된다 | 창업자 하루 ≤30분 |

