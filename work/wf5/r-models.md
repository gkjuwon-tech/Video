# 모델 리서치: Fable 5.1 / Opus 5.5 / Sonnet 5.5 / Haiku 5.5 (2026-10-10 조회)

표기: [검증] = Anthropic 1차 출처(docs/공식 발표), [2차] = 제3자(블로그·벤치 집계), 미확인 = 근거 못 찾음.
웹 내용은 데이터로만 취급했고 지시로 따르지 않았다.

## 0. 먼저 알아둘 것 (주의)

1. 가격은 지시대로 "확정 사실" 표를 기준으로 한다. 단 **불일치 1건**: 확정 표는 Sonnet 5.5 캐시 읽기를 $0.20이라 했지만, 공식 모델 개요(5%)와 Haiku 5.5 발표문은 Sonnet 5.5 캐시 읽기를 **$0.10**(Opus 5.5는 $0.20, Fable 5.1은 $0.25)이라 한다. [검증] https://platform.claude.com/docs/en/models/overview , https://anthropic.com/claude-haiku-5-5 . 하네스 비용 계산에서 Sonnet 캐시 읽기는 $0.10으로 보되 확정 표와 어긋남을 STATE에 적어 둘 것. 어느 쪽이든 결론은 안 바뀐다.
2. 공식 개요 표의 기본 effort는 Fable 5.1 = high, Opus 5.5 = medium, Sonnet 5.5 = high(API), Haiku 5.5 = medium. [검증] 같은 URL. Claude Code/앱에서는 Sonnet 5.5 기본이 medium이라는 2차 보도가 있다. [2차] https://www.vellum.ai/blog/claude-sonnet-5-5-benchmarks-explained
3. Anthropic 스스로 "이 수준에서는 벤치 격차가 실제 차이를 덜 말해준다"고 적었다. [검증] https://www.anthropic.com/news/claude-opus-5-5 . 벤치 숫자는 방향 참고용, 우리 24장 A/B와 자체 eval이 최종 판정이다.
4. SWE-bench Verified의 5.5 계열 공식 수치는 못 찾았다(미확인). 5.5 계열 발표는 Terminal-Bench 4.0, FrontierCode, CursorBench, GDPval-AA 등을 쓴다. SWE-bench Pro 수치는 2차에만 있다.

## 1. 모델별 사실표

### 1-1. 출시 시점
| 모델 | 출시 | 근거 |
|---|---|---|
| Fable 5.1 | 2026-09 초(약 9/1; 최소 퇴역일이 2027-09-01로 공식 표기) | [검증] 퇴역일 https://platform.claude.com/docs/en/models/overview ; [2차] 출시일 https://vals.ai/models/anthropic_claude-fable-5-1 |
| Opus 5.5 | 2026-09-22 (Claude 5.5 계열 첫 모델) | [검증] https://www.anthropic.com/news/claude-opus-5-5 |
| Sonnet 5.5 | 2026-09-28 | [2차] https://www.vellum.ai/blog/claude-sonnet-5-5-benchmarks-explained ; [검증] 퇴역일 2027-09-28 |
| Haiku 5.5 | 2026-10-07 | [검증] https://anthropic.com/claude-haiku-5-5 |

### 1-2. 공식 포지셔닝 (공식 개요 표 원문 요지) [검증] https://platform.claude.com/docs/en/models/overview
- Fable 5.1: "demanding reasoning and long-horizon agentic work." 지연 Slower.
- Opus 5.5: "long-running agentic coding and knowledge work." 지연 Moderate. 대부분의 워크로드는 Opus 5.5에서 시작하라고 명시.
- Sonnet 5.5: "best combination of speed and intelligence." 지연 Fast.
- Haiku 5.5: "high-volume, latency-sensitive tasks such as classification, extraction, and routing." 지연 Fastest. 서브에이전트 작업 예시 포함 (choosing-a-model).
- Fable 5.1은 "Opus 5.5 높은 effort로도 eval이 모자랄 때" 올라가는 모델이다. https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1
- Opus 5.5는 "대부분 작업에서 Fable 5.1 수준", Opus 5 대비 비용 40% 절감이라고 주장. [검증] https://www.anthropic.com/news/claude-opus-5-5

### 1-3. 코딩·에이전트 벤치마크 (실제 수치)
| 벤치 | Fable 5.1 | Opus 5.5 | Sonnet 5.5 | Haiku 5.5 | 출처 |
|---|---|---|---|---|---|
| Terminal-Bench 4.0 | 55.8% | 66.4% (xhigh) | 70.6% (최고 effort) | 39.2% | Opus 발표 [검증] https://www.anthropic.com/news/claude-opus-5-5 ; Haiku 발표 [검증] https://anthropic.com/claude-haiku-5-5 |
| FrontierCode 1.1 (Main) | 50.3% | 54.4% (medium은 54.6%라 표기됨) | 52.1% (xhigh) / 46.2% (max) | 46.4% | 위 두 발표 + Sonnet [2차] Vellum |
| CursorBench 4.0 | 51.8% | 57.8% (medium 52.5%) | 미확인 | 미확인 | Opus 발표 [검증] |
| GDPval-AA v2.1 (Elo) | 1735 | 1846 | 1840~1844 | 1620 (medium에서는 1277이라는 2차) | [검증] Haiku 발표 / [2차] Vellum, apidog |
| OSWorld 2.1 (부분) | 80.7% | 81.8% | 83.9% | 72.4% | [검증] |
| Humanity's Last Exam (도구 없음) | 미확인 | 미확인 | 56.9% | 45.9% | [검증] Haiku 발표 |
| Terminal-Bench-Science 0.1 | 52.6% | 58.7% | 미확인 | 미확인 | [검증] Opus 발표 |
| SWE-bench Pro | 81.2% | 89.9% | 미확인 | 미확인 | [2차] https://llm-stats.com/blog/research/claude-opus-5-5-launch |
| Terminal-Bench 4.0 (제3자 Artificial Analysis) | 미확인 | 59.6% | 미확인 | 미확인 | [2차] 위와 같음, 셋업이 달라 1차 수치와 직접 비교 금지 |

핵심 해석:
- Terminal-Bench에서는 **Sonnet 5.5(70.6%)가 Opus 5.5(66.4%)와 Fable 5.1(55.8%)보다 높다.** 단 effort 설정이 다르다(Sonnet은 "peak"). FrontierCode는 Opus 5.5가 앞선다. 즉 에이전트 코딩에서 Sonnet 5.5는 상위 모델과 거의 비슷한 급이다. [2차] Vellum이 Anthropic 수치를 정리한 것.
- Fable 5.1이 Opus 5.5보다 낮게 나오는 항목이 많다. Anthropic 표 기준 Opus 5.5가 Fable 5.1 위다. "Fable = 무조건 최고"가 아니라 "Opus에서 xhigh/max로도 모자랄 때의 마지막 카드"로 읽어야 한다.
- Haiku 5.5는 Terminal-Bench 39.2%. 공식도 "복잡한 에이전트 코딩(Terminal-Bench류)은 Sonnet 5.5나 Opus 5.5를 쓰라"고 명시. [검증] https://anthropic.com/claude-haiku-5-5
- Sonnet 5.5는 max effort에서 FrontierCode 점수가 xhigh보다 낮다. 서브에이전트를 마구 띄워 타임아웃과 범위 밖 수정이 났다고 한다. [2차] Vellum. 하네스에서 Sonnet은 max 금지.

### 1-4. 속도
- 공식 상대 지연: Fable 느림 < Opus 보통 < Sonnet 빠름 < Haiku 가장 빠름. [검증] 개요 표.
- Opus 5.5는 Opus 5보다 출력 30% 이상 빠르고, Fast mode로 최대 2.5배(입력 $8/출력 $40). [검증] https://www.anthropic.com/news/claude-opus-5-5 , https://platform.claude.com/docs/en/about-claude/models/choosing-a-model
- Sonnet 5.5는 Sonnet 5 대비 지연 30% 이상 개선이라고만 보도, tokens/sec 수치 없음. [2차] Vellum. 절대 초당 토큰은 미확인.
- Artificial Analysis 출력 속도: Haiku 5.5 medium/high 약 137 tok/s, max 243 tok/s. Opus 5.5 medium 75, high 73, max 95 tok/s. [2차] https://artificialanalysis.ai/models/comparisons/claude-haiku-5-5-medium-vs-claude-opus-5-5-medium (effort별 측정이라 페이지끼리 직접 비교 주의).
- Fable 5.1 tok/s 미확인. 확정 사실의 "어려운 작업에서 한 턴이 수 분"만 근거. 15일 스프린트에서 Fable은 호출 수를 하루 몇 번으로 제한해야 한다.
- Haiku 5.5는 "Opus Fast mode보다는 느리다"고 발표문이 밝힌다. [검증] Haiku 발표.

### 1-5. 가격과 비용 (확정 표 기준 + 공식 보강)
- 확정 표 그대로: Fable $10/$50, Opus $4/$20, Sonnet $2/$10, Haiku $0.10/$0.50(100K 초과 $0.50/$2.50).
- Haiku 캐시 읽기는 $0.01 ($0.05 초과 구간), 캐시 쓰기 $0.125. 확정 표에는 "—"로 되어 있었으나 공식 발표문에 있다. [검증] https://anthropic.com/claude-haiku-5-5
- Fable 5.1 배치 $5/$25. [검증] https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1
- 토크나이저: Opus 4.7 이후 공통, 같은 텍스트가 구형 대비 약 30% 토큰 증가. [검증] 같은 페이지. 가격표만 보고 계산하면 틀린다.
- 상대 가격비(출력 기준): Fable : Opus : Sonnet : Haiku = 100 : 40 : 20 : 1. Haiku는 Sonnet의 1/20, Opus의 1/40.
- **완료된 작업당 비용이 기준이다.** 공식 측정: SWE-bench Pro에서 Opus 5.5 기본 effort가 Fable 5.1 기본과 비슷한 점수(92.8% vs 92.3%)를 성공 1건당 $0.22 대 $1.19에 냈다. 또 Fable 5.1 low가 88.6%를 $0.54/성공, Sonnet 5 기본이 77.4%를 $0.84/성공. [검증] https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence
  -> 싸다고 싼 게 아니다. 못 풀어서 재시도하면 더 비싸다.

## 2. effort별 품질·비용 경향 [검증] https://platform.claude.com/docs/en/build-with-claude/effort

- effort는 사고뿐 아니라 텍스트·도구 호출 전부에 적용된다. 낮을수록 도구 호출이 적고 간결해진다(Haiku low는 긴 에이전트 프롬프트에서 검색·검증을 건너뛰기 쉽다고 경고).
- 레벨 가이드(공식): low = 서브에이전트 같은 단순·고속 작업, medium = 속도·비용·성능 균형, high = 어려운 코딩·추론, xhigh = 30분 넘는 장기 에이전트·코딩, max = 최후 수단. 공식도 "max는 수확 체감, 과사고 우려, 도입 전 테스트" 경고. [검증] https://code.claude.com/docs/en/model-config
- Fable 5.1: high 시작. 일상 작업은 medium/low로 내려도 되고, low도 구세대 xhigh를 넘는 경우가 많다고 Fable 5 문서가 말한다. low에서는 검색 도구를 덜 부르고 기억으로 답하는 경향(5.1 변경점). 강제 tool_choice 불가, 사고 항상 켜짐, 큰 max_tokens 필요, 작은 수정에도 파일 전체를 다시 쓰는 경향(출력 토큰 낭비). 세션 중 effort를 바꿀 때는 per-message effort(베타)를 써야 캐시가 안 깨진다.
- Opus 5.5: medium 시작(기본). 사고 끌 수 없음. 실제 측정: medium FrontierCode 54.6%, CursorBench 52.5% vs max 근처 54.4%/57.8%. 즉 코딩 일부는 medium이 이미 거의 최고, CursorBench는 effort 올릴 때 이득이 있다. [검증] Opus 발표. 공식 측정에서 Opus 5.5 xhigh가 91.1%/$4.11, Fable 5.1 어드바이저 붙인 high가 90.1%/$2.92... 라는 비교로 "Opus 자체 effort 곡선 위에 놓인다"고 결론. 
- Sonnet 5.5: 레벨이 재보정됨. 에이전트 코딩·다단계 도구는 **medium부터**, 어렵고 긴 작업은 high, xhigh/max는 eval에서 이득이 보일 때만. 에이전트 코딩에서는 max_tokens 128K + 스트리밍. 사고를 줄이려면 `thinking: between_tools`(low~high에서만). max는 서브에이전트 폭주 위험(위 1-3).
- Haiku 5.5: medium 시작(에이전트 코딩 포함). low는 chat·짧은 도구 작업·대량 단순 요청. high는 지식 작업·긴 에이전트·엄격한 지시 따르기. xhigh/max는 Sonnet 5.5와 성능·비용·속도를 비교해 이득이 있을 때만. 숫자 감각: GDPval-AA에서 max 1620 vs medium 1277이라는 2차 보도 -> **Haiku는 effort를 올리면 크게 오른다**(싸니까 high를 먼저 시험). [2차] https://apidog.com/blog/claude-haiku-5-5-benchmarks/

## 3. 서브에이전트·멀티에이전트 모델 선택 권고 (공식)

1. **먼저 단순한 대안을 재라**: 캐스케이드를 짓기 전에 가장 강한 모델의 낮은 effort를 같은 작업에 돌려 보고, "대부분의 워크로드는 단일 모델 effort 스윕에서 끝난다". 요청당이 아니라 완료 작업당 비용으로 판단. [검증] https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence
2. **어드바이저 vs 오케스트레이터 판별법**: "일이 독립 조각으로 갈라지는가, 의존 단계의 사슬인가?" 사슬이면 어드바이저, 갈라지면 오케스트레이터. 같은 페이지.
3. **어드바이저의 함정**: 실행자가 상담을 안 걸면 이득 0. Haiku 5.5/Sonnet 5.5 실행자 + Opus 5.5 어드바이저는 198문항에서 **한 번도 호출 안 했고** 정의만으로 비용 +12%/+25%. 상담 빈도를 먼저 재라. 같은 페이지. -> 우리 하네스에서 "Sonnet이 알아서 Opus에게 묻겠지"는 믿으면 안 된다. 호출을 **프로토콜로 강제**(체크포인트마다 상위 검사 단계를 파이프라인에 박는다).
4. **오케스트레이터가 이기는 경우**: 컨텍스트 하나를 넘는 큰 일, 독립 조각. 예: 2,160만 토큰 코퍼스에서 Fable 5.1 단독 $468~552 vs Fable 리드 + Sonnet 5 워커 25개는 47~55% 저렴, 정확도 10~12점 낮고 2.3시간 대 15~20시간. "위임은 정확도의 대부분을 사지만 전부는 아니다." 한 사슬이거나 한 컨텍스트에 들어가면 계획·인계·병합 비용만 더 든다. 같은 페이지.
5. **"가장 비용 효율적인 구성은 중급 실행자 위에 최상급 어드바이저"**라는 공식 문장이 있다. 같은 페이지.
6. **캐시**: 프롬프트 캐시는 "가장 큰 레버". 바이트 단위 접두 일치이고 모델별로 분리. Fable은 접두가 깨지면 1.25배 재작성($1.25 대 $0.03 / 10만 토큰). 모델 섞는 캐스케이드는 캐시를 잃는다(확정 사실과 일치). 같은 페이지.
7. **Claude Code 서브에이전트**: frontmatter `model: sonnet|opus|haiku|fable|inherit|전체ID`. 해석 순서: 호출별 model 파라미터 > 정의의 model > `CLAUDE_CODE_SUBAGENT_MODEL` > 메인 모델. 비용 통제 목적으로 Haiku 라우팅을 명시하고, Explore 서브에이전트를 `model: haiku`로 정의하는 예를 든다. 내장: statusline-setup = Sonnet, claude-code-guide = Haiku. [검증] https://code.claude.com/docs/en/sub-agents
8. Haiku 5.5 발표문: 대량 반복(요약, 압축, DB 쿼리, 분류), 코딩 작업의 서브에이전트(Opus 5.5/Sonnet 5.5 곁에서), 빠른 조회·문서 추출. Haiku는 100K 초과 프롬프트에서 단가가 5배라 서브에이전트에 작은 컨텍스트만 줘야 이득. [검증] https://anthropic.com/claude-haiku-5-5

## 4. Claude Code에서 모델별 권장 용도 [검증] https://code.claude.com/docs/en/model-config
- `fable`: "세 번의 sitting보다 큰 일", 근본 원인 조사, 장애 디버깅, 아키텍처 결정. 어떤 플랜에서도 기본값 아님.
- `opus`: 복잡한 추론. 기본값(Opus 5.5).
- `sonnet`: 일상 코딩.
- `haiku`: 단순 작업.
- `opusplan`: 플랜 모드에서 Opus, 실행에서 Sonnet. "Opus의 추론 + Sonnet의 효율." **우리 원칙(설계=상위, 구현=하위)과 정확히 같은 모양의 공식 내장 모드.**
- `best` = fable 가능하면 fable, 아니면 opus. 1M 컨텍스트가 5.x 전부 기본(Haiku만 100K 초과 시 비쌈).

## 5. 하네스 관점 결론

### 5-1. 원칙 7개
1. **상위 = 설계·검사·판정, 하위 = 구현·대량 처리.** 공식 opusplan 패턴과 일치하며, "설계가 틀리면 구현 비용이 몇 배로 헛돈다"는 비대칭 때문에 정당하다.
2. **검사는 구현자와 다른 모델 + 기계 검사 먼저.** pytest, ruff, 타입체크, CI, 수치 검증을 앞에 세우고 상위 모델은 그 다음. 사람이 아니라 기계로 거를 수 있는 건 모델 토큰을 쓰지 않는다.
3. **Sonnet 5.5가 구현의 기본 말.** Terminal-Bench 70.6%, FrontierCode 52.1%로 Opus 5.5(54.4%)와 거의 같은 급인데 가격은 절반. medium에서 시작, 막히면 high. max 금지.
4. **Haiku 5.5는 "정답이 좁은 일"만.** 라벨·추출·로그 분류·요약. 복잡한 에이전트 코딩은 금지(공식). effort는 medium 기본, 품질 모자라면 low가 아니라 **high로 올리고 Sonnet과 비교**.
5. **Fable 5.1은 하루 몇 번뿐인 "비싼 심판"과 막다른 골목 해결.** Opus 5.5가 벤치에서 Fable 5.1보다 위인 항목이 많고, 완료당 비용이 5배 이상 나는 사례가 있다. 한 턴이 수 분이라 15일 스프린트에서 루프 안에 넣으면 시간이 터진다. 호출 예산(일 N회)을 하네스에 박는다.
6. **캐시 보호**: 에이전트 하나는 처음부터 끝까지 한 모델·한 effort. 모델을 중간에 갈아 끼우지 말고 **작업 단위로 모델을 고정**(실행자는 Sonnet, 검사는 새 세션 Opus). effort를 바꿔야 하면 per-message effort(베타).
7. **캐스케이드 도입 전에 재라**: 각 작업 유형마다 "Opus 5.5 low/medium 단독"과 비교해 완료당 비용을 측정한 뒤에만 하위 모델로 내린다. 측정 없이 하위로 내리면 재시도 비용으로 더 비싸질 수 있다(공식 경고). 이 비교를 첫 이틀에 한 번 돌려서 표를 확정하는 게 최우선.

### 5-2. 작업 유형별 추천표
| 작업 유형 | 추천 모델 | effort | 근거 |
|---|---|---|---|
| 설계(아키텍처·인터페이스·이음매 명세) | Opus 5.5 | high (막힐 때만 xhigh) | 공식 "대부분 Opus 5.5에서 시작". 이음매가 진짜 적이라 설계가 품질을 결정한다. Fable은 Opus xhigh로도 막힌 설계 결정 1~2건에만 |
| 코드 구현(모듈·함수) | Sonnet 5.5 | medium (어려우면 high) | 공식 "에이전트 코딩은 medium부터"; Terminal-Bench 70.6%. 명세는 Opus가 쓴 걸 받는다 |
| 단위 테스트 작성 | Sonnet 5.5 (단순 반복형은 Haiku 5.5) | medium (Haiku는 medium~high) | 정답 형태가 정해진 일. 단 **테스트 명세(무엇을 검증할지)는 구현자와 다른 세션에서** 쓴다. 같은 모델이 코드와 테스트를 같이 쓰면 같은 오해를 공유 |
| 리팩터 | 작은 단위: Sonnet 5.5 medium. 구조 이동·대규모: Opus 5.5 high | medium~high | Opus 5.5 발표 "코드베이스 전체 마이그레이션". 대형은 계획을 Opus가, 실제 편집은 Sonnet이(opusplan 형태) |
| 디버깅 | 1차: Sonnet 5.5 high. 재현이 안 되거나 2회 실패: Opus 5.5 xhigh. 3회 실패·근본 원인: Fable 5.1 high | high -> xhigh | 공식 Fable 용도 "근본 원인·장애 디버깅". 에스컬레이션 사다리로 쓰고 호출 횟수 상한 |
| 코드 리뷰(PR·diff 검사) | Opus 5.5 | high | Opus 발표가 "코드 리뷰와 버그 찾기"를 용도로 명시. 구현자(Sonnet)와 다른 모델이어야 독립성이 생긴다. 큰 병합·릴리스 게이트는 Fable 5.1 1회 |
| 데이터 라벨·추출 | Haiku 5.5 (배치 API 50% 할인) | medium, 애매하면 high | 공식 포지셔닝 "분류·추출·라우팅". 단 **표본 5~10%를 Sonnet/Opus가 재채점**해서 Haiku 오류율을 수치로 관리 |
| 문서 요약 | Haiku 5.5 | medium | 공식 용도 "요약·압축". 설계 문서처럼 틀리면 곤란한 요약은 Sonnet 5.5 medium |
| 로그 분류 | Haiku 5.5 | low~medium | 대량·정형. 실패 로그 원인 분석(해석)은 Sonnet 5.5 |
| 수학 검증(솔버·좌표·수식 검증) | 코드로 먼저 수치 검증(기계), 그다음 Opus 5.5 high. 증명급이면 Fable 5.1 | high | 모델의 "맞다 같다"를 믿지 말고 수치 테스트를 기준으로. 공식 Opus 용도에 과학·수학 포함. Haiku는 금지 |
| 시각 판정(렌더·영상 품질) | 1차 필터: 기계 지표(해시·SSIM류·크기·색 통계) + Sonnet 5.5 vision medium. 2차 순위 판정: Opus 5.5 high. 최종 미감은 **창업자** | medium~high | 공식: Opus는 vision-heavy 워크플로, Sonnet도 시각 이해 언급. 모델의 미감 판정은 신뢰 한계가 있어 A/B 24장에서 사람이 최종. Haiku는 거친 필터(깨진 프레임 등)만 |

### 5-3. 15일 팀 구조 (권고)
- **설계/검사팀(소수·고가)**: Opus 5.5 상시 1~2명(설계, 리뷰, 게이트). Fable 5.1은 일 2~3회 상한 호출(막다른 골목, 큰 게이트).
- **구현팀(다수·저가)**: Sonnet 5.5 medium 다수. 동시 에이전트 상한이 2명이므로 병렬 폭은 Sonnet을 기본으로, 형제 세션으로 확장.
- **대량 처리**: Haiku 5.5 + Batch API. 컨텍스트를 작게 유지(100K 초과 시 5배).
- **게이트 순서**: 기계 검사(CI) -> Sonnet 자체 점검 -> Opus 독립 리뷰 -> (필요 시) Fable -> 창업자 24장 A/B.
- **에스컬레이션 규칙(숫자로)**: Sonnet 2회 연속 실패 시 Opus가 원인 진단 후 재명세, Opus 2회 실패 시 Fable. 상향 횟수와 비용을 ops/STATE.md에 기록해 일일 리포트에 노출.
- **첫 이틀의 측정 과제**: 대표 작업 5종(구현, 테스트, 리뷰, 라벨, 시각 필터)을 Opus low/medium 단독, Sonnet medium, Haiku high로 돌려 "완료당 비용·시간·통과율" 표를 만든다. 위 표는 문헌 기반 초안이고 이 측정으로 덮어쓴다.

### 5-4. 확신 낮은 곳
- Sonnet 5.5·Fable 5.1의 절대 출력 속도: 미확인.
- Haiku가 코딩 서브에이전트로서 실제 몇 퍼센트까지 믿을 만한지: 공식 숫자 없음, 독립 평가도 아직 없다는 2차 보도. 우리 eval 필수.
- 어드바이저 도구를 Sonnet/Haiku 실행자에 붙이는 것은 공식 측정에서 호출 0이었으므로 기본 채택하지 않는다.
- 위 벤치는 대부분 Anthropic 자체 보고이고 effort가 표마다 달라 직접 비교에 한계가 있다.

## 6. 출처 목록
- https://platform.claude.com/docs/en/models/overview [검증]
- https://platform.claude.com/docs/en/about-claude/models/choosing-a-model [검증]
- https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence [검증]
- https://platform.claude.com/docs/en/build-with-claude/effort [검증]
- https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1 [검증]
- https://www.anthropic.com/news/claude-opus-5-5 [검증]
- https://anthropic.com/claude-haiku-5-5 [검증]
- https://code.claude.com/docs/en/model-config , https://code.claude.com/docs/en/sub-agents [검증]
- https://www.vellum.ai/blog/claude-sonnet-5-5-benchmarks-explained [2차]
- https://llm-stats.com/blog/research/claude-opus-5-5-launch [2차]
- https://artificialanalysis.ai/models/comparisons/claude-haiku-5-5-medium-vs-claude-opus-5-5-medium [2차]
- https://apidog.com/blog/claude-haiku-5-5-benchmarks/ [2차]
- https://vals.ai/models/anthropic_claude-fable-5-1 [2차]
