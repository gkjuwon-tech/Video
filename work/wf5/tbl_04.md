
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

고정 입력 24장 매일 A/B가 최대 레버다. 평가 모델은 R-EVAL 방화벽 뒤에 둔다. 단위 15개, 코드 9.1k줄, 10.3 에이전트-일.

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
| **EVL-15** | 자기 대국 왕복 회귀 (엔진 세계→입력 한 장→파이프라인→정답 새 시점 비교) | `eval/self_play.py` | DAT-09, PRD-01, EVL-04 | 3/2 | ◐ | G+V | 0.6k | S·m | O→O | 0.75 | 5.0 | M | 정답 세계가 있어 발명 영역의 오차를 직접 잰다. 우리 렌더러 룩 과적합은 실사 쌍과 병행해 감시 |

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
