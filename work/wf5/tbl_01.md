
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
