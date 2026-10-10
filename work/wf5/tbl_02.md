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
