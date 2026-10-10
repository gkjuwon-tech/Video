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
| **MDL-10** | M5 TexelFM native 추론 (WNF 고정 타일 격자·마모장 계산·T8 비트 동일) | `buc/models/texelfm/native.py, wear.py` | MDL-08, FND-03 | 3/3 | ◐ | U+G | 0.5k | S·m | O→O | 0.75 | 6.2 | M | 결정론 검사로 판정 |
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
