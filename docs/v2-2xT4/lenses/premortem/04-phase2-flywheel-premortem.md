> **출처 메모.** 2×T4 + 수십M from-scratch 제약으로 다시 돌린 설계 워크숍에서 [Phase 2 + 저예산 플라이휠] 렌즈 담당이 쓴 건설적 프리모템과 개정안이다. 최종 결정은 [`docs/PARADIGM-v2.md`](../../../PARADIGM-v2.md)를 따른다. 이 문서는 Higgsfield Soul 하이브리드 결정 **이전**에 작성됐다.

---

# Phase 2 "작은 뇌, 정확한 우주, 큰 탐색" (2×T4판) 건설적 프리모템

> **대상.** `phase2-flywheel-shoestring` 제안서 v2-T4 전문. 스크래치패드에 있는 FULL본 기준이다.
> **기준.** `docs/PARADIGM.md` §0.5·§1·§2·§4·§7·§8·§10, T4 리서치 보고서(2026-10-10), 병행 렌즈 두 개(compute-budget-solver, tiny-foundation-models).
> **표기.**
> - 출처: [검증] 1차 출처 / [2차] 2차 출처 / [추정] 내 계산
> - 신규성: [신규] 이 제약에서 나온 아이디어 / [조합] 기존 부품의 새 결합 / [부품] 차용
> - 판정: **맞음 / 보정 / 틀림**
>
> **결론.**
> - 뼈대는 옳다. 물리는 공짜 검증기, 사람은 저대역 소유자, 이미지 교사는 정지 순간의 판사라는 구조다.
> - 깨지는 곳은 다섯 군데다.
>   1. 교사 단가와 정체성
>   2. 검증 비용
>   3. 시간 증인의 식별 가능성
>   4. 실현 처리량
>   5. 데이터 계보
> - 다섯 곳 모두 후퇴가 아니라 업그레이드로 고친다. 목표는 하나도 낮추지 않는다.

---

# Part 1. 숫자 재검증

## 1.0 가장 큰 교정 다섯 개

**1. 교사 단가가 4–5배 틀렸다.**
- 제안서는 SDXL급 @512² 0.2 s, 즉 18k TC/T4-h로 전체 회계를 짰다.
- 이미지 모델은 하나만 빌릴 수 있다. Phase 1 렌즈들은 Z-Image 계열(6.15B, Apache-2.0)로 수렴하고 있으니 Phase 2도 같은 교사를 써야 한다.
- SDXL은 1024² 네이티브라서, SDXL을 쓴다 해도 512² 판정은 OOD다.
- Z-Image @512² fp16은 forward 1회 1.57×10¹³ FLOPs다. T4에서 0.8–1.05 s, 즉 3.4–4.6k TC/T4-h다.
- INT8은 5.5–8k TC/T4-h로 추정하며 검증이 필요하다.

**2. Coverage–Precision에서 검증 비용 $c_{ver}$이 빠졌다.**
- best-of-4096 생성 자체는 21 s다(3.2×10¹⁴ FLOPs).
- 하지만 4096개를 MJPC로 전수 검증하면 4096 × 1.5–6분 = **100–400 T4-h**다.

**3. 실현 처리량을 3배 과소평가했다.**
- "블록당 80 s"는 Chaos Bank warm start가 이미 있다는 전제의 값이다.
- Day-1 혼합(소 60%, 중 30%, 대 10%)은 가중평균 **231 s**다.
- 그래서 1회전은 1,000블록이 아니라 약 **350블록/일**이다.

**4. F_φ-S 스텝 비용을 30–50배 과소평가했다.**
- 4M GNS(message passing 10회, latent 128, 이웃 20)는 1만 노드에서 스텝당 2.6×10¹¹ FLOPs다.
- 26–52 ms이지 1 ms가 아니다.
- SMC 루프 안에 넣으면 1024 입자 기준으로 스텝마다 30–50 s가 든다.

**5. FLIP의 dt가 문서 안에서 두 개로 섞였다.**
- "시뮬 1초에 0.5–3분"은 CFL 적응 dt ≈ 1–2 ms에서만 성립한다.
- 와인잔 계산은 dt 0.2 ms 고정을 썼다. 이 경우 top-8은 **20–50분**이다(문서값 10–25분).
- CFL 적응을 쓰면 **2–12분**이다.

## 1.1 전체 대조표

| # | 위치 | 제안서 값 | 판정 | 교정값과 근거 |
|---|---|---|---|---|
| 1 | §0-1 | 교사의 선불 컴퓨트: "SD v1 7×10²², 1년 예산의 70배" | 보정 | SD v1 산식(A100 15만 시간)은 맞지만 우리 교사가 아니다. Z-Image는 **3.4–4.5×10²³**이다(314K H800-h, MFU 30–40% 가정). 1년 예산 1.0×10²¹의 **340–450배**다 |
| 2 | §0-1, §5.1 | 학습 파라미터 125M, 1세대 287 T4-h(6일) | 보정 | F_φ-S를 4M→0.6M로 줄이고 PercepNet-S 3M을 추가해 **123M**이다. 소형 모델 실효를 12 TF로, F_φ와 Reader의 낮은 MFU, Z-Image 단가를 반영하면 **약 320 T4-h(6.7일)**이다. SR-LoRA가 필요하면 60–115를 더한다 |
| 3 | §0-3, P1 | best-of-4096 20–40 s | 보정 | 생성만 보면 맞다(15 TF에서 21 s). 검증 비용이 빠졌다. MJPC 전수 검증은 100–400 T4-h, 캐스케이드를 쓰면 6–18분(Part 2 F2) |
| 4 | P1 | $P_{hit}(4096,q{=}10^{-3})\approx98\%$ | 맞음 | 98.3%다. 단 스크리닝 재현율 $r$을 곱해야 한다: $1-(1-qr)^N$ |
| 5 | §0-4, §4.3 | BIS: (K,Δ) 3–4조합으로 0.3–15 Hz | 맞음(조건부) | K≤12일 때 Δ = 1/30, 1/5, 1.2 s 세 조합으로 0.07–15 Hz를 덮는다. 단 g 스케일 왜곡과 균일 시간 스케일 왜곡은 원리상 식별할 수 없다(F3) |
| 6 | §0-7 | T4/H100: 텐서 1/25, 대역 1/10 | 맞음 | H100 SXM 3.35 TB/s 기준이다. PCIe H100이면 대역은 1/6 |
| 7 | §1.1 | 대형 GEMM 25–45 TF | 보정 | [검증] HGEMM 실측 최고 41.6 TF(클럭 1590 MHz 고정). 지속 부하에서는 70W 쓰로틀로 더 낮다 |
| 8 | §1.1 | 학습 실효 계획값 16 TF | 보정 | d≥512인 큰 GEMM은 16 TF로 둔다. **50M 미만·d≤448 모델은 12 TF**로 둔다. T4 roofline 능선은 203 FLOP/B이고, d=384 정방 GEMM의 연산 강도는 약 192로 경계에 걸린다 |
| 9 | §1.1 | 48 T4-h/일 = 2.8×10¹⁸, 90일 2.5×10²⁰, 1년 1.0×10²¹ | 맞음 | 16 TF 기준이다. 12 TF면 ×0.75 |
| 10 | §1.1 | v1 대비 1/600–1/1,200, 2c 누계 대비 1/3,600–1/7,200 | 맞음 | — |
| 11 | §1.1 | Kaggle 주 60 T4-h(상시의 18%) | 보정 | T4×2가 쿼터를 1배로 소모할 때만 성립한다. 2배로 소모하면 30 T4-h(9%)다. vCPU 수도 미확인이다(2–4) |
| 12 | §1.1 | 320 GB/s | 맞음 | datasheet마다 300과 320으로 다르다. ECC를 켜면 가용 메모리는 약 15 GB다 |
| 13 | §1.1 | INT8 130 TOPS | 보정 | 피크값이다. 실측은 cuBLASLt 74.9 TOPS, INT4 CUTLASS 114 TOPS |
| 14 | §1.1 | **TC = 0.2 s @512²(SDXL), 18k TC/T4-h** | **틀림** | 교사를 Phase 1과 같은 Z-Image로 통일하면 @512² fp16 0.8–1.05 s, 3.4–4.6k TC/T4-h다. INT8은 5.5–8k(검증 필요). SDXL이어도 512²는 OOD다 |
| 15 | §1.1 | CFG를 쓰면 2 TC | 보정 | 판정은 조건부 단일 경로로 하므로 1 TC다. Turbo는 CFG를 쓰지 않는다. CFG가 필요한 곳은 DMD real score(2 TC)뿐이다 |
| 16 | §1.1 | 1024²는 ×4–5 | 맞음 | Z-Image는 ×3.9(6.1×10¹³ FLOPs, 3.1–4.1 s) |
| 17 | §1.1 | DMD 1스텝 6–8 TC-eq | 맞음 | real score(CFG 2) + fake score 1 + fake score 학습 fwd/bwd 3 + VAE encoder fwd/bwd ≈ 7 |
| 18 | §1.2 | Phase 2 1년차 약 6,000 T4-h | 맞음 | 기간별 몫 12.5%, 40%, 55%를 적용하면 6,460 |
| 19 | §1.2 | TC에 연 2,500–3,000 T4-h | 보정 | SDXL 단가로는 45–54M TC였다. Z-Image fp16/INT8 혼합(평균 약 5k/T4-h)이면 **연 13–15M TC**다 |
| 20 | §1.4 | LoRA: fake-score r32 23M, Chrono r16 12M, identity r8 6M | 보정 | Z-Image(30층, d=3840)의 attention q/k/v/o 기준으로 각각 29.5M, 14.7M, 7.4M이다. 모두 교사의 0.5% 이하 |
| 21 | §1.4, §7 | 텍스트 인코더 CLIP-L+bigG 1.6 GB | 틀림 | Z-Image의 텍스트 인코더는 Qwen3-4B다. fp16 8 GB, INT8 약 4 GB, 임베딩 폭 2560 |
| 22 | §4.3 | 크로노 판정 1회 60 TC, 12 s | 보정 | 조건부 단일 경로라 30 TC다. Z-Image INT8 13–18 s, fp16 23–30 s |
| 23 | §4.3 | Chrono-LoRA: 3만 장, 10 T4-h | 틀림 | Z-Image에서 3만 장 × 3 epoch fwd/bwd는 27만 TC-eq, 약 64 T4-h다. 2만 샘플 1 pass로 줄이면 15–25 T4-h |
| 24 | §4.4 | Gold 사건 시각 ±2 ms | 보정 | 240fps(프레임 간격 4.17 ms) + 블러 + 오디오로 가능하다. 단 장치별 AV 오프셋(최대 수십 ms)과 rolling shutter 행 지연(2–4 ms)을 보정해야 한다. Sensor Purity Test가 필요하다(Part 3) |
| 25 | §5.1 | EPD-S 6.6×10¹⁷, 11 T4-h, 1024개 2–5 s | 맞음 | 12 TF면 15 T4-h. 1024개는 약 4 s |
| 26 | §5.1 | MoPri-S 1.9×10¹⁸, 33 T4-h | 맞음 | 12 TF면 44 T4-h. 학습 토큰은 MDM(약 6.3×10⁹ frame-token)과 같은 규모다 |
| 27 | §5.1 | HeroProxy 4096장 INT8 1.5 s | 보정 | 추론은 맞다(0.7–1.5 s). 4096장을 256²로 렌더하는 데 약 16 s(장당 4 ms)가 따로 든다 |
| 28 | §5.1 | **MAN-S 96만 TC, 53 T4-h** | **틀림** | Z-Image fp16이면 200–267 T4-h다. Exhaust-to-Shader로 사전학습하고 DMD를 5k 스텝만 돌리면 약 41 T4-h |
| 29 | §5.1 | MAN-S 4K 1장 0.3 s(INT8) | 맞음 | 장당 약 1.7 TFLOPs이고 대역폭을 포함하면 0.1–0.3 s |
| 30 | §5.1 | **F_φ-S 4M, 1 ms/스텝(1만 노드), 15 T4-h** | **틀림** | 실제는 26–52 ms/스텝이다. scatter/gather 때문에 MFU가 낮아 학습도 40–80 T4-h가 든다. 재설계안(0.6M, MP 3, latent 64, k=8)은 1–2.6 ms/스텝, 약 8 T4-h |
| 31 | §5.1 | Reader-S 6×10¹⁸ FLOPs, 100–115 T4-h | 보정 | 6ND 공식은 RAFT식 상관 볼륨이나 HRNet식 다해상도 구조에 맞지 않는다. FLOPs는 2–3×10¹⁸이지만 MFU가 낮다. R-Kpt를 ImageNet 초기화 없이 학습하면 스케줄이 2–3배 늘어서 **115–150 T4-h** |
| 32 | §5.1 | Denoiser-S 렌더 22 + 학습 8 | 맞음 | T4 OptiX 경로추적을 40–100M path-sample/s로 가정하면 512²×1024 spp가 장당 2.7–6.7 s, 2만 장에 15–37 h다. 유리와 액체는 ×2. 반드시 Cycles 내장 디노이저를 끈다 |
| 33 | §5.1 | TTS-S + Voc-S 120–240 T4-h | 보정 | GAN 보코더는 스케줄이 길어서 200–400 T4-h |
| 34 | §6.1 | 휴머노이드 MJX(T4) 1–3×10⁵ step/s | 보정 | 접촉을 포함하면 3×10⁴–10⁵(추정)이다. MJX(JAX)가 sm_75를 지원하는지는 문서에 없다. MuJoCo Warp를 우선한다(Warp CUDA 13 빌드는 sm_75 지원 [검증]) |
| 35 | §6.1 | FLIP 스텝 60–150 ms → 시뮬 1초에 0.5–3분 | 보정 | dt 1–2 ms(CFL 적응)에서만 성립한다. dt 0.2 ms 고정이면 5–12.5분 |
| 36 | §6.2 | **와인잔 FLIP top-8 10–25분** | **틀림** | dt 0.2 ms 고정이면 8 × 2,500스텝으로 20–50분이다. CFL 적응을 쓰면 2–12분 |
| 37 | §6.2 | 도미노 50(N=1024) 3–12분, 공(N=4096) 20–60 s | 맞음 | 재계산 결과 1.54×10⁷, 1.84×10⁷ scene-step으로 일치 |
| 38 | §6.2 | MJPC 접촉창(10 s 모션) 1.5–6분 | 보정 | 휴머노이드 처리량을 보정하면 2.6–8.5분이다. MJPC-lite(rollout 128, 레퍼런스 warm start)면 0.6–2.1분 |
| 39 | §6.4 | Chaos Bank 10⁶ rollout, 64 MB | 맞음 | 소형 템플릿 10⁶개 구축은 3×10⁹ step이다. CPU 16코어(10⁵ step/s)로 약 8 h |
| 40 | §7 | GPU-A 12–13 GB | 틀림(교사 교체 시) | Z-Image fp16은 12.3 GB이고 활성을 더하면 15 GB를 넘는다. INT8 판정(6.2 GB)과 단계별 가중치 스왑으로 재배치한다 |
| 41 | §7 | PCIe allreduce 64 MB 10–25 ms | 맞음 | P2P가 없으면 호스트를 경유하므로 상단값이다 |
| 42 | §9 | 2a 샷당 증분 3–6 T4-h | 보정 | 교사 단가를 반영하면 3.5–7 T4-h |
| 43 | §10.1 | **실현 블록당 80 s, 1회전 1일** | **틀림(Day-1)** | 뱅크가 없는 Day-1 가중평균은 231 s, 하루 350블록이다. 뱅크로 스텝을 5배 줄이면 65 s, 하루 1,000블록 |
| 44 | §10.1 | L3 12.8만 TC 7 h, BSRT 8만 TC 4.5 h | 틀림 | Z-Image 기준 L3(블록당 64 TC)는 INT8 8.9 h, fp16 15 h다. BSRT 1만 장 × 4 TC는 fp16 9.4 h → 하루 4k장으로 줄여 3.8 h |
| 45 | §10.2 | 10회전 후 하루 8,000블록 | 보정 | Z-Image 단가에서는 L3 커버리지를 25% 이하로 게이팅해야 성립한다. 아니면 4,000–6,000 |
| 46 | §14 | 90일 430 T4-h | 맞음 | 항목 합계는 398이다. 개정안은 약 470 |
| 47 | §16 R5 | "교사를 SANA급으로" | 틀림 | Phase 1이 교사를 바꾸지 않는 한 단일 교사 규칙 위반이다. SANA는 라이선스 출처도 상충한다. 삭제한다 |

---

# Part 2. 가장 실패할 7지점과 업그레이드

## F1. 단일 교사 가격 충격과 "Turbo는 판사가 아니다"

**증상.** 플라이휠 1회전이 1.3–2일로 늘어난다. DR(dream residual) 점수의 분산이 커서 proxy와 교사의 Spearman 상관이 0.5에서 오르지 않는다.

**기술적 원인.**
- (i) 회계가 SDXL @512²로 잡혀 있다.
- (ii) 512²는 SDXL과 Z-Image 모두 네이티브(약 1MP) 밖일 가능성이 있다. 판정 노이즈가 커진다.
- (iii) Z-Image-Turbo는 DMD 계열로 증류된 샘플러다.
  - $\hat x_0$가 사후평균 $\mathbb E[x_0\mid x_t]$가 아니라 모드 근방의 샘플이다.
  - 그래서 DR은 가능도가 아니라 샘플 거리가 되고, DMD의 real score 역할도 못 한다.
  - 임의 σ에서 SDEdit을 하면 학습된 t가 아니어서 불안정하다.

**업그레이드.**
- (a) **One-Teacher, Two-Mode** [조합]
  - Z-Image Base(비증류, 2026-01 공개, Apache-2.0 [2차])를 판정·score 경로로 쓴다.
  - Z-Image-Turbo를 샘플 경로로 쓴다.
  - 둘은 아키텍처, VAE, 텍스트 인코더를 공유하는 한 계열이다. 이것을 "이미지 모델 하나"로 해석해 달라고 창업자에게 판결을 요청한다(Q0).
  - 판결이 부정이면 Turbo 하나에 **Score-Restoration LoRA(SR-LoRA)**를 붙인다.
    - r32, attention + MLP, 약 70M(교사의 1.1%).
    - 교사 자기 샘플 3만 장과 자체 실사진으로 σ 그리드 전 구간에서 flow-matching 디노이징 능력을 복원한다.
    - 60–115 T4-h가 들고 Phase 1과 공유한다.
- (b) **TC 가격표 매개변수화.** 모든 예산을 $c_{TC}$(모델, 해상도, 정밀도)의 함수로 적는다. 값은 E1에서 확정한다.
- (c) **판정 정밀도 게이트.** INT8/INT4 판정은 1024² fp16 판정과의 Spearman ≥0.9일 때만 쓴다.
- (d) **Feature-Tap Bus** (tiny 렌즈와 공유). 판정 forward 1회에서 DR과 중간 특징을 함께 로그한다. proxy와 PercepNet-S의 학습 데이터가 되므로 TC 1회당 정보량이 늘어난다.
- (e) **불확실성 게이트 L3.** proxy 앙상블 분산 상위 25%와 무작위 10%만 교사로 보낸다.

## F2. 검증 비용 누락: "작은 생성기 × 큰 탐색"은 검증도 쌀 때만 성립한다

**증상.** best-of-N을 키울수록 MJPC가 병목이 된다. N=4096을 전수 검증하면 100–400 T4-h다.

**원인.**
- 비용식은 $C=N(c_{gen}+c_{ver})$다.
- $c_{ver}$(MJPC) / $c_{gen}$(5 ms) ≈ 10⁴–10⁵이다. 제안서의 경제학은 $c_{ver}$를 0으로 둔 셈이다.

**업그레이드: ID-Screen 검증 캐스케이드** [조합]

| 단계 | 내용 | 대상 | 비용 |
|---|---|---|---|
| L0 | 운동학 불변량: 발 미끄럼, 관통, 관절 한계, 각속도 상한. GPU 배치 | 4096개 | <1 s |
| L1 | **역동역학 유령력(inverse dynamics ghost force).** `mj_inverse` + 프레임별 접촉력 QP로, 접촉으로 설명되지 않는 루트 렌치를 구한다. RGF $p_{real}$ 분위수로 순위를 매긴다 | 4096개 × 300프레임 | 프레임당 50–100 µs, CPU 16코어로 **4–8 s** |
| L2 | MJPC-lite: rollout 128, 레퍼런스 warm start | 상위 8개 | 개당 0.6–2.1분 |
| L3 | TPRF 교사 판정(§5.2) | 상위 4개 | 8–16 TC |
| **합계** | | | **10 s 클립 하나에 6–18분** (T4 1장 + CPU) |

정식화도 고친다.

$$P_{hit}=1-(1-q\,r)^N,\qquad C=N(c_{gen}+c_0+c_1)+N_2c_{MJPC}+N_3c_{TC}$$

여기서 $r$은 L0·L1 스크린이 분지(basin) 구성원을 놓치지 않는 재현율이다.

## F3. 크로노 증인의 게이지 비식별성

**증상.** P2-X1의 "g 스케일" 팔이 AUC ≈ 0.5에서 움직이지 않는다. 혹은 0.5를 넘는데, 그 이유가 블러 길이나 노출 개수 같은 가짜 단서다.

**원인: 차원 해석.**
- 정지 사진은 미터 스케일 $s$도, 스트로브 간격 $\Delta$도 모른다.
- 탄도 궤적의 이미지는 $(x_0,\ s\,v\Delta,\ s\,g\Delta^2)$에만 의존한다.
- 그래서 g를 κ배 하는 것은 $\Delta\to\Delta/\sqrt\kappa$로 정확히 흡수된다. 균일 시간 스케일 왜곡도 마찬가지다.
- 여기에 교사가 "보기 좋은 등간격"을 선호하는 미감 편향이 겹친다.

**업그레이드: GICC (Gauge-Invariant Chrono Contrast)** [신규]
- 왜곡 가족을 유사변환군 $G=\{(s,\lambda)\}$의 궤도와 직교하게 제한한다.
- **채점용 $\mathcal W_{inv}$:**
  - 시간 역전(소산계에서)
  - 구간별 ease-in-out 재매개화. 2차 차분이 일정하다는 성질을 깨뜨린다. 비디오 모델의 병이 바로 이것이다
  - 반발 결합 파괴: 높이 비 ≠ (간격 비)²
  - 진자 등시성 파괴
  - 보행 duty factor 이동
- **음성 대조 $\mathcal W_{ctrl}$:** g×κ, 균일 λ. AUC 0.5±0.05가 나와야 정상이다.
- **소유권 재배정.**
  - "시간의 눈금"(절대 템포, g)은 I4, 오디오 간격 증인, RGF가 소유한다.
  - "시간의 모양"은 크로노 증인이 소유한다.
- 덤: 비디오 모델이 실제로 지는 곳이 "모양"(ease-in-out, 포물선이 아닌 부유)이다. 그래서 판사가 경쟁자의 결함을 정확히 겨냥하게 된다.

## F4. 실현 처리량

**증상.** Day-1 하루 생산량이 목표의 35%에 그친다.

**원인.**
- 블록당 80 s는 뱅크가 있다는 전제였다.
- FLIP dt를 혼용했다.
- F_φ-S 비용을 1/30로 잡았다.
- MJX의 sm_75 지원이 불명이다.
- 휴머노이드 처리량을 과대평가했다.

**업그레이드.**
- (a) **Bank-First 시드기.**
  - 0–4주에 놀고 있는 CPU 야간 시간으로 템플릿 5종 × 10⁵–10⁶ rollout을 굴린다. 소형 10⁶개에 약 8 h다.
  - 1회전 규모를 350 → 1,000으로 래칫한다.
- (b) CFL 적응 FLIP(CFL≈3)과 multi-fidelity를 같이 쓴다. 와인잔 top-8이 2–12분이 된다.
- (c) **F_φ-S를 0.6M으로 줄이고 SMC 루프 밖으로 뺀다.**
  - 최종 top-k 실현에만 쓰고, 100 Hz 제어율로 적용한다.
  - SMC 루프 안에서는 PCGD-0만 쓴다.
- (d) 백엔드: CPU MuJoCo(결정론적)와 MuJoCo Warp를 쓴다. MJX는 X0에서 sm_75 동작을 확인한 뒤에만 쓴다.

## F5. 데이터 계보 누수 (상세는 Part 3)

- AMASS와 HumanML3D는 SMPL(학습된 통계 바디 모델)을 거쳐 만들어졌다.
- Cycles 기본 디노이저(OIDN, OptiX AI), COLMAP의 학습 특징과 어휘 트리, 폰 ISP의 ML 기능, torchvision의 기본 사전학습 가중치가 시스템 경로에 몰래 들어올 수 있다.

**업그레이드.** **Lineage Ledger CI**와 **Sensor Purity Test**.

## F6. GPU-A 메모리와 TC 폭주 (MAN-S DMD, BSRT)

**원인.**
- Z-Image fp16(12.3 GB)과 래스터·BSRT 버퍼를 한 장에 같이 올릴 수 없다.
- MAN-S DMD는 96만 TC-eq, 200–267 T4-h가 든다.
- BSRT를 하루 1만 장 하면 9.4 h가 든다.

**업그레이드.**
- (a) **Exhaust-to-Shader** [조합]
  - BSRT가 만드는 세탁 쌍 $(x_r,x_p)$, G-buffer, 같은 MAN 노이즈를 그대로 MAN-S의 지도학습 데이터로 쓴다.
  - 목표는 $r^*=\mathrm{HP}_{>b_y}(x_p-x_r)$, 즉 라벨 대역보다 높은 대역의 잔차다. 지도학습은 3 T4-h다.
  - 그 뒤 DMD는 5k 스텝만 돌린다(16만 TC-eq, 33–44 T4-h).
- (b) **GPU-A는 교사 전용**으로 둔다.
  - 판정은 INT8(6.2 GB)로 한다.
  - BSRT와 DMD 단계에서만 fp16 가중치로 스왑한다. pinned 호스트 메모리에서 12.3 GB를 올리는 데 약 1.3 s다.
  - 래스터와 렌더는 GPU-B로 옮긴다.
- (c) LoRA 학습은 B1–2 + gradient checkpointing(약 14.5 GB)이나, INT8 동결 베이스 + fp16 LoRA(약 10 GB, 속도는 실측 필요)로 한다.
- (d) BSRT 쿼터: 초기 2만 장 코퍼스를 만든 뒤 하루 4k장으로 줄인다. Reader-S 실사 오차가 정체될 때만 늘린다.

## F7. Reader-S의 단안 lift 과신

**원인.**
- 사전학습된 깊이 모델이나 인체 모델 없이 단안 웹 영상에서 metric 3D를 얻는 것은 불량조건이다.
- DA-V2 저자들도 "합성만으로 학습한 깊이는 일반화 문제가 심각하다"고 썼다.
- 그런데 제안서는 RGF $p_{real}$과 PCGD-0 보정을 단안 lift 품질에 묶어 두었다.

**업그레이드: Lift Ladder + Gauge-matched Lift** [신규 정식화]
- 사다리는 네 칸이다.
  1. **L-time.** 사건 시각. 단안으로 충분하다(R-Event, 오디오 onset, 2D 트랙).
  2. **L-shape.** 단안 2D 궤적의 게이지 불변 통계. GICC 보정과 EPD-S 모양 통계에 쓴다.
  3. **L-metric.** 지면과 크기 단서가 있을 때만.
  4. **L-3D.** 다시점만. Garage Gold Set, 라이선스가 확인된 멀티캠.
- 절대 물리량($p_{real}(F_g)$, PCGD-0 파라미터)은 L-3D에서만 추정한다.
- 사이클 게이트: lift → 재렌더 → R-Kpt/R-Flow로 다시 측정한 오차가 τ 이하여야 통과한다.

---

# Part 3. 제약 위반 점검: 몰래 들어온 사전학습 모델

| # | 위치 | 끼어든 학습 모델 | 판정 | from-scratch 또는 고전 대체 |
|---|---|---|---|---|
| 1 | MoPri-S 데이터. §1.3은 AMASS를 D0로 분류했다 | AMASS는 MoSh++가 **SMPL-H**(학습된 통계 바디 모델)로 마커를 피팅해 만든 관절이다. HumanML3D 모션은 AMASS에서 파생됐다. 계보상 D3에 해당하고 비상업 라이선스다 | **위반 위험** | CMU 원본 ASF/AMC(D0), 100STYLE 원본 BVH(Xsens, CC BY 4.0은 2차 출처로만 확인 [2차]), Garage 마커 모캡(D0). 텍스트-모션 쌍은 **Act-from-Text 캡처**로 만든다: 배우가 적힌 문장을 연기하므로 구성상 D2다. AMASS와 HumanML3D는 연구 ablation에만 쓴다 |
| 2 | Denoiser-S 학습용 렌더, 데모 렌더 | Cycles 기본 디노이저: OIDN(Intel 사전학습), OptiX AI(NVIDIA 사전학습) | **위반** | `denoiser=None`을 강제하고 CI가 .blend 설정을 검사한다. 디노이즈는 Denoiser-S만 한다 |
| 3 | COLMAP/GLOMAP | 4.0부터 ALIKED/LightGlue(ONNX)가 들어 있다. 어휘 트리는 외부 이미지로 학습한 k-means 코드북이다 | 위반 위험 | `feature_type=SIFT`와 sequential/spatial 매칭을 강제한다. 어휘 트리는 우리 데이터로 직접 만든다(CPU 수 시간) |
| 4 | Garage 폰 촬영 | 일부 기기의 슬로모 AI 프레임 보간, 멀티프레임 합성, 시맨틱 톤매핑, AI 안정화 | **위반(라벨 오염)** | **Sensor Purity Test** [조합]: 1 kHz 이진 카운터 LED 10개 + 피에조 클릭 보드($30–80). 보간된 프레임은 카운터 상태가 섞여 보인다. 같은 보드로 rolling shutter readout 시간과 장치별 AV 오프셋을 측정한다. HDR과 AI 기능을 끈 수동 앱만 쓴다 |
| 5 | Expr-S 데이터(Q4) | ARKit 얼굴 추적 | 판결 전 금지 | **Dot-Face**: 얼굴 점 마커 + 폰 3대 삼각측량 + 최소자승 블렌드셰이프 솔브(D1) |
| 6 | MAN-S, Denoiser-S의 perceptual loss | LatentLPIPS 공개 가중치, VGG, LPIPS | 위반(학습 경로) | 교사 내부 특징(TFP)을 **PercepNet-S**(3M, from scratch)로 증류한다. LPIPS는 평가에만 쓴다 |
| 7 | 교사 부속 | sdxl-vae-fp16-fix 같은 제3자 파인튜닝 가중치 | 회색 | 교사 원본 가중치와 upcast만 쓴다. Z-Image VAE의 fp16 NaN은 E1에서 측정하고, 필요하면 VAE만 fp32로 돌린다 |
| 8 | R-Flow, R-Kpt 구현 | torchvision RAFT의 기본 가중치, HRNet 공식 설정의 ImageNet 초기화 | **위반(기본값 함정)** | `weights=None`을 강제한다. 체크포인트 해시 금지 목록을 둔다. 첫 스텝에서 가중치 초기화 통계를 검사한다 |
| 9 | 한국어 TTS-S | mecab-ko(세종 말뭉치로 학습한 CRF)를 쓰는 G2P | 위반 위험 | 규칙 기반 자모 G2P, 또는 자체 학습 G2P |
| 10 | Event-Demand Lift 메타데이터 | 자동 생성 자막(ASR), 자동 태그 | D3 | 업로더가 직접 쓴 제목과 태그만 쓴다(D2) |
| 11 | 오디오 간격 증인 | AudioSet 분류기로 단 자동 태그 | D3 | Freesound의 사람 태그만 쓴다 |
| 12 | Reader 데이터 | COCO, FlyingChairs/Things, Sintel | 순도는 통과, **라이선스 위험** | 상업 경로는 자체 절차적 "FlyingStuff" 렌더 + Garage 키포인트 사람 라벨로 간다 |
| 13 | §18 부품 목록 | Move.ai, Wonder Studio | 학습 모델 제품 | 설계 선례로만 언급한다. 파이프라인에 넣지 않는다고 명시한다 |
| 14 | Lingua Franca | Qwen3-4B 텍스트 인코더 | 이미지 모델 부속 | 임베딩 조건으로만 쓰도록 허용을 요청한다(Q3). LLM 생성 용도는 금지다. 판결이 부정이면 ShotScript 닫힌 어휘 + 자체 8M 텍스트 인코더로 간다 |
| 15 | 평가 | VGGT(GeCo), VBench, VideoPhy-2 자동평가 VLM, LPIPS | 평가 전용이면 허용 | CI가 시스템 경로에서 import를 차단한다 |
| 16 | 교사 양자화 | Nunchaku INT4 체크포인트 | 같은 모델이므로 허용 | 판정 일치도 게이트를 통과해야 쓴다 |

**Lineage Ledger CI** [조합]
- 모든 데이터, 라벨, 가중치에 계보 DAG 해시를 붙인다.
- 시스템 경로 아티팩트의 조상 노드는 다섯 종류만 허용한다: 교사, 우리 학습, 고전 알고리즘, 센서, 사람.
- 그 밖의 노드가 하나라도 있으면 빌드를 실패시킨다.
- 자동 검사 항목: import 금지 목록, Blender 설정, COLMAP feature_type, 체크포인트 해시, 폰 영상 메타데이터.

---

# Part 4. 2×T4로 30일 안에 돌릴 결정적 실험 5개

### E1. TJC: Teacher-as-Judge Calibration (Phase 1 Day-3 bake-off와 같이 돌린다)

- **가설.**
  - H1a: 512² INT8 판정이 1024² fp16 판정과 Spearman ≥0.9로 일치한다.
  - H1b: 비증류 경로(Base 또는 Turbo+SR)의 DR이 실사 대 렌더, 정상 대 결함을 AUC ≥0.8로 가른다. Turbo 원본 DR은 이보다 0.1 이상 낮다.
  - H1c: fp16 NaN이 10⁴회 중 0회다.
- **셋업.**
  - 이미지: Garage 실사 1,000장 + 매칭 렌더 1,000장 + 결함 합성 1,000장.
  - 조건: 해상도 {512, 768, 1024} × 정밀도 {fp16, INT8, INT4} × 경로 {Base, Turbo} × σ 3개.
  - 지연, VRAM, NaN을 모두 로그한다.
- **T4-h.** 약 25. 1024² 기준치 9천 TC에 약 9 h, 나머지 조건에 12 h, 벤치 4 h.
- **성공.** H1a·H1b·H1c 모두 통과하고 가격표가 확정된다.
- **실패하면.**
  - 기본 해상도를 768²로 올리고 L3 게이트를 25%에서 15%로 좁힌다.
  - INT8이 실패하면 Nunchaku INT4를 시험한다.
  - Base 판결이 부정이고 Turbo DR이 불량이면 SR-LoRA에 착수한다(Phase 1과 공동).

### E2. GICC Chrono (P2-X1 개정)

- **가설.** 교사는 "모양" 왜곡 4종 중 2종 이상을 AUC ≥0.75로 잡는다. 게이지 대조(g×0.7, g×1.4, 균일 λ)에는 0.5±0.05로 반응하지 않는다.
- **셋업.**
  - Sensor Purity를 통과한 폰 240fps 고정 카메라로 6범주(바운스, 투척, 진자, 낙하, 점프, 회전) × 40클립을 찍는다.
  - 2,000장을 합성한다(가산, 매트, 장노출).
  - 왜곡은 실제의 다른 프레임을 고르는 것만으로 만든다. common random numbers를 쓴다.
  - 사람 2AFC 300쌍으로 기준선을 잡는다.
- **T4-h.** 약 30. 판정 8.4만 TC × 0.5 s ≈ 12 h, 조건부 Chrono-LoRA 15 h, 여유 3 h.
- **성공.** 위 가설 + held-out 범주에서도 ≥0.7.
- **실패하면.**
  - Chrono-LoRA, 라이트 페인팅 모드, (K,Δ) 재설계 순으로 시도한다.
  - 그래도 안 되면 크로노는 히어로 미감 선택 전용으로 돌리고, 시간의 모양은 ID-Screen과 실통계로 맡긴다(H7).
  - 게이지 대조에서 AUC > 0.6이 나오면 노출 개수와 블러 길이를 정규화하고 다시 잰다.

### E3. Verified Coverage–Precision (P2-X4를 앞당기고 축소)

- **가설.** ID-Screen 순위가 MJPC 성공을 AUC ≥0.8로 예측한다. 그러면 검증 비용이 10 s 클립당 20분 이하가 되고, 검증된 best-of-256이 단일 샘플을 2AFC에서 15%p 이상 이긴다.
- **셋업.**
  - MoPri-S(32M)를 전체 스케줄의 60%만 학습한다. 데이터는 CMU 원본과 100STYLE(D0).
  - 4범주 × 10프롬프트. best-of-{1, 16, 256, 4096}.
  - 무작위 200개는 MJPC로 전수 검증해 순위 AUC 교정에 쓴다.
- **T4-h.** 약 40(학습 26, MJPC 11, 기타 3). 사람 평가 1,500쌍 $1–2k.
- **실패하면.**
  - ID-Screen에 접촉 스케줄 일관성 항을 더하고, MJPC 성패로 MoFeas-S(2M)를 학습한다.
  - 2AFC가 미달이면 MoPri를 64–80M으로 키우고 TPRF RWR을 앞당긴다.

### E4. SMC + Chaos Bank (P2-X0 + X3 축소)

- **가설.** 뱅크 warm start가 있으면 중형 과제가 N=256에서 제약 충족 ≥90%, 블록당 ≤90 s T4-eq를 달성한다. 플라이휠 혼합 평균은 ≤80 s다.
- **셋업.**
  - 1주차 X0 벤치: CPU MuJoCo, MuJoCo Warp, MJX 가능 여부, Warp FLIP/MPM, gsplat 256², TensorRT INT8.
  - 템플릿 3종으로 뱅크 10⁵개를 CPU 야간에 만든다.
  - 과제 20 × 제약 5 × N {64, 256, 1024} × 뱅크 유/무.
- **T4-h.** 약 20, CPU 야간 4일.
- **성공.** 위 가설 + "조종된 rollout" 식별률 ≤55%.
- **실패하면.** 허용오차 창 학습, 잔여력 예산 확대, Feas-S 앞당김, 뱅크를 10⁶개로.

### E5. BSRT 보존성 + Exhaust-to-Shader 종자 (P2-X5 축소)

- **가설.**
  - H5a: 세탁한 쌍의 flow GT가 보존된다. DIS 잔차 EPE ≤0.5 px, 기각률 ≤20%.
  - H5b: R-Flow-S를 합성 + BSRT로 학습하면 합성만 쓸 때보다 Garage 희소 실측 flow의 EPE가 15% 이상 낮다.
  - H5c(보조): 같은 쌍으로 지도학습한 MAN-S가 Appearance 2AFC에서 5%p 이상 오른다.
- **셋업.** 렌더 1만 장 → BSRT(장당 4 TC) → R-Flow-S를 합성만/합성+BSRT 두 조건으로 학습 → MAN-S 지도학습.
- **T4-h.** 약 52(BSRT 10, 학습 36, MAN-S 3, 평가 3).
- **실패하면.**
  - 기각률이 높으면 σ*를 한 단계 낮추고 DIS로 GT를 되돌려 보정한다.
  - 실사 이득이 부족하면 렌더 다양성(조명, 센서 노이즈, 물리 블러)을 넓히고 Gold 라벨로 파인튜닝한다.

**합계와 게이트.**
- 합계 약 **167 T4-h**로 30일 용량 1,440 T4-h의 12%다.
- Kaggle 모드에서는 E1 축소판, E2, E4(CPU는 외부)만 돌린다. 약 70 T4-h.
- **30일 게이트:** E1 가격표 확정은 필수다. 여기에 E2·E3·E4 중 2개를 통과하면 90일 본계획으로 간다.

---

# Part 5. 개정 제안서 전체본: Phase 2 v2.1-T4

## "작은 뇌, 정확한 우주, 큰 탐색"

> **렌즈.** Phase 2 설계자 + 저예산 데이터 플라이휠 전략가.
> **기준.** PARADIGM v1 §0.5, §1, §2, §4, §7, §8, §9, §10. premortem 04-phase2-v2.
> **이번 개정.** 프리모템 F1–F7을 반영했다.
> **표기.** [신규], [조합], [부품], [v1 유지·축소·폐기·미룸]. "(추정)"이 붙은 값은 E1–E4에서 실측값으로 바꾼다. 목표는 하나도 낮추지 않았다.

## 0. TL;DR

1. **학습이 할 일이 아닌 곳에서 컴퓨트를 빼냈다.**
   - native WFM($10^{23}$–$10^{25}$)과 2천만 클립 lift는 폐기한다.
   - 그 자리를 세 가지가 메운다.
     - (a) 빌린 교사. Z-Image 계열의 사전학습 컴퓨트는 3.4–4.5×10²³으로, 우리 1년 예산의 **340–450배**가 선불로 들어가 있다.
     - (b) 학습이 없는 고전 물리, 광학, 음향.
     - (c) 테스트타임 탐색.
   - 작은 모델은 이 셋이 메우지 못하는 저차원 잔차만 배운다. from-scratch 모델은 10개, **약 123M**이다.
   - **1세대 학습은 약 320 T4-h, 2×T4로 6.7일**이다. SR-LoRA가 필요하면 60–115 T4-h를 더한다.
2. **화폐는 TC(교사 호출)이고 가격표로 운영한다.**
   - Z-Image @512² 기준 fp16은 3.4–4.6k TC/T4-h, INT8은 5.5–8k(추정)다.
   - 판정은 비증류 경로로 하고 샘플링은 Turbo로 한다(One-Teacher, Two-Mode, Q0).
   - Teacher Exhaust, Feature-Tap, 불확실성 게이트 L3, Judge Cascade로 블록당 TC를 래칫한다.
3. **Coverage–Precision Split v2** [신규 정식화]. 물리 사영이 넓혀 준 분지(basin) 위에서 best-of-N을 돌리는데, 검증도 캐스케이드(ID-Screen)로 싸게 만든다. 10 s 모션 best-of-4096의 검증 포함 비용은 6–18분이다.
4. **크로노포토 증인 + GICC** [신규]. 시간을 사진 한 장으로 접는다.
   - 교사는 "시간의 모양"만 판정한다: 탄도 2차 차분, 반발 결합, 소산 방향, 등시성.
   - "시간의 눈금"(g, 템포)은 물리 불변량과 오디오가 판정한다.
   - 차원 해석으로 보면 사진 한 장은 눈금을 알 수 없다.
5. **저주파 서사는 사람이 소유한다.** ShotScript와 폰 연기(Act, don't prompt)로 정한다.
6. **Reader는 직접 만든다.** 고전 SfM/flow + Reader-S(26M) + BSRT. lift는 식별 가능한 사다리(L-time, L-shape, L-metric, L-3D)에서만 쓴다.
7. **물리는 T4에서 상대적으로 싸다.**
   - 중형 강체 SMC는 뱅크가 없으면 3–13분, 있으면 1–3분이다.
   - 유리는 Voronoi + 강체, 유체는 CFL 적응 FLIP과 multi-fidelity로 한다.
8. **로드맵(M0 기준, 승리 조건은 v1 그대로).**
   - 2a: M3–M9
   - 2b: M8–M16
   - 2c: M14–M28 (병목은 사람 데이터)
   - 2d: M26–M42 이후
9. **플라이휠 처리량.**
   - Day-1은 하루 350블록이다.
   - 뱅크와 Feas-S가 붙는 약 5회전부터 하루 1,000블록이다.
   - 10회전 목표는 하루 4,000–8,000블록이다. 상단은 L3 커버리지를 25% 이하로 게이팅할 때 가능하다.
10. **스케일 원칙.** 텐서 계약, 데이터 자산, μP는 모델 크기와 무관하다. 컴퓨트가 1000배로 늘면 같은 슬롯에 큰 모델을 끼우면 된다.

## 1. 전제와 회계

### 1.1 T4 회계 단위

| 항목 | 값 |
|---|---|
| FP16 텐서 | 피크 65 TF. 실측 HGEMM 최고 41.6 TF(클럭 고정). 지속 부하는 70W 쓰로틀 |
| 학습 실효 계획값 | 큰 GEMM 16 TF/장(1 T4-h = 5.8×10¹⁶). **50M 미만 소형은 12 TF/장(4.3×10¹⁶)** |
| 상시 2×T4 | 하루 48 T4-h. 90일 4,320 T4-h(1.9–2.5×10²⁰). 1년 17,520 T4-h(0.75–1.0×10²¹) |
| Kaggle | 주 30 세션-시간. T4×2가 쿼터를 1배로 소모하면 주 60 T4-h, 2배면 30. 세션 12 h. vCPU 2–4(미확인) |
| 메모리 | 16 GB(ECC를 켜면 약 15 GB), 320 GB/s. 정수 연산 실측 INT8 74.9 TOPS, INT4 114 TOPS. RT 코어 40개, NVENC |
| 호스트 | 16 vCPU 이상, 64 GB RAM 이상. CPU는 세 번째 가속기이고, 교사 가중치 pinned 스왑에도 RAM이 필요하다 |
| 툴체인 | sm_75 고정. FA2 없음. SDPA mem-efficient 또는 flash-attention-turing(head_dim 64). Triton matmul 경로 금지. cuBLAS, CUTLASS, Warp, gsplat이 기본. Docker로 버전 고정 |

### 1.2 TC 가격표 [신규 운영]. E1에서 확정한다

| 교사, 해상도, 정밀도 | forward 1회 (T4) | TC/T4-h |
|---|---|---|
| Z-Image(6.15B) 512² fp16 | 0.8–1.05 s | 3.4–4.6k |
| 〃 512² INT8 (검증 필요) | 0.45–0.65 s | 5.5–8k |
| 〃 512² INT4 Nunchaku (T4 미검증) | 0.3–0.5 s | 7–12k |
| 〃 768² fp16 | 1.7–2.3 s | 1.6–2.1k |
| 〃 1024² fp16 | 3.1–4.1 s | 0.9–1.2k |

- **계획값.**
  - 판정은 0.6 s(INT8 @512²), 즉 약 6k TC/T4-h로 잡는다. E1에서 512²가 OOD로 나오면 768²로 올린다.
  - 생성, BSRT, DMD는 0.9 s(fp16), 즉 약 4k TC/T4-h로 잡는다.
- **CFG.** 판정은 조건부 단일 경로라 1 TC다. DMD real score만 CFG 2 TC다. DMD 1스텝은 약 7 TC-eq다.

### 1.3 1년차 Phase 2 컴퓨트 (약 6,000 T4-h)

| 용도 | T4-h/년 | 비중 |
|---|---|---|
| 작은 모델 학습 (3–4세대 포함) | 900–1,300 | 15–22% |
| TC: 판정, 슬라이스 DM, LoRA, BSRT | 2,500–3,000 (13–15M TC) | 40–50% |
| GPU 시뮬 | 400–700 | 7–11% |
| 렌더 | 900–1,200 | 15–20% |
| 벤치와 여유 | 300–500 | 5–8% |

- **관찰.** 레버는 "더 큰 모델"이 아니라 "더 싼 판정"과 "더 싼 검증"이다.
- 학생의 절반(EPD-S, MoPri-S, Feas-S, F_φ-S)은 픽셀을 보지 않는다. 이들의 데이터를 만드는 데는 TC가 들지 않는다.

### 1.4 데이터 순도 등급과 계보 장부

| 등급 | 정의 | 예 | 시스템 경로 학습 |
|---|---|---|---|
| D0 | 센서, 모캡, 시뮬 GT | 우리 렌더와 시뮬, Garage, **CMU 원본**, 100STYLE BVH | 허용 |
| D1 | 고전 알고리즘 라벨 | SIFT-COLMAP/GLOMAP 포즈, 배경차분, 오디오 onset, ChArUco, Dot-Face 솔브 | 허용 |
| D2 | 사람 주석 | COCO keypoint, Act-from-Text 쌍, 업로더 태그 | 허용 |
| D3 | 학습 모델이 개입한 라벨 | SpatialVID pose/depth, Motion-X, **AMASS/HumanML3D(SMPL)**, ASR 자막, ARKit | **금지.** 평가와 연구 ablation 전용 |

- **Lineage Ledger CI**를 둔다(Part 3).
- 라이선스는 등급과 별개로 관리한다. 상업 경로는 CMU, 100STYLE(CC BY 4.0, 1차 확인 필요), 자체 캡처만 쓴다.

### 1.5 빌린 것 한 장

- **교사는 Z-Image 계열 하나다.** Base(판정, score)와 Turbo(샘플)를 같은 모델로 보는 판결을 요청한다(Q0).
  - 판결이 부정이면 Turbo + **SR-LoRA**(r32, attention + MLP, 약 70M, 1.1%)로 간다.
- **어댑터** (교사 6.15B 대비).

| 어댑터 | 크기 | 비율 |
|---|---|---|
| fake-score LoRA r32 | 29.5M | 0.48% |
| Chrono-LoRA r16 | 14.7M | 0.24% |
| identity LoRA r8 | 캐릭터당 7.4M | 0.12% |

  - 동시에 장착하는 합은 2% 이하로 둔다. SR-LoRA를 쓸 때는 나머지를 순차로 장착한다.
- **Lingua Franca** [조합]. Qwen3-4B 텍스트 인코더의 임베딩을 캐시해서 작은 모델들의 조건으로 쓴다(Q3). 생성 용도로는 쓰지 않는다.
- **LLM 금지.** 사건 스크립트는 ShotScript의 `event` 구문으로 받고 결정론적 파서로 처리한다.
- **Perceptual loss.** 교사 중간 특징(TFP)을 증류한 PercepNet-S를 쓴다. LPIPS는 평가에만 쓴다.

## 2. 다섯 원리 (제약판)

**P1. Coverage–Precision Split v2.**

$$P_{hit}=1-(1-q\,r)^N,\quad \mathfrak B(A)=\{x:\Pi_{phys}(x)\in A\},\quad C=N(c_{gen}+c_0+c_1)+N_2c_{MJPC}+N_3c_{TC}$$

- 물리 사영이 좋은 결과의 얇은 다양체를 두꺼운 분지로 넓힌다.
- 생성기는 분지 위에 $q\ge10^{-3}$의 확률질량만 두면 된다.
- 검증은 ID-Screen 캐스케이드로 하고, 재현율 $r\ge0.95$를 게이트로 둔다.
- 이 경제학은 정확한 검증기가 있는 곳, 즉 물리에서만 성립한다.

**P2. Tiny Federation.** 각 모델의 크기를 맡은 대역의 내재 차원에 맞춘다. 물리가 이미 아는 것은 0 파라미터로 둔다.

**P3. TC는 화폐다.**
- Teacher Exhaust는 (입력 해시, σ, 교사 버전, 해상도, 정밀도, 점수, 특징 탭)을 기록한다.
- 판정 캐스케이드는 4단이다: 불변량 → RGF/ID-Screen → INT8 proxy → 교사.
- L3로는 불확실성 상위 25%와 무작위 10%만 보낸다.

**P4. 사람이 저대역을 소유한다.** 0–1 Hz 서사는 감독이 정한다.

**P5. 실리콘을 남김없이 쓴다.**

| 유닛 | 담당 |
|---|---|
| Tensor FP16 | 교사 생성, 학생 학습 |
| INT8 / INT4 | 판정과 critic |
| CUDA FP32 | 시뮬 |
| RT 코어 | 굴절, 반사, 음선 추적 |
| CPU | MuJoCo, SfM, ID-Screen, 뱅크 구축 |
| NVENC | 프리뷰 인코딩 |

GPU에서 돌릴 이유가 없는 일은 CPU에서 돌린다.

## 3. 시스템 한 장과 구멍 메우기 지도

```
ShotScript(event/blocking) ⊕ 자유 텍스트 → Qwen3-4B 임베딩(캐시)  ⊕ 폰 연기(2c)
  ▼
[W_canon] Phase 1 세계 + Reveal-Bundle Demand edit-closure
          물성 ψ = 재질 클래스 → 공학 핸드북 표, 충돌 프록시 = V-HACD
  ▼
[Events] EPD-S 28M (슬롯 고정 inpainting, Feas-S guidance) ← Chaos Bank 검색
  ▼
[Realize] Milestone SMC: CPU MuJoCo | MuJoCo Warp | Voronoi 강체 | CFL 적응 FLIP | MPM | XPBD   (학습 0)
          PCGD-0 (루프 안) · F_φ-S 0.6M (top-k에만)
          캐릭터: MoPri-S best-of-N → ID-Screen → MJPC-lite → TPRF
          판정: 불변량 I1–I5 → RGF → INT8 proxy(Hero/Chrono/Feas) → TC(Base 판정, GICC) → 감독
  ▼
[Block] (E, s(t), seeds, sim 버전, lineage) → Merkle commit → Block Bank
  ▼
[Shoot] DP-0 γ(τ) → 3DGRUT 셔터 적분 → MAN-S → Denoiser-S → 렌즈·센서 (Read-Only, I6)
        음향: J → FEM 모달 → image source + RT 음선 추적 (학습 0)
```

메움 표기: (a) 빌린 교사, (b) 구조와 고전 알고리즘, (c) 테스트타임 탐색, (d) 작은 모델.

| 능력 | 2×T4판 | 메움 |
|---|---|---|
| 정지 순간의 외관 | 교사 slice DM (Base 판정) | (a)+(c) |
| 저주파 의미 | 감독, 폰 연기, EPD-S, 텍스트 임베딩 | 사람+(d)+(a) |
| 시간의 모양 | 크로노 증인 GICC | (a) |
| 시간의 눈금 | I4, 오디오 간격 증인, Garage | (b) |
| 동역학 | 고전 시뮬 + PCGD-0 + F_φ-S | (b)+(d) |
| 혼돈 제어 | Milestone SMC + Chaos Bank | (c)+(b) |
| 캐릭터 모션 | MoPri-S × ID-Screen × MJPC-lite | (d)+(c)+(b) |
| 셰이딩 | MAN-S (Exhaust-to-Shader + DMD) | (d)+(a) |
| 숨은 면 | Reveal-Bundle Demand | (c)+(a) |
| 실영상 이해 | SIFT-SfM, DIS + Reader-S, Lift Ladder | (b)+(d)+(a) |
| 소리 | 모달 합성, 음성은 녹음 | (b) |
| 판정 비용 | Exhaust, Feature-Tap, 캐스케이드 | (d) |

## 4. 판사 재편: 비디오 모델 0개, 판사 넷과 사람 하나

### 4.1 증인 대체 매트릭스

| v1 증인의 임무 | 대체 수단 |
|---|---|
| 저주파 의미, 연출 | 감독(ShotScript, 폰 연기), EPD-S |
| 거친 타이밍 | 실통계(L-time) + 크로노 GICC |
| 퍼포먼스 latent z | 감독의 테이크 선택, 연기 lift |
| 희귀 사건 공급원 | 상상하지 않고 굴린다: 세트피스 × SMC, Chaos Bank |
| 평가 | 외부 API는 평가 전용. Arena와 Spectrum Report |

### 4.2 기울기 라우팅

| 판사 | 판정 대상 | 기울기 또는 선택이 흐르는 곳 | 차단 |
|---|---|---|---|
| 교사 슬라이스 (Base, TC) | 정지 공간 사실감 | θ, MAN-S, intrinsics, slot. s(t)와 E는 선택만 | s(t), E, ψ |
| 크로노 GICC | 시간의 모양 (0.3–15 Hz) | EPD-S 거친 타이밍, MoPri 저대역 (RWR과 선택만) | 외관, 눈금 |
| 물리 (시뮬, 불변량, RGF, ID-Screen) | 불변량, 고대역, 눈금 | PCGD-0, F_φ-S, MJPC | 외관 |
| 실통계 (Garage, 오디오, 모캡) | 간격과 타이밍 분포 | EPD-S (MMD), PCGD-0 | 외관 |
| 감독 | 0–1 Hz 서사 | 스크립트 슬롯, 테이크 | — |
| 외부 API | 평가 | 없음 | 전부 |

### 4.3 크로노포토 증인 + GICC [신규, 확신도 중]

**렌더 모드.** 고정 증인 카메라 $c_w$, 스트로브 시각 $t_k=t_0+k\Delta$, $k<K\le12$.

$$X_{add}=\mathrm{Cam}\Big(\sum_k\mathcal R_{strobe}(W(t_k),c_w)\Big),\quad X_{mat}=\mathcal R(W_{bg})\odot\prod_k(1-m_k)+\sum_k m_k\odot\mathcal R(W(t_k)),\quad X_{long}=\tfrac1S\!\int\!\mathcal R(W(t))dt$$

- $X_{add}$: Edgerton식 가산 다중노출. 어두운 배경과 섬광 조명.
- $X_{mat}$: 시퀀스 사진식 매트 합성. 깊이 순서를 지킨다.
- $X_{long}$: 장노출. LED 마커만 남기는 라이트 페인팅 모드도 있다.

**BIS (Band-Isometric Strobe).**
- 노출 간 화면 변위를 교사가 결정하는 공간 대역에 맞춘다: $\Delta_b=d^*_b/\lVert v_{screen}\rVert$, $d^*_b\approx24$–64 px.
- 변위는 증인 카메라의 거리와 줌으로 맞춘다.
- 조합 세 개로 0.07–15 Hz를 덮는다: Δ = 1/30 s, 1/5 s, 1.2 s.

**GICC 점수.**

$$\Delta_\chi(h)=\mathrm{DR}_b(X(h))-\mathbb E_{w\sim\mathcal W_{inv}}\mathrm{DR}_b(X(h\circ w)),\qquad \text{조건: }\ \mathrm{AUC}(\mathcal W_{ctrl})\in[0.45,0.55]$$

- **채점용 $\mathcal W_{inv}$:** 시간 역전(소산계), 구간 ease-in-out, 반발 결합 파괴, 진자 등시성 파괴, 보행 duty factor 이동.
- **음성 대조 $\mathcal W_{ctrl}$:** g×κ, 균일 λ. 사진은 $(s,\Delta)$ 게이지를 모르므로 이 둘은 원리상 판정 대상이 아니다. 대조가 0.5에서 벗어나면 가짜 단서가 새는 것이다.
- common random numbers를 쓴다. 가산 합성은 순서에 불변이므로 순열이 아니라 간격 왜곡을 쓴다.

**비용과 사용처.**
- 판정 1회 = 모드 3 × σ 2 × (원본 1 + 왜곡 4) = 30 TC다. INT8로 13–18 s.
- 그래서 캐스케이드 상위 k개에만 쓰고, 나머지는 ChronoProxy-S가 맡는다.
- 쓰는 곳: SMC 가중치, RWR 보상, QBC로 $(c_w,K,\Delta)$ 선택.

**보정 데이터.**
- Sensor Purity를 통과한 240fps 실영상에서 배경 median과 차분 마스크로 합성한다(D1).
- 왜곡 negative는 실제의 다른 프레임을 고르는 것만으로 만든다. 보간은 쓰지 않는다.

**실패하면.** Chrono-LoRA(r16, 15–25 T4-h) → 라이트 페인팅 → 미감 전용으로 강등(H7).

**쇼케이스.** 에저턴 헌정: 1957년 우유 왕관에서 2026년 와인 왕관으로. 같은 블록에서 스트로보 스틸, 불릿타임, 램프를 뽑는다.

### 4.4 실제 증인 (사전학습 모델 0개)

**Garage Gold Set** [조합]
- 폰 3–4대로 240fps를 찍는다. 장비는 $0–2k.
- 보정과 동기화: ChArUco, 박수와 오디오 상호상관.
- **Sensor Purity Test**로 다음을 확인하고 측정한다: 프레임이 실제 노출인지, rolling shutter readout 시간, 장치별 AV 오프셋.
- 강체 6DoF: 컬러 마커, 삼각측량, Kabsch.
- 사건 시각: 블러 + 오디오 onset으로 ±2 ms. 행 지연과 AV 오프셋을 보정한 뒤의 값이다.
- v0 목표: 6시간, 약 2만 사건.

**오디오 간격 증인** [조합]

$$\Delta t_{ij}=(t^{ev}_i-t^{ev}_j)+\beta(d_i-d_j)+(n_i-n_j)$$

- 클립별 AV 오프셋이 간격에서 상쇄된다.
- 연속 바운스 간격의 비 $T_{k+1}/T_k=e$가 반발계수다. 같은 방법으로 cadence와 드리블 주기를 얻는다.
- 태그는 사람이 단 것만 쓴다(D2).

**모캡.** CMU 원본, 100STYLE, Garage, Act-from-Text 캡처. AMASS는 연구 전용이다.

**Event-Demand Lift.**
- 사건 그래프 수요가 비어 있는 범주만 lift한다. 연 1e4–1e5 클립이고 CPU로 하루 약 9 h다.
- Lift Ladder를 따른다.

| 단 | 내용 | 조건 |
|---|---|---|
| L-time | 사건 시각 | 단안 가능 |
| L-shape | 게이지 불변 2D 통계 | 단안 가능 |
| L-metric | metric 궤적 | 지면과 크기 단서가 있을 때 |
| L-3D | 3D 궤적 | 다시점만 |

- 절대 물리 통계는 L-3D에서만 추정한다.
- 사이클 게이트: 재렌더 후 Reader-S로 다시 잰 오차가 τ 이하일 때만 쓴다.

### 4.5 RGF on T4 [v1 유지]

- 유령력(ghost force)은 lift된 궤적에 CPU 역동역학을 적용해 구한다.
- $p_{real}$은 범주별로 L-3D에서 추정한다.
- Reality Residual Budget은 그대로 둔다.
- 같은 엔진이 ID-Screen L1이 된다.

### 4.6 시간 연방 지도 v2.1

| 공간 \ 시간 | 저 (0–1 Hz) | 중 (1–8 Hz) | 고 (>8 Hz) |
|---|---|---|---|
| 레이아웃과 서사 | 감독, EPD-S | EPD-S + 크로노(모양) | 사건(물리) |
| 형태와 모션 | MoPri-S + 크로노 | MoPri-S + MJPC | 접촉, PCGD-0 |
| 외관 | 교사 (canonical) | 렌더러 (블러 적분) | 렌더러, 센서, MAN-S |
| 확률적 물질 | Stam 바람장 | SMF sway | 파티클 |
| 카메라 | DP-0 | 리그 동역학 | 손 모델 |
| **시간의 눈금** | I4, 오디오 | I4, 오디오 | 물리 |

## 5. 작은 파운데이션 모델 카탈로그

### 5.0 공통 레시피

- **처음부터 학습한다.** `weights=None`을 강제하고 Lineage CI로 확인한다.
- **μP.**
- **fp16 안정화:** dynamic loss scaling, QK-norm, z-loss, fp32 softmax/LayerNorm, gradient clipping.
- **커널:** head_dim 64. SDPA mem-efficient 또는 flash-attention-turing. 넓고 얕게(d≥384, 토큰 배치 ≥32k).
- **배포:** TensorRT. critic은 INT8, 샘플러는 fp16.
- **디퓨전 계열:** flow matching 후 4-step 증류(학습 비용 +30%).
- **학습 FLOPs:** $6ND\times1.3$. 소형은 1 T4-h = 4.3×10¹⁶. GNN과 RAFT류는 실측 MFU로 따로 잡는다.

### 5.1 카탈로그

| 모델 | 파라미터 | 아키텍처 | 입력 → 출력 | 데이터 (순도) | 학습 FLOPs | T4-h | 추론 (T4) | 메모리 학습/추론 |
|---|---|---|---|---|---|---|---|---|
| **EPD-S** | 28M | Transformer 12L, d384, 6헤드×64, self + cross-attn | 슬롯 $B{\times}256{\times}128$ + 문맥(slot ≤128, 텍스트 임베딩 2560→384) → 속도장 + 이산 logits | 뱅크와 시뮬 사건 2–5M(D0), Garage(D0), 오디오 주변분포(D1) | 6.6×10¹⁷ | 15 + 증류 5 | 1개 4–8 ms, 1024개 약 4 s | 6 / <1 GB |
| **MoPri-S** | 32M | 10L, d448, 7헤드×64, 비AR | $x\in\mathbb R^{B\times T\times140}$ (T≤300, 30fps; 6D×22관절 + 루트 + 접촉) + 텍스트, 사건 트랙, 스타일, 뼈 길이, 키프레임 마스크 → $\hat x_0$ | CMU 원본, 100STYLE, Garage, Act-from-Text (D0/D2). 증강 ×10, MJPC 재시뮬 ×3 | 1.9×10¹⁸ | 44 + 13 | 1개 5–10 ms, 4096개 21–40 s | 7 / 4 GB |
| ID-Screen | 0 | 역동역학 + 접촉 QP (고전) | 후보 → 유령력 순위 | — | — | 0 | 4096개 4–8 s (CPU16) | — |
| **Feas-S** | 5M | 6L, d256 | 사건 집합 + slot → P(성공), 예상 비용 | SMC 성패(D0) | 5×10¹⁶ | 1 | <1 ms | <1 GB |
| **HeroProxy-S** | 14M | ViT 8L, d384, p16 @256², 헤드 4개 앙상블 | 7ch → 대역 DR 2개 + 16² DR 맵 + 불확실성 | Exhaust ≥1M (누적) | 1.5×10¹⁷ | 3.5 | INT8 4096장 약 1.5 s (+렌더 16 s) | 4 / 0.2 GB |
| **ChronoProxy-S** | 12M | 같음 | 크로노 합성 → $\Delta_\chi$ (GICC) | Exhaust (크로노) | 1×10¹⁷ | 2.5 | 같음 | 같음 |
| **MAN-S** | 1.5M | U-Net 3스케일 32/64/128, 수용영역 16² | G-buffer 40ch (+MAN ξ 8ch) → 잔차 RGB, $\tanh\cdot\tau$ | **Exhaust-to-Shader 쌍 2만** + DMD 5k 스텝 + R1-GAN(자체 실사 패치) | 16만 TC-eq | **약 41** | 4K 0.1–0.3 s | 교사와 동거 10–14 / 0.3 GB |
| **PercepNet-S** [신규] | 3M | latent CNN | 교사 latent 쌍 → 지각 거리 | Feature-Tap(TFP) 증류 | 1×10¹⁷ | 2 | <1 ms | 2 / 0.1 GB |
| PCGD-0 | 약 300 (NN 아님) | 재질쌍 30 × {e~Beta, μ~LogNormal, compliance, OU 토크} | 접촉 → 확률적 파라미터 | Garage L-3D + 오디오. SMC-ABC | CPU | 0 | — | — |
| **F_φ-S** (2b 이후) | **0.6M** | 잔차 GNN, MP 3, latent 64, k=8, z8 | 입자 그래프 → 잔차 가속도. **top-k에만 적용** | Garage L-3D + 시뮬 | 약 3×10¹⁷ (낮은 MFU) | 약 8 | 1만 노드 1–2.6 ms/스텝 | 2 / 0.3 GB |
| **Reader-S** | 26M | R-Flow 3M(RAFT-S식) / R-Kpt 18M(HR-lite, ImageNet 초기화 없음) / R-Seg 4M / R-Event 1M | 영상 → flow, 2D 키포인트, 동적 마스크, 사건 | 우리 렌더(D0, FlyingStuff) + BSRT + COCO kpt(D2; 상업 경로는 Garage 라벨) | 2–3×10¹⁸ (낮은 MFU) | 115–150 | 1080p 30–80 ms | 8–10 / 2 GB |
| R-Lift | 0 | 삼각측량, BA, PnP, MoPri prior analysis-by-synthesis | 2D → 사다리별 라벨 | — | CPU | 0 | — | — |
| **Denoiser-S** | 1M | OIDN식 U-Net | 저spp + albedo + normal → 디노이즈 | 2만 쌍 (16 vs 1024 spp, 내장 디노이저 off) | 렌더 15–37 + 학습 8 | 25–45 | 4K 0.15–0.3 s | 4 / 0.3 GB |
| 2c: Expr-S | 12M | 시간축 Transformer | mel → 블렌드셰이프 52 + viseme | **Dot-Face** 캡처(D1) | — | 10–20 | 실시간 | — |
| 2c 후반: TTS-S + Voc-S | 20M + 14M | Matcha/VITS급 + Vocos급, 규칙 G2P | 텍스트 → 음성 | 라이선스 한국어 단일 화자 | — | 200–400 | 실시간 | — |

- **합계.** 2a–2b는 약 123M(모델 10개 + ID-Screen), 2c까지 약 169M이다.
- **1세대 학습.** EPD 20 + MoPri 57 + Feas 1 + Proxy 6 + MAN-S 41 + PercepNet 2 + F_φ 8 + Reader 130 + Denoiser 35 + Chrono-LoRA 20 = **약 320 T4-h**. 2×T4로 6.7일이다. 판결에 따라 SR-LoRA 60–115 T4-h를 더한다(Phase 1과 공유).
- **선례.** MDM(약 18–25M [2차], 2080 Ti 1장 3일), Learned Motion Matching, GNS, OIDN, RAFT(5.3M, 합성 데이터로 from scratch). 모션, 물리, 디노이징은 원래 작은 모델의 영역이다.

### 5.2 모델별 핵심 설계

**EPD-S** [v1 EPD v2 축소]
- 사건 슬롯은 $e_i=(b,t,\mathbf x,k,p,q,\mathbf m,\ell)$이다. 참여자 $p,q$는 세계 slot pointer다.
- 스크립트 제약은 집합 공간에서 슬롯을 고정하는 inpainting으로 준다. Feas-S로 guidance를 건다.
- 범주별 간격 분포를 오디오 증인과 MMD로 맞추는 보조 손실을 둔다. 이 MMD는 게이지 불변 통계에만 건다.
- 라벨 등급(존재, 순서, 타이밍)별로 손실을 마스크한다.

**MoPri-S**는 3단계로 학습한다.
- **A단계.** 모캡 flow matching.
- **B단계.** MJPC 재시뮬로 데이터를 약 3배 늘린다.
- **C단계. TPRF RWR** [신규 조합]
  - TPRF = **Teacher Pose Restoring Force**.
  - 후보의 정지 순간을 절차적 인체로 렌더한다.
  - 교사로 σ_L1(형태 대역)에서 SDEdit을 한 뒤, R-Kpt로 키포인트가 얼마나 움직였는지 잰다.
  - 부자연스러운 포즈는 교사가 "고쳐서" 많이 움직인다. CG 룩은 대부분 고대역에 있어서 키포인트를 거의 움직이지 않는다. 그래서 DR보다 덜 오염된다.
  - 가중 회귀는 ID-Screen과 MJPC를 통과한 샘플에만 건다.
- **긴 모션.** 겹치는 창을 lockstep으로 동시에 디노이즈한다(시간 방향 MultiDiffusion). `order_purity_check`로 검사한다.

**BSRT (Band-Safe Reality Transfer)** [조합/신규]

$$\sigma^*=D^{-1}(b_y{+}1,0.9),\quad x_p=\mathrm{Dec}\big(\mathcal D_{\sigma^*}(\mathrm{Enc}(x_r);\xi_{MAN})\big)$$

- Turbo 경로에서는 σ*를 학습된 t로 스냅한다. Base 경로에서는 σ*를 그대로 쓴다.
- MAN 노이즈를 latent 격자로 적분해서 프레임 사이 세탁을 일관되게 만든다.
- DIS 정렬 잔차가 0.5 px를 넘으면 기각한다.
- 산출 쌍은 Reader-S 학습과 Exhaust-to-Shader 양쪽에 쓴다.

**MAN-S** [v1 유지·축소]
- Read-Only다. 알파와 기하를 출력하지 않고 $\lVert r\rVert\le\tau$로 제한한다.
- 상태 φ를 입력으로 받는다.
- 모션블러 구간에서는 서브프레임 2–4개에만 적용한다.

## 6. T4 위의 물리

### 6.1 처리량 (추정, E4에서 확정)

| 시뮬 | 규모 | 장치 | 처리량 |
|---|---|---|---|
| 강체 소형 | batch 4096 | T4 MuJoCo Warp | 3×10⁵–10⁶ scene-step/s |
| 강체 중형 (접촉 약 200) | batch 1024 / 16코어 | T4 / CPU MuJoCo | 2–8×10⁴ / 5×10⁴–2×10⁵ |
| 강체 대형 | batch 256 | T4 / CPU | 5×10³–2×10⁴ |
| Voronoi 파편 300 | batch 64–256 | T4 | 1–3×10⁴ |
| 휴머노이드 | batch 1024 / CPU | T4 / CPU | 3×10⁴–10⁵ / 3–10×10⁴ |
| MPM 30만 입자 | 1 | T4 | 시뮬 1초에 1–2분 |
| **FLIP 50만, 256³** | 1 | T4 | CFL 적응(dt 1–2 ms)이면 시뮬 1초에 0.5–3분. dt 0.2 ms 고정이면 5–12.5분 |
| Eulerian 연기 256³ | 1 | T4 | 시뮬 1초에 0.5–1분 |
| XPBD 천 2만 정점 / 헤어 1만 가닥 | — | T4 | 시뮬 1초에 2–10 s / 10–30 s |

- 유리는 MPM으로 하지 않는다. 파동속도가 5.3 km/s라서 dt ≲ 2×10⁻⁷ s, 0.5초에 260만 스텝이 필요하다.
- 대신 Voronoi 사전 파쇄(seed = 사건 마크) + 강체로 한다.
- MJX(JAX)는 sm_75 지원이 불명이다. X0에서 확인되기 전에는 쓰지 않는다.

### 6.2 Milestone SMC

| 과제 | N | scene-step | 뱅크 없음 | 뱅크 있음 (스텝 5배↓) |
|---|---|---|---|---|
| 공, 병 (3 s) | 4096 | 1.8×10⁷ | T4 20–60 s | 동일 |
| 도미노 50, 핀볼 (10 s) | 1024 | 1.5×10⁷ | T4 3–13분, CPU 1.3–5분 | 0.6–2.6분 |
| 미니 Rube Goldberg (15 s) | 256 | 5.8×10⁶ | T4 5–19분 | 1–4분 |
| 와인잔 (0.5 s) | 강체 256 → FLIP top-8 | — | 강체 <1분 + FLIP 2–12분(CFL 적응) | 동일 |
| 캐릭터 10 s | best-of-4096 | — | ID-Screen 4–8 s + MJPC-lite 8개 5–17분 | — |

**Day-1 가중평균.**
- 소 60%·중 30%·대 10%로 섞으면 블록당 231 s(T4-eq)다.
- 뱅크가 붙으면 65 s다.

### 6.3 Multi-fidelity SMC (유체, MPM)

1. 1/8 해상도 프록시(3만 입자)와 proxy critic으로 거시 결정 변수를 고른다.
2. top-8만 원 해상도로 다시 시뮬한다.
3. 사건 시각은 강체 운동이 결정하므로 해상도가 바뀌어도 거의 유지된다.
4. 미세 혼돈은 Hero-Instant의 다양성으로 쓴다.

### 6.4 Chaos Bank [조합; Many-Worlds Browsing'07, motion matching]

- 놀고 있는 CPU로 템플릿별 10⁵–10⁶ rollout을 굴린다.
- 사건 그래프 시그니처와 타이밍 벡터를 kd-tree로 색인한다.
- 결정론적 CPU MuJoCo이므로 rollout당 64 B(템플릿, 파라미터, seed, sim 버전)만 저장한다. 10⁶개면 64 MB다.
- 같은 뱅크가 EPD-S와 Feas-S의 학습 데이터가 된다.
- **Bank-First:** 시드기 0–4주에 먼저 만든다.

### 6.5 캐릭터 물리: MJPC predictive sampling [부품, PHC는 2d로 미룸]

- 접촉창에서만 MJPC-lite를 쓴다. rollout 128개, 레퍼런스 warm start.
- 잔여력은 RGF 예산 안으로 제한한다.
- 학습 정책(PHC급)은 샘플 10⁹–10¹⁰이 필요하므로 2d로 미룬다.

## 7. 16 GB × 2 배치

| 모드 | GPU-A "교사 전용" | GPU-B "학생, 시뮬, 렌더" | CPU (16 vCPU 이상, 64 GB) |
|---|---|---|---|
| **F: 플라이휠** | 판정: Z-Image INT8 6.2 + VAE fp32 0.34 + 활성(512², B4) 2–3 = **약 9–10 GB**. BSRT·DMD 단계만 fp16 12.3 GB로 스왑(약 1.3 s), B1–2 + checkpointing으로 약 14.5 GB | 학생 학습 6–8 + MuJoCo Warp 1–2 + INT8 proxy 0.3 + 래스터(L2, 데이터 렌더) 1–2 = **10–13 GB**. Qwen3-4B INT8 4 GB는 새 프롬프트가 있을 때만(또는 CPU) | MuJoCo 16 프로세스, ID-Screen, 뱅크 구축, GLOMAP(SIFT), 교사 가중치 pinned 26 GB |
| **H: 히어로 샷** | 교사 slice polish. 512² band-iso 또는 768² 타일 | FLIP 1.5 + 3DGRT BVH 3–4 + MAN-S·Denoiser 0.3 + 4K AOV 1 = 7–9 GB | 모달 FEM, 오디오 |
| **T: 학습 버스트** | DDP 2-way로 같은 모델 복제 | 같음 | 데이터 로더 |

- 기본은 DDP가 아니라 역할 분담이다. 두 카드 사이로는 latent, 상태, 점수만 오간다.
- 32M 모델의 fp16 gradient(64 MB)를 PCIe Gen3로 allreduce하면 10–25 ms다.
- **Kaggle 모드:** GPU 구성은 같다. CPU 작업은 외부의 저가 16 vCPU 머신으로 뺀다. GPU 쿼터를 T4×2가 몇 배로 소모하는지 확인해야 한다.

## 8. 세계와 외관

### 8.1 Reveal-Bundle Demand [신규 조합]

$$\mathcal R=\bigcup_\tau\mathrm{Vis}(W(t(\tau)),\Gamma(\tau))\setminus\mathcal V_0$$

- 실현된 사건 집합 $E$에 실제로 나오는 편집과, 세계선 원뿔 안에서 드러나는 면에만 반사실 TC를 쓴다.
- 탁상 장면에서 slot이 50개이고 사건이 1–5개를 건드리면 TC가 10–50배 줄어든다.
- 비회귀: $\mathcal R$의 거친 레벨은 bedrock에 이미 있다. 같은 레벨 안에서 lockstep으로 푼다.
- 새 테이크가 새 면을 드러내면 copy-on-write 브랜치에서 그 면의 미세 대역만 연다.
- 파괴 단면은 절차적 볼륨 재질로 만들고, 히어로 컷만 교사로 polish한다.

### 8.2 캐릭터 외관

- 기반 메시는 MakeHuman류 절차적 인체(CC0)이고, 자체 22관절 BVH 스켈레톤을 쓴다.
- identity LoRA(r8, 7.4M)는 캐릭터당 5–10 T4-h, 한 번만 든다.
- 얼굴: v0는 블렌드셰이프와 규칙 viseme, v1은 Dot-Face로 학습한 Expr-S다. 눈은 고전 각막 굴절 모델로 만든다.

### 8.3 음향: 학습 파라미터 0개

| 소리 | 방법 |
|---|---|
| 충돌, 타격 | 사건 충격량 J → FEM 고유모드(CPU, 에셋당 수 초–수 분) |
| 파괴 | 확률적 임펄스 |
| 액체 | Minnaert 기포 확률 합성 |
| 전파 | image source + RT 코어 음선 추적 |
| 바이노럴 | 측정 HRTF |
| 음성 | 녹음 → TTS-S |

## 9. 로드맵

GPU 몫은 M0–3 10–15%, M3–9 30–50%, M9 이후 50% 이상이다.

| 단계 | 시점 | 새 부품 | 병목 | 우회 | 샷당 증분 비용 | 승리 조건 (v1 그대로) |
|---|---|---|---|---|---|---|
| **2a Frozen Action & Speed Ramp** | M3–M9 (시드 M0–M3) | MAN-S, Proxy, Denoiser-S, (Chrono-LoRA) | slice TC, RT 렌더, 아레나 콘텐츠 300개(1,100–1,700 T4-h) | Reveal-Bundle, 캐스케이드, Voronoi, CFL 적응 FLIP, 세계 30 × 사건 10, 1080p 정규화 | 3.5–7 T4-h | BUPS-FA ≥70%, Appearance Turing ≤65%, GeCo ≤0.03, TSR 밴드 안, RST ≤1프레임 |
| **2b 강체·관절체 + 살아있는 세계** | M8–M16 | PCGD-0, F_φ-S, EPD-S, Feas-S, Reader-S | sim look, 볼륨 렌더, Physics-IQ의 lift 의존 | Garage ×3, 오디오 반발 분포, 뱅크, Lift Ladder, 절차적 생명 | 2–4 T4-h | Physics-IQ·VideoPhy-2 1위, Motion Turing ≤60%, CSI KL ≤0.5×, BUPS ≥65%, 일반 Arena ≥50%, GCM ±20% |
| **2c 캐릭터·연기** | M14–M28 | MoPri-S v2(TPRF), Expr-S, identity LoRA, R-Kpt v2 | 사람 데이터, 천·헤어 | Act, don't prompt. Garage 모캡 연 20–50 h, Act-from-Text | 4–8 T4-h + 캐릭터당 5–10 | 연기 축 ≥55%, 멀티숏 연속성 ≥70%, 얼굴 ≤75%, foot skate <2 cm/s, 관통 <1%, eyeline <3° |
| **2d 완전 4D + 인터랙션** | M26–M42+ | 손 조작(MJPC → 학습 정책), 다중 행위자 EPD | 손-물체, 에셋 다양성 | Merkle 청크 스트리밍, §13 다이얼 | — | 모든 범주 Arena ≥60%, Revisit 기울기 ≈0, 인과 정확성 1위 |

- **2a와 2b가 거의 밀리지 않는 이유.** 비용이 시뮬, 렌더, TC에 있는데, 이 항목들은 T4가 상대적으로 덜 손해 본다.
- **2c가 밀리는 이유.** FLOPs가 아니라 사람 데이터가 병목이다.
- **범주 정복 순서.**

| 순서 | 범주 |
|---|---|
| 1 | Frozen Action (유리, 물 왕관) |
| 2 | 강체 연쇄 |
| 3 | 파괴 |
| 4 | 액체 hero |
| 5 | 식생과 바람 |
| 6 | 운동선수 단독 |
| 7 | 댄스 |
| 8 | 2인 대화 |
| 9 | 손 조작 |

- Kaggle 모드에서는 2a를 M12–M18에 노린다.

## 10. 플라이휠

```
 ┌(1) 제안: ShotScript × 세트피스 × EPD-S/MoPri-S (Feas-S, Chaos Bank 검색)
 │(2) 실현: SMC, PCGD-0, MJPC-lite   ← 학습 0
 │(3) 판정: 불변량 → RGF/ID-Screen → INT8 proxy → TC (불확실성 상위 25% + 무작위 10%)
 │(4) 커밋: Block Bank + Exhaust (+ Feature-Tap) + Lineage
 │(5) 증류: EPD-S, MoPri-S(RWR), Feas-S, Proxy, PercepNet, Reader-S, MAN-S(Exhaust-to-Shader)
 │(6) 읽기: Event-Demand Lift (Lift Ladder) → p_real, 통계, PCGD-0 재보정
 └(7) 가속: 제안↑ → N↓ / proxy↑ → TC↓ / 같은 하루에 블록↑
```

AlphaZero와 대응시키면 다음과 같다.

| AlphaZero | 우리 |
|---|---|
| 탐색 | SMC, 캐스케이드 |
| 정책 | EPD-S, MoPri-S |
| 가치 | Feas-S, proxy |
| 심판 | 물리, 교사, 실통계 |

### 10.1 1회전 시간표 (상시 2×T4 + 16 vCPU, Phase 2 몫 2/3)

| 단계 | 장치 | 회전 0 (350블록, 뱅크 없음) | 정상 상태 (1,000블록, 뱅크와 Feas-S) |
|---|---|---|---|
| 제안 | GPU-B | 0.4 h | 1 h |
| 실현 | CPU 60% / GPU-B 40% | 231 s × 350 = 22.5 h-eq (CPU 13 + GPU 9) | 65–80 s × 1,000 = 같은 시간 |
| L2 렌더 + proxy | GPU-B | 0.1 h | 0.3 h |
| L3 | GPU-A INT8 | 350 × 64 TC = 3.1 h | 1,000 × 64 × 커버리지 60% = 5.3 h |
| 데이터 렌더 | GPU-B | 0.2 h | 0.5 h |
| BSRT | GPU-A fp16 | 4k × 4 TC = 3.8 h | 3.8 h |
| 증류 | GPU-B | 6 h | 10 h |
| lift | CPU | 5 h | 10 h |
| **합계** | | GPU-A 7 h / GPU-B 16 h / CPU 18 h | GPU-A 9–13 h / GPU-B 21 h / CPU 23 h → **1일/회전** |

- 시드기(몫 10–15%)에는 4–6일에 1회전이다.
- Kaggle은 "세션 = 세대"로 운영한다. 주 1회전, 규모 1/3이다.

### 10.2 래칫 지표

| 지표 | 회전 0 | 10회전 목표 |
|---|---|---|
| SMC 제약 충족률/입자 | 0.5–2% | 5–15% |
| 블록당 L3 TC | 64 | 8–16 (커버리지 ≤25%) |
| 하루 블록 | 350 | **4,000–8,000** (상단은 c_TC ≤0.5 s일 때) |
| proxy–교사 Spearman | 약 0.5 | ≥0.85 |
| Reader Gold 오차 | 기준선 | −30% |
| search gain | 큼 | 단조 감소 |
| ID-Screen 재현율 | 측정 | ≥0.95 |

### 10.3 붕괴 방지

- 배치마다 실데이터 앵커를 30% 이상 둔다.
- held-out Gold 게이트에서 2% 넘게 퇴행하면 롤백한다.
- 엔트로피와 커버리지를 감시한다.
- 무작위 L3를 10% 둔다.
- Exhaust에 교사 버전, 해상도, 정밀도 태그를 붙인다.
- **판정 정밀도 드리프트 감시:** 주 1회 fp16 1024²로 재판정한다.

## 11. v1 장부

| v1 항목 | 판정 | 대체와 이유 |
|---|---|---|
| Native WFM, T1–T3 | 폐기 | $10^{23}$–$10^{25}$가 필요하다. Tiny Federation + 탐색으로 대체 |
| PLAC, 비디오 증인, PC-ELBO, 진술 그래프 | 폐기 (규칙) | 크로노 GICC가 대조와 QBC를 재활용한다 |
| Lift 스택 (DA3, VGGT, GVHMR, SAM2 등) | 폐기 | SIFT-SfM, DIS, 배경차분 + Reader-S + Lift Ladder |
| 정적 5천만·동적 2천만 클립 lift | 축소 | 연 1e4–1e5 Event-Demand |
| Gold Set ($0.3–0.6M) | 축소·미룸 | Garage + Sensor Purity. 1000fps는 2b에 대여 |
| AV-Physics Timing Solver, Blur-as-Measurement, RGF | 유지 | 오디오 간격 증인 추가, RGF는 ID-Screen과 공유 |
| EPD v2 | 축소 | EPD-S 28M |
| INTENT/EMOTION의 VLM 판정 | 폐기 | 사람 + 자체 소형 분류기 |
| PCGD (F_φ 0.3–1B) | 축소 | PCGD-0 → F_φ-S 0.6M (루프 밖) |
| Milestone SMC + Hero-Instant | 유지·조정 | 뱅크, CPU 경로, multi-fidelity |
| Edit-Closed full | 축소 | Reveal-Bundle Demand |
| MAN, slice DM | 유지·축소 | MAN-S 1.5M + Exhaust-to-Shader |
| SMPL-X, FLAME, **AMASS/HumanML3D** | 미룸 (판결 필요) | 자체 스켈레톤, CMU 원본, Garage |
| PHC, MaskedMimic | 미룸 (2d) | MJPC-lite |
| Codec-Avatar급 얼굴 | 미룸 | Dot-Face + Expr-S |
| 사전학습 TTS | 폐기 → 미룸 | 녹음 → TTS-S (규칙 G2P) |
| **SDXL급 교사 가정** | **폐기** | 단일 교사 규칙에 따라 Phase 1과 같은 Z-Image 계열 |
| Photo-Slice DR 보상 | 변경 | TPRF |
| Temporal DoF, 시간 PSH, Gabor 물결 | 유지·앞당김 | 렌더 측만 바꾸면 되어 싸다 |
| Builder–Reader self-play | 중심화 | BSRT |
| GeCo, VBench, Physics-IQ 등 | 평가 전용 | CI로 시스템 경로에서 격리 |
| H1′ go/no-go | 재정의 | H1-T4: $10^{18}$–$10^{20}$ 구간의 기울기 |

## 12. 이 제약 덕분에 좋아지는 점

1. **반복 속도.** 30M 모델 학습은 몇 시간이면 끝난다. 주당 수십 번 실험할 수 있다.
2. **인수분해가 필수가 된다.** 실패를 부품 단위로 가를 수 있고, Factorized Turing과 1:1로 대응한다.
3. **차원 해석이 정직해진다.** 비디오 판사가 없으니 "사진 한 장으로 무엇을 알 수 있는가"를 명시해야 했다. 그 결과가 GICC다.
4. **검증이 공짜로 생긴다.** 역동역학 한 번이 RGF, ID-Screen, PCGD 보정을 동시에 먹인다.
5. **배포가 쉬워진다.** 교사를 뺀 스택 전체가 소비자 GPU에서 돈다. 감독용 실시간 프리뷰가 가능하다.
6. **학습 0 부품은 감가상각이 없다.** 물리, 음향, 광학은 컴퓨트가 늘어도 재학습할 필요가 없다.
7. **데이터 자산과 계보 장부 자체가 해자다.** Block Bank, Exhaust, Gold, Lineage.
8. **전기요금.** 140W, 전기장판 두 장이다.

## 13. 스케일 원칙

- **S1.** 텐서 계약을 고정한다. 모델 크기는 구현 디테일이다.
- **S2.** μP와 같은 데이터 파이프라인. 30M에서 찾은 하이퍼파라미터가 3B로 넘어간다.
- **S3.** 탐색과 상각 사이의 다이얼. 컴퓨트가 늘면 먼저 학생을 키우고 N과 L3 커버리지를 줄인다.
- **S4.** Merkle 블록, Exhaust, Lineage는 모델과 독립이다.
- **S5.** 교사는 Dreamer 계약 슬롯에 끼운다. 가격표와 Exhaust는 버전 태그로 다시 보정한다.
- **S6.** 고전 부품은 정확도 다이얼만 돌린다.
- **S7.** fp16-safe 코드는 bf16, fp8에서 그대로 돈다. INT8 critic은 배포 계층에만 둔다.
- **S8.** 스케일 순서.

| 컴퓨트 | 무엇을 키우는가 |
|---|---|
| ×10 | 1024² 판정, MoPri 300M, N ×10 |
| ×100 | Phase 1 학생 1B, EPD와 Reader 300M |
| ×1000 | v1 T1 경로 |

## 14. 첫 90일 (Phase 2 시드, 약 470 T4-h)

| 기간 | 실험 | 예산 (T4-h) |
|---|---|---|
| 1–4주 | **E1 TJC, E2 GICC, E3 Verified C–P, E4 SMC+Bank, E5 BSRT** (Part 4) | 167 |
| 2–8주 | P2-X2 Garage + PCGD-0 + RGF. 성공: Motion Turing ≤60%, RGF AUC(실제 vs 순진한 시뮬) ≥0.75, (실제 vs PCGD-0) ≤0.6 | 10 + CPU |
| 5–10주 | P2-X4 MoPri-S 전체 + TPRF. 성공: 보행 Motion Turing ≤60% | 50 |
| 6–12주 | P2-X5 Reader-S 전체. 성공: R-Flow가 DIS 대비 EPE −30%, R-Kpt COCO AP ≥65, BSRT 이득 −15% | 120 |
| 8–13주 | P2-X6 Frozen Action 1샷. 성공: 미니 BUPS-FA ≥60%, GeCo ≤0.03(평가 전용) | 60 |
| 1–6주 | P2-X7 Spectrum Report(고전 추적) | 5 + API $2–5k |
| 6–13주 | MAN-S, Denoiser-S, PercepNet-S | 60 |

- **게이트.**
  - 30일: E1 + (E2, E3, E4 중 2개).
  - 90일: X3(E4)·X6·E2 중 2개 이상을 통과하면 2a 본진행이다.
  - 실패한 실험은 그 실험의 업그레이드를 다음 분기 1순위로 올린다.
- **Kaggle 모드.** E1 축소판, E2, E4만 진행한다.
- **돈.** 폰 $0–2k, LED·Sensor Purity 보드 $100–300, API 평가 $2–5k, 사람 평가 $5–15k.

## 15. 비회귀 점검 (순차 아웃페인팅 금지)

| 부품 | 판정 | 근거 |
|---|---|---|
| EPD-S | 통과 | 사건 집합 디퓨전. 슬롯은 병렬이고 스크립트 제약은 집합 공간 inpainting이다 |
| MoPri-S (긴 모션) | 통과 | 창들을 lockstep으로 동시에 디노이즈한다. `order_purity_check`로 OIE를 검사한다 |
| 실현 (시뮬) | 통과 | 상태공간 ODE 적분이지 픽셀 생성이 아니다 |
| Judge Cascade, GICC, ID-Screen, TPRF | 통과 | 판정은 읽기다. 다른 영역 생성의 조건으로 쓰지 않는다 |
| Reveal-Bundle Demand | 통과 | 세계 공간에서 처리하고 거친 레벨이 이미 있다. 같은 레벨에서 lockstep |
| BSRT, Exhaust-to-Shader | 통과 | 학습 데이터 전용이다. 렌더 경로에 없다 |
| MAN-S, Denoiser-S | 통과 (I6) | 결정론적이고 알파·기하 출력이 없다. 물질에 고정된 노이즈만 쓴다. Denoiser-S는 잔차 크기를 제한한다 |
| Chaos Bank | 통과 | 상태공간 초기 입자로만 쓴다 |

## 16. 리스크와 업그레이드 (후퇴 없음)

| # | 리스크 | 감지 지표 | 업그레이드 |
|---|---|---|---|
| R1 | 크로노가 모양에 둔감하다 | E2 AUC <0.65 | Chrono-LoRA, 라이트 페인팅, (K,Δ) 재설계 → 미감 전용으로 강등(H7) |
| R2 | sim look이 남는다 | Motion Turing >65% | Garage ×3–10, F_φ-S, OU 미세 토크, 1000fps 대여 |
| R3 | MoPri 품질과 다양성 부족 | foot skate, 2AFC | best-of-N 확대, TPRF, 64–80M, Act 비중 확대 |
| R4 | Reader가 웹 영상을 못 읽는다 | Gold 오차, 사이클 기각률 | Lift Ladder 하향, σ* 재보정, 쉬운 범주부터 |
| R5 | TC 초과 | 블록당 TC가 줄지 않음 | 가격표 재측정, INT4, 커버리지 게이트 강화, Feature-Tap (교사 교체는 하지 않음) |
| R6 | 라이선스 | 법무 게이트 | CMU, 100STYLE(확인), 자체 캡처, FlyingStuff |
| R7 | sm75 라이브러리 미지원 | X0 | CPU MuJoCo, MuJoCo Warp, Taichi. 툴체인 Docker 고정 |
| R8 | Kaggle 쿼터와 vCPU | 회전 주기 | 외부 CPU 머신, 범위를 2a로 한정 |
| R9 | fp16 불안정 (학생과 교사) | NaN, loss scale 붕괴 | QK-norm, z-loss, fp32 softmax/LN, clipping. 교사 VAE는 fp32 |
| R10 | 플라이휠 붕괴 | Gold 퇴행, 엔트로피 하락 | 앵커 30%, 무작위 L3 10%, 롤백 |
| R11 | 아레나 콘텐츠가 병목 | 2a 일정 | 세계 재사용, 1080p 정규화 |
| R12 | 일반 T2V 의미 범위에서 진다 | 일반 Arena | 범주 순서를 지킨다. 목표는 그대로 |
| **R13** | 교사 판결 불가, Turbo DR 불량 | E1 H1b | SR-LoRA (Phase 1 공동) |
| **R14** | 게이지 누출 (가짜 단서) | 대조 AUC >0.6 | 노출 개수·블러 정규화, 왜곡 가족 재설계 |
| **R15** | 계보 누수 | Lineage CI 실패 | 해당 아티팩트 격리 후 재생성 |
| **R16** | ID-Screen 재현율 부족 | r <0.95 | 접촉 일관성 항 추가, MoFeas-S, L2 확대 |

## 17. 창업자가 결정할 질문

0. **Z-Image Base와 Turbo를 "이미지 모델 하나"로 보는가?**
   - 기본 요청은 "예"다. 판정은 Base, 샘플은 Turbo로 쓴다.
   - "아니오"면 Turbo + SR-LoRA로 간다.
1. **D3 데이터.** 기본값은 금지, 평가와 연구 전용이다. AMASS와 HumanML3D도 여기에 포함하는가?
2. **SMPL-X와 FLAME은 에셋인가, 금지된 모델인가?** 기본값은 자체 스켈레톤이다.
3. **Qwen3-4B 임베딩을 작은 모델의 조건으로 써도 되는가?** 생성 용도는 금지한다.
4. **캡처 기기 내장 ML.** 기본값은 Sensor Purity를 통과한 프레임만 쓰고 ARKit은 금지다.
5. **2c 기본 방향을 "Act, don't prompt"로 할 것인가?**
6. **Phase 2 GPU 몫** 15%, 40%, 50% 이상 안을 승인하는가?
7. **호스트 사양** 16 vCPU 이상, 64 GB 이상을 확보할 수 있는가?
8. **Garage 예산과 1000fps 대여 시점.**
9. **COCO와 FlyingThings의 상업 경로 사용.** 기본값은 연구 전용이다.

## 18. 신규성 장부

| 확신도 | 항목 |
|---|---|
| **중상** | 크로노포토 증인 + BIS + **GICC**(유사변환군 직교 왜곡 + 게이지 음성 대조). 이미지 교사를 "시간의 모양" 판정기로 쓴다 |
| **중** | Coverage–Precision Split v2(검증 캐스케이드 포함 경제학), **Gauge-matched Lift / Lift Ladder**, BSRT, **Exhaust-to-Shader**, **TPRF**, Reveal-Bundle Demand, TC 가격표 + Teacher Exhaust + Judge Cascade, 데이터 순도 D0–D3 + **Lineage Ledger** |
| **중하 / 조합** | ID-Screen(역동역학 유령력 사전 스크린), One-Teacher Two-Mode / SR-LoRA, **Sensor Purity Test**, Dot-Face, Act-from-Text, 오디오 간격 증인, Event-Demand Lift, Garage Gold, 세션 = 세대, 실리콘 전 유닛 스케줄링 |
| **부품 (정직하게 차용)** | Many-Worlds Browsing, motion matching, MuJoCo MPC, MDM, GNS, OIDN 구조, RAFT 구조, Voronoi 파쇄, Minnaert, Zheng & James, Stam, Tessendorf, MakeHuman, V-HACD, COLMAP/GLOMAP(SIFT), DIS, ChArUco, μP, Nunchaku/SVDQuant, 커뮤니티 de-distillation. Move.ai와 Wonder Studio는 선례로만 언급한다 |

> **마무리.** "그리지 말고 찍어라"에 2×T4판이 두 줄을 더한다. **"배우지 말고 굴려라, 상상하지 말고 찍어라."** 그리고 이번 프리모템이 한 줄을 더 붙인다. **"사진이 모르는 것은 사진에게 묻지 마라."**
> - 눈금은 뉴턴과 마이크가 잰다. 모양은 사진이 판정한다. 서사는 사람이 정한다.
> - 검증까지 싸야 작은 생성기 × 큰 탐색이 이긴다.
> - 그렇게 하면 비싸게 배우는 부분은 약 123M 파라미터, 2×T4로 약 일주일이다. 나머지는 전부 탐색, 판정, 촬영에 쓴다. 개소리 같지만 계산은 맞는다. 이번엔 검증 비용까지 넣고 맞췄다.

---

**참조 파일.**
- 원 제안서 전문: `docs/v2-2xT4/lenses/04-phase2-flywheel.md`
- 검산 스크립트: `작업 중 사용한 검산 스크립트(레포 미포함)`
- 기준 문서: `docs/PARADIGM.md`
- 병행 렌즈: `docs/v2-2xT4/lenses/01-compute-budget-solver.md`, `docs/v2-2xT4/lenses/02-tiny-foundation-models.md`

**출처 (이번에 추가 확인한 것).**
- [Z-Image Base(비증류) 공개, comfyui-wiki](https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release)
- [Z-Image Base·Edit 가이드](https://localaimaster.com/blog/z-image-base-edit-guide)
- [100STYLE 논문 (Xsens 캡처, 1,125분)](https://arxiv.org/pdf/2201.04439), [100STYLE Zenodo](https://zenodo.org/record/8127870), [100STYLE CC BY 4.0 표기 (2차, 재판매처)](https://booth.pm/en/items/8872639)
- [PyTorch CUDA 13 / Turing 지원 공지](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432), [NVIDIA JAX 26.01 릴리스 노트](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-26-01.html)
- [Z-Image Turbo 해상도 (1024 네이티브, 2차)](https://help.scenario.com/en/articles/z-image-turbo-models-the-essentials/), [fal Z-Image Turbo](https://fal.ai/models/fal-ai/z-image/turbo)
