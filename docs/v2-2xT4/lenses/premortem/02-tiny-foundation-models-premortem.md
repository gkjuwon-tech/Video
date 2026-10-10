> **출처 메모.** 2×T4 + 수십M from-scratch 제약으로 다시 돌린 설계 워크숍에서 [작은 파운데이션 모델 + 데이터 엔진] 렌즈 담당이 쓴 건설적 프리모템과 개정안이다. 최종 결정은 [`docs/PARADIGM-v2.md`](../../../PARADIGM-v2.md)를 따른다. 이 문서는 Higgsfield Soul 하이브리드 결정 **이전**에 작성됐다.

---

# 렌즈 05 건설적 프리모템: 2×T4 위의 작은 파운데이션 모델

> 기준 자료: `docs/PARADIGM.md` v1.0, 렌즈 05 원안(tiny-foundation-models), T4 리서치(2026-10-10).
>
> 표기 규칙:
> - 출처 등급: **[검증]** 1차 출처 확인 / **[추정]** FLOPs·대역폭으로 계산, 실측 필요 / **[신규]** 이 제약 때문에 새로 생긴 아이디어 / **[조합]** 부품은 있고 결합이 새로움 / **[부품]** 차용.
> - 시간 단위: 따로 적지 않으면 T4 1장 기준이다. "2×T4 h"는 두 장을 동시에 쓴 벽시계 시간이다.
> - **NFE-eq**: 교사 1 NFE를 1024² 기준으로 환산한 단위다. 768² 1회 = 0.55, 512² 1회 = 0.26 (FLOPs 비율).

---

## 0. 판정 요약

1. **원안의 뼈대는 맞다.** 화폐는 교사 NFE라는 관점, 접착제 가설, 기생 학습(Feature Tap Bus), 교사 운송(teacher transport)과 결정 곡선 라벨 마스크, 라벨 계보 장부, 관할 이양(Federal Devolution)은 그대로 유지한다.
2. **숫자 하나가 전체를 흔든다. "교사 1 NFE = 1.4 s(SDXL급)"이다.**
   - SDXL에 대해서는 비관적이다. 실측 앵커를 역산하면 0.6–0.9 s다.
   - 이기는 데 필요한 6B급 교사에 대해서는 2.5배 낙관적이다. Z-Image-Turbo는 1024²에서 3.1–4.1 s다.
   - SDXL(Elo 678)을 기본값으로 쓰면 미감 목표가 구조적으로 막힌다. 상용 1위 대비 기대 승률은 SDXL 5%, Z-Image 19%다.
3. **개정안의 핵심은 더 비싼 교사를 쓰면서 세계당 교사 비용은 줄이는 것이다.** 세 가지 장치를 쓴다.
   - 판정 해상도 분할(Judge-Resolution Demand Split) [신규]
   - 투기적 대역 커밋(Speculative Band Commit) [신규]
   - 재질 견본 은행(Material Swatch Bank) + 예시 조건 LBO(ICX-LBO) [신규/조합]
   - 결과: 방 세계 하나에 드는 교사 비용이 약 2,900에서 약 760 NFE-eq로 줄고(×3.8), 관할 이양 후에는 약 360이 된다. Z-Image fp16 기준 0.35–0.75 T4-h다.
4. **PWMT(테스트타임 학습)는 1순위에서 내린다.** 블러 또는 타일링, GPU1 경합, 수요 의존성이 문제다. PWMT의 통찰(세계 안에서 교사를 amortize, 재질 정체성 유지)은 Swatch Bank + ICX로 옮긴다. 덤으로 native 모드의 T8 비트 동일성이 복구된다.
5. **숨은 위반 15곳을 찾았다.** 대표 예시는 v1 SelfCaption(VLM), v1 ℒ_ocr(OCR), COLMAP 사전학습 vocab tree와 ALIKED, 폰 ISP의 ML 처리, 서드파티 Lightning LoRA, NC 데이터셋 학습, Fun-ControlNet, P2 포즈 추정기다. 15곳 모두 from-scratch 모델이나 고전 알고리즘으로 대체한다.
6. **30일 결정적 실험 5개에 약 200 T4-h가 든다.** 상시 30일 용량(1,440 T4-h)의 14%이고, Kaggle 모드로도 돌릴 수 있다.

> 비유: 교사는 판사, 수십M 모델은 서기다. 판사는 시간당 1,030건만 본다. 서기가 초안을 쓰고 판사는 도장만 찍는다.

---

## 1. 숫자 재검증 (Task 1)

### 1.1 계획 상수: 원안 vs 리서치 vs 개정

| 항목 | 원안 | 리서치 | 개정값 | 판정 |
|---|---|---|---|---|
| 소형 모델 학습 실효 처리량 | 12 TF/장 | 15–20 TF/장 (GEMM 위주) | 2D conv와 xfmr는 12, 3D conv는 6–8 | **유지.** 소형 모델과 3D conv는 리서치 하단보다 낮다 |
| 90일 FLOPs | 1.9×10²⁰ | 2.3–3.1×10²⁰ | 소형 모델 1.9×10²⁰, 큰 GEMM 2.3–3.1×10²⁰ | 둘 다 맞다. 용도가 다르다 |
| v1 대비 배율 | 10⁴배 작다 | 약 1/400 | 카탈로그 FLOPs는 v1 CWM/LBO의 약 **1/6,700** | 원안의 "10⁴"는 반올림 과장이다 |
| roofline ridge | 203 FLOP/B (피크) | HGEMM 실측 41.6 TF | **130(실측)–203(피크)** | d=256 xfmr는 memory-bound가 아니라 경계선이다. 규칙(d≥384, C≥64)은 유지 |
| SDXL 1 NFE @1024² | 1.4 s (MFU 7%) | Fooocus 역산 약 0.55 s | **0.6–0.9 s (계획 0.75)** | **틀림.** 1.9배 과대 |
| SDXL UNet FLOPs | 6.6 TF | — | 층별 계산 6.2–6.6 TF | 맞음 |
| Z-Image-Turbo 1 NFE | "6B급 교체 옵션"만 언급, 비용 미반영 | 1024² 3.1–4.1 s | 1024² **3.5** / 768² **2.0** / 512² **1.0** s. INT4 W4A4는 ×0.5–0.6 (검증 필요) | **누락.** 교체 경로 비용이 반영되지 않았다 |
| VAE 왕복 @1024² | 1–1.5 s | 미측정 | enc 약 5 TF + dec 약 8.5 TF → **1.1–1.7 s**. 768² 0.6–0.95, 512² 0.3–0.4 [추정] | 대체로 맞음. decode 한 번이 SDXL NFE 1.3회분이다 |
| NFE/T4-h | 2,570 | — | SDXL 약 4,800. Z-Image 1024² 1,030 / 768² 1,800 / 512² 3,600 | 수정 |
| 교사 운송 1개 | 4.3 s → 2만/T4-day | — | Z-Image 512² 3 NFE + VAE = 3.4 s → **2.5만**. 768² 6.85 s → **1.26만** | 수정 (Turbo는 중간 t에서 3스텝) |
| band-iso 뷰 1개 | 6.6 s, 1.3만/day, 1024² = 64² 타일 256개 | — | Z-Image 768² 4 NFE + VAE = 8.85 s → **9.8천 뷰/T4-day** (뷰당 64² 타일 144개, 하루 141만 타일) | 수정 |
| Kaggle | 780 T4-h | 약 780 | 780. 쿼터를 GPU 장 단위로 세면 **최악 390** | 하한 추가 |

SDXL UNet 6.6 TF 계산 근거(1024², latent 128²):
- 128²/320ch conv는 약 0.4 TF다.
- 64²/640ch(transformer 깊이 2)는 약 1.4 TF다.
- 32²/1280ch(transformer 깊이 10, down·mid·up)는 약 4.4 TF다.
- 합계 6.2 TF에 업/다운샘플을 더하면 약 6.5 TF다.

Z-Image 계산 근거:
- 공식: FLOPs ≈ 2·6.15e9·N + 4·N²·3840·30
- 1024²: N≈4,300이고 61 TF다.
- 768²: N≈2,450이고 34 TF다.
- 512²: N≈1,200이고 16 TF다.
- 실효 처리량 15–20 TF/s를 적용했다.

### 1.2 교사 선택이 미감 상한이다 (Elo 기대 승률)

| 교사 | Elo | 라이선스 | vs 상용 1위(1198) | vs SDXL | 1024² 1 NFE (T4) | 비고 |
|---|---|---|---|---|---|---|
| SDXL 1.0 | 678 | OpenRAIL++ | **4.8%** | — | 0.75 s | 빠르고 fp16이 안전하지만 상한이 낮다 |
| FLUX.2 klein 4B | 863 | Apache 2.0 | 12.7% | 72% | 2.0–2.7 s | 4-step, I2I와 멀티 레퍼런스 내장 |
| **Z-Image-Turbo** | **941** | **Apache 2.0** | **18.6%** | **82%** | 3.1–4.1 s | 8 NFE, CFG 없음, bf16 네이티브 |
| Qwen-Image-2512 | 999 | Apache 2.0 | 24.1% | 88% | fp16 불가. INT4로 5–6 s [추정] | 단일 T4를 거의 독점한다 |
| Sana-Sprint 1.6B | 754 | 출처 상충 | 9% | 61% | 1–2 s/장 | **판정자로 부적합.** DC-AE f32라서 latent Nyquist 위 디테일을 디코더가 지어낸다 |

- 주의: SP의 상대는 이미지 SOTA가 아니라 비디오 모델의 프레임이다. 이미지 아레나 Elo를 비디오 프레임 미감으로 옮기는 환산식은 없다(리서치 미확인 #13). 그래서 E1에서 직접 잰다.
- 필요한 교사 수준은 Bradley–Terry 가산성으로 역산한다. v1의 TP 목표(우리 렌더 vs 교사 스틸) ≥ 45%를 그대로 두면 다음이 성립한다.
  - logit(SP_ours) = logit(SP_teacher) + logit(0.45)
  - 따라서 SP_ours ≥ 50%가 되려면 **교사 스틸 SP ≥ 55%**가 필요하다. 이 값이 E1의 합격선이다.

### 1.3 원안의 내부 오류와 불일치

| # | 위치 | 원안 | 바른 값 / 문제 |
|---|---|---|---|
| 1 | §4 합계 | "Flow-S를 빼면 2×T4로 약 1.5일" | Flow-S만 빼면 139 T4-h = **2.9일**이다. G-Adapter까지 빼야 69 T4-h = 1.4일이다 |
| 2 | M4 PWMT | 표 "메타 1×10¹⁷ / 5 h", 본문 "약 10 T4-h" | Reptile 내부 루프(200세계 × 10태스크 × 300스텝)를 넣으면 약 4×10¹⁷이다. 2×T4로 4.6 h = **9.3 T4-h**. 본문이 맞고 표가 틀렸다 |
| 3 | M4 PWMT | TTT "15–30 s, GPU0이 교사를 돌리는 동안 병렬로 끝남" | 이것은 1회 비용이다. 세계 하나에 (슈퍼블록 × 재질 × 레벨 × 라운드) ≈ 60회가 필요하므로 **GPU1에서 15–30분/세계**다. GPU1은 렌더, fit, LBO와 경합한다 |
| 4 | §4.1 | "IMU 300시간이면 약 10M 모델의 Chinchilla 점" | 실토큰 5,400만 개는 **2.7M** 모델의 Chinchilla 점이다. 8M 모델에는 ×5.5 증강이 필수다 |
| 5 | M5 BRP vs §6.4 | BRP 학습에 ScanNet++/DL3DV lift 300, 같은 문서에서 "ScanNet++는 평가 전용" | 자기모순이다. 두 데이터셋 모두 비상업 약관(DL3DV는 검증 필요)이다. 학습은 **자체 캡처**로 한다 |
| 6 | M3 데이터 | MatSynth 4천 + ambientCG 2천 + Poly Haven 700 | MatSynth가 ambientCG와 Poly Haven 등을 모은 집합이라 중복 계산이다. 고유 재질은 약 **4–5천**이다 |
| 7 | §11 X-S0 | "교사 NFE ≤ 1.6 s @1024² fp16" | 6B급 교사는 이 기준으로 정의상 탈락한다. 교사와 해상도별 기준으로 바꾼다(E1) |
| 8 | §6.3 | W* "GPU당 1.5–6개/day" | 원안 자체 수치(Tier-3 1,700 T4-h에 680개 = 2.5 T4-h/세계 = 9.6개/T4-day)와 맞지 않는다 |
| 9 | §7.3 | DP-0 탐색 "후보 10³ × 720p 프리뷰 약 8분" | 10³ × 192프레임 = 19.2만 프레임이다. T4 3DGS 100–300 fps로 **11–32분**이 든다. 수제 지각(160×90 통계)으로 1차 채점하고 상위 10개만 720p로 그리면 **2–4분**이다 |
| 10 | 리서치 보정 | "WDM: 교사 두 개(real/fake) fp16 24.6 GB" | fake score를 같은 base에 LoRA on/off로 두면 가중치는 **1벌**(12.3 GB, INT4 3.5 GB)이다. DMD 생성기의 기울기는 교사 역전파 없이 점수 차로 계산되고, 역전파는 LoRA 학습에만 필요하다 |
| 11 | 리서치 보정 | "SDXL fp16 경로 검증됨(fp16-fix VAE)" | fp16-fix VAE는 서드파티 파인튜닝 가중치라 **금지**다. mid-block fp32 upcast로 대체하고, decode 시간은 +30–60% [추정]이다 |

### 1.4 확인된 숫자 (그대로 유지)

- 4,320 T4-h, GPU 시간 장부 합계, 교사 호출 비중 약 57%, 소형 모델 학습 비중 약 11%
- 카탈로그 FLOPs 합 4.5×10¹⁸, 파라미터 180M / 69M / 249M
- 모델별 FLOPs: MRH 순전파 0.2 TF, LBO-S 30 GF/타일, BRP 0.49–0.63 TF/순전파와 42M
- LBO-S 추론: 방 4.4×10¹⁴ FLOPs, 20–40 s. Powers of Ten 약 3분
- 메모리 지도(SDXL 가정) 합계, PCIe 전송량(탭 18 MB → 1.5 ms), roofline 규칙
- Chinchilla 표(IMU 제외), u-μP와 GradScaler 기본값(2¹⁶, growth 2000), cuFFT fp16의 2의 거듭제곱 제약

---

## 2. 가장 실패할 7지점과 업그레이드 (Task 2)

### F1. 교사 경제학: 미감 상한 × T4 속도 × fp16 (치명도 최상)

- **원인**
  - 원안은 SDXL급을 기본값으로 두므로 SP 상한이 정해진다.
  - 6B DiT로 바꾸면 NFE 비용이 4.7배(3.5/0.75)로 뛴다.
  - bf16으로 학습된 DiT는 fp16에서 overflow 위험이 있다. 잔차 스트림 누적값과 SwiGLU 중간값이 65,504를 넘을 수 있다.
  - Turbo 증류 모델은 학습된 소수의 t에서만 신뢰할 수 있어, WSSC의 임의 결정 창과 맞지 않는다.
- **업그레이드**
  1. **Day-10 교사 토너먼트(E1).** 결정 지표는 "SP per T4-h"다.
     - 사전 순위는 Z-Image-Turbo, FLUX.2 klein 4B, SDXL(대조군) 순이다.
     - Sana-Sprint는 위 표의 이유로 제외한다.
  2. **FP16 Rescue** [조합]. 함수를 바꾸지 않는 정확한 재매개변수화다.
     - (i) 잔차 스트림 스케일 1/s: 모든 블록 출력 투영 W_o, 입력 임베딩, adaLN의 잔차 쪽 출력을 1/s로 줄인다. RMSNorm은 스케일 불변이므로 함수가 그대로이고, overflow 여유만 s배 커진다.
     - (ii) SwiGLU 재균형: W_up과 W_gate 중 선형 쪽을 1/s, W_down을 s배로 둔다. 출력은 같고 중간값만 1/s가 된다.
     - (iii) norm, softmax, RoPE는 fp32로 계산하고, SDPA mem-efficient 커널은 fp32로 누산한다.
     - (iv) NFE 1천 회 동안 블록별 max-abs를 프로파일링해 s를 고른다. 비용은 0이다.
  3. **교사를 두 경로로 운용한다.**
     - **INT4 W4A4** (Nunchaku SVDQuant, Turing INT4): 초안 생성과 데이터 엔진의 대량 호출에 쓴다. 판정과 커밋에는 E2의 판정 충실도 테스트를 통과해야만 쓴다.
     - **fp16 파이프라인 분할**: 층 0–14는 GPU0, 15–29는 GPU1에 둔다. 경계 전송은 768² 뷰 하나에 19 MB, 약 1.6 ms다.
  4. **Turbo t-격자 문제.** 결정 창을 8-step 격자에 스냅한다. 창이 1.5옥타브를 넘으면 **t-densify LoRA**를 학습한다.
     - 교사 자기 궤적을 자기증류한다.
     - rank 32 전 선형층 약 72M으로, 6.15B의 1.2%라 "아주 가벼운 어댑터" 범위 안이다.
     - 비용은 약 20 T4-h다.
  5. **미감은 교사만의 몫이 아니다.** DTC, Gaffer, 물리 렌즈와 필름 룩이 더하는 몫을 E5에서 따로 잰다(같은 세계, 룩 on/off).

### F2. NFE 회계 붕괴: 세계당 교사 비용이 플라이휠을 멈춘다

- **원인**
  - band-iso 조건(z비 ≤ 2^¼, θ ≤ 33°) 때문에 뷰 안에서 소유권을 갖는 유효 픽셀 비율 η는 약 25%다.
  - 최미세 레벨 2×10⁷ texel을 교사가 모두 보려면 768² 뷰 544개 × 라운드 2 × K 3이 필요하다.
  - 그러면 방 세계 하나가 약 **2,900 NFE-eq**, Z-Image fp16으로 **2.8 T4-h**다.
  - 원안의 W* 680개는 1.4 s/NFE 가정이다. Z-Image로는 약 300개로 줄어 플라이휠이 돌기 전에 멈춘다.
- **업그레이드**
  1. **판정 해상도 분할 (Judge-Resolution Demand Split)** [신규]
     - v1 §7.1에 따라 SP는 1080p로 정규화해서 판정한다.
     - Ray-Bundle Demand를 1080p와 4K로 두 번 계산한다. ℓ*₁₀₈₀까지는 교사 관할, ℓ*₄ₖ − ℓ*₁₀₈₀(+1옥타브, 8K면 +2)은 학생 관할로 둔다.
     - 거리 3 m, 화각 60° 기준 1080p 픽셀 footprint는 1.8 mm, 4K는 0.9 mm다.
     - texel의 75%를 차지하는 최미세 레벨이 교사 장부에서 빠진다.
  2. **투기적 대역 커밋 (Speculative Band Commit)** [신규]
     - 학생(ICX-LBO)이 대역 b의 초안을 쓴다.
     - 교사는 t_end(b)에서 1 NFE로 E2 잔차 DR_rel만 잰다. 가장 정면인 2뷰, 즉 κ_v = 2로 검증한다.
     - DR_rel ≤ 1.2이고 seam 검사를 통과하면 수락한다. 거부된 블록만 정식 소유권 절차(K=3)를 밟는다.
     - **v1 E2의 고정점 정의("교사가 고칠 게 없다고 말하는 세계")가 그대로 수락 테스트가 된다.**
     - 기대 비용은 블록당 κ_v/κ + (1−a)·K다. 수락률 a = 0.65면 0.5 + 1.05 = 1.55 NFE로, 기존 3 NFE의 절반이다.
  3. **관할 이양 + TJF KPI** (원안 유지). 재질 클래스 × 레벨 셀마다 2AFC ≤ 55%를 통과하면 ℓ*₁₀₈₀ 레벨도 학생에게 넘긴다.
  4. **세계별 NFE 장부 공식.** 모든 세계에 대해 기록한다.
     $$\mathrm{NFE_{eq}}=\sum_{l}\Big\lceil\tfrac{A_l}{P_v\,\eta}\Big\rceil\,\kappa\,R_l\,\big[\tfrac{\kappa_v}{\kappa}+(1-a_l)K\big]\,c_{res}$$
     - A_l: 레벨 l의 수요 texel 수
     - P_v: 뷰 픽셀 수
     - η: 소유 픽셀 비율
     - κ: 블록당 겹치는 뷰 수
     - R_l: Jacobi 라운드 수
     - c_res: 해상도별 FLOPs 계수
  5. **결과.** 방 세계 비용이 세대별로 G0 약 2,900, G1 약 1,150, G2 약 760, G3(관할 이양 후) 약 360 NFE-eq로 줄어든다(§5.1.4). W* 목표는 약 1,300개(fp16)다.

### F3. 근사 디코더(TAE-T)가 결정 대역을 오염시킨다

- **원인**
  - TAESD급 2M 디코더는 latent Nyquist 위, 즉 최고 대역을 뭉갠다.
  - 원안은 inner loop 전체를 TAE로 돌리고 full VAE는 commit에서만 쓴다. 그러면 소유권 투표의 최고 대역이 TAE 블러에 맞춰 fit되어 SFR이 무너진다. v1이 고치려던 병(평균에 의한 블러)이 디코더를 통해 다시 들어오는 셈이다.
- **업그레이드: 노이즈 마스킹 근사 규칙 (Noise-Masked Approximation Rule)** [신규]
  - 스텝 t에서 근사 연산자 A(TAE enc/dec)를 써도 되는 조건은 둘 중 하나다.
    - 대역 b의 근사 오차 e_A(b)가 그 대역의 노이즈 표준편차 σ_t·√P_ε(b)의 10% 미만이다.
    - 대역 b가 아직 미결정 상태다(b > b_hi(σ)).
  - 결정 창의 마지막 스텝, 즉 대역 o*가 commit되는 스텝만 full VAE를 쓴다.
  - 효과:
    - full VAE 호출이 뷰·레벨당 K회에서 1회로 준다.
    - 768², K=3 기준 뷰당 시간이 8.55 s에서 6.9 s로 줄어든다(−19%).
    - 이 규칙의 정의상, 결정 중인 대역의 정확도는 손실이 없다.
  - TAE-T의 대역별 전달함수와 임계 t*는 E2에서 측정한다.

### F4. PWMT: 테스트타임 학습이 v1의 병을 되살린다

- **원인 (세 가지가 겹친다)**
  1. MSE로 TTT하면 조건부 평균, 즉 블러가 나온다. flow로 TTT하면 수십 샘플 × 300스텝에서 암기, 즉 타일링 반복이 나온다.
  2. GPU1에서 세계당 15–30분을 쓰며 렌더·fit과 경합한다(§1.3 #3).
  3. 수요가 확장되면 가중치가 바뀌어 T8 비트 동일성을 잃는다. 원안도 이 점은 인정했다.
- **업그레이드: 재질 견본 은행 + 예시 조건 LBO (Material Swatch Bank + ICX-LBO)** [신규/조합]
  - **견본 생성.** bedrock commit 때 재질 인스턴스 m × 교사 관할 레벨 l마다 표준 견본 1장을 만든다.
    - 형태: 평면 패치, 고정 중립 조명, band-iso 정사영, 512², 4 NFE
    - 견본은 (seed, m, l, y)의 순수 함수이고 수요와 무관하다. bedrock의 재질 정의의 일부로 함께 commit된다.
  - **예시 조건.** ICX-LBO는 해시로 고른 견본 타일 4개(64², intrinsics 12ch)를 cross-attention으로 받는다.
    - 학습 때 "같은 재질의 다른 패치"를 예시로 주는 메타학습을 하므로 테스트타임 학습이 없다.
  - **효과**
    - 재질 정체성 일관성(PWMT의 덤)을 유지한다.
    - flow + WNF 생성이라 블러와 타일링이 생기지 않는다.
    - 수요 독립이라 native 모드에서 **T8 비트 동일성이 복구**된다.
    - GPU1 부하가 0이다.
  - **비용.** 재질 24개 × 레벨 3 × 4 NFE @512² = 288 NFE ≈ 75 NFE-eq다. fp16 4.4분, INT4 2.4분.
  - **대조군.** PWMT-FiLM(FiLM 파라미터 약 5만 개만 flow 손실로 TTT)은 E4의 대조 arm으로 남긴다.

### F5. MRH: 합성에서 실사로의 일반화 실패와 자가 라벨 순환

- **원인**
  - 원안의 MRH 학습 데이터는 절차 장면 운송, Hypersim(합성), W* 자가 라벨뿐이다.
  - Depth Anything V2 저자들은 "합성만으로 학습하면 일반화 문제가 심각하다"고 보고했다.
  - 실사와 고전 GT를 쓰는 차선이 없다. 자가 라벨은 솔버의 오류를 굳힌다.
- **업그레이드: 실사 MVS 차선 (Real-MVS Lane, 교사 노이즈 실사)** [조합]
  1. 자체 캡처 장면(미러리스 RAW, 장면당 200–500장)을 COLMAP 4.x(SIFT) + GLOMAP + PatchMatch MVS + 기하 일관성 필터로 처리해 depth, normal, 신뢰도 마스크를 얻는다. 계보 태그는 `GT-classical-real`이다.
  2. **실사 이미지를 교사 VAE로 encode하고, 임의 t의 노이즈를 더해 1 NFE를 돌리면 탭을 얻는다.** 생성은 필요 없다. 라벨 유효 대역은 결정 곡선 D(o,t)로 정하며 운송과 같은 규칙을 쓴다.
  3. 배치 구성 규칙: GT-proc ≥ 30%, GT-classical-real ≥ 25%, 자가 라벨 ≤ 25%.
  4. 기준선: 같은 데이터로 학습한 22M from-scratch 픽셀 depth 망. 두 모델의 차이로 **교사 특징 배당(teacher-feature dividend)**을 정량화한다(E3).
  - 실패하면:
    - 솔버 안의 band-iso x̂0에 고전 평면 스윕 MVS를 직접 적용한다.
    - MRH는 coarse geometry와 intrinsics만 맡는다.

### F6. 병목은 GPU가 아니다: 툴체인, 전력, 호스트, 쿼터

- **원인**
  1. Triton ≥3.3은 sm<80에서 `tl.dot`을 FMA로 폴백한다 [검증]. torch.compile max-autotune, FlexAttention, 커스텀 Triton GEMM이 경고 없이 5–10배 느려진다.
  2. FA2/3/4는 Turing을 지원하지 않는다.
  3. 70W 전력 쓰로틀이 일찍 걸린다 [검증].
  4. Kaggle 제약:
     - 세션 12 h, 주 약 30 h다.
     - vCPU와 디스크 규모는 미확인이다(작업 디스크 약 20 GB는 2차 출처).
     - Z-Image fp16 가중치만 약 20 GB라 작업 디스크를 넘는다.
  5. 절차 생성, COLMAP, 데이터 로딩이 CPU 병목이 된다. 장시간 솔버는 세션 경계에서 끊긴다.
- **업그레이드**
  - **툴체인 고정**
    - PyTorch CUDA 13.x 빌드로 고정한다(T4는 지원 하한선이다).
    - inductor GEMM 백엔드는 `ATEN`으로 고정한다.
    - attention은 SDPA mem-efficient 또는 flash-attention-turing(head_dim 64/128; fwd 최대 2.19배, bwd 1.35–1.51배)을 쓴다.
    - Triton은 elementwise fusion에만 쓴다.
  - **클럭 고정.** `nvidia-smi -lgc`로 지속 가능한 클럭을 고정해 쓰로틀 진동을 없앤다. 값은 E1에서 TF/W 곡선을 재서 정한다.
  - **GPU 절차 생성** [조합]. ProcRoom/ProcProp의 voxel과 G-buffer 생성을 Warp 커널로 GPU에서 돌린다(Warp CUDA 13 빌드는 sm_75 이상 [검증]). CPU 의존이 사라져 Kaggle에서도 데이터 주조소가 돈다.
  - **Jacobi 라운드 Merkle 체크포인트** [조합]
    - W^(r+1) = F(W^(r), seed)이므로 라운드 경계가 자연스러운 체크포인트다. 상태는 Merkle 루트와 샤드로 표현한다.
    - 재개 후 결과는 native 모드에서 비트 동일, solver 모드에서는 원래 실행과 같은 OIE 범주를 유지한다.
  - **Kaggle 운용**
    - 교사 가중치는 Kaggle Models로 마운트한다(작업 디스크에 포함되지 않음).
    - 데이터는 생성 즉시 소비한다.
    - CPU 주조는 외부 저가 VM에서 돌린다.
    - Kaggle 이용약관이 상업 R&D 사용을 허용하는지 확인해야 한다(검증 필요).

### F7. Phase 2: 비디오 증인과 포즈 추정기가 사라진 뒤의 데이터 기아

- **원인**
  - v1 Phase 2는 저주파 안무를 비디오 증인(Wan)에게, 실영상 lift를 GVHMR, WHAM, FoundationPose, MoSca에게 맡겼다. 모두 금지다.
  - MoPrior 데이터는 Chinchilla의 0.01배다. 사람 동작(2c)이 막힌다.
- **업그레이드: Capture-first + 합성 인간 키포인트** [조합]
  - **자체 mocap 파이프라인**
    - 멀티캠 리그 8–12대(동기화한 GoPro 또는 RPi HQ, $3–8k)
    - 고전 보정(체커보드, 번들 조정)
    - **Kp2D-S**(12M, from scratch). MakeHuman(CC0) 인체를 우리 렌더러로 렌더하고 교사로 운송한 데이터로 학습한다.
    - 다시점 삼각측량과 시간 평활(고전)
  - **보조 센서.** 자체 제작 IMU 스트랩(Madgwick 필터).
  - **사건과 타이밍.** EPD-S는 물리 sim으로 데이터를 무제한 만든다. 실사건 타이밍은 자체 캡처 오디오 onset(고전) + Timing-S로 얻는다.
  - **SMPL-X 대체.** SMPL-X는 MPI가 학습한 통계 모델이고 비상업 약관이다. 자체 스캔 PCA 체형 공간(고전) 또는 MakeHuman 파라메트릭 모델로 바꾼다.
  - **원안 유지 + 정직한 표기.** H7(이미지 교사 + 물리 + 실사건 prior)을 본선으로 승격하는 결정은 유지한다. 2c 일정은 캡처 규모에 묶인다고 명시한다.

---

## 3. 제약 위반 점검 (Task 3)

| # | 위치 | 몰래 끼어든 사전학습물 | 판정 | from-scratch / 고전 대체 |
|---|---|---|---|---|
| 1 | v1 §1.8 WSSC `SelfCaption(Render_v(W),…)` | 캡셔닝 VLM | **위반** (시스템 경로) | 캡션 작성기(모델 없음): MRH sem 최근접(교사 TE로 한 번 임베딩한 명사구 사전) + 문법 템플릿 + 재질 그래프 이름 + albedo 클러스터 색 이름 |
| 2 | v1 §2.4 ℒ_ocr | OCR 모델 | **위반** | **Glyph-S 4M** (OFL 폰트 Noto/나눔 × 우리 렌더러 × 운송). SYMBOL 문자열은 MSDF라서 구성상 정확하다. Glyph-S는 "SYMBOL 밖에서 생긴 글자"만 검출한다 |
| 3 | COLMAP 4.0 ALIKED/LightGlue 옵션, `vocab_tree_flickr100K`, ORB-SLAM3 `ORBvoc` | 학습된 특징·매처, 외부 이미지로 학습한 k-means 코드북 | **엄격 모드 위반** | SIFT 고정, 비디오는 sequential 매칭. 필요하면 우리 이미지로 vocab tree를 학습한다(k-means) |
| 4 | 폰 카메라 ISP(Deep Fusion/Smart HDR), ProRAW 계산 합성 | 입력 데이터에 ML 처리 흔적 | 계보 오염(회색) | MVS, 오라클, MRH 실사 차선은 미러리스 RAW로 찍는다. 폰은 IMU 전용. 계보 태그 `sensor-ISP-ML` 신설 |
| 5 | ARKit/ARCore VIO | 블랙박스이고 ML을 포함할 수 있다 | 회색 | OpenVINS / VINS-Mono (고전) |
| 6 | SDXL-Lightning / Hyper-SD / DMD2 LoRA (원안 "조건부 허용") | 제3자 사전학습 어댑터 = 두 번째 사전학습물 | **위반** ("이미지 모델 하나") | 공식 few-step 배포본(Z-Image-Turbo)을 쓰거나 우리가 학습한 LoRA를 쓴다. DMD2는 CC BY-NC-SA라 이중으로 탈락 |
| 7 | Z-Image Fun-ControlNet-Union, TAESD/TAEF1, sdxl-vae-fp16-fix | 제3자 학습 가중치 | **위반** | G-Adapter-DiT, TAE-T, fp32 upcast |
| 8 | 교사 TE = Qwen3-4B (사실상 LLM) | LLM | **회색. 창업자가 결정할 사항** | **임베딩 전용 원칙(Embedding-only Doctrine)**: DiT가 소비하는 은닉층 출력만 쓴다. 문자열을 벡터로 바꾸는 용도만 허용하고, 디코딩(생성)은 금지한다. 결과는 캐시한다 |
| 9 | 원안 BRP의 ScanNet++/DL3DV, Flow-S의 FlyingThings/Chairs, MoPrior의 AMASS/HumanML3D, Hypersim | 모델은 아니지만 비상업·연구 약관 | 상업 위반 위험 | 학습은 자체 캡처 500장면, 우리 렌더러 flow, CMU mocap + 자체 캡처로 한다. 공개셋은 평가 전용. Hypersim은 약관을 확인할 때까지 엄격 모드에서 제외 |
| 10 | Blender Cycles OIDN / OptiX AI 디노이저, Infinigen 렌더 기본값 | 학습된 디노이저 | **위반** (엄격 모드) | 끄고, raster + 교사 운송을 쓴다. Infinigen에서는 지오메트리만 가져온다 |
| 11 | 빠른 3DGS 초기화 계열(InstantSplat 등의 DUSt3R/MASt3R/단안 깊이 초기화) | 사전학습 기하 모델 | **위반** | COLMAP SfM 점 초기화 + FastGS / gsplat MCMC |
| 12 | v1 DTC p̂_T의 "단안 깊이", Σ_u의 RAFT/UFM, sem의 SigLIP/DINO | 사전학습 비전 모델 | 위반 (원안이 이미 처리) | MRH 디포커스/intrinsics/sem 헤드, DIS/LK + RAFT-S 1M |
| 13 | P2: GVHMR/WHAM/MANO/FoundationPose/MoSca, Event Type Contract의 VLM 판정, UTTERANCE의 TTS, SAM 3D Body | 사전학습 인체·포즈·음성·VLM | **위반** | Kp2D-S + 멀티캠 삼각측량, 사람 판정 + 자체 소형 분류기, 녹음 음성(자체 TTS-S는 장기) |
| 14 | 평가 모델(GeCo의 VGGT/UFM, LPIPS, ArcFace, CLIP Droste, OCR TPR)이 시스템으로 새는 경로: best-of-N 자동 선택, 자동 HP 탐색 신호, CRM 학습 라벨, early stopping | 평가 모델 | 누수되면 위반 | **규칙을 명문화한다.** 평가 모델의 출력은 사람이 읽는 보고서에만 쓴다. 자동 선택, 학습, 게이트 자동화에는 쓰지 않는다(Goodhart와 위반을 동시에 막는다) |
| 15 | Nunchaku가 배포한 사전 양자화 Z-Image INT4 | 같은 교사의 변환물 | 허용 (지식 원천 동일) | 계보를 단순하게 두려면 우리 보정셋으로 자체 양자화한다 |

원안의 판정 중 그대로 유지하는 것: 교사 동결 특징 + from-scratch 헤드(MRH), 교사 LoRA ≤ 2%, from-scratch G-Adapter, TPD(LPIPS 대체), CC0/CC-BY 에셋은 모델이 아니라 데이터라는 판정, 고전 알고리즘 전반, 평가 전용 GeCo.

---

## 4. 30일 결정적 실험 (Task 4)

| ID | 기간 | T4-h | 결정하는 것 |
|---|---|---|---|
| E1 교사 토너먼트 + FP16 Rescue | D1–D10 | 33 | 어느 교사로 이기는가 |
| E2 X1-slim + 판정 충실도 + TAE 규칙 | D8–D18 | 27 | 대역 판정 기계가 이 교사에서 작동하는가 |
| E3 MRH: 노이즈 t 독심 + 실사 일반화 | D10–D28 | 83 | 사전학습 기하 모델 없이 기하를 읽는가 |
| E4 NFE 경제: Spec-commit + Swatch Bank | D14–D30 | 48 | 플라이휠이 도는가 |
| E5 Oracle A/B-lite + Hand-S v0 | D5–D30 | 12 | T4로도 이기는 축이 있는가 |
| **합계** | | **약 203** | 상시 30일 용량 1,440의 14%. Kaggle(약 258)에서는 E3와 E4를 절반 규모로 돌린다 |

### E1. 교사 토너먼트 + FP16 Rescue

- **가설.** T4에서 도는 Apache 교사 중 하나 이상이 아래 네 조건을 모두 만족한다.
  - (i) fp16(Rescue 적용) 또는 INT4에서 NaN 0/1만 회
  - (ii) 512²에서 1.1 s 이하, 768²에서 2.3 s 이하 per NFE
  - (iii) 멈춘 세계 프롬프트에서 교사 스틸 SP ≥ 55% (vs Seedance/Kling 프레임)
  - (iv) 스냅된 t에서 SDEdit이 저대역을 보존한다
- **셋업**
  - 후보:
    - Z-Image-Turbo: fp16 + Rescue / INT4 Nunchaku / W8A16
    - FLUX.2 klein 4B: fp16
    - SDXL: fp16, 대조군
  - 측정 1(속도): 해상도 512/768/1024 × 배치 1/2에서 30분 지속 부하를 걸고 `nvidia-smi` throttle reason을 기록한다.
  - 측정 2(안정성): 교사마다 NaN 감사 1만 NFE.
  - 측정 3(미감): SP 미니 패널. 6범주 × 10프롬프트 × seed 2. 경쟁 스틸은 같은 프롬프트로 만든 Seedance/Kling 클립의 무작위 프레임이다(API 비용 $300–600). 평가자 20명, 300쌍.
- **T4-h:** 처리량 8 + NaN 감사 12.5 + SP 이미지 3 + 결정 곡선 프로브 3.5 + 자체 INT4 보정 6 = **약 33**
- **성공 기준.** (i)–(iv)를 모두 만족한 후보 중 SP가 가장 높은 것을 고른다.
- **실패하면 다음 수**
  - fp16이 깨지면: 파이프라인 분할 + Rescue + W8A16으로 간다.
  - 모든 후보가 SP 55%에 못 미치면: Qwen-Image-2512 INT4(Elo 999)를 단일 교사로 시험하고 Spec-commit으로 비용을 흡수한다. E5의 룩 상승분도 함께 본다. 목표는 낮추지 않는다.

### E2. X1-slim: 결정 곡선, 판정 충실도, TAE 규칙

- **가설.**
  - (a) 선택한 교사의 D(o,t)가 스냅된 Turbo t-격자에서 한 옥타브를 분리한다.
  - (b) INT4 판정이 fp16 판정과 같다고 볼 수 있다: |ΔD| ≤ 0.05, DR 순위상관 ≥ 0.9.
  - (c) TAE-T가 노이즈 마스킹 규칙을 만족하는 t*가 존재하고, latent Nyquist 아래에서 대역 전달함수가 0.95 이상이다.
- **셋업**
  - 이미지 500장(자체 RAW 사진 + 절차 렌더) × t 12개 × seed 4 @768² = 2.4만 NFE
  - INT4 반복: 150장 부분집합, 7.2천 NFE
  - TAE-T v0 학습(2×T4 2.3 h)
- **T4-h:** fp16 NFE 13.3 + decode 5.7 + INT4 3.7 + TAE 4.6 = **약 27**
- **성공 기준.** (a)(b)(c) 모두.
- **실패하면 다음 수**
  - (a) 실패: t-densify LoRA(20 T4-h)를 학습하거나 레벨 간격을 2옥타브로 넓힌다(v1의 대체안).
  - (b) 실패: INT4는 초안과 데이터에만, 검증과 커밋은 fp16으로 한다.
  - (c) 실패: TAE는 t ≥ t*에서만 쓰고, 나머지는 full VAE.

### E3. MRH v0: 노이즈 t 독심과 실사 일반화

- **가설.** 동결 교사 탭 위의 22M 헤드가 다음 두 가지를 해낸다.
  - 같은 크기의 from-scratch 픽셀 망보다 실사 holdout AbsRel이 25% 이상 좋다.
  - t=0.6에서 "x̂0 decode 후 픽셀로 추정"하는 경로보다 20% 이상 좋다.
- **셋업**
  - 운송 4만 개 (ProcRoom v0 + CC0 에셋, 512², 기생 탭)
  - 실사 MVS 차선: 자체 캡처 60장면에서 3만 장, 장면당 t 2개
  - 탭 압축: Frozen PCA(3840 → 256, 고정)로 줄여 RAM 링버퍼에 둔다
  - holdout: 자체 10장면 + ScanNet++ 20장면(평가 전용)
  - 기준선: B1 = 22M 픽셀 UNet(같은 데이터로 from scratch), B2 = decode → B1
- **T4-h:** 운송 38 + 실사 탭 18 + MVS 15 + 학습과 평가 12 = **약 83** (INT4면 약 60)
- **성공 기준 (실사 holdout)**
  - AbsRel ≤ 0.12 (t ≤ 0.3), ≤ 0.18 (t = 0.6)
  - normal median ≤ 18°
  - B1 대비 25% 이상, t=0.6에서 B2 대비 20% 이상 개선
- **실패하면 다음 수**
  1. 탭 층을 sweep한다(저렴하다).
  2. 실사 캡처를 200장면으로 늘린다.
  3. 솔버 안 평면 스윕 MVS를 대체 경로로 쓰고, MRH는 coarse geometry와 intrinsics만 맡긴다.

### E4. NFE 경제: Speculative Band Commit + Swatch Bank (facet worlds)

- **가설.** facet world(4–10 m² 평면 또는 준평면, STRATA 4레벨, 재질 6종)에서 Spec-commit(ICX-LBO v0 초안 + 1 NFE 검증)이 아래를 동시에 만족한다.
  - full ownership WSSC-lite 대비 교사 NFE가 2.5배 이상 줄어든다.
  - ΔDR_rel ≤ 5%, seam 2AFC ≤ 60%, solver OIE ≤ 0.05
  - native 최미세 레벨은 수요 순서를 8번 섞어도 Merkle 루트가 같다.
- **셋업**
  - WSSC-lite: band-iso 정사영, Chart-Direct Sync, WNF
  - facet world 24개: 절차 재질 12개 + 자체 촬영 표면 12개
  - ICX-LBO v0: 교사 타일 100만 + CC0 + 견본 5천으로 학습
  - arm 3개: full / Spec + Swatch / PWMT-FiLM
- **T4-h:** 타일 17 + 학습 8.3 + 견본 6 + 솔버 14.4 + 2AFC 렌더 2 = **약 48**
- **성공 기준.** 가설의 모든 조건을 만족하고, 교사 관할 최미세 레벨에서 수락률이 60% 이상.
- **실패하면 다음 수**
  1. 견본 수와 크기를 늘린다.
  2. 검증을 t 2개에서 한다.
  3. Spec-commit 대상을 확률적 재질(SMF류)로 한정하고, 구조 재질(CARRIER)은 교사가 계속 소유한다.

### E5. Oracle A/B-lite + Hand-S v0 (v1 X2를 T4에서)

- **가설.** 사전학습 모델 없이 고전 방법으로 lift한 완벽한 정적 실사 세계에 우리 카메라 스택을 얹으면, 같은 첫 프레임에서 경쟁 I2V를 일반 평가자 기준 70% 이상으로 이긴다.
- **셋업**
  - 자체 캡처 20장면(미러리스 RAW, 장면당 300–600장, 5장면은 HDR 브래킷)
  - COLMAP SIFT + GLOMAP + FastGS/gsplat 3DGUT. 장면당 T4로 15–25분 [추정]
  - 경로: 의도 4 × 리그 2
  - 손떨림 비교: Hand-S v0(IMU 40–60 h) / 실제 gyro 재생 / 손 없음
  - 경쟁: Kling/Seedance I2V
  - 사람 평가: 일반 40명, 전문가 10명, 약 1,500쌍
- **T4-h:** lift 8 + 렌더 3 + 학습 1 = **약 12**
- **성공 기준**
  - VP 일반 ≥ 70%, 전문가 ≥ 75%
  - GeCo ≤ 0.03 (평가 전용)
  - 카메라 튜링 2AFC(Hand-S vs 실제 gyro) ≤ 60% (D30), D60에 ≤ 55%
- **실패하면 다음 수.** v1 X2의 카메라/룩 ablation 그리드를 돌린다. IMU 캡처를 늘리고 리그 전달함수 증강을 한다.

---

## 5. 개정 제안서 v2 전체본 (Task 5)

# 렌즈 05 v2: 작은 파운데이션 모델 아키텍트와 데이터 엔진 (2×T4 개정판)

> 부제: "판사는 비싸다. 판결은 판사가, 서류는 서기가."
> 기준: PARADIGM v1.0 + 창업자 하드 제약(2×T4, 이미지 모델 1개만 빌림, 나머지는 수십M from-scratch). 목표는 v1 그대로이고 낮추지 않는다.

### 5.0 요약

1. **화폐는 교사 NFE다.**
   - 90일 상시 4,320 T4-h 중 약 52%가 교사 호출이다.
   - 수십M 카탈로그 전체(약 240M)를 처음부터 한 번 학습하는 데 **약 100 T4-h**(2×T4로 약 2일)가 든다.
   - 최적화 대상은 학습 효율이 아니라 **교사 호출 1회당 정보량**과 **교사가 판정할 필요가 없는 것을 판정하지 않는 것**이다.
2. **교사는 Z-Image-Turbo(6.15B, Apache 2.0, Elo 941)를 1차 후보로 둔다.** E1에서 확정한다. SDXL은 대조군이다.
3. **세 가지 장치로 교사 비용을 3.8–8배 줄인다.** 판정 해상도 분할, 투기적 대역 커밋, 재질 견본 은행 + ICX-LBO.
4. **접착제 가설.** 수십M 모델은 세계를 아는 모델이 아니라 지식 원천 사이의 인터페이스다. 원천은 셋이다.
   - 외형: 교사
   - 빛, 렌즈, 동역학: 렌더러, 광학, 시뮬레이터
   - 구조: 절차 생성 코드
5. **해자는 데이터 주조소다.** "기하는 코드에서, 외형은 교사에서, 라벨은 구성에서." 여기에 실사 MVS 차선과 견본 은행이 더해진다.

### 5.1 예산: 화폐는 NFE다

#### 5.1.1 계획 상수 (E1에서 실측값으로 교체)

| 항목 | 값 | 근거 |
|---|---|---|
| 소형 모델 학습 | 2D·xfmr 12 TF/장, 3D conv 6–8 | 실측 HGEMM 41.6 TF [검증], 쓰로틀과 비GEMM 연산 고려 |
| 2×T4 하루 | 2.1×10¹⁸ FLOP (소형 기준) | |
| 교사 Z-Image-Turbo fp16 | 1024² 3.5 / 768² 2.0 / 512² 1.0 s | [추정] |
| 교사 INT4 W4A4 | fp16의 0.5–0.6배 | INT4 실측 114 TOPS [검증], Nunchaku Turing 지원 [검증], T4 실측은 미확인 |
| NFE-eq | 768² = 0.55, 512² = 0.26 | FLOPs 비율 |
| NFE-eq/T4-h | fp16 1,030, INT4 약 1,900 | |
| VAE 왕복 | 1024² 1.5 / 768² 0.85 / 512² 0.4 s. decode만 하면 약 60% | [추정] |
| TAE-T decode | 1024²에서 25–40 ms | [추정] |
| 운송 샘플 | 512² 3.4 s (2.5만/T4-day), 768² 6.85 s (1.26만) | |
| band-iso 뷰 768² | 8.85 s, 9.8천 뷰 = 141만 타일/T4-day | |
| 견본(swatch) 512² | 4 NFE + decode ≈ 4.4 s, 약 2만/T4-day | |
| 총 GPU | 상시 4,320 T4-h / Kaggle 780 (최악 390) | |

#### 5.1.2 T4 roofline 규칙 (원안 유지, ridge 보정)

- ridge는 130(실측)–203(피크) FLOP/byte다.
- Transformer: d ≥ 384, head_dim 64, 토큰 배치 ≥ 32k로 둔다. 얕고 넓게 설계한다.
- 3×3 conv: 연산 강도가 약 4.5·C이므로 채널을 64 이상으로 둔다.
- elementwise는 fuse하고 `channels_last` fp16을 쓴다.
- 같은 파라미터라면 넓은 쪽이 T4에서 최대 2배 빠르다.

#### 5.1.3 GPU 시간 장부

| 항목 | 상시 T4-h | % | Kaggle T4-h | 내용 |
|---|---|---|---|---|
| E1–E5 (D1–D30) | 200 | 5% | 150 | §4 |
| Tier-2 교사 데이터 | 450 | 10% | 120 | 운송 15만, 실사 탭 6만 × t 2개, band-iso 타일 300만, 견본 2만, X1-slim, hero/Aes |
| Tier-3 솔버 W* | 1,800 | 42% | 250 | W* 약 1,360개(1,140 T4-h) + X3/X4/X6형 실험 460 + 데모 200 |
| 카탈로그 학습 | 450 | 10% | 120 | 3세대 × 약 70 + G-Adapter-DiT 36 + sweep 120 + MRH 재학습 40 + P2 seed 44 |
| 실사 lift | 200 | 5% | 30 | 자체 캡처 500장면 × 0.4 (PatchMatch MVS + FastGS) |
| 렌더·평가 | 500 | 12% | 60 | 4K 최종, 경쟁 클립 처리, GeCo(VGGT, 평가 전용, 클립당 약 30–100 s), 사람 평가 클립 |
| 버퍼 | 720 | 17% | 50 | 쓰로틀, 재실행 |
| **합계** | **4,320** | | **780** | |

#### 5.1.4 방 세계 하나의 NFE 장부 (4K 제품, 1080p 판정)

가정: η = 0.25, κ = 4, R = 2, K = 3.

| 단계 | 해상도 | G0 (원안식 full) | G2 (개정) | G2 NFE-eq |
|---|---|---|---|---|
| Hero still (Picture-First, 4장 × 8 NFE) | 1024² | 32 | 32 | 32 |
| Bedrock lockstep (48뷰 × K3) | 768² | 144 | 144 | 79 |
| Swatch Bank (24재질 × 3레벨 × 4) | 512² | 0 | 288 | 75 |
| Strata 교사 관할 (ℓ*₁₀₈₀까지 3레벨, 246뷰) | 768² | 1,476 | 약 810 (L-1 Spec a=0.65, κ_v=2 / L-2 Spec a=0.5 / L-3 R=1 full) | 약 445 |
| 최미세 (544뷰 상당) | 768² | 3,264 | 30 (5% 감사) | 17 |
| Polish / adversarial | 768² | 300 | 200 | 110 |
| **NFE-eq 합계** | | **약 2,900** | | **약 760** |
| T4-h (fp16 / INT4) | | 2.8 / 1.5 | | **0.74 / 0.40** |

- G1(판정 해상도 분할 + ICX 최미세, Spec 없음)은 약 1,150 NFE-eq다.
- G3(관할 이양으로 L-1도 학생에게)은 약 360 NFE-eq, 0.35 T4-h다.
- 세계당 비교사 비용(BRP, fit, 렌더, MRH)은 약 0.15 T4-h다.

#### 5.1.5 W* 생산 계획 (fp16 가정. INT4가 통과하면 약 2,000개)

| 세대 | 기간 | 세계 수 | T4-h/세계 | T4-h |
|---|---|---|---|---|
| G0 | D30–D45 | 60 | 3.0 | 180 |
| G1 | D45–D60 | 200 | 1.25 | 250 |
| G2 | D60–D75 | 400 | 0.9 | 360 |
| G3 | D75–D90 | 700 | 0.5 | 350 |
| **합계** | | **약 1,360** | | **1,140** |

### 5.2 허용 범위 판정

**네 가지 테스트**(원안 유지):

| 테스트 | 질문 | 통과 조건 |
|---|---|---|
| T-출처 | 새 지식 원천을 들여오는가? | 교사 1개(VAE, TE 포함), 우리 데이터, 고전 알고리즘에서만 지식이 온다 |
| T-주체 | 학습된 가중치를 누가 만들었나? | 우리가 from scratch로 학습했다. 교사 LoRA는 예외 |
| T-규모 | 교사를 바꾸는가? | 교사는 동결. 어댑터는 교사의 2% 이하(Z-Image 기준 약 120M 이하) |
| T-계보 | 라벨이 어디서 왔나? | 다른 사전학습 모델이 만든 라벨은 쓰지 않는다 |

**v2에서 추가하는 규칙 셋:**
1. **임베딩 전용 원칙.** 교사 TE(Qwen3-4B)는 DiT가 소비하는 층의 출력만 쓰고, 디코딩은 금지한다.
2. **평가 누수 금지.** 평가 모델의 출력은 자동 선택, 학습, 게이트 자동화에 쓰지 않는다.
3. **라벨 계보 태그 추가.** `GT-classical-real`, `sensor-ISP-ML`, `asset-3p`, `teacher-swatch`.

판정표는 §3 표와 원안 §2.2를 합친 것이다. 원안 판정 중 바뀐 점은 셋이다.
- few-step distill은 "공식 배포본만" 허용한다.
- ScanNet++, DL3DV, FlyingThings는 "평가 전용"이다.
- Hypersim은 "약관 확인 전 제외"다.

### 5.3 접착제 가설과 판정 해상도

| 지식 | 보유자 | 비용 |
|---|---|---|
| 사진 같은 외형, 미감, 구도 | 교사 (6.15B, 사전 컴퓨트는 남이 냄) | 호출당 1.0–3.5 s |
| 빛, 렌즈, 센서 | 렌더러, 광학 수식 | 파라미터 0 |
| 동역학 | Warp, MuJoCo-Warp, MPM | 파라미터 0 |
| 벽돌, 잎, 도시 문법 | 절차 생성 코드 | 파라미터 0 |
| 다시점 일관성 | 세계 W 하나 (T1, T2) | 정의상 |
| 물체 크기(metric) | 수작업 크기 사전 (클래스 500개, 범위) [조합] | 파라미터 0 |

**판정 해상도 분할** [신규]
- 교사는 "평가 해상도에서 보이는 것"까지만 판정한다. 그보다 미세한 옥타브는 학생이 만들고, 교사는 5%만 감사한다.
- 수요 계산이 이미 카메라별 ℓ*(x)를 주므로 추가 비용 없이 경계가 정해진다.
- 8K 납품 시 학생 관할은 +2옥타브가 되고, 교사 비용은 변하지 않는다.

**작은 모델이 통하는 이유(TinyStories 효과).** 인수분해가 각 모델의 분포를 구성상 좁게 만든다.

### 5.4 교사 선택 절차

| 순위 | 교사 | 조건 | 역할 |
|---|---|---|---|
| 1 | Z-Image-Turbo | E1 (i)–(iv) 통과 | 단일 교사. INT4는 대량 호출, fp16은 판정과 커밋 |
| 2 | FLUX.2 klein 4B | Z-Image가 fp16/INT4 모두 실패 | 단일 교사. 견본| 2 | FLUX.2 klein 4B | Z-Image가 fp16과 INT4 모두에서 실패할 때 | 단일 교사. 내장된 멀티 레퍼런스 기능으로 견본(swatch)을 레퍼런스로 줄 수 있다. 견본은 수요와 무관하게 만들어지므로 비회귀 조항에 걸리지 않는다 |
| 3 | Qwen-Image-2512 (INT4) | 모든 후보가 교사 스틸 SP 55%에 미달할 때 | 단일 교사. NFE당 5–6 s라서 Spec-commit과 관할 이양에 기대를 더 건다 |
| 대조 | SDXL | 항상 | 속도와 fp16 안전성의 기준선. 데모에는 쓰지 않는다 |

- **결정식.** SP ≥ 55%를 만족하는 후보 중에서 "SP 증분 / 세계당 T4-h"가 가장 큰 교사를 고른다.
- **교사 교체 비용.** MRH의 탭 투영층, G-Adapter, TAE-T만 다시 학습하면 된다. 합쳐 약 50 T4-h다. 나머지 카탈로그는 교사와 독립적이다.

### 5.5 카탈로그 한 장 (v2)

| ID | 이름 | 배치 | 파라미터 | 아키텍처 | 입력 → 출력 | 학습 데이터 (사전학습 모델 없이) | 학습 FLOPs | 2×T4 시간 | 추론 (T4 1장) | 메모리 / GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | TAE-T | P1-코어 | 2.4M | conv AE 64ch (enc/dec 각 1.2M) | RGB ↔ z 16×(H/8)² | 교사 VAE 쌍(기생 수집) + RAW 실사 | 2×10¹⁷ | 2.3 h | 1024² decode 25–40 ms | <1 GB, GPU0 |
| M1 | **MRH-D** | P1-코어 | 22M | Frozen PCA 탭 3개(3840→256, 고정) + DPT neck 256ch(RCU 12) + 전역 xfmr d=384 L4 + 헤드 9개 | taps{L8,L15,L23}@(H/16)² + t + pooled TE → depth, normal, albedo, rough/metal, shading, defocus, sem, σ̂, Aes, intrinsics | 운송 15만 + 실사 MVS 6만 + W* ≤25% | 1.8×10¹⁷ | 2 h | NFE당 +40–60 ms | 1.5 GB, GPU0 |
| M2 | G-Adapter-DiT | P1-조건부 | 20M | conv 피라미드 → 64² 토큰 → 블록 0/4/8/12에 주입(zero-init gate) | G-buffer(depth, normal, albedo, sem8, edge) → 잔차 토큰 | 교사 자기 샘플 + MRH 라벨, 운송 GT 4만 | 교사 역전파 4만 × 3.2 s | 18 h (파이프라인) = 36 T4-h | 샘플당 1회, 15 ms | 학습 각 장 약 11 GB / 추론 0.1 GB |
| M3 | **ICX-LBO** | P1-코어 | 26M | 3레벨 U-Net(96/192/384) + 20²에서 window attn 2개 + 예시 cross-attn + FiLM(h_l, t) | 55ch@80² + 견본 타일 4×64²×12ch → 잔차 12ch@64² | CC0 PBR(고유 4–5천) 다옥타브, 절차 재질 그래프, 교사 band-iso 타일 300만, 견본 2만, W* chart | 8×10¹⁷/세대 | 9.3 h | 5.5 ms/타일·스텝, 방 하나 20–40 s | 3.5 GB, GPU1 |
| M5 | BRP | P1-코어 | 42M | 3D U-Net 48–384, text x-attn @8³ | 부분관측 64³×32ch + TE → 완성 64³×32 | GPU ProcGen 20만 + Infinigen 3천(지오메트리만) + 자체 lift 300 | 2.9×10¹⁷ | 3.5–7 h | 청크당 약 1 s (16스텝) | 학습 6–8 GB, GPU1 |
| M6 | IPM-S | P1-코어 | 12M | ConvNeXt식(MRH neck 위) + xfmr 2층 | 영역 특징 → τ_SMF 32, CARRIER, 재질 그래프 id+param | 절차 파라미터가 곧 라벨 (운송에 기생) | 3.6×10¹⁶ | 0.4 h | 5 ms/영역 | 0.5 GB |
| M7 | RAFT-S | P1-확장 | 1M | RAFT-small | 프레임 쌍 → flow + occlusion | 우리 렌더러의 정확한 flow (FlyingThings 제외) | 3×10¹⁷ | 3.5 h | 720p 쌍 약 60 ms | 1 GB |
| M8 | Hand-S | P1-코어 | 8M | causal xfmr d=384 L5, μ-law 256 | 조건 + 과거 IMU 2 s → 다음 20 ms | 자체 IMU 300 h + 리그 전달함수 증강 ×5.5 | 2.4×10¹⁶ | 0.3 h | 10 s 샷 <50 ms | 0.2 GB |
| M9 | Path-S | P1-코어 | 19M | xfmr d=512 L6, flow matching + masked | 세계 토큰 ≤64 + TE + style → 키프레임 ≤16 | 절차 촬영 문법 500만 + VCam 퍼포먼스 캡처 + 퍼블릭 도메인 GLOMAP | 7.8×10¹⁶ | 1 h | 10 ms | 0.5 GB |
| M10 | CRM-S | P1-코어 | 12M | 경로 인코더 + 해석 통계 xfmr + 프리뷰 CNN 3M | (세계, 경로, 160×90 통계) → 점수 | 합성 선호 200만 + 사람 1.5–2만 + 전문가 2천 | 3.6×10¹⁶ (+ 프리뷰 렌더 18 T4-h) | 0.5 h | 후보당 2 ms | 0.5 GB |
| M11 | Rig-ID-S | P1-코어 | 3M | 1D conv + xfmr | 포즈 시퀀스 → 리그 클래스/파라미터/손 | 리그 시뮬 + 추출기 노이즈 모델 | 1×10¹⁵ | 0.2 h | 1 ms | — |
| M17 | **Glyph-S** [신규 편입] | P1-코어 | 4M | conv 백본 64–256ch + 검출 헤드 1.2M + (선택) CTC 인식 2.8M | 256² 크롭 → 텍스트 존재 히트맵 64² (+ 한글/라틴 문자열) | OFL 폰트(Noto/나눔) × 우리 렌더러 × 절차 배경 × 교사 운송 200만 | 1.2×10¹⁶ | 0.15 h | 2 ms/크롭 | 0.3 GB |
| M12 | EPD-S | P2 | 38M | set xfmr d=512 L12 | 세계 토큰 + 스크립트 제약 → 사건 ≤1024 | 물리 sim 500만 에피소드 + 자체 캡처 타이밍 | 3.4×10¹⁷ | 4 h | <100 ms | 2 GB |
| M13 | MoPrior-S | P2 | 10M | MDM식 xfmr | TE + 사건 제약 → 모션 | CMU mocap + 자체 멀티캠 mocap | 5.5×10¹⁶ | 0.7 h | 50 ms | 1 GB |
| M14 | F_φ-S | P2 | 4M | GNS식 | 입자 그래프 → 잔차 가속 | 자체 rig sim-to-real 쌍 | 약 10¹⁶ | <1 h | ms/스텝 | — |
| M15 | MAN-S | P2b | 2M | 픽셀별 conv | G-buffer 16² + MAN 노이즈 → 유계 잔차 | 교사 slice 판정 | — | — | <1 ms/타일 | — |
| M16 | Timing-S | P2 | 3M | log-mel CNN | 오디오 → onset, 사건 타입 | 모달 합성 + CC0 SFX | 약 10¹⁶ | 0.5 h | 실시간 | — |
| M18 | **Kp2D-S** [신규 편입] | P2 | 12M | HRNet식 256×192 | 이미지 → 2D 키포인트 25개 + 신뢰도 | MakeHuman(CC0) × 우리 렌더러 × 운송 200만 | 8.4×10¹⁶ | 1 h | 15 ms | 1 GB |

**합계**
- 파라미터: Phase 1 약 171M, Phase 2 약 69M, 전체 약 **240M**. fp16으로 480 MB이고 교사의 1/25 크기다.
- 1회 전체 학습: 2.46×10¹⁸ FLOPs. BRP 3D conv 보정과 G-Adapter-DiT를 넣어 **약 100 T4-h(2×T4 약 2일)**, G-Adapter를 빼면 약 1.3일이다.
- M4 PWMT는 폐지하고 Swatch Bank로 대체한다. PWMT-FiLM은 E4의 대조군으로만 남긴다.

**Chinchilla 표 보정**
- 원안의 표는 그대로 유지한다.
- Hand-S: 실토큰 5,400만은 2.7M 모델의 Chinchilla 점이다. 8M 모델은 증강을 거쳐야 2×에 도달한다.
- ICX-LBO: 20억 토큰, 약 3.2×.
- MoPrior-S: 10M으로 줄여도 0.03×다. 여전히 데이터 기아이고 F7에서 다룬다.

### 5.6 모델별 상세 (변경·신규 위주)

#### M1. MRH-D: 독심 헤드 (DiT판) `P1-코어` [조합/신규: 노이즈 t 융합 신호]

- **역할.** v1에서 다음을 맡던 부품들을 하나로 대체한다.
  - `Teacher → ĝ_v`의 깊이와 법선
  - dreamer depth/normal 헤드
  - explain-away 단계의 intrinsics
  - DTC의 p̂_T 추정(defocus, EV)
  - 소유권 점수의 σ̂²
  - sem과 Aes
  - DA3, VGGT, MoGe, SigLIP, DINO가 맡던 자리
- **탭.**
  - Z-Image S3-DiT 30층 중 8, 15, 23층에서 이미지 토큰만 뽑는다(텍스트 토큰은 제외).
  - 512² 입력이면 32² × 3840, 768² 입력이면 48² × 3840이다.
  - **Frozen PCA** [부품]: 1만 샘플로 층별 3840→256 PCA를 한 번 구해 고정한다.
    - 탭 크기가 94 MB에서 6 MB로 줄어든다(int8로는 3 MB).
    - 저장과 PCIe 전송이 현실적인 크기가 된다.
- **아키텍처.**
  - DPT reassemble: H/4, H/8, H/16, H/32로 재배열한다. ViT 토큰을 다중 스케일로 바꾸는 DPT 본래 용도와 같다.
  - fusion neck: 256ch, RCU 12개, FiLM(t). 약 14M.
  - 전역 xfmr: 16² 풀링 + register 8개, d=384, 4층. 약 7M.
  - 헤드 9개: 1–2M.
- **출력** (H/4 해상도, 512² 입력 기준 128²):
  - log-depth(affine-invariant), normal, albedo, rough/metal, shading, defocus σ
  - sem: TE 공간 2560차원을 PCA로 줄인 768차원
  - σ̂: 대역 4개
  - Aes
  - 전역 intrinsics 8개(화각, 초점거리, 디포커스 스케일, EV, WB, 수평선)
- **metric 스케일.** 클래스별 크기 사전과 화각 헤드로 고정한다. LLM scene graph를 대체한다.
- **데이터 (배치 비율 규칙 §5.8.4).**
  - (a) 교사 운송 15만: 결정 곡선으로 대역을 마스크한다.
  - (b) **실사 MVS 차선 6만**: RAW 캡처를 COLMAP PatchMatch로 처리하고, 교사 노이즈 탭을 붙인다.
  - (c) W* 자가 라벨: 25% 이하.
  - (d) σ̂: X1-slim의 seed 간 분산.
  - (e) Aes: 사람 쌍대 비교 2–3만.
- **손실.**
  - scale-shift 불변 log-depth L1 + gradient matching
  - normal 각도 손실
  - albedo × shading ≈ 선형화 이미지(intrinsic 일관성)
  - 대역 마스크
- **비용.**
  - 헤드 순전파 0.2 TF로, 교사 NFE의 1.3%(512²)다.
  - 학습 2 h(2×T4). 탭은 기생 수집이다. 실사 탭은 별도로 18 T4-h가 든다.
- **게이트.** E3 기준을 따르고, D60에는 실사 holdout AbsRel ≤ 0.10을 요구한다.

#### M3. ICX-LBO + Material Swatch Bank `P1-코어` [신규/조합]

- **역할.** v1의 LBO(200M)와 WDD를 통합하고, PWMT의 역할까지 흡수한다.
  - 미세 레벨 생성을 표면 chart 위의 2D 문제로 바꾼다. chart는 xatlas로 bedrock commit 시점에 만든다.
- **입력.**
  - X ∈ ℝ^{55×80×80}. 구성: 부모 업샘플 12, 기하 7, sem 8, matter 8, WNF 8, apron 8.
  - 조건: Fourier(log₂h_l) 16, 대역, t를 FiLM으로 넣는다.
  - 예시 E ∈ ℝ^{4×12×64×64}: 견본 4장. intrinsics 채널(albedo, normal, height, rough/metal)과 L_d를 함께 넣는다.
- **예시 인코더(2M).** conv stem으로 16² × 384 토큰 4묶음(1,024토큰)을 만든다. 20² 병목의 400개 query가 cross-attention으로 읽는다(0.6 GFLOPs).
- **출력.** Y ∈ ℝ^{12×64×64} 잔차.
  - **2×2 자식 블록마다 평균 0으로 사영**한다. 효도 제약이 구성상 정확히 성립한다.
  - L_d 디테일은 in-situ 조명에서 albedo와 normal 디테일로 다시 셰이딩해서 만든다. 견본의 조명은 옮기지 않는다.
- **샘플링.** rectified flow 2–3 Euler 스텝. 초기 노이즈는 WNF에서 가져온다.
- **비트 동일 조건.** 결정론 커널, 고정 타일 격자, atomic 없는 reduction.
  - 견본은 (seed, m, l, y)의 순수 함수이므로 **native 모드에서 T8 비트 동일이 성립한다.** 보장 범위는 같은 엔진 파일 + 같은 GPU 아키텍처다.
- **견본 은행 생성.**
  - 재질 인스턴스 m × 교사 관할 레벨 l마다 만든다.
  - 평면 패치, 중립 env, band-iso 정사영(DOF 무관), 512², 4 NFE.
  - MRH가 견본의 intrinsics를 단다.
  - bedrock commit의 일부로 Merkle 해시를 붙인다.
- **학습.**
  - 메타 에피소드: 같은 재질의 다른 패치 4개를 예시로 주고 목표 타일의 잔차를 맞힌다. 재질 그래프와 CC0에서 무제한으로 만들 수 있다.
  - 손실: rectified-flow MSE + 반경 파워 스펙트럼 손실(fp32 FFT) + 파인튜닝 배치의 5%에 TPD.
  - 8×10¹⁷/세대, 2×T4 9.3 h.
- **추론.**
  - 방 하나의 최미세 레벨: 2×10⁷ texel = 4.9천 타일 × 3스텝 = 4.9×10¹⁴ FLOPs → **20–40 s** (INT8 TensorRT면 약 절반).
- **관할 이양.**
  - 처음에는 ℓ*₄ₖ − ℓ*₁₀₈₀ 구간만 맡는다.
  - 재질 클래스별로 2AFC ≤ 55%를 통과하면 ℓ*₁₀₈₀ 레벨로 승격한다.
- **게이트.** E4 기준. D60에 최미세 2AFC ≤ 55%, OIE = 0.

#### M0. TAE-T + 노이즈 마스킹 규칙 `P1-코어`

- TAESD 방식이지만 Z-Image VAE(16ch)를 대상으로 처음부터 학습한다. 서드파티 TAEF1은 쓰지 않는다.
- 손실: L1 + TPD(배치의 5%) + from-scratch 패치 판별기(2M).
- **사용 규칙(F3).** 대역 b의 근사 오차가 0.1·σ_t·√P_ε(b) 미만이거나 b가 미결정이면 TAE를 쓴다. 결정 창의 마지막 스텝은 full VAE를 쓴다.
- 임계 t*와 대역 전달함수는 E2에서 측정한다.

#### M2. G-Adapter-DiT `P1-조건부`

- 쓰는 곳: v1의 `ctrl=Gbuffer`. 주로 bedrock의 높은 t 구간에서 레이아웃을 지키는 데 쓴다.
- **E2에서 높은 t의 SDEdit만으로 레이아웃이 유지되면 생략한다.** 그러면 36 T4-h가 솔버로 돌아간다.
- 학습 방식:
  - 교사는 동결한다.
  - 파이프라인 분할 + gradient checkpointing으로 512²에서 학습한다.
  - 샘플당 순전파 + 역전파 + 재계산 약 48 TF, 3.2 s.
  - 4만 샘플 = 36 T4-h.

#### M5. BRP `P1-코어` (원안 유지, 데이터만 수정)

- 역할: v1 CWM 중에서 "내부·뒷면 포함, 구멍 없음"만 맡는다. 전역 1-pass라서 비회귀 조항의 유일한 예외 범위 안에 있다.
- 데이터:
  - GPU ProcGen 20만(Warp)
  - Infinigen 3천(CPU, 방 하나 8–13분, 지오메트리만)
  - **자체 캡처 lift 300**(ScanNet++/DL3DV 대신)
- 캡션: 문법 템플릿을 TE로 임베딩한다.

#### M17. Glyph-S `P1-코어` [부품/신규 편입]

- **역할.** v1 ℒ_ocr의 OCR을 대체한다. "SYMBOL이 없는 곳에 나타난 글자"에 벌점을 주는 히트맵을 낸다.
  - 을지로 네온 데모의 한글은 SYMBOL(MSDF)이 구성상 맡는다.
- 데이터:
  - 한글 11,172자 + 라틴 문자, OFL 폰트 수십 종
  - 우리 렌더러로 간판, 종이, 화면 재질 위에 렌더한다
  - 교사로 운송해 사진 질감을 입힌다
- TPR/CER 평가는 외부 OCR을 평가 전용으로 써도 된다.

#### M8–M11. DP-0 소형 3종 + 1 (원안 유지, 보정)

- **Hand-S.**
  - 평행이동 성분은 OpenVINS(고전)로 얻는다. ARKit은 쓰지 않는다.
  - 30일 안에 40–60 h, 90일 안에 300 h를 모은다. 폰 4대와 GoPro 2대를 동시에 기록한다.
- **Path-S.** 데이터에 **VCam 퍼포먼스 캡처** [조합]를 추가한다.
  - 현업 오퍼레이터가 실제 방에서 폰과 GoPro 리그로 무브를 연기한다.
  - 영상 + IMU를 OpenVINS/GLOMAP으로 처리하면 의도와 손 데이터를 동시에 얻는다.
  - 퍼블릭 도메인 영화는 오래된 카메라 문법이라 보조로만 쓴다.
- **CRM-S.** saliency는 학습 모델을 쓰지 않고 해석적으로 계산한다.
  - 구성: 장면 그래프 주체 태그 + 휘도 대비 + 중심 편향 + MRH Aes
  - 수제 지각 64차원 @10 fps

#### M7. RAFT-S `P1-확장`

- 고전 DIS/LK가 1순위다.
- RAFT-S는 Phase 2 실사 lift에 필요해지면 학습한다(3.5 h). 데이터는 우리 렌더러로 무제한 만든다.

#### P2 (요약)

- EPD-S: 물리 sim으로 데이터를 무제한 만든다.
- MoPrior-S(10M): CMU mocap + 자체 멀티캠 mocap.
- Kp2D-S: 합성 인간으로 학습하고, 다시점 삼각측량(고전)으로 보정한다.
- F_φ-S: 자체 rig로 sim-to-real 쌍을 모은다.
- 비디오 증인은 없고, **H7이 본선**이다.

### 5.7 솔버 통합: Spec-WSSC v3-T4

```text
procedure SpecWSSC_T4(y, mode):
  # 0. Picture-First Bedrock (전역 1-pass: 비회귀 조항 3의 유일한 예외)
  P   ← CaptionComposer(y)                       # 템플릿 + 사전, 모델 없음
  H   ← Teacher.sample(P.hero, 1024², 8 NFE) × 1–4   # hero still = 관측 제약(DPS), 캔버스 아님
  G   ← MRH(taps(H)) → depth/normal/intrinsics/sem ; metric ← SizeDict ⊕ focal head
  V0  ← BRP(lift(G), TE(P))                       # 64³ 완성, 내부·뒷면 포함
  W   ← BedrockLockstep(V0, 48 cine views @768², K=3, WNF, G-Adapter?)   # T5: 평균 안전 대역
  S   ← SwatchBank(W.materials, L_T, seed)        # (seed, m, l, y)의 순수 함수, bedrock과 함께 commit
  D1080, D4K ← RayBundleDemand(DP0.prior(W), 1080p), (…, 4K)
  L_T ← levels(D1080) ; L_S ← levels(D4K) \ L_T   # 판정 해상도 분할

  # A. 교사 관할 레벨: 스케일 순서, 레벨 안에서는 Jacobi lockstep
  for l in L_T:
    V ← BandIsoViews(D1080[l])                    # η≈0.25, κ=4
    for r in 1..R_l:
      W_l^d ← ICX_LBO(W_{<l}, Gbuf, WNF(seed,l), S[·,l])          # 모든 블록 동시에 초안
      acc   ← Verify(Teacher, V_{κv=2}, Render(W_l^d), t_end(b), WNF′)  # 1 NFE/뷰, DR_rel≤1.2 ∧ seam
      Rej   ← blocks ∖ acc                        # 라운드 상태의 순수 함수
      W_l[Rej] ← Ownership(V(Rej), [t_s,t_e] 스냅, K=3,
                            enc/dec by NoiseMaskRule, ChartDirectSync,
                            apron = W_l[acc] (Dirichlet, 유계 반경))
      checkpoint(MerkleRoot(W), r)                # Kaggle 재개 지점
    HardCommit(l): graph cut + multiband blend, seed 동결

  # B. 학생 관할 레벨: native, 결정론
  for l in L_S: W_l ← ICX_LBO(...) ; Audit 5% (1 NFE)

  # C. Polish/Audit: AdvCams(CMA-ES) → 1-NFE DR → 실패 블록만 Ownership
  # D. Shoot: DP0.search(analytic stats, 10³) → top-10 720p → CRM → 4K render → lens/sensor
```

**Chart-Direct Sync** [조합]
- 대상: ℓ ≥ l_g+2이고 기하가 동결된 레벨.
- band-iso 뷰는 chart의 정사영이므로 BP는 UV lookup(선형)으로 끝난다. 동기화 과정은 다음뿐이다.
  - 소유권 마스크로 Laplacian pyramid blend
  - 뷰별 gauge(3×4 affine) 최소제곱
- 반복 최적화가 없다. 5×10⁶ texel이 1 s 이하다.
- 결과적으로 GPU1이 비어서 두 번째 교사를 올릴 수 있다.
- 부수 효과: band-iso 뷰는 z비 ≤ 1.19라서 DTC의 DOF 문제와 구조적으로 무관하다.

**비회귀 조항 점검**

| 조항 | Spec-commit | Swatch Bank | ICX-LBO |
|---|---|---|---|
| 1. 방문 순서 간선 없음 | 검증과 거부 집합이 라운드 상태의 순수 함수(Jacobi) | 수요 독립 순수 함수 | 고정 타일 격자 |
| 2. 다른 시점의 렌더 픽셀을 조건으로 쓰지 않음 | 검증은 세계 자신의 렌더를 판정할 뿐이다. 조건 입력이 아니다 | **해석 확장을 명시한다.** 견본은 bedrock commit의 재질 정의(SMF τ의 고해상판)다. 조건 출처는 더 거친 commit 레벨이다 | 부모 레벨, WNF, 견본 |
| 3. completion은 스펙트럼적 | 영역을 넓히지 않는다 | 영역 없음 | 2×2 평균 0 사영 |
| 등급 | solver OIE ≤ 0.05 | 비트 동일 | **native 비트 동일(T8 복구)** |

CI에서 `order_purity_check`를 돌린다. 수요 순서를 8번 섞고 Merkle 루트를 비교한다.

### 5.8 데이터 주조소 v2

#### 5.8.1 파이프라인

```
[Tier-1 주조]  GPU ProcGen(Warp, GPU1 유휴 시간) + CPU VM(Infinigen 지오메트리) + CC0 재질·에셋
      │ scene program = 라벨
[Tier-C 캡처]  미러리스 RAW 500장면 · IMU 300 h · VCam 퍼포먼스 · (P2) 멀티캠 mocap
      │ COLMAP 4.x SIFT + GLOMAP + Caspar BA + PatchMatch MVS + FastGS  → GT-classical-real
[GPU raster]   3DGUT/OptiX: G-buffer + 셰이딩 (AI 디노이저 off)
[Tier-2 교사]  운송 / 실사 노이즈 탭 / band-iso 타일 / 견본 / X1-slim
      │ Feature Tap Bus: Frozen PCA 탭(6 MB) → 링버퍼 → 헤드 온라인 학습 (기생)
[Tier-3 솔버]  Spec-WSSC → W* → chart 타일 → ICX-LBO, MRH, BRP (expert iteration)
[Tier-H 사람]  선호 쌍, 전문가, 2AFC
[Tier-P 물리]  Warp/MuJoCo-Warp → 사건 → EPD-S, F_φ-S
      ▼
Merkle 주소 WebDataset shard + 라벨 계보 장부
```

#### 5.8.2 하루 생산량 (fp16 기준, INT4는 ×1.6–1.8)

| 생산물 | 생산량 | 90일 계획 |
|---|---|---|
| 교사 운송 512² / 768² | 2.5만 / 1.26만 per T4-day | 15만 (12만 / 3만), 170 T4-h |
| 실사 노이즈 탭 | 2만 장 × t 2개 per T4-day | 6만 장, 36 T4-h |
| band-iso 타일 768² | 141만 타일/T4-day | 300만, 51 T4-h |
| 견본 512² | 2만/T4-day | 학습용 2만 + 세계별 견본 |
| W* | G0 8 → G3 48 per T4-day | 약 1,360 |
| 실사 lift | 장면당 0.4 T4-h. 실제 병목은 캡처 속도(1인당 하루 약 8장면) | 500 |
| voxel 장면 (GPU ProcGen) | 100만 이상/day | 20만 사용 |
| 물리 sim | 약 10¹⁰ env-step/day | 500만 에피소드 |

**저장.**
- 운송 샘플 1개는 2–3 MB, 탭은 int8 PCA로 20%만 저장한다.
- 전체 1–1.5 TB다.
- Kaggle에서는 생성 즉시 소비하고, 5만 개 이하만 저장한다.

#### 5.8.3 핵심 기법 (원안의 교사 운송 + v2 추가분)

```
절차 장면 GT ── raster ──> x_proc ── SDEdit(교사, t∈스냅격자[0.3,0.7], 3 NFE) ──> x'
실사 RAW + MVS GT ── VAE enc ──> z ── 노이즈(t) ── 1 NFE ──> taps     (생성 없음)
라벨 유효 대역 = { o : D(o,t) ≤ 0.1 }        # X1 결정 곡선이 라벨 증명서
손실 = Σ_{o∈유효} ‖Π_o(pred − GT)‖ ;  품질 필터: 저대역 불일치 > τ → 폐기
```

#### 5.8.4 라벨 계보 장부

- **태그 목록**: `GT-proc`, `GT-sensor`, `GT-classical-real`, `classical`, `teacher-transport(t, 대역)`, `teacher-swatch`, `teacher-self(MRH vK)`, `solver-W*(gen k)`, `human`, `sim`, `asset-3p`, `sensor-ISP-ML`
- **배치 규칙**:
  - GT-proc + GT-classical-real ≥ 55%
  - 자가 라벨 ≤ 25%
  - `sensor-ISP-ML`은 엄격 모드에서 제외
- **평가 전용**: ScanNet++, DL3DV, NYU

### 5.9 2×T4 배치

#### 5.9.1 Mode A: INT4가 E2 판정 충실도를 통과한 경우 (기본)

| GPU0 "Dreamer-A" | GB | GPU1 "Builder + Dreamer-B" | GB |
|---|---|---|---|
| Z-Image INT4 DiT + low-rank | 3.9 | Z-Image INT4 DiT (lockstep 단계에만) | 3.9 |
| VAE fp16 (mid-block fp32) | 0.3 | STRATA 작업 세트 (chart, UKA, BVH) | 3–5 |
| 활성 768² 배치 2 | 1.6 | 활성 | 1.6 |
| MRH 추론 + 온라인 학습 | 1.5 | 렌더러 (3DGUT + OptiX) | 1–2 |
| TAE-T, G-Adapter, 탭 링버퍼 | 0.7 | ICX-LBO 추론 (64타일) | 1–3.5 |
| MPS 소형 학습 잡 | 3–4 | TE(Qwen3-4B INT8): 프롬프트 트리 일괄 계산 후 해제 (세계당 약 1분) | (4, 일시) |
| **여유** | **약 3** | **여유** | **0–2** |

- 솔버 단계에서 lockstep NFE는 두 장에 나눈다. 그 사이 GPU1은 sync, 렌더, ICX를 돌린다.
- 야간에는 두 장 모두 Tier-2 데이터 엔진을 돌린다.

#### 5.9.2 Mode B: fp16만 가능한 경우

- **교사 파이프라인 분할.** 층 0–14는 GPU0(6.2 GB), 15–29는 GPU1(6.2 GB)에 둔다.
- **처리량.** 마이크로배치 2개를 엇갈려 흘리면 두 장이 모두 가동되어, 단일 장 fp16의 1.7–1.9배가 된다 [추정].
- **나머지 배치.** GPU0에 VAE, MRH, TAE(약 2.5 GB)를 두면 여유는 약 5 GB다. GPU1에 STRATA, 렌더러, ICX(6–8 GB)를 두면 여유는 약 1 GB다.
- **G-Adapter-DiT 학습.** 같은 분할로 각 장 약 11–12 GB가 든다.

#### 5.9.3 공통 사항

- **PCIe Gen3**(약 12 GB/s, P2P가 없으면 host 경유로 약 2배 느림)의 전송량:
  - latent: 128 KB
  - 탭: 6 MB (0.5–1 ms)
  - 파이프라인 경계: 19 MB (1.6–3 ms)
  - 25M 모델 그래디언트 allreduce: 50 MB (약 10 ms)
  - 어느 것도 병목이 아니다.
- **원칙.** 장 2개에 잡 2개다. 소형 모델은 DDP로 나누지 않는다.
- **호스트(상시).** 16–32 vCPU, 64–128 GB RAM, NVMe 2 TB.
- **Kaggle.** Mode A를 우선한다. 교사는 Kaggle Models로 마운트하고, 잡은 8 h 단위로 나누며, Jacobi 라운드마다 체크포인트를 남긴다.

### 5.10 저예산 학습 레시피

1. **기본 정밀도.** fp32 master 가중치 + autocast fp16. GradScaler(초기값 2¹⁶, growth 2000), inf가 나오면 해당 스텝을 건너뛴다.
2. **u-μP**를 transformer 기본값으로 쓴다. QK-RMSNorm, logit soft-cap 30, pre-LN 잔차 1/√(2L).
3. **norm, softmax, 손실, FFT는 fp32.** `allow_fp16_reduced_precision_reduction=False`.
4. **conv 네트**는 GroupNorm/RMSNorm을 fp32로 두고 BatchNorm은 쓰지 않는다.
5. **층별 max-abs 로깅**을 100스텝마다 하고, 2×10⁴을 넘으면 경보를 낸다.
6. **attention.** SDPA mem-efficient 또는 flash-attention-turing을 쓴다. head_dim은 64로 고정하고, dropout은 쓰지 않는다.
7. **툴체인 고정**: PyTorch CUDA 13.x, inductor GEMM 백엔드는 `ATEN`. FlexAttention과 Triton GEMM은 금지(sm<80에서 FMA로 폴백). Triton은 elementwise fusion에만 쓴다.
8. **교사 FP16 Rescue.** 잔차 스트림 1/s, SwiGLU 재균형, fp32 norm/softmax. 채택 조건은 NaN 0/1만이다.
9. **클럭 고정.** `nvidia-smi -lgc`. 지속 클럭과 TF/W는 E1에서 실측한다.
10. **μP 프록시.** 폭 1/4로 32개 설정을 sweep한다. CUDA MPS로 T4 한 장에 잡 4–8개를 겹친다. 실험 단위는 2 h 이하의 멱등 잡이고, SQLite 큐로 관리한다.
11. **무상태 샘플러.** Feistel PRP로 샘플 인덱스를 정한다. 체크포인트는 15분 또는 1,000스텝마다 원자적으로 쓴다(tmp → fsync → rename). 30–60분마다 외부로 업로드한다.
12. **솔버 재개.** Jacobi 라운드 Merkle 체크포인트로 결정론적으로 재개한다.

### 5.11 제품 추론 한 장 (방 규모, 4K, 30 s 샷, 2×T4, G2 기준)

| 단계 | Mode A (INT4 이중) | Mode B (fp16 분할) |
|---|---|---|
| Picture-First: hero 4장 + MRH + BRP | 약 1.5분 | 약 3분 |
| Bedrock lockstep (79 NFE-eq) | 약 1.5분 | 약 3분 |
| Swatch Bank (75 NFE-eq) | 약 1.2분 | 약 2.5분 |
| Strata Spec-commit (약 445 NFE-eq) | 약 7분 | 약 14분 |
| 최미세 ICX-LBO + 5% 감사 | <1분 | <1분 |
| Polish/Audit (110 NFE-eq) | 약 2분 | 약 3.5분 |
| DP-0 (해석 통계 10³ → 720p 상위 10) | 2–4분 | 2–4분 |
| 4K 720프레임 + 렌즈/센서 | 약 5분 | 약 5분 |
| **세계 + 첫 샷** | **약 20–23분** | **약 33–36분** |
| **같은 세계의 추가 샷** | **7–10분** | **7–10분** |

- G3(관할 이양 후)에서는 세계 + 첫 샷이 Mode A 약 15분이다.
- 원안의 "SDXL로 1–1.5 h"보다 빠르면서 교사의 미감은 SDXL보다 +263 Elo 높다.

### 5.12 v1에서 버리기, 미루기, 바꾸기

| v1 항목 | 결정 | 이유 | 대체 |
|---|---|---|---|
| S0 LLM World Program | 폐기 | LLM 금지 | Picture-First + 캡션 작성기(사전/템플릿) + ShotScript 파서 + 크기 사전 |
| WSSC `SelfCaption` | 폐기 | VLM | 캡션 작성기 |
| ℒ_ocr | 교체 | OCR 모델 | Glyph-S 4M |
| CWM 4–8B, LBO 200M, WDM 4–8B, 규모 가설 §9.4 | 축소 | 10²³ FLOPs | BRP 42M + ICX-LBO 26M + Swatch Bank. WDM은 Tile-DMD로 `P1-확장`: 512², fake score = 같은 base + LoRA(가중치 1벌), 파이프라인 분할 |
| WDD (L4 극미세) | 통합 | | ICX-LBO의 학생 관할 |
| PWMT (렌즈 05 원안) | 1순위에서 하향 | 블러/암기, GPU1 경합, T8 상실 | Swatch Bank + ICX. PWMT-FiLM은 대조군 |
| DA3/VGGT-Ω/π3/MegaSaM/MoGe/AnySplat | 금지 | 사전학습 | COLMAP 4.x SIFT + GLOMAP + PatchMatch + FastGS + MRH |
| RAFT/UFM (Σ_u) | 금지 | | DIS/LK, RAFT-S 1M |
| SigLIP/DINO sem | 금지 | | MRH sem (TE 공간) |
| TRELLIS/Hunyuan3D | 금지 | | 객체 스케일 WSSC + CC0/CC-BY 에셋 |
| 닫힌 SOTA art director, 닫힌 API polish | 폐기 | 이미지 모델 1개 원칙 | 같은 교사의 hero still |
| SDXL-Lightning 등 서드파티 few-step LoRA | 금지 | 두 번째 사전학습물 | 공식 Turbo 또는 자체 t-densify LoRA |
| 비디오 증인(Wan) + PLAC | 금지 | | 실제 증인(자체 캡처) + 물리 + EPD-S. H7 본선 승격 |
| P2 lift (GVHMR/WHAM/MANO/FoundationPose/MoSca), SMPL-X | 금지 / 교체 | 사전학습, NC 약관 | Kp2D-S + 멀티캠 삼각측량, 자체 체형 공간, MakeHuman |
| 정적 lift 5천만, 동적 2천만 클립 | 축소 | 컴퓨트, 약관 | 자체 캡처 500장면 + VCam 경로 + 퍼블릭 도메인 |
| X1 2,500장 × t 30 × seed 8 | 축소 | 비용 | X1-slim (E2), 27 T4-h |
| X3 arm (i) SEVA, (iv) Difix3D+ | 평가 전용 축소 | 사전학습, 느림 | 소수 장면 비교만 |
| 3DGRT 2차 광선 | 데모 렌더로 한정 | 3DGS보다 느림 | 거울 복도 데모의 최종 렌더만 (RT 코어 40개) |
| Powers of Ten, 10분 원테이크 | 연기 | 솔버 비용 | ICX가 L3까지 관할 이양을 받은 뒤 |
| 세계 VAE / Atom 토크나이저 | 연기 | 렌더 경로에 학습된 decode가 들어감 | 명시적 texel chart + BC7/JPEG-XL |
| 8K·120fps 기본 | 축소 | 렌더 속도 | 4K 기본, 8K는 히어로 샷만(교사 비용 불변, ICX +1옥타브) |
| MAN 셰이딩, Avatar Foundry, slice DM | 연기 | 교사 판정 비용 | P2b 이후 |
| H100 64장, 0.4–0.8M H100-h | 폐기 | | 4,320 T4-h. 파워포인트 속 데이터센터로 명예 퇴직 |

### 5.13 90일 일정과 게이트

| 주 | 작업 |
|---|---|
| 1 | **E1**, 잡 인프라(무상태 샘플러, 원자적 체크포인트, MPS, 툴체인 고정), FP16 Rescue, Feature Tap Bus + Frozen PCA, GPU ProcRoom v0, 캡처 시작(RAW, IMU) |
| 2 | **E1 결정**, **E2**(X1-slim, INT4 충실도), TAE-T v0, **E5** 캡처·lift, Glyph-S |
| 3 | **E3**(MRH + 실사 MVS 차선), ICX-LBO v0 + 견본 데이터, BRP v0, Hand-S v0, Path-S 문법 데이터, Rig-ID-S |
| 4 | **E4**(Spec-commit, facet world), **E5 평가**, **D30 게이트** |
| 5–6 | 방 규모 Spec-WSSC 통합, Chart-Direct Sync, G-Adapter go/no-go, IPM-S, CRM 합성 사전학습, **G0 W*** |
| 7–8 | **D60 게이트**, G1 W*, expert iteration 1세대(ICX/MRH/BRP), 사람 선호 쌍 |
| 9–10 | G2 W*, 관할 이양 1차(재질 3종을 ℓ*₁₀₈₀로), EPD-S sim 데이터, 필요하면 RAFT-S |
| 11–12 | G3 W*, INT8 TensorRT 배포(학생), 데모 세계 3개, FREEZE-TAG 사전 등록 |
| 13 | **D90 게이트**, 모델 카드 18종, 계보 장부 공개 범위 결정 |

**게이트** (실패해도 목표는 낮추지 않는다. 해당 실험의 "실패하면" 경로를 앞당긴다)

| 시점 | 통과 조건 |
|---|---|
| **D30** | E1: 교사 확정(스틸 SP ≥ 55%, 안정, 속도). E2: (a)(b)(c). E3: 실사 AbsRel ≤ 0.12, 배당 ≥ 25%. E4: NFE ×2.5 이상 감소, ΔDR ≤ 5%. E5: VP ≥ 70% |
| **D60** | 방 세계 end-to-end, NFE-eq/세계 ≤ 1,150. 최미세 ICX 2AFC ≤ 55%, OIE = 0 (native). X3형 HER ≥ 0.8, GeCo ≤ 0.04. Hand-S 카메라 튜링 ≤ 55%. W* ≥ 250 |
| **D90** | **v1 그대로**: 범주 하나 이상에서 SP ≥ 45%와 GeCo ≤ 0.03을 동시에. T2에서 최강 경쟁 모델 대비 VP ≥ 70%. 데모 릴 3편(왕복, 을지로 네온, 불릿타임 헌정: Warp MPM 와인잔). 추가로 NFE-eq/세계 ≤ 760, TJF ≤ 0.6, W* ≥ 1,000. Kaggle 모드에서는 범주 수만 1–2개로 줄이고 기준은 같다 |

### 5.14 리스크 → 업그레이드

| 리스크 | 감지 | 업그레이드 |
|---|---|---|
| 교사 미감 상한 | E1 스틸 SP < 55% | Qwen-Image-2512 INT4, DTC/Gaffer/룩 상승분을 E5에서 측정, hero 관할 1024² 집중 |
| bf16 → fp16 overflow | NaN, 블록 max-abs | FP16 Rescue, 파이프라인 분할 fp16, W8A16 |
| INT4 판정 표류 | E2 ΔD > 0.05 | INT4는 초안과 데이터에만, 검증과 커밋은 fp16 |
| Turbo t-격자가 성김 | 창 > 1.5옥타브 | t-densify LoRA, 레벨 간격 2옥타브 |
| Spec-commit 수락률 저조 | a < 0.5 | 견본 확대, 검증 t 2개, 재질별 관할 차등 |
| 수락 판정 해킹 (DR hacking) | held-out seed DR 괴리 | 검증 seed 분리(WNF′), 5% 무작위 full ownership 감사, 사람 2AFC |
| 견본과 in-situ 불일치 (모서리 때, 마모, 곡률) | 위치별 2AFC | G-buffer 조건 강화, 교사 감사 비율을 영역별 상향 |
| ICX의 "SR 룩" | 스펙트럼 편차, 2AFC | 스펙트럼 손실, TPD, 관할 이양 보류 |
| MRH 실사 일반화 | E3 holdout | 실사 캡처 확대, 평면 스윕 MVS 대체 |
| 자가 라벨 순환 | 센서 GT 괴리 | 계보 비율 규칙, 평가 홀드아웃 고정 |
| T4 쓰로틀 −20–40% | 지속 클럭 | 클럭 고정, 버퍼 17%, 야간 데이터 엔진 |
| sm_75 지원 종료 (다음 CUDA 메이저) | 릴리스 노트 | 툴체인 버전 고정, 컨테이너 스냅샷 |
| Kaggle 약관, 쿼터 계산 방식 | 확인 | 상시 모드 기본, Kaggle은 보조 |
| TE(Qwen3) 사용을 창업자가 불허 | 결정 | SDXL(CLIP TE)로 강등하거나 TE를 프롬프트 인코딩에만 한정. sem과 텍스트 조건은 MRH 시각 임베딩 공간으로 |
| 데이터 약관 (NC 데이터셋) | 법무 | 자체 캡처, 공개셋은 평가 전용 |
| P2 사람 동작 데이터 기아 | 학습 곡선 | 멀티캠 + Kp2D-S + IMU 스트랩. 컴퓨트가 아니라 장비 문제 |
| 저장 1–1.5 TB | 디스크 | 생성 즉시 소비, 탭 20%만 저장 |

### 5.15 신규 아이디어 장부

| # | 아이디어 | 태그 | 이 제약이 아니면 왜 안 나왔나 | 확신도 |
|---|---|---|---|---|
| 1 | **Speculative Band Commit**: 학생이 초안을 쓰고 교사는 1-NFE로 E2 고정점만 판정 | [신규] | 교사 호출이 쌀 때는 매번 정식 소유권을 돌린다 | 중상 |
| 2 | **판정 해상도 분할**: 1080p 판정 경계에서 교사와 학생의 관할을 나눔 | [신규] | 무한 컴퓨트에서는 모든 레벨을 교사가 본다 | 중상 |
| 3 | **Material Swatch Bank + ICX-LBO**: 수요 독립 견본으로 재질 정체성 유지 + T8 복구 | [신규/조합] | PWMT의 실패에서 나왔다 | 중상 |
| 4 | **노이즈 마스킹 근사 규칙**: 근사 enc/dec를 노이즈가 가리는 곳에서만 쓴다 | [신규] | VAE가 NFE만큼 비쌀 때만 의미가 있다 | 중상 |
| 5 | MRH 노이즈 t 독심 (원안) | [조합/신규] | 외부 depth 모델 금지 | 중상 |
| 6 | 기생 학습 + Feature Tap Bus + Frozen PCA | [신규/부품] | 교사 호출이 화폐일 때 | 중상 |
| 7 | **실사 MVS 차선**: 실사 + 고전 GT + 교사 노이즈 탭 | [조합] | 사전학습 depth 라벨러 금지 | 중상 |
| 8 | **FP16 Rescue**: 정확한 재매개변수화로 bf16 교사를 fp16에서 구조 | [조합] | BF16이 없는 하드웨어 | 중상 |
| 9 | 관할 이양 + TJF (원안) | [신규] | 교사가 비쌀 때 | 중 |
| 10 | 교사 운송 + 결정 곡선 라벨 마스크 (원안) | [조합] | | 중 |
| 11 | Chart-Direct Sync | [조합] | GPU1을 두 번째 교사에게 돌려주기 위해 | 중 |
| 12 | Jacobi 라운드 Merkle 체크포인트 | [조합] | 12 h 세션 제한 | 중상 |
| 13 | GPU ProcGen (Warp) | [조합] | Kaggle의 적은 vCPU | 중 |
| 14 | Picture-First + 캡션 작성기 + 크기 사전 | [조합] | LLM 금지 | 중 |
| 15 | 수제 지각, 합성 선호 사전학습, TPD (원안) | [조합] | 비전 백본·LPIPS 금지 | 중 |
| 16 | Kp2D-S 합성 인간 + 고전 삼각측량 | [조합] | 포즈 추정기 금지 | 중 |
| 17 | VCam 퍼포먼스 캡처 (의도와 손 동시 수집) | [조합] | DataDoP 계보 금지 | 중 |
| 18 | 교사 선택 = "SP per T4-h" 토너먼트 + BT 역산 합격선(55%) | [조합] | 교사 호출이 화폐일 때 | 중상 |

### 5.16 제약 덕분에 좋아지는 점

- **반복 속도.** 대부분의 모델은 2 h 안에 학습되고, 카탈로그 전면 재학습도 2일이다. 큰 팀이 체크포인트 하나를 기다리는 동안 우리는 세대를 몇 번 돈다.
- **E2 고정점이 장식에서 연산자로 바뀐다.** v1의 "교사가 고칠 게 없는 세계"라는 정의가 Spec-commit의 수락 테스트로 실제로 쓰인다. 제약이 이론을 기계로 만들었다.
- **T8 비트 동일성이 복구된다.** PWMT 대신 견본 은행을 쓰면서 native 경로가 다시 순수 함수가 된다.
- **Turbo 교사는 CFG가 없어서 NFE가 절반이다.** 원래 단점이던 "증류 모델"이 T4에서는 장점이 된다.
- **재현 가능한 공개 벤치.** FREEZE-TAG 기준선과 우리 파이프라인이 Kaggle 무료 T4에서 돈다. "믿지 마세요, 직접 돌려보세요"라고 말할 수 있다.
- **깨끗한 IP.** 학습된 가중치는 우리 데이터 + Apache 교사 1개 + 고전 알고리즘에서만 나온다.
- **추론이 싸다.** 학생 240M을 INT8로 돌리면 소비자 GPU와 WebGPU에서도 동작한다. 실시간 뷰어가 현실이 된다.
- **나중에 커져도 그대로 간다.** 같은 아키텍처를 μP로 키우면 된다. 데이터 주조소의 shard는 복리로 쌓인다.
- 마지막으로, T4는 원래 추론 카드다. 우리는 그 추론 카드로 판사를 고용하고, 서기는 직접 키운다. 개소리 같지만 숫자상으로는 된다. 판사 출근부(NFE 장부)만 정직하게 쓰면.

---

## 6. 창업자가 결정할 것

1. **교사 TE(Qwen3-4B)의 임베딩 전용 원칙을 승인할 것인가?** 불허하면 sem과 텍스트 조건을 MRH의 시각 임베딩 공간으로 옮기고, 교사 후보에서 SDXL(CLIP TE)의 상대 가치가 오른다.
2. **교사 선택 규칙.** "스틸 SP ≥ 55% 중 SP per T4-h 최대"를 공식 결정식으로 채택할 것인가? 미달하면 Qwen-Image-2512(INT4)를 허용할 것인가?
3. **캡처 투자.** 미러리스 RAW 장비, IMU 장비 $3–5k, 오퍼레이터, (P2) 멀티캠 $3–8k. 컴퓨트가 아니라 장비와 사람이 해자다.
4. **운용 모드.** 상시 2×T4(4,320 T4-h)를 기본으로 하고 Kaggle(780, 최악 390)을 보조로 둘 것인가? Kaggle 약관 확인이 필요하다.
5. **데이터 약관 정책.** NC 공개셋을 평가 전용으로 한정하는 엄격 모드를 기본값으로 둘 것인가?
