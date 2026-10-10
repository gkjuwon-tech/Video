> **출처 메모.** 2×T4 + 수십M from-scratch 제약으로 다시 돌린 설계 워크숍에서 [표현·렌더러·카메라 + Phase 1 승리] 렌즈 담당이 쓴 건설적 프리모템과 개정안이다. 최종 결정은 [`docs/PARADIGM-v2.md`](../../../PARADIGM-v2.md)를 따른다. 이 문서는 Higgsfield Soul 하이브리드 결정 **이전**에 작성됐다.

---

# [렌즈: 표현·렌더러·카메라 + Phase 1 승리] STRATA-lite 건설적 프리모템 → 개정안 v0.2

> **입력.** v1 `docs/PARADIGM.md`(§0.5, §1, §2, §4, §7, §8, §10), STRATA-lite 제안서(동료 렌즈), T4·소형 모델 현실 리서치(2026-10-10), 창업자 하드 제약(2×T4, 빌린 이미지 모델 1개, 나머지는 수십M 파라미터 from scratch, 목표 불변).
> **표기.** **[신규]** 이번 제약 때문에 새로 나온 것 / **[조합]** 부품은 있고 결합이 새로운 것 / **[부품]** 빌려 쓴 것 / **[수정]** 제안서 수치나 설계를 고친 것. "실측 필요"는 X-T0(1주차)에서 확정한다.
> **태도.** 제안서의 뼈대는 맞다. 세계를 한 번 만들고, 결정론적으로 렌더하고, 물리 카메라로 찍고, 가난을 미장센으로 쓴다. 이 문서는 그 뼈대가 Day 90에 어디서 부러지는지 먼저 부검하고, 부러질 자리마다 철심을 박는다. 후퇴는 하지 않는다. 철심 값은 T4-시간으로 적는다.

---

## 0. 프리모템 요약

1. **컴퓨트 회계의 골격은 맞다.** 렌더, 카메라, 소형 모델 학습 숫자는 ±30% 안에 든다. 90일 2×10²⁰ FLOPs(장당 13 TFLOPS)는 리서치 권고치(15–20 TFLOPS)보다 보수적이므로 그대로 쓴다.
2. **틀린 숫자는 9개다.**
   - (a) l=14에서 1024² band-iso 뷰의 폭은 6.4 cm가 아니라 **0.5 m**다. 그래서 미세 레벨 뷰 수는 25–50개가 아니라 **70–200개**(deep focus)다.
   - (b) 미세 레벨 Adam 상태는 1.9 GB가 아니라 **10–19 GB**다. OOM이 난다.
   - (c) TexelFM을 full-res conv-UNet으로 만들면 텍셀당 FLOPs가 SDXL과 **같다**. "8배 저렴"은 patchify 설계를 해야 성립한다.
   - (d) 8K EXR 40–80 MB는 beauty 단독일 때의 값이다. AOV를 포함하면 **0.5–0.7 GB/프레임**이다.
   - (e) 미니 벤치에서 seed를 대칭으로 맞추면 우리 세계는 80개가 아니라 **160개**다.
   - (f) GeoHead 추론은 0.4 s가 아니라 0.6–0.9 s다.
   - (g) GB-Adapter 학습은 13 T4-h가 아니라 30–85 T4-h다.
   - (h) RT 처리량 계산이 자기모순이다.
   - (i) Kaggle 가용량은 390–780 T4-h다. T4×2 쿼터 소모율이 미확인이다.
3. **가장 치명적인 실패 지점은 FLOPs가 아니라 교사 천장이다.** SDXL의 Elo는 678이다. 경쟁 비디오 프레임 미감을 Elo 850–950으로 가정하면 기대 승률이 17–27%다. 일관성을 정의로 이겨도 SP ≥ 50%는 수학적으로 거의 닫힌다.
4. **업그레이드 묶음은 다음과 같다.**
   - **[신규] Teacher Ceiling Test**: Day 18 게이트.
   - **Z-Image-Turbo INT4 대칭 운용**: 실패 시 σ-정밀도 카드 분할로 간다.
   - **[신규] OTBI + TABP**: 미세 레벨에서 소유 타일링과 닫힌 형태 역투영을 쓴다. Adam이 없어진다.
   - **[조합] Federal Router**: l≥13의 재질 셀은 TexelFM이 맡는다.
   - **[신규] MassingFM (21M)**: 3D prior 없는 bedrock의 구멍을 막는다.
   - **patchify TexelFM**.
   - **TL은 commit에 쓰지 않는다.**
5. **결과.** 4K 방 규모 세계 1개가 **1.0–1.6 T4-h**(Mode A)다. 비용은 제안서와 같은 수준인데, 교사는 Elo 678 SDXL이 아니라 **Elo 941 Z-Image**다. INT4가 실패하면 3–6 T4-h(Mode B)로 오른다.
6. **30일 결정적 실험은 5개이고 약 155 T4-h가 든다**(2×T4 상시 기준 30일 가용 약 1,080 T4-h의 14%).
7. **제약 위반·회색지대를 11곳 찾았다.** LPIPS 효도 손실, OCR 손실, CLIP 이미지·SigLIP `sem`, Cycles AI 디노이저, Fun-ControlNet, 제3자 SDXL 파생물, ALIKED/LightGlue, 교사 텍스트 인코더의 LLM식 사용, 평가 모델을 최적화 오라클로 쓰는 것, 그리고 회색지대인 DataDoP와 ARKit다. 전부 from-scratch 대체안이나 운영 규칙으로 막았다.
8. **최종 승리 조건(v1 §7.1)은 한 글자도 바꾸지 않는다.**

---

## A. 숫자 재검증표

| # | 항목 (제안서 §) | 제안서 값 | 재검증 값 | 근거·계산 | 판정 |
|---|---|---|---|---|---|
| A1 | 90일 FLOPs (§1.1) | 2×10²⁰ @13 TFLOPS/장 | 2.0–3.1×10²⁰ | 리서치: HGEMM 실측 피크 41.6 TFLOPS(1590 MHz 고정), 지속 시 쓰로틀. 학습 실효 15–20 권고 | 유지 (보수적) |
| A2 | v1 대비 비율 | 1/3,000–1/6,000 | Phase 1 누계(6–12×10²³) 기준이면 맞다. v1 90일 계획(75–90k H100-h ≈ 1.1–1.3×10²³) 기준이면 1/400–1/600 | 두 기준을 병기한다 | 보완 |
| A3 | 1 H100-h 환산 | 25–30 T4-h | 20–31 T4-h | 1.44×10¹⁸ ÷ (13–20 TFLOPS × 3600) | 수정 |
| A4 | Kaggle 90일 | 780 T4-h | **390–780 T4-h** | T4×2가 쿼터를 1배로 소모하는지 2배로 소모하는지 미확인. 주 30h는 비공식 | **수정** |
| A5 | SDXL 1024² forward | 0.6–0.8 s | 0.55–0.8 s | Fooocus Colab T4: 2장에 74 s, 30 step, CFG → forward당 약 0.55 s | 유지 |
| A6 | SDXL 512² hero search | 0.16 s | 속도는 맞다. **품질 판정은 틀린다** | SDXL은 512²에서 품질이 무너진다(리서치) → 후보 순위가 왜곡된다 | 수정: SDXL이면 768² 이상 |
| A7 | Z-Image-Turbo T4 | step당 4–6 s | fp16 forward **3.1–4.1 s**(1024²), 0.75–1.1 s(512²). INT4(Nunchaku Turing) **1.4–2.2 s** | forward당 61 TFLOPs ÷ 15–20 TFLOPS. INT4 실측 114 TOPS(FP16의 2.7배), 저랭크 분기 오버헤드로 1.6–2.5배 가정 | 수정, 실측 필요 |
| A8 | 1024² 뷰 폭, l=14 (§2.2) | 6.4 cm | **0.5 m** | 셀 128개 × $h_{14}$=3.9 mm (f8 latent 토큰 1개 = 셀 1개 = 텍셀 8×8) | **수정** |
| A9 | 4K 방 l=14 뷰 수 (§6) | 25–50 | deep focus **70–200**, 얕은 심도 Directed 25–80 | demand 면적 10–40 m² ÷ (0.25 m² × 0.77 apron 효율) × 1.3(가림·잡동사니) | **수정** |
| A10 | 4K 방 교사 레벨 뷰·forward | 약 70뷰, 210 forward | **200–420뷰, 600–1,300 forward**(K=3, 라우터 없음) | 레벨별 면적: l=13 40–80 m², l=12 80–120 m², l=11 약 150 m² | **수정** |
| A11 | 세계 1개 (SDXL) | 순수 0.41 / 오버헤드 포함 1.0–1.4 T4-h | 순수 0.55–0.8 / **1.4–2.4 T4-h** | A10 반영 | 수정 → §B F2 업그레이드 후 0.8–1.4 |
| A12 | 활성 레벨 Adam (§2.6) | 장당 1.9 GB | l=14 텍셀 1–2×10⁸ × 6ch × 16 B = **10–19 GB**(장당 5–10 GB) | param, grad, m, v가 fp32 | **수정 → TABP로 제거** |
| A13 | Oracle 방 TSQ·텍셀 (§2.5) | TSQ 1–3M, 텍셀 3–8×10⁷ | TSQ×64이면 0.6–1.9×10⁸이라 자기모순. deep focus 4K 방은 텍셀 1.5–3×10⁸, **0.5–1.0 GB**(모든 레벨 ×1.33) | 2 B/텍셀(BC7+BC5), 32 B/TSQ | 수정 |
| A14 | 8K 방 메모리 | 0.6–1 GB | **1.2–2.8 GB** | 8K에서는 l=15(0.24 mm)가 2.6 m 이내 표면 10–30 m²에 필요 | 수정 (T4에 들어감) |
| A15 | TexelFM 속도·이득 (§5.1) | 10⁸ 텍셀 60–90 s, 교사보다 약 8배 저렴 | full-res conv-UNet 30M이면 텍셀당 4–7 MFLOPs로 SDXL(픽셀당 약 6 MFLOPs)과 비슷하다. **10⁸ 텍셀에 2–5분/T4, 이득 약 1배.** patchify-4로 바꾸면 0.5–0.9 MFLOPs/텍셀, **20–40 s/T4**. SDXL 대비 8배, Z-INT4 대비 16배 | latent f8은 위치 수가 64배 적다. 픽셀 공간 네트워크는 비싸다 | **설계 수정** |
| A16 | GeoHead 추론 | 0.4 s @1024² (UNet 포함) | SDXL 0.6–0.9 s / Z-Image fp16 3.1–4.1 s. **WSSC 스텝과 같은 forward의 특징을 공유하면 추가 비용 약 20 ms** | UNet 단독이 이미 0.55 s 이상 | 수정 |
| A17 | GB-Adapter 학습 | 13 T4-h (6만 × 0.75 s, 1 epoch) | SDXL **30–50**, Z-Image **50–85** T4-h | 3–5 epoch. 샘플당 forward + 재계산 + activation-grad backward(가중치 grad 없음) ≈ forward의 3–4배 | 수정 |
| A18 | 4K secondary RT (§3.4) | 0.04–0.15 s (1–3×10⁷ ray/s) | 자기모순: 3.3M ray ÷ 1–3×10⁷ = 0.11–0.33 s. **TSQ 불투명 코어를 삼각형 closest-hit로 쏘면** 3–10×10⁸ ray/s(추정) → **15–60 ms** | Gaussian-particle 처리량을 삼각형 경로에 잘못 적용했다 | 수정 |
| A19 | 4K 프레임 총합 | 0.25–0.55 s | 0.28–0.63 s | 래스터: 4090 대비 5–8배 느림(리서치 환산) | 소폭 수정 |
| A20 | 8K120 10초 | 12–26분 | 13–30분 | A19 반영 | 소폭 수정 |
| A21 | 8K EXR (§3.6) | 프레임당 40–80 MB(AOV 포함), 10s@120 = 50–100 GB | beauty(DWAA)는 40–80 MB 맞다. AOV(깊이 32f, 법선, 모션, cryptomatte 3랭크) 무손실 포함 시 38 B/px × 33 Mpx → ZIP 후 **0.5–0.7 GB**, 10s@120 = **600–840 GB** | cryptomatte와 깊이는 손실 압축 금지 | **수정 → 저장 정책** |
| A22 | 에너지 4K120 10초 | 12 Wh | GPU만 8–18 Wh, 호스트 포함 16–37 Wh | 70 W × 2 × 3.4–7.6분 | 보완 (정직하게 병기) |
| A23 | 미니 벤치 우리 세계 (§8.2) | 80 × 2 = 160 T4-h | seed 대칭이면 **세계 160개 × 1.5–2.5 = 240–400 T4-h**(벽시계 5–8일). 클립 640개는 맞다 | 경쟁사는 seed 2개 = 영상 재샘플. 우리 seed는 세계 재샘플이어야 공정하다 | **수정** |
| A24 | GeCo 평가 | 2,560 × 30 s = 21 T4-h | VGGT를 fp32로 강제하면 클립당 30–60 s → 21–43 T4-h. fp16이 안정적이면 11–21 | VGGT 1.2B, 24프레임 × 1,369 토큰 전역 attention | 수정 |
| A25 | "교사 12B→2.6B ≈ 5배" (§6) | 5배 | FLOPs 약 16배(98 vs 6 TF), 시간 약 8–10배 | DiT GEMM이 UNet conv보다 효율적이다 | 수정 (결론은 같다) |
| A26 | CamFM-Hand 학습 | 4×10¹⁶, 1 T4-h | 약 1×10¹⁷, 2–3 T4-h | 100 h, stride 0.5 s, 50 epoch | 소폭 수정 |
| A27 | Ledger 엔트리 | 128 KB | SDXL은 맞다. Z-Image(16ch)는 512 KB → hot 5만 = 25 GB → int8로 12.8 GB | — | 수정 |
| A28 | "SDXL은 few-step 증류 불필요" | — | 맞다. 다만 Lightning/DMD2/Hyper LoRA는 **제3자 사전학습물**이라 어차피 금지(§C) | — | 유지 + 규칙화 |
| A29 | 교사 계약 "f8 공간 정렬" | — | Z-Image(Flux 계열 16ch f8) 충족, klein(FLUX.2 32ch f8, [2차]) 충족. DiT 토큰은 2×2 패치(16 px)지만 WNF는 latent 화소 단위로 적분하므로 문제없다 | — | 유지 |

---

## B. 실패 지점 7개와 업그레이드

### F1. 교사 천장: 일관성은 이기고 미감에서 진다

**부검 시나리오.** Day 90에 GeCo 0.025로 일관성은 압승한다. 그런데 SP가 31%에 그친다. 블라인드 평가자 코멘트는 "깨끗한데 2023년 사진 같다"다.

**기술적 원인.** 프레임 미감의 상한은 교사다. 렌즈와 필름 스택은 가산점을 얹을 뿐이고 상한을 바꾸지 못한다. Elo 기대 승률은 다음과 같다(비디오 프레임의 Elo 등가치는 미지라서 가정이다).

| 교사 | Elo | vs 상용 1위(1198) | vs 비디오 프레임 가정 850 | vs 가정 950 | 라이선스 |
|---|---|---|---|---|---|
| SDXL | 678 | 5% | 27% | 17% | OpenRAIL++-M |
| FLUX.2 klein 4B | 863 | 13% | 52% | 38% | Apache 2.0 |
| **Z-Image-Turbo** | **941** | 19% | **63%** | 49% | Apache 2.0 |
| Qwen-Image-2512 | 999 | 24% | 70% | 57% | Apache 2.0 |

TP(우리 렌더 vs 교사 스틸)가 45%라면 우리 SP는 교사 스틸의 승률보다 낮아진다. DTC 광학 스택이 얹는 가산점을 +5–10%p로 가정해도(X-T1에서 실측), SDXL로는 25–35%p의 구멍을 메울 수 없다.

**업그레이드.**
1. **[신규] Teacher Ceiling Test (X-T1, Day 18 게이트).** 교사 스틸 best-of-64와 경쟁 비디오 프레임을 블라인드로 단일 프레임 비교한다. C1에서 60% 이상이어야 통과다(TP 손실분 10%p 마진). 대형 팀은 이 사실을 월드 파이프라인을 다 만든 뒤에야 안다. 우리는 1.5 T4-day로 3주 차에 안다.
2. **교사 선택 규칙 변경.** "수용 세계당 교사-초 × SP"를 최소화하던 것을 **"천장 ≥ 60%라는 제약 아래에서 교사-초를 최소화"**로 바꾼다.
   - 기본은 **Z-Image-Turbo**다.
   - 예비는 **klein 4B**다. multi-reference가 내장되어 hero 스틸을 모든 뷰의 스타일 앵커로 쓸 수 있다. hero는 $y$의 일부라서 비회귀 조항을 지킨다.
   - 승급 경로는 **Qwen-Image-2512 INT4**(약 11 GB, 세계당 3–5 T4-h)다.
   - SDXL은 1–2주차 파이프라인 개발용 **엔지니어링 대역**으로만 쓴다. 교사 교체는 전체 교체만 허용한다(dreamer 계약상 파이프라인은 교사와 무관하게 설계되어 있다).
3. **속도 손실 회수.** Nunchaku W4A4(Turing INT4)를 쓰고, 두 장에 교사를 하나씩 올린 대칭 운용(Mode A)을 한다. 실패하면 σ-정밀도 카드 분할(Mode C)로 간다(§E2).
4. **교사 위에 얹는 미감 레버.**
   - DTC 광학 스택(학습 없음).
   - hero search N=64–256, 점수는 AesHead와 **[조합] Teacher-DCA**의 앙상블. Teacher-DCA는 Diffusion Classifier식 미감으로, 파라미터 0개이고 리워드 해킹에 내성이 있다.
   - 자체 시네마틱 LoRA(rank 32, 직접 촬영한 프레임 2천 장, 약 17 T4-h). 창업자 규칙의 "가벼운 어댑터" 범위다.
   - Directed 데모는 사람이 best-of-4를 고른다.

### F2. 교사-초 회계와 메모리가 동시에 터진다

**부검 시나리오.** 3주차 첫 생성 세계가 예상 1시간 대신 6시간 걸린다. 그나마 l=14 피팅에서 OOM으로 죽는다.

**기술적 원인.**
- (i) A8 오류(0.5 m인데 6.4 cm로 적음)와 demand 면적을 과소 추정해서 미세 뷰 수가 4–8배 모자라게 잡혔다.
- (ii) v1의 블록당 4–8뷰 중첩 투표를 그대로 두면 뷰 수가 다시 2–4배 는다.
- (iii) l=14 텍셀 10⁸개 이상에 Adam을 걸면 10–19 GB가 필요하다(A12).

**업그레이드.**
1. **[신규] OTBI (Owner-Tiled Band-Iso).**
   - 미세 레벨(l ≥ 12)에서는 band-iso 뷰 하나가 표면 차트 하나를 **단독 소유**하고, 이웃과는 12.5% apron으로만 겹친다.
   - 근거는 v1 T6(Partition Principle)이다. 소유 영역 안의 스펙트럼 보존은 $K$와 무관하다. 또 저대역은 이미 부모 레벨에서 커밋되어 있으므로 중복 투표로 얻는 것이 거의 없다.
   - 볼록하지 않은 소품만 4–6뷰를 쓰고, 소유권은 v1의 $a_k$ 점수로 정한다.
   - 뷰 수는 $N_l\approx1.3\,A_l/(0.77\,(1024\tau_l)^2)$이다.
   - 뷰 사이에 조건 간선이 없고, 같은 스텝의 동시 출력끼리 commit 시 multiband로만 섞이므로 순차 생성이 아니다.
2. **[신규] TABP (Texel-Aligned Back-Projection).**
   - band-iso 뷰는 정의상 화소 footprint가 텍셀 피치의 ±¼ 옥타브 안에 있고 θ≤33°다. 그래서 역투영이 거의 affine 리샘플이다. 대역 결정을 경사하강 대신 **닫힌 형태**로 할 수 있다.
     $$P_l(t)=\mathrm{Up}(P_{l-1})(t)+\sum_v\tilde m_{v,l}(t)\,\Pi_{b_l}\!\Big[g_v\big(\hat x_v(\pi_v(t))\big)-\mathrm{Up}(P_{l-1})(t)\Big]$$
   - $\Pi_{b_l}$: 파장 $[2\tau_l,4\tau_l]$의 Laplacian 대역.
   - $\tilde m$: 결정 중에는 $\mathrm{Blur}_{\lambda_b}(\mathrm{softmax}_v\tau_sa_v)$, commit 시에는 graph cut hard + multiband.
   - $g_v$: 3×4 색 gauge. 저대역에서 최소제곱으로 푼다.
   - $\pi_v$: G-buffer 깊이로 가시성을 검사하는 투영.
   - 법선과 높이는 같은 forward의 GeoHead 출력 $\hat g_v$를 같은 마스크로 융합한다. 높이는 Frankot–Chellappa 적분(고전)으로 구한다.
   - **효과.** Adam 상태가 0이 된다. 메모리는 현재 타일 배치의 누산기(0.5 GB 이하)뿐이다. 뷰당 5–10 ms이고 피팅 3 iteration(약 150 ms)을 대체한다.
   - Adam이 남는 곳은 bedrock, 실촬영 피팅, l≤11 시네마틱 뷰뿐이고, 이 레벨들은 거칠어서 메모리가 작다.
   - 대역 밖 저주파는 부모로 고정되므로 색·조명 드리프트가 구조적으로 0이다.
3. **[조합] Federal Router.**
   - l≥13에서 셀 단위로 교사와 TexelFM 중 하나를 고른다. 의미 셀(얼굴, 손, 간판 주변, 각인 있는 소품, CURVE 인접)과 구조 점수가 높은 셀(국소 방향 결맞음, 에지 밀도)은 교사가 맡고, 나머지 재질 셀은 TexelFM이 맡는다.
   - 임계값은 v1의 관할 규칙 $\mathrm{owner}=\arg\min_P\Delta_P+\mathrm{Var}_P$에 따라 **측정으로** 정한다(X-T3). 교사 몫은 25–40%로 예상한다.
4. **결과 (4K 방, deep focus).** 교사 뷰가 200–420개에서 **100–220개**로, forward가 600–1,300회에서 **250–480회**(K=2–3)로 준다. 교사를 Z-INT4로 바꿔도 세계당 비용은 SDXL 원안 수준이다.

### F3. 작은 AE(TL)가 지금 결정 중인 대역을 지운다

**부검 시나리오.** X3식 측정에서 HER이 0.62에서 멈춘다. 교사 x̂0 latent는 선명한데 세계에는 반쯤 흐린 텍스처가 쌓인다.

**기술적 원인.** band-iso 뷰에서 결정 중인 옥타브는 화소 파장 2–4 px, 즉 latent→픽셀 디코딩의 **최상위 옥타브**다. TAESD형 2–3M 디코더는 원 VAE보다 이 대역 복원력이 약하다(실측 필요). 결과적으로 가장 중요한 대역을 가장 약한 부품이 디코딩한다. 인코더 근사 오차는 교사 입력을 분포 밖으로 밀어 후광과 아티팩트를 만든다.

**업그레이드.**
1. **규칙: TL은 미리보기 전용이다.** hero early-stop, Gaffer/CRM, 중간 스텝의 저·중대역 재인코딩에만 쓴다. **결정 대역을 쓰는 commit 스텝(k=K)은 원 VAE로 디코딩한다.**
2. **[조합] 원 VAE fp16 자체 수리.** activation 스케일을 재분배하고, 필요하면 디코더를 짧게 미세조정한다. 이것은 교사에 대한 가벼운 어댑터 범위다. 제3자 `fp16-fix` 가중치는 쓰지 않는다. 1024² 디코딩이 fp32 1.5–2 s에서 fp16 0.3–0.5 s로 준다.
3. **TL 학습 목표에 교사 일관성 항을 추가한다.** $\lVert D_\theta(\mathrm{Enc_{TL}}(x),t)-D_\theta(\mathrm{Enc_{VAE}}(x),t)\rVert$, 즉 같은 t에서 교사 x̂0가 일치해야 한다.
4. **비용.** commit 디코딩은 뷰당 0.4 s × 100–220뷰 = 1–1.5분이다. 세계 비용 대비 2% 이하다.

### F4. Bedrock이 비어 있다: 3D prior 없이 360°와 뒷면을 만들 수 없다

**부검 시나리오.** FREEZE-TAG 오빗 의도에서 hero 프러스텀 뒤쪽 벽이 뷰마다 다른 방이 된다(Janus). 문 높이가 1.3 m인 방도 나온다. f/1.4 랙 포커스의 CoC가 틀려서 보케가 장난감 같다.

**기술적 원인.**
- CWM 4–8B를 폐기한 뒤 bedrock은 hero lift, 매싱, 교사 coarse lockstep뿐이다. hero 밖 영역은 교사가 coarse 뷰에서 **구멍을 메우는** 구조가 된다. 동기화가 있어도 레이아웃 대역의 소유자가 없다.
- GeoHead는 log-depth만 낸다. 스케일이 모호하고 FOV를 모른다. 그런데 STRATA 레벨도, DOF 수식 $c(z)$도 **미터**를 요구한다.

**업그레이드.**
1. **[신규] MassingFM (21M, from scratch): 레이아웃 소형 파운데이션.**
   - DiT 12층, d=384, head_dim 64.
   - 출력: shell 평면 ≤32개(바닥, 천장, 벽, 개구부), 객체 slot ≤64개(class, OBB, shape code 16D), 잔차 점유 32³(4³ 패치 → 512 토큰).
   - 조건: 교사 TE 임베딩 8토큰(encoder 전용), hero lift 부분 점유(masked).
   - 연속값은 flow matching, 이산값은 masked diffusion으로 학습한다.
   - 데이터: 자체 절차적 RoomGen/StreetGen(규칙 기반 배치, 클리어런스 제약, CGA식 파사드 문법) 300만 샘플(CPU 몇 시간), Infinigen 실내 5천 개, 자체 MVS 부분 관측(마스크 증강). 캡션은 scene graph 템플릿에서 생성한다.
   - 데이터 라벨이 무한하고 공짜다. 수십M 모델이 강한 조건(좁은 연산자 + 합성 라벨)이 그대로 성립한다.
2. **Janus 규칙.** 레이아웃 대역의 소유자는 정확히 하나다. hero 프러스텀 안은 hero, 밖은 MassingFM이다.
3. **coarse ring 16뷰 lockstep.** MassingFM 덩어리의 G-buffer를 GB-Adapter로 조건화한다. 교사의 일이 "빈 곳 채우기"에서 "**이미 있는 덩어리에 옷 입히기**"로 바뀐다.
4. **GeoHead에 FOV head를 추가하고, [조합] semantic metric prior를 쓴다.** MassingFM slot 클래스별 크기 분포(문 2.0–2.1 m, 좌석 높이 0.42–0.48 m 등)로 전역 스케일을 최소제곱으로 푼다(고전).
5. **비회귀.** MassingFM은 전역 1-pass이고, v1의 유일한 허용 예외(bedrock) 안에 있다.

### F5. TexelFM: 비용 주장 오류와 "그럴듯한 노이즈"

**부검 시나리오.** 8K 클로즈업에서 TexelFM이 간판 볼트 주변과 사람 손등에 나무결 같은 노이즈를 깐다. 그나마 생각보다 3배 느리다.

**기술적 원인.**
- (a) A15. 픽셀 공간 conv-UNet은 텍셀당 FLOPs가 latent DiT와 비슷하다.
- (b) 라우팅 없이 l=15를 전부 맡기면 의미 셀에서 2AFC가 실패한다.
- (c) 렌더 중 지연 평가를 하면 Proof-of-Shot이 TexelFM의 GPU 간 결정론에까지 의존하게 된다.

**업그레이드.**
1. **patchify-4 (space-to-depth) 설계.** 256²×6을 64²×96으로 접고, conv-UNet 3단(192/288/384ch)에 16² 토큰 attention(head_dim 64, flash-attention-turing)을 붙인다. 256² 타일 forward당 약 45 GFLOPs다.
2. **Federal Router(F2)로 의미 셀을 교사에게 돌려준다.**
3. **[신규] commit-before-shoot.** TexelFM 페이지는 렌더 전에 세계에 커밋한다. 렌더 경로에는 생성이 0개다. Proof-of-Shot은 렌더러 결정론에만 의존한다.
4. **스케일 법칙(8M/30M/60M)으로 "작은 파운데이션"의 크기를 데이터로 정한다.**

### F6. Turing 소프트웨어 절벽과 커스텀 CUDA 범위 폭발

**부검 시나리오.** 1주차에 Z-Image가 fp16에서 검은 이미지를 낸다. torch.compile은 아무것도 빨라지지 않는다. 6주차에 TSQ backward, BC7, OptiX, Defocus Pyramid, LH-split이 동시에 반쯤 완성된 상태로 렌더러가 회사를 삼킨다(v1 프리모템 F6의 재발).

**기술적 원인.**
- bf16 네이티브 DiT의 fp16 overflow.
- Triton ≥3.3이 cc<8.0에서 FMA로 폴백한다. 그래서 FlexAttention과 max-autotune이 무력하다.
- FA2/3/4가 없다.
- T4는 PyTorch, Warp, OptiX의 지원 **하한선**에 있다. 다음 CUDA 메이저에서 잘릴 위험이 있다.

**업그레이드.**
1. **[조합] Residual-stream rescaling.**
   - pre-norm DiT에서 $h_{i+1}=h_i+G_i(\mathrm{Norm}(h_i))$이고 Norm은 스케일 불변이다. 그래서 임베딩과 모든 residual write(adaLN gate 포함)에 $1/s$를 곱해도 최종 출력은 같다.
   - 학습 없이 fp16 범위를 확보한다.
   - 남는 overflow 레이어만 fp32로 돌리고, softmax와 norm은 fp32로 둔다.
2. **경로 규칙.** cuBLAS, CUTLASS, TensorRT, CUDA C++(gsplat, Warp)만 쓴다. Triton과 FlexAttention에 기대지 않는다. attention은 SDPA mem-efficient 또는 flash-attention-turing(head_dim 64/96/128)을 쓴다. 모든 자체 모델은 head_dim 64로 설계한다.
3. **툴체인 pin.** CUDA 13.x, R570 이상 드라이버, PyTorch(CUDA 13 빌드), OptiX 9.0으로 고정한다. 렌더러 버전은 Merkle에 함께 기록한다.
4. **렌더러 단계화.**
   - 1–2주: gsplat 2DGS(텍스처 없음)로 Oracle과 D14 왕복을 만든다.
   - 3–6주: TSQ 텍스처. Textured Gaussians나 BBSplat 공개 코드에서 출발한다.
   - 7주 이후: OptiX.
   - 정답 레퍼런스는 Blender Cycles(**디노이저 끔**)다.
5. **학습 레시피 기본값.** GradScaler, fp32 softmax/LN, QK-norm, z-loss, muP(40M 프록시에서 전이하는 관행).

### F7. 운영: Kaggle 반토막, 저장량 10배, 사람 평가 병목

**기술적 원인.**
- T4×2가 쿼터를 2배로 소모하면 Kaggle 90일 가용량은 390 T4-h로, 제안서 Kaggle 계획 725를 초과한다.
- 디스크 20 GB([2차])에는 Ledger(100 GB 이상)와 데이터(약 450 GB)를 둘 수 없다. 세션은 12h다.
- 8K AOV 저장량이 10배 과소 추정되었다(A21).
- 사람 판정 수요가 크다. CRM 5.5k, 미니 벤치 3,000쌍 × 3명, X-T1과 X2. GPU보다 평가 대역폭이 먼저 바닥난다.

**업그레이드.**
1. **상시 2×T4 확보를 기본으로 한다.** Kaggle은 버스트와 CI용 보조 수단으로 둔다.
2. **Kaggle에서는 stateless job으로 운용한다.**
   - job 하나는 10h 이하다.
   - 체크포인트는 레벨-스텝 단위로 남긴다. Jacobi라서 스텝 상태가 latent 수십 MB에 불과하다.
   - Ledger는 hot 10 GB 이하로 유지한다.
   - 교사 가중치는 Kaggle Model 입력으로 읽기 전용 마운트한다.
   - 산출물은 외부 스토리지로 보낸다.
3. **저장 정책.** 8K beauty는 모든 프레임을 저장한다. AOV는 4K 프록시나 24fps 키프레임만 남긴다.
4. **평가 효율.** Bradley–Terry 불확실성 기반 능동 샘플링으로 판정 수를 30–40% 절감한다(추정). X-T1의 판정쌍은 그대로 AesHead와 CRM의 학습 데이터로 재활용한다.

---

## C. 제약 위반 점검: 사전학습 모델이 몰래 들어온 곳

| # | 위치 | 들어온 것 | 판정 | from-scratch 대체 / 규칙 |
|---|---|---|---|---|
| C1 | v1 효도 손실 $\lambda\,\mathrm{LPIPS}$ (§2.5, STRATA-lite가 상속) | VGG/AlexNet | **위반** (시스템 손실) | Laplacian 피라미드 L1 + 교사 VAE 인코더 특징 거리 |
| C2 | v1 $\mathcal L_{ocr}$ | 사전학습 OCR | **위반** | **GlyphSpotter 5M**(폰트 합성으로 from scratch)이 pseudo-text 확률맵을 내고 SYMBOL 밖에서 벌점을 준다. SYMBOL은 MSDF라 OCR 없이도 정확하다 |
| C3 | v1 `sem`(SigLIP/DINO), Droste 판정용 CLIP 신호 | 사전학습 비전 인코더 | **위반**. 주의: SDXL과 Z-Image에는 CLIP **이미지** 인코더가 없다(텍스트 타워만 있다) | `sem` = class id + 교사 TE 임베딩. Droste는 Teacher-DCA로 판정 |
| C4 | 합성 데이터용 Blender Cycles | OptiX AI 디노이저, OIDN = 사전학습 신경망 | **위반 소지** (데이터 경로) | 디노이저를 끈다. GT AOV는 저 spp로도 수렴하고, RGB는 EEVEE(래스터) 또는 고 spp를 쓴다 |
| C5 | Z-Image Fun-ControlNet-Union (리서치 언급) | 별도 사전학습 어댑터. depth 라벨 자체가 사전학습 깊이 모델에서 나왔다 | **위반** (두 번째 사전학습 모델) | 자체 GB-Adapter |
| C6 | SDXL-Lightning/DMD2/Hyper LoRA, sdxl-vae-fp16-fix, TAESD | 제3자가 학습한 파생물 | **금지 권고** (교사의 "변형"이지만 남이 학습했다) | 증류 불필요(F1), 자체 VAE 수리(F3), 자체 TL |
| C7 | COLMAP 4.0 ALIKED/LightGlue | 학습 특징·매처 | **위반** | SIFT 전용 빌드 플래그를 고정하고 CI에서 검사 |
| C8 | 교사 TE(Qwen3-4B)로 scene graph 생성 | 사실상 대형 LLM | **위반** | **규칙 R-TE**: TE는 encoder-only(임베딩)로만 쓰고 autoregressive 디코딩은 금지. 레이아웃은 MassingFM, 데모는 WorldScript DSL |
| C9 | GeCo(VGGT, UFM), MEt3R(DUSt3R+DINO), ArcFace, OCR(TPR), ColorVideoVDP | 평가 모델 | 평가로는 허용. **단 자동 최적화 대상이 되면 위반**이다(CMA-ES로 GeCo를 최적화하면 VGGT를 증류하는 셈) | **규칙 R-EVAL 방화벽**: 별도 프로세스로 돌리고 결과는 사람이 보는 리포트로만 쓴다. 손실, 보상, 선택, early-stop, 자동 하이퍼파라미터 탐색에는 넣지 않는다. 사람이 하는 모델 선택은 허용 |
| C10 | CamFM-Intent 데이터 DataDoP | 궤적은 MonST3R 출력, 캡션은 GPT-4o 출력 | **회색** (사전학습 모델 출력을 데이터로 세탁) | **규칙 R-DATA**: 학습 데이터는 사람 라벨, 고전 알고리즘 출력, 교사 출력, 절차적 생성, 실측만 쓴다. DataDoP는 창업자 결정 전까지 평가와 ablation 전용. 대체는 Blender 오픈무비 GT, 자체 COLMAP(SIFT) 추출, 절차적 Toric 궤적 |
| C11 | VCam의 ARKit | 내부에 학습 컴포넌트가 있을 수 있다 | **회색** (사용자 입력 장치) | 생성 경로 밖에서만 허용. 대안은 폰 IMU + ArUco 마커 PnP(고전) |
| C12 | v1 상속: IC-Light(X6 대체안), Difix3D+(X3 arm iv), TRELLIS, SAM 3D Body와 SMPL-X(P2) | 사전학습 모델과 학습된 통계 모델 | **위반** | 폐기. SMPL-X는 비상업 라이선스이기도 하다. P2 렌즈에 전달 |
| C13 | AesHead의 AVA | 사람 라벨(제약은 통과) | 제약은 OK, **상업 라이선스는 ✗** | 연구 경로에서만 쓴다. 상업 경로는 자체 쌍대 + DCA |
| C14 | Nunchaku 사전 양자화 가중치 | 교사의 양자화본 | OK (교사의 한 형태) | 재현을 위해 자체 양자화를 권장 |
| C15 | GeoHead 실사 MVS의 스케일 | COLMAP은 스케일 미정 | OK | ArUco와 줄자로 미터 스케일을 잡는다 |

---

## D. 2×T4로 30일 안에 돌릴 결정적 실험 5개

| ID | 가설 | 셋업 | T4-h | 성공 기준 | 실패하면 (다음 수) |
|---|---|---|---|---|---|
| **X-T0** Turing Reality Check (D1–5) | (a) Z-Image 또는 klein이 fp16(+residual rescaling)에서 NaN 없이 1024² forward 4 s 이하. (b) INT4가 1.6배 이상 빠르고 품질 손실이 작다. (c) gsplat 2DGS 4K 프레임 0.6 s 이하(3M 프리미티브). (d) 지속 HGEMM 25 TFLOPS 이상. (e) NVENC 8K HEVC 10-bit. (f) OptiX 삼각형 closest-hit 3×10⁸ ray/s 이상 | 프롬프트 200개 × 교사 3종 × {fp16, INT8, INT4} × {512, 768, 1024}. NaN과 검은 이미지 비율, 속도, `nvidia-smi` throttle reason 30분 로그. 자체 촬영 장면 3개로 2DGS. OptiX 샘플 | 25 | (a)–(f) 전부. 단 (b) 실패는 허용(Mode B로 감) | rescaling 후에도 NaN이면 블록별 fp32 폴백(+30%) 또는 TRT INT8. INT4가 실패하면 Mode B/C. 래스터가 느리면 타일 컬링, half-res + 업샘플, AOV 축소. OptiX가 실패하면 평면 재귀 + SDF만 쓴다 |
| **X-T1** Teacher Ceiling (D3–18) | hero search(N=64, DCA 선별) 교사 스틸이 같은 프롬프트의 Seedance 2.5와 Wan 3.0 프레임을 단일 프레임 블라인드에서 C1 ≥ 60%, C2 ≥ 50%로 이긴다. 광학 스택을 얹으면 +5%p 이상 | 프롬프트 60개(C1 30, C2 30). 교사 3종 best-of-64(SDXL은 16, 768²). 경쟁사 2개 × seed 2 × 5초에서 클립당 무작위 프레임 3장. 판정자 일반 20명 + 전문가 5명, 약 1,500쌍. 부가 arm: 교사 스틸 + 우리 렌즈·필름 스택 | 10–12 (+API $60–480, 평가 $0.5–1k) | 최선 교사가 C1 ≥ 60%이고 광학 가산점 ≥ +5%p | 교사가 50–60%이면 자체 시네마틱 LoRA + N=256 + Corman 조명 후 재측정. 전부 50% 미만이면 Qwen-Image-2512 INT4로 승급하고 창업자 보고. **SP 게이트를 세계 기계로 메우려는 시도는 금지** |
| **X-T2** Band-Iso on T4 (D8–28) | 실촬영 거친 레벨(l≤10–11) 위에서 l=11–14를 다시 생성할 때, OTBI + TABP + WNF가 평균 융합보다 선명하다: HER ≥ 0.8(평균은 ≤ 0.5, T5 예측), 오빗 GeCo ≤ 0.03, seam 2AFC ≤ 60%, OIE ≤ 0.05, 그리고 고유 텍셀 10⁶개당 교사 5 s 이하(Z-INT4 기준) | 자체 촬영 장면 5개(방, 가게 앞, 책장, 벽돌 벽, 부엌). 거친 레벨은 COLMAP + 2DGS/TSQ를 약 4 cm로 저역통과해서 만든다. arm은 (i) 뷰별 SDEdit + 평균 BP, (ii) OTBI + TABP, (iii) ii + WNF, (iv) iii에서 commit 디코딩을 TL과 VAE로 비교, (v) iii + Router(X-T3의 TexelFM v0 사용). 대조 교사로 SDXL도 돌린다 | 50 | arm 하나 이상이 기준 전부 충족. (iv)에서 VAE commit이 TL commit보다 HER +0.1 이상 | seam이 보이면 apron 25%, multiband $\lambda_b$, 텍스처 마스킹이 큰 곳으로 seam 배치. Turbo 결정 폭이 1.5옥타브를 넘으면 world 업데이트를 $\Pi_{b_l}$로 엄격 제한하고 레벨 간격을 2옥타브로. GeCo가 실패하면 미세 레벨은 외관만 쓰고 기하는 l≤11 + 노멀맵으로 고정 |
| **X-T3** TexelFM 연방 분할 (D5–28) | patchify-4 TexelFM 30M(from scratch, CC0 PBR)이 재질 셀 l≥13에서 (a) 실사 매크로·교사 페이지 대비 2AFC ≤ 60%, (b) 텍셀당 교사 경로보다 5배 이상 저렴, (c) 재실행 시 비트 동일. 8M→30M에서 패치 FID와 SFR 오차가 단조 감소 | CC0 피라미드(ambientCG, Poly Haven, MatSynth CC0, 실측 크기). 자체 매크로 2천 장(교차편광, 포커스 스태킹). 8M·30M 학습. 홀드아웃 재질 300쌍 × 판정자 10명. 클래스별 Δ를 측정해 Router 임계값을 정한다 | 36 (8M 6, 30M 26, 평가 4) | (a)(b)(c) 전부 | 재질 클래스별 head, apron 64, 교사 커밋 페이지로 증류 데이터 추가(expert iteration), 60M으로 확장. 그래도 실패하면 Router 몫을 교사 쪽으로 옮긴다(비용은 오르고 품질은 지킨다) |
| **X-T4** Oracle Shot (D1–30) | (a) 실촬영 방 → TSQ 세계 ≤ 1.5 T4-h, 왕복 비트 동일. (b) 4K 프레임 ≤ 0.6 s. (c) Defocus Pyramid + LH-split이 Cycles MC(디노이저 끔) 대비 ΔJOD < 0.25(평가 전용). (d) X2 파일럿: 같은 첫 프레임 경쟁 I2V 대비 VP ≥ 60%. (e) Shannon 수요 회계에서 Corman 설정(밤, 안개, f/1.4)이 수요 텍셀을 3배 이상 줄인다 | 자체 촬영 10장면(400–800장, HDR 브래킷, ArUco). COLMAP 4.x(SIFT) + GLOMAP + PatchMatch → TSQ(또는 2DGS 폴백). 카메라 스택 v0. Cycles 레퍼런스 10샷 × 12프레임. 경쟁 I2V 10 × 의도 2 × 2모델 × seed 2. 렌즈·대기 설정 4종으로 수요 집합만 계산(교사 0회) | 30 (+API $40–300, 평가 $0.3–0.8k) | (a)–(e) | 피팅이 느리면 FastGS식 densify 상한(0.4M 기본). ΔJOD가 실패하면 가림 경계 위험 타일(10% 이하)만 MC(+30%). VP가 60% 미만이면 v1 X2 ablation 그리드(블러, 그레인, 리그, 톤) |

**합계 약 153 T4-h.** 상시 2×T4의 30일 가용량(75% 가동 기준) 약 1,080 T4-h의 14%다. 같은 기간에 TL(20)과 GeoHead 1단계(20)도 병행 학습한다. Kaggle 모드(30일 130–260 T4-h)에서는 X-T2를 2장면으로 줄이고, 일정을 2주 늘린다. 목표는 그대로다.

**의사결정 트리 (Day 18 / Day 30).**
- X-T0과 X-T1이 둘 다 통과하면 → Mode A(Z-INT4 대칭)로 잠근다.
- X-T1은 통과했는데 INT4가 실패하면 → Mode C(σ-분할), INT8도 실패하면 Mode B.
- klein이 Z보다 3%p 이내로 뒤지면 → klein을 고른다. 1.6배 빠르고 hero 레퍼런스 앵커를 쓸 수 있다.
- 모든 교사가 50% 미만이면 → Qwen-2512 INT4로 승급하고 창업자에게 보고한다.

---

## E. 개정 제안서 전체본: STRATA-lite v0.2

### 세계는 한 번만, 그 한 번도 카메라가 기록할 수 있는 만큼만. 그리고 교사는 최고로.

### E0. TL;DR

1. **예산.** 2×T4 90일 상시 4,320 T4-h, 75% 가동 시 약 3,200 T4-h. 계획 실효 13 TFLOPS/장, 2×10²⁰ FLOPs. v1 Phase 1 누계의 1/3,000–1/6,000이다.
2. **화폐는 교사-초(teacher-second)다.** 세계 비용의 60–75%가 교사 호출이다.
3. **[수정] 교사는 미감 천장을 먼저 재고 고른다.** 기본은 Z-Image-Turbo(6.15B, Apache, Elo 941)를 INT4로 두 장에 대칭 배치하는 것이다. 예비는 klein 4B, 승급은 Qwen-Image-2512다. SDXL은 개발 대역이다. Day 18 Teacher Ceiling Test가 게이트다.
4. **STRATA-lite.** 정본은 explicit TSQ + BC7/BC5(HDR 발광 페이지는 BC6H) 페이지다. 4K 방이 0.5–1.0 GB, 8K가 1.2–2.8 GB라서 T4 한 장에 통째로 올라간다.
5. **[신규] OTBI + TABP.** 미세 레벨은 단독 소유 타일링으로 생성하고 닫힌 형태로 역투영한다. Adam이 0이고, 뷰 수는 절반에서 4분의 1로 준다.
6. **[조합] Federal Router.** l≥13의 재질 셀은 patchify TexelFM이 맡는다. 교사보다 16배 싸고 비트 결정론적이다. 의미 셀은 교사가 맡는다.
7. **[신규] MassingFM 21M.** 절차적 데이터로 학습한 레이아웃 파운데이션이 bedrock의 3D prior가 된다. hero 밖 360°를 덩어리로 먼저 확정한다.
8. **렌더(2×T4 벽시계, 10초 샷).** 1080p24 10–22 s, 4K24 0.7–1.5분, 4K120 3.4–7.6분, 8K120 13–30분. fps는 생성 비용을 늘리지 않는다.
9. **카메라 스택은 학습 0개다.** 렌즈 처방, Hullin 플레어, 공분산 DOF·MB, Defocus Pyramid, LH-split, 센서·필름 수식.
10. **세계 1개(4K 방, Directed).** Mode A 1.0–1.6 T4-h(벽시계 30–48분), Mode B 3–6 T4-h. 벤치(Location-lite)는 1.5–2.5 T4-h.
11. **Corman–Shannon Demand.** 카메라가 기록하지 못하는 정보는 생성하지 않는다. DP는 T4-시간을 비용으로 지불한다.
12. **소형 모델 10종, 약 150M.** 학습 약 220 T4-h, 데이터 준비 약 200 T4-h다.
13. **최종 승리 조건은 v1 그대로다.**

### E1. 2×T4 회계

**E1.1 계획 상수**

| 항목 | 값 |
|---|---|
| 피크 | FP16 tensor 65 TFLOPS, INT8 130 TOPS, INT4 260 TOPS, FP32 8.1 TFLOPS, 320 GB/s, 16 GB, 70 W, RT 코어 40개, NVENC 1·NVDEC 2 |
| 실측 | HGEMM 41.6 TFLOPS, INT8 74.9 TOPS, INT4 114 TOPS(Citadel, 1590 MHz 고정). 지속 부하에서는 전력 쓰로틀이 일찍 걸린다 |
| 계획 실효 | 소형 모델 학습 13 TFLOPS/장. 교사 추론은 forward 실측값을 쓴다 |
| 일 / 90일 | 2.2×10¹⁸ / 2×10²⁰ FLOPs, 4,320 T4-h(75% 가동 3,200) |
| 환산 | 1 H100-h ≈ 20–31 T4-h |
| Kaggle | 390–780 T4-h / 90일(쿼터 소모율 미확인), 세션 12h |

**E1.2 교사 단가표 (T4 1장, 실측 필요)**

| | SDXL | FLUX.2 klein 4B | Z-Image-Turbo |
|---|---|---|---|
| forward 1024² fp16 | 0.55–0.8 s | 2.0–2.7 s | 3.1–4.1 s |
| forward 512² fp16 | 0.16 s (품질 저하) | 0.5–0.7 s | 0.75–1.1 s |
| INT4 W4A4 | — | 지원 확인 필요 | 1024²에서 1.4–2.2 s (계획 1.8) |
| 기본 NFE | 20–30 × CFG2 | 4 | 8, CFG 없음 |
| 가중치 fp16 / INT4 | 5.1 / — GB | 약 8 / 약 2.5 GB | 12.3 / 약 3.5 GB |
| TE | CLIP-L+bigG 1.6 GB | Qwen3-4B 약 8 GB | Qwen3-4B 약 8 GB |
| VAE | f8 4ch | f8 32ch [2차] | f8 16ch (Flux 계열) |
| 고유 텍셀 10⁶개당 교사-초 (미세 레벨) | 2.5 s (K=3) | 6 s (K=2) | fp16 9.4 s / INT4 4.7 s (K=2) |

**자체 연산 단가.**
- TL 인코드+디코드 1024²: 30–50 ms
- 원 VAE 디코드 1024²: fp16 수리 후 0.3–0.5 s
- TSQ 래스터 1024²: forward 10–20 ms
- TABP: 뷰당 5–10 ms
- TexelFM: 10⁸ 텍셀에 20–40 s
- 4K 시네마 프레임: 0.28–0.63 s

**E1.3 부족분을 무엇으로 메우는가**

수단은 넷이다. (a) 빌린 교사의 사전 컴퓨트, (b) 학습 없는 구조, (c) 테스트타임 최적화, (d) 우리 소형 모델.

| v1 구성요소 | v0.2 대체 | 수단 |
|---|---|---|
| LLM World Program | 텍스트 모드: hero search + MassingFM(TE 임베딩, encoder 전용). 데모: WorldScript DSL | (a)(d) |
| CWM 4–8B bedrock | MassingFM + hero lift(GeoHead) + coarse ring lockstep | (a)(c)(d) |
| LBO 학생 | TexelFM(재질 셀) + 세계별 WSSC-lite(OTBI+TABP) | (c)(d) |
| 3D object prior | MassingFM slot + 절차적·CC0 에셋 | (b)(d) |
| SelfCaption VLM | 구조화 캡션(prompt_tree@scale, class, 스케일, lensTags) | (b) |
| RAFT/UFM 보정 광류 | DIS, TV-L1 | (b) |
| DA3/VGGT/AnySplat lift | COLMAP 4(SIFT) + GLOMAP + PatchMatch + TSQ 피팅, GeoHead | (a)(b)(c) |
| 3DGRUT | TSQ 래스터 + OptiX(삼각형 코어) + 평면 재귀 + DIEL SDF | (b) |
| 렌즈·센서·필름 | 그대로(수식) | (b) |
| DP-0, CRM, IMU 손 | CamFM ×3 + 1 kHz 리그 시뮬 | (b)(d) |
| 닫힌 art director | hero search + AesHead + Teacher-DCA | (a)(d) |
| $\mathcal L_{ocr}$, LPIPS 효도, SigLIP `sem` | GlyphSpotter, Laplacian + 교사 특징 거리, class id + TE | (a)(b)(d) |

**운영 규칙 3개(§C).** R-TE(TE는 encoder 전용), R-EVAL(평가 방화벽), R-DATA(데이터 출처 제한).

### E2. 교사: 계약 v0.2와 운용 모드

**계약.**
1. 빌린 이미지 모델은 1개다. 자체 경량 어댑터(GB-Adapter, rank 32 이하 LoRA, VAE fp16 수리, 양자화)만 붙인다. 제3자 어댑터, LoRA, 증류물은 금지한다.
2. fp16이 안전해야 한다(residual rescaling 허용). 아니면 INT8/INT4 경로가 있어야 한다.
3. f8 공간 정렬 latent여야 한다.
4. σ에 접근할 수 있어야 한다. Turbo 계열은 학습된 σ 점으로 SDEdit 창을 스냅한다.
5. 상업 라이선스여야 한다.
6. 속도: Mode A에서 실효 2 s/1024² 이하, Mode B에서 4 s 이하.
7. **천장: X-T1 C1 ≥ 60%.**

**선택 지표.** 천장 제약을 만족하는 후보 중에서 수용 세계당 교사-초를 최소화한다.

**운용 모드.**

| 모드 | 조건 | 배치 | 교사 처리량(1024² forward/h) |
|---|---|---|---|
| **A (기본)** | INT4 통과 | 두 장 모두 교사 INT4 + 세계 shard(Morton 짝/홀). 장 사이에는 apron과 Ledger 해시만 오간다 | 약 4,000 |
| **C [신규] σ-정밀도 카드 분할** | INT4는 탐색용으로만 쓸 만함 | GPU0 fp16: commit σ 점, hero 최종, GeoHead 특징. GPU1 INT4: 탐색 스텝, adversarial, Gaffer + 세계·렌더 | 약 2,500 |
| **B** | INT4·INT8 실패 | GPU0 교사 fp16 전용. GPU1에 VAE, TL, TexelFM, 세계, 래스터, 소형 모델(파이프라인) | 약 1,000 |

**TE 처리.** 세계마다 구조화 캡션 조합(노드 약 50 × 스케일 8 × 렌즈 4, 1,600개 이하)을 한 번에 인코딩해 디스크에 캐시하고 TE를 해제한다. GPU1에서 30초 정도, CPU에서도 5–15분이다.

### E3. STRATA-lite v0.2 매개체

**E3.1 원칙 여섯**
1. 병목은 교사 시간이다. 남는 메모리로 교사 호출을 아낀다(Ledger).
2. 정본은 explicit이다. `z_slot`(int8×16)은 예약해 두고 Phase 1.5 학생 때 쓴다.
3. 하드웨어가 공짜로 주는 기능을 쓴다. BC 디코드, 바이리니어, RT 삼각형, NVENC.
4. **[수정]** 미세 레벨은 Router가 나눈다. 재질 셀은 TexelFM, 의미 셀은 교사다.
5. 모든 생성은 레벨 안에서 Jacobi(동기식)다. 그래서 두 장으로 완전히 병렬화된다. **비회귀 조항이 곧 하드웨어 효율이다.**
6. **[신규] commit-before-shoot.** 렌더 경로에서는 생성 연산이 0개다.

**E3.2 레벨 체계 [수정: 전역 고정]**

$h_l=64\cdot2^{-l}$ m를 모든 lot에 고정한다. lot이 64 m를 넘으면 l<0을 쓴다. 이렇게 해야 TexelFM의 스케일 조건과 Ledger 키가 물리적으로 일관된다. 텍셀 피치는 $\tau_l=h_l/8$이고, 4K S35 35mm 기준 $\phi(z)=1.85\times10^{-4}z$다.

| l | 셀 | 텍셀 | 4K 필요 최대 거리 | 1024² 뷰 폭 | 결정 주체 |
|---|---|---|---|---|---|
| ≤6 | ≥1 m | ≥12.5 cm | — | ≥128 m | bedrock: MassingFM + hero + coarse ring |
| 7–10 | 50–6.25 cm | 6.25 cm–7.8 mm | 340–42 m | 64–8 m | 교사(시네마틱 뷰, lockstep) |
| 11–12 | 3.1–1.6 cm | 3.9–1.95 mm | 21–10.5 m | 4–2 m | 교사(OTBI + TABP) |
| 13–14 | 7.8–3.9 mm | 0.98–0.49 mm | 5.3–2.6 m | 1–0.5 m | Router: 의미 셀 교사 / 재질 셀 TexelFM |
| 15 | 1.95 mm | 0.24 mm | 1.3 m (8K 2.6 m) | 0.25 m | TexelFM (8K 의미 셀만 교사) |
| ≥16 | ≤0.98 mm | ≤0.12 mm | ≤0.66 m | — | TexelFM |
| PSH | SYMBOL(MSDF), POINT, CURVE, GLINT, CARRIER | — | 게이팅 면제 | — | DSL, 수식 |

**E3.3 자료구조.** 제안서의 STRATA-lite v0를 유지하고 다음만 바꾼다.
- `pages`: 기본은 BC7(L_d log-delta) + BC5(법선·높이)로 128 B/8×8. HDR 발광 페이지(네온, 광원 주변)는 **BC6H**(signed half, 1 B/텍셀)를 쓴다.
- `meta`: h_0 = 64 m로 고정하고 lot extent를 따로 기록한다. `router_map`(셀별 owner 종류)과 `massing`(shell, slots)을 추가한다.
- `ledger` 엔트리 latent는 int8로 양자화한다(Z-Image 기준 256 KB).
- `wnf`는 저장하지 않는다(0 B). Philox(seed, l, cell, k)를 즉석 계산한다.

**TSQ (Textured Soft Quad)** [조합: 2DGS, BBSplat, Textured Gaussians]
- 32 B SoA. 반폭 $(s_u,s_v)$, 가장자리 $\exp(-(|u|^n+|v|^n))$, 8×8 페이지.
- RT에서는 **불투명 코어 삼각형 2개만** closest-hit로 쏜다. soft edge는 반사 안에서 근사로 처리한다.
- 예외는 v1 C6 그대로다. VOL(SGGX), DIEL(SDF), 아바타(P2c).

**E3.4 세계별 메모리 [수정]**

| 세계 | 레벨 | TSQ | 텍셀 | RAM/디스크 | GPU |
|---|---|---|---|---|---|
| Oracle 방 (실촬영, 4K) | ≤14 | 2–5M | 1.5–3×10⁸ | 0.5–1.0 GB | 상주 |
| 을지로 네온 (4K, 밤·안개·f/1.4) | ≤15(의미 셀) | 2–4M + POINT 5k + SYMBOL 300런 | 0.8–2×10⁸ | 0.3–0.7 GB | 상주 |
| 거울 복도 | ≤15 | 3–6M | 2–4×10⁸ | 0.6–1.2 GB | 상주 |
| 불릿타임 바 + 와인잔 | ≤16 + DIEL | 2–4M + SDF 8×10⁷ voxel | 1–2×10⁸ | 0.5–0.9 GB | 상주 |
| 같은 방 8K | ≤15 | 6–12M | 4–10×10⁸ | 1.2–2.8 GB | 상주 |
| 10분 원테이크 집 | ≤15 | 25–50M | 1–2×10⁹ | 3–6 GB | 경로 주변 1.5–3 GB 스트리밍 |
| Powers of Ten 8K | −6…22 | 수요 원뿔 5–15M | 2–5×10⁸ | 1–2 GB | 옥타브 2–3개 스트리밍 |

### E4. 생성 경로 v0.2 (WSSC-lite)

```text
procedure WSSC_lite_v02(y, mode∈{directed, location-lite}):
  E  ← TE.encode(structured_captions(y))                 # R-TE: encoder 전용, 1회, 캐시
  H  ← HeroSearch(y; N=64 @512² NFE 3 → top16 NFE 8 → top4 @1024² SDEdit 4)
        score = AesHead ⊕ DCA(top16) − λ·C_T4(Corman) − GlyphSpotter(pseudo-text)
        (directed: 사람 best-of-4)
  ĝ_H ← GeoHead(H) ; s ← SemanticScale(ĝ_H, slots)       # 미터 + FOV
  M  ← MassingFM(E, lift(ĝ_H, s))                         # 전역 1-pass (bedrock 예외)
  W  ← Bedrock(M, H): coarse TSQ ← mesh(M);
        ring V_c(16) lockstep K=4, ctrl = GB-Adapter(Gbuf(M));
        owner(layout band) = H (frustum 안) | M (밖)       # Janus 방지
  D  ← RayBundle ∪ CormanShannon(CamFM.prior(W), lens plan)
  for l = l_g+1 … L_T:                                     # 스케일 순서
     (B_T, B_X) ← Router(D[l])
     V ← OTBI(B_T, l) ∪ HeroCams ∪ AdvCams(l)
     ∀v: z_v ← SDEdit init at σ-point s(l), noise ε_v = WNF(seed, l, ·, k=0)
     for k = 1..K_l (K=2 미세, 3 중간, 4 coarse):           # lockstep (Jacobi)
        ∀v: (x̂0_v, ĝ_v) ← Teacher(z_v, σ_k, c_v, GB(v))   # GeoHead는 같은 forward의 특징을 읽는다
        ∀v: x_v ← (k<K_l) ? TL.dec(x̂0_v) : VAE.dec(x̂0_v)  # F3 규칙
        W ← TABP(W, {x_v, ĝ_v}, m̃, gauge)                  # 닫힌 형태, Adam 없음
        ∀v: z_v ← step(z_v, γ_k·Enc(R_v(W)) + (1−γ_k)·x̂0_v, WNF(k))
     HardCommit(l, B_T): graph cut + multiband; seed 동결; Merkle
     W[l, B_X] ← TexelFM(parent = W[l−1], WNF(l), mat, h_l)  # 같은 레벨 형제, 순서 없음
  W[>L_T] ← TexelFM (8K: 의미 셀만 교사)
  Audit/Polish: AdvCams(64 @512²) → 열린 대역만 2라운드
  Shoot: CamFM-Intent + CRM + rig sim + Gaffer + Corman → commit → Render
```

**Judgment Ledger [조합].**
- key = blake3(블록 Merkle ‖ 뷰 정의 ‖ σ 점 ‖ WNF seed ‖ teacher id ‖ 양자화 모드 ‖ caption hash).
- value = int8 x̂0 latent + gauge 통계.
- 재사용: 세션 재개, Merkle 불변 블록 편집, 거친 레벨을 공유하는 A/B, 벤치 의도 공유. 카메라만 바꾼 재촬영은 교사 호출 자체가 0이다.
- 예상 적중률 40–70%(추정).

**Proof-of-Shot [신규].**
- 영상과 함께 Merkle root, ShotScript, 렌더러·툴체인 pin을 공개한다.
- forward 커널은 정수 atomic만 쓰고 (깊이, atom id)로 stable sort한다.
- 같은 아키텍처·툴체인이면 바이트 동일, 다른 아키텍처면 1 LSB 이내(X-T4에서 교차 검증).

**비회귀 점검**

| 부품 | 조건으로 쓰는 것 | 판정 |
|---|---|---|
| MassingFM | y, hero lift | 전역 1-pass. v1의 유일한 예외 범위 |
| OTBI + TABP | 커밋된 부모 레벨, WNF, 같은 스텝의 동시 x̂0 | Jacobi. 이웃 타일 출력을 조건으로 쓰지 않는다 |
| Router | — | 연산자만 고르고 순서축은 바꾸지 않는다 |
| TexelFM | 부모 페이지, WNF, 재질 | 비트 결정론 native LBO (T8) |
| klein hero 레퍼런스(선택 시) | hero ∈ y | 렌더 픽셀이 아니므로 허용 |
| 수요 확장 | 더 거친 커밋 레벨 + WNF | 세계 공간 생성 |

### E5. 렌더러 on Turing

**E5.1 선택**

| 후보 | 결정 |
|---|---|
| gsplat(2DGS/3DGS, 3DGUT) | primary 베이스. 1–2주는 2DGS 그대로, 3–6주에 TSQ 텍스처 커널로 확장 |
| 3DGRT/3DGRUT | primary는 폐기. 3DGS보다 느리고 1세대 RT 코어의 any-hit 비용이 크다 |
| OptiX(TSQ 코어 삼각형) | secondary. 거울·크롬·유리 픽셀만, 깊이 2 이하. 7주 이후 |
| 평면 거울 재귀 래스터 | 거울 복도 주력. 깊이 12, 총비용은 프레임의 2배 이하 |
| DIEL SDF 레이마칭 | 와인잔, 분산(Abbe) 포함 |
| Blender Cycles(디노이저 끔) | 정답 레퍼런스(ΔJOD), 합성 데이터 일부 |
| 메쉬 + BC export → Vulkan/WebGL | 웹 뷰어와 엔진 export |

**E5.2 핵심 장치.**
- **Dual-cut LOD**: v1 C7, 16×16 타일 단위, +20–40%.
- **공분산 대수**: v1 식 그대로.
- **Defocus Pyramid** [조합: Kraus–Strengert'07]: CoC가 8 px를 넘는 atom은 $2^{-k}$ 해상도 레이어에서 렌더한다.
- **Longuet-Higgins Split** [조합]: 회전 블러는 셔터 중간 자세 오버스캔 렌더 1회 + 호모그래피 $H(\tau)=KR(\tau)R_{mid}^\top K^{-1}$ 적분으로 정확히 처리한다. 평행이동 블러만 공분산으로 넣는다. 롤링셔터는 행별 $\tau(y)$다. 그래서 1 kHz 손떨림 블러가 거의 공짜로 정확해진다.
- **초록 무한 터널**: 반사마다 $R(\lambda)\approx0.9\,T_{glass}(\lambda)^2$를 곱한다.

**E5.3 렌즈·센서·필름 스택(학습 0개).** Kolb 출사동 LUT, 디포커스, 조리개 스프라이트 보케, Hullin 고스트, FFT 베일링·회절 스타, 비네팅, LH-split 셔터, Poisson-Gauss 센서 + Malvar 디모자이크, 3D LUT, 할레이션, Newson 그레인 타일 뱅크(유일한 프레임별 랜덤), 게이트 위브. 4K에서 60–120 ms, 8K에서 0.25–0.5 s.

**E5.4 렌더 시간 [수정]**

| 프레임당(T4 1장) | 1080p | 4K | 8K |
|---|---|---|---|
| primary(dual-cut) | 50–100 ms | 0.2–0.45 s | 0.8–1.8 s |
| secondary(삼각형, 20% 픽셀, 깊이 2) | 5–15 ms | 15–60 ms | 60–240 ms |
| 렌즈·센서·필름 | 15–30 ms | 60–120 ms | 0.25–0.5 s |
| **합계** | **0.07–0.15 s** | **0.28–0.63 s** | **1.1–2.5 s** |

| 10초 샷 벽시계(짝·홀 분담, I/O +20%) | 24 fps | 60 fps | 120 fps |
|---|---|---|---|
| 1080p | 10–22 s | 25–54 s | 50–108 s |
| 4K | 0.7–1.5분 | 1.7–3.8분 | 3.4–7.6분 |
| 8K | 2.6–6분 | 6.6–15분 | 13–30분 |

**에너지.** 4K120 10초가 GPU만 8–18 Wh, 호스트 포함 16–37 Wh다. 스마트폰 1–2회 충전 분량이다.

**저장 [수정].**
- 8K beauty EXR(DWAA): 40–80 MB/프레임.
- AOV는 4K 프록시나 24fps 키프레임만 무손실로 저장한다. 8K 전 프레임 AOV는 10초에 600–840 GB라서 금지한다.
- 리뷰 사본은 NVENC HEVC 10-bit 8K다(실측 필요).

**라이브 프리뷰.** 1080p 절반 품질(RT·필름 끔)로 장당 25–45 fps가 나온다. VCam 입력은 폰 IMU + ArUco PnP가 기본이고 ARKit는 선택이다.

### E6. 카메라

**리그(학습 0개).** v1 리그 14종을 1 kHz ODE로 시뮬한다. 의도 층은 min-jerk, 2/3 power law, 지연 피드백이다.

**CamFM 3종**: §E7 표.

**DP 탐색.**
- 후보 1,000개를 CRM과 리그 시뮬만으로 먼저 채점한다(수 초).
- 상위 50개를 360p 48프레임 프리뷰로 렌더한다(각 0.3 s 이하).
- Gaffer는 조명 8개 이하를 CEM 200회로 찾는다. 평가 1회는 렌더 + AesHead @512²다. 약 2분 걸린다.

**Corman–Shannon Demand v0.2 [수정].** 점 주파수 평가는 jinc의 0점($c\nu\approx1.22$)에서 수요를 잘못 생략한다. 그래서 대역 평균으로 바꾼다.

$$\mathcal I_l(x)=\max_t\frac1{|B_l|}\int_{B_l}\log_2\!\Big(1+\frac{|H(\nu;x,t)|^2S_l(x)}{N_{shot}+N_{read}+N_{grain}}\Big)d\nu,\quad B_l=[\tfrac12f_l\phi,\,f_l\phi]\cap[0,\tfrac12]$$
$$H=\mathrm{MTF}_\theta\cdot\tfrac{2J_1(\pi c\nu)}{\pi c\nu}\cdot\mathrm{sinc}(\nu v\Delta t)\cdot\mathrm{sinc}(\nu)\cdot e^{-\beta z}$$

- 수요 집합은 $\{(l,b):\max_b\mathcal I_l\ge\epsilon\}\cup$승격 구조이고, $\epsilon=0.05$–0.1 bit/sample이다.
- Corman Optimizer는 $\max\ \mathrm{CRM}+\beta_a\mathrm{Aes}-\lambda C_{T4}$다. λ는 감독 UI의 예산 슬라이더다.
- 벤치에서 의도가 렌즈나 조명을 지정하면 Corman은 그 값을 바꾸지 못한다. 결과는 on/off 둘 다 보고한다.

### E7. 소형 모델 총괄 (전부 from scratch, 교사 특징·출력은 허용 데이터)

| 모델 | 파라미터 / 아키텍처 | 입력 → 출력 | 데이터 (사전학습 모델 없이) | 학습 FLOPs | T4-h (2×T4 벽시계) | 추론 | GPU 메모리 |
|---|---|---|---|---|---|---|---|
| **TL-3M** | enc 1.2M + dec 1.8M, TAESD형 conv | $x$[B,3,H,W] ↔ $z$[B,16,H/8,W/8] (교사 latent) | 자체 촬영·렌더 + 교사 생성물 + CC0/PD 사진 20만. 타깃은 교사 VAE. 손실: L1 + 교사 VAE 특징 + from-scratch PatchGAN + 교사 x̂0 일관성 | 약 5×10¹⁷ | 18–20 (약 10 h) | 1024²에 30–50 ms | <0.1 GB |
| **GB-Adapter** | 30M: G-buffer 16 px 패치 → transformer 6층 d=512(hd 64) → zero-init 투영 5개(DiT 층 0/6/12/18/24 hidden 잔차) | $g$[B,12,H,W] (log 깊이1, 법선3, 알베도3, sem 임베딩4, 레벨 마스크1) → 5×[B,N_tok,3840] | 자체 촬영 MVS 3–4만 쌍(COLMAP SIFT + GLOMAP + PatchMatch), EEVEE/Cycles(디노이저 끔) 합성 1만 | 동결 교사 역전파가 지배(샘플당 약 3 s @512²) | Z: 50–85 / SDXL: 30–50 | 교사 시간 +5–8% | 학습: GPU0 약 14 GB(512² batch 1, ckpt) / 추론 0.06 GB |
| **GeoHead** | 25M DPT-lite, 교사 중간 토큰 4개 층 + FOV head | 4×[B,4096,3840] @1024² → log 깊이, 법선, 불확실성 @H/2 + FOV | 합성 6만(CC0 모델·HDRI, 절차적 방·거리, GT 깊이·법선·FOV) + 실사 MVS 2만(ArUco 미터). 선례: 디퓨전 특징 + 합성 데이터로 일반화(Marigold, 참고만) | head 약 10¹⁷, 특징 추출이 지배 | 50–60 (512² 2 epoch + 1024² 1만 미세조정) | WSSC forward와 공유 시 +20 ms | 0.1 GB + 교사 |
| **AesHead** | 5M MLP, 교사 3개 층 pooled | [B,3×3840] → 스칼라(BT) | 자체 쌍대 5k + 전문가 500(X-T1 재활용). AVA 5만은 연구 경로 전용 | 작음 | 12–15 (특징 추출) | 512²에 0.5 s(INT4) | 작음 |
| **Teacher-DCA** [조합] | 0개 | 교사 조건 2개 × t 2개의 denoising 오차 차 | — | 0 | 0 | 후보당 4 forward(상위 16개에만) | — |
| **TexelFM** | 30M(비교 8M/60M): space-to-depth 4 → 64²×96, conv-UNet 192/288/384 + 16² attention(hd 64), flow matching + shortcut 4 step | 부모 [B,6,320,320] + WNF ξ[B,4,320,320] + 재질 emb 64 + $h_l$ Fourier 16 + sem 마스크 → 자식 잔차 [B,6,256,256] | CC0 PBR(ambientCG, Poly Haven, MatSynth CC0) 실측 크기 옥타브 피라미드, 자체 매크로 2–5k, solver 커밋 페이지 | 1.2×10¹⁸ (+ ablation 0.9×10¹⁸) | 26 + 20 | 10⁸ 텍셀에 20–40 s (INT8 12–25 s) | 1–2 GB |
| **MassingFM** [신규] | 21M DiT 12층 d=384 hd 64, FM(연속) + masked(이산) | TE 8토큰 + hero lift 부분 점유 → shell ≤32 + slot ≤64(class, OBB, shape 16D) + 점유 32³ | 절차적 RoomGen/StreetGen 300만, Infinigen 실내 5k, 자체 MVS 부분 관측 | 약 2.3×10¹⁷ | 5–10 | 16 step 0.3 s | 0.5 GB |
| **CamFM-Intent** | 15M DiT 8층 d=384 | 키프레임 16×12D + 세계 토큰 128×16D + TE 1토큰 + 스타일 32D → velocity | Blender 오픈무비 .blend GT(CC-BY), 자체 COLMAP 추출(PD·라이선스 클립), 절차적 Toric 궤적, CameraBench 사람 라벨(라이선스 확인). DataDoP는 평가 전용 | 약 8×10¹⁶ | 2 | 궤적 1,000개에 1 s | 작음 |
| **CamFM-Hand** | 5M 1D transformer 6층 d=256 | 200 Hz gyro+accel 4초(800×6 → 100토큰) + 리그·의도 조건 → ω 잔차 | 자체 GoPro GPMF 100h 이상(카메라 2–3대, 오퍼레이터 3명, 리그 4종) | 약 10¹⁷ | 2–3 | 실시간보다 빠름 | 작음 |
| **CamFM-CRM** | 8M (궤적 enc 4M + latent CNN 3M + 세계 1M) | 궤적 + 키프레임 8장 TL latent [8,16,32,32] + 세계 토큰 → 스칼라 | 사람 쌍대 5k + 500 → 1.5–2만 | 작음 | 1 | 1 ms | 작음 |
| **GlyphSpotter** | 5M U-Net-lite | [B,3,H,W] → 텍스트 확률 [B,1,H/4,W/4] + 평면 | 폰트 합성(무한), 배경은 교사 생성물, 음성 샘플 포함 | 작음 | 2–3 | 1024²에 30 ms | 작음 |
| **(선택) 시네마틱 LoRA** | rank 32, 교사 어댑터 | — | 자체 촬영 프레임 2천 장 | — | 약 17 | 0 | GPU0 약 14 GB(학습) |
| **합계** | **약 150M (+LoRA)** | | | **약 3.5×10¹⁸** | **약 210–250** | | |

**학습 레시피 기본값.** fp16 + GradScaler, fp32 softmax/LN, QK-norm, z-loss, head_dim 64, muP 프록시, Triton·FlexAttention 금지. 수십M 모델의 학습 FLOPs는 병목이 아니다(50M을 Chinchilla 비율로 학습해도 약 1시간). 비싼 것은 교사 특징 추출과 사람 판정이다.

### E8. 16 GB 두 장 배치

**Mode A (장당)**

| 항목 | GB |
|---|---|
| 교사 DiT W4A4 | 3.5 |
| 교사 활성(1024², batch 1–2) | 1.0–1.5 |
| VAE(fp16 수리) + 타일 디코드 | 1.0 |
| TL, GB-Adapter, GeoHead, AesHead, GlyphSpotter + 활성 | 0.6 |
| TexelFM(TRT, 타일 16개) | 1.0 |
| 세계 shard: 커밋 페이지 + TSQ | 0.3–1.0 |
| 세계 shard: 활성 타일 fp16 + TABP 누산기 | 0.5–1.0 |
| 래스터 버퍼(1024² 뷰 2장, G-buffer, 정렬) | 1.2 |
| CUDA 컨텍스트, 단편화 | 0.6 |
| **합계** | **9.7–11.9 / 가용 약 15** |

**Mode B.**
- GPU0: 교사 fp16 12.3 + 활성 1.0 + 헤드 0.3 + 컨텍스트 0.5 = **14.1 GB**(1024² batch 1이 상한).
- GPU1: VAE 1.0, TL, TexelFM 1.0, 세계 1–2.5, 래스터 1.2, CamFM 0.2, 컨텍스트 0.6 = 5–6.6 GB. TE 캐싱 중에만 +8 GB.
- PCIe 트래픽은 view-step당 latent 0.5 MB + G-buffer 25 MB로 약 2 ms다.

**학습 모드.**
- GPU0: GB-Adapter 또는 LoRA. 동결 fp16 교사 12.3 + ckpt 활성 1.2 + 어댑터·Adam 0.5 = 14.0 GB.
- GPU1: TexelFM, MassingFM, CamFM, TL, GeoHead(INT4 교사 특징)를 동시에 돌린다. 8 GB 미만.

**8K 렌더 모드(장당).** 세계 BC 1.2–2.8 + 8K 프레임과 AOV 1.5 + 정렬 리스트 1–2 + Defocus Pyramid 0.4 + FFT 1.8 + OptiX BVH 0.6–1.2 = **7–10 GB**. 짝·홀 프레임을 나눠 맡는다.

**호스트(권장).**
- CPU 16–32코어: COLMAP/GLOMAP, 1 kHz 리그, EXR, 절차적 데이터.
- RAM 128 GB: Ledger hot(int8) 13 GB, 세계 버전, EXR 큐.
- NVMe 4 TB: 데이터 약 450 GB, Ledger cold 약 200 GB, 합성 렌더, 출력.

**Kaggle.** RAM, CPU 수, 디스크, T4×2 쿼터 소모율이 모두 미확인이다.
- 교사는 Kaggle Model 입력으로 읽기 전용 마운트한다.
- job 하나는 10 h 이하이고, 레벨-스텝 단위로 체크포인트한다.
- Ledger hot은 10 GB 이하, 산출물은 외부 스토리지로 보낸다.
- 8K EXR은 포기하고 NVENC로만 출력한다.
- MVS와 절차적 데이터는 로컬 CPU에서 만든다.
- 이용약관(ToS)을 확인한다.

### E9. 세계 1개의 값 (4K 방·가게 앞, Directed, 15초 경로, deep focus 중간값)

| 단계 | 내용 | 교사 forward (1024² 환산) | Mode A T4-h | Mode B 교사 카드 h | SDXL T4-h |
|---|---|---|---|---|---|
| S1 Hero | 64 @512² NFE 3 → 16 NFE 8 → 4 @1024² + DCA | 약 92 | 0.05 | 0.10 | 0.10 |
| S2 Bedrock | MassingFM + GeoHead + ring 16 × K4 @768² | 약 36 | 0.02 | 0.04 | 0.02 |
| S3 Demand | Ray-Bundle + Corman–Shannon | 0 | ~0 | ~0 | ~0 |
| S4a 교사 레벨 l=7–14 | OTBI 뷰 100–220 × K 2–4, Router 적용 | 250–480 | 0.13–0.24 | 0.25–0.48 | 0.05–0.11 |
| S4b commit VAE | 뷰 100–220 × 0.4 s | — | 0.01–0.025 | (GPU1) | 0.01–0.03 |
| S4c TexelFM | 재질 셀 l=13–15, 1.5–4×10⁸ 텍셀 | 0 | 0.01–0.03 | (GPU1) | 0.01–0.03 |
| S6 Audit/Polish | adversarial 64 @512² + 2라운드 × 30뷰 × K2 | 약 138 | 0.07 | 0.14 | 0.03 |
| S7 Shoot planning | CamFM + CRM + Gaffer 200 | 약 56 | 0.03 | 0.06 | 0.02 |
| S8 Render | 4K24 15초 = 360프레임 | 0 | 0.03–0.06 | (GPU1) | 0.03–0.06 |
| **순수 합계** | | **약 570–800** | **0.35–0.50** | **0.6–0.85 h** | **0.27–0.40** |
| **공학 오버헤드 ×2.5–3** | | | **0.9–1.5 T4-h** (벽시계 27–45분) | **벽시계 1.5–2.6 h = 3–5.2 T4-h** | 0.7–1.2 |
| 벤치(Location-lite, 의도 공통 Π) | 커버리지 ×1.5–2 | | 1.5–2.5 | 4–8 | 1.2–2 |
| 데모 히어로(best-of-4 + polish) | | | 4–6 | 10–15 | 3–5 |
| 8K 추가 | l=15 TexelFM + 의미 셀 교사 | +60–240 | +0.15–0.45 | +0.3–0.9 | +0.1–0.3 |
| 10분 원테이크 집 | 방 5–8개 | | 6–10 | 15–25 | 5–8 |
| Powers of Ten | 옥타브당 프레임 1장 분량 + 의미 정합 재시도 | | 5–15 | 12–35 | 4–12 |

**결론.** OTBI, TABP, Router가 Elo 941 교사의 값을 치른다. v1 추정(100–350 T4-h) 대비 약 1/100이다. Mode B로 떨어지면 세계가 3배 비싸지지만 90일 예산 안에 든다(§E12).

### E10. Phase 1 쇼케이스 순서

정렬 기준은 (구조적 승리의 크기) ÷ (T4-h × 리스크)다.

| 순서 | 데모 | Day | 세계 T4-h | 렌더 T4-h | 필요 모델 | 비디오 모델이 못하는 이유 |
|---|---|---|---|---|---|---|
| 1 | 왕복 v0 (실촬영 방, 차이 맵) | 14 | 1–1.5 | 0.05 | 없음 (2DGS 폴백 허용) | 돌아오는 길에 세계를 다시 상상한다 |
| 2 | 리그 스왑 6종 × 룩 50종 | 21 | 0 | 0.5 | CamFM-Hand | 테이크마다 세계가 바뀐다 |
| 3 | "Shot on 2×T4" 8K120 정지 세계 슬로모션 | 28 | 0 (+TexelFM l=15, 표기) | 0.5–1.0 | TexelFM v0 | 네이티브 8K120을 낼 수 없다 |
| 4 | 을지로 네온 (Corman: 밤, f/1.4 랙 포커스, 한글 SYMBOL, 보케, 플레어 직선, 웅덩이) | 55–60 | 4–6 | 0.3 | 교사 잠금, GB-Adapter, GeoHead, MassingFM(거리), GlyphSpotter | 한글이 끓고, 보케가 깊이와 무관하고, 플레어가 떠다닌다 |
| 5 | 왕복 v1 + 하룻밤 1,000 테이크 | 60 | 0 | 6.6 | — | 1,000번 생성하면 1,000개의 세계가 나온다 |
| 6 | 거울 복도 + 초록 무한 터널 | 62 | 2–4 | 0.3 | TL, GB-Adapter, TexelFM | 거울 속 세계를 다시 상상한다 |
| 7 | 경로 이식 릴 T4± | 65 | (벤치) | 0.2 | — | 공정한 A/B |
| 8 | Corman 슬라이더 (같은 장면 0.1–10 T4-h) | 70 | 2 | 0.2 | — | 우리만의 메시지 |
| 9 | 1999→2026 불릿타임 헌정 (와인잔 정지) | 75–85 | 4–8 + Hero FX CPU 4–12 h | 0.5–1 | TL, GB-Adapter, AesHead | 정지를 요청하면 숨을 쉰다 |
| 10 | 체호프의 컵 10분 원테이크 | 85–90 | 6–10 (+polish 4) | 1–2 | TexelFM, CamFM-Intent, MassingFM | extend할 때마다 drift가 쌓인다 |
| 11 | 직접 날아보기 (웹 뷰어) | 60+ | 0 | 0 | — | 상호작용이 불가능하다 |
| 12 | Powers of Ten | 90+ | 5–15 | 0.2 | TexelFM, 의미 효도(class id) | 스케일 전환에서 녹는다 |
| 13 | 빛만 흐르는 세계 / 매직아워 | 90+ | +3 | 0.3 | GeoHead 법선 | 조명 상태가 없다 |

**미루는 것.** 불릿타임 속 사람과 군중 얼굴은 2c 근처로 미룬다. 교사 1개로 다시점 정체성을 지키기 어렵다. Vision Pro 양안 광택은 P1-확장이다.

### E11. FREEZE-TAG on 2×T4

**운영 원칙.**
1. 세계 1개가 의도 4–12개를 다 찍는다.
2. T1·T2는 텍스트(또는 첫 프레임) 전용이다. WorldScript와 사람 개입은 금지한다.
3. Corman on/off를 둘 다 보고한다. 자해 카테고리도 공개한다.
4. 평가 모델은 R-EVAL 방화벽 뒤에 둔다.
5. T4± 경로 추출은 COLMAP(SIFT) 우선이다. VGGT는 벤치 트랙에서만 폴백으로 쓰고 표기한다.

| 항목 | 미니 벤치 (Day 90) | 전체 벤치 (Day 90 이후) |
|---|---|---|
| 구성 | C1 + C2 × 40 프롬프트 × 의도 4(push-in, orbit, 왕복, 핸드헬드 워크) × seed 2 | 240 프롬프트 × 의도 12, 약 500쌍 + bullet-time |
| **우리 세계 [수정]** | 160개 × 1.5–2.5 = **240–400 T4-h** (벽시계 5–8일) | 480–960개 → 0.7–2.4k T4-h |
| 우리 렌더 | 클립 640개 1080p + 4K 마스터, 약 3 T4-h | 약 12 T4-h |
| 경쟁사 API | 3모델 × 1,920클립 × 8 s = 15,360 s × $0.05–0.4/s = **$0.8–6.1k** (가격 확인) | $5–38k |
| GeCo 등 평가 [수정] | 2,560클립 × 15–60 s = **11–43 T4-h** | 약 100 T4-h |
| 사람 평가 | 3,000쌍 × 3명 → 능동 샘플링으로 약 2,000쌍. $1–5k + 전문가 $3–6k | $50–80k |

**단계 목표 (최종은 불변).**

| 시점 | 목표 |
|---|---|
| **Day 30** | X-T0–X-T4 통과 또는 업그레이드 확정. 교사 잠금(Day 18). 왕복 비트 동일. 4K 프레임 0.6 s 이하. X2 파일럿 VP ≥ 60%. l=13–14에서 HER ≥ 0.8인 arm 1개 이상. TexelFM 2AFC ≤ 60% |
| **Day 60** | X2 VP ≥ 70%. C1 생성 세계 20개에서 GeCo ≤ 0.04, HER ≥ 0.8, OIE ≤ 0.05(v1). SP ≥ 35%, TP ≥ 40%(중간 목표). 세계당 Mode A 2 T4-h 이하 |
| **Day 90** | C1에서 SP ≥ 45%와 GeCo ≤ 0.03을 동시에(v1). T2 VP ≥ 70%. 미니 벤치 사전 등록. 릴 3+1편 |
| **최종 (불변)** | SP ≥ 50% (vs Wan 3.0, Seedance 2.5)이면서 GeCo ≤ 0.03. TP ≥ 45%. VP T1 65/75%, T2·T4± 80%, T3 90%. TPR ≥ 0.99, MAT ≥ 90%, CI ≤ 실사 + 5%p. MTF50@8K ≥ 0.25. 카메라 튜링 ≤ 55% |

### E12. 90일 계획과 예산

**E12.1 주차별 계획**

| 주 | 표현·렌더러 | 교사·생성 | 카메라 | 데이터·평가 | 게이트 |
|---|---|---|---|---|---|
| 1 | **X-T0**. STRATA-lite v0.2 자료구조(TSQ, BC7/BC5/BC6H, Merkle, h_0 고정) | X-T0 교사 3종, residual rescaling. X-T1 생성 시작 | 카메라 스택 v0 | 자체 촬영(HDR, ArUco), GoPro 손 데이터, 경쟁 클립 발주 | — |
| 2 | COLMAP 4 SIFT + GLOMAP → 2DGS/TSQ, dual-cut | SDXL 대역으로 WSSC-lite 골격, TL 학습, TexelFM 피라미드 | 리그 6종 | X-T1 판정 | **D14 왕복 v0** |
| 3 | Defocus Pyramid, LH-split, 출사동 LUT, Hullin, FFT | **X-T2**, **X-T3**(8M/30M) | Hand 학습 | **X-T1 → 교사 잠금(D18)**, X0-lite | — |
| 4 | Ledger, Proof-of-Shot, 8K 경로, X-T4 ΔJOD | X-T2/X-T3 결과, Router 보정, GeoHead 학습 | 리그 스왑 릴 | X2 파일럿, FREEZE-TAG 하네스 | **D30** |
| 5–6 | PSH(SYMBOL 한글 MSDF, POINT, CURVE), TSQ 텍스처 완성 | GB-Adapter, MassingFM, VAE fp16 수리, WSSC-lite v0.2 통합 | Intent 학습, Corman v0 | C1 생성 세계 10개 | — |
| 7–8 | 평면 재귀 거울, OptiX secondary | 을지로 네온, (선택) 시네마틱 LoRA | Gaffer, CRM 쌍대 수집 | X2 본실행, C1 세계 20개 | **D60** |
| 9–10 | DIEL SDF + 분산, Hero FX, 스트리밍 | TexelFM 60M, **X-C** | DP-0-lite 탐색 | 미니 벤치 생성 시작 | — |
| 11–12 | 4K/8K 최종 렌더, 웹 export | 데모 polish | 데모 경로 | 미니 벤치 160세계, 사람 평가 | — |
| 13 | — | — | — | 기술 보고서, 사전 등록, 릴 3+1편 | **D90** |

**E12.2 예산 (T4-h)**

| 항목 | 상시 Mode A | 상시 Mode B | Kaggle |
|---|---|---|---|
| X-T0, 렌더러 개발·프로파일링 | 60 | 60 | 30 |
| X-T1, 교사 bake-off | 20 | 20 | 15 |
| 소형 모델 학습(+LoRA) | 240 | 240 | 110 (60M, AVA, LoRA 생략) |
| 데이터 준비(MVS 35, 합성 120, 타깃·특징 45) | 200 | 200 | 40 (EEVEE만, MVS는 로컬 CPU) |
| X1-lite 교사 보정(σ 점 스냅, 결정 곡선) | 20 | 40 | 10 |
| 실촬영 Oracle 세계 30개 | 45 | 45 | 25 (15개) |
| 개발 세계 250개 | 300 | 750 (Ledger 40% 전제) | 75 (60개) |
| 미니 벤치 160세계 | 320 | 800 | 160 (C1 80세계) |
| 데모 히어로 | 60 | 150 | 40 |
| X-C Corman | 100 | 250 | 30 |
| 렌더 | 80 | 80 | 30 |
| 평가 | 60 | 60 | 30 |
| 소계 | 1,505 | 2,695 | 595 |
| 실패 버퍼 | 450 (30%) | 400 (15%) | — |
| **합계 / 가용** | **약 1,955 / 3,200** | **약 3,095 / 3,200** | **595 / 390–780** |

**남는 예산의 용도.**
- 상시 Mode A의 여유 약 1,250 T4-h: TexelFM 스케일링, 장면별 WDM 실험 2–3개(150–300), 전체 벤치 선생성, Qwen-2512 승급 대비.
- Mode B는 여유가 거의 없다. 개발 세계는 768² 환산과 Ledger 재사용으로 운용한다.
- Kaggle 390 T4-h 시나리오에서는 **목표를 낮추지 않고 일정을 1.5–2배 늘린다**(D90 게이트 → D135–180). 상시 2×T4 확보를 권고한다.

### E13. ★ 제약이 무기가 되는 지점

| 메시지 | 근거 |
|---|---|
| 결정론은 정의에서 나온다 | 왕복 비트 동일성의 추가 비용이 0이다 |
| 재촬영의 한계비용은 렌더뿐이다 | 1080p 10초 테이크가 벽시계 10–22 s다. 하룻밤 1,000 테이크, 세계는 하나 |
| 8K·120fps는 구조에서 나온다 | 생성 비용은 fps와 무관하다. 8K는 +0.15–0.45 T4-h |
| **[신규] 미감 천장을 3주 차에 안다** | Teacher Ceiling Test 12 T4-h. 대형 팀은 파이프라인을 다 만든 뒤에 안다 |
| **[신규] Adam 없는 세계 생성** | TABP. GPU가 남아돌았다면 아무도 닫힌 형태를 찾지 않았을 것이다 |
| **[신규] 라벨이 무한한 작은 파운데이션** | MassingFM, TexelFM, GlyphSpotter는 절차적·CC0 데이터라 라벨 비용이 0이다 |
| INT4가 숨은 무기다 | T4 INT4 실측 114 TOPS(FP16의 2.7배)로 6B 교사가 두 장에 하나씩 올라간다 |
| 70W 촬영장 | 4K120 10초 ≈ 스마트폰 1–2회 충전 분량 |
| Proof-of-Shot | 해시를 공개하면 누구나 다시 렌더해 바이트 단위로 같은지 확인할 수 있다 |
| 가난이 미장센이 된다 | Corman. X-C로 검증 |
| 인수분해가 생존 조건이 된다 | 작은 모델은 정규화된 국소 입력에서만 통한다. 그래서 대역별 `git blame`으로 시스템을 감사할 수 있다 |

### E14. v1에서 버리는 것, 미루는 것

| v1 항목 | 처리 | 이유 |
|---|---|---|
| H100 64장, 0.4–0.8M H100-h | 폐기 | 제약 |
| CWM 4–8B + LBO 사전학습 10²³ | **폐기** → MassingFM 21M + TexelFM 30M + 세계별 solver | 10²³ ÷ 2.2×10¹⁸/일 ≈ 125년 |
| 장면 규모 amortized WDM | 연기 → 장면별 WDM 2–3개(여유 예산) | 교사 역전파 + fake score로 장면당 50–100 T4-h |
| LLM World Program | 폐기 → MassingFM + TE encoder, WorldScript DSL | 대형 LLM 금지(R-TE) |
| 닫힌 SOTA hero, GAN real 집합 | 폐기 → 교사 hero search + AesHead + DCA | 시스템 경로 금지 |
| DA3/VGGT/AnySplat/MegaSaM | 폐기 → COLMAP/GLOMAP/PatchMatch + GeoHead | 금지 |
| SelfCaption VLM, SigLIP/DINO `sem`, CLIP 신호 | 폐기 → 구조화 캡션, class id + TE, DCA | 금지 |
| LPIPS 효도 손실, $\mathcal L_{ocr}$ | **교체** → Laplacian + 교사 특징, GlyphSpotter | 금지(§C) |
| RAFT/UFM | 교체 → DIS/TV-L1 | 금지 |
| TRELLIS/Hunyuan3D, IC-Light, Difix3D+ | 폐기 | 금지 |
| latent `z` 정본 | 연기(슬롯만 예약) → explicit TSQ + BC | 디코더(학생)가 없고, BC 디코드는 공짜다 |
| 3DGRUT 단일 렌더러 | 변경 → TSQ 래스터 + OptiX 삼각형 + 평면 재귀 + SDF | T4 RT 처리량 |
| MC 16–64 spp DOF 타일 | 변경 → Defocus Pyramid (MC는 레퍼런스 검증용) | 4K 프레임 비용 3배 이상 |
| NRC, 경로추적 GI | 폐기(P1) | RAS로 굽는다 |
| 블록당 4–8뷰 중첩 투표(미세 레벨) | **변경** → OTBI 단독 소유 | T6. 뷰 수 1/2–1/4 |
| 미세 레벨 Adam 피팅 | **변경** → TABP | 10–19 GB OOM |
| Location 모드 기본 | 변경 → Directed 기본, 벤치는 Location-lite | 비용 1.5–2배 |
| X3 arm 6개 | 축소 → X-T2(평균 / OTBI+TABP / +WNF / Router), WDM 장면 1–2개 | 컴퓨트 |
| 사람 bullet-time, 군중, 아바타 | 연기(2c) | 교사 1개로 다시점 정체성 유지가 어렵다 |
| PIS 2.0 VDP 위험 배분 | 연기 | — |
| IMU 300h | 축소 → 100h + 확장 | 인력 |
| 실시간 뷰어 | **앞당김** | 시청자 기기가 렌더한다 |

### E15. 신규성 장부

| 확신도 | 항목 |
|---|---|
| **[신규] 높음** | Corman–Shannon Demand + Corman Optimizer(v0.2: 대역 평균 OTF). **Teacher Ceiling Test를 생성 세계 시스템의 Day-18 게이트로 쓰는 프로토콜** |
| **[신규] 중상** | **OTBI + TABP**(band-iso 정렬을 이용해 대역 결정을 닫힌 형태로 동기화하고 Adam을 없앰). **MassingFM**(절차적 데이터로 학습한 레이아웃 소형 파운데이션을 bedrock 소유자로). **σ-정밀도 카드 분할**. commit-before-shoot + Proof-of-Shot. T4용 TSQ. 거울 터널의 철분 초록을 포렌식에 쓰는 것. 벤치에서 "세계 1개 = 의도 N개" 상각을 비용 회계로 정식화 |
| **[조합]** | Federal Router(v1 관할 규칙의 측정 기반 운용). patchify TexelFM. Residual-stream rescaling(fp16). Teacher-DCA 미감. GeoHead 특징 공유 forward. semantic metric prior. Longuet-Higgins Split. Defocus Pyramid + 렌즈 게이팅. Judgment Ledger. Blender 오픈무비 카메라를 DP 정답 라벨로 |
| **[부품]** | gsplat, 2DGS, BBSplat, Textured Gaussians, OptiX, Kolb, Hullin, Newson, Malvar, Kraus–Strengert, TAESD형 AE, T2I-Adapter, DPT, Diffusion Classifier, SVDQuant/Nunchaku, COLMAP/GLOMAP, Frankot–Chellappa, Infinigen, CC0 PBR, BC6H/BC7/BC5 |

### E16. 리스크 → 업그레이드

| # | 리스크 | 감지 | 업그레이드 (후퇴 없음) |
|---|---|---|---|
| R1 | 교사 천장 | X-T1 < 60% | LoRA, N=256, 광학 스택, 교사 교체, Qwen-2512 승급 |
| R2 | INT4 품질이 미세 대역을 깎는다 | INT4 vs fp16 미세 대역 SFR > 2% | Mode C(commit은 fp16) |
| R3 | fp16 overflow | NaN 비율 | residual rescaling, 블록별 fp32, INT8 |
| R4 | Turbo의 임의 σ SDEdit 실패 | 결정 곡선이 1.5옥타브 초과 | σ 점 스냅, $\Pi_{b_l}$ 엄격 제한, 레벨 간격 2옥타브 |
| R5 | OTBI seam | 2AFC > 60% | apron 25%, multiband, 텍스처 마스킹 seam |
| R6 | MassingFM 레이아웃이 단조롭다 | 다양성 지표, 사람 판정 | 절차 문법 확장, Infinigen 비율 증가, 텍스트 조건 강화 |
| R7 | GeoHead 스케일·FOV 오류 | free-space veto 비율, 문 높이 분포 | semantic prior 가중 증가, MVS 실사 확대 |
| R8 | TexelFM 의미 셀 실패 | 2AFC > 60% | Router 이동, 60M, 클래스별 head |
| R9 | T4 래스터 속도 | X-T0 | 타일 컬링, half-res + 업샘플, 렌더는 밤에 |
| R10 | OptiX 동작·속도 | X-T0 | 평면 재귀 + SDF만 |
| R11 | 다른 아키텍처에서 비트 동일성 붕괴 | 교차 GPU diff | 보장 범위 명시, toolchain pin |
| R12 | Kaggle 쿼터·디스크 | 주간 잔량 | stateless job, 일정 연장, 상시 확보 |
| R13 | AesHead/DCA 리워드 해킹 | held-out 사람 판정 괴리 | 앙상블, 사람 스팟 체크, 판정쌍 재학습 |
| R14 | Corman 룩이 단조롭다 / 체리피킹 비판 | 범주별 SP | on/off 병기, 낮·맑음 프롬프트 의무 포함 |
| R15 | 엔지니어링 범위 폭발 | 렌더러 PR 비중 | 단계화(2DGS → TSQ → OptiX), CUDA C++만 |
| R16 | 라이선스(AVA, DL3DV, ScanNet++ NC, SDXL OpenRAIL, CameraBench) | 법무 게이트 | 상업 경로는 자체 촬영, CC0, 절차 데이터, Apache 교사 |
| R17 | T4 툴체인 지원 종료 | CUDA 로드맵 | 버전 pin, 컨테이너 동결 |

### E17. 다른 렌즈, 리서처, 창업자에게

**세계 엔진 렌즈에 주는 것.**
- `render(view, level_mask, θ)`: 최종 렌더와 같은 코드다.
- `TABP`, OTBI 뷰 생성기, Router 맵, WNF Philox 패스, Ledger, Corman–Shannon 수요 집합, MassingFM 출력(shell, slot).

**세계 엔진 렌즈에 요청하는 것.**
- view-step 예산을 §E9 범위로 맞춰 달라.
- 미세 레벨 다중 뷰 투표를 OTBI로 바꾸는 데 동의해 달라(T6 근거).

**데이터·소형 모델 렌즈와.** TL, GB-Adapter, GeoHead, TexelFM, MassingFM의 소유자를 한 명씩 정하자.

**Phase 2 렌즈에 주는 것.**
- TSQ의 slot과 inst, 고전 시뮬(MPM, FLIP, 강체)로 atom을 변형하는 훅, DIEL과 Hero FX.
- 정지 슬라이스는 그대로 2a의 렌더 경로다.
- SMPL-X는 학습된 통계 모델이고 비상업 라이선스라는 점(§C12)을 함께 전달한다.

**리서처 검증 요청.**
1. T4 실측: Z-Image, klein, SDXL의 해상도별·정밀도별 forward 시간.
2. Nunchaku INT4가 T4(sm_75)에서 동작하는지, 속도는 얼마인지.
3. bf16 네이티브 DiT의 fp16 NaN과 residual rescaling 효과.
4. Turbo 계열의 임의 σ SDEdit 품질.
5. 비디오 모델 프레임과 이미지 아레나 Elo의 관계(X-T1).
6. Kaggle T4×2 쿼터 소모율, RAM, 디스크.
7. CUDA BC6H/BC7 텍스처 객체 지원 버전.
8. Turing NVENC 8K HEVC 10-bit 처리량.
9. 경쟁 API 초당 가격.
10. FLUX.2 klein base(비증류) 공개 여부.

**창업자 결정 요청.**
1. **R-TE**: 교사 TE는 encoder 전용이고 생성적 사용을 금지한다. 승인 여부.
2. **R-DATA**: 다른 사전학습 모델이 라벨을 붙인 공개 데이터셋(DataDoP 등)을 학습에 쓸지, 평가 전용으로 둘지.
3. **R-EVAL**: 평가 모델은 사람의 모델 선택까지만 허용하고 자동 최적화는 금지한다. 승인 여부.
4. **교사 승급**: X-T1 실패 시 Qwen-Image-2512(20B, INT4)를 허용할지. 세계 비용이 3배가 된다.
5. **상시 2×T4 확보**: Kaggle 단독이면 일정이 1.5–2배 늘어난다.

> **맺음말.** 비디오 모델은 같은 세계를 240번 다시 그린다. 우리는 세계를 한 번 짓는다. 덩어리는 MassingFM이 먼저 놓고, 그 위의 옷은 교사가 대역마다 입히고, 결은 TexelFM이 짠다. 그 한 번도 카메라가 기록할 수 있는 만큼만 짓는다. 이 프리모템에서 가장 중요한 숫자는 FLOPs가 아니라 교사 Elo였다. 그래서 그것을 18일째에 재고, 그 값을 OTBI, TABP, Router, INT4로 치른다. 세계 하나에 1–1.5 T4-h, 4K120 10초에 스마트폰 한 번 충전할 전기가 든다. 개소리 같지만 장부는 맞는다. 그리지 말고, 찍어라.

**참조 파일:** docs/PARADIGM.md, README.md, docs/research/landscape-2026-10.md, docs/lenses/premortem/02-representation-v0.2.md
