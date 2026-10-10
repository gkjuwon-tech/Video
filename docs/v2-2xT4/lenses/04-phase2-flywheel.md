> **출처 메모.** 2×T4 + 수십M from-scratch 제약으로 다시 돌린 설계 워크숍에서 [Phase 2 + 저예산 플라이휠] 렌즈 담당이 쓴 원안이다. 최종 결정은 [`docs/PARADIGM-v2.md`](../../PARADIGM-v2.md)를 따른다. 이 문서는 Higgsfield Soul 하이브리드 결정 **이전**에 작성됐다.

---

# Phase 2 v2-T4: "작은 뇌, 정확한 우주, 큰 탐색"

### 2×T4에서 블록 우주를 돌리는 설계와 저예산 데이터 플라이휠

> 렌즈: **Phase 2 설계자 + 저예산 데이터 플라이휠 전략가**
> 기준 문서: `docs/PARADIGM.md` §0.5, §1, §2, §4, §7, §8, §9, §10, 그리고 `lenses/premortem/04-phase2-v2.md`
> 표기:
> - **[신규]** 이 제약 때문에 새로 생긴 아이디어. 공개된 선례를 찾지 못했다.
> - **[조합]** 부품은 이미 있고, 묶는 방식이 새롭다.
> - **[부품]** 있는 것을 빌려 쓴다.
> - **[v1 유지 / 축소 / 폐기 / 미룸]** v1 항목을 어떻게 처리하는지.
>
> "(추정)"이 붙은 처리량 수치는 모두 1–2주차 벤치(P2-X0)에서 실측값으로 바꾼다. 목표는 하나도 낮추지 않았다.

---

## 0. TL;DR

1. **v1 Phase 2 예산은 대부분 학습이 할 일이 아닌 곳에 쓰이고 있었다.** v1 Phase 2 컴퓨트는 native WFM 사전학습($10^{23}$–$10^{25}$)과 수천만 클립 lift가 대부분을 차지했다. 이 둘은 버린다. 대신 세 가지가 일한다.
   - (a) 빌린 이미지 교사. SD v1 하나에만 약 $7\times10^{22}$ FLOPs가 들어갔다. 우리 1년 예산의 약 70배가 이미 선불로 들어가 있는 셈이다.
   - (b) 학습이 필요 없는 고전 물리, 광학, 음향.
   - (c) 테스트타임 탐색(Milestone SMC).

   작은 모델들은 이 셋이 메우지 못하는 **저차원 잔차**만 배운다. 결과적으로 Phase 2의 학습 파라미터는 모델 10개, 합계 약 125M이고, **1세대 전체 학습은 약 290 T4-h, 2×T4로 약 6일**이다.
2. **병목은 학습이 아니라 교사 호출(TC), 실데이터, 렌더다.** 그래서 회계 단위를 FLOPs가 아니라 TC로 잡는다. 모든 TC를 기록해서 INT8 소형 critic으로 증류하고(**Teacher Exhaust**), critic 단계를 거쳐 교사를 부르는 **Judge Cascade**를 둔다. 플라이휠이 한 번 돌 때마다 블록당 TC가 줄어든다.
3. **Coverage–Precision Split** [신규 정식화]. 물리 시뮬레이터는 정확하고 공짜인 검증기(verifier)이고, 사영기(projector) 역할도 한다. 그래서 생성기는 정밀할 필요가 없다. 물리 사영이 끌어들일 수 있는 **분지(basin)에 떨어뜨리기만** 하면 된다. 같은 비용이면 30M 모델은 3B 모델보다 100배 많이 뽑는다. T4에서 모션 best-of-4096은 약 20–40초다.
4. **비디오 증인을 크로노포토 증인(Chronophotographic Witness)으로 바꾼다** [신규, 이 제약이 아니었으면 아무도 안 했을 것]. 시간을 사진 한 장으로 접는 방식이다. 예: 에저턴(Edgerton)의 스트로보 다중노출, 마레(Marey)의 크로노포토, 장노출 궤적. 이미지 교사는 "튀는 공의 스트로보 사진"을 수없이 봤다. 그래서 이미지 교사가 약 0.3–15 Hz 대역의 **시간 증인**이 된다. 보정 데이터는 실제 240fps 영상에서 고전 합성만으로 만든다.
5. **저주파 서사는 사람이 소유한다.** LLM도 비디오 모델도 쓸 수 없다. 그래서 "누가 무엇을 언제" 하는지(0–1 Hz)는 감독의 ShotScript와 폰으로 찍은 연기가 정한다(**Act, don't prompt**). 기계는 중·고대역(타이밍 미시통계, 접촉, 물리)을 맡는다. 시간 연방에서 가장 낮은 관할권을 사람에게 준 것이다.
6. **Reader는 직접 만든다.** VGGT, DA3, GVHMR, SAM2는 금지다. 대신 고전 SfM과 optical flow에, 우리 렌더로만 학습한 Reader-S(약 26M)를 더한다. 이미지 교사로 합성 렌더를 실사처럼 "세탁"하는데, 라벨이 있는 주파수 대역은 건드리지 않는다. 이것이 **Band-Safe Reality Transfer(BSRT)** [조합/신규]다.
7. **T4에서 물리는 상대적으로 싸다.**
   - T4는 H100 대비 실효 텐서 FLOPs가 약 1/25다. 메모리 대역은 약 1/10이다.
   - 시뮬과 렌더는 메모리 대역에 묶이므로 학습보다 덜 손해 본다. 그리고 시뮬은 CPU에서도 돈다.
   - 중형 강체 SMC(N=1024, 10초)는 3–12분(추정)이다. 유리 파괴는 MPM이 아니라 Voronoi + 강체로 한다. 유체는 multi-fidelity SMC로 처리한다.
8. **로드맵은 프로젝트 시작(M0) 기준이다.**
   - 2a Frozen Action: M3–M9. v1과 거의 같다.
   - 2b: M8–M16.
   - 2c: M14–M28. FLOPs가 아니라 사람 데이터가 병목이어서 v1보다 6–10개월 늦다.
   - 2d: M26–M42 이후.
   - 승리 조건은 v1 그대로 둔다.
9. **플라이휠 1회전은 약 1일이다.** 단, Phase 2가 GPU의 약 2/3를 쓰는 시기 기준이다. 시드 기간에는 4–6일, Kaggle 모드에서는 주 1회다. 10회전 뒤 목표는 블록당 TC 4–8배 감소, SMC 입자 수 4–16배 감소, 하루 블록 생산량 8배 증가다.
10. **스케일 원칙.** 텐서 계약, 데이터 자산(Block Bank, Teacher Exhaust, Gold Set), μP 파라미터화는 모델 크기와 무관하다. 컴퓨트가 1000배로 늘면 같은 슬롯에 큰 모델을 끼우면 된다. 아키텍처는 바뀌지 않는다.

---

## 1. 전제와 회계

### 1.1 T4 회계 단위

| 항목 | 값 | 메모 |
|---|---|---|
| FP16 텐서 피크 | 65 TFLOPS | 70W 전력 캡 때문에 클럭이 떨어진다. 대형 GEMM 실측은 대략 25–45 TFLOPS(추정) |
| 소형 모델 학습 실효 | 13–20 TFLOPS/장, **계획값 16** | 30M급은 커널 런치, LayerNorm, optimizer 비중이 커서 MFU가 약 25%다 |
| **1 T4-h (학습)** | **$5.8\times10^{16}$ FLOPs** | |
| 상시 2×T4 | 48 T4-h/일 = $2.8\times10^{18}$/일. 90일 $2.5\times10^{20}$, 1년 $1.0\times10^{21}$ | v1 Phase 1 누계($5.8\times10^{23}$–$1.2\times10^{24}$)의 1/600–1/1,200. v1 2c 누계의 1/3,600–1/7,200 |
| Kaggle T4×2 | 주 약 60 T4-h (주 30 세션-시간 × 2장 가정, **확인 필요**). 상시의 약 18% | 세션은 9–12시간. CPU는 4 vCPU급 |
| 메모리 대역 | 320 GB/s (H100의 약 1/10) | 시뮬과 렌더는 대역폭에 묶이므로 텐서 연산(약 1/25)보다 덜 손해 본다 |
| FP32 / INT8 / RT | 8.1 TFLOPS / 130 TOPS / RT 코어 40개 | FP32는 시뮬, INT8은 critic 추론(TensorRT), RT는 굴절·반사·음선 추적 |
| **TC (Teacher Call)** | 교사 forward 1회 @512² ≈ 0.2 s (SDXL급 fp16, 추정) → **약 18k TC/T4-h** | CFG를 쓰면 2 TC. DMD 1 샘플-스텝은 6–8 TC-eq. 1024²는 ×4–5. SANA급 교사면 ×0.3–0.6 |
| 호스트 | **≥16 vCPU, ≥64 GB RAM 가정** | CPU가 세 번째 가속기다(시뮬, SfM, lift). 이 가정이 깨지면 §6 수치를 다시 잡는다 |

교사 선택은 Phase 1 렌즈 몫이다. 이 문서는 보수적으로 SDXL급(UNet 2.6B, fp16)을 기준으로 잡았다.
- SDXL VAE는 fp16에서 오버플로가 나므로 fp32로 돌린다. 교사 부속이다.
- T4 커널 선택: FlashAttention-2는 쓸 수 없다. PyTorch SDPA mem-efficient 백엔드나 FA1/xformers를 쓴다.

### 1.2 Phase 2에서 컴퓨트는 어디로 가는가 (1년차 Phase 2 트랙, 약 6,000 T4-h 가정)

| 용도 | T4-h/년 | 비중 |
|---|---|---|
| 작은 모델 학습 (3–4세대 재학습 포함) | 800–1,200 | 15–20% |
| **TC**: 판정, 슬라이스 DM, MAN-S/LoRA 학습, BSRT | 2,500–3,000 | **40–50%** |
| GPU 시뮬 (CPU로 못 하는 FLIP, MPM, 대형 배치) | 400–700 | 7–11% |
| 렌더 (데이터, 데모, 아레나 납품) | 900–1,200 | 15–20% |
| 벤치, 평가, 여유분 | 300–500 | 5–8% |

> **핵심 관찰.** Phase 2에서 학습은 전체의 1/5 이하다. 그래서 레버는 "더 큰 모델"이 아니라 **"더 싼 판정"**과 **"더 똑똑한 제안"**이다.
> 그리고 Phase 2 학생의 절반은 **픽셀을 아예 보지 않는다**(EPD-S, MoPri-S, Feas-S, F_φ-S). 이 학생들의 데이터 생성에는 TC가 들지 않는다.

### 1.3 데이터 순도 등급 [신규 정책]

"데이터는 허용, 모델은 금지"라는 규칙을 우회로 없이 지키기 위한 장부다.

| 등급 | 정의 | 예 | 시스템 경로 학습 |
|---|---|---|---|
| **D0** | 센서, 모캡, 시뮬 GT | 우리 렌더와 시뮬, Garage Gold Set, CMU·AMASS 원본 모캡, Physics 101 측정값 | 허용 |
| **D1** | 고전 알고리즘이 만든 라벨 | COLMAP/GLOMAP 포즈, 배경차분 마스크, 오디오 onset, ArUco | 허용 |
| **D2** | 사람이 단 주석 | COCO keypoints, HumanML3D 텍스트 | 허용 |
| **D3** | 사전학습 모델이 만든 라벨 | SpatialVID의 pose/depth, DataDoP 궤적(MonST3R), Motion-X. AIST++는 2D 검출기 기반 triangulation이어서 확인이 필요하다 | **기본 금지. 평가와 분석에만 쓴다.** 창업자 판결 필요(§17) |

라이선스는 등급과 별도로 따진다. AMASS와 SMPL 계열은 비상업 라이선스다. 그래서 상업 경로는 CMU 모캡과 자체 캡처로 간다.

### 1.4 빌린 것 한 장

- **이미지 교사 1개**: UNet, VAE, 텍스트 인코더를 묶어 이미지 모델 하나로 본다.
- **가벼운 어댑터**는 교사 파라미터의 2% 미만으로 제한한다.
  - fake-score LoRA r32 (약 23M)
  - Chrono-LoRA r16 (약 12M)
  - 캐릭터별 identity LoRA r8 (약 6M)
- **Teacher-as-Lingua-Franca** [조합]. 모든 작은 모델은 교사의 텍스트 인코더 임베딩(CLIP-L + bigG)을 조건으로 받는다. 임베딩은 미리 계산해서 캐시한다. 작은 모델들이 의미 일반화를 공짜로 얻는다. 이것이 규칙("텍스트 인코더는 이미지 모델의 일부") 안에 들어가는지는 창업자 확인이 필요하다(§17).
- **LLM은 금지된다.** 그래서 Phase 2 사건 스크립트는 ShotScript DSL의 `event` 구문으로 받고, 결정론적 파서로 처리한다. 자유 텍스트는 교사 텍스트 임베딩으로만 들어온다.
- **perceptual loss도 교사 안에서 만든다.** VGG와 LPIPS는 사전학습 모델이라 학습 루프에 넣지 않는다. 교사 VAE latent 거리와 교사 UNet 중간 feature로 대신한다(선례: LatentLPIPS). LPIPS는 평가에만 쓴다.

---

## 2. 다섯 원리 (제약 버전)

**P1. Coverage–Precision Split** [신규 정식화]

$$
P_{hit}(N)=1-\big(1-q_\phi(\mathfrak B(A))\big)^N,\qquad
\mathfrak B(A)=\{x:\ \Pi_{phys}(x)\in A\},\qquad
C=N(c_{gen}+c_{ver})+k\,c_{TC}
$$

- $A$는 좋은 결과의 집합이다. 얇은 다양체라서 거기 정확히 떨어뜨리기는 어렵다.
- 그런데 물리 사영 $\Pi_{phys}$(MJPC 추종, 접촉 투영, SMC rejuvenation)가 있으면 목표가 두꺼운 **분지** $\mathfrak B(A)$로 넓어진다.
- 그러면 생성기가 할 일은 분지 위에 확률질량 $q\ge10^{-3}$을 두는 것뿐이다. $N=4096$이면 $P_{hit}\approx98\%$다.
- 비용 $c_{gen}$은 파라미터 수에 비례한다. 그래서 같은 예산이면 작은 모델 × 큰 탐색이 이긴다. 정확한 검증기가 있는 영역(물리)에서만 성립하는 경제학이다.
- 언어 모델에는 이런 검증기가 없다. 우리에게는 뉴턴이 있다.

**P2. Tiny Federation.** 각 모델의 크기는 **그 모델이 맡은 대역의 내재 차원**에 맞춘다.
- 사건 집합: 슬롯 256개 이하.
- 인체 모션: 프레임당 약 140차원, 저랭크.
- 셰이딩 잔차: 16×16 국소 영역.

물리가 이미 아는 것을 다시 배우는 파라미터는 0개로 둔다.

**P3. TC는 화폐다.** 모든 교사 호출을 (입력 해시, σ, 교사 버전, 점수)로 기록한다(**Teacher Exhaust**). 이 로그로 INT8 critic을 증류해서 판정을 4단 캐스케이드로 만든다.
1. 불변량
2. RGF
3. INT8 proxy
4. 교사

호출의 10%는 무작위 후보에 쓴다. proxy가 자기 편향을 강화하는 echo chamber를 막기 위해서다.

**P4. 사람이 저대역을 소유한다.** 0–1 Hz 서사(누가, 무엇을, 언제)는 감독의 스크립트와 연기가 정한다. 기계는 그보다 높은 대역을 정확하게 만든다. 이것은 v1 §8.2 Spectral Division of Labor에서 증인 자리에 사람을 넣은 것이다.

**P5. 실리콘을 남김없이 쓴다.** 연산 유닛마다 일을 정해 둔다.

| 유닛 | 담당 |
|---|---|
| Tensor FP16 | 교사, 학생 학습 |
| INT8 | critic 캐스케이드 |
| CUDA FP32 | GPU 시뮬 |
| RT 코어 | 굴절, 반사, 음선 추적 |
| CPU | MuJoCo, SfM, lift, 모달 해석 |
| NVENC | 프리뷰 인코딩 |

규칙: **GPU에서 돌릴 이유가 없는 일은 CPU에서 돌린다.** CPU MuJoCo는 결정론적이어서 Merkle 재현성에도 유리하다.

---

## 3. 시스템 한 장과 "구멍 메우기 지도"

```
ShotScript (event/blocking 구문) ⊕ 자유 텍스트 → 교사 텍스트 임베딩(캐시)   ⊕ 폰 연기 영상(선택, 2c)
  │
  ▼
[W_canon] Phase 1 세계(다른 렌즈) + Reveal-Bundle Demand edit-closure (TC, 사건이 드러낼 면만)
  │       물성 ψ = 재질 클래스 → 공학 핸드북 표 (학습 0), 충돌 프록시 = V-HACD (고전)
  ▼
[Events] EPD-S 28M: 사건 집합 제안 (스크립트 슬롯 고정 inpainting, Feas-S guidance)
  │      Chaos Bank 검색으로 세트피스 warm start
  ▼
[Realize] Milestone SMC — CPU MuJoCo | GPU MJX/Warp | Voronoi 강체 | FLIP | MPM | XPBD   (학습 0)
  │       PCGD-0 확률 물성(+F_φ-S 4M, 2b+) · 캐릭터: MoPri-S 32M best-of-N → MJPC 접촉창 사영
  │       Judge Cascade: 불변량 I1–I5 → RGF → INT8 proxy(Hero/Chrono/Feas) → TC(슬라이스+크로노) → 감독
  ▼
[Block] (E, s(t), seeds, sim 버전) → Merkle commit → Block Bank (= 플라이휠 데이터)
  │
  ▼
[Shoot] DP-0 세계선 γ(τ) → 3DGRUT 셔터 적분 → MAN-S 셰이딩 → Denoiser-S → 렌즈·센서 (Read-Only, I6)
        음향: 사건 J → FEM 모달 합성 → 이미지 소스 + RT 코어 음선 추적 → 바이노럴/앰비소닉 (학습 0)
```

**구멍 메우기 지도.** v1이 큰 모델로 해결하던 능력을 무엇으로 메우는지 정리한다. (a) 빌린 교사, (b) 구조와 고전 알고리즘, (c) 테스트타임 최적화, (d) 작은 모델이다.

| 능력 | v1 | 2×T4 판 | 메움 |
|---|---|---|---|
| 정지 순간의 외관 | 이미지 교사 slice DM | 같음 | (a)+(c) |
| 저주파 의미(누가, 무엇을, 언제) | 비디오 증인 + LLM | 감독 스크립트, 폰 연기, EPD-S, 교사 텍스트 임베딩 | 사람 + (d) + (a) |
| 시간 구조의 사실감(궤적 모양, 간격) | 비디오 증인(≤ f_c) | **크로노포토 증인** | (a) |
| 사건 타이밍 통계 | 대규모 lift 실사건 prior | Garage Gold Set, 오디오 간격 증인, 시뮬 | (b)+(d) |
| 동역학 | PCGD (F_φ 0.3–1B) | 고전 시뮬 + PCGD-0 확률 물성 + F_φ-S 4M | (b) |
| 혼돈 제어 | Milestone SMC | 같음. N 조정, CPU 경로, Chaos Bank 추가 | (c)+(b) |
| 캐릭터 모션 | motion diffusion 0.1–1B + PHC | MoPri-S 32M × best-of-N + MJPC | (d)+(c)+(b) |
| 움직이는 물체의 셰이딩 | MAN 0.2–0.5B | MAN-S 1.5M | (d)+(a) |
| 숨은 면과 조명 분해 | Edit-Closed full | Reveal-Bundle Demand | (c)+(a) |
| 실영상 이해(lift) | DA3, VGGT, GVHMR, SAM2 등 | 고전 SfM/flow + Reader-S(BSRT) | (b)+(d)+(a) |
| 소리 | 모달 합성 + TTS | 모달 합성(학습 0), 음성은 녹음 | (b) |
| 커버리지 연출 | Coverage Compiler | 같음(고전) | (b) |
| 판정 비용 | 교사 직접 호출 | Judge Cascade + Teacher Exhaust | (d) |

---

## 4. 판사 재편: 비디오 모델 0개, 판사 넷과 사람 하나

### 4.1 증인 대체 매트릭스

v1의 비디오 증인이 하던 일을 하나씩 대체한다.

| v1 증인의 임무 | 대체 수단 | 근거 데이터 |
|---|---|---|
| 저주파 의미, 연출, 안무 | **감독(ShotScript, 폰 연기)**, EPD-S(교사 텍스트 임베딩 조건), 실제 증인 검색 | D0/D2 |
| 거친 사건 타이밍(≤ f_c) | 실사건 통계 + **크로노포토 증인** | Gold Set, 오디오, 240fps |
| 의도와 퍼포먼스 latent z | 감독 테이크 선택, 연기 lift | 사용자 영상 |
| 희귀 사건을 "상상"해 주는 공급원 | **상상하지 않고 굴려 본다.** 절차 세트피스 × SMC 탐색, Chaos Bank | 시뮬(D0) |
| 평가 | 외부 API(Seedance, Kling, Veo, Wan 3.0)를 **평가 전용**으로 쓴다. Arena와 Spectrum Report | API 출력은 학습 금지 |

### 4.2 기울기 라우팅 표 (v1 §8.2 개정)

| 판사 | 판정 대상 | 기울기 또는 선택이 흐르는 곳 | 차단 |
|---|---|---|---|
| 이미지 교사 슬라이스 (TC) | 정지 순간의 공간 사실감 | θ, MAN-S, intrinsics, slot (기울기). s(t)와 E는 **선택만** | s(t), E, ψ |
| **크로노포토 증인** (교사 + Chrono-LoRA) | 궤적 모양, 노출 간격, 포즈 연속성 (0.3–15 Hz) | EPD-S의 거친 타이밍, MoPri-S의 저대역 (RWR과 선택만) | 외관, 접촉 고대역 |
| 물리 (고전 시뮬 + 불변량 + RGF) | 불변량, 고대역 동역학, 접촉 | PCGD-0 ψ 분포, F_φ-S, MJPC | 외관 |
| 실사건 통계 (Gold Set, 오디오, 모캡) | 타이밍과 간격 분포 | EPD-S (MMD 정규화), PCGD-0 | 외관 |
| 감독 (사람) | 0–1 Hz 서사 | 스크립트 슬롯(하드 제약), 테이크 선택 | — |
| 외부 API | Arena 비교 | 없음 | 전부 |

### 4.3 크로노포토 증인 (Chronophotographic Witness, "에저턴 판사") [신규, 확신도 중]

**발상.** 비디오 모델이 금지되면 질문이 바뀐다. "**시간을 담고 있는 사진은 무엇인가?**" 답은 이미 사진사에 있다.
- 에저턴의 스트로보 다중노출(튀는 공의 포물선, 1957년 우유 왕관)
- 마레(Marey)와 머이브리지(Muybridge)의 크로노포토
- 스포츠 시퀀스 사진
- 장노출 궤적과 라이트 페인팅
- 사진 판독기(photo-finish)

이미지 교사는 이런 사진을 수없이 봤다. 그러니 **4D 블록을 시간이 접힌 정지 사진 한 장으로 렌더하면, 이미지 교사가 동역학을 판정한다.**

**렌더 모드.** 고정 증인 카메라 $c_w$를 두고 스트로브 시각 $t_k=t_0+k\Delta$, $k<K$를 고른다.

$$
X_{add}=\mathrm{Cam}\Big(\sum_k \mathcal R_{strobe}(W(t_k),c_w)\Big),\qquad
X_{mat}=\mathcal R(W_{bg})\odot\prod_k(1-m_k)+\sum_k m_k\odot\mathcal R(W(t_k)),\qquad
X_{long}=\tfrac1S\!\int\!\mathcal R(W(t))\,dt
$$

- $X_{add}$는 Edgerton식 가산 다중노출이다. 어두운 배경에 섬광 조명을 쓰고, 렌더러로 물리적으로 정확하게 만든다.
- $X_{mat}$는 시퀀스 사진식 매트 합성이다. 깊이 순서를 지킨다.
- $X_{long}$은 장노출이다. LED 마커만 남기는 라이트 페인팅 모드도 있다.

**Band-Isometric Strobe (BIS)** [신규]. v1의 band-isometric 뷰를 시간 축으로 옮긴 것이다.
- 노출 간 화면 변위를 교사가 결정하는 공간 대역에 맞춘다: $\Delta_b=d^*_b/\lVert v_{screen}\rVert$, $d^*_b\approx24$–$64$ px @512².
- 관측 가능한 시간 대역은 $f\in[1/(K\Delta),\,1/(2\Delta)]$다.
- $(K,\Delta)$ 조합 3–4개로 0.3–15 Hz를 타일처럼 덮는다.

**점수. 시간 왜곡 대조(PC-ELBO의 이미지판).**

$$
\Delta_\chi(h)=\mathrm{DR}_b\big(X(h)\big)-\mathbb E_{w\sim\mathcal W}\,\mathrm{DR}_b\big(X(h\circ w)\big),\qquad w_\chi=e^{-\beta\Delta_\chi}
$$

- $\mathcal W$는 단조 시간 왜곡의 가족이다.
  - 등속화
  - ease-in-out 평활화 (비디오 모델식 병)
  - 중력 스케일 ×0.7, ×1.4
  - 시간 역전 (에너지가 흩어지는 계에서는 튀는 높이가 커진다)
- common random numbers로 같은 ε와 σ를 쓴다. 노출 하나하나의 외관이 같으므로 렌더 아티팩트는 상쇄되고, **시간 구조만** 남는다.
- 가산 합성은 순서에 불변이다. 그래서 순열 대신 **간격 왜곡**을 쓴다.

**사용처.**
1. SMC 가중치
2. EPD-S와 MoPri-S의 RWR 보상
3. QBC(v1 재활용): 가설 사이에서 의견이 가장 갈리는 $(c_w,K,\Delta)$를 고른다
4. 비용: 판정 1회 = 모드 3 × σ 2 × CFG 2 × (원본 1 + 왜곡 4) = 60 TC, 약 12초. 그래서 캐스케이드 상위 k개에만 쓴다. 나머지는 ChronoProxy-S(INT8)가 맡는다.

**보정 데이터는 사전학습 모델 0개로 만든다.**
- 고정 카메라 240fps 실영상에서 고전 합성으로 만든다. 배경 median, 배경차분 마스크(D1)를 쓰고 가산 또는 매트로 합성한다.
- 왜곡 negative는 **다른 실제 프레임을 고르는 것**만으로 만든다. 240fps면 충분히 촘촘하다. 보간으로 새 프레임을 합성하지 않으므로 모든 노출이 진짜 사진이고, 달라지는 것은 간격뿐이다.
- 교사가 둔감하면 Chrono-LoRA를 붙인다. r16, 약 12M, 교사의 0.5% 미만이다. 실제 합성 약 3만 장으로 학습하고 약 10 T4-h가 든다. "아주 가벼운 어댑터" 범위 안이다.

**정직한 한계.**
- 교사가 "보기 좋은 등간격"을 선호하는 미감 편향을 가질 수 있다. 그래서 대조 점수만 쓴다.
- 선례:
  - Grid Diffusion(CVPR'24)은 비디오를 이미지 격자로 생성한다.
  - 블러와 장노출 연구가 있다.
  - 이미지 생성기를 스트로보 합성으로 **물리 및 타이밍 판정기**로 쓴 선례는 찾지 못했다. 선례 조사는 계속한다.
- 실패하면(LoRA를 붙여도 AUC < 0.65): 크로노 증인은 "모션 궤적의 미감"(히어로 선택)에만 쓴다. 시간 사실감은 RGF와 실통계로 맡긴다(v1 H7 경로).

**덤으로 생기는 쇼케이스.** 에저턴 헌정: 1957년 우유 왕관에서 2026년 와인 왕관으로. 같은 블록에서 스트로보 스틸, 불릿타임, 스피드 램프를 모두 뽑는다.

### 4.4 실제 증인 (사전학습 모델 0개)

- **Garage Gold Set** [조합]
  - 폰 3–4대로 240fps 슬로모를 찍는다. ChArUco로 보정하고, 박수와 오디오 상호상관으로 동기화한다.
  - 컬러 마커, 배경차분, 삼각측량, Kabsch로 강체 6DoF를 얻는다.
  - 사건 시각은 240fps + Blur-as-Measurement + 오디오 onset으로 ±2 ms를 노린다.
  - v0 목표: 6시간, 약 2만 사건. 장비 $0–2k.
- **오디오 간격 증인** [조합/신규 소].
  $$\Delta t_{ij}=t^{on}_i-t^{on}_j=(t^{ev}_i-t^{ev}_j)+\beta(d_i-d_j)+(n_i-n_j)$$
  - v1 F6을 괴롭힌 클립별 AV 오프셋 $\delta_{clip}$은 **간격에서 상쇄된다.** 음원이 움직이지 않으면 거리 항도 사라진다.
  - 연속 바운스 간격의 비 $T_{k+1}/T_k=e$는 반발계수다. 웹 오디오 수천 개와 메타데이터 범주("탁구공", "농구공")만으로 **재질별 반발계수 분포**를 학습 0으로 얻는다.
  - 같은 방법으로 발걸음 cadence, 드리블 주기, 문 닫힘 시간, 유리 파괴 지속시간도 얻는다.
  - v1의 AV-Physics Timing Solver는 4D 거리가 있는 Gold Set에서 유지한다.
- **모캡과 측정 데이터셋**: CMU(D0), AMASS(D0, 비상업), Physics 101(D2).
- **Event-Demand Lift** [조합]. 웹 영상은 Reader-S로 **사건 그래프 수요가 비어 있는 범주만** lift한다(능동 획득). 연간 1e4–1e5 클립이다. v1은 2천만 클립이었다.

### 4.5 RGF on T4 [v1 유지]

- ghost force는 lift된 궤적에 고전 역동역학(inverse dynamics)을 적용해 구한다. CPU에서 싸다.
- $p_{real}(F_g)$는 범주별로 Gold Set에서 추정한다.
- Reality Residual Budget은 그대로 둔다.
- lift 품질이 오를 때마다 $p_{real}$을 다시 추정한다.

### 4.6 시간 연방 지도 v2-T4

| 공간 \ 시간 | 저 (0–1 Hz) | 중 (1–8 Hz) | 고 (>8 Hz) |
|---|---|---|---|
| 레이아웃과 서사 | **감독**, EPD-S, 실제 증인 검색 | EPD-S 타이밍 + 크로노 증인 | 사건(물리) |
| 형태와 모션 | MoPri-S best-of-N + 크로노 증인 | MoPri-S + MJPC 사영 | 물리 접촉, PCGD-0 |
| 외관 | 이미지 교사 (canonical) | 렌더러 (블러 적분) | 렌더러, 센서, MAN-S |
| 확률적 물질 | 절차적 바람장(Stam) | SMF sway (seed) | 파티클 물리 |
| 카메라 | DP-0 (다른 렌즈) | 리그 동역학 | 손 모델 |

---

## 5. 작은 파운데이션 모델 카탈로그

### 5.0 공통 레시피 (T4와 스케일을 함께 고려한 설정)

- **처음부터(from scratch) 학습한다.** μP 파라미터화를 써서 30M에서 3B로 하이퍼파라미터를 옮길 수 있게 한다.
- **fp16 + dynamic loss scaling.** QK-norm을 쓰고 softmax는 fp32로 누적한다. SDPA mem-efficient 백엔드(sm75 지원)를 쓴다.
- **배포는 TensorRT.** critic과 셰이더는 INT8, flow/디퓨전 샘플러는 fp16이다.
- **교사 텍스트 임베딩은 사전계산해 캐시한다.** 텍스트 인코더는 상주하지 않는다.
- **디퓨전 계열**은 flow matching으로 학습하고 4-step으로 증류한다(consistency 또는 flow-map 계열, 학습 비용 +30%).
- **학습 FLOPs** = $6ND\times1.3$ (attention 오버헤드). 1 T4-h = $5.8\times10^{16}$.

### 5.1 카탈로그 표

| 모델 | 파라미터 | 아키텍처 | 입력 → 출력 | 데이터 (순도) | 학습 FLOPs | T4-h | 추론 (T4) | 학습/추론 메모리 |
|---|---|---|---|---|---|---|---|---|
| **EPD-S** (사건) | 28M | Transformer 12L, d384, self + cross attention | 사건 슬롯 $B{\times}256{\times}128$ + 문맥(세계 slot ≤128, 텍스트 토큰 ≤77) → 연속 속도장 + 이산 logits | 시뮬 사건 2–5M(D0), Gold Set(D0), 오디오 간격 주변분포(D1), 모캡 접촉(D0) | $6.6{\times}10^{17}$ | 11 (+4 증류) | 4-step 1개 약 2 ms, 1024개 약 2–5 s | 6 GB / <1 GB |
| **MoPri-S** (모션) | 32M | Transformer 10L, d448, 비AR 전체 시퀀스, cross-attn은 격층 | $x\in\mathbb R^{B\times T\times140}$, T≤300(30fps): 6D 관절 22개 + 루트 + 접촉. 조건: 텍스트, 사건 트랙 16ch, 스타일, 뼈 길이, 키프레임 마스크 → $\hat x_0$ | CMU, 100STYLE, 자체 캡처(D0). 연구용 AMASS. HumanML3D 텍스트(D2). 증강 ×10 | $1.9{\times}10^{18}$ | 33 (+10 증류) | 1개 약 5 ms, **best-of-4096 약 20–40 s** | 7 GB / 4 GB (배치 분할) |
| **Feas-S** (실현 가능성) | 5M | Transformer 6L, d256 | 사건 집합 + 세계 slot → P(SMC 성공), 예상 rejuvenation 비용 | 플라이휠 SMC 성패(D0, 공짜) | $5{\times}10^{16}$ | 1 | 마이크로초급 | <1 GB |
| **HeroProxy-S** | 14M | ViT 8L, d384, patch16 @256² | RGB + depth + normal + slot edge (7ch) → 대역별 DR 2개 + 16×16 DR 맵 | Teacher Exhaust ≥1M (교사 판정) | $1.5{\times}10^{17}$ | 3 (세대마다) | INT8 4096장 약 1.5 s | 4 GB / 0.2 GB |
| **ChronoProxy-S** | 12M | 같음 | 스트로보, 매트, 장노출 합성 → $\Delta_\chi$ | Exhaust (크로노) | $1{\times}10^{17}$ | 2 | 같음 | 같음 |
| **MAN-S** (셰이딩) | 1.5M | U-Net 3스케일 (32/64/128ch), 수용영역 16×16 | G-buffer 약 40ch (albedo, normal, rough/metal, depth, canonical 좌표, 상태 φ 11, 시선, 직접광 성분, **MAN 노이즈 ξ 8ch**) → 잔차 RGB, $\tanh\cdot\tau$ 제한 | 교사 DMD + R1-GAN(자체 실사 패치 D0) + 교사 VAE perceptual | TC 지배: 30k 스텝 × B4 × 8 TC-eq ≈ 96만 TC | **53** | 4K 1장 약 0.3 s(INT8), 1080p 약 0.1 s | 12 GB (교사 동거) / 0.3 GB |
| **PCGD-0** (확률 물성) | 약 300 (NN 아님) | 재질쌍 30개 × {반발계수 $e$~Beta, $\mu_s,\mu_k$~LogNormal, 구름 저항, compliance, spin coupling, OU 미세 토크 σ,τ} | 접촉 이벤트 → 확률적 파라미터 | Gold Set + 오디오 반발 분포. SMC-ABC(고전) | CPU | 0 | — | — |
| **F_φ-S** (2b 이후) | 4M | GNS식 message passing 10회, latent 128 + 확률 latent z8 | 입자 그래프 상태 → 잔차 가속도 | Gold Set lift + 시뮬(D0) | $8{\times}10^{17}$ | 15 | 스텝당 약 1 ms (1만 노드) | 3 GB / 0.5 GB |
| **Reader-S** (4종) | 26M | R-Flow 3M (RAFT-S식) / R-Kpt 18M (HR-lite heatmap) / R-Seg 4M / R-Event 1M (시간축 CNN) | 영상 → flow, 2D 키포인트, 동적 마스크, 사건 (유형, 시각) | 우리 렌더(정확한 GT, D0) + **BSRT** + COCO kpt(D2) + FlyingChairs/Things/Sintel(D0, 라이선스 확인) | 약 $6{\times}10^{18}$ | 100–115 (+BSRT 22) | 1080p 프레임당 약 30–80 ms | 8–10 GB / 2 GB |
| R-Lift | 0 | 고전 최적화: 삼각측량, BA, PnP, MoPri-S prior를 쓴 analysis-by-synthesis | 2D 트랙 → 3D 궤적 | — | CPU | 0 | — | — |
| **Denoiser-S** | 1M | OIDN식 U-Net, **처음부터 학습** | 저spp radiance + albedo + normal → 디노이즈 | 우리 렌더 쌍 (16 vs 1024 spp) 2만 쌍 | 렌더 22 + 학습 8 | 30 | 4K 약 0.2 s | 4 GB / 0.3 GB |
| *2c:* Expr-S | 12M | 시간축 Transformer | mel → ARKit식 52 블렌드셰이프 + viseme | 자체 캡처(§17 판결) | — | 10–20 | 실시간 | — |
| *2c 후반:* TTS-S + Voc-S | 20M + 14M | Matcha/VITS급 + Vocos급 | 텍스트 → 음성 | 라이선스 한국어 단일 화자 | — | 120–240 | 실시간 | — |

> **합계.** 2a–2b 시점 모델은 10개, 약 125M이다. 2c를 포함하면 약 170M이다.
> **1세대 전체 학습:** EPD-S 15, MoPri-S 43, Feas-S 1, Proxy 5, MAN-S 53, F_φ-S 15, Reader-S 115, Denoiser-S 30, Chrono-LoRA 10을 더하면 **약 287 T4-h**다. 2×T4로 약 6일이다. v1이 이 자리에 잡았던 수십만 H100-h 대신이다.
> **선례:** MDM(약 18M, CLIP은 동결해 별도), Learned Motion Matching(수 MB급 네트워크로 게임 캐릭터 구동), GNS(수 M 이하로 입자 물리 학습), OIDN(약 1M U-Net). **모션, 물리, 디노이징은 원래 작은 모델의 영역이다.** 비디오 모델은 외관과 의미까지 한 몸에 넣느라 커진 것이다.

### 5.2 모델별 핵심 설계

**EPD-S** [v1 EPD v2 축소]
- 사건 슬롯은 v1과 같다: $e_i=(b,t,\mathbf x,k,p,q,\mathbf m,\ell)$. 참여자 $p,q$는 세계 slot을 가리키는 pointer다.
- 스크립트 제약("t=8s에 잔이 깨진다")은 사건 집합 공간에서 슬롯을 고정하는 inpainting으로 준다.
- Feas-S로 classifier guidance를 건다.
- 실사건 데이터가 적은 부분은 두 가지로 보충한다. 범주별 사건 간격 분포를 오디오 간격 증인과 MMD로 맞추는 보조 손실을 건다. 라벨 등급(존재, 순서, 타이밍)에 따라 손실을 마스크한다.

**MoPri-S**는 3단계로 학습한다.
1. **A단계: 모캡 flow matching.** 33 T4-h.
2. **B단계: 물리 증강.** MJPC로 섭동 추종을 재시뮬해 데이터를 약 3배 늘린다.
3. **C단계: Photo-Slice RWR** [신규 조합]. 후보 모션의 임의 순간을 절차적 인체로 "스포츠 사진"처럼 렌더해 교사 DR을 매기고, 크로노 합성의 $\Delta_\chi$와 함께 보상 가중 회귀를 한다.
   - 모캡에 없는 범주(높이뛰기 배면, 브레이크댄스 프리즈)를 **사진으로 배운다.** 웹에는 동작 중 정지 사진이 수십억 장 있고, 교사는 그 분포를 안다.
   - 기울기 라우팅상 이것은 선택과 RWR일 뿐이고, 물리 검증(MJPC, RGF)을 통과한 샘플만 쓴다.
- **긴 모션**(분 단위)은 겹치는 창을 lockstep으로 동시에 디노이즈한다(시간 방향 MultiDiffusion). 확정된 창을 조건으로 다음 창을 만들지 않는다(§15).

**Band-Safe Reality Transfer (BSRT)** [조합/신규]. Reader-S가 실사 도메인으로 일반화하게 만드는 장치다.

$$
\sigma^*=D^{-1}\big(b_y{+}1,\,0.9\big),\qquad x_p=\mathrm{Dec}\Big(\mathcal D_{\sigma^*}\big(\mathrm{Enc}(x_r);\ \xi_{MAN}\big)\Big),\qquad \epsilon_y\le D(b_y,\sigma^*)\lesssim0.1
$$

- 렌더 $x_r$의 라벨(flow, 키포인트, 마스크 경계)은 대역 $b_y$ 이하에 있다. v1 Dreamer Calibration의 결정 곡선 $D(o,t)$로 **$b_y$보다 높은 대역만 다시 결정되는 σ**에서 SDEdit을 한다.
- 노이즈는 **물질에 고정한 MAN 노이즈**를 쓴다. 그래서 프레임 사이에서도 세탁이 일관되고, flow GT가 유효하게 남는다.
- 샘플마다 고전 검사(DIS flow로 $x_r\leftrightarrow x_p$ 정렬 확인)를 하고 통과하지 못하면 기각한다.
- 정리가 아니라 **보정에 기댄 근사 보장**이다.

**MAN-S** [v1 유지·축소]
- Read-Only Camera를 지킨다. 알파와 기하 출력이 없고 $\lVert r\rVert\le\tau$로 제한한다.
- 상태 조건부 외형 $\mathcal A_\Theta$는 입력 φ로 흡수한다.
- 모션블러 구간에서는 서브프레임 8개가 아니라 2–4개에만 MAN-S를 적용한다. 셰이딩 잔차는 천천히 변하기 때문이다.

---

## 6. T4 위의 물리

### 6.1 시뮬 처리량 (추정, P2-X0에서 확정)

| 시뮬 | 규모 | 장치 | 처리량 (추정) | 비고 |
|---|---|---|---|---|
| 강체 소형 (공, 병, 바디 ≤10) | batch 4096 | T4 (MJX/MuJoCo-Warp) | $3\times10^5$–$10^6$ scene-step/s | |
| 강체 중형 (도미노 50, 핀볼, 접촉 약 200) | batch 1024 | T4 | $2\times10^4$–$8\times10^4$ | |
| 〃 | 16코어 | **CPU MuJoCo** | $5\times10^4$–$2\times10^5$ | 결정론적이고 GPU를 쓰지 않는다. 중형에서는 CPU가 경쟁력 있다 |
| 강체 대형 (Rube Goldberg, 200–500 바디) | batch 256 | T4 / CPU | $5\times10^3$–$2\times10^4$ | |
| Voronoi 파편 300개 (유리, 도자기) | batch 64–256 | T4 | $1$–$3\times10^4$ | |
| 휴머노이드 (MJPC rollout) | batch 1024 | T4 / CPU | $1$–$3\times10^5$ / $3$–$10\times10^4$ | |
| MPM 300k 입자 (눈, 모래, 젤리) | 1 | T4 | 스텝당 5–10 ms (dt $10^{-4}$) → 시뮬 1초에 1–2분 | |
| FLIP 500k 입자, 256³ | 1 | T4 | 스텝당 60–150 ms → 시뮬 1초에 0.5–3분 | 압력 MGPCG가 시간을 지배한다 |
| Eulerian 연기 256³ | 1 | T4 | 스텝당 0.1–0.2 s → 시뮬 1초에 0.5–1분 | |
| XPBD 천 2만 정점 | batch 16 | T4 | 시뮬 1초에 2–10 s | |
| 헤어 1만 가닥 × 32 노드 | 1 | T4 | 시뮬 1초에 10–30 s | |

> **유리는 MPM으로 하지 않는다.** 유리는 $E\approx70$ GPa, $\rho\approx2500$이라 파동속도가 $c\approx5.3$ km/s다. $dx=1$ mm면 CFL 조건상 $dt\lesssim2\times10^{-7}$ s이고, 0.5초에 260만 스텝이 필요하다. H100에서도 낭비다. VFX 관행대로 **Voronoi 사전 파쇄(seed = 사건 마크) + 강체**로 한다. 결정론적이고 싸다.

### 6.2 Milestone SMC 입자 수 [v1 유지·조정]

| 과제 | N | 지평 | dt | scene-step (재시뮬 ×1.5) | T4 시간 | CPU16 시간 |
|---|---|---|---|---|---|---|
| 공, 병 (BUPS 소형) | **4096** | 3 s | 1 ms | $1.8\times10^7$ | 20–60 s | 2–6분 |
| 도미노 50, 핀볼 | **1024** | 10 s | 1 ms | $1.5\times10^7$ | 3–12분 | 1.5–5분 |
| 미니 Rube Goldberg | **256** (Chaos Bank warm start) | 15 s | 1 ms | $5.8\times10^6$ | 5–19분 | 4–12분 |
| 와인잔 파괴 (Frozen Action) | 강체 256 → FLIP top-8 | 0.5 s | 0.5 ms / 0.2 ms | 강체 $3.8\times10^5$ + FLIP 8×2,500 스텝 | 강체 1분 미만 + FLIP 10–25분 | — |
| 캐릭터 접촉창 (MJPC) | 제어 스텝당 rollout 512개, 예측 0.4 s, 50 Hz | 10 s 모션 (접촉 30%) | 2 ms | 약 $1.5\times10^7$ | 1.5–6분 | 5–17분 |

v1의 "N=4096, $4\times10^7$ env-step, 분 단위"는 **소형 장면에서만** 그대로 유지된다. 중형 이상은 N을 줄이고, 그 손실을 Feas-S가 개선한 제안과 Chaos Bank warm start로 메운다.

### 6.3 Multi-fidelity SMC (유체, MPM)

1. 거시 결정 변수(충돌 속도와 각도, 채운 양, 파쇄 seed, 바람)는 1/8 해상도 프록시(3만 입자, 약 30배 저렴)와 proxy critic으로 고른다.
2. top-8만 원 해상도로 다시 시뮬하고, 미시 seed만 바꾼다.
3. 사건 시각 제약(충돌 시각 ±1프레임)은 강체 운동이 정하므로 해상도가 바뀌어도 거의 유지된다.
4. 미세한 혼돈은 **Hero-Instant의 다양성**으로 쓴다. 혼돈을 레버로 삼는 것이다(v1 유지).

### 6.4 Chaos Bank [조합; 부품: Many-Worlds Browsing'07, motion matching]

- 밤이나 학습 중 놀고 있는 CPU로 세트피스 템플릿별 rollout을 $10^5$–$10^6$개 미리 굴린다. 템플릿 예: 도미노 줄, 유리 파괴, 병, 계단의 공, 젠가.
- 사건 그래프 시그니처와 타이밍 벡터를 kd-tree로 색인한다.
- 요청이 오면 top-N을 검색해서 SMC 초기 입자로 쓴다. 스텝을 5–20배 줄인다.
- CPU MuJoCo는 결정론적이다. 그래서 rollout을 (템플릿, 파라미터, seed, sim 버전) 약 64 B로 저장할 수 있다. $10^6$개면 64 MB다.
- 같은 뱅크가 EPD-S와 Feas-S의 학습 데이터도 된다.

### 6.5 캐릭터 물리: MJPC predictive sampling [부품, v1 PHC를 미룸]

- 범용 추종 정책(PHC급)은 $10^9$–$10^{10}$ 샘플이 필요하다. T4 한 장으로 1–3주가 걸린다.
- 대신 **MuJoCo MPC식 predictive sampling**(학습 0)을 **접촉창에서만** 쓴다. 이것이 PhysDiff식 사영이다.
- 잔여력은 RGF 예산 안으로 제한한다.
- 학습 정책은 컴퓨트가 늘어나는 2d로 미룬다.

---

## 7. 16 GB × 2 배치

| 모드 | GPU-A "판정·렌더" | GPU-B "학습·시뮬" | CPU (≥16 vCPU, 64 GB) |
|---|---|---|---|
| **F: 플라이휠의 하루** | 교사 UNet fp16 5.2 + VAE 0.2 + fake-score LoRA·Adam 0.5 + 활성(512², B4, checkpointing) 4–5 + 래스터·BSRT 버퍼 1.5 = **12–13 GB** | 학생 1개 학습(≤40M: 가중치·Adam 0.6 + 활성 5–7) + MJX/Warp 배치 1–2 + INT8 proxy 엔진 0.3 + proxy용 소형 래스터 0.5 = **9–11 GB** | MuJoCo 16 프로세스 약 4 GB, GLOMAP/COLMAP 8–16 GB, Block Bank 샤드, Exhaust 로그 |
| **H: 히어로 샷** | 교사 + 슬라이스 polish. 1024²는 14–15 GB로 빠듯하므로 512² band-iso 뷰나 768² 타일로 돌린다 | FLIP 500k + 256³ 약 1.5 GB, 3DGRT BVH(3M Gaussians) 3–4 GB, MAN-S·Denoiser-S 0.3, 4K AOV 1 GB = **7–9 GB** | 모달 FEM 해석, 오디오 |
| **T: 학습 버스트 (DDP 2-way)** | 같은 모델 복제 8–12 GB | 같음 | 데이터 로더 |

- NVLink가 없다. PCIe Gen3에서 32M 모델의 fp16 gradient 64 MB를 allreduce하면 약 10–25 ms다. 스텝이 300–500 ms이므로 확장 효율은 약 95%다.
- **기본은 DDP가 아니라 역할 분담(A/B)**이다. 두 카드 사이로는 작은 데이터(latent, 상태, 점수)만 오간다. 모델 병렬은 쓰지 않는다.
- 텍스트 인코더(CLIP-L + bigG, fp16 1.6 GB)는 임베딩을 미리 계산한 뒤 내린다.
- **Kaggle 모드:** GPU 구성은 같다. CPU가 4 vCPU뿐이므로 SMC는 GPU로 하고, lift와 CPU 시뮬은 Kaggle 밖의 저가 CPU 머신이나 노트북으로 뺀다.

---

## 8. 세계와 외관

### 8.1 Reveal-Bundle Demand: 사건이 드러낼 면만 닫는다 [신규 조합, v1 Edit-Closed 축소]

$$
\mathcal R=\bigcup_{\tau}\mathrm{Vis}\big(W(t(\tau)),\Gamma(\tau)\big)\setminus\mathcal V_0,\qquad
\mathcal L_{EC}^{T4}=\mathbb E_{e\in E_{real},\ c:\ \mathrm{Vis}(c)\cap\mathcal R\neq\varnothing}\,\mathrm{KL}\big(p_{\mathcal R}(\cdot\mid e,c)\,\Vert\,p^{img}(\cdot\mid y_e)\big)
$$

- v1은 모든 slot과 광원에 대해 반사실 편집을 30–50% 섞었다.
- 우리는 **실현된 사건 집합 $E$에 실제로 나오는 편집**과 **세계선 원뿔 안에서 드러나는 영역 $\mathcal R$**에만 반사실 TC를 쓴다. Lens-Gated Ray-Bundle Demand를 4D로 확장한 것이다.
- 탁상 장면에서 slot이 50개이고 사건이 그중 1–5개를 건드린다면 TC가 10–50배 줄어든다.
- 비회귀 조건: $\mathcal R$의 거친 레벨은 이미 존재한다(bedrock은 내부와 뒷면을 포함한다). $\mathcal R$은 같은 레벨의 다른 영역과 같은 lockstep 라운드에서 풀린다.
- 새 테이크가 새로운 면을 드러내면 copy-on-write 브랜치에서 그 면의 미세 대역만 연다. v1의 Frozen-Seed Band-Commit을 그대로 쓴다.
- 파괴 단면은 절차적 볼륨 재질(고전)로 만들고, 히어로 컷만 교사로 polish한다.

### 8.2 캐릭터 외관 [v1 Avatar Foundry v2 축소]

- 기반 메시는 MakeHuman류 절차적 인체(CC0)다. SMPL-X는 판결 전까지 시스템 경로에서 쓰지 않는다(§17).
- identity LoRA(r8, 약 6M)로 분포 매칭을 한다. **캐릭터당 약 5 T4-h**이고 한 번만 든다.
- 얼굴: v0는 블렌드셰이프와 음소→viseme 규칙, v1은 Expr-S다. 눈은 고전 각막 굴절 모델로 만든다.

### 8.3 음향: 학습 파라미터 0개 [v1 유지]

| 소리 | 방법 |
|---|---|
| 충돌, 타격 | 사건 충격량 $\mathbf J$로 모달 합성. 모드는 사면체 메시 FEM 고유값 해석으로 얻는다(CPU, 에셋당 수 초–수 분) |
| 파괴 | 확률적 임펄스 |
| 액체 | Minnaert 기포 확률 합성. Harmonic Fluids를 축소한 것이다 |
| 전파 | 이미지 소스 + **RT 코어 음선 추적** |
| 바이노럴 | 측정 HRTF (데이터) |
| 음성 | 2c 초반에는 녹음한 목소리, 후반에는 TTS-S |

---

## 9. 로드맵 2a–2d 재산정

이 표의 전제: Phase 1은 병렬로 진행한다. Phase 2의 GPU 몫은 M0–3에 10–15%, M3–9에 30–50%, M9 이후 50% 이상이다.

| 단계 | v1 시점 (M0 환산) | **2×T4 시점** | 새로 학습하는 부품 | **컴퓨트 병목** | 우회 | 샷당 증분 비용 (세계 제외) | 승리 조건 (v1 그대로) |
|---|---|---|---|---|---|---|---|
| **2a Frozen Action & Speed Ramp** | 약 M6–M9 | **M3–M9** (시드 M0–M3) | MAN-S, Hero·ChronoProxy, Denoiser-S, Chrono-LoRA | (1) 정지 슬라이스 polish의 TC. (2) 유리와 액체의 RT 렌더. (3) **아레나 콘텐츠 300개 생산**: 약 1,000–1,500 T4-h | Reveal-Bundle Demand, Judge Cascade, Voronoi 강체, multi-fidelity FLIP, RT 코어 + Denoiser-S. **세계 30개 × 사건 10개 재사용**("세계는 한 번, 사건은 무한"). 아레나는 1080p 정규화 | **3–6 T4-h**. 재테이크는 0.5–1.5 | BUPS-FA에서 각 경쟁자 대비 ≥70%, 정지 슬라이스 Appearance Turing ≤65%, GeCo ≤0.03, 램프 TSR 실제 밴드 안, RST ≤1프레임 |
| **2b 강체·관절체 + 살아있는 세계** | 약 M9–M15 | **M8–M16** | PCGD-0 → F_φ-S, EPD-S v1, Feas-S, Reader-S v1 | (1) sim look 보정용 실데이터. (2) 볼륨과 액체 렌더. (3) Physics-IQ는 실영상 조건이라 lift 품질에 의존 | Garage Gold Set ×3, 오디오 반발 분포, Chaos Bank. 절차적 생명(Stam 바람, Tessendorf 바다, Eulerian 연기, modal 식생). 실영상 조건은 Reader-S + 교사 DPS 관측 제약으로 세계화 | 2–4 T4-h | Physics-IQ, VideoPhy-2 1위, Motion Turing ≤60%, CSI KL ≤ 최강 경쟁자 ×0.5, BUPS ≥65%, 일반 Arena ≥50%, GCM ±20% |
| **2c 캐릭터·연기** | 약 M15–M21 | **M14–M28** | MoPri-S v2(C단계), Expr-S, identity LoRA, R-Kpt v2 | **데이터**(모션, 얼굴, 연기). 천과 헤어 렌더. 캐릭터당 TC | **Act, don't prompt**: 폰 연기를 Reader-S로 lift해 MoPri-S prior와 MJPC를 거쳐 재촬영한다. best-of-N, 블렌드셰이프, 녹음 음성, Coverage Compiler(고전). Garage 모캡으로 연간 20–50시간 | 4–8 T4-h + 캐릭터당 5 | 연기 축 ≥55%, 멀티숏 연속성 ≥70%, 얼굴 Appearance Turing ≤75%, foot skate <2 cm/s, 관통 <1%, eyeline <3° |
| **2d 완전 4D + 인터랙션** | 약 M27–M39 | **M26–M42+** | 손 조작 (MJPC 기반 → 학습 정책), 다중 행위자 EPD-S, 미리보기 emulator | 손-물체 상호작용, 에셋 다양성, 분 단위 블록(시뮬과 저장) | 시간 방향 Merkle 청크 스트리밍 블록, MJPC 손. §13의 스케일 다이얼 | — | 모든 범주 Arena ≥60%, Revisit 시간 기울기 ≈0, 인과 정확성 1위 |

**2a와 2b가 거의 밀리지 않는 이유.** 이 두 단계의 비용은 시뮬, 렌더, TC인데, T4가 상대적으로 덜 손해 보는 항목이다. 사전학습은 처음부터 필요 없었다.

**2c가 밀리는 이유.** FLOPs가 아니라 **사람 데이터**가 병목이기 때문이다. 우회는 "연기는 생성하지 말고 찍어라"다. 프로젝트 슬로건을 연기에도 그대로 적용한 것이다.

**범주 정복 순서.** 각 범주에서 BUPS ≥ 목표를 먼저 달성한 뒤 일반 Arena로 넓힌다.

| 순서 | 범주 |
|---|---|
| 1 | Frozen Action (유리, 물 왕관) |
| 2 | 강체 연쇄 (도미노, Rube Goldberg) |
| 3 | 파괴 |
| 4 | 액체 hero |
| 5 | 식생과 바람 |
| 6 | 운동선수 단독 동작 |
| 7 | 댄스 |
| 8 | 2인 대화 |
| 9 | 손 조작 |

Kaggle 모드에서는 2a를 M12–M18에 노린다. 2b 이후는 상시 2×T4가 있어야 현실적이다.

---

## 10. 플라이휠: 큰 사전학습 없이 도는 자기개선 루프

```
 ┌─(1) 제안: ShotScript 템플릿 × 절차 세트피스 × EPD-S/MoPri-S (Feas-S guidance, Chaos Bank 검색)
 │    ▼
 │ (2) 실현: Milestone SMC (CPU/GPU), PCGD-0, MJPC        ← 학습 0, 정확한 물리
 │    ▼
 │ (3) 판정 캐스케이드: 불변량 → RGF → INT8 proxy → TC(슬라이스 + 크로노) [10%는 무작위 후보]
 │    ▼
 │ (4) 커밋: Block Bank (Merkle) + Teacher Exhaust 로그
 │    ▼
 │ (5) 증류: EPD-S·MoPri-S(RWR), Feas-S(성패), Proxy-S(Exhaust), Reader-S(렌더 + BSRT), MAN-S
 │    ▼
 │ (6) 읽기: Reader-S로 실영상 Event-Demand lift → p_real, 사건 통계, PCGD-0 재보정
 └─(7) 가속: 더 좋은 제안 → SMC N↓ / 더 좋은 proxy → 블록당 TC↓ / 같은 하루에 블록↑
```

이것은 AlphaZero 구조와 대응한다. 탐색은 SMC와 캐스케이드, 정책은 EPD-S와 MoPri-S, 가치는 Feas-S와 proxy, 심판은 물리, 교사, 실통계다.

### 10.1 1회전 시간표 (상시 2×T4 + 16 vCPU, Phase 2 몫 약 2/3, 1회전 = 블록 1,000개)

| 단계 | 장치 | 작업량 | 시간 |
|---|---|---|---|
| 제안 | GPU-B | 장면 1,000개 × 256 제안 (4-step) | 약 1 h |
| 실현 | CPU 60%, GPU-B 40% | 블록당 평균 약 80 s 상당 (소 60%, 중 30%, 대 10%) | CPU 약 13 h, GPU-B 약 9 h |
| 판정 L2 | GPU-B INT8 | 256² 렌더 약 25만 장 + proxy | 약 0.3 h |
| 판정 L3 | GPU-A | 1,000 × 128 TC = 12.8만 TC | 약 7 h |
| 데이터용 렌더 | GPU-A | 1,000 × 64프레임 512² + AOV | 약 0.5 h |
| BSRT | GPU-A | 이미지 1만 장 × 8 TC | 약 4.5 h |
| 증류 (증분 재학습) | GPU-B | EPD-S 2, Feas-S 0.5, Proxy 2, Reader-S 4, MoPri-S RWR 2 | 약 10 h |
| 실영상 lift | CPU | 클립 300개 (GLOMAP + Reader-S + R-Lift) | CPU 약 10 h (병렬) |
| **합계** | | GPU-A 약 12 h, GPU-B 약 20 h, CPU 약 23 h | **약 1일/회전** |

- 시드기(Phase 2 몫 10–15%)에는 4–6일에 1회전이다.
- Kaggle 모드에서는 세션 하나가 회전의 한 조각이다. 체크포인트와 Block Bank 샤드는 Kaggle Dataset에 둔다(용량 한도는 확인 필요). **주 1회전, 규모 1/3**이다. 세션을 세대 경계와 맞추는 이 운영 방식을 "세션 = 세대"라고 부른다.

### 10.2 래칫 지표

| 지표 | 1회전 | 10회전 목표 |
|---|---|---|
| SMC 입자당 제약 충족률 | 0.5–2% | 5–15% |
| 블록당 L3 TC | 128 | 16–32 |
| **하루 블록 생산량** | 1,000 | **8,000** |
| proxy와 교사의 Spearman 상관 | 약 0.5 | ≥0.85 |
| Reader-S Gold Set 오차 (PCK, EPE) | 기준선 | −30% |
| search gain (SMC 최선 − 학생 단일 샘플) | 큼 | 단조 감소. 이것이 래칫의 증거다 |

### 10.3 붕괴 방지

- 배치마다 **실데이터 앵커를 30% 이상** 둔다: Gold Set, 모캡, 오디오 통계, COCO.
- **held-out Gold 게이트**: 2% 넘게 퇴행하면 그 세대를 롤백한다.
- 사건 그래프 엔트로피와 범주 커버리지를 감시한다.
- L3 호출의 10%는 무작위 후보에 쓴다. proxy echo chamber를 막기 위해서다.
- Exhaust 로그에는 교사 버전 태그를 붙인다.

---

## 11. v1 장부: 버리는 것, 미루는 것, 줄이는 것, 유지하는 것

| v1 항목 | 판정 | 이유 | 대체 |
|---|---|---|---|
| Native WFM (정적 4–8B, 사건 0.5–2B, F_φ 0.3–1B, 셰이더 0.2–0.5B, 정책 0.1–1B), T1–T3 | **폐기** | $10^{23}$–$10^{25}$ FLOPs. 우리 1년이 $10^{21}$ | Tiny Federation(약 125M) + 테스트타임 탐색 |
| PLAC, 비디오 증인(Wan 2.x), PC-ELBO, 진술 그래프(비디오 제안) | **폐기 (규칙)** | 비디오 모델 금지 | 크로노포토 증인(PC 대조와 QBC를 이미지판으로 재활용), 실제 증인, 감독 |
| Spectral Division (f_c 실측) | 유지 (측정용) | 경쟁 모델 결손 측정과 크로노 대역 설계에 필요 | 고전 추적(KLT, RANSAC 자기운동 보상)으로 측정 |
| Lift 스택 (DA3, VGGT-Ω, π3, MonST3R, GVHMR, WHAM, FoundationPose, MoSca, SAM2, CoTracker3) | **폐기** | 사전학습 모델 | 고전 SfM, flow, 배경차분 + Reader-S (BSRT) |
| 정적 5천만, 동적 2천만 클립 lift | **축소** → 연 1e4–1e5 | 컴퓨트 | Event-Demand Lift, 오디오 간격 증인 |
| Gold Set (1000fps 다시점 + LiDAR 50시간, $0.3–0.6M) | **축소, 미룸** | 예산 | Garage Gold Set ($0–2k). 1000fps는 2b 이후 대여 |
| AV-Physics Timing Solver, Blur-as-Measurement | 유지 | 고전 | 오디오 간격 증인 추가 |
| EPD v2 + Event Type Contract | 유지·축소 | — | EPD-S 28M |
| INTENT/EMOTION의 VLM 판정 | **폐기** | VLM 금지 | 사람 critic + 자체 라벨로 학습한 소형 분류기 |
| PCGD (F_φ 0.3–1B) | 축소 | — | PCGD-0(약 300 파라미터) → F_φ-S 4M |
| RGF + Reality Residual Budget | 유지 | 고전 통계 | — |
| Milestone SMC + Hero-Instant | 유지·조정 | — | N을 장면 크기별로, CPU 경로, multi-fidelity, Chaos Bank, Judge Cascade |
| Edit-Closed World (모든 편집 30–50%) | 축소 | TC | Reveal-Bundle Demand |
| 상태 조건부 외형 + MAN, slice DM | 유지·축소 | — | MAN-S 1.5M |
| Avatar Foundry v2 | 유지 | — | identity LoRA + 절차적 인체 |
| SMPL-X, FLAME | **미룸 (판결 필요)** | 학습된 통계 모델 + 비상업 라이선스 | 자체 스켈레톤(BVH식 22관절) + 절차적 인체 + 고전 리타기팅 |
| PHC, MaskedMimic | **미룸 (2d)** | RL 샘플 $10^{9}$–$10^{10}$ | MJPC predictive sampling |
| Codec-Avatar급 얼굴 latent 256–512 | **미룸** | 데이터 | 블렌드셰이프 + Expr-S |
| 사전학습 TTS | **폐기 → 미룸** | 금지 | 녹음 음성 → TTS-S |
| Harmonic Fluids | 축소 | 비용 | Minnaert 확률 합성 |
| Single-cause audio, Coverage Compiler, 테이크, 분기 우주, 재조합 엔진 | 유지 | 학습 0 | — |
| Temporal DoF, 시간 PSH, Gabor 물결 | 유지, **앞당김** | 학습 0, 렌더 측에서만 처리해 싸다 | 2a·2b 쇼케이스 |
| Builder–Reader self-play | **유지·중심화** | Reader를 직접 만들어야 하므로 | BSRT |
| Factorized Turing, BUPS Arena | 유지 | 비용이 GPU가 아니라 돈(사람 평가) | — |
| GeCo, MEt3R, VBench, Physics-IQ 등 | 평가 전용 | 규칙 | — |
| H1′ go/no-go ($10^{20}$–$10^{22}$) | **재정의** | $10^{22}$ 불가 | **H1-T4**: $10^{18}$–$10^{20}$ 구간에서 등사실감 지점의 우리 총 FLOPs를 경쟁 모델 공개 추정치와 비교해 기울기를 잰다 |

---

## 12. 이 제약 덕분에 오히려 좋아지는 점

1. **반복 속도.** 30M 모델은 몇 시간 만에 학습이 끝난다. 주당 수십 번 실험할 수 있다. 8B 모델을 한 번 학습하는 동안 우리는 200번 돈다.
2. **인수분해가 필수가 된다.** 실패를 어느 부품 탓인지 명확히 가를 수 있다. Factorized Turing Battery와 1:1로 대응한다.
3. **테스트타임 탐색이 싸진다.** Coverage–Precision 경제학이 처음으로 성립한다.
4. **배포가 쉬워진다.** 교사를 뺀 Phase 2 추론 스택 전체가 소비자 GPU와 노트북에서 돈다. 실시간 미리보기와 감독용 온디바이스 툴이 가능하다.
5. **학습 0 부품은 감가상각이 없다.** 물리, 음향, 광학, 연출은 컴퓨트가 1000배로 늘어도 재학습할 필요가 없다.
6. **데이터 자산이 주인공이 된다.** Block Bank, Gold Set, Teacher Exhaust는 모델과 독립된 해자다.
7. **의존성이 단순해진다.** 외부 파운데이션 모델은 교사 하나뿐이라 법무가 깔끔하다.
8. **전기요금.** 2 × 70W = 140W다. 전기장판 두 장 수준이고, 우리 GPU 클러스터는 냉장고 옆 콘센트 하나로 돈다. H100 64장 계획서는 액자에 걸어 둔다.

---

## 13. 컴퓨트가 늘어날 때 그대로 스케일되기 위한 원칙

- **S1. 텐서 계약을 고정한다.** 각 연방 부품은 (입력 텐서, 출력 텐서, 판사 라우팅)으로 정의한다. 모델 크기는 구현 디테일이다.
- **S2. μP + 같은 데이터 파이프라인.** 30M에서 찾은 하이퍼파라미터가 3B로 넘어간다. 스케일링 곡선을 소규모에서 미리 잰다.
- **S3. 탐색과 상각 사이의 다이얼.** 테스트타임 예산(N, 블록당 TC)과 학생 크기를 같은 Pareto 곡선 위에서 고른다. 컴퓨트가 늘면 먼저 학생을 키우고 N을 줄인다. 래칫과 같은 방향이다.
- **S4. 데이터 포맷은 모델과 독립이다.** Merkle 블록과 Exhaust 로그(입력 해시, σ, 교사 버전, 점수)를 쓴다.
- **S5. 교사는 Dreamer 계약 슬롯에 끼운다.** 더 큰 이미지 모델로 바꿔도 인터페이스는 같다. Exhaust는 버전 태그로 다시 보정한다.
- **S6. 고전 부품은 정확도 다이얼만 돌린다.** 시뮬 해상도, spp, SMC N.
- **S7. 수치.** fp16-safe(QK-norm, loss scaling)로 짠 코드는 bf16, fp8에서 그대로 돈다. T4 전용 트릭(INT8 critic)은 배포 계층에만 둔다.
- **S8. 스케일 순서.**

| 컴퓨트 | 무엇을 키우는가 |
|---|---|
| ×10 | 교사 1024² 판정, MoPri 300M, SMC N ×10 |
| ×100 | Phase 1 세계 학생 1B, EPD 300M, Reader 300M (여전히 우리 모델) |
| ×1000 | v1 WFM T1 경로 |

전 구간에서 비회귀 조항과 불변량 I1–I6은 그대로다.

---

## 14. 첫 90일: Phase 2 시드 실험

예산은 약 430 T4-h로 전체 4,320의 약 10%이고, CPU는 대부분 Phase 2가 쓴다.

| ID | 주 | 가설과 셋업 | 성공 기준 | 실패하면 (업그레이드) | 예산 |
|---|---|---|---|---|---|
| **P2-X0** T4 벤치 | 1–2 | CPU MuJoCo, MJX, Warp/Newton(sm75 지원 확인), Taichi MPM/FLIP, TC 지연(512/1024), 래스터와 RT fps, INT8 처리량 | 이 문서의 "(추정)"을 모두 실측값으로 바꾼다 | sm75 미지원 라이브러리는 CPU나 Taichi로 대체 | 10 T4-h |
| **P2-X1** Chrono-Cal | 2–6 | 6범주(바운스, 투척, 진자, 낙하, 점프, 회전) 실제 합성 2천 장 vs 왜곡 5종. 교사 DR 대조. 사람 2AFC 기준선 | 시간 역전, g 스케일, 평활 중 2종 이상에서 AUC ≥0.75 (LoRA 허용). held-out 범주에서도 ≥0.7 | Chrono-LoRA 확대. 그래도 안 되면 미감 전용으로 강등하고 H7 경로 | 30 T4-h |
| **P2-X2** Garage Gold + PCGD-0 + RGF | 2–8 | 폰 3–4대 240fps, 6시간, 물체 20종 × 사건 6종. 웹 오디오 5천 클립(반발 분포) | 외관을 고정한 Motion Turing(S vs R) ≤60%. RGF AUC: 실제 vs 순진한 시뮬 ≥0.75, 실제 vs PCGD-0 ≤0.6 | 캡처 3배, F_φ-S 앞당김 | 10 T4-h + CPU |
| **P2-X3** SMC on T4 | 3–9 | 과제 20개(도미노, 핀볼, 유리) × 제약 5세트, N ∈ {64, 256, 1024, 4096}, CPU vs GPU, Chaos Bank 유무 | 충족률 ≥90%, T4 10분 이하 또는 CPU 20분 이하. "조종된 rollout" 식별률 ≤55% | 허용오차 창 학습, 잔여력 예산 확대, Feas-S guidance | 20 T4-h |
| **P2-X4** Coverage–Precision | 4–10 | MoPri-S 학습. best-of-{1, 16, 256, 4096} + MJPC 검증 | best-of-256 + 검증이 단일 샘플보다 2AFC에서 15%p 이상 앞섬. 보행 Motion Turing ≤60% | MoPri 64–80M(여전히 수십M), Photo-Slice RWR 앞당김 | 60 T4-h |
| **P2-X5** Reader-S × BSRT | 6–12 | R-Flow, R-Kpt를 처음부터 학습. BSRT on/off 비교 | Gold Set과 Sintel에서 R-Flow가 DIS보다 EPE 30% 이상 낮음. R-Kpt COCO AP ≥65. BSRT로 실사 오차 −15% 이상 | 렌더 다양성 확대, σ* 재보정, Gold 라벨로 미세조정 | 120 T4-h |
| **P2-X6** Frozen Action 1샷 | 8–13 | 와인잔: Reveal-Bundle EC, SMC, Hero-Instant(크로노 포함), 스피드 램프, 모달 오디오, 에저턴 스틸 | 미니 BUPS-FA에서 Seedance, Kling, Veo(API, 평가 전용) 대비 ≥60% (2a 출시 목표는 70%). 정지 궤도 GeCo ≤0.03 (GeCo는 평가 전용) | 액체는 메시 표면 path tracing + MAN-S에 굴절 G-buffer 추가 | 60 T4-h |
| **P2-X7** Spectrum Report v0 | 1–6 | 경쟁 클립의 모션 PSD 결손을 고전 추적으로 측정 | v1 X0 기준(4개 중 3개 이상에서 결손) | 축을 질량 시그니처와 CSI로 옮김 | 5 T4-h + API $2–5k |
| (병행) MAN-S, Denoiser-S | 6–13 | 2a 준비 | Appearance arm +15%p | 용량과 수용영역 확대 | 83 T4-h |

**90일 게이트.** X1, X3, X6 중 둘 이상을 통과하면 2a 본진행(M3)으로 간다. 실패한 실험은 그 실험의 업그레이드를 다음 분기 1순위로 올린다. 목표는 낮추지 않는다.

**Kaggle 모드.** X0, X1, X3 소형, X4만 진행한다. 나머지는 미룬다.

**돈.** 폰 $0–2k, 스트로보 LED와 조명 $100–300, 인쇄 보드, API 평가 $2–5k, 사람 평가 $5–15k.

---

## 15. 비회귀 점검 (순차 아웃페인팅 금지)

| 부품 | 판정 | 근거 |
|---|---|---|
| EPD-S | 통과 | 사건 **집합** 디퓨전이다. 슬롯은 병렬이고 스크립트 제약은 집합 공간 inpainting이다. 픽셀 캔버스가 없다 |
| MoPri-S (긴 모션) | 통과 | 겹치는 창을 lockstep으로 동시에 디노이즈한다. 확정된 창을 조건으로 쓰지 않는다. 창 처리 순서를 바꿔도 결과가 같은지 `order_purity_check`로 OIE를 검사한다 |
| 실현 (시뮬) | 통과 | 인과적인 것은 **상태공간 ODE 적분**이고 픽셀 생성이 아니다. 세계는 원래 그렇게 돈다 |
| Judge Cascade, 크로노 증인 | 통과 | 판정은 **읽기**다. 후보 자신의 렌더로 고를 뿐, 다른 영역 생성의 조건으로 쓰지 않는다 |
| Reveal-Bundle Demand | 통과 | 세계 공간에서 처리하고, 거친 레벨이 이미 존재한다. 같은 레벨 안에서 lockstep으로 풀고, 렌더 픽셀을 조건으로 쓰지 않는다 |
| BSRT | 통과 | Reader 학습 데이터 전용이다. 렌더 경로에 없다 |
| MAN-S, Denoiser-S | 통과 (I6) | 결정론적이고 알파·기하 출력이 없다. 노이즈는물질에 고정된 MAN 노이즈만 쓰므로 프레임 단위 latent가 없다. Denoiser-S는 64 spp 이상의 입력에 잔차 크기 제한을 걸어 콘텐츠를 지어내지 못하게 한다 |
| Chaos Bank 검색 | 통과 | 상태공간 초기 입자로만 쓰고 픽셀은 쓰지 않는다 |

---

## 16. 리스크와 업그레이드 (후퇴 없음)

| # | 리스크 | 감지 지표 | 업그레이드 |
|---|---|---|---|
| R1 | 크로노 증인이 시간 구조에 둔감하다 | P2-X1 AUC < 0.65 | Chrono-LoRA 확대, BIS의 (K, Δ) 재설계, 라이트 페인팅 모드. 그래도 안 되면 미감 전용으로 강등하고 RGF와 실통계로 간다(H7) |
| R2 | sim look이 남는다 | Motion Turing > 65% | Gold Set 캡처 ×3–10, PCGD-0을 F_φ-S로, OU 미세 토크 보정, 1000fps 대여 |
| R3 | MoPri-S의 다양성과 품질이 부족하다 | foot skate, 2AFC, 범주 커버리지 | best-of-N 확대, Photo-Slice RWR, MoPri를 64–80M으로, Act-don't-prompt 비중 확대 |
| R4 | Reader-S가 웹 영상을 못 읽는다 | Gold Set 오차, lift 기각률 | 실제 증인을 Gold Set, 오디오 간격, 모캡으로 좁힌다. BSRT σ* 재보정. Event-Demand Lift를 쉬운 범주부터 |
| R5 | TC 예산이 초과된다 | 블록당 TC가 줄지 않음 | proxy 재학습 주기 단축, 판정 해상도 하향, 교사를 SANA급으로(Phase 1 렌즈와 공동 결정) |
| R6 | 라이선스 (AMASS, SMPL 비상업, 데이터셋 지역 조항) | 법무 게이트 | 상업 경로는 CMU, 100STYLE(확인), 자체 캡처로만 |
| R7 | 라이브러리가 sm75를 지원하지 않는다 (Warp, Newton, MJX) | P2-X0 | CPU MuJoCo, Taichi로 대체 |
| R8 | Kaggle 쿼터와 4 vCPU | 회전 주기 | CPU 작업을 외부 저가 머신으로, 범위를 2a로 한정 |
| R9 | fp16 학습 불안정 | loss scale 붕괴, NaN | QK-norm, μP, fp32 softmax와 LayerNorm, gradient clipping |
| R10 | 플라이휠 붕괴 (자기증류 echo chamber) | Gold 게이트 퇴행, 엔트로피 하락 | 실데이터 앵커 30% 이상, 무작위 L3 10%, 세대 롤백 |
| R11 | 아레나 콘텐츠 생산이 병목이 된다 | 2a 일정 | 세계 30개 × 사건 10개 재사용, 1080p 정규화, 캐스케이드 |
| R12 | 일반 T2V 의미 범위(아무 프롬프트)에서 진다 | 일반 Arena | 범주 정복 순서를 지킨다. 감독 스크립트와 Act-don't-prompt를 쓴다. §13 스케일 다이얼. 목표는 낮추지 않고 순서만 정한다 |

---

## 17. 창업자가 결정할 질문

1. **D3 데이터.** 사전학습 모델이 라벨을 만든 공개 데이터셋(SpatialVID의 pose와 depth, Motion-X, DataDoP)을 학습에 써도 되는가? 기본값은 금지다. 평가와 분석에만 쓴다.
2. **SMPL-X와 FLAME.** 학습된 통계 바디 모델을 "에셋"으로 볼 것인가, 금지된 "사전학습 모델"로 볼 것인가? 비상업 라이선스 문제도 있다. 기본값은 자체 스켈레톤과 절차적 인체다.
3. **교사 텍스트 인코더를 작은 모델들의 조건 입력으로 재사용**하는 것을 허용 범위로 해석해도 되는가(Lingua Franca)?
4. **캡처 기기의 내장 ML.** 예를 들어 폰 ARKit 얼굴 추적을 "센서"로 보고 데이터 수집에 써도 되는가?
5. **2c 제품 방향.** "Act, don't prompt", 즉 연기와 음성은 사용자가 찍는 방식을 기본으로 할 것인가?
6. **Phase 2의 GPU 몫.** 기간별로 15%, 40%, 50% 이상으로 가는 안을 승인하는가?
7. **호스트 사양.** 16 vCPU 이상, 64 GB RAM 이상을 확보할 수 있는가? CPU는 세 번째 가속기라서 이 가정이 Phase 2 일정을 좌우한다.
8. **Garage Gold Set 예산과 1000fps 대여 시점.**

---

## 18. 신규성 장부

| 확신도 | 항목 |
|---|---|
| **중상 (이 제약이 아니었으면 안 나왔을 것)** | **크로노포토 증인 + Band-Isometric Strobe + 시간 왜곡 대조.** 이미지 교사를 스트로보, 크로노포토, 장노출로 동역학과 타이밍 판정기로 쓴다. 보정 데이터는 실영상에서 고전 프레임 선택만으로 만든다. 가장 가까운 선례는 Grid Diffusion |
| **중** | Coverage–Precision Split (물리 사영 분지 위의 best-of-N 경제학), **BSRT** (라벨 대역 보존 교사 세탁 + MAN 노이즈로 프레임 간 일관), **Reveal-Bundle Demand** (사건이 게이트하는 edit-closure), Teacher Exhaust + Judge Cascade (TC를 화폐로 보는 회계), Photo-Slice RWR (정지 사진으로 모션 prior 보정), 데이터 순도 등급 D0–D3 |
| **중하 / 조합** | 오디오 간격 증인 (AV 오프셋이 상쇄되는 반발계수와 cadence 분포), 사람이 저대역을 소유하는 시간 연방, Event-Demand Lift, Garage Gold Set, 세션 = 세대 운영, 실리콘 전 유닛 스케줄링 |
| **부품 (정직하게 차용)** | Many-Worlds Browsing과 motion matching (Chaos Bank), MuJoCo MPC, MDM, GNS, OIDN 구조, Voronoi 파쇄, Minnaert 기포, Zheng & James 모달 음향, Stam 바람, Tessendorf, MakeHuman, V-HACD, GLOMAP/COLMAP, DIS flow, ChArUco, LatentLPIPS, μP, Move.ai와 Wonder Studio류 폰 퍼포먼스 캡처 |

> **마무리.** v1은 "그리지 말고 찍어라"였다. 2×T4판은 한 줄을 더한다: **"배우지 말고 굴려라, 상상하지 말고 찍어라."**
> - 이미지 교사는 사진 한 장만 판정할 수 있다. 그래서 시간을 사진 한 장으로 접어 보여 준다.
> - 물리는 공짜이고 정확하다. 그래서 작은 생성기는 대충 던지기만 하면 되고, 정밀함은 뉴턴이 맡는다.
> - 사람만 아는 저주파 서사는 사람에게 맡긴다.
>
> 이렇게 하면 비싸게 학습해야 하는 부분은 약 125M 파라미터와 2×T4 엿새로 줄어든다. 나머지 컴퓨트는 전부 탐색과 판정, 그리고 찍는 데 쓴다. 70W 카드 두 장으로 블록 우주를 만든다. 개소리 같지만, 계산은 맞는다.
