> **출처 메모.** 2×T4 + 수십M from-scratch 제약으로 다시 돌린 설계 워크숍에서 [표현·렌더러·카메라 + Phase 1 승리] 렌즈 담당이 쓴 원안이다. 최종 결정은 [`docs/PARADIGM-v2.md`](../../PARADIGM-v2.md)를 따른다. 이 문서는 Higgsfield Soul 하이브리드 결정 **이전**에 작성됐다.

---

# [렌즈: 표현·렌더러·카메라 + Phase 1 승리] STRATA-lite
### 세계는 한 번만 만들고, 테이크마다 드는 비용은 렌더뿐이다: T4 두 장(70와트씩)으로 8K 120fps 촬영하기

> **입력.** v1 `docs/PARADIGM.md`(§0.5, §1, §2, §4, §7, §8, §10), `docs/lenses/02`, `03`, 프리모템 v0.2/v2, 지형도 리포트, 2026-10-10 창업자 하드 제약(2×T4, 사전학습 이미지 모델은 1개만 빌림, 나머지는 수십M 파라미터로 from scratch, 목표는 그대로).
> **표기.** **[신규]** 이번 제약 때문에 새로 나온 것 / **[조합]** 부품은 있고 결합이 새로움 / **[부품]** 빌려 씀. "실측 필요"라고 적은 수치는 1주차 X-T0에서 확정한다.
> **태도.** H100 64장은 남의 집 이야기다. 우리 촬영장은 70W 두 장, 합쳐서 헤어드라이어의 1/11이다. 이 문서는 그 전력으로 Seedance와 Kling을 이기는 회계 장부다.

---

## 0. TL;DR

1. **예산.** 2×T4를 90일 상시 가동하면 4,320 T4-h, 가동률 75%면 약 3,200 T4-h다. 계획에 쓰는 실효 성능은 장당 13 TFLOPS이고, 그러면 하루 2.2×10¹⁸ FLOPs, 90일 2×10²⁰ FLOPs다. v1 Phase 1 누계(0.4–0.8M H100-h ≈ 6–12×10²³ FLOPs)의 **1/3,000–1/6,000**이다. 그래서 v1을 작게 줄이는 것으로는 안 되고, 방식 자체를 바꾼다.
2. **비용 단위는 FLOPs가 아니라 교사 호출 시간(teacher-second, TS)이다.** SDXL급 교사 1024² forward 1회가 약 0.65초다. 세계 생성 비용의 약 70%가 교사 호출이다. 그래서 이 렌즈의 결정은 전부 교사 호출 수를 줄이는 쪽으로 정한다.
3. **STRATA-lite.** 정본(canonical)을 latent에서 explicit atom과 BC7 텍셀 페이지로 바꾼다. 4K Directed 세계는 0.2–0.6 GB, 8K는 0.6–1.5 GB라서 T4 한 장에 통째로 올라간다. 병목은 메모리가 아니라 교사 시간이다.
4. **[신규] TSQ (Textured Soft Quad).** T4에 맞춘 프리미티브다. 하드웨어 BC7 디코드를 쓰고, 삼각형이라 RT 코어가 교차를 가속하며, 피팅이 싸고, 게임 엔진·웹으로 바로 export된다.
5. **렌더러.** primary는 gsplat 기반 TSQ 래스터(미분 가능)다. secondary는 OptiX를 쓰되 거울·유리 픽셀에만 쏜다. 평면 거울은 재귀 래스터로 처리한다. 3DGRT를 primary로 쓰는 안은 버린다.
6. **10초 샷 렌더 시간(2×T4 벽시계).** 1080p24 10–20초, 4K24 0.6–1.3분, 4K120 3–7분, 8K24 2.5–5.5분, 8K120 12–26분. **fps는 생성 비용을 늘리지 않는다.**
7. **카메라 스택은 학습이 전혀 없다.** 특허 렌즈 처방 레이트레이싱, Hullin 플레어, 공분산 DOF·모션블러, **[조합] Longuet-Higgins Split**(회전 모션블러를 호모그래피 적분으로 정확히 계산), Defocus Pyramid, 센서·필름 수식으로 이루어진다. 4K 프레임당 60–120 ms다.
8. **세계 1개 비용.** 4K 방 규모 Directed 세계가 1–2 T4-h(벽시계 30–60분)다. v1 추정 4–12 H100-h(≈100–350 T4-h)의 약 1/100이다. 이유는 네 가지다. 교사가 2.6B로 작고, 경로가 요구하는 부분만 만들고(Directed demand), 가장 미세한 옥타브는 TexelFM이 맡고, 렌즈 게이팅(lens gating)이 배경 디테일을 지운다.
9. **★[신규] Corman–Shannon Demand.** 카메라가 기록하지 못하는 정보는 생성하지 않는다. 촬영감독(DP) 목적함수에는 T4-시간이 비용 항으로 들어간다. 얕은 심도, 연기, 밤, 역광, 모션블러는 100년 된 저예산 촬영 기법이고, 동시에 사람들이 가장 영화적이라고 느끼는 룩이다. 돈이 없어서 쓰는 기법이 그대로 미장센이 된다.
10. **이 렌즈의 소형 모델.** 9종 합계 약 125M 파라미터이고, 학습에 약 130 T4-h, 데이터 준비에 약 80 T4-h가 든다(90일 예산의 7%). 작은 모델 학습은 싸다. 비싼 것은 세계(교사 호출)와 사람 평가다.
11. **쇼케이스 순서.** 왕복(실촬영 세계, D14) → 리그 스왑 → "Shot on 2×T4" 8K120 → 을지로 네온(Corman 쇼케이스) → 거울 터널 → 경로 이식 → 불릿타임 헌정 → 10분 원테이크 → Powers of Ten.
12. **FREEZE-TAG.** 우리는 세계 1개로 카메라 의도 12개를 전부 찍고, 경쟁사는 12번 따로 생성한다. Day 90 미니 벤치는 API 비용 $1–6k, GeCo 평가 약 20 T4-h다. **최종 승리 조건은 v1 그대로 둔다.**

---

## 1. 숫자부터: 2×T4 회계

### 1.1 계획 상수

| 항목 | 값 | 비고 |
|---|---|---|
| 피크 | FP16 tensor 65 TFLOPS, INT8 130 TOPS, FP32 8.1 TFLOPS, 320 GB/s, 16 GB, 70 W, RT 코어 40개(1세대), NVENC·NVDEC | BF16, TF32, FlashAttention-2 없음. FA1/xformers/SDPA mem-efficient 사용 |
| 실효(대형 GEMM) | 25–40 TFLOPS | 70 W 전력 상한 때문에 지속 부하에서 클럭이 떨어진다. **실측 필요** |
| 계획 실효(수십M 모델 학습) | **13 TFLOPS/장** (MFU 약 20%) | 작은 모델은 커널 런치와 메모리 대역이 지배한다 |
| 하루 / 90일 | 2.2×10¹⁸ / 2×10²⁰ FLOPs | 창업자 가정(3.5–4×10¹⁸/일)의 약 60%로 잡았다. 실제로 더 나오면 보너스 |
| 환산 | 1 H100-h ≈ 25–30 T4-h | v1의 MFU 40% 기준 |
| 상시 / Kaggle | 4,320 T4-h / 약 780 T4-h (주 30 h × 2장 × 13주) | Kaggle 주간 쿼터는 공식 문서에 없고 계정마다 다르다. 세션 상한은 12 h로 보고됨 |

### 1.2 교사 호출 시간 기준 단가표 (T4 1장)

| 연산 | 시간 | 비고 |
|---|---|---|
| SDXL UNet forward 1024² fp16 (PyTorch SDPA) | 0.6–0.8 s (계획값 0.65) | **실측 필요** |
| 같은 연산, TensorRT fp16 / INT8 W8A8 | 0.45–0.55 / 0.35–0.45 s | INT8은 노이즈가 큰 스텝에만 쓴다. [조합] σ-정밀도 스케줄, §5.3 |
| SDXL UNet forward 768² / 512² | 약 0.35 / 약 0.16 s | |
| SDXL VAE decode 1024² fp32 | 1.5–2 s | fp16 NaN을 피하려고 fp32로 돌린다. commit 시에만 쓴다 |
| **TL-2.5M**(우리 tiny AE) enc+dec 1024² | 0.03–0.05 s | 루프 안에서 VAE 대신 쓴다 |
| TSQ 래스터 1024² forward / forward+backward | 10–20 / 40–60 ms | |
| TexelFM 30M, 텍셀 10⁸개, 4 step | 60–90 s | |
| 4K 시네마 프레임 1장 렌더 | 0.25–0.55 s | §3.6 |

**view-step 하나**는 렌더 → TL 인코드 → 교사 1회 → TL 디코드 → 세계 피팅 3 iteration이다. 장당 약 0.95초이고, 두 장을 공간 분할로 병렬 운용하면 **초당 약 2.1회, 시간당 약 7,500회**다.

### 1.3 부족분을 무엇으로 메우는가

수단은 넷이다. (a) 빌린 교사가 이미 치른 사전학습 컴퓨트, (b) 학습이 필요 없는 구조(기하·광학·물리), (c) 세계별 테스트타임 최적화, (d) 우리 소형 모델.

| v1 구성요소 (규모) | T4 대체 | 수단 |
|---|---|---|
| LLM World Program | 텍스트 전용 모드에서는 **교사 자신이 world programmer**다(hero search → GeoHead lift). 데모용 directed 모드에서는 WorldScript DSL을 쓴다 | (a)(d) |
| CWM 4–8B (bedrock) | hero still lift + 매싱(massing) + 고전 완성 + 교사 coarse 뷰 lockstep(전역 1-pass) | (a)(b)(c) |
| LBO 학생 (amortized) | **TexelFM 30M**은 최미세 옥타브만, 나머지는 세계별 WSSC-lite | (c)(d) |
| 3D object prior (TRELLIS 등) | 절차적 에셋(Blender)과 CC0 모델(데이터) | (b) |
| SelfCaption VLM | 구조화 캡션: prompt_tree@scale + sem class + 스케일 + lensTags | (b) |
| RAFT/UFM 보정 광류 | DIS / TV-L1 고전 광류 | (b) |
| DA3/VGGT/AnySplat 실사 lift | COLMAP/GLOMAP + PatchMatch MVS + TSQ 피팅 | (b)(c) |
| 3DGRUT 단일 렌더러 | TSQ 래스터 + OptiX(선택 픽셀) + 평면 재귀 | (b) |
| 렌즈·센서·필름 | 그대로. 원래부터 수식이다 | (b) |
| DP-0 DiT + CRM + IMU 손 | **CamFM** 15M + 5M + 8M | (d) |
| 닫힌 미감 판사 / art director | **AesHead** 5M (동결 SDXL 특징 위) + hero search | (a)(d) |
| GeCo용 VGGT/UFM | 그대로 쓰되 **평가 전용**이고 시스템 경로에는 넣지 않는다 | — |

**정직성 규칙.** 우리 모델의 학습 손실과 데이터 생성 경로에도 금지된 사전학습 모델을 쓰지 않는다. 예를 들어 LPIPS(VGG) 대신 SDXL 특징 기반 perceptual loss나 from-scratch 판별기를 쓴다. 외부 데이터셋은 데이터로 취급한다.

---

## 2. STRATA-lite: 16GB 두 장에 맞춘 세계 매개체

### 2.1 설계 원칙 다섯

1. **병목은 메모리가 아니라 교사 시간이다.** 4K 세계가 1 GB 미만이다. 남는 메모리로 교사 호출을 아낀다(Judgment Ledger, 모든 레벨을 완전 저장).
2. **정본은 explicit이다.** latent 정본은 그것을 풀어 줄 디코더(학생)가 있을 때만 의미가 있고, 학생은 Phase 1.5의 일이다. 포맷에 `z` 슬롯(int8 d=16)은 예약해 둔다.
3. **하드웨어가 공짜로 주는 기능을 쓴다.** BC7/BC5 텍스처 디코드, 바이리니어 필터, RT 코어 삼각형 교차, NVENC가 그렇다.
4. **최미세 옥타브는 교사가 아니라 TexelFM이 맡는다.** v1 연방제의 L4에 해당한다. 레벨별 교사 호출 수는 등비수열이라 4분의 3이 최미세 레벨에 몰리는데, 그 고리를 여기서 끊는다.
5. **모든 생성 연산은 Jacobi 방식(레벨 안 동기식)이다.** 그래서 두 장에 공간 분할로 완전히 병렬화된다. **비회귀 조항이 곧 하드웨어 효율이다.** 순차 아웃페인팅은 구조상 파이프라인을 못 만든다.

### 2.2 레벨 체계와 숫자

셀 크기는 $h_l=h_0 2^{-l}$이고, TSQ 1장이 셀 하나와 8×8 텍셀 페이지를 맡으므로 텍셀 피치는 $\tau_l=h_l/8$이다. 4K S35(픽셀 피치 6.48 µm), 35mm에서 픽셀 footprint는 $\phi(z)=1.85\times10^{-4}z$다.

예시는 lot $h_0=64$ m:

| l | 셀 $h_l$ | 텍셀 $\tau_l$ | 4K/35mm에서 이 레벨이 필요한 최대 거리 | 결정 주체 |
|---|---|---|---|---|
| ≤6 | ≥1 m | ≥12.5 cm | — | **bedrock**: hero lift + 매싱 + 교사 coarse lockstep |
| 7–10 | 50–6.25 cm | 6.3 cm–7.8 mm | 약 340–42 m | 교사, 시네마틱 뷰(C2 판결) |
| 11–14 | 3.1 cm–3.9 mm | 3.9–0.49 mm | 21–2.6 m | 교사, band-isometric 뷰, 스칼라 t |
| 15 | 1.95 mm | 0.24 mm | 1.3 m (8K는 2.6 m) | **TexelFM**(재질 셀) / 교사(의미 셀: 소품, 얼굴, 간판 주변) |
| 16+ | ≤0.98 mm | ≤0.12 mm | 0.66 m (8K는 1.3 m) | TexelFM (WDD) |
| PSH 구조 | — | — | 레벨 게이팅 면제 | SYMBOL(폰트 MSDF), POINT(nits), CURVE: 전부 DSL과 수식 |

lot 크기는 방 16 m, 거리 64–128 m, Powers of Ten 4096 m로 둔다. 교사가 판정하는 옥타브는 4K에서 7–8개다. 교사 뷰 하나(1024²)는 레벨 $l$ 셀 128개 폭을 덮는다. l=14면 6.4 cm 너비의 "매크로 사진"이다.

### 2.3 자료구조

```
World (STRATA-lite v0)
├─ meta     : seed u128, units=m, up=+Z, h_0, L, l_g, mode∈{directed, location-lite},
│             teacher_id(hash), merkle_root, toolchain pins(CUDA/driver/renderer ver)
├─ blocks[l]: sparse hash (l,i,j,k:int32) → Block(16³ cells)        # 스트리밍·Merkle·샤딩 단위
│             { occ bitset 512B, tsq_range u32×2, page_range u32×2, state u8,
│               own_summary u16, unc_summary f16, hash u64(blake3) }
├─ tsq (SoA, 32 B) : μ f16×3 (block-rel) | q oct-packed 32b | s_uv f16×2 | thick f16
│                    | opacity u8 | edge_n u8 (soft↔hard) | level u8 | flags u8
│                    | L_d RGB9E5 u32 | F0/rough/metal u8×3 | sem u16(class) | own u16 | unc u8
├─ pages    : 8×8 texel / TSQ.  BC7(L_d delta, 64B) + BC5(normal/height delta, 64B) = 128 B
├─ vol      : 3D Gaussian + SGGX (fuzzy matter: 잎·털·연기 aggregate), 48 B
├─ structs  : SYMBOL{font id, utf-8, baseline, size, emissive nits, spectrum}
│             POINT{pos, nits, lobe} | CURVE{bezier chain, width, L_d} | GLINT | CARRIER(ψ grid)
├─ iface    : DIEL narrow-band SDF bricks 8³ f16 (IOR, Abbe)
├─ far_field: octahedral HDR(RGB9E5) 3 bands + depth, Hosek–Wilkie/Prague sky params
├─ lights   : world_lights(sun/sky/IES/emissive) | shot_rigs[shot] | atmosphere{β, g}
├─ wnf      : 저장 0바이트 — Philox(seed, l, cell, k)를 렌더 중에 즉석 계산
├─ ledger   : Judgment Ledger 인덱스 (§2.7)
└─ z_slot   : (예약) int8[16]/cell — Phase 1.5 학생 도입 시 활성화
```

정본이 explicit이어서 생기는 이득이 두 가지 있다. 디코드 비용이 0이고, 교사가 판정한 렌더와 최종 렌더가 같은 코드 경로를 쓴다(v1의 train/test gap 0 원칙 유지). TexelFM 레벨은 feed-forward라서 Adam 상태가 필요 없고, 그만큼 메모리가 남는다.

### 2.4 [신규] TSQ (Textured Soft Quad): T4에 맞춘 프리미티브 [조합: 2DGS, BBSplat, Textured Gaussians]

- **형태.** 방향을 가진 사각형이다. 반폭 $(s_u,s_v)$, 국소 좌표에서 일반화 가우시안 가장자리 $\exp(-(|u|^{n}+|v|^{n}))$을 쓰고, $n=2$면 부드러운 surfel, $n\ge8$이면 단단한 쿼드가 된다. 불투명도를 따로 가진다.
- **외관.** ray-quad 교차로 얻은 $(u,v)$에서 8×8 페이지를 하드웨어 바이리니어로 샘플한다. $L_d$는 scene-linear HDR 델타를 log 인코딩해 BC7로, 법선과 높이는 BC5로 저장한다.
- **T4에 맞는 이유.**
  - **메모리.** 텍셀당 2 B(BC7+BC5)다. 순수 가우시안으로 같은 고주파를 내려면 2–4 텍셀마다 64 B짜리 가우시안이 하나씩 필요하다. 그래서 **약 60–120배 압축**이다.
  - **RT 코어.** Turing RT 코어는 삼각형 교차를 하드웨어로 가속한다. TSQ의 불투명 코어는 삼각형 2개다. 3DGRT식 정이십면체 프록시 + any-hit 방식은 1세대 RT 코어에서 느리다.
  - **피팅.** 프리미티브 수가 적어서 정렬과 역전파가 싸다.
  - **export.** "텍스처가 붙은 쿼드"는 곧 표준 메쉬라서 USD, Unreal, 웹 뷰어로 그대로 나간다.
- **예외는 v1 C6 판결 그대로다.** 퍼지 물질은 VOL(SGGX)로, 유전체는 SDF로, 아바타는 Phase 2c로 미룬다.

### 2.5 세계별 메모리

| 세계 | $h_0$ | 레벨 | TSQ 수 | 텍셀 | RAM/디스크 | GPU 작업 집합 |
|---|---|---|---|---|---|---|
| Oracle 방 (실촬영) | 16 m | 0–14 | 1–3M | 3–8×10⁷ | 0.15–0.3 GB | 전부 상주 |
| 을지로 네온 거리 (4K, Corman) | 64 m | 0–15 | 2–4M + POINT 5k + SYMBOL 300런 | 3–6×10⁷ | 0.2–0.4 GB | 전부 상주 |
| 거울 복도 | 32 m | 0–15 | 2–4M | 4–8×10⁷ | 0.25–0.5 GB | 전부 상주 |
| 불릿타임 바 + 와인잔 | 16 m | 0–16 + DIEL SDF | 2M + SDF 8×10⁷ voxel | 3×10⁷ | 0.4 GB (SDF 0.16) | 전부 상주 |
| 같은 방 8K | 16 m | 0–16 | 4–8M | 2–3×10⁸ | 0.6–1 GB | 전부 상주 |
| 10분 원테이크 집 (4K) | 64 m | 0–15 | 20–40M | 약 10⁹ | 3–6 GB | 경로 주변 1.5–3 GB 스트리밍 |
| Powers of Ten (8K) | 4096 m | 0–22 | 수요 원뿔만 5–15M | 2–5×10⁸ | 1–2 GB | 옥타브 2–3개 스트리밍 |

### 2.6 16GB 두 장 배치

**솔버 모드(세계 생성).** 두 장이 각각 교사와 피팅을 다 맡고, 블록을 Morton 순서 짝/홀로 나눈다. 장 사이에 오가는 것은 경계 apron과 Ledger 해시뿐이다. 그래서 NVLink가 없어도 PCIe Gen3면 충분하다.

| 항목 | T4 #0 | T4 #1 |
|---|---|---|
| SDXL UNet fp16 (2.57B) | 5.1 GB | 5.1 GB |
| SDXL 텍스트 인코더 (CLIP-L + bigG fp16) | — | 1.6 GB (임베딩을 캐시한 뒤 CPU로 내릴 수 있음) |
| SDXL VAE fp32 (commit 전용) | 0.33 | — |
| TL, GB-Adapter, GeoHead, AesHead | 0.2 | 0.2 |
| UNet 활성값 (batch 2 @1024², no grad) | 3.0 | 3.0 |
| 세계 shard: 활성 레벨 param + grad + Adam | 1.9 | 1.9 |
| 세계 shard: 커밋된 거친 레벨 (BC7) | 0.2 | 0.2 |
| 래스터 버퍼 (뷰 2장, G-buffer, 정렬, 역전파) | 1.2 | 1.2 |
| **합계 / 여유 (가용 약 15 GB)** | **약 11.9 / 3.1** | **약 13.2 / 1.8** |

OOM이 나면 텍셀 Adam 모멘트를 pinned CPU 메모리로 내리고(PCIe 약 10 GB/s), 뷰 배치를 1로 줄인다.

**8K 렌더 모드(장당).** 세계 전체 BC7 0.6–1 GB, 8K RGBA fp32와 AOV(깊이, 법선, 모션, cryptomatte 3랭크) 1.5 GB, (atom, tile) 정렬 리스트 1–2 GB, Defocus Pyramid 0.4 GB, FFT 버퍼(8K 패딩 3채널 complex64) 1.6 GB, OptiX BVH 0.5–1 GB를 합쳐 **6–8 GB**다. 두 장이 짝·홀 프레임을 나눠 맡는다.

**학습 모드.** GB-Adapter는 동결 SDXL을 통과해 역전파해야 한다. 640²에서 gradient checkpointing을 켜면 UNet 5.1 + 활성값 6–7 + 옵티마이저 0.5로 12–13 GB다. 나머지 소형 모델은 4 GB 미만이다.

**CPU와 디스크(권장).**
- RAM 64–128 GB. 세계 전체 버전 ≤6 GB, Ledger hot set(엔트리당 128 KB × 5만 = 6.4 GB), EXR writer 큐(8K 프레임 260 MB × 16장 = 4 GB), 1 kHz 리그 시뮬을 올린다.
- NVMe 2 TB. Ledger cold 약 100 GB, 데이터(CC0 PBR 8K 약 150 GB, 합성 렌더 약 200 GB, MVS 약 100 GB), 출력(8K EXR 10초@120fps = 50–100 GB)을 둔다.

**Kaggle 모드.** RAM과 디스크 크기는 미검증이다. 세션이 끝날 때 Ledger와 Merkle 커밋을 외부 스토리지나 Kaggle Dataset 버전으로 넘긴다. 8K EXR은 포기하고 NVENC HEVC 10-bit로만 출력한다. MVS 전처리는 로컬 CPU에서 돌린다.

### 2.7 Judgment Ledger [조합]와 Proof-of-Shot

- **Ledger.** key는 `blake3(block Merkle hash ‖ canonical view(level, block, band-iso index) ‖ σ step ‖ WNF seed ‖ teacher id ‖ caption hash)`이고, value는 교사 x̂0 latent(4×128×128 fp16 = 128 KB)와 보정 잔차 통계다.
- **재사용되는 경우.**
  - 크래시나 Kaggle 12시간 세션 종료 후 재개. 레벨 commit 단위로 정확히 이어 간다.
  - Merkle 해시가 바뀌지 않은 블록의 편집.
  - 카메라만 바꾼 재촬영.
  - 거친 레벨을 공유하는 A/B 실험.
  - 벤치에서 의도 12개가 같은 세계를 공유할 때.
- **효과.** 개발 반복에서 적중률을 40–70%로 예상한다. 실효 교사 비용이 절반 근처로 준다. 내용 주소 메모이제이션(Nix, Bazel, git)을 교사 판정에 적용한 것이다.
- **[신규] Proof-of-Shot(검증 가능한 영상).** 영상과 함께 (세계 Merkle root, ShotScript, 렌더러 버전, 툴체인 pin)을 공개한다. 누구든 다시 렌더해 차이가 0인지 확인할 수 있다.
  - forward 커널에 atomic을 쓰지 않고, 정렬 키를 (깊이, atom id)로 고정해 stable sort한다.
  - 같은 아키텍처에 같은 툴체인이면 바이트 단위로 동일하고, 아키텍처가 다르면 1 LSB 이내다.
  - 이렇게 하면 왕복 비트 동일성이 골라낸 결과가 아니라는 것을 시청자가 직접 확인할 수 있다.

### 2.8 비회귀 점검 (STRATA-lite 부품별)

| 부품 | 조건으로 쓰는 것 | 판정 |
|---|---|---|
| TexelFM | 부모 레벨 페이지, 유계 apron, WNF, 재질 파라미터 | **비트 결정론 native LBO.** T8 정확. 렌더 픽셀은 조건으로 쓰지 않는다 |
| bedrock | 전역 1-pass, 모든 coarse 뷰 lockstep | v1의 유일한 허용 예외 그대로 |
| 수요 확장 | 더 거친 커밋 레벨과 WNF | 세계 공간 생성이다 |
| Corman–Shannon | 생성할 레벨의 집합만 바꾼다 | 순서축(스케일)은 그대로 |

---

## 3. 렌더러 on Turing

### 3.1 선택

| 후보 | sm_75 지원 | 장점 | 문제 | 결정 |
|---|---|---|---|---|
| **gsplat** (CUDA, 2DGS/3DGS, 3DGUT 통합) | 지원 예상 (빌드 실측) | 미분 가능. 3DGUT로 왜곡과 롤링셔터 처리 | 텍스처 surfel이 없다 | **primary 베이스.** TSQ 커널 확장 |
| 3DGRUT / 3DGRT (OptiX) | README에 최소 아키텍처 명시 없음. RT 코어 GPU 권장 → T4도 될 가능성이 높으나 실측 필요 | 반사, 굴절 | 1세대 RT 코어 40개, any-hit 비용 | **secondary 전용**, 거울·유리 픽셀만 |
| 평면 거울 재귀 래스터 | 고전 기법 | RT 불필요, 깊은 재귀, LOD가 자연히 감소 | 평면만 가능 | **거울 복도 주력** |
| Blender Cycles (OptiX) | 지원 | 정답 레퍼런스 | 느림 | 검증용 레퍼런스(ΔJOD)와 합성 데이터 생성 |
| 메쉬 export → Vulkan/WebGL | 지원 | 실시간 | soft edge 근사 | 웹 뷰어와 엔진 export |

### 3.2 Dual-cut LOD on T4

v1 C7 판결대로 $I=(1-f)I^{(l_a)}+fI^{(l_a+1)}$를 쓴다(λ-연속, T9). 컷 선택은 **16×16 타일 단위**로 하고, $f\in(0.05,0.95)$인 타일만 두 번 렌더한다. 프레임 평균 비용은 +20–40%다. α-fade는 계속 금지한다.

### 3.3 공분산 대수 + 두 가지 T4 절약 장치

atom마다 화면 공분산을 $\Sigma'=J\Sigma J^\top+\Sigma_{pix}+(c/4)^2I+\tfrac{\Delta t^2}{12}\dot{\mathbf m}_{trans}\dot{\mathbf m}_{trans}^\top$, 불투명도를 $\alpha'=\alpha\sqrt{\det\Sigma^0/\det\Sigma'}$로 둔다.

**(1) Defocus Pyramid [조합: Kraus–Strengert'07, Lee'09].**
- CoC $c>c_0=8$ px인 atom은 레이어 $k=\lceil\log_2(c/c_0)\rceil$에서 $2^{-k}$ 해상도로 렌더한다. 레이어 안 잔여 CoC $c/2^k\le c_0$는 공분산으로 처리한다.
- 레이어들은 깊이 순서로 premultiplied 합성한다.
- 이렇게 하는 이유는 fill-rate다. 120 px CoC 가우시안 하나가 픽셀 1.4만 개를 칠한다. 렌즈 게이팅 때문에 이 atom들은 어차피 거친 레벨에서 온 것이라 디테일 손실은 0이다.
- 부분 가림(partial occlusion)은 근사다. 20개 샷을 Cycles MC 레퍼런스와 비교해 ΔJOD < 0.25(ColorVideoVDP, 평가 전용)를 검증한다(X-R).

**(2) [조합] Longuet-Higgins Split.**
- 모션 필드는 $\dot{\mathbf u}=\frac1ZA(\mathbf u)\mathbf t+B(\mathbf u)\boldsymbol\omega$다. **회전 항은 깊이와 무관**하다.
- 순수 회전 카메라의 이미지는 정확히 $H(\tau)=KR(\tau)R_{mid}^\top K^{-1}$ 호모그래피다. 시차도 disocclusion도 없다.

$$I\approx\frac1N\sum_k H(\tau_k(y))_\#\Big[I^{\,\Sigma^{trans}_{mb}}_{\,T_{mid}}\Big]_{\text{overscan}}$$

- 셔터 중간 자세로 오버스캔 렌더를 **한 번** 하고, 평행이동 블러만 공분산으로 넣는다. 회전 블러는 호모그래피 적분으로 정확히 계산한다.
- 롤링셔터는 행마다 $\tau_k(y)$를 쓰고, 왜곡은 undistorted 공간에서 처리한 뒤 다시 왜곡한다.
- 핸드헬드 손떨림은 회전이 지배적이므로 **1 kHz 손 지터 블러가 거의 공짜로 정확해진다.**
- 1,500 px 휩팬 스트릭은 저해상 누적과 경로 prefix-sum으로 처리한다. 4K 기준 10–40 ms다.
- 회전 블러 = 호모그래피라는 것은 비균일 디블러링 문헌(Whyte'10)의 표준 모델이다. 이것을 렌더러 분해와 1 kHz 리그 시뮬에 결합한 것이 우리 몫이다.

### 3.4 거울: 평면 재귀와 초록 터널

- **평면 거울.** 가상 카메라 $C'=\mathrm{Refl}_M(C)$로 클립 평면과 스텐실을 걸어 재귀 래스터한다. 깊이 12까지 간다. 반사가 거듭될수록 가상 거리가 멀어져 LOD가 자동으로 거칠어지므로 **총비용이 단일 프레임의 2배 이하**다.
- **[신규 데모 디테일] 초록 터널.** 실제 거울은 뒷면 코팅이라 빛이 플로트 유리를 반사마다 두 번 통과한다. 유리 속 철분이 흡수하므로 반사마다 $R(\lambda)\approx0.9\cdot T_{glass}(\lambda)^2$가 곱해진다. 그래서 무한 거울 터널은 **점점 초록빛으로 어두워진다.** 비디오 모델은 이 물리를 모른다. 학습이 필요 없는 광학 포렌식 소재다.
- **곡면 거울, 크롬, 유리.** OptiX로 TSQ 불투명 코어 삼각형을 추적한다. 해당 픽셀(대개 5–30%)에만 깊이 2까지 쏜다.
  - T4 Gaussian-particle RT 처리량은 초당 1–3×10⁷ ray로 추정한다(실측 필요).
  - 4K에서 20% 픽셀이면 0.04–0.15 s다.
- **DIEL(와인잔).** SDF 레이마칭으로 분산(Abbe 수)까지 처리하고, 4K 프레임당 2–4 s다.

### 3.5 렌즈·센서·필름 스택: 학습 0, 전부 수식

| 블록 | 방법 / 수식 | 파라미터 출처 | 4K 비용 (T4) |
|---|---|---|---|
| 렌즈 처방 → 출사동 LUT | Kolb'95 실광선 추적. 필드별 동공 모양(cat-eye), 왜곡, breathing $f_{eff}=f(1+\beta m)$, 횡·종 색수차 | **특허에 공개된 처방**(데이터) | 오프라인 1회 (렌즈당 CPU 수 분) |
| DOF | $c(z)=\frac{f^2}{N(z_f-f)}\frac{\lvert z-z_f\rvert}{z}$, $\Sigma_{dof}=(c/4)^2I$ + Defocus Pyramid | — | 20–60 ms |
| 보케 하이라이트 | POINT/GLINT 조리개 스프라이트(필드별 LUT 모양, onion ring, 구면색수차 프린지)를 레이어 해상도에서 scatter | LUT | 10–30 ms |
| 고스트 플레어 | Hullin'11. 2회 반사 경로 $N(N-1)/2$개, 고스트는 광학 중심과 광원을 잇는 직선 위에 놓인다 | 처방 | 5–20 ms (광원 상위 16개) |
| 베일링 글레어, 회절 스타 | HDR ⊛ PSF (FFT). PSF = $\lvert\mathcal F\{\text{조리개 다각형}\}\rvert^2$. 짝수 N은 스파이크 N개, 홀수 N은 2N개 | 블레이드 수 | 약 15 ms |
| 비네팅 | $\cos^4$ + 기계적 비네팅(출사동 LUT) | — | 1 ms |
| 셔터, 모션블러 | LH-split + 롤링 $t(y)=t_0+y\tau_{line}$ | 셔터각, 라인 시간 | 10–40 ms |
| 센서 | $x=\mathrm{Poisson}(aL\tau g)/a+\mathcal N(0,\sigma_r^2)$, full-well, ADC, Bayer + Malvar–He–Cutler 디모자이크 | 데이터시트 | 5–10 ms |
| 톤, 룩 | 공개 log 커브 → 3D LUT 65³ (print 2383 에뮬레이션 등) | 공개 LUT | 2 ms |
| 할레이션 | 하이라이트 임계값 → 적색 가중 블러 → 가산 | 필름 스톡 | 5 ms |
| 그레인 | Newson'17 Boolean 모델로 미리 생성한 **그레인 타일 뱅크**, 프레임마다 seed 오프셋. **유일한 프레임별 랜덤** | 스톡별 입자 반경 | 3–5 ms |
| 게이트 위브 | 서브픽셀 결정론 오프셋 | — | 1 ms |
| **합계** | | | **60–120 ms** (8K는 약 4배) |

**학습된 파라미터는 0개다.** 모든 손잡이가 mm, f-stop, ISO, ms 같은 물리 단위라서, 실제 촬영 현장과 같은 언어로 룩을 주고받을 수 있다.

### 3.6 렌더 시간표

프레임당 시간(T4 1장, 시네마 품질: dual-cut, 공분산 DOF·MB, Defocus Pyramid, 20% 픽셀 RT, 전체 렌즈·필름 스택, AOV 포함)은 보수적 추정이고 X-T0에서 실측한다.

| | 1080p | 4K UHD | 8K |
|---|---|---|---|
| primary 래스터 (dual-cut 평균 1.3배) | 40–70 ms | 150–300 ms | 0.6–1.2 s |
| secondary RT | 10–40 ms | 40–150 ms | 0.15–0.6 s |
| 렌즈·센서·필름 | 15–30 ms | 60–120 ms | 0.25–0.5 s |
| **합계** | **0.07–0.14 s** | **0.25–0.55 s** | **1.0–2.2 s** |

**10초 샷 벽시계** (2×T4가 짝·홀 프레임 분담, I/O +20%):

| | 24 fps (240 프레임) | 60 fps (600) | 120 fps (1,200) |
|---|---|---|---|
| 1080p | 10–20 s | 25–50 s | 50–100 s |
| 4K | 0.6–1.3 min | 1.5–3.3 min | 3–6.6 min |
| 8K | 2.4–5.3 min | 6–13 min | **12–26 min** |

- **생성 비용과의 관계.**
  - fps는 생성 비용을 늘리지 않는다(1배).
  - 해상도는 렌즈 게이팅된 수요만큼만 생성을 늘린다. 4K→8K는 TexelFM 옥타브 하나와 의미 셀 일부에 대한 교사 호출이 더해질 뿐이다(+0.3–0.5 T4-h, §6).
- **저장.** 8K EXR(half, AOV 포함, DWAA 압축)은 프레임당 40–80 MB다. 10초@120fps면 50–100 GB다. 리뷰 사본은 Turing NVENC HEVC 10-bit 8K로 만든다(8K 인코드 실측 필요).
- **에너지.** 4K120 10초는 2 × 70 W × 약 5분 ≈ **12 Wh로, 스마트폰 1회 충전량 수준**이다. 8K120은 약 47 Wh다.

### 3.7 라이브 프리뷰

1080p 절반 품질(RT·필름 끔)로 장당 25–40 fps가 나온다. 폰 VCam(ARKit 포즈를 websocket으로 받음)으로 **T4 한 장에서 라이브 촬영**이 된다. ARKit는 입력 장치로 취급하고 시스템 모델이 아니다.

---

## 4. 카메라: Kinetic Camera + CamFM (DP-0-lite)

### 4.1 리그 (학습 0)

v1 §3.3의 리그 14종을 1 kHz ODE로 CPU에서 시뮬한다(질량, 감쇠, 홀로노믹 제약, 진자 $\omega_n$, PID). 의도 층은 min-jerk, 2/3 power law, 지연 피드백이다. **학습이 필요한 것은 사람 손뿐이다.**

### 4.2 CamFM: 카메라 소형 파운데이션 모델 3종

| 모델 | 파라미터 / 아키텍처 | 입력 → 출력 텐서 | 데이터 (사전학습 모델 없이) | 학습 | 추론 |
|---|---|---|---|---|---|
| **CamFM-Intent** | 15M, DiT 8층 d=384, flow matching(연속값) + masked(이산값) | 키프레임 16×12D (pos3, rot6d, f, d_f, N) + 세계 토큰 128×16D (scene-graph box 64개 + 4³ 풀링 점유 64개) + 텍스트 1토큰 (SDXL CLIP-L pooled 768 → proj, **교사의 일부**) + 스타일 32D → 키프레임 velocity | DataDoP 29K, CameraBench, **Blender 오픈무비 .blend의 정확한 카메라·렌즈·포커스 애니메이션**(CC-BY, 정답 라벨), 라이선스 클립에 대한 자체 COLMAP 추출 | 약 10¹⁶ FLOPs, 1 T4-h 미만 | 궤적 1,000개에 1 s |
| **CamFM-Hand** | 5M, 1D transformer 6층 d=256 | 200 Hz gyro(+accel) 4초 = 800×6 → patch 8 → 100토큰. 조건(rig, $\omega_n$, ζ, $v_{intent}$, 피로, 오퍼레이터 emb) → ω 잔차 시퀀스 | **자체 GoPro GPMF 100시간 이상**(카메라 2–3대, 오퍼레이터 3명, 리그 4종, 4주), Gyroflow 공유 로그, Betaflight blackbox | 약 4×10¹⁶ FLOPs, 1 T4-h | 실시간보다 빠름 |
| **CamFM-CRM** | 8M. 궤적 encoder 4M + 키프레임 TL latent CNN 3M + 세계 토큰 1M | (궤적, 키프레임 8장의 4×32×32 latent, 세계 토큰) → 스칼라. Bradley–Terry 손실 | 사람 쌍대 비교: 90일에 일반 5k + 전문가 500, 이후 1.5–2만 + 2천 (v1) | 1 T4-h | 궤적당 1 ms |

**DP 탐색 비용.** 후보 1,000개를 CRM과 리그 시뮬만으로 먼저 채점하고(렌더 없음, 수 초), 상위 50개만 360p 48프레임 프리뷰를 렌더해(각 약 1 s) 재채점한다. 샷당 1–2분이다. Gaffer는 조명 ≤8개를 CEM으로 200회 평가한다(각각 렌더 + AesHead @768² ≈ 0.4 s). 약 80 s다.

### 4.3 ★[신규] Corman–Shannon Demand: 저예산 촬영 기법이 곧 미장센

> 로저 코먼은 이틀 만에 찍은 영화로 유명하다. 연기 기계, 밤 촬영, 얕은 심도, 역광 실루엣은 B급 영화가 세트 값을 아끼던 수법이다. 동시에 관객이 가장 "영화 같다"고 느끼는 룩이다. 우리는 이 사실을 수식으로 만든다.

**(1) Shannon Demand: 카메라가 기록할 수 있는 정보만 생성한다.** 세계 점 $x$, 레벨 $l$(세계 주파수 $f_l=1/2\tau_l$), 경로 Γ, 렌즈 θ, 노출과 센서 $s$, 대기 β에 대해 다음을 정의한다.

$$
\mathcal I_l(x)=\max_t\ \log_2\!\Big(1+\frac{\big[\mathrm{MTF}_\theta(\nu)\cdot\tfrac{2J_1(\pi c\nu)}{\pi c\nu}\cdot\mathrm{sinc}(\nu\,v\Delta t)\cdot\mathrm{sinc}(\nu)\cdot e^{-\beta z}\big]^2\,S_l(x)}{N_{shot}(L+L_{air},\tau,EI)+N_{read}+N_{grain}}\Big),\qquad \nu=f_l\,\phi(x,t)\le\tfrac12
$$

- $S_l(x)$는 재질 클래스별 대역 신호 전력 prior다. 멱법칙 $\propto\bar L^2C^2f_l^{-\alpha}$를 따르고, CC0 스캔과 TexelFM 통계로 측정한다.
- $L_{air}$는 안개가 더하는 airlight다. 대비는 깎고 shot noise는 올린다.
- 수요 집합은 $\mathcal D=\{(l,b):\max_{x\in b}\mathcal I_l(x)\ge\epsilon\}\cup\text{승격 구조}$이고, $\epsilon\approx0.05$–0.1 bit/sample로 둔다.
- v1 Ray-Bundle Demand의 일반화다. CoC와 모션블러뿐 아니라 **센서 노이즈, 그레인, 안개, 회절 MTF**까지 생성을 생략하는 근거가 된다.

**(2) Corman Optimizer: DP가 T4-시간을 지불한다.**

$$
\max_{\Gamma,\theta,\mathcal L,\beta}\ \mathrm{CRM}(\Gamma)+\beta_a\,\mathrm{Aes}(\Gamma,\theta,\mathcal L)-\lambda\,C_{T4}(\Gamma,\theta,\mathcal L,\beta)\quad\text{s.t. ShotScript 의도 제약},\qquad C_{T4}=\sum_{(l,b)\in\mathcal D}c_l\lvert b\rvert
$$

- $c_l$은 레벨 $l$의 단위 면적당 교사 호출 시간(또는 TexelFM 비용)이다.
- λ는 **예산 다이얼**이다. 감독 UI에서 슬라이더로 노출한다.
- 탐색 변수에는 **스모크 머신(β)**, 조명 리그, 노출, ISO, 조리개가 포함된다. 안개는 froxel 단일 산란과 섀도맵으로 렌더하고, 볼류메트릭 광선(god ray)은 덤으로 나온다.

**(3) 반증 가능한 예측(X-C).** 20개 프롬프트에서 Corman을 켜고 끈 세계를 같은 의도로 비교한다.

- 예측: Corman-on이 **대규모 세계(거리, 원테이크, 8K)에서 T4-h를 3–10배 줄이면서 SP는 뒤지지 않는다**(Δ ≥ −3%p).
- 방 하나 4K에서는 hero search가 비용을 지배하므로 절감이 20–40%에 그칠 것이다. 이것도 정직하게 보고한다.
- 예측이 맞으면 논문 하나다: *Budget-Aware Cinematography for Generative Worlds*.

**(4) 이 제약이 없었다면 아무도 하지 않았을 이유.**
- 컴퓨트가 넉넉한 팀은 범용성을 위해 모든 픽셀을 선명하게 만든다.
- 비디오 모델은 명시적인 수요가 없어서 같은 원리를 쓸 수가 없다. 프레임마다 모든 픽셀 값을 낸다.
- 우리는 세계와 카메라를 분리했고 GPU가 두 장뿐이다. 그 두 조건이 겹쳐야 "카메라를 예산 장치로 쓴다"는 발상이 나온다.

**(5) 벤치 공정성.** FREEZE-TAG 의도가 렌즈나 조명을 지정하면 Corman은 그 값을 바꾸지 못한다. 결과는 항상 Corman on/off를 둘 다 보고한다.

---

## 5. 이 렌즈의 소형 모델 총괄 + 교사 계약

### 5.1 모델 표

| 모델 | 파라미터 / 아키텍처 | 입력 → 출력 | 데이터 (사전학습 모델 없이) | 학습 FLOPs | T4-h (2×T4 벽시계) | 추론 | GPU 메모리 |
|---|---|---|---|---|---|---|---|
| **TL-2.5M** (Tiny Latent) | 2.5M, TAESD형 conv enc/dec | $x$[B,3,H,W] ↔ $z$[B,4,H/8,W/8] (SDXL latent 공간) | 우리 렌더, 자체 촬영, 공개 이미지 20만 장. 인코더 타깃 = SDXL VAE(교사의 일부). 손실 = L1 + SDXL 특징 perceptual + from-scratch 판별기 | 약 5×10¹⁷ | 약 20 (10 h). 타깃 생성 8 T4-h 포함 | 1024² 30–50 ms | <0.1 GB |
| **GB-Adapter** | 30M, T2I-Adapter형 4-스케일 conv 피라미드 → SDXL down-block 잔차 | $g$[B,12,H,W] (log 깊이1, 법선3, 알베도3, sem 임베딩4, 레벨 마스크1) → {320@H/8, 640@H/16, 1280@H/32} | **자체 촬영과 공개 다시점 50장면의 COLMAP PatchMatch MVS 깊이·법선 + 원본 사진**(3–4만 쌍), Blender 합성 1만 | 동결 SDXL 역전파가 지배: 6만 샘플 × 0.75 s | 13 (6 h) + MVS 35 | 교사 시간 +5% | 0.06 GB + 활성값 |
| **GeoHead** | 25M, DPT-lite, 동결 SDXL UNet up-block 특징 (t=200) | {1280@32², 640@64², 320@128²} → log 깊이, 법선, 불확실성 @H/2 | Blender/절차 합성 6만 (Poly Haven CC0 모델·HDRI, GT 깊이·법선), 실사 MVS 2만 | 특징 추출이 지배 | 44 (22 h) | UNet 포함 1024²에 0.4 s | 0.1 GB + UNet |
| **AesHead** | 5M, MLP on pooled SDXL 특징 | 특징 → 스칼라 (BT) | AVA 25만 (**라이선스 확인**) + 자체 쌍대 비교 5k | 특징 추출 11 T4-h | 15 (8 h) | 512²에 0.16 s | 작음 |
| **TexelFM** | 30M(비교용 8M/60M), conv-UNet + 저해상 attention, flow matching + shortcut 4 step | 부모 타일 320²×6 (log $L_d$ RGB, 높이, 법선2) + WNF ξ[4,320²] + 재질 emb + 스케일 $h_l$ Fourier → 자식 잔차 256²×6 | **CC0 PBR 스캔**(ambientCG, Poly Haven, MatSynth 중 CC0)을 실측 크기 메타데이터로 옥타브 피라미드화 + 자체 매크로 촬영 + solver W* 페이지(expert iteration) | 1.2×10¹⁸ | 26 (13 h) + 스케일 비교 20 | 텍셀 10⁸개에 60–90 s | 1–3 GB |
| **CamFM** ×3 | 15M + 5M + 8M | §4.2 | §4.2 | 약 10¹⁷ | 3 | §4.2 | 작음 |
| **GlyphSpotter** (선택) | 5M, 소형 segmentation | hero still → 텍스트 영역 마스크와 평면 | 폰트 합성으로 무한 생성 | 작음 | 2 | 30 ms | 작음 |
| **합계** | **약 125M** | | | **약 3×10¹⁸** | **약 130 + 데이터 준비 약 80** | | |

**TexelFM이 왜 핵심인가.**
- 교사가 최미세 옥타브를 맡으면 뷰가 4배 늘어난다. TexelFM은 같은 텍셀 수를 **약 8배 싸게, 비트 결정론적으로** 만든다(텍셀당 FLOPs 비교).
- 덕분에 **8K와 Powers of Ten이 컴퓨트가 아니라 구조 문제**가 된다.
- 의미가 있는 미세 디테일(얼굴, 소품, 간판 주변)은 연방제 규칙에 따라 교사에게 남긴다.

**30M 모델로 충분한 이유([조합], 개념).** 이것이 우리의 사전학습 대체다. 구조가 입력을 정규화한다.
- band-isometric 뷰로 footprint가 균일하다.
- 미터 단위 스케일이 주어진다.
- 조명은 intrinsics로 분리된다.
- 카메라를 이미 안다.

그래서 작은 모델이 보는 분포의 복잡도가 비디오 모델이 보는 것보다 훨씬 낮다. 불변성을 학습으로 얻지 않고 구조로 공짜로 얻는다.

### 5.2 교사 계약 (STRATA-lite가 요구하는 것)과 선택

**요구 조건.**
1. sm_75에서 fp16이 안전할 것.
2. T4에서 1024² step당 0.8 s 이하일 것.
3. **공간 정렬 latent(f8 권장)**일 것. WNF 적분과 대역 소유권은 latent 토큰 footprint가 균일하다는 가정에 기댄다.
4. σ 접근(오픈 가중치)이 될 것.
5. 상업적 사용 라이선스일 것.

**기본안: SDXL 계열 1개**(UNet 2.57B, VAE f8 4ch, CreativeML OpenRAIL++-M). Turing fp16에서 동작한 실적이 가장 많다. WSSC의 결정 창 안에서는 DDIM 큰 스텝 2–4회로 충분해서 **few-step 증류 LoRA도 필요 없다.**

**3일 bake-off 후보.**
- **Z-Image-Turbo**(6B, Apache-2.0, 8 NFE): 미감이 강하지만 T4에서 step당 약 4–6 s로 추정되고 fp16 안정성이 미확인이다.
- **SANA-1.6B**: 빠르고 4K 네이티브지만 DC-AE f32라 토큰 하나가 32×32 px다. 대역 결정 대부분이 디코더 안에서 일어나 소유권 이론과 충돌하고, 라이선스도 확인해야 한다.

**판정 지표는 "수용된 세계당 교사 호출 시간 × SP"다.** 한 모델만 쓴다는 규칙상 교체는 전체 교체만 허용한다.

**가벼운 어댑터.** GB-Adapter와 (선택) 자체 시네마틱 스틸 LoRA(rank 32, 5–10 T4-h)만 둔다. 학습 데이터는 우리가 직접 찍은 프레임이다.

### 5.3 [조합] σ-정밀도 스케줄

T4의 강점은 INT8 130 TOPS다. 노이즈가 큰 스텝(bedrock, coarse, adversarial 탐색)은 교사를 W8A8로, commit이 걸린 미세 스텝은 fp16으로 돌린다. 예상 이득은 교사 시간 1.3–1.6배 단축이다. timestep-aware 양자화 문헌의 결합이고, X-T0에서 미세 대역 SFR 손실이 2% 미만인지 확인한다.

---

## 6. 세계 1개의 값 (4K, 방이나 가게 앞 규모, Directed, 15초 경로)

| 단계 | 내용 | 교사 forward 수 | T4-h (순수) |
|---|---|---|---|
| S1 Hero search | 후보 128개 @512² 20 step APG(×2), 8 step 시점에 AesHead + Corman 비용으로 조기 탈락 → 상위 8개 @1024² SDEdit 12 step | 약 3,500 @0.16 + 190 @0.65 | 0.19 |
| S2 Bedrock | GeoHead lift + 매싱 + coarse 시네마틱 뷰 16개 × 4 step × CFG 2 (lockstep) | 128 | 0.05 (피팅 포함) |
| S3 Demand | Ray-Bundle + Corman–Shannon | 0 | ~0 |
| S4 Strata, 교사 레벨 ($l_g$+1..14) | 뷰 약 70개 × 3 step (l=14: 25–50, l=13: 7–13, 거친 레벨 약 10) | 210 | 0.05 |
| S4 TexelFM (l=15) | 텍셀 5×10⁷–10⁸ | 0 | 0.02 |
| S6 Audit/Polish | adversarial 카메라 100개 @512² + 2라운드 × 뷰 50개 | 200 | 0.03 |
| S7 Shoot planning | CamFM + CRM + Gaffer 200회 @768² | 200 | 0.03 |
| S8–9 Render | 4K24 15초 = 360 프레임 | 0 | 0.04 |
| **합계 (순수)** | | **약 4,400** | **약 0.41** |
| 공학 오버헤드 ×2.5–3.5 (파이썬, I/O, 재시도) | | | **1.0–1.4** |
| 벤치용 (Location-lite, 의도 12개 공통 Π) | S4·S6 ×3 | | 1.5–2.5 |
| 데모 히어로 (best-of-4 + 사람 선택 + polish) | | | 4–6 |
| 8K 추가 | TexelFM l=16 + 의미 셀에 한해 l=15 교사 | +180 | +0.3–0.5 |

v1 추정(4–12 H100-h ≈ 100–350 T4-h) 대비 약 1/100이다. 출처를 나눠 보면 다음과 같다.

- 교사 크기(12B → 2.6B): 약 5배
- Directed 수요: 3–5배
- 최미세 옥타브 TexelFM: 약 4배
- 렌즈 게이팅과 Corman: 1.3–10배

**대가는 미감 상한이 SDXL급이라는 것이다(§13 R1).** 대형 세계 예시: 10분 원테이크 집은 순수 약 2.5 T4-h, 오버헤드 포함 5–8 T4-h다. Powers of Ten은 옥타브마다 수요가 프레임 1장 분량뿐이라 교사 호출은 약 500회지만, 의미 정합을 위한 재시도까지 넣으면 5–15 T4-h다.

---

## 7. Phase 1 쇼케이스 재정렬 (2×T4 현실)

정렬 기준은 (구조적 승리 크기) ÷ (T4-h × 리스크)다.

| 순서 | 데모 | Day | 세계 | 세계 T4-h | 렌더 T4-h | 필요한 소형 모델 | 비디오 모델이 구조적으로 못하는 이유 |
|---|---|---|---|---|---|---|---|
| 1 | **왕복 v0**: 실촬영 방, 정방향·역방향 차이 맵 | 14 | 자체 촬영 + COLMAP + TSQ 피팅 | 1–2 | 0.05 | 없음 | 돌아오는 길에 세계를 다시 상상한다 |
| 2 | **리그 스왑 6종 × 룩 50종** | 21 | 1번과 같은 세계 | 0 | 0.5 (4K 6개 + 1080p 50개) | CamFM-Hand | 리그 개념이 없다. 테이크마다 세계가 달라진다 |
| 3 | **[신규] "Shot on 2×T4": 8K120 정지 세계 슬로모션** | 28 | 1번과 같은 세계 | 0 | 0.4–0.9 | 없음 | 네이티브 8K120을 낼 수 없다 (대부분 1080p, 4K는 업스케일) |
| 4 | **을지로 네온 (Corman 쇼케이스)**: 밤, f/1.4 랙 포커스, 한글 간판(폰트 SYMBOL이라 100% 정확), 보케 볼, 플레어 직선, 웅덩이 반사 | 45–55 | 생성 (WorldScript + SDXL) | 3–6 (best-of-4) | 0.3 (4K24 30초) | TL, GB-Adapter, GeoHead, AesHead, (GlyphSpotter) | 한글이 끓는다. 보케가 깊이와 무관하다. 플레어가 떠다닌다 |
| 5 | **왕복 v1 (생성 세계) + "하룻밤 1,000 테이크"** | 55 | 4번과 같은 세계 | 0 | 6.6 (1080p 10초 × 1,000 = 벽시계 3.3 h) + 4K 10개 | — | 1,000번 생성하면 1,000개의 다른 세계가 나온다 |
| 6 | **거울 복도와 초록 무한 터널** | 60 | 절차적 건축 + SDXL 텍스처 | 2–4 | 0.3 | TL, GB-Adapter, (TexelFM) | 거울 속 세계를 매번 다시 상상한다. 철분 때문에 초록이 되는 것을 모른다 |
| 7 | **경로 이식 릴 T4±** | 65 | 벤치 세계 | (벤치) | 0.2 | — | (공정한 A/B) |
| 8 | **1999 → 2026 불릿타임 헌정**: 와인잔이 깨지는 순간 정지 | 75–85 | Hero FX: Voronoi 파쇄 + FLIP 왕관을 t*에서 정지 (CPU 4–12 h) + 바 배경 (Corman 얕은 심도) | 4–8 | 0.5–1 (DIEL 레이마칭) | TL, GB-Adapter, AesHead | 정지를 요청하면 숨을 쉰다 |
| 9 | **체호프의 컵: 10분 원테이크** | 85–90 | 생성된 집, 방 5–8개, 스트리밍 | 5–8 (+데모 polish 4) | 1–2 (4K24 10분) | +TexelFM, CamFM-Intent | 컨텍스트 밖은 잊는다. extend할 때마다 drift가 쌓인다 |
| 10 | Powers of Ten (10⁵배 줌, 8K) | 90+ | 생성, 1차원 줌 수요 | 5–15 | 0.2 | TexelFM, 의미 효도 | 스케일 전환에서 녹는다 |
| 11 | 직접 날아보기 (웹 뷰어) | 60+ | export (메쉬 + BC7) | 0 | 0 (시청자 기기에서 렌더) | — | 상호작용 자체가 불가능하다 |
| 12 | **[신규] Corman 슬라이더**: 같은 장면을 예산 0.1–10 T4-h로 | 70 | 4번, 8번 재사용 | 2 | 0.2 | — | (우리만의 메시지) |
| 13 | 빛만 흐르는 세계 / 매직아워 | 90+ | intrinsics 필요 | +3 | 0.3 | GeoHead 법선 | 조명 상태가 없다 |

**뒤로 미루는 것.**
- bullet-time 속 사람 8개와 군중 얼굴은 Phase 2c 근처로 미룬다. 교사 1개로 다시점 정체성을 지키는 일은 리스크가 크다.
- Vision Pro 양안 광택은 싸다(두 장이 좌우 눈을 하나씩 렌더). 그래서 Day 90 이후 P1-확장으로 남긴다.

---

## 8. FREEZE-TAG on 2×T4

### 8.1 운영 원칙

1. **세계 1개가 의도 12개를 다 찍는다**(Location-lite Π₁₂). 우리 생성 비용은 프롬프트 수에 비례하고, 경쟁사는 프롬프트 × 의도 × seed에 비례한다.
2. **T1·T2는 텍스트(또는 첫 프레임) 전용 모드로 돌린다.** 프롬프트 → SDXL hero search → GeoHead lift → WSSC-lite 순서다. **WorldScript나 사람 개입은 금지한다.** 데모는 Directed 모드라고 따로 표기한다.
3. **Corman on/off를 둘 다 보고하고, 자해 카테고리(불, 연기, 털)도 공개한다**(v1 원칙 유지).
4. **평가 모델은 평가 전용이다.** VGGT·UFM(GeCo), MEt3R(DUSt3R + DINO), LPIPS, OCR(TPR), ArcFace를 T4에서 돌린다. VGGT는 bf16으로 학습된 모델이라, fp16 autocast에서 overflow가 나면 fp32로 돌린다(1.2B × 4 B = 4.8 GB, 들어간다).
5. **T4± 경로 추출은 고전 BA(COLMAP)를 먼저 쓴다.** 실패하면 평가 전용으로 VGGT를 쓰되 벤치 트랙에만 한정한다.

### 8.2 비용

| 항목 | 미니 벤치 (Day 90) | 전체 벤치 (v1 규모, Day 90 이후) |
|---|---|---|
| 구성 | 2범주 (C1 "밤의 실내·거리 + 텍스트·네온", C2 "건축 실내 + 거울·크롬") × 40 프롬프트 × 의도 4 (push-in, orbit, 왕복, 핸드헬드 워크) × seed 2 | 240 프롬프트 × 의도 12, 약 500쌍 + bullet-time |
| 우리 세계 | 80 × 2 T4-h = **160 T4-h** (2×T4로 약 3.5일) | 240 × 2.5 = **600 T4-h** (약 12.5일) |
| 우리 렌더 | 클립 640개 1080p + 4K 마스터, 약 2 T4-h | 약 10 T4-h |
| 경쟁사 API | Seedance 2.5, Kling 3.0, Wan 3.0 × 1,920클립 × 8초 = 15,360초 × $0.05–0.4/s = **$0.8–6.1k** | 6개 모델 × seed 4 → **$5–38k** (가격 확인 필요) |
| GeCo 등 평가 | 약 2,560클립 × 30 s = **약 21 T4-h** | 약 100 T4-h |
| 사람 평가 | 3,000쌍 × 3명: 크라우드 $0.5–1.5k + 전문가 15명 × 4 h에 $3–6k | v1 $50–80k |

### 8.3 단계 목표 (최종 승리 조건은 v1 §7.1 그대로)

| 시점 | 목표 | v1 대비 |
|---|---|---|
| **Day 30** | X2 파일럿(자체 촬영 10장면): VP 일반 ≥ 60%(예비). LCE 정확히 대칭인 경로 = **비트 동일**. 4K 프레임 ≤ 0.6 s/T4. X0-lite 스펙트럼 블로그 | 렌더 성능 게이트 신설 |
| **Day 60** | X2 VP ≥ **70%** (v1 그대로). C1 생성 세계 20개: **GeCo ≤ 0.04, HER ≥ 0.8, OIE ≤ 0.05** (v1 그대로), SP ≥ 35%(중간 목표 신설). 세계당 ≤ 3 T4-h | SP 중간 목표 추가 |
| **Day 90** | C1에서 **SP ≥ 45%와 GeCo ≤ 0.03을 동시에** (v1 그대로). T2에서 C1 최강 경쟁 모델 대비 VP ≥ 70% (v1 그대로). 미니 벤치 사전 등록. 데모 릴 3+1편(왕복, 을지로, 불릿타임 + "Shot on 2×T4"). X-C 결과 | 범주를 C1으로 명시 |
| **최종 (불변)** | SP ≥ 50% (vs Wan 3.0, Seedance 2.5) **이면서** GeCo ≤ 0.03. TP ≥ 45%. VP T1 65/75%, T2·T4± 80%, T3 90%. TPR ≥ 0.99, MAT ≥ 90%, CI ≤ 실사 + 5%p. MTF50@8K ≥ 0.25. 카메라 튜링 ≤ 55% | **낮추지 않는다** |

---

## 9. ★ 제약이 무기가 되는 지점

| 메시지 | 근거 (숫자) | 데모 / 포맷 |
|---|---|---|
| **결정론은 컴퓨트가 아니라 정의에서 나온다** | 왕복 비트 동일성의 추가 비용이 0이다 | 차이 맵: 상대는 크리스마스 트리처럼 빛나고 우리는 새까맣다 |
| **재촬영의 한계비용은 렌더뿐이다** | 1080p 10초 테이크가 벽시계 15초, 4K는 약 1분 | "하룻밤 1,000 테이크, 세계는 하나" 모자이크 |
| **8K·120fps는 구조에서 나온다** | 생성 비용은 fps와 무관하다. 8K는 TexelFM 옥타브 하나만 더한다(+0.3–0.5 T4-h) | "Shot on 2×T4" 8K120 릴 |
| **[신규] 70W 촬영장** | 4K120 10초 ≈ 12 Wh(스마트폰 1회 충전), 세계 하나 ≈ 140 Wh(노트북 2회 충전) | "헤어드라이어 1/11로 찍었습니다" |
| **[신규] Proof-of-Shot** | Merkle root, ShotScript, toolchain pin을 공개하면 재렌더 결과가 바이트 단위로 같다 | 모든 영상에 해시와 "직접 재렌더하기" 링크 |
| **세계 파일을 그대로 배포한다** | 4K 세계 0.2–0.6 GB. 시청자 기기가 렌더하므로 우리 서빙 비용은 0 | "믿지 마세요. 직접 날아보세요." |
| **가난이 미장센이 된다** | Corman: 대형 세계에서 3–10배 절감, SP는 비열등(X-C로 검증) | Corman 슬라이더, 로저 코먼 오마주 |
| **세트는 밤에 짓고 아침에 찍는다** | 세계 1개 30–60분, 하룻밤에 10–16개 | 스튜디오 리듬 자체가 영화 촬영 일정과 같다(dailies) |
| **반복 속도** | 실험 하나가 1시간이라 하루 20회 이상 반복한다 | 블로그 "작은 팀의 OODA 루프" |
| **인수분해가 생존 조건이 된다** | 작은 모델은 정규화된 국소 입력에서만 통한다 → 대역별 `git blame`으로 시스템을 감사할 수 있다 | 기술 보고서 |

---

## 10. 90일 (이 렌즈) + T4-h 예산

### 10.1 주차별 계획

| 주 | 표현·렌더러 | 카메라 | 데이터·평가 | 게이트 |
|---|---|---|---|---|
| 1 | **X-T0**: SDXL s/step(PyTorch, TRT, INT8), gsplat sm_75 빌드, OptiX, NVENC 8K, fp16 안정성. STRATA-lite v0 (블록, TSQ, BC7) | 카메라 스택 v0 (thin-lens 공분산, 셔터, 센서, LUT) | 자체 촬영 시작 (HDR 브래킷), GoPro 손 데이터 수집 | — |
| 2 | COLMAP/GLOMAP → TSQ 피팅, dual-cut | 리그 6종 1 kHz 시뮬 | TL-2.5M 학습 | **D14 왕복 v0** |
| 3 | Defocus Pyramid, LH-split, 출사동 LUT, Hullin, FFT 글레어 | Hand 학습 | X0-lite (경쟁 클립 + 고전 광류) | — |
| 4 | Ledger, Proof-of-Shot, 8K 경로 | 리그 스왑 릴 | X2 파일럿, FREEZE-TAG 하네스 | **D30** |
| 5–6 | WSSC-lite 연동 (band-iso 뷰, WNF 래스터 패스, 소유권 저장), PSH (SYMBOL 한글 MSDF, POINT, CURVE) | Intent 학습, Corman v0 | GB-Adapter, GeoHead 학습 | — |
| 7–8 | 평면 재귀 거울, OptiX secondary | Gaffer, CRM 쌍대 수집 | X2 본실행, C1 생성 세계 20개 | **D60** |
| 9–10 | DIEL SDF + 분산, Hero FX 연동, 스트리밍 | DP-0-lite 탐색 | TexelFM 학습 (8/30/60M), **X-C** | — |
| 11–12 | 4K/8K 최종 렌더, 웹 export | 데모 경로 | 미니 벤치 80세계, 사람 평가 | — |
| 13 | — | — | 기술 보고서, 벤치 사전 등록, 릴 3+1편 | **D90** |

### 10.2 예산

| 항목 | 상시 (T4-h) | Kaggle (T4-h) |
|---|---|---|
| X-T0, 렌더러 개발·프로파일링 | 60 | 30 |
| 소형 모델 학습 | 130 | 70 (TexelFM 스케일 비교 생략) |
| 데이터 준비 (MVS, 특징) | 80 | 0 (로컬 CPU) |
| X1-lite 교사 보정 (500장 × t 15 × seed 4) | 20 | 10 |
| 실촬영 Oracle 세계 30개 | 45 | 25 |
| 개발용 생성 세계 약 250개 | 375 | 180 |
| 미니 벤치 세계 | 160 (80개) | 80 (40개) |
| 데모 히어로 세계 | 50 | 40 |
| X-C Corman 실험 | 120 | 50 |
| 렌더 (데모, 1,000 테이크, 50 룩, 벤치) | 80 | 40 (8K EXR 없음) |
| 평가 | 60 | 30 |
| 실패 버퍼 (약 30%) | 360 | 170 |
| **합계** | **약 1,480 / 가용 약 3,200** | **약 725 / 가용 약 780** |

상시 운용이면 남는 약 1,700 T4-h를 세 곳에 쓴다.
- TexelFM 스케일링 연구
- **장면별 WDM 실험**(v1 X3 arm vi), 장면 2–3개 × 50–100 T4-h
- 전체 벤치 세계를 미리 생성

Kaggle 운용이면 계획대로 가되 여유가 0이다. 그래서 Day 90 범위를 C1 한 범주로 고정한다.

---

## 11. v1에서 버리는 것, 미루는 것

| v1 항목 | 처리 | 이유 |
|---|---|---|
| H100 64장, 0.4–0.8M H100-h | **폐기** | 제약 |
| CWM 4–8B + LBO 사전학습 (10²³) | **폐기** → TexelFM 30M + 세계별 solver | 10²³ ÷ 2.2×10¹⁸/일 ≈ 125년 |
| 장면 규모 amortized WDM 학생 | **연기** → 장면별 WDM 실험 2–3개만 | 2.6B 교사를 통과하는 역전파 + fake score로 장면당 50–100 T4-h |
| LLM World Program | **폐기** → 텍스트 모드는 교사 hero search + GeoHead, 데모는 WorldScript DSL | 대형 LLM 금지 |
| 닫힌 SOTA hero still, GAN real 집합 | **폐기** → SDXL hero search + AesHead | 시스템 경로에 둘 수 없음 |
| DA3/VGGT-Ω/AnySplat/MegaSaM | **폐기** → COLMAP/GLOMAP + MVS, GeoHead | 금지 |
| SelfCaption VLM, SigLIP/DINO `sem` | **폐기** → 구조화 캡션, class id + SDXL 텍스트 임베딩 | 금지 |
| RAFT/UFM 보정 광류 | DIS/TV-L1로 교체 (평가 전용 사용은 허용) | 금지 |
| TRELLIS/Hunyuan3D object prior | **폐기** → 절차적 에셋, CC0 모델 | 금지 |
| latent `z` 정본, 30–60배 압축 | **연기** (슬롯만 예약) → explicit TSQ + BC7 | 디코더(학생)가 없고, BC7 하드웨어 디코드가 공짜 |
| 3DGRUT 단일 렌더러 (3DGRT primary) | **변경** → TSQ 래스터 + OptiX(선택 픽셀) + 평면 재귀 | T4의 RT 처리량, Turing any-hit 비용 |
| MC 16–64 spp DOF 타일 | **변경** → Defocus Pyramid (MC는 레퍼런스 검증용) | 4K에서 프레임 비용이 3배를 넘는다 |
| NRC, 경로추적 GI | **폐기** (Phase 1) | RAS로 구움 |
| Location 모드 기본값 | **변경** → Directed 기본, 벤치는 Location-lite Π₁₂ | 비용 3–5배 |
| X3의 6개 arm | 축소: (i) 평균 재구성 baseline (SEVA는 평가 baseline으로만), (v) WSSC-lite, (vi) WDM 장면 1–2개 | 컴퓨트 |
| 사람 bullet-time, 군중, 아바타 | **연기** (2c 근처) | 교사 1개로 다시점 정체성 유지가 어렵다 |
| Powers of Ten | Day 90 이후로 유지 (비용은 생각보다 싸다. 리스크는 의미 정합) | — |
| PIS 2.0 VDP 위험 배분 | **연기** | — |
| 실시간 뷰어, VR | **앞당김** | T4는 렌더 카드이고, 웹은 시청자 기기가 렌더한다 |

---

## 12. 신규성 장부 (이 렌즈)

| 확신도 | 항목 |
|---|---|
| **[신규] 높음** | **Corman–Shannon Demand + Corman Optimizer**: 카메라 채널 용량으로 생성 수요를 정하고, DP 목적함수에 T4-시간 비용을 넣는다. 저예산 촬영 기법이 비용-미감 Pareto 전선 위에 있다는 가설 |
| **[신규] 중상** | TSQ를 T4에 맞춘 세계 프리미티브로 쓰는 것(BC7, RT 삼각형, 엔진 export를 함께 노린 설계). Proof-of-Shot. 거울 터널의 철분 초록을 생성 비디오 포렌식에 쓰는 것. 벤치에서 "세계 1개 = 의도 12개" 상각을 비용 회계로 정식화한 것 |
| **[조합]** | Longuet-Higgins Split 렌더러 (회전 = 호모그래피 정확 적분, 1 kHz 리그 결합). Defocus Pyramid + 렌즈 게이팅. Judgment Ledger (교사 판정의 내용 주소 메모이제이션, 세션 선점에 강함). σ-정밀도 스케줄 (INT8/FP16). "구조적 정규화가 곧 사전학습 대체"라는 소형 모델 논증. Blender 오픈무비 카메라를 DP 정답 라벨로 쓰는 것 |
| **[부품]** | gsplat, 3DGUT, 2DGS, BBSplat, Textured Gaussians, OptiX, Kolb'95, Hullin'11, Newson'17, Malvar–He–Cutler, 평면 반사 재귀, Kraus–Strengert, TAESD형 AE, T2I-Adapter, DPT, COLMAP/GLOMAP, CC0 PBR 데이터, BC7/BC5 |

---

## 13. 리스크와 업그레이드 (T4 고유)

| # | 리스크 | 감지 | 업그레이드 (후퇴 없음) |
|---|---|---|---|
| R1 | **SDXL급 미감 상한 때문에 SP가 부족하다** (이 렌즈의 최대 리스크) | D60 SP < 35% | 범주를 C1에 집중, Corman, hero search N을 256으로, DTC + 필름 스택, 자체 시네마틱 LoRA. 그래도 부족하면 교사 bake-off로 **전체 교체**(Z-Image-Turbo INT8). 일관성은 정의로 이기고, 미감은 교사 위에 광학을 얹어서 이긴다(TP ≥ 45%) |
| R2 | T4 래스터가 추정보다 느리다 (4K·8K fill-rate) | X-T0 | 타일 컬링, Defocus Pyramid 강화, AOV 축소. 렌더는 임계 경로가 아니므로 밤에 돌린다 |
| R3 | fp16 overflow (교사, 우리 모델) | NaN 비율 | TL로 VAE fp16 문제를 우회, loss scaling, 문제 레이어만 fp32 |
| R4 | 피팅 중 16 GB OOM | 프로파일링 | Adam 모멘트를 CPU로, 블록 스트리밍, 뷰 batch 1, TexelFM 레벨은 Adam 0 |
| R5 | NVLink 없음, PCIe Gen3 | 장 간 대기 시간 | 공간 분할 Jacobi (장 간 트래픽은 apron과 해시뿐) |
| R6 | Kaggle 쿼터와 세션 상한 | 주간 잔량 | Ledger, Merkle 커밋으로 정확히 재개. 우선순위는 Oracle → 을지로 → 미니 벤치 40 |
| R7 | TexelFM이 의미 있는 미세 디테일에 실패한다 | 2AFC > 60% | 연방제 라우팅: 의미 셀은 교사, 재질 셀만 TexelFM. 60M으로 확장 |
| R8 | GeoHead 깊이 오류 | free-space veto 비율 | DSL 매싱 prior, 다시점 lockstep 교정, 실사 MVS 데이터 확대 |
| R9 | T4에서 OptiX 성능이나 동작이 부족하다 | X-T0 | 거울은 평면 재귀로, 유리는 SDF 레이마칭으로. RT는 히어로 픽셀만 |
| R10 | 아키텍처가 다르면 비트 동일성이 깨진다 | 교차 GPU diff | forward 결정론 커널, toolchain pin, 보장 범위를 "같은 아키텍처 바이트 동일, 다른 아키텍처 ≤1 LSB"로 명시 |
| R11 | 라이선스 (SDXL OpenRAIL++, AVA, DL3DV, ScanNet++, Hypersim, MatSynth 일부 CC-BY) | 법무 게이트 | 상업 경로는 자체 촬영, CC0, Blender 합성 우선 |

---

## 14. 다른 렌즈와 리서처에게

**다른 렌즈에 주는 인터페이스.**
- 세계 엔진(solver) 렌즈에 주는 것: `render(view, level_mask, θ)`(미분 가능, 최종 렌더와 같은 코드), `BP`, 소유권·저술권 저장, WNF 주소(Philox 래스터 패스), Ledger, Corman–Shannon 수요 집합.
- 세계 엔진 렌즈에 요청하는 것: WSSC-lite의 view-step 예산을 §6 표 범위에 맞춰 달라.
- 데이터·소형 모델 렌즈와는 TL, GB-Adapter, GeoHead를 공유한다. 중복 학습을 막기 위해 소유자를 한 명으로 정하자.
- Phase 2 렌즈에 주는 것: TSQ에 slot과 inst가 있다. 고전 시뮬(MPM, FLIP, 강체)의 상태로 atom을 변형하는 훅, DIEL과 Hero FX를 공유한다. 정지 슬라이스는 그 자체로 Phase 2a(Frozen Action)의 렌더 경로다.

**리서처 검증 요청.**
1. T4 지속 실효 TFLOPS (GEMM, 소형 모델 학습)
2. T4에서 SDXL s/step (PyTorch, TensorRT, INT8)
3. gsplat sm_75 빌드와 3DGUT 사용법
4. 3DGRUT/OptiX의 T4 동작 여부와 ray 처리량
5. Turing NVENC 8K HEVC 10-bit 인코드
6. Kaggle 2×T4의 RAM, 디스크, 세션 상한, 쿼터 (공식 문서에 없고 계정마다 다름)
7. 각 경쟁 API의 초당 가격
8. 라이선스: SDXL, AVA, MatSynth, DL3DV, Blender 오픈무비, Z-Image, SANA
9. VGGT를 T4 fp16으로 돌릴 때 안정성

---

> **맺음말.** 비디오 모델은 같은 세계를 240번 다시 그린다. 우리는 그 세계를 한 번만 짓고, 그 한 번마저 카메라가 기록할 수 있는 만큼만 짓는다. 8K, 120fps, 재촬영, 왕복은 렌더 문제라서 GPU 개수와 상관이 없고, 렌더는 T4가 원래 잘하는 일이다. 비싼 것은 교사의 판단 하나뿐이고, 그 판단은 Ledger에 남겨 같은 질문을 두 번 하지 않는다. 남는 부족분은 연기 기계와 f/1.4 렌즈로 메운다. "개소리 같지만 해낼 거야"를 숫자로 바꾸면 세계 하나에 1–2 T4-h, 4K120 10초 렌더에 스마트폰 한 번 충전할 전기다. 그리지 말고, 찍어라.

**Sources (이번 검증 검색):** [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut), [3DGUT (arXiv 2412.12507)](https://arxiv.org/html/2412.12507v2), [NVIDIA 3DGUT 블로그 (gsplat 통합)](https://developer.nvidia.com/blog/?p=99680), [NVIDIA Tesla T4](https://www.nvidia.com/en-eu/data-center/tesla-t4/), [Baseten SDXL 최적화](https://www.baseten.co/blog/sdxl-inference-in-under-2-seconds-the-ultimate-guide-to-stable-diffusion-optimiza), [JarvisLabs SDXL GPU 가이드](https://jarvislabs.ai/ai-faqs/best-gpu-for-stable-diffusion), [Free Cloud GPUs in 2026 (Kaggle 한도)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [aimultiple free cloud GPU](https://aimultiple.com/free-cloud-gpu)

참조 파일: docs/PARADIGM.md, docs/lenses/02-representation-strata.md, docs/lenses/03-cinematography.md, docs/lenses/premortem/02-representation-v0.2.md, docs/research/landscape-2026-10.md
