> **출처 메모.** 설계 워크숍에서 [이론가 / first-principles] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 건설적 프리모템과 개정안(v2)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../../PARADIGM.md)의 판결록(§0.5)을 따른다.
>
> **주의: 앞부분이 유실된 문서다.** 워크숍 출력이 §4의 비용 표 끝부분부터만 남았고, 그 앞(실패 지점과 업그레이드를 다룬 앞 장들)은 없다. 첫 줄이 표 중간에서 시작하는 것은 그 때문이다. 여기서 처음 나오는 용어(Partition Principle, World-Scale Causality, Frozen-Seed Band-Commit, SMF, WDD 등)는 `PARADIGM.md`가 남은 본문을 근거로 정의를 확정했다.

---

더 | 0.1~0.3 |
| **합계** | | **약 1.3~2.3** |

## 5. Spectral Federalism v2: 측정된 관할, prior-agnostic

### 5.1 Phase 1 관할 지도 (채널 × 세계 대역)

| 채널 \ 대역 | DC (전역) | L0 레이아웃 | L1 형태·실루엣 | L2 재질·중간 텍스처 | L3 미세 (≤ latent Nyquist) | L4 극미세 (> latent Nyquist) |
|---|---|---|---|---|---|---|
| 기하 | scene graph (LLM) | 단일 캔버스 lift + 3D object prior | dreamer depth head (partition, 한 옥타브 선행) | $f_{\text{par}}$까지만 기하, 나머지는 normal | dreamer normal | WDD normal detail |
| 외관 | 팔레트, 시간대 | 단일 캔버스 | 이미지 prior (partition) | 이미지 prior + intrinsic dreamer | 이미지 prior, polish에서는 API dreamer | WDD (seeded) |
| 확률적 물질 | matter class 분포 | 캔버스 segmentation | SMF 통계 투표 ($\tau$) | $G_{SM}$ 인스턴싱 | $G_{SM}$ | $G_{SM}$ LOD seed |
| 시점 의존 | env, sun | env fit | 3DGRT로 $W$를 반사 | intrinsic dreamer의 roughness | — | — |
| 사진 | 렌즈 패키지 | Camera·Sensor만 (dreamer는 `<plate>`) | 〃 | 〃 | 〃 | 〃 |
| 조명 강조 | 스타일 | Gaffer (샷 단위) | 〃 | 〃 | — | — |

### 5.2 관할 배정 규칙

관할은 사람이 정하지 않고 측정으로 정합니다.
$$
\mathrm{owner}(\text{cell})=\arg\min_P\ \Delta_P(\text{cell})+\mathrm{Var}_P(\text{cell})
$$

- **Worldness Gap.** 대역 $b$, 노이즈 $\sigma$에서 prior $P$가 얼마나 세계처럼 일관되게 그리는지를 잽니다.
  $$\Delta_P(\sigma,b)=\mathbb E\|M_{ij}\odot\Pi_b(\mathrm{warp}_{i\to j}\mathcal D^P_\sigma(x_i;\xi)-\mathcal D^P_\sigma(x_j;\xi))\|$$
- **Calibration 데이터.** 3DGS 렌더가 아니라 **실사 프레임**을 씁니다(ScanNet++ DSLR + laser scan, DL3DV 원본 프레임 + COLMAP pose). 렌더를 쓰면 dreamer의 jitter가 아니라 재구성 아티팩트를 측정하게 됩니다.
- **Dreamer 계약.** 어떤 prior든 아래를 제공하면 연방에 들어올 수 있습니다.
  - (i) 조건부 샘플: 렌더 또는 노이즈 상태, 프롬프트, G-buffer를 받아 샘플을 냅니다.
  - (ii) calibration 프로파일: $\Sigma_u$(flow로 측정한 jitter), $\Sigma_\eta(f)$(잔차 스펙트럼), $t_b$, $\rho(f,\sigma)$
  - (iii) 선택 사항: σ 접근권. σ에 접근할 수 있는 prior만 BIRTH에 참여합니다.

### 5.3 Prior-agnostic 소유권

- 닫힌 SOTA 이미지 모델(edit API)은 노이즈를 고정할 수 없어서 $\rho\approx0$입니다.
- **Partition Principle에 따르면 소유권은 $\rho$가 필요 없습니다.** 그래서 이런 모델도 POLISH의 L3 owner 후보로 들어올 수 있습니다.
  - 절차: 렌더 → edit → flow로 $u$ 정렬 → gauge 제거 → 대역별 후보
- 결과적으로 그달 가장 좋은 이미지 모델을 학습 없이 바로 씁니다. 단, 이용약관은 확인해야 합니다.
- 기본 라이선스 스택은 Qwen-Image(Apache-2.0)와 HiDream-I1(MIT)입니다. FLUX.1-dev는 연구 arm에만 씁니다.

## 6. 카메라와 사진

### 6.1 Kinetic Camera
$$
M\dot v=u_{\text{intent}}(t)+F_{\text{rig}}(s)-Dv+\xi_{\text{op}}(t),\qquad \mathrm{PSD}(\xi_{\text{op}})=\Phi_{\text{rig}}(f\mid \text{focal},\text{intent},\text{genre})
$$

| Rig | 지문 |
|---|---|
| Handheld | 호흡 0.2~0.3Hz + 1~6Hz 피크. 진폭은 초점거리에 반비례하도록 조건부 |
| Steadicam | 0.5~1Hz 감쇠 진자 |
| Dolly / Slider | jerk ≈ 0, ease in/out, 레일 미세진동 |
| Crane / Jib | 붐 관성, 멈출 때 overshoot와 settle |
| Drone / FPV | 저역 돌풍, gimbal 잔차, bank-turn 결합 |

- **Intent prior.** trajectory diffusion을 DataDoP(29K 샷), CameraBench taxonomy, 영화 클립에서 MegaSaM/ViPE로 추출한 궤적으로 학습합니다. 조건은 shot language와 $W$의 3D saliency입니다.
- **Cinematography as Search.** 세계가 고정돼 있으니 카메라 경로를 탐색할 수 있습니다. 보상은 샷 단위입니다.
  $$\sum_t[\mathrm{Aes}+\mathrm{Comp}+\mathrm{Parallax}]+\log p_{\text{intent}}(\Gamma)-\lambda\,\mathrm{CWS}-\mu\,\mathrm{TPI}$$
- **재촬영(re-shoot)이 제품 기능입니다.** 같은 세계에서 테이크를 여러 번 찍고, 샷 간 매치컷도 만듭니다.

### 6.2 Lens와 Sensor ($\mathcal S$)
- **렌즈:** thin-lens 또는 anamorphic DOF(깊이가 metric이라 물리적으로 정확), cat-eye bokeh, lens breathing, 비네팅, flare
- **셔터:** 180° 규칙, rolling shutter
- **센서:** OLPF, read/shot noise, halation, grain(프레임별 seed), LUT 또는 톤 커브
- 이 효과들은 **오직 여기서만** 생깁니다. dreamer는 `<plate>` 모드로만 그립니다.

### 6.3 Gaffer
- 샷마다 조명 세트를 최적화합니다.
  $$L^\*=\arg\max_L\ \mathbb E_t[\mathrm{Aes}(x_t)-\lambda\,\mathrm{DR_{rel}}(x_t)]$$
- 렌더는 deferred shading입니다.
  $$L_o=L_{\text{plate}}+\sum_l f_r(a,\rho,n;\omega_l,\omega_o)L_lV_l$$
- 조명은 샷 안에서 고정되므로 T1이 유지됩니다. 실제 촬영처럼 셋업마다 조명을 다시 칩니다.

### 6.4 Temporal-Spectral Renderer
- StopThePop(미리보기)과 3DGRT/3DGUT(최종)
- band-fade LOD: $w_b(\phi)=\mathrm{clamp}(\log_2\frac{f_{\text{Nyq}}}{f_b\phi},0,1)$
- OLPF σ ≈ 0.35px, stratified 32~64 spp

## 7. Phase 2: 시간 축 열기

### 7.1 Lagrangian World
$$
W(t)=\Big(\{g_i\}_{\text{canon}},\ T_o(t)\circ\mathrm{LBS}_{\theta(t)}\circ\textstyle\sum_mB_{im}c_m(t),\ a_i+\Delta a_i(t),\ \tau_c(t),\ E(\mathbf x,t)\Big)
$$
- **외관.** Phase 1 엔진이 canonical에 bake합니다. $\Delta a$는 low-rank이고 event로 시작되는 재질 변화(젖음, 그을림, 파손)로만 허용합니다. 그래서 morphing이 표현 단계에서 막힙니다.
- **사람.** SAM 3D Body와 SMPL-X로 기하를 잡고, 얼굴은 인스턴스 잠금으로 owner를 하나만 둡니다.
- **확률적 물질의 동역학.** seeded procedural sway를 씁니다. 바람장 PSD는 비디오에서 측정하고, 파티클은 물리로 움직입니다.

### 7.2 Event-first Time
1. Event set $\mathcal E=\{(\tau_e,\text{type},\text{participants})\}$을 point process prior에서 먼저 샘플링합니다.
2. 그다음 hybrid dynamics를 풉니다. 접촉 순간에는 $\Delta v_n=-(1+e)v_n$ 같은 속도 불연속이 들어갑니다.
3. 결과적으로 $\mathbb E[H(t-\tau)]=F_\tau$ 문제(타이밍 hedging으로 계단이 시그모이드로 뭉개지는 현상)가 원인 수준에서 사라집니다.

### 7.3 시간 연방 지도

| 공간 \ 시간 | 저 (0~1Hz) | 중 (1~8Hz) | 고 (>8Hz) |
|---|---|---|---|
| 레이아웃 | 비디오 prior (궤적 공간 distill, 저해상도) | Lagrangian motion prior | event / 물리 |
| 형태 | motion prior (skeleton, LBS, low-rank) | motion prior | 물리 (contact, MPM/NGFF) |
| 외관 | 이미지 prior (canonical) + event 재질 | 렌더러 (모션블러 적분) | 렌더러 / 센서 |
| 확률적 물질 | 바람장 (비디오 prior) | SMF sway (seeded) | 파티클 물리 |
| 카메라 | intent prior | rig 동역학 | operator PSD |

### 7.4 데이터와 지표
- **Motion prior 학습 데이터.** SpatialVID, Stereo4D, OmniWorld를 MoVieS, VGGT-Ω, Shape of Motion, CoTracker3/TAPIP3D로 4D 재구성합니다. 그 위에 Lagrangian motion diffusion을 학습합니다.
- **Phase 2 지표**
  - Identity Drift: canonical과 t초 시점의 ArcFace/DINO 유사도(10초, 60초)
  - Event Sharpness Index: 접촉으로 생긴 변화의 10~90% rise time. 실사는 24fps에서 1~2프레임입니다.
  - VBench-2.0 Physics
  - 정적 부분의 GeCo
  - Elo

## 8. 학습 신호와 플라이휠

| 부품 | 학습 신호 | 데이터 |
|---|---|---|
| Dreamer calibration | $\Sigma_u$(RAFT/UFM flow), $\Sigma_\eta(f)$, $\rho(f,\sigma)$, $t_b$ | ScanNet++ DSLR, DL3DV 원본 프레임, Infinigen |
| `<plate>` LoRA | diffusion loss + 속성별 negative CFG | sharp·deep-DOF 프레임 약 5만 장 + 합성 약 2만 장 |
| RGB-D/normal head | Marigold식 fine-tune | Hypersim, OmniWorld, DL3DV(DA3 pseudo-GT) |
| 불확실성 head $\hat\sigma^2$ | seed 8개 사이 픽셀 분산을 회귀 | 위와 동일 |
| RE-FT (ρ가 낮을 때만) | $\mathcal L_{\text{eq}}+\mathcal L_{\text{spectral}}+\mathcal L_{\text{adv}}$ | anchored noise 쌍 |
| EQ-VAE + denoiser 적응 | $\mathcal L_{\text{eq}}$(local homography) + LPIPS + spectral | 일반 이미지 |
| WDD | 저역 LPIPS 일치 + 고역 adversarial/spectral | 실사 crop(footprint를 맞춤) |
| $G_{SM}$ | $(\tau,\text{instance})$ 조건부 point-set flow | Infinigen, 파티클, VDB |
| Intent prior / rig PSD | trajectory diffusion, PSD 피팅 | DataDoP, CameraBench, 영화 클립 |
| Aesthetic judge | Q-Align 계열 + DR + 구도 | — |

### 플라이휠 (Expert Iteration)
- **역할 분담**
  - Judge: 이미지 prior와 연방 prior들
  - Search: WSSC v2
  - Student: 두 개입니다.
    - **World Layout Flow:** 단일 캔버스에서 coarse $W$를 만듭니다.
    - **World Detail Flow:** octree structured latent입니다. TRELLIS 계열을 장면 규모로 확장한 것입니다.
- **루프**
  1. $W_0\sim p_\phi$
  2. 세계 공간 SDEdit으로 $\sigma_{\text{mid}}$에서 WSSC v2를 시작
  3. $W^\*$ 산출
  4. $(y,W^\*)$로 $\phi$ 학습
  5. 반복
- **후처리 학습.** 이미지 teacher로 DMD2 분포 매칭을 하고, 카메라는 경로 근처에 중요도를 두고 샘플링합니다. WSSC의 $W^\*$는 fake-score의 real anchor로 써서 mode collapse를 막습니다.
- **래칫 지표.** search gain $=\mathrm{DR_{rel}}(\text{student})-\mathrm{DR_{rel}}(\text{WSSC})$가 줄어들어야 하고, polish 반복 수도 줄어야 합니다.

## 9. 평가: FrozenBench v0

**프로토콜**
- 프롬프트 40개 × {10s orbit, 30s loop-return, 60s 원테이크(매크로↔와이드)} + bullet-time 사람 8개
- 트랙 두 개: T2V와 I2V(우리 hero 프레임을 첫 프레임으로 제공)
- 경쟁 모델은 seed 3개 중 중립 평가자가 고른 best-of-3를 씁니다.

| 지표 | 정의 | v2 목표 |
|---|---|---|
| GeCo fused | $\sqrt{M\,e_m^2+e_s^2}$, UFM flow vs VGGT rigid flow | ≤ 0.04 (실사 바닥 0.02) |
| MEt3R | DUSt3R warp feature 유사도 | 실사 수준 |
| LCE | 닫힌 경로 첫·끝 프레임의 LPIPS/PSNR | PSNR ≥ 35dB |
| FV | 비강체 잔차 > 1px인 track 비율 | ≈ 0 |
| $\mathrm{SFR_{prior}}(f)$ | $P_{\mathcal R(W,C)}(f)/P_{\text{dream}(C)}(f)$ | 전 대역 ≥ 0.9 |
| $\mathrm{SFR_{real}}(f)$ | 같은 클래스 실사 대비 radial power | 0.9~1.1 |
| $\mathrm{DR_{rel}}$ | 다중 σ dream residual / 실사 중앙값 (held-out seed) | ≤ 1.2 |
| TPI | $\frac{\mathbb E\|\mathrm{HP}(M(x_t-\mathrm{warp}\,x_{t-1}))\|^2}{\mathbb E\|\mathrm{HP}(Mx_t)\|^2}$ | 실사 바닥의 1.5배 이내 |
| CWS | 카메라 jerk PSD와 영화 PSD 분포 사이의 $W_1$ | 영화 분포 안 |
| CI | 확률적 물질의 cardboard 비율 | ≤ 실사 + 5%p |
| Elo | blind pairwise (일반 + 촬영감독) | 10s ≥ 45%, 전장 ≥ 65% |

**Phase 1 승리 조건 (v2)**
- (a) GeCo, FV가 실사 바닥값 수준
- (b) LCE ≈ 0
- (c) $\mathrm{SFR_{prior}}\ge0.9$이고 $\mathrm{DR_{rel}}\le1.2$. 프레임 품질이 이미지 모델 수준이라는 뜻입니다.
- (d) TPI, CWS가 실사·영화 분포 안
- (e) Elo: 10초 샷 parity, 전장 B1~B6 ≥ 65%

## 10. 신규성 요약

**Crown jewels (v2)**
1. σ-대역 분할 소유권 + Partition Principle(K-불변)
2. World-Scale Causality ($\lambda_k$ 지연 + ownership window)
3. Frozen-Seed Band-Commit
4. Camera Owns Photography (plate dreamer + Gaffer)
5. Stochastic Matter Field
6. Calibrated, prior-agnostic Spectral Federalism

**빌려 쓰는 부품**
- Barooah–Hespanha의 drift 이론
- anchored noise 계열: Warped Noise, Integral Noise Warping
- 동기 디노이징: SyncTweedies
- IDU, Difix3D+
- EQ-VAE
- 렌더러: 2DGS, StopThePop, 3DGRT/3DGUT
- 재구성: DA3, VGGT-Ω, MoGe-2
- 3D prior: TRELLIS, Hunyuan3D, SAM 3D
- Infinigen, RGB↔X
- 카메라 데이터: DataDoP, CameraBench
- DMD2, Expert Iteration

자세한 겹침표는 §3.2에 있습니다.

## 11. 90일 실행 계획

| 주 | 트랙 A: 엔진 | 트랙 B: 렌더·카메라 | 트랙 C: 평가·데이터 |
|---|---|---|---|
| 1–2 | HSVW, octree DNA, dreamer harness, calibration v0 | 2DGS + StopThePop, band-fade LOD, sensor v0 | SFR/DR/GeCo/MEt3R/TPI/CWS/CI 구현, PSD 추출 |
| 3–6 | **X1**, `<plate>` LoRA | **X4** | FrozenBench 프롬프트, 경쟁 클립 수집 |
| 5–10 | **X2**, EQ-VAE arm, WDD v0 | Gaffer v0, 3DGRT 반사 | 평가 패널 셋업 |
| 6–11 | **X3** (SMF) | 4K/120 파이프라인 | — |
| 9–13 | 데모 3편: Frozen Tsunami 60s 원테이크, Bullet-time 사람, Macro→Wide 숲 | 최종 렌더 | **X5**, 기술 보고서, FrozenBench 공개 |

**관문 (Gates)**
- **Day 30:** X1 (i)~(iii) 통과
- **Day 60:** X2 중복 ≤ 5%, Elo +150 / X4 TPI 3배 감소
- **Day 90:** X5 기하 3배, 10초 parity, 전장 ≥ 65%

관문을 통과하지 못하면 목표를 낮추지 않습니다. §4 실험별 "실패 시 다음 수"의 업그레이드를 앞당깁니다.

## 12. 리스크 레지스터

| 리스크 | 감지 지표 | 업그레이드 대응 |
|---|---|---|
| plate 모드가 미감을 깎음 | plate와 일반 dreamer의 Aes 차이 > 0.3 | Gaffer와 그레이드 LUT 학습으로 미감 회복, LoRA rank 축소 |
| DR reward hacking (denoiser를 속이는 세계) | held-out seed DR과 학습 seed DR의 괴리 | dreamer 2종 앙상블, held-out seed·카메라, 인간 점검 |
| EQ-VAE 재구성 품질 저하 | 재구성 LPIPS/FID 상승 | 결정된 대역만 re-encode하는 pixel sync arm |
| 이음매가 보임 | seam 검출 2AFC > 60% | 인스턴스 잠금 강화, VP-blend의 $\rho$ 재보정, soft commit 앞당김 |
| 사람과 얼굴 | 뷰 간 ArcFace < 0.7 | 얼굴 전용 owner, SAM 3D Body 기하 |
| 경쟁 모델 업데이트 (Seedance 3, Veo 4) | AA Elo 변화 | B1~B6은 구조적 우위라 업데이트 영향을 덜 받음, FrozenBench 정기 재측정 |
| 세계 하나당 연산 > 3 H100h | 프로파일링 | distilled polish, student amortization |
| 라이선스 | — | Qwen-Image / HiDream-I1 기본, 연구 전용 모델은 기준선으로만 |

## 13. 맺음말

- 비디오 diffusion은 같은 세계를 240번 다시 상상하면서, 그 상상들이 우연히 일치하기를 바랍니다.
- v1은 세계를 한 번만 상상했습니다. 하지만 상상하는 사람이 여럿이었고, 순서가 없었고, 사진과 세계를 구분하지 못했습니다.
- v2는 거기에 순서와 분업을 넣습니다.
  - 전역 변수를 먼저 정하고, 레이아웃은 한 캔버스가 정하고, 이후 대역은 세계 스케일 순서로 결정합니다.
  - 대역마다 가장 잘 아는 owner가 한 번만 결정합니다.
  - seed는 DNA로 고정합니다.
  - 확률적 물질은 통계로 합의합니다.
  - 사진은 카메라가 찍고, 조명은 gaffer가 칩니다.
- 이 순서가 바로 할리우드의 제작 순서입니다. 우리는 이것을 정리와 연산자로 옮겼습니다.
- Phase 1에서는 frozen-time 샷에서 이미지 모델 수준의 프레임과 비디오 모델이 구조적으로 만들 수 없는 테이크로 이기고, FrozenBench로 그 결과를 숫자로 남깁니다. Phase 2에서는 같은 원리를 시간 축으로 확장합니다(event-first, 시간 연방).
- 프레임마다 새로 뽑는 랜덤은 여전히 필름 그레인 하나뿐입니다.
