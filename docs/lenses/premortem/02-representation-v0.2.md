> **출처 메모.** 설계 워크숍에서 [표현 / representation] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 건설적 프리모템과 개정안(STRATA v0.2)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../../PARADIGM.md)의 판결록(§0.5)을 따른다.

---

# STRATA 건설적 프리모템: [representation] 렌즈 리뷰와 v0.2 개정안

*공동창업자 리뷰 메모 · 기준일 2026-10-10 · 대상: STRATA v0.1 설계 메모*

과제와 섹션의 대응은 다음과 같다.
- 과제 1+2 (실패 지점과 업그레이드): §1
- 과제 3 (신규성 점검): §2
- 과제 4 (90일 실험): §3
- 과제 5 (개정 제안서 전체본): §4

---

## 0. 요약

**진단.** STRATA v0.1의 뼈대는 맞다. 세계를 먼저 만들고, 시간 순서 체인 대신 스케일 순서 트리를 쓰고, 세계를 시드의 순수 함수로 두고, 물리 카메라로 찍는다는 네 가지다. 실패는 뼈대가 아니라 관절에서 난다. 관절은 다섯 군데다. teacher와 세계를 잇는 인터페이스, "대역(band)"이라는 은유를 불투명한 기하에 그대로 쓴 곳, 부트스트랩의 경제성, 외관의 정본을 "재질"에 둔 결정, 그리고 "일관된 세계면 아름답다"는 가정이다.

| # | 실패 지점 | 가능성 | 터지는 곳 | 업그레이드 |
|---|---|---|---|---|
| F1 | teacher 인터페이스: 픽셀별 timestep, latent 스펙트럼, CFG 편향, 프롬프트 Janus, 폐쇄형 teacher | 매우 높음 | S²R 전체 | **U1 Band-Isometric Critic** |
| F2 | 부트스트랩 비용(샷당 수백 GPU-시간), Band-LRM 순환, CAT3D형 기반암 | 매우 높음 | 데이터 엔진, 기반암 | **U2 World Distribution Matching**: teacher는 화가가 아니라 심판 |
| F3 | "대역 ⊕"가 불투명 기하에서는 정의되지 않음 (반투명 구멍, shimmer) | 높음 | 모든 줌·도약 샷 | **U3 누적 디코드 + Dual-cut λ-블렌딩** |
| F4 | "상관 길이 ≈ h_l" 가정 붕괴: 긴 결맞음 구조, 텍스트, 점광원·글린트·가는 선 | 높음 | 데모 샷 1, 4, 5 | **U4 Perceptual-Scale Hierarchy** |
| F5 | 재질을 정본으로 두면 CG 룩과 이중 그림자가 생김 | 높음 | Elo | **U5 Radiance-Anchored Shading** |
| F6 | atom 5종 × 렌더 모드 2개 × 미분 가능 버전으로 범위가 폭발하고 train/test gap이 생김 | 높음 | 팀 속도 | **U6 Unified Kernel Atom + 단일 트레이서** |
| F7 | 일관된 "촬영지"는 아름다운 "세트"가 아님 | 중상 | Elo 60% 목표 | **U7 Directed World** |

**Atlas 피벗을 읽는 법.** World Labs는 splat 기반 Marble에서 프레임 생성형 Atlas로 무게를 옮겼다. 명시적 세계가 시네마틱 화질에서 진 이유는 넷으로 나눌 수 있다. v0.2는 이 넷을 각각 정면으로 겨냥한다. 업계가 포기한 지점이 우리의 출발점이다.

| 원인 | 대응 |
|---|---|
| (a) splat 개수 상한과 단일 스케일 | U3, U4 |
| (b) 구운 조명과 반사 부재 | U5, U6 |
| (c) 비디오 teacher의 화질 상한 (FlashWorld, Lyra) | U1, U2 |
| (d) 촬영지 수준의 미감 | U7 |

**가장 강하게 거는 새로움 세 가지** (자세한 내용은 §2).
1. **PSH와 Scale-Time Duality v2.** 원소마다 "프리필터를 거친 뒤에도 보이는가(가시도)"와 "결맞음 길이"로 레벨을 배정한다. 카메라 footprint에 따라 지연 평가하고, 렌더는 dual-cut으로 λ에 대해 연속이 되게 한다.
2. **Band-isometric 이미지 critic을 쓰는 amortized World Distribution Matching.** 장면 규모의 다층 세계를 이미지 teacher가 한 대역씩, 자기 네이티브 해상도에서 판정한다.
3. **구조 주소 순수성(structure-addressed purity).** 위상 퍼텐셜과 구조 주소 노이즈 덕분에, 장거리 결맞음을 가진 구조까지 순서와 무관한 국소 생성으로 만들 수 있다.

---

## 1. 실패 지점 7개와 업그레이드

각 실패는 네 부분으로 정리한다. 증상, 기술적 원인, 조기 경보 지표, 업그레이드다.

### F1. teacher 인터페이스: 이론상의 스펙트럼 라우팅이 실제 이미지 모델에서는 작동하지 않는다

**증상.** S²R을 돌린 첫 주에 다음이 나타난다.
- 깊이 경계(전경 인물과 배경 하늘 사이)를 따라 후광(halo)이 생긴다.
- 레벨을 내려갈수록 채도가 타오른다(fried).
- 이끼 클로즈업 뷰 안에 교토 정원 전체가 축소판으로 다시 나타난다(Droste 효과).
- 커밋된 대역이 VAE를 왕복할 때마다 조금씩 물러진다.

**기술적 원인.**
1. **픽셀별 timestep $T[p]$는 "베이스 동결 + LoRA"와 함께 갈 수 없다.**
   - FLUX, SD3, Qwen-Image 계열 MMDiT는 스칼라 $t$를 AdaLN 변조(shift, scale, gate)를 통해 모든 토큰에 똑같이 넣는다.
   - 같은 MLP로 토큰별 변조를 계산하는 것 자체는 구조상 가능하다. 하지만 토큰마다 노이즈 수준이 다른 입력은 학습 분포 밖(OOD)이다. 깨끗한 토큰과 노이즈 섞인 토큰이 서로 어텐션하는 상황은 Diffusion Forcing처럼 처음부터 그렇게 학습해야 보정된다.
   - Differential Diffusion은 학습 없이 돌아가지만, 실체는 "픽셀마다 시작 시점이 다른 소프트 인페인팅"이다. 아직 풀리지 않은 픽셀에는 노이즈를 입힌 원본을 매 스텝 다시 넣는다.
   - 깊이 불연속에서는 $T$가 몇 옥타브씩 점프하므로 후광이 생긴다. 또 전체 샘플링이 가장 거친 대역의 시작 시점에 묶인다.
2. **latent 공간은 깨끗한 파워법칙이 아니다.**
   - VAE가 f=8이고 2×2 patchify를 하면 토큰 하나가 16×16px을 덮는다. 화면 주기 16px 아래의 3–4 옥타브는 디퓨전 시간이 아니라 VAE 디코더가 정한다.
   - latent 스펙트럼에 고주파가 과하게 섞여 있다는 보고도 있다(Skorokhodov et al. 2025, "Improving the Diffusability of Autoencoders").
   - 따라서 v0.1 §7.3의 $\nu^*(t)$ 역함수는 latent teacher에서 그대로 성립하지 않는다.
3. **Tweedie 추정이 편향된다.** CFG 3–7을 쓰면 $\hat x_0$가 과포화되고 대비가 과해진다. FLUX.1-dev처럼 guidance-distilled 모델은 애초에 참 조건부 평균을 내지 않는다. 14개 레벨을 지나는 동안 이 편향이 쌓인다.
4. **프롬프트 Janus.** 모든 뷰에 같은 장면 프롬프트를 주면, 정면 매크로 뷰도 "교토 정원 사진"처럼 보이려고 한다. Generative Powers of Ten이 스케일별 프롬프트를 손으로 써야 했던 이유가 이것이다.
5. **최고 미감의 teacher는 닫혀 있다.** 재노이즈, Tweedie 동기화, score distillation은 모두 score 또는 velocity에 접근해야 한다. GPT Image 2, Nano Banana Pro, Seedream 5급은 API로만 쓸 수 있으므로 이 역할을 할 수 없다.

**조기 경보 지표.**
- 깊이 엣지 ±8px 띠의 고주파 에너지와 내부 고주파 에너지의 비
- 레벨당 채도 드리프트 $\Delta C^*$
- 뷰 이미지와 장면 프롬프트의 CLIP 유사도가 스케일과 상관없이 높게 유지되는 현상 (Droste 신호)

**업그레이드 U1: Band-Isometric Critic.** 픽셀별 timestep을 만들지 않는다. 대신 timestep 하나로 충분한 시점을 고른다. footprint 불균일을 teacher 안에서 풀지 않고 카메라 선택으로 없애는 것이 핵심이다.

- **등척 대역 시점(band-isometric view)의 정의.** 레벨 $l$의 블록 $b$를 보는 카메라 $v$는 다음을 만족해야 한다.
$$\Big|\log_2\phi_v(p)-\log_2\phi^*_l\Big|\le\tfrac14\quad\forall p\in\Omega_v,\qquad \phi^*_l=\frac{h_l}{s^*}$$
  - $s^*$는 대역 $l$이 teacher의 결정 옥타브 $o^*$에 놓이도록 하는 셀당 픽셀 수다. 보정으로 정하며, 16px 토큰 teacher라면 대략 16–32다.
  - 커밋된 기하에서 블록의 지배적 법선 군집을 찾고, 군집마다 망원(근사 직교) 정면 시점을 둔다.
  - 깊이비 $z_{\max}/z_{\min}\le2^{1/4}$를 지키려면 거리는 블록 깊이 범위의 약 5.3배 이상이어야 한다. 입사각은 $\theta\le33^\circ$로 제한한다. 이를 어기는 픽셀은 $m_v(p)=0$으로 마스크한다.
  - 블록 표면의 95% 이상을 덮을 때까지 시점을 추가한다. 보통 4–8개다.
  - 이렇게 하면 한 뷰 안에서 대역 $l$이 같은 화면 옥타브에 놓인다. 그래서 스칼라 $t$ 하나로 충분하고, teacher는 손대지 않아도 되며, 미감도 보존된다. 미세 레벨의 teacher 뷰는 결국 "정면 망원 매크로 텍스처 사진"이 된다. 이미지 모델이 가장 강하고 가장 일관된 영역이다.
  - 코스 레벨(구도와 조명)은 시네마틱 뷰를 쓰고 $t$를 전 구간에서 뽑는다. 코스 대역은 높은 노이즈에서 한꺼번에 결정되므로 footprint 불균일이 문제가 되지 않는다.
- **스펙트럼 결정 곡선 보정 (spectral decision calibration).** teacher와 VAE 조합마다 실측한다.
$$D(o,t)=\frac{\mathbb{E}_{x_0}\operatorname{Var}_{\epsilon}\big[\mathrm{BP}_o\big(\mathrm{Dec}(\hat u_0(u_t))\big)\big]}{\mathbb{E}_{x_0}\big\|\mathrm{BP}_o(x_0)\big\|^2}$$
  - $\mathrm{BP}_o$는 픽셀 공간의 Laplacian 대역통과 필터다.
  - $t_{\text{start}}(o^*)=D^{-1}(o^*,0.9)$, $t_{\text{end}}(o^*)=D^{-1}(o^*,0.1)$로 둔다.
  - **한 옥타브 분리 조건:** $D(o^*{+}1,\,t_{\text{end}})\ge0.5$이고 $D(o^*{-}1,\,t_{\text{start}})\le0.5$. 이 조건을 만족하는 $o^*$를 고른다(§3 E1).
- **자기 캡션 (self-captioning).** 뷰의 조건은 장면 프롬프트가 아니라, 세계가 지금 그 뷰에서 보여 주는 내용이다.
$$c_v=\mathrm{VLM}\big(\mathrm{Render}_v(W^{(l-1)}),\ \mathrm{sem}|_v,\ \text{화폭}\approx W_{px}\phi^*_l\,\text{m},\ \text{lens}\big)\ \oplus\ \text{prompt\_tree}(l)$$
  teacher가 받는 질문이 "교토 정원 사진은 어떻게 생겼나"에서 "폭 9cm짜리 이끼 낀 화강암 정면 망원 사진은 어떻게 생겼나"로 바뀐다. Droste 효과는 원천 차단된다. 캡션은 (block, level, view) 단위로 캐시한다.
- **가이던스 위생.**
  - APG(Adaptive Projected Guidance, Sadat et al. ICLR 2025) 또는 CFG-rescale을 쓴다.
  - 미세 레벨에서는 autoguidance(Karras 2024)나 guidance 1.0–1.5를 쓴다.
  - 저주파 색은 커밋된 부모 값으로 고정한다. 효도 제약이 이 역할을 겸한다.
- **teacher 두 개 체계.**
  - score critic은 open-weight 모델이 맡는다. Apache-2.0인 Qwen-Image급을 우선하고, FLUX.2-dev급은 라이선스를 확인한 뒤 쓴다.
  - 폐쇄형 SOTA는 샘플 분포로만 쓴다. 그 출력을 real 집합으로 삼아 판별기($\mathcal{L}_{\text{GAN}}$, U2)를 학습시키면, score에 접근하지 않고도 그 분포 쪽으로 당길 수 있다. 히어로 스틸 조건으로도 쓴다.
- **부수 효과.** 등척 뷰에서는 latent 토큰 하나가 세계 표면의 거의 같은 면적을 덮는다. 그래서 world-anchored noise(v0.1 §7.4)를 latent 토큰 단위로 정확히 적분할 수 있다. v0.1에 있던 픽셀 노이즈와 latent 노이즈의 불일치가 사라진다.

### F2. 부트스트랩이 끝나지 않는다: 비용, 순환, 그리고 CAT3D로 돌아간 기반암

**증상.**
- 히어로 샷 하나를 폴리시하는 데 수백 GPU-시간이 든다.
- native 모델 학습에 필요한 $10^5$개 규모의 세계를 만들 수 없다.
- Band-LRM은 학습 데이터가 없다. 그래서 GD 리프팅으로 대체되고 더 느려진다.
- 기반암은 다시점 동시 생성(CAT3D/SEVA 스타일) 뒤에 VGGT/DA3 재구성을 붙인 구조다. 그래서 랜드스케이프 §7.5가 짚은 "평균되며 흐려지는" 문제를 그대로 물려받는다.
- 건물 내부와 뒤편은 정준 카메라에 보이지 않으므로 비어 있다.

**기술적 원인.**
- **비용 추정** (가정을 명시한다).
  - 10초 Powers-of-Ten 경로의 고유 셀은 40–160M이다(v0.1 §4.5). 64×64 표면셀 블록으로 나누면 1–4만 블록이다.
  - 블록당 뷰 12개, 정제 3스텝, teacher 1회 호출(12–20B MMDiT, 1024², H100에서 약 0.3초)을 가정하면 순수 teacher 시간만 30–120 H100-시간이다.
  - LIFT와 ARBITRATE를 포함하면 **샷당 60–240 H100-시간**이다. 학습 세계 $10^5$개면 약 $10^7$ H100-시간이 된다.
- **순환.** Band-LRM(다시점 대역 이미지에서 대역 latent로)은 (입력, 정답) 쌍이 필요한데, 그 정답이 S²R의 출력이다.
- **기반암의 구멍.** 정준 카메라는 보이는 것만 그린다. 방 하나 같은 가림 영역은 코스 스케일 구조라서, 미세 레벨의 국소 연산자로는 만들 수 없다. 남는 선택지는 "볼 때 만든다"인데, 이는 경로 의존 금지 원칙과 충돌한다. 그렇지 않으면 구멍이 남는다.

**조기 경보 지표.** 블록당 H100-초, 기반암 재구성의 다시점 잔차(MEt3R), 렌더 대비 원본 뷰의 고주파 에너지 손실률, 가림 영역 점유율.

**업그레이드 U2: Teacher-as-Critic, 즉 World Distribution Matching(WDM).** 이미지 teacher의 역할을 화가에서 심판으로 바꾼다. teacher는 세계를 그리지 않는다. 세계 생성기 $G_\theta$가 만든 세계를 여러 시점에서 판정할 뿐이다. 모든 시점의 판정이 같은 세계 변수로 흘러가므로 일관성은 구조로 보장되고, 품질은 분포 매칭을 통해 teacher 수준에 수렴한다. 평균을 낼 "뷰 이미지"가 애초에 존재하지 않으니 CAT3D식 블러가 생길 자리도 없다. 창업자의 원안인 "이미지 디퓨전에게 세계를 여러 시점에서 그리게 한다"를 가장 급진적으로 실현하는 방법이다.

- **Stage A: 현실 prior.**
  - DL3DV-10K, SpatialVID 정적 구간, Sekai, ScanNet++, MatrixCity(항공과 거리를 아우르는 합성 도시)를 DA3, VGGT-Ω, AnySplat으로 초기화하고 STRATA로 피팅한다. 주로 코스에서 중간 레벨까지다.
  - Infinigen 합성 장면을 더한다. 내부와 뒤편을 포함한 완전한 장면이고, sub-mm GT와 재질 GT가 있다.
  - 이 데이터로 코스 STFT를 flow matching으로 학습한다.
  - 재구성에서 관측되지 않은 셀은 손실에서 빼고 `unobserved` 토큰으로 표시한다. 완성(completion) 능력은 합성 장면과 Stage B의 critic이 가르친다. Stage B는 원래 관측되지 않았던 영역에도 카메라를 놓는다.
  - 결과적으로 기반암은 **전역 1패스로 내부와 뒤편까지 포함해 생성**된다. 구멍이 없고 hole-filling도 없다.
- **Stage B: WDM.**
$$\nabla_\theta\mathcal{L}_{\text{DM}}=\mathbb{E}_{C,t,\epsilon}\Big[w_t\big(\hat u_0^{\text{fake}}(u_t,t,c_C)-\hat u_0^{\text{real}}(u_t,t,c_C)\big)^{\!\top}\frac{\partial u}{\partial\theta}\Big],\qquad u=\mathcal{E}_{\text{vae}}\Big(T_{\text{disp}}\big(\mathrm{Render}_C(G_\theta(\text{seed},y))\big)\Big)$$
  - critic 카메라 분포: $C\sim0.6\,\Pi_{\text{iso}}(l)+0.3\,\Pi_{\text{cine}}(y)+0.1\,\mathcal{M}$. 각각 등척 뷰, 시네마틱 뷰, 미니맥스 아카이브다.
  - 등척 카메라에서는 $t\sim U[t_{\text{end}}(l),\,t_{\text{start}}(l)]$로 뽑는다. 그 대역만 판정하라는 뜻이다. 그래디언트는 레벨 $l$의 생성 경로로만 흐르고, 커밋된 $Z_{<l}$에는 stop-grad를 건다. v0.1의 그래디언트 라우팅이 여기서는 구조 자체가 된다.
  - fake score는 teacher의 LoRA(rank 64–128)다. 렌더 위에서 denoising loss로 학습하고, DMD2처럼 two time-scale(5:1)로 업데이트한다.
  - $\mathcal{L}_{\text{GAN}}$은 fake-score 백본의 특징 위에 판별기를 둔다. real 집합은 footprint를 맞춘 실사 패치(매크로 사진, 시네마틱 스틸)와 폐쇄형 teacher의 샘플이다. teacher 편향을 교정하고 폐쇄형 분포를 흡수하는 역할이다.
  - **미니맥스 에너지 교체.** v0.1의 "디퓨전 손실 = implausibility"는 이미지 복잡도와 상관이 있어서, 못생긴 뷰가 아니라 복잡한 뷰를 찾는다. v0.2는 $\mathcal{M}\leftarrow\operatorname{top\text{-}k}_{C\in\Pi_{\text{cine}}}\|\hat u_0^{\text{fake}}-\hat u_0^{\text{real}}\|$를 쓴다. 생성기의 렌더 분포가 teacher 분포에서 가장 멀리 벗어난 뷰를 정확히 가리키는 값이다.
- **Stage C.** 경로 영상 보상으로 Render-RL을 한다(U7).
- **S²R의 위치 변경.** S²R은 test-time 히어로 폴리시로 내려간다. 경로가 요구하는 블록만 대상으로 하고, WDM 출력에서 시작해 K=1–2 스텝만 돈다. 비용이 1–2자릿수 줄어든다.
- **비용 감각.**
  - WDM 한 스텝은 뷰 6개 × (teacher 순전파 2회 + VAE 인코더 역전파 + 렌더 역전파)로, H100에서 3–5초다.
  - 소규모 장면 실험은 64 GPU × 2만 스텝으로 1–2일이다.
  - 전 레벨 amortized 본학습은 수천 H100-일 규모로, 이미지 생성기 증류(DMD2)와 같은 자릿수다.
- **FlashWorld와 갈리는 지점.** FlashWorld는 비디오 teacher(Wan2.2-5B, 480p)와 pixel-aligned Gaussian 때문에 상한이 묶였다. WDM에서는 각 레벨이 teacher의 네이티브 해상도로 판정되므로, 세계의 유효 해상도 상한이 대략 teacher 해상도 × $2^{L-l_g}$가 된다.

### F3. "대역 ⊕"는 불투명한 기하에서 성립하지 않는다

**증상.**
- Powers of Ten 줌에서 LOD 전환 구간마다 표면이 반투명해지고 배경이 비친다(see-through).
- 앞뒤 순서가 픽셀마다 뒤바뀌며 반짝인다(shimmer).
- 숲이 VOL aggregate에서 SURF 잎으로 넘어가는 레벨에서 밝기가 출렁인다.

**기술적 원인.**
1. **덧셈이 정의되지 않는다.** Laplacian 피라미드의 덧셈은 이미지나 높이장 같은 선형 신호에서만 의미가 있다. 불투명한 기하에 "대역을 더한다"는 연산은 정의되지 않는다. 잎 표면은 숲 볼륨에 더해지는 것이 아니라 숲 볼륨을 **대체**한다.
2. **페이드가 구멍을 만든다.** GATHER는 컷 레벨의 atom만 $\alpha_{\text{lod}}$로 페이드한다.
   - 부모를 그리지 않으면 $\alpha_{\text{lod}}<1$인 구간에 구멍이 뚫린다.
   - 부모와 자식을 둘 다 그리면 커버리지가 $1-\alpha(1-\alpha)$가 되어, $\alpha=0.5$에서 **배경이 25% 새어 나온다.**
   - 깊이가 거의 같은 두 층의 순서가 픽셀마다 바뀌므로 shimmer가 생긴다.
3. **latent 사영은 외관을 보장하지 않는다.** $B_l\leftarrow B_l-\mathcal{U}\mathcal{D}(B_l)$는 디코더가 선형일 때만 외관 공간의 효도를 보장한다. 실제 디코더는 비선형이다. 외관 공간 사영을 커밋마다 최적화로 풀면 비용이 너무 크다.
4. 따라서 "팝핑은 수학적으로 존재할 수 없다"는 v0.1의 주장은 지금 형태로는 성립하지 않는다.

**조기 경보 지표.** 줌 경로에서 $\|I_{t+1}-\mathrm{warp}(I_t)\|$ 중 레벨 경계와 동기화된 성분, 커버리지 맵 최소값 < 0.99.

**업그레이드 U3: 누적 디코드 + Dual-cut λ-블렌딩.** "대역"은 latent 안에만 두고, 렌더는 늘 완전한 세계 두 장을 외관 공간에서 섞는다.

- **latent는 잔차로, 디코드는 누적으로.** $z_l$은 VAR식 다중스케일 잔차 latent다($\mathrm{Up}(Z_{<l})$에 대한 잔차). 디코더 $D_l(Z_{\le l})$은 그 레벨의 **완전한 atom 집합** $\hat W^{(l)}$을 낸다. 각 레벨이 자기 스케일에 맞는 표현을 처음부터 고르므로 VOL에서 SURF로 넘어가는 타입 전환이 자연스럽다.
- **Dual-cut 렌더.**
$$\lambda(p)=\log_2\frac{h_0}{\kappa\,\phi_{\text{eff}}(p)},\quad l_a=\lfloor\lambda\rfloor,\quad f=\lambda-l_a,\qquad I(p)=(1-f)\,I^{(l_a)}(p)+f\,I^{(l_a+1)}(p)$$
  - $I^{(l)}$은 영역마다 $\min(l,\text{leaf})$ 레벨의 완전한 세계를 렌더한 이미지다.
  - **정리 (λ-연속성).** 정수 $\lambda$에서 양쪽 극한이 모두 $I^{(l)}$이므로 $I(p)$는 $\lambda$에 대해 연속이다. 카메라가 연속으로 움직이면 $\lambda$도 연속이다. 따라서 **팝핑은 구조적으로 0이다.** 각 컷이 완전히 불투명한 세계라서 반투명 구멍도 생기지 않는다.
  - 비용은 primary 2배다. $f<0.05$이거나 $f>0.95$인 픽셀은 컷 하나만 렌더한다.
- **효도 손실의 역할 재정의.** 이 손실은 이제 팝핑이 아니라 **크로스디졸브 고스팅**(두 컷의 실루엣이 이중으로 보이는 현상)을 줄이는 데 쓴다.
$$\mathcal{L}_{\text{fil}}=\mathbb{E}_l\,\mathbb{E}_{C:\,\phi_{\text{eff}}=h_l/\kappa}\Big[\big\|k_{\text{pix}}*I_C^{(l+1)}-k_{\text{pix}}*I_C^{(l)}\big\|_1+\lambda\,\mathrm{LPIPS}\Big]$$
  컷 $l$이 정확히 충분해지는 footprint에서 두 컷이 같아 보여야 한다는 조건이다. 커밋 시점에 사영할 필요가 없고, 디코더 학습과 WDM에 손실로 걸면 된다.
- **의미 효도 (semantic filial).** 자식 `sem`의 면적 가중 평균과 부모 `sem`의 코사인 유사도가 0.9 이상이어야 한다. "돌멩이가 개구리로 변하는" 문제는 저주파 외관이 아니라 의미 층에서 막는다. 다만 원거리에서 정말 모호했던 것이 근거리에서 분해되는 것은 허용한다. 실제 세계의 줌도 그렇게 작동한다.

### F4. "상관 길이 ≈ h_l" 가정이 깨지는 곳에서 데모가 죽는다

**증상 (데모 샷 기준).**
- **샷 1 (Powers of Ten).** 항공 구간에서 송전선, 안테나, 레일이 사라진다. 실제 항공 사진에서는 폭이 서브픽셀이어도 선으로 보인다.
- **샷 5 (네온 거리).**
  - 간판 글자가 블록 경계에서 글리프 수프(glyph soup)가 된다.
  - f/1.4 배경의 네온과 가로등이 보케 볼이 아니라 뿌연 글로우가 된다.
- **벽돌과 타일.** 블록 경계마다 줄눈 위상이 어긋나는 전위(dislocation)가 생기고, 겹침 구간에서는 대비가 꺼진다.
- **눈밭과 물결.** 반짝임(glint)이 멀리서 보면 플라스틱처럼 매끈해진다.

**기술적 원인.**
1. **긴 결맞음.** 인공물의 미세 구조(줄눈, 타일, 펜스, 레일, 직물 조직, 나뭇결)는 주기가 $h_l$ 수준이어도 결맞음 길이는 $10^2$–$10^3\,h_l$이다. 반경 $r_l$짜리 국소 연산자는 이웃 블록의 위상을 알 수 없다. Jacobi 패스로는 의존 반경이 패스당 $r$씩만 늘어난다.
2. **VP 블렌딩의 맹점.** Heitz–Neyret 블렌딩은 확률적(stochastic) 텍스처용이다. 원 논문도 구조적 텍스처에서는 고스팅이 생긴다고 밝힌다. 위상이 $\pi$ 어긋난 $\cos kx$와 $\cos(kx+\pi)$를 $w=\tfrac12$로 섞으면 0이 되고, 경계에서 대비가 사라진다. 이것이 정확히 "뭉개짐"이다.
3. **에너지가 큰 작은 것.** 점광원, 글린트, 가는 고대비 선은 기하 크기로는 미세 레벨이지만, 픽셀에 주는 영향은 코스 스케일까지 살아남는다. 프리필터 평균은 에너지는 보존해도 분포(희소한 피크)를 잃는다. lens gating이 이것들을 지우면 **보케 볼이 사라진다.** 보케 볼은 얕은 심도 영상미의 핵심이다.
4. **의미의 결맞음.** 텍스트는 글자 사이의 의미 의존이 블록 크기를 넘는다.

**조기 경보 지표.** 블록 경계에서 국소 FFT로 잰 위상 점프, 줌 OCR의 문자 오류율(CER), f/1.4 렌더의 보케 볼 개수 재현율, 항공샷에서 가는 선의 대비.

**핵심 관찰.** v0.1의 스케일-시간 이중성은 등방적 파장 대역을 가정했다. 실제 디퓨전의 스펙트럼 자기회귀는 원소별이고 비등방적이다.
- 길이 $\ell$, 폭 $w$인 선은 길이 방향으로 저주파 에너지를 가지므로, 높은 노이즈에서 이미 결정된다.
- 델타 함수(점광원)는 스펙트럼이 평탄하고 저주파 진폭이 총 에너지에 비례하므로, 밝을수록 일찍 결정된다.

즉 이 실패는 **이중성을 틀리게 적용해서** 생긴 것이다. 업그레이드는 이중성을 버리는 것이 아니라 바르게 쓰는 것이다.

**업그레이드 U4: Perceptual-Scale Hierarchy(PSH).** 원소는 픽셀에 영향을 줄 수 있는 가장 거친 레벨에 산다.

- **가시도 레벨.**
$$V_e(h)=\frac{\big\|k_h*a_e\big\|_\infty}{k_h*\bar L},\qquad l_{\text{vis}}(e)=\min\{l:\ V_e(h_l)\ge\gamma_{\text{vis}}\},\quad \gamma_{\text{vis}}\approx0.02$$
  - 확률적 텍스처: $V\sim(s/h)^{1\text{–}2}$이고 $l_{\text{vis}}\approx\log_2(h_0/s)$다. v0.1과 같다.
  - 선(대비 $c$, 폭 $w$): $V\approx cw/h$다. 2cm 검은 케이블이 하늘 앞에 있으면 $h\le cw/\gamma=1$m까지 살아남는다. 그래서 **2km 항공샷에도 존재한다.**
  - 점광원(HDR 대비 $c\sim10^3$–$10^5$, 면적 $A$): $V\approx cA/h^2$다. 수십 레벨 위로 승격된다.
- **결맞음 레벨.** 결맞음 길이가 $\ell_{\text{coh}}$인 구조의 **위상**은 $l_{\text{coh}}=\lceil\log_2(h_0/\ell_{\text{coh}})\rceil$에서 결정한다. 디테일은 여전히 미세 레벨에서 결정한다.
- **배정 규칙.** $l^\star(e)=\min(l_{\text{size}},\,l_{\text{vis}},\,l_{\text{coh}})$. 승격된 원소는 코스 레벨의 **Structure 토큰**으로 생성되고, 미세 레벨이 그것을 조건으로 받아 정제한다.

| 구조 | 파라미터 (텐서) | 렌더 | 미세 레벨이 받는 조건 |
|---|---|---|---|
| **CARRIER** (위상 퍼텐셜) | 정점 격자 위의 $\psi_{1,2}$ (f32, 사이클 단위), 진폭장 $A$ (f16), 프로파일 latent (f16[16]) | 국소 디퓨전의 조건으로만 쓰임 | $u_j=\operatorname{frac}\psi_j$, 단위 인덱스 $n_j=\lfloor\psi_j\rfloor$, $\mathbf{k}=\nabla\psi$ |
| **CURVE** | 체인 Bézier 또는 카테너리(끝점, 처짐), 폭 $w$, $L_d$, 스펙 | 해석적 커버리지 $\min(1,w/\phi)$, CoC만큼 폭 확장 | 곡선 국소 좌표 |
| **SYMBOL** | baseline Bézier, glyph id u16[≤64], 폰트 임베딩 f16[32], 크기, 잉크와 발광 | MSDF 벡터 글리프 (어느 줌에서나 선명) | 글리프 마스크와 재질 |
| **POINT** | 위치 f32×3, HDR 강도, 로브(축, 각) | 렌즈 처방 기반 조리개 스프라이트 | — |
| **GLINT** (확률 과정) | 미세면 밀도 $\rho_g$, 미세면 거칠기, NDF, salt | 절차적 이산 미세면 글린트 (Chermain 2020, Deliot–Belcour 2023 계열) | — |

- **위상 퍼텐셜이 순수성을 지킨다.**
  - $\psi$는 코스 셀의 **정점**에 저장하고 보간한다. 그래서 셀 경계에서 C0 연속이다.
  - 벽돌마다의 개성은 공간 셀이 아니라 **구조 주소 노이즈** $\xi_n=\mathrm{Philox}(\text{seed},\text{inst},n_1,n_2)$로 정한다. 벽돌 #(37,12)는 어느 블록에서 어떤 순서로 평가해도 같은 벽돌이다.
  - 국소 연산자의 반경은 그대로 유계이고 순수하다. 장거리 결맞음은 반경을 늘려서 얻는 것이 아니라 **조건으로** 들어온다.
  - 위상이 맞춰진 신호끼리만 섞으므로 상쇄가 없다. VP 블렌딩은 확률적 잔차에만 쓴다.
  - 의도된 전위(모서리, 서로 다른 벽)는 `inst` 경계에서만 허용한다.
- **lens gating 면제 규칙.**
  - 게이팅은 확률적 대역만 생략한다.
  - POINT와 GLINT 상위 $k$개는 CoC가 아무리 커도 조리개 스프라이트로 렌더하므로 보케 볼이 진짜가 된다.
  - CURVE는 CoC로 폭을 넓힌 커버리지로, SYMBOL은 블러된 MSDF로 렌더한다.
  - 승격 원소는 희소하므로, 배경 생성 비용을 약 1/1000로 줄이는 효과는 대부분 그대로 남는다.
- **텍스트는 latent에서 금지한다.** WDM 동안 OCR을 돌려서 두 경우에 벌점을 준다($\mathcal{L}_{\text{ocr}}$). SYMBOL이 없는 영역에서 글자가 검출되는 경우, 그리고 SYMBOL 문자열과 렌더의 OCR 결과가 다른 경우다.

### F5. 외관의 정본을 "재질"에 두면 CG가 된다

**증상.**
- 이미지 teacher가 그린 따뜻한 반사광과 미묘한 AO가 사라진다.
- 그림자가 이중으로 진다. 구운 그림자 위에 다시 계산한 그림자가 얹힌다.
- 피부와 천이 플라스틱처럼 보인다.
- 샷 조명 리그의 림 라이트가 불확실한 재질 위에서 반짝이 스티커처럼 뜬다.
- 한마디로 WorldGen의 "게임 룩"이 된다.

**기술적 원인.**
1. 장면 규모의 생성 이미지에서 intrinsic 분해(RGB↔X, DiffusionRenderer 계열)를 하면 그림자, AO, GI 색 번짐이 albedo로 새어 들어간다. 그 상태로 물리 조명을 다시 걸면 조명이 이중으로 적용된다.
2. probe GI(SH L2)와 NRC로는 사진 같은 미세 GI와 대기감을 재현하지 못한다.
3. 경쟁 축(Elo)은 사진다움이다. Marble이 사실적으로 보이는 이유는 조명을 **구웠기 때문**이고, Marble이 실패한 곳은 반사와 투명체다.

**조기 경보 지표.** albedo 맵의 그림자 엣지 검출률(이중 그림자 지수), teacher 원본 이미지 대비 단일 프레임 2AFC 선호도.

**업그레이드 U5: Radiance-Anchored Shading(RAS).** 세계는 카메라가 보는 것, 즉 world lighting 아래의 exitant radiance를 정본으로 저장한다. 물리는 뷰 의존 성분과 변화분만 계산한다.

$$L_o(\mathbf{x},\omega_o)=\underbrace{L_d(\mathbf{x})}_{\text{확산 radiance (HDR, 뷰 독립)}}+\underbrace{\int f_s(\omega_i,\omega_o;F_0,\alpha,\mathbf{n})\,L_W(\mathbf{x},\omega_i)\cos\theta_i\,d\omega_i}_{\text{RT 스페큘러: 세계 자신을 비춘다}}+\sum_k\Delta L_k(\mathbf{x},\omega_o)$$

- **스페큘러 항.** $L_W$는 세계 자신의 radiance로, $L_d$와 깊이 2까지의 스페큘러를 합한 값이다. 레이트레이싱하므로 거울은 세계에 실제로 있는 것을 비춘다. 카메라 뒤에 있는 것도 포함된다. 일관성을 가장 강하게 보여 주는 장면이 여기서 나오고, 이는 Marble이 약한 바로 그 지점이다.
- **분리는 구조가 강제한다.** $L_d$는 뷰 독립이므로 시점에 따라 움직이는 하이라이트를 담을 수 없다. WDM critic이 여러 시점에서 하이라이트를 요구하면, 그 에너지는 스페큘러 파라미터($F_0$, $\alpha$)로 갈 수밖에 없다. 초기화는 다시점 저백분위(min/percentile) 필터로 한다. 고전적인 스페큘러 분리 기법이다.
- **샷 리그는 가산 델타다.**
  - $\Delta L_k=(\hat\rho/\pi+f_s)\,E_kV_k$로 계산한다. albedo 추정 $\hat\rho$의 오차는 리그 기여분에만 곱해지고 베이스 이미지는 건드리지 않는다.
  - 감산(네거티브 필, 플래그)은 비율로 처리한다. $L_d'=L_d(1-\eta_k)$이고, $\eta_k$는 차단된 world 조도의 비율이다.
- **재조명은 비율 전송이 기본이다.** $L_d'=L_d\cdot\hat E_{\text{new}}/\hat E_{\text{old}}$로 처리하고, `unc`가 낮은 곳에서만 완전 PBR로 승격한다.
- **Phase 1에는 GI 경로추적이 필요 없다.** 정지 세계에서는 world lighting이 고정되어 있으므로 구운 GI는 치팅이 아니라 정답이다. 이 결정 하나로 렌더러 범위가 크게 줄어든다(U6과의 시너지).

### F6. 렌더러가 회사를 삼킨다

**증상.**
- atom 5종(SURF, VOL, STRAND, DIEL, EMIT) × Mode A/B × 미분 가능 버전을 만드는 데 9개월이 지나간다.
- critic이 학습에서 본 렌더(Mode A)와 최종 렌더(Mode B)가 다르다. 학습에서는 좋았던 세계가 시네마 렌더에서는 다르게 보인다(train/test gap).

**기술적 원인.**
- LIFT의 GD, DPS 가이던스, 미니맥스 그래디언트, WDM은 모두 렌더러를 미분해야 한다.
- strand BCSDF, 분산이 있는 유전체, SGGX 볼륨, 래스터와 RT의 하이브리드를 모두 미분 가능하게 만드는 것은 연구 과제 여러 개를 합친 규모다.
- 렌더 모드가 둘이면, critic이 판정한 이미지와 고객이 보는 이미지가 다르다.

**조기 경보 지표.** 같은 세계를 Mode A와 Mode B로 렌더했을 때의 LPIPS 차이, 렌더러 PR 대비 모델 PR의 비율.

**업그레이드 U6: Unified Kernel Atom(UKA) + 하나의 미분 가능 트레이서.**

- **기하 프리미티브를 하나로 줄인다.**
  - 기본형은 generalized-Gaussian 커널 타원체다.
  - 커널 차수 $n$(부드러운 볼륨에서 단단한 표면까지), 편평도(서펠), 신장(스트랜드 세그먼트), 선택적 cosine 변조(Gabor 캐리어)를 **연속 파라미터**로 둔다.
  - 3DGRT 논문은 이미 generalized Gaussian, 2D surfel, cosine 변조 커널을 같은 트레이서에서 다뤘다.
  - 타입은 이산 로짓이 아니라 연속 파라미터가 되고, 셰이딩 페이로드($L_d$, 스펙, SGGX, BCSDF, 글린트 참조)만 다르다.
- **유일한 예외는 유전체 계면이다.** 굴절에는 매끈한 법선이 필요하다. DIEL이 있는 레벨에만 희소 격자 위에 협대역(narrow-band) SDF 채널을 두고, 2차 패스에서 레이마칭한다.
- **렌더러는 하나다.**
  - 3DGRUT(오픈소스, 3DGUT 래스터 primary와 3DGRT 레이트레이싱 secondary의 하이브리드)를 확장한다.
  - critic 렌더(1024² 등척 뷰)와 최종 렌더(8K·120fps)에 **같은 코드 경로**를 쓴다. 차이는 spp와 2차 반사 깊이뿐이므로 train/test gap은 정의상 0이다.
  - 3DGUT의 Unscented Transform이 왜곡 렌즈와 롤링셔터를 정확히 처리한다. 공분산 대수 기반 DOF와 모션블러(v0.1 §5.2)는 유지한다.
  - RAS 덕분에 GI 경로추적은 필요 없다.
- **8K·120fps 예산** (추정). atom 10–30M개, dual-cut으로 primary 2배, 2차 반사 레이 1–4개를 가정하면 GPU 하나에 3–8fps다. 10초 1200프레임은 GPU 8장으로 수 분이다. 프로덕션 경로추적기로 내보내는 USD export는 선택 사항이다.

### F7. 일관된 "촬영지"는 아름다운 "세트"가 아니다

**증상.** 루프 클로저는 0이고 GeCo는 노이즈 바닥 수준이다. 그런데도 블라인드 Elo에서 Seedance 2.5와 Wan 3.0에 진다. 우리 영상은 "잘 찍은 부동산 워크스루"이고, 그들의 영상은 "약간 녹지만 숨 막히는 광고"다.

**기술적 원인.**
1. 이미지의 아름다움은 상당 부분 **그 시점에서만 성립하는 구도**에서 나온다. 전경 프레이밍, 역광 방향, 깊이 층위가 그렇다. 모든 시점에서 그럴듯하도록 분포를 맞춘 세계는 평균적인 "장소"로 수렴한다.
2. WorldScore를 보면 3D 계열의 주관 품질은 37–71, 비디오 계열은 49–64다. 명시적 세계의 병목은 미감이다.
3. GeCo 0.1–0.3 수준의 불일치는 일반 시청 조건에서는 잘 보이지 않는다. Elo는 일관성보다 미감에 투표한다.
4. v0.1의 미니맥스 DP는 최악의 시점을 고칠 뿐 아름다움을 더하지는 않는다.

**조기 경보 지표.** 경로를 따라간 영상의 미감 점수와 teacher 단일 이미지의 미감 점수 차이, 그리고 reveal 조건과 일반 조건 사이의 Elo 차이(§3 E2).

**업그레이드 U7: Directed World. 세계는 (seed, 의도)의 순수 함수다.**

- **의도 조건화.**
  - 조건은 $y=(\text{prompt\_tree},\ \text{hero\_stills},\ \text{shot\_intent})$이고, $\text{shot\_intent}$는 경로 스플라인 분포 $\Pi_{\text{shot}}$, 렌즈 계획, 시간대, 무드로 이루어진다.
  - **순수성의 정의를 정밀하게 하자.** 순수성은 **평가 순서**와 무관하다는 뜻이지, 조건과 무관하다는 뜻이 아니다. 의도가 바뀌면 세계도 바뀐다. 미술감독이 세트를 다시 짓는 것과 같다.
  - `shot_intent=∅`이면 "로케이션 모드"로, 여러 샷이 공유하는 세계가 된다.
- **경로 위 미감 보상** (Stage C. 코스 레벨은 Flow-GRPO, 미세 레벨은 보상 가중 증류):
$$R(W;y)=\mathbb{E}_{C\sim\Pi_{\text{shot}}}\big[A(\mathrm{video}_C)\big]+\lambda_p\,\mathrm{Par}+\lambda_b\,\mathrm{Back}+\lambda_c\,\mathrm{Comp}-\beta\max_{C\in\mathcal{M}}\Delta_{\text{DM}}(C)$$
  - $\mathrm{Par}$: 경로를 따라 잰 전경, 중경, 후경 시차 층위의 엔트로피. 3D 영상의 쾌감이 여기서 나온다.
  - $\mathrm{Back}$: 역광(contre-jour)과 림 성분의 비율.
  - $\mathrm{Comp}$: DataDoP 프레임으로 학습한 구도 critic.
  - $\Delta_{\text{DM}}$: 경로 밖에서도 무너지지 않게 하는 견고성 항.
- **DoP와 세계의 공동 최적화 (bilevel).**
  - DataDoP와 CameraBench로 학습한 경로 prior(Traj-DiT 계열)에서 시작해 세계를 생성한다.
  - 미분 가능 렌더러로 구도 보상의 경로 그래디언트를 구하고 스플라인을 고친다.
  - 이를 2–3회 반복한다. 감독이 로케이션 헌팅을 하고 세트를 손보는 순서와 같다.
  - 카메라 동역학은 실제 촬영 궤적의 jerk 스펙트럼에 맞춘다. 이것이 "버터 모션"의 반대다.
- **shot dress (샷별 세트 치팅).** shot rig의 기하 버전이다. 샷마다 copy-on-write 브랜치를 두고 코스 레벨의 소품 배치를 바꾼다. 영화 현장의 "치팅(cheating)"에 해당한다. 한 샷 안의 연속성은 정확히 유지된다.
- **평가 설계 업그레이드 (E2).**
  - Elo 프롬프트를 일관성 차이가 **눈에 보이는** 형태로 설계한다. 루프 시작과 끝을 나란히 보여 주는 reveal, 거울, 10배 줌, 120fps 슬로가 그런 예다.
  - 단일 프레임 미감 Elo는 따로 재서, teacher의 룩을 지켰는지를 분리해서 확인한다.

---

## 2. 신규성 점검: 빌려 쓸 부품과 진짜 새로운 것

| 구성요소 | 겹치는 선행 (실명) | 재분류 | 진짜 다른 점 (한 문장) |
|---|---|---|---|
| 코스→파인 계층 latent | VAR, Pyramidal Flow Matching, Relay Diffusion, XCube, LT3SD, TRELLIS(SLAT), WorldGrow, Generative Powers of Ten | **부품**: 희소 옥트리 VAE, 잔차 다중스케일 latent, fVDB, PTv3 직렬화 | 레벨을 파장이 아니라 **원소별 프리필터 가시도와 위상 결맞음**으로 배정한다(PSH). 그래서 송전선, 점광원, 줄눈 위상이 코스에서 결정된다. |
| 순수 함수와 경로 독립성 | Minecraft 월드젠, MultiDiffusion, DiffCollage, SyncTweedies. 순차형 대비군: Marble 1.1 dynamic cubes, HY-World 2.0 WorldStereo, WorldGrow, WonderWorld | **부품**: 다단계 청크 생성 패턴, 동기식 타일 합의 | **구조 주소 노이즈와 위상 퍼텐셜**로 장거리 결맞음 구조까지 국소 연산자의 순수성 안에 넣고, 평가 순서 순열 테스트로 이를 검증한다. |
| 세계 고정 노이즈 | How I Warped Your Noise, Infinite-Resolution Integral Noise Warping, Go-with-the-Flow, Consistent Flow Distillation | **부품** | 새로운 점은 작다. 등척 뷰에서 latent 토큰 footprint가 균일해져, 적분 노이즈가 latent 공간에서도 정확해진다는 정도다. |
| 이미지 teacher 분포 매칭 | DMD, DMD2, VSD(ProlificDreamer), FlashWorld(비디오 teacher, 480p), DiffSplat(물체 단위), Lyra(self-distill) | **부품**: DMD2 TTUR, GAN 항, LoRA fake score | 고해상도 **이미지** teacher가 band-isometric 시점에서 **한 대역씩만** 판정한다. 그 판정으로 장면 규모 다층 세계 생성기를 amortized로 학습한다. |
| 동기화 Tweedie (S²R) | SyncTweedies, DMV3D, 3D-Adapter, Lempitsky–Ivanov와 Waechter의 best-view 텍스처링 | **부품** (히어로 폴리시로 강등) | 이제 핵심 경로가 아니다. 소유권 annealing과 물리 중재는 test-time 옵션으로 남는다. |
| 픽셀별 timestep | Differential Diffusion, Diffusion Forcing | **대체됨** | 등척 시점 선택으로 필요 자체를 없앴다(U1). |
| LOD와 필터링 | mip-map, Mip-Splatting, Hierarchical 3DGS, Octree-GS, Nanite, SGGX, LEAN/LEADR, Loubet–Neyret | **부품** | 생성된 LOD를 **dual-cut 외관 공간 블렌딩**으로 렌더해 팝핑을 λ-연속성으로 0으로 만든다. 효도 손실은 고스팅만 맡는다. |
| lens-gated 생성 | voxel cone tracing, DOF-GS, foveated rendering | **부분적으로 새것** | 렌즈가 렌더가 아니라 **생성**을 생략시키고, 승격 원소 면제 규칙으로 보케 볼과 글린트의 진실성을 지킨다. |
| 물리 카메라 | Kolb 1995 realistic camera, Polynomial Optics(Hullin 2012), Hullin 2011 flare, 3DGUT, Newson 2017 grain | **부품 전부** | 새로운 것은 카메라가 생성 수요를 정한다는 결합뿐이다. |
| 커널과 렌더러 | 3DGRT/3DGUT(3DGRUT), EVER, Deformable Beta Splatting, 3D Convex Splatting, Textured Gaussians | **부품** | UKA는 이들의 조합이다. 타입을 연속 파라미터로 둔 설계 결정이고 연구 기여는 아니다. |
| 캐리어와 글린트 | Phasor noise(Tricard 2019), Gabor noise(Lagae 2009), Jakob 2014, Chermain 2020, Deliot–Belcour 2023, Water Wave Packets | **부품** | 이 절차적 원시들을 **생성 변수**로 승격한다. 코스 모델이 캐리어와 글린트 과정을 생성하고 국소 디퓨전이 그것을 조건으로 소비한다. |
| RAS | 게임의 lightmap + dynamic light, ratio relighting, DiffusionRenderer, RGB↔X, IC-Light | **부품** (개념) | 생성 세계의 정본 외관을 teacher가 본 radiance로 두고, 물리는 뷰 의존 성분과 변화분만 계산한다. |
| 미니맥스 카메라 | FisherRF, ViewFool, active view selection | **부분적으로 새것** | DMD 불일치 $\|\hat u^{\text{fake}}-\hat u^{\text{real}}\|$를 최악 카메라 에너지로 쓴다. |
| Directed world와 DoP | Director3D(Traj-DiT), GenDoP/DataDoP, CameraBench | **부품**: 궤적 prior | 의도를 조건으로 한 **순서 순수** 세계에 경로 위 미감 보상과 경로 밖 미니맥스를 결합한다. |
| 평가 | GeCo, MEt3R, ViewBench LCE, SCoPE, R2M-Bench, WorldScore, VBench-2.0, CameraBench | **부품** | frozen-time **내용 정합 i2v 프로토콜**, 효도 줌 점수, 세계 안에 놓은 ISO 12233 MTF 측정. |
| Phase 2 | PhysGaussian, NGFF, BulletTime, CAT4D, 4DNeX, MoVieS, Diff4Splat | **부품** | **스펙트럼 분업** WDM-4D: 이미지 teacher는 외관을, 비디오 teacher는 저해상도 운동만 판정한다(랜드스케이프 G8의 공백). |

**한 문장 테제.** STRATA는 **카메라의 해상도 질서(perceptual scale)를 세계의 생성 질서로 삼는** 최초의 표현이다. 이미지 디퓨전은 그 질서의 각 층을 등척 시점에서 판정만 한다. 그래서 일관성은 구조에서, 디테일은 수요에서, 미감은 판정과 연출에서 나온다.

**솔직한 경고.** 이미지 teacher를 쓰는 분포 매칭이 장면 규모와 2K 이상 해상도에서 안정적으로 수렴하는지는 선례가 없다(랜드스케이프 §10). 이것이 빈자리이자 가장 큰 리스크이고, §3 E3가 이를 반증하려는 실험이다.

---

## 3. 90일 결정적 실험

**전제.** 팀 5–6명, H100 32–64장. 모든 실험은 "이게 틀리면 설계의 어느 부분을 갈아 끼우는가"까지 미리 정해 둔다.

### E1. 스펙트럼 결정 곡선과 등척 정제 (D0–D30, H100 8장)

- **가설.** open-weight latent teacher(Qwen-Image급 또는 FLUX.2-dev급)에서 화면 옥타브의 결정 시점은 단조이고 충분히 날카롭다. 그래서 등척 뷰에서는 스칼라 $t$ 구간 하나가 정확히 한 옥타브를 정제한다.
- **셋업.**
  - 매크로·텍스처 4K 사진 2,000장과 건축 파사드 500장을 쓴다. 이미지마다 $t$ 30개 × seed 8개로 $\hat u_0$를 구해 $D(o,t)$를 측정한다.
  - 정제 테스트는 다음 순서다.
    1. 원본을 2× 다운샘플해 "코스 세계"를 만든다.
    2. 업샘플한 뒤 $t_{\text{start}}$로 재노이즈하고 $t_{\text{end}}$까지 디노이즈해서 $\hat x_0$를 얻는다.
    3. 자기 캡션과 APG를 켠 경우와 끈 경우를 비교한다.
  - 기준선은 SDEdit(같은 $t$), SUPIR급 디퓨전 SR이다.
- **성공 기준.**
  - $o^*$ 주변 세 옥타브에서 한 옥타브 분리 조건을 만족한다.
  - 효도 PSNR(선형 공간, 2× box-down 후) ≥ 38dB.
  - 정제 패치의 KID가 teacher 무조건부 샘플 KID의 1.3배 이내.
  - 같은 KID에서 SDEdit보다 효도 PSNR이 5dB 이상 높다.
  - 자기 캡션을 켰을 때 Droste 신호(장면 프롬프트와의 CLIP 유사도)가 50% 이상 떨어진다.
- **실패 시 다음 수.** 결정 폭이 1.5 옥타브를 넘을 때 쓸 업그레이드 경로다.
  - (a) 마지막 2 옥타브는 픽셀 공간 업샘플러 디퓨전을 미세 대역 전문가로 둔다.
  - (b) 1M장으로 per-token-$t$ 어댑터를 Diffusion-Forcing식으로 학습한다. 미감 드리프트 허용치는 aesthetic score −3% 이내다.
  - (c) 레벨 간격을 2 옥타브($h_l=h_04^{-l}$)로 넓혀 트리 깊이를 절반으로 줄인다.

### E2. Bullet-Time Bench v0와 Oracle arm (D0–D45, 주로 API 비용과 평가자 30명)

- **가설.**
  - **H2a.** 최상위 상용 모델은 frozen-time 재방문에서 측정 가능하게 실패한다. GeCo fused ≥ 0.08, LCE LPIPS ≥ 0.25, 그리고 spurious motion이 나타난다.
  - **H2b (결정적 가설).** 완벽히 일관된 실사 영상(oracle)은 재방문 중심 프롬프트에서 최강 경쟁 모델을 사람 선호로 60% 이상 이긴다.
- **셋업.**
  - 프롬프트는 48개다. 궤도 루프, 10배 푸시인, 거울과 유리, 매크로 디테일을 각 12개씩 둔다.
  - **내용 정합 프로토콜.**
    - Oracle-1: DL3DV-10K의 4K 실사 정적 장면 영상 원본.
    - Oracle-2: 같은 장면을 재구성(Zip-NeRF 또는 고밀도 3DGS)한 뒤 우리 물리 카메라 체인으로 렌더한 영상. 카메라 체인의 룩을 측정하는 용도다.
    - 경쟁 모델에는 같은 첫 프레임을 i2v 입력으로 주고, 카메라 지시 텍스트와 "time frozen" 조건을 함께 준다.
  - 대상 모델: Seedance 2.5, Wan 3.0, Kling 3.0 Omni, Veo 3.1, Gemini Omni 1.1 Flash, MiniMax H3.
  - 지표: GeCo, MEt3R, LCE, spurious motion energy, revisit 프레임의 고주파 유지율.
  - 사람 평가: 2AFC 300쌍 × 30명. 일반 조건과 reveal 조건(시작/끝 프레임을 나란히 제시)을 나누고, 단일 프레임 미감 2AFC를 따로 잰다.
- **성공 기준.** H2a가 5개 모델 중 3개 이상에서 확인된다. Oracle 선호가 전체 60% 이상, reveal 조건 70% 이상이다.
- **실패 시 다음 수.** Oracle 선호가 50% 미만이면 평가자가 일관성에 가치를 두지 않는다는 뜻이다.
  - U7(Directed World)과 U5(RAS)의 우선순위를 올린다.
  - 데모를 reveal, 거울, 줌, 8K·120fps처럼 경쟁 모델이 **물리적으로 할 수 없는** 형식으로 재설계한다.
  - 단일 프레임 미감 격차 수치가 그대로 다음 분기의 목표치가 된다.
  - 산출물 자체로 공백 G2(공개 frozen-time 벤치마크가 없음)를 채운다.

### E3. 단일 장면 WDM 대 "생성 후 재구성" (D20–D75, H100 16–32장)

- **가설.** 세계 하나(코스는 재구성, 미세 2–3개 레벨은 생성)를 이미지 teacher critic으로 분포 매칭하면 세 가지가 동시에 성립한다.
  - (i) teacher 샘플만큼 선명하다.
  - (ii) 궤도 촬영에서 GeCo가 노이즈 바닥 수준이다.
  - (iii) CAT3D/Difix 계열보다 선명하다.
  - amortization 이전에 메커니즘 자체를 반증 가능하게 검증하는 실험이다.
- **셋업.**
  - DL3DV 장면 20개를 쓴다. 약 4cm 셀까지를 재구성으로 만들고 저역통과해 디테일을 지운다.
  - 2cm, 1cm, 5mm 레벨은 UKA atom으로 생성한다.
  - 장면별 VSD 방식(fake-score LoRA rank 64), 등척 뷰와 시네마틱 뷰, 자기 캡션, APG, 세계 고정 노이즈를 쓴다. 장면당 H100 8장 × 6–10시간이다.
  - 기준선:
    - (a) SEVA로 24–80뷰를 생성한 뒤 3DGS를 피팅
    - (b) SyncTweedies식 동기화
    - (c) 피팅 후 Difix3D+식 1-step 보정과 재증류
- **성공 기준.**
  - 같은 footprint에서 상위 2 옥타브 Laplacian 에너지가 teacher 샘플의 0.9배 이상.
  - 궤도 GeCo ≤ 0.03.
  - Q-Align이 teacher 대비 −0.2 이내(5점 척도).
  - 채도 드리프트 5% 이하.
  - 최강 기준선 대비 궤도 영상 2AFC 65% 이상.
- **실패 시 다음 수.**
  - 과포화나 Janus가 남으면, critic을 S²R식 동기 샘플링(소유권 annealing)으로 바꾼 하이브리드로 간다. teacher가 판정과 함께 일부 그리기도 맡는 형태다.
  - 블러가 남으면, 실사 4K 캡처에 UKA를 직접 피팅해 표현력 상한부터 측정한다. 셀당 atom 수 $K_l$을 늘리거나 cosine 변조 커널을 켠다.

### E4. Perceptual-Scale Hierarchy 합성 스트레스 테스트 (D30–D80, H100 8장)

- **가설.** 대역만 쓰는 국소 연산자는 긴 결맞음 구조와 에너지가 큰 작은 원소에서 실패한다. PSH는 같은 연산량으로 이를 고친다.
- **셋업.**
  - Blender와 Infinigen 절차적 장면 2,000개를 만든다. 곡면 벽돌벽, 타일, 펜스, 50–500m 거리의 송전선, 간판 텍스트, 야경 광원, 눈 글린트를 포함한다.
  - GT 세계에서 구조 라벨을 추출한다(GT라서 정확하다).
  - 4레벨 STRATA와 반경 8셀짜리 국소 flow 연산자(약 150M)를 두 버전으로 학습한다. 대역만 쓰는 버전과 PSH 버전이다.
  - 측정: 블록 경계 위상 오차(국소 FFT), 줌 OCR의 CER, f/1.4 + 5레벨 게이팅에서 보케 볼 재현율과 에너지 오차(경로추적 GT 대비), 항공샷에서 가는 선의 대비.
- **성공 기준.**
  - 경계의 95%에서 위상 오차 < π/8 (대역만 쓰는 버전은 거의 균일 무작위일 것으로 예상).
  - CER ≤ 5%.
  - 보케 볼 재현율 ≥ 90%, 에너지 오차 ≤ 10%.
  - 선 대비가 GT의 20% 이내.
  - 이 모든 것을 게이팅 연산량 절감을 유지한 채로 달성.
- **실패 시 다음 수.**
  - 위상 퍼텐셜이 유기물(나뭇결, 머리카락)에 너무 경직되면, 신뢰도 가중 캐리어 혼합과 orientation-only 캐리어(phasor식)를 쓴다.
  - 승격 원소 수가 폭발하면, 에너지 클러스터링으로 lightcuts식 계층 POINT를 만든다.

### E5. RAS 대 완전 역렌더 PBR: CG 룩 테스트 (D45–D90)

- **가설.** 정지 세계에서 RAS는 teacher의 룩을 지킨다(단일 프레임 2AFC에서 teacher 원본 대비 45% 이상). 완전 역렌더 PBR은 이를 잃는다(30% 이하). 거울 일관성은 두 방식이 같다.
- **셋업.** E3의 장면 30개를 두 셰이딩 모드로 렌더하고, 양쪽에 같은 림 라이트 리그를 건다. 측정 항목은 다음과 같다.
  - 이중 그림자 지수
  - 거울 반사 오차 (같은 세계의 경로추적 GT 대비)
  - 하이라이트 고착 지수 (카메라가 움직일 때 하이라이트가 표면에 붙어 따라오는지)
- **성공 기준.** 위의 Elo 수치. 거울 오차는 두 모드 모두 PSNR ≥ 32dB. 하이라이트 고착 상관 < 0.1.
- **실패 시 다음 수.**
  - $L_d$에 스페큘러가 구워져 하이라이트가 고착되면, 다시점 저백분위 분리를 강화하고 WDM 중 $L_d$의 시점 분산 벌점을 키운다.
  - 리그 델타가 플라스틱처럼 보이면, IC-Light식 재조명 prior로 $\Delta L_k$의 이미지 공간 보정 헤드를 붙인다. 픽셀 국소 연산이라 광선 국소성 원칙이 유지된다.

### 일정과 Day-90 결정 게이트

| 주 | 0–4 | 4–8 | 8–11 | 11–13 |
|---|---|---|---|---|
| E1 | ■■■■ | | | |
| E2 | ■■ | ■■■■ | ■ | |
| E3 | | ■■■ | ■■■ | ■ |
| E4 | | ■■■ | ■■■ | ■ |
| E5 | | | ■■ | ■■ |

| 결과 | 다음 분기 |
|---|---|
| E1, E3 통과 | amortized WDM 본학습 착수 (코스 레벨과 미세 3레벨) |
| E1 실패 | per-token-$t$ 어댑터 또는 픽셀 미세 전문가 경로 |
| E2에서 Oracle 미달 | Directed World와 reveal 중심 데모를 1순위로 |
| E4 통과 | Powers of Ten(샷 1)과 네온 거리(샷 5)의 생산 착수 |
| E5 통과 | 거울·유리 복도(샷 2)와 재조명(샷 8) 착수 |

---

## 4. 개정 제안서: STRATA v0.2 전체본

*부제: 카메라의 해상도 질서대로 세계를 퇴적하고, 이미지 디퓨전은 그 퇴적을 판정한다.*

### 4.0 테제와 변경 요약

> 비디오 AI는 매 프레임 세계를 다시 상상한다. STRATA는 세계를 한 번, **카메라가 분해할 수 있는 질서대로** 퇴적한다. 이미지 디퓨전은 각 층을 등척 시점에서 판정한다. 일관성은 구조에서, 디테일은 수요에서, 미감은 판정과 연출에서 나온다.

| 영역 | v0.1 | v0.2 |
|---|---|---|
| 레벨 배정 | 등방적 파장 대역 | **PSH**: 크기, 가시도, 결맞음 중 최솟값. 구조 5종을 승격 |
| teacher 인터페이스 | 픽셀별 $T[p]$ | **등척 대역 시점** + 스칼라 $t$ + 자기 캡션 + APG |
| 부트스트랩 | S²R 데이터 엔진 | **WDM**: teacher는 심판. S²R은 히어로 폴리시로 강등 |
| 기반암 | 다시점 생성 → 재구성 | 재구성 prior와 WDM으로 학습한 **코스 월드 모델의 전역 1패스** (내부 포함) |
| LOD | 대역 atom + α 페이드 | **누적 디코드 + dual-cut λ-블렌딩**: 팝핑이 구조적으로 0 |
| 외관 정본 | 재질(PBR) + 잔차 | **RAS**: radiance 정본, 물리는 스페큘러와 변화분만 |
| 프리미티브 | 5종 atom | **UKA** 하나 + 유전체 SDF 채널 |
| 렌더러 | Mode A/B | **3DGRUT 하이브리드 하나** (학습과 최종이 같은 코드) |
| 미감 | 세계 + 샷 조명 리그 | **Directed World**: 의도 조건, 경로 보상, DoP 공동 최적화, shot dress |
| 미니맥스 | 디퓨전 손실 에너지 | DMD 불일치 에너지 |

### 4.1 제1원리

- **콘 질의 가능한 세계** (유지). $\mathcal{Q}_W(\mathbf{o},\mathbf{d},\rho)\to L\in\mathbb{R}^3_{\ge0}$이고, 카메라는 이 질의들의 적분 연산자다(v0.1 §1의 식).
- **광선 국소성 원칙** (유지). (세계, 광선)에 대한 결정론적 함수는 시간적으로 일관된다. 프레임 단위 latent가 없는 픽셀 국소 신경 연산은 깜빡일 수 없다.
- **순수성 v2.** $W=G(\text{seed},y)$. 임의의 수요 순서 순열 $\pi$에 대해 $\mathrm{realize}(\pi(\mathcal{D}))\equiv\mathrm{realize}(\mathcal{D})$다. 결정론적 커널을 쓰면 비트 단위로 같고, 그렇지 않으면 커밋 메모이제이션이 정본을 고정한다. 조건 $y$(의도 포함)가 바뀌면 세계가 바뀌는 것은 허용된다.
- **비회귀 조항 (non-regression clause).** 아래 세 조건을 지키면 순차 hole-filling이 아니다. CI에서는 수요 순서를 무작위로 8번 섞어 Merkle 루트가 같은지 검사한다.
  1. 의존 그래프는 스케일 DAG와 같은 레벨의 동기식 Jacobi 라운드뿐이다. 방문 순서 간선은 없다.
  2. 어떤 영역도 **다른 시점에서 먼저 생성된 렌더 픽셀**을 조건으로 생성되지 않는다. 조건은 오직 더 거친 커밋 레벨에서 온다.
  3. completion은 스펙트럼적이다. 기존 지지 영역(support) 안에 고주파를 더할 뿐 영역을 넓히지 않는다. 유일한 예외는 전역 1패스 기반암이다.

### 4.2 Perceptual-Scale Hierarchy와 Scale-Time Duality v2

- **격자.** $h_l=h_0\,2^{-l}$, $h_0=4096$m. $l_g$는 셀 크기 1–8m 구간(약 $l=9$–12)이다.
- **원소 배정.**
$$l^\star(e)=\min\big(l_{\text{size}}(e),\ l_{\text{vis}}(e),\ l_{\text{coh}}(e)\big),\qquad l_{\text{vis}}=\min\Big\{l:\tfrac{\|k_{h_l}*a_e\|_\infty}{k_{h_l}*\bar L}\ge\gamma_{\text{vis}}\Big\},\qquad l_{\text{coh}}=\Big\lceil\log_2\tfrac{h_0}{\ell_{\text{coh}}}\Big\rceil$$
- **Duality v2.** 레벨은 원소가 픽셀 위로 처음 떠오르는 스케일이고, 이것이 디퓨전에서 그 원소가 결정되는 시간이며, 카메라 거리로 환산하면 $l^*(z)=\lceil\log_2(h_0f_{px}/\kappa z)\rceil$이다.
  - 확률적 텍스처에서는 v0.1과 같다.
  - 선, 빛, 결맞음 구조에서는 더 거친 레벨로 간다. 실제 디퓨전이 이것들을 일찍 결정하는 것과 맞는다.
- **구조 5종.** CARRIER, CURVE, SYMBOL, POINT, GLINT. 각 구조의 텐서 구성은 §1 F4 표와 같다.
- **구조 라벨 추출 (Stage A 데이터).**
  - CARRIER: 재구성 텍스처의 국소 FFT로 지배 파수벡터를 찾고, Goldstein 위상 언래핑으로 $\psi$를 얻는다.
  - CURVE: 3D 얇은 구조 검출.
  - SYMBOL: 프레임 OCR 결과를 평면에 리프팅한다.
  - POINT: HDR 하이라이트 검출.
  - GLINT: 비디오의 시간적 반짝임 통계.
  - 합성 데이터에서는 GT를 그대로 쓴다.

### 4.3 자료구조: 텐서 명세

```
World v0.2
├─ meta       : seed(u128), units=m, up=+Z, h_0, L, l_g, y(intent), merkle_root
├─ lattice[l] : sparse hash (l,i,j,k:int32) → Cell              (fVDB/NanoVDB 스타일)
├─ vtx[l]     : sparse hash (l,i,j,k) → CarrierVertex           (CARRIER가 있는 레벨만)
├─ structs    : struct_id → Structure{type, level, inst, params}; cell→[struct_id] 공간 인덱스
├─ iface[l]   : narrow-band SDF (f16/voxel-corner)              (DIEL이 있는 레벨만)
├─ far_field  : r>R_far 구각: multi-band octahedral HDR + depth + 물리 하늘 파라미터
├─ objects    : inst_id → {class, prompt, xform, locks, phys(P2)}
├─ world_lights / shot_rigs / shot_dress(CoW branches)
└─ caches     : UKA atoms(LRU), BVH per level, caption cache, NRC(리그와 재조명용)
```

**Cell v0.2** (미세 레벨 기준 약 50–60B)

| 필드 | 타입 | 의미 |
|---|---|---|
| `z` | int8/f16[d_l], d_l=64/32/16 | 잔차 latent (VAR식) |
| `occ` | u8 | 자식 점유 마스크 |
| `split` | f16 | 더 미세한 레벨이 존재할 확률 |
| `sem` | f16[8] | 의미 임베딩 (의미 효도의 대상) |
| `inst`, `body` | u32, u32 | 객체 소유권, Phase 2 바디 |
| `sref` | u32 | 이 셀이 참조하는 구조 리스트 오프셋 |
| `unc` | f16[2] | aleatoric, epistemic 불확실성 |
| `own` | u16 | 대역 저술권 (critic 뷰 또는 폴리시 뷰 id) |
| `state` | u8 | virtual, decoded, committed, locked |
| `hash` | u64 | Merkle 콘텐츠 해시 |

**CarrierVertex.** $\psi$ f32[2] (사이클 단위, $10^4$ 사이클에서도 약 0.001 정밀도), $A$ f16. 정점당 10B다.

**UKA atom** (디코드 캐시, 약 64B)

| 그룹 | 필드 |
|---|---|
| 기하 | μ f16×3 (셀 상대좌표), q (32bit packed), s f16×3, 불투명도 σ f16, 커널 차수 n u8, cos 변조 (k f16×2, 위상 f16) 옵션 |
| 외관 (RAS) | $L_d$ RGB9E5 (HDR), $F_0$, roughness, aniso (u8×3), 법선 오버라이드 oct16 |
| 페이로드 | 종류 u8 + 인덱스: SGGX (f16×6), BCSDF (f16×3), 글린트 참조 |
| 메타 | own u16, unc u8 |

**메모리** (Powers of Ten, 8K, 10초)

| 항목 | 규모 |
|---|---|
| 사용 레벨 | 12–26 |
| 고유 셀 | 40–160M (latent int8 기준 1–4GB) |
| 승격 구조 | $10^5$–$10^6$ (수십 MB) |
| 프레임당 UKA | 10–30M (dual-cut 구간은 최대 2배) |

정식 포맷은 latent와 구조이고, atom은 캐시다.

### 4.4 생성 모델

- **분해.** $\Sigma_l$은 레벨 $l$에 배정된 구조다.
$$p(W\mid y)=p\big(S_{\le l_g},Z_{\le l_g},\Sigma_{\le l_g}\mid y\big)\prod_{l>l_g}p(S_l\mid\cdot)\;p\big(z_l,\Sigma_l\mid S_l,Z_{<l},\Sigma_{<l},y\big)$$
- **Coarse World Model (CWM)**, $l\le l_g$, 전역 1패스.
  - 희소 transformer로, full attention 토큰 수는 약 256k 이하다.
  - 토큰 종류는 셀 토큰, 정점 토큰(CARRIER의 $\psi$ 회귀), 구조 집합 토큰이다. 구조 집합은 DETR식 집합 헤드로 예측하고 학습 때 Hungarian 매칭을 쓴다. SYMBOL 문자열은 소형 AR 헤드가 만든다.
  - 조건: LLM 텍스트 인코더, Plücker 임베딩을 붙인 히어로 스틸 토큰, 의도 토큰(경로 스플라인 제어점, 렌즈 계획), 레벨별 register 토큰 약 64개(태양 방향, 노출, 스타일).
  - 미터 단위 3D RoPE를 $h_l$로 스케일한다.
- **Local Band Operator (LBO)**, $l>l_g$.
  - 반경 $r_l$짜리 윈도우 어텐션이고, few-step($K\le4$)으로 증류한다.
  - 입력:
    - 부모의 누적 문맥
    - 캐리어 평가값 $(u_j,\ \mathrm{emb}(\xi_{n}),\ \mathbf{k})$
    - 곡선 국소 좌표
    - 셀 노이즈 $\xi=\mathrm{Philox}(\text{seed},l,i,j,k,\text{step})$와 구조 노이즈 $\xi_n=\mathrm{Philox}(\text{seed},\text{inst},n_1,n_2)$
  - Jacobi 패스는 $P\le2$회다. VP 블렌딩은 확률적 잔차에만 적용한다.
  - 수용장이 유계라서 의존 원뿔이 정확히 알려지고, 지연 평가도 정확하다.
- **디코더.** $D_l(Z_{\le l},\Sigma_{\le l})$는 **완전한** UKA 집합 $\hat W^{(l)}$을 낸다. atom 오프셋은 tanh로 셀 근방에 묶는다.

### 4.5 학습

**World VAE**
$$\mathcal{L}_{\text{VAE}}=\sum_l\mathbb{E}_{C\sim\Pi_l}\big[\ell_1+\lambda_s(1-\text{SSIM})+\lambda_p\text{LPIPS}+\lambda_d\|\Phi_{\text{DINO}}(I)-\Phi_{\text{DINO}}(\hat I)\|\big]+\lambda_f\mathcal{L}_{\text{fil}}+\lambda_{\text{sem}}\mathcal{L}_{\text{sem-fil}}+\lambda_a\mathcal{L}_{\text{adv}}+\beta\,\mathrm{KL}+\lambda_r\mathcal{L}_{\text{reg}}+\lambda_{eq}\mathcal{L}_{\text{equi}}$$
- $\Pi_l$은 레벨 $l$이 최미세 필요 레벨이 되는 band-matched 카메라 분포이고, 타깃은 픽셀 프리필터를 적용한 GT 렌더다.
- $\mathcal{L}_{\text{equi}}$는 서브셀 평행이동 등변성 손실이다(StyleGAN3의 교훈).
- $\mathcal{L}_{\text{reg}}$는 불투명도 희소성, 스케일 범위, $L_d$의 시점 분산 벌점이다.

**Stage A: 현실 prior.**
- 데이터: DL3DV-10K, ScanNet++, SpatialVID 정적 구간, Sekai, MatrixCity(항공에서 거리까지), Infinigen 합성(내부 포함, 재질 GT), Objaverse-XL 배치.
- 초기화와 피팅: DA3, VGGT-Ω, AnySplat로 초기화하고 STRATA로 피팅한다.
- 관측되지 않은 셀은 마스크한다.
- CWM과 LBO를 flow matching으로 학습한다: $\mathbb{E}\|v_\theta(z_l^t,t,l,\text{ctx})-(\epsilon-z_l)\|^2$.

**Stage B: WDM.**
```
WDM_STEP(batch):
  for (seed, y) in batch:
    W_c ← CWM_θ(seed, y)                                   # 전역, few-step
    l ~ p(l);  b ~ Demand(Π_shot(y)) ∪ Uniform(surface) ∪ M  # 학습 블록 샘플
    W_l(b) ← LBO_θ(cone(b,l); W_c, Σ, ξ, ξ_n)               # 유계 원뿔
    Cams ← BAND_ISO(b,l)[4] ∪ Π_cine(y)[2]
    for C in Cams:
      x ← RenderDualCut(W, C);  u ← E_vae(T_disp(x))
      c ← CaptionCache(b,l,C) or VLM(x, sem, scale)
      t ← U[t_end(l), t_start(l)] if C∈ISO else U[0.02, 0.98]
      u_t ← (1−t)u + tε
      g ← w_t·( x̂0_fake(u_t,t,c) − x̂0_real^{APG}(u_t,t,c) )
      L_DM  += ⟨stopgrad(g), u⟩                            # ∇ = g·∂u/∂θ
      L_GAN += softplus(−Disc(feat_fake(u_t,t)))           # real: footprint 정합 실사 + 폐쇄형 teacher 샘플
    L = L_DM + λ_G L_GAN + λ_f L_fil + λ_sem L_sem-fil + λ_ocr L_ocr + λ_r L_reg
    θ ← θ − η∇θ L      # ISO 카메라의 그래디언트는 레벨 l로만 (Z_<l은 stop-grad)
  fake score LoRA: 생성기 1스텝마다 렌더로 denoising 5스텝 (TTUR)
  M ← top-k_{C∈Π_cine} ‖x̂0_fake − x̂0_real‖ (CMA-ES)
```

**Stage C: Render-RL과 Directed World** (§4.8). 코스 레벨은 Flow-GRPO/DanceGRPO식, 미세 레벨은 보상 가중 또는 적대적 증류로 학습한다. 보상은 이미지가 아니라 경로를 따라 렌더한 영상에서 계산한다. 일관성 항은 필요 없다. 구조가 이미 보장한다.

**순수성 증류.** 교사는 넓은 윈도우를 쓰는 멀티스텝 모델이고, 학생은 $K\le4$, 엄격한 반경 $r_l$, Philox 노이즈를 쓴다. consistency, shortcut, adversarial distillation으로 학습한다.

**비용 감각.** 본학습은 수천 H100-일이다. 추론은 미세 토큰 약 $10^8$개 × 4스텝 × 토큰당 약 0.4 GFLOP으로 약 $1.6\times10^{17}$ FLOPs이고, H100 8장에서 약 1분이다. lens gating이 이를 더 줄인다.

### 4.6 S²R-Polish (test-time 히어로 옵션)

```
POLISH(path, budget):
  D ← Demand(path) ∩ HeroBlocks
  parallel for (l, b) in D (레벨 오름차순, 레벨 안에서는 순서 무관):
    V ← BAND_ISO(b,l) ∪ (HeroCams ∩ b)
    x_v ← (1−t_s)·Render_v(W) + t_s·ε_v^{world-anchored};   t_s = t_start(l)
    repeat K∈{1,2}:
      x̂0_v ← Teacher(x_v, t, c_v)            # 스칼라 t, 자기 캡션, APG
      B ← LIFT(W^{(l-1)}, {x̂0_v}, ω(τ_k))   # 소유권 annealing
      B ← ARBITRATE(B)                       # 패럴랙스 → BRDF 로브 → 환각 충돌은 소유권으로 선택(평균 금지)
      x_v ← re-noise(Render_v(W ⊕ B), ε̃(seed,k))
    COMMIT(b, l, B);  unc_epi ← cross-view residual var
```
소유권 가중은 $\omega_{v,p}=\operatorname{softmax}_v(-\tau_k[\log\phi_v-\log|\cos\theta_v|])$이고 $\tau_k$는 점점 커진다. 순수성과 비회귀 조항을 지킨다. 조건은 더 거친 커밋 레벨에서만 온다.

### 4.7 렌더링과 물리 카메라

**GATHER v2**
```
GATHER(rig R, shutter [t0, t0+Δt]):
  for each primary sample (3DGUT raster, 또는 RT):
    λ ← log2(h_0 / (κ·φ_eff))   # φ_eff = pixel ⊕ CoC ⊕ motion ⊕ rolling (공분산 대수)
    I_a ← trace/raster with level-mask ⌊λ⌋;  I_b ← level-mask ⌊λ⌋+1  (f∈(0.05,0.95)일 때만)
    I ← (1−f)I_a + f I_b
  promoted: POINT/GLINT(top-k) → 조리개 스프라이트;  CURVE → 해석적 커버리지(CoC 확장);  SYMBOL → MSDF
  virtual cell을 만나면 DEMAND (값 불변, 순수)
```

**셰이딩 (RAS).**
$$L_o=L_d+L_s^{RT}(F_0,\alpha,\mathbf{n};L_W)+\sum_k\Delta L_k$$
- 가산 리그: $\Delta L_k=(\hat\rho/\pi+f_s)E_kV_k$
- 감산 리그: $L_d(1-\eta_k)$
- 재조명: 비율 전송 $L_d\cdot\hat E_{\text{new}}/\hat E_{\text{old}}$. `unc`가 낮은 곳만 PBR로 승격한다.
- 볼륨(SGGX)과 스트랜드(BCSDF)는 페이로드별 위상함수로 단일 산란만 계산한다. 다중 산란은 $L_d$에 구워져 있다.

**광수송.** 3DGRUT 하이브리드 하나를 쓴다. primary는 3DGUT 래스터, secondary는 3DGRT 레이트레이싱이다. 유전체는 SDF 레이마칭으로 처리하고 분산(Abbe 수)과 선택적 커스틱을 지원한다. Phase 1에는 GI 경로추적이 없다. NRC는 리그와 재조명용으로만 쓴다.

**공분산 대수** (v0.1 §5.2 유지).
- 화면 공분산: $\Sigma'=J\Sigma J^\top+\Sigma_{\text{pix}}+\Sigma_{\text{dof}}(z)+\Sigma_{\text{mb}}$
- 착란원: $c(z)=\frac{f^2}{N(z_f-f)}\frac{|z-z_f|}{z}$, $\Sigma_{\text{dof}}=(c/4)^2I$
- 모션블러: $\Sigma_{\text{mb}}=\frac{\Delta t^2}{12}\dot{\mathbf m}\dot{\mathbf m}^\top$
- 에너지 보존: $\alpha'=\alpha\sqrt{\det\Sigma^0/\det\Sigma'}$

**Lens-gated 생성.** 확률적 대역은 $\phi_{\text{eff}}$에 따라 생략한다. 예를 들어 50mm f/1.4, 초점 2m, 배경 20m라면 약 5레벨을 생략하므로 배경 생성 비용이 약 1/1000로 줄어든다. 승격 구조는 면제한다.

**렌즈, 셔터, 센서** (v0.1에서 유지하는 목록).
- 렌즈: 렌즈 처방전을 레이트레이싱해 출사동 LUT를 만든다(cat-eye, 블레이드, onion-ring, 구면색수차 프린지). 그 외 Hullin 고스트 플레어, FFT 베일링 글레어, 회절 스타(짝수 N이면 N개, 홀수 N이면 2N개), 아나모픽, 렌즈 브리딩, 횡·종 색수차, $\cos^4$ 비네팅과 기계적 비네팅.
- 셔터: 롤링 $t(y)=t_0+y\,\tau_{\text{line}}$, 셔터각.
- 센서: 광자 Poisson 노이즈, 리드 노이즈, full-well 클리핑, ADC, Bayer와 디모자이크, 로그 커브, look LUT.
- 필름: 할레이션, Newson Boolean 그레인(프레임마다 시드), 게이트 위브.
- Neural Sensor: 픽셀 국소 연산이라 깜빡임이 원천적으로 불가능하다.

**8K·120fps 예산.** primary 2배(dual-cut), 2차 레이 1–4개 기준 GPU당 3–8fps로 추정한다. 10초 1200프레임은 GPU 8장으로 수 분이다. 24fps에서 120fps로 올려도 생성 비용은 1배이고 렌더 비용만 5배가 된다.

### 4.8 Directed World, DoP, 샷 조명과 샷 세트

- **의도 조건화와 보상.** U7의 $R(W;y)$를 쓴다. 경로 밖 견고성은 미니맥스 아카이브 $\mathcal{M}$이 맡는다.
- **DoP 공동 최적화.**
  1. 경로 prior(DataDoP/CameraBench 기반 Traj-DiT)에서 경로를 뽑는다.
  2. 세계를 생성한다(CWM 1패스).
  3. 구도, 시차, 역광 보상의 그래디언트로 스플라인을 고친다.
  4. 2와 3을 2–3회 반복한다.
  - 카메라 동역학은 실제 촬영 궤적의 jerk 스펙트럼에 맞춘다.
- **shot_rigs** (유지). 카메라 상대 좌표의 필름 조명이고 RAS의 가산·감산 델타로 렌더한다.
- **shot_dress** (신규). 샷별 copy-on-write 브랜치로, $l\le l_g+2$에서 소품 배치를 치팅한다. 샷 안의 연속성은 정확히 유지된다.
- **로케이션 모드.** `shot_intent=∅`. 여러 샷이 공유하는 세계로, 루프 클로저와 재방문 데모에 쓴다.

### 4.9 평가: Bullet-Time Bench v1

- **비교 대상.**
  - 비디오: Seedance 2.5, Wan 3.0, Kling 3.0 Omni, Veo 3.1, Gemini Omni 1.1 Flash, MiniMax H3.
  - 세계 계열: Marble 1.1 Studio export, HY-World 2.1, FlashWorld, Voyager, Lyra 2.0(평가 전용 라이선스).
- **프로토콜.** E2의 내용 정합 i2v를 쓴다. 우리 렌더나 실사의 첫 프레임을 공통 입력으로 준다. 프롬프트는 frozen-time 지시를 포함하고 8개 데모 샷 카테고리로 나눈다.

| 지표 | 정의 | Phase 1 목표 |
|---|---|---|
| GeCo fused | motion과 structure 잔차 | ≤ 0.03 (실영상 바닥 약 0.02) |
| LCE / Revisit | 왕복 궤적 LPIPS, DINOv2 | ≤ 0.05 |
| MEt3R | 다시점 특징 일관성 | 실영상 수준 |
| 효도 줌 점수 | $\text{LPIPS}(\downarrow_k I_{\text{close}},\ \text{crop}(I_{\text{far}}))$ | ≤ 0.1 |
| 텍스처 헤엄 지수 | $\|\text{flow}(\text{HP}(I_t),\text{HP}(I_{t+1}))-\text{flow}_{\text{rigid}}\|$ | 센서 노이즈 바닥 |
| 해상도 | 세계 안에 놓은 ISO 12233 차트의 MTF50 | ≥ 0.25 cy/px @8K |
| 광학 정합성 | CoC와 깊이의 얇은 렌즈 피팅 잔차, 회절 스파이크 수, 거울 오차(경로추적 GT 대비) | 잔차 < 5%, 스파이크 수 정확, 거울 PSNR ≥ 32dB |
| 승격 원소 | 보케 볼 재현율, 항공샷 선 대비, 줌 OCR CER | ≥ 90%, ±20%, ≤ 5% |
| 사람 선호 | 영상 2AFC Elo, 단일 프레임 미감(teacher 대비) | ≥ 60% (reveal 조건 ≥ 70%), ≥ 45% |
| 과스무딩 지수 | 카메라 jerk 고주파 파워와 실제 촬영의 비 | 0.8–1.2 |

### 4.10 Phase 2: STRATA-4D v0.2

- **바디 계층** (유지). static, rigid, articulated, deformable(MPM/FEM), cloth, strand, fluid/smoke, particle/event. atom은 바디에 귀속되며 바디 id, 로컬 좌표, 스키닝 가중치를 가진다.
- **시간 방향 PSH.**
  - 운동 원소도 가시도와 결맞음으로 시간 레벨을 배정한다. 셔터 $\Delta t$보다 빠른 대역은 대역 에너지에서 계산한 $\Sigma_{\text{mb}}$로 통계적으로 렌더한다.
  - 프레임 나이퀴스트와 $1/\Delta t$ 사이 대역은 명시적으로 유지한다. 그래서 왜건 휠 효과가 진짜로 재현된다.
  - 장거리 시간 결맞음(주기 운동, 걸음걸이)은 시간 캐리어 $\psi(t)$로 둔다.
- **물리가 고주파 생성기다.** CWM은 초기 상태, 물리 파라미터, 의도와 힘(저주파)만 생성한다. 접촉, 충돌, 천이 탁 펴지는 순간은 시뮬레이터(MPM, rigid, cloth, SPH, NGFF식 학습 힘장)가 만든다.
- **Gabor 물결.** 물의 CARRIER 위상을 분산 관계 $\omega=\sqrt{g|\mathbf{k}|\tanh(|\mathbf{k}|d)}$로 진행시킨다. Phase 1의 구조 원시가 그대로 Phase 2의 동역학 원시가 된다.
- **WDM-4D: 스펙트럼 분업 (랜드스케이프 G8의 공백).**
  - 외관 critic은 이미지 teacher다. 프레임별 등척 뷰에서 공간 대역을 판정한다.
  - 운동 critic은 비디오 teacher다. 480p 이하의 저해상 프록시 렌더(albedo 셰이딩)에서 **저주파 공간 대역의 시간 변화만** 판정한다. 비디오 모델의 공간 화질 약점은 이 역할에서 상관이 없다.
  - 데이터는 SpatialVID, Stereo4D, OmniWorld를 MoVieS와 VGGT-Ω로 재구성해 쓰고, 시간과 카메라는 BulletTime식으로 분리한다.
- **영속성.** 정적인 부분은 커밋된 지층으로 남는다. 사건은 시간 청크 Merkle 위의 copy-on-write 편집이다. 세계는 남고 사건이 일어난다.

### 4.11 신규성 장부 v0.2 (요약. 자세한 내용은 §2)

| 확신도 | 항목 |
|---|---|
| **높음** | PSH와 Duality v2, dual-cut λ-연속 생성 LOD, 구조 주소 순수성 |
| **중상** | band-isometric 이미지 critic WDM. 수렴 리스크가 있고 E3가 검증한다. |
| **중** | RAS를 생성 세계의 정본으로 쓰는 것, lens-gated 생성과 승격 면제, DMD 불일치 미니맥스, Directed World |
| **중하** | 스펙트럼 분업 WDM-4D. 선례가 없고 Phase 2에서 검증한다. |
| **부품** | 물리 카메라 체인, 3DGRUT, SGGX, 글린트, phasor, DMD2, 피드포워드 재구성, 벤치 지표 |

### 4.12 로드맵과 팀

| 기간 | 목표 | 산출물 |
|---|---|---|
| D0–90 | E1–E5 | Bullet-Time Bench v0 리포트, 단일 장면 WDM 4K 궤도 루프 데모, 결정 게이트 |
| M3–M9 | amortized WDM (CWM과 미세 3레벨), 방에서 건물 규모, 4K·60fps, Directed World v0 | 데모 샷 2, 4, 7, 8. 비교군 대비 GeCo와 Elo 공개 |
| M9–M18 | 전체 PSH, Powers of Ten(10⁵배 줌), 8K·120fps, 공개 Elo 아레나 | 데모 샷 1, 3, 5, 6. Phase 2 프로토타입(rigid, cloth, 물 캐리어) |

**팀 구성** (5–6명).
- teacher와 WDM 2명
- 표현과 렌더러(UKA, dual-cut, RAS) 2명
- 데이터와 재구성(구조 라벨 추출 포함) 1명
- 평가와 사람 연구 1명

### 4.13 다른 렌즈와의 인터페이스 (API v0.2)

```python
world  = strata.create(seed, y=Intent(prompt_tree, hero_stills, shot_intent=None|PathDist), scan=None)
demand = world.demand(path, lens_rig, shutter, fps, res)   # {(l, block)} ∪ promoted structs; 값 불변
world.realize(demand, mode="native" | "polish")            # 순수·병렬·메모이즈
frames = strata.render(world, path, lens_rig, shot_rig, sensor, look, lod="dualcut")
world.structures(region)                 # CARRIER/CURVE/SYMBOL/POINT/GLINT 조회·편집
world.branch(shot_id).dress(region, op)  # shot dress (CoW, 새 merkle_root)
world.uncertainty(region)                # DoP 플래너용: 아직 결정되지 않은 영역
world.query_cone(o, d, rho)              # 제1원리
assert world.order_purity_check(demand, permutations=8)   # 비회귀 조항 CI
```

### 요약 세 줄

1. **비디오 AI는 매 프레임 세계를 다시 상상한다.** STRATA는 세계를 카메라가 분해할 수 있는 질서대로 한 번 퇴적하고, 카메라로 질의한다.
2. **이미지 디퓨전은 화가가 아니라 심판이다.** 등척 시점에서 한 대역씩만 판정하므로, 평균 낼 뷰가 없고 블러도 없다.
3. **송전선, 보케 볼, 벽돌 줄눈, 간판 글자는 크기가 아니라 보이는 정도로 산다.** 그래서 2km 상공에서 무당벌레까지 내려가는 동안 무엇도 사라지거나 변하지 않는다. 카메라를 한 바퀴 돌려 처음 자리로 돌아와 "봐, 아직 거기 있지?"라고 말할 수 있다.
