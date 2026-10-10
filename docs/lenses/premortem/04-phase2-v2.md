> **출처 메모.** 설계 워크숍에서 [Phase 2 / 4D / 데이터 플라이휠] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 건설적 프리모템과 개정안(v2)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../../PARADIGM.md)의 판결록(§0.5)을 따른다.

---

# 건설적 프리모템: Block Universe Cinema (Phase 2) → v2

> 기준일은 2026-10-10이고, 2026-10-09 지형 리포트를 반영했다. 인용 가운데 확신이 낮은 것은 "(확인 필요)"로 표시했다.
> 각 실패 지점은 [실패 시나리오 → 기술적 원인 → 업그레이드 → 측정] 순서로 쓴다. 목표는 하나도 낮추지 않았고, 2a의 승리 조건은 오히려 올렸다.

---

## 0. 진단 요약

원안의 뼈대는 맞다. 시간은 상태에만 붙고, 확률은 원인에 두며, 카메라는 블록(block)을 읽기만 한다. 이 뼈대가 무너지는 곳은 수식 안이 아니라 **경계면**이다.

| # | 경계면 | 실패 내용 | 업그레이드 |
|---|---|---|---|
| F1 | Phase 1 → Phase 2 | 넘어온 세계가 움직이지 못한다. 숨은 면이 없고, 조명이 bake되어 있고, 객체가 분리되어 있지 않다 | **Edit-Closed World** |
| F2 | 상태 → 픽셀 | 결정론적 셰이더가 공간 방향으로 사후 평균을 내서 CG 룩이 된다 | 상태 조건부 외형, **Matter-Anchored Noise 셰이딩**, 슬라이스 분포 매칭 |
| F3 | 시뮬 → 현실 | 시뮬이 현실보다 깨끗하고, 물리 판사는 현실 영상까지 기각한다 | **PCGD**와 실분포로 보정한 판사(**RGF**) |
| F4 | 비디오 모델 → 우리 | 증인이 자기 병(고주파 결손)을 우리에게 옮긴다 | 대역 분업, 진술 그래프, 순열 대조 ELBO, 실제 증인 |
| F5 | 계획 → 실현 | 혼돈계와 접촉이 많은 계에서 ESA가 수렴하지 않는다 | **Chaos-as-Controllability** (Milestone SMC) |
| F6 | 웹 데이터 → 라벨 | 오디오 타이밍을 믿을 수 없다(AV 오프셋, 후시 foley) | **AV-Physics Timing Solver**와 Blur-as-Measurement |
| F7 | 우리 → 시장 | 스스로 채점하는 시험지를 쓰고, 첫 전장을 잘못 골랐다 | **Factorized Turing Battery**, BUPS Arena, 2a를 Frozen Action으로 |

**v2의 통합 원리.** 블록이 생성 변수다. 세 교사가 각자 자기 대역만 판정한다: **이미지 교사**는 임의 시공간 슬라이스의 공간 사실감을, **실분포로 보정한 물리 검증기**는 불변량과 고주파 동역학을, **실사건 prior**는 사건 타이밍 통계를 맡는다. 비디오 모델은 **저주파 증인**으로 역할을 낮춘다. 교사가 서로의 영역에 손대지 못하도록 기울기 라우팅(gradient routing)으로 막는다.

창업자 테제의 가장 큰 자산은 이미지 디퓨전의 품질이다. 그런데 원안은 Phase 2에서 이것을 Avatar Foundry에만 썼다. v2는 이미지 디퓨전을 움직이는 세계의 **모든 순간**을 판정하는 판사로 되돌려 놓는다.

---

## 1–2. 실패 지점과 업그레이드

### F1. Phase 1 세계가 '움직일 수 없는 세계'로 넘어온다

**실패 시나리오.** 2b 데모에서 배우가 컵을 든다. 컵이 있던 자리의 테이블은 회색 얼룩이거나 floater 구름이다. 컵 그림자는 원래 자리에 그대로 남는다(ghost shadow). 들어 올린 컵은 원래 자리의 조명을 그대로 지니고 있어서 오려 붙인 것처럼 보인다. 컵 테두리에는 테이블 Gaussian 몇 개가 함께 끌려 올라간다.

**기술적 원인.**
1. **기울기가 0이다.** Phase 1의 유력 경로(지형 P1: 렌더 후 이미지 교사로 DMD)에서 Gaussian $i$가 받는 기울기는 $\partial L/\partial\phi_i=\sum_{c,\mathbf{u}}\frac{\partial L}{\partial x_c(\mathbf{u})}\,T_i(\mathbf{u},c)\,\alpha_i(\mathbf{u})\,\frac{\partial \mathbf{c}_i}{\partial\phi_i}$ 이다. 컵 밑 표면은 샘플링한 모든 카메라에서 $T_i\alpha_i=0$이므로 기울기가 정확히 0이다. 그러면 그 영역은 world-native prior가 정하게 되는데, 이 prior를 학습한 lift 데이터(DL3DV, SpatialVID)에도 컵 밑은 찍혀 있지 않다. **데이터와 교사가 같은 자리에서 동시에 비어 있는 것이다.**
2. **조명이 bake된다.** DMD는 "이 뷰에서 그럴듯한가"만 보상한다. 그래서 그림자, AO, 상호반사가 색에 섞여 들어간다. 생성된 세계를 나중에 intrinsic 분해(RGB↔X 등)하려 해도 ill-posed이고, 그 오차는 물체가 움직이는 순간 드러난다.
3. **객체 분할이 사후에 이루어진다.** 장면 단위로 생성된 Gaussian은 객체 경계에 걸쳐 있다. SAM으로 나중에 분할하면 경계 Gaussian이 갈라지면서 '끌림'이 생긴다.
4. 원안 §1.6은 이 문제를 Phase 1 팀과의 **계약**으로 넘겼다. 계약 조항은 있지만 그것을 만족시킬 **학습 신호**가 없다.

**업그레이드: Edit-Closed World (반사실 편집에 닫힌 세계).**

*정의.* 세계 $W$가 편집 분포 $p_{edit}$에 닫혀 있다는 것은 다음을 뜻한다. $e\sim p_{edit}$, $c\sim p_{cam}$, $L'\sim p_{light}$ 어느 것을 골라도 $\mathcal{R}(e(W),c,L')$가 이미지 교사의 manifold 위에 있다.

- **표현.** Gaussian마다 객체 slot $o_i\in\{0..K\}$를 *생성 변수*로 둔다. 사후 분할이 아니다. 외형은 intrinsics $(a_i,n_i,r_i,m_i)$에 작은 잔차 radiance $f_i\in\mathbb{R}^8$를 더한 것이다. 렌더는 $\text{shade}(a,n,r,m;L,\text{vis})+g(f,\omega)$이고, 잔차에 에너지 패널티 $\lambda_{res}\lVert g\rVert^2$를 건다.
- **편집 분포.** $p_{edit}$는 *Phase 2 사건 모델의 저해상 버전 그 자체*다. 포함하는 편집은 객체 $k$ 제거, $k$의 $SE(3)$ 이동, 관절 $j$를 $q_{max}$까지 여는 것(문, 서랍), 광원 교체(태양각, 램프 on/off), 인물 slot 제거다.
- **학습 신호.** DMD2 형태로 쓴다.
$$\nabla_\phi L=\mathbb{E}_{e,c,L',t}\Big[\big(s_{fake}(x_t,t\mid y_e)-s^{img}_{real}(x_t,t\mid y_e)\big)\tfrac{\partial x}{\partial\phi}\Big],\quad x=\mathcal{R}\big(e(G_\phi(\epsilon)),c,L'\big)$$
$y_e$는 편집 내용을 반영한 캡션이다. 배치의 30~50%를 반사실 샘플로 채운다.
- **각 문제가 해결되는 경로.**
  - 컵 밑 면: '컵 제거' 렌더에서 보이게 되므로 기울기를 받는다.
  - bake된 그림자: '컵 제거' 렌더에서는 드리울 물체가 없는 그림자가 된다. 교사가 벌점을 주면 그림자는 visibility 항으로 밀려난다.
  - bake된 조명: '광원 교체' 렌더에서 이중 그림자가 생기고 벌점을 받는다. 그래서 intrinsics 쪽으로 밀려난다.
  - 잘못된 slot: '이동' 렌더에서 찢어짐이 생기고 벌점을 받는다. 그래서 slot이 교정된다.
- **hole-filling이 아닌 이유.**
  - (i) **순서가 없다.** 모든 편집이 매 스텝 같은 $W$에 동시에 기울기를 준다. 확정된 캔버스를 조건으로 빈 곳을 채우는 단계가 어디에도 없다.
  - (ii) 숨은 면은 편집 뷰뿐 아니라 원래 뷰에서도 간접 효과(접촉 그림자, AO, 반사)를 통해 함께 제약된다.
  - (iii) **추론 때는 편집이 없다.** amortized 생성기는 닫힌 세계를 한 번에 낸다. 반사실 렌더는 학습 때만 쓰는 탐침이다. 원안 §1.5의 "카메라가 생기기 전에 세계가 완결된다"는 주장을, v2는 처음으로 학습 신호로 뒷받침한다.
- **Phase 1이 덤으로 얻는 것.** 재조명과 time-of-day가 공짜가 된다. 그래서 Phase 1 팀에게도 이득인 계약이 된다.
- **측정.**
  - RRS(Reveal Realism Score): 드러난 영역에 대한 2AFC.
  - Reveal-MEt3R.
  - GSI(Ghost Shadow Index): $\mathrm{corr}(\Delta\text{luminance}_{k\ \text{제거}},\ \text{shadow}_k)$.
  - 합성 GT(Infinigen/Blender)에서 잰 albedo leak.

### F2. 움직이는 물질이 CG처럼 보인다

**실패 시나리오.** 2c 대화 장면이다. 기하와 연속성은 완벽한데 피부가 플라스틱 같다. 옷 주름은 LBS 고무처럼 미끄러지고, 머리카락 하이라이트는 움직이지 않는다. 전문가 패널은 "게임 컷신"이라고 평가하고, Arena 연기 축은 30%대에 머문다. WorldScore에서 3D 계열이 주관 품질 37~71로 진 패턴이 Phase 2에서 그대로 반복된다.

**기술적 원인.**
1. **공간 방향의 사후 평균.** 원안 §2.8의 셰이더는 결정론적 회귀기다. 그런데 같은 G-buffer에 대응하는 실제 외형은 다봉 분포다(직물 섬유 하이라이트, 모공 스펙큘러, 땀). L2나 perceptual 회귀의 최적해는 $\mathbb{E}[x\mid\text{Gbuf}]$다. 원안이 §2.1에서 비디오 모델을 비판한 사후 평균 회귀가 **공간에서 그대로 재현된다.**
2. **"외형 텐서는 시간 불변"이라는 주장은 identity에 대해서만 맞다.** 주름의 자기그림자, 늘어난 피부의 색 변화, 젖음, 머리카락 Marschner 로브의 이동은 모두 상태의 함수다. 작은 포즈 잔차 MLP $\delta$로는 용량이 모자란다.
3. **Avatar Foundry의 NeRF-W식 처리.** 뷰마다 appearance code를 학습했다가 렌더할 때 버리면 공통분모만 남는다. 이것이 CAT3D식 블러다(지형 §7-5). 미세 불일치를 평균으로 해결하면 품질 상한을 스스로 깎게 된다.

**업그레이드 (a): 상태 조건부 외형. identity는 불변 인자로 표현한다.**
$$A_i(t)=\mathcal{A}_\Theta\big(\theta_i,\ \varphi_i(s(t)),\ L(t),\ \omega\big),\qquad \varphi_i=[\log(F_i^\top F_i)_{(6)},\ \Delta\text{AO},\ \text{접촉압},\ \text{젖음},\ \text{온도},\ \lVert v\rVert]$$
$\theta_i\in\mathbb{R}^{32}$는 Gaussian마다 하나씩 있고 시간에 따라 변하지 않는다. 시간 불변성을 외형 **값**이 아니라 **인자**에 건다. 그래서 identity는 정의상 보존되고, 외형은 물리가 정한 대로 바뀐다.

**업그레이드 (b): Matter-Anchored Noise(MAN) 생성형 셰이딩.**

셰이딩 잔차를 회귀로 구하지 않고 *샘플링*한다. 이때 노이즈를 픽셀이 아니라 **물질**에 붙인다.
- **물질 노이즈장.** canonical 공간에 다해상도 해시 셀 $q$를 두고, 셀마다 i.i.d. $\eta(q)\sim\mathcal{N}(0,I_k)$를 둔다. 픽셀 노이즈는 다음과 같다.
$$\xi(\mathbf{u})=\lvert\Omega_\mathbf{u}\rvert^{-1/2}\sum_{q\in\Omega_\mathbf{u}}\eta(q)$$
$\Omega_\mathbf{u}$는 픽셀 $\mathbf{u}$의 4×4 supersample이 canonical 좌표 G-buffer를 통해 가리키는 셀의 집합이다. 셀 크기가 픽셀 footprint의 1/4 이하가 되는 해상도 레벨을 고른다. 그러면 픽셀마다 분산이 1이고 픽셀끼리 독립이다. Warped Noise의 integral noise를 **변형되는 3D 물질 좌표**로 옮긴 것이다.
- **셰이더.** $r(\mathbf{u})=g_\varphi(\text{Gbuf}_{N(\mathbf{u})},\ \xi(\mathbf{u}),\ \omega)$ 형태의 1-step generator다. 수용영역은 $N(\mathbf{u})=16\times16$으로 제한한다. 학습에는 이미지 교사 DMD, GAN(R1), 그리고 낮은 가중의 paired perceptual loss를 쓴다.
- **일관성 논증.** 같은 물질점이 두 프레임이나 두 뷰에서 같은 G-buffer 이웃과 같은 $\xi$를 가지면 출력도 같다. 출력이 달라지는 것은 상태, 뷰, 조명이 *실제로* 다를 때뿐이다. 그래서 모공의 반짝임이 물질에 붙어서 움직인다. texture swimming과 평균화가 동시에 사라진다.
- **hole-filling이 아닌 이유: Read-Only Camera 불변식.**
  - 렌더 경로에는 $W$에 쓰는 연산이 하나도 없다.
  - $g_\varphi$는 기하, 불투명도, 객체를 만들 수 없다. 출력에 알파 채널이 없고, $\lVert r\rVert\le\tau$로 크기를 제한하며, 실루엣 바깥은 마스크로 막는다.
  - 콘텐츠를 지어낼 자유도가 구조적으로 없다.

**업그레이드 (c): 슬라이스 분포 매칭과 기울기 라우팅.** 4D 블록의 임의 슬라이스 $(c,t,e,L')$를 렌더해서 이미지 교사로 판정한다. 정지 프레임 한 장은 사진 한 장이다. 그래서 이미지 교사는 움직이는 세계의 *순간*을 판정하는 데 가장 적합한 판사다.
- 기울기는 $\{\theta,\Theta,\varphi,\delta,o_i,\text{intrinsics}\}$에만 흘려보낸다.
- 물리 상태 $s(t)$, 사건 $E$, 물성 $\psi$로는 stop-gradient를 건다. 교사가 사진발만 좋은 물리를 만들어 내지 못하게 하려는 것이다.
- 물리 쪽 미감은 F5의 *선택*으로만 반영한다.

**업그레이드 (d): Avatar Foundry v2.** appearance code를 평균하지 않는다. 대신 아바타를 무작위 포즈, 표정, 조명, 뷰로 렌더한 결과가 identity 레퍼런스를 조건으로 한 이미지 교사의 manifold 위에 있도록 분포 매칭을 한다. 그러면 하나의 아바타가 모든 뷰에서 동시에 그럴듯해지는 해로 수렴한다.

**측정.** Appearance Turing(실제 궤적을 고정하고, 우리 렌더와 hold-out 카메라의 실제 footage를 비교), warping error(시간 flicker), 패널이 "CG 같다"고 답한 비율.

### F3. 시뮬레이터가 현실보다 깨끗하고, 물리 판사가 현실을 기각한다

**실패 시나리오.** Rube Goldberg 데모에서 공은 정확히 $g$로 떨어지지만 튀는 모양이 '게임 물리'처럼 보인다. 반발이 너무 일정하고 미끄러짐이 없으며, 쿠션은 젤리처럼 움직인다. 2c에서는 PHC를 추종하는 캐릭터가 상체가 뻣뻣하고 미세하게 떨린다. PLAC 물리 판사에 *실제 영상*을 넣어 보면 상당수가 ghost force 임계를 넘어 비물리적이라고 판정된다.

**기술적 원인.**
1. **모델 불일치.** 반발계수가 일정하고, Coulomb 마찰을 쓰고, 접촉이 이상적이고, 공기저항이 없다. 실제 접촉은 미세 변형, 스핀 커플링, 표면 거칠기 때문에 결과가 흩어진다.
2. **"결과는 결정론" 원칙이 실제 미시 산포까지 지운다.** 그 결과 TSR 중간 대역이 실제보다 낮거나, 시뮬 jitter 때문에 높아진다.
3. **물리 추종 정책의 한계.** 캐릭터 추종 정책은 kinematic 모캡보다 자연스러움이 떨어진다는 것이 알려져 있다.
4. **판사가 우리 시뮬 모델이 맞다고 전제한다.** 실제 영상을 lift해도 모델 불일치와 lift 노이즈 때문에 $F_{ghost}>0$이 나온다. 임계를 낮추면 현실을 기각하고, 높이면 비디오 모델의 엉터리 운동을 통과시킨다. 원안에는 이 임계를 보정하는 절차가 없다.
5. **9종 솔버의 양방향 결합.** 젖은 천, 물속 캐릭터, 움직이는 물체 주변의 연기 같은 장면은 VFX 현장에서도 샷마다 sim TD가 붙는 문제다. 불안정해지고 튜닝 부담이 폭발한다.

**업그레이드: Physics-Constrained Generative Dynamics(PCGD)와 실분포 보정 판사.**

*원칙을 고친다.* "확률은 원인에, 결정론은 결과에"를 **"확률은 원인과 미시 잔차에, 보장은 불변량에"**로 바꾼다.

- **하드 불변량.**
  - identity $\theta$는 바뀌지 않는다.
  - 질량과 Gaussian 수가 보존된다. 위상이 바뀌는 것은 FRACTURE, SPLASH, IGNITE 같은 *사건 게이트*를 통해서만이다.
  - 비관통: $\text{SDF}\ge-1\text{mm}$.
  - 중력은 metric $g$다.
  - 사건 마크에 적힌 충격량 $\mathbf{J}$는 시뮬의 운동량 변화와 ±5% 안에서 일치해야 한다.
- **생성형 동역학.** 모든 물질을 Gaussian과 1:1로 대응하는 통합 입자 기질 위에서 다룬다(PhysGaussian 계열). 강체는 shape-matching 클러스터, 천은 정점 입자, 유체는 입자로 표현한다.
$$s_{t+\Delta}=\Pi_{\mathcal{I}}\Big(\mathrm{Sim}_{base}(s_t,a_t;\psi)+F_\phi(s_t,a_t,\psi,\xi_t)\Big)$$
  - $\Pi_\mathcal{I}$는 불변량 투영이다.
  - $\mathrm{Sim}_{base}$는 Warp/Newton, XPBD, MPM 같은 고전 솔버다.
  - $F_\phi$는 입자 그래프 트랜스포머이고 확률적 잔차 $\xi_t$를 받는다.
  - 결합은 그래프의 *엣지*로 표현되므로 양방향 결합이 기본값이 된다.
  - 학습 데이터는 두 가지다. (i) 고전 솔버 롤아웃으로 커버리지를 확보한다. (ii) 실제 4D lift(Stereo4D, 스포츠 멀티캠, 자체 rig)로 **sim-to-real 잔차만** 배운다.
- **캐릭터.** kinematic motion diffusion을 1차 생성기로 쓴다. 자연스러움은 데이터에서 온다. 물리는 *접촉 창*에서만 PhysDiff식으로 투영한다.
- **판사: Real-Reference Ghost Force(RGF).** 범주별로 실제 lift 영상의 ghost force 분포 $p_{real}(F_g)$를 구한다. 판정은 고정 임계가 아니라 두 표본 검정이다.
$$\text{accept}(h)\iff \Pr_{p_{real}}\big[F_g\ge F_g(h)\big]\ge0.05,\qquad \text{배치 수준: }\mathrm{MMD}\big(\{F_g(h_b)\},\,p_{real}\big)<\epsilon$$
  기준은 "lift된 현실만큼 물리적이어라. 더도 말고 덜도 말고"다.
- **부산물: Reality Residual Budget.** $p_{real}$ 안에 들어가는 미세 잔여력은 *정의상 어떤 판사도 현실과 구별할 수 없다.* 이 예산을 F5에서 보이지 않는 제어 채널로 쓴다.
- **측정.** Motion Turing에서 시뮬과 실제를 구별하는 식별률, 실분포 대비 TSR/CSI 거리, ghost-force AUC(실제 대 생성).

### F4. 증인이 병을 옮긴다

**실패 시나리오.** PLAC 커밋 100만 건으로 EPD를 amortize한다. 6개월 뒤 우리 모션이 Wan처럼 미끈해진다. Spectrum Report에서 우리 TSR이 비디오 모델 쪽으로 이동해 있다. 같은 시기에 ELBO 증인은 동작이 큰 가설을 체계적으로 기각한다.

**기술적 원인.**
1. **스펙트럼 오염.** lift한 제안 궤적에는 비디오 모델의 고주파 결손이 그대로 남아 있다. 물리 판사는 *불가능한* 운동만 잡아낸다. 매끄러운 제스처는 물리적으로 충분히 가능하므로 *밋밋하지만 가능한* 운동은 걸러지지 않는다. amortization이 이 병을 우리 모델로 증류한다.
2. **접근성.** critic 모드는 가중치가 있어야 쓸 수 있다. 최상위 모델인 Seedance 2.5, Wan 3.0, Kling 3.0, Veo는 닫혀 있다. 열려 있는 것은 Wan 2.1/2.2(Apache-2.0), HunyuanVideo 1.5, LTX, SkyReels V3 정도다. 따라서 "경쟁자의 진보가 우리의 연료"라는 말은 닫힌 모델에는 성립하지 않는다.
   - 상용 API 약관은 출력물을 경쟁 모델 학습에 쓰지 못하게 하는 경우가 많다. 모델별로 법무 확인이 필요하다.
   - HunyuanVideo 계열 라이선스에는 지역 제외 조항(EU, 영국, 한국)이 있는 것으로 알고 있다(확인 필요). 한국 법인이라면 Wan 2.x를 우선 써야 한다.
3. **ELBO의 stillness bias.** 가설마다 disocclusion과 렌더 아티팩트가 다르고, 우리 렌더는 비디오 모델 입장에서 OOD다. 동작이 클수록 아티팩트가 많이 드러나므로 정지에 가까운 가설이 유리해진다. GeCo에서 거의 움직이지 않는 LTX의 점수가 좋게 나온 것과 같은 구조다.
4. **lift 실패.** 녹아내리는 제안을 known-shape로 lift하면 쓸모없는 궤적이 나온다.

**업그레이드: 대역 분업, 진술 그래프, 순열 대조 ELBO, 실제 증인.**
- **Spectral Division of Labor.** 범주별 crossover 주파수를 실측한다(§4의 X1).
$$R(f)=\log\frac{P_{gen}(f)}{P_{real}(f)}\ \ (\text{ego-motion 보상 가속도 PSD}),\qquad f_c=\min\{f:\ R(f')<-3\text{dB}\ \ \forall f'\ge f\}$$
  증인의 증거는 $f_c$ 이하 대역에서만 쓴다. 궤적은 $\hat s^{low}_m=\mathrm{LPF}_{f_c}(\hat s_m)$로 저역 통과시키고, 사건 시각은 $\pm1/(2f_c)$ 정도의 거친 창 정보로만 쓴다. 고주파 대역은 PCGD와 실사건 타이밍 prior가 정한다.
- **진술 그래프(Testimony Graph).** 제안 영상을 Reader와 VLM에 넣어 그래프로 바꾼다. 그래프에는 사건 타입, 참여자, 부분 순서, 거친 시간 창이 담긴다. 교차신문은 픽셀 트랙이 아니라 이 그래프를 맞춰 보는 것이다(참여자 정렬, 순서 일치, 시간 창 겹침). 그래서 녹아내림이나 identity drift가 있어도 흔들리지 않는다.
- **Permutation-Contrast ELBO (PC-ELBO).** 오픈 가중치 모델에만 쓴다.
$$\Delta_m(\gamma)=\mathrm{score}\big(\mathcal{R}(h_m,\gamma)\big)-\mathbb{E}_{\pi}\,\mathrm{score}\big(\pi\circ\mathcal{R}(h_m,\gamma)\big)$$
  - $\pi$는 길이 $1/f_c$ 블록 단위의 시간 순열이다. 프레임별 아티팩트는 그대로 두고 시간 구조만 깨뜨린다.
  - 가설과 순열 사이에 common random numbers(같은 $\epsilon$, 같은 $\sigma$)를 써서 분산을 줄인다.
  - 이렇게 하면 비디오 모델이 *이 시간 구조*를 얼마나 믿는지를 아티팩트와 분리해서 잴 수 있다.
  - 원안의 능동 카메라 QBC는 유지하되, 거리 $D$를 $\Delta_m-\Delta_{m'}$로 바꾼다.
- **실제 증인(Real Witnesses).** lift 코퍼스에서 사건 그래프로 실제 클립 top-k를 검색하고, 그 타이밍 통계를 증인으로 쓴다(ReMoDiffuse식 retrieval). 환각이 없고 약관 문제도 없다. 비디오 모델은 *희귀 사건을 상상해 주는 공급원*으로만 남긴다.
- **졸업 기준을 정량화한다.** 인턴 호출 비율과, 증인에 의존한 사건의 TSR 거리를 추적한다.

### F5. 혼돈계에서 사건 계획이 실현되지 않는다

**실패 시나리오.** 도미노와 Rube Goldberg 데모에서 "t=8.0s에 유리잔이 깨진다"는 제약을 준다. ESA를 20회 돌려도 수렴하지 않는다. 수렴하더라도 해 안에 눈에 보이는 '유령 밀기'가 들어 있다.

**기술적 원인.**
1. **기울기가 망가진다.** 접촉이 많은 시스템에서 미분 가능 시뮬의 기울기는 불연속이고 분산이 크다(Suh et al., ICML 2022). 혼돈 연쇄에서는 민감도가 $\lVert\partial s(T)/\partial s_0\rVert\sim e^{\lambda T}$로 커지므로 10초짜리 연쇄에서 기울기가 폭발한다.
2. **CEM이 비싸다.** 제어가 수백 차원이면 롤아웃이 $10^4\sim10^6$번 필요하다.
3. **재계획이 순환한다.** 재계획(B)은 $t^*$ 이후의 사건을 바꾼다. 하드 제약이 여러 개 있으면 계획이 돌고 돈다.
4. **물리 사건과 비물리 사건이 섞여 있다.** INTENT, GAZE_SHIFT 같은 비물리 사건이 하나의 Match()에 들어간다. 그런데 비물리 사건은 시뮬 검출기가 없다.
5. **결정 변수를 제어 $a$로만 잡았다.** 하지만 세계 자체도 우리가 생성하는 것이다.

**업그레이드: Chaos-as-Controllability (Milestone SMC).**

*핵심 통찰은 혼돈이 적이 아니라 레버라는 것이다.* 민감도가 크다는 것은 아주 작은 차이로 다양한 결과에 도달할 수 있다는 뜻이다. 초기조건, 물성, 소품 배치는 아직 아무도 **촬영하지 않은** 자유도다.
- **결정 변수.**
  - $\Delta s_0$: 초기 자세와 속도.
  - $\Delta\psi$: 물성 prior의 신뢰구간 안에서만.
  - $\Delta$layout: 자유 소품의 배치. 레이아웃 prior의 로그확률 손실이 $\epsilon$ 이하여야 한다.
  - 제어 $a$.
  - $\iota$: 접촉 시점의 미세 임펄스. Reality Residual Budget, 즉 $p_{real}(F_g)$의 95분위 이하로 제한한다.
- **제약.**
  - 사건 제약은 감독이 정한 허용오차 창 $(t_e\pm\tau_e,\ \mathbf{x}_e\pm r_e)$로 준다.
  - **잠긴 프레임 제약**: 사용자가 이미 승인한 프레임이 있으면, 그 프레임의 재투영 변화가 0.5px 이하여야 한다.
- **알고리즘.**
```
particles n=1..N (N=4096): Δs0ⁿ ~ N(0, Σ_prior), Δψⁿ ~ p(ψ), layoutⁿ ~ p(layout | W)
for milestone m = 1..M  (사건 제약 시간순):
    GPU 병렬 시뮬 t_{m-1} → t_m + τ_m          # Warp/MJX/Newton
    wₙ ← exp(-d(Êₙ ∩ window_m, e_m)² / 2σ_m²) · p_layout(ₙ) · 1[불변량 I1–I5]
    if ESS = 1/Σwₙ² < N/2: systematic resampling
    rejuvenation:
       (a) 직전 접촉에 미세 임펄스 ι 주입 (Reality Residual Budget 이내)
       (b) 또는 Δs0 섭동 후 t=0부터 재시뮬 (혼돈이 차이를 증폭해 다양성 회복)
return argmax_n  wₙ · HeroScore_n          # 미감은 '선택'으로만 반영
```
  비용은 10s @1kHz × 4096입자로 세대당 약 $4\times10^7$ env-step이다. GPU 강체 시뮬 기준 분 단위로 끝난다.
- **Event Type Contract.** 사건 타입마다 (스키마, 실현기, 검출기, 허용오차, 검증기)를 명시한다. Match()는 타입별로 분리해서 정의한다(v2 §3의 표 참조).
- **Hero-Instant Selection.** 감독이 정한 hero 시각 $t_h$에서 입자마다 슬라이스를 이미지 교사로 점수 매기고, 그 점수를 선택 가중치로 쓴다. 물리는 손대지 않고 사진발은 선택으로 얻는다. bullet time의 정지 순간이 *물리적으로 가능한 것 가운데 가장 아름다운 파편 배치*가 되는 이유다.
- **빌려 올 선행 연구.** Twigg & James 2007(Many-Worlds Browsing), Chenney & Forsyth 2000(plausible multibody solutions), Popović et al. 2000/2003, SAMCON(Liu et al. 2010).

### F6. 데이터의 시간 정보를 믿을 수 없다

**실패 시나리오.** $10^{10}$개 사건 라벨로 EPD를 학습했는데 CSI가 실제 분포와 맞지 않는다. 사건 타이밍 분포도 ±100ms로 퍼져 있다. 원인을 추적해 보니 영화·드라마 클립에서 '음속 보정'이 사건 시각을 틀린 방향으로 옮겨 놓았다.

**기술적 원인.**
1. **AV 오프셋.** 웹 영상은 트랜스코딩, 블루투스 마이크, 편집 때문에 클립마다 수십~100ms 이상의 AV 오프셋이 흔하다. 보정하려는 항(34m 거리면 100ms)과 오차 항의 크기가 같다.
2. **영화·광고 오디오는 대부분 후시 녹음이나 foley다.** 그림에 맞춰 싱크하므로 전파 지연이 0이다. 여기에 음속 보정을 하면 사건 시각이 $d/c$만큼 반대 방향으로 밀린다.
3. **귀속할 수 없다.** 음악이나 내레이션이 덮여 있고, 음원이 여럿이고, mono라서 어떤 onset이 어떤 사건인지 정할 수 없다.
4. **빠른 사건일수록 lift가 약하다.** 빠른 사건에는 블러와 가림이 따르고, lift는 바로 이 구간에서 가장 약하다. 날카로운 움직임을 가르쳐야 할 데이터가 가장 부정확한 셈이다.
5. **VFR 타임스탬프 지터와 rolling shutter.**

**업그레이드 (a): AV-Physics Timing Solver와 Sync Style 분류.**

클립마다 다음 모델을 세운다.
$$t^{on}_i=t^{ev}_i+\beta\,d_i+\delta_{clip}+n_i,\quad n_i\sim\text{Laplace},\quad t^{ev}_i\in\Big[\tfrac{k_i}{f}-\tfrac{1}{2f},\ \tfrac{k_i}{f}+\tfrac{1}{2f}\Big]$$
비디오 관측은 구간 검열(interval-censored)로 다룬다. 블러 측정(b)을 쓰면 이 구간이 줄어든다. 거리 $d_i$는 4D 재구성에서 얻는다. $(\beta,\delta_{clip},\{t^{ev}_i\})$는 강건 최대우도로 추정한다.
- $\beta\approx1/c=2.92\,\text{ms/m}$이고 신뢰구간이 0을 배제하면 **physical sync(동시녹음)**로 분류한다. 이 클립의 사건은 서브프레임 타이밍 라벨로 쓴다.
- $\beta\approx0$이면 **cinematic sync(그림 싱크)**로 분류한다. 타이밍 라벨로는 쓰지 않고, 오디오 생성의 '영화 싱크 관례'를 학습하는 데이터로 쓴다. 출력 단계에서 physical과 cinematic 싱크를 고를 수 있는 스타일 스위치가 된다.
- **식별 조건.** $\Delta d/c\gg\sigma_t/\sqrt N$이어야 한다. 예를 들어 $\Delta d=10$m(29ms), $\sigma_t=10$ms, $N=10$이면 3ms 정밀도로 분리된다. 실내처럼 $\Delta d<3$m인 클립은 '식별 불가'로 표시한다.
- **귀속.** onset과 사건의 Hungarian 매칭 비용에 재질쌍 모달 템플릿과의 스펙트럼 상관을 넣는다.

**업그레이드 (b): Blur-as-Measurement.** 원안 §2.7의 순방향 셔터 모델을 측정기로 거꾸로 쓴다.
$$\min_{s(\cdot),S}\ \sum_k\rho\Big(I_k-\tfrac1S\!\int_{t_k-S/2}^{t_k+S/2}\!\mathcal{R}\big(W(s(t')),\Gamma_k\big)dt'\Big)+\lambda R_{phys}(s)$$
노출 시간 $S$는 사건이 없는 프레임에서 (블러 길이 / 이미지 속도)로 추정한다. 충돌이 일어난 프레임에서 블러 streak가 V자로 꺾이는 지점이 접촉 시각을 서브프레임 정밀도로 알려 준다.

**업그레이드 (c): Gold Set과 라벨 등급.**
- 1000fps 다시점 카메라에 거리가 다양한 마이크를 붙여 자체 촬영한 데이터 50시간으로 검출기의 precision/recall과 타이밍 오차 곡선을 보정한다.
- 라벨을 세 등급으로 나눈다: **존재 라벨**, **순서 라벨**, **타이밍 라벨**(사후 $\sigma<5$ms인 사건만).
- 원안의 "$10^{10}$ 사건"은 존재 라벨의 수다. 타이밍 라벨은 그 1~5%로 정직하게 다시 잡는다.

### F7. 스스로 채점하는 시험지, 그리고 잘못된 첫 전장

**실패 시나리오.** 18개월 차에 BTT, RST, Revisit에서 3σ 이상 앞섰다고 보고한다. 그런데 일반 Arena 승률은 45%다. 투자자는 "구조적으로 질 수 없는 시험만 이겼다"고 평가한다. 첫 데모인 2a(살아있는 정지세계)는 호수 반사, 불꽃, 연기 렌더가 약해서 오히려 비디오 모델 쪽이 더 예쁘다.

**기술적 원인.**
1. **BTT, RST, Revisit은 명시적 세계가 정의상 이기는 시험이다.** 필요하지만 시장의 판결은 아니다.
2. **H1 토이 실험이 순환논증이다.** 식생과 물 도메인에서 비디오 DiT와 World 모델을 비교하는데, 손으로 만든 스펙트럼과 바람 prior에 들어간 '지식'은 컴퓨트로 잡히지 않는다. 지표까지 우리 홈그라운드(BTT)에서 고르면 비교가 성립하지 않는다.
3. **Arena에서 졌을 때 원인을 분해할 수 없다.** 모션 때문인지, 외형 때문인지, 연출 때문인지 알 수 없다.
4. **2a 전장의 선택이 나쁘다.** 물, 불, 연기는 (i) 비디오 모델이 이미 잘하는 ambient motion이고(Generative Image Dynamics가 2D만으로 충분함을 보였다), (ii) 3DGS가 가장 약한 view-dependent, 반투명, 굴절 렌더 영역이다(지형 G5). 구조적 우위는 가장 작고 렌더 위험은 가장 큰 전장이다.
5. **표적이 움직인다.** Seedance 2.5는 단일 패스로 30초를 만들고, Kling 3.0에는 camera path motion control이 있다. 12~18개월 뒤에는 GEN3C류 3D 캐시를 단 상용 모델이 Revisit 격차를 좁힐 수 있다.

**업그레이드 (a): Factorized Turing Battery.** 실패한 위치를 분해하는 3-arm 평가다.
- *Motion Turing*: 외형을 고정한다(같은 재구성 에셋, 렌더러, 셔터). 궤적만 {실제, 우리 것, lift한 비디오 모델 것} 셋을 비교한다.
- *Appearance Turing*: 궤적을 실제 것으로 고정한다. 렌더만 {hold-out 카메라의 실제 footage, 우리 렌더}로 비교한다.
- *Joint Arena*: 최종 결과물끼리 비교한다.

**업그레이드 (b): BUPS(Block Universe Prompt Suite) Arena.**
- 프롬프트는 시장이 실제로 요구하는 '시간 문법' 샷으로 구성한다: 동작 중 bullet time, speed ramp(1x → 0.05x → 1x), 같은 사건을 다른 각도에서 재촬영, 1000fps 슬로모, dolly zoom.
- 경쟁자에게는 best-of-4와 자사 도구 전부를 허용한다(Kling 3.0 camera path, Seedance 2.5의 레퍼런스 30개).
- BTT 같은 구조 시험은 *제품 기능*으로 다시 분류한다. 판결은 블라인드 선호로만 내린다.
- 일반 프롬프트 Arena는 별도 트랙에서 계속 추적한다.

**업그레이드 (c): 첫 전장을 2a = Frozen Action & Speed Ramp로 바꾼다.** 사건 한가운데서 시간이 멈춘 세계를 궤도 촬영한 뒤 speed ramp로 실시간으로 복귀하는 샷이다. 예: 깨지는 유리잔의 파편, 물 왕관, 공중 도약한 댄서, 흩날리는 머리카락.
- (i) 정지 슬라이스 하나하나가 사진이다. 그래서 이미지 교사가 완벽한 판사가 되고, Phase 1의 외형 품질이 그대로 넘어온다.
- (ii) 세계 시간으로 0.3~1.5초짜리 짧은 시뮬 창만 있으면 된다. F3과 F5의 위험이 최소다.
- (iii) 비디오 모델은 동작 중 bullet time을 구조적으로 만들지 못한다. BulletTime(CVPR'26)도 아직 합성 2k 장면 수준의 연구 단계이고, 상용 공백 G2에 해당한다.
- (iv) 광고, 뮤직비디오, 스포츠 하이라이트에서 실수요가 있는 샷이다. 현실에서는 카메라 어레이가 필요한 고비용 샷이다.

**업그레이드 (d): H1을 H1′로 재정의한다.** κ는 *인간이 판정한 외형·모션 사실감이 같은 등품질 지점*에서 잰다. 컴퓨트에는 학습과 추론(시뮬, 렌더, 신경 셰이더, SMC 롤아웃)을 모두 포함한다.

---

## 3. 신규성 점검

### 3.1 원안의 [신규] 항목 재판정

| 원안 항목 | 겹치는 선행 연구 | 판정 | 진짜 다른 점 |
|---|---|---|---|
| EPD + ESA | Add-and-Thin(NeurIPS'23, TPP 디퓨전), Neural STPP(Chen et al. ICLR'21), Transformer Hawkes(ICML'20), CLoSD, PhysDiff(ICCV'23), WonderPlay(2025), PhysGen/PhysGen3D, Witkin & Kass'88 | 부분 신규. 핵심으로 유지 | 사건 집합 디퓨전과 계획-시뮬 폐루프는 각각 있다. 새로운 것은 물리·연기·서사 사건을 **하나의 타입 계약 언어**로 묶어 여러 실현기에 나눠 주고, 라벨이 실영상 lift와 동시녹음 오디오에서 오며, 최종 진실을 *실현된 사건*으로 삼는다는 점이다 |
| 증인 교차신문 | Diffusion Classifier(ICCV'23), SDS-4D(4D-fy, AYG), PhysDreamer, FisherRF·ActiveNeRF(정보 이득 시점 선택), QBC(Seung'92) | 신규(조합). 역할은 부트스트랩으로 축소 | 4D 세계가 가설 사이의 판별력이 최대인 시점을 골라, 순열 대조 ELBO로 **$f_c$ 이하 대역에서만** 증언을 받는 판정 구조 |
| 오디오 onset과 음속 보정 | Visually Indicated Sounds(CVPR'16), SyncNet 계열 AV sync, 스포츠 판독의 오디오 접촉 검출(UltraEdge류), Hearing Anything Anywhere(CVPR'24) | 신규 유지. v2에서 강화 | 4D 거리-지연 회귀의 기울기 $\beta$로 동시녹음과 그림 싱크를 클립 단위로 판별하고, 그 결과로 생성 모델용 서브프레임 사건 라벨을 대규모로 만든다 |
| 적대적 시공간 카메라 감사 | ViewFool(NeurIPS'22), EG3D식 무작위 포즈 GAN 학습 | 부분 신규 | 적대적 시점은 원래 인식 모델의 견고성 평가용이었다. (자세×시각) min-max를 **4D 생성기의 학습 신호**로 쓰는 것이 다르다 |
| Factorized Recombination | **BEDLAM(CVPR'23)**, AGORA, HUMANISE/TRUMANS, Kubric, Infinigen, Putting People in Their Place(CVPR'23) | **부품으로 재분류** | BEDLAM이 실제 모캡 × 합성 장면 × 카메라 재조합을 이미 했다. 차이는 lift한 실세계와 물리 리타게팅 필터 정도로 작다 |
| 공유 바람장, GCM | Stam'97(stochastic wind), Shinya & Fournier'92, Habel et al.'09, 게임 엔진의 전역 바람장 | **부품** | 그래픽스 표준이다. GCM 지표만 작은 기여다 |
| TSR/CSI/MI/BTT/RST/Revisit | GeCo(변형, 허위 운동), MEt3R, SCoPE·ViewBench LCE·R2M(Revisit), FVMD(2024), Physics-IQ, VideoPhy-2 | MI와 Revisit은 부품, TSR·CSI·BTT-A는 기여 | 과한 부드러움을 **스펙트럼 결손과 crossover $f_c$**로 수치화한 것이 기여다. 또 VBench의 motion smoothness 지표(프레임 보간 prior 기반)가 이 *질병에 상을 준다*는 점을 보고서에서 지적한다 |
| 분기 블록 우주 | Twigg & James'07 Many-Worlds Browsing, 게임 세이브 | **부품** | 생성형 4D에서 쓰는 테이크 UX 정도의 차이다 |
| (미표기) Coverage Compiler | He, Cohen, Salesin'96 *The Virtual Cinematographer* | **부품** | 생성한 실사 4D에 적용한다는 점만 다르다 |
| (미표기) Spectral Gaussians | Generative Image Dynamics(CVPR'24), 3D Cinemagraphy(CVPR'23), LivingWorld(2026) | **부품** | — |
| (미표기) 외형과 동역학의 분리 학습 | 지형 P6/G8 | 개념은 선행 | v2의 기여는 실측한 $f_c$, 기울기 라우팅, 물리와 실사건으로 채운 고대역이다 |

### 3.2 v2에서 새로 들어간 항목

| v2 항목 | 겹치는 선행 연구 | 진짜 다른 점 |
|---|---|---|
| Edit-Closed World | DMD2, FlashWorld, ProlificDreamer(분포 매칭 3D), DiffusionRenderer·RGB↔X(intrinsics), GALA3D(구성적), pix2gestalt(amodal) | **Phase 2 사건 prior가 만들 수 있는 모든 반사실 편집에서도 이미지 교사의 분포 위에 있도록** 하나의 세계를 학습한다. 드러날 수 있는 숨은 면과 조명 분해를 하나의 목적함수로 동시에 얻는다 |
| MAN 셰이딩 | Warped Noise(integral noise), Difix3D+, 지형 P4(세계에 고정한 노이즈, 정적 장면) | 노이즈를 **변형되고 시뮬되는 물질의 canonical 좌표**에 분산 정규화해 붙인 1-step 생성 셰이더. 시간, 시점, 변형을 가로질러 같은 물질점이면 같은 샘플이 나온다 |
| RGF와 Reality Residual Budget | RFC(Yuan & Kitani NeurIPS'20), SimPoE, PhysDiff | ghost force를 0과 비교하지 않고 **실제 lift 영상의 분포와 두 표본 검정**한다. 그 분포를 보이지 않는 제어 예산으로 다시 쓴다 |
| Spectral Division과 PC-ELBO | 지형 P6, Diffusion Classifier, Seeing the Arrow of Time(Pickup et al. CVPR'14) | 범주별로 *실측한* crossover 주파수와, 블록 순열 대조로 아티팩트를 분리하는 시간 구조 점수 |
| Milestone SMC | Twigg & James'07, Chenney & Forsyth'00, Popović'00/'03, SAMCON'10 | 생성 사건 prior가 목표를 주고, 생성 세계 prior가 섭동의 그럴듯함을 판정하고, 잠긴 프레임 제약과 실분포 잔여력 예산이 붙는다 |
| AV-Physics Timing Solver | AV sync, 음원 위치 추정 | 위 3.1과 같다. 추가로 sync style을 생성 출력의 스타일 변수로 쓴다 |
| Frozen Action과 Hero-Instant | BulletTime(CVPR'26), BulletGen, CAT4D | 정지 순간을 *물리 롤아웃 집합에서 이미지 교사로 선택*하고, 같은 블록에서 speed ramp로 실시간까지 이어 간다 |
| Factorized Turing Battery | O'Sullivan et al.'03(물리 애니메이션의 시각적 충실도 지각 평가) | 생성 모델과 SOTA 비디오 모델을 3-arm으로 분해해 비교한다 |

### 3.3 바퀴를 재발명하지 말 것: 부품표

- **재구성과 lift**: DA3, VGGT-Ω, π3, MapAnything, MonST3R, CoTracker3, TAPIP3D, GVHMR/WHAM/TRAM, MANO, FoundationPose, Shape-of-Motion/MoSca, MoVieS, SAM2.
- **세계와 4D 생성**: TRELLIS SLAT, FlashWorld의 dual-mode DMD2 설계, DiffSplat, 4DGS, CAT4D, Diff4Splat, 4DNeX(free 4D field 층).
- **물리**: Newton, MuJoCo-Warp/MJX, PhysGaussian(MPM), XPBD, FleX의 unified particle 개념, GNS/MeshGraphNets, PhysTwin, NGFF, Tessendorf FFT 바다.
- **캐릭터**: SMPL-X, FLAME, GaussianAvatars, Animatable Gaussians, Relightable Gaussian Codec Avatars, PHC/MaskedMimic, PhysDiff, CLoSD, VASA-1(얼굴 동역학 latent의 발상).
- **렌더**: 3DGRT/3DGUT, Difix3D+, DiffusionRenderer, RGB↔X, Warped Noise.
- **평가**: GeCo, MEt3R, WorldScore, VBench-2.0, FVMD, Physics-IQ, VideoPhy-2, SCoPE/ViewBench/R2M, CameraBench.
- **데이터**: DL3DV-10K, SpatialVID, Stereo4D, OmniWorld, Sekai, AIST++, AMASS, BEDLAM 파이프라인, DataDoP.
- **연출과 음향**: Virtual Cinematographer idiom, Zheng & James 모달·파괴 음향, Harmonic Fluids, SoundSpaces, Steam Audio.

---

## 4. 90일 결정적 실험 (4~6인 팀, GPU 64~128장)

### X1. Spectrum Report v0와 crossover $f_c$ 실측 (1~5주)
- **가설.** 상용 최상위 모델의 ego-motion 보상 가속도 PSD는 실제 영상보다 고주파가 부족하다. 결손이 시작되는 $f_c$는 범주별로 1.5~8Hz에 있다. 이 결손은 사람의 "둥둥 뜬다 / 미끈하다" 판정과 상관이 있다.
- **셋업.**
  - 범주 6개(낙하·바운스, 충돌·파괴, 스포츠 타격, 보행·달리기, 댄스, 손 조작) × 프롬프트 40개 × 모델 4개(Seedance 2.5, Wan 3.0, Kling 3.0 Omni, Veo 3.1) × seed 2개로 약 1,920클립을 만든다.
  - 실제 레퍼런스는 범주마다 200클립이다. 자체 240fps 촬영분과 Stereo4D/SpatialVID에서 고른 60fps 이상 클립을 쓴다.
  - 파이프라인: DA3 pose → ego-motion 보상 → SAM2 마스크 위 CoTracker3 트랙 → 가속도 → Welch PSD → $R(f)$, $f_c$ (bootstrap CI).
  - 보조 지표로 CSI와 g-check를 함께 본다.
  - 평가자 30명이 2AFC와 "too smooth" Likert에 답한다. VBench motion smoothness 점수도 같이 계산한다.
- **성공 기준.**
  - 4개 모델 중 3개 이상에서 $f_c$ 위 대역의 $R<-3$dB(95% CI).
  - Spearman ρ(HFER 결손, 사람의 too-smooth 판정) ≥ 0.4.
  - VBench smoothness와 결손 사이의 양의 상관(VBench가 질병에 상을 준다는 증거).
- **실패하면.**
  - 결손이 없으면: 질병 (3)은 스펙트럼 문제가 아니다. CSI와 사건 인과 오류로 축을 바꾸고, 피치는 (1)(2)와 제어 가능성 중심으로 다시 쓴다.
  - 결손은 있는데 사람이 반응하지 않으면: TSR을 최적화 대상에서 빼고 진단 지표로만 쓴다.
- **비용.** API 약 $2~5k, 1인 5주, 약 2k GPU-h.

### X2. Factorized Turing Battery v0: 강체·연체 (2~10주)
- **가설.**
  - (Motion arm) 외형을 고정하면 우리 물리 모션(S)과 실제 모션(R)을 구별하는 식별률이 60% 이하이고, S가 lift한 비디오 모델 모션(V)보다 70% 이상 선호된다.
  - (Appearance arm) intrinsics + MAN 셰이더가 결정론적 셰이더보다 15%p 이상 선호되고, warping error는 늘지 않는다.
- **셋업.**
  - 240fps 폰 3대로 rig를 만든다. 동기화는 박수와 오디오 상호상관으로 한다.
  - 물체 10종 × 사건 6종(낙하, 바운스, 굴림, 미끄러짐, 넘어짐, 충돌)으로 약 60클립을 찍는다.
  - 배경과 물체를 각각 3DGS로 재구성하고, 다시점 BA와 FoundationPose로 실제 6DoF 궤적을 얻는다.
  - S는 초기조건만 실제에서 가져오고, $\psi$는 VLM prior에 X5 보정을 적용한다. V는 첫 프레임에서 Wan 2.2 I2V로 생성한 뒤 known-shape로 lift한다.
  - 같은 렌더러와 셔터로 원래 시점과 45° 새 시점에서 렌더한다.
  - Appearance arm은 세 번째(hold-out) 카메라의 실제 footage를 비교 대상으로 쓴다.
  - 평가자 40명. 지표는 TSR, CSI, g-check, ghost force.
- **성공 기준.** 위 가설 수치에 더해 S의 CSI KL이 V의 1/2 이하.
- **실패하면.**
  - S와 R의 식별률이 65%를 넘으면 sim look이 확인된 것이다. 원인을 반발·마찰, 미시 진동, 블러로 분해하고, PCGD 잔차 학습을 1순위로 올린다. 캡처를 500클립으로 늘린다.
  - S 대 V 선호가 55% 미만이면 강체 영역에서는 비디오 모델이 이미 충분한 것이다. 쐐기를 Frozen Action과 Reshoot 같은 구조 축으로 옮긴다.
  - MAN이 효과가 없으면 셰이더 용량과 수용영역을 키우고, 이미지 교사 DMD의 비중을 올린다.
- **비용.** 2인 8주, 장비 $5k 미만, 3k GPU-h 미만.

### X3. Edit-Closed World: 장면별 반사실 분포 매칭 (1~12주)
- **가설.** 이미지 교사 DM에 반사실 편집(제거, 이동, 재조명)을 30~50% 섞는다. 그러면 (i) 숨은 지지면이 순차 인페인팅 기준선 이상으로 사실적이면서 다시점 일관성이 높고, (ii) albedo-조명 누설이 절반 이하로 줄어든다.
- **셋업.**
  - 장면 40개를 쓴다. 20개는 Blender/Infinigen 합성으로 GT albedo와 숨은 면이 있고, 20개는 생성 또는 재구성한 장면이다.
  - 표현: slot이 붙은 Gaussian, intrinsics, 잔차 radiance.
  - 교사: 상업적 사용이 가능한 오픈 이미지 모델. 라이선스를 확인해야 한다. FLUX.1-dev는 비상업 조건으로 알고 있고, Qwen-Image는 Apache-2.0으로 알고 있다(확인 필요). per-scene LoRA fake score를 쓴 VSD/DMD 방식이다.
  - 조건: (A) 가시 뷰만 DM, (B) A에 반사실 편집 추가, (C) **기준선**: 객체를 제거한 뒤 다시점 순차 인페인팅을 하고 재구성.
  - 측정: RRS(2AFC), Reveal-MEt3R, GSI, 합성 장면의 albedo leak, 원래 뷰의 품질 저하(Q-Align).
- **성공 기준.**
  - B가 C보다 사실감 선호 50% 이상이고, Reveal-MEt3R가 C의 0.5배 이하.
  - albedo leak이 A 대비 50% 이상 감소.
  - 원래 뷰 품질 저하 2% 이하.
  - 장면당 2 GPU-h 이하.
- **실패하면.**
  - 교사가 숨은 면을 밀어내지 못하면 **compositional route**로 간다. 객체는 TRELLIS/Hunyuan3D급 객체 생성기로 *객체 공간에서 뒷면까지 완전하게* 만들고 합성한다. 객체 단위로 완결된 생성이고 캔버스 위 순차 채우기가 아니므로 hole-filling이 아니다.
  - 조명 누설이 줄지 않으면 재조명 편집 비율을 높이고 DiffusionRenderer식 intrinsic 교사를 추가한다.

### X4. Chaos-as-Controllability (3~10주)
- **가설.** 도미노 50개, 핀볼식 공-장애물, 5단 미니 Rube Goldberg에서 두 가지 섭동만으로 사건 제약(±1프레임@24fps, ±2cm) 충족률 90% 이상, 시간 10 GPU-분 이하를 달성한다. 섭동은 물성 prior 범위 안의 섭동과, 출력 카메라에서 0.5px 이하로만 보이는 초기조건 섭동이다. 비교 대상인 diff-sim 수리(Warp)와 제어만 쓰는 CEM은 50% 이하에 그친다.
- **셋업.** 과제 20개 × 제약 세트 5개. MuJoCo-Warp 또는 Newton, $N=4096$. Milestone SMC는 rejuvenation (a)와 (b)를 따로 돌려 비교한다(ablation). 잠긴 프레임 제약도 시험한다.
- **성공 기준.** 위 가설 수치에 더해, 평가자가 '조종된 롤아웃'과 '조종되지 않은 롤아웃'을 구별하는 식별률이 55% 이하.
- **실패하면.** 감독 허용오차 창을 학습으로 넓히고, 보이지 않는 보조력을 X5의 예산 안에서 허용한다. 그래도 안 되면 EPD에 feasibility critic guidance를 걸어 *실현 가능한 계획*만 생성하게 한다.

### X5. 실분포 보정 판사와 Sync-Sound 판별 (1~8주)
- **가설.**
  - (a) lift한 실제 영상의 ghost force는 0이 아니고(중앙값이 기존 임계의 30% 초과), 생성물과 분리된다(AUC ≥ 0.8).
  - (b) 거리-지연 회귀로 동시녹음과 foley를 90% 이상 정확하게 구분한다. 동시녹음 클립의 사건 시각 오차는 1000fps GT 대비 중앙값 5ms 이하다.
- **셋업.**
  - (a) 인체 실제 클립 400개(Stereo4D, SpatialVID, AIST++)와 같은 동작을 프롬프트로 생성한 클립 400개를 비교한다. GVHMR로 lift하고 PHC로 추종한 뒤 RFC 정의로 ghost force를 계산한다. 강체는 X2 데이터를 재사용한다.
  - (b) 자체 촬영 3시간: 대여한 1000fps 카메라, 폰 여러 대, 5~60m 거리의 사건(킥, 박수, 문, 타격), LiDAR 거리. 여기에 수동 라벨(sync, foley, music)을 붙인 웹 클립 300개를 더한다.
- **실패하면.**
  - (a) AUC가 0.65 미만이면 lift 노이즈가 지배적인 것이다. 물리 판사를 보상으로 쓰는 것을 보류하고, 다시점 lift(스포츠, 자체 rig)의 정밀도부터 올린다. PLAC도 보류한다.
  - (b)가 실패하면 타이밍 라벨을 자체·라이선스 동시녹음으로 한정하고, 웹 오디오는 cinematic sync 학습에만 쓴다.

**90일 종합 게이트.** X2의 Motion arm과 X3이 모두 통과하면 2a(Frozen Action) 제품화로 간다. 하나만 통과하면, 실패한 arm의 업그레이드를 다음 분기 1순위로 둔다. 어느 경우든 목표는 낮추지 않는다.

---

# 개정 제안서 v2: Block Universe Cinema

### 시공간 블록을 생성하고, 세 교사가 각자 자기 대역만 판정한다

> 렌즈는 Phase 2(움직이는 세계) 설계와 데이터·스케일링 전략이다. 기준일은 2026-10-10이다. **[v2]**는 원안에서 바뀐 부분이다. **[신규]**는 공개된 선행을 찾지 못한 항목이고, **[부품]**은 빌려 쓰는 항목이다.

## 0. TL;DR

1. **정식화(유지).** 비디오는 4D 블록을 카메라 세계선 $\gamma(\tau)$를 따라 셔터 적분한 결과다. **[v2]** identity는 시간 불변인 *인자* $\theta$에 두고, 외형은 상태의 함수로 둔다.
2. **원칙(개정).** 확률은 원인(세계, 사건)과 미시 잔차에 두고, 보장은 불변량에 건다. 카메라는 읽기 전용이다.
3. **세 교사와 한 증인.** 이미지 교사는 임의 시공간 슬라이스의 공간 사실감을 판정한다. 실분포로 보정한 물리 검증기는 불변량과 고대역 동역학을 판정한다. 실사건 prior는 타이밍 통계를 맡는다. 비디오 모델은 $f_c$ 이하만 다루는 증인이다. 기울기 라우팅으로 서로의 영역을 넘지 못하게 한다.
4. **Edit-Closed World [신규].** Phase 1 세계는 Phase 2 사건 prior가 만들 수 있는 모든 편집에서도 사실적이도록 학습된다.
5. **EPD v2와 Milestone SMC [신규].** 사건 집합이 계약 언어 역할을 한다. 혼돈은 제어 레버로 쓰고, 실분포 잔여력 예산은 보이지 않는 제어 채널이 된다.
6. **외형 [신규].** 상태 조건부 외형과 Matter-Anchored Noise 셰이딩을 쓰고, 슬라이스 분포 매칭으로 학습한다.
7. **데이터 [신규].** Blur-as-Measurement, AV-Physics Timing Solver, Gold Set을 쓴다. 라벨은 존재, 순서, 타이밍 세 등급으로 나눈다.
8. **평가와 로드맵.** Factorized Turing Battery와 BUPS Arena를 쓴다. 첫 전장은 **Frozen Action & Speed Ramp**다.
9. **스케일링.** 검증 가능한 보상(재투영, 실분포 물리, cycle)과 test-time SMC를 쓴다. go/no-go는 H1′, 즉 등사실감 지점에서의 전체 컴퓨트로 판단한다.

## 1. 정식화

### 1.1 렌더 등식
$$V(\tau,\mathbf{u})=\frac{1}{S(\tau)}\int_{t(\tau)-S/2}^{t(\tau)+S/2}\mathcal{R}_{MAN}\Big(W\big(t'+\rho(\mathbf{u})\big),\ \Gamma(\tau),\ \eta\Big)(\mathbf{u})\,dt'$$
- $\rho(\mathbf{u})=(v/H-\tfrac12)T_{readout}$는 rolling shutter다. $\Gamma=(T_{wc},K,d_f,N,\text{distortion},\text{breathing})$다.
- $t(\tau)$는 카메라 시간을 세계 시간으로 바꾸는 사상이다. 1이면 정상 속도, 0과 1 사이면 진짜 슬로모션, 0이면 bullet time, 음수면 역재생, 불연속이면 컷이다.
- $\eta$는 물질에 고정한 노이즈장(§5.2)이다. 세계선은 $\gamma(\tau)=(t(\tau),\Gamma(\tau),S(\tau))$다.

### 1.2 생성 체인 [v2]
$$W_{canon}\sim p_\omega(W\mid c)\ \text{(edit-closed)},\quad (E,z)\sim p_\theta(E,z\mid W_{canon},c)$$
$$s=\mathrm{Realize}\big(E,z,W_{canon};\ \Delta s_0,\Delta\psi,\Delta\text{layout},a,\iota\big),\quad W(t)=\mathrm{Deform}(W_{canon},s(t)),\quad A_i(t)=\mathcal{A}_\Theta(\theta_i,\varphi_i(s(t)),L(t),\omega)$$
```
c (스크립트/프롬프트)
 └→ W_canon          [확률]        edit-closed 세계: slot·intrinsics·θ·ψ-prior(평균, 분산)·어포던스
     └→ (E, z)       [확률]        사건 집합(타입 계약) + 퍼포먼스 latent
         └→ Realize  [확률·제약]   Milestone SMC over (Δs0, Δψ, Δlayout, a, ι)
             │                     PCGD: Π_I(Sim_base + F_φ(·, ξ))   ← 불변량 I1–I5 하드
             └→ s(t)
                 └→ W(t), A(t)  [결정]   Deform + 상태 조건부 외형
                     └→ V       [결정]   γ로 셔터 적분, MAN 셰이딩 (읽기 전용, I6)
```

### 1.3 불변량 [v2]

| ID | 불변량 | 근거 / 효과 |
|---|---|---|
| I1 | identity $\theta_i$는 시간 불변 | 일관성 붕괴(1)가 정의상 불가능하다 |
| I2 | Gaussian 수와 객체 질량은 위상 사건(FRACTURE, SPLASH, IGNITE, MERGE)에서만 변한다 | 녹아내림(2)은 사건 없이는 일어날 수 없다 |
| I3 | 비관통, $\text{SDF}\ge-1$mm | 관통 아티팩트가 없다 |
| I4 | metric 중력 $g=9.81$ | 둥둥 뜨는 낙하(3)가 없다 |
| I5 | 사건 마크의 $\mathbf{J}$와 시뮬의 $\Delta p$가 ±5% 안에서 일치 | 같은 $\mathbf{J}$가 소리를 만들어 싱크가 정의상 맞는다 |
| I6 | **Read-Only Camera**: 렌더 경로에서 $W$로 쓰는 연산이 0개 | hole-filling 회귀를 구조적으로 막는다 |

### 1.4 세 교사와 기울기 라우팅 [신규]

| 신호 | 판정 대상 | 기울기가 흐르는 곳 | 차단되는 곳 |
|---|---|---|---|
| 이미지 교사 DM: 슬라이스 $(c,t,e,L')$ | 임의 정지 순간의 공간 사실감 | $\theta,\Theta,\varphi_{shader},\delta,o_i$, intrinsics | $s(t),E,\psi$ (선택만 허용) |
| 물리 검증기 RGF (실분포 보정) | 불변량, 고대역 동역학 | $F_\phi$, 정책 $\pi$, $\psi$ | 외형 |
| 실사건 prior (타이밍 라벨) | 사건 간격과 타이밍 통계 | EPD | 외형 |
| 비디오 증인 (오픈 가중치, $\le f_c$) | 저주파 의미와 연출 | EPD의 거친 타이밍, 의도, $z$ | 고대역 상태, 외형 |
| Reader cycle | 세계의 가독성(숨은 모순이 없는지) | 전부 (낮은 가중) | — |
| 적대적 카메라 감사 | 결함 | 생성기 전부 | — |

비디오 증인의 저역 제한은 렌더를 $f_c$로 시간 prefilter한 다음 교사의 네이티브 latent 속도로 넣는 방식으로 구현한다. 그러면 교사가 고대역을 밀어낼 수 없다.

### 1.5 4D 세계 텐서 구조 [v2]

| 구성요소 | 텐서 | 시간 의존 |
|---|---|---|
| 정적 Gaussian | $\mu(3),q(4),\log s(3),\alpha(1)$ + intrinsics $a(3),n(3),r(1),m(1)$ + 잔차 $f(8)$ + identity $\theta(32)$ + slot $o_i$ | 없음 |
| canonical 좌표 / 해시 셀 id | $x_c(3)$, 다해상도 셀 인덱스 (MAN 노이즈용) | 없음 |
| 객체 에셋 $k$ | $G_k$ (위와 같은 스키마), 리그 $B_k$, skin weight top-4 | 없음 |
| 물성 | $\psi_k$의 평균과 분산(약 12×2) | 위상 사건에서만 |
| 충돌 프록시, 어포던스 | SDF / convex, 어포던스 맵 | 없음 |
| 상태 | $s_k(t)$: 강체 13, 인체 약 330+얼굴 100, 입자 $P\times16$ | **있음** |
| 상태 특징 | $\varphi_i(s)$: log-strain 6, ΔAO, 접촉압, 젖음, 온도, 속도 | 상태의 함수 |
| 미시 잔차 | $\xi_t$ (PCGD 잡음), $\iota$ (잔여력) | **있음** |
| 사건 집합 | $E$: $M\le1024$, 슬롯당 약 64 | 시간 좌표 자체 |
| 광원, 바람장 | $L_j(t)$, $\mathbf{u}(\mathbf{x},t)$ | **있음** |
| 세계선 | $\gamma(\tau)$, 약 20차원 / 프레임 | 촬영 변수 |

압축비는 원안과 같다. 순진한 4DGS가 약 $1.8\times10^{10}$ float인 데 비해 우리는 약 $10^8$ float로, 대략 200배 작다. 생성 모델이 다루는 시간 의존 변수는 저차원이다.

### 1.6 블록에서 공짜로 얻는 것

- **Coverage Compiler [부품: Virtual Cinematographer'96].** action line을 계산하고 master, OTS, CU, 인서트를 자동 배치한다. eyeline은 3D gaze로 정의상 맞고, 컷 간 연속성 비용이 0이다. ProRes 멀티캠과 EDL로 내보낸다.
- **테이크.** $z$만 다시 샘플하고 blocking은 고정한다. 디렉션은 사건·의도 공간의 CFG로 준다.
- **분기 우주 [부품: Many-Worlds Browsing'07].** $s(t_b)$에서 갈라지는 포크다.
- **시간 리맵과 셔터.** 시뮬이 내부적으로 240~1000Hz로 돌아가므로 진짜 슬로모션이 나온다. speed ramp, 셔터 각도 스타일, dolly zoom $f\propto\lVert x_{cam}-x_{subj}\rVert$를 지원한다.
- **우리만의 카메라.** slit-scan, multi-perspective, 벽을 통과하는 원테이크.

### 1.7 이것이 hole-filling이 아닌 이유 [v2, 세 겹의 보장]

1. **세계는 촬영 전에 닫혀 있다(§2).** 드러날 수 있는 모든 면은 학습 단계에서 반사실 편집을 통해 같은 세계와 *동시에* 최적화되었다. 추론에서는 한 번에 출력된다.
2. **시간 방향은 물리 적분이다(§4).** 계획은 블록 전체에 대해 한 번에 세우고, 실현은 인과적으로 한다. 픽셀 캔버스는 어디에도 없다.
3. **렌더는 읽기 전용이다(I6, §5.2).** 셰이더에는 콘텐츠를 지어낼 채널(알파, 기하, 객체)이 없다. 같은 물질점, 같은 상태, 같은 뷰면 출력도 같다.

## 2. 세계: Edit-Closed World (Phase 1 계약 v2) [신규]

- **목적함수.**
$$\mathcal{L}_{EC}=\mathbb{E}_{e\sim p_{edit},\,c\sim p_{cam},\,L'\sim p_{light}}\ \mathrm{KL}\Big(p_{\mathcal{R}}(\cdot\mid e,c,L')\ \Big\Vert\ p^{img}(\cdot\mid y_e)\Big)$$
DMD2 기울기로 최적화하고, 반사실 샘플 비율은 $\rho_{cf}=0.3\sim0.5$로 둔다. $p_{edit}$는 Phase 2 EPD의 저해상 버전이고, 둘을 함께 업데이트한다. 사건 모델이 새 사건을 배우면 세계도 그 사건에 대해 닫힌다.
- **amortized 생성기.** World DiT(§7.6)가 $W_{canon}$을 한 번에 출력한다. 1단계는 lift 데이터로 flow matching을 하고, 2단계는 이미지 교사로 $\mathcal{L}_{EC}$ post-train을 한다.
- **Phase 1 계약 v2.**
  1. metric scale. DA3/VGGT-Ω의 metric head와 객체 크기 prior로 교차검증한다.
  2. slot $o_i$는 생성 변수다.
  3. intrinsics: 잔차 radiance 에너지 비율 15% 이하.
  4. edit-closure 점수: RRS, GSI 기준을 통과해야 한다.
  5. $\psi$ prior는 평균과 분산을 함께 준다(SMC가 분산을 쓴다).
  6. 충돌 프록시와 어포던스.
  7. canonical 좌표와 셀 id를 안정적으로 유지한다(MAN용).
- **Phase 1이 얻는 것.** 재조명, time-of-day, 소품 재배치가 공짜가 된다.
- **지표.** RRS, Reveal-MEt3R, GSI, albedo leak, 원래 뷰 품질 유지(Q-Align, GeCo).

## 3. 사건: EPD v2

- **표현.** 슬롯은 $e_i=(b_i,t_i,\mathbf{x}_i,k_i,p_i,q_i,\mathbf{m}_i,\ell_i)$다. 새로 추가한 $\ell_i\in\{\text{물리},\text{연기},\text{서사},\text{환경}\}$는 층을 나타낸다. 마크 $\mathbf{m}$에는 충격량 $\mathbf{J}$, 상대속도, 재질쌍, 파편 시드, 의도·감정 임베딩, 강도가 들어간다.
- **모델과 손실 [부품: Add-and-Thin식 TPP 디퓨전].**
  - 사건 토큰끼리 self-attention을 하고, 세계·에셋·텍스트에는 cross-attention을 한다. $t$는 Fourier feature로 넣는다.
  - 연속 성분은 flow matching, 이산 성분은 masked diffusion으로 학습한다.
  - 손실은 원안과 같고, 라벨 등급에 따른 마스킹을 추가한다. 존재 라벨만 있는 사건은 $t$ 손실을 0으로 하고, 순서 라벨만 있으면 순위 손실만 건다.
- **조건.** 스크립트 제약은 사건 집합 공간에서의 inpainting(슬롯 고정)으로 준다. feasibility critic $F_\phi(E,W)$로 guidance를 거는데, 그 라벨은 SMC 실현 성공 여부다.
- **데이터.** (i) 실제 lift에서 얻은 존재·순서 라벨, (ii) 동시녹음으로 보정한 타이밍 라벨, (iii) 시뮬이 실현한 사건 $\hat E$(완전 라벨), (iv) 실제 증인 검색 코퍼스.

**Event Type Contract [신규].**

| 타입 | 층 | 실현기 | 검출기 | 허용오차 | 검증기 |
|---|---|---|---|---|---|
| CONTACT_ON/OFF, IMPULSE_EXT | 물리 | PCGD / 강체 | 접촉쌍 + 상대속도 부호 | ±2ms, ±2cm | RGF, I5 |
| FRACTURE | 물리 | MPM 손상 / Voronoi(시드=마크) | 손상 임계 | ±1ms | I2, 시드 재현성 |
| SPLASH, IGNITE, EXTINGUISH | 물리 | 유체 / 연소 방출원 | 방출 시작 | ±5ms | 질량·에너지 예산 |
| FOOTSTRIKE, GRASP, RELEASE | 물리-연기 | 캐릭터 정책 + 접촉 투영 | 말단 속도와 높이 | ±1프레임 | foot skate, 관통 |
| GAZE_SHIFT | 연기 | 머리-눈 IK | 시선 벡터 변화 | ±80ms, ±3° | 각도 오차 |
| UTTERANCE_ON/OFF | 연기 | TTS → 음소 → viseme | VAD + 음소 정렬 | ±20ms | LSE-C/D |
| INTENT, EMOTION_SHIFT | 서사 | 플래너 → 정책 목표, $z$ 곡선 | 학습된 행동 인식기 | ±0.5s 창 | VLM과 인간 critic |
| LIGHT_ON/OFF, GUST | 환경 | 광원·바람장 파라미터 | 직접 | 정확 | — |

## 4. 실현: PCGD와 Milestone SMC

### 4.1 PCGD [신규 조합; 부품: PhysGaussian, FleX, GNS/MeshGraphNets, PhysTwin, NGFF]
$$s_{t+\Delta}=\Pi_{\mathcal{I}}\Big(\mathrm{Sim}_{base}(s_t,a_t;\psi)+F_\phi(s_t,a_t,\psi,\xi_t)\Big)$$
- 기질은 Gaussian과 1:1인 입자 그래프다. 강체는 shape-matching 클러스터, 천은 정점, 유체와 입상체는 입자다. 결합은 엣지로 표현하므로 양방향 결합이 기본이다.
- $F_\phi$는 sim-to-real 잔차만 학습한다. 데이터는 Stereo4D, 스포츠 멀티캠, 자체 rig다.
- 고전 솔버는 두 가지 역할을 맡는다. 하나는 *base 동역학*이고, 다른 하나는 *검증기*다. 최종 렌더에서는 고정밀 모드를 쓰고, 미리보기에서는 학습 emulator를 쓴다.

### 4.2 RGF 판사 [신규]
- 범주별 $p_{real}(F_g)$를 둔다. 개별 판정은 $p$-값 0.05 이상, 배치 판정은 MMD 기준이다.
- **Reality Residual Budget**: $\iota$는 $p_{real}$의 95분위 이하로 제한한다.
- $p_{real}$는 lift 품질이 오를 때마다 다시 추정한다. 판사가 데이터 품질과 함께 엄격해진다.

### 4.3 Milestone SMC [신규 조합; 부품: Twigg & James'07, Chenney & Forsyth'00, Popović'00/'03, SAMCON'10]
- 결정 변수는 $(\Delta s_0,\Delta\psi,\Delta\text{layout},a,\iota)$이고, 제약은 사건 허용오차 창과 잠긴 프레임 0.5px다.
- 알고리즘은 F5의 의사코드와 같다. 기본 $N=4096$, 세대당 $4\times10^7$ env-step이다.
- **Hero-Instant Selection.** 각 입자의 $t_h$ 슬라이스를 이미지 교사로 점수 매겨 선택 가중치로 쓴다. 물리는 바꾸지 않는다.
- ESA-B(재계획)는 SMC가 실패한 경우의 폴백으로만 쓴다.

### 4.4 캐릭터 스택 [v2]
- **몸.** SMPL-X와 Gaussian LBS를 기본으로 하고, 상태 조건부 외형 $\mathcal{A}_\Theta$로 주름, 근육, 피부 늘어남을 처리한다.
- **모션.** kinematic motion diffusion을 1차로 쓴다(EPD 사건과 의도로 조건화). 접촉 창에서만 PhysDiff식 투영을 하고 RGF 예산을 적용한다. PHC/MaskedMimic은 접촉이 많은 구간(격투, 조작)에만 쓴다.
- **얼굴.** FLAME 100차원으로는 미세 연기를 담기에 용량이 부족하다. 다시점 얼굴 데이터에서 학습한 고차원(256~512) expression latent를 쓴다(Codec Avatar 계열). 오디오와 텍스트에서 이 latent로 가는 동역학은 VASA-1의 발상을 3D로 옮겨, lift한 대규모 talking 영상으로 학습한다. 눈은 별도의 각막 굴절과 catchlight 모델로 만든다.
- **Avatar Foundry v2.** identity 레퍼런스를 조건으로 한 이미지 교사 분포 매칭으로 학습한다. 뷰 코드 평균은 하지 않는다.

### 4.5 물질별 모듈

| 대상 | 기질 | base 솔버(부품) | 생성 대상 |
|---|---|---|---|
| 강체, 관절체 | 클러스터 입자 | Newton, MuJoCo-Warp | 초기조건, 외력 사건 |
| 연체, 입상체, 파괴 | MPM 입자 | GPU MPM(PhysGaussian) | 재료 파라미터, 충격 사건 |
| 천, 머리카락 | 정점 / Cosserat 입자 | XPBD | 패턴, 헤어 에셋 |
| 액체 | FLIP 입자와 표면, 원경은 Tessendorf | FLIP / FFT | 방출원, 경계 |
| 연기, 불 | sparse voxel과 world-space 난류 디테일 | Eulerian | 방출원, 온도 |
| 식생 | Spectral Gaussians와 공유 바람장 [부품] | modal | 바람 스펙트럼 |
| 비물리(VFX, 변신) | Free 4D field [부품: 4DGS, CAT4D, Diff4Splat] | 없음 | 4D 직접 생성. 물리 보장은 포기하지만 I1과 I6은 유지한다 |

## 5. 외형

### 5.1 상태 조건부 외형 [신규 조합]
$A_i(t)=\mathcal{A}_\Theta(\theta_i,\varphi_i(s(t)),L(t),\omega)$이다. identity는 $\theta$에 두고, 변화는 $\varphi$가 담당한다.

### 5.2 Matter-Anchored Noise 셰이딩 [신규]
$$\xi(\mathbf{u})=\lvert\Omega_\mathbf{u}\rvert^{-1/2}\sum_{q\in\Omega_\mathbf{u}}\eta(q),\qquad r(\mathbf{u})=g_\varphi\big(\text{Gbuf}_{N(\mathbf{u})},\xi(\mathbf{u}),\omega\big),\quad \lVert r\rVert\le\tau$$
- 1-step generator이고 수용영역은 16×16이다. 출력은 셰이딩 잔차(밝기·색 변조)뿐이다. 알파와 기하 출력은 없고, 실루엣 밖은 마스크로 막는다.
- 학습: 이미지 교사 DMD, GAN(R1), 낮은 가중의 paired perceptual loss.
- 보장: 같은 물질점, 같은 상태, 같은 뷰면 출력이 같다. 그래서 swimming과 평균화가 동시에 사라진다.

### 5.3 슬라이스 분포 매칭 [신규]
4D 블록에서 $(c,t,e,L')$를 샘플하고 물리 셔터로 렌더한 뒤 이미지 교사 DMD를 적용한다. 라우팅은 §1.4를 따른다.

### 5.4 조명 (세 겹, v2)
1. 정적 부분은 intrinsics로 shading하고 잔차 radiance를 더한다(edit-closed라 재조명이 가능하다).
2. 움직이는 객체와 그 영향 영역은 3DGRT/3DGUT로 그림자와 반사를 계산한다 [부품].
3. 잔차는 MAN 셰이더가 처리한다. intrinsic 교사로는 RGB↔X와 DiffusionRenderer를 쓴다 [부품].

### 5.5 물리 카메라 [부품]
- 셔터 적분 블러는 $\Sigma_{blur}=\Sigma+\tfrac{S^2}{12}\mathbf{v}\mathbf{v}^\top$로 계산하고, 회전 운동은 서브프레임 샘플 8~16개로 처리한다.
- thin-lens DoF, focus pull, rolling shutter, 톤 커브, 센서 grain을 지원한다.
- 충돌 전후 프레임에서 블러 방향이 정확히 뒤집힌다.

## 6. 비디오 모델의 자리: 저주파 증인 [v2]

1. **$f_c$ 실측.** 범주별 $R(f)$에서 구하고, Spectrum Report와 함께 분기마다 갱신한다.
2. **진술 그래프.** 제안 영상을 (사건 타입, 참여자, 부분 순서, 거친 시간 창)으로 바꾸고 그래프 매칭으로 교차신문한다.
3. **PC-ELBO.** $\Delta_m(\gamma)=\mathrm{score}(\mathcal{R}(h_m,\gamma))-\mathbb{E}_\pi\mathrm{score}(\pi\circ\mathcal{R}(h_m,\gamma))$에 CRN을 쓴다. 오픈 가중치 모델(Wan 2.x, Apache-2.0)만 쓴다.
4. **능동 카메라.** $\gamma^*=\arg\max_\gamma\sum_{m<m'}w_mw_{m'}\lvert\Delta_m(\gamma)-\Delta_{m'}(\gamma)\rvert$를 고르고, 사후 가중치를 $w_m\leftarrow w_m e^{\beta\Delta_m(\gamma^*)}$로 갱신한다. 2~3회 반복한다.
5. **실제 증인.** lift 코퍼스에서 사건 그래프로 검색한 top-k의 타이밍 통계를 쓴다(ReMoDiffuse식).
6. **증거 사용 규칙.** 증인 궤적은 $\mathrm{LPF}_{f_c}$를 거친 것만 쓰고, 사건 시각은 $\pm1/(2f_c)$ 창으로만 쓴다. 고대역은 PCGD와 실사건 prior가 정한다. Lift는 원안대로 known-shape 저차원 fitting을 하되, robust loss와 identity drift 검출을 붙인다.
7. **졸업.** 인턴 호출 비율이 5% 미만이 되고, 증인에 의존한 사건의 TSR 거리가 실제-실제 노이즈 밴드 안에 들면 졸업으로 본다.
8. **법무.** 닫힌 API 출력은 *평가에만* 쓴다. 학습 사용 여부는 약관을 검토한 뒤 정한다.

## 7. 데이터 플라이휠 v2

### 7.1 소스
원안의 표를 유지한다: 드론, 부동산, 브이로그(DL3DV, Sekai), 영화(DataDoP), **스포츠 멀티캠**, 공연, 댄스(AIST++), 손-객체(Ego-Exo4D, HOI4D), 240fps 슬로모, 모캡(AMASS, BEDLAM), 다시점 스튜디오, 자체 캡처, 합성. v2에서는 동적 4D 소스로 **SpatialVID(동적 7k시간), Stereo4D(metric 4D 10만 이상), OmniWorld**를 먼저 쓴다.

### 7.2 Lift 파이프라인 v2
```
원본 클립
 → [필터] 카메라 운동·시차·동적 비율·미감·컷 검출 / VFR 타임스탬프 정규화
 → [기하] DA3 / VGGT-Ω / π3 (metric) + rolling shutter 보정
 → [정적/동적] MonST3R식 마스크 → 정적 부분은 W_canon에 fit
 → [동적 4D] 사람 GVHMR/WHAM(SMPL-X) · 손 MANO · 객체 FoundationPose / MoSca(비강체)
 → [서브프레임] Blur-as-Measurement (사건 ±3프레임 창에서 셔터 역적분, S 추정)
 → [사건] 후보 검출 → Event Type Contract 검출기
 → [오디오] onset → AV-Physics Timing Solver → {physical sync | cinematic sync | 식별 불가}
 → [물리] 시뮬 추종 → ghost force → 범주별 p_real 갱신 (RGF)
 → [피팅] (W_canon, s(t), E, γ) + 신뢰도 (홀드아웃 프레임 PSNR/LPIPS, 타이밍 σ)
출력 라벨 등급: 존재 / 순서 / 타이밍(σ<5ms)
```

### 7.3 AV-Physics Timing Solver [신규]
$t^{on}_i=t^{ev}_i+\beta d_i+\delta_{clip}+n_i$로 모델링하고 구간 검열 likelihood를 쓴다. $\beta$로 sync style을 분류한다. 식별 조건은 $\Delta d/c\gg\sigma_t/\sqrt N$이다. cinematic sync 클립은 오디오 생성의 스타일 데이터로 쓴다.

### 7.4 Gold Set
1000fps 다시점, 다중 거리 마이크, LiDAR로 50시간을 확보한다. 검출기 precision/recall과 타이밍 오차 곡선을 보정하는 데 쓰고, 독점 해자가 된다.

### 7.5 재조합 엔진 [부품: BEDLAM 방식]
(W, γ, P, A)를 어포던스 키로 매칭하고, 물리 리타게팅으로 실패한 조합은 기각한다. 카메라는 샷 문법을 유지하면서 리타게팅한다. 단, 검증 게이트를 통과한 조합만 쓰고, 배치의 50% 이상은 실제 lift 데이터로 고정한다.

### 7.6 Native World Foundation Model
- **토크나이저.** 정적 부분은 2m 청크마다 12.5cm sparse voxel(16³) × 8채널 latent를 두고, voxel당 Gaussian 32개를 둔다. 각 Gaussian은 intrinsics, $\theta$, slot을 가진다. 에셋은 SLAT를 perceiver로 512~2048 토큰으로 줄이고, 사건은 1024 토큰이다. 10초 장면이 약 50~64k 토큰이다.
- **World DiT(multi-stream).** 연속 성분은 flow matching, 이산 성분은 masked diffusion으로 학습하고, 여기에 §1.4의 신호들(렌더 기반 DMD 포함)을 라우팅 표대로 더한다.
- **규모 가설.** 정적 4~8B, 에셋 1~3B, 사건 0.5~2B, 동역학 $F_\phi$ 0.3~1B, 셰이더 $g_\varphi$ 0.2~0.5B, 정책 0.1~1B.
- **추론 경제성.** 1080p 10초 비디오 DiT는 약 50만 토큰에 attention 비용이 제곱으로 붙는다. 우리는 64k 토큰을 한 번 생성하고, SMC와 렌더만 돌린다. 카메라를 10대로 늘려도 추가 비용은 렌더 비용 수준이다.

### 7.7 Builder–Reader self-play와 적대적 감사
```
W ~ B_θ;  V = render(W, γ ~ 카메라 문법 prior, 센서·압축 열화)
R_φ ← 지도학습 d(R_φ(V), W)                    # 완전 GT
R_φ로 실영상 재-lift → p_real(F_g), 실사건 prior, 코퍼스 갱신
B_θ ← RL(r):  r = r_RGF + r_cycle + r_real + r_img-slice − r_adv
       (각 항은 §1.4 라우팅 표의 대상 텐서로만 기울기 전달)
적대적 시공간 카메라: min_θ max_ψ D(R(B_θ(c), C_ψ))   [부분 신규]
붕괴 방지: 실데이터 ≥50%, 커버리지 정규화, 검증 게이트, 실제 홀드아웃 상시 감시
```

### 7.8 권리
라이선스 스톡, 부동산 플랫폼, 스포츠 리그 멀티캠, 공연사와 수익을 공유하는 파트너십, 자체 캡처, 배우 퍼포먼스 마켓플레이스(동의 기반)로 확보한다. 오픈 모델은 라이선스 지역 조항까지 검토한다.

## 8. 평가 v2

### 8.1 Factorized Turing Battery [신규 프로토콜]
- **Motion Turing**: 외형을 고정하고 궤적 {실제, 우리, 비디오 lift}를 비교한다.
- **Appearance Turing**: 실제 궤적을 고정하고 {hold-out 실제 footage, 우리 렌더}를 비교한다.
- **Joint Arena**.
- 판정은 2AFC 식별률(50%가 완벽)과 선호율로 한다.

### 8.2 BUPS Arena
- 프롬프트는 시간 문법 샷이다: 동작 중 bullet time, speed ramp, 재촬영, 1000fps 슬로모, dolly zoom, 원테이크.
- 경쟁자: Wan 3.0, Seedance 2.5, MiniMax H3, Kling 3.0 Omni(motion control 포함), Veo 3.1 / Gemini Omni 1.1 Flash, Runway Gen-4.5. 각각 best-of-4와 자사 도구를 쓴다.
- 판정: 각 경쟁자 대비 승률의 95% CI 하한이 50%를 넘어야 한다.
- 별도 트랙: 일반 프롬프트 Arena. 프롬프트의 20%는 비디오 모델에 유리한 것으로 채운다.

### 8.3 지표 세트

| 지표 | 출처 | 겨냥하는 문제 |
|---|---|---|
| GeCo fused (정지 슬라이스, 정적 배경) | 부품 | (1)(2) |
| MEt3R, Revisit(SCoPE/ViewBench LCE, R2M 상대 비교) | 부품 | (1) |
| Physics-IQ, VideoPhy-2, FVMD | 부품 | 물리, 모션 분포 |
| **TSR**(PSD 기울기, HFER, $f_c$), **CSI** | 우리 | (3) |
| MI = GeCo 변형 항 + 객체별 강체·ARAP 잔차 | 부품+ | (2) |
| **BTT-A**(동작 중 bullet time), **RST**(재촬영 궤적과 타이밍) | 우리 (제품 시험) | (1) |
| **RRS, GSI, albedo leak** | 우리 | edit-closure |
| **ghost-force AUC, RGF 거리** | 우리 | 물리 사실성 |
| 인체 지표: foot skate, 관통, MANO 잔차, ArcFace, LSE-C/D, eyeline | 부품 | 2c |

### 8.4 Video Physics Spectrum Report
모든 상용 모델의 $R(f)$, $f_c$, CSI, g-check를 공개한다. VBench의 motion smoothness가 질병에 상을 준다는 상관 분석도 함께 싣는다. 문제를 시장이 읽을 수 있는 숫자로 만드는 것이 목적이다.

## 9. 로드맵 v2 (목표는 원안 이상으로 유지)

### 2a. Frozen Action & Speed Ramp (Phase 1 이후 +3~6개월) [v2 신규 단계]
- **범위.** 사건 한가운데서 정지한 세계를 만든다: 유리 파괴, 물 왕관, 폭발 파편, 도약 댄서(포즈 고정), 흩날리는 머리카락과 천. 그 상태에서 궤도 촬영을 하고 speed ramp(0 → 0.05x → 1x)로 실시간까지 복귀한다. 세계 시간 창은 0.3~1.5초다.
- **기술.** Edit-closed 세계, 짧은 창의 PCGD/MPM/FLIP, Milestone SMC와 Hero-Instant Selection, 슬라이스 DM, 정지 순간의 오프라인 렌더(3DGRT와 액체 표면 path tracing)를 쓴다.
- **킬러 데모.** 실사 바에서 와인잔이 깨지는 순간 정지한다. 와인 왕관과 파편을 360° 궤도로 돈 뒤 1x로 복귀하고, 반대 각도에서 재촬영하면 파편 하나하나가 같다.
- **승리 조건.**
  - BUPS-FA(n≥300)에서 각 경쟁자 대비 Arena 70% 이상.
  - 정지 슬라이스의 Appearance Turing 식별률 65% 이하.
  - 정지 궤도 구간 GeCo fused 0.03 이하(실제 바닥값 0.02).
  - ramp 구간 TSR이 실제 밴드 안.
  - RST 사건 타이밍 1프레임 이하.
- **막히면 업그레이드.** 액체 렌더가 약하면 액체만 메시 표면으로 path tracing하고, MAN 셰이더에 굴절 G-buffer(IOR, 두께)를 추가한다.

### 2b. 강체·관절체와 살아있는 세계 (+6~12개월)
- **범위.** 낙하, 충돌, 쌓기와 무너짐, 문과 서랍, 차량, 파괴, 입상체, 연체, Rube Goldberg, 그리고 식생, 바람, 연기, 물, 구름, time-lapse.
- **기술.** PCGD + RGF, Milestone SMC, 공유 바람장과 Spectral Gaussians(부품), 물성 증인 캠페인(RGF 기반).
- **킬러 데모.** 원안의 Rube Goldberg 3캠과 1000fps 장면, 60초 드론 orbit 중 돌풍과 하이퍼랩스.
- **승리 조건.**
  - Physics-IQ와 VideoPhy-2 1위.
  - Motion Turing(S 대 R) 식별률 60% 이하.
  - CSI KL이 최고 경쟁자의 0.5배 이하.
  - 강체 MI가 노이즈 수준.
  - BUPS Arena 65% 이상, 일반 Arena 50% 이상.
  - GCM 지연이 $d/U$의 ±20% 이내.
- **막히면 업그레이드.** sim look이 남으면 실데이터 캡처 캠페인을 10배로 늘려 $F_\phi$ 잔차를 키운다. 물이나 불이 약하면 해당 영역만 시뮬과 학습 난류 디테일로 대체한다.

### 2c. 캐릭터·연기 (+12~18개월)
- **범위.** 단일·다중 인물, 보행, 댄스, 스포츠, 대화와 립싱크, 클로즈업, 손, 의상, 머리카락, 동물.
- **기술.** §4.4 스택, Avatar Foundry v2, 고차원 expression latent, Coverage Compiler.
- **킬러 데모.** 2인 대화를 하나의 4D 테이크로 찍어 master, OTS, CU를 자동으로 얻는다. 디렉션별 테이크 3개, 치마와 긴 머리가 있는 120fps 댄스.
- **승리 조건.**
  - 전문가 패널 기준 Arena 연기 축 55% 이상, 멀티숏 연속성 70% 이상.
  - 얼굴 Appearance Turing 식별률 75% 이하.
  - foot skate 2cm/s 미만, ArcFace 0.8 이상, eyeline 3° 미만, 관통률 1% 미만.
- **막히면 업그레이드.** 얼굴 전용 실제 증인 코퍼스(talking 영상 lift), 퍼포먼스 캡처 마켓플레이스, 라이트 스테이지 파트너십.

### 2d. 완전한 4D와 인터랙션 (+24~36개월)
- **범위.** 다인물 상호작용, 손-객체 조작, 캐릭터와 상호작용하는 유체·파괴, 분 단위 원테이크, 다장면 영화, 실시간 인터랙티브.
- **기술.** native WFM이 세계와 사건을 직접 생성하고, SMC가 실현하며, 증인은 졸업한다. 분기 우주를 쓴다.
- **승리 조건.**
  - 모든 범주 Arena 60% 이상.
  - Revisit 일관성의 시간 기울기가 0에 가깝다.
  - 인과 정확성 1위.
  - 5분 원테이크를 멀티캠으로 재촬영했을 때 RST 기준을 만족한다.

## 10. 스케일링 가설과 컴퓨트 v2

| 가설 | 내용 | 검증 |
|---|---|---|
| **H1′** | 인간이 판정한 외형·모션 사실감이 같은 지점에서, 일관성·물리 축의 전체 컴퓨트(학습+추론)가 $C_{world}\le C_{video}/\kappa$, $\kappa\ge10$ | $10^{20}\sim10^{22}$, 같은 데이터 원천, Factorized Turing 점수로 fit. M6~12 go/no-go |
| H2 | lift한 세계 하나가 비디오 $K_{eff}$개(5~50)의 가치를 가진다 | 원안과 같음 |
| H3′ | 모션 손실은 *타이밍 라벨이 붙은* 사건 수의 멱법칙을 따른다 | 존재 라벨과 타이밍 라벨을 분리 측정 |
| H4 | self-play 결합. 천장은 실데이터 앵커가 정한다 | 소규모 3세대 |
| H5′ | SMC 입자 수 $N$에 대해 제약 충족률과 Arena가 $\propto\log N$, 비디오 best-of-N보다 높은 포화점 | 같은 비용에서 비교 |
| **H6** | 반사실 비율 $\rho_{cf}$에 대해 RRS가 0.3~0.5에서 포화하고, 원래 뷰 품질 저하는 2% 이하 | X3 확장 |
| **H7** | 슬라이스 DM, 물리, 실사건 prior만으로(비디오 고대역 없이) Motion/Appearance Turing 동률에 도달 | 2b 중반 |

| 항목 | H100-시간 |
|---|---|
| 정적 lift 5천만 클립 | 0.1~0.2M |
| 동적 lift 2천만 클립 + blur 측정 + timing solver | 0.3~0.5M |
| 반사실 DM 추가 교사 호출 (T1~T2) | +0.1~0.3M |
| Milestone SMC 실현 100만 샷 (약 10 GPU-분/샷) | ~0.17M |
| $F_\phi$, MAN 셰이더, 정책 RL | 0.1~0.3M |
| WFM T0 / T1 / T2 / T3 | 7k / 70k / 0.7M / 7M |
| Gold Set 캡처 | 장비와 인력 약 $0.3~0.6M |
| **2c까지 누계** | **약 2.5~5M (약 $6~14M)** |
| **2d 프런티어** | **+8~10M** |

H1′가 소규모에서 성립하지 않으면 원인을 분해한다. 토크나이저 문제인지, lift 품질 문제인지, 라우팅 문제인지, 상태와 외형의 분리 지점 문제인지 보고 표현을 업그레이드한다.

## 11. 공간 음향 v2

- **Single-cause multimodality(유지).** 충돌과 타격은 사건 $\mathbf{J}$로 모달 합성(Zheng & James)을 구동하고, 파괴, 유체 기포(Harmonic Fluids), 발소리(사건과 지면 재질), 대사(UTTERANCE → TTS, 같은 음소 타임라인이 입술을 구동), 바람(공유 바람장), 앰비언스로 이어진다. 모두 부품이다.
- **전파.** image-source와 확률적 음선 추적을 쓰고 RT60은 Sabine 식으로 검증한다. 도플러는 시변 지연선에서 자연스럽게 나온다: $\tau(t)=\lVert x_L(t)-x_k(t-\tau)\rVert/c$.
- **[v2] Sync Style 스위치.** *physical*은 전파 지연을 적용하고, *cinematic*은 그림 싱크 관례를 따른다. cinematic 관례는 F6에서 분류한 cinematic sync 데이터로 학습한다.
- 마이크 세계선을 따로 둘 수 있다(붐, 라발리에). 출력은 HRTF 바이노럴, Ambisonics, ADM 메타데이터가 붙은 Atmos 객체 스템이다.

## 12. 부가 능력
- OpenUSD로 세계와 카메라를 내보낸다. AOV(깊이, 법선, 모션벡터, cryptomatte)와 **intrinsics 패스**를 함께 낸다. edit-closed이므로 포스트에서 재조명이 가능하다.
- 스테레오, VR180, 6DoF 볼류메트릭.
- 로보틱스·피지컬 AI 시뮬레이터: metric 스케일, 물성 분산, RGF로 보정된 동역학을 그대로 쓸 수 있다.
- 인터랙티브: 분기 우주와 실시간 렌더(미리보기는 $F_\phi$ emulator).

## 13. 리스크와 업그레이드 v2

| 리스크 | 업그레이드 |
|---|---|
| 숨은 면과 조명 bake | Edit-Closed World. 안 되면 객체 공간에서 완결된 compositional 생성(캔버스 채우기 아님) |
| 움직이는 물질의 CG 룩 | 상태 조건부 외형, MAN 셰이딩, 슬라이스 DM, 3DGRT |
| sim look, 현실을 기각하는 판사 | PCGD 실잔차 학습, RGF 두 표본 판정 |
| 증인의 병 전염과 약관 | 대역 분업, 진술 그래프, PC-ELBO, 실제 증인, 닫힌 API는 평가 전용 |
| 혼돈계의 비수렴 | Milestone SMC, Reality Residual Budget, Event Type Contract |
| 웹 오디오와 블러 라벨 오염 | Timing Solver, Blur-as-Measurement, Gold Set, 라벨 등급 |
| 자기 채점과 잘못된 첫 전장 | Factorized Turing, BUPS Arena, 2a = Frozen Action |
| 비디오 모델의 급속한 개선 | 구조 축(BTT-A, RST, 커버리지, 납품물, edit-closure 재조명)은 유지된다. 오픈 모델이 좋아지면 저주파 증인도 함께 좋아진다 |
| 오픈 모델 라이선스(지역, 비상업) | Apache-2.0 계열을 우선 쓰고 법무 게이트를 둔다 |
| H1′ 불성립 | 원인을 분해한 뒤 표현을 업그레이드한다(후퇴하지 않는다) |

## 14. 첫 12개월 (Phase 1과 병행)

| 기간 | 실험·시스템 | 게이트 |
|---|---|---|
| M0–3 | **X1** Spectrum Report v0와 $f_c$ | 결손 확인 → Report 공개 |
| M0–3 | **X2** Factorized Turing v0 (강체·연체, MAN arm 포함) | Motion 식별률 60% 이하 / 외형 +15%p |
| M0–3 | **X3** Edit-Closed 장면별 DM | RRS ≥ 기준선, leak −50% |
| M0–3 | **X4** Milestone SMC | 충족률 90% 이상 |
| M0–3 | **X5** RGF 보정과 Sync-Sound 판별 | AUC ≥ 0.8, 정확도 ≥ 90%, ≤ 5ms |
| M3–6 | **2a 시스템**: Frozen Action & Speed Ramp, Hero-Instant, BUPS-FA Arena v1 | Arena 70% 이상 |
| M3–6 | Lift v2 100만 클립, Gold Set 50시간, 타이밍 라벨 코퍼스 | 타이밍 σ 분포 |
| M6–9 | EPD v2와 Event Type Contract, PCGD v1(강체·MPM), Edit-Closed amortized v1 | CSI·TSR가 실제 밴드 안 |
| M6–12 | **H1′ go/no-go** ($10^{20}\sim10^{22}$) | κ 추정 |
| M9–12 | 캐릭터 스택 프로토타입, Coverage Compiler 데모, self-play 3세대, 적대적 카메라 감사 | 2인 대화 멀티캠, RST |

---

### 맺음말

Phase 1은 촬영할 공간이 있으면 비디오가 카메라 경로 문제가 된다는 것을 보여 준다. Phase 2 v2는 여기서 한 걸음 더 나간다. 촬영할 *블록*이 있으면 움직임은 판정의 문제가 된다. 사진 한 장을 가장 잘 아는 이미지 교사가 모든 순간을 판정하고, 현실의 잔차까지 아는 물리 검증기가 모든 인과를 판정하며, 실제 소리와 블러에서 읽어 낸 사건이 모든 타이밍을 판정한다. 비디오 모델은 자신이 정확히 아는 대역에서만 증언한다. 확률은 원인과 미시 잔차에 두고, 보장은 불변량에 걸며, 카메라는 읽기만 한다. 첫 전장은 시간이 멈춘 사건의 한가운데다. 그 정지 순간의 매 프레임은 이미지 디퓨전이 이미 이기고 있는 사진 한 장이다.
