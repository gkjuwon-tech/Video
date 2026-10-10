> **출처 메모.** 리서치 1건, 렌즈 4개의 원안, 각 렌즈의 건설적 프리모템 개정안을 수석 아키텍트가 하나로 통합한 명세다. 이 레포의 기준 문서이고, 원자료는 [`docs/research/`](research/)와 [`docs/lenses/`](lenses/)에 있다.

---

# Block Universe Cinema: 통합 패러다임 명세 v1.0

> 기준일: 2026-10-10.
> 입력: 지형도 리포트(2026-10-09)와 4개 렌즈(first-principles, representation, cinematography, phase2-4d-and-flywheel)의 원안 및 프리모템 개정안.
> 신규성 표기: **[신규]** 공개된 선례를 찾지 못함 / **[조합]** 부품은 이미 있고 결합 방식이 새로움 / **[부품]** 기존 것을 빌려 씀.
> 배치 태그: `P1-코어` `P1-확장` `P2` `장기`.
> first-principles 개정안은 앞부분이 잘린 채 입력되었다. 그 안의 용어(Partition Principle, World-Scale Causality, Frozen-Seed Band-Commit, SMF, WDD, TPI, CI, `<plate>`)는 남아 있는 본문과 맥락을 근거로 이 문서에서 정의를 확정했다.

---

## 0. 이름, 테제, 슬로건

### 0.1 이름 후보

| 후보 | 뜻 | 장점 | 약점 |
|---|---|---|---|
| **Block Universe Cinema (BUC)** | 물리학의 블록 우주(모든 시각이 이미 존재하는 4D 블록). Phase 1은 시간 t가 상수인 슬라이스, Phase 2는 블록 전체 | Phase 1과 2를 한 이름으로 묶는다. "시간도 카메라가 고르는 좌표"라는 핵심이 이름에 들어 있다 | 학술적으로 들린다 |
| Profilmic | 영화이론 용어 profilmic event, 즉 카메라 앞에 놓인 실재 | 창업자 테제를 그대로 담는다. 브랜드로 쓰기 좋다 | 일반인에게 낯설다 |
| STRATA | 세계를 주파수 지층(stratum)으로 쌓는다 | 기술 정체성이 뚜렷하다 | 매개체(representation) 이름에 가깝다 |
| Worldshot | 세계를 찍는다 | 직관적이다 | 흔한 조어다 |
| Set & Shoot | 세트를 짓고 찍는다 | 현업 언어다 | 정지된 세계라는 인상을 준다 |

**추천: Block Universe Cinema.** 패러다임과 논문 이름으로 쓴다. 구성요소 이름은 다음과 같이 고정한다.

- 세계 매개체: **STRATA**
- 세계 솔버(탐색): **WSSC v3**
- 세계 학생(정책): **WDM 생성기**
- AI 촬영감독: **DP-0**
- 벤치마크: **FREEZE-TAG Bench(얼음땡)**

Profilmic은 회사나 제품 브랜드 후보로 남겨 둔다.

### 0.2 한 문장 테제

> **비디오는 그려 내는 것이 아니라 관측한 결과다.** 확률은 세계에만 둔다(Phase 2부터는 사건에도 둔다). 이미지 디퓨전은 그 세계를 모든 주파수 대역, 모든 시점, 모든 순간에서 판정하는 심판 역할을 한다. 카메라, 렌즈, 센서는 세계를 결정론적으로 읽기만 한다.

### 0.3 슬로건

- 메인: **"그리지 말고, 찍어라." (Don't draw it. Shoot it.)**
- 보조: "세계는 한 번, 테이크는 무한." / "프레임마다 새로 뽑는 랜덤은 필름 그레인 하나뿐."

### 0.4 한 장 요약

```
 y (프롬프트/스크립트)
   │  LLM World Program: prompt_tree@scale, scene graph(slot·metric), matter class, light/shot intent
   ▼
 [Bedrock] Coarse World Model (CWM): 전역 1-pass, 레벨 ≤ l_g (내부·뒷면 포함, 구멍 없음)
   │  hero still = 관측 제약 (DPS), 캔버스 아님
   ▼
 [Demand] DP-0 경로 분포 + 렌즈 계획 → Ray-Bundle Demand → {(level, block)} ∪ 완전성 집합
   ▼
 [Strata] 레벨 l = l_g+1 … L (스케일 순서. 같은 레벨 안에서는 순서 없음, lockstep)
   │  solver: WSSC v3 (band-isometric 뷰 × 이미지 교사 × World Noise Field × 대역 분할 소유권)
   │  student: LBO (amortized, WDM로 학습) ── Expert Iteration으로 solver와 묶임
   ▼  commit (Merkle, seed 동결)
 [Audit/Polish] 고정점 반복 + adversarial camera (Π 위 sup) + Gaffer/shot dress(directed 모드)
   ▼
 [Shoot] 3DGRUT 렌더 (dual-cut LOD, RAS 셰이딩, RT 반사) → 렌즈·셔터·센서·필름 → 8K·120fps, EXR AOV
          ※ Read-Only Camera: 렌더 경로에서 W에 쓰는 연산 0개, 프레임 단위 latent 0개
```

### 0.5 렌즈 충돌 판결록

| # | 쟁점 | 대립하는 안 | 판결 | 근거 |
|---|---|---|---|---|
| C1 | 세계 수렴 메커니즘 | 샘플 기반 동기화(FP의 WSSC, Cine의 PLWD, Rep v0.1의 S²R) vs 기울기 기반 분포 매칭(Rep v0.2의 WDM, P2의 slice DM) | **둘 다 쓰되 역할을 나눈다.** solver는 WSSC v3(샘플형), student는 WDM(기울기형). Expert Iteration으로 연결한다(§1.9) | WDM이 장면 규모에서 수렴한 선례가 없다(지형 §10). 반대로 solver는 샷당 수~수백 GPU-h가 든다. 두 형태는 같은 고정점 목표(E2)를 서로 다른 통계로 강제한다. X3에서 장면 단위로 정면 비교해 데모 경로를 정한다 |
| C2 | 교사 인터페이스 | 픽셀별 timestep T[p](Rep v0.1, FP의 원소·뷰별 결정대역) vs band-isometric 뷰(Rep v0.2) | **미세 레벨은 band-isometric 뷰와 스칼라 t, 코스 레벨만 시네마틱 뷰** | MMDiT는 t를 AdaLN에 스칼라로 넣는다. 토큰마다 t가 다르면 OOD이고 깊이 경계에 후광이 생긴다. 등척 뷰는 footprint 불균일을 교사 안에서 풀지 않고 카메라 선택 단계에서 없앤다 |
| C3 | 소유권 방식 | graph-cut hard(FP) vs softmax annealing(Rep) vs 후반 스텝만(Cine) | **결정 중인 대역에서 annealed 소유권을 쓰고 commit 시점에 hard partition + multiband blend.** 투표 전에 explain-away cascade를 먼저 돌린다 | Partition Principle(§1.4)에 따르면 소유 영역 안의 스펙트럼 보존은 K와 ρ에 무관하다. annealing은 초반에 모드를 합의시키고, hard commit은 블러를 막는다 |
| C4 | 렌즈 효과 처리 | `<plate>` LoRA(FP v2) vs RAW-pinhole 파인튜닝 + 역카메라(Cine v1) vs Distill-Through-Camera(Cine v2) | **DTC가 1순위, `<plate>`는 실패 시 대체안** | DTC는 교사를 고미감 manifold 위에 그대로 둔다. θ에 따라 바뀌는 효과는 식별성 논증에 따라 세계에 구워질 수 없다. plate 모드는 미감이 깎인다(FP 리스크 레지스터 자체 인정) |
| C5 | 외형의 정본 | RAS: radiance 정본(Rep) vs intrinsics 우선, edit-closed(P2) | **Dual-Anchored Appearance:** 정적 외형은 L_d가 정본, intrinsics 채널은 필수. 잔차 radiance 예산은 이동 가능 slot과 그 그림자를 받는 면에만 강제한다 | Phase 1의 Elo는 구운 GI가 지킨다(Marble이 사실적으로 보이는 이유). Phase 2의 편집성은 이동하는 부분에서만 필요하다 |
| C6 | 프리미티브 | Unified Kernel Atom(Rep) vs textured surfel atlas(FP) vs 하이브리드 5종(Cine) | **UKA 하나에 payload를 붙이고 예외 3개만 둔다:** 유전체 SDF, 시뮬 볼륨 VDB, 아바타 리그 바인딩 | 렌더러 범위가 폭발하는 것과 train/test gap을 막는다. FP의 texel patch는 UKA payload로 흡수한다 |
| C7 | LOD 렌더 | 대역 α-fade(Rep v0.1) vs dual-cut λ-블렌딩(v0.2) | **dual-cut** | 불투명 기하에서 대역 ⊕ 연산은 정의되지 않는다. α=0.5에서 배경이 25% 새어 나온다. dual-cut은 λ에 대해 연속이다(T9) |
| C8 | 세계의 목적함수 | Π 위 minimax(FP) vs Directed World(Rep U7) | **모드 둘:** Location(Π 위 minimax, 다중 샷과 재방문용), Directed(경로 중요도 + 미감 보상 + shot dress) | "어떤 경로로 찍어도"라는 약속과 "광고급 한 컷"이라는 요구는 서로 다른 제품 모드다 |
| C9 | Phase 2a 첫 전장 | 살아있는 정지세계(P2 원안) vs Frozen Action & Speed Ramp(P2 v2) | **Frozen Action.** 살아있는 세계는 2b로 옮긴다 | 정지 슬라이스는 곧 사진이므로 이미지 교사가 완벽한 판사가 된다. 시뮬 창이 짧다. 비디오 모델은 구조적으로 만들 수 없다. 물, 불, 연기는 3DGS가 가장 약하고 비디오 모델이 이미 잘하는 영역이다 |
| C10 | 동역학 원칙 | "결과는 결정론"(P2 원안) vs "보장은 불변량, 미시 잔차는 확률"(v2) | **v2** | 결정론적 시뮬은 현실보다 깨끗해서 sim look이 생긴다. 판사도 실분포로 보정해야 한다 |
| C11 | 비디오 모델의 역할 | 인턴이자 증인(P2 원안) vs 저주파 증인(v2) | **v2.** 닫힌 API는 평가 용도로만 쓴다 | 증인이 자기 병(고주파 결손)을 옮기는 것을 막는다. 약관 리스크도 있다 |
| C12 | 벤치 프레임 | 강체성 중심(Cine v1, Rep) vs 빈 사분면 SP×MI(Cine v2, P2 F7) | **빈 사분면.** GeCo를 MI의 코어로 빌린다 | GeCo가 이미 했다는 리뷰를 차단한다. 시장의 판결은 미감과 일관성을 동시에 이겨야 나온다 |
| C13 | latent 시간 Nyquist | 단정(FP, Cine v1) vs 가설(Cine v2, P2 v2) | **가설 H-ν로 둔다.** X0에서 VAE 왕복 실험으로 원인별 기여도를 측정한다 | 비디오 VAE 디코더는 비선형 생성기라서 Nyquist가 하드 리밋은 아니다 |
| C14 | Drift = effective resistance | FP v1은 신규로 주장 vs FP v2는 차용으로 인정 | **차용(Barooah–Hespanha 그래프 추정 이론).** 아웃페인팅 비판에 적용한 것만 우리 몫 | 신규성 장부를 정직하게 유지한다 |
| C15 | Factorized Recombination 신규성 | 신규(P2 원안) vs 부품(v2) | **부품(BEDLAM 계열)** | 선례가 있다 |

---

## 1. 핵심 인수분해와 핵심 수식

### 1.1 인수분해 (E1)

$$
\boxed{V_k(\mathbf u)=\mathcal S_k\!\left(\frac{1}{S}\int_{t(\tau_k)-S/2}^{t(\tau_k)+S/2}\!\!\int_{\mathcal A}\mathrm{Cam}_\theta\,\mathcal R\big(W(t'+\rho(\mathbf u)),\ \Gamma(\tau_k,\boldsymbol\xi_a)\big)(\mathbf u)\,d\boldsymbol\xi_a\,dt'\right),\qquad
(W,E,z,\gamma)\sim p_\omega(W\mid y)\,p_\theta(E,z\mid W,y)\,p_\pi(\gamma\mid W,y)}
$$

- $\mathcal R$: scene-linear HDR, all-in-focus 렌더. 결정론적이다.
- $\mathrm{Cam}_\theta$: 렌즈 연산자. 비네팅, 플레어, 노출, 톤 커브를 포함한다.
- $\mathcal S$: 센서와 필름. 프레임마다 iid 그레인을 넣는데, 이것이 **유일한 프레임별 랜덤**이다.
- $\rho(\mathbf u)$: rolling shutter 행 지연.
- $\gamma(\tau)=(t(\tau),\Gamma(\tau),S(\tau))$: 카메라 세계선(worldline).
  - $dt/d\tau=1$: 정상 재생. $0<dt/d\tau<1$: 진짜 슬로모션. $=0$: bullet time. $<0$: 역재생. 불연속: 컷.
- **Phase 1:** $W(t)\equiv W$, $E=\varnothing$.
- **Phase 2:** $W(t)=\mathrm{Deform}(W_{canon}, s(t))$. 확률은 원인($W$, $E$)과 미시 잔차에만 둔다.

### 1.2 인수분해에서 나오는 정리들

| # | 정리/명제 | 내용 | 보장 수준 | 겹침/출처 |
|---|---|---|---|---|
| T1 | Type-safe Consistency | 정적 $W$에서 두 뷰가 공통으로 보는 점 $p$에 대해 $x_i(\pi_ip)=L_W(p,\omega_i)$. 같은 plenoptic 함수의 샘플이므로 Lambertian 성분은 항등적으로 일치한다 | 정의상 | — |
| T2 | Loop Closure | $C_t=C_s\Rightarrow x_t=x_s$ (그레인 off). 비트 단위로 같다 | 정의상 | — |
| T3 | Drift = Effective Resistance | 조건부 그래프에서 간선 오차의 분산이 ε²이면 $\mathrm{Var}[\theta_i-\theta_j]=\varepsilon^2R_{\text{eff}}(i,j)$. 순차 아웃페인팅과 AR 비디오는 path graph라 $\lvert i-j\rvert$. 우리는 star graph라 $2\varepsilon^2_{fit}$이고 렌더 후에는 0 | 표준 결과 | **[부품]** Barooah–Hespanha. 아웃페인팅이 퇴화라는 논증에 적용 |
| T4 | Dedup Dividend | $r=THW/U$. 10초 24fps 1080p 느린 카메라 기준 $r\approx10$–25. 같은 예산이면 고유 texel 하나에 10–25배 공을 들일 수 있다 | 추정 | — |
| T5 | Spectral Deficit of Averaging | $\mathrm{SFR}_b=1-\frac{P_n}{P_s+P_n}(1-\rho_b)(1-\tfrac1K)$. 고주파에서는 $\approx1/K$. L2 재구성은 평균이므로 스펙트럼을 깎는다(CAT3D와 Lyra의 블러를 예측한다) | 가우시안 성분 가정 하의 정리. X1에서 검증 | **[조합]** |
| T6 | Partition Principle | 대역 $b$를 단일 소유자가 결정하면 소유 영역 안에서 $\mathrm{SFR}_b=1$이고 $K$, $\rho$와 무관하다. 손실은 seam 폭 $\lambda_b\propto1/f_b$ 안에만 생긴다 | 정리. seam은 2AFC로 검증 | **[조합]** Waechter'14와 Burt–Adelson의 view selection을 디노이징 루프 안으로 |
| T7 | Proximity Precedence | 세계 주파수 $f^w$가 뷰 $k$에서는 $f^w z_k/(F_k\cos\theta_k)$로 보인다. 그러므로 대역은 footprint가 작은 뷰부터 결정된다. 근접 우선 소유가 결정의 인과 순서와 같다 | 파워법칙 스펙트럼 가정 하의 근사 정리 | **[조합]** Dieleman의 spectral autoregression 관점 |
| T8 | Path/Order Independence | native LBO(유계 수용장, Philox 노이즈)에서 레벨 귀납으로 증명된다. 어떤 수요 순서든 비트 단위로 같다 | 정리(결정론 커널). solver 모드는 근사(OIE) | **[조합]** Minecraft 월드젠 패턴 |
| T9 | λ-Continuity | dual-cut 렌더 $I=(1-f)I^{(l_a)}+fI^{(l_a+1)}$은 λ에 대해 연속이다. 팝핑은 구조적으로 0 | 정리 | — |
| T10 | Lens Identifiability | θ(초점, 조리개, EV, 플레어)에 따라 변하는 효과는 θ를 무작위로 바꾸는 증류에서 세계에 구워질 수 없다. 세계는 all-in-focus HDR로 수렴한다 | 식별성 논증. X4에서 검증 | **[조합]** HDR-NeRF, RawNeRF 원리를 생성 증류에 적용 |

### 1.3 세계는 고정점이다 (E2)

$$
\boxed{\ \forall b,\ \forall C\in\Pi,\ \theta\sim\hat p_T:\quad \Pi_b\Big[\mathcal D^{\mathrm{own}(b)}_{\sigma_b}\big(\mathrm{Cam}_\theta\mathcal R(W^\star,C);\ \xi_W\big)-\mathrm{Cam}_\theta\mathcal R(W^\star,C)\Big]\approx0\ \ (\text{calibrated norm})\ }
$$

이미지 모델에게 우리 세계의 어떤 프레임을 보여 주고 "σ 스케일에서 고쳐 보라"고 하면 "고칠 게 없다"는 답이 돌아오는 세계를 말한다. 이 목표를 두 형태로 운용한다.

- **샘플형 (solver, WSSC v3):** $W_{n+1}=(1-\gamma_n)W_n+\gamma_n\,\mathrm{Sync}_{\sigma_n}\big(W_n,\{\mathcal D_{\sigma_n}(\mathrm{Cam}_\theta\mathcal R(W_n,C);\xi^{(n)}_W)\}_{C\in V_n}\big)$, $\sum\gamma_n=\infty$, $\sum\gamma_n^2<\infty$.
  - 각 렌더를 고정점(모드 근방)으로 끌어당긴다. 히어로 품질을 내는 경로다.
  - gradient가 아니라 샘플을 쓰므로 SDS의 mode-seeking(과포화)을 피한다.
  - contraction 조건은 엄밀히 증명하지 못했다. 열린 과제다.
- **기울기형 (student, WDM, DMD2식):** $\nabla_\phi\mathcal L=\mathbb E_{C,\theta,t}\big[w_t(\hat x_0^{fake}-\hat x_0^{real})^\top\partial u/\partial\phi\big]$, $u=\mathrm{Enc}(\mathrm{Cam}_\theta\mathcal R(G_\phi(\epsilon,y),C))$.
  - 렌더의 **분포**를 교사 분포에 맞춘다. seed마다 다양성이 보존되고 amortize할 수 있다.
- 두 형태는 같은 목표(렌더가 교사 manifold 위에 있을 것)를 다른 통계로 강제한다. 그래서 solver의 출력 $W^\star$를 student의 fake-score real anchor로 쓸 수 있다(§1.9).

### 1.4 동기화 연산자: Spectral Partition Consensus (E3)

세계 신호를 대역으로 분해한다. 대역은 STRATA 레벨과 레벨 내 sub-band(texel mip)다. $\mathrm{BP}_k$는 뷰 $k$를 세계로 back-projection하는 연산이다.

$$
\big[\mathrm{Sync}_\sigma(W,\{\hat x_k\})\big]_{i,b}=
\begin{cases}
\dfrac{\sum_k w_k^{cal}\,\mathrm{BP}_k(\hat x_k)_{i,b}}{\sum_k w_k^{cal}} & b<b_{lo}(\sigma):\ \text{결정 완료}\ (\rho_b\to1,\ \text{T5에 따라 평균 안전})\\[1.6ex]
\sum_k m_{k,b}(i)\,\mathrm{BP}_k(\hat x_k)_{i,b},\quad m_{k,b}=\mathrm{Blur}_{\lambda_b}\big(\mathrm{softmax}_k(\tau_s a_k(i,b))\big) & b\in\mathcal B(\sigma):\ \textbf{결정 중}\Rightarrow\text{소유권}\ (\tau_s\uparrow,\ \text{commit 시 hard})\\[1.6ex]
W_{i,b}\ \text{(WNF 노이즈 상태 유지)} & b>b_{hi}(\sigma):\ \text{미결정}
\end{cases}
$$

$$
a_k(i,b)=\underbrace{\log\tfrac{F_k\cos\theta_k(i)}{z_k(i)}}_{\text{footprint}}+\underbrace{\log T_k(i)}_{\text{가시성}}-\lambda_u\underbrace{\hat\sigma_k^2(i,b)}_{\text{보정된 불확실성}}+\lambda_e\underbrace{e_k(i)}_{\text{화면 가장자리 거리}}+\lambda_h\underbrace{h_k(b)}_{\text{hero 우선, 저대역만}}
$$

hard commit 시점의 소유자는 $o_b=\arg\min_o\sum_i-a_{o_i}(i,b)+\lambda\sum_{(i,j)}[o_i\ne o_j]\cdot\mathrm{seam}(i,j)$ (Potts, graph cut)로 정한다. seam 비용은 텍스처 마스킹이 큰 곳에서 낮게 둔다.

- **World-Scale Causality** [신규]. 대역 $b$의 결정권은 footprint가 $h_b$에 맞는 뷰(그 레벨의 band-isometric 뷰)에만 준다. 다른 뷰는 결정 창 $[t_{end}(b),t_{start}(b)]$ 밖에서 follower로만 남는다. follower는 세계에서 다시 렌더되고, 저대역 투표에만 참여한다. T7을 스케줄로 바꾼 것이다.
- **Janus 방지.** 최저 대역(레이아웃)의 소유자는 정확히 하나다. CWM 또는 hero 관측이 맡는다.
- **Free-space veto.** 어떤 뷰가 비어 있다고 본 공간은 내용을 소유하지 못해도 비어 있음에 대한 거부권을 갖는다. floater를 제거하는 장치다.
- **Stochastic Matter Field(SMF) 예외** [신규]. 풀, 잎, 자갈, 머리카락, 거품처럼 인스턴스 수준 결정이 뷰마다 충돌하는 물질은 픽셀을 투표에 부치지 않는다. **통계 τ**(밀도, 방향 분포, 크기 분포, 색 통계)만 합의하고, 인스턴스는 $G_{SM}(\tau,\text{instance seed})$가 생성한다. 지표는 CI(Cardboard Index), 즉 카드처럼 납작해 보이는 비율이다.

### 1.5 World Noise Field (WNF): 노이즈에 주소를 준다

세 렌즈의 world-anchored noise, Surface Noise Field, Philox 위치 주소 노이즈, 구조 주소 노이즈, Matter-Anchored Noise를 하나의 장(field)으로 통합한다.

$$
\xi_s=\Phi^{-1}\big(\mathrm{Philox}(\text{seed},\ell,s_{id},k)\big)\in\mathbb R^{C},\qquad
\varepsilon_v[j]=\frac{\sum_s w_{v,j}(s)\,\xi_s}{\sqrt{\sum_sw_{v,j}(s)^2}},\quad w_{v,j}(s)=a_s\cdot\mathrm{cov}_{v,j}(s)\cdot T_{v,j}(s)
$$

- 각 $\varepsilon_v[j]$는 정확히 $\mathcal N(0,I)$이다. footprint가 겹치지 않으면 서로 독립이다. 두 뷰가 같은 표면을 보면 footprint 겹침 비율만큼 상관된다. 교사가 가정하는 iid는 깨지지 않고, T5의 $\rho_b$를 올리는 레버가 된다.
- **Nested.** 거친 셀의 노이즈는 미세 노이즈를 정규화해 더한 값이다. 줌인은 같은 주사위를 더 자세히 읽는 일이 된다.
- band-isometric 뷰에서는 latent 토큰 하나의 표면 footprint가 균일하다. 그래서 적분이 latent 공간에서도 정확하다. 픽셀 노이즈와 latent 노이즈가 어긋나는 문제가 사라진다.
- 변형판:
  - CARRIER용 구조 주소 $\xi_n=\mathrm{Philox}(\text{seed},\text{inst},n_1,n_2)$
  - 하늘용 HEALPix 방향 노이즈
  - **Phase 2:** canonical 물질 좌표에 고정한 MAN(§8.4)
- **Frozen-Seed Band-Commit** [신규]. commit된 대역은 값과 seed를 함께 동결한다. polish는 열린 대역에만 "새로 뽑되 뷰끼리 공유하는" seed $\xi^{(n)}$를 쓴다. 동결된 대역을 다시 열려면 copy-on-write 브랜치를 만들어야 한다.
- 선례: How I Warped Your Noise(ICLR'24), Infinite-Resolution Integral Noise Warping(ICLR'25), Go-with-the-Flow(CVPR'25), SyncNoise, Consistent Flow Distillation. **새로운 점:** 장면 규모, 다중 LOD 표면 위의 단일 장이고, 생성 동기화에서 ρ를 올리는 레버로 이론화했다는 것.

### 1.6 Dreamer Calibration과 Explain-Away Cascade

**꿈 연산자를 측정 채널로 모델링한다.**

$$\mathcal D_\sigma(\mathcal R(W,C))=\mathcal R_u(W,C)+\eta,\qquad u\sim\mathcal N(0,\Sigma_u(\sigma,\text{content})),\quad \eta_b\sim\mathcal N(0,\Sigma_\eta(f_b,\sigma))$$

**보정.** 3DGS 렌더가 아니라 **실사 프레임**으로 한다. 데이터는 ScanNet++ DSLR + laser scan, DL3DV 원본 + COLMAP이다. 측정 항목은 다음과 같다.

- 결정 곡선: $D(o,t)=\mathbb E\,\mathrm{Var}_\epsilon[\mathrm{BP}_o(\mathrm{Dec}(\hat u_0))]/\mathbb E\lVert\mathrm{BP}_o(x_0)\rVert^2$
- $t_{start}(o^*)=D^{-1}(o^*,0.9)$, $t_{end}(o^*)=D^{-1}(o^*,0.1)$
- 한 옥타브 분리 조건: $D(o^*{+}1,t_{end})\ge0.5$, $D(o^*{-}1,t_{start})\le0.5$
- $\Sigma_u$: RAFT/UFM flow로 측정
- $\Sigma_\eta(f)$: 잔차 스펙트럼으로 측정
- $\rho(f,\sigma)$: WNF를 공유했을 때 측정
- **Worldness Gap** $\Delta_P(\sigma,b)=\mathbb E\lVert M_{ij}\odot\Pi_b(\mathrm{warp}_{i\to j}\mathcal D^P_\sigma(x_i;\xi)-\mathcal D^P_\sigma(x_j;\xi))\rVert$

**융합 likelihood.**

$$-\log p(\hat x_k\mid W)=\min_u\lVert u\rVert^2_{\Sigma_u^{-1}}+\sum_b\lVert\Pi_b(\hat x_k-\mathcal R_u(W,C_k))\rVert^2_{\Sigma_\eta(f_b)^{-1}}$$

미세 warp는 $u$가 흡수하므로 ghosting이 생기지 않는다. 고유 랜덤성이 큰 대역은 가중치가 낮아지고 결정권이 소유자에게 넘어간다.

**Cascade** (설명할 수 있는 불일치는 투표에 부치지 않는다).

1. **물리.** 시점 의존 radiance는 BRDF와 env/광원으로 피팅한다.
2. **Gauge.** 뷰별 노출과 WB affine(3×4), sub-pixel warp $g_k$를 피팅한 뒤 버린다.
3. **렌즈.** DTC가 $\mathrm{Cam}_\theta$로 설명한다.
4. **Dream jitter.** $u$로 설명한다.
5. **잔여 충돌.** 소유권 투표로 넘긴다.

**Dreamer 계약 (prior-agnostic).** 아래를 제공하는 prior는 무엇이든 연방에 들어올 수 있다.

- (i) 조건부 샘플
- (ii) 보정 프로파일 $(\Sigma_u,\Sigma_\eta,t_b,\rho)$
- (iii) 선택 사항: σ 접근권. σ에 접근할 수 없는 닫힌 API 모델은 ρ≈0이다. 하지만 T6 덕분에 POLISH 단계의 L3 소유자 후보로는 들어올 수 있다(렌더 → edit → flow 정렬 → gauge 제거). 단, 약관을 확인해야 한다.

### 1.7 Spectral Federalism (Phase 1 관할 지도) [신규]

> 헌법 제1조: 어떤 prior도 자기 관할 주파수 대역 밖에서는 발언할 수 없다.
> 관할 배정 규칙: $\mathrm{owner}(\text{cell})=\arg\min_P\Delta_P(\text{cell})+\mathrm{Var}_P(\text{cell})$. 사람이 정하지 않고 측정으로 정한다.

| 채널 \ 대역 | DC (전역) | L0 레이아웃 | L1 형태·실루엣 | L2 재질·중간 텍스처 | L3 미세 (latent Nyquist 이하) | L4 극미세 (latent Nyquist 초과) |
|---|---|---|---|---|---|---|
| 기하 | LLM scene graph (metric) | CWM + 3D object prior (TRELLIS/Hunyuan3D, slot 단위) | dreamer depth head (partition) | 시차가 있는 구간까지만 기하, 나머지는 normal | dreamer normal | WDD normal detail |
| 외관 | 팔레트, 시간대 | CWM + hero 관측 (DPS) | 오픈 이미지 교사 (partition) | 교사 + intrinsic head | 교사. polish 때는 닫힌 API dreamer(ρ≈0 허용) | **WDD**: seed 결정론적 detail diffusion |
| 확률적 물질 | matter class 분포 | 캔버스 segmentation | SMF 통계 투표 τ | $G_{SM}$ 인스턴싱 | $G_{SM}$ | $G_{SM}$ LOD seed |
| 승격 구조 (PSH) | — | CARRIER 위상 ψ, POINT, CURVE, SYMBOL (가시도·결맞음으로 승격) | 〃 | 〃 | — | — |
| 시점 의존 | env, sun | env fit | RT로 $W$ 자신을 반사 | intrinsic roughness | — | — |
| 사진 | 렌즈 패키지 | **카메라만** (DTC: 교사는 카메라를 통과한 이미지만 본다) | 〃 | 〃 | 〃 | 〃 |
| 조명 강조 | 스타일 | Gaffer (샷 단위) | 〃 | 〃 | — | — |

Phase 2에서는 여기에 시간 주파수 축을 더한다(§8.2). 연방 원리는 뷰들 사이(§1.4)와 prior들 사이(§1.7)에 똑같이 적용된다. **평균 내지 말고, 그 대역을 가장 잘 아는 자에게 맡겨라.**

### 1.8 의사코드: WSSC v3 (solver)

```text
procedure WSSC_v3(y, mode∈{location, directed}):
  # ---- 0. 준비 ----
  P  ← WorldProgram_LLM(y)              # prompt_tree@scale, scene graph(slot, metric bbox), matter classes, light/shot intent
  H  ← HeroStills(P)  [선택; 닫힌 SOTA = art director, ToS 확인]
  W  ← CWM(P, H) or Bootstrap(P, H)     # levels ≤ l_g: structure, coarse z, PSH structs, far field, lights, slots, intrinsics
  Π  ← (mode=location) ? Π_broad(P) : Π_shot(P) ⊕ margin
  D  ← RayBundleDemand(DP0.prior(W,P), lens plan) ∪ CoverageSet(Π)   # {(l, block)}
  calib ← DreamerProfile(teacher)        # D(o,t), Σ_u, Σ_η, ρ, Δ_P

  # ---- A. STRATA: 스케일 순서, 레벨 안에서는 lockstep ----
  for l = l_g+1 … L:
    blocks ← D[l]
    V ← ∪_b BandIsoViews(b, l)           # |log2 φ_v − log2 φ*_l| ≤ ¼, z_max/z_min ≤ 2^¼, θ ≤ 33°, 4–8 views/block
        ∪ (HeroCams declared in y) ∪ AdvCams(W, Π, l)
    [t_s, t_e] ← calib.window(o*)        # 한 옥타브 결정 창
    ∀v: θ_v ~ p̂_T(θ | v, y);  c_v ← SelfCaption(Render_v(W), sem, scale=W_px·φ*_l m, lensTags(θ_v)) ⊕ prompt_tree(l)
        z_v ← √ᾱ_{t_s}·Enc(Cam_θv R_v(W)) + √(1−ᾱ_{t_s})·ε_v(WNF; seed, l, ·, k=0)
    for k = 1..K   (K = 2–4, distilled teacher, APG guidance; fine levels use guidance 1.0–1.5)
      ∀v (병렬, lockstep): (x̂0_v, ĝ_v) ← Teacher(z_v, t_k, c_v, ctrl = Gbuffer_v(W))     # RGB (+depth/normal head)
      # explain-away cascade
      W.spec ← FitViewDep(W, x̂0);  g ← FitGauge(W, x̂0);  u ← FitJitter(Σ_u)
      W.SMF.τ ← StatVote(stochastic-matter cells)                                        # 통계만 합의
      # 대역 동기화 (E3)
      W ← FitThroughCamera(W, {x̂0_v, θ_v}):
            L = Σ_v ‖G_ωc*(Cam_θv R_v(W) − x̂0_v)‖_ρ                     # 저역: robust 합의
              + Σ_v Σ_{p∈own_k(v)} ‖H_ωc*(·)‖₁                          # 결정 대역: annealed 소유권
              + λ_down‖G_{ω_{l−1}}*(R(W_l) − R(W_{l−1}))‖₁               # 부모 대역 보존 (filial)
              + λ_attr L_attr + λ_geo L_geo(ĝ) + FreeSpaceVeto
      ∀v: z0_v ← γ_k·Enc(Cam_θv R_v(W)) + (1−γ_k)·x̂0_v                 # γ_k → 1
          z_v  ← SamplerStep(z_v, z0_v, t_k → t_{k+1}, ε_v(WNF, k))
    HardCommit(l, blocks): graph cut 소유자 + multiband blend(λ_b ∝ 1/f_b); seed 동결; Merkle hash; unc_epi ← cross-view var

  # ---- B. POLISH / AUDIT (고정점, E2) ----
  repeat until sup_{C∈Π̂} DR_rel(W, C) ≤ 1.2  or budget:
    V_n ← V ∪ AdvCams(W, Π)        # CMA-ES over pose×focal(14–200mm)×aperture; energy = calibrated dream residual
    W   ← (1−γ_n)W + γ_n·Sync_σn(W, {D_σn(Cam_θ R(W,C); ξ^(n))}_{C∈V_n})   # 열린 최미세 대역만

  # ---- C. SHOOT ----
  Γ ← DP0.search(W, P, rig, style);  L_shot ← Gaffer(W, Γ);  (directed) W_shot ← ShotDress(W, Γ)
  return Sensor(Lens(∫∫ R(W_shot, Γ(t+τ, ξ_a)) dξ dτ))  for each frame
```

**비용 감각.**

| 규모 | 조건 | H100-h |
|---|---|---|
| 방 규모, 4K 오빗 | 레벨 4–6개, 레벨당 뷰 200–400, 4-step 12B급 교사 | 4–12 |
| 10분 원테이크 대형 세계 | — | 50–300 |
| Powers of Ten (10⁵배 줌) | — | 60–240 |

solver는 데이터 엔진과 히어로 경로를 맡는다. 제품 추론은 student가 맡는다.

### 1.9 Student와 Expert Iteration (AlphaZero for Worlds)

- **Student:**
  - $p_\phi(W\mid y)$ = CWM(전역, 레벨 ≤ $l_g$) + LBO(Local Band Operator, 유계 반경 윈도우 attention, $K\le4$, Philox 노이즈)
- **학습 단계:**
  - **Stage A (현실 prior).** lift한 세계(DL3DV-10K, ScanNet++, SpatialVID 정적 구간, Sekai, MatrixCity, Infinigen 완전 장면)로 flow matching을 한다. 관측되지 않은 셀은 `unobserved`로 마스크한다.
  - **Stage B (WDM).**
    $$\nabla_\phi\mathcal L_{DM}=\mathbb E_{C\sim0.6\Pi_{iso}(l)+0.3\Pi_{cine}(y)+0.1\mathcal M,\ \theta\sim\hat p_T,\ e\sim p^{P1}_{edit},\ t}\Big[w_t\big(\hat x_0^{fake}-\hat x_0^{real,APG}\big)^\top\tfrac{\partial u}{\partial\phi}\Big],\quad u=\mathrm{Enc}\big(\mathrm{Cam}_\theta\mathcal R(e(G_\phi(\text{seed},y)),C)\big)$$
    - ISO 카메라는 $t\in[t_e(l),t_s(l)]$에서 뽑고, 기울기는 레벨 $l$로만 흘린다($Z_{<l}$은 stop-grad).
    - fake score는 교사 LoRA(rank 64–128)로 두고, TTUR 5:1로 학습한다.
    - $\mathcal L_{GAN}$의 real 집합: footprint를 맞춘 실사 패치, WSSC의 $W^\star$ 렌더, 약관이 허용하는 경우 닫힌 교사 샘플.
    - $e\sim p^{P1}_{edit}$: 광원 교체와 slot 제거를 30% 비율로 섞는다(Edit-Closure lite, §2.7).
    - minimax 에너지: $\mathcal M\leftarrow\text{top-}k_C\lVert\hat x_0^{fake}-\hat x_0^{real}\rVert$.
  - **Stage C (Render-RL / Directed).** 보상은 경로 영상에서 계산한다. 코스 레벨은 Flow-GRPO, 미세 레벨은 보상 가중 증류를 쓴다. **일관성 항은 넣지 않는다. 구조가 이미 보장하기 때문이다.**
- **래칫.**
  1. $W_0\sim p_\phi$
  2. 세계 공간 SDEdit으로 $\sigma_{mid}$에서 WSSC v3를 시작한다
  3. $W^\star$
  4. $(y,W^\star)$로 $\phi$를 학습한다
  5. 반복
  - 진척 지표: search gain $=\mathrm{DR_{rel}}(\text{student})-\mathrm{DR_{rel}}(\text{WSSC})$가 줄고, polish 반복 수가 줄어야 한다.

### 1.10 비회귀 조항 (Non-Regression Clause)

다음 세 조건을 만족하면 순차 hole-filling이 아니다.

1. **방문 순서 간선이 없다.** 의존 그래프는 스케일 DAG와 같은 레벨 안의 동기식 Jacobi 라운드뿐이다.
2. **어떤 영역도 다른 시점에서 먼저 생성된 렌더 픽셀을 조건으로 쓰지 않는다.** 조건은 더 거친 commit 레벨, WNF, $y$에서만 온다.
3. **completion은 스펙트럼적이다.** 기존 지지 영역 안에 고주파를 더할 뿐, 영역을 넓히지 않는다. 유일한 예외는 전역 1-pass bedrock이다.

**검증과 정직한 등급.**

- CI에서 `order_purity_check`를 돌린다. 수요 순서를 무작위로 8번 섞어 Merkle 루트를 비교한다.
- 지표 OIE $=\mathrm{LPIPS}(\mathcal R(W_{A\to B}),\mathcal R(W_{B\to A}))$.
- **native LBO는 비트 단위로 같다(T8).**
- **solver 모드는 OIE ≤ 0.05로 근사한다.** 나중에 확장되는 영역의 경계에서만 같은 레벨 이웃 apron(반경 r)이 Dirichlet 경계조건으로 들어간다. 이것은 반경이 유계인 세계 공간 조건이다. 오차 누적 깊이는 경로 길이가 아니라 $O(\log \text{스케일 범위})$에 비례한다.

---

## 2. 세계 매개체: STRATA v1

### 2.1 요구사항

| # | 요구 | 충족 수단 |
|---|---|---|
| R1 | 경로 독립 | 순수 레벨 귀납(T8), WNF |
| R2 | 스케일 완비와 무모순 | PSH 격자, filial 손실, dual-cut |
| R3 | 시점 의존 효과를 유도로 얻을 것 | RAS + RT 스페큘러, BRDF |
| R4 | 조명 분리 | Dual-Anchored Appearance, shot rig, Gaffer |
| R5 | 생성 친화적 텐서 | 정렬된 sparse octree + 잔차 latent |
| R6 | 편집과 소유권 | slot, Merkle, copy-on-write |
| R7 | 동역학 준비 | slot, ψ prior, rig, θ(identity) |
| R8 | 렌더 비용 $O(\text{px}\cdot\log\text{scale})$ | lens-gated demand, LOD |

### 2.2 자료구조

```
World (STRATA v1)
├─ meta      : seed u128, units=m, up=+Z, h_0=4096 m, L≈26, l_g (셀 1–8 m, l≈9–12),
│              y = {prompt_tree@scale, hero_stills, shot_intent | ∅}, mode∈{location,directed}, merkle_root
├─ lattice[l]: sparse hash (l,i,j,k:int32) → Cell                      # fVDB/NanoVDB
├─ vtx[l]    : CarrierVertex {ψ f32[2] (사이클), A f16}               # CARRIER가 있는 레벨만
├─ structs   : id → {type∈{CARRIER,CURVE,SYMBOL,POINT,GLINT}, level, inst, params}; cell→[id] 인덱스
├─ smf       : matter_id → {class, τ f16[32], instancer seed}         # Stochastic Matter Field
├─ iface[l]  : narrow-band SDF f16 (유전체 계면이 있는 레벨만)
├─ far_field : 구각 L=4–8, multi-band octahedral HDR + depth, 물리 하늘(Hosek–Wilkie/Prague) 파라미터
├─ objects   : inst → {class, prompt, xform, slot, locks, ψ_prior(μ,Σ), rig(P2), affordance, collision proxy}
├─ lights    : world_lights[] (sun/sky/practical IES/emissive) | shot_rigs[shot] (카메라 상대) | shot_dress[shot] (CoW)
├─ calib     : dreamer profiles {D(o,t), Σ_u, Σ_η(f), ρ(f,σ), Δ_P}
└─ caches    : UKA atoms (LRU), BVH per level, caption cache, NRC, GI probes
```

**Cell** (미세 레벨 기준 약 60–70 B)

| 필드 | 타입 | 의미 |
|---|---|---|
| `z` | int8/f16[d_l], d_l = 64/32/16 | VAR식 잔차 latent ($\mathrm{Up}(Z_{<l})$에 대한 잔차). identity $\theta$와 intrinsics가 여기서 디코드된다 |
| `occ`, `split` | u8, f16 | 자식 점유 마스크, 더 미세한 레벨이 존재할 확률 |
| `sem` | f16[8] | 의미 임베딩. 의미 효도의 대상 |
| `inst`, `slot`, `body` | u32, u16, u32 | 객체 소유, 생성 변수인 slot(사후 분할 아님), P2 바디 |
| `sref` | u32 | 참조하는 구조 리스트 오프셋 |
| `unc` | f16[2] | aleatoric, epistemic (교차 뷰 잔차 분산) |
| `own`, `auth` | u16, f16 | 대역 저술권(교사·뷰·생성기 id)과 소유 마진(`git blame`) |
| `b*`, `σ_i` | u8, f16 | 결정된 대역 인덱스, 원소별 노이즈 레벨(heterogeneous σ) |
| `state` | u8 | virtual / decoding / committed / locked |
| `hash` | u64 | Merkle 콘텐츠 해시. 캐시 키이자 버전 |

노이즈는 위치로 주소를 잡고(Philox), 캐시는 콘텐츠로 주소를 잡는다(Merkle). 결과적으로 세계에 git이 생긴다.

### 2.3 Unified Kernel Atom (UKA): 디코드 캐시

| 그룹 | 필드 | 크기 |
|---|---|---|
| 기하 | μ f16×3 (셀 상대), q packed 32 bit, s f16×3, 불투명도 f16, 커널 차수 n u8 (부드러운 볼륨 ↔ 단단한 표면), 편평도와 신장(s로 표현), cos 변조 캐리어 (k f16×2, 위상 f16) 선택 | ~32 B |
| 외관 | $L_d$ RGB9E5 (scene-linear HDR, world light 기준), albedo u8×3, normal oct16, $F_0$/roughness/aniso/metallic u8×4 | ~16 B |
| 잔차 | SG lobe ≤2 (예산 제한) | 0–16 B |
| payload | kind u8 + index: texel patch 8×8 int8 (FP의 surfel atlas), SGGX f16×6 (원거리 잎·털 aggregate), BCSDF f16×3 (strand), glint ref | 가변 |
| 메타 | own u16, unc u8, slot u16 | 5 B |

**예외는 세 가지뿐이다.**

1. 유전체 계면: `iface`의 narrow-band SDF, IOR, Abbe 수.
2. 시뮬레이션 볼륨: Hero FX의 불과 연기는 NanoVDB.
3. 아바타: SMPL-X에 UKA를 LBS로 바인딩.

정식 포맷은 **latent + 구조**이고, atom은 캐시다. 압축률은 디코드한 atom 대비 30–60배다.

### 2.4 Perceptual-Scale Hierarchy (PSH): 원소를 픽셀에 영향을 주는 가장 거친 레벨에 둔다 [신규]

$$
l^\star(e)=\min\big(l_{size},\,l_{vis},\,l_{coh}\big),\qquad
l_{vis}=\min\Big\{l:\tfrac{\lVert k_{h_l}*a_e\rVert_\infty}{k_{h_l}*\bar L}\ge\gamma_{vis}\approx0.02\Big\},\qquad
l_{coh}=\Big\lceil\log_2\tfrac{h_0}{\ell_{coh}}\Big\rceil
$$

- 확률적 텍스처: $V\sim(s/h)^{1\text{–}2}$. 크기 레벨에 그대로 남는다.
- 선(대비 $c$, 폭 $w$): $V\approx cw/h$. 하늘 앞의 2cm 케이블은 1m 셀까지 살아남는다. 그래서 2km 항공샷에도 존재한다.
- 점광원과 글린트: $V\approx cA/h^2$. 수십 레벨 위로 승격된다. 그래서 f/1.4 보케 볼이 진짜로 나온다.
- 결맞음 구조(벽돌 줄눈, 타일, 펜스, 직물): **위상** $\psi$는 $l_{coh}$에서 정하고, 디테일은 미세 레벨에서 정한다.

| 구조 | 파라미터 | 렌더 | 미세 레벨이 받는 조건 |
|---|---|---|---|
| CARRIER | 정점 격자 위의 $\psi_{1,2}$, 진폭 $A$, profile f16[16] | 국소 연산의 조건 | $u_j=\mathrm{frac}\,\psi_j$, 단위 인덱스 $\lfloor\psi_j\rfloor$, $\mathbf k=\nabla\psi$, 구조 주소 노이즈 $\xi_n$ |
| CURVE | 체인 Bézier/카테너리, 폭, $L_d$, 스펙 | 해석적 커버리지 $\min(1,w/\phi)$, CoC로 폭 확장 | 곡선 국소 좌표 |
| SYMBOL | baseline, glyph id u16[≤64], 폰트 임베딩, 잉크/발광 | MSDF 벡터 글리프 | 글리프 마스크 |
| POINT | 위치, HDR 강도, 로브 | 렌즈 처방 기반 조리개 스프라이트 | — |
| GLINT | 미세면 밀도, NDF, salt | 이산 미세면 글린트(Chermain'20, Deliot–Belcour'23) | — |

- 텍스트는 latent에서 금지한다. $\mathcal L_{ocr}$로 SYMBOL이 없는 곳에서 글자가 나오거나 문자열이 어긋나면 벌점을 준다.
- **Scale-Time Duality v2:** 원소의 레벨은 그 원소가 픽셀 위로 처음 떠오르는 스케일이고, 그것이 곧 디퓨전에서 결정되는 시점이며, 카메라 거리로는 $l^*(z)=\lceil\log_2(h_0f_{px}/\kappa z)\rceil$이다.

### 2.5 LOD: dual-cut 렌더와 효도 손실

$$
\lambda(p)=\log_2\frac{h_0}{\kappa\,\phi_{eff}(p)},\quad l_a=\lfloor\lambda\rfloor,\ f=\lambda-l_a,\qquad I(p)=(1-f)\,I^{(l_a)}(p)+f\,I^{(l_a+1)}(p)
$$

- $I^{(l)}$은 해당 영역을 레벨 $\min(l,\text{leaf})$의 **완전한** 세계로 렌더한 것이다. 레벨마다 VOL에서 SURF로 타입이 전환된다.
- $f<0.05$이거나 $f>0.95$이면 컷을 하나만 렌더한다. primary 비용은 최대 2배다.
- 효도 손실은 크로스디졸브 고스팅을 억제한다.
  $$\mathcal L_{fil}=\mathbb E_l\mathbb E_{C:\phi_{eff}=h_l/\kappa}\big[\lVert k_{pix}*I_C^{(l+1)}-k_{pix}*I_C^{(l)}\rVert_1+\lambda\,\mathrm{LPIPS}\big]$$
- 의미 효도: 자식 `sem`의 면적 가중 평균과 부모 `sem`의 코사인 유사도가 0.9 이상이어야 한다. 줌인했더니 돌멩이가 개구리로 바뀌는 일을 막는다.

### 2.6 외관: Dual-Anchored Appearance + 조명

$$
L_o(\mathbf x,\omega_o)=L_d(\mathbf x)+\underbrace{L_s^{RT}(F_0,\alpha,\mathbf n;L_W)}_{\text{세계 자신을 비추는 RT 스페큘러, 깊이 ≤2}}+\sum_k\Delta L_k,\qquad
\mathcal L_{int}=\big\lVert L_d-\mathrm{shade}(a,\mathbf n,r,m;L_{world},V)-g\big\rVert
$$

- **$L_d$가 정본이다(RAS).** Phase 1은 정지 세계이므로 구운 GI가 치팅이 아니라 정답이다. 그래서 Phase 1에서는 GI 경로추적을 하지 않는다.
- **intrinsics는 필수 채널이다.** 잔차 에너지 비율을 15% 이하로 두는 강제는 slot≠0인 객체와 그 그림자를 받는 면에만 건다(C5 판결).
- **귀속 제약** (Cine PIS 2.0): $\lVert r(x,\cdot)\rVert\le B_{mat}$. 무광은 0.5 JND, 반광택은 1–2 JND, 광택과 금속은 물리 로브 밖 잔차를 1 JND 이하로 둔다. 하이라이트는 곡률에서 예측한 specular flow를 따르도록 $\mathcal L_{specflow}$를 건다. 이렇게 해서 material-from-motion 오인(Doerschner'11)을 막는다.
- **shot_rigs:** 카메라 상대 좌표의 키, 필, 림, 바운스, 네거티브 필, 플래그.
  - 가산: $\Delta L_k=(\hat\rho/\pi+f_s)E_kV_k$
  - 감산: $L_d(1-\eta_k)$
  - 샷 안에서는 고정한다. 그림자가 미끄러지지 않는다.
- **재조명:** 기본은 비율 전송 $L_d\cdot\hat E_{new}/\hat E_{old}$. `unc`가 낮은 곳만 완전한 PBR로 승격한다.

### 2.7 Edit-Closure lite (Phase 2와의 계약) `P1-코어` → full은 `P1-확장`

- WDM과 polish 배치의 30%를 반사실 샘플로 채운다: 광원 교체, 히어로 slot 제거.
- 컵 밑의 숨은 면, 구운 그림자, 구운 조명이 기울기를 받는다. 컵을 제거한 렌더에서 드리울 물체 없는 그림자가 벌점을 받으므로 그림자가 visibility 항으로 밀려난다.
- full 버전(Phase 2 사건 prior 전체로 닫힘)은 §8.4에서 다룬다.

### 2.8 렌더러: 3DGRUT 하이브리드 하나

- primary: 3DGUT 래스터. Unscented Transform으로 왜곡 렌즈와 rolling shutter를 처리한다.
- secondary: 3DGRT. 반사와 굴절, 깊이 ≤2.
- 유전체는 SDF 레이마칭(분산 지원)으로, 볼륨은 SGGX/HG 단일 산란으로 처리한다.
- 공분산 대수(DOF와 MB 해석 근사):
  $$\Sigma'=J\Sigma J^\top+\Sigma_{pix}+\big(\tfrac{c(z)}4\big)^2I+\tfrac{\Delta t^2}{12}\dot{\mathbf m}\dot{\mathbf m}^\top,\quad \alpha'=\alpha\sqrt{\det\Sigma^0/\det\Sigma'}$$
  - $c(z)=\frac{f^2}{N(z_f-f)}\frac{\lvert z-z_f\rvert}{z}$
  - 깊이 불연속 × 블러 크기가 큰 타일(<10%)만 MC 16–64 spp. 검증 기준은 ColorVideoVDP ΔJOD < 0.25.
- 승격 구조: POINT와 GLINT는 조리개 스프라이트(cat-eye, 블레이드, onion-ring), CURVE는 해석적 커버리지, SYMBOL은 MSDF.
- **교사가 판정한 렌더와 최종 렌더가 같은 코드 경로를 쓴다.** 차이는 spp와 2차 반사 깊이뿐이므로 train/test gap은 0이다.
- 예산: 8K에서 GPU당 3–8 fps. 10초 1200프레임은 GPU 8장으로 수 분이다.

### 2.9 메모리 (Powers of Ten, 8K, 10초)

| 항목 | 규모 |
|---|---|
| 레벨 | 12–26 |
| 고유 셀 | 40–160M (int8 latent 1–4 GB) |
| 승격 구조 | 10⁵–10⁶ (수십 MB) |
| 프레임당 UKA | 10–30M (dual-cut 구간은 최대 2배) |

---

## 3. 가상 물리 카메라 + AI 촬영감독 (DP-0)

### 3.1 상태와 노출 적분

$$
s(t)=\big(p,R,f,d_f,N,\theta_{sh},\text{fps},\text{sensor},\kappa_a,k_{1..3},p_{1,2},t_{ro},EI,WB,d_{pupil}(f,d_f)\big),\qquad
\tau_{exp}=\frac{\theta_{sh}}{360^\circ}\cdot\frac{1}{\text{fps}}
$$

리그 시뮬은 1 kHz로 돌리고, 렌더는 서브프레임으로 샘플링한다. 모션블러는 Longuet-Higgins–Prazdny 장을 그대로 따른다. 평행이동 항은 $1/Z$에 비례하고 회전 항은 깊이와 무관하다.

예: S35 4K, 35mm, 0.5초에 360° 휩팬이면 화면 속도는 약 72,000 px/s이고 블러는 약 1,500 px다.

### 3.2 렌즈·셔터·센서와 "Camera Owns Photography" [조합]

- **렌즈.** 특허로 공개된 렌즈 처방을 레이트레이싱해 출사동 LUT를 만든다(Kolb'95). Hullin'11 고스트 플레어($g_k=c+\alpha_k(l-c)$, 광학 중심과 광원을 잇는 직선 위), FFT veiling glare, 회절 스타(짝수 N이면 N개, 홀수 N이면 2N개), 아나모픽(타원 보케, 가로 스트릭), breathing $f_{eff}=f(1+\beta m)$, 횡·종 색수차, $\cos^4$ 비네팅과 기계적 비네팅, 버티고 $f(t)=f_0Z_s(t)/Z_s(0)$.
- **센서와 필름.** $x=\mathrm{Poisson}(aL\tau g)/a+\mathcal N(0,\sigma_r^2)$, Bayer와 디모자이크, log 커브, LUT/2383 프린트, 할레이션, Newson'17 Boolean 그레인(프레임마다 seed), 게이트 위브.
- **DTC (Distill-Through-Camera).** 교사는 맨 세계를 보지 않는다. 언제나 $x=\mathrm{Cam}_\theta\mathcal R(W,c)$, $\theta\sim\hat p_T(\theta\mid c,y)$, $y(\theta)=\text{prompt}\oplus\text{lensTags}(\theta)$만 본다.
  - $\hat p_T$는 교사 샘플 수백 장으로 추정한다. 디포커스 맵과 단안 깊이로 thin-lens를 피팅해 (d_f, N)을 얻고, 휘도 히스토그램으로 EV와 톤을 얻는다.
  - T10에 따라 보케, 플레어, 노출, 비네팅은 세계 밖으로 빠진다.
  - **solver(§1.8)도 `FitThroughCamera`로 같은 원리를 쓴다.**
  - 실패 시 대체안: `<plate>` LoRA(sharp, deep-DOF 프레임 약 5만 장과 합성 2만 장, 속성별 negative CFG).
- **귀속 원칙(헌법 제1조).** 화면의 모든 변화는 카메라, 세계, 매체 중 정확히 하나에 물리적으로 귀속되어야 한다. 좋은 노이즈와 나쁜 노이즈를 가르는 것은 진폭이 아니라 귀속 가능성이다(GSC 지표).

### 3.3 Kinetic Camera: 질량 있는 카메라 [조합]

$$
M\dot v=u_{intent}(t)+F_{rig}(s)-Dv+\xi_{op}(t),\qquad \xi_{op}\sim p_{hand}\big(\omega_{res}^{200\text{Hz}}\mid\text{rig},\omega_n,\zeta,v_{intent},\text{profile}\big)
$$

- **의도 층.** min-jerk(Flash–Hogan), 2/3 power law(Lacquaniti'83), Fitts의 법칙, 지연 피드백 $u=K_pe(t-\delta)+K_d\dot e(t-\delta)$ ($\delta$ = 150–250 ms. Phase 2의 피사체 추종에 쓴다).
- **손 층** [신규 데이터원]. GoPro GPMF(gyro 약 200 Hz), Sony gyro 메타데이터, DJI 텔레메트리, Betaflight blackbox, Gyroflow 로그를 300시간 이상 모아 조건부 생성 모델(소형 DiT 또는 neural SDE)을 학습한다. 피로 모델(떨림 ∝ F), 결정적 순간의 숨 참기(호흡 성분 off)를 포함한다.
- **리그 라이브러리 14종** (식별 가능한 파라미터만 둔다).

| 리그 | 제약 다양체 (지문) | 동역학 | 지각 시그니처 |
|---|---|---|---|
| 트라이포드 + 플루이드 헤드 | 순수 회전 + 입사동 호 (r≈0.1 m) | 점성 드래그, 정지마찰 | ease-in/out, 출발 순간의 "턱", 팬 중의 미세 시차 |
| 돌리 / 슬라이더 | 수평 스플라인, σ_h < 5 mm | 그립, 이음매 v/L | 완벽한 측면 시차, ≤1.5 m/s |
| 지브 / 크레인 / 테크노크레인 | 반경 L 구면 (+L(t)) | 진자 ω_n, 오버슛 1–3 cm | 진자처럼 안착 |
| 스테디캠 | 높이 일정, f_gait/2 측면 sway | drop time 1.5–2.5 s | 떠 있지만 무게가 있다 |
| 3축 짐벌 | 회전은 완벽 + 수직 보빙 | PID 대역폭 | 닌자 워크 |
| 숄더 핸드헬드 | 보행 수직 피크 1.6–2.2 Hz | IMU 손 모델 | 오버슛 후 교정 |
| 시네 드론 | 고도 큼, yaw 지연, **카메라 수평 유지** | 기체 기울기는 화면에 보이지 않음 | 큰 시차, 바람 드리프트 |
| FPV | 롤과 횡가속 결합 $\tan\phi=a_{lat}/g$ | 업틸트 20–40° | 기울어진 수평선이 곧 속도감, 젤로 |
| 케이블캠 / 카마운트 / 모션컨트롤 / 수중 | 현수선 / ISO 8608 노면 PSD / 재현성 완벽 / 부력+항력 | — | 중계 흔들림 / 노면 질감 / 지나친 완벽 / 합법적으로 떠다님 |

### 3.4 ShotDNA v2: 식별 가능성 분할 [신규]

| 층 | 대역 | 데이터 | 방법 |
|---|---|---|---|
| I (의도) | ≤1 Hz | 영화·다큐 SfM(MegaSaM/VGGT-Ω/DA3), DataDoP 29K, CameraBench taxonomy | Toric 키프레임 + min-jerk |
| R (리그) | 기하 | 합성 + IMU 라벨 | 홀로노믹 지문 → (클래스, ω_n, ζ, 한계) |
| H (손) | 0.5–100+ Hz | IMU 코퍼스 | $p_{hand}$ |

추출기 전달함수 $\mathrm{PSD}_{SfM}=\lvert H\rvert^2\mathrm{PSD}_{gyro}+N$을 측정한 뒤, 영화에서는 식별 가능한 대역만 쓴다.

### 3.5 DP-0 v2: prior + 탐색 + 카메라 전용 선호 모델 [조합/신규]

- **입력.**
  - 세계 토큰: scene graph(클래스, 3D bbox, saliency), 64³ 점유, 광원, 파노라마 임베딩
  - 텍스트, $z_{style}\in\mathbb R^{256}$, 음악 비트, 편집 문맥(앞 샷 ShotDNA, 액션 축), PIS risk 맵
- **Prior.** 키프레임 토큰 위의 DiT. 연속값은 flow matching, 이산값은 masked prediction이다.
- **탐색.** CEM/MCTS. 루프 안에서 Toric 솔버, 1 kHz 리그·손 시뮬, 720p 프리뷰(초당 100프레임 이상)를 돌리고 후보 $10^3$–$10^4$개를 채점한다.
- **보상.**
  $$R=\mathrm{CRM}+\beta_1\mathrm{PR}+\beta_2\mathrm{Wipe}+\beta_3\mathrm{RevealBeat}+\beta_4\mathrm{Clearance}+\beta_5\log p_{intent}-\beta_6\mathrm{CWS}-\beta_7\mathrm{Risk_{PIS}}-\infty\cdot\mathrm{collision}$$
  - PR(Parallax Richness): saliency 가중 깊이 층별 각속도 분포의 엔트로피.
  - Wipe: 전경 가림체가 saliency 픽셀을 X% 이상 새로 드러내는 이벤트.
  - Clearance: FPV 틈새를 지날 때의 time-to-contact 마진.
  - CWS: 카메라 jerk PSD와 영화 분포 사이의 W₁.
- **CRM (Camera Reward Model)** [신규]. 세계를 고정하고 렌더가 결정론적이므로, 경로만 다른 쌍은 차이가 오직 카메라에서 나온다.
  - 데이터: 세계 120–200개 × 경로 40–50개, 720p 8초 프리뷰, 일반 1.5–2만 쌍과 전문가 2천 쌍.
  - Bradley–Terry로 학습하고 전문가 DPO를 더한다.
  - 데일리 모드에서 감독이 고른 것이 그대로 능동학습 데이터가 된다.
- **동기 플래너.**
  $$N(t)=\sum_o\mathrm{sal}_o[\tfrac{d}{dt}\mathrm{vis}_o]_++\eta\sum 1[\text{역광·플레어}](t)+\zeta\,\mathrm{Wipe}(t),\qquad R_{rhythm}=\mathrm{corr}(\mathrm{peaks}(N),\text{beats})-\mu_1\text{overload}-\mu_2\text{boredom}$$
  멈춘 스펙터클의 정점이 음악의 드롭과 같은 순간에 드러나도록 한다.
- **Gaffer.** $L^*=\arg\max_L\mathbb E_t[\mathrm{Aes}(x_t)-\lambda\mathrm{DR_{rel}}(x_t)]$. 조명 ≤8개, 파라미터 10–50개, 샷 안에서는 상수다.
- **Shot dress** (directed 모드). $l\le l_g+2$에서 샷별로 소품 배치를 바꾼다(copy-on-write). 한 샷 안의 연속성은 정확히 유지된다.
- **PIS 2.0 결합** `P1-확장`.
  - $\Omega_x=\lVert v_\perp+\omega\times d_{pupil}\rVert/Z$
  - 망막 속도 $v_R=v_{screen}-\mathrm{SPEM}(v_{target})$, 여기서 SPEM은 Daly 모델 $v_E=\min(0.82v_T+0.15°/s,80°/s)$
  - ColorVideoVDP로 risk 맵을 만든다. risk가 높은 영역에는 solver 뷰, 이른 $t_{own}$, LOD+1을 더 배정한다. risk는 DP-0 비용항으로도 들어간다.
  - 정리하면 **세계는 카메라에게 어디가 안전한지 알려 주고, 카메라는 세계에게 어디서 정직해야 하는지 알려 준다.**

### 3.6 ShotScript (카메라 DSL)

```
world  "hanok_first_snow" @merkle:3f9a…   mode=directed
format 6K 2.39:1 @24fps, shutter 180°, ACEScct → print_2383
camera A: rig steadicam(sled=18kg)  hand imu_profile("calm_veteran", seed=42)
          lens anamorphic(40mm, T2.3, 2.0x)  focus track(@girl.eye_left, puller_lag=0.25s)
fx "snow_frozen": sim=particles(snow, wind=1.2m/s, seed=7) freeze t*=3.2s
shot "first_snow_reveal" 0s..24s:
  lighting cheat add rect(1.2m, @girl+back(2m)+up(1.5m), 5600K, +1.5EV)
  0s frame @girl at (1/3,0.42) size 0.35 ; 0-6s orbit @girl 90° keep_height
  10s step_on crane(arm=7m) ; 16s rise 6m tilt_down ; 20s reveal @snowflakes_frozen on beat(3) ; 24s hold breath_hold
constraints: axis_of_action(@girl,@grandma) keep_side=left; no_collision; max_speed 1.8 m/s; horizon ±0.5°
budget: demand=ray_bundle, consistency=auto(PIS)
```

**컴파일 체인:** ShotScript → Toric → 의도 스플라인 → 1 kHz 리그·손 → Camera State Stream(USD 확장) → `demand_register` → (필요하면) 세계 정제 → 렌더.

**인터페이스:** 자연어, DSL, 폰 VCam(ARKit, 1:10 스케일), FPV 조종기(Betaflight rates), 타임라인 UI, USD/FBX/Alembic/Unreal/모션컨트롤 로봇 export.

---

## 4. Phase 1 전체 파이프라인

### 4.1 단계별 입력·출력·모델

| 단계 | 입력 | 출력 | 모델/알고리즘 | 비용 (방 규모, 4K 30s) |
|---|---|---|---|---|
| S0 World Program | 프롬프트/스크립트 | prompt_tree@scale, scene graph(slot, metric bbox, 재질 클래스), matter class, light intent, shot intent | LLM (Claude/GPT급 또는 Qwen3) | 초 단위 |
| S1 Art Direction (선택) | S0 | hero still 1–4장 + lensTags + 팔레트 | 닫힌 SOTA(약관 확인) 또는 오픈 교사 | 분 단위 |
| S2 Bedrock | S0, S1 | 레벨 ≤ $l_g$: 구조, 코스 latent, PSH 구조, far field, world lights, slot, intrinsics. **내부와 뒷면 포함** | CWM 전역 1-pass, hero는 DPS 관측 제약. 부트스트랩: 파노라마+깊이 + 다시점 joint pass + DA3/VGGT-Ω/AnySplat 재구성 + 코스 WDM. **코스 레벨은 T5에 따라 평균해도 안전하다**($P_n/(P_s+P_n)$이 작음) | 0.1–1 H100-h |
| S3 Demand | DP-0 경로 분포, 렌즈 계획 | {(level, block)}, 완전성 집합 | Ray-Bundle Demand (§4.3) | 초 |
| S4 Strata | S2, S3 | commit된 미세 레벨 | **solver:** WSSC v3 (§1.8) / **native:** LBO few-step | 4–12 H100-h / ~1 H100-min |
| S5 Hero FX (선택) | 스펙터클 지시 | Dielectric/Volume/Curves/Avatar 프리미티브 | t*에서 멈춘 FLIP/SPH, 파쇄, 연소, MPM 시뮬 + CEM 아트 디렉팅(DTC 점수) | 1–4 H100-h |
| S6 Audit/Polish | W | $\sup_\Pi\mathrm{DR_{rel}}\le1.2$ | 고정점 반복 + adversarial camera | 1–3 H100-h |
| S7 Shoot Planning | W, intent | 최종 경로 Γ, Gaffer 조명, shot dress | DP-0 탐색 + CRM | 0.2 H100-h |
| S8 Render | W, Γ | scene-linear HDR 프레임 + AOV | 3DGRUT, dual-cut, RAS, 공분산 블러 | 수 분 (8 GPU) |
| S9 Lens/Sensor/Look | HDR 프레임 | ProRes/EXR, 멀티레이어 AOV(깊이, 법선, cryptomatte) | 물리 렌즈, 센서, 필름 | 수 분 |

### 4.2 "렌더 시점에는 어떤 픽셀도 새로 환각하지 않는다": 성립 범위

**불변식 I6 (Read-Only Camera).** 렌더 경로 $\mathcal S\circ\mathrm{Cam}\circ\mathcal R$은 $W$에 대한 결정론적 함수다. 프레임 단위 latent가 없고, $W$에 쓰는 연산도 없다.

| 요소 | 성립 여부 | 이유 / 한정 |
|---|---|---|
| 기하, 텍스처, 재질, 샷 내 조명 | **엄밀히 성립** | T1, T2 |
| 렌즈, 셔터, 센서 효과 | **엄밀히 성립** | 물리 연산자 |
| WDD(L4 극미세) | **성립(지연 평가)** | 렌더 중에 처음 평가될 수는 있다. 하지만 값은 $(\text{seed},\text{context},\text{scale})$의 순수 함수이고 메모이즈된다. 경로나 프레임과 무관하므로 "읽을 때 한 번 생성"이지 프레임마다 생성하는 것이 아니다 |
| 수요 확장 (경로가 여유분 밖으로 나갈 때) | **성립(세계 공간)** | 새 콘텐츠가 생기긴 한다. 하지만 세계 공간에서, 더 거친 레벨과 WNF만 조건으로 생긴다. 렌더 픽셀을 조건으로 쓰지 않는다. native는 비트 동일, solver는 OIE ≤ 0.05 |
| 신경 셰이딩 잔차(선택; Phase 2 MAN) | **조건부 성립** | 학습된 결정론적 함수이고, 출력은 $\lVert r\rVert\le\tau$로 제한되며 알파·기하 채널이 없다. 콘텐츠 자유도는 0이지만 미세 셰이딩은 학습된 함수가 정한다 |
| 필름 그레인, 센서 노이즈 | **의도적으로 불성립** | 매체 층이다. 실제 카메라도 같다. 왕복 테스트에서는 끄거나 포즈에 고정한다 |
| S²R/WSSC polish | 렌더 경로가 아님 | 교사는 세계에만 쓰고 프레임에는 쓰지 않는다 |

### 4.3 Lens-Gated Ray-Bundle Demand [조합]

$$
\mathrm{Cov}=\bigcup_t\bigcup_{\xi\in\mathcal A}\bigcup_{\tau\in[0,\tau_{sh}]}\mathrm{Vis}\big(\mathrm{ray}(s(t+\tau),\mathbf u,\xi)\big),\quad
p_{eff}=\max(p_{px},c_{px},b_{px}),\quad
\ell^*(x)=\Big\lceil\log_2\frac{\max_t f_{px}\cos\theta/Z\cdot\mathrm{vis}\cdot p_{px}/p_{eff}}{D_0}\Big\rceil+1
$$

- Cov 전체는 레벨 $\ell_{cov}$ 이상으로 존재해야 한다. 전경 보케 가장자리 너머(partial occlusion)와 휩팬의 frustum 밖도 포함한다.
- 50mm T1.4, 초점 2m, 배경 10m이면 CoC는 약 120 px이고 **약 7옥타브가 생략**된다. 휩팬 스트릭이 1,500 px이면 약 10옥타브가 생략된다. 보케가 생성비를 할인해 주는 셈이다.
- **면제 규칙:** 승격 구조(POINT, GLINT 상위 k, CURVE, SYMBOL)는 게이팅에서 빠진다. 그래서 보케 볼과 가는 선이 진짜로 남는다.

---

## 5. 세 고질병: 원인, 해법, 보장

통일 진단(FP): 아직 결정하지 않은 변수(위치, 정체, 위상, 타이밍)를 **틀린 공간(픽셀)에서 평균 내면** off-manifold 출력이 나온다.
통일 처방: "Decide before you draw." 결정 변수를 명시적으로 만들고, 원래 공간에서 이산적으로 결정한 뒤, 결정론적으로 렌더한다.

| 고질병 | 원인 (근거) | 해법 | 보장 수준 |
|---|---|---|---|
| **(1) 일관성 붕괴** | 상태 변수가 없다. 세계가 attention 컨텍스트 안에만 있다. 조건부 그래프가 path/band 형태라 drift가 선형으로 쌓인다(T3). GeCo: Sora 2, Veo 3.1의 정적 장면 오차가 실사 바닥 0.02의 3–10배 | $W$를 상태 변수로 두고 star graph로 만든다. 렌더는 결정론적이다 | **정리(T1, T2, T3).** 남는 것은 렌더러 앨리어싱뿐이고, Mip 필터로 처리하는 신호처리 문제다. 단, 생성 단계의 다시점 불일치는 세계 하나로 흡수되어 "일관되지만 틀린" 디테일로 남을 수 있다. 이것은 일관성 문제가 아니라 품질 문제이고, (2)에서 다룬다 |
| **(2) 뭉개짐·녹아내림** | (a) 위치 hedging: $\mathbb E_u[I(\cdot-u)]=I*p_u$ (blur kernel = 위치 불확실성). (b) posterior mean에서 위상이 식별되지 않으면 진폭이 수축한다: $B(\omega)\approx A^2(1-\rho)$. (c) 프레임당 연산 예산 부족. (d) 3D 쪽 원인: 다시점 평균 재구성(T5, CAT3D/Lyra 블러) | 렌더 시점에는 평균할 위치가 없다(결정론적). 생성 시점에는 WNF로 ρ↑, 대역 분할 소유권(T6), explain-away, SMF, PSH 승격. 디테일 예산은 dedup으로 10–25배(T4) | **렌더 단계는 정리(시간 축 블러 0).** **생성 단계는 구조적 설계 + 실험 목표.** 지표: SFR_prior ≥ 0.9, HER ≥ 0.8, DR_rel ≤ 1.2. T6은 소유 영역 안에서 정리이고, seam은 2AFC로 검증한다 |
| **(3) 과한 부드러움** | (a) 지터 성분의 조건부 평균 회귀("핸드헬드 1000개의 평균은 짐벌 샷"). (b) 미감 데이터 필터. (c) 가설 H-ν: 시간 4배 압축 VAE(24fps → latent 6 Hz)의 디코더 보간. (d) 꿈결 같은 parallax: 강체 flow 모델로 설명되지 않는 잔차 | 카메라는 질량, 리그, IMU로 배운 손을 갖는다. 위상 샘플 하나를 **실제로 뽑는다.** parallax는 T1에 따라 정의상 강체다. Phase 2에서는 Event-first로 타이밍을 먼저 샘플한다($\mathbb E[H(t-\tau)]=F_\tau$ 문제를 제거) | **카메라 parallax는 정리.** 손떨림 스펙트럼은 실험 목표: D_PSD, CWS가 영화 분포 안, 카메라 튜링 2AFC ≤ 55%. H-ν는 X0에서 원인별로 분해 측정한다 |

**보너스(Cine의 가짜 냄새 16종 중 구조적으로 사라지는 것).** 텍스처 수영(TS), 속도와 무관한 블러(BVC), 깊이와 무관한 보케(DDC), 떠다니는 플레어(FLT), 미끄러지는 그림자, 텍스트 끓음(TPR), 그레인 끓음(GSC), 재질 오인(MAT).

---

## 6. 순차 아웃페인팅 및 선행작과의 차이

### 6.1 순차 아웃페인팅 vs Block Universe

| 축 | 순차 아웃페인팅·인페인팅 (거부) | 비디오 디퓨전 | **BUC** |
|---|---|---|---|
| 생성 순서의 축 | 방문 순서(시간) | 프레임(시간) | **스케일 × 노이즈 레벨** |
| 상태 | 2D 캔버스(과거 픽셀) | 컨텍스트 창 | **세계 공간 diffusion 상태 $W_\sigma$** |
| 노이즈 | 프레임마다 새로 | 프레임마다 새로 | **세계 표면 위의 단일 장 (WNF)** |
| 동결 기준 | 먼저 그린 픽셀은 영원히 확정 | — | **대역**: 한 주파수 대역이 모든 영역에서 동시에 확정 |
| 정보 흐름 | 인과적(뒤가 앞을 못 바꿈) | 창 안에서만 | 결정 대역 안에서 양방향, 뒷면 뷰가 앞면 레이아웃에 거부권 |
| 그래프 / $R_{\text{eff}}$ | path → $\Theta(n)$ | band → $\Theta(n)$ | star → $O(1)$ |
| 오류 누적 | 경로 길이에 비례 | 프레임 수에 비례 | $O(\log\text{스케일})$ ≈ 15–25, 고정 |
| 순서 의존성 | OIE ≫ 0 | — | OIE = 0 (native), ≤ 0.05 (solver) |
| 구멍 | 매 스텝 생김 | 매 프레임 | 없음 (모든 곳이 어떤 대역으로 이미 존재) |

### 6.2 가장 가까운 선행작

| 선행 | 하는 일 | 상한·한계 | 우리가 가져오는 것 | 진짜 다른 점 |
|---|---|---|---|---|
| World Labs Marble 1.1 (+Studio) | splat 세계 생성, 경로, export | SPZ 2M 상한, 구운 조명, room-scale, 정적 | 제품 형태 | 다중 스케일 PSH, RT 반사, 이미지 교사 대역 판정, 물리 카메라 + DP-0 |
| World Labs Atlas / Genie 3 / GWM-1 / WorldPlay | 프레임 생성형 월드 모델 | 명시적 상태가 없어 기억 지평에 한계, 결정론 없음 | — | 업계와 반대 방향의 베팅. Atlas 피벗의 원인 넷(splat 상한, 구운 조명, 비디오 교사 상한, 촬영지 수준 미감)을 각각 PSH, RAS+RT+intrinsics, 이미지 교사, DTC+Gaffer+DP-0으로 겨냥한다 |
| NVIDIA Lyra 2.0 | Wan2.1 비디오 → DA3 3DGS | 832×480, FID 43–51, 비디오의 블러와 노출 불일치 상속, 연구 전용 라이선스 | 메모리·재구성 부품 | 교사가 이미지 모델이고, 세계가 증류물이 아니라 생성 변수다 |
| HY-World 2.0/2.1 | 파노라마 → 궤적 확장 → 합성 | 순차 확장(거부된 패러다임에 가까움), 80B+17B 연쇄 | 파노라마 원경 | 전역 bedrock + lockstep, 순서 불변 |
| FlashWorld | MV 모드 → 3D 모드 DMD2 | 비디오 교사(Wan2.2-5B), 480p, pixel-aligned로 고정 밀도 | dual-mode DMD2 설계 | 이미지 교사가 band-isometric 뷰에서 대역별로 판정. 유효 해상도 ≈ 교사 해상도 × $2^{L-l_g}$ |
| CAT3D / Bolt3D | 다시점 생성 → 재구성 | 평균에 따른 블러(T5), 512급 | 다시점 joint 생성(bedrock 부트스트랩) | 매 스텝 lockstep 동기화 + 분할 소유권 + WNF |
| SEVA | 임의 카메라 NVS, 30초 루프 | 매번 프레임을 그림, flicker | — | 세계가 있으므로 프레임을 생성하지 않는다 |
| GEN3C / Voyager | 3D 캐시 → 비디오 디퓨전 출력 | 최종 픽셀을 비디오 모델이 그림, 540–720p | — | 최종 픽셀은 결정론적 렌더 |
| Director3D / GenDoP | 궤적 + Gaussian, 궤적 생성 | 연출 의도가 약하고 해상도 낮음 / 물리 없음 | Traj-DiT, DataDoP | 의도·리그·손 분해, 세계 안에서 탐색, CRM |
| Difix3D+ | 1-step 이미지 디퓨전으로 렌더 보정 | 노이즈가 세계에 묶이지 않음 | 1-step enhancer | 결과를 세계에 재증류(solver). Phase 2에서는 물질 고정 노이즈(MAN) |
| SyncTweedies / MultiDiffusion / DMV3D / 3D-Adapter | 공통 공간 동기화 | 평균 기반 | 동기화 뼈대 | 평균이 아니라 σ-대역 소유권, 보정된 likelihood, band-isometric |
| Generative Powers of Ten / VAR / XCube / LT3SD / TRELLIS | 다중 스케일, 계층 latent | 2D 또는 객체·실내 한정 | 계층 latent | 미터 단위 레벨 + 카메라 footprint + 외관 공간 효도 + 지연 평가 + PSH |

---

## 7. Phase 1 승리 조건

### 7.1 FREEZE-TAG Bench v1 (얼음땡 벤치)

> 세계가 "얼음!"을 외치면 카메라만 움직일 수 있다. 그 밖에 무엇이든 움직이면 "땡"이다. v1에서는 규칙이 하나 더 붙는다. "예쁘기까지 해야 얼음이다."

**프레이밍.** 정지 세계는 정답이 정해진 비디오 평가다(세계의 움직임 = 0). 이 아이디어의 선례는 GeCo이고, 우리는 GeCo를 **MI(Motion Integrity)의 코어로 빌린다.** 우리가 더하는 것은 다음과 같다.

- SP×MI 빈 사분면 Pareto 프레이밍
- 양방향 경로 이식 T4±
- 광학 포렌식
- MAT, BLI, OIE

**프롬프트.**

- 세계 6범주 × 40개 = 240개: 실내+텍스트, 건축, 자연, 시간정지 스펙터클, 재질 스트레스(거울, 유리, 크롬, 수면, 털), 스케일 극단
- 카메라 의도 12종: push-in, pull-out, 측면 트럭, 360° 오빗, 크레인 업 리빌, FPV 다이브, 핸드헬드 워크, 왕복, 휩팬, 돌리 줌, 랙 포커스, 60초 이상 배회
- 표현 3종 앙상블: "static scene, camera moves" / "bullet time, frozen moment" / "everything perfectly still". 경쟁 모델에게는 GeCo 기준으로 셋 중 최선을 준다
- 위를 조합해 약 500쌍. 여기에 bullet-time 사람 8개를 더한다
- **자해 카테고리**(불, 연기, 커스틱, 털, 반투명)를 일부러 넣고 결과도 공개한다
- **차트 프롬프트**: 세계 안에 ISO 12233과 Siemens star 차트를 둔다

**트랙.**

| 트랙 | 정의 | 목적 |
|---|---|---|
| T0 Oracle | 실제 촬영한 정적 장면 + 우리 카메라 스택 vs 같은 첫 프레임의 경쟁 I2V | 상한선 측정. 세계 팀 목표 = "오라클 점수의 X% 이내" |
| T1 짧은 T2V | 경쟁 모델의 홈그라운드. 고시차, 고주파 스트레스 | 정면 승부 |
| T2 같은 첫 프레임 I2V | 출발선을 공정하게 | |
| T3 롱폼 | 60초, 180초. 경쟁 모델은 공식 extend 사용 | 구조적 우위 |
| **T4± 양방향 경로 이식** | (+) 경쟁 영상에서 경로를 추출(GeCo 가중 robust BA)해 같은 첫 프레임의 우리 세계에서 렌더. (−) DP-0 경로를 경쟁 모델의 카메라 제어(Kling 3.0 motion control camera path, Veo/Flow)에 입력. 추출 편향은 우리 렌더(정답 경로를 앎)로 보정 | 지금까지 없던 수준으로 깨끗한 A/B |
| T5 카메라 튜링 | 같은 세계에서 실제 gyro 재생 / DP-0 / 경쟁 경로를 2AFC | 이미지 품질과 분리한 카메라 지능 |
| T6 스테레오/VR | 네이티브 스테레오 vs 최선의 스테레오 변환 | BLI, presence, SSQ |

**경쟁 대상.**

- 비디오: Wan 3.0, Seedance 2.5, MiniMax H3, Kling 3.0 Omni, Veo 3.1, Gemini Omni 1.1 Flash
- 세계: Marble 1.1(Studio), Atlas(접근 가능하면), Lyra 2.0(평가 전용), HY-World 2.1, FlashWorld, Voyager, GEN3C, Project Genie

**지표.**

| 축 | 지표 | 정의 |
|---|---|---|
| 미감 | **SP / TP** | 무작위 시점 단일 프레임의 블라인드 쌍대 비교. 경쟁 프레임 대비(SP), 교사 자신의 스틸 대비(TP). Bradley–Terry, 95% CI |
| 일관성 | **GeCo fused** | $\sqrt{M e_m^2+e_s^2}$ (UFM flow vs VGGT rigid flow). 실사 바닥 0.02 |
| | MEt3R, LCE/PAL, Revisit(SCoPE, R2M 상대 비교) | 다시점 특징, 왕복 LPIPS, 재방문 |
| | FV | 비강체 잔차가 1 px를 넘는 track 비율 |
| 디테일 | BS(ℓ,Δ) 끓음 스펙트럼, 청크 지문(주기 4/8 피크), TPI | 강체 워프 후 Laplacian 밴드 잔차, VAE 청크 깜빡임, 고역 시간 누설 |
| | SFR_real(f), SFR_prior(f), DR_rel, HER | 실사 대비 radial power, 교사 꿈 대비, dream residual, 고역 보존율 |
| | MTF50 | 세계 안의 ISO 12233. 네이티브 8K vs 업스케일 |
| | TPR, IDP, 효도 줌 점수 | 텍스트 영속(편집거리 < 0.1), ArcFace, $\mathrm{LPIPS}(\downarrow_kI_{close},\mathrm{crop}(I_{far}))$ |
| 광학 | BVC, DDC, FLT, BLI | 블러-속도 $R^2$, 디포커스-깊이 $R^2$, 플레어 직선 이탈각, 양안 광택 |
| 지각 | MAT, GSC, CI | 무광 재질 귀속, 그레인-구조 상관, cardboard 비율 |
| 카메라 | D_PSD, CWS, FTD, 질량 시그니처 | gyro 보정 후 각속도 PSD $W_1$, jerk PSD, 궤적 임베딩 FD, 오버슛 분포 |
| 정직성 | OIE | 세계 생성기 순서 불변성 |

**사람 평가와 공정성.**

- 일반 풀과 전문가 풀(촬영감독, 컬러리스트, VFX 슈퍼바이저 50–100명)을 따로 둔다. 쌍마다 3명 이상, 주의력 체크를 넣는다. 일반 조건과 **reveal 조건**(시작·끝 프레임을 나란히 제시)을 나눈다.
- 같은 프롬프트 문장을 쓰고, 경쟁 모델에게는 자기 도구를 최대한 허용한다. seed 4개를 뽑아 무작위 결과와 블라인드 best-of-4를 둘 다 보고한다. 1080p와 같은 코덱으로 정규화하고, 경쟁 영상에 프레임 보간을 하지 않는다.
- **사전 등록**, 제3자 학계 파트너, 원본 출력·프롬프트·코드 공개(`pip install freezetag-bench`), 서비스 약관 점검.

**Phase 1 승리 조건** (최종 목표이고 낮추지 않는다).

| 항목 | 목표 |
|---|---|
| **빈 사분면** | SP ≥ 50% (vs Wan 3.0, Seedance 2.5) **이면서** GeCo ≤ 0.03 |
| TP (교사 스틸 대비) | ≥ 45% |
| VP T1 일반 / 전문가 | ≥ 65% / ≥ 75% |
| VP T2, T4± | ≥ 80% |
| VP T3 롱폼 | ≥ 90% |
| LCE | 정확히 대칭인 경로: 비트 동일(PSNR = ∞, grain off). 근사 복귀: LPIPS ≤ 0.01 |
| TPR / MAT / CI | ≥ 0.99 / ≥ 90% / ≤ 실사 + 5%p |
| SFR_prior / DR_rel / TPI | 전 대역 ≥ 0.9 / ≤ 1.2 / 실사 바닥의 1.5배 이내 |
| MTF50 @8K | ≥ 0.25 cy/px |
| 카메라 튜링 | 2AFC ≤ 55% |

### 7.2 쇼케이스 Top 10

| # | 데모 | 증명하는 것 | 비디오 모델이 구조적으로 못하는 이유 | 바이럴 포인트 | 배치 |
|---|---|---|---|---|---|
| 1 | **왕복 "들어갔다 나와"**: 정방향·역방향 차이 맵 | T1, T2 | 돌아오는 길에 세계를 다시 상상한다 | 경쟁 모델의 차이 맵은 크리스마스 트리처럼 빛나고 우리는 새까맣다. #왕복챌린지로 프롬프트를 공개한다 | P1-코어 |
| 2 | **1999 → 2026 불릿타임 헌정**: 와인잔이 깨지는 순간 정지, 파편과 와인 왕관 사이 360° 오빗 | 정지 세계 + Hero FX + 광학 | 정지를 요청하면 숨을 쉰다(H3, GeCo의 지구본) | 매트릭스 오마주, 음악 드롭에 맞춘 reveal, 가상 스틸카메라 120대 링을 보여 주는 메이킹 | P1-확장 |
| 3 | **을지로 네온 한글 간판 거리**: f/1.4 랙 포커스, 보케 볼, 플레어 직선, 웅덩이 반사 | SYMBOL, POINT, DDC, FLT, HDR | 글자가 끓고, 보케가 깊이와 무관하며, 플레어가 떠다닌다 | "최소한 일관되게 틀린다", 플레어 직선 오버레이 | P1-코어 |
| 4 | **거울과 유리 복도**: 카메라 뒤에 있는 것까지 비친다 | RT 반사, 굴절 | 거울 속 세계를 매번 다시 상상한다 | "거울은 거짓말을 못 한다" | P1-코어 |
| 5 | **라쇼몽 × 리그 스왑**: 같은 세계를 6개 리그와 50개 스타일로 | 세계와 카메라의 분리, Kinetic Camera | 테이크마다 다른 세계가 나온다 | 6분할 화면에 "어느 게 제일 무서워 보여?" 투표 | P1-코어 |
| 6 | **경로 이식 릴 T4±**: Kling이 고른 카메라 무브를 우리 세계에서 | 공정한 A/B | — | 같은 무브, 다른 세계를 좌우로 나란히 | P1-코어 |
| 7 | **체호프의 컵 / AI 러시안 아크**: 10분(목표 90분) 원테이크, 5분 뒤 컵의 립스틱 자국이 그대로 | 장시간 drift 0 | 컨텍스트 길이, extend할 때마다 쌓이는 drift | 숨은그림찾기 챌린지, 유튜브 프리미어 라이브 | P1-확장 |
| 8 | **Powers of Ten**: 2km 항공에서 무당벌레까지 10⁵배 연속 줌 | PSH, 효도, 의미 줌 | 스케일 전환에서 녹는다 | Eames 오마주 원테이크 | P1-확장 (스트레치) |
| 9 | **빛만 흐르는 세계 + 매직아워 무한 연장**: 멈춘 파도 위로 해가 지고 등대 빛이 훑는다 | 재조명 가능한 세계 | 조명 상태가 없다 | 촬영감독 리액션 영상 | P1-확장 |
| 10 | **직접 날아보기**: 웹/VR 실시간 뷰어, FPV 파일럿 챌린지, Vision Pro 양안 광택 | 상호작용, BLI | 상호작용 자체가 불가능하다 | 모든 영상 밑에 "믿지 마세요. 직접 날아보세요." 링크, SIGGRAPH RTL과 BIFF VR 부스 | P1-확장 |

---

## 8. Phase 2: 움직이는 세계

### 8.1 정식화 확장과 불변량

$$
W_{canon}\sim p_\omega(W\mid c)\ \text{(edit-closed)},\quad (E,z)\sim p_\theta(E,z\mid W_{canon},c),\quad s=\mathrm{Realize}(E,z,W_{canon};\Delta s_0,\Delta\psi,\Delta\text{layout},a,\iota)
$$
$$
W(t)=\mathrm{Deform}(W_{canon},s(t)),\qquad A_i(t)=\mathcal A_\Theta\big(\theta_i,\varphi_i(s(t)),L(t),\omega\big)
$$

**원칙 (C10 판결).** 확률은 원인($W$, $E$, $z$)과 미시 잔차에 둔다. 보장은 불변량에 건다. 카메라는 읽기 전용이다.

| ID | 불변량 | 효과 |
|---|---|---|
| I1 | identity 인자 $\theta_i$는 시간 불변(외형 **값**이 아니라 **인자**에 건다) | morphing이 type 수준에서 금지된다 |
| I2 | Gaussian 수와 질량은 위상 사건(FRACTURE, SPLASH, IGNITE, MERGE)에서만 변한다 | 녹으려면 "녹는 사건"이 있어야 한다 |
| I3 | 비관통, SDF ≥ −1 mm | 관통 아티팩트 없음 |
| I4 | metric $g=9.81$ | 1m 자유낙하 0.45 s, 둥둥 뜨지 않는다 |
| I5 | 사건 마크의 $\mathbf J$와 시뮬 $\Delta p$가 ±5% 안에서 일치 | 소리와 그림의 싱크가 정의상 맞는다 |
| I6 | Read-Only Camera | hole-filling 회귀를 구조적으로 막는다 |

압축: 순진한 4DGS(1M × 59 × 300 ≈ $1.8\times10^{10}$ float) 대비 약 $10^8$ float로, 약 200배 작다. 시간에 따라 변하는 것은 물리 상태뿐이다.

### 8.2 세 교사와 한 증인: 비디오 모델을 교사로 쓰는 구조

| 신호 | 판정 대상 | 기울기가 흐르는 곳 | 차단되는 곳 |
|---|---|---|---|
| **이미지 교사 slice DM** $(c,t,e,L')$ | 임의 정지 순간의 공간 사실감 | $\theta,\Theta,g_\varphi,\delta,o_i$, intrinsics | $s(t),E,\psi$ (선택만 허용) |
| **물리 검증기 RGF** (실분포 보정) | 불변량, 고대역 동역학 | $F_\phi$, 정책, $\psi$ | 외형 |
| **실사건 prior** (타이밍 라벨) | 사건 간격과 타이밍 통계 | EPD | 외형 |
| **비디오 증인** (오픈 가중치, $\le f_c$) | 저주파 의미, 연출, 안무 | EPD의 거친 타이밍, 의도, $z$ | 고대역 상태, 외형 |
| Reader cycle / 적대적 시공간 카메라 감사 | 가독성 / 결함 | 전부 (낮은 가중) | — |

**비디오 모델을 교사로 쓰는 프로토콜.** PLAC(Propose–Lift–Adjudicate–Commit)를 부트스트랩으로 쓰고, 저주파 증인으로 졸업시킨다.

1. **PROPOSE.** 우리 세계를 리드 카메라로 렌더해 조건으로 준다. 오픈 비디오 모델(Wan 2.x, Apache-2.0)이 480p 저해상 초안 $M$개를 낸다.
2. **LIFT.** 형상, 리그, 카메라를 이미 알고 있으므로 남은 미지수는 저차원 상태 $s(t)$뿐이다. IK/PnP 수준의 fitting이고, robust loss와 identity drift 검출을 붙인다.
3. **Spectral Division of Labor** [신규].
   - 범주별 crossover $f_c=\min\{f:R(f')<-3\text{dB}\ \forall f'\ge f\}$, $R(f)=\log P_{gen}/P_{real}$ (ego-motion 보상 가속도 PSD)를 실측한다.
   - 증인의 궤적은 $\mathrm{LPF}_{f_c}$를 거친 것만, 사건 시각은 $\pm1/(2f_c)$ 창으로만 쓴다.
   - 고대역은 PCGD와 실사건 prior가 정한다.
4. **진술 그래프.** 제안 영상을 (사건 타입, 참여자, 부분 순서, 거친 시간 창)으로 바꾼 뒤 그래프 매칭으로 교차신문한다. 그래서 녹는 제안에도 흔들리지 않는다.
5. **PC-ELBO.**
   $$\Delta_m(\gamma)=\mathrm{score}(\mathcal R(h_m,\gamma))-\mathbb E_\pi\,\mathrm{score}(\pi\circ\mathcal R(h_m,\gamma))$$
   $\pi$는 길이 $1/f_c$ 블록 단위의 시간 순열이고 common random numbers를 쓴다. 렌더 아티팩트에 따른 stillness bias를 분리하고, 비디오 모델이 *이 시간 구조*를 얼마나 믿는지를 잰다.
6. **능동 카메라 (QBC).** $\gamma^*=\arg\max_\gamma\sum_{m<m'}w_mw_{m'}\lvert\Delta_m(\gamma)-\Delta_{m'}(\gamma)\rvert$로 고르고, $w_m\leftarrow w_me^{\beta\Delta_m(\gamma^*)}$로 갱신한다. 2–3회 반복한다.
7. **실제 증인.** lift 코퍼스에서 사건 그래프로 실제 클립 top-k를 검색하고 그 타이밍 통계를 쓴다. 환각과 약관 리스크가 없다.
8. **ADJUDICATE / COMMIT.** RGF로 판정하고, 물리로 투영한 궤적을 진실로 commit한다. 비디오 모델의 픽셀은 최종 결과물에 한 픽셀도 들어가지 않는다. 비디오 모델이 "누가 들어온다"고 제안하면, 그 인물은 3D 에셋으로 생성되어 세계에 들어온다.
9. **졸업.** 인턴 호출 비율이 5% 미만이고, 증인에 의존한 사건의 TSR 거리가 실사-실사 노이즈 밴드 안이면 졸업이다. 닫힌 API의 출력은 평가 전용이다.

**시간 연방 지도** (FP의 Lagrangian World와 P2 v2를 통합).

| 공간 \ 시간 | 저 (0–1 Hz) | 중 (1–8 Hz) | 고 (>8 Hz) |
|---|---|---|---|
| 레이아웃 | 비디오 증인 (궤적 공간, 저해상) | Lagrangian motion prior | 사건, 물리 |
| 형태 | motion prior (skeleton, LBS, low-rank) | motion prior | 물리 (contact, MPM/NGFF) |
| 외관 | 이미지 교사 (canonical) + 사건 재질 | 렌더러 (모션블러 적분) | 렌더러, 센서 |
| 확률적 물질 | 바람장 (증인) | SMF sway (seeded) | 파티클 물리 |
| 카메라 | intent prior | 리그 동역학 | 손 모델 (IMU) |

### 8.3 사건과 실현

**EPD v2** [조합]
- 사건 슬롯: $e_i=(b_i,t_i\in[0,T],\mathbf x_i,k_i,p_i,q_i,\mathbf m_i,\ell_i)$, $M\le1024$.
  - 마크 $\mathbf m$: 충격량 $\mathbf J$, 상대속도, 재질쌍, 파편 seed, 의도·감정 임베딩.
  - 층 $\ell\in$ {물리, 연기, 서사, 환경}.
- 연속 성분은 flow matching, 이산 성분은 masked diffusion으로 학습한다. 라벨 등급별로 손실을 마스크한다(존재 / 순서 / 타이밍).
- 스크립트 제약("t=8s에 유리잔이 깨진다")은 사건 집합 공간에서의 slot inpainting으로 준다. 픽셀 캔버스가 아니다.
- 선례: Add-and-Thin, Neural STPP, CLoSD, PhysDiff, Witkin & Kass'88.

**Event Type Contract** [신규]: 타입마다 (실현기, 검출기, 허용오차, 검증기)를 명시한다.

| 타입 | 실현기 | 허용오차 | 검증 |
|---|---|---|---|
| CONTACT/IMPULSE | PCGD | ±2 ms, ±2 cm | RGF, I5 |
| FRACTURE | MPM/Voronoi (seed = 마크) | ±1 ms | I2, seed 재현성 |
| FOOTSTRIKE/GRASP | 캐릭터 정책 + 접촉 투영 | ±1 프레임 | foot skate, 관통 |
| UTTERANCE | TTS → 음소 → viseme | ±20 ms | LSE-C/D |
| GAZE_SHIFT | 머리-눈 IK | ±80 ms, ±3° | 각도 오차 |
| INTENT/EMOTION | 플래너 → 정책, $z$ 곡선 | ±0.5 s | VLM과 인간 판정 |

**PCGD (Physics-Constrained Generative Dynamics)** [조합]
$$s_{t+\Delta}=\Pi_{\mathcal I}\big(\mathrm{Sim}_{base}(s_t,a_t;\psi)+F_\phi(s_t,a_t,\psi,\xi_t)\big)$$
- Gaussian과 1:1인 통합 입자 그래프 위에서 돈다. 강체는 shape-matching 클러스터, 천은 정점, 유체는 입자다. 결합이 엣지로 표현되므로 양방향 결합이 기본값이다.
- $F_\phi$는 sim-to-real 잔차만 학습한다. 데이터는 Stereo4D, 스포츠 멀티캠, 자체 rig다.

**RGF (Real-Reference Ghost Force)** [신규]
- $F_g$는 RFC 정의를 따른다. 판정은 $\Pr_{p_{real}}[F_g\ge F_g(h)]\ge0.05$, 배치 단위로는 MMD.
- 기준은 "lift한 현실만큼만 물리적이어라"다.
- **Reality Residual Budget:** $p_{real}$의 95분위 이하의 잔여력은 정의상 어떤 판사도 현실과 구별하지 못한다. 이 예산을 보이지 않는 제어 채널 $\iota$로 쓴다.

**Milestone SMC (Chaos-as-Controllability)** [조합]
```
particles n=1..4096: Δs0 ~ N(0,Σ_prior), Δψ ~ p(ψ), layout ~ p(layout|W)
for milestone m (사건 제약 시간순):
   GPU 병렬 시뮬 t_{m-1} → t_m+τ_m      (Warp/MJX/Newton)
   w_n ← exp(−d(Ê_n ∩ window_m, e_m)²/2σ_m²) · p_layout · 1[I1–I5] · 1[잠긴 프레임 재투영 ≤ 0.5 px]
   ESS < N/2 → systematic resampling; rejuvenation: 접촉 시 ι 주입(예산 이내) 또는 Δs0 섭동 후 재시뮬
return argmax_n w_n · HeroScore_n(이미지 교사로 t_h 슬라이스 채점)   # 미감은 '선택'으로만 반영
```
- 세대당 약 $4\times10^7$ env-step이고 분 단위로 끝난다.
- 혼돈이 레버가 된다. 민감도가 크다는 것은 작은 섭동으로 다양한 결과에 닿을 수 있다는 뜻이다.
- 선례: Twigg & James'07, Chenney & Forsyth'00, SAMCON.

### 8.4 외형과 세계 (Phase 2)

- **Edit-Closed World** [신규].
  $$\mathcal L_{EC}=\mathbb E_{e\sim p_{edit},c,L'}\mathrm{KL}\big(p_{\mathcal R}(\cdot\mid e,c,L')\Vert p^{img}(\cdot\mid y_e)\big)$$
  - $p_{edit}$는 EPD의 저해상 버전이다(제거, $SE(3)$ 이동, 관절 개방, 광원 교체, 인물 slot 제거). 반사실 샘플 비율은 0.3–0.5.
  - 사건 모델이 새 사건을 배우면, 세계도 그 사건에 대해 닫힌다.
  - 지표: RRS, Reveal-MEt3R, GSI, albedo leak.
- **상태 조건부 외형.** $\varphi_i=[\log(F^\top F)_{(6)},\Delta\mathrm{AO},\text{접촉압},\text{젖음},\text{온도},\lVert v\rVert]$.
- **MAN (Matter-Anchored Noise) 셰이딩** [신규].
  $$\xi(\mathbf u)=\lvert\Omega_{\mathbf u}\rvert^{-1/2}\sum_{q\in\Omega_{\mathbf u}}\eta(q),\qquad r=g_\varphi(\mathrm{Gbuf}_{16\times16},\xi,\omega),\quad \lVert r\rVert\le\tau$$
  - 1-step generator이고 알파·기하 출력이 없다. canonical 물질 셀 노이즈를 쓰므로, 같은 물질점·상태·뷰면 출력이 같다. 모공의 반짝임이 물질에 붙어서 움직인다.
  - 이것은 WNF의 Lagrangian 버전이다. Cine의 Lagrangian SNF와 같은 것이다.
- **Avatar Foundry v2.** identity 레퍼런스를 조건으로 이미지 교사 분포 매칭을 한다. 뷰별 코드를 평균하지 않는다. 얼굴은 고차원 expression latent(256–512, Codec Avatar 계열)로 표현하고, 눈은 각막 굴절과 catchlight를 별도로 만든다.

### 8.5 로드맵과 승리 조건

| 단계 | 시점 (Phase 1 이후) | 범위 | 킬러 데모 | 승리 조건 |
|---|---|---|---|---|
| **2a Frozen Action & Speed Ramp** | +3–6개월 | 사건 한가운데 정지(유리 파괴, 물 왕관, 폭발 파편, 도약 댄서, 흩날리는 머리와 천) → 궤도 → ramp 0→0.05x→1x. 세계 시간 창 0.3–1.5 s | 바에서 와인잔이 깨지는 순간 정지, 360°, 1x로 복귀, 반대 각도 재촬영에서 파편 하나하나가 같다 | BUPS-FA Arena에서 각 경쟁 모델 대비 ≥ 70%. 정지 슬라이스 Appearance Turing 식별률 ≤ 65%. 정지 궤도 GeCo ≤ 0.03. ramp 구간 TSR이 실사 밴드 안. RST 타이밍 ≤ 1 프레임 |
| **2b 강체·관절체 + 살아있는 세계** | +6–12개월 | 낙하, 충돌, 쌓기, 문과 서랍, 차량, 파괴, 입상체, 연체, Rube Goldberg. 식생, 바람, 연기, 물, 구름, time-lapse (Spectral Gaussians와 공유 바람장은 부품) | Rube Goldberg 3캠 + 1000fps. 60초 드론 오빗에서 돌풍이 밀밭을 건너고 하이퍼랩스로 밤까지 | Physics-IQ와 VideoPhy-2 1위. Motion Turing ≤ 60%. CSI KL ≤ 최강 경쟁 모델 × 0.5. 강체 MI가 노이즈 수준. BUPS ≥ 65%, 일반 Arena ≥ 50%. GCM 지연이 $d/U$의 ±20% 이내 |
| **2c 캐릭터·연기** | +12–18개월 | 단일·다중 인물, 보행, 댄스, 스포츠, 대화와 립싱크, 클로즈업, 손, 의상, 머리카락, 동물 | 2인 대화 하나의 4D 테이크에서 master/OTS/CU 자동 생성(Coverage Compiler), 디렉션별 테이크 3개, 120fps 댄스 | 전문가 패널 연기 축 ≥ 55%, 멀티숏 연속성 ≥ 70%. 얼굴 Appearance Turing ≤ 75%. foot skate < 2 cm/s, ArcFace ≥ 0.8, eyeline < 3°, 관통 < 1% |
| **2d 완전 4D + 인터랙션** | +24–36개월 | 다인물 상호작용, 손-객체 조작, 캐릭터-유체·파괴 상호작용, 분 단위 원테이크, 다장면, 실시간 | 여러 방을 지나는 5분 원테이크를 멀티캠으로 재촬영, 10분 뒤 돌아와도 세계가 그대로 | 모든 범주 Arena ≥ 60%. Revisit 시간 기울기 ≈ 0. 인과 정확성 1위 |

**Phase 2 부가 능력.**
- **Temporal DoF** [신규] `P2`. $\mathrm{rate}(x)=\exp(-(Z(x)-Z_{tf})^2/2\sigma_T^2)$, $t_{eff}=t_0+\int\mathrm{rate}\,dt$. 한 깊이 층만 시간이 흐르고, 초점면을 옮기면 다른 층이 깨어난다("랙 타임"). slit-scan이 화면 좌표를 시간에 대응시켰다면 이것은 깊이를 시간에 대응시킨다.
- **시간 방향 PSH.** 셔터보다 빠른 대역은 $\Sigma_{mb}$로 통계적으로 렌더한다. 프레임 Nyquist와 $1/\Delta t$ 사이 대역은 명시적으로 유지하므로 왜건 휠 효과가 진짜로 재현된다.
- **Gabor 물결.** CARRIER 위상을 분산 관계 $\omega=\sqrt{g\lvert\mathbf k\rvert\tanh(\lvert\mathbf k\rvert d)}$로 진행시킨다.
- **Single-cause audio.** 사건 $\mathbf J$로 모달 합성(Zheng & James), 파괴 음향, Harmonic Fluids, 발소리, UTTERANCE→TTS(같은 음소 타임라인이 입술을 구동)를 만든다. 전파는 image-source + 음선 추적, 도플러는 시변 지연선에서 자연히 나온다. **Sync Style 스위치**(physical / cinematic). 출력은 바이노럴, Ambisonics, ADM 메타데이터가 붙은 Atmos 객체 스템.
- **분기 블록 우주** [부품: Many-Worlds Browsing], **Coverage Compiler** [부품: Virtual Cinematographer'96], **테이크**($z$만 다시 샘플).

---

## 9. 데이터 플라이휠과 native World Foundation Model

### 9.1 소스

| 용도 | 소스 |
|---|---|
| 정적 세계 | DL3DV-10K(4K), ScanNet++, RealEstate10K(전진 위주이고 baseline이 작아 loop closure 학습에 불리하다), ACID, Sekai(5,000시간 이상), MatrixCity, Matrix-Pano 116K, Infinigen(완전 GT), Objaverse-XL |
| 동적 4D | SpatialVID(동적 7,089시간), Stereo4D(metric 10만 이상), OmniWorld, AIST++, Ego-Exo4D, HOI4D, BEDLAM, AMASS, **스포츠 리그 멀티캠**(자연 발생 4D GT) |
| 카메라 | DataDoP 29K, CameraBench, **IMU 코퍼스**(GoPro, Sony, DJI, Betaflight, Gyroflow) |
| 보정과 Gold | ScanNet++ DSLR, 자체 오라클 장면, **Gold Set**(1000fps 다시점 + 다중 거리 마이크 + LiDAR 50시간) |

### 9.2 Lift 파이프라인 (Reader)

```
원본 → [필터] 카메라 운동·시차·동적 비율·미감·컷 / VFR 정규화
     → [기하] DA3 / VGGT-Ω / π3 (metric) + rolling shutter 보정
     → [정적/동적] MonST3R식 마스크 → 정적 부분은 STRATA fit (+ PSH 라벨: 국소 FFT → Goldstein 위상 언래핑 ψ, 얇은 구조, OCR 리프팅, HDR 하이라이트)
     → [동적 4D] GVHMR/WHAM (SMPL-X), MANO, FoundationPose, MoSca
     → [서브프레임] Blur-as-Measurement (셔터 역적분, S 추정)
     → [사건] Event Type Contract 검출기
     → [오디오] onset → AV-Physics Timing Solver:
                t_on = t_ev + β·d + δ_clip + n (구간 검열 likelihood)
                β≈1/c → physical sync(타이밍 라벨) | β≈0 → cinematic sync(스타일 데이터) | Δd<3m → 식별 불가
     → [물리] 시뮬 추종 → ghost force → 범주별 p_real 갱신 (RGF)
     → [피팅] (W_canon, s(t), E, γ) + 신뢰도 (holdout 프레임 PSNR/LPIPS, 타이밍 σ)
라벨 등급: 존재 / 순서 / 타이밍(σ < 5 ms, 전체의 1–5%)
```

### 9.3 학습 단계와 플라이휠

1. **Stage A (현실 prior).** lift 세계로 CWM, LBO, EPD를 flow matching한다. 관측되지 않은 셀은 마스크한다. 완성 능력은 Infinigen과 Stage B의 critic이 가르친다.
2. **Stage B (분포 매칭).** WDM(§1.9)과 Edit-Closure(§8.4), slice DM을 하고, §8.2 라우팅 표에 따라 기울기를 나눈다.
3. **Stage C (Render-RL / Directed).** 경로 영상 보상을 쓴다.
4. **Expert Iteration.** WSSC v3가 탐색하고, student가 정책이고, 교사들이 judge다. solver의 $W^\star$가 student의 real anchor가 된다. 3D 정답 데이터 없이 래칫이 돈다.
5. **Builder–Reader self-play** `P2`. Builder가 세계를 만들고 Reader가 렌더를 읽는다. Reader로 실영상을 다시 lift하면 $p_{real}$과 사건 prior가 갱신된다.
   - 보상: $r=r_{RGF}+r_{cycle}+r_{real}+r_{img\text{-}slice}-r_{adv}$
   - 붕괴 방지: 배치의 50% 이상은 실데이터, 검증 게이트, 홀드아웃 상시 감시.
6. **재조합 엔진** [부품: BEDLAM]. (W, γ, P, A)를 어포던스 키로 매칭하고, 물리 리타게팅에 실패한 조합은 기각한다.

### 9.4 아키텍처와 토큰

- **토크나이저.**
  - 정적 부분: 2m 청크 → 12.5cm sparse voxel 16³ × 8ch → voxel당 UKA 32개(intrinsics, θ, slot 포함)
  - 원경: layered panorama latent
  - 에셋: SLAT → perceiver 512–2048 토큰
  - 사건: ≤1024 토큰
  - PSH 구조 집합: DETR식 헤드 + SYMBOL용 AR 헤드
  - **10초 장면 ≈ 50–64k 토큰.** 1080p 10초 비디오 DiT는 약 50만 토큰이다.
- **CWM:** 전역 full attention(≤256k 토큰), 미터 단위 3D RoPE를 $h_l$로 스케일, register 64개(태양, 노출, 스타일), hero Plücker 토큰, 의도 토큰.
- **LBO:** 반경 $r_l$ 윈도우, $K\le4$, Jacobi 패스 ≤2. 분산 보존 VP 블렌딩은 확률적 잔차에만 쓴다.
- **규모 가설:** 정적 4–8B, 에셋 1–3B, 사건 0.5–2B, $F_\phi$ 0.3–1B, $g_\varphi$ 0.2–0.5B, 정책 0.1–1B.
- **추론 경제성:** 세계 생성은 한 번이다. 카메라를 10대로 늘리거나 길이를 늘리거나 120fps로 올려도 한계 비용은 렌더 비용뿐이다(T4). 그래서 **세계 단위로 과금**할 수 있다.

### 9.5 스케일링 가설

| 가설 | 내용 | 검증 |
|---|---|---|
| H-dedup | 같은 연산에서 고유 texel당 품질 예산이 비디오 대비 10–25배 | X3에서 HER, SFR로 간접 측정 |
| H-ratchet | expert iteration 세대가 거듭될수록 search gain이 단조 감소 | 소규모 3세대 |
| **H1′ (go/no-go)** | 인간이 판정한 외형·모션 사실감이 같은 지점에서, 일관성·물리 축의 전체 컴퓨트(학습+추론)가 $C_{world}\le C_{video}/\kappa$, $\kappa\ge10$ | $10^{20}$–$10^{22}$, 같은 데이터 원천, Factorized Turing 점수로 fit |
| H2 | lift 세계 하나가 비디오 $K_{eff}$ = 5–50개 가치 | |
| H3′ | 모션 손실은 **타이밍 라벨이 붙은** 사건 수의 멱법칙 | 라벨 등급별 분리 측정 |
| H5′ | SMC 입자 수 N과 solver 반복에 대해 품질이 ∝ log N이고, 비디오 best-of-N보다 높은 포화점 | 같은 비용 비교 |
| H6 | 반사실 비율 0.3–0.5에서 RRS가 포화하고 원래 뷰 품질 저하 ≤ 2% | X6 확장 |
| H7 | 이미지 교사 + 물리 + 실사건 prior만으로(비디오 고대역 없이) Turing 동률 | 2b 중반 |

### 9.6 컴퓨트 견적 (H100 bf16, MFU 40% ≈ $1.44\times10^{18}$ FLOP/h)

| 항목 | H100-h |
|---|---|
| 정적 lift 5천만 클립 | 0.1–0.2M |
| WSSC 데이터 엔진 1–2만 세계 × ~10 h | 0.1–0.2M |
| WDM post-train (Phase 1 student) | 0.05–0.12M |
| CWM/LBO 사전학습 T1 ($10^{23}$) | ~70k |
| **Phase 1 누계** | **약 0.4–0.8M (~$1–2M)** |
| 동적 lift 2천만 + blur + timing | 0.3–0.5M |
| 반사실 DM, SMC 실현 100만 샷, $F_\phi$/MAN/정책 | 0.4–0.8M |
| WFM T2 ($10^{24}$) | ~0.7M |
| Gold Set 캡처 | 장비와 인력 $0.3–0.6M |
| **2c까지 누계** | **약 2.5–5M (~$6–14M)** |
| 2d 프런티어 (T3, $10^{25}$) | +8–10M |

---

## 10. 첫 90일

### 10.1 결정적 실험

| ID | 가설 | 셋업 | 성공 기준 | 실패하면 (업그레이드) |
|---|---|---|---|---|
| **X0** Spectrum Report | 상용 최상위 모델은 (a) 3–12 Hz 각속도 PSD가 실제보다 ≥ 6 dB 부족하고, (b) 정지를 요청해도 FV > 0이며, (c) VBench smoothness가 결손과 양의 상관이다(질병에 상을 준다). 원인을 VAE / 조건부 평균 / 필터로 분해할 수 있다 | GoPro gyro GT 300개로 추출기 $H(f)$ 보정. 4모델 × 6범주 × 40프롬프트 × seed 2. Wan2.2/HunyuanVideo VAE 왕복. 평가자 30명 | 4개 중 3개 이상에서 결손(95% CI), ρ(결손, too-smooth 판정) ≥ 0.4 | 결손이 없으면 축을 질량 시그니처(오버슛, 가속 프로파일)와 CSI로 옮긴다. 블로그는 어느 경우든 나간다 |
| **X1** Dreamer Calibration & Spectral Deficit | (a) 오픈 교사의 $D(o,t)$가 한 옥타브를 분리한다. (b) 평균 융합의 SFR이 T5 예측의 ±10% 안이다. (c) 분할 소유권은 SFR ≥ 0.9이고 seam 2AFC ≤ 60%다 | Qwen-Image / HiDream-I1(상업용), FLUX(연구용). 매크로·파사드 2,500장 × t 30 × seed 8. GT 깊이 실스캔 장면에서 K = 4/8/16 다시점 → L2 3DGS / 평균 / 분할. WNF 유무별 ρ와 Worldness Gap | (a)(b)(c) 모두. 자기 캡션으로 Droste 신호 −50% | 결정 폭 > 1.5 옥타브면: 미세 2옥타브는 픽셀 공간 refiner, per-token-t 어댑터(1M장, 미감 −3% 이내), 또는 레벨 간격을 2옥타브로 |
| **X2** Oracle A/B (T0) | 완벽한 정적 세계 + 우리 카메라 스택이 경쟁 I2V를 이기고 빈 사분면에 들어간다 | 자체 촬영 20장면(400–800장, HDR 브래킷) + DL3DV 상위 10개, 3DGUT/H3DGS. 의도 4 × 리그 2. 일반 40명, 전문가 15명, 약 3,000쌍 | VP 일반 ≥ 70%, 전문가 ≥ 75%, GeCo ≤ 0.03, SP ≥ 50% | 카메라/룩 ablation 그리드(블러, 그레인, 리그, 톤). spurious motion이 "생명감"으로 선호되면 매체만 살아 있는 허용 미세운동(빛, 공기 중 입자)을 넣는다. 결과 점수를 세계 팀 목표로 고정한다 |
| **X3** Fusion Showdown (핵심) | WSSC v3와 장면별 WDM 중 하나 이상이 생성 후 재구성 계열보다 선명하면서 기하가 실사 바닥에 근접한다 | DL3DV 20 + 생성 5. 코스(약 4 cm)는 재구성 후 저역통과. 2 cm/1 cm/5 mm 레벨 생성. 비교: (i) SEVA 24–80뷰 + 3DGS, (ii) i + MRF 소유권, (iii) SyncTweedies식, (iv) Difix3D+ 재증류, (v) WSSC v3, (vi) 장면별 WDM(VSD, LoRA 64) | HER ≥ 0.8, 오빗 GeCo ≤ 0.03, TP ≥ 40%, OIE ≤ 0.05, 최강 기준선 대비 2AFC ≥ 65%, 채도 드리프트 ≤ 5% | 블러가 남으면 t_mid부터 소유권, γ 조정, UKA $K_l$ 증가, cos 변조 커널. ρ가 오르지 않으면 픽셀 refiner + WNF. 과포화와 Janus가 남으면 하이브리드(critic + 동기 샘플) |
| **X4** DTC | 확률적 카메라를 통과시킨 증류가 보케·플레어를 굽지 않는 HDR 세계를 주고 미감을 유지한다 | X3 결과 장면 10개. (A) deep focus + 역카메라, (B) DTC, (C) DTC + Gaffer/샷 조명 치팅 | BBI ≤ 0.1, 플레어 문신 0, 동적 범위 ≥ 14 stop (GT ±1), TP ≥ 45%, 전문가 선호 C > B ≥ 60% | $\hat p_T$를 EM으로 재추정, 초점면 선명도 제약. 미감이 떨어지면 샷 공유 LUT를 카메라 파라미터로 연다. 최후에는 `<plate>` LoRA |
| **X5** PSH stress | 대역만 쓰는 국소 연산자는 결맞음 구조와 에너지가 큰 작은 원소에서 실패하고, PSH는 같은 연산량으로 이를 고친다 | Blender/Infinigen 2,000장면(벽돌, 타일, 펜스, 50–500m 송전선, 간판, 야경, 눈 글린트), 4레벨 STRATA, 150M 국소 flow | 경계 95%에서 위상 오차 < π/8, CER ≤ 5%, 보케 볼 재현율 ≥ 90%, 선 대비 ±20%, 게이팅 절감 유지 | 유기물에는 orientation-only phasor 캐리어. 승격 원소가 폭발하면 lightcuts식 계층 POINT |
| **X6** Appearance Anchoring | RAS는 교사의 룩을 지키고(TP ≥ 45%) 완전 역렌더 PBR은 잃는다(≤ 30%). 광원·slot 반사실 30%로 albedo leak −50% | X3 장면 30개 × 두 셰이딩 모드 + 같은 림 리그. 합성 GT 20장면 | 위 수치. 거울 PSNR ≥ 32 dB, 하이라이트 고착 상관 < 0.1, 원래 뷰 저하 ≤ 2% | 고착이 생기면 다시점 저백분위 분리를 강화하고 $L_d$ 시점 분산 벌점. 리그 델타가 플라스틱처럼 보이면 IC-Light식 픽셀 국소 보정 헤드. 숨은 면이 실패하면 객체 공간 compositional 생성 |
| **X7** Camera Intelligence | (a) 처음 보는 세계에서 CRM 정확도 ≥ 70%. (b) CRM 탐색 경로가 모방 경로보다 ≥ 65% 선호. (c) IMU 손 모델이 튜링 2AFC ≤ 55% | 세계 120 × 경로 40, 720p 8초, 일반 1.5만 쌍 + 전문가 2천 쌍. IMU 300시간 이상 | (a)(b)(c) | 세계 토큰 조건 + 능동학습. gyro는 회전만 있으므로 VIO로 평행이동 성분 보강 |
| **X8** Phase 2 seed | Milestone SMC가 도미노·핀볼·유리 과제에서 제약 충족 ≥ 90%(±1 프레임, ±2 cm), 10 GPU-분 이하. Frozen Action 프로토타입 1샷 | MuJoCo-Warp/Newton, N = 4096, 과제 20 × 제약 5. 와인잔 파괴 MPM 0.5 s + hero-instant 선택 | 위 수치 + "조종된 롤아웃" 식별 ≤ 55% | 허용오차 창 학습, 잔여력 예산 확대, feasibility critic guidance |

### 10.2 주차별 계획

| 주 | 엔진 (세계) | 표현·렌더 | 카메라·DP | 데이터·평가 | Phase 2 seed |
|---|---|---|---|---|---|
| 1 | 교사 harness, 라이선스 검토, WNF v0 | STRATA v0 자료구조, 3DGRUT fork | 카메라 스택 v0 (thin-lens, 셔터, 센서) | 지표 구현 착수(GeCo, MEt3R, BS, TPI), **X0 수집** | — |
| 2 | **X1** 결정 곡선 | UKA, dual-cut v0 | 리그 6종 시뮬 | 오라클 장면 촬영 시작, IMU 코퍼스 수집 | — |
| 3 | X1 Spectral Deficit, Worldness Gap | 공분산 블러 + MC 타일 | Gyroflow/GPMF 파이프라인 | **X0** 추출기 보정 | — |
| 4 | **Day-30 게이트 준비**, WSSC v3 v0 | RAS + intrinsics 채널 | 손 모델 v0 학습 | **X0 블로그 공개**, **X2** 렌더 | — |
| 5 | **X3** arm i–iv 기준선 | 렌즈 처방 LUT, 플레어 | DP-0 prior (DataDoP) | X2 사람 평가 진행 | — |
| 6 | X3 arm v (WSSC), **X4** DTC | PSH 구조 렌더 (POINT, CURVE, SYMBOL) | Toric + 탐색 루프 | **X5** 합성 데이터 생성 | X8 시뮬 harness |
| 7 | X3 arm vi (WDM 장면별) | **X5** 학습 | **X7** 쌍 렌더 | **X2 결과**, FREEZE-TAG 프롬프트 초안 | X8 SMC |
| 8 | **Day-60 게이트**, X3 중간 결과 | Ray-Bundle Demand | X7 평가 수집 | 경쟁 클립 생성 시작 | X8 와인잔 MPM |
| 9 | X3 반복, SMF v0 | **X6** RAS vs PBR | CRM 학습 | T4± 경로 추출 | Hero FX 연동 |
| 10 | Hero FX v0 (FLIP 정지 + CEM), adversarial camera | X6 + Edit-Closure lite | DP-0 v0 + Gaffer | X5 결과 | hero-instant 선택 |
| 11 | 데모 세계 3개 생성 | 4K/8K 최종 렌더 경로 | 데모 경로 탐색 | **FREEZE-TAG v1 사전 등록** | X8 결과 |
| 12 | 데모 polish | 최종 렌더 | 리그 스왑 릴 | 사람 평가 본실행 | Frozen Action 1샷 |
| 13 | **Day-90 게이트** | — | — | 기술 보고서, 벤치 공개, 데모 릴 3편(왕복, 불릿타임 헌정, 을지로 네온) | 2a 계획 확정 |

### 10.3 게이트 (실패 시 목표를 낮추지 않고, 해당 실험의 "실패하면" 업그레이드를 앞당긴다)

| 시점 | 통과 조건 |
|---|---|
| **Day 30** | X1 (a)(b)(c) 통과. X0 결손 확인 |
| **Day 60** | X2 VP ≥ 70%. X3에서 하나 이상의 arm이 HER ≥ 0.8, GeCo ≤ 0.04, OIE ≤ 0.05 |
| **Day 90** | 하나 이상의 범주 세트에서 SP ≥ 45%, GeCo ≤ 0.03. T2 최강 경쟁 모델 대비 VP ≥ 70%. 데모 릴 3편. X8 충족률 ≥ 90%. X3의 승자 arm으로 Phase 1 데모 경로 확정 |

### 10.4 최소 팀

| 역할 | 인원 | 담당 |
|---|---|---|
| 세계 엔진 리드 (디퓨전, DMD/WDM) | 1 | X3(vi), student, expert iteration |
| 세계 엔진 (동기화, 보정) | 1 | X1, WSSC v3, WNF, SMF |
| 표현·렌더러 | 1 | STRATA, UKA, dual-cut, RAS, 3DGRUT |
| 카메라 물리·렌즈 | 1 | 렌즈/센서, Ray-Bundle, X4 |
| 카메라 지능 (DP) | 1 | 손 모델, DP-0, CRM, X7 |
| 데이터·재구성 | 1 | 오라클 촬영, lift, PSH 라벨, X5 |
| 평가·사람 연구 | 1 | X0, X2, FREEZE-TAG |
| Phase 2·시뮬 | 1 | X8, Hero FX |

권장은 8명이다. 최소 6명까지 줄일 수 있다(표현과 카메라 물리 통합, 데이터와 평가 통합). 인프라는 파트타임이나 계약직으로 채운다.

### 10.5 컴퓨트와 예산

| 항목 | 규모 |
|---|---|
| GPU | H100 64장 전용 (90일 용량 ≈ 138k H100-h), 9–12주차에 128장으로 burst |
| 사용 예상 | X0 2k, X1 4k, X2 3k, **X3 25–40k**, X4 6k, X5 4k, X6 5k, X7 4k, X8 3k, CWM/LBO 파일럿 10k, 데모 10k → **약 75–90k H100-h** |
| 비용 | 컴퓨트 $0.2–0.35M + 경쟁 API $10–20k + 사람 평가 $50–80k + 촬영 장비·대여 $15–30k (1000fps 대여, 폰 rig, HDR) |

---

## 11. 리스크 → 업그레이드

| # | 리스크 | 감지 지표 | 업그레이드 (후퇴 없음) |
|---|---|---|---|
| 1 | WDM이 장면 규모에서 수렴하지 않는다(선례 없음) | DR_rel 정체, 과포화, Janus | solver 경로가 데모를 책임진다. expert iteration으로 solver 출력을 real anchor로. 레벨 커리큘럼, 하이브리드(critic + 동기 샘플) |
| 2 | 소유권 seam이 퀼트처럼 보인다 | seam 2AFC > 60% | WNF로 ρ↑, seam을 텍스처 마스킹이 큰 곳에 배치, multiband $\lambda_b$, EQ-VAE와 RE-FT(spectral + adversarial로 블러 해법을 차단) |
| 3 | 교사 인터페이스(후광, 채도, Droste) | 깊이 엣지 고주파 비, $\Delta C^*$, CLIP 신호 | band-isometric, 자기 캡션, APG/CFG-rescale, 저주파 색 고정 |
| 4 | 명시적 세계의 CG 룩과 촬영지 수준 미감(Atlas 피벗의 이유) | SP/TP < 게이트 | DTC, Gaffer, Directed World, shot dress, RAS, DP-0 + CRM. SP를 게이트로 둔다 |
| 5 | 얇은 구조, 투명체, 털, 물, 불 | 자해 카테고리 점수 | PSH(CURVE), BCSDF/SGGX payload, DIEL SDF + 3DGRT, Hero FX 시뮬 |
| 6 | 텍스트와 구조 패턴 붕괴 | CER, 위상 오차 | SYMBOL, CARRIER + 구조 주소 노이즈, $\mathcal L_{ocr}$ |
| 7 | 세계당 연산 과다 | 프로파일링 > 3 H100-h(제품) | lens-gated demand(보케 할인), student amortization, distilled 교사, dedup |
| 8 | 렌더러 범위 폭발, train/test gap | 렌더러 PR 비중, 모드 간 LPIPS | UKA 하나 + 3DGRUT 하나, 같은 코드 경로 |
| 9 | 벤치가 우리에게 유리하다거나 GeCo와 중복이라는 비판 | 리뷰 반응 | GeCo를 코어로 차용, 빈 사분면, T1 홈그라운드, 자해 카테고리, 사전 등록, 제3자 |
| 10 | 경쟁 모델이 일관성을 개선한다(3D 캐시, Atlas, Seedance 3, Veo 4) | AA Elo, FREEZE-TAG 정기 재측정 | 구조적 우위(결정론, 순서 불변, 광학, 재촬영, 세계 단위 과금, 8K·120)는 유지된다. 오픈 모델은 우리의 저주파 증인이 된다 |
| 11 | 라이선스(교사, 데이터, 닫힌 API 약관, 지역 조항) | 법무 게이트 | Apache-2.0/MIT 기본 스택. FLUX는 연구용. 닫힌 모델은 art director와 평가용. HunyuanVideo 지역 조항 확인. 스톡과 리그 파트너십 |
| 12 | 멈춘 세계가 지루하다 | 일반 VP, 이탈률 | 동기 플래너, Wipe와 PR 보상, Hero FX, 빛만 흐르는 세계, 2a Frozen Action |
| 13 | 카메라 지능이 노이즈를 학습한다 | 리그 분류 불안정, PSD에 SfM 지터 | 식별 가능성 분할, IMU 코퍼스, 추출기 전달함수 보정 |
| 14 | DR reward hacking(디노이저를 속이는 세계) | held-out seed DR과 학습 seed DR의 괴리 | dreamer 2종 앙상블, held-out seed와 카메라, 인간 점검 |
| 15 | (P2) sim look, 판사가 현실까지 기각 | Motion Turing > 65%, ghost-force AUC < 0.65 | PCGD 실잔차 학습, RGF 두 표본 판정, 캡처 10배 확대 |
| 16 | (P2) 증인이 병을 전염 | 우리 TSR이 비디오 모델 쪽으로 이동 | Spectral Division, PC-ELBO, 진술 그래프, 실제 증인 |
| 17 | (P2) 혼돈계 비수렴 | 충족률 < 50% | Milestone SMC, Reality Residual Budget, Event Type Contract |
| 18 | (P2) 웹 오디오와 블러 라벨 오염 | 타이밍 σ 분포 | AV-Physics Timing Solver, Blur-as-Measurement, Gold Set, 라벨 등급 |
| 19 | 사람과 얼굴 | 뷰 간 ArcFace < 0.7 | 얼굴 단일 owner 잠금, Avatar 프리미티브, SAM 3D Body / SMPL-X |
| 20 | H1′ 불성립 | κ < 10 | 원인 분해(토크나이저, lift, 라우팅, 상태/외형 분리 지점) 후 표현 업그레이드 |

---

## 12. 열린 질문 (창업자가 결정할 것)

1. **첫 제품 형태.** (a) 크리에이터·촬영감독용 툴(세계 + ShotScript + 재촬영), (b) T2V형 API(프롬프트 → 완성 영상), (c) VFX·버추얼 프로덕션 B2B(USD/EXR 납품). 어느 것이 FREEZE-TAG 승리를 가장 빨리 매출로 바꾸는가?
2. **Location 모드 vs Directed 모드 기본값.** shot dress와 샷별 조명 치팅을 기본으로 허용할 것인가? 정직한 세계(재방문)와 광고급 한 컷 사이의 브랜드 선택이다.
3. **닫힌 SOTA 이미지 모델의 사용 범위.** art director(hero still), GAN real 집합, prior-agnostic polish 소유자 중 어디까지 쓸 것인가? 아니면 순수 오픈 스택(Qwen-Image, HiDream)으로 갈 것인가? 법무 판단이 필요하다.
4. **자체 이미지 교사 학습 여부.** 장기적으로 "세계를 판정하기에 최적인 교사"(Worldness Gap 최소, plate·intrinsic 헤드 내장)를 직접 학습할 것인가?
5. **test-time solver의 제품화.** 샷당 수~수백 GPU-h인 히어로 모드를 고가 티어로 팔 것인가, 내부 데이터 엔진으로만 쓸 것인가? 세계 단위 과금 가격대는?
6. **벤치마크 공개 시점과 자해 카테고리.** 90일 차에 우리가 지는 칸까지 공개할 것인가?
7. **Phase 2 트랙을 Day 0부터 운영할지.** 1인 seed 트랙(X8)으로 유지할 것인가, Phase 1 Day-90 게이트 이후에 본격화할 것인가?
8. **캡처 인프라 투자.** 오라클 장면, Gold Set(1000fps, 다중 마이크, LiDAR), 멀티캠 rig 배포를 어느 규모로 할 것인가? 이것이 독점 해자가 된다.
9. **오픈소스 전략.** STRATA 포맷, FREEZE-TAG, 렌더러 확장, ShotScript 중 무엇을 공개해서 표준을 선점할 것인가?
10. **실시간 뷰어(웹, VR, FPV)의 우선순위.** 바이럴 효과는 크지만 엔지니어링 비용이 Phase 1 핵심과 경쟁한다.
11. **런칭 첫 컷 하나.** 매트릭스 오마주(Hero FX 의존, 리스크 높음)인가, 왕복 챌린지(구조적 승리, 리스크 낮음)인가?
12. **하이브리드 허용 여부.** 경쟁 비디오 모델을 저주파 증인 이상으로 쓰는 것(예: 세계 bootstrap의 후보 생성기)을 허용할 것인가? 순수성과 속도 사이의 선택이다.
13. **초상권과 아바타 정책.** 퍼포먼스 캡처 마켓플레이스(동의, 로열티)와 촬영감독 스타일·손 라이선스의 수익 배분 구조를 어떻게 할 것인가?

---

## 부록 A. 아이디어 배치 레지스트리

| 아이디어 | 출처 렌즈 | 배치 |
|---|---|---|
| 인수분해 E1, Read-Only Camera(I6), 정리 T1–T10 | FP, Rep, Cine, P2 | P1-코어 |
| WSSC v3 (WSSC + PLWD + S²R 대역 규칙) | FP, Cine, Rep | P1-코어 |
| band-isometric 뷰, 자기 캡션, APG, 스펙트럼 결정 곡선 보정 | Rep v0.2 | P1-코어 |
| World Noise Field (SNF, world-anchored, Philox, 구조 주소), Frozen-Seed Band-Commit | Cine, FP, Rep | P1-코어 |
| 대역 분할 소유권, Partition Principle, World-Scale Causality, explain-away cascade | FP | P1-코어 |
| Dreamer Calibration, Worldness Gap, dreamer 계약 | FP | P1-코어 |
| Spectral Federalism (공간) | FP | P1-코어 |
| Stochastic Matter Field | FP v2 | P1-코어 |
| STRATA 격자, PSH 구조 5종, dual-cut, 효도·의미 효도 손실 | Rep | P1-코어 |
| UKA + 3DGRUT 단일 렌더러, 공분산 대수 | Rep | P1-코어 |
| Dual-Anchored Appearance (RAS + intrinsics), 귀속 제약 $B_{mat}$ | Rep, P2, Cine | P1-코어 |
| DTC, Gaffer, shot rigs | Cine, FP, Rep | P1-코어 |
| Lens-gated Ray-Bundle Demand + 승격 면제 | Rep, Cine | P1-코어 |
| 물리 카메라 스택, Kinetic Camera, 리그 14종, IMU 손 모델 | FP, Cine | P1-코어 |
| DP-0 v0 (prior + 탐색), 동기 플래너, ShotScript | Cine | P1-코어 |
| adversarial camera 감사 (DR / DMD 에너지) | FP, Rep | P1-코어 |
| 비회귀 조항, OIE, order_purity_check | Cine, Rep | P1-코어 |
| FREEZE-TAG Bench (빈 사분면, T0, T4±, 광학 포렌식, BLI, MAT) | Cine, Rep, FP | P1-코어 |
| Edit-Closure lite (광원, 히어로 slot) | P2 | P1-코어 |
| WDM student, expert iteration | Rep, FP | P1-확장 (Day 90 이후 본학습) |
| Hero FX (시뮬 정지 스펙터클) | Cine | P1-확장 |
| Directed World bilevel, shot dress | Rep | P1-확장 |
| PIS 2.0 VDP risk 배분기 | Cine | P1-확장 |
| CRM, ShotDNA v2 전체 | Cine | P1-확장 |
| Powers of Ten 의미 줌, WDD 무한 디테일 | Rep, FP | P1-확장 |
| 빛만 흐르는 세계, 재조명, 매직아워 | Cine | P1-확장 |
| EQ-VAE, RE-FT | FP | P1-확장 (X3 실패 시 1순위) |
| 실시간 뷰어, VR, FPV 조종기, 양안 광택 | Cine | P1-확장 |
| 원클릭 정지 커버리지 | Cine | P1-확장 |
| 블록 우주 정식화, 세계선 시간 사상 | P2 | P2 |
| EPD v2 + Event Type Contract, Event-first Time | P2, FP | P2 |
| Milestone SMC + Hero-Instant, PCGD, RGF + Reality Residual Budget | P2 | P2 |
| 상태 조건부 외형, MAN 셰이딩, slice DM, Edit-Closed full | P2 | P2 |
| 세 교사 + 한 증인, Spectral Division, PC-ELBO, QBC, 진술 그래프, 실제 증인 | P2 | P2 |
| Lagrangian World, identity conservation, 시간 연방 | FP | P2 |
| Temporal DoF / 랙 타임, 시간 PSH, 왜건 휠, Gabor 물결 | Cine, Rep | P2 |
| 공유 바람장, GCM | P2 | P2 (2b, 부품) |
| Single-cause audio, AV-Physics Timing Solver, Blur-as-Measurement, Gold Set | P2 | P2 |
| Coverage Compiler, 테이크, 분기 블록 우주 | P2 | P2 |
| Factorized Turing Battery, BUPS Arena, TSR/CSI | P2 | P2 (Appearance Turing은 P1 평가에도 사용) |
| Native WFM T3 프런티어 | P2 | 장기 |
| 실시간 인터랙티브, 게임, 인터랙티브 영화 | P2 | 장기 |
| 로보틱스·피지컬 AI 시뮬레이터 시장 | P2 | 장기 |
| Atmos 객체 스템, 마이크 세계선 | P2 | 장기 |
| 촬영감독 스타일·손 라이선스, 퍼포먼스 캡처 마켓플레이스 | Cine, P2 | 장기 |
| 모션컨트롤 로봇 export, LED 월 없는 버추얼 프로덕션 | Cine | 장기 |
| 재조합 엔진 $10^9$ 규모, Builder–Reader self-play 스케일 | P2 | 장기 |
| 자체 이미지 교사 | (신규 제안) | 장기 |

## 부록 B. 신규성 장부 (확신도)

| 확신도 | 항목 |
|---|---|
| **높음** | (1) PSH + Scale-Time Duality v2 + 구조 주소 순수성. (2) 디노이징 루프 안의 σ-대역 분할 소유권(T5 Spectral Deficit와 T6 Partition Principle을 결합)과 band-isometric 이미지 교사 판정. (3) T4± 양방향 경로 이식. (4) BLI를 생성 비디오 지표로 사용. (5) Temporal DoF / 랙 타임. (6) AV-Physics Timing Solver(β로 sync style 판별). (7) Event Type Contract 기반 사건 집합 생성과 실현된 사건을 진실로 삼는 원칙 |
| **중상** | WNF를 장면·다중 LOD의 ρ-레버로 쓰는 것, DTC(렌즈 설명소거), Ray-Bundle Demand + 승격 면제, ShotDNA v2 식별 분할, CRM, Edit-Closed World, MAN 셰이딩, RGF + Reality Residual Budget, Dreamer Calibration / Worldness Gap, Spectral Federalism, WDM 장면 규모 amortization(수렴 리스크는 X3가 반증) |
| **중** | Dual-Anchored Appearance(RAS 정본), Directed World, PIS 2.0, Hero FX + sim-in-the-loop 아트 디렉팅, 빈 사분면 프레이밍, adversarial camera 감사, Milestone SMC 조합, SMF, OIE |
| **부품 (정직하게 차용)** | Barooah–Hespanha drift 이론, DMD/DMD2, VSD, SyncTweedies, MultiDiffusion, DiffCollage, Warped Noise 계열, Diffusion Forcing, TRELLIS/XCube/VAR, mip/Mip-Splatting/H3DGS/Octree-GS, 3DGRT/3DGUT/3DGRUT, SGGX, phasor/Gabor noise, 글린트, Kolb/Hullin/Newson 카메라 부품, Virtual Cinematographer, Many-Worlds Browsing, BEDLAM 재조합, Spectral Gaussians(Generative Image Dynamics), 공유 바람장, GeCo/MEt3R/WorldScore/VBench-2.0/Physics-IQ, DA3/VGGT-Ω/MoGe/MegaSaM, Difix3D+, RGB↔X/DiffusionRenderer, PhysGaussian/NGFF/Newton/MuJoCo-Warp, Zheng & James 음향 |

> **마무리.** 비디오 디퓨전은 같은 세계를 240번 다시 상상하면서 그 상상들이 우연히 일치하기를 바란다. 우리는 세계를 한 번, 카메라가 분해할 수 있는 순서대로 쌓는다. 이미지 디퓨전은 그 세계를 대역마다 판정하고, 결정 직전의 주사위까지 세계 표면에 새겨 둔다. 렌즈와 센서는 그 세계를 읽기만 한다. Phase 1에서는 멈춘 세계에서 예쁘기(SP)와 정직하기(MI)를 동시에 이긴다. 두 축을 함께 이긴 시스템은 아직 없다. Phase 2에서는 그 세계에 사건을 일으키고, 그 사건을 찍는다. 그리지 말고, 찍어라.
