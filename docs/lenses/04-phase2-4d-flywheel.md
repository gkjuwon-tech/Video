> **출처 메모.** 설계 워크숍에서 [Phase 2 / 4D / 데이터 플라이휠] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 원안(v1)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../PARADIGM.md)의 판결록(§0.5)을 따른다.

---

# Phase 2 설계서: Block Universe Cinema

### 움직이는 세계를 만들고, 시공간 안에서 촬영한다

> 렌즈: Phase 2(움직이는 세계) 설계자 겸 데이터/스케일링 전략가
> 기준: 2026년 중반까지 공개된 연구를 기준으로 했다. 그 뒤에 나온 연구와는 겹칠 수 있다.
> 표기: 굵은 글씨의 **[신규]** 는 공개된 선행 사례를 찾지 못했고, 우리가 처음일 가능성이 높다고 판단한 아이디어다. 확신도도 같이 적었다.

---

## 0. TL;DR

1. **정식화.** Phase 2는 Phase 1에 시간 차원 하나를 더한 것이다. 단, 시간은 **외형(appearance)에 붙지 않고 상태(state)에만 붙는다.** 수백만 개 primitive의 외형 텐서는 시간이 지나도 바뀌지 않는다. 시간에 따라 변하는 것은 수천~수만 차원의 물리 상태뿐이다. 비디오는 이 4D 블록을 카메라 세계선(worldline) γ(τ)을 따라 셔터 시간만큼 적분한 결과다.
2. **설계 원칙: "확률은 원인에, 결정론은 결과에" (Stochasticity at causes, determinism at consequences).** 디퓨전 비디오는 확률을 결과물인 픽셀에 둔다. 3대 고질병은 모두 여기서 나온다. 우리는 확률을 두 군데에만 둔다. ① 무엇이 존재하는가(세계), ② 무엇이 언제 일어나는가(사건). 나머지는 물리와 렌더링, 즉 결정론적 함수가 처리한다.
3. **[신규] Event-First Generation.** 움직임을 생성할 때 생성 변수는 프레임도 궤적도 아니다. **시공간 marked point process**, 즉 연속 시간 좌표 위의 사건 집합이다. 사건은 접촉, 충격, 파괴, 발화, 의도 변경 같은 것들이고, 사건과 사건 사이는 시뮬레이터가 채운다. 움직임의 날카로움, 소리 싱크, 편집 가능성은 모두 이 구조에서 나온다.
4. **[신규] 증인 교차신문(Cross-Witness Interrogation).** 비디오 모델을 경쟁자가 아니라 증인으로 쓴다. 우리 4D 세계를 우리가 고른 각도에서 렌더해 비디오 모델에게 보여주고, 비디오 모델의 모션 제안을 4D로 lift한다. 그 제안은 물리 판사와 다른 각도의 증인으로 교차 검증한다. 어느 각도에서 물어볼지는 **정보 이득(information gain)이 최대가 되도록 능동 선택**한다.
5. **로드맵.** 2a 살아있는 정지세계 → 2b 강체·관절체 → 2c 캐릭터·연기 → 2d 완전한 4D와 인터랙션. 단계마다 두 축에서 승리 조건을 정의한다. 하나는 비디오 모델이 구조적으로 통과할 수 없는 시험(Bullet-Time, Re-shoot, Slow-mo, Revisit)이고, 다른 하나는 공정한 블라인드 Arena다.
6. **데이터.** 인터넷 비디오를 (세계 W, 카메라 세계선 γ, 퍼포먼스 P, 사건 E)로 분해한 뒤 **곱(product)으로 재조합**한다(Factorized Recombination Engine). 학습은 Builder–Reader self-play와 검증 가능한 보상(verifiable reward)을 쓴 RL로 스케일한다. **[신규]** 오디오 onset으로 사건 타이밍에 서브프레임 라벨을 붙이고, 4D 재구성에서 얻은 거리로 음속 지연을 보정한다.
7. **스케일링 가설.** 일관성·물리 축에서는 같은 품질에 비디오 모델의 1/10 이하 컴퓨트로 도달한다(Factorization Dividend κ ≥ 10). 또 우리 검증기는 객관적이므로 test-time compute 스케일링이 비디오 모델보다 훨씬 잘 먹힌다.

---

## 1. 정식화: Phase 2 = Phase 1 + 시간 차원

### 1.1 핵심 등식

Phase 1(정지 세계):
$$V(\tau) = \mathcal{R}\big(W,\ \Gamma(\tau)\big)$$

Phase 2(블록 우주):
$$V(\tau, \mathbf{u}) = \frac{1}{S(\tau)}\int_{t(\tau)-S/2}^{\,t(\tau)+S/2} \mathcal{R}\Big(W\big(t' + \rho(\mathbf{u})\big),\ \Gamma(\tau)\Big)(\mathbf{u})\ dt'$$

- $\mathbf{u}=(u,v)$는 픽셀 좌표다. $\rho(\mathbf{u}) = (v/H - \tfrac12)\,T_{\text{readout}}$는 rolling shutter 때문에 생기는 행별 시간 오프셋이다.
- $\Gamma(\tau) = (T_{wc}\in SE(3),\ K,\ d_f,\ N,\ \text{distortion},\ \text{breathing})$는 자세, 내부 파라미터, 초점거리, 조리개, 렌즈 특성이다.
- $t(\tau)$는 **카메라 시간 τ를 세계 시간 t로 바꾸는 사상**이다. $dt/d\tau=1$이면 정상 속도, $0<dt/d\tau<1$이면 진짜 슬로모션, $=0$이면 bullet time, $<0$이면 역재생이고, 불연속이면 컷이다.
- 따라서 **카메라 세계선** $\gamma(\tau) = (t(\tau), \Gamma(\tau), S(\tau))$는 시공간 안의 경로다. Phase 1의 "카메라는 경로만 지정하면 된다"는 원칙이 시간 축까지 그대로 확장된다.

세계가 시간에 따라 전개되는 방식:
$$W(t) = \mathrm{Deform}\big(W_{\text{canon}},\ s(t)\big),\quad s(\cdot) = \mathrm{Sim}\big(s_0, a(\cdot); \psi\big),\quad a(\cdot) = \pi(\cdot \mid E, z),\quad (E, z) \sim p_\theta(E, z \mid W_{\text{canon}}, c)$$

**생성 체인:**

```
c (스크립트/프롬프트)
 └→ W_canon        [확률]  Phase 1 세계 + 객체 중심 에셋 + 물성 + 리그
     └→ (E, z)     [확률]  사건 집합 + 퍼포먼스 latent
         └→ a(t)   [결정]  제어 (정책/컨트롤러)
             └→ s(t)  [결정]  물리 상태 (시뮬레이터)
                 └→ W(t)  [결정]  변형된 세계
                     └→ V  [결정]  카메라 세계선 γ로 촬영 (셔터 적분)
```

확률적인 단계는 두 개뿐이다. 이 구조가 3대 고질병을 하나씩 원인 수준에서 없앤다.

| 고질병 | 디퓨전 비디오에서 생기는 이유 | 블록 우주에서 사라지는 이유 |
|---|---|---|
| (1) 일관성 붕괴 | 외형을 매 프레임 확률적으로 다시 그린다 | 외형 텐서가 시간 불변이므로 identity는 정의상 바뀔 수 없다 |
| (2) 뭉개짐·녹아내림 | 형상 변화에 어떤 제약도 없다 | 형상 변화는 명시적인 물리 과정(파괴 사건, MPM 소성 변형)으로만 일어난다. 녹으려면 "녹는 사건"이 있어야 하므로 우연히 녹을 수 없다 |
| (3) 과한 부드러움 | temporal latent 압축, 사후 평균, 거리 학습 | 사건 타이밍을 명시적으로 샘플링하고, 접촉은 솔버가 충격으로 계산하며, 중력은 metric 단위다(§2) |

### 1.2 4D 세계 W의 텐서 구조

| 구성요소 | 텐서 | 형상(예시) | 시간 의존 |
|---|---|---|---|
| 정적 배경 Gaussians | $G_{bg}$: μ(3), q(4), log s(3), α(1), 외형 특징 f(C) | $N_{bg}\times(11+C)$, 예: 5M × 59 | 없음 |
| 내재 채널(intrinsics) | albedo(3), normal(3), roughness(1), metallic(1) | $N_{bg}\times 8$ | 없음 |
| 객체 에셋 k (canonical) | $G_k$ | $N_k \times (11+C)$ | 없음 |
| 리그 | 본 $B_k$, rest pose $\in SE(3)^{B_k}$, skin weight(top-4 sparse) | $N_k\times 4\times 2$ | 없음 |
| 물성 | $\psi_k$ = (ρ, μ_s, μ_d, e, E, ν, damping, 재질 클래스) | $\approx 12$ | 파괴 사건이 일어날 때만 바뀜 |
| 충돌 프록시 | SDF 또는 convex decomposition | sparse | 없음 |
| **상태 궤적** | $s_k(t)$ | $T_{sim}\times d_k$ (강체 13, 인체 ~330+얼굴 100) | **있음** |
| 변형체 시뮬 상태 | 천 $V_c\times 6$, 머리카락 $S\times P\times 6$, MPM 입자 $P\times 16$ | | **있음** |
| 신경 잔차 변형 | $\delta_k(\mathbf{x}_c, s_k(t))$ (작은 MLP 또는 triplane) | 파라미터 | 상태의 함수 |
| 장(field) | 바람 $\mathbf{u}(\mathbf{x},t)$ = 스펙트럼 파라미터, 유체·연기 sparse voxel | | **있음** |
| **사건 집합** | $E = \{e_i\}_{i\le M}$ (§2.3) | $M\times(\sim 64)$ | 시간 좌표 그 자체 |
| 광원 | $L_j(t)$: 위치, 세기, 색, 방향 | $J\times 10$ | **있음** |
| 카메라 세계선 | $\gamma(\tau)$ | $T_{out}\times \sim 20$ | 촬영 변수 |

### 1.3 "시간은 상태에만 붙는다": 압축비

- 순진한 4DGS(프레임마다 Gaussian 파라미터를 저장): 1M Gaussian × 59 float × 300프레임(10s@30fps) ≈ **1.8×10¹⁰ float**
- 우리 방식: 1M × 59(시간 불변) + 에이전트 10개 × 300 × 300 + 천 정점 20k × 6 × 300 ≈ **9×10⁷ float**
- 약 **200배** 작고, 시간에 따라 변하는 부분은 모두 물리적 의미가 있다. 생성 모델이 다뤄야 하는 시간 의존 변수의 차원이 이렇게 작다. Phase 2를 합리적인 컴퓨트로 풀 수 있는 근거가 바로 이것이다.

### 1.4 블록 우주에서 공짜로 얻는 것들

**(a) Coverage Compiler: 멀티캠 커버리지를 자동으로.** 4D blocking에서 두 인물의 머리 위치를 이어 action line(180° 라인)을 계산한다. 그 한쪽 반공간에서 표준 커버리지를 자동 배치한다. master wide, 투 숏, OTS(over-the-shoulder) A/B, CU A/B, 인서트가 기본 세트다. eyeline match는 3D gaze 벡터가 있으므로 정의상 맞는다. 컷과 컷 사이 연속성(손에 든 컵의 물 높이, 담배 길이)도 하나의 세계에서 나오므로 0 비용이다. 결과물은 ProRes 멀티캠과 편집 XML/EDL로 내보낸다.

**(b) 테이크(Take).** 퍼포먼스 latent $z$만 다시 샘플링하고 blocking 제약(시점 t에 마크 위치 x)은 고정한다. 정지 세계, 조명, 의상은 비트 단위로 같다. 감독 디렉션("더 망설이듯이")은 사건·의도 공간에서 classifier-free guidance로 반영한다.

**(c) 분기 우주(Branching Block Universe).** 시점 $t_b$의 상태 $s(t_b)$는 명시적이고 완전하다. 그래서 $W^{(a)}, W^{(b)}$가 $[0,t_b]$를 공유하는 트리를 만들 수 있다. 새 분기를 만드는 비용은 $t_b$ 이후의 사건 생성과 시뮬레이션뿐이다. what-if 편집, 인터랙티브 영화, 게임으로 바로 이어진다.

**(d) 시간 리맵과 셔터 룩.** 진짜 슬로모션은 보간이 아니다. 시뮬레이터가 내부적으로 240~1000Hz로 돌기 때문에 실제 동역학이 그대로 나온다. bullet time, speed ramp도 공짜다. 셔터 각도도 스타일 변수로 쓸 수 있다. 180°는 필름 룩, 45°는 「라이언 일병 구하기」 같은 스타카토 액션 룩이다. dolly zoom은 $f(\tau) \propto \|x_{cam}(\tau)-x_{subj}\|$로 정확하게 계산된다.

**(e) 우리만의 카메라.** Γ는 꼭 pinhole일 필요가 없다. 열마다 다른 시간을 보는 slit-scan(time-slice) 카메라, multi-perspective 카메라, 벽을 통과하는 원테이크(「1917」, 「버드맨」 스타일) 경로처럼 물리적 카메라로는 찍을 수 없는 문법도 그냥 경로 지정 문제가 된다.

### 1.5 이것이 hole-filling이 아닌 이유

창업자가 거부한 방식은 이렇다. 카메라 뷰 기준으로 빈 곳을 순차적으로 그린다. 그러면 생성 결과가 카메라 경로에 의존하게 되고, 본질적으로 프레임 생성과 같다.

블록 우주에서는 모든 것이 모든 시간에 이미 존재한다.

- **세트는 촬영 전에 전부 짓는다.** 컵을 들어 올리면 드러나는 테이블 면, 문이 열리면 보이는 옆방, 서랍 안쪽은 모두 세계를 만들 때(카메라가 존재하기 전에) 전체 장면을 문맥으로 해서 3D world space에서 한 번에 생성된다. 드러나는 것(reveal)은 그냥 드러나는 것이지 새로 지어내는 것이 아니다.
- **시간 방향 전개도 픽셀 캔버스가 아니라 완전한 상태에 대한 물리 적분이다.** 현실 자체가 이렇게 굴러간다. 또 사건 계획은 전체 시간 구간에서 동시에 생성된다(§2.3). 원칙은 "계획은 전체(블록)로, 실현은 인과적으로(물리)"이다.

### 1.6 Phase 1에 대한 역요구사항

Phase 1 세계는 처음부터 움직일 준비가 되어 있어야 한다. Phase 1 팀에게 다음 여섯 가지를 계약으로 요구한다.

1. **Metric scale(m 단위).** 중력, 관성, 음속 지연 계산에 모두 필요하다.
2. **객체 중심 scene graph.** 움직일 수 있는 객체는 독립된 에셋이어야 하고, 뒷면과 아랫면까지 완전한 amodal geometry를 가져야 한다. 객체 밑의 지지면도 완전해야 한다.
3. **내재 외형(intrinsics)과 광원 추정.** 적어도 움직이는 객체와, 그 그림자를 받는 표면에는 있어야 한다.
4. **SimReady 태그.** 재질 클래스, 밀도, 마찰, 관절축(문 경첩, 서랍 레일)이 필요하다.
5. **충돌 프록시**(SDF 또는 convex).
6. **어포던스 맵**(앉기, 잡기, 걷기, 기대기).

---

## 2. 움직임은 픽셀이 아니라 원인의 공간에서 생성한다

### 2.1 과한 부드러움의 구조적 원인

1. **Temporal latent 압축.** Wan 2.1, HunyuanVideo 계열 VAE는 시간 방향으로 4배 압축한다. 24fps라면 latent 시간 해상도는 6Hz이고, latent 시간 축의 Nyquist 주파수는 **3Hz**다. 그보다 빠른 시간 성분은 별도의 latent 시간 샘플을 갖지 못하고 채널 차원에 접혀 들어간다. 디퓨전 모델은 이것을 약한 채널 간 상관으로 학습해야 하고, 그래서 denoising과 distillation 과정에서 가장 먼저 사라진다. 정보가 원리적으로 0이 되지는 않으므로 "반드시 소실된다"고까지 말하지는 않겠다. 다만 가장 취약한 성분인 것은 분명하다.
2. **사후 평균 회귀(posterior mean regression).** 높은 noise 수준에서 denoiser의 출력은 $\mathbb{E}[x_0\mid x_\sigma]$다. 충돌 타이밍이 다봉 분포라면 평균은 "번진 충돌"이 된다. 실제 서비스는 4~8 step으로 distill된 sampler와 CFG를 쓰기 때문에, 고주파 시간 성분에서 모드 평균화가 남는다.
3. **손실 가중.** 접촉은 픽셀 면적 × 시간으로 보면 아주 작은 영역이다. 산 하나와 공 하나의 충돌이 L2 손실에서 같은 무게를 받을 수 없다.

**스펙트럼 논증.** 시각 $t_c$에 속도가 $\Delta v$만큼 점프하면 위치 궤적에 꺾임(kink)이 생기고, 그 Fourier 꼬리는 $|X(\omega)|\sim |\Delta v|/\omega^2$로 멱함수 감쇠한다. 반면 저주파 latent를 매끄럽게 보간한 궤적은 그보다 빠르게(지수적으로) 감쇠한다. 과한 부드러움은 이렇게 **측정 가능한 스펙트럼 결손**이다. 이를 재는 지표가 §4.0의 TSR이다.

우리의 대응은 위 원인과 하나씩 짝지어진다.
(1) 사건은 연속 시간 좌표를 갖고, 시뮬레이터는 240~1000Hz로 돈다. 프레임 격자가 아예 없다.
(2) 타이밍은 저차원 공간에서 **100 step 이상으로 명시적으로 샘플링**한다. 토큰이 1k 이하라서 이 비용은 무시할 만하고, 따라서 distillation이 필요 없다. 그래서 하나의 모드를 확정한다.
(3) 사건이 일급 토큰이므로 충돌 하나와 산 하나가 같은 토큰 무게를 갖는다.

중력에 대해서도 말해 둔다. 세계가 metric이므로 $g=9.81\,\mathrm{m/s^2}$가 보장된다. 1m 자유낙하는 0.45s가 걸린다. 비디오 모델의 "둥둥 뜨는(floaty)" 낙하나 미니어처처럼 보이는 효과는 대부분 스케일 오판 때문이고, 우리에게는 원인부터 없다.

### 2.2 생성 변수의 계층

| 층 | 변수 | 차원/빈도 | 생성 방식 |
|---|---|---|---|
| L4 의도·스토리 | 장면 비트, 인물 목표, 감정 곡선 | 수십 토큰 | LLM 플래너 + 텍스트 |
| **L3 사건** | $E$: 접촉, 충격, 파괴, 발화, 발화(utterance), 시선 전환, 의도 전환 | 수십~수천 개, 연속 시간 | **EPD (§2.3)** |
| L2 제어 | 관절 토크/목표, 외력, 방출원 파라미터 | 30~120Hz | 정책 $\pi$, 컨트롤러 |
| L1 상태 | $s(t)$ | 240~1000Hz | 시뮬레이터 (결정론) |
| L0 외형 | $G$ | 시간 불변 | Phase 1 + 에셋 |

### 2.3 [신규, 확신도 높음] Event-Point-Process Diffusion (EPD)

**표현.** 사건 슬롯 $M_{max}$개(예: 1024)를 두고, 각 슬롯은 다음과 같다.
$$e_i = \big(b_i,\ t_i,\ \mathbf{x}_i,\ k_i,\ p_i,\ q_i,\ \mathbf{m}_i\big)$$
- $b_i\in\{0,1\}$: 슬롯 존재 여부. 사건 개수 자체가 확률 변수다.
- $t_i\in[0,T]$: **연속 시간.** 프레임 격자가 없다.
- $\mathbf{x}_i\in\mathbb{R}^3$: 세계 좌표.
- $k_i$: 타입. `CONTACT_ON/OFF, IMPULSE_EXT, FRACTURE, IGNITE, EXTINGUISH, SPLASH, GRASP, RELEASE, FOOTSTRIKE, GAZE_SHIFT, UTTERANCE_ON/OFF, INTENT(goal), EMOTION_SHIFT, LIGHT_ON/OFF, GUST` 등.
- $p_i, q_i\in\{0..K\}$: 참여자 포인터. 0은 환경이다.
- $\mathbf{m}_i\in\mathbb{R}^{d_m}$: 마크. 충격량 $\mathbf{J}\in\mathbb{R}^3$, 상대속도, 재질쌍 임베딩, 의도·감정 임베딩(텍스트 임베딩을 투영한 것), 강도가 들어간다.

**모델.** 사건 토큰끼리는 self-attention을 하고, 세계 토큰(정적 청크, 에셋, 어포던스)과 텍스트에는 cross-attention을 한다. $t_i$는 Fourier feature로 임베딩한다. 사건들 사이의 self-attention이 인과 구조를 학습한다. 예를 들어 A가 밀면 0.4초 뒤 B가 넘어지고, 넘어지면 소리가 나고, C가 그쪽을 돌아본다.

**노이징과 손실.** 연속 성분 $(t,\mathbf{x},\mathbf{m})$에는 flow matching을, 이산 성분 $(b,k,p,q)$에는 masked(absorbing) diffusion을 쓴다.
$$\mathcal{L} = \mathbb{E}\Big[\lambda_c\, b_i\,\|\hat{x}^{c}_0 - x^{c}_0\|^2 + \lambda_k\,\mathrm{CE}(\hat k,k) + \lambda_b\,\mathrm{BCE}(\hat b,b) + \lambda_p\,\mathrm{CE}(\hat p,p)+\lambda_q\,\mathrm{CE}(\hat q,q)\Big]$$
학습할 때는 시간순으로 정렬해 슬롯에 배치하지만, 모델 자체는 순서 불변 구조다.

**조건과 제약.** 스크립트 제약("t=8s에 유리잔이 깨져야 함", "t=5s에 배우가 문 앞 마크에")은 해당 슬롯을 고정하는 inpainting으로 준다. 이것은 사건 집합 공간에서의 조건부 생성이지 픽셀 캔버스가 아니다. 이것으로 Witkin & Kass(1988)의 Spacetime Constraints가 생성 모델 안에서 다시 살아난다. 샘플링 단계에서는 실현 가능성 critic $F_\phi(E,W)$로 guidance를 준다. 이 critic은 시뮬레이터에서 "이 사건 집합이 실현 가능했나"를 라벨로 삼아 학습한다.

**왜 새로운가.** 모션 디퓨전(MDM, CLoSD)은 궤적이나 프레임 격자 위에서 생성한다. 시공간 point process를 다루는 연구는 지진학이나 신경 스파이크 분야에 있다. 하지만 **모든 객체 타입과 모든 모달리티(시각, 소리, 편집)를 하나로 묶는 통화로서 사건 집합 디퓨전을 비디오 생성의 1차 latent로 쓴 사례는 내가 아는 한 없다.**

### 2.4 Event–Simulation Alternation (ESA): 계획과 실현의 정합

```
입력: W_canon, c, 시간 구간 [0, T]
E ← sample p_θ(E | W_canon, c)                       # 감독의 계획
for k = 1..K_max:
    a ← Controllers(E, z)        # INTENT/GOAL → 정책 목표, IMPULSE_EXT → 외력
    s, Ê ← Sim(s_0, a; ψ)        # Ê: 시뮬레이터에서 실제로 일어난 사건 (자동 검출)
    Δ, t* ← Match(E_phys, Ê)     # (t, x, type, 참여자)에 대한 Hungarian 매칭, 최초 불일치 시각
    if Δ < ε: break
    A) 수리: a[t*-δ, T] ← argmin_a Δ(E, Sim(a))         # diff-sim(Warp) 또는 CEM
    B) 재계획: E ← Ê[0,t*] ∪ sample p_θ(E(t*,T] | Ê[0,t*], W_canon, c)
return s, Ê      # 진실은 시뮬레이터가 실현한 사건이다
```

- B는 MPC(model predictive control)식 재계획이다. 이미 일어난 일은 물리가 확정하고, 그 뒤만 다시 계획한다.
- **최종 출력은 계획된 사건 E가 아니라 실현된 사건 Ê다.** 생성 모델은 감독이고 판사는 물리다.
- 겹치는 연구: CLoSD(Tevet et al., 2025)는 인체에 대해 diffusion planner와 RL tracking을 closed loop로 묶었다. ESA는 이를 강체, 유체, 천, 소리까지 일반화하고, 계획 공간을 사건 집합으로 바꾼 것이다.

### 2.5 물질별 동역학 모듈

| 대상 | 상태 표현 | 엔진 | 생성되는 것 | 겹치는 기존 연구 |
|---|---|---|---|---|
| 강체 | SE(3)+속도 (13) | Newton(NVIDIA·DeepMind·Disney, 2025), MuJoCo-Warp | 외력·초기조건 사건 | PhysGen (ECCV'24) |
| 관절체(문, 차량, 기계) | root + joint q | 같음 | 관절 구동 사건 | |
| 연체·입상체·파괴 | MPM 입자, Gaussian과 1:1 | GPU MPM | 재료 파라미터, 충격 사건 | PhysGaussian (CVPR'24), PhysDreamer (ECCV'24) |
| 천 | 메시 정점 + 속도, 삼각형마다 Gaussian을 barycentric 바인딩 | XPBD 또는 projective dynamics | 재단 패턴/재질 | Gaussian Garments 계열 |
| 머리카락 | guide strand(Cosserat rod) + 보간 | XPBD | 헤어스타일 에셋 | GaussianHair |
| 액체 | FLIP/SPH + 표면 Gaussian, 원거리 바다는 Tessendorf FFT 스펙트럼 | | 방출원, 경계 | Tessendorf |
| 연기·불 | sparse voxel 밀도/온도 + 3D wavelet turbulence 디테일 | Eulerian | 방출원, 온도 | Kim et al. 2008, tempoGAN |
| 식생 | **Spectral Gaussians** + 공유 바람장 (§4.1) | modal | 바람 스펙트럼 | Generative Image Dynamics (CVPR'24)의 3D 확장 |
| 비물리 현상(마법, VFX, 변신) | **Free 4D field**: world-space 4D Gaussian 생성 | 없음 | 4D 디퓨전 직접 생성 | 4DGS, L4GM |

마지막 행은 중요한 안전장치다. 시뮬레이터로 덮을 수 없는 현상은 물리 보장을 포기한다. 하지만 world space 안에 있으므로 **3D 일관성은 여전히 정의상 보장된다.** 표현은 하나의 스펙트럼이다: 물리 시뮬레이터 → 학습된 emulator(GNS, MeshGraphNets 계열) → free 4D field. 세 경우 모두 촬영 원리는 같다.

### 2.6 캐릭터 스택

- **골격.** 인체는 SMPL-X 계열(55관절, 6D 회전)을 쓴다. 동물과 크리처는 SMAL을 쓰거나 리깅 자동화(UniRig, Make-It-Animatable 계열)로 임의 형태에 대응한다.
- **몸.** Gaussian을 템플릿 메시에 LBS로 바인딩하고, 포즈 의존 신경 잔차 $\delta(\mathbf{x}_c,\theta)$를 더한다(Animatable Gaussians, 3DGS-Avatar 계열).
- **얼굴.** FLAME식 표정 $\psi\in\mathbb{R}^{100}$ + 턱/안구/눈꺼풀, GaussianAvatars식 바인딩, 주름용 동적 신경 텍스처를 쓴다. **눈은 별도의 굴절 모델(각막 + catchlight)로 만든다.** 사람이 실사감을 판단할 때 가장 먼저 보는 곳이 눈이다.
- **의상.** 앞의 천 시뮬레이션을 쓰고, 몸 SDF와 충돌시킨다. 젖은 옷은 MPM과 결합한다.
- **제어.** 물리 시뮬 캐릭터에 생성형 tracking 정책(PHC, MaskedMimic 계열)을 붙이고, kinematic 계획은 EPD 사건과 의도에 조건화된 모션 디퓨전이 낸다. 얼굴과 발화는 스크립트 → TTS → 음소 타이밍 → viseme과 표정으로 이어진다. 소리와 입이 같은 타임라인에서 나온다.
- **퍼포먼스 latent $z$.** 인물별, 테이크별 감정과 에너지 곡선 및 미세 습관(tics)을 담는다.
- **Avatar Foundry.** 이미지 디퓨전으로 다시점, 다표정 이미지 세트를 생성한 뒤 아바타를 fitting한다. 창업자가 예측한 대로 생성된 뷰끼리 미세하게 불일치한다. 이것은 **뷰마다 appearance code를 두고 fitting한 뒤(NeRF-W 방식) 렌더할 때는 그 코드를 버리는 방식**으로 흡수한다. 불일치는 코드가 흡수하고, 공통된 것만 아바타에 남는다.

### 2.7 날카로움을 지키면서 모션블러는 정확하게

모션블러는 셔터 적분으로 계산한다. 속도 $\mathbf{v}$로 선형 운동하는 Gaussian을 셔터 $S$ 동안 적분하면 근사적으로 다음 Gaussian이 된다.
$$\Sigma_{\text{blur}} = \Sigma + \tfrac{S^2}{12}\,\mathbf{v}\mathbf{v}^\top$$
여기서 $S^2/12$는 구간 $[-S/2, S/2]$ 균등분포의 분산이다. 회전 운동(바퀴, 프로펠러)에는 sub-frame 샘플 8~16개를 쓴다. 비디오 모델은 블러를 텍스처처럼 흉내 낸다. 우리는 블러를 **물리량(속도 × 셔터)으로 계산**한다. 그래서 충돌 순간 앞뒤 프레임의 블러 방향이 정확히 뒤집힌다. 날카로움과 블러가 동시에 사실적인 것은 이 구조에서만 가능하다.

### 2.8 조명 상호작용: 결정론적 G-buffer neural shader

움직이는 객체는 정적 세계에 그림자와 반사를 만든다. 굽힌 radiance만으로는 부족하다. 그래서 세 겹으로 처리한다.

1. **정적 부분**은 baked radiance를 쓴다. Phase 1 품질을 그대로 유지한다.
2. **움직이는 객체와 그 영향 영역**은 intrinsic 채널과 추정 광원으로 deferred shading을 하고, 그림자와 반사는 Gaussian ray tracing(3DGRT, 3DGUT 계열)으로 계산한다.
3. **잔차는 neural shader가 보정한다.** 이 shader의 조건은 세 가지다.
   - (i) 입력은 4D 세계의 G-buffer(깊이, 법선, albedo, 객체 ID, world-space feature)뿐이다.
   - (ii) 수용영역이 제한된 결정론적 함수다. 샘플링을 하지 않는다.
   - (iii) 출력은 shading 잔차뿐이다.

이 shader는 부분 completion처럼 보일 수 있다. 하지만 **콘텐츠를 지어낼 자유도가 없다.** 같은 세계 상태와 같은 시점이면 같은 결과가 나오는 함수라서 일관성이 구조적으로 깨지지 않는다. 그래서 거부된 hole-filling과는 원리가 다르다. intrinsic 분해 학습의 교사로는 RGB↔X(SIGGRAPH'24)와 DiffusionRenderer(CVPR'25)를 쓴다.

---

## 3. 비디오 모델을 인턴으로 쓴다: PLAC 프로토콜과 증인 교차신문

### 3.1 Propose–Lift–Adjudicate–Commit (PLAC)

```
입력: 현재 세계 W[0,t0], 의도 c, 후보 수 M, 증인 수 N
1) PROPOSE
   γ_lead ← argmax_γ [행위 참여자의 가시 면적 − 가림]
   cond  ← R(W[t0-2s, t0], γ_lead)                     # 우리 세계를 렌더해서 보여준다
   {V̂_m} ← p_vid(· | cond, c),  m=1..M                  # 480p 저해상 초안이면 충분하다
2) LIFT   (알려진 모델에 대한 저차원 fitting)
   추적 초기화: cond 프레임의 픽셀↔Gaussian 대응을 우리가 이미 안다 → identity가 완벽한 트랙
   CoTracker3 등으로 생성 프레임까지 추적
   ŝ_m = argmin_s Σ_t Σ_p ρ(π(γ, X_p(s(t))) − u_p(t)) + λ_phys R_phys(s) − λ_prior log p_motion(s)
   Ê_m ← Detect(ŝ_m)
3) ADJUDICATE
   (a) 물리 판사: 시뮬 tracking 컨트롤러로 ŝ_m을 추종시킨 뒤
       ghost force  F_ghost = min_{λ,τ} ‖M(q)q̈ + h(q,q̇) − Jᵀλ − Sτ‖   (residual force control 개념)
       → 관측된 운동을 만드는 데 필요한 '유령 힘'이 클수록 비물리적
   (b) 설명되지 않은 잔차: r(u,t) = ‖V̂_m − R(W⊕ŝ_m, γ_lead)‖_feat
       큰 잔차 덩어리는 VLM이 분류
         - 환각 → 버린다
         - 새 엔티티(누가 들어옴) → 3D 에셋 생성을 요청하고 lift된 위치에 삽입
         - 이펙트(연기, 불꽃) → 장/파티클 방출원 생성
   (c) 증인 교차신문 (§3.2)
4) COMMIT
   s* ← Sim tracking(argmax 가설)    # 물리 투영된 궤적이 진실
   W ← W ⊕ s*, E ← E ∪ Ê*
```

**Lift가 쉬운 이유.** 일반적인 monocular 4D 재구성은 어려운 문제다. 하지만 여기서는 형상, 외형, 리그, 카메라를 **이미 다 알고 있다.** 남은 미지수는 저차원 상태 $s(t)\in\mathbb{R}^{d}$($d$는 수십~수백)뿐이다. 그래서 IK나 PnP 수준의 잘 정의된 최적화가 된다. 비디오 모델에게는 픽셀 품질이 필요 없으므로 저해상, distill된 초안을 요청한다. 비용이 5~10배 줄어든다.

**인턴이 제안하고, 세계 빌더가 짓는다.** 비디오 모델이 "누가 문으로 들어온다"고 제안하면, 그 사람은 픽셀로 남지 않는다. 3D 에셋으로 생성되어 세계에 들어온다. 비디오 모델의 픽셀은 최종 결과물에 한 픽셀도 들어가지 않는다.

### 3.2 [신규, 확신도 높음] 증인 교차신문과 능동 카메라 배치

비디오 모델은 한 각도에서 보고 진술하는 증인이고, 환각을 한다. 우리에게는 그 증인을 **임의의 각도로 데려가 다시 물어볼 수 있는 4D 세계**가 있다.

**두 가지 모드.**

- **Critic 모드(ELBO 점수).** 가설 $h_m$을 시점 $\gamma_j$에서 렌더한 뒤, 비디오 모델의 디퓨전 손실로 그럴듯함을 잰다.
  $$\mathrm{score}_j(h_m) = -\mathbb{E}_{\sigma\in[\sigma_{lo},\sigma_{hi}],\,\epsilon}\Big[w(\sigma)\,\big\|\hat\epsilon_\phi(\mathcal{R}(W\oplus h_m,\gamma_j)_\sigma,\sigma,c)-\epsilon\big\|^2\Big]$$
  중간 noise 대역을 써서 모션 수준의 구조를 본다. **같은 시점, 같은 외형에서 모션 가설끼리만 상대 비교**하므로 ELBO의 텍스처 편향이 상쇄된다(Diffusion Classifier, Li et al. ICCV'23의 원리를 4D 검증에 적용한 것이다).
- **진술(testimony) 모드.** $\gamma_j$에서 우리 세계를 렌더한 조건 프레임을 주고 독립 제안을 받은 뒤, 그것을 lift해서 어느 가설을 지지하는지 본다.

**어느 각도로 물을지 정하기(query-by-committee over cameras).**
$$\gamma^* = \arg\max_{\gamma\in\mathcal{C}_{\text{valid}}} \sum_{m<m'} w_m w_{m'}\ D\Big(\mathcal{R}(W\oplus h_m,\gamma),\ \mathcal{R}(W\oplus h_{m'},\gamma)\Big)$$
$D$는 DINO feature 위의 트랙 차이다. 가설들이 가장 다르게 보이는 각도에서 물어보고, 사후 가중치를 갱신한다: $w_m \leftarrow w_m\exp(\beta\cdot \mathrm{score}_{\gamma^*}(h_m))$. 이를 2~3회 반복한다.

**예시.** 정면 증인은 "공이 컵을 맞혔다"고 말한다. 측면 카메라에서 렌더하면 공이 컵 5cm 앞을 지나간다. 측면 증인(critic)은 "맞혔다" 가설의 측면 렌더에 낮은 점수를 준다. 그 가설은 기각된다.

**새로운 점.** 비디오 모델을 SDS(score distillation)의 기울기 원천으로 쓰는 4D 생성(4D-fy, Dream-in-4D, Align Your Gaussians, PhysDreamer)은 이미 있다. 그러나 **(i) 비디오 모델을 이산 제안자와 스칼라 판정자로만 쓰고, (ii) 4D 세계가 증인의 시점을 능동적으로 선택하며, (iii) 물리가 최종 판사인 증거 판정 구조**는 내가 아는 한 없다. SDS의 고질병인 과포화와 평균화는 고차원 픽셀 기울기에서 생긴다. 우리는 이 구조로 그것을 원인 수준에서 피한다.

### 3.3 상태 공간 score distillation (보조 수단)

미세 조정이 필요할 때만 쓴다. ELBO 기울기를 저차원 $s$나 사건 마크 $\mathbf{m}$으로 역전파한다($\nabla_s \mathrm{score}$). 이때 매 step마다 물리 투영을 한다($s \leftarrow \mathrm{Sim}(\mathrm{track}(s))$). 차원이 수백 수준이므로 SDS의 분산 문제가 작다.

### 3.4 Amortization: 인턴의 졸업

1. **인턴기**: PLAC로 커밋된 (세계, 의도, 사건, 궤적) 쌍을 축적한다.
2. **감수기**: 그 데이터로 EPD와 모션 prior를 학습한다(expert iteration). 이제 우리 모델이 제안하고 비디오 모델은 critic만 맡는다.
3. **졸업**: 보상 = 물리 + 교차 증인 ELBO + 인간 선호로 RL fine-tuning을 한다(DDPO, Diffusion-DPO 계열). 비디오 모델 호출은 경계 사례로만 줄어든다.

**전략적 의미.** 경쟁 비디오 모델이 좋아질수록 우리 인턴도 좋아진다. **경쟁자의 진보가 우리의 연료가 된다.**

### 3.5 기존 연구와의 경계

| 기존 | 무엇을 하나 | 우리와의 차이 |
|---|---|---|
| Lyra (NVIDIA'25) | 비디오 디퓨전을 3DGS 디코더로 self-distill(4D 포함) | 비디오 latent를 진실로 삼는다. 우리는 물리와 교차 증인이 진실이다 |
| CAT4D, 4DiM, SV4D, 4Real, DimensionX, GenXD | 멀티뷰 비디오 생성 → 4D 재구성 | 움직임의 원천이 여전히 픽셀 디퓨전이다(평균화와 비물리성이 남는다) |
| Gen3C, ReCamMaster, TrajectoryCrafter | 3D 캐시나 비디오 재촬영 | 2D 생성기가 최종 픽셀을 만든다 |
| PhysDreamer, DreamPhysics | 비디오 prior로 재료 파라미터 최적화(SDS) | 단일 시점 SDS다. 능동 증인과 사건 구조가 없다 |
| Genie 3 | 인터랙티브 world model(프레임 생성) | 명시적 상태가 없어서 기억 지평에 한계가 있다. 우리는 상태가 완전하다 |
| Cosmos | 피지컬 AI용 비디오 world model | 픽셀 공간 world model이다 |

---

## 4. Phase 2 내부 로드맵과 "이긴다"의 측정

### 4.0 공통 평가 프로토콜

**블라인드 Arena.**
- 경쟁 상대는 측정 시점의 SOTA 전부다(Veo 3.x, Sora 2, Kling 2.x, Seedance, Runway, Wan 등).
- 경쟁자 쪽에는 4개 생성물 중 별도 평가자가 고른 best-of-4를 준다. 우리도 best-of-4로 맞춘다.
- 평가 축은 미감, 모션 사실성, 일관성, 물리, 연기 다섯 개다.
- 패널은 일반 평가자와 전문가 패널(촬영감독, VFX 슈퍼바이저, 배우/연출)로 나눈다.
- 프롬프트의 20%는 **비디오 모델에게 유리한 것**(스타일화, 군중 혼잡, 추상)으로 넣어 자기기만을 막는다.
- 승리 판정: 각 경쟁자 대비 승률의 95% CI 하한이 50%를 넘어야 하고, 단계별 목표 승률을 달성해야 한다.

**신규 정량 지표.**

| 지표 | 정의 | 겨냥하는 고질병 |
|---|---|---|
| **TSR** (Temporal Spectral Realism) | dense 2D 트랙(CoTracker3)에서 카메라 ego-motion을 보상한 속도 PSD를 Welch로 구한다. log-log 기울기 β와 HFER $=\int_{4}^{12\text{Hz}}P/\int_{0.5}^{12\text{Hz}}P$의 분포를, 같은 범주 실제 영상(60fps 이상 레퍼런스)과 Wasserstein-1로 비교 | (3) 과한 부드러움 |
| **CSI** (Contact Sharpness Index) | 충돌 검출 지점에서 $\max|a|_{\pm2\text{f}}/\mathrm{median}|a|_{\pm0.5s}$ 분포를 실제와 KL로 비교 | (3) |
| **MI** (Melt Index) | SAM2 마스크 → 깊이 lift → 객체마다 최적 강체(관절체면 ARAP) 정합 → 정규화 잔차. 강체로 라벨된 객체라면 0이어야 한다 | (2) 뭉개짐 |
| **BTT** (Bullet-Time Test) | 동작 중 무작위 시점 t*에 시간을 멈추고 90~180° 궤도를 돈다. MEt3R(CVPR'25)과 재투영 일관성으로 측정 | (1) |
| **RST** (Re-shoot Test) | 같은 사건을 반대 각도에서 다시 촬영하게 한다. lift한 3D 궤적을 Procrustes 정렬한 오차, 사건 타이밍 차이(프레임), identity 유사도 | (1) |
| **Revisit** | 카메라가 장소를 떠났다가 1/5/10분 뒤 돌아온다. 정렬한 뷰끼리 DreamSim 비교 | (1) |
| **g-check** | 자유낙하 구간을 metric lift한 뒤 맞춘 가속도의 상대오차 | (3) floaty |
| **GCM** (Gust Coherence) | 거리 d만큼 떨어진 식물 흔들림 신호의 상호상관 피크 지연과 $d/U_{wind}$ 비교 | 2a 사실성 |
| 인체 지표 | foot skate(cm/s, 접촉 중 수평 속도), 관통 깊이(mm), 손 무결성(MANO fit 잔차, 손가락 수), ArcFace identity(컷/각도 사이), LSE-C/D 립싱크, 컷 간 eyeline 오차(°) | 2c |
| 기존 벤치 | Physics-IQ (Motamed et al.'25), VideoPhy-2, PhyGenBench | 물리 |

**판 짜기 전략.** TSR, CSI, MI, BTT, RST, GCM은 **오픈 벤치마크로 먼저 공개**한다. 첫 보고서는 "Video Physics Spectrum Report"로, 모든 상용 모델의 스펙트럼 결손을 수치로 보여준다. 문제를 시장이 읽을 수 있게 만들면, 그 문제의 해법을 가진 우리가 기준을 정하는 쪽이 된다.

### 4.1 2a: 살아있는 정지세계 (Phase 1 이후 +3~6개월)

- **범위**: 물(바다, 강, 폭포, 잔물결), 불(촛불, 모닥불), 연기와 수증기, 식생과 풀밭, 구름과 구름 그림자, 비와 눈, 조명 깜빡임, 하루 시간 경과(time-lapse), 원경의 군중과 차량(ambient agent).
- **핵심 기술 ① Spectral Gaussians.** Gaussian $i$의 변위를 다음처럼 둔다.
  $$\mathbf{d}_i(t)=\mathrm{Re}\sum_{k=1}^{K}\mathbf{A}_{ik}\,e^{j(\omega_k t+\phi_{ik})},\quad \mathbf{A}_{ik}\in\mathbb{C}^3$$
  확률적 텍스처(물결, 불꽃)는 학습된 PSD $S_i(\omega)$와 시공간 공분산 커널을 갖는 Gaussian process에서 샘플링한다. 생성기는 영역별 PSD 파라미터만 예측한다(저차원). 그래서 반복되지 않는 무한 길이 운동이 가능하고, 루프가 필요하면 이산 주파수로 정확하게 루프를 만든다. Generative Image Dynamics(Li, Tucker, Snavely, Holynski, CVPR'24)의 2D spectral volume을 3D primitive 단위로 올린 것이다.
- **핵심 기술 ② [신규, 확신도 중간] 공유 바람장(shared wind field).** 모든 식생, 연기 기둥, 깃발, 머리카락, 물결을 **같은 3D 바람장** $\mathbf{u}(\mathbf{x},t)=\bar{\mathbf{U}} + \text{gust}(\mathbf{x}-\bar{\mathbf{U}}t)$가 구동한다. gust 전선은 바람의 속도로 이동한다. 그래서 밀밭에 돌풍이 파도처럼 번지는 모습이 생긴다. 비디오 모델이 만든 나무들은 서로 독립적으로 흔들린다. 실제 세계에서는 흔들림 사이에 공간적 지연 상관이 있다. GCM으로 측정할 수 있고, 같은 바람장이 바람 소리도 만든다(§6).
- **핵심 기술 ③.** 연기와 불은 sparse voxel 시뮬레이션에 world-space 난류 디테일 생성을 더한다. 디테일도 3D에서 생성하므로 일관성이 유지된다. 시간 경과는 relightable 정적 세계에 태양 궤적을 적용하고 구름 그림자를 ray tracing한다.
- **킬러 데모.** 60초 연속 드론 orbit. 노을 속 호숫가 마을에서 돌풍이 밀밭을 건너고, 굴뚝 연기가 같은 바람에 꺾이고, 호수에 잔물결이 인다. 돌풍 한가운데서 bullet-time으로 정지한 뒤 180° 돌고, 같은 카메라 경로로 밤까지 하이퍼랩스로 이어진다.
- **승리 조건.** "living landscape" 세트(n ≥ 300)에서 Arena 승률 ≥ 65%. BTT와 Revisit에서 최고 경쟁자보다 3σ 이상 앞선다. TSR의 W1이 실제-실제 간 노이즈 밴드 안에 든다. GCM 지연이 $d/U$의 ±20% 이내다.
- **막히면 업그레이드.** 스펙트럼 운동이 "인공적인 루프"처럼 보이면, 물과 연기 전 영역을 시뮬레이션 기반으로 바꾸고 3D 학습 난류로 디테일을 생성한다.

### 4.2 2b: 강체·관절체 (+6~9개월)

- **범위**: 낙하, 구름, 충돌, 쌓기와 무너짐, 문과 서랍, 차량(미끄러짐 없는 구름, 서스펜션), 파괴(유리, 도자기), 입상체(모래), 연체(젤리, 쿠션), Rube Goldberg 장치.
- **기술.** Newton/MuJoCo-Warp와 MPM(PhysGaussian 방식)을 쓴다. 물성 추정은 VLM 재질 분류 → 파라미터 prior → PLAC 증인으로 검증한다. 공이 튀는 높이가 이상하면 측면 증인이 지적한다. 파괴 사건 FRACTURE의 마크에는 파편 시드가 들어가서 **재촬영해도 파편 패턴이 동일**하다.
- **킬러 데모.** 실사 거실에서 도미노와 Rube Goldberg 체인이 이어진다. 동시 3캠 커버리지에, 유리잔이 깨지는 장면은 1000fps 진짜 슬로모션으로 보여준다. 이어서 반대 각도에서 재촬영하는데 파편 하나하나가 같다.
- **승리 조건.** Physics-IQ와 VideoPhy-2에서 모든 비디오 모델보다 명확한 차이로 1위. CSI 분포의 실제 대비 KL이 최고 경쟁자의 절반 이하. 강체의 MI가 0에 가깝다(측정 노이즈 수준). RST 사건 타이밍 차이 ≤ 1프레임. Arena ≥ 60%.
- **막히면 업그레이드.** 물성 오류가 지배적이면 대규모 "물성 증인 캠페인"을 돌린다. 범주별로 PLAC를 실행해 재질 prior 데이터셋을 구축한다.

### 4.3 2c: 캐릭터·연기 (+12~18개월)

- **범위**: 단일 및 다중 인물 연기, 걷기, 달리기, 춤, 스포츠, 대화(립싱크), 얼굴 클로즈업, 손, 의상, 머리카락, 동물.
- **킬러 데모.** 2인 대화 장면. 하나의 4D 테이크에서 master, OTS 2개, CU 2개가 자동으로 나온다. 감독 디렉션을 반영한 테이크 3개를 만들고, 치마와 긴 머리가 있는 댄스를 120fps 슬로모션으로 보여준다.
- **승리 조건.** 전문가 패널 기준 Arena "연기" 축 ≥ 55%. 가장 어려운 축이므로 목표를 현실적으로 두되 비대칭 우위로 보완한다. "멀티숏 연속성" 축은 ≥ 70%. foot skate < 2cm/s, 컷 간 ArcFace ≥ 0.8, eyeline 오차 < 3°, 천-몸 관통률 < 1%.
- **막히면 업그레이드.** 얼굴과 미세 연기에서 지면 세 가지를 한다.
  - (i) 비디오 모델의 클로즈업 제안을 FLAME과 신경 잔차로 lift하는 **얼굴 전용 PLAC**를 돌린다.
  - (ii) 배우 퍼포먼스 캡처 마켓플레이스를 만든다(라이선스·수익 공유).
  - (iii) Avatar Foundry를 multi-view 라이트 스테이지 데이터로 보강한다.

### 4.4 2d: 완전한 4D 생성과 인터랙션 (+24~36개월)

- **범위**: 다인물 상호작용(격투 합, 군중), 손-객체 조작(OmniGrasp, ManipTrans 계열 정책), 파괴와 유체가 캐릭터와 상호작용하는 장면, 분 단위 원테이크, 다장면 영화, 실시간 인터랙티브(게임, 인터랙티브 영화).
- **기술.** native WFM이 사건과 세계를 직접 생성하고 PLAC는 경계 사례만 맡는다. 분기 우주를 쓴다. 상태가 완전하므로 **장기 기억의 지평이 사실상 무한**하다.
- **킬러 데모.** 여러 방을 통과하는 5분 원테이크를 찍고, 같은 사건 전체를 멀티캠으로 재촬영한다. 이어 사용자가 카메라나 인물을 조작하는 인터랙티브 모드에서 10분 뒤 돌아와도 세계가 그대로다.
- **승리 조건.** 모든 범주에서 Arena ≥ 60%. Revisit 일관성의 시간 기울기가 0에 가깝다(경쟁자는 시간에 따라 감쇠한다). 인과 정확성(VLM 판정 + 인간 판정)에서 1위.

---

## 5. 데이터 플라이휠과 스케일링

### 5.1 소스 인벤토리

| 소스 | 예시 데이터셋/채널 | 주 용도 | 비고 |
|---|---|---|---|
| 드론/항공 | ACID 계열, 라이선스 스톡 | 대규모 정적 세계, 카메라 문법 | |
| 부동산/실내 투어 | RealEstate10K(~1만 영상), 투어 플랫폼 | 실내 세계, amodal 학습 | 플랫폼과 파트너십 |
| 워킹 브이로그, FPV | DL3DV-10K | 거리 수준 세계, 원경 에이전트 | |
| 영화/드라마 | E.T. dataset(Courant et al. ECCV'24), MovieNet | 카메라 세계선 문법, 연기, 조명 | 권리 리스크가 큼 |
| **스포츠 중계** | 리그 아카이브 | **같은 사건을 여러 대의 실제 카메라로 찍은 자연 데이터** → 4D 일관성 GT | 리그와 라이선스 |
| 공연 멀티캠 | 콘서트 아카이브 | 다중 증인 데이터 | |
| 댄스/피트니스 | AIST++ | 인체 모션 | |
| 손-객체 | Ego4D, Ego-Exo4D, EPIC-KITCHENS, HOI4D | 조작, 다시점 | |
| 고속 슬로모션 | 240fps 폰 슬로모, 고속 카메라 | **날카로움 GT**, TSR 레퍼런스 | |
| 모캡 | AMASS(~수십 시간), Motion-X, BEDLAM | 인체 prior | |
| 다시점 스튜디오 | Panoptic Studio, DNA-Rendering | 얼굴과 몸 GT | |
| 자체 캡처 | 멀티캠, 360, LiDAR 리그 | 고정밀 GT, 독점 | 해자 |
| 합성 | Unreal, Infinigen | 완전한 GT | |

규모 감각을 잡아 보자. 업로드 속도로 흔히 인용되는 분당 500시간 이상을 기준으로 하면 연간 약 2.6억 시간이다. 이 중 "카메라가 움직이고, 대체로 정적이거나 재구성 가능한" 비율이 3~5%만 되어도 연간 천만 시간 규모가 된다. 우리 목표는 2a/2b용 1M 시간, 전체 단계용 5M 시간의 lift다. **권리 문제는 정면으로 다룬다(§5.8).**

### 5.2 Lift 파이프라인 (Reader)

```
원본 클립
 → [필터] 카메라 운동, 시차 점수, 동적 비율, 미감, 화질, 컷 검출
 → [기하] VGGT / π3 / MapAnything / CUT3R (feed-forward 카메라 + metric 포인트맵)
 → [정적/동적 분리] MonST3R 방식 동적 마스크 + 모션 세그멘테이션
 → [동적 4D] CoTracker3 / SpatialTracker / TAPIP3D (3D 트랙)
            사람: GVHMR / WHAM / TRAM (월드 좌표 SMPL-X), 손: MANO 피팅
            객체: FoundationPose 계열 6DoF, 비강체는 Shape-of-Motion / MoSca 방식 모션 basis
 → [사건 검출] 3D 상대속도 부호 반전 + 가속도 스파이크 + 근접 → 후보 사건
            + [신규] 오디오 onset 정합 (§5.2.1)
 → [세계 피팅] 우리 표현으로 feed-forward 피팅 (W_canon, s(t), E, γ)
            상위 5% 클립은 최적화로 정밀 피팅
 → [품질 게이트] 홀드아웃 프레임 예측: 학습에 쓰지 않은 프레임을 렌더해 PSNR/LPIPS 측정
            → 클립마다 신뢰도 가중치 부여
출력: (W_i, γ_i, P_i, E_i, A_i, 신뢰도)
```

#### 5.2.1 [신규, 확신도 높음] 오디오로 서브프레임 사건 타이밍 라벨링

30fps 비디오는 접촉 시각을 33ms 단위로밖에 알 수 없다. 48kHz 오디오의 onset(spectral flux 피크)은 약 0.02ms 해상도를 갖는다. 그런데 오디오는 음속만큼 늦게 도착한다. 4D 재구성에서 사건 위치와 카메라 위치를 이미 알고 있으므로 지연을 보정할 수 있다.
$$t_{\text{event}} = t_{\text{onset}} - \frac{\|\mathbf{x}_{event}-\mathbf{x}_{cam}\|}{c_{sound}}$$
34m 떨어진 사건이라면 100ms, 즉 3프레임이 보정된다. 이렇게 사건 타이밍에 **서브프레임 정밀도의 라벨**을 붙인다. 사건의 마크(충격 세기, 재질)도 오디오 스펙트럼에서 약지도(weak supervision)로 얻는다.

선행 연구로는 소리와 시각의 연합 학습(Owens et al. "Visually Indicated Sounds" CVPR'16, Sound of Pixels)이 있다. 하지만 **4D 재구성 거리로 음속을 보정한 오디오 onset을 생성 모델의 사건 타이밍 라벨로 쓴 사례는 찾지 못했다.**

규모: 인물 중심 영상 1M 시간 × 3600s × 초당 약 2~3개의 접촉 사건 ≈ **10¹⁰개 사건**. AMASS 같은 모캡과는 자릿수가 다른 모션 데이터다.

### 5.3 [신규, 확신도 중간~높음] Factorized Recombination Engine

모든 lift 결과를 네 개의 직교 코퍼스로 분해한다.

- **세계** $\{W_a\}$
- **카메라 세계선** $\{\gamma_b\}$: 피사체 상대 좌표로 정규화해 "샷 문법"으로 저장한다. 예: "피사체 1.5m 앞에서 35mm로 dolly-in, 2.1s".
- **퍼포먼스** $\{P_c\}$: 모션, 사건, 그리고 퍼포먼스가 요구하는 지지면과 어포던스 키.
- **에셋** $\{A_d\}$

이를 재조합한다.
$$(W_a,\ \mathrm{Retarget}(\gamma_b \mid W_a, \text{subject}),\ \mathrm{PhysRetarget}(P_c \mid W_a),\ A_d)\ \to\ \text{render} \to \text{합성 비디오 + 완전한 GT}$$

- **Affordance-keyed 매칭.** $P_c$의 접촉 사건은 필요한 지지 형상을 명시한다. 예를 들어 "앉기"면 높이 0.4~0.5m, 수평 법선이 필요하다. 이것을 $W_a$의 어포던스 맵과 매칭한다.
- **물리 리타게팅.** 물리 기반 캐릭터 정책이 새 지형(계단, 경사)에 맞춰 퍼포먼스를 조정한다. 조정에 실패하면(넘어지면) 그 조합은 기각한다. 실패 자체가 검증기 역할을 한다.
- **카메라 리타게팅.** 샷 문법을 유지하면서 충돌 없는 경로를 계획한다.
- **규모.** $N_W\times N_\gamma\times N_P$가 각각 10⁷이고 유효율이 10⁻⁶이라도 조합 공간은 10¹⁵이다. 실제로는 10⁹ 렌더를 샘플링한다. **데이터가 합이 아니라 곱으로 늘어난다.**
- **겹치는 연구**: 로보틱스의 domain randomization과 procedural 합성(Infinigen)이 있다. **새로운 점은 인터넷에서 lift한 실제 분포의 구성요소를 어포던스 키와 물리 검증으로 곱해 재조합한다는 것**이다.

### 5.4 Builder–Reader self-play와 검증 가능한 보상

```
Builder B_θ: (c) → W (4D)        Reader R_φ: video → Ŵ (4D)
loop:
  W ~ B_θ;  V = render(W, γ ~ 카메라 문법 prior, 센서·압축 열화 시뮬)
  R_φ ← R_φ - ∇ d(R_φ(V), W)                      # 완전한 GT로 지도학습
  R_φ로 인터넷 영상을 다시 lift → 더 좋은 실데이터 → B_θ 사전학습 갱신
  B_θ ← RL(B_θ, r)                                  # GRPO / Diffusion-DPO 계열
  r = r_phys (ghost force, 관통, 에너지 비보존 패널티)
    + r_cycle (R_φ(render(W)) ≈ W : 세계가 '읽힐 수 있음' = 숨은 모순 없음)
    + r_real  (실제 lift 세계와의 판별자)
    + r_aes   (이미지 미감 모델, 무작위 시점)
    - r_adv   (적대적 시공간 카메라, 아래 참조)
```

**[신규, 확신도 중간~높음] 적대적 시공간 카메라 감사(Adversarial Spacetime Camera Audit).** 카메라 정책 $C_\psi$는 결함 검출기 $D$(floater, 관통, 녹아내림, 손가락 오류, 텍스처 늘어짐)의 출력을 최대화하는 (자세, 시각)을 찾는다.
$$\min_\theta\ \max_\psi\ D\big(\mathcal{R}(B_\theta(c),\ C_\psi)\big)$$
세계가 명시적이어야 가능한 학습이다. 어떤 (자세, 시각)이든 렌더할 수 있기 때문이다. 비디오 모델은 이런 식으로 감사할 수 없다. 프레임 밖은 존재하지 않기 때문이다.

**검증 가능한 보상이 갖는 의미.** LLM의 수학·코드 RL이 폭발적으로 스케일된 이유는 보상을 검증할 수 있었기 때문이다. 픽셀 비디오에는 객관적 검증기가 없다. **4D 세계에는 있다**: 재투영 일관성, 물리 잔차, cycle 재구성. 이것이 우리 스케일링 전략의 중심이다.

**붕괴 방지.**
- 배치의 50% 이상은 실제 lift 데이터로 고정한다.
- 신선한 실데이터를 계속 스트리밍한다.
- 임베딩 공간 커버리지로 다양성을 정규화한다.
- 검증 게이트를 통과한 합성 데이터만 쓴다.
- 실제 홀드아웃 세트로 Reader 정확도를 상시 감시한다.

### 5.5 Native World Foundation Model 구조

- **World tokenizer(계층형 sparse latent).**
  - 정적 세계: 2m 청크 → 청크마다 12.5cm sparse voxel(16³) × 8채널 latent(TRELLIS의 structured latent를 장면 스케일로 확장) → voxel마다 Gaussian 32개로 디코드. 원경과 하늘은 layered panorama latent로 처리한다.
  - 에셋: 객체마다 SLAT(64³ sparse) → perceiver로 512~2048 토큰.
  - 사건: 최대 1024 토큰. 카메라: 선택적 토큰. 텍스트.
  - 10s 장면 기준 약 **32k(정적) + 20k(에셋 10개) + 1k(사건) ≈ 50~64k 토큰.**
- **World DiT(multi-stream).** 스트림은 {정적 청크, 에셋, 사건, 텍스트}이고 결합 attention을 한다. 목적함수는 연속 latent에 flow matching, 이산 사건 필드에 masked diffusion이다. 여기에 **렌더 손실**을 더한다. Gaussian으로 디코드한 뒤 원본 lift 프레임 시점에서 렌더하고 perceptual loss를 건다.
- **규모 가설.** 정적 세계 4~8B, 에셋 1~3B, 사건/모션 0.5~2B, 캐릭터 정책 0.1~1B(시뮬 RL).
- **추론 경제성.** 1080p, 10s 비디오 DiT는 약 61 latent 프레임 × 120 × 68 ≈ 50만 토큰에 attention 비용이 제곱으로 붙는다. 우리는 세계 64k 토큰을 한 번 생성하고 시뮬레이션을 돌리면 된다. 3DGS 렌더는 실시간이다. **카메라 앵글을 10개로 늘리거나 길이를 늘려도 한계 비용은 렌더 비용 정도다.** 커버리지가 공짜라는 말은 경제적으로도 사실이다.

### 5.6 스케일링 법칙 가설과 go/no-go 실험

품질 축 $q$에 도달하는 데 필요한 컴퓨트를 $C(q)$라 하자.

| 가설 | 내용 | 검증 실험 |
|---|---|---|
| **H1 Factorization Dividend** | 일관성·물리 축에서 $C_{world}(q)\le C_{video}(q)/\kappa$, $\kappa\ge10$ | 같은 데이터 원천으로 $C\in\{10^{20},10^{21},10^{22}\}$에서 비디오 DiT와 World 모델을 각각 학습하고 BTT, RST, TSR, Arena 곡선을 fit. 기울기나 절편 차이로 κ 추정 → **최우선 go/no-go** |
| **H2 Camera Amortization** | lift된 세계 하나는 비디오 $K_{eff}$개의 가치가 있다($K_{eff}\approx$ 재구성 내 독립 시점 수, 5~50) | 비디오 N개 학습 vs 세계 N/K개 학습에서 같은 일관성에 도달하는 K를 측정 |
| **H3 Event-count Scaling** | 모션 모델의 손실은 프레임 수가 아니라 **사건 수**의 멱법칙을 따른다: $L(N_E)=A N_E^{-\alpha}+L_\infty$ | 사건 10⁶~10¹⁰에서 CSI, TSR, 물리 지표 측정 |
| **H4 Self-play Coupling** | Reader 정확도는 합성 세계 수에 대해 로그 선형으로 오르고, Builder 품질은 Reader 정확도를 따라간다. 천장은 실데이터 앵커가 정한다 | 소규모 루프 3세대 |
| **H5 Test-time Scaling** | 검증기가 객관적이므로 PLAC 후보 수 N에 대해 $Q\propto\log N$이 비디오 모델보다 높은 포화점까지 이어진다 | best-of-N 곡선을 같은 비용에서 비디오 모델과 비교 |

반대 방향의 위험도 적어 둔다. 외형 사실감 축에서는 표현의 천장 $L_\infty^{world}$가 비디오 모델보다 높을 수 있다. 업그레이드 방향은 primitive 용량, 신경 외형, ray-traced Gaussian, G-buffer shader다. 이 부분은 **렌더러가 감당하므로 모델 파라미터 스케일링과 분리**된다.

### 5.7 컴퓨트 견적

H100 bf16 dense 약 989 TFLOPs에 MFU 40%를 가정하면 약 $1.44\times10^{18}$ FLOP/시간이다. B200이면 대략 2배 이상이다.

| 항목 | 산정 | H100-시간 |
|---|---|---|
| 정적 세계 lift (5천만 클립 × 20s) | feed-forward 2fps → 2×10⁹ 프레임 + 상위 5% 최적화(2분/클립) | 0.1~0.2M |
| 동적 4D lift (2천만 클립 × 10s) | HMR, 트래킹, 6DoF, 피팅 약 30~60 GPU-s/클립 | 0.2~0.4M |
| PLAC 커밋 100만 건 | 480p distill된 인턴 8개 후보 + ELBO 증인 약 5 GPU-분/건 | ~0.1M |
| 캐릭터 정책 RL | 수십억 시뮬 step, 기술 다양화 | 0.05~0.2M |
| WFM T0 (10²² FLOPs) | go/no-go 실험 | ~7k |
| WFM T1 (10²³) | 2a/2b 제품 | ~70k |
| WFM T2 (10²⁴) | 2c | ~0.7M |
| WFM T3 (10²⁵) | 2d 프런티어 | ~7M |
| **2c까지 누계** | | **약 2~4M (≈ $5~12M)** |
| **2d 프런티어** | | **+8~10M** |

전략적 주장은 이렇다. **우리는 10²⁴에서, 10²⁵ 비디오 모델을 일관성·물리·커버리지 축으로 이긴다(H1).** H1이 소규모(10²⁰~10²²)에서 확인되지 않으면 스케일 투자를 하지 않는다. 그 경우 후퇴가 아니라 표현을 업그레이드한다(§8).

### 5.8 데이터 권리: 해자로 바꾼다

- 공개 크롤링에만 의존하지 않는다. 라이선스 스톡, 부동산 투어 플랫폼, 드론 촬영자 네트워크, **스포츠 리그(멀티캠 아카이브는 다른 곳에 없는 4D 데이터다)**, 공연 기획사와 수익 공유 파트너십을 맺는다.
- 자체 캡처를 운영한다. 360 + 멀티캠 + LiDAR 리그를 렌탈 하우스와 프로덕션 현장에 배포하고 캡처 대가를 지불한다.
- 배우 퍼포먼스 캡처는 동의와 라이선스를 갖춘 마켓플레이스로 운영한다. 2c의 얼굴 데이터와 초상권 리스크를 동시에 해결한다.
- 합성 데이터(재조합 엔진)의 비중이 커질수록 원천 데이터 의존도는 줄어든다.

---

## 6. 공간 음향과 부가 능력

### 6.1 하나의 원인에서 모든 감각이 나온다 (Single-cause multimodality)

소리는 별도 모델이 만든 결과물이 아니라 **같은 사건의 다른 렌더링**이다.

| 소리 | 원인 | 합성 |
|---|---|---|
| 충돌, 타격 | CONTACT/IMPULSE 사건, 충격량 $\mathbf{J}$, 재질쌍 | 모달 합성 $s(t)=\sum_n a_n(\mathbf{J},\mathbf{x}_c)e^{-d_n t}\sin(2\pi f_n t)$. 모드 $(f_n,d_n)$은 메시와 (E, ρ, ν)의 FEM 고유해석에서 구한다 (Zheng & James 2011 계열) |
| 파괴 | FRACTURE 사건, 파편 | 파편 모달 + 파괴 음향 (Zheng & James 2010) |
| 물 | 유체 시뮬레이션의 기포 생성 | 기포 진동 (Harmonic Fluids, 2009) 또는 학습된 잔차 |
| 발소리 | FOOTSTRIKE 사건 + 지면 재질 맵 | 재질별 foley 생성(사건 조건부) |
| 대사 | UTTERANCE 사건 → TTS | 같은 음소 타임라인이 입술을 구동한다 |
| 바람 | **공유 바람장**의 식생 근처 속도 | 공력 소음(aeroacoustic) 모델 |
| 군중, 앰비언스 | 원경 에이전트 밀도 | 신경 오디오 모델(세계 조건부) |

볼 수 있는 충돌의 세기와 소리의 세기 및 음색이 같은 $\mathbf{J}$에서 나온다. 싱크는 정의상 완벽하다. MMAudio 같은 video-to-audio 모델이 겪는 싱크 오차와 재질 불일치가 원인 수준에서 없다.

### 6.2 음향 전파 렌더링

- **기하**: Gaussian/SDF에서 음향용 메시를 추출하고, VLM 재질 분류로 흡음계수 $\alpha_i$를 붙인다.
- **초기 반사**는 image-source, **후기 잔향**은 확률적 음선 추적으로 계산한다. 방 단위 RT60은 Sabine 식 $RT_{60}=0.161\,V/\sum_i S_i\alpha_i$로 검증한다. 선행 연구: SoundSpaces 1/2(Chen et al.), Steam Audio 계열.
- **도플러는 따로 구현하지 않는다. 지연에서 자연스럽게 나온다.**
  $$p(t) = \sum_k\sum_{\text{path}} g_{\text{path}}(t)\, s_k\big(t-\tau_{\text{path}}(t)\big),\qquad \tau_{\text{dir}}(t)=\frac{\|\mathbf{x}_L(t)-\mathbf{x}_k(t-\tau_{\text{dir}})\|}{c}$$
  시변 지연선(delay line)의 $d\tau/dt$가 곧 도플러다. 카메라와 마이크가 움직여도, 음원이 움직여도 정확하다.
- **청자 세계선**: 기본값은 카메라다. 하지만 **마이크 세계선을 따로 둘 수 있다**(붐 마이크, 라발리에). 실제 프로덕션 문법과 같다.
- **출력**: HRTF 바이노럴, Ambisonics(FOA/HOA, VR180용), 그리고 **Dolby Atmos 등 object-based 스템을 ADM 메타데이터와 함께** 출력한다. 객체마다 위치가 있으므로 가능하다. 포스트 프로덕션 업계에서는 바로 쓸 수 있는 납품물이다.

### 6.3 기타 부가 능력 (블록 우주라서 자연스럽게 생기는 것들)

- **VFX 파이프라인 납품.** OpenUSD로 세계와 카메라 export, 깊이/법선/모션벡터/객체 ID/cryptomatte AOV 패스, 라이팅 재조정. Houdini, Unreal, Nuke로 바로 넘어간다.
- **스테레오 3D, VR180, 6DoF 볼류메트릭 비디오.** 렌더 시점만 추가하면 된다.
- **로보틱스와 피지컬 AI 시뮬레이터.** 같은 세계에 물리와 metric 스케일과 어포던스가 있다. Cosmos, Genie와 겹치는 두 번째 시장이고, 데이터 플라이휠도 공유한다.
- **인터랙티브.** 분기 우주와 실시간 렌더로 게임, 인터랙티브 영화, 라이브 가상 프로덕션을 지원한다.

---

## 7. "아무도 안 해봤을 것" 목록

| # | 아이디어 | 확신도 | 가장 가까운 선행 연구 | 무엇이 새로운가 |
|---|---|---|---|---|
| 1 | **Event-Point-Process Diffusion + ESA**: 사건 집합을 비디오 생성의 1차 latent로 쓰고, 시뮬레이터와 교대로 정합 | 높음 | MDM, CLoSD(인체 궤적 + 시뮬), 지진학/신경 분야 point process | 모든 객체 타입과 모달리티를 하나로 묶는 통화, 연속 시간, 실현된 사건이 곧 진실 |
| 2 | **증인 교차신문 + 능동 카메라 배치**: 4D 세계가 비디오 모델에게 물어볼 각도를 정보 이득으로 선택 | 높음 | SDS-4D, Diffusion Classifier, query-by-committee | 2D 생성기를 이산 제안자와 판정자로만 쓰고, 4D가 시점을 능동 선택하며, 물리가 판사 |
| 3 | **오디오 onset + 음속 지연 보정으로 서브프레임 사건 라벨** | 높음 | Visually Indicated Sounds, Sound of Pixels | 4D 거리 기반 음속 보정과 생성 모델용 사건 타이밍 지도 |
| 4 | **적대적 시공간 카메라 감사**(min-max over pose × time) | 중간~높음 | 적대적 예제, 포즈 공간 어택 | 명시적 4D 세계에서만 가능한 학습 신호 |
| 5 | **Factorized Recombination Engine** (W × γ × P × A, 어포던스 키 + 물리 리타게팅) | 중간~높음 | domain randomization, Infinigen | 인터넷에서 lift한 실분포 구성요소의 곱 재조합 |
| 6 | **공유 바람장과 Gust Coherence 지표** | 중간 | 3D cinemagraph, Generative Image Dynamics | 공간적으로 상관된 돌풍 전파, 시각과 청각의 공동 원인 |
| 7 | **TSR, CSI, MI, BTT, RST 벤치마크 세트** | 중간~높음 (지표 묶음으로서) | Physics-IQ, VideoPhy, MEt3R | 과한 부드러움을 스펙트럼 결손으로 정량화 |
| 8 | **분기 블록 우주(테이크 트리)** | 중간 | 게임 세이브/로드, Genie 계열 | 생성형 4D 세계의 상태 포크와 테이크 문법 |

---

## 8. 리스크와 업그레이드 (후퇴하지 않는다)

| 리스크 | 업그레이드 |
|---|---|
| 외형 사실감의 천장(피부 SSS, 털, 투명·굴절, 카우스틱) | ray-traced Gaussian(3DGRT/3DGUT), primitive별 신경 외형, 결정론적 G-buffer shader(§2.8). 렌더러 쪽 개선이라 모델 스케일과 분리된다 |
| 시뮬레이터가 덮지 못하는 현상 | Free 4D field 계층. 물리는 포기해도 3D 일관성은 유지된다 |
| 인간 연기와 얼굴의 불쾌한 골짜기 | Avatar Foundry, 얼굴 전용 PLAC, 퍼포먼스 캡처 마켓플레이스, 라이트 스테이지 파트너십 |
| 물성 추정 오류("돌이 젤리처럼") | 물성 증인 캠페인, 재질 prior 데이터셋 |
| 시뮬레이션 폭주와 비용 | 학습된 emulator(GNS 계열)로 미리보기, 정밀 시뮬은 최종 렌더에만. ESA가 시뮬 실패를 감지하고 재계획 |
| 비디오 모델이 급격히 좋아짐 | 그만큼 인턴이 좋아진다. PLAC가 직접 수혜를 받는다. 동시에 BTT, RST, 커버리지, 납품물 축에서 구조적 우위는 유지된다 |
| H1(κ ≥ 10)이 소규모에서 불성립 | 원인을 분해한다. 토크나이저 문제면 sparse latent 해상도와 계층을, 데이터 문제면 lift 품질 게이트와 Reader self-play를 강화한다. 표현 문제면 상태/외형 분리 지점을 재설계한다 |
| 데이터 권리 | 라이선스, 파트너십, 자체 캡처, 합성 비중 확대(§5.8) |

---

## 9. 첫 12개월 실험 계획 (Phase 1과 병행)

| 기간 | 실험 | 산출물 / 게이트 |
|---|---|---|
| M0–3 | **E1 Factorization Dividend 토이**: 식생과 물 도메인에서 1~2B 비디오 DiT vs World 모델, 10²⁰~10²¹ | κ 추정. **go/no-go** |
| M0–3 | **E2 벤치마크**: TSR, CSI, MI, BTT, RST 구현 후 모든 상용 모델 측정 | "Video Physics Spectrum Report" 공개 |
| M3–6 | **2a 시스템**: Spectral Gaussians, 공유 바람장, Tessendorf 바다, 연기 시뮬 | Arena v1, 킬러 데모 |
| M3–6 | **Lift v1** 100만 클립, **오디오 onset 사건 라벨링** 10만 클립 | 사건 타이밍 정확도(고속 카메라 GT 대비) |
| M6–9 | **EPD + ESA** (강체, Newton/Warp) | CSI/TSR이 실제 분포 노이즈 밴드 안에 들어오는지 |
| M6–9 | **PLAC v1**: 오픈 비디오 모델(Wan, HunyuanVideo, LTX 계열)을 인턴으로. 교차 증인 v1 | 커밋 수용률, ghost force 분포, 인간 판정 일치도 |
| M9–12 | **Builder–Reader self-play** 소규모 3세대 | H4 곡선 |
| M9–12 | **캐릭터 스택 프로토타입**(단일 인물 → 의상 시뮬), **Coverage Compiler 데모** | 2인 대화 장면 멀티캠 데모, RST |

---

### 마무리

Phase 1은 촬영할 공간이 있으면 비디오는 카메라 경로 문제라는 것을 보여준다. Phase 2는 이것을 한 단계 더 밀어붙인다. 촬영할 사건이 있으면 움직임은 물리 문제이고, 사건이 생성의 대상이 된다. 비디오 모델은 프레임을 그리고, 우리는 사건을 일으킨 뒤 그것을 찍는다. 확률은 원인에, 결정론은 결과에 둔다. 3대 고질병이 원인 수준에서 사라지는 이유도, 멀티캠과 테이크와 슬로모션과 소리가 공짜인 이유도, 스케일링에 검증 가능한 보상이 붙는 이유도 모두 이 한 문장에서 나온다.
