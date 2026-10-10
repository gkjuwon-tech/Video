> **출처 메모.** 설계 워크숍에서 [이론가 / first-principles] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 원안(v1)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../PARADIGM.md)의 판결록(§0.5)을 따른다.

---

# 이론가 렌즈 설계안: 프레임을 그리지 말고 세계를 풀자
## World-Synchronous Spectral Consensus (WSSC) + Spectral Federalism

> **한 줄 요약.** Kling은 프레임마다 우주를 새로 창조하는 아주 바쁜 신이에요. 240번을 창조하다 보니 매번 조금씩 다르게 만들고(일관성 붕괴), 지쳐서 대충 그리고(뭉개짐), 카메라맨 몸무게를 0kg으로 설정해 버렸어요(과한 부드러움). 우리는 우주를 **한 번만** 만들고 **카메라맨을 고용하는 제작자**예요. 할리우드는 100년 동안 이렇게 영화를 찍었어요. 우리는 그 방식을 수학으로 다시 정리하려는 거예요.

---

## 0. 이론가가 증명하려는 것

1. **인수분해 $x_t=\mathcal R(W,C_t)$는 일관성을 loss에서 type system으로 옮깁니다.** 정적 세계에서는 multi-view consistency, loop closure, 길이와 상관없는 zero-drift가 **정리(theorem)** 가 됩니다. 기대하는 성질이 아니에요.
2. **세 고질병은 사실 한 병입니다.** "아직 결정되지 않은 기하·이산 변수를 틀린 공간(픽셀)에서 평균 낸다"는 병이요. 뭉개짐의 blur kernel은 *위치 불확실성 분포* 자체이고, 과한 부드러움의 smoothing kernel은 *타이밍·위상 불확실성 분포* 자체입니다.
3. **그래서 Phase 1의 승패는 질문 하나로 좁혀집니다.** "서로 미세하게 어긋난 멀티뷰 꿈들을 **power spectrum 손실 없이** 하나의 세계로 합칠 수 있는가?" 평균을 내면 결정 대역(decision band)의 에너지가 정확히 $(1-\rho)(1-1/K)$만큼 줄어듭니다(Spectral Deficit Theorem). 그러니 **평균 대신 영역 소유권(ownership)** 으로 갑니다.
4. **크라운 주얼(★, 아무도 안 해봤다고 확신하는 것):** **Spectral Ownership Consensus + Spectral Federalism.**
   - 노이즈 레벨 σ가 정해주는 결정 주파수 대역에서만 소유권 투표를 합니다.
   - 각 prior(이미지 / 멀티뷰 / 비디오 / 물리 / 카메라)에는 가장 믿을 만한 (공간 × 시간) 주파수 대역에만 관할권을 줍니다. 헌법 제1조는 "어떤 prior도 자기 관할 주파수 밖에서는 발언할 수 없다"입니다.
5. **Phase 2는 같은 식에서 시간 축을 하나 더 여는 일입니다.** Lagrangian World(모든 원소가 ID와 궤적을 가진 세계)에 Event-first Time을 얹습니다. 시간 축 drift도 star graph라서 O(1)입니다.

---

## 1. 인수분해가 주는 정리들

### 1.1 정식화

**기존 패러다임.** $p_\theta(x_{1:T}\mid y)$, $x\in\mathbb R^{T\times H\times W\times 3}$. 세계·카메라·시간이 픽셀 텐서 하나에 뒤엉켜 있습니다.

**우리 패러다임:**
$$
W\sim p(W\mid y),\qquad C_{1:T}\sim p(C_{1:T}\mid y,W),\qquad
x_t=\mathcal S\!\left(\int_{\text{shutter}}\!\int_{\text{aperture}}\mathcal R\big(W,\;C(t+\tau,u)\big)\,du\,d\tau\right)
$$

- $\mathcal R$: 결정론적 renderer
- $\mathcal S$: sensor/film model (tone curve, halation, grain)
- Phase 1: $W$ 정적
- Phase 2: $W\to W(t)$

여기서 중요한 건 **확률성이 $W$와 $C$에만 있고 $\mathcal R$에는 없다는 점**입니다. 매 프레임 새로 뽑는 랜덤은 **필름 그레인**뿐이에요. 진짜 카메라도 그렇게 하니까 이건 리얼리즘이지 불일치가 아닙니다.

### 1.2 정리 목록

**Theorem 1 (Type-safe Consistency).** 정적 $W$, 두 프레임 $i,j$, 둘 다에서 보이는 세계 점 $p$에 대해:
$$x_i(\pi_i(p)) = L_W(p,\omega_i),\quad x_j(\pi_j(p)) = L_W(p,\omega_j)$$
두 관측은 **하나의 plenoptic function $L_W$의 샘플**입니다. Lambertian 성분은 완전히 같습니다.
- 증명: 정의 그 자체입니다(이게 핵심이에요).
- 일관성 오차는 "작은" 게 아니라 **항등적으로 0**입니다. 남는 건 renderer aliasing뿐이고, 이건 생성 문제가 아니라 Mip-Splatting이나 trilinear mip으로 끝나는 **신호처리 문제**입니다.

**Theorem 2 (Loop Closure).** $C_t=C_s$이면 $x_t=x_s$입니다. 5분짜리 원테이크로 한 바퀴 돌아와도 첫 프레임과 마지막 프레임이 비트 단위로 같아요. Kling에게는 이 문장을 쓸 권리가 없습니다.

**Theorem 3 (Drift = Effective Resistance).** 프레임(뷰)을 노드로, "서로 맞추라는 조건부 제약"을 간선으로 둡니다. 간선마다 분산 $\varepsilon^2$인 독립 오차가 생긴다고 가정하면 두 뷰 사이 불일치의 분산은
$$\operatorname{Var}[\theta_i-\theta_j]=\varepsilon^2\,R_{\text{eff}}(i,j)$$
즉 **조건부 그래프의 effective resistance**와 같습니다.

| 생성 방식 | 그래프 | $R_{\text{eff}}(i,j)$ |
|---|---|---|
| 순차 아웃페인팅 / AR 비디오 | path graph | $\lvert i-j\rvert$ |
| 고정 윈도우 $w$ 비디오 diffusion을 chunk로 이어 붙임 | band graph | $\Theta(\lvert i-j\rvert)$ (윈도우는 기울기만 줄임) |
| 단일 clip full attention | 완전 그래프 (soft, 학습된 제약) | $\approx 2/n$, 단 clip 안에서만 |
| **우리: 모든 뷰가 $W$ 하나에 연결** | **star graph (hard)** | **$2\varepsilon^2_{\text{fit}}$, 렌더 후에는 0** |

결론: **drift는 conditioning graph의 지형 문제입니다.** 아웃페인팅이 퇴화인 이유가 여기서 수식으로 나와요. 아웃페인팅은 path graph 위의 consensus라서 drift가 길이에 선형으로 늘어납니다. 창업자의 직감이 정리로 증명되는 순간입니다.

**Theorem 4 (Free Resolution / FPS / Duration + Dedup).** 생성 비용은 $O(|W|)$이고 렌더 비용은 $O(THW)$이지만 거의 공짜입니다. 120fps, 8K, 10분이 생성 비용을 늘리지 않아요.

여기에 **중복 제거 정리**가 따라옵니다.
- 비디오 모델은 같은 벽을 240번 다시 그립니다.
- redundancy ratio를 $r=\frac{THW}{U}$로 둡니다($U$ = footprint 해상도로 센 unique visible world sample 수).
- 10초 24fps 1080p, 카메라가 천천히 움직이는 정적 장면이면 $THW\approx5\times10^8$, $U\approx2\text{–}5\times10^7$이라 $r\approx10\text{–}25$입니다.
- 그래서 같은 연산 예산으로 **unique texel 하나당 10~25배 더 정성 들여 그릴 수 있습니다.** 우리는 벽을 한 번 그리되 240배 공들여 그립니다.

**Proposition (차원 세기 직관).** 정적 장면 비디오의 consistent manifold $\mathcal M$ 차원은 대략 $\dim W + 6T$이고, ambient 차원은 $D=THW\cdot3\approx1.5\times10^9$(10초 1080p)입니다.
- isotropic 오차라면 에너지 중 $1-\dim\mathcal M/D\approx 90\%+$가 "불일치 방향"에 떨어집니다.
- 실제 오차는 isotropic이 아니라서 이건 상한 직관입니다.
- 그래도 메시지는 분명해요. 픽셀 모델은 얇은 manifold 위를 외줄 타기 하고, **우리 생성기의 치역(range)은 정의상 $\mathcal M$ 안에 있습니다.**

### 1.3 보장되지 않는 것: 전쟁터가 옮겨갑니다

**보장되지 않는 것:** 모든 시점에서 $\mathcal R(W,C)$가 아름답고 prior-typical한가.

이게 핵심 전환이에요. **일관성은 공짜가 되고, 품질이 전쟁이 됩니다.** 그리고 이 전쟁은 측정 가능하고, 최적화 가능하고, adversarial camera로 감사(audit)할 수도 있습니다(§3.8). 반대로 "프레임들끼리 맞춰라"라는 Kling의 전쟁은 구조적으로 보장할 수단이 없어요.

**Phase 1 승리 조건:**
- (a) Rigidity Residual ≈ 0 (정리로 확보)
- (b) Loop Closure Error ≈ 0 (정리로 확보)
- (c) **프레임 단위 품질 ≈ 최상위 이미지 모델 품질**

(c)만 해내면 끝입니다. 비디오 모델의 프레임 단위 미감은 구조적으로 이미지 모델보다 낮기 때문이에요(§2.2의 per-pixel compute와 VAE 압축 논증). 그리고 (c)는 **"spectral deficit 없이 융합할 수 있는가"** 로 환원됩니다. 이론 전체가 이 한 점을 향합니다.

---

## 2. 병리학: 병 하나, 증상 셋

> **통일 진단:** 생성 모델이 아직 결정하지 못한 변수 $\zeta$(위치, 정체, 타이밍, 위상)를 **픽셀 공간에서 평균(hedge)** 하면 off-manifold 출력이 나옵니다. 혼합 posterior $p(x)=\sum_m\pi_m p_m(x)$의 MMSE 추정치 $\sum_m \pi_m\mu_m$은 어느 모드에도 속하지 않아요.
>
> **통일 처방:** 결정 변수를 **명시적으로 만들고**($W$, $C$, events), **원래 공간에서 이산적으로 결정한 뒤**(ownership, sampling), **결정론적으로 렌더**합니다. "Decide before you draw."

### 2.1 일관성 붕괴: 상태 변수가 없으니 보존 법칙도 없습니다
- 물리에서 보존 법칙은 상태 변수가 있어야 성립합니다. 픽셀 비디오 모델에는 "이 컵의 정체(identity)"를 담는 변수가 **어디에도 없고**, 픽셀들에 흩어져 있습니다. 그러니 컵이 머그잔으로 서서히 바뀌어도 막을 장치가 없어요.
- 수학적으로는 Theorem 3(path/band graph의 선형 drift)과 차원 세기 직관 두 가지입니다.
- **처방:** $W$라는 **상태 변수**를 도입합니다. 물질 보존이 type으로 강제됩니다.

### 2.2 뭉개짐: blur kernel = 위치 불확실성 분포
**(a) 위치 hedging.** 물체 위치가 변위 $u\sim p(u)$만큼 불확실한데 픽셀 공간에서 평균을 내면
$$\mathbb E_u[I(\cdot-u)] = I * p_u$$
**blur kernel이 바로 모션 불확실성 분포**가 됩니다. 손가락 위치가 2px 불확실하면 손가락이 2px짜리 kernel로 녹아요. 고속 동작, 손, 머리카락에서 특히 녹는 이유가 이겁니다.

**(b) 잠재 공간 Nyquist.**
- 비디오 VAE는 보통 시간 4×, 공간 8×(최근엔 16×까지) 압축합니다.
- 16~24fps에서 4× 시간 압축이면 latent rate가 4~6Hz이고 **시간 Nyquist는 2~3Hz**입니다.
- 그 이상의 움직임과 디테일은 전부 decoder의 *추측*입니다. 이 정보 없이 학습된 decoder는 에너지가 부족한 보간을 내놓습니다.

**(c) Per-pixel compute.**
- 12B 이미지 모델은 FLOPs 전부를 프레임 1장에 씁니다.
- 14B 비디오 모델은 81프레임에 나눠 씁니다.
- 그래서 프레임 단위 디테일 예산이 구조적으로 적습니다.

**처방:** 기하 불확실성은 $W$를 만드는 단계에서 **이산적으로 결정**합니다(ownership). 렌더는 결정론적이라 렌더 시점에 위치를 평균 낼 일 자체가 없어요. 디테일 예산은 Theorem 4의 dedup 덕분에 오히려 늘어납니다.

### 2.3 과한 부드러움: smoothing kernel = 타이밍·위상 불확실성
**(a) 위상 평균.** 손떨림은 $j(t)=\sum_m A_m\sin(\omega_m t+\phi_m)$이고 $\phi_m$은 랜덤입니다. $\mathbb E_\phi[j(t)]=0$이에요.
- posterior가 넓은 대역에서 샘플러가 평균 쪽으로 무너지면(few-step distillation, 강한 CFG, latent Nyquist 위 대역) **손떨림이 통째로 사라집니다.**
- 그래서 AI 카메라맨은 **손이 없습니다.** 몸무게 0kg의 유령 드론 같은 "floaty" 느낌이 여기서 나와요.

**(b) 타이밍 hedging.** 충돌 시각 $\tau$가 불확실하면
$$\mathbb E[H(t-\tau)] = F_\tau(t)$$
계단 함수가 **CDF(시그모이드)** 로 바뀝니다. 탁, 쾅 해야 할 순간이 스르르가 돼요.

**(c) 꿈결 같은 parallax.** 정적 장면을 강체 카메라로 찍으면 optical flow는 반드시
$$\mathbf u(\mathbf x)=\tfrac{1}{Z(\mathbf x)}A(\mathbf x)\mathbf t + B(\mathbf x)\boldsymbol\omega$$
꼴입니다(6DoF + depth map).
- 픽셀 모델의 flow는 이 꼴로 완전히 설명되지 않고, 그 **잔차가 "꿈결 느낌"의 정체**입니다.
- 사람 시각계는 parallax 불일치에 아주 민감해서, 잔차가 있으면 "뭔가 떠다닌다"고 느낍니다.
- 이 잔차를 **Rigidity Residual(RR)** 이라는 평가 지표로 씁니다(§10).

**처방:**
- 카메라는 **질량·관성·조작자를 가진 물리 객체**로 둡니다(§6). 위상 샘플을 하나 *실제로 뽑습니다.*
- Phase 2에서는 **event를 먼저 샘플링**합니다(§8). $\tau$를 먼저 뽑으면 $H(t-\tau)$가 날카로워요.
- parallax는 Theorem 1 덕분에 정의상 강체적입니다.

### 2.4 처방 요약

| 증상 | 수학적 원인 | 평균 내는 변수 | 우리의 구조적 제거 |
|---|---|---|---|
| 일관성 붕괴 | 상태 변수 없음, path graph drift | 정체(identity) | $W$ = 상태, star graph |
| 뭉개짐 | $I*p_u$, latent Nyquist, per-pixel compute 부족 | 위치 $u$ | 결정 대역 ownership, 결정론적 렌더, dedup |
| 과한 부드러움 | 위상·타이밍 평균, 비강체 flow | $\phi,\tau$ | Kinetic Camera, Event-first, 강체 렌더 |

---

## 3. 핵심 엔진: 불일치하는 꿈들을 하나의 세계로 (WSSC)

### 3.1 평균은 왜 죽음인가: Spectral Deficit Theorem

**Theorem 5 (Spectral Deficit of Averaging).**

설정:
- 세계 영역 하나를 $K$개 뷰가 봅니다.
- 주파수 대역 $b$에서 각 뷰의 제안을 세계로 back-project한 값을 $\hat x_k^{(b)} = s^{(b)} + n_k^{(b)}$로 둡니다.
  - 공유 성분 파워: $P_s(b)$
  - 뷰별 독자 결정 파워: $P_n(b)$
  - 쌍별 상관: $\rho_b$

평균을 내면:
$$
\mathrm{SFR}_b \equiv \frac{\mathbb E\|\bar x^{(b)}\|^2}{P_s+P_n} = 1-\frac{P_n}{P_s+P_n}(1-\rho_b)\Big(1-\frac1K\Big)
$$

해석:
- 고주파 대역은 독립 결정이 지배합니다($P_s\ll P_n$, $\rho\approx0$). 그래서 $\mathrm{SFR}\approx 1/K$입니다.
- $K=8$이면 그 대역 파워가 12.5%, 진폭이 35%로 떨어집니다. **"세탁기에 돌린 텍스처"** 가 됩니다.
- **L2로 3DGS를 재구성하는 것 = 평균 = 스펙트럼 학살**입니다. L2 최적해가 평균이니까요. L1이면 median이라 조금 낫지만 여전히 손실입니다.
- CAT3D / Marble / Lyra 계열의 "멀티뷰 생성 후 재구성"이 이미지 모델보다 흐린 이유를 **정량적으로 예측**합니다. 반증 가능한 예측이라 실험 A(§12)로 바로 검증할 수 있어요.

치료법은 두 가지이고 둘 다 씁니다.
1. **$\rho$를 올립니다.** 꿈들끼리 처음부터 비슷하게 결정하게 만드는 거예요. world-anchored noise(§3.5)와 render-equivariance fine-tuning(§9)이 여기에 해당합니다.
2. **평균을 안 냅니다.** 소유권으로 갑니다. 소유 영역 내부에서는 $\mathrm{SFR}=1$이 정확히 유지됩니다.

### 3.2 결정 주파수: σ가 무엇을 결정 중인지 알려줍니다

자연 이미지 스펙트럼은 $S(f)\approx Af^{-\alpha}$($\alpha\approx2$)입니다. Flow matching $x_t=(1-t)x_0+t\epsilon$에서 주파수별 SNR은 $(1-t)^2S(f)/t^2$입니다. SNR=1이 되는 **결정 주파수**는
$$
f_c(t)=\Big(\tfrac{A(1-t)^2}{t^2}\Big)^{1/\alpha}\;\;\overset{\alpha=2}{\propto}\;\;\tfrac{1-t}{t}
$$
- 스텝 $t_s\to t_{s-1}$에서 실제로 "결정되는" 대역은 $\mathcal B_s=[f_c(t_s),\,f_c(t_{s-1})]$입니다.
- 이 관점은 Dieleman의 "diffusion은 spectral autoregression" 분석과 같은 계열이라 우리 독창은 아닙니다.
- **우리가 새로 하는 건 이걸 멀티뷰 동기화 연산자의 분기 조건으로 쓰는 것**입니다.

### 3.3 Proximity Precedence Lemma (근접 우선 원칙)

- 세계 원소 $i$의 세계 주파수 $f^w$(cycles/m)는 뷰 $k$에서 이미지 주파수
  $$f^{\text{img}}_k = f^w\cdot \frac{z_k(i)}{F_k\cos\theta_k(i)}\quad(\text{cycles/px})$$
  로 나타납니다.
- 이 대역이 "결정"되는 조건은 $f^{\text{img}}_k<f_c(t)$입니다.

**Lemma.** 동기 디노이징에서 각 세계 대역은 **footprint $z/(F\cos\theta)$가 작은 뷰부터 순서대로** 결정됩니다.

따라서:
- footprint 기반 소유권은 휴리스틱이 아니라 **결정의 자연스러운 인과 순서**입니다.
- 먼 뷰는 그 대역에서 아직 posterior mean(미결정) 상태라 **애초에 충돌하지 않습니다.**
- 가까운 뷰는 디테일을, 먼 뷰는 레이아웃을 결정합니다. 카메라 세계의 사법부가 저절로 생기는 셈이에요.

### 3.4 동기화 연산자: Spectral Ownership Consensus (핵심 수식 E2)

세계 신호를 대역별로 분해합니다(surfel texel의 mip/Laplacian pyramid, §5). $\mathrm{BP}_k$는 뷰 $k$에서 세계로의 back-projection입니다. 원소 $i$, 대역 $b$, 뷰 $k$에 대해:

$$
\big[\mathrm{Sync}_\sigma(W,\{\hat x_k\})\big]_{i,b}=
\begin{cases}
\dfrac{\sum_k w_k\,\mathrm{BP}_k(\hat x_k)_{i,b}}{\sum_k w_k} & b\ \text{이미 결정됨 (모든 뷰 합의)}\ \Rightarrow\ \text{평균} = \text{잔여 노이즈 제거}\\[2ex]
\sum_k m_{k,b}(i)\,\mathrm{BP}_k(\hat x_k)_{i,b},\ \ m_{k,b}=\mathrm{Blur}_{\lambda_b}\!\big(\mathbb 1[k=o_b(i)]\big),\ \lambda_b\propto 1/f_b & b\in\mathcal B_s\ \Rightarrow\ \textbf{소유권}\\[2ex]
W_{i,b}\ \text{(world-anchored noise 상태 유지)} & b\ \text{미결정}\ \Rightarrow\ \text{건드리지 않음}
\end{cases}
$$

**소유자 결정:**
$$
o_b=\arg\min_{o}\;\sum_i -a_{o_i}(i,b)+\lambda\!\!\sum_{(i,j)\in\mathcal E}\![o_i\neq o_j]\quad(\text{graph cut, Potts})
$$

**Authority 점수:**
$$
a_k(i,b)=\underbrace{\log\tfrac{F_k\cos\theta_k(i)}{z_k(i)}}_{\text{footprint}}+\underbrace{\log T_k(i)}_{\text{visibility}}-\lambda_u\underbrace{\hat\sigma^2_k(i,b)}_{\text{denoiser 불확실성 (calibrated)}}+\lambda_e\underbrace{e_k(i)}_{\text{이미지 가장자리와의 거리}}+\lambda_h\underbrace{h_k(b)}_{\text{hero 뷰 우선 (저대역만)}}
$$

- **Seam 처리:** 대역 $b$는 폭 $\lambda_b\propto1/f_b$로 블렌딩합니다(Burt–Adelson multi-band). 저주파는 넓게, 고주파는 좁게 섞으니 이음매도 블러도 없습니다.
- **Janus 문제 해결:** 최저 대역(전역 레이아웃)의 소유자는 **딱 하나**입니다. hero 뷰, 또는 federalism으로 지정된 멀티뷰 prior가 맡아요(§7). 얼굴이 두 개 생길 수가 없습니다.
- **Free-space 투표권:** 어떤 뷰가 그 공간을 "비어 있다"고 보면 그 뷰는 내용을 소유하지는 못해도 **비어 있음에 대한 거부권**을 가집니다(space carving의 생성형 버전). floater가 제거됩니다.

### 3.5 World-anchored Noise: 꿈들이 같은 주사위를 굴리게 하기

- 각 surfel texel에 seed $\xi$를 둡니다. 표면 parameterization 위에 white noise 측도 $dN$을 정의합니다.
- 뷰 $k$의 픽셀 $p$ 노이즈는
  $$\epsilon_k(p)=|A_p|^{-1/2}\int_{A_p}dN$$
  이고, $A_p$는 픽셀 footprint입니다. 단위분산 가우시안이고, footprint가 겹치지 않으면 픽셀끼리 독립입니다. diffusion이 요구하는 marginal을 그대로 유지해요.
- latent diffusion이라면 latent cell footprint에 고정(anchor)합니다.
- 효과: §3.1의 $\rho_b$가 올라갑니다. 같은 표면을 보는 뷰들이 **같은 주사위 결과**로 디테일을 결정하기 시작합니다.
- 정직한 출처: 2D 버전은 *How I Warped Your Noise*(Chang+ ICLR'24)와 *Go-with-the-Flow*(CVPR'25)에서 나왔고, 3D 편집 쪽 SyncNoise가 인접합니다. 이걸 **생성 동기화의 $\rho$-레버로 스펙트럼 이론 안에 넣은 것**이 우리 몫입니다.

### 3.6 Inconsistency Explanation Cascade와 Dreamer Calibration ★

> **원칙: 설명할 수 있는 불일치는 투표에 부치지 않습니다.**

뷰 간 차이를 다음 순서로 **해명(explain away)** 하고, 끝까지 남는 진짜 충돌만 소유권 투표로 보냅니다.
1. **물리:** view-dependent radiance(specular, Fresnel)는 BRDF와 env light로 피팅합니다. 반사가 뷰마다 다른 건 정상이라 지우면 안 돼요.
2. **Gauge:** 뷰별 노출, 화이트밸런스 affine(3×4), sub-pixel warp $g_k$를 피팅한 뒤 **버립니다**(NeRF-W appearance code와 같은 계열).
3. **Dream channel:** 아래 calibration 모델로 설명합니다.
4. **잔여 충돌:** 소유권 투표.

**★ Dreamer Calibration: 이미지 모델을 꿈꾸는 측정기기로 보고 렌즈처럼 보정합니다.**

꿈 연산자 $\mathcal D_\sigma$(σ까지 노이즈를 넣었다가 디노이즈)를 **측정 채널**로 모델링합니다:
$$
\mathcal D_\sigma(\mathcal R(W,C)) = \mathcal R_{u}(W,C)+\eta,\qquad u\sim\mathcal N(0,\Sigma_u(\sigma,\text{content})),\quad \eta_b\sim\mathcal N(0,\Sigma_\eta(f_b,\sigma))
$$
- $u$: dream jitter. 미세한 기하 흔들림이고 warp field입니다.
- $\eta$: 대역별 디테일 재환각(re-hallucination)입니다.

**보정 방법:**
- 실제 3D 장면(ScanNet++, DL3DV, Matterport, Infinigen 합성)의 GT 뷰를 렌더해서 $\mathcal D_\sigma$에 넣습니다.
- 결과와 GT 사이의 RAFT flow로 $\Sigma_u$를 얻고, 잔차 스펙트럼으로 $\Sigma_\eta$를 얻습니다.
- 공유 anchored noise 아래에서 뷰 간 $\rho(f,\sigma)$도 측정합니다.

**쓰임새:**
- **융합 likelihood:**
  $$-\log p(\hat x_k\mid W)=\min_u\|u\|^2_{\Sigma_u^{-1}}+\sum_b\|\Pi_b(\hat x_k-\mathcal R_u(W,C_k))\|^2_{\Sigma_\eta(f_b)^{-1}}$$
  미세 warp는 $u$가 흡수하니까 ghosting이 생기지 않습니다. 고유 랜덤성이 큰 대역은 가중치가 낮아져서 소유자에게 결정권이 넘어갑니다.
- **Authority의 $\hat\sigma^2$ 항** 공급
- **이상화된 $f_c(\sigma)$ 대신 실측 결정 대역** 공급

**Worldness Gap $\Delta(\theta)$.** 이미지 모델이 얼마나 세계 모델인지를 재는 수치입니다.
$$
\Delta(\theta;\sigma,b)=\mathbb E_{W_{gt},C_i,C_j}\Big\|M_{ij}\odot\Pi_b\big(\mathrm{warp}_{i\to j}\mathcal D_\sigma(\mathcal R(W_{gt},C_i);\xi)-\mathcal D_\sigma(\mathcal R(W_{gt},C_j);\xi)\big)\Big\|
$$
prior 선택(FLUX / SD3.5 / Qwen-Image / HiDream 등), federalism 관할권 배정(§7), fine-tuning loss로 씁니다.

내가 아는 범위(2026 중반)에서 robust loss나 appearance embedding은 흔합니다. 하지만 **생성 prior의 오차 분포를 GT로 보정해서 정식 likelihood로 쓰는 것**은 본 적이 없습니다.

### 3.7 세계 = 고정점 (핵심 수식 E1)

$$
\boxed{\,W^\star=\mathrm{Sync}_{\sigma}\Big(W^\star,\ \big\{\mathcal D_{\sigma}\big(\mathcal R(W^\star,C);\ \xi\big)\big\}_{C\in V}\Big)\quad\text{for all }\sigma\downarrow 0\,}
$$

**해석:** 우리 세계의 아무 프레임이나 이미지 모델에게 보여주고 "σ 스케일에서 고쳐 봐"라고 했을 때, 모델이 **"고칠 게 없는데요"** 라고 답하는 세계입니다. 대역 $b$마다 그 대역을 결정하는 노이즈 레벨에서
$$\Pi_b\nabla\log p_{\sigma_b}(\mathcal R(W^\star,C))\approx0$$
이 성립합니다.

**반복(stochastic approximation):**
$$
W_{n+1}=(1-\gamma_n)W_n+\gamma_n\,\mathrm{Sync}_{\sigma_n}\big(W_n,\{\mathcal D_{\sigma_n}(\mathcal R(W_n,C);\xi^{(n)})\}_{C\in V_n}\big),\quad \textstyle\sum\gamma_n=\infty,\ \sum\gamma_n^2<\infty
$$
- $\xi^{(n)}$은 반복마다 **새로 뽑되 뷰 사이에서는 공유**합니다.
- gradient가 아니라 **샘플 기반**이므로 SDS의 mode-seeking(과포화, 매끈한 플라스틱 질감)을 피합니다.
- σ annealing과 Robbins–Monro step size 조건을 함께 쓰면 고정점 집합으로의 수렴 논증 틀이 섭니다(엄밀한 contraction 조건은 미해결 과제로 남깁니다).

**정직한 겹침:** Iterative Dataset Update(Instruct-NeRF2NeRF), Difix3D+, ISM/SDI 계열과 뼈대가 같습니다. **새로운 점은 Sync가 평균이 아니라 Spectral Ownership이라는 것, 그리고 likelihood가 calibration에서 나온다는 것**입니다.

### 3.8 Shootable Set Minimax와 Adversarial Camera (핵심 수식 E3)

$$
W^\star=\arg\min_{W\in\mathrm{Fix}}\ \sup_{C\in\Pi}\ \mathfrak B(W,C),\qquad
\mathfrak B=\sum_\sigma\omega_\sigma\|\mathcal D_\sigma(\mathcal R)-\mathcal R\|_{\text{calib}}+\lambda_h(1-\mathrm{cov})+\lambda_s\,\mathrm{stretch}
$$
- 첫째 항은 다중 스케일 Dream Residual, 둘째 항은 구멍, 셋째 항은 바늘 모양 Gaussian 같은 늘어짐입니다.
- **Adversarial camera:** CMA-ES나 SE(3) gradient로 $\Pi$ 안에서 **세계가 가장 못생겨 보이는 시점**을 찾고, 그 시점을 $V$에 넣어 고정점 반복으로 수리합니다.
- **보장:** $\sup_\Pi\mathfrak B\le\varepsilon$이면 감독이 $\Pi$ 안에서 **어떤 경로를 골라도** 모든 프레임이 $\varepsilon$ 품질입니다. "모든 카메라에 대해 아름다운 세계"가 minimax로 정식화됩니다.
- **할리우드 세트 원리 (quotient world):** $W\sim_\Pi W' \iff \mathcal R(W,C)=\mathcal R(W',C)\ \forall C\in\Pi$. 세계는 $\Pi$ 위에서만 일관되면 됩니다.
  - 세트장은 앞면만 있고, 매트 페인팅과 forced perspective도 정당한 해입니다.
  - 최적화기가 이 동치류를 자유롭게 쓸 수 있습니다. 먼 배경은 matte shell로 처리해서 싸게 끝냅니다.
  - **목표는 무한한 세계지만, 비용은 카메라가 볼 만큼만 냅니다.**

### 3.9 왜 아웃페인팅이 아닌가

| 축 | 순차 아웃페인팅·인페인팅 | WSSC |
|---|---|---|
| 동결 기준 | **순서**: 먼저 그린 픽셀이 영원히 확정 | **주파수**: 대역 하나가 모든 영역에서 동시에 확정 |
| 정보 흐름 | 인과적(뒤가 앞을 못 바꿈) | 결정 대역 안에서 양방향, 뒷면 뷰가 앞면 레이아웃에 거부권 행사 가능 |
| 그래프 | path/tree → $R_{\text{eff}}=\Theta(n)$ | star와 co-visibility → $O(1)$ |
| 상태 | 2D 캔버스(과거 프레임 픽셀) | **세계 공간 diffusion 상태 $W_\sigma$**, 노이즈가 세계에 삽니다 |
| 구멍 | 매 스텝 생김 | 없음: 모든 원소가 $\sigma_{\max}$에서 노이즈로 태어나 그것을 보는 모든 뷰가 동시에 디노이즈 |

**부분 completion이 필요한 유일한 경우:** 감독이 $\Pi$를 나중에 넓혀서 아무도 안 본 영역이 생길 때입니다. 이때는 **world-space Diffusion Forcing**을 씁니다.
- 원소마다 개별 노이즈 레벨 $\sigma_i$를 두는 heterogeneous-σ world입니다. Diffusion Forcing(Chen+ '24)은 시퀀스 토큰에 이걸 했고, 우리는 세계 원소에 적용합니다.
- 해당 영역은 이웃 문맥 속에서 빠르게 따라잡고(catch-up), 다시 전역 동기 루프에 합류해서 **미결정 대역은 계속 수정 가능한 상태로 남습니다.**
- 캔버스를 고정하고 빈칸을 채우는 것과는 위상학적으로 다른 작업입니다.

---

## 4. 알고리즘 (의사코드)

```text
procedure WSSC(y, Π, priors, calib):
  # ---------- 0. 준비 ----------
  V   ← AnchorCover(Π, K)            # visibility×footprint greedy set cover, hero 뷰 h 포함
  W   ← WorldInit(y)                 # 모든 원소 σ_max; geometry proxy = 저대역 소유 prior가 결정
  ξ   ← SeedField(W)                 # world-anchored noise seeds

  # ---------- A. BIRTH: 세계 동기 디노이징 ----------
  for s = S … 1:
    σ ← σ_s ; B ← DecisionBands(σ, calib)                # (원소, 뷰)별 결정 대역 (근접 우선)
    parallel for k in V:
      G_k ← RenderGBuffer(W, C_k)                        # depth, normal, 결정된 대역의 색, coverage
      z_k ← NoisyState(Render(W_decided, C_k), RenderNoise(ξ, C_k), σ)
      x̂_k ← Federal_Tweedie(z_k, σ, y, G_k)              # 대역별 관할 prior의 Tweedie 추정 (§7)
    W.geo ← SOC_geometry(W, {depth(x̂_k)}) + FreeSpaceVeto
    g     ← FitGauge(W, {x̂_k}) ; W.spec ← FitViewDep(W, {x̂_k}, g)   # explain-away cascade
    for band b:
      if b decided:   W_b ← CalibWeightedMean_k BP_k(x̂_k)_b
      if b ∈ B:       o_b ← GraphCut(a_k(·,b)); W_b ← MultiBandBlend(BP_{o_b}(x̂)_b, λ_b ∝ 1/f_b)
      else:           keep W_b
    if s ∈ AuditSteps: V ← V ∪ AdversarialCameras(W, Π, σ)

  # ---------- B. POLISH: 고정점 반복 ----------
  for n = 1 … N:
    σ_n ← anneal(n); ξ⁽ⁿ⁾ ← FreshSharedSeeds()
    V_n ← V ∪ AdversarialCameras(W, Π)
    W   ← (1-γ_n)·W + γ_n·Sync_{σ_n}(W, {D_{σ_n}(R(W,C); ξ⁽ⁿ⁾)}_{C∈V_n})
    if sup_{C∈Π̂} 𝔅(W,C) < 𝔅_realphoto: break          # 실사 사진 수준의 dream residual에 도달

  # ---------- C. SHOOT ----------
  Γ ← CameraSearch(y, W, rig, style)                    # Kinetic Camera, 후보 10^3~10^4개 저해상도 렌더 후 채점
  return [ Sensor( ∫∫ R(W, Γ(t+τ, u)) du dτ ) for t in frames ]
```

**연산량 추정 (Phase 1 데모 기준):**
- FLUX급 12B prior, $K\approx64\text{–}200$ 앵커, $S\approx28$ 스텝, 반복마다 VAE encode/decode 포함
- 대략 세계 하나당 H100 2~6시간
- 느리지만 합법적이고, 제품 단계에서는 §9 flywheel로 amortize합니다.

---

## 5. 표현(Representation): Surfel-Atlas World (SAW)

창업자가 말한 대로 "매개체는 아무거나"이므로, **ownership과 대역 분해가 수학적으로 깨끗한 표현**을 고릅니다.

| 구성요소 | 텐서 구조 (원소당) | 역할 |
|---|---|---|
| **Textured Surfels** ($N_s\approx$ 2–20M) | $\mu\in\mathbb R^3$, $q\in\mathbb R^4$, $s\in\mathbb R^2$, $\alpha$, **texel patch $T\in\mathbb R^{P\times P\times C}$** ($P\in\{4,8,16\}$, footprint에 따라 적응), $C$ = albedo 3 + roughness 1 + normal perturbation 3 + view-dependent latent 4 = 11 | 표면. **texel mip level = 대역 $b$** 로 σ↔LOD 동형이 직접 성립 |
| 메타데이터 | LOD $\ell$, 대역별 owner $o\in\mathbb Z^B$, authority $a\in\mathbb R^B$, 결정 대역 인덱스 $b^\*$, confidence $\kappa$, seed $\xi$(uint64), 원소별 $\sigma_i$(heterogeneous-σ) | ownership·anchored noise·Diffusion Forcing |
| **Volumetric Gaussians** ($N_v\approx$ 1–10M) | 표준 3DGS 59 floats | 머리카락, 잎, 연기, 물보라 같은 fuzzy 물질 |
| **Matte Shells** ($L$ = 4–8) | 깊이 $r_l$의 구면 RGBA equirect(8K×4K) | 원경과 하늘, forced perspective, 정확한 원거리 parallax |
| Lighting / Materials | env map + 광원, PBR 파라미터 | specular 해명, Phase 2 relight |
| **Camera** | SE(3) 연속시간 cumulative B-spline + optics track(focal, focus distance, aperture, shutter angle, anamorphic squeeze) + rig 상태 | §6 |

- **Renderer:** 2D Gaussian rasterizer, footprint 기반 trilinear mip(anti-aliasing), 셔터와 조리개를 확률적으로 적분(최종 렌더 시 16~64 sample/frame), 필요하면 Gaussian ray tracing으로 반사·굴절 처리.
- **메모리:** surfel 10M × 평균 6×6×11 texel × fp16 ≈ 8GB로, 80GB GPU에 여유 있게 들어갑니다.
- **(선택) 무한 줌 디테일:** 세계에는 coarse 대역과 seed만 두고, 극초고주파는 공유 detail synthesizer $G(\text{local context},\xi,\text{scale})$가 렌더 시점에 결정론적으로 생성합니다. seed가 세계에 고정돼 있으니 몇 번을 다시 찍어도 같은 디테일이 나옵니다.

---

## 6. Kinetic Camera: 질량 있는 카메라와 Cinematography as Search

**동역학:** 상태 $s=(p,v,q,\omega)$에 대해
$$
M\dot v = u_{\text{intent}}(t)+F_{\text{rig}}(s)-Dv+\xi_{\text{op}}(t),\qquad \xi_{\text{op}}\sim\text{colored noise},\ \mathrm{PSD}=\Phi_{\text{rig}}(f)
$$

| Rig | $\Phi_{\text{rig}}$ / 동역학의 지문 |
|---|---|
| Handheld | 호흡 0.2~0.3Hz + 조작자 흔들림 1~6Hz 피크 |
| Steadicam | 0.5~1Hz 감쇠 진자, "떠 있지만 무게 있는" |
| Dolly / Slider | jerk ≈ 0, ease-in/out, 레일 미세진동 |
| Crane / Jib | 붐 관성, 정지 시 overshoot와 settle |
| Drone / FPV | 저역 돌풍 + gimbal 잔차, FPV는 bank-turn 결합 |

- **Intent prior:** 실제 영화에서 SfM으로 추출한 카메라 궤적(E.T. the Exceptional Trajectories 계열, 영화 클립에 MegaSaM 적용, 비영화로는 RE10K·DL3DV)으로 학습한 **trajectory diffusion**. shot language("slow push-in → whip pan")와 $W$ 자체(피사체 위치, saliency)에 조건부입니다.
- **Optics:** focus pull과 lens breathing, 화면 가장자리의 cat-eye bokeh, anamorphic flare, whip pan의 rolling shutter, 180° shutter 규칙에 맞는 motion blur. **깊이가 metric이라 DOF가 물리적으로 정확합니다.** 픽셀 모델은 DOF를 흉내만 냅니다.
- **Cinematography as Search:** 세계가 고정돼 있으니 카메라 경로를 **탐색**할 수 있습니다. 저해상도 렌더는 100fps 이상 나오므로 후보 $10^3\text{–}10^4$개를 aesthetic, composition(삼분할·leading line), **parallax richness**("parallax is our flex"), rig 스펙트럼 적합도로 채점합니다.
  - Kling은 카메라 경로 하나를 시도할 때마다 영상 전체를 새로 생성해야 합니다.
  - 우리는 **재촬영(re-shoot)** 이 제품 기능입니다. 같은 세계, 다른 테이크.

---

## 7. ★★ Spectral Federalism: 주파수 연방제

**관찰:**
- 이미지 prior는 고주파 디테일의 신이지만 전역 3D 레이아웃에는 약합니다($\Delta(\theta)$의 저대역 값이 큼).
- 멀티뷰 prior(CAT3D, MVDream류)는 저대역 일관성은 좋지만 해상도와 디테일이 약합니다.
- 비디오 prior는 안무(저주파 모션)는 알지만 고주파는 VAE Nyquist 밑에서 녹아 있습니다.
- 물리는 충돌의 순간을 알고, 카메라 모델은 카메라 스펙트럼을 압니다.

**헌법:** (공간 주파수 × 시간 주파수) 평면을 대역 셀 $b$로 나누고, 셀마다 **calibration으로 측정한 신뢰도**가 가장 높은 prior 하나에게 관할권을 줍니다.

$$
W^\star=\arg\min_W\sum_{b}\ \lambda_b\ \Big\|\Pi_b\big(\mathcal R(W)-\mathcal D^{\,\mathrm{owner}(b)}_{\sigma_b}(\mathcal R(W))\big)\Big\|^2_{\text{calib}},\qquad \mathrm{owner}(b)=\arg\min_{P}\ \Delta_P(b)+\text{Var}_P(b)
$$

**관할 지도 (Phase 2 전체; Phase 1은 첫 열 + 카메라 대역만 사용):**

| 공간 \ 시간 | 저주파 (0~1Hz: 구도, 안무) | 중주파 (1~8Hz: 걸음, 제스처) | 고주파 (>8Hz: 충돌, 스냅, 떨림) |
|---|---|---|---|
| **저 (레이아웃)** | 멀티뷰 / 비디오 diffusion (저해상도) | 비디오 diffusion + Lagrangian motion prior | Event / 물리 모델 |
| **중 (형태, 실루엣)** | 멀티뷰 + 이미지 prior | Motion prior (skeleton, skin) | 물리 (contact, deformation, MPM) |
| **고 (텍스처, 모공, 머리카락)** | **이미지 prior** (canonical에 bake) | Renderer (motion blur 적분) | Renderer / Sensor (shutter, grain) |
| **카메라 대역** | Intent prior (shot language) | Rig 동역학 | Operator tremor PSD |

**왜 새로운가:**
- Magic123이나 DreamCraft3D는 2D와 3D prior를 **가중합**합니다. 모든 prior가 모든 대역에서 동시에 발언하니 서로 평균이 됩니다.
- 순차 파이프라인(멀티뷰 → 재구성 → 비디오 refine)은 뒤 단계가 앞 단계의 대역을 침범해서 다시 녹입니다.
- **대역별로 배타적인 관할권을, calibration으로 측정한 신뢰도에 따라 배정하는 것**은 내가 아는 한 아무도 하지 않았습니다.
- §3의 Spectral Ownership(뷰 사이 연방제)과 §7(prior 사이 연방제)은 **같은 원리의 두 축**입니다. *평균 내지 말고, 가장 잘 아는 자에게 그 대역을 맡겨라.*

---

## 8. Phase 2: 시간은 축 하나 더일 뿐

### 8.1 Lagrangian World
$$
W(t)=\Big(\{g_i\}_{\text{canonical}},\ \underbrace{T_o(t)}_{\text{object SE(3)}}\circ\underbrace{\mathrm{LBS}_{\theta(t)}}_{\text{articulation}}\circ\underbrace{\textstyle\sum_{m}B_{im}c_m(t)}_{\text{low-rank 잔여 변형}},\ \underbrace{a_i+\Delta a_i(t)}_{\text{appearance dynamics}},\ \underbrace{E(\mathbf x,t)}_{\text{Eulerian: 불, 연기, 물}}\Big)
$$
$$x_t=\mathcal S(\textstyle\int\!\!\int\mathcal R(W(t+\tau),C(t+\tau,u)))$$

**정리의 확장:**
- **Identity Conservation:** 원소 $i$의 외관은 $a_i+\Delta a_i(t)$뿐이고, $\Delta a$는 low-rank에 lighting·material event로만 설명되도록 penalty를 줍니다. **얼굴이 서서히 다른 사람이 되는 morphing이 type 수준에서 금지됩니다.**
- **Temporal star graph:** 모든 프레임이 canonical $W_0$에 연결되므로 drift는 영상 길이에 대해 $O(1)$입니다(Theorem 3의 시간판).
- **No motion hedging:** 운동은 궤적 공간에 있으니 픽셀에서 모션 가설을 평균 낼 일이 없습니다. motion blur는 물리 적분으로 얻습니다.

### 8.2 Event-first Time
1. Event set $\mathcal E=\{(\tau_e,\text{type}_e,\text{participants}_e)\}$을 **먼저** point process prior에서 샘플링합니다. 예: "유리잔이 떨어져 깨진다"는 낙하 시작, 바닥 접촉, 파편 분리.
2. 연속 동역학을 event에 조건부로 풉니다. 접촉 시 $\Delta v_n=-(1+e)v_n$ 같은 **속도 불연속**을 넣는 hybrid dynamical system입니다.
3. §2.3(b)의 $\mathbb E[H(t-\tau)]=F_\tau$ 문제가 원인 수준에서 사라집니다. 타이밍을 먼저 결정하면 탁, 쾅이 돌아옵니다.

### 8.3 Space-time 동기화
- 뷰를 $(C_k,t_j)$로 인덱싱하고, 같은 시각에서는 멀티뷰 SOC, 시간 방향으로는 **Lagrangian 소유권**을 씁니다. 시각 $t_j$에서 결정된 텍스처가 궤적을 타고 모든 시각으로 전파되도록 canonical에 bake합니다.
- 이미지 prior의 폴리싱은 canonical appearance로 들어가니까 **한 번 고치면 모든 프레임이 동시에 좋아집니다.**
- **Motion prior 데이터:** 인터넷 비디오 4D 재구성(MegaSaM, MonST3R, VGGT 계열, Shape of Motion, CoTracker3, TAPIP3D)으로 얻은 3D 궤적에 **Lagrangian motion diffusion**을 학습합니다. 비디오 diffusion은 저주파 대역에서만 trajectory-space distillation으로 동원합니다(federalism).
- **Phase 2 승리 논리:** Phase 1의 정적 품질(이미지 모델급 프레임)을 그대로 유지하고, 시간 축에서 물리·event 대역이 픽셀 모델의 시간 Nyquist 한계를 넘습니다. 그래서 **모든 비디오 모델을 이깁니다.**

---

## 9. 학습 신호와 플라이휠: AlphaZero for Worlds

**Expert Iteration (Anthony+ '17, AlphaZero의 세계 버전):**
- **Judge:** 이미지 prior와 federalism priors. 게임 결과에 해당합니다.
- **Search / Solver $\mathcal S$:** WSSC. 느리지만 강합니다. MCTS에 해당합니다.
- **Student $p_\phi(W\mid y)$:** 세계 네이티브 생성기. 장면 스케일의 structured sparse latent DiT이고, TRELLIS류의 장면판입니다.

**루프:**
1. $W_0\sim p_\phi$
2. 세계 공간 SDEdit으로 중간 σ에서 $\mathcal S$ 시작
3. $W^\star$ 산출
4. $(y,W^\star)$로 $\phi$ 학습
5. 반복

품질 래칫이 돌아갑니다. **3D 정답 데이터 없이** 세계 생성기가 이미지 모델 품질 쪽으로 끌려 올라가고, 실제 스캔 데이터는 regularizer로만 씁니다.

**Render-Equivariance Fine-Tuning (RE-FT):** 꿈꾸는 자들이 의회 투표 전에 미리 합의하게 만드는 학습입니다.
$$
\mathcal L_{\text{eq}}=\mathbb E\big\|M_{ij}\odot(\mathrm{warp}_{i\to j}F_\theta(\mathcal R(N,C_i),G_i)-F_\theta(\mathcal R(N,C_j),G_j))\big\|^2+\mathcal L_{\text{diff}}+\lambda\,\mathcal L_{\text{spectral}}
$$
- $F_\theta$: few-step 결정론적 sampler
- $N$: world-anchored noise
- 목표: $\Delta(\theta)$를 줄이고 $\rho_b$를 올리는 것
- 위험: 가장 쉬운 equivariant 함수는 블러입니다. 그래서 **$\mathcal L_{\text{spectral}}$(power spectrum 일치)** 와 adversarial 항으로 막습니다. 뒤로 물러서는 대신 이렇게 보강합니다.

---

## 10. 평가 지표: Kling/Seedance 격파 프로토콜

**프롬프트 세트:** 100개 × 카메라 지시. 예: "시간이 멈춘 해일과 등대, 360° orbit", "얼어붙은 폭발을 FPV로 관통", "크레인 업 reveal", "출발점으로 복귀", "60초 원테이크".

경쟁 모델에는 "time frozen, everything perfectly still, camera orbits …"로 지시합니다.

| 지표 | 정의 | 우리 예측 |
|---|---|---|
| **RR** Rigidity Residual | CoTracker3 track → 정적 장면 bundle adjustment 후 재투영 오차(px@1080p) | < 0.3px (renderer 한계) |
| **FV** Freeze Violation | 비강체 잔차 > 1px인 track 비율 ("멈춘 세계인데 숨 쉬는 물결") | ≈ 0 |
| **LCE** Loop Closure Error | 닫힌 경로의 첫·끝 프레임 LPIPS/PSNR | PSNR → ∞ |
| **SFR(f)** Spectral Fidelity | 프레임 radial power spectrum / 같은 클래스 실사 기준 | 전 대역 ≈ 1 |
| **DR** Dream Residual | 다중 σ에서 이미지 prior가 "고치고 싶어 하는 양" (실사 사진을 기준점으로) | 실사 수준 |
| **CWS** Camera Weight Score | 추정 카메라 jerk PSD와 실제 영화 PSD 분포 사이의 Wasserstein 거리 | 작음 (질량 있음) |
| **Long-take curve** | 10s / 60s / 300s에서 RR·DR | 평탄 (Theorem 3, 4) |
| Human Elo | 촬영감독 패널과 일반 평가자 blind A/B | 승 |
| 공짜 지표 | 네이티브 4K, 120fps, 재촬영 | 경쟁 불가 |

---

## 11. 선행연구와의 겹침과 새로움

| 우리 구성요소 | 가장 가까운 선행 | 같은 점 | **진짜 새로운 점** |
|---|---|---|---|
| 세계 생성 후 카메라 렌더 | World Labs Marble, HunyuanWorld, Lyra, Bolt3D, Wonderland, CAT3D, ViewCrafter | 3D 표현을 거친 렌더 | **스펙트럼 손실 없는 융합 이론과 연산자**, 품질 보장 minimax |
| 세계 동기 디노이징 | SyncTweedies, MultiDiffusion, SyncDreamer, MVDiffusion, TexFusion | 매 스텝 공통 공간에서 동기화 | **평균이 아니라 σ-결정대역 소유권** + Spectral Deficit 정리 |
| 고정점 dream-project | Instruct-NeRF2NeRF(IDU), Difix3D+, ISM/SDI | 렌더 → 생성 보정 → 재적합 | 공유 anchored seed, calibrated likelihood, 대역별 Sync |
| SDS류 | DreamFusion, SJC, VSD | prior를 세계 최적화에 사용 | gradient가 아닌 샘플 기반이라 mode-seeking 회피 |
| 소유권과 seam | Waechter+ '14 view selection, Burt–Adelson '83 | MRF view selection, multi-band blend | **디노이징 루프 안에서 σ와 결합**, 근접 우선 Lemma |
| Anchored noise | How I Warped Your Noise, Go-with-the-Flow, SyncNoise | 노이즈 수송 | 3D 표면 적분 노이즈를 **$\rho$-레버로 이론화** |
| Explain-away | NeRF-W appearance code, robust loss | gauge 피팅 | **★ Dreamer Calibration** (생성 prior를 측정기기로 보정), **Worldness Gap $\Delta(\theta)$** |
| 다중 prior 결합 | Magic123, DreamCraft3D | 2D와 3D prior 가중합 | **★★ Spectral Federalism** (대역별 배타적 관할) |
| 이종 σ | Diffusion Forcing | 토큰별 노이즈 레벨 | 세계 원소별 σ, 아웃페인팅 없는 Π 확장 |
| 카메라 | CameraCtrl, MotionCtrl, E.T. | 궤적 조건부와 궤적 데이터 | **물리 rig와 operator PSD**, cinematography as search |
| Phase 2 | Shape of Motion, PhysGaussian, CAT4D, SV4D | 4D Gaussian과 물리 | Lagrangian identity conservation, **event-first**, 시간 연방제 |
| 플라이휠 | Expert Iteration, AlphaZero | 탐색을 정책으로 증류 | 이미지 prior가 judge인 **3D-GT-free 세계 생성 래칫** |
| Drift 이론 | 없음 | | **Drift = $R_{\text{eff}}$** (아웃페인팅 퇴화의 정량 증명) |

이론가로서 내가 확신하는 아무도 안 해본 것은 다음 넷입니다.
1. **Spectral Deficit 정리 + σ-결정대역 Ownership** (멀티뷰 생성 블러의 정량 원인과 해법)
2. **Spectral Federalism** (prior 간 대역 관할)
3. **Dreamer Calibration / Worldness Gap** (생성 모델을 측정기기처럼 보정)
4. **Drift = Effective Resistance** (조건부 그래프 위상으로 일관성 분류)

---

## 12. 2~4주 안에 반증 가능한 실험

- **실험 A: Spectral Deficit 검증 (최우선).**
  - 방법: 실제 스캔 장면의 GT depth를 조건으로 이미지 모델이 $K$개 뷰를 생성하게 합니다. 그 뒤 (i) L2 3DGS 재구성, (ii) 평균 동기화, (iii) SOC로 각각 융합하고 SFR(f)를 측정합니다.
  - 예측: (i)과 (ii)는 $1-\frac{P_n}{P_s+P_n}(1-\rho)(1-1/K)$를 따르고, (iii)은 ≈ 1입니다.
  - 의미: 맞으면 **이론과 해법을 한 번에 확보**합니다.
- **실험 B: Worldness Gap 벤치마크.**
  - 방법: 오픈 이미지 모델 4~5개의 $\Delta(\theta;\sigma,b)$를 anchored noise 유무로 측정합니다.
  - 산출: prior 선택, $\rho_b$ 상승폭 측정, federalism 관할 초안.
- **실험 C: Bullet-time 데모.**
  - 방법: "얼어붙은 해일이 등대를 덮치는 골든아워" 세계 하나를 만들고, 30초 orbit과 dolly-through를 찍습니다. Kling과 Seedance에 같은 프롬프트를 넣고 RR, FV, LCE, SFR, Elo로 비교합니다.
  - 의미: **Phase 1 가설 증명의 첫 영상**입니다.
- **실험 D: Kinetic Camera 블라인드 테스트.**
  - 방법: 실제 영화 200샷에서 PSD를 피팅한 뒤, 같은 세계에서 "유령 드론 궤적"과 "질량 있는 steadicam"을 평가자에게 보여줍니다.
  - 예측: "무게감"이 미감 점수를 크게 올립니다.

---

## 13. 맺음말

정리하면 이렇습니다.
- 비디오 diffusion은 같은 세계를 240번 **다시 상상**하면서, 240번의 상상이 우연히 일치하기를 바랍니다.
- 우리는 세계를 **한 번** 상상하고, 그 상상이 모든 카메라 앞에서 버티는지 **정리로 보장하고 minimax로 감사**합니다.
- 평균은 틀린 공간에서 쓰면 해로운 연산입니다. 소유권은 근접 우선이고, 각 prior는 자기 주파수에서만 발언합니다.
- Phase 1에서는 멈춘 세계에서 Kling의 프레임을 이미지 모델급 미감으로 이기고, Phase 2에서는 시간 축을 열어 event로 쾅 하고 끝냅니다.

Kling은 0kg 유령 드론을 띄우고, 우리는 ARRI와 steadicam 장비와 진짜 사람의 손떨림을 띄웁니다. 우리가 프레임마다 새로 뽑는 랜덤은 필름 그레인 하나뿐입니다.
