> **출처 메모.** 설계 워크숍에서 [이론가 / first-principles] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 건설적 프리모템과 개정안(v2)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../../PARADIGM.md)의 판결록(§0.5)을 따른다.
>
> **복구 메모.** 처음 커밋할 때는 워크숍 출력이 두 메시지로 나뉘어 뒷부분만 저장돼 앞부분이 빠져 있었다. 에이전트 대화 기록에서 전체 원문을 복구해 교체했다. v1 `PARADIGM.md`는 복구 전의 잘린 본문을 입력으로 받아 작성됐다.

---

# 건설적 프리모템: WSSC를 WSSC v2로

## 0. 요약

- **뼈대는 맞습니다.** 인수분해 $x_t=\mathcal S(\mathcal R(W,C_t))$, star graph, "평균 대신 소유권"은 v2에서도 그대로 씁니다. 구현이 실패하는 곳은 **누가, 언제, 어떤 대역을 결정하느냐**의 세부입니다.
- **가장 위험한 7곳**
  - **F1 맥락 없는 결정.** 근접 우선 Lemma 때문에 가장 자세한 뷰가 맥락을 가장 모르는 뷰가 됩니다.
  - **F2 Latent 이음매.**
  - **F3 폴리시 루프의 자기모순.** seed를 반복마다 새로 뽑아 평균하면 고대역 파워가 $\gamma/(2-\gamma)$로 줄어듭니다.
  - **F4 사진 굽기.** 이미지 모델은 세계가 아니라 사진을 그립니다.
  - **F5 확률적 물질.** 잎, 물보라 같은 영역이 cardboard parallax를 만듭니다.
  - **F6 소유권 패치워크.**
  - **F7 샷 단위 패배.** popping, shimmer, 그리고 싸울 샷을 잘못 고르는 문제입니다.
- **업그레이드 핵심**
  - Single-Canvas Layout + World-Scale Causality (footprint-staggered log-SNR 스케줄)
  - Partition Principle: 평균도 곱도 아닌 분할만이 K에 불변입니다.
  - Frozen-Seed Band-Commit
  - Camera Owns Photography: Clean-Plate Dreamer + Gaffer
  - Stochastic Matter Field
  - Temporal-Spectral Renderer
- **정정.** Drift = $R_{\text{eff}}$는 Barooah–Hespanha(2007–08)의 기존 결과를 적용한 것입니다. crown jewel 목록에서 내리고 빌려 쓰는 부품으로 옮깁니다.
- **90일 실험 5개.** 마지막 실험은 FrozenBench v0입니다. Wan 3.0, Seedance 2.5, MiniMax H3, Kling 3.0 Omni, Veo 3.1 / Gemini Omni 1.1 Flash와 정면으로 붙입니다.

---

## 1–2. 실패 지점 7개와 각각의 업그레이드

| # | 실패 지점 | 발생 확률 | 치명도 | 업그레이드 |
|---|---|---|---|---|
| F1 | 맥락 없는 결정: 레이아웃 콜라주, 근접 우선의 역설 | 매우 높음 | 치명 | U1 Single-Canvas Layout + World-Scale Causality |
| F2 | Latent 이음매: VAE 왕복, latent 스펙트럼, decoder 대역 | 높음 | 큼 | U2 Latent-Texel World + World Detail Decoder |
| F3 | 폴리시 반복 평균이 Theorem 5를 스스로 위반 | 설계상 확실 | 큼 | U3 Frozen-Seed Contraction + Band-Commit |
| F4 | 사진(보케, 비네팅, 조명 연출)이 세계에 구워짐 | 매우 높음 | 큼(미감) | U4 Camera Owns Photography + Gaffer |
| F5 | 확률적 물질의 cardboard parallax | 매우 높음(데모 프롬프트에 직격) | 치명 | U5 Stochastic Matter Field |
| F6 | 소유권 패치워크(hybrid-image 아티팩트) | 높음 | 중~큼 | U6 Partition Principle + 인스턴스 잠금 + VP 이음매 |
| F7 | 프레임은 이기고 샷에서 짐 | 중~높음 | 사업상 치명 | U7 Temporal-Spectral Renderer + 전장 설계 + FrozenBench |

---

### F1. 맥락 없는 결정: 레이아웃 콜라주와 '근접 우선의 역설'

**실패 장면.** "얼어붙은 해일과 등대, 360° orbit". 앵커 64개가 $\sigma_{\max}$에서 동시에 출발하면 이런 세계가 나옵니다.
- 앞쪽 뷰와 뒤쪽 뷰가 각자 등대를 결정해서 등대가 두 개가 됩니다.
- 등대 벽 클로즈업 앵커는 "나무 판자"를 결정합니다.
- hero 뷰가 못 보는 해일 뒤편은 graph cut이 아무 뷰에나 나눠 줍니다. 바다와 절벽이 패치워크가 됩니다.

**기술적 원인**

1. **세계가 coarse-to-fine으로 태어나지 않습니다.**
   - 뷰 $k$의 footprint를 $\phi_k=z_k/(F_k\cos\theta_k)$ [m/px]라 하면 세계 주파수 $f^w$는 이미지 주파수 $f^w\phi_k$로 보입니다.
   - flow matching의 log-SNR을 $\lambda=\ln\frac{(1-t)^2}{t^2}$, 자연 스펙트럼을 $Af^{-\alpha}$로 두면 결정 조건은 다음과 같습니다.
     $$\lambda+\ln A-\alpha\ln(f^w\phi_k)\ \ge 0$$
   - 같은 $\lambda$에서 가까운 뷰는 더 미세한 세계 스케일을 결정합니다. 모든 뷰가 각자 프레임의 **최저 주파수**를 동시에 결정하는데, 그 최저 주파수가 가리키는 세계 스케일이 뷰마다 다릅니다.
2. **근접 우선의 역설.**
   - v1의 Proximity Precedence Lemma 자체는 맞습니다.
   - 하지만 함의는 반대입니다. 가장 권위 있는 뷰(footprint 최소)가 그 세계 대역을 자기 프레임 크기 스케일에서, 즉 프레임 밖 맥락 없이 결정합니다.
   - 클로즈업은 자기가 보는 것이 등대 벽이라는 사실을 모른 채 벽돌이나 판자를 고릅니다.
3. **Prompt leakage.** 모든 뷰에 전역 프롬프트 "lighthouse"를 넣으면 각 뷰가 자기 프레임 안에 등대를 하나씩 그립니다. MultiDiffusion 파노라마에서 객체가 반복되는 아티팩트와 같은 현상입니다.
4. **Hero 뷰 혼자서는 최저 대역을 덮지 못합니다.** hero 뷰는 360° orbit이 보는 세계의 30~40%만 봅니다. 나머지 최저 대역을 graph cut이 아무 뷰에나 배분하니 레이아웃이 콜라주가 됩니다.
5. **흐린 Tweedie에서 기하를 뽑습니다.**
   - 높은 σ의 $\hat x$는 posterior mean이라 흐립니다. 여기서 뽑은 단안 depth는 뷰마다 스케일이 다른 쓰레기입니다.
   - texel에 고정된 seed는 기하가 바뀌어 재파라미터화되는 순간 깨집니다.

**업그레이드 U1: Single-Canvas Layout + World-Scale Causality**

**U1a. 최저 대역은 단 하나의 캔버스에서 결정합니다.** 이미지 모델이 가장 잘하는 일은 *한 장 안에서 구도를 일관되게 잡는 것*입니다. 그래서 세계의 L0 대역(장면 크기의 1/10 이상 스케일)은 한 번의 디노이징 패스로 한 캔버스 위에서 결정합니다.

- **Inside-out(실내, 풍경, 워크스루).** equirect 360° 파노라마 한 장(4096×2048)을 씁니다. equirect LoRA를 단 이미지 모델이나 HY-Pano 2.0으로 만듭니다.
- **Outside-in(hero 물체 orbit).** 3×3 turnaround sheet 한 장(2048²)을 씁니다. Zero123++/Era3D의 grid-in-one-image 트릭처럼 한 캔버스 안에서 9시점을 그리게 합니다.
- **전역 변수(DC 대역)는 캔버스보다 먼저 정합니다.** 태양 방향, 시간대, 팔레트, 렌즈 패키지입니다. 이건 공간 영역이 아니라 전역 변수라서 캔버스 여러 장이 같은 값을 조건으로 **동시에** 생성됩니다.
- **여러 공간이 이어질 때.** LLM이 먼저 symbolic layout(scene graph와 floor plan)을 정합니다. 캔버스들은 그 semantic render를 ControlNet-seg 조건으로 받아 동시에 생성되고, 겹치는 영역은 L0에서 partition으로 합의합니다.
- **Lift.** 캔버스에서 DA3, VGGT-Ω, MoGe-2로 metric pointmap을 얻습니다. hero 물체의 뒷면은 image-to-3D prior(TRELLIS 계열, Hunyuan3D 2.x, SAM 3D Objects)로 채웁니다. 인스턴스는 SAM으로 lift합니다.
- **아웃페인팅이 아닌 이유.** 캔버스는 통째로 한 번에 생성됩니다. 동결 영역이 넓어지는 과정도, 경로 순서도 없습니다. 이 단계에서 결정되는 건 **L0 대역뿐**이고 L1 이상은 모든 영역에서 여전히 열려 있습니다.

**U1b. World-Scale Causality: footprint-staggered log-SNR 스케줄.** 뷰 $k$의 시계를 전역 시계보다 늦춥니다.
$$\boxed{\lambda_k=\lambda-\alpha\ln r_k,\qquad r_k=\phi_{\text{ref}}/\phi_k\ \ (\ge 1)}$$
- 대입하면 결정 조건이 $\lambda+\ln A-\alpha\ln(f^w\phi_{\text{ref}})\ge0$가 되어 **$k$가 사라집니다**. 모든 뷰가 같은 세계 대역을 같은 순간에 결정하므로 세계가 world-scale 기준으로 coarse-to-fine으로 태어납니다.
- $\alpha=2$이면 footprint 한 옥타브당 지연은 $2\ln2\approx1.39$ nats입니다. $\alpha$는 F2의 U2c에서 실측합니다.
- 부수효과로 가까운 뷰가 늦게 합류하므로 전체 forward 수가 30~40% 줄어듭니다.
- 한 뷰 안에서도 원근 때문에 footprint가 변합니다. 1차로는 뷰별 중앙값 $\phi$와 U1c로 처리하고, 2차로는 per-token timestep fine-tune(Diffusion Forcing 방식)으로 픽셀 단위 $\lambda$까지 갑니다.

**U1c. Ownership window.** 뷰 $k$는 이미지 주파수가 $\mathcal W_k=[\kappa/W_k,\ f_{\text{Nyq}}]$ 안에 드는 세계 대역만 소유할 수 있습니다($\kappa\approx4$, 즉 프레임의 1/4보다 작은 구조만).
- 프레임 크기 스케일의 구조는 그 프레임에게 맥락이 부족합니다. 그래서 그 구조를 중간 주파수로 보는 넓은 뷰가 소유하고, 클로즈업은 그 결정을 렌더로 받아 봅니다.
- 와이드 샷이 설정하고 클로즈업이 디테일을 채우는 촬영 문법이 규칙으로 들어옵니다.

**U1d. Render-captioned local prompt.** 옥타브마다 VLM이 현재 세계 렌더 $\mathcal R(W,C_k)$를 캡션해서 뷰별 프롬프트를 만듭니다. 전역 brief에서는 스타일 토큰만 공유하고, 프레임 밖에 있는 hero 객체 이름은 프롬프트에서 뺍니다.

**U1e. 기하는 외관보다 한 옥타브 앞서고, 시차가 보이는 대역까지만 결정합니다.**
- 기하 대역 $b$는 외관 대역 $b{+}1$이 결정되기 직전에 정합니다. 그 시점의 $\hat x$는 대역 $b$까지 이미 선명하므로 거기서 depth head나 DA3로 추정하고, 같은 partition 규칙과 free-space veto로 융합합니다.
- **Parallax relevance band.** 깊이 $z$에서 진폭 $\delta z$인 기하 디테일은 $\Pi$ 안의 최대 baseline $B$에서
  $$\Delta p\approx F B\,\delta z/z^2\ \ge\ \tau_p\ (=0.5\text{px})$$
  일 때만 기하로 결정합니다. 그보다 작은 디테일은 normal 또는 texel로 bake합니다.
  - 예: $z=5$m, $F=2000$px, $B=2$m이면 $\delta z\ge3.1$mm입니다.
  - 예: $z=50$m이면 $\delta z\ge31$cm입니다.
- **노이즈 DNA는 표면이 아니라 ambient sparse octree에 둡니다(§5.3).** 표면이 셀 안에서 움직여도 seed는 그대로이고 면적 가중치만 다시 정규화됩니다. 그래서 기하를 정제해도 anchored noise가 유지됩니다.

빌려 쓸 부품: HY-Pano 2.0, equirect LoRA, Zero123++/Era3D grid trick, DA3, VGGT-Ω, MoGe-2, TRELLIS/Hunyuan3D/SAM 3D, Diffusion Forcing.

---

### F2. Latent 이음매: 픽셀 이론과 latent 모델 사이의 틈

**기술적 원인**

1. **스펙트럼 가정이 다릅니다.** v1의 이론은 픽셀 공간에서 $\alpha\approx2$를 가정합니다. 그런데 FLUX나 Qwen-Image는 16ch, 8× VAE latent에서 디노이즈합니다. latent 스펙트럼은 픽셀보다 평평합니다(고주파 과다, Skorokhodov+ 2025의 diffusability 분석). 그러면 $f_c(\sigma)$가 엉뚱한 대역을 "지금 결정 중"이라고 지목합니다.
2. **VAE 왕복 drift.** 픽셀 공간 sync는 매 스텝 다음을 반복합니다.
   - $K$개 decode → back-project → 세계 렌더 → $K$개 re-encode
   - $E(D(z))\neq z$라서 $S\times N$번의 왕복 동안 색과 대비가 drift하고 고대역이 깎입니다.
   - 소유권으로 합성한 이미지는 encoder 입장에서 off-manifold입니다.
3. **Latent cell 해상도 문제.** latent cell은 8px이고 FLUX는 2×2 patchify를 하므로 토큰은 16px입니다. anchored noise는 이 단위로만 정렬됩니다. 뷰 사이에 sub-cell 오정렬이 생기면 $[f_{\text{lat}}/2,\ f_{\text{lat}}]$ 대역에서 $\rho$가 급락합니다.
4. **Decoder가 맡는 대역.** latent Nyquist(약 1/16 cyc/px) 위의 대역은 **결정론적 decoder**가 마지막에 한 번에 만듭니다. 이 대역에는 σ도 노이즈도 없어서 동기 디노이징으로 조화시킬 기회가 아예 없습니다. 게다가 decoder는 shift-equivariant가 아니어서 뷰마다 다른 미세 디테일을 냅니다.

**업그레이드 U2: Latent-Texel World + World Detail Decoder (WDD)**

**U2a. 세계가 latent texel을 직접 보유합니다.**
- 각 surfel이 $Z_i\in\mathbb R^{p\times p\times16}$를 가집니다. latent rasterization $\mathcal R_{\text{lat}}(W,C)$로 노이즈 상태를 만들고 sync도 latent에서 합니다. 매 스텝의 decode/encode가 사라집니다.
- 전제 조건은 VAE를 local warp에 등변(equivariant)하게 fine-tune하는 것입니다.
  $$\mathcal L_{\text{eq}}=\mathbb E_H\|D(H\!\circ\! z)-H\!\circ\! D(z)\|_1+\lambda_p\,\text{LPIPS}+\lambda_s\big\|\log P_{D(H\circ z)}(f)-\log P_{H\circ D(z)}(f)\big\|_1$$
  여기서 $H$는 surfel 투영 Jacobian에서 샘플한 local affine 또는 homography입니다. EQ-VAE(Kouzelis+ 2025)와 scale-equivariance 정규화(Skorokhodov+ 2025)를 확장한 것입니다.
- denoiser는 새 latent에 LoRA로 적응시킵니다.

**U2b. WDD: latent Nyquist 위의 대역은 texel 공간에서 디코드합니다.**
- $G_\psi(Z_{\text{nbhd}},\ \xi_{\text{octree}},\ \ell)\to$ texel pyramid 레벨 $\ell>\ell_{\text{lat}}$ (RGB/PBR).
- 학습 신호:
  - (i) $\text{LowPass}(\mathcal R(G(Z)))\approx\text{LowPass}(D(\mathcal R_{\text{lat}}(Z)))$ (LPIPS)
  - (ii) 고역은 같은 footprint의 실사 crop을 기준으로 adversarial loss와 spectral matching
  - (iii) 시점 일관성은 texel 공간이라 **구조적으로 공짜**
- seed가 고정돼 있으므로 다시 찍어도 같은 디테일이 나오고 무한 줌도 됩니다. v1에서 선택 사항이던 detail synthesizer를 필수 부품으로 올립니다.

**U2c. 결정 대역을 실측합니다.** 대역별 결정 시각을 다음처럼 정의하고 latent와 pixel 양쪽에서 측정해 `DecisionBands` 테이블로 씁니다. $\alpha=2$ 가정은 버립니다.
$$t_b=\arg\min_t\{\mathrm{Var}_\xi[\Pi_b\hat x_t]\le\tfrac12\mathrm{Var}_\xi[\Pi_b\hat x_{t_{\max}}]\}$$

**병행 arm(후퇴가 아닌 보험).** 픽셀 sync는 유지하되 **결정된 대역만** re-encode합니다. latent Laplacian으로 대역을 나누고, 아직 결정되지 않은 대역은 그 뷰 자신의 latent를 그대로 둡니다.

빌려 쓸 부품: EQ-VAE, Latent Radiance Fields(3D-aware VAE, ICLR'25), FLUX/Qwen-Image VAE, Difix3D+(1-step fixer).

---

### F3. 폴리시 루프가 스펙트럼 학살을 되살립니다 (설계 내부의 모순)

**원인.** v1의 E1 반복은 다음과 같습니다.
$$W_{n+1}=(1-\gamma_n)W_n+\gamma_n Y_n,\qquad Y_n=s+h_n$$
$\xi^{(n)}$을 반복마다 새로 뽑으므로, 결정 대역 위에서 새로 환각된 성분 $h_n$은 반복끼리 독립입니다(파워 $P_h$).

- **상수 γ의 정상상태:**
  $$\mathrm{Var}[h\text{-성분}]=\frac{\gamma^2P_h}{1-(1-\gamma)^2}=\frac{\gamma}{2-\gamma}P_h$$
  $\gamma=0.1$이면 파워의 5.3%만 남습니다. 진폭으로는 23%입니다.
- **Robbins–Monro($\sum\gamma^2<\infty$):** 0으로 수렴합니다. $K\to\infty$ 평균과 같습니다.
- v1은 뷰 축에서 금지한 평균을 **반복 축에서 다시 하고 있습니다.**
- 게다가 $\mathcal D_\sigma$는 sampler라서 실사 사진에서도 $\mathcal D_\sigma(x)\ne x$입니다(결정 대역 위를 다시 환각합니다). 고정점 조건이 영원히 성립하지 않으니 루프는 계속 섞기만 합니다.

**업그레이드 U3: Frozen-Seed Contraction + Band-Commit**

- **Seed는 세계의 DNA입니다.** $\xi$를 (원소, σ-level)마다 한 번만 뽑고 세계가 존재하는 동안 고정합니다.
  - 그러면 $\mathcal D_\sigma(\cdot;\xi)$는 결정론적입니다(고정 seed로 σ에서 ODE).
  - $\Phi(W)=\mathrm{Sync}(W,\{\mathcal D_\sigma(\mathcal R(W,C);\xi)\})$도 결정론 사상이 됩니다.
  - $W$가 안정되면 재환각 결과도 매번 같으므로, 평균을 해도 지워지는 게 없습니다.
- **대역을 나눠 처리합니다.**
  - $b\le b_c(\sigma_n)$: 이미 합의된 대역입니다. 평균해도 됩니다($\rho\approx1$, 잔여 노이즈 제거 효과).
  - $b>b_c(\sigma_n)$: **commit 방식**을 씁니다. 제안 $W'=W$ with $W_b\leftarrow Y_b^{\text{owner}}$를 만들고, 아래 에너지 기준으로 $\Delta E<0$이면 수락, 아니면 $e^{-\Delta E/T_n}$ 확률로 수락합니다(annealed MH). 평균은 없습니다.
    $$E(W)=\sum_{C\in V_n}\mathrm{DR}_{\text{calib}}(W,C)+\lambda_{\text{seam}}\mathrm{Seam}+\lambda_{\text{st}}\mathrm{Stretch}$$
- **고정점을 분포적으로 재정의합니다.** "고칠 게 없는 세계"가 아니라 "실사 사진만큼만 고치고 싶은 세계"가 목표입니다.
  $$\mathrm{DR_{rel}}(W,C)=\frac{\mathrm{DR}(\mathcal R(W,C))}{\mathrm{median}_{\text{real}}\mathrm{DR}}\in[0.8,1.2]\quad\forall C\in\Pi$$
  DR 평가에는 **held-out seed**를 씁니다. 세계 DNA로 평가하면 세계가 자기 seed에 과적합되고, denoiser를 속이는 adversarial 패턴이 생깁니다.
- **Reopen.** adversarial camera가 $\mathrm{DR_{rel}}>1.3$인 영역을 찾으면 그 (영역, $b\ge b^*$) 대역만 새 DNA 버전으로 다시 엽니다. 공간 순서가 아니라 대역 단위로 다시 여는 것이라 아웃페인팅과는 위상이 다릅니다.
- **비용.** polish는 1~4 step으로 distill한 dreamer(Difix3D+형 또는 자체 DMD distill)로 돌립니다.

빌려 쓸 부품: IDU(Instruct-NeRF2NeRF), Difix3D+, DreamTime(σ annealing), annealed MH.

---

### F4. 이미지 모델은 '세계'가 아니라 '사진'을 그립니다

**증상**
- 배경 보케가 세계에 구워져서, 카메라가 다가가면 텍스처가 흐립니다.
- 비네팅이 구석을 어둡게 만들고, flare가 하늘에 줄무늬로 남습니다.
- 각 뷰가 자기 구도에 맞춰 연출한 rim light와 local contrast가 서로 다릅니다. explain-away가 이걸 평균 내면 결과는 납작한 **CG 룩**입니다.
- 유리와 물의 반사가 텍스처로 칠해집니다.

**원인**
- 이미지 prior가 배운 분포는 $p(\text{Grade}\circ\text{Sensor}\circ\text{Lens}\circ\text{Camera}\circ\text{Light}\circ\text{World})$입니다. 세계만이 아닙니다.
- v1의 authority는 초점이 맞았는지 모릅니다.
- v1의 gauge(3×4 affine)는 공간적으로 변하는 렌즈 효과와 local tone mapping을 표현하지 못합니다.
- **더 근본적인 원인:** 이미지 모델 미감의 상당 부분은 *그 프레이밍에 맞춘* 조명과 그레이딩 연출에서 나옵니다. 이건 카메라 쪽 변수입니다. 영화 현장에서도 gaffer는 셋업마다 조명을 다시 칩니다.

**업그레이드 U4: Camera Owns Photography (Clean-Plate Dreamer + Gaffer)**

- **Plate 모드 dreamer.** `<plate>` LoRA를 학습합니다.
  - 데이터: DL3DV/RE10K 프레임 중 defocus estimator로 걸러낸 전 화면 sharp 프레임, EXIF 기준 f/8~f/16 사진, Infinigen/UE pinhole 렌더
  - `<bokeh>`, `<vignette>`, `<flare>`에는 negative CFG를 겁니다.
  - 출력은 고정된 log 톤 공간의 *plate radiance*입니다. 목표는 장면 미감은 유지하고 사진 효과는 0으로 만드는 것입니다.
- **Authority에 초점 항을 추가합니다.** $-\lambda_{\text{dof}}\beta_k(i)$를 더하고($\beta$: 추정 blur, px), $\beta>1$px이면 그 뷰는 소유할 수 없습니다.
- **Intrinsic 채널.** RGB↔X나 Marigold-IID 계열 dreamer가 albedo, roughness, normal을 만듭니다. 이것이 relightable layer가 되고 L1~L2 대역을 소유합니다.
- **반사는 그리지 않고 렌더합니다.**
  - 유리, 물, 금속은 ray-traced Gaussians(3DGRT/3DGUT)로 $W$ 자신을 반사합니다.
  - dreamer가 그린 반사와의 잔차는 텍스처로 쓰지 않습니다. 화면 밖 내용이나 roughness에 대한 **세계의 증거**로 보고 역렌더링에 씁니다.
- **사진은 카메라가 만듭니다.** 다음은 전부 $\mathcal S$와 $C$의 몫입니다.
  - thin-lens/anamorphic DOF, cat-eye bokeh, lens breathing
  - shutter 적분, rolling shutter, OLPF
  - 센서 노이즈, halation, grain(프레임별 seed), LUT
- **Gaffer(샷에 붙는 조명)**
  - 조명 세트: $L_{\text{shot}}=\{(\text{type},\ \text{pose}\in SE(3),\ \text{size},\ I,\ \text{CCT},\ \text{softness},\ \text{gobo})\}_{\le 8}$
  - 샷마다 $\max_{L}\ \mathbb E_t[\mathrm{Aes}(x_t)-\lambda\,\mathrm{DR_{rel}}(x_t)]$로 최적화하고, 샷 안에서는 고정합니다.
  - 렌더: $L_o=L_{\text{plate}}+\sum_l f_r(a,\rho,n;\omega_l,\omega_o)L_lV_l$ ($V_l$: 3DGRT shadow ray)
  - **Theorem 1은 샷 단위로 유지됩니다.** 조명은 샷 동안 $C$의 고정된 일부입니다. 샷이 바뀌어도 세계(기하·재질)는 같고 조명만 다릅니다. 실제 촬영과 똑같은 구조이고, 이렇게 해서 이미지 모델의 미감을 규칙 안에서 되찾습니다.

빌려 쓸 부품: RGB↔X, DiffusionRenderer, Marigold-IID, 3DGRT/3DGUT, NeRF-W gauge, defocus estimator.

---

### F5. 확률적 물질: 잎, 풀, 털, 물보라, 연기, 파편

**증상.** "parallax is our flex"가 "parallax is our tell"이 됩니다. 숲이 판지에 그린 그림처럼 움직입니다(cardboard parallax). v1의 데모 프롬프트인 해일 물보라, 얼어붙은 폭발, 숲이 정확히 이 경우입니다.

**원인**
- 결정해야 할 변수는 3D 미세구조(잎 수천 장의 위치와 방향)입니다. 그런데 어떤 뷰도 이걸 일관되게 결정하지 않습니다. 뷰 간 $\rho\approx0$입니다.
- 평균을 내면 SFR이 $1/K$로 떨어집니다.
- 소유권을 쓰면 한 뷰의 잎이 proxy 표면에 칠해져서 내부 시차가 0이 됩니다.
- 뷰별 depth로 volumetric Gaussian을 세우면, depth가 잎 단위로 일관되지 않아 floater 구름이 생깁니다.

**업그레이드 U5: Stochastic Matter Field (SMF). 통계로 투표하고, seed로 3D를 만듭니다**

- **원리.** 확률적 텍스처의 정체는 개별 잎이 아니라 *통계*(Julesz ensemble)입니다. 모든 뷰가 합의할 수 있는 것은 통계이고, 개별 인스턴스는 seed로 결정론적으로 만들면 일관성이 정의상 정확합니다.
- **분류.** 영역마다 matter class를 정합니다.
  - $m\in\{\text{surface},\ \text{thin-stochastic},\ \text{particulate},\ \text{participating media}\}$
  - 근거: dreamer depth의 분산, 통계 정상성(stationarity), VLM 라벨
- **셀 토큰.** $\tau_c\in\mathbb R^{64}$에 다음을 담고, seed는 octree DNA를 참조합니다.
  - 종/재질 임베딩, 밀도
  - 방향 분포 vMF$(\mu,\kappa)$, 크기 분포(log-normal)
  - plate 색의 평균과 공분산, clumping
- **인스턴싱.** $G_{SM}(\tau_c,\xi_c,\ell)$가 실제 깊이 분포를 가진 3D Gaussian 집합(또는 instanced card)을 결정론적으로 만듭니다. LOD별 seed 계층 덕분에 무한 줌이 됩니다.
- **통계 투표(partition은 $\tau$에 적용).**
  $$\tau^\*=\arg\min_\tau\sum_k w_k\big\|\Phi(\mathcal R(G_{SM}(\tau,\xi),C_k))-\Phi(\hat x_k)\big\|^2$$
  $\Phi$는 마스크 영역의 DINOv2/VGG Gram 행렬, 색 모멘트, steerable pyramid 방향 에너지입니다(Gatys, Portilla–Simoncelli 계열 통계). 그 뒤 frozen-seed polish에서는 $\tau$만 바꾸고 seed는 건드리지 않습니다.
- **$G_{SM}$ 학습.** Infinigen nature 생성기(나무, 풀, 바위), 파티클 스냅샷(물보라, 파편), VDB 연기를 Gaussian으로 바꿔서 $(\tau,\text{instance})$ 조건부 point-set flow를 학습합니다. 90일 안에는 Infinigen의 절차적 파라미터를 CMA-ES로 dreamer 통계에 맞추는 inverse-procedural 경로로 3개 카테고리를 합니다.
- **아웃페인팅이 아닌 이유.** 빈칸을 채우는 단계가 없습니다. 물질은 통계로부터 모든 곳에서 동시에 인스턴스화되고, 일관성은 seed 결정론으로 정확히 보장됩니다.

빌려 쓸 부품: Infinigen, Gatys/Portilla–Simoncelli 통계, DINOv2, PhysGaussian(Phase 2 동역학).

---

### F6. 소유권 패치워크: 프랑켄슈타인 이음매

**증상**
- 얼굴의 반은 뷰 A, 반은 뷰 B에서 옵니다.
- 간판 글자가 이음매에서 끊기고, 창문 격자의 위상이 튑니다.
- 중간 대역은 A에서, 고대역은 B에서 오면 엣지가 형태와 어긋납니다. hybrid-image 아티팩트입니다(Oliva+ 2006).
- 이음매 띠에서 고대역 파워가 2/3로 줄어 halo가 보입니다.

**원인**
1. 텍셀 단위 Potts graph cut은 의미 구조를 모릅니다.
2. **대역 결합.** owner($b{+}1$)는 *자기 뷰 버전의* $b$ 대역에 맞춰 디테일을 결정했습니다. 그런데 sync가 $b$ 대역을 다른 owner의 것으로 바꿨다면 위상이 어긋납니다.
3. 선형 ramp 블렌드는 $\overline{\sum w^2}=2/3$이라 이음매에서 파워가 $\rho+(1-\rho)\cdot\tfrac23$으로 줄어듭니다.

**업그레이드 U6: Partition Principle + 인스턴스 잠금 + 분산 보존 이음매**

- **명제(Partition Principle).**
  - 설정: $x=\sum_kw_kx_k$, $\sum w_k=1$, $w_k\ge0$, $x_k=s+n_k$(등상관 $\rho<1$).
  - 그러면 $\mathbb E\|x\|^2=P_s+P_n[\rho+(1-\rho)\sum_kw_k^2]$입니다.
  - simplex 위에서 $\sum w_k^2\le1$이고, 등호는 one-hot일 때만 성립합니다.
  - 따라서 **대역 파워 보존(SFR=1) ⇔ 분할(one-hot)** 입니다.
  - **따름정리(K-불변성).** mixture 평균의 파워는 $K$에 따라 줄어듭니다. product-of-experts 샘플은 posterior 분산이 $P_n/K$로 줄어 온도가 $1/K$가 됩니다(과포화, 플라스틱 질감). 분할만 $K$와 무관합니다. 그래서 **앵커를 더 넣어도 절대 흐려지지 않습니다.**
- **인스턴스 잠금.** $\hat x$에서 SAM으로 인스턴스를 뽑아 세계로 lift합니다. 얼굴, 글자, 간판, 창 격자, 로고, 사람 같은 구조적 클래스는 $b\ge b_{\text{lock}}$인 모든 대역에서 인스턴스당 owner를 하나만 둡니다(인스턴스 안 authority 합이 최대인 뷰).
- **교차 대역 연속성.** 에너지에 $\lambda_x\sum_{i,b}[o_{i,b}\ne o_{i,b+1}](1-h_{i,b})$를 더합니다. $h=1$은 두 후보의 footprint 비가 2 이상일 때뿐입니다. 근접이 정당화하는 handoff만 허용합니다.
- **이음매 배치.** GraphCut Textures(Kwatra+ 2003) 방식으로, 두 후보의 gradient 크기를 이음매 비용으로 씁니다. 이음매가 엣지를 피해 저텍스처 경로로 갑니다.
- **분산 보존 블렌드(VP-blend).** 평균 0인 대역통과 계수에 대해
  $$x_b=\frac{\sum_k m_kx_{k,b}}{\sqrt{\sum_{k,l}m_km_l\rho_{kl,b}}}$$
  로 섞으면 파워가 정확히 보존됩니다. Heitz–Neyret(2018)의 histogram-preserving blending을 대역별로 일반화한 것입니다. 구조적 내용 위로는 이음매가 지나가지 않으므로 ghosting도 없습니다.
- **Soft commit.** 이음매는 최종 commit보다 한 옥타브 먼저 확정합니다. 그러면 다음 스텝의 dreamer가 아직 열린 대역에서 이음매를 조화시킬 기회를 가집니다.

---

### F7. 프레임은 이기고 샷에서 집니다

**증상**
- 카메라가 회전하면 Gaussian이 정렬 순서 변경으로 popping합니다.
- dolly-in에서 LOD가 popping합니다.
- 240배 공들인 고주파가 Nyquist 근처에서 shimmer와 moiré를 냅니다.
- 모션블러 샘플이 부족하면 계단이 보입니다.
- 프레임 단위 미감으로 고른 카메라 경로는 리듬이 어색합니다.
- 10초 orbit에서는 평가자가 비디오 모델의 불일치를 잘 못 알아채서 Elo가 비깁니다.

**원인**
- sort-based alpha blending이 primitive 단위로 정렬합니다(StopThePop이 지적한 문제).
- LOD 전환이 이산적이고, OLPF가 없습니다.
- 카메라 탐색 보상이 프레임 단위입니다.
- 비디오 모델의 약점(긴 테이크, 재방문, 스케일 변화)이 드러나지 않는 샷에서 싸웁니다.

**업그레이드 U7: Temporal-Spectral Renderer + 전장 설계 + 벤치마크 소유**

- **렌더러**
  - per-pixel hierarchical sort(StopThePop)를 씁니다. 최종 프레임은 3DGRT/3DGUT ray tracing으로 렌더합니다(rolling shutter와 왜곡 렌즈 지원).
  - **Band-fade LOD.** $W$가 Laplacian pyramid이므로 $\mathcal R=\sum_b w_b(\phi)\,\mathrm{Band}_b$, $w_b(\phi)=\mathrm{clamp}(\log_2\frac{f_{\text{Nyq}}}{f_b\phi},0,1)$로 렌더합니다. footprint에 대해 연속이라 LOD popping이 수학적으로 생길 수 없습니다.
  - OLPF(가우시안 σ≈0.35px), 셔터 적분은 stratified 32~64 샘플입니다.
- **카메라 탐색 보상을 샷 단위로 바꿉니다.**
  $$\sum_t[\mathrm{Aes}+\mathrm{Comp}+\mathrm{Parallax}]+\log p_{\text{intent}}(\Gamma)-\lambda\,\mathrm{CWS}-\mu\,\mathrm{TPI}$$
- **전장(Battleground) 6종.** 구조적으로 우리가 이기는 샷입니다.
  - B1: 60초 이상 원테이크와 재방문
  - B2: 매크로에서 와이드까지 1000× 연속 줌
  - B3: 얼어붙은 파편 속을 지나는 FPV(고밀도 시차)
  - B4: 같은 세계에서 찍은 멀티테이크와 매치컷 시퀀스(샷 간 연속성)
  - B5: 네이티브 4K/120fps와 metric DOF rack focus
  - B6: 정지한 사람 주위의 **bullet-time**. frozen-time의 상징적인 샷이고, 비디오 모델이 spurious motion으로 가장 많이 깨지는 곳입니다.
  - 10초 orbit은 parity가 목표입니다.
- **FrozenBench 공개.** frozen-time 벤치마크가 없다는 공백(G2)을 우리가 채워서 판정 기준 자체를 소유합니다.

---

## 3. 신규성 점검

### 3.1 정정: crown jewel에서 내릴 것

- **Drift = $R_{\text{eff}}$.** Barooah & Hespanha(IEEE CSM 2007; IEEE TSP 2008)가 이미 보였습니다. 상대 측정으로 추정할 때 BLUE 오차분산이 effective resistance와 같다는 결과입니다. SLAM에서도 Khosoussi+(2019)가 그래프 연결성과 추정 정확도를 연결했습니다.
  - 우리 몫은 이걸 *생성 모델의 conditioning graph*에 적용해서 아웃페인팅 drift가 길이에 선형인 이유를 설명하는 데까지입니다.
  - 신경망의 soft constraint에 대해서는 정리가 아니라 모델(analogy)입니다.
- **Spectral Deficit의 수식.** 등상관 변수 평균의 분산 공식이라 교과서 수준입니다. 텍스처 매핑의 "blending 대신 selection" 원칙도 Lempitsky & Ivanov(2007), Waechter+(2014)에 있습니다.
  - 우리 몫은 이 공식을 대역별·σ별 *예측 도구*로 쓰는 것, 그리고 Partition Principle의 K-불변성입니다.
- **대역별 관할 메커니즘.** Factorized Diffusion(Geng+ CVPR'24)은 한 모델 안에서 주파수 대역마다 다른 프롬프트를 배타적으로 적용합니다. eDiff-I와 Wan 2.2 MoE는 σ(대략 대역)별 전문가를 둡니다.
  - 우리 몫은 *이종 prior*(이미지, 멀티뷰, 3D, 물리, 카메라) 사이의 관할을 calibration 측정으로 배정하고, 그 공통 공간을 세계로 둔다는 점입니다.

### 3.2 겹침표와 빌려 쓸 부품

| 구성요소 | 겹치는 선행 작업 | 빌려 쓸 부품 | 진짜 다른 지점 (한 문장) |
|---|---|---|---|
| 세계 생성 후 렌더 | Marble 1.1/Studio, Lyra 2.0, Lyra(1), HY-World 2.0/2.1, FlashWorld, CAT3D/Bolt3D, Wonderland | 2DGS·3DGUT 렌더러, DA3·VGGT-Ω·WorldMirror 2.0 재구성 | 이들은 비디오나 멀티뷰 teacher의 화질 상한을 물려받고, 우리는 이미지 prior를 분할 소유권으로 세계에 직접 bake해서 프레임 상한을 이미지 모델 수준으로 올린다. |
| 동기 디노이징 | SyncTweedies, MultiDiffusion, DiffCollage, TexFusion, SyncMVD | $\hat x$ sync 루프 | 평균 대신 σ-결정대역 분할을 쓰고, world-scale causality($\lambda_k$ 지연)로 누가 언제 결정할지를 정한다. |
| 시점 품질 기반 영역 선택 | TEXTure(keep/refine/generate), Text2Tex, Paint3D | 시점 각도 휴리스틱 | 그들은 순차 인페인팅의 마스크로 쓰고, 우리는 동시 디노이징 안에서 대역별 소유권으로 쓴다. |
| 레이아웃 | WorldGen(LLM layout), HY-Pano 2.0, Zero123++/Era3D grid | 캔버스 생성기, grid trick | 단일 캔버스는 L0만 결정하고 그 위 대역은 전부 열어 둔다(층위 분리). |
| Anchored noise | How I Warped Your Noise, Go-with-the-Flow, Integral Noise Warping(Deng+ ICLR'25), SyncNoise | Brownian-bridge 계층 샘플링 | 노이즈를 ambient octree DNA에 둬서 기하 정제에도 유지되고, 반복 간에도 고정해서 F3을 막는다. |
| 고정점 dream-project | IDU, Difix3D+, ISM/SDI, DreamTime | 1-step fixer, σ annealing | frozen-seed 결정론 사상과 band-commit MH로 반복 평균 결손 $\gamma/(2-\gamma)$를 없앤다. |
| Latent/VAE | EQ-VAE, Skorokhodov+ diffusability, Latent Radiance Fields | 등변 정규화 손실 | 디코딩을 texel 공간으로 옮겨(WDD) 시점 등변성을 학습이 아니라 구조로 얻는다. |
| 사진/조명 분리 | RGB↔X, DiffusionRenderer, NeRF-W | intrinsic dreamer, gauge | 이미지 prior의 분포를 세계 인자와 카메라 인자로 나눠, 사진 효과와 조명 강조를 샷에 귀속시킨다. |
| 확률적 물질 | Infinigen, Gatys/Portilla–Simoncelli | 절차적 생성기, 텍스처 통계 | 일관성 단위를 인스턴스에서 '앙상블 통계 + seed'로 바꾸고, 통계는 이미지 prior가 투표하고 3D는 결정론적으로 만든다. |
| 다중 prior | Magic123, DreamCraft3D, Factorized Diffusion, eDiff-I | — | 측정한 신뢰도로 배타적 관할을 주고, black-box API dreamer까지 받아들이는 prior-agnostic 연방이다. |
| 카메라 | CameraCtrl, MotionCtrl, E.T., DataDoP/GenDoP, CameraBench, Director3D, VFX camera shake | 궤적 데이터, taxonomy | 고정된 세계 위에서 탐색과 재촬영을 하고, rig PSD를 평가 지표(CWS)까지 묶는다. shake 자체는 VFX의 기존 관행이다. |
| Phase 2 | Shape of Motion, MoVieS, CAT4D, 4DNeX, Diff4Splat, PhysGaussian, NGFF, BulletTime | 4D 재구성, 물리 | event-first, 정체성 보존을 표현 수준에서 강제, 시간 축 연방(외관=이미지, 저주파 운동=비디오, 접촉=물리). |
| 플라이휠 | Expert Iteration, AlphaZero, FlashWorld(DMD2 cross-mode), Lyra self-distillation | DMD2 | teacher가 네트워크가 아니라 이미지 prior를 judge로 쓰는 *탐색*이고, 3D GT 없이 세계 분포를 끌어올린다. |
| 평가 | GeCo, MEt3R, ViewBench LCE, SCoPE, WorldScore, VBench-2.0, AA Arena | 지표 구현 | frozen-time 시네마틱 전용 벤치마크로 기하, 스펙트럼, 시간, 카메라 무게, Elo를 묶은 것은 처음이다. |

### 3.3 v2 crown jewels (정직한 버전)

1. **σ-대역 분할 소유권 + Partition Principle(K-불변)**
2. **World-Scale Causality** (footprint-staggered log-SNR + ownership window)
3. **Frozen-Seed Band-Commit** (반복 축의 결손 제거)
4. **Camera Owns Photography** (plate dreamer + Gaffer)
5. **Stochastic Matter Field**
6. **Calibrated, prior-agnostic Spectral Federalism**

여기에 전략 자산으로 **FrozenBench**가 더해집니다.

---

## 4. 90일 결정적 실험 5개

전제: 팀 5~6명(diffusion 2, 그래픽스/렌더 1, 3D 비전 1, 평가/인프라 1, 창업자), H100 8~16장.

### X1. 분할 vs 평균 vs 곱: 융합 이론의 생사 (1–6주)

**가설**
- H1a: 평균과 L2 재구성의 대역 파워는 $1-\eta_b(1-\rho_b)(1-1/K)$ (단 $\eta_b=P_n/(P_s+P_n)$)를 ±0.1 안에서 따른다.
- H1b: product-of-experts는 채도와 대비가 부푼다(온도 $1/K$).
- H1c: 분할은 $K$와 무관하게 SFR ≥ 0.9를 유지한다.
- H1d: fresh-seed polish의 고대역 파워는 $\gamma/(2-\gamma)$로 수렴하고, frozen-seed polish는 유지된다.

**셋업**
- **기하를 고정해서 생성 문제와 분리합니다.** 20장면을 씁니다. ScanNet++ DSLR 10장면(laser mesh), DL3DV 10장면(DA3 + 2DGS mesh)입니다.
- Dreamer: Qwen-Image(Apache-2.0) + depth/normal ControlNet + `<plate>` 프롬프트. LoRA는 4주차부터 붙입니다.
- $K\in\{8,24,48\}$
- Arms:
  - A1: 독립 생성 후 L2 2DGS 재구성 (CAT3D식)
  - A2: SyncTweedies식 $\hat x$ 평균
  - A3: product-of-experts (precision fusion 후 샘플)
  - A4: σ-대역 partition (pixel sync)
  - A5: A4 + octree anchored noise
  - A6: A5 + polish 세 가지 버전 {fresh seed / frozen seed / band-commit}
  - A7(선택): A6 + black-box API dreamer를 polish 고대역 owner 후보로 추가

**측정**
- 장면당 held-out 카메라 16개에서 $\mathrm{SFR_{prior}}(f)$ (radial 8대역)
- 실측 $\rho_b,\eta_b$로 계산한 예측 SFR
- held-out seed로 잰 $\mathrm{DR_{rel}}$, MEt3R
- 채도·대비 팽창률
- 인간 2AFC: A6 프레임 vs 같은 카메라에서 dreamer가 그린 원본

**성공 기준**
- (i) A1/A2에서 $|\Delta\mathrm{SFR}|$(실측 − 예측) ≤ 0.1
- (ii) A5의 SFR ≥ 0.9 (0.35 cyc/px 이하 모든 대역), $K$=8→48에서 변화 ≤ 0.05
- (iii) 20회 반복 후 fresh-seed의 최고대역 SFR ≤ 0.3, frozen과 commit은 ≥ 0.85
- (iv) 2AFC에서 A6가 dreamer 원본 대비 ≥ 40%(근사 parity), A1 대비 ≥ 75%

**실패 시 다음 수**
- (i)이 실패하면 latent 스펙트럼으로 이론을 재보정합니다(U2c).
- (ii)가 이음매 때문에 실패하면 U6을 앞당깁니다. 정렬 오차(ghosting) 때문이면 dream jitter $u$를 flow로 추정해 back-projection 전에 정렬합니다.
- (iv)가 실패하면 RE-FT로 $\rho$를 올리고, 그래도 안 되면 WDD를 앞당겨 latent Nyquist 위를 세계가 직접 생성하게 합니다.

**비용:** 약 300 H100h.

### X2. 텍스트에서 세계로: Single-Canvas + World-Scale Causality (5–10주)

**가설.** v1의 BIRTH(모든 뷰가 $\sigma_{\max}$에서 동시 출발, 전역 프롬프트)는 레이아웃 콜라주와 객체 중복을 만든다. U1은 이를 없애면서 프레임 품질을 유지한다.

**셋업**
- 프롬프트 40개: inside-out 20, outside-in 20. 그중 6개는 사람을 포함합니다.
- $\Pi$: orbit, dolly, crane tube
- Arms:
  - B0: v1 BIRTH
  - B1: + local prompt
  - B2: + single-canvas layout
  - B3: + FSS(footprint-staggered schedule) + ownership window
  - B4: + 기하 한 옥타브 선행 + parallax band
- 외부 기준선: FlashWorld, HY-World 2.0, (가능하면) Marble 1.1 export를 같은 경로로 렌더합니다.

**측정**
- 중복률: VLM과 detector가 360° 렌더에서 hero 인스턴스 수를 셉니다.
- Janus율, VLM judge의 레이아웃 일관성
- GeCo(렌더라서 바닥값 확인용), $\mathrm{SFR_{prior}}$, $\mathrm{DR_{rel}}$
- Elo: 기준선 세계들 대비, 그리고 hero 뷰 단일 이미지 대비

**성공 기준**
- B4의 중복과 Janus ≤ 5% (B0는 30% 이상으로 예상)
- Elo(B4) − Elo(최고 기준선) ≥ +150
- hero 뷰 $\mathrm{DR_{rel}}$ ≤ 1.3

**실패 시 다음 수**
- lift 오차가 문제면 시트를 여러 장 쓰고 VGGT-Ω로 재구성합니다.
- 클로즈업의 맥락 부족이 남으면 $\kappa$를 키우고, 와이드 렌더를 IP-Adapter 레퍼런스로 넣습니다.
- 기하가 떨리면 기하 소유를 한 옥타브 coarse하게 올리고 나머지는 normal로 bake합니다.

### X3. Cardboard Test: Stochastic Matter Field (6–11주)

**가설.** 표면 소유권으로 처리한 잎과 물보라는 Cardboard Index가 높다. SMF는 CI를 50% 이상 줄이고, 2AFC에서 65% 이상 이긴다.

**셋업**
- 12장면: 숲 가장자리, 밀밭, 얼어붙은 해일 물보라, 얼어붙은 폭발 파편, 털 클로즈업, 연기
- Infinigen(나무, 풀), 파티클 인스턴서(물보라, 파편), VDB와 procedural turbulence(연기)
- $\tau$는 CMA-ES로 맞춥니다(클러스터당 렌더 2k회 이하). 그 뒤 frozen-seed polish를 돌립니다.

**측정**
- **Cardboard Index(CI):** 확률적 물질 마스크 안에서 CoTracker3 track 패치(반경 15px)를 봅니다. 예상 시차가 1px보다 큰데도 단일 homography 잔차가 0.3px 미만으로 설명되는 패치의 비율입니다.
- TPI
- Gram/DINO 통계 거리를 dreamer 자신의 뷰 간 통계 분산과 비교
- 2AFC와 "가짜 같다" 응답률

**성공 기준:** CI 50% 이상 감소, 통계 거리 ≤ dreamer 뷰 간 분산, 2AFC ≥ 65%.

**실패 시 다음 수**
- 통계는 맞는데 "CG 나무"처럼 보이면 $G_{SM}$을 실사 재구성(DL3DV 식생, Stereo4D)으로 학습하고 DR distill을 추가합니다.
- CMA-ES가 너무 느리면 amortized $\tau$-encoder로 바꿉니다.

### X4. 카메라와 렌더러만 바꿨을 때 (2–6주, 병렬, 저비용)

**가설**
- (a) popping과 shimmer가 'CG 느낌'의 1차 원인이다. U7 렌더러로 TPI가 3배 이상 줄어든다.
- (b) 질량 있는 카메라가 유령 드론보다 무게감과 미감에서 이긴다.
- (c) 원본 실사 비디오와 섞은 real-or-render 테스트에서 35% 이상을 속인다.

**셋업**
- **세계를 고정해서 생성 문제와 분리합니다.** DL3DV 원본 비디오가 있는 10장면을 2DGS로 재구성합니다.
- DataDoP와 CameraBench 300샷에서 rig별 jerk PSD를 피팅합니다.
- Arms: 렌더러 {vanilla 3DGS, U7} × 카메라 {B-spline 유령, kinetic(rig별)}

**측정:** TPI, CWS, 2AFC(일반 평가자 30명 + 촬영감독 5명), real-or-render 혼동률.

**성공 기준:** TPI 3배 이상 감소, CWS 50% 이상 감소, kinetic 2AFC ≥ 65%, 혼동률 ≥ 35%.

**실패 시 다음 수**
- 흔들림이 '아마추어 촬영'으로 읽히면 PSD 진폭을 초점거리, shot intent, 장르에 조건부로 둡니다.
- popping이 남으면 최종 프레임 전체를 3DGRT로 렌더합니다.

### X5. FrozenBench v0: 정면 승부 (9–13주)

**가설.** WSSC v2가 frozen-time 샷에서 Wan 3.0, Seedance 2.5, MiniMax H3, Kling 3.0 Omni, Veo 3.1 / Gemini Omni 1.1 Flash를 이긴다. 기하 지표는 3배 이상, 10초 orbit은 Elo parity, 전장 샷은 65% 이상이다.

**셋업**
- 샷: 프롬프트 40개 × {10s orbit, 30s loop-return, 60s 원테이크(매크로↔와이드)}, 그리고 bullet-time 사람 8개
- 두 트랙을 돌립니다.
  - T2V: 텍스트만 줍니다.
  - I2V: 우리 hero 프레임을 첫 프레임으로 줘서 콘텐츠 조건을 맞춥니다.
- 경쟁 모델은 seed 3개 중 중립 평가자가 고른 best-of-3를 씁니다. 30초와 60초 샷은 각 모델의 최대 단일 패스와 공식 extend 기능을 씁니다.

**지표:** GeCo fused, MEt3R, LCE, FV, $\mathrm{SFR_{real}}$, $\mathrm{DR_{rel}}$(held-out seed), TPI, CWS, Elo(pairwise 2,000건 이상, 일반 평가자 30명 + 촬영감독 8명).

**성공 기준**
- GeCo ≤ 0.04 (경쟁 모델은 0.07~0.27로 예상)
- LCE PSNR ≥ 35dB
- 10초 샷 승률 ≥ 45%, 전장 샷 ≥ 65%
- 60초 원테이크에서 촬영감독 선호 ≥ 70%

**실패 시 다음 수.** 카테고리별로 패배를 분해합니다.
- 미감에서 졌다면 Gaffer와 그레이드 학습을 강화합니다.
- 디테일에서 졌다면 WDD와 API dreamer의 고대역 소유를 늘립니다.
- 사람에서 졌다면 얼굴·몸 전용 owner와 SAM 3D Body 기하를 넣습니다.
- 결과와 상관없이 FrozenBench는 공개합니다.

---

## 5. 개정 제안서 전체본

# WSSC v2: 세계를 한 번 꿈꾸고, 대역마다 가장 잘 아는 자가 한 번만 결정하고, 사진은 카메라가 찍는다

> Kling과 Seedance는 프레임마다 우주를 새로 만드는 바쁜 신입니다. v1은 우주를 한 번만 만드는 제작자였습니다. v2는 그 제작자에게 **세트 디자이너(single canvas), 미술팀(partition), 소품팀(stochastic matter), 촬영감독(kinetic camera), 조명감독(gaffer)** 을 붙입니다. 할리우드가 분업을 하는 이유가 그대로 수식으로 들어옵니다.

## 0. 원칙 여섯 개

1. **Decide before you draw.** 결정 변수($W$, $C$, event)를 명시적으로 두고, 원래 공간에서 이산적으로 결정하고, 결정론적으로 렌더합니다.
2. **Partition, never average or multiply.** 대역마다 owner는 하나입니다. 평균은 파워를 잃고, 곱은 온도를 잃습니다.
3. **World-scale causality.** 세계는 world-scale coarse-to-fine으로 태어납니다. 클로즈업은 맥락을 결정하지 않습니다.
4. **Seeds are DNA.** 한 번 뽑고 끝까지 고정합니다.
5. **Bake, don't paint.** 모든 개선은 $W$로 들어갑니다. 프레임마다 새로 뽑는 랜덤은 grain뿐입니다. per-frame neural enhancer는 쓰지 않습니다. enhancer는 polish 단계의 dreamer로만 씁니다.
6. **Camera owns photography.** 렌즈, 센서, 그레이드, 조명 강조는 $C$와 $\mathcal S$의 몫입니다.

## 1. 정식화와 정리

### 1.1 인수분해
$$
x_t=\mathcal S_{\text{shot}}\!\Big(\int_{\text{shutter}}\!\!\int_{\text{aperture}}\!\mathcal R\big(W,\,L_{\text{shot}},\,C(t+\tau,u)\big)\,du\,d\tau;\ \xi^{\text{grain}}_t\Big)
$$
$$
G\sim p(G\mid y),\quad W\sim p(W\mid y,G),\quad C\sim p(C\mid y,W,\text{intent}),\quad L_{\text{shot}}\sim p(L\mid W,C,\text{style})
$$
- $G$는 전역 변수(DC 대역)입니다.
- 확률성은 $G$, $W$(DNA $\xi$ 포함), $C$, $L_{\text{shot}}$, grain에만 있습니다. 렌더러 $\mathcal R$은 결정론적입니다.

### 1.2 구조적 보장
- **T1 Type-safe Consistency(샷 단위).** 한 샷 안의 모든 프레임은 하나의 plenoptic function $L_{W,L_{\text{shot}}}$에서 뽑은 샘플입니다. 남는 오차는 렌더러 aliasing뿐이고, 이건 신호처리(prefilter, OLPF)로 해결합니다.
- **T2 Loop Closure.** $C_t=C_s$이면 $x_t=x_s$입니다(grain 제외).
- **T3 Free Resolution/FPS/Duration + Dedup.** 생성 비용은 $O(|W|)$입니다. redundancy ratio $r=THW/U\approx10\text{–}25$(10s, 24fps, 1080p 정적 장면)이므로, unique texel 하나당 그만큼 더 공들일 수 있습니다.
- **M1 Drift 모델.** Barooah–Hespanha의 결과를 적용합니다. 조건부 그래프에서 $\mathrm{Var}[\theta_i-\theta_j]=\varepsilon^2R_{\text{eff}}(i,j)$입니다.
  - path graph(아웃페인팅, AR)는 $|i-j|$
  - band graph(chunk 이어 붙이기)는 $\Theta(|i-j|)$
  - star graph(우리)는 $O(1)$

### 1.3 융합 이론 (v2의 수학적 핵심)
- **T5 Spectral Deficit.**
  $$\mathrm{SFR}_b=1-\eta_b(1-\rho_b)(1-1/K),\qquad \eta_b=P_n/(P_s+P_n)$$
- **P1 Partition Principle.** convex 결합 $x=\sum w_kx_k$의 대역 파워는 $P_s+P_n[\rho+(1-\rho)\sum w_k^2]$입니다. 따라서 SFR=1이 되는 것은 one-hot일 때뿐입니다.
  - **따름정리(K-불변성):** 분할의 결과 통계는 앵커 수 $K$와 무관합니다. 카메라를 늘려도 흐려지지 않습니다.
  - 대조: product-of-experts는 posterior 분산이 $P_n/K$로 줄어듭니다(온도 $1/K$, SDS류의 과포화).
- **L1 Proximity Precedence와 그 역설.** 같은 $\lambda$에서 가까운 뷰가 더 미세한 세계 스케일을 더 먼저, 그리고 맥락 없이 결정합니다.
- **T6 World-Scale Causality.**
  $$\lambda_k=\lambda-\alpha\ln(\phi_{\text{ref}}/\phi_k)$$
  로 두면 세계 대역 $f^w$의 결정 시각이 모든 뷰에서 같아집니다. 그리고 ownership window $\mathcal W_k=[\kappa/W_k,f_{\text{Nyq}}]$가 맥락 없는 소유를 막습니다.
- **L2 Iteration Deficit.** seed를 새로 뽑는 반복 평균의 정상상태 파워는 $\gamma/(2-\gamma)\cdot P_h$입니다. 따라서 seed는 고정하고, 결정 대역 위는 commit 방식으로 처리합니다.
- **L3 Parallax Relevance.** 기하 디테일은 $FB\,\delta z/z^2\ge0.5$px일 때만 기하로 결정하고, 나머지는 normal/texel로 bake합니다.
- **차원 직관.** 정적 장면 비디오의 consistent manifold $\dim\mathcal M\approx\dim W+6T\ll THW\cdot3$입니다. 우리 생성기의 치역은 정의상 $\mathcal M$ 안에 있습니다.

## 2. 병리학: 두 진영의 병과 우리의 처방

| 진영 | 증상 | 수학적 원인 | v2의 구조적 제거 |
|---|---|---|---|
| 비디오 모델 | 일관성 붕괴 | 상태 변수 없음, path/band graph drift | $W$를 상태로, star graph |
| 비디오 모델 | 뭉개짐 | 위치 hedging $I*p_u$, latent 시간 Nyquist 2~3Hz, per-pixel compute 부족 | 이산 소유권, 결정론 렌더, dedup |
| 비디오 모델 | 과한 부드러움 | 위상·타이밍 평균 $\mathbb E[H(t-\tau)]=F_\tau$, 비강체 flow | Kinetic Camera, Event-first, 강체 렌더 |
| 세계-먼저 계열 (CAT3D, Lyra, Marble) | 흐림 | 재구성 = 평균(T5) | Partition(P1) |
| 세계-먼저 계열 | 비디오 화질 상속 | 비디오 teacher(FlashWorld, Lyra 2.0)의 상한 | 이미지 prior를 세계에 직접 bake |
| 세계-먼저 계열 | CG 룩 | 조명 bake, 카메라·센서 부재, 반사 불가 | Camera Owns Photography, Gaffer, 3DGRT |

> 업계의 결론("명시적 세계로는 시네마틱 화질이 안 나온다", World Labs가 Atlas로 방향을 튼 것)은 **평균하는 융합**과 **사진을 세계로 굽는 실수**에서 나온 결론입니다. v2는 둘 다 구조적으로 제거합니다. 바로 그 지점이 우리가 거는 베팅입니다.

## 3. 표현: Hierarchical Surfel–Voxel World (HSVW)

| 구성요소 | 규모 (데모 기준) | 원소당 텐서 | 역할 |
|---|---|---|---|
| **Ambient octree** (sparse voxel) | leaf $2\text{–}8\times10^7$, 최소 셀 1~2mm(Π 근처)에서 1m(원경)까지 | occ/SDF fp16; instance id int32; matter class uint8; **DNA** $z^{(\ell)}\in\mathbb R^{16}$ fp16, 계층 일관 $z_{\text{parent}}=\sum_c\sqrt{A_c/A}\,z_c$ (top-down 조건부 샘플링으로 lazy refine); commit bitmask uint8 ($B=8$); 원소별 $\sigma_i$ fp16 | 비계, 노이즈 DNA, 이종 σ |
| **Surfels** (2DGS) | $2\times10^6\text{–}2\times10^7$ | $\mu\in\mathbb R^3,q\in\mathbb R^4,s\in\mathbb R^2,\alpha$; **Laplacian texel pyramid** $\{T_b\in\mathbb R^{P_b\times P_b\times10}\}$ (plate radiance 3, albedo 3, roughness 1, metallic 1, tangent normal 2), $P_b$는 footprint에 적응; latent texel $Z\in\mathbb R^{p\times p\times16}$(U2a arm); owner $o\in\mathbb Z^B$ uint16, authority $a\in\mathbb R^B$ fp16 | 표면, 대역 = mip level |
| **SMF cells** | $10^4\text{–}10^6$ | $\tau\in\mathbb R^{64}$, DNA 참조, bbox, LOD | 확률적 물질. 렌더 시 $G_{SM}$로 결정론적 인스턴싱 |
| **Media volume** | VDB | density, albedo, emission, turbulence seed | 연기, 안개, 구름 |
| **Matte shells** | 4~8층 | equirect RGBA 8K×4K + depth offset | 원경, 하늘, forced perspective |
| **World lighting** | — | env HDR 2K fp16, sun(dir, angular radius, radiance), practicals | 세계 조명 |
| **Shot lights** (Gaffer) | 샷당 8개 이하 | type, pose $SE(3)$, size, $I$, CCT, softness, gobo | 샷 조명 |
| **Camera** | 샷당 | $SE(3)$ cumulative cubic B-spline(초당 knot 10~20), optics track(f, focus, N, shutter°, squeeze, breathing), rig state, sensor profile(OLPF σ, read noise, grain seed, halation, LUT) | 촬영 |

- **메모리:** octree 약 2.4GB + surfel(평균 약 1KB × $10^7$) 약 10GB + shells 약 2GB로 합계 약 15GB입니다. 80GB GPU에 여유 있게 들어갑니다.
- **렌더러:** 2DGS + StopThePop(미리보기), 3DGRT/3DGUT(최종, 반사·굴절·rolling shutter), band-fade LOD, OLPF, stratified 셔터·조리개 적분(32~64 spp).

## 4. 엔진: WSSC v2 파이프라인

**데이터 흐름**
1. brief → 전역 변수 $G$ → scene graph와 shot list → 셋업 가능 집합 $\Pi$
2. **L0 단일 캔버스** → Lift → $W(\lambda_{L0})$
3. **[BIRTH 루프]** G-buffer와 노이즈 상태 렌더 → plate dreamer → explain-away → 기하 선행 → partition commit → SMF 투표
4. **[POLISH]** frozen-seed band-commit + adversarial camera
5. **[SHOOT]** 카메라 탐색 → Gaffer → 센서 → 프레임, 그리고 세계·카메라 데이터 export

**Authority (v2)**
$$
a_k(i,b)=\log\tfrac{1}{\phi_k(i)}+\log T_k(i)-\lambda_u\hat\sigma^2_k(i,b)+\lambda_e e_k(i)-\lambda_{\text{dof}}\beta_k(i)+\lambda_h h_k(b)+\lambda_c c_k(i)-\infty\cdot[f^{\text{img}}_k(b)\notin\mathcal W_k]
$$
- $\beta$: 추정 defocus
- $c_k(i)$: 지역 내용과 세계 semantic 라벨의 CLIP/VLM 일치도

**Owner 에너지 (대역 순서대로 α-expansion을 surfel kNN 그래프 위에서 풉니다)**
$$
E(o)=\sum_{i,b}-a_{o_{i,b}}(i,b)+\lambda_s\!\!\sum_{(i,j),b}\!w^{\text{grad}}_{ij}[o_{i,b}\ne o_{j,b}]+\lambda_x\!\sum_{i,b}[o_{i,b}\ne o_{i,b+1}](1-h_{i,b})+\infty\!\!\sum_{\text{inst}\in\mathcal S}\sum_{i,j\in\text{inst},\,b\ge b_{\text{lock}}}\![o_{i,b}\ne o_{j,b}]
$$

**의사코드**
```text
procedure WSSC_v2(brief y, style s, priors P, calib K):
  # Stage 0. Pre-production: DC 대역 = 전역 변수
  G     ← DecideGlobals(y, s)               # 태양 방향, 시간대, 팔레트, 렌즈 패키지, 화면비
  SG    ← SceneGraph(y, G)                  # 객체, 재질, matter class, symbolic layout
  Shots ← ShotList(y, SG)                   # intent 문장: "slow push-in → whip pan"
  Π     ← ShootableSet(Shots, IntentPrior, r_Π)

  # Stage 1. L0: 단일 캔버스 레이아웃 (한 번에 결정)
  Canv  ← LayoutCanvases(SG, G)             # equirect 또는 3×3 sheet; 여러 장이면 동시 생성 + L0 partition
  W     ← Lift(Canv)                        # DA3/VGGT-Ω/MoGe-2, hero 물체 image-to-3D, SAM instance lift
  W.matter ← ClassifyMatter(W, Canv)
  ξ     ← DNA(W)                            # ambient octree, 면적 정규화, 영구 고정
  V     ← AnchorCover(Π, W)                 # Nyquist coverage: Π의 모든 (i,b)에 대해
                                            #   in-window owner k, φ_k(i) ≤ min_{C∈Π} φ_C(i);
                                            #   커버 못 하는 대역은 WDD(seeded)에 배정

  # Stage 2. BIRTH: world-scale causal 동기 디노이징
  for λ in Schedule(λ_L0 → λ_max):
    B_now ← DecisionBands(λ, K)             # 실측 결정 대역 (U2c)
    parallel for k in V with λ_k = λ − α·ln r_k ≥ λ_join:
      if joining(k): z_k ← a(λ_k)·R(W_dec, C_k) + s(λ_k)·RenderNoise(ξ, C_k)
      if new_octave: prompt_k ← RenderCaption(W, C_k)
      (x̂_k, d̂_k, n̂_k, β_k, σ̂²_k, inst_k) ← Dreamer_plate(z_k, λ_k, prompt_k, Gbuf(W, C_k))
    (g, spec) ← ExplainAway(W, {x̂_k})        # gauge, 반사 = 3DGRT(W), 렌즈 잔차 제거
    if GeometryOctaveDue(λ):                 # 기하가 외관보다 한 옥타브 앞섬
      W.geo ← PartitionFuse({d̂_k}, bands ≤ f_par) ⊕ FreeSpaceVeto ; ReprojectDecided(W)
    for b in B_now:
      o_b ← Owners(E(o))                     # window, instance-lock, cross-band, seam cost
      W_b ← VPBlend({BP_k(x̂_k)_b}, o_b, λ_b ∝ 1/f_b, ρ_b) ; Commit(b, soft = one octave early)
    for cell in SMF with stat-band ∈ B_now: τ_cell ← StatVote({x̂_k}, cell)
    for k: z_k ← DDIM(z_k; x̃_k = R(W_dec, C_k) ⊕ x̂_k|undecided)   # ε는 (z − a·x̃)/s로 내재

  # Stage 3. POLISH: frozen-seed band-commit
  for n in 1..N:
    σ_n ← anneal(n) ; V_n ← V ∪ AdversarialCameras(W, Π)
    Y ← PartitionSync(W, {D_σn(R(W,C); ξ)}_{C∈V_n})    # ξ 고정, 1~4 step distilled dreamer (+ API dreamer)
    W_{≤b_c} ← (1−γ)·W_{≤b_c} + γ·Y_{≤b_c}              # 합의된 대역만 평균
    for b > b_c(σ_n): MH_commit(W_b → Y_b ; E = Σ DR_calib + Seam + Stretch)
    if region R has DR_rel > 1.3: Reopen(R, b ≥ b*, new DNA version)
    stop if max_{C∈Π̂} DR_rel(held-out seeds) ≤ 1.2 and min SFR_prior ≥ 0.9

  # Stage 4. SHOOT
  for shot in Shots:
    Γ ← CameraSearch(W, shot, rig, reward_shot)       # 저해상도 후보 10^3~10^4개
    L ← Gaffer(W, Γ, shot, G)
    frames ← [ S( ∫∫ R(W, L, Γ(t+τ, u)) du dτ ; grain_t ) for t ]
  return frames, W, {Γ}, {L}                           # 세계와 카메라 데이터 = VFX export 제품
```

**아웃페인팅이 아닌 이유 (v2 전체)**
- **순서의 기준이 다릅니다.** 동결은 *세계 대역* 순서로 일어나고, 모든 영역에서 동시에 진행됩니다.
- **정보 흐름이 다릅니다.** 결정 중인 대역 안에서는 정보가 양방향으로 흐릅니다(뒷면 뷰도 거부권을 가집니다).
- **그래프가 다릅니다.** 조건부 그래프는 항상 star입니다.
- **확장 방식이 다릅니다.** 부분 completion이 필요한 경우는 두 가지뿐입니다. $\Pi$를 나중에 넓힐 때와 reopen할 때입니다. 둘 다 원소별 heterogeneous-σ(world-space Diffusion Forcing)로 처리하고, 미결정 대역은 계속 수정 가능한 상태로 남습니다.

**세계 하나당 연산 예산 (H100 기준)**

| 단계 | 내용 | H100-시간 |
|---|---|---|
| Layout과 Lift | 4K 캔버스 1~3장, DA3/VGGT-Ω, image-to-3D 1~5개 | 0.1 |
| BIRTH | 96뷰 × 28스텝 × 0.65(FSS 절감) ≈ forward 1,750회 (20B, 1024²) + 기하·소유권 | 0.6~0.8 |
| SMF | CMA-ES ≤ 2k 렌더/클러스터 | 0.2~0.5 |
| POLISH | 30회 × 112뷰 × 2-step distilled | 0.3~0.6 |
| SHOOT | 후보 $10^4$개 저해상도 탐색 + 4K 64spp 최종 렌더 | 0.1~0.3 |
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
