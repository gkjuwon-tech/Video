> **출처 메모.** 설계 워크숍에서 [표현 / representation] 렌즈 담당이 다른 렌즈와 독립적으로 쓴 원안(STRATA v0.1)이다. 렌즈 간 충돌과 최종 결정은 [`docs/PARADIGM.md`](../PARADIGM.md)의 판결록(§0.5)을 따른다.

---

# STRATA: 촬영할 수 있는 세계의 매개체를 처음부터 다시 만든다
*표현(representation) 급진주의 렌즈를 맡은 공동창업자의 설계 메모 v0.1*
*부제: 카메라는 세계를 그리지 않는다. 세계에 질의(query)한다.*

---

## 0. 우리의 적은 "건망증 걸린 창조주"다

비디오 디퓨전 모델을 사람처럼 그려 보면 이렇다. 매 프레임 우주를 새로 창조하는 전지전능한 신인데, 건망증이 심하다. 3초 전에 만든 머그컵 손잡이가 왼쪽이었는지 오른쪽이었는지 기억하지 못한다. 그래서 카메라가 돌면 손잡이가 슬그머니 반대편으로 옮겨 간다(일관성 붕괴). 들킬까 봐 경계선을 살짝 녹이고(뭉개짐), 녹인 티가 안 나게 모든 움직임을 버터처럼 문질러 버린다(과한 스무딩).

창업자가 꼽은 3대 고질병은 **병 하나에서 나온 증상 세 개**다. 병명은 *기억해야 할 세계가 존재하지 않음(no persistent world state)*이다. 프레임과 프레임 사이에 실재(實在)가 없으니 모델은 매 프레임 "그럴듯함"을 다시 협상해야 한다. 그 협상 비용이 일관성, 선명도, 움직임의 진정성으로 빠져나간다.

그래서 처방도 하나다. **세계를 먼저 존재하게 하고, 카메라는 그 세계에 질의만 하게 한다.** 이 렌즈의 주장은 다음과 같다.

> Phase 1의 승패는 "어떤 생성 모델을 쓰느냐"보다 "무엇을 생성하느냐(매개체)"에서 먼저 갈린다. 매개체가 틀리면 아무리 거대한 모델이라도 틀린 것을 아주 잘 만들 뿐이다.

이 문서가 제안하는 다섯 개의 축은 다음과 같다.

1. **생성의 순서축을 시간(time)에서 스케일(scale)로 바꾼다.** 순차 아웃페인팅과 비디오 디퓨전은 둘 다 "시간 순서 체인"이다. 그래서 창업자 말대로 둘은 같은 원리이고, 프레임을 덜 만드는 아웃페인팅은 퇴화다. 우리는 "스케일 순서 트리"로 간다. 패러다임 전환의 정체는 이 한 줄이다.
2. **세계는 시드(seed)의 순수 함수(pure function)다.** 카메라가 어떤 순서로 돌아다녀도 세계는 비트 단위로 같다. 마인크래프트 월드 시드 원리를 디퓨전에 옮겨 심는 셈이다.
3. **줌인은 디노이징이다 (Scale-Time Duality).** LOD 피라미드와 생성 궤적은 같은 대상이다.
4. **렌즈가 세계에게 얼마나 디테일해져야 하는지 알려준다 (Lens-gated bands).** 보케 뒤나 모션블러 속에는 디테일을 생성할 필요조차 없다.
5. **뷰 의존 효과는 외우지 않고 물리에서 유도한다.** 이미지 모델이 미감을 위해 치는 "조명 치팅"은 영화 현장처럼 *샷 조명 리그*로 합법화한다.

매개체 이름은 **STRATA(지층)**다. 세계를 주파수 대역의 지층으로 쌓는다. 기반암(bedrock)에는 산맥과 건물 배치가, 표토(topsoil)에는 이끼 포자와 니트의 올이 있다. 한 번 퇴적(commit)된 지층은 다시 파헤치지 않는다. 고고학자도 감동할 영속성이다.

전체 데이터 흐름을 먼저 보여 준다.

```
[prompt tree / hero stills / phone scan]
        │
        ▼
 (A) Bedrock  (l ≤ l_g) : 전역(global) 동시 생성
     native STFT  또는  joint multi-view diffusion(체인 아님) → feed-forward recon → encode
        │  commit (Merkle hash)
        ▼
 (B) Demand : camera path + lens rig + minimax 카메라 → 필요한 (level, block) 집합
     ※ demand는 '무엇을 계산할지'만 정하고 '값'에는 절대 영향을 주지 않는다 (순수성)
        │
        ▼
 (C) Realize l = l_g+1 … L : 국소 순수 연산자 (native few-step)  ⊕ [옵션] S²R 이미지 교사 폴리시
     filial projection → commit
        │
        ▼
 (D) Render : band-gated traversal → typed atoms → raster(primary) + RT(secondary) → HDR AOVs
        │
        ▼
 (E) Physical camera : lens(DOF·bokeh·flare·breathing·CA) → shutter(MB·rolling) →
     sensor(Poisson·read noise·CFA) → log → look → 8K·120fps
```

---

## 1. 제1원리: "촬영할 수 있는 세계"란 정확히 무엇인가

창업자는 "가상의 카메라를 띄워 촬영할 수 있는 2D 이상의 세계면 된다"고 했다. 이걸 수학으로 박제한다.

**정의 (콘 질의 가능한 세계, cone-queryable world).** 세계 $W$는 다음 질의에 답하는 함수다.

$$\mathcal{Q}_W(\mathbf{o}, \mathbf{d}, \rho) \;\to\; L \in \mathbb{R}^{3}_{\ge 0}\quad(\text{scene-linear HDR radiance})$$

원점 $\mathbf{o}$, 방향 $\mathbf{d}$, 원뿔 반각(footprint) $\rho$인 콘(cone)을 따라 들어오는 복사휘도를 $\rho$ 스케일로 사전 필터링(prefilter)한 값이다. mip-NeRF와 Zip-NeRF의 콘을 렌더링 트릭에서 **세계의 정의**로 승격시킨 것이다.

**카메라는 이 질의들의 적분 연산자다.**

$$I[p] = \mathcal{S}\!\left( \int_{\text{shutter}}\!\int_{\text{aperture}}\!\int_{\text{pixel}}\!\int_\lambda s(\lambda)\,T_{\text{lens}}(\mathbf{u},\mathbf{x},\lambda)\; \mathcal{Q}_W\big(\mathbf{r}(\mathbf{x},\mathbf{u},t)\big)\; d\lambda\, d\mathbf{x}\, d\mathbf{u}\, dt \right)$$

$\mathcal{S}$는 센서다(광자 샷 노이즈, 리드 노이즈, 클리핑, 디모자이크, 로그 인코딩). 이 식이 말하는 바는 **$W$가 고정이면 영상의 모든 프레임 간 관계는 자동으로 일관된다**는 것이다. 일관성은 노력해서 얻는 게 아니라 정의에서 공짜로 나온다.

**따름정리: 광선 국소성 원칙(ray-locality principle).** (세계, 광선)에 대한 결정론적 함수는 무엇이든 시간적으로 일관된다. 프레임 단위의 숨은 변수(frame-level latent)만 없으면 뉴럴 셰이더를 써도 깜빡임은 원천적으로 불가능하다. 뒤에서 뉴럴 렌더링을 마음껏 쓰는 법적 근거가 이것이다. 반대로 화면 전체 문맥을 보고 프레임마다 다시 생성하는 순간 깜빡임이 바로 돌아온다. 비디오 디퓨전이 하는 일이 정확히 그것이다.

여기서 매개체가 만족해야 할 요구사항 8개가 나온다.

| # | 요구사항 | 이유 |
|---|---|---|
| R1 | 경로 독립성 (path independence) | 세계는 카메라보다 먼저 존재해야 한다. 같은 장소는 어느 경로로 가도 같아야 한다. |
| R2 | 스케일 완비성과 무모순 | 항공샷(큰 ρ)과 매크로(작은 ρ)는 같은 세계를 다르게 필터링한 결과여야 한다. 줌인했더니 다른 물건이 나오면 끝이다. |
| R3 | 뷰 의존 효과를 "유도" | 반사, 굴절, 스페큘러가 외운 값이 아니라 광수송에서 나와야 360° 돌아도 맞는다. |
| R4 | 조명 분리 | 재조명(relighting), 그리고 샷 조명 리그 |
| R5 | 생성 친화적 텐서 구조 | native world diffusion의 타깃이 되려면 정렬된 희소 격자(sparse lattice)여야 한다. |
| R6 | 편집성과 소유권(ownership) | 객체 단위 잠금, 교체, 이동 |
| R7 | 동역학 준비 | Phase 2를 위한 바디, 물리 파라미터, 시간 대역 |
| R8 | 렌더 비용 $O(\text{pixels}\cdot\log(\text{scale range}))$ | 8K·120fps를 장난감 취급하기 위해 |

---

## 2. 후보 매개체 전수조사: 애정 어린 디스(roast) 대회

먼저 다들 수고 많았다는 말을 해 둔다. 이제부터 디스한다.

- **3DGS (Kerbl 2023).** 이름부터 정규분포인데 뭘 기대하나. 무한 지지(infinite support) 커널이라 태생이 흐릿해서, 수백만 개를 겹쳐야 겨우 선명해진다. 래스터는 빠르다. 하지만 생성 타깃으로 보면 비정렬 집합(가변 개수, 순열 불변)이라 디퓨전이 싫어한다. 구면 조화(SH)로 외운 view-dependence는 저주파밖에 못 담는다. 멀티스케일은 Mip-Splatting, Hierarchical 3DGS, Octree-GS로 나중에 덧붙이는 구조라 팝핑이 생긴다. StopThePop이 정렬 팝핑은 줄였지만 근본적 해결은 아니다.
- **2DGS / 서펠.** 표면을 디스크로 표현해서 노멀이 생기고, 셰이딩과 재조명의 기반이 된다. 대신 머리카락이나 연기처럼 퍼지한 것은 약하다.
- **3DGRT / 3DGUT / EVER.** 3DGRT는 가우시안을 레이트레이싱하고, 3DGUT는 Unscented Transform으로 왜곡 렌즈와 롤링셔터를 래스터에서 정확히 처리하며, EVER는 정확한 볼륨 적분을 한다. 우리의 물리 카메라와 궁합이 최고다. 2차 광선(반사·굴절)도 가능하다.
- **컴팩트 커널 계열 (Deformable Beta Splatting, 3D Convex Splatting, Triangle Splatting, Textured Gaussians와 Gabor 커널 계열 시도들).** 가우시안의 흐리멍덩함에 대한 반란군이다. 경계가 날카롭고 프리미티브당 정보량이 많다. **뭉개짐을 퇴치하는 핵심 재료**다.
- **메시 + PBR.** 업계 표준이다. 재조명, 물리, 편집, 8K·120 모두 최강이다. 단점은 위상(topology)이 생성모델에게 악몽이라는 것, 그리고 털, 잎, 연기, 물보라에 약하다는 것이다. 모범생인데 머리카락을 못 그린다.
- **NeRF / 해시 그리드 (Instant-NGP, Zip-NeRF, SMERF, Binary Opacity Grids).** 품질이 좋고 콘 기반 안티앨리어싱의 원조다. 하지만 세계를 신경망 가중치라는 금고에 넣고 잠근 셈이라, 편집하려면 금고털이가 필요하다. 생성 타깃으로도 가중치는 최악이다. 굽기(baking)를 하면 빨라지지만 그 순간 다른 매개체가 된다.
- **희소 복셀 latent (TRELLIS의 SLAT, XCube, LT3SD, SCube/InfiniCube).** 생성 타깃으로서 현재 가장 큰 교훈을 준 계열이다. 정렬된 희소 격자에 디코더를 붙이면 가우시안, 메시, RF 무엇이든 나온다. 다만 해상도 상한이 있고 장면 스케일 계층은 아직 미완이다.
- **라이트필드 / plenoptic (Light Field Networks, LVSM).** view-dependence는 신의 영역이다(그냥 다 저장하거나 학습한다). 하지만 4D/5D라 저장량이 폭발하고 기하가 없으니 물리와 재조명이 불가능하다. 완벽한 사진첩인데 공을 던지면 "그게 뭔데?"라고 한다. LVSM은 3D 귀납편향 없이도 뷰 합성이 된다는 놀라운 증거지만, 여기서 일관성은 보증이 아니라 기대다.
- **MPI / LDI / 레이어드 메시 (Stereo Magnification, LLFF, Immersive Light Field Video).** 레이어가 곧 이미지라 이미지 prior와의 궁합은 최고다. 하지만 시야 범위가 좁은 종이 인형극 무대다. 카메라가 옆으로 30cm만 가도 무대 뒤 합판이 보인다.
- **파노라마 + 깊이 / 레이어드 월드 (HunyuanWorld 1.0, Matrix-3D).** 이미지 모델에서 바로 나와서 빠르다. 하지만 한 점에서 본 세계라 이동하면 가림 구멍(occlusion hole)이 생긴다. 그 구멍을 메우는 순간 우리가 금지한 길로 들어간다.
- **비디오 모델 내부의 암묵적 세계 (Genie 3 류).** 동역학이 네이티브이고 인터랙티브하다. 하지만 영속성의 상한은 분 단위 기억이고, 8K는 불가능하다. 우리가 대체하려는 패러다임의 최강자다.
- **생성형 3D 월드 계열 (World Labs Marble, Lyra, Bolt3D, Wonderland, Gen3C, CAT3D, SEVA).** 이미지·비디오 디퓨전에서 3DGS로 가는 길을 이미 열어 준 고마운 선배들이다. 다만 대부분 (a) 단일 스케일이고, (b) 생성 후 재구성(post-hoc)하거나 비디오 latent를 디코딩하는 방식이며, (c) 카메라는 핀홀 장난감이고, (d) 순수성이나 경로 독립성 보장이 없다.

### 2.1 평가표

◎ 탁월, ○ 양호, △ 조건부 또는 추가기술 필요, ✕ 근본적 한계

| 매개체 | 8K·120 렌더 | view-dep (반사/굴절/스페큘러) | 멀티스케일/무한 디테일 | 재조명 | 이미지 prior 궁합 | 편집성 | Phase 2 확장 | native 생성 타깃 |
|---|---|---|---|---|---|---|---|---|
| 3DGS | ○ (LOD 필요) | △ (SH 저주파) | △ (사후 LOD, 팝핑) | ✕~△ | ○ (splatter image, DiffSplat) | ○ | ○ (4DGS, PhysGaussian) | △ (비정렬 집합) |
| 2DGS/서펠 | ○ | △ | △ | △ (노멀 有) | ○ | ○ | ○ | △ |
| 3DGRT/3DGUT/EVER | △~○ | ◎ (2차 광선) | △ | ○ | ○ | ○ | ○ | △ |
| Beta/Convex/Triangle splat | ○ | △ | △ | △ | ○ | ○ | ○ | △ |
| 메시 + PBR | ◎ | ◎ (PT) | ○ (Nanite류) | ◎ | △ (UV/위상) | ◎ | ◎ (단, 퍼지 ✕) | △~✕ |
| NeRF/해시 그리드 | ✕ (굽기 전) | ○ | ○ (콘) | △ | △ | ✕ | △ | ✕ (가중치) |
| 희소 복셀 latent | ○ (디코더 의존) | ○ | ○ (옥트리) | △ | △~○ | ○ | ○ | ◎ |
| 라이트필드 | △ (저장 폭발) | ◎ | △ | ✕ | ○ | ✕ | ✕ | △ (LVSM 암묵) |
| MPI/LDI/레이어 | ◎ | △ | ✕ | ✕ | ◎ | △ | ✕ | ○ (범위 ✕) |
| 파노라마+깊이 | ◎ | △ | ✕ | ✕ | ◎ | △ | ✕ | ○ (범위 ✕) |
| 비디오 모델 암묵 세계 | ✕ | ○ (학습됨) | ✕ | ✕ | ◎ | △ | ◎ (단, 영속성 ✕) | ◎ |
| **STRATA (제안)** | ◎ (band-gated) | ◎ (재질+RT+예산 잔차) | ◎ (네이티브 대역) | ○~◎ | ◎ (S²R, 스펙트럼 라우팅) | ◎ (소유권, Merkle) | ◎ (바디 계층, 시간 대역) | ◎ (정렬 희소 계층 latent) |

### 2.2 디스 대회에서 건진 교훈 다섯 개

- **L1. 뭉개짐의 절반은 매개체 탓이다.** 무한 지지 커널에 끝에 가서 평균 내는 융합을 더하면 흐려진다. 그래서 컴팩트 커널, 대역 구조, 소유권 기반 융합(§7)으로 간다.
- **L2. 생성 타깃으로서 비정렬 점집합은 최악이고, 정렬된 희소 격자가 최고다.** TRELLIS와 XCube가 준 교훈이다.
- **L3. view-dependence를 외우면(SH, 라이트필드) 일관성이 깨지고, 유도하면(재질과 광수송) 일관성이 공짜다.**
- **L4. 멀티스케일은 옵션이 아니라 매개체의 척추여야 한다.** LOD를 나중에 붙이면 팝핑과 모순이 생긴다.
- **L5. 프리미티브 종류는 부차적이다.** 본질은 대역 구조, 커밋 의미론, 카메라 결합이다. 프리미티브는 갈아 끼울 수 있는 디코더 플러그인일 뿐이다. 가우시안 대 메시 논쟁은 반찬 투정이고, 메인 요리는 계층 구조다.

---

## 3. ★ 핵심 통찰: 스케일-시간 이중성 (Scale-Time Duality), "줌인은 디노이징이다"

### 3.1 두 개의 관찰

**관찰 1: 디퓨전은 스펙트럼 자기회귀다.** 자연 이미지의 파워 스펙트럼은 대략 $S(\nu) \approx A\nu^{-\alpha}$ ($\alpha\approx 2$)이다. Flow matching $x_t=(1-t)x_0+t\epsilon$에서 주파수 $\nu$의 SNR은 다음과 같다.

$$\mathrm{SNR}(\nu,t)=\frac{(1-t)^2 A \nu^{-\alpha}}{t^2},\qquad \nu^*(t)=A^{1/\alpha}\Big(\tfrac{1-t}{t}\Big)^{2/\alpha}$$

시점 $t$에서 $\nu^*$ 아래 대역은 결정되었고 위 대역은 미결정이다. 미결정 대역의 Tweedie 추정 $\hat x_0=\mathbb{E}[x_0\mid x_t]$는 조건부 평균이다. 위상이 랜덤한 디테일의 조건부 평균은 진폭이 0으로 가므로, 결국 **사전 필터링된(mip) 버전**이 된다. (Dieleman의 "diffusion is spectral autoregression"(2024), IHDM(Rissanen 2023), Blurring Diffusion(Hoogeboom 2022)이 이 관점의 선행이다.)

**관찰 2: 카메라는 자기 해상도만큼만 본다.** 깊이 $z$, 초점거리 $f_{px}$(픽셀 단위)인 카메라의 픽셀 footprint는 $\phi = z/f_{px}$ (m/px)이다. 이 카메라에게 필요한 것은 $\phi$로 사전 필터링된 세계뿐이다.

### 3.2 이중성 선언

세계 피라미드를 $W^{(l)}=\mathcal{F}_{h_l}W$, $h_l=h_0 2^{-l}$로 정의하면 세 가지가 성립한다.

- **디퓨전 시간 $t$, 레벨 $l$, 카메라 거리 $z$가 같은 축이다.**
$$l^*(z)=\Big\lceil \log_2 \frac{h_0\, f_{px}}{\kappa\, z}\Big\rceil$$
- **생성 분해 (scale-autoregressive):**
$$p(W\mid y)=p\big(W^{(0)}\mid y\big)\prod_{l=1}^{L}p\big(B_l\mid W^{(l-1)},y\big),\qquad W^{(l)}=W^{(l-1)}\oplus B_l$$
- **생성 중간 상태는 쓰레기 노이즈가 아니라 먼 카메라에게 완벽히 유효한 세계다.** 생성은 곧 점진적 LOD다. 카메라가 다가오면 세계가 디노이징된다.

수치 예시를 들자. 8K(7680px 폭), 풀프레임 36mm 센서, 50mm 렌즈라면 $f_{px}\approx 10{,}667$이다. 1m 거리에서 $\phi\approx 94\,\mu\text{m/px}$, 30cm 매크로에서 약 28µm이다. 2km 상공 24mm 항공샷에서는 약 0.39m/px이다. $h_0=4096\text{m}$로 잡으면 이 한 샷이 쓰는 레벨은 대략 12에서 26까지 **14개 레벨**이다. "Powers of Ten" 한 컷이 지층 14개를 관통하는 셈이다.

### 3.3 함정과 업그레이드: 프리필터는 외관 공간에서 정의한다

나뭇잎 기하를 블러하는 것과 숲을 렌더한 이미지를 블러하는 것은 다르다. 투과율이 비선형이기 때문이다. 그래서 $\mathcal{F}$는 렌더된 외관에서 정의한다.

$$\mathrm{Render}_C(\mathcal{F}_\sigma W)\approx k_{\text{pix}} * \mathrm{Render}_C(W)\qquad \forall C:\ \phi_C\ge\sigma$$

이것이 의미하는 바는, **코스 레벨은 흐릿한 표면이 아니라 통계적 매질(aggregate)이어야 한다**는 것이다. 멀리서 본 숲은 SGGX 마이크로플레이크 분포(Heitz 2015)를 가진 볼륨이고, 멀리서 본 머리카락도 볼륨이다. Hybrid mesh-volume LoD(Loubet & Neyret 2017)의 계보다. 그래서 STRATA에는 SGGX를 가진 VOL atom 타입이 있다(§4).

### 3.4 효도 제약 (Filial Consistency Constraint)

$$\mathcal{F}_{h_{l-1}}\big(W^{(l-1)}\oplus B_l\big)=W^{(l-1)}$$

자식 대역은 부모의 스케일에서 보면 없는 것처럼 보여야 한다. 자식은 부모 앞에서 튀면 안 된다는, 한국적 가치를 수식으로 구현한 최초의 3D 표현이다(아마도). 의미는 이렇다. **멀리서 본 것과 가까이서 본 것이 절대 모순되지 않는다.** 비디오 모델에서 줌인하면 돌멩이가 개구리로 변신하는 현상이 바로 이 제약을 어긴 것이다.

강제하는 방법은 두 가지다.
1. 디코더가 출력하는 대역 기여가 프리필터 아래에서 0이 되도록 학습한다(Laplacian 피라미드 성질).
2. 커밋 시점에 사영(projection)한다. latent 공간에서는 $B_l\leftarrow B_l-\mathcal{U}\mathcal{D}(B_l)$, 외관 공간에서는 $\min_{B}\|\mathcal{F}(\mathrm{Render}(W^{(l-1)}\oplus B))-\mathrm{Render}(W^{(l-1)})\|$을 몇 스텝 푼다.

부산물로 **팝핑이 구조적으로 해결된다.** 대역이 부모 스케일에서 평균 0이므로, 대역을 페이드 인/아웃해도 시각적으로 연속이다. 팝핑을 해킹으로 숨기는 게 아니라 수학적으로 존재할 수 없게 만든다.

### 3.5 이 이중성에서 나오는 것들

- (a) **즉시 프리뷰.** 1초 만에 항공샷이 나오고, 클로즈업은 그동안 굽는다.
- (b) **계산량은 카메라가 실제로 분해(resolve)하는 양에 비례한다.**
- (c) **렌즈 게이팅** (§5.2)
- (d) **교사 라우팅** (§7.3)
- (e) **무한 줌** (§6.5)

**선행과의 관계를 솔직히 밝힌다.** 이미지 쪽 next-scale prediction은 VAR(Tian 2024), 해상도를 가로지르는 flow는 Pyramidal Flow Matching(Jin 2024)과 Relay Diffusion, 일관된 줌은 Generative Powers of Ten(Wang 2024), 3D 계층 latent는 XCube와 LT3SD, 그리고 mip-mapping은 1983년부터 있었다. **새로운 점은 세 가지다.** (1) 스케일 축을 물리 단위(미터)로 정의하고 카메라 footprint에 묶어서, 생성 중간 상태가 실제 카메라용 렌더 가능한 LOD가 된다. (2) 효도 제약을 렌더된 외관 공간에서 정의한다. (3) 카메라 광학이 수요를 정하는 지연 평가(lazy evaluation)를 한다.

---

## 4. STRATA 자료구조: 텐서 수준 명세

### 4.1 전체 구조

```
World
├─ meta        : seed(u128), units=m, up=+Z, version, prompt_tree, merkle_root
├─ lattice     : L+1 levels,  h_l = h_0·2^-l   (h_0 = 4096 m → h_26 ≈ 61 µm)
│   └─ level l : sparse hash map   key=(l, i, j, k : int32) → Cell    (NanoVDB/fVDB 스타일)
├─ far_field   : r > R_far 구각(shell): multi-band octahedral HDR + depth(느린 패럴랙스)
│                + 물리 하늘 모델(Hosek–Wilkie / Prague) 파라미터
├─ objects     : inst_id → Object{class, prompt, xform, locks, phys_params(P2)}
├─ world_lights: sun/sky, practicals(IES), (emissive atoms는 셀 안에)
├─ shot_rigs   : shot_id → camera-relative film lights {key, fill, rim, bounce, neg-fill, flags}
└─ caches      : decoded atoms(LRU), per-cell GI probes(SH L2), neural radiance cache
```

### 4.2 Cell 레코드

| 필드 | 타입 | 크기 | 의미 |
|---|---|---|---|
| `z` | f16[d_l] (또는 int8) | d_l = 64(coarse)/32(mid)/16(fine) | **대역 latent**: 이 셀이 기여하는 $h_l \sim h_{l+1}$ 대역 |
| `occ` | u8 | 1B | 자식 8개 점유 마스크 (구조) |
| `split` | f16 | 2B | 더 미세한 대역이 존재할 확률 |
| `sem` | f16[8] | 16B | 의미 임베딩 (SigLIP/DINO 특징 PCA) |
| `inst` | u32 | 4B | **객체 소유권** (편집과 물리의 단위) |
| `body` | u32 | 4B | Phase 2 바디 id |
| `unc` | f16[2] | 4B | (aleatoric, epistemic) 불확실성 |
| `own` | u16 | 2B | **대역 저술권(provenance)**: 이 대역을 쓴 뷰나 생성기 id |
| `state` | u8 | 1B | `virtual` / `decoded` / `committed` / `locked` |
| `hash` | u64 | 8B | Merkle 콘텐츠 해시 (캐시 키, 버저닝) |

미세 레벨 기준 셀당 약 50~70B다.

**노이즈는 위치 주소, 캐시는 콘텐츠 주소로 둔다.**
- 노이즈: $\xi(\text{seed}, l, i, j, k, \text{step})$를 Philox 카운터 기반 PRNG(Salmon 2011)로 만든다. 편집과 무관하게 안정적이어서, 작은 편집은 작은 변화로 이어진다(생성기가 Lipschitz일 때).
- 캐시: Merkle 해시를 쓴다. 부모가 편집되면 하위 캐시가 자동으로 무효화된다.
- 결과적으로 **세계의 git**이 생긴다. `git blame`을 치면 이 조약돌을 어느 모델의 어느 뷰가 저술했는지 나온다. 편집은 copy-on-write 브랜치이고, 공유는 머클 루트 하나로 한다.

### 4.3 Typed Atoms (디코드된 캐시, 셀당 고정 슬롯 K_l개, opacity-gated)

공통 헤더: `μ`(셀 상대좌표, f16×3), `q`(f16×4), `s`(f16×3), `β`(커널 샤프니스: beta kernel, 클수록 날카로운 경계), `α`(u8), `type`(u8).

| 타입 | 용도 | 추가 필드 |
|---|---|---|
| **SURF** (2D 커널 서펠) | 대부분의 표면 | OpenPBR-lite 재질(base_color, roughness, metallic, specular, anisotropy+dir, sheen, clearcoat, subsurface radius/weight). 마이크로 디테일은 **Gabor carrier ≤2개**(접평면 파수 $\mathbf{k}\in\mathbb{R}^2$, 높이/알베도 진폭, 위상) 또는 8×8 texel patch(int8). **잔차 radiance**는 SG lobe 2개(예산 제한) |
| **VOL** (3D 커널) | 안개, 연기, 원거리 잎/털 aggregate | $\sigma_t$, albedo, 위상함수: **SGGX 행렬(6)** 또는 HG $g$ |
| **STRAND** (곡선) | 머리카락, 풀, 전선, 케이블 | 제어점 4개, 반경, BCSDF(Marschner/d'Eon: longitudinal shift, width, absorption) |
| **DIEL** (유전체) | 유리, 물, 얼음 | IOR, transmission color, thickness/closed 플래그, 분산(Abbe 수) |
| **EMIT** | 광원 표면 | HDR emission, IES 프로파일 id |

STRAND와 DIEL을 1급 시민으로 둔 이유는 **전선, 머리카락, 유리, 물보라가 비디오 디퓨전이 가장 먼저 녹이는 것들**이기 때문이다. 적의 약점에 전용 무기를 배치한다.

**뷰 의존 잔차 예산.** 잔차 SG lobe는 재질이 광택성일 때만 허용한다.

$$\mathcal{L}_{vd}=\lambda\sum_i \|\mathbf{a}^{SG}_i\|_1\cdot\big(1-g(\text{roughness}_i,\text{metallic}_i)\big)$$

이게 없으면 잔차 채널이 진짜 불일치를 흡수해서, 카메라를 움직일 때 표면이 반짝반짝 shimmer하는 저주가 걸린다.

### 4.4 ★ 조명: 세계 조명과 샷 조명의 분리

이미지 모델 미감의 상당 부분은 **이 각도에서만 성립하는 조명**에서 나온다(림 라이트, 얼굴의 키 라이트, 배경의 분리광). 순진하게 세계로 합치면 이것은 불일치로 처리되어 지워지거나 뭉개진다.

하지만 영화 현장도 정확히 그렇게 찍는다. **셋업마다 조명을 다시 친다.** 그래서 STRATA는 이렇게 나눈다.
- `world_lights`: 해, 하늘, 가로등, 발광체. 영속적이다.
- `shot_rigs`: 카메라 상대 좌표의 필름 조명(키, 필, 림, 바운스, 네거티브 필, 플래그). 샷 안에서 애니메이션할 수 있고 물리 렌더된다.

결과는 이렇다. 샷 안에서는 셰이딩이 물리적으로 완벽히 일관되고, 미감은 촬영감독(DP)급이며, 세계 자체는 오염되지 않는다. 이미지 모델의 치팅을 합법적인 영화 문법으로 귀화시키는 것이다.

재조명에서 재질 분해가 불확실한 영역(`unc` 높음)은 비율 전송(ratio transfer)으로 처리한다. 구운 디테일은 보존하고 셰이딩만 바꾼다.

$$L'=L_{\text{baked}}\cdot\frac{\mathrm{Render}(M,\ \mathcal{L}_{\text{new}})}{\mathrm{Render}(M,\ \mathcal{L}_{\text{old}})}$$

### 4.5 메모리 예산 (Powers of Ten 샷: 2km 항공에서 1cm 무당벌레 매크로, 8K)

| 항목 | 규모 |
|---|---|
| 사용 레벨 | 12–26 (14개) |
| 프레임당 활성 셀 | 약 2–8M |
| 10초 경로 고유 셀 | 약 40–160M |
| latent 저장 (int8 기준) | 약 1–4 GB |
| 프레임당 디코드 atom | 약 10–30M (약 1 atom/pixel) |

**latent가 정식 포맷이고, atom은 캐시다.** NVIDIA Neural Texture Compression과 같은 철학이다. 압축률은 디코드된 atom 대비 30–60배다.

---

## 5. 렌더러와 가상 물리 카메라: 렌즈가 세계에게 말을 건다

### 5.1 Band-gated traversal

```
GATHER(camera rig R, shutter [t0, t0+Δt]):
  Q ← root cells ∩ frustum  (shutter 동안의 카메라 궤적 + 최대 CoC 만큼 팽창)
  while Q not empty:
    c ← pop(Q)
    φ_eff ← world footprint at c   # pixel ⊕ CoC ⊕ motion ⊕ rolling-shutter (§5.2)
    if h_{l(c)} ≤ κ·φ_eff  or  leaf(c):
        emit atoms(c) with α_lod = smoothstep(log2(κ·φ_eff / h_{l(c)}))   # 효도 제약 → 팝핑 없음
        continue
    if c.state == virtual: DEMAND(c)        # 순수 생성기로 realize (§6), 값은 경로와 무관
    push children(c)
```

### 5.2 카메라는 공분산 대수(covariance algebra)다

atom $i$의 화면 공분산은 다음과 같다.

$$\Sigma'_i = J_i\Sigma_iJ_i^\top + \underbrace{\Sigma_{\text{pix}}}_{\text{픽셀 적분}} + \underbrace{\Sigma_{\text{dof}}(z_i)}_{\text{보케}} + \underbrace{\Sigma_{\text{mb},i}}_{\text{모션블러}}$$

- **투영**: 3DGUT처럼 Unscented Transform을 쓰고, sigma point마다 노출 시각이 다르다. 왜곡(Brown–Conrady, fisheye)과 롤링셔터 $t(y)=t_0+y\,\tau_{\text{line}}$를 정확히 처리한다.
- **착란원(CoC)**: $c(z)=\dfrac{f^2}{N(z_f-f)}\dfrac{|z-z_f|}{z}$ (센서 mm 단위, 픽셀로 변환). 균일 디스크를 가우시안으로 근사하면 $\Sigma_{\text{dof}}=(c/4)^2 I$.
- **모션블러**: $\Sigma_{\text{mb}}=\dfrac{\Delta t^2}{12}\dot{\mathbf m}\dot{\mathbf m}^\top$, $\Delta t=\dfrac{\theta_{\text{shutter}}}{360^\circ}\cdot\dfrac{1}{\text{fps}}$. 비선형 궤적이면 서브프레임으로 샘플링한다.
- **에너지 보존**: $\alpha'_i=\alpha_i\sqrt{\det\Sigma^{0}_i/\det\Sigma'_i}$, 여기서 $\Sigma^0=J\Sigma J^\top+\Sigma_{\text{pix}}$.
- **대역 게이팅용 유효 footprint**: $\phi_{\text{eff}}=\dfrac{z}{f_{px}}\sqrt{\lambda_{\min}(\Sigma_{\text{pix}}+\Sigma_{\text{dof}}+\Sigma_{\text{mb}})}$

**★ Lens-gated bands: 렌즈가 세계에게 얼마나 디테일해질지 알려준다.**

수치로 보자. 8K, 50mm f/1.4, 초점 2m, 배경 20m라면 센서상 CoC는 약 0.82mm이고 픽셀로는 약 176px이다. 배경 footprint가 약 44배 커지므로 **약 5레벨 덜 생성해도 된다.** 표면 셀 수는 레벨당 4배씩 늘어나니, **배경 디테일 생성 비용이 약 1/1000로 줄어든다.**

물리적 정확성도 따라온다. 비디오 모델은 보케 속에 텍스처를 환각하거나 보케 모양을 틀리게 그린다. 우리는 기하에서 보케를 계산한다. 모션블러 속도 마찬가지다. 빠르게 패닝하는 프레임에서는 세계가 알아서 덜 상세해진다.

### 5.3 보케, 플레어, 글레어: 렌즈 처방전(prescription) 기반

- **보케 하이라이트**: 휘도가 $\tau\cdot$노출 이상이고 CoC가 지배적인 atom은 *조리개 스프라이트*로 렌더한다. 필드 위치마다 출사동(exit pupil) 형상을 쓰는데, 경통 비네팅에 의한 cat-eye, 블레이드 수와 라운딩, 비구면 연마 때문에 생기는 onion-ring, 구면색수차 프린지(채널별 CoC 스케일)를 포함한다. 이 형상은 렌즈 처방전(구면/비구면 면 시퀀스)을 레이트레이싱해서 LUT로 미리 계산해 둔다.
- **고스트 플레어**: Hullin et al. 2011 방식. **베일링 글레어**: HDR 프레임과 PSF의 FFT 컨볼루션. **회절 스타**: 조리개 다각형의 푸리에 변환에서 나온다. 블레이드가 짝수 N개면 스파이크 N개, 홀수 N개면 2N개다. 비디오 모델은 이 규칙을 모른다. 우리는 안다.
- **아나모픽**: 2배 스퀴즈로 타원 보케와 수평 스트릭을 만든다.
- **렌즈 브리딩**: 얇은 렌즈에서 $s'=\dfrac{f z_f}{z_f-f}$이므로 포커스를 당기면 배율이 바뀐다. 렌즈별 LUT로 처리한다.
- **색수차**: 횡(채널별 방사 스케일)과 종(채널별 $z_f$). **비네팅**: $\cos^4$에 기계적 비네팅을 더한다.

### 5.4 센서와 필름

- 광자 수 $n\sim\text{Poisson}(Q\,E\,t_{\exp})$, 리드 노이즈 $\mathcal{N}(0,\sigma_r^2)$, full-well 클리핑, ADC 양자화, Bayer CFA와 디모자이크(진짜 모아레까지 나온다). 그다음 카메라 로그 커브, look LUT, 디스플레이 변환 순서다.
- **필름 에뮬레이션**: 할레이션(하이라이트에서 붉은 주황 번짐), 그레인은 Newson et al. 2017의 Boolean 모델(해상도 독립, 프레임마다 시드), 게이트 위브(서브픽셀 흔들림).
- 센서 노이즈는 **프레임마다 독립**이다. 실제 카메라가 그렇다. 이건 의도된 시간적 노이즈라 리얼리즘을 더해 준다. 세계의 텍스처는 절대 헤엄치지 않고 센서 노이즈만 춤춘다. 진짜 영상이 정확히 이렇게 생겼다.
- **Neural Sensor (옵션)**: 특정 룩의 RAW-to-grade 매핑을 학습한 픽셀 국소 소형 네트워크. 광선 국소성 원칙에 따라 깜빡임이 원천 불가능하다.

### 5.5 광수송 모드

- **Mode A (프리뷰/베이크)**: primary 래스터(3DGUT 스타일, StopThePop식 계층 정렬)에 재질 셰이딩, probe GI, SG 잔차를 더한다.
- **Mode B (시네마)**: 하이브리드다. primary는 래스터, secondary는 레이트레이싱이다. 볼륨은 EVER/3DGRT, 서펠은 ray-disk 교차, 스트랜드는 ray-curve 교차, DIEL은 분산 포함 굴절. GI는 path tracing과 Neural Radiance Cache(Müller 2021)로, 커스틱은 DIEL atom에 photon mapping을 옵션으로 쓴다.
- **거울은 세계에 실제로 있는 것을 비춘다. 카메라 뒤에 있는 것까지.** 비디오 모델은 원리적으로 이걸 보장할 수 없다. 거울 복도 샷 하나로 상대 모델의 멘탈을 깰 수 있다.

### 5.6 8K·120fps 예산과 "시간 해상도는 거의 공짜"

8K는 7680×4320 = 33.2MP다. 10초 120fps면 1200프레임, 약 $4.0\times10^{10}$ 픽셀이다. primary 래스터가 GPU당 약 1 Gpx/s라면 GPU 하나로 약 40초다. Mode B는 5–20배 더 걸려서 8 GPU로 수 분이다.

핵심은 다음 비대칭이다.

| | 비디오 디퓨전 | STRATA |
|---|---|---|
| 생성 비용이 비례하는 것 | 프레임 수 × 해상도 | **고유한 세계 콘텐츠량** |
| 24→120fps | 생성 비용 약 5배 (사실상 불가) | 생성 ×1, 렌더만 ×5 |
| 1080p→8K | 생성 비용 약 16배 이상 (불가, 업스케일러가 환각) | lens-gated로 필요한 대역만 추가 |

**슬로모션과 하이프레임레이트는 우리의 무기다.** 경쟁자에게는 가장 비싼 것이 우리에게는 가장 싼 것이다.

---

## 6. ★ 순차 hole-filling 없이 무한 디테일 얻기: 세계는 시드의 순수 함수다

### 6.1 마인크래프트에게 배운다

마인크래프트 세계는 `f(seed, x, z)`다. 청크가 어떤 순서로 로드되든 모든 플레이어가 같은 산을 본다. 나무나 구조물을 심는 population 단계는 이웃 청크를 필요로 하는데, 그래서 **유한한 이웃 의존을 가진 다단계 생성**을 한다. 우리는 Perlin 노이즈 자리에 디퓨전을 넣는다. 마인크래프트 월드 시드를 이미지 모델 수준의 미감으로 만드는 셈이다.

### 6.2 정의

$$B_l(c)=R^{(l)}_\theta\Big(W^{(l-1)}\big|_{\mathcal N_{r_l}(c)},\ \xi\big(\text{seed},l,\mathcal N_{r_l}(c)\big),\ y\Big)$$

- 코스 레벨($l\le l_g$, 셀이 대략 1–8m)은 유한한 기반암 영역에서 **전역 동시 생성**한다. 가림 영역, 카메라 뒤편, 건물 내부까지 전부 포함한다.
- 미세 레벨($l>l_g$)은 수용장(receptive field)이 유한한 **국소 순수 연산자**다(few-step으로 증류, §9.3).

**정리 (경로 독립성).** 임의의 두 카메라 경로 $P_1,P_2$에서 둘 다 평가한 셀 $c$에 대해 $B_l(c)$는 동일하다.

*증명.* 레벨에 대한 귀납법이다. $B_l(c)$는 seed, 조건 $y$, 그리고 유한 이웃의 $W^{(l-1)}$만의 함수이고, 귀납 가정에 의해 그것들은 경로와 무관하다. ∎

결정론적 커널이 전제다. 현실에서는 커밋 메모이제이션이 최초 평가를 정본(canonical)으로 고정하므로 수치 오차까지 포함해 영속성은 정확히 지켜진다.

양자역학 농담을 하나 하자면, 관측 전의 디테일은 확률 분포다(**슈뢰딩거의 디테일**). 하지만 우리 세계는 **숨은 변수 이론**이고, 시드가 그 숨은 변수다. 아인슈타인이 옳았다. 우리 세계 한정이다.

### 6.3 근본적 차이: 생성의 순서축을 시간에서 스케일로

| | 순차 아웃/인페인팅 캔버스 | 비디오 디퓨전 | **STRATA** |
|---|---|---|---|
| 조건부 의존 그래프(DAG) | 방문 순서 체인 | 시간 순서 체인 | **스케일 트리 + 국소 격자** |
| 세계가 경로에 의존하는가 | 예 | 예 | **아니오 (정리)** |
| 오차 누적 깊이 | O(경로 길이) | O(프레임 수) | **O(log 스케일 범위) ≈ 15–25, 고정** |
| completion의 종류 | 공간적 (모르는 곳을 발명) | 공간+시간적 | **스펙트럼적 (아는 저주파에 고주파 추가)** |
| 구멍 | 있음, 메워야 함 | 매 프레임 | **없음 (모든 곳이 항상 어떤 대역으로 존재)** |
| 가림 영역과 뒤편 | 볼 때 발명 | 볼 때 발명 | **기반암에 이미 존재** |
| 드리프트 | 누적 (들어갈 때 부엌, 나올 때 화장실) | 누적 | **기반암이 고정이라 불가능** |

> **구멍을 메우지 않는다. 초점을 맞춘다.**

창업자의 지적("캔버스를 돌려 빈 곳을 채우는 건 결국 비디오 디퓨전 원리이고 퇴화다")은 정확하다. 둘 다 시간 순서 체인이기 때문이다. 우리 체계에도 completion이 있지만 종류가 다르다. 전역 저주파에 조건화된 스펙트럼 completion이다. 대역 $l$의 상관 길이는 대략 $h_l$이므로 조건부 엔트로피가 국소적이고 유계다. 체인도 없고 드리프트도 없다. 같은 "조건부 생성"이라는 단어를 쓰지만, 조건이 걸리는 그래프가 근본적으로 다르다.

### 6.4 의존 원뿔(dependency cone), Jacobi 패스, 분산 보존 블렌딩

- **블록 분할**: 레벨 $l$은 고정 격자 블록(한 변 64–128셀)과 apron으로 나눈다.
- **Jacobi 패스**: 같은 레벨 내 결합이 필요하면 동기식으로 패스를 돈다.
$$B^{(p)}_l(c)=R\big(W^{(l-1)},\ B^{(p-1)}_l\big|_{\mathcal N(c)},\ \xi\big)$$
  패스마다 의존 반경이 $r$씩 자란다. $P$번 돌면 원뿔 반경은 $P\cdot r$로 유한하고, 함수는 여전히 순수하다. MultiDiffusion의 동기식 버전과 같은 원리인데, 순서가 없다.
- **겹침 블렌딩**: 분산 보존 블렌딩을 쓴다(Heitz & Neyret 2018, histogram-preserving blending).
$$z=\frac{\sum_b w_b z_b}{\sqrt{\sum_b w_b^2}}$$
  디테일 텍스처를 섞을 때 대비가 죽는 문제가 없다. 게다가 부모와 해시 노이즈를 공유하므로 경계 근처에서 $z_a\approx z_b$이고, 블렌딩은 원래 아주 약하게만 일한다.
- **비용**: 의존 원뿔은 레벨에 대한 기하급수로 유계다. 수요(demand)는 **계산할 집합만** 정한다.

### 6.5 의미 줌 (semantic zoom)과 무한 디테일

`prompt_tree`를 스케일 범위로 키잉한다. 예를 들면 `"교토 정원" @ l≤14`, `"이끼 낀 화강암" @ 15–20`, `"이끼 포자낭과 물방울" @ 21+`. 레벨별 CFG도 스케줄링한다. 코스에서는 프롬프트 충실도를 높게, 미세 레벨에서는 자연스러운 텍스처를 위해 낮게 둔다. Generative Powers of Ten의 3D 판이면서 지연 평가가 되고 순수한 버전이다. 원리적으로 잎에서 엽맥, 세포까지 내려가는 것도 가능하다. Phase 1 데모의 오프닝 샷 후보다.

---

## 7. 부트스트랩: 이미지 디퓨전에게 세계를 그리게 하는 법, S²R (Stratified Synchronized Refinement)

창업자의 원래 통찰은 "이미지 디퓨전에게 세계를 여러 시점에서 그리게 하라"였다. S²R은 그 통찰을 **흐려지지 않고 일관되게** 만드는 방법이다. native 모델의 학습 데이터 엔진이자, 히어로 샷을 위한 프리미엄 폴리시 경로다.

### 7.1 교사(teacher) 업그레이드: 미감 손실 없이 LoRA 수준으로만

- **T1. 픽셀별 timestep 맵 $T[p]$ 입력.** Differential Diffusion과 Diffusion Forcing의 계보다.
- **T2. G-buffer 헤드.** depth, normal, albedo, roughness, metallic, semantic을 함께 출력한다(RGB↔X, Marigold, GeoWizard 계보). 이렇게 하면 각 뷰가 RGB만이 아니라 **intrinsic 샘플 전체**를 제안하게 된다.
- **T3. 노출 조건화.** HDR 브래킷용이다(Bracket Diffusion 계보).

베이스는 동결하므로 이미지 모델의 미감 prior가 보존된다. 동기화는 모델 바깥, 세계 공간에서 한다. multi-view 파인튜닝으로 미감을 깎아 먹는 대신 원본 모델을 그대로 쓰면서 일관성은 우리가 강제한다.

### 7.2 기반암(bedrock)

낮은 해상도에서 정준(canonical) 카메라 격자 전체를 **한 번의 joint pass**로 생성한다. sparse cross-view attention을 쓰는 CAT3D/SEVA 스타일이다. 앵커에서 그룹으로 가는 2단 계층까지는 허용하지만 체인은 금지한다. 원거리용 파노라마도 함께 만든다. 그다음 feed-forward 기하(VGGT, Depth Anything 3, AnySplat, MapAnything 계열)로 STRATA의 $l\le l_g$ 레벨을 인코딩한다. 나중에는 native 모델로 대체된다.

### 7.3 ★ 스펙트럼 라우팅과 footprint 정합 timestep

뷰 $v$의 픽셀 $p$에서 footprint는 $\phi_v(p)=\dfrac{z_v(p)}{f_{px}\,|\cos\theta_v(p)|}$이고, 대역 $l$의 화면 주파수는 $\nu_l(p)\propto\phi_v(p)/h_l$이다. §3.1의 공식을 뒤집으면 **대역 $l$이 결정되는 순간의 timestep**이 나온다.

$$T_v[p]=\Big(1+\sqrt{\nu_l(p)^{\alpha}/A}\Big)^{-1}$$

latent 공간 교사라면 $A$와 $\alpha$는 교사의 실측 캘리브레이션 곡선으로 대체한다. $t$별로 $\hat x_0$의 대역 에너지가 안정되는 지점을 재면 된다.

- **시작 상태**: $W^{(l-1)}$를 렌더하고, 픽셀마다 $T_{\text{start}}[p]$(대역 $l$ 미결정)로 노이즈를 입힌다.
- **종료 지점**: $T_{\text{end}}[p]$(대역 $l$은 결정, 대역 $l+1$은 미결정)에서 멈추고 Tweedie $\hat x_0$를 취한다. 이 추정은 대역 $l$을 포함하면서 더 미세한 대역은 사전 필터링된 상태다. **커밋해야 할 바로 그 상태**다.
- **레벨당 교사 스텝은 2–4번이다.** 한 옥타브는 $(1-t)/t$를 $2^{\alpha/2}\approx2$배 바꾸는 구간이기 때문이다. 14레벨 전체를 돌아도 이미지 한 장 뽑는 스텝 수와 비슷하다.
- **그래디언트 라우팅**: $w_{l'}(p,t)=\exp\!\big(-\tfrac{(l'-l^*(p,t))^2}{2s^2}\big)$. 커밋된 레벨은 가중치 0이다. 교사의 미세 디테일 업데이트가 코스 기하에 새어 들어가 기하가 지저분해지는 SDS류의 고질병을 막는다.

> 각 픽셀은 자기 깊이에 맞는 노이즈 레벨에서, 정확히 자기 대역만 결정한다.

### 7.4 세계에 박힌 노이즈 (world-anchored noise)

$$\varepsilon_v[p]=\frac{1}{\sqrt{|A_{v,p}|}}\sum_{s\in A_{v,p}}\sqrt{a_s}\;\xi_s,\qquad \xi_s=\text{Philox}(\text{seed},\,l_{\text{fine}},\,s)$$

$A_{v,p}$는 픽셀 $p$가 보는 표면 위 서브셀 집합이고 $a_s$는 면적이다. 이렇게 하면 세 가지가 성립한다.
- 뷰 안에서는 i.i.d.라서 교사의 가정을 만족한다.
- 뷰 사이에서는 footprint가 겹치는 만큼 상관된다.
- 스케일이 달라도 같은 기저 노이즈를 공유한다.

효과: **뷰들이 처음부터 같은 디테일을 환각하는 방향으로 시드된다.** 선행은 How I Warped Your Noise(Chang 2024)의 integral noise와 Consistent Flow Distillation(ICLR 2025)의 3D 일관 노이즈다. 우리가 확장한 부분은 멀티스케일 footprint 적분과 순수 격자 키잉이다.

### 7.5 S²R 루프

```
S2R_REFINE_LEVEL(l, demanded_blocks):
  parallel for block b in demanded_blocks:                       # 순서 무관, 순수
    V ← CANONICAL_CAMERAS(seed, l, b)        # band-matched 거리 d_l = f_px·h_l/κ_px,
                                             # 표면 노멀 분포 기반 방향, 해시 지터, 6–24개
    V ← V ∪ (HERO_CAMS ∩ b) ∪ MINIMAX_CAMS(l, b)                 # §7.7
    for v in V:
      x0_v ← RENDER_v(W^{(l-1)});  T_v ← FOOTPRINT_TIMESTEPS(v, l)
      ε_v  ← WORLD_ANCHORED_NOISE(seed, v)
      x_v  ← (1 − T_v)⊙x0_v + T_v⊙ε_v
    for k in 1..K:                                               # K = 2–4
      for v in V: (x̂0_v, ĝ_v) ← TEACHER(x_v, T_v, prompt(v), EV)  # Tweedie + G-buffer
      B ← LIFT(W^{(l-1)}, {x̂0_v, ĝ_v}, ω(τ_k), routing w)         # Band-LRM 1-pass 또는 GD 수 스텝
      B ← ARBITRATE(B)                                           # 패럴랙스 / 스페큘러 / 환각충돌
      B ← FILIAL_PROJECT(B, W^{(l-1)})
      for v in V:
        x̃0_v ← RENDER_v(W^{(l-1)} ⊕ B)                           # 동기화된 Tweedie
        T_v  ← SCHEDULE(T_v, k)
        x_v  ← (1 − T_v)⊙x̃0_v + T_v⊙ε̃_v(seed, k)                  # 세계 고정 노이즈로 재노이즈
    COMMIT(b, l, B);  unc_epi ← cross-view residual variance
  VP_BLEND(overlaps)
```

**소유권 가중 (ownership annealing):**

$$\omega_{v,p}=\operatorname{softmax}_v\!\Big(-\tau_k\big[\log\phi_v(p)-\log|\cos\theta_v(p)|\big]\Big),\qquad \tau_k\uparrow$$

초반에는 부드러운 합의, 후반에는 승자독식이다. 셀마다 **가장 가깝고 가장 정면인 뷰**가 대역의 저술권을 가진다(`own` 필드). 다른 뷰들은 저주파 합의에만 기여한다. best-view 텍스처 매핑(Lempitsky & Ivanov 2007, Waechter 2014)을 디퓨전 동기화로 옮긴 것이다.

**불일치 중재 (ARBITRATE):** 셀 $c$의 뷰 간 잔차 $r_{c,v}$를 다음 순서로 처리한다.
1. **패럴랙스로 설명되는가?** 기하 대역을 gradient step으로 고친다.
2. **BRDF 로브로 설명되는가?** 조명 리그 아래에서 $(F_0,\text{roughness})$를 피팅하고, 잔차가 $\gamma$ 이상 줄면 재질로 승격한다.
3. **나머지는 환각 충돌이다.** 소유권으로 하드 선택하고, 평균은 절대 내지 않는다. 그리고 `unc_epi`에 기록한다.

### 7.6 왜 이건 안 흐려지는가

1. **세계 고정 노이즈** 덕분에 뷰들이 애초에 같은 디테일을 제안한다.
2. **매 스텝 동기화** 덕분에 모드를 평균 내는 대신 모드에 합의한다. 끝에 가서 평균 내면 블러가 되고, 매 스텝 합의하면 선명해진다.
3. **소유권** 덕분에 충돌 디테일은 평균이 아니라 선택된다.
4. **Tweedie와 프리필터의 이중성** 덕분에 레벨마다 정확히 한 옥타브만 결정하므로, 미결정 대역을 억지로 합의시키지 않는다.

선행: SyncTweedies(2024), DMV3D(2024), 3D-Adapter(2024)가 3D-in-the-loop 동기화의 계보다. 우리는 거기에 스케일 층화, 세계 고정 노이즈, 소유권 annealing, 물리 중재, 순수 블록을 더했다.

### 7.7 ★ 미니맥스 촬영감독 (adversarial cinematography): 악당 DP를 고용한다

$$C^*=\arg\max_{C\in\Pi_{\text{cine}}}\ \mathcal{E}\big(\mathrm{Render}_C(W^{(l)})\big),\qquad \mathcal{E}(x)=\mathbb{E}_{t\sim U[t_a,t_b],\,\epsilon}\big\|\epsilon_\phi(x_t,t)-\epsilon\big\|^2$$

$\mathcal{E}$는 디퓨전 손실을 implausibility 에너지로 쓴 것(ELBO의 대리)이거나 판별기다. 포즈, 초점거리 14–200mm, 조리개 공간에서 CMA-ES와 미분 가능 렌더러 그래디언트로 탐색한다. **세계를 가장 못생기게 찍는 카메라 top-k**를 찾아서, 열려 있는 레벨 윈도우 안에서 다음 정제 라운드의 정준 세트에 추가한다.

제품 약속은 "**어떤 경로로 찍어도**"다. 평균이 아니라 최악의 경우(sup)를 최적화한다. 선행으로 FisherRF(능동 뷰 선택), ViewFool(분류기용 적대적 시점)이 있지만, 생성 세계의 품질 미니맥스로 쓴 예는 없는 것으로 안다.

### 7.8 HDR은 코스 대역 현상이다

해, 하늘, 램프는 저주파다. 그래서 노출 브래킷 $\{-4,-2,0,+2,+4\}$ EV 생성은 $l\le l_g+2$에서만 한다. 비용은 작고 효과는 크다. 플레어와 블룸이 진짜가 되는 지점이다. 표시용 SDR이 아닌 **scene-linear HDR**을 저장하지 않으면 §5.3의 렌즈 효과 전부가 가짜가 된다.

### 7.9 비용

부트스트랩은 비싸다. 샷당 GPU-시간 단위가 든다. 그래서 이건 **데이터 엔진이자 히어로 폴리시**이고, 제품 추론은 native 모델(§9)이 맡는다.

---

## 8. World VAE / World Latent

### 8.1 입력
- (a) atoms: 재구성 결과, S²R 출력, 합성 GT
- (b) 포즈 있는 이미지: LRM 스타일 인코더, 실제 캡처용

### 8.2 구조
- **옥트리 U-Net + sparse transformer.** bottom-up 인코더로 레벨별 특징을 만들고, top-down으로 계층 사후분포 $q(z_l\mid x,z_{<l})$를 만든다(NVAE와 XCube 계보).
- **직렬화**: Hilbert/Morton 순서와 윈도우 어텐션(Point Transformer V3 스타일).
- **미터 인식 3D RoPE**: 위치를 미터로 넣고 $h_l$로 스케일한다. 모델이 지금 다루는 게 1mm인지 1km인지 안다.
- **디코더 $D_l$**: $(z_l,\text{parent ctx})$에서 $K_l$개의 typed atom 슬롯(타입 로짓, opacity gating), 자식 점유 로짓, split 로짓을 낸다. atom 오프셋은 tanh로 셀 근방에 묶어 국소성을 보장한다.

### 8.3 손실

$$\mathcal{L}=\sum_l\mathbb{E}_{C\sim\Pi_l}\Big[\ell_1+\lambda_{s}(1-\text{SSIM})+\lambda_{p}\text{LPIPS}+\lambda_{d}\big\|\Phi_{\text{DINO}}(I)-\Phi_{\text{DINO}}(\hat I)\big\|\Big]+\lambda_g\mathcal{L}_{\text{geo}}+\lambda_m\mathcal{L}_{\text{mat}}+\lambda_f\mathcal{L}_{\text{filial}}+\lambda_a\mathcal{L}_{\text{adv}}+\beta\,\mathrm{KL}+\lambda_r\mathcal{L}_{\text{reg}}$$

- $\Pi_l$: 레벨 $l$이 최미세 필요 대역인 band-matched 카메라 분포. 타깃은 **픽셀 프리필터를 적용한 GT 렌더**라서 LOD 일관성이 손실 안에 내장된다.
- $\mathcal{L}_{\text{adv}}$: 미세 레벨 렌더에 패치 GAN을 건다. SD-VAE가 선명함을 지킨 방식이다.
- $\mathcal{L}_{\text{reg}}$: 잔차 radiance 예산(§4.3), opacity sparsity, 스케일 범위.
- **Alias-free 증강**: 서브셀 평행이동과 회전, 그리고 $\|E(T_\delta x)-\mathrm{Interp}_\delta E(x)\|$ 등변성 손실을 쓴다(StyleGAN3의 교훈). 격자 아티팩트가 생성 모델로 번지는 것을 막는다.

### 8.4 latent 통계
$d_l$은 64/32/16이고, 레벨별 채널 정규화를 한다. 압축률은 atom 대비 30–60배다.

---

## 9. Native World Diffusion: Scale-Time Flow Transformer (STFT)

이름이 Short-Time Fourier Transform과 겹친다. 그런데 둘 다 국소 주파수 분해라 오히려 맞는 이름이다. 운명이다.

### 9.1 분해

$$p(W\mid y)=p(S_0,z_0\mid y)\prod_{l\ge1}p(S_l\mid z_{<l},y)\;p(z_l\mid S_l,z_{<l},y)$$

$S_l$(구조, 즉 활성 셀)은 부모의 `occ`/`split` 로짓에서 이산적으로 샘플링한다. **구조가 디테일과 함께 자란다.** TRELLIS의 2단(구조 다음 latent)을 레벨 축 전체로 일반화한 것이다.

### 9.2 아키텍처
- 레벨 임베딩을 가진 단일 네트워크를 전 레벨이 공유한다. 토큰은 레벨 $l$의 활성 셀이다.
- 컨텍스트: 부모 토큰(깨끗한 상태)을 cross-attention으로, 레벨별 전역 register 토큰 약 64개, LLM 텍스트 인코더, 코스 셀에 앵커된 **영역 프롬프트(semantic paint)**, Plücker 임베딩을 붙인 히어로 이미지 토큰.
- 어텐션: $l\le l_g$에서는 full attention(약 256k 토큰 이하). 미세 레벨에서는 **유계 반경의 윈도우 어텐션**이고, 이것이 순수성의 조건이다.
- 목적함수: $\mathcal{L}=\mathbb{E}\big\|v_\theta(z_l^t,t,l,\text{ctx})-(\epsilon-z_l)\big\|^2$

### 9.3 순수성 증류 (purity distillation)
교사는 넓은 윈도우의 멀티스텝 모델이다. 학생은 $K\le4$ 스텝, 엄격한 반경 $r_l$, Philox(seed, l, cell, step) 노이즈를 쓰고 consistency, shortcut, adversarial distillation으로 학습한다. **수용장이 유계라서 의존 원뿔이 정확히 알려지고, 지연 평가가 정확해진다.**

### 9.4 학습 데이터 믹스

| 소스 | 주 담당 레벨 | 역할 |
|---|---|---|
| 절차적 합성 (Infinigen 계열 자연/실내, sub-mm GT, 재질 GT) | 미세 | 효도 구조, 재질 진실, 무한 디테일 |
| 실제 재구성 (DL3DV-10K, ScanNet++, MegaScenes, RealEstate10K 계열, 항공 사진측량) | 중간~코스 | 리얼리즘 |
| S²R 데이터 엔진 | 전 레벨 | 이미지 모델급 미감 분포 |
| 객체 (Objaverse-XL 등) 합성 배치 | 중간 | 다양성 |

### 9.5 Render-RL 후학습

$$R(W)=\mathbb{E}_{P\sim\Pi_{\text{cine}}}\big[A(\text{video}_P)\big]-\beta\max_{C\in\mathcal{M}}\mathcal{E}(C)+\gamma\,\text{MTF}(W)$$

**일관성 항은 필요 없다. 구조적으로 이미 보장된다.** 코스 레벨(레이아웃, 조명, 구도)에는 Flow-GRPO/DanceGRPO 스타일을, 미세 레벨에는 보상 가중 또는 적대적 증류를 쓴다. 보상을 이미지가 아니라 경로를 따라 렌더한 영상에서 계산한다는 점이 핵심이다.

### 9.6 조건화 모드
- **text → world**, 영역 프롬프트 페인팅
- **image(s) → world**: 히어로 스틸을 관측으로 쓰고 $\nabla_z\|\mathrm{Render}_{C_h}(D(z))-I_h\|$로 DPS식 가이던스를 건다. 히어로 프레임은 이미지 모델 원본 그대로 재현된다.
- **scan → 초현실 세계 (현실 초해상도)**: 폰으로 스캔한 실제 공간을 코스 레벨에 고정하고 미세 대역만 생성한다. 내 방을 8K 매크로까지 촬영할 수 있게 된다.
- **편집**: 셀을 잠그고 영역 프롬프트를 바꾼 뒤 같은 시드로 재생성하면 그 영역만 바뀐다. 순수성 덕분에 편집이 국소적이고 안정적이다.

### 9.7 추론 비용 (대략)
미세 토큰 약 $10^8$ × 4 스텝 × 토큰당 약 0.4 GFLOP(약 200M 파라미터 국소 모델)이면 약 $1.6\times10^{17}$ FLOPs이고, 8×H100급에서 약 1분이다. lens gating으로 실제로는 더 줄어든다.

---

## 10. Phase 2: STRATA-4D, 움직이는 세계

- **바디 계층**: `body` 타입은 static, rigid, articulated(스켈레톤), deformable(MPM/FEM, PhysGaussian 계보), cloth/shell, strand, fluid/smoke, particle/event다. atom은 바디에 귀속된다(바디 id, 로컬 좌표, 스키닝 가중치).
- **시간도 스케일 축이다 (시간 방향 Scale-Time Duality).** 바디 궤적을 시간 웨이블릿 대역으로 표현한다. 셔터 $\Delta t$보다 빠른 대역은 $\Sigma_{\text{mb}}$(대역 에너지로부터 계산하는 블러 공분산)로만 통계적으로 렌더한다. 프레임 나이퀴스트와 $1/\Delta t$ 사이 대역은 명시적으로 유지하는데, 그래야 **왜건 휠 효과**까지 진짜로 재현된다. 헬기 프로펠러가 멈춰 보이는 그 현상이다. 비디오 모델은 이걸 못 하거나 우연히만 한다.
- **물리가 고주파 생성기다.** 생성 모델은 초기 상태, 물리 파라미터, 의도와 힘(저주파)만 생성하고, 시뮬레이터가 접촉, 충돌, 천이 탁 펴지는 순간(고주파 불연속)을 만든다. 픽셀 디퓨전의 스펙트럼 편향은 불연속을 문질러 버리고, 그게 **버터 모션**의 정체다. 시뮬레이터는 문지르지 않는다.
- **Gabor 물결**: 물 표면의 디테일 대역을 이루는 Gabor atom의 위상을 분산 관계로 진행시킨다.
$$\omega=\sqrt{g|\mathbf{k}|\tanh(|\mathbf{k}|d)}$$
  프레임별 생성 없이 물리적으로 맞는 파도가 나온다. Water Wave Packets(Jeschke & Wojtan 2017)과 Tessendorf FFT ocean의 계보다. Phase 1의 디테일 프리미티브가 그대로 Phase 2의 동역학 프리미티브가 된다.
- **비디오 prior**: 비디오 디퓨전을 시공간 스펙트럼 라우팅으로 교사로 쓴다. 비디오 timestep을 시간 대역에 대응시킨다.
- **영속성**: 정적인 부분은 커밋된 지층으로 남는다. 이벤트는 시간 축 위의 국소 편집이고, 시간 청크별 Merkle을 둔다. **세계는 남고 사건은 일어난다.**
- **Native 4D diffusion**: 공간 latent 대역과 바디 제어 궤적의 시간 대역을 함께 디노이징한다. 공간 방향의 순수성은 그대로 유지한다.

---

## 11. 평가: Bullet-Time Bench (불릿타임 벤치)

Phase 1은 매트릭스 불릿타임을 아무 세계에서나 찍는 일이다. 비교 대상은 Seedance, Kling, Veo, Sora에 "시간 정지, 카메라만 이동"을 지시한 결과다.

### 11.1 데모 샷 8선 (각 샷이 매개체의 어떤 기능을 고문하는가)

| # | 샷 | 고문 대상 |
|---|---|---|
| 1 | **Powers of Ten**: 항공에서 무당벌레까지 연속 줌 $10^5$배 | 스케일-시간 이중성, 효도 제약, 의미 줌 |
| 2 | **거울과 유리 복도** | RT 반사·굴절, 카메라 뒤편 반사 |
| 3 | **얼어붙은 물보라 불릿타임** | DIEL, 얇은 수막, 분산 |
| 4 | **털과 이끼 매크로** | STRAND, SGGX aggregate, f/2.8 보케 |
| 5 | **비 오는 네온 거리, 정지** | HDR, 플레어, 회절 스타, 웅덩이 반사 |
| 6 | **정지한 축제 군중 사이 FPV, 120fps** | 가림, 모션블러, 롤링셔터 |
| 7 | **360° 궤도 루프 클로저** | 영속성 |
| 8 | **같은 세계의 골든아워에서 블루아워로** | 재조명, 샷 조명 리그 |

### 11.2 지표

- **재투영 광도 오차 (RPE)**:
$$E_{\text{rep}}=\frac{1}{|\Omega|}\sum_{p\in\Omega}\Big\|I_t[p]-I_{t+k}\big[\pi_{t+k}\big(\pi_t^{-1}(p,D_t[p])\big)\big]\Big\|$$
  경쟁자는 VGGT/MegaSaM으로 포즈와 깊이를 추정하고, 우리는 GT를 쓴다. 목표는 센서 노이즈 바닥 수준이다.
- **루프 클로저 거리**: 궤도의 시작 프레임과 끝 프레임 사이 LPIPS와 DINO 거리.
- **MEt3R** (2025, multi-view 일관성 지표).
- **재구성 가능성**: 프레임 90%로 3DGS를 피팅하고 나머지 10%에서 PSNR/LPIPS를 잰다.
- **효도 점수 (줌 모순)**: $\text{LPIPS}\big(\downarrow_k I_{\text{close}},\ \text{crop}(I_{\text{far}})\big)$
- **디테일 안정성과 텍스처 헤엄 지수**: 고주파 에너지의 시간 변동계수, 그리고 $\|\text{flow}(\text{HP}(I_t),\text{HP}(I_{t+1}))-\text{flow}_{\text{rigid}}\|$
- **과스무딩 지수**: 카메라 가속도와 jerk의 고주파 파워를 실제 촬영 궤적과 비교한 비율.
- **광학 정합성**: CoC 대 깊이의 얇은 렌즈 모델 피팅 잔차, 회절 스파이크 수와 블레이드 수의 일치 여부, 합성 장면에서 path-traced GT 대비 거울 오차.
- **해상도**: 세계 안에 **ISO 12233 차트 객체를 실제로 배치**하고 MTF50을 잰다. 경쟁자의 업스케일 8K 대비 우리 네이티브 8K다. 세계 속에 해상도 차트를 놓는다는 것 자체가 우리 패러다임의 시위다.
- **인간 Elo 아레나**: 촬영감독과 일반 시청자의 쌍대 선호.

**Phase 1 목표치**: RPE는 노이즈 바닥, 루프 클로저는 약 0, 8K에서 MTF50 ≥ 0.25 cy/px, "정지 세계" 프롬프트에서 최강 경쟁자 대비 Elo 승률 60% 이상.

---

## 12. 약점을 후퇴가 아니라 업그레이드로

| 리스크 | 업그레이드 |
|---|---|
| 이미지 모델의 미감이 뷰에 의존한다 (구도와 조명 치팅) | **샷 조명 리그**를 매개체에 포함하고, 히어로 프레임은 앵커링으로 원본 재현하고, 미니맥스 카메라를 쓴다 |
| CG 또는 게임 룩 | 외관은 이미지 prior에서 오고(작가가 손으로 만든 게 아님), 물리 카메라 전 체인과 Neural Sensor, 필름 에뮬레이션을 적용한다. CG와 실사를 가르는 것의 절반은 카메라다 |
| 얇은 구조와 투명체 | STRAND, DIEL 1급 타입, G-buffer 교사 |
| 재질 분해의 모호성 | 잔차 예산, intrinsic 교사, 불확실성 게이트 비율 재조명 |
| 연산과 메모리 | 지연 평가, latent 저장, lens gating(배경 약 1/1000) |
| 동기화 수렴 실패와 블러 | 세계 고정 노이즈, 소유권 annealing, 레벨당 한 옥타브씩만 결정 |
| native 모델의 데이터 부족 | S²R 데이터 엔진, 절차적 합성, 실제 재구성의 3중 공급 |
| 비트 단위 결정론 | 결정론적 커널, 커밋 메모이제이션이 정본 |
| 의미 전환 줌 (잎에서 세포로) | 스케일 키잉 prompt_tree, 레벨별 CFG |
| 경계 이음매(seam) | Jacobi 패스, 분산 보존 블렌딩, 공유 부모와 공유 노이즈 |

---

## 13. ★ 신규성 장부: 솔직 버전

내가 아는 범위(2026년 중반까지)를 기준으로 적는다.

| ★ | 아이디어 | 가장 가까운 선행 | 우리의 새로운 점 | 확신도 |
|---|---|---|---|---|
| 1 | **스케일-시간 이중성 세계**: LOD 피라미드 = 생성 궤적, 물리 미터 레벨, 카메라 footprint가 디노이징 깊이를 결정, 중간 상태가 렌더 가능 | VAR, Pyramidal Flow, Relay, Powers of Ten, XCube, LT3SD, mip-map | 세 축(디퓨전 시간, 레벨, 카메라 거리)을 하나로 묶고, 외관 공간 효도 제약과 지연 평가를 결합 | **높음** |
| 2 | **경로 독립 순수 세계 함수**: 유계 의존 원뿔, Jacobi 패스, 형식적 경로 독립성 정리, 생성 순서축을 시간에서 스케일로 | 마인크래프트 월드젠, 청크 기반 장면 생성(NuiScene, BlockFusion: 순차적), MultiDiffusion | 디퓨전 월드 생성에서 순서 무관성과 지연 평가의 정확성을 보장, 오차 누적 깊이 O(log 스케일) | **높음** |
| 3 | **footprint 정합 픽셀별 timestep 교사와 스펙트럼 라우팅** | Differential Diffusion, Diffusion Forcing, SDS 타임스텝 annealing | 깊이와 초점거리에서 유도한 픽셀별 $T[p]$로 정확히 한 옥타브만 결정하고, 그래디언트를 레벨로 라우팅 | 중상 |
| 4 | **Lens-gated band generation**: 광학이 생성 범위를 결정 | voxel cone tracing, DOF-GS | 보케와 모션블러가 생성 자체를 생략시킴 (렌더 최적화가 아니라 생성 최적화) | 중상 |
| 5 | **샷 조명 리그를 매개체의 구성요소로**: 이미지 모델 조명 치팅의 합법화 | 실제 영화 조명 관행, 재조명 연구 | 생성 세계의 영속성과 샷별 미감의 분리 설계 | 중 |
| 6 | **미니맥스 촬영감독**: 디퓨전 손실을 implausibility로 쓰는 최악 카메라 마이닝 | FisherRF, ViewFool | 생성 세계 품질의 sup 최적화 | 중 |
| 7 | **외관 공간 효도 제약과 SGGX aggregate를 생성 LOD로** | SGGX, Loubet–Neyret LoD | 생성 대역의 커밋 조건으로 사용 | 중 |
| 8 | **소유권 annealing 동기화 Tweedie와 물리 중재** | SyncTweedies, DMV3D, 3D-Adapter, best-view 텍스처링 | 패럴랙스, 스페큘러, 환각 3분류 중재와 승자독식 annealing | 중 |
| 9 | **Gabor atom을 Phase 1 디테일과 Phase 2 물결 동역학의 공통 프리미티브로** | Gabor/텍스처드 스플랫 시도들, Water Wave Packets | 생성 대역 프리미티브가 그대로 물리 동역학 프리미티브가 됨 | 중하 |

**"이건 아무도 안 해봤다"고 가장 강하게 거는 것은 1과 2의 결합이다.** 세계를 미터 단위 스케일 트리 위의 순수 함수로 정의하고, 그 트리를 디퓨전 궤적 자체로 만들고, 카메라 광학이 어느 가지를 평가할지 결정하게 하는 것이다.

---

## 14. 다른 렌즈 동료들과의 인터페이스, 그리고 마무리

다른 렌즈(카메라와 연출, 미감, 동역학, 시스템)와 맞물릴 최소 API는 이 정도다.

```python
world  = strata.create(seed, prompt_tree, bedrock=None | scan | hero_stills)
demand = world.demand(camera_path, lens_rig, shutter, fps, res)   # → {(level, block)}, 값 불변
world.realize(demand, mode="native" | "s2r_polish")               # 순수·병렬·메모이즈
frames = strata.render(world, camera_path, lens_rig, shot_rig, sensor, look)
world.uncertainty(region)   # 카메라 경로 플래너용: 어디가 아직 '슈뢰딩거'인가
world.edit(region, op)      # copy-on-write 브랜치, 새 merkle_root
world.query_cone(o, d, rho) # 제1원리 그 자체
```

### 요약 세 줄
1. **비디오 AI는 매 프레임 세계를 다시 상상하고, 우리는 세계를 한 번 퇴적시킨 뒤 카메라로 질의한다.**
2. **아웃페인팅과 비디오 디퓨전은 시간 순서 체인이고, STRATA는 스케일 순서 트리다.** 구멍을 메우지 않고 초점을 맞춘다. 줌인이 곧 디노이징이다.
3. **렌즈, 셔터, 센서는 후처리가 아니라 세계에게 얼마나 디테일해지라고 명령하는 주체다.** 그래서 8K·120fps는 비용이 아니라 무기다.

비디오 디퓨전은 건망증 걸린 창조주다. 우리는 창조주를 해고하지 않는다. 그 창조주(이미지 디퓨전)에게 딱 한 번, 대신 모든 스케일에서 아주 꼼꼼하게 세계를 창조하게 하고, 그 결과를 지층으로 굳힌 다음 촬영감독을 들여보낸다. 그러면 Phase 1에서 Seedance와 Kling이 "멈춘 세계"를 찍으려다 손잡이를 이사시키는 동안, 우리는 같은 머그컵을 2km 상공에서 내려와 손잡이의 유약 기포까지 8K·120fps로 찍고, 카메라를 한 바퀴 돌려 처음 자리로 돌아와 "봐, 아직 거기 있지?"라고 말한다.

Phase 2에서는 그 머그컵을 떨어뜨린다. 깨지는 순간의 고주파는 디퓨전이 아니라 물리가 만들고, 파편은 세계에 남는다.
