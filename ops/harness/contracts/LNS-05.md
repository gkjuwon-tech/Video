# LNS-05 렌즈 수차: 왜곡 · 횡색수차 · cat-eye · Hullin 고스트 · veiling glare · 회절 별

**목적.** PARADIGM §7.6 "보케 스프라이트, Hullin 고스트, FFT veiling glare, 회절 스타" + 렌즈 처방의 왜곡·색수차·cat-eye 를 근사식으로 구현한다(학습 0). 선형 장면 영상에 작동한다. 실제 광선추적 처방은 LNS-06 몫이고, 여기는 그 아래의 수식 부품과 `LENS_RX_NAMES` 별 기본 스펙을 준다.
**소유.** `buc/lens/optics.py`(`buc/lens/__init__.py` 금지, 네임스페이스), 테스트 `tests/lns_05/`(동결 `LNS-05.tests.json`). **의존.** LNS-02 `buc.lens.dof.aperture_kernel`·`_check_blades` 규칙, FND-02 `Theta`·`Frames`, FND-06 `color`, LNS-01 `LENS_RX_NAMES`, numpy·scipy. **torch import 금지**, import 부작용·전역 RNG·RuntimeWarning 금지, 결정론, 입력 불변, 새 배열 반환.

## 규약
영상 `(H,W)` 또는 `(H,W,3)`(색 함수는 3채널 필수), float32/64만(그 밖은 `TypeError`), 비유한·잘못된 모양·범위 밖 인자·bool 수치는 `ValueError`. `coc_px` 는 `(H,W)` 비음수 유한. 반환 dtype은 입력과 같다. 화소 중심이 정수 `(x=열,y=행)`, 주점 `c=((W-1)/2,(H-1)/2)`, **정규화 반지름** `r=|p-c|/R`, `R=hypot(W-1,H-1)/2`(모서리 r=1). 각도 `t`는 방향 `(cos t, sin t)`, 도 단위, y는 아래. 조리개 회전·날 수는 `dof.aperture_kernel`과 같은 규약(날 수 0 또는 3..18, 변 법선 `rot+(2i+1)·180/N`). 선택 인자는 키워드 전용.

## 공개 API (`from buc.lens import optics`)
상수: `GLARE_EXPONENT=1.5`, `GLARE_R0_FRAC=0.03`, `STAR_CHROMA=(1.12,1.0,0.88)`(R,G,B 파장비), `STAR_PSF_SIZE=257`, `STAR_APERTURE_FRAC=0.25`, `CATSEYE_MAX_SHIFT=0.5`.
```python
# 왜곡. 후방 사상 out(p) = in(c + (p-c)·s(r)),  s(r) = 1 + k1 r² + k2 r⁴.  k1>0 배럴, k1<0 핀쿠션.
distortion_scale(r, k1, k2) -> ndarray
apply_distortion(img, k1, k2, *, border="edge"|"zero", nearest=False, center=None) -> ndarray  # k=0 이면 복사, nearest=깊이/라벨용
undistort_image(img, k1, k2, *, border="edge", nearest=False, center=None)   # out(q)=in(ideal_to_distorted(q)); apply_distortion의 역
ideal_to_distorted(xy, k1, k2, shape_hw, center=None) -> (...,2)   # 이상(핀홀) 화소 -> 왜곡 영상 위치. r_d(1+k1 r_d²+k2 r_d⁴)=r_q 를 뉴턴으로 푼다
distorted_to_ideal(xy, k1, k2, shape_hw, center=None) -> (...,2)   # 닫힌 식 p·s(r)
# 유효: 유한, |k|<=2, 1+3k1 r²+5k2 r⁴ > 0 (r∈[0,1]; 단조). 아니면 ValueError.
apply_lateral_ca(rgb, red_px=0.0, blue_px=0.0, *, border="edge")   # (H,W,3). 채널 c 를 배율 1+e_c (e=px/R) 로 확대, 초록 기준. 복사(휘도 보존) 표본화. |px|<0.5R
# cat-eye: 반지름 d/2 조리개 두 개를 간격 o=CATSEYE_MAX_SHIFT·strength·radial_frac·d 만큼 (방향 angle_deg) 어긋나게 겹친 교집합(질량중심을 중심에 맞춤)
catseye_kernel(diameter_px, radial_frac, angle_deg, strength, blades=0, rotation_deg=0.0) -> (K,K) f32  # 합 1, 크기 = dof.aperture_kernel(d) 와 같음, radial_frac·strength=0 이거나 d<=1 이면 aperture_kernel 과 배열 동일
apply_catseye(rgb, coc_px, *, strength, blades=0, blade_rot_deg=0.0) -> ndarray  # 단일 층 산란 블러. 화소마다 자기 CoC·(r,angle)의 커널로 퍼뜨림. CoC<=1 px 는 선명. 각도 양자화 <=8°, 프레임 경계 d/2 안은 비규정
@dataclass(frozen=True) GhostSpec(scale, size_frac, gain, tint=(1,1,1))   # |scale|<=3, size_frac∈[0.005,0.5], gain>=0, tint 3개 >=0. 위반 ValueError
DEFAULT_GHOSTS: tuple[GhostSpec,...]   # >=4개, scale<0 과 scale>1 이 모두 있다
ghost_layer(rgb_lin, ghosts, *, threshold=0.8, blades=0, blade_rot_deg=0.0) -> ndarray  # 가산 층(비음수, 수치잡음 -1e-6·max 허용)
veiling_glare(rgb_lin, amount, *, r0_frac=GLARE_R0_FRAC) -> ndarray   # amount∈[0,1], r0_frac∈(0,1]
star_psf(blades, rotation_deg=0.0, *, size=257, aperture_frac=0.25) -> (size,size,3) f32   # size 홀수 >=33, aperture_frac∈(0,0.5]
star_layer(rgb_lin, *, blades, rotation_deg=0.0, threshold=0.8, size_frac=0.25) -> ndarray  # 단위 세기 층
@dataclass(frozen=True, kw_only=True) OpticsSpec   # 필드와 기본값 (이 순서)
  k1=0, k2=0, ca_red_px=0, ca_blue_px=0, catseye=0 [0,1], blades=0, blade_rot_deg=0, glare=0 [0,1], glare_r0_frac=0.03,
  ghosts=() (시퀀스 -> tuple), ghost_threshold=0.8, star_amount=0 [0,1], star_size_frac=0.25, star_threshold=0.8
spec_from_theta(theta: Theta, width_px: int) -> OpticsSpec
apply_optics(rgb_lin, spec: OpticsSpec) -> ndarray
apply_optics_frames(frames: Frames, theta: Theta, *, spec=None) -> Frames
```

## 수식·의미
- **고스트(Hullin 근사).** `E=max(rgb-threshold,0)`(채널별). 고스트 k: E 를 중심 기준 `p' = c + scale·(p-c)` 로 **광속 보존** 쌍선형 옮기고(scale<0 은 중심 건너편), `aperture_kernel(size_frac·W, blades, rot)` 로 퍼뜨린 뒤 `gain·tint` 곱. 고스트는 가산적이고 광원–중심 직선 위에 놓인다. 총 광속 = `gain·tint_c·ΣE_c`(프레임 안).
- **veiling glare.** `out=(1-a)·x + a·(h*x)/(h*1)`, `h(r)=(1+(r/r0)²)^-1.5`, `r0=r0_frac·W` px, 채널 독립. `h*1`로 나눠 가장자리 정규화(균일 장 불변). amount=0 은 복사. 선형·가산적.
- **회절 별.** 조리개(지름 `aperture_frac·size/s_c` 의 N각형·원, 부분화소 면적 적분)의 `|FFT|²`, 중심 index `size//2`, 채널마다 합 1(채널 c 의 조리개는 `STAR_CHROMA[c]` 로 나눈다: 중심값 비 s⁻²). 광선: N 짝수 N개, 홀수 2N개, 방향 = 변 법선과 그 반대. `star_layer`=E 를 PSF(전체 폭이 `size_frac·W` px 가 되게 재표본, 합 1)로 컨볼루션(광속 보존, blades=0 은 에어리).
- **스펙 체인 `apply_optics`.** `out = glare(x) + ghost_layer(x) + star_amount·star_layer(x)`(셋 다 원본 x 에서), 이어서 `apply_lateral_ca`, 마지막에 `apply_distortion`. 빈 스펙은 복사. **`spec.catseye` 는 apply_optics 가 소비하지 않는다**(블러는 DOF 몫, 호출자가 `apply_catseye(strength=spec.catseye)`).
- **`spec_from_theta`.** `lens_rx`별 표(값은 구현 자유, 아래 순서만 고정), `blades=int(theta.extra.get("blades",0))`, ghosts 는 DEFAULT_GHOSTS 의 gain 을 렌즈별로 곱한 것, `ca_*_px = 비율·width_px`(해상도에 선형), `star_amount` 는 `f_stop` 에 비감소(f/2.8 이하 0, f/16 에서 >0, <=1). 순서 제약: spherical_clean 은 k1=k2=0, CA 0, catseye<=0.1, glare<=0.03; vintage_petzval 은 catseye>=0.5, 적·청 CA 부호 반대·비영, k≠0; soft_single_coat 은 glare>=max(0.05, 3×clean), ghost gain 합>=2×clean; 모든 렌즈의 ghost gain 합과 catseye 는 clean 이상. 미등록 `lens_rx`·width_px<1·비정수 `ValueError`, Theta 아니면 `TypeError`.
- **`apply_optics_frames`.** `srgb_to_linear` → 프레임마다 `apply_optics(spec or spec_from_theta(theta,W))` → `linear_to_display`(클립). `(N,H,W)` AOV 는 같은 왜곡으로 **최근접** 사상, 나머지 AOV·fps·해시는 유지. 비-Frames `TypeError`.

## 수용 기준
`tests/lns_05/` 전부 통과(CPU, `-m "not gpu"`, <60 s, 결정론, 동결 해시 `LNS-05.tests.json`): 왜곡 해석 오라클·점 사상·역왜곡 왕복·직선 휨 방향, CA 중심 이동 `r0·px/R`, cat-eye 렌즈 넓이·방향·에너지, 고스트 위치·광속·조리개 모양, 글레어 FFT 오라클, 별 광선 수·각도(N=4..8, 회전 0/17°), 스펙 순서·체인 합성 일치, Frames 이음매, torch 미로드, 320×180 전 스택 <6 s. 작성자는 스크래치 프로토타입으로 전부 12 s 안에 통과하는 것을 확인했다. 룩의 취향 판정(V)과 GPU 이식은 테스트 밖이다.

## 하류 이음매 · 요청(범위 밖)
- LNS-06 은 `catseye_kernel`·`GhostSpec`·`star_psf`·`OpticsSpec` 위에 처방별 반경 커널(스월, 애너모픽 squeeze)을 얹는다. 시그니처 변경은 이 계약서와 함께 올린다.
- LNS-08 은 순방향에 `apply_optics`, 역사슬 오라클에 `undistort_image`·`ideal_to_distorted` 를 쓴다(CA 역은 `-px/(1+px/R)` 근사, 고스트·글레어는 소유권 마스킹).
- **요청 1.** `apply_catseye`(단일 층)를 깊이 층 DOF(LNS-02)와 어떻게 합칠지(예: 원경 층 교체)는 LNS-06/08 이 정해야 한다. **요청 2.** LNS-02/03/04 출력 `Frames` 는 표시값이라 >1 하이라이트가 없어 `apply_optics_frames` 의 고스트·별은 약하다. 선형 float AOV 가 필요하다(LNS-04 요청 2와 동일).
