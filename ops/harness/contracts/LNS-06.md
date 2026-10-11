# LNS-06 애너모픽·빈티지 처방: 출사동 커널(Kolb 근사) · 스월 · 스퀴즈 · 수평 플레어 · 주변부 번짐

**목적.** PARADIGM §7.6 "렌즈 처방 → 출사동 LUT", "빈티지 처방(필드 커버처·스월 보케·주변부 번짐)", "애너모픽(2× 스퀴즈, 타원 보케, 수평 플레어)". 광선추적 LUT 대신 **닫힌 식 근사**로 구현한다(학습 0). 실제 Kolb LUT가 나중에 오더라도 `exit_pupil_kernel` 시그니처를 유지해 갈아 끼운다. 수차 부품(왜곡·CA·글레어·고스트·별·cat-eye)은 LNS-05가 이미 줬으므로 여기서 **다시 만들지 않고** 그 위에 얹는다.
**소유.** `buc/lens/prescriptions.py`(`buc/lens/__init__.py` 금지), 테스트 `tests/lns_06/`(동결 `LNS-06.tests.json`). **의존.** `buc.lens.optics`(LNS-05: `OpticsSpec`·`catseye_kernel`·`apply_catseye`·`spec_from_theta`·`apply_optics`·`apply_distortion`·`CATSEYE_MAX_SHIFT`), `buc.lens.theta`(`LENS_RX_NAMES`), `buc.core.color/contracts`, numpy·scipy. **torch import 금지**, import 부작용·전역 RNG·RuntimeWarning 금지, 결정론, 입력 불변, 새 배열 반환.

## 규약
영상은 LNS-05와 같다: `(H,W,3)` float32/float64(그 밖은 `TypeError`, 모양·비유한·범위 밖·bool 수치는 `ValueError`), 화소 중심이 정수 `(x=열,y=행)`, 주점 `c=((W-1)/2,(H-1)/2)`, `r_frac=|p-c|/R`, `R=hypot(W-1,H-1)/2`. 각도 `t`는 방향 `(cos t, sin t)`, 도, y는 아래, **"각도 증가 방향" = x축에서 y축 쪽**. 선택 인자는 키워드 전용.

## 공개 API (`from buc.lens import prescriptions`; `__all__` 에 모두 포함)
상수: `SWIRL_MAX_TWIST_DEG=60.0`, `EDGE_SIGMA_FRAC=0.015`, `EDGE_R_INNER=0.35`, `SQUEEZE_MAX=3.0`, `DEFAULT_STREAK_TINT=(0.55,0.75,1.0)`.
```python
@dataclass(frozen=True, kw_only=True) Prescription   # 필드와 기본값(이 순서); 검증은 __post_init__
  optics: OpticsSpec = OpticsSpec()  # LNS-05 스펙(왜곡·CA·cat-eye·날 수·글레어·고스트·별). OpticsSpec 이 아니면 TypeError
  squeeze: float = 1.0               # [1, SQUEEZE_MAX]  애너모픽 수평 스퀴즈
  swirl: float = 0.0                 # [0,1]  스월 보케 세기
  edge_blur: float = 0.0             # [0,1]  주변부 번짐(상면 만곡)
  streak: float = 0.0                # [0,1]  수평 플레어 이득
  streak_len_frac: float = 0.35      # (0,1]  플레어 길이 / 폭
  streak_tint: tuple = DEFAULT_STREAK_TINT   # 길이 3, 각 >=0 (시퀀스 -> tuple)
  streak_threshold: float = 0.8      # >=0
  name: str = "custom"               # 비어 있지 않은 str
  # 읽기 전용 속성: catseye, blades, blade_rot_deg  (= optics 의 값)
pupil_open_fraction(r_frac, catseye) -> float   # 원형 조리개 cat-eye 개구율 (2/π)(acos q − q√(1−q²)), q = CATSEYE_MAX_SHIFT·r_frac·catseye
exit_pupil_kernel(diameter_px, r_frac, angle_deg, rx: Prescription) -> (K,K) float32   # 합 1, 비음수
edge_softness(rgb, amount) -> ndarray
streak_layer(rgb_lin, *, gain, length_frac=0.35, tint=DEFAULT_STREAK_TINT, threshold=0.8) -> ndarray   # 가산 층(비음수)
apply_prescription_bokeh(rgb, coc_px, rx) -> ndarray      # coc_px (H,W) 비음수 유한(수치 dtype, 아니면 TypeError/ValueError)
RX_TABLE: Mapping[str, Mapping]    # 키 == set(LENS_RX_NAMES); 각 행은 swirl, edge_blur, streak (+선택 streak_len_frac, streak_tint) = Prescription 키워드
prescription_from_theta(theta: Theta, width_px: int) -> Prescription
apply_prescription(rgb_lin, rx) -> ndarray       # 선형 장면용
apply_prescription_frames(frames: Frames, theta: Theta, *, rx=None) -> Frames
```

## 수식·의미
- **exit_pupil_kernel.** `d<=1` 이면 `[[1]]`. 아니면 ① 바탕 `B = optics.catseye_kernel(d, r_frac, angle, rx.catseye, rx.blades, rx.blade_rot_deg)`. ② 크기 `K=2·ceil(d·√s/2)+1`(s=squeeze; s=1 이면 B와 같다). ③ **스월**: `τ=swirl·SWIRL_MAX_TWIST_DEG·r_frac`(도). 출력 화소 p(중심 기준) 를 `q=(p_x·√s, p_y/√s)` 로 보내고, `ρ=|q|` 에서 `B` 를 `각도 −τ·ρ/(d/2)` 만큼 되돌려 쌍선형 표본화(밖은 0) — 형상이 각도 증가 방향으로 ρ에 비례해 비틀린다. 면적·반경 분포 보존. ④ **스퀴즈**: 위 `q` 사상이 곧 스퀴즈다(가로 1/√s, 세로 √s, 기하평균 보존; 분산비 `var_y/var_x = s²`). ⑤ 합 1로 재정규화. **`τ=0` 이고 `s=1` 이면 `B` 와 배열이 동일**(`np.array_equal`). 질량중심은 커널 중심.
- **pupil_open_fraction** 은 LNS-05 cat-eye 렌즈(블레이드 0)의 넓이 / 원 넓이의 닫힌 식. LNS-04/08이 기계적 비네팅에 쓸 수 있다(블러 커널은 합 1로 에너지 보존).
- **edge_softness.** `out=(1−w)·x + w·G(x)`, `w=clip((r−EDGE_R_INNER)/(1−EDGE_R_INNER),0,1)²`, `G`=`scipy.ndimage.gaussian_filter(σ=amount·EDGE_SIGMA_FRAC·W, mode="nearest", truncate=4.0)`(공간만, 채널 독립). `amount=0` 은 복사, `r<=EDGE_R_INNER` 는 입력과 비트 동일.
- **streak_layer.** `E=max(x−threshold,0)`. 행마다 `h(k)=exp(−|k|/L)/Z`(k=−(W−1)..W−1, `L=length_frac·W`, 합 1)와 합성 → 세로 3탭 `[.25,.5,.25]`(밖은 0) → `gain·tint`(채널별). 광원이 중앙이고 L이 작으면 광속≈`gain·tint_c·ΣE_c`.
- **apply_prescription_bokeh.** 화소마다 자기 CoC 와 `(r_frac, angle)` 의 `exit_pupil_kernel` 로 퍼뜨린다(LNS-05 `apply_catseye` 와 같은 구조: CoC<=1 px 는 선명, 각도 7.5°·r 1/16·d 단계 양자화, 프레임 경계 안쪽 `half` 이내는 비규정). **`swirl=0` 이고 `squeeze=1` 이면 `optics.apply_catseye(rgb, coc, strength=rx.catseye, blades=rx.blades, blade_rot_deg=rx.blade_rot_deg)` 와 같다**(atol 1e-5).
- **prescription_from_theta.** `optics=optics.spec_from_theta(theta,width_px)`(그대로), `squeeze=float(theta.extra.get("squeeze", 2.0 if anamorphic_2x else 1.0))`, 나머지는 `RX_TABLE[lens_rx]`. `(lens_rx=="anamorphic_2x") != (squeeze==2.0)` 는 `ValueError`. 값은 구현 자유, **순서 제약**: spherical_clean 은 swirl=0, edge_blur<=0.1, streak=0; vintage_petzval 은 swirl>=0.5, edge_blur>=0.5, optics.catseye>=0.5; vintage_double_gauss 는 swirl∈[0.1,0.4]이고 petzval swirl>=2×그 값, edge_blur 는 clean<dg<petzval; anamorphic_2x 는 squeeze=2, streak>=0.3, swirl<=0.1, tint 의 B>R; soft_single_coat 는 edge_blur>=0.2, optics.glare>=0.05; anamorphic 외엔 squeeze=1·streak=0. swirl/edge_blur/streak/len_frac 는 해상도 불변, 수차 px 만 폭에 선형. 미등록 lens_rx·잘못된 width_px 는 `ValueError`, Theta 아니면 `TypeError`.
- **apply_prescription.** `out = optics.apply_optics(edge_softness(x, rx.edge_blur), rx.optics)` + (`rx.streak>0` 이면) `streak_layer(x, gain=rx.streak, length_frac=rx.streak_len_frac, tint=rx.streak_tint, threshold=rx.streak_threshold)`(스트릭은 원본 x 에서, 왜곡 뒤에 더한다). 기본 `Prescription()` 은 복사. **`rx.optics.catseye` 와 스월·스퀴즈 블러는 소비하지 않는다**(호출자가 `apply_prescription_bokeh`).
- **apply_prescription_frames.** `srgb_to_linear` → 프레임마다 `apply_prescription(rx or prescription_from_theta(theta,W))` → `linear_to_display`. `(N,H,W)` AOV 는 `rx.optics` 의 왜곡으로 **최근접** 사상, 나머지 AOV·fps·해시 유지. 비-Frames·비-Prescription 은 `TypeError`.

## 수용 기준
`tests/lns_06/` 전부 통과(CPU, `-m "not gpu"`, <60 s, 결정론, 동결 해시 `LNS-06.tests.json`): 스월 축 회전량(0.3/0.6/1.0 → 약 12°/24°/40°)·r_frac 비례·각도 방향·질량/이차 모멘트 보존, 스퀴즈 분산비 s²·기하평균 보존, cat-eye 개구율 해석식과 커널 넓이 일치, 점광원 출력=해당 위치 커널(광속 보존), LNS-05 이음매(`apply_catseye` 동일), 스트릭·번짐의 수식 오라클, 순서 제약, 체인 합성 일치, Frames 이음매, torch 미로드, 320×180 체인 <12 s. 작성자는 스크래치 참조 구현으로 전부 통과를 확인했다(구현은 이 저장소에 없음). 룩의 취향 판정(V)과 GPU 이식은 테스트 밖이다.

## 하류 이음매 · 요청(범위 밖)
- LNS-08 은 순방향에 `apply_prescription`(+DOF 단계에서 `apply_prescription_bokeh`), `prescription_from_theta` 를 쓴다. 역사슬 오라클은 `rx.optics` 로 LNS-05 의 `undistort_image` 를 쓴다(번짐·스트릭·고스트는 소유권 마스킹).
- **요청 1.** 깊이 층 DOF(LNS-02)와 `apply_prescription_bokeh`(단일 층)를 합치는 방식(원경 층 교체 등)은 LNS-08 이 정한다. **요청 2.** 표시값 `Frames` 에는 >1 하이라이트가 없어 스트릭·고스트가 약하다. 선형 float AOV 가 필요하다(LNS-04/05 요청과 동일). **요청 3.** 실제 Kolb 광선추적 LUT 는 이 근사식을 대체할 때 `exit_pupil_kernel` 시그니처를 지킬 것.
