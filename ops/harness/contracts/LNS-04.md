# LNS-04 센서·필름: Poisson-Gauss · ADC · 65³ LUT · 할레이션 · 비네팅 · Newson 그레인

**목적.** 렌즈·셔터 뒤, 화면에 고정되는 마지막 물리 스택(PARADIGM §7.6 "센서"+"룩"). 학습 파라미터 0, 순수 수식. 체인 순서는 **EV 게인 → 비네팅 → 할레이션 → 센서(Poisson-Gauss·만충전 클립·ADC) → 로그 부호화 → 65³ LUT → 그레인**. 그레인이 유일한 프레임별 난수(센서 잡음 포함 두 확률 요소는 `seed`·`frame`으로 결정).
**소유.** `buc/lens/sensor_film.py`(`buc/lens/__init__.py` 금지, 네임스페이스), 테스트 `tests/lns_04/`(동결 `LNS-04.tests.json`). 의존: FND-06 `buc.core.color`, FND-02 `Theta`·`Frames`, LNS-01 `buc.lens.theta.LUT_NAMES`, numpy·scipy. **torch import 금지**, 전역 RNG·시각 금지(난수는 `np.random.Generator`, 스트림은 `(seed, frame, 용도)`로 분리), RuntimeWarning 금지. 모든 입력 배열 불변, 새 배열 반환.

## 공개 API (`from buc.lens import sensor_film as sf`)
정본 상수: `LUT_SIZE=65`, `LOG_MIDGRAY=0.18`, `LOG_STOPS=(-8.0,6.0)`, `GRAIN_SIGMA_MAX=0.08`, `GRAIN_RADIUS_FRAC=0.0012`, `GRAIN_RADIUS_MIN_PX=0.75`, `HALATION_THRESHOLD=0.8`, `HALATION_SIGMA_FRAC=0.02`, `HALATION_TINT=(1.0,0.35,0.12)`, `HALATION_GAIN=1.0`, `VIGNETTE_MAX_DROP=0.8`, `FULL_WELL_PER_UM2=1500.0`, `FULL_WELL_MIN/MAX=1500/100000`, `READ_NOISE_E=2.5`, `ADC_BITS=12`.
```python
@dataclass(frozen=True, kw_only=True)
class SensorSpec: full_well_e: float = 20000.0; read_noise_e: float = 3.0; bits: int = 12   # >0 유한 / >=0 / int 1..16, 위반 ValueError
sensor_spec_from_theta(theta, width_px) -> SensorSpec      # 피치[um]=sensor_width_mm*1000/width_px, FW=clip(1500*피치², 1500, 1e5), 읽기잡음 READ_NOISE_E, bits ADC_BITS
adc_quantize(x, bits) -> f32                               # clip[0,1], 최근접 반올림, k/(2^bits-1) 격자, 멱등
sensor_capture(rgb_lin, spec, *, seed=0, frame=0, noise=True) -> f32   # 임의 모양 float 배열
log_encode(x) / log_decode(e)                              # e=clip((log2(max(x,tiny)/0.18)-lo)/(hi-lo),0,1), 입력 dtype 유지
identity_lut(size=65) -> (S,S,S,3) f32                     # [r][g][b] 인덱스, 노드 i ↔ i/(S-1)
make_lut(name, size=65) -> (S,S,S,3) f32 [0,1]             # name ∈ LNS-01 LUT_NAMES(미등록·size<2 ValueError), 결정론, 매번 새 복사본
apply_lut3d(coded, lut) -> f32                             # 마지막 축 3, 입력 [0,1] 클램프 후 삼선형(아핀 정확), 정육면체·크기≥2 아니면 ValueError
apply_look(rgb_lin, lut: str | ndarray) -> f32 [0,1]       # 선형 -> 로그 -> LUT. "none"은 LUT 우회 = color.linear_to_display(비트 동일)
vignette_gain(h, w, strength) -> (h,w) f32                 # 1 - 0.8*strength*r², r=중심거리/반대각선(모서리 화소 중심에서 r=1), 상하좌우 대칭
apply_vignette(rgb, strength) / apply_halation(rgb_lin, amount) / halation_layer(rgb_lin, amount)   # (H,W,3), amount∈[0,1]
grain_field(u, *, strength, radius_px, seed=0, frame=0) -> f32   # u (H,W)|(H,W,C) ∈[0,1] 표시값
apply_sensor_film(rgb_lin, theta, *, seed=0, frame=0, spec=None, noise=True) -> f32 (H,W,3) 표시값 [0,1]
apply_sensor_film_frames(frames: Frames, theta, *, seed=0, spec=None, noise=True) -> Frames
```
`seed`는 int `[0,2^64)`, `frame`은 int `[0,2^32)`(bool·float은 TypeError, 범위 밖 ValueError). float32/64 외 dtype TypeError, NaN/inf·잘못된 모양·범위 밖 인자는 ValueError(ContractError 포함). `seed` 등 옵션은 키워드 전용.

## 수식·의미
- **센서.** 입력 1.0 = 만충전 노출. 전자수 `~Poisson(min(x,·)·FW)`를 FW에서 자르고 **그 뒤** `N(0, read_noise_e²)`를 더한 뒤 `[0,1]` 클립과 ADC. 따라서 `Var = x/FW + σr²/FW² + q²/12`(q=1/(2^bits-1)), 평균은 x(포화 미만). 만충전 위(x≥1.5)는 `1-O(σr/FW)`로 포화(읽기잡음 0이면 정확히 1.0). 음수 입력은 0. 화소·채널 독립. `noise=False`면 양자화만(시드 무관).
- **로그·LUT.** `neutral_709` 노드 = `linear_to_display(log_decode(좌표))`(= sRGB 기준 표시 변환), `make_lut("none")`은 같은 표. 나머지 8룩은 그 위의 그레이딩: teal_orange(그림자 청록·하이라이트 주황), bleach_bypass(채도↓·대비↑), kodak_warm/fuji_cool(R>B / B>R), mono_silver(R=G=B), faded_matte(검정≥0.03·흰색≤0.98), neon_night(채도↑), cross_process(채널별 곡선). 모든 룩: 회색축 채널별 비감소, 인접 노드 점프 <0.15, 서로 구별됨.
- **비네팅/할레이션.** 할레이션 층 `amount·GAIN·tint_c·G_σ*max(Y-0.8,0)`, `Y=color.luminance`, σ=0.02·W(폭에 비례), 영 경계, 층은 비음수·HDR 비클램프, 에너지는 광원 초과분에 비례. amount=0은 비트 복사.
- **그레인(Newson 계열, 구현 자유).** 상수 u₀에서 평균 u₀, 화소 표준편차 `strength·0.08·2√(u₀(1-u₀))`(반경과 무관), 채널 독립, 정규화 자기상관 ρ(r/2)∈[0.5,0.92], ρ(r)∈[0.15,0.6], ρ(≥2.5r)≈0, 타일 반복 흔적 없음(512²에서 지연 96~256 px 상관 <0.1), 프레임·시드마다 새 실현(상관 <0.1). strength=0은 비트 복사. 체인에서 `radius_px=max(0.75, 0.0012·W)`.
- **체인.** `ev`는 `2^ev` 게인으로 LNS-04가 소유(하류가 다시 적용하지 말 것). `wb_kelvin/wb_tint`는 **LNS-04 범위 밖**(미소유, 아래 요청). `grain/halation/vignette/lut`은 θ 필드 그대로, lut 미등록은 ValueError. `noise=False`는 센서 잡음과 그레인을 모두 끈다(ExplainAway 왕복·결정론 비교용; ADC 양자화는 남는다).
- **Frames 어댑터.** `Frames.rgb`(표시값) → `srgb_to_linear` → i번째 프레임을 `frame=i`로 체인 → 새 `Frames`(fps·shot_hash·world_hash 유지, aov는 복사 유지). 비-Frames TypeError. HDR(>1) 원천은 배열 수준 `apply_sensor_film`을 직접 쓴다.

## 불변식
결정론(같은 입력·seed·frame → 같은 바이트, 같은 numpy 환경) · 입력 불변 · 출력 `[0,1]` float32 · 기본 θ+`noise=False`는 `linear_to_display`와 3e-3 이내(12비트 양자화) · 센서 잡음과 그레인 스트림 독립 · 순수 import 부작용 없음.

## 수용 기준
`tests/lns_04/` 전부 통과(CPU, <60 s, `-m "not gpu"`, 동결 해시 `LNS-04.tests.json`): ADC 격자, 사진 전달 곡선(분산 기울기 = 1/FW ±5%), 포화·검정·결정론·독립성, 삼선형 LUT 아핀 정확성, 룩별 성격, 비네팅·할레이션 오라클(scipy 가우시안), 그레인 평균·표준편차·상관 길이·무타일링, 체인 순서(공개 조각의 합성과 일치), Frames 이음매, torch 미로드, 360×640 전 스택 <8 s. `V`(사람 시각 판정: 룩·그레인 취향)와 GPU 이식은 이 테스트 밖이다.

## 하류 이음매
LNS-07(CPL)은 이 모듈 뒤가 아니라 앞에 놓인다(선형 장면). LNS-08은 `apply_sensor_film(…, noise=False)`를 순방향으로, `vignette_gain`·`halation_layer`·`make_lut`·`log_encode`를 역사슬 오라클로 쓴다(역변환은 RD-03/RD-06 소유; 센서/그레인은 역변환 불가이므로 소유권 마스킹). RD-03은 `GRAIN_SIGMA_MAX`·`VIGNETTE_MAX_DROP`·`HALATION_*`와 `vignette_gain` 형태를 추정 모델로 맞춘다. 상수 변경은 호환 파괴이므로 계약서와 함께 올린다.
**요청(범위 밖).** (1) WB(`wb_kelvin/wb_tint`) 적용 단위가 백로그에 없다. LNS-08(체인 합성기)이 맡을지 결정 필요. (2) LNS-02/03 출력 `Frames`는 표시값이라 >1 하이라이트가 없다. 할레이션 품질이 필요하면 선형 부동소수 AOV를 상류가 제공해야 한다.
