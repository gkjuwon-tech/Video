# WLD-17 far-field — 팔면체 HDR · Hosek–Wilkie형 하늘 · 지평선 · 지면 연장

**목적.** 프레임 밖(특히 뒤통수·오빗 끝) 하늘과 지면을 **해석식 + 파라미터 0**으로 채우는 무한원 환경. 결과는 FND-02 `FarField`(팔면체 `(S,S,3)` f32)로 `Strata.set_far_field`에 들어간다. 팔면체 모서리(−z)와 네 변에서 값이 이어져 **이음매가 없어야** 한다.
**소유.** `buc/world/farfield.py`(namespace import, `__init__.py` 없음), 테스트 `tests/wld_17/`(동결 `WLD-17.tests.json`). 의존: numpy, `buc.core.contracts`(FarField·ContractError)만. **torch·scipy 금지(런타임 솔버 불필요)**, import 부작용·전역 RNG 없음, 순수·결정론(같은 입력 → 비트 동일), 입력 불변. 예외는 전부 `class FarFieldError(contracts.ContractError)`(ValueError 하위).
**단위·규약.** 선형 Rec.709, 휘도 **kcd/m²**(천정 휘도 ≈ 1–20, 조도 klux). 세계 up은 임의 축이라 **`up`을 명시 인자**로 받는다(PRD-03 스텁은 y 아래, INV-01은 z 위). 방위 = 위에서 내려다볼 때 **시계방향**: `dir = cos e (cos a·north + sin a·east) + sin e·up`, `east = north × up`. 팔면체 맵은 **세계 축에 고정**(up과 무관): `+z`=중심 `(.5,.5)`, `±x`=`(1|0,.5)`, `±y`=`(.5,1|0)`, `−z`=네 모서리. 텍셀 `[i,j]` 중심 = `u=(j+.5)/S, v=(i+.5)/S`; `p=(x,y)/(|x|+|y|+|z|)`, z<0이면 `p←((1−|p_y|)sgn p_x, (1−|p_x|)sgn p_y)`(`sgn 0=+1`), `uv=(p+1)/2`.

## 공개 API
상수: `FARFIELD_VERSION="farfield/v0"`, `DEFAULT_SIZE=256`, `MIN_SIZE=8`, `MAX_SIZE=4096`, `SUN_ANGULAR_RADIUS_DEG=0.2665`, `TURBIDITY_RANGE=(1.0,10.0)`, `DEFAULT_SUN_ELEV_DEG=45.0`, `DEFAULT_SUN_AZIM_DEG=180.0`.
```python
oct_encode(d (...,3)) -> uv (...,2) in [0,1]      # d 정규화, 영벡터·비유한 → FarFieldError
oct_decode(uv (...,2)) -> d (...,3) 단위
texel_centers(size) -> (S,S,2);  texel_dirs(size) -> (S,S,3);  texel_solid_angle(size) -> (S,S) f64  # size는 [8,4096] int 아니면 FarFieldError. dω=4/S²·l1³, l1=|x|+|y|+|z|, 합≈4π
sample(far: FarField|ndarray(S,S,3), d (...,3)) -> (...,3) f64   # 쌍선형. 범위 밖 텍셀은 변 거울 접힘: j<0→(S−1−i, −1−j), j≥S→(S−1−i, 2S−1−j), i도 같게(모서리는 두 번 적용). clamp 금지
hw_shape(coeffs(9), cos_theta, gamma) -> ndarray   # F=(1+A·e^{B/(cosθ+.01)})(C+D·e^{Eγ}+F·cos²γ+G·χ(H,γ)+I·√cosθ), χ(H,γ)=(1+cos²γ)/(1+H²−2H cosγ)^1.5, 순서 (A..I), cosθ는 [0,1]로 클램프
hw_params(turbidity, albedo, sun_elev_deg) -> (coeffs (3,9) f64, Z (3,) f64)  # RGB별 계수와 천정 스케일, 사본 반환, 범위 밖·비유한 → FarFieldError
@dataclass(frozen=True, kw_only=True) FarFieldSpec:
    up: 3-튜플; north: 3-튜플|None=None; sun_elev_deg=45.0 [-90,90]; sun_azim_deg=180.0 (mod 360); turbidity=2.5 [1,10]; albedo=0.3 [0,1]
    exposure=1.0 (>0); camera_height_m=1.6 (>0); horizon_blend_deg=1.5 (>0); ground_albedo_rgb: 3-튜플|None=None ([0,1]); size=256 [8,4096] int; include_sun=False (엄격 bool)
    # __post_init__: up 정규화, north는 up에 직교화·정규화(None이면 |up_k|가 가장 작은 축(동률은 낮은 k)을 직교화), north∥up·비유한·모양 오류·bool/문자열 → FarFieldError. up·north·ground_albedo_rgb는 float 튜플(해시 가능).
    sun_dir() -> (3,);  to_meta() -> dict(json allow_nan=False 가능; 키: version,up,north,sun_elev_deg,sun_azim_deg,turbidity,albedo,exposure,camera_height_m,horizon_blend_deg,ground_albedo_rgb,size,include_sun)
    @classmethod from_meta(m) -> spec   # version 불일치·모르는 키·누락 → FarFieldError, from_meta(to_meta(s)) == s
    build() -> contracts.FarField       # f32 (S,S,3) C-연속, 유한, ≥0
sky_radiance(spec, d) -> (...,3)  # HW형 하늘, 지평선 아래도 cosθ=0 클램프로 평가, 유한·≥0 (음수는 0으로 클램프), ×exposure
sun_irradiance(spec) -> (3,)      # 법선 조도 E_n(klux, ×exposure), 태양 고도 ≤0이면 정확히 0
sky_irradiance(spec) -> (3,)      # 상반구 ∫L cosθ dω (오차 ≤2%)
ground_lit(spec) -> (3,)          # alb/π·(sky_irradiance + sun_irradiance·sin(max(e,0))), alb = ground_albedo_rgb 또는 (albedo,)*3
environment_radiance(spec, d) -> (...,3)   # e=d·up ≥0이면 sky_radiance. e<0이면 w·sky_radiance(d_h)+(1−w)·ground_lit, d_h=d의 수평 성분(영이면 north), 깊이 δ=asin(−e)(도), w=exp(−δ/horizon_blend_deg). 태양 원반 제외
ground_distance(spec, d) -> (...,) # 지면 연장 평면까지 광선 거리 h/(−d·up), 아래 방향이 아니면 inf
up_world_from_obs(obs, up_cam) -> (3,)     # Rᵀ·up_cam 정규화 (Obs.R 은 세계→카메라)
spec_for_world(obs: Obs, up_cam, theta_in: Theta, **overrides) -> FarFieldSpec
```
`hw_params` 계수는 (T, albedo, elev)의 **결정론적 해석 근사**여도 된다. 원 Hosek–Wilkie 계수표는 레포에 없고 검증 불가이므로 임의 숫자를 "HW 데이터"라 주장하지 말고, 근사 출처(예: 폐형 Perez/Preetham 모형에 대한 오프라인 최소제곱 적합)를 모듈 docstring에 밝힐 것. 빌드 경로에 반복 솔버를 두지 않는 것을 권장(상수 내장).
`build()`: 텍셀 중심 방향에서 `environment_radiance`를 f64로 평가해 f32로. `include_sun=True`이고 `sun_elev_deg>0`이면 `ΔL=E_n·k(d)`를 더한다. k는 각거리 가우시안(σ≥1.5텍셀각), `Σ k·Ω_texel=1`로 정규화 → 해 방향 최대, 15° 밖 ≤1e-3·최대. 고도 ≤0이면 변화 없음.
`spec_for_world`: `up=up_world_from_obs`, `north` = 광축 `Rᵀe_z`의 수평 성분(광축∥up이면 `Rᵀe_x`의 수평 성분). `theta_in.sun_elev_deg/azim_deg`가 있으면 그대로(**방위 0 = 입력 카메라 전방, 90 = 카메라 오른쪽**), 없으면 기본값. overrides는 spec 필드(up·north 제외).

## 불변식
맵은 세계 축 고정이며 up이 바뀌어도 코덱은 같다. 해 자오면에 대해 `sky_radiance`는 거울 대칭, 세계 축을 함께 회전하면 값 불변. `environment_radiance`는 지평선(e=0)에서 방위마다 연속(±0.05°에서 상대차 <6%), 깊이 ≥40°에서 방위 무관 = `ground_lit`. `texel_solid_angle` 합 = 4π(1e-3). `build()`는 `sample`로 읽을 때 뒤통수(−z)와 네 변에서도 해석값과 일치(≤2%). `exposure`는 하늘·조도·지면에 선형.

## 수용 기준 (`tests/wld_17/`, CPU, 합 <60 s, `-m "not gpu"`)
코덱 왕복 1e-9·축 위치·입체각 합과 몬테카를로 밀도 / 쌍선형 표본(텍셀 중심 정확, 선형장 재현 <1/S, 접힘선 연속 — clamp는 실패) / `hw_shape` 폐형 오라클 1e-12·`sky_radiance`=HW×Z 1e-9·거울 대칭·천정 휘도 1–20·파랑비·태양 근방 광휘·고도 단조·태양 지평선 아래·탁도·알베도 / 지평선 연속·지면 연장·헤이즈 대역 단조·조도 구적 일치(3–5%)·직접광 비율 0.5–0.95·태양 원반 에너지 보존(1%) / 검증 거부(파라미터화 29종)·메타 왕복·결정론 / Obs·RD-02 `up_cam`·수평선 부호 일치 / `Strata` 왕복·루트 / torch 미로드.

## 하류 이음매
WLD-03: `spec_for_world(obs, g.up_cam, theta_in).build()` → `Strata.set_far_field`, `meta["far_field"]=spec.to_meta()`(재생성용; FarField 자체는 `hash_array`로 Merkle). WLD-18/RDR: 광선이 빗나가면 `sample(world.far_field, d_world)`, 지면 연장 평면은 `ground_distance`와 `ground_lit`(맵과 같은 색)을 쓴다. 입력의 하늘 섹터 lockstep(교사)은 이 맵을 초기값으로 덮어쓴다. 실내 장면은 만들지 않는다(호출자 책임). **합의 필요:** `Theta.sun_azim_deg`의 기준(여기서는 입력 카메라 전방 = 0, 시계방향)을 LNS 계열이 같게 쓸 것.
