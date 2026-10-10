# LNS-01 θ 스키마·사전분포 샘플러·수제 프리셋 20 v0

**목적.** FND-02 `Theta`(룩 프리셋 θ_look)의 물리 범위·어휘·`extra` 키를 못 박고, (a) 결정론 사전분포 샘플러(MRH 증강·InverseLook 초기값·테스트용), (b) 손으로 만든 프리셋 20(출처 i, 권리상 안전)을 제공한다. 학습 파라미터 0, 순수 수식·데이터.
**소유.** `buc/lens/theta.py`, `presets/*.yaml`(레포 루트, 파일 20개), 테스트 `tests/lns_01/`(동결). 의존: FND-02 `buc.core.contracts`(`Theta`, `ContractError`), numpy, PyYAML(컨테이너에 있음; pyproject 의존 추가는 FND 몫 — 요청 사항). **torch 금지**. `buc/lens/__init__.py`는 만들지 않는다(namespace, RDR 단위와 동일).

## 공개 API (`from buc.lens import theta`)
- 상수: `SCHEMA_VERSION="1.0.0"`; `LUT_NAMES=("none","neutral_709","teal_orange","bleach_bypass","kodak_warm","fuji_cool","mono_silver","faded_matte","neon_night","cross_process")`; `LENS_RX_NAMES=("spherical_clean","vintage_petzval","vintage_double_gauss","anamorphic_2x","soft_single_coat")`; `EXTRA_DEFAULTS={"squeeze":1.0,"blades":9,"cpl_angle_deg":0.0,"rolloff":0.5,"atmos_g":0.0}`; `HARD_BOUNDS: dict[str,(lo,hi)]`(아래 표); `PRESETS_DIR: Path`(= 이 파일 기준 `../../presets`, cwd 무관).
- `default_theta() -> Theta` = `Theta()` + 완전한 extra(EXTRA_DEFAULTS). `complete_extra(theta) -> Theta`: 새 객체, 모르는 extra 키는 보존, 아는 키는 정본 타입(squeeze·cpl_angle_deg·rolloff·atmos_g `float`, blades `int`)으로 강제하고 빠진 키는 기본값으로 채운다. 정수값 float(8.0)은 blades로 허용, 그 밖(8.5·bool·문자열·None)은 `ContractError`.
- `validate_theta(theta) -> None`: `theta.validate()` 후 추가 검사, 위반 시 `ContractError`(예외 타입은 이것뿐). 검사: HARD_BOUNDS(모든 수치 필드, `focus_m`은 유한), `lut∈LUT_NAMES`, `lens_rx∈LENS_RX_NAMES`, `fov_x_deg()∈[3,130]`, extra 아는 키 범위·타입(빠진 키는 기본값 취급), `(lens_rx=="anamorphic_2x") == (squeeze==2.0)`이고 squeeze∈{1.0,2.0}, 애너모픽이면 `aspect>=2.0`, `sun_elev_deg`와 `sun_azim_deg`는 둘 다 None이거나 둘 다 값.
- `theta_to_dict(theta) -> dict`: Theta 필드명 그대로 평평한 JSON 직렬화 가능 dict(`__contract__` 없음, extra는 중첩 dict, 값은 파이썬 기본형). `theta_from_dict(d) -> Theta`: 모르는 최상위 키·dict 아님 `ContractError`, `complete_extra` 적용 후 `validate_theta`. 왕복 `theta_from_dict(theta_to_dict(t)).content_hash()==complete_extra(t).content_hash()`.
- `exposure_time_s(theta) -> float` = `shutter_deg/(360*fps)`.
- `sample_theta(rng: np.random.Generator | int, *, fixed: Mapping[str,Any] | None = None) -> Theta`: int는 `np.random.default_rng(seed)`. 항상 `validate_theta` 통과 + 완전한 extra. `fixed`: Theta 필드명 또는 extra 키를 고정(나머지는 표대로 샘플). `fov_x_deg`·모르는 키는 `ContractError`; `focal_mm`만 고정하면 `sensor_width_mm=24.89`; `lens_rx` 고정 시 aspect는 그 렌즈의 표로 샘플(애너모픽이면 squeeze=2.0); 고정 값이 제약을 깨면 `ContractError`.
- `sample_thetas(n: int, seed: int = 0, *, fixed=None) -> list[Theta]`: i번째 = `sample_theta(np.random.default_rng(np.random.SeedSequence([seed, i])), fixed=fixed)`. 따라서 같은 인자 → 같은 `content_hash`, **접두 안정성** `sample_thetas(5,s)==sample_thetas(9,s)[:5]`(해시), 시드가 다르면 달라진다. n<0 `ValueError`, n=0 빈 리스트.
- `@dataclass(frozen=True) Preset(name, description, theta: Theta, provenance: str, tags: tuple[str,...])`. `list_presets() -> tuple[str,...]`(정렬), `load_preset(name_or_path) -> Preset`(`.yaml/.yml` 경로 또는 이름; 없으면 `KeyError`), `load_presets() -> dict[str,Preset]`.
- 프리셋 파일 형식: 최상위 `name`(=파일 stem), `description`, `provenance`, `tags`(선택), `theta`(= `theta_from_dict` 입력; extra는 일부만 적어도 됨). 필수 키 누락·`theta` 오류는 `ContractError`.

## 물리 범위 HARD_BOUNDS (닫힌 구간; 샘플·프리셋은 모두 이 안)
aspect [1,3] · focal_mm [4,600] · sensor_width_mm [4.8,70] · f_stop [0.95,32] · focus_m [0.3,1000] · shutter_deg [1,360] · fps [8,240] · nd_stops [0,12] · cpl [0,1] · ev [-5,5] · wb_kelvin [2000,12000] · wb_tint [-0.6,0.6] · grain·halation·vignette·rim [0,1] · atmos_beta [0,0.1] · sun_elev_deg [-20,90] · sun_azim_deg [0,360) · extra: blades 정수 [3,18] · cpl_angle_deg [0,180) · rolloff [0,1] · atmos_g [0,0.95].

## 사전분포 (필드 독립, 아래 결합 제외; 확률은 표 값 ± 샘플 오차)
- 센서 폭 mm 선택: 8.8:.05, 17.3:.15, 24.89:.40, 36.0:.30, 54.12:.10. **가로 화각** `fov_x_deg` 로그균등 [15°,95°] → `focal_mm = sensor/(2·tan(fov/2))`.
- f_stop 로그균등 [1.2,22] · focus_m 로그균등 [0.5,50] · wb_kelvin 로그균등 [2500,10000].
- shutter_deg 선택 {45:.05, 90:.15, 144:.05, 172.8:.05, 180:.60, 270:.10} · fps {24:.50, 25:.10, 30:.15, 48:.05, 60:.10, 120:.10}.
- 영 팽창(확률 p로 정확히 0, 아니면 균등): nd_stops p=.6 U[0.3,10] · cpl p=.7 U[0.2,1] · grain p=.3 U[0.05,0.7] · halation p=.5 U[0.05,0.6] · vignette p=.3 U[0.05,0.7] · rim p=.5 U[0.1,1] · atmos_beta p=.4 로그균등 [1e-4,2e-2].
- ev 정규(0,0.7)를 [-3,3]로 클립 · wb_tint 정규(0,0.15)를 [-0.5,0.5]로 클립.
- lut: none .35, neutral_709 .15, 나머지 8개 각 .0625. lens_rx: spherical_clean .6, 나머지 4개 각 .1.
- aspect: 애너모픽이 아니면 {1.3333:.10, 1.5:.05, 1.7778:.55, 1.85:.10, 2.0:.05, 2.39:.15}, 애너모픽이면 {2.0:.2, 2.39:.7, 2.76:.1}. squeeze는 lens_rx로 결정(2.0/1.0).
- 태양: 확률 .25로 `sun_elev_deg=sun_azim_deg=None`, 아니면 elev U[-5,75], azim U[0,360). `fixed`에 태양 값이 있으면 None 분기를 건너뛰고 고정 안 된 쪽만 샘플; `fixed["sun_elev_deg"]=None`이면 둘 다 None(azim을 값으로 고정하면 `ContractError`).
- extra: blades 정수 U{5..12} · cpl_angle_deg U[0,180) · rolloff U[0,1] · atmos_g U[0,0.9].

## 수제 프리셋 20 (`presets/<name>.yaml`, `provenance: handmade`)
이름과 서명 제약(그 밖은 자유, 서로 다른 θ여야 함): `neutral_reference`(focal 35, lut none, grain=halation=vignette=0, spherical_clean) · `cinema_scope_anamorphic`(anamorphic_2x, aspect 2.39±.01) · `portrait_85_shallow`(fov_x≤30°, f_stop≤2) · `wide_landscape_deep`(fov_x≥70°, f_stop≥8, focus≥5) · `golden_hour_warm`(sun_elev∈[2,15], lut kodak_warm) · `midday_harsh`(sun_elev≥55) · `blue_hour_dusk`(sun_elev∈[-6,0]) · `night_neon`(lut neon_night, f_stop≤2, grain≥.4) · `foggy_morning`(atmos_beta≥.005, sun_elev≤20) · `vintage_petzval_swirl`(vintage_petzval, vignette≥.4) · `super8_home_movie`(sensor 5.79±.01, aspect 1.3333±.01, fps=18, grain≥.6) · `imax_large_format`(sensor≥50, grain≤.1) · `telephoto_compression_sports`(focal≥200, fps≥48, shutter≤180) · `macro_closeup`(focus_m≤.6, f_stop≤5.6) · `high_speed_slowmo`(fps≥120, shutter≤180) · `long_exposure_nd_water`(nd≥4, shutter≥270, cpl>0) · `cpl_blue_sky_polarized`(cpl≥.8, sun_elev∈[30,70]) · `bleach_bypass_war`(lut bleach_bypass, grain≥.3) · `teal_orange_blockbuster`(lut teal_orange) · `noir_monochrome`(lut mono_silver, vignette≥.3, grain≥.3). **다양성**(20개 전체): 서로 다른 θ; lut ≥8종, lens_rx 5종 전부, 센서 폭 ≥5종, fps ≥5종, aspect(소수 2자리) ≥4종; 초점거리 최대/최소 ≥10; f_stop 최소 ≤1.6·최대 ≥11; focus_m 최소 ≤0.6·최대 ≥8; fov_x 최소 ≤12°·최대 ≥80°; 태양 없음 ≥1개·있음 ≥10개; grain>0 ≥8, halation>0 ≥3, vignette>0 ≥5, rim>0 ≥2, cpl>0 ≥2, nd>0 ≥2; 노출 시간 모두 [1/4000, 1/8] s. 프리셋 텍스트에 닫힌 외부 모델 이름("soul" 포함 문자열)을 쓰지 않는다(수제 출처 i).

## 불변식
1. 샘플러·로더는 전역 난수·시각·cwd를 쓰지 않는다. 입력 `Mapping`/`Generator` 외 상태 없음.
2. 샘플·프리셋의 값은 파이썬 `float/int/str/None`(numpy 스칼라 금지) — `json.dumps(t.to_dict())` 가능, `Theta.from_dict(t.to_dict())`가 해시 동일.
3. 경계: 프리셋 20과 모든 샘플이 `validate_theta`를 통과하고 HARD_BOUNDS 안.

## 수용 기준
`tests/lns_01/` 전부 통과(CPU, <60 s, `-m "not gpu"`, 동결 해시 `LNS-01.tests.json`): 어휘·범위·extra 검증과 오류 경로, 사전분포 4000표본의 한계 분포(허용 오차 6σ급)와 결합 제약, 결정론·접두 안정성·`fixed`, 프리셋 20 이름·서명·다양성·출처·cwd 무관 로드, 왕복.

## 하류 이음매
LNS-04는 `lut` 이름을 65³ LUT로, `grain/halation/vignette`를 화면 고정 효과로 매핑(`none`=항등). LNS-05/06은 `lens_rx` 이름(+`squeeze`, `blades`)을 처방/출사동으로, LNS-07은 `cpl`+`cpl_angle_deg`, LNS-02는 `focal/f_stop/focus/sensor`로 CoC를, LNS-03은 `exposure_time_s`를 쓴다. LNS-08·RD-20·DAT-03은 `sample_thetas`·`HARD_BOUNDS`·`theta_to/from_dict`를 쓴다. 필드 추가는 `Theta.extra`로만 하고 `SCHEMA_VERSION`을 올린다. 사용자 이미지의 θ_in은 공유 프리셋 라이브러리에 쌓지 않는다(프리셋은 수제·CAPTURE·ASSET만).
