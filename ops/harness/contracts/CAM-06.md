# CAM-06 IMU 파이프 v0 — GPMF · 폰/공개 IMU 파서 · 정렬 · 증강 ×5.5

**목적.** CamFM-Hand(MDL-17, 손떨림 8M)의 학습 입력: 센서 파일(GoPro GPMF, 폰·TUM VI·TII 계열 CSV)을 200 Hz 균일 `ImuClip`으로 바꾸고, 기기 간 시계를 맞추고, 리그 전달함수 증강으로 ×5.5로 늘린다(CT21: 8M 모델은 증강 ×5.5 필수). 학습 0, 순수 수식. 데이터 원천은 `docs/DATA-PLAN.md`의 공개 데이터(TUM VI·TII Racing, CC BY 4.0; R1 허용, R2: 정답이 IMU 센서). CC-BY 표기(`ATTRIBUTION.md`)는 데이터 단위의 몫이고, 이 모듈은 `meta`(`dataset`,`license` 등)를 파서→클립→증강 변형까지 그대로 전달한다.
**소유.** `buc/camera/imu.py`, 테스트 `tests/cam_06/`(동결). 의존: numpy, scipy(`signal`,`interpolate`), 선택적으로 FND-06 `buc.core.geom`. **torch 금지**, import 부작용·전역 RNG 없음(`default_rng(SeedSequence(..))`만). `buc/camera/__init__.py` 만들지 않는다. 공개 이름은 `__all__`에. 잘못된 값 `ValueError`, 잘못된 형식 `TypeError`. 입력 불변.
**규약.** SI: rad/s, m/s²(중력 포함), s, float64. 열 순서는 센서 원본(x,y,z) 그대로(파서는 축 치환 안 함); 센서→카메라(OpenCV)는 하류가 `rotate_frame`(`x_cam = R x_sensor`)으로. 상수: `RATE_HZ=200.0`, `AUG_FACTOR=5.5`, `G_STD`, `MAX_GAP_S=0.1`, `WINDOW_LEN=400`(2 s), `GRAVITY_LP_HZ=0.5`, `RIG_KINDS=("lowpass","highpass")`.

## 자료형 (frozen dataclass, 배열은 복사본·읽기전용·float64, `meta`는 깊은 복사된 JSON 가능 dict, 아니면 `TypeError`)
- `Series(t (N,), v (N,3))`: N≥2, 유한, `t` 엄격 증가 아니면 `ValueError`.
- `RawImu(gyro: Series|None, accel: Series|None, source: str, device: str="", meta: dict={})`. 센서별 자기 시각. 둘 다 None이면 `ValueError`.
- `ImuClip(t0, rate_hz, gyro (N,3), accel (N,3), source="", device="", meta=None)`: N≥1, 모양 일치·유한·`rate_hz>0`. `.n`, `.duration_s = n/rate_hz`, `.t = t0 + arange(n)/rate_hz`, `.as_array() -> (N,6)`(자이로|가속도), `.slice(i0, i1)`(t0 갱신, 빈 구간 `ValueError`).

## 파서
- `parse_gpmf(data: bytes, *, rate_hz=None, device="gopro") -> RawImu` (`source="gpmf"`). KLV(키 4B·형식 1B·구조크기 1B·반복 2B 빅엔디언, 4B 패딩), `DEVC`⊃`STRM`⊃{`STMP`(J, µs)·`SCAL`(1개 또는 축별 3개)·`GYRO`/`ACCL`(3축 s/S/l/L/f)}. 값 = 정수/SCAL(SCAL 없으면 1). 다른 스트림(GPS5 등)은 건너뛴다. 시각: 블록 k의 표본 j = `STMP_k/1e6 + j/rate`(블록마다 STMP에 못박음), `rate` = (마지막 블록 앞 표본 수 합)/((STMP_마지막−STMP_처음)/1e6), 블록이 하나뿐이면 `rate_hz` 필수. `rate_hz`를 주면 항상 그 값을 쓴다. STMP가 하나도 없으면 `rate_hz` 필수, 시각 `i/rate`(0부터). STMP가 일부만 있거나 시각이 증가하지 않으면 `ValueError`. 자이로·가속도 속도가 달라도(400/200 Hz) 된다. 선언한 페이로드가 파일 끝을 넘는 KLV는 그 항목(DEVC면 통째로)과 이후를 버리고 `meta["truncated"]=True`. `meta`: `truncated`, `gyro_rate_hz`, `accel_rate_hz`(가속도가 있을 때). GYRO가 없거나 bytes 계열이 아니면 각각 `ValueError`/`TypeError`.
- `parse_phone_csv(source, *, sensor="both", fmt="auto", time_unit=None, gyro_unit="rad/s", accel_unit="m/s2", columns=None, device="phone") -> RawImu`. `source`: `Path`, 줄바꿈 없는 `str`(경로), 줄바꿈 있는 `str`(내용), `bytes`, 텍스트 스트림. 첫 비어있지 않은 줄이 헤더(앞의 `#` 제거), 이후 `#`로 시작하는 줄과 빈 줄은 무시, 구분자는 쉼표. `fmt="auto"`: 헤더가 `#timestamp [ns]`로 시작하면 `"tumvi"`(열 순서 t(ns), w_xyz(rad/s), a_xyz(m/s²); `source="tumvi"`, 시각은 정수 ns/1e9), 아니면 `"generic"`(`source="csv"`): 열 `t|time|timestamp`(대소문자 무시), `gx,gy,gz`, `ax,ay,az`; `columns={"t":..,"gx":..,...}`로 이름을 바꿀 수 있다. `time_unit` ∈ {s,ms,us,ns}(generic 기본 s). `gyro_unit` ∈ {rad/s, deg/s}, `accel_unit` ∈ {m/s2, g}(g는 ×`G_STD`). `sensor`: `"both"`(자이로 필수, 가속도 열이 없으면 `accel=None`), `"gyro"`, `"accel"`(가속도만, `gyro=None`; 센서별 파일용). 열 없음·숫자 아님·NaN/inf·시각 비증가·행 2개 미만·모르는 단위/`fmt`/`sensor`는 `ValueError`.
- `merge_raw(gyro_raw, accel_raw) -> RawImu`: 자이로는 앞의 것, 가속도는 뒤의 것, `source`·`device`·`meta`는 앞의 것. 어느 쪽이 비면 `ValueError`.

## 균일화
- `split_segments(raw, max_gap_s=MAX_GAP_S) -> list[RawImu]`: 두 센서 중 하나라도 간격이 `max_gap_s`를 넘으면 그 틈의 **중점**에서 둘 다 자른다. 조각의 어느 센서든 표본 2개 미만이면 버린다. `source`·`device`·`meta` 유지.
- `resample_uniform(raw, rate_hz=RATE_HZ, *, max_gap_s=MAX_GAP_S, t0=None) -> ImuClip`: 격자 시작 = 두 센서가 겹치는 구간의 시작(`t0`를 주면 그 값, 겹침 밖이면 `ValueError`), `n = floor((끝−시작)·rate+1e-9)+1`. 틈 > `max_gap_s`, 가속도/자이로 없음, 겹침 없음, `rate_hz≤0`은 `ValueError`. 보간은 3차(스플라인급): 원 표본 속도의 10% 이하 사인파 오차 ≤ 0.2% 진폭. 원 속도 > 목표 속도면 먼저 영위상 저역 통과(앨리어싱 방지: 150 Hz 성분은 200 Hz로 내릴 때 ≥26 dB 감쇠, 10 Hz는 보존).

## 정렬 (기기 간 시계)
- `estimate_offset(ref, other, *, max_lag_s=2.0, min_overlap_s=5.0) -> AlignEstimate(offset_s, score, lag_samples)`: 회전·장착과 무관한 **자이로 노름** `|ω|`의 평균 제거 피어슨 상관을 정수 지연마다 구하고 최대점을 포물선으로 부표본 보정. 의미: `t_ref = t_other + offset_s`(other가 ref보다 `offset_s`만큼 늦은 시계). `|offset_s| ≤ max_lag_s`(+1/rate), 두 클립의 `t0` 차이는 임의. `offset_s = (ref.t0−other.t0) + (lag_samples+δ)/rate`, |δ|≤0.5, `lag_samples`는 `other[i]≈ref[i+lag]`를 만드는 정수 지연. `score`∈[−1,1]. 속도가 다르거나 `max_lag_s≤0`이거나 겹침이 `min_overlap_s` 미만이면 `ValueError`.
- `shift_clip(clip, offset_s) -> ImuClip`: `t0 += offset_s`.
- `align_pair(ref, other, *, max_lag_s=2.0) -> (ref_cut, other_aligned, est)`: `other`를 `t_other = t_ref − offset_s`에서 3차 보간해 ref 격자에 올리고, 둘 다 겹치는 구간으로 같은 `t0`·`n`으로 자른다(ref는 슬라이스만, 값 불변; other는 장착 회전하지 않음).

## 증강 (×5.5, 결정론)
- `rotate_frame(clip, R) -> ImuClip`: 자이로·가속도 모두 `x @ R.T`. R이 회전이 아니면(det≠+1, 직교 오차>1e-6, 모양) `ValueError`.
- `gravity_estimate(clip, fc_hz=GRAVITY_LP_HZ) -> (N,3)`: 가속도의 영위상 2차 저역(양끝 4 s 안쪽에서 "상수+4 Hz 떨림"의 상수를 0.01 m/s² 이내로 복원). 이하 "동적 가속도" = 가속도 − 중력 추정.
- `warped_length(n, scale) = floor((n−1)/scale+1e-9)+1`. `time_warp(clip, scale)`: 운동을 `scale`배 빠르게(`scale>0` 유한). 자이로 `ω'(t)=s·ω(st)`, 동적 가속도 `s²` 배, 중력 성분 유지, 3차 보간, 길이 `warped_length`. `scale=1`이면 항등(1e-9).
- `rig_filter(clip, *, kind, fn_hz, zeta, gain=1.0)`: 리그 전달함수. 2차계 `lowpass: ωn²/(s²+2ζωn s+ωn²)`, `highpass: s²/(…)`(ωn=2π·fn). 쌍선형 변환, **인과적**(자이로 경로), 초기 상태는 첫 표본의 직류 정상상태(상수 입력 → 시작 과도 없음; 고역 통과의 상수 입력 → 0). 자이로 → `gain·H(ω)`, 가속도 → `g + gain·H(a−g)`. 해석 이득 대비 ≤3%(정상상태). 조건: `0<fn_hz<0.45·rate`, `zeta>0`, `gain>0`, kind∈`RIG_KINDS`, 아니면 `ValueError`.
- `perturb_sensor(clip, *, seed, gyro_sigma=0.0, accel_sigma=0.0, gyro_bias=(0,0,0), accel_bias=(0,0,0))`: `x + bias + σ·N(0,1)`. RNG = `default_rng(SeedSequence([seed, 0xCA06]))`에서 자이로 (N,3) → 가속도 (N,3) 순. σ<0, 바이어스가 (3,) 유한이 아니면 `ValueError`.
- `AugmentConfig`(frozen, kw 전용, 기본값): `rot_deg_max=30.0`, `time_scale=(0.85,1.18)`, `rig_prob=0.6`, `rig_kinds=("lowpass","highpass")`, `rig_fn_hz=(1.0,20.0)`, `rig_zeta=(0.3,1.2)`, `rig_gain=(0.5,1.5)`, `gyro_sigma=(0.0,0.02)`, `accel_sigma=(0.0,0.1)`, `gyro_bias_max=0.01`, `accel_bias_max=0.05`. 범위는 lo≤hi·유한, `time_scale`·`rig_fn_hz`·`rig_zeta`·`rig_gain` 하한 >0, `rig_fn_hz` 상한 <0.45·`RATE_HZ`, `rig_prob`∈[0,1], 음수 σ·한계 금지 — 위반은 `ValueError`.
- `sample_augmentation(cfg=None, *, seed=0) -> dict` (JSON 가능, numpy 형 금지): `{"rotvec": [3 floats, |rotvec|≤rot_deg_max], "time_scale", "rig": None | {"kind","fn_hz","zeta","gain"}, "noise": {"seed": int≥0, "gyro_sigma","accel_sigma","gyro_bias":[3],"accel_bias":[3]}, "crop": None}`. 값은 `cfg` 범위 안(축소된 범위면 그 값 그대로), `rig`는 확률 `rig_prob`. 같은 `(cfg, seed)` → 같은 dict.
- `apply_augmentation(clip, aug) -> ImuClip`: 순서 **회전(rotvec→Rodrigues) → time_warp → rig_filter(있으면) → perturb_sensor → crop(`[start, length]`, 있으면)**. `aug`의 키가 위 스키마와 다르거나 값이 무효(rotvec≠3, crop이 범위 밖/길이 0, 하위 호출의 조건 위반)면 `ValueError`. 결과 `meta` = 입력 `meta` + `"aug": aug 사본`. 기록만으로 비트 재현된다. `augment_clip(clip, cfg=None, *, seed=0)` = `apply_augmentation(clip, sample_augmentation(cfg, seed=seed))`.
- `augment_dataset(clips, *, factor=AUG_FACTOR, cfg=None, seed=0) -> list[ImuClip]`: `factor`는 유한 ≥1(아니면 `ValueError`). 출력 = 원본 전부(입력 순서, 값 그대로) 다음에 출처 0,1,2… 순, 출처 안에서 변형 1,2,… 순. 모든 출력의 `meta`에 `origin`(입력 인덱스), `variant`(원본 0), `aug`(원본 None) 추가, 나머지 `meta`·`source`·`device` 유지. 출처 i의 변형은 `n_i`표본에 대해 길이 합이 **정확히** `floor((factor−1)·n_i+0.5)`: 변형을 하나씩 만들어(`SeedSequence([seed, i, v, 0xCA06])`로 `sample_augmentation`, 다른 클립과 무관) 남은 예산 이하면 통째로, 넘으면 예산 길이만큼 `crop`한 것을 마지막으로 낸다. 같은 `(clips, factor, cfg, seed)` → 비트 동일, 전역 RNG 불변.
- `make_windows(clip, *, length=WINDOW_LEN, hop=None) -> float32 (M, length, 6)`: 열 = 자이로 3 | 가속도 3, `hop` 기본 `length//2`, `M=(n−length)//hop+1`(n<length면 0). `length<1`·`hop<1`은 `ValueError`.

## 불변식
입력 불변 · 같은 입력·시드 → 비트 동일 · 회전 증강은 노름 보존 · `time_warp(1.0)`·항등 `aug`는 항등 · 증강 결과는 `meta["aug"]`만으로 재현 · 원본이 먼저, 변형 길이 합이 예산과 정확히 일치.

## 수용 기준
`tests/cam_06/` 전부 통과(CPU, <60 s, 동결 해시 `CAM-06.tests.json`, GPU 테스트 없음): 합성 GPMF 바이너리(STMP 못박기, 199.8 Hz, 지터, 400/200 Hz, 축별 SCAL, int32, 잘림) · TUM VI/일반 CSV(단위, 센서별 병합, 오류) · 리샘플 정확도·앨리어싱·틈 · 알려진 오프셋 회복(≤1.5 ms, 노이즈 ≤3 ms) · 증강 해석 대조(time_warp s·s², 리그 이득 3%, 인과성, 선형성) · 센서 열화 통계 · ×5.5 예산 정확 일치·재현·결정론 · 윈도 · torch 미로드.

## 하류 이음매
MDL-17(CamFM-Hand): `augment_dataset(clips)` 출력에 `make_windows(clip)`을 적용해 (M,400,6) float32를 만든다(μ-law 256-bin 양자화·20 ms 패치는 MDL-17 몫). 학습/검증 분할은 `meta["origin"]` 단위로 해야 한다(변형이 같은 원본을 공유 → 누수 방지). CAM-01 리그 시뮬레이터의 14종 전달함수와 맞출 때는 `rig_filter`의 `(kind, fn_hz, zeta)`로 근사한다. 데이터 단위(TUM VI/TII 다운로드·`ATTRIBUTION.md`)는 `parse_phone_csv(..., device=..)` 결과를 `meta`로 꼬리표 붙여 넘긴다. 알려진 한계: GPMF `ORIN` 축 방향·`MTRX` 보정, 클록 드리프트(ppm), 평행이동 적분(OpenVINS 몫)은 다루지 않는다. 의존 추가 요청 없음(scipy는 이미 pyproject에 있음).
