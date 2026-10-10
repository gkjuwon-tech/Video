# DAT-02 Realify + 탭 동시 추출 패스 — SDEdit 2–3 NFE · 결정 곡선 대역 마스크

**목적.** 합성 렌더(DAT-01)를 엔진 교사로 SDEdit하는 **한 번의 패스**에서 (i) 기하 GT가 결정 곡선 유효 대역 안에서 보존된 사진풍 이미지 x′와 (ii) 그 패스의 교사 탭(TCH-03, MRH 학습 입력)을 동시에 얻는다(PARADIGM §6.3: `t∈스냅[0.2,0.7]`, 2–3 NFE, `라벨 유효 대역={o: D(o,t)≤0.1}`, `저대역 불일치>τ → 폐기`). 추가 NFE 없음.
**소유.** `buc/data/realify.py`, 테스트 `tests/dat_02/`(동결). 의존: TCH-02 `TeacherSampler`·`SigmaTable`·`add_noise`, TCH-03 `TapBus`·`TapSet`·`TapRing`, FND-04 `buc.gov.prov`, FND-03 `det.derive_seed/hash_parts`, DAT-01 `SynthSample`(덕 타이핑), DAT-05 `ShardWriter`·`npy_encode`. torch·numpy·pillow만. CUDA·diffusers 없이 import, import 부작용 없음.
**잠정.** `provisional_curve`와 `DEFAULT_TAU`는 TCH-05(X1 결정 곡선) 전의 자리값이다. TCH-05는 `DecisionCurve`를 만들어 넘기면 된다.

## 옥타브 (영상 공간, 정본 정의)
`fftfreq(H)`·`fftfreq(W)` 빈의 `r=max(|fy|,|fx|)`(cycles/px). 옥타브 o(0=가장 고주파)는 `r∈(0.5·2^-(o+1), 0.5·2^-o]`, 즉 `o=floor(log2(0.5/r))`; `o≥n_oct-1`과 DC(r=0)는 `n_oct-1`. 하드 마스크라 분해는 분할: `Σ_o Π_o(x)=x`, 직교(에너지 합 보존), 실수 출력.
- `max_octaves(h,w)->int = floor(log2(min(h,w)))`
- `octave_map(h,w,n_oct)->LongTensor[h,w]`(FFT 배치 순서), `octave_split(x[...,H,W], n_oct)->Tensor[n_oct,...,H,W]`, `band_project(x, valid: Sequence[bool])->Tensor`(n_oct=len(valid), 유효 옥타브 합, 빈 valid `RealifyError`). float64로 계산, 입력 dtype으로 반환. `n_oct<1` `RealifyError`.

## 결정 곡선
`@dataclass(frozen=True) DecisionCurve(sigmas: tuple[float,...], D: tuple[tuple[float,...],...], table_key: str|None=None, name="custom")`: σ 유한·엄격 내림차순·(0,1], D는 `[n_sigma][n_oct]`(n_oct≥1, 행 길이 동일), 값 유한 ∈[0,1]. 위반 `RealifyError`. 리스트·ndarray 입력은 float 튜플로 정규화.
- `n_octaves`, `at(sigma, tol=1e-9)->tuple[float,...]`(곡선에 없는 σ `RealifyError`), `band_mask(sigma, thresh=D_THRESH)->tuple[bool,...]` = `D(o,σ) ≤ thresh`(**경계 포함**).
- `key -> str` = `det.hash_parts(CURVE_TAG, name, table_key or "", n_sigma, n_oct, *[struct.pack("<d",v) for v in sigmas + D 행 우선]).hex()`.
- `provisional_curve(table, n_oct)`: σ마다 `r=((1-σ)/σ)·2^(o-2)`, `D=1/(1+r²)`; `name="provisional-v0"`, `table_key=table.key`.

## 입력·결과
- 상수: `REALIFY_TAG="buc.dat02.realify.v1"`, `CURVE_TAG="buc.dat02.curve.v1"`, `SIGMA_RANGE=(0.2,0.7)`, `ALLOWED_STEPS=(2,3)`, `D_THRESH=0.1`, `DEFAULT_TAU=0.05`. 예외 `RealifyError(ValueError)`.
- `@dataclass(frozen=True, kw_only=True) RealifyItem(key: str, image: ndarray[H,W,3] f32 표시값 [0,1], prov, caption: str, labels: Mapping[str, ndarray]={})`. key는 DAT-05 key 정규식. label 이름 `^[A-Za-z0-9_]+$`, 예약어 `rgb`·`meta`·`tap<숫자>` 금지.
- `item_from_synth(sample, *, key, caption) -> RealifyItem`: `image=clip(sample.rgb_display(),0,1)` f32, `labels={gbuf,depth,sem,inst}` 복사, `prov=as_record(sample.prov)`.
- `@dataclass(frozen=True) RealifyResult`: `key, image`(x′ `[H,W,3]` f32 [0,1]), `taps: TapSet`(층별 int8 `[1,k,gh,gw]`, `key=item.key`, `prov=result.prov`, `ephemeral=False`), `band_valid: tuple[bool]`, `lowband_err: float`, `keep: bool`, `reject_reason: None|"no_band"|"lowband"`, `sigma`(스냅 시작 σ), `tap_sigma`, `steps`, `seed`(뷰 노이즈 시드), `labels`(복사), `prov`(= `derive("realify",[item.prov])`), `curve_key`, `table_key`, `ring_seq: int|None`. `shard_fields()->dict[str,bytes]`: `rgb.png`(uint8 RGB = `round(image·255)`), `tap<l>.npy`(int8 `[k,gh,gw]`, `npy_encode`), `<label>.npy`, `meta.json`(정렬 키·무공백 JSON: `v,key,sigma,tap_sigma,steps,seed,band_valid,n_oct,lowband_err,tau,d_thresh,curve_key,table_key,basis_fp(hex),grid,tap_layers,keep`).

## 패스
`realify(sampler, bus, items, *, curve, sigma, seed, steps=2, tau=DEFAULT_TAU, d_thresh=D_THRESH, token_offset=0, tap_step=-1, ring=None) -> list[RealifyResult]`(items 순서).
1. **검증은 교사 호출(encode·forward) 전에 전부**: items 비어 있지 않음·타입, key 중복, 이미지 모양 동일·유한·[0,1]·`spec.check_image_size`; 방화벽 — 각 prov가 `why_not_train==()` ∧ `bits==0` ∧ `ns=="train_ok"`가 아니면 `FirewallError`; `σ_s=table.snap(sigma).sigma ∈ SIGMA_RANGE`; `steps∈ALLOWED_STEPS`(요청 NFE; **유효 n=min(steps, len(table)-start)**, 표 끝에서 잘림 — 기본 표에서 σ_s=0.3이면 n=1); `tap_step∈[-n,n)` 정수; `curve.at(σ_s)` 존재, `curve.table_key`가 None이 아니면 `sampler.table.key`와 같음, `curve.n_octaves ≤ max_octaves(H,W)`; `tau≥0` 유한, `d_thresh∈[0,1]`; `bus.attached`; ring 기저 지문 = `bus.basis.fingerprint`(아니면 `TapBasisError`); seed ∈[0,2^64). 값 위반 `RealifyError`, 타입 `TypeError`.
2. 청크 = `sampler.max_views_per_call or len(items)`. 청크마다: `x_in=2·image-1`(`[B,3,H,W]`) → `teacher.encode_image` → `sampler.sdedit(x0, [caption…], sigma=sigma, seeds=[det.derive_seed(seed, key)…], steps=n)`와 **같은 궤적**(허용 1e-6), 변환기 호출은 정확히 n회. 그중 `tap_step` 번째 forward 하나만 `bus.capture(derive("realify",청크 prov), grid=(H//16,W//16), token_offset)` 안에서 돈다. x′ = `clamp((decode_latent(마지막 x̂0)+1)/2, 0, 1)`. `tap_sigma=table[start+(tap_step mod n)]`, `result.steps=n`.
3. `band_valid=curve.band_mask(σ_s, d_thresh)`. `lowband_err = RMS(band_project(x′−image, band_valid))`(3·H·W 원소, float64). 유효 옥타브가 없으면 err=0.0, `"no_band"`. `keep = any(valid) ∧ err ≤ tau`, 아니면 `"lowband"`.
4. ring이 있으면 keep인 결과의 TapSet만 push(`ring_seq`).
`write_results(writer, results)->int`: keep인 것만 `writer.write(key, shard_fields(), prov)`, 쓴 개수 반환.

## 불변식
뷰 독립(결과는 배치 구성·순서·청크와 무관: 이미지 1e-5, 탭 ±1 LSB) · 같은 입력 → 비트 동일 · 시드·σ가 다르면 다르다 · 입력 배열 불변 · 거부된 호출은 교사·링을 건드리지 않는다.

## 수용 기준
`tests/dat_02/` 전부 통과(CPU, 가짜 교사, <60 s): 옥타브 분할·직교·코사인 위치, 곡선 검증·키 공식·잠정 곡선 수치, sdedit 오라클 대조·NFE 수·탭 오라클(탭 step의 상태에서 직접 캡처), 대역 마스크 경계 포함, 결정론적 저대역 판정(무노이즈 체커 교사: 옥타브 3 이하만 바뀜), 방화벽 카나리아(USER·EXTGEN:test·HOLDOUT·COMP), 사전 검증 무호출, 링·샤드 왕복, DAT-01 렌더 연동. `@pytest.mark.gpu`: 실물 교사 512² 1장 — **미검증**(실제 토큰 오프셋, 실제 결정 곡선, τ 값).

## 하류 이음매
MDL-03: `TapSet.features(basis)` + `tap_sigma`(FiLM t) + labels, 손실은 `band_project(pred−GT, band_valid)`(라벨 해상도가 다르면 같은 옥타브 정의로 재계산). DAT-03(θ-Realify)는 `RealifyResult.image`를 입력으로. DAT-05 샤드 필드 이름은 위 `shard_fields`가 정본. TCH-05는 `DecisionCurve(table_key=table.key)`를 공급.
