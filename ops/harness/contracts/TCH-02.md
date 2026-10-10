# TCH-02 교사 샘플러 v0 — σ 스냅 표 · SDEdit · x̂0 · 뷰 배치 lockstep

**목적.** 모든 교사 호출(RD-07, WLD-07/08/19, DAT-02/15, TCH-05/06/11, MDL-07/11)의 **단일 입구**. Turbo는 학습된 8개 σ에서만 믿을 수 있으므로(PARADIGM CT3, "σ 점 스냅"), 임의 σ 요청은 표의 점으로 반올림하고 그 점들 사이만 오간다. 수식은 flow matching: `z_σ=(1-σ)·x0+σ·ε`, 속도 `v=ε-x0`, `x̂0=z-σ·v`.
**소유.** `buc/teacher/sample.py`, 테스트 `tests/tch_02/`(동결). 의존: TCH-01(`ZImageTeacher`, `TeacherSpecError`, `TeacherError`, `TeacherNumericError`), FND-03(`det.derive_seed`, `det.hash_parts`). import는 torch·numpy·위 두 모듈만 쓰고 diffusers 없이 되며 부작용이 없다. **소유 밖 요청: 없음.**

## 변환기 호출 규약 (diffusers ZImagePipeline과 동일; GPU 테스트가 실물로 확인)
`out = teacher.transformer(x_list, t, cap_list, return_dict=False)`; `x_list[i]`는 `[C,1,h,w]`(`teacher.dtype`), `t`는 `float32[n]` = **`1-σ`**, `cap_list[i]`는 `[L_i,D]`. `out`이 tuple이면 `out[0]`, 원소 `[C,1,h,w]`의 list. **속도 `v = -stack(out)`**(float32로 올려서 계산). 샘플러 쪽 산술은 전부 float32.

## 공개 API (`buc.teacher.sample`)
- `FLOW_SHIFT=3.0`, `N_STEPS=8`. `SigmaError(TeacherSpecError)`(ValueError 계열).
- `@dataclass(frozen=True) SigmaTable(points: tuple[float,...], name="custom")`: 점은 유한·**엄격 내림차순**·`(0,1]`, 1개 이상(아니면 `SigmaError`). `from_flow_shift(n=8, shift=3.0)`: `u_i=1-i/n (i=0..n-1)`, `σ_i=shift·u/(1+(shift-1)u)`; 기본은 `(1, 21/22, 0.9, 5/6, 0.75, 9/14, 0.5, 0.3)`. 끝점 0은 표에 없고 암묵(`σ_next=0`). `key->str` = `det.hash_parts("buc.tch02.table.v1", name, *[struct.pack("<d",p) for p in points]).hex()`. `snap(sigma)->Snap(index, sigma, requested)`: **|σ-점| 최소, 동률은 큰 σ(앞 인덱스)**; σ는 유한이고 `(0,1]`(아니면 `SigmaError`). `index_of(sigma, tol=1e-9)->int`: 표 밖이면 `SigmaError`. `window(lo, hi)->tuple[int,...]`: `lo<=σ<=hi`인 인덱스(내림차순), 비면 중앙값의 스냅 1개, `lo>hi`면 `SigmaError`. `schedule(start, steps)->tuple[(σ,σ_next),...]`, 마지막 σ_next=0.0; `start`가 범위 밖이거나 `steps<1` 또는 `start+steps>len`이면 `SigmaError`.
  **표는 잠정값이다.** 실제 가중치의 학습 σ와 같은지는 모르며 GPU 테스트가 확인한다. TCH-05(결정 곡선)가 표를 덮어쓸 수 있고, 그래서 표는 데이터(`TeacherSampler(table=)`)다.
- `DEFAULT_TABLE`, `snap_sigma(sigma, table=None)->Snap`.
- `add_noise(x0, sigma, noise)=(1-σ)x0+σ·noise`(σ∈[0,1]), `euler_step(z, x0, sigma, sigma_next)=(σn/σ)·z+(1-σn/σ)·x0`(0<σ≤1, 0≤σn<σ, 아니면 `SigmaError`; shape 불일치 `ValueError`; float32 산술·출력; **σn==0이면 x0와 비트 동일**). 속도 Euler `z+(σn-σ)v`와 같은 식이다(v=(z-x̂0)/σ).
- `@dataclass(frozen=True) StepInfo(step, sigma, sigma_next, view_lo, view_hi)`.
- `@dataclass(frozen=True) SampleResult`: `x0_hat`(마지막 forward의 원 예측, hook 전), `z`(마지막 상태, `float32[B,C,h,w]`), `z_start`(첫 forward 입력), `sigmas`(forward마다의 σ, tuple), `sigma_end`(끝 상태의 σ, 표 끝까지 가면 0.0), `start_index`, `requested_sigma`, `steps`, `calls`(변환기 호출 수), `table_key`. grad=False면 전부 `requires_grad=False`.
- `TeacherSampler(teacher, *, table=None, max_views_per_call=None)`: 속성 `teacher, table`.
  - `predict_x0(z, sigma, cond, *, grad=False, observer=None)->Tensor[B,C,h,w]`: **표의 점에 정확히 있는 σ**에서 forward 1회(청크 단위)로 x̂0. 표 밖이면 `SigmaError`(여기서는 조용히 스냅하지 않는다).
  - `sdedit(x0, cond, *, sigma, seeds, steps=2, level=0, k=0, x0_hook=None, observer=None, grad=False)->SampleResult`: `sigma`를 스냅 → `z_start=add_noise(x0, σ_s, ε)`, `ε_i=teacher.make_noise(seeds_i,1,8h,8w,level=level,k=k)[0]`(뷰 i의 노이즈는 배치 구성과 무관) → 표의 연속 `steps`점을 `denoise`.
  - `denoise(z, cond, *, start_index, steps, x0_hook=None, observer=None, grad=False)->SampleResult`: `z`는 이미 σ=`table[start_index]` 상태. 매 step: 한 번에 전 뷰 forward → `x̂0` → `x0_hook(x̂0, StepInfo)`가 있으면 `x0'`(같은 shape, 아니면 `ValueError`) → `z←euler_step(z,x0',σ,σn)`.
  - `generate(cond, *, seeds, height, width, steps=None, level=0, k=0, **kw)->SampleResult`: `z_start=ε`(σ=1), 표 전체(기본 8 NFE). `sdedit(sigma=1.0)`과 비트 동일.
- 입력: `x0/z`는 `float[B,C=spec.latent_channels,h,w]`, 유한, `spec.check_image_size(8h,8w)` 통과, B≥1. `cond`는 `Sequence[str]`(→`teacher.encode_text`) 또는 `Sequence[Tensor[L,D]]`(혼합은 `TypeError`), 길이 B 또는 1(브로드캐스트), 아니면 `ValueError`. `seeds`는 int(→`seeds[i]=det.derive_seed(seed,i)`) 또는 길이 B 시퀀스. `observer(StepInfo)`는 **변환기 호출 직전마다**(step 우선, 청크 순서) 호출; `max_views_per_call`이면 뷰를 연속 청크로 나눠 부르고 결과는 청크 크기와 무관하다. `x0_hook`은 step마다 전 배치에 1회.

## 불변식
1. **뷰 독립(lockstep)**: 같은 step의 뷰들은 한 호출에 묶이지만 뷰 i의 결과는 배치 구성·순서·청크와 무관(뷰를 섞으면 결과도 같은 순서로 섞인다; order_purity).
2. 같은 입력 → 비트 동일. 시드가 다르면 다르다.
3. 완전 디노이저(변환기가 `v=(z-x*)/σ`)이면 어떤 시작 σ·step 수에서도 `x0_hat==x*`; 표 끝까지 가면 `z==x0'` 비트 동일.
4. 검증(σ, 모양, cond, seeds, hook)은 **변환기를 부르기 전에** 실패한다. 변환기 출력이 비유한이면 `TeacherNumericError`, 개수·모양 불일치는 `TeacherError`.
5. grad=False는 `no_grad`; grad=True는 `x0`/`z`로 역전파가 흐르고(MDL-07), 교사 파라미터는 동결 그대로.

## 수용 기준
`tests/tch_02/` 전부 통과(CPU, 가짜 변환기, <60 s): 표 수치·스냅 규칙·창·키, add/euler 수식, 변환기 규약(`t=1-σ`, 부호, dtype), 해석적 오라클(완전·아핀 디노이저) 경로 대조, 뷰 독립·순서·청크, 시드·level 의미, hook·observer, 오류 우선 검증, grad 흐름, bf16 허용 오차. `@pytest.mark.gpu`(실행하지 못했음, 5090 필요): 표가 스냅샷 스케줄러(shift 3.0)와 5e-3 이내, SDEdit 왕복 PSNR 하한, 실물 lockstep vs 단일 뷰 허용 오차.

## 하류 이음매
TCH-03 탭 훅은 `observer`로 (step, σ, 뷰 범위) 문맥을 받는다. TCH-04는 `SigmaTable.key`·`StepInfo`를 키 재료로 쓴다. WLD-07의 `γ·Enc(render)+(1-γ)x̂0`는 `x0_hook`, Ring-R의 띠 덮어쓰기·재노이즈는 `x0_hook`과 `add_noise`(k=step)로 한다. TCH-11은 `generate`, DAT-02는 `sdedit`(σ∈스냅[0.2,0.7], 2–3 step), RD-07은 `predict_x0`/`sdedit(steps=1)`, MDL-07은 `grad=True`.
