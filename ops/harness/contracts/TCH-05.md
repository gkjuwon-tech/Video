# TCH-05 결정 곡선 X1-slim · 가난선 프로브 v0 — 옥타브별 "누가 정했나"와 σ 채택 규칙

**목적.** E2 (a): SDEdit 창이 어느 옥타브를 입력에서 보존하고 어느 옥타브를 교사가 새로 정하는지(결정 곡선 D(o,σ))를 σ 표 점마다 잰다. 그 결과로 DAT-02의 σ 스냅 점, 라벨 유효 대역 마스크(`{o : D ≤ 0.1}`), 레벨 간격(R4: 결정 폭 >1.5옥타브면 2옥타브 간격 또는 SR-LoRA)이 정해진다. E2 (b)의 가난선(DR 비 ≤1.1 ∧ 2AFC ≤0.55)은 **외부가 잰 값을 받아** 경계로 바꾼다.
**소유.** `buc/teacher/decision_curve.py`, 테스트 `tests/tch_05/`(동결). 의존: TCH-02(`TeacherSampler.sdedit/.table`, `SigmaTable`), FND-06(`buc.core.pyramid.laplacian_pyramid`), numpy, torch. diffusers·transformers·scipy.optimize를 import하지 않고 import 부작용 0(전역 RNG 불변). **소유 밖 요청: 없음.**

## 정의 (이 숫자들이 해석을 정한다)
- **옥타브 o** = `laplacian_pyramid(img_float64, levels)[o]`, o=0이 최미세, 클수록 굵다(피치 2^o px). 잔차는 제외. 입력 `(H,W)` 또는 `(H,W,C)`, `min(H,W) ≥ 2^(levels+1)`(아니면 `DecisionCurveError`).
- **D(o)** = Σ‖B_o(x)−B_o(y)‖² / Σ(‖B_o(x)‖²+‖B_o(y)‖²) ∈ [0,2]. 합은 이미지·시드 전체에 걸친 **풀링**(비율의 합이 아니라 합의 비율). y=x → 0, y=−x → 2, y=g·x → (1−g)²/(1+g²), y=0 → 1, 독립 동에너지 → ≈1. 스케일·x↔y 대칭 불변. 분모 ≤1e-18이면 **NaN**(조용히 0으로 바꾸지 않는다).
- **spread(o)** = Σ_s‖B_o(y_s)−B̄_o‖² / Σ_s‖B_o(y_s)‖² ∈ [0,1] (시드 간 분산 비율, σ̂²의 원천). 독립 시드 n개 ≈ 1−1/n. 분모 ≤1e-18이면 NaN.
- **가장자리 `edge_octave(d, tau)->float`**: 조각 선형 보간한 d를 **굵은 쪽(o=L−1)부터** 훑어 처음 `d>tau`로 넘어가는 실수 위치. 굵은 쪽 끝이 이미 `>tau`면 `+inf`, 전부 `≤tau`면 `-inf`. NaN·1차원 아님·tau≤0은 `DecisionCurveError`. 예: d=[.9,.7,.5,.2,.05] → tau=.1: 3+2/3, tau=.5: 2.0.
- **폭 `decision_width(d, lo=0.1, hi=0.5)`** = edge(lo) − edge(hi) ≥ 0, 어느 한쪽이 무한이면 NaN. `lo<hi` 아니면 오류.
- **`valid_mask(d, tau=0.1)->bool[L]`**: 굵은 쪽에서 이어진 접두만(`d≤tau`), 중간에 끊기면 그 미세 쪽은 전부 False. 한 칸 = `ceil(edge)` 이상. DAT-02의 대역 마스크.

## 공개 API
- 상수 `VALID_TAU=0.1, DECIDED_TAU=0.5, WIDTH_MAX=1.5, SEP_TARGET=1.0, SEP_TOL=0.5, MONO_TOL=0.5, LEVELS=4`. `DecisionCurveError(ValueError)`. `__all__`에 아래 이름 전부.
- `band_curve(inputs: Sequence[ndarray], outputs: Sequence[Sequence[ndarray]], levels=4) -> BandCurve(levels, decision[L], spread[L], input_energy[L], n_images, n_seeds)`: N장 × 시드 S≥2, 모양 동일·유한·정수 levels(bool 불가). 내부 float64, 입력 불변, 비트 결정론.
- `@dataclass(frozen=True, eq=False) DecisionCurve(indices, sigmas, decision[n,L], spread[n,L], input_energy[L], levels, steps, n_images, n_seeds, table_key)`: indices 엄격 오름차순(창 시작 표 인덱스), 모양 불일치는 오류. `edges(tau)`, `widths(lo,hi)`, `valid_mask(sigma, tau=0.1)`(표의 σ와 1e-9 이내만; **스냅하지 않고** 아니면 오류), `to_dict()/from_dict()`(JSON, NaN↔null, 왕복 동일). `max_abs_delta(a,b)->float`(INT4 |ΔD|용; indices·levels 다르거나 NaN이면 오류).
- `TeacherCodec(teacher)`: `encode(x)=teacher.encode_image(x)`, `decode(z)=teacher.decode_latent(z)`(clamp 기본). 코덱 규약은 `encode([N,3,H,W])→잠재`, `decode(잠재)→[N,3,H,W]`.
- `run_x1_slim(sampler, codec, images[N,3,H,W], cond, *, seeds, levels=4, steps=2, indices=None, level=0, k=0) -> DecisionCurve` (`@torch.no_grad`). 순서: **검증(sampler·codec 호출 전)** → `lat=encode(images)` 1회 → 기준 `ref=decode(lat)`(**원본이 아니라 코덱 왕복**과 비교해 VAE 손실을 교사 결정과 분리) → σ 바깥·시드 안쪽 루프로 `sampler.sdedit(lat, cond, sigma=table.points[j], seeds=int(seed), steps=steps, level=level, k=k)`(추가 kwargs 없음, 전 이미지 한 호출) → 출력은 `x0_hat`(마지막 forward의 원 예측)을 `decode`. 시드는 σ마다 같은 목록(공통 난수). 기본 `indices=range(len(table)-steps+1)`. 시드 ≥2개·서로 다른 int(bool/float 불가), 인덱스 정렬·`j+steps≤len(table)`, 이미지 유한·3채널·4차원 텐서. `x0_hat` 모양이 `lat`과 다르면 오류. 기준이 평탄하면 해당 칸 NaN.
- `forward_count(n_images, n_seeds, n_indices, steps) -> int`(곱, 양의 정수만).
- **해석 `interpret(curve, *, width_max=1.5, sep_target=1.0, sep_tol=0.5, mono_tol=0.5, valid_tau=0.1, decided_tau=0.5) -> Interpretation(verdicts, shifts, adopted, monotone, e2a_pass, fallback)`**. `SigmaVerdict(index, sigma, edge_valid, edge_decided, width, usable)`: edge_valid=edge(valid_tau), edge_decided=edge(decided_tau), `usable = 두 가장자리 유한 ∧ width ≤ width_max`. `shifts[i] = edge_decided(j) − edge_decided(j+steps)`(파트너가 곡선에 없거나 무한이면 NaN). `adopted` = usable ∧ `|shift − sep_target| ≤ sep_tol`인 표 인덱스(오름차순 tuple). `monotone` = 유한 edge_decided가 인덱스 증가에 따라 `+mono_tol` 이상 오르지 않음. `e2a_pass = len(adopted) ≥ 2`. `fallback`: 통과면 `"none"`, 유한 폭이 2개 미만이면 `"inconclusive"`, 그 밖은 `"widen_level_gap_or_sr_lora"`. 곡선에 NaN이 있거나 파라미터가 `0<valid<decided`, `width_max>0`, `sep_target>0`, `sep_tol≥0`, `mono_tol≥0`을 어기면 `DecisionCurveError`.
- **가난선 프로브** `poverty_line(student_dr, teacher_dr, *, afc=None, ratio_max=1.1, afc_max=0.55) -> PovertyLine(n_octaves, passes, ratio)`: 옥타브 o별(o=0 최미세) `ratio=student/teacher`(교사>0·유한, 학생≥0), 통과 = `ratio ≤ ratio_max`(포함) ∧ (afc 있으면 `afc ≤ afc_max`, 포함). **학생 소유 옥타브 = 최미세에서 이어진 통과 접두의 길이**. `boundary_mm(footprint_mm) = footprint_mm·2^n`, `teacher_pixel_fraction = 0.25^n`. `octave_pitch_mm(o, footprint_mm)=footprint_mm·2^o`. `initial_o_s_mm(judge_mm, measured_mm)=max(...)`(PARADIGM CT7, 양의 유한수만).

## 불변식
순수·결정론(같은 입력 비트 동일, 시드 순서를 바꾸면 rtol 1e-9) · 모든 검증은 sampler/codec 호출 전 · 입력 불변 · 조용한 NaN 치환 없음 · 항등 교사 → D=0, spread=0 · 완전 디노이저(x̂0가 σ·시드와 무관) → spread≈0, 모든 σ 행 동일.

## 수용 기준
`tests/tch_05/` 전부 통과(CPU, <60 s, 동결 해시): D/spread 닫힌 식·풀링·옥타브 국소화(체커는 o=0에만)·FND-06 에너지 대조, 가장자리·폭·마스크의 손계산값과 독립 오라클, 컨테이너·JSON·`max_abs_delta`, `interpret`의 채택/탈락/불확정/단조성 사례, 실행기 호출 규약·코덱 왕복 기준·오류 우선 검증·알려진 결정 구조 오라클(블러+노이즈 교사), TCH-02 실물 `TeacherSampler`+가짜 변환기 연동, 가난선 접두 규칙. `@pytest.mark.gpu`(**실행하지 못했음**, 5090 필요): 실물 교사 곡선이 유한하고 σ에 단조.

## 열린 해석 (Fable 몫, 잠정값)
임계(0.1/0.5, 폭 1.5, 분리 1±0.5 옥타브, 채택 ≥2점)와 "두 step 창 = edge_decided 이동량" 해석, 옥타브 방향(0=최미세)은 PARADIGM E2 (a)·R4 문구를 구현 가능한 수식으로 옮긴 **작성자의 선택**이다. 사람 2AFC·DR·INT4 측정(E2 (b)–(d))은 이 단위 밖이며 `poverty_line`/`max_abs_delta`는 그 수치를 받기만 한다.

## 하류 이음매
DAT-02는 `adopted` σ와 `curve.valid_mask(σ)`를 쓰고(σ는 표 점만), WLD-12/PRD는 `PovertyLine.boundary_mm`과 `initial_o_s_mm`으로 o_s를 시드한다. MDL-03 σ̂²는 `spread`. 곡선은 `to_dict()` JSON으로 저장하고 `table_key`로 σ 표 일치를 확인한다.
