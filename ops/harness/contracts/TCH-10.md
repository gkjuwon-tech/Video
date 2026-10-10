# TCH-10 양자화 — INT8(SmoothQuant)·W4A4(SVDQuant식 저랭크)·TRT, 자체 보정셋 + prov 검사 (T4 부록)

**목적.** 엔진 교사를 우리 손으로 양자화한다(PARADIGM §1.4 INT8 TRT SmoothQuant α 0.5–0.8, W4A4 Nunchaku SVDQuant, §5 "우리 보정셋, prov 검사 통과", 사전 양자화 가중치 금지, CT32 "양자화 보정셋으로 Soul이 흐르지 않는다"). CPU에서는 **수치 에뮬레이션(fake-quant)** 이 정본이고, TRT 엔진·Nunchaku 커널은 GPU 부록(미확인, E0)이다. 5090 BF16 경로에는 쓰지 않는다(순서 최후 N).
**소유.** `buc/teacher/quant.py`, `tests/tch_10/`(동결). 의존: FND-04(`buc.gov.prov`), TCH-09(`Fp32Island`). import는 torch·numpy·표준 라이브러리만, tensorrt·nunchaku·CUDA 없이 되고 부작용 없음(백엔드는 `importlib.util.find_spec`로만 확인).

## 공개 API (`buc.teacher.quant`)
- `QUANT_TAG="buc.quant.v1"`, `MODES=("int8","w4a4")`. 예외 `QuantError(RuntimeError)` ⊃ `QuantSpecError(+ValueError)`, `BackendUnavailable`. prov 위반은 FND-04 `FirewallError`.
- `qmax(bits)`: 8→127, 4→7(대칭·제한 범위), 그 밖 `QuantSpecError`.
- `absmax_scale(x, bits, *, group_size=None) -> float32`: None이면 0-d `max(|x|.max(),1e-8)/qmax`; g면 마지막 축을 g씩 묶어 모양 `x.shape[:-1]+(K//g,)`(K%g≠0 `QuantSpecError`).
- `quantize(x, scale, bits, *, group_size=None) -> int8` = `clamp(round(x/scale), ±qmax)`(scale 비유한·≤0 `QuantSpecError`). `dequantize(q, scale, *, group_size=None) -> float32`. `fake_quant(x, bits, *, group_size=None, scale=None)` = dequantize∘quantize, **x의 dtype** 반환(scale None이면 absmax_scale).
- `smooth_factors(act_absmax[K], w_absmax[K], alpha) -> float32[K]` = `clamp(a,1e-5)^α / clamp(w,1e-5)^(1-α)`. α∉[0,1]·모양 불일치 `QuantSpecError`.
- `lowrank_split(w[O,K], rank) -> (up[O,r], down[r,K], resid[O,K])` float32, r=min(rank,O,K), float64 SVD 절단(`up=U·S`, `down=Vh`), `up@down+resid == w`. rank 0이면 resid=w.
- `@dataclass(frozen=True) QuantSpec(mode, alpha=0.5, smooth=True, group_size=64, rank=32, skip=())`: 생성 시 검증(`QuantSpecError`), skip은 정렬·중복 제거 tuple. `w_bits`/`a_bits`(int8→8, w4a4→4), `to_dict()`(JSON 가능).
- `QuantLinear(nn.Module).from_linear(linear, spec, act_absmax[K])`(입력 Linear 불변; nn.Linear 아님·act 모양 불일치·w4a4 K%g≠0이면 `QuantSpecError`). 버퍼: `inv_smooth[K]`(=1/s, smooth=False면 1), `qweight` int8 [O,K], `w_scale`, `bias`(float32|None), int8: `a_scale` 0-d, `lora_up/lora_down`=None; w4a4: `a_scale`=None, `lora_up[O,r]`,`lora_down[r,K]` float32. W' = W·s(열).
  - **int8**: 가중치 출력 채널별 `w_scale[O]`=행 absmax/127, 활성 **정적 텐서별** `a_scale`=max(act_absmax/s)/127(보정값, 실행 시 클램프).
  - **w4a4**: W'=up@down+R, R을 입력 축 g묶음 4비트(`w_scale[O,K/g]`), 활성은 **토큰별 g묶음 동적** 4비트, 저랭크 가지는 양자화하지 않은 x'로 float32.
  - `forward(x)`: float32로 계산, x.dtype 반환. x'=x·inv_smooth; y = Qa(x')@deq(R)ᵀ (+ (x'@downᵀ)@upᵀ) + b. `dequant_weight()` = deq(R)(+up@down) [O,K](스무딩 공간; ·inv_smooth면 원 W 근사).
- `collect_act_absmax(model, batches, *, run_fn=None, names=None) -> dict[str, float32[K]]`: Linear 입력의 채널별 |x| 최댓값 누적(no_grad, 출력 불변, 끝나면 훅 제거·예외 시도). names=None이면 `Fp32Island` 밖의 모든 Linear. 호출 안 된 모듈은 키 없음. 비유한 입력 `QuantError`. run_fn(model,batch) 기본 `model(batch)`.
- `@dataclass(frozen=True) CalibSample(id: str, data: Tensor, prov: ProvLike)`. `build_calib_set(candidates, n, *, seed=0, strict=True) -> CalibSet`: id 중복·모양 불일치·n∉[1,가용] `ValueError`, data가 텐서가 아니면 `TypeError`. **strict면 후보 하나라도 `train_ok`가 아니면 `FirewallError`**(Soul·USER·HOLDOUT 등 fail closed); strict=False면 버리고 `rejected`(정렬 id)에 남긴다. 선택: 통과 후보를 id 정렬 후 `random.Random(seed).sample` → 입력 순서 무관·결정론.
- `@dataclass(frozen=True) CalibSet(samples, prov, digest, rejected=())`(직접 생성 가능; 소비자는 `prov`를 다시 검사한다): `samples`, `ids`, `prov`=`derive("quant-calib", 선택 prov)`, `digest`(sha256 hex, 선택 집합의 id·dtype·shape·바이트, 선택 순서 무관), `rejected`, `__len__`, `batches(bs) -> list[Tensor]`(선택 순서로 stack).
- `assert_not_prequantized(state_dict)`: 정수형 텐서(0-d int64 카운터 제외)나 양자화 접미사 키(`qweight, qzeros, scales, wscales, w_scale, a_scale, weight_scale, input_scale, smooth, smooth_factor, lora_up, lora_down, proj_up, proj_down, SCB, absmax, quant_map`)가 있으면 `FirewallError`(제3자 사전 양자화 금지).
- `weight_digest(model) -> str`, `quant_key(spec, calib_digest, weight_digest="", engine_hash="") -> str`(sha256 hex, 정렬 JSON + QUANT_TAG; Exhaust 키의 "양자화 모드 ‖ TRT 엔진 해시").
- `quantize_model(model, spec, calib, *, weight_prov, run_fn=None, batch_size=1) -> QuantReport` (제자리). **변경 전에** 모두 검사: 타입(`TypeError`), 이중 적용(`model._buc_quant`→`QuantError`), `derive("quantize",[weight_prov, calib.prov], updates_shared_weights=True)` + `assert_train_ok`(→`FirewallError`), `assert_not_prequantized`, skip의 없는 이름(`QuantSpecError`). 대상 = 섬 밖 Linear − skip; 보정 중 안 불렸거나 w4a4에서 K%g≠0이면 float 유지. `QuantReport(mode, quantized, skipped, prov, calib_digest, weight_digest, key)`(이름은 정렬 tuple, skipped = 모든 Linear − quantized, weight_digest는 양자화 전). `model._buc_quant = report`.
- `spearman(a,b)`(동률 평균 순위; 길이 불일치·n<2·비유한 `ValueError`; 분산 0이면 0.0), `precision_gate(ref, test, *, min_rho=0.9, max_delta=0.05) -> PrecisionGate(rho, mean_abs_delta, max_abs_delta, n, ok)`, ok = n≥3 ∧ rho≥min_rho ∧ mean|ΔD|≤max_delta(PARADIGM 판정 정밀도 게이트; |ΔD|를 평균으로 해석).
- `tensorrt_available()`, `nunchaku_available() -> bool`. `build_trt_engine(onnx_path, out_path, *, calib, precision="int8", workspace_gb=1.0) -> str`(엔진 sha256): calib 타입(`TypeError`)·precision∈{int8,fp16}(`QuantSpecError`)·calib prov(`FirewallError`)를 먼저 보고, tensorrt가 없으면 `BackendUnavailable`.

## 불변식
1. |fake_quant(x)−x| ≤ scale/2(클램프 밖 제외). smoothing은 float64에서 정확한 동치: `(x/s)@(W·s)ᵀ = x@Wᵀ`.
2. 보정셋·양자화 산출물의 prov는 train_ok가 아니면 만들어지지 않는다(fail closed, 원자적).
3. 결정론: 같은 입력·seed → 같은 선택·digest·key·출력.

## 수용 기준
`tests/tch_10/` 전부(CPU, <60 s). 수치(장난감 Linear 256→128, 이상치 채널 4개 ×60): int8 α0.5 상대오차 ≤0.03이고 스무딩 없는 int8보다 ≥2.5배 작음; w4a4(α0.5, r32) ≤0.06이고 r0보다 ≥1.5배 작음; 이상치 없는 입력 int8 ≤0.025; 저랭크 구조 가중치에서 r4가 r0의 ≤0.5배. 장난감 MLP 전체 int8 ≤0.03, w4a4 ≤0.12. `@pytest.mark.gpu`: TRT INT8 엔진·Nunchaku W4A4(미실행, E0).

## 하류 이음매
교사 로더는 FP16 Rescue(TCH-09) 후 `quantize_model`; Exhaust/TFU 키는 `QuantReport.key`; 판정 대리 전환은 `precision_gate`; DAT-05·GOV-01은 `CalibSet.prov`를 조상 DAG에 넣는다. **소유 밖 요청: 없음**(GPU 백엔드 의존성 tensorrt·nunchaku는 GPU 러너 환경에만).
