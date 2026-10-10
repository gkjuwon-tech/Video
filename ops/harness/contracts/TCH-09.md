# TCH-09 FP16 Rescue — 잔차 1/s · SwiGLU 재균형 · FP32 섬 (T4 부록)

**목적.** BF16 네이티브 교사(Z-Image-Turbo DiT)를 BF16이 없는 T4에서 **학습 없이, 수학적으로 동치로** FP16에서 돌린다(PARADIGM §5 "FP16 Rescue", E0 (a)). (i) 잔차 스트림에 들어가는 모든 쓰기(입력 임베딩, 스트림 파라미터, 블록 출력 W_o/W_down 또는 adaLN gate 행, sandwich-norm이면 post-norm 가중치)에 1/s를 곱하고, 스트림을 읽는 norm의 eps를 1/s²배 해 정확히 보상한다. (ii) SwiGLU는 선형 쪽 W_up을 1/t, W_down 입력 쪽을 t배(바이어스는 그대로). (iii) norm·softmax 등은 FP32 섬, SDPA는 FP32 누산. (iv) max-abs 프로파일로 s, t를 2의 거듭제곱으로 고른다. 5090 BF16 경로에는 쓰지 않는다(순서 최후).
**소유.** `buc/teacher/t4_fp16.py`, `tests/tch_09/`(동결). 의존: TCH-01(`ZImageTeacher.transformer`, 예외 계층은 독립). import는 torch·numpy·scipy만, CUDA·diffusers 없이 되고 부작용 없음. **소유 밖 요청: 없음.**

## 공개 API (`buc.teacher.t4_fp16`)
- `FP16_MAX = 65504.0`. 예외 `RescueError(RuntimeError)` ⊃ `RescuePlanError(+ValueError)`.
- 원시 연산(제자리, `no_grad`, 모듈 반환, 2의 거듭제곱이 아니어도 동작):
  `scale_output_(linear, f, rows=None)`: `W[rows]*=f`, `b[rows]*=f`(rows는 `slice` 또는 None=전체). `scale_input_(linear, f)`: `W*=f`, **b 불변**. `scale_affine_(norm, f)`: `weight*=f`, bias 있으면 `bias*=f`; weight가 None/없음이면 `RescuePlanError`. `scale_eps_(norm, f)`: float 속성 `eps`(없으면 `variance_epsilon`)에 `*=f`; 둘 다 없거나 None이면 `RescuePlanError`. linear 계열은 `nn.Linear` 인스턴스가 아니면 `RescuePlanError`.
- `@dataclass(frozen=True) SwiGLUSite(up: str, gate: str, down: str, scale: float)`.
- `@dataclass(frozen=True) RescuePlan(s: float, embed=(), stream_params=(), gate_rows=(), residual_out=(), stream_norms=(), unscale=(), swiglu=(), fp32_islands=())` — 모두 `get_submodule`/`get_parameter` 이름의 tuple. 의미:
  `embed`: Linear → `scale_output_(1/s)`. `stream_params`: 파라미터(패드 토큰 등) `*= 1/s`. `gate_rows`: `(linear, start, stop)` → 해당 행만 `scale_output_(1/s)`(선형 gate). `residual_out`: Linear면 `scale_output_(1/s)`, norm(가중치+eps 보유)이면 `scale_affine_(1/s)`(sandwich post-norm). `stream_norms`: `scale_eps_(1/s²)`. `unscale`: 스트림을 norm 없이 읽는 Linear → `scale_input_(s)`. `swiglu`: `up`→`scale_output_(1/t)`, `down`→`scale_input_(t)`(gate 불변). 같은 모듈이 여러 항목에 있으면 인자는 곱으로 합성된다.
  `validate()`: s, t는 유한·≥1·2의 정수 거듭제곱(아니면 `RescuePlanError`); 한 필드 안 이름 중복 금지; `gate_rows`는 `0<=start<stop`.
- `apply_rescue(model, plan) -> RescueReport` (제자리). **모든 이름·타입·행 범위를 먼저 검증하고, 하나라도 틀리면 아무것도 바꾸지 않고** `RescuePlanError`. 이미 적용된 모델(`model._buc_fp16_rescue` 표시)이면 `RescueError`. 스케일 후 `fp32_islands`를 `apply_fp32_islands`로 감싼다. `RescueReport(s, touched: tuple[str,...] 정렬, islands: tuple[str,...])`.
- `Fp32Island(nn.Module)`(속성 `inner`, 감쌀 때 `inner`를 float32로 변환): 부동소수 텐서 인자를 float32로 올려 `inner`를 autocast 끈 채 실행하고, 부동소수 출력(텐서/튜플/리스트)을 **첫 부동소수 입력의 dtype**으로 되돌린다. `model.half()`/`.to(float16|bfloat16)` 뒤에도 `inner`의 부동소수 파라미터·버퍼는 float32로 남는다(장치 이동은 따른다).
- `apply_fp32_islands(model, names) -> list[str]`: 각 서브모듈을 `Fp32Island`로 교체(이미 섬이면 건너뜀, 멱등), 없는 이름은 변경 전 `RescuePlanError`. `cast_fp16(model)`: 섬 밖은 float16, 섬은 float32; 모델 반환.
- `sdpa_fp32(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> Tensor`: float32로 계산(F.scaled_dot_product_attention 의미 동일), `q.dtype`으로 반환. `fp32_sdpa()` 컨텍스트: 그 안에서 `torch.nn.functional.scaled_dot_product_attention`을 `sdpa_fp32`로 바꾸고 나갈 때(예외 포함) 원래 객체로 복원.
- `MaxAbsProfiler(model, names=None)` 컨텍스트: names=None이면 루트 제외 모든 잎 모듈. 훅으로 `out_max[name]`(부동소수 출력 |x| 최댓값의 누적 max, 비유한 값 제외), `in_max[name]`(첫 부동소수 위치 인자), `nonfinite[name]`(입·출력에 비유한 값이 있던 호출 수), `calls[name]`. 출력을 바꾸지 않고, 나갈 때 모든 훅을 제거한다. 없는 이름은 `RescuePlanError`.
- `choose_scale(max_abs: float | Mapping[str,float], *, headroom=8.0, fp16_max=FP16_MAX, max_s=2.0**16) -> float`: (매핑이면 값의 최댓값) `max_abs/s <= fp16_max/headroom`을 만족하는 **가장 작은 2의 거듭제곱 s≥1**. 비유한·음수 입력이나 s>max_s가 필요하면 `RescueError`.
- `upper95(k, n) -> float`: 단측 95% Clopper-Pearson 상한(k=0이면 `1-0.05**(1/n)`, k=n이면 1.0). n<1 또는 k∉[0,n]은 `ValueError`.
- `audit_images(images, *, black_thresh=0.02) -> AuditReport(n_images, n_nonfinite, n_black, upper95_fail, ok)`: 원소는 `[B,3,H,W]` 또는 `[3,H,W]`, 값 범위 [-1,1]. 비유한 값이 있으면 nonfinite, 아니고 `max((x+1)/2) <= black_thresh`면 black(배타). `ok = n_images>0 and 실패 0`.
- `zimage_plan(transformer, s, *, swiglu_scales=None, islands=True) -> RescuePlan`: diffusers `ZImageTransformer2DModel` 이름 규약(블록 `layers.*`/`noise_refiner.*`/`context_refiner.*`, `attention_norm1/2`, `ffn_norm1/2`, `feed_forward.w1(gate)/w3(up)/w2(down)`, `x_embedder`, `cap_embedder`, 패드 토큰, `final_layer`)으로 sandwich 계획을 만든다(`swiglu_scales`: float=모든 사이트, Mapping=`down` 이름→t, None=1.0; `islands`면 모든 norm을 섬으로). 해당 구조(`layers`)가 없으면 `RescuePlanError`. 실제 이름 정합은 GPU 테스트만 확인한다(E0, 미확인).

## 불변식
1. 2의 거듭제곱 s,t에서 rescue는 float64로 **원래 출력과 상대 1e-10 이내**(pre-norm·sandwich·직접 readout 모두). 스트림 max-abs는 정확히 1/s, SwiGLU 중간값은 1/t.
2. 검증 실패는 변경 전. 이중 적용 금지. import 부작용 없음.
3. FP32 섬 안의 계산은 float32, 밖의 dtype은 보존.

## 수용 기준
`tests/tch_09/` 전부 통과(CPU, 장난감 DiT, <60 s): 원시 연산, 계획 검증·원자성·이중 적용, float64 동치(eps 보상 포함), 스트림·SwiGLU 축소 비율, FP16 구조 실증(원본 FP16은 비유한, rescue+섬+fp32 SDPA는 유한이고 FP32 기준 상대오차 ≤3e-2), 섬 dtype, SDPA FP32 정확도, 프로파일러, choose_scale, upper95, 감사. `@pytest.mark.gpu`(미실행, T4/CUDA+가중치 필요): Z-Image FP16 rescue 출력 유한·BF16 대비 근접.

## 하류 이음매
TCH-10(양자화)은 `MaxAbsProfiler`(보정 통계)·`Fp32Island`·`fp32_sdpa`를 재사용하고 rescue된 가중치를 양자화한다. E0 (a) NaN 감사(1만 NFE)는 `audit_images`·`upper95`로 보고한다. TCH-02 샘플러는 변경 없이 rescue된 `teacher.transformer`를 부른다(호출 규약 동일).
