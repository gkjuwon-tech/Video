# TCH-06 Teacher-DCA · DR 판정 — 조건 2개 × σ 2개의 denoising 오차 차, DR_rel 합격선

**목적.** 파라미터 0짜리 교사 판정기 두 가지. (1) **Teacher-DCA**(Diffusion Classifier식): 이미지 `x`가 조건 `c_a`와 `c_b` 중 어느 쪽에 더 맞는지. 상위 k 재순위(Gaffer), 사진풍 드리프트 감사 `Δ_photo`, 발명 영역 seed 선택에 쓴다. (2) **DR**(교사 denoising 잔차): Spec-commit의 `DR_rel ≤ 1.2`, 감사 S7의 `sup DR_rel`, 가난선 `o_s`의 `≤ 1.1`, E0 (e)의 DR 판별 AUC가 쓰는 스칼라. 후보당 **4 fwd**(2 조건 × 2 σ), DR은 σ당 1 fwd.
**소유.** `buc/teacher/dca.py`, 테스트 `tests/tch_06/`(동결). 의존: TCH-02(`TeacherSampler.predict_x0`, `SigmaTable.index_of`, `add_noise`, `SigmaError`), TCH-01(`TeacherSpecError`). 변환기는 **TCH-02 경유로만** 부른다(σ 스냅 표, `t=1-σ`, `v=-out` 규약을 다시 쓰지 않는다). import는 torch·numpy·위 모듈뿐이며 diffusers 없이 되고 부작용이 없다. **소유 밖 요청: 없음.**

## 수식 (flow matching, TCH-02와 동일)
`z_σ=(1-σ)x0+σε`, `v=ε-x0`, `x̂0=z-σ·v̂`. 그러면 `v̂-v=(x0-x̂0)/σ`.
- 잔차 `r(x̂0,x0)=mean_{C,h,w}(x̂0-x0)²` (float32, 뷰별). **DR** = σ 점들에 대한 `r`의 평균. `DR_rel=DR(후보)/DR(기준)`.
- 속도 오차 `e=r/σ²=mean(v̂-v)²`. ε-오차는 `(1-σ)²/σ²·r`이라 σ=1에서 0으로 퇴화하므로 **쓰지 않는다**(표의 모든 점에서 조건 수가 좋은 v-공간을 쓴다).
- `DCA(x;c_a,c_b)=mean_σ[e(x,c_b)-e(x,c_a)]`. **양수 = x가 c_a에 더 맞는다.** 같은 `z`(공통 난수)를 두 조건에 쓴다.
- `Δ_photo=DCA(F;photo,style)-DCA(I;photo,style)`, 양수면 사진풍으로 끌렸다.

## 공개 API (`buc.teacher.dca`)
- `DCA_SIGMAS=(0.75,0.5)`, `DR_SIGMAS=(0.5,)`, `DR_REL_MAX=1.2`, `DR_REL_MAX_OS=1.1`, `DR_FLOOR=1e-12`. `DCAError(TeacherSpecError)`(ValueError 계열).
- `residual(x0_hat, x0)->float32[B]`; `velocity_error(x0_hat, x0, sigma)->float32[B]`(`r/σ²`; σ∉(0,1]이면 `SigmaError`). 모양 불일치·비유한은 `DCAError`.
- `@dataclass(frozen=True) DRResult(dr: float32[B], per_sigma: float32[B,S], sigmas: tuple, calls: int, table_key: str)`; `dr=per_sigma.mean(1)`.
  `teacher_dr(sampler, x0, cond, *, seeds, sigmas=DR_SIGMAS, level=0)->DRResult`. σ마다 `z=add_noise(x0,σ,ε_j)`, `x̂0=sampler.predict_x0(z,σ,cond)`, `per_sigma[:,j]=r`.
- `@dataclass(frozen=True) DCAResult(dca: float32[B], err_a: float32[B,S], err_b: float32[B,S], sigmas: tuple, calls: int, table_key: str)`; `err_*`는 v-공간 `e`, `dca=(err_b-err_a).mean(1)`.
  `teacher_dca(sampler, x0, cond_a, cond_b, *, seeds, sigmas=DCA_SIGMAS, level=0)->DCAResult`. 변환기 호출 순서는 **σ 우선(주어진 순서 그대로, 정렬 안 함), 같은 σ 안에서 a→b**; 기본이면 `calls==4`(청크가 있으면 ×청크 수).
- `delta_photo(sampler, frame, ref, c_photo, c_style, *, seeds, sigmas=DCA_SIGMAS, level=0)->float32[B]`(두 이미지에 같은 시드; 8 fwd).
- `dr_rel(dr, dr_ref, *, floor=DR_FLOOR)->float64`: 원소별 비, `dr_ref`는 스칼라 또는 같은 길이. `dr_ref<floor`·비유한·`dr<0`이면 `DCAError`.
- `@dataclass(frozen=True) DRVerdict(rel: float64[B], accept: bool[B], max_rel: float)`; `dr_verdict(dr, dr_ref, *, max_rel=DR_REL_MAX)`: **`rel<=max_rel`이면 합격(경계 포함)**, `max_rel`은 유한 양수.
- `dr_auc(dr_real, dr_fake)->float`: `P(fake>real)+0.5·P(동률)` 전 쌍. 1-D 비어 있지 않은 유한 입력.
- `rank_topk(scores, k)->int64[min(k,n)]`: 큰 점수 먼저, **동률은 낮은 인덱스 먼저**. `k<1`·비유한·1-D 아님은 `DCAError`.
- 입력은 TCH-02와 같다: `x0 float[B,C,h,w]`, `cond_*`는 `Sequence[str]` 또는 `Sequence[Tensor[L,D]]`(길이 B 또는 1, a와 b가 다른 종류여도 된다), `seeds`는 int 또는 길이 B 시퀀스. 노이즈 `ε_j=teacher.make_noise(seed_i,1,8h,8w,level=level,k=j)[0]`(`j`=sigmas 안의 위치). `sigmas`는 **표의 점**이어야 하고(아니면 `SigmaError`), 비었거나 중복이면 `DCAError`.

## 불변식
1. `dca(a,b)=-dca(b,a)`, `dca(a,a)==0`(비트). 뷰 i의 결과는 배치 구성·순서·청크와 무관(`seeds`를 시퀀스로 줄 때). 접두 불변: `sigmas=(s0,)`의 결과는 `(s0,s1)` 결과의 첫 열.
2. 같은 입력 → 비트 동일. 완전 디노이저에서 `err_c[:,j]=mean((x*(c)-x0)²)/σ_j²`(시드 무관). `err_c·σ²==teacher_dr(c).per_sigma`(같은 시드·σ).
3. 모든 검증(σ, 모양, 두 cond, 시드)은 **첫 변환기 호출 전에** 실패한다. 변환기 비유한은 `TeacherNumericError`, 개수·모양 불일치는 `TeacherError`(TCH-02에서 전파).
4. 항상 `no_grad`; 결과 텐서는 `requires_grad=False`(입력이 grad를 요구해도).

## 수용 기준
`tests/tch_06/` CPU 통과(<60 s, 가짜 변환기+진짜 `load_teacher`): 잔차·v-오차 수식, 완전·아핀 해석 오라클 대조, 공통 난수·호출 순서·`t=1-σ`, 대칭·영, 뷰 독립·청크·접두, 결정성·시드 민감도, 사전 검증, `dr_rel/verdict` 경계, AUC 손계산, `rank_topk` 동률, grad 차단, bf16 허용 오차, 사용자 표(`table=`). `@pytest.mark.gpu`(실행하지 못했음, 5090 필요): 실물 DCA 부호와 DR 판별 AUC 하한. 기준선이 실제로 E0 (e) ≥0.7을 넘는지는 모른다.

## 하류 이음매
WLD-12 Spec-commit: 합격 = `dr_verdict(teacher_dr(...,seeds=held-out WNF′).dr, 기준 dr).accept`. WLD-16 감사: `sup DR_rel`은 호출자가 `dr_rel(...).max()`로. Gaffer 재순위·hero 폐지 후 seed 선택: `rank_topk(teacher_dca(...).dca, 16)`. MRH-Aes ⊕ DCA 앙상블의 가중은 이 단위 밖(점수만 낸다). TCH-04 원장은 `table_key`, `sigmas`, `seeds`, `level`을 키 재료로 쓴다. Δ_photo는 감사 지표일 뿐 레벨 판정(PCI)이 아니다.
