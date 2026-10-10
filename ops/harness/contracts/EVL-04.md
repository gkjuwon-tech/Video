# EVL-04 첫 프레임 KPI v0 — FFD · FFG · ΔE00 · ΔG · FP(t) · OIR (R-EVAL)

**목적.** pipeline.upg §5.2 KPI 정의를 한 모듈로 고정한다. 평가 전용 LPIPS는 **별도 프로세스**(R-EVAL)에서만 돌고, 결과는 사람이 읽는 보고서·KPI 숫자로만 나간다(최적화 신호 금지, 반환값은 순수 `float`).
**소유.** `eval/first_frame.py`, 테스트 `tests/evl_04/`(동결). 의존 FND-06(`buc.core.color`, `buc.core.pyramid`)만; `buc.core` 밖 `buc.*` import 금지. 모듈 수준 import에 torch·lpips·torchvision 등 금지(워커 안에서만 지연 import). import 부작용 없음. `eval/`은 패키지가 아니다(테스트는 파일 경로로 적재; 워커는 `python eval/first_frame.py --worker --backend B`로 같은 파일을 실행, 레포 루트를 `sys.path`에 넣는다).
**영상 규약.** 입력은 표시값(sRGB 부호화) `(H,W,3)` float32/64 [0,1] 또는 uint8(/255). 그 밖 dtype `TypeError`, ndim/채널/NaN/범위(±1e-6)/빈 영상 `ValueError`. 비교 쌍은 크기 같아야 한다(`ValueError`; 같은 크롭은 호출자 몫). 입력 불변, 결정론(같은 입력 → 같은 값), 반환은 내장 `float`.

## 공개 API
```python
SCHEMA_VERSION=1; R_EVAL=True; REF_SHORT=1080; GRAIN_BLUR_SIGMA=0.7; PATCH=32; SAL_FRAC=0.10
GRAIN_F_LO=1/16; GRAIN_F_HI=0.5; FP_MIN_COVIS=0.05; FFD_FLOOR_MAX=0.015; DE00_FLOOR_MAX=0.5
class EvalWorkerError(RuntimeError); class EvalModelUnavailable(EvalWorkerError)
class LpipsClient:    # 워커 프로세스 1개를 소유
    def __init__(self, backend="lpips", *, net="alex", device="cpu", timeout_s=120.0)
    backend: str; is_perceptual: bool; worker_pid: int
    def distance(a, b) -> float     # a,b 표시값 (H,W,3), 검증은 부모에서 먼저
    def info() -> {"pid","backend","torch_loaded","lpips_loaded"}; def close()  # 멱등, 회수까지; with 지원
def proxy_lpips(a, b) -> float
def as_display(img, name="img") -> float32 (H,W,3)
def prepare(img, *, r_eff=None, short_side=1080, sigma=0.7) -> float32   # Ψ 다음 B
def ffd(inp, f0, client, *, r_eff=None) -> float
def delta_e00(inp, f0, *, mask=None, patch=32, min_cover=0.5) -> float
def estimate_flow(a, b) -> float32 (H,W,2)        # 채널 (dx,dy), a(p) ≈ b(p+flow(p))
@dataclass(frozen=True) class FFGResult: median: float; p95: float; n: int
def ffg(inp, f0, *, mask=None, ref_short=1080) -> FFGResult
def ffg_sal(inp, f0, region_mask, *, ref_short=1080) -> float      # 영역 P95
def top_fraction_mask(score, frac=0.10) -> bool (H,W)
def grain_band_ratios(inp, f0, *, n_bands=6) -> ndarray(n_bands,); def delta_g(inp, f0, *, n_bands=6) -> float
@dataclass(frozen=True) class FPCurve: times, lpips, de00, covis: ndarray(N); slope_lpips, slope_de00: float; backend: str
def fp_curve(frames, warped_inputs, covis_masks, client, *, times=None, r_eff=None) -> FPCurve
@dataclass(frozen=True) class OIRResult: n,k:int; rate,ci_low,ci_high,p_value:float; as_dict()
def oir_counts(k, n) -> OIRResult; def oir(correct) -> OIRResult     # bool/0·1 정수 1차원
@dataclass(frozen=True) class Targets: ffd,ffg,ffg_sal,de00,oir (Optional[float]); TARGETS: dict[str,Targets]
@dataclass(frozen=True) class TargetVerdict: ok: bool; checks: dict[str, bool|None]
def evaluate_targets(metrics: Mapping, klass: str) -> TargetVerdict
@dataclass(frozen=True) class FirstFrameReport: schema, eval_only, backend, is_perceptual, r_eff, ffd, ffg_median, ffg_p95,
    ffg_sal_p95, de00, dg, oir, klass, verdict;  as_dict(); to_json(); kpi_fields() -> {"ffd","ffg","de00"}
def first_frame_report(inp, f0, client, *, r_eff=None, sal_mask=None, oir_responses=None, klass=None) -> FirstFrameReport
```
`client`는 `distance(a,b)->float`, `backend: str`, `is_perceptual: bool`을 가진 아무 객체(테스트는 대역을 넣는다).

## 정의
- **Ψ·B(`prepare`).** 목표 짧은변 `min(r_eff, short_side)`; 짧은변이 목표 이하면 **업샘플 없음**. 아니면 FND-06 `resize(...,"bilinear")`, 긴 변은 half-up 반올림. 이어서 `gaussian_blur(σ=0.7)`(표시값에 적용). **FFD** = `client.distance(prepare(입력), prepare(F0))`(입력이 앞).
- **ΔE00.** 좌상단부터 `patch×patch` 온전한 패치만; 패치별 **선형광** RGB 평균(mask 있으면 마스크 안 픽셀만) → Lab → `delta_e_2000` → 패치 평균. 마스크 점유 ≥ `min_cover`(경계 포함)인 패치만 쓰고 없으면 `ValueError`. 영상이 패치보다 작아도 `ValueError`. 흰↔검정 = 100.
- **광류/FFG.** OpenCV(DIS) 없음 → numpy/scipy 자체 구현(피라미드 Lucas-Kanade 등, 정확도는 테스트가 정한다). 휘도(Rec.709) 기준. `ffg`는 크기 `|flow|` × `ref_short / min(H,W)`(px@1080p)의 중앙값·P95(`n`=사용 픽셀 수). 마스크는 bool (H,W), 비면 `ValueError`. `top_fraction_mask`: 상위 `ceil(frac·N)`개, 동점은 래스터 순서가 앞선 쪽.
- **ΔG.** 휘도(Rec.709, 평균 제거) 주기도 PSD `|FFT|²/(H·W)`, 반경 주파수(cy/px)를 `[1/16, 0.5]`에서 로그 간격 `n_bands`대역(마지막 대역은 상단 포함), 대역 평균 PSD(바닥 1e-12). `grain_band_ratios` = 대역별 `log2(P_F0/P_입력)`(부호 있음), `delta_g` = 그 절댓값 평균. 빈 대역이면 `ValueError`.
- **FP(t).** 호출자가 입력을 t로 워프해 넘긴다(여기서 워프하지 않는다). 프레임 i: `covis[i]`=마스크 평균. `< FP_MIN_COVIS`면 lpips·de00 = NaN. 아니면 공동 가시 밖 픽셀을 프레임 값으로 채운 워프 입력과 프레임을 `prepare` 후 `client.distance(워프입력, 프레임)`; de00 = `delta_e00(워프입력, 프레임, mask=공동가시)`. 기울기 = 유한 점의 최소제곱 기울기(`times` 기본 `0..N-1`; 유한 점 <2면 NaN). `times` 단조 요구는 없고 유한·길이 N만.
- **OIR.** Wilson 95% 구간, 정확 이항 양측 p(귀무 0.5). `n<1`, `k∉[0,n]` `ValueError`.
- **TARGETS(§5.5).** `L1_photo_green`(.05,.5,.5,1.5,.55) · `L1_ai_yellow_safe`(.06,1.0,.5,2.0,.60) · `L1_ai_yellow_ext_full`(.10,2.5,1.0,2.0,.65) · `escher_red_safe`(.08,1.5,.75,2.5,—) · `L2_painting_safe`(.10,1.5,.75,3.0,—); 순서 ffd,ffg,ffg_sal,de00,oir. 경계값 통과(`<=`), 목표나 측정이 없으면 `None`(실패 아님), NaN은 `False`, `ok = all(c is not False)`. 알 수 없는 `klass`·지표 키 `ValueError`.

## R-EVAL 불변식
1. 부모 프로세스에 torch/lpips가 로드되지 않는다(프로세스 import 시에도, 클라이언트 사용 뒤에도). 워커는 별도 PID이고 `close()`가 회수한다.
2. 백엔드 `"lpips"`(기본)는 패키지·가중치가 없으면 `EvalModelUnavailable` — **조용히 `proxy`로 바꾸지 않는다.** `"proxy"`는 numpy 전용 대역(`proxy_lpips`: 항등 0·대칭·블러/노이즈/JPEG 품질에 단조)이고 `is_perceptual=False`; 그 사실이 보고서에 `backend`/`is_perceptual`로 남는다. proxy 수치는 공개 KPI가 아니다.
3. 워커 사망/타임아웃은 `EvalWorkerError`(멈추지 않음). 부모의 입력 검증 실패는 워커를 건드리지 않고 클라이언트는 계속 쓸 수 있다.
4. 보고서는 `eval_only=True`, JSON 직렬화 가능, 유한 수치. 시스템 경로(`buc/`)가 `eval`을 import하는 것은 GOV-05 린트의 몫이다.

## 수용 기준
`tests/evl_04/` 전부 통과(CPU, <60 s, 동결 해시 `EVL-04.tests.json`): 격리(새 프로세스 import, 워커 PID/회수/강제 종료, 소스 import 검사), Ψ·B 정확 크기·값, FFD 전달 순서, ΔE00(100·패치 평균·선형광·저역·마스크), 광류 합성 이동 중앙오차 <0.1 px, FFG 단위 환산, ΔG 백색잡음 진폭비 → 2.0 log2, FP 곡선·기울기·NaN 처리, OIR scipy 대조, 목표표, 보고서/KPI 이음매. `@pytest.mark.gpu`: 진짜 LPIPS 워커(`lpips` 필요).

## 하류 이음매
WLD-14: `ffd` > `TARGETS[...].ffd`면 InputPolish. EVL-12(X-FF0 바닥 `FFD_FLOOR_MAX`, `DE00_FLOOR_MAX`)·EVL-13·EVL-15·TCH-07: `LpipsClient`+`ffd`/`fp_curve`. EVL-07·EVL-09·PRD-05: `FirstFrameReport.as_dict()`/`kpi_fields()` → `buc.core.contracts.KPI(ffd, ffg, de00)`. 의존성 요청: 진짜 LPIPS용 `lpips` 패키지를 선택 extra(`eval`)로 FND가 추가.
