# WLD-08 뷰별 SDEdit + 평균 역투영 폴백 — 실제 화면에 나가는 발명 채움 (S2 확장)

**목적.** WLD-03 `DepthWorld` 의 숨은 층(LDI 층 1, `hidden` 텍셀, 색은 임시 최근접 복사)을 교사 발명으로 덮는다. 관측 시점에서 ±20° 이내 가상 카메라 몇 대로 세계를 찍고(RDR-02), 한 번의 lockstep SDEdit(σ≈0.5)으로 다시 그리게 한 뒤, 각 숨은 텍셀을 그 텍셀이 **보이는 뷰들**에 역투영해 **선형광 산술평균**으로 색을 정한다. 의미 패스·OTBI(WLD-07/09)가 이것을 이기러 오고, 이기지 못하면 이것이 출하 경로다. 관측 영역(층 0, 층 1의 비숨김)은 절대 건드리지 않는다.
**소유.** `buc/world/sdedit_avg.py`, 테스트 `tests/wld_08/`(동결 `WLD-08.tests.json`). **의존.** WLD-03 `depth_world`(`DepthWorld`, `build_depth_world`, `DepthWorldConfig`), WLD-01 `strata`(`Strata`, `OWNERS`), RDR-02 `quads.render_ldi`, FND-02 `contracts`, FND-06 `geom`·`color`, TCH-02 샘플러(덕 타입), numpy·scipy·torch. `diffusers`·`transformers` import 금지, CUDA 초기화 금지, 전역 RNG 불변, 입력 불변, 결정론, 경고 없음.

## 샘플러 규약(덕 타입; TCH-02 `TeacherSampler` 와 호환)
`sampler.teacher`: `.encode_image(x[B,3,H,W] float32 in [-1,1]) -> latent`, `.decode_latent(z) -> [B,3,H,W] in [-1,1]`, `.spec.check_image_size(h,w)`(16의 배수). `sampler.table`: `.snap(sigma)->(.index,.sigma)`, `len()`. `sampler.sdedit(x0, cond, *, sigma, seeds, steps)` -> 객체(`.z` 잠재 `[B,C,h,w]`, `.sigma_end`). 속성·메서드가 없으면 `TypeError`.

## 공개 API (`buc.world.sdedit_avg`)
```python
NAME = "sdedit_avg"; VERSION = "0.1.0"
class SDEditAvgError(contracts.ContractError)
@dataclass(frozen=True, kw_only=True) class SDEditAvgConfig:
    sigma=0.5 ((0,1] 유한); max_angle_deg=20.0 ((0,45]); n_per_side=2 (int 1..8); pivot_m=None (None|유한>0)
    erode_px=2 (int 0..8); depth_tol_rel=0.05 ((0,0.5]); min_votes=1 (int ≥1); prompt="a photograph" (str)
    # 위반(bool 정수, NaN 포함) ValueError
@dataclass(frozen=True, kw_only=True, eq=False) class ViewSpec:
    c2w:(4,4) f64; K:(3,3) f64; width:int; height:int; angle_deg:float; k1:float=0; k2:float=0
def auto_pivot_m(dw) -> float          # median(dw.z[dw.near]); near 없으면 median(dw.z)
def plan_views(dw, cfg=None) -> tuple[ViewSpec, ...]
@dataclass(frozen=True, kw_only=True, eq=False) class SDEditAvgResult:
    strata: Strata; world: World                  # world == strata.to_world()
    filled: (H,W) bool; votes: (H,W) int16; unfilled: (H,W) bool   # filled = hidden∧votes≥min_votes, unfilled = hidden∧¬filled
    views: tuple[ViewSpec]; view_disocc_frac: (n,) f64; view_hole_frac: (n,) f64
    cfg: SDEditAvgConfig; steps: int; sigma_snapped: float
    def meta(self) -> dict
def sdedit_avg_fill(dw: DepthWorld, sampler, *, cfg=None, views=None, seed: int = 0) -> SDEditAvgResult
def make_world_impl(sampler, *, depth_cfg=None) -> spine.Impl   # Impl("S2","real",NAME,VERSION,fn)
```
`dw`가 DepthWorld 아니면 `TypeError`, cfg가 Config/None 아니면 `TypeError`, `views` 비었거나 크기가 obs와 다르면 `ValueError`, seed가 음수·bool이면 `ValueError`, `make_world_impl`의 `depth_cfg`가 `DepthWorldConfig`/None이 아니면 `TypeError`. 검증 실패는 샘플러를 부르기 전에 난다.

## 알고리즘 (관측 가능한 결과 = 계약)
1. **뷰 계획.** 관측 카메라 좌표계에서 피벗 `P=(0,0,d)`, `d=pivot_m 또는 auto_pivot_m`. 각 `θ`에 대해 카메라 중심 `c=(−d sinθ, 0, d(1−cosθ))`, 회전 `R_y(θ)`(광축 `(sinθ,0,cosθ)`가 P를 본다), `c2w = c2w_obs · [R_y(θ)|c]`. `θ`는 `[−A,…,−A/n, +A/n,…,+A]`(A=`max_angle_deg`, n=`n_per_side`, 0 제외, 증가순; θ<0이면 카메라는 관측 카메라 +x 쪽). K·크기·k1·k2는 obs와 같다.
2. **대상.** `hidden = dw.hidden`. 비어 있으면 샘플러를 **부르지 않고** 채울 것 없이 반환(`filled` 전부 False).
3. **뷰 렌더.** 뷰마다 `render_ldi(dw.world, …)`(기본 옵션). `alpha<0.99` 화소는 가장 가까운 `alpha≥0.99` 화소 색으로 대체(검은 구멍을 교사에 주지 않는다). 표시값 `x=2·linear_to_srgb(clip(rgb,0,1))−1`. 크기가 16의 배수가 아니면 오른쪽·아래를 edge 복제로 올림하고 출력에서 자른다. `n`뷰를 한 배치 `[n,3,H16,W16]`로 `teacher.encode_image` 1회.
4. **SDEdit 1회(lockstep).** `sn=table.snap(cfg.sigma)`, `steps=len(table)−sn.index`(표 끝까지). `sampler.sdedit(z0, [cfg.prompt], sigma=cfg.sigma, seeds=seed, steps=steps)` **정확히 1회**, B=뷰 수. `sigma_end≠0`·`z` 모양 불일치·디코드 비유한이면 `SDEditAvgError`. 출력 `y=srgb_to_linear((decode+1)/2)`.
5. **투표.** 숨은 텍셀 p의 3D점(층 1 깊이로 `geom.unproject_world`)을 뷰 i로 `geom.project`. 투표 조건: (a) z>1e-3이고 반올림 화소가 영상 안, (b) 그 화소가 `core_i`(= `layer_id==1`을 3×3으로 `erode_px`회 침식, 영상 밖은 0)에 속함, (c) `|depth_i[px]−z| ≤ depth_tol_rel·z`. 표본은 `y_i`의 쌍선형 보간(가장자리 클램프). `votes`=투표 뷰 수.
6. **평균.** `votes≥min_votes`인 텍셀 색 = 투표 표본의 **선형광 산술평균**, `clip[0,1]` float32. 그 밖의 숨은 텍셀은 이전 색 그대로.
7. **출력 세계.** `dw.strata` 깊은 복사: 층 1 `rgba[...,:3]`만 hidden∧filled에서 바뀐다(알파·깊이·층 0·비숨김 비트 동일). TSQ `own==ENGINE` 행의 `color`는 그 행 소스 화소(`mu`를 obs로 재투영해 반올림)의 새 층 1 색으로 갱신, 다른 열·행은 비트 동일. `strata.meta["sdedit_avg"]=meta()`, `world=strata.to_world()`(`world.meta["strata_root"]` 일치).
8. **스테이지.** `make_world_impl(sampler, depth_cfg=None)`: `fn({"read"}, ctx)` = `build_depth_world(read, cfg=depth_cfg)` → `sdedit_avg_fill(cfg=SDEditAvgConfig(**params), seed=params.get("_seed", ctx.seed))` → `{"world": res.world}`. params에서 `_seed`를 빼고 모르는 키는 `ValueError`.

## meta()
`json.dumps(allow_nan=False)` 가능, 키 ⊇ `name, version, cfg(dict), n_views, angles_deg, sigma_requested, sigma_snapped, steps, hidden_px, filled_px, unfilled_px, filled_frac, view_disocc_frac(list), view_hole_frac(list)`. `view_disocc_frac[i]=mean(layer_id==1)`, `view_hole_frac[i]=mean(alpha<0.5 ∧ layer_id≠1)`(화면 밖 구멍은 채우지 않는다; MDL-12 몫).

## 불변식
관측 포즈 렌더 불변(채움 전후 1e-6) · 층 0·층 1 알파/깊이 비트 동일 · 가려져 어느 뷰에서도 보이지 않는 텍셀은 채우지 않는다 · 같은 입력·시드 → 같은 `merkle_root` · `check_stage_io("S2")` 통과 · 뷰 평균이라 1뷰 투표는 그 뷰 표본 그대로.

## 수용 기준 (`tests/wld_08/`, CPU <60 s, 가짜 교사)
API·검증·지연 import · 뷰 기하(거리·각도·피벗 중심·부호) · 해석적 가림 오라클과 투표 수 일치 · 정답 오라클 교사에서 숨은 텍셀 평균 오차 ≤0.01(채우기 전 ≥0.03), 상자색 누출 없음 · 비계획 뷰(15°, −17°) 시각 대리 지표 ≤0.01 · 선형광 평균·미채움 보존·관측 불변 · lockstep 1회 호출 규약(σ·steps·cond·seeds) · 입력 영상=렌더 · 16 비배수 크기 · 실제 `TeacherSampler`+가짜 변환기 이음매 · S2 스테이지. `@pytest.mark.gpu`: 실물 Z-Image는 **실행하지 못했음**.
**미검증(V, 창업자 눈):** 실제 SDEdit 발명의 질감·이음매, 숨은 영역 가장자리 `erode_px` 띠(층 0과 같은 평면에서 맞닿아 투표 못 받음, 임시 색 유지), 중앙 미관측(어느 뷰에서도 가려진) 텍셀, 실사 VAE 왕복 충실도.

## 하류 이음매
WLD-10/11: `result.strata`·`unfilled`·`filled`(이음매 후보), WLD-14: `result.world`, WLD-16: `plan_views`·`view_disocc_frac`로 감사 카메라 범위. WLD-07/09는 같은 `x0_hook`/`add_noise`로 대체해 `SDEditAvgResult` 모양을 유지한다. 시그니처를 바꾸면 이 계약서와 함께 올린다.
