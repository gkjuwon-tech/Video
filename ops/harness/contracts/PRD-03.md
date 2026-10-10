# PRD-03 스텁 팩 v0 — 고전 판독 · 평면 세계 · 고정 무브 (걷는 해골)

**목적.** FND-02 서명만 맞추는 가장 단순한 스테이지 구현. 실물(RD-21·WLD-01·CAM-02 …)이 없거나 실패해도 `입력 → 판독 → 세계 → 샷`이 항상 돈다(PRD-01 폴백 카드). 못생겨도 계약이 맞물린다는 증거가 목적이다.
**소유.** `buc/pipeline/stubs/*.py`(`__init__.py` 없이도 `import buc.pipeline.stubs.<mod>`가 된다), 테스트 `tests/prd_03/`(동결). **의존.** `buc.core.contracts`(FND-02), `buc.pipeline.spine`(PRD-01: `Impl`, `Registry`), numpy, 표준 라이브러리. torch·scipy 금지, import 부작용·전역 RNG 없음. 입력 객체를 변이하지 않는다. 같은 (입력, params) → 비트 동일 출력(`ctx.seed`는 `Shot.seed`에만 쓰고 그 밖에 영향 없음). 알 수 없는 params 키(`_seed` 제외)·범위 밖 값은 `ValueError`.

## 모듈과 공개 API
각 모듈은 `NAME: str`, `VERSION = "0.1.0"`, 스테이지 함수 `fn(inputs: dict, ctx: StageContext) -> dict`를 낸다.
| 모듈 | NAME | 함수 | 스테이지 |
|---|---|---|---|
| `read_classic` | `classic_read` | `classic_read` | S-1 `{image}→{read}` |
| `world_flat` | `flat_world` | `flat_world` | S2 `{read}→{world}` |
| `move_fixed` | `fixed_move` | `fixed_move` | S8 `{world, read}→{shot}` (world는 무시) |
| `pack` | — | `identity_polish`(S8.5: `{"world": inputs["world"]}` 그대로), `stub_impls() -> dict[str, Impl]`, `register_stubs(registry, *, replace=False) -> Registry` | S8.5 + 묶음 |

`stub_impls()` = 키 `("S-1","S2","S8","S8.5")` 순서, 값은 `Impl(stage, kind="stub", name=NAME, version=VERSION, fn)`(S8.5의 name은 `identity_polish`). `register_stubs`는 각각 `registry.register(impl, replace=replace)`하고 registry를 돌려준다(중복이면 PRD-01의 `ValueError`). **S9-10(렌더)·S11(KPI) 스텁은 이 팩에 없다**(RDR-02/06·EVL 몫). 스파인 전체 실행에는 그 구현을 따로 끼우거나 `until="S8.5"`로 멈춘다.

## 동작
**classic_read** (params: `focal_mm=35.0`, `cam_height_m=1.5`, `z_near_m=0.5`, `z_far_m=50.0`; 모두 유한 양수, `z_near_m ≤ z_far_m`)
- `theta_in = Theta(aspect=W/H, focal_mm=focal_mm)`(그 밖은 기본값). `obs = Obs(K=theta_in.intrinsics(W,H), R=I, t=0, width=W, height=H)`: 세계축 = 입력 카메라축(OpenCV, y 아래), 카메라는 원점.
- `image_lin`: uint8/255, uint16/65535, float32는 그대로를 `max(·,0)` 후 전달함수 적용. `linear-srgb`는 항등, `adobe-rgb`는 `v^2.2`, 그 밖(`srgb`·`display-p3`·`rec2020`·`unknown`)은 sRGB EOTF(`v≤0.04045 ? v/12.92 : ((v+0.055)/1.055)^2.4`). 색역 변환은 하지 않는다. float32 결과(float64로 계산).
- `depth`(수평선 가정 평면 깊이, 수평선 = 주점 행 `cy=(H-1)/2`): 행 `v>cy`는 `clip(fy·cam_height_m/(v-cy), z_near_m, z_far_m)`, 그 외(수평선 이상)는 `z_far_m`. 열마다 같다. float32, 항상 유한·양수.
- 고정 필드: `input_hash=image.h_I`, `prov=image.prov`, `level="L1"`, `scale_s=1`, `sigma_s=1`(가정 스케일이라 크다), `depth_conf`=0 전부 `(H,W)`(깊이가 가정값), `fallbacks=("stub:classic_read",)`, `meta={"stub":"classic_read"}`.

**flat_world** (params: `n_layers=3`, 정수 1–16)
- 유한 깊이의 `dmin,dmax`로 기하 경계 `e_j = dmin·(dmax/dmin)^(j/n)`, j=0..n(유한 깊이가 없으면 `dmin=dmax=1`). 화소 구간 `b = clip(searchsorted(e[1:-1], depth, side="right"), 0, n-1)`(`inf`·최대 깊이는 마지막 구간). 층 j: `rgb = clip(image_lin, 0, 1)`(모든 층 동일), 알파 = `b==j`이면 1 아니면 0(직선 알파, 화소마다 정확히 한 층만 1), 깊이 = 상수 `sqrt(e_j·e_{j+1})`(float32). 앞→뒤 정렬(j 오름차순).
- `World(obs=read.obs, layers, theta_in, level=read.level, read_hash=read.content_hash(), prov=read.prov, far_field=None, meta={"stub":"flat_world","n_layers":n,"bin_edges":[e_0..e_n]})`.
- 따라서 합성(앞→뒤 over) = `clip(image_lin,0,1)` 정확히, 그리고 관측 카메라에서 층을 찍으면 원래 영상이 나온다(상수 z 카드는 같은 화소로 투영).

**fixed_move** (params: `n_frames=25`(정수 ≥1), `fps=None`(→`theta_in.fps`, 0<fps≤240), `push_frac=0.10`(−0.5…0.5), `z_s=None`(→ 유한 깊이의 중앙값, 유한 양수))
- `C0=obs.c2w()`, `τ_k=k/(N-1)`(N=1이면 0), `d_k=push_frac·z_s·τ_k`, `c2w_k = C0·T(0,0,d_k)`(광축 선형 전진, 자세 고정). `t_k=k/fps`, `K_k=obs.K`, `focus_m_k = z_s − d_k`, `width/height=obs`, `fps`, `theta = theta_in에서 fps만 교체`, `move="stub_push_in"`, `rig="locked"`, `seed=ctx.seed`. 첫 프레임은 관측 포즈와 정확히 같다(`first_pose_matches`).

## 불변식
- 세 스텁 출력은 `check_stage_io`를 통과한다(taint 비감소: prov를 그대로 전달). 체인 `S-1→S2→S8→S8.5` 출력 `World`는 `Shot`·`Obs`와 일관.
- 스텁 출력은 `ctx.params["_seed"]`·`ctx.seed`와 무관(Shot.seed 제외) → `Spine.run` 반복·새 프로세스에서 같은 root.
- 모든 StageIO는 `impl_kind="stub"`이며, 실물이 실패해 폴백했을 때 PRD-01이 `fallback`을 기록한다.

## 수용 기준
`tests/prd_03/` 전부 통과(CPU, <60 s, `-m "not gpu"`, 동결 `PRD-03.tests.json`): 모듈·등록 API / 전달함수·깊이 오라클(float32 허용오차) / 색역별 전달 / 계약 검사 / 층 분할 정확성과 합성 항등 / 무브 공식 오라클(비단위 관측 포즈 포함) / params 검증 / 스파인 실행·캐시 hit·부분 재실행·실물 실패 폴백 / RDR-02 `render_ldi`가 관측 포즈에서 입력을 재현(atol 1e-4)하고 샷 전체가 렌더됨. GPU 테스트 없음.

## 하류 이음매
PRD-01: `register_stubs(reg)`로 등록하고 실물은 `kind="real"`로 같은 스테이지에 끼운다(스텁 코드 불변). PRD-02 CLI의 `--stub` 경로와 EVL-02(일일 A/B 껍데기)가 이 팩으로 `until="S8.5"`까지 돈다. RDR-02/06은 `World.layers`(상수 깊이 카드)와 `Shot`만 읽는다. 해골 클립은 `fallbacks`·`depth_conf=0`·`sigma_s=1`로 "가짜 판독"임이 표시된다 — KPI·벤치가 이 표시를 읽어 스텁 결과를 점수에서 구분해야 한다(EVL 몫). 이름·버전·공식을 바꾸면 캐시 키가 바뀌므로 `VERSION`과 이 계약서를 함께 올린다.
