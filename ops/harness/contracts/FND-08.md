# FND-08 골든 픽스처·기하 정답 합성 장면 10종 v0

**목적.** 정답(깊이·법선·인스턴스·카메라·색)을 해석적으로 아는 소형 장면 10개를 결정론적으로 만들고 커밋해 둔다. 레인들은 서로의 코드를 읽지 않고 이 장면으로만 만난다(이음매 시험대). 장면은 **무작위 없이** 해석적 광선 투사로만 만든다.
**소유.** `tools/make_fixtures.py`, `tests/fixtures/*`(생성물: `<이름>.npz` 10개 + `manifest.json`, 합계 ≤ 6 MB). 테스트 `tests/fnd_08/`(동결). 의존: numpy, `buc.core.contracts`(FND-02), `buc.core.geom`·`color`(FND-06). **torch 금지**, import 부작용(전역 RNG·파일 쓰기) 없음. `tools/`는 패키지가 아니므로 파일 경로로 적재하고, 스크립트로 직접 실행해도 `buc`가 import되도록 저장소 루트를 (없으면) `sys.path`에 넣는다(PYTHONPATH 불필요).

## 공개 API (`tools/make_fixtures.py`)
- 상수: `FIXTURE_VERSION = "1"`, `FIX_W, FIX_H = 96, 64`, `SCENE_NAMES`(아래 표 순서 10개 튜플), `LIGHT_DIR`(= normalize(−0.3,−0.7,−0.6), float64 (3,)), `AMBIENT = 0.2`, `SKY_LIN = (0.25, 0.40, 0.75)`, `CHECK_A = (0.8,0.8,0.8)`, `CHECK_B = (0.1,0.1,0.1)`, `CHECK_M = 0.25`.
- `@dataclass(frozen=True, kw_only=True) View`: `obs: Obs`, `image_lin (H,W,3) float32 ≥0`, `depth (H,W) float32`(z-깊이, 하늘 +inf), `normals (H,W,3) float32`(카메라 좌표, 단위, 카메라를 향함, 하늘 (0,0,−1)), `instance (H,W) int32`(0 = 하늘). 배열은 C-연속.
- `@dataclass(frozen=True, kw_only=True) Scene`: `name: str`, `views: tuple[View, ...]`(views[0] = 기준), `gt: dict[str, str|bool|int|float|np.ndarray]`. `content_hash() -> str`(64 hex): 이름·gt·모든 뷰의 obs·배열 전부를 덮는 `contracts.content_hash` 기반(어느 한 값이 바뀌면 달라진다).
- `build_scene(name) -> Scene`(모르는 이름은 `KeyError`), `build_all() -> dict[str, Scene]`(SCENE_NAMES 순서). 순수·결정론(같은 입력 → 같은 바이트, 전역 RNG 불변·미사용). 10종 전체 < 10 s.
- `write_fixtures(out_dir) -> dict`(npz 10개와 `manifest.json` = `{"version": "1", "scenes": {이름: {"file": "<이름>.npz", "hash": scene.content_hash()}}}` 기록, 같은 dict 반환), `load_scene(name, root=None) -> Scene`(root 기본 = 저장소의 `tests/fixtures`; dtype·shape·gt 값 타입까지 비트 무손실 왕복), `check_fixtures(out_dir) -> bool`(재생성 해시와 manifest·파일이 모두 일치하면 True; 파일 없음·불일치는 False, 예외 없음).
- `to_input_image(scene, view=0) -> InputImage`(픽셀 = `color.linear_to_srgb8(image_lin)`, color_space "srgb", prov taint {"ENGINE"}), `to_read_packet(scene, view=0) -> ReadPacket`(image_lin·depth·obs·normals 그대로, `theta_in` = sensor 36 mm·`focal_from_fov`·aspect=W/H, input_hash = `to_input_image(...).h_I`, prov 같음).
- CLI: `python3 tools/make_fixtures.py [--out DIR]`(기본 tests/fixtures에 생성, 종료 0), `--check`(DIR을 쓰지 않고 `check_fixtures`; 일치 0, 불일치 1).

## 렌더 규약 (모든 값 float64로 계산 후 저장)
카메라: K = `geom.intrinsics_from_fov(fov_x, 96, 64)`. 요(yaw) θ: c2w 회전 = [[c,0,s],[0,1,0],[−s,0,c]], 중심 = eye, `Obs.R` = R_c2wᵀ, `t = −R·eye`. 세계는 오른손·미터·**y 아래**. 화소 중심(정수 좌표) 하나에서 광선 하나(표본추출·안티앨리어싱 없음): 카메라 광선 (x,y,1) = (`undistort`((u−cx)/fx, (v−cy)/fy), 1), 세계 광선 d = Rᵀ(x,y,1), 원점 = 카메라 중심. 교점 매개변수 t가 곧 z-깊이. 가장 가까운 양의 교점을 취하고, 같은 평면 위의 패치·원판은 평면을 덮는다. 법선은 카메라를 향한 면 법선(세계), 카메라 법선 = R·n.
색(선형): 평면 기본 알베도 = 체커 — 법선 최대 성분 축을 뺀 나머지 두 세계 좌표 (a,b)에 대해 `(⌊a/CHECK_M⌋+⌊b/CHECK_M⌋)` 짝수 → CHECK_A, 홀수 → CHECK_B. `lit`: 알베도·(AMBIENT + 0.8·max(0, n_w·LIGHT_DIR)). `emissive`: 알베도 그대로. 하늘은 음영 없이 SKY_LIN.

## 장면 (96×64, 평면은 `n·X=d`, n은 카메라를 향한 세계 단위 법선)
| 이름 | fov_x | 카메라 | 기하 (인스턴스 id) | 음영 |
|---|---|---|---|---|
| plane_frontal | 60° | 항등 | 평면 z=4 (1) | emissive |
| plane_tilted | 60° | 항등 | 평면 n=(0.3,−0.2,−1)/‖·‖, (0,0,4) 통과 (1) | lit |
| ground_horizon | 70° | 항등 | 바닥 y=1.5 (1), 위는 하늘 | lit |
| sphere_lit | 50° | 항등 | 구 c=(0.1,−0.05,3) r=0.9 알베도 (.7,.5,.3) (1), 하늘 | lit |
| box_pair | 60° | yaw −8°, eye (0.4,−0.1,−0.5) | 벽 z=6 (1); 상자 min(−.9,−.4,2.5) max(−.1,.6,3.3) 알베도 (.2,.5,.8) (2); 상자 min(.3,0,3.5) max(1.3,.8,4.5) 알베도 (.8,.6,.2) (3) | lit |
| corridor | 70° | 항등 | 바닥 y=1 (1), 천장 y=−1 (2), 왼벽 x=−1.5 (3), 오른벽 x=1.5 (4), 끝벽 z=10 (5) | lit |
| two_layer_card | 60° | 항등 | 배경 z=5 (1); 카드: 평면 z=2 안의 x∈[−.6,.6], y∈[−.4,.4] 사각형 알베도 (.8,.2,.2) (2) | lit |
| two_view_parallax | 60° | 뷰0 항등, 뷰1 yaw −3°·eye (0.3,0,0) | 벽 z=6 (1), 구 c=(−.2,.1,3) r=.6 알베도 (.7,.5,.3) (2) | lit |
| distortion_dots | 60°, k1=−0.2, k2=0.05 | 항등 | 평면 z=3 알베도 (.02,.02,.02) (1); 평면 위 원판 35개(중심 (0.4i, 0.4j, 3), i∈−3..3 안쪽 루프, j∈−2..2 바깥 루프, 반경 0.1 m, 알베도 1.0, id 2+k) | emissive |
| color_chart | 60° | 항등 | 평면 z=2.5 알베도 = srgb8 코드 (40,40,40) (25); 패치 24개 0.4 m 정사각, 열 c∈0..5는 x∈[−1.2+.4c, −.8+.4c], 행 r∈0..3은 y∈[−.8+.4r, −.4+.4r], id = 6r+c+1, 알베도 = `srgb8_to_linear(코드)` | emissive |

차트 코드(행 우선): r0 회색 0,51,102,153,204,255 · r1 (255,0,0)(0,255,0)(0,0,255)(0,255,255)(255,0,255)(255,255,0) · r2 (115,82,68)(194,150,130)(98,122,157)(87,108,67)(133,128,177)(103,189,170) · r3 (214,126,44)(80,91,166)(193,90,99)(94,60,108)(157,188,64)(224,163,46).

## gt 필수 키 (해석적 정답)
- 공통: `shading` ("lit"|"emissive"). 평면이 있는 장면(sphere_lit 제외): `plane_n (P,3) float64`, `plane_d (P,) float64` — 전체 평면(카드 제외)을 id 순으로.
- sphere_lit·two_view_parallax: `sphere_c (3,)`, `sphere_r float`. box_pair: `box_min`, `box_max` `(2,3)`(id 2,3).
- two_layer_card: `card_min`, `card_max (3,)`; `ldi_rgba (2,H,W,4) float32`, `ldi_depth (2,H,W) float32` — 층0 = 카드(알파 1 = 카드 화소, 아니면 0·rgb 0; 깊이는 카드 화소는 카드 깊이, 그 밖은 층1 깊이), 층1 = 가려진 곳까지 모두 렌더한 배경(알파 1 전부). 층0을 층1 위에 합성하면 `view.image_lin`.
- two_view_parallax: `T_01 (4,4) float64 = w2c(뷰1)·c2w(뷰0)`.
- distortion_dots: `dots_world (35,3)`, `dots_uv (35,2) = obs.project(dots_world)`, `dot_radius = 0.1`.
- color_chart: `patch_codes (24,3) uint8`, `patch_lin (24,3) float32`, `patch_rects (24,4) float64`(연속 가장자리 [u0,u1,v0,v1] = 핀홀 투영), `chart_bg_code = 40`.

## 불변식
모든 뷰 `Obs`는 FND-02 검증 통과(fx=fy, 주점 ((W−1)/2,(H−1)/2)). `instance==0 ⇔ depth==+inf`. depth>0, NaN 없음. 법선 단위(1e-5)·카메라를 향함(n_c·광선<0). `to_read_packet`은 `ReadPacket` 검증을 통과하고 `check_stage_io("S-1", {"image": ii}, {"read": rp})`가 통과한다. 장면 해시는 서로 다르다. 저장된 파일 = 재생성(해시 일치).

## 수용 기준
`tests/fnd_08/` 전부 통과(CPU, <60 s, 동결 해시 `FND-08.tests.json`): 장면별 독립 해석 렌더(깊이 rtol 2e-6, 법선 2e-6, 색 1e-5, 체커 경계 ≤0.5 % 화소 예외)와의 대조, 알려진 값(plane_frontal 깊이 ≡ 4, 지평선 행 31/32 경계, 끝벽 깊이 ≡ 10, 구 최소 깊이), 점 중심 오차 ≤0.35 px(왜곡 포함), 차트 패치 화소 비트 일치와 sRGB8 왕복, 두 뷰 깊이 전이·`plane_homography` 일치, LDI → `World` 구성, 해시·결정론·왕복·manifest·`--check` 변조 탐지·torch 미로드.

## 하류가 기대할 이음매
RD-01/RD-21 → `to_read_packet`/`to_input_image` 정답 대조, RDR-02 → plane_frontal/tilted·two_view의 `plane_homography`, WLD-01 → `two_layer_card`의 `ldi_*`, CAM-01/02·EVL-02 → 지평선·복도 소실점·`T_01`, LNS-04·EVL-04 → color_chart 패치, 렌즈 왜곡 단위 → distortion_dots. 장면을 늘릴 땐 이름을 끝에 붙이고 `FIXTURE_VERSION`을 올린다(기존 10종은 불변).
