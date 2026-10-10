# CAM-02 무브 레시피 · 촬영 메뉴 봉투 Π_g — 푸시인 · 슬라이더 · ±20° 오빗 · 랙

**목적.** 고정 무브 4종의 **경로 정의식**(벤치 모드 CT-I19의 "사전 등록 정의식"의 원천), 그 경로를 CAM-01 리그로 실현하는 길, 촬영 메뉴 3등급의 **기하 봉투 Π_g**와 최소 등급 판정을 만든다(pipeline.upg §3.7·F-P6·F-P12, bench I1–I5). 학습 0, 순수 수식.
**소유.** `buc/camera/moves.py`, 테스트 `tests/cam_02/`(동결). 의존: numpy, `buc.camera.rigs`(CAM-01), `to_shot`에서만 `buc.core.contracts`(함수 안에서 import). **torch 금지**, import 부작용·전역 RNG 사용 없음. `buc/camera/__init__.py`는 만들지 않는다.
**규약.** 세계 오른손·미터, 카메라 OpenCV, 포즈는 4×4 **c2w** float64. `C_0` = 입력 사진의 카메라(c2w). 피사체 점 `p_s = C_0·(0,0,z_s)`(광축 위, 깊이 `z_s`). 모든 무브는 **종점 포함** 격자: `N = round(T·fps)+1` 프레임, `t_k = k/fps`, 마지막 프레임이 종점(`T·fps`가 정수가 아니면 `ValueError`). 시간 프로파일 min-jerk `s(τ)=10τ³−15τ⁴+6τ⁵`, `τ=k/(N−1)`.

## 상수
`MOVE_NAMES=("push_in","slider","orbit","rack_focus")`, `GRADES=("safe","extended","full")`(🟢🟡🔴), `BENCH_INTENTS=("I1","I2","I3","I4","I5")`, `BENCH_DURATION_S=8.0`, `BENCH_FPS=24.0`, `M_MAX_SEMANTIC=1.5`, `M_MAX_MATERIAL=4.0`, `BOUND_TOL=1e-9`.

## 공개 API
- `MoveSpec(name, z_s, duration_s=8.0, fps=24.0, m=1.0, slide_zs=0.0, orbit_deg=0.0, round_trip=False, focus_from_m=None, focus_to_m=None, to_blur=False)` frozen dataclass, `.n_frames`. 검증 `ValueError`: 모르는 이름, `z_s·duration_s·fps·m` 비양수/비유한, 해당 없는 필드가 기본값이 아님(`m`은 push_in만, `slide_zs`는 slider만, `orbit_deg`는 orbit만(`|·|≤360`), 초점 구간·`to_blur`는 rack_focus만), rack_focus는 `focus_from_m`·`focus_to_m`(>0) 필수.
- `recipe(name, z_s, **override) -> MoveSpec`: 기본값 push_in `m=1.5`, slider `slide_zs=0.10`, orbit `orbit_deg=20.0`, rack_focus `focus_from_m=z_s`(`focus_to_m` 필수).
- `bench_spec(intent, z_s) -> MoveSpec`: I1 push_in m=1.5 · I2 orbit +20° · I3 orbit +20° `round_trip=True` · I4 orbit 90° · I5 orbit 180°, 모두 8 s·24 fps. 그 밖은 `ValueError`.
- `frame_times(spec) -> (N,)`, `ease_profile(tau, round_trip=False) -> s`(τ를 [0,1]로 클램프. 왕복은 `s(2τ)`(τ≤½) / `s(2(1−τ))`).
- `ideal_path(spec, c0=None) -> MovePath`, `rig_path(spec, c0=None, *, seed=0) -> MovePath`. `c0`는 유한 SE(3) 4×4(기본 단위행렬, 복사 보관), 아니면 `ValueError`.
- `MovePath`: `spec`, `c0`, `t (N,)`, `c2w (N,4,4)`, `focus_m (N,)`, `subject (3,)`(=p_s), `rig: str`, `traj`(Trajectory|None), `.n_frames`, `.to_shot(theta, width, height, *, seed=0)`.
- `to_shot(path, theta, width, height, *, seed=0) -> Shot`(FND-02): `K=theta.intrinsics(W,H)` 전 프레임 동일, `move=spec.name`, `rig=path.rig`, `fps=spec.fps`, 배열은 복사. `abs(theta.fps−spec.fps)>1e-9`이면 `ValueError`.
- `measure(c2w, z_s) -> PathMeasure(alpha_deg, yaw_deg, m_peak, disp_m)`, `Envelope`, `envelope(grade, *, m_max=1.5, observed_depth_m=inf)`, `min_grade(spec, *, m_max, observed_depth_m) -> str`, `min_grade_path(c2w, z_s, *, m_max, observed_depth_m) -> str`.

## 경로 정의식 (C_0 국소 좌표, `L_k = C_0⁻¹C_k`, `c2w_k = C_0·L_k`)
- **push_in**: `L=T(0,0,d)`, `d = z_s(1−1/m)·s(τ)`(광축 직선 전진, 자세 고정; 최종 배율 `z_s/(z_s−d_end)=m`; `m<1`은 후퇴).
- **slider**: `L=T(slide_zs·z_s·s(τ), 0, 0)`(+x, 자세 고정).
- **orbit**: `α=radians(orbit_deg)·s(τ)`, `L=T(P)R_y(α)T(−P)`, `P=(0,0,z_s)`. 피사체는 항상 `(0,0,z_s)`(화면 중앙), 거리 `z_s` 일정, CAM-01 `crane_jib(radius=z_s, q=(α,0), home=C_0)`의 FK와 같다. `+`각 = 카메라는 C_0 왼쪽(−x)으로 이동하며 오른쪽(+x)을 향해 회전.
- **rack_focus**: 포즈는 `C_0` 고정. 초점 `1/focus = 1/f₀ + s(τ)(1/f₁−1/f₀)`(디옵터에서 min-jerk).
- **round_trip**(모든 무브): `s`를 왕복 프로파일로 교체 → `c2w`가 시간 대칭(`c2w[k]=c2w[N−1−k]`), 마지막 = 첫 포즈(1e-12), 가운데 = 편도의 종점.
- 다른 무브의 `focus_m_k = ‖p_s − c_k‖`(피사체에 초점). 모든 경로는 `C_0`에 등변: `ideal_path(spec, c0).c2w = c0·ideal_path(spec).c2w`.

## 리그 실현
`rig_path`: push_in→`dolly`, slider→`slider`, orbit→`crane_jib(geom radius=z_s)`, 모두 `home_c2w=C_0`, `Intent`(편도 `[0,T]`, 왕복 `[0,T/2,T]`, 정지 웨이포인트, tail 0)를 `simulate`해 `traj.frame_poses(fps)`. `path.rig`는 리그 이름, `traj`는 Trajectory. rack_focus는 리그 없음(`rig="locked"`, `traj=None`, 이상 경로와 동일). `ideal_path`의 `rig`는 `"ideal"`. 벤치 크기 무브(위 TIER1 3종, m∈[0.5,4], |slide|≤0.15, |orbit|≤90°)에서 이상 경로와의 오차: 위치 ≤1e-4·z_s, 회전행렬 원소 ≤1e-4. 같은 입력 → 비트 동일(TIER1은 시드 무관).

## 봉투 Π_g
`measure`: `p_s=c2w[0]·(0,0,z_s)`, `v_k=c_k−p_s`, `f_k`=c2w의 z축. `alpha_deg=max∠(v_k,v_0)`(오빗각), `yaw_deg=max∠(f_k,−v_k)`(조준 오차; 롤은 측정 안 함), `m_peak=max ‖v_0‖/‖v_k‖`, `disp_m=max‖c_k−c_0‖`. 열이 아니거나 비유한·비회전·`z_s≤0`·카메라가 피사체 위면 `ValueError`.
`Envelope(grade, max_alpha_deg, max_yaw_deg, max_slide_zs, m_max, max_disp_m, ir_cap_mean, ir_cap_max)` frozen:
| 등급 | α | yaw | 슬라이드 | m | disp | IR cap 평균/최대 |
|---|---|---|---|---|---|---|
| safe | 20 | 20 | 0.15 z_s | m_max | 깊이 | 0.20 / 0.30 |
| extended | 90 | 90 | ∞ | m_max | 깊이 | 0.65 / 0.85 |
| full | ∞ | ∞ | ∞ | ∞ | ∞ | None / None |
`contains(measure)`: α·yaw는 `≤max+BOUND_TOL`, m·disp는 `≤max·(1+BOUND_TOL)`. `contains_path(c2w,z_s)`. `contains_spec(spec)`: `to_blur`면 safe 불가, `|slide_zs|≤max_slide_zs`, 그리고 `ideal_path(spec)`의 `measure`가 `contains`. 알려진 한계: 경로 측정은 슬라이드와 작은 오빗을 구별하지 못하므로 슬라이드 한도는 `contains_spec`만 검사한다. 불변식: safe ⊂ extended ⊂ full(`contains`·`contains_spec` 모두), `contains_spec ⇒ contains_path(ideal)`(소리), `min_grade`는 가장 작은 포함 등급. 결과: 오빗 ≤20° safe, ≤90° extended, 그 위 full; 푸시인 `m≤m_max` safe(그 위 full); 녹은 배경으로 랙(`to_blur`) extended. 얼굴·에셔·IR_info 등급화는 PRD-04의 몫이다(이 단위는 기하만).

## 수용 기준
`tests/cam_02/` 전부 통과(CPU, <60 s, `-m "not gpu"`, 동결 해시 `CAM-02.tests.json`): 상수·검증·bench 정의 / 4종 정의식(1e-12)·배율·피사체 중앙·평균 α=10°·왕복 대칭·디옵터 랙 / CAM-01 FK와의 동치·리그 추종 / 봉투 표·경계·포함·소리 / Shot 유효성·첫 포즈·torch 미로드.

## 하류 이음매
EVL-10(벤치): `bench_spec(I1..I5, z_s)`→`ideal_path`/`rig_path`→`to_shot`이 사전 등록 경로이며 DP-0 항(β_v, IR_C, 도착 모드)은 이 경로를 바꾸지 못한다. CAM-03(DP-0): `envelope(g).contains_path`가 Γ∈Π_g 검사, `ir_cap_*`가 cap_g. PRD-04(메뉴): `min_grade`/`min_grade_path`, 후보 무브는 `recipe`. WLD-21(Demand): `ideal_path`로 수요 포즈 열 생성. LNS-02: `traj.pose_at`로 셔터 하위 샘플링. 의존 추가 요청 없음.
