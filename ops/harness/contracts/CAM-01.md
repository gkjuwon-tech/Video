# CAM-01 리그 1 kHz ODE v0 — 14종 · min-jerk · OU 외란

**목적.** 카메라 리그 14종을 1 kHz 상미분방정식으로 CPU에서 시뮬해 "의도 → 물리 → 포즈" 분해의 물리층을 만든다(`docs/lenses/03-cinematography.md` §3.5–3.6, PARADIGM-v2 §7.8). 학습 0, 순수 수식. 먼저 슬라이더·달리·오빗(`crane_jib`) 3종(`TIER1`)이 정확해야 한다. 핸드헬드는 OU 잡음 근사.
**소유.** `buc/camera/rigs.py`, 테스트 `tests/cam_01/`(동결). 의존: numpy, scipy(허용), 필요하면 FND-06 `buc.core.geom`. **torch 금지**(import 시 로드되면 실패), import 부작용·전역 RNG 사용 없음. `buc/camera/__init__.py`는 만들지 않는다(namespace, lens·render와 동일).
**규약.** 세계 오른손·미터, 카메라 OpenCV(x 오른쪽, y 아래, z 앞), 포즈는 4×4 **c2w** float64(FND-06). 리그의 일반화 좌표 `q`(m 또는 rad)만 포즈를 정한다(홀로노믹: 포즈 = `home_c2w @ L(q)`, 제약 위반 불가).

## 상수 · 순수 함수
`DT=1e-3`, `RATE_HZ=1000`, `V_EPS=1e-3`, `PARAM_KEYS=("mass","damping","quad_drag","stiffness","coulomb","w2","kp","ki","kd","ff_gain","u_max","ou_sigma","ou_tau")`, `TIER1=("slider","dolly","crane_jib")`, `RIG_NAMES`(14, 순서 고정) = tripod_fluid, dolly, slider, crane_jib, techno_crane, steadicam, gimbal_3axis, shoulder_handheld, cine_drone, fpv, cablecam, car_mount, motion_control, underwater.
- `min_jerk_profile(tau) -> (s, ds, d2s)` 정규화 시간 τ(스칼라·배열, [0,1]로 클램프): `s=10τ³−15τ⁴+6τ⁵`와 τ에 대한 1·2계 도함수.
- `min_jerk_path(times, waypoints, t) -> (x, v, a)`: `times (K,)`·`waypoints (K,n)`, 구간마다 min-jerk(웨이포인트마다 정지, `v=a=0`). `t`가 (M,)이면 각각 (M,n), 스칼라면 (n,). 범위 밖은 끝점 유지(`v=a=0`).
- `power_law_speed(curvature, gain=1.0)`: `v = gain·κ^(−1/3)`(각속도 `κ^(2/3)`). κ≤0이 하나라도 있으면 `ValueError`.
- `ou_noise(n, dt, tau, sigma, rng: np.random.Generator) -> (n,)`: 정확 이산화 OU(`a=e^(−dt/τ)`, 정상 분산 σ², 첫 표본도 정상분포, σ=0이면 영). `n≥1, dt>0, tau>0, sigma≥0` 아니면 `ValueError`. 같은 `rng` 상태 → 같은 출력.
- `Intent(times, waypoints, tail_s=0.0)` frozen dataclass: `times (K≥2,)` 엄격 증가·`times[0]==0`, `waypoints (K,n_q)`, 유한, `tail_s≥0`, 아니면 `ValueError`; 입력을 복사해 둔다. `.duration_s = times[-1]+tail_s`, `.reference(t) -> (q, qd, qdd)`(= `min_jerk_path`), `Intent.point_to_point(q0, q1, duration_s, tail_s=0.0)`.

## 리그 ODE (자유도마다 독립, RK4, 스텝 1 ms)
```
m q̈ = u − c q̇ − c2 |q̇| q̇ − k q − f_c tanh(q̇/V_EPS) − m·w2·sin(q)      (평형 q=0)
```
파라미터 `PARAM_KEYS`는 각각 자유도별 길이 `n_q` 벡터(스칼라는 브로드캐스트): mass(m>0), damping(c), quad_drag(c2), stiffness(k), coulomb(f_c), w2(진자 ω²), kp·ki·kd·ff_gain·u_max(>0)·ou_sigma(≥0)·ou_tau(>0). 추가로 스칼라 `delay_s≥0`. 위반·모르는 키·잘못된 길이는 `ValueError`.
**제어**(스텝 k, ZOH: 한 스텝 동안 u 일정): `u = clip(ff_gain·u_ff(k) + kp·e_d + kd·ė_d + ki·I, ±u_max) + d_k`. `u_ff` = 기준 궤적의 역동역학(위 식 우변에 `q_ref, q̇_ref, q̈_ref`를 넣은 값). `e_d, ė_d` = 인덱스 `j=max(k−round(delay_s/DT), 0)`의 `q_ref−q`, `q̇_ref−q̇`(지연 피드백, 처음 δ 동안은 e(0)). `I_{k+1}=I_k+e_d·DT`, `I_0=0`. `d_k`는 `ou_sigma>0`인 자유도의 OU 외란(`ou_tau`), `np.random.default_rng(np.random.SeedSequence([seed, 0xCA01]))` 하나에서 자유도 순서로 뽑는다. 같은 시드 → 비트 동일, 전역 RNG 불변.

## 클래스
- `make_rig(name, *, params=None, geom=None, home_c2w=None) -> Rig`. 모르는 이름 `KeyError`. `default_params(name) -> dict`(`PARAM_KEYS`+`"delay_s"`, 값은 float 튜플·float, 호출마다 새 객체).
- `Rig`: `name`, `n_q`, `q_scale (n_q,)`(전형적 최대 이동량, 테스트·하류가 의도를 만들 때 쓴다), `params`(dict, 값 `(n_q,)` 배열·`delay_s` float), `geom`(`axis` 단위 3벡터, `radius` m>0; 이외 키 `ValueError`, 0벡터 축·radius≤0 `ValueError`), `home_c2w`(4×4 유한 SE(3), 복사 보관; 기본 단위행렬).
  - `fk(q) -> c2w`: `q (...,n_q)` → `(...,4,4)`, `home_c2w @ L(q)`. `q=0`이면 홈.
  - `simulate(intent, *, seed=0, q0=None, qd0=None) -> Trajectory`: 기준 = `intent.reference`, 길이 `round(duration_s/DT)+1`, `q0` 기본 `waypoints[0]`, `qd0` 기본 0. 자유도·형상 불일치·비유한 `ValueError`.
  - `simulate_force(force, duration_s, *, q0=None, qd0=None, seed=0, disturbance=False) -> Trajectory`: 개루프. `force`는 `(n_q,)`/스칼라 또는 `t -> (n_q,)` 콜러블(스텝마다 격자 시각 `t_k=k·DT`에서 한 번 호출). 제어·ff·지연 없음, OU 외란은 `disturbance=True`일 때만. `duration_s>0` 아니면 `ValueError`. `q_ref=None`.
- `Trajectory`(dataclass): `rig`, `t (N,)`, `q`·`qd`·`q_ref`·`u (N,n_q)`(`u[k]`는 [t_k,t_{k+1}]에 인가, 마지막 행은 직전 복사), `pose (N,4,4)`, `dt=DT`. `q_at(times)`, `pose_at(times)`: (q,q̇) 3차 에르미트 보간 후 FK(범위 밖 `ValueError`). `frame_poses(fps) -> (M,4,4)`: `t_k=k/fps`, `M=floor(T·fps+1e-9)+1`.

## 운동학 L(q) (홈 카메라 좌표계; `R_y,R_x,R_z`는 표준 회전행렬, `T`는 평행이동)
- slider `(s)`: `T(s·axis)`, axis 기본 (1,0,0). dolly `(s)`: 같음, axis 기본 (0,0,1). `axis`는 정규화해 저장.
- **crane_jib `(φ,ψ)` / techno_crane `(φ,ψ,ΔL)`**: 피벗 `P=(0,0,R)`(R=`geom["radius"]`, 기본 3.0), `Rot=R_y(φ)R_x(ψ)`, 반지름 `r=R(+ΔL)`: 회전 `Rot`, 위치 `P + Rot·(0,0,−r)`. 항상 피벗을 바라보고 거리 `r`(오빗; φ=±20°는 CAM-02의 ±20° 오빗).
- tripod_fluid `(pan,tilt)`: `R_yR_x`. gimbal_3axis `(pan,tilt,roll)`: `R_yR_xR_z`. steadicam `(x,z,pan,roll)`: `T(x,0,z)R_yR_z`. shoulder_handheld·fpv·motion_control `(x,y,z,yaw,pitch,roll)`: `T(x,y,z)R_yR_xR_z`. cine_drone·underwater `(x,y,z,yaw)`: `T(x,y,z)R_y`. cablecam `(s,θ)`: `T(s·axis)R_z(θ)`, axis 기본 (1,0,0). car_mount `(s,h)`: `T(s·axis + h·ŷ)`, axis 기본 (0,0,1).

## 기본 파라미터 요구(구현자가 값을 정하되 아래를 지킨다)
모든 리그 안정·유한. ff_gain=1. 외란 없는 리그(`ou_sigma` 전부 0)는 `Intent.point_to_point(0, 0.3·q_scale, 1.2 s, tail 1.8 s)`에서 `|q−q_ref|≤0.01·q_scale`, 종점 오차 ≤0.005·q_scale. 외란 리그는 RMS 오차 ≤0.1·q_scale. TIER1(slider, dolly, crane_jib)·motion_control은 `ou_sigma=0`, `delay_s=0`. slider `q_scale` 0.5–2 m·질량 작게, dolly `q_scale` 2–10 m·질량 ≥10×slider, crane_jib `q_scale[0]≥20°`. shoulder_handheld `delay_s`∈[0.15,0.25], `ou_sigma>0`. steadicam·cablecam `w2>0`(진자), tripod_fluid `coulomb>0`, cine_drone·underwater `quad_drag>0`, car_mount `ou_sigma>0`(노면).

## 수용 기준
`tests/cam_01/` 전부 통과(CPU, <60 s, `-m "not gpu"`, 동결 해시 `CAM-01.tests.json`): min-jerk·멱법칙·OU 통계 / 질량-감쇠(1e-9)·2차 항력·스프링·진자(에너지 1e-9, 타원적분 주기 2e-4)·쿨롱 해석해 / ff·PD·I항·지연·포화 / 14종 FK 표 일치(1e-12)·레일·오빗 반지름 1e-9 / 14종 시뮬 유한·결정론·추종 / 에르미트 보간(1e-9).

## 하류 이음매
CAM-02(무브): `make_rig(...).simulate(Intent...).frame_poses(fps)` 로 c2w 포즈 열을 얻는다. 렌즈·셔터(LNS-02)는 `pose_at`으로 노출 구간을 하위 샘플링한다. MDL-19(Rig-ID)·DAT-13/14는 `RIG_NAMES` 인덱스를 클래스 레이블로, `default_params`·`params`를 회귀 목표로 쓴다. 이 단위는 렌더·학습을 하지 않는다. 의존 추가 필요 없음(요청 없음).
