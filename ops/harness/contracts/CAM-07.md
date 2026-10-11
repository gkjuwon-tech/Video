# CAM-07 충돌 · 클리어런스 · 가시성 유틸 — 장애물 SDF · 경로 클리어런스 · 차폐/프러스텀/깊이 가시성

**목적.** DP-0(CAM-03)의 `−∞·collision`과 β₄·Clearance 항, PRD-04 메뉴의 "푸시인 가능 거리", 피사체 가림 판정에 쓰는 **순수 기하 질의**(PARADIGM-v2 §7 보상식). 세계 표현에 묶이지 않도록 장애물은 프리미티브 3종으로 받는다. 학습 0.
**소유.** `buc/camera/clearance.py`, 테스트 `tests/cam_07/`(동결). **numpy와 표준 라이브러리만**(torch·다른 buc 모듈 import 금지, import 부작용·전역 RNG 없음, 입력 불변, 결정론). `buc/camera/__init__.py`는 만들지 않는다. 모든 공개 이름은 `__all__`. 의존 FND-02는 규약만 쓴다.
**규약.** 세계 오른손·미터, 카메라 OpenCV(x 오른쪽, y 아래, z 앞), 포즈 4×4 **c2w**, 카메라 위치 = `c2w[:3,3]`. 영상 픽셀 중심은 정수 좌표, 영상 범위 `u∈[−0.5, W−0.5]`, `v∈[−0.5, H−0.5]`. 깊이는 광축 z-깊이(`+inf` 허용, NaN 금지). 입력 검증 실패는 `ValueError`(점 배열은 마지막 축 3, 유한).

## 상수
`CAMERA_RADIUS_M=0.10`, `COMFORT_M=0.5`, `SURF_TOL=1e-6`(차폐 판정에서 선분 양 끝 이만큼은 무시), `PATH_TOL_M=1e-4`, `OBSTACLE_TYPES=("sphere","box","halfspace")`.

## 장애물 (frozen, kw-only, `eq=False`; 배열은 float64 복사 보관)
- `Sphere(center, radius>0)`, `Box(center, half_extents>0, rotation=None)`(`rotation` 3×3 열 = 상자 축의 세계 방향, 직교정규·det=+1, 기본 단위행렬), `HalfSpace(normal, offset)`(고체 = `n·x ≤ offset`, 저장 시 `‖n‖=1`로 정규화하며 offset도 같이 나눔; 바닥·벽용).
- `.signed_distance(points) -> ndarray` : `(...,3)` → `(...)`. **정확한 부호거리**(밖 양수, 안 음수, 표면 0). 장면 SDF는 1-립시츠다(경로 하한 증명의 전제).
- `.blocks_segment(a, b)`, `.to_dict()`(JSON 가능; 키 `type`+필드, 상자는 `rotation` 3×3 포함), `obstacle_from_dict(d)`(필수 키 누락·모르는 키·모르는 type은 `ValueError`; 상자 `rotation`은 선택).

## 공개 함수
- `camera_positions(x) -> (N,3)`: `(N,4,4)`·`(4,4)`·`(N,3)`·`(3,)` 입력(N≥1, 유한; 4×4는 마지막 행 [0,0,0,1]) → 복사본.
- `clearance(obstacles, points) -> float | ndarray` : 장면 SDF = 각 장애물 SDF의 최솟값(장애물 없음 = `+inf`). `(3,)` 입력이면 파이썬 float.
- `path_clearance(obstacles, path, *, tol=PATH_TOL_M) -> PathClearance` : `path`는 `camera_positions`가 받는 형태, **프레임 점을 잇는 폴리라인 전체**(프레임 사이 얇은 벽 터널링을 잡는다)의 최소 clearance. `PathClearance(min_clearance, lower_bound, at_frame, point, tol)`: `lower_bound ≤ 진짜 최소 ≤ min_clearance`, `min_clearance − lower_bound ≤ tol`, `point` = `min_clearance`를 얻은 3-점(튜플, `clearance(point)==min_clearance`), `at_frame` = 그 점의 소수 프레임 인덱스(`[0,N−1]`). 장애물 없음: `inf, inf, 0.0, path[0]`. `tol≤0`·비유한이면 `ValueError`. (1-립시츠 구간 분할로 `(f(a)+f(b)−길이)/2` 하한을 쓰면 된다.)
- `collision_free(obstacles, path, *, camera_radius=CAMERA_RADIUS_M, margin=0.0, tol=PATH_TOL_M) -> bool` : **소리(sound)**. `lower_bound ≥ camera_radius + margin`일 때만 True(True이면 진짜 최소 ≥ 임계). `camera_radius`·`margin` < 0이면 `ValueError`.
- `free_distance(obstacles, origin, direction, *, radius=CAMERA_RADIUS_M, max_dist=100.0, tol=1e-6) -> float` : 광선을 따라 clearance ≥ radius를 유지하며 갈 수 있는 거리(구면 추적, 보수적: 진짜 값 이하, 정면 충돌은 `tol` 이내). 이미 위반이면 0.0, 상한 `max_dist`. `direction` 영벡터·비유한, `radius<0`, `max_dist≤0`은 `ValueError`.
- `segment_visible(obstacles, a, b) -> bool | ndarray` : a→b 선분이 어떤 고체의 내부도 지나지 않으면 True. 양 끝 `SURF_TOL`은 무시하므로 **바깥에서 본 표면 점은 보이고**, 뒷면 점·고체 속 점·고체 속 카메라는 가려진다. a,b는 브로드캐스트(`(…,3)`), 대칭. 스칼라 입력은 파이썬 bool.
- `visibility_matrix(obstacles, c2w, points, *, K=None, size=None, margin_px=0.0, near=1e-3) -> (N,M) bool` : 프레임 i에서 점 j가 보임 ⇔ 카메라 z>near **그리고**(K,size=(W,H) 주어지면) 투영이 영상 안쪽 `margin_px` 여유 안 **그리고** `segment_visible`. `K`와 `size`는 둘 다 주거나 둘 다 없어야 한다. `c2w`는 (N,4,4) 또는 (4,4), 유한·회전(1e-6) 아니면 `ValueError`. `visible_fraction(...)` = 행 평균 `(N,)`(같은 인자).
- `depth_visible(points, K, c2w, depth, *, rel_tol=0.01, abs_tol=0.0) -> (M,) bool` : 렌더 깊이 버퍼 z-버퍼 검사. 점이 카메라 앞에서 영상 안(`floor(u+0.5)` 최근접 화소)에 투영되고 `z ≤ depth·(1+rel_tol)+abs_tol`이면 True. `c2w`는 단일 포즈, `depth (H,W)` NaN이면 `ValueError`.
- `clearance_score(min_clearance, *, radius=CAMERA_RADIUS_M, comfort_m=COMFORT_M) -> float | ndarray` : `x=clip((c−radius)/comfort_m,0,1)`, `3x²−2x³`(β₄ 입력, 단조, `inf`→1). `comfort_m≤0`·`radius<0`·NaN은 `ValueError`.
- `evaluate_path(obstacles, c2w, *, camera_radius, margin, comfort_m, subject_points=None, K=None, size=None, min_visible=0.0, tol) -> ClearanceReport(ok, collision, min_clearance, lower_bound, worst_frame, score, min_visible_fraction)` : `collision = not collision_free`; `subject_points`가 있으면 `min_visible_fraction`=프레임별 `visible_fraction`의 최솟값(없으면 None), `ok = not collision and min_visible_fraction ≥ min_visible`; `score = clearance_score(min_clearance, radius=camera_radius, comfort_m)`.

## 불변식
SDF 1-립시츠·회전/평행이동 등변 / `lower_bound ≤ 진짜 최소 ≤ min_clearance` / `collision_free`는 소리 / 프레임 점이 모두 멀어도 프레임 사이 관통은 충돌 / `segment_visible(a,b)==segment_visible(b,a)` / 입력 배열 불변경, 같은 입력 → 비트 동일, 전역 RNG 불변.

## 수용 기준
`tests/cam_07/` 전부 통과(CPU, <60 s, 동결 해시 `CAM-07.tests.json`): 상수·검증·직렬화 / 구·상자·반공간 SDF(해석해 1e-12, 상자는 표면 표본 브루트포스 대조) / 경로 clearance(구 해석해, 조밀 표본, 터널링, 하한·상한 관계) / 충돌 임계·소리성·`free_distance` / 선분 가시성(표면 규칙, 조밀 표본 대조) / 프러스텀 경계(±1e-3 px 근방)·차폐 스윕 / 깊이 버퍼 / 점수 / 보고서 / torch 미로드.

## 하류 이음매
CAM-03(DP-0): `collision_free(obstacles, Γ.c2w)`가 보상의 −∞ 항, `evaluate_path(...).score`가 β₄, `visible_fraction`이 피사체 가림 페널티. CAM-02 `ideal_path(...).c2w` 그대로 `path`에 넣는다. PRD-04: `free_distance`로 푸시인·후퇴 한도 표시. WLD 쪽은 장애물을 `to_dict` 형태(JSON)로 넘기고 이 단위가 `obstacle_from_dict`로 받는다. 알려진 한계: 경로는 프레임 점의 직선 연결이며(프레임 사이 곡률 무시), 카메라는 구(반지름)로 근사, 카메라 방향·렌즈 프러스텀 충돌은 보지 않는다. 의존 추가 요청 없음.
