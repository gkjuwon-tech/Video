# CAM-04 얼굴 봉투 규칙 — F0–F3 · α_σ · 미관측 얼굴면 5%

**목적.** 인물 입력의 DP-0 하드 제약(pipeline.upg §10, NC-P8, F-P9, CT-I18): 카메라 경로가 얼굴 관측 각도 봉투와 "미관측 얼굴면 노출" 한도를 지키는지 판정한다. 한 장으로는 보지 않은 얼굴면을 만들 수 없으므로 어기는 경로는 폐기한다(`FaceViol_σ`). 학습 0, 순수 수식.
**소유.** `buc/camera/face_envelope.py`, 테스트 `tests/cam_04/`(동결). **numpy와 표준 라이브러리만**(torch·buc 다른 모듈 import 금지, import 부작용·전역 RNG 없음). `buc/camera/__init__.py`는 만들지 않는다. 의존 FND-02는 규약(좌표·c2w)만 쓴다. 모든 공개 이름은 `__all__`에 둔다.
**규약.** 세계 오른손·미터, 포즈는 4×4 **c2w** float64(CAM-02·FND-02 Shot과 같음). 카메라 위치 = `c2w[:3,3]`. 입력 검증 실패는 `ValueError`.

## 얼굴 좌표계
`FaceSpec(*, center, front, up, h_f, sigma_z=0.0, head_radius_m=0.10)` frozen kw-only dataclass. `center`(머리 중심)·`front`(얼굴이 향한 방향)·`up`은 (3,) float64 복사본으로 저장: `front` 정규화, `up`은 front에 직교화 후 정규화, `right = cross(up, front)`(속성). `h_f`∈(0,1] = C_0에서 얼굴 높이/프레임 높이, `sigma_z`≥0(m, inf 허용) = 얼굴 깊이 불확실도, `head_radius_m`>0. 비유한·영벡터·up∥front는 `ValueError`. `to_dict()`(JSON 가능: 키 center,front,up,h_f,sigma_z,head_radius_m)와 `from_dict(d)`(필수 center,front,up,h_f; 모르는 키·누락은 `ValueError`; 왕복 오차 ≤1e-12).
카메라 위치 `c`의 얼굴 각: `v=c−center`, `x=v·right, y=v·up, z=v·front`, **yaw=atan2(x,z)**, **pitch=atan2(y,hypot(x,z))**(도), `dist=‖v‖`. `face_angles(face, cam_pos) -> (yaw, pitch, dist)`: (3,) 또는 (N,3) 입력, 같은 모양 출력. `dist ≤ head_radius_m`이면 `ValueError`.

## 규칙 (상수는 모듈에 같은 이름으로 노출)
`GRADES=("safe","extended","full")`, `CATEGORIES=("F0","F1","F2","F3")`, `EPS_F=0.03`, `W_F_M=0.15`, `BACK_YAW_DEG=120`, `H_F1_MAX=0.03`, `H_F2_MAX=0.15`, `FACE_CAP_HALF_DEG=45`, `HEAD_RADIUS_M=0.10`, `M_TOL=0.02`, `ANGLE_TOL_DEG=1e-6`, `UNOBS_TOL=1e-9`, 표 `YAW_TABLE_DEG={F0:inf,F1:inf,F2:30,F3:15}`, `PITCH_TABLE_DEG={F0:inf,F1:inf,F2:15,F3:10}`, `M_MAX={F0:inf,F1:1.5,F2:1.0,F3:1.0}`, `UNOBS_MAX={F0:.05,F1:.03,F2:.05,F3:.05}`.
- `alpha_sigma_deg(sigma_z_m, *, eps_f=EPS_F, w_f_m=W_F_M)` = degrees(asin(min(1, eps_f·w_f/σ))); σ=0→90, σ=inf→0, σ<0·NaN·인자≤0은 `ValueError`. (1 cm→26.7°, 2 cm→13.0°, 3 cm→8.63°)
- `face_category(h_f, observed_yaw_deg=0.0, *, has_face=True) -> str`: 얼굴 없음 또는 감은 |관측 yaw|>120° → F0(120 정확히는 아님); h_f≤3% F1; ≤15% F2; 그 위 F3. h_f∉(0,1]은 `ValueError`(has_face일 때).
- `limits(category, sigma_z_m=0.0, *, margin_deg=0.0) -> FaceLimits(category, yaw_deg, pitch_deg, m_max, unobs_max, grades, alpha_sigma_deg)` frozen. F2·F3의 `yaw_deg = max(0, min(표, α_σ) − margin_deg)`(margin은 검출 yaw MAE 여유, yaw만 줄임), F0·F1은 무제한. `grades`: F0·F1 `GRADES`, F2 `("safe","extended")`, F3 `("safe",)`. F3⊂F2⊂F1⊆F0(모든 한계), σ가 클수록 yaw는 줄기만 한다.

## 미관측 얼굴면 노출
머리 = 반지름 r의 구, **얼굴면 = 구 위 `front` 중심 반각 45° 캡**. 카메라 `c`에서 캡 점 `n`(단위 법선)이 보인다 ⇔ `n·(c−center) > r`(거리·머리 크기가 반영되며, 멀리서는 반구). `unobserved_exposure(face, cam0_pos, cam_pos) -> float | (N,)` = 캡 면적 중 `cam_pos`에서는 보이고 `cam0_pos`(입력 카메라 C_0)에서는 안 보이는 비율 ∈[0,1]. 절대 오차 ≤0.005(캡 안 ≥4096점 결정론 준균일 표본을 얼굴 기저에서 만들 것 → 강체 이동 불변), 같은 입력 → 비트 동일, 두 카메라 위치가 머리 안이면 `ValueError`.

## 경로 판정
`evaluate_path(face, c2w, *, c0=None, margin_deg=0.0, grade=None) -> FaceVerdict`. `c2w` (N,4,4), N≥1, 유한, 마지막 행 [0,0,0,1]; `c0`(4×4, 기본 `c2w[0]`; 도착 샷은 명시) 위치가 관측 카메라. 프레임마다: Δyaw=|wrap(yaw_k−yaw_0)|, Δpitch=|pitch_k−pitch_0|, 배율 `m_k=d_0/d_k`, `h_f_k=h_f·m_k`. 범주 = `face_category(max h_f_k, yaw_0)`(경로 전체 h_f). 위반(고정 순서) `"yaw"`: max Δyaw>`limits.yaw_deg`+ANGLE_TOL; `"pitch"`; `"push"`: max m>`m_max`+M_TOL; `"unobserved"`: 프레임별 노출의 **최대**>`unobs_max`+UNOBS_TOL; `"grade"`: `grade`가 범주의 `grades`에 없음(grade 인자가 `GRADES` 밖이면 `ValueError`). `FaceVerdict`(frozen): `ok, category, yaw_obs_deg, h_f_max, m_peak, yaw_dev_deg, pitch_dev_deg, unobs_max, limits, violations: tuple[str], allowed_grades`(모두 파이썬 float/tuple/str, `==` 비교 가능).
`evaluate_faces(faces, c2w, *, c0=None, margin_deg=0.0, grade=None) -> FaceReport(ok, verdicts, violations, allowed_grades)`: `faces`는 `FaceSpec`·`to_dict()` 딕셔너리(=`ReadPacket.faces` 원소 형태)의 열; `violations`=합집합(고정 순서), `allowed_grades`=교집합(`GRADES` 순서); 빈 열은 ok·전 등급. `face_viol_sigma(faces, c2w, **kw) -> bool` = `not ok`.
`orbit_budget_deg(face, cam0_pos, direction, *, margin_deg=0.0, step_deg=0.25) -> float`: 얼굴 `up` 축 둘레 거리·pitch 일정 수평 오빗을 `direction=±1` 쪽으로 `step_deg` 격자로 훑어, yaw 한계(표·α_σ·margin)와 노출 한도를 처음 넘기 직전의 각도(≤180)를 돌려준다. 같은 오빗을 `evaluate_path`에 넣으면 이 각도까지는 ok, +1° 넘으면 위반. 뒷모습은 대략 ±55° (거리 1.5–8 m에서 48–62°).

## 불변식
회전·평행이동 불변, 좌우(±yaw) 대칭, 정면 관측이면 노출 0, 노출은 관측에서 멀어질수록 단조 증가, 입력 배열 불변경.

## 수용 기준
`tests/cam_04/` 전부 통과(CPU, <60 s, 동결 해시 `CAM-04.tests.json`): 상수·α_σ 표·범주 경계·한계표 포함 관계 / 노출 해석해(원거리 0, 1/2, 1)와 독립 구적 대조 / 경로 판정(yaw·pitch·push·노출 3%/5%·margin·grade·도착 샷·다중 얼굴) / orbit_budget와 evaluate_path 일관성 / FaceSpec 검증·직렬화 / torch 미로드.

## 하류 이음매
CAM-03(DP-0): `face_viol_sigma(faces, Γ.c2w, c0=C_0)`가 보상의 −∞ 항(FaceViol_σ). PRD-04(메뉴): `evaluate_faces(...).allowed_grades`·`orbit_budget_deg`로 열리는 등급과 각도 예산 표시(봉투로 줄면 "봉투 적용" 꼬리표). RD-15(얼굴 맵)는 `FaceSpec.to_dict()`를 `ReadPacket.faces`에 넣는다(σ_z=MRH σ̂ ⊕ 머리 모형 잔차, yaw 검출 MAE는 `margin_deg=2×MAE`로). EVL-14 X-FACE-Z는 `alpha_sigma_deg`와 실측 붕괴 각을 비교. 알려진 한계: 얼굴면은 캡 모델(뒤통수·귀는 제외, 의미 패스/링 교사 몫), 다중 얼굴은 독립 판정. 의존 추가 요청 없음.
