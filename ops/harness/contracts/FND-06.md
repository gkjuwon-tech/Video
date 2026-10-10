# FND-06 기하·색·필터 수학 유틸 v0 — geom · color · pyramid

**목적.** 전 단위가 쓰는 공용 수학의 단일 원천. 좌표·색·리샘플 규약을 한 곳에 못박아 이음매 오차(반 픽셀, 감마, 경계)를 막는다.
**소유.** `buc/core/geom.py`, `buc/core/color.py`, `buc/core/pyramid.py`. 테스트 `tests/fnd_06/`(동결). 의존: numpy, scipy(허용). **torch 금지**(import 시 torch가 로드되면 실패). import 부작용 없음(전역 RNG 불변).
**공통.** 순수 함수, 입력 불변(제자리 수정 금지), 결과는 새 배열. 내부 계산은 float64. 비트 결정론(같은 입력 → 같은 바이트). 잘못된 값 `ValueError`, 잘못된 dtype `TypeError`. NaN은 조용히 바꾸지 않고 전파(`visible_mask`만 False).

## 규약 (FND-02와 동일)
세계 오른손·미터. 카메라 OpenCV(x 오른쪽, y 아래, z 앞). **픽셀 중심이 정수 좌표**, 영상 영역은 `[-0.5, W-0.5) × [-0.5, H-0.5)`, 주점 `((W-1)/2, (H-1)/2)`. 스케일 s: `u' = (u+0.5)·s − 0.5`. 포즈 4×4는 열벡터 규약. `w2c`는 세계→카메라, `c2w`는 그 역. 왜곡 `x_d = x(1 + k1 r² + k2 r⁴)`(정규화 좌표), `Obs.project`와 같은 식.

## geom.py (모두 float64 반환)
- `fov_from_focal(focal_mm, sensor_width_mm) -> float`(도, 2·atan(s/2f)), `focal_from_fov(fov_x_deg, sensor_width_mm) -> float`(범위 밖 ValueError). `intrinsics_from_fov(fov_x_deg, W, H) -> K(3,3)`, `intrinsics_from_focal(focal_mm, sensor_width_mm, W, H)`(`Theta.intrinsics`와 1e-9 일치), `fov_x_deg_from_K(K, W)`.
- `scale_intrinsics(K, sx, sy=None)`, `crop_intrinsics(K, x0, y0)`(`cx -= x0`).
- 회전: `rodrigues(rvec)->R`, `rotvec_from_matrix(R)->rvec`(θ∈[0,π], 0과 π 근처에서 안정), `is_rotation(R, atol=1e-6)->bool`, `project_to_so3(M)`(SVD, det +1 강제), `rotation_angle_deg(Ra, Rb)`(측지각, 작은 각도 정밀: atan2 형).
- 포즈: `make_pose(R,t)`, `invert_pose(T)`(해석적), `relative_pose(T_w2c_a, T_w2c_b) = T_b·T_a⁻¹`, `interp_pose(T0, T1, s)`(회전 측지 보간, 평행이동 선형), `look_at(eye, target, *, up) -> c2w`(up 필수. y축 = −up 수직 성분. eye==target·시선∥up·up=0이면 ValueError).
- 투영: `distort(xy, k1, k2)`, `undistort(xy_d, k1, k2)`(수렴 1e-10, 단조 영역), `project(K, R, t, X(...,3), k1=0, k2=0) -> (uv(...,2), z(...))`(`Obs.project`와 1e-9 일치, z≤0의 uv는 무의미), `unproject(K, uv, depth, k1=0, k2=0) -> Xc`(광축 z-깊이=depth), `unproject_world(K, R, t, uv, depth, k1=0, k2=0)`, `pixel_grid(H, W) -> (H,W,2)`(순서 `(u,v)`), `visible_mask(uv, z, W, H) -> bool`(z>0 이고 위 반열린 영역, NaN은 False).
- 평면: `plane_homography(K_src, K_dst, R, t, n, d) -> H`(평면 `n·X=d`는 src 카메라 좌표, n 단위벡터, d≠0, `X_dst = R X + t`, `H = K_dst (R + t nᵀ/d) K_src⁻¹`, `H[2,2]=1`로 정규화), `apply_homography(H, uv)`.

## color.py
- 전달함수(IEC 61966-2-1, 문턱 0.04045 / 0.0031308, 범위 밖은 홀함수 확장, 클램프 없음): `srgb_to_linear(x)`, `linear_to_srgb(x)`. float32→float32, float64→float64, 정수 dtype은 TypeError. 8비트: `srgb8_to_linear(u8)->float32`(0→0, 255→1 정확), `linear_to_srgb8(x)->uint8`(클램프, 최근접 반올림; 256 코드 왕복 동일).
- 색공간 이름: `COLOR_SPACES = ("srgb","display-p3","rec2020","linear-srgb","adobe-rgb")`(FND-02의 `unknown`은 `decode_to_linear`에서만 srgb로 취급). `eotf(x, space)`, `oetf(x, space)`(P3=sRGB 곡선, rec2020=BT.2020 α=1.09929682680944 β=0.018053968510807, adobe-rgb=γ 563/256, linear-srgb=항등).
- 행렬: 원색 xy와 D65(0.3127, 0.3290)에서 float64로 유도(반올림 표 금지). `WHITE_XYZ`(Y=1), `rgb_to_xyz_matrix(space)`, `xyz_to_rgb_matrix(space)`. `convert_linear(rgb(...,3), src, dst)`: 같은 공간은 항등(비트), 색역 밖 음수는 보존. `decode_to_linear(pixels uint8|uint16|float32 (...,3), color_space) -> float32`(정수는 /255, /65535, float32는 부호화 값, EOTF 후 Rec.709 선형으로; 그 밖 dtype TypeError, 채널≠3·모르는 이름 ValueError). `linear_to_display(x) -> float32`([0,1] 클램프 후 sRGB 부호화; `Frames.rgb`용).
- `luminance(rgb_lin)`(= sRGB 행렬 Y행), `rgb_lin_to_xyz`, `xyz_to_rgb_lin`, `xyz_to_lab(xyz, white=None)`, `lab_to_xyz(lab, white=None)`(CIE ε=216/24389, κ=24389/27, 기본 white=`WHITE_XYZ`), `rgb_lin_to_lab`, `lab_to_rgb_lin`, `delta_e_2000(lab1, lab2, kL=1, kC=1, kH=1)`(Sharma 2005, 방송, 대칭, 색상각 경계 처리).

## pyramid.py (입력 `(H,W)` 또는 `(H,W,C)`, float32|float64 외 TypeError, dtype 보존)
- `gaussian_kernel_1d(sigma, truncate=4.0)`(반경 `int(truncate·σ+0.5)`, 합 1), `gaussian_blur(img, sigma, border="reflect", truncate=4.0)`(분리형. scipy `gaussian_filter`와 같은 경계: reflect=반화소 대칭, edge, zero). 
- `pyr_down(img, border="reflect")`: 축별 `[1,3,3,1]/8`, 출력 중심 = 입력 `2i+0.5`(반화소 정렬 → `scale_intrinsics(K, 0.5)`와 일치), **출력 크기 `H//2, W//2`**(홀수면 마지막 행/열은 필터 지지로만 쓰임), 크기<2 ValueError. `pyr_up(img, out_shape=None)`: 축별 `out[2i]=¼a[i-1]+¾a[i]`, `out[2i+1]=¾a[i]+¼a[i+1]`(가장자리 복제), 크기 `2H, 2W`; `out_shape`는 각 축 `2h` 또는 `2h+1`만(`2h+1`의 덧행은 가장자리 복제).
- `gaussian_pyramid(img, levels) -> list`(길이 levels+1, 0단계 = 입력 값), `laplacian_pyramid(img, levels)`(대역 `G_l − up(G_{l+1}, G_l.shape)`, 마지막은 잔차 `G_levels`), `laplacian_reconstruct(pyr)`(완전 복원: float64 1e-12), `laplacian_low_swap(img, low, levels)`(잔차를 `low`로 교체해 복원; `low` 크기가 `G_levels`와 다르거나 levels<1이면 ValueError).
- `resize(img, out_hw, method="bilinear")`, method ∈ {nearest, area, bilinear, bicubic(a=−0.5), lanczos(a=3)}. **기준 구현 = Pillow `Image.resize`('F' 모드, 채널별, `reducing_gap=None`)**: 반화소 중심 `(x+0.5)·scale`, 축소 시 지지 폭을 scale배(안티앨리어싱), 가장자리는 지지 절단 후 재정규화, nearest는 `floor((x+0.5)·scale)`. float32에서 Pillow와 2e-5 이내. 크기 동일이면 값 그대로(복사), 상수 영상 보존, area 정수 배율은 블록 평균.
- `sample_bilinear(img, uv(...,2), border="edge") -> (...,) 또는 (...,C)`: 좌표는 픽셀 중심 정수. border `edge`(복제) 또는 `zero`(영상 밖을 0으로 두고 섞음 = scipy `grid-constant`). NaN 좌표 → NaN.

## 불변식
`project∘unproject` 항등(1e-7 px) · `Obs`/`Theta`와 수치 일치 · 영상 리샘플 무게중심 이동 = `scale_intrinsics`의 `(u+0.5)s−0.5`(2e-4 px) · 라플라시안 완전 복원 · 짝수 크기 `pyr_down`은 평균 보존 · sRGB 오라클 대비 1e-13.

## 수용 기준
`tests/fnd_06/` 전부 통과(CPU, <60 s, 동결 해시 `FND-06.tests.json`): 알려진 값(FOV 90° 가장자리 = u=W−0.5, 50 mm/36 mm = 39.5978°, sRGB 0.5, D65 흰색, BT.2407 행렬, Sharma ΔE00 14쌍), scipy·Pillow 대조, 회전/포즈/호모그래피 오라클, 경계·오류 처리, FND-02 `Theta/Obs/InputImage/Frames` 교차 검사, torch 미로드.

## 하류 이음매
FND-08(정답 합성 장면)·RDR-02(쿼드 래스터: `plane_homography`, `project`, `sample_bilinear`)·RD-01(`decode_to_linear`, `scale/crop_intrinsics`, `resize`)·LNS-04(sRGB·Lab·`luminance`)·CAM-01/06(`look_at`, `interp_pose`, `rodrigues`)·EVL-04(`delta_e_2000`, `rgb_lin_to_lab`)·TCH-05(`gaussian_blur`, 피라미드)·DAT-10(`resize` 체인)·DAT-07·INV-04가 이 모듈만 쓴다. 규약 변경은 호환 파괴이므로 이 계약서와 함께 올린다.
