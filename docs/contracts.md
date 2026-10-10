# 데이터 계약 v0 (FND-02)

구현: `buc/core/contracts.py` · 상수 `CONTRACT_VERSION = "0.1.0"` · 테스트: `tests/fnd_02/` · 계약서: `ops/harness/contracts/FND-02.md`

스테이지 사이 모든 이음매의 단일 원천이다. numpy 와 표준 라이브러리만 쓰며(torch 금지) CPU 에서 import 된다.
스텁과 실물 구현은 이 타입만 보고 맞물린다. 주 버전이 다른 직렬화 데이터는 `ContractVersionError` 로 거부한다.

## 공통 규약

| 항목 | 규약 |
|---|---|
| 세계 좌표 | 오른손계, 단위 미터 |
| 카메라 좌표 | OpenCV: x 오른쪽, y 아래, z 앞(광축) |
| 픽셀 | 픽셀 중심이 정수 좌표. 영상 중심 주점은 `((W-1)/2, (H-1)/2)` |
| 영상 | 채널 끝 `(H,W,3)`, `float32`, 장면 선형(Rec.709 원색). 표시값(`Frames.rgb`)만 `[0,1]` |
| 깊이 | 광축 z-깊이(m), `float32`, `>0`. 하늘은 `+inf` 허용. NaN 금지 |
| 포즈 | `Obs.R, Obs.t` 는 세계 -> 카메라. `Shot.c2w` 는 카메라 -> 세계 (4x4) |
| 생성 | 모든 계약 클래스는 `@dataclass(kw_only=True)`. 위치 인자는 `TypeError` |
| 변환·검증 | `__post_init__` 이 배열을 정본 dtype·C-연속으로 바꾼 뒤 `validate()` 를 호출한다. 위반은 `ContractError`. 수치 필드의 NaN 은 언제나 거부 |

모든 계약 클래스의 메서드: `validate()`, `to_dict()`, `from_dict(d)`, `content_hash()`.

- `to_dict()`: JSON 직렬 가능. `"__contract__": 클래스명`, `"__v__": CONTRACT_VERSION` 을 포함한다. 배열은
  `{"__ndarray__": base64, "dtype": "<f4", "shape": [...]}` (리틀엔디안, 비트 무손실). 비유한 float 은 `{"__float__": "inf"}`.
- `from_dict(d)`: 주 버전이 다르면 `ContractVersionError`, 클래스명 불일치는 `ContractError`. 같은 주 버전의 알 수 없는 새 필드는 무시한다.
- `content_hash()`: 64자 hex(sha256). dict 키 순서 무관, list == tuple, int / float / bool 구분, 배열은 dtype(리틀엔디안) + shape +
  바이트(메모리 배치·바이트 순서 무관, 1 ulp 민감). 해시 함수는 `content_hash` 한 곳에만 있어 v1 에서 blake3 로 바꿀 수 있다.
- 모듈 함수: `content_hash(x)`, `dumps(obj) -> str`, `loads(s) -> obj` (`__contract__` 로 타입 디스패치).
- 상수: `LEVELS = ("L1","L1.5","L2a","L2b")`, `TAINT_BASES`, `STAGES`, `STAGE_SIGNATURES`.

## 타입

### Prov (출처 오염)
`taint: frozenset[str]`, `mode="engine"`, `roots: tuple[str]`(정렬·중복 제거), `ns="engine"`.
taint 원소는 `BASE` 또는 `BASE:<한정자>`, BASE 는 `USER, EXTGEN, CAPTURE, ASSET, ENGINE, SI, HOLDOUT, COMP` 중 하나.
속성 `ext`(EXTGEN), `comp`(COMP), `hold`(HOLDOUT). `Prov.merge(*ps)`: taint·roots 합집합, mode 같으면 유지 아니면 `"mixed"`,
ns 는 하나라도 `"user_vault"` 면 `"user_vault"`, 아니면 첫 값.

### Theta (촬영 파라미터)
기본값: aspect 16/9, focal_mm 35, sensor_width_mm 24.89, f_stop 4, focus_m 3, shutter_deg 180, fps 24, nd_stops 0, cpl 0, ev 0,
wb_kelvin 6500, wb_tint 0, lut "none", grain 0, halation 0, vignette 0, lens_rx "spherical_clean", atmos_beta 0,
sun_elev_deg None, sun_azim_deg None, rim 0, extra {}. 필드 추가는 `extra` 로 한다.

범위: focal (1,2000], sensor (1,100], f_stop [0.7,64], focus (0,inf], shutter (0,360], fps (0,240], nd [0,15], ev [-10,10],
wb [1500,20000], wb_tint [-1,1], cpl·grain·halation·vignette·rim [0,1], atmos_beta >= 0, sun_elev [-90,90], sun_azim [0,360).
`fov_x_deg()`, `intrinsics(W,H) -> (3,3) float64` (fx = fy = focal/sensor * W, 주점은 영상 중심).

### Obs (관측 카메라)
`K (3,3)`, `R (3,3)`, `t (3,)`, `width`, `height`, `k1=0`, `k2=0`, `mask=None (H,W) bool`, `l_obs_hard=0`, `l_obs_part=0`.
R, t 는 세계 -> 카메라(OpenCV). K 마지막 행 `[0,0,1]`, fx, fy > 0. R 직교·det +1(atol 1e-6). `0 <= l_obs_hard <= l_obs_part`.
`w2c()`, `c2w()` (4x4), `center() = -R^T t`, `project(X (N,3)) -> (uv (N,2), z (N,))`:
`x = Xc/z`, `r2 = x^2 + y^2`, 왜곡 계수 `1 + k1 r2 + k2 r2^2` 를 곱한 뒤 `u = fx x + cx`.

### InputImage
`pixels (H,W,3)` (`uint8 | uint16 | float32` 만), `color_space ∈ {srgb, display-p3, rec2020, linear-srgb, adobe-rgb, unknown}`,
`prov`, `exif={}`, `consent=True`. 속성 `h_I = content_hash(pixels)`.

### ReadPacket (S-1 출력)
`image_lin (H,W,3)`, `depth (H,W)`, `obs`, `theta_in`, `input_hash`, `prov`, `scale_s=1`(>0), `sigma_s=0`, `level="L1"`,
`depth_conf=None [0,1]`, `normals=None` (단위벡터, atol 1e-3), `esc`, `faces`, `lights`, `dyn`, `fallbacks`, `meta`.
image 는 >= 0 유한. `dyn` 키는 `{blur_vel, support_graph, matter, artic_slots}`(P2 예약) 만 허용.

### LDILayer, FarField, World
- `LDILayer(rgba (H,W,4) float32 [0,1] 직선 알파, depth (H,W) float32 > 0)`
- `FarField(rgb (S,S,3) float32 >= 0 유한)` 팔면체 매핑
- `World` (STRATA-lite-min v0): `obs`, `layers: tuple[LDILayer]` (1장 이상), `theta_in`, `level`, `read_hash`, `prov`,
  `far_field=None`, `meta={}`, `schema="STRATA-lite-min/v0"`. 층은 obs 크기이고 앞 -> 뒤 정렬: 두 층 모두 알파 > 1e-3 인 픽셀에서
  `depth[i] <= depth[i+1] * (1 + 1e-6)`.

### Shot
`c2w (N,4,4)`, `t (N,)`, `K (N,3,3)`, `focus_m (N,)`, `width`, `height`, `fps`, `theta`, `move="static"`, `rig="locked"`, `seed=0`.
`t[0] = 0`, 간격 `1/fps` (atol 1e-9), 회전 직교·det +1, 마지막 행 `[0,0,0,1]`, `focus_m > 0`.
`n_frames`, `duration_s = N / fps`, `first_pose_matches(obs, rot_tol_deg=0.1, trans_tol_m=1e-3)`.

### Frames
`rgb (N,H,W,3) float32 [0,1]` (표시값), `fps`, `aov={}` (각 배열의 첫 축이 N), `shot_hash=""`, `world_hash=""`.

### KPI
`ffd, ffg, de00, ir_geo, ir_area, ir_info, m_eff, esc, cheat_share, tfu, gpu_h, wall_s` (모두 `Optional[float]=None`),
`ladder_step`, `proof_of_shot`, `extra`. ffd, ffg, de00, esc, tfu, gpu_h, wall_s >= 0. ir_*, cheat_share ∈ [0,1]. m_eff > 0. NaN 금지.

### StageIO
`stage ∈ STAGES`, `impl_kind ∈ {stub, real}`, `impl_name`, `impl_version`, `inputs: {이름: hash}`, `outputs: {이름: hash}`,
`params={}`, `wall_s=0`, `fallback=None`.
`cache_key = content_hash((CONTRACT_VERSION, stage, impl_kind, impl_name, impl_version, inputs, params))` — outputs·wall_s·fallback 은 제외.

## 스테이지 이음매

`STAGE_SIGNATURES[stage] = (입력 {이름: 타입}, 출력 {이름: 타입})`, v0 은 다음 6개만:

| 스테이지 | 입력 | 출력 |
|---|---|---|
| `S-1` | image: InputImage | read: ReadPacket |
| `S2` | read: ReadPacket | world: World |
| `S8` | world: World, read: ReadPacket | shot: Shot |
| `S8.5` | world: World, read: ReadPacket | world: World |
| `S9-10` | world: World, shot: Shot | frames: Frames |
| `S11` | frames, read, world, shot | kpi: KPI |

`check_stage_io(stage, inputs, outputs)`: 서명 없는 스테이지, 키 누락·초과, 타입 불일치는 `ContractError`. 교차 검사:

- S-1: `read.input_hash == image.h_I`, obs 크기 == 영상 크기, `read.prov.taint ⊇ image.prov.taint`
- S2: `world.read_hash == read.content_hash()`, world.obs 해시 == read.obs 해시, taint 포함
- S8: `shot.first_pose_matches(read.obs)`
- S8.5: 출력 `world.obs` == 입력 `world.obs`
- S9-10: frames 의 N·H·W == shot

`make_stage_io(stage, impl_kind, impl_name, impl_version, inputs, outputs, params=None, wall_s=0.0, fallback=None)`:
`check_stage_io` 를 먼저 돌리고 각 값의 `content_hash()` 로 `StageIO` 를 채운다.

## 하류 이음매

RD-21 -> ReadPacket · WLD-01 -> World(STRATA 내부는 자유) · LNS-01 -> Theta · CAM-02 -> Shot · RDR-02/06 -> Frames ·
PRD-01 -> StageIO·check_stage_io·cache_key · PRD-03 스텁은 STAGE_SIGNATURES 만 맞춘다 · PRD-05/EVL -> KPI.
