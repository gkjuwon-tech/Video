# DAT-01 절차 장면 GT 렌더 파이프 (G-buffer 12ch · 디노이저 OFF)

**목적.** MRH(M3)·G-Adapter(M4) 학습 GT의 원천. INV-01 `SceneProgram`(= 라벨)을 카메라로 찍어 선형 HDR rgb와 **G-buffer 12채널**(M4 입력 `[B,12,H,W]`)·깊이·sem·inst·광류를 만든다. 셰이딩은 해석식(점광원 람베르트), 학습된 디노이저(OIDN·OptiX AI)와 모든 화소 간 필터는 **금지**.
**소유.** `buc/data/synth_render.py`(네임스페이스 패키지, `__init__.py` 없음). 테스트 `tests/dat_01/`(동결 `DAT-01.tests.json`). 의존: INV-01(`SceneProgram`, `raycast`, 상수 `SEM_*`/`MATERIALS`/`SLOT_CLASSES`/`FURNITURE_CLASSES`, `sem_id`), FND-02(`Prov`), FND-06(`geom`, `color`). RDR-02/05는 규약과 이음매(아래). numpy·scipy 외 금지(`scipy.ndimage/signal`, cv2, skimage, diffusers, transformers, `buc.teacher/models/train` import 금지). GPU 없이 import·실행, import 부작용·전역 RNG 변경 금지, 입력 불변.

## 규약
세계 오른손·미터·z 위. 카메라 OpenCV(`c2w`), 픽셀 중심 정수. depth = 광축 z-깊이(미스 `+inf`). 법선 = **카메라 좌표** 단위벡터, 카메라를 향함(`n·X_c<0`). 광류 `(du,dv)=π_B(X)−π_A(X)`(RDR-05와 동일, X는 A의 화소 중심 광선이 맞은 점). rgb·albedo는 선형 Rec.709.

## 공개 API
```python
N_GBUF = 12
GBUF_CHANNELS = ("log_depth","normal_x","normal_y","normal_z","albedo_r","albedo_g","albedo_b",
                 "sem_structure","sem_opening","sem_furniture","sem_accessory","level_mask")
SEM_GROUP_NAMES = ("structure","opening","furniture","accessory")
ALBEDO: dict[str, (r,g,b)]               # MATERIALS 16종 + "door" + "window", 각 성분 [0.02, 0.95]
DEFAULT_MATERIAL_BY_CLASS: dict[str,str] # SLOT_CLASSES 전부 -> MATERIALS 원소 ("default" 슬롯 재질의 해석)
def sem_group(sem) -> int   # VOID -1; floor/wall/ceiling 0; door/window 1; FURNITURE_CLASSES 슬롯 2; 그 밖 슬롯 3; 모르는 id ValueError
@dataclass(frozen=True, kw_only=True, eq=False)
class SynthCamera: c2w (4,4), K (3,3), width:int, height:int, k1=0.0, k2=0.0   # float64로 복사. 모양·비유한·비직교 R·fx/fy<=0·정수 아님/<1 -> ValueError
@dataclass(frozen=True, kw_only=True)
class SynthOpts: spp:int=1 (1,4,9,16 중 하나, 아니면 ValueError); ambient=0.05; exposure=0.05 (유한, >=0); bg=(0,0,0) (길이 3, 유한, >=0)
@dataclass(frozen=True, kw_only=True, eq=False)
class SynthSample:
    rgb (H,W,3) f32 선형 HDR(클램프 없음) | gbuf (12,H,W) f32 | depth (H,W) f32 | sem (H,W) i16 | inst (H,W) i32
    flow (H,W,2) f32 | flow_valid (H,W) bool | camera: SynthCamera | flow_camera: SynthCamera|None | prov: Prov | meta: dict
    속성(gbuf 슬라이스, 복사 아님): log_depth (H,W), normal (3,H,W), albedo (3,H,W), sem4 (4,H,W), level_mask (H,W)
    rgb_display() -> f32 = color.linear_to_display(rgb) | to_arrays() -> dict | content_hash() -> 64 hex sha256
def render_gbuffer(program, camera, *, flow_to=None, opts=None) -> SynthSample
def default_camera(program, seed:int, *, width=256, height=256, fov_x_deg=75.0) -> SynthCamera
def translate_camera(camera, d_cam) -> SynthCamera      # c2w[:3,3] += R_c2w @ d_cam, 나머지 동일
def render_room(seed, *, width=256, height=256, fov_x_deg=75.0, room_cfg=None, opts=None, flow_dx_m=0.05) -> SynthSample
```
타입 오류(program이 SceneProgram 아님, camera/opts/flow_to 타입)는 `TypeError`, `flow_to`의 영상 크기가 다르면 `ValueError`. 위치 인자 금지(키워드 전용 부분).

## G-buffer 12ch (맞은 화소 / 미스 화소)
0 `log_depth` = ln(depth) / 0 · 1–3 법선(카메라 좌표) / 0 · 4–6 albedo / 0 · 7–10 `sem4` = `sem_group`의 one-hot(합 1) / 0 · 11 `level_mask` = 1 / 0. 미스는 12채널 전부 0, `depth=inf, sem=0, inst=-1, flow=0, flow_valid=False, rgb=opts.bg`. gbuf에 NaN/inf 없음. 기하(depth·sem·inst·법선·albedo·광류)는 항상 **화소 중심 광선 1개**(= `raycast`와 바이트 동일: depth/sem/inst)에서 온다(spp 무관).
albedo: 셸 면은 `program.matter`의 floor/wall/ceiling 재질(없으면 wood/paint_white/paint_white), 개구부 화소는 `ALBEDO["door"|"window"]`, 슬롯 화소는 `slot.material`("default"면 `DEFAULT_MATERIAL_BY_CLASS[cls]`)의 `ALBEDO`.

## 셰이딩 (그림자·간접광·반사 없음, v0)
`rgb = albedo ⊙ (ambient + exposure · Σ_i rgb_i·power_i·max(0, n_w·ω_i) / (4π²·d_i²))`, ω_i=(pos_i−P)/|·|, **d_i=max(|pos_i−P|, 0.5 m)**, 광원은 `id` 오름차순 합(슬롯·광원 입력 순서와 무관하게 바이트 동일). n_w=카메라를 향한 세계 법선. 모든 `Light.kind`를 점광원으로 취급. 미스 화소는 bg.
`spp=n²`: rgb만 n×n 정규 격자 부화소 평균, 오프셋 `(i+0.5)/n−0.5`(i=0..n−1, x·y 모두, 모든 화소 동일·무작위 없음). 각 샘플은 그 광선의 자기 기하로 셰이딩, 구현은 `K[0,2]-=dx; K[1,2]-=dy`로 `raycast`한 것과 같다.
**디노이저 OFF(화소 독립).** 어떤 채널도 이웃 화소를 섞지 않는다: 영상 일부를 `geom.crop_intrinsics`로 자른 렌더는 전체 렌더의 같은 부분과 1e-5 안에서 일치. `meta["denoiser"]="off"`.

## 광류
`flow_to=None`이면 flow=0, flow_valid=False. 아니면 맞은 화소의 X를 A(왜곡 k1,k2 포함)로 역투영해 B(자기 왜곡 포함)에 투영, `flow_valid = hit & (z_B > 1e-3)`, 무효는 0(NaN 금지). 부화소 정밀, 영상 밖 투영도 유효.

## 카메라·방 파이프
`default_camera`: (program, seed)의 결정론 함수(전역 RNG 불변), K=`intrinsics_from_fov(fov_x_deg,W,H)`, k=0. 눈은 셸 다각형 안, 벽·비-rug 슬롯 발자국 밖에서 여유 ≥ 사다리 {0.5,0.4,0.3,0.2} m 중 찾아지는 가장 큰 값, z∈[1.0, min(1.8, 높이−0.5)], 피치 ±15° 이내·롤 0. 후보 16개를 순서대로 뽑아 32×32 프로브에서 슬롯 화소 ≥5%인 첫 후보(없으면 최대 비율)를 쓴다.
`render_room(seed,…)` = `generate_room(seed, room_cfg)` → `default_camera(prog, seed)` → `flow_to=translate_camera(cam,(flow_dx_m,0,0))`(None이면 광류 없음) → `render_gbuffer`. 닫힌 방이라 미스 없음.
`meta` = `{"renderer":"buc.data.synth_render.v1","denoiser":"off","spp","program_hash","seed","grammar"}`(값은 str/int/float/bool). `prov = Prov()`(taint 비어 있음, ns "engine"). `to_arrays()` 키 = rgb,gbuf,depth,sem,inst,flow,flow_valid,c2w,K(위 dtype, c2w·K는 f64). `content_hash()`는 to_arrays 바이트+meta로, 같은 입력 → 같은 해시.

## 수용 기준
`tests/dat_01/` 전부 통과(CPU, <60 s, 결정론, 동결): 상수·검증·import 위생, 해석 상자 방 오라클(깊이 2e-5, 법선, sem, 알베도, 개구부, 미스), 셰이딩 공식(전 화소, 노출 선형성, 후면 클램프, 거리 하한, 그림자 없음), spp 평균 의미, 크롭 불변(디노이저 OFF), 광류 닫힌형·geom 오라클·무효 규칙, RDR-05 광류·법선과 RDR-02 항등 렌더 교차검증, `default_camera` 제약, `render_room` 불변식·결정론·계보, 속도(256² <12 s).

## 하류 이음매
DAT-02: `rgb`·`gbuf`·`depth`·`prov`로 Realify 입력/라벨(`sem`, `inst`)을 만든다. DAT-09: `program_hash`·`inst`·`sem`을 구성상 정확한 라벨로 쓴다. MDL-07: `gbuf`(12,H,W)가 G-buffer 입력(채널 순서 고정). DAT-05: `to_arrays()`·`meta`·`prov`를 샤드에 쓴다. 채널 순서·단위·`meta` 키를 바꾸면 이 계약서와 함께 올린다.
