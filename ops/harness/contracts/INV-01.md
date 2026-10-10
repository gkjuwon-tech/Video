# INV-01 문법 코어 + RoomGen v0 — 장면 프로그램 · 레이캐스트 · G1–G13 에너지

**목적.** MRH/MassingFM 합성 데이터의 원천. 장면 프로그램(SceneProgram) = 라벨. 방(RoomGen v0)을 만들고, 카메라에서 깊이·의미·슬롯 id·법선을 쏘고, 문법 규칙 위반을 에너지로 잰다. 장르·OOG 문법(INV-02/03)은 이 코어 위에 얹는다.
**소유.** `buc/invent/grammar/core.py`, `buc/invent/grammar/room.py`(네임스페이스 패키지, `__init__.py` 불필요). 테스트 `tests/inv_01/`(동결 `INV-01.tests.json`).
**의존.** FND-02(`buc.core.contracts.content_hash`), FND-06(`buc.core.geom`). numpy만(scipy 허용, **torch 금지**: import 시 torch가 로드되면 실패). **numba는 선택**: 없으면 순수 numpy 경로, 있으면 JIT(import 때 컴파일 금지, 첫 호출 때). import 부작용·전역 RNG(np.random, random) 변경 금지, 입력 불변, GPU 불필요.

## 규약
세계 오른손·미터, **z 위**, 바닥 z=0, 천장 z=`shell.height`. 카메라는 FND-06(OpenCV, 픽셀 중심 정수, `c2w`). 슬롯: `center` = OBB 중심, `size` = (sx,sy,sz) 전체 길이, 로컬 +x = 정면(front), `yaw` = z축 CCW(로컬→세계). 셸 `corners`는 CCW, 변 i = corner i → (i+1)%M, 안쪽 = 진행 방향 왼쪽, 개구부 `s` = 변 시작점에서의 거리.

## 상수
`SLOT_CLASSES`(앞 16개 고정·순서 불변, 추가만 허용) = bed sofa table chair desk shelf cabinet wardrobe lamp plant tv person rug picture stool counter. `FURNITURE_CLASSES` = bed sofa table desk shelf cabinet wardrobe counter. `SUPPORTS` = floor wall slot ceiling. `OPENING_KINDS` = door window. `LIGHT_KINDS` = window lamp sun other. `MATERIALS`(+ 슬롯만 "default"). `SEM_VOID,FLOOR,WALL,CEILING,DOOR,WINDOW = 0..5`, `SEM_SLOT_BASE = 8`, `sem_id(cls) = 8 + SLOT_CLASSES.index(cls)`(모르는 클래스 ValueError). `ENERGY_NAMES = ("G1",…,"G13")`. `HAS_NUMBA: bool`. `class GrammarError(ValueError)`.

## 데이터 (모두 `@dataclass(frozen=True, kw_only=True, eq=False)`, 배열은 float64 C-연속으로 강제, 위반은 `GrammarError`)
- `Shell(corners (M,2) 3≤M≤16, height>0, walls_present: tuple[bool]*M = 전부 True, floor=True, ceiling=True)`. CCW·면적>0 아니면 거부. `n_edges, edge_vectors(), edge_lengths(), perimeter(), area()`.
- `Opening(kind, edge:int, s0<s1, z0<z1)` — 범위 밖(벽 밖)은 **허용**(G2가 점수를 매긴다).
- `Slot(id:int, cls, center(3), size(3)>0, yaw=0, support="floor", support_id=-1, material="default")`. `z0, z1, footprint() -> (4,2)`.
- `Light(id, kind, pos(3), rgb(3)≥0, power≥0, source: None | ("opening"|"slot", int))`.
- `SceneProgram(grammar:str, seed:int, shell, openings=(), slots=(), lights=(), matter={}, meta={})`. 생성 시 `validate()`: 개구부 `edge` 범위, 슬롯/광원 id 중복 금지, `matter` 키 ⊆ {floor,wall,ceiling}·값 ∈ MATERIALS. `slot_by_id(i)`(없으면 KeyError), `to_dict()`(JSON 가능, `"__contract__": "SceneProgram"`), `from_dict`(다른 계약명은 GrammarError), `content_hash()`(= FND-02 `content_hash(to_dict())`, 64 hex, 1 ulp 민감, 비트 무손실 왕복). 규칙 위반 프로그램은 구성 가능해야 한다(에너지가 잰다).

## 레이캐스트
```python
@dataclass(frozen=True, kw_only=True, eq=False)
class RayCastResult: depth (H,W) f32; sem (H,W) i16; inst (H,W) i32; normal (H,W,3) f32; def visible_slots() -> {slot_id: 픽셀수}
def raycast(program, *, c2w, K, width:int, height:int, k1=0.0, k2=0.0, backend="auto") -> RayCastResult
```
광선 = 픽셀 중심 → `geom.unproject(K, uv, 1.0, k1, k2)`(왜곡 포함) → `c2w` 회전. **depth = 광축 z-깊이**(광선 매개변수 t, z_c=1 정규화). 표면 = 바닥(다각형 안만)·천장·존재하는 벽(수직 사각형)·슬롯 OBB 6면. 첫 교차만. 개구부는 **벽 평면에 칠해진다**(깊이·법선은 벽과 같고 sem만 DOOR/WINDOW, 겹치면 `openings` 앞 순서가 이김). sem: 셸은 위 상수, 슬롯은 `sem_id`. inst: 슬롯 id, 셸·미스 -1. 미스(열린 벽 너머, 바닥/천장 없음): depth=+inf, sem=0, inst=-1, normal=(0,0,0). 법선은 단위 벡터이며 광선을 마주 본다(n·d<0). 슬롯 순서는 결과에 영향 없음(바이트 동일). `RuntimeWarning` 없음(축 평행 광선 포함). 잘못된 입력(c2w/K 모양·비유한·비직교 R, width/height 정수 아님·<1, 모르는 backend)은 `ValueError`, 위치 인자는 `TypeError`. `backend` ∈ auto|numpy|numba; numba 미설치에서 "numba"는 `RuntimeError`; 설치돼 있으면 numpy와 rtol 1e-6로 일치. 64×64 한 장 < 0.5 s(numpy 경로).

## 에너지 (G1–G13 골격)
```python
@dataclass(frozen=True, kw_only=True, eq=False) class CameraSpec: c2w (4,4), K (3,3), width, height
@dataclass(frozen=True, kw_only=True, eq=False) class EnergyContext: cameras: tuple[CameraSpec]=(); allowed_classes: frozenset|None=None; symmetry: ("x"|"y", float)|None=None
@dataclass(frozen=True, kw_only=True, eq=False) class GrammarEnergy: terms: {G1..G13: float}; weights: {…}; total: float; def vector() -> (13,) f64
RULES: dict[str, Callable[[SceneProgram, EnergyContext], float]]   # 13개, ENERGY_NAMES 순서
def grammar_energy(program, ctx=None, *, weights=None) -> GrammarEnergy   # total = Σ w·term, 기본 w=1; 모르는 키·음수·NaN 가중치는 ValueError
```
모든 항은 유한·≥0·결정론, 슬롯/광원 순서와 평행이동에 불변. 허용 기준: 슬롯 접촉·지지 허용오차 0.02 m(벽 0.05 m).
- **G1** 폐합 = Σ(없는 벽 변 길이) + (floor 없으면 둘레) + (ceiling 없으면 둘레).
- **G2** 출입 = max(0, 1−문 수) + (벽 밖 개구부 수: s0<0, s1>변 길이, z0<0, z1>height) + (같은 변에서 s·z 구간이 **열린 구간으로 겹치는** 개구부 쌍 수; 맞닿음은 겹침 아님).
- **G3** 동선 = 문마다 문 앞 1 m 띠(가로 = 문 폭 [s0,s1]∩변, 깊이 = 안쪽 법선 0..1 m)에서 **막힌 가로 길이의 합집합**(m). 장애물 = `support=="floor"`이고 `cls!="rug"`인 슬롯 발자국이 그 가로 위치의 깊이 선분과 교차. 분해능 ≤0.02 m 이하이거나 해석적, 오차 ≤0.03 m. (주 동선 폭 ≥0.6 m 검사는 v1.)
- **G4** 지지 = 위반 슬롯의 부피 sx·sy·sz 합. floor: |z0|≤0.02. ceiling: |z1−height|≤0.02. slot: `support_id` 슬롯이 존재·자기 자신 아님·|z0−그 z1|≤0.02·중심이 그 발자국 안. wall: 발자국 코너와 임의 벽 변(존재 여부 무관) 선분 최소 거리 ≤0.05.
- **G5** 빛 인과 = 출처 없는 광원 수. 유효: `("opening",i)`가 존재하고 `kind=="window"`, 또는 `("slot",i)`가 존재하고 `cls=="lamp"`. 그 외(None 포함) 위반.
- **G6** 반복 = 변마다 창(`window`) 중심 s를 정렬한 간격 g_i, 창이 3개 이상인 변에서 Σ|g_i − mean(g)|, 변들의 합.
- **G7** 대칭 = `ctx.symmetry=(axis,a)`이면 슬롯마다 중심을 축 평면 `axis=a`로 거울 반사한 점과 **같은 cls** 슬롯 중심(자기 포함)의 최근접 3D 거리 합. 없으면 0.
- **G8** 양식 = `ctx.allowed_classes`가 있으면 그 밖 cls 슬롯 수, None이면 0.
- **G9–G12**(실외 연속·스케일 연출·반사 정합·겹 연속)은 v0에서 항상 0.0(INV-03/04/07이 채운다).
- **G13** 발명 인물 = `cls=="person"` 슬롯 중 어떤 `ctx.cameras`에서든 (카메라 앞 z_c>0, 머리 중심 (x,y,z1−0.12)이 영상 안, 정면 (cos yaw, sin yaw)·(카메라 xy − 슬롯 xy) > 0, h_f = 0.22·fy/z_c/height > 0.01)인 슬롯 수(슬롯당 1회). 카메라 없으면 0.

## RoomGen v0
```python
@dataclass(frozen=True, kw_only=True) class RoomGenConfig:
    area_range=(9.0,40.0); aspect_range=(1.0,2.2); height_range=(2.4,3.2); l_shape_prob=0.25
    furniture_range=(3,9); accessory_range=(0,6); n_doors_range=(1,2); n_windows_range=(0,4); people=0
def generate_room(seed: int, cfg: RoomGenConfig|None = None) -> SceneProgram
```
검증(ValueError): 범위 lo≤hi, area lo≥9, aspect lo≥1, height lo≥2.4, l_shape_prob∈[0,1], people≥0, n_doors lo≥1, 개수 범위 lo≥0; seed는 음이 아닌 정수(bool·float 거부, numpy 정수 허용).
출력: `grammar=="room"`, `seed==seed`. 같은 (seed,cfg) → 같은 `content_hash`(프로세스·PYTHONHASHSEED 무관), 전역 RNG 불변. 셸: 직사각(4) 또는 L자(6, 직각·CCW·오목 모서리 1개), 면적·높이는 범위 안, 모든 변 ≥1.0 m, 벽·바닥·천장 모두 존재. 문 1–2개(서로 다른 변, 폭 0.7–1.2, z0=0, 1.9≤z1≤2.4; 요청이 2이고 직사각 area≥14면 정확히 2), 창은 문 없는 한 변에 등간격(창 수는 요청 이하; 요청 하한 ≥1이고 문 없는 긴 변이 있으면 최소 1개), 개구부는 벽 안·서로 비겹침. 슬롯: 바닥 가구(FURNITURE_CLASSES, support floor)는 요청 수 k 이하(**area≥25 m²·k≤8이면 정확히 k**), 부속(chair plant stool tv picture rug, tv는 가구 위 support slot, picture는 support wall), 천장 램프(cls lamp, support ceiling) ≥1. 모든 슬롯 발자국이 방 다각형 안, z∈[0,height], 바닥 지지 슬롯(rug 제외)끼리 관통 없음, 문 앞 띠 비움. 광원 = 램프당 1 + 창당 1, 모두 유효 출처. `people=n`이면 `person` 슬롯(크기 약 0.5×0.5×1.7, 벽을 향함)을 n명 이하(area≥25면 정확히 n). 기본(people=0)은 사람 없음. **어떤 방도 G1–G13 전부 0**(카메라 없음, 허용오차 1e-6). 방은 닫혀 있다: 슬롯 밖 자유 지점에서 큐브 6면으로 쏘면 모든 광선이 맞는다. 다양성: 30개 시드에서 해시 전부 다름, 면적 폭 넓음, 클래스 ≥9종. 속도 목표 장면당 수십 ms(테스트 상한 평균 0.4 s).

## 수용 기준
`tests/inv_01/` 전부 통과(CPU, <60 s, 결정론, 동결): 상수·데이터 검증·직렬화·해시, torch 미로드·PYTHONHASHSEED 불변·전역 RNG 불변, 해석해(벽/바닥/천장/슬롯/45° 회전)와 **독립 오라클** 대조(깊이 2e-5, sem/inst 99.5%, 법선), 왜곡·개구부·가림·열린 벽·경고 없음, 에너지 각 항 공식대로(허용오차 명시), RoomGen 불변식·설정 존중·다양성·L자 비율(이항 3σ)·속도.

## 하류 이음매
INV-04는 `SceneProgram`·`Slot.footprint()`·`raycast`(깊이·sem·inst)로 카빙 32³·부분 점유·RHC 샘플을 만든다(점유 격자는 INV-04 소유). INV-07은 `grammar_energy`/`RULES`/`EnergyContext`에 E_gram을 얹고 G9–G12를 채운다. DAT-01은 `SceneProgram.matter`·`Slot.material`·`Light`로 셰이딩하고 `raycast`를 G-buffer 교차 오라클로 쓴다. INV-02/03은 `generate_*`가 `SceneProgram`을 내고 같은 `RULES`를 확장한다(상수 추가만, 기존 id 불변). 시그니처를 바꾸면 이 계약서와 함께 올린다.
