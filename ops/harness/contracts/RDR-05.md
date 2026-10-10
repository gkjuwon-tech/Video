# RDR-05 AOV (depth·normal·flow·원자 id)·관측 bitset·안정 정렬 v0

**목적.** RDR-02 래스터의 화소마다 "지배 원자(어느 층의 어느 텍셀)"를 기록하고 법선·광류·관측 여부를 얹는다. IR(EVL-05)·halo 지표(EVL-07)·마스크·Proof-of-Shot(RDR-09)의 단일 원천이다. 렌더 결과(rgb·alpha·depth·layer_id)는 RDR-02와 **바이트 동일**이어야 한다.
**소유.** `buc/render/aov.py`(namespace, `__init__.py` 없음). 테스트 `tests/rdr_05/`(동결 `RDR-05.tests.json`). 의존: RDR-02(`render_ldi`, `RenderOpts`, `render_shot`; 비공개 헬퍼 import는 허용하나 `quads.py`는 고치지 않는다), FND-02(`World`, `Shot`, `Frames`), FND-03(`canonical_order`/`segment_reduce`/`hash_*`), FND-06(`geom`). numpy·scipy·torch(CPU)만. GPU 없이 import·실행, import 부작용·전역 RNG 변경 없음. 슬라이스 범위: 지배 원자·depth·법선·광류·bitset 모두 포함(반사 관측 등 O 확장은 P 단계).

## 규약 (RDR-02와 동일)
세계 오른손·미터, 카메라 OpenCV, 픽셀 중심 정수. **원자** = LDI 텍셀 `(층 l, 행 v, 열 u)`, `atom_id = l*H*W + v*W + u`(int64, H,W = `world.obs` 크기, FND-02가 층 크기를 obs와 같게 강제). `-1` = 없음. 원자 위치 = 관측 카메라에서 텍셀 중심을 `layer.depth`로 역투영(`obs.k1,k2` 반영)한 세계 점.

## 공개 API
```python
@dataclass(frozen=True, kw_only=True)
class Cam: c2w: ndarray(4,4); K: ndarray(3,3); k1: float = 0.0; k2: float = 0.0   # render_ldi와 같은 검증(ValueError)
@dataclass(frozen=True, kw_only=True)
class AOV:
    rgb (H,W,3) f32; alpha (H,W) f32; depth (H,W) f32; layer_id (H,W) i16   # = render_ldi 출력과 바이트 동일
    normal (H,W,3) f32      # 목표 카메라 좌표 단위벡터, atom_id<0 이면 0
    atom_id (H,W) int64     # 지배 원자, 없으면 -1
    flow (H,W,2) f32        # (du,dv) px = π_B(X) − π_A(X), X = 지배 원자의 세계 점. 무효면 0 (NaN 금지)
    flow_valid (H,W) bool   # atom_id>=0 이고 X가 B 카메라 near_m 앞이면 True
def n_atoms(world) -> int                                   # len(layers)*H*W
def pack_atom_id(layer, v, u, *, width, height) -> int64    # 브로드캐스트, 범위 밖 ValueError
def unpack_atom_id(atom_id, *, width, height) -> (layer, v, u)   # int64 배열, 음수 ValueError
def atom_points(world, atom_id) -> float64 (...,3)          # 세계 점, id<0 이면 NaN, 범위 밖 ValueError
def render_aov(world, *, c2w, K, width, height, k1=0.0, k2=0.0, opts=None, flow_to: Cam|None=None) -> AOV
def render_shot_aov(world, shot, opts=None) -> Frames
class ObsBitset: words uint64[ceil(n/64)], n
    from_bool(flags) (classmethod; bool 배열 아니면 TypeError, C순서 평탄화) / to_bool() -> bool[n] / count() -> int
    test(atom_id) -> bool (같은 모양; id<0 False, id>=n ValueError) / content_hash() -> hex str
    | & - (합·교·차; n이 다르면 ValueError)
def observed_atoms(world, *, vis_min=0.5, alpha_eps=1e-3) -> ObsBitset     # n_atoms(world)비트
def atom_pixel_counts(atom_id, n) -> int64[n]               # 임의 모양, id<0 무시, id>=n ValueError
def sort_fragments(pixel, depth, atom_id) -> int64[m]       # (pixel, depth, atom_id) 사전식 오름차순 안정 정렬 순열
```
`render_aov`의 입력 검증·`TypeError`(위치 인자)는 `render_ldi`와 같다. `flow_to=None`이면 flow=0, flow_valid=False. 다른 타입의 `flow_to`는 `TypeError`.

## 의미
1. **지배 원자.** 화소의 `layer_id` 층에서, 그 화소에 걸친(RDR-02의 깊이 컬링을 통과한) 텍셀 중 스플랫 가중치 `텐트·알파·면적/(rx·ry)`가 최대인 텍셀(동률은 작은 atom_id). 따라서 `atom_id<0 ⇔ layer_id<0`, 아니면 `atom_id // (H*W) == layer_id`. 항등 카메라·정수 평행이동에서는 정확히 같은 (v,u) 텍셀(또는 정수 이동된 텍셀)이다. 층 안 깊이 불연속의 구멍 화소는 -1.
2. **법선.** 텍셀 법선은 관측 카메라 좌표에서 역투영 점의 유한차분(이웃이 알파>alpha_eps·유한 깊이·`|Δz|<=z_tol_rel·z`이면 중앙차분, 한쪽만 유효하면 단측, 한 축에 이웃이 없으면 `-P/|P|`)의 외적을 카메라를 향하게(`n·P<=0`) 정규화한 것. 화소 법선 = 지배 원자의 법선을 `R_t·R_oᵀ`로 목표 카메라 좌표로 돌린 값. 평면은 정확하다(상수 깊이 카드의 법선 = 관측 카메라 `(0,0,-1)`).
3. **광류.** X = `atom_points`, A = 이번 카메라, B = `flow_to`. 부화소 정밀(화소 중심이 아니라 원자의 투영 위치 기준). 카메라가 +x로 dx 움직이면 깊이 Z 카드는 `du = -fx·dx/Z`.
4. **관측 bitset.** `observed_atoms`: 텍셀 `(l,v,u)`가 관측 ⇔ `alpha_l>alpha_eps` 이고 앞 층들의 투과율 `∏_{k<l}(1−alpha_k) >= vis_min` 이고 (`obs.mask`가 있으면) `mask[v,u]`. 비트 배치(계약): 원자 i ↔ 워드 `i>>6`의 비트 `i&63`(LSB=0), n 이후 꼬리 비트는 0, dtype uint64. `content_hash()` = `hash_parts(b"buc.obsbitset.v1", n, hash_array(words)).hex()`.
5. **IR 이음매.** `IR_geo = 1 − mean(bits.test(atom_id)[atom_id>=0])`(프레임·화소 평균). `atom_pixel_counts`는 FND-03 `segment_reduce`(정수 고정소수점)로 세어 화소·프레임 순서에 비트 불변.
6. **정렬.** `sort_fragments`는 입력 순서를 섞어도 정렬된 `(pixel,depth,atom_id)` 열이 같고, 키가 완전히 같은 항목은 입력 순서를 유지(안정). `depth` NaN·길이 불일치 `ValueError`, `pixel`·`atom_id`가 정수 dtype이 아니거나 `depth`가 float이 아니면 `TypeError`. 비교는 IEEE 값(−0.0 == +0.0, +inf 허용).

## `render_shot_aov`
`render_shot(world, shot, opts)`의 `rgb`·`fps`·`shot_hash`·`world_hash`와 aov(`depth`,`alpha`,`layer_id`)는 그대로(바이트 동일)에 aov 키 `normal (N,H,W,3) f32`, `atom_id (N,H,W) i64`, `flow (N,H,W,2) f32`, `flow_valid (N,H,W) bool`을 더한다. 프레임 i의 flow는 i→i+1(프레임별 K 사용), 마지막 프레임은 flow=0·flow_valid=False.

## 불변식
결정론(같은 입력 → 같은 바이트) · `world` 불변 · RuntimeWarning·NaN 없음 · 법선은 단위·카메라 향함 · 이진 알파 층에서 `layer_id` 영역과 `atom_id` 층이 일치.

## 수용 기준
`tests/rdr_05/` 전부 통과(CPU, <60 s, 동결 `RDR-05.tests.json`): 원자 id 패킹, 항등·정수 이동 지배 원자, RDR-02 바이트 동일, 법선 오라클(카드·기울어진 평면·회전 카메라), 광류 닫힌형·`geom` 오라클·무효 규칙, 찢김/가림에서 지배 원자와 -1, bitset 배치·집합 연산·해시, `observed_atoms` 규칙, AOV+bitset으로 계산한 IR_geo 해석값, 안정 정렬 오라클·순열 불변, 세는 값 순열 불변, `render_shot_aov` 이음매, 검증 오류.

## 하류 이음매
EVL-05는 `atom_id`+`ObsBitset`로 IR을, EVL-07은 `depth/alpha/layer_id/atom_id`로 실루엣 띠·halo를, DAT-01은 이 aov 키를 G-buffer로, RDR-09는 `content_hash`·`atom_pixel_counts`·`hash_array(aov)`로 왕복 동일을 검증한다. 시그니처·키 이름·비트 배치를 바꾸면 이 계약서와 함께 올린다.
