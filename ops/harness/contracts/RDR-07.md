# RDR-07 평면 재귀 반사 · secondary 광선 (Warp BVH 폴백)

**목적.** 평면 거울(거울·유리창·웅덩이·거울 복도)을 RDR-02 래스터 위에서 **가상 카메라 재귀**로 물리 반사 렌더한다(반사 문신 F-P4의 렌더 쪽, PARADIGM §렌더러 "평면 거울 재귀 래스터, 깊이 12"). 곡면 크롬·유리용 secondary 광선은 `trace_closest`(삼각형 closest-hit)로 두고, 백엔드는 OptiX 대신 **Warp `wp.Mesh` BVH**(GPU), 없으면 numpy 기준 구현으로 폴백한다.
**소유.** `buc/render/mirror.py`(네임스페이스, `__init__.py` 없음). 테스트 `tests/rdr_07/`(동결 `RDR-07.tests.json`).
**의존.** RDR-02 `buc.render.quads`(`render_ldi`, `RenderOpts`), FND-02 `contracts`(World/LDILayer/Shot/Frames), FND-06 `geom`·`color`. numpy 필수, `warp`는 **지연 import**(모듈 import 시 `sys.modules`에 `warp` 없음). GPU 없이 import·실행, 전역 RNG 불변, RuntimeWarning 없음.

## 공개 API (`from buc.render import mirror`)
```python
@dataclass(frozen=True, kw_only=True)
class Mirror:
    center: ArrayLike            # (3,) 평면 위 점(세계 m)
    normal: ArrayLike            # (3,) 반사면 방향. 저장 시 단위화(float64). 0벡터·비유한 -> ValueError
    u_axis: ArrayLike | None = None   # 평면 내 축. 저장 시 법선 성분 제거 후 단위화. 법선과 평행 -> ValueError
    half_size: tuple[float, float] | None = None  # (hu, hv) m, u축·v축(v = normal × u). None=무한 평면. <=0 -> ValueError
    reflectance: tuple[float, float, float] = (0.9, 0.9, 0.9)  # 반사마다 곱하는 선형 RGB, 각 [0,1] 밖 ValueError
    two_sided: bool = False
    # 속성 d = normal·center  (평면: normal·X = d). half_size 가 있는데 u_axis None 이면 ValueError

@dataclass(frozen=True, kw_only=True)
class MirrorOpts:
    max_depth: int = 2           # 따라가는 반사 횟수 상한, 0..16 (밖 ValueError)
    max_renders: int = 64        # render_ldi 호출 예산(1차 포함), >=1
    min_weight: float = 1e-3     # [0,1)
    terminal_rgb: tuple = (0, 0, 0)   # 따라가지 않는 거울의 입사 복사휘도(선형)
    clip_eps_m: float = 1e-4     # >=0
    quad: RenderOpts | None = None    # 모든 수준의 render_ldi 에 그대로 전달(bg 포함)

@dataclass(frozen=True, kw_only=True)
class MirrorRender:
    rgb: (H,W,3) f32 선형;  alpha: (H,W) f32;  depth: (H,W) f32 1차 z-깊이
    layer_id: (H,W) int16;  mirror_id: (H,W) int16 (첫 거울, 없으면 -1)
    bounce: (H,W) int8 (경로가 맞은 거울 수, 종단 거울 포함, <= max_depth+1)
    virtual_depth: (H,W) f32 (경로 끝점을 펼친 허상의 1차 카메라 z-깊이, 없으면 +inf)
    n_renders: int

def reflection_matrix(normal, d) -> np.ndarray            # (4,4) 하우스홀더, X' = X - 2(n·X - d) n
def virtual_camera(c2w, K, width, mirror) -> tuple[np.ndarray, np.ndarray]
def ray_mirror_hits(mirrors, *, c2w, K, width, height, k1=0.0, k2=0.0, near_m=1e-3) -> tuple[np.ndarray, np.ndarray]
def clip_world(world, normal, d, *, side=1, eps=1e-4) -> World
def render_mirrors(world, mirrors, *, c2w, K, width, height, k1=0.0, k2=0.0, opts=None) -> MirrorRender
def render_mirror_shot(world, mirrors, shot, opts=None) -> Frames

class BackendUnavailable(RuntimeError)
@dataclass(frozen=True, kw_only=True)
class RayHits: t (N,) f64 (+inf 미스), tri (N,) int64 (-1), bary (N,2) f64 (미스는 0)
def available_backends() -> tuple[str, ...]               # 항상 "numpy" 포함; warp import + CUDA 장치 있으면 "warp"
def trace_closest(origins, dirs, tris, *, t_min=1e-6, t_max=inf, backend="auto") -> RayHits
def reflect_dirs(d, n) -> np.ndarray                      # d - 2(d·n̂)n̂, |d| 보존
def fresnel_schlick(cos_theta, f0) -> np.ndarray          # f0 + (1-f0)(1-clip(cos,0,1))^5, 브로드캐스트
```
키워드 인자는 키워드 전용(위치 인자 `TypeError`). `mirrors`는 `Mirror`의 수열(아니면 `TypeError`), 빈 수열 허용. c2w/K/크기 검증은 RDR-02와 같은 `ValueError`.

## 의미 (관측 가능한 결과만 계약)
1. **가상 카메라.** `M = reflection_matrix(n, d)`, `F = diag(-1,1,1,1)`. `c2w_v = M @ c2w @ F`(회전 det +1), `K_v` = K 에서 `K[0,1] -> -K[0,1]`, `K[0,2] -> width-1-K[0,2]`. 가상 카메라 영상을 좌우로 뒤집으면(`[:, ::-1]`) 원 카메라에서 본 거울상: 점 X 의 가상 투영 u'' 에 대해 `width-1-u''` = 원 카메라에서 `M X` 의 투영(왜곡 포함).
2. **거울 적중.** 화소 중심 광선(목표 왜곡 `k1,k2` 를 `geom.undistort` 로 풂)과 평면의 교점이 사각형 안(`|u|<=hu, |v|<=hv`)이고 z-깊이 >= `near_m`, 앞면(`dir·n < 0`, two_sided면 양면)일 때. 가장 가까운 거울. `ray_mirror_hits` 는 `(mirror_id int16 (H,W), z float64 (H,W) +inf)`.
3. **클립.** `clip_world`: `(n, d)` 를 `|n|` 으로 나눠 단위화(0벡터 `ValueError`), side ∈ {1,-1}. 각 텍셀을 `world.obs` 로 역투영한 X 에 대해 `side·(n·X - d) <= eps` 이면 알파 0(rgb·깊이·나머지 불변, 새 World, 입력 불변).
4. **재귀** L(view, clip, 남은 깊이, 가중 w): `prim = render_ldi(clip 적용 world, view, k1,k2, opts.quad)`. 이 view 의 거울 적중(직전 거울 제외, 교점이 clip 평면 뒤면 제외)이 `prim.alpha < 0.5` 또는 `prim.depth > z_hit` 이면 거울 화소(**v0: 화소 단위 이진 선택**). 거울 i 화소의 rgb = `R_i ⊙ fliplr(L(virtual_camera(view,i), clip=(n_i,d_i, 시점 쪽), 깊이-1, w·max R_i))`. 남은 깊이 0, `w·max R_i < min_weight`, 예산 소진 중 하나면 `R_i ⊙ terminal_rgb`. 순회는 깊이 우선, 거울 번호 오름차순, 필요한 화소가 있을 때만 자식 렌더. 거울 화소: alpha 1, depth = z_hit, layer_id -1, bounce = 1 + 자식 bounce(종단은 1), virtual_depth = 자식의 virtual_depth(뒤집어 가져옴, 종단은 +inf). 자식 렌더 1회 = `n_renders` 1 증가(거울 복도 사슬에서 `n_renders = 1 + 따라간 단계 수`).
5. 거울이 없거나 아무 거울도 보이지 않으면 rgb/alpha/depth/layer_id 는 `render_ldi` 와 **바이트 동일**, mirror_id -1, bounce 0, virtual_depth = depth, n_renders 1.
6. `render_mirror_shot`: 프레임마다 `render_mirrors`, `Frames(rgb=linear_to_display(rgb), aov={"depth","alpha","layer_id","mirror_id","bounce","virtual_depth"}, shot_hash, world_hash)`.

## secondary 광선
`trace_closest`: 양면 삼각형 (M,3,3), M=0 이면 전부 미스. dirs 는 내부 단위화(0벡터 `ValueError`), `t` 는 미터. `t_min < t <= t_max` 중 최근접, 동률은 작은 삼각형 번호. bary `(b1,b2)`: `P = (1-b1-b2)v0 + b1 v1 + b2 v2`. `backend`: `"auto"`(warp 가능하면 warp, 아니면 numpy, 예외 없음), `"numpy"`, `"warp"`(불가면 `BackendUnavailable`), 그 외 `ValueError`. numpy 는 결정론. warp 결과는 numpy 와 t 상대 1e-4, tri 일치(경계 제외).

## 불변식
결정론(같은 입력 → 같은 바이트), 입력 불변, NaN 없음. 거울 평면 뒤의 물체는 그 반사에 나타나지 않는다. 반사 k회 경로의 색 = `∏R · (도달 복사휘도)` → 반사율이 초록 쪽이면 깊어질수록 초록(초록 터널). `n_renders <= max_renders`.

## 수용 기준
`tests/rdr_07/` 전부 통과(CPU, <60 s, `-m "not gpu"`): API·검증·지연 import·결정론·무거울 동일성, 반사 행렬·가상 카메라(`geom.project` 대조, 왜곡 포함)·적중·클립 오라클, 바닥 거울 등가식(`R ⊙ fliplr(render_ldi(clip, 가상))`, 1e-4), 점 착지(1차·반사 0.35 px), 깊이·허상 깊이 AOV, 전경 가림·평면 뒤 누출 없음·뒷면·양면, 거울 복도 광선 오라클(bounce·`R^k·bg`, 화소 99%), max_depth·min_weight·예산 등가, 깊이 12 속도, shot 이음매, `trace_closest` 오라클·경계·성능. warp 대조는 `@pytest.mark.gpu`.

## 하류 이음매
EVL-07 반사 문신 지수는 `mirror_id`·`virtual_depth`·`bounce` AOV 를 쓴다. RD-17(반사 판독)은 `Mirror`(평면 RANSAC 결과)와 `virtual_camera`(C_0^R)를 쓴다. RDR-05 는 AOV 키를 그대로 확장, LNS 후처리는 `rgb` 선형·`depth` 를 받는다. DIEL(유리·수면 굴절)·곡면 크롬은 `trace_closest`·`reflect_dirs`·`fresnel_schlick` 위에 짓는다. OptiX 백엔드는 범위 밖(이름 `"optix"` 예약).
