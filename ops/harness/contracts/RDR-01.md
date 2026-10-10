# RDR-01 2DGS 래퍼 v0 — 미분 래스터 · dual-cut LOD · COLMAP 점 초기화 피팅

**목적.** 프로젝트의 primary 미분 래스터. 2D 가우시안 surfel(2DGS)을 임의 카메라로 찍어 선형 RGB·알파·깊이·노멀을 내고, 역전파로 피팅한다. 백엔드 둘: `"torch"`(순수 PyTorch 기준 구현, CPU·CUDA, float32/64, **정본**)와 `"gsplat"`(`gsplat.rasterization_2dgs`, CUDA float32). sm_120 JIT가 실패해도 torch 백엔드가 폴백이다.
**소유.** `buc/render/splat2d.py`, `buc/render/fit.py`. 테스트 `tests/rdr_01/`(동결 `RDR-01.tests.json`). **의존.** RUN-05(gsplat 설치·`TORCH_EXTENSIONS_DIR` 캐시), FND-06(`geom`, `color`). numpy·scipy·torch 허용.
**import 규칙.** 두 모듈 import는 `gsplat`을 import하지 않고 CUDA를 초기화하지 않는다(지연 import). 전역 RNG 불변.

## 규약 (FND-02/06과 동일)
세계 오른손·미터, 카메라 OpenCV, `c2w` (4,4) 카메라→세계, 픽셀 중심 정수 좌표. 색은 선형 Rec.709, 활성화 없이 그대로 합성(SH 없음, v0).

## splat2d.py
```python
class BackendUnavailable(RuntimeError)
@dataclass(kw_only=True, eq=False)
class Surfels:                      # 모두 torch.Tensor, 같은 dtype(float32|float64)·device
    means: (N,3); quats: (N,4) wxyz(정규화 전); log_scales: (N,2); opacity_logits: (N,); colors: (N,3)
    # __post_init__: numpy 입력은 tensor로. 모양·dtype 불일치·비유한 → ValueError
    n: int (property); scales()=exp; opacities()=sigmoid; rotations()->(N,3,3); normals()->(N,3)
    params() -> [means, quats, log_scales, opacity_logits, colors]   # 이 순서
    requires_grad_(flag=True)->self; detach(); clone(); to(device=None, dtype=None); subset(idx|mask)
    @classmethod cat(items); @classmethod empty(dtype=float32); save(path) / @classmethod load(path)  # npz, 비트 왕복
@dataclass(frozen=True, kw_only=True)
class SplatOpts: bg=(0.,0.,0.); near=0.01; far=1e4; backend="auto"   # 잘못된 값 ValueError
@dataclass(kw_only=True, eq=False)
class SplatRender: rgb (H,W,3); alpha (H,W); depth (H,W); normal (H,W,3); backend: str
    def numpy(self) -> dict[str, np.ndarray]   # float32, grad 없음
def resolve_backend(name: str, device) -> str  # "auto": env BUC_SPLAT_BACKEND 우선, CUDA 장치+gsplat import 성공이면 "gsplat", 아니면 "torch"
def render(surfels, *, c2w, K, width, height, opts=None) -> SplatRender
```
`R(q)`의 열 = (t_u, t_v, n) 세계 방향. surfel 점 `p + s_u·u·t_u + s_v·v·t_v`. 화소 x의 광선과 surfel 평면의 교점에서 (u,v)를 구한다(2DGS 광선-surfel 교차).
- 화소 알파: `α_i = min(0.999, o_i·exp(−½·min(u²+v², 2‖x−c_i‖²)))`, `c_i` = 중심 투영(px, 저역 필터 σ=1/√2 px), `α_i < 1/255`는 버림. 3σ 밖 컬링은 허용.
- 정렬: 중심의 카메라 z 오름차순(안정, 동률은 입력 번호). 중심 z<near 또는 >far는 컬링. 양면.
- 합성: `T_i=∏_{j<i}(1−α_j)`, `w_i=α_i T_i`; `rgb=Σw c + T·bg`, `alpha=Σw`, `depth=Σw z_i/alpha`(z_i = 교점의 카메라 z), `alpha<1e-3`이면 depth=+inf·normal=0. `normal` = 세계 단위 노멀(카메라 쪽으로 뒤집어 `w` 가중 합 후 정규화).
- 백엔드 출력 dtype = 입력 dtype. `backend="gsplat"`인데 CPU 텐서·gsplat 없음 → `BackendUnavailable`. `"auto"`에서 gsplat import/JIT 실패 시 경고 후 torch로 폴백. 잘못된 카메라(모양, 비유한, 비직교 R, fx/fy≤0, width/height 정수 아님·<1) → ValueError.

**dual-cut LOD** (PARADIGM C7, 16×16 타일)
```python
def lod_lambda(depth, K, h0, kappa=1.0)       # λ = log2(h0·fx/(κ·depth)); depth=inf → −inf. numpy→numpy, tensor→tensor
@dataclass(kw_only=True, eq=False)
class DualCutRender: rgb; alpha; depth; normal; level (H,W); n_tiles: int; n_tiles_double: int; cuts_rendered: tuple[int,...]
def render_dual_cut(cuts: Sequence[Surfels], lam, *, c2w, K, width, height, opts=None, tile=16, band=(0.05, 0.95)) -> DualCutRender
```
`cuts[0]`이 가장 거친 레벨. `lam`은 스칼라 또는 (H,W) 배열/텐서. 타일 λ = 타일 안 유한 값의 평균(없으면 0), `[0, L−1]`로 클램프. `l_a=min(⌊λ⌋, max(L−2,0))`, `f=λ−l_a`; `f≤band[0]→0`, `f≥band[1]→1`. 타일 출력 `rgb,alpha = (1−f)·I^{l_a} + f·I^{l_a+1}`, depth·normal은 우세 컷(f<0.5면 l_a). `level = l_a+f`. `n_tiles_double` = 0<f<1인 타일 수, `cuts_rendered` = 실제로 렌더한 컷 번호(오름차순). 정수 λ에서 양쪽 극한이 같다(λ-연속). NaN·모양 불일치·빈 cuts·tile<1 → ValueError.

## fit.py
```python
@dataclass(frozen=True, kw_only=True)
class ColmapPoints: ids int64 (N,); xyz float64 (N,3); rgb uint8 (N,3); error float64 (N,); track_len int64 (N,)
def read_colmap_points3d(path) -> ColmapPoints      # points3D.txt|.bin, 디렉터리면 .bin 우선. id 오름차순. 깨진 형식 ValueError
def init_from_points(xyz, rgb=None, *, k_nn=3, k_normal=8, scale_mult=1.0, min_scale=1e-4, opacity=0.1,
                     cam_centers=None, dtype=torch.float32) -> Surfels
def init_from_rgbd(rgb_lin, depth, *, K, c2w, stride=1, alpha=None, opacity=0.99, scale_mult=0.5, dtype=torch.float32) -> Surfels
@dataclass(frozen=True, kw_only=True)
class FitView: image (H,W,3) 선형; c2w; K; mask=None
@dataclass(frozen=True, kw_only=True)
class FitOpts: iters=300; lr_means=1.6e-4; lr_quats=1e-3; lr_scales=5e-3; lr_opacity=5e-2; lr_colors=2.5e-3;
    lambda_ssim=0.2; lambda_dist=0.0; lambda_normal=0.0; batch_views=0; prune_opacity=0.0; bg=(0,0,0); seed=0; backend="auto"
@dataclass(kw_only=True, eq=False)
class FitResult: surfels (detached); losses: list[float] (len=iters, 스텝 전 손실); psnr: tuple[float,...] (뷰별 최종)
def fit(init, views, opts=None) -> FitResult
def psnr(a, b, mask=None) -> float              # 10·log10(1/MSE), 같으면 inf
```
- `init_from_points`: means=xyz; 스케일(두 축) = `max(min_scale, scale_mult·k_nn 최근접 거리 평균)`(k는 `min(k,N−1)`); 노멀 = `k_normal` 이웃 PCA 최소 고유벡터, `cam_centers`가 있으면 가장 가까운 카메라 쪽(`n·(c−p)≥0`); `R(q)[:,2]=n`, det +1. rgb: uint8 → `color.srgb8_to_linear`, float → 선형 그대로, None → 0.5. 불투명도 `opacity`. N<2·비유한 → ValueError.
- `init_from_rgbd`: 유효 화소(유한·양수 깊이, α>1e-3)마다 하나, 행·열 `0, stride, …`. 접선 = 카메라 x·y축(정면 surfel), 중심 = `geom.unproject_world`, 스케일 `scale_mult·stride·depth/fx`(·/fy), 불투명도 `opacity·α`.
- `fit`: Adam(파라미터 그룹별 lr), 손실 `(1−λ_ssim)·L1 + λ_ssim·(1−SSIM 11×11 σ1.5)` + 2DGS 왜곡·노멀 정합 정규화(λ>0일 때). `batch_views=0`이면 매 반복 전 뷰. 같은 seed·장치 → 비트 동일. `init` 불변. `prune_opacity>0`이면 끝에 그 미만 surfel 제거.

## 불변식
결정론(같은 입력 → 같은 바이트, CPU) · 입력 불변 · NaN/경고 없음(카메라 뒤·edge-on 포함) · float64 경로에서 autograd = 중앙 차분 · 입력 순서 무관(정렬 후 동일).

## 수용 기준 (`tests/rdr_01/`, CPU <60 s)
API·검증·지연 import·백엔드 선택 · 단일 surfel: 피크 = o, 질량 `2πσ²o`(5%), σ_px = fx·s/z(4%), 중심 착지 = `geom.project`(0.03 px), 비등방 축 각(1°) · 기울어진 surfel 깊이 = 광선-평면 교점(1e-4 상대)·노멀 · over 합성 공식(1e-5) · 컬링 · FD 기울기(float64) · dual-cut 블렌드·스냅·클램프·λ-연속·타일 평균·통계 · COLMAP txt/bin 동치 · 초기화 수치 · rgbd 항등 재현 · 피팅 수렴(손실 ≤1/4, PSNR) · 결정론 · 속도(64×48, 1000개 fwd+bwd <8 s). `@pytest.mark.gpu`: gsplat ↔ torch 수치 일치(2e-3)·기울기 코사인 >0.99.

## 하류 이음매
RDR-03(TSQ)은 `Surfels`·`render` 시그니처와 합성 규칙을 확장, RDR-08 체적은 같은 정렬·합성에 합류, WLD-03은 `init_from_rgbd`/`init_from_points`, WLD-07은 `fit`/`render`, RUN-08은 `render(backend="gsplat")` 시간 측정. `SplatRender.numpy()`의 `rgb/alpha/depth`는 RDR-02 `QuadRender`와 같은 뜻(선형, z-깊이, +inf).
