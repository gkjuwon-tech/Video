# RDR-04 EWA 비등방 필터·C_0 정합 렌더 필터 (카디널 B-스플라인 사전필터)

**목적.** 첫 프레임 충실도(FFD 바닥)를 정하는 읽기 필터. 모든 프레임(프레임 0 포함)이 같은 함수로 텍스처를 읽는다. (1) 보간은 **카디널 B-스플라인**(사전필터로 노드 보간, 보간 필터의 MTF를 관측 대역 안에서 역보정), (2) 축소는 **EWA 비등방 가우시안**이되 **J가 확대·항등이면 블러 0**(항등 카메라 = 입력 비트 근접). 원천: pipeline.upg.md F-P3 "C_0 정합 렌더 필터", X-FF0 게이트(FFD ≤ 0.015, MTF 비 ≥ 0.9 @0.35 cy/px).
**소유.** `buc/render/filters.py`(네임스페이스 패키지, `__init__.py` 불필요). 테스트 `tests/rdr_04/`(동결 `RDR-04.tests.json`). 의존: RDR-02(`quads`, 이음매 시험에서만), FND-06 `pyramid.sample_bilinear`(모드 `"bilinear"` 대조용 허용). numpy·scipy(`scipy.ndimage.spline_filter`/`map_coordinates` 사용 가능) 허용, torch 불필요. import 부작용·전역 RNG 변경·RuntimeWarning 금지, GPU 없이 동작(GPU 테스트 없음).

## 규약
좌표 `uv=(x,y)`, **화소 중심이 정수**, 영상 영역 `[-0.5, W-0.5)×[-0.5, H-0.5)`. 영상은 선형 값(float), `(H,W)` 또는 `(H,W,C)`. 알파가 있으면 호출자가 **전치 알파**로 넣는다(필터는 영상에 선형이므로 쓰레기 rgb가 새지 않는다). 경계는 **거울(whole-sample, x=0·W-1 둘레 대칭)** 고정. 야코비안 `jac[...,i,j] = ∂uv_i/∂xy_j`(원본 px / 목표 px, i=원본축, j=목표축).

## 공개 API (`__all__` 에 모두 포함)
```python
DEGREES = (1, 3, 5);  MODES = ("bilinear", "bspline", "ewa")
@dataclass(frozen=True, kw_only=True)
class EwaOpts:
    aa_scale: float = 0.5        # κ >= 0
    max_aniso: float = 16.0      # >= 1
    max_sigma_px: float = 8.0    # > 0, 원본 px, 안티앨리어싱 σ 상한
    recon_sigma_px: float = 0.0  # >= 0, 등방 추가 σ(0 = 순수 보간). 기본 0 이어야 항등이 정확
    # __post_init__: 비실수·비유한·범위 밖은 ValueError
def bspline_basis(x, degree=3) -> ndarray[float64]        # 중심 B-스플라인 β^n, 지지 |x|<(n+1)/2
def interp_mtf(freq, degree=3, *, prefiltered=True) -> ndarray[float64]
    # prefiltered: sinc(f)^(n+1) / B_n(f), B_3=(4+2cos2πf)/6, B_5=(66+52cos2πf+2cos4πf)/120, B_1=1. False: sinc^(n+1). 우함수
@dataclass(frozen=True, eq=False)
class PrefilteredTexture: coef: ndarray[float64]; degree: int; shape: tuple   # coef 모양 = 입력 모양
def prefilter_texture(img, *, degree=3) -> PrefilteredTexture   # 거울 경계 정확 사전필터. 입력 불변·읽기 전용 허용
def bspline_sample(tex, uv) -> ndarray[float64]                  # uv (...,2) -> (...[,C]) 순수 카디널 보간
def ewa_covariance(jac, *, opts=None) -> ndarray[float64]        # (...,2,2) -> (...,2,2) 원본 px² 공분산
def ewa_sample(tex, uv, jac, *, opts=None) -> ndarray[float64]   # jac 는 uv.shape[:-1]+(2,2) 로 브로드캐스트
def homography_uv_jacobian(H, xy) -> tuple[ndarray, ndarray]     # (uv (...,2), jac (...,2,2)), 분모 c<=0 이면 둘 다 NaN(경고 없음)
@dataclass(frozen=True, eq=False)
class WarpResult: image: ndarray[float32]; valid: ndarray[bool]   # (h,w[,C]), (h,w)
def warp_homography(img, H, out_hw, *, mode="ewa", opts=None, fill=0.0) -> WarpResult
```
`warp_homography`: `H` 는 **목표→원본** 3×3(목표 화소 (x,y) → 원본 uv), `out_hw=(h,w)` 양의 정수. 모드 `bilinear`(사전필터·AA 없음, `sample_bilinear` edge 경계, 대조군), `bspline`(AA 없음), `ewa`. `valid` = 분모>0 이고 uv 가 원본 영역 안. 무효 화소는 `fill`. 출력 float32, 항상 유한.
**입력 검증.** 영상 dtype 이 float 가 아니면 `TypeError`; ndim∉{2,3}·빈 배열·비유한 `ValueError`. degree∉DEGREES, mode 오류, H 모양·비유한·특이(|det| ≤ 1e-12·‖H‖³), out_hw 오류, fill 비유한, uv 마지막 축≠2·비유한, jac 모양·비유한·브로드캐스트 불가 → `ValueError`. uv/jac 비수치 → `TypeError`. `PrefilteredTexture` 가 아닌 tex → `TypeError`. `degree`·`mode`·`opts`·`fill`·`interp_mtf` 의 `prefiltered` 는 키워드 전용(위치로 주면 `TypeError`).

## EWA 의미 (관측 가능한 결과만 계약)
`J = U diag(s1≥s2) Vᵀ`(SVD), `s2' = max(s2, s1/max_aniso)`, `σ_i = min(aa_scale·√max(s_i'²−1, 0), max_sigma_px)`, `Σ = U diag(σ1²+r², σ2²+r²) Uᵀ`(r=recon_sigma_px). 결과 = 카디널 보간 연속함수를 `N(0, Σ)` 로 평활한 값(정규화 가중, 직류 보존). 따라서 `s_i ≤ 1`(확대·항등)이면 Σ=0, `ewa_sample == bspline_sample`. 공분산은 J 의 좌특이벡터·특이값에만 의존. 구현(탭 배치·구적)은 자유, 이득 오차 ≤ 0.012(등방)·0.03(클램프 시험) 이내.

## 불변식
- 결정론(같은 입력 → 같은 바이트), 입력 불변, 영상에 선형(채널 독립·`a·img1+b·img2`), 상수 영상은 어떤 J·uv 에서도 상수.
- 사전필터 보간은 정수 노드에서 입력 재현(오차 < 1e-9), 3차 다항식 재현(안쪽), 거울 경계 대칭, 해석 `interp_mtf` = 수치 커널 푸리에(±2e-3). 0.35 cy/px: 3차 0.916 ≥ 0.9, 쌍선형 0.657.
- 항등 `warp_homography`(ewa·bspline)는 입력과 max 오차 < 1e-5. 부화소 이동·회전 왕복 RMSE(대역 제한 영상, σ=0.15)는 < 0.004 이고 쌍선형의 0.2배 미만. 반화소 이동 이득(cos 줄무늬): 0.10 cy/px ≥ 0.995, 0.20 ≥ 0.98, 0.30 ≥ 0.91, 0.35 ≥ 0.82(쌍선형의 1.6배 이상).
- 비등방: y 로만 8배 축소해도 가로 0.25 cy/px 줄무늬 이득 0.97–1.03, `max_aniso=1`(등방)이면 < 0.05, 세로 앨리어싱 < 0.02. 2배 축소: 통과대역(0.1 cy/px) ≥ 0.93, 0.35 cy/px 앨리어싱 < 0.25(쌍선형의 0.55배 미만).

## 수용 기준
`tests/rdr_04/` 전부 통과(CPU, <60 s, 결정론, 동결 해시 `RDR-04.tests.json`): API·검증·기저·사전필터·MTF·EWA 공분산·주파수 응답·호모그래피 야코비안·항등 루프·왕복·유효 마스크·RDR-02 `render_ldi` 및 평면 호모그래피 오라클 대조(평균 오차 < 0.01/0.005)·속도(160×120×3 2배 축소 < 15 s, 256×256×3 항등 < 5 s).

## 하류 이음매
- WLD-02(입력 역투영 TABP): `prefilter_texture`/`warp_homography` 로 입력→텍셀 격자 역투영(카디널 사전필터가 관측 대역 MTF를 역보정). EVL-12(X-FF0): 항등 루프 FFD 측정에 `warp_homography(mode="ewa")` 사용.
- RDR-02 래스터가 소프트 에지 텐트 대신 `ewa_covariance`(Σ)를 풋프린트로 쓸 수 있다. 시그니처를 바꾸면 이 계약서와 함께 올린다. 미결: 텍셀 스플랫(scatter)용 EWA 는 이 단위 밖(수집 gather 형만 제공).
