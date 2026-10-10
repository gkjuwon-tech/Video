# RDR-02 평면 쿼드·LDI 래스터 v0 (빠른 경로: 알파·소프트 에지)

**목적.** `World.layers`(LDILayer 앞→뒤)를 임의의 목표 카메라에서 찍어 선형 RGB·알파·깊이·층 id를 만든다. 슬라이스용 빠른 경로이며, 2DGS(RDR-01)와 TSQ(RDR-03)가 나중에 같은 시그니처로 대체한다.
**소유.** `buc/render/quads.py`(네임스페이스 패키지, `__init__.py` 불필요). 테스트 `tests/rdr_02/`(동결 `RDR-02.tests.json`).
**의존.** FND-02(`World`, `LDILayer`, `Obs`, `Shot`, `Frames`), FND-06(`geom`, `color`, `pyramid`). numpy·scipy·torch 허용(내부 구현은 자유, torch는 CPU만으로 동작해야 함). import 부작용·전역 RNG 변경 금지, GPU 없이 import·실행.

## 규약 (FND-02/06과 동일)
세계 오른손·미터, 카메라 OpenCV, 픽셀 중심 정수 좌표, 영상 `[-0.5, W-0.5)`. `c2w` = 카메라→세계 (4,4). 층의 rgb는 **선형** Rec.709, 알파는 **직선(비전치)**. 출력도 선형(표시값 변환은 `render_shot`에서만).

## 공개 API
```python
@dataclass(frozen=True, kw_only=True)
class RenderOpts:
    bg: tuple[float,float,float] = (0,0,0)   # 모든 층 뒤의 선형 배경색
    edge_px: float = 1.0                     # 최소 필터 반경(목표 px) = 소프트 에지 폭
    z_tol_rel: float = 0.1                   # 같은 층 안에서 최근접 깊이의 (1+z_tol_rel)배보다 먼 텍셀은 가려짐
    alpha_eps: float = 1e-3                  # 알파 <= eps 텍셀은 건너뜀
    near_m: float = 1e-3                     # 목표 카메라 z < near_m 인 텍셀은 건너뜀
    max_footprint_px: float = 8.0            # 텍셀 풋프린트 반경 상한
    # __post_init__: edge_px>0, z_tol_rel>=0, alpha_eps>=0, near_m>0, max_footprint_px>=1, bg 길이 3 유한 아니면 ValueError

@dataclass(frozen=True, kw_only=True)
class QuadRender:
    rgb: np.ndarray       # (H,W,3) float32 선형, 배경 위에 합성된 값. 범위는 입력 색의 볼록 껍질 안
    alpha: np.ndarray     # (H,W) float32 [0,1], 1 - prod(1 - A_l)
    depth: np.ndarray     # (H,W) float32 목표 카메라 광축 z-깊이(m). 합성 가중 기대값. 가중합 < 1e-3 이면 +inf. NaN 금지
    layer_id: np.ndarray  # (H,W) int16, 가중치 w_l 최대인 층(동률은 작은 번호=앞). alpha < 1e-3 이면 -1

def card_layer(rgba, depth_m) -> LDILayer                 # 상수 깊이 평면 카드. depth_m 유한·양수 아니면 ValueError
def render_ldi(world: World, *, c2w, K, width: int, height: int,
               k1: float = 0.0, k2: float = 0.0, opts: RenderOpts | None = None) -> QuadRender
def render_shot(world: World, shot: Shot, opts: RenderOpts | None = None) -> Frames
```
`c2w`·`K`는 배열류(리스트·float32 허용). `k1,k2`는 **목표** 카메라 왜곡(FND-06 `distort`와 같은 식); 원본 왜곡은 `world.obs.k1,k2`로 `geom.unproject`에서 처리. 잘못된 입력(모양, 비유한, 비직교 R, fx/fy<=0, width/height 정수 아님·<1, k NaN)은 `ValueError`; 위치 인자는 `TypeError`.

## 알고리즘 의미 (구현 방식은 자유, 관측 가능한 결과만 계약)
1. 층마다 텍셀(원본 픽셀) 중심을 `obs`로 역투영해 목표로 투영하고, 텍셀 사각형(코너 ±0.5 px)의 투영 크기로 풋프린트를 정한다: 반경 `r = clip(max(edge_px, 투영 크기), edge_px, max_footprint_px)`, 각 텍셀은 면적 `Ax`(목표 px²)만큼의 알파 질량을 필터(단위 적분)로 퍼뜨린다. 항등 카메라에서 정확히 항등이다.
2. **전치 알파(premultiplied)로 필터링**한다. 투명 텍셀의 rgb 쓰레기가 새면 안 된다. 층 색 `C_l = Σw·a·c / Σw·a`, 층 알파 `A_l = min(1, Σw·a)`.
3. 같은 층 안에서 한 목표 화소에 겹치면 최근접 깊이 `z_min`의 `(1+z_tol_rel)`배보다 먼 텍셀은 버린다. 연결 메시가 없으므로 깊이 불연속은 **찢어진다**(고무 시트 없음, 탈가림은 알파 0 구멍).
4. 층 합성(앞→뒤, 직선 알파 over): `w_l = A_l·∏_{k<l}(1−A_k)`, `rgb = Σ w_l C_l + (1−Σ w_l)·bg`, `depth = Σ w_l z_l / Σ w_l`.
5. `world.far_field`는 v0에서 쓰지 않는다(배경은 `opts.bg`).

## 불변식
- 결정론: 같은 입력 → 같은 바이트(rgb/alpha/depth/layer_id). 입력(`world`)은 불변, RuntimeWarning 없음.
- 항등 카메라(`c2w`=obs 포즈, 왜곡 포함)는 층을 재현. 상수 층·over 연산은 1e-5 이내로 정확.
- 알파 질량 보존: 이진 알파 카드의 에지 위치가 알파 합으로 0.2 px 이내 정확(부화소), 전이 폭 <= 3 px, 에지에서 알파 단조(링잉 없음), `edge_px`를 키우면 전이만 넓어지고 위치는 보존.
- 정면 평면은 FND-06 `plane_homography` 오라클과 평균 오차 <0.01(확대 2배까지, 핀홀 없음 alpha>0.98), 기울어진 평면은 평균 <0.02, 축소는 평균 <0.03.
- 테두리 오염(어두운/밝은 프린지)과 쓰레기 rgb 누출 없음. 카메라 뒤·근평면 카드는 건너뛰어도 NaN·경고 없음.
- 깊이 AOV는 목표 카메라 z-깊이(앞으로 0.5 m 가면 3.0→2.5, 회전 시 화소별 평면 z와 3e-3 이내).

## `render_shot` (Frames 이음매)
프레임마다 `render_ldi(world, c2w=shot.c2w[i], K=shot.K[i], width=shot.width, height=shot.height, opts=opts)`(프레임별 K 허용). 결과 `Frames(rgb=linear_to_display(각 rgb) (N,H,W,3) float32, fps=shot.fps, aov={"depth": (N,H,W) f32, "alpha": (N,H,W) f32, "layer_id": (N,H,W) int16}, shot_hash=shot.content_hash(), world_hash=world.content_hash())`. 이 단위는 DOF·셔터·필름을 하지 않는다(LNS-02/03/04가 `QuadRender`·`Frames` 뒤에 붙는다).

## 수용 기준
`tests/rdr_02/` 전부 통과(CPU, <60 s, 결정론, 동결 해시 `RDR-02.tests.json`): API·검증·결정론·Frames 이음매, 항등·호모그래피 오라클·시차·점 착지(`geom.project` 대조, 목표 왜곡 포함), over 합성·AOV, 소프트 에지 질량 보존·프린지 없음, 층 내 찢김·가림, 확대 핀홀 없음, 속도(320x180x4층 1프레임 <8 s, 160x90 8프레임 <20 s, 층 4배에 시간 약 8배 이내).

## 하류 이음매
RDR-04가 필터 반경 `edge_px`/풋프린트를 EWA로 교체, RDR-05가 `layer_id`·`depth`·`alpha`를 AOV로 확장, LNS-02/03/04는 `render_ldi` 출력(선형 rgb·depth)을 후처리, WLD-03/08과 CAM-05는 `render_ldi`/`render_shot` 시그니처만 쓴다. 시그니처를 바꾸면 이 계약서와 함께 올린다.
