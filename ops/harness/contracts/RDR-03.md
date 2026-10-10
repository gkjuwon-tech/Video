# RDR-03 TSQ 텍스처 쿼드 커널 — 8×8 페이지 바이리니어 · n≥2 에지 · 미분 · BC7/BC5 페이지

**목적.** PARADIGM §7.3 TSQ(Textured Soft Quad)를 찍는 미분 래스터. 2DGS surfel(RDR-01)에 (1) 초타원 에지 지수 `n≥2`(n=2 부드러운 surfel, n≥8 단단한 쿼드)와 (2) TSQ마다 8×8 선형 RGB 페이지(바이리니어)를 더한다. 백엔드 둘: `"torch"`(정본, CPU·CUDA, float32/64)와 `"cuda"`(`tsq_kernel.cu` JIT, CUDA float32, 하드웨어 바이리니어 허용). 페이지 저장은 BC7(L_d 로그 코드) + BC5(노멀/높이) = 128 B/페이지. 폴백은 RDR-01 2DGS.
**소유.** `buc/render/tsq.py`, `buc/render/tsq_kernel.cu`. 테스트 `tests/rdr_03/`(동결 `RDR-03.tests.json`). **의존.** RDR-01(`splat2d.Surfels`, `render`, `BackendUnavailable`, 합성 규칙), FND-06 `geom`. numpy·scipy·torch만.
**import 규칙.** `import buc.render.tsq`는 CUDA 초기화·컴파일을 하지 않는다(`load_extension()`에서 지연 JIT). 전역 RNG 불변.

## 공개 API (`buc/render/tsq.py`)
```python
PAGE = 8; LMIN = 2.0**-16
BackendUnavailable                      # = buc.render.splat2d.BackendUnavailable (같은 클래스 재수출)
@dataclass(kw_only=True, eq=False)
class TSQ:   # torch.Tensor, 같은 dtype(float32|float64)·device. numpy 입력은 tensor로
    means (N,3); quats (N,4) wxyz; log_scales (N,2); opacity_logits (N,); edge_n (N,); pages (N,8,8,3) 선형
    # 모양·dtype 불일치·비유한·edge_n<2 → ValueError
    n; dtype; device; scales(); opacities(); rotations()->(N,3,3); normals()->(N,3)
    params() -> [means, quats, log_scales, opacity_logits, pages, edge_n]      # 이 순서
    requires_grad_(flag=True)->self; detach(); clone(); to(device=None, dtype=None); subset(idx|mask)
    @classmethod cat(items); @classmethod empty(dtype=float32); save(path)/@classmethod load(path)  # npz 비트 왕복
    @classmethod from_surfels(s: Surfels, *, edge_n=2.0) -> TSQ   # pages = colors 상수 복제
    to_surfels() -> Surfels                                         # colors = 페이지 64 텍셀 평균
@dataclass(frozen=True, kw_only=True)
class TSQOpts: bg=(0.,0.,0.); near=0.01; far=1e4; backend="auto"   # backend ∈ {auto, torch, cuda}, 잘못된 값 ValueError
@dataclass(kw_only=True, eq=False)
class TSQRender: rgb (H,W,3); alpha (H,W); depth (H,W); normal (H,W,3); backend: str
    def numpy(self) -> dict[str, np.ndarray]                       # float32, grad 없음
def resolve_backend(name, device) -> str   # auto: env BUC_TSQ_BACKEND 우선, CUDA 장치+확장 로드 성공이면 "cuda", 아니면 "torch"
def load_extension()                       # JIT 모듈. CUDA·nvcc 없음/컴파일 실패 → BackendUnavailable (다른 예외 금지)
def edge_falloff(u, v, n) -> Tensor        # exp(-½(|u|^n+|v|^n)), 브로드캐스트
def sample_page(pages, idx, uv) -> Tensor  # (N,8,8,C),(M,) long,(M,2) → (M,C) 바이리니어·클램프
def render(tsq, *, c2w, K, width, height, opts=None) -> TSQRender
# BC 코덱 (numpy, CPU)
def encode_bc7(img: uint8 (H,W,4)) -> bytes;  def decode_bc7(data, width, height) -> uint8 (H,W,4)
def encode_bc5(img: uint8 (H,W,2)) -> bytes;  def decode_bc5(data, width, height) -> float32 (H,W,2) ∈[0,1]
@dataclass(frozen=True, kw_only=True)
class PageBlob: n: int; lo: float32 (N,); hi: float32 (N,); bc7: bytes (64N); bc5: bytes|None (64N); nbytes: int (property, bc7+bc5 길이)
def compress_pages(pages, nh=None) -> PageBlob   # pages (N,8,8,3) ≥0 선형; nh (N,8,8,2) ∈[-1,1]. NaN·모양 → ValueError
def decompress_pages(blob) -> float32 (N,8,8,3); def decompress_nh(blob) -> float32 (N,8,8,2) | None
```

## 래스터 의미 (RDR-01과 같은 규약: 세계 오른손·OpenCV·`c2w`·픽셀 중심 정수)
- 광선-평면 교점에서 지역 좌표 `u = r·t_u / s_u`, `v = r·t_v / s_v` (`R(q)` 열 = t_u, t_v, n). 교점 유효 = `|n·d|>1e-6` 이고 `t>0`.
- 에너지 `E = ½(|u|^n + |v|^n)`(유효할 때, 아니면 +∞), 저역 `E_lp = ‖x − c‖²`(px, c=중심 투영). `α = min(0.999, o·exp(−min(E, E_lp)))`, `α<1/255`이면 0. n은 `max(edge_n, 2)`로 쓴다.
- 색: 페이지 `page[j, i]`의 텍셀 중심은 `u_i = −1 + (2i+1)/8`(열 i ↔ u), `v_j`(행 j ↔ v). 텍셀 좌표 `x = 4(u+1) − 0.5`, `y = 4(v+1) − 0.5`를 `[0,7]`로 클램프한 바이리니어. 교점 무효면 `(u,v)=(0,0)`.
- 정렬(중심 카메라 z 오름차순, 안정)·near/far 컬링·over 합성·depth(교점 z, 무효면 중심 z)·normal(카메라 쪽 뒤집기, w 가중 정규화)·`alpha<1e-3`이면 depth=+inf, normal=0 — RDR-01과 동일.
- **이음매 정리:** `edge_n=2` + 상수 페이지 ⇒ `splat2d.render`와 같은 결과(float64 1e-7).
- 미분: 모든 `params()`에 autograd. `u=0`, n 큰 값(float32 64), 스침각에서도 기울기 유한(NaN/inf 없음).

## BC 규약
- 블록 4×4, 래스터 순서(블록 행 우선, 블록 안 화소 행 우선). `encode_bc7`은 **모드 6만** 낸다(바이트0 하위 7비트 = 0x40). `decode_bc7`은 모드 6을 사양대로 비트 정확히(`((64−w)e0 + w·e1 + 32)>>6`, 앵커 1비트 생략) 풀고, 모드 바이트 0(예약)은 (0,0,0,0), 미지원 모드는 ValueError.
- BC5 = BC4(R) 8 B + BC4(G) 8 B. 디코드는 D3D 부동 팔레트(`r0>r1`: 7분할, 아니면 5분할 + 0, 1)를 /255.
- 페이지 로그 코드: `y = log2(max(L, LMIN))`, 페이지마다 `lo = f32(min y)`, `hi = f32(max(max y, lo + 1e-3))`, `code = rint(255·(y−lo)/(hi−lo))` → BC7 RGBA(code, 255). 복원 `L = 2^(lo + c/255·(hi−lo))`. 페이지 p는 이미지 `(8N, 8)` 세로 쌓기의 블록 4개 = `bc7[64p:64p+64]`. nh는 `rint((x+1)/2·255)` → BC5, 복원 `2·d − 1`.
- v0 범위 밖: BC6H(HDR 발광), 하드웨어 BC 텍스처 샘플링(GPU 테스트에서만 선택적), RGB9E5 기준값(lo/hi는 f32 메타 8 B/페이지).

## 불변식
결정론(같은 입력 → 같은 바이트, CPU) · 입력 불변 · NaN/경고 없음(카메라 뒤·edge-on 포함) · float64 autograd = 중앙 차분 · 입력 순서 무관.

## 수용 기준 (`tests/rdr_03/`, CPU <60 s)
API·검증·지연 import·`tsq_kernel.cu`(`__global__` 포함) 존재·백엔드 선택 · 독립 numpy 오라클(`tests/rdr_03/rdr03_helpers.py`)과 랜덤 장면 일치(float64 1e-6, skew 포함) · n=2 상수 페이지 = RDR-01(1e-7) · 해석 에지 프로파일(1e-9)·텍셀 중심 착지·페이지 방향(90° 회전)·클램프 · 기울어진 TSQ 깊이 = 광선-평면 · edge-on/카메라 뒤 유한 · 타일링(n=16: α>0.98 비율 ≥0.65, n=2: ≤0.15, 이음매 α>0.8) · `edge_falloff`/`sample_page` 해석 기울기 · render FD 기울기(6개 파라미터, float64) · u=0·n=64(float32) 유한 · 텍스처 복원(200회, 페이지 RMSE <0.02)·기하 복원(150회, 4 mm, 크기 2%) · BC7 모드 6 디코드 오라클 비트 일치·인코드 PSNR ≥36 dB·상수 블록 ±1 · BC5 오라클 일치·PSNR ≥38 dB · 페이지 코덱 lo/hi 정확·코드 오차(평균 ≤4.5, 최대 ≤28 코드)·64/128 B · 압축 페이지 렌더 PSNR ≥35 dB · 속도(64×48, 300개 fwd+bwd <8 s). `@pytest.mark.gpu`: cuda ↔ torch(3e-3, 하드웨어 1/256 바이리니어), 기울기 코사인 >0.99.

## 하류 이음매
WLD-07 의미 패스는 `TSQ`·`render`로 TSQ 미분 피팅, WLD-09 OTBI/TABP는 `sample_page`·`compress_pages`/`decompress_pages`로 페이지에 쓰고 커밋, RDR-08·RD-12는 `TSQRender`를 `SplatRender`와 같은 뜻으로 읽는다. 폴백 시 `TSQ.to_surfels()` → RDR-01. 시그니처 변경은 이 계약서와 함께.
