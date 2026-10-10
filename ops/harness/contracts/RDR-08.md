# RDR-08 STRAND/SGGX 체적 v0 — 털·잎·솜털, 매팅 α를 싣는 그릇

**목적.** 가는 구조(털 커튼·잎·솜털)를 표면 하나로 세우면 고무막이 된다(F-P5, D15 야크 털 커튼). 이 단위는 (1) SGGX 미세플레이크 수학, (2) SGGX 비등방 소광을 갖는 3D 가우시안 체적, (3) 화면 공간 가닥(STRAND) 래스터, (4) 매팅 α→체적 밀도 변환(RD-18 경계 분해의 그릇, 폴백 사다리 "α를 밀도로"), (5) 가닥→SGGX 체적 LOD를 순수 PyTorch(CPU 정본, autograd)로 제공한다.
**소유.** `buc/render/vol.py`. 테스트 `tests/rdr_08/`(동결 `RDR-08.tests.json`). **의존.** RDR-01(`splat2d` 카메라 규약·합성 규칙), FND-06(`geom`). numpy·scipy·torch만. import 시 gsplat import·CUDA 초기화·전역 RNG 변경 금지.

## 규약
RDR-01과 같다: 세계 오른손·미터, OpenCV 카메라, `c2w` (4,4), 화소 중심 정수, 화소 (u,v)의 광선 = `C + t·ω`, `ω = normalize(R·K⁻¹[u,v,1])`. 색은 선형, 직선 알파. 잘못된 카메라 → ValueError(RDR-01과 같은 검사).

## SGGX (numpy 입력 → numpy, torch → torch; `(...,3,3)`·`(...,3)` 브로드캐스트; 방향은 내부에서 정규화)
```python
sggx_fiber(t, roughness)    # S = I − t tᵀ + r² t tᵀ   (최대 고유값 1, σ(t)=r)
sggx_surface(n, roughness)  # S = n nᵀ + r² (I − n nᵀ)
sggx_sigma(S, w)            # σ(w) = sqrt(wᵀ S w)   투영 면적
sggx_D(S, m)                # D(m) = 1 / (π·sqrt(det S)·(mᵀ S⁻¹ m)²)
sggx_phase_specular(S, wi, wo)  # D(h)/(4σ(wi)), h = normalize(wi+wo); |wi+wo|<1e-9 → 0. wi·wo 모두 산란점에서 바깥쪽
```
항등식: `∫_{S²} max(0,w·m) D(m) dm = σ(w)`, `∫ f_p dwo = 1`, `σ(wi)f(wi,wo) = σ(wo)f(wo,wi)`, S 스칼라배에 f_p 불변. roughness ∉ (0,1] → ValueError.

## 데이터
```python
@dataclass(kw_only=True, eq=False)
class VolGaussians:   # torch.Tensor, 같은 dtype(float32|64)·device. numpy 입력은 tensor로
    means (N,3); quats (N,4) wxyz; log_scales (N,3); log_density (N,); colors (N,3); sggx (N,3,3) SPD
    n; densities()=exp; rotations()->(N,3,3) 열=축; covariances()=R diag(s²) Rᵀ
    params() -> [means, quats, log_scales, log_density, colors]; requires_grad_(); detach(); to(device=None,dtype=None); subset(idx|mask)
@dataclass(kw_only=True, eq=False)
class Strands:
    points (S,P,3) P≥2; radius (S,P) m >0 ((S,)는 (S,P)로 확장); opacity (S,) ∈ (0,1]; colors (S,3)
    n; params() -> [points, radius, opacity, colors]; detach(); to(...); subset(...)
```
모양·dtype 불일치, 비유한, sggx 비대칭(>1e-6·max|S|)·비양정치, 범위 위반 → ValueError.

## 렌더
```python
@dataclass(frozen=True, kw_only=True)
class VolOpts: bg=(0.,0.,0.); near=0.01; far=1e4; filter_sigma=0.5; max_alpha=0.999; min_alpha=1/255; cutoff=4.0
@dataclass(kw_only=True, eq=False)
class VolRender: rgb (H,W,3); alpha (H,W); depth (H,W); def numpy() -> dict[str, np.ndarray]  # float32
def render_vol(*, vol=None, strands=None, c2w, K, width, height, opts=None, backdrop=None) -> VolRender
```
- **가우시안 조각** (화소 광선마다): `Σ⁻¹`, `d=C−μ`, `a=ωᵀΣ⁻¹ω`, `b=ωᵀΣ⁻¹d`, `t*=−b/a`, `m²=dᵀΣ⁻¹d−b²/a`. 광학 깊이 `τ = ρ·σ_S(ω)·sqrt(2π/a)·exp(−½m²)`(직선 전체 적분, 소광 = ρ·σ(ω)), `α = min(max_alpha, 1−e^{−τ})`. 조각 깊이 = 점 `C+t*ω`의 카메라 z, 색 = colors. `m² > cutoff²` 또는 깊이 ∉ [near, far]이면 버린다(min_alpha는 가우시안에 적용하지 않음). 저역 필터 없음(v0).
- **가닥 조각**: 선분 (P_k,P_{k+1})의 끝점을 투영(a,b px, z_a,z_b). 한 끝이라도 z ∉ [near,far]면 선분 컬링. `t = clamp((x−a)·(b−a)/|b−a|², 0, 1)`(|b−a|²<1e-12이면 0), `d = |x − (a+t(b−a))|`, `z(t)=(1−t)z_a+t z_b`, `r(t)` 선형, `w = r(t)·sqrt(fx·fy)/z(t)` px, `α_seg = o·[Φ((d+w)/σ_f) − Φ((d−w)/σ_f)]`(σ_f = filter_sigma). 가닥 하나는 화소당 조각 하나: 선분 중 α 최대(동률은 작은 번호), α는 max_alpha로 클램프, `α < min_alpha`면 버림. 깊이 = 그 선분의 z(t), 색 = 가닥 색. 따라서 `∫ α dd = 2w·o`(얇을 때 열 합 = o·2w).
- **backdrop**(선택): `rgb`(전치 곱, bg=0으로 렌더), `alpha`, `depth` 키/속성(numpy 또는 tensor, (H,W[,3]))을 가진 것. 화소당 조각 하나(깊이 = depth, α = alpha, 색 = rgb/alpha)로 합류. RDR-01 `render(..., opts=SplatOpts(bg=(0,0,0)))` 결과를 그대로 넣을 수 있다. 모양 불일치 → ValueError.
- **합성**(RDR-01과 같은 규칙, 화소별 정렬): 조각을 깊이 오름차순 안정 정렬(동률: 가우시안 번호 → 가닥 번호 → backdrop). `T_i=∏_{j<i}(1−α_j)`, `rgb=Σα_iT_ic_i + T·bg`, `alpha=Σα_iT_i`, `depth=Σα_iT_iz_i/alpha`, alpha<1e-3이면 depth=+inf. 출력 dtype = 입력 dtype. 빈 입력 → bg, alpha 0, depth inf.

## 변환
```python
def alpha_to_tau(alpha, max_alpha=0.999)   # −log(1 − clip(α, 0, max_alpha)); numpy→numpy, tensor→tensor
def tau_to_alpha(tau)                       # 1 − exp(−τ)
def matte_to_vol(alpha, fg_rgb, depth, *, K, c2w, stride=1, footprint=1.0, thickness=None, tangent=None,
                 roughness=0.1, min_alpha=1/255, dtype=torch.float32) -> VolGaussians
def strands_to_vol(strands, *, roughness=0.1, min_std=0.0) -> VolGaussians
def scatter_colors(vol, *, light_dir, light_rgb, cam_center, albedo=None) -> torch.Tensor  # (N,3)
```
- `matte_to_vol`: 행·열 `0, stride, …`의 화소 중 깊이 유한·양수이고 `α ≥ min_alpha`인 것마다 하나(행 우선 순서). μ = `geom.unproject_world`(z-깊이), 축 = 카메라 x·y·z(세계), 표준편차 `s_x = footprint·stride·z/fx`, `s_y = footprint·stride·z/fy`, `s_z = thickness`(None이면 `0.25·s_x`). sggx = I(tangent None) 또는 `sggx_fiber(tangent, roughness)`(tangent (3,) 또는 (H,W,3) 세계). 밀도: 원천 광선 ω_src = normalize(μ−C)에서 피크 광학 깊이가 `alpha_to_tau(α)/(2π·footprint²)`가 되게 `ρ = τ_t/(2πf²)/(σ_S(ω_src)·sqrt(2π/a))`. 색 = fg_rgb. 격자 합이 2πf²이므로 원천 뷰에서 내부 α·`αF+(1−α)B`가 재현되고, 시점을 옮기면 3D 층처럼 시차가 난다.
- `strands_to_vol`: 길이 L>1e-9인 선분마다 하나. μ = 중점, 축1 = 접선 t(나머지 두 축은 det +1 정규직교 보완), `s_t = max(0.5L, min_std)`, `s_p = max(r̄, min_std)`(r̄ = 끝 반지름 평균), sggx = `sggx_fiber(t, roughness)`, `ρ = 2·o·r̄·L / ((2π)^{3/2}·s_t·s_p²)`, 색 = 가닥 색. 그래서 `∫τ dA = 2 o r̄ L σ(ω)` = 가닥의 투영 커버리지.
- `scatter_colors`: `wi = −normalize(light_dir)`, `wo = normalize(cam_center − μ_i)`, 결과 `albedo ⊙ light_rgb · f_p(S_i, wi, wo)`(albedo 기본 = colors). 그림자 없음(v0). Gaffer 림 이음매.

## 불변식
결정론(CPU 같은 바이트) · 입력·전역 RNG 불변 · NaN/경고 없음(카메라 뒤, 길이 0 선분, edge-on) · alpha = 1−∏(1−α_i)(순서 무관) · 입력 순서 순열에 출력 불변 · float64에서 autograd = 중앙 차분.

## 수용 기준 (`tests/rdr_08/`, CPU <60 s)
SGGX 항등식(구면 구적 0.5–1%) · 단일 가우시안 τ = 수치 선적분 오라클(1e-3 상대)·깊이 = t*의 z · 섬유 비등방 비 = roughness · 합성 공식·backdrop 가림(1e-6) · 가닥 프로파일 오라클(1e-5)·얇은 가닥 열 합 o·2w(3%)·관절 이중 계산 없음·깊이 보간·교차 정렬 · 매트 재현(내부 α 2e-3, 색 3e-3, τ 질량 2%)·선형 램프·시차 3 px·섬유 매트 비 · strands_to_vol 질량 동치(4%) · 검증·import 격리 · gradcheck(float64) · 속도(64×48, 가우시안 1500 + 가닥 100×7점 fwd+bwd < 10 s). `@pytest.mark.gpu`: CUDA 결과 = CPU(2e-4).

## 하류 이음매
RD-18 경계 분해 → `matte_to_vol(α, F, depth)`(배경 B는 별도 층). WLD-03은 RDR-01 surfel을 `backdrop`으로 합류. 털 에셋(CC0 메시 + 털)은 `Strands`, 원거리는 `strands_to_vol`. Gaffer 림은 `scatter_colors`. `VolRender.numpy()`의 rgb/alpha/depth는 `SplatRender`·`QuadRender`와 같은 뜻.
