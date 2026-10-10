# LNS-03 셔터 (Longuet-Higgins split · 롤링 · 180°)

**목적.** 렌더된 셔터 **중간 자세** 영상 1장(선형 rgb, 선택 깊이)과 카메라 궤적으로 노출 시간 동안의 모션블러를 적분한다. PARADIGM-v2 §7.5 LH-split: 회전 블러는 호모그래피 $H(\tau)=K\,R_{w2c}(\tau)\,R_{w2c,mid}^\top K^{-1}$ 적분으로 **정확히**, 평행이동 블러는 깊이로 화소별 1차 근사, 롤링 셔터는 행마다 $\tau(y)$. 기본은 θ의 180° 셔터(`exposure_time_s = shutter_deg/(360·fps)`).
**소유.** `buc/lens/shutter.py`(`buc/lens/__init__.py` 만들지 않음). 테스트 `tests/lns_03/`(동결 `LNS-03.tests.json`).
**의존.** FND-02 `contracts`(`Shot`,`Frames`,`Theta`), FND-06 `geom`(`interp_pose`,`project`,`unproject`,`apply_homography`)·`color`·`pyramid.sample_bilinear`, numpy·scipy. torch 선택(CPU import·실행 가능). import 부작용·전역 RNG·RuntimeWarning 금지. 무작위 지터 없음(결정론).

## 규약
OpenCV 카메라, `c2w`(카메라→세계, 4×4), 픽셀 중심 정수 좌표 `(u,v)=(열,행)`, 깊이는 광축 z(m). 입력·출력 rgb는 **선형** (H,W,3).

## 공개 API (`from buc.lens import shutter`)
```python
@dataclass(frozen=True, kw_only=True)
class ShutterOpts:
    samples: int = 0          # 0=자동: clip(ceil(2*max|blur_vec|)+1, 1, max_samples). >0이면 정확히 그 수
    max_samples: int = 64     # >= 1
    readout_s: float = 0.0    # 롤링 판독 시간(위→아래), 0=글로벌. >=0 유한
    centered: bool = True     # True: [t-T/2, t+T/2], False: [t, t+T]
    translation: bool = True  # 깊이가 있으면 평행이동 블러 포함
    border: str = "edge"      # "edge" | "zero" (sample_bilinear 의미)
    # 범위 밖·비유한·samples/max_samples 정수 아님(bool 포함)·border 미지 → ValueError

@dataclass(frozen=True, kw_only=True)
class ShutterResult:
    rgb: np.ndarray        # (H,W,3) float32 선형
    blur_px: np.ndarray    # (H,W,2) float32 = x(t_close(y)) - x(t_open(y)), 화소의 세계점 궤적 끝점 차(px)
    samples: int           # 실제 쓴 시간 샘플 수

exposure_time(shutter_deg, fps) -> float                  # deg/(360 fps); deg∉(0,360]·fps<=0·비유한 ValueError
exposure_window(t, fps, shutter_deg, *, centered=True) -> tuple[float, float]
sample_times(t0, t1, n) -> np.ndarray[float64]            # 중점 층화: t0+(k+0.5)(t1-t0)/n, k=0..n-1
row_time_offsets(height, readout_s) -> np.ndarray[float64]  # (H,) readout*(y/(H-1)-0.5), H=1이면 [0]
pose_at(c2w, t, tau) -> np.ndarray                       # 스칼라 tau→(4,4), 1D tau→(M,4,4)
rotation_homography(K, c2w_mid, c2w_tau) -> np.ndarray   # (3,3) 또는 c2w_tau (M,4,4)→(M,3,3)
apply_shutter(rgb, depth=None, *, K, c2w, t, t_mid, shutter_deg, fps, opts=None) -> ShutterResult
apply_shutter_frames(frames: Frames, shot: Shot, opts=None) -> Frames
```
`apply_shutter`의 키워드 인자는 키워드 전용(위치 인자 TypeError).

## 수식·의미
- **pose_at**: `c2w` (N,4,4), `t` (N,) 엄격 증가. N=1이면 상수. 아니면 `j = clip(searchsorted(t,tau,'right')-1, 0, N-2)`, `s=(tau-t_j)/(t_{j+1}-t_j)`(구간 밖은 끝 구간 외삽, s<0·s>1), 결과 `geom.interp_pose(c2w_j, c2w_{j+1}, s)`(회전 측지, 위치 선형). 매듭점에서 입력과 1e-12.
- **rotation_homography** = `K · R_τᵀ · R_mid · K⁻¹` (R = c2w[:3,:3], 정규화 없음). 중간 자세 화소 → τ 자세 화소. 같은 회전이면 항등(1e-12).
- **노출 창**: `T=exposure_time`, centered면 `[t_mid−T/2, t_mid+T/2]`, 아니면 `[t_mid, t_mid+T]`. 행 y의 창은 `+row_time_offsets(H, readout_s)[y]`.
- **blur_px**: 깊이 z가 유한하면 `X = c2w_mid ∘ unproject(K, p, z)`, 아니면(깊이 None·+inf·`translation=False`) 무한원점(회전만, 호모그래피로 투영). `x(τ)`=pose_at(τ)로 투영. 카메라 뒤로 가는 점은 0. NaN 금지.
- **적분(gather)**: `out(p) = (1/n) Σ_k I_mid(q_k(p))`, `τ_k = sample_times(창(y))`. 회전 부분은 **정확한 역호모그래피** `q = H(τ_k)⁻¹ p`(1차 근사 `2p − Hp` 금지). 평행이동 부분은 p의 깊이로 계산한 변위 `d_k(p) = x_full(τ_k) − x_rot(τ_k)`를 빼는 1차 근사 `q = H⁻¹p − d_k(p)`를 권장(구현 자유, 아래 불변식만 계약). 샘플링은 `sample_bilinear(…, border=opts.border)`.
- **빠른 경로**: `max|blur_px| ≤ 1e-6`이면 rgb 비트 복사(float32), samples=1.
- **apply_shutter_frames**: 프레임 i마다 `lin = color.srgb_to_linear(frames.rgb[i])`, 깊이 `frames.aov.get("depth")[i]`(없으면 None), `apply_shutter(lin, depth, K=shot.K[i], c2w=shot.c2w, t=shot.t, t_mid=shot.t[i], shutter_deg=shot.theta.shutter_deg, fps=shot.theta.fps, opts)`. 출력 `Frames(rgb=linear_to_display, fps, aov=입력 유지 + "blur_px" (N,H,W,2) f32, 해시 유지)`. N·H·W가 shot과 다르면 ValueError. 입력 불변.

## 불변식 (테스트 허용 오차)
1. 결정론(바이트 동일), 입력 불변, float32, NaN·경고 없음, 정적 카메라(롤링 포함) → 비트 복사.
2. 볼록성: 출력은 채널별 입력 [min,max] 안(1e-5).
3. 순수 회전(요·롤, 깊이 None/유한 둘 다)은 256샘플 역호모그래피 오라클과 내부 MAE < 2.5e-4, max < 2e-3(1차 근사는 실패). blur_px는 `geom.project` 오라클과 1e-3 px.
4. 임펄스 요 블러: 에너지 보존(1%), 무게중심 이동 <0.25 px(centered), 수평 분산 ≈ L²/12(±15%+0.1), L ∝ 셔터 각(90°/180°/360°). `centered=False`면 무게중심이 −L/2 쪽으로.
5. 롤: 주점 화소는 불변, 반경 r 의 블러 길이 ≈ r·Δθ(±5%), 방향 접선.
6. 평행이동(정면 평면 깊이 Z): 블러 길이 fx·v·T/Z(±5%), 깊이 2배면 절반. 달리 인: blur_px 방사형(FOE=주점, 코사인 >0.99), 크기 ≈ r·v·T/Z.
7. 롤링: 짧은 셔터 요 회전에서 수직 막대의 행별 위치가 `project` 오라클과 0.15 px, 가운데 행 불변.
8. 속도: 320×180, 최대 블러 ≈30 px 1프레임 < 4 s(CPU).

## 수용 기준
`tests/lns_03/` 전부 통과(CPU, <60 s, 결정론, 동결 해시 `LNS-03.tests.json`).

## 하류 이음매
- 슬라이스 체인: RDR-02 `render_shot` → LNS-02 `apply_dof_frames` → **LNS-03 `apply_shutter_frames`** → LNS-04. `Shot.t[i]`가 셔터 중간 시각이고 `Shot.c2w[i]`가 그 렌더 자세다.
- `aov["blur_px"]`는 R3 모션블러 방향장 m(x) 라벨(MRH·EPD-S 블러 속도계)로 쓴다. 달리 인은 FOE 방사형.
- 오버스캔 렌더는 아직 없다(가장자리 클램프). RDR가 오버스캔을 주면 `geom.crop_intrinsics`로 K를 맞춰 넘긴다. 롤링 판독 시간의 장치별 값은 CAP 보드 측정(PARADIGM §1323)에서 `ShutterOpts.readout_s`로 들어온다. 시그니처를 바꾸면 이 계약서와 함께 올린다.
