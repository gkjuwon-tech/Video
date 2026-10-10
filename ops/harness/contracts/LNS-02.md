# LNS-02 DOF · Defocus Pyramid · 포커스 브리딩 · 랙 스케줄

**목적.** RDR-02 `render_ldi` 출력(선형 rgb, 카메라 z-깊이)과 `render_shot`의 `Frames` 뒤에 붙는 물리 피사계 심도. 박막 렌즈 CoC $c(z)=\frac{f^2}{N(z_f-f)}\frac{|z-z_f|}{z}$, CoC가 8 px를 넘는 부분은 Defocus Pyramid(Kraus–Strengert'07, $2^{-k}$ 해상도), 포커스 브리딩 $f_{eff}=f(1+\beta m)$, 디옵터 공간 랙 스케줄. **핵심은 칼선 방지**다. 녹은 전경 실루엣은 CoC 폭으로 부드럽게 넘어가야 하고, 선명한 전경 둘레에는 배경 헤일로나 어두운 링이 생기면 안 된다.
**소유.** `buc/lens/dof.py`(`buc/lens/__init__.py`는 만들지 않는다, 네임스페이스). 테스트 `tests/lns_02/`(동결 `LNS-02.tests.json`). **의존.** FND-02 `contracts`(`Theta`, `Shot`, `Frames`), FND-06 `color`·`pyramid`, numpy·scipy(torch는 선택이지만 CPU에서 import·실행 가능해야 한다). import 부작용 금지, 전역 RNG 금지, RuntimeWarning 금지.

## 공개 API (`from buc.lens import dof`)
```python
@dataclass(frozen=True, kw_only=True)
class DofOpts:
    blades: int = 0                 # 0 = 원형 조리개, 3..18 = 다각형
    blade_rot_deg: float = 0.0
    pyramid: bool = True            # Defocus Pyramid 사용
    pyramid_threshold_px: float = 8.0   # >= 2
    max_coc_px: float = 64.0        # >= 1, CoC 상한(클립)
    in_focus_px: float = 0.5        # [0, 2), 이 값 이하 CoC는 선명
    breathing_beta: float = 0.0     # [-1, 1]
    # 범위 밖·비유한·blades 정수 아님(bool·float 포함)은 ValueError

@dataclass(frozen=True, kw_only=True)
class DofResult:
    rgb: np.ndarray      # (H,W,3) float32 선형
    coc_px: np.ndarray   # (H,W) float32, min(|CoC|, max_coc_px)

coc_px(depth, *, focal_mm, f_stop, focus_m, sensor_width_mm, width_px: int, signed=False) -> np.ndarray[float32]
coc_from_theta(depth, theta: Theta, width_px: int, *, focus_m=None, signed=False) -> np.ndarray
defocus_level(coc, threshold_px=8.0) -> np.ndarray[int]   # 0 if |c|<=thr else ceil(log2(|c|/thr))
aperture_kernel(diameter_px, blades=0, rotation_deg=0.0) -> np.ndarray[float32]
apply_dof(rgb, depth, *, focal_mm, f_stop, focus_m, sensor_width_mm, opts=None) -> DofResult
apply_dof_theta(rgb, depth, theta, *, focus_m=None, opts=None) -> DofResult
rack_schedule(n_frames, fps, focus_from_m, focus_to_m, *, start_s=0.0, duration_s=None, ease="smooth") -> np.ndarray[float64]
breathing_scale(focal_mm, focus_m, beta, ref_focus_m=inf) -> float | np.ndarray
breathing_K(K, scale) -> np.ndarray          # fx, fy만 scale배, 주점 유지, 입력 불변
apply_breathing(img, scale, K, *, nearest=False) -> np.ndarray
apply_dof_frames(frames: Frames, shot: Shot, opts=None) -> Frames
```

## 수식·의미
- **CoC(px)** = `f²/(N·(z_f−f))·|z−z_f|/z · width_px/sensor_width_mm` (f, z_f, z는 같은 단위). `signed=True`면 부호는 `sign(z−z_f)`(앞은 음, 뒤는 양). `z=+inf`면 극한 `f²/(N(z_f−f))`. 입력 모양 보존. `focus_m <= focal_mm/1000`, NaN, 비양수 depth, f_stop·focal·sensor ≤0 또는 비유한, width_px<1 이면 ValueError.
- **aperture_kernel**: d ≤ 1이면 `[[1.0]]`, 아니면 크기 `2⌈d/2⌉+1` 정방, 비음수, 합 1, 중심 대칭, 원형이면 `E[r²]≈d²/8`(±12%). 다각형은 지름 d 원에 내접하며 `rotation_deg`가 반영된다(360°/blades 주기).
- **apply_dof**: 입력 rgb (H,W,3) 유한 실수, depth (H,W) 양수(+inf 허용). 모양 불일치·NaN·비양수 깊이는 ValueError, 렌즈 인자는 키워드 전용(위치 인자 TypeError). 구현 방식은 자유지만 관측 가능한 결과가 아래 불변식을 만족해야 한다. 권장 구조는 부호 CoC로 깊이 슬라이스를 나누고, 슬라이스마다 전치 알파(rgb·coverage)를 조리개 커널로 퍼뜨린 뒤 먼 것부터 가까운 것 순서로 over 합성하는 방식이다. 가려진 배경은 pull-push로 채우고, 크기는 이산 커널 사이를 선형 보간한다. CoC > threshold인 슬라이스는 `defocus_level` 단계에서 **위상 보존** 다운·업샘플(FND-06 `pyr_down`/`pyr_up` 같은 텐트 계열)로 처리하고, 추가된 블러(E[r²])를 커널 지름에서 보상한다.
- **apply_dof_theta**: `theta.focal_mm/f_stop/sensor_width_mm`, `focus_m` 인자가 없으면 `theta.focus_m`. `opts=None`이면 `DofOpts(blades=int(theta.extra.get("blades", 0)))`. 같은 인자의 `apply_dof`와 바이트까지 같아야 한다.
- **rack_schedule**: `t_i=i/fps`, `dur = duration_s` 또는 `(n−1)/fps − start_s`(dur ≤ 0이면 전부 from), `u=clip((t−start)/dur,0,1)`, `e=u`(linear) 또는 `3u²−2u³`(smooth), **디옵터 보간** `1/z = 1/z0 + (1/z1−1/z0)·e`. 끝점은 정확히 입력값이고 inf도 허용한다. n=1이면 `[from]`. n<1·비정수, fps≤0, focus≤0·NaN, ease 미지, start<0, duration≤0·비유한은 ValueError.
- **breathing_scale** = `(1+β·m(z))/(1+β·m(z_ref))`, `m(z)=f/(z−f)`(m(inf)=0). |β|>1, NaN, z≤f면 ValueError. 배열 z를 받으면 배열을 돌려준다. **apply_breathing**: `out(p) = in(c + (p−c)/s)`, c=(K[0,2],K[1,2]), 쌍선형·가장자리 클램프(`nearest=True`면 최근접, 깊이용), s=1이면 정확한 복사, (H,W)와 (H,W,C) 모두 받는다.
- **apply_dof_frames**: `frames.aov["depth"]`가 필요하고 N·H·W가 shot과 같아야 한다(아니면 ValueError). 프레임마다 `lin = color.srgb_to_linear(rgb[i])`. β≠0이면 `s_i = breathing_scale(theta.focal_mm, focus_m[i], β, ref_focus_m=focus_m[0])`로 rgb(쌍선형)와 depth(최근접)를 `shot.K[i]` 주점 기준으로 확대한다. 이어서 `apply_dof(theta 렌즈, focus_m[i], opts or DofOpts(blades=theta.extra.blades|0))`를 적용한다. 출력은 `Frames(rgb=linear_to_display, fps, aov=입력 aov 유지 + depth(브리딩 반영) + "coc_px" (N,H,W) f32, 해시 유지)`이고 입력은 바꾸지 않는다.

## 불변식 (테스트 허용 오차)
1. 모든 CoC ≤ in_focus_px → rgb 비트 동일 복사. 결정론(같은 입력 → 같은 바이트), 입력 불변, NaN·경고 없음, +inf 깊이 처리.
2. 볼록성: 출력은 채널별 입력 [min,max] 안(1e-4). **상수 색 + 임의 깊이 계단(근·원·inf)이면 출력은 상수(1e-3)**: 테두리·불연속에서 링·헤일로가 없어야 한다.
3. 균일 평면 위 임펄스: 에너지 보존(0.2%), 중심 이동 <0.25 px(피라미드 단계 포함), `E[r²]=c²/8`(±18%), 반경 c/2+0.1c+1.5 밖 질량 <3%. CoC 4→36 px 스윕에서 스폿 E[r²]는 엄격 단조 증가한다.
4. 텍스처 평면: 이상 원판 컨볼루션 오라클과 내부 MAE <0.02(c=6, 14, 30). 피라미드 on/off 차이는 내부 MAE <0.02, max <0.06.
5. **칼선 방지 A**(선명 전경 / 녹은 배경 c>10): 전경 내부 불변(0.02), 배경으로 전경 색 번짐 <0.1, 배경 정규화 >0.9(어두운 링 없음).
6. **칼선 방지 B**(녹은 전경 c∈(12,30) / 선명 배경): 실루엣을 가로지르는 프로파일이 단조이고, 인접 화소 간 점프는 <2.5/c+0.02, 10–90% 폭은 [0.4c, 1.3c], 실루엣 밖 c/4 지점의 전경 기여 >0.08, c/2+3 px 밖 배경은 불변(0.02).
7. 랙 연속성: CoC 0.5 px 간격 스윕에서 프레임 간 변화량이 이웃의 2배(+3e-4)를 넘지 않는다(8·16·32 px 단계 경계에서 튐 없음). 근·원 두 평면 랙에서 근평면 선명도는 단조 감소, 원평면 선명도는 단조 증가하고 끝점에서는 원본 선명도(1e-3)다.
8. 속도: 320×180, 최대 CoC >35 px 한 프레임 <6 s(CPU).

## 수용 기준
`tests/lns_02/` 전부 통과(CPU, <60 s, 결정론, 동결 해시 `LNS-02.tests.json`). 작성자는 스크래치 프로토타입(슬라이스+pull-push+피라미드, 약 200줄)으로 52개 전부 2.5 s 안에 통과하는 것을 확인했다. 수신 화소 CoC로만 gather하는 순진한 구현은 5·6·7에서 실패한다.

## 하류 이음매
- CAM-* 단위는 `rack_schedule` 출력을 그대로 `Shot.focus_m`에 넣는다. 렌더 단계에서 브리딩을 하려면 `breathing_K`로 프레임별 K를 만든다(그때는 `apply_dof_frames`의 β=0).
- LNS-05/06(수차·처방)은 `aperture_kernel`·Defocus Pyramid 단계 위에 반경별 커널(cat-eye, 스월, 애너모픽 squeeze)을 얹는다. 시그니처를 바꾸면 이 계약서와 함께 올린다.
- LNS-03(셔터)·LNS-04(센서·필름)는 `apply_dof_frames` 출력 `Frames`(표시값) 뒤에 붙는다. `aov["coc_px"]`는 MRH defocus 헤드 학습 라벨과 IR_info(φ_eff) 계산에 쓴다.
