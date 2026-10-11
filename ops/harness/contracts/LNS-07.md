# LNS-07 CPL 편광 필터: Rayleigh 하늘 편광 · 필터 축 · Fresnel 반사 편광 v0

**목적.** 렌즈 앞 원편광(CPL) 필터를 **장면 선형 영상**에 곱하는 한 줄 수식(PARADIGM §7.6). 하늘은 단일 산란 Rayleigh 편광도 `DoP(γ)=P_max·sin²γ/(1+cos²γ)`(γ=시선과 태양 사이 각), 편광 방향(E벡터)은 산란면에 수직. 필터 축 φ_f에서 화소 이득 `g = (1-s) + s·τ·(1 + DoP·cos2(φ-φ_f))`(s=θ.cpl). 태양과 90°인 하늘이 가장 짙어지고 축을 돌리면 밝기가 오간다. 학습 파라미터 0, 순수 수식, 난수 없음.
**소유.** `buc/lens/cpl.py`(`buc/lens/__init__.py` 금지, 네임스페이스), 테스트 `tests/lns_07/`(동결 `LNS-07.tests.json`). 의존: LNS-04(체인 위치만, 코드 import 없음), FND-02 `buc.core.contracts.Theta`, LNS-01 `buc.lens.theta.EXTRA_DEFAULTS`, numpy. **torch·scipy 금지**, 전역 RNG·시각·경고(RuntimeWarning) 금지. 입력 배열 불변, 새 배열 반환.

## 공개 API (`from buc.lens import cpl`)
상수: `P_MAX=0.75`(하늘 최대 편광도), `TAU=0.40`(비편광광 투과율; PARADIGM "T≈0.35–0.45". 완전 편광+축 정렬은 최대 `2·TAU`).
```python
rayleigh_dop(gamma_rad, p_max=P_MAX) -> ndarray f64            # p_max*sin²γ/(1+cos²γ), 모양 유지. p_max∉(0,1] 또는 비유한 -> ValueError
fresnel_dop(theta_i_rad, n=1.33) -> ndarray f64                # 반사광 편광도 |Rs-Rp|/(Rs+Rp). 입사각∈[0,π/2], n>1 아니면 ValueError. 브루스터각 atan(n)에서 1
sun_dir_world(elev_deg, azim_deg, *, up, north=None) -> (3,) f64   # FarFieldSpec.sun_dir 와 같은 규약: cos e(cos a·N + sin a·E) + sin e·U, E=N×U
view_rays(K, h, w) -> (h,w,3) f64                              # 화소(정수 중심) 단위 시선, OpenCV 카메라 좌표 (K⁻¹[u,v,1]) 정규화
sky_polarization(rays, sun_cam, *, p_max=P_MAX) -> (dop (H,W) f64, evec (H,W,3) f64)
polarizer_axis(angle_deg) -> (3,) f64                          # (cos a, sin a, 0): 영상 +x 에서 +y(아래)로 잰다
polarizer_gain(dop, evec, angle_deg, *, strength=1.0, tau=TAU) -> (H,W) f64
apply_polarizer(rgb_lin, dop, evec, angle_deg, *, strength=1.0, tau=TAU) -> rgb   # 일반 진입점(수면·유리 반사용)
apply_cpl(rgb_lin, theta, *, R_w2c, up, north=None, K=None, sky_mask=None) -> rgb  # 하늘 편광 진입점
compensation_ev(strength, tau=TAU) -> float                   # -log2((1-s)+s·tau) >= 0
```
- `evec` = 단위 E벡터(카메라 좌표) `normalize(ray × sun_cam)`; ray∥sun(외적 <1e-9)이면 영벡터이고 dop=0. `cos2φ ≡ 2(evec·axis)²-1`(광축 근처 시선이면 정확히 cos2(φ-φ_f)). 이득은 `(1-s)+s·τ·(1+dop·cos2φ)`: 0<이득, `s=0`이면 정확히 1, 모든 각 평균=`(1-s)+sτ`, 직교 두 축의 이득 합=`2((1-s)+sτ)`, 축 주기 180°.
- `north=None`이면 FarFieldSpec과 같은 규칙(up의 |성분| 최소 축을 up에 수직 투영·정규화). up 영벡터·north∥up은 ValueError.
- `apply_cpl`: `s=theta.cpl`, 축 `theta.extra.get("cpl_angle_deg", 0.0)`, 태양 `theta.sun_elev_deg/sun_azim_deg`(둘 중 하나라도 None이면 태양 없음 -> dop=0, 균일 이득 `(1-s)+sτ`). `sun_cam = R_w2c @ sun_dir_world(...)`. `K=None`이면 `theta.intrinsics(W,H)`. `sky_mask`: (H,W) bool 또는 [0,1] float(부드러운 가장자리), dop에 곱해진다. `None`=전 화면이 하늘(하류가 `~isfinite(depth)`를 넘긴다). 구름·지면처럼 편광 안 된 화소는 마스크 0 -> 이득 `(1-s)+sτ`.
- 영상은 `(H,W,3)` float32/float64 유한(HDR>1·0 허용, 클램프 없음). 내부 float64 계산 후 **입력 dtype으로 되돌린다**. 채널 무관 같은 이득(중성 필터). `s==0`이면 (검증 후) 비트 동일한 **복사본**.
- `sky_polarization`의 `sun_cam`은 (3,) 단위벡터(|‖·‖-1|≤1e-6), `rays`는 `(...,3)`, 아니면 ValueError.
- 오류: dtype 비부동소수 `TypeError`; 모양·비유한·R_w2c 비회전·K 모양·마스크 모양/범위·strength∉[0,1]·tau∉(0,1]·dop∉[0,1] `ValueError`(ContractError 포함); theta가 `Theta` 아니면 `TypeError`. 키워드 전용 인자.

## 불변식
결정론(같은 입력 -> 같은 바이트) · 입력 불변 · 출력 유한·비음수(입력 비음수일 때) · 선형(입력을 k배 하면 출력도 k배) · 이득 ∈ `[τ(1-P_MAX), τ(1+P_MAX)]`(s=1) · 카메라를 광축으로 α(영상 x에서 y쪽이 양) 굴리고 필터 축 φ로 찍은 것 = 굴리지 않고 축을 φ+α로 찍은 것(광축 화소) · 순수 import 부작용 없음.

## 수용 기준
`tests/lns_07/` 전부 통과(CPU, <60 s, `-m "not gpu"`, 동결 해시 `LNS-07.tests.json`): Rayleigh/Fresnel 오라클·특수각(0, 90°, 브루스터), 이득 닫힌 꼴(정렬 τ(1+P), 직교 τ(1-P), 45° τ), 각 평균·직교 합·180° 주기, 태양 규약(FarFieldSpec 대조), 시선 광선 기하, E벡터 직교성, 광각 전 화면 독립 오라클 대조(여러 θ·카메라 방향), 마스크·태양 없음·s=0 항등, 선형성·HDR 보존·dtype 유지, 롤-축 동등성, LNS-04 체인 이음매(`compensation_ev`로 노출 보상 시 중간 회색 복원), 오류 경로, torch 미로드, 360×640 <3 s. 하늘색 취향 등 `U`(사람 시각) 판정은 이 테스트 밖이다.

## 하류 이음매
LNS-08(체인 합성기)이 **LNS-04보다 앞**, 선형 장면에서(LNS-02 렌즈 효과와의 앞뒤는 합성기가 정하되 센서·LUT보다는 앞) `apply_cpl`을 부른다. 프레임마다 그 프레임의 `R_w2c`를 넘긴다(카메라가 돌면 편광 패턴이 움직인다). **노출 보상은 이 모듈이 하지 않는다**: 필터가 실제로 `≈1.32 EV`(s=1) 어둡게 하므로 θ.ev에 `compensation_ev`를 더할지는 LNS-08이 정한다(`ev`는 LNS-04가 적용). 수면·유리는 호출자가 `fresnel_dop`와 `normalize(ray × normal)`(s편광 E벡터)로 `dop, evec`를 만들어 `apply_polarizer`로 넣는다. LNS-08은 `sky_mask`를 깊이 `+inf`에서 만든다.
**요청(범위 밖).** (1) PARADIGM의 `T/2` 접두사와 "T≈0.35–0.45" 서술이 모호해 `TAU`를 비편광 투과율(0.40)로 해석했다. 다르게 정해지면 상수와 계약 동시 변경. (2) 월드 `up/north` 규약이 단일하지 않다: 호출자가 FarFieldSpec과 같은 `up/north`를 넘겨야 태양 방위가 맞는다.
