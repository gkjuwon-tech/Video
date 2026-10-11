# LNS-08 θ 사슬 합성기 + ExplainAway 역사슬 왕복

**목적.** 화면 고정 스택(WB→EV→비네팅→할레이션→센서→65³ LUT→그레인)을 θ 하나로 합성하는 `look`과, 그 **수치 역사슬** `explain`을 준다. 성질 `explain(look(x))≈x`(소유 영역, **증명 가능한 오차 한계**)와 `look(explain(y))≈y`(치역)가 "관측 소유권"의 토대다: 역변환이 닫힌 화소만 소유(owned)하고, 그레인·클립·포화·랭크 결손은 소유권 비트로 가린다. 학습 파라미터 0. 대역 마스크·Wiener(RD-04), 비네팅/그레인 추정(RD-03), 톤 추정(RD-06)은 범위 밖.
**소유.** `buc/lens/chain.py`(`buc/lens/__init__.py` 금지, torch 금지, 전역 RNG·시각 금지), `tests/lns_08/`(동결), 이 계약서. **`tests/test_roundtrip.py`**: 구현자가 만드는 얇은 스모크(공개 API로 `explain(look(x,Theta()))` 소유 100 %·`error_bound` 이내 한 건; 동결 대상 아님, 게이트의 경로 존재 검사용). 의존: FND-02 `Theta`·`Frames`, FND-06 `color`, LNS-01 `buc.lens.theta`, LNS-04 `buc.lens.sensor_film as sf`. DOF·셔터·CPL·광학(장면 의존)은 상류 단계이며 이 모듈은 부르지 않는다.

## 공개 API (`from buc.lens import chain`)
상수: `CHAIN_ORDER=("wb","ev","vignette","halation","sensor","lut","grain")`, `WB_REF_KELVIN=6500.0`, `WB_MIRED_STOPS=0.004`, `WB_TINT_STOPS=1.0`, `OWN_OK=0, OWN_SENSOR_BLACK=1, OWN_SENSOR_SAT=2, OWN_LUT_LOST=4, OWN_HALO_TAINT=8`, `SAT_MARGIN_LSB=1.5`, `LUT_INV_TOL=2e-5`, `HALO_TAINT_EPS=5e-7`, `BLACK_CODE_EPS=2e-3`, `HALO_SOURCE_MAX=16.0`. `__all__`에 모두 포함.
```python
wb_gains(wb_kelvin, wb_tint) -> (3,) f64          # m=1e6/K-1e6/6500; raw=2**[-0.004m, -1.0*tint, +0.004m]; gains=raw/color.luminance(raw)
apply_wb(rgb_lin, theta) -> f32 / remove_wb(rgb_lin, theta) -> f32   # 이득을 f32로 내려 곱함/나눔. HDR 비클램프. 기본 θ면 비트 동일
look(rgb_lin, theta, *, seed=0, frame=0, spec=None, noise=True) -> f32 (H,W,3) 표시값   # == sf.apply_sensor_film(apply_wb(x,θ), θ, ...) 바이트 동일
look_frames(frames: Frames, theta, *, seed=0, spec=None, noise=True) -> Frames   # sRGB→선형→i번째를 frame=i로 look. fps·해시 유지, aov 복사
grain_sigma(display, theta) -> f32                # strength*GRAIN_SIGMA_MAX*2*sqrt(u(1-u)) (표시 영역 화소 σ)
invert_look(display, lut: str) -> (s_lin f32 (...,3), status u8 (...))   # apply_look(·,lut)의 역: 센서 출력 선형값. 비트는 BLACK/LUT_LOST만
@dataclass(frozen=True) Explained: linear f32 (H,W,3); status u8 (H,W); error_bound f32 (H,W,3); owned -> bool (H,W) (== status==0)
explain(display, theta, *, spec=None) -> Explained
```
`wb_kelvin`은 카메라 WB 설정이고 일광(6500 K) 장면에서 낮을수록 푸르다(b/r 증가); tint>0은 마젠타(g↓). 입력 규칙은 LNS-04와 동일: float32/64만(TypeError), NaN/inf·잘못된 모양·범위 밖(`explain`·`invert_look`·`grain_sigma`의 표시값은 [0,1])은 ValueError, 비 `Theta` TypeError, 미등록 lut ValueError, `seed/frame` 검증은 LNS-04에 위임, `spec`은 `SensorSpec`/None. 입력 불변, 출력은 새 배열, 같은 입력→같은 바이트, `noise=False`는 seed·frame 무관.

## 역사슬 `explain` (노이즈 없는 `look`의 역)
순서: `invert_look`(LUT⁻¹·로그 복호) → 할레이션 고정점 `u=max(s-halation_layer(u),0)`(수렴까지) → `/(V·2^ev·g_c)`로 EV·비네팅·**WB 제거** → `linear`는 WB 이전 장면 선형. `spec`(기본 `sensor_spec_from_theta(θ,W)`)에서 ADC 비트만 쓴다(q=1/(2^bits-1)).
- `status` 비트(조합 가능): **SENSOR_SAT** 어느 채널이든 센서 출력 ≥ 1-1.5q(`linear`는 하한값이며 `look`하면 같은 y); **SENSOR_BLACK** `lut!="none"`이고 코드값 e_c<2e-3(로그 바닥 아래, `none`엔 바닥 없음); **LUT_LOST** 룩이 그 색에서 단사 아님(출력 클립 채널·랭크 결손·수렴 실패, `mono_silver`는 전 화소); **HALO_TAINT** `halation>0`이고 포화 지시자의 가우시안(σ=0.02W, truncate 5, 영 경계) 블러 >5e-7(포화 광원의 미지 휘도가 헤일로로 번지는 범위. 광원 노출 ≤`HALO_SOURCE_MAX`일 때 건전). LUT_LOST 화소의 `linear`는 최선 추정(mono는 회색 최소노름)이며 `look∘explain≈id`는 mono 전체와 그 외 LUT_LOST가 아닌 화소에서 성립.
- `error_bound`(소유 화소): `(q/2 + LUT_INV_TOL)·(1 + a·tint_c/(1 - a·Y_H)) / (g_c·2^ev·V)`, a=halation, tint=`sf.HALATION_TINT`, Y_H=`color.luminance(tint)`, V=`sf.vignette_gain`. 비소유 화소는 `+inf`. 이 식이 **건전**해야 한다: 소유 화소는 `|linear-x|≤error_bound`.
- `invert_look` 정확도: 소유(status==0) 색에서 센서값 오차 ≤`LUT_INV_TOL`(65³ 삼선형 보간 자체의 역, `none`은 sRGB 복호).

## 불변식
1. **건전성(거짓 소유 없음):** 임의 θ(LNS-01 샘플러 포함)·장면에서 `owned ⇒ |explain(look(x,θ,noise=False)).linear - x| ≤ error_bound`. 2. **완전성 하한:** 포화 없는 중간톤 장면에서 `none/neutral_709/bleach_bypass/faded_matte` ≥98 %, 클립 룩도 ≥80 % 소유; 클립 없는 룩의 비포화·바닥 위 색은 항상 소유. 3. `look∘explain≈id`: LUT_LOST가 아닌 화소(+mono 전체)에서 표시값 차 ≤2e-3. 4. 소유 화소 `status==0`, 알려진 비트 외 없음. 5. 노이즈 경로(`noise=True` 입력)도 유한값을 내고 평탄 패치에서 무편향(|평균오차|<3e-3). 6. 모듈 import에 torch 없음, 입력 불변.

## 수용 기준
`tests/lns_08/` 전부 통과(CPU, 60 s 이내, `-m "not gpu"`, 동결 `LNS-08.tests.json`): 공개 표면·상수·오류 경로·결정론·격리, WB 수식·단조·휘도 보존, `look`이 LNS-04 공개 조각 합성과 일치(WB 선행), 10룩 전부의 `invert_look` 정확도·BLACK/LUT_LOST 의미·클립 룩의 거짓 소유 없음, `error_bound`가 위 식과 일치(비트수·비네팅·EV·WB·할레이션 반영)하며 건전, SAT/BLACK/HALO_TAINT 비트의 위치·건전성, 할레이션이 실제로 제거됨(층 무시 시 실패), `look∘explain`, 멱등, 룩 교체 재촬영, 그레인 σ(몬테카를로 ±6 %), 노이즈 경로, 144×256 `explain` <8 s. 구현 지침: `explain`은 3.5k 화소 장면을 ≈0.3 s 안에 끝낼 것(전체 스위트 예산 60 s). `V`(사람 판정)·GPU 이식은 범위 밖.

## 하류 이음매
RD-04(ExplainAway)는 `explain`의 `linear/status/error_bound`를 받아 대역 마스크·그레인 Wiener를 얹고, 그레인 σ는 `grain_sigma`로 노이즈 가림 규칙(σ_read≥σ_grain)에 쓴다. RD-03/06은 `look`을 순방향 오라클로 쓴다(추정 θ̂의 자기 일관 검사). DAT-03/09는 `look`/`look_frames`로 θ 적용본을, `explain`으로 ExplainAway 통과본을 만든다. 룩 교체: `look(explain(look(x,θ₁),θ₁).linear, θ₂)`가 소유 영역에서 `look(x,θ₂)`와 일치. WB는 LNS-04 요청 (1)에 따라 여기서 소유하고 EV는 LNS-04 소유라 이중 적용 금지.
**요청(범위 밖).** (1) 포화 광원의 HDR 휘도를 헤일로 에너지로 역산하는 일은 RD-03 몫(여기선 HALO_TAINT로 가림). (2) 상수 변경은 호환 파괴: 계약서와 함께 올린다.
