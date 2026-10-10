# Block Universe Cinema 통합 패러다임 명세 v2.1: 2×T4 + 수십M from-scratch + 미감 연방제 에디션

> **출처 메모.** 이 문서는 v1 기준 문서(`docs/PARADIGM.md` v1.0), 2×T4 제약 리서치(2026-10-10), 4개 렌즈의 원안과 프리모템 개정안, v2.0 초안, 그리고 v2.0 이후에 나온 창업자 결정과 미감 A/B 5판(Higgsfield Soul 하이브리드)을 수석 아키텍트가 하나로 통합한 명세다. 4개 렌즈는 compute-budget-solver, tiny-foundation-models, representation-renderer-demos, phase2-flywheel-shoestring이다.
>
> **v2.0 → v2.1에서 고친 것.**
> 1. **잘렸던 입력 복원.** v2.0을 쓸 때 tiny 개정안은 §5.4 중간부터, representation 개정안은 X-T2 중간부터, phase2 원안은 §16부터만 들어왔다. 세 문서의 전체본을 다시 읽고 놓친 내용을 복원했다. 숫자 교정 장부(§1.8), FP16 Rescue의 SwiGLU 재균형(§3.5), 판정 해상도 분할과 세계별 NFE 장부식(§4.1), OTBI 뷰 수 식과 TABP 닫힌형(§4.1), Teacher-DCA(§5.3), 크로노포토 증인의 세 합성식과 BIS(§9.3), 증인 대체 매트릭스(§9.2), 실리콘 전 유닛 배정(§1.9), Phase 2 비회귀 점검(§4.5), 몰래 들어온 사전학습물 통합 목록(§6.6) 등이다. v2.0이 "맥락으로 정의를 확정했다"고 적었던 PWMT, OTBI/TABP, X-T0/X-T1, Spec-commit은 원문 정의와 대조해 바로잡았다.
> 2. **Soul 하이브리드 통합.** 창업자는 Higgsfield **Soul Cinema / Soul 2.0을 닫힌 미감 교사**(아트 디렉터 + 최종 광택 + 개념 아트)로 채택했다. 이 결정을 판결록(§0.5), 4분류(§2), 이미지 모델 선정(§3), 파이프라인(§4), 카탈로그(§5), 데이터 계보(§6.6), 카메라 스택(§7.6, §7.8), 쇼케이스와 벤치(§8), 90일 계획(§11), 리스크(§13), 열린 질문(§14)에 일관되게 반영했다. 새 절 **§3.8 미감 부검 결과와 미감 연방제**를 넣었다.
> 3. **카탈로그 정본 이름.** 렌즈마다 달랐던 수십M 모델 이름을 정본 하나로 통일하고 별칭표(§5.2)를 남겼다. 파라미터 합계와 2×T4 학습 시간을 다시 계산했다(§5.6).
> 4. **v2.1 감사 반영.** 정합성 비평 40건과 누락 비평 36건을 반영했다. 비회귀(세계 경유 광택의 위치), 예산 구멍(Hybrid-A 세계), Plan S 범위와 메일 문구, Soul 방화벽의 기생 학습 경로가 핵심이다. 반영 내역과 미반영 사유는 부록 C에 있다.
>
> **렌즈 약어.** CB = compute-budget-solver, TF = tiny-foundation-models, RR = representation-renderer-demos, P2 = phase2-flywheel-shoestring. (o)는 원안, (u)는 프리모템 개정안이다. FD = 창업자 결정과 미감 A/B 문서.
>
> **표기.**
> - 신규성: [신규] 이 제약 때문에 새로 나온 것, [조합] 부품은 있고 결합이 새로운 것, [부품] 차용한 것, [운용] 운영 규칙.
> - 근거: [검증] 1차 출처, [2차] 2차 출처, (추정) 계산 가정, (실측) E0에서 숫자로 바꿀 값.
> - 배치 태그: `P1-코어`, `P1-확장`, `P2`, `장기`.
> - 4분류: (a) 빌림(오픈 엔진 교사), (a′) 빌림(닫힌 미감 교사 Soul, 콘텐츠 전용), (b) 계산(학습 없는 구조), (c) 테스트타임 최적화(TTO), (d) 만듦(수십M from scratch).
> - 승인 단계: **Plan A**(승인 없이: 콘텐츠와 사람 경유 룩), **A+**(세계 피팅 승인), **S**(소형 모델 학습 승인). §3.7.4. S의 하위 단계 S1·S2는 파이프라인 단계 S0–S11(§4.2)과 다른 이름이다. 스케일 원칙은 SC1–SC9(§10.3)로 쓴다.
> - 계보 태그: `SOUL`(Soul 원본 픽셀), `SOUL-DERIVED`(Soul 픽셀을 입력으로 계산한 산출물), `SOUL-INSPIRED`(사람이 Soul 스틸을 보고 만든 파라미터·스크립트). §6.6.

---

## 0. 테제, 슬로건, 무엇이 바뀌었나, 판결록

### 0.1 한 줄 테제 v2.1

> **비디오는 관측이다. 관측에 드는 진짜 비용은 이미지 교사의 판단 하나뿐이다. 우리는 그 판단을 의미가 사는 대역에서만 사고, 나머지는 물리·광학·기하가 공짜로 주는 것과 우리가 직접 키운 작은 서기들로 메운다. 그리고 미감은 한 모델의 것이 아니라 연방의 것이다. 카메라와 빛과 필름이 대부분을 소유하고, 개념과 미술은 닫힌 미감 교사가, 재질과 기하는 오픈 엔진 교사가 소유한다.**

- v1 테제는 그대로다. 확률은 세계(Phase 2부터는 사건)에만 둔다. 이미지 디퓨전은 심판이다. 카메라는 읽기만 한다.
- v2는 "심판 호출이 곧 화폐"라는 회계 원리를 더했다.
- v2.1은 **미감 연방제**를 더한다. 빌린 눈은 둘이다. 하나는 측량 기사(오픈 엔진 교사)이고, 하나는 아트 디렉터(Soul)다. 아트 디렉터의 붓은 세계와 최종 프레임에만 닿는다. 우리 서기들의 교과서에는 한 획도 들어가지 않는다(Soul 방화벽).

### 0.2 슬로건

- 메인(유지): **"그리지 말고, 찍어라."**
- v2 보조: **"대신 아껴서 찍어라."**, **"배우지 말고 굴려라, 상상하지 말고 찍어라."**, **"사진이 모르는 것은 사진에게 묻지 마라."**
- v2.1 보조: **"측량은 Z가, 촬영은 우리가, 미술은 Soul이."**, **"영화 같음의 8할은 카메라에 있다."**
- 피치: **"남이 이미 낸 10²³을 빌리고, 물리가 공짜로 주는 일관성을 줍고, 우리 돈은 '이 세계'와 '작은 접착제'에만 쓴다. 최고의 미감은 빌려 오되, 손은 우리 것이다. T4 두 장, 냉장고 세 대 분량의 전기로."**

### 0.3 v1 → v2.1: 바뀐 것과 안 바뀐 것

| 항목 | v1 | v2.1 | 이유 |
|---|---|---|---|
| 목표 (Phase 1·2 승리 조건) | §7.1, §8.5 | **그대로** | 창업자 지시. 하나도 낮추지 않는다 |
| 인수분해 E1, 정리 T1–T10, WNF, 대역 소유권, 비회귀 조항, DTC, PSH, FREEZE-TAG | 핵심 | **그대로** | 하드웨어와 무관한 이론이다. 제약 아래서 오히려 생존 조건이 된다 |
| 컴퓨트 | H100 64장, 0.4–0.8M H100-h | **2×T4, 90일 4,320 T4-h, 실효 2.3–3.1×10²⁰ FLOPs** (v1 90일 계획의 약 1/400–1/600) | 하드 제약 |
| **엔진 교사** [v2.1 개명] | 12–20B 오픈 교사 | **Z-Image-Turbo 6.15B(Apache-2.0, 동결, 정밀도 사다리). E1에서 통제성·속도·라이선스 기준으로 재오디션**(FLUX.2 klein 4B, Z-Image Base, ERNIE-Image-Turbo) | 구조·노이즈·프로브·래칫에는 가중치와 σ에 접근할 수 있는 교사가 필요하다 |
| **미감 교사** [v2.1 신규] | 닫힌 SOTA art director(약관 확인), 닫힌 API POLISH 소유자 후보(v1 §1.6 (iii)) | **Higgsfield Soul Cinema / Soul 2.0. 닫힌 모델, 콘텐츠 전용**: 아트 디렉터, 최종 광택, 개념 아트, 평가 앵커. **Soul 방화벽** | 창업자 결정. A/B 50축 중 Soul 41승. v2.0은 "두 번째 이미지 모델"이라며 금지했다가 v2.1에서 부활 |
| **미감 소유권** [v2.1 신규] | 교사 + DTC + Gaffer | **미감 연방제**: 10개 미감 축 중 약 8개(화면비, 렌즈 원근, 심도, 시간대·광원, 노출, 그레이딩, 필름 질감, 프레이밍)는 물리 카메라·조명·필름·DP-0가, 2개(미술 생활감, 인물·서사·개념)는 Soul이, 재질·기하는 엔진 교사가 소유 | A/B에서 Soul 승리의 약 78%가 물리 스택 소관(§3.8) |
| World Program | LLM | **MassingFM(24M)**(절차 문법으로 학습) + CaptionComposer(코드) | 대형 LLM 금지 |
| Bedrock | CWM 4–8B | **Picture-First**: hero(엔진 교사 "측량 모드", 또는 A+에서 Soul 개념 아트) → MRH 독심 → MassingFM → 의미 패스 lockstep | 10²³ 학습 불가. hero 구도에 이미 10²³이 들어 있다 |
| 3D 지각 | DA3/VGGT/MoGe/TRELLIS/SigLIP | **MRH(24M)**(독심 헤드: 엔진 교사 특징을 읽는다) | 사전학습 3D 모델 금지 |
| 미세 레벨 solver | WSSC v3, 뷰 중첩 투표 + Adam 피팅 | **WSSC-T4**: Grid-Locked OTBI 차트, 닫힌형 동기화(TABP), VAE·Adam 0 | 16 GB OOM, 교사 시간 절약 |
| Student | LBO 200M + WDM 4–8B | **TexelFM-35**(디테일 연산자, 가난선 아래) + 관할 이양 래칫. WDM은 연기 | 교사 역전파가 T4에서 비싸고, 장면 규모 수렴 선례도 없다 |
| 정본 | latent z + UKA 캐시 | **explicit TSQ + BC7/BC5/BC6H 페이지** (`z_slot` 예약) | 디코더가 없다. BC 디코드는 하드웨어가 공짜로 해 준다 |
| 렌더러 | 3DGRUT 하이브리드 | **gsplat-TSQ 래스터 + OptiX(삼각형)/Warp BVH + 평면 재귀 + DIEL SDF** | 1세대 RT 코어 40개, any-hit 비용 |
| 렌즈 라이브러리 | 처방 + 수식 | **+ 빈티지 수차 처방(필드 커버처, 스월, 주변부 번짐), 원편광(CPL) 필터, 룩 프리셋 θ_look** [v2.1] | A/B가 보여 준 "영화 같음"의 출처 |
| Phase 2 비디오 증인 | 저주파 증인(Wan) | **금지** → 크로노 증인(GICC) + 실제 증인 + 감독 | 비디오 모델 금지. H7이 본선으로 승격 |
| 화폐 | H100-h | **TFU**(엔진 교사 forward 1회, Z-Image 1024² 기준). Soul은 T4 밖의 **크레딧**으로 따로 계상 | 병목은 학습 FLOPs가 아니라 교사 호출이다 |
| 카탈로그 이름 | — | **정본 이름 + 별칭표**(§5.2) [v2.1] | 렌즈마다 같은 모델을 다른 이름으로 불렀다 |

### 0.4 한 장 요약

```
 y (프롬프트 / ShotScript / WorldScript)
   │  Qwen3-4B TE (encoder 전용, 세계당 1회 캐시) ⊕ CaptionComposer (템플릿, 모델 0)
   │
   ├─[Concept·Look] Soul (닫힌, 콘텐츠 전용)                                                     (a′)
   │        A : 사람이 웹에서 개념 아트 1–4장 → 사람이 룩 측정 → θ_look 프리셋, WorldScript 전사
   │        A+: MCP 개념 아트 → (설명 제거 후) 엔진 교사 약한 σ 통과 → MRH 독심 판독 → 관측 제약
   │        prov=SOUL → soul_vault.  데이터셋·손실·보상·보정·체크포인트로는 한 바이트도 가지 않는다
   ▼
 [Hero]     엔진 교사 "측량 모드" hero search 64@512² → 4@1024² × MRH-Aes ⊕ Teacher-DCA × Corman 비용 × Glyph 벌점   (a)+(d)+(c)
   ▼
 [Bedrock]  MRH lift → MassingFM(24M): shell·slot·광원 → 문법·CC0 에셋 → L0 조명 bake       (d)+(b)
   │        의미 패스: 48 cine 뷰 @768² lockstep, G-Adapter, TSQ 미분 피팅, CMA-ES 거부권    (a)+(c)
   │        Swatch Bank: 재질 × 레벨 견본 (bedrock commit의 일부, 순수 함수)                (a)
   ▼
 [Demand]   CamFM prior + Ray-Bundle + Corman–Shannon ("카메라가 기록 못 하면 만들지 않는다") (b)+(d)
   ▼
 [Strata]   교사 관할: Grid-Locked OTBI 차트 + Fold 뷰 → 닫힌형 고정소수점 sync (VAE 0, Adam 0) (a)+(b)
   │        Spec-commit 구간: TexelFM 초안 → 교사 1-NFE 판정 → 거부 블록만 소유권             (d)+(a)
   │        가난선 아래: TexelFM native, 비트 결정론(T8)                                      (d)
   ▼
 [Audit]    적대 카메라 CMA-ES @512² → 열린 대역만 재판정 (CoW)                              (c)
   ▼
 [Plan]     CamFM(의도·손·CamFM-CRM) + 1 kHz 리그 + Gaffer(θ_look) + DP-0(전경 레이어링, 스케일 레시피) → Γ   (b)+(d)+(c)
   ▼
 [Polish]   (선택, A+) Soul 세계 경유 POLISH: Γ 키프레임 렌더 → Soul 구조 보존 광택 → CoW로 다시 연 1개 Jacobi 라운드
            (엔진 교사 뷰와 동시, 소유 마스크 ∧ Soul 2K Nyquist 이하) → 엔진 교사 재판정 → commit(prov=authored)   (a′)+(a)
   ▼
 [Shoot]    TSQ 래스터 + RT → 렌즈(빈티지·CPL)·센서·필름(θ_look) → 4K/8K · 120fps                (b)
            저장 = (Merkle 루트, Γ, Soul 자산 해시). 프레임은 캐시일 뿐이다. 렌더 경로에 생성 연산은 0이다.
   ▼
 [Still]    (선택) Soul 스틸 광택(키 아트, 포스터, 썸네일) = 콘텐츠(A). 세계로 돌아오지 않는다
            비디오 프레임별 광택은 금지(프레임마다 다시 그리는 것은 우리가 이기려는 그 병이다)    (a′)
   ※ 모든 엔진 교사 호출은 Teacher Exhaust Ledger에 기록되고, 같은 호출이 여섯 군데에 쓰인다. 그중 두 군데가 학습 데이터다.
     Soul이 개입한 호출은 Tap Bus를 끈 별도 프로세스에서 돌고, SOUL-DERIVED로 표시되어 학습 분할에서 자동으로 빠진다.
```

### 0.5 렌즈 충돌 판결록 v2.1

| # | 쟁점 | 대립안 | 판결 | 근거 |
|---|---|---|---|---|
| CT1 | 빌릴 교사 | SDXL [TF(o), RR(o)] vs Z-Image-Turbo [CB, TF(u), RR(u), P2] | **엔진 교사 = Z-Image-Turbo(현직).** E1에서 klein 4B, Z-Image Base, ERNIE-Image-Turbo와 엔진 점수로 재오디션한다(§3.2). SDXL은 1–2주차 개발 대조군으로만 쓴다. **미감은 CT31로 분리했다** | v2.0의 근거(Elo 941 대 678, 기대 승률 82%)는 엔진 교사의 재질 품질로서 여전히 유효하다. 다만 미감 상한은 이제 연방(CT36)이 정한다. 라이선스는 Apache-2.0 하나로 끝난다 |
| CT2 | 교사 천장 합격선 | 55% [TF(u), BT 역산] vs 60% [RR(u), CB(u)] | **엔진 트랙: 녹색 ≥60%, 황색 55–60%(DTC, Gaffer, LoRA 의무), 적색 <55%.** 하이브리드 트랙: Soul 천장 SP_S(E1-S)를 재고, 우리 렌더 대 Soul 스틸 기준선을 **TP_S\* = 1 − SP_S**로 둔다. **이 기준선은 사람 보고서의 진단 항목이다. 게이트도 승리 조건도 아니고, 엔진 교사 잠금·승급·트랙 개방에 쓰지 않는다**(§8.1은 그대로, Q6) | Bradley–Terry에서 logit은 더해진다. SP_ours = 50%이려면 TP = 1 − SP_anchor다. v1의 TP ≥45%는 SP_E = 55%에 정확히 대응한다 |
| CT3 | 판정 경로 | Base(판정) + Turbo(샘플) [P2] vs Turbo 단독 + σ 스냅 [CB, RR] | **기본은 Turbo 단일 가중치 + 학습된 σ 점 스냅.** Base 병용은 창업자 Q1. E0에서 Turbo의 DR 판별 AUC가 0.7 미만이면 SR-LoRA(t-densify LoRA, 약 70M, 1.1%)에 착수한다 | 미감이 Soul로 넘어가면서 Base(비증류, 연속 σ, CFG)의 상대 가치가 올랐다. Q1을 재상정한다 |
| CT4 | 학습 실효 TFLOPS | 20 [CB(o)], 15 [CB(u)], 13 [RR], 12 [TF, P2] | **큰 GEMM 모델(d≥512) 15 TF/장, 소형(<50M, d≤448) 12 TF/장, 3D conv 6–8** | HGEMM 실측 최고 41.6 TF [검증], 70W 쓰로틀, roofline 능선 130(실측)–203(피크) FLOP/B |
| CT5 | 비용 화폐 | TFU [CB], NFE-eq [TF], 교사-초 [RR], TC@512² [P2] | **TFU 하나.** Z-Image 1024² forward = 6.2×10¹³ FLOPs. 512² = 0.25 TFU, 768² = 0.56 TFU. TF의 NFE-eq와 같은 단위다. Soul은 T4를 쓰지 않으므로 **크레딧**으로 따로 센다 | 모든 실험과 게이트를 한 단위로 비교하려는 것이다 |
| CT6 | 미세 레벨 동기화 | Grid-Locked 겹침 ω=2 soft [CB] vs OTBI 단독소유 + TABP [RR] vs Chart-Direct Sync [TF] | **Grid-Locked OTBI**: 세계 축 정렬 차트 위의 단독소유 타일 + 25% apron, 닫힌형 고정소수점 합. VAE 0, Adam 0 | T6에 따르면 소유 영역 안의 SFR=1은 K와 무관하다. apron 밖의 중첩은 TFU 낭비다. 세 렌즈 모두 Adam 제거에 동의했다 |
| CT7 | 교사와 학생의 관할 경계 | 가난선 o_s(m) [CB] vs 판정해상도 분할 + Spec-commit [TF] vs Federal Router [RR] | **관할 이양 래칫 하나로 통합**(§4.1): 교사 소유 / Spec-commit / TexelFM native의 3구역. Router가 의미 셀을 이양 대상에서 뺀다. 초기 o_s = max(1080p 판정 경계, E2 측정값) | 세 제안은 v1 헌법(관할은 측정으로 정한다)을 각자 다른 단위로 자른 것이다 |
| CT8 | 미세 연산자 구조 | BO-40 s2d + 마모장 [CB], ICX-LBO 26M 견본 + 2×2 평균0 [TF], TexelFM 30M patchify [RR] | **TexelFM-35 하나**(정본명, 별명 "디테일 연산자"): s2d×4 + 견본 cross-attn + 마모장 + 2×2 평균0 사영 | 네 장치가 서로 다른 실패를 막는다. s2d는 roofline, 사영은 효도, 마모장은 새집 CG 룩, 견본은 재질 정체성을 맡는다 |
| CT9 | 견본 출처 | Texel Bank(타 세계) [CB], Swatch Bank(세계 내) [TF(u)], PWMT(세계별 TTT) [TF(o)] | **Swatch Bank 1순위. Texel Bank는 학습 데이터와 타 세계 검색 전용. PWMT는 폐지**(PWMT-FiLM은 E4 대조군으로만) | PWMT(Per-World Material TTT)는 MSE면 블러, flow면 타일링 암기를 낳고, GPU1에서 세계당 15–30분 경합하며, T8을 잃는다. Swatch는 (seed, m, l, y)의 순수 함수라 T8을 되살린다 |
| CT10 | Bedrock 순서 | 프로그램 우선 [CB] vs Picture-First [TF, RR] | **텍스트 모드는 Picture-First + Program-Backed, directed 모드는 DSL 프로그램 우선.** v2.1: Soul 개념 아트는 A에서는 사람이 DSL로 전사하고, A+에서는 hero와 같은 관측 제약(DPS)이 된다 | CB 자신의 F2 분석에서 프로그램이 L1을 정하면 데칼 룩이 되고 TP가 28%로 떨어진다. hero는 관측 제약이지 캔버스가 아니다 |
| CT11 | 레이아웃 모델 | WP-30 [CB], MassingFM [RR], BRP 42M [TF], CaptionComposer [TF] | **MassingFM(24M, 코어, WP-30을 흡수) + BRP(42M, `P1-확장`, 지형과 유기 장면) + CaptionComposer(코드)** | 실내와 거리의 엔트로피는 슬롯 문법으로 덮인다. 볼륨 완성은 비정형 장면에서만 필요하다 |
| CT12 | 교사 특징 헤드 | Free-rider probe 20M [CB], MRH-D 22M [TF], GeoHead 25M + AesHead 5M [RR] | **MRH(24M) 하나**(정본명, 별명 "독심 헤드"; Aes와 FOV 헤드 포함). 미감 점수는 MRH-Aes ⊕ Teacher-DCA(파라미터 0) 앙상블. 한계비용은 교사 fwd의 1.5% 이하(FP16 1024² 기준, INT8 기준 1.5–2.3%) | 같은 역할이다. 탭을 한 번만 뽑아야 PCIe와 디스크를 아낀다 |
| CT13 | 학습용 지각 손실 | T-LPIPS-S [CB], TPD(교사 특징 직접) [TF], PercepNet-S 3M [P2(u)], Laplacian + 교사 특징 [RR] | **T-LPIPS-S 4M 하나**(PercepNet-S를 흡수). 교사 특징 직접 거리(TPD)는 라벨과 평가에만 쓴다 | 손실 안에서 6B 교사를 역전파하면 16 GB를 넘고 비용이 3배가 된다 |
| CT14 | 루프 안 VAE | 0회 [CB] vs 노이즈 마스킹 규칙 [TF] vs TL은 미리보기 전용 [RR] | **평면 차트는 VAE 0회. 곡면·투시 경로는 노이즈 마스킹 규칙(TAE-T). commit 마지막 스텝은 원 VAE 1회** | 근사 오차는 노이즈가 가려 주는 곳에서만 허용된다. 결정 대역을 가장 약한 부품이 디코딩하면 안 된다 |
| CT15 | 미분 래스터 | gsplat-TSQ [RR] vs nvdiffrast + 차트 메시 [CB] | **gsplat-TSQ 하나.** 1–2주차는 2DGS 그대로, 3–6주차에 TSQ로 확장한다. 차트는 TSQ 페이지의 UV 묶음이다. RT는 OptiX(TSQ 코어 삼각형)를 쓰고 안 되면 Warp BVH로 폴백한다 | 렌더러 범위 폭발을 막고 train/test gap 0 원칙을 지킨다 |
| CT16 | 8K 프레임 시간 | 0.3–0.8 s [CB] vs 1.1–2.5 s [RR] | **RR 보수치로 계획하고 CB는 낙관 경계로 둔다.** E0에서 실측 | dual-cut과 fill-rate까지 세면 RR이 정직하다 |
| CT17 | 세계 1개 계획 비용 | 4 T4-h [CB], 0.9–1.5 [RR], 약 0.7 [TF] | **계획 2 T4-h(Mode A INT8), 2.8(Mode B FP16), 최악 6** | §4.4에서 다시 계산했다. 순수 1.0 T4-h에 공학 오버헤드 ×2 |
| CT18 | 두 장 배치 | INT8 두 반구 [CB], Mode A/C/B [RR], fp16 파이프라인 분할 [TF], GPU-A 교사 전용 [P2] | **Mode A(양자화 복제) 기본. Mode B는 TF의 fp16 파이프라인 분할. Mode C는 σ-카드 분할.** 운영은 주야 시분할로 한다 | 단독 14.1 GB는 OOM 여유가 없다. 파이프라인 분할은 단일 장 대비 1.7–1.9배 처리량이다 |
| CT19 | 광류 | Flow-S 5.3M [CB], RAFT-S 1M [TF], DIS/TV-L1 [RR], R-Flow 3M [P2] | **P1 시스템 경로는 고전 DIS/TV-L1. 학습 광류는 P2 Reader-S 안의 R-Flow 3M 하나** | P1에서 광류는 dream jitter 정렬 용도라 고전 알고리즘으로 충분하다 |
| CT20 | 카메라 모델 크기 | 25/8/15 [CB], 19/8/12 + Rig-ID 3 [TF], 15/5/8 [RR] | **CamFM-Intent 20M, CamFM-Hand 8M, CamFM-CRM 12M, Rig-ID-S 3M** | 데이터 규모(합성 궤적 수백만, IMU 300 기기-시간)에 맞췄다 |
| CT21 | IMU 규모 | 300 h [CB, TF] vs 100 h [RR] | **300 기기-시간**(6대 동시 기록이면 운영자 약 50시간). D60까지 100 | 기기-시간과 사람-시간을 구분하면 둘 다 맞는 말이었다. 실토큰 5,400만은 2.7M 모델의 Chinchilla 점이라 8M에는 증강 ×5.5가 필수다 |
| CT22 | 합성 GT 생산 | Cycles 15만 + Infinigen 300만 [CB(o)] vs 자체 렌더러 [전원(u)] | **자체 렌더러(같은 코드 경로) + 자체 문법.** Infinigen은 3–5k 장면과 에셋만. Cycles는 디노이저를 끈 레퍼런스(ΔJOD) 전용 | 원안대로면 숨은 비용이 1,250–5,000 T4-h와 40–65만 CPU-h다 |
| CT23 | D3 데이터(사전학습 모델이 라벨한 공개셋) | 전원 기본 금지 | **시스템 경로 학습 금지, 평가와 연구 ablation만.** AMASS·HumanML3D(SMPL 계보), DataDoP(MonST3R), SpatialVID 포함. Lineage CI가 강제한다 | 라벨 계보 순도 |
| CT24 | MoPri-S 크기 | 32M [P2] vs 10M (데이터 기아) [TF] | **데이터가 정한다**: 16M으로 시작하고 자체 모캡 + Act-from-Text가 50시간을 넘으면 32M | CMU + 100STYLE ≈ 28–30시간은 32M에 Chinchilla 0.1× 수준이다 |
| CT25 | F_φ 크기 | 4M [TF, P2(o)] vs 0.6M [P2(u)] | **0.6M, SMC 루프 밖 top-k에만 적용** | 4M GNS는 1만 노드에서 스텝당 26–52 ms가 든다 |
| CT26 | Day-90 벤치 규모 | 40 세계 [CB] vs 160 [RR] vs 240×12 [v1] | **미니 벤치 160 세계(C1+C2), 라이브러리 = 벤치.** 전체 벤치는 Day 90 이후 | 우리 seed는 세계 재샘플이어야 공정하다. 프롬프트를 4주차에 사전 등록한다 |
| CT27 | Phase 2 일정 기준 | Phase 1 이후 +3–6개월 [v1] vs M0 기준 [P2(u)] | **M0(프로젝트 시작) 기준. 시드 트랙은 Day 0부터** | P2 비용은 시뮬, 렌더, TC에 몰려 있어 T4 손해가 상대적으로 작다. Chaos Bank는 일찍 쌓아야 한다 |
| CT28 | Phase 2 비디오 모델 | 저주파 증인 [v1 C11] | **금지.** 크로노 증인(GICC) + 실제 증인 + 감독(사람)으로 대체 | 하드 제약. v1 H7을 본선으로 승격한다 |
| CT29 | 운용 모드 | 상시 vs Kaggle | **상시 2×T4를 권고하고 Kaggle은 보조** | 같은 카드인데 가동률이 5.6배(쿼터가 2배 차감되면 11배) 차이 난다 |
| CT30 | 폰 내장 ML (ARKit, 뉴럴 ISP) | VCam 입력 [RR(o)] vs 금지 [P2, TF] | **데이터 경로와 시스템 경로에서 금지.** 기본은 ArUco PnP + OpenVINS. ARKit을 라이브 프리뷰 입력 장치로 쓸지는 Q8 | ISP 환각이 미세 대역 통계를 오염시킨다 |
| **CT31** | **미감 교사** | v2.0: 빌린 이미지 모델 하나, 닫힌 모델은 평가 전용 vs FD: Soul 하이브리드 | **하이브리드 채택.** 엔진 교사(오픈, 동결; 노이즈·프로브·판정·래칫) + 미감 교사(Soul, 닫힌; 아트 디렉터·최종 광택·개념 아트·평가 앵커). Soul은 절대 버리지 않는 핵심 자산이다(창업자) | A/B 5판 50축에서 Soul 41승·Z 1승·8무. v1 §1.6 (iii)이 열어 둔 "σ 없는 닫힌 모델(ρ≈0)의 POLISH 소유자" 슬롯을 T6 근거로 부활시킨다 |
| **CT32** | **Soul 방화벽** | — | **Soul 픽셀은 세계 W(콘텐츠)와 최종 렌더로만 흐른다.** 데이터셋, 손실, 보상, 보정, 체크포인트, Texel Bank, 래칫, Exhaust 학습 분할, Feature-Tap Bus 링버퍼, 양자화 보정셋, 교사 VAE 쌍 수집기로는 흐르지 않는다. `SOUL`/`SOUL-DERIVED`/`SOUL-INSPIRED` 태그 + STRATA `own`·`prov` 필드 + Merkle 태그 + Lineage CI + 격리 저장소(soul_vault) | 약관: "You may not use the Service or outputs to train, fine-tune, or improve any AI or machine-learning model". 예외는 서면 승인뿐이다 |
| **CT33** | **세계 피팅의 회색지대** | 장면 표현 최적화 = 모델 학습인가? | **회색지대로 명시한다.** 기본은 Plan A(사람 경유: 사람이 보고 측정하고 전사한다). Soul 픽셀을 관측으로 세계를 자동 피팅하는 것, 룩 θ 자동 역추정, LUT 자동 피팅은 A+ 서면 승인 뒤에만 한다 | 세계 W는 "AI 모델"이 아니라고 주장할 수 있지만, 미분 피팅은 학습과 수학적으로 같다. 해석 리스크를 사람 경유로 피하고 승인을 요청한다 |
| **CT34** | **Soul 생성 경로와 예산** | 웹 무료 vs MCP | **웹 무료·Unlimited(약 2,900회)는 창업자가 웹에서 직접 뽑는 사람 경로 전용.** MCP 자동 호출은 크레딧 예산(약 1,600회분, 추정)으로 따로 잡는다. **웹 앱 자동화로 무료 생성을 우회하지 않는다** | 공식 도움말: MCP 생성은 항상 크레딧을 쓰고, 무료·Unlimited는 웹에서만 적용된다 |
| **CT35** | **엔진 교사 선정 기준** | 미감(Elo) [v2.0] vs 통제성·속도·라이선스 [FD] | **엔진 점수 ES / 수용 세계당 T4-h**를 최대화한다. ES = 차트 뷰 재질 2AFC(물리 조명 아래) + MRH 실사 AbsRel + DR 판별 AUC + SDEdit 저대역 보존. 미감은 E1의 하이브리드 트랙에서만 잰다. ES가 3%p 이내면 빠른 쪽을 고른다 | 미감은 Soul과 물리 카메라 몫이 됐다. Z-Image의 "지루한" 디폴트(평광, 팬포커스, 정면, 가림 없음)는 측량 기사로는 이상적이다 |
| **CT36** | **미감 소유권** | v2.0 §5.5 "미감은 오직 빌린 교사의 몫" vs 연방 | **미감 연방제.** 축 1–8(화면비, 렌즈 원근, 심도, 시간대·광원, 노출, 그레이딩, 필름 질감, 프레이밍)은 물리 스택(θ_look, Gaffer, 렌즈·센서·필름, DP-0), 축 9–10(미술 생활감, 인물·서사·개념)은 Soul(개념 아트, Soul ID) + 사람, 재질·기하·식물학적 디테일은 엔진 교사 | Soul 승리 41개 중 32개(78%)가 물리 소관. 단, 영향력 기준으로는 해석형 프롬프트의 격차 대부분이 개념 축이다(CT40) |
| **CT37** | **비디오 프레임별 Soul 광택** | 프레임마다 광택 vs 금지 | **금지.** 허용은 둘뿐이다. (i) 스틸(키 아트, 포스터, 썸네일)의 광택 = 콘텐츠(A). (ii) 히어로 키프레임 광택을 세계로 전파하는 POLISH 소유(A+, Shoot 전에 CoW로 다시 연 1개 Jacobi 라운드에서 엔진 교사 뷰와 동시에, 소유 마스크 안, Soul 2K Nyquist 이하 대역만, 엔진 교사 재판정 후 commit) | 프레임별 재그리기는 깜박임과 GeCo 붕괴를 부르고, "카메라는 읽기만 한다"를 깬다. 비회귀 조항 2를 지키는 길은 세계 경유뿐이다 |
| **CT38** | **벤치 트랙** | — | **Engine(Soul 0) / Hybrid-A(사람이 측정한 룩 프리셋 라이브러리, 4주차 동결) / Hybrid-A+(프롬프트별 MCP 개념 아트) 분리 보고.** 공개 Kaggle 벤치는 Engine과 Hybrid-A만(Hybrid-A 공개는 Q13 회신 후) | 닫힌 모델과 크레딧은 공개 재현이 안 된다. 프롬프트별 사람 개입은 T1·T2 규칙 위반이므로 프리셋은 사전 동결한다 |
| **CT39** | **카탈로그 이름** | 렌즈별 상이(MRH/Free-rider/GeoHead, LBO-S/BO-40/ICX-LBO/TexelFM, TAE-T/TL, G-Adapter/GB-Adapter, Hand/Path/CRM 등) | **정본 이름 + 한국어 별명 + 별칭표**(§5.2). 문서, 코드, 체크포인트, Lineage 노드가 정본 ID를 쓴다 | 계보 CI는 이름으로 조상을 추적한다. 이름이 둘이면 누수 검사가 뚫린다 |
| **CT40** | **미감 격차를 재는 법** | 동일 가중 축 세기 [FD 1–4판] vs 영향력 | **"카메라만 바꿨을 때 격차가 얼마나 닫히나"를 블라인드로 직접 잰다**(X-LA, 미감 부검 3분해: 카메라·조명 몫 / 연출 몫 / 개념 몫) | 야크 판: 축 개수로는 물리 78%가 유지되지만 영향력의 대부분은 개념 해석이었다. 축 세기에는 맹점이 있다 |
| **CT41** | **TP의 기준 스틸** | v1 TP = 교사 스틸 대비 | **v1 TP(엔진 교사 스틸 대비 ≥45%)는 그대로 둔다.** TP_S(우리 렌더 대 Soul 스틸)를 진단 지표로 추가하고, 진단 기준선 TP_S\* = 1 − SP_S(E1-S)를 함께 보고한다(게이트 아님, Q6) | 목표는 낮추지 않고 측정만 늘린다 |

### 0.6 v1 판결 C1–C15의 운명

| v1 판결 | v2.1 |
|---|---|
| C1 solver(WSSC) + student(WDM) | **변경.** solver는 WSSC-T4, student는 TexelFM + MassingFM + MRH, 관할 이양 래칫으로 연결한다. WDM은 연기 |
| C2 미세 레벨은 band-iso + 스칼라 t | **유지.** Grid-Locked 차트 뷰가 band-iso의 극단형이다 |
| C3 annealed → hard 소유권 | **변형.** 미세 레벨은 처음부터 단독소유(OTBI)로 하고 apron에서만 multiband. 의미 패스는 annealed 유지 |
| C4 DTC 1순위 | **유지 + 확장.** p̂_T 추정은 MRH의 defocus·EV 헤드가 맡는다. v2.1: Soul 스틸의 구워진 심도·그레인·그레이딩·비네팅을 θ_soul로 설명해 세계 밖으로 빼는 데도 같은 원리를 쓴다(§3.8.4) |
| C5 Dual-Anchored Appearance | **유지 + 보강.** L0 조명 대역은 물리 bake가 소유하고 L_d가 정본이다 |
| C6 UKA + 예외 3 | **변경.** TSQ + 예외 3(VOL/SGGX, DIEL SDF, 아바타는 P2c) |
| C7 dual-cut | **유지** (16×16 타일 단위) |
| C8 Location/Directed | **유지.** Directed가 기본이고 벤치는 Location-lite |
| C9 2a = Frozen Action | **유지** |
| C10 불변량 + 미시 잔차 | **유지** |
| C11 비디오 저주파 증인 | **폐기 (CT28)**. 닫힌 비디오 API는 평가 전용 |
| C12 빈 사분면 | **유지** |
| C13 H-ν 가설 | **유지** (X0는 고전 추적으로 측정) |
| C14, C15 신규성 정직성 | **유지.** Grid-Locked는 [조합]으로 하향했다(MultiDiffusion, SyncMVD 선례) |
| v1 §1.6 (iii) "σ 없는 닫힌 API = ρ≈0, T6 덕에 POLISH 소유자 후보, 약관 확인" | **v2.0 금지 → v2.1 부활(Soul, A+ 조건부).** v1 Q3("닫힌 SOTA의 사용 범위")에 대한 창업자 답이다: art director·개념 아트·평가 앵커는 Plan A, POLISH 소유는 A+, GAN real 집합 등 학습 용도는 S 전까지 금지 |

---

## 1. 하드웨어 현실과 컴퓨트 예산

### 1.1 T4 사실표

| 항목 | 값 | 설계상 의미 |
|---|---|---|
| 칩 | TU104, sm_75. CUDA 2,560, Tensor 320, **RT 40**, 70W 패시브 [검증] | OptiX 하드웨어 삼각형 교차를 쓸 수 있다 |
| FP16 텐서 | 피크 65 TF, **실측 HGEMM 41.6 TF**(1590 MHz 고정, 피크의 64%) [검증] | 전력 쓰로틀이 일찍 걸린다. 계획은 15 TF/장(큰 GEMM), 12 TF/장(소형), 3D conv 6–8 |
| INT8 / INT4 | 피크 130/260, **실측 74.9 / 114.4 TOPS** [검증] | INT4는 FP16의 2.75배다. Hopper에서 빠진 INT4가 여기엔 있는 숨은 무기 |
| FP32 | 8.1 피크, SGEMM 7.17 [검증] | FP32 경로를 실수로 타면 6–8배 손해 |
| 정밀도 | BF16·TF32·FP8 없음 [검증] | FP16 + loss scaling. bf16으로 학습된 교사는 FP16 수술(Rescue) 필요 |
| 메모리 | 16 GB(ECC를 켜면 사용 가능 약 15 GB), 300–320 GB/s, L2 4 MiB | roofline 능선: FP16 약 203 FLOP/B(피크), 130(실측), INT8 약 406 |
| 인터커넥트 | PCIe Gen3 x16, 실효 약 12 GB/s, NVLink 없음 | tensor parallel은 비효율적이다. 역할로 나눈다(두 반구) |
| 툴체인 하한선 | PyTorch CUDA 13, Warp CUDA 13, OptiX 모두 하한이 Turing [검증] | **Docker로 버전을 고정한다.** 다음 CUDA 메이저에서 잘릴 위험이 있다 |

### 1.2 소프트웨어 규약 (sm_75)

| 영역 | 쓴다 | 금지 / 주의 |
|---|---|---|
| 프레임워크 | PyTorch CUDA 13.x(고정), cuBLAS/CUTLASS GEMM, Inductor는 원소별 퓨전만(GEMM 백엔드 `ATEN` 고정) | Triton ≥3.3 `tl.dot`은 cc<8.0에서 FMA로 폴백한다 [검증]. max-autotune, FlexAttention, 커스텀 Triton GEMM 금지 |
| 어텐션 | SDPA mem-efficient, **flash-attention-turing**(fwd 최대 2.19×, bwd 1.35–1.51×, hd 64/96/128, dropout 없음) [검증] | FA2/3/4 없음. FA1 bwd는 hd ≤64. **우리 모델은 head_dim 64**, 교사는 hd 128(fwd 전용) |
| 추론 | TensorRT 10.x FP16/INT8, Nunchaku v1.2 W4A4. 20-series INT4 지원 [검증], Z-Image-Turbo 4-bit 가중치 공개 [검증]. **Z-Image의 Turing INT4 커널 경로와 T4 속도는 미확인(E0)** | Nunchaku의 T4 실측은 없다(E0). 사전 양자화 가중치 대신 우리 보정셋으로 자체 양자화한다(계보 단순화) |
| 클럭 | `nvidia-smi -lgc`로 지속 가능한 클럭 고정 [운용] | 쓰로틀 진동 제거. 값은 E0의 TF/W 곡선으로 정한다 |
| 3D·시뮬 | gsplat(JIT, 3DGUT), OptiX(Turing 이상, R570+), Warp(CUDA 13 빌드는 sm_75 이상), MuJoCo Warp, CPU MuJoCo | MJX(JAX)는 sm_75 확인 전에는 쓰지 않는다. tiny-cuda-nn fully-fused는 미검증 |
| SfM | COLMAP 4.2.1 `feature_type=SIFT` + GLOMAP(`global_mapper`) + Caspar GPU BA + PatchMatch MVS, 비디오는 sequential 매칭 | ALIKED/LightGlue(ONNX), `vocab_tree_flickr100K`, ORB-SLAM3 `ORBvoc`(외부 이미지로 학습한 코드북) 금지. 필요하면 우리 이미지로 vocab tree를 k-means한다. CI가 빌드 플래그를 검사한다 |
| 렌더 레퍼런스 | Blender Cycles(OptiX), **디노이저 OFF**, Infinigen은 지오메트리만 | OIDN, OptiX AI 디노이저 금지(사전학습 CNN) |
| Kaggle | 교사 가중치는 Kaggle Models 읽기 전용 마운트(작업 디스크 약 20 GB 밖), 잡 ≤8 h 단위(세션 12 h 안에서 체크포인트 여유), Jacobi 라운드 Merkle 체크포인트 | Kaggle 약관의 상업 R&D 허용 여부를 D1에 확인한다(Q4) |

### 1.3 예산표 (두 장 합계)

| 항목 | 1일 | 30일 | 90일 | 근거 |
|---|---|---|---|---|
| T4-시간 | 48 | 1,440 | **4,320** | — |
| 학습 비관 (12 TF/장) | 2.1×10¹⁸ | 6.2×10¹⁹ | 1.9×10²⁰ | 작은 GEMM, 쓰로틀 |
| **학습 기본 (15 TF/장)** | **2.6×10¹⁸** | **7.8×10¹⁹** | **2.3×10²⁰** | 실측 HGEMM의 36% |
| 학습 낙관 (20 TF/장) | 3.5×10¹⁸ | 1.0×10²⁰ | 3.1×10²⁰ | 큰 GEMM, s2d, 퓨전 |
| 추론 INT8 (24 TOPS-eq/장) | 4.1×10¹⁸ | 1.2×10²⁰ | 3.7×10²⁰ | attention은 FP16 |
| 교사 용량 (전부 교사에 쓸 때) | 6.6만 TFU (INT8) / 9.6만 (INT4) / 3.8만 (FP16) | — | — | §1.4 |
| 전력 (GPU만) | 3.4 kWh | 101 kWh | 302 kWh | 70W×2. 호스트를 포함하면 2–3배 |

**스케일 감각.**

- 처음부터 비디오 모델을 학습하는 길은 닫혀 있다. 2.3×10²⁰ FLOPs의 Chinchilla 최적점은 1.4B 모델 + 2.8×10¹⁰ 토큰이다. 1080p 10초 클립(약 49만 토큰)으로 환산하면 약 5.7만 클립, 즉 영상 158시간을 한 번 보는 수준이다. 경쟁자는 10²⁴–10²⁵를 쓴다. 자릿수로 4개 차이다.
- 빌린 엔진 교사의 사전학습 연산은 3.4–4.5×10²³이다(314K H800-h [2차], MFU 30–40% 가정). 우리 90일 예산의 **약 1,500배(1,100–1,900배)**다. Soul의 사전학습 연산은 공개되지 않았다. 우리는 그 결과물을 콘텐츠로만 빌린다.
- 수십M 모델 학습은 싸다. 50M 모델을 20 tok/param으로 학습하면 6ND = 3×10¹⁷ FLOPs이고, 2×T4(15–20 TF/장)로 **약 2–3시간**이다. 리서치 원문의 "약 1시간"은 HGEMM 피크에 가까운 가정이다.
- 카탈로그 전체(Phase 1 213.5M) 1세대 학습은 약 9.2–13.1×10¹⁸ FLOPs로 **90일 예산의 약 4–6%**다(§5.6, 상단은 G-Adapter 3 epoch). 학습 T4-h 몫이 FLOPs 몫보다 큰 이유는 소형 모델의 MFU가 낮아서다.
- **진짜 병목은 교사 forward, 데이터 생산, 엔지니어링, 사람 평가, 그리고 v2.1부터는 사람의 아트 디렉션 시간이다.** 학습 FLOPs가 아니다.

### 1.4 TFU 가격표 [신규 운용] (E0에서 확정)

1 TFU = Z-Image(6.15B) 1024² forward 1회 = 6.2×10¹³ FLOPs. 구성은 2PN 5.35×10¹³(이미지 4,096 + 텍스트 약 256 토큰)과 attention 8.7×10¹²이다. 512²는 0.25 TFU, 768²는 0.56 TFU, 256²는 0.1 TFU다. TF 렌즈의 NFE-eq와 같은 단위다.

| 정밀도 (같은 가중치) | 1024² s/fwd | 512² s/fwd | TFU/T4-h | 쓰는 곳 |
|---|---|---|---|---|
| FP16 수술판 (PyTorch + FA-turing) | 4.5 (3.1–4.5) | 0.9–1.2 | 800 | 최종 재판정, commit, 폴백 |
| INT8 (TRT, SmoothQuant α 0.5–0.8) | **2.6** (2.2–3.0) | 0.6–0.7 | **1,385** | 결정 창, 판정. **기본 계획값** |
| W4A4 (Nunchaku SVDQuant, **Z-Image Turing 경로 미확인 가정**) | 1.8 (1.4–2.2) | 0.45 | 2,000 | 큰 σ, 탐색, adversarial, hero 1차 |

- FP16 4.5 s는 리서치의 순수 GEMM 추정 3.1–4.1 s(61–62 TF ÷ 15–20 TF)에 Rescue FP32 섬과 SDPA FP32 누산 오버헤드 약 10%를 얹은 보수치다. 1024² 1장(8 NFE)은 36 s이고, 리서치 추정은 26–35 s다.

| 비교 | 값 |
|---|---|
| 교사 VAE decode 1024² | FP16(자체 수리) 0.3–1.0 s, FP32 타일 1.5–2 s |
| TAE-T decode 1024² | 25–40 ms |
| Exhaust 기록 (int8 x̂0 256 KB + 20% PCA 탭) | 호출당 약 0.3 MB |
| 엔진 교사가 바뀌면 | TFU를 새 교사의 1024² fwd로 재정의하고 가격표·Exhaust를 버전 태그로 재보정한다. FLUX.2 klein 4B면 1 fwd ≈ 0.65 TFU(Z-Image 기준, 추정) |

### 1.5 상시 2×T4 vs Kaggle 쿼터 모드

| 항목 | 상시 (기본) | Kaggle 1배 소모 | Kaggle 2배 소모 |
|---|---|---|---|
| 90일 T4-h | 4,320 (가동률 75–85%면 3,200–3,700) | 771 | 386 |
| 세션 | 무제한 | 12 h [검증], 주 약 30 h [2차] | 동일 |
| vCPU / RAM / 디스크 | 16–32 / 128 GB / NVMe 4 TB 권장 | 2–4 / 약 30 GB / 20 GB [2차, 미확인] | 동일 |
| 운영 | 주야 시분할 | 세션 = 세대, 20분 체크포인트, Jacobi 라운드 Merkle 재개 | + 범위를 C1 하나로 고정 |
| Day-90 범위 | 전체 계획 | 데모 3편 + 범주 1개(일정 동일) | **목표는 그대로, 일정만 1.5–2배(D135–180)** |

- **창업자 권고:** 같은 2×T4를 상시로 돌린다. 클라우드 T4 기준 90일에 $1.5–2.2K이고(Colab Enterprise T4 $0.42/h [2차]), 자체 서버도 된다. "T4 두 장"이라는 제약은 그대로 지키고 가동률만 올리는 것이다.
- **CPU 농장 [신규 운용]:** 절차 생성, Infinigen, COLMAP, ID-Screen, Chaos Bank는 GPU 쿼터 밖의 CPU에서 돌린다. Kaggle에서는 외부 저가 16 vCPU 머신으로 돌리고, 레이아웃·G-buffer 생성은 Warp 커널로 GPU에서 돌린다(GPU ProcGen [조합]).
- Soul은 T4도 Kaggle도 쓰지 않는다. 운용 모드와 무관하게 같은 크레딧 예산(§3.7.5)을 쓴다.

### 1.6 메모리 맵 (장당 사용 가능 약 15 GB)

| 모드 | GPU0 | GPU1 | 호스트 RAM (128 GB 권장, 최소 64) | 디스크 (NVMe 4 TB) |
|---|---|---|---|---|
| **솔브 Mode A (INT8 복제, 기본)** | 교사 INT8 6.2 (W4A4면 3.5) + 활성(1024² B2, FA-turing) 2.0 + TRT 작업공간 1.0 + VAE FP16(mid-block FP32) 1.0 + MRH 0.1 + G-Adapter 0.05 + TAE-T·LatentDown/Up 0.02 + Exhaust 링버퍼 0.5 + ctx 0.5 ≈ **11.4 GB** (W4A4 8.7) | 교사 INT8 복제본 6.2(lockstep 구간에만) + TSQ·페이지 0.5–1.0 + Z_tex latent 타일 2.0 + TABP 누산기 0.5 + TexelFM(TRT) 1.0 + 래스터 버퍼 1.2 + BVH 0.3 + ctx 0.5 ≈ **12.2–12.7 GB**. 피팅 구간에는 복제본을 내리고(pinned 재적재 0.5 s) gsplat 버퍼 1.5 + Adam 0.6을 올린다 | 교사 가중치 pinned(FP16 12.3 + INT8 6.2 + TE 8) 약 27 GB, Ledger hot(int8) 10–13 GB, 세계 피라미드 2–6 GB, EXR 큐 4 GB | — |
| **솔브 Mode B (FP16 파이프라인 분할)** | 층 0–14 6.2 + 활성 1.0 + VAE 1.0 + MRH·TAE-T 0.2 + ctx 0.5 ≈ 9 GB | 층 15–29 6.2 + 세계 3–4 + 래스터 1.2 + TexelFM 1.0 + ctx 0.5 ≈ 12–13 GB | 경계 전송 19 MB/스텝(1.6–3 ms) | — |
| **솔브 Mode C (σ-카드 분할)** | 교사 FP16 수술판 12.3 + 활성 1.0 + 헤드 0.3 + ctx 0.5 ≈ 14.1 GB (commit σ 점, hero 최종, 탭) | 교사 W4A4 3.5 + 세계·렌더 6–8 (탐색 스텝, adversarial, Gaffer) | — | — |
| **학습** | G-Adapter/LoRA: NF4·INT8 동결 교사 3.5–6.2 + ckpt 활성(512² B1–4) 6 + 어댑터·Adam 0.5 ≈ 10.5–13 GB. 또는 교사 공장: INT8 6.2 + 활성 2 + 탭 0.5 ≈ 8.7 GB | TexelFM-35 약 9 GB, 또는 MPS로 소형 잡 2–4개(각 1–4 GB) | WebDataset 샤드 스트리밍 | 데이터 450 GB, PCA-128 int8 탭 캐시 300 GB |
| **렌더 8K** | 짝수 프레임: 세계 BC 1.2–2.8 + 프레임·AOV 1.5 + 정렬 1–2 + Defocus Pyramid 0.4 + FFT 1.8 + BVH 0.6–1.2 ≈ **7–10 GB** | 홀수 프레임, 동일 | EXR/ProRes 인코딩은 CPU 바운드(16코어 이상) | 8K EXR은 최종 테이크만(beauty 10초 120fps에 50–100 GB) |
| **P2 플라이휠** | 교사 전용: INT8 판정 6.2 + VAE FP32 0.34 + 활성(512² B4) 2–3 ≈ 9–10 GB. BSRT/DMD 구간만 FP16 12.3으로 스왑(1.3 s) | 학생 학습 6–8 + MuJoCo Warp 1–2 + INT8 proxy 0.3 + 래스터 1–2 ≈ 10–13 GB | CPU MuJoCo 16 프로세스, ID-Screen, 뱅크 구축 | Chaos Bank 64 MB/10⁶ rollout |
| **Soul 판독 (A+ 전용)** | 엔진 교사 INT8 + MRH, Soul 스틸 2K → 1024² 다운스케일 1장 판독 약 3–5 s(1–2 TFU). **Feature-Tap Bus를 끈 별도 프로세스**에서 돌고, 탭·latent·VAE 쌍은 링버퍼에 들어가지 않는다 | — | soul_vault는 학습 잡 계정이 읽을 수 없는 별도 마운트 | soul_vault 수 GB |
| **Kaggle 모드** (RAM 약 30 GB) | 교사 INT8 6.2 + 활성 2.0 + VAE 1.0 + 헤드 0.2 + ctx 0.5 ≈ 10 GB. FP16 수술판은 commit 직전에 Kaggle Models 마운트에서 적재(재적재 시간 실측 필요) | 세계·렌더 6–8 GB 또는 소형 모델 학습 | **pinned는 INT8 6.2 GB만.** TE는 CPU에서 인코딩한 뒤 바로 해제. Ledger hot은 2–3 GB로 줄이고 나머지는 비동기 업로드. **Mode C는 불가**(FP16 수술판 상주 불가) | 20 GB (§6.7) |

PCIe 트래픽은 병목이 아니다. 뷰당 latent 128²×16 fp16 = 512 KB이고, 100뷰 라운드에 51 MB, 약 4 ms다. 25M 모델 gradient allreduce는 50 MB, 약 10 ms다. 원칙은 **"두 장에 잡 두 개"**다. 소형 모델은 DDP로 쪼개지 않는다.

### 1.7 Day-1 T4 진실표 (E0, 2시간이면 이 문서의 (추정)이 숫자로 바뀐다)

| 순위 | 측정 항목 | 합격선 [RR X-T0 복원] | 떨어지면 |
|---|---|---|---|
| **1** | **교사·VAE의 FP16 NaN·검은 이미지 비율**(Rescue 후, 1만 NFE, 교사·정밀도별) | 0/10,000 + commit σ에서 FP16 수술판 별도 2,000 | 블록별 FP32 폴백(+30%) 또는 INT8 판정, Mode B |
| 2 | HGEMM 512–8192 크기별 지속 TF, 클럭, 전력, throttle reason, TF/W 곡선(클럭 고정값 결정) | 지속 HGEMM ≥25 TF | 모든 시간 재계획(아래 규칙) |
| 3 | 엔진 교사 fwd: FP16 수술판, TRT-FP16, INT8, W4A4 × 512/768/1024. 재오디션 후보(klein 4B FP16·INT8·W4A4, ERNIE-Image-Turbo INT8)와 Z-Image Base fwd 같은 표 | FP16 1024² ≤4.5 s, INT8 ≤3.0 s, **INT4 ≥1.6×FP16** | INT4 실패는 허용(Mode B/C) |
| 4 | SDPA vs flash-attention-turing | 측정 | — |
| 5 | VAE FP16/FP32 | 측정 | FP32 타일 |
| 6 | gsplat/2DGS 4K fwd, 1024² fwd+bwd | **4K ≤0.6 s @3M 프리미티브** | 타일 컬링, half-res + 업샘플, AOV 축소 |
| 7 | OptiX closest-hit vs Warp BVH, 컨테이너의 `libnvoptix` 존재 여부 | **closest-hit ≥3×10⁸ ray/s** | Warp BVH, 평면 재귀 + SDF만 |
| 8 | NVENC 8K HEVC 10-bit | 동작 | CPU 인코딩 |
| 9 | CUDA BC7/BC5/BC6H 텍스처 하드웨어 디코드 | 동작, 바이리니어 필터 포함 | 셰이더 디코드(+비용 실측) |
| 10 | COLMAP 4.2 SIFT + GLOMAP + Caspar 장면당 처리량 | 측정(CPU 농장 사이징) | vCPU 증설 |
| 11 | MuJoCo Warp sm_75 동작·처리량 | 동작 | CPU MuJoCo |
| 12 | pinned 27 GB 재적재 시간, PCIe(카드 간, 호스트↔카드) | 재적재 ≤1 s/6 GB | 복제본 상주 |
| 13 | Kaggle 쿼터 소모율 | 측정 | §1.5 2배 열 |

**규칙:** 실효가 30% 낮게 나오면 모든 시간을 1.4배로 다시 계획한다. 결론은 바뀌지 않는다.

### 1.8 숫자 교정 장부 [v2.1 복원] (렌즈 프리모템이 고친 숫자)

v2.0은 일부 렌즈의 앞부분을 받지 못해 아래 교정 중 몇 개가 근거 없이 결론만 남았다. 근거와 함께 복원한다. 교정 ID는 `NC`로 쓴다(§5.2·§5.4의 P2 카탈로그 ID N1–N10과 구분).

| # | 항목 | 이전 값 (출처) | 교정값 | 근거 |
|---|---|---|---|---|
| NC1 | SDXL 1 NFE @1024² | 1.4 s [TF(o)] | **0.55–0.8 s** | Fooocus Colab T4 역산(2장 74 s, 30 step, CFG) [TF(u), RR(u)] |
| NC2 | Z-Image-Turbo 1 NFE @1024² | 비용 미반영 [TF(o)], step당 4–6 s [RR(o)] | **FP16 3.1–4.5 s, INT8 2.2–3.0, W4A4 1.4–2.2** (추정) | 61–62 TF ÷ 15–20 TF = 3.1–4.1 s(+FP32 섬·누산 오버헤드 10% → 4.5). INT8은 24 TOPS-eq 기준 2.2–3.0 s |
| NC3 | 1024² band-iso 뷰 폭 @l=14 | 6.4 cm [RR(o)] | **0.5 m** | 셀 128개 × h₁₄ 3.9 mm (latent 토큰 1개 = 셀 1개) |
| NC4 | 4K 방 미세 레벨 뷰 수 | 25–50 | **deep focus 70–270, Directed 25–80** | 수요 면적 10–40 m² ÷ (0.25 m² × apron 효율 0.77) × 가림 1.3 |
| NC5 | 미세 레벨 Adam 상태 | 장당 1.9 GB | **10–19 GB(장당 5–10)** → OOM | 텍셀 1–2×10⁸ × 6ch × 16 B. TABP로 제거(§4.1 P2) |
| NC6 | TexelFM 이득 | 교사보다 8배 저렴 | full-res conv면 **약 1배**. patchify(s2d×4)여야 8–16배 | 텍셀당 4–7 MFLOPs vs 0.5–0.9 MFLOPs |
| NC7 | 8K EXR | 40–80 MB/프레임(AOV 포함) | beauty 40–80 MB, **AOV 포함 0.5–0.7 GB** | 38 B/px × 33 Mpx = 무압축 1.26 GB → ZIP 무손실 후 0.5–0.7 GB. cryptomatte·깊이는 무손실 |
| NC8 | 미니 벤치 우리 세계 | 80 | **160(290–400 T4-h, 세계당 1.8–2.5)** | 우리 seed는 세계 재샘플이어야 공정 |
| NC9 | GeCo 평가 | 21 T4-h | **11–43 T4-h** | VGGT를 fp32로 강제하면 클립당 30–60 s |
| NC10 | Kaggle 90일 | 780 T4-h | **386–771**(리서치 "약 780") | T4×2의 쿼터 차감률 미확인 |
| NC11 | IMU 300 h | "10M 모델의 Chinchilla 점" | 실토큰 5,400만 = **2.7M** 모델의 Chinchilla 점. 8M은 증강 ×5.5 필수 | [TF(u)] |
| NC12 | CC0 재질 수 | MatSynth 4천 + ambientCG 2천 + Poly Haven 700 | **고유 약 4–5천** | MatSynth가 두 소스를 포함한다 |
| NC13 | DP-0 탐색 | 후보 10³ × 720p 약 8분 | 그대로면 **11–32분**. 해석 통계로 1차 채점 + 상위만 프리뷰하면 1–4분 | 19.2만 프레임 ÷ 100–300 fps |
| NC14 | G-Adapter 학습 | 13 T4-h [RR(o)] | **36–110 T4-h** | 샘플-패스 4–12만(4만 샘플이면 1–3 epoch, RR(u)의 더 작은 데이터면 3–5 epoch) × 48 TF = 1.9–5.8×10¹⁸. 동결 교사 역전파(fwd의 3–4배) |
| NC15 | 4K secondary RT | 0.04–0.15 s (자기모순) | **15–60 ms** | TSQ 불투명 코어 삼각형 closest-hit 3–10×10⁸ ray/s(추정) |
| NC16 | Ledger 엔트리 | 128 KB | Z-Image 16ch 512 KB → **int8 256 KB**. hot 5만 = 12.8 GB | [RR(u)] |
| NC17 | 1 H100-h | 25–30 T4-h | **20–31 T4-h** | 1.44×10¹⁸ ÷ (13–20 TF × 3600) |
| NC18 | v1 대비 비율 | "10⁴배 작다" / 1/3,000–1/6,000 | v1 90일 계획 대비 **1/400–1/600**, Phase 1 누계 대비 1/3,000–1/6,000. 카탈로그 FLOPs는 v1 CWM/LBO의 약 1/6,700 | 기준을 병기한다 |
| NC19 | PWMT 비용 | "15–30 s, GPU0과 병렬" | 세계당 약 60회 → **GPU1에서 15–30분** | 슈퍼블록 × 재질 × 레벨 × 라운드 |
| NC20 | WDM 메모리 | 교사 두 벌 24.6 GB | fake score = 같은 base + LoRA on/off → **가중치 1벌** | [TF(u)] |
| NC21 | 에너지 4K120 10초 | 12 Wh | **GPU만 8–18 Wh, 호스트 포함 16–37 Wh** | 정직하게 병기 |

### 1.9 실리콘 전 유닛 배정 [v2.1 복원: P2(o) P5]

| 유닛 | 담당 |
|---|---|
| FP16 텐서 | 엔진 교사 FP16 commit·재판정, 학생 학습 |
| INT8 / INT4 텐서 | 교사 판정·탐색(INT8, W4A4), critic 캐스케이드(TensorRT INT8), 학생 배포 |
| CUDA FP32 | GPU 시뮬(MPM, FLIP, Warp), 렌즈·센서 FFT, 고정소수점 리덕션 |
| RT 코어 40개 | 거울·크롬·유리 secondary, L0 조명 bake, 음선 추적 |
| NVENC / NVDEC | 리뷰 인코딩(8K HEVC 10-bit, 실측 필요), 라이브 프리뷰 |
| CPU 16–32코어 | MuJoCo(결정론), COLMAP/GLOMAP, 1 kHz 리그, 절차 생성, graph cut, EXR 인코딩, 모달 FEM |
| (T4 밖) Soul 클라우드 | 개념 아트, 룩 레퍼런스, 스틸 광택. 크레딧과 사람 시간으로 계상 |

규칙: **GPU에서 돌릴 이유가 없는 일은 CPU에서 돌린다.** CPU MuJoCo는 결정론적이라 Merkle 재현성에도 유리하다.

### 1.10 T4 밖의 자원 장부 [v2.1]

| 자원 | 90일 예산 | 단위 | 비고 |
|---|---|---|---|
| Soul 웹 생성(사람 수동) | 약 2,900회 | 생성 1회 = 스틸 1장(배치 최대 4)으로 계상 | 창업자 구독의 무료·Unlimited. **사람만**. 자동화 금지. **'회'가 이미지 장수인지 배치 생성인지 미확인이다.** 배치 단위면 스틸 용량이 최대 4배가 된다. 배분표(§3.7.5)는 이미지 장수 기준으로 보수 계상하고, 실제 계량 단위는 D1–3에 MCP 단가와 함께 실측해 고친다 |
| Soul MCP 생성(자동) | 약 1,600회분(추정) | 크레딧. Soul V2 MCP 1회 약 10 크레딧, $1 ≈ 16 크레딧(3rd-party 추정, 미검증). '1회'의 단위(스틸/배치)도 같이 실측 | 약 1.6만 크레딧 ≈ $1,000 내외. **창업자 보유 크레딧을 먼저 쓰고 부족분만 구매한다(상한 $1K).** 잔고는 D1–3에 확인한다 |
| 아트 디렉터 시간 | 주 10–20시간 | 사람-시간 | 개념 아트 선별, 룩 수동 측정, WorldScript 전사, IP 점검 |
| 경쟁 비디오 API | $3–12K | 달러 | 평가 전용 |
| 사람 평가 | $13.5–32K | 달러 | E1-S, 미감 부검, TP_S 앵커 포함 |

---

## 2. 4분류 표: 빌린 것 / 만든 것 / 계산하는 것 / 테스트타임 최적화

### 2.1 차입 장부

**빌린 것 = 오픈 엔진 교사 + Soul(닫힌, 콘텐츠 전용).** 학습 데이터로 흐를 수 있는 빌린 지식은 엔진 교사뿐이다.

$$
C_{\text{eff}}=\underbrace{C^{\text{eng}}_{\text{borrow}}}_{(a)\ \approx4\times10^{23}}+\underbrace{C^{\text{soul}}_{\text{borrow}}}_{(a')\ \text{미공개, 콘텐츠 전용}}+\underbrace{C_{\text{struct}}}_{(b)\ \text{학습 0}}+\underbrace{N_wC_{\text{tto}}}_{(c)\ \approx5\times10^{16}/\text{세계}}+\underbrace{C_{\text{own}}}_{(d)\ \approx2\times10^{19}},\qquad \text{실제 지출}=C_{\text{own}}+N_wC_{\text{tto}}+C_{\text{data}}+\text{크레딧}
$$

- (a) **빌린 엔진 교사**(오픈, 동결)가 재질, 형태, 의미, 사진다운 외형을 맡는다. MRH와 G-Adapter로 꺼내 쓰고, 노이즈를 통제하고, 중간층을 읽고, 래칫의 학습 데이터를 만든다. 학습 데이터로 흐를 수 있는 빌린 지식은 이것뿐이다.
- (a′) **빌린 미감 교사 Soul**(닫힌)이 개념 아트, 미술·의상의 생활감, 인물과 서사, 촬영 취향의 레퍼런스를 맡는다. **세계 W(콘텐츠)와 최종 렌더로만 흐르고, 우리 모델의 학습에는 0을 기여한다**(방화벽, CT32). 그래서 C_own에 아무것도 더하지 않는다.
- (b) **구조**가 3D 일관성, 시차, 가림, 왕복 비트 동일, 광학, 강체 물리, 마모장을 맡는다. v2.1에서는 미감 10축 중 약 8축(§3.8)도 여기에 속한다. 경쟁자는 10²⁴를 쓰고도 GeCo 0.07–0.27이고, 실사 바닥은 0.02다.
- (c) **테스트타임 최적화**가 "이 세계"에 맞춘 판정과 수렴을 맡는다.
- (d) **수십M 모델**은 원천들 사이의 접착제다. 엔트로피가 낮고, 국소적이고, 대역이 제한된 함수만 배운다. P1 + P2 1세대 학습이 약 1.8–2.2×10¹⁹ FLOPs(§5.6)다.
- (e) **데이터**는 교사 공장 약 330 T4-h, CPU 약 2,700 CPU-h다(§6.5).

### 2.2 전체 구성요소 표

| 서브시스템 | v1 구성요소 | v2.1 구성요소 (정본 이름) | 분류 | 비고 |
|---|---|---|---|---|
| **엔진 교사** | 12–20B 오픈 교사 | Z-Image-Turbo DiT 6.15B (동결). E1 재오디션 후보: FLUX.2 klein 4B, Z-Image Base, ERNIE-Image-Turbo | (a) | FP16 수술/INT8/W4A4는 같은 가중치의 수치 변환이다 |
| | 교사 VAE | Z-Image VAE (f8, 16ch) | (a) | 이미지 모델의 일부. fp16 자체 수리(제3자 fp16-fix 금지) |
| | — | Qwen3-4B TE (encoder 전용, R-TE) | (a) | 세계당 1회 인코딩 후 캐시하고 언로드 |
| | (Base) | Z-Image Base | (a), 조건부 | 창업자 Q1 |
| **미감 교사** [v2.1] | 닫힌 SOTA art director(약관 확인) | **Higgsfield Soul Cinema / Soul 2.0** (닫힌, 웹 + MCP, 2K 상한, Soul ID·Soul HEX) | **(a′)** | 개념 아트, 룩 레퍼런스, 스틸 광택, 평가 앵커. 방화벽 |
| **룩 프리셋** [v2.1] | 샷 공유 LUT | θ_look = {화면비, 초점거리, f-stop, 초점, 셔터 각, ND/CPL, EV, WB, LUT·팔레트, 그레인·할레이션·비네팅, 렌즈 처방, 대기 β, 태양 고도·방위, 림} | (b) | A: 사람이 Soul 스틸을 보고 측정(`SOUL-INSPIRED`). A+: 자동 역추정(`SOUL-DERIVED`) |
| **교사 어댑터 (우리가 학습)** | ControlNet, `<plate>` LoRA, fake-score LoRA | G-Adapter 25M, Cine/Texture LoRA ≤30M(선택), Chrono-LoRA 15M(P2), fake-score LoRA 30M(P2), identity LoRA 7.4M/캐릭터(P2c), SR-LoRA 70M(조건부) | (d)→(a) | **동시 장착 합 ≤ 엔진 교사의 2%(123M).** Soul에는 어댑터를 붙일 수 없고 붙이지도 않는다 |
| **World Program** | LLM | MassingFM(24M) + CaptionComposer(템플릿·사전) + WorldScript DSL | (d)+(b) | LLM 0 |
| **Bedrock** | CWM 4–8B, DA3/VGGT/AnySplat 부트스트랩, TRELLIS | hero search(엔진 교사, 측량 모드) + (A+) Soul 개념 아트 관측 제약 + MRH lift + MassingFM + BRP(확장) + 문법·CC0 에셋 + 의미 패스 lockstep + CMA-ES 거부권 | (a)+(a′)+(d)+(b)+(c) | 전역 1-pass 예외 |
| **조명 L0** | RAS 정본 | 물리 bake(OptiX/Warp 경로추적, 8 mm, 256 spp) + hero 조명 역적합 + θ_look 태양·림 | (b)+(c) | L_d 정본 유지 |
| **3D 지각** | dreamer depth head, DA3, VGGT, MoGe, SigLIP, DINO, 닫힌 미감 판사 | MRH(24M) (깊이, 법선, intrinsics, sem, σ̂, defocus, Aes, FOV) + Teacher-DCA(파라미터 0) | (d) on (a) | 한계비용 ≤1.5%(FP16 1024² 기준, INT8 기준 1.5–2.3%) |
| **기하 제어** | `ctrl=Gbuffer` (사전학습 ControlNet) | G-Adapter(25M) | (d)→(a) | E3가 생략 여부를 판정 |
| **미세 연산자** | LBO 200M, WDD | TexelFM-35 + Swatch Bank + 마모장 | (d)+(a)+(b) | native는 비트 결정론 |
| **VAE 대리** | — | TAE-T 2.5M, LatentDown/Up-S 2×1M | (d) | 노이즈 마스킹 규칙 |
| **지각 손실** | VGG-LPIPS | T-LPIPS-S 4M | (d) | 교사 특징 증류 |
| **동기화** | Sync(Adam 피팅) | Grid-Locked OTBI + TABP + Fold 뷰 + Context Ring + 고정소수점 정렬 리덕션 + graph cut + multiband | (b) | Adam 0 |
| **노이즈** | WNF | WNF(Philox), 차트에서 ρ=1 | (b) | 저장 0 B |
| **확률적 물질** | G_SM, τ 투표 | 절차 instancer(scatter, L-system) + IPM-S 8M + STRAND/SGGX(솜털·털) | (b)+(d) | 무한 해상도 |
| **텍스트·선·점광원** | PSH, $\mathcal L_{ocr}$ | PSH(MSDF, 해석적 커버리지) + Glyph-S 4M | (b)+(d) | 한글은 SYMBOL이 구성상 정확하게 처리. 교사의 가짜 한자는 Glyph-S가 벌점 |
| **캡션** | VLM SelfCaption | 프로그램 자기지식 캡션(prompt_tree@scale, class, 스케일, lensTags) | (b) | 세계는 자기가 무엇인지 안다 |
| **Solver** | WSSC v3 | WSSC-T4 (§4.3) | (c) | 세계당 약 700 TFU(G0) |
| **Student** | WDM 4–8B | 관할 이양 래칫(TexelFM, MRH, MassingFM 세대 갱신) | (d) | WDM 연기. Soul 세계 셀은 래칫에서 제외 |
| **표현** | STRATA(latent + UKA) | STRATA-lite(TSQ + BC 페이지, `z_slot` 예약, `own`·`prov`에 Soul 표시) | (b) | §7 |
| **렌더** | 3DGRUT | gsplat-TSQ 래스터 + OptiX/Warp RT + 평면 재귀 + DIEL SDF + dual-cut + 공분산 + Defocus Pyramid + LH-split | (b) | 학습 0 |
| **렌즈·센서·필름** | 수식 | 수식 (Kolb, Hullin, FFT glare, Newson 그레인) + **빈티지 수차 처방, CPL 필터** | (b) | 학습 0 |
| **카메라** | DP-0 DiT, IMU 손, CRM | CamFM-Intent 20M, CamFM-Hand 8M, CamFM-CRM 12M, Rig-ID-S 3M + 1 kHz 리그 ODE + Toric + CEM/MCTS + 전경 레이어링 보상 + 스케일 연출 레시피 | (d)+(b)+(c) | Soul 출력으로 학습하는 것은 **CamFM-CRM(구도 평가기)만 S1**. CamFM-Intent(궤적 생성 prior)는 S2(§3.7.4) |
| **수요** | Ray-Bundle | Ray-Bundle + Corman–Shannon(대역평균 OTF) | (b) | [신규] |
| **광류** | RAFT/UFM | DIS / TV-L1 | (b) | — |
| **실사 lift** | DA3/VGGT/MegaSaM | COLMAP 4.2 SIFT + GLOMAP + Caspar + PatchMatch + TSQ 피팅 | (b)+(c) | — |
| **Hero FX** | FLIP/MPM + CEM | Warp MPM/CFL 적응 FLIP + Voronoi 파쇄 + CEM | (b)+(c) | 유리는 MPM이 아니라 Voronoi |
| **디노이저** | OIDN | 고전 à-trous/SVGF (P1), Denoiser-S 1M (P2) | (b)/(d) | — |
| **P2 사건** | EPD 0.5–2B | EPD-S 28M + Feas-S 5M + Chaos Bank | (d)+(b) | — |
| **P2 동역학** | PCGD + F_φ 0.3–1B | 고전 시뮬 + PCGD-0(약 300 파라미터) + F_φ-S 0.6M + Milestone SMC | (b)+(d)+(c) | — |
| **P2 캐릭터** | PHC, SMPL-X, GVHMR | MoPri-S 16→32M × ID-Screen × MJPC-lite + 자체 스켈레톤 + MakeHuman(CC0). (A) Soul ID는 캐릭터 개념 아트로만 | (d)+(b)+(c)+(a′) | identity LoRA를 Soul ID 출력으로 학습하는 것은 금지 |
| **P2 시간 판정** | 비디오 증인 + PLAC | 크로노 증인 GICC(엔진 교사) + 오디오 간격 + Garage Gold + 감독 | (a)+(b)+사람 | [신규]. Soul은 판사가 아니다 |
| **P2 외관** | MAN g_φ 0.2–0.5B, slice DM | MAN-S 1.5M(Exhaust-to-Shader + DMD 5k) | (d)+(a) | — |
| **P2 Reader** | lift 스택 | Reader-S 26M(R-Flow/R-Kpt/R-Seg/R-Event) + Lift Ladder | (d)+(b) | — |
| **P2 판정 대리** | — | HeroProxy-S 14M, ChronoProxy-S 12M | (d) | INT8 |
| **P2 음향** | 모달, TTS | 모달 FEM + Minnaert + image source + RT 음선. 음성은 녹음 → TTS-S(2c 후반) | (b)+(d) | — |
| **평가 전용** | GeCo(VGGT/UFM), MEt3R, LPIPS, OCR, ArcFace, VBench | 동일 + **Soul 스틸 앵커(E1-S, TP_S, 미감 부검)** | 평가 | `eval/` import 금지 린트(R-EVAL). Soul 앵커 결과는 사람 보고서로만 |

### 2.3 요약: 어디를 무엇으로 메우나

| v1에서 큰돈이 들던 곳 | 무엇으로 메우나 |
|---|---|
| CWM 10²³ 학습 | (a) hero 구도 + (d) MassingFM(절차 데이터라 라벨 공짜) + (c) 의미 패스 lockstep + (a′) Soul 개념 아트(A: 사람 전사, A+: 관측 제약) |
| LBO/WDM 10²³ | (c) 세계별 WSSC-T4 + (d) TexelFM이 가난선 아래를 맡고, 래칫으로 가난선을 올린다 |
| 3D 파운데이션 스택 | (a) 엔진 교사가 이미 가진 형태 지식을 (d) MRH로 꺼내 쓴다 |
| 닫힌 SOTA art director, 미감 판사 | (a′) Soul(개념·미술·취향 레퍼런스) + (b) 물리 카메라·조명·필름(10축 중 8축) + (d) MRH-Aes ⊕ Teacher-DCA |
| 비디오 증인과 동역학 학습 | (b) 물리 시뮬레이터 + (c) SMC와 best-of-N 탐색 + (a) 크로노 증인 + 사람 |
| lift 5천만 클립 | 자체 캡처 300–500 장면 + 수요 기반 lift(Event-Demand) |
| 8K·120fps | (b) 렌더 비용뿐이다. 생성 비용은 fps와 무관하다 |

### 2.4 두 교사의 계약 비교 [v2.1]

| 항목 | 엔진 교사 (오픈, 예: Z-Image-Turbo) | 미감 교사 (Soul, 닫힌) |
|---|---|---|
| 접근 | 가중치, σ, 중간층 특징, VAE, TE | 프롬프트, 레퍼런스 이미지(Cinema는 레퍼런스를 넣으면 프롬프트 비활성), 프리셋, Soul ID, Soul HEX. σ·특징 접근 없음(ρ≈0) |
| 역할 | 측량 기사: 재질·기하·의미 판정, 노이즈 통제, 프로브, 래칫 데이터 | 아트 디렉터·촬영 레퍼런스: 개념 아트, 룩, 스틸 광택, 평가 앵커 |
| 학습 데이터로 | 허용(Exhaust → MRH, TexelFM, T-LPIPS-S, HeroProxy-S·ChronoProxy-S) | **금지**(약관). 승인 S 범위 안에서만 소형 모델 일부 해금 |
| 판정·보상 | 허용(판정이 곧 화폐) | **금지**. 판사가 아니다. 평가 앵커 결과는 사람 보고서로만 |
| 세계 W로 | 대역 소유권 절차로 | A: 사람 경유(전사·측정)와 픽셀 그대로 붙인 콘텐츠만. A+: 관측 제약·POLISH 소유 |
| 결정론 | 엔진 해시 고정, Proof-of-Shot | 재생성 불가. **바이트와 해시를 보관**하고 Proof-of-Shot은 "Soul 자산 해시가 주어졌을 때"의 렌더 결정론을 보증한다 |
| 해상도 | 1024² 판정, 차트 무한 | **2K 상한** |
| 비용 | TFU(T4) | 크레딧(웹 무료는 사람 경로만) |
| 교체 가능성 | Dreamer 계약 슬롯, 교체 비용 50–100 T4-h | 미감 교사 슬롯. θ_look은 물리 파라미터라 교사가 바뀌어도 남는다 |

### 2.5 인수분해 배당 D1–D6과 KPI [v2.1 복원: CB(u) §5.2.1]

피치의 정량 근거다. "경쟁자보다 자릿수 4개 작은 학습 예산(§1.3)으로 왜 경쟁이 되나"를 배당 여섯 개로 나눠 적는다.

| # | 배당 | 숫자 | 메커니즘 |
|---|---|---|---|
| D1 | 구조 배당 | 10²⁴ 이상 상당, 지출 0 | 렌더러는 파라미터 0개짜리 완벽한 모델이다(T1, T2). 경쟁자는 10²⁴–10²⁵를 쓰고도 GeCo 0.07–0.27이다 |
| D2 | 표본 배당 | 같은 연산당 독립 장면 수가 이미지 쪽이 **약 300배(1024²)–1,500배(저해상 혼합)** | 이미지 표본은 토큰 4k, 1080p 10초 비디오 클립은 약 49만 토큰. 빌린 이미지 교사는 같은 연산으로 훨씬 많은 장면을 봤다 |
| D3 | dedup | 10초 영상에서 r = 10–25, 재방문이 있는 60초에서 r = 60–150 | 우리는 프레임이 아니라 **고유 텍셀** 단위로 비용을 낸다. 비디오 모델은 같은 벽을 프레임마다 다시 그린다 |
| D4 | 스펙트럼 배당 (가난선) | 옥타브 하나를 넘길 때마다 교사 픽셀이 1/4. 방 하나의 재질 패스에서 교사 뷰 443 → 100 | §4.1 P1 |
| D5 | 재사용 | 세계 1개 × 리그 6종 × 스타일 50종. 4K120 10초 샷 하나의 한계비용 약 7–15 T4-분(8K120이면 약 25–60 T4-분, §7.7) | 세계 단위 과금의 원가 근거 |
| D6 | 해상도 | 8K120은 렌더 비용만 든다. 비디오 모델은 토큰 80배, 어텐션 6,400배 | 출력 포맷과 세계 연산이 분리된다 |

- **KPI: FLOPs/고유 텍셀.** CB(u)의 교정값으로 우리 약 1.6×10⁸, Seedance 약 2×10⁸–10⁹(dedup r = 10–25, L20 한 장 가정)이다. CB(o)의 "1.4×10⁸ vs ≤3×10⁸"은 교정 전 값이라 쓰지 않는다. 가정이 많은 비교라 **내부 KPI로만** 쓰고 마케팅 문구에는 쓰지 않는다.
- 배당은 곱해지지 않는다. D2는 빌린 교사의 사전학습 효율이고, D3–D6은 우리 지출을 줄이는 항이다. 각 배당의 실측은 E2(D4), E4(D3), §7.7(D6)에서 한다.

---

## 3. 이미지 모델 선정: 엔진 교사 + 미감 교사

### 3.1 왜 교사가 둘인가

- **역할 반전 [FD (1)].** 오픈 엔진 교사(Z-Image)의 "지루한" 디폴트는 한낮 평광, 팬포커스, 정면, 빈 공간, 가림 없음이다(단, 매크로·근접에서는 Z도 얕은 심도와 보케를 굽는다. 이끼 판, §3.8.1). 최종 룩으로는 약하다. 그러나 3D 세계의 재질과 기하를 뽑는 원천으로는 이상적이다. 그림자, 심도, 그레인, 전경 가림이 구워져 있지 않아서 설명해 빼낼 것이 적다. 이것이 **측량 기사**의 일이다. 반대로 Soul 스틸에는 그림자, 심도, 그레인, 전경 가림이 구워져 있다. 세계 재료로는 다루기 어렵고, 최종 룩의 기준으로는 이상적이다. 이것이 **촬영감독·미술감독**의 일이다.
- **엔진 교사는 열려 있어야 한다.** WNF로 노이즈를 통제하고(ρ=1), σ별로 DR을 판정하고, 중간층을 MRH로 읽고, Exhaust로 래칫을 돌려야 한다. 가중치와 σ에 접근할 수 없는 모델은 이 일을 못 한다.
- **미감 교사는 닫혀 있어도 된다.** v1 §1.6 (iii)이 이미 길을 열어 두었다. σ에 접근할 수 없는 닫힌 모델은 노이즈 상관 ρ≈0이라 lockstep 대역 소유에는 못 들어오지만, T6(분할 원리) 덕분에 POLISH 단계의 소유자 후보는 될 수 있다(렌더 → 편집 → 정렬 → gauge 제거). v2.1은 이 슬롯을 Soul로 채우고, 약관 때문에 그 앞에 방화벽(§3.7.3)을 친다.
- **결과.** 엔진 교사는 미감이 아니라 **통제성·속도·라이선스·측량 품질**로 고른다(CT35). 미감은 연방이 정한다(CT36, §3.8).

### 3.2 엔진 교사 재오디션 (E1, Day 3–18)

**기준.**

| 기준 | 정의 | 측정 |
|---|---|---|
| 통제성 | σ 접근, 임의 σ SDEdit 충실도(연속 σ vs 학습 σ 점), CFG 유무, 멀티 레퍼런스, f8 정렬 latent | E2 결정 곡선, E0 (e) DR 판별 AUC |
| 프로브 가능성 | 중간층 특징이 기하·재질을 담는가 | E3 MRH 실사 AbsRel, 차입 배당 ≥30% |
| 측량 품질 | 물리 조명 아래 정면 차트 뷰 재질의 정확도("스톡 텍스처 함정", 식물학적 디테일) | E1-E 보조 arm: 차트 텍스처 vs CC0 사진 2AFC ≥45% |
| 속도 | TFU/T4-h, INT8·W4A4 경로 존재 | E0 |
| 라이선스 | 상업 사용 가능(Apache-2.0) | 법무 |
| FP16 안전성 | Rescue 후 NaN·검은 이미지 0/10,000(교사·정밀도별) + commit σ에서 FP16 수술판 별도 2,000 | E0 (a) |

**후보 (1024², T4 1장, 속도는 전부 (추정)이고 E0에서 확정).**

| 후보 | AA Elo | 파라미터 (본체 + TE) | 라이선스 | 통제성 | 측량 품질 관찰 | FLOPs/fwd | s/fwd FP16 / INT8 / W4A4 | 1 fwd = TFU | 방 1개 교사 시간 (INT8, 700 TFU 환산) | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Z-Image-Turbo** (현직) | 941 | 6.15B S3-DiT(30층, d=3840) + Qwen3-4B | Apache-2.0 | 학습 σ 8점에서만 신뢰(σ 스냅), CFG 없음. bf16 네이티브라 FP16 수술 필요 | 이끼 판에서 식물학적 디테일로 유일하게 1승. 평광·팬포커스 디폴트 = 측량 이상형. 현판에 가짜 한자(SYMBOL이 소유하므로 무해) | 6.2×10¹³ | 4.5 (3.1–4.5) / 2.6 / 1.8 | 1.0 | 약 0.5 T4-h | **기준선이자 현직** |
| Z-Image Base | — | 같은 구조 (공개·라이선스 확인 필요) | 확인 필요 | 비증류, 연속 σ, CFG(판정 1회 = 2 fwd). DR 판별력 향상 기대 | 미측정 | 6.2×10¹³ | 같음(판정은 ×2) | 1.0 (판정 2.0) | 판정 몫만 ×1.5–2 | **Q1: 판정 전용 병용 후보.** 미감이 Soul로 넘어가 매력이 커졌다 |
| FLUX.2 klein 4B | 863 | 4B + Qwen3-4B | Apache-2.0 | 4-step 증류, **멀티 레퍼런스 편집 내장**(견본을 레퍼런스로 줄 수 있음), VAE 32ch f8 [2차] | 미측정 | 약 4×10¹³ | 2.0–2.7 / 1.7 / 1.1 (Nunchaku 지원 미확인) | 약 0.65 | 약 0.33 T4-h | **도전자.** Z-Image와의 미감 격차(39:61)의 비용이 Soul 덕에 줄었다. ES가 3%p 이내면 약 1.5배 빠른 klein(INT8 2.6/1.7 s, 0.5/0.33 T4-h) |
| ERNIE-Image-Turbo | 912 (924) | 8B | Apache-2.0 | Turbo 계열(σ 스냅 예상). 24 GB GPU 권장 | 미측정 | 약 8×10¹³ | FP16 불가(가중치만 16 GB) / 3.2–4.0 / 2.2–2.8 (W4A4 경로 미확인) | 약 1.3 | 약 0.65–0.8 T4-h | 후보(INT8 전용). T4 메모리 리스크가 크다 |
| Qwen Image Max 2512 | 999 | 약 20B + 7B VL | Apache-2.0 | 8 NFE(Lightning은 제3자라 금지) | — | 1.9×10¹⁴ | — / — / 4.2–5.0 | 약 3 | 약 0.8–1.0 T4-h (W4A4, 700 × 4.2–5.0 s) | **강등.** v2.0의 "미감 승급" 역할은 Soul이 대신한다. ES가 모든 후보에서 미달일 때만 |
| SDXL (+Lightning) | 678 | 2.6B + 0.8B | OpenRAIL++ | 20–30 NFE × CFG (Lightning은 제3자, DMD2는 NC) | 2023 세대 | 6.1×10¹² | 0.55–0.8 / 0.44 / — | 0.1 | 약 0.09 T4-h (700 × 0.44 s) | 1–2주차 개발 대역만 |
| 제외 | — | Qwen-Image-2.1, FLUX.2 klein 9B, FLUX.1-dev, SD3.5 Large | 비상업·커뮤니티 | — | — | — | — | — | — | 라이선스 |
| 제외 | — | SANA / Sana-Sprint | 출처 상충 | DC-AE f32라 latent Nyquist 위 디테일을 디코더가 지어낸다 | — | — | — | — | — | 대역 소유권 이론과 충돌 |

**결정식 (CT35).** 엔진 계약(§3.6 A)을 만족하는 후보 중 **ES 증분 / 수용 세계당 T4-h**가 가장 큰 교사를 고른다. ES = 차트 뷰 재질 2AFC, MRH 실사 AbsRel, DR 판별 AUC, SDEdit 저대역 보존의 표준화 평균이다. ES 차이가 3%p 이내면 빠른 쪽을 고른다(RR(u)의 klein 규칙). 미감은 이 식에 들어가지 않는다.

### 3.3 결정과 폴백 사다리

1. **엔진 교사 = Z-Image-Turbo**(재오디션에서 지기 전까지). 시스템 경로에는 Day 18 이후 엔진 교사 하나만 남긴다. bake-off와 개발 스캐폴드는 평가 용도다.
2. **폴백 사다리:** Z-Image **FP16 수술판** → INT8 → W4A4(큰 σ에만) → **FLUX.2 klein 4B**(FP16과 INT 경로가 모두 실패할 때) → SDXL은 개발 대역으로만 쓰고 데모에는 쓰지 않는다.
3. **승급:** ES가 모든 후보에서 기준 미달이면 Qwen Image Max 2512 INT4(Q3). 세계당 교사 시간이 약 2배가 되지만 90일 안에 들어간다. **미감 미달은 승급 사유가 아니다.** 미감은 §3.8의 연방이 맡는다.
4. **교사 교체 비용:** MRH 탭 사영층, G-Adapter, TAE-T, LatentDown/Up-S를 다시 학습하면 50–100 T4-h다. 나머지 카탈로그는 교사와 독립이다. TFU 가격표와 Exhaust는 버전 태그로 다시 보정한다.
5. **[관찰] 부모 연산 차입 원칙.** 거대한 부모에서 증류된 공식 자식이 가장 좋은 차입 대상이다. 추론 비용은 자식 크기를 따르고 품질은 부모 연산을 따른다. Turbo는 CFG가 없어 NFE도 절반이다. 증류 모델이라는 단점이 T4에서는 장점이 된다.
6. **[v2.1] 측량 모드.** CaptionComposer는 차트 뷰, 견본, Realify 운송, 텍스트 모드 hero 1차 탐색에 "흐린 날 확산광, 깊은 심도, 정면, 사람 없음" 계열 측량 태그를 붙인다. 엔진 교사의 디폴트가 원래 그쪽이라 비용이 없다. 극적인 조명과 심도는 θ_look과 Gaffer가 물리로 만든다. **단, 매크로·근접에서는 엔진 교사도 심도를 굽는다**(이끼 판). 차트 뷰와 견본은 측량 태그에 더해 MRH defocus 검사를 거치고, 임계를 넘으면 재샘플해 깊은 심도를 강제한다.

### 3.4 Teacher Ceiling Test v2.1 [신규 프로토콜] (E1, Day 3–18)

- **질문.** (i) 엔진 교사 스틸이 Seedance, Kling, Wan의 "멈춘 세계" 영상에서 무작위로 뽑은 프레임을 단일 프레임 블라인드 2AFC로 이기는가(E1-E)? (ii) Soul 스틸은 얼마나 이기는가(E1-S)? (iii) 엔진 재질 + Soul 룩을 물리로 합친 것은 어디까지 가는가(X-LA0, §3.8.6)?
- **셋업.**
  - FREEZE-TAG 프롬프트 초안 6범주 × 20 = 120개
  - **E1-E 선택 정책 [RR X-T1 복원]:** 두 arm을 모두 보고한다. (i) 무작위 arm: seed 4개 중 무작위 1장(E1-S, 경쟁 프레임과 같은 정책). (ii) 파이프라인 arm: best-of-64(MRH 없이 Teacher-DCA 선별, 실제 hero search와 같은 정책). 두 천장의 선택 정책을 대칭으로 맞추려는 것이다
  - **E1-E+θ arm [RR X-T1 복원]:** 엔진 스틸에 우리 렌즈·센서·필름 스택과 사람이 측정한 θ_look을 2D 후처리로 적용한다. 성공 기준은 E1-E 대비 +5%p 이상(광학 가산점). H-LA1의 D18 조기 신호다
  - 경쟁 API로 클립 생성(API 비용 $300–600), 1080p, 같은 코덱으로 정규화
  - **E1-S:** 같은 120 프롬프트를 창업자가 웹에서 Soul Cinema로 생성한다(배치 4 → 그중 1장을 무작위로 고른다. 경쟁 프레임도 무작위라서 공정하다). 2K → 1080p, 화면비는 중앙 크롭으로 맞추고, 화면비 자체의 효과는 따로 보고한다. 웹 480회
  - 평가자 30명, 약 4,000쌍(사람 평가 $1.5–5.5K)
  - 보조: 정면 텍스처 뷰 vs CC0 사진 텍스처("스톡 텍스처 함정"), 엔진 후보별
- **판정.**
  - 엔진 트랙(CT2): C1 기준 E1-E ≥60%면 녹색, 55–60%면 황색, <55%면 적색.
  - 하이브리드 트랙: SP_S = E1-S 승률을 기록하고 **TP_S\* = 1 − SP_S**를 진단 기준선으로 보고한다(§3.8.5). 보고 항목이고 게이트가 아니다.
  - 엔진 교사 잠금과 승급은 E1-E와 ES로만 판정한다. Soul 앵커 결과(SP_S)는 이 판정에 쓰지 않는다(Q6). 엔진 트랙이 적색이면 황색 의무(DTC, Gaffer, LoRA)를 지고 X-LA로 어느 몫이 비는지 찾는다. 미감 미달은 승급 사유가 아니다(§3.3 항목 3).
  - 하이브리드 트랙 개방은 창업자 결정(CT31)이다. SP_S로 정하지 않는다.
- **의미.** Elo로는 비디오 프레임 대비 미감을 예측할 수 없다. 대형 팀은 파이프라인을 다 만든 뒤에야 천장을 안다. **우리는 3주차에 두 천장을 다 안다.**
- **규칙(RR(u) 계승).** SP 게이트를 세계 기계로 메우려는 시도는 금지한다. 천장이 모자라면 원인을 재고 그 몫을 소유한 부품을 고친다.

### 3.5 교사를 T4에서 쓸 수 있게 만드는 처방

| 처방 | 내용 | 등급 |
|---|---|---|
| **FP16 Rescue (Zero-Training Residual Rescale)** | (i) pre-norm(RMSNorm) 블록에서는 잔차 스트림 전체에 상수 1/s(예: 1/32)를 곱해도 norm 출력이 변하지 않는다. 입력 임베딩, 모든 블록 출력 투영 W_o, adaLN의 잔차 쪽 출력(gate)에 1/s를 곱해 스트림 크기만 줄인다. (ii) **SwiGLU 재균형** [v2.1 복원]: W_up·W_gate 중 선형 쪽을 1/s, W_down을 s배로 둔다. 출력은 같고 중간값만 1/s가 된다. (iii) norm, softmax, RoPE, AdaLN 변조, 최종 norm, VAE 첫 conv는 FP32 섬으로 남기고 SDPA는 FP32로 누산한다. (iv) NFE 1천 회 동안 블록별 max-abs를 프로파일링해 s를 고른다. 수학적으로 동치이고 학습이 필요 없다. sandwich-norm이면 post-norm 가중치에 곱한다(아키텍처 확인 필요) | [조합] |
| **정밀도 사다리** | 큰 σ와 탐색은 W4A4, 결정 창은 INT8, 최종 재판정과 commit은 FP16. 양자화 오차 예산을 σ_t에 비례하게 둔다 | [조합] |
| **σ 점 스냅** | Turbo는 학습된 8개 σ에서만 믿을 수 있다. SDEdit 시작점을 그 집합으로 반올림하고, "학습 스케줄의 연속 2 step + 같은 σ에서 재판정"으로 운용한다 | [부품] |
| **SR-LoRA (= t-densify LoRA)** | 결정 창이 1.5옥타브를 넘거나 DR 판별 AUC < 0.7이면, 교사 자기 궤적을 자기증류해 σ 격자를 촘촘히 한다. rank 32 전 선형층 약 70M(1.1%), 약 20 T4-h | [조합], 조건부 |
| **판정 정밀도 게이트** | INT8/INT4 판정은 1024² FP16 판정과 Spearman ≥0.9이고 \|ΔD\| ≤0.05일 때만 쓴다. 주 1회 FP16 재판정으로 드리프트를 감시한다 | [운용] |
| **어텐션** | 교사 fwd는 flash-attention-turing(hd 128). attention 몫 8.7×10¹²에서 최대 2.19배 빨라진다 | [부품] |
| **TE 처리** | 세계마다 구조화 캡션 조합(노드 약 50 × 스케일 8 × 렌즈 4, 1,600개 이하)을 한 번에 인코딩해 캐시하고 TE를 해제한다. GPU 30초 또는 CPU 5–15분 | [운용] |
| **VAE fp16 자체 수리** | activation 스케일을 재분배하고 mid-block만 FP32로 둔다. 제3자 `fp16-fix` 가중치는 쓰지 않는다. 1024² 디코딩 FP32 1.5–2 s → FP16 0.3–0.5 s | [조합] |
| **노이즈 마스킹 규칙(TAE-T 사용 규칙)** | 근사 enc/dec(TAE-T)는 대역 b의 근사 오차가 0.1·σ_t·√P_ε(b) 미만이거나 b가 아직 미결정일 때만 쓴다. **결정 대역을 쓰는 commit 스텝은 원 VAE로 디코딩한다.** TAE-T 학습에 교사 일관성 항 ‖D_θ(Enc_TAE(x),t) − D_θ(Enc_VAE(x),t)‖을 넣는다 | [신규] |
| **클럭 고정** | `nvidia-smi -lgc`, E0의 TF/W 곡선으로 값 결정 | [운용] |

### 3.6 교사 계약 v2.1

**(A) 엔진 교사 계약**

1. 빌린 오픈 이미지 모델은 하나이고 동결한다(Base를 판정 전용으로 병용할지는 Q1).
2. 붙일 수 있는 것은 우리가 처음부터 학습한 경량 어댑터(합 ≤2%), 우리 손으로 한 양자화(우리 보정셋, prov 검사 통과), FP16 Rescue, VAE fp16 자체 수리뿐이다. 제3자 LoRA, ControlNet, 파인튜닝 VAE, 사전 양자화 가중치는 Q7 판정 전까지 금지한다.
3. f8 공간 정렬 latent여야 한다. WNF와 대역 소유권이 균일한 footprint에 기대기 때문이다.
4. σ에 접근할 수 있어야 한다(오픈 가중치).
5. 상업적 사용 라이선스여야 한다.
6. E1-E에서 녹색 또는 황색이어야 한다. 적색이면 §3.4대로 황색 의무를 지고 X-LA로 빈 몫을 찾는다. Soul 앵커 결과는 이 판정에 쓰지 않는다.
7. **선택식:** ES 증분 / 수용 세계당 T4-h 최대(CT35).

**(B) 미감 교사 계약 [v2.1]**

1. 빌린 닫힌 이미지 모델은 하나다. Soul Cinema를 우선하고 Soul 2.0을 보조로 쓴다. 자산마다 모델, 버전, 날짜, 플랜, 경로(웹/MCP), 프롬프트를 기록한다.
2. **출력은 세계 W(콘텐츠)와 최종 렌더로만 흐른다**(방화벽, §3.7.3). 학습, 손실, 보상, 보정, 체크포인트에는 쓰지 않는다. 승인 S 범위가 유일한 예외다.
3. 웹 무료·Unlimited는 사람 경로에만 쓴다. MCP는 크레딧이다. 웹 앱 자동화는 쓰지 않는다.
4. 우리가 올리는 레퍼런스 이미지는 Higgsfield가 자기 학습에 쓸 수 있다(Enterprise 제외). **공개가 확정된 히어로 프레임 외의 우리 렌더는 올리지 않는다.** 미공개 데모의 세계 경유 POLISH와 X-Soul 레퍼런스 업로드는 공개 확정 뒤 또는 Enterprise 조건(Q19)에서만 한다. 제3자 권리가 있는 이미지는 올리지 않는다.
5. 상업 공개 전에 IP 점검을 통과해야 한다(§8.3).
6. 비디오 프레임별 광택은 금지한다(CT37).
7. **제거 가능성:** Soul을 빼도 Engine 트랙이 단독으로 돌아야 한다. 벤치 Engine 트랙이 이것을 상시 증명한다.

### 3.7 미감 교사 Soul: 사실, 역할, 방화벽, 승인, 예산 [v2.1 신규]

#### 3.7.1 확인된 사실 (2026-10, FD의 웹 확인)

| 항목 | 내용 | 근거 |
|---|---|---|
| Soul 2.0 | Higgsfield 자체 파운데이션 이미지 모델. 패션·에디토리얼 지향. 프리셋, 무드보드, 레퍼런스 이미지 모드, Soul ID(캐릭터 일관성), Soul HEX(색 팔레트). 품질 옵션 1.5k / 2k | FD |
| Soul Cinema | 클로즈업, 분위기, 질감 중심의 시네마틱 모델. 미감이 모델에 내장(프리셋 없음). Soul HEX, Soul ID 지원. **레퍼런스 이미지를 넣으면 프롬프트 입력이 비활성화된다.** 배치 최대 4. MCP 지원 미확인(MCP 안내 페이지에는 Soul V2만 언급) | FD |
| 해상도 | 이미지는 전 플랜 **2K 상한** | FD |
| MCP | `https://mcp.higgsfield.ai/mcp`, OAuth. Claude Code 등록: `claude mcp add --transport http --scope user higgsfield https://mcp.higgsfield.ai/mcp`. **MCP 생성은 항상 크레딧을 쓴다. Unlimited와 무료 생성은 웹에서만 적용된다** | 공식 도움말 |
| 가격 | Soul V2 MCP 1회 약 10 크레딧, $1 ≈ 16 크레딧 | 3rd-party 추정, **미검증** |
| 약관 | "You may not use the Service or outputs to train, fine-tune, or improve any AI or machine-learning model, whether or not it competes with Higgsfield." 예외는 서면 승인. 출력 소유권은 사용자, 상업 사용 제한 없음. Higgsfield는 사용자가 올린 콘텐츠(레퍼런스 포함)로 자기 모델을 학습할 수 있다(Enterprise 제외) | 약관 |
| 가용량 | 웹 무료 약 2,900회 + 크레딧, 합쳐 약 4,500회 | 창업자 추산 |

#### 3.7.2 다섯 역할

| 역할 | 무엇을 하나 | 세계·렌더로 가는 경로 | 90일 호출 (웹 / MCP) | 약관 적합성 | 단계 |
|---|---|---|---|---|---|
| **(a) 개념 아트 = 관측 제약** | 해석형 프롬프트의 개념(스케일 전환, 신화적 서사, 상호작용), 구도, 미술 생활감 | **A:** 아트 디렉터가 보고 WorldScript·소품·스케일·인물 설정으로 전사한다. 캔버스가 아니다. **A+:** 스틸의 그레인·심도·그레이딩·비네팅을 θ_soul로 설명해 뺀 뒤, 엔진 교사에 약한 σ(스냅 0.2–0.3)로 통과시켜 MRH가 깊이·법선·재질을 읽는다(독심 판독). 이것을 frustum 안 레이아웃 대역의 DPS 관측 제약으로 쓴다. 뒷면 등 미관측부는 MassingFM·BRP·엔진 교사가 개념과 일관되게 채운다 | 데모 360 / 벤치 Hybrid-A+ 640(승인 시) | A: 안전(사람의 참고). 판독 자체는 추론이지만 그 결과로 세계를 피팅하면 회색 → A+ | A / A+ |
| **(b) 룩 추출** | 팔레트, 조명, 그레이딩, 렌즈 성격 → θ_look 룩 프리셋 | **A:** 사람이 측정한다(화면비, 망원 압축과 추정 초점거리, f-stop, 태양 고도·방위, 림, EV, 컬러리스트가 매칭한 LUT, 그레인, 비네팅, 렌즈 성격). Soul HEX 팔레트는 LUT 기준색이 된다. 결과는 **룩 프리셋 라이브러리**(`SOUL-INSPIRED`). **A+:** θ_soul 자동 역추정, LUT 자동 피팅(`SOUL-DERIVED`) | 240 / — | 수동: 안전. 자동 피팅: 회색(A+) | A / A+ |
| **(c) 최종 미감 대역 소유 후보** | DP-0 경로 위 히어로 프레임의 구조 보존 광택 | **스틸**(키 아트, 포스터, 썸네일): 픽셀 그대로 = 콘텐츠(A). **세계 경유**: Shoot 전에 CoW로 다시 연 1개 Jacobi 라운드(엔진 교사 뷰와 동시)의 POLISH 소유, 소유 마스크 안, Soul 2K Nyquist 이하 대역만, 엔진 교사 재판정 후 commit(A+, v1 §1.6 iii). **비디오 프레임별: 금지**(CT37). 구조 보존 편집이 실제로 되는지는 X-Soul로 확인한다(Cinema는 레퍼런스 모드에서 프롬프트가 꺼진다) | X-Soul 160 / 200, 키 아트 200 / 데모 스틸 120 | 스틸: 안전. 세계 경유: 회색(A+). 프레임별: 패러다임 위반 | A / A+ |
| **(d) 평가 앵커** | E1-S Soul 천장, TP_S 블라인드 2AFC, 미감 부검, 버전 감시 | 없음. 사람 평가 보고서 전용(R-EVAL) | E1-S 480, TP_S 320, X-LA 140 / 버전 감시 240 | 사람 블라인드 비교이고, 결과는 사람 보고서 전용(진단)이다. 엔진 교사 잠금, 트랙 개방, 게이트, 체크포인트·하이퍼파라미터·프리셋 선택에는 쓰지 않는다(Q6): 안전에 가까운 회색. **미확인:** 발송본은 "사람이 품질을 평가할 때 Soul 스틸을 우리 렌더 옆에 둔다"고만 썼고 용도 제한은 적지 않았다. 후속 문의(티켓 #317768)에서 E1-S(경쟁 프레임 대비)와 TP_S의 용도를 밝힌다 | A |
| **(e) 콘텐츠 자산** | 세계 속 액자·포스터·간판 이미지, 원경 매트 페인팅(far-field, 무한원) | 픽셀을 그대로 붙인다(최적화 0). 2K로 충분한지는 Corman–Shannon 수요가 판정한다. 대기 원근과 심도로 흐린 원경은 2K로 충분한 경우가 많다. 무한원이라 오빗에서도 시차 문제가 없다 | 예비에서 | 안전(출력 소유, 상업 사용) | A |

#### 3.7.3 Soul 방화벽 [v2.1 신규 운용]

1. **흐름 규칙.** Soul 픽셀은 세계 W(콘텐츠)와 최종 렌더로만 흐른다. 어떤 데이터셋, 손실, 보상, 보정 테이블, 체크포인트로도 흐르지 않는다.
2. **태그.** `SOUL`(원본 픽셀), `SOUL-DERIVED`(Soul 픽셀을 입력으로 계산한 것: 판독 결과, Soul 스틸을 통과시킨 엔진 교사 호출의 Exhaust, θ_soul 자동 피팅, LUT 자동 피팅, Soul 관측으로 피팅한 세계), `SOUL-INSPIRED`(사람이 보고 만든 것: 수동 θ 프리셋, 전사한 WorldScript). STRATA 블록의 `own`에는 Soul이 저술한 셀(A+)을 표시하고, `meta.prov`에 자산 해시·모델·버전·날짜·플랜을 기록한다. Merkle 키에 `prov`가 들어간다.
3. **조상 검사.** 시스템 경로의 학습·보정 산출물의 계보 DAG에 `SOUL` 또는 `SOUL-DERIVED` 조상이 있으면 빌드를 실패시킨다. `SOUL-INSPIRED`는 콘텐츠와 프리셋에는 허용하고, 학습에는 법무 의견 전까지 기본 제외한다(Q17). S 승인 범위에서만 해금한다. **태그는 상속된다.** `SOUL-INSPIRED` 프리셋(θ_soul 수동 측정)으로 렌더한 프레임과 그 프레임이 든 사람 판정 쌍도 `SOUL-INSPIRED`다(X-LA arm C·D·K, Hybrid-A 렌더 포함). Q17 판정 전까지 CamFM-CRM·MRH-Aes 학습 쌍은 수제 프리셋(§7.6 출처 i)으로 렌더한 프레임으로만 모은다.
4. **격리.** `soul_vault/`는 학습 잡 계정이 마운트할 수 없는 별도 저장소다(권한 기반 격리). 학습 샤드 빌더, Texel Bank 색인기, TexelFM 래칫 샘플러, MRH 자가 라벨 선택기, Exhaust 학습 분할기, MRH-Aes·CamFM-CRM 쌍 빌더, HeroProxy-S·ChronoProxy-S 증류기, TFU 보정기가 모두 `prov` 검사를 거친다. **기생 학습 경로도 같다.** Feature-Tap Bus 링버퍼 기록기(TF(o)의 온라인 헤드 학습), TRT INT8/W4A4 양자화 보정셋 빌더, 교사 VAE enc/dec 쌍 수집기(TAE-T 데이터)도 `prov` 검사를 거친다. soul=1 호출은 `W.prov.soul ≠ none`인 세계의 모든 엔진 교사 호출이다(S1′ Soul 판독, S8.5 세계 경유 POLISH, 그리고 그 세계의 S2c·S4·S7·S8. §4.1 P5의 soul 비트 정의와 같다). 이 호출은 **Tap Bus를 끈 별도 프로세스**에서 돌고, 그 탭·latent·VAE 쌍은 링버퍼에 들어가지 않는다. Exhaust 학습 분할 검사는 사후 분할만 막으므로 이 실시간 경로를 따로 막는다.
5. **사람 평가 쌍.** Soul 계보 태그(`SOUL`/`SOUL-DERIVED`/`SOUL-INSPIRED`)가 붙은 자산이 든 2AFC 쌍(E1-S, TP_S, X-LA, Hybrid-A 렌더 쌍)은 MRH-Aes, CamFM-CRM 학습에서 제외한다. 사람 라벨이어도 Soul 출력을 "개선에 쓰는" 것이기 때문이다.
6. **카나리아.** 주 1회 가짜 `SOUL` 태그 샘플을 학습 샤드 빌더와 **Feature-Tap Bus 링버퍼**에 주입한다. CI가 실패해야 정상이다.
7. **세계 단위.** Soul이 닿은 세계는 라이브러리·벤치·데모 콘텐츠로 쓸 수 있지만 래칫 데이터가 아니다. 엔진 트랙 세계만 플라이휠을 돌린다.

#### 3.7.4 승인 요청 3단계 (Plan A / A+ / S)

| 단계 | 무엇을 하나 | 필요한 것 | 우리 쪽 준비 | 거절되면 |
|---|---|---|---|---|
| **Plan A** (기본, D1부터) | 콘텐츠(스틸, 키 아트, 세계 속 이미지 자산, 원경 매트), 사람 경유 룩 측정과 개념 전사, 평가 앵커 | 승인 불필요(약관 안). 평가 앵커는 확인을 구한다(발송본은 용도 제한을 적지 않아 후속 문의에서 밝힌다) | 방화벽 CI, IP 점검, soul_vault, 룩 프리셋 라이브러리 | — |
| **Plan A+** (세계 피팅) | Soul 스틸을 관측 제약으로 쓰는 세계 피팅, θ_soul 자동 역추정, LUT 자동 피팅, MRH 판독 결과로 lift, 히어로 키프레임 광택의 세계 경유 POLISH, Hybrid-A+ 벤치 트랙 | 서면 승인. "장면 표현(세계 파라미터) 최적화는 AI 모델 학습이 아니다"라는 해석 확인 또는 명시적 허가 | 피팅 산출물은 `SOUL-DERIVED`로 격리, 래칫 제외 | Plan A 유지. Hybrid-A+ 트랙과 POLISH 세계 경유는 없다 |
| **Plan S** (소형 모델 학습) | Soul 출력으로 소형 모델 일부를 학습. **S1(비생성, ≤20M):** ① CamFM-CRM(12M, 구도 평가기), ② LookPrior-S(약 2M, 프롬프트 → θ_look 물리 파라미터), ③ MRH-Aes 헤드(≤2M, MRH 본체 동결). **S2(픽셀을 출력하지 않는 생성형, ≤25M, 생성형임을 밝히고 별도 협의):** ④ CamFM-Intent(20M, 카메라 궤적만 출력), ⑤ MassingFM(24M, 박스 레이아웃만 출력). MRH 기하 헤드는 요청하지 않는다(본체 24M이 상한을 넘고 가치가 낮다)(§5.7) | 서면 승인. 범위·모델 목록·데이터 상한을 명시한다. **이미지·비디오 생성기 학습은 요청하지 않는다** | 해금 모델별 계보 브랜치(`lineage:soul-s1`, `lineage:soul-s2`) | A 또는 A+ 유지. S1만 승인되면 S2는 계속 금지 |

- **발송본에서 물은 것(3개).** Soul Cinema의 MCP/API 제공 여부와 일정, 스타트업·연구 크레딧, 업로드 레퍼런스의 학습 제외(Enterprise 조건).
- **아직 묻지 않은 것(후속 문의: 티켓 #317768 또는 영업팀 연락).** MCP 크레딧 단가와 '생성 1회'의 계량 단위(스틸/배치), 모델 버전 고정과 단종 공지, 사람이 Soul 출력을 보고 측정한 룩 프리셋을 공개 벤치와 함께 공개해도 되는지(Q13), Soul을 이름으로 비교한 블라인드 결과(TP_S, X-LA)를 공개해도 되는지, 아트 디렉터가 창업자 계정을 쓰는 것이 계정 공유 조항에 걸리는지(별도 시트 여부), 평가 앵커의 용도(E1-S·TP_S, 보고 전용), Plan S의 S1/S2 구분, 약관 해석에 대한 서면 회신. 공동 마케팅은 발송본이 이미 제안했다(부록 B 발송 기록). 골자는 부록 B.
- **마일스톤.** D1–3 메일 발송(2026-10-10 발송, 티켓 #317768, 영업팀 전달 요청) → D3–7 도메인·회사 이메일·웹사이트 확보 후 Contact Sales 폼 재접수(티켓 #317768은 전달 불가로 종결) → D30 무회신이면 Plan A 확정 운용(어차피 기본값) → D45 A+ 회신 시 X-Soul 세계 경유와 Hybrid-A+ 트랙 착수 → D60 S 회신 시 CamFM-CRM 해금 실험(S1). S2는 별도 회신 뒤.

#### 3.7.5 Soul 호출 배분표 (90일)

**웹(사람 수동 경로, 약 2,900회).** 창업자 또는 아트 디렉터가 higgsfield.ai 웹에서 직접 뽑는다. 자동화하지 않는다. 아트 디렉터는 별도 시트를 쓰거나, 창업자 계정 공유가 허용되는지 후속 문의(티켓 #317768 또는 영업팀 연락)의 회신을 받은 뒤에만 창업자 계정을 쓴다(약관 미확인, 발송본에는 이 질문이 없었다). 호출 수는 이미지 장수 기준으로 계상한다(§1.10).

| 용도 | 호출 | 연결 |
|---|---|---|
| E1-S Soul 천장 (120 프롬프트 × 배치 4) | 480 | §3.4 |
| X-LA 미감 부검 (5쌍 재현 + 신규 25 프롬프트 × 배치 4 + 이끼 변형) | 140 | §3.8.6 |
| 룩 프리셋 라이브러리 (θ 수동 측정용 레퍼런스 60룩 × 4) | 240 | §7.6 |
| 데모 개념 아트 (6편 × 약 60) | 360 | §8.3 |
| 미니 벤치 TP_S 앵커 (80 프롬프트 × 4) | 320 | §8.4 |
| X-Soul 수동 편집 실험 | 160 | §11.2 |
| 키 아트·포스터·썸네일 (Plan A 콘텐츠) | 200 | §8.3 |
| IP 재생성·대체 | 100 | §8.3 |
| **예비 (약 30%)** | **900** | 원경 매트 등 콘텐츠 자산 포함 |
| **합계** | **2,900** | |

**MCP(자동 경로, 크레딧 약 1,600회분, 추정).** Soul Cinema가 MCP에 없으면 Soul 2.0으로 돌리고 버전 태그를 남긴다.

| 용도 | 호출 | 조건 |
|---|---|---|
| Hybrid-A+ 벤치 히어로 스틸 (160 세계 × 배치 4) | 640 | A+ 승인 시. 미승인이면 예비로 |
| X-Soul 자동 스윕 (MCP가 레퍼런스·편집을 지원할 때) | 200 | — |
| 데모 최종 광택 스틸 | 120 | — |
| 버전 회귀 감시 (고정 프롬프트 20 × 월 1회 × 3개월 × 배치 4) | 240 | Soul 업데이트 감지 |
| **예비** | **400** | — |
| **합계** | **1,600** | 약 1.6만 크레딧 ≈ $1,000 내외(미검증 단가) |

### 3.8 미감 부검 결과와 미감 연방제 [v2.1 신규]

#### 3.8.1 증거: 창업자 A/B 5판

같은 간단 프롬프트로 Soul Cinema와 Z-Image(엔진 교사)를 비교하고, 미감 10축마다 승패를 매겼다.

| 판 | 프롬프트 | 유형 | 축 | Soul 승 | Z 승 | 무 | Soul 승 중 물리 / 콘텐츠 | 관찰 |
|---|---|---|---|---|---|---|---|---|
| 1–3 | "astronaut riding a horse"류, "a traditional Korean house courtyard", "an old fisherman mending a net" | 해석형 / 기록형 / 기록형 | 30 | 27 | 0 | 3 (화면비·심도) | 21 / 6 | **Soul:** 약 2.35:1 시네마스코프, 망원 압축과 얕은 심도(배경 사구가 녹음), 해 질 녘 낮은 태양과 역광·림, 로우키 노출(피사체 살짝 언더, 하늘 보존), 제한된 팔레트(라벤더 회색 + 흙색)와 필름 톤, 미디엄샷·말 머리 크롭·정면 응시, 프레임 안의 프레임과 전경 레이어링(처마, 흐린 기둥), 낡고 먼지 묻은 우주복과 바이저 너머 수염 난 얼굴(서사). **Z:** 1:1, 한낮 평광, 하이키, 팬포커스, 고채도, 전신 측면 카탈로그 구도, 새 제품 같은 우주복, 불투명 바이저 = "스톡 사진" 룩. 현판·주련에 해독 불가 가짜 한자(Soul은 글자를 피함) |
| 4 | "a mossy forest floor" | 기록형 | 10 | 7 | **1** | 2 | 6 / 1 | **Z가 처음 이긴 축: 미술·콘텐츠 정확도.** 선명한 매크로 자연 사진, 식물학적으로 정확하고 풍부(솔잎, 고사리, 조약돌). **Z도 얕은 심도와 보케를 그렸다**(매크로에서는 Z 디폴트도 팬포커스가 아니다, §3.3 항목 6). **Soul:** 거의 단색 올리브, 로우키, 이끼 솜털에만 걸린 역광, 가장자리 빈티지 번짐·방사형 블러(달리 인 같은 움직임의 흔적), 그레인·비네팅, 지평선 없는 극접사라 이끼가 산맥처럼 보이는 스케일 착각. 창업자 평: "영화라고 해도 믿겠다" |
| 5 | 야크 (간단 프롬프트) | 해석형 | 10 | 7 | 0 | 3 | 5 / 2 | **가장 극단적인 격차.** Soul은 프롬프트를 **해석**했다: 거대한 신수(神獸) 스케일의 흰 야크가 작은 두 사람에게 고개를 숙이고, 보통 크기 야크 무리가 스케일 기준점이 되고, 뿔이 하늘을 후광처럼 감싸고, 털 가장자리에 역광 림, 완벽한 좌우 대칭과 V자 계곡 소실선, 편광 같은 짙은 파란 하늘, 먼 설산의 대기 원근. Z는 **기록**했다: 보통 크기 야크 정면 증명사진, 화면 안 태양과 플레어로 흐린 노출, 줄 선 관광객, 마을 건물 = 여행사 포스터 |
| **누적** | | | **50** | **41** | **1** | **8** | **32 (78%) / 9** | 창업자 단독·비블라인드 판정, 축 분류는 주관, n=5 |

#### 3.8.2 미감 10축과 소유권 (미감 연방제)

| 축 | 내용 | 소유자 | 우리 메커니즘 | θ_look 필드 |
|---|---|---|---|---|
| 1 화면비 | 2.35–2.39:1 | 물리(카메라) | 센서 크롭, 애너모픽 처방(타원 보케, 수평 플레어) | aspect, squeeze |
| 2 렌즈 원근 | 망원 압축, 광각 과장 | 물리 | 초점거리 + 카메라 거리(DP-0) | focal, distance |
| 3 심도 | 배경이 녹는 얕은 심도 | 물리 | $c(z)$ + Defocus Pyramid, 조리개 날 | f_stop, focus, blades |
| 4 시간대·광원 방향 | 낮은 태양, 역광, 림 | 물리(Gaffer) | Hosek–Wilkie 태양 고도·방위, L0 bake, 림 라이트 ≤8 | sun_elev, sun_az, rigs |
| 5 노출 | 로우키, 하늘 보존 | 물리(센서) | EV, 하이라이트 롤오프, full-well | ev, curve |
| 6 그레이딩·팔레트 | 제한 팔레트, 필름 톤 | 물리(룩) + Soul 취향 | log → 3D LUT 65³, Soul HEX → 기준색 | lut, palette |
| 7 필름 질감 | 그레인, 할레이션, 비네팅, 빈티지 수차, 움직임의 흔적 | 물리 | Newson 그레인, 할레이션, 게이트 위브, 빈티지 렌즈 처방, LH-split 셔터 적분 | stock, halation, vignette, lens_rx, shutter |
| 8 프레이밍 | 프레임 안의 프레임, 전경 레이어링, 타이트 크롭, 대칭·소실선 | 물리(DP-0). **공짜가 아니라 취향이 필요한 영역** | CamFM-CRM + PR + Wipe + **전경 레이어링 보상** + Toric. Soul 구도로 학습하는 것은 CamFM-CRM(구도 평가기)만 S1 해금 | framing |
| 9 미술·의상 생활감 | 낡음, 먼지, 사용 흔적 | Soul(개념) + 엔진 교사(재질) | Soul 개념 전사, 마모장(곡률, AO, 접촉 빈도), 엔진 교사 재질 | — |
| 10 인물·서사·개념 | 스케일 전환, 신화, 상호작용, 얼굴 | **Soul + 사람(감독)** | 개념 아트(A: 전사, A+: 관측 제약), MassingFM 스케일 레시피, Soul ID(개념 아트에 한해) | — |
| 부: 재질 정확도 | 식물학적·재료학적 디테일 | 엔진 교사(측량) | 차트 뷰, Swatch, TexelFM | — |
| 부: 대기·하늘 | 대기 원근, 편광된 짙은 하늘 | 물리 | 볼류메트릭 산란 β, CPL 필터 | haze_beta, cpl_angle |
| 부: 미세구조 광택 | 역광에 빛나는 솜털, 털 림 | 물리 | STRAND/SGGX 미세구조 + Gaffer 림 | rigs |

**원칙.** 축 1–8은 우리 물리 카메라·조명(Gaffer, 태양 고도)·센서/필름·DP-0가 소유한다. 축 9–10은 Soul(개념 아트, Soul ID)과 감독이 소유한다. 오픈 엔진 교사는 물리 조명 아래 정면 차트 뷰의 재질과 표면만 판정하므로 "스톡 조명 룩"이 세계로 덜 새어 들어온다. 이 마지막 문장은 가설이고 X-LA H-LA5로 검증한다.

#### 3.8.3 기록형과 해석형, 축 세기의 맹점, Soul의 두 가치

- **축 세기의 맹점.** 야크 판은 축 개수로는 Soul 7승(물리 5, 콘텐츠 2)이라 누적 물리 몫 78%가 유지된다. 그러나 **영향력 기준으로는 이번 격차의 대부분이 개념 해석(스케일 전환, 신화적 서사, 상호작용)이라는 콘텐츠 축**이었다. 동일 가중으로 축을 세면 이것을 놓친다(CT40).
- **프롬프트 두 유형.**
  - **기록형**(이끼, 어부, 한옥): 격차가 주로 카메라·조명에 있다. 우리 물리 스택이 대부분 메울 수 있어야 한다.
  - **해석형**(야크, 우주인): 격차가 주로 개념에 있다. 물리로는 대체할 수 없다.
- **Soul의 가치는 둘이다.** (a) 물리로 복제할 수 있는 **촬영 취향**, (b) 물리로 대체할 수 없는 **개념 아트 디렉션**. **(b)가 Soul을 버릴 수 없는 진짜 이유다.** (a)는 룩 프리셋으로 사람 손을 거쳐 옮길 수 있다.

#### 3.8.4 A/B에서 나온 설계 반영 14개

| # | 반영 | 위치 |
|---|---|---|
| 1 | 역할 분담 반전: 엔진 교사 = 측량 기사, Soul = 촬영감독·미술감독. 측량 모드 태그 | §3.1, §3.3 항목 6 |
| 2 | **Soul 카메라 역추정 = 룩 프리셋.** 관측 제약으로 쓸 때 그레인·심도·그레이딩·비네팅은 우리 카메라 θ로 설명해 세계 밖으로 빼고(DTC·렌즈 식별성 원리), 추정한 θ_soul을 프리셋으로 저장해 모든 각도에서 재사용한다. 수동 측정은 안전(A), 자동 피팅은 회색(A+) | §3.7.2 (b), §7.6 |
| 3 | DP-0 구도 취향: 프레임 안의 프레임, 전경 레이어링(처마, 흐린 기둥), 타이트 크롭은 v1의 Wipe·PR 보상과 일치한다. **전경 레이어링 보상을 명시적으로 강화**한다. Soul 출력으로 CamFM-CRM(구도 평가기)을 학습하는 것은 방화벽 밖이고 S1 해금 대상이다. CamFM-Intent(궤적 생성 prior)는 S2 | §7.8, §5.7 |
| 4 | 텍스트: 엔진 교사는 가짜 한자를 그렸고 Soul은 글자를 피했다. **글자는 SYMBOL이 소유한다**는 원칙 재확인 | §7.2 PSH |
| 5 | 승인 요청 3단계(A/A+/S), Soul Cinema MCP 문의 | §3.7.4, 부록 B |
| 6 | "영화 같음"의 상당 부분은 스틸 속 **시간의 흔적**(카메라 이동 모션블러)과 **렌즈 성격**(빈티지 수차)이다. 우리 셔터 적분과 렌즈 처방 스택이 물리적으로 매 프레임 만든다. 렌즈 라이브러리에 빈티지 처방(필드 커버처, 스월, 주변부 번짐) 추가 | §7.6 |
| 7 | 역광에 빛나는 솜털 = STRAND/SGGX 미세구조 + Gaffer 림 라이트 | §7.3, §7.8 |
| 8 | DP-0 연출 옵션 **역-미니어처**(작은 것을 거대한 풍경처럼): 극접사 + 지평선 제거 + 얕은 심도 | §7.8 |
| 9 | 미감 부검 첫 대상은 이끼 페어(콘텐츠가 가장 단순해 카메라 몫을 가장 깨끗하게 분리, 가난선 아래 TexelFM 영역) | §3.8.6, §11.1 |
| 10 | 미감 부검을 3분해로: 카메라·조명 몫 / 연출 몫 / 개념 몫. 축 세기 대신 블라인드 격차 닫힘 | §3.8.6 |
| 11 | Soul 히어로 스틸 = 세계의 개념 아트. 엔진 교사에 약한 노이즈로 통과시켜 독심 헤드가 깊이·재질을 읽으면 Soul 내부 접근 없이 기반암을 얻는다(추론 경로, 세계 피팅은 A+). 뒷면은 BRP·엔진 교사가 개념과 일관되게 채운다 | §4.2, §4.3 |
| 12 | 스케일 연출 레시피 **거대화**: 메트릭 세계 + 보통 크기 기준물 배치 + 로우앵글 광각 + 대기 원근(볼류메트릭 산란). 이끼의 역-미니어처와 대칭 | §7.8 |
| 13 | 렌즈 라이브러리에 원편광(CPL) 필터: 태양 각도에 따라 하늘을 물리적으로 짙게 | §7.6 |
| 14 | Soul 출력은 기존 IP와 닮을 수 있다. 상업 데모 전 IP 점검 단계 | §8.3, §13 R32 |

#### 3.8.5 BT 회계: 연방제가 SP를 만드는 방식

Bradley–Terry에서 logit이 더해진다고 보면(CT2와 같은 가정), 앵커 스틸 $a$에 대해

$$
\mathrm{logit}(\mathrm{SP}_{ours})=\mathrm{logit}(\mathrm{SP}_a)+\mathrm{logit}(\mathrm{TP}_a),\qquad \mathrm{SP}_{ours}\ge0.5\iff \mathrm{TP}_a\ge1-\mathrm{SP}_a.
$$

- **엔진 앵커:** SP_E = 55%면 TP_E ≥ 45%. v1의 TP 목표와 정확히 같다.
- **Soul 앵커:** SP_S는 E1-S에서 측정하고, 계획 가정은 **65–75%**다(E1 (ii)와 같은 값). SP_S 65%면 **TP_S\* = 35%**, 75%면 **25%**로 충분하다. 75%라면 우리 렌더가 Soul 스틸에 3:1로 져도 비디오 프레임과는 동률이다. 이것이 하이브리드가 사는 여유(headroom)다.
- **연방 분해.** TP_S가 어디서 나오는지는 X-LA가 잰다. 카메라·조명 몫은 θ_look과 물리 스택이, 연출 몫은 DP-0가, 개념 몫은 Soul 개념 아트(A: 사람 전사, A+: 관측 제약)가 만든다.
- **정직한 한계.** BT 가산성은 근사다(비이행 선호, 맥락 효과). 스틸 대 비디오 프레임 비교는 단일 프레임 2AFC다. 그래서 TP_S는 진단 지표이고, **승리 조건(§8.1)은 하나도 바꾸지 않는다.** TP_S\*는 측정값에서 도출한 진단 기준선일 뿐 게이트도 목표도 아니다(Q6).

#### 3.8.6 미감 부검 3분해 실험 X-LA (Look Autopsy) [신규 프로토콜]

- **질문.** Soul과 엔진 교사 사이의 미감 격차 중 얼마가 카메라·조명 몫이고, 얼마가 연출 몫이고, 얼마가 개념 몫인가? "카메라만 바꿨을 때 격차가 얼마나 닫히나"를 블라인드로 직접 잰다.
- **arm.**

| arm | 내용 | Plan |
|---|---|---|
| S | Soul 원본 스틸 | A |
| Z | 엔진 교사 원본 스틸 | A |
| Z+ | 엔진 교사 + 시네마토그래피 프롬프트(보조: 프롬프트만으로 격차가 얼마나 줄어드나) | A |
| C (카메라·조명) | 엔진 교사 재질로 지은 세계 + **사람이 측정한 θ_soul**(화면비, 초점거리, f-stop, 초점, 태양 고도·방위, 림, EV, LUT, 그레인, 비네팅, 렌즈 성격)로 우리 스택이 렌더 | A |
| D (+ 연출) | C + Soul에 맞춘 구도·전경 레이어링·스케일 배치(사람이 DP-0를 수동 조작) | A |
| K (+ 개념) | D + Soul 개념을 사람이 WorldScript로 전사(소품, 스케일 관계, 서사 소품). A+ 승인 후에는 K+ = Soul 스틸 관측 제약 | A / A+ |

- **FD arm (ii)와의 차이.** FD가 요청한 arm (ii)(Soul 스틸을 관측 제약으로 쓴 세계)는 약관 회색지대라 K+로 미뤘다. D45에 A+ 회신이 오면 해석형 2 프롬프트(야크, 우주인)에 K+를 추가하고, 그 전까지는 K(사람 전사)로 대리한다.
- **arm C·D·K의 렌더와 판정 쌍은 `SOUL-INSPIRED`를 상속한다**(§3.7.3 3). 학습 데이터로 쓰지 않는다.
- **인물·동물 에셋 [v2.1].** P1은 인물·동물 세계를 2c로 미뤘으므로 X-LA 세계는 정지 에셋으로 짓는다. 인물(어부, 우주인, 신수 앞 두 사람)은 MakeHuman(CC0) 정적 포즈에 얼굴 비노출(뒷모습, 원거리, 바이저 반사), 동물(말, 야크)은 CC0 메시 + STRAND/SGGX 털로 짓는다. 에셋을 구하지 못한 프롬프트는 arm C·D만 돌리고 결과표에 표기한다.

- **지표.** 각 arm을 S와, 그리고 경쟁 멈춘 세계 프레임과 블라인드 2AFC로 비교한다. S 대비 승률을 P라 하면 격차 닫힘 비율은
  $$g_{cam}=\frac{P_C-P_Z}{0.5-P_Z},\quad g_{dir}=\frac{P_D-P_C}{0.5-P_Z},\quad g_{con}=\frac{P_K-P_D}{0.5-P_Z},\quad g_{res}=1-g_{cam}-g_{dir}-g_{con}.$$
- **프롬프트.** ① 이끼(X-LA0, D10–20) → ② 어부, 한옥(기록형) → ③ 야크, 우주인(해석형) → ④ 신규 25개(기록형 15, 해석형 10). 프롬프트는 사전 등록한다.
- **반증 가능한 예측.**
  - **H-LA1:** 기록형에서 g_cam ≥ 0.5.
  - **H-LA2:** 해석형에서 g_con ≥ 0.4이고 g_cam ≤ 0.4.
  - **H-LA3 (이끼, 도전):** arm C 또는 D가 S 대비 ≥50%. 엔진 교사의 정확한 식물 디테일 + Soul 룩 프리셋이 Soul 원본을 넘는다.
  - **H-LA4:** Z+는 격차의 30% 이하만 닫는다. 프롬프트로 부른 촬영은 물리 카메라를 대신하지 못한다.
  - **H-LA5:** 차트 뷰(물리 조명 아래 정면)에서 뽑은 재질은 hero 스틸에서 뽑은 재질보다 구운 조명 지수(BBI)가 낮다. "스톡 조명 룩"이 덜 샌다.
- **비용.** X-LA0 약 6 T4-h + 웹 20회. 본실험 약 70 T4-h(30 프롬프트 × arm 3개 세계, 이끼류는 작은 facet 세계) + 웹 120회 + 사람 약 3,000쌍 $2–4K.
- **약관.** 모든 θ는 사람이 측정한다(Plan A). 자동 피팅 변형(C+)과 K+는 A+ 이후에만 돌린다. Soul을 이름으로 비교한 결과의 공개 여부는 후속 문의(티켓 #317768 또는 영업팀 연락)의 회신을 따른다(발송본에는 이 질문이 없었다). 회신 전에는 내부 보고서로만 쓴다.
- **실패하면 (목표 유지, 업그레이드).**
  - 기록형에서 g_cam < 0.3이면 물리 스택에 빠진 것이 있다. 축 단위 단일 교체 ablation(화면비만, 조명만, LUT만…)으로 범인을 찾고 렌즈 라이브러리, 그레이딩, Gaffer를 보강한다.
  - 기록형에서도 개념 몫이 크면 directed 모드와 아트 디렉션 인력에 투자하고 MassingFM 스케일 레시피를 앞당긴다.
  - 해석형에서 개념 몫 g_con(K − D)이 크면 개념 전사의 가치가 수치로 드러난다. 후속 문의에는 승인 전에 얻을 수 있는 숫자만 붙인다. E3 (c)의 Soul풍 대리 판독 결과(Soul 픽셀 0)와 X-LA의 g_con이다. K 대 K+ 비교는 A+ 승인 뒤에만 나오므로 S 단계 협의 자료로 쓴다.

---

## 4. Phase 1 파이프라인 on 2×T4

### 4.1 핵심 원리 일곱

**P1. 가난선, 판정 해상도 분할, 관할 이양 래칫 (Spectral Poverty Line + Judge-Resolution Split + Devolution Ratchet)** [신규]

- **관찰.** 자연 이미지는 1/f² 스펙트럼을 따른다. 옥타브당 에너지는 거의 같은데 표본 수는 옥타브마다 4배 늘어난다. 반면 의미(프롬프트와 장거리 레이아웃에 대한 의존)는 낮은 옥타브에 몰려 있다. 대역 b를 결정하는 데 필요한 문맥은 텍셀 단위로 보면 상수다. 그래서 수용장이 고정된 수십M 연산자로 충분하다.
- **판정 해상도 분할 [TF(u) 복원].** v1 §7.1에 따라 SP는 1080p로 정규화해 판정한다. Ray-Bundle Demand를 1080p와 4K로 두 번 계산하고, ℓ\*₁₀₈₀까지는 교사 관할, ℓ\*₄ₖ − ℓ\*₁₀₈₀(8K면 +2옥타브)은 학생 관할로 둔다. 거리 3 m, 화각 60°에서 1080p 픽셀 footprint는 1.8 mm, 4K는 0.9 mm다. **텍셀의 75%를 차지하는 최미세 레벨이 교사 장부에서 빠지고, 납품 해상도를 8K로 올려도 교사 비용은 변하지 않는다.**
- **가난선 정의.** 재질 클래스 m마다 다음을 정한다.
  $$o_s(m)=\min\Big\{o:\ \tfrac{\mathrm{DR}_{\text{teacher}}(\text{TexelFM}(o,m))}{\mathrm{DR}_{\text{teacher}}(\text{Teacher}(o,m))}\le1.1\ \wedge\ \Pr_{\text{2AFC, 50쌍 감사}}\le0.55\Big\}$$
  1차 판정은 교사-판사 자동 2AFC로 하고, 사람은 표본 감사만 한다. 초기값 o_s = max(1080p 판정 경계 레벨, E2 측정값)이다.
- **3구역 관할** (셀 단위 Router, RR의 Federal Router):

| 구역 | 대역 | 소유자 | 교사 비용 |
|---|---|---|---|
| 교사 소유 | o_s(m)+2 이상 (+ 모든 의미 셀) | 교사 OTBI 소유권, K=2+1 | 뷰당 3 NFE |
| **Spec-commit** [신규] | o_s(m)+1 | TexelFM이 초안을 쓰고 교사가 t_end(b)에서 1-NFE로 E2 고정점(DR_rel ≤ 1.2 ∧ seam)을 판정한다. 가장 정면인 2뷰(κ_v = 2)로 검증하고, 거부된 블록만 소유권(K=3)으로 보낸다 | 블록당 κ_v/κ + (1−a)·K. a = 0.65면 1.55 NFE(기존 3의 절반) |
| 학생 native | o_s(m) 이하 | TexelFM, 비트 결정론 | 0 (5% 감사 1 NFE) |

- **Router 규칙:**
  - 얼굴, 손, 음식, 인쇄물, 간판 주변, 각인 있는 소품, CURVE 인접 같은 **의미 셀**과 구조 점수(국소 방향 결맞음, 에지 밀도)가 높은 셀은 이양하지 않는다.
  - 글자는 처음부터 SYMBOL이 맡는다.
  - 기본값은 콘크리트·목재·직물·석재가 2–4 mm, 얼굴·음식·인쇄물이 0.5–1 mm다. 교사 몫은 25–40%로 예상한다.
- **세계별 NFE 장부 [TF(u) 복원].** 모든 세계에 대해 기록한다.
  $$\mathrm{NFE_{eq}}=\sum_{l}\Big\lceil\tfrac{A_l}{P_v\,\eta}\Big\rceil\,\kappa\,R_l\,\big[\tfrac{\kappa_v}{\kappa}+(1-a_l)K\big]\,c_{res}$$
  A_l은 레벨 l의 수요 텍셀 수, P_v는 뷰 픽셀 수, η는 소유 픽셀 비율(band-iso에서 약 0.25), κ는 블록당 겹치는 뷰 수, R_l은 Jacobi 라운드 수, c_res는 해상도별 FLOPs 계수다.
- **래칫:**
  1. 라이브러리 세계(엔진 트랙)의 교사 판정 대역으로 TexelFM gen(k+1)을 학습한다.
  2. 판정을 통과한 재질은 o_s를 한 옥타브 올린다. 그 재질의 교사 비용이 1/4이 된다.
  3. 같은 시간에 더 많은 세계를 풀고, 데이터가 늘어나고, 1로 돌아간다.
  4. 의미 바닥에서 멈춘다.
- **세대별 비용.** 계획은 G0 약 700 TFU/세계 → G3 ≤520. TF(u)의 G3 약 360 NFE-eq(TF 기준선 G0 2,900 → G2 760 → G3 360)를 도전 목표로 둔다.
- **KPI:** TFU/세계, TJF(teacher jurisdiction fraction), Spec 수락률 a.
- **정직한 폴백 비용:** 가난선을 2 mm에서 1 mm로 내리면 방 하나에 교사 fwd가 약 570회 늘어난다. INT8로 약 25분이다.

**P2. Grid-Locked OTBI 차트 + TABP (VAE 0, Adam 0, ρ=1)** [조합 + 신규 부품]

1. **세계 축 정렬 차트.** 벽은 u = 수평, v = 중력 반대다. 바닥은 방의 주축에 맞춘다. 늘어남은 2^{1/4} 이하다. 평면 차트끼리 회전이 0이므로 latent 격자가 정렬된다.
2. **격자 스냅 OTBI (Owner-Tiled Band-Iso).** 레벨 l의 뷰를 차트 latent 격자(f8)에 정수 오프셋, 회전 0으로 스냅하고 **단독소유 타일 + 25% apron**으로 덮는다. 뷰 수는 [RR(u) 복원]
   $$N_l\approx\frac{1.3\,A_l}{0.77\,(1024\,\tau_l)^2}$$
   이고(τ_l = h_l/8, 1024τ_l은 1024² 뷰의 폭, l=14에서 0.5 m), 볼록하지 않은 소품만 4–6뷰를 쓴다. "정사영 렌더 → Enc"가 **latent 텍스처를 잘라 오는 메모리 복사**로 바뀐다(TF의 Chart-Direct Sync와 같은 효과).
3. **WNF가 정확해진다.** latent 토큰 하나가 차트 셀 하나다. apron이 겹치는 곳에서는 ρ=1이다.
4. **닫힌형 sync (TABP, Texel-Aligned Back-Projection) [RR(u) 복원].** band-iso 뷰는 화소 footprint가 텍셀 피치의 ±¼ 옥타브 안이고 θ≤33°라 역투영이 거의 affine 리샘플이다. 그래서 대역 결정을 경사하강 대신 닫힌형으로 한다.
   $$P_l(t)=\mathrm{Up}(P_{l-1})(t)+\sum_v\tilde m_{v,l}(t)\,\Pi_{b_l}\!\Big[g_v\big(\hat x_v(\pi_v(t))\big)-\mathrm{Up}(P_{l-1})(t)\Big]$$
   $\Pi_{b_l}$은 파장 $[2\tau_l,4\tau_l]$의 Laplacian 대역, $\tilde m$은 결정 중에는 $\mathrm{Blur}_{\lambda_b}(\mathrm{softmax}_v\tau_sa_v)$이고 commit 때는 graph cut hard + multiband, $g_v$는 저대역 최소제곱 3×4 색 gauge, $\pi_v$는 G-buffer 깊이로 가시성을 검사하는 투영이다. 법선과 높이는 같은 forward의 MRH 출력을 같은 마스크로 융합하고 높이는 Frankot–Chellappa로 적분한다. latent 공간에서는 Z_tex[l] ← Σ_v own_v · x̂0_v를 **int64 고정소수점으로 (텍셀 id, 뷰 id) 정렬 후 segment-reduce**하므로 순서와 무관하게 비트 단위로 같다. Adam 상태 0, 누산기 0.5 GB 이하, 뷰당 5–10 ms. 대역 밖 저주파는 부모로 고정되므로 색·조명 드리프트가 구조적으로 0이다.
5. **Fold 뷰 [신규].** 수직 모서리를 공유하는 벽 차트를 하나의 평면으로 펼친다(평면 다면체의 정확한 등거리 전개). 코너를 뷰 한가운데 둔다. 수평 모서리(바닥과 벽)는 seam-owner 투시 뷰로 처리하고 폭을 λ_b로 제한한다.
6. **Context Ring [신규].** 1024² 뷰의 바깥 64 px는 인접 차트를 **더 거친 commit 레벨**에서 펼쳐 LatentUp으로 채운다. 이 영역은 소유하지 않는다. 그 결과 교사가 "몰딩 바로 위의 벽" 같은 위치 문맥을 본다.
7. **효도 투영 [신규].** 스텝마다 Z_l ← Z_l + Up(Z_{l−1}^{commit} − Down(Z_l))를 적용한다(LatentDown/Up-S). commit할 때는 RGB Laplacian 저대역 교체(정확)로 마무리한다.
8. **commit.** 차트 graph cut(CPU maxflow) + multiband 후, 교사 VAE 디코드를 텍셀당 한 번만 한다(방 하나에 1024² 타일 약 50장).
9. **적용 범위.** 평면에 가까운 면(약 80%)이 이 경로로 간다. 곡률이 큰 블록(≤20%)은 투시 band-iso + TSQ 미분 피팅 + TAE-T(노이즈 마스킹 규칙)로 간다. 잎, 털, 풀은 SMF와 STRAND/SGGX로 간다.
10. **판정 해상도는 1024²가 최적이다 [CB(o) §4.5 복원].** 픽셀당 FLOPs(문맥 apron 포함)는 512² 9.4×10⁷, **1024² 7.4×10⁷**, 2048² 8.9×10⁷이다. 512²는 교사 분포 밖(OOD)이고 문맥이 부족하며, 2048²는 어텐션 비중이 39%로 오르고 역시 OOD다. **비용 손잡이는 해상도가 아니라 뷰 수와 관할(가난선)이다.** 그래서 512² 판정(S1 hero 1차, S2d, S7 감사, S8 Gaffer)은 E0 (d)(512² INT8 판정이 1024² FP16과 Spearman ≥0.9)를 통과한 **순위 매기기에만** 쓰고, commit과 소유 판정은 1024²(또는 768² 의미 패스)로 한다.

**P3. Picture-First, Program-Backed Bedrock** [조합]

- hero still은 관측 제약(DPS)이고 캔버스가 아니다. 텍스트 모드 hero 1차 탐색은 엔진 교사의 측량 모드로 한다.
- 레이아웃 대역의 소유자는 **frustum 안은 hero, 밖은 MassingFM**으로 정확히 하나다(Janus 방지).
- **semantic metric prior [RR(u) 복원].** MRH의 FOV 헤드와 클래스별 크기 사전(약 500 클래스: 문 2.0–2.1 m, 좌석 높이 0.42–0.48 m 등)으로 전역 스케일을 최소제곱으로 푼다(고전). STRATA 레벨과 DOF 수식이 미터를 요구하기 때문이다.
- 절차 에셋은 L1의 SDEdit 초기값일 뿐이다. L1 형태는 교사, MRH, G-Adapter가 lockstep으로 정한다.
- directed 모드에서는 DSL 프로그램이 L0를 소유한다.
- **[v2.1] Soul 개념 아트.** Plan A에서는 아트 디렉터가 개념(스케일 관계, 소품, 서사 소품, 구도)을 DSL로 전사한다. A+에서는 Soul 스틸이 hero와 같은 지위의 관측 제약이 된다. 이때 θ_soul로 구운 광학 효과를 먼저 설명해 빼고(DTC 원리), frustum 안 레이아웃 대역만 소유한다.

**P4. Corman–Shannon Demand와 연산을 아는 촬영** [신규] (§7.8)

카메라가 기록할 수 없는 정보는 생성하지 않는다. DP-0 보상에 −λ·TFU(Γ) 항이 들어간다. 그래서 보케, 밤, 연기, 역광, 모션블러는 연산 할인쿠폰이다. **시네마틱할수록 싸다.** v2.1의 A/B는 같은 사실을 반대쪽에서 확인했다. Soul이 이긴 축(얕은 심도, 로우키, 역광, 그레인, 움직임의 흔적)은 대부분 수요를 줄이는 축이다.

**P5. Teacher Exhaust Ledger + Feature-Tap Bus (기생 학습)** [신규]

- **키:** blake3(블록 Merkle ‖ 뷰 정의 ‖ σ 점 ‖ WNF seed ‖ teacher id ‖ 양자화 모드 ‖ TRT 엔진 해시 ‖ caption hash ‖ **soul 비트**)
- **값:** int8 x̂0 latent(256 KB), DR 점수, gauge 통계, (20% 표본) PCA-128 int8 탭
- **쓰임:**
  1. 세션 재개(Kaggle 12 h)
  2. Merkle 델타 솔브
  3. A/B 실험에서 거친 레벨 공유
  4. MRH, T-LPIPS-S, HeroProxy-S·ChronoProxy-S의 학습 데이터
  5. TexelFM 래칫 데이터
  6. 벤치 의도 공유
- **"교사 호출 한 번을 여섯 군데에 쓴다."** 개발 반복에서 적중률은 40–70%로 추정한다. 그중 학습 데이터는 4·5 두 군데다.
- **soul 비트 정의:** soul 비트 = (`prov.soul` ∈ {inspired, derived, authored}). `prov.soul = none`일 때만 0이다.
- **soul 비트 엔트리의 쓰임:** 1, 2, 6에만 쓴다. 3(A/B 실험 간 거친 레벨 공유)은 **같은 prov 등급 세계끼리만** 허용한다. 엔진 트랙 세계가 soul 엔트리를 재사용하면 그 세계를 `SOUL-DERIVED`로 재태깅하고 래칫에서 뺀다. 4–5(학습)에는 쓰지 않는다.
- **Feature-Tap Bus는 soul=1 호출에서 꺼진다.** 그런 호출은 별도 프로세스에서 돌고 링버퍼에 쓰지 않는다(§3.7.3 4).

**P6. 두 반구와 주야 시분할** [운용]

- **낮 10 h:** GPU0는 대화형 실험과 교사 공장, GPU1은 학습.
- **밤 14 h:** 두 장으로 라이브러리 세계를 솔브하거나 장기 학습을 돌린다.
- 모든 학습은 20분마다 체크포인트를 남기고 선점 가능하게 만든다.
- 잡 스케줄러가 TFU 예산을 집행한다.
- **"세트는 밤에 짓고 아침에 찍는다."** 데일리스 리듬 그대로다. v2.1에서는 아트 디렉터가 낮에 Soul 개념 아트와 룩 측정을 하고, 밤에 그 세계를 짓는다.

**P7. 미감 연방제와 Soul 레인** [v2.1 신규]

- **소유권.** 축 1–8은 θ_look(Gaffer, 렌즈·센서·필름)과 DP-0, 축 9–10은 Soul 개념 아트와 감독, 재질·기하는 엔진 교사(§3.8.2).
- **룩 프리셋 라이브러리.** 아트 디렉터가 Soul 레퍼런스에서 수동 측정한 θ_look 60여 개를 이름과 설명과 함께 보관한다(`SOUL-INSPIRED`). 텍스트 모드에서는 CaptionComposer가 규칙과 TE 최근접 검색으로 고른다. 파라미터 0이고 학습이 없으므로 Plan A에서 자동 파이프라인에 들어갈 수 있다. 벤치용 라이브러리는 4주차에 동결한다(CT38).
- **Soul 레인의 단계 게이트.** A: 사람 경유와 콘텐츠만. A+: 관측 제약, 자동 역추정, POLISH 세계 경유. S: §5.7의 해금 순서.
- **Soul은 판사가 아니다.** 어떤 **모델**의 학습 손실·보상, SMC 가중치, 후보 선택 점수에도 Soul 출력이 들어가지 않는다. 예외는 둘뿐이다. (i) 세계 W 자체의 피팅(A+)에서 Soul 스틸이 관측항이 되는 경우, (ii) θ_soul이 θ_look과 Gaffer CEM의 초기값이 되는 경우. 이 산출물은 `SOUL-DERIVED`(자동) 또는 `SOUL-INSPIRED`(수동)로 격리된다. 평가 앵커는 사람 보고서로만 간다(Q6).
- **2K 상한과 대역.** Soul이 소유할 수 있는 대역은 최종 프레이밍에서 Soul 스틸의 Nyquist 이하뿐이다. 그 위는 엔진 교사와 TexelFM이 소유한다. 원경처럼 수요가 낮은 곳은 Corman–Shannon이 2K로 충분하다고 판정할 수 있다.

### 4.2 단계표 (방 6×5×3 m, 표면 200 m², 4K, 15–30초 경로, Directed, Mode A INT8, gen0)

| 단계 | 입력 → 출력 | 모델·알고리즘 (분류) | 장치 | TFU | T4-분 | Soul (경로, Plan) |
|---|---|---|---|---|---|---|
| **S−1 Concept·Look** [v2.1] | y → 개념 아트 1–4장, θ_look, WorldScript 초안 | Soul 웹 생성 (a′), 아트 디렉터 측정·전사 (사람), 룩 프리셋 라이브러리 검색 (b) | 클라우드 + 사람 | 0 | 0 (사람 30–90분) | 웹 8–40회, A |
| S0 인코딩·캡션 | y → E(TE 캐시), prompt_tree@scale | Qwen3-4B TE (a), CaptionComposer (b) | GPU0 30 s 또는 CPU | 0 | 0.5 | — |
| S1 Hero | E → hero 1–4장 + MRH 판독(깊이·법선·FOV·Aes) | 엔진 교사 측량 모드 64@512² NFE3 → 16@512² NFE8 → 4@1024² SDEdit4 (a, W4A4), MRH-Aes ⊕ Teacher-DCA(상위 16) (d), Glyph-S 벌점 (d), Corman 비용 (b). directed는 사람 best-of-4 | 두 장 | 112 (탐색 96 + DCA 16) | 4.9 | — |
| **S1′ Soul 판독** [v2.1] | Soul 스틸 → θ_soul 설명 제거 → MRH 판독 → DPS 관측 제약 | 엔진 교사 1–2 NFE @σ 스냅 0.2–0.3, 2K → 1024² 다운스케일 (a) + MRH (d). Exhaust soul=1, **Tap Bus 끈 별도 프로세스** | GPU0 | 1–2 | 0.1 | MCP 4회(벤치) 또는 웹, **A+** |
| S2a Lift + Massing | hero(, Soul 판독), E → shell·slot·광원·matter, 점유 32³ (+BRP 64³) | MRH (d), MassingFM (d), 크기 사전 + FOV 헤드 (b), BRP (d, 확장) | GPU1 | 0 | <1 | — |
| S2b 초기화 + 조명 | P → TSQ 코스 세계, 세계축 차트, Fold 그래프, L0 조명 | 문법·CC0 에셋 (b), xatlas (b), OptiX/Warp 경로추적 bake (b), hero 조명 역적합 (c), θ_look 태양·대기 (b) | CPU + GPU1 | 0 | 4 (Warp 8–15) | — |
| S2c 의미 패스 | W0 → L1 형태와 코스 외관 확정 | 교사 48 cine 뷰 @768² × K3 lockstep (a) + G-Adapter (d) + MRH (d) + TSQ 미분 피팅 4라운드 + DTC (c) | 두 장 | 81 | 3.5 + 피팅 12 | — |
| S2d 레이아웃 거부권 | W → 수정된 P | CMA-ES 8 gen × 16 cand @512², 에너지 = ‖depth_MRH − depth(W)‖ + slot 충돌 (c) | 두 장 | 32 | 1.4 | — |
| S2e Swatch Bank | 재질 목록 → 재질 m × 레벨 l 견본(512², intrinsics) | 교사 band-iso 정사영 4 NFE, 측량 모드 (a) + MRH (d). (seed, m, l, y)의 순수 함수 | GPU0 | 30 | 1.3 | — |
| S3 Demand | CamFM prior + 렌즈 계획 → {(l, block)} + Router 맵 (1080p/4K 이중) | Ray-Bundle + Corman–Shannon (b), CamFM-Intent (d) | CPU/GPU1 | 0 | <0.5 | — |
| S4 재질 패스 | W, D, S → commit된 Z_tex (교사 관할 레벨, 기본 4 mm·2 mm) | Grid-Locked OTBI + TABP + Fold + Context Ring (b), 교사 INT8 K=2+1 (a), LatentUp/Down (d), Spec-commit (d+a, 래칫 후) | 두 장 | **300 → 105 (G3)** | **13 → 4.5** | — |
| S4′ commit | Z_tex → RGB·intrinsics BC 페이지 | graph cut (CPU) + multiband + VAE 디코드 1회 + Laplacian 저대역 교체 (b) | CPU + GPU1 | 0 | 1–2 (FP32 VAE 4–6) | — |
| S5 미세 패스 | 부모 페이지 → l > L_T 페이지 (2–3×10⁸ 텍셀) | TexelFM native 2–3 step (d), WNF, 마모장 (b) | GPU1 | 0 | 1.5–3 | — |
| S6 Hero FX (선택) | 지시 → DIEL/VOL/파편 정지 상태 | Warp CFL 적응 FLIP/MPM + Voronoi (b) + CEM 64회 (c) | GPU1 + CPU | 0–50 | 30–240 | — |
| S7 Audit | W → sup DR_rel ≤ 1.2 | CMA-ES 적대 카메라 @512², 열린 대역만 재판정, CoW (c). 5% 무작위 full-ownership 감사 | 두 장 | 110 | 4.8 + 3 | — |
| S8 Shoot planning | W, 의도, θ_look → Γ, 조명 리그, 렌즈 계획 | CamFM ×3 (d), 1 kHz 리그 (b), CamFM-CRM 1,000 후보(해석 통계) → 360p 프리뷰 50 (c), Gaffer CEM 200 @512²(θ_look 초기값) + MRH-Aes (c+a), 전경 레이어링·스케일 레시피 보상, Corman λ·TFU | GPU1 | 50 | 5–10 | — |
| **S8.5 세계 경유 POLISH (선택)** [v2.1] | W, Γ 키프레임 렌더 → Soul 광택 → W′(commit) | Soul 구조 보존 편집 (a′) → gauge 제거 → CoW로 다시 연 1개 Jacobi 라운드에서 엔진 교사 뷰와 동시에 소유 마스크 ∧ 2K Nyquist 이하 대역만 소유 → 엔진 교사 재판정 (a) → Commit(prov=authored) | 클라우드 + GPU0 | 0–20 | 0–2 | MCP 2–8회, **A+** |
| S9–10 렌더 + 렌즈/센서 | W, Γ → HDR + AOV → ProRes/EXR | TSQ 래스터 + RT + dual-cut + 공분산 + Defocus Pyramid + LH-split + 렌즈(빈티지·CPL)·센서·필름(θ_look) (b) | 두 장 | 0 | 4K24 15초: 1.7–3.8 | — |
| **S11 스틸 광택 (선택)** [v2.1] | 최종 렌더의 히어로 키프레임 → 광택 스틸(키 아트, 포스터, 썸네일) | Soul 구조 보존 편집 (a′). 세계 밖 산출물이고 세계로 돌아오지 않는다 | 클라우드 | 0 | 0 | MCP·웹 2–8회, **A** |

- **합계 (엔진 트랙 G0).** S1 112 + S2c 81 + S2d 32 + S2e 30 + S4 300 + S7 110 + S8 50 = **715 TFU(약 700)**, INT8로 교사 약 31 T4-분이다.
- **G3 ≤520의 산정.** S4 외 단계를 G0 그대로(415) 두는 보수 산정이라 S4가 105까지 내려가야 한다(−65%). 이 감소는 o_s 한 옥타브 상향(재질 ≥50%), Spec 수락률 a ≥0.6, Router 의미 셀 비율에 달려 있다. Day 60에 실측 S4로 다시 맞춘다. S4가 170에 머물면 세계 TFU는 약 585(계획 T4-h 약 1.85)이고, Day 90의 TFU 게이트(≤520)와 계획 T4-h 게이트(≤1.8)는 미달로 보고한다. 목표는 낮추지 않는다.
- **순서.** S8.5(세계 경유 POLISH)는 반드시 S9–10 렌더 **전**이다. commit-before-shoot(§7.1 원칙 6)과 비회귀 조항 2를 지키기 위해서다. S11 스틸 광택만 렌더 뒤에 온다.

### 4.3 의사코드: WSSC-T4 v2.1

```text
procedure WSSC_T4(y, mode∈{text, directed}, bench∈{off, location-lite}, plan∈{A, A+, S}, track∈{E, H-A, H-A+}, gloss∈{off, still, world}):
  # ---- −1. 미감 교사 레인 (Soul: 닫힌, 콘텐츠 전용, 방화벽) ----
  C ← ∅
  assert not (track = H-A+ and plan < A+)                # H-A+는 A+ 승인 뒤에만
  if track = H-A+ or (track ≠ E and mode = directed):
      C ← SoulConcept(y)                                  # directed: 사람이 웹에서 / H-A+: MCP. prov=SOUL → soul_vault
  assert not (bench ≠ off and track = H-A and C ≠ ∅)      # H-A 벤치는 동결 프리셋만, 프롬프트별 Soul 없음 (CT38)
  assert not (track = E and gloss ≠ off)                  # Engine 트랙은 Soul 0 (CT38)
  assert not (bench ≠ off and track = H-A and gloss ≠ off)   # H-A 벤치는 광택도 없음 (CT38)
  θ_look ← (track = E) ? LookFromCaption(y)               # 엔진 트랙: 캡션 규칙만
          : (track = H-A+ and plan ≥ A+ and C≠∅) ? InverseLook(C)   # A+: 자동 역추정 (SOUL-DERIVED)
          : LookPresetLibrary.select(y) ⊕ HumanLook(C)    # A: 사람 측정 프리셋 (SOUL-INSPIRED)
  # ---- 0. 인코딩 (R-TE: encoder 전용) ----
  E   ← TE.encode(CaptionComposer(y, survey_tags)); unload(TE)
  # ---- 1. Picture-First Bedrock (전역 1-pass: 비회귀 조항 3의 유일한 예외) ----
  if mode = directed:  P ← WorldScriptDSL(y ⊕ HumanTranscribe(C));  H ← ∅
  else:
    H   ← HeroSearch(E; 64@512²·NFE3 [W4A4] → 16@512²·NFE8 → 4@1024²·SDEdit4)
           score = MRH.Aes ⊕ TeacherDCA(top16) − λ·C_T4(Corman) − GlyphS(pseudo-text)
    if track = H-A+ and plan ≥ A+ and C ≠ ∅:              # Soul 개념 아트 = 관측 제약
       C̃ ← ExplainAway(C, θ_soul)                         # 그레인·심도·그레이딩·비네팅 제거
       (x̂0_C, taps_C) ← Teacher(Enc(C̃), σ=snap(0.25)) @isolated(TapBus=off)   # 링버퍼·보정셋·VAE 쌍 수집 없음
       Exhaust.log(…, soul=1)
       H ← H ∪ {(C̃, MRH(taps_C)), prov=SOUL-DERIVED}
    G_H ← MRH(taps(H));  s ← SizePrior(slots) ⊕ MRH.FOV          # semantic metric prior
    P   ← MassingFM(E, lift(G_H, s))                   # shell, slot(OBB, shape16), 광원, matter, occ 32³
  W     ← Instantiate(P, Grammar ∪ CC0 assets)           # SDEdit 초기값일 뿐, L1 소유권 없음
  W.prov.soul ← max(prov(C), prov(θ_look), prov(H))     # none < inspired < derived < authored. 사람 전사한 C는 inspired. CoW 브랜치는 상속 (§3.7.3 3)
  W.L0_light ← Bake(OptiX | Warp-BVH, lights ← FitLights(H, MRH.shading) | DSL ⊕ θ_look.sun/haze)
  charts ← xatlas_WorldAxis(W);  folds ← FoldGraph(charts)
  # 의미 패스: 레이아웃 소유 = H (frustum 안) | P (밖)
  V_c ← DP0.demand(W) ∪ Coverage(Π) → 48 cine views @768²
  for σ_k in Snap(Turbo_sched, start≈0.6, K=3):          # 큰 σ는 W4A4, 이후 INT8
      ∀v lockstep: (x̂0_v, taps_v) ← Teacher(z_v, σ_k, E_v; GAdapter(Gbuf_v(W))) @isolated(TapBus=off) if W.prov.soul ≠ none
                   ĝ_v ← MRH(taps_v);  Exhaust.log(v, σ_k, x̂0_v, taps_v, soul=(W.prov.soul ≠ none))
      W ← FitThroughCamera_TSQ(W, {x̂0_v, ĝ_v, θ_v~p̂_T}; L0_light 고정, TAE-T by NoiseMaskRule)
      ∀v: z_v ← step(z_v, γ_k·Enc(Cam_θ R_v(W)) + (1−γ_k)·x̂0_v, WNF(seed, l_g, ·, k))
  P ← CMAES_LayoutVeto(P, ‖depth_MRH − depth(W)‖ + slot 충돌, 8 gen × 16 @512²)
  S ← SwatchBank(W.materials, L_T, seed)               # (seed, m, l, y) 순수 함수; bedrock commit에 포함
  Commit(bedrock); Merkle(+prov)

  # ---- 2. Demand (판정 해상도 분할) ----
  D1080, D4K ← RayBundle ∪ CormanShannon(CamFM.prior(W), lens plan(θ_look)) @1080p, @4K
  L_T ← levels(D1080);  L_S ← levels(D4K) ∖ L_T          # 8K면 L_S에 +1옥타브

  # ---- 3. 재질 패스: 교사 관할 레벨 (스케일 순서, 레벨 안 Jacobi lockstep) ----
  for l in L_T:
     (B_T, B_S, B_X) ← Router(D1080[l], o_s(m), sem)      # 교사 소유 / Spec-commit / native
     draft[B_S] ← TexelFM(parent=W[l−1], WNF(l), S[·,l], Wear(W), mat)
     acc ← { b∈B_S : DR_rel(Teacher_INT8, Enc(R_v(draft)), t_end(l), WNF′) ≤ 1.2 ∧ seam_ok }   # κ_v=2, 1 NFE/뷰
     Own ← B_T ∪ (B_S ∖ acc)                             # 라운드 상태의 순수 함수
     V   ← OTBI(Own, l, apron=25%) ∪ FoldViews(folds, l) ∪ SeamOwner(수평 모서리, l)
     Z_tex[l] ← LatentUp(Z_tex[l−1]) noised by WNF[l] at σ_start(l)
     for σ_k in Snap(window(l)) (2 step + 같은 σ 재판정, commit 스텝은 FP16 수술판):
        ∀v∈V (두 장, Jacobi): x̂0_v ← Teacher(crop(Z_tex[l]) ⊕ ContextRing_coarse(v), σ_k, E_v) @isolated(TapBus=off) if W.prov.soul ≠ none
        Z_tex[l] ← FixedPointSegmentSum_v(own_v · x̂0_v, apron multiband)   # TABP 닫힌형, 역전파 0, VAE 0
        Z_tex[l] ← Z_tex[l] + LatentUp(Z_tex[l−1]^commit − LatentDown(Z_tex[l]))   # 효도 투영
     HardCommit(l): graphcut(chart) + multiband; RGB[l] ← VAEdec(Z_tex[l]) (1회);
                    RGB[l] ← LaplacianLowSwap(RGB[l], RGB[l−1]); RGB[l, acc] ← draft[acc];
                    seed 동결; Merkle(+engine hash, +prov);
                    if W.prov.soul = none and track = E: Exhaust → TexelFM 래칫 데이터     # 방화벽. track = E면 gloss = off(assert)라 6.5단이 돌지 않고 prov.soul = none이 끝까지 유지된다
     RGB[l, B_X] ← TexelFM(...)                           # 같은 레벨 형제, 순서 없음
  # ---- 4. 미세 패스 (가난선 아래, native, T8) ----
  for l ∈ L_S: RGB[l] ← TexelFM(parent, WNF, S, Wear, mat, h_l);  Audit 5% (1 NFE)
  # ---- 5. Audit ----
  repeat ≤2: C* ← CMAES_AdvCam(pose×f×N; DR@512²);  reopen band(C*) via copy-on-write → 3
  # ---- 6. Shoot ----
  Γ ← DP0.search(W; CamFM-CRM + β·MRH-Aes + β_FL·ForegroundLayering + β_sc·ScaleRecipe − λ·TFU(Γ))
  L_shot ← Gaffer(W, Γ; init=θ_look.rigs)
  # ---- 6.5 세계 경유 POLISH (선택, A+, v2.1): 반드시 렌더 전 (commit-before-shoot) ----
  if gloss = world and plan ≥ A+:                        # track ≠ E, H-A 벤치 아님 (위 assert)
      G ← SoulGloss(R_{keyframes(Γ)}(W)) @isolated(TapBus=off)       # Soul 구조 보존 편집, gauge 제거
      W ← PolishRound_CoW(W, G, mask=own∧band≤Nyquist_2K)   # CoW로 다시 연 1개 Jacobi 라운드, 엔진 교사 뷰와 동시
      엔진 교사 재판정(DR_rel ≤ 1.2);  Commit(+prov=authored); Merkle(+prov)
  F ← Sensor_Film(θ_look, Lens(θ_look.lens_rx, CPL)(Raster_TSQ + RT(W, Γ)))   # 렌더 경로에 생성 연산 0
  # ---- 7. 스틸 광택 (선택, A) ----
  if gloss = still:            K ← SoulGloss(keyframes(F))          # 콘텐츠 산출물, 세계로 돌아오지 않음
  # 비디오 프레임별 SoulGloss는 호출 경로 자체가 없다 (CT37)
  Firewall.assert_no_soul_ancestor(training_artifacts)
  return F                                                # 저장: (Merkle 루트, Γ, θ_look, Soul 자산 해시)
```

### 4.4 세계 1개와 데모 1편의 값

**세계 1개 (방 규모, 4K, Directed).**

| 경로 | 교사 TFU | 교사 T4-분 | 비교사 GPU T4-분 | 순수 T4-h | **계획 T4-h (×2 공학 오버헤드)** | 벽시계 (두 장) | Soul 호출 |
|---|---|---|---|---|---|---|---|
| Mode A INT8, gen0 | 약 700 | 30 | 30 | 1.0 | **2.0** | 40–60분 | 0 (엔진 트랙) |
| Mode A W4A4 혼합 | 약 700 | 21 | 30 | 0.85 | 1.7 | 35–50분 | 0 |
| Mode B FP16 분할 | 약 700 | 52 | 30 | 1.4 | **2.8** | 60–90분 | 0 |
| Mode A, 래칫 G3 | 약 520 | 22 | 30 | 0.9 | **1.8** | 35–55분 | 0 |
| 최악 (FP16, 가난선 없음, VAE 루프) | 약 1,800 | 135 | 45 | 3.0 | 6 | 하룻밤 | 0 |
| Hybrid-A (프리셋 라이브러리, **Engine 세계 재사용**: L0 재bake + 수요 차 레벨 CoW + S8–S10) | 50–100 (S8 Gaffer 50 + CoW 0–50) | 2–4 | 8–17 | 0.17–0.35 | **0.35–0.7** | 10–25분 | 0 (사전 측정한 프리셋 재사용) |
| Hybrid-A (단독으로 지을 때) | 약 700 | 30 | 30 | 1.0 | 2.0 | 40–60분 | 0 |
| Hybrid-A+ (개념 아트 관측 제약) | 약 705 | 30 | 31 | 1.0 | 2.0 | 40–60분 | MCP 4 |

**변형.**

| 변형 | T4-h |
|---|---|
| 벤치 Location-lite(의도 공통 Π, 커버리지 ×1.25–1.5) | 2.5 |
| 8K 추가(TexelFM l=15, 의미 셀만 교사) | +0.2–0.5 |
| 데모 히어로(best-of-4 hero + polish ×2) | 6–10 |
| 10분 원테이크 집(방 5–8개) | 6–10 |
| Powers of Ten | 5–15 |

- v1 추정 4–12 H100-h ≈ 80–370 T4-h(NC17 환산, 리서치 환산 100–300) 대비 **약 40–185배 싸다.**
- 출처는 넷이다. 가난선·판정 해상도 분할·래칫(4–5배), OTBI 단독소유(뷰 1/2–1/4)와 VAE·Adam 제거, Directed 수요와 Corman(1.3–10배), 교사 크기(12–20B → 6B, 2–3배).
- 참고: TF(u)의 G2 추론 경로는 세계 + 첫 샷이 Mode A 약 20–23분, 같은 세계의 추가 샷이 7–10분이다. 계획에는 위의 보수치를 쓴다.
- 전력은 세계당 약 140 Wh, 노트북 두 번 충전하는 양이다.
- **Soul은 T4 시간을 거의 더하지 않는다.** 비용은 크레딧과 사람 시간이다.
- **Hybrid-A는 Engine 세계를 재사용한다 [v2.1].** 측량 모드로 지은 bedrock과 재질 commit은 룩과 독립이다(§3.3 항목 6). 그래서 Hybrid-A는 같은 프롬프트의 Engine 세계에서 CoW 브랜치를 떠서 θ_look의 태양·대기로 L0만 다시 bake하고, θ_look의 렌즈 계획이 바꾼 수요 차이 레벨만 열고, S8–S10만 다시 돈다. 브랜치는 `prov.soul = inspired`로 태깅되어 래칫에서 빠진다(원본 Engine 세계는 그대로 래칫에 남는다). 벤치 파일럿에서 재사용이 실패하면(수요 차 CoW가 세계당 0.7 T4-h를 넘으면) 단독으로 짓고, 차액 최대 약 320 T4-h를 예비에서 쓴다.

**데모 1편.**

| 항목 | T4-h | Soul | 사람 |
|---|---|---|---|
| 개념 아트·룩 측정 | 0 | 웹 40–60 | 아트 디렉터 3–6 h |
| 히어로 세계 | 6–10 | (A+) MCP 4–8 | — |
| Hero FX | 2–10 | — | — |
| DP-0 탐색 | 1–2 | — | 감독 |
| 최종 8K120 30초(3,600 프레임 × 1.1–2.5 s) | 1.1–2.5 | — | — |
| 4K 테이크와 재렌더 ×3 | 1–2 | — | — |
| 경쟁 비교와 GeCo | 3 | — | — |
| 키 아트 광택, IP 점검 | 0 | 웹·MCP 10–20 | 1–2 h |
| **합계** | **14–30** | **50–90회** | **4–8 h** |

3+1편이면 약 **60–120 T4-h**(상시 가동 1.5–3일)다. §11.5 데모 버킷 140은 3+1+1편과 Corman 슬라이더를 담는다.

### 4.5 비회귀 점검 (순차 아웃페인팅 금지)

v1 §1.10의 세 조항: (1) 방문 순서 간선이 없다. (2) 어떤 영역도 다른 시점에서 먼저 생성된 렌더 픽셀을 조건으로 쓰지 않는다. (3) completion은 스펙트럼적이다(유일한 예외는 전역 1-pass bedrock).

**Phase 1 부품.**

| 부품 | 조항 1 | 조항 2 | 조항 3 | 등급 |
|---|---|---|---|---|
| Hero + MassingFM bedrock | 전역 1-pass | hero는 레이아웃 대역의 관측 likelihood(DPS)이고, 48 cine 뷰는 동시에 디노이즈된다 | v1이 정한 유일한 예외 | 예외 범위 안 |
| 의미 패스 lockstep | Jacobi | 초기 z는 MassingFM 세계 렌더 + WNF | — | 통과 |
| Swatch Bank | 수요와 독립인 순수 함수 | **해석 확장(명시):** 견본은 bedrock commit의 재질 정의(SMF τ의 고해상판)다. 조건 출처는 더 거친 레벨이다 | 영역 없음 | 비트 동일 |
| OTBI + TABP + Fold + Context Ring | Jacobi. 고정소수점 정렬 리덕션으로 순서와 무관 | Context Ring은 **더 거친 commit 레벨**에서만 가져온다 | 차트 전체를 동시에 디노이즈한다 | solver OIE ≤0.05 |
| Spec-commit | 수락·거부가 라운드 상태의 순수 함수 | 검증은 세계 자신의 렌더를 **판정**할 뿐 조건 입력이 아니다 | 영역 확장 없음 | OIE ≤0.05 |
| TexelFM native | 고정 타일 격자 | 부모, WNF, 견본 | 2×2 평균0 사영 | **T8 비트 동일** |
| Texel Bank | — | **다른 세계**에서만 검색한다 | — | 통과 |
| G-Adapter | — | 입력은 세계 기하(더 거친 레벨) | — | 통과 |
| Router, Corman, 판정 해상도 분할 | 연산자만 고르고 순서축은 그대로 | — | — | 통과 |

**Soul 레인 [v2.1].**

| 부품 | 조항 1 | 조항 2 | 조항 3 | 등급 |
|---|---|---|---|---|
| Soul 개념 아트, 사람 전사 (A) | — | 사람의 입력은 y(WorldScript)의 일부다 | — | 통과 |
| Soul 개념 아트 관측 제약 (A+) | 전역 1-pass bedrock 안 | Soul 스틸은 hero와 같은 지위(y의 일부)다. 세계의 렌더 픽셀이 아니다 | bedrock 예외 범위 | 예외 범위 안 |
| θ_look 룩 프리셋 | — | 카메라·조명 파라미터일 뿐 생성이 없다 | — | 통과 |
| 원경 매트 페인팅 (콘텐츠 자산) | — | y의 일부(far-field 슬롯, 무한원) | 영역 확장 아님 | 통과 |
| Soul 스틸 광택 (A) | — | 세계 밖 산출물이고 세계로 돌아오지 않는다 | — | 해당 없음 |
| Soul POLISH 세계 경유 (A+, S8.5) | Shoot 전에 CoW로 다시 연 1개 Jacobi 라운드에서 히어로 키프레임 뷰와 엔진 교사 뷰를 동시에 편집·소유. 렌더(S9–10)는 commit 뒤에만 | 조건은 **그 라운드의 세계 자신의 렌더**다. WSSC 고정점 반복의 Enc(R_v(W))와 같은 지위다(v1 §1.6 iii). 확정된 다른 영역의 픽셀로 새 영역을 만들지 않는다 | 소유 마스크 안, 2K Nyquist 이하 대역만, 영역 확장 없음 | OIE ≤0.05 (X-Soul에서 측정) |
| **Soul 비디오 프레임별 광택** | 프레임 순서가 곧 방문 순서다 | 앞 프레임의 결과가 사실상 다음 프레임의 조건이 된다 | — | **위반. 금지(CT37)** |

**Phase 2 부품 [v2.1 복원: P2(o) §15].**

| 부품 | 판정 | 근거 |
|---|---|---|
| EPD-S | 통과 | 사건 **집합** 디퓨전이다. 슬롯은 병렬이고 스크립트 제약은 집합 공간 inpainting이다. 픽셀 캔버스가 없다 |
| MoPri-S (긴 모션) | 통과 | 겹치는 창을 lockstep으로 동시에 디노이즈한다(시간 방향 MultiDiffusion). 확정된 창을 조건으로 쓰지 않는다. 창 처리 순서를 섞어 `order_purity_check`로 OIE를 검사한다 |
| 실현 (시뮬) | 통과 | 인과적인 것은 **상태공간 ODE 적분**이고 픽셀 생성이 아니다. 세계는 원래 그렇게 돈다 |
| Judge Cascade, 크로노 증인 | 통과 | 판정은 **읽기**다. 후보 자신의 렌더로 고를 뿐, 다른 영역 생성의 조건으로 쓰지 않는다 |
| Reveal-Bundle Demand | 통과 | 세계 공간에서 처리하고, 거친 레벨이 이미 존재한다. 같은 레벨 안에서 lockstep으로 풀고, 렌더 픽셀을 조건으로 쓰지 않는다 |
| BSRT | 통과 | Reader 학습 데이터 전용이다. 렌더 경로에 없다 |
| MAN-S, Denoiser-S | 통과 (I6) | 결정론적이고 알파·기하 출력이 없다. 노이즈는 물질에 고정된 MAN 노이즈만 쓰므로 프레임 단위 latent가 없다. Denoiser-S는 64 spp 이상의 입력에 잔차 크기 제한을 걸어 콘텐츠를 지어내지 못하게 한다 |
| Chaos Bank 검색 | 통과 | 상태공간 초기 입자로만 쓰고 픽셀은 쓰지 않는다 |

CI에서 `order_purity_check`를 돌린다. 수요 순서(P2는 창 순서)를 8번 섞고 Merkle 루트와 OIE를 비교한다. v2.1부터 같은 CI가 `Firewall.assert_no_soul_ancestor`와 비디오 프레임별 Soul 호출 부재를 함께 검사한다.

---

## 5. 수십M from-scratch 모델 카탈로그

### 5.1 공통 레시피

- **처음부터 학습한다.** `weights=None`을 강제하고 체크포인트 해시 금지 목록을 Lineage CI로 검사한다(torchvision RAFT, HRNet ImageNet 초기화 같은 기본값 함정을 막는다).
- **Soul 없이 학습한다.** 모든 학습 샤드는 `prov` 검사를 통과해야 한다(§3.7.3, §6.6). 이 카탈로그의 어떤 모델도 Soul 출력으로 학습하지 않는다.
- **FP16 안정화:**
  - FP32 마스터 + autocast FP16, GradScaler(2¹⁶, growth 2000), inf가 나오면 해당 스텝을 건너뛴다
  - LN, softmax, 손실, FFT, R1은 FP32. `allow_fp16_reduced_precision_reduction=False`
  - QK-RMSNorm, z-loss, logit soft-cap 30, pre-LN 잔차 1/√(2L)
  - transformer는 u-μP. conv 네트는 GroupNorm/RMSNorm(FP32), BatchNorm 금지
  - 층별 max-abs를 100스텝마다 로깅하고 2×10⁴을 넘으면 경보
- **T4 roofline:**
  - 넓고 얕게. d와 채널은 64의 배수이고 384 이상.
  - 고해상 conv는 space-to-depth로 산술강도를 능선 위로 올린다(3×3 conv 강도 ≈ 4.5·C).
  - 배치당 토큰 ≥8k(가능하면 32k).
  - head_dim 64. `channels_last` fp16.
- **μP 프록시:** 폭 1/4로 32개 설정을 sweep한다. CUDA MPS로 T4 한 장에 잡 4–8개를 겹친다. 실험 단위는 2시간 이하의 멱등 잡이고 SQLite 큐로 관리한다.
- **무상태 샘플러:** Feistel PRP로 샘플 인덱스를 정한다. 15분 또는 1,000 스텝마다 원자적 체크포인트(tmp → fsync → rename)를 쓰고, 30–60분마다 외부로 업로드한다.
- **배포:** TensorRT. critic은 INT8, 샘플러는 FP16.

### 5.2 정본 이름 체계와 별칭표 [v2.1 신규]

규칙: 문서, 코드, 체크포인트, Lineage 노드는 **정본 ID**를 쓴다. 별칭은 렌즈 원문을 찾아갈 때만 쓴다. 한국어 별명은 설명용이다.

| 정본 ID | 정본 이름 | 별명 | 별칭 (출처 렌즈) | 정본 크기 | 크기 결정 근거 |
|---|---|---|---|---|---|
| M1 | **TAE-T** | 꼬마 VAE | TAE-T 2.5M [CB], TAE-T 2.4M [TF(u)], TL-3M / TL-2.5M / TL [RR] | 2.5M | TAESD 선례(약 1.2M), Z-Image 16ch 대상 |
| M2 | **LatentDown/Up-S** | 레벨 승강기 | LatentDown/Up-S [CB] | 2×1M | 거의 선형인 국소 연산자 |
| M3 | **MRH** (Mind-Reading Heads) | 독심 헤드 | Free-rider probe(프로브) 20M [CB], MRH [TF(o)], MRH-D 22M [TF(u)], GeoHead 25M + AesHead 5M [RR] | 24M (Aes·FOV 헤드 포함) | 탭을 한 번만 뽑는다(CT12) |
| — | **Teacher-DCA** | 교사 미감 판정 | Teacher-DCA [RR] | 0 (파라미터 없음) | Diffusion Classifier식, 조건 2개 × t 2개의 denoising 오차 차. 상위 16개에만 |
| M4 | **G-Adapter** | 조종간 | G-Adapter 25M [CB], G-Adapter-DiT 20M [TF(u)], GB-Adapter 30M [RR] | 25M (조건부) | T2I-Adapter 선례, E3가 생략 여부 판정 |
| M5 | **TexelFM-35** | 디테일 연산자 | LBO-S [TF(o)], ICX-LBO 26M [TF(u)], BO-40 v2 40M [CB], TexelFM 30M [RR] | 35M | s2d×4 + 견본 cross-attn + 마모장 + 2×2 평균0(CT8) |
| M6 | **T-LPIPS-S** | 지각 자 | T-LPIPS-S 4M [CB], TPD [TF](교사 특징 직접 거리: 라벨·평가 전용), PercepNet-S 3M [P2(u)], LatentLPIPS식 [P2(o)], Laplacian + 교사 특징 [RR] | 4M | LPIPS 자체가 특징 위의 선형 헤드 |
| M7 | **MassingFM** | 덩어리 모델 | WP-30 30M [CB], MassingFM 21M [RR] (CaptionComposer는 코드 [TF]) | 24M | 레이아웃 엔트로피가 낮다 |
| M8 | **BRP** | 뒷면 완성기 | BRP 42M [TF] | 42M (`P1-확장`) | SSC 선례 |
| M9 | **IPM-S** | 물질 역추정기 | SMF-τ 5M [CB], IPM-S 8M / 12M [TF] | 8M | 역사상은 64차원 이하 |
| M10 | **CamFM-Intent** | 카메라 의도 | DP-0-S 25M [CB], Path-S 19M [TF(u)], CamFM-Intent 15M [RR] | 20M | 의도는 1 Hz 이하 저차원 신호 |
| M11 | **CamFM-Hand** | 카메라 손 | 손 모델 8M [CB], Hand-S 8M [TF], CamFM-Hand 5M [RR] | 8M | 데이터 2.7M Chinchilla, 증강 ×5.5 |
| M12 | **CamFM-CRM** | 카메라 평론가 | CRM 15M [CB], CRM-S 12M [TF(u)], CamFM-CRM 8M [RR] | 12M | 해석 통계가 비전 백본을 대신 |
| M13 | **Rig-ID-S** | 리그 감식 | Rig-ID-S 3M [TF] | 3M | 홀로노믹 지문 |
| M14 | **Glyph-S** | 글자 탐지기 | Glyph-S 4M [TF], GlyphSpotter 5M [RR] | 4M | 검출은 국소 텍스처 판별 |
| — | (P1 학습 광류 없음) | — | Flow-S 5.3M [CB], RAFT-S 1M [TF] → P1은 DIS/TV-L1, 학습 광류는 N7 R-Flow로 통합 | 0 | CT19 |
| A1 | **Cine/Texture LoRA** | 시네마틱 LoRA | 텍스처 LoRA [CB], 시네마틱 LoRA rank 32 [RR], A1 [v2.0] | 20–30M (선택) | 엔진 교사 어댑터 |
| A2 | **SR-LoRA** | 판정 보강 LoRA | SR-LoRA [P2(u)], t-densify LoRA 72M [TF(u)] | 약 70M (조건부) | 1.1%, CT3 |
| N1 | **EPD-S** | 사건 모델 | EPD-S 28M [P2], EPD-S 38M [TF(u)], EPD-S 30M [CB] | 28M | 슬롯 ≤256 |
| N2 | **MoPri-S** | 동작 사전 | MoPri-S 32M [P2], MoPrior-S 10M [TF] | 16M → 32M (데이터 게이트) | CT24 |
| N3 | **Feas-S** | 실현성 판정 | Feas-S 5M [P2] | 5M | — |
| N4a | **HeroProxy-S** | 정지 판사 대리 | HeroProxy-S 14M [P2] | 14M | Exhaust 증류 |
| N4b | **ChronoProxy-S** | 시간 판사 대리 | ChronoProxy-S 12M [P2] | 12M | — |
| N5 | **MAN-S** | 셰이딩 잔차 | MAN-S 1.5M [P2], MAN-S 2M [TF], MAN g_φ 15M [CB] | 1.5M | 수용장 16² |
| N6 | **F_φ-S** | 잔차 동역학 | F_φ-S 4M [TF, P2(o)], F_φ 8M GNN [CB], F_φ-S 0.6M [P2(u)] | 0.6M | CT25 |
| N7 | **Reader-S** | 실영상 판독기 | R-Flow 3M(= Flow-S/RAFT-S), R-Kpt 18M(= Kp2D-S 12M [TF]), R-Seg 4M, R-Event 1M(= Timing-S 3M [TF], 오디오 onset 포함) | 26M | — |
| N8 | **Denoiser-S** | 디노이저 | Denoiser-S 1M [P2] | 1M | OIDN식 |
| N9 | **Expr-S** | 표정 | Expr-S 12M [P2] | 12M (2c) | — |
| N10 | **TTS-S + Voc-S** | 음성 | TTS-S 20M + Voc-S 14M [P2] | 34M (2c 후반) | — |
| — | PCGD-0, ID-Screen, R-Lift | — | — | 약 300 / 0 / 0 (NN 아님) | 고전 |
| (S 전용 후보) | **LookPrior-S** | 룩 예측기 | 신규 | 약 2M | 프롬프트 → θ_look. **Plan S 승인 전에는 편성하지 않는다.** A·A+에서는 0 파라미터 프리셋 검색 |

- **DP-0는 모델이 아니다.** 문서에서 DP-0는 샷 계획 절차(후보 탐색 + DP-0 보상, §7.8)를 뜻한다. 그 안의 학습된 prior는 CamFM-Intent(CB 별칭 DP-0-S)이고, 평가기는 CamFM-CRM이다.
- **표기 규칙.** 크기는 이름에 붙이지 않고 괄호로 쓴다(예: MRH(24M), MassingFM(24M)). 이름 자체에 크기가 들어간 정본은 TexelFM-35 하나다.

### 5.3 Phase 1 카탈로그 (전부 `P1-코어`, 표기한 것만 예외)

| ID | 모델 | 역할 (대체하는 v1 부품) | 파라미터·아키텍처 | 입력 → 출력 | 데이터 (사전학습 모델 0, Soul 0) | 학습 FLOPs / T4-h | 추론 / 메모리 | 왜 수십M이면 되나 |
|---|---|---|---|---|---|---|---|---|
| M1 | **TAE-T** | 루프 안 VAE 대리 (곡면 경로, 프리뷰, DP) | 2.5M. conv 64ch f8 AE (enc 1.2 + dec 1.3) | RGB [B,3,H,W] ↔ z [B,16,H/8,W/8] | 교사 VAE enc/dec 쌍(교사의 일부) × 128² 크롭 400만(RAW, CC0, 우리 렌더(`prov.soul = none`만), 교사 샘플). L1 + T-LPIPS-S + from-scratch PatchGAN 3M + 교사 x̂0 일관성 항 | 5×10¹⁷ / 27 (타깃 7 포함) | 1024² decode 25–40 ms / <0.5 GB | TAESD 선례(약 1.2M). latent는 국소적이다 |
| M2 | **LatentDown/Up-S** [신규] | 레벨 간 latent 효도 투영 | 각 1M. latent 3×3 conv 6층 128ch | Z [128²×16] ↔ Z [64²×16] | M1 크롭의 (Enc(x), Enc(down2 x)) 쌍 | 10¹⁶ / 3 | <5 ms / 0.1 GB | 거의 선형인 국소 연산자 |
| M3 | **MRH** | dreamer depth head, DA3/VGGT/MoGe/DUSt3R/TRELLIS lift, SigLIP sem, 미감 판사, DTC p̂_T, σ̂² | 24M. Frozen PCA 탭(층 8/15/23, 3840→256 고정) → DPT reassemble + fusion neck 256ch(RCU 12, FiLM t) 14M + 전역 xfmr d384 L4 + register 8개 7M + 헤드 10개 2M | 탭 3×[(H/16)²×3840] + t + pooled TE → @H/4: log-depth 1(affine-inv, metric은 크기 사전), normal 3, albedo 3, rough/metal 2, shading 1, defocus 1, sem 768(TE PCA), σ̂ 4대역, Aes 1, 전역 8(FOV, 초점, EV, WB, 수평선) | (a) 운송/Realify 15만: 우리 렌더러 GT AOV + 교사 SDEdit t∈스냅[0.2, 0.7], 2–3 NFE, 결정 곡선 대역 마스크. (b) 실사 MVS 3–6만: 자체 촬영 300장면, COLMAP SIFT + GLOMAP + PatchMatch + raw LiDAR 스케일. (c) W\* 자가 라벨 ≤25%(엔진 트랙 세계만). (d) σ̂: X1-slim seed 간 분산. (e) Aes: 우리 렌더·엔진 교사 스틸·CC0 사진끼리의 사람 쌍 2–3만(데일리, X2, X7, E1-E 보조 arm). **경쟁 API 프레임(§9.2: API 출력은 학습 금지) 또는 Soul 계보 태그(`SOUL`/`SOUL-DERIVED`/`SOUL-INSPIRED`)가 든 쌍은 제외** | 헤드 2×10¹⁷ / 5–10. 특징 공장 약 80 T4-h는 데이터로 계상 | NFE당 +40–60 ms(FP16 1024² 기준 ≤1.5%, INT8 기준 1.5–2.3%) / 1.5 GB | 이해는 6B 교사가 이미 했다. 헤드는 readout일 뿐이다(Beyond Surface Statistics, Probing 3D Awareness) |
| — | **Teacher-DCA** [조합] | 미감 판사 보조(리워드 해킹 내성) | 0 파라미터 | 교사 조건 2개 × t 2개의 denoising 오차 차 | — | 0 | 후보당 4 fwd, 상위 16개에만 | Diffusion Classifier 선례 |
| M4 | **G-Adapter** (조건부) | 사전학습 ControlNet(금지) 대체 | 25M. G-buffer 16 px patchify → xfmr 6층 d512(hd 64) → zero-init 선형 주입 5곳(DiT 블록 0/6/12/18/24, 폭 3840) | G-buffer [B,12,H,W] (log-depth 1, normal 3, albedo 3, sem 4, 레벨 마스크 1) → 5×[B,N_tok,3840] 잔차 | 합성 2–3만(우리 렌더러 GT) + 실사 MVS 2만 + 교사 자기 샘플 × MRH 라벨. NF4/INT8 동결 교사를 통과하는 역전파 @512², grad ckpt | 샘플당 약 48 TF × 4–12만 샘플-패스(1–3 epoch) = 1.9–5.8×10¹⁸ / **36–110 (E3가 판정, SDEdit만으로 충분하면 생략)** | 샘플당 1회 15 ms, 교사 +1–2% / 학습 10.5–14 GB | T2I-Adapter(SDXL용 77M) 선례. 제어 신호는 저차원이다 |
| M5 | **TexelFM-35** | LBO 200M + WDD + Spec-commit 초안 작성기 | 35M. s2d×4(256²×16ch → 64²×256) → conv-res 320ch ×8 → window attn 512ch ×6 @32²(hd 64, QK-norm) + 견본 cross-attn(4 타일 → 1,024 토큰, 인코더 2M) → up → d2s×4. FiLM(log₂h_l Fourier 16, 대역, t). 출력에 **2×2 평균0 사영**(효도 정확) | 부모 [B,9,320,320](log L_d 3, albedo 3, normal-xy 2, height 1; 256 + apron 32) + WNF ξ [B,4,320,320] + 마모장 [B,3,…](곡률, AO, 접촉 빈도) + 재질 emb 64 + sem 마스크 + 견본 [B,4,12,64,64] → 잔차 [B,9,256,256] | CC0 PBR(고유 4–5k 재질, 4–8K, 실측 크기) 옥타브 피라미드, **자체 RAW 매크로 2–5만**(교차편광, 포커스 스태킹), 절차 재질 그래프, 교사 band-iso 타일 300만, W\* 페이지(래칫, 엔진 트랙만). FM + shortcut 2–4 step, 방사 스펙트럼 손실(FP32), patch-GAN(D 4M) 미세조정, FP16 GAN 안정화(R1 FP32, SN, EMA) | 4.8×10¹⁸(1,000만 샘플) / **세대당 약 90** | 10⁸ 텍셀 20–40 s(INT8 TRT 12–25 s) / 학습 9 GB, 추론 1–2 GB | 국소적이고 정상(stationary)인 파워법칙 통계다. SR 선례: Real-ESRGAN 16.7M, SwinIR 11.8M, HAT-S 9.6M. **full-res conv로 만들면 이득이 없다(NC6)** |
| M6 | **T-LPIPS-S** [신규] | VGG-LPIPS(금지) | 4M. conv 5단(32→256ch) + 층별 선형 가중 | 2×[B,3,256,256] → 거리(스칼라 + 맵) | 패치 쌍 20만(고전 열화, TexelFM·TAE-T 출력, 워프) × 교사 특징 거리 라벨(40만 fwd @256², 17–29 T4-h) → BAPPS 사람 2AFC(라이선스 확인)로 미세조정 | 10¹⁷ / 3 | 256²에 1 ms / 0.5 GB | LPIPS 자체가 특징 위의 선형 헤드다 |
| M7 | **MassingFM** [신규] | LLM World Program, CWM 레이아웃 | 24M. DiT 12층 d384(hd 64). 연속값은 flow matching, 이산값은 masked | TE 토큰(≤256×2560 → 384) + (선택) hero lift 부분 점유 32³ + slot 힌트 → shell ≤32, slot ≤64(class 약 800, OBB 2 cm 양자화, yaw, shape 16, 재질 클래스), 광원 ≤16, matter, 점유/sem 32³ | **자체 레이아웃 문법 300만 장면**(ProcTHOR식 + 가벼운 Infinigen 제약 언어 재구현, 한옥·을지로·바 문법; 장면당 50 ms, 42 CPU-h) + Infinigen-Indoors 3–5k + 문법·패러프레이즈 캡션 30만(고유) → TE 임베딩(4 T4-h) + 실사 MVS 부분 관측 | 7×10¹⁷ / 16–25 | 16 step 0.3 s / 0.5 GB | 레이아웃 엔트로피가 낮다. 문법이 라벨을 무한히 준다 |
| M8 | **BRP** `P1-확장` | CWM의 "내부·뒷면, 구멍 없음"(비정형 장면) | 42M. 3D U-Net 48–384, text x-attn @8³ | 부분 관측 64³×32ch + TE → 완성 64³×32(occ, sem 8, coarse albedo 3, normal 3, emissive, matter 16) | Warp GPU ProcGen 20만(지형, 자연) + Infinigen 3k(지오메트리만) + 자체 lift 300 | 2.9×10¹⁷ / 7–15 (3D conv는 MFU가 낮다) | 청크당 약 1 s / 학습 6–8 GB | SSC 선례 LMSCNet 0.35M |
| M9 | **IPM-S** | SMF G_SM 파라미터 역추정 | 8M. ConvNeXt식(MRH neck 위) + xfmr 2층 | 영역 특징 + 마스크 → τ_SMF 32, CARRIER(k₁, k₂, 위상, profile 16), 재질 그래프 id(약 150) + 파라미터 ≤24 | 파라미터를 아는 절차 instancer 렌더(라벨 공짜, 운송에 기생) | 5×10¹⁶ / 2 | 영역당 5 ms | 역사상은 64차원 이하다 |
| M10 | **CamFM-Intent** | DP-0 prior | 20M. DiT 8층 d384. FM(연속) + masked(이산) | 키프레임 ≤16 × [pos 3, rot6d 6, f, d_f, N, 리그 원핫 14, t] + 세계 토큰 ≤128(MassingFM slot + 4³ 점유) + TE 1토큰 + 스타일 32 → velocity | **Blender 오픈무비 .blend 카메라(CC-BY, 정확한 GT)**, 절차 Toric 문법 200–500만, 자체 GLOMAP 추출(PD·라이선스 클립 5–10k, 약 800 CPU-h), VCam 퍼포먼스 캡처. DataDoP는 평가 전용 | 5×10¹⁷ / 12 | 궤적 1,000개에 1 s | 의도는 1 Hz 이하의 저차원 신호다 |
| M11 | **CamFM-Hand** | IMU 손 모델 | 8M. causal xfmr d384 L5, μ-law 256-bin | 과거 2 s IMU(200 Hz gyro + accel, 20 ms 패치) + 리그·프로파일·의도·피로·숨 참기 → 다음 20 ms | 자체 IMU 300 기기-시간(폰 4 + GoPro 2 동시, 오퍼레이터 3–6명, 리그 4–10종), Gyroflow·Betaflight 로그, 리그 전달함수 증강 ×5.5. 평행이동은 OpenVINS(고전) | 10¹⁷ / 3 | 10 s 샷 <50 ms | 생리 떨림, 보행, 호흡은 저차원 확률과정이다 |
| M12 | **CamFM-CRM** | CRM | 12M. 경로 인코더 + **해석 통계** xfmr + 프리뷰 CNN 3M | (세계 토큰, 경로, 64-d 해석 통계 @10 fps, 160×90 프리뷰) → 점수 | 합성 선호 200만(원 경로 vs 열화 경로: 지터, 충돌, 머리 잘림, 수평 기울기, **전경 레이어링 제거**) → 사람 5k → 1.5–2만 + 전문가 500 → 2k(BT + 마진). **Soul 구도는 쓰지 않는다**(S1 해금 대상). Q17 판정 전까지 사람 쌍은 수제 프리셋(§7.6 출처 i) 렌더로만 모은다(`SOUL-INSPIRED` 상속 렌더 제외) | 5×10¹⁶ / 1–2 (+프리뷰 렌더 18) | 후보당 2 ms | 렌더러가 depth, flow, saliency를 정확히 준다 |
| M13 | **Rig-ID-S** | ShotDNA 리그 역추정 | 3M. 1D conv + xfmr | 포즈 시퀀스(SfM 노이즈 포함) → 리그 클래스, 파라미터, 손 프로파일 | 리그 시뮬 + 추출기 노이즈 $\lvert H\rvert^2$ | 10¹⁵ / 0.2 | 1 ms | 홀로노믹 지문은 저차원이다 |
| M14 | **Glyph-S** | $\mathcal L_{ocr}$의 OCR | 4M. U-Net-lite (+선택 CTC 2.8M) | [B,3,H,W] → 텍스트 확률 [B,1,H/4,W/4] (+ 한글·라틴 문자열) | OFL 폰트(Noto, 나눔) × 우리 렌더러 × 절차 배경 × 교사 운송 200만, 음성 샘플 포함. 한글 11,172자 | 1.2×10¹⁶ / 2–3 | 1024²에 30 ms | 검출은 국소 텍스처 판별이다. 엔진 교사의 가짜 한자를 잡는다 |
| A1 | Cine/Texture LoRA (선택) | `<plate>`, 미감 보정 | rank 16–32, 20–30M, QLoRA(NF4 베이스) @512² | — | 자체 RAW 1–2만 장 또는 자체 촬영 프레임 2천 장 | 10¹⁸ / 17–45 | 병합하면 0 / 약 10–14 GB | 어댑터 |

### 5.4 Phase 2 카탈로그 (P2 렌즈 v2.1 기준)

| ID | 모델 | 파라미터·아키텍처 | 입력 → 출력 | 데이터 (순도) | 학습 FLOPs / T4-h | 추론 / 메모리 | 배치 |
|---|---|---|---|---|---|---|---|
| N1 | **EPD-S** | 28M. xfmr 12층 d384, self + cross | 슬롯 [B,256,128] + 문맥(slot ≤128, TE 2560→384) → 속도장 + 이산 logits | Chaos Bank + 시뮬 사건 2–5M(D0), Garage(D0), 오디오 간격 주변분포(D1) | 6.6×10¹⁷ / 15 + 증류 5 | 1024개 약 4 s / 6 GB | 2a–2b |
| N2 | **MoPri-S** | **16M → 32M**(데이터 게이트), 10층 d448 비AR | x ∈ ℝ^{B×T×140}(T≤300, 30fps, 6D×22관절 + 루트 + 접촉) + 텍스트, 사건 트랙, 스타일, 뼈 길이, 키프레임 마스크 → x̂0 | **CMU 원본 ASF/AMC, 100STYLE BVH(CC BY 4.0 [2차]), Garage 마커 모캡, Act-from-Text 캡처**(D0/D2). 증강 ×10, MJPC 재시뮬 ×3. AMASS·HumanML3D는 연구 전용 | 1.9×10¹⁸(32M) / 44 + 13 | 4096개 21–40 s / 7 GB | 2a–2c |
| N3 | **Feas-S** | 5M, 6층 d256 | 사건 집합 + slot → P(성공), 예상 비용 | SMC 성패(D0) | 5×10¹⁶ / 1 | <1 ms | 2a– |
| N4 | **HeroProxy-S / ChronoProxy-S** | 14M / 12M. ViT 8층 d384 p16 @256², 앙상블 헤드 4개 | 7ch → 대역 DR 2 + 16² DR 맵 + 불확실성 / 크로노 합성 → Δχ(GICC) | Teacher Exhaust ≥1M(누적, soul=0만) | 2.5×10¹⁷ / 6 | INT8 4096장 약 1.5 s(+렌더 16 s) | 2a– |
| N5 | **MAN-S** | 1.5M. U-Net 3스케일 32/64/128, 수용장 16² | G-buffer 40ch + MAN ξ 8ch → 잔차 RGB, tanh·τ | **Exhaust-to-Shader 쌍 2만**(BSRT 부산물) 지도학습 3 T4-h → DMD 5k step(16만 TC-eq) + R1-GAN(자체 실사 패치) | — / **약 41** | 4K 0.1–0.3 s / 0.3 GB | 2a |
| N6 | **F_φ-S** | **0.6M**. 잔차 GNN, MP 3, latent 64, k=8, z8 | 입자 그래프 → 잔차 가속도. **top-k 실현에만**, 100 Hz | Garage L-3D + 시뮬 | 3×10¹⁷(낮은 MFU) / 8 | 1만 노드 1–2.6 ms/스텝 | 2b |
| N7 | **Reader-S** | 26M = R-Flow 3M(RAFT-S식) + R-Kpt 18M(HR-lite, ImageNet 초기화 없음) + R-Seg 4M + R-Event 1M(시각 + 오디오) | 영상 → flow, 2D 키포인트 25, 동적 마스크, 사건 | 우리 렌더(FlyingStuff, D0) + BSRT + MakeHuman(CC0) 합성 인체 + COCO kpt(D2, 상업 경로는 Garage 라벨) | 2–3×10¹⁸ / **115–150** | 1080p 30–80 ms / 8–10 GB | 2b– |
| N8 | **Denoiser-S** | 1M. OIDN식 U-Net | 저spp + albedo + normal → 디노이즈(잔차 크기 제한) | 2만 쌍(16 vs 1024 spp, **내장 디노이저 OFF**) | 렌더 15–37 + 학습 8 | 4K 0.15–0.3 s | 2a(RT 무거운 샷) |
| N9 | Expr-S | 12M. 시간축 xfmr | mel → 블렌드셰이프 52 + viseme | **Dot-Face** 캡처(점 마커 + 폰 3대 삼각측량 + 최소자승 솔브, D1) | — / 10–20 | 실시간 | 2c |
| N10 | TTS-S + Voc-S | 20M + 14M, 규칙 기반 자모 G2P | 텍스트 → 음성 | 라이선스를 확보한 한국어 단일 화자 녹음 | — / 200–400 | 실시간 | 2c 후반 |
| — | PCGD-0 | 약 300 파라미터(NN 아님): 재질쌍 30 × {e~Beta, μ~LogNormal, compliance, OU 토크} | 접촉 → 확률적 파라미터 | Garage L-3D + 오디오, SMC-ABC | CPU / 0 | — | 2a– |
| — | ID-Screen, R-Lift | 0 (고전: 역동역학 + 접촉 QP / 삼각측량·BA·PnP) | — | — | CPU / 0 | 4096개 4–8 s (CPU16) | 2a– |

선례는 MDM(약 18–25M [2차], 2080 Ti 1장 3일), Learned Motion Matching, GNS, OIDN, RAFT(5.3M, 합성 데이터로 from scratch)다. 모션, 물리, 디노이징은 원래 작은 모델의 영역이다. 비디오 모델은 외관과 의미까지 한 몸에 넣느라 커진 것이다.

### 5.5 교사 어댑터 (엔진 교사 6.15B 대비)

| 어댑터 | 크기 | 비율 | 언제 |
|---|---|---|---|
| G-Adapter (M4) | 25M | 0.41% | P1 의미 패스(조건부) |
| Cine/Texture LoRA (A1) | 20–30M | 0.3–0.5% | 선택 |
| Chrono-LoRA r16 | 14.7M | 0.24% | P2(E2 실패 시) |
| fake-score LoRA r32 | 29.5M | 0.48% | P2 MAN-S DMD |
| identity LoRA r8 | 7.4M/캐릭터 | 0.12% | P2c. **Soul ID 출력으로 학습 금지** |
| SR-LoRA r32 (A2, attn + MLP) | 약 70M | 1.1% | 조건부(CT3) |

규칙: **동시 장착 합 ≤2%(123M).** SR-LoRA를 쓸 때는 나머지를 순차로 장착한다. Soul에는 어댑터가 없다.

### 5.6 합계와 2×T4 학습 시간 재계산 [v2.1]

| 묶음 | 모델 | 파라미터 | 1세대 학습 FLOPs | T4-h | 2×T4 벽시계 |
|---|---|---|---|---|---|
| P1 코어 (조건부·확장 제외) | M1–M3, M5–M7, M9–M14 | 146.5M | 7.0×10¹⁸ | 165–180 (+ CamFM-CRM 프리뷰 렌더 18. X7의 프리뷰와 같은 렌더이고 여기서만 계상) | 3.8–4.1일 |
| G-Adapter (조건부, E3) | M4 | 25M | 1.9–5.8×10¹⁸ (1–3 epoch, 동결 교사 역전파가 지배) | 36–110 | +0.75–2.3일 |
| BRP (`P1-확장`) | M8 | 42M | 2.9×10¹⁷ | 7–15 (3D conv MFU 낮음) | +0.15–0.3일 |
| **P1 합계** | | **213.5M** | **9.2–13.1×10¹⁸** (90일 예산의 약 4–6%) | **226–323** | **4.7–6.7일** |
| P2 2a–2b | N1–N8 | 104M (MoPri-S 16M) → 120M (32M) | 약 6×10¹⁸ + 교사 경유 연산 약 2.5×10¹⁸ | 약 286–353 (Chrono-LoRA 15–25, N8 렌더 15–37 포함) | 6.0–7.4일 |
| P2 2c 추가 | N9–N10 | +46M | — | Expr-S 10–20, TTS·Voc 200–400 | 2c 시점 |
| 교사 어댑터 | A1, A2, Chrono, fake-score, identity | 동시 장착 ≤123M | — | A1 17–45, A2 약 20, fake-score는 MAN-S에 포함, identity 캐릭터당 5–10 | — |
| **전체 (P1 + P2 2c까지, 어댑터 제외)** | | **약 380M** (fp16 0.76 GB, 엔진 교사의 약 1/16) | P1 + P2 2a–2b 약 1.8–2.2×10¹⁹ (**90일 예산의 약 8–9%**) | P1 + P2 2a–2b 1세대 **512–676** | **10.7–14.1일** |

- **1 T4-h** = 4.3×10¹⁶ FLOPs(12 TF) ~ 5.4×10¹⁶(15 TF). 소형 모델은 12 TF로 계상했다.
- **FLOPs 몫(약 8–9%)보다 T4-h 몫(512–676, 90일 용량의 12–16%)이 큰 이유:** 소형 모델은 커널 런치, LayerNorm, optimizer 비중이 커서 MFU가 낮고, G-Adapter와 MAN-S는 교사 경유 연산이 지배한다.
- **TF(u)의 "카탈로그 전체 약 100 T4-h"와의 차이:** TF는 ICX-LBO 1세대를 8×10¹⁷(20억 토큰)로 잡았다. 정본 TexelFM-35는 CB 기준 1,000만 샘플(4.8×10¹⁸, 세대당 약 90 T4-h)로 보수적으로 계상한다. 1세대를 TF 규모로 줄이면 P1 합계가 약 75 T4-h 줄어든다.
- 90일 배분(§11.5)의 소형 모델 학습 650 T4-h = P1 1세대 226–323 + 핵심 2–3세대(TexelFM gen2·gen3 약 180, MRH·MassingFM 재학습 약 30) + sweep·ablation 약 120. P2 1세대의 대부분은 Phase 2 시드 버킷(470)에 들어간다.

### 5.7 Soul 방화벽의 영향과 승인 시 해금 우선순위 [v2.1]

**원칙: 이 카탈로그의 어떤 모델도 Soul 출력으로 학습하지 않는다.** Soul이 주는 것은 세계(콘텐츠), 최종 프레임, 사람 경유 프리셋뿐이다. 아래 표는 승인 S가 오면 무엇을 어떤 순서로 해금할지 미리 정한 것이다.

| 모델 | Soul로 학습하면 얻을 것 | 지금 (Plan A / A+) | 승인 S 시 해금 순위 |
|---|---|---|---|
| CamFM-CRM (12M, 비생성 평가기) | **프레이밍 취향(축 8)**: 프레임 안의 프레임, 전경 레이어링, 타이트 크롭, 대칭·소실선. 물리 소관이지만 공짜가 아닌 축 | 금지. v1 PR·Wipe 보상 + 전경 레이어링 항 + 합성 선호 + 사람 쌍(우리 렌더만, 수제 프리셋) | **S1 ①** |
| LookPrior-S (신규 후보, 약 2M, 비생성) | 프롬프트 → θ_look(물리 파라미터) 자동 예측 | 0 파라미터 프리셋 검색(사람 측정 라이브러리) | S1 ② |
| MRH-Aes 헤드 (≤2M, MRH 본체 동결, 비생성) | Soul 앵커로 보정된 미감 점수 | 금지(Soul 계보 쌍 제외) + Teacher-DCA | S1 ③ |
| CamFM-Intent (20M, 궤적 생성 prior) | Soul 구도의 카메라 경로 취향 | 금지. 합성 Toric·오픈무비 궤적 | **S2 ④** (픽셀 없는 생성형임을 밝히고 별도 협의) |
| MassingFM (24M, 레이아웃 생성 prior) | 개념 레이아웃, 스케일 연출(신수 스케일 등) | 금지. 스케일 연출 레시피(규칙) + 감독 + (A) 사람 전사 | S2 ⑤ (같음) |
| MRH 기하·재질 헤드 | Soul 스틸 판독 정확도 | 금지(판독은 추론만, A+) | **요청하지 않는다**(본체 24M이 S 상한을 넘고 가치가 낮다) |
| TexelFM, Texel Bank, T-LPIPS-S, TAE-T, LatentDown/Up-S, IPM-S, Glyph-S | 미세 대역 | 금지 | **해금하지 않는다.** 미세 대역은 엔진·물리 소관이고 Soul은 2K 상한이라 줄 것이 적다 |
| identity LoRA (P2c) | Soul ID 캐릭터 일관성 | 금지(Soul ID는 개념 아트에만) | 별도 협의 |
| 어떤 이미지 생성기든 | — | 금지 | **요청하지 않는다**(약관 정신, 경쟁 우려 차단) |
| P2 모델 전부 | — | 무관(픽셀을 안 보거나 엔진 교사 판정을 쓴다) | — |

### 5.8 접착제 가설: 왜 수십M이면 되나

| 지식 | 이미 가진 쪽 | 비용 |
|---|---|---|
| 재질, 형태, 의미, 사진다운 외형 | 빌린 엔진 교사 | TFU |
| 개념, 미술 생활감, 인물·서사 | 빌린 미감 교사 Soul | 크레딧, 학습 기여 0 |
| 촬영 취향(미감 축 1–8) | 물리 카메라·조명·필름 + 사람이 측정한 프리셋 | 파라미터 0 |
| 빛 수송, 렌즈, 센서 | 렌더러, 광학 수식 | 파라미터 0 |
| 강체, 유체, 파괴 | Warp, MuJoCo, MPM, FLIP | 파라미터 0 |
| 벽돌, 잎, 레이아웃 문법 | 절차 코드 | 파라미터 0 |
| 물체 크기(metric) | 수작업 크기 사전(약 500 클래스) | 파라미터 0 |
| 다시점 일관성 | 세계 W 하나(T1, T2) | 정의상 |

- 우리 모델이 배우는 것은 **원천들 사이의 사상(mapping)**뿐이다. 인수분해는 각 모델의 입력 분포를 구성상 좁힌다. band-iso가 footprint를 균일하게 하고, 스케일이 미터로 주어지고, 조명은 intrinsics로 분리되고, 카메라는 이미 알려져 있다. 불변성을 학습이 아니라 구조로 공짜로 얻는 셈이다.
- **TinyStories 효과 [부품]:** 좁고 깨끗한 분포에서는 10–30M 모델도 유창해진다.
- **정직한 약점:**
  - 범용 소형 생성기는 약하다(DiT-S/2 33M의 FID 68.4).
  - 오픈 도메인 T2I를 처음부터 학습하는 것은 불가능하다(MicroDiT 1.16B는 우리 90일 예산의 2–3배).
  - 그러므로 **미감은 빌린 두 교사와 물리 카메라의 연방 몫이고**, 수십M 모델은 좁은 연산자만 맡는다. v2.0의 "미감은 오직 빌린 교사의 몫"을 이렇게 고쳐 쓴다.
  - 합성 데이터만으로 학습한 깊이 모델은 일반화가 나쁘다(DA-V2 저자들의 지적). 그래서 MRH에는 실사 MVS 차선이 필수다.

---

## 6. 데이터 엔진 "주조소" (사전학습 모델 없이, Soul 없이)

### 6.1 원칙

> **"기하는 코드에서, 외형은 교사에서, 라벨은 구성에서. 미감은 데이터로 들어오지 않는다."**

- 한 샘플은 다섯 겹으로 되어 있다: (절차·센서 GT 기하) ⊗ (엔진 교사 photoreal 외형) ⊗ (교사 내부 특징, 모든 t) ⊗ (결정 곡선 대역 유효 마스크) ⊗ (계보 태그).
- 이것을 만들려면 렌더러, WNF, 결정 곡선 보정, 솔버, 절차 생성기가 모두 있어야 한다. 그래서 H100을 가진 경쟁자도 컴퓨트로는 살 수 없다. **데이터가 해자다.**
- 모델은 4일이면 다시 학습할 수 있는 소모품이다. 데이터는 Merkle 주소 샤드로 영구히 쌓인다.
- **[v2.1] 주조소는 Soul 없이 돈다.** Soul 승인이 거절되거나 약관이 바뀌어도 데이터 자산은 한 바이트도 오염되지 않는다.

### 6.2 허용 소스와 금지 소스

| 종류 | 소스 |
|---|---|
| 에셋 (모델이 아니라 데이터) | ambientCG, Poly Haven(텍스처·HDRI·모델), MatSynth CC0 부분, GSO, MakeHuman(CC0), OFL 폰트, Blender 오픈무비(CC-BY) |
| 절차 | 자체 문법(RoomGen, StreetGen, 한옥, 을지로, 바), Infinigen(장면 3–5k + 에셋, 지오메트리만), 노드 재질 그래프, Warp GPU ProcGen, Toric 촬영 문법 |
| 자체 캡처 | 미러리스/DSLR **Bayer RAW** + 매크로(교차편광), HDR 브래킷 + ArUco/ChArUco 장면 300–500개, raw LiDAR 점, IMU(폰, GoPro GPMF), VCam 퍼포먼스, Garage Gold(240fps + 오디오, P2), Dot-Face(P2c) |
| 사람 라벨 | 2AFC 선호(Soul 계보 태그 `SOUL`/`SOUL-DERIVED`/`SOUL-INSPIRED` 자산이 든 쌍과 경쟁 API 프레임이 든 쌍 제외), 전문가 패널, COCO kpt(D2, 연구 경로), BAPPS(라이선스 확인), 업로더가 단 태그 |
| 공개 다시점 | ScanNet++, DL3DV, NYU는 **평가 전용 홀드아웃**(라이선스, 연구용) |
| **금지 [v2.1]** | Soul 출력(`SOUL`), Soul 픽셀에서 계산한 모든 것(`SOUL-DERIVED`), 기본값으로 `SOUL-INSPIRED`(Q17), D3 데이터셋, NC 데이터셋의 학습 사용 |

### 6.3 파이프라인

```
[Tier-1 CPU 주조]  자체 문법(numba) + Infinigen(지오메트리) + CC0 에셋 → scene program = 라벨
                   (Kaggle: Warp GPU ProcGen으로 GPU1 유휴 시간에)
[Tier-C 캡처]      RAW · IMU · VCam · (P2) Garage/Dot-Face/멀티캠
      │ COLMAP 4.2 SIFT + GLOMAP + Caspar BA + PatchMatch MVS + TSQ 피팅  → GT-classical-real
[GPU1 래스터]      우리 렌더러: G-buffer(depth, normal, albedo, sem, flow, inst) + 셰이딩 (AI 디노이저 OFF)
[GPU0 교사 Tier-2] Realify/운송(SDEdit t∈스냅[0.2,0.7], 2–3 NFE) / 실사 노이즈 탭(1 NFE) / band-iso 타일 / 견본 / X1-slim
      │ Feature-Tap Bus: Frozen PCA-128 int8 탭 → 링버퍼(prov 검사) → 헤드 온라인 학습 (기생). soul=1 호출은 Tap Bus off
[Tier-3 솔버]      WSSC-T4 → W* → chart 페이지 → TexelFM, MRH(≤25%), MassingFM (래칫, 엔진 트랙 세계만)
[Tier-H 사람]      선호 쌍, 전문가, 2AFC 감사
[Tier-P 물리]      CPU MuJoCo / MuJoCo Warp / Warp MPM·FLIP → 사건 → EPD-S, Feas-S, Chaos Bank
      ▼
Merkle 주소 WebDataset 샤드 + 라벨 계보 장부(Lineage) ── prov 검사 ──✕── soul_vault (격리, 학습 계정 마운트 불가)
```

**핵심 기법: Realify + 탭 동시 추출 패스** [중, 교사 운송 조합]

합성 렌더를 엔진 교사로 SDEdit하는 호출 한 번에서 두 가지를 동시에 얻는다. (i) 기하 GT가 결정 곡선 유효 대역 안에서 보존된 사진 같은 이미지, (ii) 그 패스의 교사 중간 특징(MRH 학습 입력)이다. 실사 RAW는 VAE로 인코딩해 노이즈를 더하고 1 NFE만 돌려 탭을 얻는다(생성 없음).

```
절차 장면 GT ── raster ──> x_proc ── SDEdit(교사 INT8, t∈스냅[0.2,0.7], 2–3 NFE) ──> x'
실사 RAW + MVS GT ── VAE enc ──> z ── 노이즈(t) ── 1 NFE ──> taps     (생성 없음)
라벨 유효 대역 = {o : D(o,t) ≤ 0.1}        # X1 결정 곡선이 라벨 증명서
손실 = Σ_{o∈유효} ‖Π_o(pred − GT)‖ ;  저대역 불일치 > τ → 폐기
```

### 6.4 하루 생산량

| 생산물 | 방법 | 하루 생산량 | 90일 계획 |
|---|---|---|---|
| 레이아웃 장면 | 자체 문법(장면당 50 ms) | 약 170만 / vCPU-일 (16 vCPU면 약 2,700만) | 300만 |
| Infinigen 장면 | 방 하나에 CPU 8–13분 [검증] | 약 2,000 / 16 vCPU | 3–5k |
| 합성 GT 렌더 | 우리 렌더러 1024², 0.1–0.3 s | 30–80만 / T4-day (CPU 장면 공급이 상한) | 10–15만 |
| Realify + 탭 | 교사 INT8 512²(768²) 2–3 NFE + PCA | 약 4.4만(768²는 약 2만) / T4-day | 15만 (약 80 T4-h) |
| 실사 노이즈 탭 | RAW + MVS GT → VAE enc → 1 NFE × t 2개 | 약 8만 NFE / T4-day | 6만 장 (약 36 T4-h) |
| band-iso 교사 타일 | 솔브 부산물 + 768² 공장 | 9.8천–1.26만 뷰(141만 64² 타일) / T4-day | 300만 타일 |
| 견본 | 512² 4 NFE | 약 2만 / T4-day | 학습 2만 + 세계별 |
| W\* 세계 | WSSC-T4 | 밤 14 h × 2장 → 10–14개(G0, 2.0 T4-h), 약 16개(G3, 1.8 T4-h) | 300–400 (도전: TF(u) 약 1,360) |
| CC0 재질 피라미드 | 다운샘플 | 사실상 무제한(CPU) | 고유 4–5k 재질 × 옥타브 |
| RAW 매크로 | 미러리스 + 매크로 | 1인당 500–1,500장/일 | 2–5만 |
| 실사 MVS 장면 | 촬영 + COLMAP/GLOMAP + PatchMatch | 1인당 약 8장면/일, CPU 수 분–1시간/장면 | 300–500 |
| IMU | 6대 동시 기록 | 6–40 기기-시간/일 | 300 기기-시간 |
| 카메라 궤적 | Toric 문법 / Blender .blend / GLOMAP | 수백만/일 / — / 약 800 CPU-h 총 | 200–500만 + 1만 클립 |
| 지각 라벨 | 교사 특징 거리 @256² | 약 17–28만 쌍 / T4-day | 20만 쌍 |
| 글리프 | 폰트 × 렌더러 × 운송 | 무제한 | 200만 |
| 물리 롤아웃 (P2) | CPU MuJoCo / Warp | 약 10¹⁰ env-step/일 | Chaos Bank 10⁶ |

- **데이터 SDEdit은 INT8로 돌린다**(E0 (c) 통과 시). FP16(약 2.5만/T4-day)으로 돌리면 Realify만 약 145–170 T4-h가 되어 §6.5 합계는 약 360–420, §11.5 교사 데이터 공장 버킷은 420이 된다(차액은 예비에서).

### 6.5 데이터 장부 [신규 운용] (주간 리뷰에서 실측과 비교)

| 항목 | GPU T4-h | CPU-h | 규칙 |
|---|---|---|---|
| 합성 장면과 GT 렌더 | 20 | 1,500 (Infinigen 장면·에셋) | 디노이저 OFF |
| Realify + 특징 | 80 | — | PCA-128 int8 캐시 |
| 실사 기하 (MVS) | 10 | 300 | SIFT 전용 |
| 실사 노이즈 탭 | 36 | — | — |
| 지각 라벨 | 17–29 | — | — |
| 레이아웃·캡션 임베딩 | 4 | 42 | TE는 인코딩에만 |
| 카메라 궤적 | — | 800 | SIFT 전용 |
| X1-slim | 3–16 | — | — |
| band-iso 타일 공장 | 51 | — | 솔브 부산물로 대체 가능 |
| 견본 학습셋 (2만 × 512² 4 NFE) | 24 | — | 측량 모드 |
| 재생성·버퍼 (운송 768² 비중 확대, 실패 재실행) | 50–60 | — | 주간 리뷰에서 회수 |
| **합계** | **약 295–330** (§11.5 버킷 330과 상단 일치. TAE 타깃 7 T4-h는 §5.3 M1에 계상) | **약 2,700** | — |

원안에 숨어 있던 비용(Cycles GT 15만 장 1,250–5,000 T4-h, Infinigen 300만 장면 40–65만 CPU-h, 학습 손실 안의 6B 교사 역전파)을 **약 1/20로 줄였다.** Soul 생성은 이 장부에 없다(데이터가 아니므로).

### 6.6 계보: 순도 등급, 라벨 태그, SOUL 태그, Lineage CI, Sensor Purity

**순도 등급.**

| 등급 | 정의 | 예 | 시스템 경로 학습 |
|---|---|---|---|
| D0 | 센서, 모캡, 시뮬, 절차 GT | 우리 렌더·시뮬, Garage, CMU 원본, 100STYLE BVH, RAW | 허용 |
| D1 | 고전 알고리즘 라벨 | SIFT-COLMAP/GLOMAP 포즈, PatchMatch 깊이, 오디오 onset, ChArUco, Dot-Face 솔브 | 허용 |
| D2 | 사람 주석 | 2AFC(Soul 계보 태그 자산·경쟁 API 프레임이 없는 쌍), COCO kpt, Act-from-Text 쌍, 업로더 태그 | 허용 |
| D2-T | 엔진 교사 파생 | 운송, 견본, MRH 자가 라벨, W\*(엔진 트랙) | 허용. **배치의 GT(D0+D1) 비율 ≥55%, 자가 라벨 ≤25%** |
| D3 | 다른 학습 모델이 개입한 라벨 | SpatialVID pose·depth, Motion-X, **AMASS/HumanML3D(SMPL)**, DataDoP(MonST3R), ASR 자막, ARKit, 폰 뉴럴 ISP | **금지.** 평가와 연구 ablation 전용 |
| **DS [v2.1]** | **닫힌 미감 교사 출력과 그 파생** | `SOUL`, `SOUL-DERIVED`, (기본) `SOUL-INSPIRED` | **금지(약관).** 콘텐츠·평가 전용. 승인 S 범위의 해금 브랜치만 예외 |

**라벨 계보 태그 (TF(u) 목록 + v2.1).** `GT-proc`, `GT-sensor`, `GT-classical-real`, `classical`, `teacher-transport(t, 대역)`, `teacher-swatch`, `teacher-self(MRH vK)`, `solver-W*(gen k)`, `human`, `sim`, `asset-3p`, `sensor-ISP-ML`(엄격 모드 제외), **`SOUL`, `SOUL-DERIVED`, `SOUL-INSPIRED`**.

- **Lineage Ledger CI** [신규 운용]
  - 모든 데이터, 라벨, 가중치에 계보 DAG 해시를 붙인다.
  - 시스템 경로 학습 아티팩트의 조상 노드로 허용하는 것은 엔진 교사, 우리 학습, 고전 알고리즘, 센서, 사람뿐이다. 다른 것이 섞이면 빌드를 실패시킨다.
  - 자동 검사: import 금지 목록(`eval/` 패키지), Blender 디노이저 설정, COLMAP feature_type과 vocab tree, 체크포인트 해시, 폰 영상 메타데이터.
  - **[v2.1] Soul 검사:** 조상에 `SOUL`·`SOUL-DERIVED`가 있으면 실패. `SOUL-INSPIRED`는 학습 아티팩트의 조상이면 실패(기본값). soul_vault 경로가 학습 잡의 마운트 목록에 있으면 실패. Exhaust 학습 분할에 soul=1 엔트리가 있으면 실패. 사람 평가 쌍 빌더가 Soul 자산 해시 또는 `prov.soul ≠ none`인 렌더를 포함하면 실패. TRT 캘리브레이션 캐시, Feature-Tap Bus 링버퍼 덤프, 교사 VAE 쌍 캐시의 조상에 `SOUL`·`SOUL-DERIVED`가 있으면 실패. `SOUL-INSPIRED`가 있어도 실패(기본값, Q17). 주 1회 카나리아를 학습 샤드 빌더와 링버퍼에 주입하는 테스트. 비디오 렌더 경로에 SoulGloss 호출이 있으면 실패(CT37).
- **Sensor Purity Test** [조합]
  - 1 kHz 이진 카운터 LED 10개 + 피에조 클릭 보드($30–80).
  - 보간된 프레임은 카운터 상태가 섞여 보이므로 검출할 수 있다.
  - 같은 보드로 rolling shutter readout 시간과 장치별 AV 오프셋을 잰다.
  - HDR과 AI 기능을 끈 수동 앱만 쓴다.
- **RAW 원칙:** 미세 대역 데이터는 Bayer RAW + 고전 디모자이크로만 만든다. **폰 뉴럴 ISP가 그린 텍스처는 TexelFM이 배워야 할 바로 그 대역에 남의 신경망 환각을 심는다.**

**몰래 들어오는 사전학습물 통합 목록 [v2.1 복원: TF(u) 15곳 + RR(u) 15곳 + Soul].**

| # | 위치 | 몰래 들어오는 것 | 판정 | 대체·규칙 |
|---|---|---|---|---|
| 1 | v1 WSSC `SelfCaption` | 캡셔닝 VLM | 위반 | 프로그램 자기지식 캡션 + MRH sem 최근접(TE로 한 번 임베딩한 명사구 사전) |
| 2 | v1 $\mathcal L_{ocr}$ | OCR 모델 | 위반 | Glyph-S. SYMBOL은 MSDF라 구성상 정확 |
| 3 | v1 효도 손실 λ·LPIPS | VGG/AlexNet | 위반 | T-LPIPS-S, Laplacian 피라미드 L1 + 교사 특징 거리 |
| 4 | v1 `sem`(SigLIP/DINO), Droste 판정 CLIP | 사전학습 비전 인코더 | 위반 | MRH sem(TE 공간), class id, Teacher-DCA. 참고: SDXL·Z-Image에는 CLIP **이미지** 인코더가 없다 |
| 5 | COLMAP 4 ALIKED/LightGlue, `vocab_tree_flickr100K`, ORB-SLAM3 `ORBvoc` | 학습된 특징·매처, 외부 이미지 k-means 코드북 | 엄격 모드 위반 | SIFT 고정, sequential 매칭, 우리 이미지로 vocab tree |
| 6 | 폰 ISP(Deep Fusion, Smart HDR), ProRAW 계산 합성 | 입력에 ML 처리 흔적 | 계보 오염(회색) | 미러리스 RAW. 폰은 IMU 전용. `sensor-ISP-ML` 태그 |
| 7 | ARKit/ARCore VIO | 블랙박스 ML 가능 | 회색 | OpenVINS / VINS-Mono, ArUco PnP. 라이브 프리뷰 입력은 Q8 |
| 8 | SDXL-Lightning, Hyper-SD, DMD2 LoRA, sdxl-vae-fp16-fix, TAESD/TAEF1, Z-Image Fun-ControlNet-Union | 제3자 학습 파생물 | 위반 | 공식 Turbo, 자체 VAE 수리, TAE-T, G-Adapter. DMD2는 NC라 이중 탈락 |
| 9 | 교사 TE(Qwen3-4B)로 scene graph 생성 | 사실상 LLM | 생성은 위반, 임베딩은 회색(Q2) | **R-TE 임베딩 전용**: DiT가 소비하는 층 출력만, 디코딩 금지, 결과 캐시 |
| 10 | ScanNet++, DL3DV, FlyingThings/Chairs, Hypersim, AMASS/HumanML3D, AVA, DataDoP, CameraBench 학습 | NC·연구 약관, D3 | 상업 위반 위험 | 평가 전용. 학습은 자체 캡처, 우리 렌더러 flow, CMU. Hypersim은 약관 확인 전 제외 |
| 11 | Blender Cycles OIDN/OptiX AI 디노이저, Infinigen 렌더 기본값 | 학습된 디노이저 | 위반 | 끄고, 래스터 + 교사 운송. Infinigen은 지오메트리만 |
| 12 | 3DGS 빠른 초기화(InstantSplat 등의 DUSt3R/MASt3R/단안 깊이) | 사전학습 기하 모델 | 위반 | COLMAP SfM 점 초기화 + FastGS / gsplat MCMC |
| 13 | v1 DTC p̂_T의 단안 깊이, Σ_u의 RAFT/UFM | 사전학습 비전 모델 | 위반 | MRH defocus/intrinsics/sem 헤드, DIS/TV-L1 |
| 14 | P2: GVHMR/WHAM/MANO/FoundationPose/MoSca/SAM 3D Body/SMPL-X, VLM 판정, 사전학습 TTS, IC-Light, Difix3D+, TRELLIS/Hunyuan3D | 사전학습 인체·포즈·음성·VLM·재조명·3D | 위반 | Reader-S + 멀티캠 삼각측량, 자체 스켈레톤 + MakeHuman, 사람 판정, 녹음 → TTS-S, 물리 재조명, CC0 에셋 |
| 15 | 평가 모델(GeCo의 VGGT/UFM, MEt3R, LPIPS, ArcFace, OCR TPR)이 best-of-N 선택, 자동 HP 탐색, CamFM-CRM 라벨, early stopping으로 새는 경로 | 평가 모델 | 누수되면 위반 | **R-EVAL**: 별도 프로세스, 결과는 사람이 읽는 보고서로만 |
| 16 | Nunchaku 사전 양자화 Z-Image INT4 | 같은 교사의 제3자 변환물 | **Q7 판정 전 금지(기본안)** | 자체 양자화(우리 보정셋, prov 검사) |
| **17** | **[v2.1] Soul 출력의 학습 사용**: Soul 앵커 쌍(과 `SOUL-INSPIRED` 프리셋 렌더 쌍)으로 MRH-Aes·CamFM-CRM 학습, Soul 판독 호출의 탭·VAE 쌍이 링버퍼·보정셋으로 새는 것, Soul 세계를 래칫에, θ_soul 자동 피팅을 학습 라벨로, Soul ID로 identity LoRA, Soul 스틸 Exhaust를 MRH 학습에 | 닫힌 모델 출력 | **약관 위반** | Soul 방화벽(§3.7.3). 승인 S 범위만 예외 |

### 6.7 저장

| 모드 | 운용 |
|---|---|
| 상시 | 데이터 약 450 GB, PCA 탭 캐시 300 GB, Ledger cold 200 GB, 세계 라이브러리 150–300 GB, 벤치 클립 약 200 GB → **1.3–1.5 TB**(NVMe 4 TB 권장) |
| Kaggle | 생성 즉시 소비(기생 학습), 저장은 5만 개 이하, 디스크 20 GB. 산출물은 HF Hub 비공개 저장소나 GCS로 비동기 업로드 |
| **soul_vault [v2.1]** | Soul 원본 스틸과 메타(프롬프트, 모델, 버전, 날짜, 플랜, 경로, 약관 스냅샷 해시), 수 GB. 학습 잡 계정은 마운트할 수 없다. 콘텐츠 렌더 잡과 사람 평가 잡만 읽는다 |

---

## 7. STRATA-lite, 렌더러, 카메라 스택

### 7.1 원칙 여섯

1. 병목은 메모리가 아니라 교사 시간이다. 4K 방이 1 GB 안팎이라 남는 메모리로 교사 호출을 아낀다(Ledger).
2. **정본은 explicit이다.** latent 정본은 디코더(학생)가 있을 때만 의미가 있다. `z_slot`(int8×16/cell)은 Phase 1.5에 쓰려고 예약해 둔다.
3. 하드웨어가 공짜로 주는 기능을 쓴다. BC 디코드, 바이리니어 필터, RT 삼각형 교차, NVENC.
4. 미세 레벨은 Router가 나눈다. 재질 셀은 TexelFM, 의미 셀은 교사다.
5. **레벨 안의 모든 생성은 Jacobi다.** 그래서 두 장으로 공간을 나눠 완전히 병렬화할 수 있다. **비회귀 조항이 곧 하드웨어 효율이다.**
6. **commit-before-shoot** [신규]. 렌더 경로에는 생성 연산이 하나도 없다. v2.1: Soul 광택도 렌더 경로에 없다. 스틸 광택은 렌더 뒤의 별도 산출물이고, 세계 경유 광택은 Shoot 전에 CoW로 다시 연 POLISH 라운드(S8.5)에서 commit까지 끝낸다.

### 7.2 레벨 체계 (전역 고정 $h_l=64\cdot2^{-l}$ m, 64 m를 넘는 lot은 l<0)

텍셀 피치는 $\tau_l=h_l/8$이다. 4K S35, 35mm에서 픽셀 footprint는 $\phi(z)=1.85\times10^{-4}z$다.

| l | 셀 | 텍셀 | 4K에서 필요한 최대 거리 | 1024² 뷰 폭 | 결정 주체 |
|---|---|---|---|---|---|
| ≤6 | ≥1 m | ≥12.5 cm | — | ≥128 m | bedrock: hero(또는 Soul 개념 아트, A+) + MassingFM + 의미 패스 |
| 7–10 | 50–6.25 cm | 6.25 cm–7.8 mm | 340–42 m | 64–8 m | 교사(시네마틱 뷰 lockstep) |
| 11–12 | 3.1–1.6 cm | 3.9–1.95 mm | 21–10.5 m | 4–2 m | 교사(Grid-Locked OTBI), 가난선 기본 2 mm |
| 13–14 | 7.8–3.9 mm | 0.98–0.49 mm | 5.3–2.6 m | 1–0.5 m | Router: 의미 셀은 교사, 재질 셀은 Spec-commit → TexelFM |
| 15 | 1.95 mm | 0.24 mm | 1.3 m (8K 2.6 m) | 0.25 m | TexelFM (8K 의미 셀만 교사) |
| ≥16 | ≤0.98 mm | ≤0.12 mm | ≤0.66 m | — | TexelFM |
| PSH | SYMBOL(MSDF), POINT, CURVE, GLINT, CARRIER | — | 레벨 게이팅 면제 | — | DSL, 수식. **글자는 SYMBOL이 소유한다.** 엔진 교사가 현판에 그린 가짜 한자는 Glyph-S가 벌점을 주고 SYMBOL로 교체한다 |

Soul이 소유할 수 있는 대역(A+)은 최종 프레이밍에서 Soul 스틸 2K의 Nyquist 이하뿐이다. 4K 납품에서 화면 높이를 채우는 물체라면 대략 한 옥타브 아래까지다. 그 위는 위 표의 주체가 소유한다.

### 7.3 자료구조와 TSQ

```
World (STRATA-lite v2.1)
├─ meta    : seed u128, h_0=64 m(고정), lot extent, L, l_g, mode, teacher_id, quant_mode, engine_hash,
│            merkle_root, toolchain pins, router_map, massing{shell, slots},
│            prov{soul: none|inspired|derived|authored, soul_assets:[hash, model, ver, date, plan, path, tos_hash]},
│            look{θ_look id, preset 출처}
├─ blocks[l]: sparse hash (l,i,j,k) → Block(16³) {occ, tsq_range, page_range, state, own, unc, hash}
│            own u16: 저술자(엔진 교사 뷰 / TexelFM / 절차 / 사람 / SOUL(A+ 저술 셀))
├─ charts  : 세계축 차트(xatlas), Fold 그래프, chart ↔ TSQ 페이지 UV 매핑 (차트 = TSQ 페이지의 UV 묶음)
├─ tsq     : 32 B SoA {μ f16×3, q oct32, s_uv f16×2, thick, opacity, edge_n(soft↔hard), level, flags,
│            L_d RGB9E5, F0/rough/metal, sem u16, own u16, unc u8}
├─ pages   : 8×8 텍셀/TSQ. BC7(L_d log-delta) + BC5(normal/height) = 128 B. HDR 발광(네온) 페이지는 BC6H
├─ vol     : 3D Gaussian + SGGX (잎·털·연기 aggregate) · strand: BCSDF (털·이끼 솜털 근거리)
│            iface: DIEL SDF bricks 8³ f16 (IOR, Abbe)
├─ structs : SYMBOL/POINT/CURVE/GLINT/CARRIER(ψ) · far_field: octahedral HDR + Hosek–Wilkie
│            (+ 원경 매트 페인팅 슬롯: Soul 콘텐츠 자산 가능, prov 기록)
├─ lights  : world_lights | shot_rigs | atmosphere{β, g} · swatches: (m, l) → 512² 견본 + intrinsics
├─ wnf     : 0 B (Philox(seed, l, cell, k) 즉석 계산) · ledger: Exhaust 인덱스(soul 비트 포함)
└─ z_slot  : (예약) int8[16]/cell
```

**TSQ (Textured Soft Quad)** [조합: 2DGS, BBSplat, Textured Gaussians]

- **형태:** 방향을 가진 사각형이다. 가장자리는 $\exp(-(|u|^n+|v|^n))$로 정의하며 n=2면 부드러운 surfel, n≥8이면 단단한 쿼드가 된다. 8×8 페이지를 하드웨어 바이리니어로 읽는다.
- **T4에 맞는 이유:**
  - 텍셀당 2 B라서 같은 고주파를 가우시안으로 낼 때보다 60–120배 압축된다.
  - 불투명 코어가 삼각형 2개라서 RT 코어가 closest-hit로 가속한다. 3DGRT식 정이십면체 프록시 + any-hit은 1세대 RT 코어에서 느리다.
  - 피팅이 싸다.
  - USD, Unreal, WebGL로 바로 export할 수 있다.
- **예외:** VOL(SGGX), STRAND(BCSDF), DIEL(SDF), 아바타(P2c).

### 7.4 세계별 메모리

| 세계 | 레벨 | TSQ | 텍셀 | RAM/디스크 | GPU |
|---|---|---|---|---|---|
| Oracle 방 (실촬영, 4K) | ≤14 | 2–5M | 1.5–3×10⁸ | 0.5–1.0 GB | 상주 |
| 을지로 네온 (밤, 안개, f/1.4) | ≤15 (의미 셀) | 2–4M + POINT 5k + SYMBOL 300런 | 0.8–2×10⁸ | 0.3–0.7 GB | 상주 |
| 거울 복도 | ≤15 | 3–6M | 2–4×10⁸ | 0.6–1.2 GB | 상주 |
| 불릿타임 바 + 와인잔 | ≤16 + DIEL | 2–4M + SDF 8×10⁷ voxel | 1–2×10⁸ | 0.5–0.9 GB | 상주 |
| 이끼 산맥 (극접사, 0.3 × 0.3 m) [v2.1] | ≤19 (TexelFM) + STRAND | 0.5–1.5M + 솜털 strand 0.2–1M | 0.5–1×10⁸ | 0.2–0.5 GB | 상주 |
| 신수 계곡 (야크, 원경 매트) [v2.1] | −2…15 | 3–6M + 털 SGGX/strand | 1–3×10⁸ | 0.6–1.5 GB | 상주 |
| 같은 방 8K | ≤15 | 6–12M | 4–10×10⁸ | 1.2–2.8 GB | 상주 |
| 10분 원테이크 집 | ≤15 | 25–50M | 1–2×10⁹ | 3–6 GB | 경로 주변 1.5–3 GB 스트리밍 |
| Powers of Ten 8K | −6…22 | 수요 원뿔 5–15M | 2–5×10⁸ | 1–2 GB | 옥타브 2–3개 스트리밍 |

### 7.5 렌더러 선택

| 후보 | 결정 |
|---|---|
| gsplat (2DGS/3DGS, 3DGUT) | **primary 베이스이자 유일한 미분 래스터.** 1–2주차는 2DGS 그대로(왕복 v0), 3–6주차에 TSQ 텍스처 커널로 확장. 3DGUT는 왜곡 렌즈, rolling shutter, 부드러운 볼륨용 |
| OptiX (TSQ 코어 삼각형) | **secondary.** 거울, 크롬, 유리 픽셀(5–30%)만, 깊이 ≤2. 7주차 이후. 컨테이너에 OptiX가 없으면 **Warp `wp.Mesh` BVH 폴백**(3–10배 느리지만 프레임당 +30–80 ms) |
| 평면 거울 재귀 래스터 | 거울 복도의 주력. 깊이 12, 총비용은 프레임의 2배 이하. **초록 무한 터널** [신규 데모 디테일]: 반사마다 $R(\lambda)\approx0.9\,T_{glass}(\lambda)^2$를 곱하면 철분 흡수 때문에 점점 초록빛으로 어두워진다. 비디오 모델은 이 물리를 모른다 |
| DIEL SDF 레이마칭 | 와인잔, 분산(Abbe), 4K 프레임당 2–4 s |
| 3DGRT primary | **폐기.** 3DGS보다 느리고 1세대 RT 코어의 any-hit 비용이 크다 |
| Blender Cycles (디노이저 OFF) | 정답 레퍼런스(ΔJOD)와 합성 데이터 일부 |
| 메쉬 + BC export → Vulkan/WebGL | 웹 뷰어(시청자 기기가 렌더하므로 우리 서빙 비용 0) |

**핵심 장치.**

- **dual-cut LOD**(v1 C7, 16×16 타일 단위, +20–40%), 공분산 대수(v1 식)
- **Defocus Pyramid** [조합: Kraus–Strengert'07]: CoC가 8 px를 넘는 atom은 $2^{-k}$ 해상도 레이어에서 렌더한다. fill-rate 폭발을 막는다. v2.1: 빈티지 처방의 공간 가변 PSF(주변부 스월·번짐)도 이 피라미드에서 반경별 커널로 처리한다.
- **Longuet-Higgins Split** [조합]: 회전 블러는 셔터 중간 자세 오버스캔 렌더 1회 + 호모그래피 $H(\tau)=KR(\tau)R_{mid}^\top K^{-1}$ 적분으로 정확히 계산한다. 평행이동 블러만 공분산으로 넣는다. rolling shutter는 행마다 $\tau(y)$를 쓴다. 1 kHz 손떨림 블러가 거의 공짜로 정확해진다. **A/B가 본 "시간의 흔적"(달리 인의 방사형 블러)은 이 평행이동 블러의 확장 초점(FOE) 구조 그대로다.**

### 7.6 렌즈·센서·필름 스택 (학습 파라미터 0개, 모든 손잡이가 물리 단위)

| 블록 | 방법 | 4K 비용 |
|---|---|---|
| 렌즈 처방 → 출사동 LUT | Kolb'95, 특허 처방(데이터). cat-eye, 왜곡, breathing, 색수차 | 오프라인 1회 |
| **빈티지 처방 라이브러리** [v2.1] | 옛 렌즈 특허 처방(Petzval 계열, 초기 더블 가우스, 단층 코팅). 필드 커버처와 비점수차가 **스월 보케와 주변부 번짐**을 물리적으로 만든다. 단층 코팅은 베일링 글레어와 Hullin 고스트를 키운다 | 오프라인 1회 + 주변부 가변 PSF 10–20 ms |
| **애너모픽** [v2.1] | 2× 스퀴즈 처방, 타원 보케, 수평 플레어. 2.39:1 화면비(축 1) | 출사동 LUT에 포함 |
| DOF | $c(z)=\frac{f^2}{N(z_f-f)}\frac{\lvert z-z_f\rvert}{z}$ + Defocus Pyramid | 20–60 ms |
| 보케 스프라이트, Hullin 고스트, FFT veiling glare, 회절 스타 | POINT/GLINT scatter, 2회 반사 경로, $\lvert\mathcal F\{\text{조리개}\}\rvert^2$ | 30–65 ms |
| **원편광(CPL) 필터** [v2.1] | 하늘: 단일 산란 Rayleigh 편광도 $\mathrm{DoP}(\gamma)=P_{max}\frac{\sin^2\gamma}{1+\cos^2\gamma}$ ($\gamma$ = 시선과 태양 사이 각, $P_{max}$≈0.7–0.8), 편광 방향은 산란면에 수직. 필터 축 φ_f에서 $I'=\tfrac{T}{2}I\,[1+\mathrm{DoP}\cos2(\phi-\phi_f)]$. 수면·유리 반사는 첫 정반사의 Fresnel s/p 성분으로 처리(브루스터각 근처 억제). 태양과 90°인 하늘이 가장 짙어진다. 필터 투과 T≈0.35–0.45만큼 노출 보정 | 2–5 ms |
| 셔터 | LH-split + rolling | 10–40 ms |
| 센서 | Poisson-Gauss, full-well, ADC, Bayer + Malvar–He–Cutler | 5–10 ms |
| 룩 | log → 3D LUT 65³, 할레이션, **Newson 그레인 타일 뱅크(유일한 프레임별 랜덤)**, 게이트 위브, 비네팅 | 10–15 ms |
| **합계** | — | **75–190 ms**(항목 합, 블록 병렬·퓨전 시 60–120 ms) (8K는 0.25–0.5 s). 빈티지·애너모픽·CPL을 쓰는 샷만 +10–25 ms(8K +40–100 ms) |

광학 효과는 전부 수식이다. **OIDN과 OptiX AI 디노이저는 금지**하고 고전 à-trous/SVGF를 쓴다.

**룩 프리셋 θ_look [v2.1].** 한 프리셋은 다음 물리 파라미터의 묶음이다: 화면비·스퀴즈, 센서 포맷, 초점거리, 카메라 거리 힌트, f-stop, 초점, 조리개 날 수, 셔터 각, ND, CPL 각, EV, WB, 하이라이트 롤오프, LUT(Soul HEX 팔레트 기준색), 그레인 스톡, 할레이션, 비네팅, 렌즈 처방 id, 대기 β·g, 태양 고도·방위, 림·키·필 리그 템플릿, 프레이밍 템플릿.

- **출처 세 가지.** (i) 우리가 손으로 만든 프리셋, (ii) Soul 레퍼런스를 사람이 측정한 프리셋(`SOUL-INSPIRED`, Plan A), (iii) Soul 스틸에서 자동 역추정한 θ_soul(`SOUL-DERIVED`, A+).
- **역추정 원리(A+).** Soul 스틸을 관측 제약으로 쓸 때 그레인, 심도, 그레이딩, 비네팅은 세계가 아니라 카메라가 만든 것으로 설명해 세계 밖으로 뺀다(DTC와 렌즈 식별성 원리, v1 §3.4 ShotDNA). 그렇게 추정한 θ_soul을 프리셋으로 저장하면 같은 룩을 모든 각도에서 재사용한다. 스틸 한 장의 미감이 오빗 전체의 미감이 된다.
- **라이브러리.** v0 ≥30개를 4주차(D30)에 벤치용으로 동결하고(CT38), D60까지 60여 개로 늘린다. 리그 스왑 릴(D28)의 "룩 50종"은 수제 프리셋(출처 i) 20 + 이 라이브러리 v0의 `SOUL-INSPIRED` 30이다.
- **버전 내성.** 프리셋은 물리 파라미터라 Soul 버전이 바뀌어도 그대로 남는다(R28).

### 7.7 렌더 시간 (T4 1장, 시네마 품질, (추정))

| 프레임당 | 1080p | 4K | 8K |
|---|---|---|---|
| primary (dual-cut) | 50–100 ms | 0.2–0.45 s | 0.8–1.8 s |
| secondary RT (20% 픽셀, 깊이 2) | 5–15 ms | 15–60 ms | 60–240 ms |
| 렌즈·센서·필름 | 15–30 ms | 60–120 ms | 0.25–0.5 s |
| **합계** | **0.07–0.15 s** | **0.28–0.63 s** | **1.1–2.5 s** |

| 10초 샷 벽시계 (짝·홀 프레임 분담, I/O +20%) | 24 fps | 60 fps | 120 fps |
|---|---|---|---|
| 1080p | 10–22 s | 25–54 s | 50–108 s |
| 4K | 0.7–1.5분 | 1.7–3.8분 | 3.4–7.6분 |
| 8K | 2.6–6분 | 6.6–15분 | 13–30분 |

빈티지·CPL 샷은 위 값에 4K +3–4%, 8K +4–8%를 더한다.

- **fps는 생성 비용을 늘리지 않는다.** 4K120 10초에 GPU 전력은 8–18 Wh(호스트 포함 16–37 Wh)로 스마트폰 1–2회 충전 분량이다.
- 라이브 프리뷰는 1080p 절반 품질에서 25–45 fps가 나온다. T4 한 장으로 VCam 라이브 촬영을 할 수 있다(ArUco PnP + IMU).
- **저장 원칙: "프레임 대신 세계와 경로를 저장한다"** [운용].
  - 테이크 = (Merkle 루트, Γ, θ_look, Soul 자산 해시)이고 언제든 비트 단위로 똑같이 다시 렌더할 수 있다.
  - 리뷰용은 4K ProRes 422 HQ(30 s에 약 14 GB).
  - 8K EXR(DWAA)은 납품용 최종 테이크만 만든다.
  - AOV는 4K 프록시나 24fps 키프레임만 무손실로 저장한다. 8K 전 프레임 AOV는 10초에 600–840 GB라서 금지한다.

### 7.8 카메라: 리그, CamFM, DP 탐색, 연출 레시피, Corman–Shannon

- **리그(학습 0):** v1 리그 14종을 1 kHz ODE로 CPU에서 시뮬한다(질량, 감쇠, 홀로노믹 제약, 진자, PID). 의도 층은 min-jerk, 2/3 power law, 지연 피드백이다. **학습이 필요한 것은 사람 손뿐이다.**
- **DP 탐색(샷당 1–4분):**
  1. 후보 1,000개를 CamFM-CRM의 해석 통계와 리그 시뮬만으로 채점한다(렌더 없이 수 초).
  2. 상위 50개를 360p 48 프레임으로 프리뷰한다(각 0.3 s 이하).
  3. Gaffer는 조명 ≤8개를 CEM 200회로 찾는다(평가 1회 = 렌더 + MRH-Aes @512²). θ_look의 리그 템플릿이 초기값이다. 최종 상위 4개는 MRH-Aes ⊕ Teacher-DCA 앙상블로 재순위한다(약 +4 TFU). MRH-Aes 단독 선택은 금지한다(R40).
- **DP-0 보상 v2.1.**
  $$R=\mathrm{CamFM\text{-}CRM}+\beta_1\mathrm{PR}+\beta_2\mathrm{Wipe}+\beta_3\mathrm{RevealBeat}+\beta_4\mathrm{Clearance}+\beta_5\log p_{intent}+\beta_8\mathrm{FL}+\beta_9\mathrm{FiF}+\beta_{10}\mathrm{Sc}-\beta_6\mathrm{CWS}-\beta_7\mathrm{Risk}-\lambda\,\mathrm{TFU}(\Gamma)-\infty\cdot\mathrm{collision}$$
  - **FL (전경 레이어링) [v2.1, FD (3)]:** 프레임 가장자리 띠 안에서, 주피사체보다 가까운 가림체(처마, 기둥, 잎)가 차지하는 면적을 그 가림체의 디포커스 정도로 가중해 시간 평균한다. 주피사체 saliency를 X% 이상 가리면 벌점이다. 렌더러가 깊이와 saliency를 정확히 주므로 학습이 없다.
  - **FiF (프레임 안의 프레임):** 카메라에서 주피사체로 가는 광선 묶음이 MassingFM shell의 개구부(창, 문, 처마 아래)를 지나면 가점.
  - **Sc (스케일 연출):** 아래 레시피를 의도로 고른 경우의 제약 충족도.
  - PR과 Wipe는 v1 그대로다. Soul의 구도 취향은 이 항들과 대부분 겹친다. 가중치는 사람 선호(우리 렌더만)로 맞추고, Soul 구도로 학습하는 것은 CamFM-CRM만 S1이다(§3.7.4).
- **스케일 연출 레시피 [v2.1, FD (8)(12)].**

| 레시피 | 목적 | 처방 | 데모 |
|---|---|---|---|
| **역-미니어처** | 작은 것을 거대한 풍경처럼 | 극접사(배율 ≥0.5×), 카메라 높이 = 피사체 스케일의 1–3배, **지평선 제거**(틸트 다운, 또는 배경을 역광 헤이즈로 날림), 얕은 심도의 흐림이 대기 원근처럼 읽히도록 초점을 "봉우리"에 둔다, 스케일에 맞춘 느린 이동(실제 1–3 cm/s가 산맥 위 비행으로 읽힌다), 빈티지 주변부 번짐, 역광 솜털 림 | 이끼 산맥 FPV |
| **거대화** | 보통 것을 신수 스케일로 | 메트릭 세계에서 대상 스케일을 키우고 **보통 크기 기준물**(사람, 보통 야크 무리)을 배치, 로우앵글 광각(16–24 mm, 카메라 높이 0.3–1 m), **대기 원근**(볼류메트릭 산란 β를 거리에 비례), 깊은 심도(얕은 심도는 미니어처 신호라 피한다), 느린 카메라, CPL로 짙은 하늘, 역광 림 | 신수 오빗 |

  얕은 심도는 원래 "미니어처 신호"다. 역-미니어처에서 이 신호를 거꾸로 쓰는 조건(초점 위치, 역광 헤이즈)은 X-LA 이끼 arm D에서 검증한다. **오빗과 풀백은 스케일이 진짜임을 증명한다.** 비디오 모델은 카메라가 돌면 스케일 착각을 유지하지 못한다.
- **Corman–Shannon Demand v2** [신규]: 점 주파수가 아니라 대역 평균으로 평가한다. 점 주파수로 계산하면 jinc의 0점($c\nu\approx1.22$)에서 수요를 잘못 생략한다.
  $$\mathcal I_l(x)=\max_t\frac1{|B_l|}\int_{B_l}\log_2\!\Big(1+\frac{|H(\nu;x,t)|^2S_l(x)}{N_{shot}+N_{read}+N_{grain}}\Big)d\nu,\quad H=\mathrm{MTF}_\theta\cdot\tfrac{2J_1(\pi c\nu)}{\pi c\nu}\cdot\mathrm{sinc}(\nu v\Delta t)\cdot\mathrm{sinc}(\nu)\cdot e^{-\beta z}$$
  - 수요 집합은 $\{(l,b):\max_b\mathcal I_l\ge\epsilon\}\cup$ 승격 구조이고, $\epsilon=0.05$–0.1 bit/sample이다. $\mathrm{MTF}_\theta$는 θ_look의 렌즈 처방(빈티지 주변부 포함)을 쓴다.
  - **Corman Optimizer:** $\max\ \mathrm{CamFM\text{-}CRM}+\beta_a\,\mathrm{MRH\text{-}Aes}-\lambda\,\mathrm{TFU}(\Gamma,\theta,\mathcal L,\beta)$. MRH-Aes 단독으로 최적화하지 않고 Teacher-DCA 앙상블로 쓴다(R40). λ는 감독 UI의 **예산 슬라이더**다. 탐색 변수에는 스모크 머신(β), 조명, 노출, 조리개가 포함된다.
  - **벤치 공정성:** 의도가 렌즈나 조명을 지정하면 Corman은 그 값을 바꾸지 못한다. 결과는 항상 on과 off를 둘 다 보고하고, 낮·맑음 프롬프트를 의무로 포함한다.
  - **반증 가능한 예측(X-C):** 대규모 세계(거리, 원테이크, 8K)에서 T4-h가 3–10배 줄면서 SP 비열등(Δ ≥ −3%p). 방 하나 4K에서는 절감이 20–40%에 그칠 것이다. 이것도 정직하게 보고한다.

### 7.9 결정론과 Proof-of-Shot

- **고정소수점 정렬 리덕션:** 닫힌형 sync는 int64로 누적하고 (텍셀 id, 뷰 id)로 정렬한 뒤 segment-reduce한다.
- **엔진 동결:** TRT plan, timing cache, 캘리브레이션 캐시를 아티팩트로 버전 관리하고, 엔진 해시와 드라이버 버전을 Merkle 키에 넣는다.
- **솔버 경로 설정:** `cudnn.deterministic=True`, `benchmark=False`. 렌더 MC seed = Philox(포즈 해시, 픽셀). forward는 정수 atomic만 쓰고 (깊이, atom id)로 stable sort한다.
- **Proof-of-Shot** [신규]
  - 영상과 함께 (Merkle 루트, ShotScript, θ_look, 렌더러·툴체인 pin)을 공개한다.
  - 같은 아키텍처와 같은 툴체인이면 바이트 단위로 같고, 아키텍처가 다르면 1 LSB 이내다.
  - "왕복 비트 동일"이 체리피킹이 아니라는 것을 시청자가 직접 검증할 수 있다.
  - **[v2.1] Soul이 닿은 작품:** Soul 생성 자체는 닫혀 있어 재현할 수 없다. 그래서 Soul 자산의 해시와 플랜 단계를 함께 공개하고, Proof-of-Shot은 "그 자산 바이트가 주어졌을 때"의 렌더 결정론을 보증한다고 명시한다. 스틸 광택 산출물은 영상과 구분해 표기한다. Soul이 닿은 작품의 θ_look(`SOUL-INSPIRED`·`SOUL-DERIVED` 프리셋)은 Q13 회신 전까지 해시만 공개한다.

---

## 8. Phase 1 승리 조건, 2×T4 단계 목표, 쇼케이스, 벤치

### 8.1 최종 승리 조건 (v1 §7.1 그대로, 낮추지 않는다)

| 항목 | 목표 |
|---|---|
| **빈 사분면** | SP ≥ 50% (vs Wan 3.0, Seedance 2.5) **이면서** GeCo ≤ 0.03 |
| TP (엔진 교사 스틸 대비) | ≥ 45% |
| VP T1 일반 / 전문가 | ≥ 65% / ≥ 75% |
| VP T2, T4± | ≥ 80% |
| VP T3 롱폼 | ≥ 90% |
| LCE | 정확히 대칭인 경로에서 비트 동일. 근사 복귀는 LPIPS ≤ 0.01 |
| TPR / MAT / CI | ≥ 0.99 / ≥ 90% / ≤ 실사 + 5%p |
| SFR_prior / DR_rel / TPI | 전 대역 ≥ 0.9 / ≤ 1.2 / 실사 바닥의 1.5배 이내 |
| MTF50 @8K | ≥ 0.25 cy/px |
| 카메라 튜링 | 2AFC ≤ 55% |
| **[v2.1 진단, 목표 아님]** | SP_S(E1-S), TP_S, X-LA 격차 닫힘 비율 g_cam·g_dir·g_con. 진단 기준선 TP_S\* = 1 − SP_S(게이트 아님, Q6) |

**승리 주장 규칙 [v2.1].** 승리 조건은 FREEZE-TAG의 자동 트랙(Engine, Hybrid-A, Hybrid-A+) 어디서 달성해도 우리 시스템의 승리다. 다만 어느 트랙인지 항상 표기하고, **공개 재현 주장은 Engine과 Hybrid-A 트랙으로만** 한다(CT38). Hybrid-A 공개는 Q13 회신 후다. Directed 데모는 Directed라고 따로 표기한다.

### 8.2 2×T4 단계 목표

| 시점 | 목표 |
|---|---|
| **Day 3** | [v2.1] Higgsfield 메일 발송(Plan A/A+/S), Soul 방화벽 CI v0와 soul_vault 가동 |
| **Day 18** | 엔진 교사 잠금(E1-E와 ES로만 판정. 적색이면 황색 의무, §3.4) + **E1-S로 SP_S 확정, TP_S\* 도출(보고 항목)** |
| **Day 30** | E0–E6 통과 또는 업그레이드 확정. 왕복 비트 동일. 4K 프레임 ≤0.6 s. X2 파일럿 VP ≥60%. l=11–12 수직 슬라이스 HER ≥0.8, GeCo ≤0.04, OIE ≤0.05, seam 2AFC ≤60%, TP ≥35%. TexelFM 2AFC ≤60%. MRH 실사 AbsRel ≤0.12. CamFM-Hand v0(IMU 40–60 기기-시간) 카메라 튜링 ≤60%. **[v2.1] X-LA0(이끼) 3분해 1차 결과, 룩 프리셋 라이브러리 v0(≥30) 동결, 카나리아 테스트 통과** |
| **Day 60** | X2 VP ≥70%(v1). C1 생성 세계 20개에서 GeCo ≤0.04, HER ≥0.8, OIE ≤0.05(v1). **SP ≥35%, TP ≥40%(중간 목표).** TexelFM 최미세 2AFC ≤55%, native OIE = 0. 라이브러리 ≥100. 세계당 Mode A ≤2 T4-h. CamFM-Hand 카메라 튜링 ≤55%. **[v2.1] X-LA 기록형 3 프롬프트에서 g_cam 측정(H-LA1 판정), TP_S를 TP_S\*의 80%와 비교해 보고 [보고 항목: 게이트·승리 조건 아님, Q6], 이끼 산맥 FPV 데모** |
| **Day 90** | **v1 그대로:** 범주 하나 이상에서 SP ≥45%와 GeCo ≤0.03을 동시에. T2 최강 경쟁 모델 대비 VP ≥70%. 데모 릴 3편 + "Shot on 2×T4". P2 SMC 충족률 ≥90%. **T4 추가:** TFU/세계 ≤520(G3), 세계 계획 T4-h ≤1.8, 세계 벽시계 ≤55분, 미니 벤치 사전 등록 및 실행. **[v2.1] 트랙별(Engine / H-A / H-A+) 결과 분리 보고, TP_S를 TP_S\*와 비교해 보고 [보고 항목: 게이트·승리 조건 아님, Q6], X-LA 30 프롬프트 완료, 하이브리드 쇼케이스 1편(우주인과 말)** |

### 8.3 쇼케이스 실행 순서

정렬 기준은 (구조적 승리의 크기) ÷ (T4-h × 리스크)다.

| 순서 | 데모 | Day | 세계 T4-h | 렌더 T4-h | Soul (Plan) | 필요한 것 | 비디오 모델이 구조적으로 못 하는 이유 |
|---|---|---|---|---|---|---|---|
| 1 | **왕복 v0**: 실촬영 방, 정방향·역방향 차이 맵 | 14 | 1–1.5 | 0.05 | — | 2DGS 폴백 허용 | 돌아오는 길에 세계를 다시 상상한다 |
| 2 | **리그 스왑 6종 × 룩 50종** | 28 | 0 | 0.5 | 룩 50종 = 수제 프리셋 20 + 사람 측정 `SOUL-INSPIRED` 30(라이브러리 v0) (A) | CamFM-Hand, 룩 프리셋 라이브러리 v0 | 테이크마다 세계가 바뀐다 |
| 3 | **"Shot on 2×T4"**: 8K120 정지 세계 슬로모션 [신규] | 28 | 0 (+TexelFM l=15) | 0.5–1 | — | TexelFM v0 | 네이티브 8K120을 낼 수 없다 |
| 4 | **이끼 산맥 FPV** [v2.1]: 역-미니어처, 이끼 솜털 역광, 빈티지 렌즈, 마지막에 풀백으로 숲 바닥임을 드러냄 | 45–55 | 1–2 | 0.3 | 룩 레퍼런스 (A) | TexelFM, STRAND/SGGX, 빈티지 처방, X-LA0 결과 | 카메라가 날면 스케일 착각이 무너지고 디테일이 끓는다. 풀백은 다중 스케일 세계가 하나여야 한다 |
| 5 | **을지로 네온 (Corman 쇼케이스)**: 밤, f/1.4 랙 포커스, 한글 SYMBOL, 보케 볼, 플레어 직선, 웅덩이 반사 | 55–60 | 4–6 | 0.3 | 개념 아트 (A) | 엔진 교사 잠금, G-Adapter, MRH, MassingFM(거리), Glyph-S | 한글이 끓고, 보케가 깊이와 무관하고, 플레어가 떠다닌다 |
| 6 | **왕복 v1 + "하룻밤 1,000 테이크"** | 60 | 0 | 6.6 (1080p 10초 × 1,000) | — | — | 1,000번 생성하면 세계 1,000개가 나온다 |
| 7 | **거울 복도 + 초록 무한 터널** | 62 | 2–4 | 0.3 | — | 평면 재귀, OptiX | 거울 속 세계를 다시 상상하고, 철분의 초록을 모른다 |
| 8 | **한옥 마당: 프레임 안의 프레임** [v2.1] | 60–70 | 3–5 | 0.3 | 개념 아트 (A) | 전경 레이어링·FiF 보상, SYMBOL 현판(정확한 한글·한자) | 처마 너머 시차가 일관되지 않고, 현판 글자가 끓는다 |
| 9 | **경로 이식 릴 T4±** | 65 | (벤치) | 0.2 | — | — | 공정한 A/B |
| 10 | **Corman 슬라이더**: 같은 장면을 0.1–10 T4-h로 [신규] | 70 | 2 | 0.2 | — | — | 우리만 할 수 있는 메시지 |
| 11 | **우주인과 말: 2022 → 2026** [v2.1 하이브리드 쇼케이스]: "astronaut riding a horse"(DALL·E 2/Imagen 시절의 상징적 프롬프트)를 멈춘 세계로. 해 질 녘, 갈기와 먼지가 공중에 멈춘 말 위의 우주인, 360° 오빗 | 70–80 | 6–10 + Hero FX(먼지·모래 MPM 정지) 2–4 | 0.5–1 | 개념 아트 = 관측 제약 (A+) / 사람 전사 (A), θ_look(2.35:1, 85–135 mm, f/2, 낮은 태양 림, 로우키) | MassingFM·BRP(사구), STRAND(갈기), Hero FX, 룩 프리셋, IP 점검 | 오빗하면 말의 해부가 흐르고, 갈기가 숨 쉬고, 먼지가 움직이고, 얼굴이 바뀐다. 경쟁 모델에 같은 프롬프트 + "frozen time orbit"으로 A/B |
| 12 | **1999 → 2026 불릿타임 헌정**: 와인잔이 깨지는 순간 정지 | 75–85 | 4–8 + Hero FX(Voronoi + CFL FLIP, CPU 4–12 h) | 0.5–1 | — | DIEL, Hero FX | 정지를 요청하면 숨을 쉰다 |
| 13 | **체호프의 컵**: 10분 원테이크 | 85–90 | 6–10 (+polish 4) | 1–2 | — | TexelFM, CamFM-Intent, MassingFM | extend할 때마다 drift가 쌓인다 |
| 14 | **신수 오빗** [v2.1, `P1-확장`]: 거대한 흰 야크가 작은 두 사람에게 고개를 숙인 계곡, 보통 야크 무리, 뿔의 후광, 털 림, V자 계곡, CPL 하늘, 대기 원근 | 90+ | 6–10 | 0.5 | 개념 아트 (A/A+), 원경 매트 (A) | 거대화 레시피, STRAND/SGGX 털, BRP 지형, CPL, 볼류메트릭 β | 오빗이 스케일이 진짜임을 증명한다. 비디오 모델은 강제 원근이 돌면 무너진다 |
| 15 | 직접 날아보기 (웹 뷰어) | 60+ | 0 | 0 | — | export | 상호작용 자체가 불가능하다 |
| 16 | Powers of Ten | 90+ | 5–15 | 0.2 | — | TexelFM, 의미 효도 | 스케일 전환에서 녹는다 |
| 17 | 빛만 흐르는 세계 / 매직아워 | 90+ | +3 | 0.3 | 룩 레퍼런스 (A) | MRH 법선 | 조명 상태가 없다 |

- **Day-90 릴:** 왕복, 을지로 네온, 불릿타임 헌정 + "Shot on 2×T4"(v2.0 그대로) + **하이브리드 쇼케이스 1편(우주인과 말)**. 이끼 산맥 FPV는 Day 60 중간 공개 후보다.
- **미루는 것:** 불릿타임 속 사람과 군중 얼굴은 2c로 미룬다. 엔진 교사 하나로 다시점 정체성을 지키기 어렵다. "어부"는 사람 연기라 2c이고, P1에서는 X-LA 스틸 비교에만 쓴다(MakeHuman 정적 포즈, 얼굴 비노출, §3.8.6). 우주인의 얼굴은 바이저 반사율을 높여 부분 노출로 제한해 정체성 리스크를 줄인다. Vision Pro 양안 광택은 `P1-확장`.
- **IP 점검 절차 [v2.1, FD (14)].** Soul이 닿은 상업 공개물은 모두 거친다. **범위:** 데모·키 아트 전부와 Hybrid-A+ 벤치 개념 아트 전부(640장). 벤치는 사람 체크리스트 전수 + 역이미지 검색 표본 20%로 한다.
  1. 프롬프트 위생: 실명 IP, 캐릭터, 작가, 브랜드 이름을 넣지 않는다.
  2. 사람 검토: 알려진 캐릭터, 로고, 유명 작품과 닮았는지 체크리스트로 본다(신수 야크가 기존 애니메이션 캐릭터와 닮을 위험 등).
  3. 사람이 운용하는 역이미지 검색(평가 전용 도구, 시스템 경로 밖).
  4. 상업 데모는 법무 서명.
  5. 결과를 soul_vault 메타에 기록한다. 걸리면 재생성(웹 'IP 재생성·대체' 100회, 부족하면 예비 900에서)하거나 개념을 바꾼다.
- **Soul 크레딧 표기 [v2.1 사후].** 발송본은 데모에 "Look by Higgsfield Soul" 크레딧을 넣겠다고 제안했다(부록 B 발송 기록). 크레딧은 Soul이 닿은 데모(개념 아트, 키 아트, 세계 경유 광택)에만 붙이고 Engine 트랙 결과물에는 붙이지 않는다. 이 문구는 룩(축 1–8)을 물리 스택이 소유한다는 연방제(§3.8.2)와 R38에 맞지 않는다. 영업팀과 협의할 때 바로잡는다(Q16).

### 8.4 FREEZE-TAG on 2×T4

**운영 원칙.**

1. **세계 하나가 의도 N개를 모두 찍는다**(Location-lite Π). 우리 생성 비용은 프롬프트 수에 비례하고, 경쟁사는 프롬프트 × 의도 × seed에 비례한다.
2. T1·T2는 **텍스트(또는 첫 프레임) 전용 모드**다. WorldScript와 사람 개입은 금지한다. 데모는 Directed라고 따로 표기한다.
3. Corman on/off를 둘 다 보고한다. 자해 카테고리(불, 연기, 털)도 공개한다.
4. 평가 모델(VGGT, UFM, MEt3R, LPIPS, OCR, ArcFace)은 R-EVAL 방화벽 뒤에 둔다. VGGT가 FP16에서 오버플로하면 FP32로 돌린다(4.8 GB, 들어간다).
5. T4± 경로 추출은 고전 BA(COLMAP SIFT)가 먼저다. VGGT 폴백은 벤치 트랙에만 쓰고 표기한다.
6. **라이브러리 = 벤치** [운용]. 4주차에 프롬프트를 사전 등록하고, 5–12주차 라이브러리 세계를 그 프롬프트로 만든다. 시스템이 바뀌면 Merkle 델타 솔브로 바뀐 대역만 다시 푼다.
7. **[v2.1] 트랙 분리(CT38).**

| 트랙 | Soul 사용 | 자동성 | 공개 재현 | Soul 호출 (미니 벤치) | 래칫 데이터 |
|---|---|---|---|---|---|
| **Engine** | 없음 | 완전 자동 | Kaggle에서 재현 가능 | 0 | 예 |
| **Hybrid-A** | 4주차에 동결한 사람 측정 룩 프리셋 라이브러리(`SOUL-INSPIRED`)만. 프롬프트별 사람 개입 없음 | 완전 자동(규칙 검색) | 프리셋 파일을 공개하면 재현 가능(공개는 Q13: Higgsfield 확인 후) | 0 | 아니오(`SOUL-INSPIRED`, Q17). Engine 세계의 CoW 브랜치라 원본 Engine 세계만 래칫에 남는다 |
| **Hybrid-A+** | 프롬프트별 MCP 개념 아트 = 관측 제약 | 완전 자동 | 닫힌 모델이라 재현 불가. 자산 해시 공개 | 640 (A+ 승인 시) | 아니오 |

| 항목 | 미니 벤치 (Day 90) | 전체 벤치 (Day 90 이후) |
|---|---|---|
| 구성 | C1(밤의 실내·거리 + 텍스트·네온) + C2(건축 실내 + 거울·크롬) × 40 프롬프트 × 의도 4(push-in, orbit, 왕복, 핸드헬드 워크) × seed 2 | 240 프롬프트 × 의도 12, 약 500쌍 + bullet-time |
| 우리 세계 | **Engine** 160개 × 1.8–2.5 = **290–400 T4-h**(계획 350, 벽시계 5–8일). **Hybrid-A**는 Engine 세계의 bedrock과 재질 commit을 재사용하고 L0 재bake, 수요 차 레벨 CoW, S8–S10만 다시 돈다: 160 × 0.35–0.7 = **56–112 T4-h(계획 90)**(§4.4). 재사용이 실패하면 단독으로 짓고 차액 최대 약 320은 예비에서. **Hybrid-A+**는 bedrock부터 다르므로(개념 아트 관측 제약) 별도 세계 160 × 2.0–2.5 = 320–400, A+ 승인 시 예비에서 | 480–960개, 0.7–2.4k T4-h |
| 렌더 | 클립 640개 1080p + 4K 마스터, 약 3 T4-h | 약 12 T4-h |
| 경쟁 API | 3모델 × 640클립 = 1,920클립 × 8 s = 15,360 s × $0.05–0.4 = **$0.8–6.1k** | $5–38k |
| GeCo 등 평가 | 2,560클립 × 15–60 s = 11–43 T4-h | 약 100 T4-h |
| 사람 평가 | 능동 샘플링(BT 불확실성) 약 2,000쌍 × 3명: $1–5k + 전문가 15명 $3–6k. **TP_S 앵커 쌍 별도** | $50–80k |
| Kaggle 모드 (Engine만) | C1 40 세계(1배 소모) / 20 세계(2배) | — |

**공개 벤치 [신규 운용].** FREEZE-TAG 기준선과 우리 Engine 트랙 파이프라인이 Kaggle 무료 T4×2에서 돈다. "믿지 마세요. 직접 돌려보세요." Hybrid-A는 프리셋 파일과 함께 공개한다(Q13: Higgsfield 확인 후. 회신 전에는 Engine 트랙만 공개).

---

## 9. Phase 2 on 2×T4: 작은 뇌, 정확한 우주, 큰 탐색

### 9.1 무엇이 바뀌나

- 비디오 증인, PLAC, PC-ELBO, 진술 그래프는 **폐기**(하드 제약)한다.
- native WFM($10^{23}$–$10^{25}$)과 2천만 클립 lift도 **폐기**한다. v1 Phase 2 예산은 대부분 학습이 할 일이 아닌 곳에 쓰이고 있었다. 빌린 엔진 교사 하나에만 이미 약 4×10²³(3.4–4.5×10²³, §1.3) FLOPs가 선불로 들어가 있다.
- 남는 것: 물리는 공짜 검증기, 사람은 저대역 소유자, 이미지 교사는 정지 순간의 판사다. **v1 H7("이미지 교사 + 물리 + 실사건 prior만으로 Turing 동률")이 본선으로 승격된다.**
- 판사는 Phase 1과 같은 엔진 교사다. P2(o)의 "SANA급으로 교체" 안은 삭제했다.
- **[v2.1] Soul은 Phase 2의 판사가 아니다.** σ도 특징도 없고, 약관상 보상이나 선택 신호로 쓸 수 없다. Phase 2에서 Soul의 자리는 세트·캐릭터 개념 아트(Soul ID 포함, Plan A)와 키 아트뿐이다. identity LoRA를 Soul ID 출력으로 학습하지 않는다.
- **T4에서 물리는 상대적으로 싸다.** T4는 H100 대비 실효 텐서 FLOPs가 약 1/25, 메모리 대역은 약 1/10이다. 시뮬과 렌더는 대역에 묶이므로 학습보다 덜 손해 보고, 시뮬은 CPU에서도 돈다.

### 9.2 판사 재편: 비디오 모델 0개, 판사 넷과 사람 하나

**증인 대체 매트릭스 [v2.1 복원: P2(o) §4.1].**

| v1 증인의 임무 | 대체 수단 | 근거 데이터 |
|---|---|---|
| 저주파 의미, 연출, 안무 | **감독(ShotScript, 폰 연기)**, EPD-S(교사 텍스트 임베딩 조건), 실제 증인 검색 | D0/D2 |
| 거친 사건 타이밍(≤ f_c) | 실사건 통계 + **크로노포토 증인** | Garage Gold, 오디오, 240fps |
| 의도와 퍼포먼스 latent z | 감독 테이크 선택, 연기 lift | 사용자 영상 |
| 희귀 사건을 "상상"해 주는 공급원 | **상상하지 않고 굴려 본다.** 절차 세트피스 × SMC 탐색, Chaos Bank | 시뮬(D0) |
| 평가 | 외부 API(Seedance, Kling, Veo, Wan 3.0)를 **평가 전용**으로 쓴다. Arena와 Spectrum Report | API 출력은 학습 금지 |

**기울기 라우팅.**

| 판사 | 판정 대상 | 기울기나 선택이 흐르는 곳 | 차단 |
|---|---|---|---|
| 엔진 교사 슬라이스 (TFU) | 정지 공간의 사실감 | θ, MAN-S, intrinsics, slot. s(t)와 E는 선택만 | s(t), E, ψ |
| **크로노 증인 GICC** [신규] | **시간의 모양** (0.3–15 Hz) | EPD-S 거친 타이밍, MoPri-S 저대역(RWR, 선택) | 외관, 눈금 |
| 물리 (시뮬, 불변량 I1–I6, RGF, ID-Screen) | 불변량, 고대역, **시간의 눈금** | PCGD-0, F_φ-S, MJPC | 외관 |
| 실통계 (Garage, 오디오 간격, 모캡) | 간격과 타이밍 분포 | EPD-S(MMD, 게이지 불변 통계에만), PCGD-0 | 외관 |
| 감독 (사람) | 0–1 Hz 서사 | ShotScript 슬롯, 테이크 선택, 폰 연기 | — |
| 외부 API, Soul | 평가, 콘텐츠 | 없음 | 전부 |

### 9.3 크로노포토 증인 + BIS + GICC [신규]

**발상 [v2.1 복원: P2(o) §4.3].** 비디오 모델이 금지되면 질문이 바뀐다. "**시간을 담고 있는 사진은 무엇인가?**" 답은 이미 사진사에 있다. 에저턴의 스트로보 다중노출(튀는 공의 포물선, 1957년 우유 왕관), 마레와 머이브리지의 크로노포토, 스포츠 시퀀스 사진, 장노출 궤적과 라이트 페인팅, 사진 판독기(photo-finish). 이미지 교사는 이런 사진을 수없이 봤다. 그러니 **4D 블록을 시간이 접힌 정지 사진 한 장으로 렌더하면, 이미지 교사가 동역학을 판정한다.**

- **렌더 모드.** 고정 증인 카메라 $c_w$, 스트로브 시각 $t_k=t_0+k\Delta$, $K\le12$.
  $$X_{add}=\mathrm{Cam}\Big(\sum_k \mathcal R_{strobe}(W(t_k),c_w)\Big),\qquad
  X_{mat}=\mathcal R(W_{bg})\odot\prod_k(1-m_k)+\sum_k m_k\odot\mathcal R(W(t_k)),\qquad
  X_{long}=\tfrac1S\!\int\!\mathcal R(W(t))\,dt$$
  $X_{add}$는 에저턴식 가산 다중노출(어두운 배경, 섬광 조명, 렌더러로 물리적으로 정확), $X_{mat}$는 시퀀스 사진식 매트 합성(깊이 순서 유지), $X_{long}$은 장노출(LED 마커만 남기는 라이트 페인팅 모드 포함)이다.
- **BIS (Band-Isometric Strobe)** [신규]. v1의 band-isometric 뷰를 시간 축으로 옮긴 것이다. 노출 간 화면 변위를 교사가 결정하는 공간 대역에 맞춘다: $\Delta_b=d^*_b/\lVert v_{screen}\rVert$, $d^*_b\approx24$–$64$ px @512². 관측 가능한 시간 대역은 $f\in[1/(K\Delta),\,1/(2\Delta)]$이고, Δ = 1/30 s, 1/5 s, 1.2 s 세 조합으로 0.07–15 Hz를 덮는다. 화면 변위 $d^*_b$는 증인 카메라의 거리와 줌으로 맞춘다 [P2(u) 복원].
- **차원 해석.** 정지 사진은 미터 스케일 s와 스트로브 간격 Δ를 모른다. 탄도 궤적의 이미지는 $(x_0,\ s v\Delta,\ s g\Delta^2)$에만 의존한다. 그래서 g×κ는 $\Delta\to\Delta/\sqrt\kappa$로 흡수되고, **식별할 수 없다.**
- **GICC (Gauge-Invariant Chrono Contrast):**
  $$\Delta_\chi(h)=\mathrm{DR}_b(X(h))-\mathbb E_{w\sim\mathcal W_{inv}}\mathrm{DR}_b(X(h\circ w)),\quad \mathrm{AUC}(\mathcal W_{ctrl})\in[0.45,0.55]$$
  - 채점용 $\mathcal W_{inv}$: 소산계의 시간 역전, 구간 ease-in-out(2차 차분 일정성 파괴), 반발 결합 파괴(높이 비 ≠ 간격 비²), 진자 등시성 파괴, 보행 duty factor 이동.
  - 음성 대조 $\mathcal W_{ctrl}$: g×κ, 균일 λ. 여기서 0.5를 벗어나면 가짜 단서가 새는 것이다.
  - common random numbers로 같은 ε와 σ를 쓴다. 노출 하나하나의 외관이 같으므로 렌더 아티팩트는 상쇄되고 시간 구조만 남는다. 가산 합성은 순서에 불변이라 순열 대신 간격 왜곡을 쓴다.
  - **"눈금"은 I4, 오디오, RGF가 소유하고 "모양"은 교사가 소유한다.** 덤으로, 비디오 모델이 실제로 지는 곳(ease-in-out 부유, 포물선이 아닌 궤적)을 판사가 정확히 겨냥하게 된다.
- **사용처.** SMC 가중치, EPD-S와 MoPri-S의 RWR 보상, QBC(가설 사이에서 의견이 가장 갈리는 $(c_w,K,\Delta)$ 선택).
- **비용:** 판정 1회 = 모드 3 × σ 2 × (원본 1 + 왜곡 4) = 30 TC(512²) ≈ 7.5 TFU. INT8로 18–21 s(30 TC × 0.6–0.7 s)다. 그래서 캐스케이드 상위 k개에만 쓰고 나머지는 ChronoProxy-S가 맡는다.
- **보정 데이터:** Sensor Purity를 통과한 240fps 실영상에서 배경 median과 차분 마스크로 만든다(D1). 왜곡 negative는 **실제의 다른 프레임을 고르는 것**만으로 만든다(보간 없음). 모든 노출이 진짜 사진이고 달라지는 것은 간격뿐이다.
- **교사가 둔감하면** Chrono-LoRA(r16, 약 15M, 교사의 0.24%)를 실제 합성 약 3만 장으로 학습한다(15–25 T4-h).
- **정직한 한계와 선례.** 교사가 "보기 좋은 등간격"을 선호하는 미감 편향을 가질 수 있어 대조 점수만 쓴다. Grid Diffusion(CVPR'24)은 비디오를 이미지 격자로 생성한다. 이미지 생성기를 스트로보 합성으로 **물리·타이밍 판정기**로 쓴 선례는 찾지 못했다. 실패하면(LoRA를 붙여도 AUC < 0.65) 크로노 증인은 "모션 궤적의 미감"(히어로 선택)에만 쓰고, 시간 사실감은 RGF와 실통계로 맡긴다(v1 H7 경로).
- **덤 쇼케이스: 에저턴 헌정.** 1957년 우유 왕관에서 2026년 와인 왕관으로. 같은 블록에서 스트로보 스틸, 불릿타임, 스피드 램프를 모두 뽑는다.

### 9.4 Coverage–Precision Split v2 + ID-Screen 검증 캐스케이드 [신규 정식화 / 조합]

$$P_{hit}=1-(1-q\,r)^N,\qquad \mathfrak B(A)=\{x:\Pi_{phys}(x)\in A\},\qquad C=N(c_{gen}+c_0+c_1)+N_2c_{MJPC}+N_3c_{TFU}$$

좋은 결과의 집합 A는 얇은 다양체다. 물리 사영 $\Pi_{phys}$(MJPC 추종, 접촉 투영, SMC rejuvenation)가 그것을 두꺼운 분지 $\mathfrak B(A)$로 넓힌다. 생성기는 분지 위에 $q\ge10^{-3}$만 두면 된다. 같은 예산이면 작은 모델 × 큰 탐색이 이긴다. 정확한 검증기가 있는 영역(물리)에서만 성립하는 경제학이다. 언어 모델에는 이런 검증기가 없다. 우리에게는 뉴턴이 있다. **검증도 싸야 한다.**

| 단계 | 내용 | 대상 | 비용 |
|---|---|---|---|
| L0 | 운동학 불변량(발 미끄럼, 관통, 관절 한계, 각속도 상한), GPU 배치 | 4096 | <1 s |
| L1 | **역동역학 유령력**: `mj_inverse` + 프레임별 접촉력 QP로 RGF $p_{real}$ 분위수 순위 | 4096 × 300 프레임 | CPU 16코어 4–8 s |
| L2 | MJPC-lite(rollout 128, 레퍼런스 warm start) | 상위 8 | 개당 0.6–2.1분 |
| L3 | TPRF 교사 판정(포즈 복원력) | 상위 4 | 8–16 TC |
| **합계** | 10 s 모션 best-of-4096 | — | **6–18분** (MJPC 전수 검증이면 100–400 T4-h) |

재현율 게이트: $r\ge0.95$. 판정 캐스케이드의 L3 호출 10%는 무작위 후보에 쓴다(proxy echo chamber 방지).

### 9.5 T4 위의 물리 (추정, P2-E4에서 확정)

| 시뮬 | 규모 | 처리량 / 시간 |
|---|---|---|
| 강체 소형 | batch 4096, MuJoCo Warp | 3×10⁵–10⁶ scene-step/s |
| 강체 중형 (접촉 약 200) | T4 batch 1024 / CPU 16코어 | 2–8×10⁴ / 5×10⁴–2×10⁵ (중형에서는 결정론적 CPU가 경쟁력 있다) |
| 강체 대형 (Rube Goldberg, 200–500 바디) | batch 256, T4 / CPU | 5×10³–2×10⁴ |
| 휴머노이드 | T4 / CPU | 3×10⁴–10⁵ / 3–10×10⁴ |
| Voronoi 파편 300 | batch 64–256 | 1–3×10⁴ |
| MPM 30만 입자 | 1 | 스텝당 5–10 ms, 시뮬 1초에 1–2분 |
| **FLIP 50만, 256³** | 1 | **CFL 적응**(dt 1–2 ms)이면 시뮬 1초에 0.5–3분. dt 0.2 ms 고정이면 5–12.5분. 압력 MGPCG가 지배 |
| Eulerian 연기 256³ [복원] | 1 | 스텝당 0.1–0.2 s, 시뮬 1초에 0.5–1분 |
| XPBD 천 2만 / 헤어 1만 가닥 × 32 노드 | — | 시뮬 1초에 2–10 s / 10–30 s |

- 유리는 MPM으로 하지 않는다(E≈70 GPa, ρ≈2500, 파동속도 5.3 km/s면 dx = 1 mm에서 dt ≲ 2×10⁻⁷ s). **Voronoi 사전 파쇄(seed = 사건 마크) + 강체**로 한다.
- **Milestone SMC 입자 수 [복원].**

| 과제 | N | 지평 | scene-step (재시뮬 ×1.5) | T4 시간 | CPU16 시간 |
|---|---|---|---|---|---|
| 공, 병 (BUPS 소형) | 4096 | 3 s | 1.8×10⁷ | 20–60 s | 2–6분 |
| 도미노 50, 핀볼 | 1024 | 10 s | 1.5×10⁷ | 3–12분(뱅크 있으면 0.6–2.6분) | 1.5–5분 |
| 미니 Rube Goldberg | 256 (Chaos Bank warm start) | 15 s | 5.8×10⁶ | 5–19분 | 4–12분 |
| 와인잔 파괴 (Frozen Action) | 강체 256 → FLIP top-8 | 0.5 s | 강체 3.8×10⁵ + FLIP 8×2,500 스텝 | 강체 1분 미만 + FLIP 10–25분 | — |
| 캐릭터 접촉창 (MJPC) | 제어 스텝당 rollout 512, 예측 0.4 s, 50 Hz | 10 s 모션 | 약 1.5×10⁷ | 1.5–6분 | 5–17분 |

- **multi-fidelity:** 거시 결정 변수(충돌 속도·각도, 채운 양, 파쇄 seed, 바람)는 1/8 해상도 프록시(약 30배 저렴)와 proxy critic으로 고르고, top-8만 원해상도로 미시 seed만 바꿔 다시 돌린다. 사건 시각 제약은 강체 운동이 정하므로 해상도가 바뀌어도 유지된다. 미세한 혼돈은 Hero-Instant의 다양성으로 쓴다.
- **Chaos Bank** [조합: Many-Worlds Browsing, motion matching]: 놀고 있는 CPU로 템플릿별 10⁵–10⁶ rollout을 굴린다. 사건 그래프 시그니처와 타이밍 벡터로 kd-tree 색인하고 top-N을 SMC 초기 입자로 쓴다(스텝 5–20배 절감). 결정론적 CPU MuJoCo라 rollout당 64 B, 10⁶개가 64 MB다. 같은 뱅크가 EPD-S와 Feas-S의 데이터가 된다. **Bank-First:** 0–4주차에 먼저 만든다.
- **캐릭터 물리:** 범용 추종 정책(PHC급)은 10⁹–10¹⁰ 샘플(T4 한 장 1–3주)이 든다. 대신 MuJoCo MPC식 predictive sampling(학습 0)을 **접촉창에서만** 쓴다(PhysDiff식 사영). 학습 정책은 2d로 미룬다.

### 9.6 실제 증인과 Lift Ladder

- **Garage Gold Set** [조합]
  - 폰 3–4대 240fps, ChArUco, 박수와 오디오 상호상관으로 동기화한다. 컬러 마커, 배경차분, 삼각측량, Kabsch로 강체 6DoF를 얻는다.
  - Sensor Purity를 거친 뒤 사건 시각을 ±2 ms로 잡는다(행 지연과 AV 오프셋 보정 후, Blur-as-Measurement + 오디오 onset).
  - v0 목표: 6시간, 약 2만 사건. 장비는 $0–2k.
- **오디오 간격 증인:** $\Delta t_{ij}=(t^{ev}_i-t^{ev}_j)+\beta(d_i-d_j)+(n_i-n_j)$. 클립별 AV 오프셋이 간격에서 상쇄되고, 음원이 움직이지 않으면 거리 항도 사라진다. 연속 바운스 간격 비 $T_{k+1}/T_k=e$가 반발계수다. 범주 메타데이터("탁구공", "농구공")가 사람이 단 태그인 웹 오디오만 써서 **재질별 반발계수 분포**를 학습 0으로 얻는다. 같은 방법으로 발걸음 cadence, 드리블 주기, 문 닫힘 시간, 유리 파괴 지속시간도 얻는다. v1의 AV-Physics Timing Solver는 4D 거리가 있는 Gold Set에서 유지한다.
- **Lift Ladder** [신규 정식화]

| 단 | 내용 | 조건 |
|---|---|---|
| L-time | 사건 시각 | 단안 가능 |
| L-shape | 게이지 불변 2D 통계 | 단안 가능 |
| L-metric | metric 궤적 | 지면과 크기 단서가 있을 때 |
| L-3D | 3D 궤적 | 다시점만 |

  절대 물리량($p_{real}(F_g)$, PCGD-0)은 **L-3D에서만** 추정한다. 사이클 게이트: 재렌더 후 Reader-S로 다시 잰 오차가 τ 이하일 때만 쓴다.
- **Event-Demand Lift:** 사건 그래프 수요가 비어 있는 범주만 lift한다. 연 1e4–1e5 클립(v1은 2천만), CPU로 하루 약 9시간.
- **RGF on T4 [v1 유지]:** ghost force는 lift된 궤적에 고전 역동역학을 적용해 구한다. $p_{real}(F_g)$는 범주별로 Gold Set에서 추정하고, lift 품질이 오를 때마다 다시 추정한다.

### 9.7 외관, 숨은 면, 캐릭터, 음향

- **MAN-S via Exhaust-to-Shader** [조합]: BSRT 세탁 쌍 $(x_r,x_p)$과 같은 MAN 노이즈를 그대로 지도학습 데이터로 쓴다. 목표는 $r^*=\mathrm{HP}_{>b_y}(x_p-x_r)$다. 지도학습 3 T4-h 뒤 DMD는 5k 스텝만 돌린다. **원안 200–267 T4-h → 약 41 T4-h.** Read-Only다(알파·기하 출력 없음, $\lVert r\rVert\le\tau$). 모션블러 구간에서는 서브프레임 2–4개에만 적용한다.
- **BSRT (Band-Safe Reality Transfer):** $\sigma^*=D^{-1}(b_y{+}1,0.9)$(Turbo면 학습 σ로 스냅), $x_p=\mathrm{Dec}(\mathcal D_{\sigma^*}(\mathrm{Enc}(x_r);\xi_{MAN}))$. 렌더의 라벨(flow, 키포인트, 마스크 경계)이 사는 대역 $b_y$ 위만 다시 결정되는 σ에서 SDEdit한다. 노이즈는 물질에 고정한 MAN 노이즈라 프레임 사이 세탁이 일관된다. DIS 정렬 잔차가 0.5 px를 넘으면 기각한다. 초기 2만 장 이후 하루 4k장. 정리가 아니라 보정에 기댄 근사 보장이다.
- **Reveal-Bundle Demand** [신규 조합]:
  $$\mathcal R=\bigcup_\tau\mathrm{Vis}(W(t(\tau)),\Gamma(\tau))\setminus\mathcal V_0,\qquad \mathcal L_{EC}^{T4}=\mathbb E_{e\in E_{real},\,c:\,\mathrm{Vis}(c)\cap\mathcal R\neq\varnothing}\mathrm{KL}\big(p_{\mathcal R}(\cdot\mid e,c)\,\Vert\,p^{img}(\cdot\mid y_e)\big)$$
  실현된 사건이 실제로 드러내는 면에만 반사실 TFU를 쓴다. 탁상 장면에서 TFU가 10–50배 준다. 새 테이크가 새 면을 드러내면 copy-on-write 브랜치에서 그 면의 미세 대역만 연다(Frozen-Seed Band-Commit). 파괴 단면은 절차적 볼륨 재질로 만들고 히어로 컷만 교사로 polish한다.
- **캐릭터:**
  - MakeHuman(CC0) + 자체 22관절 BVH 스켈레톤, 자체 체형 PCA. SMPL-X는 Q9 판결 전까지 쓰지 않는다.
  - **MoPri-S 3단계:** A 모캡 flow matching → B MJPC 재시뮬로 약 3배 증강 → C **TPRF RWR**(Teacher Pose Restoring Force: 정지 순간을 절차적 인체로 렌더해 엔진 교사로 형태 대역 σ에서 SDEdit하고 R-Kpt로 키포인트 이동량을 잰다. 부자연스러운 포즈일수록 교사가 많이 고친다. P2(o)의 Photo-Slice RWR을 대체했다). 가중 회귀는 ID-Screen과 MJPC를 통과한 샘플에만 건다. 긴 모션은 겹치는 창 lockstep.
  - identity LoRA(캐릭터당 5–10 T4-h, 자체 촬영·렌더 데이터). **Soul ID는 캐릭터 개념 아트로만 쓴다.**
  - 얼굴은 v0 블렌드셰이프 + 규칙 viseme, v1 Dot-Face → Expr-S. 눈은 고전 각막 굴절 모델.
  - **"Act, don't prompt":** 2c의 연기와 음성은 사용자가 폰으로 찍은 연기를 Reader-S로 lift해 MoPri-S prior와 MJPC를 거쳐 재촬영한다(Q25).
- **음향 (학습 0) [복원].**

| 소리 | 방법 |
|---|---|
| 충돌, 타격 | 사건 충격량 $\mathbf J$로 모달 합성. 모드는 사면체 메시 FEM 고유값 해석(CPU, 에셋당 수 초–수 분) |
| 파괴 | 확률적 임펄스 |
| 액체 | Minnaert 기포 확률 합성(Harmonic Fluids 축소판) |
| 전파 | 이미지 소스 + **RT 코어 음선 추적** |
| 바이노럴 | 측정 HRTF(데이터) |
| 음성 | 2c 초반 녹음, 후반 TTS-S |

**시간 연방 지도 v2.1.**

| 공간 \ 시간 | 저 (0–1 Hz) | 중 (1–8 Hz) | 고 (>8 Hz) |
|---|---|---|---|
| 레이아웃과 서사 | **감독**, EPD-S, 실제 증인 검색 | EPD-S + 크로노(모양) | 사건(물리) |
| 형태와 모션 | MoPri-S + 크로노 | MoPri-S + MJPC | 접촉, PCGD-0 |
| 외관 | 엔진 교사(canonical) + θ_look | 렌더러(블러 적분) | 렌더러, 센서, MAN-S |
| 확률적 물질 | Stam 바람장 | SMF sway | 파티클 |
| 카메라 | CamFM-Intent | 리그 동역학 | CamFM-Hand |
| **시간의 눈금** | I4, 오디오 | I4, 오디오 | 물리 |

### 9.8 로드맵 재산정 (M0 = 프로젝트 시작, 승리 조건은 v1 §8.5 그대로)

GPU 몫: M0–3 10–15%, M3–9 30–50%, M9 이후 ≥50%.

| 단계 | 시점 | 새 부품 | 병목 | 우회 | 샷당 증분 | 승리 조건 (v1) |
|---|---|---|---|---|---|---|
| **2a Frozen Action & Speed Ramp** | M3–M9 (시드 M0–M3) | MAN-S, HeroProxy-S·ChronoProxy-S, Denoiser-S, Chrono-LoRA(조건부) | 교사 TFU, RT 렌더, 아레나 콘텐츠 300개(1,100–1,700 T4-h) | Reveal-Bundle, 캐스케이드, Voronoi, CFL FLIP, 세계 30 × 사건 10("세계는 한 번, 사건은 무한"), 1080p 정규화 | 3.5–7 T4-h | BUPS-FA ≥70%, Appearance Turing ≤65%, GeCo ≤0.03, TSR 밴드 안, RST ≤1 프레임 |
| **2b 강체·관절체 + 살아있는 세계** | M8–M16 | PCGD-0, F_φ-S, EPD-S, Feas-S, Reader-S | sim look, Physics-IQ의 lift 의존 | Garage ×3, 오디오 반발 분포, 뱅크, Lift Ladder, 절차적 생명(Stam 바람, Tessendorf 바다, Eulerian 연기, 모달 식생) | 2–4 T4-h | Physics-IQ·VideoPhy-2 1위, Motion Turing ≤60%, CSI KL ≤0.5×, BUPS ≥65%, 일반 Arena ≥50%, GCM ±20% |
| **2c 캐릭터·연기** | M14–M28 | MoPri-S v2(TPRF), Expr-S, identity LoRA, R-Kpt v2 | **사람 데이터**(FLOPs 아님), 천·헤어 | "Act, don't prompt", Garage 모캡 연 20–50 h, Act-from-Text, Dot-Face, 멀티캠 8–12대 | 4–8 T4-h + 캐릭터당 5–10 | 연기 축 ≥55%, 멀티숏 연속성 ≥70%, 얼굴 ≤75%, foot skate <2 cm/s, 관통 <1%, eyeline <3° |
| **2d 완전 4D + 인터랙션** | M26–M42+ | 손 조작(MJPC → 학습 정책), 다중 행위자 EPD | 손-물체, 에셋 다양성 | Merkle 청크 스트리밍, 스케일 다이얼 | — | 모든 범주 Arena ≥60%, Revisit 기울기 ≈0, 인과 정확성 1위 |

- **2a와 2b가 거의 밀리지 않는 이유:** 비용이 시뮬, 렌더, TFU에 몰려 있는데 T4는 이 항목들에서 상대적으로 덜 손해 본다. 사전학습은 처음부터 필요 없었다.
- **2c가 밀리는 이유:** FLOPs가 아니라 사람 데이터가 병목이다. 우회는 "연기는 생성하지 말고 찍어라"다.
- **범주 정복 순서:** Frozen Action → 강체 연쇄 → 파괴 → 액체 hero → 식생·바람 → 운동선수 단독 → 댄스 → 2인 대화 → 손 조작.
- **Kaggle 모드:** 2a를 M12–M18에 노린다.
- **H1′ 재정의 [복원].** v1의 H1′ go/no-go(10²⁰–10²²)는 10²²가 불가능하므로 **H1-T4**로 바꾼다: 10¹⁸–10²⁰ 구간에서 등사실감 지점의 우리 총 FLOPs를 경쟁 모델 공개 추정치와 비교해 기울기를 잰다.

### 9.9 Phase 2 1년차 컴퓨트 (약 6,000 T4-h)

| 용도 | T4-h/년 | 비중 |
|---|---|---|
| 작은 모델 학습 (3–4세대) | 900–1,300 | 15–22% |
| TFU: 판정, slice, LoRA, BSRT (약 3.5–4.2M TFU) | 2,500–3,000 | 40–50% |
| GPU 시뮬 | 400–700 | 7–11% |
| 렌더 | 900–1,200 | 15–20% |
| 벤치와 여유 | 300–500 | 5–8% |

**관찰:** 레버는 "더 큰 모델"이 아니라 **"더 싼 판정"과 "더 싼 검증"**이다. 학생의 절반(EPD-S, MoPri-S, Feas-S, F_φ-S)은 픽셀을 보지 않는다. 그래서 이들의 데이터 생산에는 TFU가 0이다.

---

## 10. 플라이휠과 "컴퓨트가 늘면 그대로 스케일" 원칙

### 10.1 Phase 1 플라이휠: 관할 이양 래칫 (AlphaZero-lite for Worlds)

```
 ┌(1) 솔브: WSSC-T4가 세계 W*를 만든다 (밤 2장, 10–16개/밤, 엔진 트랙)
 │(2) 배출: Teacher Exhaust (x̂0, DR, 탭) + commit 페이지 → Merkle 샤드 (soul=0만)
 │(3) 증류: TexelFM gen(k+1) ← o_s 바로 위 교사 대역 / MRH ← W* 자가 라벨(≤25%) / MassingFM ← W* 레이아웃
 │(4) 이양: 재질별 o_s 재측정 (자동 2AFC + 50쌍 감사) → Spec 수락률 a↑ → TJF↓
 └(5) 가속: TFU/세계↓ → 하루 세계 수↑ → (1)
        ※ Soul이 닿은 세계(Hybrid 트랙, 데모)는 이 고리 밖에 있다. 플라이휠은 Soul 없이 돈다.
```

| 래칫 지표 | G0 (Day 30) | G3 (Day 90) 목표 |
|---|---|---|
| TFU/세계 (방, 4K) | 약 700 | ≤520 (도전 360) |
| TJF | 약 0.85 | ≤0.6 |
| Spec 수락률 a | — | ≥0.6 |
| 재질 중 o_s가 한 옥타브 오른 비율 | 0 | ≥50% |
| 세계 벽시계 (Mode A) | ≤1 h (계획 40–60분, §4.4. E4 수직 슬라이스만 ≤4 h 허용) | ≤55분 (계획 35–55분) |
| 세계 계획 T4-h (Mode A) | 2.0 | ≤1.8 |
| 라이브러리 (엔진 트랙) | 10 | ≥300 |

- **붕괴 방지:** 배치마다 GT(D0+D1) ≥55%, 자가 라벨 ≤25%, Spec 5% 무작위 full-ownership 감사, held-out seed(WNF′)로 판정, ScanNet++ 평가 홀드아웃 고정.

### 10.2 Phase 2 플라이휠

```
 ┌(1) 제안: ShotScript × 세트피스 × EPD-S/MoPri-S (Feas-S, Chaos Bank 검색)
 │(2) 실현: SMC, PCGD-0, MJPC-lite                ← 학습 0
 │(3) 판정: 불변량 → RGF/ID-Screen → INT8 proxy → TFU (불확실성 상위 25% + 무작위 10%)
 │(4) 커밋: Block Bank + Exhaust + Lineage
 │(5) 증류: EPD-S, MoPri-S(RWR), Feas-S, HeroProxy-S·ChronoProxy-S, Reader-S, MAN-S(Exhaust-to-Shader)
 │(6) 읽기: Event-Demand Lift (Lift Ladder) → p_real, PCGD-0 재보정
 └(7) 가속: 제안↑ → N↓ / proxy↑ → TFU↓
```

AlphaZero와의 대응: 탐색은 SMC와 캐스케이드, 정책은 EPD-S와 MoPri-S, 가치는 Feas-S와 proxy, 심판은 물리·교사·실통계다.

**1회전 시간표 [P2(u) §10.1] (상시 2×T4 + 16 vCPU, Phase 2 몫 약 2/3).** v2.0이 되살렸던 P2(o) 숫자(실현 블록당 80 s, L3 6–7 h, BSRT 1만 장 4.5 h)는 P2(u)가 "틀림(Day-1)"으로 교정한 값이라 버린다. 뱅크가 없는 Day-1 혼합(소 60%, 중 30%, 대 10%)의 실현 비용은 블록당 가중평균 231 s다.

| 단계 | 장치 | 회전 0 (350블록, 뱅크 없음) | 정상 상태 (1,000블록, 뱅크와 Feas-S) |
|---|---|---|---|
| 제안 | GPU-B | 0.4 h | 1 h |
| 실현 | CPU 60% / GPU-B 40% | 231 s × 350 = 22.5 h-eq (CPU 13 + GPU 9) | 65–80 s × 1,000 = 같은 시간 |
| 판정 L2 (렌더 + proxy) | GPU-B INT8 | 0.1 h | 0.3 h |
| 판정 L3 | GPU-A INT8 | 350 × 64 TC = 3.1 h | 1,000 × 64 TC × 커버리지 60% = 5.3 h (c_TC 0.5 s 기준. §1.4의 512² INT8 0.65 s면 6.9 h) |
| 데이터용 렌더 | GPU-B | 0.2 h | 0.5 h |
| BSRT | GPU-A FP16 | 이미지 4천 장 × 4 TC = 3.8 h (§9.7 하루 4k장) | 3.8 h |
| 증류 (증분 재학습) | GPU-B | 6 h | 10 h (EPD-S, Feas-S, HeroProxy-S·ChronoProxy-S, Reader-S, MoPri-S RWR) |
| 실영상 lift | CPU | 5 h | 10 h (클립 300개, GLOMAP + Reader-S + R-Lift) |
| **합계** | | **GPU-A 7 h / GPU-B 16 h / CPU 18 h** | **GPU-A 9–13 h / GPU-B 21 h / CPU 23 h → 약 1일/회전** |

| 지표 | 회전 0 | 10회전 목표 |
|---|---|---|
| 하루 블록 | 350 (뱅크 없음, 블록당 231 s T4-eq) → 뱅크·Feas-S 후 1,000 | 4,000–8,000 (상단은 L3 커버리지 ≤25%) |
| 블록당 L3 판정 | 64 TC | 8–16 TC |
| proxy–교사 Spearman | 약 0.5 | ≥0.85 |
| SMC 충족률/입자 | 0.5–2% | 5–15% |
| ID-Screen 재현율 | 측정 | ≥0.95 |
| search gain (SMC 최선 − 학생 단일 샘플) | 큼 | 단조 감소(래칫의 증거) |

- **시드기**(Phase 2 몫 10–15%)에는 4–6일에 1회전이다. Kaggle은 "세션 = 세대"로 주 1회전, 규모 1/3이다.
- **붕괴 방지:** 실데이터 앵커 30% 이상, held-out Gold가 2% 넘게 퇴행하면 롤백, 무작위 L3 10%, 사건 그래프 엔트로피·범주 커버리지 감시, 주 1회 FP16 1024² 재판정으로 정밀도 드리프트 감시, Exhaust에 교사 버전 태그.

### 10.3 스케일 원칙 [운용]

| # | 원칙 |
|---|---|
| SC1 | **텐서 계약을 고정한다.** 모델 크기는 구현 디테일이다. 슬롯에 큰 모델을 끼우면 된다 |
| SC2 | **μP와 같은 데이터 파이프라인.** 30M에서 찾은 하이퍼파라미터가 3B로 넘어간다 |
| SC3 | **탐색과 상각 사이의 다이얼.** 컴퓨트가 늘면 먼저 학생을 키우고, N과 L3 커버리지와 교사 관할을 줄인다 |
| SC4 | Merkle 블록, Exhaust, Lineage, Swatch, Chaos Bank, 룩 프리셋 라이브러리는 모델과 독립인 **복리 자산**이다 |
| SC5 | 엔진 교사는 Dreamer 계약 슬롯에 끼운다. 가격표와 Exhaust는 버전 태그로 다시 보정한다 |
| SC6 | 고전 부품(물리, 광학, 음향)은 **감가상각이 없다.** 정확도 다이얼만 돌린다 |
| SC7 | FP16-safe 코드는 bf16·fp8에서 그대로 돈다. INT8 critic은 배포 계층에만 둔다 |
| SC8 | `z_slot`은 Phase 1.5 latent 정본 학생(세계 VAE)을 위해 예약해 둔 자리다 |
| **SC9 [v2.1]** | **미감 교사도 슬롯이다.** Soul이 다른 닫힌 모델이나 미래의 오픈 미감 모델로 바뀌어도 인터페이스(개념 아트, θ_look, 스틸 광택, 평가 앵커)는 같다. θ_look은 물리 파라미터라 교사와 독립이다. 승인 S가 오면 §5.7 순서대로 해금 브랜치를 연다 |

| 컴퓨트 | 무엇을 키우나 |
|---|---|
| ×10 (T4 → L4/A10 몇 장) | 1024² 판정 기본화, TexelFM 150M, MoPri-S 슬롯 300M, N ×10, WDM 장면별 실험 |
| ×100 | Phase 1 학생 1B(CWM-lite + LBO), EPD·Reader 300M, Location 모드 기본화 |
| ×1000 | v1 T1 경로(CWM/LBO 10²³), native WFM 착수 |

전 구간에서 비회귀 조항과 불변량 I1–I6, 그리고 Soul 방화벽은 그대로다.

---

## 11. 첫 90일

### 11.1 30일 결정적 실험 (합계 약 486 T4-h = 30일 용량 1,440의 34%)

| ID | 기간 | 가설 | 셋업 | T4-h | 성공 기준 | 실패하면 (목표 유지, 업그레이드) |
|---|---|---|---|---|---|---|
| **E0** T4 진실표 + 교사 정밀도 + 판정 보정 | D1–5 | (a) FP16 Rescue 후 NaN·검은 이미지 0/10,000(교사·정밀도별) + FP16 수술판은 commit σ에서 별도 2,000(0/1,000의 95% 상한은 0.3%라 세계당 700 TFU × 300–400 세계를 보증하지 못한다). (b) INT8 1024² ≤3.0 s, W4A4 ≤2.0 s. (c) FP32 대비 LPIPS: INT8 ≤0.10, W4A4(σ>0.5) ≤0.15. (d) 512² INT8 판정이 1024² FP16과 Spearman ≥0.9. (e) Turbo DR이 실사 vs 렌더, 정상 vs 결함을 AUC ≥0.7로 가른다. (f) **[TJC·H1b 복원]** Base 또는 Turbo+SR 경로의 DR AUC ≥0.8이고 Turbo 원본보다 ≥0.1 높으면 Q1 "병용" 권고 | §1.7 진실표 + 프롬프트 200 × 정밀도 5종, 층별 max\|act\|. 실사 1,000 / 렌더 1,000 / 결함 1,000 × 해상도 3 × σ 3. **판정 경로 {Turbo, Base} × 정밀도 {FP16, INT8, W4A4}**(P2 TJC 통합). 엔진 후보(klein 4B, ERNIE-Turbo INT8) 속도·NaN 동시 측정. NaN 감사는 512² 1만 NFE × 교사·정밀도 | **62** (기존 25 + TJC·H1b 25 + NaN 감사 약 12) | (a)–(f) | FP32 섬 확대 → INT8은 t>0.5에만, 결정 창은 FP16 → TRT export 실패 시 PyTorch FP16 + FA-turing(세계 ×1.7) → klein-4B. (e)가 실패하면 SR-LoRA(Q1). (f)·(e)가 모두 약하면 기본 판정 해상도를 768²로 올리고 P2 L3 게이트(불확실성 상위)를 25% → 15%로 좁힌다 |
| **E1** Teacher Ceiling v2.1 + 엔진 재오디션 | D3–18 | (i) E1-E: 엔진 교사 스틸이 Seedance/Kling 프레임을 ≥60%, Wan을 ≥55%로 이긴다(무작위 arm과 best-of-64 Teacher-DCA arm 모두 보고). 텍스처 뷰는 CC0 사진 대비 ≥45%. (ii) **E1-S: Soul 스틸의 SP_S 측정**(가정 65–75%, §3.8.5). (iii) 엔진 후보별 ES 측정. (iv) **E1-E+θ**: 엔진 스틸 + 렌즈·센서·필름 스택 + 사람 측정 θ_look(2D 후처리)이 E1-E 대비 +5%p 이상 | §3.4. Soul은 웹 480회(사람). 엔진 후보 3종 × 차트 뷰 arm. best-of-64 arm, E1-E+θ arm [RR X-T1 복원] | 26 (+API $0.3–0.6k, 사람 $1.5–5.5k) | 엔진: 녹색 또는 황색 → **D18 엔진 교사 잠금**. 하이브리드: TP_S\* = 1 − SP_S 확정(보고 항목). (iv)는 H-LA1의 조기 신호 | 엔진 황색이면 DTC, Gaffer, LoRA 의무. 적색이면 황색 의무에 더해 X-LA로 빈 몫을 찾는다(Soul 결과는 이 판단에 쓰지 않는다). ES가 전원 미달일 때만 Qwen Image Max 2512 INT4(Q3). 게이트 범주는 이기는 범주부터 연다 |
| **E2** X1-slim: 결정 곡선 + 가난선 초기값 + INT4 판정 충실도 + TAE 규칙 | D5–14 | (a) 스냅 스케줄의 연속 2 step 창이 차트 뷰에서 한 옥타브를 분리한다. (b) TexelFM-35/p8(폭 1/4 μP 프록시, 약 8M) 프로토가 재질 ≥50%에서 1 mm 이하를 교사와 구별할 수 없게 만든다(DR 비 ≤1.1, 사람 ≤55%). (c) INT4 \|ΔD\| ≤0.05, DR 순위상관 ≥0.9. (d) TAE-T가 노이즈 마스킹 규칙을 만족하는 t\*가 있고 latent Nyquist 아래 전달함수 ≥0.95 | 정면 텍스처 300(자체 RAW + CC0) + 교사 벽면 100 × t 10 × seed 4 @512²(약 14,400 fwd). 사람 2AFC 2,000쌍. TAE-T v0 | 27 | (a)–(d) | 결정 폭 >1.5옥타브면 레벨 간격 2옥타브 또는 SR-LoRA, 중간은 TexelFM. TexelFM이 지는 재질은 가난선을 한 옥타브 내림(+25분/세계) + 견본·마모장 투입. INT4는 초안과 데이터에만. TAE는 t ≥ t\*에서만 |
| **E3** MRH 차입 배당 + G-Adapter 가능성 | D14–30 | (a) 동결 INT8 교사 탭 위 MRH가 실사 홀드아웃에서 AbsRel ≤0.10(t≤0.5), ≤0.15(t=0.7), normal ≤18°이고, 같은 헤드를 처음부터 학습한 CNN 인코더 위에 얹은 것보다 ≥30%, t=0.6에서 "x̂0 decode 후 픽셀 추정" 경로보다 ≥20% 낫다(**차입이 실제로 효과 있음을 증명**). (b) SDEdit-from-render만으로 t=0.6에서 깊이 일치 AbsRel ≤0.12. 아니면 G-Adapter v0가 ≤0.08. (c) **[v2.1] Soul풍 대리 판독(FD (11)의 사전 검증, Soul 픽셀 0):** 우리 렌더 GT에 Soul풍 θ(2.35:1, f/2, 그레인, 로우키, 비네팅)를 입힌 스틸 → θ 설명 제거 → 엔진 교사 σ 스냅 0.2–0.3 1 NFE → MRH에서 AbsRel ≤0.15이고, 설명 제거 없는 경우보다 ≥20% 개선 | 합성 3만(Realify로 특징 동시 획득) + 실사 MVS 5,000 프레임(t 2개). 비교군 학습. G-Adapter v0(25M, 512², 8K step). 홀드아웃: 자체 10장면 + ScanNet++ 20장면(평가 전용) | 82 (Realify 12, 헤드·비교군 20, G-Adapter 45, Soul풍 대리 arm 5) | (a) + (b) 중 한 경로. (c)는 A+ 판독 착수(D45)의 근거 | 탭 층 sweep, t 2개 특징 앙상블, MVS 실사 확대(200장면). G-Adapter 실패 시 late-block 주입, step 확대 → teacher-first(MRH 기하를 slot bbox로 제약). 최후엔 솔버 안 평면 스윕 MVS + MRH는 coarse·intrinsics만. (b)를 SDEdit만으로 통과하면 G-Adapter 예산 80 T4-h 절약 |
| **E4** WSSC-T4 수직 슬라이스 | D18–30 | Bedrock + 의미 패스 + Grid-Locked OTBI(4 mm, 2 mm) + TexelFM(≤1 mm)로 방 하나를 2×T4 벽시계 ≤4 h에 풀고, 오빗 GeCo ≤0.04, 왕복 비트 동일, OIE ≤0.05, seam 2AFC ≤60%, TP ≥35%, HER ≥0.8(평균 융합은 ≤0.6) | 자체 촬영 방 3–5종(거실, 바, 한옥 마루, 벽돌 벽, 부엌), 거친 레벨은 COLMAP + TSQ 4 cm 저역통과. arm: (i) 뷰별 SDEdit + 평균 BP, (ii) OTBI + TABP, (iii) ii + WNF, (iv) iii + Fold·Context Ring, (v) iv + Router + **Spec-commit(TexelFM 초안 + 1 NFE 검증, κ_v=2) + Swatch Bank**, (vi) iv + PWMT-FiLM 대조((v)의 Swatch arm과 재질 정체성·T8·GPU1 점유를 비교). commit 디코딩을 TAE vs VAE로 비교. GeCo는 평가 전용. **[TF(u) E4 복원] NFE 경제 셋업:** facet world 24개(절차 재질 12 + 자체 촬영 표면 12, 4–10 m², STRATA 4레벨) | 70 | arm 하나 이상이 전부 충족. **(v) 추가 기준:** full-ownership 대비 교사 NFE ≥2.5배 감소, ΔDR_rel ≤5%, 교사 관할 최미세 레벨 수락률 a ≥0.6, native 레벨에서 수요 순서를 8회 섞어도 Merkle 루트 동일. **[RR X-T2 복원]** VAE commit이 TAE commit보다 HER +0.1 이상 | seam: apron 25%→40%, multiband λ_b, 텍스처 마스킹이 큰 곳에 seam, seam-owner 확대. 코스 불일치: G-Adapter 앞당김. 블러: τ_s annealing. 4 h 초과: W4A4 확대, 코스 뷰 축소, Merkle 메모. **Spec 경제 미달:** 견본 수·크기 확대 → 검증을 t 2개에서 → Spec 대상을 SMF류로 한정하고 CARRIER 재질은 교사 소유 유지. **GeCo 실패:** 미세 레벨은 외관만 쓰고 기하는 l≤11 + 노멀맵으로 고정 |
| **E5** TexelFM 연방 분할 | D5–28 | patchify/s2d TexelFM-35(35M)가 재질 셀 l≥13에서 (a) 실사 매크로·교사 페이지 대비 2AFC ≤60%, (b) 텍셀당 교사 경로보다 ≥5배 싸고, (c) 재실행 시 비트 동일. μP 프록시 TexelFM-35/p8(폭 1/4, 약 8M) → 35M에서 패치 FID와 SFR 오차가 단조 감소 | CC0 피라미드 + 자체 매크로 2천 장. TexelFM-35/p8·TexelFM-35 학습. 홀드아웃 재질 300쌍 × 판정자 10명. 클래스별 Δ로 Router 임계값 결정 | 36 | (a)(b)(c) | 재질 클래스별 헤드, apron 64, 교사 커밋 페이지로 증류 데이터 추가, 60–80M. 그래도 안 되면 Router 몫을 교사 쪽으로(비용은 오르고 품질은 지킨다) |
| **E6** Oracle Shot + X2 파일럿 | D1–30 | (a) 실촬영 방 → TSQ 세계 ≤1.5 T4-h, 왕복 비트 동일. (b) 4K 프레임 ≤0.6 s. (c) Defocus Pyramid + LH-split이 Cycles MC 대비 ΔJOD <0.25. (d) 같은 첫 프레임 경쟁 I2V 대비 VP ≥60%. (e) Corman 설정(밤, 안개, f/1.4)이 수요 텍셀을 ≥3배 줄인다 | 자체 촬영 10장면(400–800장, HDR, ArUco), COLMAP 4.2 SIFT + GLOMAP + PatchMatch → TSQ(2DGS 폴백), 카메라 스택 v0, Cycles 10샷 × 12프레임, 경쟁 I2V 10 × 의도 2 × 2모델 × seed 2 | 30 (+API $40–300, 평가 $0.3–0.8k) | (a)–(e) | 피팅이 느리면 FastGS식 densify 상한(0.4M). ΔJOD 실패 시 가림 경계 위험 타일(≤10%)만 MC. VP <60%면 v1 X2 ablation 그리드 |
| **X0** Spectrum Report | D1–30 | v1 X0 그대로(상용 모델의 3–12 Hz 각속도 PSD 결손) | GoPro gyro GT + **고전 추적**(COLMAP/GLOMAP SIFT)으로 추출기 H(f) 보정. 경쟁 클립 | 5 (+API $2–5k) | v1 기준 | v1 기준. 블로그는 어느 경우든 나간다 |
| **X-LA0** 미감 부검: 이끼 [v2.1] | D10–20 | 이끼 페어에서 카메라·조명 몫이 격차의 절반 이상(g_cam ≥0.5)이고, 엔진 교사 재질 + 사람 측정 θ_soul이 Soul 원본에 ≥45%(도전 ≥50%, H-LA3) | §3.8.6의 arm S/Z/Z+/C/D. 작은 facet 세계(0.3 × 0.3 m, TexelFM 영역), 사람이 θ 측정. 웹 20회. 사람 약 400쌍 | 6 (+사람 $0.5k) | g_cam ≥0.5 | 축 단위 단일 교체 ablation으로 빠진 물리를 찾는다(빈티지 처방, 솜털 STRAND, 림, 그레이딩). 목표 유지 |
| **P2-E2** GICC 크로노 | D7–28 | 교사가 "모양" 왜곡 4종 중 ≥2종을 AUC ≥0.75로 잡고, 게이지 대조에는 0.5±0.05 | Sensor Purity를 통과한 240fps 6범주 × 40클립, 합성 2,000장, 사람 300쌍 | 30 | + held-out 범주 ≥0.7 | Chrono-LoRA → 라이트 페인팅 → (K,Δ) 재설계 → 미감 전용으로 강등(H7) |
| **P2-E3** Verified Coverage–Precision | D10–30 | ID-Screen이 MJPC 성공을 AUC ≥0.8로 예측하고, 검증된 best-of-256이 단일 샘플 대비 2AFC +15%p | MoPri-S 16M(60% 스케줄), CMU·100STYLE, 4범주 × 10프롬프트, 무작위 200개 MJPC 전수 검증 | 40 (+사람 $1–2k) | 위 수치 | 접촉 일관성 항 추가, MoFeas-S, MoPri 32M, TPRF RWR 앞당김 |
| **P2-E4** SMC + Chaos Bank + X0 벤치 | D1–21 | 뱅크 warm start가 있으면 중형 과제 N=256에서 충족 ≥90%, 블록당 ≤90 s T4-eq | CPU MuJoCo, MuJoCo Warp, MJX 확인, Warp FLIP/MPM. 템플릿 3종 뱅크 10⁵(CPU 야간 4일). 과제 20 × 제약 5 | 20 | + "조종된 rollout" 식별률 ≤55%. 플라이휠 혼합(소 60·중 30·대 10%) 평균 ≤80 s/블록 [P2(u)] | 허용오차 창 학습, 잔여력 예산 확대, Feas-S 앞당김, 뱅크 10⁶ |
| **P2-E5** BSRT 보존성 + Exhaust-to-Shader 종자 | D14–30 | 세탁 쌍의 flow GT 보존(EPE ≤0.5 px, 기각 ≤20%). 합성 + BSRT가 합성만 쓸 때보다 Garage EPE −15% | 렌더 1만 → BSRT → R-Flow 학습 → MAN-S 지도학습 | 52 | 위 수치 + H5c: MAN-S가 Appearance 2AFC +5%p [P2(u)] | σ\* 한 단계 하향, DIS로 GT 되돌림, 렌더 다양성 확대 |

**P2 공동 항목:** P2의 TJC(Teacher-as-Judge Calibration) 판정 보정과 H1b(Base·Turbo+SR 판정 경로 비교)는 E0 (f)에 통합했고, 그 25 T4-h를 E0에 되살렸다. **합계:** P1 344(E0 62 + E1 26 + E2 27 + E3 82 + E4 70 + E5 36 + E6 30 + X0 5 + X-LA0 6) + P2 142 = **486 T4-h**.

**Kaggle 모드의 30일 실험 [v2.1 복원: TF·RR·P2 렌즈, 사후 검증에서 §11.5 버킷에 맞춤].** 범위는 §11.5의 Kaggle 열 버킷(1배 126, 2배 74) 안에 든다.
- **Kaggle 1배 (합 126).** E0는 진실표와 NaN 감사만(37, TJC·H1b 25는 빼고 Q1은 기본안), E1 26, E2 27, X-LA0 6은 전량, E6는 절반(15, 오라클 5장면), E3는 Realify와 MRH 헤드 축소판(15, 비교군·G-Adapter·Soul풍 arm 제외). E4(방 2장면, facet world 12개, 약 35)와 E5(TexelFM-35/p8만, 약 8)는 Day 30 이후로 미루고 같은 열의 세계 솔브(120)와 소형 모델 학습(250) 버킷에서 쓴다. 두 실험의 판정은 +2주 뒤에 한다. P2는 P2-E2·P2-E4만(합 50 중 §11.5 Kaggle 1배 P2 시드 35 안에서). 일정은 +2주.
- **Kaggle 2배 (합 74).** E0 37, E1 26, X-LA0 6, E2는 결정 곡선 (a)만(약 5). 나머지(E2 (b)–(d), E3, E4, E5, E6)는 Day 30 이후 같은 열의 소형 모델 학습(140)과 세계 솔브(60) 버킷에서 줄여 돌린다. P2 시드는 0이다.
- 목표는 그대로다. 모든 잡은 8 h 단위로 쪼갠다.

### 11.2 이후 실험 (31–90일)

| ID | 기간 | 내용 | T4-h |
|---|---|---|---|
| X2 본실행 | W7–8 | 오라클 20장면, 평가자 약 3,000쌍, VP ≥70% | 15 |
| X3-T4 Fusion Showdown | W5–8 | DL3DV 홀드아웃 10(평가 전용) + 생성 5. arm: 평균 융합 / WSSC-T4 / WSSC-T4 + Spec-commit / 장면별 WDM 1개(여유 시) | 60 (확장 시 +90은 예비에서) |
| X4 DTC | W6–7 | BBI ≤0.1, 플레어 문신 0, TP ≥45%. v2.1: Soul 스틸 설명 제거(θ_soul) 정확도 arm 추가(A+ 승인 시) | 20 |
| X5 PSH stress | W6–9 | 우리 렌더러로 2,000장면(벽돌, 타일, 펜스, 송전선, 간판, 야경, 글린트), 국소 연산자는 TexelFM | 25 |
| X6 Appearance Anchoring | W9–10 | RAS + 물리 bake vs 완전 역렌더, albedo leak −50% | 30 |
| X7 Camera Intelligence | W7–10 | CamFM-CRM 정확도 ≥70%, 탐색 경로 선호 ≥65%, 카메라 튜링 ≤55%. v2.1: 전경 레이어링·FiF 보상 on/off arm. 학습 쌍은 수제 프리셋 렌더만(`SOUL-INSPIRED` 제외) | 25 (프리뷰 렌더 18은 §5.6 P1 합계에 계상) |
| X-C Corman | W9–11 | 20 프롬프트 on/off, 대규모 세계 3–10배 절감 + SP 비열등 | 100 |
| **X-LA 미감 부검 본실험** [v2.1] | W5–12 | §3.8.6: 기록형(어부, 한옥) → 해석형(야크, 우주인) → 신규 25. 빈티지 처방·CPL을 실렌즈 사진과 비교하는 X-Lens 검증 포함. H-LA1–5 판정 | 70 (+웹 120, 사람 $2–4k) |
| **X-Soul 구조 보존 편집** [v2.1] | W6–10 | 우리 히어로 프레임 → Soul(2.0 레퍼런스 모드, Cinema 레퍼런스 모드, 프롬프트 변형) → 구조 보존 여부. 지표: G-buffer 대비 에지·저대역 드리프트, 팔레트 ΔE, 미감 이득(2AFC). A: 스틸 광택만. A+ 승인 시 세계 경유 POLISH 라운드 1회와 OIE 측정 | 5 (+웹 160, MCP 200) |
| E1-S 후처리 | W3 | Soul 스틸 정규화(2K → 1080p, 화면비 처리), TP_S 앵커 셋 구성 | 2 |
| P2-X6 Frozen Action 1샷 | W8–13 | 미니 BUPS-FA ≥60%, GeCo ≤0.03 | 60 (P2 시드 버킷) |

**X-Soul의 판정.** 저대역 드리프트 < τ이고 미감 이득 ≥ +10%p면 Soul을 "최종 미감 대역 소유 후보"로 승격한다(스틸은 A, 세계 경유는 A+). 구조를 깨면 Soul은 개념 아트와 룩 레퍼런스에 머문다. 어느 쪽이든 비디오 프레임별 광택은 하지 않는다(CT37).

### 11.3 주차별 계획 (GPU0 낮 / GPU1 낮 / 밤 두 장 / CPU 농장·사람 / Soul·법무·아트)

| 주 | GPU0 (낮 10 h) | GPU1 (낮 10 h) | 밤 14 h (두 장) | CPU 농장·사람 | Soul·법무·아트 [v2.1] | 게이트 |
|---|---|---|---|---|---|---|
| 1 | **E0**, Docker 툴체인 고정, FP16 Rescue | gsplat/2DGS 빌드, 카메라 스택 v0, Exhaust Ledger, Lineage CI | E0 NaN 스윕, **E1 스틸 생성** | 문법 v0, 에셋, RAW·IMU·오라클 촬영 시작, 경쟁 클립 발주, P2 뱅크 구축 | **D1–3 Higgsfield 메일**, soul_vault + 방화벽 CI v0, **E1-S 웹 생성 시작**, 룩 측정 양식 | Day 1, Day 3, Day 5 |
| 2 | **E2** 결정 곡선, T-LPIPS 라벨 | TAE-T, LatentDown/Up, COLMAP → TSQ | P2-E4 SMC, Realify 공장 | 합성 장면 1,000, MVS, E1 판정 | E1-S 생성 완료, **X-LA0 이끼 θ 측정**, 룩 프리셋 v0 착수, D3–7 도메인·회사 이메일·웹사이트 확보 후 Contact Sales 폼 재접수(티켓 #317768은 전달 불가로 종결) | **D14 왕복 v0** |
| 3 | Realify 공장(MRH), **E3 시작**, X-LA0 렌더 | **E5** TexelFM-35/p8·TexelFM-35, Defocus Pyramid, LH-split | P2-E2 GICC, BSRT | **E1 → D18 엔진 교사 잠금, E1-S → TP_S\* 도출(보고)**, 합성 3,000, X0 | X-LA0 사람 평가 | D18 |
| 4 | **E3** MRH + G-Adapter v0 | **E4** 방 3–5 × arm 6, Ledger, Proof-of-Shot, 8K 경로 | E4 계속, P2-E3/E5 | 문법 300만 + 캡션, **FREEZE-TAG 프롬프트 사전 등록**, 리그 스왑 릴, "Shot on 2×T4"(D28) | **룩 프리셋 라이브러리 v0(≥30) 동결(Hybrid-A)**, 카나리아 테스트, Plan 상태 확정 | **Day 30** |
| 5–6 | WSSC-T4 v1 통합(MassingFM, Swatch, Spec-commit), X3-T4 | MassingFM, CamFM-Intent, T-LPIPS-S, Corman v0, 빈티지·CPL 처방 | 라이브러리(주 약 25–30 세계, 밤 용량의 약 1/3), C1 세계 10 | GLOMAP 궤적, PSH(SYMBOL 한글 MSDF) | X-LA 기록형(어부, 한옥), 이끼 산맥 FPV 개념·룩 | — |
| 7–8 | X4 DTC, G-Adapter v1, X7(전경 레이어링 arm) | **TexelFM gen2(래칫)**, 평면 재귀 거울, OptiX/Warp secondary | 라이브러리 → 100, C1 세계 20, **을지로 네온**, **이끼 산맥 FPV** | X2 본실행, CamFM-CRM 쌍 수집(우리 렌더·수제 프리셋만) | X-LA 해석형(야크, 우주인), X-Soul 수동, **D45 A+ 회신 시 Hybrid-A+ 착수** | **Day 60** |
| 9–10 | Hero FX(Voronoi + CFL FLIP 와인잔, 먼지 MPM), **X-C** | DIEL SDF, DP-0 탐색, Reader-S(P2) | **미니 벤치 세계 생성(라이브러리 = 벤치)**, 거울 복도, 한옥 마당 | 경쟁 클립 T4± 경로 추출, Corman 슬라이더 | **우주인과 말 개념 아트 + IP 점검**, X-LA 신규 25, D60 S 회신 시 CamFM-CRM 해금 실험 브랜치(S1) | — |
| 11–12 | 데모 polish, 불릿타임 헌정, 체호프의 컵, 우주인과 말 | 4K/8K 최종 렌더, 웹 export, P2-X6 | 벤치 Engine 160 세계 + Hybrid-A 160(Engine 세계 재사용 재촬영) 완료 + 렌더 (+A+면 Hybrid-A+ 160 세계) | 사람 평가 본실행(TP_S 앵커 포함) | 키 아트 광택(스틸), 데모 IP 최종 점검 | — |
| 13 | Day-90 게이트 | — | — | 기술 보고서, 벤치 공개(Kaggle 노트북, Engine. H-A는 Q13 회신 시), 릴 3+1+1편 | Soul 사용 보고서(호출, 크레딧, 플랜, 방화벽 감사 로그) | **Day 90** |

### 11.4 게이트

게이트를 못 넘어도 목표는 낮추지 않는다. 해당 실험의 "실패하면" 업그레이드를 앞당긴다.

| 시점 | 통과 조건 |
|---|---|
| **Day 3** | [v2.1] 메일 발송, soul_vault 격리 확인, 방화벽 CI가 카나리아를 잡는다 |
| **Day 5** | E0 (a)(b) 또는 FP16 수술판 ≤5 s/fwd |
| **Day 14** | E2 (a)(b), 왕복 v0 비트 동일 |
| **Day 18** | E1-E 녹색 또는 황색 → 엔진 교사 잠금(적색이면 황색 의무, §3.4. Soul 결과는 쓰지 않는다). E1-S로 SP_S 확정(보고 항목) |
| **Day 30** | E3 (a), E4 전 기준을 충족하는 arm 1개 이상, E5 (a)(b)(c), E6 (a)–(d), X-LA0 판정. P2: E0 가격표 확정 + (P2-E2, E3, E4) 중 2개. [v2.1] 룩 프리셋 v0 동결 |
| **Day 60** | §8.2 Day 60 행 |
| **Day 90** | §8.2 Day 90 행 (v1 조건 그대로 + T4 추가 조건 + 트랙별 보고) |

### 11.5 90일 T4-h 배분

| 버킷 | 상시 (4,320) | Kaggle 1배 (771) | Kaggle 2배 (386) |
|---|---|---|---|
| 30일 결정적 실험 (P1 몫, X-LA0·엔진 재오디션·TJC 포함, §11.1) | 344 | 126 (Kaggle 축소판, §11.1) | 74 |
| 소형 모델 학습 (P1 1세대 + 핵심 2–3세대 + ablation, §5.6) | 650 | 250 | 140 |
| 교사 데이터 공장 (§6.5) | 330 | 60 | 30 |
| 세계 솔브 (개발 약 250 + 라이브러리 + 오라클 30) | 605 | 120 | 60 |
| FREEZE-TAG 미니 벤치: Engine 350 + Hybrid-A 90(Engine 세계 재사용, §8.4) | 440 | 80 (Engine만, C1 40 세계) | 40 (Engine만, 20 세계) |
| 데모 3+1 + 하이브리드 1 + Corman 슬라이더 | 140 | 40 | 30 |
| 이후 실험 (X2–X7, X-C 275 + X-LA 70 + X-Soul 5 + E1-S 2) | 352 | 0 | 0 |
| 렌더, DP-0, Hero FX | 150 | 30 | 12 |
| 평가 (GeCo 등) | 80 | 30 | 0 |
| Phase 2 시드 (P2-E2..E5 142 + 학습 + 2a 프로토) | 470 | 35 | 0 |
| **예비·유휴** (1–3주차 코드 공백 포함) | **759 (18%)** | 0 | 0 |
| **판정** | 전체 계획. A+ 승인 시 Hybrid-A+ 벤치 160 세계(320–400)를 예비에서 쓰면 예비는 약 360–440(8–10%). Hybrid-A 재사용이 실패해 단독으로 지으면(+최대 약 320) 예비는 더 줄어든다. 둘이 겹치면 Hybrid-A+를 80 세계로 줄인다 | 데모 + 범주 1개, Engine 트랙만, 일정 동일 | **일정 1.5–2배(D135–180), 목표 동일. 상시 확보를 권고** |

### 11.6 Soul 크레딧과 승인 마일스톤 [v2.1]

| 시점 | 마일스톤 | 크레딧·웹 소진 목표 (누적) | 규칙 |
|---|---|---|---|
| D1–3 | 메일 발송(부록 B), MCP 등록·OAuth, soul_vault, 창업자 보유 크레딧 잔고 확인 | 웹 50 / MCP 0 | MCP 단가와 '1회'의 계량 단위(스틸/배치)를 실측해 §1.10을 고친다 |
| D14 | Contact Sales 폼 재접수 회신 확인(재접수는 D3–7), E1-S 완료 | 웹 550 / MCP 20 | — |
| D30 | Plan 상태 확정(무회신이면 Plan A로 운용 지속), 룩 프리셋 v0 동결 | 웹 1,000 / MCP 150 | MCP 잔고 <70%면 버전 감시 빈도를 반으로 |
| D45 | A+ 회신 시 Hybrid-A+ 트랙과 X-Soul 세계 경유 착수 | 웹 1,500 / MCP 400 | A+ 거절이면 640을 예비로 돌린다 |
| D60 | S 회신 시 CamFM-CRM 해금 브랜치(`lineage:soul-s1`) | 웹 2,000 / MCP 800 | MCP 잔고 <30%면 데모 스틸과 벤치 순으로만 |
| D90 | Soul 사용 보고서, 다음 분기 예산 | 웹 2,900 / MCP 1,600 | 초과 사용 금지. 웹 자동화 금지 |

- **소진 감시.** 주간 리뷰에서 웹·MCP 사용량을 용도별로 대조한다(§3.7.5 배분표). 용도 밖 사용은 아트 디렉터 승인이 필요하다.
- **약관 스냅샷.** 메일 발송일과 매월 1일에 약관 전문을 해시로 보관한다. 약관이 바뀌면 R27 절차를 탄다.

### 11.7 팀과 돈

- **팀:** v1 최소 팀(8명, 최소 6명)을 유지한다. 컴퓨트가 줄었다고 엔지니어링이 줄지 않는다. **병목은 이제 사람이다.**
  - 역할: 세계 엔진 2(solver/동기화, 학생/래칫), 표현·렌더러 1, 카메라 물리·렌즈 1, 카메라 지능 1, 데이터·캡처 1, 평가·사람 연구 1, Phase 2·시뮬 1.
  - **[v2.1] 아트 디렉터·룩 담당 0.5–1명**(창업자가 겸할 수 있다): Soul 개념 아트 선별, θ 수동 측정, WorldScript 전사, IP 점검. 법무는 외부 파트타임.
- **90일 돈:**

| 항목 | 금액 |
|---|---|
| 컴퓨트(상시 클라우드 기준) | $1.5–2.2K |
| 호스트(자체 서버라면, 16–32 vCPU, 128 GB, NVMe 4 TB) | $3–5K 1회 |
| 경쟁 API | $3–12K |
| 사람 평가 (E1-S, X-LA, TP_S 앵커 포함) | $13.5–32K |
| 캡처 장비(미러리스 + 매크로 $1–2K, IMU 리그 $3–5K, Garage·LED 보드 $0.1–2K, 빈티지 렌즈 2–3종 실촬영 비교 $0.3–1K) | 합계 $4.4–10K |
| **[v2.1] Soul**: 구독(기존) + MCP 크레딧 약 1.6만. 창업자 보유 크레딧을 먼저 쓰고 부족분만 구매(미검증 단가) | 0–$1K (상한) |
| **[v2.1] 법무**: 약관 해석, 승인 협상, IP 점검 서명 | $2–5K |

v1($0.3–0.45M) 대비 **컴퓨트 비용이 100분의 1 이하로 줄었다.** 남은 돈은 사람, 데이터, 그리고 빌린 미감에 쓴다.

---

## 12. v1에서 보류하거나 폐기한 것, 제약 덕분에 좋아진 것

### 12.1 버리는 것과 미루는 것

| v1 항목 | 처리 | 이유 | 대체 |
|---|---|---|---|
| H100 64장, 0.4–0.8M H100-h | **폐기** | 하드 제약 | 4,320 T4-h. 파워포인트 속 데이터센터로 명예 퇴직 |
| CWM 4–8B + LBO 사전학습 10²³ | **폐기** | 2.1–2.6×10¹⁸/일로 약 105–130년 | MassingFM + BRP + 의미 패스 + TexelFM + 래칫 |
| WDM student (DMD2, fake-score LoRA 64–128) | **연기** | T4에서 교사 backward는 forward의 약 3배이고 메모리도 크다. 장면 규모 수렴 선례도 없다 | 샘플형 솔버 + 래칫. 여유 예산으로 장면별 WDM 실험 1–3개(fake score = 같은 base + LoRA, 가중치 1벌) |
| LLM World Program, VLM SelfCaption | **금지** | 대형 LLM과 다른 사전학습 모델 금지 | MassingFM, CaptionComposer, WorldScript DSL, 자기지식 캡션 |
| 닫힌 SOTA hero / art director | v2.0 금지 → **v2.1 부활(Soul, Plan A)** | 창업자 결정. A/B 50축 중 41승 | Soul 개념 아트(사람 전사), 룩 프리셋, 평가 앵커. 방화벽 |
| 닫힌 API POLISH 소유자 (v1 §1.6 iii) | v2.0 금지 → **v2.1 조건부 부활(A+)** | T6 근거, 약관 회색지대 | Shoot 전 CoW 라운드의 세계 경유 POLISH(S8.5). 비디오 프레임별 광택은 금지 |
| GAN real 집합에 닫힌 교사 샘플 | **금지 유지** | 약관(학습 사용) | 자체 실사 패치 |
| DA3, VGGT-Ω, π3, AnySplat, MegaSaM, MonST3R, MoGe, TRELLIS, Hunyuan3D | **시스템 경로 금지** | 사전학습 모델 | MRH, MassingFM, COLMAP/GLOMAP(SIFT), CC0 에셋. VGGT/UFM은 GeCo 평가 전용 |
| `ctrl=Gbuffer` 사전학습 ControlNet (Z-Image Fun-ControlNet 포함) | **금지** | 제3자 사전학습 네트워크 | G-Adapter |
| RAFT/UFM 보정 광류, LPIPS(VGG), SigLIP/DINO sem, OCR 모델 | **교체** | 금지 | DIS/TV-L1, T-LPIPS-S, MRH sem(TE 공간), Glyph-S |
| OIDN, OptiX AI 디노이저 (렌더와 데이터 생성 모두) | **금지** | 사전학습 CNN | spp 확대 + à-trous/SVGF, Denoiser-S(P2) |
| 폰 사진, ARKit 깊이, 폰 뉴럴 ISP | **교체** | 신경망 라벨, 환각 | Bayer RAW, MVS + raw LiDAR, OpenVINS |
| latent `z` 정본, UKA | **연기·변경** | 디코더가 없다 | explicit TSQ + BC 페이지, `z_slot` 예약 |
| 3DGRUT 단일 렌더러 (3DGRT primary) | **변경** | T4의 RT 처리량, Turing any-hit 비용, 1 mm 디테일을 Gaussian으로 표현하려면 10⁸개가 필요 | gsplat-TSQ 래스터 + OptiX 삼각형(선택 픽셀) + 평면 재귀 + DIEL SDF |
| MC 16–64 spp DOF 타일 | **변경** | 4K 프레임 비용이 3배를 넘는다 | Defocus Pyramid. MC는 레퍼런스 검증에만 |
| 미세 레벨 블록당 4–8뷰 중첩 투표, Adam 피팅 | **변경** | T6: 중첩은 낭비이고, Adam 상태 때문에 10–19 GB OOM | Grid-Locked OTBI + 닫힌형 TABP |
| PWMT (렌즈 TF 원안) | **폐지** (PWMT-FiLM은 E4 대조군) | 블러/암기, GPU1 경합, T8 상실 | Swatch Bank + 견본 조건 TexelFM |
| NRC, 경로추적 GI (P1 렌더 경로) | **폐기** | 비용 | RAS + 물리 bake |
| Location 모드 기본값 | **변경** | 비용 1.5–2배 | Directed 기본, 벤치는 Location-lite |
| 정적 lift 5천만, 동적 2천만 클립, WSSC 데이터 엔진 1–2만 세계 × 10 h | **폐기** | 예산 | 자체 캡처 300–500장면, Event-Demand Lift, 라이브러리 300–400 세계 × 2 T4-h |
| Cycles GT 15만, Infinigen 300만 (렌즈 원안) | **폐기** | 숨은 비용(§6.5) | 자체 렌더러 + 자체 문법 |
| X1 규모 (60만 fwd) | **1/40로 축소** | 예산 | X1-slim 14,400 fwd (E2) |
| X3 arm 6개 | **축소** | 금지 모델(SEVA, Difix3D+)은 기준선 평가 전용 | E4 arm 6개, X3-T4 |
| per-token-t 어댑터 1M장, `<plate>` LoRA 7만 장 | **축소·연기** | 예산 | 필요할 때 A1 LoRA 1–2만 장 |
| PIS 2.0 VDP 위험 배분 | **연기** | 엔지니어링 대역폭 | P1-확장 |
| Gold Set ($0.3–0.6M, 1000fps, LiDAR 50 h) | **축소·연기** | 비용 | Garage Gold + Sensor Purity. 1000fps는 2b에 대여 |
| 비디오 증인(Wan), PLAC, PC-ELBO, 진술 그래프 | **금지** | 비디오 모델 금지 | 크로노 GICC(PC 대조와 QBC를 이미지판으로 재활용), 실제 증인, 감독 |
| PCGD F_φ 0.3–1B, EPD 0.5–2B, MAN 0.2–0.5B, 정책 0.1–1B | **축소** | 수십M 제약 | F_φ-S 0.6M, EPD-S 28M, MAN-S 1.5M, MJPC-lite |
| Edit-Closed World (모든 편집 30–50%) | **축소** | TC | Reveal-Bundle Demand |
| PHC, MaskedMimic | **연기 (2d)** | 샘플 10⁹–10¹⁰ | MJPC-lite |
| SMPL-X, FLAME, AMASS/HumanML3D, GVHMR/WHAM, SAM 3D Body | **연기·교체 (Q9)** | 학습된 통계 모델, 비상업 라이선스, D3 | 자체 스켈레톤, MakeHuman, CMU 원본, Garage, R-Kpt(Reader-S) + 멀티캠 삼각측량 |
| Codec-Avatar급 얼굴 latent 256–512 | **연기** | 데이터 | 블렌드셰이프 + Expr-S |
| INTENT/EMOTION의 VLM 판정 | **폐기** | VLM 금지 | 사람 critic + 자체 라벨로 학습한 소형 분류기 |
| 사전학습 TTS, IC-Light | **폐기·연기** | 금지 | 녹음 → TTS-S, 물리 재조명 |
| Harmonic Fluids | **축소** | 비용 | Minnaert 확률 합성 |
| Spectral Division(f_c 실측), AV-Physics Timing Solver, Blur-as-Measurement, RGF + Reality Residual Budget | **유지** | 고전, 측정용 | 고전 추적(KLT, RANSAC 자기운동 보상), 오디오 간격 증인 추가 |
| Temporal DoF, 시간 PSH, Gabor 물결 | **유지, 앞당김** | 학습 0, 렌더 측에서만 처리해 싸다 | 2a·2b 쇼케이스 |
| Builder–Reader self-play | **유지·중심화** | Reader를 직접 만들어야 한다 | BSRT |
| Single-cause audio, Coverage Compiler, 테이크, 분기 우주, 재조합 엔진 | **유지** | 학습 0 | — |
| H1′ go/no-go (10²⁰–10²²) | **재정의** | 10²² 불가 | H1-T4(§9.8) |
| 8K·120fps EXR 상시 저장 | **폐기** | 10초에 600–840 GB | (Merkle, Γ) 저장, 최종 테이크만 EXR |
| §9.4 규모 가설, WFM T2/T3 | **재정의** | 규모 제약 | "WFM" = 수십M 전문가 연방 + 빌린 엔진 교사 1 + 빌린 미감 교사 1(콘텐츠 전용) + 구조. 컴퓨트가 늘면 §10.3 다이얼을 돌린다 |
| Powers of Ten, 10분 원테이크 | **유지(P1-확장)** | 컴퓨트로는 가능(5–15 T4-h), 리스크는 의미 정합 | Day 85–90 이후 |
| 실시간 웹 뷰어 | **앞당김** | T4는 원래 렌더 카드다. 시청자 기기가 렌더한다 | Day 60 이후 |

### 12.2 제약 덕분에 좋아진 것

1. **반복 속도.** 수십M 모델은 학습이 몇 시간에서 하룻밤이면 끝나고, 카탈로그 전체를 다시 학습해도 2×T4로 약 1–2주다. 주당 10–20개 실험을 돌릴 수 있다. 큰 팀이 체크포인트 하나를 기다리는 동안 우리는 세대를 여러 번 넘긴다.
2. **인수분해가 선택이 아니라 물리 법칙이 된다.** 모듈 경계가 명확하고, 부품마다 단위 테스트를 할 수 있고, 비용이 어디서 새는지 TFU로 보인다. 실패한 대역은 `git blame`으로 찾는다.
3. **라이선스가 짧아진다.** 학습된 가중치는 모두 우리 데이터, Apache-2.0 엔진 교사 하나, 고전 알고리즘에서 나온다. Soul은 학습 계보 밖에 있어서 가중치의 라이선스를 오염시키지 않는다.
4. **결정론이 기본값이 된다.** 작은 모델, Philox 노이즈, 고정소수점 리덕션, 동결 엔진 덕분에 Proof-of-Shot이 가능하다.
5. **미감과 원가가 같은 방향을 본다.** 보케, 밤, 연기, 역광, 모션블러가 연산 할인쿠폰이다(Corman). 로저 코먼 B급 영화의 생존술이 관객이 가장 영화적이라고 느끼는 룩과 같다. **v2.1의 A/B가 이것을 반대편에서 확인했다.** Soul이 이긴 축의 대부분이 수요를 줄이는 축이었다.
6. **E2 고정점이 장식에서 연산자로 바뀐다.** v1에서 "교사가 고칠 게 없는 세계"는 이론적 정의였다. v2에서는 Spec-commit의 수락 테스트로 실제 기계가 됐다.
7. **차원 해석이 정직해진다.** 비디오 판사가 없으니 "사진 한 장이 무엇을 알 수 있는가"를 명시해야 했고, 그 결과가 GICC다.
8. **데이터가 정직해진다.** RAW 원칙, 같은 코드 경로로 만든 합성 GT, Lineage CI를 거치면 미세 대역은 남의 신경망이 그린 것이 아니라 빛이 실제로 만든 것이 된다.
9. **Turbo 교사의 단점이 장점이 된다.** CFG가 없어 NFE가 절반이고, 증류 모델이라 부모 컴퓨트를 빌린다.
10. **배포가 싸다.** 교사를 뺀 스택 약 0.2–0.4B는 INT8로 소비자 GPU와 WebGPU에서 돈다. 온프렘 스튜디오, 실시간 뷰어, 세계 단위 과금에서 원가 우위가 생긴다.
11. **재현 가능한 공개 벤치를 낼 수 있다.** FREEZE-TAG Engine 트랙이 Kaggle 무료 T4에서 돈다.
12. **이야기가 생긴다.** "T4 두 장, 140W(전기장판 두 장)로 10²⁴를 이긴다."
13. **[v2.1] 미감이 측정 가능한 연방이 된다.** 미감을 한 모델의 신비로 두지 않고, 축마다 소유자를 정하고, 격차가 어느 몫인지 블라인드로 잰다(X-LA). 큰 팀은 더 큰 이미지 모델을 기다리고, 우리는 빠진 몫을 고친다.
14. **[v2.1] 지루한 교사가 최고의 측량 기사가 된다.** 미감을 엔진 교사에게서 떼어 내자, 평광·팬포커스라는 약점이 세계 재료로서의 장점으로 뒤집혔다. 엔진 교사는 미감이 아니라 통제성과 속도로 고를 수 있게 됐다.

---

## 13. 리스크 → 업그레이드 (후퇴 없음)

| # | 리스크 | 감지 | 업그레이드 |
|---|---|---|---|
| R1 | **미감 천장** (엔진 교사 Z-Image Elo 941, 상용 1위 대비 기대 승률 19%) | E1-E <60%, E1-S, D60 SP <35%, X-LA 몫별 격차 | **하이브리드: Soul 개념 아트 + θ_look 연방으로 몫별 수리(X-LA).** 엔진 황색이면 DTC + Gaffer + A1 LoRA + hero N=256 의무. ES가 전원 미달일 때만 Qwen Image Max 2512 INT4(Q3). 교사가 이기는 범주부터 게이트를 연다 |
| R2 | bf16 교사의 FP16 오버플로 | NaN 비율, 블록 max-abs | FP16 Rescue(잔차 재스케일 + SwiGLU 재균형), FP32 섬 확대, Mode B 파이프라인 분할, W8A16 |
| R3 | INT8/INT4가 T4에서 안 되거나 판정이 표류 | E0 (b)(c)(d), 주간 FP16 재판정 괴리 | Mode C(commit은 FP16), INT4는 탐색·초안·데이터에만. 최악 경로도 세계당 계획 약 6 T4-h(순수 3.0) |
| R4 | Turbo의 임의 σ SDEdit 실패, DR 판별력 부족 | 결정 폭 >1.5옥타브, E0 (e) AUC <0.7 | σ 점 스냅, Π_b 엄격 제한, 레벨 간격 2옥타브, SR-LoRA, Q1(Base) |
| R5 | 차트·OTBI 이음매, 코너 위상 불일치 | seam 2AFC >60%, 코너 위상 오차 | apron 25%→40%, Fold 뷰, seam-owner 투시 뷰 확대, 텍스처 마스킹이 큰 곳에 seam, LatentUp/Down 효도 투영 |
| R6 | CG 룩, 데칼 룩 (절차 기하와 교사 꿈의 불일치) | TP <35%, 데칼 지수 | Picture-First, 관할 재배정(프로그램은 L0만), G-Adapter 앞당김, 의미 패스 뷰 48→64, θ_look 마모·그레이딩 |
| R7 | TexelFM의 "새집 CG" 또는 "SR 룩" | 마모 통계 대비 실사, 스펙트럼 편차, 2AFC | 기하 유도 마모장, Swatch 강화, RAW 데이터 확대, 60–80M, Router 몫을 교사 쪽으로 |
| R8 | MassingFM 단조로움, 자유 프롬프트 OOD, 개념 해석 부족 | 다양성 지표, 프롬프트 커버리지, X-LA 해석형 g_con | 문법 확장, 패러프레이즈 규칙, Infinigen 비중 확대, hero 관측 가중 확대, 스케일 연출 레시피, DSL·아트 디렉터 폴백(Soul 개념 전사) |
| R9 | MRH의 실사 일반화 실패 | E3 홀드아웃 AbsRel | 실사 MVS 차선 확대, 두 t 특징 앙상블, 평면 스윕 MVS 대체, W\* 자가 라벨 ≤25% 유지 |
| R10 | Spec-commit DR 해킹 (판정을 속이는 초안) | held-out seed(WNF′) DR 괴리 | 검증 seed 분리, 5% 무작위 full-ownership 감사, 사람 2AFC 표본 |
| R11 | 숨은 데이터·엔지니어링 비용 폭주 | 데이터 장부 대비 실측 | 주간 리뷰, 원천별 T4-h·CPU-h 상한, 같은 코드 경로 합성 GT |
| R12 | 두 장 GPU 경합 (솔브 vs 학습) | GPU 가동률 <70%, 잡 대기 | 주야 시분할, 20분 체크포인트로 선점 가능화, TFU 스케줄러 |
| R13 | 컨테이너 OptiX 부재, gsplat sm_75 빌드 문제 | E0 | Warp BVH 폴백, 2DGS 단계화, CUDA C++만 사용 |
| R14 | 결정론 붕괴 (TRT 재빌드, atomic) | 재솔브 해시 불일치, order_purity_check 실패 | 고정소수점 정렬 리덕션, 엔진 해시를 Merkle 키에, deterministic 커널, 보장 범위 명시 |
| R15 | 엔지니어링 범위 폭발 | 렌더러 PR 비중, 마일스톤 지연 | 미분 래스터 하나(gsplat-TSQ), 단계화(2DGS → TSQ → OptiX), 렌즈 라이브러리 확장은 처방 데이터 추가로만, Phase 2는 1인 시드 트랙 |
| R16 | 사람 평가가 병목 | 2AFC 대기열, 비용 | 교사-판사 자동 2AFC + 50쌍 감사, BT 능동 샘플링(30–40% 절감), 데일리 모드 선택을 데이터로 재활용(Soul 계보·경쟁 API 프레임 쌍 제외) |
| R17 | 계보 누수 (몰래 들어온 사전학습 모델) | Lineage CI 실패 | 해당 아티팩트 격리 후 재생성, import 금지 린트, 체크포인트 해시 목록, §6.6 통합 목록 |
| R18 | Kaggle 쿼터 2배 차감, 디스크 20 GB, RAM 30 GB, 약관(상업 R&D) | Day-1 확인 | 상시 확보(Q4), stateless 8 h 잡, Kaggle Models 마운트, §1.6 Kaggle 메모리 행, 일정 연장, 생성 즉시 소비. 약관이 상업 R&D를 막으면 Kaggle은 공개 벤치 재현용으로만 |
| R19 | T4 툴체인 지원 종료 (다음 CUDA 메이저) | 릴리스 노트 | Docker 이미지와 드라이버 고정, 컨테이너 스냅샷 |
| R20 | 데이터 라이선스 (BAPPS, MatSynth 일부 CC-BY, 100STYLE, COCO, AVA, 공개 다시점 NC) | 법무 게이트 | 상업 경로는 자체 캡처, CC0, 절차 데이터만. 공개셋은 평가 전용 |
| R21 | (P2) 크로노 증인이 시간 모양에 둔감, 또는 게이지 누출 | GICC AUC <0.65, 대조 AUC >0.6 | Chrono-LoRA, 노출 개수·블러 정규화, 라이트 페인팅, 미감 전용 강등(H7) |
| R22 | (P2) sim look | Motion Turing >65% | Garage ×3–10, PCGD-0 → F_φ-S, OU 미세 토크, 1000fps 대여 |
| R23 | (P2) 검증 재현율 부족 | ID-Screen r <0.95 | 접촉 일관성 항, MoFeas-S, L2 확대 |
| R24 | (P2c) 사람 동작 데이터 기아 | MoPri-S 학습 곡선 | Act-from-Text, Garage 모캡 연 20–50 h, 멀티캠 + R-Kpt 삼각측량. 컴퓨트가 아니라 장비와 사람 문제다 |
| R25 | 경쟁 모델의 일관성 개선 (Seedance 3, Veo 4) | AA Elo, FREEZE-TAG 정기 재측정 | 구조적 우위는 유지된다: 결정론, 순서 불변, 광학, 재촬영, 8K·120, 세계 단위 과금, Proof-of-Shot |
| **R26** | **약관 해석: 세계 피팅이 "모델 학습"으로 해석될 위험** | 법무 의견, 메일 회신 | Plan A 기본(사람 경유), A+ 서면 승인 전 자동 피팅 0, 방화벽 CI, 약관 스냅샷 보관 |
| **R27** | **약관 변경**(학습 금지 범위 확대, 출력 소유·상업 조건 변경, 서비스 종료) | 월간 약관 해시 diff | **제거 가능성 원칙**: Engine 트랙이 단독으로 돈다. 기존 자산은 생성 시점 약관 스냅샷으로 관리한다. 룩 프리셋은 물리 파라미터라 남는다 |
| **R28** | **모델 버전 변경·단종**(Soul 2.0 후속, Cinema 변경) | 버전 감시 프롬프트 월 1회 | 자산마다 버전 태그. θ_look은 교사와 독립. 확정된 개념 아트는 이미 콘텐츠다. 데모 재현은 soul_vault 바이트로 |
| **R29** | **크레딧 소진·가격 인상** | 주간 소진표(§11.6) | 배분표 우선순위(벤치 > 데모 스틸 > 감시). 크레딧이 부족하면 MCP 용도를 축소한다(Hybrid-A+ 160 → 80 세계, 버전 감시는 분기 1회). 웹으로 옮기는 것은 원래 사람 경로 용도(데모 개념 아트, 스틸 광택, 룩 레퍼런스)에 한한다. **자동 파이프라인 입력(Hybrid-A+ 벤치 등)을 사람이 웹에서 대신 생성해 공급하지 않는다**(CT34, Hybrid-A+의 완전 자동 정의). 추가 구매는 창업자 승인. **웹 자동화 금지** |
| **R30** | **2K 상한** | — | Soul 픽셀은 4K·8K 영상 미세 대역의 원천이 아니다(Nyquist 이하만, A+). 원경은 Corman–Shannon이 2K 충분성을 판정. 스틸 광택은 2K로 납품하거나 표기 |
| **R31** | **미감 불일치**(Soul의 구운 조명·광학이 물리 세계와 충돌, 개념 frustum과 MassingFM 뒷면 사이의 Janus) | TP_S 정체, 조명·이음매 2AFC | θ_soul로 설명 제거(DTC), Gaffer 재적합, Janus 규칙(frustum 안 = Soul 개념, 밖 = MassingFM), BRP·엔진 교사가 개념과 일관되게 채움. A에서는 사람이 물리에 맞게 번역해 전사 |
| **R32** | **IP 유사**(Soul 출력이 기존 캐릭터·작품과 닮음) | IP 점검 절차(§8.3) | 프롬프트 위생, 사람 검토, 역이미지 검색, 법무 서명, 재생성(웹 'IP 재생성·대체' 100회, 부족하면 예비 900에서) |
| **R33** | **승인 거절** | 메일 회신 | **Plan A로 운용**: 룩 프리셋 라이브러리(Hybrid-A), 사람 전사, 스틸 콘텐츠, 평가 앵커. X-LA가 개념 몫을 수치로 보여 주면 directed 모드와 아트 디렉션 인력에 투자한다. 목표는 그대로 |
| **R34** | **업로드 콘텐츠 유출**(비Enterprise는 Higgsfield가 업로드로 학습 가능) | 업로드 로그 | 미공개 세계·데모·제3자 이미지 업로드 금지, 텍스트 프롬프트 위주, Enterprise 조건 문의, X-Soul 레퍼런스는 공개가 확정된 프레임만, 미공개 데모의 세계 경유 POLISH는 공개 확정 뒤 또는 Enterprise 조건에서만(§3.6 B4) |
| **R35** | **방화벽 누수** | 카나리아 실패, 계보 감사 | 권한 격리, CI, 누수 시 해당 아티팩트 격리 후 재학습(소형 모델이라 수일 안), 감사 로그 보존 |
| **R36** | **벤치 재현성·공정성 비판**(닫힌 모델이 파이프라인에) | 외부 피드백 | 트랙 분리(CT38), Engine·Hybrid-A 공개, Hybrid-A+ 자산 해시 공개, 승리 주장에 트랙 표기 |
| **R37** | **MCP에 Soul Cinema가 없음** | MCP 도구 목록 | Soul 2.0으로 MCP, Cinema는 웹 사람 경로. 버전 태그로 구분. 메일로 일정 문의 |
| **R38** | **정체성 희석·벤더 종속**("그리지 말고 찍어라"인데 그리는 모델에 기댄다) | 이해관계자 반응 | Soul은 미술팀(개념과 레퍼런스)이지 촬영이 아니다. 영상의 모든 프레임은 세계를 찍은 것이다. 제거 가능성 원칙, 미감 교사 슬롯(§10.3 SC9) |
| **R39** | **광택 스틸과 영상 프레임의 혼동**(마케팅에서 Soul 광택 스틸을 영상 프레임처럼 보임) | 공개물 검토 | 스틸 광택 산출물 표기 규칙, Proof-of-Shot에서 구분, 영상 A/B에는 광택 스틸을 쓰지 않는다. Soul 크레딧은 Soul이 닿은 작품에만 붙이고 Engine 트랙 결과물에는 붙이지 않는다(§8.3) |
| **R40** | **미감 점수 리워드 해킹** [RR(u) R13 복원]: MRH-Aes가 hero search, Gaffer CEM 200회, DP-0 β·MRH-Aes, Corman β_a에 동시에 쓰이고, Soul·경쟁 API 쌍을 빼서 학습 쌍이 2–3만으로 줄었다 | held-out 사람 판정과 MRH-Aes 점수의 괴리, 탐색 상위 후보의 사람 2AFC 정체 | MRH-Aes ⊕ Teacher-DCA 앙상블 고정, 탐색 상위 k의 10% 사람 스팟 체크, 판정 쌍으로 주기 재학습, Gaffer·Corman에서 MRH-Aes 단독 사용 금지(최종 재순위는 앙상블) |

---

## 14. 창업자가 결정할 열린 질문

| # | 질문 | 기본안 (판정이 없을 때) |
|---|---|---|
| Q1 | **Z-Image Base와 Turbo를 "엔진 교사 하나"로 볼 것인가?** (Base는 판정, Turbo는 샘플). v2.1: 미감이 Soul로 넘어가 Base의 연속 σ·판별력이 더 중요해졌다 | Turbo 단일 가중치 + 필요하면 SR-LoRA(1.1%). 단 E1 재오디션에서 Base 판정 arm을 함께 잰다 |
| Q2 | **R-TE: Qwen3-4B TE를 encoder로만 쓰고, 그 임베딩을 작은 모델들(MassingFM, CamFM, EPD-S, MoPri-S)의 조건으로 쓰는 것(Lingua Franca)을 허용할 것인가?** 생성형 사용은? | 임베딩은 허용, 생성형 사용은 금지. 불허하면 자체 8M 텍스트 인코더 + ShotScript 닫힌 어휘로 간다 |
| Q3 | **엔진 점수(ES)가 모든 후보에서 미달이면 Qwen Image Max 2512(20B, INT4, Apache)로 승급할 것인가?** 세계 비용은 약 2배. v2.1: 미감 미달은 승급 사유가 아니다 | 승급 허용(ES 미달일 때만) |
| Q4 | **상시 2×T4를 확보할 것인가?** 클라우드 90일 $1.5–2.2K 또는 자체 서버. 호스트 16–32 vCPU, 128 GB, NVMe 4 TB | 상시 확보. Kaggle은 보조와 공개 벤치용. Kaggle 약관의 상업 R&D 허용 여부를 D1에 확인한다 |
| Q5 | **R-DATA: D3 데이터셋**(AMASS/HumanML3D, DataDoP, SpatialVID, Motion-X)을 학습에 쓸 것인가? | 평가와 연구 ablation만 |
| Q6 | **R-EVAL: 평가 모델**(VGGT, UFM, LPIPS, OCR)은 사람의 모델 선택에만 쓰고 자동 최적화 신호로는 쓰지 않는다. **Soul 평가 앵커**는 더 좁게, 사람 보고서(진단)에만 쓰고 엔진 교사 잠금, 트랙 개방, 게이트, 체크포인트·하이퍼파라미터·프리셋 선택에는 쓰지 않는다(약관의 "improve any AI model"과 충돌을 피한다). 승인하는가? | 승인 |
| Q7 | **제3자 파생 가중치**(Lightning LoRA, fp16-fix VAE, Fun-ControlNet, 사전 양자화 가중치)를 쓸 것인가? | 금지. 공식 배포판과 자체 어댑터·양자화만 |
| Q8 | **캡처 기기 내장 ML**: ARKit를 VCam 라이브 프리뷰 입력 장치로만 쓰는 것, 폰 ARKit 얼굴 추적을 "센서"로 보는 것을 허용할 것인가? (데이터 경로에서는 금지) | 금지. ArUco PnP + OpenVINS, Dot-Face |
| Q9 | **SMPL-X/FLAME은 에셋인가, 금지된 모델인가?** 비상업 라이선스 문제도 있다 | 금지. 자체 스켈레톤 + MakeHuman(CC0) + 자체 체형 PCA |
| Q10 | **캡처와 사람 평가 투자**: RAW·매크로 $1–2K, IMU 리그 $3–5K, Garage·Sensor Purity $0.1–2K, 빈티지 렌즈 실촬영 비교 $0.3–1K, (P2c) 멀티캠 $3–8K, 사람 평가 90일 $13.5–32K | 승인 권고. 컴퓨트가 아니라 장비와 사람이 해자다 |
| Q11 | **Phase 2 GPU 몫** M0–3 10–15%, M3–9 30–50%, M9 이후 ≥50%를 승인하는가? 시드 트랙을 Day 0부터 1인으로 운영하는가? | 승인 |
| Q12 | **Location vs Directed 기본값**, shot dress와 샷별 조명 치팅 허용 여부 | Directed 기본. 벤치는 Location-lite |
| Q13 | **오픈소스 범위**: FREEZE-TAG(Kaggle에서 실행 가능), STRATA-lite 포맷, ShotScript, Proof-of-Shot 검증기, 룩 프리셋 라이브러리(Hybrid-A 재현용) | 벤치·검증기·H-A 프리셋은 공개, 포맷은 Day 90 이후에 판단. 프리셋 공개는 후속 문의(티켓 #317768 또는 영업팀 연락)의 회신으로 Higgsfield 입장 확인 후(발송본에는 이 질문이 없었다) |
| Q14 | **첫 제품과 런칭 첫 컷** (v1 Q1, Q11 계승): 툴 / API / B2B | 툴(세계 + ShotScript + 재촬영 + 룩 프리셋) |
| Q15 | **장기: 컴퓨트가 늘면 자체 이미지 교사를 학습할 것인가?** (§10.3 ×100–×1000) | 장기 과제로 둔다 |
| **Q16** | **Higgsfield 승인 요청 범위와 시점.** A+와 S를 함께 요청할 것인가? 공동 마케팅을 제안할 것인가? [사후 추가] Contact Sales 폼이 요구하는 웹사이트와 회사 이메일 도메인을 먼저 확보할 것인가? 발송본의 크레딧 문구 "Look by Higgsfield Soul"을 바로잡을 것인가? | A+와 S는 D1에 함께 요청했다(부록 B 발송 기록). 후속 문의에서 S1(비생성 ≤20M)과 S2(픽셀 없는 생성형 prior ≤25M)를 나눠 다시 밝힌다. 이미지·비디오 생성기 학습은 요청하지 않는다. 공동 마케팅은 이미 제안했다. 크레딧 문구는 영업팀과 협의할 때 Soul의 실제 몫(개념 아트·미술)에 맞게 바로잡는다(R38). 고객지원은 티켓을 영업팀에 전달하지 않는다고 답했다(07:28Z). 그래서 도메인, 회사 이메일, 한 장짜리 웹사이트를 D3–7에 만들고 폼으로 다시 접수한다 |
| **Q17** | **`SOUL-INSPIRED`(사람이 Soul 스틸을 보고 만든 프리셋·스크립트)를 학습(래칫, CamFM-CRM·MRH-Aes 쌍 등)에 써도 되는가?** 그 프리셋으로 렌더한 프레임과 판정 쌍은 태그를 상속한다(§3.7.3 3) | 법무 의견 전까지 제외. 콘텐츠·프리셋으로는 허용 |
| **Q18** | **승리 주장 트랙.** Hybrid-A+ 결과로도 승리를 주장할 것인가? | 모든 자동 트랙에서 주장 가능하되 트랙을 표기한다. 공개 재현 주장은 Engine·Hybrid-A만 |
| **Q19** | **Soul 예산.** MCP 크레딧 약 1.6만(약 $1K, 미검증)과 Enterprise 플랜 검토 | 1.6만 크레딧 한도. 창업자 보유 크레딧을 먼저 쓰고 부족분만 구매(상한 $1K, D1–3 잔고 확인). Enterprise는 업로드가 필요한 X-Soul 세계 경유가 유의미할 때(A+ 이후) |
| **Q20** | **엔진 교사 교체 규칙.** ES가 3%p 이내면 약 1.5배 빠른 klein 4B로 바꿀 것인가? | 예. 교체 비용 50–100 T4-h를 감수한다 |
| **Q21** | **비디오 프레임별 Soul 광택 금지(CT37)를 제품 규칙으로 승인하는가?** | 승인. 광택은 스틸과 세계 경유뿐이다 |
| **Q22** | **Soul ID 인물과 얼굴을 상업 데모에 쓸 것인가?** (초상·IP) | 실존 인물 닮음 금지. P1 데모는 바이저 뒤나 원거리의 작은 인물만 |
| **Q23** | **첫 공개 컷 순서.** 왕복 챌린지 / 이끼 산맥 FPV / 우주인과 말 | 왕복(리스크 낮음) + "Shot on 2×T4"를 먼저, Day 60에 이끼 산맥 FPV, Day 90에 우주인과 말(하이브리드 쇼케이스) |
| **Q24** | **아트 디렉터 인력.** 창업자 겸임인가, 채용인가? | 90일은 창업자 + 파트타임. X-LA가 개념 몫이 크다고 말하면 채용 |
| **Q25** | **(P2) 2c 제품 방향과 실데이터 투자** [P2(o) §17 복원]: "Act, don't prompt"(연기와 음성은 사용자가 찍는다)를 기본으로 할 것인가? Garage Gold 예산과 1000fps 대여 시점은? 호스트 16 vCPU·64 GB 이상을 확보하는가(CPU는 세 번째 가속기)? | 예. Garage $0–2k, 1000fps는 2b에 대여, 호스트 확보 |
| **Q26** | **(P2) COCO keypoint와 FlyingThings를 상업 경로 학습에 쓸 것인가?** [P2(u) Q9 복원] | 연구 경로만. 상업 경로는 Garage 사람 라벨과 우리 렌더(FlyingStuff) |

---

## 부록 A. 신규성 장부 v2.1 (확신도)

| 확신도 | 항목 |
|---|---|
| **높음 [신규]** | Corman–Shannon Demand + Corman Optimizer(대역평균 OTF, λ·TFU). Teacher Ceiling Test를 Day-18 게이트로 쓰는 프로토콜(v2.1: 엔진·Soul 두 천장) |
| **중상 [신규]** | 가난선 + 판정 해상도 분할 + 관할 이양 래칫(교사-판사 자동 2AFC, 3구역 Router; CB(u) 중상, TF 중상/중). 크로노포토 증인 + BIS + **GICC**(유사변환군 직교 왜곡 + 게이지 음성 대조; P2(u) 중상). Speculative Band Commit. OTBI + TABP(Adam 없는 세계 생성). Fold 뷰 + Context Ring. Swatch Bank + 견본 조건 TexelFM(T8 복구). MRH 노이즈-t 독심(픽셀이 생기기 전에 기하 융합). Teacher Exhaust Ledger + Feature-Tap 기생 학습. MassingFM(절차 학습 레이아웃 파운데이션). 노이즈 마스킹 근사 규칙. Proof-of-Shot + commit-before-shoot. 초록 무한 터널 포렌식. **[v2.1] 미감 연방제(축별 소유권) + 미감 부검 3분해(X-LA: 카메라·연출·개념 몫을 블라인드 격차 닫힘으로 측정)** |
| **중 [조합]** | Realify + 탭 동시 추출 패스(교사 운송 [조합], CB 중). T-LPIPS-S(교사 특징 증류 지각 척도, CB 중). Lift Ladder([신규 정식화], P2 중). Grid-Locked Latent Charts(MultiDiffusion, SyncMVD, TexFusion 계열에 STRATA 피치, WNF ρ=1, 가난선을 결합). LatentDown/Up-S 효도 투영. G-Adapter(T2I-Adapter 구조, from scratch). FP16 Rescue(잔차 재스케일 + SwiGLU 재균형). σ-정밀도 사다리와 σ-카드 분할. 기하 유도 마모장. Picture-First, Program-Backed bedrock. 물리 bake Bedrock. Coverage–Precision Split v2 + ID-Screen. Exhaust-to-Shader. BSRT. TPRF. Reveal-Bundle Demand. Sensor Purity Test. 데이터 순도 D0–D3 + Lineage CI. Longuet-Higgins Split. Defocus Pyramid. 해석 통계 CamFM-CRM + 합성 선호 사전학습. Blender 오픈무비 카메라를 DP 정답 라벨로. Teacher-DCA 미감. semantic metric prior. **[v2.1] θ_soul 역추정 룩 프리셋(DTC·렌즈 식별성의 확장: 스틸 한 장의 미감을 모든 각도로). BT 앵커 게이트 TP\* = 1 − SP_anchor. 전경 레이어링·프레임 안의 프레임 보상. 스케일 연출 레시피(역-미니어처·거대화). Soul 방화벽 계보 CI(닫힌 모델 출력의 학습 격리). 닫힌 미감 교사의 T6 POLISH 소유(v1 §1.6 iii의 구현)** |
| **[운용]** | TFU 화폐(CB [운용]), 주야 시분할, 라이브러리 = 벤치, 프레임 대신 세계와 경로 저장, CPU 농장, 세션 = 세대, 데이터 장부, 부모 연산 차입 원칙, 상시 vs Kaggle 회계, **[v2.1] 벤치 트랙 분리, 승인 3단계(A/A+/S), 웹 사람 경로와 MCP 크레딧 경로 분리, 측량 모드 프롬프트, 정본 이름 체계** |
| **[부품] (정직하게 차용)** | Z-Image-Turbo, **Higgsfield Soul Cinema / Soul 2.0(닫힌, 콘텐츠 전용)**, Nunchaku/SVDQuant, SmoothQuant, flash-attention-turing, gsplat/2DGS/3DGUT, OptiX, Warp, MuJoCo(Warp, MPC), COLMAP/GLOMAP/Caspar, PatchMatch, xatlas, TAESD, DPT, T2I-Adapter, Diffusion Classifier, RAFT 구조, MDM, GNS, OIDN 구조, Kolb/Hullin/Newson/Malvar, Kraus–Strengert, Frankot–Chellappa, **Rayleigh 하늘 편광·Fresnel(CPL), 빈티지 렌즈 특허 처방**, BC6H/BC7/BC5, Infinigen, ProcTHOR 문법, CC0 에셋, μP/u-μP, Wortsman 처방, Many-Worlds Browsing, motion matching, Voronoi 파쇄, Minnaert, Zheng & James, Grid Diffusion(선례), TinyStories 효과, v1의 [부품] 전부 |

---

## 부록 B. Higgsfield 서면 승인 요청 메일 골자 (Plan A / A+ / S)

**발송 기록 (2026-10-10).** 아래 골자를 다듬어 support@higgsfield.ai로 보냈다. 제목은 "Request for written permission to use Soul outputs (Block Universe Cinema)"다. 약 30분 뒤 고객지원팀이 답했다(티켓 #317768). 서면 라이선스 허가, 모델·API 제공, 크레딧, 레퍼런스 이미지 조건, 쇼케이스 파트너십은 모두 Pricing 페이지 Enterprise 항목의 Contact Sales 폼(https://higgsfield.ai/pricing)으로 접수하면 영업팀이 직접 연락한다는 내용이다. 그 폼은 회사 이메일과 웹사이트를 요구해서 쓸 수 없었다. 그래서 06:57Z에 같은 티켓으로 후속 메일을 보냈다. 원 메일을 제휴·영업팀에 직접 전달해 달라고 요청했고, 필요한 정보는 이메일로 보내겠다고 적었다. 07:28Z에 두 번째 회신이 왔다. 폼은 회사 이메일만 받고, 제휴·대형 계약 제안은 그 폼이 공식 경로이며, 고객지원 채널에는 따로 알려 줄 영업 주소가 없다는 내용이다. 내용에 대한 거절이 아니라 접수 경로 안내다. 영업팀은 아직 메일을 보지 못했다. 그래서 D14까지 기다리지 않고 도메인과 회사 이메일, 한 장짜리 웹사이트를 먼저 만든 뒤 폼으로 다시 접수한다(Q16). 영업팀이 연락해 오면 이메일로 S1/S2 구분과 남은 질문(§3.7.4 "아직 묻지 않은 것")을 보낸다. 웹사이트·도메인 확보 여부는 Q16에 있다. 회신이 오기 전까지는 Plan A만 쓴다(§3.7.4).

- **발송본이 실제로 말한 것.** (1) 콘텐츠 사용과 "사람이 품질을 평가할 때 Soul 스틸을 우리 렌더 옆에 둔다"는 것의 확인 요청. 평가 결과의 용도 제한(보고 전용)은 적지 않았다. (2) Soul 스틸 한 장을 장면 하나의 피팅 목표로 쓰고 같은 단계에서 카메라 설정을 재는 것(A+). (3) 소형 내부 모델 학습(S). 예로 든 것은 "카메라를 어디에 둘지와 구도를 배우는 약 20M 구도 모델" 하나다. 질문은 MCP/API 제공, 스타트업·연구 크레딧, 업로드 레퍼런스의 학습 제외 셋이다. 대가로 데모의 "Look by Higgsfield Soul" 크레딧, 결과 조기 공유, 우주인과 말 공동 쇼케이스, NDA 하 설계 문서 공유와 통화를 제안했다.
- **Plan S 문구의 해석 규칙.** 발송본의 "약 20M camera framing model"은 크기와 설명이 CamFM-Intent(20M, 궤적 생성 prior, S2 ④)에 가깝다. 생성형이라는 사실은 밝히지 않았다. 그래서 이 문구를 근거로 온 승인은 S1 ① CamFM-CRM(12M, 비생성)에만 적용한다. S2는 생성형임을 밝힌 별도 회신 전까지 금지한다. 후속 문의에서는 "12M 비생성 구도 평가기(S1)"와 "20M 궤적 생성 prior(S2, 생성형)"로 바로잡아 다시 묻는다.
- **크레딧 문구.** 표기는 §8.3 규칙을 따른다. 문구는 영업팀과 협의할 때 바로잡는다(Q16).

**구성 (한국어 요지).**

1. 우리가 누구인지, 왜 Soul인지: 물리 카메라로 정지 세계를 찍는 파이프라인, 미감 A/B에서 Soul이 압도적이었다는 사실.
2. **Plan A(확인 요청):** 출력 소유·상업 사용 조항 안에서 Soul 스틸을 콘텐츠(키 아트, 세계 속 이미지, 원경)와 사람의 레퍼런스로 쓰고, 사람 블라인드 평가의 비교 대상으로 쓴다(결과는 보고용이고 학습·튜닝·모델 선택 신호로 쓰지 않는다). 사람이 Soul 출력을 보고 측정한 카메라·룩 프리셋을 만들고, 이를 공개 벤치와 함께 공개할 수 있다. 어떤 모델의 학습에도 쓰지 않는다. 이것이 약관 안이라는 우리 이해를 확인해 달라고 요청한다.
3. **Plan A+(승인 요청):** Soul 스틸을 관측으로 삼아 3D 장면 표현(신경망이 아닌 명시적 세계 파라미터)을 최적화하는 것, 카메라·룩 파라미터를 자동 추정하는 것. 이것이 약관의 "AI 모델 학습"에 해당하지 않는다는 해석 확인 또는 명시적 허가.
4. **Plan S(승인 요청, 두 단계):** (S1) 생성 모델이 아닌 20M 이하 소형 모델 3종(카메라 구도 평가기 12M, 룩 프리셋 예측기 약 2M(물리 카메라·조명 파라미터 출력), 동결 백본 위 미감 점수 헤드 ≤2M). (S2) 픽셀을 출력하지 않는 25M 이하 생성형 prior 2종(카메라 궤적만 출력하는 경로 prior 20M, 박스 레이아웃만 출력하는 3D 레이아웃 prior 24M). S2는 생성형임을 밝히고 따로 묻는다. 이미지·비디오 생성기는 학습하지 않는다. 데이터 상한과 감사 로그를 제공한다.
5. 질문: Soul Cinema의 MCP 제공 여부·일정, MCP 크레딧 단가와 '생성 1회'의 계량 단위(스틸/배치), 버전 고정과 단종 공지, Enterprise 조건(업로드 학습 제외), 사람이 측정한 룩 프리셋(물리 카메라·조명 파라미터)을 공개 벤치와 함께 공개해도 되는지, Soul을 이름으로 비교한 블라인드 결과를 공개해도 되는지, 아트 디렉터가 창업자 계정을 써도 되는지(아니면 별도 시트), 공동 마케팅.
6. 안전장치: Soul 방화벽(계보 태그, 격리 저장소, CI, 카나리아), 요청 시 감사 로그 공유.

**영문 초안 핵심 문단.**

> We build a pipeline that constructs a 3D world once and films it with a physically simulated camera; every video frame is a render of that world, not a generated frame. In side-by-side A/B comparisons, Soul Cinema was by far the strongest aesthetic reference we found, and we want to use it as our art director.
>
> We understand the following to be within your Terms and ask you to confirm: (A) using Soul outputs as content (key art, in-world images, matte backdrops) and as human references; using them as comparison anchors in human blind evaluations whose results are used only for reporting, never as a training, tuning, or model-selection signal; and making human-measured camera/look presets by viewing Soul outputs, which we may publish with an open benchmark. We will not use outputs to train, fine-tune, or improve any AI or machine-learning model.
>
> We request written permission for: (A+) optimizing explicit 3D scene parameters (not neural network weights) against Soul outputs used as observations, including automatic estimation of camera and look parameters; and (S) training a small number of models that never output pixels on Soul outputs, with a data cap and audit logs, in two tiers: (S1) non-generative models under 20M parameters (a camera-composition critic, a look-preset predictor that outputs physical camera and lighting parameters, an aesthetic scoring head on a frozen backbone); (S2) at most 25M-parameter generative priors that output only camera trajectories or 3D bounding-box layouts. We disclose that S2 models are generative and ask about them separately. We will never train an image or video generator on Soul outputs.
>
> We would also like to know whether Soul Cinema will be available through your MCP server, the MCP credit pricing and whether one generation is metered per image or per batch, your policy on version pinning and deprecation notices, Enterprise terms for uploaded references, whether we may publish look presets (physical camera and lighting parameters measured by humans while viewing Soul outputs) with our open benchmark, whether we may publish blind-comparison results that name Soul, whether a collaborator may use the founder's account or needs a separate seat, and whether you would be open to co-marketing ("Shot on 2×T4, art-directed with Soul").

---

## 부록 C. v2.1 변경 이력

**v2.1 감사 반영 (정합성 비평 C1–C40, 누락 비평 V1–V36).** 76건을 모두 반영했다. 그중 15건(C.2의 9개 항목)은 다른 지시와 충돌하거나 원문 렌즈와 대조한 결과에 따라 고쳐서 반영했다(아래 C.2). 지시 전체를 반영하지 않은 항목은 없다.

### C.1 반영 요약

| 영역 | 바뀐 것 | 위치 | 지시 |
|---|---|---|---|
| **비회귀** | 세계 경유 POLISH(A+)를 Shoot 전 **S8.5**로 옮겼다. CoW로 다시 연 1개 Jacobi 라운드에서 엔진 교사 뷰와 동시에 편집·소유하고, 재판정 뒤 commit한 다음에 렌더한다. 렌더 뒤에는 스틸 광택(S11)만 남는다. §0.4 각주를 "여섯 군데에 쓰이고 그중 두 군데가 학습 데이터"로 고쳤다 | §0.4, CT37, §3.7.2 (c), §4.2, §4.3, §4.5, §7.1, §12.1 | C2, C30 |
| **Plan S 범위** | S1(비생성 ≤20M: CamFM-CRM, LookPrior-S, MRH-Aes 헤드)과 S2(픽셀 없는 생성형 ≤25M: CamFM-Intent, MassingFM, 생성형임을 밝히고 별도 협의)로 나눴다. MRH 기하 헤드는 요청하지 않는다. 메일 국문·영문을 같은 범위로 고쳤다 | §2.2, §3.7.4, §3.8.2, §3.8.4, §5.7, §7.8, Q16, 부록 B | C1, V2 |
| **예산 구멍** | Hybrid-A는 Engine 세계의 bedrock·재질 commit을 재사용하고 L0 재bake, 수요 차 CoW, S8–S10만 다시 돈다(세계당 0.35–0.7 T4-h). 벤치 버킷 350 → 440. Hybrid-A+는 별도 세계(예비에서) | §4.4, §8.4, §11.3, §11.5 | C3, V9 |
| **Soul 방화벽** | (i) Feature-Tap Bus 링버퍼, TRT 양자화 보정셋, 교사 VAE 쌍 수집기를 prov 검사 대상에 넣고, soul=1 호출은 Tap Bus를 끈 별도 프로세스로 돌린다. 카나리아를 링버퍼에도 주입. (ii) MRH-Aes 학습 쌍에서 경쟁 API 프레임과 Soul 계보 태그 쌍을 뺐다. (iii) `SOUL-INSPIRED` 상속 규칙(프리셋 렌더와 판정 쌍). (iv) P7 문구를 "어떤 모델의" 학습 손실·보상·선택 점수로 좁히고 두 예외(세계 피팅, θ_soul 초기값)를 명시. (v) prov.soul의 비트 정의, Exhaust 용도 3 제한, 래칫 게이트 `prov.soul = none ∧ track = E` | CT32, §1.6, §3.7.3, §4.1 P5·P7, §4.3, §5.3 M3·M12, §6.2, §6.3, §6.6, R16 | C4, C12, C14, C16, C17, V1, V34 |
| **의사코드** | H-A+ 트랙에서만 MCP·InverseLook·관측 제약. H-A 벤치에서 C≠∅이면 assert | §4.3 | C15 |
| **약관·메일** | Q6 Soul 앵커 용도 축소. 영문 Plan A를 "확인 요청"으로. 프리셋 공개, Soul 이름 비교 결과 공개, 계정 공유, 계량 단위 질문 추가. 업로드는 공개 확정 프레임만. R29: 자동 파이프라인 입력을 사람이 웹으로 대신 생성하지 않는다 | §1.10, §3.6 B4, §3.7.2, §3.7.4, §3.7.5, §8.4, R29, R34, Q6, Q19, 부록 B | C13, C20, C24, C40, V14, V24, V35, V36 |
| **교사·속도·NaN** | FP16 3.1–4.5 s 근거 정리, Nunchaku Z-Image Turing 경로 미확인 표기, NaN 기준 0/10,000 + 수술판 2,000, Day-1 진실표에 순위·합격선·추가 측정 5종, 사전 양자화 가중치는 Q7 전 금지 | §1.2, §1.4, §1.7, NC2, §3.2, §6.6 #16, E0 | C5, C18, C19, V6, V7 |
| **30일 실험** | E0 (f) TJC·H1b 복원(+25)과 NaN 감사(+12), E1 best-of-64·E1-E+θ arm(+6), E3 Soul풍 대리 판독 arm(+5), E4 Spec-commit·Swatch·facet world 24개와 X-T2 판정(+20), E5·E2 정본 이름, P2-E4·P2-E5 기준, Kaggle 모드 실험 범위. 합계 418 → **486 T4-h** | §11.1 | C22, V3, V5, V8, V16, V18, V22, V26, V28 |
| **산술** | S1 112(DCA 포함), S1′ 1–2 TFU, S4 G3 105, G3 계획 1.8 T4-h·35–55분, 밤당 세계 10–16, 벤치 1.8–2.5, G-Adapter 1.9–5.8×10¹⁸, 카탈로그 FLOPs 4–6%·8–9%, 하루 생산량 4종, 데이터 장부 TAE 이중 계상 제거, 데모 합계 14–30, API 클립 수, 캡처 $4.4–10K, 렌즈 스택 75–190 ms, 105–130년, 80–370 T4-h, NC4·NC7·NC10, klein 약 1.5배, Qwen 0.8–1.0, SDXL 0.09, MRH 한계비용 INT8 병기 | §1.3, §1.8, §2.1, §3.2, §4.2, §4.4, §5.3, §5.6, §6.4, §6.5, §7.6, §8.4, §10.1, §11.7, §12.1 | C6, C8, C9, C29, C31–C33, C35, C37, C39, V10 |
| **게이트** | 세계 벽시계 G0 ≤1 h(E4만 ≤4 h), G3 ≤55분, Day 90 계획 T4-h ≤1.8. TP_S 항목에 "하이브리드 진단 게이트, 승리 조건 아님". Day 30 CamFM-Hand v0 | CT2, §8.2, §10.1 | C7, C25, V27 |
| **Phase 2** | 1회전 시간표를 P2(u) 두 열 표로 교체(회전 0 350블록 / 정상 1,000블록), GICC 18–21 s, Chrono-LoRA 15–25, BIS 변위 규칙, §9.1 FLOPs, Q26 | §9.1, §9.3, §5.6, §10.2, Q26 | C10, C11, V4, V23, V30 |
| **복원** | §2.5 인수분해 배당 D1–D6과 FLOPs/고유 텍셀 KPI, P2 끝 판정 해상도 계산(1024² 최적, 512²는 순위 매기기에만), R40 미감 점수 리워드 해킹, §1.6 Kaggle 메모리 행, X-LA 인물·동물 에셋 경로와 FD arm (ii) 표기, 이끼 판 "Z도 얕은 심도·보케" | §1.6, §2.5, §3.1, §3.3, §3.8.1, §3.8.6, §4.1, §7.8, §13 | V12, V13, V15, V17, V20, V25, V32 |
| **이름·표기** | 교정 장부 NC1–NC21, 크기는 괄호 표기, MoPri-S·HeroProxy-S/ChronoProxy-S, DP-0 = 절차 주석, 렌즈 별칭(AesHead, CRM, TL 규칙, Free-rider) 제거, PWMT "폐지"로 통일, 신규성 장부 하향(높음은 Corman–Shannon과 Teacher Ceiling Test만) | §0.3, CT9, CT11, CT12, §1.8, §5.2, §6.3, §12.1, 부록 A | C23, C27, C28, V11, V22 |
| **기타** | 라이브러리 페이스(주 25–30 세계), 리그 스왑 릴 D28·룩 50 = 수제 20 + `SOUL-INSPIRED` 30, IP 점검 범위(Hybrid-A+ 640장 포함), Kaggle 잡 8 h·약관 확인, SP_S 가정 65–75% 통일, P1 도전 목표 표현, Realify t 범위 통일 | §1.2, §3.8.5, §4.1, §5.3, §6.3, §7.6, §8.3, §11.3, Q4, R18 | C21, C26, C34, C36, C38, V19, V21, V29, V31, V33 |

**90일 배분 변화.** 30일 실험 276 → 344, 미니 벤치 350 → 440, 예비 917(21%) → **759(18%)**. A+ 승인 시 Hybrid-A+ 320–400을 쓰면 예비 약 360–440(8–10%).

### C.2 고쳐서 반영한 것 (지시와 다른 값이나 방식을 택한 이유)

| 지시 | 지시 내용 | 실제 반영 | 이유 |
|---|---|---|---|
| C6 | S4 G3 = 127 (다른 단계 393) | **S4 G3 = 105** | C37을 함께 반영해 S1이 90 → 112(DCA 16 포함)가 되면서 다른 단계 합이 415가 됐다. G3 ≤520을 지키려면 105다. −65%가 공격적이라는 점과 S4가 170에 머물 때의 결과(약 585 TFU, 게이트 미달 보고)를 §4.2에 적었다 |
| C1 / V2 | C1: 해금 목록을 메일 문구(비생성 ≤20M)에 맞춰 줄임. V2: 메일을 ≤25M, 픽셀 비출력 5종으로 넓힘 | **S1/S2 두 단계** | 두 지시가 충돌한다. S1은 C1을 따라 비생성 3종만, S2는 V2를 따라 픽셀을 출력하지 않는 생성형 2종을 "생성형임을 밝히고 따로" 묻는다. MRH 기하 헤드는 C1이 S2로 보냈지만 본체가 24M이고 가치가 낮아 요청 목록에서 뺐다 |
| C3 / V9 | C3: Hybrid-A 350을 예비에서. V9 (a): 재사용, 세계당 0.3–0.5, 버킷 420 | **재사용, 세계당 0.35–0.7, 버킷 440** | 측량 모드 재질은 룩과 독립이라 재사용이 원리에 맞는다. 다만 재촬영에 S8 Gaffer CEM 50 TFU와 L0 재bake가 다시 들어가 V9의 추정보다 높게 잡았다. 재사용이 실패하면 단독 건설 차액(최대 약 320)을 예비에서 쓴다고 명시했다 |
| C9 / V10 | Realify 생산량 교정 또는 FP16 유지 시 버킷 증액 | **INT8로 4.4만/T4-day, 80 T4-h 유지** | 데이터 SDEdit은 E0 (c)의 INT8 충실도 게이트 대상이다. FP16으로 돌릴 때의 비용(145–170 T4-h, 버킷 420)은 §6.4에 폴백으로 적었다 |
| C10 / V4 | C10: L3 8–12 h, BSRT 4천 장 × 8 TC 5.8 h. V4: P2(u) 표 그대로 | **P2(u) 표(BSRT 4k × 4 TC = 3.8 h)**, L3는 5.3 h(c_TC 0.5 s)–6.9 h(§1.4의 0.65 s) 병기 | 원문 P2(u) §10.1과 대조했다. BSRT의 TC 수는 원문이 4다. L3는 커버리지 60%가 원문에 있어 C10의 전수 가정보다 작다. c_TC 차이만 병기했다 |
| V12 | KPI "우리 약 1.4×10⁸ vs Seedance ≤3×10⁸" | **1.6×10⁸ vs 2×10⁸–10⁹, 내부 KPI 전용** | 원문 CB(u) O9가 이 숫자를 교정했다. 지시의 값은 CB(o)의 교정 전 값이다 |
| C26 / V21 | C26: Day 21 룩 20–30종. V21: 50종 = 수제 20 + `SOUL-INSPIRED` 30, Day 28 | **V21 방식 + §7.6은 C26 방식(v0 ≥30, D60까지 60여 개)** | 릴의 메시지(룩 50종)를 지키면서 v0 동결 일정과 맞는다 |
| C8 | M4를 1–3 epoch로 | 반영 + **NC14 근거를 "샘플-패스 4–12만"으로 통일** | NC14의 "3–5 epoch"(RR(u), 더 작은 데이터)와 모순되지 않게 단위를 샘플-패스로 바꿨다 |
| V5 / V6 | 예비 892, E0 50 | **E0 62, 예비 759** | V6의 NaN 감사 12 T4-h와 V8·V16·V3, Hybrid-A 버킷을 함께 반영한 최종값이다 |

### C.3 반영하지 않은 것

- **없음.** 다만 두 지시는 해당 위치에 고칠 대상이 없어 일부만 반영했다.
  - C34의 "§0.3 CT29 근거": CT29에는 780이라는 숫자가 없다(5.6배·11배는 771 기준으로 이미 맞다). NC10만 고쳤다.
  - C23의 "Aes": §2.2와 §5.3에서 MRH 출력 헤드 목록 안의 "Aes"는 모델 이름이 아니라 MRH 내부 헤드 이름이라 그대로 두었다. 독립 점수로 쓰인 곳(의사코드, DP-0 보상, Corman Optimizer)은 MRH-Aes로 고쳤다.

### C.4 v2.1 사후 검증 반영

사후 검증 23건(구조 5, 숫자 8, 방화벽 10)을 원문과 대조했다. 23건 모두 결함이 실재했다. 21건은 그대로 반영했고 2건은 일부만 반영했다. 메일 관련 5건은 실제 발송 스레드(티켓 #317768)와도 대조했다. 구조 감사와 숫자 감사가 §6.5 합계를 따로 잡아 같은 수정 하나로 처리했다.

| 영역 | 바뀐 것 | 위치 |
|---|---|---|
| **산술** | §6.5 합계 293–328 → **295–330**, FP16 폴백 380–420 → **360–420**. §5.6 P2 2a–2b 315–325 → **286–353**(6.0–7.4일), 전체 546–643 → **512–676**(10.7–14.1일, 90일 용량의 12–16%). D5 한계비용을 4K120 10초 7–15 T4-분(8K120 25–60)으로. G3 미달 시 계획 T4-h 2.0 → 1.85. R3 최악 경로를 계획 6(순수 3.0)으로. Kaggle P2 실험 70 → 50. IP 재생성 몫의 이름을 배분표와 맞췄다 | §2.5, §4.2, §5.6, §6.4, §6.5, §8.3, §11.1, R3, R32 |
| **Kaggle 버킷** | §11.1 Kaggle 범위를 §11.5 버킷(1배 126, 2배 74)에 맞추고 합계를 적었다. 버킷이 아니라 범위를 고쳤다. 2배 열은 버킷을 범위에 맞추면 다른 버킷을 절반 가까이 깎아야 해서다. 빠진 실험은 Day 30 이후 같은 열의 학습·세계 솔브 버킷에서 돈다 | §11.1 |
| **평가 앵커** | 엔진 교사 잠금, 승급, 엔진 계약 6은 E1-E와 ES로만 판정한다. "하이브리드 트랙 개방" 조건을 지우고 개방은 창업자 결정으로 정의했다. TP_S\*는 게이트가 아니라 보고 항목이다. §3.7.2 (d)와 Q6을 P7·부록 B의 "보고 전용"에 맞췄다. 발송본에 용도 제한이 없었다는 사실은 "미확인"으로 적었다 | CT2, CT41, §3.4, §3.6 A6, §3.7.2, §3.8.5, §8.1, §8.2, §11.1 E1, §11.3, §11.4, Q6 |
| **발송 기록** | 06:57Z 후속 메일(폼 사용 불가, 티켓을 영업팀에 전달 요청)과 회신 위치(Pricing 페이지 Enterprise 항목)를 적었다. 발송본의 실제 요청·질문·대가, Plan S 문구 해석 규칙(승인은 S1 ① CamFM-CRM에만 적용)을 넣었다. "D14 메일 후속"은 "D7/D14 티켓 전달 확인"으로, 질문 목록은 "물은 것 3개"와 "아직 묻지 않은 것"으로 나눴다. "메일 회신(부록 B 5)"은 "후속 문의 회신"으로 고쳤다 | 부록 B, §3.7.4, §3.7.5, §3.8.6, §11.3, §11.6, Q13, Q16 |
| **크레딧·공개** | "Look by Higgsfield Soul" 표기 규칙(Soul이 닿은 작품에만)을 넣었다. Q16에 문구 수정과 웹사이트·도메인 확보를 더했다. Soul이 닿은 작품의 θ_look은 Q13 회신 전까지 해시만 공개하고, Hybrid-A 공개는 Q13 회신 뒤로 했다 | CT38, §7.9, §8.1, §8.3, R39, Q16 |
| **의사코드·방화벽** | 시그니처에 gloss를 넣고 Engine 트랙과 H-A 벤치의 광택 금지 assert를 더했다. Instantiate 직후 W.prov.soul을 대입하고, prov.soul ≠ none인 세계의 의미·재질 패스 교사 호출을 격리했다. §3.7.3 4의 soul=1 정의를 그 세계의 모든 엔진 교사 호출로 넓혔다. §6.6 링버퍼·VAE 쌍 검사에 `SOUL-INSPIRED`를 더했고, M1의 "우리 렌더"는 prov.soul = none만 쓴다 | §3.7.3, §4.3, §5.3 M1, §6.6 |
| **X-LA 후속 근거** | 후속 문의에는 승인 전에 얻을 수 있는 숫자(E3 (c), g_con)만 붙인다. K 대 K+ 비교는 S 단계 협의 자료다 | §3.8.6 |
| **구조** | 마무리와 참조 파일을 문서 끝으로 옮겼다. 부록 C 머리의 "9건"을 "15건(C.2의 9개 항목)"으로 고쳤다. 스케일 원칙 S1–S9를 SC1–SC9로, R38을 "(§10.3 SC9)"로 바꾸고, 머리 표기에 S1·S2 구분 문장을 넣었다 | 머리 표기, §10.3, R38, 부록 C |

**일부만 반영한 것.**

- **Plan S 하위 단계 이름(S1/S2)은 그대로 둔다.** 감사는 PS1/PS2 같은 개명을 제안했다. 계보 브랜치(`lineage:soul-s1`, `lineage:soul-s2`), 부록 C 이력, 후속 문의에 보낼 구분이 모두 S1/S2를 쓴다. 이름을 바꾸면 브랜치와 이력이 어긋난다. 그래서 스케일 원칙만 SC로 바꾸고, 파이프라인 단계와 다른 이름이라는 문장을 머리 표기에 넣었다.
- **래칫 내보내기 위치는 그대로 둔다.** 감사는 래칫·Exhaust 내보내기를 최종 commit 뒤로 옮기자고 했다. Engine 트랙 광택 금지 assert를 넣으면 track = E 세계가 Soul에 닿을 경로가 없다. 그래서 위치는 두고 이유를 주석으로 적었다.

---

> **마무리.** 비디오 모델은 같은 세계를 240번 다시 그린다. 우리는 세계를 한 번 짓는다. 덩어리는 MassingFM이 놓고, 옷은 엔진 교사가 대역마다 입히고, 결은 TexelFM이 짠다. 그 한 번도 카메라가 기록할 수 있는 만큼만 짓는다.
>
> v2.1에서 배운 것은 미감에도 장부가 있다는 사실이다. A/B 50축에서 Soul이 41번 이겼고, 그중 32번은 우리 카메라와 빛과 필름이 원래 소유했어야 할 축이었다. 나머지는 개념이었다. 그래서 촬영 취향은 사람 손으로 옮겨 물리로 재현하고, 개념은 아트 디렉터에게 빌리고, 그 아트 디렉터의 붓이 우리 서기들의 교과서에 닿지 않게 방화벽을 친다. 빌린 눈은 둘, 배운 손은 우리 것이다.
>
> 세계 하나에 2 T4-h, 4K120 10초에 스마트폰 한 번 충전할 전기가 든다. 개소리 같지만 장부는 맞는다. 그리지 말고, 찍어라. 대신 아껴서 찍어라.

**참조 파일:** [`PARADIGM.md`](PARADIGM.md) (v1 기준 문서), [`README.md`](../README.md), [`research/landscape-2026-10.md`](research/landscape-2026-10.md), [`lenses/`](lenses/), [`v2-2xT4/`](v2-2xT4/) (2×T4 워크숍 원안·프리모템)
