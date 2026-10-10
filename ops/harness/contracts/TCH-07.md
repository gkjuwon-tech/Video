# TCH-07 교사 선정 시험 — TCT-v3 NVF · ES_v3 결정식 (Z 현직 vs klein 도전자)

**목적.** models.upg §1.2–1.4(CT-M1)·§2(TCT-v3)의 **판정 논리**를 사전 등록 가능한 순수 함수로 고정한다. 사람 평가·GPU 생성·경쟁 API는 이 단위 밖(측정값을 받기만 한다). 기본 경로는 **Z 유지**: klein은 증거가 게이트를 통과하고 결정식이 이길 때만 바뀐다.
**소유.** `buc/teacher/tct.py`, `tests/tch_07/`(동결). 의존: TCH-08(`KleinSpec`, `choose_l2b_path`, `KLEIN_TABLE`), TCH-02(`DEFAULT_TABLE`), FND-03(`det.hash_parts`). 순수 파이썬+numpy, 부작용 없음, `eval/` import 금지(EVL-04는 별도 프로세스 레인이 낸 숫자만 받는다). 소유 밖 요청: 없음. GPU 테스트: 없음(측정은 E0/E1-v3 레인 몫).
**공통 규칙.** 모든 비율·문턱 비교는 `1e-12` 허용 오차의 경계 포함(`>=`/`<=`). 검증 실패는 `TctError(ValueError)`(하위: `FirewallError`, `PreRegistrationError`, `SelectionError`). 지표는 "클수록 좋다"로 부호를 맞춘 float를 받는다.

## 공개 API (`buc.teacher.tct`)
- 상수: `INCUMBENT="z-image-turbo"`, `CHALLENGER="flux2-klein-4b"`, `CRITERIA=("K1".."K7")`, `WEIGHTS_L1`(.10,.10,.20,.20,.15,.10,.15), `WEIGHTS_L2B`(.30,.20,.05,.15,.10,.10,.10), `Z_CLIP=.5`, `TIE_PP=.02`, `SWAP_PP=.05`, `NVF_R_ENTRY=.30`, `WORLD_MIX`(.5/.35/.15), `WINDOW_DAYS=(45,75)`, `ENGINE_FREEZE_DAY=60`, `LOCK_DAY=15`, `GATE_DAY=18`, `SEAM_RATIO_MAX=1.3`, `W4A4_FACTOR=.846`, `T4H_PER_TFU=1/1385`, `SELECTION_SETS`, `FORBIDDEN_SELECTION_SETS`.
- 사전 등록: `PreRegistration(w_l1, w_l2b, tie_pp, swap_pp, nvf_r_entry, mix).validate()/key()->64hex`; `PREREG`. 합 1·비음수·길이 7·`tie_pp<swap_pp`가 아니면 `PreRegistrationError`. `key`는 어떤 값이 바뀌어도 달라진다.
- 점수: `normalize(m, m_ref)` = `clip((m-m_ref)/|m_ref|, ±.5)`(키 정확히 K1..K7, 유한, 기준 ≠0); `engine_score(m, m_ref, weights)`(ES는 Z 대비 분수, 0.02=2%p); `world_t4h(per_grade, mix)`(1.0/1.6/1.9 → 1.345).
- 게이트: `GateInputs(license_ok, fp16_nan, fp16_trials, fp16_ref_nan, fp16_ref_trials, mem_peak_gib, host_completed, nvf_r)`, `gate_report()->GateReport(ok, failed)`(순서 license, fp16, memory, nvf_r_entry; 시행 수 <10,000/2,000은 실패, 메모리 ≤15 GiB, nvf_r ≥.30). `klein_license_gate(KleinSpec)->LicenseGate(ok, reason)`(`validate()` 예외를 False로).
- 방화벽: `check_selection_inputs(tags)` — `SELECTION_SETS`(R-real, R-syn, O-photo-dev, O-style-2a, O-style-2b, E-escher) 밖은 `FirewallError`(COMP·홈 출력·S-closed 포함). `home_away_gap(h,a)->(gap, gap>.05)`, `seam_ok(seam, inner)`.
- 등급: `grade_engine(nvf_hc, nvf_obs)`(녹 ≥.60∧obs ≥.95, 황 ≥.55 — obs 미달은 황으로 강등, 그 외 적), `L2bEvidence(nvf, z_style, delta_photo, tau_photo)`, `grade_l2b`(녹 ≥.55∧z≤1.5∧Δ≤τ, 황 ≥.50, 적), `grade_l2a`(녹/황만), `sp_i2v(hc,tp)`=sigmoid(logit+logit), `required_tp(hc)=1-hc`, `nvf_obs(views[(de00_median, edge_drift_px)])`(≤2.0, ≤1.0 px; NaN 불합격), `z_style(d, self_d)`, `self_tau(v)=2·std(ddof=1)`, `wilson(k,n)->Rate`, `bradley_terry(items, results[(승,패)], prior=0)->dict[str,log강도(평균0)]`, `bt_prob`.
- 선정: `Candidate(id, metrics, t4h_world, gates, sources, ref_contract, swap_ok, nvf_s, ref_t4h_world)`; `select_teachers(candidates, ring_r: Mapping[id, L2bEvidence], *, prereg=PREREG, expected_prereg_key=None, adapter_ok=True)->Selection(engine, ref, outcome, path, es_l1, es_l2b, score, eligible, excluded, tie_band, prereg_key)`, `.to_dict()`(JSON), `.lock()->LockedConfig(engine, ref, outcome, key)`(frozen).
  **규칙.** (1) Z는 기준(`m_ref`)이라 반드시 포함. (2) 게이트 통과 집합 A; 탈락 사유는 `excluded`. (3) `score=ES_L1/T4h`; **ES>0인 후보가 있으면 비율 최대, 없으면 ES 최대**를 T*로. (4) T* 와 |ΔES_L1| ≤ 2%p인 밴드에서 **T4h 최소**(동률: ES 큰 쪽 → Z → id). (5) Ring-R이 L2b 녹색이면 O1(`ring_r`). 아니면 R={A∖엔진, ref_contract∧swap_ok, nvf_s−ring_r ≥ +5%p}; 비면 O4(`ref_adapter`, 어댑터 불가면 `cards_2_5d`); 아니면 T_ref=argmax(Δnvf/ref_t4h_world) → |ΔES_L1| ≤ 2%p∧더 빠르면 **O3**(엔진 교체, ref 없음) 아니면 **O2**(`klein_ref`). `path`는 TCH-08 `choose_l2b_path`와 같다. 입력 순서와 무관·결정론. 방화벽은 후보의 `sources`에, 사전 등록 키는 `expected_prereg_key`에 적용(불일치 `PreRegistrationError`).
- COMP 게이트(잠금 뒤에만): `CompRecord(input_id, grade, registered, win)`, `comp_gate(locked, records, *, nvf_obs_rate)->CompReport`: 1차(등록 성공만)·폴백 포함 두 수치, 등급은 **둘 중 낮은 값**, 등록 실패율 >20% 플래그, 등급별 표, 적색이면 `needs_human_q3`. `LockedConfig`가 아니면 `TctError`.
- 교체 창: `can_replace(day, "engine"|"ref", delta)` — 엔진은 D45만, 참조 슬롯은 D45·D75, 둘 다 Δ ≥ +5%p.
- 계획: `ARMS`(Z-plain, Z-Ring, klein-plain, klein-ref, Z-Ring+theta, Z-RingR; `Arm(engine, seeds, nfe, tfu)`, NFE는 σ 표 길이), `FILL_ARMS`, `arm_seeds`, `fill_cost_tfu(n_poses, *, w4a4=True)`(990포즈 ≈ 40,059 TFU), `tfu_to_t4h`(≈28.9), `LITE_*`.

## 불변식
COMP·홈 출력은 선정에 못 들어온다 · 가중치는 사전 등록 키로만 바뀐다 · 라이선스 미확인/9B klein은 항상 A 밖(Z 유지) · 같은 입력 → 같은 `to_dict()` · COMP 게이트는 `LockedConfig` 없이 못 돈다.

## 수용 기준
`tests/tch_07/` 전부 통과(CPU, <10 s, 동결): 상수·사전 등록 키, z 정규화·ES 손계산(0.27/0.14), 게이트 경계, 방화벽, 등급 경계, 로짓 가산성, Wilson·Bradley-Terry(닫힌 식·복원), 잠금 사례(2%p 경계, 비율 대 ES), O1–O4와 `choose_l2b_path` 일치, 순열 불변, COMP 보수 등급·등록 실패 플래그, 교체 창, 비용 29 T4-h.

## 열린 해석 (Fable 몫, 잠정값)
ES≤0 처리(비율 무의미 → ES 최대), 밴드를 T* 기준으로 잡은 것, 보수 등급(min), obs 미달=황, `NVF-S`=L2b의 NVF 항, `self_tau`=2σ(ddof=1)은 사양이 모호해 **작성자의 선택**이다.

## 하류 이음매
D15 잠금은 `Selection.lock()`의 `key`를 계보 태그로, D18 보고는 `CompReport.to_dict()`, D45/D75는 `can_replace`. EVL-04 `LpipsClient` 수치는 호출자가 지표로 넣는다.
