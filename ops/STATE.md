# 운영 상태

## ⏸️ 일시 정지 (D1 03:45Z, 창업자 지시: "공장 다 스톱, 사용량 99%, 월요일 9시에 보자, 돈 안 나가게")
- 재개: **2026-10-12(월) 21:00 KST = 12:00Z**, Claude 주간 한도 리셋 뒤.
- 멈춘 것: 공장장 정기 트리거 `trig_01GL7Ww48KbFj51yDHhG5EQW`(꺼짐), 5시간 보고 `trig_01BFrwWLxm7wufRA4SEwgYNQ`(꺼짐), night2 체크인(삭제), 공장장 세션 중지 요청, M14 빌드 워크플로 중지, M3 빌드 워크플로(빌드 단계에서 멈춤).
- GPU 55234730: M6 라벨·로그를 HF `backup/day1_stop/`에 백업 후 자체 정지 → 확인되면 파기(저장비도 0).
- 완성 모델: M1(`ckpt/night1/tae_t/`), M13(`ckpt/day1/m13/`), M2(`ckpt/day1/m2/`, 30k 스텝, +7.4/+5.6 dB, 야크 +4.7/+3.6 dB). M6는 라벨 단계 중 정지(라벨 백업으로 재개 가능).
- 재개 순서: (1) 트리거 2개 다시 켜기, (2) M3 워크플로 `Workflow({scriptPath: .../m3-spike-pull-forward-wf_e4247626-c45.js, resumeFromRunId: "wf_e4247626-c45"})`로 이어서(읽기·명세 캐시, `ops/gpu/day1/m3_mrh.py` 5.1k줄 WIP 커밋됨), M14 워크플로도 `wf_4a2dc1bf-673` 재개, (3) 새 5090 띄우고 M6(라벨 HF에서 복원) → M3 → 파이프라인 야크 v2.
- **돈:** Vast 크레딧 약 $29(03:00Z 기준). 창업자: 22일 이후 최대 30만원 충전 가능. 그 전까지 24시간 가동 금지, 할 일 있을 때만 켜기(하루 평균 약 4.7 GPU-시간). 5090 $0.508/h.

## 🏭 15일 스프린트 (D0 = 2026-10-10)
- **하네스 v1 가동:** 헌법 `ops/harness/README.md`, 백로그 `ops/harness/backlog.json`(167단위, 수직 슬라이스 45개 우선), 선택기 `pick.py`, 기계 게이트 `gate.py`(동결 테스트 해시·compile·ruff·pytest), 상태 갱신 `update.py`, 장부 `ledger.jsonl`. 교대조 워크플로 `ops/workflows/buc-shift.js`(args batches 4, k 2). 교대조가 끝나면 오케스트레이터가 바로 다음 교대조를 띄운다.
- 모델 배정: 선택·게이트·커밋 Haiku, 계약서+테스트 Sonnet(난이도·위험 ≥4는 Opus medium), 구현은 백로그 impl_model, 수리 Sonnet→Opus, 위험 ≥4 검토 Opus low. Fable은 교대조에서 쓰지 않음.
- 보고: 5시간마다(트리거 `trig_01BFrwWLxm7wufRA4SEwgYNQ`, cron 0 */5). 창업자 질문 시 즉시.
- **공장장 세션:** `session_01XA9TirqQtqJiYRyewCK8uv`(Sonnet 5.5, 태그 buc-foreman)이 교대조를 상시 가동한다(지침 `ops/harness/FOREMAN.md`). 이유: 오케스트레이터 세션에서 창업자가 실행을 중단하면 백그라운드 워크플로가 같이 죽는다(2회 발생). 공장장은 GPU·키를 다루지 않는다. 진행 신호는 `ops/harness/foreman.json`.
- **상시 승인 A-001 (2026-10-10T14:51:47Z):** 창업자가 BUC 세션들끼리 세션 생성과 Workflow 실행을 매번 묻지 않고 하도록 승인했다. 원문·해시·서버 이벤트 ID는 `ops/harness/AUTHORIZATION.md`. GPU·결제·키는 여전히 오케스트레이터만.
- **D1 낮 GPU 학습 (55234730, 러너 `ops/gpu/day1/runner.sh`, 마감 13:00Z):** **M13 Rig-ID 완료**(2.93M 파라미터, 합성 리그 4만 샘플, 20 에폭 231초, 검증 정확도 75.4%·2순위 96%, 14클래스 우연 7%). 헷갈리는 쌍: 돌리↔카마운트, 짐벌↔삼각대, 시네드론↔수중, 테크노크레인↔지브(물리적으로 비슷한 부드러운 이동). HF `ckpt/day1/m13/`. **M2 LatentDown/Up 학습 중**(각 0.92M, 01:48Z 기준 step 5.8k; 평가 step 1478에서 Down PSNR 29.5 vs 기본 리샘플 26.3(+3.2 dB), Up 30.5 vs 24.5(+6.0 dB)). M2 마커가 vastai 로그에 안 보이는 현상 있음(원인 미상) → `ops/gpu/day1/extra.sh` 원격 슬롯으로 진단. **M6**: 리뷰 수리가 중단돼 Sonnet이 재수리 중, 끝나면 `ops/gpu/day1/M6_READY` 푸시. **파이프라인 첫 야크** `ops/gpu/day1/pipe_yak.py`(skeleton 1샷 + scaffold 6샷, 임시 깊이 Depth-Anything-V2-Small은 평가 전용 라벨) → extra.sh로 백그라운드 실행, 결과 HF `video/pipe_d1/`. **결과(02:05Z): skeleton S1 + scaffold S1–S4 완료(1920×826, 24fps, 샷당 약 2분).** S1 푸시인은 볼 만함(뿔 옆 약간 번짐). S2·S3·S4는 뿔·얼굴 뒤 가림 해제 영역에 얼룩말 줄무늬 늘어남(고무막) + S3 오른쪽 검은 테두리. 원인: 구멍 메우기 없음(WLD-08 교사 경로가 `TeacherNumericError: 변환기 출력 NaN`으로 폴백), RD-18 매팅 low_conf 기각. skeleton S1은 스텁 렌더에 검은 가로 띠(스텁 버그, 무해). **교사 NaN은 `buc/teacher/zimage.py` 경로 버그로 추정(같은 박스에서 diffusers ZImagePipeline은 정상) → M3 탭 추출도 같은 경로라 Sonnet 진단 결과(02:40Z): CPU에서 diffusers와 호출 규약 1e-6 일치, 원인 미증명. 가장 유력한 용의자 = 결정론 모드가 켜는 fill_uninitialized_memory(빈 텐서를 NaN으로 채움). 수리: 그 채움만 끄고(결정론 유지), 비유한 출력 시 first_bad·tried 진단 + 단계별 재시도(math SDPA, GEMM 플래그). 게이트 TCH-01/02/03·WLD-08 통과, 커밋됨. GPU에서 확인 필요(M6 라벨 단계가 끝나 Z-Image 자리가 나면).** M6는 자체 diffusers 로더라 영향 없음, 02:08Z M6_READY 푸시.
- **중간보스 LTX-2.5 비교 (D1, COMP 평가 전용·학습 금지):** 창업자 지시로 야크 6샷을 LTX-2.5(22B distilled, bf16, 1920×832, 24fps, 시드 42)로 생성. 별도 인스턴스 **55306560**(A100 PCIe 80GB, 일본, $1.05/h, 디스크 220GB, 라벨 buc-comp-ltx25). 부트 루프 `ops/gpu/comp/boot.sh`가 `ops/gpu/comp/ltx_yak.sh`를 받아 실행(푸시하면 원격 수정), 끝나면 또는 5시간 뒤 스스로 정지. 결과 HF `comp/ltx25/`. 라이선스 LTX-2.x Community License(연매출 $10M 미만 무료). 보스 순서: 바닥선(깊이 워프) → 중간보스 LTX-2.5 → 최종보스 Seedance. **결과(D1 00:50Z): 6샷 완료(샷당 약 2–3분, 1차는 gcc 없어 전부 실패 → 수정).** 화질은 바닥선보다 압도적(구멍·번짐 없음, 털 클로즈업 디테일, 아크에서 옆모습 그럴듯한 발명). 카메라 지시 준수는 낮음: S1 과한 돌진, S2 슬라이더 대신 푸시인, S4 크레인이 탑다운까지 과하게, S5 '카메라 고정' 무시하고 줌인·랙 포커스 없음, S6 돌리 줌 아님(그냥 푸시인), S3 아크만 대체로 맞음(회전 과다). 실제 기준선은 바닥선이 아니라 LTX 화질이다.
- **데이터 계획 (창업자, D1):** 아이폰 촬영 숙제 기본 취소, 공개 데이터로 대체. `docs/DATA-PLAN.md`(라이선스 검증 워크플로 public-data-hunt 결과). 핵심: M3 실사=DIODE(MIT), M11=TUM VI(CC BY), M5=Poly Haven·ambientCG·MatSynth CC0, M8=Infinigen·OmniObject3D·GSO·ABO·Objaverse(CC-BY/CC0 필터), M1=raw.pixls.us CC0·SIDD. 대형 실내 깊이셋(ScanNet·Matterport·Replica 등)은 전부 비상업이라 탈락. ARKitScenes는 Apple 서면 확인 전까지 평가 전용.
- **승리 조건 (창업자, D1):** "일관성만으로 이기면 패배, **순수 화질로 이긴다**." 블라인드 패널 질문은 화질 하나뿐. 매일 A/B도 화질 지표(선명도·미세 질감·결함) 중심. `docs/DEMO-D15.md` §5 갱신.
- **D15 데모 지정 (창업자, 2026-10-10T15:00Z):** 초원의 하얀 거대 야크 이미지 한 장으로 멈춘 세계 + 다양한 카메라 무브. 사양 `docs/DEMO-D15.md`, 원본은 HF 금고 `demo/d15_hero_yak.webp`(공개 레포에 넣지 않음). RDR-01·RDR-08·RD-18·RD-05를 수직 슬라이스로 승격, 관련 단위에 `demo: d15-yak` 태그.
- **공장장 체크인:** 공장장은 send_later 권한이 막혀 있어 오케스트레이터가 매시간 :03에 공장장 세션을 깨우는 트리거 `trig_01GL7Ww48KbFj51yDHhG5EQW`를 걸었다. 창업자가 공장장 세션에서 직접 교대조 상시 가동을 지시했다(14:57Z).
- **밤샘 잡 night1 (창업자 지적: "학습도 영상도 안 하면 뭘 하냐" → GPU 밤새 가동):** 명세 `ops/gpu/night1/SPEC.md`. ① 야크 바닥선 영상 6샷(순진한 깊이 워프, 평가 전용) ② Z-Image 교사 코퍼스(이미지+VAE 잠재, HF `corpus/night1/`) ③ M1 TAE-T 스파이크 학습(HF `ckpt/night1/tae_t/`). 마감 22:45Z(07:45 KST) 뒤 스스로 정지. 코드는 워크플로 night1-gpu-jobs(Sonnet 구현·Opus 검토·Sonnet 수리, 9에이전트, 테스트 110개 통과)가 작성. **배포: `ops/gpu/runner.sh`(= night1/runner.sh) 푸시 16:0xZ.** GPU 상주 루프가 5분 안에 받아 실행. TAE는 --micro-batch 16 --mem-gb 4.5(코퍼스와 VRAM 공유). 진행은 `python3 ops/gpu/vlogs.py 55234730`의 N1_* 마커(주의: 이번 밤 러너는 `cut` 버퍼링 때문에 코퍼스·TAE 마커가 늦게 보인다 → HF 금고의 corpus/night1/·ckpt/night1/tae_t/ 업로드로 확인. 러너는 awk fflush로 고침, 다음 배포부터 적용). **결과(16:30Z):** 바닥선 6샷 GPU 130초 완료·업로드(video/night1/), 코퍼스 첫 샤드 500장 16:24Z(약 23장/분, 범주 비율 명세대로, 잠재 std 1.05·NaN 0). **M1 TAE-T v0 첫 체크포인트 16:57Z:** 스텝 1,464(약 15분), 평가 64장 PSNR D(z_t) 29.4 dB / D(E(x)) 28.7 dB, latent MSE 0.135. HF `ckpt/night1/tae_t/`.
- **아침 상황 (2026-10-10T23:15Z = 10/11 08:15 KST):** 하네스 57/167 완료(밤사이 +48), 수직 슬라이스 43/49(남은 것 RD-05·RD-08·RD-21·WLD-03·WLD-08·WLD-17), 막힘 0. RD-02 막힘은 동결 테스트 헬퍼 버그(np.cross 2D, numpy 2)였다 → 오케스트레이터가 고치고 재동결. night1 완료: 교사 코퍼스 **4,755장**(실패 0, HF corpus/night1/ 샤드 10), **M1 TAE-T v0 102,430스텝: PSNR D(z_t) 38.5 dB / D(E(x)) 최고 36.7 dB**(2.52M 파라미터, ckpt_best.pt), GPU 07:45 KST 자동 정지(55234730 stopped), Vast 크레딧 $31.08(밤새 $3.92).
- **⚠ 7일 사용량 경고 (seven_day allowed_warning, 리셋 2026-10-12T12:00Z):** 공장장 누적 API 환산 $143, Opus 호출 34%. 조치: buc-shift.js **절약 모드 기본**(Opus는 난이도·위험 5의 계약서·검토만, 2차 수리 Sonnet), 공장장 체크인 2시간 간격(`3 */2 * * *`), 교대조 완료 즉시 재가동 금지. 창업자 답: **주간 84% 사용**(23:20Z). 결정: 수직 슬라이스 남은 6단위(RD-05·RD-08·RD-21·WLD-03·WLD-08·WLD-17)만 `slice_only` 교대조로 끝내고, **공장 일시정지 → 2026-10-12T12:05Z 자동 재개**. GPU는 정지 유지(크레딧 $31 ≈ 61 GPU-시간, Phase 1 학습 예산 50–90시간이라 아껴야 함). pick.py `--slice-only`, buc-shift.js args `slice_only`. 일시정지: 공장장 체크인 trig_01GL7Ww48KbFj51yDHhG5EQW·5시간 보고 trig_01BFrwWLxm7wufRA4SEwgYNQ 꺼 둠. 재개 trig_0147LtS5z2a265uEmcMshC6V(10/12 12:05Z, 이 세션), 슬라이스 결과 확인 trig_01TnGcMuZPN9HWhgoenaJiuW(10/11 02:18Z).
- **장부 병합:** `.gitattributes`에 `ops/harness/ledger.jsonl merge=union`.
- 오케스트레이터 세션 워커가 20:40Z–23:10Z 재시작으로 멈췄었다(공장장은 계속 가동).
- **E0 통과 (15:26Z, 인스턴스 55234730):** Z-Image-Turbo BF16 on RTX 5090 = **1024² 3.04 s/장**(9 스텝, 0.38 s/fwd), 512² 0.9 s, 로드 후 VRAM 20.7 GB, 정점 23.3 GB, 다운로드 237 s, diffusers 0.41.0, torch 2.9.1+cu128, CPU 16. 결과 `ops/gpu/e0_result.json`. 시험 이미지는 vastai logs가 줄을 489자에서 잘라 회수 실패(마커는 400자 이하로).
- **GPU 교체 (15:17Z):** 한국 5090 55227600이 50분간 기동 실패 → 자동 교체 **55234730**(한국, machine 143424, $0.508/h 디스크 포함). 옛 55227600은 창업자가 권한을 허용한 뒤 오케스트레이터가 파기했다(15:3x Z). ensure_up의 destroy는 [y/N] 확인 프롬프트 때문에 실패했었다 → 확인 입력 후 목록으로 검증하도록 고침.
- **밤사이 GPU 체인:** `ops/gpu/night_chain.sh` = 기동 보장(ensure_up) → E0 결과 회수(`ops/gpu/e0_result.json`) → **GPU 정지**(유휴 과금 방지, 디스크만 과금) → 커밋. 러너 단위(RUN-*)가 준비되면 `vastai start instance ID`로 다시 켠다.
- **한도 리셋 직후 깨우기:** `trig_01PcFv7tauA9qF4ycUjpFgc9`(1회, 19:12Z)가 공장장을 깨운다.
- **GPU 기동 감시:** `ops/gpu/ensure_up.py`가 인스턴스가 기한(생성 후 50분) 안에 안 뜨면 파기하고 검증된 5090(≤$0.50/h, 중국 제외)을 새로 빌려 같은 부팅 스크립트로 띄운다. 현재 인스턴스는 `ops/gpu/instance.json`.
- 하네스 설계 워크플로(buc-harness-design)는 창업자 지시로 폐기. 조사 산출물(work/wf5/r-*.md, units.json)은 백로그 원천으로 사용.

## ▶️ 재개 (창업자 복귀, 결제 예정)
- 지금 단계: **15일 스프린트용 에이전트 하네스 설계** (워크플로 `buc-harness-design`, 스크립트 `ops/workflows/buc-harness-design.js`). 산출물: `work/wf5/r-models.md`, `r-practice.md`, `r-inventory.md`, `design-cost.md`(Opus), `design-quality.md`(Fable), `harness-spec.v1.md`, `redteam.md`, **`harness-spec.md`(최종)**.
- 창업자 원칙: 구현은 하위 모델(Sonnet 5.5, Haiku 5.5), 설계와 검사는 상위 모델(Opus 5.5, Fable 5.1). 비용·시간 때문에 전부 Opus로 하지 않는다.
- 모델 가격(공식 레퍼런스, 2026-10-06): Fable 5.1 $10/$50, Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 (MTok 입력/출력).
- 환경 제약: 이 컨테이너 CPU 4개 → 워크플로당 동시 에이전트 2명. 병렬은 워크플로 다중 실행 또는 형제 클라우드 세션으로.
- **🟢 GPU 가동 중:** Vast 인스턴스 **55227600** = 한국 RTX 5090 꿀매물(machine 35043, $0.236/h 디스크 포함, CUDA 12.8, 이미지 pytorch 2.9.1-cuda12.8). 라벨 `buc-e0-kr5090`. 부팅 즉시 `ops/gpu/e0_bringup.sh`(공개 raw URL에서 당겨 옴) 실행: Z-Image-Turbo BF16 기동·속도·VRAM·출력 정상 여부 → `vastai logs`로 회수(`ops/gpu/wait_e0.sh`). 끝나도 `sleep infinity`로 작업대로 유지.
- **Hugging Face:** 창업자 토큰(쓰기 권한)을 이 컨테이너 `~/.cache/huggingface/token`(600)에만 저장, 레포에는 없음. 체크포인트·산출물 금고 = 비공개 저장소 **`sdsdsdsdsdsdsd123/buc-artifacts`**. 같은 토큰을 Vast **계정 환경변수 HF_TOKEN**으로 등록해 새로 만드는 GPU 인스턴스에 자동 주입(현재 인스턴스 55227600에는 재생성 전까지 없음). 스프린트 후 재발급 권고.
- **Vast API 키:** 창업자가 채팅으로 준 키를 이 컨테이너의 `~/.config/vastai/vast_api_key`(권한 600)에만 저장. 레포에는 절대 없음. `ops/persist.sh`에 비밀 검문소(커밋 내용에 키 값이 섞이면 중단). 컨테이너가 리셋되면 키가 사라지므로 창업자에게 다시 받아야 한다. 스프린트 후 키 재발급 권고.
- (이전) **GPU 최종 (창업자, 같은 날 갱신):** 창업자가 Vast.ai 인증을 해결하고 **$35 충전 완료**. 다시 **Vast.ai**로 간다(RunPod 불필요). 한국 5090 $0.213 매물은 그사이 사라짐. 현재 후보: 5090 캐나다 BC $0.414(신뢰도 0.998, CPU 32, RAM 31GB) / 4090 한국 $0.376(CPU 32, RAM 63GB). 중국 호스트는 GitHub·HF 접속 위험으로 제외. $35 ≈ 5090 약 85시간. 시크릿 `VAST_API_KEY`, `HF_TOKEN` 대기. D0 전에는 렌트하지 않는다(유휴 과금 방지).
- (이전) **GPU 업체 변경 (창업자):** Vast.ai는 한국 전화번호 인증이 안 돼서 결제 불가 → **RunPod RTX 4090 커뮤니티 $0.34/h(재고 많음)**. 5090은 RunPod에서 $0.69라 제외, T4는 RunPod에 없음. Phase 1 예상 5–10만원, 첫 충전 $35. 시크릿: `RUNPOD_API_KEY`, `HF_TOKEN`. 러너는 업체 중립(공개 레포에서 작업을 당겨 가고 결과·체크포인트는 HF 비공개 저장소로).
- 다음: 하네스 명세 검토 → 하네스 구현(하위 모델) → 기획서 v3 마무리 → 결제 후 D0.

## ⏸️ (이전) 프로젝트 중단 기록 (2026-10-10)
- 창업자 결정으로 프로젝트를 접었다. 결제(Vast.ai 충전)는 하지 않았고, 쓴 돈은 0원이다.
- 진행 중이던 워크플로(기획서 v3 렌즈·프리모템)는 중단했다. 그때까지 나온 메모는 `work/wf3/`에 있다(렌즈 4편 1차 메모, 일부 프리모템, 리서치 일부, Day 15 예측 패널 전체).
- 매시간 모니터링 트리거 `trig_01BFrwWLxm7wufRA4SEwgYNQ`는 **비활성화**했다(삭제 아님). 재개할 때 다시 켜면 된다.
- 다시 시작할 때: 이 파일의 아래 기록과 `ops/RESUME.md`를 따른다. 기준 문서는 `docs/PARADIGM-v2.md`(v2.1), I2V 전환 결정은 `work/wf3/inputs/07-founder-decision-i2v.md`, 컴퓨트 결론은 "Vast.ai RTX 5090을 필요할 때만, Phase 1 약 3–7만원"이다.

---

## (중단 전 기록)

## 지금 단계
**기획서 v3 (I2V 우선) 제작 중: 1/3단계 (렌즈 + 리서치 + 프리모템).**
- 워크플로: `buc-v3-lenses` (스크립트 `ops/workflows/buc-v3-lenses.js`). 산출물: `work/wf3/lens/{pipeline,models,rights,bench}.memo.md`, `*.upg.md`(프리모템 개정), `research.memo.md`.
- 입력: `work/wf3/inputs/07-founder-decision-i2v.md`(I2V 결정과 분석), `06-founder-decisions-soul-ab.md`.
- 다음: 2단계 집필(수석 아키텍트가 목차·판결록·정본 숫자표 확정 → 절별 집필 → 조립) → 3단계 검증(정합성·누락·방화벽 감사 → 수정) → `docs/PARADIGM-v3.md` 커밋.

## 기획서 다음: 15일 스프린트 (창업자 지시)
- D0 = 기획서 v3 확정일. D15에 같은 사진 한 장으로 Seedance I2V보다 눈으로 좋은 영상.
- 소형 모델 범위를 좁히지 않는다. Phase 1 카탈로그 전원을 만든다.
- **컴퓨트 결정 (2026-10-10):** Vast.ai **RTX 5090**(32GB, BF16, RT 코어)을 필요할 때만 켠다. 1순위는 한국 호스트 1장($0.213/h). 없으면 2×5090($0.83/h) 또는 2×4090($0.72/h). Phase 1 완주 계산 비용은 약 2–4만원(디버깅·재실행·디스크·전송 포함 현실치 3–7만원). 창업자 충전 5만원.
- 학습·개발은 5090. "2×T4로 촬영" 주장은 스프린트 끝에 T4를 몇 시간 빌려 실측한다(제안, 창업자 확인 대기). T4 전용 우회(FP16 수술, INT8/W4A4 강제)는 5090 경로에서 불필요.
- Kaggle은 약관(개인·비상업, 계정 공유 금지) 때문에 주력에서 제외. T4 사냥 워크플로는 중단(1차 결과: ops/viz/t4_hunt.py, gpu_value.py).
- 차트: `ops/viz/sprint_plan.py`.

## 블로커 / 창업자 결정 대기
1. **Vast.ai:** 가입 + 5만원 충전 + API 키를 Network secrets에 `VAST_API_KEY`로. 체크포인트 저장용 Hugging Face 쓰기 토큰을 `HF_TOKEN`으로(비공개 저장소). 채팅에 붙여넣지 않는다. 새 컨테이너에서 읽힌다.
2. **Kaggle 약관 (2026-10-10 원문 확인, kaggle.com/terms):** "You will not share your account or password with anyone"(계정 공유 금지), "You will only use the Services for your own internal, personal, non-commercial use, and not on behalf of or for the benefit of any third party"(개인·비상업 전용). → 친구 계정 빌려 쓰기는 계정 공유, 스타트업 R&D는 비상업 조항과 충돌. Kaggle을 주력 컴퓨트로 쓰지 않는 쪽을 권고. 참고: Colab Pro/Pro+ 연동 시 Kaggle GPU 주 +15/+30시간(실험 프로모션).
3. **(해결)** 컴퓨트: Vast 5090 필요할 때만 가동으로 결정.
4. **사람 데이터:** CamFM-Hand용 IMU(폰 들고 걷기), MRH용 실사 방 촬영, TexelFM용 RAW 매크로.
5. **Seedance 비교 영상:** 창업자의 Higgsfield 구독(Seedance 포함)으로 같은 사진 I2V 생성.

## 모니터링
- 매시간 트리거: `trig_01BFrwWLxm7wufRA4SEwgYNQ` (매시 :24 UTC, 이 세션으로 발화). 없으면 같은 프롬프트로 다시 만든다.

## 완료 기록
- `docs/PARADIGM-v2.md` v2.1 + 사후 검증 23건 반영 (커밋 275894c, c7d9587).
- Higgsfield: 티켓 #317768. Contact Sales 폼은 회사 이메일 필요. I2V 전환으로 데모 이후로 보류.
