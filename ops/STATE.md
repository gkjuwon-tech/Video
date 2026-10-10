# 운영 상태

## 🏭 15일 스프린트 가동 중 (D0 = 2026-10-10)
- **하네스 v1 가동:** 헌법 `ops/harness/README.md`, 백로그 `ops/harness/backlog.json`(167단위, 수직 슬라이스 45개 우선), 선택기 `pick.py`, 기계 게이트 `gate.py`(동결 테스트 해시·compile·ruff·pytest), 상태 갱신 `update.py`, 장부 `ledger.jsonl`. 교대조 워크플로 `ops/workflows/buc-shift.js`(args batches 4, k 2). 교대조가 끝나면 오케스트레이터가 바로 다음 교대조를 띄운다.
- 모델 배정: 선택·게이트·커밋 Haiku, 계약서+테스트 Sonnet(난이도·위험 ≥4는 Opus medium), 구현은 백로그 impl_model, 수리 Sonnet→Opus, 위험 ≥4 검토 Opus low. Fable은 교대조에서 쓰지 않음.
- 보고: 5시간마다(트리거 `trig_01BFrwWLxm7wufRA4SEwgYNQ`, cron 0 */5). 창업자 질문 시 즉시.
- **공장장 세션:** `session_01XA9TirqQtqJiYRyewCK8uv`(Sonnet 5.5, 태그 buc-foreman)이 교대조를 상시 가동한다(지침 `ops/harness/FOREMAN.md`). 이유: 오케스트레이터 세션에서 창업자가 실행을 중단하면 백그라운드 워크플로가 같이 죽는다(2회 발생). 공장장은 GPU·키를 다루지 않는다. 진행 신호는 `ops/harness/foreman.json`.
- **상시 승인 A-001 (2026-10-10T14:51:47Z):** 창업자가 BUC 세션들끼리 세션 생성과 Workflow 실행을 매번 묻지 않고 하도록 승인했다. 원문·해시·서버 이벤트 ID는 `ops/harness/AUTHORIZATION.md`. GPU·결제·키는 여전히 오케스트레이터만.
- **D15 데모 지정 (창업자, 2026-10-10T15:00Z):** 초원의 하얀 거대 야크 이미지 한 장으로 멈춘 세계 + 다양한 카메라 무브. 사양 `docs/DEMO-D15.md`, 원본은 HF 금고 `demo/d15_hero_yak.webp`(공개 레포에 넣지 않음). RDR-01·RDR-08·RD-18·RD-05를 수직 슬라이스로 승격, 관련 단위에 `demo: d15-yak` 태그.
- **공장장 체크인:** 공장장은 send_later 권한이 막혀 있어 오케스트레이터가 매시간 :03에 공장장 세션을 깨우는 트리거 `trig_01GL7Ww48KbFj51yDHhG5EQW`를 걸었다. 창업자가 공장장 세션에서 직접 교대조 상시 가동을 지시했다(14:57Z).
- **밤샘 잡 night1 (창업자 지적: "학습도 영상도 안 하면 뭘 하냐" → GPU 밤새 가동):** 명세 `ops/gpu/night1/SPEC.md`. ① 야크 바닥선 영상 6샷(순진한 깊이 워프, 평가 전용) ② Z-Image 교사 코퍼스(이미지+VAE 잠재, HF `corpus/night1/`) ③ M1 TAE-T 스파이크 학습(HF `ckpt/night1/tae_t/`). 마감 22:45Z(07:45 KST) 뒤 스스로 정지. 코드는 워크플로 night1-gpu-jobs(Sonnet 구현·Opus 검토)가 작성.
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
