# 운영 상태 (2026-10-10 기준, 매시간 갱신)

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
