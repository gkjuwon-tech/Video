# 운영 상태 (2026-10-10 기준, 매시간 갱신)

## 지금 단계
**기획서 v3 (I2V 우선) 제작 중: 1/3단계 (렌즈 + 리서치 + 프리모템).**
- 워크플로: `buc-v3-lenses` (스크립트 `ops/workflows/buc-v3-lenses.js`). 산출물: `work/wf3/lens/{pipeline,models,rights,bench}.memo.md`, `*.upg.md`(프리모템 개정), `research.memo.md`.
- 입력: `work/wf3/inputs/07-founder-decision-i2v.md`(I2V 결정과 분석), `06-founder-decisions-soul-ab.md`.
- 다음: 2단계 집필(수석 아키텍트가 목차·판결록·정본 숫자표 확정 → 절별 집필 → 조립) → 3단계 검증(정합성·누락·방화벽 감사 → 수정) → `docs/PARADIGM-v3.md` 커밋.

## 기획서 다음: 15일 스프린트 (창업자 지시)
- D0 = 기획서 v3 확정일. D15에 같은 사진 한 장으로 Seedance I2V보다 눈으로 좋은 영상.
- 소형 모델 범위를 좁히지 않는다. Phase 1 카탈로그 전원을 만든다.
- 컴퓨트: Kaggle 3계정(주 약 90 세션-시간). T4×2 쿼터 차감률(1배/2배) 미확인 → 15일 193–386 T4-h.
- 차트: `ops/viz/sprint_plan.py`.

## 블로커 / 창업자 결정 대기
1. **Kaggle 자격 증명:** 클라우드 환경 설정의 Network secrets(또는 환경 변수)에 `KAGGLE_USERNAME_1..3`, `KAGGLE_KEY_1..3`. 채팅에 붙여넣지 않는다. 새 세션(또는 새 컨테이너)에서 읽힌다.
2. **Kaggle 약관:** 계정 3개가 서로 다른 실제 사람 3명의 계정인지 확인(1인 다계정은 약관 위반으로 정지 위험).
3. **컴퓨트 부족분:** Kaggle만으로는 Phase 1 스프린트 필요량(500–775 T4-h)의 약 25–75%. 상시 2×T4 임대(15일 720 T4-h) 여부.
4. **사람 데이터:** CamFM-Hand용 IMU(폰 들고 걷기), MRH용 실사 방 촬영, TexelFM용 RAW 매크로.
5. **Seedance 비교 영상:** 창업자의 Higgsfield 구독(Seedance 포함)으로 같은 사진 I2V 생성.

## 모니터링
- 매시간 트리거: `trig_01BFrwWLxm7wufRA4SEwgYNQ` (매시 :24 UTC, 이 세션으로 발화). 없으면 같은 프롬프트로 다시 만든다.

## 완료 기록
- `docs/PARADIGM-v2.md` v2.1 + 사후 검증 23건 반영 (커밋 275894c, c7d9587).
- Higgsfield: 티켓 #317768. Contact Sales 폼은 회사 이메일 필요. I2V 전환으로 데모 이후로 보류.
