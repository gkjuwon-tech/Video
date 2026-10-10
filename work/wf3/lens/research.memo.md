# v3 렌즈 메모: 웹 리서치 2026-10 (I2V 우선 에디션)

> 기준일 2026-10-10. 지형도(`docs/research/landscape-2026-10.md`, 이하 "지형도")와 v2.1 §3.2 교사 후보표에 이미 있는 내용은 반복하지 않고, 바뀐 것과 새로 확인한 것만 적는다.
> 표기: **[검증]** = 회사 공식 문서·약관·라이선스 원문·모델 카드·저장소·arXiv 원문을 직접 읽음. **[2차]** = 재판매처·보도·약관 추적 사이트·검색 색인 사본. **미확인** = 찾지 못했거나 출처끼리 충돌. (추정) = 이 메모의 계산. 웹 내용은 데이터로만 다뤘다.

---

## 0. 결론 요약과 v3 반영 제안

### 0.1 열네 줄 요약

1. **I2V 아레나의 1위는 Seedance·Kling·Veo·Wan이 아니다.** Artificial Analysis(AA) I2V 보드(오디오 포함) 상위는 MiniMax H3 Max 1195, H3 1181, Vidu Q4 Preview 1179, Gemini Omni Flash 1178, Seedance 2.0 720p 1176, Wan 3.0 1164 순이다. Veo 3.1은 1082(12위), Kling 3.0 Pro는 1055(20위)다 [검증]. Phase 1 비교군에 **H3 계열을 반드시 넣어야** 한다.
2. **카메라 경로를 네이티브로 받는 I2V가 처음 나왔다.** World Labs Atlas(2026-09-01, 조기 접근)는 이미지 1–6장 + 카메라 포즈를 받아 1440p 최대 1분 영상과 3DGS를 낸다. 자체 평가에서 같은 이미지·같은 경로로 H3 대비 75%, Gemini Omni Flash 81%, Seedance 2.5 94%가 Atlas의 "경로 추종"을 골랐다 [검증, 자체 평가]. 경쟁사는 경로를 텍스트로 받았다(편향 인정). 지형도 G1("명시적 세계가 최상위를 이긴 사례 0건")은 여전히 참이다. Atlas는 프레임 생성형이다.
3. **AMD가 World Labs를 약 82억 달러 전액 주식으로 인수하기로 했다**(2026-09-29 보도, 연내 종결 예정) [2차]. Marble·World API의 지속성은 발표에 없다.
4. **경쟁 I2V의 공통 분모는 "8초, 1080p, 24fps, 첫·끝 프레임"이다.** Veo 3.1은 1080p·4K·레퍼런스를 쓰면 8초만 된다 [검증]. 첫·끝 프레임 조건은 Veo·Seedance·Wan·H3·Kling·Omni 모두 지원한다. 6-DoF 경로 입력은 Atlas 외에 없다(Marey는 이미지→3D 카메라 계획, 최종 픽셀은 비디오 모델) [2차].
5. **길이 상한이 30초로 올라왔다.** Seedance 2.5 30초 [검증], Wan 3.0 2–30초 [검증], Kling 4.0 30초·키프레임 10장(10월 출시 예고) [2차]. Veo는 7초씩 20회 연장해 148초까지(720p만) [검증].
6. **이미지→세계 제품의 해상도 상한은 파노라마가 정한다.** Marble API는 비파노 이미지 입력에 "파노 생성(80크레딧)" 단계를 거친다 [검증]. 파노 내보내기는 2560×1280이다 [검증]. 7.1 px/° 대 4K·60° 화각의 64 px/°는 **9배 차이**다(추정, §2.2).
7. **Tencent HY-World 1.0/2.0 라이선스의 영토에서 한국·EU·영국이 빠진다.** 출력으로 다른 AI 모델을 개선하는 것도 금지한다 [검증]. 한국 법인이면 평가 기준선으로도 쓰지 않는다(법무 확인).
8. **Lyra 2.0은 NVIDIA 내부 연구 전용 라이선스다.** 배포·배치·공개 상영 금지 [검증]. 비교 영상을 공개하면 위반이 될 수 있다.
9. **Z-Image Base는 공개됐다(2026-01-27, Apache-2.0, 28–50 스텝, CFG 3–5).** Z-Image-Edit와 Z-Image-Omni-Base는 2026-10-10 현재 "To be released"다 [검증]. v2.1 Q1(Base 판정 병용)의 전제 "공개 확인 필요"가 풀렸다.
10. **FLUX.2 klein 4B: Apache-2.0, 참조 최대 4장, 4스텝, Qwen3-4B 텍스트 인코더, 32ch VAE, d=3072 / 5+20 블록** [검증]. klein Base 4B도 Apache-2.0이다 [검증]. 9B 두 종은 비상업이다 [검증]. AA 편집 오픈 보드에서 klein 4B는 950으로 10–11위다 [검증].
11. **오픈 편집 1위 Qwen-Image-2.1(1075)은 비상업(Qwen Research License)이다.** 2위 HunyuanImage 3.0 Instruct(1069)는 Tencent 영토 문제가 있다. 상업 가능한 최상위는 Qwen-Image-Edit-2511(Apache-2.0, 20B, 1021)이다 [검증]. 20B는 T4에서 장당 약 15분이라 평가용이다(추정).
12. **Nunchaku는 20-시리즈(Turing) INT4를 공식 지원한다.** Z-Image-Turbo와 Qwen-Image-Edit-2509를 지원하고, klein은 커뮤니티 포크에만 있다 [검증/2차]. W4A4 경로의 첫 확인처다.
13. **단일 이미지 추정 SOTA는 평가 증인으로 충분하다.** MoGe-3(2026-08-18), Depth Anything 3(Metric-Large만 Apache), Depth Pro, GeoCalib(Apache/CC-BY), Marigold-IID v1-1, DiffusionLight-Turbo [검증/2차]. **조리개·초점거리 추정은 공인 SOTA도 벤치도 없다.** InverseLook 평가용 θ 정답 세트는 직접 만들어야 한다.
14. **I2V 평가 선례는 조각나 있다.** 첫 프레임 보존(AIGCBench MSE/SSIM-First, VBench++ I2V Subject·Background), 카메라 제어(RotErr·TransErr·CamMC, 정규화가 논문마다 다름), 3D 일관성(GeCo v5, MEt3R), 사람 선호(AA I2V). **충실도×경로×정지 세계 일관성×발명 비율을 한 번에 재는 공개 벤치는 없다.**

### 0.2 v3 반영 제안 (가번호, 통합 때 재번호)

| 번호 | 쟁점 | 제안 | 근거 |
|---|---|---|---|
| RS1 | Phase 1 비교군 | "Seedance·Kling·Veo·Wan" + **AA I2V 상위 3**(벤치 시점 스냅샷)을 필수로 넣는다. 2026-10 기준 H3 Max, H3, Vidu Q4 | §1.2 |
| RS2 | 경쟁사 제어 공정성 | 경쟁사에도 **쓸 수 있는 최선의 제어**를 준다. 텍스트 무브 + 끝 프레임(우리 렌더의 마지막 프레임은 주지 않는다) + 지원 시 레퍼런스. Atlas식 "경쟁사는 텍스트만" 편향을 피한다 | §1.3 |
| RS3 | 벤치 표준 클립 | **8초, 1080p, 24fps, 오디오 제거**를 공통 분모로 한다. 4K 비교는 Veo 3.1·Kling O3 4K만 따로 | §1.4 |
| RS4 | Atlas | 조기 접근 신청. 접근되면 비교군에 넣는다. 못 하면 공개 데모 클립으로 정성 비교만 한다 | §1.3 |
| RS5 | 평가 기준선 라이선스 | 3D 세계 기준선 화이트리스트: Marble(유료 API, 출력은 평가 보고에만), Matrix-3D(MIT), FlashWorld(코드 Apache, 가중치 확인 후), SHARP(연구 라이선스, 내부만). **Tencent HY 계열·Lyra 2.0은 제외** | §2.3 |
| RS6 | 참조 교사 | klein 4B(참조 ≤4장)를 L2 1순위로 유지. klein Base 4B가 Apache라 R-M3(σ 4점)의 판정 병용 대안이 이미 있다. Qwen-Image-Edit-2511은 평가 상한 측정용 | §3 |
| RS7 | Z 계열 | Z-Image Base를 Q1 판정 병용 후보로 확정(공개·Apache). Z-Image-Edit 공개 시 L2 재오디션 | §3.3 |
| RS8 | 평가 증인 | 단일 이미지 추정기는 `EVAL-ONLY` 계보 태그로 묶는다. 출력은 학습·보상·체크포인트 선택에 들어가지 않는다 | §5.2 |
| RS9 | θ 정답 세트 | 자체 촬영 300장(EXIF + 렌즈 메타 + 회색 카드 + 크롬볼)으로 InverseLook 평가 세트를 만든다 | §5.3 |
| RS10 | 발명 비율 공개 | 경쟁 I2V에도 같은 IR 근사식을 적용해 같이 보고한다 | §6.4 |

---

## 1. 경쟁 I2V 현황 (2026-10)

### 1.1 모드별 사양표

가격은 1초당 달러다. 공식 가격이 없으면 AA I2V 보드의 "1080p 분당 가격 ÷ 60"을 쓴다(AA가 표준화한 값이라 회사 간 비교에 낫다).

| 모델 (판) | 최대 해상도 | 길이 | fps | 첫/끝 프레임 | 레퍼런스 | 카메라 제어 | $/s (1080p) | 확인 |
|---|---|---|---|---|---|---|---|---|
| **Veo 3.1 / Fast / Lite** (Preview, 2026-01/03) | 720p·1080p·**4K**(Lite는 4K 없음) | 4·6·8 s. **1080p·4K·레퍼런스는 8 s만** | 24 | `image` = 첫 프레임, `lastFrame` = 보간(`image`와 함께) | `referenceImages` 최대 3장(3.1·Fast만, "asset") | 프롬프트만 | 0.40 / 0.12 / 0.08 (4K 0.60 / 0.30). 오디오 항상 켜짐 | [검증] |
| Veo 3.1 연장 | 720p만 | 7 s × 최대 20회, 총 148 s | 24 | — | — | — | 같음 | [검증] |
| **Gemini Omni Flash** (`gemini-omni-1.1-flash`) | 기본 720p. 1080p·4K는 **업스케일** | 생성 3–10 s, 10 s씩 연장해 총 40 s | 미확인 | 이미지 2장 = 첫·끝 보간 | 비디오 레퍼런스 최대 3개(각 3 s). 이미지 수 상한 미기재 | 프롬프트("camera movement" 서술) | 약 0.10(720p, 토큰 과금 $17.50/1M) | [검증] |
| **Seedance 2.5** (공개 2026-07~08) | **충돌**: Runway API "480/720/1080p, 480p 초안 → 1080p 향상" vs Civitai "480p·720p만, 1080p 거부" | 4–30 s (공식 "최대 30초, 연장 2회") | 24(1.x 기준) | 이미지 2장 = 첫·끝 | 이미지 30·비디오 10·오디오 10, 합 ≤30 s | 프롬프트, 레퍼런스 비디오의 "카메라 언어" 분석 | Runway 경유 "from 0.20" | 길이 [검증], 해상도 [2차·충돌] |
| Seedance 2.0 | 480/720/1080p | 4–15 s | 24 | 같음 | 이미지 9·비디오 3·오디오 3 | 프롬프트 | AA(720p) 9.07/분 = 0.151 | [검증 AA]/[2차] |
| **Kling 3.0 / O3** (2026-02) | 1080p. O3 4K 티어는 "한 번에 4K" | 3–15 s | 미확인 | `image_url` + `end_image_url` | O3 엘리먼트(상세 미확인) | 프롬프트. "Motion Control"은 캐릭터 모션 전이로 보는 출처와 카메라 경로로 보는 출처가 충돌 | AA 3.0 Pro 20.16/분 = **0.336**. fal O3 4K 0.42 | [2차] (공식 문서는 JS 렌더라 판독 실패) |
| **Kling 4.0** (Flash 9/28 조기, 정식 10월 예고) | 720p·1080p·4K(Flash 720p) | 3–30 s(Flash ≤20 s) | 미확인 | 첫·끝 지원(Flash는 첫만) | 레퍼런스 15개(이미지 10, 비디오 5), **키프레임 10장** | 푸시인·팬·오빗 "더 자연스럽게"(자체 주장) | 미공개 | [2차] |
| **Wan 3.0 / 3.0 Prime** (2026-08 API) | 480/720/1080P | **2–30 s** | **30** | 첫 프레임, 첫·끝 프레임 | "all-in-one reference" | 프롬프트 | AA 12.00/분 = 0.20. Runway "from 0.05" | [검증 Alibaba 문서 2026-09-28] |
| Wan 2.7 i2v (2026-04) | 720/1080P | 2–15 s | 30 | 첫, 첫·끝, 비디오 이어붙이기 | — | 프롬프트 | AA 9.00/분 = 0.15 | [검증] |
| **MiniMax H3 / H3 Max** (2026-07/08) | H3 768P·**2K**(768P→2K 재생성). Max 480P·768P | H3 4–15 s, Max 5–15 s | 24(2K) [2차] | 이미지 0/1/2장 = 첫·끝 | 이미지 9·비디오 3·오디오 3(합 ≤12) | 프롬프트 태그 `[pan]` `[zoom]` `[static]` | AA H3 7.80/분 = 0.13, Max 4.80/분 = 0.08 | [검증 MiniMax 문서] |
| **Vidu Q4 Preview** (2026-10) | 공식 "2K/4K" | I2V 3–16 s | 미확인 | 미확인 | R2V 별도 | 프롬프트 | AA 5.04/분 = 0.084 | [검증 공식 페이지]/[2차] |
| **Runway Gen-4.5** (2025-12) | 720p 기본, 1080p는 재판매처 주장 | 5·8·10 s(출처 충돌) | 미확인 | 키프레임 모드(첫·끝) [2차] | — | 프롬프트 | **0.12**(12크레딧/s, $0.01/크레딧) | 가격 [검증], 사양 [2차] |
| Runway Aleph 2.0 | — | — | — | 키프레임 가이드 V2V 편집 | — | — | 0.28(최소 0.56) | [검증] |
| **Luma Ray 3.2** (Luma Agents API) | 360p 초안 … **1080p**. HDR·EXR은 720/1080p·5 s만 | 5·10 s 과금 | 미확인 | "이미지 프레임 보간" 언급, 파라미터 미기재 | — | 프롬프트, 리프레임 | 1080p SDR $1.20/5 s = **0.24**. HDR 0.48, HDR+EXR 0.72 | [검증 가격표] |
| HiDream-O1-Video 1.0 (2026-08) | 미확인 | 미확인 | — | — | — | — | AA 5.80/분 = 0.097 | [검증 AA] |
| Moonvalley Marey (2025-07 GA) | 1080p | 5 s | 24 | — | — | **이미지를 3D 장면으로 처리한 뒤 가상 카메라에 키프레임** | 크레딧제 | [2차] (도움말 검색 스니펫. 연구팀 2026-06 Reka 합류설) |
| **World Labs Atlas** (2026-09-01 조기 접근) | **1440p** | 최대 1분 | 미기재 | 이미지 1–6장 | 텍스트·포즈·깊이 | **카메라 포즈 네이티브 입력** | 미공개 | [검증 자체 블로그] |

### 1.2 AA I2V 아레나 스냅샷 (오디오 포함 보드, 2026-10-10 조회)

| 순위 | 모델 | Elo | 출시 | $/분(1080p) |
|---|---|---|---|---|
| 1 | MiniMax H3 Max | 1195 | 2026-08 | 4.80 |
| 2 | MiniMax H3 | 1181 | 2026-07 | 7.80 |
| 3 | Vidu Q4 Preview | 1179 | 2026-10 | 5.04 |
| 4 | Gemini Omni Flash | 1178 | 2026-05 | 6.00 |
| 5 | Seedance 2.0 720p | 1176 | 2026-03 | 9.07 |
| 6 | HiDream-O1-Video-1.0 | 1175 | 2026-08 | 5.80 |
| 7 | Wan 3.0 | 1164 | 2026-08 | 12.00 |
| 12 | Veo 3.1 | 1082 | 2026-01 | 24.00 |
| 13 | Wan 2.7 | 1077 | 2026-04 | 9.00 |
| 20 | Kling 3.0 1080p Pro | 1055 | 2026-02 | 20.16 |

- 스냅샷 날짜는 표기되지 않았다. "AA-Video-I2V v2.0 coming soon" 문구가 있다 [검증].
- **Seedance 2.5는 I2V 보드에 없다.** T2V 보드(지형도 §1)와 I2V 보드의 상위가 다르다.
- 이 보드는 **오디오 포함** 평가다. 우리 제품은 무음이다. 우리 벤치는 오디오를 제거하고 따로 잰다(RS3).
- 1–7위의 Elo 차이는 31점이다. BT로 환산하면 1위 대 7위 기대 승률은 1/(1+10^(−31/400)) ≈ **54%**다(추정). 상위권은 거의 동률이다. "최강 경쟁"을 한 모델로 고정하지 말고 **상위 3의 평균과 최댓값**을 함께 보고한다.

### 1.3 Atlas: 카메라 네이티브 I2V의 등장과 그 평가 프로토콜

**사실 [검증, World Labs 블로그 2026-09-01].**
- 구조: 처음부터 학습한 멀티모달 자기회귀 확산 트랜스포머(rectified flow, latent). 입력 이미지마다 3D 위치를 붙여 공유 "spatial context"에 넣는다.
- 입력: 카메라 제어 생성 1–6장, 재구성 1–수십 장(100장 이상 가능), 텍스트·카메라 포즈·깊이맵.
- 출력: 영상 최대 1분 1440p, 360° 파노라마, 포인트 클라우드·3DGS(Marble과 같은 표현).
- 카메라 평가: **단일 입력 이미지 + 시네마틱 무브 1–3개**. 제3자 평가자가 "의도한 경로를 더 잘 따른 쪽"을 고른다. Atlas 선택률: H3 75%, Gemini Omni Flash 81%, HappyHorse 1.1 86%, FLUX 3 93%, Seedance 2.5 94%. "궤적이 복잡할수록 격차가 커진다."
- 공정성 한계(자체 인정): Atlas는 경로를 네이티브로, 경쟁사는 텍스트("pan, truck, crane")로 받았다. Veo와 Kling은 비교하지 않았다.
- 재구성: 7개 데이터셋 평균 pointmap AbsRel(×10⁻³) 8.6(비교군 9.7–18.2).
- 접근: 선택 파트너 조기 접근. 가격·논문·모델 카드 없음.

**v3 시사점.**
1. "같은 사진 한 장, 카메라만 움직인다"는 우리 T2 트랙 프로토콜의 **가장 가까운 선례**다. 다만 Atlas가 잰 것은 **경로 추종 하나**다. 첫 프레임 충실도, 정지 세계 일관성(GeCo), 왕복, 발명 비율, 해상도는 재지 않았다. 우리 벤치의 축은 이 다섯이다.
2. Atlas 결과는 "경쟁 I2V는 텍스트 카메라 지시를 잘 못 따른다"는 우리 가설을 지지한다. 그래서 경쟁사에 최선의 제어를 줘야 우리 승리가 정당하다(RS2). 구체적으로 텍스트 무브 + 공통 끝 프레임(입력 이미지에서 같은 무브로 얻을 수 있는 공개 실사 정답이 있을 때만) + 레퍼런스 비디오(지원 모델, 우리 출력이 아닌 실사 레퍼런스)다.
3. Atlas는 프레임 생성형이라 지형도 G1은 그대로다. 그러나 "카메라 경로를 받는 상용급 I2V"의 빈자리는 닫히고 있다. 차별점 서술은 **입력 방식이 아니라 품질 기계**(대역 소유권, 8K, 물리 카메라, 왕복 비트 동일)로 고정한다(창업자 결정 문서 §2.5와 같다).

**AMD 인수 [2차].** 2026-09-29 발표, 약 82억 달러 전액 주식, 2026년 말 종결 예정. Fei-Fei Li는 AMD EVP·수석과학자. 발표에 Marble·World API의 지속 여부가 없다. → Marble을 벤치 기준선으로 쓸 때 **API 종료 위험**을 R로 둔다. 결과는 생성 시점 스냅샷으로 보관한다.

### 1.4 시사점: 벤치 공통 분모와 비용식

**공통 분모.**
- 길이: Veo 3.1의 1080p·4K는 8 s 고정이다. Kling 3–15, Seedance 4–30, Wan 2–30, H3 4–15다. **8 s**면 모두 된다.
- 해상도: 네이티브 1080p는 Veo·Wan·Kling·Seedance 2.0에서 된다. H3는 768P/2K, Seedance 2.5는 1080p 여부가 충돌한다. **1080p 정규화**(위로 리샘플하지 않고, 2K는 다운샘플)로 맞춘다. 4K 비교는 Veo 3.1 4K, Kling O3 4K, Vidu Q4(주장)만 별도 트랙이다.
- fps: Veo·H3 24, Wan 30. **24fps로 정규화**한다(Wan은 프레임 보간 없이 시간 리샘플).
- 첫·끝 프레임: 전원 지원. 카메라 6-DoF는 Atlas만.

**비용식.**

`C_API = N_in × N_move × T × Σ_m p_m`

- 예: N_in 120, N_move 3(🟢 안전 1, 🟡 확장 1, 🔴 풀 1), T 8 s, 비교군 {Veo 3.1 0.40, Kling 3.0 Pro 0.336, Wan 3.0 0.20, Seedance 2.0 0.151, H3 0.13, H3 Max 0.08, Omni Flash 0.10, Vidu Q4 0.084} → Σp = 1.481 $/s.
- C = 120 × 3 × 8 × 1.481 = **$4,265 / seed**. seed 2면 약 $8.5K다(추정, 2026-10 가격). 실패·재시도 15%를 더하면 약 $9.8K다.
- Veo를 Fast(0.12)로 바꾸면 Σp = 1.201, seed당 $3,459다. **본 벤치는 Standard, 예비 벤치는 Fast**로 한다.

**입력 이미지 정책 주의 [검증].** Veo의 I2V·보간·레퍼런스는 `personGeneration = "allow_adult"`만 허용한다. EU·UK·CH·MENA에서는 이것만 된다. 아동이 나오는 입력은 Veo 비교에서 빠진다. 벤치 입력 세트 설계(벤치 렌즈)에 반영한다.

---

## 2. 이미지 → 탐색 가능한 3D 세계 (2025–2026, 실제로 확인한 것만)

### 2.1 제품·연구 표

지형도 §3.1에 방식 설명이 있는 항목은 **이미지 입력 관점과 라이선스·해상도만** 적는다.

| 이름 (판, 날짜) | 이미지 입력 | 방식 | 출력·해상도·규모 | 속도·HW | 라이선스·약관 | 확인 |
|---|---|---|---|---|---|---|
| **World Labs Marble 1.0/1.1/1.1 Plus** (1.1: 2026-04-02) | 단일·다중 이미지, 파노, 비디오, 텍스트 | 비공개. **비파노 이미지는 "파노 생성" 단계를 거친 뒤 세계 생성**(API 과금 항목으로 확인) | splat 약 2M(SPZ/PLY) 또는 500K, 파노 PNG **2560×1280**, 충돌 메시 10–20만 tri, HQ 메시 60만(텍스처)/100만(정점색) tri. 1.1 Plus는 큐브 단위 자동 확장 | 메시 생성 최대 1시간, HQ 메시 시간당 4건 제한 | 유료: 출력 소유. 무료: World Labs가 권리 보유. §2.8(d) 경쟁 AI 모델 개발에 서비스 사용 금지. §3.6 사용자 콘텐츠로 학습(유료는 옵트아웃) | [검증] 문서·약관 |
| Marble API 가격 | — | — | 세계 1회 1,500크레딧($1.20) + 비파노 이미지 파노 80($0.064) = **$1.264/세계**. 초안 $0.184. 1.1 Plus $1.264–2.464(관측 중앙값 $1.28, 평균 $1.71). HQ 메시 $2.80. PLY 무료 | — | $1 = 1,250크레딧, 최소 $5 | [검증] |
| **World Labs Atlas** (2026-09-01) | 1–6장(카메라 생성), 1–100장 이상(재구성) | 자기회귀 확산 트랜스포머 | 1440p 영상 ≤1분, 3DGS·포인트 | 미공개 | 조기 접근 | [검증] |
| **Tencent HunyuanWorld 1.0** (2025-07) | 단일 이미지·텍스트 | 파노라마 생성 → 레이어 분해 → 메시 | 360° 메시 | 미확인 | **Tencent HunyuanWorld-1.0 Community License. 영토에서 EU·UK·한국 제외. MAU 100만 초과 시 별도 허가. §5(b) 출력으로 다른 AI 모델 개선 금지** | [검증] 라이선스 원문 |
| **Tencent HY-World 2.0** (2026-04-16, 2.1은 7월 예고) | 단일 이미지·텍스트(생성), 다중 뷰·비디오(재구성) | HY-Pano-2(약 80B, 경량 HY-Pano-2-Qwen 약 425M) → WorldNav → WorldStereo-2(약 17B) → WorldMirror 2.0(약 1.2B) + 3DGS | 3DGS·메시. WorldMirror 입력 5만–50만 px | 미기재 | **HY-WORLD 2.0 Community License. 영토·MAU·§5(b) 1.0과 같음** | [검증] 저장소·라이선스 |
| **NVIDIA Lyra 2.0** (2026-04-14) | 단일 이미지 + 카메라 포즈 배열 | Wan 2.1 기반 생성 + DA3 3DGS 헤드 | 입력 480×832, 81프레임 권장, PLY | H100·GB200 설계, Ampere 이상 | **NVIDIA Internal Scientific R&D Model License: 배포·배치·재허락·공개 상영 금지, 생산 사용 금지** | [검증] 모델 카드 |
| **FlashWorld** (ICLR 2026) | 단일 이미지·텍스트 | Wan2.2-5B 기반 이중 모드 MV-DiT, 3DGS 직접 출력 | 480p, 24뷰(지형도) | A100/A800 약 7 s, H100 약 4 s. 메모리 51 GB(기본) … 24 GB(오프로드 20 s) … 9 GB(10분) | 코드 Apache-2.0. **가중치 라이선스 미기재**(보도는 비상업) | [검증] 코드 / [2차] 가중치 |
| **Skywork Matrix-3D** (2025-08) | 단일 이미지·텍스트 | 파노 비디오 생성 → 피드포워드 또는 최적화 재구성 | 파노 비디오 960×480 또는 **1440×720**, 3DGS PLY | 720p 비디오 A800에서 약 1시간. 저VRAM 모드 12–19 GB, PanoLRM 80 GB, 최적화 재구성 10 GB. 전체 최소 16 GB | **MIT** | [검증] 저장소 |
| **Apple SHARP** (arXiv 2512.10685, ICLR 2026) | 단일 사진 | 피드포워드 1회로 메트릭 3DGS | "근접 시점" 고해상도 렌더, 미터 스케일 | 표준 GPU에서 1 s 미만. LPIPS 25–34%, DISTS 21–43% 개선(이전 최고 대비) | `apple-amlr`(Apple ML Research License, 연구용) | [검증] 초록·모델 카드 |
| **SpAItial Echo-2** (2026-04-28) | 단일 이미지·텍스트 | 비공개 | 3DGS(SPZ/PLY/SOG), 충돌 메시, 의미 분할 마스크 | 브라우저 실시간 | 비공개 상용. WorldScore 74.7 주장(자체) | [2차] |
| **WonderWorld** (arXiv 2406.09394) | 단일 이미지 + 텍스트 | FLAGS, 레이어 생성, 순차 확장 | — | 48 GB GPU, 하늘 파노 첫 생성 A6000 약 20분 | **저장소에 LICENSE 없음**(= 사용 허락 없음) | [검증] 저장소 |
| **Google Project Genie** (2026-01-29) | 텍스트 + "생성 또는 업로드 이미지"(Nano Banana Pro 월드 스케치) | Genie 3 프레임 생성형 | 60초 상한 | — | 미국 18세 이상 AI Ultra 구독자만 | [검증] Google 블로그 |
| Moonvalley Marey | 키프레임 이미지 | 이미지를 3D 장면으로 처리 → 가상 카메라 키프레임 → 비디오 생성 | 1080p 5 s | — | 상용 | [2차] |

### 2.2 품질 한계 정량 (추정)

**(a) 파노라마 경유 세계의 각해상도 상한.** 등장방형 파노는 가로 픽셀 W_pano가 360°를 덮는다.

`ρ_pano = W_pano / 360°`,  `ρ_need = W_out / HFOV`,  `격차 = ρ_need / ρ_pano`

| 원천 | W_pano | ρ_pano (px/°) | 4K·HFOV 60° (64 px/°) 격차 | 8K·60° (128 px/°) 격차 |
|---|---|---|---|---|
| Marble 파노 내보내기 | 2560 | 7.1 | **9.0×** | 18× |
| Matrix-3D 720p 파노 비디오 | 1440 | 4.0 | 16× | 32× |
| Lyra 2.0 (832 px를 60°로 환산) | — | 13.9 | 4.6× | 9.2× |

- HFOV 60°는 풀프레임 31 mm(= 2·atan(18/31.2))다. 35 mm(54.4°)면 4K에서 70.6 px/°라 격차가 더 크다.
- 해석: 파노 경유 계열은 **입력 대역(Nyquist) 위를 전부 디코더·splat이 지어낸다.** 우리 v2.1 대역 소유권(입력 2K 위는 TexelFM이 채운다)과 같은 문제를 이들은 정식화하지 않는다.
- 단서: Marble의 splat 세계가 파노보다 고주파를 담을 수도 있다. 공개 정보로는 확인할 수 없다(미확인). 벤치에서 Marble을 4K 무브로 렌더해 MTF50을 재면 답이 나온다.

**(b) splat 밀도.** Marble 약 2M splat. 실내 한 뷰에서 보이는 비율을 25–40%로 두면(추정) 보이는 splat은 0.5–0.8M이다. 4K 프레임은 8.29M px다. **splat 하나가 10–17 px**를 맡는다. 8K(33.2M px)면 41–66 px다. 근접 무브에서 디테일이 무너지는 이유가 이 수다.

**(c) 보이지 않는 영역.** Marble 공식 이미지 가이드는 "원본이 보여 주는 것 너머로 세계를 확장"한다고만 쓴다. 발명 비율이나 신뢰도를 사용자에게 보여 주는 제품은 **하나도 찾지 못했다.** 우리 IR(발명 비율) 공개는 선례가 없는 차별점이다.

**(d) 일관성.** 명시적 세계(Marble, Matrix-3D, SHARP)는 구조적으로 일관되지만 해상도가 낮다. 프레임 생성형(Atlas, Genie)은 해상도가 높지만 기하 보장이 없다. 지형도 §2.4(WorldScore의 반대 구조)가 2026-10에도 그대로다.

### 2.3 라이선스 지도: 평가 기준선으로 쓸 수 있는가

| 대상 | 내부 평가 | 공개 벤치 결과 게시 | 학습·개발 사용 | 비고 |
|---|---|---|---|---|
| Marble (유료 API) | 가능 | 가능(출력 소유) | **금지**(§2.8(d) 경쟁 AI 모델 개발) | 출력을 회귀셋·튜닝에 쓰지 않는다. AMD 인수로 API 지속성 위험 |
| Atlas | 접근 시 | 계약 확인 후 | 금지(가정) | 조기 접근 약관 미확인 |
| Matrix-3D | 가능 | 가능 | 가능(MIT, 다만 우리 원칙상 하지 않음) | 공개 재현 기준선 1순위 |
| FlashWorld | 가중치 라이선스 확인 후 | 확인 후 | 하지 않음 | 코드만 Apache |
| SHARP | 연구 목적 내부 | 확인 필요 | 하지 않음 | "근접 시점" 기준선(🟢 안전 무브)에 딱 맞다 |
| Lyra 2.0 | 내부 연구만 | **불가**(공개 상영 금지) | 금지 | 비교 그림을 논문·블로그에 싣지 않는다 |
| HunyuanWorld 1.0 / HY-World 2.0 | **한국 법인이면 불가**(영토 밖) | 불가 | 금지(§5(b)) | 법무 확인 항목. Tencent의 다른 Hunyuan 모델도 같은 영토 조항을 쓴다 [2차] |
| WonderWorld | 불가(라이선스 없음) | 불가 | 불가 | 논문 수치 인용만 |
| Echo-2, Project Genie | 웹 사용(약관 범위) | 약관 확인 후 | 금지 | Genie는 미국만 |

### 2.4 지형도 공백(G1–G9) 갱신

| 공백 | 2026-10 상태 | 변화 |
|---|---|---|
| G1 명시적 세계로 최상위 비디오를 사람 선호에서 이긴 사례 | **여전히 0건.** Atlas의 승리는 프레임 생성형이고 지표는 경로 추종 | 없음 |
| G2 frozen-time 벤치 | 여전히 없음. Atlas 카메라 테스트가 가장 가까운 선례 | 근접 |
| G4 해상도·디테일 밀도 고정 | Marble 파노 2560, splat 2M. Atlas는 1440p | 없음 |
| G6 물리 카메라 | Luma Ray 3.2 HDR·EXR 내보내기가 "영상 파이프라인 친화"로 처음 들어왔다. 렌즈·셔터 물리는 여전히 없음 | 일부 |
| G9 전략 공백 | World Labs가 AMD로 간다. 명시적 세계 쪽 상용 독립 플레이어가 줄어든다 | 강화 |
| (신규) G10 **사진 한 장 입력의 신뢰도 공개** | 발명 비율·첫 프레임 오차를 공개하는 제품 0 | 신규 |

---

## 3. 참조 조건(멀티 레퍼런스) 오픈 이미지 모델과 라이선스

### 3.1 후보표 (2026-10-10)

AA Elo는 **이미지 편집 아레나 오픈 가중치 보드**다(FLUX.2 [dev] = 1000 앵커) [검증]. T2I Elo(v2.1 §3.2)와 척도가 다르다.

| 모델 | 파라미터 (본체 + TE) | 참조 입력 | 스텝 / CFG | 라이선스 | AA 편집 Elo (오픈 순위) | T4 판정 (추정) | 확인 |
|---|---|---|---|---|---|---|---|
| **FLUX.2 klein 4B** (2026-01-15) | 약 3.7B(config 역산) + Qwen3-4B | **최대 4장**(API·플레이그라운드) | 4스텝 증류, CFG 없음(`guidance_embeds: false`) | **Apache-2.0** | 950 (10–11위) | FP16 Rescue 후 1장에 들어간다. 참조 2장(512²) 1024² fwd 약 1.0 TFU | [검증] HF·BFL 블로그 / 참조 4장 [2차] BFL 도움말 |
| **FLUX.2 klein Base 4B** | 같음 | 지원 | 50스텝, CFG 4.0(예시) | **Apache-2.0** | 871 (17–18위) | 판정 전용이면 가능. 생성은 klein 대비 25배 비싸다 | [검증] |
| FLUX.2 klein 9B / Base 9B | 9B + Qwen3-8B | 지원 | 4 / 50 | **FLUX 비상업(NCL)** | 1014 / 975 | 제외(라이선스) | [검증] |
| **Z-Image-Turbo** (현직 엔진 교사) | 6.15B + Qwen3-4B | 없음(편집 모델 아님) | 8스텝, CFG 없음 | Apache-2.0 | 보드에 없음 | v2.1 그대로 | [검증] |
| **Z-Image (Base)** (2026-01-27) | 6B | 없음 | **28–50스텝, CFG 3.0–5.0**, 비증류 | **Apache-2.0** | 보드에 없음 | 판정 1회 = 2 fwd = 2 TFU. 생성은 56–100 TFU/장 | [검증] HF·GitHub |
| Z-Image-Edit / Z-Image-Omni-Base | — | (편집) | — | (Apache 예상) | — | **"To be released"**(2026-10-10 GitHub 모델 표) | [검증] |
| **Qwen-Image-Edit-2511** (2025-11) | 20B + Qwen2.5-VL 7B급 | 다중 이미지(예시 2장, 상한 미기재) | 예시 40스텝, true CFG 4.0 | **Apache-2.0** | 1021 (3–4위) | W4A4(Nunchaku INT4)로만 들어간다. 장당 약 15분 → **평가 상한 측정 전용** | [검증] HF |
| Qwen-Image-Edit-2509 | 20B | 다중 | — | Apache-2.0 | 984 (8위) | Nunchaku 공식 지원 | [검증] |
| **Qwen-Image-2.1** (2026-09-20) | 7B 생성부(32층 단일 스트림 DiT) | **최대 10장** | — | **Qwen Research License: 비상업 전용** | **1075 (1위)** | 제외(라이선스). 외부 출력도 학습 금지 원칙 | [검증] HF·LICENSE |
| HunyuanImage 3.0 Instruct | 대형 MoE | 지원 | — | Tencent 커뮤니티 라이선스(영토 조항) [2차] | 1069 (1–2위) | 제외(한국 영토 문제 가능) | [2차] |
| LongCat-Image-Edit (Meituan) | 약 6B [2차] | 단일 이미지(예시) | 50스텝, CFG 4.5 | Apache-2.0 | 939 (12–13위) | 오프로드로 18 GB 요구 → T4 빠듯 | [검증] HF |
| JoyAI-Image-Edit-Plus (JD, 2026-05) | 8B MLLM + 16B MMDiT | 1–6장 | — | Apache-2.0 [2차] | 보드 미등재 | 24B라 평가 전용 | [2차] |
| Step1X-Edit v1p2 | — | — | — | 미확인 | 925 | — | 미확인 |
| FLUX.1 Kontext [dev] | 12B | 1장 | — | FLUX 비상업 | 861 | 제외 | [검증] |

### 3.2 klein 4B 해부 (모델 저장소 config, [검증])

| 부품 | 값 | 의미 |
|---|---|---|
| 트랜스포머 | `Flux2Transformer2DModel`, 이중 스트림 5 + 단일 스트림 20 = 25블록, 헤드 24 × 128 = d 3072, `mlp_ratio` 3.0, `patch_size` 1, `in_channels` 128 | VAE 32ch에 2×2 패치를 미리 접은 128ch 입력이다. 1024² = 64×64 = **4,096 토큰**(f16 격자) |
| 텍스트 | `joint_attention_dim` 7680 = 3 × 2560 | Qwen3-4B(d 2560, 36층) 은닉 3개 층을 이어 붙인다 |
| TE | `Qwen3ForCausalLM`, d 2560, 36층, FFN 9728 | Z-Image와 같은 크기급 TE다. 같은 체크포인트인지는 미확인(호스트 pinned 39 vs 47 GB 판정, 모델 렌즈 §1.7) |
| VAE | latent 32ch, 블록 [128, 256, 512, 512], 배치정규화 통계 사용 | f8 공간 정렬(패치 전). v2.1 계약 A3(f8 정렬) 충족 |
| 정밀도 | 가중치 BF16 | **FP16 Rescue 필요**(Z와 같은 처방) |

**파라미터 역산(추정).** 이중 블록 = 스트림당 (QKV 3d² + 출력 d² + SwiGLU 9d²) × 2 = 26d². 단일 블록 = 융합 입력 9d² + 출력 4d² = 13d². 합 5×26 + 20×13 = 390d² = 390 × 3072² ≈ **3.68B**. 임베더(문맥 7680→3072 약 24M)를 더해 약 3.7B다. 공칭 "4B"와 맞는다.

**fwd 비용(추정).** `FLOPs ≈ 2·P·N + 4·L·d·N²` (P 3.68×10⁹, L 25, d 3072)

| 조건 | 토큰 N | 선형 | 어텐션 | 합 | TFU (Z 1 fwd 6.2×10¹³ = 1) |
|---|---|---|---|---|---|
| 1024², 참조 0, 텍스트 512 | 4,608 | 3.4×10¹³ | 0.65×10¹³ | 4.1×10¹³ | 0.66 |
| + 참조 2장 512² (각 1,024) | 6,656 | 4.9×10¹³ | 1.4×10¹³ | 6.3×10¹³ | **1.01** |
| + 참조 4장 1024² (각 4,096) | 20,992 | 15.5×10¹³ | 13.5×10¹³ | 29×10¹³ | **4.7** |

- 참조 4장 1024²는 어텐션이 선형과 맞먹는다. **참조는 512²로 줄여 넣는 것을 기본**으로 한다. 견본(Swatch)은 원래 작은 패치라 손해가 없다.
- 참조 토큰은 매 스텝 타임스텝 변조를 받으므로 스텝 간 KV 캐시는 일반적으로 안 된다(추정, E0에서 확인).
- T4 시간: v2.1 표의 klein 실효 속도(4×10¹³ / 2.0–2.7 s ≈ 15–20 TF FP16)를 쓰면 참조 2장 fwd는 **3.1–4.2 s(FP16), 2.0–2.7 s(INT8)**다. 4스텝 + VAE면 장당 9–18 s다. 모델 렌즈 E0 (h) 목표(INT8 2-ref ≤3.0 s)와 맞는다.
- 메모리: FP16 가중치 7.4 GB + VAE 0.2 GB + 활성(6.7k 토큰) 약 0.5–1 GB ≈ **8–9 GB**. TE는 한 번 인코딩해 캐시하고 내린다. 공식 "약 13 GB"는 TE 동시 적재 기준으로 보인다(추정).

**Qwen-Image-Edit-2511 비용(추정).** 20B, d 3072, 60층. 목표 1024²(4,096) + 참조 1024²(4,096) + 텍스트 약 300 = 8,492 토큰. 선형 2 × 2×10¹⁰ × 8,492 = 3.4×10¹⁴, 어텐션 4 × 60 × 3072 × 8,492² = 5.3×10¹³. fwd 3.9×10¹⁴ = **6.3 TFU**. true CFG ×2, 40스텝 → **약 507 TFU/장**(Z-Turbo 장당 8 TFU의 63배). v2.1 W4A4 실효 1.8 s/TFU를 쓰면 **약 15분/장**이다. 메모리는 INT4 가중치 약 10–12 GB + TE 오프로드로 한 장에 겨우 들어간다. 시스템 경로 후보가 아니다. Lightning 계열 증류 LoRA는 제3자라 계약상 금지다(v2.1 §3.6 A2).

### 3.3 Z-Image 계열 상태 [검증]

| 날짜 | 사건 |
|---|---|
| 2025-11-26 | Z-Image-Turbo 공개 |
| 2025-12-01 | 기술 보고서 arXiv(2511.22699) |
| 2025-12-08 | AA T2I 종합 8위, 오픈 1위(공식 README 주장) |
| **2026-01-27** | **Z-Image (Base) 공개**: "full-capacity, undistilled", 28–50스텝, CFG 3–5, Apache-2.0 |
| 2026-10-10 | **Z-Image-Omni-Base, Z-Image-Edit: "To be released"**(HF·ModelScope 모두) |

- **함의.** v2.1 Q1("Z-Image Base: 공개·라이선스 확인 필요")의 전제가 풀렸다. 판정 전용 병용은 계약 A1의 "교사 하나"와 충돌하지 않는다. 같은 구조·같은 VAE·같은 TE라 어댑터 재학습 비용이 거의 없다(같은 latent 공간). 판정 1회 2 TFU다.
- Z-Image-Edit는 Turbo 공개(2025-11) 이후 약 10개월째 "예정"이다. **v3 계획에 Z-Image-Edit를 전제하지 않는다.** 공개되면 L2 재오디션(모델 렌즈 Q-M6).
- Nunchaku가 Z-Image-Turbo INT4를 공식 지원한다 [검증]. Base는 미확인.

### 3.4 T4(sm_75) 실행 경로 확인표

| 경로 | 상태 | 확인 |
|---|---|---|
| FP16 | 모든 후보가 BF16 네이티브다. FP16 Rescue(v2.1 §3.5)가 전제다 | [검증] 모델 카드 dtype |
| INT8 (W8A8) | Turing INT8 텐서코어 있음. 공식 INT8 체크포인트는 없다. 우리 손 양자화(계약 A2)가 맞다 | — |
| **W4A4 (Nunchaku SVDQuant)** | v0.2.0(2025-04-05) "20-시리즈 호환", v1.2.0(2026-01-12) "20-시리즈 INT4 지원". 지원 모델: FLUX.1 계열, Qwen-Image(+Edit 2509), **Z-Image-Turbo**. **klein은 공식 미지원**(커뮤니티 포크 "Vitoom Nunchaku"가 sm_75 휠과 klein을 낸다고 주장) | [검증] 공식 README / 포크 [2차] |
| NVFP4·FP8 | Blackwell·Ada 전용. T4에서 불가. BFL의 klein FP8/NVFP4 공개본은 우리와 무관 | [검증] |

- 계약 A2는 사전 양자화 가중치와 제3자 커널을 Q7 판정 전까지 막는다. Nunchaku는 **커널 라이브러리**이고 가중치는 우리 보정셋으로 우리가 양자화한다. 커뮤니티 포크는 쓰지 않는다. klein W4A4는 공식 지원이 나오거나 우리 손으로 포팅할 때까지 미확인으로 둔다.

### 3.5 참조 교사 결정식 입력값 갱신

| v2.1/모델 렌즈 가정 | 2026-10 사실 | 조치 |
|---|---|---|
| klein 멀티 레퍼런스 "내장 [2차]" | **최대 4장** [2차, BFL 도움말]. 4B·Base 4B 모두 Apache [검증] | Reference Ring의 슬롯 상한 4. 견본 2 + 입력 이미지 1 + 스타일 앵커 1 |
| klein σ 4점 문제(R-M3)의 대안 "klein Base 공개 확인 시" | **klein Base 4B 공개·Apache** [검증] | R-M3 대응을 "조건부"에서 "준비됨"으로 바꾼다 |
| Z-Image Base "확인 필요" | 공개·Apache [검증] | Q1을 판정 병용 실험(E2)으로 승격 |
| Z-Image 편집 계열 | 미공개 [검증] | 전제하지 않는다 |
| 오픈 편집 최상위 | Qwen-Image-2.1(비상업), HunyuanImage 3.0 Instruct(영토) | 둘 다 제외. 상업 가능 최상위는 Qwen-Image-Edit-2511(1021)이고 평가 전용 |

---

## 4. 이미지 생성기 약관: 출력 소유·학습 제한 원문 (짧은 인용)

권리 렌즈 §3.1 표와 겹친다. 여기서는 **원문 인용, URL, 확인 등급**만 붙인다. 인용은 영어 원문 그대로 짧게 둔다.

### 4.1 인용표

| 생성기 (판·날짜) | 출력 소유 (원문) | 학습·개발 제한 (원문) | 생성기의 사용자 콘텐츠 학습 | 확인 | URL |
|---|---|---|---|---|---|
| **Higgsfield** (약관 2026-08-27 발효, 도움말 2026-09-12 수정) | "You own your outputs." / "does not restrict the commercial use of outputs" (§4.4) | "You may not use the Service or outputs to train, fine-tune, or improve any AI or machine-learning model" (§5.1(iii), §5.2(iv)). 예외: "models you train with your own inputs … or with Higgsfield's written permission" | "Higgsfield may also use content to train and improve its AI models." Enterprise는 제외. 삭제하면 이후 중단 | [검증] 공식 도움말 + 공식 블로그 | https://higgsfield.ai/creator-hub/help-center/account/who-owns-my-generations-and-can-i-use-them-commercially , https://higgsfield.ai/blog/terms-of-use-privacy-policy-update |
| **OpenAI** (이용약관, 2026-01-01 발효로 보도) | (비즈니스 약관) 고객이 "own all Output" | 금지 행위: "Use Output to develop models that compete with OpenAI" | 소비자 콘텐츠는 학습 가능, 옵트아웃 [2차] | **[2차]** 공식 페이지 403. 검색 색인 원문 | https://openai.com/policies/row-terms-of-use/ , https://openai.com/policies/business-terms |
| **Midjourney** (2026-02-12 또는 05-05 판, 충돌) | 연 매출 100만 달러 넘는 회사(와 직원)는 "Pro" 또는 "Mega" 구독이어야 자기 Assets를 소유 | 경쟁 제품·서비스 개발·제공 목적 사용 금지, 자동화 도구로 접근·생성 금지 | 회사에 영구·취소 불가 라이선스 [2차] | **[2차]** 공식 페이지 403. 약관 추적 사이트 사본 | https://docs.midjourney.com/hc/en-us/articles/32083055291277-Terms-of-Service |
| **Google 소비자 서비스** (서비스 약관 **2026-07-30 발효**) | "Google won't claim ownership over that content." | 금지: "using AI-generated content from our services to develop machine learning models or related AI technology" | — | [검증] | https://policies.google.com/terms |
| **Gemini API** (추가 약관 2026-04-28 수정) | "Google won't claim ownership over that content." | "You may not use the Services to develop models that compete with the Services" | 무료: "uses the content you submit … and any generated responses to provide, improve, and develop". 유료: "doesn't use your prompts … to improve our products"(위반 탐지용 한시 기록) | [검증] | https://ai.google.dev/gemini-api/terms |
| **BFL FLUX [dev] 비상업 라이선스 v2.0** (FLUX.2 [dev], klein 9B 포함) | "We claim no ownership rights in and to the Outputs." | "You may not use the Output to train, fine-tune, or distill a model that is competitive with a FLUX [dev] Model." | — | [검증] 라이선스 원문 | https://huggingface.co/black-forest-labs/FLUX.2-dev (게이트), BFL 라이선스 원문 |
| **World Labs** (약관 2026-01-21 발효) | 무료: "World Labs retains all rights in and to any Output generated by Free Account Users". 유료: "Paid Account Users own all rights, title, and interest in and to Outputs" (§3.3) | §2.8(d): 서비스로 "develop competing AI models" 금지 | §3.6: 사용자 콘텐츠를 "training of World Labs AI Models"에 사용. 무료는 취소 불가, 유료는 옵트아웃 가능 | [검증] | https://worldlabs.ai/terms-of-service |
| **Tencent HY-World 2.0 / HunyuanWorld 1.0** (커뮤니티 라이선스) | — | §5(b): 저작물이나 출력으로 "improve any other AI model" 금지 | — | [검증] 라이선스 원문 | https://github.com/Tencent-Hunyuan/HY-World-2.0 , https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 |
| 〃 영토 조항 | §1(l) "the worldwide territory, excluding the territory of the European Union, United Kingdom and South Korea." | §4: 월간 활성 사용자 100만 초과 시 별도 허가 | — | [검증] | 같음 |
| **Qwen Research License** (Qwen-Image-2.1, 2026-09-20) | — | §2(a) "FOR NON-COMMERCIAL PURPOSES ONLY". §2(b) 상업 목적은 별도 라이선스. §4(b) 출력으로 배포 모델을 학습하면 Qwen 표기 의무 | — | [검증] | https://huggingface.co/Qwen/Qwen-Image-2.1 |
| **NVIDIA Lyra 2.0** | — | 내부 과학 연구·개발 전용. 배포·배치·재허락·공개 상영 금지 | — | [검증] 모델 카드 요지 | https://huggingface.co/nvidia/Lyra-2.0 |
| **Apache-2.0 오픈 모델** (Z-Image, Z-Image-Turbo, klein 4B, klein Base 4B, Qwen-Image-Edit-2511) | 출력 조건 없음 | 출력 사용 제한 없음 | — | [검증] 모델 카드 태그 | 각 HF 모델 카드 |

### 4.2 이번에 새로 확인된 것 (권리 렌즈 표에 더할 것)

1. **Google 소비자 약관이 2026-07-30판으로 바뀌었다.** 금지 문구가 "AI-generated content from our services"로 넓다. Gemini 앱·Nano Banana 소비자 출력이 사용자 업로드로 들어오면 **우리가 그것으로 개발하는 순간** 위반 소지가 있다. USER 방화벽의 근거가 하나 더 생긴다.
2. **World Labs 약관은 무료 계정 출력의 권리를 World Labs가 갖는다.** 사용자가 Marble 무료 계정으로 만든 렌더를 우리에게 올리면, 사용자는 그 이미지의 권리자가 아니다. 업로드 보증 조항(권리 렌즈 §2)이 이 경우를 막아야 한다. 벤치에서 Marble을 돌릴 때는 **유료 계정**이어야 결과를 공개할 수 있다.
3. **Tencent 영토 조항은 세계 생성 기준선 선택을 바꾼다.** 한국에서 HY-World를 내려받아 돌리는 것 자체가 허락 밖일 수 있다. 법무 항목 L-new: "회사·개발자 소재지와 Tencent 영토 조항".
4. **Higgsfield 신약관(2026-08-27)은 라이선스를 "서비스 운영에 필요한 범위"로 좁혔다고 공식 블로그가 쓴다.** 그러나 도움말은 여전히 "content to train and improve its AI models"를 적는다. 둘 다 [검증]이고 내용이 겹친다. 학습 사용은 남아 있다.
5. **오픈 최상위 편집 모델 두 개가 상업 불가다**(Qwen-Image-2.1 비상업, HunyuanImage 3.0 영토). 오픈 모델 출력 화이트리스트(모델 렌즈 §4.8)에서 둘 다 "금지"로 넣는다.

### 4.3 함의 (요지만, 상세는 권리 렌즈)

| 질문 | 답 | 근거 |
|---|---|---|
| 사용자가 Soul·GPT Image·Midjourney·Gemini 이미지를 올려 영상을 받는 것 | 생성기 약관이 막지 않는다. 사용자 쪽 의무(플랜, 매출 요건)는 업로드 보증으로 사용자에게 | §4.1 소유 조항 |
| 그 업로드로 우리가 학습 | **금지**(Higgsfield·Google 명시, OpenAI·Midjourney 경쟁 조항) | §4.1 제한 조항 |
| 그 업로드로 우리가 코드·HP를 고침(개발) | Google·Higgsfield 문구상 위험. 권리 렌즈 CT-R3(프록시 재현 규칙) 유지 | §4.1 |
| 경쟁 I2V API에 외부 생성 이미지를 입력 | 그 API가 입력으로 학습하면 출처 약관과 부딪칠 수 있다. Gemini API 유료 등급은 개선에 쓰지 않음 [검증]. 나머지는 미확인 → 공개 벤치는 우리 촬영·CC0·Apache 모델 출력만 | §4.1 Gemini API |

---

## 5. 단일 이미지 기하·내재·조명·카메라 추정 SOTA (평가 전용 증인)

**원칙.** 이 절의 모델은 **시스템 경로에 넣지 않는다.** 우리 판독기(MRH, InverseLook, FitLights)를 채점하는 외부 증인으로만 쓴다. 출력은 학습·보상·체크포인트 선택에 들어가지 않는다(from-scratch 순도와 라이선스 둘 다를 지킨다). 계보 태그는 `EVAL-ONLY(model, license-hash)`다.

### 5.1 과제별 후보표

| 과제 | 후보 (날짜) | 출력 | 크기 | 라이선스 | 우리 KPI 연결 | 확인 |
|---|---|---|---|---|---|---|
| 메트릭 pointmap + FOV | **MoGe-3** (2026-08-18 공지) | 메트릭 pointmap·깊이·내부 파라미터·마스크·법선 | ViT-L 370M, ViT-g 1.25B | 코드 MIT, 가중치 미기재 | 첫 프레임 기하 오차, 에셔 점수(추정기 간 불일치), 미터 스케일 | [검증] GitHub README |
| 〃 | MoGe-2 (arXiv 2507.02546) | 메트릭 pointmap, FOV, 법선 | 35M / 104M / 326–331M | 코드 MIT, 가중치 미기재 | 같음. ViT-S(35M)는 T4에서 대량 채점용 | [검증] |
| 메트릭 깊이 + 초점거리 | **Depth Pro** (Apple, arXiv 2410.02073) | 2.25 MP 메트릭 깊이, `focallength_px` | — | 저장소 LICENSE(종류 미기재) | InverseLook FOV 헤드 채점 | [검증] 저장소 / 라이선스 미확인 |
| 깊이·포즈·내부 파라미터·3DGS | **Depth Anything 3** (2025-11-14, 1.1판 재학습) | 단안·다시점 깊이, 포즈, 내부 파라미터, 3DGS, 메트릭(Metric 판) | Small 0.08B, Base 0.12B, Large 0.35B, Giant 1.15B, Nested 1.40B | **Small·Base·Metric-Large·Mono-Large만 Apache-2.0.** Large·Giant·Nested는 CC BY-NC 4.0 | 경쟁 I2V 프레임의 포즈·깊이(경로 추종, IR 근사). 메트릭 = `focal × out / 300` | [검증] GitHub |
| 단일 이미지 보정 | **GeoCalib** (ECCV 2024) | 중력(roll, pitch), 초점·FOV, 왜곡 k1/k2(pinhole·radial·divisional) | — | 코드 Apache-2.0, 가중치 CC BY 4.0 | 에셔 점수의 수평선·소실 성분, θ 렌즈 왜곡 채점 | [검증] |
| 〃 | AnyCalib (ICCV 2025) | 픽셀별 광선 회귀, 카메라 모델 무관, 크롭·늘림에 강함 | — | 미확인 | 크롭·리사이즈된 AI 이미지 입력의 FOV | [검증] 논문 / 라이선스 미확인 |
| 내재 분해 | **Marigold-IID v1-1** (2025-05-15) | Appearance: albedo·roughness·metallicity. Lighting: albedo·확산 음영·비확산 잔차 | SD 기반 | 코드 Apache-2.0, **가중치 RAIL++-M** | MRH albedo/shading 채점, FitLights 잔차 | [검증] GitHub |
| 〃 | Careaga & Aksoy "Colorful Diffuse Intrinsic Image Decomposition in the Wild" (TOG 2024) | albedo, 유색 확산 음영, 비확산 잔차 | — | 미확인 | 유색 조명(혼합 색온도) 장면 채점 | [2차] |
| 〃 (비디오) | DiffusionRenderer (NVIDIA, CVPR 2025) | G-buffer(법선, 깊이, albedo, roughness, metallic) | 비디오 확산 | NVIDIA Source Code License, 비상업 | 우리 렌더 클립의 역렌더 일관성(내부만) | [2차] |
| 조명 | **DiffusionLight-Turbo** (arXiv 2507.01305) | 크롬볼 인페인팅으로 HDR 환경맵 | SDXL + Turbo LoRA | 미확인(SDXL 기반 → OpenRAIL++ 계열 추정) | FitLights 환경광·키라이트 방향 채점 | [검증] 초록 / 라이선스 미확인 |
| 노출 메타 | EXIF as Language (CVPR 2023) | 이미지 패치와 EXIF의 공동 임베딩. 왜곡 추정·포렌식 | — | 미확인 | 렌즈·센서 단서 보조 | [2차] |
| 다시점 기하 | VGGT, π³/Pi3X, MapAnything, VGGT-Ω, WorldMirror 2.0 | 지형도 §6 | — | VGGT-Ω 등은 지형도 참조. **WorldMirror는 Tencent 영토 조항** | GeCo·MEt3R 계산 | 지형도 |

### 5.2 평가 증인 앙상블 (의사코드)

증인 하나를 정답으로 믿지 않는다. **앙상블 중앙값과 분산**을 함께 쓴다. 분산 자체가 에셔 점수의 성분이 된다(파이프라인 렌즈 §2.9와 접속).

```
  # EVAL-ONLY. 결과는 R-EVAL 보고서로만 간다. 학습·보상·체크포인트 선택 금지.
def witness_geometry(I):                       # I: 입력 이미지(원 해상도)
    W = {"moge3": MoGe3_L, "da3m": DA3_Metric_L, "dpro": DepthPro}   # 상업 가능 라이선스만 기본
    out = {k: m(I) for k, m in W.items()}       # 각자 depth(px), fov_x
    fov = median([o.fov_x for o in out.values()])
    fov_spread = MAD([o.fov_x for o in out.values()])          # 도(°)
    D = {k: o.depth * (median_depth_all / median(o.depth)) for k, o in out.items()}  # 스케일 정렬
    disagree = mean_over_pixels(std_k(log D[k]))               # 로그 깊이 불일치
    g = GeoCalib(I)                                            # roll, pitch, k1
    return dict(fov=fov, fov_spread=fov_spread, disagree=disagree,
                roll=g.roll, pitch=g.pitch, k1=g.k1, fov_geocalib=g.fov)

def score_inverselook(theta_hat, I):            # 우리 θ 추정 채점
    w = witness_geometry(I)
    e_fov = abs(theta_hat.fov - w["fov"])                       # 증인 대비 오차
    flag  = e_fov > 3 * max(w["fov_spread"], 1.0)               # 증인끼리도 갈리면 판정 보류
    return e_fov, flag
```

- **T4 비용(추정).** MoGe-2 ViT-L은 A100·3090에서 FP16 60 ms다(공식). T4는 3090 대비 FP16 약 1/2–1/3이라 약 120–180 ms다. 입력 1장당 증인 3개 + GeoCalib ≈ 1 s 미만이다. 벤치 330장 × 수 회는 무시할 수준이다.
- **라이선스 규칙.** 기본 앙상블은 상업 가능 라이선스(MIT·Apache·CC BY)만 쓴다. CC BY-NC(DA3 Giant)·RAIL++-M(Marigold)·비상업(DiffusionRenderer)은 **내부 진단 보고에만** 쓰고, 공개 수치에는 쓰지 않는다(법무 확인 전까지).

### 5.3 공백: 렌즈·초점·조리개 추정에는 공인 SOTA가 없다

- 찾은 것: 초점거리·FOV(Depth Pro, MoGe, GeoCalib, AnyCalib), 왜곡(GeoCalib k1/k2), 디포커스 **디블러링**(RDDM AAAI 2025 등). **단일 이미지에서 f-number와 초점 거리(focus distance)를 함께 추정하는 공개 SOTA·벤치는 찾지 못했다**(미확인 = 없음으로 간주).
- 물리: 착란원 `c = K·|s − s_f| / s`, `K = f² / (N·(s_f − f))` (s 피사체 거리, s_f 초점 거리, N f-number). 깊이 s(p)와 블러 c(p)를 여러 깊이에서 알면 **초점면 s_f와 K는 정해진다**. 그러나 블러를 픽셀로만 재므로 센서 크기를 모르면 **N은 환산 조리개(풀프레임 등가)로만** 정해진다(추정, 식의 자유도). 렌더에는 환산값이면 충분하다(같은 그림). 실제 난점은 둘이다. (i) 질감 없는 영역에서는 블러를 잴 수 없다. (ii) 깊이 증인 없이는 s_f 앞뒤의 같은 블러를 구분하지 못한다(전후 모호성). 깊이 증인이 있으면 풀린다. 그래서 InverseLook의 DoF 성분은 "깊이 증인 + 블러 지도 + 렌즈 세트 사전"으로 s_f와 환산 N을 푸는 문제다.
- **θ-GT 세트 제안 (RS9).** 자체 촬영 300장. 카메라 2대(풀프레임·APS-C) × 렌즈 5종(24/35/50/85/135 mm) × 조리개 3(f/1.8, f/4, f/11) × 장면 10. 장면마다 회색 카드(WB·노출 정답), 크롬볼(조명 정답), 줄자(스케일 정답)를 한 컷 따로 찍는다. EXIF + 렌즈 메이커노트의 초점 거리 필드를 정답으로 쓴다(제조사별 필드 확인 필요). 비용은 사람 2일(추정). 순도 등급 D0(실측)이고 `bench-holdout`로 묶는다.
- 이 세트가 없으면 "InverseLook이 θ를 맞췄다"는 주장은 증인 일치도일 뿐이다. 벤치 공개 전에 만든다.

---

## 6. I2V 평가 지표·벤치 선례

지형도 §5(MEt3R, GeCo, WorldScore, VBench-2.0, 왕복 계열, CameraBench)는 반복하지 않는다. **I2V 고유 축**(첫 프레임 보존, 카메라 경로 추종)과 2026년 갱신만 적는다.

### 6.1 선례표

| 축 | 지표·벤치 | 정의 (요지) | 한계 | 확인 |
|---|---|---|---|---|
| 첫 프레임 보존 | **AIGCBench** MSE(First), SSIM(First), Image-GenVideo CLIP | 생성 첫 프레임 대 입력 이미지의 MSE·SSIM. CLIP은 입력 대 모든 프레임 | 픽셀 지표라 그레인·톤 차이에 과민. 저주파·고주파를 구분하지 않음 | [2차] 리뷰 요약 |
| 〃 | **VBench++ I2V Subject** | DINOv1 특징. 입력↔각 프레임 유사도와 연속 프레임 유사도의 가중 평균(가중치 비공개) | 입력 위치가 모델마다 달라 "첫 프레임"이 아니라 "전 구간 유사도"다 | [검증] arXiv HTML |
| 〃 | **VBench++ I2V Background** | DreamSim 특징(배경 변화에 민감), 같은 2항 가중 평균 | 같음 | [검증] |
| 카메라(분류) | **VBench++ Camera Motion** | 네 가장자리 점을 CoTracker로 추적 → 휴리스틱으로 7종(팬 좌·우, 틸트 상·하, 줌 인·아웃, 정지) 분류 → 프롬프트와 일치율 | 오빗·크레인·트럭·달리와 줌을 구분 못 함. 우리 🟡·🔴 무브를 못 잰다 | [검증] |
| 이미지 세트 | **VBench++ Image Suite** | Pexels·Pixabay 출처. 픽셀 면적 기준 85.6%가 4K–8K. 전경 6범주, 배경 4범주(건축, 풍경, 실내, 추상). 1:1·7:4·8:5·16:9 크롭 | 주 피사체가 화면을 작게 차지하도록 골랐다(크롭 생존) | [검증] |
| 카메라(포즈) | **RotErr, TransErr, CamMC** (MotionCtrl·CameraCtrl·CamI2V 계보) | SfM(COLMAP·GLOMAP)으로 생성 영상의 포즈를 추정해 명령 포즈와 비교. RealCam-I2V: 1,000샘플 × 5회 시도 | **스케일 정규화가 논문마다 다르다**(RealCam-I2V 클립별 장면 스케일, CamGeo 무정규화 L2, MeRoPE 최대 이동 노름). 논문 간 수치 비교 불가 | [2차] 검색 요약, 논문 ID 확인 |
| 카메라(사람) | **Atlas 카메라 테스트** | 같은 이미지 + 무브 1–3개, 제3자 평가자가 경로 추종 2AFC | 자체 실행, 경쟁사는 텍스트 입력 | [검증] |
| 정지 세계 일관성 | **GeCo** (v5, 2026-08-19) | 움직임 잔차 + 구조 잔차 융합, 정지 장면 변형·가림 불일치 검출, 학습 없는 가이던스 손실로도 사용 | 코드 공개 여부 미확인. Kling·Seedance 미측정(지형도) | [검증] arXiv 목록 |
| 사람 선호 | **AA I2V 아레나** | 오디오 포함 패널 투표. "AA-Video-I2V v2.0 coming soon" | 정지 세계·카메라 무브 전용 아님 | [검증] |
| 데이터 | **I2V-Bench** (ConsistI2V) | YouTube 2,950개, 16범주. VBench식 차원 | 2024 세대 | [2차] |

### 6.2 카메라 경로 지표: 우리가 쓸 정의 (정규화 고정)

명령 경로 `{R*_t, T*_t}`(우리 DP-0 출력)와 추정 경로 `{R_t, T_t}`(DA3 Apache 판 또는 VGGT로 생성 프레임에서 추정)를 비교한다.

1. 첫 프레임 정렬: 두 경로 모두 `R_0 = I, T_0 = 0`으로 옮긴다.
2. **스케일 고정 규칙:** `s = argmin_s Σ_t ‖T*_t − s·T_t‖²` (Umeyama 스케일만, 회전·이동은 1번에서 고정). 명령 경로의 최대 이동 노름 `M = max_t ‖T*_t‖`로 나눠 무차원화한다. MeRoPE식 정규화와 같은 계열이다.
3. 지표:
   - `RotErr = (1/T) Σ_t arccos((tr(R*_tᵀ R_t) − 1)/2)` [도]
   - `TransErr = (1/T) Σ_t ‖T*_t − s·T_t‖₂ / M` [무차원]
   - `CamMC = (1/T) Σ_t ‖[R*_t | T*_t/M] − [R_t | s·T_t/M]‖_F`
4. 추정기 바닥값: **우리 렌더**(명령 경로가 정답인 영상)에 같은 추정기를 돌린 값을 `floor`로 보고한다. 경쟁 결과는 `지표 − floor`로 함께 적는다. GeCo의 "실제 영상 바닥 0.02"와 같은 논리다.
5. 시도 5회(추정기 무작위성), 중앙값 보고(RealCam-I2V 프로토콜 계승).

**주의.** 경쟁 I2V는 텍스트로 경로를 받으므로 크기·속도가 어긋나는 것은 당연하다. 그래서 **방향 일치**(무브 종류, 회전축 부호)와 **크기 오차**를 나눠 보고한다. 방향은 VBench++ 7종을 우리 무브 12종(푸시인, 풀아웃, 트럭 좌·우, 페데스탈 상·하, 팬, 틸트, 오빗 ±, 크레인, 랙 포커스)으로 넓힌 분류기로 잰다(분류기는 우리 렌더로 보정).

### 6.3 첫 프레임 보존: 우리가 쓸 정의

`FFD = DreamSim(F_0, I_in)` 를 주 지표로, `LPIPS`, `DISTS`, `ΔE00(평균 색차)`를 보조로 둔다. 근거: AIGCBench는 MSE·SSIM(픽셀), VBench++는 DINO·DreamSim(의미·배경)을 썼다. 우리는 그레인·톤까지 보는 지각 지표가 필요하다.

- **대역 분리 보고:** `FFD_low = FFD(↓4(F_0), ↓4(I_in))`, `FFD_high = FFD_full − FFD_low` (추정 정의). 저대역은 레이아웃·톤, 고대역은 질감이다. 입력 시점의 대역 소유권(입력이 저·중대역 소유)과 1:1로 대응한다.
- **왕복 프레임:** 경로가 c_0으로 돌아오는 무브(🟡·🔴)는 마지막 프레임에도 FFD를 잰다. ViewBench LCE(지형도 §2.2)와 같은 정의다.
- 경쟁 I2V의 F_0은 입력과 거의 같게 나온다(첫 프레임 조건). 그래서 **F_0 FFD는 변별력이 낮다.** 변별력은 **t = 1 s, 2 s 프레임의 "같은 시점 영역" 보존**에서 나온다. 6.4의 관측 마스크 안에서 입력을 재투영해 비교한다(`FFD_obs(t)`).

### 6.4 경쟁 I2V의 발명 비율(IR) 근사 (의사코드)

우리 IR은 세계 소유권 지도에서 정확히 나온다. 경쟁 영상은 프레임만 있으므로 증인 기하로 근사한다.

```
  # EVAL-ONLY 증인: DA3(Apache 판) 또는 VGGT로 프레임 포즈·깊이, MoGe로 입력 깊이
def invention_ratio(frames, I_in, tau=0.05):
    K0, D0 = moge(I_in).intrinsics, moge(I_in).depth          # 입력 시점 기하
    poses, depths, Ks = da3_multiview([I_in] + frames)           # I_in을 0번 뷰로 고정
    s = scale_align(depths[0], D0)                               # 증인 간 스케일 정렬
    ir_t = []
    for t, F in enumerate(frames, start=1):
        X = backproject(depths[t] * s, Ks[t], poses[t])          # 프레임 t 픽셀의 3D 점
        u, z = project(X, K0, pose_identity)                     # 입력 카메라로 투영
        inside = in_frustum(u, I_in.shape) & (z > 0)
        visible = inside & (z <= D0.sample(u) * (1 + tau))       # 깊이 시험(가림)
        ir_t.append(1 - visible.mean())                          # 관측 안 된 픽셀 비율
    return mean(ir_t), max(ir_t)

  # 같은 마스크로 FFD_obs(t): visible 영역에서 I_in을 프레임 t로 재투영해 DreamSim 비교
```

- **검증:** 우리 렌더에 이 근사를 돌려 정확한 IR과의 오차를 보고한다(근사 바닥값). 오차가 크면 경쟁 IR을 구간으로만 적는다.
- **해석 규칙:** 같은 무브에서 경쟁 IR이 우리보다 **낮으면** 경쟁 모델이 명령보다 덜 움직인 것이다(경로 추종 실패의 다른 얼굴). IR은 항상 6.2의 경로 지표와 같이 보고한다.

### 6.5 공백과 우리 벤치의 위치

| 축 | 선례 | 공개 상용 비교 | 우리 |
|---|---|---|---|
| 첫 프레임 보존 | AIGCBench, VBench++ | VBench++ 리더보드(오픈 위주) | FFD(대역 분리) + FFD_obs(t) |
| 카메라 경로 | RotErr·TransErr·CamMC, Atlas 2AFC | Atlas 자체 테스트만 | 정규화 고정 + floor 보고 + 방향/크기 분리 |
| 정지 세계 일관성 | GeCo, MEt3R, WorldScore | GeCo(Sora 2·Veo 3.1·Wan 2.2) | GeCo ≤0.03 목표(v2.1 그대로) |
| 발명 비율 | **없음** | 없음 | IR 정확값(우리) + 근사(경쟁) |
| 에셔(입력의 3D 비성립) | **없음** | 없음 | 증인 불일치 + GeoCalib 성분(§5.2) |
| 해상도 | MTF 계열(지형도) | 없음 | MTF50, 8K 트랙 |

**결론.** 다섯 축을 한 프로토콜로 묶은 공개 벤치는 없다. FREEZE-TAG v3(T2 트랙)를 공개하면 **"사진 한 장 → 카메라 무브" 평가의 첫 표준 후보**가 된다. 단 공개 재현성을 위해 입력은 자체 촬영·CC0·Apache 모델 출력만 쓴다(§4.3). VBench++ 이미지 세트(Pexels·Pixabay)는 재배포 조건이 각 사이트 라이선스에 묶이므로 **링크·ID 목록으로만** 공개한다(확인 필요).

---

## 7. 미확인·상충 장부 (v3 본문 인용 전 재확인)

| # | 항목 | 상태 | 재확인 방법 |
|---|---|---|---|
| U1 | Kling 3.0/O3/4.0 공식 사양·가격 | 공식 문서가 JS 렌더라 판독 실패. 재판매처·AA 값만 | 벤치 계정으로 API 문서 직접 확인 |
| U2 | Kling "Motion Control"이 카메라 경로인가 캐릭터 모션 전이인가 | 출처 충돌 | 실제 API 호출 |
| U3 | Seedance 2.5 I2V 1080p 지원 | Runway API "1080p 향상" vs Civitai "1080p 거부" | BytePlus ModelArk 문서 |
| U4 | Kling 4.0 정식 출시일·가격 | "10월" 예고만 | 출시 후 갱신 |
| U5 | Runway Gen-4.5 I2V 해상도·길이 | 재판매처 간 충돌(5/8/10 s, 720p/1080p) | Runway API 레퍼런스 |
| U6 | AMD–World Labs 인수 세부·Marble 지속성 | 보도만(2026-09-26~29 날짜 충돌) | AMD 보도자료 |
| U7 | Atlas 접근 조건·약관 | 미공개 | 조기 접근 신청 |
| U8 | FlashWorld 가중치 라이선스 | 저장소 미기재, 보도는 비상업 | HF 모델 카드 |
| U9 | Marble splat 세계가 파노(2560×1280)보다 고주파를 담는가 | 미확인 | 벤치에서 MTF50 실측 |
| U10 | HunyuanWorld 1.0 공개일(2025-07)·방식(파노→레이어→메시) | 기억·2차 기반 | 기술 보고서 |
| U11 | klein 4B 참조 상한 4장 | BFL 도움말(검색 요약) [2차] | 공식 기술 사양 페이지(현재 404) |
| U12 | klein 4B와 Z-Image의 TE가 같은 Qwen3-4B 체크포인트인가 | 구조 동일만 확인 | 가중치 해시 비교 |
| U13 | Qwen-Image 계열 구조(d 3072, 60층) | 기억 기반, 비용 계산에 사용 | 모델 config |
| U14 | Nunchaku Turing에서 Z-Image-Turbo·Qwen-Edit 실제 동작 | 공식은 "20-시리즈 INT4 지원" + 모델 지원 목록. 조합 실측 없음 | E0 (h) |
| U15 | OpenAI·Midjourney 약관 현행 원문 | 공식 페이지 403, 추적 사이트·색인 사본 [2차] | 브라우저 직접 열람 후 해시 보관 |
| U16 | Depth Pro, MoGe 가중치, AnyCalib, DiffusionLight-Turbo, Careaga & Aksoy 라이선스 | 미기재·미확인 | 각 LICENSE 파일 |
| U17 | VBench++ I2V Subject·Background 가중치 | 논문 본문에 없음 | 공개 코드 |
| U18 | GeCo 코드 공개 | 미확인 | 저자 페이지 |
| U19 | AA I2V 보드 스냅샷 날짜 | 표기 없음(Vidu Q4 Preview 2026-10 등재로 10월 이후) | 벤치 시점에 재조회 |
| U20 | Midjourney 소유 조건의 매출 기준(총매출 vs 연매출), 약관 판 날짜 | 2차 출처끼리 충돌 | 원문 |
| U21 | Tencent 영토 조항이 회사 소재지 기준인가 사용 장소 기준인가 | 원문은 "Territory" 정의만 | 법무 |

## 8. 출처 (URL)

**경쟁 I2V**
- Veo 3.1 사양: https://ai.google.dev/gemini-api/docs/veo [검증]
- Gemini API 가격(2026-10-09): https://ai.google.dev/gemini-api/docs/pricing [검증]
- Gemini Omni Flash: https://ai.google.dev/gemini-api/docs/omni [검증]
- Runway API 모델·가격: https://docs.dev.runwayml.com/guides/models/ , https://docs.dev.runwayml.com/guides/pricing/ [검증]
- Wan(Alibaba Model Studio, 2026-09-28): https://www.alibabacloud.com/help/en/model-studio/video-generate-edit-model [검증]
- MiniMax H3: https://platform.minimax.io/docs/guides/video-generation [검증]
- Luma Ray 3.2 가격: https://docs.agents.lumalabs.ai/guides/pricing/ [검증]
- Seedance 2.5: https://seed.bytedance.com/en/seedance2_5 [검증], https://developer.civitai.com/orchestration/recipes/seedance [2차]
- Kling: https://fal.ai/models/fal-ai/kling-video/o3/4k/image-to-video/llms.txt , https://fal.ai/learn/tools/what-is-kling-4-0 [2차]
- Vidu Q4: https://www.vidu.com/vidu-q4 [검증]
- AA I2V 아레나: https://artificialanalysis.ai/video/leaderboard/image-to-video [검증]
- Atlas: https://www.worldlabs.ai/blog/atlas [검증]
- AMD–World Labs: https://www.sdxcentral.com/news/amd-acquires-ai-research-firm-world-labs-in-all-stock-deal-worth-82b/ [2차]
- Marey: https://help.moonvalley.com/en/articles/11725660-camera-motion-from-image [2차]

**이미지 → 3D 세계**
- Marble 문서: https://docs.worldlabs.ai/marble/release-notes , https://docs.worldlabs.ai/marble/export/specs , https://docs.worldlabs.ai/api/pricing , https://docs.worldlabs.ai/marble/create/prompt-guides/image-prompt [검증]
- World Labs 약관: https://worldlabs.ai/terms-of-service [검증]
- HY-World 2.0: https://github.com/Tencent-Hunyuan/HY-World-2.0 (License.txt) [검증]
- HunyuanWorld 1.0: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 (LICENSE) [검증]
- Lyra 2.0: https://huggingface.co/nvidia/Lyra-2.0 [검증]
- FlashWorld: https://github.com/imlixinyang/FlashWorld [검증]
- Matrix-3D: https://github.com/SkyworkAI/Matrix-3D [검증]
- SHARP: https://arxiv.org/abs/2512.10685 , https://huggingface.co/apple/Sharp [검증]
- WonderWorld: https://github.com/KovenYu/WonderWorld [검증]
- Echo-2: https://radiancefields.com/spaitial-ai-announces-new-model-echo-2 [2차]
- Project Genie: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/ [검증]

**참조 조건 오픈 모델**
- FLUX.2 klein 4B / Base 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B , https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B (config: transformer, text_encoder, vae) [검증]
- klein 공개 블로그(2026-01-15): https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence [검증]
- Z-Image: https://github.com/Tongyi-MAI/Z-Image , https://huggingface.co/Tongyi-MAI/Z-Image [검증]
- Qwen-Image-Edit-2511: https://huggingface.co/Qwen/Qwen-Image-Edit-2511 [검증]
- Qwen-Image-2.1: https://huggingface.co/Qwen/Qwen-Image-2.1 (LICENSE) [검증]
- LongCat-Image-Edit: https://huggingface.co/meituan-longcat/LongCat-Image-Edit [검증]
- AA 편집 오픈 보드: https://artificialanalysis.ai/image/leaderboard/editing/open-weights [검증]
- Nunchaku: https://github.com/nunchaku-tech/nunchaku [검증]

**약관** — §4.1 표의 URL.

**추정기(평가 전용)**
- MoGe(-2/-3): https://github.com/microsoft/MoGe [검증]
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3 [검증]
- Depth Pro: https://github.com/apple/ml-depth-pro , https://arxiv.org/abs/2410.02073 [검증]
- GeoCalib: https://github.com/cvg/GeoCalib [검증]
- AnyCalib: https://arxiv.org/abs/2503.12701 [검증]
- Marigold: https://github.com/prs-eth/Marigold [검증]
- DiffusionLight-Turbo: https://arxiv.org/abs/2507.01305 [검증]
- DiffusionRenderer: https://github.com/nv-tlabs/diffusion-renderer [2차]
- EXIF as Language: https://arxiv.org/abs/2301.04647 [2차]

**평가 지표**
- VBench++: https://arxiv.org/html/2411.13503 [검증]
- GeCo: https://arxiv.org/abs/2512.22274 [검증]
- AIGCBench: https://arxiv.org/abs/2401.01651 [2차]
- ConsistI2V / I2V-Bench: https://arxiv.org/abs/2402.04324 [2차]
- RealCam-I2V: https://arxiv.org/abs/2502.10059 , CameraCtrl: https://arxiv.org/abs/2404.02101 , CamGeo: https://arxiv.org/abs/2605.30895 , MeRoPE: https://arxiv.org/abs/2609.01252 , Track2View: https://arxiv.org/abs/2606.15534 [2차]
