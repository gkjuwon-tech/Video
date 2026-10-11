# 밤 잡 night2 · M14 Glyph-S 스파이크 명세 (2026-10-11 밤 → 2026-10-12 08:00 KST)

스프린트 D1(15일 중 첫날). 백로그 MDL-20(검출기, todo) + DAT-12(글리프 합성, todo)의 **스파이크 레인**이다. 정식판은 나중에 MDL-20/DAT-12 단위가 `buc/models/glyph/detector.py`, `buc/data/glyph_synth.py`로 만든다. 이 스파이크는 RTX 5090 한 장에서 **GPU 1.5시간 이하**(10월 22일까지 GPU 크레딧 부족)로 돌아가는 독립 잡이다.

- 코드: `ops/gpu/night2/m14_glyph.py`(잡 한 파일) + `ops/gpu/night2/test_m14_glyph.py`(pytest, CPU). 다른 파일은 만들지 않는다. 폰트 표(아래 §3)는 잡 파일 안의 상수 `FONTS`로 넣는다.
- `buc/`, `tests/`, `ops/harness/`는 건드리지 않는다. `buc/` import도 이 잡에서는 하지 않는다(필요한 것이 없고, `buc.core.det`가 blake3를 import 시점에 불러 Vast 이미지에서 깨진다). RUN-06 하네스(`buc.train.*`)도 쓰지 않는다(이유: blake3 import, `deterministic=True` 기본값이 업샘플 역전파를 막음, 절대 마감·N2 마커·DONE.json·평가 훅 없음, `batch_fn`이 메인 스레드 동기 실행). 골격은 `ops/gpu/day1/m2_latent_lift.py`를 **복사**한다(import 금지).
- 왜곡 함수는 `ops/gpu/day1/m6_tlpips.py`의 `op_jpeg`, `op_blur`, `op_noise`, `op_resample`, `op_color`를 복사한다(그 파일은 모듈 수준에서 torch를 불러오므로 import 금지).

## 0. 08:00 KST에 있을 것 (예상)

| 항목 | 예상 | 근거 |
|---|---|---|
| 모델 | **M14 Glyph-S v0**, U-Net-lite **4,028,354 파라미터**(CPU에서 측정), 입력 [B,3,H,W] → 로짓 [B,2,H/4,W/4]. ch0 = 텍스트 확률(PARADIGM 인터페이스 [B,1,H/4,W/4]), ch1 = 가짜 글리프 확률(벌점용) | §5 |
| 금고 | `ckpt/night2/m14/` 에 ckpt_best.pt(EMA 가중치), ckpt_last.pt, metrics.jsonl, previews/*.jpg, fonts_manifest.json, data_manifest.json, licenses/, config.json, DONE.json | §10 |
| 학습량 | 중앙값 약 **1.2M 고유 합성 샘플**(384²), 재사용 2회로 약 **2.4M 샘플-뷰**, 약 **37k 스텝**(배치 64). 비관 0.86M/27k, 낙관 1.7M/48k(상한) | §7 |
| GPU 시간 | 잡 벽시계 합계 **5,100–5,400초 이하**(준비 + 학습 4,320초 + 평가 + 마무리) | §7 |
| 합성 지표(예상) | val_id 픽셀 AP 0.90–0.95, F1@0.5 0.85–0.90 · val_font(학습에 없던 폰트 4종) AP 0.85–0.92 · 영역 재현율(em ≥ 16 px) 0.95 이상, em 9–16 px 0.80–0.90 · 텍스트 없는 질감 이미지 FP율 3–10%(털·풀이 가장 어려움) · 가짜/진짜 AUROC 0.90–0.97 | §8, 추정치 |
| 모르는 것 | **실제 교사 이미지의 가짜 글자**에 대한 성능은 라벨이 없으니 숫자로 못 낸다(R2: 신경망 라벨 금지). 실내 교사 이미지 top-K 시트와 야크·경쟁 모델 프레임 시트를 사람이 보고 판정한다 | §8.6 |
| 백로그 진척 | MDL-20: 스파이크 v0 체크포인트(정식 아님, 계약서 없음). DAT-12: 생성기 범위의 약 70%(폰트 57개·라이선스 기록·11,172 음절·가짜 글리프 8종·질감 음성)가 잡 안에 있음. 빠진 것: `buc/data` 모듈화, 샤드·출처 방화벽 연결, CTC 문자열(범위 밖) | — |

실패 시나리오와 그때 08:00 모습: (a) CPU를 다른 잡이 먹으면 고유 샘플이 0.5M 수준으로 줄지만 체크포인트는 나온다. (b) 폰트 다운로드가 막히면 jsDelivr 미러로 간다. 둘 다 실패하면 `fatal`로 끝나고 체크포인트는 없다(첫 3분 안에 드러난다). (c) 교사 코퍼스가 없으면 절차 배경만으로 학습한다(`bank:0` 마커). (d) 러너가 `^N2_` 마커를 걸러내면 학습은 되지만 로그가 안 보인다(§11).

## 1. 역할과 출력 정의 (결정)

- PARADIGM-v2 M14 행: "[B,3,H,W] → 텍스트 확률 [B,1,H/4,W/4]", 그리고 엔진 점수 `− GlyphS(pseudo-text)`(876행), PSH 표(1382행) "교사가 현판에 그린 가짜 한자는 Glyph-S가 벌점을 주고 SYMBOL로 교체".
- 채널 하나로는 "진짜 글자"와 "가짜 글자"를 동시에 뜻할 수 없다. 가짜를 0으로 학습하면 배경과 구별이 안 되어 벌점을 줄 수 없다. **결정: 헤드를 2채널로 학습한다.**
  - ch0 `glyph`: 글자 모양 잉크 영역(진짜든 가짜든). 목표 `t0 = max(real, fake)`. PARADIGM의 [B,1,H/4,W/4] 텍스트 확률은 `sigmoid(ch0)`이다.
  - ch1 `fake`: 가짜 글리프 영역. 목표 `t1 = fake`.
  - 공개 함수: `text_prob(logits) -> [B,1,H/4,W/4]`, `fake_prob(logits) -> [B,1,H/4,W/4]`, `pseudo_text_penalty(logits) = mean(sigmoid(ch0)·sigmoid(ch1))`(이미지당 스칼라).
  - 헤드는 1×1 conv라 나중에 오케스트레이터가 1채널만 원하면 ch0만 쓰면 된다. 이 결정은 메타(`config.head="glyph+fake"`)와 DONE.json에 기록한다.
- 해상도: H, W는 4의 배수여야 한다(아니면 `ValueError`). 내부적으로 32의 배수로 reflect 패딩하고 출력을 `H/4 × W/4`로 잘라 돌려준다. 학습 크롭은 384²(= 12×32)다.
- 선택 CTC 인식기(2.8M)는 이 스파이크에 넣지 않는다.

## 2. 데이터 규칙 (DATA-PLAN 준수)

- 라벨은 **전부 우리 렌더러의 잉크 알파**에서 나온다. OCR, 신경망 마스크, 신경망 필터는 쓰지 않는다(R2).
- 폰트: google/fonts 저장소의 SIL OFL 1.1 폰트만 쓴다. 상업 학습 허용(R1). 라이선스는 `fonts_manifest.json`과 `licenses/<family>/OFL.txt`로 기록해 체크포인트와 함께 올린다. OFL은 출력물 표기 의무가 없고 CC-BY를 쓰지 않으므로 `ATTRIBUTION.md`는 생기지 않는다.
- 교사 코퍼스(night1, Z-Image-Turbo 출력 4,755장)는 **배경으로만** 쓴다(PARADIGM 행의 "교사 운송"). 라벨은 그 위에 우리가 그린 잉크뿐이다. **indoor 범주(서울 식당·카페·지하철, 약 30%)는 학습 배경에서 뺀다.** 교사가 그린 가짜 간판이 섞여 있어 0으로 라벨하면 정반대를 가르치기 때문이다. indoor는 정성 평가에만 쓴다.
- 창업자 야크 이미지(`demo/d15_hero_yak.webp`)와 경쟁 모델 출력(`comp/ltx25/` 등)은 **평가 전용**이다. 학습 샘플 생성 경로에서 절대 읽지 않는다(테스트로 확인, §12 T15).
- 컨테이너의 DejaVuSans, unifont, wqy-zenhei는 **테스트 픽스처 전용**이다. 학습 폰트 목록(`FONTS`)에 들어가면 안 된다(unifont는 GPL/OFL 이중, wqy는 GPL+예외).
- FND-04 방화벽 차이: `buc/gov/prov.py`의 `TRAIN_LICENSES`에 `OFL-1.1`이 없다. 스파이크는 방화벽을 호출하지 않고 매니페스트에 기록만 한다. DAT-12/MDL-20 정식화 전에 오케스트레이터가 `TRAIN_LICENSES`에 OFL-1.1 추가를 결정해야 한다(DONE.json `notes`에 남긴다).
- 비밀: `HF_TOKEN`은 환경변수에서만 읽는다. 출력·파일·argv·매니페스트 어디에도 쓰지 않는다. 없으면 업로드를 생략하고 `hf_token:"no"`만 기록한다.

## 3. 폰트 (57파일, 52가문, 170.5 MB, 모두 OFL-1.1, 검증 완료)

- URL: `https://raw.githubusercontent.com/google/fonts/bd8f81ddb5c74d5c8897b36ad88b440266245103/ofl/<dir>/<file>` (커밋 고정, `[`/`]`는 `%5B`/`%5D`). 실패하면 `https://cdn.jsdelivr.net/gh/google/fonts@bd8f81ddb5c74d5c8897b36ad88b440266245103/ofl/<dir>/<file>`로 간다. 라이선스는 같은 경로의 `<dir>/OFL.txt`, 저작권 줄은 `<dir>/METADATA.pb`의 `copyright` 필드다.
- 다운로드: urllib, 스레드 8개, 파일당 타임아웃 60초, 3회 재시도. 저장 위치는 `<out>/fonts/`(`.part` → `os.replace`). sha256을 `FONTS` 표와 비교하고, 다르면 경고만 하고 실제 해시를 매니페스트에 쓴다. 이미 있고 해시가 맞으면 다시 받지 않는다. 디스크 여유가 1 GB 미만이면 시작하지 않는다(`fatal`).
- `FONTS` 상수: 항목마다 `dir, file, cls ∈ {full, ksx, latin}, hangul_n, axes=[(tag,min,default,max)…], size, sha256`. 값은 정찰 때 만든 매니페스트(`…/scratchpad/fonts_manifest.json`, 57항목)에서 그대로 옮긴다. fontTools는 박스에 없으므로 커버리지는 이 표를 믿는다.
- 분류:
  - **full(11,172 음절 전부, 21파일)**: NotoSansKR[wght], NotoSerifKR[wght], NanumGothic R/B/EB, NanumMyeongjo R/B/EB, NanumGothicCoding, NanumPenScript, NanumBrushScript, GothicA1 R/B, GowunBatang, GowunDodum, IBMPlexSansKR, PoorStory, SingleDay, HiMelody, Stylish, BlackAndWhitePicture.
  - **ksx(KS X 1001 2,350 음절만 쓴다, 17파일)**: BlackHanSans, DoHyeon, Jua, Sunflower-Medium, Gaegu, Dokdo, EastSeaDokdo, Gugi, SongMyung, CuteFont, YeonSung, Hahmlet[wght], BagelFatOne, GasoekOne, Diphylleia, Orbit, KirangHaerang. 일부는 2,350자보다 많이 갖고 있지만 추가분 목록이 없으므로 2,350자 집합만 쓴다(17개 모두 2,350자 전부 보유 확인). 2,350자 집합은 `euc_kr` 코덱으로 오프라인 도출한다(검증: 정확히 2,350, 한자 4,888).
  - **latin(19파일)**: NotoSans, NotoSerif, NotoSansMono, Roboto, OpenSans, Lato, Montserrat, PlayfairDisplay, Oswald, BebasNeue, Pacifico, Lobster, Caveat, RobotoMono, Anton, Merriweather, Inter, DancingScript, SourceCodePro. 한글 폰트도 라틴을 갖고 있으므로 라틴 블록에도 쓴다.
- **검증용 보류 폰트(학습에서 완전 제외)**: NanumPenScript(full, 손글씨), GowunBatang(full, 명조), BlackHanSans(ksx, 굵은 장식), Montserrat(latin). 학습 폰트는 나머지 53개.
- 가변 폰트 함정: NotoSansKR 기본값은 wght 100(Thin), NotoSerifKR는 200이다. 반드시 `font.set_variation_by_axes([...])`로 **fvar 순서대로 모든 축 값**을 넣는다(wght만 바꾸고 나머지는 default). 굵기는 {300, 400, 500, 700, 900} 중 하나를 고르고 축 범위로 자른다. 이 호출은 객체를 바꾸므로 `(path, px, wght)`마다 FreeTypeFont 객체 하나를 워커별 LRU(최대 512)에 캐시한다. `getlength` 결과도 `(font, px, char)`로 캐시한다(측정 시간의 8%).
- 한자: KS X 1001 한자 4,888자 중 NotoSansKR/NotoSerifKR에 실제 있는 글자만 쓴다. 준비 단계에서 PIL로 탐침한다(`font.getmask(ch)`의 바이트·크기 해시를 U+E000·U+10FFFD의 .notdef 해시와 비교, fontTools 결과와 일치 확인됨). 결과는 `<out>/fonts/coverage.json`에 캐시한다. 한글 폰트 전체를 탐침하지는 않는다(표를 믿는다).
- 두부(tofu) 방지: 폰트가 갖지 않은 글자를 진짜 텍스트로 라벨하면 거짓 양성이다. ksx 폰트는 2,350자 밖 음절을, latin 폰트는 한글·한자를 절대 받지 않는다(테스트 T5).
- `ImageFont.Layout.BASIC`을 쓴다(완성형 한글, 셰이핑 불필요, 빠름).

## 4. 합성 데이터 (CPU, 결정론)

모든 샘플은 `(seed, split_id, global_idx)`의 순수 함수다. 샘플마다 `rng = np.random.default_rng([seed, split_id, global_idx])`. 샘플러 상태가 없으니 재개하면 같은 스트림이 이어진다. split_id: 0 train, 1 val_id, 2 val_font, 3 val_tex, 4 val_fake. 평가 집합은 global_idx를 0부터 쓰지만 split_id가 달라 학습과 겹치지 않는다.

### 4.1 배경 (S = 384)
| 비율 | 원천 | 내용 |
|---|---|---|
| 45% | 교사 뱅크 | landscape·fur/hair·material·surreal 범주(약 3,300장)에서 `i % 10 != 0`인 것 약 2,970장. 무작위 512² 크롭(짧은 변의 0.5–1.0 배 크기를 512로 리사이즈) 한 장을 JPEG q90 바이트로 만들어 `<out>/bank/bank_train.bin` + `offsets.npy`(약 0.3 GB)에 저장한다. 워커는 `np.memmap`으로 공유해 읽고, 디코드(약 3 ms)한 뒤 무작위 384² 크롭·좌우 반전 없음·밝기/색 지터를 한다. `i % 10 == 0`(약 330장)은 평가 전용 뱅크 `bank_eval.bin`이다 |
| 35% | 절차 | fBm(3–6옥타브, 작은 격자 cv2.resize 합), 2색 LUT 팔레트(70%는 채도 낮춤), 그라디언트, 보케 원판 |
| 20% | 구조 질감 전면 | §4.4의 질감 생성기(털·풀·벽돌·잎·나뭇결·울타리/블라인드·타일)를 화면 전체에 깐다 |

- 뱅크 만들기: `$N1_ROOT/img`, `$N1_ROOT/meta`가 있으면 그대로 쓴다. 없고 `--no-fetch`가 아니면 HF 금고 `corpus/night1/gen_*.tar`에서 **img/와 meta/만** 푼다(m2의 `extract_shard`, `make_hf_fetchers` 복사, lat/는 건너뜀). 600초 안에 뱅크가 안 되면 교사 비율을 0으로 바꾸고(절차 70%, 질감 30%) `N2_M14_CACHE {"bank":0}`을 남기고 계속한다. 이 경우 config의 `bank` 필드가 달라지므로 재개 비교에도 들어간다.
- DAT-01 `render_room`은 쓰지 않는다. 장당 0.16–0.31초로 느리고, 무늬 없는 상자라 배경 사전으로 약하다.

### 4.2 진짜 텍스트 블록
- 이미지당 블록 수: 0개 8%, 1개 30%, 2개 32%, 3개 20%, 4개 10%.
- 문자 체계: ko 55%, latin 25%, ko+latin 혼합 12%, 한자(Noto KR만) 5%, 숫자·기호 3%.
- 한글 음절 선택: full 폰트는 40% 확률로 11,172자 균등, 60%는 KS X 1001 2,350자 또는 잡 안의 한국어 단어 목록(간판·표지 어휘 약 300개, 손으로 쓴 상수 `KO_WORDS`). ksx 폰트는 2,350자에서만 고른다. 라틴은 단어 목록 + 무작위 의사단어 + 숫자 + 구두점.
- 11,172자 커버리지: 워커가 청크마다 사용한 음절 비트셋(1.4 KB)을 돌려주고 메인이 OR 한다. 진행 마커에 `hs`(본 음절 수)를 넣는다. 예상: 중앙값 시나리오에서 균등 추첨 약 300만 회, 음절당 평균 약 270회, 첫 2만 스텝 안에 11,172자 전부.
- 글자 크기 em: 7–120 px 로그 균등(S/3.2). 줄 1–3개, 줄 간격 1.15–1.5, 자간 조절 30%.
- 색: 블록 주변 평균 휘도와의 대비가 45 이상인 경우 88%, 낮은 대비 12%. 효과: 판(패널) 40%, 외곽선 22%, 그림자 20%, 글로우 8%.
- 기하: 블록 캔버스(uint8 4채널: real, fake, panel, fx)를 **한 번의** `cv2.warpPerspective`로 변형한다. 회전 N(0, 4°), 15%는 ±35°까지, 3%는 세로쓰기. 모서리 지터는 블록 크기의 12% 이하.
- 겹침 금지: 변형된 블록의 잉크(알파 > 20)가 기존 잉크와 겹치면 그 블록을 버리고 다시 뽑는다(최대 6회).
- 합성은 sRGB에서 알파 블렌딩한다(단순화, 기록만 해 둔다).

### 4.3 가짜 글리프 (우리 렌더러로 만드는 음성 → ch1 양성)
- 블록이 가짜일 확률 0.35. 가짜 블록의 60%는 전부 가짜, 40%는 글자 칸의 15–60%만 가짜다. 그래서 한 줄 안에 진짜와 가짜가 섞인 글자 단위 라벨이 생긴다.
- 종류와 비율:
  | 종류 | 비율 | 방법 |
  |---|---|---|
  | `shuffle` | 0.18 | 글자 잉크의 연결 성분(자모 묶음)을 `cv2.connectedComponents`로 나눠 이동·회전·크기 변경 |
  | `patch` | 0.14 | 64–128 px로 그린 글자를 2×2 또는 3×3 칸으로 잘라 순서 섞기, 뒤집기, 기울이기 |
  | `melt` | 0.14 | 부드러운 무작위 변위장(em의 7–16%)으로 `cv2.remap`, 선택적으로 침식·팽창 |
  | `chimera` | 0.14 | 서로 다른 두 글자 겹치기. 1/3은 한자 A의 왼쪽 절반 + 한자 B의 오른쪽 절반(Noto KR, PARADIGM 378행의 "가짜 한자") |
  | `jamo` | 0.14 | 음절을 산술 분해(`s=c-0xAC00, L=s//588, V=(s%588)//28, T=s%28`)한 뒤 같은 폰트의 호환 자모(U+3131–3163) 2–3개를 틀린 배치·회전·크기로 한 칸에 그림 |
  | `strokes` | 0.12 | 폰트 획 두께에 맞춘 LINE_AA 선·호·원 3–8개 |
  | `mirror` | 0.07 | 좌우 또는 상하 반전. 원본과 반전본의 IoU가 0.8 이상이면(대칭 글자) 버림 |
  | `latin_pseudo` | 0.07 | 라틴 글자에 `patch`/`melt`/`shuffle` 적용 |
- 가짜 판정 규칙: 결과 잉크와 원래 글자 잉크의 IoU가 0.6 이상이면 "아직 읽히는 글자"로 보고 다시 만든다(최대 4회, 실패하면 그 칸은 `strokes`). `jamo`는 우연히 맞는 음절이 될 수 있으니 같은 자모로 조합한 진짜 음절과도 IoU를 비교한다.
- 학습 이미지에는 수평 반전 증강을 하지 않는다(반전 글자 자체가 가짜 종류다).

### 4.4 하드 음성: 글자 비슷한 질감 (라벨 0, 가중치 2)
- 이미지의 50%에 1–3개 조각(블록 크기 32–192 px), 그리고 §4.1의 전면 질감 20%.
- 생성기(전부 numpy/cv2, 각각 3–8 ms 목표):
  - `fur`: 흐름장(fBm 각도)을 따라 짧은 곡선 획 2,000–6,000개, 획 두께 1–2 px, 털빛 팔레트.
  - `grass`: 아래가 넓고 위가 뾰족한 세로 잎 300–1,200개, 기울기 ±25°, 초록·마른 풀 팔레트.
  - `brick`: 엇갈림 쌓기 격자, 줄눈 2–6 px, 벽돌마다 색 지터와 잡음, 원근 변형.
  - `foliage`: 회전된 타원·잎 모양 다각형 200–800개, 겹침, 명암.
  - `wood`: 사인 + fBm 나뭇결. `fence`/`blinds`: 평행 막대. `tiles`/`window_grid`: 격자. `barcode`: 무작위 폭 세로 막대. `hatching`, `dashed`, `scribble`(무작위 폴리라인), `greeked`(회색 막대로 된 가짜 문단 모양).
- 색은 주변 평균 휘도 근처로 둔다(정찰 프로토타입의 무작위 원색은 교사 영역과 너무 달랐다).
- 하드 음성 잉크 영역(3×3 팽창)은 손실 가중치 2.0을 받는다.

### 4.5 전역 열화 (m6 왜곡 함수 복사 + 추가)
다운·업 리샘플(0.4–0.95, linear/cubic) 50%, 가우시안 블러(σ 0.4–2.0 px) 55%, 모션 블러 12%, 감마·대비·밝기 LUT, 화이트밸런스 이득, 비네팅, 잡음(미리 만든 16타일 뱅크를 무작위 롤), 색수차 이동 18%, JPEG q25–95 60%(`cv2.imencode`). 열화는 이미지에만 적용하고 라벨은 변형 전 잉크에서 만든다(블러로 퍼진 잉크를 라벨이 쫓지 않는다).

### 4.6 라벨 (H/4)
- real, fake 각각 전해상도 uint8 잉크 맵(블록별로 같은 호모그래피로 변형, max 누적).
- `area = cv2.resize(ink/255, (S/4, S/4), interpolation=cv2.INTER_AREA)` → `t = clip(4 · GaussianBlur(area, σ=1.0), 0, 1)`. 소프트 영역 목표다(정찰 시트로 시각 확인한 레시피). `t0 = max(t_real, t_fake)`, `t1 = t_fake`.
- 가중치 맵 `w`(H/4): 기본 1. em < 9 px 블록 영역(팽창 1칸)은 0(읽기 어려운 미세 글자는 무시). 하드 음성 잉크 영역은 2.
- 반환 형식(워커 → 메인, 청크 = 샘플 16개): `img uint8 [16,3,384,384]`, `tgt uint8 [16,2,96,96]`(×255), `w uint8 [16,96,96]`(×100), `meta`(블록 목록: 4각 좌표 H/4, em, script, kind, real/fake/hardneg), `seen` 비트셋. 정규화와 channels_last 변환은 GPU에서 한다.
- 평가 지표의 이진 정답은 `t ≥ 0.5`다. 이 선택(소프트 BCE 목표 + 0.5 이진화)은 config에 `label="area4_blur1_x4"`로 기록한다.

### 4.7 CPU 워커
- **워커 수 = min(8, len(os.sched_getaffinity(0)) − 2, --workers)**, 기본 8. `multiprocessing.get_context("forkserver").Pool`을 **CUDA를 만지기 전에** 만든다. 워커 초기화: `cv2.setNumThreads(1)`, `OMP_NUM_THREADS=1`, `os.nice(5)`, torch import 금지.
- DataLoader 공유메모리를 쓰지 않는다(Docker의 /dev/shm 64 MB 위험). 결과는 파이프로 numpy를 돌려받는다(청크 7.5 MB, 300 img/s면 약 140 MB/s).
- `pool.imap`으로 청크 번호 순서대로 받고, 미리 받기 깊이는 2×워커 청크(약 120 MB)다. 학습 배치 b = 청크 4b…4b+3.
- 측정(정찰 샌드박스, 4 vCPU): 384²에서 프로세스 1개 21.4 ms/샘플(46.8 img/s), 4개 152.6 img/s(효율 81%). 5090 박스에서 8워커 예상 **200–400 img/s, 중앙값 280**. 처음 200스텝에서 `ips_cpu`를 실측해 마커로 낸다.

## 5. 모델: U-Net-lite (4,028,354 파라미터)

```
입력 x: uint8 → (x/255 − 0.5)/0.5, [B,3,H,W], H,W는 4의 배수(내부 32배수 reflect 패딩)
stem   Conv3x3 s2 3→24                         H/2
e0     ResBlock(24)                             H/2
d1     Conv3x3 s2 24→48,  e1 ResBlock(48)×2     H/4
d2     Conv3x3 s2 48→96,  e2 ResBlock(96)×2     H/8
d3     Conv3x3 s2 96→144, e3 ResBlock(144)×2    H/16
d4     Conv3x3 s2 144→192,e4 ResBlock(192)×2    H/32   (수용영역: 큰 글자 em 120 px 대응)
u3     nearest↑ + concat(e3) → ResBlock(336→144)  H/16
u2     nearest↑ + concat(e2) → ResBlock(240→96)   H/8
u1     nearest↑ + concat(e1) → ResBlock(144→48)   H/4
head   GN(8)+SiLU → Conv3x3 48→48 → SiLU → Conv1x1 48→2   → 로짓 [B,2,H/4,W/4]
ResBlock(ci,co) = x·skip(1x1 if ci≠co) + Conv3x3(SiLU(GN8(Conv3x3(SiLU(GN8(x))))))
```
- 너비 상수 `WIDTHS=(24,48,96,144,192)`. 시작할 때 파라미터 수를 세어 `[3.2M, 4.8M]` 창 안인지 확인하고 시작 마커에 `params`, `in_range`를 넣는다. 창 밖이면 `fatal`.
- 헤드 마지막 1×1의 bias는 ch0 −2.5, ch1 −3.5로 초기화한다(양성 비율 약 8%, 2%).
- 계산량(CPU 훅 측정): 384²에서 전진 약 10.0 GFLOPs/장 → 학습 약 30 GFLOPs/장. 저장 활성 약 63 MB/장(CPU bf16 autocast 측정, CUDA에서는 GroupNorm이 fp32라 약 90 MB로 본다).
- 1024² 추론은 전진 약 71 GFLOPs라 PARADIGM 목표 30 ms 안에 들 것으로 본다. 마지막 평가에서 1024² 배치 1의 지연을 재서 DONE.json에 `lat_1024_ms`로 넣는다.

## 6. 손실과 최적화

- `L = BCE(ch0, t0; w) + 0.5·Dice(ch0, t0; w) + BCE(ch1, t1; w) + 0.5·Dice(ch1, t1; w)`
  - BCE는 소프트 목표, 픽셀 가중치 `w`, 합 / Σw.
  - Dice는 마이크로배치 단위 소프트 Dice(`1 − (2Σwpt + 1)/(Σwp + Σwt + 1)`). ch1은 배치의 Σt1 < 1이면 Dice 항을 뺀다.
- AdamW lr 2e-3, betas (0.9, 0.99), weight decay 0.05(GroupNorm·bias 제외), 그래디언트 클립 1.0.
- **진척률 기반 스케줄**: `p = max(step/total_steps, train_s/train_budget_s)`. 처음 600스텝 선형 워밍업, 그다음 코사인으로 `lr·(0.02 + 0.98·0.5·(1+cos(πp)))`. `train_s`는 학습 루프 누적 시간(ckpt_last에 저장, 재시작마다 합산)이라 재개해도 스케줄이 이어지고, CPU가 느려도 예산 끝에 lr이 바닥에 닿는다.
- EMA 0.999(평가·best·업로드는 EMA 가중치). bf16 autocast, channels_last, TF32 켬, `torch.compile` 끔(시작 시간과 위험).
- 배치 64 = 마이크로배치 32 × 누적 2. **배치 재사용 R = 2**: 스텝 s는 고유 배치 `s // 2`를 쓰고, 두 번째 사용(`s` 홀수)에는 GPU에서 증강한다. 증강: 4 정렬 무작위 크롭 352²(라벨도 같은 위치 88²), 밝기 ±10%, 대비 ±15%, 채널 이득 ±5%, 잡음 σ ≤ 3/255. 시드 `(seed, s)`로 결정론. 고유 샘플은 CPU가 상한이고 GPU는 2배 빨리 돌 수 있으니 이렇게 GPU를 더 쓴다.

## 7. 예산, 스텝, 시간

| 구간 | 상한 |
|---|---|
| 준비: 폰트 다운로드(170 MB) + 한자 탐침 + 뱅크(약 2,970장 디코드·크롭, 8프로세스) + 검증 집합 생성(3,840장) + 풀 예열 | 420 s(뱅크는 600 s에서 포기) |
| 학습 루프 | `--train-budget-s 4320` (1.2 GPU-h) |
| 평가: 학습 300 s에 첫 평가, 이후 900 s마다(300, 1200, 2100, 3000, 3900) + 마지막 | 회당 45 s 이하, 합 약 270 s |
| 마무리: 저장 + 최종 평가(정성 시트 포함) + 업로드 join | 300 s |
| **합계(벽시계, 재시작 포함 누적)** | **`--budget-s 5400`** |

- 누적 벽시계 `elapsed_total_s`(준비·평가·업로드 포함)를 ckpt_last에 저장하고 재시작마다 합산한다. `elapsed_total_s ≥ budget_s − 300`이거나 `train_s ≥ train_budget_s`이면 학습을 멈추고 마무리로 간다(reason `budget`).
- `--total-steps 48000`은 상한이다(낙관 시나리오). 보통은 예산이 먼저 닿는다.
- 시나리오(배치 64, R = 2, GPU가 CPU의 2배 이상을 소화한다고 가정; 384² 학습 30 GFLOPs/장이면 GPU 실효 20–30 TFLOPs에서 650–1,000 img/s):
  | CPU img/s | 고유 샘플 (4,320 s) | 샘플-뷰 | 스텝 |
  |---|---|---|---|
  | 200 (비관, 다른 잡과 경쟁) | 0.86M | 1.73M | 27k |
  | 280 (중앙) | 1.21M | 2.42M | 37.8k |
  | 400 (낙관) | 1.73M | 3.07M(상한 도달) | 48k |
- PARADIGM 행 목표(200만 샘플, 1.2×10¹⁶ FLOPs)와 비교: 중앙 시나리오는 FLOPs로 약 7×10¹⁶(384² 기준)으로 넘지만, 고유 샘플은 약 1.2M으로 200만에 못 미친다. 진행 보고에 이 차이를 그대로 적는다.
- 마감: `N2_DEADLINE`(없으면 `N1_DEADLINE`) = **1791759600**(2026-10-12 08:00 KST = 23:00Z). `now ≥ deadline − finish_margin_s(600)`이면 멈추고 마무리. 예산을 다 쓰려면 늦어도 **06:20 KST(1791753600)** 에 시작해야 한다. 권장 시작은 01:00 KST 이전.
- 정지 사유 우선순위: `signal > deadline > budget > max_steps > total_steps`.

## 8. 평가 (EMA 가중치, 고정 집합, 메모리에 보관)

검증 집합은 준비 단계에서 한 번 만들어 RAM에 둔다(합 3,840장, 약 1.8 GB). 디스크에 캐시하지 않는다.

| 집합 | split_id | 장수 | 내용 |
|---|---|---|---|
| `val_id` | 1 | 1,536 | 학습 폰트, 배경은 평가 뱅크(`i%10==0`) + split 1 절차 시드 |
| `val_font` | 2 | 768 | **보류 폰트 4종만**(NanumPenScript, GowunBatang, BlackHanSans, Montserrat), 배경도 평가 시드 |
| `val_tex` | 3 | 768 | 텍스트 없음: 질감 6군(fur, grass, brick, foliage, wood/fence/blinds, barcode/grid/hatching) 각 96장 + 평가 뱅크 교사 크롭(hair/material/landscape) 192장 |
| `val_fake` | 4 | 768 | 모든 이미지에 진짜 블록 ≥ 1 + 가짜 블록 ≥ 1, 가짜 종류 8종 균등 |

지표(H/4 해상도, 마스크 `w > 0`):
1. **픽셀 AP / F1** (ch0, 이진 정답 `t0 ≥ 0.5`): 1,000칸 히스토그램으로 GPU에서 AP, F1@0.5, 최대 F1과 그 임계값. `val_id`, `val_font` 각각. 같은 방식으로 ch1의 AP도(`val_fake`).
2. **영역 재현율**: 블록 메타의 4각형(H/4)마다 정답 픽셀의 50% 이상이 `sigmoid(ch0) ≥ 0.5`이면 검출. em 구간 9–16, 16–32, 32+ px와 script(ko, latin, mix, hanja)별 재현율.
3. **질감 FP율**(`val_tex`): 픽셀 FPR(`p0 ≥ 0.5` 비율), 이미지 FP율(3×3 연결 성분 중 넓이 ≥ 4칸인 것이 하나라도 있으면 FP), 평균·최대 확률. 질감 종류별로.
4. **가짜 대 진짜 분리**(`val_fake`): 글자 칸 영역마다 평균 `sigmoid(ch1)`. 가짜 칸 대 진짜 칸 AUROC(전체 + 종류 8종별), 이미지 단위로 `pseudo_text_penalty`의 가짜 있음/없음(`val_id`의 가짜 없는 이미지) AUROC.
5. **best 점수** = `0.5·AP0(val_font) + 0.3·AUROC_fake + 0.2·(1 − tex_img_fp)`. 오르면 ckpt_best.pt(EMA 가중치 + config + 지표)를 원자적으로 쓴다.
6. **정성(마지막 평가에서만, 평가 전용 원천)**: indoor 교사 이미지 64장(결정론적 선택)을 1024² 그대로 추론해 `pseudo_text_penalty` 상위 16장의 시트 `previews/teacher_indoor_topk.jpg`. HF 금고에 `demo/d15_hero_yak.webp`, `comp/ltx25/`가 있으면 받아서 추론만 하고 `previews/evalonly_yak.jpg`, `previews/evalonly_comp.jpg`와 발화율(픽셀 `p0 ≥ 0.5` 비율, `p1` 평균)을 DONE.json에 넣는다. 이 파일들은 `<out>/evalonly/`에만 두고 생성기 경로에서 읽지 않는다. 없으면 건너뛴다(경고만).

## 9. 프리뷰 이미지

- 평가마다 `previews/val_step{S:06d}.jpg`: 4×4 타일(`val_id` 8장 + `val_fake` 8장, 결정론 고정), 타일마다 [원본 | 예측 오버레이 | 정답 오버레이]를 가로로 256 px씩. 오버레이: ch0 ≥ 0.5 초록 반투명, ch1 확률 빨강 열지도, 블록 4각형 윤곽(진짜 흰색, 가짜 빨강, 미세 글자 회색).
- `previews/tex_fp_step{S:06d}.jpg`: `val_tex`에서 최대 확률 상위 16장.
- JPEG q85, 장당 1.5 MB 이하. 업로드에는 가장 최근 것과 첫 평가 것만 넣는다.

## 10. 체크포인트, 재개, DONE, 업로드

- 출력 폴더 `--out` 기본 `<root>/night2/m14` (`root` 기본 `$N1_ROOT=/workspace/night1`).
- 시작 시 `clean_tmp`로 `*.tmp`, `*.part` 삭제. 디스크 여유 3 GB 미만이면 경고, 1 GB 미만이면 `fatal`. 이 잡의 디스크 사용 상한은 약 1 GB(폰트 0.17 + 뱅크 0.35 + ckpt 0.1 + 프리뷰)다.
- `ckpt_last.pt`(원자 저장: tmp → fsync → `os.replace`): `step, config, model, ema, opt, best, elapsed_total_s, train_s, micro_batch, hangul_seen(bitset), saved_at`. 평가 직전과 직후, 그리고 마무리에서 저장한다. 약 65 MB.
- `config`(재개·DONE 비교 대상): `widths, head, crop, batch, reuse, seed, gen_version, label, font_set_sha(학습 폰트 sha256 목록의 해시), heldout_fonts, bank(교사 뱅크 사용 여부)`. 다르면 경고 후 새로 시작(이전 ckpt_last는 `ckpt_last.prev.pt`로 이름 바꿔 보존).
- `load_ckpt`는 `weights_only=True`를 먼저 시도하고 실패하면 대체 경로로 간다(m2와 같다).
- **DONE.json**(원자 저장): reason이 `total_steps`, `max_steps`, `budget`일 때만 쓴다. `deadline`, `signal`에서는 쓰지 않는다(다음 밤에 ckpt_last에서 이어 간다). 내용: `status, reason, step, unique_samples, elapsed_total_s, train_s, best 지표, lat_1024_ms, head, notes`(OFL-1.1 방화벽 추가 요청 포함).
- `_already_done()`: DONE.json + ckpt_last step ≥ 목표 또는 reason `budget` + config 일치이면 `N2_M14 {"status":"already_done"}`과 `N2_M14_DONE`을 찍고 아무것도 불러오지 않고 rc 0으로 끝난다. 그래서 keep-alive 루프가 재실행해도 예산을 더 쓰지 않는다.
- 실패 처리(m2 복사): 스텝 실패는 `warn(step_failed)` 후 다음으로, 연속 100회면 `status:"fatal"`, rc 2. OOM(cuDNN/cuBLAS 할당 실패 포함, `is_oom`)이면 마이크로배치를 반으로(32 → 16 → 8, 배치 64 구성은 그대로, 누적 횟수만 늘어남). 비유한 손실·그래디언트는 그 스텝을 건너뛰고, 연속 25회면 ckpt_last에서 복원. 평가 실패는 경고만 하고 체크포인트를 잃지 않는다. 워커 풀이 죽으면 한 번 다시 만들고, 또 죽으면 `fatal`.
- `"fatal"` 문자열은 진짜 치명 마커에만 쓴다(러너가 이 문자열로 판정한다).
- **업로드**: `<out>/upload_stage/`에 복사(ckpt_last.pt, ckpt_best.pt, metrics.jsonl, previews 최신, fonts_manifest.json, data_manifest.json, config.json, licenses/, DONE.json 있으면) 후 `HfApi(token).upload_folder(folder_path=…, path_in_repo="ckpt/night2/m14", repo_id="sdsdsdsdsdsdsd123/buc-artifacts", repo_type="model")`를 백그라운드 스레드로. 평가마다 한 번, 이전 업로드가 아직 돌면 건너뜀. 3회 시도(5 s, 15 s 대기), 실패는 `warn(upload_failed)`만. 마지막 업로드는 최대 240 s join. `--no-upload` 또는 토큰 없음이면 생략.
- `fonts_manifest.json`: 폰트마다 `family, file, url(고정 커밋), mirror_used, sha256(실제), expected_sha256, cls, hangul_n, axes, license:"OFL-1.1", license_url, copyright(METADATA.pb), ofl_txt_sha256, heldout(bool)`. `licenses/<family>/OFL.txt` 원문 52개(약 230 KB)를 같이 올린다.
- `data_manifest.json`: 규칙 진술(R1 OFL 상업 허용, R2 라벨 = 자체 렌더 잉크, 교사 = 배경 전용, indoor 제외, 평가 전용 원천 목록), 교사 인덱스 범위(train/eval/indoor 개수와 선택 규칙), split_id 정의, 보류 폰트, 생성기 버전·비율 표, 본 음절 수.

## 11. 마커와 러너 요구사항

- 접두어 `N2_M14`, `N2_M14_CACHE`, `N2_M14_EVAL`, `N2_M14_DONE`. 한 줄 JSON, `separators=(',',':')`, **400자 이하**(`fit_marker`가 긴 필드를 잘라냄), `print(..., flush=True)`. 모든 문자열은 `scrub()`을 거친다(토큰 값 그대로 + 정규식 `hf_[A-Za-z0-9]{8,}`).
- 시작: `N2_M14 {"start":…,"root":…,"hf_token":"yes|no","upload":…,"deadline":…,"budget_s":5400,"torch":…,"params":4028354,"in_range":true,"workers":8,"mem_gb":6,"head":"glyph+fake"}`.
- 준비: `N2_M14_CACHE {"fonts":57,"bytes":…,"mirror":…,"hanja":…,"bank":2970,"bank_eval":330,"val":3840,"s":…}`.
- 진행(240 s마다, 5분 이하 보장): `N2_M14 {"step","p","loss","l0","l1","lr","ips_cpu","ips_gpu","qwait","mb","mem","hs","el","eta"}`. `qwait`은 배치를 기다린 시간 비율(CPU 병목 판정), `mem`은 `max_memory_allocated` GB.
- 평가: `N2_M14_EVAL {"step","ap_id","f1_id","ap_font","rr16","rr9","tex_fp","tex_fpr","auc_fake","ap_fake","best"}`.
- 끝: `N2_M14_DONE {"status":"ok|already_done|fatal","reason":…,"step":…,"uniq":…,"el":…,"best":…}`. 정상 종료(마감 정지 포함)마다 반드시 찍는다.
- 러너(이 명세 범위 밖, night2 러너를 만들 때 필요): day1 `marks()` 필터는 `^(D1_|N1_)`만 통과시키므로 `^N2_`를 추가해야 한다. DONE 경로는 `$N1_ROOT/night2/m14/DONE.json`. 환경 `HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True N2_DEADLINE=1791759600`. 로그는 `logs/night2/`로 업로드. 공유 박스이므로 다른 잡이 돌면 `vastai stop`을 하지 않는다. 버퍼링 대비로 비콘 패딩(약 42줄)을 쓴다. 실행 예:
  `keep m14 python -u code/m14_glyph.py --root $N1_ROOT --mem-gb 6 --workers 8`

## 12. CLI

`--root`(env N1_ROOT, 기본 /workspace/night1) · `--out`(기본 `<root>/night2/m14`) · `--deadline`(env N2_DEADLINE → N1_DEADLINE) · `--finish-margin-s 600` · `--budget-s 5400` · `--train-budget-s 4320` · `--mem-gb 6.0` · `--workers 8`(8 초과는 8로 자름) · `--batch 64` · `--micro-batch 32` · `--reuse 2` · `--crop 384` · `--total-steps 48000` · `--max-steps`(없음) · `--lr 2e-3` · `--warmup 600` · `--first-eval-s 300` · `--eval-every-s 900` · `--log-every-s 240` · `--max-consec-errors 100` · `--seed 14` · `--no-upload` · `--no-fetch`.

- CUDA 메모리 상한: `PYTORCH_CUDA_ALLOC_CONF`를 CUDA 초기화 전에 설정하고 `torch.cuda.set_per_process_memory_fraction(6·2³⁰ / total_memory)`. 예상 최고치: 마이크로배치 32 × 약 90 MB + 작업공간 ≈ 4–4.5 GB. 평가는 `no_grad`, 배치 64.
- `main(argv, deps)`는 `now, sleep, uploader, stop_event, fonts(경로 목록 주입), fetch, pool_factory, device`를 주입받는다(테스트용). SIGTERM/SIGINT는 stop 이벤트를 세운다.

## 13. CPU 테스트 계획 (`test_m14_glyph.py`, pytest, 합계 60 s 이하, GPU·네트워크 없음)

픽스처: 임시 폴더, 주입 폰트 = `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`(latin) + `/usr/share/fonts/opentype/unifont/unifont.otf`(full 한글 흉내, **테스트 전용**). 파일이 없으면 `pytest.skip`. 작은 설정: S = 64 또는 96, 배치 2–4, 워커 0(인라인) 또는 2, 가짜 교사 뱅크 = 무작위 이미지 몇 장, `--no-upload` 또는 가짜 uploader.

| # | 확인 |
|---|---|
| T1 | 결정론: 같은 `(seed, split, idx)` 두 번 → 이미지·라벨·메타 바이트 동일. split만 바꾸면 다름. 인라인 생성 = 2워커 풀 생성 |
| T2 | 형상·범위: img uint8 [N,3,S,S], tgt [N,2,S/4,S/4] ∈ [0,255], w ∈ {0,100,200}. `t0 ≥ t1` 모든 픽셀 |
| T3 | 라벨 정렬: 빈 배경에 블록 하나를 알려진 호모그래피로 그려 잉크 무게중심(전해상도/4)과 라벨 무게중심 차이 ≤ 0.5칸. 잉크 없는 곳 라벨 0 |
| T4 | 하드 음성·질감 생성기 8종 이상: 출력 형상, 라벨 전부 0, 가중치 2 영역 존재. `val_tex` 샘플은 t0 합 = 0 |
| T5 | 커버리지 가드: ksx 폰트에 2,350자 밖 음절 요청 0회(1,000샘플), latin 폰트에 한글·한자 0회. `euc_kr` 도출 집합 크기 2,350 / 4,888. .notdef 탐침이 U+E000을 없음으로 판정 |
| T6 | 가짜 글리프 8종: 각각 형상, ch1에 양성, 원본과 IoU < 0.6, mirror 대칭 거부, 혼합 블록에서 글자 단위 real/fake 분리 |
| T7 | 보류 폰트: train split은 보류 폰트를 절대 안 고르고, `val_font`는 보류 폰트만 고름 |
| T8 | 모델: `count_params()`가 4,028,354이고 창 안. 입력 [2,3,96,96] → [2,2,24,24]. 4의 배수 아니면 ValueError. 100×100이 아닌 100×132 같은 32 비배수도 패딩으로 [*,25,33] |
| T9 | 한 스텝 학습(CPU, S=64): 손실 유한, 파라미터 변화, Dice/BCE 가중치 0 영역 무시 |
| T10 | 재개: `max_steps=2` 실행 후 `max_steps=4` 재실행 = 처음부터 4스텝 실행과 배치 해시열 동일. config 바꾸면 경고 + 새로 시작 + `ckpt_last.prev.pt` 생성 |
| T11 | 마감: 주입 `now`를 `deadline−600` 이후로 → 학습 멈춤, ckpt_last 저장, `N2_M14_DONE` 출력, **DONE.json 없음**, rc 0 |
| T12 | 예산: 주입 시계로 `elapsed_total_s`가 `budget_s−300`을 넘김 → reason `budget`, DONE.json 있음. 재실행하면 `already_done`, 학습 스텝 0회 |
| T13 | 마커: 모든 출력 줄이 400자 이하(거대한 페이로드 주입), JSON 파싱 가능, 접두어 `N2_M14` |
| T14 | 비밀: `HF_TOKEN=hf_fakeTOKEN123456`, 토큰이 든 예외를 던지는 가짜 uploader → stdout·stderr·out 폴더 모든 파일에 토큰 없음. 토큰 없으면 `hf_token:"no"`이고 uploader 호출 0회 |
| T15 | 평가 전용 격리: `evalonly/`에 파일을 두고 생성기 1,000샘플을 만들 때 그 경로를 연 적이 없음(`open` 감시 주입) |
| T16 | OOM 처리: 마이크로배치 > 16에서 OOM을 던지는 가짜 모델 → 16으로 줄고 진행. 비유한 손실 25회 주입 → ckpt_last 복원 |
| T17 | 지표: 완벽 예측 AP = 1, 상수 예측 AP = 양성 비율, AUROC 대칭성, 영역 재현율을 손으로 만든 블록 2개로 검산 |
| T18 | 업로드 스테이징: 파일 목록이 명세와 같고, uploader가 `ckpt/night2/m14`로 호출됨, fonts_manifest 항목에 license `OFL-1.1`과 sha256 존재 |
| T19 | 프리뷰: 오버레이 함수 출력 형상과 JPEG 저장, 1.5 MB 이하 |

## 14. 남은 결정 (오케스트레이터)
1. 2채널 헤드(glyph + fake) 채택 여부. 이 스파이크는 2채널로 학습하고 ch0이 PARADIGM 1채널 인터페이스를 그대로 만족한다.
2. `buc/gov/prov.py` `TRAIN_LICENSES`에 `OFL-1.1` 추가(DAT-12/MDL-20 정식화 전).
3. indoor 교사 이미지를 계속 정성 평가 전용으로 둘지. 실제 교사 가짜 글자의 정량 평가를 하려면 사람 라벨(신경망 아님) 소량이 필요하다.
