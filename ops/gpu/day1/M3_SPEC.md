# M3 MRH 스파이크 GPU 잡 명세 (D1, 2026-10-11) — `ops/gpu/day1/m3_mrh.py`

창업자 질문 "교수님 그림 그릴 때 뇌 도촬해서 깊이 맞추는 건 언제 학습하냐"에 대한 답: **오늘 학습한다.** GPU가 05:00Z쯤 비면 교사 특징 추출이 약 1.5시간 걸리고, **07:00Z 전후에 첫 평가 숫자와 야크 깊이 미리보기**가 나온다. 학습은 09:00Z 전후에 끝난다(6,000스텝). GPU가 늦게 비면 하는 데까지 하고 night2에서 이어 간다.

이 문서는 스파이크 레인이다(`ops/gpu/day1/SPEC.md` 공통 규칙을 그대로 따른다). 코드는 `ops/gpu/day1/m3_mrh.py`와 `ops/gpu/day1/test_m3_mrh.py` 두 파일에만 쓴다. `buc/`, `tests/`, `ops/harness/`, `runner.sh`, `extra.sh`는 고치지 않는다. `buc/`는 import해서 읽기만 한다. 아래 API 이름은 독자 4명이 코드에서 확인한 것이다. 빌더는 실제 시그니처를 다시 읽고 맞춘다.

---

## 0. 고정 결정 요약

| 항목 | 결정 |
|---|---|
| 교사 읽기 조건 | sigma **0.3**(`sampler.table.snap(0.3)`, 인덱스 7), 캡션 **`SURVEY_CAPTION`** 고정(`buc.read.mrh_call`, RD-07과 같음), 층 8/15/23, `token_offset=0`, 이미지당 순전파 1회(블록 23 뒤 조기 종료) |
| PCA-128 기저 | night1 코퍼스 600장(ENGINE) + RoomGen 렌더 400장(ENGINE). 이미지·층마다 토큰 96개, 층당 N=96,000. DIODE·야크·평가 이미지는 **넣지 않는다**. 한 번 맞추고 동결 |
| 실사 데이터 | DIODE train에서 선택 추출 **3,661장**(실외 2,958 = 81%, 실내 703), RGB는 원본 768×1024, 정답은 절반 해상도. DIODE val 771장은 평가 전용 |
| 합성 데이터 | RoomGen/DAT-01 장면 3,000개 × 시점 2 = **6,000장**, 512×384. 합성 dev 200장(시드 0–199). 6,000장 모두 realify(절반 sigma 0.5·2 NFE, 절반 sigma 0.3·1 NFE, 두 경우 모두 탭 sigma 0.3) |
| 깊이 표현 | **ln(미터 z-깊이)**. 손실 = 이동 불변 대역 로그 깊이(지수 a=1) + 0.15 × 전역 로그 스케일 항(SILog λ=0.85와 같다) |
| 모델 | 기본 `MRHConfig()`, 17,516,303 파라미터. 탭 입력은 성분별 std로 고정 정규화 |
| 학습 | 6,000스텝, AdamW 3e-4, warmup 300 + cosine, EMA 0.999, 넥만 bf16, VRAM 상한 10 GiB |
| 헤드라인 지표 | DIODE val, **중앙값 스케일 정렬** AbsRel·δ1(실내/실외 분리). 원시 미터와 스케일+이동 정렬 값도 같은 표에 적는다 |
| 기준선 | 상수 깊이, Depth-Anything-V2-Small(Apache-2.0, **평가 비교 숫자로만** 쓴다) |
| 마커 | `D1_M3`, `D1_M3_DATA`, `D1_M3_EVAL`, `D1_M3_DONE`(400자 이하) |
| 업로드 | `ckpt/day1/m3/`(ckpt·기저·지표·표), `ckpt/day1/m3/preview/`(야크·DIODE 미리보기), `corpus/day1/m3_taps/`, `corpus/day1/diode/` |

---

## 1. 데이터

### 1a. DIODE 실사 레인 (CPU + 네트워크, 시작 즉시)
- 원천: `https://diode-dataset.s3.amazonaws.com/{val,train}.tar.gz`(val 2,774,625,282 B, train 86,747,209,190 B). 압축 해제는 스트리밍으로만 한다(`tarfile.open(fileobj=BufferedReader(ResumableHTTP), mode="r|gz", bufsize=4<<20)`). tarball 전체는 디스크에 두지 않는다. 같은 TCP 끊김은 `Range` + `If-Match: <ETag>`로 같은 압축 오프셋에서 이어 받는다(412·416·403은 치명, 429·5xx는 재시도, 최대 8회, 백오프 min(60, 2·2^k) 초). 프로세스가 재시작되면 0바이트부터 다시 받는다. 이미 끝난 샘플(`<stem>.gt.npz`가 있음)은 일 없이 건너뛴다.
- 순서: **val 먼저**(약 2–5분), 그다음 train(28 MB/s면 약 52분, 링크가 빠르면 CPU 한계로 25분 안팎).
- 선택(`Selection.keep(info)`, 파일 이름만 보는 순수 함수, profile `spike`): pitch 인덱스 {1,2,3}(−10°/0°/+10°), 실외는 yaw stride 3, 실내는 yaw stride 6, 위상 `yaw_i % stride == int(scan) % stride`. 결과 train 3,661장(실외 2,958 = 81%, 실내 703). 179개 스캔을 모두 포함한다. 실외 비중이 큰 것은 초원 야크 데모 때문이다. val은 771장 전부(실내 325, 실외 446).
- **Dev(모델 선택용, train 안에서 장면 단위로 떼어 냄):** 장면 `scene_00004`(실내), `scene_00009`, `scene_00012`(실외). 약 190장. 학습에서 빼고 `ckpt_best` 선택에만 쓴다. 한 스캔 안의 crop은 83% 겹치므로 이미지 단위 무작위 분할은 금지한다.
- 저장(`$N1_ROOT/day1/m3/data/diode/{train,val}/{indoors|outdoor}/scene_xxxxx/scan_xxxxx/`):
  - `<stem>.png`: 원본 바이트 그대로(재인코딩 없음). 1024×768 RGB. RGB는 줄이지 않는다(토큰 격자 48×64).
  - train `<stem>.gt.npz`: `depth_m` float16 (384,512) = 2×2 칸의 유효 픽셀 기하평균(없으면 0), `vcnt` uint8 (384,512) = 유효 픽셀 수 0..4.
  - val `<stem>.gt.npz`: `depth_m` float16 (768,1024)(무효는 0), `valid` = `np.packbits(valid, axis=-1)` uint8 (768,128).
  - 유효 마스크 = `(mask > 0.5) & isfinite(depth) & (depth > 0)`. depth는 `(768,1024,1)`을 squeeze한다. mask dtype은 f32/f64가 섞여 있다. `np.load(..., allow_pickle=False)`만 쓴다. 출력 경로는 정규식 그룹(숫자와 고정 단어)으로만 만든다.
  - `.gt.npz`는 `.part` → fsync → `os.replace` 순서로 **마지막에** 쓴다. 이 파일이 완료 표지다.
  - `SOURCE.json`(url, etag, size, 관측 md5, Selection 파라미터와 `sel_hash`), `LICENSE_DIODE.txt`(MIT 원문 + 사이트 문장 + arXiv:1908.00463), `val/EVAL_ONLY.txt`.
- 디스크: train 약 5.4 GB, val 약 1.5 GB.
- 완료 표지: `stages/diode_val.done.json`, `stages/diode_train.done.json`. 스트림 끝까지 읽고 PNG 멤버 수가 맞을 때만 쓴다(train 25,458, val 771).
- 끝나면 부분집합을 약 1 GB tar로 묶어 `corpus/day1/diode/{train,val}_NNN.tar`로 올린다(shard 0에 LICENSE). night2가 87 GB를 다시 받지 않게 하기 위해서다. 우선순위는 가장 낮다.
- 정답 법선: D1에서는 쓰지 않는다(DIODE 쪽 법선 손실 가중치 0). 공식 법선 tar(135 GB, 좌표계가 다름)와 깊이에서 유도한 법선은 night2 이후 선택 사항이다.

### 1b. 합성 레인 (CPU, 시작 즉시)
- 생성: `buc.invent.grammar.room.generate_room(seed)` → `buc.data.synth_render.default_camera(prog, v, width=512, height=384, fov_x_deg=U(55,90) from rng([seed,v]))` → `render_gbuffer(prog, cam)`(spp=1). `generate_room`의 ValueError(시드의 약 0.37%)는 건너뛴다. `(inst >= 0).mean() < 0.02`인 시점은 버리고 시도는 최대 3K회까지 한다.
- 시드: train **100000–102999 × 2 시점 = 6,000장**. 합성 dev는 **0–199 × 1 시점 = 200장**(학습 금지). 키는 `r{seed:06d}v{k}`.
- 표시용 이미지: `rgb_display()`는 **쓰지 않는다**(시점의 31%가 하얗게 날아감). 시점마다 자동 노출 + Reinhard를 쓴다. `x = rgb·0.35/p70(lum)`, `linear_to_display(x/(1+x))`.
- 병렬: `mp.get_context("fork").Pool(6)`, `os.nice(10)`, OMP/BLAS 스레드 1, 워커에서 torch를 import하지 않는다. 예상 5–8장/초이므로 6,200장에 약 15–25분.
- 저장(`data/synth/{train,dev}/`): `<key>.npz`(gbuf float16 (12,384,512), sem int16) + `<key>.png` + `<key>.json`(seed, view, fov, program_hash). 약 0.15 MB/장, 합계 약 1 GB.
- realify(교사 단계 F2b): `RealifyItem`을 손으로 만든다. `caption=SURVEY_CAPTION`(항목마다 다른 캡션 금지), `labels={"gbuf": f16, "sem": i16}`. `R.realify(sampler, bus, items, curve=R.provisional_curve(sampler.table, R.max_octaves(384,512)), sigma=…, seed=…, steps=…, token_offset=0, tap_step=-1, ring=None)`. 키 해시 짝수는 sigma 0.5·steps 2, 홀수는 sigma 0.3·steps 1이다. 두 경우 모두 `tap_sigma`가 0.3이다. 평가 편향을 막으려고 tau는 기본값 0.05를 쓰고, keep=False 샘플은 학습에서 빠진다. 모드별 keep 비율과 lowband_err는 `D1_M3` 마커에 남긴다. 합성 dev 200장은 `tau=1e9`로 모두 남긴다.
- 실외 합성(열린 셸 + 태양)은 D1에서 하지 않는다. 실외 신호는 DIODE에서 온다.

### 1c. 출처(provenance) 등급과 방화벽

| 원천 | prov 태그 | `why_not_train` | 쓰임 |
|---|---|---|---|
| RoomGen 렌더(자체 코드, 분석적 레이캐스트) | `["ENGINE"]`, roots `roomgen:<seed>` | () → train_ok | 기저 맞춤, 학습, 합성 dev |
| night1 Z-Image 코퍼스(이미지, 라벨 아님) | `["ENGINE"]`, roots `night1:<i>` (meta에 ip 플래그가 있으면 제외) | () → train_ok | 기저 맞춤만 |
| DIODE train(MIT, FARO 레이저) | `["ASSET:MIT"]`, roots `diode:<scene>/<scan>` | 현재 **거부** (`ASSET:MIT`, TRAIN_LICENSES에 MIT 없음) | 아래 FND-04 규칙에 따름 |
| DIODE val | `["ASSET:MIT", "HOLDOUT:diode_val"]` | 거부(hold) | 평가 전용 |
| 야크 | `Prov(taint={"USER","HOLDOUT:r_eval"}, ns="user_vault")` (`pipe_yak.make_input_image`와 같음) | 거부(ext, hold) | 평가 미리보기 전용 |
| Depth-Anything-V2 출력 | 저장 경로가 `eval/`뿐, prov 기록 없음 | — | 비교 숫자로만 |

- **DIODE를 `CAPTURE`·`ENGINE`·`ASSET:CC-BY`로 바꿔 다는 것은 출처 세탁이므로 금지한다.** 정당한 경로는 **FND-04 개정** 하나뿐이다. `TRAIN_LICENSES`에 `"MIT"`를 추가한다(`buc/gov/prov.py:41`, `ops/harness/contracts/FND-04.md:18`, `tests/fnd_04/test_tags.py:67`, `gate.py --freeze FND-04`). 이 개정은 오케스트레이터가 하고 창업자 확인이 필요하다. 창업자는 이미 `docs/DATA-PLAN.md`에서 DIODE를 학습 OK로 결정했다. 빌더는 이 파일들을 건드리지 않는다.
- 잡의 동작:
  - DIODE train/dev 탭은 F2c에서 **항상 추출**한다. 방화벽 판정 `P.why_not_train(P.as_record(["ASSET:MIT"]))`이 비어 있지 않으면 탭을 `taps_pending/diode/*.npz`에 둔다. 이 디렉터리는 학습 경로가 아니며 로더가 읽지 않는다.
  - 학습 시작(T) 직전과 재시작마다 `promote_pending()`이 판정을 다시 한다. 통과하면 `ShardWriter(purpose="train")`로 승격해 쓴다(작성기가 방화벽을 다시 검사한다). 통과하지 않으면 합성만으로 학습하고 `D1_M3 {"diode_train":"blocked_fnd04"}`를 남긴다.
  - 도메인 구성은 학습 시작 시점에 고정한다. `run_id` 둘 중 하나: `m3d1-mix`(DIODE 0.6 + 합성 0.4) 또는 `m3d1-syn`(합성 1.0). 개정이 학습 도중에 들어오면 night2에서 `m3d1-mix`를 새로 시작한다(syn 가중치로 웜 스타트 허용, 옵티마이저는 새로).
- 기저 맞춤 prov = `P.derive("pca_fit", [코퍼스·렌더의 prov 전부])`. 하나라도 train_ok가 아니면 `fit_pca_basis`가 FirewallError를 낸다. 기저 맞춤 목록을 만드는 함수 `fit_set()`은 코퍼스와 합성 train 경로만 받는다(테스트로 고정).
- **R2:** 모든 정답은 레이저(DIODE)나 분석적 렌더(RoomGen)에서 온다. 캡션은 고정 문자열이다. Depth-Anything-V2, pipe_yak 깊이, 교사 출력은 **어디에서도 라벨이 되지 않는다.** DA-V2 결과는 `day1/m3/eval/`에만 저장하고, 학습 로더는 `shards/train/` 밖의 경로를 읽으면 assert로 멈춘다.
- **야크:** `hf_hub_download(sdsdsdsdsdsdsd123/buc-artifacts, "demo/d15_hero_yak.webp")`로 `$N1_ROOT/day1/m3/private/`에만 받는다. 기저 맞춤·ShardWriter·TapRing·학습 로더에 절대 들어가지 않는다(`ring=None`, 임시 TapSet). 야크 탭(`private/yak_taps.npz`)과 미리보기는 박스 로컬과 비공개 금고 `ckpt/day1/m3/preview/`에만 둔다. 레포 디렉터리(`repo_m3/` 등) 아래에 쓰려 하면 `safe_out_path()`가 예외를 낸다. `HF_TOKEN`은 환경변수에서만 읽고 출력하지 않는다(`scrub()`).

---

## 2. 교사 특징 추출 (GPU, 하위 프로세스 `--stage teacher`)

### 2.1 VRAM 게이트와 해제
- 게이트: 60초마다 `torch.cuda.mem_get_info()[0] >= 24e9`가 되면 통과한다. 기다리는 동안 5분마다 `D1_M3 {"stage":"gpu_wait","free_gb":…,"m6":"<logs/m6.run.log의 teacher released 여부>"}`를 남긴다. M6 로그는 참고만 하고 판정은 `mem_get_info`로 한다.
- 통과하면 `torch.cuda.set_per_process_memory_fraction(22 GiB / total)`을 건다. M2 2.9 + M3 22 = 24.9 GB < 32 GB.
- 교사 상주량: 변환기와 VAE 약 12.5 GB. 텍스트 인코더(Qwen3-4B 약 8 GB)는 `SURVEY_CAPTION`을 한 번 인코딩하고 바로 `teacher.release_text_encoder()`를 부른다. 임베딩은 `day1/m3/cap_embed.pt`에 캐시하고, 재시작 때 캐시가 있으면 텍스트 인코더를 올리지 않는다. 최대 약 21 GB + 활성값 1–3 GB이므로 22 GiB 상한 안에 들어간다.
- 해제: 교사 단계 끝(정상, 마감, 예외 모두)에 `bus.detach()` → `del sampler, bus, teacher` → `gc.collect()` → `torch.cuda.empty_cache()`. 그다음 `torch.cuda.memory_allocated()==0`, `memory_reserved()==0`, `mem_get_info` free가 로드 전 대비 10 GB 이상 회복됐는지 확인하고 `D1_M3 {"teacher":"released",…}`를 남긴다. 그 뒤 **하위 프로세스가 종료**하므로 CUDA 컨텍스트까지 확실히 반환된다. 부모 프로세스는 종료 후 `mem_get_info`로 한 번 더 확인한다.

### 2.2 로드와 백엔드
- `find_teacher_cache`는 `pipe_yak.py`의 로직을 복사해 쓴다. `Z.load_teacher(spec, device="cuda", te_device="cuda", hub_cache=hub, seed=0, deterministic=False)`. 스파이크는 속도를 위해 `deterministic=False`로 돌리고 meta에 `det:false`를 기록한다. `sampler = TeacherSampler(teacher)`(`max_views_per_call=None`, 배치 하나가 순전파 1회).
- **교사 NaN 버그 대응(STATE.md: `buc/teacher/zimage.py` 경로에서 `TeacherNumericError`가 났다):** F0에서 `buc` 백엔드를 시험한다. NaN이나 `TeacherNumericError`가 나면 `backend="diffusers"`로 바꾼다. 이 경로는 M6 방식이다. `ZImageTransformer2DModel.from_pretrained(subfolder="transformer", bf16)`과 VAE를 로드하고, 정규화 `(z−shift)·scale`은 vae.config에서 가져온다. 캡션 임베딩은 `ZImagePipeline.encode_prompt`로 한 번 만든다. `transformer(xs, t=1−sigma, caps, return_dict=False)`, 잡음은 `torch.Generator("cpu").manual_seed(derive_seed(seed,key))`로 만든다. TapBus는 `T.TapBus(basis).attach(transformer.layers)`로 똑같이 붙인다. 백엔드 이름은 기저 meta, 모든 샘플 meta, ckpt hparams에 남긴다. RD-07을 읽을 때도 같은 백엔드를 써야 한다(위험 목록 참고).

### 2.3 한 이미지 읽기 (RD-07 `MrhReader`와 같은 절차)
1. `x = 2·img01 − 1`, float32 [B,3,H,W]. H와 W는 16의 배수여야 한다(DIODE 768×1024, 합성 512×384, 코퍼스는 1024²를 가운데 768×1024로 자르고 1344×576은 그대로 둔다).
2. `x0 = teacher.encode_image(x)` → [B,16,H/8,W/8].
3. `eps = cat(teacher.make_noise(det.derive_seed(SEED, key), 1, H, W))`, `z = add_noise(x0, 0.3, eps)`.
4. `with torch.no_grad(), bus.capture(rec, grid=(H//16, W//16), token_offset=0, key=key, ring=None) as cap: sampler.predict_x0(z, 0.3, [SURVEY_CAPTION]*B)`.
   - **조기 종료:** 블록 23에 두 번째 훅을 단다. TapBus 훅이 먼저 실행된 뒤 비공개 예외 `_StopAfterTap`을 던지고, `with` 블록 **안에서** 잡는다. 블록 24–29를 건너뛰므로 약 20% 빨라진다. F0에서 조기 종료 유무에 따라 int8 탭이 같은지 확인한다(불일치 비율 ≤ 0.1%).
5. `cap.tapset.layers[l]`: int8 [B,128,H/16,W/16]. 이미지별로 `.cpu().clone()`한다.
- 배치: 768×1024는 B=2, 512×384는 B=4(같은 해상도끼리만 묶는다).
- 처리량 추정(E0에서 외삽, 미측정): 768×1024 약 0.35–0.45 s/장(조기 종료 포함), 512×384 realify 약 0.15–0.25 s/장, 공유 시 ×1.3–1.5.

### 2.4 교사 단계 순서 (각 단계에 완료 표지)

| 단계 | 일 | 개수 | 예상(전용) | 상한 |
|---|---|---|---|---|
| F0 smoke | 512² 그라디언트 1장 + 코퍼스 1장 + DIODE val 1장. 훅 출력 형상 `(1, n_img + pad32(L_cap), 3840)`, `S − n_img ∈ {32, 64}`, 출력 유한, x0-hat PSNR이 정상인지(`tests/tch_02/test_tch02_gpu_real.py`의 sdedit roundtrip과 같은 방식), 각 탭 층이 정확히 한 번 실행되는지, 조기 종료 결과가 같은지, 768×1024와 512×384에서 순전파 시간과 `max_memory_allocated` 기록 | 3 | 5–10분 | 15분. 실패하면 **잡을 멈추고** fatal |
| F1 기저 | RawTap 훅(`out[:, :n_img]`)으로 코퍼스 600 + 렌더 400(train 시드 앞쪽)을 sigma 0.3에서 읽는다. 이미지·층마다 토큰 96개를 시드 고정 `torch.Generator(1234)`로 뽑는다. 층별 L2 노름 상위 0.1%를 버리고 `T.fit_pca_basis(samples, prov=derive("pca_fit",…), k=128, meta={sigma:0.3, caption, backend, teacher_rev})`. 진단: 따로 떼어 둔 토큰 10%에서 int8 대 float 투영의 상대오차, 성분별 \|q\|==127 비율, 설명 분산. 상대오차가 10%를 넘으면 상위 1%를 버리고 다시 맞춘다(최대 2회). `basis.save(day1/m3/basis.pt)` 후 지문(fingerprint) 기록 | 1,000 | 10–15분(+ CPU 맞춤 2분) | 30분 |
| F2a 평가 탭 | DIODE val 771장(prov HOLDOUT) → `eval/taps/val/<stem>.npz`(int8 3층). 야크 1장은 면적 589,824 px 근처로 비율을 유지하고 각 변을 16의 배수로 맞춘다 → `private/yak_taps.npz`. 둘 다 샤드가 아니다 | 772 | 5–7분 | 15분 |
| F2b 합성 realify | 6,000 train + 200 dev, 위 1b 규칙 | 6,200 | 20–25분 | 40분 |
| F2c DIODE train/dev | `.gt.npz`가 생긴 샘플부터 처리한다(스트림과 병행). 방화벽 판정에 따라 shards 또는 taps_pending에 쓴다 | 3,661 | 25–30분 | 50분 |
| F3 해제 | 2.1 절차 | — | 1분 | — |

- 기저는 **동결**한다. `basis.pt`가 있으면 F1을 건너뛴다. 다시 맞추려면 `--refit-basis`를 명시해야 하고, 그 경우 모든 샤드와 ckpt가 무효가 된다(`basis_fp` 불일치는 `TapBasisError`).
- 기저 meta의 해상도 구성(768×1024, 512×384, 1344×576)은 추출 분포와 같다.

### 2.5 저장 형식 (컴팩트 샘플, DAT-05 tar)
- 학습 샤드: `ShardWriter(day1/m3/shards/train/{diode|synth}, max_samples=500, purpose="train")`. dev 샤드는 `shards/dev/{diode|synth}`(같은 작성기, 키 접두어 `dv_`). 필드는 다음과 같다.
  - `tap8.npy`, `tap15.npy`, `tap23.npy`: int8 [128, gh, gw].
  - `lab4.npz`: `log_depth` float32 [H/4,W/4], `mask` uint8 [H/4,W/4], (합성만) `normal` float16 [3,H/4,W/4]. H/4 라벨은 `downsample_labels(stride=4)`와 **같은 규칙**이다. 블록 16 px가 모두 유효할 때만 마스크가 1이고, 로그 깊이는 평균, 법선은 평균 후 재정규화한다. DIODE는 절반 해상도 GT에서 `vcnt` 합이 16인지 보는 방식으로 만든다. 클램프는 [0.25, 400] m.
  - `meta.json`: `v="buc.d1.m3.sample.v1"`, `key`, `domain`(diode|synth), `env`(indoors|outdoor|room), `scene`, `basis_fp`(hex), `tap_layers=[8,15,23]`, `tap_sigma=0.3`, `band_valid`(합성은 realify 결과, DIODE는 **전부 True**. 실사는 다시 칠하지 않으므로 GT가 입력 이미지와 모든 대역에서 맞다), `keep`, `lowband_err`, `realify` 모드, `backend`, `det`.
  - `canonical gbuf`(12ch, 전체 해상도)는 쓰지 않는다. 768×1024 f16이면 장당 19 MB이고 3,661장이면 69 GB라 디스크에 들어가지 않는다. MDL-03 `decode_sample`/`MRHDataset`도 쓰지 않는다(전부 RAM에 올리는 eager 방식이고 gbuf가 필요하다). 대신 잡 로컬 `LazyShardDataset`을 쓴다. `verify_shard(purpose="train")`를 한 번 하고, 멤버는 `os.pread`로 그때그때 읽고, 초기화 때 meta를 전부 검사한다.
- 크기: DIODE 약 1.4 MB/장 × 3,661 ≈ 5.2 GB, 합성 약 0.43 MB/장 × 6,200 ≈ 2.7 GB, val 탭 약 0.9 GB. 총 약 9 GB.
- `blake3`가 없으면 `pipe_yak.ensure_blake3()` 방식의 shim을 쓰고 meta에 `blake3:"shim"`을 기록한다.
- 업로드: 샤드는 `corpus/day1/m3_taps/{train,dev}/{diode,synth}/`, 기저는 `ckpt/day1/m3/basis.pt`. val 탭과 야크 탭은 `eval/day1/m3/`이 아니라 `ckpt/day1/m3/eval_taps/`(비공개 금고)에 둔다.

---

## 3. 학습 (GPU, 하위 프로세스 `--stage train`, 교사를 내린 뒤)

- **모델:** `MRH(MRHConfig(), seed=0)`, 파라미터 17,516,303개(넥 15.94M, 헤드 1.58M). 입력은 `feat_l = q·scale_l / tap_std_l`이다. `tap_std`는 정렬된 학습 키 중 앞 256개에서 구한 층별·성분별 std(최소 1e-6)이고, ckpt hparams에 저장한다. 나중에 내보낼 때 `reassemble[i].proj.weight`에 접어 넣을 수 있다. FiLM t는 `tap_sigma`(0.3).
- **손실**(잡 로컬 `m3_loss`, `buc.models.mrh.train`의 band projection·`band_valid_at_stride`·`_vector_term`·`_mean_or_zero`를 재사용한다):
  - 깊이(이동 불변, a=1): 샘플마다 `Bp=BP(m·p)`, `Bm=BP(m)`, `Bg=BP(m·g)`, `b=<Bm,Bg−Bp>/(<Bm,Bm>+1e-8)`, `r=Bp+b·Bm−Bg`, `L_d=<r,r>/n`. 지수가 1로 고정되므로 거듭제곱 왜곡이나 깊이 반전을 허용하지 않는다(계약 손실의 자유 지수 a 문제를 피한다).
  - 스케일: `L_s = (Σm(p−g)/Σm)²`. 합성과 DIODE 모두 ln(미터) 정답이므로 미터 스케일을 약하게 배운다(가중치 0.15, SILog λ=0.85와 같다). 원시 미터 평가가 가능해진다.
  - 법선: `nmask = mask·(|n_gt| > 0.5)`. 유효 픽셀이 16개 이상인 샘플만(즉 합성만) 계산하고 가중치 0.5. DIODE 배치에서 법선 헤드 grad는 정확히 0이어야 한다(잡 안에서 첫 DIODE 배치에 assert).
  - sem 0(무작위 표 자리표시자라 의미가 없다), albedo 0.
  - `total = 1.0·L_d + 0.15·L_s + 0.5·L_n`. 유효 픽셀이 16 미만인 샘플은 깊이 항에서 뺀다.
  - 넥은 `autocast(bf16)`, 헤드와 손실은 `autocast(enabled=False)` fp32. TF32 켬, `cudnn.benchmark=True`, `deterministic=False`.
- **배치·도메인:** 스텝마다 `random.Random(f"{seed}-dom-{step}")`로 도메인을 고른다(mix: DIODE 0.6, 합성 0.4 / syn: 합성 1.0). DIODE는 전역 배치 8(마이크로 4 × 누적 2, 768×1024), 합성은 전역 16(마이크로 16, 512×384). 도메인 안의 순서는 `(seed, domain, epoch)` 순열이다. 재개해도 같은 데이터 순서가 나온다. 4스레드 프리페처를 쓴다.
- **최적화:** AdamW lr 3e-4, betas (0.9, 0.99), wd 0.01, warmup 300 → cosine으로 0.1·lr(6,000에서), grad clip 1.0, EMA 0.999(초반에는 `min(0.999,(1+s)/(10+s))`). `total_steps=6000`, `warmup=300`, 손실 가중치, `run_id`, `basis_fp`, `tap_std` 해시, 도메인 구성을 `config_hash`에 넣는다. 불일치하면 ckpt를 무시하고 경고한다.
- **메모리:** `set_per_process_memory_fraction(10 GiB/total)`. OOM이 나면 마이크로배치를 반으로 줄이고(누적은 2배) ckpt에 저장한다. 한 번 줄이면 다시 늘리지 않는다. 비유한 grad는 스텝을 건너뛰고, 25회 연속이면 `ckpt_last`를 다시 읽고, 100회 연속 실패면 fatal.
- **속도 추정:** 768×1024 B=8 fwd+bwd 약 18 TFLOP → 0.3–0.6 s/스텝(미측정). 6,000스텝에 약 50–90분. 첫 5스텝에서 `max_memory_allocated`와 스텝 시간을 마커로 남긴다.
- **체크포인트:** `day1/m3/ckpt_last.pt` = {version, step, config, config_hash, model, ema, optimizer, scheduler, micro_batch, rng, best, tap_std, basis_fp, saved_at}. 약 280 MB. 500스텝이나 10분마다(먼저 오는 쪽), 매 평가 전, 멈출 때 원자적으로 저장한다(`.tmp` → fsync → `os.replace`). `ckpt_best.pt`는 EMA 가중치만(약 70 MB). DIODE dev 중앙값 정렬 AbsRel이 최소일 때 저장하고, syn 런이면 합성 dev 기준이다. 시작할 때 `clean_tmp()`. 재개 때 로컬 ckpt가 없으면 금고 `ckpt/day1/m3/ckpt_last.pt`를 받는다.
- **평가 주기:** 500스텝에 첫 평가(첫 숫자를 빨리 보기 위해), 그 뒤 1,000스텝마다 + 마지막. EMA로 평가한다.
- **마감:** `N1_DEADLINE`(13:00Z = 1791723600). 학습은 `now ≥ deadline − 600`이면 멈춘다. 마지막 단계는 ckpt_last 저장 → 평가(DA-V2 숫자는 캐시를 쓴다) → 미리보기 → 업로드 → 업로드 스레드 join(240초)이다. SIGTERM·SIGINT는 이벤트 플래그로 받아 같은 마지막 단계를 거친다.
- **업로드**(HfApi, 데몬 스레드, 3회 재시도, 실패해도 치명 아님, 토큰이 없으면 `hf_token:"no"`만 기록): `ckpt/day1/m3/` ← `ckpt_last.pt`, `ckpt_best.pt`, `basis.pt`, `metrics.jsonl`, `eval_table.json`, `eval_table.md`, `stages.json`, `SOURCE_DIODE.json`. `ckpt/day1/m3/preview/` ← 야크·DIODE 미리보기.

### 마커 (한 줄 JSON, `fit_marker`로 400자 이하로 자름, `scrub` 적용, `flush=True`)
- `D1_M3 {"start":true,"root":…,"hf_token":"yes|no","deadline":…,"torch":…,"stage":…}`, 진행 상황은 240초마다(`stage`, `step`, `loss`, `l_d`, `l_s`, `l_n`, `lr`, `sps`, `mb`, `vram_gb`, `free_gb`).
- `D1_M3_DATA {"split","seen","kept","written","skip","bad","pos_gb","mbps","reconn"}`(DIODE) / `{"synth":…,"done":…,"rate":…}`(렌더), 60초마다.
- `D1_M3_EVAL {"step","val_med_absrel","val_med_d1","in_absrel","out_absrel","raw_absrel","ss_absrel","dav2_ssd_absrel","const_med_absrel","dev_absrel","preview":"ckpt/day1/m3/preview/yak_step000500.png"}`.
- `D1_M3_DONE {"status":"done|deadline|fatal","step",…}`. `done`일 때만 `day1/m3/DONE.json`을 원자적으로 쓴다. `deadline`이면 DONE.json 없이 정상 종료(rc 0)하고 night2가 이어 간다. `fatal`이면 rc 2.
- 마커가 아닌 출력에는 'error'라는 단어를 쓰지 않는다(러너의 marks 필터가 grep한다).

---

## 4. 평가 (정직하게)

- **대상:** DIODE val 771장(실내 325, 실외 446). 학습과 기저 맞춤에 쓰지 않은 장면 19–24. 모델 선택은 dev(train 장면 00004/00009/00012)로만 하고, val은 보고용이다. 평가 탭은 `eval/taps/val/`에서만 읽는다.
- **예측:** EMA MRH → log_depth [H/4] → 로그 공간에서 bilinear(`align_corners=False`)로 768×1024까지 올린다. GT는 전체 해상도이고 마스크는 `valid & 0.25 ≤ d ≤ 350 m`. 예측은 [1e-3, 1e4] m로 클램프하고 클램프된 비율을 기록한다.
- **정렬(모든 행에 같은 정의):**
  1. **원시 미터:** `d̂ = exp(p)`.
  2. **중앙값 스케일(헤드라인):** `d̂·median(d/d̂)`. 스케일만 고치고 모양(지수, 이동)은 고치지 않으므로 기하적으로 가장 정직하다.
  3. **스케일+이동(깊이):** 깊이 공간 최소제곱 `s·d̂+t`(Marigold 관례).
  4. **스케일+이동(시차):** 시차 공간 최소제곱(MiDaS 관례). DA-V2와 사과 대 사과로 비교하기 위해 넣는다.
- **지표:** AbsRel, δ1(<1.25), RMSE-log. 이미지별 평균을 집계하고 전체·실내·실외로 나눈다. N을 적는다. 헤드라인에는 이미지 부트스트랩 95% CI(1,000회, 시드 0)를 붙인다.
- **기준선 (같은 표):**
  - (i) 상수: 원시 미터 = `exp(DIODE train 선택분 유효 픽셀 평균 ln d)` 한 값. 중앙값 정렬 = 이미지마다 GT 중앙값 상수(정답 스케일을 아는 바닥선). 스케일+이동 = 퇴화(평균).
  - (ii) **Depth-Anything-V2-Small**(`depth-anything/Depth-Anything-V2-Small-hf`, Apache-2.0, transformers, GPU 약 1.5 GB). 출력은 상대 시차이므로 **4(시차 스케일+이동)만 유효**하고 1·2·3은 n/a로 적는다. 교사 게이트를 기다리는 동안 B0 단계에서 한 번 계산한다(VRAM 상한 3 GiB). 이미지별 지표와 미리보기 8장만 `eval/dav2/`에 캐시하고, **예측 맵은 학습 경로 어디에도 쓰지 않는다.** DA-V2가 DIODE를 학습에 봤는지는 확인하지 않았다(제로샷으로 알려져 있음). 표 각주에 적는다.
  - (iii) MRH 행은 1–4 모두.
- **합성 dev**(200장): 중앙값 정렬 AbsRel·δ1, 법선 각도 평균·중앙값·%<11.25/22.5/30°(GT를 정규화한 뒤).
- **산출:** `eval_table.json`과 `eval_table.md`. 헤드라인은 "DIODE val 중앙값 정렬 AbsRel/δ1(실내/실외)"이고, 원시 미터와 스케일+이동은 보조 열이다. 시차 정렬은 DA-V2 비교용이라고 표에 명시한다. 정렬하지 않은 원시 미터가 나쁘게 나와도 숨기지 않는다.
- **미리보기**(비공개 금고 `ckpt/day1/m3/preview/`):
  - `yak_step{step:06d}.png`, `yak_latest.png`: 2×2 패널. 입력 썸네일 | MRH 깊이(중앙값 정규화, turbo 색표는 numpy로 구현, 가까울수록 따뜻한 색) | DA-V2 깊이(평가 참고용 라벨 표시) | 점구름 미리보기.
  - 점구름: HFOV 60°를 가정해 역투영하고, numpy z-버퍼 스플랫(1 px)으로 yaw −12°, 0°(중앙값의 0.3배만큼 돌리 인), +12° 세 시점에서 그린다. `yak_pcd_step….png`, 폭 768.
  - `diode_val_strip.png`: 고정 val 6장(실내 3, 실외 3). RGB | GT | MRH | DA-V2.
  - 이 그림들은 레포에 절대 커밋하지 않는다.

---

## 5. 단계 계획과 시간 예산

현재 01:51Z(1791683459), 마감 13:00Z. 출시(빌드, 검토, 푸시, extra.sh 반영)는 약 04:00–04:30Z로 가정한다. G는 VRAM 게이트가 통과한 시각이다.

| 단계 | 자원 | 시작 | 예상 | 완료 표지 |
|---|---|---|---|---|
| P0 준비 | CPU | 출시 | 1분 | — (매번 실행: env, 마감, clean_tmp, 디스크 확인, 방화벽 탐침, blake3) |
| D-val, D-train | CPU·네트 (하위 프로세스, nice 10) | 출시 | 5분 + 30–90분 | `stages/diode_{val,train}.done.json` |
| R 렌더 | CPU 6 procs (하위 프로세스) | 출시 | 15–25분 | `stages/render.done.json` |
| B0 DA-V2 기준선 | GPU ≤3 GiB | 출시 + D-val 완료 | 3–5분 | `stages/dav2.done.json` |
| G 게이트 | — | — | 05:00Z쯤으로 가정 | — |
| F0–F3 교사 | GPU ≤22 GiB | G | 약 1.5시간(상한 2.5시간) | `stages/f0…f2c.done.json`, `basis.pt` |
| T 학습 | GPU ≤10 GiB | F3 | 50–90분 | `DONE.json` |
| E 평가 | T 안에서 | 500스텝, 이후 1,000마다 | 회당 약 2분 | `metrics.jsonl` |

- **G = 05:00Z 시나리오:** F0 05:10, F1 05:30, F2a 05:40, F2b 06:05, F2c 06:35(DIODE 스트림이 그때까지 끝나 있어야 함), F3 06:40, **첫 평가와 야크 미리보기 약 06:55Z**, 학습 끝 약 08:30Z. 13:00Z보다 4시간 넘게 이르다.
- **게이트가 늦을 때의 규칙**(`R = deadline − now`, 게이트 통과 시점에 판정):
  - R ≥ 4.5시간: 전체 계획.
  - 2.5시간 ≤ R < 4.5시간: F2b 상한 3,000장, F2c 상한 2,000장(장면 균형 접두부), 학습은 시간 예산만큼.
  - 1.25시간 ≤ R < 2.5시간: F0, F1, F2a, F2b 1,500장 + F2c 800장 → 40분 학습 → 평가와 미리보기 1회.
  - R < 1.25시간: 교사를 올리지 않는다. `D1_M3_DONE {"status":"deadline","stage":"gpu_wait"}`, rc 0. night2가 이어 간다.
  - 교사 단계 안에서는 `deadline − 1800` 이후 새 배치를 시작하지 않는다.
- **멱등성:** 모든 단계는 `stages/<name>.done.json` = {stage, config_hash, finished_at, counts}로 끝을 표시한다. config_hash가 같으면 건너뛰고, 다르면 다시 한다(기저는 예외: 위 동결 규칙). F2 단계는 키별 진행 기록(`progress_<stage>.jsonl`, 샤드를 닫은 뒤 키를 추가)으로 샘플 단위로 이어 간다. 하위 프로세스는 pid 파일 + `/proc/<pid>/cmdline` 확인으로 중복 실행을 막는다.
- **night2:** 같은 명령에 `--run night2 --marker-prefix N2_M3`(마감은 night2 러너의 `N1_DEADLINE` 환경변수). 로컬 파일이 없으면 금고에서 복원한다(`basis.pt`, 샤드, `ckpt_last.pt`, DIODE 부분집합). 남은 F 단계 → T 재개 순서로 간다.

---

## 6. 출시 방법 (extra.sh 블록, 오케스트레이터가 넣는다. 이 잡은 extra.sh·runner.sh를 고치지 않는다)

```bash
M3_RUN=1
if [ ! -f "$R/.m3_launched_$M3_RUN" ] && ! pgrep -f m3_mrh.py >/dev/null; then
  touch "$R/.m3_launched_$M3_RUN"
  # pipe 블록과 같은 codeload tarball 받기 → dst "$R/repo_m3" (| x repo_m3)
  OUT=/proc/$PPID/fd/1; [ -w "$OUT" ] || OUT=/dev/null
  setsid nohup bash -c 'cd "$1"; export PYTHONPATH="$1/repo_m3:$1/code" PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=1; n=0
    while [ "$(date +%s)" -lt $((N1_DEADLINE-300)) ] && [ $n -lt 15 ]; do
      python -u "$1/repo_m3/ops/gpu/day1/m3_mrh.py" --root "$1" --stage all >> "$1/logs/m3.run.log" 2>&1
      rc=$?; [ $rc -eq 0 ] && break; [ -f "$1/day1/m3/DONE.json" ] && break
      n=$((n+1)); echo "D1_M3 {\"restart\":$n,\"rc\":$rc}" >> "$1/logs/m3.run.log"; sleep 60
    done; echo "D1_M3 {\"exit\":\"$(date -u +%FT%TZ)\"}" >> "$1/logs/m3.run.log"' _ "$R" </dev/null >/dev/null 2>&1 &
  setsid nohup bash -c 'sleep 2; tail -n +1 -F "$1/logs/m3.run.log" 2>/dev/null | grep --line-buffered -E "^D1_M3|Traceback" | awk "{ print substr(\$0, 1, 400); fflush() }" > "$2"' _ "$R" "$OUT" </dev/null >/dev/null 2>&1 &
  echo "D1_X m3 launched out=$OUT"
fi
```
- `PIPE_RUN`·`PIPE_RUN2` 값은 그대로 둔다(바꾸면 파이프가 다시 뜬다). 코드는 브랜치에 **푸시해야** 박스에 도달한다(raw 캐시 + 180초 폴링).
- 로그 이름은 `logs/m3.run.log`다. 러너가 마지막에 `logs/m*.log`를 `logs/day1/`에 올린다.
- 래퍼는 rc 0(done 또는 deadline)이면 멈추고, rc ≠ 0이면 60초 간격으로 최대 15회 다시 띄운다.
- **박스 정지 위험:** M13·M2·M6가 다 끝나면 러너가 `.main_done` → `self_stop`으로 가고 분리된 M3도 죽는다. 대응: M6가 끝나 가는데(D1_M6_DONE 직전) M3 `DONE.json`이 아직 없으면, 오케스트레이터가 extra.sh 새 판에 전경 대기 블록을 넣는다. `while [ ! -f day1/m3/DONE.json ] && [ $(date +%s) -lt $((N1_DEADLINE-300)) ] && pgrep -f m3_mrh.py >/dev/null; do echo "D1_X m3hold"; sleep 120; done`. 대기 중에는 원격 수정이 막힌다는 대가가 있다.
- night2 런처는 새 `M3_RUN` 값을 쓴다(플래그 파일이 디스크에 남아 있으므로).

---

## 7. CPU 테스트 계획 (`ops/gpu/day1/test_m3_mrh.py`, pytest, 60초 이하, GPU·외부 네트워크·diffusers 없음)

가짜 객체는 `deps` 딕셔너리로 주입한다. `tests/`의 fakes는 import하지 않고, 필요하면 `tests/dat_02/dat02_fakes.py`의 FakeDiT·FakeTeacher 구조를 본떠 테스트 파일 안에 작게 만든다(블록 30개, D=64, 기저 k=8, token_offset 0).

1. **DIODE 이름과 선택:** `parse_member`가 indoors/outdoor 철자와 정크·`..` 경로를 처리하는지(None 반환). `Selection.keep`이 이름만 보는 순수 함수이고 순서와 무관한지. spike profile 개수(가짜 CSV 25,458행 → 3,661, 실외 2,958). dev 장면(00004/00009/00012)이 train 인덱스에서 빠지는지.
2. **스트리밍 재개:** localhost `http.server`(Range, If-Match, 중간 끊김 3회 주입)로 작은 `.tar.gz`. 결과 바이트가 같은지, md5가 같은지, 412는 치명, 재실행하면 done 샘플을 건너뛰는지, `.part`를 정리하는지.
3. **GT 축약:** 절반 해상도 기하평균과 `vcnt`, 그리고 H/4 라벨이 `buc.models.mrh.train.downsample_labels`(같은 GT로 만든 12ch gbuf)와 1e-5 안에서 같은지. val packbits 왕복.
4. **렌더:** 실제 `render_gbuffer` 2장면×2시점, 64×48. 결정론(프로세스 1개 대 2개에서 바이트가 같은지), 톤맵이 [0,1]이고 하얗게 날아가지 않는지, 키 정규식, ValueError 시드를 건너뛰는지.
5. **교사 경로(가짜):** 훅이 층 8/15/23에서 한 번씩만 실행되는지. 조기 종료 탭 == 전체 순전파 탭. `max_views_per_call` 분할이 TapError를 내는지 확인(우리는 None을 쓴다). 캡션이 늘 `SURVEY_CAPTION`인지. sigma 0.3.
6. **VRAM 게이트와 해제:** 가짜 `mem_get_info` 순열(10 → 30 GB)에서 기다렸다가 통과하는지. 마감 직전에는 `status=deadline` rc 0으로 끝나는지. 해제 검증 함수가 allocated/reserved≠0이면 경고하는지.
7. **기저:** 가짜 토큰으로 `fit_pca_basis` → 저장/로드 지문. 상위 노름 제거. **기저 맞춤 목록에 HOLDOUT·USER·ASSET:MIT 샘플이 들어가면 FirewallError**가 나는지, `fit_set()`이 val·야크 경로를 받지 않는지.
8. **방화벽 흐름:** 현재 어휘에서 DIODE train → `taps_pending`에 가고 shards에는 없는지. `monkeypatch`로 `P.TRAIN_LICENSES`에 MIT를 넣으면 `promote_pending`이 승격하는지(테스트 안에서만). 야크·val 키가 어떤 샤드에도 없는지. 학습 로더가 `shards/train/` 밖을 읽으면 assert하는지.
9. **샤드 왕복:** `ShardWriter`(blake3 shim 포함) → `LazyShardDataset`. basis_fp 불일치는 TapBasisError, keep=False는 거부, meta 검사.
10. **손실:** p에 상수를 더해도 `L_d`가 변하지 않는지(1e-6). `L_s`는 상수에 반응하는지. 반전 깊이(−ln d)는 큰 손실. DIODE(법선 없음) 배치에서 법선 헤드 grad == 0. 결과 유한.
11. **학습 루프**(tiny MRHConfig, `tests/mdl_03/mdl03_helpers.py::tiny_cfg`와 같은 크기): 3스텝 → ckpt 저장 → 새 프로세스 객체로 재개하면 스텝이 이어지고 다음 배치 인덱스가 같은지. OOM 주입 시 마이크로배치가 반으로 줄고 저장되는지. 가짜 시계로 마감 → DONE.json 없음 + `D1_M3_DONE status=deadline`. total_steps에 도달하면 DONE.json. config_hash 불일치면 ckpt를 무시하는지. 기본 `MRHConfig()` 파라미터 수가 15M–20M인지.
12. **평가 지표:** 완벽한 예측 → AbsRel 0, δ1 1. 중앙값 정렬이 스케일 불변인지. 스케일+이동(깊이·시차) 닫힌 해. 상수 기준선. 마스크와 범위. 실내/실외 분리. 부트스트랩 CI가 결정론적인지.
13. **미리보기:** 색표와 점구름 스플랫 형상이 맞고 NaN이 없는지. `safe_out_path`가 레포 루트 아래 경로를 거부하는지. 미리보기 업로드 경로가 `ckpt/day1/m3/preview/`로 시작하는지.
14. **마커와 비밀:** 모든 마커가 400자 이하인지. `HF_TOKEN=hf_fakeTOKEN123456`을 넣었을 때 stdout, 로그, metrics, meta 어디에도 나타나지 않는지.
15. **단계 멱등성:** done 표지가 있으면 두 번째 실행이 건너뛰는지(가짜 카운터). config_hash가 바뀌면 다시 하는지. `basis.pt`가 있으면 `--refit-basis` 없이는 F1이 돌지 않는지.

GPU 박스에서 처음 실행할 때 F0 smoke가 실제 가중치 검사를 대신한다(통과하지 못하면 fatal).

---

## 8. 위험

1. **DIODE가 방화벽에 막혀 있다**(`ASSET:MIT` ∉ TRAIN_LICENSES). FND-04 개정이 없으면 오늘 학습은 합성 전용(`m3d1-syn`)이 된다. 실사 정답 없이 실외·먼 거리 깊이 품질은 약하다. **오늘 가장 큰 결정 사항**이며 창업자/오케스트레이터 몫이다.
2. **교사 NaN 버그**(`buc/teacher/zimage.py`). diffusers 백엔드로 피하지만, 그러면 RD-07 읽기 경로와 특징이 미세하게 다를 수 있다. 백엔드 이름을 기록하고, RD-07이 같은 백엔드로 읽도록 맞춰야 한다.
3. **실제 가중치에서 검증된 적 없음:** 토큰 배치(offset 0, 행 우선, `pad32`), TapBus, realify는 가짜 객체에서만 돌았다. F0가 실패하면 오늘 M3는 멈춘다.
4. **VRAM 경합:** M6 라벨 단계(약 21 GB)가 끝나지 않으면 게이트가 열리지 않는다. 늦게 열리면 축소 계획으로 가거나 night2로 넘긴다.
5. **int8 PCA 품질:** 실제 은닉 상태의 거대 활성 토큰이 max/127 스케일을 키울 수 있다. 상위 노름 제거와 재맞춤으로 대응하고, 그래도 나쁘면 TCH-03 개정(백분위 스케일)을 요청한다.
6. **도메인 차이:** DIODE RGB는 FARO HDR 톤맵이고, RoomGen은 저폴리 CG다(realify로 일부 완화). 합성 정답은 거친 대역만 감독한다. 경계 선명도는 DIODE에서만 배운다.
7. **미터 스케일 항**(`L_s`): 단일 이미지 스케일은 모호하다. 실내 합성(최대 8 m)과 실외 DIODE(최대 수백 m)가 섞이면 원시 미터 오차가 클 수 있다. 헤드라인은 중앙값 정렬이고, 원시 값은 정직하게 보조로 적는다.
8. **박스 자동 정지:** 다른 잡이 일찍 끝나면 러너가 M3를 죽인다. 6절의 전경 대기 블록으로 대응하고, 남은 일은 night2에서 금고로 재개한다.
9. **네트워크와 디스크:** DIODE train 87 GB를 한 번은 다 받아야 한다(gzip은 시크 불가). 프로세스가 재시작되면 앞부분을 다시 받는다. 여유 디스크가 25 GB 미만이면 선택 stride를 두 배로 하고, 8 GB 미만이면 쓰기를 멈춘다.
10. **CPU 경합:** 16 vCPU를 M2, pipe_yak, DIODE 압축 해제(약 14 ms/MB), 렌더 6 procs가 나눠 쓴다. 모두 nice 10이다. M2의 데이터 로딩이 느려질 수 있다.
11. **평가 정직성:** 합성 train keep=False 선택 편향(dev는 tau 1e9로 피함). DA-V2의 DIODE 노출 여부는 확인하지 않았다. 중앙값 정렬 상수 기준선은 정답 스케일을 알고 있다는 점을 표에 명시한다.
12. **수치 추정치:** 처리량, VRAM, 스텝 시간은 E0와 CPU 실험에서 외삽했다. F0와 첫 5스텝의 측정값으로 상한을 다시 계산한다(`D1_M3` 마커로 보고).
