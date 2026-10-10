# 낮 잡 day1 (2026-10-11 08:30 → 22:00 KST) 명세: 꼬마 모델 M13 · M2 · M6 스파이크 학습

창업자 지시: "그냥 돌려 풀로". 오늘 낮 동안 GPU로 꼬마 모델을 더 만든다. **스파이크 레인**(하네스 백로그와 별개, 정식판은 나중에 MDL 단위가 만든다). 코드는 `ops/gpu/day1/` 아래에만 둔다. `buc/`, `tests/`, `ops/harness/`는 건드리지 않는다(공장장 교대조가 쓰는 경로). 단 `buc/` 코드를 **읽고 import 해서 쓰는 것**은 된다(아래 명시한 것만).

## 공통 실행 환경 (night1과 같다)
`ops/gpu/night1/SPEC.md`의 "실행 환경", "Z-Image-Turbo 사용법", "테스트" 절을 그대로 따른다. 요약과 추가 사항:
- RTX 5090 32GB, torch 2.9.1(더 새 API 금지), diffusers(ZImagePipeline), transformers, huggingface_hub, numpy, Pillow, opencv-python-headless, imageio 설치됨. 추가 패키지 금지(scipy는 있음).
- 작업 폴더 `N1_ROOT=/workspace/night1`(어젯밤 산출물이 그대로 있다), 오늘 산출물은 `$N1_ROOT/day1/<잡>/`.
- **어젯밤 자산 (GPU 디스크에 있음, 없으면 HF 금고에서 받는다):**
  - 교사 코퍼스 4,755장: `$N1_ROOT/img/{i:08d}.webp`, `$N1_ROOT/lat/{i:08d}.npy`(float16, (16,H/8,W/8), `(z−shift)·scale` 정규화), `$N1_ROOT/meta/{i:08d}.json`(prompt, seed, h, w, category). 1024² 70%, 1344×576 30%. HF 금고 `corpus/night1/gen_*.tar`(같은 레이아웃).
  - M1 TAE-T v0: `$N1_ROOT/tae/ckpt_best.pt`(HF `ckpt/night1/tae_t/ckpt_best.pt`), 모델 코드 `ops/gpu/night1/tae_train.py`(여기서 모델 클래스를 import 해 인코더·디코더로 쓴다).
- 레포 코드: 러너가 레포 브랜치 tarball을 `$N1_ROOT/repo`에 풀고 `PYTHONPATH`에 넣는다. 쓸 수 있는 것: `buc.camera.rigs`(CAM-01, 순수 numpy·scipy), `buc.camera.moves`(CAM-02). 계약서는 `ops/harness/contracts/CAM-01.md`, `CAM-02.md`.
- 비밀 `HF_TOKEN`은 값 출력 금지, 없으면 업로드 생략(`hf_token=no`만 기록). 업로드 실패는 경고만.
- 마커: 한 줄 JSON, **400자 이하**, `flush=True`. 접두어 `D1_M13`, `D1_M2`, `D1_M6`. 5분에 한 번 이상 진행 마커.
- 마감 `N1_DEADLINE`(유닉스 초, 22:00 KST = 2026-10-11T13:00:00Z = 1791723600). 마감 전 600초에 마지막 저장·평가·업로드 후 정상 종료. 재실행하면 이어서 간다.
- 업로드 대상: HF 금고 `ckpt/day1/<m13|m2|m6>/`(ckpt_best.pt, ckpt_last.pt, metrics.jsonl, 결과 그림).
- 체크포인트는 원자적 저장(임시 파일 → rename). 학습 루프의 한 스텝·한 배치 실패는 기록하고 넘어간다. 연속 100회 실패면 중단.
- GPU를 셋이 나눠 쓴다: M6 라벨 생성이 Z-Image(약 21GB)를 올리는 동안 M2·M13은 각각 VRAM 3GB 이하로(`torch.cuda.set_per_process_memory_fraction`). M6 학습 단계는 라벨 생성이 끝나고 Z-Image를 내린 뒤 시작한다.

## 잡 1 · M13 Rig-ID-S `m13_rigid.py` (가장 먼저, 짧게)
- 역할: 카메라 포즈 시퀀스(구조복원 잡음 포함)를 보고 **어떤 리그로 찍었는지**(14종) 맞힌다. 레퍼런스 영상의 카메라 무빙을 복제하는 ShotDNA의 첫 단계.
- 데이터(전부 합성, CPU 병렬): `buc.camera.rigs`의 14종 리그마다 무작위 의도(웨이포인트 2–5개, 길이 3–10 s)와 무작위 파라미터(`default_params` 주변)로 `simulate` → `frame_poses(24)` c2w 포즈열. 리그별 같은 수, 총 학습 4만, 검증 4천(시드 분리). `multiprocessing.Pool`(코어 수−2). 캐시 파일(`.npz`)로 저장해 재실행 때 다시 만들지 않는다.
- 구조복원 잡음(추출기 흉내): 전역 스케일 미지(평행이동을 경로 길이로 정규화), 회전 잡음 0.1–0.5°, 평행이동 잡음 0.5–2%, 느린 드리프트(OU), 프레임 5% 결측(보간), 시작 포즈를 원점으로.
- 입력 특징: 프레임 간 상대 회전(rot6d)·상대 평행이동(정규화)·누적 경로, 길이를 240프레임으로 자르거나 패딩+마스크.
- 모델: 약 3M. 1D conv 3층(커널 5, 채널 128→192) + transformer 인코더 4층 d192 + [CLS] → 14-way 분류 헤드(+ 보조 회귀 헤드: 리그 파라미터 정규화값, 있으면).
- 학습: AdamW 3e-4, 배치 256, bf16 autocast, 20 에폭 또는 마감 중 먼저. 5분마다 마커, 에폭마다 검증 정확도·혼동 행렬.
- 산출: `metrics.jsonl`(epoch, loss, val_acc, top2), `confusion.png`(14×14, 리그 이름 축), ckpt. 예상 GPU 15분 이하.

## 잡 2 · M2 LatentDown/Up-S `m2_latent_lift.py`
- 역할: 해상도 레벨 사이 잠재 변환. Down: `Z[16,128,128] → Z[16,64,64]` (= 교사 VAE가 절반 크기 이미지에 내는 잠재), Up: 그 역. 각 약 1M.
- 데이터: 코퍼스 이미지 x(1024² 또는 1344×576)와 저장된 잠재 z = Enc(x). GPU에서 교사 VAE로 `z_d = Enc(down2(x))`를 계산해 캐시(`$N1_ROOT/day1/m2/zd/{i}.npy`, float16, 같은 정규화). down2는 antialias 면적 평균(`F.interpolate(..., mode="area")`). VAE만 올린다(diffusers `AutoencoderKL`을 Z-Image 스냅샷의 `vae` 하위 폴더에서 로드, bf16). 평가용 400장 고정(인덱스 기준 결정론), 나머지 학습.
- 모델: 각 잠재 3×3 conv 6층 128ch(GroupNorm+SiLU, 잔차). Down은 2층째에 stride 2, Up은 nearest 2× 업샘플 후 conv. 출력에 입력의 단순 리샘플(area/bilinear)을 더하는 잔차 구조(쉬운 기준선 위에 보정만 배운다).
- 손실: 잠재 MSE + 0.1 × 이미지 공간 L1(M1 TAE-T 디코더로 둘 다 풀어서, TAE-T는 동결). 무작위 크롭 64²(Up 입력)/128²(Down 입력) 정렬.
- 학습: AdamW 1e-3, 배치 32, bf16, 마감 또는 30k 스텝. 15분마다 평가: 평가 400장에서 latent MSE와, 교사 VAE 디코드 PSNR(예측 잠재 vs 정답 잠재 디코드), **기준선(area/bilinear 리샘플) 대비 개선율**.
- 산출: metrics.jsonl, 비교 그림(원본 | 기준선 리샘플 디코드 | 우리 디코드), ckpt(Down·Up 각각). 예상 GPU 1–2시간.

## 잡 3 · M6 T-LPIPS-S `m6_tlpips.py`
- 역할: 두 이미지가 **사람 눈에 얼마나 다른지** 재는 4M 자. 금지된 VGG-LPIPS를 대신한다. 라벨은 교사(Z-Image)의 내부 특징 거리(Teacher Perceptual Distance).
- 단계 A · 쌍 만들기(CPU 병렬): 코퍼스에서 256² 무작위 크롭 x. 왜곡 y = 다음 중 1–2개 조합(강도 무작위): JPEG(q 10–90), 가우시안 블러(σ 0.5–3), 가우시안 잡음(σ 2–20/255), 다운·업 리샘플(×0.25–0.75), 언샤프 과샤프닝, 색 지터(밝기·대비·채도 ±15%), 미세 기하 왜곡(평행이동 0.5–3 px, 회전 ±2°), **M1 TAE-T 왕복**(우리 꼬마 압축기, GPU), **교사 VAE 왕복**. 학습 쌍 6만, 검증 쌍 3천 + 검증 삼조(ref, a, b) 2천.
- 단계 B · 교사 라벨(GPU, Z-Image 상주): x와 y를 교사 VAE로 인코딩 → 같은 고정 잡음·같은 시각 σ=0.3(flow matching 보간 `z_t = (1−σ)·z + σ·ε`, ε은 쌍마다 같은 시드)으로 섞음 → 빈 프롬프트 임베딩으로 트랜스포머 **한 번** 전진 → 블록 8, 15, 23의 은닉 상태를 forward hook으로 받는다(블록 리스트와 forward 시그니처는 diffusers ZImagePipeline/ZImageTransformer 소스에서 확인: 로컬에서 `pip install diffusers` 해서 소스를 읽어라. 가중치는 필요 없다). 거리 = 층별로 토큰 벡터를 채널 방향 단위 정규화한 뒤 제곱 차의 토큰 평균, 세 층 평균. 토큰 격자 거리 맵(16×16)도 저장. 배치 처리(쌍 32개 = 64장씩). 라벨은 `labels_*.npz` 조각으로 저장해 재개 가능.
- 단계 C · 학습(Z-Image 내린 뒤): 학생 = conv 5단(32→64→128→192→256ch, stride 2) + 단별 채널 정규화 차 → 1×1 선형 가중(음이 아니게 softplus) → 공간 평균 = 스칼라, 단별 맵 업샘플 합 = 거리 맵. 약 4M. 손실: 로그 거리 MSE + 0.5 × 순위 손실(배치 내 쌍 순서, margin ranking). AdamW 1e-3, 배치 64, bf16, 마감 또는 20k 스텝.
- 평가(15분마다): 검증 쌍에서 교사 거리와의 **Spearman 상관**, 검증 삼조에서 **교사와 판정 일치율(2AFC)**, 기준선 비교(픽셀 L2, SSIM 근사)와 같이 기록.
- 산출: metrics.jsonl, 산점도(교사 거리 vs 학생 거리), ckpt. 예상 GPU: 라벨 1.5–2.5시간 + 학습 1시간.

## 순서와 동시 실행 (러너가 한다)
1. M13 (CPU 데이터 생성 + 짧은 GPU 학습) — 시작 즉시.
2. M6 단계 A·B (Z-Image 상주) — 시작 즉시, M13과 동시.
3. M2 — M6 단계 B와 동시(VAE만, 3GB 이하).
4. M6 단계 C — B가 끝나면.
모든 잡은 독립 프로세스. 하나가 죽어도 나머지는 계속.

## 테스트 (이 컨테이너, CPU, 잡당 60초 안)
`ops/gpu/day1/test_<잡>.py`(pytest). GPU·네트워크·diffusers 없이: 가짜 VAE·가짜 트랜스포머(블록 리스트가 있는 작은 nn.Module)·작은 가짜 코퍼스(임시 폴더에 무작위 이미지·잠재 몇 개)를 주입. 확인: 데이터 생성 결정론과 재개, 왜곡 함수 형상·범위, 훅이 정확히 지정 층에서 잡히는지, 거리 대칭성·자기 거리 0, 모델 파라미터 수 범위(M13 2–4M, M2 각 0.7–1.5M, M6 3–5M), 한 스텝 학습 후 손실 유한·체크포인트 저장/재개, 마감 처리, 마커 400자 이하, 토큰 비노출. M13은 실제 `buc.camera.rigs`로 리그 몇 종을 짧게 시뮬레이션해 본다.
