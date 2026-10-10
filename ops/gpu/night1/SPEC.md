# 밤샘 잡 night1 (2026-10-10 → 10-11 아침) 명세

창업자 지시: 밤새 GPU를 놀리지 말고 **학습**과 **영상**을 만든다. 하네스 백로그(buc/ 아래 단위)와 별개인 **스파이크 레인**이다. 코드는 `ops/gpu/night1/` 아래에만 둔다. `buc/`, `tests/`, `ops/harness/`는 건드리지 않는다(공장장 교대조가 쓰는 경로).

## 실행 환경 (GPU 서버)
- Vast.ai RTX 5090 32GB, CUDA 12.8, 도커 이미지 `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` (**torch 2.9.1**. 더 새 API 쓰지 말 것), Python 3.11 계열, CPU 12–16, RAM 30GB+, 디스크 90GB(Z-Image 가중치 33GB가 이미 HF 캐시에 있다).
- E0가 이미 설치한 것: `diffusers`(ZImagePipeline 포함), `transformers`, `accelerate`, `safetensors`, `hf_transfer`, `sentencepiece`, `huggingface_hub`, `numpy`, `Pillow`. 러너가 추가로 `opencv-python-headless imageio imageio-ffmpeg`를 설치한다. 그 밖의 패키지는 쓰지 않는다.
- 작업 폴더: `/workspace/night1` (환경 변수 `N1_ROOT`로 바꿀 수 있게).
- 비밀: 환경 변수 `HF_TOKEN`(쓰기 권한). **값을 절대 출력·로그·파일에 쓰지 않는다.** 없으면 업로드를 건너뛰고 로그에 `hf_token=no`만 남긴다.
- HF 비공개 금고: `repo_id="sdsdsdsdsdsdsd123/buc-artifacts"`, `repo_type="model"`. 업로드는 `huggingface_hub.HfApi(token=os.environ["HF_TOKEN"]).upload_file(...)` 또는 `upload_folder(...)`. 업로드 실패는 경고만 찍고 계속 간다(잡을 죽이지 않는다).
- 로그: 표준출력 한 줄 JSON 마커로 낸다. 오케스트레이터는 `vastai logs`(마지막 수백 줄)로만 본다. 마커 접두어는 잡별로 `N1_VID`, `N1_GEN`, `N1_TAE`. 10분에 한 번 이상 진행 마커를 찍는다.
- 마감: 환경 변수 `N1_DEADLINE`(유닉스 초). 이 시각이 되면 하던 일을 정리(마지막 저장·업로드)하고 정상 종료한다.
- 모든 잡은 **중간에 죽어도 다시 실행하면 이어서** 간다(이미 만든 파일은 건너뛴다).

## Z-Image-Turbo 사용법 (E0에서 쓰는 것과 같게)
```python
from huggingface_hub import snapshot_download
from diffusers import ZImagePipeline
path = snapshot_download("Tongyi-MAI/Z-Image-Turbo", ignore_patterns=["assets/*"])
pipe = ZImagePipeline.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False); pipe.to("cuda")
img = pipe(prompt=p, height=h, width=w, num_inference_steps=9, guidance_scale=0.0,
           generator=torch.Generator("cuda").manual_seed(seed)).images[0]   # PIL
vae = pipe.vae   # diffusers AutoencoderKL 계열, 잠재 16채널, 공간 1/8
```
VAE 인코딩: 입력은 `[-1,1]` 범위 `(B,3,H,W)` bf16. `vae.encode(x).latent_dist.mean`을 쓰고, `scaling_factor`와 `shift_factor`는 `vae.config`에서 `getattr(..., 기본값)`으로 읽어 `(z - shift) * scale`로 정규화해 저장한다(기본값: scale 1.0, shift 0.0). 디코딩은 그 역. 정확한 채널 수는 코드에서 `vae.config.latent_channels`로 읽는다(하드코딩 금지).

## 잡 C · 바닥선 영상 `baseline_video.py` (가장 먼저, 짧게)
- 목적: D15 데모 이미지(거대 야크)로 **순진한 2.5D 깊이 워프** 영상을 만든다. 이건 우리 모델이 아니라 **이겨야 할 바닥선**이다. 영상 왼쪽 위에 작은 글씨로 `BASELINE (naive depth warp)`를 넣는다.
- 입력: HF 금고 `demo/d15_hero_yak.webp` (2000×861). `hf_hub_download`로 받는다. 토큰이 없으면 `--image 경로` 인자로 받은 파일을 쓴다.
- 깊이: `transformers.pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)` (Apache-2.0, 평가 전용 레인. 학습 데이터로 쓰지 않는다). 출력은 상대 역깊이(클수록 가깝다). 사람 두 명이 서 있는 영역(이미지 좌표 x 900–1130, y 640–840)의 중앙값을 거리 1.0으로 정규화하고, 하늘·먼 산은 최대 거리 40으로 자른다. `depth = 1 / (a*disp + b)` 꼴로 두 점(사람=1.0, 최소 disp 백분위 2% = 40)에 맞춘다.
- 렌더: 픽셀을 3D 점으로 올려(가로 화각 50°의 핀홀) 새 카메라로 투영. z-버퍼는 `torch` 산포(`scatter_reduce(..., "amin")`)로, 색은 z-버퍼 승자에서 가져온다. 구멍은 `cv2.inpaint`(TELEA, 반경 3)로 채운다. 깊이 불연속(이웃 깊이 비 > 1.15)은 찢어서(점만 투영하므로 자연히 찢어진다) 구멍으로 둔다. 출력 해상도 1920×826(원본 비율), 24fps, `imageio`+`imageio-ffmpeg`로 H.264 mp4 (crf 18, yuv420p).
- 샷 6개 (docs/DEMO-D15.md와 같은 이름), 사람까지 거리 d=1.0 기준:
  - S1 푸시인 8초: 카메라 z를 0 → +0.30d (전진), 이징 smoothstep.
  - S2 슬라이더 8초: x를 −0.06d → +0.06d, y는 약간 낮게(+0.01d 아래).
  - S3 아크 10초: 사람(거리 1.0, 화면 중앙 아래쪽)을 축으로 요 각도 −12° → +12°, 카메라는 항상 그 점을 바라본다.
  - S4 크레인 업 8초: y를 +0.03d(낮음) → −0.10d(높음), 약간 아래로 틸트해 사람을 화면에 유지.
  - S5 랙 포커스 6초: 카메라 고정. 초점 거리를 0.35(전경 양귀비) → 1.0(사람) → 1.6(야크 눈)으로 옮긴다. 블러는 착란원 반경 r = k·|1/z − 1/z_f| (최대 18px)로, 깊이 층 8개로 나눠 층별 가우시안 블러 후 뒤에서 앞으로 합성.
  - S6 돌리 줌 6초: 카메라가 0 → −0.4d 뒤로 빠지며 초점거리를 사람 크기가 일정하도록 키운다.
- 출력: `/workspace/night1/video/baseline_yak_S{1..6}.mp4`, 6샷 이어 붙인 `baseline_yak_all.mp4`, 샷별 첫·중간·끝 프레임 3×6 접촉 시트 `baseline_yak_sheet.jpg`. HF 금고 `video/night1/`에 업로드.
- 마커: `N1_VID {"shot":"S1","frames":192,"s":..}`, 끝에 `N1_VID_DONE {...}`.
- 예상 시간: 5분 안팎. 실패해도 다른 잡은 진행한다.

## 잡 A · 교사 코퍼스 `corpus_gen.py` (밤새)
- 목적: Z-Image로 이미지를 대량 생성하고 같은 이미지의 VAE 잠재를 짝으로 저장한다. M1 TAE-T 증류쌍, 열화 라이브러리, 평가 입력의 공용 원료.
- 프롬프트: 코드 안의 조합 문법으로 결정론적으로 만든다(시드 = 인덱스). 범주와 비율:
  - 실내 30% (서울 식당·카페·좁은 주방·거실·작업실·지하철역, 소품 많고 반사 있는 장면)
  - 풍경 20% (산·초원·바닷가·숲·도시 원경, 대기 원근)
  - 털·머리카락·가는 구조 20% (야크·하이랜드 소·양·장모견·갈대·깃털·술 장식, 역광)
  - 재질 클로즈업 15% (나무·직물·돌·녹슨 금속·유리병·물방울)
  - 초현실 시네마틱 스케일 15% (거대한 동물/사물과 아주 작은 사람, 시네마스코프 구도)
  - 수식어: 시간대·날씨·조명(역광·흐림·네온·골든아워)·렌즈(24/35/50/85mm, 얕은 심도 여부)·사진/영화 스틸 톤. 실존 인물·브랜드·로고·글자 지시는 넣지 않는다.
- 해상도: 1024×1024 70%, 1344×576(2.33:1) 30%. 9 스텝, guidance 0.
- 저장(인덱스 i, 8자리): `img/{i}.webp`(품질 92), `lat/{i}.npy`(float16, `(C,H/8,W/8)`, 위 정규화 적용), `meta/{i}.json`(prompt, seed, h, w, category, t_gen). 500장마다 tar 샤드(`shards/gen_{k:04d}.tar`, img+lat+meta)로 묶어 HF 금고 `corpus/night1/`에 올리고, 올린 샤드의 원본 파일은 디스크에 그대로 둔다(TAE 학습이 읽는다).
- VAE 인코딩은 생성 직후 같은 프로세스에서 배치로(최대 8장) 한다.
- 디스크 보호: 남은 용량이 8GB 아래로 내려가면 생성을 멈추고 `N1_GEN {"stop":"disk"}`.
- 마커: 10분마다 `N1_GEN {"n":..,"rate_img_per_min":..,"vram_GB":..,"last_shard":..}`.
- TAE 학습과 같은 GPU를 나눠 쓴다. Z-Image 상주 약 20GB, 학습은 4GB 이하.

## 잡 B · M1 TAE-T 스파이크 학습 `tae_train.py` (A와 동시에, 밤새)
- 목적: 교사 VAE를 흉내 내는 작은 자동인코더(합계 약 2.5M 파라미터). 인코더: RGB → 잠재(C채널, 1/8), 디코더: 잠재 → RGB. 하네스의 MDL-01이 나중에 정식으로 만들 때 출발점과 기준 곡선이 된다.
- 구조(권장): 인코더 = conv3×3 스템(32) → [다운샘플 stride2 + 잔차블록 2개] × 3 (채널 32→64→96→128) → conv1×1 → C. 디코더 = 대칭(conv1×1 C→128 → [잔차블록 2개 + nearest 업샘플 + conv] × 3 → conv3×3 → 3, tanh 없이 출력 후 clamp는 평가 때만). GroupNorm(8)+SiLU. 파라미터 수를 시작할 때 출력하고 2.0M–3.2M 범위에 들게 폭을 맞춘다.
- 데이터: 잡 A가 쓴 `img/`·`lat/` 짝. 1000쌍이 생길 때까지 기다린다(30초마다 확인). 매 에폭 파일 목록을 새로 읽는다. 마지막 64쌍은 평가용으로 고정(인덱스 기준으로 결정론적 선택, 학습에서 제외).
- 크롭: 256×256 랜덤(8의 배수 정렬)과 그에 맞는 잠재 32×32. 좌우 뒤집기.
- 손실: `L_enc = MSE(E(x), z_t)`, `L_dec = L1(D(z_t), x) + 0.5·L1(D(E(x)), x) + 0.1·L1(∇D(z_t), ∇x)`(소벨 기울기). 합 = `L_enc + L_dec`.
- 최적화: AdamW lr 1e-3, wd 1e-4, 배치 32, 1000스텝 워밍업 후 코사인(전체 길이는 마감까지 남은 시간으로 추정하지 말고 상수 200k로 둔다), bf16 autocast, grad clip 1.0. DataLoader worker 4.
- 평가: 15분마다 평가 64쌍 전체 해상도에서 `PSNR(D(z_t), x)`, `PSNR(D(E(x)), x)`, `latent MSE` 계산. 비교 이미지 4장(원본 | 교사 잠재→우리 디코더 | 우리 인코더→우리 디코더)을 `tae/samples/step_{s}.jpg`로 저장.
- 체크포인트: 15분마다 `tae/ckpt_last.pt`(모델·옵티마이저·스텝, 원자적 저장: 임시 파일 후 rename), 평가 최고 PSNR이면 `tae/ckpt_best.pt`. HF 금고 `ckpt/night1/tae_t/`에 last·best·최신 샘플·`metrics.jsonl` 업로드.
- 재개: `ckpt_last.pt`가 있으면 이어서 학습.
- 마커: 5분마다 `N1_TAE {"step":..,"loss":..,"l_enc":..,"l_dec":..,"img_per_s":..}`, 평가 때 `N1_TAE_EVAL {...}`.

## 테스트 (이 컨테이너, CPU)
- 각 잡에 `test_<잡>.py`(pytest)를 둔다. GPU·네트워크·diffusers·transformers 없이 돌아야 한다: 무거운 의존성은 함수 인자로 주입하거나 가짜 객체로 바꿔 끼운다(예: 가짜 파이프라인이 무작위 PIL 이미지와 가짜 VAE를 돌려준다, 가짜 깊이 함수).
- 확인할 것: 프롬프트 문법의 결정론과 범주 비율, 파일 레이아웃과 재개(두 번 돌려도 중복 없음), 샤드 tar 내용, 마감 처리, TAE 파라미터 수 범위·크롭 정렬·한 스텝 학습 후 손실 유한·체크포인트 저장/재개, 카메라 궤적 함수(샷별 시작·끝 자세), 투영/z-버퍼(앞의 점이 이긴다), 짧은 mp4(작은 해상도, 몇 프레임) 실제 인코딩.
- 실행: `python3 -m pytest -q ops/gpu/night1` 이 60초 안에 통과해야 한다.
