> **출처 메모.** 2026-10-10 기준 웹 조사로 만든 2×T4 하드웨어·소프트웨어·이미지 모델·소형 모델 선례 보고서다. [검증]/[2차]/[추정] 표기를 지킨다. 이미지 모델의 T4 속도는 대부분 FLOPs 추정치이므로 Day-1 실측으로 교체한다.

---

# 리서치 보고서: 2×T4 + 수십M from-scratch 제약 검증 (기준일 2026-10-10)

**표기.** [검증]은 1차 또는 공식 출처로 확인한 값이다. [2차]는 2차 출처만 있는 값이다. [추정]은 내가 FLOPs와 대역폭으로 계산한 값으로, 실측이 필요하다. [신규]는 이 제약 때문에 새로 나온 설계 아이디어다.

---

## 0. 요약

- **수십M 모델 학습은 병목이 아니다.** 50M 모델을 Chinchilla 비율로 학습해도 2×T4에서 약 1시간이다. 200 tok/param으로 과학습해도 약 1일이다.
- **진짜 병목은 두 가지다.**
  - 빌린 이미지 교사의 forward 1회 비용. 프로젝트의 진짜 화폐다.
  - 고전 파이프라인(SfM, 렌더)의 처리량.
- **T4 함정은 다섯 가지다.**
  - BF16, TF32, FP8이 없다.
  - Triton ≥3.3은 텐서코어를 쓰지 않는다(FMA 폴백).
  - FA2/3/4가 없다.
  - 70W 전력 제한 때문에 클럭이 떨어진다.
  - 최신 오픈 이미지 모델이 대부분 bf16 네이티브다.
- **숨은 무기는 세 가지다.**
  - INT8/INT4 텐서코어. 실측 INT4 114 TOPS로 FP16의 2.7배다.
  - RT 코어 40개.
  - CUDA 13과 Warp가 아직 sm_75를 지원한다.

---

## 1. T4 하드웨어와 소프트웨어 현실

### 1.1 하드웨어 스펙 검증

| 항목 | 브리핑 값 | 검증 결과 | 출처 |
|---|---|---|---|
| FP16 텐서 | 65 TFLOPS | [검증] "Mixed Precision (FP16/FP32) 65 TFLOPS". Tesla급이라 FP32 누산도 full-rate | [T4 datasheet](https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf) |
| INT8 / INT4 | 130 / 260 TOPS | [검증] | 위 datasheet |
| FP32 | 8.1 TFLOPS | [검증] | 위 datasheet |
| 메모리 | 16GB GDDR6, 320 GB/s | [검증] 대역폭은 문서마다 다르다. 메인 datasheet는 300 GB/s, 가상화 datasheet와 Citadel 논문은 320 GB/s(메모리 클럭 5,001 MHz). L2는 4 MiB | [vGPU datasheet](https://www.Nvidia.com/content/dam/en-zz/Solutions/design-visualization/solutions/resources/documents1/Datasheet_NVIDIA_T4_Virtualization.pdf), [Citadel T4 microbenchmark](https://arxiv.org/pdf/1903.07486) |
| 코어 | — | [검증] CUDA 2,560, Tensor 320, **RT 40**, 70W 패시브 | 위 datasheet들 |
| **실측 GEMM** (1590 MHz 고정) | — | [검증] HGEMM(cuBLAS) **41.6 TFLOPS**(피크의 64%), INT8 cuBLASLt **74.9 TOPS**, INT4 CUTLASS **114.4 TOPS**, SGEMM 7.17 TFLOPS | [Citadel](https://arxiv.org/pdf/1903.07486) Table 4.3 |
| **전력 쓰로틀** | — | [검증] cuBLAS GEMM에서 **power-limit throttling이 아주 일찍 걸린다**. 행렬이 클수록 클럭이 더 떨어진다. 85°C에서 thermal throttling이 겹쳐 클럭이 급락한다. 최고 클럭은 처음 몇 초만 유지된다 | 위 논문 §4.5 |

**학습 계획치 보정 [추정].** 실측 최고가 41.6이고 지속 부하에서는 쓰로틀이 걸린다. 비GEMM 연산 비중도 있다. 그래서 학습 실효치는 **장당 15–20 TFLOPS(명목 대비 MFU 23–31%)**로 잡을 것을 권한다. 브리핑의 20–25는 상단이다.

| 운용 모드 | 일당 FLOPs | 90일 FLOPs | 90일 T4-h |
|---|---|---|---|
| 상시 2×T4 | 2.6–3.5×10¹⁸ | **2.3–3.1×10²⁰** | 4,320 |
| Kaggle 쿼터 (주 30h × 2장, 쿼터 계산 방식은 미확인) | — | 4.2–5.6×10¹⁹ | 약 780 |
| v1 계획 (75–90k H100-h) | — | 약 1.1–1.3×10²³ | (H100 1장 ≈ T4 20–25장) |

v1 대비 약 **1/400**이다.

### 1.2 소프트웨어 스택

| 요소 | sm_75(T4) 상태 | 출처 |
|---|---|---|
| **PyTorch** | [검증] CUDA 13.x 빌드는 Turing(7.5)부터 Blackwell까지 계속 지원한다. 2.15부터 없어지는 CUDA 12.6 빌드는 Maxwell, Pascal, Volta에만 영향을 준다. 2.14가 CUDA 12.x 마지막 릴리스다(2026-09-04 공지). **T4는 지원 하한선 바로 위에 있다** | [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) |
| BF16 | [검증] 네이티브 지원 없음. 에뮬레이션으로 돌아가지만 속도 이득이 없다(V100 사례에서 FP32와 비슷하거나 느림) | [forum1](https://discuss.pytorch.org/t/bfloat16-native-support/117155), [forum2](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629) |
| TF32, FP8 | [검증] 없음. TensorRT 10.x 지원표에서도 T4는 FP32/FP16/INT8만 있다 | [TRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html) |
| **SDPA** | [검증] Turing에서 PyTorch attention은 xformers 계열 Memory-Efficient 커널을 쓴다. flash 백엔드는 사용 불가 | [flash-attention-turing README](https://github.com/ssiu/flash-attention-turing) |
| FlashAttention | [검증] FA2는 Ampere/Ada/Hopper, FA3는 Hopper, FA4는 Hopper/Blackwell만 지원한다. 공식 레포가 Turing 사용자를 **flash-attention-turing**으로 안내한다. 이 포크는 fwd+bwd, head dim 64/96/128, causal/GQA/varlen을 지원하고 dropout은 지원하지 않는다. T4에서 SDPA 대비 **fwd 최대 2.19배, bwd 1.35–1.51배**. FA1은 Turing을 지원하지만 구버전 문서에 "head dim > 64 backward는 A100 필요"라는 제한이 있다 | [FA README](https://github.com/Dao-AILab/flash-attention), [ssiu](https://github.com/ssiu/flash-attention-turing), [FA 0.2.7](https://pypi.org/project/flash-attn/0.2.7) |
| xformers | [2차] 문서상 mem-efficient attention은 cc > 6.0을 지원한다. 최신 휠의 sm75 포함 여부는 미확인 | [xformers ops 문서 미러](https://https-facebookresearch_github_io.proxy.bugsmash.io/xformers/components/ops.html) |
| **Triton** | [검증] PR #5066(2024-11 머지): **cc < 80에서 tl.dot이 MMA를 쓰지 않고 FMA로 폴백**한다. README 지원 하드웨어는 "CC 8.0+"로 바뀌었다. 메인테이너는 "best effort, 테스트하지 않음"이라고 했다. Triton 3.3부터 적용 | [PR #5066](https://github.com/triton-lang/triton/pull/5066), [Triton README](https://github.com/triton-lang/triton), [triton-windows](https://github.com/woct0rdho/triton-windows) |
| torch.compile | [추정] Inductor의 원소별·리덕션 퓨전은 쓸 만하다. matmul은 기본값이 cuBLAS라 괜찮다. 다만 max-autotune Triton GEMM 템플릿, **FlexAttention**, 커스텀 Triton 커널은 텐서코어를 쓰지 못하거나 테스트되지 않은 경로다 | 위 PR에서 추론 |
| TensorRT | [검증] 10.x는 SM 7.5 이상을 지원하고 T4에서 FP16/INT8을 쓴다. TensorRT-RTX는 Turing을 기본 포함하지 않는다(별도 엔진 권장) | [TRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html), [TRT-RTX](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html) |
| **Nunchaku (SVDQuant W4A4)** | [검증] v0.2.0에서 20-series 호환, v1.2.0에서 **20-series INT4**를 지원한다. FLUX.1, SANA, Qwen-Image, **Z-Image-Turbo(4-bit, 2025-12-20)**를 지원하고 `flux.1-dev-turing.py` 예제가 있다. T4 실측은 없음 | [nunchaku](https://github.com/nunchaku-ai/nunchaku) |
| FP16 학습 안정성 | [검증] muP 논문의 GPT-3 6.7B 재현은 FP16에서 발산이 잦아 FP32로 학습했다. Wortsman et al.은 attention logit 성장과 output logit 발산이 **소형 모델에서 높은 LR로 재현되고, 대형 모델 처방이 그대로 통한다**고 보고했다 | [muP](https://arxiv.org/abs/2203.03466), [Wortsman](https://arxiv.org/abs/2309.14322) |

### 1.3 렌더와 3D 최적화

| 요소 | 상태 | 출처 |
|---|---|---|
| gsplat | [검증] CUDA로 첫 실행 때 JIT 빌드. 원본 3DGS 대비 **학습 메모리 최대 4배 절감, 15% 빠름**. MCMC, absgrad, 안티앨리어싱, 배치·멀티 GPU 지원. **3DGUT 통합**(`with_ut`, rolling shutter, 왜곡·fisheye·FTheta). 최소 CC 문서는 없다 | [gsplat docs](https://docs.gsplat.studio/main/), [NVIDIA blog](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/), [1.5.3](https://radiancefields.com/gsplat-1-5-3-released-by-nerfstudio) |
| 3DGRUT | [검증] CUDA 11.8–13.0 지원. "3DGRT requires dedicated ray-tracing hardware and remains slower than 3DGS". 벤치는 RTX 5090/L40 기준 | [3dgrut](https://github.com/nv-tlabs/3dgrut) |
| OptiX | [검증] 현재 다운로드 페이지 기준 Turing 이상 지원(T4가 하한선). OptiX 9.0은 R570+, 9.1은 R590+ 드라이버 필요 | [OptiX DL](https://developer.nvidia.com/designworks/optix/download), [9.0 release](https://forums.developer.nvidia.com/t/optix-9-0-release/322842) |
| **3DGS 소요 시간** | [검증] RTX 4090, Mip-NeRF360 평균: 3DGS 20.9분(2.63M Gaussians, 피크 9.89GB), DashGaussian 6.35분, **FastGS 1.93분(0.40M Gaussians, 피크 5.21GB)**. Colab T4 원본 3DGS 7K iter는 15–30분으로 보고됨 | [FastGS](https://arxiv.org/html/2511.04283v2), [DEV Colab](https://dev.to/nomurasan/building-a-3d-scene-from-30-photos-getting-gaussian-splatting-running-on-colab-4n5p) |
| T4 환산 | [추정] 4090 대비 FP32는 10배, 대역폭은 3.15배 차이 → 5–8배 느리다고 본다. **FastGS류 10–16분/장면**, 원본 3DGS 30K는 1.7–2.8h. 메모리는 둘 다 16GB 안에 들어간다 | — |
| tiny-cuda-nn | [2차] fully-fused MLP는 공유 메모리를 많이 쓴다. README는 "2080 Ti 이상에서 동작할 가능성"이라고 쓴다. T4도 같은 sm_75이지만 실측은 없다. 대안은 CutlassMLP | [tcnn](https://github.com/NVlabs/tiny-cuda-nn) |

### 1.4 물리 시뮬레이터와 절차적 생성

| 요소 | 상태 | 출처 |
|---|---|---|
| NVIDIA Warp | [검증] CUDA 12.x 빌드는 sm_52 이상, **CUDA 13.x 빌드는 sm_75 이상**. fp16 atomic은 cc 7.0 이상 | [Warp compat](https://nvidia.github.io/warp/latest/user_guide/compatibility.html) |
| Newton / MuJoCo Warp | [2차] Newton은 warp.sim을 일반화한 것이고 MuJoCo Warp가 주 백엔드다. 별도 최소 CC 문서는 없어서 Warp를 따른다고 추론 | [newton PyPI](https://pypi.org/project/newton/) |
| Genesis / Taichi | 미확인. Genesis 문서에는 CUDA 백엔드만 있고 최소 CC 표기가 없다. 자체 컴파일러 "Quadrants"로 전환했다는 언급(채용공고)이 있다 | [Genesis install](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) |
| Blender Cycles OptiX | [2차] 3.0 매뉴얼 기준 OptiX는 CC 5.0 이상이고 RTX(Turing 이상)가 권장 등급. 5.x 요구사항은 미확인 | [Blender manual](https://docs.blender.org/manual/en/3.0/render/cycles/gpu_rendering.html) |
| Infinigen | [검증] Indoors 단일 방(fast_solve)은 **CPU로 약 8–13분**. 원본 논문 기준 1080p 이미지 2장에 평균 3.5h(그중 GPU 렌더 약 1h) | [HelloRoom](https://infinigen.cs.princeton.edu/docs/latest/HelloRoom.html), [Infinigen](https://arxiv.org/pdf/2306.09310) |

---

## 2. T4 두 장에서 쓸 이미지 생성 모델 후보

### 2.1 미감 순위 (Artificial Analysis T2I arena, 2026-10-10 조회) [검증]

상용 1위는 GPT Image 2.5 Sunburst로 Elo 1198이다.

| 모델 | Elo | 파라미터 | 라이선스 | 텍스트 인코더 / VAE | 비고 |
|---|---|---|---|---|---|
| Qwen-Image-2.1 | 1035 | 7B(생성부) | **비상업 전용** | — | 탈락 |
| Qwen Image Max 2512 | 999 | 약 20B | Apache 2.0 | Qwen2.5-VL급 | fp16 40GB라 INT4 필수. Nunchaku 지원 |
| **Z-Image Turbo** | **941** | **6.15B** S3-DiT(30층, d=3840) | **Apache 2.0** | Qwen3-4B / Flux 계열 VAE(16ch) | 8 NFE, CFG 없음, 공식 dtype은 bf16, "16GB에 들어감" |
| FLUX.2 klein 9B | 941 | 9B | FLUX NCL(비상업) | Qwen3 8B | 탈락 |
| ERNIE Image (Turbo) | 912 (924) | 8B | Apache 2.0 | — | 24GB GPU 권장 |
| **FLUX.2 klein 4B** | **863** | 4B | **Apache 2.0** | Qwen3-4B / FLUX.2 VAE(32ch, [2차]) | 4-step 증류, T2I·I2I·멀티 레퍼런스 편집, 약 13GB |
| FLUX.1 dev / SD3.5 Large | 840 | 12B / 8.1B | NC / Community | — | 크고 느림 |
| FLUX.1 schnell | 803 | 12B | Apache 2.0 | T5-XXL | klein 4B에 품질·속도 모두 밀림 |
| Lumina Image v2 | 782 | 2.6B | Apache?(출처 상충) | Gemma2-2B | — |
| SD3.5 Medium | 760 | 2.5B | Stability Community(매출 $1M 미만 무료, [2차]) | — | — |
| Sana Sprint 1.6B | 754 | 1.6B | NSCL v2 vs Apache(출처 상충) | Gemma2-2B / DC-AE f32 | H100 0.1s, 4090 0.31s |
| **SDXL 1.0** (+Lightning) | **678** | UNet 2.6B | OpenRAIL++(Lightning도 OpenRAIL++, **DMD2는 CC BY-NC-SA**) | CLIP-L+bigG / SDXL VAE | fp16 경로 검증됨(fp16-fix VAE) |

출처: [AA leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image), [pinggy 정리](https://pinggy.io/blog/best_free_open_source_ai_image_generators_to_self_host/), [Z-Image-Turbo 카드](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [Z-Image 논문](https://huggingface.co/papers/2511.22699), [FLUX.2 klein](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence), [Comfy klein](https://docs.comfy.org/tutorials/flux/flux-2-klein), [Sana](https://github.com/NVlabs/Sana), [Sana-Sprint 카드](https://replicate.com/nvidia/sana-sprint-1.6b/readme), [SD3.5M](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), [Lightning 라이선스(2차)](https://www.promptlayer.com/models/sdxl-lightning), [DMD2 라이선스(2차)](https://huggingface.co/Zuntan/NoobHyperDmd), [sdxl-vae-fp16-fix](https://www.promptlayer.com/models/sdxl-vae-fp16-fix)

**Elo로 본 기대 승률.**

| 대결 | 기대 승률 |
|---|---|
| Z-Image vs 상용 1위 | 약 19% |
| Z-Image vs klein 4B | 61% |
| Z-Image vs SDXL | 82% |
| Qwen-2512 vs Z-Image | 58% |

**교사가 프레임 미감의 상한이다.** 이 아레나는 비디오 모델 프레임과 직접 비교가 안 된다(아래 미확인 목록 참고).

### 2.2 T4 속도 [추정, 실측 필요]

T4 이미지 생성 실측 공개 자료는 거의 없다. 유일한 실측 앵커는 Fooocus(SDXL) Colab T4에서 이미지 2장에 약 74초라는 보고다([출처](https://www.stablediffusiontutorials.com/2023/12/install-run-fooocus-on-pc-colab.html)).

나머지는 다음 식으로 계산했다.
- forward FLOPs ≈ 2·P·N(토큰 수) + attention
- T4 fp16 실효 15–20 TFLOPS
- 참고로 Sana 공식 지연(A100 추정)은 0.6B 0.9s, 1.6B 1.2s, FLUX-dev 23s다.

| 모델 | 1024² 토큰 | forward당 FLOPs | T4 forward | 1024² 장당 | 512² 장당 | VRAM 배치 |
|---|---|---|---|---|---|---|
| Z-Image-Turbo (8 NFE) | 약 4.3K | 약 61 TF | 3.1–4.1s | **26–35s** | 6–9s | DiT fp16 12.3GB를 GPU0에 단독. TE 8GB는 GPU1 또는 CPU에서 1회 실행 후 캐시. INT4로 약 3.5GB |
| FLUX.2 klein 4B (4 step) | 약 4.3K | 약 40 TF | 2.0–2.7s | **9–12s** | 2–3s | DiT 약 8GB + TE 분리 |
| SDXL-Lightning (4 step) | latent 128² | — | 약 0.55s(Fooocus 역산) | **3.5–5s** | (SDXL은 512에서 품질 저하) | 전체 약 7GB, 1장으로 충분 |
| Sana-Sprint 1.6B (1–2 step) | 1K (DC-AE f32) | 약 4 TF | 0.2–0.3s | **1–2s** (DC-AE 디코드 포함) | <1s | 3.2GB + Gemma 5GB |
| FLUX.1 schnell (4 step) | 4.6K | 약 125 TF | 6–8s | 25–33s | — | fp16 24GB라 두 장에 분할하거나 INT4 |
| Qwen-Image-2512 (Lightning 8) | 4.3K | 약 190 TF | 10–13s | 80–100s(fp16 환산) | 20–25s | INT4 필수 |

**메모리 배치 원칙 [추정].**
- NVLink가 없고 PCIe Gen3이라 tensor parallel은 비효율적이다.
- 역할을 나눈다. **GPU0 = 교사 DiT 전용. GPU1 = VAE, 세계(Gaussians), 렌더러, 수십M 모델, 학습.**
- 텍스트 인코더는 prompt tree당 1회만 돌린다.

### 2.3 조건부 생성과 편집 지원

| 모델 | 지원 내용 | 출처 |
|---|---|---|
| Z-Image-Turbo | Fun-ControlNet-Union(Canny/HED/**Depth**/Pose/MLSD, 1328px 학습) | [comfyui-wiki](https://comfyui-wiki.com/en/news/2025-12-02-z-image-turbo-controlnet-union) |
| FLUX.2 klein | I2I와 멀티 레퍼런스 내장 | — |
| Sana | ControlNet(2025-02). Sprint+ControlNet은 H100 0.25s | — |
| SDXL | depth/canny ControlNet과 T2I-Adapter 생태계가 가장 두껍다 | — |

모두 오픈 가중치라 **노이즈 레벨에 접근할 수 있다**(SDEdit 가능).

주의할 점이 있다. Turbo, Lightning, Sprint 같은 증류 모델은 **학습된 소수의 t에서만 신뢰할 수 있다**. WSSC의 임의 σ 창 [t_start, t_end]에서는 가장 가까운 학습 t로 스냅해야 한다. 이건 실측 항목이다.

### 2.4 교사 forward 회계 (v1 WSSC 기준) [추정]

v1 기준 세계 하나당 forward 수는 레벨 4–6 × 뷰 200–400 × K 2–4 = **1.6K–9.6K**다.

| 교사와 해상도 | T4 1장 기준 세계당 시간 |
|---|---|
| Z-Image 1024² | 1.4–11 h |
| Z-Image 512² | 21분–2.8 h |
| klein 4B 512² | 13분–1.9 h |
| SDXL-Lightning 1024² | 15분–1.5 h |
| Sana-Sprint | 5–48분 |

v1 표의 "방 규모 4–12 H100-h"는 **100–300 T4-h**에 해당해 그대로는 불가능하다. 뷰 수, 해상도, K를 합쳐 약 100배 줄여야 한다. VAE encode/decode도 매 스텝 뷰마다 들어가므로 비중이 크다(미측정).

---

## 3. 수십M 파라미터 from-scratch 선례

환산 가정 [추정]: 2080 Ti ≈ T4의 1.5–2배, V100 ≈ T4의 2–3배, H100 ≈ T4의 20–25배.

| 모델 | 파라미터와 아키텍처 | 데이터 (사전학습 없이) | 원 컴퓨트 | 2×T4 환산 [추정] | 출처 |
|---|---|---|---|---|---|
| **RAFT** | **5.3M**(업샘플러 제외 4.8M), small 1.0M. 반복 GRU + 4D correlation. 추론 211 GFLOPs | **합성만 사용**(FlyingChairs+Things)해 from scratch | **2×2080Ti**, 200K iter, 배치 12. mixed precision 1-GPU 스크립트 있음. 1080Ti에서 1088×436 10fps | 거의 동급, 시간 1.5–2배 | [RAFT](https://ar5iv.labs.arxiv.org/html/2003.12039), [torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_large.html), [repo](https://github.com/princeton-vl/RAFT) |
| **Real-ESRGAN** | RRDBNet 약 16.7M([2차]) | **순수 합성 2차 열화**(HR 이미지만 있으면 됨) | 4×V100, ESRNet 1M + GAN 400K iter, 배치 48 | ESRNet만 2–3주. 축소 스케줄 필요 | [paper](https://arxiv.org/pdf/2107.10833) |
| **MDM** | 8층 Transformer, d=512, T=1000 cosine (파라미터 수는 논문에 없음) | 모션캡처. 텍스트 조건은 **frozen CLIP**(우리 규칙에서는 교체 대상) | **RTX 2080 Ti(Turing) 1장 약 3일**, 500K step, 배치 64 | 2×T4로 약 2.5–3일 | [MDM](https://ar5iv.labs.arxiv.org/html/2209.14916) |
| **MVSplat** | **12.0M**, plane-sweep cost volume, feed-forward 3DGS | RealEstate10K 67,477 장면 | 22fps (백본 초기화가 from scratch인지 미확인) | — | [MVSplat](https://arxiv.org/abs/2403.14627) |
| TinyStories | 1M–33M(35M) | 합성 동화 텍스트 | V100 1장 ≤30h | ≤1.3–1.9일 | [TinyStories](https://arxiv.org/abs/2305.07759) |
| modded-nanoGPT | 124M, 목표 val loss 3.28 | FineWeb, 330M 토큰 미만 | **기록 #92: 0.665분(2026-08-30, 8×H100)**. llm.c 기준선은 45분. Muon, FP8, FA3 사용 | 이론상 약 2h(2.5×10¹⁷). FP8/FA3가 없으니 실제로는 수 시간 | [repo](https://github.com/KellerJordan/modded-nanogpt) |
| airbench | 소형 CNN | CIFAR-10 | A100에서 94% 3.29s(논문), 이후 2.6s | 분 단위 | [paper](https://arxiv.org/abs/2404.00498) |
| DiT-S/2 | 33M, 6.06 Gflops | ImageNet-256 | 400K × 256 | 약 13–17h | [DiT](https://arxiv.org/pdf/2212.09748) |
| EDM2-XS | 125M | ImageNet-512, 클래스 조건 | — | — | [EDM2](https://arxiv.org/pdf/2312.02696) |
| MicroDiT | 1.16B | 3,700만 장 + 합성 이미지 | **$1,890, 8×H100 2.6일**, COCO FID 12.7 | **약 210–260일 = 우리 90일 예산의 2–3배** | [MicroDiT](https://arxiv.org/abs/2407.15811) |
| TAESD | enc/dec 각각 약 1.2M([2차]), SD VAE는 34M/49M | **SD VAE에서 증류** | — | — | [diffusers](https://huggingface.co/docs/diffusers/v0.36.0/api/models/autoencoder_tiny) |
| Depth Anything V2-S | 24.8M | 대형 교사 의사라벨. 저자들은 "**합성만으로 학습하면 일반화 문제 심각**"이라고 썼다 | — | (구조 참고용. 사용 금지) | [DA-V2](https://arxiv.org/pdf/2406.09414) |
| muP | 40M 프록시로 6.7B에 전이, 튜닝 비용은 사전학습의 7% | — | — | — | [muP](https://arxiv.org/abs/2203.03466) |

생성 품질 지표:
- DiT-S/2는 FID **68.4**로, **범용 소형 생성기는 약하다**.
- EDM2-XS는 FID 3.53, 가이던스 사용 시 2.91로, 좁은 도메인에서는 강하다.

**정리.**
- 수십M 모델은 **좁고 인수분해된 연산자**에서 강하다. 광류, SR, 모션, 기하, feed-forward splat이 그렇다.
- 데이터는 **합성 열화**나 **고전 알고리즘 의사정답**으로 공짜 라벨을 만들 수 있을 때 강하다.
- 오픈 도메인 T2I를 from scratch로 하는 것은 2×T4로는 불가능하다. 그래서 이미지 모델을 빌리는 결정이 맞다.

---

## 4. Kaggle과 Colab (2026)

**Kaggle**
- [검증] 세션은 CPU/GPU 12h, TPU 9h(2026-09-21 기준 문서).
- [2차] 주간 GPU 쿼터는 공식 수치가 없다. 계정 설정에 표시되고 변동하며, 사용자들은 약 30h로 보고한다.
- GPU는 P100 1장 또는 T4×2(일부 출처는 beta로 표기).
- Colab Pro/Pro+를 연동하면 주 +15h/+30h.
- 디스크 20GB는 [2차].
- T4×2가 쿼터를 1배로 소모하는지 2배로 소모하는지, RAM, CPU 수는 미확인.
- 출처: [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [aimultiple](https://aimultiple.com/free-cloud-gpu)

**Colab 무료**
- 한도는 비공개이고 최대 약 12h. 사용자 상호작용이 끊기면 VM이 삭제된다.
- 무료 등급에서는 SSH나 웹 UI 사용 불가.
- [2차] Colab Enterprise T4는 시간당 $0.42(2026-09).
- 출처: 위와 같음

**함의.** Kaggle 모드는 상시 대비 약 18%다. 9–12h 단위 체크포인트와 재개가 필수이고, 장시간 솔버는 세션을 넘겨 resumable해야 한다.

---

## 5. 고전 SfM/MVS 처리량

- **COLMAP 4.0(2026-03-14)에 GLOMAP이 통합**됐다(`global_mapper`).
- **4.1(2026-06)에서 GPU BA "Caspar"가 추가**됐다. Ceres CUDA 대비 중·대규모 문제에서 10–100배 빠르다고 한다.
- 4.0에 **ALIKED/LightGlue(ONNX)**가 추가됐는데, 학습 모델이라 우리 규칙상 시스템 경로에서는 금지다. **SIFT만 쓴다.** 최신은 4.2.1(2026-09-29). [changelog](https://colmap.github.io/changelog.html)

**GLOMAP 논문의 평균 매핑 시간(초, 매칭 제외)** [검증], [GLOMAP](https://arxiv.org/html/2407.20219):

| 데이터셋 | COLMAP | GLOMAP |
|---|---|---|
| ETH3D SLAM | 1115 | 134 |
| IMC 2023 | 4051 | 497 |
| MIP360 | 346 | 221 |
| LaMAR | 354,660 | 12,405 |

**추출과 매칭 비용.**
- KITTI 100프레임: GPU SIFT 추출 2.0s, sequential 매칭 3.7s. 전체 346s의 대부분은 매핑이다([CuSfM](https://arxiv.org/pdf/2510.15271)).
- 4만 장: GPU 추출 약 1h, 매칭 약 2h([H-3DGS](https://arxiv.org/pdf/2406.12080)).
- [추정] 비디오 장면 하나(200–300프레임)에 sequential + GLOMAP 조합이면 **수 분이고 CPU가 병목**이다. Kaggle의 적은 vCPU가 실제 제약이 될 수 있다.

---

## 6. v1 구성요소별 영향 (리서치 근거 기반 권고)

| v1 요소 | 판정 | 근거와 대체 |
|---|---|---|
| S0 World Program (LLM) | **제거** | 대형 LLM 금지. 대체안은 교사 텍스트 인코더 임베딩 + 절차적 데이터(Infinigen)로 학습한 수십M 레이아웃 모델. 텍스트 인코더(Qwen3-4B)를 LLM 용도로 쓰는 것은 규칙 해석이 필요하다 |
| S1 닫힌 SOTA art director, POLISH 닫힌 API dreamer | **제거** | 시스템 경로 금지. 평가에만 사용 |
| S2 부트스트랩(DA3/VGGT/AnySplat), TRELLIS/Hunyuan3D | **제거** | COLMAP 4(SIFT) + GLOMAP + Caspar BA + MVS 의사정답으로 학습한 자체 수십M 깊이/splat 모델(MVSplat 12M급). MegaDepth식 접근 |
| Dreamer 보정용 RAFT/UFM flow | **대체** | 자체 RAFT-small(1.0M)을 합성 데이터로 from scratch 학습(선례 있음) |
| 3DGRT 2차 광선 | **연기** | RT 코어는 있지만 3DGS보다 느리다 |
| 8K·120fps 기본 출력 | **연기** | 렌더 예산 |
| WDM student (12B 교사 + fake-score LoRA 64–128) | **축소 또는 연기** | 교사 두 개(real/fake)의 fp16 가중치만 24.6GB(Z-Image 기준)이고 활성 메모리도 크다. 512² 이하, 그래디언트 체크포인팅, 두 장 분할로만 가능 [추정] |
| Phase 2 비디오 모델 증인 | **제거** | 물리 시뮬(Warp/Newton) + 자체 모션 모델(MDM 선례: 2080Ti 1장 3일) |
| GeCo (VGGT 사용) | 유지 | 평가 전용으로 허용 |
| gsplat/3DGUT, graph cut, WNF(Philox) | **유지** | 학습이 없는 구조이고 sm_75에서 돈다 |

**[신규] 아이디어 (리서치에서 파생)**

1. **교사 forward 회계.** 모든 실험과 게이트를 "T4 교사-forward" 단위로 예산화한다.
2. **head_dim=64 규약.** FA1 backward 제약과 flash-attention-turing에 맞춘다.
3. **텍스트 임베딩 캐시와 CPU 오프로드.** 8GB 회수.
4. **빌린 VAE를 TAESD식 1–2M 디코더로 증류.** VAE는 이미지 모델의 일부라 허용된다. 프리뷰와 대량 판정을 가속한다.
5. **INT4 교사(Nunchaku, Turing INT4).** INT4 실측 114 TOPS는 FP16 41.6의 2.7배다.
6. **Small-scale proxy 원칙(Wortsman) + muP**를 학습 레시피 기본값으로 둔다. QK-norm, z-loss, fp32 softmax/LN, GradScaler 포함.
7. **FastGS급 0.4M Gaussians를 장면 예산의 기본값**으로 둔다. T4에서 렌더와 메모리의 균형점이다.

---

## 7. 2×T4 + 수십M from-scratch 제약에서 설계자가 알아야 할 결정적 사실 Top 15

1. **실효 컴퓨트.** T4 HGEMM 실측 최고는 41.6 TFLOPS(피크의 64%)이고 70W 전력 제한으로 지속 부하에서 클럭이 떨어진다. 계획치는 장당 15–20 TFLOPS다. **90일 2.3–3.1×10²⁰ FLOPs**(v1의 약 1/400). Kaggle 모드는 4–6×10¹⁹.
2. **수십M 학습은 싸다.** 50M 모델을 20 tok/param으로 학습하면 약 1h, 200 tok/param은 약 1일이다. **병목은 교사 forward, 렌더, SfM이다.** 학습 FLOPs가 아니다.
3. **교사 forward가 화폐다.** Z-Image-Turbo 1024² forward 1회 ≈ 61 TFLOPs ≈ T4에서 3–4s [추정]. v1 WSSC(1.6K–9.6K forward/세계)는 세계당 1.4–11 T4-h다. **512² band-iso 뷰, 뷰 수와 K 감축, INT4로 약 100배 줄여야 한다.**
4. **BF16/TF32/FP8가 없다.** bf16은 에뮬레이션이라 속도 이득이 없다. Z-Image와 Sana 공식 경로가 bf16이라 **fp16 NaN·검은 이미지 위험이 Day-1 실측 1순위**다. SDXL은 fp16-fix VAE로 검증된 안전 경로다.
5. **Triton ≥3.3은 cc<8.0에서 tl.dot을 FMA로 폴백**한다(지원 하드웨어 CC 8.0+, best effort). FlexAttention, 커스텀 Triton, max-autotune에 기대지 말고 **cuBLAS/CUTLASS/CUDA C++(gsplat, Warp)를 기본 경로**로 쓴다.
6. **Attention.** FA2/3/4는 쓸 수 없다. SDPA는 Turing에서 mem-efficient 커널을 쓴다. **flash-attention-turing은 fwd 최대 2.19배, bwd 1.35–1.51배**(hd 64/96/128, dropout 없음). FA1은 backward가 hd ≤64로 제한된다. **head_dim 64로 설계한다.**
7. **T4는 지원 하한선에 있다.** PyTorch CUDA 13 빌드, Warp CUDA 13 빌드, OptiX 모두 최소가 Turing이다. 지금은 안전하지만 **다음 CUDA 메이저에서 잘릴 위험**이 있으니 툴체인 버전을 고정한다.
8. **INT8/INT4가 숨은 무기다.** 실측 INT8 74.9 TOPS, INT4 114 TOPS다. Nunchaku v1.2가 20-series INT4를 지원하고 Z-Image-Turbo 4-bit도 있다. 6–20B 교사가 16GB에 들어간다. T4 실측은 미확인.
9. **교사 후보.** 16GB에 fp16으로 들어가는 최고 미감 Apache 모델은 **Z-Image-Turbo(Elo 941, 6.15B, 1024²에서 26–35s [추정])**다. 속도 대안은 **FLUX.2 klein 4B(863, Apache, 9–12s [추정], 편집 내장)**다. SDXL-Lightning(678)은 가장 빠르고 안전하지만 미감이 낮다. 상용 1위(1198) 대비 Z-Image의 기대 승률은 약 19%다.
10. **두 장 사용법.** NVLink 없이 PCIe Gen3이라 **역할을 분리한다.** GPU0 = 교사 DiT, GPU1 = VAE, 세계, 렌더, 학습. 텍스트 인코더는 1회 실행 후 캐시한다.
11. **3DGS는 T4에서 현실적이다.** FastGS는 4090 기준 1.93분, 0.40M Gaussians, 5.2GB이고 T4에서 약 10–16분으로 추정된다. gsplat은 메모리 4배 절감과 3DGUT(rolling shutter, 왜곡) 통합으로 **v1 primary 렌더 경로를 유지할 수 있다.** 3DGRT는 연기한다.
12. **합성 데이터로 학습한 소형 모델 선례는 강하다.** RAFT 5.3M(합성으로 from scratch, 2×2080Ti), Real-ESRGAN(순수 합성 열화), MDM(2080Ti 1장 3일)이 있다. 다만 **합성만으로 학습한 깊이 모델은 일반화가 나쁘다**(DA-V2). 기하 모델에는 **실영상 + 고전 SfM/MVS 의사정답**이 필수다.
13. **소형 범용 생성기는 약하고 from-scratch T2I는 불가능하다.** DiT-S/2(33M)는 FID 68.4다. MicroDiT(1.16B)조차 우리 90일 예산의 2–3배가 든다. 그래서 **미감은 오직 빌린 교사의 몫**이고, 수십M 모델은 인수분해된 좁은 연산자만 맡는다.
14. **FP16 학습은 안정화 장치가 기본이다.** muP의 6.7B 재현은 FP16에서 발산해 FP32로 학습했다. Wortsman에 따르면 불안정은 소형 모델에서도 재현되고 처방(QK-norm, z-loss 등)이 통한다. GradScaler, fp32 softmax/LN, QK-norm을 기본값으로 둔다.
15. **고전 SfM 스택이 성숙했다.** COLMAP 4.0에 GLOMAP이 통합됐고 4.1에 GPU BA(Caspar)가 들어갔다. GLOMAP 매핑은 IMC 2023에서 497s 대 4051s다(매칭 제외). **ALIKED/LightGlue는 학습 모델이라 금지하고 SIFT만 쓴다.** Kaggle은 세션 12h, 주 약 30h(비공식)이므로 체크포인트와 재개가 필수다.

---

## 8. 불확실·미확인 항목

1. **모든 이미지 모델의 T4 실측 속도.** 2.2절 표는 SDXL(Fooocus 역산)을 빼면 FLOPs 추정이다. 512/768/1024 해상도와 NFE별로 Day 1에 실측해야 한다.
2. **bf16 네이티브 모델의 fp16 수치 안정성**(Z-Image, klein, Sana). NaN, 검은 이미지, VAE 오버플로가 나는지 확인해야 한다.
3. **Nunchaku INT4의 T4 실측.** 20-series 지원은 확인했고 T4는 언급이 없다. Z-Image와 klein의 Turing INT4 경로 존재 여부와 속도도 미확인.
4. **증류 교사의 임의 t SDEdit 품질.** Turbo/Lightning/Sprint를 WSSC 결정 창에 쓸 수 있는지 모른다.
5. **라이선스 상충.** Sana와 Sana-Sprint 가중치(NSCL v2 vs Apache), Lumina 2.0(Apache 여부 불명확), SD3.5 Community License의 $1M 기준(2차 출처). 데이터셋(DL3DV, RealEstate10K, ScanNet++)의 상업 사용 가능 여부는 조사하지 않았다.
6. **torch.compile/Inductor의 T4 실제 영향.** Triton FMA 폴백에서 내가 추론한 것이다. xformers 최신 휠에 sm75가 포함되는지도 미확인.
7. **T4 지속 실효 TFLOPS.** 서버 냉각과 전력 설정에 크게 좌우된다. 우리 장비에서 `nvidia-smi` throttle reason을 보며 측정해야 한다.
8. **FastGS와 3DGS의 T4 환산(4090 대비 5–8배 가정)**, 3DGRT의 T4 속도, tiny-cuda-nn fully-fused MLP의 T4 동작 여부.
9. **gsplat, Newton, MuJoCo Warp, Genesis, Taichi, Blender 5.x의 최소 CC.** 문서화되어 있지 않고 Warp를 따른다고 추론했다.
10. **Kaggle 세부.** T4×2 쿼터 소모율, RAM, CPU 수, 디스크. Colab 무료 한도는 비공개이고 출처끼리 상충한다.
11. **선례 수치 일부.** MDM 파라미터 수(논문에 없음), Real-ESRGAN 16.7M과 TAESD 1.2M(2차 출처), MVSplat 백본이 사전학습 초기화인지 여부(from scratch 선례로 쓸 수 있는지 영향).
12. **규칙 해석.** 교사의 텍스트 인코더(Qwen3-4B, 사실상 LLM)를 "이미지 모델의 일부"로 프롬프트 인코딩 외 용도(scene graph 생성 등)에 쓸 수 있는지. **창업자가 결정할 사항이다.**
13. **이미지 아레나 Elo와 비디오 모델 프레임 미감의 관계.** Phase 1 목표("단일 프레임에서 Wan/Seedance 대비 50% 이상")를 이 Elo로 예측할 수 없다. 별도 블라인드 측정이 필요하다.
14. **COLMAP 4.x(SIFT + GLOMAP + Caspar) 실측.** 장면당 처리량과 CPU 병목을 우리 하드웨어에서 측정한 적이 없다.
