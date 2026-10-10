> **출처 메모.** 2026-10-09 기준 웹 조사로 만든 지형도다. 수치 상당수는 벤더 발표나 3rd-party 보도에 기대고 있고, 확인하지 못한 항목은 §10에 따로 적었다. 인용하기 전에 원문을 다시 확인할 것.

---

# 지형도 리포트: "세계를 먼저 만들고 촬영한다" (기준일 2026-10-09)

## 0. 요약

1. 상용 비디오 최상위권은 지금 이렇게 갈립니다. Artificial Analysis T2V Arena 스냅샷(날짜 표기 없음, 10/9 조회) 기준 1위 Wan 3.0(1156), 2위 Utopai X, 3위 Seedance 2.5(1143), 4위 MiniMax H3, 6위 FLUX 3 순입니다. Kling 3.0 Omni는 15위(1015)입니다. Veo 4는 아직 없고, Sora 앱은 2026년 4월 종료로 보입니다.
2. 상용 모델의 카메라 제어는 대부분 텍스트 지시, 첫·끝 프레임, 레퍼런스 비디오 수준입니다. 6-DoF 경로를 넣었을 때 기하가 보장되는 상용 모델은 확인하지 못했습니다.
3. 정적 장면에서 카메라가 움직일 때 상용 모델이 실패한다는 근거는 정량으로 나와 있습니다. GeCo가 Sora 2와 Veo 3.1을 측정했고, 정적 장면의 fused 오차가 0.07–0.27로 실제 영상 바닥값 0.02의 3–10배였습니다. 주요 실패는 표면 녹아내림, 가려졌던 영역의 불일치, 정적 프롬프트인데 물체가 움직이는 현상입니다. 단, Seedance와 Kling을 같은 기준으로 잰 공개 벤치마크는 찾지 못했습니다.
4. "세계 먼저" 계열은 이미 많습니다. Marble(World Labs), Lyra 2.0(NVIDIA), HY-World 2.0/2.1(Tencent), FlashWorld, CAT3D/Bolt3D, SEVA, Voyager/GEN3C, Matrix-3D, WorldGen(Meta), Echo-2(SpAItial)가 있습니다. 그런데 시네마틱 품질로 최상위 비디오 모델과 사람 선호도를 정면 비교한 사례는 없습니다. 측정 지표도 LPIPS/FID/WorldScore뿐입니다.
5. 업계 흐름은 거꾸로 가고 있습니다. World Labs는 Marble(splat) 다음 제품으로 Atlas(autoregressive diffusion transformer, 1440p, 1분 비디오)를 냈습니다. Genie 3, Runway GWM-1, HY-World 1.5 WorldPlay도 모두 프레임 생성형입니다. 명시적 세계로 프레임 생성형보다 높은 화질을 낸다는 쪽에 거는 회사는 사실상 없습니다. 이 자리가 비어 있습니다.
6. WorldScore가 이 구도를 수치로 보여 줍니다. 3D 계열은 카메라 제어 85–94, 3D 일관성 81–90으로 높지만 주관 품질은 37–71입니다. 비디오 계열은 카메라 제어 22–40에 주관 품질 49–64입니다. 두 축을 동시에 이긴 시스템은 아직 없습니다. 다만 이 스냅샷에는 2026년 상용 모델이 들어 있지 않습니다.
7. 시점 간 불일치 문제를 학계가 지금 푸는 방식은 두 가지입니다. 비디오 모델 prior를 쓰는 방식(Lyra 2.0, FlashWorld)은 화질 상한이 비디오 모델 수준에 묶입니다. 재구성으로 평균하는 방식(CAT3D)은 결과가 흐려집니다. 고해상도 이미지 모델을 teacher로 써서 세계 자체를 생성 변수로 삼는 경로는 장면 규모로 해 본 사례를 확인하지 못했습니다.

---

## 1. 상용·오픈 비디오 모델 현황 (2026-10 기준)

| 모델 | 최신 버전 / 출시 | 해상도·길이·fps (출처 간 충돌 많음) | 카메라 제어 | 비고 |
|---|---|---|---|---|
| Seedance (ByteDance Seed) | 2.0: 2026-02 (2/12 설). 2.5: 2026-07-31 (8/8 설도 있음) | 2.0: 4–15s, 480p/720p/1080p. 2K·4K 주장은 서로 충돌. 2.5: 단일 패스 30s, multi-turn 연장. 해상도는 720p 상한설과 4K설이 충돌 | 텍스트 지시와 레퍼런스. 2.5는 이미지 30, 비디오 10, 오디오 10개 레퍼런스. "3D pre-visualization"은 Caixin 보도에만 있음 | AA Elo: 2.5가 3위(1143), 2.0이 7위(1116) |
| Kling (Kuaishou) | 3.0 / 3.0 Omni: 2026-02-04~06. 3.0 Turbo와 Omni 업그레이드: 2026-06-17(공식 확인 못 함) | 3–15s, 1080p 표준. 4K는 Omni 4K 티어. 60fps는 미확인 | camera path와 motion vector를 받는 "motion control" 모델, 최대 6컷 multi-shot | AA: 3.0 Omni 1080p Pro가 15위(1015) |
| Veo (Google) | 3.1: 2025-10. Veo 4는 발표 없음(2026-08 시점) | 4/6/8s, 720p/1080p, 2026-01부터 4K 업스케일 | Flow의 카메라 제어, 첫·끝 프레임, 다중 키프레임 정밀도는 부족 | I/O 2026의 비디오 발표는 Gemini Omni Flash. Omni 1.1 Flash(2026-08)는 10초 단위로 40초까지 연장, 이미지 5·비디오 3 레퍼런스. AA 8위(1113) |
| Sora (OpenAI) | Sora 2: 2025-09-30 | 15–25s 1080p(보도 기준) | — | 보도상 2026-03~04에 standalone 제품 종료. OpenAI 페이지 간에도 서로 맞지 않음 |
| Runway | Gen-4.5: 2025-12. Aleph 2.0. GWM-1: 2025-12-11 | Gen-4.5 API는 720p, 2–10s | 첫 프레임 입력. GWM-1은 camera pose 등 action 조건 | GWM-1은 Gen-4.5 위에 올린 autoregressive real-time 모델(Worlds/Robotics/Avatars) |
| MiniMax Hailuo | H3(Hailuo 3.0): 2026-07-30~08-03. H3 Max: 2026-09-02 | 4–15s, native 768p(2K는 regenerate), Extend로 약 30s. 4K60 주장은 신뢰 낮음 | 레퍼런스 12개(이미지 9, 비디오 3, 오디오 3) | AA 4위(1137)와 5위. open weights 약속은 라이선스 제약 가능성 |
| Wan (Alibaba) | 오픈은 2.1/2.2(Apache-2.0)까지. 2.5–2.7은 closed. 3.0: 2026-08-24 API | 3.0: 480p–1080p, 초당 $0.0425–0.17 | — | AA 1위(1156). Wan 3.0 open 여부는 상충 |
| HunyuanVideo (Tencent) | 1.5: 2025-11, 8.3B, 오픈 | 480–720p 5–10s, SR로 1080p | — | 2.0은 확인 못 함 |
| Luma | Ray3(2025, 16-bit HDR), Ray3.14(2026-01, native 1080p), Ray3.2(2026 중반, 미검증) | — | 첫·끝 프레임(Ray3 Modify) | — |
| Skywork | SkyReels V2(diffusion forcing, 무한 길이), V3(2026-01, 오픈), V4(2026-03) | — | — | AA 9위 V4 |
| Moonvalley Marey | 2025-07 GA | 5s, 24fps, 1080p | 단일 이미지에서 3D-aware free camera | 2026-06 Reka 합류설(미검증) |

기타 AA 상위권: Utopai X(2위), FLUX 3(BFL, 6위), Grok Imagine Video 1.5, HappyHorse 1.0/1.1(Alibaba-ATH), Agnes-Video-2.5.

**시사점.** 현 최상위 상용 모델의 물리적 상한은 단일 패스 15–30초, 네이티브 720p–1080p입니다(4K는 대부분 업스케일이거나 상위 티어). Phase 1의 비교 대상은 Seedance 2.5, Wan 3.0, Kling 3.0 Omni, Gemini Omni 1.1 Flash, Veo 3.1입니다.

출처: [AA T2V leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video), [Seedance 2.5 – TechNode](https://technode.com/2026/07/31/bytedance-launches-seedance-2-5-video-generation-model/), [Caixin Seedance 2.5](https://www.caixinglobal.com/2026-06-24/bytedance-targets-july-launch-of-upgraded-ai-video-model-102456839.html), [Seedance 2.0 spec(segmind)](https://blog.segmind.com/seedance-2-0-video-duration-length-cost-use-cases/), [Kling 3.0 press(Bernama)](https://web11.bernama.com/en/press/news.php?id=2520757), [Kling 3.0 Turbo(AtlasCloud)](https://www.atlascloud.ai/blog/guides/kling-3.0-turbo-kling-omni), [Kling 3.0 spec(muapi)](https://muapi.ai/kling-3), [Veo 4 미출시(pixverse)](https://pixverse.ai/es/blog/veo-4-release), [Gemini Omni 1.1 Flash(Google blog)](https://blog.google/innovation-and-ai/technology/developers-tools/build-with-gemini-omni-1-1-flash/), [Sora 상태(igorslab)](https://www.igorslab.de/en/openai-is-not-completely-shutting-down-sora-but-following-mixed-signals-is-apparently-pulling-the-plug-on-its-current-product-strategy/), [Runway Gen-4.5(eWeek)](https://www.eweek.com/news/runway-ai-video-model/), [Runway GWM-1](https://runwayml.com/research/introducing-runway-gwm-1), [MiniMax H3(NYU RITS)](https://rits.shanghai.nyu.edu/ai/minimax-releases-h3-2k-video-with-native-audio-open-weights-promised/), [Wan 3.0(OpenRouter)](https://openrouter.ai/alibaba/wan-3.0), [HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870), [Luma Ray3.14](https://www.businesswire.com/news/home/20260126744711/en), [SkyReels V3](https://huggingface.co/Skywork/SkyReels-V3-V2V-14B), [Marey(TechCrunch)](https://techcrunch.com/2025/07/08/moonvalleys-ethical-ai-video-model-for-filmmakers-is-now-publicly-available)

---

## 2. 알려진 실패 모드 (근거 있는 것 위주)

### 2.1 정적 장면과 카메라 이동: GeCo가 가장 직접적인 증거

- **프로토콜.** 정적 장면 프롬프트 80개를 object, indoor, outdoor, appearance-complex 네 범주로 나눕니다. slow/fast 궤적에 seed 4개씩, 모델당 320클립입니다. 약 3초 window를 8fps 이하로 리샘플합니다. 정지 화면에 가까운 출력이 점수에서 이득을 보지 않도록 Total Motion과 Mean Motion을 함께 보고합니다.
- **지표.** 창 안 center frame i와 다른 frame j에 대해 계산합니다.
  - motion 잔차: `e_m(p) = ||F_obs(p) − F_rigid(p)||₂ / f`. 여기서 `F_obs`는 UFM 광류, `F_rigid`는 VGGT depth와 pose에서 유도한 rigid flow이고, co-visible 픽셀에서만 계산합니다.
  - structure 잔차: `e_s(p) = |D_{j→i}(p) − D_i(p)| / D_i(p)`
  - fused: `e(p) = sqrt(M_covis(p)·e_m(p)² + e_s(p)²)`
  - 전 과정이 differentiable이라 sampling guidance loss로도 쓸 수 있습니다.
- **결과(낮을수록 좋음).**

  | 모델 | object | indoor | outdoor | appearance-complex |
  |---|---|---|---|---|
  | Sora 2 | 0.069 | 0.169 | 0.265 | 0.206 |
  | Veo 3.1 | 0.090 | 0.233 | 0.217 | 0.210 |
  | Wan 2.2 | 0.140 | 0.279 | 0.437 | — |

  실제 영상의 noise floor는 약 0.02이고, 생성 영상은 대개 0.1–0.3입니다. LTX-Video가 낮게 나온 것은 움직임이 거의 없어서입니다.
- **실패 유형.**
  1. Geometric deformation: co-visible 강체 표면이 늘어나거나 휘거나 녹아내립니다.
  2. Occlusion inconsistency: 가려졌다 다시 보이는 물체가 바뀌어 있고, disocclusion 영역이 매번 새로 생성됩니다.
  3. Spurious object motion: 정적 프롬프트인데 지구본이 계속 돕니다. 동결 세계 프롬프트에 대한 정면 반례입니다.
- **한계.** Kling과 Seedance는 측정 대상에 없습니다.
- 출처: [GeCo arXiv 2512.22274](https://arxiv.org/html/2512.22274)

### 2.2 긴 궤적: 공간 망각, 드리프트, loop closure 실패

- **Lyra 2.0(NVIDIA, 2026-04)이 정의한 두 실패.** spatial forgetting은 context 밖으로 나간 영역을 다시 볼 때 새로 지어내는 현상입니다. temporal drifting은 autoregressive 오차가 누적되는 현상입니다. [arXiv 2604.13036](https://arxiv.org/abs/2604.13036)
- **ViewBench**(arXiv 2602.07854): yaw, pitch, roll의 왕복 궤적에서 Loop Closure Error를 `LPIPS(첫 프레임, 원위치로 돌아왔을 때의 프레임)`로 정의합니다. [pdf](https://arxiv.org/pdf/2602.07854)
- **SCoPE**(arXiv 2606.27345): truck, orbit, dolly, non-palindromic loop로 구성한 Revisit 프로토콜입니다. camera return error, ViPE 기반 trajectory error, LPIPS와 DINOv2 similarity를 봅니다. [pdf](https://arxiv.org/pdf/2606.27345)
- **Generative View Stitching**: RE10K처럼 baseline이 작은 데이터로 학습하면 forward-orbit-backward 궤적에서 loop closure에 실패한다고 스스로 보고했습니다. [pdf](https://arxiv.org/pdf/2510.24718)
- **WorldCrafter**(2026-09): 폐곡선 궤적의 revisit LPIPS가 Lyra 2.0 0.487, 자기 방법 0.255입니다. 0.255도 실사 기준으로는 꽤 다른 이미지입니다. [pdf](https://arxiv.org/pdf/2609.24984)
- **R2M-Bench**(2026-08): 절대 유사도로는 revisit 기억과 일반적 시각 안정성을 분리할 수 없다는 이유로 상대 비교를 제안합니다. [pdf](https://arxiv.org/pdf/2608.27328)

### 2.3 뭉개짐과 과도하게 부드러운 움직임

- **Tex4D 부록**: video diffusion은 image diffusion보다 결과가 더 부드럽습니다. 고주파 디테일을 시간축에서 유지하기가 어렵기 때문에 공간 디테일을 희생해 시간 일관성을 얻는다는 설명입니다. 창업자가 말한 "뭉개짐"과 "이상하게 부드러움"을 학계가 직접 인정한 기술입니다. [pdf](https://arxiv.org/pdf/2410.10821)
- **"How I Warped Your Noise"(ICLR 2025)**: image prior로 비디오를 만들 때 생기는 고주파 flicker와 texture sticking의 원인을 noise sampling으로 봅니다. 해법은 noise를 광류를 따라 이동시키는 것입니다. 노이즈를 세계 좌표에 고정한다는 아이디어의 선례입니다. [review](https://liner.com/review/how-i-warped-your-noise-a-temporallycorrelated-noise-prior-for)
- **Seedance 2.0 리뷰**(벤더 블로그라 신뢰도 중간): 텍스트가 warp되거나 flicker합니다. 긴 클립에서 drift가 생기고 레퍼런스를 무시하는 경우가 있습니다. 반대로 360° orbit에서 "rubber-limb"가 Gen-3보다 적었다는 평도 있습니다. [pixverse review](https://pixverse.ai/en/blog/seedance-2-0-review-prompts-and-use-cases)
- **Project Genie**: 예술적 스타일은 강하지만 photorealism과 object interaction은 약하다는 평가입니다. [TechCrunch](https://techcrunch.com/2026/01/29/i-built-marshmallow-castles-in-googles-new-ai-world-generator-project-genie)
- **Stable Virtual Camera(SEVA)** 자체 한계: 물체를 관통하는 궤적, 복잡한 궤적, 입력에서 먼 시점에서 flicker가 납니다. 사람, 동물, 물 같은 dynamic texture도 약합니다. [arXiv 2503.14489](https://arxiv.org/pdf/2503.14489)

### 2.4 WorldScore: 비디오 계열과 3D 계열의 반대 구조

ICCV 2025 논문입니다. 아래는 프로젝트 페이지 스냅샷이고, 열은 제가 순서대로 매핑했습니다. 2026년 상용 모델은 들어 있지 않습니다.

| 모델 | Static | CamCtrl | 3D Consist | Photo Consist | Style | Subjective |
|---|---|---|---|---|---|---|
| Gen-3 | 60.71 | 29.47 | 68.31 | 87.09 | 62.82 | 63.85 |
| Hailuo | 57.55 | 22.39 | 67.18 | 62.82 | 54.91 | 52.44 |
| Wan2.1 | 57.56 | 23.53 | 78.74 | 78.36 | 77.18 | 59.38 |
| LucidDreamer | 70.40 | 88.93 | 90.37 | 90.20 | 48.10 | 58.99 |
| WonderWorld | 72.69 | 92.98 | 86.87 | 85.56 | 70.57 | 49.81 |
| Text2Room | 62.10 | 94.01 | 88.71 | 88.36 | 37.23 | 36.69 |
| Voyager (하이브리드) | 77.62 | 85.95 | 81.56 | 85.99 | 84.89 | 71.09 |

- 3D 계열은 제어와 일관성에서 이기고 미감에서 집니다. 비디오 계열은 그 반대입니다.
- 하이브리드인 Voyager가 현재 static 최고점입니다.
- 출처: [WorldScore](https://haoyi-duan.github.io/WorldScore/), [HF leaderboard](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard)

### 2.5 공백

- **frozen-time 전용 공개 벤치마크가 없습니다.** bullet-time이나 frozen-time orbit에서 Seedance와 Kling을 정량 비교한 공개 자료를 찾지 못했습니다.
- **연구 쪽 접근.**
  - BulletTime(CVPR 2026): world-time과 camera pose를 분리한 4D positional encoding과 AdaLN을 씁니다. 데이터는 Blender 합성 약 2k장면, 약 20k영상입니다. [arXiv 2512.05076](https://arxiv.org/pdf/2512.05076)
  - CAT4D: multi-view video diffusion으로 bullet-time 뷰를 생성합니다.
  - BulletGen(Meta): bullet-time 생성으로 4DGS를 보정합니다. [arXiv 2506.18601](https://arxiv.org/html/2506.18601v2)

---

## 3. "세계 먼저 만들고 촬영" 계열 선행 연구·제품

### 3.1 명시적 세계 (explicit world → render)

**World Labs Marble 1.0 / 1.1 / 1.1 Plus**
- 일정: 2025-09 beta, 2025-11-12 GA, 1.1은 2026-04-02.
- 표현: Gaussian splat. SPZ는 2M 또는 500K splat, PLY도 지원. mesh는 Pro 플랜.
- 생성 방식: 비공개. 텍스트, 이미지, 비디오, 3D 레이아웃(Chisel)을 입력으로 받습니다.
- 한계: room-scale입니다. 1.1 Plus는 "dynamic cubes"로 1패스에 최대 5 unit까지 확장합니다. 사람과 동물에는 최적화되어 있지 않습니다. 조명이 bake되어 있습니다.
- 시네마틱 지향: 부분적입니다. Studio에서 camera path를 만들고 video export를 할 수 있습니다. 다만 Studio의 enhance 기능은 3rd-party 설명뿐입니다.

**World Labs Atlas**
- 일정: 2026-09-01 early access.
- 표현과 방식: multimodal autoregressive diffusion transformer입니다. 명시적 카메라 입력을 받고 3D도 출력합니다.
- 한계: 논문, 코드, 독립 벤치마크가 없습니다.
- 시네마틱 지향: 그렇습니다. 1440p, 최대 1분 비디오(자체 주장)입니다.

**NVIDIA Lyra 2.0**
- 일정: 2026-04, arXiv 2604.13036.
- 생성 방식: Wan2.1-14B(VAE 8×8×4, 16ch, 832×480)로 walkthrough 비디오를 만들고, Depth Anything 3의 3DGS head로 feed-forward 재구성합니다. mesh는 OpenVDB와 marching cubes로 뽑습니다.
- 메모리와 학습:
  - frame별 depth, camera, 1/8 point cloud를 cache합니다.
  - 가시 점 기준으로 과거 프레임을 최대 5장 greedy 검색합니다.
  - canonical 좌표 맵을 forward-warp해서 attention의 Q·K에만 더합니다. V에는 넣지 않습니다.
  - 확률 0.7로 history에 t~U(0,0.5) 노이즈를 넣고 1-step으로 복원하는 self-augmented history 학습을 합니다.
  - DMD로 4-step까지 distill합니다.
- 한계: 해상도가 480p급입니다. DL3DV 기준 FID 43–51입니다. 정적 장면만 다룹니다. DL3DV 데이터의 노출 불일치를 그대로 물려받습니다. 라이선스는 연구 전용이고 생산 사용을 금지합니다.
- 시네마틱 지향: 아닙니다. sim과 game 자산이 목표입니다.

**NVIDIA Lyra (1)**
- 일정: ICLR 2026.
- 생성 방식: video diffusion의 latent에서 3DGS decoder를 학습하는 self-distillation입니다. RGB decoder가 teacher, 3DGS decoder가 student입니다. 4D로 확장했습니다.
- 한계: 해상도와 디테일이 비디오 latent 해상도에 묶입니다.

**Tencent HY-World 2.0 / 2.1**
- 일정: 2.0은 2026-04-16, 2.1은 2026-07.
- 표현: 3DGS와 mesh.
- 생성 방식: HY-Pano 2.0(약 80B)으로 360° 파노라마를 만들고, WorldNav가 궤적을 계획하고, WorldStereo 2.0(약 17B)이 궤적을 따라 확장하고, WorldMirror 2.0(약 1.2B, feed-forward)과 3DGS로 합성합니다.
- 한계: 궤적을 따라 순차적으로 확장하므로 우리가 거부한 패러다임에 가깝습니다. 모델이 거대합니다.
- 시네마틱 지향: 아닙니다. 엔진, sim, 게임 자산이 목표입니다.

**HY-World 1.5 WorldPlay**
- 일정: 2025-12.
- 생성 방식: streaming video diffusion에 Reconstituted Context Memory를 붙였습니다. 24fps입니다.
- 명시적 세계가 아니고 프레임 생성형입니다.

**FlashWorld**
- 일정: ICLR 2026, arXiv 2510.13678.
- 표현: pixel-aligned 3DGS. 픽셀마다 depth, quaternion, scale, opacity, SH를 예측하고, 중심은 `o + d·r`입니다.
- 생성 방식:
  - Wan2.2-5B를 초기값으로 한 dual-mode multi-view DiT입니다. 24 views, 480p, Plücker raymap 조건입니다.
  - MV-oriented 모드는 latent v-pred, 3D-oriented 모드는 Gaussian을 렌더해 re-encode합니다.
  - DMD2 cross-mode distillation을 합니다. MV 모드가 teacher(품질), 3D 모드가 student(일관성)이고, 4-step {1000,900,750,500}입니다.
  - 손실은 GAN, R1, 그리고 CMC loss 0.1입니다.
  - 장면당 약 9초입니다.
- 한계:
  - 3D 모드만 쓰면 흐립니다.
  - WorldScore 3D 일관성이 85.87로 LucidDreamer의 90.37보다 낮습니다.
  - 장면 규모와 다양성이 데이터에 묶입니다.
  - 거울 반사와 미세 기하가 약합니다.
- 시네마틱 지향: 아닙니다.

**CAT3D (Google)**
- 일정: NeurIPS 2024.
- 표현: NeRF.
- 생성 방식: multi-view LDM으로 많은 뷰를 생성한 뒤 robust NeRF로 재구성합니다.
- 한계: 생성 뷰가 서로 불일치하고, 재구성에서 평균되며 흐려집니다. 공개되지 않았습니다.
- 시네마틱 지향: 아닙니다.

**Bolt3D (Google)**
- 일정: ICCV 2025.
- 표현: pixel-aligned Gaussian.
- 생성 방식: multi-view LDM이 appearance와 pointmap latent를 함께 생성하고, Gaussian head가 opacity와 scale을 예측합니다. 6.25초입니다.
- 한계: 장면 규모가 작습니다. 데이터는 SfM으로 만든 기하입니다.
- 시네마틱 지향: 아닙니다.

**Stable Virtual Camera (SEVA)**
- 일정: ICCV 2025.
- 표현: 없음. 생성형 NVS입니다.
- 생성 방식: 임의 카메라를 조건으로 한 multi-view diffusion에 2-pass procedural sampling을 써서 30초 loop closure를 만듭니다.
- 한계: 복잡한 궤적에서 flicker가 납니다. 사람과 물이 약합니다.
- 시네마틱 지향: 일부 그렇습니다.

**GEN3C (NVIDIA)**
- 일정: CVPR 2025.
- 표현: colored point cloud cache.
- 생성 방식: cache를 렌더한 결과를 video diffusion의 조건으로 넣습니다.
- 한계: 결국 비디오 모델이 프레임을 그리는 구조이고, 빈 곳은 비디오 모델이 채웁니다.
- 시네마틱 지향: 아닙니다.

**Voyager (HunyuanWorld)**
- 일정: 2025-09.
- 표현: RGB-D 비디오에서 point cloud.
- 생성 방식: world-consistent RGB-D video diffusion에 point culling cache를 붙인 autoregressive 방식입니다.
- 한계: 540p이고 GPU 메모리를 60GB 씁니다.
- 시네마틱 지향: 아닙니다. 다만 WorldScore static 1위입니다.

**Wonderland (Snap)**
- 일정: CVPR 2025.
- 표현: 3DGS.
- 생성 방식: camera-guided video latent에서 feed-forward로 3DGS를 만듭니다.
- 한계: 해상도가 latent에 묶입니다.

**Matrix-3D (Skywork)**
- 일정: 2025-08.
- 표현: 3DGS.
- 생성 방식: mesh render를 조건으로 한 panoramic video diffusion 뒤에 feed-forward 또는 최적화 기반 재구성을 합니다. Matrix-Pano 116K 데이터셋을 함께 냈습니다.
- 한계: 360° 파노라마라 뷰 하나당 해상도가 낮습니다.

**Director3D**
- 일정: NeurIPS 2024.
- 표현: 3DGS.
- 생성 방식: Traj-DiT(text에서 camera path)로 궤적을 만들고, GM-LDM이 pixel-aligned Gaussian을 만들고, SDS++로 다듬습니다.
- 한계: 궤적이 연출 의도보다 기하적 타당성만 따릅니다(GenDoP 지적). 해상도가 낮습니다.
- 시네마틱 지향: 카메라 연출을 다룬 유일한 선례입니다.

**DiffSplat**
- 일정: ICLR 2025.
- 표현: splat grid(object 단위).
- 생성 방식: T2I diffusion을 splat latent를 출력하도록 fine-tune하고 rendering loss를 더합니다.
- 한계: object 수준입니다.
- 의미: 이미지 모델이 splat latent를 하나의 스타일처럼 학습한다는 증거입니다.

**WonderWorld / WonderJourney / LucidDreamer / Text2Room / ViewCrafter**
- 일정: 2023–2024.
- 표현: 각각 FLAGS(layered Gaussian surfel), mesh, 3DGS 등.
- 생성 방식: 순차 아웃페인팅이나 인페인팅으로 확장합니다.
- 한계: seam과 해상도 문제가 있습니다. 우리가 거부한 패러다임 그 자체입니다.

**Meta WorldGen**
- 일정: 2025-11, CVPR 2026.
- 표현: trimesh와 navmesh.
- 생성 방식: LLM 레이아웃, 이미지 레퍼런스, image-to-3D, 분해와 텍스처링 순서입니다. 최대 50×50m.
- 한계: 게임 그래픽 수준이지 photoreal이 아닙니다.

**WorldGrow**
- 일정: AAAI 2026.
- 표현: TRELLIS식 structured latent block.
- 생성 방식: 3D block inpainting과 coarse-to-fine.
- 한계: 3D-FRONT 실내에 한정됩니다. block을 순차 확장합니다.

**SpAItial Echo / Echo-2**
- 일정: 2025–26.
- 표현: Gaussian splat(PLY/SPZ export).
- 생성 방식: 비공개입니다.
- 한계: 공개 정보가 적습니다.

**Difix3D+ (NVIDIA)**
- 일정: CVPR 2025.
- 생성 방식: 1-step image diffusion으로 렌더 아티팩트를 고치고, 그 결과를 3D로 재증류하며, 추론 시 enhancer로도 씁니다.
- 의미: 이미지 모델을 최종 렌더러로 쓰는 선례입니다.

출처: [Marble 1.1](https://rits.shanghai.nyu.edu/ai/world-labs-releases-marble-1-1-auto-expanding-3d-world-generation), [Marble export docs](https://docs.worldlabs.ai/marble/export/specs), [Marble release notes](https://docs.worldlabs.ai/marble/release-notes), [Atlas(SiliconANGLE)](https://siliconangle.com/2026/09/01/fei-fei-lis-world-labs-debuts-atlas-a-world-model-showcase-for-advanced-spatial-intelligence/), [RTFM](https://worldlabs.ai/blog/rtfm), [Lyra 2.0 html](https://arxiv.org/html/2604.13036v1), [Lyra 2.0 HF](https://huggingface.co/nvidia/Lyra-2.0), [Lyra](https://arxiv.org/html/2509.19296v1), [HY-World 2.0 GitHub](https://github.com/Tencent-Hunyuan/HY-World-2.0), [HY-World 2.0 PDF](https://3d-models.hunyuan.tencent.com/world/world2_0/HY_World_2_0.pdf), [HY-World 1.5 report](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf), [FlashWorld](https://arxiv.org/html/2510.13678v1), [CAT3D](https://arxiv.org/html/2405.10314v1), [Bolt3D](https://arxiv.org/html/2503.14445v1), [SEVA](https://arxiv.org/pdf/2503.14489), [GEN3C](https://arxiv.org/pdf/2503.03751), [Voyager](https://arxiv.org/pdf/2506.04225), [Wonderland](https://arxiv.org/html/2412.12091v2), [Matrix-3D](https://arxiv.org/html/2508.08086v1), [Director3D](https://arxiv.org/pdf/2406.17601), [DiffSplat](https://arxiv.org/pdf/2501.16764), [WonderWorld](https://arxiv.org/pdf/2406.09394), [WorldGen](https://arxiv.org/pdf/2511.16825), [WorldGrow](https://arxiv.org/pdf/2510.21682), [Echo(maginative)](https://www.maginative.com/article/synthesia-co-founder-raises-13m-to-build-the-holy-grail-of-ai-interactive-3d-worlds-from-text/), [Difix3D+](https://arxiv.org/pdf/2503.01774)

### 3.2 프레임 생성형 월드 모델 (경쟁 패러다임)

- **Genie 3**(2025-08): 720p, 24fps, real-time, 수 분간 일관성 유지. **Project Genie**(2026-01-29, US Ultra 구독자 대상): 60초 월드, photorealism이 약합니다. Genie 4는 없습니다. [Wikipedia](https://en.wikipedia.org/wiki/Genie_(world_model)), [TechCrunch](https://techcrunch.com/2026/01/29/i-built-marshmallow-castles-in-googles-new-ai-world-generator-project-genie)
- **Runway GWM-1 Worlds**: Gen-4.5 기반 autoregressive real-time 모델입니다. [Runway](https://runwayml.com/research/introducing-runway-gwm-1)
- **World Labs RTFM**(2025-10): 명시적 3D 없이 프레임에 3D pose를 붙여 주변 프레임을 검색하는 "context juggling"을 합니다. 이어서 Atlas(2026-09)가 나왔습니다.
- **NVIDIA Cosmos Predict 2.5**: 2B와 14B, flow 기반, 2억 클립, 최대 30초. Cosmos 3는 README 언급뿐입니다. [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5)

**해석(제 판단).** World Labs조차 splat에서 프레임 생성형으로 무게를 옮기고 있습니다. 명시적 세계를 그대로 렌더하면 시네마틱 화질이 안 나온다는 업계 합의로 읽힙니다. 우리 테제는 이 합의에 정면으로 반대하는 베팅입니다.

---

## 4. 4D·동적·물리 (Phase 2 관련)

- **Feed-forward 4D 재구성**
  - MoVieS(CVPR 2026): pixel-aligned "dynamic splatter pixels"에 per-pixel motion displacement를 붙였습니다. 장면당 0.93초입니다. [arXiv 2507.10065](https://arxiv.org/pdf/2507.10065)
  - VGGT-Ω(CVPR 2026 oral): 동적 장면에서 카메라와 depth는 내지만 motion mask, scene flow, 4D 모델은 내지 않습니다.
- **비디오 diffusion 기반 4D 생성**
  - 4DNeX: Wan 계열 fine-tune으로 RGB와 XYZ를 합친 6D video를 만들고 dynamic point cloud로 출력합니다. 4DNeX-10M 데이터셋을 냈습니다. [arXiv 2508.13154](https://arxiv.org/pdf/2508.13154)
  - Diff4Splat(CVPR 2026): 단일 이미지, 궤적, 텍스트에서 deformable 3DGS를 1-pass로 만듭니다. 약 30초입니다. [arXiv 2511.00503](https://arxiv.org/pdf/2511.00503)
  - CAT4D: multi-view video diffusion과 deformable 3DGS를 씁니다. [arXiv 2411.18613](https://arxiv.org/pdf/2411.18613)
  - Full-4D: 단일 비디오에서 multi-view video를 합성한 뒤 4DGS로 만듭니다.
  - Lyra는 4D 확장을 언급만 했습니다.
- **시간과 카메라 분리**: BulletTime(CVPR 2026, §2.5).
- **월드 단위 4D**
  - TeleWorld(2026-01): 생성, 재구성, guidance의 폐루프로 4D 시공간 표현을 유지합니다. [arXiv 2601.00051](https://arxiv.org/pdf/2601.00051)
  - LivingWorld(2026-04): 단일 이미지에서 구름, 물, 연기 같은 environmental dynamics를 hash 기반 motion field로 만듭니다. 확장 1스텝에 9초 걸리는데, 이는 순차 확장이라는 뜻입니다. [alphaXiv 2604.01641](https://www.alphaxiv.org/abs/2604.01641)
- **물리**
  - NGFF(ICLR 2026): 3DGS 인지와 neural force field로 상호작용 4D 비디오를 만듭니다. 기존 Gaussian 시뮬레이터보다 두 자릿수 빠르다고 하고, Veo3보다 낫다고 주장합니다. GSCollision 64만 영상(약 4TB)을 냈습니다. [arXiv 2602.00148](https://www.arxiv.org/abs/2602.00148)
  - PhysGaussian, Phys4DGen, Phy124: 3DGS를 MPM 입자로 다룹니다(2024).
- **공백.** 장면 규모 4D, 정체성 유지, 물리, 시네마틱 렌더를 동시에 갖추고 비디오 SOTA와 비교한 사례는 없습니다. 대부분 object 단위, driving, ambient motion 중 하나입니다.

---

## 5. 일관성·세계 품질 평가 지표

| 지표·벤치 | 정의(요지) | 쓸모 |
|---|---|---|
| **MEt3R** (CVPR 2025) | 이미지 쌍에 DUSt3R를 돌려 pointmap을 얻고, DINO 계열 feature를 상대 시점으로 warp한 뒤 overlap mask 위 cosine similarity를 잽니다. `MEt3R = 1 − ½(S₁₂+S₂₁)` 꼴이고 pose가 필요 없습니다. view-dependent 효과에 둔감하게 설계됐습니다. | 생성 뷰 간 일관성 |
| **GeCo** (2025-12) | §2.1의 fused 잔차입니다. differentiable이고 실제 영상 바닥값이 약 0.02입니다. | frozen-world 영상의 핵심 지표 후보 |
| **WorldScore** (ICCV 2025) | 3,000개 예시, controllability·quality·dynamics. DROID-SLAM 재투영 오차로 3D 일관성을, 추정 pose로 카메라 제어를 잽니다. photometric, style, subjective 지표 포함. | 3D와 비디오 통합 비교. 2026 상용 모델은 미포함 |
| **VBench-2.0** | Human Fidelity, Controllability, Creativity, Physics, Commonsense를 VLM·LLM과 전문 도구로 평가하고 사람 선호와 맞춰 봅니다. | 일반 비디오 품질 |
| Loop closure / revisit | ViewBench LCE(LPIPS), SCoPE Revisit, R2M-Bench(상대 비교), WorldCrafter revisit LPIPS | 되돌아오기 |
| 재투영 오차 | GeoVideo, WorldScore, Lyra 2.0이 사용 | 기하 |
| Lyra 2.0 세트 | SSIM, LPIPS, FID, Subjective, Style, CamCtrl, Reproj. 3D 쪽은 LPIPS-P/G, FID | 세계 생성 표준에 가까운 묶음 |
| 사람 선호 | Artificial Analysis Video Arena Elo | "Seedance를 이긴다"를 검증하는 실질적 심판 |
| 카메라 연출 | CameraBench(약 3,000영상, 시네마토그래퍼와 만든 taxonomy), DataDoP(29K 실제 shot 궤적) | "미친 카메라 무빙"의 평가와 학습 |

출처: [MEt3R](https://arxiv.org/abs/2501.06336), [GeCo](https://arxiv.org/abs/2512.22274v1), [WorldScore](https://arxiv.org/pdf/2504.00983), [VBench-2.0](https://arxiv.org/html/2503.21755v2), [GeoVideo](https://arxiv.org/pdf/2512.03453), [CameraBench](https://arxiv.org/abs/2504.15376), [GenDoP/DataDoP](https://arxiv.org/pdf/2504.07083)

---

## 6. 데이터셋과 feed-forward 재구성

**정적·카메라 데이터**
- **DL3DV-10K**: 10,510개 영상, 5,120만 프레임, 4K, 65개 POI 유형. Lyra 2.0과 FlashWorld의 핵심 학습·평가 데이터입니다. [arXiv 2312.16256](https://arxiv.org/pdf/2312.16256)
- **RealEstate10K**: YouTube 부동산 영상에서 약 8만 클립, 약 1천만 프레임, SfM pose. 수치는 기억 기반이라 원문 확인이 필요합니다. 전진 위주이고 baseline이 작아서 loop closure 학습에 불리합니다(GVS 지적). [project](https://google.github.io/realestate10k/)
- **ACID**: 해안 드론 영상, RE10K 포맷. 정확한 규모는 미확인입니다. [mirror](https://hyper.ai/en/datasets/20569)
- **Sekai**: 100개국 이상, 750개 도시의 walking·drone 영상 5,000시간 이상. Sekai-Game은 약 60시간. [arXiv 2506.15675](https://arxiv.org/pdf/2506.15675)
- **Matrix-Pano**: depth와 궤적이 붙은 정적 파노라마 비디오 116K.
- **Voyager 데이터**: 실사와 UE 렌더 10만 클립 이상.

**동적 데이터**
- **SpatialVID**(CVPR 2026): 원본 21,000시간 이상에서 2.7M 클립, 동적 콘텐츠 7,089시간. per-frame pose, depth, dynamic mask, 구조화 캡션, motion instruction이 붙어 있습니다. [arXiv 2509.09676](https://arxiv.org/pdf/2509.09676)
- **OmniWorld**(ICLR 2026): 합성 OmniWorld-Game과 공개 데이터 큐레이션, 3억 프레임 이상, RGB, depth, pose 등. [arXiv 2509.12201](https://arxiv.org/pdf/2509.12201)
- **Stereo4D**(CVPR 2025 oral): 인터넷 VR180 스테레오 영상에서 metric 스케일과 장기 3D 궤적을 가진 4D 장면 10만 개 이상. [arXiv 2412.09621](https://arxiv.org/html/2412.09621v2)

**연출 데이터**
- **DataDoP**: 영화와 다큐 29K shot의 궤적(MonST3R로 추정), depth, 연출 캡션.

**feed-forward 재구성**
- **DUSt3R, MASt3R, CUT3R**: 기반 계열입니다.
- **VGGT**(CVPR 2025): 기반 모델입니다.
- **π3, MapAnything, AnySplat**: arXiv ID는 기억 기반이라 검증하지 못했습니다.
- **Depth Anything 3**(2025-11): VGGT보다 pose 정확도가 약 23.6%, π3보다 약 16.7% 높다는 요약이 있습니다. Gaussian head가 있고 Lyra 2.0이 채택했습니다. [arXiv 2511.10647](https://arxiv.org/pdf/2511.10647)
- **VGGT-Ω**(CVPR 2026 oral, arXiv 2605.15195): 0.2B에서 10B, 수천에서 200만 시퀀스까지 power-law 스케일링을 보였습니다. 1,800만 개 비디오로 self-supervised 학습을 했습니다. 1B 체크포인트에 benchmark contamination 가능성이 있다고 저자가 공지했습니다(3rd-party 보도, 원문 미확인). [arXiv](https://arxiv.org/abs/2605.15195), [learnopencv](https://learnopencv.com/vggt-vs-vggt-%cf%89-vggt-omega-a-complete-guide-to-feed-forward-3d-reconstruction/)
- **WorldMirror 2.0**(Tencent, 약 1.2B): any-input에서 3D로.
- **MoVieS**: 동적 장면용(§4).

**파이프라인 표준형(Voyager, SpatialVID, Lyra 2.0 공통).** 인터넷 비디오를 받아 shot을 분할하고 정적·동적을 필터링합니다. 그다음 pose와 depth를 추정하고(VGGT, DA3, MegaSaM, ViPE 계열), dynamic mask를 만들고, 캡션을 붙여 학습 튜플을 만듭니다.

---

## 7. 우리 아이디어와 가장 가까운 선행 작업 Top 10, 그리고 아직 Seedance/Kling을 못 이긴 이유

1. **World Labs Marble 1.1(+Studio) / Atlas**: 제품 측면에서 가장 가깝습니다. 세계를 만들고 camera path를 넣고 video export까지 됩니다.
   - splat 상한(SPZ 2M)과 bake된 조명 때문에 클로즈업 디테일, 반사, 투명체가 약합니다.
   - room-scale이고 cube 확장도 최대 5 unit입니다. 사람과 동물이 비최적입니다. 정적입니다.
   - 비디오 모델과 공개 head-to-head 비교가 없습니다.
   - 회사 자신이 Atlas라는 프레임 생성형으로 1440p 비디오를 노립니다. 명시적 렌더만으로는 부족하다는 내부 판단으로 해석됩니다(제 추정).
2. **FlashWorld**: 기술적으로 가장 가깝습니다. 이미지급 품질을 내는 모드를 3D 일관 모드로 DMD2 distill합니다.
   - teacher가 이미지 모델이 아니라 Wan2.2-5B 비디오 모델이라 품질 상한이 비디오 수준입니다.
   - 480p에 24뷰이고, Gaussian이 pixel-aligned라 장면 규모와 디테일 밀도가 고정됩니다.
   - 3D 일관성이 LucidDreamer보다 낮습니다. 시네마틱 평가를 하지 않았습니다.
3. **Lyra 2.0**: 긴 궤적에서 일관된 비디오를 만들고 3DGS로 올립니다.
   - 세계가 생성 비디오의 증류물이라 비디오의 블러, 드리프트, 노출 불일치를 물려받습니다.
   - 832×480에 FID 43–51입니다. DA3 head를 downsample해 Gaussian 수가 4배 적습니다.
   - 정적 장면만 다루고 연구 전용 라이선스입니다.
4. **HY-World 2.0/2.1**: 파노라마, 궤적 계획, stereo 확장, feed-forward 합성으로 이어지는 오픈 파이프라인입니다.
   - 확장이 궤적을 따라 순차적이라 seam과 누적 오류 위험이 있습니다(우리가 거부한 방식과 가까움).
   - 목표가 게임과 sim 자산입니다. 80B와 17B 모델을 연쇄로 돌려야 합니다.
5. **CAT3D / Bolt3D**: 많은 뷰를 생성한 뒤 재구성합니다. 창업자 원안과 거의 같은 구조입니다.
   - 뷰 간 미세 불일치가 재구성에서 평균되어 흐려지는 문제를 정면으로 해결하지 못했습니다.
   - 512급 해상도이고 공개되지 않았습니다.
6. **Stable Virtual Camera(SEVA)**: 임의 카메라, 30초, loop closure.
   - 명시적 세계 없이 매번 프레임을 그리는 구조입니다. 복잡한 궤적에서 flicker가 납니다.
7. **GEN3C / Voyager**: 3D cache를 렌더해 조건으로 쓰고 비디오 diffusion이 출력합니다.
   - 최종 픽셀은 여전히 비디오 모델이 그리므로 뭉개짐과 과도한 부드러움을 물려받습니다.
   - 빈 곳을 비디오 모델이 채우는 hole-filling 구조입니다. 해상도는 540–720p입니다.
8. **Lyra(1) / Wonderland**: video latent에서 3DGS를 decode합니다.
   - 디테일이 video latent 해상도(8× downsample)에 묶입니다.
9. **Director3D**: 텍스트에서 시네마틱 궤적과 Gaussian을 함께 생성합니다. 카메라 연출을 다룬 유일한 선례입니다.
   - 2024년 작업이고 해상도가 낮습니다. SDS++ 특유의 과포화가 있습니다.
   - 궤적이 연출 의도를 반영하지 않습니다(GenDoP 지적).
10. **Matrix-3D / Meta WorldGen / SpAItial Echo-2** (묶음)
    - Matrix-3D는 파노라마를 쓰기 때문에 뷰당 해상도가 낮습니다.
    - WorldGen은 mesh라 게임 수준의 사실감입니다.
    - Echo-2는 공개 정보가 부족합니다.
    - 셋 다 시네마틱 비교를 하지 않았습니다.

(참고로 DiffSplat은 이미지 prior를 splat으로 직접 바꿀 수 있다는 증거로, Difix3D+는 이미지 모델을 최종 렌더 보정기로 쓰는 선례로 중요합니다.)

---

## 8. 아무도 제대로 채우지 못한 빈자리 (우리가 들어갈 자리)

- **G1. 명시적 세계 렌더로 최상위 비디오 모델을 사람 선호에서 이긴 사례가 0건입니다.**
  - 3D 계열 평가는 LPIPS, FID, WorldScore에서 멈춰 있습니다.
  - Seedance 2.5, Wan 3.0, Kling 3.0, Veo 3.1과 frozen-time 카메라 무빙을 블라인드 Elo로 붙인 공개 결과가 없습니다.
- **G2. frozen-time 벤치마크가 없습니다.** 재료는 GeCo(spurious motion, deformation), MEt3R, ViewBench LCE, CameraBench taxonomy로 이미 다 있습니다. 묶은 사람이 없을 뿐입니다.
- **G3. 이미지 teacher를 쓰고 세계 자체를 생성 변수로 삼는 장면 규모 생성기가 없습니다.**
  - 현재 생성기들은 비디오 teacher(FlashWorld, Lyra 계열), object 단위(DiffSplat), 장면별 최적화(SDS/VSD 계열) 중 하나입니다.
  - 2K–4K 이미지 모델(GPT Image 2, Nano Banana Pro, Seedream 5급)을 장면 세계 생성기의 분포 매칭 teacher로 쓴 사례를 찾지 못했습니다.
- **G4. 해상도와 디테일 밀도가 고정되어 있습니다.**
  - 모든 계열이 480p급 pixel-aligned Gaussian이거나 고정 splat 수입니다.
  - 카메라가 다가가면 디테일이 일관되게 늘어나는 multi-scale 세계 생성이 없습니다. Marble의 LOD는 렌더링 쪽이지 생성 쪽이 아닙니다.
- **G5. view-dependent 효과와 조명이 비어 있습니다.**
  - splat은 조명을 bake합니다. 거울과 반사는 FlashWorld가 스스로 한계로 적었습니다.
  - 시네마틱의 핵심인 스펙큘러 하이라이트의 이동, 유리, 물은 공백입니다.
- **G6. 물리적 카메라가 없습니다.**
  - 렌즈(DoF, focus pull, 아나모픽, 비네팅), 셔터(motion blur), 센서(grain, 노출)를 세계 렌더에 미분 가능하게 넣은 생성 시스템이 없습니다.
  - 과도하게 부드러운 움직임의 반대말은 24fps와 180° 셔터라는 물리입니다.
- **G7. 세계를 이해하는 감독이 없습니다.**
  - DataDoP와 CameraBench는 있지만, 생성된 세계의 3D saliency와 구도에 맞춰 궤적을 계획하는 시스템은 없습니다.
  - Director3D의 궤적은 연출 의도를 반영하지 못했습니다.
- **G8. Phase 2의 장면 규모 4D가 비어 있습니다.**
  - 정체성을 유지하는 동적 세계, 물리, 시네마틱 렌더를 함께 갖춘 시스템이 없습니다.
  - 데이터(SpatialVID, Stereo4D, OmniWorld, Sekai)와 동적 재구성(MoVieS, VGGT-Ω)은 이미 있습니다.
  - 외관은 이미지 teacher가, 동역학은 비디오 teacher가 맡는 분리 학습은 시도된 적이 없습니다.
- **G9. 전략적 공백이 있습니다.**
  - 대형 플레이어가 모두 프레임 생성형으로 이동했습니다(Genie 3, GWM-1, Atlas, WorldPlay).
  - 명시적 세계 쪽 연구 결과물은 연구 전용 라이선스가 많습니다(Lyra 2.0).
  - 상업용 고품질 명시적 세계 기반 비디오 스택을 노리는 경쟁이 상대적으로 비어 있습니다.

---

## 9. 리서치에서 도출한 설계 시사점

새 패러다임 후보입니다. 겹치는 선행 연구는 명시했습니다.

### P1. 세계 자체가 denoising 대상인 생성 + 이미지 teacher 분포 매칭

- **표현.** `W = {W_ℓ}_{ℓ=0..L}` 계층 세계로 둡니다.
  - 각 레벨은 sparse voxel 또는 octree에 고정된 Gaussian 집합입니다.
  - Gaussian 하나는 `g = (μ∈ℝ³, q∈ℝ⁴, s∈ℝ³, α∈ℝ, a∈ℝ³ albedo, n/ρ, f∈ℝ^{d≈32} neural appearance)`입니다.
  - 여기에 env lighting `E`를 더합니다.
  - 토큰화는 TRELLIS식 structured latent `z ∈ ℝ^{N_vox×C}`를 장면 단위로 확장한 형태입니다.
- **1단계 (world-native prior).** 인터넷 비디오를 DA3나 VGGT-Ω로 재구성하고 W로 fit합니다(DL3DV, SpatialVID 정적 구간, Sekai). 그 위에서 `p(z)`의 flow matching을 학습합니다.
- **2단계 (이미지 teacher post-train).** 렌더 `x = R(G_φ(ε), c)`에 대해 DMD2식 분포 매칭을 합니다.

  `∇_φ L = E_{c~p(c|path), t}[(s_fake(x_t,t) − s_real^{img}(x_t,t)) · ∂x/∂φ]`

  - `s_real`은 고정된 고해상도 이미지 모델입니다.
  - `p(c|path)`는 계획된 카메라 경로 주변에 중요도를 둔 샘플링입니다.
  - 모든 뷰의 gradient가 같은 W를 통과하므로 일관성은 구조적으로 강제됩니다.
  - 미세 불일치는 평균되지 않습니다. 하나의 세계가 모든 뷰에서 동시에 그럴듯하도록 수렴합니다. CAT3D식 블러와의 차이가 여기서 나옵니다.
- **겹침.** FlashWorld(DMD2 cross-mode, 비디오 teacher, 480p, pixel-aligned), DiffSplat(object), ProlificDreamer VSD(장면별 최적화), SyncTweedies, DMV3D, 3D-Adapter(재구성 in-the-loop denoising).
- **새로운 점.** 장면 규모를 한 번에 생성하는 amortized 생성기, 고해상도 이미지 teacher, 경로 인지 샘플링, multi-scale 세계의 조합입니다.

### P2. 디테일을 필요할 때 생성하는 zoomable 세계 (아웃페인팅이 아닌 이유)

- 레벨 `ℓ+1`은 `Down(W_{ℓ+1}) ≈ W_ℓ` 제약 아래 생성되는 주파수 대역 정제입니다. world-space cascaded super-resolution입니다.
- 새로운 의미 콘텐츠를 빈 곳에 채우지 않습니다. 이미 결정된 세계의 고주파만 결정합니다.
- 경로 위 영역은 렌더 전에 모든 레벨을 동시에 denoise합니다.
- 대규모 장면은 overlapping tile을 매 스텝 동시에 합의시킵니다(DiffCollage나 SyncTweedies식). 어떤 tile도 다른 tile보다 먼저 확정되지 않습니다. WonderWorld, WorldGrow, HY-World의 순차 확장과 본질적으로 다릅니다.

### P3. 미분 가능한 물리 카메라

- 셔터 적분 `I = (1/S)·Σ_k R(W, c(τ₀ + k·Δτ/S))`로 진짜 motion blur를 만듭니다. thin-lens DoF, rolling shutter, 노출과 톤 커브, 센서 grain을 더합니다.
- 프레임은 고정된 세계에서 렌더되므로 텍스처가 프레임마다 다시 샘플링되지 않습니다. texture swimming과 sticking의 원인 자체가 없습니다(Tex4D, Warped-Noise 근거).

### P4. 세계 좌표에 고정된 노이즈를 쓰는 neural 렌더 보정

- G-buffer(albedo, normal, roughness, f)를 1-step enhancer에 넣어 view-dependent 효과와 마감을 처리합니다(Difix3D+ 계열).
- enhancer의 노이즈를 세계 표면 점에 고정하고 투영해서 씁니다. 그러면 enhancer가 세계 점에 대해 결정적으로 동작해 flicker가 생기지 않습니다.
- **겹침.** Difix3D+, GEN3C, Warped Noise.
- **새로운 점.** 노이즈를 세계에 고정해 렌더 보정의 시간 일관성을 확보하는 것입니다.

### P5. 세계를 보는 DoP

- DataDoP와 CameraBench로 궤적 prior를 학습합니다.
- 생성된 W의 3D saliency, 구도, 시차 노출, focus target을 reward로 삼아 궤적을 최적화합니다(Director3D Traj-DiT, GenDoP 확장).

### P6. Phase 2

- 세계를 `W(τ) = W_canon ⊕ motion field ⊕ physics state`로 둡니다.
- 외관은 이미지 teacher, 동역학은 비디오 teacher의 저주파 motion score로 분리 학습합니다.
- 데이터는 SpatialVID, Stereo4D, OmniWorld를 MoVieS 또는 VGGT-Ω로 재구성해 씁니다.
- 시간과 카메라는 BulletTime식으로 분리합니다.
- 물리는 NGFF와 MPM 계열을 씁니다.

### 검증 프로토콜 (Phase 1 승리 조건 후보)

- **프롬프트.** 동일한 "frozen time + camera orbit/dolly/crane/loop" 프롬프트를 Seedance 2.5, Wan 3.0, Kling 3.0 Omni, Veo 3.1, Gemini Omni 1.1에 넣습니다.
- **지표.**
  - GeCo fused: 목표 0.03 이하, 실제 영상 바닥값 0.02에 근접.
  - MEt3R.
  - LCE: 왕복 궤적의 LPIPS.
  - spurious motion energy.
  - 고주파 디테일 유지율: revisit 프레임의 Laplacian 에너지 비율.
  - Q-Align.
  - 블라인드 pairwise Elo.

---

## 10. 불확실하거나 확인하지 못한 것

- Seedance 2.0/2.5의 네이티브 해상도는 720p, 1080p, 2K, 4K 주장이 서로 충돌합니다. 2.5의 "3D pre-visualization"은 Caixin 한 곳에만 있습니다.
- Kling 3.0의 60fps와 네이티브 4K, 3.0 Turbo(2026-06-17)는 공식 출처로 확인하지 못했습니다.
- AA 리더보드 스냅샷은 날짜가 표기되어 있지 않습니다. Utopai X, FLUX 3(video), HappyHorse 같은 모델의 실체는 검증하지 않았습니다.
- Sora 종료 시점은 OpenAI 페이지 간에도 서로 맞지 않습니다. Wan 3.0의 open weights 여부도 상충합니다.
- Atlas의 수치(1440p, 1분)는 모두 World Labs 자체 주장입니다. Marble의 생성 방식은 비공개이고, Studio "enhance"는 3rd-party 설명뿐입니다.
- HY-World 2.1의 기술 세부, Cosmos 3의 스펙, SpAItial Echo-2의 방법은 확인하지 못했습니다.
- WorldScore HF 리더보드에 2026년 상용 모델이 추가됐는지 확인하지 못했습니다. §2.4 표의 열 매핑은 제가 파싱한 것입니다.
- GeCo 이후 Seedance나 Kling을 포함한 정적 장면 기하 벤치마크(PDI-Bench 등)는 확인하지 못했습니다.
- RealEstate10K 규모, π3, MapAnything, AnySplat의 arXiv ID는 기억 기반입니다. ACID의 정확한 규모는 미확인입니다.
- VGGT-Ω 1B의 contamination 공지는 3rd-party 보도로만 확인했습니다.
- §9는 리서치를 바탕으로 한 가설입니다. 특히 P1의 이미지 teacher 분포 매칭이 장면 규모, 2K 이상 해상도에서 안정적으로 수렴하는지는 검증된 선례가 없습니다. 바로 그 점이 빈자리이자 리스크입니다.
