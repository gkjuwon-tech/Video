# 공개 데이터 계획 (D1, 2026-10-11) — 창업자 아이폰 촬영 대체

창업자 결정: 아이폰 16 Pro 촬영(IMU·LiDAR 방·RAW 매크로)은 양과 품질이 부족하므로 **전문 장비로 찍은 공개 데이터**로 대체한다. 조사 워크플로 `public-data-hunt`(Sonnet 탐색 4 + Opus 라이선스 검증 4)가 라이선스 원문을 직접 읽고 판정했다. 근거 요약은 `work/data/data_hunt_summary.txt`.

## 규칙 (학습 데이터)
- **R1 상업 학습 허용**: NC·연구 전용·학술 전용·AI 학습 금지·"경쟁 서비스 금지" 조항이 있으면 학습 금지(평가만).
- **R2 신경망 라벨 금지**(PARADIGM CT23): 정답이 센서(레이저·LiDAR·모션캡처·IMU), 고전 계산(COLMAP·ORB-SLAM·BA), 합성 렌더에서 나와야 한다. 신경망이 만든 마스크·깊이·포즈·캡션은 쓰지 않는다.
- **CC-BY는 출처 표기 의무**: 쓰는 순간 `ATTRIBUTION.md`에 데이터셋·저자·라이선스를 기록한다. **CC-BY-SA**(Hypersim)는 공유 조건 위험이 있으니 파생 데이터를 재배포하지 않고, 가중치에 미치는지는 법률 확인 전까지 보조로만.
- 경쟁 모델 출력(LTX-2.5, Seedance)은 COMP: 평가 전용, 학습 금지.

## 모델별 데이터원
| 모델 | 학습 OK (조건) | 비고 |
|---|---|---|
| **M3 MRH 실사 차선** | **DIODE**(MIT, FARO 레이저, 실내+실외 약 2.5만 장, 깊이 84 GB + 법선 131 GB) — 주력 실사. TUM RGB-D(CC BY 4.0, 소량, VGA) | 합성 보조: **Hypersim**(CC BY-SA 3.0, 1.9 TB, 선택 다운로드), **TartanAir V2**(CC BY 4.0, V2만), **BlendedMVS**(CC BY 4.0, 원천 사진 권리 미확인 → 보조), **Infinigen/Infinigen-Indoors**(BSD-3, 직접 생성, 외부 에셋 경로 쓰지 말 것). **ARKitScenes**: 라이선스 두 조항이 충돌 → **평가 전용**, Apple에 서면 확인 요청(ARKitScenes-license@group.apple.com) 후 승격 |
| **M11 CamFM-Hand** | **TUM VI**(CC BY 4.0, 200 Hz 자이로·가속도, 사람이 들고 걷는 리그, 정적 IMU 111시간), **TII Racing 드론**(CC BY 4.0, 사람 조종 FPV IMU 500 Hz + 모션캡처), TUM RGB-D(모션캡처 100 Hz 경로, 자이로 없음), Zenodo 22690904(CC BY 4.0, 검증용) | 리그 다양성은 CAM-01 리그 시뮬레이터로 합성 |
| **M10 CamFM-Intent · M12 CRM** | **Blender 오픈 무비 .blend 카메라 곡선**(영화별 CC BY 1.0–4.0, 영화마다 확인), 자체 Toric 문법 합성, TUM 경로 | DataDoP·E.T.는 R2 위반(신경망 포즈)으로 평가 전용. RealEstate10K는 원본이 YouTube 영상이라 제외 |
| **M13 Rig-ID** | CAM-01 합성 리그 + TUM VI(손) + TII(FPV) | |
| **M5 TexelFM** | **Poly Haven**(CC0, **공식 API로만**), **ambientCG**(CC0, 스캔 기반 우선), **MatSynth**(라이선스 필드 CC0만, ShareTextures 출처 항목 제외), **CGBookCase**(CC0), **TextureCan**(CC0) | 털·모피 근접 질감은 공백(ShareTextures는 일괄 다운로드 금지 조항 → 평가만). 추론 때는 입력 사진 자체의 견본 타일을 쓴다 |
| **M1 정식 (RAW 사실감)** | **raw.pixls.us**(샘플별 CC0만, rawsamples.ch 출신 NC 제외), **SIDD**(MIT, 저자 서면 확인 권장) | |
| **M8 BRP** | **Infinigen**(직접 생성: 바위·나무·지형·생물, 360° 정답), **OmniObject3D**(CC BY 4.0, 6천 개 실물 스캔; 신경망 매팅 마스크는 버림), **Google Scanned Objects**(CC BY 4.0, 1,030개), **ABO**(CC BY 4.0, 가구 7.9천 + 회전 사진), **Poly Haven 모델**(CC0, 자연물), **Objaverse 1.0**(개체별 라이선스 CC-BY 4.0·CC0만), **Smithsonian 3D**(CC0, 1차 약관 페이지 접속 실패 → 재확인 후) | |
| **M7 MassingFM** | 자체 레이아웃 문법 + Infinigen-Indoors(직접 생성) + ABO 가구 | 실제 방 배치 데이터는 상업 허용이 없다(전부 NC) → 합성만 |

## 탈락 (학습 금지, 사유)
ScanNet, ScanNet++, Matterport3D/HM3D, Replica, Taskonomy, ETH3D, Tanks and Temples, KITTI, Waymo, Mid-Air, MatrixCity, Structured3D, 3D-FRONT/3D-FUTURE, ZInD, HSSD-200, ShapeNet (모두 비상업·학술 전용) · EuRoC, UZH-FPV, ADVIO, Newer College, Oxford Spires, Hilti, UZH 이벤트 카메라, Aria 계열, RoNIN, TLIO (NC·연구 전용·불명) · DataDoP, E.T. (신경망 라벨) · MIT-Adobe FiveK, RAISE, Zurich RAW-to-RGB, DTD, Textures.com, Quixel Megascans (연구 전용·독점) · Unsplash, Pexels (경쟁 서비스·ML 조항, 미재확인) · nyuuzyou/cc0-textures (출처 불명 재배포).

## 공백 (공개 데이터로 못 메우는 것)
1. **실제 식당·카페 실내 + 레이저 깊이**: 상업 허용 데이터가 없다. DIODE 소량 + Hypersim/Infinigen 합성으로 간다. (선택) 창업자 아이폰 LiDAR 몇 곳은 **평가 전용 홈 경기장**으로 가치가 있다.
2. **다중 리그 실제 손떨림**(폰·고프로·짐벌·숄더): TUM VI 한 리그뿐. (선택) 창업자 폰 들고 걷기 20–30분이 "폰" 리그 클래스를 채우는 가장 싼 방법.
3. **살아 있는 동물 실물 3D(털·변형)**: 없음. Objaverse 동물(아티스트 제작) + Smithsonian 표본 + Infinigen 생물로 대체.
4. **4K 이상 실사 + 정렬된 깊이**: 없음.

## 할 일
- 데이터 단위(DAT-04, DAT-06, DAT-07, DAT-08, DAT-13, CAM-06, INV-09)는 이 문서를 데이터 원천으로 쓴다. 다운로드는 GPU 서버(또는 디스크 큰 별도 인스턴스)에서, 필요한 부분만 선택적으로.
- Apple에 ARKitScenes 상업 사용 서면 확인 메일(창업자 승인 후 발송).
- `ATTRIBUTION.md` 신설: CC-BY 데이터를 처음 쓰는 커밋에서 함께 추가.
