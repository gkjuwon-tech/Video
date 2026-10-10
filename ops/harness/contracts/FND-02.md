# FND-02 데이터 계약 v0 — World·ReadPacket·Obs·Shot·Theta·KPI·StageIO

**목적.** 스테이지 사이 모든 이음매의 단일 원천. 스텁↔실물 교체(PRD-01/PRD-03)가 이 타입만 보고 맞물린다.
**소유.** `buc/core/contracts.py`, `docs/contracts.md`(타입별 필드·단위·좌표계 설명, 아래 7+3 타입 이름을 모두 담는다). 테스트 `tests/fnd_02/`.
**의존.** numpy·표준 라이브러리만(torch 금지, CPU import). 해시는 `hashlib.sha256`(FND-03 blake3와 독립; v1에서 교체 가능하게 `content_hash` 한 곳에 둔다).

## 공통 규약
- 좌표: 세계 오른손·미터. 카메라 OpenCV(x 오른쪽, y 아래, z 앞). 픽셀 중심은 정수 좌표 → 영상 중심 주점은 `((W-1)/2, (H-1)/2)`.
- 영상: 채널 끝 `(H,W,3)`, `float32`, 장면 선형(Rec.709 원색). 깊이: 광축 z-깊이 m, `float32`, `>0`, 하늘은 `+inf` 허용(NaN 금지).
- 모든 계약 클래스는 `@dataclass`, **키워드 생성**, `__post_init__`에서 배열을 정본 dtype·C-연속으로 강제 변환한 뒤 `validate()`를 호출한다(위반 시 `ContractError`).
- 메서드(모든 계약 클래스): `validate() -> None`, `to_dict() -> dict`(json.dumps 가능, `"__contract__": 클래스명`, `"__v__": CONTRACT_VERSION` 포함, 배열은 `{"__ndarray__": base64, "dtype": "<f4", "shape": [...]}`로 비트 무손실), `@classmethod from_dict(d)`(주 버전이 다르면 `ContractVersionError`, 클래스명 불일치는 `ContractError`), `content_hash() -> str`(64자 hex).
- 모듈 함수: `content_hash(x)`(dict 키 순서 무관, list≡tuple, int≠float≠bool 구분, 배열은 dtype(리틀엔디안)+shape+바이트 — 메모리 배치 무관), `dumps(obj)->str`, `loads(s)->obj`(`__contract__`로 타입 디스패치).
- 상수: `CONTRACT_VERSION = "0.1.0"`, `LEVELS = ("L1","L1.5","L2a","L2b")`, `TAINT_BASES = {"USER","EXTGEN","CAPTURE","ASSET","ENGINE","SI","HOLDOUT","COMP"}`, `STAGES`(순서: `"S-1","S0","S1","S2","S3","S4","S5","S6","S7","S8","S8.5","S9-10","S11"`), `STAGE_SIGNATURES`.
- 예외: `class ContractError(ValueError)`, `class ContractVersionError(ContractError)`.

## 타입 (필드 = 키워드, 기본값)
- **Prov**(taint: frozenset[str]=∅, mode="engine", roots: tuple[str]=(), ns="engine"). taint 원소는 `BASE` 또는 `BASE:<한정자>`(BASE∈TAINT_BASES). roots는 정렬·중복 제거. 속성 `ext`(EXTGEN 존재), `comp`, `hold`. `Prov.merge(*ps)`: taint·roots 합집합, mode 같으면 유지 아니면 `"mixed"`, ns는 하나라도 `"user_vault"`면 `"user_vault"` 아니면 첫 값.
- **Theta**(aspect=16/9, focal_mm=35, sensor_width_mm=24.89, f_stop=4, focus_m=3, shutter_deg=180, fps=24, nd_stops=0, cpl=0, ev=0, wb_kelvin=6500, wb_tint=0, lut="none", grain=0, halation=0, vignette=0, lens_rx="spherical_clean", atmos_beta=0, sun_elev_deg=None, sun_azim_deg=None, rim=0, extra={}). 범위: focal (1,2000], sensor (1,100], f_stop [0.7,64], focus (0,∞], shutter (0,360], fps (0,240], nd [0,15], ev [-10,10], wb [1500,20000], tint/cpl·grain·halation·vignette·rim: tint [-1,1], 나머지 [0,1], atmos_beta ≥0, sun_elev [-90,90], sun_azim [0,360). `fov_x_deg()`, `intrinsics(W,H)->(3,3) float64`(fx=fy=focal/sensor·W, 주점 중심).
- **Obs**(K (3,3), R (3,3), t (3,), width, height, k1=0, k2=0, mask=None (H,W) bool, l_obs_hard=0, l_obs_part=0). R·t는 세계→카메라. K 마지막 행 [0,0,1], fx,fy>0; R 직교·det +1(atol 1e-6); 0 ≤ l_obs_hard ≤ l_obs_part. `w2c()`/`c2w()` (4,4), `center()`=-Rᵀt, `project(X (N,3)) -> (uv (N,2), z (N,))`(x=Xc/z; r²; ×(1+k1r²+k2r⁴); u=fx·x+cx).
- **InputImage**(pixels (H,W,3) uint8|uint16|float32, color_space∈{"srgb","display-p3","rec2020","linear-srgb","adobe-rgb","unknown"}, prov, exif={}, consent=True). 속성 `h_I`=content_hash(pixels).
- **ReadPacket**(S−1 출력: image_lin, depth, obs, theta_in, input_hash, prov, scale_s=1, sigma_s=0, level="L1", depth_conf=None [0,1], normals=None 단위벡터(atol 1e-3), esc={}, faces=(), lights={}, dyn={}, fallbacks=(), meta={}). image ≥0 유한; depth·image·mask가 obs의 (H,W)와 일치; scale_s>0; dyn 키 ⊆ {"blur_vel","support_graph","matter","artic_slots"}(P2 예약).
- **LDILayer**(rgba (H,W,4) float32 [0,1] 직선 알파, depth (H,W) float32 >0). **FarField**(rgb (S,S,3) float32 ≥0 유한, 팔면체).
- **World**(STRATA-lite-min v0: obs, layers: tuple[LDILayer] ≥1, theta_in, level, read_hash, prov, far_field=None, meta={}, schema="STRATA-lite-min/v0"). 층은 obs 크기, 앞→뒤 정렬(둘 다 α>1e-3인 픽셀에서 depth[i] ≤ depth[i+1]·(1+1e-6)).
- **Shot**(c2w (N,4,4), t (N,), K (N,3,3), focus_m (N,), width, height, fps, theta, move="static", rig="locked", seed=0). t[0]=0, 간격 1/fps(atol 1e-9), 회전 직교·det+1, 마지막 행 [0,0,0,1], focus>0. `n_frames`, `duration_s`=N/fps, `first_pose_matches(obs, rot_tol_deg=0.1, trans_tol_m=1e-3)`.
- **Frames**(rgb (N,H,W,3) float32 [0,1] 표시값, fps, aov={} 각 배열 첫 축 N, shot_hash="", world_hash="").
- **KPI**(ffd, ffg, de00, ir_geo, ir_area, ir_info, m_eff, esc, cheat_share, tfu, gpu_h, wall_s: 모두 Optional[float]=None, ladder_step=None, proof_of_shot=None, extra={}). ffd·ffg·de00·esc·tfu·gpu_h·wall_s ≥0; ir_*·cheat_share ∈[0,1]; m_eff >0; NaN 금지.
- **StageIO**(stage, impl_kind∈{"stub","real"}, impl_name, impl_version, inputs: dict[str,hash], outputs: dict[str,hash], params={}, wall_s=0, fallback=None). 속성 `cache_key` = content_hash(CONTRACT_VERSION, stage, impl_kind, impl_name, impl_version, inputs, params) — outputs·wall_s·fallback 제외.

## 스테이지 이음매
`STAGE_SIGNATURES[stage] = (입력 {이름: 타입}, 출력 {이름: 타입})`, v0은 다음 6개만:
`S-1` {image:InputImage}→{read:ReadPacket} · `S2` {read}→{world:World} · `S8` {world, read}→{shot:Shot} · `S8.5` {world, read}→{world} · `S9-10` {world, shot}→{frames:Frames} · `S11` {frames, read, world, shot}→{kpi:KPI}.
`check_stage_io(stage, inputs, outputs)`: 서명 없는 스테이지·키 누락/초과·타입 불일치 → ContractError. 교차 검사: S-1 read.input_hash==image.h_I, obs 크기==image, read.prov.taint ⊇ image.prov.taint; S2 world.read_hash==read.content_hash(), world.obs 해시==read.obs, taint ⊇; S8 shot.first_pose_matches(read.obs); S8.5 출력 world.obs==입력 obs; S9-10 frames N·H·W == shot.
`make_stage_io(stage, impl_kind, impl_name, impl_version, inputs, outputs, params=None, wall_s=0.0, fallback=None) -> StageIO`: check_stage_io 후 각 값의 content_hash로 채운다.

## 수용 기준
테스트 `tests/fnd_02/` 전부 통과(CPU, <60 s): 왕복 비트 무손실·해시 안정·위반 거부·투영 수치(1e-9)·이음매 교차 검사·cache_key 의미.

## 하류 이음매
RD-21 → ReadPacket, WLD-01 → `World`로 내보내기(STRATA 내부는 자유), LNS-01 → Theta 생성(필드 추가는 `extra`), CAM-02 → Shot, RDR-02/06 → Frames, PRD-01 → StageIO·check_stage_io·cache_key, PRD-03 스텁은 STAGE_SIGNATURES만 맞춘다, PRD-05/EVL → KPI. FND-04가 Prov 규칙을 강화해도 필드는 유지.
