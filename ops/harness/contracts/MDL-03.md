# MDL-03 M3 MRH 본체 v0 — PCA 탭 → DPT neck(FiLM t) → 조밀 헤드 8종 · 대역 사영 손실 · 학습 배선

**목적.** 엔진 교사 탭(TCH-03, 층 8/15/23, PCA-128 int8)을 읽어 H/4 해상도로 깊이·법선·알베도·rough/metal·shading·defocus·sem·σ̂를 내는 독심 헤드 본체(PARADIGM §5.3 M3: "DPT reassemble + fusion neck 256ch(RCU 12, FiLM t) 14M + 헤드"). v0 학습은 깊이·법선·sem만(r-inventory: 알베도·rough 등은 P 단계, 가중치 0). 전역 xfmr·전역 헤드 28은 MDL-04, MRH.obs는 MDL-05.
**소유.** `buc/models/mrh/neck.py`, `heads.py`, `train.py`; 테스트 `tests/mdl_03/`(동결). 의존: TCH-03 `buc.teacher.taps`(`TAP_LAYERS`, `PCA_DIM`, `PCABasis`, `TapSet`, `TapBasisError`), DAT-02 `buc.data.realify`(`band_project`, `max_octaves`, `REALIFY_TAG`, 샤드 필드 정본), DAT-05 `buc.data.shards`(`iter_shards`, `Sample`, `npy_decode`), RUN-06 `buc.train.loop`(`Trainer`, `LoopConfig`), `buc.train.amp.AmpConfig`. torch·numpy만. CUDA 없이 import, import 부작용·전역 RNG 변경 없음.
**소유 밖 요청.** `buc/models/__init__.py`·`buc/models/mrh/__init__.py`는 만들지 않는다(네임스페이스 패키지로 import된다). 필요하면 FND 단위에 요청.

## neck.py
- 예외 `MRHSpecError(ValueError)`(설정·입력 모양/값 위반).
- `@dataclass(frozen=True) MRHConfig(tap_layers=(8,15,23), tap_dim=128, neck_ch=256, rcu_per_fusion=4, head_ch=128, sem_dim=768, n_sigma_bands=4, t_embed_dim=128)`. tap_layers는 엄격 오름차순 정수 3개(≥0), 나머지는 양의 정수(bool 불가); 위반 `MRHSpecError`. `to_dict()`는 JSON 직렬화 가능한 dict(tap_layers는 list).
- `class ResidualConvUnit(nn.Module)`: 3×3 conv(C→C) 정확히 2개 + 잔차. 기본 설정에서 neck 안 개수 = `3·rcu_per_fusion` = 12.
- `class MRHNeck(nn.Module)(cfg)`; `forward(taps: Mapping[int, Tensor[B,k,gh,gw]], t) -> NeckOut`. `NeckOut(NamedTuple): feat4[B,C,4gh,4gw], feat16[B,C,gh,gw]`(C=neck_ch; feat16은 MDL-04 전역 xfmr 입력). DPT식: tap_layers[0]→stride 4, [1]→8, [2]→16 재조립 후 coarse→fine 융합 3블록. **FiLM(t)**: 각 융합 블록에 `γ=1+Δγ(e(t)), β(e(t))`, Δγ·β 선형층은 **영 초기화**(초기 출력은 t와 무관).
- 입력 검증: taps 키 집합 = tap_layers(빠짐·남음 `MRHSpecError`), 텐서는 부동소수(int8 등 비부동소수 `TypeError`: 먼저 `TapSet.features(basis)`), 4차원·층간 같은 모양·채널 = tap_dim(아니면 `MRHSpecError`). `t`는 float 또는 `Tensor[B]`(스칼라·원소 1개는 배치로 브로드캐스트, 그 밖의 길이≠B는 `MRHSpecError`), 유한 ∈(0,1], 아니면 `MRHSpecError`.
- **BatchNorm 금지**(뷰 독립: 표본 출력이 배치 구성과 무관, 1e-5).

## heads.py
- `head_channels(cfg) -> dict[str,int]`(이 순서): `log_depth 1, normal 3, albedo 3, rough_metal 2, shading 1, defocus 1, sem sem_dim, sigma_hat n_sigma_bands`. `HEAD_NAMES`는 같은 순서의 튜플. `SIGMA_MIN=1e-3`.
- `class MRHHeads(nn.Module)(cfg)`: `forward(feat4) -> dict[str, Tensor[B,c,h,w]]`(h,w = feat4의 공간). 출력 정의: log_depth 항등(affine-invariant), normal·sem = `x/max(‖x‖₂,1e-6)`(채널 축 단위벡터), albedo·rough_metal = sigmoid, shading·defocus = softplus(≥0), sigma_hat = softplus+`SIGMA_MIN`(>0).
- `class MRH(nn.Module)(cfg=MRHConfig(), *, seed: int|None=None)`: 속성 `cfg, neck, heads`. seed가 주어지면 지역 생성기로 초기화(같은 seed → 같은 state_dict, 전역 RNG 불변). `forward(taps, t) -> dict`(heads 출력), `features(taps, t) -> NeckOut`. 파라미터 전부 fp32.
- 예산(기본 설정): neck ∈ [9M, 18M], heads ∈ [0.5M, 3.5M], 합 ≤ 20M(전역 xfmr 7M과 함께 26M 상한).

## train.py — 라벨·손실
- 예외 `MRHDataError(ValueError)`. 상수 `STRIDE=4`, `MIN_VALID_PX=16`, `SEM_TEMP=0.07`, `N_SEM_IDS=64`.
- `band_valid_at_stride(valid, stride=4) -> tuple[bool]`: s=log2(stride)(2의 거듭제곱 아니면 `MRHSpecError`). `len(valid)>s`이면 `valid[s:]`, 아니면 `(valid[-1],)`(DAT-02 옥타브 정의에서 전해상 옥타브 o ↔ 1/stride 해상 옥타브 o−s).
- `downsample_labels(gbuf[B,12,H,W], sem[B,H,W], stride=4) -> dict`: `mask`=(avg_pool(level_mask)≥1−1e-6) float{0,1} `[B,1,h,w]`; `log_depth`=avg_pool(ch0)·mask; `normal`=avg_pool(ch1–3)를 ‖·‖>1e-6이면 정규화(아니면 0)·mask; `albedo`=avg_pool(ch4–6)·mask; `sem`=`sem[:, ::stride, ::stride]` int64. H,W가 stride 배수가 아니면 `MRHDataError`.
- `default_sem_table(n_ids=N_SEM_IDS, dim=768, seed=0) -> Tensor[n_ids,dim]`: 결정론 단위 행(TE-PCA 임베딩 전 자리값; 같은 seed → 동일, 다른 seed → 다름, 전역 RNG 불변).
- `@dataclass(frozen=True) LossWeights(depth=1.0, normal=1.0, sem=0.5, albedo=0.0)`.
- `mrh_loss(out, batch, *, sem_table, weights=LossWeights(), ignore_sem_ids=(0,)) -> (Tensor 스칼라, dict[str,float])`. v′=`band_valid_at_stride(band_valid_b)`, `BP=realify.band_project(·, v′)`, m=mask.
  - **depth(대역 안 affine-inv):** 표본마다 `Bp=BP(m·p)`, `Bm=BP(m)`, `Bg=BP(m·g)`; (a,b)=`argmin‖a·Bp+b·Bm−Bg‖²`, float64 2×2 정규방정식 + 대각 1e-8(미분 가능, detach 없음). `S_b=‖a·Bp+b·Bm−Bg‖²/max(n_b,1)`, n_b=Σm.
  - **normal / albedo:** `S_b=Σ_c‖BP(m·(pred−gt4))‖²/max(n_b,1)`.
  - 연속 항 L = `n_b≥MIN_VALID_PX`인 표본의 S_b 평균(없으면 그래프에 붙은 0).
  - **sem:** `logits=⟨ŝ, table_i/‖table_i‖⟩/SEM_TEMP`, sem4 라벨 중 `ignore_sem_ids`·범위 밖 제외, CE 평균(없으면 0). 대역 사영 없음.
  - total = Σ w·L(metrics는 파이썬 float; 가중치 0인 항은 계산은 해도 그래프에 기여 0, 해당 헤드 파라미터의 grad는 None 또는 0). metrics: `loss/depth, loss/normal, loss/sem, loss/albedo, sem_acc`.
- 평가 지표: `depth_absrel(pred_log[B,1,h,w], gt_depth[B,1,h,w], mask) -> Tensor[B]`: 마스크 화소에서 log 공간 최소제곱 affine(전 대역) 후 `mean(|exp(a·p+b)−d|/d)`. `normal_angle_deg(pred, gt, mask) -> Tensor[B]`: 마스크 평균 `acos(clamp(⟨n̂,n⟩,−1,1))`(도).

## train.py — 데이터·학습
- 배치는 dict(RUN-06 `to_device` 호환): `taps{l:[B,k,gh,gw] f32}, t[B] f32, band_valid tuple[tuple[bool]], gbuf[B,12,H,W] f32, sem[B,H,W] int64, keys tuple[str]`.
- `decode_sample(sample: Sample, basis, cfg) -> dict`(한 표본, B 축 없음): `meta.json`의 `v==REALIFY_TAG`, `basis_fp==basis.fingerprint.hex()`(아니면 `TapBasisError`), `tap_layers==cfg.tap_layers`, `keep`가 참, `gbuf/sem` 존재, H=16·gh·W=16·gw(위반 `MRHDataError`). 탭 = `npy_decode(tap<l>.npy)`(int8 `[k,gh,gw]`) × `basis.scale` = `TapSet.features`와 **비트 동일**. `t=meta.tap_sigma`, `band_valid=tuple(meta.band_valid)`.
- `collate(examples) -> batch dict`(모양 불일치 `MRHDataError`). `class MRHDataset(samples, basis, cfg)`: 전부 즉시 디코드, `len`, `basis_fp`, `batch(indices: int64[b]) -> dict`(순서 보존). `MRHDataset.from_shards(paths, basis, cfg)` = `iter_shards(paths)`(verify·방화벽은 DAT-05).
- `@dataclass(frozen=True) MRHTrainConfig(model=MRHConfig(), weights=LossWeights(), lr=3e-4, weight_decay=0.01, sem_table_seed=0, n_sem_ids=N_SEM_IDS, ignore_sem_ids=(0,))`, `to_dict()`.
- `make_trainer(tcfg, dataset, *, run_id, ckpt_dir, max_steps, batch_size, seed=0, amp=None, hub=None, log=None, **loop_kw) -> Trainer`: `MRH(tcfg.model, seed=seed)`, AdamW(lr, wd), `LoopConfig(num_samples=len(dataset), amp=amp or AmpConfig("fp32","cpu"), hparams={"mrh": tcfg.to_dict(), "basis_fp": dataset.basis_fp.hex()}, …)`, `batch_fn=dataset.batch`, `compute_loss=(model,batch)→mrh_loss(model(batch["taps"],batch["t"]), batch, …)`. 기저가 바뀌면 config_hash가 바뀌어 재개가 거부된다(`CheckpointMismatchError`).

## 수용 기준
`tests/mdl_03/` 전부 통과(CPU, 결정론, <60 s): 설정 검증·to_dict, 모양(H/4·비정사각), 출력 범위·단위벡터, FiLM 영 초기화(t 무관)와 섭동 뒤 t 의존, 뷰 독립, seed 결정론·전역 RNG 불변, 기본 설정 파라미터 예산·RCU 12, 입력 검증, 옥타브 재매핑·라벨 다운샘플 오라클, depth 손실 affine 불변·무효 대역 무시·numpy 오라클 대조, normal 손실 오라클, sem CE 오라클·무시 id, 가중치 0 헤드 무기울기, 지표 오라클, 샤드 왕복(DAT-02 `RealifyResult.shard_fields` → DAT-05 → 디코드 = `TapSet.features`), 기저 불일치·keep=False 거부, 소형 설정 과적합(80 스텝, 손실 ≤ 초기의 25%), 중단·재개 비트 동일, hparams에 기저 지문. `@pytest.mark.gpu`: CUDA fp16 1 스텝 유한. **미검증:** 실물 교사 탭에서의 품질(AbsRel·법선 각도, E3 문턱), sem 실제 TE-PCA 임베딩, σ̂ 학습 신호(v0 미학습).

## 하류 이음매
RD-07: `MRH.forward(TapSet.features(basis), tap_sigma)`(ext 임시 탭도 같다), 타일 혼합은 H/4 출력 위에서. MDL-04: `MRH.features(...).feat16`와 `cfg.neck_ch`. MDL-05·RD-15: `out["sem"]`과 `sem_table` 로짓. 체크포인트 hparams `basis_fp`로 기저 일치를 확인한다. 헤드 이름·채널·활성 정의를 바꾸면 이 계약서를 올린다.
