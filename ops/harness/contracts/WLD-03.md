# WLD-03 깊이 세계 빌더 v0 — 미터 깊이 → 불연속 찢기 → 2층 LDI → 2DGS(TSQ 행) 초기화 · 매팅 합류 자리

**목적.** S2 실물. `ReadPacket`(RD-21: `depth`=미터 z-깊이, `image_lin`, `depth_conf`)으로 `Strata`(WLD-01)와 `World`를 짓는다. 고무 시트를 막는 세 장치: (1) 깊이 불연속(상대 점프 > `rel_jump`)에서 **찢는다**(깊이 평활·보간 금지, 화소 깊이 그대로), (2) 전경 뒤로 배경을 `ext_px`만큼 잇는 **숨은 층**(LDI 층 1)을 둔다(가림 해제가 구멍·전경 늘어남 대신 배경 깊이로 보인다; 색은 임시, WLD-08이 덮는다), (3) RD-18 매팅이 주어지면 띠 화소를 `α·F`(전경 깊이)와 `B`(배경 깊이)로 **분해**한다(혼합색이 배경에 남는 halo 방지). 관측 시점 합성은 입력과 같다.
**소유.** `buc/world/depth_world.py`, 테스트 `tests/wld_03/`(동결 `WLD-03.tests.json`). **의존.** numpy·scipy(`ndimage`), FND-02 `contracts`, FND-06 `geom.unproject_world`, WLD-01 `strata`(`Strata`, `OWNERS`, `LEVEL_RANGE`), RD-18 `edge_unmix.depth_edges`(선택; 같은 식). `torch`·`buc.render.*`·`buc.pipeline.spine`은 **함수 안 지연 import**(모듈 import 시 `buc.render.*`는 로드되지 않는다; torch는 WLD-01→`det` 경유로 이미 로드될 수 있다. CUDA 초기화 금지). import 부작용·전역 RNG 변경 없음, 입력 불변, 결정론.

## 공개 API
```python
NAME = "depth_world"; VERSION = "0.1.0"; N_LAYERS = 2
class DepthWorldError(contracts.ContractError)
@dataclass(frozen=True, kw_only=True) class DepthWorldConfig:
    rel_jump=0.10 (>0 유한); ext_px=16.0 (≥0 유한); near_m=1e-3 (>0); far_m=1e4 (>near_m, 유한)
    tsq_stride=2 (int≥1, bool 금지); alpha_eps=1e-3 ([0,0.5)); opacity=0.99 ((0,1]); scale_mult=0.5 (>0); edge_n=2.0 ([2,64])
    # 위반 ValueError
@dataclass(frozen=True, kw_only=True) class DepthWorld:
    strata: Strata; world: World            # world == strata.to_world()
    z: (H,W) f64; sky, near, far, hidden, matted: (H,W) bool
    tear_x: (H,W-1) bool; tear_y: (H-1,W) bool
    n_tsq: int; tsq_layer: (n_tsq,) int8
    def meta(self) -> dict
def build_depth_world(read: ReadPacket, *, cfg=None, matte=None, far_field: FarField|None=None) -> DepthWorld
def tsq_to_surfels(cols: Mapping[str, ndarray], obs: Obs, *, dtype=None) -> Surfels     # RDR-01, 지연 torch
def depth_world_stage(inputs: dict, ctx) -> {"world": World}                            # S2 {read}→{world}
def make_world_impl() -> spine.Impl                                                      # Impl("S2","real",NAME,VERSION,depth_world_stage)
```
`read`가 `ReadPacket`이 아니면 `TypeError`, cfg가 `DepthWorldConfig`/None이 아니면 `TypeError`, far_field가 FarField/None이 아니면 `TypeError`. `matte`는 덕 타입(RD-18 `UnmixResult` 호환): 속성 `alpha (H,W)`, `fg (H,W,3)`, `bg (H,W,3)`, `band (H,W) bool`. 속성 없음 `TypeError`; 모양 불일치·비유한·alpha∉[0,1]·fg/bg<0 `ValueError`.

## 알고리즘 (관측 가능한 결과 = 계약)
기호: `I=image_lin`(f64), `D=read.depth`, `K,R,t,k1,k2 = read.obs`.
1. **깊이.** `sky = ¬isfinite(D) ∨ D ≥ far_m`; `z = where(sky, far_m, max(D, near_m))`(f64, 그 밖 변형 없음).
2. **찢기.** 4-이웃 쌍 (p,q)가 `|z_p−z_q| > rel_jump·min(z_p,z_q)`이면 찢김. `tear_x[y,x]`=(y,x)–(y,x+1), `tear_y[y,x]`=(y,x)–(y+1,x). 찢긴 쌍의 작은 z 쪽 `near`, 큰 z 쪽 `far`(RD-18 `depth_edges(z, rel_jump)`와 같다).
3. **매팅(matte 있을 때, 띠 화소 p).** `near`·`far`가 모두 비지 않으면 `zf=z[EDT 최근접 near]`, `zb=z[EDT 최근접 far]`(`scipy.ndimage.distance_transform_edt(~mask, return_indices=True)`). `zb > zf·(1+rel_jump)`이면 `matted[p]=True`.
4. **숨은 층.** `far`가 비지 않으면 `d, q` = `~far`의 EDT 거리·최근접 far 화소. `hidden = (d ≤ ext_px) ∧ (z[q] > z·(1+rel_jump)) ∧ ¬matted`. 색 `c_h = B[q]`(matte 있고 `band[q]`) 아니면 `I[q]`; 깊이 `z[q]`.
5. **층**(정확히 2장, 앞→뒤, rgb는 `clip(·,0,1)` f32, 직선 알파):
   - 층 0: 기본 `(I, 1, z)`. matted 화소: `(F_p, α_p, zf)`, 단 `α_p ≤ alpha_eps`면 α=0.
   - 층 1: matted `(B_p, 1, zb)`, hidden `(c_h, 1, z[q])`, 그 밖 `(0, 0, z)`.
   - 따라서 관측 시점 합성 = `clip(I)`(matte 없음, 1e-6), matte 있음은 `αF+(1−α)B`(띠 밖은 I).
6. **TSQ 행**(2DGS 초기화). 층 0 → 층 1 순, 각 층 안 격자 `v=0,s,2s,…`·`u=0,s,…`(s=`tsq_stride`) 행 우선, 그 층 α > alpha_eps인 화소만. 열:
   `mu = geom.unproject_world(K,R,t,(u,v),z_L,k1,k2)` · `normal = −R[2,:]`(카메라를 향함) · `s_uv = scale_mult·s·z_L·(1/fx, 1/fy)` · `opacity = cfg.opacity·α_L` · `edge_n = cfg.edge_n` · `color = max(원색,0)`(층 0: I 또는 F, 층 1: B 또는 c_h, **클립 안 함**) · `level = clip(rint(log2(8·fx/(s·z_L))), LEVEL_RANGE)` · `sem = 0` · `own`: 층 0 비매팅 INPUT(1), matted 화소(두 층) INPUT_PART(2), hidden ENGINE(4) · `unc`: 층 0 = `rint(255·(1−depth_conf))`(depth_conf None이면 255), 층 1 = 255. 행이 ≥1이면 `add_tsq` 한 번.
7. **Strata/World.** `Strata(obs=read.obs, theta_in=read.theta_in, level=read.level, read_hash=read.content_hash(), prov=read.prov, seed=0, meta={"depth_world": meta()})`, `add_layer`×2, `set_far_field(far_field)`(None이면 생략; 하늘 화소는 v0에서 층 0의 `far_m` 깊이에 남는다). blocks·charts는 비운다(WLD-02/04 몫). `world = strata.to_world()`.
8. `tsq_to_surfels`: means=mu, `log_scales=log(s_uv)`, `opacity_logits=logit(clip(opacity,1e-6,1−1e-6))`, colors=color, 회전 열 `(t_u, t_v, n)`: `t_u = normalize(R[0,:] − (R[0,:]·n)n)`, `t_v = n × t_u`(det +1). 결과 `Surfels.normals() = normal`. 기본 dtype float32.
9. **스테이지.** `depth_world_stage`: params = `ctx.params`에서 `_seed` 뺀 것 → `DepthWorldConfig(**params)`(모르는 키 `ValueError`), matte·far_field 없음.

## meta()
`json.dumps(allow_nan=False)` 가능, 키 ⊇ `name, version, cfg(dict), n_layers, n_tsq, n_tear_pairs, hidden_frac, matted_frac, sky_frac, matte(bool), far_field(bool)`.

## 불변식
결정론(같은 입력 → 같은 `merkle_root`·`to_bytes`) · `read` 불변 · NaN·경고 없음 · `check_stage_io("S2")` 통과 · 층 깊이는 {z, zf, zb, z[q]} 값만 쓴다(평활·보간 깊이 없음) · 찢김 없는 매끈한 기울기 면(인접 비 < 1+rel_jump)은 hidden·tear 0 · `world.meta["strata_root"] == strata.merkle_root().hex()`.

## 수용 기준 (`tests/wld_03/`, CPU <60 s)
API·검증·지연 import(`buc.render.*` 미로드) · 깊이 클립/하늘 · 찢김 오라클(9%/11% 양쪽, 매끈 경사 0) · 층 공식·합성 항등(matte 유무) · TSQ 열 오라클(왜곡·회전 관측, 재투영 1e-3 px) · S2 이음매·Strata 루트·far_field 전달 · **시각 대리 지표**(RDR-02 `render_ldi`, 전경 2 m/배경 6 m, 옆 0.3 m 이동): 실루엣 띠 밖 중간 깊이 ≤0.5 %, 가림 해제 띠 α≥0.99·깊이 6 m ≥95 %(`ext_px=0`이면 구멍 ≥80 %), 관측 영역 GT 대비 평균 색 오차 ≤0.01 · **halo**: 혼합 링 장면에서 matte 없음 보라 화소 ≥15, matte 있음 ≤2 · 실제 RD-18 `edge_unmix` 합류 합성 ≈ I · RDR-01 `init_from_rgbd`와 surfel·렌더 일치(1e-4) · 결정론·meta JSON.
**미검증(V, 창업자 눈):** 실사진의 찢김 품질·숨은 층 색(임시 최근접 복사)·털·잎 경계, 2K 속도, 하늘의 far_field 분리(v0는 층 0에 둔다).

## 하류 이음매
WLD-08: `world.layers[1]`의 `own=ENGINE` 행/`hidden` 마스크를 발명 채움 대상으로 덮는다. WLD-07: `tsq_to_surfels(strata.tsq.columns, obs)` → RDR-01 `fit`. WLD-05·20: `world`·`strata`. RD-21/RD-18: 패킷에 `unmix`가 실리면 스테이지가 `matte`로 넘긴다(v1, 이 계약서와 함께 올린다). WLD-17: `far_field` 그대로 `set_far_field`. RDR-02: `render_ldi(world, …)`.
