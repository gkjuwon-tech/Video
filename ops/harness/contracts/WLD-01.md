# WLD-01 세계 컨테이너 STRATA-lite-min v0 — LDI 층·far-field·meta/prov·blocks·charts·TSQ SoA·Merkle·직렬화

**목적.** S2 이후 세계 W의 가변 작업 컨테이너. 내부는 STRATA(PARADIGM-v2 §7.3)의 v0 부분집합이고, 바깥으로는 FND-02 `World`로 내보낸다. 저장 = (Merkle 루트 + 직렬화 바이트), 왕복 비트 동일.
**소유.** `buc/world/strata.py`(`buc/world/__init__.py` 없이 namespace로 import). 테스트 `tests/wld_01/`. 의존: numpy, `buc.core.contracts`(Obs·Theta·Prov·LDILayer·FarField·World·ContractError·WORLD_SCHEMA), `buc.core.det`(hash_parts·hash_array·hash_bytes·merkle_root_keyed). torch 금지, CPU import.

## 상수
`STRATA_SCHEMA = contracts.WORLD_SCHEMA` · `H0_M = 64.0` · `BLOCK_CELLS = 16` · `LEVEL_RANGE = (-8, 24)`(양끝 포함) · `MAGIC = b"BUCSTRATA0"`
`OWNERS = {"NONE":0,"INPUT":1,"INPUT_PART":2,"INPUT_REFL":3,"ENGINE":4,"TEXELFM":5,"PROC":6,"HUMAN":7,"SOUL":8}`
`BLOCK_STATES = ("empty","bedrock","proposed","committed","dirty")`
`TSQ_FIELDS`(순서 고정, 이름 → (dtype 문자열, 꼬리 shape)): `mu ("<f4",(3,))`, `normal ("<f4",(3,))`, `s_uv ("<f4",(2,))`, `opacity ("<f4",())`, `edge_n ("<f4",())`, `color ("<f4",(3,))`(L_d 선형), `level ("<i1",())`, `sem ("<u2",())`, `own ("<u2",())`, `unc ("|u1",())`.
예외 `class StrataError(contracts.ContractError)`. 모든 검증 실패·역직렬화 실패는 이것(또는 하위 ContractError)로 낸다.

## 함수
- `cell_size(l) -> float` = `64·2^-l`(정확), `texel_pitch(l)` = cell_size/8, `block_extent(l)` = 16·cell_size. l이 int가 아니거나 LEVEL_RANGE 밖 → StrataError.
- `block_of(points (N,3), l) -> int64 (N,4)` = `[l, ⌊x/E⌋, ⌊y/E⌋, ⌊z/E⌋]`, E=block_extent(l), float64로 계산. 비유한 → StrataError.
- `canon_json(x) -> bytes` = `json.dumps(x, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode()`. dict 키는 재귀적으로 str만, NaN/inf·직렬화 불가 → StrataError.

## 타입
- `@dataclass(frozen=True) Block(state: str, own: int, unc: int = 255, tsq_start: int = 0, tsq_stop: int = 0)`. 생성 시 검증(위반 → StrataError): state∈BLOCK_STATES, 정수 필드는 int(bool 제외), own∈OWNERS 값, unc∈[0,255], 0≤start≤stop. `digest() -> bytes` = `hash_parts(b"buc.strata.block.v0", state, own, unc, tsq_start, tsq_stop)`.
- `TSQTable`: `columns: dict[str, ndarray]`(키 = TSQ_FIELDS 전부, 정본 dtype·C-연속), `n`(행 수). `TSQTable()` 빈 표. `append(**cols) -> (start, stop)`: 모든 필드를 같은 행 수로 받아(누락·초과·길이 불일치·shape 불일치 → StrataError) 검증 후 덧붙인다; 실패 시 표 불변. 검증: mu·color·s_uv·opacity·edge_n 유한, normal 단위(atol 1e-3), opacity∈[0,1], s_uv>0, edge_n∈[2,64], color≥0, level∈LEVEL_RANGE, own∈OWNERS 값. dtype 변환이 값을 바꾸면(예: level 200을 i1로) StrataError.
- `Strata(*, obs: Obs, theta_in: Theta, level: str, read_hash: str, prov: Prov, seed: int = 0, meta: dict|None = None, layers=(), far_field: FarField|None = None)`. 속성: `obs, theta_in, level, read_hash, prov, seed, meta`(깊은 복사 보관), `layers: list[LDILayer]`, `far_field`, `blocks: dict[(l,i,j,k), Block]`, `charts: dict[str, dict]`, `tsq: TSQTable`, `schema = STRATA_SCHEMA`.
  - 메서드: `validate()`, `add_layer(layer) -> int`(검증 실패 시 불변), `set_far_field(ff|None)`, `set_block(key, block)`, `get_block(key) -> Block|None`, `set_chart(chart_id, rec: dict)`(rec 깊은 복사), `add_tsq(**cols) -> (start, stop)`, `composite() -> (rgb (H,W,3) f32, alpha (H,W) f32)`, `section_entries() -> dict[str, bytes]`, `merkle_root() -> bytes(32)`, `proof(key: str) -> (index, n, proof: list[bytes], item: bytes)`, `to_world() -> World`, `@classmethod from_world(world) -> Strata`, `to_bytes() -> bytes`, `@classmethod from_bytes(b) -> Strata`, `save(path)`, `@classmethod load(path)`.

## 불변식 (validate)
공개 속성(obs·theta_in·level·read_hash·prov·seed)은 직접 대입 가능하고 배열은 쓰기 가능하다. `merkle_root`·`section_entries`·`to_bytes`·`to_world`는 **호출 시점 상태**로 계산한다(낡은 캐시 금지).
seed∈[0,2^128) int · level∈contracts.LEVELS · read_hash 비지 않은 str · meta는 canon_json 가능, 예약 키 `strata_root`·`strata_seed` 금지 · 층은 obs (H,W), FND-02 World와 같은 앞→뒤 규칙(둘 다 α>1e-3인 픽셀에서 depth[i] ≤ depth[i+1]·(1+1e-6)) · block 키는 int 4-튜플, l∈LEVEL_RANGE, i,j,k∈[-2^31,2^31) · block의 tsq 범위 ⊆ [0, tsq.n] · chart_id는 `[A-Za-z0-9_.-]+`.

## composite
앞→뒤 직선 알파 over(검은 배경): `rgb = Σ_i c_i α_i Π_{j<i}(1-α_j)`, `alpha = 1-Π_i(1-α_i)`; 층 0장이면 0. float64 계산 후 f32.

## Merkle (바이트 배치 = 계약)
`merkle_root() = det.merkle_root_keyed(section_entries())`. 섹션(키 → 값):
`schema`·`level`·`read_hash` → utf-8 · `obs`·`theta_in`·`prov` → `bytes.fromhex(x.content_hash())` · `seed` → `seed.to_bytes(16,"little")` · `meta` → `hash_parts(b"buc.strata.meta.v0", canon_json(meta))` · `layer/{i:04d}` → `hash_parts(b"buc.strata.layer.v0", hash_array(rgba), hash_array(depth))` · `far_field`(있을 때만) → `hash_array(rgb)` · `block/{l}/{i}/{j}/{k}`(10진, 음수는 `-`) → `Block.digest()` · `chart/{id}` → `hash_parts(b"buc.strata.chart.v0", canon_json(rec))` · `tsq/{name}`(모든 필드, 0행도) → `hash_array(column)`.
결과는 삽입 순서와 무관하고, 어떤 값의 1 ulp 변화도 루트를 바꾼다. `proof(key)`: 잎 항목 `u64le(len(k))+k+v`의 키 바이트순 index·n·`det.merkle_proof`; `det.merkle_verify(item, index, n, proof, root)`가 참. 없는 키 → KeyError.

## 직렬화
`to_bytes()`는 `MAGIC`로 시작, 결정론(같은 내용 → 같은 바이트, blocks·charts·meta 삽입 순서 무관), 배열 비트 무손실, pickle 금지. 루트를 담고 `from_bytes`는 재계산 루트와 대조한다. 어떤 1바이트 변조·잘림·잘못된 MAGIC도 StrataError(권장: 본문 전체 blake3 꼬리, 패딩 금지). `from_bytes(to_bytes(s))`는 같은 루트, 같은 `to_bytes()`.

## World 이음매
`to_world()`: 층 ≥1 아니면 StrataError. `World(obs, layers=tuple(layers), theta_in, level, read_hash, prov, far_field, meta={**meta, "strata_root": root.hex(), "strata_seed": f"{seed:032x}"})` — blocks·charts·tsq는 World에 싣지 않는다(루트로만 결속). `from_world(w)`: 위 두 키를 meta에서 빼고 seed를 복원(없으면 0), blocks·charts·tsq 빈 상태. 블록·차트·TSQ가 빈 Strata는 `from_world(s.to_world())`가 같은 루트.

## 수용 기준
`tests/wld_01/` 전부 통과(CPU, <60 s, 동결 `WLD-01.tests.json`): 레벨 수치, block_of, composite(atol 1e-6), 불변식 위반 거부와 실패 시 불변, Merkle 섹션 바이트 오라클 일치, 삽입 순서 무관·1 ulp 민감·증명, 직렬화 왕복·결정론·변조 거부, World 왕복과 FND-02 계약 통과.

## 하류 이음매
WLD-03 `add_layer`·`add_tsq`(2DGS 초기화 = TSQ 행)·`set_far_field` · WLD-02 `Block.own`=INPUT/INPUT_PART · WLD-04 `set_chart`(rec 형식은 WLD-04 소유) · WLD-15 `section_entries`·`proof`·`to_bytes`(CoW·델타) · WLD-17 FarField · RDR·S2 이음매는 `to_world()`만 본다. v1에서 섹션 값 배치를 바꾸면 태그 `v0→v1`과 이 계약서를 함께 올린다.
