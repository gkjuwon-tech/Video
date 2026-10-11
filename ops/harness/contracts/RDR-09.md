# RDR-09 Proof-of-Shot·왕복 비트 동일 검증 v0

**목적.** "같은 입력 바이트면 같은 샷 바이트"라는 헤드라인 주장을 **누구나 다시 돌려 확인할 수 있는 증거**로 만든다. (1) 샷의 프레임 지문을 Merkle 루트로 묶은 `ProofOfShot`을 만들고 검증하고, (2) 정확히 대칭인 왕복(나갔다 돌아오는) 경로에서 돌아오는 프레임이 나가는 프레임과 **바이트까지 같음**을 판정한다. 영상 인코딩(RDR-06)은 범위 밖이며 증거는 `Frames` 바이트 기준이다.
**소유.** `buc/render/pos.py`(namespace, `__init__.py` 없음), 테스트 `tests/rdr_09/`(동결 `RDR-09.tests.json`). 의존: RDR-05 `buc.render.aov`(`render_shot_aov`, `observed_atoms`), RDR-02 `RenderOpts`, FND-02(`World`, `Shot`, `Frames`), FND-03 `buc.core.det`(`hash_parts`, `hash_array`, `merkle_*`). numpy·scipy·torch(CPU)·blake3만, GPU 없이 import·실행, import 부작용·전역 RNG 변경 없음.

## 지문 (바이트 배치는 계약)
`frame_leaf(frames, i, *, keys=None) -> str`(64자 소문자 hex) = `hash_parts(b"buc.pos.frame.v1", hash_array(rgb[i]), k1, hash_array(aov[k1][i]), k2, ...).hex()`. `keys=None`이면 frames의 모든 aov 키, 아니면 `sorted(set(keys))`(없는 키 `ValueError`, i 범위 밖 `IndexError`). 키는 정렬해서 쓰므로 dict 삽입 순서와 무관. **인덱스는 지문에 넣지 않는다**(왕복 비교의 전제). 비트 단위: −0.0과 +0.0, dtype 차이도 다른 지문. `frame_digests(frames, *, keys=None) -> tuple[str,...]`. `leaf_root(leaves) -> str` = `merkle_root([bytes.fromhex(x)...]).hex()`(빈 입력은 `EMPTY_ROOT`, 64-hex가 아니면 `ValueError`). 상수 `SCHEMA="buc.pos/v1"`, `RENDERER_PIN: str`, `ROUNDTRIP_KEYS=("alpha","atom_id","depth","layer_id","normal")`(flow·flow_valid는 i→i+1이라 왕복에서 뒤집히므로 제외). `toolchain_pin() -> dict[str,str]` 키는 정확히 `renderer(=RENDERER_PIN), python, numpy, scipy, torch, blake3, machine, system`, 호출마다 새 dict. `class PosError(ValueError)`.

## IR
`ir_geo(atom_id: (N,H,W) int, bits: ObsBitset) -> IRGeo(per_frame: tuple[float], mean: float, max: float)`(frozen dataclass, 파이썬 float). 프레임별 `1 - 관측화소/유효화소`(유효 = id>=0, 정수 계수 후 나눗셈), `mean`은 **화소 풀링** `1 - Σ관측/Σ유효`, `max = max(per_frame)`. 유효 화소가 0이면 0.0. 3차원이 아니면 `ValueError`.

## 증명
`opts_dict(opts=None) -> dict`: `RenderOpts` 필드 6개(`bg`는 float 리스트). `ProofOfShot`(frozen, kw_only): `schema, shot_hash, world_hash, theta_hash(=shot.theta.content_hash()), toolchain, opts, n_frames, width, height, fps(float), aov_keys(정렬 tuple), leaves(tuple[str]), root, ir, roundtrip, extra`. `ir = {"obs_hash"(=observed_atoms(world).content_hash()), "ir_geo_mean", "ir_geo_max", "per_frame": list}`. `roundtrip = None`(샷이 비대칭이거나 n<2) 아니면 `{"keys": list(ROUNDTRIP_KEYS 중 aov에 있는 것), "pairs", "identical", "max_abs_diff"}`. `__post_init__`은 구조만 검사(schema, 해시 필드 64-hex, `len(leaves)==n_frames`; 위반 `PosError`). 루트 일치는 검증이 판정한다. `to_dict()`(JSON 가능, 새 객체), `from_dict(d)`(엄격: 모르는/빠진 키·dict 아님 `PosError`), `to_json()` = `json.dumps(to_dict(), sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)`, `from_json`(JSON 오류 `PosError`), `content_hash() = hash_parts(b"buc.pos.v1", to_json().encode()).hex()`. `write_proof(proof, path) -> Path`(상위 디렉터리 생성, `<이름>.part`에 쓰고 `os.replace`, 내용 = `to_json()+"\n"`), `read_proof(path)`(없음 `FileNotFoundError`, 깨짐 `PosError`).

`make_proof(world, shot, *, opts=None, frames=None, extra=None) -> ProofOfShot`: `frames=None`이면 `render_shot_aov(world, shot, opts)`. 준 `frames`는 `shot_hash`/`world_hash`가 일치해야 한다(`ValueError`). `atom_id` aov가 없으면 `PosError`, world/shot 타입 오류·위치 인자 초과 `TypeError`. `extra`는 JSON 값(str 키 dict, list, str, int, 유한 float, bool, None)만, 아니면 `PosError`; 깊은 복사해 보관, 검증 대상이 아니다(Soul 자산 해시·플랜 단계 등을 싣는 자리).

`verify_proof(proof, world, shot, *, frames=None) -> Verdict`: 불일치로는 예외를 내지 않는다. `frames=None`이면 **증명에 적힌 opts로** 재렌더. `Verdict`(frozen) 필드 `ok, world_ok, shot_ok, theta_ok, toolchain_ok, shape_ok, root_ok, frames_ok, ir_ok, roundtrip_ok`(파이썬 bool), `bad_frames`(오름차순 tuple[int]), `reason`(실패한 검사 이름을 이 순서로 쉼표 연결, 없으면 `""`). `shape_ok` = 프레임 수·H·W·fps·aov 키 일치(아니면 `frames_ok=ir_ok=roundtrip_ok=False`, `bad_frames=()`). `root_ok` = `leaf_root(proof.leaves)==proof.root`. `frames_ok` = 재계산 지문이 모두 일치(`bad_frames`는 불일치 인덱스). `ir_ok`/`roundtrip_ok` = 재계산 dict가 증명과 같음(주장 삭제·대칭 아닌 샷의 왕복 주장도 불일치). **`toolchain_ok`는 참고용**이며 `ok`에 들어가지 않는다(바이트가 같으면 성립). `ok` = toolchain·extra를 뺀 나머지의 AND.
포함 증명: `inclusion_proof(proof, i) -> list[str]`(범위 밖 `IndexError`), `verify_inclusion(proof, i, leaf, path) -> bool`(예외 없이 False).

## 왕복
`mirror_index(n) -> int64[n]` = `n-1-i`(n<0 `ValueError`). `out_and_back(shot) -> Shot`: n>=2 필요(`ValueError`), 프레임 `0..n-1, n-2..0`(2n-1장, 정점 한 번), `c2w/K/focus_m`은 비트 복사, `t=arange/fps`, 나머지 필드 유지. `is_palindromic_shot(shot, *, atol=0.0) -> bool`: `c2w, K, focus_m`이 `[i]`와 `[n-1-i]`에서 같은가(`atol=0`은 바이트 정확, 아니면 `allclose(rtol=0)`). `lsb_diff(a, b) -> int`: `rint(clip(x,0,1)*255)` 정수 최대 차(모양 불일치 `ValueError`). `RoundTrip`(frozen, `eq=False`): `n_frames, pairs(=n//2), identical, pair_identical(tuple[bool]), first_mismatch(int|None), keys, max_abs_diff(rgb, float), max_lsb_diff, diff_map (pairs,H,W) float32`. `roundtrip_check(frames, *, keys=None)`: 프레임 i와 n-1-i(i<n//2)를 rgb와 `keys`(None이면 `ROUNDTRIP_KEYS`∩aov, 정렬)의 **바이트**로 비교(−0.0≠+0.0). `verify_roundtrip(world, shot, opts=None)`: 대칭이 아니면 `ValueError`, 렌더 후 `roundtrip_check`.

## 불변식
결정론(같은 world·shot·opts·툴체인 → 같은 증명 JSON 바이트, 독립 객체로 다시 만들어도) · 입력 불변 · RuntimeWarning 없음 · 샷 프레임 순서·계산 순서와 무관한 프레임 지문 · 지문은 1 ulp와 aov 한 칸에도 민감 · 정확히 대칭인 경로에서 `identical=True`, `diff_map` 전부 0.

## 수용 기준
`tests/rdr_09/` 전부 통과(CPU, <60 s, 동결 `RDR-09.tests.json`): 지문 오라클·비트 민감도, Merkle 루트, IR 해석값(합성·항등·정수 이동), 왕복 비트 동일과 음성 대조군(1 ulp, aov 한 칸, 잡음, 근사 대칭), 증명 필드·결정론·직렬화·엄격 파싱, 월드·샷·옵션·잎·루트·IR·왕복 주장 변조 탐지, 포함 증명, 파일 왕복. GPU 테스트 없음(렌더러가 CPU 경로).

## 하류 이음매
PRD-05는 `proof.content_hash()`를 `KPI.proof_of_shot`에, `proof.ir`를 IR 칸에 싣는다. EVL-02/05는 `ir_geo`·`roundtrip_check`를 재사용한다. 공개 검증기(Q13)는 `read_proof` → `verify_proof`만 호출한다. 교차 아키텍처 "1 LSB 이내" 층은 원본 프레임이 있어야 하므로 `lsb_diff`만 제공하고 해시 층은 같은 툴체인 전용이다. 키·배치를 바꾸면 `v1`→`v2`와 이 계약서를 함께 올린다.
