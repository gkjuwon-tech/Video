# FND-03 결정론 유틸 v0 — Philox WNF · int64 segment-reduce · blake3 Merkle

**목적.** 비트 동일 재현(Proof-of-Shot, T8, order_purity)의 바닥. 아래 바이트 배치는 캐시 키·원장·샤드 주소로 디스크에 남으므로 구현 상세가 아니라 **계약**이다.
**소유.** `buc/core/det.py`, 테스트 `tests/fnd_03/`(동결). 의존: numpy, torch(CPU), `blake3`. CUDA 없이 import 가능.
**소유 밖 요청.** pyproject 의존성에 `blake3>=1.0` 추가(FND 단위). `buc/core/__init__.py`는 없어도 namespace로 동작.

## 공개 API
**Philox / WNF**
- `philox4x32(ctr: uint32[...,4], key: uint32[...,2], rounds=10) -> uint32[...,4]`. Random123 Philox4x32-R(승수 0xD2511F53/0xCD9E8D57, 키 증가 0x9E3779B9/0xBB67AE85), 브로드캐스트, 입력 불변. dtype≠uint32 `TypeError`, 마지막 축 크기 오류 `ValueError`.
- `wnf(seed, level, cell, k=0, channels=4, dtype=np.float32) -> [*cell.shape, channels]` 표준정규. 순수 함수(전역 RNG 불변), 배치·순서·cell 정수 dtype과 무관(ρ=1). 스칼라 cell → `(channels,)`.
  키 `(seed & 0xFFFFFFFF, seed >> 32)`. 블록 b의 카운터 `(cell & 0xFFFFFFFF, cell >> 32, (level & 0xFFFF) | (b << 16), k)`. 출력 `(w0,w1)`,`(w2,w3)`마다 Box-Muller(`u=(w+0.5)/2^32`, `r=√(−2 ln u1)`, `(r cos 2πu2, r sin 2πu2)`). 블록 하나 = 4채널, 앞에서 `channels`개. float64로 계산 후 `dtype` 캐스트.
  범위 위반 `ValueError`: seed∈[0,2^64), level∈[0,2^16), k∈[0,2^32), cell∈[0,2^63), channels≥1. cell이 정수 아님 `TypeError`.
- `cell_id(i, j) -> int64 = (i<<31)|j` (i∈[0,2^32), j∈[0,2^31), 아니면 `ValueError`). `wnf_tile(seed, level, i0, j0, h, w, k=0, channels=4, dtype) -> [h,w,channels]`: apron이 겹치는 타일은 겹친 셀에서 비트 동일.
- `derive_seed(seed, *parts) -> int[0,2^64)` = `int.from_bytes(hash_parts(seed,*parts)[:8],"little")`.

**고정소수점 리덕션**
- `fixed_encode(x, frac_bits) -> int64`: `round_half_even(x·2^frac_bits)` 정확. frac_bits∈[1,48] 아니면 `ValueError`; 비유한 `ValueError`; |결과|≥2^63 `OverflowError`. `fixed_decode(q, frac_bits) -> float64`.
- `canonical_order(segment_ids, view_ids=None) -> int64[n]`: (segment, view) 사전식 **안정** 정렬 인덱스.
- `segment_reduce(values, segment_ids, num_segments, view_ids=None, weights=None, frac_bits=24) -> np.int64[num_segments, *values.shape[1:]]`. 입력은 numpy/torch(CPU). 항 = `fixed_encode(float64(v)·float64(w), frac_bits)`(항별 양자화) 후 세그먼트별 정수 합. **순서·view_ids 유무와 무관하게 비트 동일**, 빈 세그먼트 0. 길이 불일치·id 범위 밖·비유한 `ValueError`. **최종 합**이 int64 밖이면 `OverflowError`(감김 금지, 중간 부분합은 무관).
- `segment_reduce_int_torch(q: int64 Tensor, segment_ids, num_segments) -> int64 Tensor`: `index_add_`, CPU·CUDA 동일. dtype≠int64 `TypeError`, id 범위 밖 `ValueError`, 오버플로 검사 없음.

**해시·Merkle** (blake3, 32바이트 `bytes`)
- `hash_bytes(bytes-like)` 표준 blake3. `hash_parts(*parts)` = H(`b"buc.parts.v1"` + u32le(개수) + 각 `태그(1)+u64le(길이)+페이로드`), 태그 bytes-like `b` / str(utf-8) `s` / int `i`(16바이트 LE signed, −2^63≤v<2^64). 그 외·bool `TypeError`, 범위 밖 `OverflowError`.
- `hash_array(a)` = H(`b"buc.array.v1"` + 리틀엔디안 dtype 문자열(`<f4`,`|u1`) + `\0` + u8(ndim) + u64le(shape…) + C순서 LE 바이트). 논리 값 기준(메모리 배치·엔디안 무관), numpy/torch/스칼라. object dtype `TypeError`.
- `merkle_leaf(x)=H(0x00‖x)`, `merkle_node(l,r)=H(0x01‖l‖r)`, `EMPTY_ROOT=H(b"buc.merkle.empty")`, `merkle_root(items)`: 홀수 노드는 복제 없이 **승격**. 비 bytes-like `TypeError`.
- `merkle_proof(items, index) -> list[bytes]`(아래→위 형제, 승격 레벨 생략, 범위 밖 `IndexError`), `merkle_verify(item, index, n, proof, root) -> bool`(예외 없이 False).
- `merkle_root_keyed(entries: Mapping | Iterable[(key,value)])`: key bytes/str(utf-8), value bytes-like(아니면 `TypeError`). 정규화 key 중복 `ValueError`. key 바이트순 정렬, 잎 = `u64le(len(key))+key+value`. **입력 순서와 무관**. 빈 입력 `EMPTY_ROOT`.

**솔버 경로** `enable_determinism(seed=None)`: `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` 미설정 시 `":4096:8"`. seed가 있으면 random·numpy·torch(CUDA 포함) 시드, 없으면 시드 불변.

## 불변식
WNF 값은 (seed, level, cell, k, 채널)만의 함수 · 정수 리덕션은 어떤 순열에도 비트 동일 · Merkle 도메인 분리(0x00/0x01)로 second-preimage·CVE-2012-2459형 모호성 없음 · 모든 해시 입력은 길이 프레이밍 · import 부작용 없음.

## 수용 기준
`tests/fnd_03/` 전부 통과(CPU, <60 s): Random123 KAT와 순수 파이썬 오라클 대조, 정규성 통계, 카운터 배치 고정, 타일 겹침 ρ=1, 항별 정확 오라클과 8회 셔플 비트 동일, 오버플로 경계, 해시·Merkle 바이트 배치 고정, 모든 인덱스 증명, 8회 셔플 `merkle_root_keyed` 불변. `@pytest.mark.gpu`: `segment_reduce_int_torch` CUDA=CPU.

## 하류 이음매
RUN-06 `derive_seed` · TCH-01/04 `hash_parts`/`hash_array` · RDR-05/09 `canonical_order`/`segment_reduce` · WLD-01/15 `merkle_root_keyed`/`merkle_proof` · DAT-05 샤드 주소 · MDL-10 `wnf_tile` · GOV-04·EVL-11 셔플 후 루트 비교. 배치 변경은 호환 파괴이므로 태그 버전(`v1`→`v2`)과 이 계약서를 함께 올린다. float32 WNF 비트 동일은 같은 numpy/libm 빌드(동결 툴체인) 안에서만 보장, 교차 플랫폼은 float64 기준 1e-9.
