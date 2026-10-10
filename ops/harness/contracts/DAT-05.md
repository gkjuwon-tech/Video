# DAT-05 샤드 I/O v1 — WebDataset tar · Merkle 주소 · prov 검사

**목적.** 학습·개발 데이터가 디스크에 쌓이는 유일한 길. 샤드는 내용 주소(Merkle 루트)로 이름이 붙고, 모든 샘플은 계보(prov)를 달고 들어오며, 방화벽(FND-04)을 통과하지 못하면 **쓰기에서도 읽기에서도** 거부된다(닫힌 쪽 실패). 카나리아 C1(`USER`)·C5(`EXTGEN:test`)가 걸리는 곳.
**소유.** `buc/data/shards.py`, 테스트 `tests/dat_05/`(동결). 의존: FND-03 `buc.core.det`(`hash_bytes`·`hash_parts`·`merkle_root`·`merkle_root_keyed`), FND-04 `buc.gov.prov`(`as_record`·`why_not_train`·`why_not_dev`·`FirewallError`), numpy, 표준 라이브러리 `tarfile`·`json`. **`webdataset` 패키지에 의존하지 않는다**(설치돼 있지 않음. 포맷만 호환). CUDA 없이 import 가능, import 부작용 없음.
**범위 밖.** ext 샘플용 vault 샤드(HMAC 주소)는 만들지 않는다: 쓰기 경로가 ext·comp·hold·비 `train_ok` ns를 전부 거부하므로 주소는 항상 평문 blake3.

## 디스크 계약 (바꾸면 `SHARD_TAG`·`SAMPLE_TAG` 버전을 올린다)
- 파일 `<dir>/<root_hex>.tar`(root = 아래 Merkle 루트, 소문자 hex 64자) + 선택 사이드카 `<root_hex>.json`(매니페스트, 파생 캐시. 진실은 tar).
- tar: **USTAR**, 멤버 `<key>.<ext>`(key에 `.` 없음, 첫 `.` 앞이 key = WebDataset 규칙). 메타 mtime=0, mode=0o644, uid=gid=0, uname=gname="", `tarfile` 기본 블록킹. 샘플 멤버는 연속, 샘플 안은 ext 오름차순. 샘플마다 사이드카 멤버 `<key>.prov.json`(예약 ext) = `json.dumps(ProvRecord.to_dict(), sort_keys=True, separators=(",",":"))`의 ASCII 바이트(정규형). 같은 입력 → **같은 tar 바이트**.
- key `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`, ext `^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*$`(≤32자, `prov.json` 제외).
- `sample_digest = hash_parts(SAMPLE_TAG, key, prov_json_bytes, n_fields, ext_1, hash_bytes(payload_1), …, ext_n, hash_bytes(payload_n))`(ext 오름차순, 필드에 prov.json 제외, `n_fields`는 int). `SAMPLE_TAG="buc.shard.sample.v1"`.
- `root = merkle_root([sample_digest …])`(**tar 순서**). `SHARD_TAG="buc.shard.v1"`. 매니페스트 JSON(sort_keys, 무공백): `v, root, n, keys, nbytes, tar_blake3, roots, ns, prov_bits`.

## 공개 API (`buc.data.shards`)
- 예외: `ShardError(ValueError)`(형식 위반), `ShardIntegrityError(ShardError)`(주소·해시 불일치). 방화벽 위반은 `buc.gov.prov.FirewallError`를 그대로 올린다.
- `@dataclass(frozen=True) Sample(key: str, fields: dict[str, bytes], prov: ProvRecord)` — fields에 `prov.json` 없음.
- `@dataclass(frozen=True) ShardInfo(root: bytes, n: int, keys: tuple[str,...], nbytes: int, tar_hash: bytes, roots: tuple[str,...], ns: str, prov_bits: int, path: str)`; `.address -> root.hex()`, `.to_dict()`(매니페스트 dict, path 제외). `roots` = 샘플 prov.roots 합집합(정렬). `prov_bits`는 항상 0, `ns`는 항상 `"train_ok"`.
- `class ShardWriter(out_dir, *, max_samples=1000, max_bytes=None, purpose="train", branch="main")`; 컨텍스트 매니저. `write(key: str, fields: Mapping[str, bytes-like], prov: ProvLike) -> None`, `close() -> list[ShardInfo]`, 속성 `infos`(끝난 샤드). 한 샘플을 쓴 뒤 `n==max_samples` 또는 페이로드 합 ≥ `max_bytes`면 샤드를 닫는다. 샤드는 임시 파일에 쓰고 `os.replace`로 `<root_hex>.tar` 확정(같은 이름이 이미 있고 바이트가 같으면 멱등, 다르면 `ShardIntegrityError`). 샘플이 0개면 파일 없음. `with` 중 예외면 열린 샤드를 버리고(부분 샤드·임시 파일 남기지 않음) 이미 닫힌 샤드만 남긴다. close 뒤 write는 `RuntimeError`. `out_dir`은 없으면 만든다. `purpose ∈ {"train","dev"}`, `branch ∈ {"main","optin-v1"}`(dev는 main만), 아니면 `ValueError`. `max_samples<1`·`max_bytes<1` `ValueError`.
- `verify_shard(path, *, expected_root=None, purpose="train", branch="main") -> ShardInfo`: tar 전체를 읽어 루트를 재계산. 주소 비교 대상 = `expected_root` 또는 파일명 stem(64 hex일 때). 불일치·사이드카 매니페스트와 불일치 → `ShardIntegrityError`; tar 손상·잘림·샘플 0개·멤버 이름이 `<key>.<ext>`(위 정규식)가 아니거나 일반 파일이 아님·prov 사이드카 없음/비정규형/잘못된 버전·필드 없는 샘플·중복 key·불연속 key·중복 ext → `ShardError`. 그 다음 모든 샘플에 방화벽 → `FirewallError`.
- `iter_shard(path, *, purpose="train", branch="main", verify=True) -> Iterator[Sample]`(지연: 오류는 첫 `next()`). verify=True면 먼저 `verify_shard`(실패하면 한 샘플도 내보내지 않음). verify=False여도 샘플 단위 형식 검사(사이드카 필수·정규형)와 **방화벽 위반 샘플 차단은 그대로**다(위반 샘플 앞의 정상 샘플만 나간다). `iter_shards(paths, **kw)`는 주어진 순서로 이어 붙인다.
- `dataset_root(roots: Iterable[bytes]) -> bytes` = `merkle_root_keyed({r.hex(): r})`. 샤드 목록 순서와 무관, 중복 `ValueError`.
- `find_shards_with_root(directory, root: str) -> list[str]`: `^[0-9a-f]{64}\.json$` 매니페스트 중 `roots`에 root가 든 tar 경로(정렬). 깨진 매니페스트 `ShardError`. 삭제 잡(GOV-01)용.
- `npy_encode(a: np.ndarray) -> bytes`(C순서로 정규화, pickle 금지 → object dtype `ValueError`), `npy_decode(b: bytes) -> np.ndarray`.

## 방화벽 규칙 (쓰기·읽기 동일)
purpose train: `why_not_train(rec, branch)==()`, dev: `why_not_dev(rec)==()`. 여기에 **샤드 불변식 추가**: `rec.ext|comp|hold` 비트 0, `rec.ns=="train_ok"`(FND-04는 ns를 판정에 안 쓰므로 여기서 닫는다). 위반은 `FirewallError`, 메시지에 key. 거부된 `write`는 상태를 바꾸지 않는다(이후 쓰기 가능). prov는 필수(None·str → `TypeError`, 기본값 없음). 입력 검증: key 형식·writer 수명 안 중복 `ValueError`(타입 `TypeError`), fields 비었거나 ext 형식·`prov.json` 사용 `ValueError`, 페이로드 non-bytes-like `TypeError`.

## 수용 기준
`tests/dat_05/` 전부 통과(CPU, <60 s): 바이트 배치(독립 오라클이 만든 tar·루트와 비트 동일), 같은 입력 두 디렉터리 동일 파일명·바이트, 롤오버(개수·바이트), 라운드트립 비트 동일, 중단 시 잔여물 없음, 방화벽 진리표와 시드 고정 무작위 300건(쓰기 허용 ⇔ train_ok ∧ 비트 0 ∧ ns), 우회 제작 tar(루트 일치하는 `USER` 샤드)를 읽기에서 거부, 변조(바이트 반전·잘림·개명·prov 삭제·비정규 JSON)를 검출, dev/train 목적 분리, `dataset_root` 셔플 불변, 삭제용 루트 검색.

## 하류 이음매
DAT-02/07 `ShardWriter`(샘플 = png/npy + prov: 부모 prov를 `derive`한 레코드) · MDL-01/03·TCH-03 `iter_shards`(학습 입력은 반드시 이 경로) · RUN-06 샘플러가 `ShardInfo.n/keys`로 결정론적 순서 · GOV-01 Lineage CI 규칙 1은 `verify_shard`, 삭제 잡은 `find_shards_with_root`, 데이터 매니페스트는 `dataset_root`. 샤드 안 필드 이름(`png`, `npy`, …)의 의미는 각 생산 단위의 계약이다.
