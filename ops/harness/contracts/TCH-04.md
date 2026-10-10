# TCH-04 Exhaust Ledger v0 — blake3 키 · int8 x̂0 · ext/hold 분리 · 세션 재개

**목적.** 교사 호출 산출(x̂0)을 키로 주소 지정해 int8로 저장·재사용한다. 쓰임은 (1) 세션 재개 (2) 같은 프로젝트 Merkle 델타 솔브뿐이다(권리 렌즈: ext는 세계 간 공유·학습 금지). **키가 방화벽의 일부**다: (ext, comp, hold, ns) 비트가 키에 들어가고, ext 키는 프로젝트 키 없이는 계산·열거할 수 없다.
**소유.** `buc/teacher/exhaust.py`, 테스트 `tests/tch_04/`(동결). 의존: TCH-02(`SigmaTable`, `SigmaError`), FND-03(`det.hash_parts`; `blake3`), FND-04(`as_record`, `merkle_parts`, `content_address_kind`, `train_ok`, `FirewallError`). import는 torch·numpy·blake3·위 모듈만, 부작용 0.
**소유 밖 요청.** 실제 봉인(K_P, KMS)은 `Sealer`를 꽂는 쪽(GOV/RUN)의 몫. 이 단위는 암호를 구현하지 않고 sealer 없는 ext 쓰기·읽기를 `FirewallError`로 거부한다.

## 공개 API (`buc.teacher.exhaust`)
- 상수 `KEY_TAG="buc.tch04.key.v1"`, `QUANT_MODE="int8-sym-ch-v1"`, `Q_MAX=127`, `EXT_TTL_S=72*3600`, `PURPOSES=("resume","delta","share","train")`. 예외 `ExhaustError(RuntimeError)` ⊃ `ExhaustSpecError(+ValueError)`, `ExhaustNumericError(+FloatingPointError)`; `FirewallError`는 FND-04 것을 재노출. `__all__` 선언.
- `quantize_x0(x0: Tensor[C,h,w]) -> (q: int8[C,h,w], scale: float32[C])`: 채널별 `scale=max|x|/127`(float32), `q=round_half_even(x/scale)`를 [-127,127]로 clamp; 전부 0인 채널은 `scale=0,q=0`; 다른 부동소수 dtype은 float32로 올려 계산, 입력 불변·detach. 비유한 `ExhaustNumericError`, ndim≠3·비부동소수 `ExhaustSpecError`. `dequantize_x0(q, scale)=q.float()*scale[:,None,None]`. 오차 `|x-deq| ≤ scale/2`.
- `KeyParts(world: bytes[32], view: str, sigma: float, seed: int, teacher_id: str, caption_hash: str(소문자 hex 64), table: SigmaTable, height: int, width: int, level=0, k=0, precision="bf16", quant=QUANT_MODE)` frozen dataclass. 생성 시 검증(정수는 bool 불가, seed∈[0,2^64), height/width>0, level/k≥0, 문자열 비어 있지 않음)해 `ExhaustSpecError`; `sigma`가 표의 점이 아니면 `SigmaError`.
- `compute_key(parts, prov, *, project_id=None, project_key=None) -> str`(hex 64). `body = det.hash_parts(KEY_TAG, *merkle_parts(prov), content_address_kind(prov), proj, world, view, struct.pack("<d", table_point), seed, level, k, teacher_id, caption_hash, table.key, height, width, precision, quant)`, `proj = project_id if ext else ""`, `table_point = table.points[table.index_of(sigma)]`. 비-ext 키 = `body.hex()`; ext 키 = `blake3.blake3(body, key=project_key).hexdigest()`. ext인데 `project_id`/`project_key`가 없으면 `FirewallError`, 형식 오류(빈 id, 32바이트 아닌 키)는 `ExhaustSpecError`. 비-ext에서 project 인자는 무시.
- `teacher_id(teacher) -> str` = `f"{spec.repo}@{spec.revision or '-'}|{dtype}|{fingerprints['transformer'].hex()}|{fingerprints['vae'].hex()}"`; `fingerprints is None`이면 `ExhaustSpecError`. `caption_hash`는 `teacher.text.key(prompt)`.
- `Sealer`(Protocol): `seal(plaintext, *, aad: bytes, project_id: str) -> bytes`, `open(blob, *, aad, project_id) -> bytes`(변조 시 예외).
- `ExhaustEntry(key, q, scale, stats: dict[str,float], prov: ProvRecord, created: float)` + `x0_hat`, `nbytes`(디스크 파일 크기); `PutResult(key, created, conflict, nbytes)`. `allowed_purposes(prov)`: 항상 `("resume","delta")`, `bits==0 ∧ train_ok(prov)`이면 `+("share","train")`.
- `ExhaustLedger(root, *, sealer=None, clock=time.time, ext_ttl_s=EXT_TTL_S)`: `root`, `stats`(복사본: hits, misses, writes, conflicts, corrupt, expired, denied).
  - `key(...)`=`compute_key`; `entry_path(key, prov, *, project_id=None)`: 비-ext `root/<ns>/<key[:2]>/<key>.exh`, ext `root/<ns>/<proj16>/<key[:2]>/<key>.exh`, `proj16=det.hash_parts("buc.tch04.proj.v1", project_id).hex()[:16]`.
  - `put(parts, x0_hat, prov, *, project_id=None, project_key=None, stats=None) -> PutResult`: latent `(h,w)==(height//8,width//8)`가 아니면 `ExhaustSpecError`; `stats`는 {str: 유한 수}(아니면 `ExhaustSpecError`). **write-once**: 유효 항목이 있으면 덮어쓰지 않고 `created=False`, (q,scale)이 다르면 `conflict=True`. 깨진 파일은 없는 것으로 보고 다시 쓴다. 같은 폴더 임시 파일(`<key>.exh.tmp.<접미>`) → 원자적 교체, 실패해도 부분·임시 파일이 남지 않는다.
  - `get(parts, prov, *, purpose, project_id=None, project_key=None) -> ExhaustEntry | None`: 순서 = purpose 검증(모르면 `ExhaustSpecError`) → `purpose ∉ allowed_purposes(prov)`면 **파일을 열기 전에** `FirewallError`(`denied`+1) → 키 계산 → 읽기. `None`마다 `misses`+1, 깨짐(매직·길이·체크섬·헤더 key/prov 불일치·봉인 열기 실패)은 추가로 `corrupt`+1, 히트는 `hits`+1. ext는 `clock()-created > ext_ttl_s`이면 파일 삭제 후 `None`(`expired`+1; 정확히 TTL은 유효).
  - `get_or_compute(parts, prov, compute, *, purpose="resume", ...) -> ExhaustEntry`: 히트면 `compute` 0회, 미스면 1회 호출(`Tensor` 또는 `(Tensor, stats)`), 저장 후 **저장된 양자화 값**을 돌려준다(미스·히트의 `x0_hat` 비트 동일). 방화벽·자격 검사는 `compute` 호출 전. `compute` 예외·비유한·모양 불일치면 아무것도 저장하지 않는다.
  - `iter_keys(ns, project_id=None) -> list[str]`(정렬; `ns∉NAMESPACES` `ExhaustSpecError`): ext 구역은 `project_id` 없이 `FirewallError`, 있으면 그 프로젝트 것만. `iter_train(branch="main")`(키 순; 모르는 branch `ValueError`): **`train_ok` 구역만** 열고, 헤더 prov가 `bits==0 ∧ train_ok(prov, branch)`가 아니거나 봉인 파일이면 건너뛴다(위조 대비). ext·comp·hold 구역은 열지 않는다.
  - `expire() -> int`(TTL 지난 ext 삭제; sealer 불필요), `purge_project(project_id) -> int`(그 프로젝트 ext 항목 삭제 수; 빈 id `ExhaustSpecError`), `recover() -> int`(남은 `*.exh.tmp*` 삭제).
- 파일 배치: 평문 `b"BUCX1"+u32le(헤더 길이)+헤더 JSON(정렬 키: v, key, prov=to_dict(), shape, stats, created, quant)+int8 q(C순서)+float32le scale+blake3(앞 전체)`; ext는 `b"BUCS1"+f64le(created)+sealer.seal(평문, aad=key.encode()+f64le(created), project_id)`. 프로젝트 키·원본 `project_id`는 디스크에 쓰지 않는다.

## 불변식
1. 키는 입력 필드 하나만 달라도 달라지고 프로세스·인스턴스와 무관하게 같다. (ext, comp, hold, ns)가 다르면 다르다. ext 키는 project_id·project_key에 종속된다.
2. ext는 sealer 없이 디스크에 못 가고 디스크에 q 평문이 없다. ext·comp·hold는 학습·공유 불가, ext는 열거도 프로젝트 안에서만.
3. 저장 오차 ≤ scale/2, 크기 ≈ fp32의 1/4. 읽기는 손상을 값으로 돌려주지 않는다. 쓰기는 원자적·write-once.

## 수용 기준
`tests/tch_04/` 전부 통과(CPU, 가짜 sealer, <60 s): 양자화 오라클·오차 상한, 키 골든(`hash_parts`·keyed blake3 재계산)·필드 민감도·비트 분리·ext 자격, 저장·왕복·파일 배치·손상 8종·원자성·write-once·동시 쓰기, 방화벽(용도 표 600건, sealer 평문 부재, iter_train 격리·위조 무시, 프로젝트 열거 금지, TTL·purge), 세션 재개(끊긴 8-step 디노이즈가 끊기지 않은 실행과 비트 동일, 남은 호출 수만 실행, 프로젝트 간 재사용 0), import 안전성. `@pytest.mark.gpu`(미실행): CUDA 텐서 저장이 CPU와 동일.

## 하류 이음매
WLD-15는 `get(..., purpose="delta")`·`get_or_compute`, TCH-03/DAT-05/TCH-10 학습 분할은 `iter_train`만 쓴다. `KeyParts.world`는 z에 영향 주는 모든 입력(x0_hook의 렌더 포함)을 덮는 Merkle 루트여야 한다(호출자 책임). 레이아웃·`KEY_TAG` 변경은 호환 파괴이므로 `v1`→`v2`와 이 문서를 함께 올린다.
