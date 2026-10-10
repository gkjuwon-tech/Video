# RUN-03 체크포인트·HF Hub 비공개 저장소 동기 v0

**목적.** GPU 서버(SSH 불가)와 이 컨테이너 사이의 체크포인트·산출물 이동. 비공개 저장소 한 곳이 금고이고, **반쯤 올라간 파일이 완성본으로 보이지 않게** 한다. 바이트 I/O만 한다(가중치를 해석하지 않는다).
**소유.** `buc/rt/hf_sync.py`, 테스트 `tests/run_03/`(동결). 표준 라이브러리만 쓴다. `huggingface_hub`는 `HfHubClient` 생성 때만 **지연 import**(import 시 torch·numpy·huggingface_hub 금지, CPU에서 import 가능).
**소유 밖 요청.** `buc/rt/__init__.py`(없어도 동작).

## 상수
`DEFAULT_REPO_ID="sdsdsdsdsdsdsd123/buc-artifacts"`, `DEFAULT_REPO_TYPE="model"`, `TOKEN_ENV="HF_TOKEN"`, `DEFAULT_PREFIX="ckpt"`, `SIDECAR_SUFFIX=".sha256.json"`.

## 예외 (전부 `HubSyncError(RuntimeError)`의 하위)
`HubAuthError` · `HubNotFoundError`(`FileNotFoundError`도 상속) · `TransientHubError` · `IntegrityError` · `ConflictError` · `RepoNotPrivateError`.

## 클라이언트 이음매 (`Protocol HubClient`, 4개)
`list_files(prefix: str) -> list[str]`(접두어로 거르지 않아도 된다, HubSync가 `prefix + "/"`로 다시 거른다) · `upload_file(local: Path, remote: str) -> None` · `download_file(remote: str, local: Path) -> None`(없으면 `HubNotFoundError`) · `is_private() -> bool`.
`HfHubClient(*, repo_id, repo_type, token)`: 실물 어댑터. huggingface_hub 예외를 위 예외로 옮기고, 저장소를 만들 때는 항상 `private=True`. 라이브러리가 없으면 생성 시 `HubSyncError`(메시지에 토큰 없음).

## 공개 API
- `resolve_token(env=None, token_path=None) -> str|None`: `env`(기본 `os.environ`)의 `HF_TOKEN`을 먼저, 공백뿐이거나 없으면 `token_path`(기본 `~/.cache/huggingface/token`) 파일. 앞뒤 공백 제거, 비면 `None`.
- `redact(text, secrets) -> str`: 빈 비밀은 무시, 나머지는 `***`로 치환.
- `HubConfig(*, token, repo_id=DEFAULT_REPO_ID, repo_type=DEFAULT_REPO_TYPE)` frozen dataclass, `token`은 `repr=False`.
- `RetryPolicy(max_attempts=4, base_delay=1.0, factor=2.0, max_delay=30.0)` frozen. `max_attempts≥1, base_delay≥0, factor≥1, max_delay≥base_delay` 아니면 `ValueError`. k번째 실패 뒤 대기 `min(max_delay, base_delay·factor^(k-1))`, 지터 없음, 마지막 실패 뒤엔 대기 없이 그 예외를 올린다.
- `SyncResult(ok, uploaded=False, remote_path=None, sha256=None, size=None, skipped=None, error=None)` frozen. `CheckpointRef(run_id, step, remote_path)` NamedTuple.
- 순수 함수: `validate_run_id(s)`(전체 일치 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`) · `validate_remote_path(p)`(빈값, 앞 `/`, `\`, NUL, 빈·`.`·`..` 구간, 512자 초과) 위반은 `ValueError` · `checkpoint_name(step, ext="pt") -> "step_%09d.<ext>"`(step은 `int`≥0: bool·비정수 `TypeError`, 음수 `ValueError`; ext `[A-Za-z0-9]+`) · `parse_checkpoint_name(name) -> (step, ext)|None`(`step_`+9자리 이상 숫자+`.`+ext 전체 일치만) · `checkpoint_path(run_id, step, ext="pt", prefix="ckpt") -> "<prefix>/<run_id>/<name>"`.
- `HubSync(client|None, *, prefix="ckpt", retry=RetryPolicy(), sleep=time.sleep, secrets=(), log=None)`; `client=None`은 비활성(토큰 없음): 아래 엄격 메서드는 전부 `HubAuthError("hf_token=no")`.
  - `HubSync.from_env(*, env=None, token_path=None, repo_id=..., repo_type=..., client_factory=None, **kw)`: 토큰 없으면 비활성 인스턴스(팩토리 호출 안 함), 있으면 `client_factory(repo_id=, repo_type=, token=)`(기본 `HfHubClient`)로 만들고 `secrets=(token,)`.
  - `verify_private()`: `is_private()` 거짓이면 `RepoNotPrivateError`. 모든 업로드 전에 부른다(참은 캐시, 거짓은 캐시 안 함).
  - `upload_file(local, remote, *, overwrite=False) -> SyncResult` · `download_file(remote, dest, *, force=False) -> Path`.
  - `upload_checkpoint(local, run_id, step, *, overwrite=False)`: ext는 로컬 확장자(없으면 `ValueError`). `list_checkpoints(run_id) -> list[CheckpointRef]`(step 오름차순, 완성본만, 다운로드 없음) · `latest_checkpoint(run_id)` · `download_checkpoint(run_id, dest_dir, step=None, *, force=False) -> Path`(`dest_dir/<이름>`, 없으면 `HubNotFoundError`) · `fetch_latest(run_id, dest_dir) -> Path|None`(없으면 `None` = 새로 시작).
  - `try_upload_checkpoint(...) -> SyncResult`: **최선 노력**. `HubSyncError`·`OSError`는 WARNING 로그 후 `ok=False, error=<가림 처리된 문자열>`. 비활성이면 `ok=False, skipped="no_token"` + 로그에 `hf_token=no`. 인자 오류(`ValueError`·`TypeError`: run_id·step·임시 파일명)는 그대로 올린다.

## 디스크 배치 (계약)
원격 `<prefix>/<run_id>/step_000000123.pt` + 사이드카 `<같은 경로>.sha256.json`. 사이드카 바이트 = `json.dumps({"v":1,"sha256":<hex64>,"size":<int>}, sort_keys=True, separators=(",",":")).encode()`. 해시 `hashlib.sha256`, 1 MiB 스트리밍.

## 불변식
1. **완성본 = 데이터 + 사이드카.** 데이터 업로드가 성공한 뒤에만 사이드카를 올린다. 한쪽만 있으면 `list/latest`에 안 보인다. 최신은 **숫자** step 기준. 다른 run(`a` vs `ab`)·prefix와 섞이지 않는다.
2. 업로드 멱등: 원격 사이드카가 같은 sha256·크기면 클라이언트 업로드 0회(`uploaded=False, skipped="up_to_date"`). 다르면 `ConflictError`(`overwrite=True`면 덮어씀). 사이드카 없는 원격 데이터는 미완성이라 충돌이 아니라 다시 올린다.
3. 다운로드: 사이드카를 먼저 읽는다(없으면 `HubNotFoundError`, 데이터는 받지 않음). 데이터는 `<dest>.part`로 받아 크기·sha256 검증 뒤 `os.replace`. 실패하면 기존 `dest` 보존, `.part` 등 잔재 없음. 사이드카가 JSON·`v==1`·`sha256` 64 hex·`size` 정수≥0이 아니면 `IntegrityError`. `force=False`이고 로컬이 이미 일치하면 데이터 다운로드 0회.
4. 재시도는 `TransientHubError`·`IntegrityError`만, 클라이언트 호출 단위(다운로드는 받기+검증 단위)로. 인증·없음·충돌·그 외 예외는 1회 만에 그대로 전파.
5. 공개 저장소면 아무것도 올리지 않는다. `*.tmp`·`*.part`는 `ValueError`, 없는 파일은 `FileNotFoundError`(검사는 네트워크 호출 전).
6. 토큰은 로그·`SyncResult`·`repr`·올라가는 예외 메시지(`type(e)(redacted)`로 다시 만든다)·소스에 나타나지 않는다. 소스에 토큰 리터럴·`private=False` 금지. 시계·난수 의존 없음(`sleep` 주입).

## 수용 기준
`tests/run_03/` 전부 통과(CPU, <60 s, 인메모리 가짜 허브): 이름·경로 표, 사이드카 바이트 고정, 업로드 순서·부분 실패, 숫자 최신·고아 무시, 백오프 `[1,2,4]`·상한·즉시 실패, 멱등·충돌·덮어쓰기, 무결성(영구·일시 손상, 절단, 크기, 불량 사이드카)과 원자성, 비공개 검사, 최선 노력·비활성, 토큰 해석·가림, from_env 배선, import 순수성. `@pytest.mark.gpu`: 실제 비공개 저장소 왕복(`HF_TOKEN`과 시험용 `BUC_HF_TEST_REPO` 없으면 skip, 운영 금고에는 쓰지 않음).

## 하류 이음매
RUN-06 `ckpt.py`: 원자 체크포인트를 쓴 뒤 30–60분마다 `try_upload_checkpoint`(실패해도 학습 지속), 시작 때 `fetch_latest`로 재개. RUN-02 러너: `HubSync.from_env()`로 산출물 업로드. 사이드카 형식·원격 이름 변경은 호환 파괴이므로 `"v"`를 올리고 이 계약서를 함께 고친다.
