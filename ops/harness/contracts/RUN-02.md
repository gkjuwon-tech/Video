# RUN-02 pull 러너 v0

**목적.** GPU 서버는 SSH가 막혀 있다. 그래서 서버가 GitHub 브랜치에서 잡 명세(`jobs/*.yaml`)를 **당겨** 실행하고, 상태·지표·작은 산출물을 같은 브랜치에 **푸시**한다(큰 산출물은 RUN-03 허브). 여러 러너·재시작·푸시 경쟁이 있어도 **같은 잡의 같은 시도를 두 번 실행·계상하지 않고**, 죽은 뒤에는 이어서 돈다. 1차 범위: `kind` = `train`, `render` 두 종.
**소유.** `buc/rt/runner.py`, `jobs/*.yaml`(최소 1개, 예: `jobs/example-smoke.yaml`, 실서버가 바로 집지 않게 `enabled: false` 권장), 테스트 `tests/run_02/`(동결). 의존: 표준 라이브러리(`subprocess`, `fcntl`, `hashlib`, `json`), `buc.core.ledger`(선택 주입), RUN-03 `HubSync`(선택 주입). **import 시 torch·numpy·yaml 금지**(PyYAML은 `parse_job_yaml` 안에서 지연 import), 부작용 없음.
**소유 밖 요청.** pyproject 의존성 `pyyaml>=6`.

## 상수·예외
`JOB_SCHEMA="buc.job/v1"`, `STATE_SCHEMA="buc.job_state/v1"`, `RESULT_SCHEMA="buc.job_result/v1"`, `KINDS=("train","render")`, `JOBS_DIR="jobs"`, `RUNS_DIR="runs/jobs"`.
`RunnerError(RuntimeError)` ⊃ `JobSpecError`(`ValueError`도 상속), `PushConflict`, `LockHeld`.

## 잡 명세 `JobSpec` (frozen dataclass, kw_only)
`id:str`(전체 일치 `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`), `kind`∈KINDS, `cmd:tuple[str,...]`(비지 않은 str 리스트만; 문자열 하나는 오류), `env:dict[str,str]`(키 `[A-Z_][A-Z0-9_]*`, 값 str; 키에 `TOKEN|SECRET|PASSWORD|KEY`(대소문자 무관) 포함 금지 = 비밀은 서버 환경에서만), `timeout_s:float=3600.0`(>0, 유한, int→float), `max_attempts:int=3`(≥1), `outputs:tuple[str,...]=()`(out 디렉터리 기준 상대 glob; 절대경로·`..` 금지), `gpu:bool=True`, `priority:int=0`, `enabled:bool=True`. bool을 숫자로 받지 않는다. 위반·모르는 키는 `JobSpecError`. YAML의 `schema` 키는 있으면 `JOB_SCHEMA`여야 한다.
- `from_dict(d)`, `to_dict()`(위 필드 전부, tuple→list, `schema` 제외), `parse_job_yaml(text, *, stem=None) -> JobSpec`(`stem`이 주어지면 `id`와 같아야 함).
- `spec_hash -> str` = `sha256(json.dumps(h, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()).hexdigest()`, `h` = `to_dict()`에서 **`id, kind, cmd, env, timeout_s, outputs, gpu`만**(실행에 영향 없는 `max_attempts, priority, enabled`는 제외).

## 원격 이음매 (`Protocol JobRemote`, 경로는 저장소 상대 `/` 구분)
`sync() -> None`(원격 최신으로 맞춤) · `read(path) -> bytes|None` · `list(prefix) -> list[str]`(그 아래 파일 경로) · `publish(files: Mapping[str, bytes], message: str) -> str`: 현재 HEAD 위에 파일을 쓰고 커밋·푸시. 원격이 앞서 있으면(non-fast-forward) **로컬을 원격으로 되돌린 뒤** `PushConflict`(그 파일들은 반영 안 됨). 바뀐 게 없으면 커밋 없이 HEAD 반환.
`GitRemote(repo_dir, *, branch, remote="origin", git="git")`: git CLI 어댑터(`fetch`+`reset --hard <remote>/<branch>`, `ls-files`, 커밋은 `-c user.name/user.email` 지정, `push <remote> HEAD:<branch>`). 오류 메시지에서 URL 자격 증명(`//user:pass@`)은 가린다.

## 저장소 배치 (계약)
`jobs/<id>.yaml` 명세 · `runs/jobs/<id>/state.json` = `{"schema":STATE_SCHEMA,"job_id","spec_hash","status":"running"|"retry"|"done"|"failed","attempt":int,"runner_id","lease_until":float|null,"updated_at":float,"error":str|null}` · `runs/jobs/<id>/result.json` = `{"schema":RESULT_SCHEMA,"job_id","spec_hash","status","attempt","runner_id","exit_code":int|null,"timed_out":bool,"wall_s":float,"gpu_h":float,"metrics":dict|null,"outputs":[{"path","size","sha256","pushed":bool,"remote":str|null}],"log_tail":str}` · 작은 산출물 `runs/jobs/<id>/out/<상대경로>`. JSON은 `sort_keys`·UTF-8.

## 실행기 이음매
`ExecResult(exit_code:int|None, timed_out:bool, wall_s:float, log_tail:str)` frozen. `Protocol Executor.run(cmd, *, cwd, env, timeout_s, heartbeat_s, on_heartbeat) -> ExecResult`. `SubprocessExecutor()`: 새 프로세스 그룹, stdout+stderr 합침, `heartbeat_s`마다 `on_heartbeat()`, 기한 넘으면 그룹 kill → `timed_out=True`. `log_tail`은 마지막 ≤200줄·≤16KB.

## `Runner(remote, *, runner_id, work_root, executor=None, ledger=None, hub=None, clock=time.time, environ=None, lease_s=900.0, heartbeat_s=60.0, max_push_bytes=5_000_000, usd_per_h=0.0, log=None)`
- **잡 디렉터리** `work_root/<id>/<spec_hash[:12]>/`(시도 사이에 지우지 않음 = 재개 재료), 산출 `out/`, 로컬 저널 `attempts/<n>.json`(실행 종료 직후 원자적 기록: exit_code·timed_out·wall_s·log_tail).
- **잡 환경** = `environ`(기본 `os.environ`) + `spec.env` + `BUC_JOB_ID, BUC_JOB_KIND, BUC_ATTEMPT, BUC_RESUME("1" if attempt>1 else "0"), BUC_JOB_DIR, BUC_OUT_DIR, BUC_METRICS(=out/metrics.json)`. cwd = 잡 디렉터리.
- `pending() -> list[JobSpec]`: `sync()` 후 `jobs/*.yaml`(파싱 실패·stem 불일치는 건너뛰고 로그) 중 `enabled`이고 다음 중 하나: 상태 없음 · 상태의 `spec_hash` ≠ 현재(새 판, attempt 1부터) · `retry` · `running`이면서 `lease_until ≤ clock()` 또는 `runner_id`가 자기 자신. 순서 `(-priority, id)`. `done`/`failed`(같은 해시)는 제외.
- `claim(spec) -> int|None`: 다음 시도 번호 n(상태 없거나 해시 다르면 1, `retry`·만료 `running`은 attempt+1, **자기 `running`이면서 로컬 저널 `attempts/<attempt>.json`이 있으면 같은 n**)으로 `running` 상태를 publish. `PushConflict`면 sync 후 다시 판단(최대 3회); 남이 살아 있는 리스로 가졌으면 `None`. n>`max_attempts`면 실행 없이 `failed`(`error="attempts_exhausted"`)를 publish하고 `None`.
- `run_job(spec, attempt) -> JobResult`: 저널이 있으면 **재실행하지 않고** 저널로 마무리, 없으면 실행→저널. 실행 중 하트비트마다 리스 갱신 publish(실패는 로그만). 마무리: 지표(`BUC_METRICS`가 JSON 객체면 dict, 아니면 None), `outputs` glob 매칭 파일 sha256·크기, 개별·누적 ≤`max_push_bytes`는 저장소로, 넘으면 `hub.upload_file(local, f"jobs/{id}/{spec_hash[:12]}/{rel}")`(hub 없거나 실패면 `pushed=False, remote=None`, 치명 아님). 장부가 있으면 `gpu_h`(gpu 잡만 `wall_s/3600`, 아니면 0)와 `usd`(`usd_per_h·wall_s/3600`)를 `run_id=f"job:{id}"`, `stage=kind`, `idem_key=f"job:{id}:{spec_hash[:16]}:{n}:<kind>"`로 기록. 상태: exit 0·미타임아웃 → `done`, 아니면 n<max면 `retry`, 아니면 `failed`. state+result+작은 산출물을 **한 커밋**으로 publish(충돌 시 sync 후 재시도 ≤3회). 그 전에 원격 상태의 (`runner_id`, `attempt`)가 자기 것이 아니면 아무것도 쓰지 않고 `status="lost"`.
- `JobResult`(frozen): `job_id, spec_hash, status("done"|"retry"|"failed"|"lost"), attempt, exit_code, timed_out, wall_s, metrics, outputs`.
- `run_once() -> JobResult|None`: `work_root/runner.lock`에 `fcntl.flock(LOCK_EX|LOCK_NB)`(잡혀 있으면 `LockHeld`) → `pending` → 첫 잡부터 `claim` 성공한 하나만 `run_job`. 없으면 `None`. `serve(*, max_jobs=None, poll_s=300.0, sleep=time.sleep, stop=None) -> int`(실행한 잡 수). `last_activity: float|None`(claim·하트비트·마무리 시각, RUN-01 `activity` 공급용).
- **비밀 위생.** `environ`에서 이름에 TOKEN/SECRET/PASSWORD/KEY가 든 값(길이 ≥4)은 `log_tail`·`error`·커밋 내용에서 `***`로 가린다. 환경 값은 result·state에 쓰지 않는다.
- `main(argv=None) -> int`: `--repo --branch --runner-id --work-root [--once] [--max-jobs N] [--lease-s] [--heartbeat-s]`. `--once`는 `run_once` 한 번, 잡 없으면 0.

## 불변식
같은 (id, spec_hash, attempt)는 최대 1회 실행·1회 계상 · `done`은 다시 돌지 않음(명세 해시가 바뀌면 새 판) · 살아 있는 남의 리스는 빼앗지 않음 · 푸시 경쟁에서 진 쪽은 실행하지 않음 · 잡 디렉터리는 시도 사이 보존 · 비밀은 저장소에 안 올라감.

## 수용 기준
`tests/run_02/` 전부 통과(CPU, <60 s; 가짜 원격·로컬 bare git 저장소·실제 짧은 서브프로세스): 명세 검증·해시 오라클, pending 규칙·순서, 경쟁·리스 만료·자기 재개, 저널 기반 무재실행 재개, 재시도·소진, 타임아웃, 지표·산출(작은 것 푸시/큰 것 허브), 장부 멱등, 비밀 가림, 로컬 락, git 끝-끝(두 클론 경쟁 포함), 저장소 `jobs/*.yaml` 유효성, import 순수성.

## 하류 이음매
RUN-01 `activity=lambda iid: runner.last_activity`. RUN-03 `HubSync.from_env()`를 `hub`로. RUN-04 큐는 `jobs/*.yaml`을 쓰는 쪽(priority로 순서). RUN-07은 `runs/jobs/*/result.json`과 장부 `run_id LIKE 'job:%'`를 읽는다. 상태·결과 형식 변경은 `*_SCHEMA` 주 버전을 올리고 이 계약서를 함께 고친다.
