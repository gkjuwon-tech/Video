# RUN-05 GPU 서버 부트스트랩 v0

**목적.** 새로 빌린 Vast RTX 5090(sm_120, torch 2.9.1+cu128 이미지)에서 **한 번만** 무거운 준비를 한다: GPU·torch 진단 → 파이썬 의존성 → HF 가중치 prefetch → gsplat 설치·JIT 캐시 빌드. 재실행은 캐시를 보고 건너뛴다(멱등). Blackwell 지뢰(torch가 sm_120을 모름, CUDA<12.8, pip가 cu128 torch를 갈아엎음, JIT 무한 대기, PyPI 실패 시 소스 빌드)를 기계적으로 막는다.
**소유.** `ops/gpu/bootstrap.sh`(bash, `set -u`, `set -x` 금지), 테스트 `tests/run_05/`(동결). 의존: RUN-01(`rent(onstart=…)`이 이 스크립트를 부른다).

## 호출
`bash ops/gpu/bootstrap.sh` (인자 없음). 모든 설정은 환경 변수:
| 변수 | 기본 | 뜻 |
|---|---|---|
| `BUC_ROOT` | `/workspace/buc` | 캐시·결과 루트 |
| `BUC_PYTHON` | `python` | GPU 작업용 인터프리터(테스트는 스텁 주입) |
| `HF_HOME` | `$BUC_ROOT/hf` | 미리 설정돼 있으면 유지 |
| `TORCH_EXTENSIONS_DIR` | `$BUC_ROOT/torch_extensions` | gsplat JIT 캐시. 미리 설정돼 있으면 유지 |
| `TORCH_CUDA_ARCH_LIST` | 진단 cap에서 `"M.m"`(5090 → `12.0`) | 미리 설정돼 있으면 유지 |
| `MAX_JOBS` | `min(nproc, 16)` | 미리 설정돼 있으면 유지 |
| `BUC_MODELS` | `Tongyi-MAI/Z-Image-Turbo` | 공백 구분 HF repo id 목록 |
| `BUC_PIP_PKGS` | `diffusers transformers accelerate safetensors huggingface_hub hf_transfer sentencepiece ninja jaxtyping rich` | |
| `BUC_GSPLAT_VERSION` | `1.5.3` | |
| `BUC_RETRIES` / `BUC_RETRY_SLEEP` | `3` / `15` | 모델당 prefetch 시도 수 / 시도 사이 초 |
| `BUC_GSPLAT_TIMEOUT` | `2400` | gsplat 단계 호출당 초(`timeout`) |
| `BUC_SKIP_DEPS`/`BUC_SKIP_PREFETCH`/`BUC_SKIP_GSPLAT`=`1` | — | 그 단계 `skip` |
| `BUC_FORCE=1` | — | 모든 스탬프 무시 |

## 파이썬 이음매 (스텁 가능해야 한다)
GPU·네트워크 작업은 **전부** `$BUC_PYTHON` 호출로만 하고, 호출마다 환경 `BOOT_STEP`을 단다. 스크립트는 `curl`·`wget`·`git`을 직접 부르지 않는다. 부기(JSON 쓰기·파싱)는 PATH의 `python3`(표준 라이브러리만, torch import 금지)를 써도 된다. `nvidia-smi`는 정보용·선택(없거나 실패해도 무시).
- `probe`: stdout에 `PROBE {json}` 한 줄. 키 `cuda_available:bool, torch:str, cuda:str|null, gpu:str, cap:[int,int], arch_list:[str], bf16:bool`. 판정(실패 시 env `reason`): rc≠0 또는 줄 없음/깨짐 → `probe_failed`; `cuda_available` 거짓 → `no_cuda`; `cap[0]≥12`이고 cuda < 12.8(숫자 비교) → `cuda_too_old_for_sm120`; `f"sm_{M}{m}"`가 `arch_list`에 없음 → `torch_lacks_sm_{M}{m}`.
- `deps`: `$BUC_PYTHON -m pip install … -c <constraints> <BUC_PIP_PKGS>`.
- `prefetch`: 모델마다 환경 `BOOT_MODEL=<repo>`, `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER=1`. rc 0 = 성공. 토큰은 파이썬이 `HF_TOKEN` 환경에서 직접 읽는다(argv 금지).
- `gsplat_install`: 1차 `-m pip install … gsplat==<VER>`(source `pypi`), 1차 install 또는 jit 실패 시 2차 `-m pip install --no-build-isolation … git+https://github.com/nerfstudio-project/gsplat@v<VER>`(source `git`). install이 성공했을 때만 그 뒤 `gsplat_jit`(작은 래스터 1회로 JIT 컴파일·수치 확인, rc 0 = 성공). 둘 다 실패면 30. 두 단계 모두 `TORCH_CUDA_ARCH_LIST`, `TORCH_EXTENSIONS_DIR`, `MAX_JOBS` 환경과 `timeout $BUC_GSPLAT_TIMEOUT`.
- **torch 보호:** 모든 pip install은 `-c $BUC_ROOT/constraints.txt`를 달고, 그 파일에 `torch==<probe torch>`가 있다. argv에 `torch`/`torchvision`/`torchaudio` 요구사항 토큰 금지.

## 순서·코드
probe → deps → prefetch(모델 순서대로) → gsplat. 실패 코드: env **10**(이후 호출 0, 나머지 `skip`), deps **11**(이후 `skip`), prefetch **20**(gsplat은 계속), gsplat **30**. 종료 코드 = 실패한 첫 단계 코드, 없으면 0. 잠금 `$BUC_ROOT/bootstrap.lock`(`flock -n`)을 다른 프로세스가 쥐고 있으면 `BOOT_LOCKED` 출력, **75**로 즉시 종료(파이썬 호출 0).
prefetch는 모델당 최대 `BUC_RETRIES`회, 실패 사이 `BUC_RETRY_SLEEP`초.

## 캐시(스탬프, `$BUC_ROOT/stamps/`)
probe는 매번. deps 키 = (pkgs, torch). prefetch 키 = repo. gsplat 키 = (VER, cap, torch, cuda) **그리고** `TORCH_EXTENSIONS_DIR`가 비어 있지 않을 것. 성공 시에만 스탬프. 적중이면 호출 0, 상태 `cached`.

## 산출
- `$BUC_ROOT/bootstrap.json`(성공·실패 모두, 원자적 교체): `{"v":1,"ok":bool,"code":int,"ts_start","ts_end","hf_token":"yes"|"no", "paths":{"root","hf_home","torch_extensions_dir"}, "env":{"status":"ok"|"fail","reason":str|null,"torch","cuda","gpu","cap":[M,m],"arch"}, "deps":{"status"}, "prefetch":{"status","models":[{"repo","status":"ok"|"cached"|"fail","attempts":int}]}, "gsplat":{"status","version","source":"pypi"|"git"|null,"attempts":["pypi",…]}}`. 단계 status ∈ `ok|cached|fail|skip`(prefetch 전체: 모델 중 fail 있으면 fail, 모두 cached면 cached, 아니면 ok). `ok == (code==0)`.
- `$BUC_ROOT/env.sh`(env ok일 때): `export HF_HOME=… TORCH_EXTENSIONS_DIR=… TORCH_CUDA_ARCH_LIST=… HF_HUB_ENABLE_HF_TRANSFER=1` 줄. 비밀 없음.
- stdout 마커 `BOOT_START`, `BOOT_ENV … arch=<arch> …`, `BOOT_RESULT ok=<0|1> code=<n> …`, 각 `BOOT_*` 줄 ≤400자(vastai logs 절단). 전체 출력은 `$BUC_ROOT/logs/bootstrap.log`에도.

## 불변식
HF_TOKEN 값은 stdout·stderr·json·env.sh·로그·argv 어디에도 없다 · cu128 torch를 pip가 바꿀 수 없다 · 재실행은 probe 외 호출 0 · 실패해도 json은 남는다.

## 수용 기준
`tests/run_05/` 전부 통과(CPU, 스텁 `$BUC_PYTHON`, <60 s): 문법, 정상 경로 호출 순서·환경, 진단 4종 거절, 단계별 종료 코드·skip, prefetch 재시도, gsplat pypi→git 대체·timeout, 스탬프 멱등·무효화(torch 변경, 캐시 비움, FORCE), torch 제약, 잠금 75, 비밀 위생, env.sh. `@pytest.mark.gpu`: 실제 5090에서 `BUC_SKIP_PREFETCH=1`로 gsplat ok/cached.

## 하류 이음매
RUN-01 `rent(onstart=…)`·RUN-02 러너가 잡 전에 실행하고 `code`를 본다. TCH-01·RDR-01·EVL-06·RUN-06은 `source $BUC_ROOT/env.sh` 후 `HF_HOME` 캐시에서 `local_files_only`로 로드, gsplat은 `TORCH_EXTENSIONS_DIR` 캐시로 재컴파일 없이 import.
