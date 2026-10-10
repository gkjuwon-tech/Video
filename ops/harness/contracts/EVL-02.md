# EVL-02 일일 A/B 러너 v0 — 24장 고정 무브 → 전 구간 → mp4 → 전일 대비 시트 (야간 잡)

**목적.** 반복 횟수가 최대 레버다(예측 패널: 전 구간 반복 20–40회 → 50–80회). 이 단위는 밤마다 EVL-01의 24장을 **같은 크롭 파일·같은 안전 무브·같은 시드 규칙**으로 파이프라인 끝까지 돌려 mp4로 만들고, 전일 결과와 한 장의 시트로 비교하는 **장부 + 인코딩 + 시트** 층이다. 렌더 자체는 주입된 `render(job)`가 한다(스파인 연결은 `spine_renderer`). 한 장이 터져도 밤 전체가 죽지 않고, 끊겨도 이어서 돈다.
**소유.** `eval/daily_ab.py`, 테스트 `tests/evl_02/`(동결). 의존: PRD-01(`buc.pipeline.spine`, 호출만), EVL-01(`tools/inputs_qa.py`를 **파일 경로로 지연 적재**, `tools/`는 패키지가 아님), RDR-06(`buc.render.video`), FND-02·03, numpy, Pillow, 표준 라이브러리. `eval/`은 패키지가 아니다(테스트는 파일 경로 적재, 레포 루트를 `sys.path`에 넣는다). import 부작용 없음(파일·디렉터리 생성 0), `lpips`·`cv2`·`torchvision` 적재 금지(R-EVAL: 평가 모델은 EVL-04 워커 프로세스). 소스에 토큰·`hf_` 문자열 금지. `buc/`가 `eval`을 import하는 것은 GOV-05 몫.

## 공개 API (`eval/daily_ab.py`)
상수: `SCHEMA_VERSION=1`, `DEFAULT_MOVE="I1"`(🟢 푸시인), `SEED_TAG="buc.daily_ab.v1"`, `RECORD_NAME="run.json"`, `SHEET_NAME="sheet.png"`, `PSNR_CAP_DB=99.0`, `NO_PREV_RGB`, `FAILED_RGB`, `BG_RGB`, `STATUS_RGB{new,same,changed,failed,regressed}`(서로 다른 5색), `PAD=4`, `HEADER_H=28`, `LABEL_H=16`.
```python
class DailyABError(RuntimeError)
@dataclass(frozen=True) Job(item, move, seed, crop_box)      # .id .slot
@dataclass(frozen=True) Rendered(frames: Frames, root: str|None = None)
Render = Callable[[Job], Frames | Rendered]
def job_seed(base_seed, item_id, move) -> int                 # det.derive_seed(base_seed, SEED_TAG, item_id, move)
def plan_jobs(m, *, move="I1", base_seed=0, partial=False) -> list[Job]
def check_sha256(item, data: bytes) -> None                   # 금고 바이트 대조(EVL-01). 불일치 DailyABError(항목 id만, 바이트 노출 없음)
def spine_renderer(spine, load_image, *, params=None, until="S9-10", move_stage="S8") -> Render
def load_inputs(path=None, *, partial=False) -> Manifest      # EVL-01 load_manifest(require_ready=not partial)
def read_record(day_dir) -> dict; def find_previous(out_root, date) -> PrevDay|None   # PrevDay(date, dir, record)
def canonical_digest(record) -> str                            # 키 이름이 "wall_s"인 값(어디에 있든)을 뺀 정렬 JSON의 SHA-256
def clip_delta(prev_mp4, cur_mp4) -> {"comparable","mad","psnr_db"}
def sheet_geometry(n_items, thumb_size=(160,90), cols=4) -> SheetGeometry
def run_daily(manifest, render, out_root, date, *, move="I1", base_seed=0, partial=False, tag="", encoder="cpu",
              crf=18, resume=True, thumb_size=(160,90), cols=4) -> DailyResult
def main(argv=None, *, render=None) -> int
```
- **계획.** `plan_jobs`: slot 순 정렬(입력 순서 무관). `move ∉ m.moves`(None 포함) `ValueError`. `partial=False`인데 `status != ready`가 있으면 `DailyABError`(메시지에 planned id들). `partial=True`는 ready만, 하나도 없으면 `DailyABError`. `Job.crop_box = item.crop_box`(같은 크롭 파일, EVL-01), `seed = job_seed(...)`.
- **`spine_renderer`.** `image = load_image(job)`(크롭이 적용된 `InputImage`를 돌려줘야 함) 후 `spine.run(image, seed=job.seed, params=P, until=until)`. `P` = 사용자 `params` 깊은 복사 + `P[move_stage] ∪= {"move": job.move, "crop_box": list(job.crop_box)}`(입력 params 불변). `force`를 주지 않는다(PRD-01 캐시·부분 재실행을 그대로 쓴다). 결과 `artifacts["frames"]`가 없으면 `DailyABError`. 반환 `Rendered(frames, result.root)`. **S8 구현은 `ctx.params["move"]`(벤치 의도 I1..I5)와 `["crop_box"]`를 읽어야 한다(CAM-02 `bench_spec`).**
- **`run_daily(date)`.** `date`는 `YYYY-MM-DD`(달력상 유효, 아니면 `ValueError`; 시계를 읽지 않는다). 인자(thumb·cols·move)와 계획 오류는 **디스크를 건드리기 전에** 발생. 산출은 `out_root/date/`만: `<id>_<move>.mp4`(항목당 하나, `encode_frames(encoder, crf)`), `run.json`, `sheet.png`. 그 밖의 파일·임시·`.part` 잔여 없음. 전일(`find_previous`)은 읽기 전용.
- **격리.** `render`가 `Exception`을 내거나 `Frames/Rendered`가 아닌 것을 돌려주거나 인코딩·프레임 수 검사가 실패하면 그 항목만 `status:"failed"`(`error = "<타입>: <메시지>"` ≤300자), mp4는 지운다(**mp4 존재 ⇔ status ok**). 다음 항목은 계속 돈다. `KeyboardInterrupt` 등은 그대로 전파. 작업 하나가 끝날 때마다 `run.json`을 원자적으로(임시+`os.replace`) `complete:false`로 갱신, 끝에서 `true`.
- **재개(`resume=True`).** 같은 날 `run.json`이 있고 `move`·`tag`가 같으며, 항목이 `ok`·`seed` 같음·mp4 존재·`mp4_sha256` 일치일 때만 재사용(`DailyResult.reused`에 id 튜플, 장부에는 표시 없음 → 재개 결과와 처음부터 돈 결과의 `canonical_digest`가 같다). 아니면 다시 렌더. `resume=False`는 전부 다시. `tag`는 코드 식별자(커밋 해시 등, 기본 "").
- **전일 대비.** 기준선 = 날짜가 **엄격히 앞서고** `read_record`가 통과하며 `complete:true`인 가장 최근 날(깨진 JSON·스키마·디렉터리명≠date·끊긴 날은 건너뜀). 항목별 `delta{status,mad,psnr_db,prev_date}`: 오늘 ok + 전일 ok(mp4 존재)면 `frames_hash` 같으면 `same`(mad 0.0, psnr `PSNR_CAP_DB`), 다르면 `changed`(`clip_delta`: 디코드한 두 클립의 평균 절대차 0–255 단위·PSNR, 소수 6자리 반올림; 프레임 수·크기가 다르면 `comparable:false`, mad/psnr `None`); 오늘 ok + 전일 없음/실패/파일 없음 `new`; 오늘 실패 + 전일 ok `regressed`; 오늘 실패 + 전일 없음/실패 `failed`. `prev_date`는 same/changed/regressed에만.
- **`run.json`** 최상위: `schema, date, move, base_seed, tag, complete, prev_date, vault_repo, disclosure`(= `manifest.selection["disclosure"]`, 선별 공개 문구), `matrix`(= EVL-01 `coverage(m)["matrix"]`), `n_planned_skipped, items, summary, wall_s`. `summary = {n_jobs, ok, failed, deltas{new,same,changed,failed,regressed}}`. 항목: `id, slot, seed, status, mp4(상대명|null), n_frames, fps, width, height, frames_hash(Frames.content_hash), mp4_sha256, root(스파인 root|null), error, wall_s, delta`. 실패 항목의 수치 필드는 `null`. 정렬 키 JSON, 비밀 패턴 없음. `read_record` 통과 조건: JSON dict, `schema==1`, `complete` bool, `items` list, `date==디렉터리명`(아니면 `DailyABError`).
- **시트(`sheet.png`, RGB, 메타데이터 없음).** `sheet_geometry`: `cols_eff=min(cols,n)`, `rows=ceil(n/cols_eff)`, 칸 `cell_w=3·PAD+2·tw`, `cell_h=2·PAD+th+LABEL_H`, 크기 `(cols_eff·cell_w, HEADER_H+rows·cell_h)`; 항목 i(0 기준)는 행 `i//cols_eff`, 열 `i%cols_eff`, 칸 원점 `(열·cell_w, HEADER_H+행·cell_h)`; `thumb_box(i,"prev")=(x+PAD, y+PAD, …+tw, …+th)`, `"cur"`은 x가 `x+2·PAD+tw`부터, `label_box(i)=(x+PAD, y+PAD+th, x+cell_w−PAD, y+PAD+th+LABEL_H)`. 범위 밖 i `IndexError`, 잘못된 인자 `ValueError`. 배경 `BG_RGB`. **왼쪽 = 전일, 오른쪽 = 오늘**, 썸네일 = 클립의 **마지막 프레임**(크기가 같으면 디코드한 그대로, 다르면 이중선형 축소). 전일 없음 → prev 칸 `NO_PREV_RGB` 단색, 오늘 실패 → cur 칸 `FAILED_RGB` 단색. 라벨 띠는 `STATUS_RGB[delta.status]`로 채우고 글자는 (+2,+2) 이후에 그린다(띠의 좌상단 픽셀 = 상태색). 헤더에 날짜·전일·tag·disclosure 글자. 24장 전체(planned 건너뜀 제외)가 한 장.
- **`main`.** `--out DIR --date D [--manifest P] [--partial] [--move I1] [--seed 0] [--tag ""] [--renderer MOD:ATTR]`(`MOD.ATTR()`가 `Render`를 돌려주는 공장). 반환: 0 전부 ok, 1 실패 항목 있음, 2 인자·날짜·매니페스트·렌더러 해석 오류(argparse 종료도 2, 이때 디스크 무접촉). 한 줄 요약 `"<date> ok=<n> failed=<n>"`. `DailyResult(record, day_dir, record_path, sheet_path, reused)`, `.ok = (failed == 0)`.

## 불변식
1. 같은 (매니페스트, 렌더 출력, 설정) → `canonical_digest`·시트 픽셀 동일, CPU libx264 mp4 바이트 동일. 시계는 `wall_s`에만.
2. 한 항목의 실패·예외는 다른 항목에 영향이 없다. 끊긴 밤은 `resume`으로 남은 것만 돈다.
3. 전일 디렉터리는 절대 바뀌지 않는다. 24장 모드(`partial=False`)는 planned가 하나라도 있으면 아무것도 쓰기 전에 실패한다.
4. 수치 해석은 `public_ok` 항목만 공개(EVL-01); 이 단위는 장부에 `disclosure`·`matrix`를 항상 붙인다.

## 수용 기준
`tests/evl_02/` 전부 통과(CPU, <60 s, 약 35 s, `-m "not gpu"`): 시드 공식, 계획·ready 규칙, 전일 탐색(깨진/끊긴/미래/잘못된 이름 건너뜀), 시트 기하 정확값(560×460 등)과 칸 겹침 없음, 클립 델타 독립 오라클 대조, 이틀 시나리오(same/changed/regressed, 시트 칸 픽셀 = 디코드한 프레임, 라벨 색), 첫날 실패 격리, 오래된 mp4 제거, 중단→재개 결과 동일, 재사용 무효화(tag·seed·훼손), 24장 야간 실행, 썸네일 축소, 입력 검증의 디스크 무접촉, 실제 매니페스트 이음매(disclosure·matrix), CLI 종료 코드, 격리(평가 모델 미적재). `@pytest.mark.gpu`: `encoder="gpu"` NVENC 경로.

## 하류 이음매
PRD-06(비교 시트·Seedance A/B)과 EVL-03은 `read_record`·`find_previous`·`run.json`의 `items[*].mp4/frames_hash/delta`를 읽는다. EVL-07·09(KPI·시간별 보고)는 같은 날 폴더의 mp4를 입력으로 EVL-04를 부른다. RUN-07은 `main(--renderer …)`를 야간 큐에 건다(종료 코드 1 = 아침 빨간불). 스파인 실물 연결: S8 구현이 `move`/`crop_box` params를 읽고, `load_image`는 RUN-03 금고 다운로드 + `check_sha256` + 크롭을 한다. 장부 스키마·시드 공식 변경은 호환 파괴이므로 `SCHEMA_VERSION`/`SEED_TAG`와 이 계약서를 함께 올린다.
**소유 밖 요청.** 없음(Pillow·PyYAML는 기존 의존성). 실제 24장은 창업자 촬영(D1–4) 전까지 `--partial`(ready만)로 돈다.
