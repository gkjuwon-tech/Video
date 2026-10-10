# RUN-07 시간별 보고 생성기 v0 (PNG + 한국어 요약)

**목적.** 창업자용 보고 한 장: 하네스 진행, 최근 구간 지출(USD·GPU-h·TFU)·예산 사용률, 단위별 실소요. 못 읽은 수치는 "장부 없음"이라 적고 0으로 지어내지 않는다.
**소유.** `ops/report/hourly.py`, 테스트 `tests/run_07/`(동결). 의존 FND-07(`buc.core.ledger`, `buc.core.config`)만. numpy·torch·matplotlib 미사용(PIL + 표준 라이브러리). import 부작용 없음. 직접 실행 시 레포 루트(`__file__` 기준)를 스스로 `sys.path`에 넣는다. **읽기 전용:** 장부·backlog를 바꾸지 않고 없는 DB 파일을 만들지 않는다.

## 공개 API
```python
SCHEMA_VERSION=1; HOURS=24; WARN_FRAC=0.8; KST_OFFSET_S=32400
CANVAS=(1280,720); CHART_BOX=(80,150,1200,430); PROGRESS_BOX=(80,540,1200,580)   # (x0,y0,x1,y1), x1·y1 배타
COLOR_BG=(15,17,21); COLOR_USD=(122,162,247); COLOR_DONE=(158,206,106); COLOR_TRACK=(42,46,54)
FONT_CANDIDATES: tuple[str,...]       # 한글 글꼴 경로 후보. 없으면 PIL 기본 글꼴(예외 금지)
def redact(text) -> str               # hf_·sk-·ghp_ 토큰, (api_key|token|secret|password)\s*[=:]\s*\S+ (→ 이름=***), 영숫자 32자+ 연속을 ***로
def load_backlog(path) -> list[dict]  # {"items":[...]}의 items. 없으면 FileNotFoundError, 형태/JSON 오류 ValueError
def hourly_bins(events, now, hours=HOURS) -> list[float]   # 한 kind의 LedgerEvent들. 오래된 칸부터, 칸 i = [now-(hours-i)h, now-(hours-i-1)h), math.fsum
@dataclass(frozen=True, kw_only=True) class ReportData     # 아래. to_dict()는 JSON 직렬화 가능
def collect(*, now, items, ledger=None, config=None, window_h=1.0, hours=HOURS) -> ReportData
def summarize_ko(d) -> str
def bar_rects(values, box, vmax=None) -> list[tuple[int,int,int,int]]
def render_png(d, path) -> Path
def write_report(out_dir, d) -> tuple[Path, Path]          # (png, md)
def main(argv=None) -> int
```
`now`는 유닉스 초. 창은 반열림 `[now-window_h*3600, now)`; `ts==now`와 미래 이벤트는 어디에도 세지 않는다. `now` 비유한·`hours<1`·`window_h<=0`·알 수 없는 status는 `ValueError`.

## 데이터 형태
`items`: backlog 항목(`id`, `status`∈todo|doing|done|blocked, `log:[{ts,event,note?}]`). `ledger`: FND-07 `Ledger`|None. `config`: `Config`|None(캡 없음).
`ReportData`: `schema, now, window_start, window_end, window_h, hours, ledger_available`, `spend_window/spend_today/spend_total: dict[kind,float]`(장부 없으면 `{}`), `hourly: dict[kind,list[float]]`, `caps: dict[kind,(daily|None,total|None)]`, `counts`(네 status 키), `total_units`, `done_in_window: list[id]`(정렬), `durations: dict[id,시간]`, `blocked: list[(id,note)]`(정렬), `alerts: list[str]`.
- 오늘 = `now`가 속한 UTC일의 `[00:00Z, now)`. 누적 = `ts < now` 전부.
- `done_in_window`: status가 done이고 **마지막** `done` 이벤트 ts가 창 안. `durations`: 첫 `claimed` ts → 마지막 `done` ts (claimed 없으면 생략).
- `blocked` note: 마지막 `blocked` 이벤트의 note(없으면 ""), `redact` 후 80자 이하.
- `alerts`: 캡>0이고 USD 일/누적 사용률 ≥ `WARN_FRAC`(경계 포함)이면 `경고: 일 예산 {pct:.0f}% 사용` / `경고: 누적 예산 {pct:.0f}% 사용`.

## 요약 (`summarize_ko`: 결정론, 개행 종료, 30줄 이하, 비밀 비노출)
```
[BUC 시간별 보고] 2026-10-11 09:30 KST (2026-10-11 00:30Z)
진행: 완료 3/7 (42.9%) · 진행 중 1 · 막힘 1 · 대기 2        (0단위면 0.0%)
지난 1시간 완료: 2건 — A, C                                 (0건이면 "0건"만)
실소요: A 2.00h, C 0.50h                                   (비면 줄 생략)
막힘: E — 사유                                              (사유 없으면 "막힘: E")
지출(지난 1시간): $5.50 · GPU 0.25h · TFU 12.5
지출(오늘 UTC): $2.00 / 일 캡 $10.00 (20%)                  (캡 없으면 "/ 일 캡 없음", 캡 0이면 % 생략)
지출(누적): $6.50 / 누적 캡 $35.00 (19%)
경고: 일 예산 83% 사용                                      (alerts, 있을 때만)
```
장부 없으면 지출 세 줄 대신 `지출: 장부 없음 — 기록을 읽지 못했다`.

## 그림 (`render_png`)
RGB `CANVAS` PNG, 안티에일리어싱 없는 단색 채움, 같은 입력 → 같은 바이트, 임시 파일 후 `os.replace`(부모 디렉터리 생성, 잔여 없음). 배경 `COLOR_BG`, 제목·요약 한글 텍스트는 `CHART_BOX` 밖.
- `CHART_BOX`: `hourly["usd"]` 막대(없으면 빈 차트)를 `bar_rects` 사각형 그대로 `COLOR_USD`로 채운다. 박스 안에 글자 금지, 눈금선은 `COLOR_USD`가 아닌 색만.
- `bar_rects`: n칸 `s=(x1-x0)/n`; `xa=round(x0+i*s+0.15*s)`, `xb=round(x0+(i+1)*s-0.15*s)`(배타), `h=round((y1-y0)*v/vmax)`, 사각형 `(xa, y1-h, xb, y1)`. `vmax` 기본 `max(values)`(0이면 1.0). 음수·비유한 값, 퇴화 박스는 `ValueError`.
- `PROGRESS_BOX`: 왼쪽 `round(w*done/total)` 픽셀만 `COLOR_DONE`, 나머지 `COLOR_TRACK`.

## 쓰기와 CLI
`write_report`: UTC `d.now` 기준 `hourly-YYYYMMDDTHHMMZ.png`·`.md`. md = `summarize_ko(d)` 그대로(재실행은 같은 바이트로 덮어씀).
`main`: `--out`(기본 `work/reports`) `--backlog`(기본 `ops/harness/backlog.json`) `--ledger`(기본 `load_config().ledger_path`, `BUC_*` 반영) `--now UNIX` `--hours` `--window-h`. 성공: stdout JSON 한 줄 `{"ok":true,"png","md"}`, 반환 0. backlog 없음/깨짐: `{"ok":false,"why"}`, 반환 1, 파일 쓰지 않음. 장부 파일 없음: 열지 않고 `ledger_available=False`로 진행. 잘못된 인자: `SystemExit(2)`.

## 수용 기준
`tests/run_07/` 전부 통과(CPU, <60 s, 동결 해시): 위 규칙의 경계 수치, 요약 줄 정확 일치, 비밀 가림, 막대·진행 막대 픽셀, 결정론, CLI 코드, 읽기 전용, import 위생.

## 하류 이음매
오케스트레이터·공장장은 `python3 ops/report/hourly.py --out work/reports`로 `.md`(본문)와 `.png`(첨부)를 얻는다. 장부는 FND-07 `Ledger.total/events`로만 읽는다. 소유 밖 요청 없음.
