# DAT-16 크기 사전 테이블 v1 — 약 500 클래스 로그정규 (μ, σ) + 검사기

**목적.** R6 미터 스케일(RD-08)이 쓰는 수작업 크기 사전. 클래스 c의 물체 크기 h는 로그정규, 즉 `ln h ~ N(μ_c, σ_c²)`, `μ_c = ln(median_m)`. RD-08은 `log s* = Σ w_k(μ_k − ln ĥ_k)/Σ w_k`, `w_k = q_k/(σ_k² + σ̂²_h,k)`를 푼다(pipeline.upg 2.6). 같은 표가 학습의 "크기 거짓말"(×0.5–2) 판정과 에셔 r_size에도 쓰인다. 값 하나가 틀리면 장면 전체 미터 스케일이 틀어지므로 **데이터와 검사기를 따로 둔다**: 검사기가 구조·단위·내부 일관성을 기계로 거르고, 의미(실제 크기)는 도메인 검수와 동결 앵커 테스트가 거른다.
**소유.** `data/size_prior.yaml`, `tools/size_prior_check.py`, 테스트 `tests/dat_16/`(동결). 의존: FND-01(레포 골격)만. 도구는 **표준 라이브러리 + PyYAML + numpy**만, torch·cv2·`buc` import 금지, import 부작용 없음(파일은 `load_table`/`main` 때만 읽는다), 순수·결정론. 파일 경로로 적재 가능해야 한다(`tools/`는 패키지 아님. 스크립트로도 실행: `python3 tools/size_prior_check.py [--path P --min-classes N --max-classes N --min-core N]`).

## 데이터 형태 (`data/size_prior.yaml`, UTF-8)
```yaml
schema: 1
unit: m
classes:
  door_interior:                 # 정본 ID ^[a-z][a-z0-9_]{1,39}$
    dim: height                  # height|width|length|diameter|depth: 크기를 말하는 축(세워 둔 것은 height, 긴 것은 length, 둥근 것은 diameter)
    median_m: 2.03               # 중앙값(미터) = exp(μ). 정수 허용, bool·문자열 금지
    sigma_log: 0.04              # σ (자연로그 단위), 0.02 ≤ σ ≤ 0.9
    range_m: [1.88, 2.2]         # 중앙 95% 구간 ≈ median·exp(±2σ). 작성자의 이중 입력(오타 검출용)
    group: architecture          # GROUPS 중 하나
    tier: 1                      # 1 = 핵심(수직 슬라이스, 도메인 검수 우선), 2 = 나머지
    aliases: [door, interior_door]   # 선택. 정규형(소문자·밑줄)으로 저장
    note: "..."                  # 선택, 문자열
```
키는 `dim median_m sigma_log range_m group tier`(필수) + `aliases note`(선택)뿐이고 다른 키는 오류. 상수: `DIMS={height,width,length,diameter,depth}`, `GROUPS={animal,appliance,architecture,clothing,container,electronics,food,furniture,misc,nature,person,plant,sport,street,tableware,tool,vehicle}`.
`GROUP_BOUNDS`(중앙값 허용 구간, m): architecture 0.02–150, furniture 0.1–4, appliance 0.05–3, tableware 0.01–1, electronics 0.01–3, vehicle 0.5–30, person 0.03–2.2, animal 0.005–8, plant 0.005–60, street 0.05–30, tool 0.01–3, sport 0.02–3, container 0.02–3, clothing 0.02–2, food 0.01–1, nature 0.005–200, misc 0.005–100 (cm를 m로 착각한 오류 검출용).

## 불변식 (검사기 `check_raw`가 강제)
- 규모: `Limits(min_classes=480, max_classes=600, min_core=150)`(기본 `DEFAULT_LIMITS`), tier 1 ≥ min_core. `Limits`는 frozen dataclass, `0 ≤ min_core ≤ min_classes ≤ max_classes` 아니면 `ValueError`.
- 정본 ID와 별칭은 합쳐서 전역 유일(자기 이름·같은 클래스 안 중복 포함).
- 값: 유한, `0.005 ≤ median_m ≤ 200`, `0 < lo < hi`, `lo ≤ median ≤ hi`; 일관성 `|ln median − ½ln(lo·hi)| ≤ 0.15` 이고 `|σ − ln(hi/lo)/4| ≤ 0.35σ + 0.01`; 중앙값은 그룹 허용 구간 안.
- 입력은 어떤 쓰레기(None·정수·빈 dict·클래스 값이 None)라도 **예외 없이** 문제 목록으로 돌려준다. 입력을 바꾸지 않는다.

## 공개 API (`tools/size_prior_check.py`)
```python
SCHEMA_VERSION = 1; DIMS; GROUPS; GROUP_BOUNDS; DEFAULT_PATH: Path   # = <repo>/data/size_prior.yaml (파일 위치 기준)
class SizePriorError(ValueError): problems: tuple[str, ...]     # str(e) = 줄바꿈으로 이은 problems
@dataclass(frozen=True) Limits(min_classes=480, max_classes=600, min_core=150);  DEFAULT_LIMITS
def read_raw(path) -> dict          # YAML 적재. 중복 매핑 키 -> E_DUP, 파싱·파일 오류 -> E_PARSE (SizePriorError). safe_load만
def check_raw(raw, *, limits=DEFAULT_LIMITS) -> list[str]   # 정렬·중복 없음. 빈 목록 = 유효
@dataclass(frozen=True) ClassPrior(name, dim, median_m, sigma_log, range_m: tuple[float,float], group, tier, aliases: tuple[str,...], note: str|None)
    .mu_log -> ln(median_m);  .quantile(p) -> median·exp(σ·Φ⁻¹(p)) (0<p<1 아니면 ValueError)
    .z(size_m) -> (ln size − μ)/σ (size ≤0·비유한 ValueError);  .contains(size_m, k=2.0) -> |z| ≤ k
@dataclass(frozen=True) SizePriorTable(classes: dict[str, ClassPrior])
    len / 정렬 순회 / `name in t`(정본만) / `t[name]`(정본만, 없으면 KeyError)
    .resolve(text) -> ClassPrior   # strip·소문자·공백/하이픈 -> '_' 정규화 후 정본, 그다음 별칭. 없으면 KeyError
    .names(*, group=None, tier=None) -> tuple[str,...] (정렬);  .core() = names(tier=1);  .groups()
    .arrays(names) -> (mu_log, sigma_log)  # float64 (n,), 별칭 허용, 모르는 이름 KeyError, 빈 입력 (0,)
def load_table(path=None, *, limits=DEFAULT_LIMITS) -> SizePriorTable   # read_raw + check_raw, 문제가 있으면 SizePriorError(모든 problems)
def main(argv=None) -> int          # 0 유효("OK <n> classes ..." 출력), 1 문제(줄마다 출력), 2 읽기·파싱 불가
```
**문제 문자열** = `"<코드>: <클래스|<table>>: <설명>"`. 코드: `E_SCHEMA`(키·타입·버전·단위), `E_NAME`(이름·별칭 형식), `E_DUP`(별칭·키 중복), `E_ENUM`(dim·group·tier 값), `E_VALUE`(수치 범위·유한성·range), `E_CONSIST`(median·σ·range 불일치), `E_UNIT`(그룹 허용 구간 밖), `E_COUNT`(규모·핵심 수), `E_PARSE`. 타입이 틀리면(bool·문자열 수치·tier가 1.0) E_SCHEMA, 타입은 맞고 값이 틀리면 E_ENUM/E_VALUE.

## 표 내용 요구 (동결 테스트가 검증)
- 정본 앵커 표 `tests/dat_16/dat16_anchors.py`(약 70 클래스: dim·group·중앙값 구간·σ 구간)가 **정본**이다. 문 2.0–2.1 m, 좌석 높이 0.42–0.48 m, 계단 단 0.15–0.19 m, 성인 키 1.5–1.9 m(사양 문구)를 포함하며 전부 tier 1이다. 크기 순서 사슬, 쌍 비율(문/성인 키 등), 규격품 σ ≤ 0.08 대 변동품 σ ≥ 0.15, 복붙 방지(중앙값 서로 다른 값 ≥ 45 %, 자릿수 폭 ≥ 2.5)도 검사한다.
- 별칭 계약: `person`·`human` → `person_adult`, `door` → `door_interior`. `chair car tree dog cat bicycle laptop bench book`은 정본으로 존재.
- 12개 이상의 group 사용, 주요 그룹(건축·가구·사람·동물·탈것·거리·식물·식기·전자)은 각 8개 이상.

## 수용 기준
`tests/dat_16/` 전부 통과(CPU, <60 s, 실제로 수 초): 검사기 변조 탐지(코드별, 단위 오류 ×100, 중복 YAML 키 함정), CLI 종료 코드·출력, 로그정규 수학(분위수·z), `resolve`·`arrays`, 실제 표의 규모·앵커·순서·σ 위생, RD-08 로그 가중 최소제곱 이음매(중앙값 입력에서 스케일 비트 복원, 몬테카를로 2σ 포함률 93–97.5 %).

## 하류 이음매
- RD-08 `buc/read/metric_scale.py`: `t.arrays(classes)`로 μ,σ를 받아 위 식을 푼다. 검출 어휘(RD-07)는 `t.resolve(text)`로 정본에 매핑하고 `KeyError`는 "사전 없음"(그 객체는 LS에서 제외)으로 취급한다. 학습 '크기 거짓말' 생성기·에셔 r_size는 `ClassPrior.z / contains / quantile`.
- `tools/`는 패키지가 아니다. `buc` 쪽에서 쓰려면 경로 적재 또는 얇은 래퍼가 필요하다(이 단위는 만들지 않음; 요청 사항).
- 표 값의 의미(어느 축을 재는가)는 `dim`이 정한다. 값 변경은 도메인 검수(Opus) 후 `tier`, `note`에 근거를 남긴다.
