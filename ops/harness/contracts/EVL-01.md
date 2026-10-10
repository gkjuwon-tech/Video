# EVL-01 고정 입력 24장 매니페스트 — 출처 태그·크롭 규약·커버리지 매트릭스

**목적.** EVL-02 일일 A/B·EVL-03 경쟁 비교·D15 데모가 매일 같은 24장, 같은 크롭 파일, 같은 무브로 돈다. 이 단위는 그 24장의 **장부(manifest)와 기계 검사기**다. 선별 편향(이기는 칸만 고르기, 예측 패널: 실사 안전 0.40 대 그림체 풀 0.04)이 승률 해석을 오염시키지 않게 구성(커버리지 매트릭스)을 규칙으로 고정하고, 계보 태그(FND-04)로 24장이 학습·보정·회귀로 새지 않음을 강제한다. 매트릭스 의미는 Fable 적대 검토 대상(이 단위 밖).
**소유.** `eval/inputs24/manifest.yaml`, `tools/inputs_qa.py`, 테스트 `tests/evl_01/`(동결). 의존 FND-01, 그리고 `buc.gov.prov`(FND-04, 표준 라이브러리 전용)만 import. 도구는 표준 라이브러리 + PyYAML + `buc.gov.prov`만(torch·numpy 금지). 레포 루트를 `sys.path`에 넣는 것 외 import 부작용 없음. `tools/`는 패키지가 아니다(파일 경로 적재, 스크립트 실행 `python3 tools/inputs_qa.py [--path P] [--ready] [--matrix]`).
**공개 레포에 이미지 원본을 넣지 않는다.** 원본은 HF 비공개 금고 `VAULT_REPO`에 있고 매니페스트는 경로·SHA-256·크기만 적는다. `eval/` 아래 이미지 파일 0개.

## 데이터 형태 (`manifest.yaml`)
최상위 키 정확히: `schema: 1`, `name: inputs24`, `vault_repo: sdsdsdsdsdsdsd123/buc-artifacts`, `moves`(비지 않은 중복 없는 ⊂ `MOVES`=I1..I5, bench §2.3), `selection`{`method` 비지 않은 str, `candidate_pool` int ≥24, `seed` int, `disclosure` 비지 않은 str(데모 릴 "선별했음" 캡션 문구)}, `items`(list).
항목 필수 키 정확히(+선택 `note` str, `moves` list): `slot` 1..24, `id` = `in{slot:02d}`, `title` `^[a-z][a-z0-9_]{1,39}$`(전역 유일), `status` planned|ready, `kind` photo|gen|painting, `origin` capture|asset|extgen|engine|unverified, `category` C1..C6(bench §2.3; C6 = 그림체), `features` ⊂ `FEATURES`={text,reflective,fur,people,face,night,indoor,outdoor,macro,far,bokeh}(중복 불가), `difficulty` easy|mid|hard, `picked_by` founder|random|third_party|hero, `taint` 비지 않은 list[str], `generator` str|null, `vault_path` str|null, `sha256` 64자 소문자 hex|null, `size` [w,h] int|null, `crop` {`aspect` ∈ `ASPECTS` 키, `mode` center|manual, `box` [x0,y0,x1,y1] int, 선택 `reason`}|null.
`ASPECTS` = native, 1:1, 4:3, 3:2, 16:9, 21:9, 2.39:1, 9:16. bool은 int가 아니다.
**코드 배정:** 키 누락·여분·타입 틀림·형식(title 정규식, 빈 moves·taint·disclosure, 고정값 schema/name/vault_repo 불일치)은 E_SCHEMA; 타입은 맞고 열거 밖 값(kind, origin, category, features, difficulty, picked_by, status, moves 원소, crop.aspect·mode)은 E_ENUM; 같은 리스트 안 중복(features·moves)은 E_DUP.

## 불변식 (`check_raw`)
- 정확히 24항목, slot 1..24 각 1회, id 일치(E_SLOT). title·sha256·vault_path 중복은 E_DUP.
- `status: ready` ⇒ vault_path·sha256·size·crop 모두 non-null. `planned` ⇒ sha256·size·crop null(E_READY). `require_ready=True`면 planned 항목마다 정확히 하나의 E_READY. planned 항목도 kind·category·features 등 **칸 정보는 필수**(매트릭스는 촬영 전에 고정).
- 일관성(E_CONSIST): photo ⇒ origin ∈ {capture, asset, unverified}; gen ⇒ {extgen, engine, unverified}; painting ⇒ {asset, extgen, engine, unverified}; painting ⇔ C6; generator non-null ⇔ origin ∈ {extgen, engine}.
- 계보(E_TAINT, `buc.gov.prov` 사용): 모든 태그가 `normalize_tag` 통과(별칭 `SOUL` 허용); **모든 항목에 `EVAL-ONLY`**(⇒ `train_ok`·`dev_ok` 거짓: 24장은 학습·보정·회귀셋 금지); capture ⇒ `CAPTURE`; asset ⇒ 한정자 있는 `ASSET:<lic>`; extgen ⇒ `EXTGEN:<generator>`; engine ⇒ `ENGINE`; unverified ⇒ `ext_bit` 참(권리 미확인은 닫힌 쪽).
- 경로(E_PATH): vault_path는 상대 POSIX, `..`·선행 `/`·`\`·`://` 금지, 확장자 ∈ .webp .png .jpg .jpeg .heic .exr .tif .tiff.
- 값(E_VALUE): size 긴 변 512–8192; 크롭 후 긴 변 ≥512, origin capture면 ≥1920(미러리스 2K); sha256 형식; candidate_pool ≥24.
- 크롭(E_CROP): box는 `0 ≤ x0 < x1 ≤ w`, `0 ≤ y0 < y1 ≤ h`. center ⇒ box == `center_crop_box(w,h,aspect)`. manual ⇒ `reason` 비지 않고 `|(bw/bh)/a − 1| ≤ 0.01`(native면 a = w/h).
- 히어로(E_HERO, docs/DEMO-D15.md): slot 1 = `HERO` — title `d15_hero_yak`, vault_path `demo/d15_hero_yak.webp`, sha256 `7c1fc8dcebcfb485e93cff2b206405dd4af155acf3085213df82128876362bda`, size [2000,861], status ready, origin unverified, kind ∈ {photo, gen}, category C3, picked_by hero, features ⊇ {fur, people, bokeh, far, outdoor}, crop native·center(전체 프레임, 2.32:1 유지). `picked_by: hero`는 slot 1에만.
- 비밀(E_SECRET): 어떤 문자열에도 `hf_[A-Za-z0-9]{20,}`, `sk-[A-Za-z0-9]{20,}` 패턴 금지.
- 커버리지(E_COVER, `COVERAGE` 상수): kind photo 10–14, gen 6–10, painting 3–5; capture ≥8; extgen ≤10; C1..C5 각 ≥2이고 각각 photo ≥1; text ≥2, (text∧indoor) ≥1, reflective ≥2, people ≥3, night ≥2, fur ≥1; hard ≥6, easy ≤10; random+third_party ≥6; public_ok ≥10.
- `public_ok` = `not ext_bit(taint) and origin != "unverified"`(공개 주장은 공개 출처만, bench CT-B2).
- 예외 없이 문제 목록 반환(None·int·list·빈 dict·None 항목 등 어떤 쓰레기도). 입력을 바꾸지 않는다. 결과는 정렬·중복 없음. 문제 문자열 = `"<코드>: <inNN|item[i]|<manifest>>: <설명>"`. 코드: E_SCHEMA(키·타입·버전·고정값), E_ENUM, E_SLOT, E_DUP, E_READY, E_CONSIST, E_TAINT, E_PATH, E_VALUE, E_CROP, E_HERO, E_SECRET, E_COVER, E_PARSE.

## 공개 API (`tools/inputs_qa.py`)
```python
SCHEMA_VERSION = 1; N_ITEMS = 24; VAULT_REPO: str; MOVES; KINDS; ORIGINS; CATEGORIES; FEATURES; ASPECTS: dict[str, float|None]
HERO: dict; COVERAGE: dict; DEFAULT_PATH: Path   # <repo>/eval/inputs24/manifest.yaml (파일 위치 기준)
class ManifestError(ValueError): problems: tuple[str, ...]
def read_raw(path) -> dict                        # safe_load. 중복 키 E_DUP, 파일·파싱 오류 E_PARSE (ManifestError)
def center_crop_box(w, h, aspect: str) -> tuple[int,int,int,int]
    # native → (0,0,w,h). 아니면 a=ASPECTS[aspect]; w/h > a 이면 bh=h, bw=round_half_up(h·a), 아니면 bw=w, bh=round_half_up(w/a);
    # x0=(w−bw)//2, y0=(h−bh)//2. 모르는 aspect·w,h ≤0·비int ValueError
def check_raw(raw, *, require_ready=False) -> list[str]
@dataclass(frozen=True) Item(slot, id, title, status, kind, origin, category, features: frozenset[str], difficulty, picked_by,
    taint: tuple[str,...] (정규화·정렬), generator, vault_path, sha256, size: tuple[int,int]|None,
    crop_aspect: str|None, crop_box: tuple[int,int,int,int]|None, public_ok: bool)
@dataclass(frozen=True) Manifest(items: tuple[Item,...] (slot 순), moves: tuple[str,...], selection: dict, vault_repo: str)
    .hero -> Item(slot 1); .by_slot(n) -> Item (없으면 KeyError); .ready() -> tuple[Item,...]
def load_manifest(path=None, *, require_ready=False) -> Manifest   # 문제가 있으면 ManifestError(전부)
def coverage(m: Manifest) -> dict   # {"kind","origin","category","difficulty","picked_by","features": {값: 개수, 모든 열거값 0 포함},
                                    #  "matrix": {C1..C6: {photo,gen,painting: n}}, "public_ok": n, "ready": n, "total": 24}
def main(argv=None) -> int          # 0 "OK 24 items (ready r/24) ..."(--matrix면 다음 줄에 coverage JSON), 1 문제 줄마다, 2 읽기·파싱 불가
```

## 실제 매니페스트 요구
지금은 히어로만 ready, 나머지 23칸은 planned(창업자 촬영 D1–4 후 sha256·size·crop을 채운다. **해시를 지어내지 않는다**). 계획 모드 `load_manifest()`가 통과해야 하고, 위 커버리지를 칸 정보만으로 만족해야 한다(실사 12 안팎 미러리스 2K + 간판·글자 실내, 외부 생성기 8 안팎, 그림 4 안팎). `eval/` 아래 이미지 0개, 비밀 패턴 0개.

## 수용 기준
`tests/evl_01/` 전부 통과(CPU, 수 초): 크롭 수학, 변조 탐지(코드별 30여 종), 쓰레기 입력 무예외·무변경, ready 모드, coverage 집계 정확값, read_raw 중복 키·파싱 오류, CLI 종료 코드, 실제 매니페스트(히어로 고정값, 구성, 전 항목 train_ok 거짓, 이미지·비밀 없음), torch 미적재.

## 하류 이음매
EVL-02: `load_manifest(require_ready=True)`로 24장·`moves`·`crop_box`를 받아 같은 크롭 파일로 렌더하고, 시트에 `coverage(m)["matrix"]`와 `selection.disclosure`를 붙인다. EVL-03·PRD-06: 경쟁 입력은 같은 `crop_box`(bench F-P12 "같은 크롭·같은 파일"), 주점은 크롭 좌표에서 다시 푼다(R1). 금고 다운로드는 RUN-03 몫이고 받은 바이트의 SHA-256을 `Item.sha256`과 대조해야 한다. 공개 수치는 `public_ok` 항목만.
