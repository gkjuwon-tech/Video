# GOV-03 렌더 경로 린트 세트 v1 — no_input_pixels_in_render_path · canvas_extension_lint · refl_bake_lint · preset_lineage_check

**목적.** I6(Read-Only Camera)와 비회귀 가드 R-NR1·R-NR11·(e)를 기계 검사로 만든다(pipeline.upg §5.4-1, §9.3–9.5, §2 표 (e)). F_0 = Render(W, C_0, θ_in): 렌더 경로는 입력 버퍼를 import·참조하지 않고, 교사는 입력 크롭으로 초기화되지 않고, 반사 영역은 입력으로 덮이지 않으며, θ_in 은 공유 프리셋이 되지 않는다. 모든 판정은 **정적(AST, 실행 안 함)·결정론·닫힌 쪽**(파싱 실패 = 위반 `SYNTAX`). 프라그마/예외 주석 없음.
**소유.** `buc/gov/lints.py`(표준 라이브러리 + `buc.gov.prov`; `buc.lens.theta`·`buc.core.contracts`는 프리셋 함수 안에서만 지연 import → 모듈 import 시 torch·numpy 미적재). 테스트 `tests/gov_03/`(동결). 의존: FND-02(`Theta`), FND-04 `buc.gov.prov`(`as_record`, `ProvRecord`), LNS-01(`complete_extra`, `load_preset`, `Preset`).
**범위 밖.** 런타임 데이터 흐름 추적, 근사(1 ulp 이상 다른) θ 복제 탐지, 학습 방화벽(GOV-01), 카나리아 주입 잡(GOV-02).

## 공개 API (`buc.gov.lints`)
상수: `LINT_TAG="buc.lints.v1"`, `LINTS=("no_input_pixels_in_render_path","canvas_extension_lint","refl_bake_lint","preset_lineage_check")`, `REPO_ROOT`(=이 파일 `parents[2]`), `RENDER_PATH_DIRS=("buc/render","buc/lens","buc/camera")`, `CANVAS_DIRS=("buc/teacher","buc/world","buc/invent","buc/render","buc/lens","buc/camera","buc/models")`, `REFL_DIRS=CANVAS_DIRS`, `PRESET_SCAN_DIRS=("buc",)`, `PRESET_EXEMPT_DIRS=("buc/gov",)`, `INPUT_MODULES=("buc.read",)`, `INPUT_IDENTS={"image_lin","input_image","input_pixels","input_rgb","input_buffer","I_w"}`, `INPUT_ATTRS=INPUT_IDENTS|{"pixels"}`, `INPUT_TYPES={"InputImage","ReadPacket"}`, `IMAGE_READ_FUNCS={"imread","read_image","decode_image","decode_jpeg","decode_png"}`, `OUTPAINT_RE=(?i)(out_?paint|un_?crop|extend_?canvas|canvas_?exten(d|sion)|expand_?canvas|border_?extend|extend_?border)`(search), `INIT_KWARGS={"init","init_image","init_latent","init_latents","x0","x_init","z0","z_init","image","latents","start_image","source_image","init_rgb"}`, `PAD_FUNCS={"pad","copyMakeBorder"}`, `REFL_RE=(?i)^(m_?refl\w*|refl(ection)?_?mask\w*|mask_?refl\w*)$`(match), `MASKED_FUNCS={"masked_scatter(_)","masked_fill(_)","putmask","place","copyto","index_put(_)"}`, `BLEND_FUNCS={"lerp","mix","blend","composite","alpha_blend","over"}`, `PRESET_ALLOWED_BASES={"ENGINE","CAPTURE","ASSET"}`, `PRESET_PROVENANCE={"handmade":"ENGINE","engine":"ENGINE","capture":"CAPTURE"}`.

`@dataclass(frozen=True, order=True) Violation(path:str, line:int, lint:str, rule:str, detail:str)` + `to_dict()`(키 5개). 모든 함수의 결과는 **정렬·중복 제거된 list**. line 은 노드의 `lineno`(호출은 호출 시작 줄), SYNTAX 는 오류 줄.

**용어.** *입력 참조* = 부분식에 Name∈INPUT_IDENTS∪INPUT_TYPES 또는 Attribute.attr∈INPUT_ATTRS∪INPUT_TYPES. *반사 식별자* = Name.id/Attribute.attr 가 REFL_RE. *callee 끝 이름* = 호출 대상의 마지막 Name/Attribute. 문자열·주석은 무시.

`scan_render_source(src, path="<src>", module=None)` (lint 1). 상대 import 는 `module` 로 해석(파일명이 `__init__.py`면 module 자체가 패키지, 아니면 부모); module=None 이면 상대 import 무시. 규칙:
- `INPUT_IMPORT`: `import X`, `from X import n`(X 와 X.n 후보), `importlib.import_module("lit")`/`__import__("lit")` 중 INPUT_MODULES 또는 그 하위로 해석되는 것. detail=일치한 점 이름.
- `DYNAMIC_IMPORT`: import_module/`__import__` 인자가 문자열 리터럴이 아님.
- `INPUT_NAME`: Name·함수 인자·호출 키워드 이름 ∈ INPUT_IDENTS, Attribute.attr ∈ INPUT_ATTRS. detail=식별자.
- `INPUT_TYPE`: Name/Attribute/`from … import` 이름 ∈ INPUT_TYPES.
- `IMAGE_READ`: 별칭 해석 후 `PIL.Image.open`, 또는 callee 끝 이름 ∈ IMAGE_READ_FUNCS.

`check_render_path(root=REPO_ROOT, dirs=RENDER_PATH_DIRS)`: dirs 아래 `*.py`(없는 디렉터리 건너뜀, `__pycache__` 제외)를 path=root 기준 POSIX 상대 경로, module=경로에서 유도(`__init__` 제거)로 스캔 + `TRANSITIVE_IMPORT`: 렌더 모듈에서 root 아래 `buc.*` 정적 import 그래프(함수 안 import·리터럴 import_module 포함; `from X import n` 은 X.n 이 root 에 모듈/패키지로 있거나 입력 모듈이면 X.n, 아니면 X; import 별칭마다 간선)를 정렬 순 BFS. 처음 닿은 INPUT_MODULES 모듈까지의 최단 사슬이 3개 이상이면 위반, line=렌더 파일의 첫 간선 줄, detail=`"a -> b -> buc.read.x"`.

`scan_canvas_source(src, path, module=None)` (lint 2): `OUTPAINT_API`(def/class 이름, Name, Attribute, import 이름·별칭이 OUTPAINT_RE; detail=그 이름), `INPUT_INIT`(호출 키워드 ∈ INIT_KWARGS 이고 값이 입력 참조; detail=키워드), `PAD_INPUT`(callee 끝 이름 ∈ PAD_FUNCS 또는 점 이름이 `ImageOps.expand`로 끝나고 어떤 인자·키워드 값이 입력 참조).

`scan_refl_source(src, path, module=None)` (lint 3): `REFL_INPUT` detail ∈ {`where`: callee 끝 `where`, 첫 인자에 반사 식별자, 2·3번째 인자 중 입력 참조; `assign`: Assign/AugAssign 대상의 Subscript 인덱스에 반사 식별자, 값이 입력 참조; `mult`: `*` 의 한쪽이 *직접* 반사 식별자(Subscript 를 벗긴 Name/Attribute; `1-m` 은 아님), 다른 쪽이 입력 참조; `masked`/`blend`: callee 끝 ∈ MASKED_FUNCS/BLEND_FUNCS, 인자에 반사 식별자와 입력 참조가 모두}. `ABSORB_NO_REFL`: *잔차 흡수 이름*(소문자에 `absorb` 와 `resid`·`lowband`·`low_band` 중 하나) — def 는 REFL_RE 파라미터가 있고 본문에서 Load 되어야; 호출은 위치 인자에 반사 식별자가 있거나, REFL_RE 키워드가 리터럴 None/False 가 아니거나, 다른 키워드 값에 반사 식별자가 있어야. detail=함수 이름.

`scan_preset_source(src, path, module=None)` (lint 4, AST): `THETA_IN_TO_PRESET` = callee 식의 이름들 중 하나라도 `preset`(대소문자 무시)을 포함하고 인자·키워드 값에 Name/Attribute `theta_in`; 또는 Subscript 대입 대상의 값 식이 `preset` 포함, 값에 `theta_in`.
`preset_provenance_record(s) -> ProvRecord`: PRESET_PROVENANCE 또는 `asset:<lic>`→`ASSET:<lic>`, 그 밖(정확 일치만, str 아님 포함) `ValueError`.
`why_not_preset(prov) -> tuple[str,...]`(정렬): as_record 실패 `("UNREADABLE",)`; 빈 taint `NO_TAINT`; 베이스∉허용 `TAINT:<tag>`(정규화된 태그); ns≠train_ok `NS:<ns>`.
`@dataclass(frozen=True) PresetEntry(name, theta, prov)`. `check_presets(entries, *, project_thetas=())`: entry 는 PresetEntry 또는 LNS-01 Preset(provenance 문자열 → 모르면 `PROVENANCE`/detail=문자열). 사유 `R:D` → rule=R, detail=D, path=name, line=0. `THETA_IN_COPY`: `complete_extra(theta).content_hash()` 가 project_thetas 의 같은 해시와 정확히 일치, detail=해시 앞 16자.
`check_preset_dir(presets_dir=None, *, project_thetas=())`(None=LNS-01 `PRESETS_DIR`): `*.yaml|*.yml` 를 `load_preset`; 실패 → path=파일명, `UNREADABLE`.
`scan_paths(lint, root=REPO_ROOT, dirs=None, exempt=())`: 해당 lint 의 스캐너로 기본 디렉터리(lint 4 는 PRESET_EXEMPT_DIRS 면제) 스캔; 모르는 lint `ValueError`.
`run_all(root=REPO_ROOT, lints=LINTS) -> dict[lint, list]`(LINTS 순서): lint 1 = check_render_path, lint 4 = AST + `root/presets` 있으면 check_preset_dir. `main(argv=None) -> int`: `--root`, `--lint`(반복, choices=LINTS); stdout 마지막 줄 JSON `{"ok","violations":[to_dict…]}`(Violation 순 정렬); 0 깨끗/1 위반/2 argparse. `python3 -m buc.gov.lints` 동작.

## 불변식
1. 코드를 실행·import 하지 않는다(AST만). 같은 입력 → 같은 결과.
2. 현재 저장소: `run_all(REPO_ROOT)` 위반 0(렌더 경로·캔버스·반사·프리셋 20개 모두 깨끗).
3. 규칙 완화(상수 축소)는 LINT_TAG 를 올리고 Fable 검토.

## 수용 기준
`tests/gov_03/` 전부 통과(CPU, <60 s): 규칙별 진리표(위반·정상 쌍), 상대 import 해석, 전이 사슬, 줄 번호·detail·정렬, 실행 안 함, 범위 디렉터리, 프리셋 출처·ns·θ_in 복제, run_all 카나리아(린트마다 1건), CLI 종료 코드·JSON, 저장소 깨끗, torch·numpy 미적재.

## 하류 이음매
CI 워크플로가 `python3 -m buc.gov.lints` 실행(요청: .github 추가는 운영 단위). PRD-01 쇼트 끝 `Lint.no_input_pixels_in_render_path()` = `check_render_path()` 가 빈 리스트인지. 새 렌더 단계 디렉터리는 RENDER_PATH_DIRS 에 추가(TAG 올림). θ_in 은 프리셋 *이름*으로 기록하고 값 복사로 공유 라이브러리에 넣지 않는다. GOV-02 카나리아는 `run_all` 이 각 린트에서 1 을 내는지 본다.
