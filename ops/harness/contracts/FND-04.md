# FND-04 계보 태그·prov 비트 v1 — taint·ext/comp/hold·BLOCK_TRAIN/DEV

**목적.** 방화벽의 어휘. 권리 렌즈 v3(`work/wf3/lens/rights.upg.md` §1.2–1.3) 태그 체계, `derive` 전파, `train_ok`/`dev_ok`/`signal_class`, Merkle 키용 (ext, comp, hold, ns) 비트를 **한 곳**에 둔다. GOV-01·DAT-05·TCH-03/04/10·EVL-03/10·GOV-06이 이것만 import한다. 모든 판정은 **닫힌 쪽 실패(fail closed)**: 모르는 태그·라이선스·역할은 금지 쪽.
**소유.** `buc/gov/prov.py`(표준 라이브러리만, torch·numpy 금지, import 부작용 없음), 테스트 `tests/fnd_04/`(동결). 의존: FND-01(패키지). FND-02 `contracts.Prov`는 duck-typing으로만 받는다(import 강제 아님).

## 태그 문법
`BASE` 또는 `BASE:<한정자>`(첫 `:`로 분리, 한정자 비어 있지 않고 공백 없음). str 아니면 `TypeError`, 모르는 BASE·문법 위반은 `TagError`.
`KNOWN_BASES` = ENGINE, CAPTURE, ASSET, OPEN-GEN, HUMAN, SI, TWIN, COMMISSION, USER, USER-DERIVED, BIO, EXTGEN, EXTGEN-DERIVED, EXTGEN-INSPIRED, COMP, HOLDOUT, PUB-SEALED, EVAL-ONLY, IP-FLAG, OPTIN, TEL-SYS, TEL-CAGG, D3, NC.
- 한정자 필수: `EXTGEN:<g>`, `EXTGEN-DERIVED:<g>`, `EXTGEN-INSPIRED:<g>`, `TWIN:<c>`, `OPEN-GEN:<m>:<h>:<role>`(정확히 3필드, role ∈ {train, dev-eval, bench}), `COMMISSION:<s>[+<s>…]`(s ∈ {eval, train, publish}, 중복 불가).
- `IP-FLAG`(가림 전) / `IP-FLAG:masked`(가림 후). 다른 한정자 `TagError`. `ASSET[:lic]`(lic 없으면 라이선스 미상). 나머지 BASE의 한정자는 자유 문자열.
- 별칭(`normalize_tag`): `SOUL`→`EXTGEN:soul`, `SOUL-DERIVED`→`EXTGEN-DERIVED:soul`, `SOUL-INSPIRED`→`EXTGEN-INSPIRED:soul`, `COMMISSION` scope는 eval<train<publish 순으로 정렬.

## 상수
`EXT_BASES` = {USER, USER-DERIVED, BIO, EXTGEN, EXTGEN-DERIVED, EXTGEN-INSPIRED}
`BLOCK_TRAIN` = EXT_BASES ∪ {COMP, HOLDOUT, PUB-SEALED, EVAL-ONLY, TEL-CAGG, OPTIN, D3, NC}
`BLOCK_DEV` = EXT_BASES ∪ {COMP, HOLDOUT, PUB-SEALED, EVAL-ONLY, TEL-CAGG, OPTIN}
(BASE 기준 매칭, 한정자 무관. §1.3 의사코드보다 엄격하다: §1.2 표의 HOLDOUT·INSPIRED·TEL-CAGG·OPTIN "개발 금지/J만/브랜치만"을 반영.)
`TRAIN_LICENSES` = {CC0, PD, CC-BY, Apache-2.0} · `MODES` = ("none","inspired","derived","authored")(순서 = 강도) · `NAMESPACES` = ("user_vault","ext_vault","comp_vault","pub_bench","train_ok")(앞일수록 엄격) · `PROV_TAG` = "buc.prov.v1".

## 공개 API
- 예외: `TagError(ValueError)`, `ProvError(ValueError)`, `FirewallError(RuntimeError)`.
- `normalize_tag(s) -> str`, `parse_tag(s) -> tuple[str, str|None]`(정규화 후 (base, qual)).
- `@dataclass(frozen=True) ProvRecord(taint: frozenset[str]=∅, mode="none", roots: tuple[str,...]=(), ns: str|None=None)`: taint 정규화, roots 정렬·중복 제거, mode∉MODES·ns∉NAMESPACES `ProvError`. ns=None이면 `implied_ns(taint)`; 명시 ns가 그보다 느슨하면 `ProvError`. 값 동등·해시 가능. 속성 `ext`, `comp`, `hold`, `bits`. `to_dict()`(JSON 가능, `"v": PROV_TAG`, 정렬 리스트) / `from_dict(d)`(v 불일치 `ProvError`).
- `as_record(x) -> ProvRecord`: ProvRecord 그대로, `.taint/.mode/.roots/.ns`를 가진 객체(FND-02 Prov: mode "engine"→"none", "mixed"→"derived", ns "engine"→"train_ok", 결과 ns = most_restrictive(변환값, implied)), Mapping(to_dict 형태, v 없어도 됨), str 아닌 iterable[str](태그 집합). 단일 str·bytes·None `TypeError`. 아래 함수의 `x`는 모두 이것을 거친다.
- `implied_ns(taint) -> str`: USER·USER-DERIVED·BIO→user_vault, 그 외 EXTGEN 계열→ext_vault, COMP·EVAL-ONLY→comp_vault, HOLDOUT·PUB-SEALED→pub_bench, 아니면 train_ok. `most_restrictive(*ns) -> str`(빈 입력 `ValueError`).
- 비트: `ext_bit(x)` = taint∩EXT_BASES≠∅ · `comp_bit(x)` = COMP 존재 · `hold_bit(x)` = HOLDOUT 또는 PUB-SEALED 존재 · `prov_bits(x) -> int` = ext·1 | comp·2 | hold·4 · `merkle_parts(x) -> ("buc.prov.v1", prov_bits, ns)`(FND-03 `hash_parts`에 그대로) · `content_address_kind(x)` = "hmac" if ext else "sha256".
- `derive(op: str, parents: Sequence[ProvLike], *, human_ref=False, updates_shared_weights=False) -> ProvRecord`. op 빈 문자열·parents 비면 `ValueError`. 부모 taint를 태그별로 변환해 합집합: `USER[:q]`→`USER-DERIVED[:q]`; `EXTGEN:g`→`EXTGEN-DERIVED:g`; human_ref면 EXTGEN·EXTGEN-DERIVED·EXTGEN-INSPIRED `:g` → `EXTGEN-INSPIRED:g`(USER 계열에는 INSPIRED가 생기지 않는다); 나머지는 그대로. roots 합집합. mode = "inspired" if human_ref ∧ 어떤 부모 ext else max(부모 mode). ns = most_restrictive(부모 ns…, implied). `updates_shared_weights ∧ 결과.ext` → `FirewallError`("CT-M5").
- `why_not_train(x, branch="main") -> tuple[str,...]`(정렬, 비면 허용). main: BLOCK_TRAIN 태그 그 자체, `IP-FLAG`(가림 전), OPEN-GEN role≠train, COMMISSION scope∌train, ASSET lic∉TRAIN_LICENSES(미상 포함), mode=="inspired" ∧ ext면 `"Q17"`. branch "optin-v1": BASE ∉ {CAPTURE, ENGINE, ASSET, HUMAN, OPTIN}인 태그. 그 밖의 branch `ValueError`. `train_ok(x, branch="main") = not why_not_train(...)`.
- `why_not_dev(x)` / `dev_ok(x)`: BLOCK_DEV 태그, `IP-FLAG`(가림 전), OPEN-GEN role ∉ {train, dev-eval}, COMMISSION scope∌eval.
- `signal_class(x) -> "S" | "J"`: dev_ok면 "S". `assert_train_ok(x, what="")`/`assert_dev_ok(x, what="")`: 위반 시 `FirewallError`(메시지에 사유 태그 포함).

## 불변식
1. **단조성:** derive 결과가 train_ok(dev_ok)이면 모든 부모가 train_ok(dev_ok). ext/comp/hold 비트는 부모 OR. ns는 부모 어느 것보다 느슨하지 않다. human_ref도 이를 깨지 않는다(세탁 금지).
2. derive는 부모 순서·중복과 무관(같은 레코드), 재적용해도 taint 불변(`derive(op,[derive(op,[p])]).taint == derive(op,[p]).taint`, human_ref=False).
3. 정규화는 멱등, 별칭 결과와 새 이름은 같은 레코드.
4. `merkle_parts` 배치는 디스크 계약: 바꾸면 `PROV_TAG` 버전을 올린다.

## 수용 기준
`tests/fnd_04/` 전부 통과(CPU, <5 s): 태그 문법·별칭·거부, 태그별 train/dev 진리표(§1.2 표), 카나리아 C1(`USER`)·C5(`EXTGEN:test`) 거부, 조건 규칙(IP-FLAG·OPEN-GEN·COMMISSION·ASSET·Q17·optin), 비트·Merkle 배치 고정, ns 엄격도, derive 변환·FirewallError, 시드 고정 무작위 600건 단조성·순서 무관, FND-02 Prov 변환, JSON 왕복.

## 하류 이음매 · 소유 밖 요청
GOV-01 `why_not_train`/`derive`로 조상 DAG 검사 · DAT-05·TCH-10 샤드·보정셋 `assert_train_ok` · TCH-03 ext면 Tap 우회(`ext_bit`) · TCH-04 키에 `merkle_parts`·`content_address_kind` · EVL-03 `COMP[:vendor]` · EVL-10/GOV-06 `hold_bit`·`signal_class`. **요청:** FND-02 `TAINT_BASES`에 USER-DERIVED·BIO·EXTGEN-DERIVED 등을 더하고 `Prov.ext`를 FND-04 `ext_bit` 정의(USER 포함)로 맞출 것 — 지금 FND-02 `Prov.ext`는 USER를 빼므로 방화벽 판정에 쓰면 안 된다. `buc/gov/__init__.py`는 없어도 namespace로 동작.
