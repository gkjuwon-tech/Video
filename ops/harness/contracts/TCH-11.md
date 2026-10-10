# TCH-11 입구 생성 Entry-Gen — Z-Image T2I 8 NFE로 AI풍 dev·학습 입력 생산

**목적.** 엔진 교사 T2I를 **입구 생성**으로만 쓴다(CT-M9): 프롬프트 -> 8 NFE 이미지 1장, **선택 없음**(hero search 폐지). dev·학습 입력(AI풍 구운 심도·질감), T1-E 재현, 주조소·Self-Input 입력(DAT-09)용. TCH-02 `generate`를 청크마다 부르고 `decode_latent`로 풀어 PNG로 만드는 단순 루프다.
**소유.** `buc/teacher/entry_gen.py`, 테스트 `tests/tch_11/`(동결). 의존: TCH-02 `TeacherSampler`, TCH-01 `decode_latent`·`spec.check_image_size`, FND-03 `det`, FND-04 `ProvRecord`, numpy·torch·PIL. diffusers 없이 import, 부작용 없음. **소유 밖 요청: 없음.**

## 공개 API (`buc.teacher.entry_gen`)
- 상수: `ENTRY_TAG="buc.tch11.entry.v1"`, `MODES=("survey","cinematic","plain")`, `MODE_SUFFIX: Mapping[str,str]`(키 = MODES; `plain`은 `""`, 나머지는 비어 있지 않고 서로 다르며 포함 관계가 아님. 권장 문구: survey = 확산광·깊은 심도·정면, cinematic = 시네마틱 스틸·아나모픽·얕은 심도·그레인), `DEFAULT_TAINT=("ENGINE",)`.
- 예외: `EntryGenError(ValueError)` ⊃ `BannedPromptError`.
- `compose_prompt(prompt, mode="survey")->str`: `prompt.strip()`; suffix가 비면 그대로, 아니면 `f"{p}, {suffix}"`. str 아님 `TypeError`; 공백뿐·모르는 mode `EntryGenError`.
- `check_prompts(prompts, banned=())->tuple[str,...]`: 시퀀스(str 단독 `TypeError`), 원소 str(아니면 `TypeError`), strip 후 비면 `EntryGenError`, `banned`(정규식 str 또는 컴파일된 패턴; str은 대소문자 무시)에 하나라도 걸리면 `BannedPromptError`(메시지에 걸린 프롬프트). 금칙 목록은 CI 소유(rights 규칙 15), 여기서는 받아 적용만 한다.
- `entry_seed(seed:int, index:int)->int = det.derive_seed(seed, ENTRY_TAG, index)`; seed ∉ [0,2^64) 또는 index<0 이면 `ValueError`.
- `entry_key(prompt, mode, seed, height, width, table_key, level=0, k=0)->str` = `det.hash_parts(ENTRY_TAG,"key",prompt,mode,seed,height,width,table_key,level,k).hex()[:32]` (DAT-05 key 정규식 만족, **index와 무관**).
- `to_uint8_rgb(img: Tensor[B,3,H,W] in [-1,1])->ndarray uint8[B,H,W,3]`: `clamp` 후 `floor((x+1)*127.5+0.5)`. ndim≠4·채널≠3·비유한은 `EntryGenError`.
- `@dataclass(frozen=True) EntryImage(key, index, prompt, caption, mode, seed, level, k, height, width, image, prov, table_key, nfe)`: `prompt`=strip한 원문, `caption`=`compose_prompt` 결과, `image`=uint8 `[H,W,3]` C-연속 **읽기 전용**, `prov`=`ProvRecord`, `nfe`=그 장을 만든 교사 forward 수(= `len(sampler.table)`). 메서드: `png_bytes()`(RGB PNG, 무손실·결정적), `image_f32()`(`uint8/255` float32 `[H,W,3]`; RealifyItem.image 자리), `meta()->dict`(키 정확히 `v,key,index,prompt,caption,mode,seed,level,k,height,width,nfe,table_key,prov(=prov.to_dict())`, JSON 직렬화 가능, `v=ENTRY_TAG`), `shard_fields()->{"rgb.png": png_bytes(), "meta.json": json.dumps(meta, sort_keys, separators=(",",":"), ensure_ascii=False).encode()}`.
- `generate_entries(sampler, prompts, *, seed=0, height=512, width=512, mode="survey", batch_size=4, level=0, k=0, banned=(), taint=DEFAULT_TAINT)->list[EntryImage]`. `seed`는 int(-> 장마다 `entry_seed(seed, i)`) 또는 길이 N 시퀀스(그대로 사용; 길이 불일치 `ValueError`). 프롬프트를 `batch_size`개씩 연속 청크로 나눠 청크마다 `sampler.generate(captions, seeds=, height=, width=, level=, k=)` 1회 -> `teacher.decode_latent(res.z)` -> `to_uint8_rgb`. 스텝은 항상 표 전체(**기본 8 NFE**, 표를 바꾸면 그 길이). `taint`는 `ProvRecord(taint=frozenset(taint), mode="none")`로 만들고 `ext`·`comp`·`hold` 비트가 서면 `EntryGenError`(입구 생성 출력은 외부 계보가 아니다). 빈 `prompts`는 `[]`.
- `write_entries(entries, out_dir)->list[Path]`: `out_dir`을 만들고 장마다 `{key}.png`(=`rgb.png`)와 `{key}.json`(=`meta.json`)을 tmp+replace로 쓴다. 반환은 PNG 경로(입력 순서). 같은 내용 재기록은 멱등. 빈 입력 `[]`.

## 불변식
1. **검증은 교사 forward 전에 끝난다**: 금칙·빈 프롬프트·mode·크기(16의 배수, `spec.check_image_size`)·batch_size≥1·seed·taint 위반은 어느 청크에서 발견되든 변환기 호출 0, 텍스트 인코딩 0.
2. **배치 독립**: `batch_size`·청크 구성과 무관하게 장 i의 `key`·`image`는 비트 동일(TCH-02 lockstep). 시퀀스를 뒤집고 시드도 뒤집으면 결과도 뒤집힌다.
3. 같은 입력 -> 같은 `key`·픽셀·PNG 바이트. 시드·프롬프트·mode·level·k·표가 다르면 `key`가 다르다. 반복 프롬프트도 index가 다르면 시드·그림이 다르다.
4. `taint`는 이미지·키를 바꾸지 않는다. 기본 출처는 `ENGINE`(`train_ok`·`dev_ok`).
5. 전역 RNG·grad 모드·교사 파라미터를 건드리지 않는다. 가중치로 학습하지 않는다(CT-M5).

## 수용 기준
`tests/tch_11/` 전부 통과(CPU, 진짜 `load_teacher`+아핀 가짜 변환기·가짜 VAE, <60 s): 독립 오라클(`sampler.generate` -> `decode_latent` -> `to_uint8_rgb`)과 픽셀 비트 동일, 변환기 호출 정확히 8회·`t=1-σ`·인코딩 문장 = 합성 캡션, 배치 크기 1/2/5의 결과 동일과 호출 수(40 / 8+8+8 / 8), 결정성·시드 민감도·중복 프롬프트·순서 순수성, mode·level·k·표 변경 효과, 출처 기본/사용자 지정/ext 거부, 오류 우선 검증 전수, PNG 왕복·메타·샤드 필드·디스크 쓰기, 불변 객체, import 순수성(diffusers 등 미적재). `@pytest.mark.gpu`(**실행하지 못했음**, 5090 필요): 실물 512² 8 NFE가 상수/검정 이미지가 아니고 프롬프트마다 다르며 같은 GPU에서 재현(평균 차 ≤0.5), 단일 뷰와 배치의 차 ≤3 레벨.

## 하류 이음매
DAT-09·DAT-02는 `EntryImage.image_f32()`·`caption`·`prov`로 `RealifyItem`을 만들거나 `shard_fields()`를 DAT-05 `ShardWriter`에 그대로 넣는다(key는 정규식 통과). T1-E 재현은 `mode="cinematic"`, 시드는 호출자가 고정(선택 없음). 래칫·내부 라이브러리는 `mode="survey"`. 금칙 목록은 호출자가 `banned=`로 넘긴다.
