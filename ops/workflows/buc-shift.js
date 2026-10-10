export const meta = {
  name: 'buc-shift',
  description: 'BUC harness shift: pick ready units, contract+frozen tests (Sonnet/Opus), implement (Haiku/Sonnet), independent gate, repair ladder, Opus review for risky units, commit',
  phases: [
    { title: 'Pick', detail: 'Haiku runs the picker' },
    { title: 'Contract', detail: 'Sonnet high (Opus medium for diff/risk ≥4) writes contract + tests, freezes hashes' },
    { title: 'Build', detail: 'implement with backlog impl_model' },
    { title: 'Gate', detail: 'Haiku runs gate.py; repair Sonnet → Opus' },
    { title: 'Review', detail: 'Opus low for risk ≥4' },
    { title: 'Commit', detail: 'Haiku: update backlog, heartbeat, persist' },
  ],
}

const R = '/home/user/Video'
const BATCHES = (args && args.batches) || 4
const K = (args && args.k) || 2

const RULES = `[BUC 하네스 공통 규칙] 저장소: ${R} (Block Universe Cinema: 사진 한 장으로 3D 세계를 짓고 물리 카메라로 찍는 I2V. 파이썬 패키지 buc/). 먼저 ${R}/ops/harness/README.md 를 읽어라.
- 너는 이 작업 단위의 소유 경로, tests/<id 소문자_밑줄>/, ops/harness/contracts/<ID>.md 만 만들거나 고친다. 다른 파일은 건드리지 않는다(필요하면 결과에 요청으로 적는다). git commit/push 하지 마라.
- 사양 출처: ${R}/work/wf5/r-inventory.md (단위 ID로 grep해서 해당 행·설명만 읽어라), ${R}/docs/PARADIGM-v2.md, ${R}/work/wf3/lens/pipeline.upg.md·models.upg.md (I2V v3), ${R}/work/wf3/inputs/07-founder-decision-i2v.md, 의존 단위의 ${R}/ops/harness/contracts/*.md 와 기존 buc/ 코드.
- 실행 환경: CPU 전용 컨테이너(python3, torch CPU, numpy, scipy, pillow, pytest). GPU 코드는 CPU에서도 import 가능해야 하고, GPU가 필요한 테스트는 @pytest.mark.gpu 로 표시한다.
- 비밀 값(키·토큰)을 파일·출력에 쓰지 마라. 성공을 지어내지 마라. 모르면 모른다고 적어라.`

const PICK = { type: 'object', properties: { items: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, name: { type: 'string' }, paths: { type: 'array', items: { type: 'string' } }, deps: { type: 'array', items: { type: 'string' } },
  diff: { type: 'number' }, risk: { type: 'number' }, verify: { type: 'string' }, impl_model: { type: 'string' }, impl_effort: { type: 'string' },
  review: { type: 'string' }, slice: { type: 'boolean' }, why: { type: 'string' } }, required: ['id', 'name', 'paths', 'diff', 'risk', 'impl_model', 'impl_effort', 'review'] } },
  ready: { type: 'number' }, done: { type: 'number' }, total: { type: 'number' } }, required: ['items'] }
const DONE = { type: 'object', properties: { ok: { type: 'boolean' }, files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' }, requests: { type: 'string' } }, required: ['ok', 'notes'] }
const GATE = { type: 'object', properties: { pass: { type: 'boolean' }, raw: { type: 'string' } }, required: ['pass', 'raw'] }
const REVIEW = { type: 'object', properties: { verdict: { type: 'string', enum: ['pass', 'fix'] }, issues: { type: 'array', items: { type: 'string' } } }, required: ['verdict', 'issues'] }

const tdir = (id) => `tests/${id.toLowerCase().replace(/-/g, '_')}`
const owned = (it) => `소유 경로: ${it.paths.join(', ')} / 테스트: ${tdir(it.id)}/ / 계약서: ops/harness/contracts/${it.id}.md`
const head = (it) => `[단위] ${it.id} ${it.name} (난이도 ${it.diff}, 위험 ${it.risk}, 검증 ${it.verify || '-'}, 수직 슬라이스 ${it.slice ? '예' : '아니오'})\n${owned(it)}\n배정 이유: ${it.why || ''}`

async function gate(it, label) {
  return agent(`${RULES}\n${head(it)}\n\n명령 하나만 실행하라: cd ${R} && python3 ops/harness/gate.py ${it.id} ; 그 JSON 출력 전체를 raw에, 출력의 "pass" 값을 pass에 넣어 반환하라. 아무것도 고치지 마라.`,
    { label: `gate:${it.id}:${label}`, phase: 'Gate', model: 'haiku', effort: 'low', schema: GATE })
}

async function repair(it, model, effort, why, label) {
  return agent(`${RULES}\n${head(it)}\n\n게이트가 실패했다. 계약서(ops/harness/contracts/${it.id}.md)와 동결 테스트(${tdir(it.id)}/, 수정 금지)를 기준으로 구현만 고쳐라. 고친 뒤 직접 \`python3 ops/harness/gate.py ${it.id}\` 를 돌려 통과를 확인하라. 테스트 자체가 틀렸다고 판단되면 고치지 말고 ok=false와 이유를 적어라.\n\n<gate_output>\n${(why || '').slice(0, 6000)}\n</gate_output>`,
    { label: `repair:${it.id}:${label}`, phase: 'Gate', model, effort, schema: DONE })
}

// 절약 모드(기본 켬, 2026-10-10 7일 사용량 경고): Opus는 난이도·위험 5 단위의 계약서와 검토에만 쓴다.
// 2차 수리는 Sonnet이 다른 각도로 한 번 더. args.economy === false 이면 원래 배정(≥4 Opus)으로 돌아간다.
const ECO = !(args && args.economy === false)

async function runUnit(it) {
  const upper = ECO ? (it.diff >= 5 || it.risk >= 5 || it.impl_model === 'opus') : (it.diff >= 4 || it.risk >= 4 || it.impl_model === 'opus')
  const used = []
  // 1) 계약서 + 테스트 (구현자와 다른 에이전트)
  const cm = upper ? 'opus' : 'sonnet'
  const c = await agent(`${RULES}\n${head(it)}\n\n너는 **계약서·테스트 작성자**다. 구현은 하지 마라.
1. 사양 출처에서 이 단위의 요구를 찾아 읽고, 의존 단위 계약서·코드를 확인하라.
2. ops/harness/contracts/${it.id}.md 에 계약서를 써라: 목적, 공개 API(함수·클래스 시그니처와 타입), 데이터 형태, 불변식, 소유 파일, 수용 기준, 하류가 기대할 이음매. 간결하게(2–5KB).
3. ${tdir(it.id)}/ 에 pytest 테스트를 써라: CPU에서 결정론적으로 60초 안에 끝나고, 수용 기준을 실제로 검증한다(수치는 허용 오차와 함께). 사소한 존재 확인만으로 채우지 마라. GPU 필요 테스트는 @pytest.mark.gpu.
4. \`python3 ops/harness/gate.py --freeze ${it.id}\` 를 실행해 테스트 해시를 동결하라.
ok=true는 계약서와 테스트를 쓰고 동결까지 했을 때만.`,
    { label: `contract:${it.id}`, phase: 'Contract', model: cm, effort: upper ? 'medium' : 'high', schema: DONE })
  used.push(`contract:${cm}`)
  if (!c || !c.ok) return { id: it.id, status: 'blocked', models: used.join(','), note: `계약 실패: ${c ? c.notes : 'agent null'}` }
  // 2) 구현 (하위)
  const b = await agent(`${RULES}\n${head(it)}\n\n너는 **구현자**다. 계약서 ops/harness/contracts/${it.id}.md 와 동결 테스트 ${tdir(it.id)}/ 를 읽고 소유 경로에 구현하라. 테스트·계약서는 수정 금지. 다 쓰면 \`python3 ops/harness/gate.py ${it.id}\` 로 통과를 확인하고, 실패하면 고쳐서 다시 돌려라(최대 4회). 하류가 쓸 수 있게 공개 API는 계약서와 정확히 일치시켜라.`,
    { label: `build:${it.id}`, phase: 'Build', model: it.impl_model, effort: it.impl_effort, schema: DONE })
  used.push(`build:${it.impl_model}`)
  // 3) 독립 게이트 + 수리 사다리
  let g = await gate(it, 'g1')
  if (!g || !g.pass) {
    const r1 = await repair(it, 'sonnet', 'high', g && g.raw, 'r1'); used.push('repair:sonnet')
    g = await gate(it, 'g2')
    if (!g || !g.pass) {
      const r2m = ECO ? 'sonnet' : 'opus'
      const r2 = await repair(it, r2m, 'high', `${g && g.raw}\n(2차 수리: 1차 수리가 실패했다. 가정을 처음부터 다시 점검하라.)`, 'r2'); used.push(`repair:${r2m}`)
      g = await gate(it, 'g3')
      if (!g || !g.pass) return { id: it.id, status: 'blocked', models: used.join(','), note: `게이트 3회 실패: ${(g && g.raw || '').slice(0, 400)} ${r2 ? r2.notes : ''}` }
    }
  }
  // 4) 위험 단위 검토 (상위)
  const rvm = ECO && it.risk < 5 && it.review !== 'F' ? 'sonnet' : 'opus'
  if (it.risk >= 4 || it.review === 'F') {
    const rv = await agent(`${RULES}\n${head(it)}\n\n너는 **검토자**다. 계약서, 동결 테스트, 구현(git diff 및 소유 파일)을 읽고 판정하라: 계약서의 수용 기준을 실제로 만족하는가, 테스트가 수용 기준을 제대로 검증하는가, 하류 이음매(공개 API·데이터 형태)가 사양과 맞는가, 명백한 버그·비결정성·비밀 노출·소유 경로 밖 수정이 있는가. 사소한 스타일은 무시. 고쳐야 할 것이 있으면 verdict=fix와 구체적 지시 목록.`,
      { label: `review:${it.id}`, phase: 'Review', model: rvm, effort: rvm === 'opus' ? 'low' : 'medium', schema: REVIEW })
    used.push(`review:${rvm}`)
    if (rv && rv.verdict === 'fix' && rv.issues.length) {
      await repair(it, 'sonnet', 'high', `검토 지적:\n- ${rv.issues.join('\n- ')}`, 'rv'); used.push('repair:sonnet')
      g = await gate(it, 'g4')
      if (!g || !g.pass) return { id: it.id, status: 'blocked', models: used.join(','), note: `검토 수정 후 게이트 실패: ${(g && g.raw || '').slice(0, 400)}` }
    }
  }
  return { id: it.id, status: 'done', models: used.join(','), note: (b && b.requests) ? `요청: ${b.requests}`.slice(0, 400) : '' }
}

const summary = []
phase('Pick')
await agent(`명령 하나만 실행하라: cd ${R} && python3 ops/harness/pick.py --k 0 --reset-doing ; 출력 JSON을 그대로 items(빈 배열)와 함께 반환하라.`, { label: 'reset', phase: 'Pick', model: 'haiku', effort: 'low', schema: PICK })
for (let bi = 0; bi < BATCHES; bi++) {
  const p = await agent(`명령 하나만 실행하라: cd ${R} && python3 ops/harness/pick.py --k ${K} --claim ; 출력 JSON을 그대로 구조화해 반환하라(items의 모든 필드를 빠짐없이).`,
    { label: `pick:${bi + 1}`, phase: 'Pick', model: 'haiku', effort: 'low', schema: PICK })
  if (!p || !p.items || !p.items.length) { log(`batch ${bi + 1}: ready 단위 없음, 교대조 종료`); break }
  log(`batch ${bi + 1}/${BATCHES}: ${p.items.map(x => x.id).join(', ')} (완료 ${p.done}/${p.total}, 대기 ${p.ready})`)
  const res = (await parallel(p.items.map(it => () => runUnit(it)))).map((r, i) => r || { id: p.items[i].id, status: 'blocked', models: '', note: 'unit pipeline crashed' })
  summary.push(...res)
  phase('Commit')
  const lines = res.map(r => `python3 ops/harness/update.py ${r.id} ${r.status} --models '${r.models}' --note ${JSON.stringify(r.note || '').replace(/'/g, '')}`).join(' && ')
  await agent(`다음 명령들을 저장소 ${R} 에서 순서대로 실행하라. 실패해도 다음으로 진행하고 결과를 notes에 요약하라.
1) ${lines}
2) git status --porcelain 을 보고, 이번 배치 단위(${res.map(r => r.id).join(', ')})의 소유 경로·tests·contracts 와 ops/harness/backlog.json, ops/harness/ledger.jsonl, ops/gpu/heartbeat.json 이 아닌 변경 파일이 있으면 그 목록을 notes에 적고 \`git checkout -- <파일>\`(추적 파일) 또는 \`rm\`(새 파일)로 되돌려라. 단 work/, ops/harness/contracts/ 는 건드리지 마라.
3) python3 -c "import json,time; json.dump({'epoch': int(time.time()), 'by': 'shift'}, open('ops/gpu/heartbeat.json','w'))"
4) ops/persist.sh "shift: ${res.map(r => `${r.id}=${r.status}`).join(' ')}"`,
    { label: `commit:${bi + 1}`, phase: 'Commit', model: 'haiku', effort: 'low', schema: DONE })
}
return { units: summary, done: summary.filter(s => s.status === 'done').length, blocked: summary.filter(s => s.status === 'blocked').map(s => `${s.id}: ${s.note}`) }
