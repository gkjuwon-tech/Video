#!/usr/bin/env bash
# 밤사이 GPU 체인: 기동 보장(안 뜨면 교체) → E0 결과 회수 → 결과 커밋. 이후 GPU는 상주 루프가 받는 runner.sh 밤샘 잡으로 넘어간다.
# (STOP_AFTER_E0=1 이면 E0 뒤 GPU를 정지한다.)
# 사용: ops/gpu/night_chain.sh INSTANCE_ID LOG_DIR
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO" || exit 1
IID="$1"; W="$2"; mkdir -p "$W"
python3 ops/gpu/ensure_up.py "$IID" 50 25 3 || { echo "GPU 기동 실패"; ops/persist.sh "gpu: bring-up failed" >/dev/null 2>&1; exit 2; }
ID=$(python3 -c "import json;print(json.load(open('ops/gpu/instance.json'))['id'])")
echo "E0 대기: $ID"
ops/gpu/wait_e0.sh "$ID" "$W/e0_$ID.log" 80; RC=$?
# E0 결과 회수: 숫자는 레포에, 이미지는 스크래치에.
python3 - "$W/e0_$ID.log" "$W" <<'EOF'
import base64, json, re, sys
log = open(sys.argv[1], errors="replace").read() if __import__("os").path.exists(sys.argv[1]) else ""
m = re.findall(r"E0_RESULT (\{.*\})", log)
res = json.loads(m[-1]) if m else {"error": "E0_RESULT 없음", "tail": log[-1500:]}
json.dump(res, open("ops/gpu/e0_result.json", "w"), ensure_ascii=False, indent=1)
chunks = [d for _, d in sorted((int(i), d) for i, d in re.findall(r"E0_IMG (\d+) (\S+)", log))]
if chunks:
    open(sys.argv[2] + "/e0_soju.jpg", "wb").write(base64.b64decode("".join(chunks)))
print("E0 결과:", {k: res.get(k) for k in list(res)[:8]})
EOF
# GPU는 끄지 않는다: E0 뒤 상주 루프가 ops/gpu/runner.sh(밤샘 잡)를 받아 실행하고, 잡이 끝나면 스스로 정지한다.
# 러너가 없을 때만 정지하려면 STOP_AFTER_E0=1.
if [ "${STOP_AFTER_E0:-0}" = 1 ]; then
  vastai stop instance "$ID" --raw >/dev/null 2>&1 && echo "GPU 정지: $ID"
fi
python3 - "$ID" "${STOP_AFTER_E0:-0}" <<'PYEOF'
import json, sys, time
p = "ops/gpu/instance.json"
d = json.load(open(p)); d.update(id=int(sys.argv[1]), state="stopped_after_e0" if sys.argv[2] == "1" else "e0_done_runner_next", updated=int(time.time()))
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PYEOF
ops/persist.sh "gpu: E0 result (rc=$RC)" 2>&1 | tail -1
exit $RC
