#!/usr/bin/env bash
# 밤사이 GPU 체인: 기동 보장(안 뜨면 교체) → E0 결과 회수 → GPU 정지(유휴 과금 방지) → 결과 커밋.
# 정지된 인스턴스는 디스크(받아 둔 Z-Image 가중치 포함)만 과금된다. 다시 쓸 때 vastai start instance ID.
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
chunks = re.findall(r"E0_IMG (\S+)", log)
if chunks:
    open(sys.argv[2] + "/e0_soju.jpg", "wb").write(base64.b64decode("".join(chunks)))
print("E0 결과:", {k: res.get(k) for k in list(res)[:8]})
EOF
# 유휴 과금 방지: 할 일(러너)이 생길 때까지 정지.
vastai stop instance "$ID" --raw >/dev/null 2>&1 && echo "GPU 정지: $ID"
python3 - "$ID" <<'EOF'
import json, sys, time
p = "ops/gpu/instance.json"
d = json.load(open(p)); d.update(id=int(sys.argv[1]), state="stopped_after_e0", updated=int(time.time()))
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
EOF
ops/persist.sh "gpu: E0 result (rc=$RC), instance stopped" 2>&1 | tail -1
exit $RC
