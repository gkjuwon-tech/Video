#!/usr/bin/env bash
# 03:45Z 창업자 지시: 전부 정지(사용량 99%, 월요일 21:00 KST 재개), 돈 안 나가게.
# M6 라벨(작음)·manifest·체크포인트·로그를 HF 금고에 백업한 뒤 인스턴스를 스스로 정지한다. 오케스트레이터가 확인 후 파기한다.
set -u
R=${N1_ROOT:-/workspace/night1}; cd "$R" || exit 0
x() { while IFS= read -r l; do echo "D1_X $1 ${l:0:360}"; done; }
[ -f "$R/.final_stop_1" ] && exit 0
touch "$R/.final_stop_1"
echo "D1_X final_stop begin $(date -u +%FT%TZ)"
for p in $(ps -eo pid,args | grep -E '[p]ython -u (code/m6_tlpips|code/m3_mrh|.*m14_glyph)' | awk '{print $1}'); do kill -TERM "$p" 2>/dev/null; done
sleep 30
python - "$R" <<'PY' 2>&1 | x backup
import os, sys, glob
from huggingface_hub import HfApi
R = sys.argv[1]; tok = os.environ.get("HF_TOKEN")
api = HfApi(token=tok)
m6 = os.path.join(R, "day1", "m6")
files = sorted(glob.glob(os.path.join(m6, "labels_*.npz")) + glob.glob(os.path.join(m6, "*.json")) + glob.glob(os.path.join(m6, "*.jsonl")) + glob.glob(os.path.join(m6, "ckpt_*.pt")))
files += sorted(glob.glob(os.path.join(R, "logs", "*.log")))
n = 0
for i in range(0, len(files), 50):
    from huggingface_hub import CommitOperationAdd
    ops = []
    for p in files[i:i + 50]:
        rel = os.path.relpath(p, R)
        ops.append(CommitOperationAdd(path_in_repo="backup/day1_stop/" + rel, path_or_fileobj=p))
    try:
        api.create_commit(repo_id="sdsdsdsdsdsdsd123/buc-artifacts", repo_type="model", operations=ops, commit_message="day1 stop backup")
        n += len(ops)
    except Exception as e:
        print("fail", type(e).__name__, str(e)[:200])
print("ok files", n, "of", len(files))
PY
echo "D1_X final_stop backup_done $(date -u +%FT%TZ)"
[ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1 | tail -1 | x stop
