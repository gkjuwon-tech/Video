#!/usr/bin/env bash
# night1 러너. GPU 서버의 상주 루프(ops/gpu/e0_bringup.sh)가 ops/gpu/runner.sh 를 받아 exec 한다(배포 = 그 경로로 복사해 푸시).
# 순서: 지정 인스턴스 확인 → 의존성 → 바닥선 영상(짧게) → 코퍼스 생성 + TAE-T 학습(동시, 마감까지, 죽으면 재시작) → 로그 업로드 → 스스로 정지.
# 비밀 값은 출력하지 않는다(HF_TOKEN은 있음/없음만).
set -u
BASE='https://raw.githubusercontent.com/gkjuwon-tech/Video/refs/heads/claude/video-generation-new-paradigm-qriszj'
export N1_ROOT=${N1_ROOT:-/workspace/night1}
export N1_DEADLINE=${N1_DEADLINE:-1791672300}   # 2026-10-10T22:45:00Z = 07:45 KST
HARD_STOP=$((N1_DEADLINE + 1800))                # 마감 30분 뒤에는 무조건 정지
IDLE_MAX=${IDLE_MAX:-9000}
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
mkdir -p "$N1_ROOT/code" "$N1_ROOT/logs"; cd "$N1_ROOT" || exit 1

fetch() { python -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen(sys.argv[1]+'?t='+sys.argv[2], timeout=30).read().decode())" "$BASE/$1" "$(date +%s)"; }
self_stop() {
  echo "N1_STOP $(date -u +%FT%TZ) reason=$1"
  [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" || true
  sleep 60; exit 0
}
# 마커 줄만 표준출력으로 보낸다(vastai logs가 줄을 약 489자에서 자르므로 400자로 자른다. cut은 파이프에서 4KB씩 버퍼링해 마커가 몇 시간씩 안 보이므로 awk fflush를 쓴다). 전체 로그는 logs/ 에.
marks() { grep --line-buffered -E '^N1_|Traceback|Error|error|Killed' | awk '{ print substr($0, 1, 400); fflush() }'; }

echo "N1_START $(date -u +%FT%TZ) id=${CONTAINER_ID:-?} hf_token=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no) deadline=$N1_DEADLINE"

# 지정 인스턴스가 아니면 일하지 않는다(교체 전 인스턴스가 늦게 살아나 같은 잡을 두 번 돌리는 것을 막는다). 원래 데드맨 루프로 돌아간다.
WANT=$(fetch ops/gpu/instance.json 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
if [ -n "$WANT" ] && [ -n "${CONTAINER_ID:-}" ] && [ "$WANT" != "$CONTAINER_ID" ]; then
  echo "N1_NOT_DESIGNATED want=$WANT me=$CONTAINER_ID"
  while true; do
    hb=$(fetch ops/gpu/heartbeat.json 2>/dev/null | python -c "import json,sys; print(int(json.load(sys.stdin).get('epoch',0)))" 2>/dev/null || echo 0)
    [ $(( $(date +%s) - hb )) -gt "$IDLE_MAX" ] && self_stop heartbeat_stale
    sleep 300
  done
fi

# 마감 감시: 무슨 일이 있어도 HARD_STOP에 정지.
( while true; do [ "$(date +%s)" -gt "$HARD_STOP" ] && self_stop hard_deadline; sleep 120; done ) &

pip install -q --no-cache-dir opencv-python-headless imageio imageio-ffmpeg 2>&1 | tail -1
for f in baseline_video.py corpus_gen.py tae_train.py; do
  fetch "ops/gpu/night1/$f" > "code/$f" && [ -s "code/$f" ] && echo "N1_FETCH ok $f" || echo "N1_FETCH_FAIL $f"
done

# 1) 바닥선 영상: 짧은 잡. 최대 40분. 실패해도 다음으로.
timeout 2400 python -u code/baseline_video.py --root "$N1_ROOT" 2>&1 | tee -a logs/video.log | marks
echo "N1_VIDEO_EXIT $(date -u +%FT%TZ)"

# 2) 코퍼스 생성과 TAE-T 학습을 동시에. 둘 다 재개 가능하므로 죽으면 마감 전까지 다시 띄운다(최대 20회).
keep() {  # keep NAME CMD...
  local name=$1; shift; local n=0
  while [ "$(date +%s)" -lt $((N1_DEADLINE - 300)) ] && [ $n -lt 20 ]; do
    "$@" 2>&1 | tee -a "logs/$name.log" | marks
    n=$((n + 1)); echo "N1_RESTART $name n=$n $(date -u +%FT%TZ)"; sleep 30
  done
}
keep corpus python -u code/corpus_gen.py --root "$N1_ROOT" &
GEN=$!
sleep 60
keep tae python -u code/tae_train.py --root "$N1_ROOT" --finish-margin-s 600 --micro-batch 16 --mem-gb 4.5 &
TAE=$!
wait $GEN; wait $TAE

# 3) 로그 업로드 후 정지.
python - <<'PY' 2>&1 | marks
import os, glob
tok = os.environ.get("HF_TOKEN")
if not tok:
    print("N1_LOGS skip hf_token=no")
else:
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    for p in glob.glob(os.path.join(os.environ["N1_ROOT"], "logs", "*.log")):
        try:
            api.upload_file(path_or_fileobj=p, path_in_repo="logs/night1/" + os.path.basename(p),
                            repo_id="sdsdsdsdsdsdsd123/buc-artifacts", repo_type="model")
            print("N1_LOGS up", os.path.basename(p))
        except Exception as e:
            print("N1_LOGS fail", os.path.basename(p), type(e).__name__)
PY
self_stop night1_done
