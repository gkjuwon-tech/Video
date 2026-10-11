#!/usr/bin/env bash
# 비교용 GPU 부트 루프: ops/gpu/comp/ltx_yak.sh를 깃허브에서 받아 실행한다. 스크립트가 바뀌면(푸시) 다시 실행한다(원격 수리).
# 전부 끝나거나(.all_done) 부팅 후 5시간이 지나면 스스로 정지한다(과금 누수 방지).
set -u
BASE='https://raw.githubusercontent.com/gkjuwon-tech/Video/refs/heads/claude/video-generation-new-paradigm-qriszj'
T0=$(date +%s); LAST=""
python -m pip install -q --no-cache-dir vastai >/dev/null 2>&1
stop_me() {
  echo "C_STOP $(date -u +%FT%TZ) reason=$1"
  [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" || true
  sleep 120; exit 0
}
fetch() { python -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen(sys.argv[1]+'?t='+sys.argv[2], timeout=30).read().decode())" "$BASE/$1" "$(date +%s)"; }
while true; do
  [ -f /workspace/comp/.all_done ] && stop_me all_done
  [ $(( $(date +%s) - T0 )) -gt 18000 ] && stop_me max_5h
  if fetch ops/gpu/comp/ltx_yak.sh > /tmp/c.sh 2>/dev/null && [ -s /tmp/c.sh ]; then
    H=$(md5sum /tmp/c.sh | cut -d' ' -f1)
    if [ "$H" != "$LAST" ]; then LAST=$H; echo "C_RUN $H"; bash /tmp/c.sh; fi
  fi
  [ -f /workspace/comp/.all_done ] && stop_me all_done
  sleep 120
done
