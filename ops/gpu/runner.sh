#!/usr/bin/env bash
# day1 러너: GPU 상주 루프(ops/gpu/e0_bringup.sh)가 ops/gpu/runner.sh 로 받아 exec 한다.
# M13(장비 감식반)과 M2(엘리베이터)를 바로 학습하고, M6(눈썰미 자)는 레포에 ops/gpu/day1/M6_READY 가 생기면 받아서 시작한다.
# 마감 N1_DEADLINE(22:00 KST) 뒤 정리·업로드하고 스스로 정지한다. 비밀 값은 출력하지 않는다.
set -u
BASE='https://raw.githubusercontent.com/gkjuwon-tech/Video/refs/heads/claude/video-generation-new-paradigm-qriszj'
BRANCH='claude/video-generation-new-paradigm-qriszj'
export N1_ROOT=${N1_ROOT:-/workspace/night1}
export N1_DEADLINE=${N1_DEADLINE:-1791723600}   # 2026-10-11T13:00:00Z = 22:00 KST
HARD_STOP=$((N1_DEADLINE + 1800))
IDLE_MAX=${IDLE_MAX:-9000}
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$N1_ROOT/repo:$N1_ROOT/code:${PYTHONPATH:-}"
mkdir -p "$N1_ROOT/code" "$N1_ROOT/logs" "$N1_ROOT/repo"; cd "$N1_ROOT" || exit 1

fetch() { python -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen(sys.argv[1]+'?t='+sys.argv[2], timeout=30).read().decode())" "$BASE/$1" "$(date +%s)"; }
self_stop() {
  echo "D1_STOP $(date -u +%FT%TZ) reason=$1"
  [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" || true
  sleep 60; exit 0
}
# 마커만 표준출력으로(줄마다 즉시, 400자 자르기). 전체 로그는 logs/.
marks() { grep --line-buffered -E '^(D1_|N1_)|Traceback|Error|error|Killed' | awk '{ print substr($0, 1, 400); fflush() }'; }

echo "D1_START $(date -u +%FT%TZ) id=${CONTAINER_ID:-?} hf_token=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no) deadline=$N1_DEADLINE"

# 지정 인스턴스가 아니면 일하지 않는다(원래 데드맨 루프로).
WANT=$(fetch ops/gpu/instance.json 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
if [ -n "$WANT" ] && [ -n "${CONTAINER_ID:-}" ] && [ "$WANT" != "$CONTAINER_ID" ]; then
  echo "D1_NOT_DESIGNATED want=$WANT me=$CONTAINER_ID"
  while true; do
    hb=$(fetch ops/gpu/heartbeat.json 2>/dev/null | python -c "import json,sys; print(int(json.load(sys.stdin).get('epoch',0)))" 2>/dev/null || echo 0)
    [ $(( $(date +%s) - hb )) -gt "$IDLE_MAX" ] && self_stop heartbeat_stale
    sleep 300
  done
fi
( while true; do [ "$(date +%s)" -gt "$HARD_STOP" ] && self_stop hard_deadline; sleep 120; done ) &

# scipy: buc.camera.rigs(CAM-01)가 쓴다. 없으면 M13 데이터 생성이 전부 실패한다(D1 01:25Z 실제 발생).
pip install -q --no-cache-dir opencv-python-headless imageio imageio-ffmpeg scipy 2>&1 | tail -1
python -c "import scipy, cv2, imageio; print('D1_DEPS ok scipy', scipy.__version__)" || echo "D1_DEPS fail"
# 레포 코드(buc.camera.* 등): 브랜치 tarball을 repo/ 에 푼다.
python - "$BRANCH" "$N1_ROOT/repo" <<'PY' && echo "D1_REPO ok" || echo "D1_REPO fail"
import io, sys, tarfile, urllib.request, shutil, os
url = f"https://codeload.github.com/gkjuwon-tech/Video/tar.gz/refs/heads/{sys.argv[1]}"
data = urllib.request.urlopen(url, timeout=120).read()
dst = sys.argv[2]; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
    top = t.getmembers()[0].name.split("/")[0]
    for m in t.getmembers():
        if not m.name.startswith(top + "/") or m.name == top: continue
        m.name = m.name[len(top) + 1:]
        t.extract(m, dst, filter="data")
PY
for f in day1/m13_rigid.py day1/m2_latent_lift.py night1/tae_train.py; do
  fetch "ops/gpu/$f" > "code/$(basename "$f")" && [ -s "code/$(basename "$f")" ] && echo "D1_FETCH ok $f" || echo "D1_FETCH_FAIL $f"
done

keep() {  # keep NAME CMD... : 마감 5분 전까지 죽으면 다시 띄운다(최대 15회). 성공 판정은 이번 실행의 로그만 본다.
  local name=$1; shift; local n=0
  while [ "$(date +%s)" -lt $((N1_DEADLINE - 300)) ] && [ $n -lt 15 ]; do
    "$@" 2>&1 | tee "logs/$name.run.log" | marks
    cat "logs/$name.run.log" >> "logs/$name.log"
    [ -f "$N1_ROOT/day1/$name/DONE.json" ] && break
    if grep -qE "_DONE" "logs/$name.run.log" && ! grep -qE '"fatal"' "logs/$name.run.log"; then break; fi
    n=$((n + 1)); echo "D1_RESTART $name n=$n $(date -u +%FT%TZ)"; sleep 60
  done
}
keep m13 python -u code/m13_rigid.py --root "$N1_ROOT" --procs 8 &
P13=$!
keep m2 python -u code/m2_latent_lift.py --root "$N1_ROOT" --finish-margin-s 600 --mem-gb 2.9 &
P2=$!

# M6: 검토·수리가 끝나 레포에 M6_READY 가 생기면 시작.
(
  while [ "$(date +%s)" -lt $((N1_DEADLINE - 3600)) ]; do
    if fetch ops/gpu/day1/M6_READY >/dev/null 2>&1; then
      fetch ops/gpu/day1/m6_tlpips.py > code/m6_tlpips.py && [ -s code/m6_tlpips.py ] && echo "D1_FETCH ok m6" && \
        keep m6 python -u code/m6_tlpips.py --root "$N1_ROOT"
      break
    fi
    sleep 300
  done
) &
P6=$!

# 원격 잡 슬롯: 레포의 ops/gpu/day1/extra.sh 가 바뀌면 받아서 실행한다(재부팅 없이 잡 추가·수리). 본 잡이 다 끝나면 멈춘다.
(
  LAST=""
  while [ "$(date +%s)" -lt $((N1_DEADLINE - 300)) ]; do
    if fetch ops/gpu/day1/extra.sh > /tmp/extra.sh 2>/dev/null && [ -s /tmp/extra.sh ] && ! grep -q "404: Not Found" /tmp/extra.sh; then
      H=$(md5sum /tmp/extra.sh | cut -d' ' -f1)
      if [ "$H" != "$LAST" ]; then LAST=$H; echo "D1_EXTRA_RUN $H"; bash /tmp/extra.sh 2>&1 | tee -a logs/extra.log | marks; echo "D1_EXTRA_END $H"; fi
    fi
    [ -f "$N1_ROOT/.main_done" ] && break
    sleep 180
  done
) &
PX=$!
wait $P13; wait $P2; wait $P6
touch "$N1_ROOT/.main_done"; wait $PX

python - <<'PY' 2>&1 | marks
import os, glob
tok = os.environ.get("HF_TOKEN")
if tok:
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    for p in glob.glob(os.path.join(os.environ["N1_ROOT"], "logs", "m*.log")):
        try:
            api.upload_file(path_or_fileobj=p, path_in_repo="logs/day1/" + os.path.basename(p),
                            repo_id="sdsdsdsdsdsdsd123/buc-artifacts", repo_type="model")
            print("D1_LOGS up", os.path.basename(p))
        except Exception as e:
            print("D1_LOGS fail", os.path.basename(p), type(e).__name__)
PY
self_stop day1_done
