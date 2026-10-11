#!/usr/bin/env bash
# day1 원격 잡 슬롯(runner.sh가 md5가 바뀌면 받아 실행). 출력은 marks 필터를 거치므로 진단 줄은 D1_X 로 시작한다.
# 1) M2 진단: 마커가 한 줄도 없다(01:27Z 이후). 프로세스·로그·스택을 본다.
# 2) 파이프라인 첫 야크(pipe_yak.py)를 백그라운드로 띄운다(한 번만, PIPE_RUN 이 바뀌면 다시).
set -u
R=${N1_ROOT:-/workspace/night1}; cd "$R" || exit 0
x() { while IFS= read -r l; do echo "D1_X $1 ${l:0:360}"; done; }

echo "D1_X diag $(date -u +%FT%TZ)"
ps -eo pid,ppid,etime,stat,pcpu,rss,args | grep -E '[m]2_latent|[m]13_rigid|[m]6_tlpips|[p]ipe_yak' | x ps
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | x gpu
ls -la day1/m2 2>&1 | head -12 | x m2dir
echo "lines=$(wc -l < logs/m2.run.log 2>/dev/null)" | x m2log
tail -n 12 logs/m2.run.log 2>/dev/null | x m2tail
P=$(ps -eo pid,args | grep -E '[p]ython -u code/m2_latent_lift' | awk '{print $1}' | head -1)
if [ -n "$P" ]; then
  pip install -q --no-cache-dir py-spy >/dev/null 2>&1
  timeout 60 py-spy dump --pid "$P" 2>&1 | grep -vE '^\s*$' | head -40 | x m2stack
fi
python -c "import transformers; print('transformers', transformers.__version__)" 2>&1 | tail -1 | x deps

# --- 파이프라인 첫 야크 ---
PIPE_RUN=1
if [ ! -f "$R/.pipe_launched_$PIPE_RUN" ]; then
  touch "$R/.pipe_launched_$PIPE_RUN"
  python -c "import transformers" 2>/dev/null || pip install -q --no-cache-dir transformers 2>&1 | tail -1 | x pip
  python - "$R/repo_pipe" <<'PY' 2>&1 | x repo
import io, sys, tarfile, urllib.request, shutil, os
url = "https://codeload.github.com/gkjuwon-tech/Video/tar.gz/refs/heads/claude/video-generation-new-paradigm-qriszj"
data = urllib.request.urlopen(url, timeout=120).read()
dst = sys.argv[1]; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
    top = t.getmembers()[0].name.split("/")[0]
    for m in t.getmembers():
        if not m.name.startswith(top + "/") or m.name == top: continue
        m.name = m.name[len(top) + 1:]
        t.extract(m, dst, filter="data")
print("ok", os.path.isfile(os.path.join(dst, "ops/gpu/day1/pipe_yak.py")))
PY
  # 로그 출구: 이 스크립트를 띄운 슬롯 서브셸의 표준출력(= 컨테이너 로그). 슬롯이 막히지 않게 모든 fd를 떼고 띄운다.
  OUT=/proc/$PPID/fd/1
  [ -w "$OUT" ] || OUT=/dev/null
  setsid nohup bash -c '
    cd "$1" && PYTHONPATH="$1/repo_pipe:$1/code" python -u "$1/repo_pipe/ops/gpu/day1/pipe_yak.py" --root "$1" > "$1/logs/pipe.log" 2>&1
    echo "D1_PIPE_EXIT rc=$? $(date -u +%FT%TZ)" >> "$1/logs/pipe.log"
  ' _ "$R" </dev/null >/dev/null 2>&1 &
  setsid nohup bash -c '
    sleep 2; tail -n +1 -F "$1/logs/pipe.log" 2>/dev/null | grep --line-buffered -E "^D1_PIPE|Traceback" | awk "{ print substr(\$0, 1, 400); fflush() }" > "$2"
  ' _ "$R" "$OUT" </dev/null >/dev/null 2>&1 &
  echo "D1_X pipe launched out=$OUT"
fi
echo "D1_X end $(date -u +%FT%TZ)"
