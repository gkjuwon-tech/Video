#!/usr/bin/env bash
# 세션 작업물(스크래치패드)을 레포 work/로 옮기고 커밋·푸시한다. 언제 돌려도 안전하다(멱등).
# 사용: ops/persist.sh "커밋 메시지 요약"
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MSG="${1:-checkpoint}"
SP="${BUC_SCRATCH:-$(ls -d /tmp/claude-*/*/*/scratchpad 2>/dev/null | head -1)}"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"

if [ -n "$SP" ] && [ -d "$SP" ]; then
  for d in $(cd "$SP" && ls -d wf3 wf5 wf6 wf7 2>/dev/null); do
    [ -d "$SP/$d" ] || continue
    mkdir -p "$REPO/work/$d"
    # 큰 바이너리는 제외하고 텍스트 산출물만 옮긴다. 로컬 절대 경로는 자리표시자로 바꾼다.
    (cd "$SP/$d" && find . -type f \( -name '*.md' -o -name '*.json' -o -name '*.js' -o -name '*.txt' -o -name '*.csv' \) -size -5M) | while read -r f; do
      mkdir -p "$REPO/work/$d/$(dirname "$f")"
      sed -e "s#$SP#__SCRATCH__#g" -e "s#$REPO#__REPO__#g" "$SP/$d/$f" > "$REPO/work/$d/$f"
    done
  done
fi

cd "$REPO" || exit 1
git add -A 2>/dev/null
if git diff --cached --quiet; then echo "persist: 변경 없음"; exit 0; fi
# 비밀 검문소: 로컬 비밀 파일의 값이 커밋 내용에 섞여 있으면 중단한다(값은 레포에 두지 않고 실행 시 읽는다).
for secret_file in "$HOME/.config/vastai/vast_api_key" "$HOME/.cache/huggingface/token"; do
  [ -s "$secret_file" ] || continue
  if git diff --cached | grep -qF -- "$(cat "$secret_file")"; then
    echo "persist: 비밀 값이 커밋에 섞여 있어 중단합니다 ($secret_file)" >&2
    git reset -q; exit 2
  fi
done
git commit -q -F - <<EOF
ops: persist work state ($MSG)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DXDTH1gRJUrQo4xe26AmAr
EOF
for i in 1 2 3 4; do git pull --rebase --autostash -q origin "$BRANCH" 2>/dev/null; git push -u origin "$BRANCH" && exit 0; sleep $((2**i)); done
echo "persist: push 실패" >&2; exit 1
