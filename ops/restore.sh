#!/usr/bin/env bash
# 새 세션(또는 리셋된 컨테이너)에서 레포 work/의 작업물을 이 세션의 스크래치패드로 되돌린다.
# 사용: ops/restore.sh [스크래치패드 경로]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SP="${1:-${BUC_SCRATCH:-$(ls -d /tmp/claude-*/*/*/scratchpad 2>/dev/null | head -1)}}"
[ -n "$SP" ] || { echo "restore: 스크래치패드 경로를 인자로 주세요" >&2; exit 1; }
mkdir -p "$SP"
for d in "$REPO"/work/*/; do
  name="$(basename "$d")"
  mkdir -p "$SP/$name"
  (cd "$d" && find . -type f) | while read -r f; do
    mkdir -p "$SP/$name/$(dirname "$f")"
    sed -e "s#__SCRATCH__#$SP#g" -e "s#__REPO__#$REPO#g" "$d/$f" > "$SP/$name/$f"
  done
  echo "restore: work/$name → $SP/$name"
done
for f in "$REPO"/ops/workflows/*.js; do
  [ -f "$f" ] || continue
  sed -e "s#__SCRATCH__#$SP#g" -e "s#__REPO__#$REPO#g" "$f" > "$SP/$(basename "$f")"
  echo "restore: 워크플로 스크립트 $(basename "$f") → $SP (경로 치환 완료)"
done
