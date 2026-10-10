#!/usr/bin/env bash
# E0 로그를 기다린다: E0_DONE 또는 오류 표시가 나오거나 시간 초과면 종료. 로그는 인자로 받은 파일에 저장.
ID="$1"; OUT="$2"; MAXMIN="${3:-75}"
end=$(( $(date +%s) + MAXMIN*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  vastai logs "$ID" --tail 400 > "$OUT.tmp" 2>&1 && mv "$OUT.tmp" "$OUT"
  if grep -q "E0_DONE" "$OUT" 2>/dev/null; then echo "E0_DONE 감지"; grep -E "E0_(START|ENV|DIFFUSERS|RESULT)" "$OUT"; exit 0; fi
  if grep -q -E "Traceback|CUDA error|out of memory|No space left" "$OUT" 2>/dev/null; then echo "오류 감지"; tail -40 "$OUT"; exit 1; fi
  sleep 60
done
echo "시간 초과"; tail -30 "$OUT"; exit 2
