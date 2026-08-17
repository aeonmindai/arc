#!/usr/bin/env bash
# Wait for the probe to reach a milestone. ONE ssh per 5 min, unbuffered output.
for i in $(seq 1 40); do
  sleep 300
  OUT=$(runcrate ssh arc-w53-paged -- 'grep -E "^(=====|OK:|FAIL:|W53_DONE|SERVER|RUN [AB] FAILED|\[o(ff|n)\])" /root/w53.log | tail -25; echo "---NVIDIA---"; nvidia-smi --query-gpu=memory.used --format=csv,noheader' 2>/dev/null)
  printf '=== poll %s %s ===\n%s\n' "$i" "$(date -u +%H:%M)" "$OUT"
  case "$OUT" in
    *W53_DONE*) echo "PROBE_FINISHED"; break ;;
  esac
done
