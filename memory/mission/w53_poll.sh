#!/usr/bin/env bash
# One low-frequency poller. ONE runcrate ssh per 5 min, never batched.
for i in $(seq 1 40); do
  sleep 300
  OUT=$(runcrate ssh arc-w53-paged -- 'tail -c 1800 /root/w53.log' 2>/dev/null)
  echo "=== poll $i $(date -u +%H:%M) ==="
  echo "$OUT"
  case "$OUT" in
    *W53_DONE*) echo "PROBE_FINISHED"; break ;;
  esac
done
