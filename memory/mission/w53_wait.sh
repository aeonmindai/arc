#!/usr/bin/env bash
# Wait until the probe reaches the first SERVE step or dies. ONE ssh per 5 min.
# Exits (one notification) as soon as a milestone lands.
while true; do
  sleep 300
  OUT=$(runcrate ssh arc-w53-paged -- 'grep -cE "^===== [456]" /root/w53.log; grep -cE "^(FAIL:|W53_DONE)" /root/w53.log' 2>/dev/null)
  N=$(echo "$OUT" | tr -d '\r' | paste -sd+ - | bc 2>/dev/null || echo 0)
  if [ "${N:-0}" -gt 0 ]; then
    runcrate ssh arc-w53-paged -- 'tail -c 2500 /root/w53.log' 2>/dev/null
    echo "MILESTONE_REACHED"
    break
  fi
done
