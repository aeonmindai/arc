#!/usr/bin/env bash
# Wait until both legs finish (or something dies). ONE ssh per 5 min.
while true; do
  sleep 300
  N=$(runcrate ssh arc-w53-paged -- 'grep -cE "^(W53_DONE|FAIL:|RUN [AB] FAILED|SERVER \[)" /root/w53.log' 2>/dev/null | tr -dc '0-9')
  if [ "${N:-0}" -gt 0 ]; then
    runcrate ssh arc-w53-paged -- 'sed -n "/===== 4 RUN A/,\$p" /root/w53.log | tail -c 5000' 2>/dev/null
    echo "W53_MILESTONE"
    break
  fi
done
