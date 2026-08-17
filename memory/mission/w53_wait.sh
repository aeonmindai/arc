#!/usr/bin/env bash
# Final waiter: ONE runcrate ssh per 5 min, exits when the probe writes W53_DONE.
for i in $(seq 1 12); do
  sleep 300
  DONE=$(runcrate ssh arc-w53-paged -- 'grep -c W53_DONE /root/w53.log 2>/dev/null || echo 0' 2>/dev/null | tr -dc '0-9')
  if [ "${DONE:-0}" != "0" ]; then
    runcrate ssh arc-w53-paged -- 'cat /root/w53_result.txt 2>/dev/null; echo "=== gen_on ==="; cat /root/gen_on.json 2>/dev/null' 2>/dev/null
    echo "W53_DONE_READ"
    exit 0
  fi
done
echo "W53_TIMEOUT_60MIN"
