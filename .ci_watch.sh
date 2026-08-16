#!/bin/bash
for i in $(seq 1 50); do
  out=$(gh -R aeonmindai/arc pr checks 73 2>/dev/null)
  n_pending=$(echo "$out" | grep -c pending)
  if [ "$n_pending" = "0" ] && [ -n "$out" ]; then
    fails=$(echo "$out" | grep -Ev "	pass	")
    if [ -n "$fails" ]; then echo "FAILURES:"; echo "$fails"; else echo "ALL CHECKS PASS"; fi
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT waiting for CI"
