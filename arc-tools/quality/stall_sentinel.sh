#!/usr/bin/env bash
# Stall sentinel — watches one log file for growth; a session-2 lesson turned
# into a tool (the step-6→7 chain once sat 30 minutes behind a zombie
# wait-loop before anyone noticed).
#
#   stall_sentinel.sh <logfile> <stall_secs> [pid_to_kill]
#
# If <logfile> does not grow for <stall_secs> seconds:
#   - drops STALL_<basename> into /srv/arcstatus (picked up by
#     status_server.sh's snapshot, so the driving agent sees it on next poll);
#   - if [pid_to_kill] is given: SIGTERM that EXACT PID, escalate to SIGKILL
#     after 30s. PID-kills only — never a pkill pattern (session-2 lesson:
#     pattern kills took out the wrong generation of the chain).
#
# Exits after firing once, or silently when pid_to_kill exits on its own.
# One sentinel per watched step; start it right after you capture the step's
# PID with $!.
#
# Usage: nohup bash stall_sentinel.sh /root/logs/bake4.log 900 $BAKE_PID \
#          > /root/logs/sentinel_bake.log 2>&1 &
set -u
LOGFILE=${1:?usage: stall_sentinel.sh <logfile> <stall_secs> [pid_to_kill]}
STALL_SECS=${2:?usage: stall_sentinel.sh <logfile> <stall_secs> [pid_to_kill]}
KILL_PID=${3:-}
STATUS_DIR=/srv/arcstatus
MARKER="$STATUS_DIR/STALL_$(basename "$LOGFILE")"
mkdir -p "$STATUS_DIR"

size_of() { wc -c < "$LOGFILE" 2>/dev/null || echo 0; }

LAST_SIZE=$(size_of)
LAST_CHANGE=$(date +%s)
echo "sentinel: watching $LOGFILE (stall=${STALL_SECS}s, pid=${KILL_PID:-none})"
while true; do
  sleep 30
  # Watched process already gone -> nothing to guard.
  if [ -n "$KILL_PID" ] && ! kill -0 "$KILL_PID" 2>/dev/null; then
    echo "sentinel: pid $KILL_PID exited; standing down"
    exit 0
  fi
  NOW=$(date +%s)
  SIZE=$(size_of)
  if [ "$SIZE" != "$LAST_SIZE" ]; then
    LAST_SIZE=$SIZE
    LAST_CHANGE=$NOW
    continue
  fi
  if [ $((NOW - LAST_CHANGE)) -ge "$STALL_SECS" ]; then
    MSG="$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOGFILE frozen ${STALL_SECS}s at ${SIZE} bytes (pid ${KILL_PID:-none})"
    echo "$MSG" > "$MARKER"
    echo "sentinel: STALL — $MSG"
    if [ -n "$KILL_PID" ]; then
      kill "$KILL_PID" 2>/dev/null
      sleep 30
      kill -0 "$KILL_PID" 2>/dev/null && kill -9 "$KILL_PID" 2>/dev/null
      echo "sentinel: killed pid $KILL_PID (TERM then KILL)"
    fi
    exit 1
  fi
done
