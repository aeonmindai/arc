#!/usr/bin/env bash
# Session status server — bootstrap FIRST THING after SSH works (session-3
# lesson: the Runcrate token TTL is ~8h; when it expires mid-session the
# driving agent loses ssh_execute but can still poll plain HTTP).
#
# What it does:
#   - snapshots session state into /srv/arcstatus/status.txt every 30s:
#     UTC time, GPU util/mem, tails of every log under /root/logs, results
#     inventory, and any STALL_* markers the stall sentinel dropped;
#   - serves /srv/arcstatus on 0.0.0.0:8899 (python3 http.server).
#
# Poll from anywhere:  curl -s http://<BOX_IP>:8899/status.txt
# Results tarballs copied into /srv/arcstatus/ are downloadable the same way
# (session-3 used exactly this as the results escape hatch).
#
# PID discipline (session-2 lesson — kill PIDs, never patterns): both loop and
# server PIDs land in /srv/arcstatus/pids; teardown is
#   kill $(cat /srv/arcstatus/pids) 2>/dev/null
#
# Usage: nohup bash status_server.sh > /root/logs/status_server.log 2>&1 &
set -u
STATUS_DIR=/srv/arcstatus
LOG_DIR=${LOG_DIR:-/root/logs}
RESULTS_DIR=${RESULTS_DIR:-}
PORT=${STATUS_PORT:-8899}

mkdir -p "$STATUS_DIR" "$LOG_DIR"
: > "$STATUS_DIR/pids"

snapshot() {
  {
    echo "== arc session status @ $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
    echo
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
      || echo "nvidia-smi unavailable"
    echo
    for f in "$LOG_DIR"/*.log; do
      [ -e "$f" ] || continue
      echo "--- tail -3 $f ($(wc -c < "$f") bytes) ---"
      tail -3 "$f"
      echo
    done
    if [ -n "$RESULTS_DIR" ] && [ -d "$RESULTS_DIR" ]; then
      echo "--- results inventory ---"
      ls -la "$RESULTS_DIR"
      echo
    fi
    if ls "$STATUS_DIR"/STALL_* >/dev/null 2>&1; then
      echo "!!! STALL MARKERS PRESENT !!!"
      for m in "$STATUS_DIR"/STALL_*; do
        echo "$m: $(cat "$m")"
      done
    fi
  } > "$STATUS_DIR/status.txt.tmp" 2>&1
  mv "$STATUS_DIR/status.txt.tmp" "$STATUS_DIR/status.txt"
}

# HTTP server (background, PID recorded)
cd "$STATUS_DIR"
python3 -m http.server "$PORT" --bind 0.0.0.0 >/dev/null 2>&1 &
HTTP_PID=$!
echo "$HTTP_PID" >> "$STATUS_DIR/pids"

# Snapshot loop (this process, PID recorded)
echo "$$" >> "$STATUS_DIR/pids"
echo "status_server up: http://0.0.0.0:$PORT/status.txt (http pid $HTTP_PID, loop pid $$)"
while true; do
  snapshot
  sleep 30
done
