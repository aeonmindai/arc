#!/usr/bin/env bash
# Stall sentinel v2 — three-signal stall detection for a whole session.
#
# Jish, 2026-08-13: "another problem with running a robot in the machine, you
# don't know when shit stalls, so fucked".
#
# stall_sentinel.sh (v1) watches ONE log for growth and PID-kills it. That
# misses two real failure shapes seen on the boxes:
#   * a step that is genuinely working but writes nothing for 20 minutes (a
#     long cargo link, a bake between per-layer lines) — v1 would kill it;
#   * a chain that DIED between steps: no work processes, no new log bytes, and
#     nothing will ever move again — v1 stands down silently.
#
# v2 therefore requires THREE signals to agree before crying stall:
#
#   logs frozen  AND  GPU utilization < 5%  AND  work processes still alive
#       => STALL_DETECTED   (something is wedged, not busy)
#
#   no work processes  AND  the chain's completion marker is absent
#       => STALL_DEADCHAIN  (the chain fell over between steps)
#
# It never kills anything. Killing is v1's job (armed per step, with that
# step's exact PID). v2 only reports, into /srv/arcstatus/stall.txt and
# stall_latest.txt, which the status digest folds into the polled page.
#
# Usage:
#   nohup bash stall_sentinel2.sh /root/logs "S6_COMPLETE" /root/logs/s6.log \
#     > /root/logs/sentinel2.log 2>&1 &
#
#   $1  log DIRECTORY to watch (every *.log under it)
#   $2  completion marker that means "the chain finished on purpose"
#   $3  file to grep for that marker
#   $4  minutes of agreement before STALL_DETECTED fires (default 6)
#   $5  polls with no work processes before STALL_DEADCHAIN fires (default 4)
#
# PID discipline: `pgrep -f` patterns are BRACKETED on the first character
# ('[m]istralrs') so the sentinel can never match its own command line — the
# same trick that stops `pkill -f` from murdering the invoking SSH command
# (session-2 lesson, OPERATING_PROTOCOL §4.4).
set -uo pipefail

LOG_DIR=${1:-/root/logs}
DONE_MARKER=${2:-S6_COMPLETE}
DONE_FILE=${3:-/root/logs/s6.log}
STALL_MINUTES=${4:-6}
DEAD_POLLS=${5:-4}
STATUS_DIR=${STATUS_DIR:-/srv/arcstatus}
POLL_SECS=${POLL_SECS:-60}

mkdir -p "$STATUS_DIR" "$LOG_DIR"

# Work this session actually does. Bracketed first char, see header.
WORK_PAT='[m]istralrs (quantize|serve|calibrate)|[r]un_gsm8k|[b]atch_load_probe|[s]peed_probe|[q]tip_gemv_tune|[c]argo (test|run|build)|[s]6_upload_uqff'

signature() {
  # mtime+size of every watched log, collapsed to one hash. Works on GNU stat;
  # falls back to `ls` on anything else.
  { stat -c '%Y %s %n' "$LOG_DIR"/*.log 2>/dev/null \
      || ls -l --time-style=+%s "$LOG_DIR"/*.log 2>/dev/null \
      || ls -l "$LOG_DIR"/*.log 2>/dev/null; } | md5sum 2>/dev/null | cut -d' ' -f1
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -dc '0-9'
}

work_procs() {
  pgrep -cf "$WORK_PAT" 2>/dev/null | tr -dc '0-9'
}

report() {
  local msg="$(date -u +%Y-%m-%dT%H:%M:%SZ) $1"
  echo "$msg" | tee -a "$STATUS_DIR/stall.txt"
  tail -3 "$STATUS_DIR/stall.txt" > "$STATUS_DIR/stall_latest.txt" 2>/dev/null
}

last_sig=$(signature)
quiet=0
dead=0
echo "sentinel2: watching $LOG_DIR (stall=${STALL_MINUTES} agreeing polls of ${POLL_SECS}s, done marker '$DONE_MARKER' in $DONE_FILE)"

while true; do
  sleep "$POLL_SECS"

  # Finished on purpose: stand down without a marker.
  if grep -q "$DONE_MARKER" "$DONE_FILE" 2>/dev/null; then
    echo "sentinel2: '$DONE_MARKER' seen; standing down"
    exit 0
  fi

  sig=$(signature)
  util=$(gpu_util); util=${util:-0}
  work=$(work_procs); work=${work:-0}

  if [ "$sig" = "$last_sig" ] && [ "$util" -lt 5 ] && [ "$work" -gt 0 ]; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
  last_sig="$sig"

  if [ "$quiet" -ge "$STALL_MINUTES" ]; then
    report "STALL_DETECTED: $work work proc(s) alive, logs frozen ${quiet} polls, gpu ${util}%"
    quiet=0
  fi

  if [ "$work" -eq 0 ]; then
    dead=$((dead + 1))
    if [ "$dead" -ge "$DEAD_POLLS" ]; then
      report "STALL_DEADCHAIN: no work procs and '$DONE_MARKER' absent — the chain fell over"
      dead=0
    fi
  else
    dead=0
  fi
done
