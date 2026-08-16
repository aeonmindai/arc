#!/usr/bin/env bash
# Session-6 status digest — ONE small page a 60 s watchdog can poll for ~0
# tokens.
#
# status_server.sh serves /srv/arcstatus on :8899 and writes a verbose
# status.txt (tail -3 of every log). That is fine for a human but expensive for
# a watchdog that polls every 60 s for eight hours. This writes a SECOND file,
# /srv/arcstatus/s6.txt, containing only:
#
#   * elapsed wall + spend so far + the budget trip-wire state
#   * the last N machine-greppable event markers from the driver log
#   * one nvidia-smi line
#   * any stall markers
#
# The watchdog does a set-difference against its previous poll and prints only
# NEW event lines, so idle hours cost nothing.
#
# THE TOKEN NEVER COMES NEAR THIS FILE. It greps a fixed marker allow-list out
# of the driver log; the upload script prints no secret, and the allow-list
# could not carry one even if it did.
#
# Usage:
#   S6_START_EPOCH=$(date +%s) nohup bash s6_status_digest.sh > /root/logs/digest.log 2>&1 &
#
# Env:
#   S6_LOG          driver log to grep      (default /root/logs/s6.log)
#   S6_START_EPOCH  session start, epoch s  (default: file mtime of S6_LOG)
#   S6_RATE_HR      $/hr for the spend line (default 4.92)
#   S6_WIRE_H       anti-hang teardown wire, h (default 7.75; NOT a budget cap —
#                   Jish tops up, this only stops a hung step billing forever)
#   S6_EVENTS       how many markers to show (default 12)
set -uo pipefail

STATUS_DIR=${STATUS_DIR:-/srv/arcstatus}
S6_LOG=${S6_LOG:-/root/logs/s6.log}
S6_RATE_HR=${S6_RATE_HR:-4.92}
S6_WIRE_H=${S6_WIRE_H:-7.75}
S6_EVENTS=${S6_EVENTS:-12}
PERIOD=${PERIOD:-20}

mkdir -p "$STATUS_DIR"
if [ -z "${S6_START_EPOCH:-}" ]; then
  S6_START_EPOCH=$(stat -c %Y "$S6_LOG" 2>/dev/null || date +%s)
fi
echo "$$" >> "$STATUS_DIR/pids"

# The complete marker vocabulary the driver emits. Keep in sync with s6_driver.sh
# (test_s6_driver.sh asserts every marker the driver can print appears here).
MARKERS='^::::::|^HEALTH_|^BUILD_|^BEAM_|^BAKE_|^PACE\[|^UPLOAD_|^BATCH\[|^BATCHSWEEP\[|^SPEED\[|^COHERENCE|^GSM8K\[|^DEGEN\[|^GATE\[|^WINNERS:|^TUNE_|^CALIB_|^MTP_|^RESULTS_TGZ|^PULL_WITH|^S6_COMPLETE|^ABORT_|^ABORTED_AWAITING_DECISION|^FAILED_ASSERTION|^DECISION_|^RESUMING_IN_PLACE|^PAUSE|^PAUSED|^DELETE_INSTANCE_NOW|^BIN_SOURCE|^CACHE_|^BINCACHE_|^SKIP_|^TRIPWIRE|^FAIL:|panicked at|CUDA_ERROR|out of memory'

while true; do
  now=$(date +%s)
  elapsed=$(( now - S6_START_EPOCH ))
  {
    # A demand to delete outranks everything: a forgotten paused box is exactly
    # the failure DOCTRINE D10 exists to prevent, and this page is what the 60 s
    # watchdog reads. It goes FIRST so no amount of scrolling can hide it.
    if [ -f "$STATUS_DIR/DELETE_ME" ]; then
      printf 'DELETE_INSTANCE_NOW %s\n' "$(cat "$STATUS_DIR/DELETE_ME")"
      printf '  results: %s\n' "$STATUS_DIR/s6_results.tgz"
    elif grep -qa '^ABORTED_AWAITING_DECISION' "$S6_LOG" 2>/dev/null \
         && ! grep -qa '^DECISION_RECEIVED' "$S6_LOG" 2>/dev/null; then
      printf 'ABORTED_AWAITING_DECISION — box is PAUSED, write a decision or it self-condemns\n'
      grep -a '^ABORTED_AWAITING_DECISION' "$S6_LOG" | tail -1
      grep -a '^FAILED_ASSERTION' "$S6_LOG" | tail -1
    fi
    printf '== arc session 6 @ %s ==\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    awk -v e="$elapsed" -v r="$S6_RATE_HR" -v w="$S6_WIRE_H" 'BEGIN{
      h = e/3600.0;
      printf "cum_h=%.2f spend_usd=%.2f wire_h=%.2f%s\n", h, h*r, w,
             (h > w ? "  <<< BUDGET_WIRE: TEAR DOWN NOW" : "");
    }'
    printf -- '-- gpu: '
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm \
      --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    printf -- '-- events (last %s):\n' "$S6_EVENTS"
    grep -aE "$MARKERS" "$S6_LOG" 2>/dev/null | tail -"$S6_EVENTS"
    if [ -s "$STATUS_DIR/stall_latest.txt" ]; then
      printf -- '-- stall:\n'
      cat "$STATUS_DIR/stall_latest.txt"
    fi
  } > "$STATUS_DIR/s6.txt.tmp" 2>&1
  mv "$STATUS_DIR/s6.txt.tmp" "$STATUS_DIR/s6.txt"
  sleep "$PERIOD"
done
