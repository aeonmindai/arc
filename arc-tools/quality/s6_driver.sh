#!/usr/bin/env bash
# =============================================================================
# GPU SESSION 6 DRIVER — one prepared script, run detached, zero thinking on
# the box. Companion doc: GPU_SESSION_RUNBOOK_6.md (read it before renting).
#
#   file_upload  -> /root/s6_driver.sh
#   (nohup bash /root/s6_driver.sh > /root/logs/s6.log 2>&1 < /dev/null &)
#
# NEVER paste this into ssh_execute as a heredoc: ssh_execute mangles heredocs
# and complex quoting (session-1 lesson). Upload the file, then `bash` it.
#
# Everything long runs detached; every kill is a PID kill captured from `$!`
# (a `pkill -f` once murdered the invoking SSH command). Every step emits
# machine-greppable markers that s6_status_digest.sh folds into
# /srv/arcstatus/s6.txt for a 60 s watchdog.
#
# =============================================================================
#                     ***  NEVER DEBUG ON A PAID BOX.  ***
#
# Jish, 2026-08-14: "if you spot any problem on gpu, tear down, fix on cpu, try
# again — loop, don't worry about balance."
#
# The instant any ABORT-IF trips, this driver harvests EVERYTHING it has into
# /srv/arcstatus/s6_results.tgz, writes results/s6_failure.json with the exact
# failed assertion (expected vs got, plus which log in the tarball explains
# it), and exits 10. It never retries a failed step, never "just checks one
# thing", and never leaves work in flight. The orchestrator then pulls the
# tarball, DELETES the instance, and diagnoses on CPU where compute is free.
#
# The tarball is the deliverable of a FAILED session, not a consolation prize:
# a box that dies with nothing pulled makes the next attempt blind and we pay
# twice for the same ignorance. So the harvest is unconditional and always
# happens BEFORE the delete — including on the health-gate path, where the one
# file it saves (box_health.json) is exactly what decides whether the next box
# should be in a different region.
#
# BUT "never debug on a paid box" means never SIT AND THINK on one. It does not
# mean throwing away 70 minutes of paid setup to fix a typo.
# Jish, 2026-08-14: "loop teardowns only makes sense if its more than 5 minutes
# of code work, if its less, continue holding the box."
#
# So an abort HARVESTS and then PAUSES: work stopped, server down, results
# already tarred and pullable, box alive, waiting for one decision. The
# arithmetic is printed in the pause banner so the call is mechanical:
#
#   idle box            $4.92/hr = $0.082/min
#   re-entry PRE-upload  boot ~10 + 149 GB download ~35 + build ~25
#                        = ~70 min = ~$5.74 before any work happens
#   re-entry POST-upload boot ~10 + cached binary <1 + 68 GB UQFF ~15
#                        = ~26 min = ~$2.13   (no weights, no bake)
#   resume in place      0 min = $0
#
# Hold if the fix is shorter than the re-entry; delete if it is longer.
#
# D10 ("never leave a GPU idle") still governs, and a forgotten paused box is
# EXACTLY the failure D10 exists to prevent — so the pause has a hard idle
# timeout (PAUSE_IDLE_SECS, default 900 s) after which the driver demands an
# immediate DELETE and stops waiting. See the runbook: the on-box timer cannot
# delete the instance, it makes the delete unmissable inside 60 s.
# =============================================================================
#
# STEP ORDER — priority order, with ONE deliberate deviation, stated here so
# nobody has to re-derive it at 3am:
#
#   S0  box health gate           BEFORE the 149 GB download ($30 lesson, s5a)
#   S1  bootstrap: build + download + MTP telemetry patch + eval data
#   S2  CUDA beam parity gate     <- THE GATE. No pass, no bake, no session.
#   S3  bake with beam            header + pace gates
#   S4  UPLOAD THE UQFF           <- irreversible value; before all measurement
#   S5  serve + BATCH SWEEP       <- THE headline (DOCTRINE D2), never measured
#   S6  GSM8K n=100 2048-cap      re-measure the PROVISIONAL 87.0
#   S7  gen-2 GEMV sweep (98)     server down
#   S8  calibration probe         server down — closes DOCTRINE D11's question
#   S9  tuned serve + batch re-sweep
#   S10 MTP acceptance            first valid number ever
#   S11 voting GSM8K k=5          the 90+ attempt (D6)
#   S12 tar + teardown            NEVER CUT
#
# The deviation: mission priority lists voting (8) above MTP (9) and
# calibration (10). Voting costs ~90 min for ONE number and degrades
# gracefully (resume-safe JSON, keep n>=60); MTP and calibration cost 20 and
# 25 min for one number each and degrade to NOTHING if cut. Running the two
# cheap ones first and letting the expensive one absorb the trip-wire buys
# three numbers instead of one. Voting is therefore last and TRUNCATABLE, not
# droppable.
#
# Resume:  S6_FROM=S5 bash s6_driver.sh     (skips steps before S5)
# Subset:  S6_ONLY=S7,S8 bash s6_driver.sh
# Dry run: see test_s6_driver.sh (mock binaries on PATH, no GPU, no model)
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------- seams
# Every external command goes through a variable so test_s6_driver.sh can
# inject a mock. Defaults are the real thing.
S6_ROOT=${S6_ROOT:-/root}
LOG_DIR=${LOG_DIR:-$S6_ROOT/logs}
STATUS_DIR=${STATUS_DIR:-/srv/arcstatus}

# boot_run161_h200.sh picks the largest-free mount; mirror its choice.
if [ -z "${WORKROOT:-}" ]; then
  WORKROOT=$S6_ROOT
  for m in /ephemeral /workspace /mnt /root; do
    [ -d "$m" ] && WORKROOT=$m && break
  done
fi
ARC=${ARC:-$WORKROOT/work/arc}
V4_DIR=${V4_DIR:-$WORKROOT/models/DeepSeek-V4-Flash}
Q=${Q:-$ARC/arc-tools/quality}
RESULTS=${RESULTS:-$Q/results}
UQFF_DIR=${UQFF_DIR:-$V4_DIR/uqff}

S6_CARGO=${S6_CARGO:-cargo}
S6_PY=${S6_PY:-python3}
S6_MISTRALRS=${S6_MISTRALRS:-$ARC/target/release/mistralrs}
S6_BOOT=${S6_BOOT:-$S6_ROOT/boot.sh}
S6_HEALTH_GATE=${S6_HEALTH_GATE:-$S6_ROOT/box_health_gate.sh}
S6_SENTINEL=${S6_SENTINEL:-$S6_ROOT/stall_sentinel.sh}
S6_UPLOAD_PY=${S6_UPLOAD_PY:-$S6_ROOT/s6_upload_uqff.py}
S6_BINCACHE_PY=${S6_BINCACHE_PY:-$S6_ROOT/s6_bin_cache.py}
S6_TOKEN_FILE=${S6_TOKEN_FILE:-$S6_ROOT/.hf_token}

# ---------------------------------------------------------------- policy
ARC_BRANCH=${ARC_BRANCH:-master}
HF_REPO=${HF_REPO:-aeonmind/DeepSeek-V4-Flash-UQFF-qtip2}
BASE_MODEL=${BASE_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$ARC/chat_templates/deepseek_v4.json}
PORT=${PORT:-1234}
RATE_HR=${RATE_HR:-4.92}

# W=256 OR NO BAKE. There is no width ladder and no "degraded" rung.
# Jish, 2026-08-14: "beam should work on 256, if fail, tear down, spin up, loop
# until alright, any other problem -> do the fucking same."
# W=128 costs -0.004 matmul cos and the width is STAMPED INTO THE ARTIFACT
# forever. Shipping it because the good one was inconvenient is the same
# species of shortcut DOCTRINE D4 bans. If 256 does not hold on this box, the
# answer is a different box, not a worse artifact.
BEAM_WIDTH=${BEAM_WIDTH:-256}
GSM8K_N=${GSM8K_N:-100}
GSM8K_MAXTOK=${GSM8K_MAXTOK:-2048}
VOTES=${VOTES:-5}
BATCHES=${BATCHES:-1,8,16,32,64}
MTP_DEPTH=${MTP_DEPTH:-2}
CALIB_SAMPLES=${CALIB_SAMPLES:-8}
CALIB_BOX_SECS=${CALIB_BOX_SECS:-1500}   # 25 min hard box on S8

# ------------------------------------------------------- single-attempt wires
# These are ANTI-HANG bounds for one attempt, not budget caps. Jish tops up;
# the reason a step is not allowed to run forever is that a hung step teaches
# us nothing while it bills, not that we are rationing dollars.
S6_ATTEMPT=${S6_ATTEMPT:-1}
PAUSE_IDLE_SECS=${PAUSE_IDLE_SECS:-900}       # hard idle timeout on the paused state
REENTRY_MIN_PRE_UPLOAD=${REENTRY_MIN_PRE_UPLOAD:-70}
REENTRY_MIN_POST_UPLOAD=${REENTRY_MIN_POST_UPLOAD:-26}
S6_USE_BIN_CACHE=${S6_USE_BIN_CACHE:-1}       # try the cached binary before building
BIN_CACHE_PREFIX=${BIN_CACHE_PREFIX:-arc-bin}
WIRE_BAKE_H=${WIRE_BAKE_H:-2.85}
WIRE_UPLOAD_H=${WIRE_UPLOAD_H:-3.30}
WIRE_LASTCHANCE_H=${WIRE_LASTCHANCE_H:-6.00}
WIRE_STOP_H=${WIRE_STOP_H:-7.25}
WIRE_TEARDOWN_H=${WIRE_TEARDOWN_H:-7.75}

DRYRUN=${S6_DRYRUN:-0}
S6_FROM=${S6_FROM:-S0}
S6_ONLY=${S6_ONLY:-}

START_EPOCH=${S6_START_EPOCH:-$(date +%s)}
export S6_START_EPOCH=$START_EPOCH

SERVE_PID=""
BAKE_PID=""
FAIL_LOG="s6.log"   # which log the current step would blame; used by abort()

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULTS" 2>/dev/null

# ---------------------------------------------------------------- helpers
mark()  { echo ":::::: $* ::::::"; date -u +%H:%M:%S; }
say()   { echo "$*"; }
skipm() { echo "SKIP_$1 $2"; }

# Exit code 10 = this attempt failed. The ONE rule this driver exists to
# enforce, and the reason every terminal path funnels through one function:
#
#   ***  NEVER DEBUG ON A PAID BOX.  ***
#
# Not one step. Not "let me just check one thing while I'm here." The instant
# an ABORT-IF trips, we harvest everything we have and the box dies. Whether
# the cause was our code or a bad rental gets decided from the pulled
# artifacts, on a laptop, off the clock. An in-session retry of a failed step
# on the same box is never allowed: the box is the one variable we cannot
# control, and $4.92/hr is the worst possible place to think.
#
# The driver cannot delete the instance itself (that is the orchestrator's
# Runcrate MCP job), so its whole responsibility is to make the pull-and-delete
# unmissable AND to guarantee the tarball exists BEFORE anyone deletes.
#
#   abort <STEP_TAG> <assertion-id> <expected> <got> <message>
#
# All five are mandatory. A failure report that does not say expected-vs-got is
# a report someone has to rent a box to reproduce.
abort() {
  local step=$1 assertion=$2 expected=$3 got=$4 msg=$5
  echo "ABORT_$step $msg"
  echo "FAILED_ASSERTION $assertion  expected=[$expected]  got=[$got]"
  {
    printf '{\n  "attempt": %s,\n  "step": "%s",\n  "assertion": "%s",\n' \
      "$S6_ATTEMPT" "$step" "$assertion"
    printf '  "expected": "%s",\n  "got": "%s",\n' \
      "$(jstr "$expected")" "$(jstr "$got")"
    printf '  "message": "%s",\n' "$(jstr "$msg")"
    printf '  "logs_in_tarball": [%s],\n' "$(printf '"results/%s"' "$(basename "${FAIL_LOG:-s6.log}")")"
    printf '  "elapsed_h": %s,\n  "attempt_cost_usd": %s,\n' \
      "$(elapsed_h)" "$(awk -v e="$(elapsed_h)" -v r="$RATE_HR" 'BEGIN{printf "%.2f", e*r}')"
    printf '  "gpu": "%s",\n' "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    printf '  "arc_commit": "%s",\n' "$(cd "$ARC" 2>/dev/null && git rev-parse HEAD 2>/dev/null)"
    printf '  "utc": "%s"\n}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$RESULTS/s6_failure.json" 2>/dev/null
  cp "$RESULTS/s6_failure.json" "$STATUS_DIR/s6_failure.json" 2>/dev/null
  harvest "FAILED at $step"
  pause_for_decision "$step" "$assertion"
}

# The paused state. Reached only from abort(), and only AFTER harvest() has
# already produced a pullable tarball — so a decision to delete is never a
# decision to lose evidence.
#
# Decisions arrive as one line in $STATUS_DIR/s6_decision (one ssh_execute):
#   resume:S5   continue from step S5 — bake and upload are NOT redone
#   retry       re-run the step that just failed, on this box, after a fix
#   hold:N      extend the pause by N seconds (a fix is in progress)
#   delete      give up now
# The file is consumed on read, so a stale decision cannot loop the driver.
pause_for_decision() {
  local failed_step=$1 assertion=$2
  local dfile="$STATUS_DIR/s6_decision"
  local reentry=$REENTRY_MIN_PRE_UPLOAD phase="PRE-upload"
  if [ -f "$RESULTS/.uploaded" ]; then
    reentry=$REENTRY_MIN_POST_UPLOAD; phase="POST-upload"
  fi
  rm -f "$dfile"
  echo "ABORTED_AWAITING_DECISION step=$failed_step assertion=$assertion idle_timeout_s=$PAUSE_IDLE_SECS"
  echo "  Box is PAUSED: no work processes, no server, results already tarred and pullable."
  echo "  HOLD-OR-DELETE ARITHMETIC ($phase): idle costs \$0.082/min; re-entering"
  echo "    costs ~${reentry} min ~= \$$(awk -v m="$reentry" -v r="$RATE_HR" 'BEGIN{printf "%.2f", m*r/60}')."
  echo "    => HOLD if the fix is under ~${reentry} min of code work. DELETE if it is longer."
  echo "  Decide with ONE ssh_execute:"
  echo "    echo resume:S5 > $dfile   # continue from S5 (bake/upload NOT redone)"
  echo "    echo retry     > $dfile   # re-run $failed_step here after an in-session fix"
  echo "    echo hold:1800 > $dfile   # a fix is in progress, extend the pause"
  echo "    echo delete    > $dfile   # give up now"
  local waited=0 deadline=$PAUSE_IDLE_SECS
  while [ "$waited" -lt "$deadline" ]; do
    if [ -s "$dfile" ]; then
      local d
      d=$(head -1 "$dfile" | tr -d '[:space:]')
      rm -f "$dfile"
      echo "DECISION_RECEIVED $d after ${waited}s"
      case "$d" in
        delete|abort|teardown) demand_delete "operator chose delete" ;;
        hold:*)
          local extra=${d#hold:}
          deadline=$((deadline + extra))
          echo "PAUSE_EXTENDED by ${extra}s (new deadline ${deadline}s)"
          ;;
        retry)
          echo "RESUMING_IN_PLACE from $failed_step (no bake, no re-upload)"
          exec env S6_START_EPOCH="$START_EPOCH" S6_ATTEMPT="$S6_ATTEMPT" \
            S6_FROM="$failed_step" S6_ONLY="" bash "$0"
          ;;
        resume:*)
          local tgt=${d#resume:}
          echo "RESUMING_IN_PLACE from $tgt (no bake, no re-upload)"
          exec env S6_START_EPOCH="$START_EPOCH" S6_ATTEMPT="$S6_ATTEMPT" \
            S6_FROM="$tgt" S6_ONLY="" bash "$0"
          ;;
        *) echo "DECISION_UNKNOWN '$d' — ignoring, still waiting" ;;
      esac
    fi
    nap 10; waited=$((waited + 10))
    if [ $((waited % 120)) -eq 0 ]; then
      echo "PAUSED ${waited}s/${deadline}s — no decision yet (idle cost so far \$$(awk -v w="$waited" -v r="$RATE_HR" 'BEGIN{printf "%.2f", w*r/3600}'))"
    fi
    [ "$DRYRUN" = "1" ] && [ "$waited" -ge 80 ] && break
  done
  demand_delete "idle timeout after ${deadline}s with no decision"
}

# The one path that could reintroduce a forgotten idle box (D10). It cannot
# call Runcrate itself, so it makes the delete impossible to miss: a marker
# file the status digest prints as its FIRST line, and a marker the 60 s
# watchdog greps for and acts on without asking.
demand_delete() {
  echo "DELETE_INSTANCE_NOW ($1)"
  echo "delete: $1 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/DELETE_ME" 2>/dev/null
  echo "  Results are already at $STATUS_DIR/s6_results.tgz — pull, then DELETE (not stop)."
  echo "  Fix on CPU (free), then rent again with S6_ATTEMPT=$((S6_ATTEMPT + 1))."
  budget_line
  exit 10
}

# A cached binary that passes --version can still be the wrong binary. Its
# DEFINITIVE smoke is the first real use, so the two steps that constitute one
# (the bake header at S3, /health at S5) call this before they abort.
# Rebuilding off a bad cache is NOT debugging on a paid box — it is
# invalidating a cache we chose to trust, with no diagnosis involved — so it is
# the one retry the policy allows, and only once.
rebuild_from_source_if_cached() {
  local why=$1
  [ "${BIN_FROM_CACHE:-0}" = "1" ] || return 1
  echo "CACHE_REJECTED $why — the cached binary failed its first real use; rebuilding from source (once)"
  BIN_FROM_CACHE=0
  rm -f "$S6_MISTRALRS"
  ( cd "$ARC" && $S6_CARGO build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn" ) \
    2>&1 | tail -3 | tee -a "$LOG_DIR/build_rebuild.log"
  [ -x "$S6_MISTRALRS" ] && { echo "BIN_SOURCE=built-from-source (after cache rejection)"; return 0; }
  return 1
}

# JSON string escaping for the failure report (quotes, backslashes, newlines).
jstr() { printf '%s' "$1" | tr '\n\t' '  ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# THE tar-and-publish path. Shared by success and failure on purpose: the
# tarball is the deliverable of a FAILED session too, not a consolation prize.
# A box that dies with nothing pulled makes the next attempt blind, and we pay
# twice for the same ignorance. Unconditional, and always BEFORE the delete —
# including on the health-gate path, where there is barely anything to tar and
# it still runs, because "barely anything" includes box_health.json, which is
# exactly what decides whether the next box should be in a different region.
harvest() {
  local why=${1:-done}
  echo ":::::: HARVEST ($why) ::::::"
  kill_pid "${SERVE_PID:-}" 4 2>/dev/null; SERVE_PID=""
  kill_pid "${BAKE_PID:-}" 4 2>/dev/null;  BAKE_PID=""
  mkdir -p "$RESULTS" 2>/dev/null
  cp "$LOG_DIR"/*.log "$RESULTS/" 2>/dev/null
  cp "$STATUS_DIR"/box_health*.json "$RESULTS/" 2>/dev/null
  cp "$STATUS_DIR"/stall*.txt "$RESULTS/" 2>/dev/null
  ( cd "$(dirname "$RESULTS")" && tar czf "$S6_ROOT/s6_results.tgz" "$(basename "$RESULTS")/" ) 2>/dev/null
  cp "$S6_ROOT/s6_results.tgz" "$STATUS_DIR/" 2>/dev/null
  if [ -s "$STATUS_DIR/s6_results.tgz" ]; then
    echo "RESULTS_TGZ: $STATUS_DIR/s6_results.tgz ($(wc -c < "$STATUS_DIR/s6_results.tgz" | tr -d ' ') bytes, $(tar tzf "$S6_ROOT/s6_results.tgz" 2>/dev/null | wc -l | tr -d ' ') entries)"
    echo "PULL_WITH: curl -s -o s6_results.tgz http://<BOX_IP>:8899/s6_results.tgz"
  else
    echo "FAIL: could not write the results tarball — pull /root/logs/ file by file before deleting"
  fi
  # The token must not outlive the box, on ANY path (session 5 leaked one and
  # it had to be rotated).
  if [ -f "$S6_TOKEN_FILE" ]; then
    shred -u "$S6_TOKEN_FILE" 2>/dev/null || rm -f "$S6_TOKEN_FILE"
    [ -f "$S6_TOKEN_FILE" ] && echo "FAIL: token file still present" || echo "TOKEN_DELETED"
  fi
  budget_line
}

elapsed_h() {
  awk -v s="$START_EPOCH" -v n="$(date +%s)" 'BEGIN{printf "%.3f", (n-s)/3600.0}'
}
past() {  # past <hours> -> 0 when elapsed exceeds it
  awk -v e="$(elapsed_h)" -v w="$1" 'BEGIN{exit !(e > w)}'
}
budget_line() {
  awk -v e="$(elapsed_h)" -v r="$RATE_HR" 'BEGIN{printf "cum_h=%.2f spend_usd=%.2f\n", e, e*r}'
}

# Sleep, but collapse to a heartbeat under S6_DRYRUN so test_s6_driver.sh can
# exercise the whole chain in seconds instead of hours.
nap() { if [ "$DRYRUN" = "1" ]; then sleep 0.2; else sleep "$1"; fi; }

# Should this step run? Honors S6_FROM (ordered resume) and S6_ONLY (subset).
step_order() { case "$1" in
  S0) echo 0;; S1) echo 1;; S2) echo 2;; S3) echo 3;; S4) echo 4;; S5) echo 5;;
  S6) echo 6;; S7) echo 7;; S8) echo 8;; S9) echo 9;; S10) echo 10;;
  S11) echo 11;; S12) echo 12;; *) echo 99;; esac; }

want() {
  local s=$1
  if [ -n "$S6_ONLY" ]; then
    case ",$S6_ONLY," in *",$s,"*) return 0;; *) return 1;; esac
  fi
  [ "$(step_order "$s")" -ge "$(step_order "$S6_FROM")" ]
}

# Launch a stall sentinel bound to an exact PID (v1 = per-step kill escalation).
arm_sentinel() {
  local logf=$1 secs=$2 pid=$3 tag=$4
  [ -x "$S6_SENTINEL" ] || [ -f "$S6_SENTINEL" ] || return 0
  nohup bash "$S6_SENTINEL" "$logf" "$secs" "$pid" \
    > "$LOG_DIR/sentinel_$tag.log" 2>&1 &
}

kill_pid() {  # PID kill only. Never a pattern (it would kill the caller).
  local pid=$1 grace=${2:-8}
  [ -z "$pid" ] && return 0
  kill "$pid" 2>/dev/null
  nap "$grace"
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  return 0
}

wait_health() {  # wait_health <seconds>
  local limit=${1:-600} waited=0
  [ "$DRYRUN" = "1" ] && limit=20
  while [ "$waited" -lt "$limit" ]; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "SERVE_UP after ${waited}s"; return 0
    fi
    [ -n "$SERVE_PID" ] && ! kill -0 "$SERVE_PID" 2>/dev/null && {
      echo "SERVE_DIED before health (pid $SERVE_PID)"; return 1; }
    nap 5; waited=$((waited + 5))
  done
  echo "SERVE_TIMEOUT after ${limit}s"; return 1
}

# start_serve <logname> [extra args...]
start_serve() {
  local logname=$1; shift
  local uqff0
  uqff0=$(ls "$UQFF_DIR"/*.uqff 2>/dev/null | sort | head -1)
  if [ -z "$uqff0" ]; then echo "SERVE_NO_UQFF in $UQFF_DIR"; return 1; fi
  # shellcheck disable=SC2086
  nohup env ${ARC_QTIP_TUNE_TABLE:+ARC_QTIP_TUNE_TABLE=$ARC_QTIP_TUNE_TABLE} \
    ${ARC_MTP_LOG_ACCEPTANCE:+ARC_MTP_LOG_ACCEPTANCE=$ARC_MTP_LOG_ACCEPTANCE} \
    "$S6_MISTRALRS" serve -p "$PORT" -m "$V4_DIR" -a deepseekv4 \
    --from-uqff "$uqff0" --prefix-cache-n 0 \
    --chat-template "$CHAT_TEMPLATE" "$@" \
    > "$LOG_DIR/$logname.log" 2>&1 < /dev/null &
  SERVE_PID=$!
  echo "SERVE_PID=$SERVE_PID log=$LOG_DIR/$logname.log uqff=$uqff0"
  arm_sentinel "$LOG_DIR/$logname.log" 900 "$SERVE_PID" "$logname"
  if wait_health 900; then
    return 0
  fi
  # Never leave a half-dead server behind: callers test SERVE_PID to decide
  # whether their step can run at all.
  kill_pid "$SERVE_PID" 8
  SERVE_PID=""
  return 1
}

stop_serve() { kill_pid "$SERVE_PID" 8; SERVE_PID=""; nap 5; }

# =============================================================================
mark "S6 DRIVER START  branch=$ARC_BRANCH arc=$ARC v4=$V4_DIR beam=$BEAM_WIDTH"
budget_line
echo "ENV gates that must be UNSET all session (except where a step sets them):"
env | grep -E '^(ARC_QUANT_ATTENTION|ARC_QTIP_REFINE_SCALES|ARC_V4_WINDOW_ONLY|ARC_V4_STANDARD_DENSE|ARC_V4_NO_ABSORBED_DECODE|ARC_FORCE_NAIVE_SDPA|ARC_TIME_DECODE|ARC_ALLOW_UNSTAMPED_QTIP|ARC_QTIP_HESSIAN)=' \
  && abort ENV_DIRTY "banned_gate_exported" "none of ARC_QUANT_ATTENTION/ARC_QTIP_REFINE_SCALES/ARC_V4_*/ARC_FORCE_NAIVE_SDPA/ARC_TIME_DECODE/ARC_ALLOW_UNSTAMPED_QTIP/ARC_QTIP_HESSIAN exported" "$(env | grep -cE '^(ARC_QUANT_ATTENTION|ARC_QTIP_REFINE_SCALES|ARC_V4_WINDOW_ONLY|ARC_V4_STANDARD_DENSE|ARC_V4_NO_ABSORBED_DECODE|ARC_FORCE_NAIVE_SDPA|ARC_TIME_DECODE|ARC_ALLOW_UNSTAMPED_QTIP|ARC_QTIP_HESSIAN)=') exported (listed above)" "a banned gate is exported into the session environment; unset it in the launching shell and re-run" \
  || echo "ENV_CLEAN"

# =============================================================================
# S0 — BOX HEALTH GATE, before the 149 GB download.  ~2 min.
# s5a: 99% util at 132 W of 700 W = starved on host<->device transfers, ~3
# min/layer with a binary proven at 30 s/layer. Cost ~1.5 h and ~$7.
# =============================================================================
if want S0; then
  mark "S0 box-health"
  FAIL_LOG=health.log
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "HEALTH_COMPUTE_CAP=$CC"
  # has_qtip_kernels is set by mistralrs-quant/build.rs iff compute_cap >= 8.0.
  # Below that, EVERY cuda parity test in S2 returns Ok(()) VACUOUSLY.
  awk -v c="${CC:-0}" 'BEGIN{exit !(c+0 >= 8.0)}' \
    || abort BOX_COMPUTE_CAP "nvidia-smi --query-gpu=compute_cap" ">= 8.0" "${CC:-<no answer>}" "compute_cap below 8.0: mistralrs-quant/build.rs:113 would not set has_qtip_kernels, so every CUDA parity test in S2 returns Ok(()) SILENTLY and the gate passes vacuously. Rent an H100/H200-class box (cap 9.0). results/health.log"
  DRV=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9][0-9.]*' | grep -o '[0-9][0-9.]*$')
  TK=$(nvcc --version 2>/dev/null | grep -o 'release [0-9][0-9.]*' | grep -o '[0-9][0-9.]*$')
  echo "HEALTH_CUDA driver_max=$DRV toolkit=$TK"
  if [ -n "$DRV" ] && [ -n "$TK" ]; then
    NEWEST=$(printf '%s\n%s\n' "$DRV" "$TK" | sort -V | tail -1)
    [ "$TK" != "$DRV" ] && [ "$NEWEST" = "$TK" ] \
      && abort BOX_TOOLKIT "nvcc release vs nvidia-smi driver max CUDA" "toolkit <= $DRV" "toolkit $TK" "the CUDA toolkit is newer than the driver supports; the build succeeds and then dies at runtime with CUDA_ERROR_UNSUPPORTED_PTX_VERSION (cost session 1 forty minutes). Fixable in-image: apt-get install -y cuda-toolkit-${DRV/./-} then export CUDA_HOME=/usr/local/cuda-$DRV — but per policy this box dies and the next one is rented from an image whose toolkit matches. results/health.log"
  fi
  if [ "$DRYRUN" = "1" ]; then
    bash "$S6_HEALTH_GATE" --json "$STATUS_DIR/box_health.json" 2>&1 | tee "$LOG_DIR/health.log"
  else
    bash "$S6_HEALTH_GATE" --burn-secs 60 --json "$STATUS_DIR/box_health.json" \
      2>&1 | tee "$LOG_DIR/health.log"
  fi
  rc=${PIPESTATUS[0]}
  if [ "$rc" != "0" ]; then
    echo "HEALTH_FAIL rc=$rc"
    FAILLINE=$(grep -m1 '^FAIL ' "$LOG_DIR/health.log")
    abort BOX_HEALTH "box_health_gate.sh exit code" "0 (all gates PASS)" "$rc — ${FAILLINE:-see results/health.log}" "the rental is bad. DELETE it and rent a different box, preferably a different region. Do NOT debug the rental: s5a ran 99% util at 132W of a 700W limit (transfer-starved) and cost ~1.5h and ~\$7 before anyone noticed; abandoning here costs ~\$0.30 because the gate precedes the 149 GB download. results/health.log + results/box_health.json (the JSON is what tells you offline whether to change region)."
  fi
  echo "HEALTH_PASS"
  cp "$STATUS_DIR/box_health.json" "$RESULTS/box_health.json" 2>/dev/null
fi

# =============================================================================
# S1 — Bootstrap: build ∥ 149 GB download, MTP telemetry patch, eval corpora.
# ~55 min.  NO cudnn: it cost -62% decode (5.45 vs 14.58 tok/s, session 4).
# =============================================================================
if want S1; then
  mark "S1 bootstrap"
  FAIL_LOG=boot.log
  # BINARY CACHE. A cache HIT skips the ~25 min cargo build. It is only taken
  # when the arc commit, driver version, CUDA toolkit, compute cap and glibc
  # all match the box the binary was built on, and the sha256 matches — those
  # are the five things that make a Linux CUDA binary refuse to run, or run
  # wrongly, somewhere else. A code fix changes arc_commit, so the cache
  # invalidates itself on exactly the loop where a stale binary would be
  # dangerous. See s6_bin_cache.py for the full reasoning.
  BIN_FROM_CACHE=0
  if [ "$S6_USE_BIN_CACHE" = "1" ] && [ -s "$S6_TOKEN_FILE" ] && [ -d "$ARC/.git" ]; then
    if $S6_PY "$S6_BINCACHE_PY" pull --repo-id "$HF_REPO" --prefix "$BIN_CACHE_PREFIX" \
         --token-file "$S6_TOKEN_FILE" --arc "$ARC" --out "$S6_MISTRALRS" \
         2>&1 | grep -v -i token | tee "$LOG_DIR/bincache.log" | grep -q BINCACHE_HIT; then
      # Pre-trust smoke: proves the file executes and its dynamic objects
      # resolve. The DEFINITIVE smoke is its first real use (the bake header at
      # S3, or /health at S5); a failure there rebuilds from source.
      if "$S6_MISTRALRS" --version > "$LOG_DIR/bincache_smoke.log" 2>&1 \
         && ! ldd "$S6_MISTRALRS" 2>/dev/null | grep -q "not found"; then
        BIN_FROM_CACHE=1
        echo "BIN_SOURCE=cache ($(head -1 "$LOG_DIR/bincache_smoke.log"))"
        echo "CACHE_SMOKE=pass (--version + ldd; definitive smoke is first real use)"
      else
        echo "CACHE_SMOKE=fail — cached binary does not run here; building from source"
        rm -f "$S6_MISTRALRS"
      fi
    else
      echo "BIN_SOURCE=cache-miss (see results/bincache.log for which field mismatched)"
    fi
  fi

  if [ "$BIN_FROM_CACHE" = "1" ] && [ "${S6_SKIP_BOOT:-0}" != "1" ]; then
    # Still need the repo (arc-tools, chat_templates, cargo examples) and the
    # weights unless we are on the post-upload path, but NOT the cargo build.
    echo "BOOT_MODE=cache (repo + data only, build skipped)"
  fi

  if [ "${S6_SKIP_BOOT:-0}" != "1" ]; then
    ARC_BRANCH=$ARC_BRANCH nohup bash "$S6_BOOT" > "$LOG_DIR/boot.log" 2>&1 < /dev/null &
    BOOT_PID=$!
    echo "BOOT_PID=$BOOT_PID"
    arm_sentinel "$LOG_DIR/boot.log" 1800 "$BOOT_PID" boot
    waited=0
    while [ "$waited" -lt 7200 ]; do
      grep -q "BOOTSTRAP_COMPLETE" "$LOG_DIR/boot.log" 2>/dev/null && break
      grep -q "^FAIL:" "$LOG_DIR/boot.log" 2>/dev/null \
        && abort BUILD "boot.sh completes without FAIL:" "no FAIL: line" "$(grep -m1 '^FAIL:' "$LOG_DIR/boot.log")" "bootstrap failed; the line above is boot.sh's own diagnosis. results/boot.log"
      kill -0 "$BOOT_PID" 2>/dev/null || break
      nap 30; waited=$((waited + 30))
    done
    grep -q "BOOTSTRAP_COMPLETE" "$LOG_DIR/boot.log" 2>/dev/null \
      || abort BUILD "BOOTSTRAP_COMPLETE in boot.log" "present within 7200s" "absent after ${waited}s" "bootstrap neither completed nor reported FAIL: — most likely the 149 GB download or the cargo build is still crawling. results/boot.log + results/dl.log"
  fi
  [ -x "$S6_MISTRALRS" ] || abort BUILD "release binary exists" "$S6_MISTRALRS executable" "missing or not executable" "boot.sh claimed BOOTSTRAP_COMPLETE but produced no binary. results/boot.log"
  echo "BIN_SOURCE=built-from-source"

  # MTP acceptance telemetry. VERIFIED 2026-08-14: at master d6ceaf1ad
  # log_acceptance_rate() has ZERO call sites (mtp_pipeline.rs:690), so S10
  # would measure nothing — the exact session-3 failure. The session-2-era
  # patch NO LONGER APPLIES (PR #30 rewrote the file); s6_mtp_acceptance_
  # telemetry.patch is regenerated against d6ceaf1ad and compiles clean.
  #
  # BUT agent AK is landing a PERMANENT call site in mistralrs-core. If that
  # merged first, this checkout already logs and applying the patch on top
  # would add a SECOND call and double-count acceptance. So the patch is a
  # no-op whenever a call site is already present — the grep is the authority,
  # not the merge order.
  if grep -rq "log_acceptance_rate()" --include='*.rs' "$ARC/mistralrs-core/src" 2>/dev/null; then
    skipm MTP_PATCH "a log_acceptance_rate() call site is already in this checkout (agent AK's permanent fix landed) — NOT applying the patch, which would double-count acceptance. Telemetry still needs ARC_MTP_LOG_ACCEPTANCE=1 unless AK made it unconditional; S10 verifies either way."
  elif git -C "$ARC" apply --check "$Q/patches/s6_mtp_acceptance_telemetry.patch" 2>/dev/null; then
    git -C "$ARC" apply "$Q/patches/s6_mtp_acceptance_telemetry.patch" \
      && echo "PATCH_OK s6_mtp_acceptance_telemetry"
    ( cd "$ARC" && $S6_CARGO build --release -p mistralrs-cli --features "cuda flash-attn" ) \
      2>&1 | tail -3 | tee -a "$LOG_DIR/build_patch.log"
    [ -x "$S6_MISTRALRS" ] || abort BUILD "release binary after the MTP patch rebuild" "$S6_MISTRALRS executable" "missing" "the MTP telemetry patch applied but the rebuild failed. results/build_patch.log — reproduce with the patch on CPU."
  else
    skipm MTP_PATCH "s6_mtp_acceptance_telemetry.patch does not apply to this checkout — S10 will measure nothing; record that, do not debug"
  fi

  bash "$Q/fetch_data.sh" 2>&1 | tail -3 | tee "$LOG_DIR/fetch_data.log"
  grep -q "DATA_READY" "$LOG_DIR/fetch_data.log" || skipm DATA "fetch_data.sh did not print DATA_READY — GSM8K steps will fail"
  echo "BUILD_OK $(cd "$ARC" && git log --oneline -1 2>/dev/null)"
fi

# =============================================================================
# S2 — CUDA BEAM PARITY GATE.  ~15 min.  THIS IS THE GATE.
#
# PR #33's kernel is projected 31-62 min for a 44-layer bake against a measured
# ~6.2 h exhaustive bake, but NO GPU HAS EVER RUN IT. The default was
# deliberately not flipped: hardware parity must pass first.
#
# W=256 OR NO BAKE. There is no width ladder, no "largest passing width", no
# degraded rung behind a flag. cuda_beam_matches_cpu_beam_bit_for_bit iterates
# W = 64 -> 128 -> 256 and fails at the FIRST bad width, so a W=256 failure
# still tells us plenty offline — but it does not license baking at 128.
# Shipping a checkpoint we KNOW is worse (-0.004 matmul cos at 128, -0.014 at
# 64), with the inferior width stamped into it forever, because the good one
# was inconvenient, is the same species of shortcut DOCTRINE D4 bans.
#
# And we do NOT fall back to a ~6.2 h exhaustive bake: that buys the previous
# generation at full price and leaves nothing for measurement — the exact
# mistake session 5 was killed for.
#
# On failure the box dies. Diagnosis happens on CPU, where the fixtures are
# free and a Viterbi fixture costs nothing but patience.
# =============================================================================
BAKE_W=$BEAM_WIDTH
if want S2; then
  mark "S2 beam-parity"
  FAIL_LOG=beam_parity.log
  ( cd "$ARC" && $S6_CARGO test -p mistralrs-quant --release --features cuda cuda_ -- --nocapture ) \
    > "$LOG_DIR/beam_parity.log" 2>&1
  rc=$?
  cp "$LOG_DIR/beam_parity.log" "$RESULTS/beam_parity.txt" 2>/dev/null
  grep -E "^test .*(cuda_beam|cuda_exhaustive|cuda_3d|cuda_search_plan).* \.\.\. " \
    "$LOG_DIR/beam_parity.log" | sed 's/^/BEAM_TEST /' || true
  grep -E "^test result:" "$LOG_DIR/beam_parity.log" | sed 's/^/BEAM_RESULT /' || true

  # Vacuity guards. The tests return Ok(()) silently when the kernels are not
  # compiled in, and with a message when no CUDA device answers.
  grep -q "CUDA not available; skipping" "$LOG_DIR/beam_parity.log" \
    && abort BEAM_VACUOUS "beam parity tests must actually run" "no 'CUDA not available; skipping' line" "the test binary printed it" "the parity tests SKIPPED — no CUDA device was visible to them, so their 'passes' mean nothing and a bake on this box would be unproven. results/beam_parity.txt"
  grep -qE "^test result: ok\..* 0 failed" "$LOG_DIR/beam_parity.log" || true

  if [ "$rc" = "0" ]; then
    echo "BEAM_PARITY[W=64]=PASS"; echo "BEAM_PARITY[W=128]=PASS"; echo "BEAM_PARITY[W=256]=PASS"
    BAKE_W=$BEAM_WIDTH
    echo "BEAM_GATE=W${BAKE_W} (bit-identical to the CPU beam at every width)"
  else
    FAILW=$(grep -oE '\bW=(64|128|256): CUDA beam differs' "$LOG_DIR/beam_parity.log" \
            | head -1 | grep -oE '[0-9]+')
    MISMATCH=$(sed -n 's/.*CUDA beam differs from the CPU beam in \([0-9]*\/[0-9]*\) bytes.*/\1/p' \
            "$LOG_DIR/beam_parity.log" | head -1)
    FAILED_TEST=$(grep -m1 -E '^test .* \.\.\. FAILED' "$LOG_DIR/beam_parity.log" | sed 's/^test //; s/ \.\.\. FAILED//')
    echo "BEAM_FAILING_WIDTH=${FAILW:-unknown}"
    grep -E "panicked at|assertion|differs from the CPU beam|^test result:" \
      "$LOG_DIR/beam_parity.log" | head -12
    abort BEAM_PARITY "CUDA beam == CPU beam, byte-identical, at W=$BEAM_WIDTH" \
      "0 mismatched bytes at W=64, W=128 and W=256" \
      "${FAILED_TEST:-cargo test exit $rc} failed at W=${FAILW:-unknown}${MISMATCH:+, ${MISMATCH} bytes differ}" \
      "the CUDA beam kernel does not reproduce the CPU beam on this silicon. W=$BEAM_WIDTH or no bake — there is no degraded rung and no exhaustive fallback. Reproduce on CPU for free: cargo test -p mistralrs-quant qtip::search_bench (the CPU beam is PR #29's beam_quantize_row) and diff against kernels/qtip/qtip_beam.cu. Full test output: results/beam_parity.txt"
  fi
  echo "$BAKE_W" > "$RESULTS/beam_width_used.txt"
elif [ -s "$RESULTS/beam_width_used.txt" ]; then
  BAKE_W=$(cat "$RESULTS/beam_width_used.txt")
fi

# =============================================================================
# S3 — BAKE with the beam kernel.  projected 31-62 min at W=256.
#
# Gates, in order of how much money each one has already cost us:
#   * bake header MUST print. The GPU fast path returned before
#     log_bake_header until PR #33, so no GPU bake has ever emitted one. If it
#     is absent, this binary predates the fix and we cannot prove what search
#     ran -> ABORT (DOCTRINE D4: the artifact must be self-describing).
#   * header must name the beam we asked for, never a substituted search.
#   * no "QTIP GPU quantize fallback" line (that is the ~11 min/layer CPU crawl).
#   * ISQ thread policy must read 1 (24 host threads on one device = 4-9
#     min/layer with no warning; session-5 trap).
#   * pace: at 3 min >= 2 layers, at 10 min >= 6 layers. Calibrated for the
#     beam's projected 42-85 s/layer, NOT for exhaustive's 510 s/layer.
# =============================================================================
if want S3; then
  mark "S3 bake W=$BAKE_W"
  FAIL_LOG=bake.log
  [ "$BAKE_W" = "$BEAM_WIDTH" ] || abort BAKE "bake width" "W=$BEAM_WIDTH" "W=$BAKE_W" "refusing to bake at anything but W=$BEAM_WIDTH — there is no degraded rung"
  mkdir -p "$UQFF_DIR"
  export ARC_QTIP_BEAM=$BAKE_W
  export MISTRALRS_ISQ_SINGLETHREAD=1
  nohup env ARC_QTIP_BEAM="$BAKE_W" MISTRALRS_ISQ_SINGLETHREAD=1 \
    "$S6_MISTRALRS" quantize text -m "$V4_DIR" -a deepseekv4 --isq qtip2 \
    -o "$UQFF_DIR/" --uqff-base-model "$BASE_MODEL" --uqff-repo-id "$HF_REPO" \
    > "$LOG_DIR/bake.log" 2>&1 < /dev/null &
  BAKE_PID=$!
  BAKE_T0=$(date +%s)
  echo "BAKE_PID=$BAKE_PID"
  arm_sentinel "$LOG_DIR/bake.log" 1200 "$BAKE_PID" bake

  # POLL for the header rather than sleeping 3 minutes blind: on a healthy box
  # it appears within seconds, so a missing header aborts in ~10 s instead of
  # burning 3 minutes of rental before we learn the binary is unprovenanced.
  HDR_DEADLINE=180
  [ "$DRYRUN" = "1" ] && HDR_DEADLINE=20
  HDR=""
  while [ $(( $(date +%s) - BAKE_T0 )) -lt "$HDR_DEADLINE" ]; do
    HDR=$(grep -m1 "QTIP bake \[" "$LOG_DIR/bake.log" 2>/dev/null)
    [ -n "$HDR" ] && break
    # The process exiting is not by itself an error (a tiny fixture bake ends
    # fast); read the log once more before judging.
    kill -0 "$BAKE_PID" 2>/dev/null || { sleep 1; HDR=$(grep -m1 "QTIP bake \[" "$LOG_DIR/bake.log" 2>/dev/null); break; }
    sleep 2
  done
  if [ -z "$HDR" ] && rebuild_from_source_if_cached "no bake header"; then
    kill_pid "$BAKE_PID"
    echo "RETRY_BAKE once with the freshly built binary"
    nohup env ARC_QTIP_BEAM="$BAKE_W" MISTRALRS_ISQ_SINGLETHREAD=1 \
      "$S6_MISTRALRS" quantize text -m "$V4_DIR" -a deepseekv4 --isq qtip2 \
      -o "$UQFF_DIR/" --uqff-base-model "$BASE_MODEL" --uqff-repo-id "$HF_REPO" \
      >> "$LOG_DIR/bake.log" 2>&1 < /dev/null &
    BAKE_PID=$!; BAKE_T0=$(date +%s)
    nap 20
    HDR=$(grep -m1 "QTIP bake \[" "$LOG_DIR/bake.log" 2>/dev/null)
  fi
  if [ -z "$HDR" ]; then
    kill_pid "$BAKE_PID"
    abort BAKE_HEADER "bake log contains 'QTIP bake ['" "a header line within ${HDR_DEADLINE}s" "no header (log is $(wc -c < "$LOG_DIR/bake.log" 2>/dev/null | tr -d ' ') bytes)" "before PR #33 the GPU path returned BEFORE log_bake_header (mistralrs-quant/src/qtip/mod.rs:1437,:1653), so this binary cannot prove which search produced the checkpoint. Refusing to bake an unprovenanced artifact (D4). Check offline that the built tree really contains #33. results/bake.log"
  fi
  echo "BAKE_HEADER: $HDR"
  case "$HDR" in
    *"search=viterbi-beam(W=$BAKE_W)"*) echo "BAKE_SEARCH_OK W=$BAKE_W" ;;
    *"search=greedy"*) kill_pid "$BAKE_PID"; abort BAKE_GREEDY "bake header search field" "search=viterbi-beam(W=$BAKE_W)" "$HDR" "the bake selected GREEDY. DOCTRINE D4: greedy is banned forever, structurally — no env, flag or config path may reach it and a bake handed it must hard-error. This is a CODE REGRESSION in the ban (mistralrs-quant/src/qtip/mod.rs QtipRotation::for_mode / greedy_ban_tests.rs), not a box problem. results/bake.log" ;;
    *) kill_pid "$BAKE_PID"; abort BAKE_SEARCH "bake header search field" "search=viterbi-beam(W=$BAKE_W)" "$HDR" "the header names a different search than ARC_QTIP_BEAM=$BAKE_W requested. cuda_search_plan (qtip/mod.rs:714) may TRANSLATE (a beam >= 2^L runs the exhaustive kernel) but must never SUBSTITUTE. results/bake.log" ;;
  esac
  case "$HDR" in
    *"rotation=hadamard-"*) echo "BAKE_ROTATION_OK" ;;
    *) kill_pid "$BAKE_PID"; abort BAKE_ROTATION "bake header rotation field" "rotation=hadamard-N" "$HDR" "rotation is OFF. D11 makes the block-diagonal Hadamard the permanent default, its signs are serialized into the UQFF and consumed by the forward path, and a no-rotation artifact is a different (worse) product (fp4-sourced experts: 0.887 vs 0.963 matmul cos). results/bake.log" ;;
  esac

  # `grep -c` EXITS 1 on zero matches, and `set -o pipefail` makes a
  # `grep -c ... | grep -qx 0` pipeline report that 1 — which reads as "the
  # fallback fired". Count into a variable instead; the dry run caught this.
  FB=$(grep -c "QTIP GPU quantize fallback" "$LOG_DIR/bake.log" 2>/dev/null || true)
  if [ "${FB:-0}" -gt 0 ]; then
    grep -m1 "QTIP GPU quantize fallback" "$LOG_DIR/bake.log"
    kill_pid "$BAKE_PID"
    abort BAKE_FALLBACK "QTIP GPU quantize fallback count" "0" "$FB" "the bake fell back to the CPU Viterbi pipeline (~11 min/layer ~= 8 h). The warn line printed above names the exact condition and IS the deliverable — read it in results/bake.log and fix the named cause on CPU."
  fi
  ISQ_LINE=$(grep -m1 -E "ISQ thread policy|Applying (immediate )?ISQ .*threads" "$LOG_DIR/bake.log")
  echo "BAKE_ISQ_THREADS: ${ISQ_LINE:-<none>}"
  case "$ISQ_LINE" in
    *"1 thread"*) : ;;
    "") echo "BAKE_ISQ_WARN no thread-policy line yet" ;;
    *) echo "BAKE_ISQ_WARN not 1 thread — MISTRALRS_ISQ_SINGLETHREAD=1 was exported; if the pace gate below fails this is why" ;;
  esac

  # The pace gate is only meaningful AT the 3-minute mark, and the header poll
  # above may have returned in seconds. Close the gap before counting.
  if [ "$DRYRUN" != "1" ]; then
    while [ $(( $(date +%s) - BAKE_T0 )) -lt 180 ]; do sleep 10; done
  fi
  L3=$(grep -c "Detected INT4" "$LOG_DIR/bake.log")
  echo "PACE[t=3m]=$L3 layers"
  if [ "$DRYRUN" != "1" ] && [ "${L3:-0}" -lt 2 ]; then
    kill_pid "$BAKE_PID"
    abort BAKE_PACE "'Detected INT4' lines at t=3min" ">= 2 (beam projection 42-85 s/layer)" "$L3" "bake pace too slow at 3 min. Either the beam kernel is not really engaged (check results/bake.log for the header width) or the box is transfer-starved like s5a (99% util at 132W of 700W). Compare results/box_health.json power against the 200W floor offline."
  fi
  if [ "$DRYRUN" != "1" ]; then
    nap 420
    L10=$(grep -c "Detected INT4" "$LOG_DIR/bake.log")
    echo "PACE[t=10m]=$L10 layers"
    [ "${L10:-0}" -ge 6 ] || { kill_pid "$BAKE_PID"; abort BAKE_PACE "'Detected INT4' lines at t=10min" ">= 6" "$L10" "projected 44-layer bake > 75 min against a 31-62 min projection. results/bake.log + results/box_health.json"; }
  fi

  waited=0
  while kill -0 "$BAKE_PID" 2>/dev/null; do
    nap 30; waited=$((waited + 30))
    if past "$WIRE_BAKE_H"; then
      kill_pid "$BAKE_PID"
      abort BAKE_TRIPWIRE "bake completes before the anti-hang wire" "done by ${WIRE_BAKE_H}h cumulative" "still running at $(elapsed_h)h" "the bake overran its wire; it is hung or the box is starved. results/bake.log has the last layer line — count layers/min offline."
    fi
    [ "$DRYRUN" = "1" ] && break
  done
  BYTES=$(du -sb "$UQFF_DIR" 2>/dev/null | cut -f1)
  SHARDS=$(ls "$UQFF_DIR"/*.uqff 2>/dev/null | wc -l | tr -d ' ')
  echo "BAKE_OK bytes=${BYTES:-0} shards=${SHARDS:-0} layers=$(grep -c 'Detected INT4' "$LOG_DIR/bake.log")"
  grep -E "QTIP bake \[|ISQ thread policy|QTIP GPU quantize fallback" "$LOG_DIR/bake.log" \
    > "$RESULTS/bake_log_excerpt.txt" 2>/dev/null
  if [ "$DRYRUN" != "1" ]; then
    [ "${SHARDS:-0}" -ge 1 ] || abort BAKE_SIZE "*.uqff shards in $UQFF_DIR" ">= 1" "0" "the bake exited without writing a shard. results/bake.log"
    awk -v b="${BYTES:-0}" 'BEGIN{exit !(b > 40e9 && b < 120e9)}' \
      || abort BAKE_SIZE "uqff directory size" "40-120 GB (expect ~68 GB / 7 shards)" "$(awk -v b="${BYTES:-0}" 'BEGIN{printf "%.1f GB", b/1e9}') / ${SHARDS:-0} shards" "the artifact is the wrong size — do not publish it. results/bake.log"
  fi
  unset ARC_QTIP_BEAM MISTRALRS_ISQ_SINGLETHREAD
fi

# =============================================================================
# S4 — UPLOAD THE UQFF.  ~25 min.  THE HIGHEST-VALUE IRREVERSIBLE STEP.
# After this, nobody ever bakes again (FACTS.md: "bake ONCE and reuse the
# UQFF"). It runs BEFORE every measurement so a session that dies at S7 still
# leaves the artifact behind.
# Token: 0600 file, never echoed, never in the status page, deleted at S12.
# =============================================================================
if want S4; then
  mark "S4 upload-uqff"
  FAIL_LOG=upload.log
  MANIFEST="$UQFF_DIR/arc_bake_manifest.json"
  {
    printf '{\n'
    printf '  "produced_by": "arc-tools/quality/s6_driver.sh (GPU session 6)",\n'
    printf '  "arc_commit": "%s",\n' "$(cd "$ARC" && git rev-parse HEAD 2>/dev/null)"
    printf '  "base_model": "%s",\n' "$BASE_MODEL"
    printf '  "isq": "qtip2",\n'
    printf '  "beam_width": %s,\n' "$BAKE_W"
    printf '  "bake_header": "%s",\n' "$(grep -m1 'QTIP bake \[' "$LOG_DIR/bake.log" 2>/dev/null | sed 's/"/\\"/g')"
    printf '  "gpu": "%s",\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    printf '  "utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '}\n'
  } > "$MANIFEST"
  cp "$MANIFEST" "$RESULTS/arc_bake_manifest.json" 2>/dev/null

  if [ ! -s "$S6_TOKEN_FILE" ]; then
    skipm UPLOAD "no token at $S6_TOKEN_FILE — file_upload it (chmod 600) and re-run with S6_ONLY=S4. THE BAKE IS ONLY WORTH SOMETHING IF IT IS PUBLISHED."
  else
    chmod 600 "$S6_TOKEN_FILE" 2>/dev/null
    set +x
    $S6_PY "$S6_UPLOAD_PY" --folder "$UQFF_DIR" --repo-id "$HF_REPO" \
      --token-file "$S6_TOKEN_FILE" --private \
      --commit-message "Arc session-6 qtip2 bake (beam W=$BAKE_W)" \
      2>&1 | grep -v -i "token" | tee "$LOG_DIR/upload.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" != "0" ]; then
      echo "UPLOAD_FAIL rc=$rc — RETRY ONCE (upload_folder resumes, it does not restart)"
      $S6_PY "$S6_UPLOAD_PY" --folder "$UQFF_DIR" --repo-id "$HF_REPO" \
        --token-file "$S6_TOKEN_FILE" --private \
        --commit-message "Arc session-6 qtip2 bake (beam W=$BAKE_W) retry" \
        2>&1 | grep -v -i "token" | tee -a "$LOG_DIR/upload.log"
      rc=${PIPESTATUS[0]}
      [ "$rc" = "0" ] || echo "UPLOAD_FAIL_FINAL rc=$rc — continue the session, but the bake is NOT published; retry from the driving machine before teardown"
    fi
    UP=$(grep -c "UPLOAD_OK" "$LOG_DIR/upload.log" 2>/dev/null || true)
    [ "${UP:-0}" -gt 0 ] && echo "UPLOAD_VERIFIED (every local file listed back from the hub)"
  fi
  # Mark the phase so the pause banner quotes the cheaper re-entry arithmetic.
  grep -q "UPLOAD_OK" "$LOG_DIR/upload.log" 2>/dev/null && touch "$RESULTS/.uploaded"

  # Push the binary to the cache while the token is already in play. ~200 MB,
  # seconds, and it is what turns a future ~70 min re-entry into ~26 min. Skip
  # when this binary CAME from the cache (nothing new to publish).
  if [ -s "$S6_TOKEN_FILE" ] && [ "${BIN_FROM_CACHE:-0}" != "1" ]; then
    $S6_PY "$S6_BINCACHE_PY" push --repo-id "$HF_REPO" --prefix "$BIN_CACHE_PREFIX" \
      --token-file "$S6_TOKEN_FILE" --arc "$ARC" --binary "$S6_MISTRALRS" \
      --built-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      2>&1 | grep -v -i token | tee -a "$LOG_DIR/bincache.log" | tail -2
  fi

  if past "$WIRE_UPLOAD_H"; then
    echo "TRIPWIRE past $WIRE_UPLOAD_H h at end of S4 — expect to lose S11 (voting truncates)"
  fi
fi

# =============================================================================
# S5 — Serve + THE HEADLINE: full-serving batch sweep.  ~20 min.
# DOCTRINE D2: production is B=32-128. Aggregate tok/s, per-user AT batch,
# TTFT, $/Mtok. NEVER measured end-to-end before — s4's 1,006 tok/s was a
# kernel microbench, not the engine.
# =============================================================================
if want S5; then
  mark "S5 serve + batch sweep"
  FAIL_LOG=serve_baseline.log
  if ! start_serve serve_baseline; then
    rebuild_from_source_if_cached "server never came up" && start_serve serve_baseline
  fi
  [ -n "$SERVE_PID" ] || abort SERVE "GET /health on port $PORT" "200 within 900s of launch" "no healthy response" "the server never came up on the freshly baked UQFF (and a rebuild off the binary cache did not help, if one applied). results/serve_baseline.log + results/bincache.log"
  ( cd "$Q" && $S6_PY "$Q/run_coherence.py" --skip-facts ) 2>&1 | tail -4 | tee "$LOG_DIR/coherence.log"
  ( cd "$Q" && $S6_PY "$Q/speed_probe.py" --label s6_b1_diag ) 2>&1 | tail -3 | tee "$LOG_DIR/speed_b1.log"
  SWEEP_ARGS="--label s6_baseline --batches $BATCHES"
  past "$WIRE_LASTCHANCE_H" || SWEEP_ARGS="$SWEEP_ARGS --include-128"
  # shellcheck disable=SC2086
  ( cd "$Q" && $S6_PY "$Q/batch_load_probe.py" $SWEEP_ARGS ) 2>&1 | tee "$LOG_DIR/batch_baseline.log"
  grep -E "^BATCH\[|^BATCHSWEEP\[" "$LOG_DIR/batch_baseline.log" || echo "FAIL: batch probe produced no BATCH lines"
fi

# =============================================================================
# S6 — GSM8K n=100, 2048-cap, 0-shot greedy.  ~75 min.
# 87.0% is PROVISIONAL: PR #35 changed decode math on 4/5 expert paths
# (SwiGLU clamp, incl. the shared expert every token traverses) and moved YaRN
# off ratio-0 layers {0,1,43}. A number either side of 87.0 is NOT a
# regression — it is the first number measured on math that matches the
# reference. Scored evals: ONE request at a time.
# =============================================================================
if want S6; then
  mark "S6 gsm8k baseline (re-measure the provisional 87.0)"
  ( cd "$Q" && $S6_PY "$Q/run_gsm8k.py" --n "$GSM8K_N" --max-tokens "$GSM8K_MAXTOK" ) \
    > "$LOG_DIR/gsm8k_plain.log" 2>&1 &
  GSM_PID=$!
  arm_sentinel "$LOG_DIR/gsm8k_plain.log" 1800 "$GSM_PID" gsm8k
  while kill -0 "$GSM_PID" 2>/dev/null; do
    nap 60
    past "$WIRE_STOP_H" && { kill_pid "$GSM_PID"; echo "TRIPWIRE gsm8k stopped at the $WIRE_STOP_H h wire (JSON keeps whatever n it reached)"; break; }
    [ "$DRYRUN" = "1" ] && break
  done
  grep -E "^GSM8K\[" "$LOG_DIR/gsm8k_plain.log" | tail -2 || echo "FAIL: no GSM8K line"
fi

# =============================================================================
# S7 — gen-2 GEMV sweep, 98 variants (44 gen-1 + 54 gen-2).  ~20 min.
# Server DOWN (perplexity/bench and serve cannot share the device).
# NOTE: the header documents `gen1`/`gen2`/id-lists for ARC_TUNE_VARIANTS.
# There is NO `all` keyword — an unrecognized spec falls back to the full grid
# with a warning. UNSET the variable to sweep everything cleanly.
# =============================================================================
if want S7; then
  mark "S7 gen-2 gemv sweep"
  stop_serve
  unset ARC_TUNE_VARIANTS
  ( cd "$ARC" && $S6_CARGO run --release -p mistralrs-quant --example qtip_gemv_tune --features cuda ) \
    > "$LOG_DIR/gemv_sweep.log" 2>&1
  rc=$?
  cp "$LOG_DIR/gemv_sweep.log" "$RESULTS/gemv_tune_sweep.txt" 2>/dev/null
  grep -E "^=== qtip2b GEMV autotune|WINNER \[" "$LOG_DIR/gemv_sweep.log" | head -6
  if [ "$rc" != "0" ]; then
    skipm GEMV_SWEEP "sweep exited $rc — keep the baked defaults and move on (no debugging)"
  else
    # Parse the marker line, NEVER glob: a session-4 glob missed the filename
    # and the 'tuned' serve silently ran without the table.
    TUNE_TABLE=$(grep '^WINNER_TABLE_WRITTEN: ' "$LOG_DIR/gemv_sweep.log" | tail -1 | cut -d' ' -f2)
    if [ -s "${TUNE_TABLE:-/nonexistent}" ]; then
      cp "$TUNE_TABLE" "$RESULTS/gemv_tune_winners.json"
      export ARC_QTIP_TUNE_TABLE="$RESULTS/gemv_tune_winners.json"
      echo "WINNERS: $ARC_QTIP_TUNE_TABLE"
    else
      skipm TUNE_TABLE "no WINNER_TABLE_WRITTEN line (or empty file) — S9 will measure the BAKED defaults, which is still a valid row"
    fi
  fi
fi

# =============================================================================
# S8 — Calibration probe: the diag(H) channel spread of a V4 expert layer.
# Closes DOCTRINE D11's open question: below ~1e4 shuts the no-rotation +
# Hessian door, above ~1e5 reopens it.
#
# HARD RISK, stated up front: `mistralrs calibrate` loads the model
# UNQUANTIZED (calibrate.rs sets from_uqff: None and normal.rs:644 puts a
# calibration run on the regular device with loading_isq=false). The V4
# checkpoint is ~149 GB on disk against 141 GB of HBM. It may not fit. That is
# WHY this step is time-boxed and why the OOM is itself a recordable finding
# ("Arc cannot calibrate its own flagship on one box; `calibrate` has no
# --from-uqff"). Do NOT debug it here.
# =============================================================================
if want S8; then
  mark "S8 calibration probe"
  stop_serve
  WBYTES=$(du -sb "$V4_DIR" 2>/dev/null | cut -f1)
  HBM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "CALIB_PREFLIGHT weights_bytes=${WBYTES:-0} hbm_mib=${HBM_MIB:-0}"
  CAL="$RESULTS/v4_expert_stats.arccalib"
  ( cd "$ARC" && timeout "$CALIB_BOX_SECS" "$S6_MISTRALRS" calibrate text \
      -m "$V4_DIR" -a deepseekv4 --samples "$CALIB_SAMPLES" --gram none \
      --out "$CAL" ) > "$LOG_DIR/calibrate.log" 2>&1
  rc=$?
  if [ "$rc" != "0" ] || [ ! -s "$CAL" ]; then
    grep -iE "out of memory|cuda error|failed|panicked" "$LOG_DIR/calibrate.log" | head -4
    skipm CALIB "calibrate exited $rc without an artifact. If the reason is OOM, that IS the finding: the flagship cannot be calibrated on one H200 because `calibrate` has no --from-uqff and loads unquantized. File it; do not debug."
  else
    ( cd "$ARC" && $S6_CARGO run --release -p mistralrs-quant --example stats_info -- "$CAL" --layers ) \
      > "$LOG_DIR/calib_layers.log" 2>&1
    cp "$LOG_DIR/calib_layers.log" "$RESULTS/calib_layers.txt" 2>/dev/null
    # An expert-stack layer: prefer a row whose `experts` column is populated,
    # else the widest in_features row. Columns: isq name in_feat tokens gram experts
    IDX=$(awk 'NF>=6 && $1 ~ /^[0-9]+$/ && $6 != "-" {print $1; exit}' "$LOG_DIR/calib_layers.log")
    [ -z "$IDX" ] && IDX=$(awk 'NF>=4 && $1 ~ /^[0-9]+$/ {if ($3+0 > m) {m=$3+0; k=$1}} END{print k}' "$LOG_DIR/calib_layers.log")
    if [ -n "$IDX" ]; then
      ( cd "$ARC" && $S6_CARGO run --release -p mistralrs-quant --example stats_info -- "$CAL" --layer "$IDX" ) \
        > "$LOG_DIR/calib_layer.log" 2>&1
      cp "$LOG_DIR/calib_layer.log" "$RESULTS/calib_layer_$IDX.txt" 2>/dev/null
      # stats_info prints: "diag/tokens  min ... max ...  (dynamic range 1181.0x)"
      # Capture the number with sed, not a second grep -oE: `[0-9.eE+]+` also
      # matches the "e" in "range" and yields two lines (dry run caught this).
      SPREAD=$(sed -n 's/.*dynamic range \([0-9.eE+-]*\)x.*/\1/p' "$LOG_DIR/calib_layer.log" | head -1)
      echo "CALIB_SPREAD layer=$IDX diag_h_dynamic_range=${SPREAD:-unknown}"
      echo "CALIB_VERDICT $(awk -v s="${SPREAD:-0}" 'BEGIN{ if (s+0 <= 0) print "unknown"; else if (s+0 < 1e4) print "BELOW 1e4 -> D11 stands: no-rotation+Hessian stays ruled out"; else if (s+0 > 1e5) print "ABOVE 1e5 -> D11 reopens: re-run the sensitivity sweep at this dispersion"; else print "1e4..1e5 -> inconclusive band, keep rotation, widen the fixture family"}')"
    else
      skipm CALIB_LAYER "could not pick an expert layer out of stats_info --layers"
    fi
  fi
fi

# =============================================================================
# S9 — Tuned serve + batch re-sweep.  ~12 min.  The before/after that turns
# a kernel win into a fleet number (kernel gains are ~Bx leveraged, D2).
# =============================================================================
if want S9; then
  mark "S9 tuned batch sweep"
  FAIL_LOG=serve_tuned.log
  if [ -z "${ARC_QTIP_TUNE_TABLE:-}" ]; then
    skipm TUNED_SWEEP "no winner table from S7 — the S5 sweep already covers the baked defaults"
  else
    start_serve serve_tuned || abort SERVE "GET /health on port $PORT (tuned)" "200 within 900s of launch" "no healthy response" "the server came up untuned in S5 but not with ARC_QTIP_TUNE_TABLE set — suspect the winner table. results/serve_tuned.log + results/gemv_tune_winners.json"
    # The env must reach the SERVE process, not just this shell (session-4 bug).
    tr '\0' '\n' < "/proc/$SERVE_PID/environ" 2>/dev/null | grep -q ARC_QTIP_TUNE_TABLE \
      && echo "TUNE_TABLE_IN_SERVE_ENV=yes" || echo "TUNE_TABLE_IN_SERVE_ENV=NO — the sweep result is NOT applied; label the numbers accordingly"
    ( cd "$Q" && $S6_PY "$Q/batch_load_probe.py" --label s6_tuned --batches "$BATCHES" ) \
      2>&1 | tee "$LOG_DIR/batch_tuned.log"
    grep -E "^BATCHSWEEP\[" "$LOG_DIR/batch_tuned.log" || true
    ( cd "$Q" && $S6_PY "$Q/speed_probe.py" --label s6_b1_tuned ) 2>&1 | tail -3
  fi
fi

# =============================================================================
# S10 — MTP acceptance: the first valid number ever.  ~20 min.
# Every prior MTP number is void (PR #30: the draft got embed(T0) for BOTH
# h_proj and e_proj over a KV cache that never saw the prompt). The telemetry
# call site comes from the S1 patch — master's log_acceptance_rate() has ZERO
# call sites, which is why sessions 3/4/5 measured nothing.
# =============================================================================
if want S10; then
  mark "S10 mtp acceptance"
  stop_serve
  ARC_MTP_LOG_ACCEPTANCE=1 start_serve serve_mtp --mtp-depth "$MTP_DEPTH" \
    || skipm MTP "mtp server never became healthy"
  if [ -n "$SERVE_PID" ]; then
    grep -E "MTP (speculative decode engaged|requested)" "$LOG_DIR/serve_mtp.log" | head -2
    if grep -q "has no MTP head" "$LOG_DIR/serve_mtp.log"; then
      skipm MTP "$(grep -m1 'has no MTP head' "$LOG_DIR/serve_mtp.log") — that line IS the deliverable"
    else
      ( cd "$Q" && $S6_PY "$Q/speed_probe.py" --label s6_mtp"$MTP_DEPTH" --tokens 256 ) \
        2>&1 | tail -3 | tee "$LOG_DIR/speed_mtp.log"
      N=$(grep -c "MTP acceptance" "$LOG_DIR/serve_mtp.log")
      if [ "${N:-0}" -ge 1 ]; then
        grep "MTP acceptance" "$LOG_DIR/serve_mtp.log" | tail -10 \
          | tee "$RESULTS/mtp_acceptance.txt"
        echo "MTP_ACCEPT $(grep 'MTP acceptance rate' "$LOG_DIR/serve_mtp.log" | tail -1)"
      else
        skipm MTP_TELEMETRY "0 acceptance lines after 256 decoded tokens: the S1 patch is not in this binary, or the env did not reach the serve process ($(tr '\0' '\n' < "/proc/$SERVE_PID/environ" 2>/dev/null | grep -c ARC_MTP_LOG_ACCEPTANCE) match). Record and move on."
      fi
    fi
  fi
fi

# =============================================================================
# S11 — Voting GSM8K k=5, confidence-weighted: the 90+ attempt (DOCTRINE D6).
# Unblocked by #28 (batch prefill had NO causality — every prior batched
# quality result is invalid) and #21 (per-seq xs_history). ~90 min at n=100.
# TRUNCATABLE, not droppable: the JSON is rewritten after every problem, so
# stopping at the wire keeps whatever n it reached. Keep n >= 60 (CI ~±12pp).
# =============================================================================
if want S11; then
  mark "S11 voting gsm8k k=$VOTES"
  stop_serve
  start_serve serve_vote || skipm VOTING "server never became healthy"
  if [ -n "$SERVE_PID" ]; then
    VN=$GSM8K_N
    past "$WIRE_LASTCHANCE_H" && VN=60 && echo "TRIPWIRE past $WIRE_LASTCHANCE_H h — voting starts at n=60"
    ( cd "$Q" && $S6_PY "$Q/run_gsm8k.py" --n "$VN" --max-tokens "$GSM8K_MAXTOK" \
        --votes "$VOTES" --vote-mode confidence_weighted ) \
      > "$LOG_DIR/gsm8k_votes.log" 2>&1 &
    V_PID=$!
    arm_sentinel "$LOG_DIR/gsm8k_votes.log" 1800 "$V_PID" voting
    while kill -0 "$V_PID" 2>/dev/null; do
      nap 60
      past "$WIRE_STOP_H" && { kill_pid "$V_PID"; echo "TRIPWIRE voting stopped at the $WIRE_STOP_H h wire; JSON keeps its partial n"; break; }
      [ "$DRYRUN" = "1" ] && break
    done
    grep -E "^GSM8K\[" "$LOG_DIR/gsm8k_votes.log" | tail -2 || echo "FAIL: no voting GSM8K line"
  fi
fi

# =============================================================================
# S12 — Results out + teardown prep.  NEVER CUT.
# =============================================================================
# S12 always runs at the end of a full chain (it is never a trip-wire cut), but
# a deliberate subset run (S6_ONLY=S4 to retry an upload, say) must NOT tar and
# shred the token out from under a session that is still going.
if [ -n "$S6_ONLY" ] && ! want S12; then
  echo "SKIP_S12 (S6_ONLY=$S6_ONLY) — results and token left untouched"
  budget_line
  exit 0
fi

mark "S12 teardown"
harvest "S6 complete"
tar tzf "$S6_ROOT/s6_results.tgz" 2>/dev/null | head -30
echo "S6_COMPLETE — now DELETE the instance (delete, not stop) and verify with list_instances."
