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
S6_TOKEN_FILE=${S6_TOKEN_FILE:-$S6_ROOT/.hf_token}

# ---------------------------------------------------------------- policy
ARC_BRANCH=${ARC_BRANCH:-master}
HF_REPO=${HF_REPO:-aeonmind/DeepSeek-V4-Flash-UQFF-qtip2}
BASE_MODEL=${BASE_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$ARC/chat_templates/deepseek_v4.json}
PORT=${PORT:-1234}
RATE_HR=${RATE_HR:-4.92}

BEAM_WIDTH=${BEAM_WIDTH:-256}       # requested; S2 may lower it to 128
BEAM_MIN_PUBLISHABLE=${BEAM_MIN_PUBLISHABLE:-128}
GSM8K_N=${GSM8K_N:-100}
GSM8K_MAXTOK=${GSM8K_MAXTOK:-2048}
VOTES=${VOTES:-5}
BATCHES=${BATCHES:-1,8,16,32,64}
MTP_DEPTH=${MTP_DEPTH:-2}
CALIB_SAMPLES=${CALIB_SAMPLES:-8}
CALIB_BOX_SECS=${CALIB_BOX_SECS:-1500}   # 25 min hard box on S8

# Trip-wires, cumulative hours from driver start (see the runbook's cost model).
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

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULTS" 2>/dev/null

# ---------------------------------------------------------------- helpers
mark()  { echo ":::::: $* ::::::"; date -u +%H:%M:%S; }
say()   { echo "$*"; }
abort() { echo "ABORT_$1 $2"; teardown_hint; exit 1; }
skipm() { echo "SKIP_$1 $2"; }

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

teardown_hint() {
  echo "TRIPWIRE teardown required: tar results, download, DELETE the instance."
  budget_line
}

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
  && abort ENV_DIRTY "one of the banned gates is exported — unset it and restart" \
  || echo "ENV_CLEAN"

# =============================================================================
# S0 — BOX HEALTH GATE, before the 149 GB download.  ~2 min.
# s5a: 99% util at 132 W of 700 W = starved on host<->device transfers, ~3
# min/layer with a binary proven at 30 s/layer. Cost ~1.5 h and ~$7.
# =============================================================================
if want S0; then
  mark "S0 box-health"
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "HEALTH_COMPUTE_CAP=$CC"
  # has_qtip_kernels is set by mistralrs-quant/build.rs iff compute_cap >= 8.0.
  # Below that, EVERY cuda parity test in S2 returns Ok(()) VACUOUSLY.
  awk -v c="${CC:-0}" 'BEGIN{exit !(c+0 >= 8.0)}' \
    || abort BOX_COMPUTE_CAP "compute_cap=$CC < 8.0 — qtip kernels would not compile in and S2 would pass vacuously"
  DRV=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9][0-9.]*' | grep -o '[0-9][0-9.]*$')
  TK=$(nvcc --version 2>/dev/null | grep -o 'release [0-9][0-9.]*' | grep -o '[0-9][0-9.]*$')
  echo "HEALTH_CUDA driver_max=$DRV toolkit=$TK"
  if [ -n "$DRV" ] && [ -n "$TK" ]; then
    NEWEST=$(printf '%s\n%s\n' "$DRV" "$TK" | sort -V | tail -1)
    [ "$TK" != "$DRV" ] && [ "$NEWEST" = "$TK" ] \
      && abort BOX_TOOLKIT "toolkit $TK newer than driver max $DRV — runtime UNSUPPORTED_PTX_VERSION. apt-get install cuda-toolkit-${DRV/./-} and re-export CUDA_HOME first"
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
    abort BOX_HEALTH "box_health_gate exit $rc — DELETE THIS INSTANCE AND RE-RENT a different box/region. Do NOT debug the rental; deleting costs \$0.08, s5a cost \$7. Nothing below has run, so nothing is lost."
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
  if [ "${S6_SKIP_BOOT:-0}" != "1" ]; then
    ARC_BRANCH=$ARC_BRANCH nohup bash "$S6_BOOT" > "$LOG_DIR/boot.log" 2>&1 < /dev/null &
    BOOT_PID=$!
    echo "BOOT_PID=$BOOT_PID"
    arm_sentinel "$LOG_DIR/boot.log" 1800 "$BOOT_PID" boot
    waited=0
    while [ "$waited" -lt 7200 ]; do
      grep -q "BOOTSTRAP_COMPLETE" "$LOG_DIR/boot.log" 2>/dev/null && break
      grep -q "^FAIL:" "$LOG_DIR/boot.log" 2>/dev/null \
        && abort BUILD "boot.sh reported: $(grep -m1 '^FAIL:' "$LOG_DIR/boot.log")"
      kill -0 "$BOOT_PID" 2>/dev/null || break
      nap 30; waited=$((waited + 30))
    done
    grep -q "BOOTSTRAP_COMPLETE" "$LOG_DIR/boot.log" 2>/dev/null \
      || abort BUILD "no BOOTSTRAP_COMPLETE after ${waited}s — see $LOG_DIR/boot.log"
  fi
  [ -x "$S6_MISTRALRS" ] || abort BUILD "$S6_MISTRALRS missing after bootstrap"

  # MTP acceptance telemetry. VERIFIED 2026-08-14: master's
  # log_acceptance_rate() still has ZERO call sites (mtp_pipeline.rs:690), so
  # S10 would measure nothing — the exact session-3 failure. The session-2-era
  # patch NO LONGER APPLIES (PR #30 rewrote the file); s6_mtp_acceptance_
  # telemetry.patch is regenerated against d6ceaf1ad and compiles clean.
  if git -C "$ARC" apply --check "$Q/patches/s6_mtp_acceptance_telemetry.patch" 2>/dev/null; then
    git -C "$ARC" apply "$Q/patches/s6_mtp_acceptance_telemetry.patch" \
      && echo "PATCH_OK s6_mtp_acceptance_telemetry"
    ( cd "$ARC" && $S6_CARGO build --release -p mistralrs-cli --features "cuda flash-attn" ) \
      2>&1 | tail -3 | tee -a "$LOG_DIR/build_patch.log"
    [ -x "$S6_MISTRALRS" ] || abort BUILD "rebuild after the MTP patch produced no binary"
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
# FALLBACK IS EXPLICIT AND NARROW. cuda_beam_matches_cpu_beam_bit_for_bit
# iterates W = 64 -> 128 -> 256 and fails at the FIRST bad width, naming it in
# the assert ("W=128: CUDA beam differs..."). So:
#   * all pass                -> bake at 256 (quality-neutral, PR #29)
#   * 256 fails, 128 passes   -> bake at 128 (-0.004 cos; acceptable, FLAGGED)
#   * only 64 passes          -> DO NOT BAKE. -0.014 cos is a real regression
#                                and the width is stamped into the artifact
#                                forever. Tear down; fix on CPU.
#   * none pass               -> tear down.
# In NO case do we fall back to a ~6.2 h exhaustive bake inside a paid session:
# that is $30 of a $48.54 balance for the old generation at full price, and it
# is exactly the mistake session 5 was killed for.
# =============================================================================
BAKE_W=$BEAM_WIDTH
if want S2; then
  mark "S2 beam-parity"
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
    && abort BEAM_VACUOUS "the parity tests skipped: no CUDA device visible to the test binary. The 'passes' mean nothing."
  grep -qE "^test result: ok\..* 0 failed" "$LOG_DIR/beam_parity.log" || true

  if [ "$rc" = "0" ]; then
    echo "BEAM_PARITY[W=64]=PASS"; echo "BEAM_PARITY[W=128]=PASS"; echo "BEAM_PARITY[W=256]=PASS"
    BAKE_W=$BEAM_WIDTH
    echo "BEAM_GATE=W${BAKE_W} (all widths bit-identical to the CPU beam)"
  else
    FAILW=$(grep -oE '\bW=(64|128|256): CUDA beam differs' "$LOG_DIR/beam_parity.log" \
            | head -1 | grep -oE '[0-9]+')
    echo "BEAM_FAILING_WIDTH=${FAILW:-unknown}"
    case "${FAILW:-0}" in
      256) BAKE_W=128 ;;
      128) BAKE_W=64  ;;
      *)   BAKE_W=0   ;;
    esac
    if [ "$BAKE_W" -lt "$BEAM_MIN_PUBLISHABLE" ]; then
      grep -E "panicked at|assertion|differs from the CPU beam|test result:" \
        "$LOG_DIR/beam_parity.log" | head -12
      abort BEAM_PARITY "CUDA beam parity failed at W=${FAILW:-all} (rc=$rc). The largest passing width (${BAKE_W}) is below the publishable floor ${BEAM_MIN_PUBLISHABLE} (W=64 costs -0.014 matmul cos vs exhaustive, and the width is STAMPED into the artifact forever). Per the runbook: do NOT fall back to a ~6.2 h exhaustive bake inside a paid session. Capture $LOG_DIR/beam_parity.log, tar, DELETE the instance, fix on CPU."
    fi
    echo "BEAM_GATE=W${BAKE_W} (DEGRADED: W=${FAILW} failed parity; W=${BAKE_W} costs ~-0.004 matmul cos vs exhaustive — FLAG THIS IN THE RESULTS)"
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
  [ "$BAKE_W" -ge "$BEAM_MIN_PUBLISHABLE" ] || abort BAKE "no publishable beam width from S2"
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
  if [ -z "$HDR" ]; then
    kill_pid "$BAKE_PID"
    abort BAKE_HEADER "no 'QTIP bake [...]' header in the first 3 min. Before PR #33 the GPU path returned BEFORE log_bake_header, so this binary cannot prove which search produced the checkpoint. Refusing to bake an unprovenanced artifact (D4)."
  fi
  echo "BAKE_HEADER: $HDR"
  case "$HDR" in
    *"search=viterbi-beam(W=$BAKE_W)"*) echo "BAKE_SEARCH_OK W=$BAKE_W" ;;
    *"search=greedy"*) kill_pid "$BAKE_PID"; abort BAKE_GREEDY "the bake header says greedy. DOCTRINE D4: greedy is banned forever, structurally. This is a code regression, not a session problem." ;;
    *) kill_pid "$BAKE_PID"; abort BAKE_SEARCH "header does not name viterbi-beam(W=$BAKE_W): '$HDR'. cuda_search_plan may translate but must never substitute a search." ;;
  esac
  case "$HDR" in
    *"rotation=hadamard-"*) echo "BAKE_ROTATION_OK" ;;
    *) kill_pid "$BAKE_PID"; abort BAKE_ROTATION "header says rotation off — D11 makes Hadamard rotation the permanent default; a no-rotation artifact is a different (worse) product" ;;
  esac

  # `grep -c` EXITS 1 on zero matches, and `set -o pipefail` makes a
  # `grep -c ... | grep -qx 0` pipeline report that 1 — which reads as "the
  # fallback fired". Count into a variable instead; the dry run caught this.
  FB=$(grep -c "QTIP GPU quantize fallback" "$LOG_DIR/bake.log" 2>/dev/null || true)
  if [ "${FB:-0}" -gt 0 ]; then
    grep -m1 "QTIP GPU quantize fallback" "$LOG_DIR/bake.log"
    kill_pid "$BAKE_PID"
    abort BAKE_FALLBACK "the bake fell back to the CPU pipeline (~11 min/layer = ~8 h). The warn line above names the condition and IS the deliverable."
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
    abort BAKE_PACE "only $L3 layer(s) in 3 min. The beam projection is 42-85 s/layer (2-4 layers per 3 min); <2 means the beam is not really engaged or the box is starved. Re-run box_health_gate.sh --with-bake-probe; on FAIL delete and re-rent."
  fi
  if [ "$DRYRUN" != "1" ]; then
    nap 420
    L10=$(grep -c "Detected INT4" "$LOG_DIR/bake.log")
    echo "PACE[t=10m]=$L10 layers"
    [ "${L10:-0}" -ge 6 ] || { kill_pid "$BAKE_PID"; abort BAKE_PACE "only $L10 layers in 10 min (need 6): projected 44-layer bake > 75 min, which breaks the cost model"; }
  fi

  waited=0
  while kill -0 "$BAKE_PID" 2>/dev/null; do
    nap 30; waited=$((waited + 30))
    if past "$WIRE_BAKE_H"; then
      kill_pid "$BAKE_PID"
      abort BAKE_TRIPWIRE "bake still running at the $WIRE_BAKE_H h trip-wire. $(budget_line)"
    fi
    [ "$DRYRUN" = "1" ] && break
  done
  BYTES=$(du -sb "$UQFF_DIR" 2>/dev/null | cut -f1)
  SHARDS=$(ls "$UQFF_DIR"/*.uqff 2>/dev/null | wc -l | tr -d ' ')
  echo "BAKE_OK bytes=${BYTES:-0} shards=${SHARDS:-0} layers=$(grep -c 'Detected INT4' "$LOG_DIR/bake.log")"
  grep -E "QTIP bake \[|ISQ thread policy|QTIP GPU quantize fallback" "$LOG_DIR/bake.log" \
    > "$RESULTS/bake_log_excerpt.txt" 2>/dev/null
  if [ "$DRYRUN" != "1" ]; then
    [ "${SHARDS:-0}" -ge 1 ] || abort BAKE_SIZE "no .uqff shards written"
    awk -v b="${BYTES:-0}" 'BEGIN{exit !(b > 40e9 && b < 120e9)}' \
      || abort BAKE_SIZE "uqff dir is ${BYTES} bytes; expected ~68 GB (40-120 GB band)"
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
  start_serve serve_baseline || abort SERVE "server never became healthy — see $LOG_DIR/serve_baseline.log"
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
  if [ -z "${ARC_QTIP_TUNE_TABLE:-}" ]; then
    skipm TUNED_SWEEP "no winner table from S7 — the S5 sweep already covers the baked defaults"
  else
    start_serve serve_tuned || abort SERVE "tuned server never became healthy"
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
mark "S12 teardown"
stop_serve
cp "$LOG_DIR"/*.log "$RESULTS/" 2>/dev/null
( cd "$Q" && tar czf "$S6_ROOT/s6_results.tgz" results/ ) 2>/dev/null
cp "$S6_ROOT/s6_results.tgz" "$STATUS_DIR/" 2>/dev/null
ls -l "$STATUS_DIR/s6_results.tgz" 2>/dev/null
echo "RESULTS_TGZ: $STATUS_DIR/s6_results.tgz"
tar tzf "$S6_ROOT/s6_results.tgz" 2>/dev/null | head -30

# The token must not outlive the session (session-5: a token was shared in
# plaintext and had to be rotated).
if [ -f "$S6_TOKEN_FILE" ]; then
  shred -u "$S6_TOKEN_FILE" 2>/dev/null || rm -f "$S6_TOKEN_FILE"
  [ -f "$S6_TOKEN_FILE" ] && echo "FAIL: token file still present" || echo "TOKEN_DELETED"
fi

budget_line
echo "S6_COMPLETE — now DELETE the instance (delete, not stop) and verify with list_instances."
