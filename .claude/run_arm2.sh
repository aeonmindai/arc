#!/bin/bash
# run_arm2.sh <armname> <ragged 0|1> <tokens> [expect]
#
# Same binary, same commit, same model, same prompt geometry as the 02:2x run.
# EXACTLY ONE variable changes: --tokens (64 -> $TOKENS).
#
# Hypothesis under test: peak decode residency is NOT a constant 8. It is
#   N_resident = admission_rate x sequence_lifetime   (Little's Law)
# where admission_rate is set by prompt bucketing (9 distinct prompt lengths ->
# 9 prompt buckets, one bucket per engine step, ~11 s each at V4's 11.8 ms per
# prompt token) and sequence_lifetime = tokens / per_user_rate.
# At tokens=64: 0.29 seq/s x (64/2.66 s) = 7.0  -> observed peak 8.
# Multiplying lifetime by 8 must move residency. If it does not, there IS a cap.
set -uo pipefail
ARM=$1; RAGGED=$2; TOKENS=$3; EXPECT_ARG=${4:-}
PORT=18234
BIN=/root/budget-chain/arc/target/release/mistralrs
SRC=/root/models/v4-src
UQFF=/root/models/v4
LOG=/root/logs/serve_${ARM}.log
RES=/root/results/${ARM}.json
REPO=/root/budget-chain/arc

echo "=== ARM=$ARM RAGGED=$RAGGED TOKENS=$TOKENS $(date -u +%H:%M:%S) ==="
echo "BIN_SHA256=$(sha256sum $BIN | cut -d' ' -f1)"
echo "GIT_SHA=$(git -C $REPO rev-parse HEAD)"

# Bracket trick: the pattern '[m]istralrs' matches the string 'mistralrs' but
# this script's own /proc cmdline contains '[m]istralrs', which does not.
# `pkill -f` otherwise kills the shell that launched it -- that happened three
# times in the previous session.
pkill -f '[m]istralrs serve -p '"$PORT" 2>/dev/null; sleep 5
: > "$LOG"

# Set in BOTH arms, so the A/B differs only in the ragged flags.
# `prefill_starvation_floor()` (default_scheduler.rs:222) is resolved ONCE at
# FixedBucketingManager::new(), so it must be in the process env at launch.
# Without it, `select_running_bucket` gives the step to whichever bucket has the
# highest SUMMED priority, and a decode cohort of N outscores a 3-4 sequence
# prompt bucket by ~N*log2(L) -- so admission slows down as the cohort grows.
# That negative feedback loop is what pins residency, not any constant.
export ARC_PREFILL_FLOOR_STEPS=${FLOOR:-1}
echo "FLOOR: ARC_PREFILL_FLOOR_STEPS=$ARC_PREFILL_FLOOR_STEPS (both arms)"

if [ "$RAGGED" = "1" ]; then
  export ARC_V4_XS_PER_SEQ=1
  export ARC_MTP_PER_SEQ_KV=1
  echo "FLAGS: ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1 (both, set BEFORE launch)"
else
  unset ARC_V4_XS_PER_SEQ ARC_MTP_PER_SEQ_KV
  echo "FLAGS: none (baseline control)"
fi

setsid nohup stdbuf -oL "$BIN" serve -p $PORT -m "$SRC" -a deepseekv4 \
  --from-uqff "$UQFF/qtip2-0.uqff" --max-seqs 64 --prefix-cache-n 0 \
  </dev/null >>"$LOG" 2>&1 &
sleep 2
PID=$(pgrep -f '[m]istralrs serve -p '"$PORT" | head -1)
echo "SERVER_PID=$PID"

for i in $(seq 1 240); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "READY after $((i*5))s"; break; fi
  kill -0 "$PID" 2>/dev/null || { echo "FATAL: server died during load"; tail -30 "$LOG"; exit 2; }
  sleep 5
done

echo "--- /proc/$PID/environ ---"
tr '\0' '\n' < /proc/$PID/environ | grep -E '^ARC_(V4_XS_PER_SEQ|MTP_PER_SEQ_KV)=' || echo "(no ARC ragged flags in process env)"
echo "EXCL[pre]: compute-apps=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | tr '\n' ',')] mypid=$PID"

cd $REPO
PIDARG=""
[ "$RAGGED" = "1" ] && PIDARG="--pid $PID"
EXPECT=""
[ -n "$EXPECT_ARG" ] && EXPECT="--expect-running $EXPECT_ARG"

stdbuf -oL python3 arc-tools/ragged_v4_batch_sweep.py --port $PORT --log "$LOG" \
  $PIDARG $EXPECT --batches 1,32 --tokens "$TOKENS" --distinct 9 --out "$RES"
SWEEP_RC=$?
echo "SWEEP_RC=$SWEEP_RC"
echo "EXCL[post]: compute-apps=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | tr '\n' ',')]"

echo "--- residency, chronological ---"
grep -oE '[0-9]+ running, [0-9]+ waiting' "$LOG" | cat -n
echo "--- ArcSched/ArcKV instrumentation ---"
grep -a 'ragged-decode capability resolved' "$LOG" | sed -e 's/\x1b\[[0-9;]*m//g' | tail -5
grep -a 'decode bucketing' "$LOG" | sed -e 's/\x1b\[[0-9;]*m//g' | tail -15
echo "BEACON_COUNT=$(grep -c ARC-RAGGED-MASK-ENGAGED "$LOG")"
pkill -f '[m]istralrs serve -p '"$PORT" 2>/dev/null
echo "=== ARM $ARM COMPLETE rc=$SWEEP_RC $(date -u +%H:%M:%S) ==="
