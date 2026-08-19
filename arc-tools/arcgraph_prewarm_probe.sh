#!/bin/bash
# Is the surviving capture-time allocation the thing that corrupts the heap?
#
# CONTROL   : capture as-is            -> expect the MISS line, expect the crash
# TREATMENT : pre-warm that exact size -> expect NO MISS line; does it survive?
#
# ⚠️ The treatment is a PROBE, NOT A FIX. Pre-warming a size hides the symptom;
# the real fix is that the buffer stops being allocated per decode step. A green
# treatment arm must not be merged as a solution.
#
# GUARDED BOTH DIRECTIONS, because absence alone proves nothing:
#   * the MISS line must be PRESENT in the control  -> the log would have shown it
#   * the MISS line must be ABSENT in the treatment -> the pre-warm actually took
# If the control does not show the miss, the run is UNPROVEN, not a pass.
#
# A treatment arm that still crashes is EQUALLY VALUABLE: it eliminates the last
# capture-time allocation and forces the search elsewhere. Reported as a result.
set -u
L=/root/logs/arcgraph-prewarm; mkdir -p $L; S=$L/STATUS.txt; : > $S
REPO=/root/arc; SRC=/root/models/v4-src; UQFF=/root/models/v4-uqff/qtip2b-0.uqff; PORT=1234
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $S; }
PID=""
cleanup(){ [ -n "$PID" ] && kill $PID 2>/dev/null; sleep 3; [ -n "$PID" ] && kill -9 $PID 2>/dev/null; pkill -f "mistralrs serve -p $PORT" 2>/dev/null; return 0; }
trap cleanup EXIT INT TERM

OCC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c .)
[ "${OCC:-0}" = "0" ] || { say "ENV_FAIL: $OCC foreign process(es) on the GPU"; say "RESULT: UNANSWERED"; exit 2; }
BIN=$REPO/target/release/mistralrs
grep -qa "ARC prewarm" $BIN || { say "ENV_FAIL: binary lacks the prewarm probe — stale build"; say "RESULT: UNANSWERED"; exit 2; }
say "binary carries the probe; mtime $(date -u -r $BIN +%H:%M:%S)"

leg(){ # $1 name, $2 extra env
  local n=$1 extra=$2 log=$L/$n.server.log
  say "=== leg $n ($extra) ==="
  cd $REPO || return 1
  # shellcheck disable=SC2086
  env $extra ARC_CAPTURE_STREAM=1 ARC_V4_CAPTURE_PROBE=1 ARC_CANDLE_ALLOC_CACHE=1 \
      ARC_GRAPH_WARMUP=8 ARC_GRAPH_DEFERRED_PASSES=3 \
      $BIN serve -p $PORT -m $SRC -a deepseekv4 --from-uqff $UQFF \
      --chat-template chat_templates/deepseek_v4.json \
      --prefix-cache-n 0 --paged-attn off --max-seqs 1 --max-seq-len 4096 >$log 2>&1 &
  PID=$!
  local ok=0
  for _ in $(seq 1 90); do curl -fsS --max-time 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $PID 2>/dev/null || break; sleep 5; done
  [ $ok = 1 ] || { say "  server never healthy"; cleanup; PID=""; return 1; }
  curl -s --max-time 600 localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"Count from one to twenty in words, then stop.","max_tokens":24,"temperature":0}' >$L/$n.resp.json 2>&1
  local tok; tok=$(python3 -c "import json;print(json.load(open('$L/$n.resp.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  local alive="DIED"; kill -0 $PID 2>/dev/null && alive="ALIVE"
  local miss; miss=$(awk '/capture started for batch_size/{f=1} f && /MISS during capture/' $log 2>/dev/null | wc -l | tr -d ' ')
  local glibc; glibc=$(grep -ciE 'malloc_consolidate|free\(\):|malloc\(\):|corrupted|double free' $log 2>/dev/null)
  say "  tokens=$tok server=$alive capture_time_misses=$miss glibc_diagnostics=${glibc:-0}"
  [ "${miss:-0}" != "0" ] && awk '/capture started/{f=1} f && /MISS during capture/' $log | grep -oE 'size [0-9]+' | sort -u | sed 's/^/    /' | tee -a $S
  echo "$tok" >$L/$n.tok; echo "${miss:-0}" >$L/$n.miss; echo "${glibc:-0}" >$L/$n.glibc; echo "$alive" >$L/$n.alive
  cleanup; PID=""; sleep 4; return 0
}

leg control ""                                  || say "control leg incomplete"
leg treatment "ARC_GRAPH_PREWARM_SIZES=172032"  || say "treatment leg incomplete"

say "=== VERDICT ==="
cm=$(cat $L/control.miss 2>/dev/null||echo NA); tm=$(cat $L/treatment.miss 2>/dev/null||echo NA)
cg=$(cat $L/control.glibc 2>/dev/null||echo NA); tg=$(cat $L/treatment.glibc 2>/dev/null||echo NA)
ct=$(cat $L/control.tok 2>/dev/null||echo NA);  tt=$(cat $L/treatment.tok 2>/dev/null||echo NA)
ca=$(cat $L/control.alive 2>/dev/null||echo NA); ta=$(cat $L/treatment.alive 2>/dev/null||echo NA)
say "control   : misses=$cm glibc=$cg tokens=$ct server=$ca"
say "treatment : misses=$tm glibc=$tg tokens=$tt server=$ta"
if [ "$cm" = "0" ] || [ "$cm" = "NA" ]; then
  say "UNPROVEN: the control did NOT show a capture-time miss, so the treatment's"
  say "  absence proves nothing — the log was never shown capable of reporting it."
elif [ "$tm" != "0" ]; then
  say "PREWARM DID NOT TAKE: treatment still missed ($tm). Hypothesis untested, not refuted."
elif [ "$tg" = "0" ] && [ "$tt" != "0" ]; then
  say "✅ CONFIRMED BY CONSTRUCTION: removing the capture-time allocation removed the"
  say "  corruption (control missed and died; treatment did not miss and served $tt tokens)."
  say "  REMINDER: pre-warming is a PROBE. The fix is that the buffer stops"
  say "  allocating per decode step."
else
  say "🔴 ELIMINATED, NOT FIXED: the miss is gone ($tm) and it STILL corrupts"
  say "  (glibc=$tg, tokens=$tt, server=$ta). The last capture-time allocation is"
  say "  NOT the cause. This is a result: the search moves elsewhere."
fi
say "RESULT: COMPLETE"
