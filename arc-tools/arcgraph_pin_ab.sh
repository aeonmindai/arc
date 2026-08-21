#!/bin/bash
# Does pinning the xs window remove the CUDA-graph capture heap corruption?
#
# ONE BINARY, ONE VARIABLE. Both arms are /root/wt-ladder's binary; the only
# difference is ARC_V4_XS_PIN_WINDOW. That is a cleaner control than my
# pre-warm probe ever was — pre-warming seeded a hard-coded size, this changes
# the actual allocation behaviour.
#
#   PIN  arm : flag unset      -> xs_rolling logs "PINNED"
#   CTRL arm : flag = 0        -> xs_rolling logs "RESIZING"
#
# 🔴 THIS SCRIPT CANNOT PRODUCE A VERDICT ON master AS OF cc5487ad3. DO NOT
# BOOK GPU TIME FOR IT UNTIL THE FLAG BELOW EXISTS.
#
# `ARC_V4_XS_PIN_WINDOW` is read by NOTHING in the tree — the name appears only
# in this file. Neither does the "xs rolling window is PINNED/RESIZING" log line
# this script greps for, at `xs_rolling.rs:241` or anywhere else. The nearest
# real flag is `ARC_V4_XS_PER_SEQ` (`kv_cache/xs_rolling.rs:233`), which is a
# different thing: per-sequence ragged decode, not window pinning.
#
# So both arms would run in the SAME state, which is exactly the failure the
# assertion below was written to catch. It does catch it — the run exits 2 with
# "PIN A/B IS VOID" rather than reporting "the pin is free" — but only after two
# full model loads and two generations have been paid for. Verified during the
# env-flag polarity sweep, 2026-08-21.
#
# ⚠️ THE ASSERTION THAT MAKES THE COMPARISON MEAN ANYTHING.
# Both strings are compiled into the binary — presence proves nothing. What
# proves the arms differ is the RUNTIME log line, emitted once per process by
# `xs_rolling.rs:241`. A mistyped flag name gives two identical arms that agree
# perfectly and report "the pin is free". So: PINNED must appear in one log and
# RESIZING in the other, or this prints PIN A/B IS VOID and exits 2 without a
# verdict. (Method borrowed from the ladder chain, which built its own A/B this
# way for exactly this reason.)
#
# PRE-REGISTERED OUTCOMES (arc-tools/prereg/prewarm_probe_outcomes.md, committed
# before any of this ran — they are NOT being chosen now to fit the result):
#   P1 miss gone AND survives      -> CONFIRMED: the per-step allocation was it
#   P2 miss gone BUT still crashes -> ELIMINATED, NOT FIXED (equal weight)
#   P3 miss still present          -> CANNOT ANSWER, exit 2
set -u

L=/root/logs/arcgraph-pin; mkdir -p $L; S=$L/STATUS.txt; : > $S
BIN=${BIN:-/root/wt-ladder/target/release/mistralrs}
SRC=/root/models/v4-src; UQFF=/root/models/v4-uqff/qtip2b-0.uqff; PORT=1237
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $S; }
PID=""

# Fail before spending GPU time, not after. `strings` on the binary is enough:
# if the flag name is not compiled in, nothing reads it, and the two arms are
# the same arm. Set PIN_AB_FORCE=1 to run anyway (e.g. against a binary built
# from a branch that adds the flag).
if [ "${PIN_AB_FORCE:-0}" != "1" ] && [ -r "$BIN" ] \
   && ! strings "$BIN" 2>/dev/null | grep -q ARC_V4_XS_PIN_WINDOW; then
  say "🔴 REFUSING TO RUN: $BIN does not contain ARC_V4_XS_PIN_WINDOW, so the"
  say '   ctrl arm (ARC_V4_XS_PIN_WINDOW=0) and the pin arm (unset) are the SAME'
  say "   arm. This would burn two model loads to reach 'PIN A/B IS VOID'."
  say "   See the header comment for what to do instead."
  say "RESULT: UNANSWERED (flag not present in binary)"
  exit 2
fi

cleanup(){ [ -n "$PID" ] && kill $PID 2>/dev/null; sleep 3; [ -n "$PID" ] && kill -9 $PID 2>/dev/null; pkill -f "mistralrs serve -p $PORT" 2>/dev/null; return 0; }
trap cleanup EXIT INT TERM

# Exclusivity: a neighbour makes both the timings and any dmesg attribution
# unsound, and this box is shared.
OCC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c .)
[ "${OCC:-0}" = "0" ] || { say "ENV_FAIL: $OCC foreign compute process(es) on the GPU"; say "RESULT: UNANSWERED"; exit 2; }
[ -x "$BIN" ] || { say "ENV_FAIL: no binary at $BIN"; say "RESULT: UNANSWERED"; exit 2; }
say "BIN=$BIN mtime=$(date -u -r $BIN +%H:%M:%S)"
say "ref=$(cd /root/wt-ladder 2>/dev/null && git log --oneline -1 || echo unknown)"

leg(){ # $1 name, $2 extra env
  # NOT `local n=$1 extra=$2 log=$L/$n.server.log` — bash expands $n while the
  # `local` builtin is still processing its own arguments, so under `set -u`
  # that is "n: unbound variable" and the script dies before the first arm.
  local n=$1
  local extra=$2
  local log=$L/$n.server.log
  say "=== arm $n (${extra:-<flag unset>}) ==="
  cd /root/wt-ladder || return 1
  # shellcheck disable=SC2086
  env $extra RUST_LOG=info \
      ARC_CAPTURE_STREAM=1 ARC_V4_CAPTURE_PROBE=1 ARC_CANDLE_ALLOC_CACHE=1 \
      ARC_GRAPH_WARMUP=8 ARC_GRAPH_DEFERRED_PASSES=3 \
      $BIN serve -p $PORT -m $SRC -a deepseekv4 --from-uqff $UQFF \
      --chat-template chat_templates/deepseek_v4.json \
      --prefix-cache-n 0 --paged-attn off --max-seqs 1 --max-seq-len 4096 >$log 2>&1 &
  PID=$!
  local ok=0
  for _ in $(seq 1 90); do curl -fsS --max-time 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $PID 2>/dev/null || break; sleep 5; done
  if [ $ok != 1 ]; then say "  server never healthy"; tail -4 $log | sed 's/^/    /' | tee -a $S; cleanup; PID=""; echo 0 >$L/$n.tok; return 1; fi
  curl -s --max-time 600 localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"Count from one to twenty in words, then stop.","max_tokens":24,"temperature":0}' >$L/$n.resp.json 2>&1
  local tok; tok=$(python3 -c "import json;print(json.load(open('$L/$n.resp.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  local alive="DIED"; kill -0 $PID 2>/dev/null && alive="ALIVE"
  # Misses only AFTER capture begins; before that they are the warmup passes
  # doing their job and share the same warning string.
  local miss; miss=$(awk '/capture started for batch_size/{f=1} f && /MISS during capture/' $log 2>/dev/null | wc -l | tr -d ' ')
  local started; started=$(grep -c "capture started for batch_size" $log 2>/dev/null)
  local glibc; glibc=$(grep -ciE 'malloc_consolidate|free\(\):|malloc\(\):|corrupted|double free' $log 2>/dev/null)
  local mode="NONE"
  grep -q "xs rolling window is PINNED"   $log 2>/dev/null && mode="PINNED"
  grep -q "xs rolling window is RESIZING" $log 2>/dev/null && mode="RESIZING"
  say "  mode=$mode tokens=$tok server=$alive capture_started=$started misses=$miss glibc=${glibc:-0}"
  [ "${miss:-0}" != "0" ] && awk '/capture started/{f=1} f && /MISS during capture/' $log | grep -oE 'size [0-9]+' | sort -u | sed 's/^/    /' | tee -a $S
  echo "$tok" >$L/$n.tok; echo "${miss:-0}" >$L/$n.miss; echo "${glibc:-0}" >$L/$n.glibc
  echo "$alive" >$L/$n.alive; echo "$mode" >$L/$n.mode; echo "${started:-0}" >$L/$n.started
  cleanup; PID=""; sleep 4; return 0
}

leg ctrl "ARC_V4_XS_PIN_WINDOW=0" || say "ctrl arm incomplete"
leg pin  ""                        || say "pin arm incomplete"

say "=== VERDICT ==="
cmode=$(cat $L/ctrl.mode 2>/dev/null||echo NA); pmode=$(cat $L/pin.mode 2>/dev/null||echo NA)
cmiss=$(cat $L/ctrl.miss 2>/dev/null||echo NA); pmiss=$(cat $L/pin.miss 2>/dev/null||echo NA)
cg=$(cat $L/ctrl.glibc 2>/dev/null||echo NA);   pg=$(cat $L/pin.glibc 2>/dev/null||echo NA)
ct=$(cat $L/ctrl.tok 2>/dev/null||echo NA);     pt=$(cat $L/pin.tok 2>/dev/null||echo NA)
ca=$(cat $L/ctrl.alive 2>/dev/null||echo NA);   pa=$(cat $L/pin.alive 2>/dev/null||echo NA)
cs=$(cat $L/ctrl.started 2>/dev/null||echo 0);  ps=$(cat $L/pin.started 2>/dev/null||echo 0)
say "ctrl : mode=$cmode misses=$cmiss glibc=$cg tokens=$ct server=$ca capture_started=$cs"
say "pin  : mode=$pmode misses=$pmiss glibc=$pg tokens=$pt server=$pa capture_started=$ps"

# 1. THE ARMS MUST ACTUALLY DIFFER.
if [ "$cmode" != "RESIZING" ] || [ "$pmode" != "PINNED" ]; then
  say "🔴 PIN A/B IS VOID — expected ctrl=RESIZING and pin=PINNED, got ctrl=$cmode pin=$pmode."
  say "   The two arms are not demonstrably different behaviour, so NO comparison"
  say "   between them means anything. Most likely a flag-name or log-level problem."
  say "RESULT: UNANSWERED"; exit 2
fi
say "✅ arms differ: ctrl=RESIZING, pin=PINNED"

# 2. THE CONTROL MUST REPRODUCE WHAT WE ARE TRYING TO REMOVE.
if [ "$cs" = "0" ]; then
  say "UNPROVEN: capture never started in the control, so its miss count is vacuous."
  say "RESULT: UNANSWERED"; exit 2
fi
if [ "${cmiss:-0}" = "0" ]; then
  say "UNPROVEN: the control shows NO capture-time miss, so the pin removing one"
  say "   proves nothing — the log was never shown capable of reporting it."
  say "RESULT: UNANSWERED"; exit 2
fi

# 3. Pre-registered outcomes.
if [ "${pmiss:-1}" != "0" ]; then
  say "P3 CANNOT ANSWER: the pin arm still shows $pmiss capture-time miss(es)."
  say "   The pin did not remove the varying allocation on this path. Says nothing"
  say "   in EITHER direction about whether that allocation is the corruption."
  say "RESULT: UNANSWERED"; exit 2
elif [ "${pg:-1}" = "0" ] && [ "${pt:-0}" != "0" ] && [ "$pa" = "ALIVE" ]; then
  say "✅ P1 CONFIRMED: pinning removed the capture-time allocation AND the"
  say "   corruption (ctrl: misses=$cmiss glibc=$cg tokens=$ct $ca; pin: misses=0"
  say "   glibc=$pg tokens=$pt $pa). The per-step-varying xs allocation was the"
  say "   heap corruption. This is a MEMORY-SAFETY fix, not only a throughput one."
else
  say "🔴 P2 ELIMINATED, NOT FIXED: the miss is gone (pin misses=0) and it STILL"
  say "   corrupts (glibc=$pg tokens=$pt server=$pa). The last capture-time"
  say "   allocation is NOT the cause. Reported with the same weight as P1 — the"
  say "   search moves to the committed next suspects: cuBLAS workspace leak(),"
  say "   event-tracking vs leak()'s manual refcount decrements, RELAXED capture."
fi
say "RESULT: COMPLETE"
