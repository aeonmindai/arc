#!/bin/bash
# ArcKV/Fp8 — nsys legs: `cuMemcpyDtoHAsync_v2` copies and kernel launches PER
# DECODE STEP, before (CPU round trip) and after (fused device kernel).
#
# These are COUNTS. Box contention changes wall-clock, not how many D2H calls a
# decode step makes, so a contended run still yields a valid count — but the
# card must physically have room: V4 is ~79 GB of the H200's 143 GB, so two of
# these do not fit, and an OOM mid-trace is an environment failure (exit 2), not
# a result.
#
# Counted with arc-tools/kv_fp8_count_per_step.py, which pins the step count
# from the trace and was first validated against the recorded baseline (it
# reproduces 44.09 D2H/step and 9,131.5 launches/step).
set -u
OUT=${OUT:-/root/kvfp8}
BASE=${BASE:-/root/budget-chain}
BIN=${BIN:-/root/arc-wt/fp8-target/release/mistralrs}
DUR=${DUR:-280}
GEN=${GEN:-3000}
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
unset ARC_TIME_DECODE V4_STATS V4_NAN_DEBUG V4_TRACE ARC_PROFILE ARC_GPU_ACT_QUANT
[ -x "$BIN" ] || { echo "FATAL_NO_BINARY $BIN"; exit 2; }
command -v nsys >/dev/null || { echo "FATAL_NO_NSYS"; exit 2; }

# Holding the bench lock is not the same as the previous holder's process having
# exited; gate on what the card actually reports.
wait_for_vram() {
  local need=${1:-100000} i=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "${free:-0}" -ge "$need" ] && { echo "vram ok: ${free} MiB free"; return 0; }
    i=$((i + 1))
    [ "$i" -gt 240 ] && { echo "FATAL_NO_VRAM free=${free}MiB after 20 min"; exit 2; }
    sleep 5
  done
}

for arm in before:cpu after:fused; do
  A=${arm%%:*}
  M=${arm##*:}
  echo "=== leg $A (ARC_KV_FP8_MODE=$M) $(date -u +%T) ==="
  wait_for_vram 100000
  rm -f "$OUT/$A".nsys-rep "$OUT/$A".sqlite
  ARC_KV_FP8_MODE=$M nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --cuda-memory-usage=false --duration="$DUR" --kill=sigterm \
    --force-overwrite=true --output="$OUT/$A" \
    "$BIN" bench -m "$BASE/src" -a deepseekv4 \
    --from-uqff "$BASE/uqff/qtip2-0.uqff" --max-seqs 1 --prefix-cache-n 0 \
    --prompt-len 64 --gen-len "$GEN" --iterations 1 --warmup 0 \
    >"$OUT/$A.nsys.log" 2>&1
  RC=$?
  echo "NSYS_RC=$RC"
  # 143 = SIGTERM, which is how --kill ends the window and is EXPECTED.
  [ "$RC" = "139" ] && { echo "FATAL_TARGET_SIGSEGV"; exit 2; }
  grep -qi "out of memory" "$OUT/$A.nsys.log" && { echo "FATAL_OOM $A"; exit 2; }
  [ -f "$OUT/$A.nsys-rep" ] || { echo "FATAL_NO_REPORT $A"; exit 2; }
  nsys stats --force-export=true --report cuda_api_sum --format csv \
    --output "$OUT/${A}_api" "$OUT/$A.nsys-rep" >/dev/null 2>&1
  [ -f "$OUT/$A.sqlite" ] || { echo "FATAL_NO_SQLITE $A"; exit 2; }
done

echo "=== ANALYSIS ==="
for A in before after; do
  echo "--- $A ---"
  python3 "$OUT/count.py" "$OUT/$A.sqlite" 20 || exit 2
done
