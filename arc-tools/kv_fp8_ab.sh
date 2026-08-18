#!/usr/bin/env bash
# ArcKV/Fp8 — interleaved A-B-A-B decode timing for the fused E4M3 kernel.
#   A = ARC_KV_FP8_MODE=cpu    (43 blocking D2H per step)
#   B = ARC_KV_FP8_MODE=fused  (one device kernel, 0 D2H)
#
# A-then-B is banned on this box: a 3.3% monotonic drift once exceeded the arm
# difference. The drift within each arm is printed next to the delta, and a
# delta smaller than the drift is not a result.
#
# LOCK DISCIPLINE. The lock covers exactly one thing: a leg (model load, which
# allocates GPU memory, plus the timed run). It is taken immediately before and
# released immediately after — NOT held across parsing, and NOT wrapped around
# the whole script. Six chains share this H200; one holding the lock through
# post-processing starved the trellis chain for 40 minutes while the card read
# 0 %, 0 MiB, 78 W.
set -u
OUT=${OUT:-/root/kvfp8}
BASE=${BASE:-/root/budget-chain}
BIN=${BIN:-/root/arc-wt/fp8-target/release/mistralrs}
LOCK=${LOCK:-/root/locks/bench.lock}
LOCKWAIT=${LOCKWAIT:-3600}
REPS=${REPS:-2}
GEN=${GEN:-260}
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
unset ARC_TIME_DECODE V4_STATS V4_NAN_DEBUG V4_TRACE ARC_PROFILE ARC_GPU_ACT_QUANT
[ -x "$BIN" ] || { echo "FATAL_NO_BINARY $BIN"; exit 2; }

exec 9>"$LOCK"

leg() { # $1 = tag, $2 = mode
  local tag=$1 mode=$2 log="$OUT/$1.log" free rc ms
  flock -w "$LOCKWAIT" 9 || { echo "FATAL_LOCK_TIMEOUT ${LOCKWAIT}s"; exit 2; }
  # Holding the lock is not the same as the card being free: V4 is ~79 GB of
  # 143 GB and the previous holder's process can outlive its script.
  for _ in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "${free:-0}" -ge 100000 ] && break
    sleep 5
  done
  [ "${free:-0}" -ge 100000 ] || { echo "FATAL_NO_VRAM ${free}MiB"; exit 2; }

  ARC_KV_FP8_MODE=$mode "$BIN" bench -m "$BASE/src" -a deepseekv4 \
    --from-uqff "$BASE/uqff/qtip2-0.uqff" --max-seqs 1 --prefix-cache-n 0 \
    --prompt-len 64 --gen-len "$GEN" --iterations 1 --warmup 0 >"$log" 2>&1
  rc=$?
  flock -u 9   # release BEFORE parsing

  [ $rc -eq 0 ] || {
    echo "FAIL_BENCH $tag rc=$rc"
    # A process killed by a GPU fault dies with no error line. This box carries
    # ~1,485 Xid faults, so check before blaming the code.
    dmesg 2>/dev/null | grep -i xid | tail -3
    tail -5 "$log"
    exit 1
  }
  # "| Decode (260 tokens) | 13.9 +- 0.0 | 71.80 ms/T |". An empty parse must
  # FAIL, not silently become a 0.0 ms delta.
  ms=$(grep -oE "[0-9]+\.[0-9]+ ms/T" "$log" | tail -1 | grep -oE "^[0-9]+\.[0-9]+")
  [ -n "$ms" ] || { echo "FAIL_NO_MS_PER_T $tag"; tail -5 "$log"; exit 1; }
  echo "$tag $mode $ms"
}

: >"$OUT/ab.txt"
for i in $(seq 1 "$REPS"); do
  leg "A$i" cpu   | tee -a "$OUT/ab.txt"
  leg "B$i" fused | tee -a "$OUT/ab.txt"
done

echo "=== RESULT ==="
awk '
  { v[substr($1,1,1) substr($1,2)] = $3; arm[substr($1,1,1)] += $3; n[substr($1,1,1)]++ }
  END {
    a = arm["A"]/n["A"]; b = arm["B"]/n["B"];
    printf "A (cpu round trip) mean %.2f ms/token over %d legs\n", a, n["A"];
    printf "B (fused kernel)   mean %.2f ms/token over %d legs\n", b, n["B"];
    printf "delta              %.2f ms/token (%+.2f%%)\n", a-b, -100.0*(a-b)/a;
    if (n["A"]>1) printf "drift within A     %+.2f%%  (A1 %.2f -> A%d %.2f)\n", \
      100.0*(v["A" n["A"]]-v["A1"])/v["A1"], v["A1"], n["A"], v["A" n["A"]];
    if (n["B"]>1) printf "drift within B     %+.2f%%  (B1 %.2f -> B%d %.2f)\n", \
      100.0*(v["B" n["B"]]-v["B1"])/v["B1"], v["B1"], n["B"], v["B" n["B"]];
    print "A delta smaller than the drift is not a result.";
  }' "$OUT/ab.txt"
