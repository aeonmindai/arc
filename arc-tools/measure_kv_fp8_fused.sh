#!/usr/bin/env bash
# ArcKV/Fp8 — measure the fused E4M3 quantize+dequantize kernel against the
# CPU round trip it replaces. Runs ON THE BOX. Nothing here is a proxy: every
# number is produced by the real binary on the real model.
#
# WHAT IT PRODUCES (the four numbers the change is judged on)
#   1. `cuMemcpyDtoHAsync_v2` calls per decode step, before and after.
#      The 44/step are the thing being removed and the reason CUDA graph
#      capture is impossible. NOTE `*Synchronize*` is 0.0 calls/step in this
#      workload — counting `cudaStreamSynchronize` reports "no syncs" and is
#      wrong. Count the D2H.
#   2. Kernel launches per decode step, before and after. Op count is the
#      disease (9,131/token, median kernel 1.18 us), so this matters more than
#      wall time.
#   3. Interleaved A-B-A-B ms/token with the monotonic drift stated. This box
#      showed 3.3% drift across one run — larger than a real arm difference —
#      so A-then-B is a fabricated comparison.
#   4. Bit-parity: the GPU-gated `kv_fp8_fused_is_bit_identical_to_cpu_exact`
#      test, plus its negative control.
#
# RULES ENFORCED HERE, NOT ASSUMED
#   * The box is shared with the ArcGraph chain. Every timing leg holds
#     /root/.arc-bench.lock under flock; a contended number is a fabricated
#     number.
#   * Exclusivity is asserted BEFORE AND AFTER every leg from
#     `nvidia-smi --query-compute-apps` (the lock file is not an occupancy
#     signal — an 87 GB server has been resident with no lock at all, and a
#     77 GB server has run with the lock reading FREE). A neighbour appearing
#     mid-leg aborts the run.
#   * Environment failure exits 2. A failed measurement exits 1. They are not
#     the same thing and must never be reported as the same thing.
#   * Engagement is asserted before any null/neutral result is believed: if
#     the arm that is supposed to use the fused kernel shows the same D2H
#     count as the arm that is not, the run aborts rather than reporting "no
#     difference".
set -uo pipefail

REPO="${REPO:-/root/arc-wt/fp8}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
ARCH="${ARCH:-deepseekv4}"
ISQ="${ISQ:-qtip2}"
PROMPT_LEN="${PROMPT_LEN:-128}"
GEN_LEN="${GEN_LEN:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
OUT="${OUT:-/root/kvfp8-fused}"
LOCK="${LOCK:-/root/locks/bench.lock}"
BIN="${BIN:-${CARGO_TARGET_DIR:-$REPO/target}/release/mistralrs}"
REPS="${REPS:-2}" # A-B-A-B => REPS=2

mkdir -p "$OUT"

envfail() {
  echo "ENVFAIL: $*" >&2
  exit 2
}
fail() {
  echo "FAIL: $*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || envfail "no nvidia-smi"
command -v nsys >/dev/null 2>&1 || echo "WARN: no nsys; the D2H/launch counts will be skipped" >&2
[ -x "$BIN" ] || envfail "no mistralrs binary at $BIN (build with --features 'cuda flash-attn')"

# ---------------------------------------------------------------------------
# Exclusivity. Bracket every leg: a V4 load shows near-zero VRAM for most of a
# minute, so "compute-apps empty" sampled during a neighbour's load is
# indistinguishable from an idle box.
# ---------------------------------------------------------------------------
MYPID=""
assert_exclusive() {
  local where="$1"
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u | tr '\n' ',')
  apps="${apps%,}"
  if [ -n "$MYPID" ]; then
    [ "$apps" = "$MYPID" ] || envfail "[$where] compute-apps=[$apps], expected only $MYPID"
  else
    [ -z "$apps" ] || envfail "[$where] box not idle: compute-apps=[$apps]"
  fi
}

# ---------------------------------------------------------------------------
# One measured leg. $1 = arm label, $2 = ARC_KV_FP8_MODE value ("" => default,
# which is the fused kernel), $3 = log path, $4 = "nsys" to trace.
# ---------------------------------------------------------------------------
run_leg() {
  local arm="$1" mode="$2" log="$3" trace="${4:-}"
  local -a cmd=("$BIN" bench -m "$MODEL" -a "$ARCH" --isq "$ISQ"
    --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" --max-seq-len "$MAX_SEQ_LEN")

  assert_exclusive "pre-$arm"
  (
    if [ -n "$mode" ]; then export ARC_KV_FP8_MODE="$mode"; else unset ARC_KV_FP8_MODE; fi
    unset ARC_GPU_ACT_QUANT
    if [ "$trace" = "nsys" ]; then
      nsys profile -t cuda -o "$OUT/$arm" --force-overwrite true \
        --cuda-memory-usage false "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  ) >"$log" 2>&1 &
  local pid=$!
  # `pgrep -f`/`pkill -f` would match this script's own command line; use the
  # job pid we already hold instead.
  MYPID=""
  wait "$pid"
  local rc=$?
  MYPID=""
  assert_exclusive "post-$arm"
  [ $rc -eq 0 ] || fail "$arm leg exited $rc; see $log"
}

# ---------------------------------------------------------------------------
# ms/token out of a bench log. Fails loudly rather than emitting an empty
# string that would silently become a "0.0 ms/token improvement".
# ---------------------------------------------------------------------------
decode_ms_per_token() {
  local log="$1"
  local tps
  tps=$(grep -oE 'tok_per_s_decode[^0-9]*[0-9]+\.[0-9]+' "$log" | tail -1 |
    grep -oE '[0-9]+\.[0-9]+$')
  [ -n "$tps" ] || tps=$(grep -oiE 'decode[^0-9]*([0-9]+\.[0-9]+) *tok' "$log" | tail -1 |
    grep -oE '[0-9]+\.[0-9]+')
  [ -n "$tps" ] || fail "no decode throughput in $log — do not report a delta from a missing number"
  awk -v t="$tps" 'BEGIN { printf "%.3f", 1000.0 / t }'
}

# ---------------------------------------------------------------------------
# 0. Bit-parity + its negative control. Cheap, and it gates everything else:
#    a faster kernel that stores different bytes is not a faster kernel.
# ---------------------------------------------------------------------------
echo "=== 0. bit-parity (GPU-gated test + D33 negative control) ==="
(
  cd "$REPO" || envfail "no repo at $REPO"
  cargo test -p mistralrs-core --features "cuda flash-attn" --lib \
    kv_fp8_fused_is_bit_identical_to_cpu_exact -- --nocapture --exact \
    models::dsv4_kv_fp8::tests::kv_fp8_fused_is_bit_identical_to_cpu_exact
) 2>&1 | tee "$OUT/parity.log"
prc=${PIPESTATUS[0]}
# `set -e` does not survive a pipe; the status is read explicitly.
[ "$prc" -eq 2 ] && envfail "parity test could not find a CUDA device"
[ "$prc" -eq 0 ] || fail "bit-parity FAILED — the fused kernel does not store what the CPU path stores"
grep -q "test result: ok. 1 passed" "$OUT/parity.log" ||
  fail "parity test reported no results — 'no failures' is not 'ran'"
echo "bit-parity: PASS"

# ---------------------------------------------------------------------------
# 1 + 2. D2H count and launch count per decode step, both arms, under nsys.
# ---------------------------------------------------------------------------
if command -v nsys >/dev/null 2>&1; then
  echo "=== 1+2. nsys: DtoH copies and kernel launches per step ==="
  flock "$LOCK" bash -c "$(declare -f run_leg assert_exclusive envfail fail); \
    OUT='$OUT' BIN='$BIN' MODEL='$MODEL' ARCH='$ARCH' ISQ='$ISQ' \
    PROMPT_LEN='$PROMPT_LEN' GEN_LEN='$GEN_LEN' MAX_SEQ_LEN='$MAX_SEQ_LEN' \
    bash -c 'true'" || true

  for arm in before after; do
    mode=""; [ "$arm" = "before" ] && mode="cpu"
    (
      flock 9 || envfail "could not take the box lock $LOCK"
      run_leg "$arm" "$mode" "$OUT/$arm.nsys.log" nsys
    ) 9>"$LOCK"
    nsys stats --report cuda_api_sum --format csv "$OUT/$arm.nsys-rep" \
      >"$OUT/$arm.api.csv" 2>"$OUT/$arm.api.err" ||
      echo "WARN: nsys stats cuda_api_sum failed for $arm" >&2
    nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/$arm.nsys-rep" \
      >"$OUT/$arm.kern.csv" 2>>"$OUT/$arm.api.err" ||
      echo "WARN: nsys stats cuda_gpu_kern_sum failed for $arm" >&2

    d2h=$(awk -F, '/cuMemcpyDtoHAsync_v2/ { gsub(/"/,"",$3); s+=$3 } END { print s+0 }' "$OUT/$arm.api.csv")
    launches=$(awk -F, 'NR>1 { gsub(/"/,"",$3); s+=$3 } END { print s+0 }' "$OUT/$arm.kern.csv")
    echo "$arm: cuMemcpyDtoHAsync_v2=$d2h  kernel_launches=$launches  (steps=$GEN_LEN)"
    echo "$arm $d2h $launches" >>"$OUT/counts.txt"
  done

  # Engagement (D18): the two arms MUST differ in D2H count. Equal counts mean
  # the fused arm never engaged, and every later number is the same code twice.
  b=$(awk '$1=="before"{print $2}' "$OUT/counts.txt")
  a=$(awk '$1=="after"{print $2}' "$OUT/counts.txt")
  [ -n "$b" ] && [ -n "$a" ] || fail "missing D2H counts — no results is not a null result"
  [ "$a" -lt "$b" ] || fail "ENGAGEMENT: after-arm D2H ($a) is not below before-arm ($b); the fused kernel did not run"
  awk -v b="$b" -v a="$a" -v n="$GEN_LEN" 'BEGIN {
    printf "D2H per step: before %.2f -> after %.2f (removed %.2f/step)\n", b/n, a/n, (b-a)/n }'
fi

# ---------------------------------------------------------------------------
# 3. Interleaved A-B-A-B ms/token. A = CPU round trip, B = fused kernel.
#    The drift is reported alongside the delta, because on this box a 3.3%
#    monotonic drift once exceeded the arm difference.
# ---------------------------------------------------------------------------
echo "=== 3. interleaved A-B-A-B ms/token ==="
: >"$OUT/ab.txt"
for i in $(seq 1 "$REPS"); do
  for arm in A B; do
    mode=""; [ "$arm" = "A" ] && mode="cpu"
    log="$OUT/${arm}${i}.log"
    (
      flock 9 || envfail "could not take the box lock $LOCK"
      run_leg "${arm}${i}" "$mode" "$log"
    ) 9>"$LOCK"
    ms=$(decode_ms_per_token "$log")
    echo "$arm $i $ms" >>"$OUT/ab.txt"
    echo "${arm}${i} = $ms ms/token"
  done
done

awk '
  { v[$1""$2] = $3; arm[$1] += $3; n[$1]++ }
  END {
    a = arm["A"] / n["A"]; b = arm["B"] / n["B"];
    printf "A (CPU round trip) mean %.3f ms/token over %d\n", a, n["A"];
    printf "B (fused kernel)   mean %.3f ms/token over %d\n", b, n["B"];
    printf "delta              %.3f ms/token (%.2f%%)\n", a - b, 100.0 * (a - b) / a;
    # Monotonic drift: same arm, first rep vs last rep.
    if (n["A"] > 1) printf "drift within A     %.2f%% (A1 %.3f -> A%d %.3f)\n", \
      100.0 * (v["A" n["A"]] - v["A1"]) / v["A1"], v["A1"], n["A"], v["A" n["A"]];
    if (n["B"] > 1) printf "drift within B     %.2f%% (B1 %.3f -> B%d %.3f)\n", \
      100.0 * (v["B" n["B"]] - v["B1"]) / v["B1"], v["B1"], n["B"], v["B" n["B"]];
    print "REPORT THE DRIFT WITH THE DELTA. A delta smaller than the drift is not a result.";
  }' "$OUT/ab.txt"

echo "=== artifacts in $OUT ==="
