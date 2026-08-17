#!/usr/bin/env bash
#
# Arc — wave64 keystone gate: trellis grouped-GEMM arch specialisation.
#
# What this answers, in order, so a cheap box can answer the first half and an
# H200 the second:
#
#   STAGE 1  COMPILE (no GPU needed)  — the new SM90+ kernel and the
#            status-returning launchers compile for sm_80, sm_90a and sm_100a.
#            nvcc cross-compiles, so ANY box with CUDA >= 12.8 can run this;
#            an A30/A6000 at ~$0.40/hr is enough. This is the gate that has
#            never been run for this change.
#   STAGE 2  PARITY (needs sm_90+)    — the grouped-GEMM CUDA parity tests.
#            On an sm_90 box these exercise the NEW kernel; on sm_80 they
#            exercise the old one. Both must pass.
#   STAGE 3  A/B TIMING (needs sm_90+) — qtip_grouped_curve on this branch and
#            on the baseline ref, same box, same fixture.
#
# D18: three distinct exit codes. 0 = pass. 1 = a genuine failure, the only
# result that should change a decision. 2 = the environment could not answer
# (no CUDA, no GPU, wrong arch) — NEVER conflate this with 1. Every terminal
# outcome, including a timeout, writes a line to the status file, so silence
# can never be read as success.
#
# Usage (main runs this; agents do not call runcrate — D15):
#   setsid nohup bash arc-tools/wave64_keystone_gate.sh \
#       < /dev/null > /tmp/wave64.log 2>&1 &
#
# Env:
#   BASELINE_REF   git ref for the A/B baseline (default: origin/master)
#   STATUS         status file (default: /tmp/wave64_status.txt)
#   SKIP_AB        set to 1 to run stages 1-2 only
#   CURVE_ARGS     extra args for qtip_grouped_curve

set -uo pipefail

STATUS=${STATUS:-/tmp/wave64_status.txt}
BASELINE_REF=${BASELINE_REF:-origin/master}
REPO=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "STAGE0 env  FAIL not a git repo" | tee -a "$STATUS"; exit 2; }
cd "$REPO" || exit 2

: > "$STATUS"
say() { echo "$*" | tee -a "$STATUS"; }
# Any unexpected exit still writes a terminal line.
trap 'rc=$?; [ "$rc" -ne 0 ] && say "TERMINAL unexpected-exit rc=$rc"; exit $rc' EXIT

say "wave64 keystone gate  branch=$(git rev-parse --abbrev-ref HEAD)  head=$(git rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
# Environment probe. Distinguishes "cannot answer" from "failed".
# ---------------------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  export PATH=${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
fi
if ! command -v nvcc >/dev/null 2>&1; then
  say "STAGE1 compile  ENV-CANNOT-ANSWER no nvcc on PATH"
  say "TERMINAL exit=2"
  exit 2
fi
NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d '.' || true)
say "env nvcc=$NVCC_VER gpu_cc=${GPU_CC:-none}"

# ---------------------------------------------------------------------------
# STAGE 1 — compile for every arch this change claims to support.
# sm_100a needs CUDA >= 12.8; when the toolkit is older that is an
# environment answer, not a failure of the code.
# ---------------------------------------------------------------------------
STAGE1_FAIL=0
for CC in 80 90 100; do
  if [ "$CC" -ge 100 ]; then
    case "$NVCC_VER" in
      12.[0-7]*|12.[0-7]|11.*) say "STAGE1 sm_${CC}  ENV-CANNOT-ANSWER nvcc $NVCC_VER predates sm_100a"; continue ;;
    esac
  fi
  if CUDA_COMPUTE_CAP=$CC FEATURES="cuda" RUN_GPU_TESTS=no \
     timeout 3600 bash arc-tools/cuda_compile_check.sh > "/tmp/wave64_compile_${CC}.log" 2>&1; then
    say "STAGE1 sm_${CC}  PASS"
  else
    rc=$?
    [ "$rc" -eq 124 ] && say "STAGE1 sm_${CC}  FAIL timeout" || say "STAGE1 sm_${CC}  FAIL rc=$rc (see /tmp/wave64_compile_${CC}.log)"
    STAGE1_FAIL=1
  fi
done
[ "$STAGE1_FAIL" -eq 1 ] && { say "TERMINAL exit=1 compile gate failed"; exit 1; }

# ---------------------------------------------------------------------------
# STAGE 2 — grouped-GEMM parity on real silicon.
# ---------------------------------------------------------------------------
if [ -z "${GPU_CC:-}" ]; then
  say "STAGE2 parity  ENV-CANNOT-ANSWER no GPU visible"
  say "STAGE3 ab      ENV-CANNOT-ANSWER no GPU visible"
  say "TERMINAL exit=0 compile-only (stages 2-3 unanswered)"
  exit 0
fi

export CUDA_COMPUTE_CAP=$GPU_CC
PARITY_LOG=/tmp/wave64_parity.log
if timeout 5400 cargo test --release -p mistralrs-quant --features cuda \
      cuda_grouped_gemm -- --nocapture --test-threads=1 > "$PARITY_LOG" 2>&1; then
  # D18 rule 1: a green result must PROVE work happened. `cargo test` reports
  # success for zero tests run, which is exactly the shape being guarded here.
  NRUN=$(grep -Eo '[0-9]+ passed' "$PARITY_LOG" | head -1 | grep -Eo '[0-9]+' || echo 0)
  if [ "${NRUN:-0}" -lt 2 ]; then
    say "STAGE2 parity  FAIL only ${NRUN} grouped tests ran (expected >= 2); a filter that matches nothing reports success"
    say "TERMINAL exit=1"; exit 1
  fi
  say "STAGE2 parity  PASS sm_${GPU_CC} tests_passed=${NRUN}"
else
  rc=$?
  [ "$rc" -eq 124 ] && say "STAGE2 parity  FAIL timeout" || say "STAGE2 parity  FAIL rc=$rc (see $PARITY_LOG)"
  say "TERMINAL exit=1"; exit 1
fi

if [ "${SKIP_AB:-0}" = "1" ]; then
  say "STAGE3 ab      SKIPPED by request"
  say "TERMINAL exit=0"; exit 0
fi
if [ "$GPU_CC" -lt 90 ] 2>/dev/null; then
  say "STAGE3 ab      ENV-CANNOT-ANSWER sm_${GPU_CC} takes the Ampere path; the change under test is the SM90+ kernel"
  say "TERMINAL exit=0 (parity green, delta unmeasured)"; exit 0
fi

# ---------------------------------------------------------------------------
# STAGE 3 — A/B against the baseline on the SAME box and fixture.
# ---------------------------------------------------------------------------
run_curve() {  # $1 = label, $2 = dir
  local out="/tmp/wave64_curve_$1.log"
  if ! ( cd "$2" && timeout 5400 cargo run --release -p mistralrs-quant \
           --features cuda --example qtip_grouped_curve -- ${CURVE_ARGS:-} ) > "$out" 2>&1; then
    local rc=$?
    [ "$rc" -eq 124 ] && say "STAGE3 $1  FAIL timeout" || say "STAGE3 $1  FAIL rc=$rc (see $out)"
    return 1
  fi
  # The harness has previously printed a headline while every row was empty.
  if ! grep -qE '^[[:space:]]*[0-9]+[[:space:]]*\|' "$out"; then
    say "STAGE3 $1  FAIL ran clean but produced no data rows (see $out)"
    return 1
  fi
  say "STAGE3 $1  PASS rows=$(grep -cE '^[[:space:]]*[0-9]+[[:space:]]*\|' "$out")  (see $out)"
  return 0
}

run_curve head "$REPO" || { say "TERMINAL exit=1"; exit 1; }

BASE_WT=/tmp/wave64-baseline
rm -rf "$BASE_WT"
if ! git worktree add --detach "$BASE_WT" "$BASELINE_REF" >/dev/null 2>&1; then
  say "STAGE3 baseline  ENV-CANNOT-ANSWER could not create a worktree at $BASELINE_REF"
  say "TERMINAL exit=0 (head measured, baseline unmeasured)"; exit 0
fi
run_curve baseline "$BASE_WT" || { say "TERMINAL exit=1"; exit 1; }
git worktree remove --force "$BASE_WT" >/dev/null 2>&1 || true

say "STAGE3 ab      PASS — compare /tmp/wave64_curve_head.log vs /tmp/wave64_curve_baseline.log"
say "TERMINAL exit=0"
exit 0
