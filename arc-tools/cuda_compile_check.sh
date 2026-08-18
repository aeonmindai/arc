#!/usr/bin/env bash
#
# Arc — CUDA compile + kernel-smoke gate.
#
# Run this FIRST on any Linux box with the CUDA toolkit, BEFORE downloading
# model weights. It compiles every Arc-authored CUDA kernel (QTIP + the
# arc-cuda-graph decode/gemv/FlashMLASparse kernels) with the exact feature
# set the rental uses, and — if a GPU is actually present — runs the QTIP
# CUDA parity tests that exercise the kernels at runtime.
#
# This is the local mirror of .github/workflows/cuda_compile_check.yaml
# (which runs the same compile step for free, with no GPU). Use it to fail
# fast on the build box instead of discovering a kernel problem after a
# 148 GB download + 30 min of ISQ.
#
# Usage:
#   bash arc-tools/cuda_compile_check.sh                 # auto-detect arch
#   CUDA_COMPUTE_CAP=90 bash arc-tools/cuda_compile_check.sh   # force Hopper
#   FEATURES="cuda" bash arc-tools/cuda_compile_check.sh       # skip flash-attn
#
# Exit codes: 0 = all gates passed. Non-zero with a `FAIL:` line otherwise.

set -uo pipefail

# Exact feature set the rental build uses (rental_h100_v4_flash.sh step 4).
# Override with FEATURES="cuda" on a small box where flash-attn would OOM.
FEATURES=${FEATURES:-cuda flash-attn}
RUN_GPU_TESTS=${RUN_GPU_TESTS:-auto}   # auto | yes | no

step() { echo; echo "::::::: $* :::::::"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

# --- repo root (script lives in arc-tools/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || fail "cannot cd to repo root"

step "1/5 toolchain sanity"
command -v cargo >/dev/null || fail "cargo not found — install rust first"
CUDA_HOME_GUESS=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
export CUDA_HOME=${CUDA_HOME:-${CUDA_HOME_GUESS:-/usr/local/cuda}}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
command -v nvcc >/dev/null || fail "nvcc not found at \$CUDA_HOME/bin ($CUDA_HOME) — install the CUDA toolkit"
ok "nvcc: $(nvcc --version | grep -i release | sed 's/^ *//')"

# --- compute capability: env wins; else detect from GPU; else default 90 ---
GPU_PRESENT=no
GPU_CC=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU_PRESENT=yes
  # Actual device compute capability (75=T4, 80=A100, 90=H100). Tracked
  # SEPARATELY from the CUDA_COMPUTE_CAP *compile target*, which may be forced
  # higher than the device — e.g. cross-compiling for sm_90 on a T4.
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  GPU_CC=${GPU_CC:-0}
fi
if [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
  if [ "$GPU_PRESENT" = yes ] && [ "$GPU_CC" -gt 0 ] 2>/dev/null; then
    CUDA_COMPUTE_CAP=$GPU_CC
  else
    # No GPU (or undetectable cap): compile for Hopper, the V4 Flash rental target.
    CUDA_COMPUTE_CAP=90
  fi
fi
export CUDA_COMPUTE_CAP
ok "CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP (gpu_present=$GPU_PRESENT, device_cc=$GPU_CC)"
if [ "$CUDA_COMPUTE_CAP" -lt 80 ] 2>/dev/null; then
  echo "WARN: QTIP kernels require sm_80+ (has_qtip_kernels gate in mistralrs-quant/build.rs)."
  echo "      Compiling for sm_${CUDA_COMPUTE_CAP} will NOT enable the QTIP path."
  echo "      Set CUDA_COMPUTE_CAP=90 to compile-check the kernels the rental will use."
fi

step "2/5 compile Arc CUDA kernels (mistralrs-quant + arc-cuda-graph) [features: cuda]"
# `--lib`: build.rs runs nvcc on every .cu, but no binary is linked. Keeps this
# step GPU-less-safe (arc-cuda-graph's gemv_bench bin would otherwise link -lcuda).
cargo build -p mistralrs-quant --lib --features cuda 2>&1 | tail -40 || fail "mistralrs-quant --features cuda FAILED. This step does TWO things: nvcc compiles the .cu files, then rustc type-checks the cuda-gated Rust. A 'Compiling N of N kernels' line above means the KERNELS were fine and the RUST did not compile -- read the rustc errors, not the nvcc output."
cargo build -p arc-cuda-graph --lib --features cuda 2>&1 | tail -40 || fail "arc-cuda-graph --features cuda FAILED (nvcc kernels AND/OR cuda-gated Rust -- see above)"
ok "Arc CUDA kernels compiled for sm_${CUDA_COMPUTE_CAP}"

step "3/5 compile-check CUDA-gated tests (--no-run)"
cargo test --no-run -p mistralrs-quant --features cuda 2>&1 | tail -5 || fail "qtip cuda-gated tests did not compile"
ok "cuda-gated test harness compiled"

step "4/5 compile rental binaries with EXACT rental features [features: ${FEATURES}]"
# shellcheck disable=SC2086
cargo build --release -p arc-cli -p mistralrs-cli --features "${FEATURES}" 2>&1 | tail -5 \
  || fail "arc-cli/mistralrs-cli did not compile with features '${FEATURES}' — this is the rental step-4 build"
ok "rental binaries compiled (features: ${FEATURES})"

step "5/5 GPU kernel smoke tests (QTIP parity) — runtime validation"
SHOULD_RUN=no
case "$RUN_GPU_TESTS" in
  yes) SHOULD_RUN=yes ;;
  no)  SHOULD_RUN=no ;;
  # Gate on the ACTUAL device cap, not the compile target. Code built for a
  # newer arch than the GPU fails to load with CUDA_ERROR_INVALID_PTX (e.g.
  # sm_90 binaries on an sm_75 T4), and QTIP itself needs sm_80+. So auto-run
  # only when the device is sm_80+ AND can execute what we compiled (target <= device).
  auto) [ "$GPU_PRESENT" = yes ] && [ "$GPU_CC" -ge 80 ] 2>/dev/null \
        && [ "$CUDA_COMPUTE_CAP" -le "$GPU_CC" ] 2>/dev/null && SHOULD_RUN=yes ;;
esac
if [ "$SHOULD_RUN" = yes ]; then
  # These tests build a QTIP layer on CUDA and assert GPU output matches the
  # CPU reference (cos sim >= 0.999). cuda_quantize_matches_cpu_dequantize_cos_sim
  # exercises the prefix-grouped Viterbi *quantize* kernel — the highest-risk,
  # never-run-on-GPU-until-now path. A hang here fails fast in ~1 min.
  cargo test --release -p mistralrs-quant --features cuda -- --nocapture --test-threads=1 \
      cuda_quantize_matches_cpu_dequantize_cos_sim \
      cuda_dequantize_matches_cpu_viterbi_rotation \
      cuda_forward_matches_cpu_viterbi_rotation \
      cuda_rotate_x_matches_cpu \
      cuda_fused_gemv_matches_dequant_matmul \
      cuda_fused_gemv_matches_dequant_matmul_no_rotation \
      qtip_gather_forward_cuda_matches_cpu \
      2>&1 | tee /tmp/qtip_gpu_smoke.log || fail "QTIP GPU kernel smoke tests failed (log: /tmp/qtip_gpu_smoke.log)"
  # The tests no-op (print "CUDA not available; skipping" and return Ok) if a
  # CUDA device can't be opened — which would make the line above pass without
  # exercising a kernel. Require the parity output and reject the skip path.
  grep -q "cos sim" /tmp/qtip_gpu_smoke.log || fail "QTIP GPU smoke tests did not actually run a kernel (no 'cos sim' output)"
  grep -q "CUDA not available; skipping" /tmp/qtip_gpu_smoke.log && fail "QTIP GPU smoke tests skipped — Device::new_cuda(0) failed on this box"
  ok "QTIP GPU kernels run correctly on this device (parity >= 0.999)"
else
  echo "SKIP: GPU smoke tests not run (gpu_present=$GPU_PRESENT, device_cc=$GPU_CC, compiled_for=$CUDA_COMPUTE_CAP, mode=$RUN_GPU_TESTS)."
  echo "      Compile gate passed. Runtime kernel validation requires an sm_80+ GPU"
  echo "      whose arch can run the compiled kernels — a free Colab/Kaggle T4 is sm_75,"
  echo "      and cross-compiling for sm_90 on it stays compile-only (you cannot run"
  echo "      sm_90 binaries on sm_75)."
fi

echo
echo "================================================================================"
echo "  CUDA COMPILE CHECK PASSED for sm_${CUDA_COMPUTE_CAP} (features: ${FEATURES})"
[ "$SHOULD_RUN" = yes ] && echo "  + QTIP GPU kernel smoke tests PASSED"
echo "  The rental build (rental_h100_v4_flash.sh step 4) will not fail on Arc CUDA."
echo "================================================================================"
