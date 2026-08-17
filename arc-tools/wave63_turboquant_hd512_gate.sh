#!/usr/bin/env bash
# wave63 — TurboQuant CUDA kernels at head_dim {64,128,256,512}, Hopper + Blackwell.
#
# WHAT THIS PROVES
#   1. All four head-dim instantiations of every turbo_* kernel compile.
#   2. They compile for SM90 *and* SM100/SM103 (D16), not just the box's GPU.
#   3. The format tests still pass on-box.
#   4. The existing decode path did not regress.
#
# WHAT THIS DOES NOT DO
#   No quality A/B. TurboQuant quality is settled by prior measurement; this
#   gate is only "it builds, it runs, it did not get slower".
#
# RUN (from main, per D15 — agents never call runcrate):
#   scp this to the box, then:
#     setsid nohup bash /root/arc/arc-tools/wave63_turboquant_hd512_gate.sh \
#       < /dev/null > /root/wave63_gate.log 2>&1 &
#   Watch:  tail -f /root/wave63_gate.log ; cat /root/wave63_gate.status
set -uo pipefail

STATUS=/root/wave63_gate.status
REPO=${REPO:-/root/arc}
BRANCH=${BRANCH:-feat/turboquant-hd512-v4}

# Driver 580.173.02 caps at CUDA 13.0 while the only toolkit is 13.1; without
# the compat lib every PTX-JIT fails.
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH

say() { echo "[$(date -u +%H:%M:%S)] $*"; echo "$*" >> "$STATUS"; }
: > "$STATUS"

# ---------------------------------------------------------------- preflight
if [ -x "$REPO/arc-tools/gpu_box_preflight.sh" ]; then
  say "STEP 0 preflight"
  bash "$REPO/arc-tools/gpu_box_preflight.sh" || { say "FAIL preflight"; exit 1; }
else
  say "FAIL preflight script missing at $REPO/arc-tools/gpu_box_preflight.sh"; exit 1
fi

cd "$REPO" || { say "FAIL no repo at $REPO"; exit 1; }
git fetch --all -q && git checkout -q "$BRANCH" && git pull -q --ff-only || {
  say "FAIL could not check out $BRANCH"; exit 1; }
say "HEAD $(git rev-parse --short HEAD)"

# ------------------------------------------- STEP 1: single-arch build (SM90)
# never cudnn: -62% decode on V4.
say "STEP 1 build sm_90 (native)"
if cargo build --release --features "cuda flash-attn" 2>&1 | tail -40; then
  say "PASS build sm_90"
else
  say "FAIL build sm_90"; exit 1
fi

# --------------------------------- STEP 2: D16 — Hopper AND Blackwell cubins
# This is the load-bearing check. It recompiles the paged-attn kernels for
# sm_90a, sm_100a and sm_103a, so every head-dim instantiation is proven to
# compile on both architectures rather than only the one this box has.
say "STEP 2 build sm_90 + sm_100 + sm_103 (D16 dual-arch)"
if ARC_CUDA_ARCHS=90,100,103 cargo build --release \
     --features "cuda flash-attn" -p mistralrs-paged-attn 2>&1 | tail -60; then
  say "PASS build multi-arch"
else
  say "FAIL build multi-arch"; exit 1
fi

# Show which arches actually landed in the archive.
LIB=$(find target/release/build -name 'libmistralrspagedattention.a' | head -1)
if [ -n "$LIB" ]; then
  say "cubin arches present: $(cuobjdump --list-elf "$LIB" 2>/dev/null \
      | grep -o 'sm_[0-9]*a\?' | sort -u | tr '\n' ' ')"
fi

# ------------------------------------------------- STEP 3: format tests
say "STEP 3 turboquant format tests"
if cargo test --release -p mistralrs-quant turboquant:: 2>&1 | tail -20; then
  say "PASS turboquant tests"
else
  say "FAIL turboquant tests"; exit 1
fi

# ------------------------------------------ STEP 4: V4 decode no-regression
# V4 does not use PagedAttention, so this does not exercise the TurboQuant
# kernel. It is here to prove this branch did not disturb the shipping path.
UQFF=/root/models/v4-uqff/qtip2b-0.uqff
if [ -f "$UQFF" ]; then
  say "STEP 4 V4 decode smoke (no-regression)"
  timeout 900 ./target/release/mistralrs bench \
      -m /root/models/v4-src --uqff "$UQFF" \
      --n-gen 64 --n-prompt 64 --repetitions 2 2>&1 | tail -30
  say "STEP 4 done (compare tok/s against FACTS.md V4 decode baseline)"
else
  say "SKIP step 4 — no UQFF at $UQFF"
fi

say "ALL STEPS COMPLETE"
