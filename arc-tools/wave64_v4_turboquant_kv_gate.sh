#!/usr/bin/env bash
# wave64 — TurboQuant KV storage for DeepSeek-V4 (`ARC_V4_TURBOQUANT=1`).
#
# WHAT THIS PROVES
#   1. The edited TurboQuant .cu still compiles — for SM90 *and* SM100/SM103
#      (D16). This is the FIRST compile of the `qs` / `softcapping` fix: it was
#      written on macOS where nvcc does not exist, and the `cuda` module is
#      cfg-gated so `cargo check` never even parsed the Rust side of it.
#   2. V4 runs with ARC_V4_TURBOQUANT=1 — it serves tokens, not a stack trace.
#      (wave43-BU turned FP8 KV on unmeasured and killed every V4 request.)
#   3. TurboQuant actually ENGAGES rather than silently no-opping. The codec
#      emits one line on its first eviction; absent that line the flag did
#      nothing and every other number below is meaningless. This check is the
#      whole point — D18: the absence of a signal must not read as success.
#   4. It does not cost decode throughput.
#   5. It buys KV memory.
#
# WHAT THIS DOES NOT DO
#   No quality A/B, no GSM8K. TurboQuant quality is settled by prior
#   measurement. This gate is only "it builds, it engages, it isn't slower".
#
# RUN (from main, per D15 — agents never call runcrate):
#   scp this to the box, then:
#     setsid nohup bash /root/arc/arc-tools/wave64_v4_turboquant_kv_gate.sh \
#       < /dev/null > /root/wave64_gate.log 2>&1 &
#   Watch:  tail -f /root/wave64_gate.log ; cat /root/wave64_gate.status
set -uo pipefail

STATUS=/root/wave64_gate.status
REPO=${REPO:-/root/arc}
BRANCH=${BRANCH:-feat/turboquant-v4-cachedk}
MODEL=${MODEL:-/root/models/v4-src}
UQFF=${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}
OUT=${OUT:-/root/wave64}

# Driver 580.173.02 caps at CUDA 13.0 while the only toolkit is 13.1; without
# the compat lib every PTX-JIT fails.
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export RUST_LOG=${RUST_LOG:-info}

mkdir -p "$OUT"
say() { echo "[$(date -u +%H:%M:%S)] $*"; echo "$*" >> "$STATUS"; }
: > "$STATUS"

# ---------------------------------------------------------------- preflight
if [ -x "$REPO/arc-tools/gpu_box_preflight.sh" ]; then
  say "STEP 0 preflight"
  bash "$REPO/arc-tools/gpu_box_preflight.sh" || { say "FAIL preflight"; exit 1; }
else
  say "WARN preflight script absent (it lands with the box-preflight branch); continuing"
fi

cd "$REPO" || { say "FAIL no repo at $REPO"; exit 1; }
git fetch --all -q && git checkout -q "$BRANCH" && git pull -q --ff-only || {
  say "FAIL could not check out $BRANCH"; exit 1; }
say "HEAD $(git rev-parse --short HEAD)"

# ------------------------------------------- STEP 1: build (compiles the .cu)
# never cudnn: -62% decode on V4.
say "STEP 1 build sm_90 (native) — first compile of the qs/softcapping fix"
if cargo build --release --features "cuda flash-attn" 2>&1 | tail -40; then
  say "PASS build sm_90"
else
  say "FAIL build sm_90 — the qs/softcapping edit does not compile"; exit 1
fi

# --------------------------------- STEP 2: D16 — Hopper AND Blackwell cubins
say "STEP 2 build sm_90 + sm_100 + sm_103 (D16 dual-arch)"
if ARC_CUDA_ARCHS=90,100,103 cargo build --release \
     --features "cuda flash-attn" -p mistralrs-paged-attn 2>&1 | tail -60; then
  say "PASS build multi-arch"
else
  say "FAIL build multi-arch"; exit 1
fi
LIB=$(find target/release/build -name 'libmistralrspagedattention.a' | head -1)
if [ -n "$LIB" ]; then
  say "cubin arches present: $(cuobjdump --list-elf "$LIB" 2>/dev/null \
      | grep -o 'sm_[0-9]*a\?' | sort -u | tr '\n' ' ')"
fi

# ------------------------------------------------- STEP 3: host tests on-box
say "STEP 3 host tests"
if cargo test --release -p mistralrs-core --lib v4_turbo 2>&1 | tail -20 \
   && cargo test --release -p mistralrs-core --lib turbo_and_dense 2>&1 | tail -10 \
   && cargo test --release -p mistralrs-quant turboquant:: 2>&1 | tail -10; then
  say "PASS host tests"
else
  say "FAIL host tests"; exit 1
fi

if [ ! -f "$UQFF" ]; then
  say "SKIP steps 4-6 — no UQFF at $UQFF (set UQFF=...)"
  say "ALL STEPS COMPLETE (build-only)"
  exit 0
fi

BENCH=(./target/release/mistralrs bench -m "$MODEL" --uqff "$UQFF"
       --n-gen 128 --n-prompt 2048 --repetitions 3)

# ------------------------------------------------------ STEP 4: A — flag OFF
say "STEP 4 baseline: ARC_V4_TURBOQUANT unset"
timeout 1800 env -u ARC_V4_TURBOQUANT "${BENCH[@]}" > "$OUT/a_off.txt" 2>&1
say "STEP 4 exit=$?"
tail -25 "$OUT/a_off.txt"

# ------------------------------------------------------- STEP 5: B — flag ON
say "STEP 5 turboquant: ARC_V4_TURBOQUANT=1"
timeout 1800 env ARC_V4_TURBOQUANT=1 "${BENCH[@]}" > "$OUT/b_on.txt" 2>&1
say "STEP 5 exit=$?"
tail -25 "$OUT/b_on.txt"

# ------------------------------------------- STEP 6: did it actually ENGAGE?
# The load-bearing check. Both markers must appear with the flag on and
# neither may appear with it off. A green throughput number from a run where
# the codec never fired is the failure this gate exists to catch.
say "STEP 6 engagement"
ON_BUILT=$(grep -c "V4 KV storage: TurboQuant" "$OUT/b_on.txt")
ON_FIRED=$(grep -c "V4 TurboQuant KV engaged" "$OUT/b_on.txt")
OFF_BUILT=$(grep -c "V4 KV storage: TurboQuant" "$OUT/a_off.txt")
OFF_FIRED=$(grep -c "V4 TurboQuant KV engaged" "$OUT/a_off.txt")
say "  flag ON : slots_built=$ON_BUILT  codec_fired=$ON_FIRED"
say "  flag OFF: slots_built=$OFF_BUILT codec_fired=$OFF_FIRED"
grep -m1 "V4 TurboQuant KV engaged" "$OUT/b_on.txt" | tee -a "$STATUS"
if [ "$ON_BUILT" -ge 1 ] && [ "$ON_FIRED" -ge 1 ] \
   && [ "$OFF_BUILT" -eq 0 ] && [ "$OFF_FIRED" -eq 0 ]; then
  say "PASS engagement — the codec ran only with the flag on"
else
  say "FAIL engagement — the flag did not do what it says; ignore steps 4/5 numbers"
fi

# ---------------------------------------------------------- STEP 7: KV bytes
# Peak GPU memory over a long-context run, both settings. The saving is on the
# compressed region only (the window stays dense), so it grows with context.
say "STEP 7 peak GPU memory, 8K context"
for mode in off on; do
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) \
      > "$OUT/mem_$mode.txt" 2>/dev/null &
  MPID=$!
  if [ "$mode" = on ]; then E=(env ARC_V4_TURBOQUANT=1); else E=(env -u ARC_V4_TURBOQUANT); fi
  timeout 1800 "${E[@]}" ./target/release/mistralrs bench -m "$MODEL" --uqff "$UQFF" \
      --n-gen 256 --n-prompt 8192 --repetitions 1 > "$OUT/mem_run_$mode.txt" 2>&1
  kill $MPID 2>/dev/null
  say "  peak MiB ($mode): $(sort -n "$OUT/mem_$mode.txt" | tail -1)"
done

say "ALL STEPS COMPLETE"
say ""
say "EXTRACT THESE FIVE NUMBERS:"
say "  1. STEP 6 must say PASS. If it does not, nothing else counts."
say "  2. decode tok/s, flag OFF   (grep -i 'decode' $OUT/a_off.txt)"
say "  3. decode tok/s, flag ON    (grep -i 'decode' $OUT/b_on.txt)"
say "     -> claim: ON/OFF >= 0.98. Below that, eviction is costing decode."
say "  4. peak MiB off vs on at 8K ctx (STEP 7)"
say "     -> claim: ON < OFF. The saving is ~3.9x on the compressed region."
say "  5. prefill tok/s both files -> must be unchanged (prefill evicts nothing)."
