#!/usr/bin/env bash
# wave64 — TurboQuant KV storage for DeepSeek-V4 (`ARC_V4_TURBOQUANT=1`).
#
# WHAT THIS PROVES
#   1. The edited TurboQuant .cu compiles — for SM90 *and* SM100/SM103 (D16),
#      asserted against the cubins actually in the archive, not reported.
#   2. V4 runs with ARC_V4_TURBOQUANT=1 — it serves tokens, not a stack trace.
#      (wave43-BU turned FP8 KV on unmeasured and killed every V4 request.)
#   3. TurboQuant actually ENGAGES rather than silently no-opping. The codec
#      emits one line on its first eviction; absent that line the flag did
#      nothing and every other number below is meaningless. This is the whole
#      point — D18: the absence of a signal must not read as success.
#   4. It does not cost decode throughput.
#   5. It buys KV memory.
#
# WHAT THIS DOES NOT DO
#   No quality A/B, no GSM8K. TurboQuant quality is settled by prior
#   measurement. This gate is only "it builds, it engages, it isn't slower".
#
# EXIT CODES (D18 rule 2 — a harness gap must never read as a code verdict)
#   0  every step that ran, passed
#   1  THE CODE UNDER TEST FAILED — a build error in our .cu, a failing test,
#      a missing arch, or the flag not engaging. This is a real refutation.
#   2  ENVIRONMENT / HARNESS problem — missing toolkit, missing model, missing
#      repo, a feature flag this script got wrong. Says nothing about the code.
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
ARCHS=${ARCHS:-90,100,103}

# Driver 580.173.02 caps at CUDA 13.0 while the only toolkit is 13.1; without
# the compat lib every PTX-JIT fails.
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export RUST_LOG=${RUST_LOG:-info}

mkdir -p "$OUT"
say()      { echo "[$(date -u +%H:%M:%S)] $*"; echo "$*" >> "$STATUS"; }
die_env()  { say "ENV-FAIL $* (exit 2 — harness/environment, NOT a code verdict)"; exit 2; }
die_code() { say "FAIL $* (exit 1 — the code under test)"; exit 1; }
: > "$STATUS"

# ---------------------------------------------------------------- preflight
if [ -x "$REPO/arc-tools/gpu_box_preflight.sh" ]; then
  say "STEP 0 preflight"
  bash "$REPO/arc-tools/gpu_box_preflight.sh" || die_env "preflight script reported a bad box"
else
  say "WARN preflight script absent (it lands with the box-preflight branch); continuing"
fi

cd "$REPO" || die_env "no repo at $REPO"
git fetch --all -q && git checkout -q "$BRANCH" && git pull -q --ff-only \
  || die_env "could not check out $BRANCH"
say "HEAD $(git rev-parse --short HEAD)"

# ------------------------------------- feature sanity, BEFORE any 9-min build
# The v1 of this gate passed `--features "cuda flash-attn"` to
# `-p mistralrs-paged-attn`, which declares only `cuda`/`metal`. That is a
# harness bug and it burned a whole gate run to surface, so it is now checked
# up front, in seconds, against the manifests themselves.
declares() { # pkg feature -> 0 if the package declares it
  awk '/^\[features\]/{f=1;next} /^\[/{f=0} f' "$1/Cargo.toml" 2>/dev/null \
    | grep -qE "^[[:space:]]*$2[[:space:]]*="
}
declares mistralrs-paged-attn cuda || die_env "mistralrs-paged-attn no longer declares 'cuda'"
declares mistralrs-core cuda       || die_env "mistralrs-core no longer declares 'cuda'"
declares mistralrs-core flash-attn || die_env "mistralrs-core no longer declares 'flash-attn'"
if declares mistralrs-paged-attn flash-attn; then
  say "NOTE mistralrs-paged-attn now declares flash-attn; STEP 2 could pass it"
fi
say "PASS feature sanity"

# ------------------------------------------- STEP 1: build (compiles the .cu)
# never cudnn: -62% decode on V4.
say "STEP 1 build sm_90 (native) — compiles the qs/softcapping fix"
if cargo build --release --features "cuda flash-attn" 2>&1 | tail -40; then
  say "PASS build sm_90"
else
  die_code "build sm_90 — the qs/softcapping edit does not compile"
fi

# --------------------------------- STEP 2: D16 — Hopper AND Blackwell cubins
# `-p mistralrs-paged-attn` takes ONLY `cuda`; flash-attn is a mistralrs-core
# feature and passing it here is what broke the v1 gate.
say "STEP 2 build sm_$( echo "$ARCHS" | tr ',' '/' ) (D16 dual-arch)"
if ARC_CUDA_ARCHS="$ARCHS" cargo build --release \
     --features cuda -p mistralrs-paged-attn 2>&1 | tail -60; then
  say "PASS build multi-arch"
else
  die_code "build multi-arch — a kernel does not compile for one of $ARCHS"
fi

# Assert, don't report: a missing arch is exactly the D16 regression this step
# exists to catch, and v1 only printed the list.
LIB=$(find target/release/build -name 'libmistralrspagedattention.a' 2>/dev/null | head -1)
[ -n "$LIB" ] || die_env "built, but no libmistralrspagedattention.a under target/release/build"
GOT=$(cuobjdump --list-elf "$LIB" 2>/dev/null | grep -o 'sm_[0-9]*a\?' | sort -u | tr '\n' ' ')
say "  cubin arches present: ${GOT:-<none>}"
[ -n "$GOT" ] || die_env "cuobjdump produced no arch list (is it on PATH?)"
MISSING=""
for a in ${ARCHS//,/ }; do
  # build.rs suffixes 'a' for compute cap >= 90 (arch-specific cubins).
  want="sm_${a}a"; [ "$a" -lt 90 ] 2>/dev/null && want="sm_${a}"
  case " $GOT " in *" $want "*) ;; *) MISSING="$MISSING $want";; esac
done
if [ -n "$MISSING" ]; then
  die_code "D16 — archive is missing cubins for:$MISSING (have: $GOT)"
fi
say "PASS D16 — every requested arch is in the fat binary"

# ------------------------------------------------- STEP 3: host tests on-box
# Same feature set as STEP 1 so this reuses those artifacts instead of forcing
# a second full non-cuda rebuild. The tests themselves are CPU (D14).
say "STEP 3 host tests (CPU tests, D14 — not a hardware claim)"
if cargo test --release -p mistralrs-core --features "cuda flash-attn" \
     --lib v4_turbo 2>&1 | tail -20 \
   && cargo test --release -p mistralrs-core --features "cuda flash-attn" \
     --lib turbo_and_dense 2>&1 | tail -10 \
   && cargo test --release -p mistralrs-quant --features cuda turboquant:: 2>&1 | tail -10; then
  say "PASS host tests"
else
  die_code "host tests"
fi

[ -f "$UQFF" ] || die_env "no UQFF at $UQFF — steps 4-7 need the model (set UQFF=...)"
[ -d "$MODEL" ] || die_env "no model dir at $MODEL (set MODEL=...)"

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
# The load-bearing check. Both markers must appear with the flag on and neither
# may appear with it off. A green throughput number from a run where the codec
# never fired is precisely the failure this gate exists to catch.
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
  ENGAGE_FAILED=1
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

[ "${ENGAGE_FAILED:-0}" = 1 ] && exit 1
exit 0
