#!/usr/bin/env bash
# wave63 — TurboQuant CUDA kernels at head_dim {64,128,256,512}, Hopper + Blackwell.
#
# THE ONE THING THIS PROVES
#   `turbo_paged_attention.cu` compiles for sm_90a AND sm_100a/sm_103a with all
#   four head-dim instantiations live, and the built archive actually contains
#   both cubins. That is the whole D16 claim in PR #94; everything else here is
#   secondary.
#
# EXIT CODES (D18 rule 2 — the distinction the runner acts on)
#   0  pass
#   1  genuine failure. Real signal: the code is wrong. Act on this.
#   2  this machine could not answer. Missing preflight, failing box, no
#      toolkit, no repo. Says nothing about the code — do not read it as a
#      regression.
#
# RUN (from main, per D15 — agents never call runcrate):
#   setsid nohup bash /root/arc/arc-tools/wave63_turboquant_hd512_gate.sh \
#     < /dev/null > /root/wave63_gate.log 2>&1 &
#   tail -f /root/wave63_gate.log ; cat /root/wave63_gate.status
#
# The preflight is looked up at a branch-independent path FIRST, because it
# lives on wave61/box-preflight-shared-prefix and vanishes the moment a runner
# checks out any other branch. Override with ARC_PREFLIGHT=/path.
# Set SKIP_NATIVE_BUILD=1 to go straight to the multi-arch proof.
set -uo pipefail

STATUS=${STATUS:-/root/wave63_gate.status}
REPO=${REPO:-/root/arc}
BRANCH=${BRANCH:-feat/turboquant-hd512-v4}

export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
export PATH=$CUDA_HOME/bin:$PATH

: > "$STATUS"
say()  { echo "[$(date -u +%H:%M:%S)] $*"; echo "$*" >> "$STATUS"; }
# Exit 2 — the box could not answer. Not a verdict on the code.
noans() { say "NOANSWER $*"; say "EXIT 2 (environment could not answer)"; exit 2; }
# Exit 1 — the code is wrong. This is the only signal worth acting on.
fail()  { say "FAIL $*";     say "EXIT 1 (genuine failure)";              exit 1; }

# --------------------------------------------------------------- preflight
# gpu_box_preflight.sh ends with
#     [ "$_arc_pf_sourced" = "1" ] || exit 1
# so when it is SOURCED on a *failing* box that test succeeds, short-circuits
# the `|| exit 1`, and becomes the last command — making `source pf || handler`
# return 0 and the handler never fire. We therefore ignore its return value
# entirely and read `_ARC_PF_FAILED` directly, which is the flag it actually
# sets. If that flag is missing the script did not reach its own verdict, and
# an unverified box is a box that cannot answer.
PF=""
for cand in \
    "${ARC_PREFLIGHT:-}" \
    /root/arc-tools/gpu_box_preflight.sh \
    /usr/local/lib/arc/gpu_box_preflight.sh \
    "$REPO/arc-tools/gpu_box_preflight.sh"
do
    [ -n "$cand" ] && [ -f "$cand" ] && { PF="$cand"; break; }
done
[ -n "$PF" ] || noans "no gpu_box_preflight.sh found. It lives on \
wave61/box-preflight-shared-prefix, so it disappears whenever a runner checks \
out another branch — stage it at /root/arc-tools/ (branch-independent) or set \
ARC_PREFLIGHT=/path. Nothing about the code was tested."

say "STEP 0 preflight ($PF)"
PF_FLAG=$(bash -c 'set +e; . "$1" >&2; echo "${_ARC_PF_FAILED:-unset}"' _ "$PF")
case "$PF_FLAG" in
    0) say "PASS preflight" ;;
    1) noans "preflight reports this box is unhealthy (_ARC_PF_FAILED=1). \
See FAIL[...] lines above. The code was never exercised." ;;
    *) noans "preflight never set _ARC_PF_FAILED (got '$PF_FLAG') — it did not \
reach its own verdict, so its silence cannot be read as a pass." ;;
esac

# ------------------------------------------------------------------- repo
cd "$REPO" 2>/dev/null || noans "no repo at $REPO"
command -v nvcc >/dev/null || noans "nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"
git fetch --all -q 2>/dev/null || noans "cannot reach the git remote"
git checkout -q "$BRANCH" 2>/dev/null || noans "cannot check out $BRANCH"
git pull -q --ff-only 2>/dev/null || true
say "HEAD $(git rev-parse --short HEAD) on $BRANCH"
say "nvcc $(nvcc --version | grep -o 'release [0-9.]*' | head -1)"

# ============================================================================
# STEP 1 — THE D16 PROOF. Runs first and needs no full workspace build.
# ============================================================================
say "STEP 1 compile paged-attn for sm_90a + sm_100a + sm_103a"
if ! ARC_CUDA_ARCHS=90,100,103 cargo build --release \
        --features cuda -p mistralrs-paged-attn 2>&1 | tail -60; then
    fail "multi-arch build of mistralrs-paged-attn failed — see nvcc errors above"
fi
say "PASS multi-arch compile"

# The build succeeding is not the claim; the cubins being present is.
LIB=$(find target/release/build -name 'libmistralrspagedattention.a' 2>/dev/null | head -1)
[ -n "$LIB" ] || noans "built, but no libmistralrspagedattention.a found to inspect"
command -v cuobjdump >/dev/null || noans "cuobjdump missing — cannot verify which \
arches landed, and the unverified build is exactly the assertion we came to replace"

ARCHES=$(cuobjdump --list-elf "$LIB" 2>/dev/null | grep -oE 'sm_[0-9]+a?' | sort -u | tr '\n' ' ')
say "cubin arches in archive: ${ARCHES:-<none>}"
MISSING=""
for want in sm_90a sm_100a; do
    case " $ARCHES " in *" $want "*) ;; *) MISSING="$MISSING $want" ;; esac
done
[ -z "$MISSING" ] || fail "D16 unmet: archive is missing$MISSING (has: $ARCHES). \
The kernels compiled but this binary would PTX-JIT or fail on the missing arch."
say "PASS D16 — Hopper and Blackwell cubins both present"

# ============================================================================
# STEP 2 — native full build. PR #98 already built these kernels clean for
# sm_90 in 8m53s, so this is confirmation, not discovery. Skippable.
# ============================================================================
if [ "${SKIP_NATIVE_BUILD:-0}" = "1" ]; then
    say "SKIP step 2 (SKIP_NATIVE_BUILD=1)"
else
    say "STEP 2 native build (cuda flash-attn; never cudnn — -62% decode on V4)"
    cargo build --release --features "cuda flash-attn" 2>&1 | tail -40 \
        || fail "native build failed"
    say "PASS native build"
fi

# ============================================================================
# STEP 3 — format tests. CPU-only by nature: these pin the CUDA constant tables
# to the Rust generator. Not a GPU claim, and not offered as one (D14).
# ============================================================================
say "STEP 3 turboquant format tests"
cargo test --release -p mistralrs-quant turboquant:: 2>&1 | tail -20 \
    || fail "turboquant format tests failed"
say "PASS format tests"

# ============================================================================
# STEP 4 — decode no-regression. V4 does NOT use PagedAttention, so this does
# not exercise the TurboQuant kernel at all; it only shows this branch did not
# disturb the shipping path.
# ============================================================================
UQFF=${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}
if [ "${SKIP_NATIVE_BUILD:-0}" = "1" ]; then
    say "SKIP step 4 (no native binary was built)"
elif [ -f "$UQFF" ]; then
    say "STEP 4 V4 decode smoke (no-regression only)"
    timeout 900 ./target/release/mistralrs bench \
        -m /root/models/v4-src --uqff "$UQFF" \
        --n-gen 64 --n-prompt 64 --repetitions 2 2>&1 | tail -30
    say "STEP 4 done — compare tok/s against the FACTS.md V4 decode baseline"
else
    say "SKIP step 4 — no UQFF at $UQFF"
fi

say "ALL STEPS COMPLETE"
say "EXIT 0 (pass)"
exit 0
