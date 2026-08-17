#!/usr/bin/env bash
#
# ArcTarget gate (wave65) — prove the arch matrix is REAL, on the box.
#
# This gate exists because the previous arch claim was false and nothing
# noticed: on 2026-08-17 a build asking for `ARC_CUDA_ARCHS=90,100,103` on a
# live H200 exited 0 and produced an archive whose only cubin was `sm_90a`. The
# compile passing was read as the architectures being present. So this gate
# never trusts a compile: it asks `cuobjdump` what is actually in the archive,
# and it includes negative controls that prove the check can still say no.
#
# LAUNCH (main runs this; agents do not get a shell — D15):
#
#   GATE=/root/arc-gate-wave65-arctarget
#   mkdir -p $GATE && cd /root/arc && git fetch origin \
#     && git checkout arctarget/multiarch-specialization \
#     && setsid nohup bash arc-tools/wave65_arctarget_gate.sh \
#          < /dev/null > $GATE/run.log 2>&1 &
#
# POLL (bounded output):
#   tail -30 /root/arc-gate-wave65-arctarget/status.txt
#
# EXTRACT when DONE appears:
#   the status.txt lines — every step prints one, including SKIP with a reason.
#
# Exit codes (D18 rule 2 — never collapse these):
#   0 = every gate passed
#   1 = a gate genuinely failed (the strategy signal)
#   2 = the environment could not answer (no toolkit, no GPU, no repo)
#
# Paths are namespaced per D19: shared /root/*.log has already caused one
# agent's output to be published as another's.

set -uo pipefail

GATE_DIR=${GATE_DIR:-/root/arc-gate-wave65-arctarget}
STATUS="$GATE_DIR/status.txt"
WORK="$GATE_DIR/work"
mkdir -p "$GATE_DIR" "$WORK"
: > "$STATUS"

RC=0
say()  { echo "$*" | tee -a "$STATUS"; }
pass() { say "PASS  $*"; }
fail() { say "FAIL  $*"; RC=1; }
skip() { say "SKIP  $*"; }               # could not answer, not a negative
envr() { say "ENV   $*"; RC=2; }

say "ArcTarget gate wave65 — $(date -u +%FT%TZ)"

# --- 0. preflight -----------------------------------------------------------
if [ -x /root/arc-tools/gpu_box_preflight.sh ]; then
  if /root/arc-tools/gpu_box_preflight.sh > "$WORK/preflight.log" 2>&1; then
    pass "box preflight"
  else
    fail "box preflight (see $WORK/preflight.log)"
    say "DONE rc=$RC"; exit $RC
  fi
else
  skip "box preflight — /root/arc-tools/gpu_box_preflight.sh not present"
fi

command -v nvcc >/dev/null 2>&1 || { envr "nvcc not on PATH"; say "DONE rc=$RC"; exit $RC; }
command -v cuobjdump >/dev/null 2>&1 || { envr "cuobjdump not on PATH"; say "DONE rc=$RC"; exit $RC; }
REPO=${REPO:-$(pwd)}
[ -f "$REPO/Cargo.toml" ] || { envr "not a repo root: $REPO"; say "DONE rc=$RC"; exit $RC; }

say "INFO  nvcc: $(nvcc --version | tail -2 | head -1)"
say "INFO  gpu:  $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"

# --- 1. does this toolkit even support the arches we claim? ------------------
# A missing sm_103a here is a TOOLKIT limit, not an Arc failure. Separating the
# two is the whole point of having a step for it.
CODES=$(nvcc --list-gpu-code 2>/dev/null | tr -d ' ')
TOOLKIT_ARCHS=""
for a in sm_80 sm_89 sm_90a sm_100a sm_103a; do
  if echo "$CODES" | grep -qx "$a"; then
    TOOLKIT_ARCHS="$TOOLKIT_ARCHS $a"
  else
    say "INFO  toolkit does NOT support $a"
  fi
done
say "INFO  toolkit supports:$TOOLKIT_ARCHS"

# Which predefined macro distinguishes an arch-SPECIFIC compile (sm_90a, where
# wgmma exists) from a plain one (sm_90, where it does not? The guard every
# future Hopper/Blackwell path needs is exactly this macro, and its name is not
# stated in the public docs — so ask the compiler instead of guessing. A path
# guarded only on `__CUDA_ARCH__ >= 900` would fail to compile the moment
# someone builds plain sm_90.
for arch in sm_90 sm_90a sm_100a; do
  echo "$TOOLKIT_ARCHS" | grep -q "$arch" || continue
  defs=$(nvcc -arch=$arch -dM -E -x cu /dev/null 2>/dev/null \
         | grep -i '__CUDA_ARCH' | sort | tr '\n' ' ')
  say "INFO  $arch predefines: ${defs:-<none captured>}"
done
# Build the request from what the toolkit can actually produce, so a toolkit
# gap fails as a toolkit gap rather than as an Arc bug.
REQ=""
for pair in "sm_80:80" "sm_90a:90" "sm_100a:100" "sm_103a:103"; do
  code=${pair%%:*}; cc=${pair##*:}
  echo "$TOOLKIT_ARCHS" | grep -q "$code" && REQ="${REQ:+$REQ,}$cc"
done
[ -n "$REQ" ] || { envr "toolkit supports none of sm_80/90a/100a/103a"; say "DONE rc=$RC"; exit $RC; }
say "INFO  requesting ARC_CUDA_ARCHS=$REQ"

cd "$REPO" || { envr "cannot cd $REPO"; say "DONE rc=$RC"; exit $RC; }

# archs_in <archive>  ->  sorted unique cubin arch list, straight from cuobjdump
archs_in() {
  cuobjdump --list-elf "$1" 2>/dev/null \
    | grep -o 'sm_[0-9]\+[af]\?' | sort -u | tr '\n' ',' | sed 's/,$//'
}

find_archive() { find target -name "$1" -newermt '-6 hours' 2>/dev/null | head -1; }

# --- 2. the real thing: multi-arch build of the quant kernels ----------------
say "STEP  building mistralrs-quant with ARC_CUDA_ARCHS=$REQ (this is the slow one)"
if ARC_CUDA_ARCHS="$REQ" cargo build -p mistralrs-quant --lib --features cuda \
     > "$WORK/quant_multi.log" 2>&1; then
  QA=$(find_archive libmistralrsquant.a)
  if [ -z "$QA" ]; then
    fail "quant archive not found after a successful build"
  else
    GOT=$(archs_in "$QA")
    say "INFO  quant archive cubins: $GOT"
    MISSING=""
    for cc in ${REQ//,/ }; do
      want="sm_${cc}"; [ "$cc" -ge 90 ] && want="sm_${cc}a"
      echo "$GOT" | grep -q "\b$want\b" || MISSING="$MISSING $want"
    done
    if [ -n "$MISSING" ]; then
      fail "quant archive is missing:$MISSING (has: $GOT)"
    else
      pass "quant archive carries every requested arch ($GOT)"
    fi
  fi
else
  # The build.rs assertion failing IS a legitimate negative signal — it means
  # nvcc did not emit what was asked. Show the reason rather than just "FAIL".
  fail "mistralrs-quant multi-arch build failed"
  say "INFO  $(grep -E 'ArcTarget:|error:' "$WORK/quant_multi.log" | tail -5)"
fi

# --- 3. the CUDA-gated Rust must type-check too ------------------------------
# A macOS `cargo check` does not type-check any of this; one PR shipped 15 such
# errors. Step 2 already compiled the lib; this covers the test targets.
if cargo test -p mistralrs-quant --features cuda --no-run > "$WORK/quant_tests.log" 2>&1; then
  pass "cuda-gated Rust type-checks (lib + tests)"
else
  fail "cuda-gated Rust does not compile"
  say "INFO  $(grep -E '^error' "$WORK/quant_tests.log" | head -5)"
fi

# --- 4. discrimination: the check must be able to say NO ---------------------
# Same crate, fewer arches. If sm_100a shows up here the check is not reading
# the archive; if it is missing from step 2 the check is not reading it either.
say "STEP  discrimination build (arc-cuda-graph, 80,90 only)"
if ARC_CUDA_ARCHS="80,90" cargo build -p arc-cuda-graph --lib --features cuda \
     > "$WORK/graph_two.log" 2>&1; then
  GA=$(find_archive libarccudagraph.a)
  GOT2=$(archs_in "$GA")
  say "INFO  arc-cuda-graph cubins: $GOT2"
  if echo "$GOT2" | grep -q 'sm_80' && echo "$GOT2" | grep -q 'sm_90a' \
     && ! echo "$GOT2" | grep -q 'sm_100a'; then
    pass "two-arch build contains exactly sm_80+sm_90a and NOT sm_100a"
  else
    fail "two-arch build produced '$GOT2' — the arch check does not discriminate"
  fi
else
  fail "arc-cuda-graph two-arch build failed"
  say "INFO  $(grep -E 'ArcTarget:|error:' "$WORK/graph_two.log" | tail -5)"
fi

# --- 5. negative control: an unverifiable claim must REFUSE to build ---------
# With cuobjdump unavailable the build has made a multi-arch claim it cannot
# back, and must fail. If this SUCCEEDS, the gate in build.rs is not running.
say "STEP  negative control (CUOBJDUMP=/bin/false must fail the build)"
touch arc-cuda-graph/build.rs
if ARC_CUDA_ARCHS="80,90" CUOBJDUMP=/bin/false \
     cargo build -p arc-cuda-graph --lib --features cuda > "$WORK/negctl.log" 2>&1; then
  fail "build SUCCEEDED with cuobjdump unavailable — the verification is not running"
else
  if grep -q "ArcTarget" "$WORK/negctl.log"; then
    pass "unverifiable arch claim refused to build"
  else
    fail "build failed for some other reason: $(grep -E '^error' "$WORK/negctl.log" | head -2)"
  fi
fi

# --- 6. no-regression: the default (unset) path still builds -----------------
say "STEP  default build (ARC_CUDA_ARCHS unset)"
touch arc-cuda-graph/build.rs
if cargo build -p arc-cuda-graph --lib --features cuda > "$WORK/graph_default.log" 2>&1; then
  GA=$(find_archive libarccudagraph.a)
  say "INFO  default cubins: $(archs_in "$GA")"
  pass "default single-arch build unchanged"
else
  fail "default build regressed"
fi

# --- 7. the wgmma descriptor probe -----------------------------------------
say "STEP  wgmma shared-memory descriptor probe"
if nvcc -arch=sm_90a -std=c++17 -O3 -o "$WORK/wgmma_probe" \
     arc-tools/probe/arctarget_wgmma_desc_probe.cu > "$WORK/probe_build.log" 2>&1; then
  "$WORK/wgmma_probe" > "$WORK/probe_run.log" 2>&1
  prc=$?
  while IFS= read -r line; do say "PROBE $line"; done < "$WORK/probe_run.log"
  case $prc in
    0) pass "descriptor encoding identified on hardware" ;;
    1) fail "descriptor probe answered negatively (see PROBE lines)" ;;
    *) skip "descriptor probe could not run (rc=$prc)" ;;
  esac
else
  # A PTX syntax error here is loud and is information: it means the wgmma
  # operand form in the probe needs the ISA doc, not that the descriptor is
  # wrong. Report it as such.
  fail "descriptor probe did not compile"
  say "INFO  $(grep -E 'error|ptxas' "$WORK/probe_build.log" | head -5)"
fi

# --- 8. Blackwell: COMPILE-ONLY. This is not a measurement. ------------------
say "STEP  Blackwell compile-only check (D14: we rent SM90; nothing here is benchmarked)"
if echo "$TOOLKIT_ARCHS" | grep -q sm_100a; then
  cat > "$WORK/tcgen05_probe.cu" <<'EOF'
// Does this toolkit assemble tcgen05 mnemonics for sm_100a? Compile only —
// no numerics are claimed and none are computed.
#include <cstdint>
__global__ void k(uint32_t* addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(addr)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
#else
    (void)addr;
#endif
}
EOF
  if nvcc -arch=sm_100a -std=c++17 -c -o "$WORK/tcgen05_probe.o" "$WORK/tcgen05_probe.cu" \
       > "$WORK/tcgen05.log" 2>&1; then
    pass "tcgen05 mnemonics assemble for sm_100a (COMPILE-ONLY, never executed)"
  else
    say "INFO  $(grep -E 'error|ptxas' "$WORK/tcgen05.log" | head -3)"
    fail "tcgen05 snippet did not assemble for sm_100a"
  fi
else
  skip "sm_100a unsupported by this toolkit — Blackwell claim NOT verifiable here"
fi

say "DONE rc=$RC"
exit $RC
