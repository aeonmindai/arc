#!/usr/bin/env bash
#
# wave66 — settle the wgmma shared-memory MATRIX DESCRIPTOR on real Hopper.
#
# WHY: `wgmma.mma_async` takes its B operand through a 64-bit descriptor whose
# bit table lives in PTX ISA section 9.7.16.5.1.2.2 — a section that is listed
# in the published table of contents and whose BODY IS TRUNCATED in the HTML
# (checked 2026-08-17). Same for the instruction's operand order in 9.7.16.5.2.
# A wrong descriptor does not fault; it reads the wrong bytes and returns wrong
# numbers, on the one kernel nobody else can adopt from us. So we do not write
# it from documentation. The hardware answers; we read the answer.
#
# WHAT IT COSTS THE BOX: one nvcc compile (~20 s) and four single-block kernel
# launches. It allocates ~1 MB and touches one SM for microseconds. It is a
# CORRECTNESS probe (bit-exact compare), so nothing else running can perturb
# its verdict — but it WILL perturb a timing measurement in flight. Schedule it
# in a gap between curve runs, not alongside one.
#
# LAUNCH (main runs this; agents do not get a shell — D15). Note there is no
# `git checkout` anywhere: the box's working tree is shared, so this reads the
# two files out of the object store and never mutates /root/arc.
#
#   GATE=/root/arc-probe-wave66-wgmma
#   BR=perf/wgmma-descriptor-probe
#   mkdir -p $GATE && cd /root/arc \
#     && git fetch origin "$BR:refs/remotes/origin/$BR" \
#     && git show "origin/$BR:arc-tools/probe/wave66_wgmma_desc_probe.sh" > $GATE/run.sh \
#     && setsid nohup bash $GATE/run.sh < /dev/null > $GATE/run.log 2>&1 &
#
# POLL (bounded output):
#   tail -40 /root/arc-probe-wave66-wgmma/status.txt
#
# EXTRACT when DONE appears: every PROBE_* line in status.txt. The load-bearing
# ones are PROBE_ARITY_ACCEPTED, PROBE_CONTROL_MMA_SYNC, PROBE_MATCH and
# PROBE_STATUS.
#
# Exit codes (D18 rule 2 — never collapse these):
#   0 = a unique descriptor encoding was identified
#   1 = the probe answered NEGATIVELY (no candidate matched, or ambiguous)
#   2 = the environment could not answer (no toolkit, no GPU, compile failed)
#
# Paths are namespaced per D19.

set -uo pipefail

GATE_DIR=${GATE_DIR:-/root/arc-probe-wave66-wgmma}
STATUS="$GATE_DIR/status.txt"
WORK="$GATE_DIR/work"
REPO=${REPO:-/root/arc}
BRANCH=${BRANCH:-perf/wgmma-descriptor-probe}
SRC_PATH=arc-tools/probe/arctarget_wgmma_desc_probe.cu

mkdir -p "$GATE_DIR" "$WORK"
: > "$STATUS"

RC=0
say()  { echo "$*" | tee -a "$STATUS"; }
pass() { say "PASS  $*"; }
fail() { say "FAIL  $*"; RC=1; }
skip() { say "SKIP  $*"; }               # could not answer, not a negative
envr() { say "ENV   $*"; RC=2; }
done_and_exit() { say "DONE rc=$RC"; exit $RC; }

say "wgmma descriptor probe wave66 — $(date -u +%FT%TZ)"

# --- 0. preflight -----------------------------------------------------------
if [ -x /root/arc-tools/gpu_box_preflight.sh ]; then
  if /root/arc-tools/gpu_box_preflight.sh > "$WORK/preflight.log" 2>&1; then
    pass "box preflight"
  else
    fail "box preflight (see $WORK/preflight.log)"
    done_and_exit
  fi
else
  skip "box preflight — /root/arc-tools/gpu_box_preflight.sh not present"
fi

command -v nvcc >/dev/null 2>&1 || { envr "nvcc not on PATH"; done_and_exit; }
say "INFO  nvcc: $(nvcc --version | tail -2 | head -1)"
say "INFO  gpu:  $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"

# The probe needs sm_90a specifically: `wgmma` does not exist on plain sm_90.
#
# This used to grep `nvcc --list-gpu-code` for `sm_90a`. That check is worthless
# — the listing never prints the `a` variants at all (CUDA 12.9 shows `sm_90`,
# `sm_100`, `sm_103` and no arch-specific forms), so it would have reported
# "toolkit cannot target sm_90a" on a toolkit that can. The inverse also bit us:
# `-arch=sm_90a` was accepted without complaint and then produced a
# `compute_90` intermediate, so ptxas rejected wgmma as "not supported on
# .target 'sm_90'".
#
# So: emit PTX and read its `.target` back. Ask the tool what it DID, not what
# it claims to support.
echo '__global__ void k(){}' > "$WORK/tgt.cu"
nvcc -gencode arch=compute_90a,code=sm_90a -ptx -o "$WORK/tgt.ptx" "$WORK/tgt.cu" \
     > "$WORK/tgt.log" 2>&1 || true
PTX_TARGET=$(grep -m1 '^\.target' "$WORK/tgt.ptx" 2>/dev/null || echo "<none>")
say "PROBE_PTX_TARGET=$PTX_TARGET"
case "$PTX_TARGET" in
  *sm_90a*) pass "toolkit emits sm_90a — wgmma is assemblable" ;;
  *) envr "asked for compute_90a, got '$PTX_TARGET' — wgmma unreachable from this CUDA"
     done_and_exit ;;
esac

# --- 1. get the source without touching the shared working tree -------------
if [ -d "$REPO/.git" ]; then
  if git -C "$REPO" show "origin/$BRANCH:$SRC_PATH" > "$WORK/probe.cu" 2>"$WORK/show.err"; then
    pass "extracted $SRC_PATH from origin/$BRANCH (working tree untouched)"
  else
    envr "could not read $SRC_PATH from origin/$BRANCH — fetch it first ($(head -1 "$WORK/show.err"))"
    done_and_exit
  fi
else
  envr "no git repo at $REPO"
  done_and_exit
fi

# --- 2. settle the ARITY by asking the compiler -----------------------------
# The register-A form of wgmma.mma_async takes either 7 or 8 operands after the
# destination — whether an `imm-trans-a` precedes `imm-trans-b` is exactly the
# thing the truncated doc will not say. A wrong arity is a ptxas ERROR, which is
# loud; a wrong descriptor is silent. So arity is settled by the compiler and
# the descriptor by the hardware, and neither is guessed.
ARITY_OK=""
BIN=""
for HAS_TRANS_A in 0 1; do
  out="$WORK/probe_ta$HAS_TRANS_A"
  if nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 -O3 \
       -DWGMMA_HAS_TRANS_A=$HAS_TRANS_A \
       -o "$out" "$WORK/probe.cu" > "$WORK/nvcc_ta$HAS_TRANS_A.log" 2>&1; then
    say "INFO  compiled with WGMMA_HAS_TRANS_A=$HAS_TRANS_A"
    if [ -z "$ARITY_OK" ]; then ARITY_OK=$HAS_TRANS_A; BIN="$out"; fi
  else
    say "INFO  rejected WGMMA_HAS_TRANS_A=$HAS_TRANS_A: $(grep -m2 -i 'error' "$WORK/nvcc_ta$HAS_TRANS_A.log" | tr '\n' ' ' | cut -c1-200)"
  fi
done

if [ -z "$ARITY_OK" ]; then
  envr "neither operand arity compiled — see $WORK/nvcc_ta0.log and $WORK/nvcc_ta1.log"
  say "PROBE_ARITY_ACCEPTED=none"
  done_and_exit
fi
say "PROBE_ARITY_ACCEPTED=$ARITY_OK"

# --- 3. run it --------------------------------------------------------------
say "STEP  running the 400-variant sweep (4 configs x 100 descriptor candidates)"
"$BIN" > "$WORK/probe.out" 2>&1
PROBE_RC=$?
if ! grep -E '^PROBE_' "$WORK/probe.out" >> "$STATUS"; then
  # Silence here would read as "it ran and found nothing", which is a different
  # claim from "it produced no verdict at all" (D18).
  say "WARN  probe emitted no PROBE_ lines (see $WORK/probe.out)"
fi

case "$PROBE_RC" in
  0) pass "unique descriptor encoding identified"; RC=0 ;;
  1) fail "probe answered negatively — see PROBE_STATUS above"; RC=1 ;;
  *) envr "probe could not run (rc=$PROBE_RC)"; RC=2 ;;
esac

done_and_exit
