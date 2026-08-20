#!/usr/bin/env bash
# Assert the QTIP beam kernels compile with NO local-memory spill.
#
# WHY THIS IS A GATE AND NOT A COMMENT
# ------------------------------------
# `qtip_beam.cu` and `qtip2b_beam.cu` both keep a per-thread `cand[]` array in
# registers and both raise `__launch_bounds__(256, 4)` to cap registers at 64.
# `__launch_bounds__` does NOT refuse to compile when it cannot reach that
# budget — it SPILLS to local memory, silently, and a spilled load inside the
# radix loop runs ~CAND x 3.87 times per timestep. Both files name this check by
# name in their source ("`LOCAL:` must remain 0") and, until now, nothing ran it.
#
# It is also the check that separates the two competing cost models for the
# K=8/V=4/L=12 encode. A naive constant bump turns `cand[2^K]` into `cand[256]`,
# which spills for certain — that spilled kernel is where the discredited
# "~1,700 s/layer / 20 h / $30" figure came from. The lane-team re-blocking in
# `qtip_beam.cu` keeps `cand[]` at 16 entries at BOTH geometries; this script is
# what turns that claim into a compiled fact instead of a hand count. (Hand
# counting C++ undercounts SASS by ~2.05x on this kernel family — do not.)
#
# THIS GATE IS PROVEN RED ON EVERY RUN
# ------------------------------------
# A resource gate whose parser silently matches nothing passes on any code at
# all, which is the failure mode that has burned this repo repeatedly. Two
# things make that impossible here:
#   1. A NEGATIVE CONTROL kernel (written to a temp dir, not the repo) with a
#      dynamically-indexed 256-float local array is compiled and pushed through
#      the SAME parser. If it is not reported as a spill, this script fails —
#      the gate has lost its teeth and says so.
#   2. A KERNEL-COUNT assertion per source. `qtip_beam.cu` must yield 4 beam
#      kernels (2 geometries x 2 codebooks); if the K=8/V=4/L=12 instantiation
#      were ever dropped, a "no spill" result would be about a kernel that no
#      longer exists. `qtip2b_beam.cu` yields 1: that rung has no stored-LUT
#      path at all (the codebook is always computed from `mult`), so its kernel
#      is not templated on the codebook the way the LUT rung's is.
#
# Note on the negative control: nvcc reports its spill as `STACK:1024`, not
# `LOCAL:`. That is why the verdict below tests BOTH fields — a LOCAL-only check
# would have called the control clean and disarmed the whole gate.
#
# NO GPU REQUIRED: nvcc + cuobjdump only.
#
# Usage:  arc-tools/qtip_beam_res_usage_check.sh [sm_90]
set -euo pipefail

ARCH="${1:-sm_90}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KDIR="${REPO_ROOT}/mistralrs-quant/kernels/qtip"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

# Mirror mistralrs-quant/build.rs exactly: a different flag set is a different
# register allocation, and therefore a measurement of a kernel we do not ship.
NVCC_ARGS=(
  -std=c++17
  -O3
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
)

# Emit the per-kernel resource lines of a cubin, one per line.
usage_lines() {
  cuobjdump -res-usage "$1" | grep -E 'REG:[0-9]+' || true
}

# Echo "SPILL" when any resource line reports nonzero LOCAL or STACK.
verdict() {
  local lines="$1" local_b stack_b
  while read -r line; do
    [ -z "${line}" ] && continue
    local_b="$(sed -n 's/.*LOCAL:\([0-9]*\).*/\1/p' <<<"${line}")"
    stack_b="$(sed -n 's/.*STACK:\([0-9]*\).*/\1/p' <<<"${line}")"
    [ -z "${local_b}" ] && continue
    if [ "${local_b}" != "0" ] || [ "${stack_b}" != "0" ]; then
      echo "SPILL"
      return
    fi
  done <<<"${lines}"
  echo "CLEAN"
}

echo "== QTIP beam local-memory (spill) gate =="
nvcc --version | tail -1
echo "arch: ${ARCH}"
echo

# ---------------------------------------------------------------------------
# 0. Negative control: prove the parser above can say SPILL at all.
# ---------------------------------------------------------------------------
cat >"${OUT}/negative_control.cu" <<'EOF'
// Dynamically-indexed local array: nvcc cannot promote this to registers, so
// the kernel is guaranteed to report LOCAL > 0. Not part of the build.
__global__ void qtip_spill_negative_control(float* out, int n) {
    float big[256];
    #pragma unroll 1
    for (int i = 0; i < 256; ++i) big[i] = (float)(i * n);
    float s = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < n; ++i) s += big[i & 255];
    out[threadIdx.x] = s;
}
EOF
nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" \
     -o "${OUT}/negative_control.cubin" "${OUT}/negative_control.cu"
nc_lines="$(usage_lines "${OUT}/negative_control.cubin")"
echo "--- negative control (MUST report a spill) ---"
echo "${nc_lines}"
if [ "$(verdict "${nc_lines}")" != "SPILL" ]; then
  echo "FAIL: the negative control did not report LOCAL>0."
  echo "      This gate cannot detect a spill and would pass on ANY kernel."
  exit 1
fi
echo "negative control: SPILL as expected — the gate has teeth."
echo

# ---------------------------------------------------------------------------
# 1. The shipped beam kernels.
#    qtip_beam.cu   : 2 geometries (K4/V2/L16, K8/V4/L12) x 2 codebooks = 4
#    qtip2b_beam.cu : 1 kernel — computed codebook only, so no codebook template
# ---------------------------------------------------------------------------
SOURCES=(qtip_beam.cu qtip2b_beam.cu)
EXPECT=(4 1)

fail=0
for i in "${!SOURCES[@]}"; do
  src="${SOURCES[$i]}"
  want="${EXPECT[$i]}"
  cubin="${OUT}/${src%.cu}.cubin"
  echo "--- ${src} ---"
  nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" -I "${KDIR}" \
       -o "${cubin}" "${KDIR}/${src}"

  # Names first, demangled when c++filt is available, so the log records WHICH
  # geometries were measured — not just that something was.
  if command -v c++filt >/dev/null 2>&1; then
    cuobjdump -res-usage "${cubin}" | grep -E '^\s*Function ' | c++filt || true
  else
    cuobjdump -res-usage "${cubin}" | grep -E '^\s*Function ' || true
  fi

  lines="$(usage_lines "${cubin}")"
  echo "${lines}"

  got="$(grep -c 'REG:' <<<"${lines}" || true)"
  if [ "${got}" -lt "${want}" ]; then
    echo "FAIL: ${src} produced ${got} kernels, expected ${want}."
    echo "      A geometry instantiation went missing; 'no spill' would be a"
    echo "      statement about a kernel that is not in the build."
    fail=1
  fi

  if [ "$(verdict "${lines}")" = "SPILL" ]; then
    echo "FAIL: ${src} spills to local memory."
    echo "      Every loop indexing cand[] must be fully unrolled, and"
    echo "      QB_MIN_BLOCKS_PER_SM must be reachable. Drop it to 3 rather"
    echo "      than shipping a spill."
    fail=1
  fi
  echo
done

if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAILED — see above."
  exit 1
fi
echo "RESULT: LOCAL:0 STACK:0 on every beam kernel (${ARCH}), all geometries present."
