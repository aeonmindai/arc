#!/usr/bin/env bash
# Parent system: ArcKernels + ArcGate.
#
# FREE SASS PROBE for the blockwise-FP8 dense-GEMM lane. NO GPU REQUIRED:
# nvcc + cuobjdump only, so it runs on the GPU-less nvcc CI lane
# (.github/workflows/cuda_compile_check.yaml) and on any dev box with the
# toolkit.
#
# WHAT IT SETTLES
# ---------------
# 1. `fp8_matmul_tiled` (blockwise_fp8_gemm.cu) — HOW ptxas actually loads the
#    shared tiles in the inner product `acc += s_input[ty][k] * s_weight[tx][k]`.
#    The source-derived cost model for this kernel (shared traffic 4.13x over
#    supply x a 4-way bank conflict from the old +4 tile padding) is a claim
#    about SCALAR 32-bit LDS. If ptxas had already vectorized those loads to
#    LDS.128, the conflict arithmetic would be about instructions that do not
#    exist and the model's 7% agreement would be a coincidence. This probe
#    prints the per-kernel LDS width census so that question is settled by the
#    compiled artifact, not by reading C++.
#      NOTE the padding interaction: the +1 pad (stride 33 floats, PR #215)
#      that kills the 4-way bank conflict ALSO forecloses LDS.128 on tile rows
#      (rows are only 4-byte aligned at stride 33). Scalar LDS here is
#      therefore the EXPECTED outcome on current source, not a defect — the
#      probe exists so a future retune that trades padding for vector width
#      does it with the census in front of them.
# 2. `blockwise_fp8_gemm_wmma.cu` — whether the "tensor-core" GEMM actually
#    contains tensor-core instructions (HMMA/HGMMA). That kernel has NEVER
#    executed on any GPU; if ptxas lowered it without a single matrix-multiply
#    instruction, every derivation in its header is void. HARD FAIL if absent.
#
# TEETH (a parser that matches nothing must not pass)
# ---------------------------------------------------
# * Kernel-count assertion: the tiled TU must yield exactly 4
#   `fp8_matmul_tiled` kernels (2 dtypes x 2 POW2_SCALE instantiations —
#   see the launcher in blockwise_fp8_gemm.cu). A count drift means an
#   instantiation was dropped and the census is about a kernel not in the
#   build.
# * Every tiled kernel must show at least one LDS — a tiled shared-memory GEMM
#   with zero shared loads means the SASS extractor lost the function body.
#
# Usage:  arc-tools/fp8_gemm_sass_check.sh [sm_90]
set -euo pipefail

ARCH="${1:-sm_90}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KDIR="${REPO_ROOT}/mistralrs-quant/kernels/blockwise_fp8"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

# Mirror mistralrs-quant/build.rs exactly: a different flag set is a different
# instruction selection, and therefore a probe of a kernel we do not ship.
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

echo "== blockwise-FP8 GEMM SASS probe =="
nvcc --version | tail -1
echo "arch: ${ARCH}"
echo

# ---------------------------------------------------------------------------
# Per-function SASS extraction.
#
# PORTABILITY LESSON (this script's first CI run went red on itself): GNU awk
# does not honor \< \> word-boundary escapes in ERE — it warns "escape
# sequence treated as plain character" and matches nothing — so the original
# awk body-splitter returned zero LDS for every function and the fail-on-zero
# guard below correctly refused to trust it. This revision does no regex body
# splitting at all: cuobjdump lists the mangled kernel names, and
# `cuobjdump -sass -fun <mangled>` extracts exactly one function's SASS. All
# counting is grep -cE with POSIX classes and bracket expressions only —
# no \< \> anywhere.
# ---------------------------------------------------------------------------
fn_names() { # $1 = cubin, $2 = fixed-string name filter
  cuobjdump -sass "$1" | grep 'Function : ' | awk '{print $NF}' | grep -F "$2" || true
}
count() { # $1 = file, $2 = POSIX ERE
  grep -cE "$2" "$1" || true
}
# SASS instruction lines carry a /*<hex addr>*/ comment; body-vs-empty is
# judged on those, so "extractor returned nothing" and "zero LDS" cannot be
# confused with each other.
RE_INSN='/[*][0-9a-fA-F]+[*]/'
# LDSM ('M' after LDS) must not count; the char after LDS is '.' or space.
RE_LDS_ALL='[[:space:]]LDS[[:space:].]'
RE_LDS_128='[[:space:]]LDS[.](U[.])?128[[:space:]]'
RE_LDS_64='[[:space:]]LDS[.](U[.])?64[[:space:]]'
RE_MMA='[[:space:]](HMMA|HGMMA)[[:space:].]'

# ---------------------------------------------------------------------------
# 1. The scalar tiled GEMM: LDS width census.
# ---------------------------------------------------------------------------
echo "--- blockwise_fp8_gemm.cu : fp8_matmul_tiled ---"
nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" \
     -o "${OUT}/gemm.cubin" "${KDIR}/blockwise_fp8_gemm.cu"

if command -v c++filt >/dev/null 2>&1; then
  cuobjdump -sass "${OUT}/gemm.cubin" | grep 'Function : ' | c++filt | grep fp8_matmul_tiled || true
fi

fail=0
extract_ok=1
n_tiled=0
sum128=0
names="$(fn_names "${OUT}/gemm.cubin" fp8_matmul_tiled)"
if [ -z "${names}" ]; then
  echo "FAIL[extractor]: no fp8_matmul_tiled function names in the cubin's"
  echo "  SASS listing — the name pass found nothing, so there is no census."
  fail=1
  extract_ok=0
fi

echo "kernel                                            insns LDS(all) LDS.128 LDS.64 scalar"
for name in ${names}; do
  n_tiled=$((n_tiled + 1))
  sfile="${OUT}/tiled_${n_tiled}.sass"
  cuobjdump -sass -fun "${name}" "${OUT}/gemm.cubin" > "${sfile}" 2>/dev/null || true
  insns="$(count "${sfile}" "${RE_INSN}")"
  lds_all="$(count "${sfile}" "${RE_LDS_ALL}")"
  w128="$(count "${sfile}" "${RE_LDS_128}")"
  w64="$(count "${sfile}" "${RE_LDS_64}")"
  scalar=$((lds_all - w128 - w64))
  sum128=$((sum128 + w128))
  printf '%-48s %6s %8s %7s %6s %6s\n' "${name:0:48}" "${insns}" "${lds_all}" "${w128}" "${w64}" "${scalar}"
  if [ "${insns}" -eq 0 ]; then
    echo "FAIL[extractor]: ${name}: 'cuobjdump -sass -fun' returned no"
    echo "  instruction lines. This is an EXTRACTION failure, not a zero-LDS"
    echo "  kernel — this function's census row is unverified."
    fail=1
    extract_ok=0
  elif [ "${lds_all}" -eq 0 ]; then
    echo "FAIL[zero-LDS]: ${name}: body extracted (${insns} instructions) but"
    echo "  it contains no shared-memory loads at all. A shared-tile GEMM"
    echo "  without LDS means the wrong function was extracted or the kernel"
    echo "  no longer stages tiles — either way the conflict model has no"
    echo "  subject here."
    fail=1
  fi
done

if [ "${n_tiled}" -ne 4 ]; then
  echo "FAIL: expected 4 fp8_matmul_tiled kernels (2 dtypes x 2 POW2_SCALE"
  echo "      instantiations), found ${n_tiled}. An instantiation was dropped;"
  echo "      the census is about kernels that are not in the build."
  fail=1
fi

# TU-level cross-check: computed on the WHOLE cubin dump, independent of the
# per-function extractor, so it survives an extractor failure and gives the
# reader a signal either way. Informational — the per-function census is the
# only thing the VERDICT may rest on (the TU includes the GEMV/MoE kernels).
cuobjdump -sass "${OUT}/gemm.cubin" > "${OUT}/gemm.sass" 2>/dev/null || true
echo "TU-level cross-check (whole cubin, informational):" \
     "LDS(all)=$(count "${OUT}/gemm.sass" "${RE_LDS_ALL}")" \
     "LDS.128=$(count "${OUT}/gemm.sass" "${RE_LDS_128}")" \
     "LDS.64=$(count "${OUT}/gemm.sass" "${RE_LDS_64}")"

# The VERDICT prints ONLY on a complete, non-empty per-function census.
# Printing it off TU-level counts (or off an empty extraction) would be a
# lying instrument: confident prose about numbers that were never measured.
if [ "${extract_ok}" -eq 1 ] && [ "${n_tiled}" -eq 4 ] && [ "${fail}" -eq 0 ]; then
  if [ "${sum128}" -eq 0 ]; then
    echo "VERDICT[tiled]: scalar LDS only, no LDS.128 — the shared loads are"
    echo "  32-bit, so the bank-conflict arithmetic (stride mod 32) is about"
    echo "  instructions that really exist, and the +1-pad fix is live."
  else
    echo "VERDICT[tiled]: ptxas emitted LDS.128 — the scalar-LDS conflict model"
    echo "  does NOT describe this build; re-derive before quoting it."
  fi
else
  echo "VERDICT[tiled]: NOT ESTABLISHED — the per-function census is empty or"
  echo "  partial (see FAILs above). Do not quote any LDS conclusion from this"
  echo "  run; the TU-level line above is a cross-check, not a verdict."
fi
echo

# ---------------------------------------------------------------------------
# 2. The WMMA GEMM: tensor-core instructions must exist.
# ---------------------------------------------------------------------------
echo "--- blockwise_fp8_gemm_wmma.cu : HMMA presence ---"
nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" \
     -o "${OUT}/wmma.cubin" "${KDIR}/blockwise_fp8_gemm_wmma.cu"

cuobjdump -sass "${OUT}/wmma.cubin" > "${OUT}/wmma.sass" 2>/dev/null || true
hmma="$(count "${OUT}/wmma.sass" "${RE_MMA}")"
echo "HMMA/HGMMA instructions in TU: ${hmma}"
if [ "${hmma}" -eq 0 ]; then
  echo "FAIL: the tensor-core GEMM contains no HMMA/HGMMA. It has never run on"
  echo "      hardware; if it also has no tensor-core instructions, every"
  echo "      derivation in its header is void and the three-arm sweep would"
  echo "      be comparing cuBLASLt against two scalar kernels."
  fail=1
fi
echo

if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAILED — see above."
  exit 1
fi
echo "RESULT: census complete (${ARCH}); 4 tiled kernels present (per-function"
echo "  extraction verified non-empty), WMMA TU carries ${hmma} HMMA/HGMMA."
