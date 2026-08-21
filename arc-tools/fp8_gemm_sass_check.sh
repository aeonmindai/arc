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
# Per-function SASS census. cuobjdump -sass emits one "Function : <mangled>"
# header per kernel; slice on those and count instruction mnemonics inside
# each slice with awk.
# ---------------------------------------------------------------------------
census() { # $1 = cubin, $2 = function-name substring, $3 = extended regex to count
  cuobjdump -sass "$1" | awk -v want="$2" -v pat="$3" '
    /Function :/ { fname = $NF; infn = index(fname, want) > 0; if (infn) { order[++n] = fname; count[fname] = 0 } }
    infn && $0 ~ pat { count[fname]++ }
    END { for (i = 1; i <= n; i++) printf "%s %d\n", order[i], count[order[i]] }
  '
}

# ---------------------------------------------------------------------------
# 1. The scalar tiled GEMM: LDS width census.
# ---------------------------------------------------------------------------
echo "--- blockwise_fp8_gemm.cu : fp8_matmul_tiled ---"
nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" \
     -o "${OUT}/gemm.cubin" "${KDIR}/blockwise_fp8_gemm.cu"

if command -v c++filt >/dev/null 2>&1; then
  cuobjdump -sass "${OUT}/gemm.cubin" | grep -E 'Function :' | c++filt | grep fp8_matmul_tiled || true
fi

# LDS.128 / LDS.64 are prefixes of the wide forms; plain "LDS " (or LDS.U)
# without a width suffix is the scalar 32-bit load.
lds_any="$(census "${OUT}/gemm.cubin" fp8_matmul_tiled '\<LDS\>|\<LDS\.')"
lds_128="$(census "${OUT}/gemm.cubin" fp8_matmul_tiled '\<LDS\.(U\.)?128\>')"
lds_64="$(census "${OUT}/gemm.cubin" fp8_matmul_tiled '\<LDS\.(U\.)?64\>')"

echo "kernel                                       LDS(all) LDS.128 LDS.64"
fail=0
n_tiled=0
while read -r name total; do
  [ -z "${name}" ] && continue
  n_tiled=$((n_tiled + 1))
  w128="$(awk -v k="${name}" '$1 == k { print $2 }' <<<"${lds_128}")"
  w64="$(awk -v k="${name}" '$1 == k { print $2 }' <<<"${lds_64}")"
  printf '%-44s %8s %7s %6s\n' "${name:0:44}" "${total}" "${w128:-0}" "${w64:-0}"
  if [ "${total}" -eq 0 ]; then
    echo "FAIL: ${name} has ZERO shared-memory loads — the SASS extractor lost"
    echo "      the function body; every census line above is unverified."
    fail=1
  fi
done <<<"${lds_any}"

if [ "${n_tiled}" -ne 4 ]; then
  echo "FAIL: expected 4 fp8_matmul_tiled kernels (2 dtypes x 2 POW2_SCALE"
  echo "      instantiations), found ${n_tiled}. An instantiation was dropped;"
  echo "      the census is about kernels that are not in the build."
  fail=1
fi

total_128="$(awk '{ s += $2 } END { print s + 0 }' <<<"${lds_128}")"
if [ "${total_128}" -eq 0 ]; then
  echo "VERDICT[tiled]: scalar LDS only, no LDS.128 — the shared loads are"
  echo "  32-bit, so the bank-conflict arithmetic (stride mod 32) is about"
  echo "  instructions that really exist, and the +1-pad fix is live."
else
  echo "VERDICT[tiled]: ptxas emitted LDS.128 — the scalar-LDS conflict model"
  echo "  does NOT describe this build; re-derive before quoting it."
fi
echo

# ---------------------------------------------------------------------------
# 2. The WMMA GEMM: tensor-core instructions must exist.
# ---------------------------------------------------------------------------
echo "--- blockwise_fp8_gemm_wmma.cu : HMMA presence ---"
nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" \
     -o "${OUT}/wmma.cubin" "${KDIR}/blockwise_fp8_gemm_wmma.cu"

hmma="$(cuobjdump -sass "${OUT}/wmma.cubin" | grep -cE '\<(HMMA|HGMMA)\>' || true)"
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
echo "RESULT: census complete (${ARCH}); 4 tiled kernels present, WMMA carries tensor-core instructions."
