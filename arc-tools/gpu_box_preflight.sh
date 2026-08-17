#!/usr/bin/env bash
#
# gpu_box_preflight.sh — assert a rented GPU box can actually produce correct
# numbers, BEFORE a 30-minute build and a 10-minute model load die at the end
# (or, worse, succeed and lie).
#
# Every check below exists because it cost a real session real money on
# `arc-mtp-radix` (H200 SXM5, driver 580.173.02 / CUDA 13.0 cap, toolkit 13.1,
# $4.92/hr). Each failure names itself: `FAIL[<CHECK>] <reason>`. Nothing here
# is advisory — a failed check exits non-zero so a run script can chain on it.
#
# Usage:
#   source arc-tools/gpu_box_preflight.sh          # export the fixed env + assert
#   ./arc-tools/gpu_box_preflight.sh               # assert only, exit 0/1
#   ./arc-tools/gpu_box_preflight.sh --quick       # skip the PTX compile probe
#   ./arc-tools/gpu_box_preflight.sh --flags "serve --mtp-depth 3 --prefix-cache-n 0"
#                                                  # also assert those CLI flags exist
#
# It is safe to source: it sets PATH/CUDA_HOME/LD_LIBRARY_PATH to the values
# this image needs, and callers inherit them.
#
# ---------------------------------------------------------------------------
# THE CHECK THAT MATTERS MOST: PTX
# ---------------------------------------------------------------------------
# `candle-kernels/build.rs` calls `cudaforge::KernelBuilder::build_ptx()`, which
# is implemented (cudaforge-0.1.5/src/builder.rs:568-575) as
#     nvcc -gencode=arch=compute_90a,code=sm_90a --ptx <file>.cu
# The `--ptx` flag means the artifact is PTX *text*, JIT-compiled by the driver
# at module load. candle-kernels is the ONLY crate in the tree that does this —
# mistralrs-core, mistralrs-quant, mistralrs-paged-attn, arc-cuda-graph and both
# candle-flash-attn crates all use `build_lib` (SASS). So `CUDA_COMPUTE_CAP` is
# powerless here: it only edits the `compute_90` inside that gencode string, it
# cannot stop PTX from being emitted.
#
# When the toolkit's PTX ISA is newer than the driver's JIT (13.1 toolkit on a
# 13.0-capped driver), the driver refuses it. Under the CUDA *driver* API, which
# is what cudarc uses, that surfaces loudly as
# `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. But under the CUDA *runtime* API the
# launch failure lands in the `cudaGetLastError()` slot, NOT in
# `cudaDeviceSynchronize()` — so a program that only checks the sync slot sees
# "no error" and reads back an unwritten buffer. Measured on this image:
#     without compat:  launch=cudaErrorUnsupportedPtxVersion  val=0
#     with compat:     launch=cudaSuccess                     val=42
# A run missing the compat path therefore does not necessarily crash. It can
# produce plausible, wrong numbers with a clean exit code. That is why this
# check is a hard gate and not a warning.
#
# The fix is NVIDIA forward compatibility (supported on datacenter GPUs): the
# `cuda-compat-*` package installs a newer userspace libcuda alongside the older
# kernel driver. Confirmed on hardware: `apt-get install cuda-compat-13-1` put
# libcuda.so.590.48.01 in /usr/local/cuda-13.1/compat and flipped the PTX arm
# from val=0 to val=42 with no rebuild.
# ---------------------------------------------------------------------------

_arc_pf_sourced=0
# shellcheck disable=SC2128
if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    _arc_pf_sourced=1
fi

ARC_PF_QUICK=0
ARC_PF_FLAGS=""
ARC_PF_MIN_DISK_GB="${ARC_PF_MIN_DISK_GB:-60}"
ARC_PF_BIN="${ARC_PF_BIN:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) ARC_PF_QUICK=1; shift ;;
        --flags) ARC_PF_FLAGS="$2"; shift 2 ;;
        --bin)   ARC_PF_BIN="$2"; shift 2 ;;
        --min-disk-gb) ARC_PF_MIN_DISK_GB="$2"; shift 2 ;;
        *) shift ;;
    esac
done

_ARC_PF_FAILED=0
_pf_ok()   { printf "  ok   [%s] %s\n" "$1" "$2"; }
_pf_fail() { printf "FAIL[%s] %s\n" "$1" "$2" >&2; _ARC_PF_FAILED=1; }
_pf_warn() { printf "  warn [%s] %s\n" "$1" "$2"; }

echo "=== arc gpu box preflight — $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --------------------------------------------------------------- CUDA_ON_PATH
# Trap: nvcc is not on the default PATH on this image; it lives under
# /usr/local/cuda-<ver>/bin. Without it, cudarc's build.rs panics with
# "`nvcc --version` failed" ~40 minutes into a session.
if [ -z "${CUDA_HOME:-}" ]; then
    for _c in /usr/local/cuda /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda-12.8; do
        [ -x "$_c/bin/nvcc" ] && { CUDA_HOME="$(readlink -f "$_c")"; break; }
    done
fi
export CUDA_HOME
case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) : ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac

if command -v nvcc >/dev/null 2>&1; then
    _pf_ok CUDA_ON_PATH "nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1) at $(command -v nvcc)"
else
    _pf_fail CUDA_ON_PATH "nvcc not on PATH and not found under /usr/local/cuda*. cudarc's build.rs will panic with \`nvcc --version\` failed. Set CUDA_HOME."
fi

# ------------------------------------------------------------------ GPU_ALIVE
if command -v nvidia-smi >/dev/null 2>&1; then
    _gpu="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    if [ -n "$_gpu" ]; then
        _pf_ok GPU_ALIVE "$_gpu"
        ARC_COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
        export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-$ARC_COMPUTE_CAP}"
    else
        _pf_fail GPU_ALIVE "nvidia-smi present but returned no device"
    fi
else
    _pf_fail GPU_ALIVE "no nvidia-smi — D14 forbids CPU-only validation, so this box cannot produce a measurement"
fi

# --------------------------------------------------------------- CUDA_COMPAT
# Driver's max CUDA vs the installed toolkit. A newer toolkit is fine ONLY if
# the forward-compat libcuda is on LD_LIBRARY_PATH ahead of the system one.
_drv_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)"
_tk_cuda="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)"
_compat_dir="$(ls -d /usr/local/cuda*/compat 2>/dev/null | tail -1)"
if [ -n "$_compat_dir" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$_compat_dir:"*) : ;;
        *) export LD_LIBRARY_PATH="$_compat_dir:${LD_LIBRARY_PATH:-}" ;;
    esac
fi
if [ -n "$_drv_cuda" ] && [ -n "$_tk_cuda" ]; then
    if [ "$(printf '%s\n%s\n' "$_tk_cuda" "$_drv_cuda" | sort -V | tail -1)" = "$_drv_cuda" ]; then
        _pf_ok CUDA_COMPAT "toolkit $_tk_cuda <= driver max $_drv_cuda; no forward-compat needed"
    elif [ -n "$_compat_dir" ]; then
        _pf_ok CUDA_COMPAT "toolkit $_tk_cuda > driver max $_drv_cuda, using forward-compat $_compat_dir ($(ls "$_compat_dir" | tr '\n' ' '))"
    else
        _pf_fail CUDA_COMPAT "toolkit $_tk_cuda is NEWER than the driver's max CUDA $_drv_cuda and no cuda-compat is installed. PTX JIT will fail — silently under the CUDA runtime API. Run: apt-get install -y cuda-compat-${_tk_cuda%%.*}-$(echo "$_tk_cuda" | cut -d. -f2)"
    fi
fi

# ------------------------------------------------------------------- PTX_JIT
# The differential probe. SASS arm must pass; PTX arm must pass too. If SASS
# passes and PTX fails, the box will run Arc's own kernels but corrupt every
# candle-kernels op — which is most elementwise/reduction work in the model.
if [ "$ARC_PF_QUICK" = "1" ]; then
    _pf_warn PTX_JIT "skipped (--quick). Do not report a number from a run that skipped this."
elif ! command -v nvcc >/dev/null 2>&1; then
    _pf_fail PTX_JIT "cannot probe without nvcc"
else
    _pf_tmp="$(mktemp -d)"
    cat > "$_pf_tmp/pt.cu" <<'CUEOF'
#include <cstdio>
__global__ void k(int *o) { *o = 42; }
int main() {
    int *d = 0;
    if (cudaMalloc(&d, 4) != cudaSuccess) { printf("malloc_fail\n"); return 2; }
    cudaMemset(d, 0, 4);
    k<<<1, 1>>>(d);
    /* The launch error lands HERE, not in cudaDeviceSynchronize(): a kernel
       that never launched has nothing to synchronise on. Checking only the
       sync slot is exactly how a PTX-JIT failure reads back as "no error"
       plus a wrong value. */
    cudaError_t le = cudaGetLastError();
    cudaError_t se = cudaDeviceSynchronize();
    int h = -1;
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
    printf("launch=%s sync=%s val=%d\n",
           cudaGetErrorName(le), cudaGetErrorName(se), h);
    return (h == 42) ? 0 : 1;
}
CUEOF
    _cc="${CUDA_COMPUTE_CAP:-90}"
    _arch="compute_${_cc}a"
    nvcc "-gencode=arch=${_arch},code=sm_${_cc}a" "$_pf_tmp/pt.cu" -o "$_pf_tmp/sass" >"$_pf_tmp/nvcc.log" 2>&1 \
        || nvcc "-gencode=arch=compute_${_cc},code=sm_${_cc}" "$_pf_tmp/pt.cu" -o "$_pf_tmp/sass" >"$_pf_tmp/nvcc.log" 2>&1
    nvcc "-gencode=arch=${_arch},code=${_arch}" "$_pf_tmp/pt.cu" -o "$_pf_tmp/ptx" >>"$_pf_tmp/nvcc.log" 2>&1 \
        || nvcc "-gencode=arch=compute_${_cc},code=compute_${_cc}" "$_pf_tmp/pt.cu" -o "$_pf_tmp/ptx" >>"$_pf_tmp/nvcc.log" 2>&1

    if [ -x "$_pf_tmp/sass" ]; then
        _sass_out="$("$_pf_tmp/sass" 2>&1)"
        case "$_sass_out" in
            *val=42*) _pf_ok PTX_JIT_SASS "cubin path correct — $_sass_out" ;;
            *) _pf_fail PTX_JIT_SASS "SASS/cubin kernel returned the wrong value: $_sass_out. This box cannot run ANY CUDA correctly." ;;
        esac
    else
        _pf_fail PTX_JIT_SASS "probe failed to compile: $(tail -3 "$_pf_tmp/nvcc.log")"
    fi

    if [ -x "$_pf_tmp/ptx" ]; then
        _ptx_out="$("$_pf_tmp/ptx" 2>&1)"
        case "$_ptx_out" in
            *val=42*) _pf_ok PTX_JIT "driver JIT correct — $_ptx_out" ;;
            *) _pf_fail PTX_JIT "PTX-JIT kernel returned the WRONG VALUE: $_ptx_out. candle-kernels ships PTX only (candle-kernels/build.rs -> cudaforge build_ptx), so every op it owns is corrupt. Note the clean exit code — this failure does NOT crash the server. Install cuda-compat and put it first on LD_LIBRARY_PATH." ;;
        esac
    else
        _pf_fail PTX_JIT "PTX probe failed to compile: $(tail -3 "$_pf_tmp/nvcc.log")"
    fi
    rm -rf "$_pf_tmp"
fi

# --------------------------------------------------------------------- NO_CUDNN
# Measured -62% decode on V4 (14.58 -> 5.45 tok/s, session 4). Never build it.
if [ -n "${ARC_BUILD_FEATURES:-}" ]; then
    case "$ARC_BUILD_FEATURES" in
        *cudnn*) _pf_fail NO_CUDNN "ARC_BUILD_FEATURES contains 'cudnn' — measured -62% decode on V4. Build --features \"cuda flash-attn\"." ;;
        *) _pf_ok NO_CUDNN "features '$ARC_BUILD_FEATURES' clean" ;;
    esac
fi

# ------------------------------------------------------------------ PY_BUILD
# pip 22.0.2 + no numpy breaks flash-attn metadata generation with
# "canonicalize_version() got an unexpected keyword argument
# 'strip_trailing_zero'". Upgrade pip/setuptools/wheel and install numpy FIRST.
if command -v python3 >/dev/null 2>&1; then
    _pipv="$(python3 -m pip --version 2>/dev/null | awk '{print $2}')"
    if [ -z "$_pipv" ]; then
        _pf_fail PY_BUILD "python3 has no pip; flash-attn and every probe script need it"
    elif [ "$(printf '%s\n23.0\n' "$_pipv" | sort -V | head -1)" = "$_pipv" ] && [ "$_pipv" != "23.0" ]; then
        _pf_fail PY_BUILD "pip $_pipv is too old — flash-attn metadata generation dies with canonicalize_version(strip_trailing_zero). Run: python3 -m pip install -U pip setuptools wheel && python3 -m pip install numpy"
    elif ! python3 -c "import numpy" >/dev/null 2>&1; then
        _pf_fail PY_BUILD "numpy missing — install it BEFORE flash-attn or the build backend fails"
    else
        _pf_ok PY_BUILD "pip $_pipv, numpy $(python3 -c 'import numpy;print(numpy.__version__)')"
    fi
fi

# ------------------------------------------------------------------- SETSID
# A plain `nohup ... &` through `runcrate ssh` does not survive the session.
# Use: setsid nohup <cmd> < /dev/null > log 2>&1 &
if command -v setsid >/dev/null 2>&1; then
    _pf_ok SETSID "present — background with: setsid nohup CMD < /dev/null > LOG 2>&1 &"
else
    _pf_fail SETSID "no setsid; a plain 'nohup &' through runcrate ssh will be killed when the ssh session ends"
fi

# --------------------------------------------------------------------- DISK
_free_gb="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "$_free_gb" ]; then
    if [ "$_free_gb" -lt "$ARC_PF_MIN_DISK_GB" ]; then
        _pf_fail DISK "${_free_gb}G free < ${ARC_PF_MIN_DISK_GB}G required"
    else
        _pf_ok DISK "${_free_gb}G free"
    fi
fi

# ---------------------------------------------------------------- CLI_FLAGS
# Assert every flag a run intends to pass actually exists, BEFORE the model
# load. `--max-seq-len` and `--max-batch-size` appear in GPU_SESSION_RUNBOOK_8
# §S3c and exist nowhere in mistralrs-cli; clap rejects the invocation and the
# server never starts.
if [ -n "$ARC_PF_FLAGS" ]; then
    _bin="${ARC_PF_BIN:-$(command -v mistralrs 2>/dev/null)}"
    if [ -z "$_bin" ] || [ ! -x "$_bin" ]; then
        _pf_warn CLI_FLAGS "no mistralrs binary given (--bin) — cannot verify flags $ARC_PF_FLAGS"
    else
        _sub="$(printf '%s' "$ARC_PF_FLAGS" | awk '{print $1}')"
        _help="$("$_bin" "$_sub" --help 2>&1)"
        _bad=""
        for _f in $ARC_PF_FLAGS; do
            case "$_f" in
                --*) printf '%s' "$_help" | grep -q -- "$_f" || _bad="$_bad $_f" ;;
            esac
        done
        if [ -n "$_bad" ]; then
            _pf_fail CLI_FLAGS "flags not accepted by '$_bin $_sub':$_bad — clap will refuse the invocation and the server will never start"
        else
            _pf_ok CLI_FLAGS "all flags exist on '$_sub'"
        fi
    fi
fi

# -------------------------------------------------------------------- verdict
echo "--- env exported by this preflight ---"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

if [ "$_ARC_PF_FAILED" = "1" ]; then
    echo "PREFLIGHT=FAIL — do not start a measurement; the numbers would not be trustworthy."
    [ "$_arc_pf_sourced" = "1" ] || exit 1
else
    echo "PREFLIGHT=PASS"
    [ "$_arc_pf_sourced" = "1" ] || exit 0
fi
