// Parent system: ArcKernels (rung owner: ArcQuant / QTIP)
//
// Fused QTIP decode + gemv at **K=8 / V=4 / L=12**, with the whole
// reproduction table staged in shared memory as bf16.
//
// WHAT IS DIFFERENT FROM `qtip_gemv.cu` (K=4/V=2/L=16)
// ----------------------------------------------------
// Same bit rate. `bpw = K/V`, so 8/4 is 2 bits per weight exactly as 4/2 is.
// Nothing here compresses harder; everything here decodes cheaper:
//
//   * The table is 2^12 x 4 bf16 = 32,768 B, not 2^16 x 2 f32 = 524,288 B.
//     32 KiB is under the 48 KiB static __shared__ limit, so it is staged once
//     per block with NO `cudaFuncSetAttribute` opt-in, and every subsequent
//     lookup is an LDS instead of a dependent, data-scattered trip to L2.
//     The sibling kernel's own header calls its 512 KiB table "small enough to
//     stay hot"; the measured reality quoted there is 388 GB/s, ~8% of H200
//     HBM, and that gather is the decode limiter.
//   * A symbol IS a byte. K=8 deletes the nibble unpack — the `& 0x0F`,
//     the `>> 4`, and the odd/even branch wrapped around them.
//   * One table read yields FOUR weights, not two.
//   * Warmup is ceil(L/K) = 2 prior symbols, not 4.
//
// WHY V=4 AND NOT V=8
// -------------------
// For `mma.m16n8k16.bf16` each thread's B fragment is 4 contiguous-k bf16
// values, so a V=4 bf16 entry is exactly 8 bytes = one LDS.64 = a complete mma
// B-operand pair: no conversion and no repacking between the table read and a
// tensor-core operand. V=8 needs L>=16 to hold the rate at 2 bpw, which puts
// the table back at 1 MiB and out of shared memory.
//
// MEASUREMENT STATUS -- READ BEFORE QUOTING A NUMBER
// -------------------------------------------------
// The 5.375 inst/weight (4.375 with the row-scale hoist) that motivates this
// geometry was produced by a standalone COMPILED PROBE, not by this file.
// **This kernel has not been compiled with nvcc and has not been run on a
// GPU.** Its instruction count is UNMEASURED. Establish it with
// `nvcc -cubin -arch=sm_90` plus unroll differencing on the inner loop before
// it is quoted anywhere. Hand-counting C++ undercounts SASS by ~2.05x on this
// kernel family, so a count that was not compiled is not a count.
//
// PARITY
// ------
// `mistralrs-quant/src/qtip/k8v4l12.rs::gemv_row(.., RowScaleHoist::Off)` is
// the reference, and the correspondence is operation for operation:
//
//     w   = __fmul_rn(cb, scale)   <->  let w = cb * scale;
//     acc = fmaf(w, x, acc)        <->  acc = w.mul_add(x, acc);
//
// `__fmul_rn` rather than `*` because build.rs compiles this directory with
// `--use_fast_math`, which implies `-fmad=true`; a bare `cb * scale` feeding
// the FMA's multiplicand is not contractible today, but writing the intent is
// what `qtip_exact_fp.cuh` exists for and costs nothing. bf16 -> f32 widening
// is exact on both sides, and `fmaf` and Rust's `f32::mul_add` are both the
// single-rounding fused op.
//
// The row-scale hoist (`ROW_SCALE_HOIST`) is the ONE thing that breaks this.
// It reassociates the sum -- `sum (cb*s)*x` becomes `s * sum cb*x` -- so it is
// a separate template parameter and the parity gate runs with it off.
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr uint32_t QK8_K           = 8;
constexpr uint32_t QK8_L           = 12;
constexpr uint32_t QK8_V           = 4;
constexpr uint32_t QK8_STATE_MASK  = (1u << QK8_L) - 1u;     // 0xFFF
constexpr uint32_t QK8_LUT_STATES  = 1u << QK8_L;            // 4096
constexpr uint32_t QK8_LUT_ENTRIES = QK8_LUT_STATES * QK8_V; // 16384

// Prior symbols consumed before a thread decodes its own slice: ceil(L/K).
// One more than strictly required -- the state that decodes symbol t is
// ((s_{t-1} & 0xF) << 8) | packed[t] and s_{t-1} & 0xF == packed[t-1] & 0xF --
// but it matches `qtip_gemv.cu`'s definition at K=4/L=16, and the Rust side
// pins both facts (`one_prior_symbol_is_the_true_minimum_warmup`).
constexpr uint32_t QK8_WARMUP_SYMS = (QK8_L + QK8_K - 1u) / QK8_K;  // 2

static_assert(QK8_LUT_ENTRIES * sizeof(__nv_bfloat16) == 32768,
              "the K=8/V=4/L=12 table must be exactly 32 KiB; that is the property that lets "
              "it live in static shared memory without a cudaFuncSetAttribute opt-in");
static_assert(QK8_LUT_ENTRIES * sizeof(__nv_bfloat16) <= 48 * 1024,
              "static __shared__ is capped at 48 KiB per block");
static_assert(QK8_K % QK8_V == 0 && (QK8_K / QK8_V) == 2,
              "this rung must stay at 2 bits per weight (bpw = K/V), or it is no longer "
              "comparable with the shipped K=4/V=2 rung");
static_assert(QK8_WARMUP_SYMS * QK8_K >= QK8_L,
              "warmup must shift at least L bits through the state register");

template <typename T>
__device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
    return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);

template <>
__device__ __forceinline__ float to_f32<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// Warp-level XOR-shuffle sum reduction.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// Stage the whole 32 KiB table into shared memory, cooperatively.
//
// 16-byte loads when the source pointer allows it (a freshly materialized
// [4096, 4] tensor is cudaMalloc-aligned, so this is the normal case), with a
// scalar fallback so a view with a nonzero element offset degrades in speed
// rather than faulting. Caller issues the __syncthreads().
__device__ __forceinline__ void stage_lut(
    __nv_bfloat16* __restrict__ lut_s, const __nv_bfloat16* __restrict__ lut_g, int threads
) {
    const int tid = threadIdx.x;
    if ((reinterpret_cast<uintptr_t>(lut_g) & 0xFu) == 0u) {
        constexpr int N_VEC = (int)(QK8_LUT_ENTRIES * sizeof(__nv_bfloat16) / sizeof(uint4)); // 2048
        const uint4* src = reinterpret_cast<const uint4*>(lut_g);
        uint4*       dst = reinterpret_cast<uint4*>(lut_s);
        for (int i = tid; i < N_VEC; i += threads) {
            dst[i] = src[i];
        }
    } else {
        for (int i = tid; i < (int)QK8_LUT_ENTRIES; i += threads) {
            lut_s[i] = lut_g[i];
        }
    }
}

// Fused decode + gemv. `y[row] = sum_k W[row,k] * x[k]`, with W reconstructed
// on the fly from `packed`, `row_scales` and the staged table. `x` must
// already be in the QTIP-rotated frame (the caller applies
// `launch_qtip_rotate_x_*` first, exactly as for the K=4 rung).
//
// Grid:  (n_blocks, 1, 1)  -- each block walks rows blockIdx.x, +gridDim.x, ...
// Block: (THREADS, 1, 1)
//
// The grid-stride over rows is what makes the staging pay. Staging costs
// 32 KiB of L2->SMEM per BLOCK, not per row, so with a grid sized to fill the
// device the cost is bounded by the grid rather than by n_rows. One block per
// row would stage 32 KiB to do (in_features/4) bytes of packed work -- 32x
// overhead at in_features=4096.
template <typename T, int THREADS, bool ROW_SCALE_HOIST>
__global__ void __launch_bounds__(THREADS)
qtip_fused_gemv_k8_v4_l12_kernel(
    const uint8_t*       __restrict__ packed,      // [n_rows, packed_per_row]
    const float*         __restrict__ row_scales,  // [n_rows]
    const __nv_bfloat16* __restrict__ lut,         // [2^L * V] bf16, 32 KiB
    const T*             __restrict__ x,           // [k_in], k_in == num_symbols*V
    T*                   __restrict__ y,           // [n_rows]
    int n_rows,
    int packed_per_row,
    int num_symbols
) {
    // 32,768 B. Plus warp_sums below; total static shared is
    // 32768 + (THREADS/32)*4 B, still far under the 48 KiB cap.
    __shared__ __align__(16) __nv_bfloat16 lut_s[QK8_LUT_ENTRIES];
    constexpr int N_WARPS = THREADS / 32;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");
    __shared__ float warp_sums[N_WARPS];

    stage_lut(lut_s, lut, THREADS);
    __syncthreads();

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // Each thread takes a contiguous slice of the row's symbols.
    const int sym_per_thread = (num_symbols + THREADS - 1) / THREADS;
    const int sym_start_raw  = tid * sym_per_thread;
    const int sym_end        = min(num_symbols, sym_start_raw + sym_per_thread);

    for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const uint8_t* row_packed = packed + (size_t)row * packed_per_row;
        const float    scale      = __ldg(row_scales + row);

        float acc = 0.0f;

        if (sym_start_raw < num_symbols) {
            // ---- Warmup ----
            // Threads i>0 replay QK8_WARMUP_SYMS prior symbols to seed the
            // state; their decoded weights belong to the previous thread's
            // slice and are NOT accumulated. Thread 0 starts from the true
            // initial state 0. A thread starting inside the first
            // QK8_WARMUP_SYMS symbols walks from 0, which is exact rather
            // than approximate for the same reason.
            uint32_t state = 0u;
            if (sym_start_raw > 0) {
                const int warm_start = max(0, sym_start_raw - (int)QK8_WARMUP_SYMS);
                for (int t = warm_start; t < sym_start_raw; ++t) {
                    const uint32_t sym = (uint32_t)__ldg(row_packed + t);
                    state = ((state << QK8_K) | sym) & QK8_STATE_MASK;
                }
            }

            // ---- Decode ----
            // One byte -> one symbol -> one 8-byte table entry -> four
            // weights. No nibble unpack and no odd/even branch: that is what
            // K=8 buys over the K=4 rung.
            for (int sym_idx = sym_start_raw; sym_idx < sym_end; ++sym_idx) {
                const uint32_t sym = (uint32_t)__ldg(row_packed + sym_idx);
                state = ((state << QK8_K) | sym) & QK8_STATE_MASK;

                // `lut_s` is 16-byte aligned and the entry is 4 bf16, so this
                // address is 8-byte aligned for every state -- the layout that
                // makes an LDS.64 (and an mma B-operand pair) possible. Whether
                // the compiler actually fuses the four 2-byte reads into one
                // 8-byte load is a codegen outcome to be confirmed with
                // `nvcc -cubin`, NOT something this comment asserts.
                const __nv_bfloat16* c = &lut_s[(size_t)state * QK8_V];
                const int x_off = sym_idx * (int)QK8_V;

                #pragma unroll
                for (int v = 0; v < (int)QK8_V; ++v) {
                    const float cb = __bfloat162float(c[v]);
                    const float xv = to_f32<T>(__ldg(x + x_off + v));
                    if constexpr (ROW_SCALE_HOIST) {
                        acc = fmaf(cb, xv, acc);
                    } else {
                        acc = fmaf(__fmul_rn(cb, scale), xv, acc);
                    }
                }
            }
        }

        // ---- Block reduction ----
        acc = warp_reduce_sum(acc);
        // The grid-stride loop reuses warp_sums every iteration, so a warp
        // that races ahead must not overwrite a value warp 0 has not read yet.
        __syncthreads();
        if (lane == 0) warp_sums[warp_id] = acc;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) {
                y[row] = from_f32<T>(ROW_SCALE_HOIST ? (v * scale) : v);
            }
        }
    }
}

// Blocks resident at once. Staging is 32 KiB of L2->SMEM per block, so this
// caps total staging traffic at 64 MiB regardless of how many rows the layer
// has. Sized to oversubscribe any current datacentre part (H100 has 132 SMs)
// without making the cap the thing that limits parallelism.
constexpr int QK8_MAX_BLOCKS = 2048;

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// `row_scale_hoist` is 0 for the bit-exact policy (per-weight scale, the one
// the parity gate pins) and nonzero for the hoisted policy. It is an int and
// not a bool so the ABI is unambiguous across the FFI boundary;
// `RowScaleHoist::as_abi` on the Rust side is the only thing that produces it.
#define QTIP_K8V4L12_GEMV_LAUNCHER(NAME, T)                                        \
    void NAME(const uint8_t*       d_packed,                                       \
              const float*         d_row_scales,                                   \
              const __nv_bfloat16* d_lut,                                          \
              const T*             d_x_rotated,                                    \
              T*                   d_y,                                            \
              int n_rows,                                                          \
              int packed_per_row,                                                  \
              int num_symbols,                                                     \
              int row_scale_hoist,                                                 \
              cudaStream_t stream) {                                               \
        constexpr int THREADS = 128;                                               \
        if (n_rows <= 0 || num_symbols <= 0) return;                               \
        const int blocks = n_rows < QK8_MAX_BLOCKS ? n_rows : QK8_MAX_BLOCKS;      \
        if (row_scale_hoist != 0) {                                                \
            qtip_fused_gemv_k8_v4_l12_kernel<T, THREADS, true>                     \
                <<<blocks, THREADS, 0, stream>>>(                                  \
                    d_packed, d_row_scales, d_lut, d_x_rotated, d_y,               \
                    n_rows, packed_per_row, num_symbols);                          \
        } else {                                                                   \
            qtip_fused_gemv_k8_v4_l12_kernel<T, THREADS, false>                    \
                <<<blocks, THREADS, 0, stream>>>(                                  \
                    d_packed, d_row_scales, d_lut, d_x_rotated, d_y,               \
                    n_rows, packed_per_row, num_symbols);                          \
        }                                                                          \
    }

QTIP_K8V4L12_GEMV_LAUNCHER(launch_qtip_fused_gemv_k8_v4_l12_bf16, __nv_bfloat16)
QTIP_K8V4L12_GEMV_LAUNCHER(launch_qtip_fused_gemv_k8_v4_l12_f16,  __half)
QTIP_K8V4L12_GEMV_LAUNCHER(launch_qtip_fused_gemv_k8_v4_l12_f32,  float)

#undef QTIP_K8V4L12_GEMV_LAUNCHER

} // extern "C"
