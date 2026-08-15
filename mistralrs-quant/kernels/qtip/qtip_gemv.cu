// QTIP CUDA fused decode + gemv kernel (single-token decode path).
//
// Replaces the dequantize+matmul forward path. The dequantize pass materializes
// the full BF16 weight matrix in HBM (~13 GB for V4 Flash's 13B active params)
// then the matmul reads it back, so the weight traffic across HBM is roughly
// 3× the actual packed-bit traffic: read packed → write BF16 → read BF16.
//
// This kernel reads the packed QTIP bytes once, decodes them in registers, and
// accumulates the matvec `y = W @ x_rotated` on the fly. Each packed byte
// holds two K=4 symbols → 4 weights (V=2 reproduction values per symbol).
// The LUT (2^16 × V floats = 512 KiB) lives in L2 cache; the matvec hits the
// LUT roughly num_symbols times per row, which is small enough to stay hot.
//
// Performance target on H100: ≥85% HBM utilization on the packed-byte read
// (2.85 TB/s of the 3.35 TB/s peak), which puts the V4 Flash decode at the
// ~1 K tok/s ceiling implied by 13B active params × 0.25 bytes/weight ≈ 3.25 GB
// per token.
//
// Correctness invariants (must match `qtip_dequantize.cu` and `mod.rs`):
//   * State recurrence:   state_t = ((state_{t-1} << K) | sym_t) & ((1<<L)-1)
//   * Initial state:      state_{-1} = 0
//   * Pack layout:        byte b stores (sym 2b) in low nibble, (sym 2b+1) in high nibble
//   * Per-row scale:      multiplied into the decoded weights before fmaf
//   * Warmup property:    for t ≥ ceil(L/K)=4 the state is determined by the
//     last L/K=4 symbols alone (the K=4 most-recent symbols shift the L=16
//     bits out of the state). We exploit this to start multiple threads at
//     arbitrary symbol offsets after consuming 4 prior symbols' worth of
//     warmup symbols. Symbols 0..3 are still seeded from state=0.
//
// Activation pre-rotation (D·H·D incoherence) must be applied to `x` BEFORE
// calling this kernel — the existing `launch_qtip_rotate_x_*` from
// `qtip_dequantize.cu` does this. The kernel itself only sees `x_rotated`.
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The LUT gather this kernel's header calls "small enough to stay hot" is the
// measured decode limiter (388 GB/s ~ 8% of HBM on H200; the stall is the
// dependent, scattered load, not the bytes). `qtip_cb_value_ldg<true>` removes
// it entirely — see qtip_codebook.cuh.
#include "qtip_codebook.cuh"

namespace {

constexpr uint32_t QTIP_K          = 4;
constexpr uint32_t QTIP_L          = 16;
constexpr uint32_t QTIP_V          = 2;
constexpr uint32_t QTIP_STATE_MASK = (1u << QTIP_L) - 1u;
// Number of prior symbols required to fully determine the trellis state when
// starting an arbitrary thread mid-row. ceil(L/K) = 4.
constexpr uint32_t QTIP_WARMUP_SYMS = QTIP_L / QTIP_K;

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

// Fused decode + gemv kernel. Computes `y[row] = sum_k W_packed[row,k] * x[k]`
// where W is reconstructed on the fly from `packed`, `row_scales`, and the
// shared `lut`. Assumes `x` is already in the QTIP-rotated frame.
//
// Grid:  (n_rows, 1, 1)
// Block: (THREADS, 1, 1)  — one CUDA block per output row.
//
// Each thread walks a contiguous slice of symbols, warms up its state from
// the QTIP_WARMUP_SYMS prior symbols (read directly from `packed`), then
// decodes its own slice and accumulates the dot product into a register
// scalar. Threads finally do a block-wide reduction and thread 0 writes
// the output.
//
// Memory traffic per row:
//   * packed bytes:        packed_per_row bytes, read once globally per row.
//   * x reads:             num_weights × sizeof(T), shared by all threads in
//                          the block (each x slice goes through L2; with a
//                          BLOCK_PER_SM=4 occupancy hint the activations
//                          mostly stay in L2 across blocks).
//   * LUT reads:           random into 512 KiB; H100 L2 = 60 MB so the LUT
//                          stays cache-hot across the whole grid.
//   * row_scales:          1 float per row.
//   * output write:        1 element per row.
template <typename T, int THREADS, bool COMPUTED_CB>
__global__ void __launch_bounds__(THREADS)
qtip_fused_gemv_v2_k4_l16_kernel(
    const uint8_t* __restrict__ packed,       // [n_rows, packed_per_row]
    const float*   __restrict__ row_scales,   // [n_rows]
    const float*   __restrict__ lut,          // [2^L * V] (unused when computed)
    const T*       __restrict__ x,            // [k_in]   (k_in == num_weights)
    T*             __restrict__ y,            // [n_rows]
    int n_rows,
    int packed_per_row,
    int num_symbols,
    unsigned int cb_mult
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int N_WARPS = THREADS / 32;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");

    const uint8_t* row_packed = packed + (size_t)row * packed_per_row;
    const float scale = __ldg(row_scales + row);

    // Each thread takes a contiguous slice of symbols. We split the row
    // evenly across THREADS, with the last thread taking the remainder.
    const int sym_per_thread = (num_symbols + THREADS - 1) / THREADS;
    const int sym_start_raw  = tid * sym_per_thread;
    const int sym_end        = min(num_symbols, sym_start_raw + sym_per_thread);

    float acc = 0.0f;

    if (sym_start_raw < num_symbols) {
        // ---- Warmup ----
        // For t < WARMUP_SYMS the trellis state genuinely depends on the
        // initial state=0; for t >= WARMUP_SYMS, the state is fully
        // determined by the last WARMUP_SYMS symbols. We therefore:
        //   * Thread 0 starts at state=0 and decodes symbols [0, sym_end).
        //   * Threads i>0 read WARMUP_SYMS prior symbols out of `packed`
        //     to seed their state, then decode their slice [sym_start_raw,
        //     sym_end).
        //
        // The warmup symbols are read but their decoded weights are NOT
        // accumulated — they belong to the prior thread's slice.

        uint32_t state = 0;
        int sym_idx = sym_start_raw;

        if (tid > 0) {
            // Walk WARMUP_SYMS prior symbols. Each may itself need warmup if
            // it's in the first 3 positions; but state=0 initial gives the
            // right value (and only the last 4 symbols' worth of state
            // survives in the L=16 register anyway, so it's exact).
            int warm_start = max(0, sym_start_raw - (int)QTIP_WARMUP_SYMS);
            // Walk every symbol from `warm_start` to `sym_start_raw - 1`,
            // shifting the state by K each step.
            for (int t = warm_start; t < sym_start_raw; ++t) {
                int byte_idx = t >> 1;
                uint8_t byte = __ldg(row_packed + byte_idx);
                uint32_t sym = (t & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
            }
        }

        // ---- Decode loop ----
        // For each symbol in [sym_idx, sym_end):
        //   1. Read its nibble from `packed`.
        //   2. Shift into state.
        //   3. Load LUT[state] → 2 weight values.
        //   4. Multiply by scale and the 2 corresponding x values.
        //   5. Accumulate.
        //
        // Hot-path optimization: we walk byte-by-byte when possible (2
        // symbols per byte), letting the compiler fuse the two LUT loads
        // and four fmaf calls.

        // Process pairs of symbols (one byte) at a time where alignment lets us.
        // The byte read goes through the read-only cache (__ldg) — same row
        // is touched by exactly one thread, so no aliasing concern.
        while (sym_idx < sym_end) {
            int byte_idx = sym_idx >> 1;
            uint8_t byte = __ldg(row_packed + byte_idx);

            // Always process the low nibble (symbol sym_idx if even, else
            // already consumed during warmup).
            bool do_lo = ((sym_idx & 1) == 0);
            uint32_t sym_lo = byte & 0x0F;
            uint32_t sym_hi = (byte >> 4) & 0x0F;

            if (do_lo) {
                state = ((state << QTIP_K) | sym_lo) & QTIP_STATE_MASK;
                // LUT path: 512 KiB FP32, L2-resident, __ldg read-only.
                // Computed path: no load at all.
                const float2 c = qtip_cb_value_ldg<COMPUTED_CB>(lut, state, cb_mult);
                float w0 = c.x * scale;
                float w1 = c.y * scale;
                int x_off = sym_idx * (int)QTIP_V;
                float x0 = to_f32<T>(__ldg(x + x_off + 0));
                float x1 = to_f32<T>(__ldg(x + x_off + 1));
                acc = fmaf(w0, x0, acc);
                acc = fmaf(w1, x1, acc);
                ++sym_idx;
                if (sym_idx >= sym_end) break;
            }

            // High nibble (the second symbol in this byte).
            state = ((state << QTIP_K) | sym_hi) & QTIP_STATE_MASK;
            const float2 c_hi = qtip_cb_value_ldg<COMPUTED_CB>(lut, state, cb_mult);
            float w0 = c_hi.x * scale;
            float w1 = c_hi.y * scale;
            int x_off = sym_idx * (int)QTIP_V;
            float x0 = to_f32<T>(__ldg(x + x_off + 0));
            float x1 = to_f32<T>(__ldg(x + x_off + 1));
            acc = fmaf(w0, x0, acc);
            acc = fmaf(w1, x1, acc);
            ++sym_idx;
        }
    }

    // ---- Block reduction ----
    // Warp-level first, then a single warp aggregates across warps.
    acc = warp_reduce_sum(acc);

    __shared__ float warp_sums[N_WARPS];
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            y[row] = from_f32<T>(v);
        }
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// `cb_mult == 0` selects the stored Gaussian LUT; nonzero selects the computed
// sum2 codebook with that MCG multiplier. See qtip_codebook.cuh.
#define QTIP_GEMV_LAUNCHER(NAME, T)                                                \
    void NAME(const uint8_t* d_packed,                                             \
              const float*   d_row_scales,                                         \
              const float*   d_lut,                                                \
              const T*       d_x_rotated,                                          \
              T*             d_y,                                                  \
              int n_rows,                                                          \
              int packed_per_row,                                                  \
              int num_symbols,                                                     \
              unsigned int cb_mult,                                                \
              cudaStream_t   stream) {                                             \
        constexpr int THREADS = 128;                                               \
        if (cb_mult != 0u) {                                                       \
            qtip_fused_gemv_v2_k4_l16_kernel<T, THREADS, true>                     \
                <<<n_rows, THREADS, 0, stream>>>(                                  \
                    d_packed, d_row_scales, d_lut, d_x_rotated, d_y,               \
                    n_rows, packed_per_row, num_symbols, cb_mult);                 \
        } else {                                                                   \
            qtip_fused_gemv_v2_k4_l16_kernel<T, THREADS, false>                    \
                <<<n_rows, THREADS, 0, stream>>>(                                  \
                    d_packed, d_row_scales, d_lut, d_x_rotated, d_y,               \
                    n_rows, packed_per_row, num_symbols, 0u);                      \
        }                                                                          \
    }

QTIP_GEMV_LAUNCHER(launch_qtip_fused_gemv_v2_k4_l16_bf16, __nv_bfloat16)
QTIP_GEMV_LAUNCHER(launch_qtip_fused_gemv_v2_k4_l16_f16,  __half)
QTIP_GEMV_LAUNCHER(launch_qtip_fused_gemv_v2_k4_l16_f32,  float)

#undef QTIP_GEMV_LAUNCHER

} // extern "C"
