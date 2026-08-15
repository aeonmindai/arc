// QTIP CUDA dequantize + activation-rotation kernels.
//
// Format: V=2, K=4, L=16. Each row stores `num_symbols/2` packed bytes
// (low nibble = symbol 2b, high nibble = symbol 2b+1). The trellis state
// updates as `state = ((state << K) | sym) & ((1<<L) - 1)`. The shared LUT
// is `(1<<L) x V` floats. Output is `n_rows x num_weights`, where
// `num_weights = num_symbols * V`.
//
// Two kernel groups:
//   1) `qtip_dequantize_v2_k4_l16_*` — packed → dtype matrix (rotated frame
//      when the QtipLayer carries rotation_signs). The kernel itself does
//      not touch rotation; it simply produces the weight matrix that lives
//      in the rotated frame as stored. Rotation is applied to `x`
//      separately (see below) so the matmul `(xR) @ (WR)^T = x @ W^T`
//      identity holds.
//   2) `qtip_rotate_x_*` — apply the block-diagonal D·H·D rotation
//      in-place to activation rows. Each block handles one rotation block;
//      grid is `(batch_size, blocks_per_row)`. Hadamard block size is the
//      compile-time `BLOCK` template arg (we instantiate 2,4,8,16,32,64,
//      128 — matches `QTIP_ROTATION_MAX_BLOCK=128` on the Rust side).
//
// Both kernel groups are SM80+ only (uses __nv_bfloat16 ops). The Rust
// side gates on `has_qtip_kernels` (set in build.rs when CC >= 80).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The codebook: either the stored 512 KiB Gaussian table or the in-register
// "sum2" computed code, selected at compile time per instantiation.
#include "qtip_codebook.cuh"

namespace {

constexpr uint32_t QTIP_K          = 4;
constexpr uint32_t QTIP_L          = 16;
constexpr uint32_t QTIP_V          = 2;
constexpr uint32_t QTIP_STATE_MASK = (1u << QTIP_L) - 1u;

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

// ---------------------------------------------------------------------------
// Dequantize kernel: one thread per row.
// ---------------------------------------------------------------------------
//
// The trellis is sequential, so a per-row decode is the natural unit of
// work. Grid: (ceil(n_rows / THREADS), 1, 1). Block: (THREADS, 1, 1).
//
// Memory traffic per row:
//   * packed reads: packed_per_row bytes (a few hundred to a few thousand)
//   * LUT lookups:  num_symbols * V random-ish accesses → relies on L2
//                   (LUT is 512 KiB total, fits in H100's 60 MB L2 many times
//                    over; for one row, the working set is much smaller).
//   * scale read:   1 float
//   * output write: num_weights * sizeof(T) coalesced stores.
//
// We pre-multiply scale into the local accumulator (free), so each write is
// a fused mul + dtype convert + store.

template <typename T, int THREADS, bool COMPUTED_CB>
__global__ void qtip_dequantize_v2_k4_l16_kernel(
    const uint8_t* __restrict__ packed,       // [n_rows, packed_per_row]
    const float*   __restrict__ row_scales,   // [n_rows]
    const float*   __restrict__ lut,          // [2^L * V] (unused when computed)
    T*             __restrict__ out,          // [n_rows, num_weights]
    int n_rows,
    int packed_per_row,
    int num_symbols,
    unsigned int cb_mult
) {
    int row = blockIdx.x * THREADS + threadIdx.x;
    if (row >= n_rows) return;

    const uint8_t* my_packed = packed + (size_t)row * packed_per_row;
    const int num_weights = num_symbols * (int)QTIP_V;
    T* my_out             = out + (size_t)row * num_weights;
    const float scale     = row_scales[row];

    uint32_t state = 0;
    #pragma unroll 4
    for (int b = 0; b < packed_per_row; ++b) {
        uint8_t byte   = my_packed[b];
        uint8_t sym_lo = byte & 0x0F;
        uint8_t sym_hi = (byte >> 4) & 0x0F;

        // First symbol (low nibble).
        state = ((state << QTIP_K) | (uint32_t)sym_lo) & QTIP_STATE_MASK;
        int out_idx_lo = (b * 2) * (int)QTIP_V;
        float2 c = qtip_cb_value<COMPUTED_CB>(lut, state, cb_mult);
        my_out[out_idx_lo + 0] = from_f32<T>(c.x * scale);
        my_out[out_idx_lo + 1] = from_f32<T>(c.y * scale);

        // Second symbol (high nibble).
        state = ((state << QTIP_K) | (uint32_t)sym_hi) & QTIP_STATE_MASK;
        int out_idx_hi = (b * 2 + 1) * (int)QTIP_V;
        c = qtip_cb_value<COMPUTED_CB>(lut, state, cb_mult);
        my_out[out_idx_hi + 0] = from_f32<T>(c.x * scale);
        my_out[out_idx_hi + 1] = from_f32<T>(c.y * scale);
    }
}

// ---------------------------------------------------------------------------
// Block-diagonal D·H·D rotation kernel (in-place on `x`)
// ---------------------------------------------------------------------------
//
// Each CUDA block handles ONE rotation block (BLOCK consecutive elements of
// one batch row). `BLOCK` is a power-of-2 in [2, 128]. `signs` is a per-
// feature ±1 vector of length in_features.
//
// Grid: (batch_size, blocks_per_row).
// Block: (BLOCK, 1, 1).
//
// Math: each block performs `x_block ← D · H · D · x_block` where:
//   D  = diag(signs_block) — sign flip
//   H  = normalized Hadamard, applied via the FWHT butterfly with the
//        standard 1/sqrt(BLOCK) scaling so the transform is involutory.
//
// Matches the CPU `apply_block_rotation` semantics in
// `mistralrs-quant/src/qtip/mod.rs:117` (signs are FP32 ±1, blocks are
// applied independently and contiguously).

template <typename T, int BLOCK>
__global__ void qtip_rotate_x_kernel(
    T*           __restrict__ x,         // [batch_size, in_features], in-place
    const float* __restrict__ signs,     // [in_features]
    int in_features
) {
    static_assert(BLOCK >= 2 && BLOCK <= 256, "BLOCK must be in [2, 256]");

    int batch_idx        = blockIdx.x;
    int block_in_row     = blockIdx.y;
    int feature_off      = block_in_row * BLOCK + threadIdx.x;
    if (feature_off >= in_features) return;
    int row_off          = batch_idx * in_features + feature_off;

    __shared__ float smem[BLOCK];

    // Load row block into shared memory in FP32, apply first D.
    float sign = signs[feature_off];
    float v    = to_f32<T>(x[row_off]);
    smem[threadIdx.x] = v * sign;
    __syncthreads();

    // FWHT butterfly. Each thread cooperates: at stride h, pair = lane ^ h.
    // (lane & h) == 0 lane writes `a + b`, the other writes `b - a` where
    // a = smem[lane] (pre-stage) and b = smem[pair_lane].
    //
    // We need to capture both pre-stage values before any writes, so we
    // load → sync → write → sync.
    int lane = threadIdx.x;
    #pragma unroll
    for (int h = 1; h < BLOCK; h *= 2) {
        int pair = lane ^ h;
        float a  = smem[lane];
        float b  = smem[pair];
        __syncthreads();
        smem[lane] = ((lane & h) == 0) ? (a + b) : (b - a);
        __syncthreads();
    }

    // 1/sqrt(BLOCK) normalization, then second D.
    float wht_scale = rsqrtf((float)BLOCK);
    float rotated   = smem[threadIdx.x] * wht_scale * sign;

    x[row_off] = from_f32<T>(rotated);
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// ----- Dequantize -----------------------------------------------------------

// `cb_mult == 0` selects the stored Gaussian LUT; nonzero selects the computed
// sum2 codebook with that MCG multiplier. See qtip_codebook.cuh.
#define QTIP_DEQ_LAUNCHER(NAME, T)                                                 \
    void NAME(const uint8_t* d_packed,                                             \
              const float*   d_row_scales,                                         \
              const float*   d_lut,                                                \
              T*             d_out,                                                \
              int n_rows,                                                          \
              int packed_per_row,                                                  \
              int num_symbols,                                                     \
              unsigned int cb_mult,                                                \
              cudaStream_t   stream) {                                             \
        constexpr int THREADS = 64;                                                \
        const int blocks = (n_rows + THREADS - 1) / THREADS;                       \
        if (cb_mult != 0u) {                                                       \
            qtip_dequantize_v2_k4_l16_kernel<T, THREADS, true>                     \
                <<<blocks, THREADS, 0, stream>>>(                                  \
                    d_packed, d_row_scales, d_lut, d_out,                          \
                    n_rows, packed_per_row, num_symbols, cb_mult);                 \
        } else {                                                                   \
            qtip_dequantize_v2_k4_l16_kernel<T, THREADS, false>                    \
                <<<blocks, THREADS, 0, stream>>>(                                  \
                    d_packed, d_row_scales, d_lut, d_out,                          \
                    n_rows, packed_per_row, num_symbols, 0u);                      \
        }                                                                          \
    }

QTIP_DEQ_LAUNCHER(launch_qtip_dequantize_v2_k4_l16_bf16, __nv_bfloat16)
QTIP_DEQ_LAUNCHER(launch_qtip_dequantize_v2_k4_l16_f16,  __half)
QTIP_DEQ_LAUNCHER(launch_qtip_dequantize_v2_k4_l16_f32,  float)

#undef QTIP_DEQ_LAUNCHER

// ----- In-place activation rotation -----------------------------------------

// Real LLMs use block sizes that are powers of 2 in [2, 128]
// (rotation_block_size() in mod.rs caps at QTIP_ROTATION_MAX_BLOCK=128
// and picks the largest power-of-2 divisor of in_features below that).

#define QTIP_ROTATE_LAUNCHER(NAME, T, BLOCK)                                       \
    static void NAME(T* d_x,                                                       \
                     const float* d_signs,                                         \
                     int batch_size,                                               \
                     int in_features,                                              \
                     cudaStream_t stream) {                                        \
        const int blocks_per_row = in_features / (BLOCK);                          \
        dim3 grid(batch_size, blocks_per_row);                                     \
        qtip_rotate_x_kernel<T, (BLOCK)>                                           \
            <<<grid, (BLOCK), 0, stream>>>(d_x, d_signs, in_features);             \
    }

QTIP_ROTATE_LAUNCHER(rot_bf16_2,   __nv_bfloat16, 2)
QTIP_ROTATE_LAUNCHER(rot_bf16_4,   __nv_bfloat16, 4)
QTIP_ROTATE_LAUNCHER(rot_bf16_8,   __nv_bfloat16, 8)
QTIP_ROTATE_LAUNCHER(rot_bf16_16,  __nv_bfloat16, 16)
QTIP_ROTATE_LAUNCHER(rot_bf16_32,  __nv_bfloat16, 32)
QTIP_ROTATE_LAUNCHER(rot_bf16_64,  __nv_bfloat16, 64)
QTIP_ROTATE_LAUNCHER(rot_bf16_128, __nv_bfloat16, 128)

QTIP_ROTATE_LAUNCHER(rot_f16_2,   __half, 2)
QTIP_ROTATE_LAUNCHER(rot_f16_4,   __half, 4)
QTIP_ROTATE_LAUNCHER(rot_f16_8,   __half, 8)
QTIP_ROTATE_LAUNCHER(rot_f16_16,  __half, 16)
QTIP_ROTATE_LAUNCHER(rot_f16_32,  __half, 32)
QTIP_ROTATE_LAUNCHER(rot_f16_64,  __half, 64)
QTIP_ROTATE_LAUNCHER(rot_f16_128, __half, 128)

QTIP_ROTATE_LAUNCHER(rot_f32_2,   float, 2)
QTIP_ROTATE_LAUNCHER(rot_f32_4,   float, 4)
QTIP_ROTATE_LAUNCHER(rot_f32_8,   float, 8)
QTIP_ROTATE_LAUNCHER(rot_f32_16,  float, 16)
QTIP_ROTATE_LAUNCHER(rot_f32_32,  float, 32)
QTIP_ROTATE_LAUNCHER(rot_f32_64,  float, 64)
QTIP_ROTATE_LAUNCHER(rot_f32_128, float, 128)

#undef QTIP_ROTATE_LAUNCHER

void launch_qtip_rotate_x_bf16(
    __nv_bfloat16* d_x,
    const float*   d_signs,
    int batch_size,
    int in_features,
    int block_size,
    cudaStream_t   stream
) {
    switch (block_size) {
        case 2:   rot_bf16_2(d_x, d_signs, batch_size, in_features, stream); break;
        case 4:   rot_bf16_4(d_x, d_signs, batch_size, in_features, stream); break;
        case 8:   rot_bf16_8(d_x, d_signs, batch_size, in_features, stream); break;
        case 16:  rot_bf16_16(d_x, d_signs, batch_size, in_features, stream); break;
        case 32:  rot_bf16_32(d_x, d_signs, batch_size, in_features, stream); break;
        case 64:  rot_bf16_64(d_x, d_signs, batch_size, in_features, stream); break;
        case 128: rot_bf16_128(d_x, d_signs, batch_size, in_features, stream); break;
        default:  break; // unsupported — caller falls back to CPU
    }
}

void launch_qtip_rotate_x_f16(
    __half*        d_x,
    const float*   d_signs,
    int batch_size,
    int in_features,
    int block_size,
    cudaStream_t   stream
) {
    switch (block_size) {
        case 2:   rot_f16_2(d_x, d_signs, batch_size, in_features, stream); break;
        case 4:   rot_f16_4(d_x, d_signs, batch_size, in_features, stream); break;
        case 8:   rot_f16_8(d_x, d_signs, batch_size, in_features, stream); break;
        case 16:  rot_f16_16(d_x, d_signs, batch_size, in_features, stream); break;
        case 32:  rot_f16_32(d_x, d_signs, batch_size, in_features, stream); break;
        case 64:  rot_f16_64(d_x, d_signs, batch_size, in_features, stream); break;
        case 128: rot_f16_128(d_x, d_signs, batch_size, in_features, stream); break;
        default:  break;
    }
}

void launch_qtip_rotate_x_f32(
    float*         d_x,
    const float*   d_signs,
    int batch_size,
    int in_features,
    int block_size,
    cudaStream_t   stream
) {
    switch (block_size) {
        case 2:   rot_f32_2(d_x, d_signs, batch_size, in_features, stream); break;
        case 4:   rot_f32_4(d_x, d_signs, batch_size, in_features, stream); break;
        case 8:   rot_f32_8(d_x, d_signs, batch_size, in_features, stream); break;
        case 16:  rot_f32_16(d_x, d_signs, batch_size, in_features, stream); break;
        case 32:  rot_f32_32(d_x, d_signs, batch_size, in_features, stream); break;
        case 64:  rot_f32_64(d_x, d_signs, batch_size, in_features, stream); break;
        case 128: rot_f32_128(d_x, d_signs, batch_size, in_features, stream); break;
        default:  break;
    }
}

} // extern "C"
