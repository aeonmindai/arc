// QTIP 2-bit dequantize kernel for V=2, K=4, L=16 trellis format.
//
// Format: packed[N rows, num_symbols/2 bytes] holds K=4-bit symbols, low
// nibble of byte b = symbol 2b, high nibble = symbol 2b+1. Each symbol
// shifts a 16-bit trellis state by K=4 bits; the new state indexes into
// a (2^L) x V LUT producing V=2 float weights per symbol.
//
// This kernel is Tier-A Greedy mode only — rotation is NOT applied here
// (use rotation-disabled QTIP checkpoints, i.e. QtipMode::Greedy). The
// stored LUT is shared across the layer.
//
// Launch: one thread per row. Sequential trellis walk per thread.
// Bandwidth-bound on LUT lookup; LUT (512 KiB) lives mostly in L2 cache.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr uint32_t K = 4;       // bits per symbol
constexpr uint32_t L = 16;      // trellis state width
constexpr uint32_t V = 2;       // weights per state
constexpr uint32_t STATE_MASK = (1u << L) - 1u;

template <typename T>
__device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}

// One thread per row. Walks the row's packed symbol stream, decodes
// the trellis sequentially, writes V*num_symbols dequantized values
// scaled by row_scales[row].
template <typename T>
__global__ void qtip_dequantize_v2_k4_l16_kernel(
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ row_scales,
    const float*   __restrict__ lut,
    T*             __restrict__ out,
    int n_rows,
    int packed_per_row,
    int num_symbols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const uint8_t* my_packed = packed + (size_t)row * packed_per_row;
    float scale = row_scales[row];
    T* my_out = out + (size_t)row * (num_symbols * V);

    uint32_t state = 0;

    // Loop over symbols. We can read 2 symbols per byte, so iterate by 2.
    for (int sym_pair = 0; sym_pair < packed_per_row; sym_pair++) {
        uint8_t byte = my_packed[sym_pair];
        uint8_t sym_lo = byte & 0x0F;
        uint8_t sym_hi = (byte >> 4) & 0x0F;

        // First (low-nibble) symbol.
        state = ((state << K) | (uint32_t)sym_lo) & STATE_MASK;
        {
            int sym_idx = sym_pair * 2;
            if (sym_idx < num_symbols) {
                float w0 = lut[state * V + 0] * scale;
                float w1 = lut[state * V + 1] * scale;
                my_out[sym_idx * V + 0] = from_f32<T>(w0);
                my_out[sym_idx * V + 1] = from_f32<T>(w1);
            }
        }

        // Second (high-nibble) symbol.
        state = ((state << K) | (uint32_t)sym_hi) & STATE_MASK;
        {
            int sym_idx = sym_pair * 2 + 1;
            if (sym_idx < num_symbols) {
                float w0 = lut[state * V + 0] * scale;
                float w1 = lut[state * V + 1] * scale;
                my_out[sym_idx * V + 0] = from_f32<T>(w0);
                my_out[sym_idx * V + 1] = from_f32<T>(w1);
            }
        }
    }
}

} // namespace

extern "C" {

void launch_qtip_dequantize_v2_k4_l16_bf16(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    __nv_bfloat16* d_out,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    cudaStream_t stream
) {
    const int threads = 64;
    const int blocks = (n_rows + threads - 1) / threads;
    qtip_dequantize_v2_k4_l16_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        d_packed, d_row_scales, d_lut, d_out, n_rows, packed_per_row, num_symbols);
}

void launch_qtip_dequantize_v2_k4_l16_f16(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    __half*        d_out,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    cudaStream_t stream
) {
    const int threads = 64;
    const int blocks = (n_rows + threads - 1) / threads;
    qtip_dequantize_v2_k4_l16_kernel<__half><<<blocks, threads, 0, stream>>>(
        d_packed, d_row_scales, d_lut, d_out, n_rows, packed_per_row, num_symbols);
}

void launch_qtip_dequantize_v2_k4_l16_f32(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    float*         d_out,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    cudaStream_t stream
) {
    const int threads = 64;
    const int blocks = (n_rows + threads - 1) / threads;
    qtip_dequantize_v2_k4_l16_kernel<float><<<blocks, threads, 0, stream>>>(
        d_packed, d_row_scales, d_lut, d_out, n_rows, packed_per_row, num_symbols);
}

} // extern "C"
