/**
 * TurboQuant Walsh-Hadamard Transform + Quantize CUDA kernels.
 *
 * Implements the TurboQuant write path as fused GPU operations:
 * 1. L2 normalization
 * 2. D·H·D rotation (random signs + Hadamard butterfly + random signs)
 * 3. Lloyd-Max scalar quantization via codebook lookup
 * 4. Sub-byte index packing (4-bit nibble or 3-bit 10-in-32)
 *
 * Reference: spiritbuun/llama-cpp-turboquant-cuda, TurboQuant ICLR 2026.
 */

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>

// ============================================================================
// Codebook constants (compiled into the binary)
// ============================================================================

// 4-bit codebook for d=128 (16 centroids)
__constant__ float CODEBOOK_4BIT_D128[16] = {
    -0.237664013127f, -0.180836062501f, -0.141805261760f, -0.110288414632f,
    -0.082828489390f, -0.057772320256f, -0.034151583096f, -0.011302500645f,
     0.011302500645f,  0.034151583096f,  0.057772320256f,  0.082828489390f,
     0.110288414632f,  0.141805261760f,  0.180836062501f,  0.237664013127f,
};

// 4-bit boundaries for d=128 (17 values)
__constant__ float BOUNDARIES_4BIT_D128[17] = {
    -1.0f, -0.209250037814f, -0.161320662130f, -0.126046838196f,
    -0.096558452011f, -0.070300404823f, -0.045961951676f, -0.022727041871f,
     0.0f,  0.022727041871f,  0.045961951676f,  0.070300404823f,
     0.096558452011f,  0.126046838196f,  0.161320662130f,  0.209250037814f, 1.0f,
};

// 3-bit codebook for d=128 (8 centroids)
__constant__ float CODEBOOK_3BIT_D128[8] = {
    -0.188397319183f, -0.118139828402f, -0.066585638471f, -0.021604320011f,
     0.021604320011f,  0.066585638471f,  0.118139828402f,  0.188397319183f,
};

// 3-bit boundaries for d=128 (9 values)
__constant__ float BOUNDARIES_3BIT_D128[9] = {
    -1.0f, -0.153268573792f, -0.092362733436f, -0.044094979241f,
     0.0f,  0.044094979241f,  0.092362733436f,  0.153268573792f, 1.0f,
};

// 2-bit codebook for d=128 (4 centroids)
__constant__ float CODEBOOK_2BIT_D128[4] = {
    -0.133041590561f, -0.039991612341f, 0.039991612341f, 0.133041590561f,
};

// 2-bit boundaries for d=128 (5 values)
__constant__ float BOUNDARIES_2BIT_D128[5] = {
    -1.0f, -0.086516601451f, 0.0f, 0.086516601451f, 1.0f,
};

// ============================================================================
// Sign arrays for D·H·D rotation (deterministic from seed=42, dim=128)
// ============================================================================

// Pre-computed random ±1 signs using xorshift64 with seed = 42 ^ (128 * 0x9E3779B97F4A7C15)
// Must match the Rust generate_signs(42, 128) output exactly.
__constant__ float SIGNS_128[128] = {
    -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1,  1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1, -1,
    -1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1, -1,  1,  1,
    -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,
};

// ============================================================================
// Walsh-Hadamard Transform kernel (in shared memory)
// ============================================================================

/**
 * In-place WHT butterfly in shared memory.
 * data: shared memory array of `dim` floats.
 * dim: must be power of 2 (64, 128, 256).
 * tid: thread index within the block.
 * n_threads: threads per block (typically dim/2 or dim).
 */
__device__ void wht_butterfly_shared(float* data, int dim, int tid, int n_threads) {
    for (int h = 1; h < dim; h *= 2) {
        __syncthreads();
        // Each thread handles one butterfly pair
        for (int idx = tid; idx < dim / 2; idx += n_threads) {
            int block_start = (idx / h) * (h * 2);
            int offset = idx % h;
            int i = block_start + offset;
            int j = i + h;
            float a = data[i];
            float b = data[j];
            data[i] = a + b;
            data[j] = a - b;
        }
    }
    __syncthreads();
}

// ============================================================================
// Scalar quantization helpers
// ============================================================================

/**
 * Quantize a single value using binary search on boundaries.
 * Returns index in [0, n_levels-1].
 */
__device__ uint8_t quantize_scalar(float x, const float* boundaries, int n_levels) {
    // Boundaries has n_levels+1 entries
    if (x <= boundaries[1]) return 0;
    if (x >= boundaries[n_levels]) return (uint8_t)(n_levels - 1);

    int lo = 1, hi = n_levels;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (x < boundaries[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return (uint8_t)(lo - 1);
}

// ============================================================================
// Fused WHT + Quantize + Pack kernel (4-bit keys)
// ============================================================================

/**
 * Fused kernel: normalize → D·H·D rotate → 4-bit quantize → nibble pack.
 *
 * Each block processes one vector (one head of one token).
 * Input:  vectors [batch, dim] as float/half
 * Output: packed [batch, dim/2] as uint8 (nibble-packed 4-bit indices)
 *         norms  [batch] as half (L2 norms)
 *
 * Grid:  (batch, 1, 1)
 * Block: (128, 1, 1) for dim=128, (256, 1, 1) for dim=256
 */
extern "C" __global__ void turbo_quantize_4bit_d128(
    const float* __restrict__ input,   // [batch, 128]
    uint8_t*     __restrict__ packed,   // [batch, 64] (nibble-packed)
    __half*      __restrict__ norms,    // [batch]
    int batch_size
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int dim = 128;

    __shared__ float smem[128];

    // Step 1: Load input to shared memory
    if (tid < dim) {
        smem[tid] = input[bid * dim + tid];
    }
    __syncthreads();

    // Step 2: Compute L2 norm via parallel reduction
    __shared__ float norm_smem[128];
    norm_smem[tid] = (tid < dim) ? smem[tid] * smem[tid] : 0.0f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            norm_smem[tid] += norm_smem[tid + s];
        }
        __syncthreads();
    }
    float norm = sqrtf(norm_smem[0]);

    // Store norm
    if (tid == 0) {
        norms[bid] = __float2half(norm);
    }

    // Step 3: Normalize to unit sphere
    if (tid < dim && norm > 1e-10f) {
        smem[tid] /= norm;
    }
    __syncthreads();

    // Step 4: First sign flip (D · x)
    if (tid < dim) {
        smem[tid] *= SIGNS_128[tid];
    }
    __syncthreads();

    // Step 5: WHT butterfly (H · D · x)
    wht_butterfly_shared(smem, dim, tid, 128);

    // Step 6: Normalize by 1/sqrt(dim)
    const float wht_scale = 0.08838834764831845f; // 1/sqrt(128)
    if (tid < dim) {
        smem[tid] *= wht_scale;
    }
    __syncthreads();

    // Step 7: Second sign flip (D · H · D · x)
    if (tid < dim) {
        smem[tid] *= SIGNS_128[tid];
    }
    __syncthreads();

    // Step 8: Quantize + nibble pack
    // Each thread handles 2 consecutive values → 1 output byte
    if (tid < dim / 2) {
        float v0 = smem[2 * tid];
        float v1 = smem[2 * tid + 1];
        uint8_t idx0 = quantize_scalar(v0, BOUNDARIES_4BIT_D128, 16);
        uint8_t idx1 = quantize_scalar(v1, BOUNDARIES_4BIT_D128, 16);
        packed[bid * (dim / 2) + tid] = (idx0 & 0xF) | ((idx1 & 0xF) << 4);
    }
}

/**
 * Fused kernel: normalize → D·H·D rotate → 3-bit quantize → 10-in-32 pack.
 *
 * Each block processes one vector (one head of one token).
 * Output: packed [batch, ceil(dim/10)*4] as uint8
 */
extern "C" __global__ void turbo_quantize_3bit_d128(
    const float* __restrict__ input,   // [batch, 128]
    uint8_t*     __restrict__ packed,   // [batch, 52] (10-in-32 packed)
    __half*      __restrict__ norms,    // [batch]
    int batch_size
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int dim = 128;

    __shared__ float smem[128];
    __shared__ uint8_t indices[128];

    // Step 1: Load input
    if (tid < dim) {
        smem[tid] = input[bid * dim + tid];
    }
    __syncthreads();

    // Step 2: L2 norm
    __shared__ float norm_smem[128];
    norm_smem[tid] = (tid < dim) ? smem[tid] * smem[tid] : 0.0f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) norm_smem[tid] += norm_smem[tid + s];
        __syncthreads();
    }
    float norm = sqrtf(norm_smem[0]);
    if (tid == 0) norms[bid] = __float2half(norm);

    // Step 3: Normalize
    if (tid < dim && norm > 1e-10f) smem[tid] /= norm;
    __syncthreads();

    // Step 4-7: D·H·D rotation (same as 4-bit kernel)
    if (tid < dim) smem[tid] *= SIGNS_128[tid];
    __syncthreads();
    wht_butterfly_shared(smem, dim, tid, 128);
    const float wht_scale = 0.08838834764831845f;
    if (tid < dim) smem[tid] *= wht_scale;
    __syncthreads();
    if (tid < dim) smem[tid] *= SIGNS_128[tid];
    __syncthreads();

    // Step 8: Quantize all elements
    if (tid < dim) {
        indices[tid] = quantize_scalar(smem[tid], BOUNDARIES_3BIT_D128, 8);
    }
    __syncthreads();

    // Step 9: 10-in-32 packing (13 groups of 10, last group has 8 elements)
    // Each of the first 13 threads packs one group
    const int n_groups = (dim + 9) / 10; // 13 for dim=128
    if (tid < n_groups) {
        int base = tid * 10;
        uint32_t word = 0;
        int count = (base + 10 <= dim) ? 10 : (dim - base);
        for (int j = 0; j < count; j++) {
            word |= ((uint32_t)indices[base + j] & 0x7) << (j * 3);
        }
        // Write as 4 bytes (little-endian)
        int out_offset = bid * n_groups * 4 + tid * 4;
        packed[out_offset + 0] = (uint8_t)(word);
        packed[out_offset + 1] = (uint8_t)(word >> 8);
        packed[out_offset + 2] = (uint8_t)(word >> 16);
        packed[out_offset + 3] = (uint8_t)(word >> 24);
    }
}

// ============================================================================
// Standalone WHT kernels (for rotating queries during attention)
// ============================================================================

/**
 * Apply D·H·D rotation to a batch of 128-dim vectors in-place.
 * Used to rotate queries before computing attention against compressed keys.
 *
 * Grid:  (batch, 1, 1)
 * Block: (128, 1, 1)
 */
extern "C" __global__ void turbo_wht_rotate_d128(
    float* __restrict__ data,  // [batch, 128], modified in-place
    int batch_size
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int dim = 128;

    __shared__ float smem[128];

    // Load
    if (tid < dim) {
        smem[tid] = data[bid * dim + tid];
    }
    __syncthreads();

    // D · x
    if (tid < dim) smem[tid] *= SIGNS_128[tid];
    __syncthreads();

    // H · D · x
    wht_butterfly_shared(smem, dim, tid, 128);

    // Normalize
    const float wht_scale = 0.08838834764831845f;
    if (tid < dim) smem[tid] *= wht_scale;
    __syncthreads();

    // D · H · D · x
    if (tid < dim) smem[tid] *= SIGNS_128[tid];
    __syncthreads();

    // Store
    if (tid < dim) {
        data[bid * dim + tid] = smem[tid];
    }
}

/**
 * Dequantize 4-bit packed indices: unpack → codebook lookup → inverse WHT → scale by norm.
 *
 * Grid:  (batch, 1, 1)
 * Block: (128, 1, 1)
 */
extern "C" __global__ void turbo_dequantize_4bit_d128(
    const uint8_t* __restrict__ packed,  // [batch, 64]
    const __half*  __restrict__ norms,   // [batch]
    float*         __restrict__ output,  // [batch, 128]
    int batch_size
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int dim = 128;

    __shared__ float smem[128];

    // Step 1: Unpack nibbles and lookup centroids
    if (tid < dim / 2) {
        uint8_t byte = packed[bid * (dim / 2) + tid];
        uint8_t idx0 = byte & 0xF;
        uint8_t idx1 = (byte >> 4) & 0xF;
        smem[2 * tid]     = CODEBOOK_4BIT_D128[idx0];
        smem[2 * tid + 1] = CODEBOOK_4BIT_D128[idx1];
    }
    __syncthreads();

    // Step 2: Inverse WHT (D·H·D is self-inverse with normalized H)
    if (tid < dim) smem[tid] *= SIGNS_128[tid];
    __syncthreads();
    wht_butterfly_shared(smem, dim, tid, 128);
    const float wht_scale = 0.08838834764831845f;
    if (tid < dim) smem[tid] *= wht_scale;
    __syncthreads();
    if (tid < dim) smem[tid] *= SIGNS_128[tid];
    __syncthreads();

    // Step 3: Scale by norm
    float norm = __half2float(norms[bid]);
    if (tid < dim) {
        output[bid * dim + tid] = smem[tid] * norm;
    }
}
