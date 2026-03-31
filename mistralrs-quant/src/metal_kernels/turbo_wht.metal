/**
 * TurboQuant Walsh-Hadamard Transform + Quantize Metal kernels.
 *
 * Metal equivalents of the CUDA turbo_wht.cu kernels.
 * Reference: Infatoshi/exllama-ish exl4_hadamard.metal, TurboQuant ICLR 2026.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Codebook constants
// ============================================================================

constant float CODEBOOK_4BIT_D128[16] = {
    -0.237664013127f, -0.180836062501f, -0.141805261760f, -0.110288414632f,
    -0.082828489390f, -0.057772320256f, -0.034151583096f, -0.011302500645f,
     0.011302500645f,  0.034151583096f,  0.057772320256f,  0.082828489390f,
     0.110288414632f,  0.141805261760f,  0.180836062501f,  0.237664013127f,
};

constant float BOUNDARIES_4BIT_D128[17] = {
    -1.0f, -0.209250037814f, -0.161320662130f, -0.126046838196f,
    -0.096558452011f, -0.070300404823f, -0.045961951676f, -0.022727041871f,
     0.0f,  0.022727041871f,  0.045961951676f,  0.070300404823f,
     0.096558452011f,  0.126046838196f,  0.161320662130f,  0.209250037814f, 1.0f,
};

constant float CODEBOOK_3BIT_D128[8] = {
    -0.188397319183f, -0.118139828402f, -0.066585638471f, -0.021604320011f,
     0.021604320011f,  0.066585638471f,  0.118139828402f,  0.188397319183f,
};

constant float BOUNDARIES_3BIT_D128[9] = {
    -1.0f, -0.153268573792f, -0.092362733436f, -0.044094979241f,
     0.0f,  0.044094979241f,  0.092362733436f,  0.153268573792f, 1.0f,
};

constant float CODEBOOK_2BIT_D128[4] = {
    -0.133041590561f, -0.039991612341f, 0.039991612341f, 0.133041590561f,
};

constant float BOUNDARIES_2BIT_D128[5] = {
    -1.0f, -0.086516601451f, 0.0f, 0.086516601451f, 1.0f,
};

// Pre-computed signs matching Rust generate_signs(42, 128)
constant float SIGNS_128[128] = {
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
// WHT butterfly in threadgroup memory
// ============================================================================

/**
 * In-place Walsh-Hadamard butterfly for 128 elements.
 * Uses SIMD shuffle for stages 1-5 (within a 32-lane SIMD group),
 * and threadgroup shared memory for stages 6-7 (cross-SIMD).
 */
inline void wht_butterfly_128(
    threadgroup float* data,
    uint tid,
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // 7 stages for dim=128: h = 1, 2, 4, 8, 16, 32, 64
    for (int h = 1; h < 128; h *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each thread handles one butterfly pair per stage
        for (int idx = (int)tid; idx < 64; idx += 128) {
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// Scalar quantization
// ============================================================================

inline uint8_t quantize_scalar_4bit(float x) {
    // Binary search on 17 boundaries for 16 levels
    if (x <= BOUNDARIES_4BIT_D128[1]) return 0;
    if (x >= BOUNDARIES_4BIT_D128[16]) return 15;
    uint lo = 1, hi = 16;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (x < BOUNDARIES_4BIT_D128[mid]) hi = mid;
        else lo = mid + 1;
    }
    return (uint8_t)(lo - 1);
}

inline uint8_t quantize_scalar_3bit(float x) {
    if (x <= BOUNDARIES_3BIT_D128[1]) return 0;
    if (x >= BOUNDARIES_3BIT_D128[8]) return 7;
    uint lo = 1, hi = 8;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (x < BOUNDARIES_3BIT_D128[mid]) hi = mid;
        else lo = mid + 1;
    }
    return (uint8_t)(lo - 1);
}

// ============================================================================
// Fused WHT + 4-bit quantize + pack kernel
// ============================================================================

kernel void turbo_quantize_4bit_d128(
    device const float* input   [[buffer(0)]],  // [batch, 128]
    device uint8_t*     packed  [[buffer(1)]],  // [batch, 64]
    device half*        norms   [[buffer(2)]],  // [batch]
    constant uint&      batch_size [[buffer(3)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (bid >= batch_size) return;
    constexpr int dim = 128;

    threadgroup float smem[128];

    // Load input
    if (tid < (uint)dim) {
        smem[tid] = input[bid * dim + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 norm via reduction
    threadgroup float norm_smem[128];
    norm_smem[tid] = (tid < (uint)dim) ? smem[tid] * smem[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < (uint)s) norm_smem[tid] += norm_smem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = sqrt(norm_smem[0]);

    if (tid == 0) norms[bid] = half(norm);

    // Normalize
    if (tid < (uint)dim && norm > 1e-10f) smem[tid] /= norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // D·H·D rotation
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    wht_butterfly_128(smem, tid, simd_lane, simd_group);
    constexpr float wht_scale = 0.08838834764831845f;
    if (tid < (uint)dim) smem[tid] *= wht_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize + nibble pack
    if (tid < (uint)(dim / 2)) {
        float v0 = smem[2 * tid];
        float v1 = smem[2 * tid + 1];
        uint8_t idx0 = quantize_scalar_4bit(v0);
        uint8_t idx1 = quantize_scalar_4bit(v1);
        packed[bid * (dim / 2) + tid] = (idx0 & 0xF) | ((idx1 & 0xF) << 4);
    }
}

// ============================================================================
// Fused WHT + 3-bit quantize + pack kernel
// ============================================================================

kernel void turbo_quantize_3bit_d128(
    device const float* input   [[buffer(0)]],
    device uint8_t*     packed  [[buffer(1)]],  // [batch, 52]
    device half*        norms   [[buffer(2)]],
    constant uint&      batch_size [[buffer(3)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (bid >= batch_size) return;
    constexpr int dim = 128;
    constexpr int n_groups = (dim + 9) / 10; // 13

    threadgroup float smem[128];
    threadgroup uint8_t indices[128];

    // Load
    if (tid < (uint)dim) smem[tid] = input[bid * dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 norm
    threadgroup float norm_smem[128];
    norm_smem[tid] = (tid < (uint)dim) ? smem[tid] * smem[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < (uint)s) norm_smem[tid] += norm_smem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = sqrt(norm_smem[0]);
    if (tid == 0) norms[bid] = half(norm);

    // Normalize
    if (tid < (uint)dim && norm > 1e-10f) smem[tid] /= norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // D·H·D rotation
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    wht_butterfly_128(smem, tid, simd_lane, simd_group);
    constexpr float wht_scale = 0.08838834764831845f;
    if (tid < (uint)dim) smem[tid] *= wht_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize
    if (tid < (uint)dim) {
        indices[tid] = quantize_scalar_3bit(smem[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 10-in-32 packing
    if (tid < (uint)n_groups) {
        int base = tid * 10;
        uint word = 0;
        int count = (base + 10 <= dim) ? 10 : (dim - base);
        for (int j = 0; j < count; j++) {
            word |= ((uint)indices[base + j] & 0x7) << (j * 3);
        }
        int out_offset = bid * n_groups * 4 + tid * 4;
        packed[out_offset + 0] = (uint8_t)(word);
        packed[out_offset + 1] = (uint8_t)(word >> 8);
        packed[out_offset + 2] = (uint8_t)(word >> 16);
        packed[out_offset + 3] = (uint8_t)(word >> 24);
    }
}

// ============================================================================
// Standalone WHT rotation kernel (for rotating queries during attention)
// ============================================================================

kernel void turbo_wht_rotate_d128(
    device float* data [[buffer(0)]],  // [batch, 128], in-place
    constant uint& batch_size [[buffer(1)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (bid >= batch_size) return;
    constexpr int dim = 128;

    threadgroup float smem[128];

    if (tid < (uint)dim) smem[tid] = data[bid * dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // D·H·D
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    wht_butterfly_128(smem, tid, simd_lane, simd_group);
    constexpr float wht_scale = 0.08838834764831845f;
    if (tid < (uint)dim) smem[tid] *= wht_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < (uint)dim) data[bid * dim + tid] = smem[tid];
}

// ============================================================================
// Dequantize 4-bit kernel
// ============================================================================

kernel void turbo_dequantize_4bit_d128(
    device const uint8_t* packed [[buffer(0)]],  // [batch, 64]
    device const half*    norms  [[buffer(1)]],  // [batch]
    device float*         output [[buffer(2)]],  // [batch, 128]
    constant uint&        batch_size [[buffer(3)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (bid >= batch_size) return;
    constexpr int dim = 128;

    threadgroup float smem[128];

    // Unpack + codebook lookup
    if (tid < (uint)(dim / 2)) {
        uint8_t byte = packed[bid * (dim / 2) + tid];
        smem[2 * tid]     = CODEBOOK_4BIT_D128[byte & 0xF];
        smem[2 * tid + 1] = CODEBOOK_4BIT_D128[(byte >> 4) & 0xF];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse WHT
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    wht_butterfly_128(smem, tid, simd_lane, simd_group);
    constexpr float wht_scale = 0.08838834764831845f;
    if (tid < (uint)dim) smem[tid] *= wht_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)dim) smem[tid] *= SIGNS_128[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scale by norm
    float norm = float(norms[bid]);
    if (tid < (uint)dim) {
        output[bid * dim + tid] = smem[tid] * norm;
    }
}
