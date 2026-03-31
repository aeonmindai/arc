/**
 * TurboQuant Paged Attention Metal Kernel
 *
 * Metal equivalent of turbo_paged_attention.cuh.
 * Computes attention over TurboQuant-compressed KV cache using codebook lookups.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Codebooks
// ============================================================================

constant float TQ_CODEBOOK_4BIT[16] = {
    -0.237664013127f, -0.180836062501f, -0.141805261760f, -0.110288414632f,
    -0.082828489390f, -0.057772320256f, -0.034151583096f, -0.011302500645f,
     0.011302500645f,  0.034151583096f,  0.057772320256f,  0.082828489390f,
     0.110288414632f,  0.141805261760f,  0.180836062501f,  0.237664013127f,
};

constant float TQ_CODEBOOK_3BIT[8] = {
    -0.188397319183f, -0.118139828402f, -0.066585638471f, -0.021604320011f,
     0.021604320011f,  0.066585638471f,  0.118139828402f,  0.188397319183f,
};

constant float TQ_CODEBOOK_2BIT[4] = {
    -0.133041590561f, -0.039991612341f, 0.039991612341f, 0.133041590561f,
};

constant float TQ_SIGNS_128[128] = {
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
// WHT
// ============================================================================

inline void tq_wht_butterfly_128(threadgroup float* data, uint tid) {
    for (int h = 1; h < 128; h *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

inline void tq_rotate_128(threadgroup float* data, uint tid) {
    if (tid < 128) data[tid] *= TQ_SIGNS_128[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_wht_butterfly_128(data, tid);
    if (tid < 128) data[tid] *= 0.08838834764831845f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) data[tid] *= TQ_SIGNS_128[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// Dequantize helpers
// ============================================================================

inline float dequant_4bit(device const uint8_t* packed, uint dim_idx) {
    uint8_t byte = packed[dim_idx / 2];
    uint8_t idx = (dim_idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    return TQ_CODEBOOK_4BIT[idx];
}

inline float dequant_3bit(device const uint8_t* packed, uint dim_idx) {
    uint group = dim_idx / 10;
    uint pos = dim_idx % 10;
    uint word = *reinterpret_cast<device const uint32_t*>(packed + group * 4);
    uint8_t idx = (word >> (pos * 3)) & 0x7;
    return TQ_CODEBOOK_3BIT[idx];
}

inline float dequant_2bit(device const uint8_t* packed, uint dim_idx) {
    uint8_t byte = packed[dim_idx / 4];
    uint shift = 6 - (dim_idx % 4) * 2;
    uint8_t idx = (byte >> shift) & 0x3;
    return TQ_CODEBOOK_2BIT[idx];
}

// ============================================================================
// TurboQuant Paged Attention (K4/V3 variant, HEAD_SIZE=128)
// ============================================================================

kernel void turbo_paged_attention_k4v3_d128(
    device float*         out          [[buffer(0)]],  // [num_seqs, num_heads, 128]
    device const float*   q            [[buffer(1)]],  // [num_seqs, num_heads, 128]
    device const uint8_t* k_packed     [[buffer(2)]],  // [total_tokens, num_kv_heads, 64]
    device const uint8_t* v_packed     [[buffer(3)]],  // [total_tokens, num_kv_heads, 52]
    device const half*    k_norms      [[buffer(4)]],  // [total_tokens, num_kv_heads]
    device const half*    v_norms      [[buffer(5)]],  // [total_tokens, num_kv_heads]
    device const int*     token_table  [[buffer(6)]],  // [num_seqs, max_tokens]
    device const int*     context_lens [[buffer(7)]],  // [num_seqs]
    constant uint&        num_kv_heads     [[buffer(8)]],
    constant uint&        max_tokens       [[buffer(9)]],
    constant uint&        num_heads_val    [[buffer(10)]],
    constant float&       scale            [[buffer(11)]],
    uint bid_head  [[threadgroup_position_in_grid]],    // flattened: head_idx + seq_idx * num_heads
    uint tid       [[thread_index_in_threadgroup]]
) {
    constexpr uint HEAD_SIZE = 128;
    constexpr uint K_BYTES = 64;   // 128 * 4bit / 8
    constexpr uint V_BYTES = 52;   // ceil(128/10)*4

    uint num_heads = num_heads_val;
    uint head_idx = bid_head % num_heads;
    uint seq_idx = bid_head / num_heads;
    int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Step 1: Load and rotate query
    threadgroup float q_rotated[128];
    if (tid < HEAD_SIZE) {
        q_rotated[tid] = q[seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_rotate_128(q_rotated, tid);

    // Step 2: Compute Q·K scores
    // Process one token at a time (simple path)
    threadgroup float logits[8192]; // Max context length supported in shared memory
    float qk_max = -FLT_MAX;

    for (int t = 0; t < context_len && t < 8192; t++) {
        int token_idx = token_table[seq_idx * max_tokens + t];

        // Each thread dequantizes one K element and computes partial dot product
        float partial = 0.0f;
        if (tid < HEAD_SIZE) {
            device const uint8_t* k_ptr = k_packed + (token_idx * num_kv_heads + kv_head_idx) * K_BYTES;
            float k_val = dequant_4bit(k_ptr, tid);
            partial = q_rotated[tid] * k_val;
        }

        // Reduce across SIMD group (32 threads)
        partial = simd_sum(partial);

        // Cross-SIMD reduction via threadgroup memory
        threadgroup float warp_sums[4];
        if (tid % 32 == 0) warp_sums[tid / 32] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float qk = 0.0f;
            for (uint w = 0; w < 4; w++) qk += warp_sums[w];
            float k_norm = float(k_norms[token_idx * num_kv_heads + kv_head_idx]);
            qk = qk * k_norm * scale;
            logits[t] = qk;
            qk_max = max(qk_max, qk);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Broadcast qk_max
    threadgroup float shared_scalar[1];
    if (tid == 0) shared_scalar[0] = qk_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qk_max = shared_scalar[0];

    // Step 3: Softmax
    float exp_sum = 0.0f;
    for (int t = (int)tid; t < context_len; t += 128) {
        float val = exp(logits[t] - qk_max);
        logits[t] = val;
        exp_sum += val;
    }
    exp_sum = simd_sum(exp_sum);
    threadgroup float warp_exp[4];
    if (tid % 32 == 0) warp_exp[tid / 32] = exp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint w = 0; w < 4; w++) total += warp_exp[w];
        shared_scalar[0] = 1.0f / (total + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = shared_scalar[0];

    for (int t = (int)tid; t < context_len; t += 128) {
        logits[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Weighted sum of values
    float acc = 0.0f;
    if (tid < HEAD_SIZE) {
        for (int t = 0; t < context_len; t++) {
            float weight = logits[t];
            if (weight < 1e-8f) continue;

            int token_idx = token_table[seq_idx * max_tokens + t];
            device const uint8_t* v_ptr = v_packed + (token_idx * num_kv_heads + kv_head_idx) * V_BYTES;
            float v_norm = float(v_norms[token_idx * num_kv_heads + kv_head_idx]);
            float v_val = dequant_3bit(v_ptr, tid) * v_norm;
            acc += weight * v_val;
        }
    }

    // Step 5: Inverse rotation and write output
    threadgroup float out_rotated[128];
    if (tid < HEAD_SIZE) out_rotated[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_rotate_128(out_rotated, tid);

    if (tid < HEAD_SIZE) {
        out[seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + tid] = out_rotated[tid];
    }
}
