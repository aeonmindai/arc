/**
 * TurboQuant Paged Attention Kernel
 *
 * A variant of the standard paged attention kernel that reads compressed KV cache
 * using TurboQuant codebook lookups instead of direct memory loads or FP8 dequantization.
 *
 * Key differences from standard paged attention:
 * 1. Q is pre-rotated via WHT before the Q·K loop (once per query)
 * 2. K is stored as packed indices; lookup centroids from codebook in shared memory
 * 3. K dot product operates in the rotated domain (no inverse rotation needed)
 * 4. V is stored as packed indices; dequantized inline via codebook lookup
 * 5. Norms are stored per-token and applied after the dot product
 *
 * This kernel handles the 4-bit key / 3-bit value (default) configuration.
 * The cache layout per block:
 *   k_cache: [num_blocks, num_kv_heads, packed_k_bytes_per_head, block_size]
 *   v_cache: [num_blocks, num_kv_heads, packed_v_bytes_per_head, block_size]
 *   k_norms: [num_blocks, num_kv_heads, block_size] (half)
 *   v_norms: [num_blocks, num_kv_heads, block_size] (half)
 *
 * Reference: TurboQuant ICLR 2026 (arXiv:2504.19874), 0xSero/turboquant Triton kernels.
 */

#include <stdint.h>
#include <cuda_fp16.h>
#include <float.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define MAX_TQ(a, b) ((a) > (b) ? (a) : (b))
#define MIN_TQ(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP_TQ(a, b) (((a) + (b) - 1) / (b))

// ============================================================================
// Codebooks (same values as turbo_wht.cu)
// ============================================================================

__constant__ float TQ_CODEBOOK_4BIT[16] = {
    -0.237664013127f, -0.180836062501f, -0.141805261760f, -0.110288414632f,
    -0.082828489390f, -0.057772320256f, -0.034151583096f, -0.011302500645f,
     0.011302500645f,  0.034151583096f,  0.057772320256f,  0.082828489390f,
     0.110288414632f,  0.141805261760f,  0.180836062501f,  0.237664013127f,
};

__constant__ float TQ_CODEBOOK_3BIT[8] = {
    -0.188397319183f, -0.118139828402f, -0.066585638471f, -0.021604320011f,
     0.021604320011f,  0.066585638471f,  0.118139828402f,  0.188397319183f,
};

__constant__ float TQ_CODEBOOK_2BIT[4] = {
    -0.133041590561f, -0.039991612341f, 0.039991612341f, 0.133041590561f,
};

// Sign array for WHT rotation (seed=42, dim=128)
__constant__ float TQ_SIGNS_128[128] = {
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
// WHT in shared memory (for query rotation)
// ============================================================================

__device__ void tq_wht_butterfly_128(float* data, int tid) {
    for (int h = 1; h < 128; h *= 2) {
        __syncthreads();
        for (int idx = tid; idx < 64; idx += 128) {
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

// Apply D·H·D rotation to a 128-dim vector in shared memory
__device__ void tq_rotate_128(float* data, int tid) {
    // D · x
    if (tid < 128) data[tid] *= TQ_SIGNS_128[tid];
    __syncthreads();
    // H · D · x
    tq_wht_butterfly_128(data, tid);
    // Scale by 1/sqrt(128)
    if (tid < 128) data[tid] *= 0.08838834764831845f;
    __syncthreads();
    // D · H · D · x
    if (tid < 128) data[tid] *= TQ_SIGNS_128[tid];
    __syncthreads();
}

// ============================================================================
// TurboQuant Paged Attention Kernel (simplified, HEAD_SIZE=128)
// ============================================================================

/**
 * Simplified TurboQuant attention for decode (single query token).
 *
 * This kernel processes one sequence per block, one head per block.
 * It computes attention over compressed KV cache using codebook lookups.
 *
 * Parameters:
 *   out:          [num_seqs, num_heads, HEAD_SIZE] output
 *   q:            [num_seqs, num_heads, HEAD_SIZE] query (not rotated)
 *   k_packed:     [total_tokens, num_kv_heads, k_packed_bytes] packed 4-bit K indices
 *   v_packed:     [total_tokens, num_kv_heads, v_packed_bytes] packed 3-bit V indices
 *   k_norms:      [total_tokens, num_kv_heads] half norms for K
 *   v_norms:      [total_tokens, num_kv_heads] half norms for V
 *   token_table:  [num_seqs, max_context_len] maps (seq, pos) → token index
 *   context_lens: [num_seqs] actual context length per sequence
 *
 * Grid:  (num_heads, num_seqs, 1)
 * Block: (128, 1, 1)
 */
template <int HEAD_SIZE, int K_BITS, int V_BITS>
__global__ void turbo_paged_attention_kernel(
    float*       __restrict__ out,           // [num_seqs, num_heads, HEAD_SIZE]
    const float* __restrict__ q,             // [num_seqs, num_heads, HEAD_SIZE]
    const uint8_t* __restrict__ k_packed,    // [total_tokens, num_kv_heads, k_bytes_per_head]
    const uint8_t* __restrict__ v_packed,    // [total_tokens, num_kv_heads, v_bytes_per_head]
    const __half*  __restrict__ k_norms,     // [total_tokens, num_kv_heads]
    const __half*  __restrict__ v_norms,     // [total_tokens, num_kv_heads]
    const int32_t* __restrict__ token_table, // [num_seqs, max_tokens_per_seq]
    const int32_t* __restrict__ context_lens,// [num_seqs]
    const int num_kv_heads,
    const int max_tokens_per_seq,
    const int num_heads,
    const float scale
) {
    static_assert(HEAD_SIZE == 128, "TurboQuant currently supports HEAD_SIZE=128 only");

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int context_len = context_lens[seq_idx];

    if (context_len == 0) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Packed sizes per head
    constexpr int K_BYTES = (K_BITS == 4) ? (HEAD_SIZE / 2) :
                            (K_BITS == 3) ? (((HEAD_SIZE + 9) / 10) * 4) :
                            (HEAD_SIZE / 4);
    constexpr int V_BYTES = (V_BITS == 4) ? (HEAD_SIZE / 2) :
                            (V_BITS == 3) ? (((HEAD_SIZE + 9) / 10) * 4) :
                            (HEAD_SIZE / 4);

    // Select codebooks
    const float* k_codebook = (K_BITS == 4) ? TQ_CODEBOOK_4BIT :
                              (K_BITS == 3) ? TQ_CODEBOOK_3BIT :
                              TQ_CODEBOOK_2BIT;
    const float* v_codebook = (V_BITS == 4) ? TQ_CODEBOOK_4BIT :
                              (V_BITS == 3) ? TQ_CODEBOOK_3BIT :
                              TQ_CODEBOOK_2BIT;

    // =========================================================
    // Step 1: Load and rotate query
    // =========================================================
    __shared__ float q_rotated[128];
    const float* q_ptr = q + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    if (tid < HEAD_SIZE) {
        q_rotated[tid] = q_ptr[tid];
    }
    __syncthreads();

    // Apply D·H·D rotation to the query
    tq_rotate_128(q_rotated, tid);

    // =========================================================
    // Step 2: Compute Q·K attention scores
    // =========================================================
    // We process one token at a time. Each thread handles one dimension.
    // This is simpler than the vectorized paged attention but correct.

    extern __shared__ char shared_mem[];
    float* logits = reinterpret_cast<float*>(shared_mem);
    float* reduce_buf = reinterpret_cast<float*>(shared_mem + context_len * sizeof(float));

    float qk_max = -FLT_MAX;

    for (int t = 0; t < context_len; t++) {
        int token_idx = token_table[seq_idx * max_tokens_per_seq + t];
        const uint8_t* k_ptr = k_packed + (token_idx * num_kv_heads + kv_head_idx) * K_BYTES;
        float k_norm = __half2float(k_norms[token_idx * num_kv_heads + kv_head_idx]);

        // Each thread loads and dequantizes one element of the key
        float k_val = 0.0f;
        if (tid < HEAD_SIZE) {
            uint8_t idx;
            if constexpr (K_BITS == 4) {
                // Nibble packing: 2 per byte
                uint8_t byte = k_ptr[tid / 2];
                idx = (tid & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            } else if constexpr (K_BITS == 3) {
                // 10-in-32 packing
                int group = tid / 10;
                int pos_in_group = tid % 10;
                uint32_t word = *reinterpret_cast<const uint32_t*>(k_ptr + group * 4);
                idx = (word >> (pos_in_group * 3)) & 0x7;
            } else {
                // 2-bit: 4 per byte
                uint8_t byte = k_ptr[tid / 4];
                int shift = 6 - (tid % 4) * 2;
                idx = (byte >> shift) & 0x3;
            }
            k_val = k_codebook[idx];
        }

        // Dot product: q_rotated · k_centroids (all in rotated domain)
        // Parallel reduction across 128 threads
        float partial = (tid < HEAD_SIZE) ? q_rotated[tid] * k_val : 0.0f;

        // Warp-level reduction
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, mask);
        }

        // Cross-warp reduction via shared memory
        __shared__ float warp_sums[4]; // 128 threads / 32 = 4 warps
        if (tid % WARP_SIZE == 0) {
            warp_sums[tid / WARP_SIZE] = partial;
        }
        __syncthreads();

        float qk = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < 4; w++) qk += warp_sums[w];
            qk = qk * k_norm * scale;
            logits[t] = qk;
            qk_max = fmaxf(qk_max, qk);
        }
        __syncthreads();
    }

    // Broadcast qk_max to all threads
    if (tid == 0) {
        reduce_buf[0] = qk_max;
    }
    __syncthreads();
    qk_max = reduce_buf[0];

    // =========================================================
    // Step 3: Softmax
    // =========================================================
    float exp_sum = 0.0f;
    for (int t = tid; t < context_len; t += blockDim.x) {
        float val = __expf(logits[t] - qk_max);
        logits[t] = val;
        exp_sum += val;
    }
    // Reduce exp_sum across threads
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, mask);
    }
    __shared__ float warp_exp_sums[4];
    if (tid % WARP_SIZE == 0) {
        warp_exp_sums[tid / WARP_SIZE] = exp_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < 4; w++) total += warp_exp_sums[w];
        reduce_buf[0] = __fdividef(1.0f, total + 1e-6f);
    }
    __syncthreads();
    float inv_sum = reduce_buf[0];

    for (int t = tid; t < context_len; t += blockDim.x) {
        logits[t] *= inv_sum;
    }
    __syncthreads();

    // =========================================================
    // Step 4: Weighted sum of values (attention · V)
    // =========================================================
    // Each thread accumulates its dimension across all tokens
    float acc = 0.0f;

    if (tid < HEAD_SIZE) {
        for (int t = 0; t < context_len; t++) {
            float weight = logits[t];
            if (weight < 1e-8f) continue; // Skip near-zero weights

            int token_idx = token_table[seq_idx * max_tokens_per_seq + t];
            const uint8_t* v_ptr = v_packed + (token_idx * num_kv_heads + kv_head_idx) * V_BYTES;
            float v_norm = __half2float(v_norms[token_idx * num_kv_heads + kv_head_idx]);

            // Dequantize value element
            uint8_t idx;
            if constexpr (V_BITS == 4) {
                uint8_t byte = v_ptr[tid / 2];
                idx = (tid & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            } else if constexpr (V_BITS == 3) {
                int group = tid / 10;
                int pos_in_group = tid % 10;
                uint32_t word = *reinterpret_cast<const uint32_t*>(v_ptr + group * 4);
                idx = (word >> (pos_in_group * 3)) & 0x7;
            } else {
                uint8_t byte = v_ptr[tid / 4];
                int shift = 6 - (tid % 4) * 2;
                idx = (byte >> shift) & 0x3;
            }

            float v_val = v_codebook[idx] * v_norm;
            acc += weight * v_val;
        }
    }

    // =========================================================
    // Step 5: Inverse rotation and write output
    // =========================================================
    // V was compressed in the rotated domain, so the accumulated output
    // is also in the rotated domain. Apply inverse rotation.
    __shared__ float out_rotated[128];
    if (tid < HEAD_SIZE) {
        out_rotated[tid] = acc;
    }
    __syncthreads();

    // D·H·D is self-inverse
    tq_rotate_128(out_rotated, tid);

    // Write output
    float* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    if (tid < HEAD_SIZE) {
        out_ptr[tid] = out_rotated[tid];
    }
}

// ============================================================================
// Launch wrapper
// ============================================================================

inline void launch_turbo_paged_attention(
    float* out,
    const float* q,
    const uint8_t* k_packed,
    const uint8_t* v_packed,
    const __half* k_norms,
    const __half* v_norms,
    const int32_t* token_table,
    const int32_t* context_lens,
    int num_kv_heads,
    int max_tokens_per_seq,
    int num_seqs,
    int num_heads,
    int head_size,
    float scale,
    int k_bits,
    int v_bits,
    cudaStream_t stream
) {
    // Shared memory: context_len * sizeof(float) for logits + scratch
    // Use max_tokens_per_seq as upper bound
    int shared_mem_size = (max_tokens_per_seq + 1) * sizeof(float);

    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(128); // One thread per head dimension

    // Dispatch based on bit-width combination
    if (head_size == 128) {
        if (k_bits == 4 && v_bits == 3) {
            turbo_paged_attention_kernel<128, 4, 3><<<grid, block, shared_mem_size, stream>>>(
                out, q, k_packed, v_packed, k_norms, v_norms,
                token_table, context_lens, num_kv_heads, max_tokens_per_seq,
                num_heads, scale);
        } else if (k_bits == 3 && v_bits == 3) {
            turbo_paged_attention_kernel<128, 3, 3><<<grid, block, shared_mem_size, stream>>>(
                out, q, k_packed, v_packed, k_norms, v_norms,
                token_table, context_lens, num_kv_heads, max_tokens_per_seq,
                num_heads, scale);
        } else if (k_bits == 3 && v_bits == 2) {
            turbo_paged_attention_kernel<128, 3, 2><<<grid, block, shared_mem_size, stream>>>(
                out, q, k_packed, v_packed, k_norms, v_norms,
                token_table, context_lens, num_kv_heads, max_tokens_per_seq,
                num_heads, scale);
        }
    }
}
