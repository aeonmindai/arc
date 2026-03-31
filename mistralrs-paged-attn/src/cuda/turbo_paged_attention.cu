/**
 * TurboQuant Paged Attention CUDA Kernels
 *
 * Complete write + read path for TurboQuant compressed KV cache.
 *
 * Write (turbo_reshape_and_cache):
 *   FP16 K/V → normalize → WHT rotate → codebook quantize → store U8 indices + F16 norms
 *
 * Read (turbo_paged_attention_v1_f16):
 *   For each cached token: load U8 index → codebook lookup → dot with rotated Q
 *   No full dequantization. Codebook in shared memory.
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

// Include the kernel definitions from the header
#include "turbo_paged_attention.cuh"

// ============================================================================
// Codebook + Signs (must match turbo_wht.cu and Rust codebook.rs)
// ============================================================================

static __constant__ float TQ_CB_4BIT[16] = {
    -0.237664013127f, -0.180836062501f, -0.141805261760f, -0.110288414632f,
    -0.082828489390f, -0.057772320256f, -0.034151583096f, -0.011302500645f,
     0.011302500645f,  0.034151583096f,  0.057772320256f,  0.082828489390f,
     0.110288414632f,  0.141805261760f,  0.180836062501f,  0.237664013127f,
};

static __constant__ float TQ_BD_4BIT[17] = {
    -1.0f, -0.209250037814f, -0.161320662130f, -0.126046838196f,
    -0.096558452011f, -0.070300404823f, -0.045961951676f, -0.022727041871f,
     0.0f,  0.022727041871f,  0.045961951676f,  0.070300404823f,
     0.096558452011f,  0.126046838196f,  0.161320662130f,  0.209250037814f, 1.0f,
};

static __constant__ float TQ_CB_3BIT[8] = {
    -0.188397319183f, -0.118139828402f, -0.066585638471f, -0.021604320011f,
     0.021604320011f,  0.066585638471f,  0.118139828402f,  0.188397319183f,
};

static __constant__ float TQ_BD_3BIT[9] = {
    -1.0f, -0.153268573792f, -0.092362733436f, -0.044094979241f,
     0.0f,  0.044094979241f,  0.092362733436f,  0.153268573792f, 1.0f,
};

static __constant__ float TQ_SIGNS[128] = {
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
// Helpers
// ============================================================================

__device__ void tq_wht_128(float* data, int tid) {
    for (int h = 1; h < 128; h *= 2) {
        __syncthreads();
        for (int idx = tid; idx < 64; idx += 128) {
            int bs = (idx / h) * (h * 2);
            int off = idx % h;
            float a = data[bs + off];
            float b = data[bs + off + h];
            data[bs + off] = a + b;
            data[bs + off + h] = a - b;
        }
    }
    __syncthreads();
}

__device__ void tq_rotate(float* data, int tid) {
    if (tid < 128) data[tid] *= TQ_SIGNS[tid];
    __syncthreads();
    tq_wht_128(data, tid);
    if (tid < 128) data[tid] *= 0.08838834764831845f;
    __syncthreads();
    if (tid < 128) data[tid] *= TQ_SIGNS[tid];
    __syncthreads();
}

__device__ uint8_t tq_quantize_4bit(float x) {
    if (x <= TQ_BD_4BIT[1]) return 0;
    if (x >= TQ_BD_4BIT[16]) return 15;
    int lo = 1, hi = 16;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (x < TQ_BD_4BIT[mid]) hi = mid;
        else lo = mid + 1;
    }
    return (uint8_t)(lo - 1);
}

__device__ uint8_t tq_quantize_3bit(float x) {
    if (x <= TQ_BD_3BIT[1]) return 0;
    if (x >= TQ_BD_3BIT[8]) return 7;
    int lo = 1, hi = 8;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (x < TQ_BD_3BIT[mid]) hi = mid;
        else lo = mid + 1;
    }
    return (uint8_t)(lo - 1);
}

// ============================================================================
// turbo_reshape_and_cache: Quantize FP16 K/V and store as U8 in paged cache
// ============================================================================

/**
 * Quantize one vector (K or V) and store as U8 indices.
 *
 * Grid:  (num_tokens * num_heads, 1, 1)
 * Block: (128, 1, 1)
 *
 * Each block handles one vector (one head of one token).
 *
 * key/value: [num_tokens, num_heads, head_size] in FP16
 * cache:     [num_blocks, num_heads, head_size/x, block_size, x] in U8
 * norms:     [num_blocks, num_heads, block_size] in FP16
 * slot_mapping: [num_tokens] -> slot index in cache
 */
template<bool IS_KEY>
__global__ void turbo_reshape_and_cache_kernel(
    const __half* __restrict__ input,  // [num_tokens, num_heads, 128]
    uint8_t* __restrict__ cache,       // [num_blocks, num_heads, 8, block_size, 16] for U8
    __half* __restrict__ norms,        // [num_blocks, num_heads, block_size]
    const int64_t* __restrict__ slot_mapping,
    int num_heads,
    int head_size,
    int block_size,
    int input_stride,                  // stride for token dim in input
    int cache_block_stride,            // stride for block dim in cache
    int cache_head_stride,             // stride for head dim in cache
    int norm_block_stride,             // stride for block dim in norms
    int norm_head_stride               // stride for head dim in norms
) {
    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int token_idx = vec_idx / num_heads;
    const int head_idx = vec_idx % num_heads;

    // Check slot mapping
    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return; // PAD token

    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    __shared__ float smem[128];

    // Step 1: Load FP16 input to shared memory as F32
    if (tid < head_size) {
        smem[tid] = __half2float(input[token_idx * input_stride + head_idx * head_size + tid]);
    }
    __syncthreads();

    // Step 2: Compute L2 norm
    __shared__ float norm_buf[128];
    norm_buf[tid] = (tid < head_size) ? smem[tid] * smem[tid] : 0.0f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) norm_buf[tid] += norm_buf[tid + s];
        __syncthreads();
    }
    float norm = sqrtf(norm_buf[0]);

    // Store norm
    if (tid == 0) {
        norms[block_idx * norm_block_stride + head_idx * norm_head_stride + block_offset] =
            __float2half(norm);
    }

    // Step 3: Normalize
    if (tid < head_size && norm > 1e-10f) {
        smem[tid] /= norm;
    }
    __syncthreads();

    // Step 4: WHT rotate (D·H·D)
    tq_rotate(smem, tid);

    // Step 5: Quantize and store as U8
    // Cache layout for U8: [num_blocks, num_heads, head_size/16, block_size, 16]
    // Each element is one U8 index. We store one index per coordinate.
    if (tid < head_size) {
        uint8_t idx;
        if constexpr (IS_KEY) {
            idx = tq_quantize_4bit(smem[tid]);
        } else {
            idx = tq_quantize_3bit(smem[tid]);
        }

        // Compute cache offset: same layout as standard paged attention U8 cache
        // [block_idx, head_idx, tid/16, block_offset, tid%16]
        const int x = 16; // U8 element size = 1 byte, x = 16/1 = 16
        int d0 = tid / x;
        int d1 = tid % x;
        int offset = block_idx * cache_block_stride +
                     head_idx * cache_head_stride +
                     d0 * block_size * x +
                     block_offset * x +
                     d1;
        cache[offset] = idx;
    }
}

// Launch wrapper
extern "C" void turbo_reshape_and_cache(
    const void* key,
    const void* value,
    void* key_cache,
    void* value_cache,
    void* k_norms,
    void* v_norms,
    const int64_t* slot_mapping,
    int num_tokens,
    int num_heads,
    int head_size,
    int block_size,
    int key_stride,
    int value_stride,
    int kv_block_stride,
    int kv_head_stride,
    int norm_block_stride,
    int norm_head_stride,
    cudaStream_t stream,
    uint32_t dtype  // 0=f16, 1=bf16
) {
    if (head_size != 128) return; // Only support head_size=128 for now

    int total_vectors = num_tokens * num_heads;
    dim3 grid(total_vectors);
    dim3 block(128);

    // Quantize keys (4-bit)
    turbo_reshape_and_cache_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(key),
        reinterpret_cast<uint8_t*>(key_cache),
        reinterpret_cast<__half*>(k_norms),
        slot_mapping,
        num_heads, head_size, block_size,
        key_stride,
        kv_block_stride, kv_head_stride,
        norm_block_stride, norm_head_stride
    );

    // Quantize values (3-bit stored as U8 indices — one per coordinate)
    turbo_reshape_and_cache_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(value),
        reinterpret_cast<uint8_t*>(value_cache),
        reinterpret_cast<__half*>(v_norms),
        slot_mapping,
        num_heads, head_size, block_size,
        value_stride,
        kv_block_stride, kv_head_stride,
        norm_block_stride, norm_head_stride
    );
}

// ============================================================================
// turbo_paged_attention_v1: Attention over TurboQuant compressed KV cache
// ============================================================================

/**
 * Decode attention over TurboQuant U8 cache.
 *
 * Grid:  (num_heads, num_seqs, 1)
 * Block: (128, 1, 1)  — one thread per head dimension
 *
 * For each cached token:
 *   - Load U8 index from K cache
 *   - Lookup codebook centroid
 *   - Dot product with rotated Q (in shared memory)
 *   - Scale by stored norm
 *
 * Then softmax + weighted V aggregation (dequantize V inline).
 */
__global__ void turbo_attn_kernel_d128(
    float* __restrict__ out,                    // [num_seqs, num_heads, 128]
    const __half* __restrict__ query,           // [num_seqs, num_heads, 128]
    const uint8_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, 8, block_size, 16]
    const uint8_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, 128, block_size]
    const __half* __restrict__ k_norms,         // [num_blocks, num_kv_heads, block_size]
    const __half* __restrict__ v_norms,         // [num_blocks, num_kv_heads, block_size]
    const uint32_t* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const uint32_t* __restrict__ context_lens,  // [num_seqs]
    int num_kv_heads,
    int max_blocks_per_seq,
    int num_heads,
    int block_size,
    float scale,
    int kv_block_stride,
    int kv_head_stride,
    int norm_block_stride,
    int norm_head_stride
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Load and rotate query into shared memory
    __shared__ float q_rot[128];
    if (tid < 128) {
        q_rot[tid] = __half2float(query[seq_idx * num_heads * 128 + head_idx * 128 + tid]);
    }
    __syncthreads();
    tq_rotate(q_rot, tid);

    // Compute Q·K scores for all cached tokens
    // Use dynamic shared memory for logits
    extern __shared__ char smem_bytes[];
    float* logits = reinterpret_cast<float*>(smem_bytes);

    float qk_max = -FLT_MAX;

    const uint32_t* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;
    int num_blocks = (context_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        int physical_block = seq_block_table[b];
        int tokens_in_block = min(block_size, context_len - b * block_size);

        for (int t = 0; t < tokens_in_block; t++) {
            int token_pos = b * block_size + t;

            // Each thread loads one K element (U8 index) and looks up codebook
            float k_val = 0.0f;
            if (tid < 128) {
                int x = 16;
                int d0 = tid / x;
                int d1 = tid % x;
                int k_offset = physical_block * kv_block_stride +
                               kv_head_idx * kv_head_stride +
                               d0 * block_size * x +
                               t * x +
                               d1;
                uint8_t k_idx = k_cache[k_offset];
                k_val = TQ_CB_4BIT[k_idx];
            }

            // Dot product: q_rot · k_codebook (parallel reduction)
            float partial = (tid < 128) ? q_rot[tid] * k_val : 0.0f;

            // Warp reduction
            for (int mask = 16; mask > 0; mask >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, mask);
            }

            // Cross-warp reduction
            __shared__ float warp_sums[4];
            if (tid % 32 == 0) warp_sums[tid / 32] = partial;
            __syncthreads();

            if (tid == 0) {
                float qk = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
                float k_norm = __half2float(
                    k_norms[physical_block * norm_block_stride +
                            kv_head_idx * norm_head_stride + t]);
                qk = qk * k_norm * scale;
                logits[token_pos] = qk;
                qk_max = fmaxf(qk_max, qk);
            }
            __syncthreads();
        }
    }

    // Broadcast qk_max
    __shared__ float shared_val[1];
    if (tid == 0) shared_val[0] = qk_max;
    __syncthreads();
    qk_max = shared_val[0];

    // Softmax
    float exp_sum = 0.0f;
    for (int i = tid; i < context_len; i += 128) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, mask);
    __shared__ float warp_exp[4];
    if (tid % 32 == 0) warp_exp[tid / 32] = exp_sum;
    __syncthreads();
    if (tid == 0) {
        float total = warp_exp[0] + warp_exp[1] + warp_exp[2] + warp_exp[3];
        shared_val[0] = __fdividef(1.0f, total + 1e-6f);
    }
    __syncthreads();
    float inv_sum = shared_val[0];
    for (int i = tid; i < context_len; i += 128)
        logits[i] *= inv_sum;
    __syncthreads();

    // Weighted sum of V
    float acc = 0.0f;
    if (tid < 128) {
        for (int b = 0; b < num_blocks; b++) {
            int physical_block = seq_block_table[b];
            int tokens_in_block = min(block_size, context_len - b * block_size);

            for (int t = 0; t < tokens_in_block; t++) {
                int token_pos = b * block_size + t;
                float weight = logits[token_pos];
                if (weight < 1e-8f) continue;

                // V cache: [num_blocks, num_kv_heads, head_size, block_size] in U8
                int v_offset = physical_block * kv_block_stride +
                               kv_head_idx * kv_head_stride +
                               tid * block_size +
                               t;
                uint8_t v_idx = v_cache[v_offset];
                float v_val = TQ_CB_3BIT[v_idx];
                float v_norm = __half2float(
                    v_norms[physical_block * norm_block_stride +
                            kv_head_idx * norm_head_stride + t]);
                acc += weight * v_val * v_norm;
            }
        }
    }

    // Inverse rotation and write output
    __shared__ float out_rot[128];
    if (tid < 128) out_rot[tid] = acc;
    __syncthreads();
    tq_rotate(out_rot, tid);

    if (tid < 128) {
        out[seq_idx * num_heads * 128 + head_idx * 128 + tid] = out_rot[tid];
    }
}

extern "C" void turbo_paged_attention_v1_f16(
    void* out,
    const void* query,
    const void* k_cache,
    const void* v_cache,
    const void* k_norms,
    const void* v_norms,
    int num_kv_heads,
    float scale,
    float softcapping,
    const uint32_t* block_tables,
    const uint32_t* context_lens,
    int block_size,
    int max_context_len,
    int num_seqs,
    int num_heads,
    int head_size,
    int max_num_blocks_per_seq,
    int q_stride,
    int kv_block_stride,
    int kv_head_stride,
    int norm_block_stride,
    int norm_head_stride,
    cudaStream_t stream
) {
    if (head_size != 128) return;

    int shared_mem = max_context_len * sizeof(float);
    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(128);

    turbo_attn_kernel_d128<<<grid, block, shared_mem, stream>>>(
        reinterpret_cast<float*>(out),
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const uint8_t*>(k_cache),
        reinterpret_cast<const uint8_t*>(v_cache),
        reinterpret_cast<const __half*>(k_norms),
        reinterpret_cast<const __half*>(v_norms),
        block_tables,
        context_lens,
        num_kv_heads,
        max_num_blocks_per_seq,
        num_heads,
        block_size,
        scale,
        kv_block_stride,
        kv_head_stride,
        norm_block_stride,
        norm_head_stride
    );
}
