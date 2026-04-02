/**
 * TurboQuant Paged Attention CUDA Kernels — Full Sub-Byte Packing
 *
 * K cache: 4-bit nibble packed (2 values per byte, 64 bytes/head for d=128)
 * V cache: 3-bit packed (10 values per u32, 52 bytes/head for d=128)
 *
 * Write: FP16 → WHT → codebook quantize → bit-pack → store
 * Read:  Load packed → bit-unpack → codebook lookup → dot with rotated Q
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

#include "turbo_paged_attention.cuh"

// ============================================================================
// Codebooks + Signs (same as before)
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
    while (lo < hi) { int mid = (lo + hi) >> 1; if (x < TQ_BD_4BIT[mid]) hi = mid; else lo = mid + 1; }
    return (uint8_t)(lo - 1);
}

__device__ uint8_t tq_quantize_3bit(float x) {
    if (x <= TQ_BD_3BIT[1]) return 0;
    if (x >= TQ_BD_3BIT[8]) return 7;
    int lo = 1, hi = 8;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (x < TQ_BD_3BIT[mid]) hi = mid; else lo = mid + 1; }
    return (uint8_t)(lo - 1);
}

// ============================================================================
// reshape_and_cache: Quantize, pack bits, store
// ============================================================================

// K kernel: quantize to 4-bit, nibble-pack (2 per byte), store 64 bytes/head
__global__ void turbo_reshape_and_cache_k(
    const __half* __restrict__ input,   // [num_tokens, num_heads, 128]
    uint8_t* __restrict__ cache,        // [num_blocks, num_heads, 4, block_size, 16] packed
    __half* __restrict__ norms,         // [num_blocks, num_heads, block_size]
    const int64_t* __restrict__ slot_mapping,
    int num_heads, int head_size, int block_size,
    int input_stride,
    int cache_block_stride, int cache_head_stride,
    int norm_block_stride, int norm_head_stride
) {
    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_idx = vec_idx / num_heads;
    const int head_idx = vec_idx % num_heads;
    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    __shared__ float smem[128];
    __shared__ uint8_t indices[128];

    // Load, normalize, rotate
    if (tid < head_size)
        smem[tid] = __half2float(input[token_idx * input_stride + head_idx * head_size + tid]);
    __syncthreads();

    // L2 norm
    __shared__ float nbuf[128];
    nbuf[tid] = (tid < head_size) ? smem[tid] * smem[tid] : 0.0f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (tid < s) nbuf[tid] += nbuf[tid + s]; __syncthreads(); }
    float norm = sqrtf(nbuf[0]);
    if (tid == 0)
        norms[block_idx * norm_block_stride + head_idx * norm_head_stride + block_offset] = __float2half(norm);
    if (tid < head_size && norm > 1e-10f) smem[tid] /= norm;
    __syncthreads();

    tq_rotate(smem, tid);

    // Quantize
    if (tid < head_size) indices[tid] = tq_quantize_4bit(smem[tid]);
    __syncthreads();

    // Nibble pack: thread tid handles pair (2*tid, 2*tid+1) → 1 byte
    // Packed K cache: [block, head, packed_byte/16, block_offset, packed_byte%16]
    // packed_bytes = head_size/2 = 64, stored in 5D with x=16
    if (tid < head_size / 2) {
        uint8_t lo = indices[2 * tid] & 0xF;
        uint8_t hi = indices[2 * tid + 1] & 0xF;
        uint8_t packed = lo | (hi << 4);

        int x = 16;
        int byte_idx = tid; // 0..63
        int offset = block_idx * cache_block_stride +
                     head_idx * cache_head_stride +
                     (byte_idx / x) * block_size * x +
                     block_offset * x +
                     (byte_idx % x);
        cache[offset] = packed;
    }
}

// V kernel: quantize to 3-bit, pack 10-in-32, store 52 bytes/head
__global__ void turbo_reshape_and_cache_v(
    const __half* __restrict__ input,   // [num_tokens, num_heads, 128]
    uint8_t* __restrict__ cache,        // [num_blocks, num_heads, 52, block_size] packed
    __half* __restrict__ norms,         // [num_blocks, num_heads, block_size]
    const int64_t* __restrict__ slot_mapping,
    int num_heads, int head_size, int block_size,
    int input_stride,
    int v_block_stride, int v_head_stride,
    int norm_block_stride, int norm_head_stride
) {
    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_idx = vec_idx / num_heads;
    const int head_idx = vec_idx % num_heads;
    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    __shared__ float smem[128];
    __shared__ uint8_t indices[128];

    if (tid < head_size)
        smem[tid] = __half2float(input[token_idx * input_stride + head_idx * head_size + tid]);
    __syncthreads();

    __shared__ float nbuf[128];
    nbuf[tid] = (tid < head_size) ? smem[tid] * smem[tid] : 0.0f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (tid < s) nbuf[tid] += nbuf[tid + s]; __syncthreads(); }
    float norm = sqrtf(nbuf[0]);
    if (tid == 0)
        norms[block_idx * norm_block_stride + head_idx * norm_head_stride + block_offset] = __float2half(norm);
    if (tid < head_size && norm > 1e-10f) smem[tid] /= norm;
    __syncthreads();

    tq_rotate(smem, tid);

    if (tid < head_size) indices[tid] = tq_quantize_3bit(smem[tid]);
    __syncthreads();

    // 10-in-32 packing: 13 groups of 10 (last group has 8 values for d=128)
    // Each group → 4 bytes. Total = 13*4 = 52 bytes.
    // Thread 0..12 each pack one group.
    const int n_groups = (head_size + 9) / 10; // 13
    if (tid < n_groups) {
        int base = tid * 10;
        uint32_t word = 0;
        int count = min(10, head_size - base);
        for (int j = 0; j < count; j++) {
            word |= ((uint32_t)indices[base + j] & 0x7) << (j * 3);
        }
        // V cache: [block, head, packed_byte_dim, block_offset]
        // packed_byte_dim = tid*4 .. tid*4+3 (4 bytes per group)
        int byte_base = block_idx * v_block_stride +
                        head_idx * v_head_stride +
                        tid * 4 * block_size +
                        block_offset;
        // Write 4 bytes, each at stride block_size apart
        cache[byte_base] = (uint8_t)(word);
        cache[byte_base + block_size] = (uint8_t)(word >> 8);
        cache[byte_base + 2 * block_size] = (uint8_t)(word >> 16);
        cache[byte_base + 3 * block_size] = (uint8_t)(word >> 24);
    }
}

extern "C" void turbo_reshape_and_cache(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    void* k_norms, void* v_norms,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_size, int block_size,
    int key_stride, int value_stride,
    int kv_block_stride, int kv_head_stride,
    int norm_block_stride, int norm_head_stride,
    cudaStream_t stream, uint32_t dtype
) {
    if (head_size != 128) return;
    int total = num_tokens * num_heads;
    dim3 grid(total);
    dim3 block(128);

    turbo_reshape_and_cache_k<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(key),
        reinterpret_cast<uint8_t*>(key_cache),
        reinterpret_cast<__half*>(k_norms),
        slot_mapping,
        num_heads, head_size, block_size,
        key_stride,
        kv_block_stride, kv_head_stride,
        norm_block_stride, norm_head_stride
    );

    // V cache strides: [num_blocks, num_heads, 52, block_size]
    int v_packed_dim = (head_size + 9) / 10 * 4; // 52
    int v_block_stride = num_heads * v_packed_dim * block_size;
    int v_head_stride = v_packed_dim * block_size;

    turbo_reshape_and_cache_v<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(value),
        reinterpret_cast<uint8_t*>(value_cache),
        reinterpret_cast<__half*>(v_norms),
        slot_mapping,
        num_heads, head_size, block_size,
        value_stride,
        v_block_stride, v_head_stride,
        norm_block_stride, norm_head_stride
    );
}

// ============================================================================
// Attention kernel with sub-byte unpacking
// ============================================================================

__global__ void turbo_attn_kernel_d128(
    float* __restrict__ out,
    const __half* __restrict__ query,
    const uint8_t* __restrict__ k_cache,    // [B, H, 4, BS, 16] nibble-packed
    const uint8_t* __restrict__ v_cache,    // [B, H, 52, BS] 3-bit packed
    const __half* __restrict__ k_norms,
    const __half* __restrict__ v_norms,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lens,
    int num_kv_heads, int max_blocks_per_seq, int num_heads,
    int block_size, float scale,
    int k_block_stride, int k_head_stride,
    int v_block_stride, int v_head_stride,
    int norm_block_stride, int norm_head_stride
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Rotate query
    __shared__ float q_rot[128];
    if (tid < 128)
        q_rot[tid] = __half2float(query[seq_idx * num_heads * 128 + head_idx * 128 + tid]);
    __syncthreads();
    tq_rotate(q_rot, tid);

    extern __shared__ char smem_bytes[];
    float* logits = reinterpret_cast<float*>(smem_bytes);
    float qk_max = -FLT_MAX;

    const uint32_t* seq_bt = block_tables + seq_idx * max_blocks_per_seq;
    int num_blocks = (context_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        int pb = seq_bt[b];
        int tib = min(block_size, context_len - b * block_size);

        for (int t = 0; t < tib; t++) {
            int token_pos = b * block_size + t;

            // Unpack 4-bit K: each thread reads its nibble from the packed byte
            float k_val = 0.0f;
            if (tid < 128) {
                int byte_idx = tid / 2;  // which packed byte (0..63)
                int nibble = tid & 1;    // low or high nibble
                int x = 16;
                int k_offset = pb * k_block_stride +
                               kv_head_idx * k_head_stride +
                               (byte_idx / x) * block_size * x +
                               t * x +
                               (byte_idx % x);
                uint8_t packed_byte = k_cache[k_offset];
                uint8_t k_idx = nibble ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
                k_val = TQ_CB_4BIT[k_idx];
            }

            // Dot product reduction
            float partial = (tid < 128) ? q_rot[tid] * k_val : 0.0f;
            for (int mask = 16; mask > 0; mask >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, mask);

            __shared__ float warp_sums[4];
            if (tid % 32 == 0) warp_sums[tid / 32] = partial;
            __syncthreads();

            if (tid == 0) {
                float qk = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
                float k_norm = __half2float(
                    k_norms[pb * norm_block_stride + kv_head_idx * norm_head_stride + t]);
                qk = qk * k_norm * scale;
                logits[token_pos] = qk;
                qk_max = fmaxf(qk_max, qk);
            }
            __syncthreads();
        }
    }

    // Broadcast qk_max
    __shared__ float sv[1];
    if (tid == 0) sv[0] = qk_max;
    __syncthreads();
    qk_max = sv[0];

    // Softmax
    float exp_sum = 0.0f;
    for (int i = tid; i < context_len; i += 128) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, mask);
    __shared__ float we[4];
    if (tid % 32 == 0) we[tid / 32] = exp_sum;
    __syncthreads();
    if (tid == 0) {
        float total = we[0] + we[1] + we[2] + we[3];
        sv[0] = __fdividef(1.0f, total + 1e-6f);
    }
    __syncthreads();
    float inv_sum = sv[0];
    for (int i = tid; i < context_len; i += 128)
        logits[i] *= inv_sum;
    __syncthreads();

    // Weighted V sum — unpack 3-bit from 10-in-32
    // V cache: [block, head, packed_dim, slot]
    // packed_dim = 52 bytes = 13 groups of 4 bytes
    // Each group of 4 bytes holds 10 values at 3 bits each
    float acc = 0.0f;
    if (tid < 128) {
        int group = tid / 10;      // which 10-in-32 group (0..12)
        int pos_in_group = tid % 10; // position within group (0..9)

        for (int b = 0; b < num_blocks; b++) {
            int pb = seq_bt[b];
            int tib = min(block_size, context_len - b * block_size);

            for (int t = 0; t < tib; t++) {
                float weight = logits[b * block_size + t];
                if (weight < 1e-8f) continue;

                // Read 4 bytes of the group, reconstruct u32
                int byte_base = pb * v_block_stride +
                                kv_head_idx * v_head_stride +
                                group * 4 * block_size +
                                t;
                uint32_t word = (uint32_t)v_cache[byte_base] |
                               ((uint32_t)v_cache[byte_base + block_size] << 8) |
                               ((uint32_t)v_cache[byte_base + 2 * block_size] << 16) |
                               ((uint32_t)v_cache[byte_base + 3 * block_size] << 24);

                uint8_t v_idx = (word >> (pos_in_group * 3)) & 0x7;
                float v_val = TQ_CB_3BIT[v_idx];
                float v_norm = __half2float(
                    v_norms[pb * norm_block_stride + kv_head_idx * norm_head_stride + t]);
                acc += weight * v_val * v_norm;
            }
        }
    }

    // Inverse rotation
    __shared__ float out_rot[128];
    if (tid < 128) out_rot[tid] = acc;
    __syncthreads();
    tq_rotate(out_rot, tid);

    if (tid < 128)
        out[seq_idx * num_heads * 128 + head_idx * 128 + tid] = out_rot[tid];
}

extern "C" void turbo_paged_attention_v1_f16(
    void* out, const void* query,
    const void* k_cache, const void* v_cache,
    const void* k_norms, const void* v_norms,
    int num_kv_heads, float scale, float softcapping,
    const uint32_t* block_tables, const uint32_t* context_lens,
    int block_size, int max_context_len,
    int num_seqs, int num_heads, int head_size,
    int max_num_blocks_per_seq, int q_stride,
    int kv_block_stride, int kv_head_stride,
    int norm_block_stride, int norm_head_stride,
    cudaStream_t stream
) {
    if (head_size != 128) return;

    int v_packed_dim = (head_size + 9) / 10 * 4; // 52
    int v_block_stride = num_kv_heads * v_packed_dim * block_size;
    int v_head_stride = v_packed_dim * block_size;

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
        block_tables, context_lens,
        num_kv_heads, max_num_blocks_per_seq, num_heads,
        block_size, scale,
        kv_block_stride, kv_head_stride,
        v_block_stride, v_head_stride,
        norm_block_stride, norm_head_stride
    );
}
