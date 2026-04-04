/**
 * Gather RoPE cos/sin from pre-computed tables using GPU position tensor.
 *
 * Standard RoPE uses CPU-side `narrow(0, position, 1)` to index the cos/sin table.
 * This kernel does the same thing entirely on GPU — reads positions from a GPU tensor,
 * gathers the corresponding cos/sin rows, and applies rotary embedding in-place.
 *
 * This is the key enabler for CUDA graph capture: positions can change between
 * graph iterations (WHILE loop) without any CPU involvement.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

/**
 * Gather cos/sin rows from the pre-computed table based on GPU position tensor,
 * then apply rotary embedding to Q and K in-place.
 *
 * The pre-computed table has shape [max_seq_len, rot_dim/2].
 * For each batch element, we read positions[b] and gather cos_table[pos] and sin_table[pos].
 *
 * GPT-NeoX style (is_neox=true): split head into [0..half, half..dim]
 *   q_out[..half] = q[..half] * cos - q[half..] * sin
 *   q_out[half..] = q[half..] * cos + q[..half] * sin
 *
 * GPT-J style (is_neox=false): interleaved pairs [0,1], [2,3], ...
 *   q_out[2i]   = q[2i]   * cos[i] - q[2i+1] * sin[i]
 *   q_out[2i+1] = q[2i+1] * cos[i] + q[2i]   * sin[i]
 */

// Apply RoPE to a single head using gathered cos/sin
// head: pointer to [head_dim] elements
// cos_row, sin_row: pointer to [rot_dim/2] elements
template<typename T>
__device__ void apply_rope_to_head(
    T* head, const T* cos_row, const T* sin_row,
    int head_dim, int rot_dim, bool is_neox, int tid, int stride
) {
    int half = rot_dim / 2;
    if (is_neox) {
        // GPT-NeoX: first half and second half
        for (int i = tid; i < half; i += stride) {
            float c = (float)cos_row[i];
            float s = (float)sin_row[i];
            float x0 = (float)head[i];
            float x1 = (float)head[i + half];
            head[i]        = (T)(x0 * c - x1 * s);
            head[i + half] = (T)(x1 * c + x0 * s);
        }
    } else {
        // GPT-J: interleaved pairs
        for (int i = tid; i < half; i += stride) {
            float c = (float)cos_row[i];
            float s = (float)sin_row[i];
            float x0 = (float)head[2 * i];
            float x1 = (float)head[2 * i + 1];
            head[2 * i]     = (T)(x0 * c - x1 * s);
            head[2 * i + 1] = (T)(x1 * c + x0 * s);
        }
    }
}

/**
 * GPU-gathered RoPE for decode (seq_len=1).
 *
 * q: [batch, num_heads, 1, head_dim]     (contiguous, in-place)
 * k: [batch, num_kv_heads, 1, head_dim]  (contiguous, in-place)
 * cos_table: [max_seq_len, rot_dim/2]    (pre-computed, read-only)
 * sin_table: [max_seq_len, rot_dim/2]    (pre-computed, read-only)
 * positions: [batch] i32                  (GPU tensor, the key enabler)
 *
 * Grid: (num_heads_total, batch)  where num_heads_total = num_heads + num_kv_heads
 * Block: (min(rot_dim/2, 128))
 */
template<typename T>
__global__ void gather_rope_decode(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ cos_table,
    const T* __restrict__ sin_table,
    const int32_t* __restrict__ positions,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rot_dim,
    int cos_stride,  // number of elements per row in cos/sin table (= rot_dim/2)
    bool is_neox
) {
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    int pos = positions[batch_idx];
    const T* cos_row = cos_table + pos * cos_stride;
    const T* sin_row = sin_table + pos * cos_stride;

    int total_heads = num_heads + num_kv_heads;
    if (head_idx >= total_heads) return;

    T* head;
    if (head_idx < num_heads) {
        // Q head
        head = q + batch_idx * num_heads * head_dim + head_idx * head_dim;
    } else {
        // K head
        int kv_idx = head_idx - num_heads;
        head = k + batch_idx * num_kv_heads * head_dim + kv_idx * head_dim;
    }

    apply_rope_to_head(head, cos_row, sin_row, head_dim, rot_dim, is_neox, tid, stride);
}

// ============================================================================
// C entry points
// ============================================================================

extern "C" void launch_gather_rope_decode_bf16(
    void* q, void* k,
    const void* cos_table, const void* sin_table,
    const int32_t* positions,
    int num_heads, int num_kv_heads,
    int head_dim, int rot_dim,
    int cos_stride,
    int batch_size,
    int is_neox,
    cudaStream_t stream
) {
    int total_heads = num_heads + num_kv_heads;
    int threads = min(rot_dim / 2, 128);
    dim3 grid(total_heads, batch_size);
    dim3 block(threads);

    gather_rope_decode<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)q, (__nv_bfloat16*)k,
        (const __nv_bfloat16*)cos_table, (const __nv_bfloat16*)sin_table,
        positions,
        num_heads, num_kv_heads, head_dim, rot_dim, cos_stride,
        (bool)is_neox
    );
}

extern "C" void launch_gather_rope_decode_f16(
    void* q, void* k,
    const void* cos_table, const void* sin_table,
    const int32_t* positions,
    int num_heads, int num_kv_heads,
    int head_dim, int rot_dim,
    int cos_stride,
    int batch_size,
    int is_neox,
    cudaStream_t stream
) {
    int total_heads = num_heads + num_kv_heads;
    int threads = min(rot_dim / 2, 128);
    dim3 grid(total_heads, batch_size);
    dim3 block(threads);

    gather_rope_decode<__half><<<grid, block, 0, stream>>>(
        (__half*)q, (__half*)k,
        (const __half*)cos_table, (const __half*)sin_table,
        positions,
        num_heads, num_kv_heads, head_dim, rot_dim, cos_stride,
        (bool)is_neox
    );
}
