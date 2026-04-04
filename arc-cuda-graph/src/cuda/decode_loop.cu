/**
 * Decode step update + WHILE loop condition kernels.
 *
 * Two paths:
 * - CUDA < 12.4: writes loop condition to device memory (host reads it)
 * - CUDA >= 12.4: calls cudaGraphSetConditional from device code (zero host involvement)
 */

#include <cuda_runtime.h>
#include <cstdint>

// Detect CUDA 12.4+ for conditional graph node device API
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
#define ARC_HAS_GRAPH_CONDITIONAL 1
#else
#define ARC_HAS_GRAPH_CONDITIONAL 0
#endif

__global__ void decode_step_update(
    const int32_t* __restrict__ sampled_tokens,     // [batch]
    int32_t* __restrict__ input_ids,                 // [batch]
    int32_t* __restrict__ positions,                 // [batch]
    int32_t* __restrict__ context_lens,              // [batch]
    int64_t* __restrict__ slot_mappings,              // [batch]
    const int32_t* __restrict__ block_tables,        // [batch, max_blocks_per_seq]
    int32_t* __restrict__ n_generated,               // [batch]
    int32_t* __restrict__ output_tokens,             // [batch, max_tokens]
    int32_t* __restrict__ finished,                  // [batch]
    int32_t* __restrict__ ring_buffer,               // [batch, ring_size] pinned
    int32_t* __restrict__ ring_write_head,           // [batch] pinned
    int32_t eos_token_id,
    int32_t max_tokens,
    int32_t block_size,
    int32_t max_blocks_per_seq,
    int32_t ring_size,
    int32_t* __restrict__ loop_condition_ptr,
    int32_t batch_size
) {
    int bid = blockIdx.x;
    if (threadIdx.x != 0) return;
    if (bid >= batch_size) return;
    if (finished[bid]) return;

    int token = sampled_tokens[bid];
    int pos = n_generated[bid];

    output_tokens[bid * max_tokens + pos] = token;
    n_generated[bid] = pos + 1;
    positions[bid] += 1;
    context_lens[bid] += 1;
    input_ids[bid] = token;

    int next_pos = positions[bid];
    int block_idx = next_pos / block_size;
    int block_offset = next_pos % block_size;
    if (block_idx < max_blocks_per_seq) {
        int physical_block = block_tables[bid * max_blocks_per_seq + block_idx];
        slot_mappings[bid] = (int64_t)physical_block * block_size + block_offset;
    }

    if (ring_buffer != nullptr && ring_write_head != nullptr) {
        int wp = ring_write_head[bid];
        ((volatile int32_t*)ring_buffer)[bid * ring_size + (wp % ring_size)] = token;
        __threadfence_system();
        ((volatile int32_t*)ring_write_head)[bid] = wp + 1;
    }

    if (token == eos_token_id || pos + 1 >= max_tokens) {
        finished[bid] = 1;
    }
}

// Host-driven loop path: write condition to device memory
__global__ void check_all_done(
    const int32_t* __restrict__ finished,
    int32_t* __restrict__ loop_condition_ptr,
    int32_t batch_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < batch_size; i++) {
        if (!finished[i]) {
            *loop_condition_ptr = 1;
            return;
        }
    }
    *loop_condition_ptr = 0;
}

// GPU-autonomous path: set WHILE conditional handle from device code (CUDA 12.4+)
#if ARC_HAS_GRAPH_CONDITIONAL
__global__ void check_all_done_conditional(
    const int32_t* __restrict__ finished,
    int32_t batch_size,
    cudaGraphConditionalHandle cond_handle
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < batch_size; i++) {
        if (!finished[i]) {
            cudaGraphSetConditional(cond_handle, 1);
            return;
        }
    }
    cudaGraphSetConditional(cond_handle, 0);
}
#endif

// ============================================================================
// C entry points
// ============================================================================

extern "C" void launch_decode_step_update(
    const int32_t* sampled_tokens,
    int32_t* input_ids, int32_t* positions,
    int32_t* context_lens, int64_t* slot_mappings,
    const int32_t* block_tables,
    int32_t* n_generated, int32_t* output_tokens, int32_t* finished,
    int32_t* ring_buffer, int32_t* ring_write_head,
    int32_t eos_token_id, int32_t max_tokens,
    int32_t block_size, int32_t max_blocks_per_seq, int32_t ring_size,
    int32_t* loop_condition,
    int batch_size, cudaStream_t stream
) {
    decode_step_update<<<batch_size, 1, 0, stream>>>(
        sampled_tokens,
        input_ids, positions, context_lens, slot_mappings,
        block_tables,
        n_generated, output_tokens, finished,
        ring_buffer, ring_write_head,
        eos_token_id, max_tokens, block_size, max_blocks_per_seq, ring_size,
        loop_condition, batch_size
    );
}

extern "C" void launch_check_all_done(
    const int32_t* finished, int32_t* loop_condition,
    int batch_size, cudaStream_t stream
) {
    check_all_done<<<1, 1, 0, stream>>>(finished, loop_condition, batch_size);
}

#if ARC_HAS_GRAPH_CONDITIONAL
extern "C" void launch_check_all_done_conditional(
    const int32_t* finished, int batch_size,
    cudaGraphConditionalHandle cond_handle,
    cudaStream_t stream
) {
    check_all_done_conditional<<<1, 1, 0, stream>>>(finished, batch_size, cond_handle);
}
#endif

// Returns 1 if CUDA 12.4+ conditional graph node API is available, 0 otherwise.
extern "C" int arc_has_graph_conditional() {
    return ARC_HAS_GRAPH_CONDITIONAL;
}
