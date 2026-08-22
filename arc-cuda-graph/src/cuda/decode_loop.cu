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

// ============================================================================
// ArcInfer/ArcGraph — device decode loop step commit
//
// The narrow sibling of `decode_step_update` above, for the CUDA-graph replay
// path (`arc-cuda-graph/src/device_loop.rs`). That kernel is written for the
// paged-attention autonomous runner and unconditionally dereferences
// `finished`, `n_generated`, `slot_mappings` and `block_tables`; the replay
// path has none of those (V4 runs with `cache_config == None`), so passing
// nulls would fault. This kernel touches only what that path actually owns.
//
// It closes the decode loop on-device:
//   sampled token  ->  the PINNED U32 buffer the captured graph reads
//                  ->  device position += 1
//                  ->  pinned+mapped host ring, for the host to drain later
//
// Ordering is load-bearing. Every range check runs BEFORE any write, and the
// ring write head is published LAST, after `__threadfence_system()`. So a token
// the host can see in the ring is guaranteed to have been committed to the
// graph input buffer, and a refused commit publishes nothing at all rather than
// letting the next launch consume a garbage id.
// ============================================================================

#define ARC_COMMIT_FAULT_NONE           0
#define ARC_COMMIT_FAULT_TOKEN_RANGE    1
#define ARC_COMMIT_FAULT_POSITION_LIMIT 2

__global__ void arc_graph_step_commit(
    const int32_t* __restrict__ sampled,   // [batch] i32, written by CudaSampler
    uint32_t* __restrict__ input_ids,      // [batch] U32, PINNED graph input
    uint32_t* __restrict__ positions,      // [batch] U32, PINNED graph input; nullable
    int32_t*  __restrict__ ring,           // [batch, ring_size] pinned + mapped
    int32_t*  __restrict__ ring_head,      // [batch] pinned + mapped
    int32_t*  __restrict__ fault,          // [1] pinned + mapped, sticky
    int32_t ring_size,
    int32_t vocab,
    int32_t position_limit,                // exclusive; <= 0 disables the check
    int32_t batch_size
) {
    int bid = blockIdx.x;
    if (threadIdx.x != 0) return;
    if (bid >= batch_size) return;

    // A fault latched by an earlier step means the graph input buffer already
    // holds a stale id. Committing on top of that would paper over it.
    if (fault != nullptr && ((volatile int32_t*)fault)[0] != ARC_COMMIT_FAULT_NONE) return;

    int token = sampled[bid];

    // ---- checks first, writes after -------------------------------------
    if (token < 0 || token >= vocab) {
        if (fault != nullptr) {
            ((volatile int32_t*)fault)[0] = ARC_COMMIT_FAULT_TOKEN_RANGE;
            __threadfence_system();
        }
        return;
    }
    if (positions != nullptr && position_limit > 0) {
        // `positions[bid]` is the slot this step just wrote KV to. The next
        // step writes at +1, and the fixed-capacity window is `position_limit`
        // wide, so +1 must stay strictly inside it.
        if ((int64_t)positions[bid] + 1 >= (int64_t)position_limit) {
            if (fault != nullptr) {
                ((volatile int32_t*)fault)[0] = ARC_COMMIT_FAULT_POSITION_LIMIT;
                __threadfence_system();
            }
            return;
        }
    }

    // ---- commit ---------------------------------------------------------
    // This single store is what removes the host from the loop: the captured
    // graph baked THIS address at capture time, so the next `cuGraphLaunch`
    // reads the token without any host round trip.
    input_ids[bid] = (uint32_t)token;
    if (positions != nullptr) {
        positions[bid] += 1u;
    }

    // ---- publish last ----------------------------------------------------
    if (ring != nullptr && ring_head != nullptr && ring_size > 0) {
        int wp = ring_head[bid];
        int slot = wp % ring_size;
        if (slot < 0) slot += ring_size;
        ((volatile int32_t*)ring)[(int64_t)bid * ring_size + slot] = token;
        // Order the token store ahead of the head store as seen by the HOST.
        // Without this the host can observe an advanced head and read the
        // previous occupant of the slot.
        __threadfence_system();
        ((volatile int32_t*)ring_head)[bid] = wp + 1;
    }
}

extern "C" void arc_launch_graph_step_commit(
    const int32_t* sampled,
    uint32_t* input_ids,
    uint32_t* positions,
    int32_t* ring,
    int32_t* ring_head,
    int32_t* fault,
    int32_t ring_size,
    int32_t vocab,
    int32_t position_limit,
    int32_t batch_size,
    cudaStream_t stream
) {
    arc_graph_step_commit<<<batch_size, 1, 0, stream>>>(
        sampled, input_ids, positions,
        ring, ring_head, fault,
        ring_size, vocab, position_limit, batch_size
    );
}
