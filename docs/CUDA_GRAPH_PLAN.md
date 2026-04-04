# Arc GPU-Autonomous Decode

**Goal:** Zero CPU involvement during decode. One `cuGraphLaunch` generates the entire response. The GPU runs forward pass → sampling → EOS check → position increment → loop, without ever returning to the host.

**No existing system does this.** vLLM, SGLang, and TensorRT-LLM all launch graphs from the CPU each step (~10μs). Arc eliminates the CPU from the decode loop entirely.

---

## How it works

```
CPU: pre-allocate KV blocks for max_tokens
CPU: initialize decode_state (positions, block_tables, first token)
CPU: cuGraphLaunch(decode_graph, stream)    ← ONE launch for the ENTIRE generation
GPU: ┌─ WHILE (any sequence active) ──────────────────────┐
GPU: │  1. Gather RoPE cos/sin from GPU position tensor   │
GPU: │  2. Forward pass (all layers, cuBLAS + attention)   │
GPU: │  3. Fused argmax/top-p sampling → next token ID     │
GPU: │  4. Step update kernel:                              │
GPU: │     - Store token in output buffer                   │
GPU: │     - Increment position                             │
GPU: │     - Update slot_mapping for next KV cache write    │
GPU: │     - Copy sampled token → input_ids for next step   │
GPU: │     - Check EOS / max_tokens → set WHILE condition   │
GPU: │  5. Write token to pinned ring buffer (streaming)    │
GPU: └────────────────────────────────────────────────────┘
CPU: poll ring buffer → stream tokens to client via SSE
CPU: cudaStreamSynchronize(stream) when generation complete
```

Per-step host overhead: **0μs**. The CPU launched once.

---

## Build order (dependencies, not tiers — ships as one unit)

### 1. GPU position tensors for RoPE

`RotaryEmbedding::forward()` currently takes `seqlen_offsets: &[usize]` (CPU) and does `self.cos.narrow(0, offset, seq_len)` per batch element. This creates CPU-computed pointers that get baked into graphs and can't update on replay.

**Change:** Add `forward_graph_mode(&self, q, k, positions_gpu: &Tensor)` that:
- Takes positions as a GPU tensor `[batch]` i32
- Runs a `gather_rope` kernel: for each batch element, reads `cos[positions[b]]` and `sin[positions[b]]` into a contiguous buffer, then applies rotary embedding in-place
- No CPU-side `narrow()`, no `Tensor::cat()`, no CPU-dependent addresses

**Files:**
- `mistralrs-core/src/layers.rs` — add `forward_graph_mode` to `RotaryEmbedding`
- `arc-cuda-graph/src/cuda/gather_rope.cu` — gather + apply RoPE kernel
- Every model's `Attention::forward()` — when in graph mode, call `forward_graph_mode`

### 2. Pre-allocated fixed input buffers

All dynamic decode inputs live at stable GPU addresses, allocated once per batch-size bucket.

**DecodeInputBuffers struct:**
```rust
struct DecodeInputBuffers {
    input_ids: Tensor,       // [padded_bs, 1] u32
    positions: Tensor,       // [padded_bs] i32
    block_tables: Tensor,    // [padded_bs, max_blocks] u32
    context_lens: Tensor,    // [padded_bs] u32
    slot_mappings: Tensor,   // [padded_bs] i64
}
```

Before graph capture: fill with valid placeholder data.
The forward pass reads from these tensors. The graph records their addresses.
The GPU-side step-update kernel writes new values directly into these buffers between WHILE iterations.

**Batch-size buckets:** [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, 64]. CUDA 12.6 constant-time launch (~2.5μs) makes caching many graphs free.

**Files:**
- `arc-cuda-graph/src/buffers.rs` — `DecodeInputBuffers` allocation and management
- `mistralrs-core/src/pipeline/mod.rs` — thread input buffers through `forward_inputs`
- `mistralrs-core/src/paged_attention/layers/paged_attention.rs` — read from fixed buffers in graph mode

### 3. GPU sampling kernels

Sampling runs on GPU inside the graph. No CPU round-trip per token.

**Kernels:**
- `fused_argmax_bf16(logits, token_ids_out, log_probs_out)` — greedy decode. Two-pass reduction over vocab dimension. Handles [batch, vocab] in one kernel.
- `fused_top_p_bf16(logits, temperature, top_p, token_ids_out, rng_state)` — nucleus sampling. Radix sort + cumulative sum in shared memory. Uses on-GPU RNG (cuRAND device API or Philox counter-based).
- `penalty_apply(logits, generated_tokens, freq_penalty, presence_penalty)` — modifies logits in-place before sampling.

**Files:**
- `arc-cuda-graph/src/cuda/sampling.cu` — all sampling kernels
- `arc-cuda-graph/src/sampling.rs` — Rust FFI + integration

### 4. Step-update kernel

Runs after sampling each iteration. Updates all decode state for the next forward pass.

```cuda
__global__ void decode_step_update(
    // Inputs (written by sampling)
    const int32_t* sampled_tokens,   // [batch] — output of argmax/top-p

    // Decode state (read-write, persistent across WHILE iterations)
    int32_t* input_ids,              // [batch, 1] — next forward pass reads this
    int32_t* positions,              // [batch] — RoPE positions
    int32_t* context_lens,           // [batch] — attention context lengths
    int64_t* slot_mappings,          // [batch] — KV cache write slots
    const int32_t* block_tables,     // [batch, max_blocks] — pre-allocated, read-only
    int32_t* n_generated,            // [batch] — tokens generated so far
    int32_t* output_tokens,          // [batch, max_tokens] — full output sequence
    bool* finished,                  // [batch] — per-sequence done flag

    // Ring buffer for streaming (pinned host memory, zero-copy)
    int32_t* ring_buffer,            // [batch, ring_size]
    int32_t* ring_write_head,        // [batch] — atomic write position

    // Constants
    int32_t eos_token,
    int32_t max_tokens,
    int32_t block_size,
    int32_t max_blocks,

    // WHILE loop condition (device-side)
    int32_t* loop_condition          // scalar — set to 0 when all sequences done
) {
    int bid = blockIdx.x;  // one block per batch element
    if (finished[bid]) return;

    int token = sampled_tokens[bid];
    int pos = n_generated[bid];

    // Store token in output
    output_tokens[bid * max_tokens + pos] = token;
    n_generated[bid] = pos + 1;

    // Update input for next forward pass
    input_ids[bid] = token;
    positions[bid] += 1;
    context_lens[bid] += 1;

    // Compute next KV cache slot from pre-allocated block table
    int next_pos = positions[bid];
    int block_idx = next_pos / block_size;
    int block_offset = next_pos % block_size;
    if (block_idx < max_blocks) {
        slot_mappings[bid] = (int64_t)block_tables[bid * max_blocks + block_idx] * block_size + block_offset;
    }

    // Write to streaming ring buffer (pinned host memory — visible to CPU immediately)
    int ring_pos = atomicAdd(&ring_write_head[bid], 1);
    ring_buffer[bid * ring_size + (ring_pos % ring_size)] = token;

    // Check termination
    if (token == eos_token || pos + 1 >= max_tokens) {
        finished[bid] = true;
    }

    // Global reduction: are ALL sequences done?
    __syncthreads();
    if (threadIdx.x == 0) {
        // Use atomicAnd across blocks to check if all are finished
        // Block 0 writes the final condition
        if (bid == 0) {
            bool all_done = true;
            for (int i = 0; i < gridDim.x; i++) {
                if (!finished[i]) { all_done = false; break; }
            }
            *loop_condition = all_done ? 0 : 1;
        }
    }
}
```

**Files:**
- `arc-cuda-graph/src/cuda/decode_loop.cu` — step-update kernel
- `arc-cuda-graph/src/decode_state.rs` — DecodeState struct management

### 5. CUDA 12.4 conditional WHILE graph node

Assemble the decode graph with a WHILE body:

```rust
// Pseudocode for graph construction
let handle = cudaGraphConditionalHandleCreate(graph, CUDA_GRAPH_COND_TYPE_WHILE);

// The WHILE body graph contains:
//   1. Gather RoPE from GPU positions
//   2. Full model forward pass (captured from Candle)
//   3. Fused sampling kernel
//   4. Step-update kernel (sets WHILE condition)

// Capture the body by recording one decode step
cudaStreamBeginCapture(stream);
model.forward_graph_mode(input_buffers, ...);
fused_argmax(logits, sampled_tokens, ...);
decode_step_update(sampled_tokens, decode_state, loop_condition, ...);
cudaStreamEndCapture(stream) → body_graph;

// Attach body as WHILE node
cudaGraphAddConditionalNode(graph, while_node, body_graph, handle);

// Instantiate and launch
cudaGraphInstantiate(graph) → exec;
cudaGraphLaunch(exec, stream);  // ONE LAUNCH, generates entire response
```

**FFI additions:**
- `cudaGraphConditionalHandleCreate`
- `cudaGraphSetConditional`
- Conditional node creation APIs

**Files:**
- `arc-cuda-graph/src/ffi.rs` — conditional node FFI bindings
- `arc-cuda-graph/src/autonomous.rs` — WHILE graph assembly
- `arc-cuda-graph/src/graph.rs` — rewrite: one method to build and launch the autonomous decode graph

### 6. Zero-copy token streaming

The step-update kernel writes tokens to pinned host memory (allocated with `cudaHostAlloc` / `cudaMallocHost`). The CPU reads from this memory without any synchronization — it's immediately visible after the GPU write (on x86 with PCIe, there's a small delay but no explicit sync needed).

**Streaming flow:**
```
Engine thread:
  launch autonomous graph
  loop:
    poll ring_buffer write_head
    if new tokens available:
      read tokens from ring_buffer
      send via SSE to client
    if all sequences finished:
      break
```

**Files:**
- `arc-cuda-graph/src/ring_buffer.rs` — pinned host memory ring buffer
- `mistralrs-core/src/engine/mod.rs` — polling loop for streaming

---

## Files summary

**New files:**
```
arc-cuda-graph/
├── Cargo.toml
├── build.rs
├── src/
│   ├── lib.rs
│   ├── ffi.rs                  # CUDA driver FFI (graph + conditional nodes)
│   ├── graph.rs                # Autonomous decode graph assembly + launch
│   ├── buffers.rs              # Pre-allocated decode input buffers
│   ├── decode_state.rs         # GPU-side decode state management
│   ├── sampling.rs             # GPU sampling Rust wrappers
│   ├── ring_buffer.rs          # Zero-copy pinned memory ring buffer
│   ├── autonomous.rs           # WHILE conditional node setup
│   └── cuda/
│       ├── gather_rope.cu      # RoPE gather from GPU positions
│       ├── sampling.cu         # fused_argmax, fused_top_p, penalty_apply
│       └── decode_loop.cu      # decode_step_update kernel
```

**Modified files:**
```
mistralrs-core/src/layers.rs                              # RotaryEmbedding::forward_graph_mode
mistralrs-core/src/models/*.rs                             # Call forward_graph_mode when in graph mode
mistralrs-core/src/pipeline/mod.rs                         # graph_wrapped_forward → autonomous path
mistralrs-core/src/pipeline/inputs_processor.rs            # Populate fixed buffers instead of fresh tensors
mistralrs-core/src/paged_attention/layers/paged_attention.rs  # Read from fixed buffers in graph mode
mistralrs-core/src/engine/mod.rs                           # Ring buffer polling for streaming
```

---

## What this achieves

- **0μs per-step CPU overhead** — GPU loops autonomously
- **~2.5μs per-generation overhead** — one graph launch
- **Zero padding waste** — fine-grained batch buckets
- **Streaming without sync** — zero-copy ring buffer
- **Model-agnostic** — only RoPE needs a graph-mode path; everything else is engine-level
- **Better than every existing system** — vLLM, SGLang, TRT-LLM all have >0μs per-step CPU cost
